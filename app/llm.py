"""Thin OpenAI-compatible wrapper used by every agent and gate.

Two helpers:
  - chat()      -> free-form text completion
  - chat_json() -> parsed JSON object (uses response_format=json_object)

Both take an optional `model=` so a caller can pick a role-specific model
(config.JUDGE_MODEL / AUDITOR_MODEL) instead of the default config.MODEL —
keeping all model/temperature/timeout/usage policy in this one place.

Robustness baked in here so agent logic stays clean:
  - rate-limit / transient-connection retries with backoff;
  - capability-aware kwargs: some models reject `response_format=json_object`
    or a custom `temperature` with a 400 — we detect that and retry without the
    offending field instead of failing the whole call;
  - a per-request timeout so a hung completion can't block a worker forever;
  - lightweight token-usage accounting (per-call log + a running total).
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import config

# One OpenAI-compatible client PER provider, built lazily and cached. Per-call model
# routing (below) picks the right one, so generation can run on a cheap model while the
# judge/auditor run on smarter ones — even on different providers in the same run.
_clients: dict[str, Any] = {}
_lock = threading.Lock()

# How many times to retry on rate-limit / transient connection errors.
_RATE_LIMIT_RETRIES = 3

# --- token-usage accounting -------------------------------------------------
# Agent 3 fans out across threads, so guard the running totals with a lock.
_usage_lock = threading.Lock()
_usage_totals = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


def _record_usage(model: str, usage: Any) -> None:
    if not usage:
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or (prompt + completion)
    with _usage_lock:
        _usage_totals["prompt"] += prompt
        _usage_totals["completion"] += completion
        _usage_totals["total"] += total
        _usage_totals["calls"] += 1
    print(f"[llm] {model} tokens: prompt={prompt} completion={completion} total={total}")


def usage_summary() -> dict[str, int]:
    """Return a copy of the cumulative token usage for this process."""
    with _usage_lock:
        return dict(_usage_totals)


def reset_usage() -> None:
    with _usage_lock:
        for k in _usage_totals:
            _usage_totals[k] = 0


# --- client routing ---------------------------------------------------------
def _provider_for_model(model: str) -> str:
    """Which provider serves this model. OpenRouter slugs are namespaced
    (`vendor/model`) or end in `:free`; everything else is treated as OpenAI. This
    lets per-role models (config.MODEL / JUDGE_MODEL / AUDITOR_MODEL) live on
    different providers without any extra config."""
    m = (model or "").strip()
    if "/" in m or m.endswith(":free"):
        return "openrouter"
    return "openai"


def _client_for(provider: str):
    """Lazily build + cache one client per provider, reading that provider's own key."""
    client = _clients.get(provider)
    if client is not None:
        return client
    with _lock:
        client = _clients.get(provider)
        if client is None:
            from openai import OpenAI
            if provider == "openrouter":
                if not config.OPENROUTER_API_KEY:
                    raise RuntimeError(
                        "A model is routed to OpenRouter but OPENROUTER_API_KEY is not set "
                        "in .env (free at https://openrouter.ai/keys)."
                    )
                client = OpenAI(api_key=config.OPENROUTER_API_KEY,
                                base_url="https://openrouter.ai/api/v1",
                                default_headers={
                                    "HTTP-Referer": "https://github.com/agentic-course-generator",
                                    "X-Title": "Agentic Course Generator",
                                })
            else:
                if not config.OPENAI_API_KEY:
                    raise RuntimeError(
                        "A model is routed to OpenAI but OPENAI_API_KEY is not set in .env "
                        "(copy .env.example to .env and fill it in)."
                    )
                client = OpenAI(api_key=config.OPENAI_API_KEY)
            _clients[provider] = client
    return client


def _call_with_retry(fn):
    """Retry fn() on rate-limit or transient connection errors with exponential backoff.

    Other exceptions propagate immediately — callers decide whether to fall back.
    """
    from openai import RateLimitError, APIConnectionError
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            return fn()
        except (RateLimitError, APIConnectionError) as exc:
            if attempt == _RATE_LIMIT_RETRIES:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"[llm] {type(exc).__name__}, retrying in {wait}s (attempt {attempt + 1}/{_RATE_LIMIT_RETRIES})")
            time.sleep(wait)


# --- capability-aware completion --------------------------------------------
def _complete(messages: list[dict], *, model: str, temperature: float | None,
              want_json: bool) -> str:
    """One completion, adapting to the model's capabilities.

    Some models reject `response_format=json_object` or a non-default
    `temperature` with a 400. Rather than fail, we strip whichever field the
    error names and retry — so the SAME call works across OpenAI, OpenRouter free
    models, and reasoning models that pin temperature to 1."""
    from openai import BadRequestError

    send_json = want_json
    send_temp = temperature is not None
    client = _client_for(_provider_for_model(model))

    for _ in range(3):   # at most: drop json, then drop temperature
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": config.LLM_TIMEOUT,
        }
        if send_temp:
            kwargs["temperature"] = temperature
        if send_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = _call_with_retry(lambda: client.chat.completions.create(**kwargs))
        except BadRequestError as exc:
            msg = str(exc).lower()
            if send_json and ("response_format" in msg or "json" in msg):
                send_json = False          # model can't do strict JSON mode — parse from text
                continue
            if send_temp and "temperature" in msg:
                send_temp = False          # model pins temperature — let it use its default
                continue
            raise
        _record_usage(model, getattr(resp, "usage", None))
        return resp.choices[0].message.content or ("{}" if want_json else "")

    raise RuntimeError("LLM call failed after stripping unsupported parameters")


# --- public helpers ---------------------------------------------------------
def chat(system: str, user: str, *, temperature: float | None = None,
         model: str | None = None) -> str:
    """Single-turn chat completion returning plain text."""
    temp = config.TEMPERATURE if temperature is None else temperature
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    return _complete(messages, model=model or config.MODEL, temperature=temp, want_json=False)


def chat_json(system: str, user: str, *, temperature: float | None = None,
              model: str | None = None) -> dict[str, Any]:
    """Chat completion constrained to a JSON object, returned parsed."""
    temp = config.TEMPERATURE if temperature is None else temperature
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    content = _complete(messages, model=model or config.MODEL, temperature=temp, want_json=True)
    return _parse_json(content)


def _parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object/array from model output, tolerating ```json fences or
    surrounding prose (needed for models that ignore strict JSON mode)."""
    text = (text or "").strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}
