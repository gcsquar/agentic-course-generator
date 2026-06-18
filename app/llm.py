"""Thin OpenAI wrapper used by every agent and gate.

Two helpers:
  - chat()      -> free-form text completion
  - chat_json() -> parsed JSON object (uses response_format=json_object)

Keeping this in one place means we can swap models, add retries, or log
token usage without touching agent logic.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import config

_client = None
_lock = threading.Lock()

# How many times to retry on rate-limit / transient connection errors.
_RATE_LIMIT_RETRIES = 3


def _get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                from openai import OpenAI
                if not config.LLM_API_KEY:
                    key_name = ("OPENROUTER_API_KEY" if config.LLM_PROVIDER == "openrouter"
                                else "OPENAI_API_KEY")
                    raise RuntimeError(
                        f"No API key for provider '{config.LLM_PROVIDER}'. Set {key_name} "
                        "in .env (copy .env.example to .env and fill it in)."
                    )
                kwargs: dict[str, Any] = {"api_key": config.LLM_API_KEY}
                if config.LLM_BASE_URL:  # OpenRouter (OpenAI-compatible endpoint)
                    kwargs["base_url"] = config.LLM_BASE_URL
                    kwargs["default_headers"] = {
                        "HTTP-Referer": "https://github.com/agentic-course-generator",
                        "X-Title": "Agentic Course Generator",
                    }
                _client = OpenAI(**kwargs)
    return _client


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


def chat(system: str, user: str, *, temperature: float | None = None) -> str:
    """Single-turn chat completion returning plain text."""
    def _call():
        resp = _get_client().chat.completions.create(
            model=config.MODEL,
            temperature=config.TEMPERATURE if temperature is None else temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    return _call_with_retry(_call)


def chat_json(system: str, user: str, *, temperature: float | None = None) -> dict[str, Any]:
    """Chat completion constrained to a JSON object, returned parsed."""
    temp = config.TEMPERATURE if temperature is None else temperature

    def _call(use_format: bool) -> str:
        kwargs: dict[str, Any] = dict(
            model=config.MODEL,
            temperature=temp,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if use_format:
            kwargs["response_format"] = {"type": "json_object"}
        resp = _get_client().chat.completions.create(**kwargs)
        return resp.choices[0].message.content or "{}"

    from openai import BadRequestError
    try:
        content = _call_with_retry(lambda: _call(True))
    except BadRequestError:
        # Some OpenRouter free models reject response_format=json_object. Retry without
        # it and parse JSON out of the text — the prompts already ask for JSON output.
        content = _call_with_retry(lambda: _call(False))
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
