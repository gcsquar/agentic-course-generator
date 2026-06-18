"""Thin OpenAI wrapper used by every agent and gate.

Two helpers:
  - chat()      -> free-form text completion
  - chat_json() -> parsed JSON object (uses response_format=json_object)

Keeping this in one place means we can swap models, add retries, or log
token usage without touching agent logic.
"""
from __future__ import annotations

import json
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
                if not config.OPENAI_API_KEY:
                    raise RuntimeError(
                        "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
                    )
                _client = OpenAI(api_key=config.OPENAI_API_KEY)
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
    def _call():
        resp = _get_client().chat.completions.create(
            model=config.MODEL,
            temperature=config.TEMPERATURE if temperature is None else temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return json.loads(resp.choices[0].message.content or "{}")
    return _call_with_retry(_call)
