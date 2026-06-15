"""Thin OpenAI wrapper used by every agent and gate.

Two helpers:
  - chat()      -> free-form text completion
  - chat_json() -> parsed JSON object (uses response_format=json_object)

Keeping this in one place means we can swap models, add retries, or log
token usage without touching agent logic.
"""
from __future__ import annotations

import json
from typing import Any

import config

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def chat(system: str, user: str, *, temperature: float | None = None) -> str:
    """Single-turn chat completion returning plain text."""
    resp = _get_client().chat.completions.create(
        model=config.MODEL,
        temperature=config.TEMPERATURE if temperature is None else temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def chat_json(system: str, user: str, *, temperature: float | None = None) -> dict[str, Any]:
    """Chat completion constrained to a JSON object, returned parsed."""
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
