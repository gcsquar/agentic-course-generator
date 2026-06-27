"""Utilities for presenting source spans as indexed sentence-like units.

The source index is an evidence format for Agent 3 and Gate 3: generation cites
stable sentence IDs instead of fragile exact quote substrings.
"""
from __future__ import annotations

import html
import re


def indexed_sentences(text: str) -> list[dict[str, str]]:
    """Split source text into stable, sentence-like units with string IDs."""
    units: list[str] = []
    for block in re.split(r"\n\s*\n+", text or ""):
        block = " ".join(block.split()).strip()
        if not block:
            continue
        # Keep short headings/bullets as their own evidence unit. Long prose blocks
        # are split at sentence boundaries, but we avoid clever NLP dependencies.
        if len(block) <= 220 or block.startswith(("#", "- ", "* ")):
            parts = [block]
        else:
            parts = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9#*\-]|[‘'\"]))", block)
        for part in parts:
            part = part.strip()
            if part:
                units.append(part)

    return [{"id": str(i), "text": unit} for i, unit in enumerate(units, start=1)]


def render_indexed_source(text: str, *, max_chars: int | None = None) -> str:
    """Render text as simple XML-ish source evidence blocks."""
    items = indexed_sentences(text)
    lines = ["<SOURCE>"]
    used = 0
    for item in items:
        escaped = html.escape(item["text"], quote=False)
        line = f'<s id="{item["id"]}">{escaped}</s>'
        if max_chars is not None and used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    lines.append("</SOURCE>")
    return "\n".join(lines)


def valid_sentence_ids(text: str) -> set[str]:
    return {item["id"] for item in indexed_sentences(text)}
