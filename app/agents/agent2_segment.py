"""Agent 2 — Curriculum Segmentation.

CONTRACT (do not change the signature — the Supervisor depends on it):
    segment(source: IngestResult, *, mock: bool = False, use_llm: bool = True,
            feedback: list[str] | None = None) -> Curriculum

Job:  clean article text  ->  a set of standalone, distinct, faithful micro-lessons.
`feedback` carries the gate's issues on a retry — use it to fix the prior attempt.
Output `Curriculum` must satisfy `gates.gate_segment` (standalone, distinct, faithful).

This is a STUB. `mock=True` returns canned lessons so the pipeline runs end-to-end.
"""
from __future__ import annotations

from contracts import IngestResult, Curriculum, Lesson


def segment(source: IngestResult, *, mock: bool = False, use_llm: bool = True,
            feedback: list[str] | None = None) -> Curriculum:
    if mock or not use_llm:
        lessons = [
            Lesson(order=1, title="Self-Attention",
                   body="What attention is and why it replaced recurrence.",
                   source_span=source.clean_text[:120], n_formulas=1,
                   description="Introduces attention as a replacement for recurrence.",
                   key_concepts=["attention", "query/key/value"]),
            Lesson(order=2, title="Multi-Head Attention",
                   body="Running several attention projections in parallel.",
                   source_span=source.clean_text[120:240],
                   description="Why multiple attention heads help.",
                   key_concepts=["multi-head attention"]),
            Lesson(order=3, title="Positional Encoding",
                   body="Why order must be injected into a permutation-invariant model.",
                   source_span=source.clean_text[240:360],
                   description="How order is added to a permutation-invariant model.",
                   key_concepts=["positional encoding"]),
        ]
        return Curriculum(
            source_url=source.url,
            key_concepts=["self-attention", "multi-head attention", "positional encoding"],
            lessons=lessons,
        )
    # TODO: real LLM segmentation into standalone lessons with source spans;
    # if `feedback` is present, fix those issues.
    raise NotImplementedError(
        "Agent 2 not implemented yet. Run with --mock, or implement segment()."
    )
