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

import re
from contracts import IngestResult, Curriculum, Lesson
import llm

def split_into_paragraphs(text: str) -> list[str]:
    """Divide text into separated abstracts (if only >= 50 symbols) for better ciatation. Criteria: \n\n"""
    raw_parts = re.split(r"\n\s*\n", text)
    paragraphs = []
    buffer = ""
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        buffer += ("  " if buffer else "") + part
        if len(buffer) >= 50:
            paragraphs.append(buffer)
            buffer = ""
    if buffer:
        if paragraphs:
            paragraphs[-1] += "  " + buffer
        else:
            paragraphs.append(buffer)
    return paragraphs

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
    
    paragraphs = split_into_paragraphs(source.clean_text)
    working_paragraphs = paragraphs[:100]

    numbered = "\n".join(
        f"[{i}] {p[:400]}{'...' if len(p) > 400 else ''}"
        for i, p in enumerate(working_paragraphs)
    )

    system_prompt = (
        "You are an expert curriculum designer. Your ONLY job is to mark WHERE each "
        "micro-lesson STARTS in the numbered paragraphs below. You do NOT write lesson text.\n"
        "RULES:\n"
        "- Produce BETWEEN 3 AND 8 segments. Returning a single segment is WRONG — the article "
        "must be split into several distinct micro-lessons.\n"
        "- Each segment gives only the paragraph index where it STARTS (`start_idx`). Segments "
        "run in order: each ends right before the next one begins, so they tile the whole text "
        "with no gaps or overlaps. Do NOT supply an end index.\n"
        "- The FIRST segment MUST have start_idx 0; start_idx values must STRICTLY INCREASE.\n"
        "- Do NOT include paragraph text, an article 'body', or any full content — indices and "
        "short metadata ONLY.\n"
        "- Start a new segment where the topic shifts; a segment's paragraphs must be coherent "
        "enough to stand alone as one lesson.\n"
        'Return a JSON OBJECT with a "segments" array, exactly like:\n'
        "{\n"
        '  "segments": [\n'
        '    {"title": "Short name", "description": "1-2 lines", "key_concepts": ["c1", "c2"], "start_idx": 0},\n'
        '    {"title": "Next topic", "description": "1-2 lines", "key_concepts": ["c3"], "start_idx": 7}\n'
        "  ]\n"
        "}\n"
    )

    user_prompt = f"SOURCE PARAGRAPHS:\n{numbered}"
    
    if feedback:
        user_prompt += "\n\nPAY ATTENTION (fix these issues from the previous attempt):\n" + "\n".join(f"- {f}" for f in feedback)

    response = llm.chat_json(
        system=system_prompt,
        user=user_prompt,
        temperature=0.2,
    )

    # chat_json always returns a dict (json_object mode can't return a bare array).
    if isinstance(response, dict):
        if "start_idx" in response or "title" in response:
            # LLM returned a single segment as a flat dict — wrap it
            lessons_data = [response]
        else:
            # Find a list whose elements are dicts (skip string lists like key_concepts)
            lessons_data = next(
                (v for v in response.values()
                 if isinstance(v, list) and v and isinstance(v[0], dict)),
                None,
            )
    elif isinstance(response, list):
        lessons_data = response
    else:
        lessons_data = None

    if not lessons_data:
        raise ValueError(f"LLM returned no lesson list. Raw response: {response}")

    if not all(isinstance(seg, dict) for seg in lessons_data):
        raise ValueError(f"LLM returned non-dict items in lesson list. Raw: {response}")

    n = len(working_paragraphs)

    # Deterministic tiling: trust the LLM only for CUT POINTS (where each lesson
    # starts), never for the spans. We sort + dedupe the starts, force the first to
    # paragraph 0, and derive each end = (next start - 1), last end = n-1. Coverage is
    # then correct BY CONSTRUCTION — no gaps, no overlaps — no matter how sloppy the
    # model's indices are. (Letting the LLM emit [start, end] left ~20% of the source
    # in no lesson across retries; see INSIGHTS.md.)
    # Clamp each start FIRST (junk/out-of-range -> 0..n-1), then sort by it, so the
    # ordering always matches the real starts. Dedupe equal starts (keep the first).
    candidates = sorted(
        ((max(0, min(_safe_int(seg.get("start_idx"), 0), n - 1)), seg) for seg in lessons_data),
        key=lambda c: c[0],
    )
    cuts: list[tuple[int, dict]] = []
    seen_starts: set[int] = set()
    for start, seg in candidates:
        if start in seen_starts:
            continue   # two lessons claiming the same start -> keep the first only
        seen_starts.add(start)
        cuts.append((start, seg))

    if not cuts:                       # degenerate: one lesson over everything
        cuts = [(0, {"title": "Full article"})]
    if cuts[0][0] != 0:                # guarantee the source is covered from the top
        cuts[0] = (0, cuts[0][1])

    lessons = []
    all_key_concepts = set()
    for i, (start, seg) in enumerate(cuts):
        end = cuts[i + 1][0] - 1 if i + 1 < len(cuts) else n - 1
        source_span = "\n\n".join(working_paragraphs[start : end + 1])

        # Extractive by design: the lesson body IS the raw source paragraphs for this
        # range. Agent 2 never writes prose, so it cannot invent facts — the single
        # rewrite step lives in Agent 3 (per learner), checked against this same span.
        lesson = Lesson(
            order=i + 1,
            title=seg.get("title", f"Micro-lesson {i+1}").strip(),
            body=source_span,
            source_span=source_span,
            description=seg.get("description", "").strip(),
            key_concepts=[kc.strip() for kc in seg.get("key_concepts", []) if kc.strip()],
            n_formulas=0,
            start_idx=start,
            end_idx=end,
        )
        lessons.append(lesson)
        all_key_concepts.update(lesson.key_concepts)

    return Curriculum(
        source_url=source.url,
        key_concepts=list(all_key_concepts),
        lessons=lessons,
    )


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default