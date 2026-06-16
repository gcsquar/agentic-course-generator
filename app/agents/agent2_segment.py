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
import llm

def split_into_paragraphs(text: str) -> list[str]:
    """Divide text into separated abstracts (if only >= 50 symbols) for better ciatation. Criteria: \n\n"""
    """Разбивает текст на осмысленные абзацы (>= 50 символов) для точного цитирования."""
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
        "You are an expert curriculum designer. Split the numbered paragraphs into semantic micro-lessons.\n"
        "REQUIREMENTS:\n"
        "- Create 3–8 micro-lessons.\n"
        "- Each micro-lesson = a continuous range of paragraphs [start_idx, end_idx] (inclusive).\n"
        "- Paragraphs within a single micro-lesson must be logically connected.\n"
        "- Micro-lessons must cover the entire provided text sequentially, without gaps or overlaps.\n"
        "- Each micro-lesson is finished on it's own and can be used as separated educational material\n"
        "Return ONLY a valid JSON array in the following format:\n"
        "[\n"
        "  {\n"
        '    "id": "seg_001",\n'
        '    "title": "Short informative name",\n'
        '    "description": "What is this part about (1-2 lines)",\n'
        '    "key_concepts": ["concept1", "concept2", ...],\n'
        '    "start_idx": 0,\n'
        '    "end_idx": 5\n'
        "  }\n"
        "]\n"
    )

    user_prompt = f"SOURCE PARAGRAPHS:\n{numbered}"
    
    if feedback:
        user_prompt += "\n\nPAY ATTENTION (fix these issues from the previous attempt):\n" + "\n".join(f"- {f}" for f in feedback)

    response = llm.chat_json(
        system=system_prompt,
        user=user_prompt,
        temperature=0.2,
    )

    response = llm.chat_json(
        system=system_prompt,
        user=user_prompt,
        temperature=0.2,
    )

    if isinstance(response, dict) and "segments" in response:
        lessons_data = response["segments"]
    elif isinstance(response, list):
        lessons_data = response
    else:
        raise ValueError(f"LLM returned an invalid response format for segmentation: {type(response)}")
    
    if not lessons_data:
        raise ValueError("LLM returned an empty response for segmentation")

    lessons = []
    all_key_concepts = set()

    for i, seg in enumerate(lessons_data):
        try:
            start = max(0, int(seg.get("start_idx", 0)))
            end = min(len(working_paragraphs) - 1, int(seg.get("end_idx", start)))
            if end < start:
                start, end = end, start
        except (TypeError, ValueError):
            start, end = 0, len(working_paragraphs) - 1

        source_span = "\n\n".join(working_paragraphs[start : end + 1])
        
        body = seg.get("body", "").strip()
        if not body:
            body = f"### {seg.get('title', 'Lesson')}\n\n{seg.get('description', 'Content pending.')}"

        lesson = Lesson(
            order=i + 1,
            title=seg.get("title", f"Micro-lesson {i+1}").strip(),
            body=body,
            source_span=source_span,
            description=seg.get("description", "").strip(),
            key_concepts=[kc.strip() for kc in seg.get("key_concepts", []) if kc.strip()],
            n_formulas=0
        )
        lessons.append(lesson)
        all_key_concepts.update(lesson.key_concepts)

    return Curriculum(
        source_url=source.url,
        key_concepts=list(all_key_concepts),
        lessons=lessons,
    )