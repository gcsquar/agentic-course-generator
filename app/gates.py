"""Quality-gate framework.

Each gate is a function (input) -> GateResult(passed, issues).
Pattern: cheap deterministic checks first, then (optionally) an LLM judge.
The Supervisor re-runs the failing agent with `issues` as feedback.

Phase 0 ships deterministic stubs; the LLM-judge halves are added in Phase A/B.
"""
from __future__ import annotations

import config
from contracts import (IngestResult, Curriculum, GateResult,
                       PersonalizedLesson, UserProfile)
import llm


def gate_ingest(result: IngestResult, *, use_llm: bool = True) -> GateResult:
    """Agent 1 gate: did we get clean, teachable article text?

    Cheap deterministic checks run always; the LLM judge runs only when
    `use_llm` is True (mock runs pass use_llm=False to stay offline).
    """
    issues: list[str] = []
    if not result.accepted:
        issues.append(f"Agent 1 rejected the source: {result.reason}")
    if len(result.clean_text) < config.MIN_ARTICLE_CHARS:
        issues.append(
            f"clean_text too short ({len(result.clean_text)} < {config.MIN_ARTICLE_CHARS} chars)"
        )
    if not issues and use_llm:
        verdict = llm.chat_json(
            system=(
                "You are a quality gate for article extraction. Decide whether the bulk of "
                "this text is substantive, teachable article content. Fail ONLY if the text "
                "is fundamentally unusable: mostly navigation/ads/link-lists, a login or "
                "paywall, marketing with no real content, or off-topic.\n"
                "DO NOT fail for cosmetic reasons: markdown formatting, headings, citation "
                "markers, reference lists, or because the excerpt is truncated mid-sentence "
                "(you are shown only the first part). Minor boilerplate leakage is acceptable "
                "as long as real teaching content dominates.\n"
                'Return JSON {"passed": boolean, "issues": [string]} — issues only for '
                "blocking problems."
            ),
            user=(
                f"Title: {result.title}\n"
                f"URL: {result.url}\n"
                f"Extracted excerpt (first 4000 chars of a longer document):\n"
                f"{result.clean_text[:4000]}"
            ),
            temperature=0.0,
        )
        if not verdict.get("passed", False):
            judge_issues = verdict.get("issues") or ["LLM judge rejected extracted article prose"]
            issues.extend(str(issue) for issue in judge_issues)
    return GateResult(passed=not issues, issues=issues)


def gate_segment(curriculum: Curriculum, source: IngestResult,
                 *, use_llm: bool = True) -> GateResult:
    """Agent 2 gate: standalone? distinct? faithful to source?

    Deterministic checks run always; the LLM faithfulness judge runs when use_llm.
    """
    issues: list[str] = []

    # --- deterministic checks ---
    if not curriculum.lessons:
        issues.append("no lessons produced")
    titles = [l.title.strip().lower() for l in curriculum.lessons]
    if len(titles) != len(set(titles)):
        issues.append("duplicate lesson titles (not distinct)")
    # Summaries legitimately contain FEWER formulas than the source, so we only
    # flag the clear hallucination signal: formulas that appear from nowhere.
    if source.n_formulas == 0 and sum(l.n_formulas for l in curriculum.lessons) > 0:
        issues.append("lessons contain formulas absent from the source (possible invention)")

    # --- LLM faithfulness judge ---
    if use_llm and curriculum.lessons:
        issues.extend(_judge_faithfulness(curriculum, source))

    return GateResult(passed=not issues, issues=issues)


def _judge_faithfulness(curriculum: Curriculum, source: IngestResult) -> list[str]:
    """Diff each lesson against the source.

    Returns only BLOCKING (high-severity) issues — hallucinations / unsupported
    claims / non-standalone lessons. Subjective 'could cover more' notes are
    low-severity and deliberately do NOT fail the gate (else retries never settle).
    """
    lessons_blob = "\n\n".join(
        f"### Lesson {l.order}: {l.title}\n{l.body}\n(claimed source excerpt: {l.source_span})"
        for l in curriculum.lessons
    )
    verdict = llm.chat_json(
        system=(
            "You are a strict faithfulness reviewer for educational lessons. "
            "Compare the generated lessons against the SOURCE article and list issues, each "
            "tagged with a severity:\n"
            '- "high": a claim, fact, number, or formula NOT supported by the source '
            "(hallucination), or a lesson that is not self-contained/standalone.\n"
            '- "low": the lesson is faithful but could cover more, or a minor topic is '
            "omitted. These are suggestions, not errors.\n"
            "Do NOT raise high severity merely because a lesson is a concise summary. "
            'Return JSON {"issues": [{"severity": "high"|"low", "lesson": int, '
            '"problem": "<one short sentence>"}]}.'
        ),
        user=(
            f"SOURCE (title: {source.title}):\n{source.clean_text[:config.SEGMENT_INPUT_CHARS]}\n\n"
            f"LESSONS:\n{lessons_blob}"
        ),
        temperature=0.0,
    )
    blocking = [
        f"Lesson {i.get('lesson', '?')}: {i.get('problem', 'unsupported content')}"
        for i in verdict.get("issues", [])
        if str(i.get("severity", "")).lower() == "high"
    ]
    return blocking


def gate_personalize(personalized: list[PersonalizedLesson], curriculum: Curriculum,
                     users: list[UserProfile], *, use_llm: bool = True) -> GateResult:
    """Agent 3 gate: every lesson tailored for every user, faithful, level-appropriate?"""
    issues: list[str] = []

    # --- deterministic checks: full coverage, no empties ---
    expected = {(u.name, l.order) for u in users for l in curriculum.lessons}
    got = {(p.user, p.order) for p in personalized}
    for user, order in sorted(expected - got):
        issues.append(f"{user}: missing personalized lesson {order}")
    for p in personalized:
        if not p.body.strip():
            issues.append(f"{p.user}: lesson {p.order} is empty")

    # --- LLM judge per user (faithful to original + fits the profile) ---
    if use_llm:
        originals = {l.order: l for l in curriculum.lessons}
        by_user: dict[str, list[PersonalizedLesson]] = {}
        for p in personalized:
            by_user.setdefault(p.user, []).append(p)
        profiles = {u.name: u for u in users}
        for user, lessons in by_user.items():
            issues.extend(_judge_personalization(profiles.get(user), lessons, originals))

    return GateResult(passed=not issues, issues=issues)


def _judge_personalization(user: UserProfile | None, lessons: list[PersonalizedLesson],
                           originals: dict[int, "object"]) -> list[str]:
    """Compare each tailored lesson to its original; block on hallucination or clear
    level/tone mismatch. Stylistic nitpicks are low-severity and don't block."""
    if user is None:
        return []
    blob = "\n\n".join(
        f"### [order {p.order}] {p.title}\n"
        f"ORIGINAL: {getattr(originals.get(p.order), 'body', '(missing)')}\n"
        f"TAILORED: {p.body}"
        for p in sorted(lessons, key=lambda x: x.order)
    )
    verdict = llm.chat_json(
        system=(
            "You review personalized lessons against their originals for one learner. "
            "List issues, each with a severity:\n"
            '- "high": the tailored lesson adds facts/numbers/formulas NOT in the original '
            "and NOT cited (hallucination), OR is clearly wrong for the learner's stated "
            "level (e.g. dense math given to a self-described beginner).\n"
            '- "low": minor tone/style mismatch or could be tailored more. Not an error.\n'
            "A faithful rephrasing at the right level is GOOD — do not raise high for that. "
            'Return JSON {"issues": [{"severity": "high"|"low", "lesson": int, '
            '"problem": "<one short sentence>"}]}.'
        ),
        user=f"LEARNER PROFILE:\n{user.raw}\n\nLESSONS:\n{blob}",
        temperature=0.0,
    )
    return [
        f"{user.name} lesson {i.get('lesson', '?')}: {i.get('problem', 'issue')}"
        for i in verdict.get("issues", [])
        if str(i.get("severity", "")).lower() == "high"
    ]
