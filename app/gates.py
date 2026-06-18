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
    # News is not teachable source material. Enforce the signal Agent 1 records in
    # meta — this also catches the inconsistent "is_news=True but accepted=True" case
    # the gate would otherwise wave through (Agent 1 gets 0 retries, so a halt here
    # is correct: re-fetching the same news URL can't make it teachable).
    if result.meta.get("is_news"):
        issues.append("source looks like a news article (time-sensitive reporting, not teachable)")
    if not issues and use_llm:
        verdict = llm.chat_json(
            system=(
                "You are a quality gate for article extraction. Decide whether the bulk of "
                "this text is substantive, teachable article content suitable for building "
                "lessons. Fail if the text is fundamentally unusable: mostly navigation/ads/"
                "link-lists, a login or paywall, marketing with no real content, or off-topic.\n"
                "ALSO fail if it is a NEWS ARTICLE or time-sensitive reporting (current events, "
                "press release, sports/market/political coverage) — even when well-written, news "
                "teaches no durable concept and is not valid source material. Evergreen tutorials, "
                "explainers, documentation, and in-depth conceptual articles are fine.\n"
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
                f"Published date (if any): {result.meta.get('published') or 'unknown'}\n"
                f"Agent 1's news suspicion (is_news): {result.meta.get('is_news')}\n"
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
    """Agent 2 gate: STRUCTURAL checks only.

    Agent 2 is extractive — each lesson body IS the raw source paragraphs for its
    paragraph range, so it cannot hallucinate and there is nothing to fact-check
    here (faithfulness moved to Agent 3, the sole rewrite step). What CAN go wrong
    is structure: no lessons, non-distinct lessons, or paragraph ranges that overlap
    (redundancy) or leave gaps (we missed part of the source). `use_llm` is accepted
    for signature compatibility but unused — these checks are deterministic.
    """
    issues: list[str] = []

    if not curriculum.lessons:
        return GateResult(passed=False, issues=["no lessons produced"])

    # A single lesson spanning the whole article tiles perfectly but isn't segmentation.
    # The coverage check below can't catch that (1 lesson trivially tiles), so guard count.
    if len(curriculum.lessons) < config.MIN_LESSONS:
        issues.append(
            f"only {len(curriculum.lessons)} lesson(s) — source not segmented "
            f"(need >= {config.MIN_LESSONS} distinct micro-lessons)"
        )

    titles = [l.title.strip().lower() for l in curriculum.lessons]
    if len(titles) != len(set(titles)):
        issues.append("duplicate lesson titles (not distinct)")

    # Deterministic coverage: the paragraph ranges must tile the source cleanly.
    issues.extend(_check_coverage(curriculum.lessons, curriculum.n_source_paragraphs))

    return GateResult(passed=not issues, issues=issues)


def _check_coverage(lessons: list, n_source_paragraphs: int = 0) -> list[str]:
    """The lessons' paragraph ranges should tile [0..N) contiguously: start at 0 and
    each lesson picks up exactly where the previous left off — no gap (source left
    in no lesson => 'we missed something') and no overlap (two lessons on the same
    paragraphs => redundancy). Exact arithmetic, no LLM. Skipped when ranges are
    unset (e.g. mock/hand-built curricula have start_idx == -1).

    `n_source_paragraphs` is the FULL source paragraph count: if the lessons stop short
    of it, the segmenter dropped the tail (e.g. a long article past SEGMENT_MAX_PARAGRAPHS)
    — which the per-lesson tiling alone can't see, since it only knows the paragraphs it got."""
    ranged = [l for l in lessons if getattr(l, "start_idx", -1) >= 0
              and getattr(l, "end_idx", -1) >= 0]
    if len(ranged) != len(lessons):
        return []   # indices unavailable — nothing deterministic to check

    issues: list[str] = []
    expected_start = 0
    for start, end, order in sorted((l.start_idx, l.end_idx, l.order) for l in ranged):
        if end < start:
            issues.append(f"lesson {order}: inverted paragraph range [{start},{end}]")
            continue
        if start > expected_start:
            issues.append(
                f"coverage gap: source paragraphs {expected_start}-{start - 1} are in no lesson"
            )
        elif start < expected_start:
            issues.append(f"lesson {order}: paragraph range overlaps the previous lesson")
        expected_start = max(expected_start, end + 1)

    # Tail loss: lessons end before the source does (the segmenter's cap dropped paragraphs).
    if n_source_paragraphs and expected_start < n_source_paragraphs:
        issues.append(
            f"dropped tail: {n_source_paragraphs - expected_start} source paragraph(s) "
            f"({expected_start}-{n_source_paragraphs - 1}) are past the segmenter cap and in no lesson"
        )
    return issues


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
        # Faithfulness is judged against the WHOLE course, not one slice: a term
        # established in another lesson is in-source, not invented. (Agent 2 is
        # extractive + tiles the article, so this concatenation ~= the full source.)
        full_source = "\n\n".join(
            l.body for l in sorted(curriculum.lessons, key=lambda x: x.order)
        )[:config.FAITHFULNESS_SOURCE_CHARS]
        by_user: dict[str, list[PersonalizedLesson]] = {}
        for p in personalized:
            by_user.setdefault(p.user, []).append(p)
        profiles = {u.name: u for u in users}
        for user, lessons in by_user.items():
            issues.extend(_judge_personalization_confirmed(profiles.get(user), lessons, originals, full_source))

    return GateResult(passed=not issues, issues=issues)


def _judge_personalization_confirmed(user: UserProfile | None, lessons: list[PersonalizedLesson],
                                     originals: dict[int, "object"], full_source: str) -> list[str]:
    """Judge once at temp 0; if it finds nothing, confirm with a second, hotter
    sample before declaring this learner's lessons clean. A FAIL is trusted at once;
    only a PASS is double-checked.

    Closes the failure mode the independent auditor caught: the Supervisor re-runs a
    failed stage until its gate passes, but one judge call is a single noisy sample,
    so the loop tends to stop on the first *lenient* sample — letting an invented
    "~3x faster" metric reach the final, gate-passed artifact. The confirmation draw
    runs hotter (not a temp-0 re-run, which would just reproduce the lenient verdict)
    so it is a genuinely independent sample; any high-severity finding blocks. Extra
    cost lands only on the attempts that were about to pass."""
    issues = _judge_personalization(user, lessons, originals, full_source, temperature=0.0)
    if not issues:
        issues = _judge_personalization(user, lessons, originals, full_source, temperature=0.5)
    return issues


def _judge_personalization(user: UserProfile | None, lessons: list[PersonalizedLesson],
                           originals: dict[int, "object"], full_source: str,
                           *, temperature: float = 0.0) -> list[str]:
    """Compare each tailored lesson to its original; block on hallucination or clear
    level/tone mismatch. Stylistic nitpicks are low-severity and don't block.

    Hallucination is judged against `full_source` (the whole course), so a concept
    established in another lesson is not mistaken for an invention.
    `temperature` lets the confirmation pass draw an independent (hotter) sample."""
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
            '- "high": the tailored lesson adds facts/numbers/formulas NOT supported anywhere '
            "in the FULL SOURCE (shown below) and NOT cited (hallucination), OR is clearly "
            "wrong for the learner's stated level (e.g. dense math given to a self-described "
            "beginner). NOTE: a term that appears elsewhere in the FULL SOURCE (another "
            "lesson) is faithful, not an invention — do not flag it.\n"
            '- "high": a FORCED or DECORATIVE analogy — it relabels the source content in '
            "the learner's own jargon without making a genuinely hard idea easier to grasp "
            "(e.g. a cooking recipe rewritten so eggs are 'activation functions' and flour is "
            "'hyperparameters'). A good analogy clarifies a real difficulty; a meaningless one "
            "just translates vocabulary and must be sent back so the analogy is removed.\n"
            '- "low": minor tone/style mismatch, could be tailored more, OR a term that IS in '
            "the FULL SOURCE but is used without much explanation. Using an in-source term "
            "without defining it is a clarity nuance, not a faithfulness error — and is "
            "appropriate for an advanced learner. Not blocking.\n"
            "A faithful rephrasing at the right level is GOOD — do not raise high for that. A "
            "term/fact present in the FULL SOURCE needs NO citation; only background ADDED from "
            "outside the source needs one — do not flag missing citations for in-source content.\n"
            'Return JSON {"issues": [{"severity": "high"|"low", "lesson": int, '
            '"problem": "<one short sentence>"}]}.'
        ),
        user=(f"LEARNER PROFILE:\n{user.raw}\n\n"
              f"FULL SOURCE (everything the course may faithfully draw on):\n{full_source}\n\n"
              f"LESSONS:\n{blob}"),
        temperature=temperature,
    )
    return [
        f"{user.name} lesson {i.get('lesson', '?')}: {i.get('problem', 'issue')}"
        for i in verdict.get("issues", [])
        if str(i.get("severity", "")).lower() == "high"
    ]
