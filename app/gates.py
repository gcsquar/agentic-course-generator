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
import source_index


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
            model=config.JUDGE_MODEL,
        )
        passed = verdict.get("passed")
        if passed is None and "accepted" in verdict:
            passed = verdict.get("accepted")
        if not passed:
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

    # Smell check (generous): a lesson spanning a huge run of paragraphs has almost certainly
    # lumped several subtopics — flag it so the segmenter re-splits. Not a hard size law (a
    # coherent subtopic may be large); it only fires on egregious dumping.
    for l in curriculum.lessons:
        if l.start_idx >= 0 and (l.end_idx - l.start_idx + 1) > config.MAX_LESSON_PARAGRAPHS:
            issues.append(
                f"lesson {l.order} spans {l.end_idx - l.start_idx + 1} paragraphs "
                f"(> {config.MAX_LESSON_PARAGRAPHS}) — likely lumps several subtopics; split it"
            )

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

    originals = {l.order: l for l in curriculum.lessons}
    for p in personalized:
        original = originals.get(p.order)
        if original is not None:
            issues.extend(_check_supporting_quotes(p, getattr(original, "body", "")))
            issues.extend(_check_supporting_sentences(p, getattr(original, "body", "")))

    # --- LLM judge per user (faithful to original + fits the profile) ---
    if use_llm:
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
    """Judge once at temp 0; a FAIL is trusted at once. A clean PASS is re-checked with a
    second, INDEPENDENT sample before it's declared clean — closing the retry-until-lucky
    loophole (INSIGHTS #2): the Supervisor re-runs a failed stage until its gate passes,
    but one judge call is a single noisy sample, so the loop tends to stop on the first
    *lenient* one — letting an invented "~3x faster" metric reach the gate-passed output.

    What makes the confirmation draw independent matters (ROADMAP 2.4). Best: a DIFFERENT
    model (`config.CONFIRM_MODEL`) — independence by construction, regardless of sampling.
    Fallback (no CONFIRM_MODEL): a hotter-temperature draw of the same judge — which is a
    genuine resample ONLY on models that actually vary with temperature; a reasoning model
    that pins temperature makes it a no-op (then confirmation is just a temp-0 re-run, no
    worse than a single sample). Either way a FAIL is trusted immediately, so extra cost
    lands only on attempts that were about to pass. Any high finding from either blocks."""
    issues = _judge_personalization(user, lessons, originals, full_source, temperature=0.0)
    if issues:
        return issues
    if not config.CONFIRM_PERSONALIZE_GATE:
        return []
    confirm_model = config.CONFIRM_MODEL
    if confirm_model and confirm_model != config.JUDGE_MODEL:
        # Genuinely independent second opinion from a different model (temperature irrelevant).
        return _judge_personalization(user, lessons, originals, full_source,
                                      temperature=0.0, model=confirm_model)
    # No distinct model configured: fall back to a hotter draw of the same judge.
    return _judge_personalization(user, lessons, originals, full_source, temperature=0.5)


def _judge_personalization(user: UserProfile | None, lessons: list[PersonalizedLesson],
                           originals: dict[int, "object"], full_source: str,
                           *, temperature: float = 0.0, model: str | None = None) -> list[str]:
    """Compare each tailored lesson to its own source span.

    The old gate judged a whole user's course in one large prompt and missed unsupported
    details buried mid-bundle. This keeps the public helper shape but performs smaller,
    stricter per-lesson reviews internally.
    """
    if user is None:
        return []
    out: list[str] = []
    for p in sorted(lessons, key=lambda x: x.order):
        original = originals.get(p.order)
        if original is None:
            continue
        out.extend(_judge_one_personalized_lesson(
            user, p, getattr(original, "body", ""), full_source_context=full_source,
            temperature=temperature, model=model
        ))
    return out


def _judge_one_personalized_lesson(user: UserProfile, lesson: PersonalizedLesson,
                                   source_span: str, *, full_source_context: str = "",
                                   temperature: float = 0.0, model: str | None = None) -> list[str]:
    """Strict faithfulness judge for one generated lesson and its source span."""
    indexed_source = source_index.render_indexed_source(source_span)
    verdict = llm.chat_json(
        system=(
            "You review ONE personalized lesson against the exact source span it rewrote. "
            "The source is indexed as `<s id=...>` sentences. List issues, each with a severity. "
            "Be strict about source support and citation entailment:\n"
            '- "high": the tailored lesson adds a factual claim, number, formula, complexity '
            "claim, tool/API detail, implementation recommendation, tradeoff, or model name "
            "that is NOT explicitly supported by the INDEXED SOURCE and is NOT clearly marked as "
            "cited external background (hallucination), OR is clearly "
            "wrong for the learner's stated level (e.g. dense math given to a self-described "
            "beginner). Topic words appearing in the source do NOT support a specific CLAIM: "
            "if the lesson describes, summarizes, or characterizes a named item/tool that the "
            "SOURCE only names without describing, that description is invented — flag it high.\n"
            '- "high": a MODEL-PROVIDED SUPPORTING SENTENCE entry cites sentence IDs that do '
            "not directly entail its claim. Sentence IDs are anchors, not proof by themselves.\n"
            '- "high": a FORCED or DECORATIVE analogy — it relabels the source content in '
            "the learner's own jargon without making a genuinely hard idea easier to grasp "
            "(e.g. a cooking recipe rewritten so eggs are 'activation functions' and flour is "
            "'hyperparameters'). A good analogy clarifies a real difficulty; a meaningless one "
            "just translates vocabulary and must be sent back so the analogy is removed.\n"
            '- "low": minor tone/style mismatch, could be tailored more, OR a term that IS in '
            "the INDEXED SOURCE but is used without much explanation. Using an in-source term "
            "without defining it is a clarity nuance, not a faithfulness error — and is "
            "appropriate for an advanced learner. Not blocking.\n"
            "Use the ARTICLE-WIDE SOURCE CONTEXT only to resolve terms, aliases, and acronyms "
            "defined elsewhere in the article; do not flag an acronym expansion when that "
            "context explicitly defines it. Keep substantive lesson claims grounded in the "
            "INDEXED SOURCE or cited external background.\n"
            "A faithful rephrasing at the right level is GOOD — do not raise high for that. A "
            "term/fact present in the INDEXED SOURCE needs NO citation; only background ADDED from "
            "outside the source needs one — do not flag missing citations for in-source content.\n"
            'Return JSON {"issues": [{"severity": "high"|"low", "lesson": int, '
            '"problem": "<one short sentence>"}]}.'
        ),
        user=(
            f"LEARNER PROFILE:\n{user.raw}\n\n"
            f"LESSON ORDER: {lesson.order}\n"
            f"TITLE: {lesson.title}\n\n"
            f"INDEXED SOURCE:\n{indexed_source}\n\n"
            f"ARTICLE-WIDE SOURCE CONTEXT FOR TERMS/ALIASES/ACRONYMS ONLY:\n"
            f"{full_source_context or source_span}\n\n"
            f"MODEL-PROVIDED SUPPORTING SENTENCES:\n"
            f"{getattr(lesson, 'supporting_sentences', [])}\n\n"
            f"CITATIONS FOR EXTERNAL BACKGROUND:\n{lesson.citations or []}\n\n"
            f"TAILORED LESSON:\n{lesson.body}"
        ),
        temperature=temperature,
        model=model or config.JUDGE_MODEL,
    )
    # The agent's retry router keys feedback to a lesson by its order number, so a finding
    # MUST carry a real one or it gets dropped. Only emit "lesson N" when N is an order the
    # judge actually saw; if it omitted the number or named a non-existent lesson, tag it
    # "lesson unspecified" so the agent broadcasts it to all of the user's lessons instead.
    out: list[str] = []
    for i in verdict.get("issues", []):
        if str(i.get("severity", "")).lower() != "high":
            continue
        problem = i.get("problem", "issue")
        try:
            order = int(i.get("lesson"))
        except (TypeError, ValueError):
            order = None
        where = f"lesson {order}" if order == lesson.order else f"lesson {lesson.order}"
        out.append(f"{user.name} {where}: {problem}")
    return out


def _check_supporting_quotes(lesson: PersonalizedLesson, source_span: str) -> list[str]:
    """Validate Agent 3's evidence contract.

    This is intentionally not a keyword blacklist: the model may include any detail it
    can ground, but each supplied quote must be a real substring of the source span.
    """
    issues: list[str] = []
    source_norm = _quote_norm(source_span)
    for idx, item in enumerate(getattr(lesson, "supporting_quotes", []) or [], start=1):
        quote = str(item.get("source_quote") or "").strip() if isinstance(item, dict) else ""
        claim = str(item.get("claim") or "").strip() if isinstance(item, dict) else ""
        if not quote:
            issues.append(f"{lesson.user} lesson {lesson.order}: supporting quote {idx} is empty")
            continue
        if not _quote_supported(quote, source_span, source_norm):
            label = f" for claim '{claim[:80]}'" if claim else ""
            issues.append(
                f"{lesson.user} lesson {lesson.order}: supporting quote {idx}{label} "
                "does not appear in the source span"
            )
    return issues


def _check_supporting_sentences(lesson: PersonalizedLesson, source_span: str) -> list[str]:
    """Validate sentence-ID evidence anchors from Agent 3."""
    issues: list[str] = []
    valid_ids = source_index.valid_sentence_ids(source_span)
    for idx, item in enumerate(getattr(lesson, "supporting_sentences", []) or [], start=1):
        if not isinstance(item, dict):
            issues.append(f"{lesson.user} lesson {lesson.order}: supporting sentence entry {idx} is invalid")
            continue
        claim = str(item.get("claim") or "").strip()
        raw_ids = item.get("sentence_ids") or item.get("source_ids") or item.get("ids") or []
        if isinstance(raw_ids, (str, int)):
            raw_ids = [raw_ids]
        ids = [str(s).strip() for s in raw_ids if str(s).strip()]
        if not claim:
            issues.append(f"{lesson.user} lesson {lesson.order}: supporting sentence entry {idx} has no claim")
        if not ids:
            issues.append(f"{lesson.user} lesson {lesson.order}: supporting sentence entry {idx} has no sentence IDs")
            continue
        bad = [sid for sid in ids if sid not in valid_ids]
        if bad:
            issues.append(
                f"{lesson.user} lesson {lesson.order}: supporting sentence entry {idx} "
                f"references unknown source sentence ID(s): {', '.join(bad)}"
            )
    return issues


def _quote_norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _quote_words(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _quote_supported(quote: str, source_span: str, source_norm: str) -> bool:
    """Return whether a quote is grounded in the source, allowing markdown noise.

    Exact normalized substring is preferred. The fallback compares the quote against
    source windows of similar length so harmless punctuation/markdown differences do
    not trigger expensive retries.
    """
    quote_norm = _quote_norm(quote)
    if not quote_norm:
        return False
    if quote_norm in source_norm:
        return True

    from difflib import SequenceMatcher
    quote_words = _quote_words(quote)
    source_words = _quote_words(source_span)
    n = len(quote_words)
    if n < 4 or not source_words:
        return False
    stride = max(1, n // 3)
    best = 0.0
    for start in range(0, max(1, len(source_words) - n + 1), stride):
        window = " ".join(source_words[start:start + n])
        score = SequenceMatcher(None, " ".join(quote_words), window).ratio()
        if score > best:
            best = score
        if best >= 0.65:
            return True
    return False
