"""Agent 3 — Profile-Based Personalization.

CONTRACT (do not change the signature — the Supervisor depends on it):
    personalize(curriculum: Curriculum, users: list[UserProfile], *,
                mock: bool = False, use_llm: bool = True,
                feedback: list[str] | None = None,
                do_research: bool = True) -> list[PersonalizedLesson]

Job:  lessons + learner profiles  ->  one tailored lesson per (user, lesson).
- `feedback` carries the gate's issues on a retry (tagged "<Name> lesson N: ...").
- `do_research` enables the gap-filling loop: use `research.py` to fetch background
  from TRUSTED sources and attach a CITED note (citations=[url]).
Output must satisfy `gates.gate_personalize` (level-appropriate, faithful, cited).

Note: learner loading lives in `user_profiles.parse_users` (shared), not here.

IMPLEMENTATION NOTES
---------------------
Per (user, lesson) pair the agent does up to two LLM calls:

  1. Draft  — rewrite the original lesson for this ONE learner (level, tone,
     interests, focus, language all come from `user.raw`, which is fed to the
     model verbatim — no separate "profile parsing" step needed). The model is
     told to never invent facts and to flag a `needs_background` topic instead
     of inventing it, if the lesson assumes something a beginner likely lacks.

  2. Weave  — only runs when (a) `do_research` is on, (b) the learner looks
     like a beginner, and (c) the draft flagged a real gap. `research.py`
     searches the web and returns text ONLY from `config.TRUSTED_DOMAINS`
     (the open web is dirty — no trusted hit means no background is added,
     rather than citing a dirty source). The returned text is folded into the
     lesson as a clearly marked, cited aside; the rest of the lesson is left
     untouched.

Retry feedback from `gates.gate_personalize` is tagged per learner+lesson
(e.g. "Mike lesson 2: ..." or "Mike: lesson 2 is empty"), so we filter the
flat `feedback` list down to the few lines relevant to the (user, lesson) pair
being regenerated and put only those in the prompt.

Users are processed in parallel (small thread pool) since each learner's
output is independent — this mirrors the original design's "parallel
PersonalizationAgent(u1..uN)" fan-out. Each user's own lessons are generated
sequentially (gap-research for lesson N may want lesson N's own draft first).

This file has no import-time dependency on Agent 1 or Agent 2's *implementation*
— only on the shared `contracts.py` dataclasses — so it can be built, tested,
and run entirely against hand-built mock `Curriculum`/`UserProfile` objects.
See tests/test_agent3_personalize.py.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib.parse

from contracts import Curriculum, Lesson, PersonalizedLesson, UserProfile
import config
import llm
import research
import source_index

# Fan-out across learners only (each learner's own lessons run sequentially).
_MAX_WORKERS = 4

_BEGINNER_LEVELS = {"beginner", "novice", "new", "intro", "introductory"}


def personalize(curriculum: Curriculum, users: list[UserProfile], *,
                mock: bool = False, use_llm: bool = True,
                feedback: list[str] | None = None,
                do_research: bool = True,
                previous: list[PersonalizedLesson] | None = None) -> list[PersonalizedLesson]:
    if mock or not use_llm:
        out: list[PersonalizedLesson] = []
        for u in users:
            for l in curriculum.lessons:
                out.append(PersonalizedLesson(
                    user=u.name, order=l.order, title=l.title,
                    body=f"[for {u.name} · {u.level}] {l.body}",
                    supporting_quotes=[],
                    supporting_sentences=[],
                ))
        return out

    feedback = feedback or []
    results: list[PersonalizedLesson] = []
    previous_by_key = {(p.user, p.order): p for p in (previous or [])}

    def _do_user(user: UserProfile) -> list[PersonalizedLesson]:
        return _personalize_for_user(user, curriculum, feedback, do_research, previous_by_key)

    if len(users) > 1:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(users))) as pool:
            futures = {pool.submit(_do_user, u): u for u in users}
            for fut in as_completed(futures):
                results.extend(fut.result())
    else:
        for u in users:
            results.extend(_do_user(u))

    # Deterministic order regardless of thread completion order: input user
    # order, then lesson order — matches what orchestrator/_write_readable expects.
    user_rank = {u.name: i for i, u in enumerate(users)}
    results.sort(key=lambda p: (user_rank.get(p.user, 0), p.order))
    return results


# --------------------------------------------------------------------- per-user

def _personalize_for_user(user: UserProfile, curriculum: Curriculum,
                           feedback: list[str], do_research: bool,
                           previous_by_key: dict[tuple[str, int], PersonalizedLesson]) -> list[PersonalizedLesson]:
    user_feedback = [f for f in feedback if user.name.lower() in f.lower()]
    out: list[PersonalizedLesson] = []
    covered: list[str] = []   # concepts taught in EARLIER lessons of this course
    for lesson in sorted(curriculum.lessons, key=lambda l: l.order):
        lesson_feedback = _feedback_for_lesson(user_feedback, lesson.order)
        previous_lesson = previous_by_key.get((user.name, lesson.order))
        if previous_lesson and feedback and not lesson_feedback:
            # On a retry, preserve lessons the gate did not complain about. Regenerating
            # clean lessons wastes calls and can introduce new unsupported claims.
            out.append(previous_lesson)
            covered.extend(lesson.key_concepts)
            continue
        out.append(_personalize_lesson(
            user, lesson, curriculum, lesson_feedback, do_research, covered, previous_lesson
        ))
        # A term introduced here is "already taught" for every later lesson, so a
        # later lesson may reference it freely instead of flagging it as background.
        covered.extend(lesson.key_concepts)
    return out


def _feedback_for_lesson(user_feedback: list[str], order: int) -> list[str]:
    """Pick the feedback lines (already filtered to THIS user) relevant to THIS lesson:

      - a line naming this lesson's order (word-boundary match, so 'lesson 1' doesn't
        also catch 'lesson 10') -> precise routing, the common case;
      - a line that names NO lesson order at all -> user-level feedback the gate couldn't
        pin to a lesson (e.g. an LLM judge that omitted the number, surfaced by the gate as
        'lesson unspecified'). Broadcast it to every lesson rather than silently drop it —
        the regex router needs a real N, and a dropped issue means the model never sees a
        wish it was supposed to fix.

    A line naming a DIFFERENT lesson's number is excluded — it belongs to that lesson."""
    this_lesson = re.compile(rf"lesson\s+{order}\b", re.IGNORECASE)
    any_lesson = re.compile(r"lesson\s+\d+\b", re.IGNORECASE)
    return [f for f in user_feedback
            if this_lesson.search(f) or not any_lesson.search(f)]


# ------------------------------------------------------------------- per-lesson

def _personalize_lesson(user: UserProfile, lesson: Lesson, curriculum: Curriculum,
                         lesson_feedback: list[str], do_research: bool,
                         covered: list[str], previous_lesson: PersonalizedLesson | None) -> PersonalizedLesson:
    repairing = bool(lesson_feedback and previous_lesson and not previous_lesson.fallback)
    if repairing:
        draft = _repair_draft(user, lesson, curriculum, lesson_feedback, covered, previous_lesson)
    else:
        draft = _generate_draft(user, lesson, curriculum, lesson_feedback, covered)
    body = (draft.get("body") or "").strip() or lesson.body
    supporting_sentences = _supporting_sentences(draft.get("supporting_sentences"))
    citations: list[str] = list(previous_lesson.citations) if repairing and previous_lesson else []

    # Fail-soft visibility (ROADMAP 3.2): if the draft LLM call failed we shipped the
    # UNtailored original — flag it so it isn't silently passed off as personalized (the
    # faithfulness gate won't catch it: the original is faithful, just not tailored).
    failed = bool(draft.get("_draft_failed"))
    if failed:
        print(f"[agent3] {user.name} lesson {lesson.order}: draft failed "
              f"({draft.get('_error', 'error')}) — shipping untailored original (flagged fallback)")

    # Gap-filling research is for BEGINNERS only: an expert reading in their own field
    # doesn't need fetched background, and adding "go deeper" Scholar links to an expert's
    # lesson reads as ungrounded external clutter (the auditor flagged exactly that). Skip
    # it on a failed draft — there's no real tailoring to fill a gap into.
    gap_topic = draft.get("needs_background")
    if do_research and gap_topic and not failed and _looks_like_beginner(user):
        query = _research_query(gap_topic, user)
        found = _safe_research(query)
        if found:
            text, url = found
            body = _weave_background(body, gap_topic, text, url)
            citations.append(url)
            print(f"[research] HIT  '{gap_topic}' -> {url}")
        else:
            body = _append_scholar_fallback(body, gap_topic, query)
            print(f"[research] MISS '{gap_topic}' — no trusted source, Scholar fallback")

    return PersonalizedLesson(
        user=user.name, order=lesson.order, title=lesson.title,
        body=body, citations=citations, supporting_quotes=[],
        supporting_sentences=supporting_sentences,
        topic_fit=draft.get("topic_fit") or "",
        fallback=failed,
    )


def _supporting_sentences(value) -> list[dict[str, object]]:
    """Normalize the model's evidence contract to {claim, sentence_ids}."""
    if not isinstance(value, list):
        return []
    out: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        raw_ids = item.get("sentence_ids") or item.get("source_ids") or item.get("ids") or []
        if isinstance(raw_ids, (str, int)):
            raw_ids = [raw_ids]
        ids = [str(s).strip() for s in raw_ids if str(s).strip()]
        if claim and ids:
            out.append({"claim": claim, "sentence_ids": ids})
    return out


def _looks_like_beginner(user: UserProfile) -> bool:
    level = (user.level or "").strip().lower()
    if level:
        return level in _BEGINNER_LEVELS
    return bool(re.search(r"\b(beginner|novice|new to)\b", user.raw or "", re.IGNORECASE))


def _research_query(gap_topic: str, user: UserProfile) -> str:
    """Build a search query adapted to the user's level."""
    level = (user.level or "").strip().lower()
    if level in _BEGINNER_LEVELS:
        return f"{gap_topic} introduction explained simply"
    if level == "expert":
        return f"{gap_topic} technical overview"
    return f"{gap_topic} key concepts overview"  # intermediate / unknown


def _append_scholar_fallback(body: str, gap_topic: str, query: str) -> str:
    """When no trusted source was found, add a Google Scholar pointer."""
    scholar_url = "https://scholar.google.com/scholar?q=" + urllib.parse.quote(query)
    note = (
        f"\n\n---\n"
        f"**To go deeper on \"{gap_topic}\":**  \n"
        f"Automated search did not find a suitable source in our trusted domains. "
        f"We recommend searching for it yourself:  \n"
        f"[Google Scholar: {query}]({scholar_url})"
    )
    return body + note


def _safe_research(query: str) -> tuple[str, str] | None:
    try:
        return research.research(query)
    except Exception:
        # A flaky network/search failure should degrade gracefully, not break the run.
        return None


# --------------------------------------------------------------------- LLM calls

def _generate_draft(user: UserProfile, lesson: Lesson, curriculum: Curriculum,
                     lesson_feedback: list[str], covered: list[str]) -> dict:
    feedback_note = ""
    if lesson_feedback:
        feedback_note = (
            "\n\nThe previous attempt for THIS learner and THIS lesson had these "
            "issues — fix them in this rewrite:\n- " + "\n- ".join(lesson_feedback)
        )

    system = (
        "You are an expert tutor rewriting one lesson for ONE specific learner, using "
        "their full profile below.\n\n"
        "STEP 1 — determine abstraction level BEFORE writing.\n"
        "Abstraction level = how concrete vs. how technical/precise the explanation should be. "
        "It is determined by TWO factors combined:\n\n"
        "Factor A — learner's stated level in their OWN domain:\n"
        "  beginner → high abstraction (intuition, plain language, everyday analogies)\n"
        "  intermediate → medium abstraction (some jargon ok, worked examples, key distinctions)\n"
        "  expert → low abstraction (precise terminology, assumes fundamentals, no hand-holding)\n\n"
        "Factor B — topic-profile alignment:\n"
        "  ALIGNED (article topic matches learner's domain/role/interests): use Factor A as-is.\n"
        "  MISALIGNED (topic is outside their domain): raise abstraction by one level regardless "
        "of Factor A. An expert in ML reading about Renaissance art is a smart non-specialist — "
        "NOT a domain beginner (don't over-simplify or patronize), but NOT a domain expert either "
        "(don't assume art-history knowledge). Give them the core idea, real-world significance, "
        "and AT MOST one analogy to their own domain — and only if it truly makes the idea click, "
        "never a forced or decorative one. Skip specialist jargon entirely.\n\n"
        "Combined abstraction levels:\n"
        "  expert + aligned    → precise, dense, minimal explanation of basics (cover tradeoffs ONLY if the source states them)\n"
        "  expert + misaligned → clear, jargon-free, core idea + an optional cross-domain analogy ONLY if it truly clarifies\n"
        "  intermediate + aligned    → worked examples, key terms defined once, some depth\n"
        "  intermediate + misaligned → accessible overview, skip sub-field specifics\n"
        "  beginner + aligned  → intuition first, step-by-step, explicit prerequisites\n"
        "  beginner + misaligned → very light touch, only the single most important takeaway\n\n"
        "Factor C — how the learner thinks about and uses information (infer from their role/profile):\n"
        "  Quantitative thinker (business, finance, engineering, data): prefer numbers, metrics, "
        "percentages, benchmarks, and concrete measurements THAT APPEAR IN THE SOURCE — surface "
        "and lead with them. If the lesson has no quantitative content, do NOT estimate or invent "
        "magnitudes ('~3x faster', 'cuts error rate in half') — present it qualitatively; making "
        "up numbers is a faithfulness violation, not personalization.\n"
        "  Qualitative thinker (humanities, arts, general audience, creative roles): prefer "
        "descriptive language ('significantly faster', 'much more accurate'). Numbers are fine "
        "when they appear in the source, but don't add or emphasize metrics that aren't central.\n"
        "  When in doubt, look at the learner's role and interests to decide: a manager or "
        "analyst likely wants numbers; a student or general reader is fine with words.\n\n"
        "STEP 2 — write the tailored lesson from SOURCE CLAIMS ONLY:\n"
        "- Faithfulness scope: every factual claim, formula, number, tradeoff, recommendation, "
        "tool/API mention, complexity statement, and implementation detail must be explicitly "
        "supported by the ORIGINAL lesson text below. Do NOT add facts from your general ML "
        "knowledge, even if they are true.\n"
        "- Earlier lessons are shown only for brief recall/context. You may refer to their topic "
        "names, but you may not introduce new factual claims from earlier lessons into this one.\n"
        "- If the learner asks for tradeoffs but the ORIGINAL lesson does not state a tradeoff, "
        "say 'The source does not cover that tradeoff' rather than filling it in.\n"
        "- Do not introduce alternate technical labels, algorithms, variants, or library names "
        "unless those exact ideas appear in the ORIGINAL lesson. Examples: do not say ordinary "
        "least squares, residual sum of squares, stochastic/mini-batch/online variants, SVD, "
        "SGDRegressor, Adam, or robust regression unless the ORIGINAL lesson says them.\n"
        "- If the lesson assumes a prerequisite that is in NEITHER the original NOR the "
        "already-taught list, do NOT invent or explain it from your own knowledge — name it "
        "in `needs_background` so it can be researched and cited.\n"
        "- Personalization changes HOW the source is presented (tone, density, level, ordering) "
        "— NEVER what it claims. Do NOT add your own critiques or judgments (e.g. 'this formula "
        "is wrong/suboptimal'), deployment or implementation advice, or tradeoff analysis the "
        "source does not state. A dense, precise rewrite for an expert is still a faithful "
        "SUMMARY — not a code review, an opinion, or a how-to.\n"
        "- If the source only NAMES or LISTS something without describing it (a linked or "
        "recommended article, a reference, a cited work, a tool), present it as named — do NOT "
        "invent its 'focus', a summary, what it is about, or why it fits. You know only what the "
        "source actually says about it; if that is just a title, give just the title.\n"
        "- Apply the abstraction level from Step 1 throughout: depth, vocabulary, "
        "example choice, and how much you explain vs. assume.\n"
        "- Use an analogy ONLY when it makes a genuinely difficult concept easier to grasp. "
        "An analogy that merely renames the content in the learner's jargon (e.g. calling "
        "recipe ingredients 'training data' or 'activation functions') adds no understanding "
        "and is worse than none — omit it. Never force one to satisfy a stated style.\n"
        "- Honor the learner's stated focus and preferred language if given in their profile.\n"
        "- Evidence contract: the source is shown as indexed `<s id=...>` sentences. Include "
        "`supporting_sentences` for the substantive claims in your rewrite. Each item must pair "
        "your claim with the sentence ID(s) that DIRECTLY support it. If no source sentence IDs "
        "directly support a claim, remove the claim.\n"
        "- In `topic_fit`: if misaligned, write one short sentence (e.g. 'Outside learner's "
        "domain — delivered as accessible overview for a smart non-specialist'). Empty if aligned.\n\n"
        'Return ONLY JSON: {"body": "<tailored lesson in markdown>", '
        '"supporting_sentences": [{"claim": "<claim in your rewrite>", '
        '"sentence_ids": ["1", "2"]}], '
        '"needs_background": "<short topic, or null>", '
        '"topic_fit": "<one sentence if misaligned, else empty string>"}'
    )
    indexed_source = source_index.render_indexed_source(
        lesson.body[:config.PERSONALIZE_LESSON_CHARS]
    )
    user_msg = (
        f"LEARNER PROFILE:\n{user.raw[:config.PERSONALIZE_PROFILE_CHARS]}\n\n"
        f"ARTICLE KEY CONCEPTS (navigation labels only; NOT factual support): "
        f"{', '.join(curriculum.key_concepts) or '(none listed)'}\n\n"
        f"EARLIER LESSON TOPICS (continuity labels only; NOT factual support for this lesson): "
        f"{', '.join(dict.fromkeys(covered)) or '(none yet — this is among the first lessons)'}\n\n"
        f"ORIGINAL LESSON (order {lesson.order}): {lesson.title}\n"
        f"{indexed_source}\n"
        f"Lesson key concepts: {', '.join(lesson.key_concepts) or '(none listed)'}"
        f"{feedback_note}"
    )
    try:
        return llm.chat_json(system=system, user=user_msg, temperature=0.1)
    except Exception as exc:
        # Fail soft: ship the untailored original rather than crash the whole run — but
        # MARK it (ROADMAP 3.2) so the fallback is visible, not silently passed off as
        # personalized. `_`-prefixed keys are internal signals, not part of the LLM schema.
        return {"body": lesson.body, "needs_background": None, "topic_fit": "",
                "_draft_failed": True, "_error": type(exc).__name__}


def _repair_draft(user: UserProfile, lesson: Lesson, curriculum: Curriculum,
                  lesson_feedback: list[str], covered: list[str],
                  previous_lesson: PersonalizedLesson) -> dict:
    """Repair a failed personalized lesson without regenerating from scratch.

    The retry loop was previously whack-a-mole: a fresh generation removed one
    unsupported claim and introduced another. This prompt constrains retries to
    deleting or minimally rewriting only the gate-flagged unsupported content.
    """
    indexed_source = source_index.render_indexed_source(
        lesson.body[:config.PERSONALIZE_LESSON_CHARS]
    )
    system = (
        "You are repairing a previous personalized lesson that failed a faithfulness gate. "
        "Your job is NOT to write a new lesson. Preserve the previous draft as much as possible.\n\n"
        "Rules:\n"
        "- Fix ONLY the listed gate issues.\n"
        "- Prefer deleting unsupported claims over replacing them.\n"
        "- If replacement is necessary, use only facts directly supported by the indexed source sentences.\n"
        "- Do NOT add new examples, formulas, methods, applications, tools, conditions, or tradeoffs.\n"
        "- Do NOT move facts from other lessons into this lesson.\n"
        "- Keep the learner-specific tone/level where possible, but factual faithfulness wins.\n"
        "- For each substantive claim that remains, include `supporting_sentences` with sentence IDs "
        "that directly support that claim. If no sentence IDs directly support it, remove the claim.\n"
        'Return ONLY JSON: {"body": "<repaired lesson markdown>", '
        '"supporting_sentences": [{"claim": "<claim in repaired lesson>", '
        '"sentence_ids": ["1", "2"]}], '
        '"needs_background": null, '
        '"topic_fit": "<same topic_fit as before, or empty string>"}'
    )
    user_msg = (
        f"LEARNER PROFILE:\n{user.raw[:config.PERSONALIZE_PROFILE_CHARS]}\n\n"
        f"ORIGINAL LESSON (order {lesson.order}): {lesson.title}\n"
        f"{indexed_source}\n\n"
        f"PREVIOUS PERSONALIZED DRAFT:\n{previous_lesson.body}\n\n"
        f"PREVIOUS SUPPORTING SENTENCES:\n{previous_lesson.supporting_sentences}\n\n"
        "GATE ISSUES TO FIX:\n- " + "\n- ".join(lesson_feedback)
    )
    try:
        return llm.chat_json(system=system, user=user_msg, temperature=0.0)
    except Exception as exc:
        return {
            "body": previous_lesson.body,
            "supporting_sentences": previous_lesson.supporting_sentences,
            "needs_background": None,
            "topic_fit": previous_lesson.topic_fit,
            "_draft_failed": True,
            "_error": type(exc).__name__,
        }


def _weave_background(body: str, gap_topic: str, background_text: str, url: str) -> str:
    system = (
        "You insert a short, cited background note into an existing lesson so a learner "
        "missing prerequisite knowledge can follow along. Rules:\n"
        "- Use ONLY the provided background text for the new note; add no outside facts.\n"
        "- Leave the rest of the lesson body UNCHANGED — append or insert the note, don't "
        "rewrite anything else.\n"
        "- Clearly mark the note (e.g. a short 'Background' aside) and cite the source URL "
        "inline.\n"
        "Return the full lesson body in markdown, nothing else — no preamble."
    )
    user_msg = (
        f"LESSON BODY:\n{body}\n\n"
        f"MISSING BACKGROUND TOPIC: {gap_topic}\n"
        f"BACKGROUND TEXT (source: {url}):\n{background_text}"
    )
    try:
        woven = llm.chat(system=system, user=user_msg, temperature=0.2)
        return woven.strip() or body
    except Exception:
        return body
