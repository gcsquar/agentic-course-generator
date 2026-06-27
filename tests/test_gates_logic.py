"""Deterministic gate tests (no LLM): the cheap structural checks that run
before any judge. These guard the boring-but-critical properties INSIGHTS #8
showed a clever gate can miss (lesson count, dropped tail, lumping).
"""
import config
import gates
import llm
from contracts import Curriculum, Lesson, IngestResult, UserProfile, PersonalizedLesson


def _lesson(order, start, end, title=None):
    return Lesson(order=order, title=title or f"L{order}", body="x",
                  start_idx=start, end_idx=end)


def _curriculum(lessons, n_src=0):
    return Curriculum(source_url="http://x", lessons=lessons, n_source_paragraphs=n_src)


_SRC = IngestResult(url="http://x", accepted=True, reason="ok", clean_text="x" * 1000)


# ----------------------------------------------------------------- _check_coverage
def test_coverage_clean_tiling_has_no_issues():
    lessons = [_lesson(1, 0, 3), _lesson(2, 4, 7), _lesson(3, 8, 9)]
    assert gates._check_coverage(lessons, 10) == []


def test_coverage_detects_gap():
    lessons = [_lesson(1, 0, 3), _lesson(2, 6, 9)]   # 4-5 missing
    issues = gates._check_coverage(lessons, 10)
    assert any("gap" in i for i in issues)


def test_coverage_detects_overlap():
    lessons = [_lesson(1, 0, 5), _lesson(2, 4, 9)]   # 4-5 in both
    issues = gates._check_coverage(lessons, 10)
    assert any("overlap" in i for i in issues)


def test_coverage_detects_inverted_range():
    lessons = [_lesson(1, 0, 4), _lesson(2, 8, 6)]   # end < start
    issues = gates._check_coverage(lessons, 10)
    assert any("inverted" in i for i in issues)


def test_coverage_detects_dropped_tail():
    lessons = [_lesson(1, 0, 4), _lesson(2, 5, 9)]   # source actually has 20 paragraphs
    issues = gates._check_coverage(lessons, 20)
    assert any("dropped tail" in i for i in issues)


def test_coverage_skipped_when_indices_unset():
    # Mock/hand-built curricula leave start_idx == -1; nothing deterministic to check.
    lessons = [Lesson(order=1, title="A", body="x"), Lesson(order=2, title="B", body="y")]
    assert gates._check_coverage(lessons, 0) == []


# ----------------------------------------------------------------- gate_segment
def test_segment_single_giant_lesson_fails_count():
    # One lesson tiles [0,9] perfectly but isn't segmentation (INSIGHTS #8).
    cur = _curriculum([_lesson(1, 0, 9)], n_src=10)
    res = gates.gate_segment(cur, _SRC, use_llm=False)
    assert not res.passed
    assert any("not segmented" in i or "lesson(s)" in i for i in res.issues)


def test_segment_duplicate_titles_flagged():
    lessons = [_lesson(1, 0, 3, "Same"), _lesson(2, 4, 7, "Same"), _lesson(3, 8, 9, "Other")]
    res = gates.gate_segment(_curriculum(lessons, 10), _SRC, use_llm=False)
    assert any("duplicate" in i for i in res.issues)


def test_segment_lump_smell_flagged():
    big = config.MAX_LESSON_PARAGRAPHS + 5
    lessons = [_lesson(1, 0, big - 1), _lesson(2, big, big + 2), _lesson(3, big + 3, big + 4)]
    res = gates.gate_segment(_curriculum(lessons, big + 5), _SRC, use_llm=False)
    assert any("lumps several subtopics" in i for i in res.issues)


def test_segment_clean_curriculum_passes():
    lessons = [_lesson(1, 0, 3), _lesson(2, 4, 7), _lesson(3, 8, 9)]
    res = gates.gate_segment(_curriculum(lessons, 10), _SRC, use_llm=False)
    assert res.passed, res.issues


def test_segment_no_lessons_fails():
    res = gates.gate_segment(_curriculum([], 0), _SRC, use_llm=False)
    assert not res.passed


# ----------------------------------------------------------- gate_personalize (det.)
def _two_lesson_curriculum():
    return _curriculum([_lesson(1, 0, 3), _lesson(2, 4, 9)], 10)


def test_personalize_missing_pair_flagged():
    users = [UserProfile(name="Mike", raw="## Mike")]
    cur = _two_lesson_curriculum()
    personalized = [PersonalizedLesson(user="Mike", order=1, title="L1", body="ok")]  # missing order 2
    res = gates.gate_personalize(personalized, cur, users, use_llm=False)
    assert any("missing personalized lesson 2" in i for i in res.issues)


def test_personalize_empty_body_flagged():
    users = [UserProfile(name="Mike", raw="## Mike")]
    cur = _two_lesson_curriculum()
    personalized = [
        PersonalizedLesson(user="Mike", order=1, title="L1", body="ok"),
        PersonalizedLesson(user="Mike", order=2, title="L2", body="   "),
    ]
    res = gates.gate_personalize(personalized, cur, users, use_llm=False)
    assert any("lesson 2 is empty" in i for i in res.issues)


def test_personalize_complete_passes_without_llm():
    users = [UserProfile(name="Mike", raw="## Mike")]
    cur = _two_lesson_curriculum()
    personalized = [
        PersonalizedLesson(user="Mike", order=1, title="L1", body="tailored one"),
        PersonalizedLesson(user="Mike", order=2, title="L2", body="tailored two"),
    ]
    res = gates.gate_personalize(personalized, cur, users, use_llm=False)
    assert res.passed, res.issues


def test_personalize_judge_gets_article_context_for_acronyms(monkeypatch):
    captured = {}

    def fake_chat_json(system, user, temperature=None, model=None):
        captured["system"] = system
        captured["user"] = user
        return {"issues": []}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    user = UserProfile(name="Rita", raw="## Rita\nBeginner")
    lesson = PersonalizedLesson(
        user="Rita",
        order=2,
        title="Boosting",
        body="GBDT means Gradient Boosted Decision Trees.",
    )

    gates._judge_one_personalized_lesson(
        user,
        lesson,
        "This lesson mentions GBDT only.",
        full_source_context="Gradient Boosted Decision Trees, also known as GBDT.",
    )

    assert "Use the ARTICLE-WIDE SOURCE CONTEXT only to resolve terms" in captured["system"]
    assert "ARTICLE-WIDE SOURCE CONTEXT FOR TERMS/ALIASES/ACRONYMS ONLY" in captured["user"]
    assert "Gradient Boosted Decision Trees, also known as GBDT" in captured["user"]


# --------------------------------------------------- confirmation-sampling (INSIGHTS #2)
# The "warmer gatekeeper": a clean PASS is double-checked with a second, INDEPENDENT
# sample (so the retry loop can't stop on the first lenient verdict), but a FAIL is
# trusted at once. Independence comes from a different model (CONFIRM_MODEL) if set, else
# a hotter temperature draw. These pin that asymmetric logic by stubbing the inner judge.

def _confirm_args():
    user = UserProfile(name="Mike", raw="## Mike")
    return user, [], {}, "full source text"


def test_confirmation_clean_then_flag_blocks(monkeypatch):
    """temp-0 finds nothing but the hotter temp-0.5 fallback sample does -> blocks."""
    monkeypatch.setattr(config, "CONFIRM_MODEL", "")   # no distinct model -> temp fallback
    calls = []

    def fake_judge(user, lessons, originals, full_source, *, temperature=0.0, model=None):
        calls.append((temperature, model))
        return [] if temperature == 0.0 else ["Mike lesson 1: invented ~3x metric"]

    monkeypatch.setattr(gates, "_judge_personalization", fake_judge)
    issues = gates._judge_personalization_confirmed(*_confirm_args())
    assert issues == ["Mike lesson 1: invented ~3x metric"]
    assert [t for t, _ in calls] == [0.0, 0.5]   # both samples drawn, hotter second


def test_confirmation_fail_is_trusted_at_once(monkeypatch):
    """A temp-0 FAIL must NOT trigger a second sample — fails are trusted immediately."""
    monkeypatch.setattr(config, "CONFIRM_MODEL", "")
    calls = []

    def fake_judge(user, lessons, originals, full_source, *, temperature=0.0, model=None):
        calls.append((temperature, model))
        return ["Mike lesson 1: decorative analogy"]

    monkeypatch.setattr(gates, "_judge_personalization", fake_judge)
    issues = gates._judge_personalization_confirmed(*_confirm_args())
    assert issues == ["Mike lesson 1: decorative analogy"]
    assert len(calls) == 1   # no confirmation draw needed


def test_confirmation_clean_twice_passes(monkeypatch):
    """Both samples clean -> no issues, but the confirmation sample WAS taken."""
    monkeypatch.setattr(config, "CONFIRM_MODEL", "")
    calls = []

    def fake_judge(user, lessons, originals, full_source, *, temperature=0.0, model=None):
        calls.append((temperature, model))
        return []

    monkeypatch.setattr(gates, "_judge_personalization", fake_judge)
    issues = gates._judge_personalization_confirmed(*_confirm_args())
    assert issues == []
    assert len(calls) == 2


def test_confirmation_uses_distinct_model_when_set(monkeypatch):
    """With CONFIRM_MODEL set, the confirmation draw uses THAT model at temp 0 (a real
    independent second opinion), not a hotter draw of the same judge."""
    monkeypatch.setattr(config, "JUDGE_MODEL", "JUDGE-A")
    monkeypatch.setattr(config, "CONFIRM_MODEL", "CONFIRM-B")
    calls = []

    def fake_judge(user, lessons, originals, full_source, *, temperature=0.0, model=None):
        calls.append((temperature, model))
        return []

    monkeypatch.setattr(gates, "_judge_personalization", fake_judge)
    gates._judge_personalization_confirmed(*_confirm_args())
    # first draw uses the default judge model (None -> resolved downstream); the
    # confirmation draw is the distinct model at temp 0, not temp 0.5.
    assert calls[0] == (0.0, None)
    assert calls[1] == (0.0, "CONFIRM-B")
