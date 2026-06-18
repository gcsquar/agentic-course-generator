"""Unit tests for the REAL Agent 3 implementation.

These do not hit OpenAI or the network: `llm.chat_json` / `llm.chat` and
`research.research` are monkeypatched with deterministic fakes. This lets us
verify Agent 3's own logic (coverage, feedback filtering, the research-gap
loop) in isolation, without depending on Agent 1/Agent 2's real code or an
API key — exactly the boundary `contracts.py` is meant to give us.
"""
from contracts import Curriculum, Lesson, PersonalizedLesson, UserProfile
from agents import agent3_personalize as a3


def _curriculum() -> Curriculum:
    return Curriculum(
        source_url="http://x",
        key_concepts=["self-attention", "positional encoding"],
        lessons=[
            Lesson(order=1, title="Self-Attention",
                   body="Attention lets each token weigh every other token.",
                   key_concepts=["self-attention"]),
            Lesson(order=2, title="Positional Encoding",
                   body="Order is injected because attention is permutation-invariant.",
                   key_concepts=["positional encoding"]),
        ],
    )


def _users() -> list[UserProfile]:
    expert = UserProfile(name="Mike", level="expert",
                          raw="## Mike\n- level: expert\n- focus: tradeoffs")
    beginner = UserProfile(name="Dana", level="beginner",
                            raw="## Dana\n- level: beginner\n- focus: intuition")
    return [expert, beginner]


def test_full_coverage_no_research(monkeypatch):
    """Every (user, lesson) pair gets a non-empty, tailored body."""
    monkeypatch.setattr(a3.llm, "chat_json",
                         lambda system, user, temperature=None:
                         {"body": "tailored: " + user[:20], "needs_background": None})

    out = a3.personalize(_curriculum(), _users(), mock=False, use_llm=True, do_research=True)

    assert len(out) == 4
    assert all(isinstance(p, PersonalizedLesson) for p in out)
    pairs = {(p.user, p.order) for p in out}
    assert pairs == {("Mike", 1), ("Mike", 2), ("Dana", 1), ("Dana", 2)}
    assert all(p.body.strip() for p in out)
    assert all(p.citations == [] for p in out)  # no gap flagged -> no research


def test_research_loop_only_for_beginners(monkeypatch):
    """A flagged gap triggers research+citation for a beginner, not for an expert."""
    def fake_chat_json(system, user, temperature=None):
        if "Dana" in user:  # beginner profile text is in the prompt
            return {"body": "draft for Dana", "needs_background": "self-attention basics"}
        return {"body": "draft for Mike", "needs_background": None}

    monkeypatch.setattr(a3.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(a3.research, "research",
                         lambda query: ("Self-attention lets a model relate tokens.",
                                       "https://en.wikipedia.org/wiki/Attention"))
    monkeypatch.setattr(a3.llm, "chat",
                         lambda system, user, temperature=None: "draft for Dana\n\n> Background: ... [wikipedia]")

    out = a3.personalize(_curriculum(), _users(), mock=False, use_llm=True, do_research=True)

    by_user = {(p.user, p.order): p for p in out}
    assert by_user[("Mike", 1)].citations == []
    assert by_user[("Dana", 1)].citations == ["https://en.wikipedia.org/wiki/Attention"]
    assert "Background" in by_user[("Dana", 1)].body


def test_no_research_when_disabled(monkeypatch):
    """do_research=False must never call the research module, even if a gap is flagged."""
    monkeypatch.setattr(a3.llm, "chat_json",
                         lambda system, user, temperature=None:
                         {"body": "draft", "needs_background": "something"})
    monkeypatch.setattr(a3.research, "research",
                         lambda query: (_ for _ in ()).throw(AssertionError("should not be called")))

    out = a3.personalize(_curriculum(), _users(), mock=False, use_llm=True, do_research=False)
    assert all(p.citations == [] for p in out)


def test_feedback_is_filtered_per_user_and_lesson(monkeypatch):
    """Only feedback lines naming THIS user and THIS lesson reach the prompt."""
    seen_prompts: list[str] = []

    def fake_chat_json(system, user, temperature=None):
        seen_prompts.append(user)
        return {"body": "fixed", "needs_background": None}

    monkeypatch.setattr(a3.llm, "chat_json", fake_chat_json)

    feedback = [
        "Mike lesson 1: too basic for an expert",
        "Dana: lesson 2 is empty",
        "Mike lesson 10: completely unrelated note about a different lesson entirely",
    ]
    a3.personalize(_curriculum(), _users(), mock=False, use_llm=True,
                   feedback=feedback, do_research=False)

    mike_lesson1_prompt = next(p for p in seen_prompts
                               if "Mike" in p and "order 1)" in p)
    assert "too basic for an expert" in mike_lesson1_prompt
    assert "completely unrelated" not in mike_lesson1_prompt  # lesson 10 must not leak into lesson 1

    dana_lesson2_prompt = next(p for p in seen_prompts
                               if "Dana" in p and "order 2)" in p)
    assert "lesson 2 is empty" in dana_lesson2_prompt


def test_llm_failure_degrades_to_original_body(monkeypatch):
    """If the LLM call raises, ship the untailored original instead of crashing."""
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(a3.llm, "chat_json", boom)
    out = a3.personalize(_curriculum(), _users()[:1], mock=False, use_llm=True, do_research=False)
    assert out[0].body == "Attention lets each token weigh every other token."


def test_mock_mode_unchanged():
    """mock=True path (used by the rest of the team before Agent 3 is real) still works."""
    out = a3.personalize(_curriculum(), _users(), mock=True)
    assert len(out) == 4
    assert all(p.body.startswith("[for ") for p in out)
