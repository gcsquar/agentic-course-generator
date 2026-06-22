"""Per-role model routing (ROADMAP 2.3), all offline.

Two things are pinned here:
  1. a model slug routes to the right provider (so generation, judging, and auditing
     can run on different models — even different providers — in one run);
  2. each role actually uses its configured model: gates -> JUDGE_MODEL, auditor ->
     AUDITOR_MODEL, so "smart judge, smarter auditor" is wired, not just documented.
"""
import config
import gates
import llm
import auditor
from contracts import IngestResult, Curriculum, Lesson, PersonalizedLesson, UserProfile
from storage import Run


# ----------------------------------------------------------- provider inference
def test_provider_inference():
    assert llm._provider_for_model("gpt-4o-mini") == "openai"
    assert llm._provider_for_model("gpt-5.4") == "openai"
    assert llm._provider_for_model("meta-llama/llama-3.3-70b-instruct") == "openrouter"
    assert llm._provider_for_model("openai/gpt-oss-120b:free") == "openrouter"
    assert llm._provider_for_model("anything:free") == "openrouter"


# ------------------------------------------------- each role uses its own model
def _capture_models(monkeypatch) -> list[str]:
    seen: list[str] = []

    def fake_chat_json(system, user, temperature=None, model=None):
        seen.append(model)
        # return something each caller accepts as "clean"
        return {"passed": True, "issues": [], "score": 90, "verdict": "pass", "findings": []}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    return seen


def test_ingest_gate_uses_judge_model(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "JUDGE-SENTINEL")
    seen = _capture_models(monkeypatch)
    result = IngestResult(url="http://x", accepted=True, reason="ok",
                          title="T", clean_text="x" * 1200, meta={})
    gates.gate_ingest(result, use_llm=True)
    assert seen and all(m == "JUDGE-SENTINEL" for m in seen)


def test_personalize_gate_uses_judge_model(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "JUDGE-SENTINEL")
    seen = _capture_models(monkeypatch)
    cur = Curriculum(source_url="http://x",
                     lessons=[Lesson(order=1, title="L1", body="b", start_idx=0, end_idx=0)])
    users = [UserProfile(name="Mike", raw="## Mike")]
    personalized = [PersonalizedLesson(user="Mike", order=1, title="L1", body="tailored")]
    gates.gate_personalize(personalized, cur, users, use_llm=True)
    assert seen and all(m == "JUDGE-SENTINEL" for m in seen)


def test_auditor_uses_auditor_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AUDITOR_MODEL", "AUDITOR-SENTINEL")
    seen = _capture_models(monkeypatch)
    run = Run(run_id="t")
    run.dir = tmp_path                 # empty dir -> auditor loads {}/[] defaults
    auditor.audit_run(run)
    assert seen == ["AUDITOR-SENTINEL"]
