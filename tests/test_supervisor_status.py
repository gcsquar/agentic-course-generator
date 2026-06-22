"""Terminal run status + early-stop (ROADMAP 2.1 / 2.2), all on mocks (no LLM).

These pin the behavior change that makes the gates actually matter: a stage whose
gate stays red no longer ships silently — the run ends `degraded` (or `halted` in
strict mode), the verdict is persisted, and retries stop once they stop converging.
"""
import json

import config
import gates
from contracts import GateResult
from storage import Run
from supervisor import Supervisor


def _sup(tmp_path, **kw) -> Supervisor:
    run = Run(run_id="test")
    run.dir = tmp_path            # keep output under pytest's tmp dir
    return Supervisor(run, **kw)


def _status_file(tmp_path) -> dict:
    return json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------- passed
def test_clean_run_is_passed(tmp_path):
    sup = _sup(tmp_path, mock=True)
    result = sup.build("http://x")
    assert sup.status == "passed"
    assert result
    data = _status_file(tmp_path)
    assert data["status"] == "passed" and data["stage_issues"] == {}


# ----------------------------------------------------------------- degraded
def test_persistent_gate_failure_marks_degraded_but_ships(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "gate_personalize",
                        lambda *a, **k: GateResult(passed=False, issues=["Mike lesson 1: bad"]))
    sup = _sup(tmp_path, mock=True)
    result = sup.build("http://x")              # default: not strict
    assert sup.status == "degraded"
    assert result, "degraded run must still ship its (flawed) output"
    data = _status_file(tmp_path)
    assert data["status"] == "degraded"
    assert "03_personalized" in data["stage_issues"]


def test_degraded_run_writes_banner_in_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "gate_personalize",
                        lambda *a, **k: GateResult(passed=False, issues=["Mike lesson 1: bad"]))
    sup = _sup(tmp_path, mock=True)
    sup.build("http://x")
    md = (tmp_path / "03_personalized.md").read_text(encoding="utf-8")
    assert "DEGRADED" in md and "Mike lesson 1: bad" in md


# ----------------------------------------------------------------- halted
def test_agent1_rejection_halts(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "gate_ingest",
                        lambda *a, **k: GateResult(passed=False, issues=["rejected source"]))
    sup = _sup(tmp_path, mock=True)
    result = sup.build("http://x")
    assert sup.status == "halted"
    assert result == []
    assert "01_ingest" in _status_file(tmp_path)["stage_issues"]


def test_strict_halts_on_segment_gate_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "gate_segment",
                        lambda *a, **k: GateResult(passed=False, issues=["bad segmentation"]))
    sup = _sup(tmp_path, mock=True)
    result = sup.build("http://x", strict=True)
    assert sup.status == "halted"
    assert result == []                 # Agent 3 never ran
    assert "02_curriculum" in _status_file(tmp_path)["stage_issues"]


def test_non_strict_continues_past_segment_failure(tmp_path, monkeypatch):
    # Same failing segment gate, but without --strict the run continues and degrades.
    monkeypatch.setattr(gates, "gate_segment",
                        lambda *a, **k: GateResult(passed=False, issues=["bad segmentation"]))
    sup = _sup(tmp_path, mock=True)
    result = sup.build("http://x", strict=False)
    assert sup.status == "degraded"
    assert result, "non-strict run should still produce personalized lessons"


# ----------------------------------------------------------- early-stop (2.2)
def test_stage_stops_early_when_not_converging(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ESCALATION_MODEL", "")   # isolate early-stop from escalation
    sup = _sup(tmp_path, mock=True)
    calls = {"n": 0}

    def produce(_feedback):
        calls["n"] += 1
        return object()

    # Always fails with the SAME two issues -> no convergence.
    def gate(_artifact):
        return GateResult(passed=False, issues=["issue A", "issue B"])

    artifact, result = sup._stage("x", produce=produce, gate=gate,
                                  serialize=lambda a: {})
    # MAX_RETRIES=2 would allow 3 attempts; early-stop bails after the 2nd (no drop).
    assert calls["n"] == 2
    assert not result.passed


def test_stage_keeps_retrying_while_improving(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ESCALATION_MODEL", "")
    sup = _sup(tmp_path, mock=True)
    calls = {"n": 0}

    def produce(_feedback):
        calls["n"] += 1
        return object()

    # Issue count shrinks each attempt: 3 -> 2 -> 1 (still failing) -> uses full budget.
    def gate(_artifact):
        n = calls["n"]
        return GateResult(passed=False, issues=["x"] * max(1, 4 - n))

    artifact, result = sup._stage("x", produce=produce, gate=gate,
                                  serialize=lambda a: {})
    assert calls["n"] == 3              # 1 initial + MAX_RETRIES, because each retry improved


# ----------------------------------------------------------- escalation (2.5)
def test_escalation_final_attempt_uses_stronger_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ESCALATION_MODEL", "STRONG-MODEL")
    sup = _sup(tmp_path, mock=True)
    seen: list[str] = []

    def produce(_feedback):
        seen.append(config.MODEL)     # capture the generation model active at produce time
        return object()

    def gate(_artifact):
        return GateResult(passed=False, issues=["a", "b"])   # never passes, no improvement

    artifact, result = sup._stage("x", produce=produce, gate=gate, serialize=lambda a: {})
    # base attempts plateau (early-stop after 2), then ONE escalated attempt on the strong model.
    assert seen[-1] == "STRONG-MODEL"
    assert "STRONG-MODEL" not in seen[:-1]
    assert config.MODEL != "STRONG-MODEL"   # override restored afterwards


def test_no_escalation_when_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ESCALATION_MODEL", "")
    sup = _sup(tmp_path, mock=True)
    seen: list[str] = []

    def produce(_feedback):
        seen.append(config.MODEL)
        return object()

    def gate(_artifact):
        return GateResult(passed=False, issues=["a", "b"])

    sup._stage("x", produce=produce, gate=gate, serialize=lambda a: {})
    assert len(seen) == 2                       # early-stop, no extra escalated attempt
    assert all(m == config.MODEL for m in seen)


def test_no_escalation_when_stage_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ESCALATION_MODEL", "STRONG-MODEL")
    sup = _sup(tmp_path, mock=True)
    seen: list[str] = []

    def produce(_feedback):
        seen.append(config.MODEL)
        return object()

    def gate(_artifact):
        return GateResult(passed=True)          # passes on the first attempt

    sup._stage("x", produce=produce, gate=gate, serialize=lambda a: {})
    assert seen == [config.MODEL]               # no escalation needed
