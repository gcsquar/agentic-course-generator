"""Architecture v2 — the Supervisor (Phase B).

The sequential orchestrator (orchestrator.py) just ran each stage once and let
gate failures fall through. The Supervisor instead OWNS the run:

  - state: it keeps each stage's final artifact and persists every attempt (the
    persisted run folder is the real shared state; `self.state` is a convenience
    handle for debugging/extension, not read by downstream stages today)
  - control: it runs the fixed stage order (A1 -> A2 -> A3) and decides whether to
    advance or halt after each gate
  - retries: when a gate fails it RE-RUNS that stage (bounded by MAX_RETRIES),
    threading the gate's issues back into the agent as feedback

KNOWN GAP (ROADMAP 2.1): only Agent 1 halts on a failed gate; stages 2 & 3 ship
their last attempt even if its gate never passed. Making that terminal state
explicit (passed/degraded/halted) is tracked in ROADMAP.md.
"""
from __future__ import annotations

from typing import Any, Callable

import config
import gates
import user_profiles
from storage import Run
from contracts import GateResult
from orchestrator import _write_readable
from agents import agent1_ingest, agent2_segment, agent3_personalize


class Supervisor:
    def __init__(self, run: Run, *, mock: bool = False):
        self.run = run
        self.mock = mock
        self.use_llm = not mock
        self.state: dict[str, Any] = {}   # stage name -> final artifact (debug handle; not read downstream)

    # -- public entry ----------------------------------------------------
    def build(self, url: str, *, only_user: str | None = None) -> list:
        print(f"[supervisor] run {self.run.run_id}  mock={self.mock}")

        # Agent 1 gets 0 retries: re-fetching the same URL can't fix a bad source.
        ingest, g1 = self._stage(
            "01_ingest",
            produce=lambda fb: agent1_ingest.ingest(url, mock=self.mock),
            gate=lambda a: gates.gate_ingest(a, use_llm=self.use_llm),
            retries=0,
        )
        if not g1.passed:
            print(f"[supervisor] HALT after Agent 1: {g1.issues}")
            return []
        self.run.save_md("01_ingest", f"# {ingest.title}\n\n{ingest.clean_text}")

        # Agents 2 & 3 retry WITH the gate's issues threaded back in as feedback.
        curriculum, _g2 = self._stage(
            "02_curriculum",
            produce=lambda fb: agent2_segment.segment(
                ingest, mock=self.mock, use_llm=self.use_llm, feedback=fb),
            gate=lambda a: gates.gate_segment(a, ingest, use_llm=self.use_llm),
        )

        users = user_profiles.parse_users(config.USERS_FILE)
        users = user_profiles.filter_users(users, only_user)
        personalized, _g3 = self._stage(
            "03_personalized",
            produce=lambda fb: agent3_personalize.personalize(
                curriculum, users, mock=self.mock, use_llm=self.use_llm, feedback=fb),
            gate=lambda a: gates.gate_personalize(a, curriculum, users, use_llm=self.use_llm),
            serialize=lambda a: [p.to_dict() for p in a],
        )
        _write_readable(self.run, personalized)
        print(f"[supervisor] done: {len(personalized)} personalized lessons")
        return personalized

    # -- the retry loop --------------------------------------------------
    def _stage(self, name: str, *, produce: Callable[[list[str]], Any],
               gate: Callable[[Any], GateResult],
               serialize: Callable[[Any], Any] | None = None,
               retries: int | None = None) -> tuple[Any, GateResult]:
        """Run a stage; on gate failure, re-run up to `retries` times (default
        MAX_RETRIES), feeding the gate's issues back into produce() each time.

        `produce(feedback)` builds the artifact (feedback = previous gate issues).
        `gate(artifact)` judges it. Every attempt is persisted.
        """
        if retries is None:
            retries = config.MAX_RETRIES
        feedback: list[str] = []
        artifact: Any = None
        result = GateResult(passed=False, issues=["stage did not run"])

        for attempt in range(1, retries + 2):   # 1 initial + `retries`
            artifact = produce(feedback)
            payload = serialize(artifact) if serialize else artifact.to_dict()
            self.run.save_json(f"{name}__try{attempt}", payload)

            result = gate(artifact)
            self.run.save_json(f"{name}__gate{attempt}", result.to_dict())
            status = "PASS" if result.passed else f"FAIL {result.issues}"
            print(f"[supervisor] {name} try {attempt}: {status}")

            if result.passed:
                break
            feedback = result.issues   # fed to the next produce() in TB.2

        # canonical (final) artifact for downstream stages / readers
        self.state[name] = artifact
        self.run.save_json(name, serialize(artifact) if serialize else artifact.to_dict())
        return artifact, result
