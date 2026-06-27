"""Architecture v2 — the Supervisor (Phase B).

The sequential orchestrator (orchestrator.py) just ran each stage once and let
gate failures fall through. The Supervisor instead OWNS the run:

  - state: it keeps each stage's final artifact and persists every attempt (the
    persisted run folder is the real shared state; `self.state` is a convenience
    handle for debugging/extension, not read by downstream stages today)
  - control: it runs the fixed stage order (A1 -> A2 -> A3) and decides whether to
    advance or halt after each gate
  - retries: when a gate fails it RE-RUNS that stage (bounded by MAX_RETRIES),
    threading the gate's issues back into the agent as feedback. It stops early
    once a retry stops reducing the issue count (ROADMAP 2.2) — more attempts that
    aren't converging only burn budget.
  - terminal status (ROADMAP 2.1): every run ends `passed` / `degraded` / `halted`.
    Previously only Agent 1 could halt; stages 2 & 3 shipped their last attempt
    silently even when its gate never passed, so the gates didn't actually affect
    the output. Now a stage whose gate stays red marks the run `degraded` (the
    artifact still ships, but the failing issues are recorded and surfaced) — or,
    with `strict=True`, halts the run on the first red gate. The verdict lands in
    `run_status.json`, a banner on the readable output, and the process exit code.
"""
from __future__ import annotations

import inspect
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
        self.status: str = "passed"       # passed | degraded | halted (set during build)
        self.stage_issues: dict[str, list[str]] = {}   # stage -> remaining issues (failed stages only)

    # -- public entry ----------------------------------------------------
    def build(self, url: str, *, only_user: str | None = None, strict: bool = False) -> list:
        print(f"[supervisor] run {self.run.run_id}  mock={self.mock}  strict={strict}")
        self.status = "passed"
        self.stage_issues = {}

        # Agent 1 gets 0 retries: re-fetching the same URL can't fix a bad source.
        # A bad source is always a hard halt (strict-independent) — there is nothing
        # downstream can do with content we rejected.
        ingest, g1 = self._stage(
            "01_ingest",
            produce=lambda fb: agent1_ingest.ingest(url, mock=self.mock),
            gate=lambda a: gates.gate_ingest(a, use_llm=self.use_llm),
            retries=0,
        )
        if not g1.passed:
            self._record("01_ingest", g1, strict=True)   # always halts
            print(f"[supervisor] HALT after Agent 1: {g1.issues}")
            self._finalize(url)
            return []
        self.run.save_md("01_ingest", f"# {ingest.title}\n\n{ingest.clean_text}")

        # Agents 2 & 3 retry WITH the gate's issues threaded back in as feedback.
        curriculum, g2 = self._stage(
            "02_curriculum",
            produce=lambda fb: agent2_segment.segment(
                ingest, mock=self.mock, use_llm=self.use_llm, feedback=fb),
            gate=lambda a: gates.gate_segment(a, ingest, use_llm=self.use_llm),
        )
        if not self._record("02_curriculum", g2, strict=strict):
            print(f"[supervisor] HALT (strict) after Agent 2: {g2.issues}")
            self._finalize(url)
            return []

        users = user_profiles.parse_users(config.USERS_FILE)
        users = user_profiles.filter_users(users, only_user)
        personalized, g3 = self._stage(
            "03_personalized",
            produce=lambda fb, prev=None: agent3_personalize.personalize(
                curriculum, users, mock=self.mock, use_llm=self.use_llm,
                feedback=fb, previous=prev),
            gate=lambda a: gates.gate_personalize(a, curriculum, users, use_llm=self.use_llm),
            serialize=lambda a: [p.to_dict() for p in a],
            retries=config.MAX_RETRIES_PERSONALIZE,   # own (higher) cap — see config note
        )
        self._record("03_personalized", g3, strict=strict)   # last stage: strict halt has nothing left to skip

        _write_readable(self.run, personalized, status=self.status, issues=self.stage_issues)
        self._finalize(url)
        print(f"[supervisor] done ({self.status}): {len(personalized)} personalized lessons")
        return personalized

    # -- status bookkeeping ----------------------------------------------
    def _record(self, name: str, gate: GateResult, *, strict: bool) -> bool:
        """Fold a stage's gate verdict into the run status. Returns whether the
        pipeline should CONTINUE: a passed gate continues; a failed gate halts the
        run in strict mode (return False) or marks it degraded and continues
        otherwise. Once degraded, a later passing stage does not undo it."""
        if gate.passed:
            return True
        self.stage_issues[name] = gate.issues
        if strict:
            self.status = "halted"
            return False
        self.status = "degraded"
        return True

    def _finalize(self, url: str) -> None:
        """Persist the machine-readable run verdict next to the artifacts."""
        self.run.save_json("run_status", {
            "run_id": self.run.run_id,
            "url": url,
            "status": self.status,
            "stage_issues": self.stage_issues,
        })

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
        prev_issue_count: int | None = None
        attempt = 0

        # Phase 1 — the base model, with informed retries and early-stop on non-convergence.
        for attempt in range(1, retries + 2):   # 1 initial + `retries`
            artifact, result = self._attempt(name, attempt, produce, gate, serialize, feedback,
                                             previous=artifact)
            if result.passed:
                break
            feedback = result.issues   # informed feedback for the next produce()

            # Early-stop (ROADMAP 2.2): if a retry did not REDUCE the issue count, the
            # feedback loop isn't converging on this model — more base attempts just burn
            # budget (and risk stopping on a lucky lenient sample, INSIGHTS #2). The first
            # failure always earns at least one informed retry.
            cur = len(result.issues)
            if prev_issue_count is not None and cur >= prev_issue_count:
                print(f"[supervisor] {name}: no improvement "
                      f"({prev_issue_count} -> {cur} issues) — stopping base retries")
                break
            prev_issue_count = cur

        # Phase 2 — escalation (ROADMAP 2.5): if the base model couldn't satisfy the gate,
        # make ONE last-ditch attempt on a stronger model. "Stage can't pass on this model"
        # is exactly what escalation is for — so it fires even when early-stop gave up.
        if not result.passed and config.ESCALATION_MODEL and retries > 0:
            attempt += 1
            print(f"[supervisor] {name}: escalating final attempt to {config.ESCALATION_MODEL}")
            artifact, result = self._attempt(name, attempt, produce, gate, serialize,
                                             feedback, model=config.ESCALATION_MODEL,
                                             previous=artifact)

        # canonical (final) artifact for downstream stages / readers
        self.state[name] = artifact
        self.run.save_json(name, serialize(artifact) if serialize else artifact.to_dict())
        return artifact, result

    def _attempt(self, name: str, attempt: int, produce: Callable[[list[str]], Any],
                  gate: Callable[[Any], GateResult], serialize: Callable[[Any], Any] | None,
                  feedback: list[str], *, model: str | None = None,
                  previous: Any = None) -> tuple[Any, GateResult]:
        """One produce -> persist -> gate -> persist cycle. On an escalation attempt
        (`model` set) the GENERATION model is overridden for the duration of produce()
        only — gates keep their own JUDGE_MODEL, and the override is restored even on
        error. Safe because the pipeline is sequential and produce() (including Agent 3's
        internal thread pool) completes before we restore."""
        if model:
            saved = config.MODEL
            config.MODEL = model
            try:
                artifact = self._produce(produce, feedback, previous)
            finally:
                config.MODEL = saved
        else:
            artifact = self._produce(produce, feedback, previous)

        payload = serialize(artifact) if serialize else artifact.to_dict()
        self.run.save_json(f"{name}__try{attempt}", payload)
        result = gate(artifact)
        self.run.save_json(f"{name}__gate{attempt}", result.to_dict())
        tag = "PASS" if result.passed else f"FAIL {result.issues}"
        suffix = f" [escalated:{model}]" if model else ""
        print(f"[supervisor] {name} try {attempt}{suffix}: {tag}")
        return artifact, result

    @staticmethod
    def _produce(produce: Callable, feedback: list[str], previous: Any) -> Any:
        """Call a stage producer, passing previous artifact only if it accepts it."""
        try:
            n_params = len(inspect.signature(produce).parameters)
        except (TypeError, ValueError):
            n_params = 1
        if n_params >= 2:
            return produce(feedback, previous)
        return produce(feedback)
