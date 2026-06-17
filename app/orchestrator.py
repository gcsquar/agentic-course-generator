"""Orchestration — Phase A: Sequential pipeline (the baseline).

    URL -> [A1] -(g1)-> [A2] -(g2)-> [A3] -> personalized lessons

Every stage is persisted to the run folder (audit trail + blackboard). Gates run
between stages and (for now) just report; Phase B turns the reports into retries
and grows this function into a real Supervisor.
"""
from __future__ import annotations

import config
import gates
import user_profiles
from storage import Run
from agents import agent1_ingest, agent2_segment, agent3_personalize


def run_sequential(url: str, run: Run, *, mock: bool = False) -> list:
    print(f"[orchestrator] run {run.run_id}  mock={mock}")

    # --- Agent 1 -------------------------------------------------------
    ingest = agent1_ingest.ingest(url, mock=mock)
    run.save_json("01_ingest", ingest.to_dict())
    run.save_md("01_ingest", f"# {ingest.title}\n\n{ingest.clean_text}")
    g1 = gates.gate_ingest(ingest, use_llm=not mock)
    print(f"[gate g1] passed={g1.passed} issues={g1.issues}")
    if not g1.passed:
        # Phase B: retry. For now, stop early on a hard reject.
        run.save_json("01_gate", g1.to_dict())
        return []

    # --- Agent 2 -------------------------------------------------------
    curriculum = agent2_segment.segment(ingest, mock=mock)
    run.save_json("02_curriculum", curriculum.to_dict())
    g2 = gates.gate_segment(curriculum, ingest, use_llm=not mock)
    print(f"[gate g2] passed={g2.passed} issues={g2.issues}")

    # --- Agent 3 -------------------------------------------------------
    users = user_profiles.parse_users(config.USERS_FILE)
    personalized = agent3_personalize.personalize(curriculum, users, mock=mock)
    run.save_json("03_personalized", [p.to_dict() for p in personalized])
    _write_readable(run, personalized)
    g3 = gates.gate_personalize(personalized, curriculum, users, use_llm=not mock)
    print(f"[gate g3] passed={g3.passed} issues={g3.issues}")
    print(f"[orchestrator] {len(users)} users x {len(curriculum.lessons)} lessons "
          f"= {len(personalized)} personalized lessons")
    return personalized


def _write_readable(run: Run, personalized: list) -> None:
    by_user: dict[str, list] = {}
    for p in personalized:
        by_user.setdefault(p.user, []).append(p)

    parts: list[str] = ["# Personalized Lessons\n\n"]
    for user, lessons in by_user.items():
        parts.append(f"## {user}\n\n")
        for l in sorted(lessons, key=lambda x: x.order):
            parts.append(f"### {l.order}. {l.title}\n\n")
            parts.append(l.body.strip() + "\n\n")
            if getattr(l, "topic_fit", ""):
                parts.append(f"> **Примечание:** {l.topic_fit}\n\n")
            if l.citations:
                links = " · ".join(f"[{i+1}]({url})" for i, url in enumerate(l.citations))
                parts.append(f"*Источники: {links}*\n\n")
            parts.append("---\n\n")

    run.save_md("03_personalized", "".join(parts))
