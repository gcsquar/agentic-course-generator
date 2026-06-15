# Agentic Course Generator

Turns a **single URL** into **personalized educational lessons** using three cooperating
agents, coordinated by a **Supervisor** with quality gates, a research loop, and an
independent auditor. AI School Assignment 1 — *Mastering Agentic Orchestration*.

```
URL ─▶ [Agent 1: Ingest] ─g1─▶ [Agent 2: Segment] ─g2─▶ [Agent 3: Personalize] ─g3─▶ lessons
        fetch + clean            clean text →             per-user tailoring          │
        accept/reject            micro-lessons            + research loop (cited)      ▼
                                                                              [Independent Auditor]
   Supervisor owns state · routing · retries-with-feedback · halt/escalate
```

## How the team works in parallel

`app/contracts.py` is the **integration boundary**. Each agent is one file with a fixed
signature; everything talks through the dataclasses in `contracts.py`. Every agent has a
`mock=True` mode, so **the whole pipeline runs end-to-end even before the agents are
written** — run with `--mock`. Build your agent against mocked upstream data, and
integration "just works" when you meet the contract.

### Structure

| Area | Files |
|------|-------|
| Orchestration / supervision / audit / gates | `main.py` `orchestrator.py` `supervisor.py` `auditor.py` `gates.py` |
| Shared infra | `llm.py` `storage.py` `config.py` `research.py` `user_profiles.py` |
| Shared contract (change by consensus) | `contracts.py` `users.md` |
| Agent 1 — Ingestion | `agents/agent1_ingest.py` |
| Agent 2 — Segmentation | `agents/agent2_segment.py` |
| Agent 3 — Personalization | `agents/agent3_personalize.py` |

> The agent files are **stubs** today (mock data + a `NotImplementedError` on the real
> path). Each agent file is filled in against the contract in `contracts.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY (needed only for real runs)
```

## Run

```bash
cd app
python main.py "<URL>"               # Supervisor (default) + audit  — needs real agents + key
python main.py "<URL>" --sequential  # plain sequential baseline (for comparison)
python main.py "<URL>" --mock         # offline, canned data — proves wiring, no key needed
```
Outputs land in `app/runs/<timestamp>/` (`01_ingest` → `02_curriculum` →
`03_personalized` → `99_audit`, plus a per-attempt `__try`/`__gate` audit trail).

## Tests / CI

```bash
pytest -q          # mock integration smoke tests (no key)
```
`.github/workflows/ci.yml` runs these + a mock end-to-end on every push, so a
change in one module can't silently break the contract.
