"""Integration smoke tests — run entirely on mocks (no network, no API key).

These guard the team contract: as long as each agent honors its signature and
returns the right dataclass, the orchestrator + supervisor wire up end-to-end.
CI runs these on every push, so a change in one module can't silently break others.
"""
from contracts import Curriculum, PersonalizedLesson, IngestResult
from storage import Run
from supervisor import Supervisor
from orchestrator import run_sequential
from agents import agent1_ingest, agent2_segment, agent3_personalize


def _run(tmp_path, fn):
    run = Run(run_id="test")
    # redirect output under pytest's tmp dir so we don't litter app/runs
    run.dir = tmp_path
    return fn(run)


def test_agents_honor_contracts():
    ing = agent1_ingest.ingest("http://x", mock=True)
    assert isinstance(ing, IngestResult) and ing.clean_text

    cur = agent2_segment.segment(ing, mock=True)
    assert isinstance(cur, Curriculum) and cur.lessons

    from user_profiles import parse_users
    import config
    users = parse_users(config.USERS_FILE)
    pers = agent3_personalize.personalize(cur, users, mock=True)
    assert pers and all(isinstance(p, PersonalizedLesson) for p in pers)
    assert len(pers) == len(users) * len(cur.lessons)


def test_supervisor_end_to_end(tmp_path):
    result = _run(tmp_path, lambda r: Supervisor(r, mock=True).build("http://x"))
    assert result and all(isinstance(p, PersonalizedLesson) for p in result)


def test_sequential_end_to_end(tmp_path):
    result = _run(tmp_path, lambda r: run_sequential("http://x", r, mock=True))
    assert result
