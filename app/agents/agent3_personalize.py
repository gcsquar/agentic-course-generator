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
This is a STUB. `mock=True` returns canned output so the pipeline runs end-to-end.
"""
from __future__ import annotations

from contracts import Curriculum, UserProfile, PersonalizedLesson


def personalize(curriculum: Curriculum, users: list[UserProfile], *,
                mock: bool = False, use_llm: bool = True,
                feedback: list[str] | None = None,
                do_research: bool = True) -> list[PersonalizedLesson]:
    if mock or not use_llm:
        out: list[PersonalizedLesson] = []
        for u in users:
            for l in curriculum.lessons:
                out.append(PersonalizedLesson(
                    user=u.name, order=l.order, title=l.title,
                    body=f"[for {u.name} · {u.level}] {l.body}",
                ))
        return out
    # TODO: real per-user tailoring (level/tone/interests); honor `feedback`;
    # if `do_research`, fill gaps via research.py with cited background.
    raise NotImplementedError(
        "Agent 3 not implemented yet. Run with --mock, or implement personalize()."
    )
