"""Quick real-LLM smoke test: Agent 2 + Agent 3, mock Agent 1, 1 user, no research.

Cost estimate: ~$0.01-0.03 (gpt-4o-mini, 1 user x 3 lessons x 2 LLM calls each).

Usage (from repo root):
    # 1. Make sure .env has OPENAI_API_KEY=sk-...
    # 2. python test_real_cheap.py
    # 3. python test_real_cheap.py --user Lena   <- pick one user by name
    # 4. python test_real_cheap.py --all-users   <- all 4 users (~$0.04-0.10)
"""
from __future__ import annotations

import argparse
import io
import sys

# Windows console defaults to cp1252 — force utf-8 so Cyrillic prints correctly.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, "app")

from agents.agent1_ingest import _MOCK as mock_ingest
from agents import agent2_segment, agent3_personalize
from user_profiles import parse_users
import config


def run(user_name: str | None = None, all_users: bool = False) -> None:
    all_profiles = parse_users(config.USERS_FILE)

    if all_users:
        users = all_profiles
    elif user_name:
        users = [u for u in all_profiles if u.name.lower() == user_name.lower()]
        if not users:
            names = [u.name for u in all_profiles]
            print(f"User '{user_name}' not found. Available: {names}")
            sys.exit(1)
    else:
        users = all_profiles[:1]  # cheapest default: first user only

    print(f"\nModel : {config.MODEL}")
    print(f"Users : {[u.name for u in users]}")
    print(f"Source: {mock_ingest.title!r}  ({len(mock_ingest.clean_text)} chars)\n")

    # --- Agent 2: real LLM segmentation ---
    print("[Agent 2] Segmenting into micro-lessons...")
    curriculum = agent2_segment.segment(mock_ingest, mock=False, use_llm=True)
    print(f"          {len(curriculum.lessons)} lessons produced")
    for l in curriculum.lessons:
        print(f"          [{l.order}] {l.title}")

    # --- Agent 3: real LLM personalization, research disabled ---
    print(f"\n[Agent 3] Personalizing for {[u.name for u in users]} (no web research)...")
    personalized = agent3_personalize.personalize(
        curriculum, users,
        mock=False, use_llm=True,
        do_research=False,   # disable web research: fewer LLM calls, no network
    )
    print(f"          {len(personalized)} personalized lessons produced\n")

    # --- Print output ---
    cur_user = None
    for p in sorted(personalized, key=lambda x: (x.user, x.order)):
        if p.user != cur_user:
            cur_user = p.user
            user_obj = next(u for u in users if u.name == p.user)
            print(f"\n{'='*60}")
            print(f"  {p.user}  |  level: {user_obj.level}  |  focus: {user_obj.focus}")
            print(f"{'='*60}")
        print(f"\n  Lesson {p.order}: {p.title}")
        print(f"  {'-'*50}")
        # Show first 400 chars so output stays readable
        body_preview = p.body[:400] + ("…" if len(p.body) > 400 else "")
        for line in body_preview.splitlines():
            print(f"  {line}")
        if p.citations:
            print(f"\n  Sources: {', '.join(p.citations)}")

    print(f"\nDone. Total personalized lessons: {len(personalized)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cheap real-LLM test: Agent 2 + Agent 3")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--user", metavar="NAME", help="test with one specific user")
    group.add_argument("--all-users", action="store_true", help="test all users in users.md")
    args = parser.parse_args()
    run(user_name=args.user, all_users=args.all_users)
