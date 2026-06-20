"""CLI entry point.

    python main.py <URL>               # Supervisor (Architecture v2) — default
    python main.py <URL> --sequential  # plain Sequential baseline (Architecture v1)
    python main.py <URL> --mock        # no network/LLM — proves the wiring
    python main.py <URL> --user Mike   # personalize for one user.md profile only

Output lands in runs/<timestamp>/.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

from storage import Run
from orchestrator import run_sequential
from supervisor import Supervisor


def main() -> None:
    parser = argparse.ArgumentParser(description="URL -> personalized lessons (3 agents)")
    parser.add_argument("url", help="target URL to turn into a course")
    parser.add_argument("--mock", action="store_true",
                        help="run with canned data — no network/LLM calls")
    parser.add_argument("--sequential", action="store_true",
                        help="use the plain sequential baseline instead of the Supervisor")
    parser.add_argument("--no-audit", action="store_true",
                        help="skip the independent end-of-run audit")
    parser.add_argument("--user", metavar="NAME", default=None,
                        help="personalize for a single user.md profile only, by name")
    args = parser.parse_args()

    run = Run(run_id=datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        if args.sequential:
            result = run_sequential(args.url, run, mock=args.mock, only_user=args.user)
        else:
            result = Supervisor(run, mock=args.mock).build(args.url, only_user=args.user)
    except ValueError as e:
        print(f"[main] {e}")
        sys.exit(1)

    # Independent auditor: a fresh review of the finished run (real mode only).
    if result and not args.mock and not args.no_audit:
        from auditor import audit_run
        audit_run(run)

    if not args.mock:
        import llm
        u = llm.usage_summary()
        if u["calls"]:
            print(f"[llm] run total: {u['calls']} calls, {u['total']} tokens "
                  f"(prompt={u['prompt']}, completion={u['completion']})")

    print(f"\nDone. Outputs in: {run.dir}")


if __name__ == "__main__":
    main()
