"""CLI entry point.

    python main.py <URL>               # Supervisor (Architecture v2) — default
    python main.py <URL> --sequential  # plain Sequential baseline (Architecture v1)
    python main.py <URL> --mock        # no network/LLM — proves the wiring

Output lands in runs/<timestamp>/.
"""
from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    run = Run(run_id=datetime.now().strftime("%Y%m%d_%H%M%S"))
    if args.sequential:
        result = run_sequential(args.url, run, mock=args.mock)
    else:
        result = Supervisor(run, mock=args.mock).build(args.url)

    # Independent auditor: a fresh review of the finished run (real mode only).
    if result and not args.mock and not args.no_audit:
        from auditor import audit_run
        audit_run(run)

    print(f"\nDone. Outputs in: {run.dir}")


if __name__ == "__main__":
    main()
