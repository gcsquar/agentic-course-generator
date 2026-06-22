"""Judge-quality eval runner — scores the LLM gates against the golden cases.

This makes "how good is this model/prompt as a judge?" measurable. It runs the
REAL gates (so it needs an API key and costs money — it is deliberately NOT part
of pytest/CI). Each case has a known-correct verdict; the runner reports a
confusion matrix and precision/recall over the judge's pass/fail decisions.

Usage (from repo root, with a venv + key):
    python evals/run_evals.py                       # all cases, JUDGE_MODEL, 1 run each
    python evals/run_evals.py --model gpt-4o-mini   # compare a specific model
    python evals/run_evals.py --repeats 5           # sample each case 5x (judge is noisy)
    python evals/run_evals.py --only syrniki        # one case by name substring
    python evals/run_evals.py --list                # list cases, run nothing

Positive class = "the case contains a real problem" (expect_pass=False), so:
  recall    = of the real problems, how many the judge CAUGHT (misses are dangerous)
  precision = of the things it flagged, how many were real (over-flagging hurts UX)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make app/ modules importable (same trick as tests/conftest.py)
APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP))

import config          # noqa: E402
import gates           # noqa: E402
from cases import CASES, Case   # noqa: E402


def _run_gate(case: Case) -> tuple[bool, list[str]]:
    d = case.build()
    if case.kind == "personalize":
        res = gates.gate_personalize(d["personalized"], d["curriculum"], d["users"], use_llm=True)
    elif case.kind == "ingest":
        res = gates.gate_ingest(d["ingest"], use_llm=True)
    else:
        raise ValueError(f"unknown case kind: {case.kind}")
    return res.passed, res.issues


def _substr_hit(issues: list[str], needles: list[str]) -> bool:
    blob = " \n ".join(issues).lower()
    return any(n.lower() in blob for n in needles)


def _label(expect_pass: bool, passed: bool) -> str:
    # positive class = has a real problem (expect_pass False)
    if not expect_pass and not passed:
        return "TP"   # caught a real problem
    if not expect_pass and passed:
        return "FN"   # MISSED a real problem
    if expect_pass and passed:
        return "TN"   # correctly clean
    return "FP"       # over-flagged a clean case


def run(cases: list[Case], repeats: int) -> int:
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    errors = 0
    print(f"\nModel under test: JUDGE_MODEL={config.JUDGE_MODEL}  ·  repeats={repeats}\n")
    print(f"{'case':<34} {'expect':<7} {'result':<14} reason/notes")
    print("-" * 88)

    for case in cases:
        correct = 0
        case_labels: list[str] = []
        reason_hits = 0
        forbid_hits = 0
        last_issues: list[str] = []
        last_error = ""
        for _ in range(repeats):
            try:
                passed, issues = _run_gate(case)
            except RuntimeError as e:   # almost always: no API key — nothing will work, stop
                print(f"\n[evals] cannot run gates: {e}")
                return 2
            except Exception as e:      # bad model / network / API error — record & keep going
                errors += 1
                last_error = f"{type(e).__name__}: {str(e)[:120]}"
                case_labels.append("ERR")
                continue
            lbl = _label(case.expect_pass, passed)
            case_labels.append(lbl)
            counts[lbl] += 1
            if case.expect_issue_substr and _substr_hit(issues, case.expect_issue_substr):
                reason_hits += 1
            if case.forbid_issue_substr and _substr_hit(issues, case.forbid_issue_substr):
                forbid_hits += 1
            ok = (passed == case.expect_pass) and not (
                case.forbid_issue_substr and _substr_hit(issues, case.forbid_issue_substr))
            if ok:
                correct += 1
            if not passed:
                last_issues = issues

        scored = repeats - case_labels.count("ERR")
        verdict = f"{correct}/{scored} ok" if scored else "ERR"
        notes = []
        if case.expect_issue_substr and scored:
            notes.append(f"reason {reason_hits}/{scored}")
        if forbid_hits:
            notes.append(f"!! flagged forbidden term {forbid_hits}/{scored}")
        if last_error:
            notes.append(f"!! {case_labels.count('ERR')}/{repeats} errored: {last_error}")
        label_summary = "/".join(sorted(set(case_labels)))
        flag = "OK " if (scored and correct == scored and not last_error) else (
            "ER " if not scored else ("!! " if correct == 0 else "~~ "))
        print(f"{flag}{case.name:<31} {'PASS' if case.expect_pass else 'FAIL':<7} "
              f"{verdict:<14} [{label_summary}] {' · '.join(notes)}")
        if correct < scored and last_issues:
            print(f"      sample issues: {last_issues[:3]}")

    # ---- scorecard ----
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    accuracy = (tp + tn) / total if total else float("nan")
    print("\n" + "=" * 50)
    print(f"Confusion (positive = real problem present):")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}   (n={total})")
    if total:
        print(f"  precision={precision:.2f}  recall={recall:.2f}  accuracy={accuracy:.2f}")
    if fn:
        print(f"  WARNING: {fn} missed problem(s) — the dangerous failure mode.")
    if errors:
        print(f"  WARNING: {errors} sample(s) errored (not scored) — see rows marked ER/!!.")
    print("=" * 50)
    # non-zero exit if the judge made any mistake OR any sample errored (CI / model gating)
    return 0 if (fp == 0 and fn == 0 and errors == 0) else 1


def _looks_like_openrouter(model: str) -> bool:
    """OpenRouter slugs are namespaced (vendor/model) and free ones end in ':free'."""
    return "/" in model or model.endswith(":free")


def _configure_provider(model: str | None, provider: str | None) -> None:
    """Point the LLM client at the right provider for `model`.

    `--model` alone only changes which model string we send; the client's base_url
    and key come from `config.LLM_PROVIDER`, fixed at import. So to test an OpenRouter
    free model while the default provider is OpenAI, we must also switch the client.
    We set config BEFORE the first call, so the lazily-built (and cached) client picks
    it up. Provider is explicit (--provider) or inferred from the slug shape."""
    if model:
        config.JUDGE_MODEL = model
    want = provider or ("openrouter" if (model and _looks_like_openrouter(model)) else None)
    if want == "openrouter":
        if not config.OPENROUTER_API_KEY:
            print("[evals] --model looks like an OpenRouter slug but OPENROUTER_API_KEY is not "
                  "set in .env.\n        Get a free key at https://openrouter.ai/keys and add "
                  "OPENROUTER_API_KEY=... to .env.")
            sys.exit(2)
        config.LLM_PROVIDER = "openrouter"
        config.LLM_API_KEY = config.OPENROUTER_API_KEY
        config.LLM_BASE_URL = "https://openrouter.ai/api/v1"
    elif want == "openai":
        if not config.OPENAI_API_KEY:
            print("[evals] --provider openai but OPENAI_API_KEY is not set in .env.")
            sys.exit(2)
        config.LLM_PROVIDER = "openai"
        config.LLM_API_KEY = config.OPENAI_API_KEY
        config.LLM_BASE_URL = None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run judge-quality evals against the golden cases.")
    ap.add_argument("--model", default=None, help="override JUDGE_MODEL for this run")
    ap.add_argument("--provider", default=None, choices=["openai", "openrouter"],
                    help="force the provider/endpoint (else inferred from the model slug)")
    ap.add_argument("--repeats", type=int, default=1, help="samples per case (judge is noisy)")
    ap.add_argument("--only", default=None, help="run only cases whose name contains this substring")
    ap.add_argument("--kind", default=None, choices=["personalize", "ingest"], help="filter by gate kind")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    args = ap.parse_args()

    cases = CASES
    if args.only:
        cases = [c for c in cases if args.only.lower() in c.name.lower()]
    if args.kind:
        cases = [c for c in cases if c.kind == args.kind]
    if not cases:
        print("no cases match the filter")
        sys.exit(1)

    if args.list:
        for c in cases:
            print(f"  {c.name:<34} {c.kind:<12} expect={'PASS' if c.expect_pass else 'FAIL':<4} "
                  f"({c.insight}) — {c.note}")
        return

    _configure_provider(args.model, args.provider)

    sys.exit(run(cases, args.repeats))


if __name__ == "__main__":
    main()
