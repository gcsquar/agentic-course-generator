"""Independent auditor.

A fresh, unbiased reviewer that reads a FINISHED run from disk (it never saw the
generation in-context) and judges the whole chain end-to-end:

  source  ->  lessons  ->  personalized lessons + citations

...and critiques the Supervisor's OWN gate decisions (did a gate pass something it
shouldn't have, or block something good?). Independence comes from (a) a fresh LLM
call with no shared context and (b) reading only the persisted artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
import llm
from storage import Run

_SYSTEM = (
    "You are an INDEPENDENT auditor. You did not generate any of this content and "
    "have no stake in it. You are given a finished pipeline run: the source article, "
    "the micro-lessons derived from it, the per-learner personalized lessons (some "
    "with researched background + citations), and the quality-gate verdicts the "
    "system recorded for each stage.\n"
    "Audit the run end-to-end and report problems:\n"
    "- Faithfulness: lessons/personalizations must not contradict or invent beyond the source.\n"
    "- Segmentation: lessons should be standalone, distinct, and cover the source.\n"
    "- Personalization: each lesson should genuinely fit its learner's level/tone/interests.\n"
    "- Citations: background ADDED from outside the source must include a URL. A lesson that "
    "faithfully rephrases the source needs NO citation — do not penalize its absence.\n"
    "- Gate decisions: judge whether the recorded gate verdicts were correct (flag a gate "
    "that PASSED flawed output, or one that was needlessly strict).\n"
    'Return JSON {"score": int 0-100, "verdict": "pass"|"concerns"|"fail", '
    '"findings": [{"area": str, "severity": "high"|"low", "note": str}], '
    '"summary": str}.'
)


def audit_run(run: Run) -> dict:
    d = run.dir
    ingest = _load(d / "01_ingest.json") or {}
    curriculum = _load(d / "02_curriculum.json") or {}
    personalized = _load(d / "03_personalized.json") or []
    gate_files = {p.name: _load(p) for p in sorted(d.glob("*__gate*.json"))}

    report = llm.chat_json(
        system=_SYSTEM,
        user=_compile(ingest, curriculum, personalized, gate_files),
        temperature=0.0,
    )
    run.save_json("99_audit", report)
    run.save_md("99_audit", _render(report))

    highs = [f for f in report.get("findings", []) if str(f.get("severity")).lower() == "high"]
    print(f"[auditor] verdict={report.get('verdict')} score={report.get('score')} "
          f"high-severity findings={len(highs)}")
    return report


# ---------------------------------------------------------------- helpers
def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _compile(ingest, curriculum, personalized, gate_files) -> str:
    src = (ingest.get("clean_text", "") or "")[:config.SEGMENT_INPUT_CHARS]
    lessons = "\n".join(
        f"- [{l.get('order')}] {l.get('title')}: {(l.get('body') or '')[:300]}"
        for l in curriculum.get("lessons", [])
    )
    pers = "\n".join(
        f"- {p.get('user')} [{p.get('order')}] {p.get('title')} "
        f"(cites: {p.get('citations') or 'none'}): {(p.get('body') or '')[:250]}"
        for p in personalized
    )
    gates = "\n".join(
        f"- {name}: passed={v.get('passed')} issues={v.get('issues')}"
        for name, v in gate_files.items() if v
    )
    return (
        f"SOURCE (title: {ingest.get('title','?')}, accepted={ingest.get('accepted')}):\n{src}\n\n"
        f"LESSONS:\n{lessons}\n\n"
        f"PERSONALIZED:\n{pers}\n\n"
        f"RECORDED GATE VERDICTS:\n{gates}"
    )


def _render(report: dict) -> str:
    lines = [
        "# Independent Audit",
        "",
        f"**Verdict:** {report.get('verdict')}  ·  **Score:** {report.get('score')}/100",
        "",
        report.get("summary", ""),
        "",
        "## Findings",
    ]
    findings = report.get("findings", [])
    if not findings:
        lines.append("_None._")
    for f in findings:
        lines.append(f"- **[{f.get('severity')}] {f.get('area')}** — {f.get('note')}")
    return "\n".join(lines) + "\n"
