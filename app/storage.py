"""Run-folder storage — the audit trail AND the shared-state blackboard.

Every run gets its own folder under runs/<run_id>/. Each stage writes:
  - a .json file  (machine handoff / re-load)
  - a .md file    (human-readable, keeps the source next to the output so a
                   judge can diff 'what the source said' vs 'what we produced')
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config


class Run:
    """One end-to-end run. `run_id` is caller-supplied to stay deterministic
    (no Date.now in this codebase's spirit — pass a stamp from main)."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.dir: Path = config.RUNS_DIR / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, data: dict[str, Any] | list[Any]) -> Path:
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_md(self, name: str, text: str) -> Path:
        path = self.dir / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        return path

    def load_json(self, name: str) -> Any:
        return json.loads((self.dir / f"{name}.json").read_text(encoding="utf-8"))
