"""Shared: load learner profiles from users.md.

Lives here (not inside Agent 3) so the orchestrator/supervisor can load users
without depending on an agent module. The users.md FORMAT is a shared
contract — change it by consensus, since Agent 3 reads these fields.
"""
from __future__ import annotations

import re
from pathlib import Path

from contracts import UserProfile


def parse_users(path: Path) -> list[UserProfile]:
    """Parse users.md: each `## Name` block with `- key: value` lines."""
    text = path.read_text(encoding="utf-8")
    users: list[UserProfile] = []
    blocks = re.split(r"^##\s+", text, flags=re.MULTILINE)[1:]
    for block in blocks:
        lines = block.splitlines()
        name = lines[0].strip()
        fields: dict[str, str] = {}
        for line in lines[1:]:
            m = re.match(r"\s*-\s*([\w ]+?)\s*:\s*(.+)", line)
            if m:
                fields[m.group(1).strip().lower()] = m.group(2).strip()
        users.append(UserProfile(
            name=name,
            role=fields.get("role", ""),
            level=fields.get("level", ""),
            interests=fields.get("interests", ""),
            tone=fields.get("tone", ""),
            age=fields.get("age", ""),
            region=fields.get("region", ""),
            education=fields.get("education", ""),
            experience=fields.get("experience", ""),
            languages=fields.get("languages") or fields.get("language", ""),
            focus=fields.get("focus", ""),
            raw=block.strip(),
        ))
    return users
