"""Data contracts shared by all three agents.

These dataclasses are the API between agents. An agent only ever sees the
contract it is given and only ever returns the contract it owns — so each one
can be built and tested in isolation against mocked inputs.

Flow:  URL --> Agent1 --> IngestResult --> Agent2 --> Curriculum
                                                   \\--> Agent3 (+ users) --> [PersonalizedLesson]
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------- Agent 1
@dataclass
class IngestResult:
    """Output of Agent 1 (Content Ingestion & Parsing)."""
    url: str
    accepted: bool                 # accept / reject judgment call
    reason: str                    # why accepted or rejected
    title: str = ""
    clean_text: str = ""           # cleaned, structured article text (markdown)
    description: str = ""          # 1-2 sentence summary of the article (Agent 1; read by Agent 2 + audit)
    images: list[str] = field(default_factory=list)        # descriptions of relevant images
    n_formulas: int = 0            # formulas detected (for the faithfulness gate)
    # `meta` holds soft accept/reject signals (e.g. {"score": .., "published": .., "is_news": ..})
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- Agent 2
@dataclass
class Lesson:
    """A single standalone micro-lesson."""
    order: int
    title: str
    body: str                      # self-contained lesson text (markdown)
    source_span: str = ""          # the slice of source it was built from (faithfulness)
    description: str = ""          # short summary of this lesson (Agent 2)
    key_concepts: list[str] = field(default_factory=list)  # concepts this lesson teaches
    n_formulas: int = 0
    start_idx: int = -1            # source paragraph range this lesson covers (Agent 2);
    end_idx: int = -1              # the segmentation gate uses it to check coverage. -1 = unset
    # depends_on: list[int] = field(default_factory=list)  # optional: prerequisite lesson orders (topic graph)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Curriculum:
    """Output of Agent 2 (Curriculum Segmentation)."""
    source_url: str
    key_concepts: list[str] = field(default_factory=list)  # article-level concepts (Agent 2 -> Agent 3 topic match)
    lessons: list[Lesson] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"source_url": self.source_url,
                "key_concepts": self.key_concepts,
                "lessons": [l.to_dict() for l in self.lessons]}


# ---------------------------------------------------------------- Agent 3
@dataclass
class UserProfile:
    """A user parsed from users.md.

    `raw` holds the FULL profile text — Agent 3's LLM reads that, so a field only
    needs to be structured when something uses it programmatically (e.g. language
    routing, level logic). Free-form preferences can live in `raw` alone.
    """
    name: str
    role: str = ""
    level: str = ""                # e.g. beginner / intermediate / expert
    interests: str = ""
    tone: str = ""                 # e.g. likes humor / formal / concise
    age: str = ""
    region: str = ""
    education: str = ""
    experience: str = ""
    languages: str = ""            # preferred language(s) for the output
    focus: str = ""                # what the learner specifically wants to focus on
    # learning style fields (populated from profiling form / users.md)
    reading_style: str = ""        # how they approach new material
    explanation_style: str = ""    # what kind of explanation helps most
    error_handling: str = ""       # how to handle mistakes
    pace: str = ""                 # preferred speed
    tone_note: str = ""            # specific tone preferences / annoyances
    new_terms: str = ""            # how to introduce new terminology
    background_gaps: str = ""      # how deep to go on prerequisites
    session_state: str = ""        # current energy/time state for this session
    raw: str = ""                  # the full profile blob, for the LLM to read

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonalizedLesson:
    """Output of Agent 3 — one lesson tailored to one user."""
    user: str
    order: int
    title: str
    body: str                      # tailored lesson text (markdown)
    citations: list[str] = field(default_factory=list)  # added-info sources
    topic_fit: str = ""            # non-empty when article topic poorly fits the user's interests

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- Gates
@dataclass
class GateResult:
    """Returned by every quality gate."""
    passed: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
