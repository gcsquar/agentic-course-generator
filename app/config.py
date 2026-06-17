"""Central configuration: models, thresholds, paths.

Everything tunable lives here so agents stay clean and the reflection
('did the gates help?') has one place to look.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv optional; env vars may be set another way
    pass

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# gpt-4o-mini is cheap & fast for a teaching project; override via env.
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.3

# --- Gate thresholds (Agent 1) ---
MIN_ARTICLE_CHARS = 600          # shorter than this -> probably not teachable

# --- Agent 2 (segmentation) ---
SEGMENT_INPUT_CHARS = 16000      # cap source text fed to the segmenter (cost/latency)

# --- Agent 3 (personalization) ---
PERSONALIZE_LESSON_CHARS = 6000   # cap on lesson.body fed into the draft prompt
PERSONALIZE_PROFILE_CHARS = 1500  # cap on user.raw fed into the draft prompt

# The open web is dirty: only cite background fetched from these trusted domains.
TRUSTED_DOMAINS = [
    "arxiv.org", "nature.com", "pubmed.ncbi.nlm.nih.gov",
    "stanford.edu", "mit.edu", "harvard.edu", ".edu",
    "khanacademy.org", "britannica.com", "ietf.org",
]
RESEARCH_FETCH_CHARS = 4000      # cap text pulled from a researched page

# --- Gate / retry policy ---
MAX_RETRIES = 2                  # how many times the Supervisor re-runs a failed stage

# --- Paths ---
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
USERS_FILE = ROOT / "users.md"
