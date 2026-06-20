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

# --- LLM provider (OpenAI by default; OpenRouter for FREE models) ------------
# OpenRouter is OpenAI-API-compatible and offers free models (the ":free" variants),
# so the whole pipeline can run without OpenAI credit. Put OPENROUTER_API_KEY (and
# optionally OPENROUTER_MODEL) in .env. Force a provider with LLM_PROVIDER=openrouter|openai;
# otherwise we auto-pick OpenRouter only when it is the only key present.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
if LLM_PROVIDER not in ("openai", "openrouter"):
    LLM_PROVIDER = "openrouter" if (OPENROUTER_API_KEY and not OPENAI_API_KEY) else "openai"

if LLM_PROVIDER == "openrouter":
    LLM_API_KEY = OPENROUTER_API_KEY
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    # A free, JSON-capable default; override with OPENROUTER_MODEL (free slugs end in ":free").
    MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
else:
    LLM_API_KEY = OPENAI_API_KEY
    LLM_BASE_URL = None                       # None -> OpenAI SDK uses its default endpoint
    # gpt-4o-mini is cheap & fast for a teaching project; override via OPENAI_MODEL.
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TEMPERATURE = 0.3

# Role-specific models. Generation, judging, and auditing are DIFFERENT-difficulty
# jobs and benefit from different models: judging/auditing the same output with the
# SAME model that wrote it shares its blind spots (a correlated error, not an
# independent check). Default both to MODEL so single-model setups are unchanged;
# set JUDGE_MODEL / AUDITOR_MODEL in .env to get a genuinely independent reviewer.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "").strip() or MODEL
AUDITOR_MODEL = os.getenv("AUDITOR_MODEL", "").strip() or MODEL

# Per-request timeout (seconds) for LLM completions, so a hung call can't block an
# Agent 3 worker thread forever. Override with LLM_TIMEOUT.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# --- Gate thresholds (Agent 1) ---
MIN_ARTICLE_CHARS = 600          # shorter than this -> probably not teachable

# --- Agent 2 (segmentation) ---
MIN_LESSONS = 3                  # fewer than this isn't segmentation (1 giant lesson = collapse)
# Max paragraphs fed to the segmenter. Generous so normal articles are fully covered; a
# longer source's tail is DROPPED past this — but the coverage gate now flags that loss
# instead of passing silently (it used to be a hard [:100] that hid half a long article).
SEGMENT_MAX_PARAGRAPHS = 400
# A SMELL threshold, not a hard rule: a lesson spanning more paragraphs than this has almost
# certainly lumped several subtopics (one 52-paragraph lesson held a quarter of an article).
# Generous on purpose — a genuinely coherent subtopic can be large; this only catches dumping.
MAX_LESSON_PARAGRAPHS = 30

# Faithfulness reference window: how much SOURCE the personalization gate and the
# independent auditor may see when judging "is this claim in the source?". Must be big
# enough to hold the WHOLE course, or in-source terms get flagged as invented (a real
# bug we hit — see INSIGHTS.md #6). 100k chars ~= 25k tokens, fine for 128k-context
# models. NOTE: this does not scale to very large articles; see the INSIGHTS scale caveat.
FAITHFULNESS_SOURCE_CHARS = 100000

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
