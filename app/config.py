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


def _load_openrouter_key_file() -> str:
    """Load a local OpenRouter key file without requiring it to be valid Python.

    Supports either a raw key as the first non-comment line, or a simple
    assignment like `OPENROUTER_API_KEY = "..."`. Env vars still override this.
    """
    path = Path(os.getenv(
        "OPENROUTER_API_KEY_FILE",
        str(Path(__file__).resolve().parents[1] / "openrouter_key.py"),
    )).expanduser()
    if not path.exists():
        return ""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            line = line.split("=", 1)[1].strip()
        value = line.strip().strip('"\'')
        if value.startswith("sk-"):
            return value
    return ""

# --- LLM provider (OpenAI by default; OpenRouter for FREE models) ------------
# OpenRouter is OpenAI-API-compatible and offers free models (the ":free" variants),
# so the whole pipeline can run without OpenAI credit. Put OPENROUTER_API_KEY (and
# optionally OPENROUTER_MODEL) in .env. Force a provider with LLM_PROVIDER=openrouter|openai;
# otherwise we auto-pick OpenRouter only when it is the only key present.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") or _load_openrouter_key_file()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
if LLM_PROVIDER not in ("openai", "openrouter"):
    LLM_PROVIDER = "openrouter" if (OPENROUTER_API_KEY and not OPENAI_API_KEY) else "openai"

if LLM_PROVIDER == "openrouter":
    LLM_API_KEY = OPENROUTER_API_KEY
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    # Override with OPENROUTER_MODEL when needed.
    MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-pro")
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

# Confirmation-sampling (INSIGHTS #2): the personalization judge re-checks a clean PASS
# with a SECOND, independent sample before trusting it, so the retry loop can't stop on
# one lenient verdict. Independence is real only if the second draw differs from the
# first. Set CONFIRM_MODEL to a DIFFERENT model for a genuinely independent second opinion
# (it routes to its own provider like the others). If unset, the confirmation falls back
# to a hotter-temperature draw of JUDGE_MODEL — which varies only on models that actually
# sample (reasoning models that pin temperature make that draw a no-op).
CONFIRM_MODEL = os.getenv("CONFIRM_MODEL", "").strip()

# Per-request timeout (seconds) for LLM completions, so a hung call can't block an
# Agent 3 worker thread forever. Override with LLM_TIMEOUT.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# OpenRouter routing/caching. Keep the app pinned to DeepSeek unless explicitly
# overridden; disabling fallbacks makes provider drift visible instead of silent.
OPENROUTER_PROVIDER_ONLY = [p.strip() for p in os.getenv(
    "OPENROUTER_PROVIDER_ONLY", "deepseek"
).split(",") if p.strip()]
OPENROUTER_ALLOW_FALLBACKS = os.getenv("OPENROUTER_ALLOW_FALLBACKS", "false").lower() in (
    "1", "true", "yes", "on"
)
OPENROUTER_REQUIRE_PARAMETERS = os.getenv("OPENROUTER_REQUIRE_PARAMETERS", "false").lower() in (
    "1", "true", "yes", "on"
)
OPENROUTER_CACHE = os.getenv("OPENROUTER_CACHE", "true").lower() in ("1", "true", "yes", "on")
OPENROUTER_CACHE_TTL = os.getenv("OPENROUTER_CACHE_TTL", "3600")
OPENROUTER_SESSION_ID = os.getenv("OPENROUTER_SESSION_ID", "agentic-course-generator")[:256]

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
# Cap lesson count so per-lesson personalization checks stay tractable. Agent 2
# still tiles the whole source; extra fine-grained cuts are merged.
MAX_LESSONS = int(os.getenv("MAX_LESSONS", "12"))

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
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))   # default re-runs of a failed stage

# Personalization (Agent 3) is the most expensive stage to retry (N users x M lessons x
# draft + confirmation-draw), but also the one whose gate feedback is the most actionable
# (per-learner, per-lesson "fix this") — so it gets its OWN, higher cap. NOTE: extra retries
# only buy something while the stage keeps REDUCING its issue count; the early-stop
# (ROADMAP 2.2) cuts a non-converging stage short regardless of this cap, so raising it does
# nothing for a run that plateaus on "no improvement". Override with MAX_RETRIES_PERSONALIZE.
MAX_RETRIES_PERSONALIZE = int(os.getenv("MAX_RETRIES_PERSONALIZE", "4"))

# Per-lesson personalization judging is already much smaller and stricter than the
# old whole-course judge. Keep confirmation available, but allow real runs to skip
# the second pass when cost/latency matters.
CONFIRM_PERSONALIZE_GATE = os.getenv("CONFIRM_PERSONALIZE_GATE", "true").lower() in (
    "1", "true", "yes", "on"
)

# Escalation (ROADMAP 2.5): if the base generation model can't satisfy a stage's gate
# within MAX_RETRIES (or stops converging), the Supervisor makes ONE final attempt on a
# STRONGER model. Set ESCALATION_MODEL to that model (it routes to its own provider like
# the others). Unset -> no escalation. Affects GENERATION only; gates keep JUDGE_MODEL.
ESCALATION_MODEL = os.getenv("ESCALATION_MODEL", "").strip()

# --- Paths ---
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
USERS_FILE = ROOT / "users.md"
