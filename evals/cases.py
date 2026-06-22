"""Golden eval cases — each documented INSIGHTS incident as a runnable scenario
with a KNOWN-correct gate verdict.

These exercise the LLM judges (so they need an API key and cost money — they are
NOT pytest/CI tests; run them via `python evals/run_evals.py`). The point is to
make judge quality MEASURABLE: a model that misses the syrniki analogy (recall)
or flags an in-source term as invented (precision) shows up as a failing case,
so "how smart a model do we need?" and "did this prompt change regress?" become
data, not vibes (see ROADMAP Wave 1 and INSIGHTS #7).

Each Case names the gate to run and what the right answer is:
  - expect_pass:          should the gate PASS (no blocking issue)?
  - expect_issue_substr:  informational — a correct FAIL should mention this
                          (reason match; not part of pass/fail scoring).
  - forbid_issue_substr:  a precision tripwire — the gate must NOT raise an issue
                          naming this (e.g. an in-source term flagged as invented).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from contracts import Curriculum, Lesson, PersonalizedLesson, UserProfile, IngestResult


@dataclass
class Case:
    name: str
    insight: str                 # which INSIGHTS note this reproduces
    kind: str                    # "personalize" | "ingest"
    expect_pass: bool
    build: Callable[[], dict]    # -> kwargs for the gate (see run_evals._run_case)
    expect_issue_substr: list[str] = field(default_factory=list)
    forbid_issue_substr: list[str] = field(default_factory=list)
    note: str = ""


# --------------------------------------------------------------- shared profiles
def _mike() -> UserProfile:
    return UserProfile(
        name="Mike", role="Senior ML Engineer", level="expert",
        interests="production systems, scaling, tradeoffs",
        tone="short, with math analogies",
        focus="when to use a technique and its tradeoffs",
        raw=("## Mike\n- role: Senior ML Engineer\n- level: expert\n"
             "- interests: production systems, scaling, tradeoffs\n"
             "- tone: short, likes math analogies\n"
             "- focus: when to use a technique and its tradeoffs, not derivations"))


def _one_lesson_curriculum(title: str, body: str) -> Curriculum:
    return Curriculum(source_url="http://x", key_concepts=[],
                      lessons=[Lesson(order=1, title=title, body=body,
                                      source_span=body, start_idx=0, end_idx=0)])


# =============================================================== personalize cases

def _case_syrniki() -> dict:
    """INSIGHTS #1: a faithful-but-useless decorative analogy. The judge must flag it."""
    body = ("# Syrniki (Russian Cheese Pancakes)\n\n"
            "Syrniki are small fried quark pancakes. Ingredients: 1 lb farmer's cheese, "
            "2 eggs, half a cup of flour, a quarter cup of sugar, and optional raisins. "
            "Mix the cheese, eggs, and sugar into a soft dough, form small patties, dredge "
            "them in flour, and fry in butter over medium heat until golden on both sides. "
            "Serve warm with sour cream or jam.")
    cur = _one_lesson_curriculum("Syrniki", body)
    tailored = ("**Ingredients for Syrniki** — 1 lb farmer's cheese (analogous to your "
                "model's training data), 2 eggs (like the activation functions that bind "
                "everything together), half a cup of flour (similar to hyperparameters), a "
                "quarter cup of sugar (the sweetener for your output), and optional raisins "
                "(like the noise in your data). Mix and fry the patties the way you would "
                "tune a training pipeline until they converge to golden brown.")
    pers = [PersonalizedLesson(user="Mike", order=1, title="Syrniki", body=tailored)]
    return {"users": [_mike()], "curriculum": cur, "personalized": pers}


def _case_fabricated_metric() -> dict:
    """INSIGHTS #2/#14: inventing magnitudes the source never states."""
    body = ("Gradient checkpointing trades compute for memory. Instead of storing every "
            "activation for the backward pass, it recomputes some activations during "
            "backpropagation. This lowers peak memory usage at the cost of extra forward "
            "passes, which makes it useful when a model would otherwise not fit in GPU memory.")
    cur = _one_lesson_curriculum("Gradient Checkpointing", body)
    tailored = ("Gradient checkpointing recomputes activations in the backward pass to save "
                "memory. In practice it cuts peak memory by about 3x while making training "
                "roughly 30% slower — a worthwhile trade when the model won't fit otherwise.")
    pers = [PersonalizedLesson(user="Mike", order=1, title="Gradient Checkpointing", body=tailored)]
    return {"users": [_mike()], "curriculum": cur, "personalized": pers}


def _case_invented_further_reading() -> dict:
    """INSIGHTS #15: the source only NAMES references; the rewrite invents their focus."""
    body = ("## Further reading\n\n"
            "- \"Attention Is All You Need\" (Vaswani et al., 2017)\n"
            "- \"Layer Normalization\" (Ba et al., 2016)")
    cur = _one_lesson_curriculum("Further Reading", body)
    tailored = ("For deeper study: \"Attention Is All You Need\" focuses on replacing "
                "recurrence entirely with self-attention to reach state-of-the-art BLEU "
                "scores on machine translation, while \"Layer Normalization\" shows how "
                "normalizing across feature dimensions stabilizes recurrent network training.")
    pers = [PersonalizedLesson(user="Mike", order=1, title="Further Reading", body=tailored)]
    return {"users": [_mike()], "curriculum": cur, "personalized": pers}


def _case_added_advice() -> dict:
    """INSIGHTS #14: personalization must change HOW the source is presented, never
    WHAT it claims. Adding the agent's own critique / deployment advice / tradeoff
    analysis the source never states is a faithfulness violation, distinct from the
    fabricated-number case. The judge must flag it high."""
    body = ("Cosine annealing decays the learning rate following a cosine curve from its "
            "initial value down to near zero over the course of training. It is a common "
            "schedule for training deep neural networks.")
    cur = _one_lesson_curriculum("Cosine Annealing", body)
    # The source says WHAT it is; this rewrite injects opinions, a recommendation, and
    # deployment advice the source never makes.
    tailored = ("Cosine annealing decays the LR along a cosine curve to near zero. Honestly "
                "this is often suboptimal — you should prefer a linear warmup followed by "
                "linear decay, and in production you'd want to retune the cycle length per "
                "dataset and add restarts to escape sharp minima.")
    pers = [PersonalizedLesson(user="Mike", order=1, title="Cosine Annealing", body=tailored)]
    return {"users": [_mike()], "curriculum": cur, "personalized": pers}


def _case_faithful_rewrite() -> dict:
    """Precision: a faithful, level-appropriate rephrase must PASS (no high issues)."""
    body = ("Self-attention lets each token in a sequence attend to every other token, "
            "weighting them by relevance. The weights come from scaled dot products between "
            "query and key projections, normalized with a softmax, and are used to combine "
            "the value projections into a context-aware representation for each position.")
    cur = _one_lesson_curriculum("Self-Attention", body)
    tailored = ("Self-attention computes, per token, a softmax over scaled query-key dot "
                "products and uses those weights to mix the value vectors — giving each "
                "position a context-aware representation. Dense and to the point, no "
                "derivation.")
    pers = [PersonalizedLesson(user="Mike", order=1, title="Self-Attention", body=tailored)]
    return {"users": [_mike()], "curriculum": cur, "personalized": pers}


def _case_in_source_term_deep() -> dict:
    """INSIGHTS #6: a term that appears in ANOTHER lesson is in-source, not invented.
    The judge must NOT flag a reference to it (precision against truncation). Every
    lesson is personalized so the deterministic coverage check passes and only the
    faithfulness judge is under test."""
    cur = Curriculum(
        source_url="http://x", key_concepts=[],
        lessons=[
            Lesson(order=1, title="Self-Attention",
                   body="Self-attention lets each token attend to every other token, weighted by relevance.",
                   start_idx=0, end_idx=0),
            Lesson(order=2, title="Block Structure",
                   body=("Each transformer block also applies layer normalization and residual "
                         "connections, which together keep deep stacks trainable."),
                   start_idx=1, end_idx=1),
        ])
    # Lesson 1's rewrite references "layer normalization" (introduced in lesson 2 ->
    # in-source). It must stay FAITHFUL: the block-level property (trainability) is
    # attributed to layer norm + residual connections exactly as lesson 2 states it —
    # NOT to self-attention — so the only thing under test is "is the in-source term
    # treated as invented?" (it must not be), with no misattribution confound.
    pers = [
        PersonalizedLesson(user="Mike", order=1, title="Self-Attention",
                           body=("Self-attention lets each token weigh every other by relevance — "
                                 "the core mixing step. It sits inside a transformer block which, as "
                                 "the next lesson covers, wraps it with layer normalization and "
                                 "residual connections to keep deep stacks trainable.")),
        PersonalizedLesson(user="Mike", order=2, title="Block Structure",
                           body=("Each block wraps the attention sub-layer with layer normalization "
                                 "and residual connections; together these keep deep stacks trainable.")),
    ]
    return {"users": [_mike()], "curriculum": cur, "personalized": pers}


# ==================================================================== ingest cases

def _ingest(title: str, text: str) -> IngestResult:
    return IngestResult(url="http://example.com/x", accepted=True,
                        reason="agent accepted", title=title, clean_text=text,
                        description="", meta={})   # no is_news -> let the LLM judge decide


def _case_news_rejected() -> dict:
    """INSIGHTS #3: time-sensitive reporting is not teachable. The gate judge must fail it."""
    text = ("Stocks tumbled on Thursday as investors reacted to the morning's inflation data. "
            "The S&P 500 fell 2.1% in afternoon trading and the Nasdaq dropped 2.8%, led by "
            "losses in technology shares. Federal Reserve officials signaled they may keep "
            "interest rates higher for longer, citing persistent price pressures. 'The market "
            "is repricing its expectations for rate cuts this year,' said one analyst at a major "
            "bank. Oil prices slid amid concerns about weakening demand. Traders now turn to "
            "next week's jobs report, which could further shape the central bank's path. The "
            "sell-off erased the week's earlier gains, leaving major indexes near monthly lows.")
    return {"ingest": _ingest("Stocks Tumble After Inflation Data", text)}


def _case_encyclopedic_accepted() -> dict:
    """INSIGHTS #7: an in-depth encyclopedic article IS valid source material. The gate
    judge must NOT over-reject it for lacking tutorial structure (precision)."""
    text = ("Photosynthesis is the process by which green plants, algae, and some bacteria "
            "convert light energy into chemical energy stored in glucose. It takes place mainly "
            "in the chloroplasts, organelles that contain chlorophyll, a pigment that absorbs "
            "light most strongly in the blue and red wavelengths. The process has two stages. "
            "The light-dependent reactions occur in the thylakoid membranes, producing ATP and "
            "NADPH and releasing oxygen as a byproduct of splitting water. The light-independent "
            "reactions, or Calvin cycle, occur in the stroma and use that ATP and NADPH to fix "
            "carbon dioxide into sugars. Photosynthesis underpins most food chains and maintains "
            "atmospheric oxygen, making it fundamental to life on Earth.")
    return {"ingest": _ingest("Photosynthesis", text)}


# ============================================================================ list
CASES: list[Case] = [
    Case("syrniki_decorative_analogy", "INSIGHTS #1", "personalize", expect_pass=False,
         build=_case_syrniki, expect_issue_substr=["analog"],
         note="faithful-but-useless ML analogies over a recipe -> high"),
    Case("fabricated_metric", "INSIGHTS #2/#14", "personalize", expect_pass=False,
         build=_case_fabricated_metric, expect_issue_substr=["3x", "30%", "metric", "number"],
         note="invents ~3x / 30% the source never states -> high"),
    Case("added_advice_not_in_source", "INSIGHTS #14", "personalize", expect_pass=False,
         build=_case_added_advice,
         expect_issue_substr=["advice", "recommend", "opinion", "tradeoff", "suboptimal",
                              "not in the source", "deployment", "critique"],
         note="injects critique/recommendation/deployment advice the source never states -> high"),
    Case("invented_further_reading_focus", "INSIGHTS #15", "personalize", expect_pass=False,
         build=_case_invented_further_reading, expect_issue_substr=["focus", "describe", "invent"],
         note="source only names references; rewrite invents their focus -> high"),
    Case("faithful_rewrite_passes", "precision", "personalize", expect_pass=True,
         build=_case_faithful_rewrite,
         note="a faithful, level-appropriate rephrase must pass"),
    Case("in_source_term_not_invented", "INSIGHTS #6", "personalize", expect_pass=True,
         build=_case_in_source_term_deep,
         note="a term from another lesson is in-source; a faithful reference must not fail the gate"),
    Case("news_rejected", "INSIGHTS #3", "ingest", expect_pass=False,
         build=_case_news_rejected, expect_issue_substr=["news", "time-sensitive", "reporting"],
         note="market report -> not teachable -> fail"),
    Case("encyclopedic_accepted", "INSIGHTS #7", "ingest", expect_pass=True,
         build=_case_encyclopedic_accepted,
         note="in-depth encyclopedic article -> valid source -> pass"),
]
