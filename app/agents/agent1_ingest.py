"""Agent 1 — Content Ingestion & Parsing. 

CONTRACT (do not change the signature — the Supervisor depends on it):
    ingest(url: str, *, mock: bool = False) -> IngestResult

Job:  a URL  ->  cleaned, structured article text  +  an accept/reject decision.
Output `IngestResult` must satisfy `gates.gate_ingest` (clean teachable text).

This is a STUB. `mock=True` returns canned data so the whole pipeline runs
end-to-end today; the real path raises until you implement it.
"""
from __future__ import annotations

from contracts import IngestResult

_MOCK = IngestResult(
    url="https://example.com/intro-to-transformers",
    accepted=True,
    reason="(mock) clear tutorial-style prose with sections and a worked example",
    title="A Gentle Intro to Transformers",
    clean_text=(
        "# A Gentle Intro to Transformers\n\n"
        "Transformers replaced recurrence with self-attention. Each token attends to "
        "every other token, weighted by relevance, computed as softmax(QK^T / sqrt(d)) V, "
        "where Q, K and V are linear projections of the input embeddings into queries, "
        "keys and values. Because the dot products grow with dimension, we scale by "
        "sqrt(d) to keep the softmax in a stable range. "
        "Multi-head attention runs several such projections in parallel, letting the model "
        "attend to different kinds of relationships at once, then concatenates and "
        "re-projects the heads. "
        "Positional encodings inject order into the model because attention is "
        "permutation-invariant: without them a shuffled sentence would look identical. "
        "Each block also contains a position-wise feed-forward network, residual "
        "connections and layer normalization, which together make deep stacks trainable. "
        "Stacking these blocks gives the full architecture, trained with masked or causal "
        "objectives depending on whether the task is understanding or generation."
    ),
    description="An introduction to the Transformer architecture and self-attention.",
    images=["diagram of scaled dot-product attention"],
    n_formulas=1,
)


def ingest(url: str, *, mock: bool = False) -> IngestResult:
    if mock:
        return IngestResult(**{**_MOCK.to_dict(), "url": url})
    # TODO: real ingestion — fetch + clean (trafilatura/BeautifulSoup),
    # detect formulas, and make the accept/reject judgment.
    raise NotImplementedError(
        "Agent 1 not implemented yet. Run with --mock, or implement ingest()."
    )
