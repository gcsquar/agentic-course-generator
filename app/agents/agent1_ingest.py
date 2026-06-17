"""Agent 1 — Content Ingestion & Parsing.

CONTRACT (do not change the signature — the Supervisor depends on it):
    ingest(url: str, *, mock: bool = False) -> IngestResult

Job:  a URL  ->  cleaned, structured article text  +  an accept/reject decision.
Output `IngestResult` must satisfy `gates.gate_ingest` (clean teachable text).
"""
from __future__ import annotations

import datetime
import re

import requests

import config
import llm
from contracts import IngestResult

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_NEWS_HINTS = ("/news/", "/article/", "cnn.", "bbc.", "nytimes.", "reuters.", "bloomberg.")

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

# helpers
def _fetch(url: str) -> str | None:
    """Fetch HTML. Falls back to a browser-like request if the default bot is blocked."""
    try:
        import trafilatura
    except ImportError:
        trafilatura = None

    if trafilatura is not None:
        html = trafilatura.fetch_url(url)
        if html:
            return html

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return None


def _extract_images(html: str) -> list[str]:
    """Collect human-readable descriptions of images: <img alt> + <figcaption>."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    descriptions: list[str] = []

    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        if alt:
            descriptions.append(alt)

    for cap in soup.find_all("figcaption"):
        text = cap.get_text(strip=True)
        if text:
            descriptions.append(text)

    seen: set[str] = set()
    return [d for d in descriptions if not (d in seen or seen.add(d))]


def _count_formulas(html: str, clean_text: str) -> int:
    """Best-effort formula count: rendered math in HTML + inline LaTeX in the text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rendered = len(soup.find_all("math")) + len(soup.select(".katex, .MathJax, .mwe-math-element"))
    inline_latex = len(re.findall(r"\$[^$\n]+\$", clean_text))
    return rendered + inline_latex


def _looks_like_news(url: str) -> bool:
    """Cheap URL-based guess: does this look like a news story?"""
    u = url.lower()
    return any(hint in u for hint in _NEWS_HINTS)


def _age_years(published: str) -> float | None:
    """Rough age of the article in years from a YYYY-MM-DD-ish date string."""
    if not published:
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", published)
    if not m:
        return None
    try:
        d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    return round((datetime.date.today() - d).days / 365.25, 1)


def _judge(title: str, url: str, clean_text: str) -> dict:
    """Ask the LLM whether this article is good source material for 2-3 lessons."""
    return llm.chat_json(
        system=(
            "You judge whether an article is good source material for building "
            "2-3 teaching lessons. ACCEPT substantive tutorials, explainers, "
            "documentation, or in-depth articles with real teaching content. "
            "REJECT news articles, thin or marketing pages, link lists, or anything "
            "too shallow to teach from. "
            'Return JSON: {"accepted": boolean, "reason": string, '
            '"is_news": boolean, "score": number between 0 and 1, '
            '"description": string (1-2 sentence summary)}.'
        ),
        user=(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Article text (first 4000 chars):\n{clean_text[:4000]}"
        ),
        temperature=0.0,
    )
def ingest(url: str, *, mock: bool = False) -> IngestResult:
    if mock:
        return IngestResult(**{**_MOCK.to_dict(), "url": url})

    # 1. Fetch the page (with a browser-like fallback)
    html = _fetch(url)
    if not html:
        return IngestResult(
            url=url,
            accepted=False,
            reason="Could not fetch the URL (network error, 404, or blocked).",
        )

    # extracting a clean article text as markdown+metadata
    clean_text = trafilatura.extract(
        html,
        output_format="markdown",
        include_formatting=True,
        include_links=False,
        include_comments=False,
    ) or ""

    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta else "") or ""
    published = (meta.date if meta else "") or ""

    # pull image descriptions/ccount formulas
    images = _extract_images(html)
    n_formulas = _count_formulas(html, clean_text)
    looks_like_news = _looks_like_news(url)
    age_years = _age_years(published)

    # hard reject: too short to build lessons from
    if len(clean_text) < config.MIN_ARTICLE_CHARS:
        return IngestResult(
            url=url,
            accepted=False,
            reason=(
                f"Extracted text too short ({len(clean_text)} chars, "
                f"need >= {config.MIN_ARTICLE_CHARS}) to build lessons."
            ),
            title=title,
            clean_text=clean_text,
            images=images,
            n_formulas=n_formulas,
            meta={"published": published, "age_years": age_years,
                  "is_news": looks_like_news},
        )

    # here we accept/reject judgment given by LLM
    try:
        verdict = _judge(title, url, clean_text)
    except Exception as exc:
        accepted = not looks_like_news
        verdict = {
            "accepted": accepted,
            "reason": (
                f"(heuristic fallback — LLM unavailable: {exc}) "
                + ("URL looks like news" if looks_like_news
                   else "length OK and not obviously news")
            ),
            "is_news": looks_like_news,
            "score": None,
            "description": "",
        }

    return IngestResult(
        url=url,
        accepted=bool(verdict.get("accepted", False)),
        reason=str(verdict.get("reason", "")),
        title=title,
        clean_text=clean_text,
        description=str(verdict.get("description", "")),
        images=images,
        n_formulas=n_formulas,
        meta={
            "published": published,
            "age_years": age_years,
            "is_news": bool(verdict.get("is_news")) or looks_like_news,
            "score": verdict.get("score"),
        },
    )