"""Agent 1 — Content Ingestion & Parsing.

CONTRACT (do not change the signature — the Supervisor depends on it):
    ingest(url: str, *, mock: bool = False) -> IngestResult

Job:  a URL  ->  cleaned, structured article text  +  an accept/reject decision.
Output `IngestResult` must satisfy `gates.gate_ingest` (clean teachable text).
"""
from __future__ import annotations

import datetime
import re
import time
import urllib.parse

import requests
import trafilatura

import config
import llm
from contracts import IngestResult

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# How many extra times to retry a TRANSIENT fetch failure (timeout / connection / 5xx).
# A bad source still gets 0 supervisor retries; this only makes the network resilient so
# a flaky connection doesn't halt a good URL (a permanent 4xx is not retried).
_FETCH_RETRIES = 2

# URL-based news guess. Kept deliberately narrow: this signal now HARD-FAILS the ingest
# gate, and Agent 1 gets 0 retries, so a false positive halts a good source. The LLM judge
# is the broader, content-aware news check.
#   - a "/news/" path SECTION is a strong section signal;
#   - news OUTLETS are matched by HOST on a dot boundary, so "bbc.co.uk" matches but
#     "abbc.company.com" does NOT (a plain `"bbc." in url` substring test let that through).
_NEWS_PATH_HINTS = ("/news/",)
_NEWS_DOMAINS = ("cnn.com", "bbc.com", "bbc.co.uk", "nytimes.com", "reuters.com",
                 "bloomberg.com", "apnews.com", "theguardian.com")

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
    """Fetch HTML, retrying TRANSIENT failures (timeout / connection / 5xx) with backoff.

    Falls back to a browser-like request if trafilatura's default bot is blocked. A
    permanent 4xx (404/403/410) is NOT retried — re-fetching can't fix it. This keeps a
    flaky network from halting a good URL (Agent 1 itself gets 0 supervisor retries)."""
    for attempt in range(_FETCH_RETRIES + 1):
        html = trafilatura.fetch_url(url)
        if html:
            return html

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500:
                return None        # permanent client error — don't retry
            # 5xx -> transient, fall through to retry
        except requests.RequestException:
            pass                   # connection error / timeout -> transient, retry

        if attempt < _FETCH_RETRIES:
            time.sleep(2 ** attempt)   # 1s, 2s
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
    """Cheap URL-based guess: does this look like a news story? Matches a `/news/` path
    section, or a known news OUTLET by host on a dot boundary (so `abbc.company.com`
    is NOT mistaken for `bbc.com`)."""
    parsed = urllib.parse.urlparse(url.lower())
    host = parsed.netloc.split(":", 1)[0]
    if any(hint in parsed.path for hint in _NEWS_PATH_HINTS):
        return True
    return any(host == d or host.endswith("." + d) for d in _NEWS_DOMAINS)


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
            "You judge whether an article is good SOURCE MATERIAL for building 2-3 teaching "
            "lessons. The downstream pipeline adds the pedagogy itself — it segments and "
            "rewrites this text into lessons — so the source does NOT need to be "
            "tutorial-shaped.\n"
            "ACCEPT anything with substantive, factual, teachable content: tutorials, "
            "explainers, documentation, AND in-depth reference or encyclopedic articles (a "
            "thorough Wikipedia article qualifies). Missing step-by-step structure, code "
            "examples, or exercises is NOT a reason to reject — adding those is the pipeline's "
            "job, not the source's.\n"
            "REJECT only: news / time-sensitive reporting, marketing or sales pages, link "
            "lists / navigation, stubs, or content genuinely too thin or off-topic to teach "
            "from.\n"
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

    # 1. Fetch the page (browser-like fallback + transient-failure retries inside _fetch)
    html = _fetch(url)
    if not html:
        # A fetch failure is INFRASTRUCTURE, not a content judgment — flag it so the halt
        # reads as "couldn't retrieve" rather than "rejected this source as unteachable".
        return IngestResult(
            url=url,
            accepted=False,
            reason="Could not fetch the URL after retries (timeout, network error, 4xx, or blocked).",
            meta={"fetch_failed": True},
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