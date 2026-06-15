"""Self-contained web research for Agent 3's gap-filling loop.

No API key: web search via DuckDuckGo's HTML endpoint, page text via trafilatura.
The open web is dirty, so results are filtered to config.TRUSTED_DOMAINS before
anything is cited — that filter is the quality control for the research loop.
"""
from __future__ import annotations

import urllib.parse

import requests
import trafilatura

import config

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def web_search(query: str, *, max_results: int = 10) -> list[tuple[str, str]]:
    """Return [(title, url)] from DuckDuckGo's HTML endpoint."""
    try:
        resp = requests.post("https://html.duckduckgo.com/html/",
                             data={"q": query}, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[tuple[str, str]] = []
    for a in soup.select("a.result__a"):
        url = _unwrap(a.get("href", ""))
        title = a.get_text(strip=True)
        if url:
            results.append((title, url))
        if len(results) >= max_results:
            break
    return results


def is_trusted(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d.lstrip(".")) or host.endswith(d)
               for d in config.TRUSTED_DOMAINS)


def fetch_clean(url: str) -> str:
    """Download a page and return cleaned text (capped)."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    text = trafilatura.extract(downloaded, favor_recall=True) or ""
    return text[:config.RESEARCH_FETCH_CHARS]


def research(query: str) -> tuple[str, str] | None:
    """Search, pick the first TRUSTED result with usable text, return (text, url).

    Returns None if nothing trusted and substantial is found — better to add no
    background than to cite a dirty source.
    """
    for _title, url in web_search(query):
        if not is_trusted(url):
            continue
        text = fetch_clean(url)
        if len(text) >= 400:
            return text, url
    return None


def _unwrap(href: str) -> str | None:
    """DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<encoded-url>."""
    if not href:
        return None
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs:
        return urllib.parse.unquote(qs["uddg"][0])
    return href if href.startswith("http") else None
