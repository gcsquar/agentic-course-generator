"""Unit tests for Agent 1 — all offline (no network, no API key).

These test the pure helper functions and the mock path, so they run in CI
without secrets and guard against accidental contract breakage.
"""
from agents import agent1_ingest
from agents.agent1_ingest import (
    _age_years,
    _count_formulas,
    _extract_images,
    _looks_like_news,
)
from contracts import IngestResult


def test_mock_returns_valid_contract():
    r = agent1_ingest.ingest("https://my.site/x", mock=True)
    assert isinstance(r, IngestResult)
    assert r.url == "https://my.site/x"        # url is overridden with the input
    assert r.accepted is True
    assert len(r.clean_text) >= 600            # long enough to pass the gate


def test_looks_like_news():
    assert _looks_like_news("https://www.bbc.com/news/some-story")
    assert not _looks_like_news("https://example.com/tutorial/transformers")


def test_age_years():
    assert _age_years("") is None
    assert _age_years("not-a-date") is None
    age = _age_years("2000-01-01")
    assert age is not None and age > 20


def test_extract_images_reads_alt_and_caption():
    html = (
        '<img alt="a cat">'
        '<figure><img alt=""><figcaption>a dog</figcaption></figure>'
    )
    imgs = _extract_images(html)
    assert "a cat" in imgs
    assert "a dog" in imgs


def test_count_formulas():
    html = "<math></math><span class='katex'></span>"
    text = "energy $E=mc^2$ and also $a+b$"
    # 1 <math> + 1 .katex + 2 inline LaTeX = 4
    assert _count_formulas(html, text) == 4