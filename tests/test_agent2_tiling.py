"""Deterministic tiling tests for Agent 2 (no LLM, no network).

Agent 2 trusts the LLM only for CUT POINTS (`start_idx`); the spans are computed
in code so coverage is correct BY CONSTRUCTION. These tests pin that guarantee:
no matter how sloppy the model's indices are (out of order, duplicated, out of
range, or non-integer junk), the resulting lessons must tile the source 0..N-1
with no gaps and no overlaps. This is the logic INSIGHTS claims as verified —
now it lives in the repo as a regression guard.
"""
from contracts import IngestResult
from agents import agent2_segment as a2


def _source(n_paragraphs: int) -> IngestResult:
    # Each paragraph is >= 50 chars so split_into_paragraphs keeps them separate.
    paras = [f"Paragraph number {i} explaining a distinct idea in enough words to pass the fifty character minimum."
             for i in range(n_paragraphs)]
    return IngestResult(url="http://x", accepted=True, reason="ok",
                        title="T", clean_text="\n\n".join(paras))


def _segment_with_cuts(monkeypatch, segments, n_paragraphs=10):
    """Run the real segment() with the LLM stubbed to return `segments`."""
    monkeypatch.setattr(a2.llm, "chat_json",
                        lambda system, user, temperature=None, model=None: {"segments": segments})
    return a2.segment(_source(n_paragraphs), mock=False, use_llm=True)


def _assert_perfect_tiling(curriculum, n):
    lessons = sorted(curriculum.lessons, key=lambda l: l.order)
    assert lessons, "no lessons produced"
    # first starts at 0, last ends at n-1
    assert lessons[0].start_idx == 0
    assert lessons[-1].end_idx == n - 1
    # contiguous: each lesson picks up exactly where the previous left off
    expected_start = 0
    for l in lessons:
        assert l.start_idx == expected_start, f"gap/overlap at lesson {l.order}"
        assert l.end_idx >= l.start_idx, f"inverted range at lesson {l.order}"
        expected_start = l.end_idx + 1
    assert expected_start == n, "lessons do not cover the whole source"


def test_in_order_cuts_tile(monkeypatch):
    cur = _segment_with_cuts(monkeypatch, [
        {"title": "A", "start_idx": 0},
        {"title": "B", "start_idx": 4},
        {"title": "C", "start_idx": 7},
    ], n_paragraphs=10)
    _assert_perfect_tiling(cur, 10)
    assert len(cur.lessons) == 3


def test_out_of_order_cuts_are_sorted(monkeypatch):
    cur = _segment_with_cuts(monkeypatch, [
        {"title": "B", "start_idx": 7},
        {"title": "A", "start_idx": 2},
        {"title": "C", "start_idx": 5},
    ], n_paragraphs=10)
    _assert_perfect_tiling(cur, 10)
    # first forced to 0 even though no segment claimed it
    assert cur.lessons[0].start_idx == 0


def test_duplicate_starts_are_deduped(monkeypatch):
    cur = _segment_with_cuts(monkeypatch, [
        {"title": "A", "start_idx": 0},
        {"title": "B", "start_idx": 3},
        {"title": "C", "start_idx": 3},   # duplicate — keep the first only
        {"title": "D", "start_idx": 6},
    ], n_paragraphs=10)
    _assert_perfect_tiling(cur, 10)
    starts = [l.start_idx for l in cur.lessons]
    assert len(starts) == len(set(starts)), "duplicate start survived"


def test_out_of_range_and_junk_indices(monkeypatch):
    cur = _segment_with_cuts(monkeypatch, [
        {"title": "A", "start_idx": -5},      # below range -> clamps to 0
        {"title": "B", "start_idx": 4},
        {"title": "C", "start_idx": 999},     # above range -> clamps to n-1
        {"title": "D", "start_idx": "junk"},  # non-int -> default 0 -> deduped
    ], n_paragraphs=10)
    _assert_perfect_tiling(cur, 10)


def test_single_segment_still_covers(monkeypatch):
    # A lone segment must still tile the whole source (it just won't pass the count gate).
    cur = _segment_with_cuts(monkeypatch, [{"title": "All", "start_idx": 0}], n_paragraphs=8)
    _assert_perfect_tiling(cur, 8)
    assert len(cur.lessons) == 1


def test_n_source_paragraphs_recorded(monkeypatch):
    cur = _segment_with_cuts(monkeypatch, [
        {"title": "A", "start_idx": 0},
        {"title": "B", "start_idx": 5},
    ], n_paragraphs=12)
    assert cur.n_source_paragraphs == 12
