"""Deterministic tests for llm._parse_json (no network).

INSIGHTS changelog #7 says the JSON parser was "Verified (plain / fenced / prose /
garbage) on isolated tests" — but those tests never made it into the repo. They
matter: OpenRouter free models routinely ignore strict JSON mode and wrap their
output in ```json fences or prose, so this tolerant parser is what keeps the whole
pipeline working on them. Pin it here.
"""
import llm


def test_plain_object():
    assert llm._parse_json('{"passed": true, "issues": []}') == {"passed": True, "issues": []}


def test_json_fenced():
    text = '```json\n{"a": 1, "b": "x"}\n```'
    assert llm._parse_json(text) == {"a": 1, "b": "x"}


def test_bare_fence_without_lang():
    text = '```\n{"a": 1}\n```'
    assert llm._parse_json(text) == {"a": 1}


def test_prose_around_json():
    text = 'Sure! Here is the result:\n{"score": 0.9, "verdict": "pass"}\nHope that helps.'
    assert llm._parse_json(text) == {"score": 0.9, "verdict": "pass"}


def test_array_payload():
    # json_object mode can't return a bare array, but lenient parsing of one still works.
    assert llm._parse_json('[{"start_idx": 0}, {"start_idx": 5}]') == [{"start_idx": 0}, {"start_idx": 5}]


def test_garbage_returns_empty_dict():
    assert llm._parse_json("not json at all, just text") == {}


def test_empty_returns_empty_dict():
    assert llm._parse_json("") == {}
    assert llm._parse_json(None) == {}


def test_nested_object_in_prose():
    text = 'The verdict is: {"issues": [{"severity": "high", "lesson": 2}]} done.'
    assert llm._parse_json(text) == {"issues": [{"severity": "high", "lesson": 2}]}
