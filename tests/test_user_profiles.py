"""Deterministic tests for the shared users.md parser."""
from user_profiles import parse_users, filter_users


_USERS_MD = """# Users

Some preamble that must be ignored.

## Mike
- role: Senior ML Engineer
- level: expert
- language: English
- focus: tradeoffs
- some free note: kept only in raw

## Dana
- role: Student
- level: beginner
- languages: Russian
"""


def test_parse_basic_fields(tmp_path):
    p = tmp_path / "users.md"
    p.write_text(_USERS_MD, encoding="utf-8")
    users = parse_users(p)
    assert [u.name for u in users] == ["Mike", "Dana"]
    mike = users[0]
    assert mike.role == "Senior ML Engineer"
    assert mike.level == "expert"
    assert mike.focus == "tradeoffs"


def test_language_alias_maps_to_languages(tmp_path):
    # `language` (singular) and `languages` (plural) both populate `languages`.
    p = tmp_path / "users.md"
    p.write_text(_USERS_MD, encoding="utf-8")
    users = {u.name: u for u in parse_users(p)}
    assert users["Mike"].languages == "English"   # from `- language:`
    assert users["Dana"].languages == "Russian"   # from `- languages:`


def test_raw_holds_full_block(tmp_path):
    p = tmp_path / "users.md"
    p.write_text(_USERS_MD, encoding="utf-8")
    mike = parse_users(p)[0]
    # free-form lines with no structured field still reach the model via raw
    assert "some free note" in mike.raw
    assert mike.raw.startswith("Mike")


def test_filter_users_by_name(tmp_path):
    p = tmp_path / "users.md"
    p.write_text(_USERS_MD, encoding="utf-8")
    users = parse_users(p)
    only = filter_users(users, "mike")   # case-insensitive
    assert len(only) == 1 and only[0].name == "Mike"


def test_filter_users_none_returns_all(tmp_path):
    p = tmp_path / "users.md"
    p.write_text(_USERS_MD, encoding="utf-8")
    users = parse_users(p)
    assert filter_users(users, None) == users


def test_filter_users_unknown_raises(tmp_path):
    p = tmp_path / "users.md"
    p.write_text(_USERS_MD, encoding="utf-8")
    users = parse_users(p)
    try:
        filter_users(users, "Nobody")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not found" in str(e)
