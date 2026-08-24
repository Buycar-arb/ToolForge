"""The JSONL helpers every stage relies on."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toolforge import jsonl  # noqa: E402


def test_write_read_count_and_append(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "data.jsonl"
    jsonl.write(target, [{"n": 1}, {"n": 2}])
    assert jsonl.count(target) == 2
    jsonl.append(target, {"n": 3})
    assert [r["n"] for r in jsonl.read(target)] == [1, 2, 3]


def test_malformed_lines_are_skipped(tmp_path: Path) -> None:
    target = tmp_path / "data.jsonl"
    target.write_text('{"n": 1}\nnot json\n\n{"n": 2}\n', encoding="utf-8")
    assert [r["n"] for r in jsonl.read(target)] == [1, 2]
    assert jsonl.count(target) == 3  # counts lines, not valid records


def test_record_at_is_one_based_and_bounded(tmp_path: Path) -> None:
    target = tmp_path / "data.jsonl"
    jsonl.write(target, [{"n": 1}, {"n": 2}])
    assert '"n": 1' in jsonl.record_at(target, 1)
    assert "out of range" in jsonl.record_at(target, 3)
    assert "out of range" in jsonl.record_at(target, 0)


def test_missing_file_is_reported_not_raised(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    assert jsonl.count(missing) == 0
    assert "not found" in jsonl.describe(missing)
    assert "not found" in jsonl.record_at(missing, 1)


def test_unicode_survives_a_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "data.jsonl"
    jsonl.write(target, [{"q": "工具选择：人物信息搜索"}])
    assert jsonl.read_all(target)[0]["q"] == "工具选择：人物信息搜索"
