"""Parquet <-> JSONL must survive a round trip with nested fields intact."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toolforge import jsonl  # noqa: E402

pytest.importorskip("pandas", reason="the data extra is not installed")
pytest.importorskip("pyarrow", reason="the data extra is not installed")

from toolforge.convert import to_jsonl, to_parquet  # noqa: E402

RECORDS = [
    {
        "_id": "a1",
        "question": "Were they of the same nationality?",
        "answer": "yes",
        "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
        "context": [["Scott Derrickson", ["An American director."]]],
    },
    {
        "_id": "a2",
        "question": "第二个问题？",
        "answer": "no",
        "supporting_facts": [["李四", 1]],
        "context": [["李四", ["一段中文段落。"]]],
    },
]


def test_round_trip_preserves_nested_fields(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    jsonl.write(source / "data.jsonl", RECORDS)

    assert to_parquet(source, report=lambda _m: None)
    (source / "data.jsonl").unlink()

    assert to_jsonl(source, report=lambda _m: None)
    assert jsonl.read_all(source / "data.jsonl") == RECORDS


def test_writing_to_a_separate_directory(tmp_path: Path) -> None:
    source, target = tmp_path / "in", tmp_path / "out"
    source.mkdir()
    jsonl.write(source / "data.jsonl", RECORDS)

    to_parquet(source, target, report=lambda _m: None)
    assert (target / "data.parquet").is_file()
    assert not (source / "data.parquet").exists()


def test_an_empty_directory_is_not_an_error(tmp_path: Path) -> None:
    messages: list[str] = []
    assert to_jsonl(tmp_path, report=messages.append) == []
    assert to_parquet(tmp_path, report=messages.append) == []
    assert len(messages) == 2
