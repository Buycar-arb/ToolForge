"""Small helpers for the JSONL files that flow between pipeline stages."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

PathLike = str | Path


def read(path: PathLike, *, skip_invalid: bool = True) -> Iterator[dict[str, Any]]:
    """Yield each record of a JSONL file.

    Malformed lines are skipped with a warning unless ``skip_invalid`` is False.
    """
    file = Path(path)
    with file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if not skip_invalid:
                    raise ValueError(f"{file}:{line_number}: {exc}") from exc
                print(f"[jsonl] skipping malformed line {file}:{line_number} ({exc})")


def read_all(path: PathLike, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a JSONL file into a list, optionally stopping after ``limit`` records."""
    records: list[dict[str, Any]] = []
    for record in read(path):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def write(path: PathLike, records: list[dict[str, Any]] | list[Any]) -> Path:
    """Overwrite ``path`` with ``records``.  Parent directories are created."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return file


def append(path: PathLike, record: Any) -> Path:
    """Append one record, flushing immediately so long runs stay crash-safe."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
    return file


def touch(path: PathLike) -> Path:
    """Create an empty file (and its parents) if it does not exist yet."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch(exist_ok=True)
    return file


def count(path: PathLike) -> int:
    """Number of non-empty lines, or 0 when the file is missing."""
    file = Path(path)
    if not file.is_file():
        return 0
    with file.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def describe(path: PathLike) -> str:
    """A short markdown summary of a JSONL file, for the Web UI."""
    file = Path(path)
    if not file.is_file():
        return f"`{file}` — **not found**"
    size_kb = file.stat().st_size / 1024
    modified = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"**{file.name}** · {count(file)} records · {size_kb:,.1f} KB · modified {modified}\n\n"
        f"`{file}`"
    )


def record_at(path: PathLike, index: int) -> str:
    """Pretty-printed JSON of the 1-based ``index``-th record."""
    file = Path(path)
    if not file.is_file():
        return f"// file not found: {file}"
    total = count(file)
    if total == 0:
        return "// file is empty"
    if index < 1 or index > total:
        return f"// index out of range (1-{total})"
    for position, record in enumerate(read(file), 1):
        if position == index:
            return json.dumps(record, ensure_ascii=False, indent=2)
    return "// index out of range"


def list_files(directory: PathLike, suffix: str = ".jsonl") -> list[str]:
    """Sorted file names with ``suffix`` inside ``directory`` (empty if missing)."""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.suffix == suffix)
