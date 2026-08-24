"""Convert between JSONL and Parquet.

Parquet is how the datasets ship (it is compact and typed); JSONL is what every
stage of the pipeline reads and writes.  Nested fields such as HotpotQA's
``context`` and ``supporting_facts`` cannot live in a Parquet column as-is, so
they are stored as JSON strings and parsed back on the way out.

The original release shipped six byte-identical copies of this logic in six
directories.  This is that logic, once.

::

    toolforge convert to-jsonl  data/source_qa/HotpotQA
    toolforge convert to-parquet output/data
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

PathLike = str | Path
Reporter = Callable[[str], None]


def _looks_like_json(value: Any) -> bool:
    """Whether a cell holds a JSON object or array serialised as a string."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    return (text.startswith("[") and text.endswith("]")) or (
        text.startswith("{") and text.endswith("}")
    )


def _first_value(frame: Any, column: str) -> Any:
    """The first non-null value in a column, or ``None`` if there is none.

    Column *values* decide how a column is treated, never its dtype: pandas 3
    gives string columns a dedicated ``str`` dtype instead of ``object``, so a
    dtype check silently skips exactly the columns that need converting.
    """
    present = frame[column].dropna()
    return None if present.empty else present.iloc[0]


def to_jsonl(directory: PathLike, output_dir: PathLike | None = None, *, report: Reporter = print) -> list[Path]:
    """Convert every ``*.parquet`` in ``directory`` to ``*.jsonl``.

    Columns whose first value looks like serialised JSON are parsed back into
    real lists and dicts, so the output matches the pipeline's input schema.
    Returns the files written.
    """
    import pandas as pd

    source = Path(directory)
    target = Path(output_dir) if output_dir else source
    target.mkdir(parents=True, exist_ok=True)

    files = sorted(source.glob("*.parquet"))
    if not files:
        report(f"No .parquet files in {source}")
        return []

    written: list[Path] = []
    for parquet_file in files:
        try:
            frame = pd.read_parquet(parquet_file, engine="pyarrow")
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            report(f"  ✗ {parquet_file.name}: {exc}")
            continue

        if frame.empty:
            report(f"  · {parquet_file.name} is empty, skipped")
            continue

        for column in frame.columns:
            if not _looks_like_json(_first_value(frame, column)):
                continue
            frame[column] = frame[column].apply(
                lambda cell: json.loads(cell) if _looks_like_json(cell) else cell
            )

        destination = target / f"{parquet_file.stem}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for record in frame.to_dict("records"):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        written.append(destination)
        report(f"  ✓ {parquet_file.name} → {destination.name}  ({len(frame)} records)")
    return written


def to_parquet(directory: PathLike, output_dir: PathLike | None = None, *, report: Reporter = print) -> list[Path]:
    """Convert every ``*.jsonl`` in ``directory`` to ``*.parquet``.

    Nested values are serialised to JSON strings, which :func:`to_jsonl` reverses.
    Returns the files written.
    """
    import pandas as pd

    source = Path(directory)
    target = Path(output_dir) if output_dir else source
    target.mkdir(parents=True, exist_ok=True)

    files = sorted(source.glob("*.jsonl"))
    if not files:
        report(f"No .jsonl files in {source}")
        return []

    written: list[Path] = []
    for jsonl_file in files:
        records: list[dict[str, Any]] = []
        with jsonl_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    report(f"  · {jsonl_file.name}:{line_number} skipped ({exc})")

        if not records:
            report(f"  · {jsonl_file.name} has no valid records, skipped")
            continue

        frame = pd.DataFrame(records)
        for column in frame.columns:
            if not isinstance(_first_value(frame, column), (list, dict)):
                continue
            frame[column] = frame[column].apply(
                lambda cell: json.dumps(cell, ensure_ascii=False)
                if isinstance(cell, (list, dict)) else cell
            )

        destination = target / f"{jsonl_file.stem}.parquet"
        try:
            frame.to_parquet(destination, engine="pyarrow", compression="snappy", index=False)
        except Exception as exc:  # noqa: BLE001
            report(f"  ✗ {jsonl_file.name}: {exc}")
            continue

        written.append(destination)
        report(f"  ✓ {jsonl_file.name} → {destination.name}  ({len(frame)} records)")
    return written
