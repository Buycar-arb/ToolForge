#!/usr/bin/env python3
"""Fetch the ToolForge datasets (and, optionally, the stage 1 embedding model).

    python download_data.py                    # everything
    python download_data.py --only HotpotQA    # just one corpus
    python download_data.py --with-model       # also fetch bge-m3 for stage 1

What you get, under ``data/``:

    data/source_qa/   HotpotQA & 2WikiMultihopQA — the raw input to stage 2

That is the starting point of the whole pipeline. From it:

    toolforge convert to-jsonl data/source_qa/HotpotQA
    toolforge label   data/source_qa/HotpotQA/bridge_hp.jsonl  data/labelled/output.jsonl
    toolforge generate data/labelled/output.jsonl --case case_C1 --target 100

If a ``labelled/`` folder is present in the dataset it is picked up too, which
lets you skip stage 2.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

#: Public dataset: the raw multi-hop QA that stage 2 consumes.
REPO_ID = os.getenv("TOOLFORGE_DATASET_REPO", "buycar/ToolForge")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

#: Where the hub repo puts things -> where this project wants them.  Both the
#: current and the pre-2.0 layouts are handled, so an older dataset revision
#: still lands in the right place.
RELOCATIONS = {
    "source_qa": DATA_DIR / "source_qa",
    "labelled": DATA_DIR / "labelled",
    "Stage_2/original_data": DATA_DIR / "source_qa",
    "Stage_2/label_data": DATA_DIR / "labelled",
}


def _merge(source: Path, destination: Path) -> None:
    """Move ``source`` into ``destination``, merging rather than replacing."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.is_dir() and item.is_dir():
            _merge(item, target)
            continue
        if target.exists():
            target.unlink()
        shutil.move(str(item), str(target))
    shutil.rmtree(source, ignore_errors=True)


def download_datasets(only: str | None = None) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    staging = DATA_DIR / ".hf"
    staging.mkdir(parents=True, exist_ok=True)

    patterns = [f"*{only}*"] if only else None
    print(f"Downloading from https://huggingface.co/datasets/{REPO_ID}")
    if only:
        print(f"  (only paths matching {only!r})")
    print()
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(staging),
            allow_patterns=patterns,
            ignore_patterns=["*.md"],  # keep this repo's docs, not the hub's
        )
    except (RepositoryNotFoundError, GatedRepoError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        print(
            f"\nCould not read {REPO_ID}: {type(exc).__name__}.\n"
            "If it is private or gated, log in first:\n"
            "    pip install -U huggingface_hub && hf auth login\n"
            "To pull from a different dataset, set TOOLFORGE_DATASET_REPO.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    moved = False
    for relative, destination in RELOCATIONS.items():
        arrived = staging / relative
        if arrived.is_dir():
            _merge(arrived, destination)
            print(f"  {relative}  ->  {destination.relative_to(ROOT)}")
            moved = True

    # Anything the dataset adds later still ends up somewhere sensible.  Parent
    # directories emptied by the relocations above are dropped, not moved.
    for leftover in list(staging.iterdir()):
        if leftover.name.startswith("."):
            continue
        if leftover.is_dir() and not any(
            item for item in leftover.rglob("*") if item.is_file()
        ):
            shutil.rmtree(leftover, ignore_errors=True)
            continue
        target = DATA_DIR / leftover.name
        if leftover.is_dir():
            _merge(leftover, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(leftover), str(target))
        print(f"  {leftover.name}  ->  data/{leftover.name}")
        moved = True

    shutil.rmtree(staging, ignore_errors=True)
    if not moved:
        print("  (the dataset was empty)")

    print("\nNext steps\n")
    if (DATA_DIR / "labelled").is_dir():
        print("  Labels are included — you can skip stage 2 and go straight to stage 3:")
        print("    toolforge generate data/labelled/output.jsonl --case case_C1 --target 10\n")
    elif (DATA_DIR / "source_qa").is_dir():
        print("  1. convert the Parquet files to JSONL:")
        print("       toolforge convert to-jsonl data/source_qa/HotpotQA\n")
        print("  2. stage 2 — label a small slice first:")
        print("       toolforge label data/source_qa/HotpotQA/bridge_hp.jsonl \\")
        print("                       data/labelled/output.jsonl --limit 20\n")
        print("  3. stages 3+4 — generate and validate:")
        print("       toolforge generate data/labelled/output.jsonl --case case_C1 --target 5\n")


def download_embedding_model() -> None:
    from huggingface_hub import snapshot_download

    destination = os.getenv("EMBEDDING_MODEL_PATH") or str(ROOT / "models" / "bge-m3")
    print(f"\nDownloading BAAI/bge-m3 to {destination}")
    print("  (only stage 1 — tool variant generation — needs this)\n")
    snapshot_download(repo_id="BAAI/bge-m3", local_dir=destination)
    print(f"\nDone. Set EMBEDDING_MODEL_PATH={destination} in your .env")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", metavar="NAME",
                        help="fetch only paths matching NAME, e.g. HotpotQA or inference_wiki")
    parser.add_argument("--with-model", action="store_true",
                        help="also download the bge-m3 embedding model used by stage 1")
    args = parser.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is not installed.  Run:  pip install huggingface_hub", file=sys.stderr)
        return 1

    download_datasets(args.only)
    if args.with_model:
        download_embedding_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
