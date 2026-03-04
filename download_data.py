"""
Download all datasets and optional models for ToolForge.

Usage:
    python download_data.py              # download datasets only
    python download_data.py --with-model # also download bge-m3 embedding model (Stage 1)
"""

import argparse
import os

def download_datasets():
    from huggingface_hub import snapshot_download
    print("Downloading ToolForge datasets from Hugging Face Hub...")
    print("  Repo : https://huggingface.co/datasets/buycar/ToolForge")
    print("  Files: Stage_2/original_data/ + train_and_eval_data/\n")
    snapshot_download(
        repo_id="buycar/ToolForge",
        repo_type="dataset",
        local_dir=".",
        ignore_patterns=["*.md"],  # skip HF-side README to avoid overwriting local ones
    )
    print("\nDatasets downloaded successfully.")
    print("  Stage_2/original_data/  — HotpotQA & 2WikiMultihopQA source data")
    print("  train_and_eval_data/    — Final SFT training & evaluation datasets\n")
    print("Next step: convert Parquet to JSONL before running Stage 2:")
    print("  cd Stage_2/original_data/HotpotQA && python parquet_to_jsonl.py")
    print("  cd Stage_2/original_data/2WikiMultihopQA && python parquet_to_jsonl.py")


def download_bge_model():
    from huggingface_hub import snapshot_download
    model_dir = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH", "./models/bge-m3")
    print(f"Downloading bge-m3 embedding model to {model_dir} ...")
    print("  (This is only required for Stage 1: Tool Variant Generation)\n")
    snapshot_download(
        repo_id="BAAI/bge-m3",
        local_dir=model_dir,
    )
    print(f"\nbge-m3 model downloaded to: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ToolForge datasets and models.")
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Also download the bge-m3 embedding model (needed for Stage 1 only)",
    )
    args = parser.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface_hub")
        raise SystemExit(1)

    download_datasets()

    if args.with_model:
        download_bge_model()
