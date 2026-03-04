#!/usr/bin/env python3
"""Download all required datasets for evaluation."""

import os
import sys
import argparse
from pathlib import Path
import json
import yaml
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import datasets
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install the requirements first:")
    print("  pip install datasets tqdm pyyaml")
    sys.exit(1)


class DatasetDownloader:
    """Download and cache datasets for evaluation."""

    def __init__(self, config_path: str = None, cache_dir: str = None):
        """Initialize downloader with configuration."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config',
                'datasets.yaml'
            )

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data'
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # HuggingFace cache directory
        self.hf_cache_dir = self.cache_dir / 'huggingface'
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset configuration."""
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
        return self.config['datasets'][dataset_name]

    def estimate_dataset_size(self, source: str, subset: str) -> Dict[str, Any]:
        """Estimate dataset size before downloading."""
        try:
            # Get dataset info without downloading
            dataset_info = datasets.load_dataset_builder(source, subset)

            # Get download size and dataset size
            download_size = dataset_info.info.download_size
            dataset_size = dataset_info.info.dataset_size
            num_examples = dataset_info.info.splits.get('test',
                          dataset_info.info.splits.get('train', {})).num_examples

            return {
                'download_size_mb': download_size / (1024 * 1024) if download_size else 0,
                'dataset_size_mb': dataset_size / (1024 * 1024) if dataset_size else 0,
                'num_examples': num_examples or 0
            }
        except Exception as e:
            print(f"Warning: Could not get size info for {subset}: {e}")
            return {
                'download_size_mb': 0,
                'dataset_size_mb': 0,
                'num_examples': 0
            }

    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """Download a single dataset."""
        info = self.get_dataset_info(dataset_name)
        source = info['source']
        subset = info['subset']
        split = info.get('split', 'test')  # Use split from config, default to 'test'

        # Create dataset-specific directory
        dataset_dir = self.cache_dir / subset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Check if already cached
        cache_file = dataset_dir / "data.jsonl"
        if cache_file.exists() and not force:
            print(f"✓ {dataset_name} already cached at {cache_file}")
            return True

        print(f"\nDownloading {dataset_name} ({subset}) from {source} [split: {split}]...")

        try:
            # Download dataset
            dataset = load_dataset(
                source,
                subset,
                split=split,  # Use the split from configuration
                cache_dir=str(self.hf_cache_dir)
            )

            # Process and save
            processed_data = []
            test_size = info.get('test_size', -1)

            for idx, item in enumerate(tqdm(dataset, desc=f"Processing {subset}")):
                # Handle different dataset formats
                question = item.get('question', item.get('query', ''))

                # Get answers - handle different formats
                answers = item.get('golden_answers', item.get('answer', []))
                if isinstance(answers, str):
                    answers = [answers]

                processed_item = {
                    'id': f"{subset}_{idx}",
                    'question': question,
                    'answers': answers,
                    'metadata': {
                        'dataset': subset,
                        'index': idx
                    }
                }
                processed_data.append(processed_item)

                # Apply test size limit
                if test_size > 0 and len(processed_data) >= test_size:
                    break

            # Save to cache in JSONL format (one JSON object per line)
            with open(cache_file, 'w') as f:
                for item in processed_data:
                    f.write(json.dumps(item) + '\n')

            print(f"✓ {dataset_name}: {len(processed_data)} examples saved to {cache_file}")
            return True

        except Exception as e:
            print(f"✗ Error downloading {dataset_name}: {e}")
            return False

    def download_all(self, datasets_list: List[str] = None, force: bool = False) -> Dict[str, bool]:
        """Download all configured datasets."""
        if datasets_list is None:
            datasets_list = self.config.get('active_datasets', [])

        results = {}
        total_download_size = 0
        total_dataset_size = 0

        # First, estimate sizes
        print("\n" + "="*60)
        print("DATASET SIZE ESTIMATION")
        print("="*60)

        size_info = {}
        for dataset_name in datasets_list:
            info = self.get_dataset_info(dataset_name)
            size = self.estimate_dataset_size(info['source'], info['subset'])
            size_info[dataset_name] = size

            total_download_size += size['download_size_mb']
            total_dataset_size += size['dataset_size_mb']

            print(f"\n{dataset_name} ({info['subset']}):")
            print(f"  Download size: {size['download_size_mb']:.2f} MB")
            print(f"  Dataset size:  {size['dataset_size_mb']:.2f} MB")
            print(f"  Num examples:  {size['num_examples']:,}")

        print(f"\n{'='*60}")
        print(f"TOTAL ESTIMATED SIZES:")
        print(f"  Total download: {total_download_size:.2f} MB ({total_download_size/1024:.2f} GB)")
        print(f"  Total dataset:  {total_dataset_size:.2f} MB ({total_dataset_size/1024:.2f} GB)")
        print(f"  Cache location: {self.cache_dir}")
        print(f"{'='*60}\n")

        # Ask for confirmation
        if not force:
            response = input("Do you want to proceed with downloading? (y/n): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                return results

        # Download datasets
        print("\n" + "="*60)
        print("DOWNLOADING DATASETS")
        print("="*60)

        for dataset_name in datasets_list:
            results[dataset_name] = self.download_dataset(dataset_name, force)

        # Summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)

        successful = sum(1 for v in results.values() if v)
        failed = len(results) - successful

        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")

        if failed > 0:
            print("\nFailed datasets:")
            for name, success in results.items():
                if not success:
                    print(f"  - {name}")

        # Check actual cache size
        total_cache_size = 0
        for dataset_dir in self.cache_dir.iterdir():
            if dataset_dir.is_dir():
                for file in dataset_dir.glob("*.jsonl"):
                    total_cache_size += file.stat().st_size

        print(f"\nActual cache size: {total_cache_size/(1024*1024):.2f} MB")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download datasets for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all active datasets
  python download_datasets.py

  # Download specific datasets
  python download_datasets.py --datasets nq popqa

  # Force re-download even if cached
  python download_datasets.py --force

  # Use custom cache directory
  python download_datasets.py --cache-dir ./my_cache
        """
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to download (default: all active datasets)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to datasets.yaml config file'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Directory to cache datasets'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if cached'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets without downloading'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = DatasetDownloader(
        config_path=args.config,
        cache_dir=args.cache_dir
    )

    # List datasets if requested
    if args.list:
        print("\nAvailable datasets:")
        print("-" * 40)
        for name, info in downloader.config['datasets'].items():
            active = name in downloader.config.get('active_datasets', [])
            status = "[ACTIVE]" if active else ""
            print(f"  {name:15} {info['subset']:20} {status}")
        print("\nActive datasets:", ', '.join(downloader.config.get('active_datasets', [])))
        return

    # Download datasets
    results = downloader.download_all(
        datasets_list=args.datasets,
        force=args.force
    )

    # Exit with error if any downloads failed
    if any(not v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

#   # Download all datasets with size estimation
#   python evaluations/scripts/download_datasets.py

#   # Download specific datasets
#   python evaluations/scripts/download_datasets.py --datasets nq popqa bamboogle

#   # Force re-download
#   python evaluations/scripts/download_datasets.py --force