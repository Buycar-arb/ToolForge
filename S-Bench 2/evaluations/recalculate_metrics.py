#!/usr/bin/env python3
"""Batch recalculate metrics for evaluation results."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from src.metrics.metrics import calculate_metrics


def find_result_files(directory: str) -> List[Path]:
    """Find all result JSON files in directory."""
    result_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('_results.json'):
                result_files.append(Path(root) / file)

    return sorted(result_files)


def process_file(file_path: Path, metrics_list: List[str], update: bool = True) -> Dict[str, Any]:
    """Process a single result file."""
    print(f"\nProcessing: {file_path}")

    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Get results
    if 'results' not in data:
        print(f"  Warning: No 'results' field found in {file_path}")
        return {}

    results = data['results']
    if not results:
        print(f"  Warning: Empty results in {file_path}")
        return {}

    # Calculate metrics
    metrics = calculate_metrics(results, metrics_list)

    # Print metrics
    print(f"  Dataset: {data.get('dataset', 'unknown')}")
    print(f"  Samples: {len(results)}")
    for metric, value in metrics.items():
        print(f"  {metric:20s}: {value:.4f}")

    # Update file if requested
    if update:
        old_metrics = data.get('metrics', {})
        data['metrics'] = metrics

        # Save updated file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Report changes
        if old_metrics:
            print("  Changes:")
            for metric, new_value in metrics.items():
                old_value = old_metrics.get(metric, 0.0)
                if abs(new_value - old_value) > 0.0001:
                    print(f"    {metric}: {old_value:.4f} -> {new_value:.4f}")

    return metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Batch recalculate metrics for evaluation results')
    parser.add_argument('path', nargs='?', default='./evaluations/results',
                       help='Path to results directory or file (default: ./evaluations/results)')
    parser.add_argument('--metrics', nargs='+', default=['exact_match', 'f1'],
                       help='Metrics to calculate (default: exact_match f1)')
    parser.add_argument('--no-update', action='store_true',
                       help='Do not update files, only print metrics')
    parser.add_argument('--pattern', default='*_results.json',
                       help='File pattern to match (default: *_results.json)')

    args = parser.parse_args()

    path = Path(args.path)
    update = not args.no_update

    if path.is_file():
        # Process single file
        process_file(path, args.metrics, update)
    elif path.is_dir():
        # Process directory
        result_files = find_result_files(path)

        if not result_files:
            print(f"No result files found in {path}")
            return

        print(f"Found {len(result_files)} result file(s)")
        print("=" * 60)

        all_metrics = {}
        for file_path in result_files:
            metrics = process_file(file_path, args.metrics, update)
            if metrics:
                all_metrics[str(file_path)] = metrics

        # Summary
        if all_metrics:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)

            # Calculate average metrics across all files
            avg_metrics = {}
            for metric in args.metrics:
                values = [m[metric] for m in all_metrics.values() if metric in m]
                if values:
                    avg_metrics[metric] = sum(values) / len(values)

            print(f"\nAverage across {len(all_metrics)} file(s):")
            for metric, value in avg_metrics.items():
                print(f"  {metric:20s}: {value:.4f}")
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()