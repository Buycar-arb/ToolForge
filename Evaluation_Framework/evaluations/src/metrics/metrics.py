"""Metrics calculation for evaluation."""

from typing import List, Dict, Any
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', '', s)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s.strip()


def exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Calculate exact match score."""
    if not prediction:
        return 0.0

    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if norm_pred == norm_gt:
            return 1.0
    return 0.0


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Calculate F1 score."""
    if not prediction:
        return 0.0

    def get_tokens(s):
        return normalize_answer(s).split()

    pred_tokens = get_tokens(prediction)
    scores = []

    for gt in ground_truths:
        gt_tokens = get_tokens(gt)

        if not pred_tokens and not gt_tokens:
            scores.append(1.0)
            continue
        if not pred_tokens or not gt_tokens:
            scores.append(0.0)
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            scores.append(0.0)
            continue

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        scores.append(f1)

    return max(scores) if scores else 0.0


def extract_search_stats(item: Dict[str, Any]) -> tuple:
    """
    Extract search statistics from messages or response.

    Args:
        item: Result item containing either messages or response

    Returns:
        Tuple of (search_count, iteration_count)
    """
    search_count = 0
    iteration_count = 0

    if 'messages' in item:
        # Function-based: extract from messages
        messages = item['messages']
        # Count tool responses (each tool response = one search)
        search_count = sum(1 for m in messages if m.get('role') == 'tool')
        # Count iterations (all assistant messages = reasoning iterations)
        iteration_count = sum(1 for m in messages if m.get('role') == 'assistant')

    elif 'response' in item:
        # Tag-based: extract from response
        response = item.get('response', '')
        # Count search tags
        search_count = len(re.findall(r'<search>.*?</search>', response, re.DOTALL))
        # For tag-based, iterations = search count + 1 (final answer generation)
        iteration_count = search_count + 1

    return search_count, iteration_count


def calculate_metrics(results: List[Dict[str, Any]], metrics_list: List[str]) -> Dict[str, float]:
    """Calculate all metrics for results."""
    metrics_results = {metric: [] for metric in metrics_list}

    for item in results:
        prediction = item.get('prediction', '')
        # Support both ground_truths (list) and gold_answer (string) fields
        ground_truths = item.get('ground_truths', [])
        if not ground_truths and 'gold_answer' in item:
            # Convert single gold_answer to list format
            gold_answer = item.get('gold_answer', '')
            if gold_answer:
                ground_truths = [gold_answer]

        for metric in metrics_list:
            if metric == 'exact_match':
                score = exact_match(prediction, ground_truths)
            elif metric == 'f1':
                score = f1_score(prediction, ground_truths)
            else:
                continue

            metrics_results[metric].append(score)

    # Calculate averages
    avg_metrics = {}
    for metric, scores in metrics_results.items():
        avg_metrics[metric] = sum(scores) / len(scores) if scores else 0.0

    # Add search statistics
    # First try to get from explicit fields, then extract from messages/response
    search_counts = []
    iteration_counts = []

    for item in results:
        # Check for explicit fields first
        if 'search_queries' in item:
            search_counts.append(len(item['search_queries']))
        elif 'messages' in item or 'response' in item:
            # Extract from messages or response
            searches, iterations = extract_search_stats(item)
            search_counts.append(searches)
        else:
            search_counts.append(0)

        if 'iterations' in item:
            iteration_counts.append(item['iterations'])
        elif 'messages' in item or 'response' in item:
            # Use extracted iteration count if not explicitly provided
            if len(search_counts) > len(iteration_counts):
                _, iterations = extract_search_stats(item)
                iteration_counts.append(iterations)
        else:
            iteration_counts.append(0)

    avg_metrics['avg_searches'] = sum(search_counts) / len(search_counts) if search_counts else 0.0
    avg_metrics['avg_iterations'] = sum(iteration_counts) / len(iteration_counts) if iteration_counts else 0.0

    return avg_metrics


def main():
    """Main function to recalculate metrics from existing results."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Recalculate metrics from evaluation results')
    parser.add_argument('result_file', help='Path to the result JSON file')
    parser.add_argument('--metrics', nargs='+', default=['exact_match', 'f1'],
                       help='Metrics to calculate (default: exact_match f1)')
    parser.add_argument('--output', help='Output file path (default: update original file)')
    parser.add_argument('--print', action='store_true', help='Print metrics to console')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.result_file}")
    with open(args.result_file, 'r') as f:
        data = json.load(f)

    # Get results list
    if 'results' in data:
        results = data['results']
    else:
        # Assume the file is a list of results
        results = data if isinstance(data, list) else [data]

    # Calculate metrics
    print(f"Calculating metrics: {args.metrics}")
    metrics = calculate_metrics(results, args.metrics)

    # Print metrics
    if args.print or not args.output:
        print("\nMetrics:")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")

    # Update or save file
    if args.output:
        output_file = args.output
    else:
        output_file = args.result_file

    if 'results' in data:
        # Update metrics in the original structure
        data['metrics'] = metrics
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nUpdated metrics in: {output_file}")
    elif args.output:
        # Save only metrics to new file
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to: {output_file}")

    return metrics


if __name__ == "__main__":
    main()