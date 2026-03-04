#!/usr/bin/env python3
"""Main evaluation script."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from src.models.closed_source import OpenAIModel, DeepSeekModel, ClaudeModel, GrokModel
from src.models.open_source import VLLMModel
from src.datasets.dataset_loader import BenchmarkDataset
from src.datasets.local_dataset import LocalDataset
from src.search.search_interface import SearchEngine
from src.search.tag_search import TagBasedSearch
from src.search.function_search import FunctionSearchHandler
from src.inference.tag_based_inference import TagBasedInference
from src.inference.function_inference import FunctionInference
from src.metrics.metrics import calculate_metrics
from src.utils.thread_manager import MultiThreadEvaluator, BatchProcessor


def load_config(config_dir: str) -> Dict[str, Any]:
    """Load all configuration files."""
    configs = {}
    config_files = ['models.yaml', 'datasets.yaml', 'search_engines.yaml', 'prompts.yaml']

    for file in config_files:
        path = os.path.join(config_dir, file)
        with open(path, 'r') as f:
            configs[file.replace('.yaml', '')] = yaml.safe_load(f)

    return configs


def initialize_model(model_name: str, config: Dict) -> Any:
    """Initialize model based on configuration."""
    model_config = config['models']['models'][model_name]

    if model_config['type'] == 'closed_source':
        if 'gpt' in model_name:
            return OpenAIModel(model_config)
        elif 'deepseek' in model_name:
            return DeepSeekModel(model_config)
        elif 'claude' in model_name:
            return ClaudeModel(model_config)
        elif 'grok' in model_name:
            return GrokModel(model_config)
        else:
            # Default to OpenAI model for unknown closed source models
            return OpenAIModel(model_config)
    else:
        return VLLMModel(model_config)


def initialize_search(config: Dict, search_method: str) -> Any:
    """Initialize search components."""
    search_config = config['search_engines']['search_engine']

    if search_method == 'tag':
        search_engine = SearchEngine(search_config)
        return TagBasedSearch(search_engine, search_config)
    elif search_method == 'function':
        return FunctionSearchHandler(search_config)
    else:
        raise ValueError(f"Unknown search method: {search_method}")


def create_model_factory(model_name: str, configs: Dict) -> callable:
    """Create a factory function for model instances."""
    def model_factory():
        return initialize_model(model_name, configs)
    return model_factory


def create_search_factory(configs: Dict, search_method: str) -> callable:
    """Create a factory function for search handler instances."""
    def search_factory():
        return initialize_search(configs, search_method)
    return search_factory


def get_prompt_config(config: Dict, model_name: str, search_method: str) -> Dict:
    """Get appropriate prompt configuration."""
    prompts = config['prompts']

    if search_method == 'tag':
        return prompts['prompts']['tag_based']
    elif search_method == 'function':
        # Determine model type to select correct prompt variant
        model_config = config['models']['models'][model_name]
        model_type = model_config.get('type', 'closed_source')

        if model_type == 'closed_source':
            return prompts['prompts']['function_based_closed_source']
        else:
            return prompts['prompts']['function_based_open_source']
    else:
        raise ValueError(f"Unknown search method: {search_method}")


def evaluate_single(
    question: str,
    model: Any,
    search_handler: Any,
    prompt_config: Dict,
    search_method: str
) -> Dict[str, Any]:
    """Evaluate a single question."""
    try:
        if search_method == 'tag':
            inference = TagBasedInference(model, search_handler, prompt_config)
        elif search_method == 'function':
            inference = FunctionInference(model, search_handler, prompt_config)
        else:
            raise ValueError(f"Unknown search method: {search_method}")

        result = inference.run(question)
        return result
    except Exception as e:
        print(f"Error evaluating question: {e}")
        return {
            'answer': None,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument('--model', type=str, help='Model to evaluate')
    parser.add_argument('--method', type=str, choices=['tag', 'function'], help='Search method')
    parser.add_argument('--datasets', nargs='+', help='Datasets to evaluate')
    parser.add_argument('--config_dir', default='./evaluations/config', help='Config directory')
    parser.add_argument('--output_dir', default='./evaluations/results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint_every', type=int, default=None, help='Checkpoint frequency (overrides config)')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads')
    parser.add_argument('--use_multithreading', action='store_true', help='Enable multi-threading')
    parser.add_argument('--disable_multithreading', action='store_true', help='Disable multi-threading (default)')

    args = parser.parse_args()

    # Load configurations
    print("Loading configurations...")
    configs = load_config(args.config_dir)

    # Get model and datasets from config if not specified
    model_name = args.model or configs['models']['active_model']
    datasets_to_eval = args.datasets or configs['datasets']['active_datasets']
    search_method = args.method or configs['search_engines']['search_engine']['search_method']

    # Get checkpoint_every from config or command line argument
    checkpoint_every = args.checkpoint_every
    if checkpoint_every is None:
        # Try to get from evaluation config in datasets.yaml
        checkpoint_every = configs['datasets'].get('evaluation', {}).get('checkpoint_every', 100)

    # Determine if multi-threading should be used
    # Priority: command line args > config file > default
    config_multithreading = configs['datasets'].get('evaluation', {}).get('use_multithreading', False)
    config_max_workers = configs['datasets'].get('evaluation', {}).get('max_workers', 4)
    
    if args.use_multithreading:
        use_multithreading = True
    elif args.disable_multithreading:
        use_multithreading = False
    else:
        use_multithreading = config_multithreading
    
    max_workers = args.max_workers if args.max_workers != 4 else config_max_workers

    print(f"Checkpoint every: {checkpoint_every} examples")
    print(f"Multi-threading: {'Enabled' if use_multithreading else 'Disabled'}")
    if use_multithreading:
        print(f"Max workers: {max_workers}")

    print(f"Model: {model_name}")
    print(f"Method: {search_method}")
    print(f"Datasets: {datasets_to_eval}")

    # Initialize components
    print("Initializing components...")
    if use_multithreading:
        # For multi-threading, we'll create factory functions
        model_factory = create_model_factory(model_name, configs)
        search_factory = create_search_factory(configs, search_method)
        
        # Create a single instance for testing
        model = model_factory()
        search_handler = search_factory()
    else:
        # Single-threaded mode
        model = initialize_model(model_name, configs)
        search_handler = initialize_search(configs, search_method)

    # Get prompt configuration
    prompt_config = get_prompt_config(configs, model_name, search_method)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{model_name}_{search_method}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({
            'model': model_name,
            'method': search_method,
            'datasets': datasets_to_eval,
            'timestamp': timestamp
        }, f, indent=2)

    all_results = {}

    # Evaluate each dataset
    for dataset_name in datasets_to_eval:
        print(f"\nEvaluating {dataset_name}...")

        # Load dataset
        dataset_config = configs['datasets']['datasets'][dataset_name]
        
        # Check if this is a local dataset
        if dataset_config.get('source') == 'local':
            dataset = LocalDataset(dataset_config)
        else:
            dataset = BenchmarkDataset(dataset_config)
        
        data = dataset.load()

        print(f"Loaded {len(data)} examples")

        # Evaluate using multi-threading or single-threading
        if use_multithreading:
            print(f"Using multi-threading with {max_workers} workers...")
            
            # Create multi-threaded evaluator
            evaluator = MultiThreadEvaluator(
                model_factory=model_factory,
                search_factory=search_factory,
                prompt_config=prompt_config,
                search_method=search_method,
                max_workers=max_workers,
                checkpoint_every=checkpoint_every
            )
            
            # Evaluate dataset
            results = evaluator.evaluate_dataset(
                dataset_name=dataset_name,
                data=data,
                output_dir=run_dir,
                resume_from_checkpoint=True
            )
            
        else:
            print("Using single-threading...")
    
            # Single-threaded evaluation (original logic)
            results = []
            checkpoint_file = os.path.join(run_dir, f"{dataset_name}_checkpoint.jsonl")

            # Load checkpoint if exists
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    for line in f:
                        results.append(json.loads(line))
                print(f"Resumed from checkpoint: {len(results)} completed")

            for i in tqdm(range(len(results), len(data)), desc=f"Evaluating {dataset_name}"):
                item = data[i]

                # Evaluate
                result = evaluate_single(
                    item['question'],
                    model,
                    search_handler,
                    prompt_config,
                    search_method
                )

                # Add metadata
                # Simplified result format
                simplified_result = {
                    'id': item['id'],
                    'question': item['question'],
                    'gold_answer': item['answers'][0] if item['answers'] else '',  # Use first answer as gold
                    'prediction': result.get('answer', '')
                }

                # Add method-specific data
                if search_method == 'tag':
                    simplified_result['response'] = result.get('response', '')
                elif search_method == 'function':
                    simplified_result['messages'] = result.get('messages', [])

                result = simplified_result

                results.append(result)

                # Save checkpoint
                if (i + 1) % checkpoint_every == 0:
                    with open(checkpoint_file, 'a') as f:
                        for r in results[-checkpoint_every:]:
                            f.write(json.dumps(r) + '\n')
                    print(f"\nCheckpoint saved at {i + 1}")

            # Save final checkpoint
            with open(checkpoint_file, 'w') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')

        # Calculate metrics
        metrics = calculate_metrics(results, dataset_config['metrics'])

        # Save results
        dataset_results = {
            'dataset': dataset_name,
            'num_examples': len(results),
            'metrics': metrics,
            'results': results
        }

        # Save detailed results
        with open(os.path.join(run_dir, f"{dataset_name}_results.json"), 'w') as f:
            json.dump(dataset_results, f, indent=2)

        all_results[dataset_name] = metrics

        # Print metrics
        print(f"\nMetrics for {dataset_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Save summary
    summary = {
        'model': model_name,
        'method': search_method,
        'timestamp': timestamp,
        'results': all_results
    }

    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete! Results saved to {run_dir}")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for dataset, metrics in all_results.items():
        print(f"\n{dataset}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()