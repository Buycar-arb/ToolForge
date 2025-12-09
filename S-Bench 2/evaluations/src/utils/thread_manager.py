"""Multi-threading support for evaluation tasks."""

import threading
import queue
import time
from typing import Dict, Any, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from tqdm import tqdm


class ThreadSafeCounter:
    """Thread-safe counter for tracking progress."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    def get_value(self) -> int:
        with self._lock:
            return self._value


class ThreadSafeFileWriter:
    """Thread-safe file writer for checkpoint files."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._lock = threading.Lock()
    
    def write_line(self, data: Dict[str, Any]):
        """Write a single line to the file."""
        with self._lock:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


class MultiThreadEvaluator:
    """Multi-threaded evaluator for running evaluations in parallel."""
    
    def __init__(self, 
                 model_factory: Callable,
                 search_factory: Callable,
                 prompt_config: Dict[str, Any],
                 search_method: str,
                 max_workers: int = 4,
                 checkpoint_every: int = 10):
        """
        Initialize multi-threaded evaluator.
        
        Args:
            model_factory: Function to create model instances
            search_factory: Function to create search handler instances
            prompt_config: Prompt configuration
            search_method: Search method ('tag' or 'function')
            max_workers: Maximum number of worker threads
            checkpoint_every: Save checkpoint every N completed tasks
        """
        self.model_factory = model_factory
        self.search_factory = search_factory
        self.prompt_config = prompt_config
        self.search_method = search_method
        self.max_workers = max_workers
        self.checkpoint_every = checkpoint_every
        
        # Thread-safe components
        self.counter = ThreadSafeCounter()
        self.results_lock = threading.Lock()
        self.results = []
        
    def _create_worker_components(self):
        """Create model and search components for a worker thread."""
        model = self.model_factory()
        search_handler = self.search_factory()
        return model, search_handler
    
    def _evaluate_single_item(self, item: Dict[str, Any], checkpoint_writer: Optional[ThreadSafeFileWriter] = None) -> Dict[str, Any]:
        """
        Evaluate a single item (question).
        
        Args:
            item: Dataset item containing question and metadata
            checkpoint_writer: Optional checkpoint writer for saving results
            
        Returns:
            Evaluation result
        """
        try:
            # Create components for this thread
            model, search_handler = self._create_worker_components()
            
            # Import inference classes
            if self.search_method == 'tag':
                from ..inference.tag_based_inference import TagBasedInference
                inference = TagBasedInference(model, search_handler, self.prompt_config)
            elif self.search_method == 'function':
                from ..inference.function_inference import FunctionInference
                inference = FunctionInference(model, search_handler, self.prompt_config)
            else:
                raise ValueError(f"Unknown search method: {self.search_method}")
            
            # Run inference
            result = inference.run(item['question'])
            
            # Create simplified result
            simplified_result = {
                'id': item['id'],
                'question': item['question'],
                'gold_answer': item['answers'][0] if item['answers'] else '',
                'prediction': result.get('answer', '')
            }
            
            # Add method-specific data
            if self.search_method == 'tag':
                simplified_result['response'] = result.get('response', '')
            elif self.search_method == 'function':
                simplified_result['messages'] = result.get('messages', [])
            
            # Write checkpoint if writer provided
            if checkpoint_writer:
                checkpoint_writer.write_line(simplified_result)
            
            return simplified_result
            
        except Exception as e:
            error_result = {
                'id': item['id'],
                'question': item['question'],
                'gold_answer': item['answers'][0] if item['answers'] else '',
                'prediction': '',
                'error': str(e)
            }
            
            if checkpoint_writer:
                checkpoint_writer.write_line(error_result)
            
            return error_result
    
    def evaluate_dataset(self, 
                        dataset_name: str,
                        data: List[Dict[str, Any]], 
                        output_dir: str,
                        resume_from_checkpoint: bool = True) -> List[Dict[str, Any]]:
        """
        Evaluate a dataset using multiple threads.
        
        Args:
            dataset_name: Name of the dataset
            data: List of dataset items
            output_dir: Output directory for results
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            List of evaluation results
        """
        checkpoint_file = os.path.join(output_dir, f"{dataset_name}_checkpoint.jsonl")
        checkpoint_writer = ThreadSafeFileWriter(checkpoint_file)
        
        # Load existing results if resuming
        completed_results = []
        start_index = 0
        
        if resume_from_checkpoint and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        completed_results.append(json.loads(line))
            start_index = len(completed_results)
            print(f"Resumed from checkpoint: {start_index} completed")
        
        # Prepare remaining items
        remaining_items = data[start_index:]
        
        if not remaining_items:
            print("All items already completed!")
            return completed_results
        
        print(f"Evaluating {len(remaining_items)} items with {self.max_workers} workers...")
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._evaluate_single_item, item, checkpoint_writer): item
                for item in remaining_items
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(remaining_items), desc=f"Evaluating {dataset_name}") as pbar:
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        
                        # Thread-safe result collection
                        with self.results_lock:
                            completed_results.append(result)
                        
                        # Update progress
                        pbar.update(1)
                        
                        # Print progress every checkpoint_every items
                        current_count = self.counter.increment()
                        if current_count % self.checkpoint_every == 0:
                            print(f"\nCompleted {current_count} items")
                            
                    except Exception as e:
                        item = future_to_item[future]
                        print(f"Error processing item {item['id']}: {e}")
                        pbar.update(1)
        
        print(f"Evaluation complete! Total results: {len(completed_results)}")
        return completed_results


class BatchProcessor:
    """Process items in batches with multi-threading."""
    
    def __init__(self, 
                 evaluator: MultiThreadEvaluator,
                 batch_size: int = 100):
        """
        Initialize batch processor.
        
        Args:
            evaluator: Multi-threaded evaluator instance
            batch_size: Number of items to process in each batch
        """
        self.evaluator = evaluator
        self.batch_size = batch_size
    
    def process_dataset(self, 
                       dataset_name: str,
                       data: List[Dict[str, Any]], 
                       output_dir: str) -> List[Dict[str, Any]]:
        """
        Process dataset in batches.
        
        Args:
            dataset_name: Name of the dataset
            data: List of dataset items
            output_dir: Output directory for results
            
        Returns:
            List of all evaluation results
        """
        all_results = []
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        print(f"Processing {len(data)} items in {total_batches} batches of {self.batch_size}")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(data))
            batch_data = data[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} (items {start_idx}-{end_idx-1})")
            
            # Process batch
            batch_results = self.evaluator.evaluate_dataset(
                f"{dataset_name}_batch_{batch_idx}",
                batch_data,
                output_dir,
                resume_from_checkpoint=True
            )
            
            all_results.extend(batch_results)
            
            # Save intermediate results
            intermediate_file = os.path.join(output_dir, f"{dataset_name}_batch_{batch_idx}_results.json")
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        return all_results
