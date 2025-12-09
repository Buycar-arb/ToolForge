"""Local dataset loader for custom datasets."""

from typing import Dict, List, Any, Optional
import json
import os
from .base_dataset import BaseDataset


class LocalDataset(BaseDataset):
    """Load local datasets from JSONL files."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize local dataset loader."""
        super().__init__(config)
        # Override cache directory to use the local data path
        self.local_data_path = config.get('local_data_path', f"./data/{self.subset}/data.jsonl")
        
    def load(self) -> List[Dict[str, Any]]:
        """Load dataset from local JSONL file."""
        print(f"Loading {self.subset} from local file: {self.local_data_path}")
        
        if not os.path.exists(self.local_data_path):
            raise FileNotFoundError(f"Local dataset file not found: {self.local_data_path}")
        
        data = []
        with open(self.local_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    # Validate required fields
                    if 'question' not in item:
                        print(f"Warning: Line {line_num} missing 'question' field, skipping")
                        continue
                    
                    # Ensure question ends with question mark
                    question = self.format_question(item['question'])
                    
                    # Handle answers field (can be 'answers' or 'answer')
                    answers = item.get('answers', item.get('answer', []))
                    if isinstance(answers, str):
                        answers = [answers]
                    
                    # Create standardized item
                    processed_item = {
                        'id': item.get('id', f"{self.subset}_{line_num-1}"),
                        'question': question,
                        'answers': answers,
                        'metadata': {
                            'dataset': self.subset,
                            'index': line_num - 1,
                            'source': 'local'
                        }
                    }
                    
                    # Add any additional metadata from original item
                    if 'metadata' in item:
                        processed_item['metadata'].update(item['metadata'])
                    
                    data.append(processed_item)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(data)} examples from local dataset")
        
        # Apply test size limit if specified
        if self.test_size > 0:
            data = data[:self.test_size]
            print(f"Limited to {len(data)} examples (test_size={self.test_size})")
        
        return data
