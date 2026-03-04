"""Base dataset loader."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
import os


class BaseDataset(ABC):
    """Abstract base class for datasets."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.source = config['source']
        self.subset = config['subset']
        self.test_size = config.get('test_size', -1)
        # Fix cache directory path to point to evaluations/data
        self.cache_dir = f"./evaluations/data/{self.subset}"

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load dataset and return list of examples."""
        pass

    def save_cache(self, data: List[Dict]) -> None:
        """Save data to cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "data.jsonl")
        with open(cache_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    def load_cache(self) -> Optional[List[Dict]]:
        """Load data from cache if exists."""
        cache_file = os.path.join(self.cache_dir, "data.jsonl")
        if os.path.exists(cache_file):
            data = []
            with open(cache_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        return None

    def format_question(self, question: str) -> str:
        """Ensure question ends with question mark."""
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
        return question