"""Base search interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import requests
import time


class SearchEngine(ABC):
    """Abstract base class for search engines."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url = config['url']
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.top_k = config.get('top_k', 3)

    def search(self, query: str) -> str:
        """Execute search and return formatted results."""
        payload = {
            "queries": [query],
            "topk": self.top_k,
            "return_scores": self.config.get('return_scores', True)
        }

        for retry in range(self.max_retries):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                results = response.json()['result'][0]
                return self._format_results(results)
            except Exception as e:
                if retry == self.max_retries - 1:
                    raise e
                time.sleep(2 ** retry)

    def _format_results(self, results: List[Dict]) -> str:
        """Format search results for insertion."""
        formatted = []
        for idx, doc in enumerate(results):
            content = doc['document']['contents']
            lines = content.split('\n')
            title = lines[0] if lines else ""
            text = '\n'.join(lines[1:]) if len(lines) > 1 else content
            formatted.append(f"Doc {idx + 1}(Title: {title}) {text}")
        return '\n'.join(formatted)