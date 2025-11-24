"""Tag-based search implementation."""

import re
from typing import Dict, Any, Optional, Tuple
from .search_interface import SearchEngine


class TagBasedSearch:
    """Handle tag-based search interactions."""

    def __init__(self, search_engine: SearchEngine, config: Dict[str, Any]):
        self.search_engine = search_engine
        self.config = config['tag_format']
        self.search_pattern = re.compile(
            f"{re.escape(self.config['search_tag'])}(.*?){re.escape(self.config['search_close'])}",
            re.DOTALL
        )
        self.answer_pattern = re.compile(
            f"{re.escape(self.config['answer_tag'])}(.*?){re.escape(self.config['answer_close'])}",
            re.DOTALL
        )

    def extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from text."""
        matches = self.search_pattern.findall(text)
        if matches:
            query = matches[-1].strip()
            return query[:200]  # Limit query length
        return None

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from text."""
        matches = self.answer_pattern.findall(text)
        if matches:
            return matches[-1].strip()
        return None

    def has_answer(self, text: str) -> bool:
        """Check if text contains answer tags."""
        return self.config['answer_tag'] in text and self.config['answer_close'] in text

    def format_search_results(self, results: str) -> str:
        """Format search results with information tags."""
        return f"\n{self.config['info_tag']}{results}{self.config['info_close']}\n"

    def should_continue(self, text: str) -> Tuple[bool, str]:
        """Determine if generation should continue and why."""
        if self.has_answer(text):
            return False, "answer_found"
        if self.extract_search_query(text):
            return True, "search_needed"
        return True, "continue_generation"