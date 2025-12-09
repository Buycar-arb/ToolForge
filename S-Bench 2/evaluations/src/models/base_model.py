"""Base model interface for evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import yaml


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name')
        self.max_tokens = config.get('max_tokens', 2048)
        self.temperature = config.get('temperature', 0.7)

    @abstractmethod
    def generate_with_functions(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Generate response with function/tool calling capability.

        Returns:
            Dict with 'content' and 'tool_calls' keys
        """
        pass

    @abstractmethod
    def generate_with_tags(self, prompt: str, stop_sequences: List[str] = None, **kwargs) -> str:
        """Generate response using tag-based approach with stop sequences.

        Uses completions API (raw text) for all model types.
        Stops at specified sequences like </search> or </answer>.

        Args:
            prompt: Raw text prompt to continue from
            stop_sequences: List of sequences to stop generation at
            **kwargs: Additional generation parameters

        Returns:
            Generated text up to stop sequence
        """
        pass