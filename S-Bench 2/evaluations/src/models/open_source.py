"""Open source model implementations via vLLM server."""

import json
import time
from typing import Dict, List, Any
import requests
from .base_model import BaseModel

"""
Chat API vs Completions API:

  # Completions API (generate_raw)
  prompt = "Question: What is the capital? <think>"
  response = model.generate_raw(prompt)
  # Returns: "I need to search...</think><search>capital of France</search>"

  # Chat API (generate)
  messages = [{"role": "user", "content": "Question: What is the capital?"}]
  response = model.generate(messages)
  # Returns: "I'll help you find the capital. Let me search for that information."

"""

class VLLMModel(BaseModel):
    """vLLM server model implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server_url = config['server_url']
        self.model_path = config['model_path']
        self.timeout = config.get('timeout', 30)

    def generate_with_functions(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Generate response with function/tool calling."""
        # For open source models, tools are already injected in system prompt
        # So we don't need to pass them separately
        data = {
            "model": self.model_path,
            "messages": messages,  # Tools already in system prompt
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stream": False
        }

        for retry in range(3):
            try:
                response = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()['choices'][0]['message']

                # Only return content, tool calls will be extracted from content
                return {
                    'content': result.get('content', '')
                }
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)

    def generate_with_tags(self, prompt: str, stop_sequences: List[str] = None, **kwargs) -> str:
        """Generate response using tag-based approach with stop sequences."""
        data = {
            "model": self.model_path,
            "prompt": prompt,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stop": stop_sequences, # ["</search>", "</answer>"]
            "stream": False
        }

        for retry in range(3):
            try:
                response = requests.post(
                    f"{self.server_url}/v1/completions",
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Get the response content
                content = response.json()['choices'][0]['text']

                # Append the stop sequence that was triggered
                # vLLM by default also strips stop sequences like OpenAI
                if stop_sequences and content:
                    # Check for each possible unclosed tag and append the appropriate closing
                    if '<search>' in content and '</search>' not in content:
                        content += '</search>'
                    elif '<answer>' in content and '</answer>' not in content:
                        content += '</answer>'

                return content
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)