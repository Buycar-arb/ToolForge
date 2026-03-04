"""Prompt template management utilities."""

import yaml
from typing import Dict, List, Any


class PromptManager:
    """Manage prompt templates for different models and methods."""

    def __init__(self, config_path: str = "./config/prompts.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_prompt(self, model_name: str, search_method: str, question: str) -> Dict[str, str]:
        """Get formatted prompt for a model and search method."""
        method_key = f"{search_method}_based"

        if model_name in self.config.get("model_specific", {}):
            if method_key in self.config["model_specific"][model_name]:
                prompt_template = self.config["model_specific"][model_name][method_key]
            else:
                prompt_template = self.config["prompts"][method_key]
        else:
            prompt_template = self.config["prompts"][method_key]

        formatted = {}
        if prompt_template.get("system"):
            formatted["system"] = prompt_template["system"]
        formatted["user"] = prompt_template["user"].format(question=question)

        return formatted