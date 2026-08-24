"""API-served models for the evaluation harness.

The original release carried four near-identical classes (OpenAI, Claude, Grok,
DeepSeek) that differed only in one optional request field.  They are one class
now — :class:`APIModel` — with the old names kept as aliases so existing configs
and imports keep working.

Two calling modes, matching the two inference styles in ``config/search_engines.yaml``:

``generate_with_tags``
    Search-R1 style. One HTTP call to the configured endpoint with ``stop``
    sequences; the closing tag the server strips is appended back.

``generate_with_functions``
    Standard chat completion through the shared async client, which rotates the
    keys in ``API_KEYS`` and retries transient failures.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import time
from typing import Any, Dict, List

import requests
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from .base_model import BaseModel

#: Keys for the OpenAI-compatible endpoint, comma-separated, rotated per call.
API_KEYS = [key.strip() for key in os.getenv("API_KEYS", "").split(",") if key.strip()]

#: Endpoint used by ``generate_with_functions``.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

#: Tags whose closing half the server strips when it honours a stop sequence.
_CLOSING_TAGS = ("search", "answer")


def _resolve(value: str | None) -> str:
    """Expand a ``${VAR}`` placeholder from the environment, else pass through."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    return value or ""


def _restore_stop_tag(content: str) -> str:
    """Put back the closing tag the API stripped when it hit a stop sequence."""
    for tag in _CLOSING_TAGS:
        if f"<{tag}>" in content and f"</{tag}>" not in content:
            return content + f"</{tag}>"
    return content


class AsyncChatCaller:
    """Async chat completions with key rotation and retries."""

    def __init__(self, model: str, retry_attempts: int = 15, retry_delay: int = 60) -> None:
        self.model = model
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.clients = [AsyncOpenAI(api_key=key, base_url=API_BASE_URL) for key in API_KEYS]
        self.cycle = itertools.cycle(self.clients) if self.clients else None
        self.lock = asyncio.Lock()

    async def generate(self, messages: List[Dict[str, str]], max_tokens: int = 10000) -> str | None:
        if not self.cycle:
            raise RuntimeError("No API_KEYS configured — set API_KEYS in the environment.")

        async with self.lock:
            client = next(self.cycle)

        for attempt in range(self.retry_attempts):
            try:
                response = await client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens, temperature=0.0
                )
                return response.choices[0].message.content
            except (APIConnectionError, RateLimitError, APITimeoutError, InternalServerError) as exc:
                print(f"[{self.model}] attempt {attempt + 1}/{self.retry_attempts} failed: {exc}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
            except Exception as exc:  # noqa: BLE001 - non-retryable, give up now
                print(f"[{self.model}] non-retryable error: {exc}")
                return None
        return None


class APIModel(BaseModel):
    """Any model reachable over an OpenAI-compatible HTTP API.

    Recognised config keys (``config/models.yaml``)::

        type: closed_source
        model_name: gpt-5.1          # sent as the `model` field
        api_key: ${OPENAI_API_KEY}   # ${VAR} is read from the environment
        endpoint: https://.../chat/completions
        max_tokens: 8000
        temperature: 0
        timeout: 60
        thinking: {...}              # optional, forwarded verbatim
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = _resolve(config.get("api_key"))
        self.endpoint = _resolve(config.get("endpoint"))
        self.model_name = config["model_name"]
        self.timeout = config.get("timeout", 60)
        #: Extended-thinking block, forwarded to the endpoint when present.
        self.thinking = config.get("thinking") or {}
        self.caller = AsyncChatCaller(self.model_name)

    # -- tag-based inference -------------------------------------------- #
    def generate_with_tags(self, prompt: str, stop_sequences: List[str] | None = None, **kwargs: Any) -> str:
        if not self.endpoint:
            raise ValueError(
                f"Model '{self.model_name}' needs an `endpoint` in models.yaml for tag-based inference."
            )

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stop": stop_sequences,
        }
        if self.thinking:
            payload["thinking"] = self.thinking

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last: Exception | None = None
        for retry in range(3):
            try:
                response = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return _restore_stop_tag(content) if stop_sequences and content else content
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                if retry < 2:
                    time.sleep(2**retry)
        raise last if last else RuntimeError("request failed")

    # -- function-calling inference ------------------------------------- #
    def generate_with_functions(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs: Any) -> Dict:
        """Tools are already described in the system prompt, so ``tools`` is unused."""
        try:
            content = asyncio.run(
                self.caller.generate(messages, max_tokens=kwargs.get("max_tokens", self.max_tokens))
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal to a sweep
            print(f"[{self.model_name}] function-calling request failed: {exc}")
            content = None
        return {"content": content or "", "tool_calls": []}


# Backwards-compatible names: older configs and scripts import these directly.
OpenAIModel = ClaudeModel = GrokModel = DeepSeekModel = APIModel

__all__ = [
    "APIModel",
    "AsyncChatCaller",
    "ClaudeModel",
    "DeepSeekModel",
    "GrokModel",
    "OpenAIModel",
]
