"""A single async LLM client for every stage of the pipeline.

The whole project talks to models through :class:`LLMClient`.  It gives you:

* **two providers** - ``openai`` (any OpenAI-compatible endpoint: OpenAI, Azure,
  vLLM, OpenRouter, an internal gateway...) and ``anthropic`` (the native
  Messages API).  The provider is inferred from the model id, or forced with a
  ``provider:model`` prefix such as ``anthropic:claude-opus-5``.
* **key rotation** - pass several keys and the client cycles through them.
* **retries with exponential backoff** on the transient error classes, moving on
  to the next key once a key has exhausted its attempts.
* **one call shape** - :meth:`LLMClient.complete` takes chat messages plus an
  optional system prompt and returns a string (``""`` when every attempt failed).
* **parameter negotiation** - the OpenAI chat API is no longer uniform: GPT-5
  and the reasoning models reject ``max_tokens`` and want
  ``max_completion_tokens``, while many self-hosted servers only understand
  ``max_tokens``. The client guesses from the model id and, if the endpoint
  disagrees, adapts and remembers - see :class:`ParameterStyle`.

Example
-------
>>> client = LLMClient(model="gpt-5.1")
>>> await client.complete([{"role": "user", "content": "hi"}])
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from toolforge.config import Settings
from toolforge.config import settings as default_settings

log = logging.getLogger(__name__)

Message = dict[str, Any]

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelInfo:
    """A curated preset.  Any model id works - this list only drives the UI."""

    id: str
    provider: str
    label: str
    note: str = ""


#: Presets offered in the Web UI dropdowns.  Editing this list is safe; the
#: dropdowns accept free-form text, so a model missing here still works.
MODEL_REGISTRY: tuple[ModelInfo, ...] = (
    # -- Anthropic -------------------------------------------------------- #
    ModelInfo("claude-opus-5", "anthropic", "Claude Opus 5", "highest quality, best judge"),
    ModelInfo("claude-sonnet-5", "anthropic", "Claude Sonnet 5", "balanced quality / cost"),
    ModelInfo("claude-haiku-4-5", "anthropic", "Claude Haiku 4.5", "fastest, cheapest"),
    # -- OpenAI ----------------------------------------------------------- #
    ModelInfo("gpt-5.1", "openai", "GPT-5.1", "strongest OpenAI generator"),
    ModelInfo("gpt-5.1-mini", "openai", "GPT-5.1 mini", "cheap high-volume generation"),
    ModelInfo("gpt-5", "openai", "GPT-5", ""),
    ModelInfo("gpt-4.1", "openai", "GPT-4.1", "the model used in the paper"),
    ModelInfo("gpt-4o", "openai", "GPT-4o", ""),
    # -- OpenAI-compatible third parties ---------------------------------- #
    ModelInfo("deepseek-chat", "openai", "DeepSeek Chat", "needs OPENAI_BASE_URL"),
    ModelInfo("deepseek-reasoner", "openai", "DeepSeek Reasoner", "needs OPENAI_BASE_URL"),
    ModelInfo("qwen3-8b", "openai", "Qwen3-8B (local vLLM)", "needs OPENAI_BASE_URL"),
)

#: Model ids in registry order - handy for building dropdowns.
MODEL_IDS: tuple[str, ...] = tuple(m.id for m in MODEL_REGISTRY)

_ANTHROPIC_HINTS = ("claude", "opus", "sonnet", "haiku", "fable")

#: Model families that require ``max_completion_tokens`` instead of ``max_tokens``.
_NEEDS_COMPLETION_TOKENS = ("gpt-5", "o1", "o3", "o4")


def resolve_provider(model: str) -> tuple[str, str]:
    """Split ``model`` into ``(provider, model_id)``.

    ``"anthropic:claude-opus-5"`` forces the provider; a bare id is inferred
    from the registry first and from name hints second, defaulting to
    ``"openai"`` so that OpenAI-compatible gateways keep working.

    >>> resolve_provider("claude-opus-5")
    ('anthropic', 'claude-opus-5')
    >>> resolve_provider("openai:claude-sonnet-5")   # a gateway proxying Claude
    ('openai', 'claude-sonnet-5')
    """
    if ":" in model:
        prefix, _, rest = model.partition(":")
        if prefix in {"openai", "anthropic"}:
            return prefix, rest
    for info in MODEL_REGISTRY:
        if info.id == model:
            return info.provider, model
    lowered = model.lower()
    if any(hint in lowered for hint in _ANTHROPIC_HINTS):
        return "anthropic", model
    return "openai", model


def model_choices() -> list[tuple[str, str]]:
    """``(label, value)`` pairs for a Gradio dropdown."""
    return [
        (f"{m.label}  ·  {m.note}" if m.note else m.label, m.id)
        for m in MODEL_REGISTRY
    ]


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


def _accepts(callable_object: Any, parameter: str) -> bool:
    """Whether ``callable_object`` takes a keyword argument named ``parameter``."""
    try:
        return parameter in inspect.signature(callable_object).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callables
        return True


class LLMError(RuntimeError):
    """Raised when every key and every retry has been exhausted."""


@dataclass
class ParameterStyle:
    """Which request parameters this endpoint actually accepts.

    Initialised from the model id, then corrected in place the first time the
    server rejects something.  One wasted round trip at most, and only once per
    client, which beats hard-coding a model list that goes stale.
    """

    #: ``"max_tokens"`` (classic, and every self-hosted server) or
    #: ``"max_completion_tokens"`` (GPT-5 and the reasoning models).
    token_parameter: str = "max_tokens"
    #: Some endpoints reject ``temperature`` outright. The Anthropic SDK dropped
    #: it from ``messages.create()`` in 1.0, and some reasoning models accept
    #: only their default.
    send_temperature: bool = True

    @classmethod
    def guess(cls, model: str) -> ParameterStyle:
        lowered = model.lower()
        if any(family in lowered for family in _NEEDS_COMPLETION_TOKENS):
            return cls(token_parameter="max_completion_tokens")
        return cls()

    def adapt(self, error_text: str) -> bool:
        """Adjust to what the endpoint just complained about.

        Returns True when something changed and the call is worth retrying.
        """
        message = error_text.lower()
        if "max_completion_tokens" in message and self.token_parameter == "max_completion_tokens":
            self.token_parameter = "max_tokens"
            return True
        if "max_tokens" in message and self.token_parameter == "max_tokens":
            self.token_parameter = "max_completion_tokens"
            return True
        if "temperature" in message and self.send_temperature:
            self.send_temperature = False
            return True
        return False


class LLMClient:
    """Async chat-completion client with key rotation and backoff.

    Parameters
    ----------
    model:
        Model id, optionally ``provider:``-prefixed.  Defaults to
        :attr:`Settings.generation_model`.
    api_keys:
        Overrides the keys from settings.  Rotated round-robin.
    base_url:
        Overrides the endpoint from settings.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_keys: Sequence[str] | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        config: Settings | None = None,
    ) -> None:
        self.config = config or default_settings
        self.provider, self.model = resolve_provider(model or self.config.generation_model)
        self.base_url = base_url or self.config.base_url_for(self.provider)
        self.temperature = self.config.temperature if temperature is None else temperature
        self.max_tokens = self.config.max_tokens if max_tokens is None else max_tokens

        #: Negotiated request-parameter shape for this endpoint.
        self.style = ParameterStyle.guess(self.model)

        keys = list(api_keys) if api_keys is not None else list(self.config.keys_for(self.provider))
        if not keys:
            raise LLMError(
                f"No API key configured for provider '{self.provider}'. "
                f"Set {'ANTHROPIC_API_KEY' if self.provider == 'anthropic' else 'OPENAI_API_KEY'} "
                "in your .env file (see .env.example)."
            )
        self.api_keys = keys
        self._clients = [self._build_client(key) for key in keys]
        self._cycle = itertools.cycle(range(len(self._clients)))
        self.call_count = 0

    # -- construction ---------------------------------------------------- #
    def _build_client(self, key: str) -> Any:
        if self.provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise LLMError(
                    "Model "
                    f"'{self.model}' needs the native Anthropic SDK: pip install anthropic\n"
                    "Alternatively route it through an OpenAI-compatible gateway with "
                    f"model='openai:{self.model}'."
                ) from exc
            client = AsyncAnthropic(
                api_key=key, base_url=self.base_url, timeout=self.config.request_timeout
            )
            # `temperature` was removed from messages.create() in the 1.0 SDK.
            if not _accepts(client.messages.create, "temperature"):
                self.style.send_temperature = False
            return client

        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=key, base_url=self.base_url, timeout=self.config.request_timeout
        )

    # -- transient error classes ----------------------------------------- #
    @staticmethod
    def _retryable() -> tuple[type[BaseException], ...]:
        errors: list[type[BaseException]] = [asyncio.TimeoutError]
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )

            errors += [APIConnectionError, APITimeoutError, InternalServerError, RateLimitError]
        except ImportError:  # pragma: no cover
            pass
        try:
            import anthropic

            errors += [
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
                anthropic.RateLimitError,
            ]
        except ImportError:  # pragma: no cover
            pass
        return tuple(errors)

    # -- the one public call --------------------------------------------- #
    async def complete(
        self,
        messages: Iterable[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run a chat completion and return the text (``""`` if all attempts fail).

        ``system`` is prepended as a system message for OpenAI and passed as the
        native ``system`` parameter for Anthropic, so the same call site works
        for both providers.
        """
        payload = list(messages)
        temp = self.temperature if temperature is None else temperature
        limit = self.max_tokens if max_tokens is None else max_tokens
        retryable = self._retryable()
        last_error: BaseException | None = None

        for key_round in range(len(self._clients)):
            index = next(self._cycle)
            client = self._clients[index]
            tag = f"...{self.api_keys[index][-4:]}"

            for attempt in range(self.config.retry_attempts):
                try:
                    self.call_count += 1
                    text = await self._dispatch(client, payload, system, temp, limit)
                    if text:
                        return text
                    log.warning("[%s] empty response from key %s", self.model, tag)
                    last_error = LLMError("empty response")
                except retryable as exc:
                    last_error = exc
                    log.warning(
                        "[%s] key %s attempt %d/%d failed: %s",
                        self.model, tag, attempt + 1, self.config.retry_attempts, exc,
                    )
                except Exception as exc:  # non-retryable: try the next key
                    last_error = exc
                    log.error("[%s] key %s hit a non-retryable error: %s", self.model, tag, exc)
                    break

                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self._backoff(attempt))

            if key_round < len(self._clients) - 1:
                log.info("[%s] rotating to the next API key", self.model)

        log.error("[%s] all %d key(s) exhausted: %s", self.model, len(self._clients), last_error)
        return ""

    @staticmethod
    def _is_bad_request(exc: BaseException) -> bool:
        """Whether the endpoint rejected the request itself (HTTP 400)."""
        return getattr(exc, "status_code", None) == 400 or type(exc).__name__ == "BadRequestError"

    def _backoff(self, attempt: int) -> float:
        return min(self.config.retry_delay * (2**attempt), self.config.retry_max_delay)

    async def _dispatch(
        self,
        client: Any,
        messages: list[Message],
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if self.provider == "anthropic":
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if self.style.send_temperature:
                kwargs["temperature"] = temperature
            if system:
                kwargs["system"] = system
            response = await client.messages.create(**kwargs)
            return "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )

        final = ([{"role": "system", "content": system}] + messages) if system else messages

        # Retry once per rejected parameter, at most twice overall.
        for _ in range(3):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": final,
                self.style.token_parameter: max_tokens,
            }
            if self.style.send_temperature:
                kwargs["temperature"] = temperature
            try:
                response = await client.chat.completions.create(**kwargs)
            except Exception as exc:
                if self._is_bad_request(exc) and self.style.adapt(str(exc)):
                    log.info(
                        "[%s] endpoint rejected a parameter, retrying with %s%s",
                        self.model,
                        self.style.token_parameter,
                        "" if self.style.send_temperature else " and no temperature",
                    )
                    continue
                raise
            return response.choices[0].message.content or ""
        return ""

    # -- convenience ------------------------------------------------------ #
    async def complete_text(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Shorthand for a single user turn."""
        return await self.complete([{"role": "user", "content": prompt}], system=system, **kwargs)

    async def aclose(self) -> None:
        """Close the underlying HTTP pools."""
        for client in self._clients:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LLMClient(provider={self.provider!r}, model={self.model!r}, keys={len(self.api_keys)})"
