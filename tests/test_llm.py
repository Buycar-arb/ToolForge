"""Unit tests for the shared LLM client: routing, key rotation and retries."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toolforge.config import Settings  # noqa: E402
from toolforge.llm import MODEL_IDS, LLMClient, LLMError, model_choices, resolve_provider  # noqa: E402


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.1", ("openai", "gpt-5.1")),
        ("gpt-4.1", ("openai", "gpt-4.1")),
        ("claude-opus-5", ("anthropic", "claude-opus-5")),
        ("claude-sonnet-5", ("anthropic", "claude-sonnet-5")),
        ("anthropic:claude-haiku-4-5", ("anthropic", "claude-haiku-4-5")),
        # An OpenAI-compatible gateway proxying Claude must stay on the OpenAI path.
        ("openai:claude-sonnet-5", ("openai", "claude-sonnet-5")),
        ("deepseek-chat", ("openai", "deepseek-chat")),
        ("qwen3-8b", ("openai", "qwen3-8b")),
        ("some-future-model", ("openai", "some-future-model")),
    ],
)
def test_provider_routing(model: str, expected: tuple[str, str]) -> None:
    assert resolve_provider(model) == expected


def test_registry_is_offered_to_the_ui() -> None:
    choices = model_choices()
    assert len(choices) == len(MODEL_IDS)
    assert {value for _label, value in choices} == set(MODEL_IDS)


def test_missing_key_fails_with_an_actionable_message() -> None:
    empty = Settings(openai_api_keys=[], anthropic_api_keys=[])
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        LLMClient("gpt-5.1", config=empty)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        LLMClient("claude-opus-5", config=empty)


class _Recorder:
    """Stands in for an AsyncOpenAI client and scripts its outcomes."""

    def __init__(self, tag: str, outcomes: list[object], log: list[str]) -> None:
        self.tag, self.outcomes, self.log = tag, outcomes, log
        self.chat = self

    @property
    def completions(self):
        return self

    async def create(self, **_kwargs):
        self.log.append(self.tag)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome

        class Message:
            content = outcome

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


def _client(outcomes: dict[str, list[object]], log: list[str]) -> LLMClient:
    config = replace(
        Settings(openai_api_keys=["k-aaaa", "k-bbbb"]),
        retry_attempts=2,
        retry_delay=0.0,
        retry_max_delay=0.0,
    )
    client = LLMClient("gpt-5.1", config=config)
    client._clients = [_Recorder(tag, list(outcomes.get(tag, [])), log) for tag in ("A", "B")]
    import itertools

    client._cycle = itertools.cycle(range(2))
    return client


def test_first_key_answers_without_touching_the_second() -> None:
    log: list[str] = []
    client = _client({"A": ["hello"]}, log)
    assert asyncio.run(client.complete([{"role": "user", "content": "hi"}])) == "hello"
    assert log == ["A"]


def test_transient_failure_is_retried_on_the_same_key() -> None:
    from openai import APIConnectionError

    log: list[str] = []
    error = APIConnectionError(request=None)  # type: ignore[arg-type]
    client = _client({"A": [error, "recovered"]}, log)
    assert asyncio.run(client.complete([{"role": "user", "content": "hi"}])) == "recovered"
    assert log == ["A", "A"]


def _api_error(kind: str) -> Exception:
    """Build a real openai error object, which needs a real httpx response."""
    import httpx
    import openai

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(429 if kind == "rate" else 500, request=request)
    cls = openai.RateLimitError if kind == "rate" else openai.InternalServerError
    return cls(kind, response=response, body=None)


def test_exhausted_key_rotates_to_the_next_one() -> None:
    log: list[str] = []
    limited = _api_error("rate")
    client = _client({"A": [limited, limited], "B": ["from the second key"]}, log)
    assert asyncio.run(client.complete([{"role": "user", "content": "hi"}])) == "from the second key"
    assert log == ["A", "A", "B"]


def test_total_failure_returns_empty_rather_than_raising() -> None:
    log: list[str] = []
    boom = _api_error("server")
    client = _client({"A": [boom] * 2, "B": [boom] * 2}, log)
    assert asyncio.run(client.complete([{"role": "user", "content": "hi"}])) == ""
    assert log == ["A", "A", "B", "B"]


def test_system_prompt_is_prepended_for_openai() -> None:
    captured: dict[str, object] = {}

    class Capturing(_Recorder):
        async def create(self, **kwargs):
            captured.update(kwargs)
            return await super().create(**kwargs)

    log: list[str] = []
    client = _client({}, log)
    client._clients = [Capturing("A", ["ok"], log), Capturing("B", ["ok"], log)]
    asyncio.run(client.complete([{"role": "user", "content": "hi"}], system="be terse"))
    assert captured["messages"][0] == {"role": "system", "content": "be terse"}
    assert captured["messages"][1] == {"role": "user", "content": "hi"}


# --------------------------------------------------------------------------- #
# Parameter negotiation
#
# The OpenAI chat API is no longer uniform: GPT-5 and the reasoning models want
# `max_completion_tokens`, while many self-hosted servers only know `max_tokens`.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.1", "max_completion_tokens"),
        ("gpt-5", "max_completion_tokens"),
        ("o3-mini", "max_completion_tokens"),
        ("gpt-4.1", "max_tokens"),
        ("gpt-4o", "max_tokens"),
        ("qwen3-8b", "max_tokens"),
    ],
)
def test_token_parameter_is_guessed_from_the_model(model: str, expected: str) -> None:
    from toolforge.llm import ParameterStyle

    assert ParameterStyle.guess(model).token_parameter == expected


def test_style_adapts_in_both_directions() -> None:
    from toolforge.llm import ParameterStyle

    modern = ParameterStyle(token_parameter="max_completion_tokens")
    assert modern.adapt("Unsupported parameter: 'max_completion_tokens' is not supported")
    assert modern.token_parameter == "max_tokens"

    classic = ParameterStyle(token_parameter="max_tokens")
    assert classic.adapt("Unsupported parameter: 'max_tokens' is not supported with this model")
    assert classic.token_parameter == "max_completion_tokens"

    picky = ParameterStyle()
    assert picky.adapt("Unsupported value: 'temperature' does not support 0.0")
    assert picky.send_temperature is False

    assert not ParameterStyle().adapt("something else entirely")


def test_a_rejected_parameter_is_retried_with_the_other_name() -> None:
    """A 400 naming the token parameter must be corrected, not surfaced."""
    import httpx
    import openai

    sent: list[dict[str, object]] = []

    class Negotiating(_Recorder):
        async def create(self, **kwargs):
            sent.append(dict(kwargs))
            if "max_tokens" in kwargs:
                request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
                raise openai.BadRequestError(
                    "Unsupported parameter: 'max_tokens' is not supported with this model.",
                    response=httpx.Response(400, request=request),
                    body=None,
                )
            return await super().create(**kwargs)

    log: list[str] = []
    client = _client({}, log)
    client.style.token_parameter = "max_tokens"  # deliberately the wrong guess
    client._clients = [Negotiating("A", ["adapted"], log), Negotiating("B", [], log)]

    assert asyncio.run(client.complete([{"role": "user", "content": "hi"}])) == "adapted"
    assert client.style.token_parameter == "max_completion_tokens"
    assert "max_tokens" in sent[0] and "max_completion_tokens" in sent[1]


def test_anthropic_requests_omit_temperature_when_the_sdk_rejects_it() -> None:
    """The Anthropic SDK dropped `temperature` from messages.create() in 1.0."""
    from toolforge.llm import _accepts

    def modern(*, model, messages, max_tokens, system=None): ...
    def classic(*, model, messages, max_tokens, temperature=None, system=None): ...

    assert not _accepts(modern, "temperature")
    assert _accepts(classic, "temperature")
