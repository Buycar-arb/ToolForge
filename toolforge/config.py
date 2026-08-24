"""Central configuration for ToolForge.

Every tunable lives here.  Values come from (in order of precedence):

1. explicit arguments passed in code / on the CLI
2. environment variables (a ``.env`` file at the repo root is auto-loaded)
3. the defaults in this file

Import :data:`settings` for the process-wide singleton::

    from toolforge.config import settings
    print(settings.generation_model)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root (the directory that contains the ``toolforge`` package).
ROOT_DIR = Path(__file__).resolve().parent.parent

#: Directory holding the built-in domain tool libraries (one JSONL file per tool).
DEFAULT_TOOL_BANK_DIR = ROOT_DIR / "tool_bank"

#: Legacy location kept working so existing clones/checkouts do not break.
LEGACY_TOOL_BANK_DIR = ROOT_DIR / "Stage_3" / "tool_bank" / "tools"

#: Name of the fallback ("general") search tool file inside the tool bank.
GENERAL_TOOL_STEM = "general_information_search"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_list(name: str) -> list[str]:
    raw = _env(name)
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_dotenv_file(path: Path | str | None = None) -> Path | None:
    """Load ``KEY=value`` pairs from a ``.env`` file into ``os.environ``.

    Existing environment variables always win, so ``API_KEYS=... python -m toolforge``
    overrides whatever is in the file.  Returns the file that was loaded, or ``None``.
    """
    env_path = Path(path) if path else ROOT_DIR / ".env"
    if not env_path.is_file():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


@dataclass
class Settings:
    """Runtime configuration, resolved from the environment."""

    # -- credentials ------------------------------------------------------- #
    #: Keys for an OpenAI-compatible endpoint.  Multiple keys are rotated.
    openai_api_keys: list[str] = field(default_factory=list)
    #: Keys for the native Anthropic API.  Multiple keys are rotated.
    anthropic_api_keys: list[str] = field(default_factory=list)
    #: Base URL of the OpenAI-compatible endpoint (OpenAI, Azure, vLLM, a gateway...).
    openai_base_url: str = "https://api.openai.com/v1"
    #: Base URL of the Anthropic endpoint.
    anthropic_base_url: str = "https://api.anthropic.com"

    # -- models ------------------------------------------------------------ #
    #: Model used by stages 1-3 to synthesise data.
    generation_model: str = "gpt-5.1"
    #: Model used by stage 4 to score generated dialogues.
    judge_model: str = "claude-opus-5"

    # -- sampling ---------------------------------------------------------- #
    temperature: float = 0.0
    max_tokens: int = 8192

    # -- resilience -------------------------------------------------------- #
    #: Attempts per API key before rotating to the next one.
    retry_attempts: int = 5
    #: Base seconds between retries (exponential backoff, capped).
    retry_delay: float = 5.0
    #: Upper bound for the backoff sleep.
    retry_max_delay: float = 60.0
    #: Per-request timeout in seconds.
    request_timeout: float = 180.0

    # -- retrieval --------------------------------------------------------- #
    #: A tool response contains a random number of passages in ``[min, max]``.
    rag_top_k_min: int = 5
    rag_top_k_max: int = 10

    # -- pipeline ---------------------------------------------------------- #
    #: How many distractor tools to inject alongside the gold tool.
    virtual_tool_min: int = 3
    virtual_tool_max: int = 8
    #: Concurrent in-flight requests for the batch stages.
    concurrency: int = 8

    # -- paths ------------------------------------------------------------- #
    tool_bank_dir: Path = DEFAULT_TOOL_BANK_DIR
    output_dir: Path = ROOT_DIR / "output"

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv_file()

        # ``API_KEYS`` is the historical name and still works for both providers.
        legacy = _env_list("API_KEYS")
        openai_keys = _env_list("OPENAI_API_KEY") or legacy
        anthropic_keys = _env_list("ANTHROPIC_API_KEY") or legacy

        tool_bank = _env("TOOL_BANK_DIR")
        if tool_bank:
            tool_bank_dir = Path(tool_bank)
        elif DEFAULT_TOOL_BANK_DIR.is_dir():
            tool_bank_dir = DEFAULT_TOOL_BANK_DIR
        else:  # pragma: no cover - only hit on pre-refactor checkouts
            tool_bank_dir = LEGACY_TOOL_BANK_DIR

        return cls(
            openai_api_keys=openai_keys,
            anthropic_api_keys=anthropic_keys,
            openai_base_url=_env("OPENAI_BASE_URL") or _env("API_BASE_URL") or "https://api.openai.com/v1",
            anthropic_base_url=_env("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
            generation_model=_env("GENERATION_MODEL") or _env("DEFAULT_MODEL") or "gpt-5.1",
            judge_model=_env("JUDGE_MODEL") or "claude-opus-5",
            temperature=_env_float("TEMPERATURE", 0.0),
            max_tokens=_env_int("MAX_TOKENS", 8192),
            retry_attempts=_env_int("RETRY_ATTEMPTS", 5),
            retry_delay=_env_float("RETRY_DELAY", 5.0),
            retry_max_delay=_env_float("RETRY_MAX_DELAY", 60.0),
            request_timeout=_env_float("REQUEST_TIMEOUT", 180.0),
            rag_top_k_min=_env_int("RAG_TOP_K_MIN", 5),
            rag_top_k_max=_env_int("RAG_TOP_K_MAX", 10),
            virtual_tool_min=_env_int("VIRTUAL_TOOL_MIN", 3),
            virtual_tool_max=_env_int("VIRTUAL_TOOL_MAX", 8),
            concurrency=_env_int("CONCURRENCY", 8),
            tool_bank_dir=tool_bank_dir,
            output_dir=Path(_env("OUTPUT_DIR") or ROOT_DIR / "output"),
        )

    # ------------------------------------------------------------------ #
    @property
    def general_tool_file(self) -> Path:
        """Path of the fallback ``general_information_search`` tool library."""
        return self.tool_bank_dir / f"{GENERAL_TOOL_STEM}.jsonl"

    def keys_for(self, provider: str) -> list[str]:
        """API keys configured for ``provider`` (``"openai"`` or ``"anthropic"``)."""
        return self.anthropic_api_keys if provider == "anthropic" else self.openai_api_keys

    def base_url_for(self, provider: str) -> str:
        """Endpoint configured for ``provider``."""
        return self.anthropic_base_url if provider == "anthropic" else self.openai_base_url

    def describe(self) -> str:
        """Human-readable summary with the keys masked - safe to print or log."""

        def mask(keys: list[str]) -> str:
            if not keys:
                return "(none)"
            return ", ".join(f"...{k[-4:]}" if len(k) > 4 else "****" for k in keys)

        return "\n".join(
            [
                f"generation model : {self.generation_model}",
                f"judge model      : {self.judge_model}",
                f"openai base url  : {self.openai_base_url}",
                f"openai keys      : {mask(self.openai_api_keys)}",
                f"anthropic keys   : {mask(self.anthropic_api_keys)}",
                f"tool bank        : {self.tool_bank_dir}",
                f"output dir       : {self.output_dir}",
                f"temperature      : {self.temperature}   max_tokens: {self.max_tokens}",
                f"rag top-k        : {self.rag_top_k_min}-{self.rag_top_k_max}",
            ]
        )


#: Process-wide settings, resolved once at import time.
settings = Settings.from_env()


def reload_settings() -> Settings:
    """Re-read the environment (useful after editing ``.env`` in a live session)."""
    global settings
    settings = Settings.from_env()
    return settings
