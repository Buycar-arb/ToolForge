"""Stage 4, part 2 — LLM quality scoring.

The rule checks in :mod:`toolforge.stages.validation` verify *structure*.  This
module asks a strong model to verify *reasoning*: whether each ``<think>`` block
is consistent with the action that follows it, and whether a mistaken turn is
properly reflected on by the next one.

A sample's ``total_score`` is ``rule_score + gpt_score``; only ``2`` is kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from toolforge.config import Settings
from toolforge.config import settings as default_settings
from toolforge.llm import LLMClient
from toolforge.prompts.judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT


def parse_score(response: str) -> tuple[int | None, str | None]:
    """Extract the verdict from ``<score>\\n[1]\\n</score>``.

    Returns ``(score, None)`` on success or ``(None, reason)`` when the judge's
    output could not be parsed.
    """
    matches = re.findall(r"<score>\s*\[\s*([01])\s*\]\s*</score>", response or "", re.DOTALL)
    if matches:
        # Reasoning can quote or discuss an earlier candidate verdict.  The
        # judge contract puts the actual verdict at the end, so use the last
        # syntactically valid score block rather than the first one mentioned.
        return int(matches[-1]), None
    return None, f"Could not parse a <score> block from the judge response: {(response or '')[:200]!r}"


@dataclass
class Score:
    """The full scorecard for one generated sample."""

    case: str
    #: 1 when all nine rule checks passed, else 0.
    rule_score: int
    #: 1/0 from the LLM judge, or ``"null"`` when it was not reached or failed.
    gpt_score: int | str
    uuid: str = ""
    messages: dict[str, Any] | None = None
    good_reason: str | None = None
    error_reason: str | None = None
    #: Per-check outcomes from the rule stage, for debugging.
    checks: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.rule_score + (self.gpt_score if isinstance(self.gpt_score, int) else 0)

    @property
    def accepted(self) -> bool:
        """Only a perfect 2 is written to the training set."""
        return self.total == 2

    def to_record(self) -> dict[str, Any]:
        """The row written to the score JSONL."""
        record: dict[str, Any] = {
            "case": self.case,
            "rule_score": self.rule_score,
            "gpt_score": self.gpt_score,
            "total_score": self.total,
        }
        if self.accepted:
            record["good_reason"] = self.good_reason
        else:
            record["error_reason"] = self.error_reason or ""
        if self.uuid:
            record["uuid"] = self.uuid
        if self.messages is not None:
            record["data"] = self.messages
        if self.checks:
            record["checks"] = self.checks
        return record


class DialogueJudge:
    """Scores a generated dialogue for think/action consistency."""

    def __init__(self, client: LLMClient | None = None, *, config: Settings | None = None) -> None:
        self.config = config or default_settings
        self.client = client or LLMClient(self.config.judge_model, config=self.config)

    async def score(self, messages: dict[str, Any], tool_mapping: list[dict[str, str]]) -> tuple[int | str, str]:
        """Return ``(score, raw judge response)``.

        ``score`` is ``1``/``0``, or ``"null"`` when the judge failed or its
        answer could not be parsed — in which case the raw text carries the reason.
        """
        prompt = JUDGE_USER_PROMPT.format(messages=messages, good_tool_mapping=tool_mapping)
        response = await self.client.complete(
            [{"role": "user", "content": prompt}], system=JUDGE_SYSTEM_PROMPT
        )
        if not response:
            return "null", "The judge model returned nothing."
        score, error = parse_score(response)
        if error:
            return "null", error
        return score, response
