"""Stage 2 — label each multi-hop question with a tool and a routing class.

Input is raw HotpotQA / 2WikiMultihopQA JSONL.  For every question the model
returns a reasoning trace, the tools to call in execution order, and the route
(``case1``-``case4``) that decides the shape of the stage 3 dialogue.

Three fields are added to each record and consumed by stage 3:

``reasoning``
    the ``<think>`` block — becomes the trajectory guidance for planning
``tool_select``
    ``"[tool_a,tool_b]"`` — which tool library the gold tool is drawn from
``route_select``
    ``"[case3]"`` — the routing class

Records the model never manages to label keep the same three fields with an
error marker, so nothing silently disappears from the file.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolforge import jsonl
from toolforge.config import Settings
from toolforge.config import settings as default_settings
from toolforge.llm import LLMClient
from toolforge.prompts.tool_selection import (
    TOOL_CHOOSE_SYSTEM_A,
    TOOL_CHOOSE_SYSTEM_BCD,
    TOOL_CHOOSE_USER_A,
    TOOL_CHOOSE_USER_BCD,
    TOOL_LIST,
)

log = logging.getLogger(__name__)

EventHook = Callable[[str], None]

#: Marker written into a record whose labelling never succeeded.
FAILED = "Processing failed"


class LabelParseError(ValueError):
    """The model's answer did not contain the expected sections."""


def parse_label(content: str) -> tuple[str, str, str]:
    """Split a model answer into ``(reasoning, tool_select, route_select)``.

    Expected shape::

        <think>
        ...reasoning...
        </think>
        工具选择:[tool_a,tool_b]
        路径选择:[case3]
    """
    for marker in ("<think>\n", "\n</think>", "\n工具选择:", "\n路径选择:"):
        if marker not in content:
            raise LabelParseError(f"missing {marker.strip()!r} in the model answer")

    reasoning, _, remainder = content.split("<think>\n", 1)[1].partition("\n</think>")
    tools, _, route = remainder.split("\n工具选择:", 1)[1].partition("\n路径选择:")
    return reasoning, tools, route


@dataclass
class LabelStats:
    """Totals for one labelling run."""

    total: int = 0
    labelled: int = 0
    failed: int = 0
    deferred: int = 0

    def summary(self) -> str:
        return (
            f"{self.labelled} labelled · {self.failed} failed · "
            f"{self.deferred} left for later · {self.total} read"
        )


class ToolLabeler:
    """Labels stage 2 records, with retries and bounded concurrency."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        tool_list: str = TOOL_LIST,
        force_single_call: bool = False,
        config: Settings | None = None,
    ) -> None:
        self.config = config or default_settings
        self.client = client or LLMClient(config=self.config)
        self.tool_list = tool_list
        # The ``_A`` prompt forces route ``case1``; used to top up that class.
        self.system_prompt = TOOL_CHOOSE_SYSTEM_A if force_single_call else TOOL_CHOOSE_SYSTEM_BCD
        self.user_template = TOOL_CHOOSE_USER_A if force_single_call else TOOL_CHOOSE_USER_BCD

    async def label_one(
        self,
        record: dict[str, Any],
        *,
        attempts: int = 5,
        on_answer: Callable[[dict[str, Any], str], None] | None = None,
    ) -> dict[str, Any]:
        """Label one record in place, retrying on malformed answers."""
        prompt = self.user_template.format(question=record["question"], tool_list=self.tool_list)
        last_error = "no attempt was made"

        for attempt in range(attempts):
            content = await self.client.complete(
                [{"role": "user", "content": prompt}], system=self.system_prompt
            )
            if not content:
                last_error = "the model returned nothing"
            else:
                if on_answer:
                    on_answer(record, content)
                try:
                    reasoning, tools, route = parse_label(content)
                except LabelParseError as exc:
                    last_error = str(exc)
                else:
                    record["reasoning"] = reasoning
                    record["tool_select"] = tools
                    record["route_select"] = route
                    record.pop("processing_error", None)
                    return record

            if attempt < attempts - 1:
                await asyncio.sleep(min(2**attempt, 15))

        log.warning("giving up on %r: %s", record.get("question", "")[:60], last_error)
        record["processing_error"] = f"{FAILED}: {last_error}"
        record["reasoning"] = FAILED
        record["tool_select"] = FAILED
        record["route_select"] = FAILED
        return record

    async def label_many(
        self,
        records: Sequence[dict[str, Any]],
        *,
        concurrency: int | None = None,
        on_event: EventHook | None = None,
        on_answer: Callable[[dict[str, Any], str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Label a batch, preserving input order."""
        emit = on_event or (lambda message: None)
        semaphore = asyncio.Semaphore(max(1, concurrency or self.config.concurrency))
        done = 0

        async def run(index: int, record: dict[str, Any]) -> dict[str, Any]:
            nonlocal done
            async with semaphore:
                result = await self.label_one(record, on_answer=on_answer)
            done += 1
            if done % 10 == 0 or done == len(records):
                emit(f"  labelled {done}/{len(records)}")
            return result

        return list(await asyncio.gather(*(run(i, r) for i, r in enumerate(records))))


async def run_labeling(
    input_file: Path | str,
    output_file: Path | str,
    *,
    residue_file: Path | str | None = None,
    max_records: int | None = None,
    concurrency: int | None = None,
    labeler: ToolLabeler | None = None,
    on_event: EventHook | None = None,
    on_answer: Callable[[dict[str, Any], str], None] | None = None,
) -> LabelStats:
    """Label ``input_file`` into ``output_file``.

    Records beyond ``max_records`` are written untouched to ``residue_file`` so a
    large corpus can be worked through in batches.
    """
    emit = on_event or print
    labeler = labeler or ToolLabeler()

    everything = jsonl.read_all(input_file)
    cutoff = len(everything) if max_records is None else max_records
    batch, residue = everything[:cutoff], everything[cutoff:]
    stats = LabelStats(total=len(everything), deferred=len(residue))
    emit(f"read {stats.total} records — labelling {len(batch)}, deferring {len(residue)}")

    if residue and residue_file:
        jsonl.write(residue_file, residue)
        emit(f"deferred records written to {residue_file}")

    if not batch:
        emit("nothing to label")
        return stats

    labelled = await labeler.label_many(
        batch, concurrency=concurrency, on_event=emit, on_answer=on_answer
    )
    stats.failed = sum(1 for record in labelled if record.get("tool_select") == FAILED)
    stats.labelled = len(labelled) - stats.failed

    jsonl.write(output_file, labelled)
    emit(f"wrote {output_file} — {stats.summary()}")
    return stats
