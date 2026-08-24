"""Stages 3 + 4 — generate dialogues and keep only the ones that validate.

The loop, per case:

1. take the next stage 2 record (cycling round if the file runs out),
2. generate a dialogue for the case,
3. run the nine rule checks; on failure, record the score and move on,
4. otherwise ask the LLM judge; a total of 2/2 is written to the training set.

Every attempt is scored — the score file is a complete audit trail of what was
rejected and why, which is what makes the yield numbers reproducible.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolforge import jsonl
from toolforge.config import Settings
from toolforge.config import settings as default_settings
from toolforge.stages.cases import family_for_route
from toolforge.stages.cases import get as get_case
from toolforge.stages.dialogue import (
    DialogueGenerator,
    GeneratedSample,
    GenerationError,
    SourceRecord,
)
from toolforge.stages.judge import DialogueJudge, Score
from toolforge.stages.validation import ValidationOptions, validate

log = logging.getLogger(__name__)

#: Called with a human-readable progress line.  The CLI prints it; the Web UI
#: streams it into the log pane.
EventHook = Callable[[str], None]

#: Shown in place of the judge's full prose when it rejects a sample.  The prose
#: itself stays in the score file; only the tally is summarised.
JUDGE_REJECTION = "LLM judge: think/action inconsistency"


def summarise_reason(reason: str | None) -> str:
    """Condense a rejection reason into something worth counting.

    Rule checks already come back as short labels. The judge returns a whole
    critique, which would otherwise dominate the summary with one-off entries.
    """
    text = (reason or "unknown").strip()
    if text.startswith("<reasoning>") or "<score>" in text:
        return JUDGE_REJECTION
    return text.split(";")[0].strip()[:90]


@dataclass
class CaseJob:
    """How many samples to produce for one case, and where to put them."""

    case_id: str
    target: int
    data_output: Path
    score_output: Path

    @classmethod
    def from_config(cls, case_id: str, config: dict[str, Any], output_dir: Path) -> CaseJob:
        """Build a job from the JSON block used by the CLI and the Web UI."""
        return cls(
            case_id=case_id,
            target=int(config.get("target_count", 10)),
            data_output=Path(config.get("data_output", output_dir / "data" / f"{case_id}.jsonl")),
            score_output=Path(config.get("score_output", output_dir / "scores" / f"{case_id}.jsonl")),
        )


@dataclass
class CaseProgress:
    """Running totals for one case."""

    case_id: str
    target: int
    succeeded: int = 0
    attempted: int = 0
    rejections: Counter[str] = field(default_factory=Counter)

    @property
    def success_rate(self) -> float:
        return (self.succeeded / self.attempted * 100) if self.attempted else 0.0

    @property
    def complete(self) -> bool:
        return self.succeeded >= self.target

    def summary(self) -> str:
        status = "✅" if self.complete else "⚠️"
        return (
            f"{status} {self.case_id}: {self.succeeded}/{self.target} kept "
            f"from {self.attempted} attempts ({self.success_rate:.1f}%)"
        )


class Pipeline:
    """Runs stages 3 and 4 over a set of stage 2 records."""

    def __init__(
        self,
        *,
        generator: DialogueGenerator | None = None,
        judge: DialogueJudge | None = None,
        validation_options: ValidationOptions | None = None,
        config: Settings | None = None,
    ) -> None:
        self.config = config or default_settings
        self.generator = generator or DialogueGenerator(config=self.config)
        self.judge = judge or DialogueJudge(config=self.config)
        self.validation_options = validation_options or ValidationOptions()

    # ------------------------------------------------------------------ #
    async def process_one(self, record: SourceRecord, case_id: str) -> tuple[GeneratedSample | None, Score]:
        """Generate, validate and score a single sample.

        Returns ``(sample, score)``; ``sample`` is ``None`` unless the score is 2/2.
        """
        try:
            sample = await self.generator.generate(record, case_id)
        except GenerationError as exc:
            return None, Score(case=case_id, rule_score=0, gpt_score="null", error_reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - one bad record must not kill the run
            log.exception("unexpected failure while generating %s", case_id)
            return None, Score(
                case=case_id, rule_score=0, gpt_score="null", error_reason=f"Generation exception: {exc}"
            )

        outcome = validate(sample.to_record(), case_id, self.validation_options)
        score = Score(
            case=case_id,
            rule_score=1 if outcome.passed else 0,
            gpt_score="null",
            uuid=sample.uuid,
            messages={"messages": sample.messages},
            checks=outcome.results,
        )

        if not outcome.passed:
            score.error_reason = outcome.reason
            return None, score

        verdict, detail = await self.judge.score(
            {"messages": sample.messages}, sample.gold_tool_mapping
        )
        score.gpt_score = verdict
        if score.accepted:
            score.good_reason = detail
        else:
            score.error_reason = detail
        return (sample if score.accepted else None), score

    # ------------------------------------------------------------------ #
    async def run_case(
        self,
        records: Sequence[dict[str, Any]],
        job: CaseJob,
        *,
        on_event: EventHook | None = None,
        delay: float = 1.0,
        concurrency: int | None = None,
    ) -> CaseProgress:
        """Fill one case's quota, cycling through ``records`` as needed."""
        emit = on_event or (lambda message: None)
        spec = get_case(job.case_id)  # fail fast on a bad case id

        jsonl.touch(job.data_output)
        jsonl.touch(job.score_output)

        labelled = [r for r in records if "tool_select" in r]
        if not labelled:
            emit(f"⚠️  {job.case_id}: no records carry a 'tool_select' label — did stage 2 run?")
            return CaseProgress(job.case_id, job.target)

        # A case family only makes sense for records stage 2 routed to it: asking
        # a "one call" record for a "several calls" dialogue gives the planner
        # contradictory instructions. Prefer matching records; fall back loudly.
        matching = [r for r in labelled if family_for_route(r.get("route_select", "")) == spec.family]
        if matching:
            usable = matching
        else:
            usable = labelled
            emit(
                f"⚠️  {job.case_id} suits records routed to {spec.source_route}, but none of the "
                f"{len(labelled)} records carry that route — falling back to all of them, "
                "which tends to lower the yield"
            )

        progress = CaseProgress(job.case_id, job.target)
        max_attempts = len(usable) * 2
        workers = max(1, concurrency or self.config.concurrency)
        cursor = 0
        lock = asyncio.Lock()
        emit(f"▶ {job.case_id}: target {job.target}, {len(usable)} labelled records, {workers} workers")

        async def claim() -> dict[str, Any] | None:
            nonlocal cursor
            async with lock:
                if progress.complete or progress.attempted >= max_attempts:
                    return None
                progress.attempted += 1
                record = usable[cursor % len(usable)]
                cursor += 1
                return record

        async def worker() -> None:
            while (raw := await claim()) is not None:
                try:
                    record = SourceRecord.parse(raw)
                except Exception as exc:  # noqa: BLE001
                    emit(f"  ✗ unreadable record: {exc}")
                    continue

                sample, score = await self.process_one(record, job.case_id)

                async with lock:
                    jsonl.append(job.score_output, score.to_record())
                    if sample is not None and not progress.complete:
                        jsonl.append(job.data_output, sample.to_record())
                        progress.succeeded += 1
                        emit(f"  ✓ {job.case_id} {progress.succeeded}/{job.target}")
                    elif sample is not None:
                        emit(f"  · {job.case_id} quota already met, sample scored but not stored")
                    else:
                        reason = summarise_reason(score.error_reason)
                        progress.rejections[reason] += 1
                        emit(f"  ✗ attempt {progress.attempted}: {reason}")

                if delay:
                    await asyncio.sleep(delay)

        await asyncio.gather(*(worker() for _ in range(workers)))
        emit(progress.summary())
        return progress

    # ------------------------------------------------------------------ #
    async def run(
        self,
        records: Sequence[dict[str, Any]],
        jobs: Iterable[CaseJob],
        *,
        on_event: EventHook | None = None,
        delay: float = 1.0,
        concurrency: int | None = None,
    ) -> dict[str, CaseProgress]:
        """Run every case in turn and return the per-case totals."""
        results: dict[str, CaseProgress] = {}
        for job in jobs:
            results[job.case_id] = await self.run_case(
                records, job, on_event=on_event, delay=delay, concurrency=concurrency
            )
        return results


def format_report(results: dict[str, CaseProgress]) -> str:
    """A markdown summary of a finished run."""
    if not results:
        return "No cases were run."

    kept = sum(p.succeeded for p in results.values())
    attempts = sum(p.attempted for p in results.values())
    rate = (kept / attempts * 100) if attempts else 0.0

    lines = [
        "### Run complete",
        "",
        f"**{kept}** samples kept from **{attempts}** attempts — overall yield **{rate:.1f}%**",
        "",
        "| case | kept | target | attempts | yield |",
        "|------|------|--------|----------|-------|",
    ]
    for case_id, progress in results.items():
        mark = "✅" if progress.complete else "⚠️"
        lines.append(
            f"| {mark} `{case_id}` | {progress.succeeded} | {progress.target} | "
            f"{progress.attempted} | {progress.success_rate:.1f}% |"
        )

    rejections: Counter[str] = Counter()
    for progress in results.values():
        rejections.update(progress.rejections)
    if rejections:
        lines += ["", "**Most common rejection reasons**", ""]
        lines += [f"- `{count}×` {reason}" for reason, count in rejections.most_common(8)]
    return "\n".join(lines)


def load_records(input_file: Path | str, limit: int | None = None) -> list[dict[str, Any]]:
    """Read stage 2 output, keeping only rows that carry a tool label."""
    records = jsonl.read_all(input_file, limit=limit)
    return [record for record in records if "tool_select" in record]


def route_summary(records: Sequence[dict[str, Any]]) -> str:
    """Which case families the labelled records can support, and how many of each."""
    from toolforge.stages.cases import CASE_IDS, normalise_route

    counts: Counter[str] = Counter(normalise_route(r.get("route_select", "")) for r in records)
    if not counts:
        return "no routed records"

    lines = []
    for route, count in sorted(counts.items()):
        family = family_for_route(route)
        if family:
            cases = ", ".join(c for c in CASE_IDS if c[5] == family)
            lines.append(f"  {count:>4} × {route}  →  family {family}  ({cases})")
        else:
            lines.append(f"  {count:>4} × {route or '(unlabelled)'}  →  no matching family")
    return "\n".join(lines)
