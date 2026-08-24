"""End-to-end checks that run entirely offline.

The suite drives the real stage 3 engine and the real stage 4 checks against a
scripted model (:mod:`tests.mock_llm`), so it catches regressions in bundling,
ordering, rendering and validation without spending a single API call.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.mock_llm import ScriptedGenerator, make_generator  # noqa: E402
from toolforge import toolbank  # noqa: E402
from toolforge.stages.cases import CASE_IDS, CASE_SPECS  # noqa: E402
from toolforge.stages.dialogue import SourceRecord  # noqa: E402
from toolforge.stages.pipeline import Pipeline  # noqa: E402
from toolforge.stages.validation import DIALOGUE_PATTERNS, ValidationOptions, validate  # noqa: E402

GOLD_TOOL = "person_information_search"


def build_record(distractors: int = 60) -> dict:
    """A HotpotQA-shaped record with two supporting facts and plenty of noise."""
    return {
        "_id": "test-0001",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
        "context": [
            ["Scott Derrickson", ["Scott Derrickson (born 1966) is an American director."]],
            ["Ed Wood", ["Edward Davis Wood Jr. was an American director and screenwriter."]],
            *[
                [f"Distractor {index}", [f"Filler passage number {index} about unrelated cinema history."]]
                for index in range(distractors)
            ],
        ],
        "reasoning": "Look up each person's nationality, then compare them.",
        "tool_select": f"[{GOLD_TOOL}]",
        "route_select": "[case3]",
    }


async def generate_one(case_id: str) -> tuple[list, ScriptedGenerator]:
    """Run the real engine for one case against the scripted model."""
    record = SourceRecord.parse(build_record())
    spec = CASE_SPECS[case_id]

    context = toolbank.build_context(record.gold_tool_names)
    while spec.use_fallback_tools and not context.fallback_available:
        # The gold tool happened to be general search; resample.
        context = toolbank.build_context(record.gold_tool_names)

    generator = make_generator(record, context, spec.turns)
    sample = await generator.generate(record, case_id, context)
    return sample.to_record(), generator


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_case_generates_and_validates(case_id: str) -> None:
    """Every case produces a record that passes all nine rule checks."""
    record, _ = asyncio.run(generate_one(case_id))
    outcome = validate(record, case_id)
    assert outcome.passed, f"{case_id} failed: {outcome.reason}"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_case_matches_expected_role_pattern(case_id: str) -> None:
    """The dialogue shape matches the pattern stage 4 expects for the case."""
    record, _ = asyncio.run(generate_one(case_id))
    roles = [message["role"] for message in record[1]["messages"]]
    assert roles == DIALOGUE_PATTERNS[case_id]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_tool_messages_align_with_bundles(case_id: str) -> None:
    """There is exactly one passage bundle per tool message, in order."""
    record, _ = asyncio.run(generate_one(case_id))
    tool_messages = [m for m in record[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == len(record[2]["rags"]) == len(CASE_SPECS[case_id].tool_messages)
    assert all(bundle for bundle in record[2]["rags"]), "a tool message would render empty"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_strict_reference_check(case_id: str) -> None:
    """With check 7 enabled, the passages used still match ``supporting_facts``.

    Check 7 was inert in the published release (it read the wrong record slots).
    This proves the corrected implementation passes on well-formed data, so
    ``--strict-references`` is safe to turn on.
    """
    record, _ = asyncio.run(generate_one(case_id))
    outcome = validate(record, case_id, ValidationOptions(strict_reference_check=True))
    assert outcome.passed, f"{case_id} failed with strict references: {outcome.reason}"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_argument_check_runs_where_declared(case_id: str) -> None:
    """Cases that declare check 6 emit the paired tool calls it needs."""
    record, _ = asyncio.run(generate_one(case_id))
    spec = CASE_SPECS[case_id]
    payload = record[3]["argument_check"]
    if spec.check_arguments:
        assert isinstance(payload, list) and payload, f"{case_id} declares check 6 but emitted nothing"
        start, stop = spec.argument_check_range
        assert stop <= len(payload) + 1, (
            f"{case_id} argument range {spec.argument_check_range} exceeds "
            f"the {len(payload)} tool-calling turns it produces"
        )
    else:
        assert payload == "Don't need to check"


def test_all_offered_tools_are_resolvable() -> None:
    """Every tool offered in a prompt has a schema the validator can look up."""
    record, generator = asyncio.run(generate_one("case_A2"))
    definitions = generator.tool_definitions
    for tool in record[5]["argument_tool_bank"]:
        assert tool["name"] in definitions


def test_fallback_cases_are_skipped_when_gold_tool_is_general() -> None:
    """A fallback case is refused when there is nothing to fall back to."""
    from toolforge.stages.dialogue import GenerationError

    raw = build_record()
    raw["tool_select"] = "[general_information_search]"
    record = SourceRecord.parse(raw)
    context = toolbank.build_context(record.gold_tool_names)
    assert not context.fallback_available

    generator = make_generator(record, context, turns=1)
    with pytest.raises(GenerationError, match="general search"):
        asyncio.run(generator.generate(record, "case_A4", context))


# --------------------------------------------------------------------------- #
# Route ↔ family correspondence
#
# Stage 2 labels each question with a routing class; that class decides which
# case family can sensibly be generated from it. Asking a "one call" record for
# a "several calls" dialogue hands the planner contradictory instructions.
# --------------------------------------------------------------------------- #


def test_every_family_maps_to_exactly_one_route() -> None:
    from toolforge.stages.cases import FAMILY_TO_ROUTE, ROUTE_TO_FAMILY

    assert set(ROUTE_TO_FAMILY.values()) == {"A", "B", "C", "D"}
    assert set(FAMILY_TO_ROUTE) == {"A", "B", "C", "D"}
    for route, family in ROUTE_TO_FAMILY.items():
        assert FAMILY_TO_ROUTE[family] == route


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_each_case_declares_the_route_it_needs(case_id: str) -> None:
    from toolforge.stages.cases import ROUTE_TO_FAMILY

    spec = CASE_SPECS[case_id]
    assert ROUTE_TO_FAMILY[spec.source_route] == spec.family


@pytest.mark.parametrize(
    ("label", "expected"),
    [("[case1]", "A"), ("case2", "B"), ("  [CASE3] ", "C"), ("[case4]", "D"),
     ("", None), ("[nonsense]", None)],
)
def test_route_labels_are_normalised(label: str, expected: str | None) -> None:
    from toolforge.stages.cases import family_for_route

    assert family_for_route(label) == expected


def _routed(route: str, identifier: str) -> dict:
    """A labelled record carrying a specific stage 2 routing class."""
    record = build_record()
    record["route_select"] = route
    record["_id"] = identifier
    return record


class _StubPipeline(Pipeline):
    """Records which source records a case actually drew, without calling a model."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips the real __init__
        from toolforge.config import settings

        self.config = settings
        self.generator = None
        self.judge = None
        self.validation_options = None
        self.seen: list[str] = []

    async def process_one(self, record, case_id):
        from toolforge.stages.judge import Score

        self.seen.append(record.raw["_id"])
        return None, Score(case=case_id, rule_score=0, gpt_score="null", error_reason="stub")


def _run_case(pipeline: _StubPipeline, case_id: str, records: list[dict],
              messages: list[str] | None = None) -> None:
    import asyncio
    import tempfile

    from toolforge.stages.pipeline import CaseJob

    with tempfile.TemporaryDirectory() as tmp:
        job = CaseJob(case_id, target=1,
                      data_output=Path(tmp) / "data.jsonl",
                      score_output=Path(tmp) / "scores.jsonl")
        asyncio.run(pipeline.run_case(
            records, job,
            on_event=messages.append if messages is not None else None,
            delay=0, concurrency=1,
        ))


def test_matching_records_are_preferred_over_the_rest() -> None:
    """A case must draw only from records stage 2 routed to its family."""
    pipeline = _StubPipeline()
    _run_case(pipeline, "case_C1", [_routed("[case1]", "route-A"), _routed("[case3]", "route-C")])

    assert pipeline.seen, "the case should have attempted something"
    assert set(pipeline.seen) == {"route-C"}, (
        f"case_C1 needs case3 records, but drew {set(pipeline.seen)}"
    )


def test_falling_back_is_announced_when_no_record_matches() -> None:
    """With nothing suitable available, the run says so rather than silently degrading."""
    pipeline = _StubPipeline()
    messages: list[str] = []
    _run_case(pipeline, "case_D9", [_routed("[case1]", "route-A")], messages)

    assert any("case4" in m and "falling back" in m for m in messages), messages
    # It still tries — retrying the only record it has — just loudly.
    assert set(pipeline.seen) == {"route-A"}


# --------------------------------------------------------------------------- #
# Rejection summaries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("5. Tool-RAG consistency check failed", "5. Tool-RAG consistency check failed"),
        ("5. A failed; 6. B failed", "5. A failed"),
        (None, "unknown"),
        ("<reasoning>\nthe model chose a plausible but wrong tool…\n</reasoning>\n<score>\n[0]\n</score>",
         "LLM judge: think/action inconsistency"),
        ("Assistant 2 is inconsistent <score>\n[0]\n</score>", "LLM judge: think/action inconsistency"),
    ],
)
def test_rejection_reasons_are_condensed_for_the_tally(reason: str | None, expected: str) -> None:
    """The judge's full critique must not drown out the rule-check counts."""
    from toolforge.stages.pipeline import summarise_reason

    assert summarise_reason(reason) == expected


def test_long_rule_reasons_are_truncated() -> None:
    from toolforge.stages.pipeline import summarise_reason

    assert len(summarise_reason("x" * 500)) == 90
