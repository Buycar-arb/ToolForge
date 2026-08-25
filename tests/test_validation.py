"""The nine rule checks must reject broken records, not just accept good ones."""

from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_pipeline import generate_one  # noqa: E402
from toolforge.cli import cmd_validate  # noqa: E402
from toolforge.stages.judge import parse_score  # noqa: E402
from toolforge.stages.validation import (  # noqa: E402
    CHECK_LABELS,
    ValidationOptions,
    validate,
)


@pytest.fixture(scope="module")
def good_record() -> list:
    record, _ = asyncio.run(generate_one("case_A2"))
    return record


def _fails(record: list, check: str, case_id: str = "case_A2", **options) -> None:
    outcome = validate(record, case_id, ValidationOptions(**options))
    assert not outcome.passed, f"expected check '{check}' to fail"
    assert CHECK_LABELS[check] in outcome.failures, (
        f"expected '{check}' to fail, got: {outcome.failures}"
    )


def test_baseline_record_passes(good_record: list) -> None:
    assert validate(good_record, "case_A2").passed


def test_1_wrong_role_sequence(good_record: list) -> None:
    broken = copy.deepcopy(good_record)
    broken[1]["messages"].pop()          # drop the final answer turn
    _fails(broken, "format")


def test_2_assistant_without_think_block(good_record: list) -> None:
    broken = copy.deepcopy(good_record)
    broken[1]["messages"][2]["content"] = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
    _fails(broken, "content")


def test_3_empty_tool_message(good_record: list) -> None:
    broken = copy.deepcopy(good_record)
    for message in broken[1]["messages"]:
        if message["role"] == "tool":
            message["content"] = "   "
            break
    _fails(broken, "not_empty")


def test_4_answer_does_not_match_the_gold_answer(good_record: list) -> None:
    broken = copy.deepcopy(good_record)
    broken[1]["messages"][-1]["content"] = "<think>\nx\n</think>\n<answer>\nno\n</answer>"
    _fails(broken, "answer_consistency")


def test_5_tool_message_invents_a_passage(good_record: list) -> None:
    broken = copy.deepcopy(good_record)
    for message in broken[1]["messages"]:
        if message["role"] == "tool":
            message["content"] += "\n**99**\ntitle: Fabricated\ncontent: Never retrieved."
            break
    _fails(broken, "tool_rags_consistency")


def test_6_retry_changes_an_optional_parameter(good_record: list) -> None:
    broken = copy.deepcopy(good_record)
    pairs = broken[3]["argument_check"]
    assert isinstance(pairs, list) and len(pairs) >= 2, "case_A2 must emit paired tool calls"
    definition = pairs[0]["objects"][0]["tool_definition"]
    optional = [
        name for name in definition["parameters"]["properties"]
        if name not in definition["parameters"].get("required", [])
    ]
    if not optional:
        pytest.skip("this sampled tool variant has no optional parameters")
    pairs[1]["objects"][0]["arguments"][optional[0]] = "changed on retry"
    _fails(broken, "arguments")


def test_6b_noop_retry_passes_by_default_and_strict_on_demand(good_record: list) -> None:
    # case_A2's baseline retry carries byte-identical arguments to the failed
    # call, so the published check 6 accepts it (a no-op retry).
    assert validate(good_record, "case_A2").passed
    # Opted in: a retry that changes nothing is incoherent with the failure/
    # recovery narrative and is rejected.
    _fails(good_record, "arguments", require_argument_change=True)


def test_7_reference_check_is_inert_by_default_and_strict_on_demand(good_record: list) -> None:
    broken = copy.deepcopy(good_record)
    broken[4]["argument_all_reference"] = [{"turn": 1, "data": []}]
    # Published behaviour: the check never fired.
    assert validate(broken, "case_A2").passed
    # Opted in: it does.
    _fails(broken, "reference", strict_reference_check=True)


def test_7_strict_references_match_multiple_facts_under_one_title(good_record: list) -> None:
    record = copy.deepcopy(good_record)
    record[4]["argument_all_reference"] = [{
        "turn": 1,
        "data": [
            {"title": "Same title", "content": "First fact."},
            {"title": "Same title", "content": "Second fact."},
        ],
    }]
    record[6]["supporting_facts"] = [["Same title", 0], ["Same title", 1]]
    record[6]["context"] = [["Same title", ["First fact.", "Second fact."]]]

    outcome = validate(record, "case_A2", ValidationOptions(strict_reference_check=True))
    assert outcome.results["reference"] == 1

    record[4]["argument_all_reference"][0]["data"][1]["content"] = "First fact."
    _fails(record, "reference", strict_reference_check=True)


def test_8_calls_a_tool_that_was_never_labelled(good_record: list) -> None:
    broken = copy.deepcopy(good_record)
    broken[2]["good_tool_mapping"] = [
        {"original_tool": "person_information_search", "diversity": "some_other_tool"}
    ]
    _fails(broken, "tool_consistency")


def test_9_tool_call_drops_a_required_parameter(good_record: list) -> None:
    broken = copy.deepcopy(good_record)
    import json
    import re

    def strip_required(content: str) -> str:
        def fix(match: re.Match) -> str:
            call = json.loads(match.group(1))
            if call.get("arguments"):
                call["arguments"].pop(next(iter(call["arguments"])))
            return f"<tool_call>{json.dumps(call)}</tool_call>"

        return re.sub(r"<tool_call>(.*?)</tool_call>", fix, content, flags=re.DOTALL)

    for message in broken[1]["messages"]:
        if message["role"] == "assistant" and "<tool_call>" in message["content"]:
            message["content"] = strip_required(message["content"])
    _fails(broken, "tool_bank")


def test_a_malformed_record_fails_rather_than_crashing() -> None:
    outcome = validate([{"case": "case_A1"}, {}, {}, {}, {}, {}, {}], "case_A1")
    assert not outcome.passed


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("<reasoning>ok</reasoning>\n<score>\n[1]\n</score>", 1),
        ("<score>\n[0]\n</score>", 0),
        ("<score>[1]</score>", 1),
        (
            "<reasoning>candidate <score>[1]</score></reasoning>\n<score>[0]</score>",
            0,
        ),
    ],
)
def test_judge_score_parsing(response: str, expected: int) -> None:
    score, error = parse_score(response)
    assert error is None and score == expected


@pytest.mark.parametrize("response", ["", "no score here", "<score>maybe</score>"])
def test_judge_score_parsing_failure_is_reported(response: str) -> None:
    score, error = parse_score(response)
    assert score is None and error


@pytest.mark.parametrize("contents", ["", "not json\n{still broken\n"])
def test_validate_command_rejects_files_with_no_valid_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: str,
) -> None:
    target = tmp_path / "broken.jsonl"
    target.write_text(contents, encoding="utf-8")
    args = SimpleNamespace(
        input=str(target),
        strict_references=False,
        strict_answer_format=False,
        strict_argument_change=False,
    )

    assert cmd_validate(args) == 1
    captured = capsys.readouterr()
    assert "0/0 records" in captured.out
    assert "No valid records found" in captured.err
