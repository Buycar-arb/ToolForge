"""Parsing the model's replies — the part that breaks when you change models.

The prompt asks for a ```json block. GPT-5 returns the bare object instead, and
other models wrap it in prose. All of it has to work, because a parser that only
accepts one shape silently rejects 100% of the output from a model that prefers
another.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toolforge.stages.dialogue import extract_tags, parse_dialogue_json  # noqa: E402
from toolforge.stages.labeling import LabelParseError, parse_label  # noqa: E402
from toolforge.stages.variants import parse_tool  # noqa: E402

DIALOGUE = {
    "messages": [
        {"role": "user", "content": "the question"},
        {"role": "assistant", "content": "<think>\nthinking\n</think>\n<tool_call>\n{}\n</tool_call>"},
        {"role": "tool", "content": "**1**\ntitle: T\ncontent: C"},
        {"role": "assistant", "content": "<think>\ndone\n</think>\n<answer>\nyes\n</answer>"},
    ]
}
RAW = json.dumps(DIALOGUE, ensure_ascii=False)


@pytest.mark.parametrize(
    ("label", "reply"),
    [
        ("bare object, as GPT-5 returns it", RAW),
        ("```json fence, as the prompt asks", f"```json\n{RAW}\n```"),
        ("plain ``` fence", f"```\n{RAW}\n```"),
        ("prose before", f"Here is the dialogue you asked for:\n\n{RAW}"),
        ("prose after", f"{RAW}\n\nLet me know if you need changes."),
        ("prose either side", f"Sure!\n\n```json\n{RAW}\n```\n\nHope that helps."),
    ],
)
def test_dialogue_json_survives_every_wrapping(label: str, reply: str) -> None:
    messages = parse_dialogue_json(reply)
    assert messages is not None, label
    # The leading user turn is dropped — the engine supplies its own.
    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]


def test_braces_inside_strings_do_not_truncate_the_object() -> None:
    tricky = {
        "messages": [
            {"role": "assistant", "content": 'a { brace, a "quote" and a \\ backslash }'},
        ]
    }
    messages = parse_dialogue_json(json.dumps(tricky, ensure_ascii=False))
    assert messages is not None
    assert messages[0]["content"] == tricky["messages"][0]["content"]


@pytest.mark.parametrize(
    "reply",
    ["", "   ", "Sorry, I cannot help with that.", '{"messages": [', '{"foo": 1}', "```json\nnot json\n```"],
)
def test_unusable_replies_are_rejected_not_guessed(reply: str) -> None:
    assert parse_dialogue_json(reply) is None


def test_tags_are_extracted_in_order() -> None:
    text = "<turn_1><tool_call>a</tool_call><tool_call>b</tool_call></turn_1>"
    turn = extract_tags(text, "turn_1", as_list=False)
    assert extract_tags(turn, "tool_call") == ["a", "b"]
    assert extract_tags(text, "missing") == []


# --------------------------------------------------------------------------- #
# Stage 2 label parsing
# --------------------------------------------------------------------------- #

LABEL = (
    "<think>\nthe reasoning\n</think>\n"
    "工具选择:[person_information_search]\n"
    "路径选择:[case2]"
)


def test_label_parsing() -> None:
    reasoning, tools, route = parse_label(LABEL)
    assert reasoning == "the reasoning"
    assert tools == "[person_information_search]"
    assert route == "[case2]"


@pytest.mark.parametrize(
    "reply",
    ["", "no tags at all", "<think>\nx\n</think>\n工具选择:[a]", "工具选择:[a]\n路径选择:[case1]"],
)
def test_incomplete_labels_raise(reply: str) -> None:
    with pytest.raises(LabelParseError):
        parse_label(reply)


# --------------------------------------------------------------------------- #
# Stage 1 tool parsing
# --------------------------------------------------------------------------- #

TOOL = {"name": "a_search", "description": "does a thing", "parameters": {"type": "object", "properties": {}}}


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps(TOOL),
        f"```json\n{json.dumps(TOOL)}\n```",
        f"```\n{json.dumps(TOOL)}\n```",
        json.dumps({"tool": TOOL}),
        json.dumps([TOOL]),
    ],
)
def test_tool_definitions_are_normalised(reply: str) -> None:
    tool = parse_tool(reply)
    assert tool is not None and tool["name"] == "a_search"


def test_title_is_accepted_as_a_name() -> None:
    tool = parse_tool(json.dumps({"title": "a_search", "description": "x"}))
    assert tool is not None and tool["name"] == "a_search"


@pytest.mark.parametrize("reply", ["", "not json", "{}", json.dumps({"name": "only a name"})])
def test_unusable_tool_replies_are_rejected(reply: str) -> None:
    assert parse_tool(reply) is None


# --------------------------------------------------------------------------- #
# Planned tool calls
#
# The planning prompt asks for one JSON object per <tool_call> block. Models
# mostly oblige, but sometimes pack several into one — which used to abort the
# whole sample with a raw "Extra data" JSONDecodeError.
# --------------------------------------------------------------------------- #

CALL_A = json.dumps({"name": "t", "arguments": {"query": "first", "names": ["x"]}})
CALL_B = json.dumps({"name": "t", "arguments": {"query": "second"}})


@pytest.mark.parametrize(
    ("label", "block", "expected"),
    [
        ("one object, as asked for", CALL_A, ["first"]),
        ("two objects on separate lines", f"{CALL_A}\n{CALL_B}", ["first", "second"]),
        ("two objects, comma separated", f"{CALL_A},\n{CALL_B}", ["first", "second"]),
        ("surrounding whitespace", f"\n  {CALL_A}  \n", ["first"]),
    ],
)
def test_tool_call_blocks_may_hold_several_calls(label: str, block: str, expected: list[str]) -> None:
    from toolforge.stages.dialogue import parse_tool_call_block, retrieval_query

    assert [c["arguments"]["query"] for c in parse_tool_call_block(block)] == expected, label
    # Their queries are joined so retrieval covers everything the turn asks for.
    assert retrieval_query(block) == " ".join(expected)


@pytest.mark.parametrize(
    "block",
    ["", "   ", "not json at all", json.dumps({"name": "t", "arguments": {}}),
     json.dumps({"name": "t"}), json.dumps([1, 2, 3])],
)
def test_unusable_tool_calls_skip_the_sample_cleanly(block: str) -> None:
    """A malformed plan must raise GenerationError, not a raw decode error."""
    from toolforge.stages.dialogue import GenerationError, retrieval_query

    with pytest.raises(GenerationError):
        retrieval_query(block)
