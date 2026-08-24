"""Stage 4 — rule-based validation of a generated dialogue.

Nine independent checks run over the seven-element record described in
:mod:`toolforge.stages.dialogue`.  A sample scores ``rule_score = 1`` only when
all nine pass; otherwise it is rejected and the failing checks are recorded.

The checks
----------

===  ====================================================================
 #   what it verifies
===  ====================================================================
 1   the role sequence matches the case's expected pattern
 2   assistant turns are ``<think>`` + ``<tool_call>``, ending with ``<answer>``
 3   no non-assistant message has empty content
 4   the final ``<answer>`` equals the gold answer
 5   each ``tool`` message renders exactly its passage bundle
 6   a retried tool call only changes *required* parameters
 7   the passages used equal the record's ``supporting_facts``
 8   the tools actually called match what stage 2 labelled
 9   every tool call matches a schema that was offered in the system prompt
===  ====================================================================

Two behaviours are inherited verbatim from the published release and are
**opt-in to change**, so the default reproduces the paper's numbers:

* Check 2 does not reject a malformed *final* answer block (it only warns).
  Enable :attr:`ValidationOptions.strict_final_answer_format` to enforce it.
* Check 7 read the wrong record slots in the original code and therefore always
  passed.  Enable :attr:`ValidationOptions.strict_reference_check` to run it for
  real.  Expect a lower pass rate when you do.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from toolforge.stages.cases import CASE_SPECS, CaseSpec, ToolPolicy

Record = Sequence[Any]

#: Record slot indices — see :mod:`toolforge.stages.dialogue` for the layout.
HEADER, MESSAGES, META, ARGUMENTS, REFERENCES, TOOL_BANK, SOURCE = range(7)

#: Emitted for cases where check 6 does not apply.
SKIP_ARGUMENT_CHECK = "Don't need to check"

CHECK_LABELS: dict[str, str] = {
    "format": "1. Dialogue format validation failed",
    "content": "2. Assistant content format validation failed",
    "not_empty": "3. Non-assistant field empty validation failed",
    "answer_consistency": "4. Answer consistency check failed",
    "tool_rags_consistency": "5. Tool-RAG consistency check failed",
    "arguments": "6. Argument validation failed",
    "reference": "7. Reference error at one or more stages",
    "tool_consistency": "8. Predefined tool count mismatch or inconsistent usage order",
    "tool_bank": "9. Mismatch between tool_call names/arguments and tool_bank definitions",
}


def expected_roles(spec: CaseSpec) -> list[str]:
    """The role sequence a case must produce.

    ``system, user`` then one ``assistant, tool`` pair per served passage bundle,
    then the final ``assistant`` answer.
    """
    roles = ["system", "user"]
    for _ in spec.tool_messages:
        roles += ["assistant", "tool"]
    return roles + ["assistant"]


#: ``case id -> expected role sequence``, derived from the case specs.
DIALOGUE_PATTERNS: dict[str, list[str]] = {
    case_id: expected_roles(spec) for case_id, spec in CASE_SPECS.items()
}


@dataclass
class ValidationOptions:
    """Toggles for the two checks that were inert in the published release."""

    #: Enforce ``<think>...</think>\\n<answer>...</answer>`` on the final turn.
    strict_final_answer_format: bool = False
    #: Actually run check 7 (it read the wrong record slots originally).
    strict_reference_check: bool = False


@dataclass
class ValidationResult:
    """Outcome of running all nine checks."""

    results: dict[str, int]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def reason(self) -> str:
        return "; ".join(self.failures)


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _messages(record: Record) -> list[dict[str, Any]]:
    block = record[MESSAGES]
    if isinstance(block, str):
        block = json.loads(block)
    return block["messages"]


def _passage_key(passage: dict[str, str]) -> tuple[str, str]:
    """A comparable form of a passage, ignoring whitespace and punctuation."""
    return _normalise(passage["title"]), _normalise(passage["content"])


def _normalise(text: str) -> str:
    """Lowercase, letters and digits only — used to compare rendered passages."""
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def check_format(record: Record, case_id: str, _options: ValidationOptions) -> int:
    """1. The role sequence matches the case's expected pattern."""
    try:
        messages = _messages(record)
    except Exception:
        return 0
    expected = DIALOGUE_PATTERNS[case_id]
    if len(messages) != len(expected):
        return 0
    return int(all(message.get("role") == role for message, role in zip(messages, expected, strict=True)))


def check_content(record: Record, _case_id: str, options: ValidationOptions) -> int:
    """2. Assistant turns use ``<think>`` + ``<tool_call>``, the last one ``<answer>``."""
    try:
        messages = _messages(record)
    except Exception:
        return 0

    assistants = [m for m in messages if m.get("role") == "assistant"]
    if not assistants:
        return 0

    for index, message in enumerate(assistants):
        content = (message.get("content") or "").strip()
        is_final = index == len(assistants) - 1

        if is_final:
            final_pattern = r"^<think>\s*.*?\s*</think>\s*\n\s*<answer>\s*.*?\s*</answer>\s*$"
            if not re.match(final_pattern, content, re.DOTALL):
                # The published pipeline only warned here; see the module docstring.
                if options.strict_final_answer_format:
                    return 0
                print(f"[validation] final assistant turn is not <think>+<answer>: {content[:80]!r}")
            continue

        match = re.match(r"^<think>\s*.*?\s*</think>\s*\n\s*(.*?)$", content, re.DOTALL)
        if not match:
            return 0
        remainder = match.group(1).strip()
        if not remainder:
            return 0
        if not re.match(r"^(<tool_call>\s*.*?\s*</tool_call>\s*)+$", remainder, re.DOTALL):
            return 0
    return 1


def check_not_empty(record: Record, _case_id: str, _options: ValidationOptions) -> int:
    """3. Every ``system`` / ``user`` / ``tool`` message has content."""
    try:
        messages = _messages(record)
    except Exception:
        return 0
    for message in messages:
        if message.get("role") != "assistant" and not (message.get("content") or "").strip():
            return 0
    return 1


def check_answer_consistency(record: Record, _case_id: str, _options: ValidationOptions) -> int:
    """4. The final ``<answer>`` matches the gold answer, case-insensitively."""
    try:
        messages = _messages(record)
        expected = str(record[META]["answer"]).strip()
    except Exception:
        return 0

    final = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if final is None:
        return 0
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", final.get("content", ""), re.DOTALL)
    if not match:
        return 0
    return int(match.group(1).strip().lower() == expected.lower())


def check_tool_rags_consistency(record: Record, _case_id: str, _options: ValidationOptions) -> int:
    """5. Each ``tool`` message renders exactly its passage bundle, nothing added or dropped."""
    try:
        messages = _messages(record)
        bundles = record[META]["rags"]
    except Exception:
        return 0

    rendered = [m.get("content", "") for m in messages if m.get("role") == "tool"]
    if len(rendered) != len(bundles):
        return 0

    pattern = r"\*\*(\d+)\*\*\s*\ntitle:\s*(.*?)\s*\ncontent:\s*(.*?)(?=\n\*\*\d+\*\*|\Z)"
    for content, bundle in zip(rendered, bundles, strict=True):
        parsed = [
            {"title": title.strip(), "content": body.strip()}
            for _number, title, body in re.findall(pattern, content, re.DOTALL)
        ]
        if len(parsed) != len(bundle):
            return 0
        if {_passage_key(p) for p in parsed} != {_passage_key(p) for p in bundle}:
            return 0
    return 1


def check_arguments(record: Record, case_id: str, _options: ValidationOptions) -> int:
    """6. A retried tool call may only change parameters listed in ``required``."""
    block = record[ARGUMENTS]
    if not block or block.get("argument_check") == SKIP_ARGUMENT_CHECK:
        return 1

    spec = CASE_SPECS.get(case_id)
    if spec is None or spec.argument_check_range is None:
        return 1

    pairs = block["argument_check"]
    if len(pairs) < 2:
        print("[validation] not enough assistant turns to compare arguments")
        return 1

    start, stop = spec.argument_check_range
    for index in range(start, stop, 2):
        if index + 1 >= len(pairs):
            break
        first, second = pairs[index], pairs[index + 1]
        before, after = first["objects"], second["objects"]

        if len(before) != len(after):
            print(f"[validation] tool call count changed: {len(before)} -> {len(after)}")
            return 0

        for call_before, call_after in zip(before, after, strict=True):
            if call_before.get("name") != call_after.get("name"):
                print(
                    f"[validation] tool name changed: "
                    f"{call_before.get('name')} -> {call_after.get('name')}"
                )
                return 0

            args_before = call_before.get("arguments", {})
            args_after = call_after.get("arguments", {})
            changed = [
                key
                for key in set(args_before) | set(args_after)
                if args_before.get(key) != args_after.get(key)
            ]

            definition = call_before.get("tool_definition")
            if not definition:
                print(f"[validation] assistant {first['assistant_index']} has no tool definition")
                return 0

            required = definition.get("parameters", {}).get("required", [])
            for key in changed:
                if key not in required:
                    print(f"[validation] retry changed the optional parameter '{key}'")
                    return 0
    return 1


def check_reference(record: Record, _case_id: str, options: ValidationOptions) -> int:
    """7. The supporting passages actually used equal the record's ``supporting_facts``."""
    if not options.strict_reference_check:
        # The published release read the wrong slots here, so this always passed.
        return 1
    try:
        used = [
            passage
            for turn in record[REFERENCES].get("argument_all_reference", [])
            for passage in turn.get("data", [])
        ]
        supporting = record[SOURCE].get("supporting_facts", [])
        context = record[SOURCE].get("context", [])
    except Exception as exc:
        print(f"[validation] reference check could not read the record: {exc}")
        return 0

    if len(used) != len(supporting):
        return 0

    sentences_by_title = {
        item[0]: (item[1] if isinstance(item[1], list) else [item[1]])
        for item in context
        if len(item) >= 2
    }

    for passage in used:
        title = passage.get("title", "")
        fact = next((f for f in supporting if len(f) >= 2 and f[0] == title), None)
        if fact is None:
            return 0
        sentence_id = fact[1]
        sentences = sentences_by_title.get(title)
        if sentences is None or sentence_id >= len(sentences):
            return 0
        if passage.get("content", "").strip() != sentences[sentence_id].strip():
            return 0

    return int({p.get("title", "") for p in used} == {f[0] for f in supporting if f})


def check_tool_consistency(record: Record, case_id: str, _options: ValidationOptions) -> int:
    """8. The tools called match the tools stage 2 labelled for this question.

    Three regimes, keyed off the case:

    The regime comes from the case's :class:`~toolforge.stages.cases.ToolPolicy`:
    ``ALLOW_FEWER`` (``case_D2``), ``ALLOW_EXTRA`` (the 15 cases that deliberately
    call a wrong tool first), or ``EXACT`` for everything else.
    """
    try:
        messages = record[MESSAGES].get("messages", [])
        mapping = record[META].get("good_tool_mapping", [])
        labelled = record[SOURCE].get("tool_select", "")
    except Exception as exc:
        print(f"[validation] tool consistency check could not read the record: {exc}")
        return 0

    called: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw in re.findall(r"<tool_call>(.*?)</tool_call>", message.get("content", ""), re.DOTALL):
            try:
                name = json.loads(raw.strip()).get("name", "")
            except json.JSONDecodeError:
                return 0
            if name and name not in called:
                called.append(name)

    if not (labelled.startswith("[") and labelled.endswith("]")):
        order: list[str] = []
    else:
        order = [name.strip() for name in labelled[1:-1].split(",")]

    variants = {entry.get("original_tool"): entry.get("diversity", "") for entry in mapping}
    expected = [variants[name] for name in order if variants.get(name)]

    spec = CASE_SPECS.get(case_id)
    policy = spec.tool_policy if spec else ToolPolicy.EXACT
    if policy is ToolPolicy.ALLOW_FEWER:
        return int(bool(called) and called[0] in expected)
    if policy is ToolPolicy.ALLOW_EXTRA:
        return int(all(name in called for name in expected))
    return int(called == expected)


def check_tool_bank(record: Record, _case_id: str, _options: ValidationOptions) -> int:
    """9. Every tool call names a tool that was offered, with a valid argument set."""
    try:
        offered = {
            tool["name"]: (
                set(tool["parameters"]["properties"]),
                set(tool["parameters"]["required"]),
            )
            for tool in record[TOOL_BANK]["argument_tool_bank"]
        }
        messages = record[MESSAGES]["messages"]
    except Exception as exc:
        print(f"[validation] tool bank check could not read the record: {exc}")
        return 0

    seen: list[tuple[str, frozenset[str]]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw in re.findall(r"<tool_call>(.*?)</tool_call>", message.get("content", ""), re.DOTALL):
            try:
                call = json.loads(raw.strip())
            except json.JSONDecodeError:
                return 0
            entry = (call.get("name", ""), frozenset(call.get("arguments", {})))
            if entry not in seen:
                seen.append(entry)

    valid = True
    for name, arguments in seen:
        if name not in offered:
            print(f"[validation] tool '{name}' was never offered in the system prompt")
            valid = False
            continue
        properties, required = offered[name]
        missing = required - arguments
        if missing:
            print(f"[validation] '{name}' is missing required parameters: {sorted(missing)}")
            valid = False
            continue
        unknown = (arguments - required) - properties
        if unknown:
            print(f"[validation] '{name}' got unknown parameters: {sorted(unknown)}")
            valid = False
    return int(valid)


#: The nine checks, in the order they run.
CHECKS: tuple[tuple[str, Callable[[Record, str, ValidationOptions], int]], ...] = (
    ("format", check_format),
    ("content", check_content),
    ("not_empty", check_not_empty),
    ("answer_consistency", check_answer_consistency),
    ("tool_rags_consistency", check_tool_rags_consistency),
    ("arguments", check_arguments),
    ("reference", check_reference),
    ("tool_consistency", check_tool_consistency),
    ("tool_bank", check_tool_bank),
)


def validate(record: Record, case_id: str, options: ValidationOptions | None = None) -> ValidationResult:
    """Run all nine checks and collect the failures."""
    options = options or ValidationOptions()
    results: dict[str, int] = {}
    failures: list[str] = []

    for name, check in CHECKS:
        try:
            outcome = check(record, case_id, options)
        except Exception as exc:  # a broken record fails the check, never the run
            print(f"[validation] check '{name}' raised {exc!r}")
            outcome = 0
        results[name] = outcome
        if outcome == 0:
            label = CHECK_LABELS[name]
            print(label)
            failures.append(label)

    return ValidationResult(results=results, failures=failures)
