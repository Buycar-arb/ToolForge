"""Stage 3 — turn one labelled multi-hop question into one training dialogue.

This module is the engine that executes every :class:`~toolforge.stages.cases.CaseSpec`.
It replaces the 29 hand-copied processors of the original release; the *only*
thing that varies between cases now lives in :mod:`toolforge.stages.cases`.

The recipe (identical for all 29 cases)
---------------------------------------

1. **Plan** — ask the model for the tool-calling skeleton: per turn, which
   ``<tool_call>`` to make and which ``<reference>`` passages that call should
   surface.  Driven by the planning prompt of the case *family*.
2. **Retrieve** — for each turn, run BM25 over the record's *non*-supporting
   passages to get realistic distractors, then assemble the passage bundles the
   case needs (see :class:`~toolforge.stages.cases.PassageMode`).
3. **Author** — hand the plan plus the bundles to the model and ask it to write
   the finished conversation as JSON, following the case's reasoning flow.
4. **Assemble** — package messages, bundles, references and tool schemas into
   the record layout stage 4 validates.

Output layout
-------------
:meth:`GeneratedSample.to_record` produces the seven-element list written to
disk.  Stage 4 addresses it positionally, so the order is part of the format::

    [0] {"case": ..., "uuid": ...}
    [1] {"messages": [...]}                  the training dialogue
    [2] {"rags": [...], "answer", "reasoning", "good_tool_mapping"}
    [3] {"argument_check": ...}              paired tool calls, or a skip marker
    [4] {"argument_all_reference": [...]}    supporting passages used, per turn
    [5] {"argument_tool_bank": [...]}        every tool schema offered
    [6] {...}                                the original stage 2 record
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import re
import uuid as uuid_module
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolforge import bm25, toolbank
from toolforge.config import Settings
from toolforge.config import settings as default_settings
from toolforge.llm import LLMClient
from toolforge.prompts import agent, flows, planning
from toolforge.prompts import cases as case_prompts
from toolforge.prompts import dialogue as dialogue_prompts
from toolforge.stages.cases import CaseSpec, PassageMode
from toolforge.stages.cases import get as get_case
from toolforge.toolbank import ToolContext

log = logging.getLogger(__name__)

Passage = dict[str, Any]
Message = dict[str, Any]

_PLANNING_SYSTEM = {
    "A": planning.generate_tool_call_system_prompt_A,
    "B": planning.generate_tool_call_system_prompt_B,
    "C": planning.generate_tool_call_system_prompt_C,
    "D": planning.generate_tool_call_system_prompt_D,
}


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def extract_tags(text: str, tag: str, as_list: bool = True) -> list[str] | str:
    """Pull every ``<tag>...</tag>`` block out of ``text``.

    Returns a list of the inner strings, or all of them joined by newlines when
    ``as_list`` is False.
    """
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return matches if as_list else ("\n".join(matches) if matches else "")


def _json_candidates(content: str) -> list[str]:
    """Every plausible JSON payload in a model reply, best guess first.

    Models are inconsistent about fencing: the prompt shows a ```json block and
    some oblige, while others (GPT-5 among them) return the bare object. All
    three shapes are accepted.
    """
    candidates: list[str] = []

    fenced = re.findall(r"```(?:json)?\s*\n(.*?)\n\s*```", content, re.DOTALL)
    candidates.extend(block.strip() for block in fenced)

    # Fall back to the outermost balanced object, skipping braces inside strings.
    start = content.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(content)):
            char = content[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(content[start : index + 1])
                    break
    return candidates


def parse_tool_call_block(block: str) -> list[dict[str, Any]]:
    """Parse a ``<tool_call>`` block into the objects it holds.

    The prompt asks for one JSON object per block, and models usually oblige.
    Sometimes they pack several into a single block instead — accept both rather
    than losing the sample to a "Extra data" decode error.
    """
    decoder = json.JSONDecoder()
    text = block.strip()
    objects: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            objects.append(value)
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
    return objects


def retrieval_query(block: str) -> str:
    """The text to retrieve with for one planned tool call.

    When a block holds several calls their queries are joined, so the passages
    that come back cover everything that turn asks for.
    """
    queries = [
        str(call["arguments"]["query"]).strip()
        for call in parse_tool_call_block(block)
        if isinstance(call.get("arguments"), dict) and call["arguments"].get("query")
    ]
    if not queries:
        raise GenerationError(
            f"a planned tool call carries no arguments.query: {block.strip()[:120]!r}"
        )
    return " ".join(queries)


def parse_dialogue_json(content: str) -> list[Message] | None:
    """Pull the authored ``messages`` array out of a model reply.

    Accepts a ```json block, a bare ``` block, or unfenced JSON. The leading
    ``user`` turn is dropped — the engine supplies its own. Returns ``None``
    when nothing usable is present.
    """
    if not content:
        log.warning("the authoring step returned an empty response")
        return None

    last_error: str | None = None
    for candidate in _json_candidates(content):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if not isinstance(parsed, dict) or "messages" not in parsed:
            last_error = "the JSON payload has no 'messages' array"
            continue
        messages = parsed["messages"]
        if messages and messages[0].get("role") == "user":
            messages = messages[1:]
        return messages

    log.warning(
        "no usable dialogue JSON in the model reply (%s): %s",
        last_error or "nothing that looked like JSON",
        content[:160].replace("\n", " "),
    )
    return None


# --------------------------------------------------------------------------- #
# Source records
# --------------------------------------------------------------------------- #


@dataclass
class SourceRecord:
    """One labelled multi-hop question, as produced by stage 2."""

    question: str
    answer: str
    reasoning: str
    route: str
    #: Passages listed in ``supporting_facts`` — the ones that contain the answer.
    gold_passages: list[Passage]
    #: Every other passage — the BM25 corpus that supplies realistic distractors.
    distractor_passages: list[Passage]
    #: Canonical tool names stage 2 selected for this question.
    gold_tool_names: list[str]
    #: The untouched stage 2 record, carried through to the output.
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, record: dict[str, Any] | str) -> SourceRecord:
        """Build a :class:`SourceRecord` from a stage 2 JSONL row.

        Raises ``KeyError`` if stage 2 never labelled the row (no ``tool_select``).
        """
        data = json.loads(record) if isinstance(record, str) else record

        supporting = {(title, sentence_id) for title, sentence_id in data["supporting_facts"]}
        gold: list[Passage] = []
        rest: list[Passage] = []
        for title, sentences in data["context"]:
            for sentence_id, sentence in enumerate(sentences):
                bucket = gold if (title, sentence_id) in supporting else rest
                bucket.append({"title": title, "content": sentence})

        def dedupe(passages: list[Passage]) -> list[Passage]:
            seen: set[str] = set()
            unique = []
            for passage in passages:
                if passage["content"] not in seen:
                    seen.add(passage["content"])
                    unique.append(passage)
            return unique

        return cls(
            question=data["question"],
            answer=data["answer"],
            reasoning=data["reasoning"],
            route=data["route_select"],
            gold_passages=dedupe(gold),
            distractor_passages=dedupe(rest),
            gold_tool_names=toolbank.parse_tool_select(data["tool_select"]),
            raw=data,
        )


# --------------------------------------------------------------------------- #
# Generated samples
# --------------------------------------------------------------------------- #


@dataclass
class GeneratedSample:
    """A finished dialogue plus everything stage 4 needs to validate it."""

    case_id: str
    uuid: str
    messages: list[Message]
    #: One passage bundle per ``tool`` message, in order.
    rags: list[list[Passage]]
    #: ``[{"turn": n, "data": [...]}, ...]`` — supporting passages used per turn.
    references: list[dict[str, Any]]
    #: Paired tool calls for check #6, or the skip marker.
    argument_check: list[dict[str, Any]] | str
    #: Every tool schema offered to the model in the system prompt.
    offered_tools: list[dict[str, Any]]
    answer: str
    reasoning: str
    gold_tool_mapping: list[dict[str, str]]
    source: dict[str, Any]

    def to_record(self) -> list[dict[str, Any]]:
        """The seven-element list written to the output JSONL (see module docs)."""
        return [
            {"case": self.case_id, "uuid": self.uuid},
            {"messages": self.messages},
            {
                "rags": self.rags,
                "answer": self.answer,
                "reasoning": self.reasoning,
                "good_tool_mapping": self.gold_tool_mapping,
            },
            {"argument_check": self.argument_check},
            {"argument_all_reference": self.references},
            {"argument_tool_bank": self.offered_tools},
            self.source,
        ]


class GenerationError(RuntimeError):
    """The model produced something the engine could not turn into a sample."""


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


class DialogueGenerator:
    """Generates dialogues for any of the 29 cases.

    Parameters
    ----------
    client:
        The LLM used for both the planning and authoring steps.
    tool_bank_dir / config:
        Default to the process settings.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        tool_bank_dir: Path | str | None = None,
        config: Settings | None = None,
    ) -> None:
        self.config = config or default_settings
        self.client = client or LLMClient(config=self.config)
        self.tool_bank_dir = Path(tool_bank_dir or self.config.tool_bank_dir)
        self._definitions: dict[str, dict[str, Any]] | None = None

    @property
    def tool_definitions(self) -> dict[str, dict[str, Any]]:
        """All tool variants in the bank, loaded once and cached."""
        if self._definitions is None:
            self._definitions = toolbank.load_definitions(self.tool_bank_dir)
        return self._definitions

    # ------------------------------------------------------------------ #
    async def generate(
        self,
        record: SourceRecord,
        case_id: str,
        context: ToolContext | None = None,
    ) -> GeneratedSample:
        """Generate one sample, or raise :class:`GenerationError`."""
        spec = get_case(case_id)
        context = context or toolbank.build_context(
            record.gold_tool_names, tool_bank_dir=self.tool_bank_dir, config=self.config
        )

        if spec.use_fallback_tools and not context.fallback_available:
            raise GenerationError(
                f"{case_id} needs a fallback tool, but the gold tool for this record "
                "is general search — skipping."
            )

        tool_prompt = context.tool_prompt_fallback if spec.use_fallback_tools else context.tool_prompt_standard
        offered = context.offered_fallback if spec.use_fallback_tools else context.offered_standard

        plan = await self._plan(record, spec, tool_prompt, context)
        bundles, references = self._build_bundles(record, spec, plan)
        messages = await self._author(record, spec, tool_prompt, context, plan, bundles)

        argument_check: list[dict[str, Any]] | str = "Don't need to check"
        if spec.check_arguments:
            argument_check = self._pair_tool_calls(messages)

        return GeneratedSample(
            case_id=case_id,
            uuid=str(uuid_module.uuid4()),
            messages=messages,
            rags=[bundles[ref] for ref in spec.tool_messages],
            references=references,
            argument_check=argument_check,
            offered_tools=offered,
            answer=record.answer,
            reasoning=record.reasoning,
            gold_tool_mapping=context.gold_mapping,
            source=record.raw,
        )

    # -- step 1: plan the trajectory ------------------------------------ #
    async def _plan(
        self, record: SourceRecord, spec: CaseSpec, tool_prompt: str, context: ToolContext
    ) -> dict[int, dict[str, list[str]]]:
        """Ask the model for the per-turn tool calls and supporting passages."""
        system = _PLANNING_SYSTEM[spec.family] + tool_prompt
        user = planning.generate_tool_call_user_prompt.format(
            query=record.question,
            tools=context.gold_tools,
            reference=record.gold_passages,
            answer=record.answer,
            type=record.route,
            reasoning=record.reasoning,
        )
        trace = await self.client.complete(
            [{"role": "user", "content": [{"type": "text", "text": user}]}], system=system
        )
        if not trace or not trace.strip():
            raise GenerationError("the planning step returned an empty response")

        plan: dict[int, dict[str, list[str]]] = {}
        for turn in range(1, spec.turns + 1):
            block = extract_tags(trace, f"turn_{turn}", as_list=False)
            tool_calls = extract_tags(block, "tool_call")
            references = extract_tags(block, "reference")
            if not tool_calls:
                raise GenerationError(f"the plan has no tool call for turn {turn}")
            if len(references) < len(tool_calls):
                raise GenerationError(
                    f"turn {turn}: {len(references)} reference blocks for "
                    f"{len(tool_calls)} tool calls — the plan is incomplete"
                )
            plan[turn] = {"tool_calls": tool_calls, "references": references}
        return plan

    # -- step 2: retrieve and bundle passages --------------------------- #
    def _build_bundles(
        self, record: SourceRecord, spec: CaseSpec, plan: dict[int, dict[str, list[str]]]
    ) -> tuple[dict[str, list[Passage]], list[dict[str, Any]]]:
        """Assemble the passage bundles this case needs, plus the per-turn references."""
        bundles: dict[str, list[Passage]] = {}
        references: list[dict[str, Any]] = []
        # ``MERGE_INTO_FIRST`` folds its references into turn 1's bundle, so keep
        # the pre-deduplication lists around until every turn has been processed.
        pending: dict[int, list[list[Passage]]] = {}

        for turn, mode in enumerate(spec.passages, start=1):
            tool_calls = plan[turn]["tool_calls"]
            refs = plan[turn]["references"]
            used: list[Passage] = []

            if mode is PassageMode.MERGE_INTO_FIRST:
                target = pending[1]
                for index, raw_reference in enumerate(refs[: len(tool_calls)]):
                    supporting = _parse_reference(raw_reference)
                    used.extend(supporting)
                    target[index].extend(supporting)
                references.append({"turn": turn, "data": used})
                continue

            wide = mode is PassageMode.THREE_STRIKES
            k_min = self.config.rag_top_k_min * (3 if wide else 1)
            k_max = self.config.rag_top_k_max * (3 if wide else 1)

            gold: list[list[Passage]] = []
            bad: list[list[Passage]] = []
            strikes: tuple[list[list[Passage]], ...] = ([], [], [])

            for index, call in enumerate(tool_calls):
                hits = bm25.retrieve(
                    record.distractor_passages, retrieval_query(call), k_min, k_max
                )
                supporting = _parse_reference(refs[index])
                used.extend(supporting)

                if wide:
                    third = len(hits) // 3
                    strikes[2].append(copy.deepcopy(hits[:third]))
                    strikes[1].append(copy.deepcopy(hits[third : 2 * third]))
                    strikes[0].append(copy.deepcopy(hits[2 * third :]))
                    gold.append(copy.deepcopy(hits[:third]))
                else:
                    if mode is PassageMode.GOLD_AND_BAD:
                        bad.append(copy.deepcopy(hits))
                    gold.append(copy.deepcopy(hits))
                gold[index].extend(supporting)

            pending[turn] = gold
            if mode is PassageMode.GOLD_AND_BAD:
                bundles[f"bad@{turn}"] = bm25.deduplicate(bad)
            elif wide:
                for slot, passages in enumerate(strikes, start=1):
                    bundles[f"bad{slot}@{turn}"] = bm25.deduplicate(passages)
            references.append({"turn": turn, "data": used})

        for turn, gold in pending.items():
            bundles[f"gold@{turn}"] = bm25.deduplicate(gold)
        return bundles, references

    # -- step 3: author the conversation -------------------------------- #
    async def _author(
        self,
        record: SourceRecord,
        spec: CaseSpec,
        tool_prompt: str,
        context: ToolContext,
        plan: dict[int, dict[str, list[str]]],
        bundles: dict[str, list[Passage]],
    ) -> list[Message]:
        """Render the case prompt, ask for the dialogue, and prepend system/user."""
        slots: dict[str, Any] = {
            "query": record.question,
            "right_response": record.reasoning,
            "answer": record.answer,
            "flow": flows.CASE_FLOWS[spec.case_id],
        }
        for name, reference in spec.prompt_slots.items():
            slots[name] = self._resolve_slot(reference, plan, bundles, context)

        user = case_prompts.CASE_USER_PROMPTS[spec.case_id].format(**slots)
        response = await self.client.complete(
            [{"role": "user", "content": [{"type": "text", "text": user}]}],
            system=dialogue_prompts.conversation_generate_system_prompt,
        )
        authored = parse_dialogue_json(response)
        if not authored:
            raise GenerationError("the authoring step produced no usable dialogue")

        return [
            {"role": "system", "content": agent.AGENT_SYSTEM_PROMPT + tool_prompt},
            {"role": "user", "content": agent.AGENT_USER_PROMPT.format(record.question)},
            *(message for message in authored if message.get("role") != "user"),
        ]

    @staticmethod
    def _resolve_slot(
        reference: str,
        plan: dict[int, dict[str, list[str]]],
        bundles: dict[str, list[Passage]],
        context: ToolContext,
    ) -> Any:
        """Turn a slot reference from the case spec into the value to interpolate."""
        if reference == "distractors":
            return context.distractors_standard
        if reference == "distractors_fallback":
            return context.distractors_fallback
        if reference == "general_tool":
            return context.general_tool
        kind, _, turn = reference.partition("@")
        if kind == "plan":
            return plan[int(turn)]["tool_calls"]
        return bundles[reference]

    # -- step 4: pair up tool calls for check #6 ------------------------ #
    def _pair_tool_calls(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Collect the tool calls of every assistant turn except the final answer.

        Stage 4 check #6 walks these pairwise: a retry may only change parameters
        listed in the tool's ``required`` array.
        """
        assistants = [message for message in messages if message.get("role") == "assistant"]
        if len(assistants) <= 1:
            return []

        grouped: list[dict[str, Any]] = []
        for index, assistant in enumerate(assistants[:-1]):
            calls: list[dict[str, Any]] = []
            for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", assistant.get("content", ""), re.DOTALL):
                try:
                    call = json.loads(raw.strip())
                except json.JSONDecodeError:
                    log.warning("assistant turn %d has an unparseable tool call", index + 1)
                    continue
                call["tool_definition"] = self.tool_definitions.get(call.get("name"))
                if call["tool_definition"] is None:
                    log.warning("no definition in the tool bank for '%s'", call.get("name"))
                calls.append(call)
            if calls:
                grouped.append({"assistant_index": index + 1, "objects": calls})
        return grouped


def _parse_reference(raw: str) -> list[Passage]:
    """Parse a ``<reference>`` block, which the model emits as a Python literal."""
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise GenerationError(f"unparseable <reference> block: {exc}") from None
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
