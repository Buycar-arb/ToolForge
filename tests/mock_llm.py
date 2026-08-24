"""A scripted stand-in for :class:`toolforge.llm.LLMClient`.

It lets the whole stage 3 + 4 pipeline run offline: the planning call returns a
well-formed trajectory built from the record's own gold passages, and the
authoring call returns a dialogue whose ``tool`` messages render exactly the
passage bundles the engine just assembled.  That makes every rule check
meaningful — a regression in bundling, ordering or rendering fails the test.
"""

from __future__ import annotations

import json
from typing import Any

from toolforge.prompts.dialogue import conversation_generate_system_prompt
from toolforge.stages.dialogue import DialogueGenerator


def render_tool_message(passages: list[dict[str, str]]) -> str:
    """Render a passage bundle the way the training data expects."""
    return "\n".join(
        f"**{index}**\ntitle: {passage['title']}\ncontent: {passage['content']}"
        for index, passage in enumerate(passages, 1)
    )


class ScriptedGenerator(DialogueGenerator):
    """A generator that remembers the plan and bundles for the mock client."""

    last_plan: dict[int, dict[str, list[str]]] | None = None
    last_bundles: dict[str, list[dict[str, str]]] | None = None
    last_spec: Any = None

    async def _author(self, record, spec, tool_prompt, context, plan, bundles):
        self.last_plan, self.last_bundles, self.last_spec = plan, bundles, spec
        return await super()._author(record, spec, tool_prompt, context, plan, bundles)


class MockLLMClient:
    """Answers the two prompts the engine sends, and nothing else."""

    def __init__(self, record, context, turns: int = 1) -> None:
        #: Set by :func:`make_generator` once the generator exists.
        self.generator: ScriptedGenerator | None = None
        self.record = record
        self.context = context
        self.turns = turns
        self.calls: list[str] = []

    # the engine only ever calls .complete()
    async def complete(self, messages, *, system: str | None = None, **_kwargs: Any) -> str:
        if system == conversation_generate_system_prompt:
            self.calls.append("author")
            return self._author_response()
        self.calls.append("plan")
        return self._plan_response()

    # -- planning ------------------------------------------------------- #
    def _plan_response(self) -> str:
        """One tool call per turn, with the gold passages split across turns."""
        gold_tool = self.context.gold_tools[0]
        required = gold_tool["parameters"]["required"]
        call = json.dumps(
            {"name": gold_tool["name"], "arguments": dict.fromkeys(required, "probe query")},
            ensure_ascii=False,
        )

        passages = self.record.gold_passages
        if self.turns == 1:
            per_turn = [passages]
        else:
            split = max(1, len(passages) // 2)
            per_turn = [passages[:split], passages[split:]]

        blocks = []
        for index, chunk in enumerate(per_turn, 1):
            blocks.append(
                f"<turn_{index}>\n<tool_call>\n{call}\n</tool_call>\n"
                f"<reference>\n{chunk!r}\n</reference>\n</turn_{index}>"
            )
        return "\n".join(blocks)

    # -- authoring ------------------------------------------------------ #
    def _author_response(self) -> str:
        spec = self.generator.last_spec
        plan = self.generator.last_plan
        bundles = self.generator.last_bundles
        assert spec and plan and bundles, "the engine must set the recording hooks first"

        messages: list[dict[str, str]] = []
        for reference in spec.tool_messages:
            turn = int(reference.rsplit("@", 1)[1])
            call = plan[turn]["tool_calls"][0]
            messages.append(
                {"role": "assistant", "content": f"<think>\nprobing turn {turn}\n</think>\n<tool_call>\n{call}\n</tool_call>"}
            )
            messages.append({"role": "tool", "content": render_tool_message(bundles[reference])})
        messages.append(
            {"role": "assistant", "content": f"<think>\nenough evidence\n</think>\n<answer>\n{self.record.answer}\n</answer>"}
        )
        return "```json\n" + json.dumps({"messages": messages}, ensure_ascii=False) + "\n```"


def make_generator(record, context, turns: int) -> ScriptedGenerator:
    """Wire a scripted generator and its mock client together."""
    client = MockLLMClient(record, context, turns)
    generator = ScriptedGenerator(client)  # type: ignore[arg-type]
    client.generator = generator
    return generator
