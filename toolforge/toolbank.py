"""The tool bank: 22 domain tool libraries, and how a prompt context is sampled from them.

Directory layout — one JSONL file per *domain*, one JSON tool definition per line::

    tool_bank/
      person_information_search.jsonl      <- 20+ paraphrased variants of the same tool
      medical_information_search.jsonl
      general_information_search.jsonl     <- the fallback tool
      ...

The file **stem** is the canonical tool name that Stage 2 labels refer to; the
individual lines are *variants* produced by Stage 1.  Sampling a different
variant on every record is what stops the fine-tuned model from memorising tool
names instead of learning to read tool descriptions.

:func:`build_context` turns a label ("this question needs
``person_information_search``") into everything Stage 3 needs to render a prompt.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

from toolforge.config import GENERAL_TOOL_STEM, Settings
from toolforge.config import settings as default_settings

Tool = dict[str, Any]

#: Rendered into the system prompt so the model knows the calling convention.
TOOL_PROMPT_TEMPLATE = Template(
    """# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:​
<tools>
$recall_tools
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
)


# --------------------------------------------------------------------------- #
# Reading the bank
# --------------------------------------------------------------------------- #


def load_variants(path: Path) -> list[Tool]:
    """Every tool definition inside one domain file."""
    tools: list[Tool] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tools.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[toolbank] skipping {path.name}:{line_number} ({exc})")
    return tools


def sample_variant(path: Path) -> Tool | None:
    """Pick one random variant from a domain file."""
    variants = load_variants(path)
    if not variants:
        print(f"[toolbank] {path.name} has no usable tool definitions")
        return None
    return random.choice(variants)


def domain_files(tool_bank_dir: Path) -> list[Path]:
    """All domain files in the bank, sorted for reproducible iteration."""
    folder = Path(tool_bank_dir)
    if not folder.is_dir():
        print(f"[toolbank] tool bank directory not found: {folder}")
        return []
    return sorted(p for p in folder.glob("*.jsonl") if p.is_file())


def domain_names(tool_bank_dir: Path) -> list[str]:
    """Canonical tool names (file stems) available in the bank."""
    return [p.stem for p in domain_files(tool_bank_dir)]


def load_definitions(tool_bank_dir: Path) -> dict[str, Tool]:
    """Map every *variant* name to its definition, across all domain files.

    Used by the argument-consistency check, which needs to look up the schema of
    whatever variant name the model actually emitted in a ``<tool_call>``.
    """
    definitions: dict[str, Tool] = {}
    for path in domain_files(tool_bank_dir):
        for tool in load_variants(path):
            name = tool.get("name")
            if name:
                definitions[name] = tool
            else:
                print(f"[toolbank] a tool in {path.name} has no 'name' field")
    return definitions


def describe_bank(tool_bank_dir: Path) -> dict[str, str]:
    """``{tool name: description}`` taken from the first variant of each domain."""
    summary: dict[str, str] = {}
    for path in domain_files(tool_bank_dir):
        variants = load_variants(path)
        summary[path.stem] = variants[0].get("description", "(no description)") if variants else "(empty file)"
    return summary


def parse_tool_select(raw: str | Iterable[str]) -> list[str]:
    """Normalise Stage 2's ``tool_select`` field into a list of tool names.

    Stage 2 emits it as a bracketed string, e.g. ``"[person_information_search]"``
    or ``"[a, b]"``.  Lists are passed through unchanged.
    """
    if not isinstance(raw, str):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [name for name in raw.strip().strip("[]").replace(" ", "").split(",") if name]


# --------------------------------------------------------------------------- #
# Sampling a prompt context
# --------------------------------------------------------------------------- #


@dataclass
class ToolContext:
    """Everything Stage 3 needs about tools for one source record.

    Two parallel tool sets are prepared:

    ``*_standard``
        distractor tools + the gold tool(s).  Used by the 23 cases where the
        model is expected to solve the query with a specialised tool.

    ``*_fallback``
        the same, plus the ``general_information_search`` tool.  Used by the six
        "fallback" cases (A4, B6, C9, C10, D9, D10) where the specialised tool
        must fail and the model must fall back to general search.
        :attr:`fallback_available` is False when the gold tool *is* the general
        tool, which makes those cases meaningless for this record.
    """

    #: Definitions of the gold tool variants selected for this record.
    gold_tools: list[Tool] = field(default_factory=list)
    #: ``[{"original_tool": <file stem>, "diversity": <variant name>}, ...]``
    gold_mapping: list[dict[str, str]] = field(default_factory=list)

    #: Rendered ``# Tools`` block for the standard tool set.
    tool_prompt_standard: str = ""
    #: Distractor tools only (gold tools excluded) — the pool of "wrong" tools.
    distractors_standard: list[Tool] = field(default_factory=list)
    #: Every tool offered in the standard prompt, as JSON — used by check #9.
    offered_standard: list[Tool] = field(default_factory=list)

    #: Rendered ``# Tools`` block for the fallback tool set.
    tool_prompt_fallback: str = ""
    #: Every tool offered in the fallback prompt (gold + distractors + general).
    distractors_fallback: list[Tool] = field(default_factory=list)
    #: Every tool offered in the fallback prompt, as JSON — used by check #9.
    offered_fallback: list[Tool] = field(default_factory=list)
    #: The ``general_information_search`` variant used in the fallback set.
    general_tool: Tool | None = None
    #: False when the gold tool is itself the general tool.
    fallback_available: bool = True


def _render(tools: Sequence[Tool]) -> str:
    return TOOL_PROMPT_TEMPLATE.substitute(recall_tools="".join(str(tool) for tool in tools))


def build_context(
    gold_tool_names: Sequence[str],
    *,
    tool_bank_dir: Path | str | None = None,
    virtual_tool_min: int | None = None,
    virtual_tool_max: int | None = None,
    config: Settings | None = None,
) -> ToolContext:
    """Sample the tool sets shown to the model for one source record.

    One variant is drawn per domain file; ``virtual_tool_min..max`` of the
    non-gold variants become distractors.  Everything is shuffled so the gold
    tool never sits at a fixed position.
    """
    cfg = config or default_settings
    bank = Path(tool_bank_dir or cfg.tool_bank_dir)
    low = cfg.virtual_tool_min if virtual_tool_min is None else virtual_tool_min
    high = cfg.virtual_tool_max if virtual_tool_max is None else virtual_tool_max

    gold_stems = set(gold_tool_names)
    general_path = bank / f"{GENERAL_TOOL_STEM}.jsonl"

    pool: list[Tool] = []           # one variant from every non-gold domain
    gold_tools: list[Tool] = []
    gold_mapping: list[dict[str, str]] = []
    general_from_pool: Tool | None = None
    gold_is_general = False

    for path in domain_files(bank):
        tool = sample_variant(path)
        if tool is None:
            continue
        if path.stem in gold_stems:
            gold_tools.append(tool)
            gold_mapping.append({"original_tool": path.stem, "diversity": tool.get("name", "Unknown")})
            if path.stem == GENERAL_TOOL_STEM:
                gold_is_general = True
        else:
            pool.append(tool)
            if path.stem == GENERAL_TOOL_STEM:
                general_from_pool = tool

    missing = gold_stems - {p.stem for p in domain_files(bank)}
    if missing:
        print(f"[toolbank] gold tools missing from the bank: {sorted(missing)}")

    context = ToolContext(gold_tools=gold_tools, gold_mapping=gold_mapping)
    how_many = random.randint(low, high)

    # -- standard set: distractors + gold ---------------------------------- #
    distractors = random.sample(pool, min(how_many, len(pool)))
    context.distractors_standard = copy.deepcopy(distractors)
    offered = distractors + gold_tools
    random.shuffle(offered)
    context.offered_standard = list(offered)
    context.tool_prompt_standard = _render(offered)

    # -- fallback set: distractors + gold + general ------------------------ #
    if gold_is_general:
        # The gold tool *is* general search, so there is nothing to fall back to.
        context.fallback_available = False
        return context

    general_tool = general_from_pool or sample_variant(general_path)
    if general_tool is None:
        context.fallback_available = False
        return context

    fallback = random.sample(pool, min(how_many, len(pool))) + gold_tools + [general_tool]
    random.shuffle(fallback)
    context.general_tool = general_tool
    context.distractors_fallback = copy.deepcopy(fallback)
    context.offered_fallback = list(fallback)
    context.tool_prompt_fallback = _render(fallback)
    return context
