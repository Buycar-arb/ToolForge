"""The 29 dialogue cases, declared as data instead of code.

Every case follows the *same* three-step recipe (plan a trajectory → retrieve
passages → author the dialogue).  They differ only in a handful of knobs, so
each one is a :class:`CaseSpec` and a single engine
(:mod:`toolforge.stages.dialogue`) executes all of them.

The knobs
---------

``family``
    Which planning system prompt to use — ``A``/``B`` (single turn) or
    ``C``/``D`` (two turns).  See :mod:`toolforge.prompts.planning`.

    The family is not free to choose per record: it must agree with the routing
    class stage 2 assigned, or the planning prompt receives contradictory
    instructions ("this question needs one call" *and* "produce several calls").
    :data:`ROUTE_TO_FAMILY` records the correspondence.

``turns``
    How many ``<turn_N>`` blocks the planner must produce.

``use_fallback_tools``
    True for the six cases where the model must give up on the specialised tool
    and fall back to ``general_information_search``.  These use the *fallback*
    tool set, which additionally offers the general tool.

``passages``
    Per turn, which passage bundles to build.  See :class:`PassageMode`.

``prompt_slots``
    Maps a placeholder in the case user prompt to the bundle that fills it.
    ``"gold_content_1" -> "gold@1"`` means "the gold bundle of turn 1".

``tool_messages``
    The bundles served to the model as ``tool`` messages, **in order**.  This
    order is load-bearing: stage 4 check #5 compares tool message *i* against
    ``rags[i]``.

``tool_policy``
    How strictly stage 4 check #8 compares the tools actually called against the
    tools stage 2 labelled — see :class:`ToolPolicy`.

``argument_check_range``
    ``(start, stop)`` window over the tool-calling assistant turns that stage 4
    check #6 compares pairwise (a retry may only change *required* parameters).
    ``None`` disables the check for this case.

Naming
------
A bundle reference is ``"<kind>@<turn>"``:

``gold@1``
    turn 1 retrieval **plus** the supporting passages — the call that succeeds.
``bad@1``
    turn 1 retrieval **without** them — the call that fails and triggers a retry.
``bad1@2`` / ``bad2@2`` / ``bad3@2``
    three disjoint slices of a 3x-wider turn-2 retrieval, for the fallback cases
    where the model fails repeatedly before switching to general search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ToolPolicy(str, Enum):
    """How check #8 compares the called tools against the labelled ones."""

    #: The called tools must equal the labelled tools, in order.
    EXACT = "exact"
    #: Every labelled tool must be called; extra (wrong) calls are expected.
    #: Applies to cases whose flow deliberately calls the wrong tool first.
    ALLOW_EXTRA = "allow_extra"
    #: Only the first call must be a labelled tool — the case answers with fewer
    #: calls than the label implies.  ``case_D2`` only.
    ALLOW_FEWER = "allow_fewer"


class PassageMode(str, Enum):
    """How a turn's retrieved passages are split into bundles."""

    #: One bundle: retrieval + supporting passages.  The call just works.
    GOLD_ONLY = "gold_only"
    #: Two bundles: ``bad`` (retrieval only) and ``gold`` (retrieval + support).
    #: Models a first attempt that misses, then a corrected one that hits.
    GOLD_AND_BAD = "gold_and_bad"
    #: Four bundles from a 3x-wider retrieval: three disjoint ``bad`` slices plus
    #: a ``gold`` one.  Models three failed attempts before the fallback.
    THREE_STRIKES = "three_strikes"
    #: No retrieval of its own — this turn's supporting passages are folded into
    #: turn 1's bundle.  Only ``case_D2``, where the model answers a two-hop
    #: question after a single tool call.
    MERGE_INTO_FIRST = "merge_into_first"


@dataclass(frozen=True)
class CaseSpec:
    """Declarative description of one dialogue case."""

    case_id: str
    family: str
    turns: int
    passages: tuple[PassageMode, ...]
    prompt_slots: dict[str, str]
    tool_messages: tuple[str, ...]
    use_fallback_tools: bool = False
    tool_policy: ToolPolicy = ToolPolicy.EXACT
    argument_check_range: tuple[int, int] | None = None
    description: str = ""

    #: Extra scalar slots the prompt may reference, resolved by the engine.
    static_slots: tuple[str, ...] = field(default=("query", "right_response", "answer", "flow"))

    def __post_init__(self) -> None:
        if len(self.passages) != self.turns:
            raise ValueError(f"{self.case_id}: {self.turns} turns but {len(self.passages)} passage modes")

    @property
    def source_route(self) -> str:
        """The stage 2 routing class this case expects its source record to have."""
        return FAMILY_TO_ROUTE[self.family]

    @property
    def check_arguments(self) -> bool:
        """Whether stage 4 check #6 applies to this case."""
        return self.argument_check_range is not None

    @property
    def group(self) -> str:
        """``A``, ``B``, ``C`` or ``D``."""
        return self.case_id.removeprefix("case_")[0]


#: Stage 2's routing class -> the case family that fits it.  A record labelled
#: ``case2`` (one turn, several calls) can only sensibly produce a B-family
#: dialogue, and so on.
ROUTE_TO_FAMILY: dict[str, str] = {
    "case1": "A",   # one turn, one call
    "case2": "B",   # one turn, several calls
    "case3": "C",   # two turns, one call each
    "case4": "D",   # two turns, several calls in at least one
}

#: The reverse: which routing class a family expects to be given.
FAMILY_TO_ROUTE: dict[str, str] = {family: route for route, family in ROUTE_TO_FAMILY.items()}


def normalise_route(raw: str) -> str:
    """Turn stage 2's ``"[case3]"`` into ``"case3"``."""
    return (raw or "").strip().strip("[]").strip().lower()


def family_for_route(raw: str) -> str | None:
    """The case family that suits a stage 2 route label, if it is a known one."""
    return ROUTE_TO_FAMILY.get(normalise_route(raw))


G, GB, S3, MERGE = (
    PassageMode.GOLD_ONLY,
    PassageMode.GOLD_AND_BAD,
    PassageMode.THREE_STRIKES,
    PassageMode.MERGE_INTO_FIRST,
)


def _spec(
    case_id: str,
    family: str,
    passages: tuple[PassageMode, ...],
    slots: dict[str, str],
    tool_messages: tuple[str, ...],
    *,
    fallback: bool = False,
    tool_policy: ToolPolicy = ToolPolicy.EXACT,
    argument_check: tuple[int, int] | None = None,
    tool_list: str | None = None,
    description: str = "",
) -> CaseSpec:
    """Build a spec, wiring the optional ``tool_list`` / ``general_tool`` slots."""
    slots = dict(slots)
    if tool_list:
        slots["tool_list"] = tool_list
    if fallback:
        slots["general_tool"] = "general_tool"
    return CaseSpec(
        case_id=case_id,
        family=family,
        turns=len(passages),
        passages=passages,
        prompt_slots=slots,
        tool_messages=tool_messages,
        use_fallback_tools=fallback,
        tool_policy=tool_policy,
        argument_check_range=argument_check,
        description=description,
    )


# --------------------------------------------------------------------------- #
# Group A — single turn, one tool call per attempt
# --------------------------------------------------------------------------- #
_A = [
    _spec("case_A1", "A", (G,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1"},
          ("gold@1",),
          description="Single call, succeeds immediately."),
    _spec("case_A2", "A", (GB,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1", "error_content_1": "bad@1"},
          ("bad@1", "gold@1"),
          argument_check=(0, 1),
          description="Call returns nothing useful; retry with adjusted arguments."),
    _spec("case_A3", "A", (GB,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1", "error_content_1": "bad@1"},
          ("bad@1", "gold@1"),
          tool_list="distractors", tool_policy=ToolPolicy.ALLOW_EXTRA,
          description="Wrong tool picked first, then the right one."),
    _spec("case_A4", "A", (S3,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1",
           "error_content_1": "bad1@1", "error_content_2": "bad2@1", "error_content_3": "bad3@1"},
          ("bad1@1", "bad2@1", "bad3@1", "gold@1"),
          fallback=True, argument_check=(1, 2), tool_list="distractors_fallback",
          tool_policy=ToolPolicy.ALLOW_EXTRA,
          description="Three failed attempts, then fall back to general search."),
]

# --------------------------------------------------------------------------- #
# Group B — single turn, several tool calls per attempt
# --------------------------------------------------------------------------- #
_B = [
    _spec("case_B1", "B", (G,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1"},
          ("gold@1",),
          description="Parallel calls, all succeed."),
    _spec("case_B2", "B", (GB,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1", "error_content_1": "bad@1"},
          ("bad@1", "gold@1"),
          argument_check=(0, 1),
          description="Parallel calls miss; retry with adjusted arguments."),
    _spec("case_B3", "B", (GB,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1", "error_content_1": "bad@1"},
          ("bad@1", "gold@1"),
          argument_check=(0, 1),
          description="Parallel calls miss; retry with a different decomposition."),
    _spec("case_B4", "B", (GB,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1", "error_content_1": "bad@1"},
          ("bad@1", "gold@1"),
          tool_list="distractors", tool_policy=ToolPolicy.ALLOW_EXTRA,
          description="Wrong tool set picked first, then the right one."),
    _spec("case_B5", "B", (GB,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1", "error_content_1": "bad@1"},
          ("bad@1", "gold@1"),
          tool_list="distractors", tool_policy=ToolPolicy.ALLOW_EXTRA,
          description="Partially wrong tool set, corrected on retry."),
    _spec("case_B6", "B", (S3,),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1",
           "error_content_1": "bad1@1", "error_content_2": "bad2@1", "error_content_3": "bad3@1"},
          ("bad1@1", "bad2@1", "bad3@1", "gold@1"),
          fallback=True, argument_check=(1, 2), tool_list="distractors_fallback",
          tool_policy=ToolPolicy.ALLOW_EXTRA,
          description="Three failed parallel attempts, then general search."),
]

# --------------------------------------------------------------------------- #
# Groups C and D — two turns.  C calls one tool per turn, D calls several.
# The two groups are structurally identical, so they are generated from one table.
# --------------------------------------------------------------------------- #
_TWO_TURN_TABLE: list[tuple[str, tuple[PassageMode, ...], dict[str, str], tuple[str, ...], dict]] = [
    # id  passages        prompt slots (beyond the shared ones)      tool messages        extras
    ("1", (G, G),
     {"gold_content_1": "gold@1", "gold_content_2": "gold@2"},
     ("gold@1", "gold@2"),
     {"description": "Two sequential hops, both succeed."}),

    ("3", (G, GB),
     {"gold_content_1": "gold@1", "gold_content_2": "gold@2", "error_content_2": "bad@2"},
     ("gold@1", "bad@2", "gold@2"),
     {"tool_list": "distractors", "tool_policy": ToolPolicy.ALLOW_EXTRA,
      "description": "Second hop picks the wrong tool, then corrects."}),

    ("4", (G, GB),
     {"gold_content_1": "gold@1", "gold_content_2": "gold@2", "error_content_2": "bad@2"},
     ("gold@1", "bad@2", "gold@2"),
     {"argument_check": (1, 2), "description": "Second hop misses, retried with adjusted arguments."}),

    ("5", (GB, GB),
     {"gold_content_1": "gold@1", "gold_content_2": "gold@2",
      "error_content_1": "bad@1", "error_content_2": "bad@2"},
     ("bad@1", "gold@1", "bad@2", "gold@2"),
     {"argument_check": (0, 3), "tool_list": "distractors",
      "description": "Both hops miss once and are retried."}),

    ("6", (GB, GB),
     {"gold_content_1": "gold@1", "gold_content_2": "gold@2",
      "error_content_1": "bad@1", "error_content_2": "bad@2"},
     ("bad@1", "gold@1", "bad@2", "gold@2"),
     {"tool_list": "distractors", "tool_policy": ToolPolicy.ALLOW_EXTRA,
      "description": "Both hops pick the wrong tool once, then correct."}),

    ("7", (GB, G),
     {"gold_content_1": "gold@1", "error_content_1": "bad@1", "gold_content_2": "gold@2"},
     ("bad@1", "gold@1", "gold@2"),
     {"tool_list": "distractors", "tool_policy": ToolPolicy.ALLOW_EXTRA,
      "description": "First hop picks the wrong tool, then corrects."}),

    ("8", (GB, G),
     {"gold_content_1": "gold@1", "error_content_1": "bad@1", "gold_content_2": "gold@2"},
     ("bad@1", "gold@1", "gold@2"),
     # NOTE: cases C8/D8 appear in the original stage-4 argument-check config, but the
     # original generator never emitted the paired tool-call data the check needs, so it
     # always short-circuited to "pass". Left disabled to preserve published behaviour.
     {"description": "First hop misses, retried with adjusted arguments."}),

    ("9", (G, S3),
     {"gold_content_1": "gold@1", "gold_content_2": "gold@2",
      "error_content_1": "bad1@2", "error_content_2": "bad2@2", "error_content_3": "bad3@2"},
     ("gold@1", "bad1@2", "bad2@2", "bad3@2", "gold@2"),
     {"fallback": True, "argument_check": (2, 3), "tool_list": "distractors_fallback",
      "tool_policy": ToolPolicy.ALLOW_EXTRA,
      "description": "Second hop fails three times, then falls back to general search."}),

    ("10", (S3, G),
     {"gold_content_1": "gold@1", "gold_content_2": "gold@2",
      "error_content_1": "bad1@1", "error_content_2": "bad2@1", "error_content_3": "bad3@1"},
     ("bad1@1", "bad2@1", "bad3@1", "gold@1", "gold@2"),
     {"fallback": True, "argument_check": (1, 2), "tool_list": "distractors_fallback",
      "tool_policy": ToolPolicy.ALLOW_EXTRA,
      "description": "First hop fails three times, falls back, then the second hop succeeds."}),
]

_TWO_TURN: list[CaseSpec] = []
for _family in ("C", "D"):
    for _n, _modes, _slots, _tools, _extras in _TWO_TURN_TABLE:
        _slots = dict(_slots)
        _slots["right_tool_1"] = "plan@1"
        _slots["right_tool_2"] = "plan@2"
        _TWO_TURN.append(_spec(f"case_{_family}{_n}", _family, _modes, _slots, _tools, **_extras))

# ``case_C2`` was dropped from the taxonomy; ``case_D2`` is a one-off shape where
# the model answers a two-hop question from a single tool call, so it does not
# fit the table above.
_TWO_TURN = [spec for spec in _TWO_TURN if spec.case_id != "case_C2"]
_TWO_TURN.append(
    _spec("case_D2", "D", (G, MERGE),
          {"right_tool_1": "plan@1", "gold_content_1": "gold@1"},
          ("gold@1",),
          tool_policy=ToolPolicy.ALLOW_FEWER,
          description="Both hops answered from one tool call — fewer calls than labelled."),
)


#: Every supported case, keyed by id.
CASE_SPECS: dict[str, CaseSpec] = {
    spec.case_id: spec for spec in (*_A, *_B, *_TWO_TURN)
}

#: Sorted case ids, grouped A → B → C → D.
CASE_IDS: list[str] = sorted(
    CASE_SPECS, key=lambda cid: (cid[5], int(cid[6:]))
)

#: Cases that need the ``general_information_search`` fallback tool.  If the gold
#: tool for a record *is* general search, these cases are skipped for that record.
FALLBACK_CASES: frozenset[str] = frozenset(
    cid for cid, spec in CASE_SPECS.items() if spec.use_fallback_tools
)


def get(case_id: str) -> CaseSpec:
    """Look up a case spec, with a helpful error for typos."""
    try:
        return CASE_SPECS[case_id]
    except KeyError:
        raise KeyError(
            f"Unknown case '{case_id}'. Supported cases: {', '.join(CASE_IDS)}"
        ) from None


def summary_table() -> str:
    """A markdown table of all cases — rendered in the Web UI and the docs."""
    rows = [
        "| case | turns | tool messages | fallback | arg check | tool policy | description |",
        "|------|-------|---------------|----------|-----------|-------------|-------------|",
    ]
    for cid in CASE_IDS:
        spec = CASE_SPECS[cid]
        rows.append(
            f"| `{cid}` | {spec.turns} | {len(spec.tool_messages)} | "
            f"{'yes' if spec.use_fallback_tools else '—'} | "
            f"{'yes' if spec.check_arguments else '—'} | {spec.tool_policy.value} | {spec.description} |"
        )
    return "\n".join(rows)
