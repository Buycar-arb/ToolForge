# Architecture

How the pieces fit together, and where to look when you want to change something.

```
                      ┌─────────────────────────────────────────┐
   raw multi-hop QA   │  data/source_qa/{HotpotQA,2Wiki…}       │
   (HotpotQA schema)  └──────────────────┬──────────────────────┘
                                         │
                     ┌───────────────────▼───────────────────┐
   stage 2           │  toolforge/stages/labeling.py         │
   label             │  + reasoning, tool_select, route_select│
                     └───────────────────┬───────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
┌───────▼────────┐          ┌────────────▼────────────┐        ┌──────────▼─────────┐
│ tool_bank/     │          │ stages/cases.py         │        │ prompts/           │
│ 22 libraries   │──────────▶ 29 CaseSpec records     │◀───────│ planning, cases,   │
│ ~20 variants   │  sampled  │ (data, not code)       │ chosen │ flows, dialogue    │
└────────────────┘  by       └────────────┬────────────┘  by   └────────────────────┘
                     toolbank.py          │
                                          │
                     ┌────────────────────▼────────────────────┐
   stage 3           │  toolforge/stages/dialogue.py           │
   generate          │  plan → retrieve → author → assemble    │
                     └────────────────────┬────────────────────┘
                                          │
                     ┌────────────────────▼────────────────────┐
   stage 4           │  validation.py  (9 rule checks)         │
   validate          │  judge.py       (LLM scoring)           │
                     └────────────────────┬────────────────────┘
                                          │ 2/2 only
                     ┌────────────────────▼────────────────────┐
                     │  output/data/case_XX.jsonl   ← training │
                     │  output/scores/case_XX.jsonl ← audit    │
                     └─────────────────────────────────────────┘
```

## The central idea: cases as data

The original release had 29 processor classes averaging 110 lines each, all
copies of the same three-step recipe. They now live as 29 `CaseSpec` records in
`toolforge/stages/cases.py`, and one engine in `toolforge/stages/dialogue.py`
executes all of them.

A case declares only what actually differs:

```python
_spec("case_C9", "C", (GOLD_ONLY, THREE_STRIKES),
      {"right_tool_1": "plan@1", "right_tool_2": "plan@2",
       "gold_content_1": "gold@1", "gold_content_2": "gold@2",
       "error_content_1": "bad1@2", "error_content_2": "bad2@2", "error_content_3": "bad3@2"},
      ("gold@1", "bad1@2", "bad2@2", "bad3@2", "gold@2"),
      fallback=True, argument_check=(2, 3), tool_list="distractors_fallback",
      tool_policy=ToolPolicy.ALLOW_EXTRA,
      description="Second hop fails three times, then falls back to general search.")
```

`3160` lines → `~330` lines of declarations plus `~430` lines of engine, with
the per-case behaviour visible in one screen instead of scattered across 29
copy-pasted blocks.

### Bundle references

A slot value like `"bad1@2"` names a passage bundle: *kind* `@` *turn*.

| kind | contents |
|------|----------|
| `gold@N` | turn *N*'s BM25 hits **plus** the supporting passages — the call that works |
| `bad@N` | turn *N*'s BM25 hits **without** them — the call that fails and gets retried |
| `bad1@N`, `bad2@N`, `bad3@N` | three disjoint slices of a 3× wider retrieval, for the fallback cases |
| `plan@N` | turn *N*'s planned tool calls (not passages) |

`tool_messages` lists the bundles served to the model as `tool` messages, **in
order**. That order is part of the on-disk format: stage 4's check 5 compares
tool message *i* against `rags[i]`.

## Module map

| module | responsibility |
|--------|----------------|
| `toolforge/config.py` | every tunable, resolved from `.env` + environment |
| `toolforge/llm.py` | one async client — provider routing, key rotation, backoff |
| `toolforge/toolbank.py` | reading the bank; sampling the tool sets for one record |
| `toolforge/bm25.py` | retrieval that supplies realistic distractor passages |
| `toolforge/jsonl.py` | the file format every stage speaks |
| `toolforge/convert.py` | Parquet ↔ JSONL |
| `toolforge/stages/cases.py` | the 29 case specifications |
| `toolforge/stages/dialogue.py` | the generation engine |
| `toolforge/stages/validation.py` | the nine rule checks |
| `toolforge/stages/judge.py` | LLM scoring |
| `toolforge/stages/pipeline.py` | the generate → validate → score loop |
| `toolforge/stages/labeling.py` | stage 2 |
| `toolforge/stages/variants.py` | stage 1 |
| `toolforge/prompts/` | every prompt, grouped by the job it does |
| `toolforge/webui/` | the Gradio front end |
| `toolforge/cli.py` | the `toolforge` command |

## The output record

Stage 3 writes a seven-element list per sample. Stage 4 addresses it
positionally, so the order is the format:

```
[0] {"case": "case_C1", "uuid": "..."}
[1] {"messages": [...]}                    ← the training dialogue
[2] {"rags": [...], "answer", "reasoning", "good_tool_mapping"}
[3] {"argument_check": [...] | "Don't need to check"}
[4] {"argument_all_reference": [{"turn": 1, "data": [...]}, ...]}
[5] {"argument_tool_bank": [...]}          ← every tool schema offered
[6] {...}                                  ← the original stage 2 record
```

For training you want `[1]["messages"]`.

## Adding a case

1. Add a `_spec(...)` entry to `toolforge/stages/cases.py`.
2. Add its user prompt to `toolforge/prompts/cases.py` and register it in
   `CASE_USER_PROMPTS`.
3. Add its reasoning flow to `toolforge/prompts/flows.py` and register it in
   `CASE_FLOWS`.
4. Run `pytest` — the suite parametrises over every case, so a new one is
   generated and validated automatically.

No engine changes are needed unless the case needs a genuinely new passage mode.

## Adding a provider

`toolforge/llm.py` handles two: any OpenAI-compatible endpoint, and the native
Anthropic Messages API. To add a third, extend `LLMClient._build_client` and
`_dispatch`, then add an entry to `MODEL_REGISTRY` so it shows up in the UI.

Most "new providers" need nothing at all — if they speak the OpenAI API, point
`OPENAI_BASE_URL` at them.
