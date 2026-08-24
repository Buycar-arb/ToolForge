# Behaviour notes

Things found while refactoring that a reader of the code — or of the paper —
should know about. **The default behaviour of every item below matches the
published release**, so numbers stay reproducible. Each has an opt-in switch.

---

## 1. Check 7 (reference consistency) never ran

`DataValidator.check_reference_consistency` read `data[2]["argument_all_reference"]`
and `data[3]["supporting_facts"]`. In the integrated pipeline those fields live
at index **4** and **6**:

```
[0] case header   [1] messages   [2] rags/answer/reasoning   [3] argument_check
[4] argument_all_reference        [5] argument_tool_bank      [6] source record
```

Both lookups therefore returned `[]`, `len([]) == len([])` passed, and
`set() == set()` returned 1. The check reported success on every record without
ever comparing anything.

**Now:** `toolforge/stages/validation.py` implements the check against the
documented layout, but it is gated behind `ValidationOptions.strict_reference_check`,
default `False`.

```bash
toolforge generate labelled.jsonl --case case_C1 --strict-references
toolforge validate output/data/case_C1.jsonl --strict-references
```

`tests/test_pipeline.py::test_strict_reference_check` shows all 29 cases pass
with it enabled on well-formed data, so turning it on is safe. It will reject
records where the model dropped or invented a supporting passage — which is the
point.

---

## 2. Check 2 did not enforce the final answer format

In `check_assistant_content_format`, the branch for the **last** assistant turn
printed `"error"` with the `return 0` commented out:

```python
if not re.match(pattern, content.strip(), re.DOTALL):
    print("error")
    # return 0        <- disabled
```

So a malformed closing turn was logged but accepted. (Check 4 still verified the
answer *text*, so this was never a correctness hole — only a formatting one.)

**Now:** the same behaviour by default; `ValidationOptions.strict_final_answer_format`
(`--strict-answer-format`) turns it into a rejection.

---

## 3. `case_C8` / `case_D8` were listed for check 6 but never supplied its data

`ARGUMENT_CHECK_CONFIG` mapped `case_C8` and `case_D8` to the window `(0, 1)`,
but their processors emitted `{"argument_check": "Don't need to check"}`, which
makes `_check_arguments` return 1 immediately. The configuration entry was dead.

This matters because C8/D8 are exactly the "retry with corrected arguments"
shape that check 6 exists to police — the intent was clearly to check them.

**Now:** stated explicitly in `toolforge/stages/cases.py`, where each case
carries its own `argument_check_range` (`None` for C8/D8) instead of a separate
lookup table that could drift. To enable it, give those two entries a range:

```python
# toolforge/stages/cases.py, in _TWO_TURN_TABLE, entry "8"
{"argument_check": (0, 1), "description": "First hop misses, retried with adjusted arguments."}
```

---

## 4. Dead code removed

None of these were reachable from any entry point:

The pre-refactor tree is at commit `86194c8` if you want to check any of these
against the original — see [`migration.md`](migration.md).

| removed | where it was |
|---------|--------------|
| `DataProcessor.replace_tool_names_in_reasoning` | `Stage_3/services/data_processor.py` |
| `extract_messages_brutal`, `find_closing_quote`, `merge_messages` | `Stage_3` eval `closed_source.py` |
| `extract_tool_calls_as_str_list`, `extract_reference1_simple`, `extract_reference2_simple` | `Stage_3/utils/text_utils.py` |
| `ToolManager.get_grouped_tool_calls_hybrid` | duplicate of the `BaseProcessor` method |
| `MCPCaller` | instantiated only when `mcp_api_url` was set, which no entry point did |

Unused per-case computations were also dropped — `case_C1`, for instance, built
`tool_response1_bad` and `tool_response2_bad` and then referenced neither.

---

## 5. Behaviour that is intentionally identical

- **Prompts** — all 29 case prompts, 29 flows, 4 planning prompts, the dialogue
  system prompt, the judge prompts, the agent system/user prompts and the stage 2
  tool-selection prompts were moved **byte for byte**. A test asserted this
  during the port.
- **Dialogue patterns** — the 29 expected role sequences are now *derived* from
  each case's `tool_messages` rather than hand-written. They come out identical
  to the original table, which independently confirms the case specifications.
- **Passage bundling** — retrieval width, the `//3` split for the fallback cases,
  deduplication-then-shuffle, and the order bundles are served in are unchanged.
  Bundle order is load-bearing: check 5 compares tool message *i* against `rags[i]`.
- **Tool sampling** — one variant per domain file, `virtual_tool_min..max`
  distractors, gold tools appended and shuffled; the fallback set is sampled
  separately from the standard set, exactly as before.
- **Scoring** — `rule_score + gpt_score`, only 2/2 is kept, every attempt is
  written to the score file.

---

## 6. Things that changed on purpose

| change | why |
|--------|-----|
| A malformed plan is rejected before the authoring call | The original indexed `reference1[i]` past the end and raised `IndexError`, which the caller caught as a generic failure. Same outcome, one wasted API call saved, and a message that says what was wrong. |
| One-turn cases now get the same plan sanity check as two-turn ones | Groups A and B skipped it; the only difference was a crash instead of a clean skip. |
| Concurrency inside a case | The original processed one record at a time. Set `--concurrency 1` for the old pacing. |
| Retries use exponential backoff | The original slept a flat 40 s between attempts *and* 40 s between keys, so a bad key cost minutes. Backoff starts at 5 s and caps at 60 s. |
| Empty API responses are retried | The original returned `None` and the caller treated it as a hard failure. |
