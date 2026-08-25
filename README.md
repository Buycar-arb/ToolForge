<div align="center">

<h1>
  <img src="assets/meituan.png" width="34" height="34" align="absmiddle" alt="Meituan">
  &nbsp;ToolForge
</h1>

<p><b><a href="https://arxiv.org/abs/2512.16149">A Data Synthesis Pipeline for Multi-Hop Search without Real-World APIs</a></b></p>

<p>
  <a href="https://arxiv.org/abs/2512.16149">
    <img src="https://img.shields.io/badge/arXiv-2512.16149-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/buycar/ToolForge-data">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Source%20QA-FFD21E?style=for-the-badge&labelColor=1a2230" alt="Hugging Face"></a>
  <a href="README_zh.md">
    <img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-%E6%96%87%E6%A1%A3-EC6708?style=for-the-badge&labelColor=1a2230" alt="中文文档"></a>
</p>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/models-GPT--5.1%20%7C%20Claude%20Opus%205-ec6708" alt="Models">
  <img src="https://img.shields.io/badge/cases-29-6d5bd0" alt="29 cases">
  <img src="https://img.shields.io/badge/checks-9%20rules%20%2B%20LLM%20judge-0e7490" alt="Validation">
  <img src="https://img.shields.io/badge/license-MIT-15803d" alt="MIT">
</p>

<p>
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#the-29-dialogue-cases">Cases</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

---

Teaching a model to *call tools* is easy. Teaching it to **recover** — to notice
a tool came back empty, retry with better arguments, abandon a tool that was
never the right one, and fall back to general search when nothing specialised
works — needs training data that almost never occurs in the wild.

ToolForge manufactures it. Feed it multi-hop QA; get back multi-turn
conversations in `<think>` / `<tool_call>` / `<answer>` form, in **29 distinct
failure-and-recovery shapes**, each one having survived nine structural checks
and an LLM judge before it is written to disk.

```bash
pip install -e ".[all]"
cp .env.example .env          # add one API key
toolforge doctor              # confirm it's wired up
toolforge webui               # or drive it from the command line
```

---

## What comes out

One record per generated sample. This is `case_C4` — two hops, where the second
tool call comes back useless and the model retries it with corrected arguments:

```jsonc
{"messages": [
  {"role": "system",    "content": "…agent instructions + 9 tool schemas…"},
  {"role": "user",      "content": "Were Scott Derrickson and Ed Wood of the same nationality?"},

  {"role": "assistant", "content": "<think>\nA comparison question — I need each person's
                                    nationality separately. Start with Derrickson.\n</think>\n
                                    <tool_call>{\"name\": \"people_data_investigator\",
                                    \"arguments\": {\"query\": \"Scott Derrickson nationality\"}}</tool_call>"},
  {"role": "tool",      "content": "**1**\ntitle: Scott Derrickson\ncontent: … is an American director. …"},

  {"role": "assistant", "content": "<think>\nDerrickson is American. Now Ed Wood.\n</think>\n
                                    <tool_call>{\"name\": \"people_data_investigator\",
                                    \"arguments\": {\"query\": \"Wood\"}}</tool_call>"},
  {"role": "tool",      "content": "**1**\ntitle: Wood carving\ncontent: … unrelated …"},

  {"role": "assistant", "content": "<think>\nThat query was too vague and returned nothing about
                                    the director. Retry with his full name.\n</think>\n
                                    <tool_call>{\"name\": \"people_data_investigator\",
                                    \"arguments\": {\"query\": \"Ed Wood film director nationality\"}}</tool_call>"},
  {"role": "tool",      "content": "**1**\ntitle: Ed Wood\ncontent: … was an American director …"},

  {"role": "assistant", "content": "<think>\nBoth American.\n</think>\n<answer>\nyes\n</answer>"}
]}
```

Three details that make this data hard to fake by hand, and that ToolForge gets
right for every sample:

- **The failed call really fails.** Its `tool` message is genuine BM25 output
  over the record's *non*-supporting passages. The distractors are real
  distractors, not `"error"`.
- **The tool is never the same tool twice.** Each of the 22 domain libraries
  holds ~20 paraphrases of one capability; a different one is drawn per sample.
  A model trained on this cannot memorise tool names — it has to read schemas.
- **The reflection is earned.** The judge reads every `<think>` block and asks
  whether it matches the action that follows, and whether a mistaken turn is
  actually reflected on by the next one. Nothing reaches the training set on
  structure alone.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/Buycar-arb/ToolForge.git
cd ToolForge
pip install -e ".[all]"
```

Or take only what you need — `pip install -e .` gives the core pipeline, and the
extras are `webui`, `anthropic`, `data`, `embeddings`, `eval`, `dev`.

### 2. Configure

```bash
cp .env.example .env
```

One key is enough:

```bash
OPENAI_API_KEY=sk-…              # comma-separate several; they are rotated
GENERATION_MODEL=gpt-5.1         # writes the dialogues
JUDGE_MODEL=gpt-5.1              # scores them
```

Anthropic works natively too:

```bash
ANTHROPIC_API_KEY=sk-ant-…
JUDGE_MODEL=claude-opus-5        # the provider is inferred from the model id
```

The provider is chosen from the model name — `claude-*` goes to the Anthropic
Messages API, everything else to an OpenAI-compatible endpoint. Force it with a
prefix when you need to (`openai:claude-sonnet-5` routes Claude through a
gateway; `anthropic:claude-opus-5` forces the native API). Point
`OPENAI_BASE_URL` at Azure, vLLM, OpenRouter or your own gateway and any model
they serve works unchanged.

```bash
toolforge doctor
```

tells you exactly what is still missing.

> Current models are not interchangeable at the API level — GPT-5 rejects
> `max_tokens`, the Anthropic SDK dropped `temperature`, and models disagree
> about whether to fence their JSON. ToolForge negotiates all three for you;
> [`docs/models.md`](docs/models.md) explains what it is doing and why.

### 3. Get the data

```bash
python download_data.py
toolforge convert to-jsonl data/source_qa/HotpotQA
```

### 4. Run it

<table>
<tr><th align="left" width="50%">Visual</th><th align="left" width="50%">Command line</th></tr>
<tr valign="top"><td>

```bash
toolforge webui
```

Opens `http://localhost:7860`. Five tabs:

- **Overview** — a readiness checklist
- **Tool bank** — browse libraries, edit `TOOL_LIST`, run stage 1
- **Label** — stage 2, with live progress
- **Generate** — stages 3 + 4
- **Data** — browse any output, re-run the checks

Loopback access stays anonymous. Before using `--share` or a non-loopback
`--host`, set `TOOLFORGE_WEBUI_USERNAME` and `TOOLFORGE_WEBUI_PASSWORD`;
startup fails closed when either credential is missing.

</td><td>

```bash
# stage 2 — label questions
toolforge label \
  data/source_qa/HotpotQA/bridge_hp.jsonl \
  output/labelled/output.jsonl --limit 200

# stages 3 + 4 — generate and validate
toolforge generate output/labelled/output.jsonl \
  --case case_C1 --case case_D4 --target 100
```

</td></tr>
</table>

Start with `--limit 20 --target 3` and read what comes out before scaling up.

---

## How it works

```
  data/source_qa/            raw multi-hop QA (HotpotQA schema)
         │
         ▼
  ┌──────────────┐  Stage 2 · label
  │  labeling    │  → which tool answers this, and in how many turns?
  └──────┬───────┘     adds reasoning · tool_select · route_select
         │
         ▼
  ┌──────────────┐  Stage 3 · generate                      ┌───────────────┐
  │  dialogue    │  1. plan the tool-calling trajectory      │  tool_bank/   │
  │              │  2. retrieve real distractor passages ◀───│  22 libraries │
  │              │  3. author the conversation              │  ~20 variants │
  └──────┬───────┘  4. assemble the record                  └───────────────┘
         │
         ▼
  ┌──────────────┐  Stage 4 · validate
  │  validation  │  9 rule checks  →  rule_score 0 or 1
  │  judge       │  LLM scoring    →  gpt_score  0 or 1
  └──────┬───────┘  only 2/2 is kept
         │
         ├──▶  output/data/case_XX.jsonl     the training set
         └──▶  output/scores/case_XX.jsonl   every attempt, with its verdict
```

**Stage 1** is optional and sits outside this loop: it grows the tool bank by
paraphrasing a tool definition, keeping a variant only when it is *close in
meaning* (cosine above threshold) and *far in wording* (BM25 below threshold).

### The nine checks

| # | what it verifies |
|:-:|------------------|
| 1 | the role sequence matches the case's expected pattern |
| 2 | assistant turns are `<think>` + `<tool_call>`, ending with `<answer>` |
| 3 | no `system` / `user` / `tool` message is empty |
| 4 | the final `<answer>` equals the gold answer |
| 5 | each `tool` message renders **exactly** its passage bundle — nothing invented, nothing dropped |
| 6 | a retried tool call only changes parameters the schema marks `required` |
| 7 | the passages used equal the record's `supporting_facts` |
| 8 | the tools actually called match what stage 2 labelled |
| 9 | every tool call names a tool that was offered, with a valid argument set |

Every attempt — kept or rejected — is written to the score file with its reason,
so the yield is auditable rather than asserted:

```
### Run complete
**312** samples kept from **604** attempts — overall yield **51.7%**

| case      | kept | target | attempts | yield |
|-----------|------|--------|----------|-------|
| ✅ case_C1 | 100  | 100    | 173      | 57.8% |
| ✅ case_D4 | 100  | 100    | 210      | 47.6% |

**Most common rejection reasons**
- `142×` 5. Tool-RAG consistency check failed
- ` 88×` 2. Assistant content format validation failed
```

> Two checks in the original release were inert, and are preserved that way by
> default so published numbers reproduce exactly. Both now have an opt-in flag —
> see [`docs/behaviour-notes.md`](docs/behaviour-notes.md).

---

## The 29 dialogue cases

Four families, differing in how many turns the question needs and how many calls
each turn makes:

| family | shape | cases |
|:------:|-------|-------|
| **A** | one turn, one call per attempt | `A1`–`A4` |
| **B** | one turn, several calls per attempt | `B1`–`B6` |
| **C** | two turns, one call each | `C1`, `C3`–`C10` |
| **D** | two turns, several calls per turn | `D1`–`D10` |

Within a family the cases vary the *failure*: a call that comes back empty and
is retried with corrected arguments, a wrong tool chosen first, three failed
attempts followed by a fallback to general search, or a two-hop question
answered from a single call.

```bash
toolforge cases                     # the full table
toolforge cases --case case_C9      # one case, including its reasoning flow
```

Each case is a data declaration, not a class — the whole taxonomy is one
readable file, [`toolforge/stages/cases.py`](toolforge/stages/cases.py):

```python
_spec("case_C9", "C", (GOLD_ONLY, THREE_STRIKES),
      {"gold_content_1": "gold@1", "gold_content_2": "gold@2",
       "error_content_1": "bad1@2", "error_content_2": "bad2@2", "error_content_3": "bad3@2"},
      ("gold@1", "bad1@2", "bad2@2", "bad3@2", "gold@2"),
      fallback=True, argument_check=(2, 3),
      description="Second hop fails three times, then falls back to general search.")
```

Adding a 30th case means adding one entry here plus its prompt and flow — the
test suite picks it up automatically.

---

## Repository layout

```
toolforge/               the package — everything importable
├─ config.py             every tunable, resolved from .env
├─ llm.py                one async client: provider routing, key rotation, backoff
├─ toolbank.py           reading the bank; sampling tool sets per record
├─ bm25.py               retrieval that supplies realistic distractors
├─ convert.py            Parquet ↔ JSONL
├─ cli.py                the `toolforge` command
├─ stages/
│  ├─ cases.py           ★ the 29 cases, declared as data
│  ├─ dialogue.py        ★ the generation engine
│  ├─ validation.py      the nine rule checks
│  ├─ judge.py           LLM scoring
│  ├─ pipeline.py        the generate → validate → score loop
│  ├─ labeling.py        stage 2
│  └─ variants.py        stage 1
├─ prompts/              every prompt, grouped by the job it does
└─ webui/                the Gradio front end

tool_bank/               22 domain tool libraries (JSONL, ~20 variants each)
data/                    datasets (downloaded)
tests/                   offline suite — no API key, no network
viewer/compare.html      side-by-side trajectory viewer, self-contained
Evaluation_Framework/    EM / F1 on NQ, PopQA, Musique, Bamboogle
train/                   Swift SFT scripts
docs/                    architecture · models · behaviour notes · migration
```

---

## Using it as a library

```python
import asyncio
from toolforge import CaseJob, Pipeline, load_records

records = load_records("output/labelled/output.jsonl")
jobs = [CaseJob("case_C1", target=100,
                data_output="output/data/case_C1.jsonl",
                score_output="output/scores/case_C1.jsonl")]

results = asyncio.run(Pipeline().run(records, jobs, on_event=print))
print(results["case_C1"].summary())
```

Or drive one sample at a time:

```python
from toolforge import DialogueGenerator, SourceRecord, validate

record = SourceRecord.parse(raw_stage2_row)
sample = await DialogueGenerator().generate(record, "case_C1")
print(validate(sample.to_record(), "case_C1").passed)
```

---

## Tests

The suite runs the real engine and the real checks against a scripted model, so
it needs no API key and no network:

```bash
pip install -e ".[dev]"
pytest
```

Every case is generated and validated on every run — a regression in bundling,
ordering, rendering or validation fails immediately.

---

## Training and evaluation

Both are GPU-specific and documented separately in
[`docs/training.md`](docs/training.md): Swift SFT for Qwen3, the EM / F1
harness for NQ / PopQA / Musique / Bamboogle, and the custom side-by-side
benchmark viewer.

---

## Source data

🤗 **[buycar/ToolForge-data](https://huggingface.co/datasets/buycar/ToolForge-data)**

**Raw multi-hop question answering — the input ToolForge runs on.** 257,901
questions across six corpora from HotpotQA and 2WikiMultihopQA, in the exact
slices used in the paper. This is *source* data, not generated data: it is what
you feed to the pipeline, and stage 2 is the first thing that touches it.

```bash
python download_data.py              # -> data/source_qa/
python download_data.py --with-model # + bge-m3, for stage 1's similarity gate
```

| corpus | questions | question shape |
|---|---:|---|
| `HotpotQA/bridge_hp` | 72,991 | bridge: hop 1's answer identifies hop 2's subject |
| `HotpotQA/comparison_hp` | 17,456 | comparison: "which of X and Y…" |
| `2WikiMultihopQA/compositional_wiki` | 76,481 | compositional: "the Z of the Y of X" |
| `2WikiMultihopQA/comparison_wiki` | 51,963 | comparison |
| `2WikiMultihopQA/bridge_comparison_wiki` | 34,631 | bridge *and* comparison |
| `2WikiMultihopQA/inference_wiki` | 4,379 | inference over family relations |

From there, three commands produce training data:

```bash
toolforge convert  to-jsonl data/source_qa/HotpotQA
toolforge label    data/source_qa/HotpotQA/bridge_hp.jsonl data/labelled/output.jsonl
toolforge generate data/labelled/output.jsonl --case case_C1 --target 100
```

Because you run the factory rather than download its output, you can build the
same data over *your* corpus, in *your* domain, with *your* tools — any corpus
matching the schema in the [dataset card](https://huggingface.co/datasets/buycar/ToolForge-data)
works.

The four categories the pipeline produces:

| category | meaning |
|----------|---------|
| **SRST** | single round, single tool |
| **SRMT** | single round, multi tool |
| **MRST** | multi round, single tool |
| **MRMT** | multi round, multi tool |

Which corpus yields which case family is not obvious and matters a great deal —
[`docs/choosing-source-data.md`](docs/choosing-source-data.md) has the measured
breakdown.

---

## Citation

If ToolForge is useful in your research, please cite:

```bibtex
@article{chen2025toolforge,
  title={ToolForge: A Data Synthesis Pipeline for Multi-Hop Search without Real-World APIs},
  author={Chen, Hao and Hu, Zhexin and Chai, Jiajun and Yang, Haocheng and He, Hang and Wang, Xiaohan and Lin, Wei and Wang, Luhang and Yin, Guojun and others},
  journal={arXiv preprint arXiv:2512.16149},
  year={2025}
}
```

---

<div align="center">
<sub><a href="https://arxiv.org/abs/2512.16149">Paper</a> · <a href="docs/architecture.md">Architecture</a> · <a href="docs/models.md">Models</a> · <a href="docs/behaviour-notes.md">Behaviour notes</a> · <a href="README_zh.md">中文文档</a><br>MIT licensed</sub>
</div>
