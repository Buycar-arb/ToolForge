# Evaluation Framework

EM / F1 on NQ, PopQA, Musique and Bamboogle, in either of the two inference
styles a tool-calling model can be asked to use.

| style | how the model searches | for |
|-------|------------------------|-----|
| **tag** | emits `<search>query</search>` in free text | Search-R1 style models |
| **function** | emits a standard tool call | anything with function calling |

## Setup

```bash
pip install -e ".[eval]"        # from the repository root
```

### 1. Start the retrieval server

Every run needs a retrieval backend. `rag_server/` is a FastAPI + FAISS server
over the Wikipedia dump:

```bash
python rag_server/download.py          # corpus + index (large)
bash rag_server/launch.sh              # serves http://localhost:5003/retrieve
bash rag_server/quick_test.sh          # confirm it answers
```

Point `search_engine.url` in `evaluations/config/search_engines.yaml` at it if
you serve it elsewhere.

### 2. Serve a local model (only for `type: open_source`)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3-8B \
    --port 8001 \
    --served-model-name qwen3-8b     # must match `model_path` in models.yaml
```

### 3. Credentials

API models read their keys from the environment — `models.yaml` holds
`${VAR}` references, never literals:

```bash
export OPENAI_API_KEY=sk-…
export OPENAI_ENDPOINT=https://api.openai.com/v1/chat/completions
export ANTHROPIC_API_KEY=sk-ant-…

# Function-calling mode additionally uses the rotating-key client:
export API_KEYS=sk-key-1,sk-key-2
export API_BASE_URL=https://api.openai.com/v1
```

## Run

```bash
cd evaluations
python run_evaluation.py                                    # uses the config defaults
python run_evaluation.py --model gpt-5.1 --method function --datasets bamboogle nq
python run_evaluation.py --use_multithreading --max_workers 8
```

Results land in `evaluations/results/<model>_<method>_<timestamp>/`:

```
config.json              what was run
<dataset>_checkpoint.jsonl   resumable progress — rerun to continue
<dataset>_results.json   per-example predictions plus the metrics
summary.json             metrics across all datasets
```

Recompute metrics without re-running inference:

```bash
python evaluations/src/metrics/metrics.py results/<run>/bamboogle_results.json
python evaluations/src/metrics/metrics.py results/<run>/bamboogle_results.json --print
```

## Configuration

| file | what it holds |
|------|---------------|
| `config/models.yaml` | model definitions and `active_model` |
| `config/datasets.yaml` | `active_datasets`, sizes, metrics, threading |
| `config/search_engines.yaml` | retrieval endpoint, `search_method`, function schemas |
| `config/prompts.yaml` | prompt variants per inference style |

### Adding a model

Any model reachable over an OpenAI-compatible API is one YAML block — there is a
single `APIModel` class behind all of them:

```yaml
  my-model:
    type: closed_source
    model_name: my-model
    api_key: ${MY_API_KEY}
    endpoint: https://my-gateway/v1/chat/completions
    max_tokens: 4096
    temperature: 0
    timeout: 60
```

## Metrics

| metric | definition |
|--------|------------|
| **EM** | exact match after normalisation (lowercase, strip articles and punctuation) |
| **F1** | token-level overlap between the predicted and gold answer |

## Layout

```
evaluations/
├─ run_evaluation.py     entry point
├─ config/               the four YAML files above
└─ src/
   ├─ models/            closed_source.py (APIModel) · open_source.py (vLLM)
   ├─ datasets/          HF hub and local JSONL loaders
   ├─ search/            tag-based and function-based search handlers
   ├─ inference/         the per-question inference loops
   ├─ metrics/           EM / F1
   └─ utils/             threading, prompts, logging
rag_server/              FastAPI + FAISS retrieval server
```
