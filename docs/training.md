# Training and evaluation

The data factory has no GPU dependencies. Training and inference do, and their
requirements are CUDA-version specific, so they are deliberately kept out of
`requirements.txt`.

## Fine-tuning

```bash
pip install ms-swift
```

Edit `train/train.sh` to point at your base model and the JSONL produced by
stage 3, then:

```bash
cd train && bash train.sh
```

The published configuration:

| | |
|---|---|
| base model | Qwen3, full-parameter |
| sequence length | 12,000 tokens |
| precision | BF16 + FlashAttention 2 |
| sharding | DeepSpeed ZeRO-2 |
| hardware | 4 GPUs |

`train/qwen3_mix/qwen3_think.py` supplies the chat template that keeps
`<think>` blocks intact through tokenisation — without it the reasoning traces
are stripped and the model learns to call tools without thinking first.

Training consumes `[1]["messages"]` from each generated record. Extract it with:

```bash
python -c "
import json, sys
for line in open(sys.argv[1]):
    print(json.dumps({'messages': json.loads(line)[1]['messages']}, ensure_ascii=False))
" output/data/case_C1.jsonl > train.jsonl
```

## Standard benchmarks

`Evaluation_Framework/` measures EM / F1 on NQ, PopQA, Musique and Bamboogle,
in either inference style:

- **tag-based** — the model emits `<search>query</search>` (Search-R1 style)
- **function calling** — standard tool calls

```bash
pip install -e ".[eval]"

# 1. start the retrieval server
bash Evaluation_Framework/rag_server/launch.sh

# 2. (for local models) serve the checkpoint
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/checkpoint --port 8001 --served-model-name qwen3-8b-sft

# 3. run
cd Evaluation_Framework/evaluations
python run_evaluation.py --model qwen3-8b-sft --method function --datasets bamboogle
```

Configuration lives in `Evaluation_Framework/evaluations/config/`:

| file | what it holds |
|------|---------------|
| `models.yaml` | model definitions; API keys are `${ENV_VAR}` references, never literals |
| `datasets.yaml` | which datasets to run, and their metrics |
| `search_engines.yaml` | retrieval endpoint, `search_method`, and the function schemas |
| `prompts.yaml` | prompt variants per inference style |

Recompute metrics on a finished run without re-running inference:

```bash
python evaluations/src/metrics/metrics.py evaluations/results/<run>/bamboogle_results.json
```

## The custom benchmark

Compares a fine-tuned checkpoint against a baseline on your own generated data.

```bash
cd ourbenchmark_inference_output

bash model_deploy.sh                 # serve the checkpoint with Swift + vLLM
python our_model_eval.py             # run the fine-tuned model
python open_source_model_eval.py     # run a baseline for comparison
```

Then open `viewer/compare.html` in a browser and drop the two result files into
it. The viewer is a single self-contained page — no server, no CDN, nothing
uploaded. It shows both trajectories side by side with synced scrolling,
per-file exact-match rates, and filters for the cases where the two models
disagree.

Result files are JSONL with one object per question:

```json
{"sample_id": 0, "original_query": "...", "golden_answer": "...", "messages": [...]}
```
