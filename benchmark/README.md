# Custom benchmark

Replays ToolForge-generated samples as a **live agent loop** against any model:
same system prompt, same question, and every `<tool_call>` answered by running
BM25 over that record's own passages. What the model does with those results is
entirely its own — so this measures tool-calling behaviour, not memorisation.

## Compare a fine-tuned checkpoint against a baseline

```bash
# 1. serve your checkpoint
bash deploy_model.sh /path/to/checkpoint toolforge-8b 0,1,2,3

# 2. run it
python run_benchmark.py ../output/data/case_C1.jsonl ours.jsonl \
    --model toolforge-8b --base-url http://0.0.0.0:8000/v1 --api-key EMPTY

# 3. run a baseline over an API (keys come from ../.env)
python run_benchmark.py ../output/data/case_C1.jsonl baseline.jsonl --model gpt-5.1

# 4. open ../viewer/compare.html and drop both files in
```

Each run prints a per-sample verdict and an exact-match rate at the end.

## The viewer

`viewer/compare.html` is a single self-contained page — no server, no CDN,
nothing uploaded. Drop the two result files onto it and you get:

- both trajectories side by side, with synced scrolling
- `<think>`, `<tool_call>` and `<answer>` blocks broken out and colour-coded
- retrieved passages rendered as a readable list rather than one wall of text
- exact-match rate per file
- filters for **Disagree**, **A only**, **B only**, **Both wrong** — which is
  where the interesting samples live
- keyboard navigation (`j` / `k` or the arrow keys), light/dark, EN/中文

## Options

| flag | meaning |
|------|---------|
| `--model` | model id, or the `--served-model-name` you gave vLLM |
| `--base-url` | endpoint override; omit to use `OPENAI_BASE_URL` from `.env` |
| `--api-key` | single key; omit to use the rotating keys from `.env` |
| `--limit N` | only evaluate the first N samples |
| `--top-k` | passages returned per tool call (default 10) |
| `--max-rounds` | tool rounds before giving up (default 5) |
| `-v` | print every model turn as it happens |

## Output format

One JSON object per line — the format `viewer/compare.html` reads:

```json
{"sample_id": "...", "original_query": "...", "golden_answer": "...", "messages": [...]}
```
