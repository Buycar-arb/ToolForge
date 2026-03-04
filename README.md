<div align="center">

<h1>🔧 ToolForge</h1>

<p>
  <b>English</b> | <a href="README_zh.md">中文</a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Gradio%205.x-orange?logo=gradio" alt="Gradio">
  <img src="https://img.shields.io/badge/LLM-Qwen3%20%7C%20GPT%20%7C%20Claude-green" alt="LLM">
  <img src="https://img.shields.io/badge/Inference-vLLM-purple" alt="vLLM">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<p><em>An automated SFT data synthesis pipeline for LLM tool-calling capabilities</em></p>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Data Format](#data-format)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)

---

## Overview

**ToolForge** is an end-to-end automated pipeline for synthesizing high-quality Supervised Fine-Tuning (SFT) data targeting **LLM tool-calling capabilities**. It covers the complete workflow from raw multi-hop QA data to training-ready data, model fine-tuning, and evaluation.

The pipeline trains LLMs (primarily **Qwen3**) to accurately select and invoke tools in complex multi-hop question-answering scenarios. It supports **29 distinct dialogue case types** across single/multi-turn and single/multi-tool combinations.

> **Datasets on Hugging Face** — All datasets (source data + final SFT training data) are hosted at 🤗 [buycar/ToolForge](https://huggingface.co/datasets/buycar/ToolForge). Download instructions in [Quick Start](#quick-start).

### Key Features

- 🏭 **Full pipeline automation** — from raw QA data to validated SFT training data
- 📦 **Built-in sample datasets** — HotpotQA & 2WikiMultihopQA ready to use
- 🔀 **29 dialogue case types** — A1-A4 (single-turn), B1-B6 (reflection), C1,C3-C10 (dual-tool), D1-D10 (complex)
- 🌐 **Bilingual support** — Chinese & English training datasets
- 🛠️ **22 domain tool libraries** — academic, medical, geographic, economic, and more
- ✅ **Dual-stage validation** — rule-based (9 checks) + LLM scoring
- 🖥️ **Gradio Web UI** — manage the full pipeline through a visual interface
- 📊 **Built-in evaluation** — EM/F1 on NQ, PopQA, Musique, Bamboogle

---

## Pipeline Architecture

```
Stage_2/original_data/  ← Download from HF Hub (HotpotQA & 2WikiMultihopQA)
           │
           ▼
   ┌───────────────┐
   │    Stage 1    │  Tool Variant Generation
   │   (optional)  │  → Expand tool library with semantic variants
   │               │  → BM25 + vector similarity deduplication
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │    Stage 2    │  Tool Selection Labeling
   │               │  → LLM labels tool_select & route_select
   │               │  → 4 routing paths (case1–case4)
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │    Stage 3    │  Multi-turn Dialogue Generation
   │               │  → 29 case types (A/B/C/D groups)
   │               │  → think / tool_call / answer format
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │    Stage 4    │  Data Validation & Scoring
   │               │  → 9 rule-based validation checks
   │               │  → LLM quality scoring (think-action consistency)
   └───────┬───────┘
           │
           ▼
  High-Quality SFT Data
  (SRST / SRMT / MRST / MRMT)
           │
           ▼
   ┌───────────────┐
   │    Training   │  Swift SFT (Qwen3, 4×GPU, Full Params)
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │  Evaluation   │  EM/F1 on NQ / PopQA / Musique / Bamboogle
   └───────────────┘
```

### Dialogue Case Type Overview

| Group | Cases | Scenario |
|-------|-------|----------|
| **A** | A1–A4 | Single-turn: direct answer / single tool / multi-tool / no tool |
| **B** | B1–B6 | Reflection: failed tool call → retry / adjust parameters |
| **C** | C1,C3–C10 | Dual-tool: sequential / parallel / conditional tool chaining |
| **D** | D1–D10 | Complex: multi-hop reasoning with dynamic tool selection |

---

## Project Structure

```
sft_tools/
├── Stage_1/                          # Tool Variant Generation
│   ├── generate_tool.py
│   └── tool_prompts.py
│
├── Stage_2/                          # Tool Selection Labeling
│   ├── code/
│   │   ├── llm_generate_label.py
│   │   └── tool_prompts.py
│   ├── original_data/                # ★ Sample datasets (ready to use)
│   │   ├── HotpotQA/
│   │   │   ├── bridge_hp.parquet
│   │   │   ├── comparison_hp.parquet
│   │   │   └── parquet_to_jsonl.py   # Convert to JSONL before use
│   │   └── 2WikiMultihopQA/
│   │       ├── bridge_comparison_wiki.parquet
│   │       ├── comparison_wiki.parquet
│   │       ├── compositional_wiki.parquet
│   │       ├── inference_wiki.parquet
│   │       └── parquet_to_jsonl.py
│   ├── label_data/output.jsonl       # Stage 2 labeled output (generated at runtime)
│   └── residue_data/output.jsonl
│
├── Stage_3/                          # Multi-turn Dialogue Generation
│   ├── generate_and_judge_main.py
│   ├── config/
│   ├── core/                         # API & MCP clients
│   ├── processors/                   # 29 case processors
│   ├── prompts/                      # Prompt templates
│   ├── services/                     # Generator & tool manager
│   ├── tool_bank/tools/              # 22 domain tool libraries (JSONL)
│   └── utils/                        # BM25, file, text utilities
│
├── Stage_4/                          # Data Validation & Scoring
│   ├── config/
│   ├── core/
│   ├── prompts/
│   ├── utils/
│   └── validators/
│
├── ToolForge_gradio_webui/           # Gradio Web UI
│   ├── quick_fast.py                 # Main entry point
│   ├── feature_generate_judge.py
│   ├── feature_tool_list_manager.py
│   └── feature_tool_variant_generator.py
│
├── Evaluation_Framework/             # Evaluation Framework
│   ├── evaluations/                  # Multi-dataset / multi-model eval
│   │   ├── config/                   # YAML configs
│   │   ├── scripts/
│   │   └── src/                      # Datasets, inference, metrics, models
│   └── rag_server/                   # FastAPI RAG retrieval server
│
├── ourbenchmark_inference_output/    # Custom Benchmark Evaluation
│   ├── model_deploy.sh               # Deploy fine-tuned model via Swift + vLLM
│   ├── our_model_eval.py             # Evaluate fine-tuned model
│   ├── open_source_model_eval.py     # Evaluate baseline models for comparison
│   ├── bm25_utils.py
│   └── viewer_compare.html           # Side-by-side result viewer
│
├── train/                            # Model Training
│   ├── train.sh                      # Swift SFT training script
│   └── qwen3_mix/
│       └── qwen3_think.py            # Custom Qwen3 thinking template
│
├── train_and_eval_data/              # Final Training & Eval Datasets
│   ├── train_data/
│   │   ├── chinese_data/             # MRMT / MRST / SRMT / SRST (Parquet)
│   │   └── english_data/             # MRMT / MRST / SRMT / SRST (Parquet)
│   ├── eval_data/                    # MRMT / MRST / SRMT / SRST eval sets
│   └── parquet_to_jsonl.py           # Convert to JSONL for training
│
├── .env.example                      # Environment variable template
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI-compatible API access (GPT / Claude / local vLLM)
- CUDA-capable GPU (for training and inference only)

### Installation

```bash
git clone https://github.com/Buycar-arb/ToolForge.git
cd ToolForge
pip install -r requirements.txt
```

### Download Datasets

All data files are hosted on 🤗 [buycar/ToolForge](https://huggingface.co/datasets/buycar/ToolForge).
Run the included script to download everything at once:

```bash
pip install huggingface_hub
python download_data.py
```

To also download the **bge-m3** embedding model required by Stage 1:

```bash
python download_data.py --with-model
```

This downloads:
- `Stage_2/original_data/` — HotpotQA & 2WikiMultihopQA source data (for Stage 2)
- `train_and_eval_data/` — Final SFT training & evaluation datasets (Parquet)

### Configure Environment Variables

```bash
cp .env.example .env
# Fill in your API keys and base URL
```

Load before running any script:

```bash
export $(grep -v '^#' .env | xargs)
```

| Variable | Description | Example |
|----------|-------------|---------|
| `API_KEYS` | Comma-separated API keys | `sk-key1,sk-key2` |
| `API_BASE_URL` | API endpoint | `https://api.openai.com/v1` |
| `DEFAULT_MODEL` | Generation model | `gpt-4.1` |
| `JUDGE_MODEL` | Validation model | `gpt-4.1` |
| `SENTENCE_TRANSFORMER_MODEL_PATH` | Local bge-m3 path (Stage 1 only) | `./models/bge-m3` |

### Launch the Web UI

All scripts **must be run from the project root** (`sft_tools/`):

```bash
cd sft_tools
export $(grep -v '^#' .env | xargs)
python ToolForge_gradio_webui/quick_fast.py
```

Open `http://localhost:7860`. The UI covers all pipeline stages in three tabs:
1. **Tool Variant Generator** — Stage 1
2. **Tool Labeling** — Stage 2
3. **Generation & Validation** — Stage 3+4

---

## Detailed Usage

> All scripts must be run from the **project root** (`sft_tools/`) to ensure package imports resolve correctly.

### Stage 1: Tool Variant Generation *(Optional)*

Stage 1 expands your tool library by generating semantically diverse variants. **Skip this stage** if you want to use the 22 built-in tool libraries as-is.

If you want to add or expand tools, Stage 1 requires a local bge-m3 embedding model:

```bash
python download_data.py --with-model
```

Open `Stage_1/generate_tool.py` and configure:

- **Top of file**: `MODEL = "gpt-4"` — model to use for generation
- **Inside `main()`**: `target = 20` — number of variants to generate per tool
- **Inside `main()`**: `output_file` — controlled by `OUTPUT_FILE` env var (default `./output/tool_variants.jsonl`)

Run:

```bash
python -m Stage_1.generate_tool
```

---

### Stage 2: Tool Selection Labeling

**Download the data first** (see [Download Datasets](#download-datasets) in Quick Start), then convert from Parquet to JSONL:

```bash
# HotpotQA
cd Stage_2/original_data/HotpotQA
python parquet_to_jsonl.py
# Outputs: bridge_hp.jsonl, comparison_hp.jsonl

# 2WikiMultihopQA
cd ../2WikiMultihopQA
python parquet_to_jsonl.py
# Outputs: bridge_comparison_wiki.jsonl, comparison_wiki.jsonl, ...
```

Then open `Stage_2/code/llm_generate_label.py` and set the input/output paths at the top:

```python
input_file  = "Stage_2/original_data/HotpotQA/bridge_hp.jsonl"
output_file = "Stage_2/label_data/output.jsonl"
```

Run from the project root:

```bash
python -m Stage_2.code.llm_generate_label
```

The script labels each sample with `tool_select` (which tool to call) and `route_select` (which case type: case1–case4).

---

### Stage 3+4: Dialogue Generation & Validation

Open `Stage_3/generate_and_judge_main.py` and configure the `main()` function:

```python
c_cases_config = {
    'case_C1': (
        100,                                       # target count
        "output/validated/case_C1.jsonl",          # validated data output
        "output/scores/score_C1.jsonl"             # score output
    ),
    # Add more cases as needed...
}

base_config = {
    'input_file': "Stage_2/label_data/output.jsonl",  # Stage 2 output
}
```

Run:

```bash
python -m Stage_3.generate_and_judge_main
```

Scoring:
- **2 / 2** — passes all rule checks + LLM quality check → saved to training data
- **< 2** — filtered out

---

### Model Training

Install Swift:

```bash
pip install ms-swift
```

Edit `train/train.sh` to set your model path and data path, then run:

```bash
cd train
bash train.sh
```

Key configuration:
- Model: Qwen3 (full parameter fine-tuning)
- Sequence length: 12,000 tokens
- Precision: BF16 + Flash Attention 2
- Optimizer: DeepSpeed ZeRO-2

---

## Data Format

### Input Data (Stage 2)

Each record in the source JSONL follows the HotpotQA schema:

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "type": "comparison",
  "level": "hard",
  "context": [
    ["Scott Derrickson", ["Scott Derrickson (born 1966) is an American director..."]],
    ["Ed Wood", ["Edward Davis Wood Jr. (October 10, 1924) was an American director..."]]
  ],
  "supporting_facts": [
    ["Scott Derrickson", 0],
    ["Ed Wood", 0]
  ]
}
```

### Training Data Categories

| Category | Description |
|----------|-------------|
| **SRST** | Single-Round Single-Tool |
| **SRMT** | Single-Round Multi-Tool |
| **MRST** | Multi-Round Single-Tool |
| **MRMT** | Multi-Round Multi-Tool |

Final training data is stored as Parquet in `train_and_eval_data/train_data/` (Chinese & English).

### SFT Dialogue Format

Each training sample uses the `think / tool_call / observation / answer` structure:

```json
{
  "messages": [
    {"role": "system",    "content": "...tool definitions..."},
    {"role": "user",      "content": "Were Scott Derrickson and Ed Wood of the same nationality?"},
    {"role": "assistant", "content": "<think>I need to look up both people's nationalities.</think>\n<tool_call>{\"name\": \"general_information_search\", \"arguments\": {\"query\": \"Scott Derrickson nationality\"}}</tool_call>"},
    {"role": "tool",      "content": "Scott Derrickson is an American film director..."},
    {"role": "assistant", "content": "<think>Scott Derrickson is American. Now check Ed Wood.</think>\n<tool_call>{\"name\": \"general_information_search\", \"arguments\": {\"query\": \"Ed Wood nationality\"}}</tool_call>"},
    {"role": "tool",      "content": "Ed Wood was an American director and screenwriter..."},
    {"role": "assistant", "content": "<think>Both are American, so same nationality.</think>\nyes"}
  ]
}
```

---

## Evaluation

### Standard Benchmarks

See `Evaluation_Framework/README.md` for full setup. Quick start:

```bash
cd Evaluation_Framework/evaluations
python run_evaluation.py --config config/models.yaml --dataset nq --search_mode tag
```

Supports two inference modes:
- **Tag-based** (`<search>query</search>`) — Search-R1 style
- **Function calling** — Standard OpenAI function call format

| Metric | Description |
|--------|-------------|
| **EM** | Exact Match — strict string equality after normalization |
| **F1** | Token-level F1 — overlap between predicted and gold answer tokens |

### Custom Benchmark

Evaluate your fine-tuned model against baseline models on your own data.

#### Step 1 — Deploy the fine-tuned model

Edit `ourbenchmark_inference_output/model_deploy.sh`:

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" swift deploy \
    --model /path/to/your/checkpoint \    # ← set your checkpoint path
    --infer_backend vllm \
    --tensor_parallel_size 4 \
    --max_new_tokens 8192 \
    --served_model_name history-8B
```

```bash
cd ourbenchmark_inference_output
bash model_deploy.sh
# Model served at http://0.0.0.0:8000/v1 as "history-8B"
```

#### Step 2 — Evaluate the fine-tuned model

Open `our_model_eval.py` and set paths at the top:

```python
input_path  = "path/to/stage34_validated_output.jsonl"
output_path = "path/to/our_model_results.jsonl"
```

```bash
cd ourbenchmark_inference_output
python our_model_eval.py
```

#### Step 3 — Evaluate baseline models

```bash
export API_KEYS=sk-your-key
export API_BASE_URL=https://api.openai.com/v1
```

Open `open_source_model_eval.py`, set `input_path`, `output_path`, and the model name in `APICaller(model="...")`, then run:

```bash
python open_source_model_eval.py
```

#### Step 4 — Compare results

Open `viewer_compare.html` in a browser to view the two result files side by side.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **LLM APIs** | OpenAI SDK (AsyncOpenAI), GPT-4.1, Claude-sonnet, DeepSeek, Grok |
| **Training** | [Swift (ModelScope)](https://github.com/modelscope/ms-swift), Transformers, DeepSpeed, Flash-Attention 2 |
| **Inference** | vLLM |
| **Retrieval / RAG** | bm25s, SentenceTransformer (bge-m3), FAISS |
| **Tool Protocol** | MCP (Model Context Protocol) |
| **Data Formats** | JSONL, Parquet (pyarrow), HuggingFace Datasets |
| **Web UI** | Gradio 5.x |
| **API Server** | FastAPI |
| **Async** | asyncio, aiohttp |
| **Monitoring** | Weights & Biases, TensorBoard |

---

<div align="center">

[⬆ Back to Top](#)

</div>
