<div align="center">

<h1>🔧 ToolForge</h1>

<p>
  <a href="README.md">English</a> | <b>中文</b>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/框架-Gradio%205.x-orange?logo=gradio" alt="Gradio">
  <img src="https://img.shields.io/badge/大模型-Qwen3%20%7C%20GPT%20%7C%20Claude-green" alt="LLM">
  <img src="https://img.shields.io/badge/推理-vLLM-purple" alt="vLLM">
  <img src="https://img.shields.io/badge/许可证-MIT-yellow" alt="License">
</p>

<p><em>面向大模型工具调用能力的自动化 SFT 训练数据合成流水线</em></p>

</div>

---

## 目录

- [项目概述](#项目概述)
- [流水线架构](#流水线架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [数据格式](#数据格式)
- [评测](#评测)
- [技术栈](#技术栈)

---

## 项目概述

**ToolForge** 是一个面向**大模型工具调用能力**的端到端自动化 SFT 训练数据合成流水线，覆盖从原始多跳 QA 数据到训练就绪数据、模型微调、评测的完整工作流。

本流水线旨在训练大模型（主要为 **Qwen3**）在复杂多跳问答场景下准确选择并调用工具。支持 **29 种对话 case 类型**，涵盖单轮/多轮、单工具/多工具的多种真实交互模式。

> **数据集托管在 Hugging Face** — 所有数据集（原始数据 + 最终 SFT 训练数据）均发布于 🤗 [buycar/ToolForge](https://huggingface.co/datasets/buycar/ToolForge)。下载方式见[快速开始](#快速开始)。

### 核心特性

- 🏭 **全流程自动化** — 从原始 QA 数据到验证通过的 SFT 训练数据
- 📦 **公开数据集** — HotpotQA & 2WikiMultihopQA，托管于 HF Hub，一键下载
- 🔀 **29 种对话类型** — A1-A4（单轮）、B1-B6（反思）、C1/C3-C10（双工具）、D1-D10（复杂）
- 🌐 **中英双语支持** — 同时生成中英文训练数据集
- 🛠️ **22 个领域工具库** — 涵盖学术、医疗、地理、经济等多个领域
- ✅ **双阶段数据验证** — 规则验证（9 项检查）+ LLM 质量评分
- 🖥️ **Gradio Web UI** — 通过可视化界面管理完整流水线
- 📊 **内置评测框架** — 支持 NQ、PopQA、Musique、Bamboogle 的 EM/F1 评测

---

## 流水线架构

```
Stage_2/original_data/  ← 从 HF Hub 下载（HotpotQA & 2WikiMultihopQA）
           │
           ▼
   ┌───────────────┐
   │   第一阶段    │  工具变体生成（可选）
   │               │  → 为工具库生成语义多样的变体
   │               │  → BM25 + 向量相似度双重去重
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │   第二阶段    │  工具选择标注
   │               │  → LLM 标注 tool_select 和 route_select
   │               │  → 4 种路由路径（case1–case4）
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │   第三阶段    │  多轮对话生成
   │               │  → 29 种 case 类型（A/B/C/D 组）
   │               │  → think / tool_call / answer 格式
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │   第四阶段    │  数据验证与评分
   │               │  → 9 项规则验证
   │               │  → LLM 质量打分（思考-行动一致性）
   └───────┬───────┘
           │
           ▼
  高质量 SFT 训练数据
  (SRST / SRMT / MRST / MRMT)
           │
           ▼
   ┌───────────────┐
   │   模型训练    │  Swift SFT（Qwen3，4×GPU，全参数微调）
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │   模型评测    │  EM/F1 on NQ / PopQA / Musique / Bamboogle
   └───────────────┘
```

### 对话 Case 类型说明

| 组别 | Case | 场景说明 |
|------|------|----------|
| **A 组** | A1–A4 | 单轮场景：直接回答 / 单工具调用 / 多工具调用 / 无需工具 |
| **B 组** | B1–B6 | 反思场景：工具调用失败 → 重试 / 调整参数后重新调用 |
| **C 组** | C1,C3–C10 | 双工具场景：顺序调用 / 并行调用 / 条件链式调用 |
| **D 组** | D1–D10 | 复杂场景：多跳推理中动态选择工具 |

---

## 项目结构

```
sft_tools/
├── Stage_1/                          # 第一阶段：工具变体生成
│   ├── generate_tool.py
│   └── tool_prompts.py
│
├── Stage_2/                          # 第二阶段：工具选择标注
│   ├── code/
│   │   ├── llm_generate_label.py
│   │   └── tool_prompts.py
│   ├── original_data/                # ★ 从 HF Hub 下载后放置于此
│   │   ├── HotpotQA/
│   │   │   ├── bridge_hp.parquet
│   │   │   ├── comparison_hp.parquet
│   │   │   └── parquet_to_jsonl.py   # 使用前先转换为 JSONL
│   │   └── 2WikiMultihopQA/
│   │       ├── bridge_comparison_wiki.parquet
│   │       ├── comparison_wiki.parquet
│   │       ├── compositional_wiki.parquet
│   │       ├── inference_wiki.parquet
│   │       └── parquet_to_jsonl.py
│   ├── label_data/output.jsonl       # 第二阶段标注输出（运行后自动生成）
│   └── residue_data/output.jsonl
│
├── Stage_3/                          # 第三阶段：多轮对话生成
│   ├── generate_and_judge_main.py
│   ├── config/
│   ├── core/                         # API 客户端 & MCP 客户端
│   ├── processors/                   # 29 种 case 处理器
│   ├── prompts/                      # Prompt 模板
│   ├── services/                     # 对话生成路由 & 工具管理
│   ├── tool_bank/tools/              # 22 个领域工具库（JSONL）
│   └── utils/                        # BM25、文件、文本工具
│
├── Stage_4/                          # 第四阶段：数据验证与评分
│   ├── config/
│   ├── core/
│   ├── prompts/
│   ├── utils/
│   └── validators/
│
├── ToolForge_gradio_webui/           # Gradio Web UI
│   ├── quick_fast.py                 # 主入口
│   ├── feature_generate_judge.py
│   ├── feature_tool_list_manager.py
│   └── feature_tool_variant_generator.py
│
├── Evaluation_Framework/             # 模型评测框架
│   ├── evaluations/                  # 多数据集/多模型评测
│   │   ├── config/                   # YAML 配置文件
│   │   ├── scripts/
│   │   └── src/                      # 数据集、推理、指标、模型
│   └── rag_server/                   # FastAPI RAG 检索服务
│
├── ourbenchmark_inference_output/    # 自定义 Benchmark 评测
│   ├── model_deploy.sh               # 用 Swift + vLLM 部署微调模型
│   ├── our_model_eval.py             # 评测微调模型
│   ├── open_source_model_eval.py     # 评测基座模型（对比用）
│   ├── bm25_utils.py
│   └── viewer_compare.html           # 可视化对比查看器
│
├── train/                            # 模型训练
│   ├── train.sh                      # Swift SFT 训练脚本
│   └── qwen3_mix/
│       └── qwen3_think.py            # 自定义 Qwen3 Thinking 模板
│
├── train_and_eval_data/              # 最终训练集与评估集
│   ├── train_data/
│   │   ├── chinese_data/             # MRMT / MRST / SRMT / SRST（Parquet）
│   │   └── english_data/             # MRMT / MRST / SRMT / SRST（Parquet）
│   ├── eval_data/                    # MRMT / MRST / SRMT / SRST 评估集
│   └── parquet_to_jsonl.py           # 转换为 JSONL 供训练使用
│
├── .env.example                      # 环境变量配置模板
└── requirements.txt
```

---

## 快速开始

### 环境要求

- Python 3.10+
- 兼容 OpenAI API 的接口（GPT / Claude / 本地 vLLM 均可）
- 支持 CUDA 的 GPU（仅训练和推理阶段需要）

### 安装

```bash
git clone https://github.com/Buycar-arb/ToolForge.git
cd ToolForge
pip install -r requirements.txt
```

### 下载数据集

所有数据文件托管于 🤗 [buycar/ToolForge](https://huggingface.co/datasets/buycar/ToolForge)。
运行项目内置脚本一键下载：

```bash
pip install huggingface_hub
python download_data.py
```

如果还需要下载 Stage 1 所需的 **bge-m3** 嵌入模型：

```bash
python download_data.py --with-model
```

下载内容：
- `Stage_2/original_data/` — HotpotQA & 2WikiMultihopQA 原始数据（供第二阶段使用）
- `train_and_eval_data/` — 最终 SFT 训练集和评测集（Parquet 格式）

### 配置环境变量

```bash
cp .env.example .env
# 用编辑器打开 .env，填写 API key 和接口地址
```

运行任意脚本前加载环境变量：

```bash
export $(grep -v '^#' .env | xargs)
```

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `API_KEYS` | API 密钥列表，逗号分隔 | `sk-key1,sk-key2` |
| `API_BASE_URL` | API 接口地址 | `https://api.openai.com/v1` |
| `DEFAULT_MODEL` | 数据生成模型 | `gpt-4.1` |
| `JUDGE_MODEL` | 质量验证模型 | `gpt-4.1` |
| `SENTENCE_TRANSFORMER_MODEL_PATH` | bge-m3 模型本地路径（仅第一阶段） | `./models/bge-m3` |

### 启动 Web UI

所有脚本**必须从项目根目录**（`sft_tools/`）运行：

```bash
cd sft_tools
export $(grep -v '^#' .env | xargs)
python ToolForge_gradio_webui/quick_fast.py
```

在浏览器中打开 `http://localhost:7860`，界面包含三个 Tab：
1. **工具变体生成** — 第一阶段
2. **工具标注** — 第二阶段
3. **数据生成与校验** — 第三+四阶段

---

## 详细使用说明

> 所有脚本必须从**项目根目录**（`sft_tools/`）运行，以保证 Python 包导入路径正确解析。

### 第一阶段：工具变体生成（可选）

第一阶段为工具库生成语义多样的变体。**如果使用内置的 22 个工具库，可以跳过此阶段。**

如需扩充工具库，先下载 bge-m3 嵌入模型：

```bash
python download_data.py --with-model
```

打开 `Stage_1/generate_tool.py`，配置以下变量：

- **文件顶部**：`MODEL = "gpt-4"` — 用于生成的模型名
- **`main()` 内**：`target = 20` — 每个工具生成的变体数量
- **`main()` 内**：`output_file` — 由环境变量 `OUTPUT_FILE` 控制（默认 `./output/tool_variants.jsonl`）

运行：

```bash
python -m Stage_1.generate_tool
```

---

### 第二阶段：工具选择标注

**先从 HF Hub 下载数据集**（见[下载数据集](#下载数据集)），然后将 Parquet 转换为 JSONL：

```bash
# HotpotQA
cd Stage_2/original_data/HotpotQA
python parquet_to_jsonl.py
# 输出：bridge_hp.jsonl, comparison_hp.jsonl

# 2WikiMultihopQA
cd ../2WikiMultihopQA
python parquet_to_jsonl.py
# 输出：bridge_comparison_wiki.jsonl, comparison_wiki.jsonl, ...
```

然后打开 `Stage_2/code/llm_generate_label.py`，修改顶部的路径配置：

```python
input_file  = "Stage_2/original_data/HotpotQA/bridge_hp.jsonl"
output_file = "Stage_2/label_data/output.jsonl"
```

从项目根目录运行：

```bash
python -m Stage_2.code.llm_generate_label
```

脚本会为每条样本标注 `tool_select`（调用哪个工具）和 `route_select`（case 类型：case1–case4）。

---

### 第三+四阶段：对话生成与验证

打开 `Stage_3/generate_and_judge_main.py`，修改 `main()` 函数中的配置：

```python
c_cases_config = {
    'case_C1': (
        100,                                       # 目标生成数量
        "output/validated/case_C1.jsonl",          # 验证通过数据的输出路径
        "output/scores/score_C1.jsonl"             # 评分结果输出路径
    ),
    # 可继续添加其他 case...
}

base_config = {
    'input_file': "Stage_2/label_data/output.jsonl",  # 第二阶段的输出
}
```

运行：

```bash
python -m Stage_3.generate_and_judge_main
```

评分规则：
- **2 / 2** — 规则验证全部通过 + LLM 质量检查通过 → 保存到训练数据
- **< 2** — 过滤淘汰

---

### 模型训练

安装 Swift：

```bash
pip install ms-swift
```

编辑 `train/train.sh` 设置模型路径和数据路径，然后运行：

```bash
cd train
bash train.sh
```

训练核心配置：
- 模型：Qwen3 全参数微调
- 最大序列长度：12,000 tokens
- 精度：BF16 + Flash Attention 2
- 优化器：DeepSpeed ZeRO-2

---

## 数据格式

### 输入数据（第二阶段）

`Stage_2/original_data/` 中的数据遵循 HotpotQA 格式：

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

### 训练数据分类

| 类别 | 说明 |
|------|------|
| **SRST** | 单轮单工具（Single-Round Single-Tool） |
| **SRMT** | 单轮多工具（Single-Round Multi-Tool） |
| **MRST** | 多轮单工具（Multi-Round Single-Tool） |
| **MRMT** | 多轮多工具（Multi-Round Multi-Tool） |

最终训练数据以 Parquet 格式存储于 `train_and_eval_data/train_data/`（中英文各一份）。

### SFT 对话格式

每条训练样本遵循 `think / tool_call / observation / answer` 多轮结构：

```json
{
  "messages": [
    {"role": "system",    "content": "...工具定义..."},
    {"role": "user",      "content": "Scott Derrickson 和 Ed Wood 是同一国籍吗？"},
    {"role": "assistant", "content": "<think>需要查两人的国籍。</think>\n<tool_call>{\"name\": \"general_information_search\", \"arguments\": {\"query\": \"Scott Derrickson 国籍\"}}</tool_call>"},
    {"role": "tool",      "content": "Scott Derrickson 是美国电影导演..."},
    {"role": "assistant", "content": "<think>Derrickson 是美国人，再查 Ed Wood。</think>\n<tool_call>{\"name\": \"general_information_search\", \"arguments\": {\"query\": \"Ed Wood 国籍\"}}</tool_call>"},
    {"role": "tool",      "content": "Ed Wood 是美国导演和编剧..."},
    {"role": "assistant", "content": "<think>两人都是美国人，国籍相同。</think>\n是"}
  ]
}
```

---

## 评测

### 标准基准测试

完整配置请参阅 `Evaluation_Framework/README.md`。快速启动：

```bash
cd Evaluation_Framework/evaluations
python run_evaluation.py --config config/models.yaml --dataset nq --search_mode tag
```

支持两种推理模式：
- **标签式推理**（`<search>query</search>`）— Search-R1 风格
- **Function calling** — 标准 OpenAI 函数调用格式

| 指标 | 说明 |
|------|------|
| **EM** | 精确匹配（Exact Match）— 标准化后严格字符串相等 |
| **F1** | Token 级 F1 — 预测答案与标准答案的词级重叠度 |

### 自定义 Benchmark 评测

在自定义数据上对比微调模型与基座模型的表现。

#### 第一步 — 部署微调模型

编辑 `ourbenchmark_inference_output/model_deploy.sh`：

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" swift deploy \
    --model /path/to/your/checkpoint \    # ← 修改为你的检查点路径
    --infer_backend vllm \
    --tensor_parallel_size 4 \
    --max_new_tokens 8192 \
    --served_model_name history-8B
```

```bash
cd ourbenchmark_inference_output
bash model_deploy.sh
# 模型以 "history-8B" 为名在 http://0.0.0.0:8000/v1 提供服务
```

#### 第二步 — 评测微调模型

打开 `our_model_eval.py`，修改顶部的路径：

```python
input_path  = "path/to/stage34_validated_output.jsonl"
output_path = "path/to/our_model_results.jsonl"
```

```bash
cd ourbenchmark_inference_output
python our_model_eval.py
```

#### 第三步 — 评测基座模型（对比用）

```bash
export API_KEYS=sk-your-key
export API_BASE_URL=https://api.openai.com/v1
```

打开 `open_source_model_eval.py`，设置 `input_path`、`output_path` 和 `APICaller(model="...")` 中的模型名，然后运行：

```bash
python open_source_model_eval.py
```

#### 第四步 — 可视化对比结果

在浏览器中打开 `viewer_compare.html`，加载两个结果文件，逐条对比模型输出。

---

## 技术栈

| 类别 | 技术 / 框架 |
|------|------------|
| **LLM API** | OpenAI SDK（AsyncOpenAI）、GPT-4.1、Claude-sonnet、DeepSeek、Grok |
| **训练框架** | [Swift（ModelScope）](https://github.com/modelscope/ms-swift)、Transformers、DeepSpeed、Flash-Attention 2 |
| **推理部署** | vLLM |
| **检索 / RAG** | bm25s、SentenceTransformer（bge-m3）、FAISS |
| **工具协议** | MCP（Model Context Protocol） |
| **数据格式** | JSONL、Parquet（pyarrow）、HuggingFace Datasets |
| **Web UI** | Gradio 5.x |
| **API 服务** | FastAPI |
| **异步并发** | asyncio、aiohttp |
| **训练监控** | Weights & Biases、TensorBoard |

---

<div align="center">

[⬆ 回到顶部](#)

</div>
