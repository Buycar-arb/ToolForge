<div align="center">

<h1>🔨&nbsp; ToolForge</h1>

<p><b><a href="https://arxiv.org/abs/2512.16149">ToolForge: A Data Synthesis Pipeline for Multi-Hop Search without Real-World APIs</a></b><br><sub>无需真实 API 的多跳检索数据合成流水线</sub></p>

<p>
  <a href="https://arxiv.org/abs/2512.16149">
    <img src="https://img.shields.io/badge/arXiv-2512.16149-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/buycar/ToolForge">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20%E6%95%B0%E6%8D%AE%E9%9B%86-ToolForge-FFD21E?style=for-the-badge&labelColor=1a2230" alt="Hugging Face"></a>
  <a href="README.md">
    <img src="https://img.shields.io/badge/English-README-EC6708?style=for-the-badge&labelColor=1a2230" alt="English"></a>
</p>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/models-GPT--5.1%20%7C%20Claude%20Opus%205-ec6708" alt="Models">
  <img src="https://img.shields.io/badge/cases-29-6d5bd0" alt="29 cases">
  <img src="https://img.shields.io/badge/checks-9%20rules%20%2B%20LLM%20judge-0e7490" alt="Validation">
  <img src="https://img.shields.io/badge/license-MIT-15803d" alt="MIT">
</p>

<p>
  <a href="#快速开始">快速开始</a> ·
  <a href="#工作原理">工作原理</a> ·
  <a href="#29-种对话形态">对话形态</a> ·
  <a href="docs/architecture.md">架构说明</a> ·
  <a href="#引用">引用</a>
</p>

</div>

---

让模型学会**调用**工具并不难；难的是让它学会**纠错**——察觉工具返回为空、用更好的
参数重试、放弃一开始就选错的工具、在专用工具反复失败后回退到通用搜索。这类轨迹在
真实语料中几乎不会出现。

ToolForge 负责把它造出来：输入多跳问答数据，输出 `<think>` / `<tool_call>` /
`<answer>` 格式的多轮对话，覆盖 **29 种失败与恢复形态**，且每一条都必须通过
**9 项结构校验 + 1 次 LLM 质量评审**才会被写入训练集。

```bash
pip install -e ".[all]"
cp .env.example .env          # 填一个 API key
toolforge doctor              # 检查配置是否就绪
toolforge webui               # 或者直接用命令行
```

---

## 产出的数据长什么样

每条样本一个记录。下面是 `case_C4`——两跳问题，第二次工具调用返回无用内容，
模型用修正后的参数重试：

```jsonc
{"messages": [
  {"role": "system",    "content": "…助手指令 + 9 个工具 schema…"},
  {"role": "user",      "content": "Were Scott Derrickson and Ed Wood of the same nationality?"},

  {"role": "assistant", "content": "<think>\n这是比较类问题，需要分别查两个人的国籍。
                                    先查 Derrickson。\n</think>\n
                                    <tool_call>{\"name\": \"people_data_investigator\",
                                    \"arguments\": {\"query\": \"Scott Derrickson nationality\"}}</tool_call>"},
  {"role": "tool",      "content": "**1**\ntitle: Scott Derrickson\ncontent: … is an American director. …"},

  {"role": "assistant", "content": "<think>\nDerrickson 是美国人。接着查 Ed Wood。\n</think>\n
                                    <tool_call>{\"name\": \"people_data_investigator\",
                                    \"arguments\": {\"query\": \"Wood\"}}</tool_call>"},
  {"role": "tool",      "content": "**1**\ntitle: Wood carving\ncontent: … 无关内容 …"},

  {"role": "assistant", "content": "<think>\n刚才的检索词太宽泛，没有返回导演相关信息，
                                    改用全名重试。\n</think>\n
                                    <tool_call>{\"name\": \"people_data_investigator\",
                                    \"arguments\": {\"query\": \"Ed Wood film director nationality\"}}</tool_call>"},
  {"role": "tool",      "content": "**1**\ntitle: Ed Wood\ncontent: … was an American director …"},

  {"role": "assistant", "content": "<think>\n两人都是美国人。\n</think>\n<answer>\nyes\n</answer>"}
]}
```

有三个细节是手工难以伪造、而 ToolForge 对每条样本都能保证的：

- **失败的调用是真的失败。** 它的 `tool` 内容是在该条数据的**非**支撑段落上真实跑
  BM25 得到的，干扰项是真干扰项，而不是一句 `"error"`。
- **同一个工具不会连续出现两次同名。** 22 个领域工具库、每个约 20 个语义等价的变体，
  每条样本随机抽取其一。模型无法靠背工具名蒙混，只能真的去读 schema。
- **反思是被验证过的。** 评审模型会逐条检查 `<think>` 与随后动作是否一致，以及出错的
  那一轮是否真的在下一轮被反思到。仅仅结构正确是进不了训练集的。

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/Buycar-arb/ToolForge.git
cd ToolForge
pip install -e ".[all]"
```

也可以按需安装：`pip install -e .` 只装核心流水线，可选组件有
`webui`、`anthropic`、`data`、`embeddings`、`eval`、`dev`。

### 2. 配置

```bash
cp .env.example .env
```

一个 key 就够：

```bash
OPENAI_API_KEY=sk-…              # 多个用逗号分隔，程序会自动轮换
GENERATION_MODEL=gpt-5.1         # 负责生成对话
JUDGE_MODEL=gpt-5.1              # 负责质量评审
```

也原生支持 Anthropic：

```bash
ANTHROPIC_API_KEY=sk-ant-…
JUDGE_MODEL=claude-opus-5        # 供应商由模型名自动推断
```

供应商根据模型名判断：`claude-*` 走 Anthropic 原生 Messages API，其余走
OpenAI 兼容接口。需要时可用前缀强制指定——`openai:claude-sonnet-5` 表示通过
网关调用 Claude，`anthropic:claude-opus-5` 表示强制走原生接口。把
`OPENAI_BASE_URL` 指向 Azure、vLLM、OpenRouter 或自建网关，它们提供的任何模型
都可以直接使用。

```bash
toolforge doctor
```

会明确告诉你还缺什么。

> 当前各家模型在 API 层面并不通用——GPT-5 不接受 `max_tokens`，Anthropic SDK 移除了
> `temperature`，不同模型对「是否给 JSON 加代码块围栏」的处理也不一致。ToolForge 会
> 自动协商这三件事，具体原理见 [`docs/models.md`](docs/models.md)。

### 3. 准备数据

```bash
python download_data.py
toolforge convert to-jsonl data/source_qa/HotpotQA
```

### 4. 运行

<table>
<tr><th align="left" width="50%">图形界面</th><th align="left" width="50%">命令行</th></tr>
<tr valign="top"><td>

```bash
toolforge webui
```

打开 `http://localhost:7860`，共五个标签页：

- **Overview** — 配置就绪检查清单
- **Tool bank** — 浏览工具库、编辑 `TOOL_LIST`、运行第一阶段
- **Label** — 第二阶段，带实时进度
- **Generate** — 第三、四阶段
- **Data** — 浏览任意输出文件、重跑校验

</td><td>

```bash
# 第二阶段 —— 工具标注
toolforge label \
  data/source_qa/HotpotQA/bridge_hp.jsonl \
  output/labelled/output.jsonl --limit 200

# 第三 + 四阶段 —— 生成与校验
toolforge generate output/labelled/output.jsonl \
  --case case_C1 --case case_D4 --target 100
```

</td></tr>
</table>

建议先用 `--limit 20 --target 3` 跑一小批，确认产出符合预期再放量。

---

## 工作原理

```
  data/source_qa/            原始多跳问答（HotpotQA 格式）
         │
         ▼
  ┌──────────────┐  第二阶段 · 标注
  │  labeling    │  → 这个问题该用什么工具？需要几轮？
  └──────┬───────┘     写入 reasoning · tool_select · route_select
         │
         ▼
  ┌──────────────┐  第三阶段 · 生成                         ┌───────────────┐
  │  dialogue    │  1. 规划工具调用轨迹                     │  tool_bank/   │
  │              │  2. 检索真实干扰段落              ◀──────│  22 个工具库  │
  │              │  3. 撰写多轮对话                        │  每个约 20 变体│
  └──────┬───────┘  4. 组装成记录                          └───────────────┘
         │
         ▼
  ┌──────────────┐  第四阶段 · 校验
  │  validation  │  9 项规则校验  →  rule_score 0 或 1
  │  judge       │  LLM 质量评审  →  gpt_score  0 或 1
  └──────┬───────┘  只有 2/2 会被保留
         │
         ├──▶  output/data/case_XX.jsonl     训练集
         └──▶  output/scores/case_XX.jsonl   每次尝试的评分与原因
```

**第一阶段**是可选的，独立于上述循环：它通过改写工具定义来扩充工具库，只有
**语义足够接近**（余弦相似度高于阈值）且**措辞足够不同**（BM25 相似度低于阈值）
的变体才会被保留。

### 9 项校验

| # | 校验内容 |
|:-:|---------|
| 1 | 角色序列符合该 case 的预期模式 |
| 2 | assistant 轮次为 `<think>` + `<tool_call>`，最后一轮以 `<answer>` 收尾 |
| 3 | `system` / `user` / `tool` 消息内容非空 |
| 4 | 最终 `<answer>` 与标准答案一致 |
| 5 | 每条 `tool` 消息**精确**还原对应的段落集合——不许编造，不许遗漏 |
| 6 | 重试时只能修改 schema 中标记为 `required` 的参数 |
| 7 | 用到的支撑段落与原始 `supporting_facts` 一致 |
| 8 | 实际调用的工具与第二阶段标注一致 |
| 9 | 每次工具调用的名称和参数都在 system prompt 提供的 schema 范围内 |

每一次尝试——无论保留还是丢弃——都会带着原因写入评分文件，因此产出率是可核查的：

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

> 原始版本中有两项校验实际上从未生效。为保证论文数据可复现，默认行为保持不变，
> 但都提供了开关，详见 [`docs/behaviour-notes.md`](docs/behaviour-notes.md)。

---

## 29 种对话形态

按「需要几轮」和「每轮调用几次」分为四族：

| 族 | 形态 | 对应 case |
|:--:|------|-----------|
| **A** | 单轮，每次尝试调用一次 | `A1`–`A4` |
| **B** | 单轮，每次尝试调用多次 | `B1`–`B6` |
| **C** | 双轮，每轮调用一次 | `C1`、`C3`–`C10` |
| **D** | 双轮，某轮调用多次 | `D1`–`D10` |

同一族内部的差异在于**失败方式**：调用返回为空后用修正参数重试、一开始选错工具、
连续三次失败后回退到通用搜索、或者两跳问题只用一次调用就答完。

```bash
toolforge cases                     # 完整表格
toolforge cases --case case_C9      # 查看单个 case，含推理流程
```

每个 case 都是一条数据声明而非一个类，整套分类体系集中在一个文件里
[`toolforge/stages/cases.py`](toolforge/stages/cases.py)：

```python
_spec("case_C9", "C", (GOLD_ONLY, THREE_STRIKES),
      {"gold_content_1": "gold@1", "gold_content_2": "gold@2",
       "error_content_1": "bad1@2", "error_content_2": "bad2@2", "error_content_3": "bad3@2"},
      ("gold@1", "bad1@2", "bad2@2", "bad3@2", "gold@2"),
      fallback=True, argument_check=(2, 3),
      description="Second hop fails three times, then falls back to general search.")
```

新增第 30 种形态只需在这里加一条，再补上它的 prompt 和 flow——测试会自动覆盖它。

---

## 目录结构

```
toolforge/               主包，所有可导入代码
├─ config.py             全部可调参数，从 .env 解析
├─ llm.py                统一异步客户端：供应商路由、密钥轮换、退避重试
├─ toolbank.py           读取工具库；为每条数据采样工具集合
├─ bm25.py               生成真实干扰段落的检索
├─ convert.py            Parquet ↔ JSONL
├─ cli.py                `toolforge` 命令
├─ stages/
│  ├─ cases.py           ★ 29 种形态，以数据形式声明
│  ├─ dialogue.py        ★ 生成引擎
│  ├─ validation.py      9 项规则校验
│  ├─ judge.py           LLM 评审
│  ├─ pipeline.py        生成 → 校验 → 评分 主循环
│  ├─ labeling.py        第二阶段
│  └─ variants.py        第一阶段
├─ prompts/              全部 prompt，按用途分组
└─ webui/                Gradio 前端

tool_bank/               22 个领域工具库（JSONL，每个约 20 个变体）
data/                    数据集（下载得到）
tests/                   离线测试，无需 API key 和网络
viewer/compare.html      并排轨迹对比页面，单文件自包含
Evaluation_Framework/    NQ / PopQA / Musique / Bamboogle 上的 EM / F1
train/                   Swift SFT 训练脚本
docs/                    架构 · 模型 · 行为说明 · 迁移指南
```

---

## 作为库调用

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

也可以逐条生成：

```python
from toolforge import DialogueGenerator, SourceRecord, validate

record = SourceRecord.parse(raw_stage2_row)
sample = await DialogueGenerator().generate(record, "case_C1")
print(validate(sample.to_record(), "case_C1").passed)
```

---

## 测试

测试用脚本化的假模型驱动真实引擎和真实校验，因此不需要 API key，也不需要联网：

```bash
pip install -e ".[dev]"
pytest
```

每次运行都会把 29 种形态全部生成并校验一遍——分桶、顺序、渲染、校验中任何一处
回退都会立刻暴露。

---

## 训练与评测

两者都依赖 GPU 环境，单独记录在 [`docs/training.md`](docs/training.md)：
Qwen3 的 Swift SFT、NQ / PopQA / Musique / Bamboogle 的 EM / F1 评测框架，
以及自定义 benchmark 的并排对比查看器。

---

## 数据集

🤗 [**buycar/ToolForge**](https://huggingface.co/datasets/buycar/ToolForge)

```bash
python download_data.py              # HotpotQA 与 2WikiMultihopQA
python download_data.py --with-model # 额外下载 bge-m3，用于第一阶段的相似度门控
```

下载得到 `data/source_qa/` —— 第二阶段消费的原始多跳问答数据，也是整条流水线的起点。
三条命令就能把它变成训练数据：

```bash
toolforge convert  to-jsonl data/source_qa/HotpotQA
toolforge label    data/source_qa/HotpotQA/bridge_hp.jsonl data/labelled/output.jsonl
toolforge generate data/labelled/output.jsonl --case case_C1 --target 100
```

因为你运行的是工厂本身而不是下载它的产物，所以你可以在**自己的**语料、**自己的**领域、
**自己的**工具上造出同样的数据。

流水线产出的四个类别：

| 类别 | 含义 |
|------|------|
| **SRST** | 单轮单工具 |
| **SRMT** | 单轮多工具 |
| **MRST** | 多轮单工具 |
| **MRMT** | 多轮多工具 |

哪个语料能产出哪一族对话，既不直观又很关键——实测分布见
[`docs/choosing-source-data.md`](docs/choosing-source-data.md)。

---

## 引用

如果 ToolForge 对你的研究有帮助，欢迎引用：

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
<sub><a href="https://arxiv.org/abs/2512.16149">论文</a> · <a href="docs/architecture.md">架构说明</a> · <a href="docs/models.md">模型说明</a> · <a href="docs/behaviour-notes.md">行为说明</a> · <a href="README.md">English</a><br>MIT 许可证</sub>
</div>
