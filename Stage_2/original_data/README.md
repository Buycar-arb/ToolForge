# Stage 2 — Original Data

<p>
  <b>English</b> | <a href="#中文说明">中文</a>
</p>

---

## English

This directory contains the source datasets used in **Stage 2: Tool Selection Labeling**.

The Parquet files are hosted on **Hugging Face Hub** and are NOT included in the Git repository.
Download them before running Stage 2.

### Dataset on Hugging Face

🤗 **[buycar/ToolForge](https://huggingface.co/datasets/buycar/ToolForge)**

### Download

Run the following from the **project root** (`sft_tools/`):

```bash
pip install huggingface_hub
python download_data.py
```

After downloading, convert Parquet to JSONL before running Stage 2:

```bash
# HotpotQA
cd Stage_2/original_data/HotpotQA
python parquet_to_jsonl.py
# Output: bridge_hp.jsonl, comparison_hp.jsonl

# 2WikiMultihopQA
cd ../2WikiMultihopQA
python parquet_to_jsonl.py
# Output: bridge_comparison_wiki.jsonl, comparison_wiki.jsonl, ...
```

### Directory Structure

```
original_data/
├── HotpotQA/
│   ├── bridge_hp.parquet          # 246 MB — multi-hop bridge questions
│   ├── comparison_hp.parquet      #  53 MB — comparison questions
│   └── parquet_to_jsonl.py        # conversion script
└── 2WikiMultihopQA/
    ├── bridge_comparison_wiki.parquet   #  71 MB
    ├── comparison_wiki.parquet          #  92 MB
    ├── compositional_wiki.parquet       # 135 MB
    ├── inference_wiki.parquet           #  10 MB
    └── parquet_to_jsonl.py              # conversion script
```

### Data Sources

| Dataset | Paper | Original Source |
|---------|-------|-----------------|
| HotpotQA | [Yang et al., 2018](https://arxiv.org/abs/1809.09600) | [hotpotqa/hotpot_data](https://github.com/hotpotqa/hotpot_data) |
| 2WikiMultihopQA | [Ho et al., 2020](https://arxiv.org/abs/2011.01060) | [Alab-NLP/2WikiMultihopQA](https://github.com/Alab-NLP/2WikiMultihopQA) |

---

## 中文说明

本目录存放 **第二阶段（工具选择标注）** 所需的原始数据集。

Parquet 数据文件托管于 **Hugging Face Hub**，不包含在 Git 仓库中，使用前需先下载。

### 数据集地址

🤗 **[buycar/ToolForge](https://huggingface.co/datasets/buycar/ToolForge)**

### 下载方式

在**项目根目录**（`sft_tools/`）下运行：

```bash
pip install huggingface_hub
python download_data.py
```

下载完成后，运行转换脚本将 Parquet 转为 JSONL，再执行第二阶段：

```bash
# HotpotQA
cd Stage_2/original_data/HotpotQA
python parquet_to_jsonl.py
# 输出：bridge_hp.jsonl, comparison_hp.jsonl

# 2WikiMultihopQA
cd ../2WikiMultihopQA
python parquet_to_jsonl.py
# 输出：bridge_comparison_wiki.jsonl, comparison_wiki.jsonl 等
```

### 文件说明

| 文件 | 大小 | 内容 |
|------|------|------|
| `HotpotQA/bridge_hp.parquet` | 246 MB | 多跳桥接类问题 |
| `HotpotQA/comparison_hp.parquet` | 53 MB | 对比类问题 |
| `2WikiMultihopQA/bridge_comparison_wiki.parquet` | 71 MB | 桥接+对比混合 |
| `2WikiMultihopQA/comparison_wiki.parquet` | 92 MB | 对比类问题 |
| `2WikiMultihopQA/compositional_wiki.parquet` | 135 MB | 组合推理类问题 |
| `2WikiMultihopQA/inference_wiki.parquet` | 10 MB | 推断类问题 |
