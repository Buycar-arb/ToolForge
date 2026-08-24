# Source QA data

The input to **stage 2**: raw multi-hop question answering, in HotpotQA's schema.

The Parquet files live on Hugging Face, not in this repository.

```bash
python download_data.py                          # from the repository root
toolforge convert to-jsonl data/source_qa/HotpotQA
toolforge convert to-jsonl data/source_qa/2WikiMultihopQA
```

🤗 [**buycar/ToolForge**](https://huggingface.co/datasets/buycar/ToolForge)

## What arrives

| directory | files | question type |
|-----------|-------|---------------|
| `HotpotQA/` | `bridge_hp`, `comparison_hp` | bridge (hop A → hop B) and comparison (A vs B) |
| `2WikiMultihopQA/` | `bridge_comparison_wiki`, `comparison_wiki`, `compositional_wiki`, `inference_wiki` | four reasoning shapes over Wikipedia |

## Record schema

Stage 2 reads these four fields and adds three of its own:

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "type": "comparison",
  "context": [
    ["Scott Derrickson", ["Scott Derrickson (born 1966) is an American director."]],
    ["Ed Wood", ["Edward Davis Wood Jr. was an American director and screenwriter."]]
  ],
  "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]]
}
```

`supporting_facts` is what makes the whole pipeline work: it splits `context`
into the passages that contain the answer and everything else. Stage 3 uses the
first as the gold evidence a tool call should surface, and runs BM25 over the
second to produce realistic distractors — which is how a *failed* tool call in
the generated data can be genuinely, plausibly unhelpful.

## Using your own data

Any corpus matching the schema above works. The only hard requirements are
`question`, `answer`, `context` and `supporting_facts`, with `supporting_facts`
referencing titles and sentence indices that exist in `context`.

---

# 中文说明

**第二阶段**的输入：原始多跳问答数据，采用 HotpotQA 的格式。

Parquet 文件托管在 Hugging Face，不随仓库分发：

```bash
python download_data.py                          # 在仓库根目录执行
toolforge convert to-jsonl data/source_qa/HotpotQA
toolforge convert to-jsonl data/source_qa/2WikiMultihopQA
```

`supporting_facts` 是整条流水线的关键：它把 `context` 切成「包含答案的支撑段落」
和「其余段落」。第三阶段用前者作为工具调用应当命中的黄金证据，对后者跑 BM25 生成
真实的干扰项——这正是生成数据里那些**失败的**工具调用能够真实、可信地「查不到东西」
的原因。

只要满足上述 schema，任何自有语料都可以直接使用。
