# 选择源语料 / Choosing source data

**English below.**

---

## 为什么这件事重要

第 2 阶段给每个问题标一个 **route**，而 route 决定了这条记录**只能**用来生成哪一族对话：

| route | 含义 | 对应族 | 包含的形态 |
|---|---|:---:|---|
| `case1` | 单轮，调用一次 | **A** | `case_A1`–`A4` |
| `case2` | 单轮，调用多次 | **B** | `case_B1`–`B6` |
| `case3` | 双轮，每轮一次 | **C** | `case_C1`, `C3`–`C10` |
| `case4` | 双轮，某轮多次 | **D** | `case_D1`–`D10` |

route 不是随便标的，它取决于**问题本身的结构**。而问题结构又取决于你用哪个语料——
所以「想要 D 族数据」这件事，在选语料的时候就已经决定了。

拿单领域的语料去跑 D 族，一条都出不来。`toolforge generate` 会直接告诉你：

```
⚠️  case_D9 suits records routed to case4, but none of the 40 records carry
    that route — falling back to all of them, which tends to lower the yield
```

## 各语料的 route 分布

对 🤗 `buycar/ToolForge` 的六个语料各抽样标注后实测：

| 语料 | 总条数 | 抽样 | case1→A | case2→B | case3→C | case4→D | 适合 |
|---|---:|---:|---:|---:|---:|---:|---|
| `inference_wiki` | 4,379 | 40 | **37** | 3 | 0 | 0 | **A** |
| `comparison_wiki` | 51,963 | 25 | 0 | **25** | 0 | 0 | **B** |
| `comparison_hp` | 17,456 | 25 | 4 | **21** | 0 | 0 | A, **B** |
| `compositional_wiki` | 76,481 | 30 | 4 | 0 | **25** | 1 | **C** |
| `bridge_hp` | 72,991 | 25 | **19** | 1 | 5 | 0 | **A**, C |
| `bridge_comparison_wiki` | 34,631 | 25 | 1 | **11** | **9** | **4** | **B, C, D** |

规律和语料名字对得上：

- **`inference_wiki`** —— 亲属关系推理，全在人物这一个领域，一次查询就能拿到 → 几乎全是 A
- **`comparison_*`** —— 「A 和 B 哪个先 / 是否相同」，一轮里查两个实体 → **B**（单轮多次调用）
- **`compositional_wiki`** —— 「X 的 Y 的 Z」，前一跳的结果是后一跳的输入 → **C**（双轮）
- **`bridge_comparison_wiki`** —— 桥接与比较混合，四类都有，**是唯一能稳定产出 D 族的语料**
- **`bridge_hp`** —— 桥接类，但 HotpotQA 的桥接常常一次就能查到 → 偏 A

## 配语料的建议

```bash
# A 族（4 种形态）
toolforge label data/source_qa/2WikiMultihopQA/inference_wiki.jsonl        out/A.jsonl --limit 500

# B 族（6 种）—— comparison_wiki 最纯
toolforge label data/source_qa/2WikiMultihopQA/comparison_wiki.jsonl       out/B.jsonl --limit 500

# C 族（9 种）
toolforge label data/source_qa/2WikiMultihopQA/compositional_wiki.jsonl    out/C.jsonl --limit 500

# D 族（10 种）—— 只有 bridge_comparison_wiki 稳定产出，且占比仅约 16%，要多标
toolforge label data/source_qa/2WikiMultihopQA/bridge_comparison_wiki.jsonl out/D.jsonl --limit 3000
```

**D 族最稀缺**：抽样里只有 4/25 是 case4，所以要拿到同样数量的 D 族数据，标注量得放大
六倍以上。这也是 `toolforge label --single-call` 存在的原因——某些类别天然产不够，
需要用更严格的提示词专门补足。

标注完随时可以查分布：

```bash
toolforge generate out/D.jsonl --case case_D9 --target 10
#  → 载入时会打印每个 route 有多少条、对应哪些形态
```

## 用自己的语料

只要符合 HotpotQA 的 schema 就行（见 `data/source_qa/README.md`）。想让某一族多出数据，
就往对应的问题结构上凑：

- 想要 **B 族** → 比较类问题（「X 和 Y 哪个……」）
- 想要 **C / D 族** → 组合类问题（「X 的 Y 的 Z 是什么」），且两跳最好落在**不同领域**，
  这样第 2 阶段才会选出两个不同的工具

---

# English

## Why this matters

Stage 2 labels every question with a **route**, and that route decides which case
family the record can produce:

| route | meaning | family | cases |
|---|---|:---:|---|
| `case1` | one turn, one call | **A** | `case_A1`–`A4` |
| `case2` | one turn, several calls | **B** | `case_B1`–`B6` |
| `case3` | two turns, one call each | **C** | `case_C1`, `C3`–`C10` |
| `case4` | two turns, several calls | **D** | `case_D1`–`D10` |

The route follows from the *shape of the question*, which follows from the corpus.
So "I want D-family data" is a decision you make when picking source data, not later.

## Measured route distribution

Sampled and labelled from each corpus in 🤗 `buycar/ToolForge`:

| corpus | records | sampled | case1→A | case2→B | case3→C | case4→D | good for |
|---|---:|---:|---:|---:|---:|---:|---|
| `inference_wiki` | 4,379 | 40 | **37** | 3 | 0 | 0 | **A** |
| `comparison_wiki` | 51,963 | 25 | 0 | **25** | 0 | 0 | **B** |
| `comparison_hp` | 17,456 | 25 | 4 | **21** | 0 | 0 | A, **B** |
| `compositional_wiki` | 76,481 | 30 | 4 | 0 | **25** | 1 | **C** |
| `bridge_hp` | 72,991 | 25 | **19** | 1 | 5 | 0 | **A**, C |
| `bridge_comparison_wiki` | 34,631 | 25 | 1 | **11** | **9** | **4** | **B, C, D** |

- **comparison** corpora ask "which of X and Y…", answered by querying two entities
  in one turn → **B**
- **compositional** asks "the Z of the Y of X", where each hop feeds the next → **C**
- **bridge_comparison** mixes both and is the only corpus that reliably yields **D**
- **inference** stays inside one domain, so one call usually suffices → **A**

**D is the scarce one** — roughly 16% of that corpus. Label six times as much to get
the same number of D samples. That scarcity is also why `toolforge label --single-call`
exists: some classes need a dedicated prompt to top them up.

`toolforge generate` prints the route distribution when it loads a file, and warns
when the cases you asked for do not match the records you have.
