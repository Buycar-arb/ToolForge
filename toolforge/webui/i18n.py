"""Interface language for the Web UI.

Chinese is the default; English is available with ``toolforge webui --lang en``
or ``UI_LANG=en``.  Every user-visible string in :mod:`toolforge.webui` goes
through :func:`t`, so adding a language means adding one dictionary.

Data is never translated.  The nine check labels are written to the score files
in English so runs stay comparable across languages and against the published
results; :func:`translate_log` renders them in the chosen language for display
only.
"""

from __future__ import annotations

import os
import re

#: The language in use.  Set once at startup by :func:`set_language`.
_current = "zh"

SUPPORTED = ("zh", "en")


def set_language(language: str | None) -> str:
    """Select the interface language, falling back to Chinese."""
    global _current
    choice = (language or os.getenv("UI_LANG") or "zh").strip().lower()
    _current = choice if choice in SUPPORTED else "zh"
    return _current


def current() -> str:
    return _current


def t(key: str, **kwargs: object) -> str:
    """Look up ``key`` in the active language, formatting any placeholders."""
    table = STRINGS.get(_current, STRINGS["zh"])
    text = table.get(key) or STRINGS["zh"].get(key) or STRINGS["en"].get(key) or key
    return text.format(**kwargs) if kwargs else text


# --------------------------------------------------------------------------- #
# Chinese
# --------------------------------------------------------------------------- #

ZH: dict[str, str] = {
    # -- shell -------------------------------------------------------------- #
    "app.subtitle": "大模型工具调用 SFT 数据自动化工厂 · 第 1–4 阶段",
    "tab.overview": "总览",
    "tab.toolbank": "工具库",
    "tab.label": "标注 · 第 2 阶段",
    "tab.generate": "生成 · 第 3+4 阶段",
    "tab.data": "数据",

    # -- status chips ------------------------------------------------------- #
    "chip.gen": "生成",
    "chip.judge": "评审",
    "chip.keys": "{count} 个 API key",
    "chip.libraries": "{count} 个工具库",

    # -- stage rail --------------------------------------------------------- #
    "rail.1.stage": "第 1 阶段",
    "rail.1.title": "工具库",
    "rail.1.desc": "把工具定义改写成语义等价的变体，逼模型读描述而不是背名字。",
    "rail.2.stage": "第 2 阶段",
    "rail.2.title": "标注",
    "rail.2.desc": "为每个问题标注该调用哪个工具、以及走哪条执行路径。",
    "rail.3.stage": "第 3 阶段",
    "rail.3.title": "对话生成",
    "rail.3.desc": "按 29 种形态之一，生成完整的多轮工具调用对话。",
    "rail.4.stage": "第 4 阶段",
    "rail.4.title": "校验",
    "rail.4.desc": "9 项规则校验加 LLM 评审，只有满分 2/2 才保留。",

    # -- shared components -------------------------------------------------- #
    "model.info": "可以选预设，也可以直接输入你的服务端支持的任意模型 ID。",
    "temp.label": "Temperature",
    "temp.info": "流水线求可复现就填 0；只有第 1 阶段追求多样性时才调高。",
    "tokens.label": "最大 token 数",
    "tokens.info": "长对话需要余量，8192 是比较安全的下限。",
    "status.waiting": "_点击开始后，{what}会显示在这里。_",
    "status.result": "运行结果",
    "log.label": "实时日志",
    "log.placeholder": "任务运行时进度会实时滚动显示…",

    # -- file inspector ----------------------------------------------------- #
    "inspect.path": "JSONL 文件",
    "inspect.path.info": "任意 JSONL：原始输入、标注结果、生成数据或评分文件都可以。",
    "inspect.load": "↻ 加载",
    "inspect.none": "_尚未加载文件。_",
    "inspect.notfound": "_文件不存在。_",
    "inspect.empty": "// 文件为空",
    "inspect.nothing": "// 未加载任何内容",
    "inspect.record": "第几条",
    "inspect.content": "记录内容",
    "inspect.oob": "// 序号超出范围（1-{total}）",
    "inspect.missing": "// 文件不存在：{path}",
    "inspect.summary": "**{name}** · {count} 条记录 · {size} KB · 修改于 {modified}",

    # -- overview ----------------------------------------------------------- #
    "overview.readiness": "### 就绪检查",
    "overview.recheck": "↻ 重新检查配置",
    "overview.check.gen": "生成模型",
    "overview.check.judge": "评审模型",
    "overview.check.sdk": "Anthropic SDK",
    "overview.check.bank": "工具库",
    "overview.check.bm25": "BM25 检索",
    "overview.check.env": ".env 文件",
    "overview.check.env.ok": "已加载 `{name}`",
    "overview.check.env.missing": "把 `.env.example` 复制成 `.env` 并填写",
    "overview.check.env.unused": "文件不存在——但环境变量已提供全部配置",
    "overview.check.installed": "已安装",
    "overview.check.needkey": "`{model}`，走 **{provider}** —— 需要设置 `{variable}`",
    "overview.check.model": "`{model}`，走 **{provider}**",
    "overview.check.sdk.missing": "`pip install 'toolforge[anthropic]'`",
    "overview.check.bank.detail": "{count} 个工具库，位于 `{path}`",
    "overview.check.bm25.missing": "`pip install bm25s jieba`",
    "overview.header": "| | 检查项 | 详情 |\n|---|-------|------|",
    "overview.ready": "\n\n**一切就绪，可以开始。**",
    "overview.notready": "\n\n**还不能运行** —— 待解决：{blocked}。",
    "overview.config.accordion": "当前生效的配置",
    "overview.config.label": "配置详情",
    "overview.panel": """
<div class="tf-panel">
<strong>配置</strong>来自仓库根目录的 <code>.env</code> 文件。<br><br>
界面上所有操作都有等价的命令行：<br>
<code>toolforge doctor</code> · <code>toolforge label</code> · <code>toolforge generate</code><br><br>
<span style="color:var(--tf-muted)">ToolForge v{version}</span>
</div>
""",
    "overview.body": """
### 这个工具在造什么

ToolForge 把多跳问答数据，加工成**教会模型调用工具的监督微调数据**——尤其是野生语料里
几乎不存在的那部分：选错工具后纠正、参数不对时重试、专用工具反复查不到时回退到通用搜索。

每条样本都是一段完整对话，格式为 `<think>` / `<tool_call>` / `<answer>`，并且每条都是
**挣来的**：必须通过 9 项结构校验和一次 LLM 质量评审才会写进训练集。被拒的尝试连同原因
一起留档，所以产出率是可核查的，而不是嘴上说说。

**{count} 种对话形态**，分为四族：

- **A 族** —— 单轮，每次尝试调用一次（{a_cases}）
- **B 族** —— 单轮，每次尝试调用多次（{b_cases}）
- **C 族** —— 双轮，每轮调用一次（共 {c_count} 种）
- **D 族** —— 双轮，某轮调用多次（共 {d_count} 种）

### 建议的上手顺序

1. **工具库** —— 确认 22 个工具库都在，且 `TOOL_LIST` 与之对应。
2. **标注** —— 先拿一小批（20 条）跑第 2 阶段，把输出读一遍。
3. **生成** —— 选一种形态，先要几条，看看产出长什么样。
4. 形态没问题了再放量。
""",

    # -- tool bank ---------------------------------------------------------- #
    "bank.tab.overview": "工具库总览",
    "bank.tab.toollist": "TOOL_LIST",
    "bank.tab.variants": "第 1 阶段 · 生成变体",
    "bank.note": "每个工具一个 JSONL 文件，每行一个<em>变体</em>。第 3 阶段每条样本随机抽一个变体——"
                 "这正是逼微调后的模型去读工具描述、而不是背工具名字的原因。",
    "bank.rescan": "↻ 重新扫描工具库",
    "bank.report.head": "`{path}` —— 共 {total} 个工具库，其中 {active} 个已提供给第 2 阶段。",
    "bank.report.cols": "| 在 TOOL_LIST 中 | 工具 | 变体数 | 描述 |\n|---|------|--------|------|",
    "bank.report.none": "**未找到任何工具库**，路径：`{path}`。",
    "bank.empty_file": "*（空文件）*",

    "toollist.note": "<strong>TOOL_LIST</strong> 是第 2 阶段可选的工具清单，必须和工具库保持一致："
                     "第 2 阶段选得出、但工具库里没有的工具，会让第 3 阶段直接失败。"
                     "保存会原地改写 <code>toolforge/prompts/tool_selection.py</code>。",
    "toollist.label": "提供给第 2 阶段的工具",
    "toollist.info": "勾选第 2 阶段可以选择的工具，描述自动取自工具库。",
    "toollist.reload": "↻ 重新载入",
    "toollist.selectall": "全选",
    "toollist.clear": "清空",
    "toollist.save": "💾 保存 TOOL_LIST",
    "toollist.unsaved": "_尚未保存。_",
    "toollist.saved": "已把 {count} 个工具写入 {file}。重启应用后第 2 阶段才会生效。",
    "toollist.refuse_empty": "不能写入空的 TOOL_LIST —— 那样第 2 阶段将无工具可选。",
    "toollist.not_in_bank": "工具库中不存在：{names}",
    "toollist.no_definition": "在 {file} 中找不到 TOOL_LIST 定义",

    "variants.note": "只有<strong>语义足够接近</strong>（余弦相似度高于阈值）且<strong>措辞足够不同</strong>"
                     "（BM25 相似度低于阈值）的候选才会被保留。没有本地向量模型时相似度门控会跳过、"
                     "全部保留——在 <code>.env</code> 里设置 <code>EMBEDDING_MODEL_PATH</code> 可启用"
                     "（<code>python download_data.py --with-model</code>）。",
    "variants.tool": "要改写的工具定义",
    "variants.output": "追加写入的工具库文件",
    "variants.output.info": "会先读取文件里已有的变体，所以重复运行是在原基础上继续补足。",
    "variants.model": "生成模型",
    "variants.target": "变体总数",
    "variants.cosine": "语义相似度下限（余弦）",
    "variants.cosine.info": "越高，越要求变体和原工具意思一致。",
    "variants.bm25": "词面相似度上限（BM25）",
    "variants.bm25.info": "越低，越要求变体的措辞与已有变体不同。",
    "variants.run": "▶  运行第 1 阶段",
    "variants.status": "生成结果",
    "variants.result": "### {mark} `{path}` 中已有 {produced}/{wanted} 个变体",
    "variants.shortfall": "没有全部产出——调低余弦阈值或调高 BM25 阈值，再运行一次即可继续补足。",
    "variants.bad_json": "工具定义不是合法的 JSON：{error}",
    "variants.need_fields": "工具定义至少需要 `name` 和 `description` 两个字段。",
    "variants.need_output": "请填写输出文件路径。",

    # -- labeling ----------------------------------------------------------- #
    "label.note": "第 2 阶段读取 HotpotQA / 2WikiMultihopQA 原始数据，为每条补上三个字段："
                  "<strong>reasoning</strong>（轨迹指导）、<strong>tool_select</strong>（该用哪个工具库的工具）"
                  "和 <strong>route_select</strong>（case1–case4，决定第 3 阶段生成什么形态的对话）。",
    "label.input": "输入 JSONL",
    "label.input.info": "原始多跳问答数据。还没下载的话先执行 `python download_data.py`。",
    "label.output": "输出 JSONL",
    "label.residue": "剩余数据 JSONL（可选）",
    "label.residue.info": "超出处理条数的记录会原样存到这里，方便把大语料分批做完。",
    "label.preview": "_填入输入路径后这里会显示文件概况。_",
    "label.model": "标注模型",
    "label.limit": "本次标注条数",
    "label.limit.info": "先小批量试（比如 20 条），确认输出没问题再放量。",
    "label.concurrency": "并发数",
    "label.concurrency.info": "同时在途的请求数。触发限流就调小。",
    "label.single": "强制走 case1",
    "label.single.info": "改用更严格的提示词，总是输出 case1——用于补足数量偏少的单次调用类别。",
    "label.run": "▶  运行第 2 阶段",
    "label.status": "标注结果",
    "label.inspect": "查看文件",
    "label.inspect.label": "要查看的文件",
    "label.result": """### ✅ 标注完成

| | |
|---|---|
| 读取记录数 | {total} |
| 标注成功 | **{labelled}** |
| 标注失败 | {failed} |
| 顺延到剩余文件 | {deferred} |
| 成功率 | {rate}% |

接下来把输出文件送进 **生成** 标签页。""",
    "label.no_input": "输入文件不存在：`{path}`",
    "label.no_output": "请填写输出路径。",

    # -- generate ----------------------------------------------------------- #
    "gen.note": "每次尝试会先规划工具调用轨迹、检索出真实的干扰段落、生成完整对话，"
                "然后跑 <strong>9 项规则校验</strong>和一次 <strong>LLM 评审</strong>。"
                "只有满分 2/2 才会进训练集；其余全部连同原因写入评分文件，产出率完全可查。",
    "gen.input": "第 2 阶段产出的标注 JSONL",
    "gen.input.info": "即「标注」标签页的输出文件。",
    "gen.outdir": "输出目录",
    "gen.cases": "要生成的对话形态",
    "gen.cases.info": "可以多选，会依次执行。",
    "gen.target": "每种形态生成条数",
    "gen.target.info": "指保留下来的条数，不是尝试次数——通常要试好几次才成一条。",
    "gen.advanced": "高级：逐形态配置（会覆盖上面的选择）",
    "gen.advanced.note": '留空则使用上面的选项。想按形态分别指定输出路径，可以粘贴例如：'
                         '<code>{{"case_C1": {{"target_count": 100, "data_output": "out/c1.jsonl", '
                         '"score_output": "out/c1_scores.jsonl"}}}}</code>',
    "gen.model": "生成模型",
    "gen.model.info": "负责撰写对话，质量影响最大。",
    "gen.judge": "评审模型",
    "gen.judge.info": "评估思考与行动是否一致，建议用你手上最强的模型。",
    "gen.concurrency": "并发数",
    "gen.concurrency.info": "每种形态同时进行的尝试数。",
    "gen.delay": "每次尝试间隔（秒）",
    "gen.delay.info": "给有限流的服务端留点余地。",
    "gen.vmin": "干扰工具数量 —— 最少",
    "gen.vmax": "干扰工具数量 —— 最多",
    "gen.vmax.info": "提示词里除正确工具外，还放多少个干扰工具。",
    "gen.strict": "严格程度",
    "gen.strict.refs": "启用校验 7（引用一致性）",
    "gen.strict.refs.info": "该校验在原始版本中因下标写错而从未生效。启用后会真正执行，产出率会下降。",
    "gen.strict.answer": "强制校验最后一轮 <answer> 格式",
    "gen.strict.answer.info": "原始版本遇到最后一轮格式错误只警告、不拒绝。",
    "gen.run": "▶  运行第 3+4 阶段",
    "gen.status": "运行报告",
    "gen.inspect": "查看生成数据或评分文件",
    "gen.inspect.label": "要查看的文件",
    "gen.cases_accordion": "29 种对话形态分别是什么？",
    "gen.no_input": "标注文件不存在：`{path}`。请先运行第 2 阶段。",
    "gen.bad_json": "逐形态配置不是合法的 JSON：{error}",
    "gen.unknown_case": "未知的形态 ID：{names}",
    "gen.pick_case": "请至少选择一种形态。",
    "gen.vrange": "干扰工具的最小值不能大于最大值。",
    "gen.no_records": "{path} 中没有带 'tool_select' 字段的记录 —— 第 2 阶段跑完了吗？",
    "gen.loaded": "已载入 {count} 条标注记录",

    # -- data --------------------------------------------------------------- #
    "data.note": "每个阶段都会写出 JSONL。这里可以打开其中任意一个：原始输入、标注结果、"
                 "生成的对话，或是记录了每次尝试被拒原因的评分文件。",
    "data.tab.browse": "逐条浏览",
    "data.tab.revalidate": "重新校验数据文件",
    "data.browse.label": "要浏览的文件",
    "data.revalidate.note": "对生成数据文件重新执行 9 项规则校验——改过提示词之后，"
                            "或者想知道打开更严格的开关会损失多少产出时很有用。",
    "data.revalidate.path": "生成数据 JSONL",
    "data.revalidate.run": "▶  重新校验",
    "data.revalidate.idle": "_尚未执行校验。_",
    "data.revalidate.notfound": "### ⚠️ 文件不存在：`{path}`",
    "data.revalidate.empty": "### ⚠️ 文件为空。",
    "data.revalidate.head": "### {mark} {passed}/{total} 条记录通过全部 9 项校验",
    "data.revalidate.cases": "| 形态 | 记录数 |\n|------|--------|",
    "data.revalidate.fails": "| 失败数 | 校验项 |\n|---|-------|",
    "data.revalidate.notrecord": "不是 ToolForge 生成的数据行",

    # -- runtime ------------------------------------------------------------ #
    "run.working": "⏳ 正在运行…",
    "run.done": "### ✅ 完成",
    "run.failed": "### ❌ 运行失败\n\n**{kind}：** {error}\n\n<details><summary>调用栈</summary>\n\n```\n{traceback}\n```\n\n</details>",
    "run.guard": "### ⚠️ {message}",

    # -- run report --------------------------------------------------------- #
    "report.none": "本次没有运行任何形态。",
    "report.head": "### 运行完成",
    "report.summary": "共尝试 **{attempts}** 次，保留 **{kept}** 条 —— 总产出率 **{rate}%**",
    "report.cols": "| 形态 | 保留 | 目标 | 尝试 | 产出率 |\n|------|------|------|------|--------|",
    "report.reasons": "**最常见的被拒原因**",
}

# --------------------------------------------------------------------------- #
# English
# --------------------------------------------------------------------------- #

EN: dict[str, str] = {
    "app.subtitle": "An automated SFT data factory for LLM tool-calling · stages 1 – 4",
    "tab.overview": "Overview",
    "tab.toolbank": "Tool bank",
    "tab.label": "Label · stage 2",
    "tab.generate": "Generate · stages 3 + 4",
    "tab.data": "Data",

    "chip.gen": "gen",
    "chip.judge": "judge",
    "chip.keys": "{count} API keys",
    "chip.libraries": "{count} tool libraries",

    "rail.1.stage": "Stage 1",
    "rail.1.title": "Tool bank",
    "rail.1.desc": "Paraphrase tools into variants so the model reads descriptions, not names.",
    "rail.2.stage": "Stage 2",
    "rail.2.title": "Labelling",
    "rail.2.desc": "Tag each question with the tool to call and the routing class.",
    "rail.3.stage": "Stage 3",
    "rail.3.title": "Dialogue",
    "rail.3.desc": "Author a multi-turn conversation for one of 29 case shapes.",
    "rail.4.stage": "Stage 4",
    "rail.4.title": "Validation",
    "rail.4.desc": "Nine rule checks plus an LLM judge; only 2/2 is kept.",

    "model.info": "Pick a preset or type any model id your endpoint serves.",
    "temp.label": "Temperature",
    "temp.info": "0 for reproducible pipeline runs; raise it only for stage 1 variety.",
    "tokens.label": "Max tokens",
    "tokens.info": "Long dialogues need headroom — 8192 is a safe floor.",
    "status.waiting": "_{what} appears here once you start._",
    "status.result": "Result",
    "log.label": "Live log",
    "log.placeholder": "Progress streams here while the job runs…",

    "inspect.path": "JSONL file",
    "inspect.path.info": "Any JSONL — input, labelled output, generated data or scores.",
    "inspect.load": "↻ Load",
    "inspect.none": "_No file loaded._",
    "inspect.notfound": "_File not found._",
    "inspect.empty": "// file is empty",
    "inspect.nothing": "// nothing loaded",
    "inspect.record": "Record",
    "inspect.content": "Record contents",
    "inspect.oob": "// index out of range (1-{total})",
    "inspect.missing": "// file not found: {path}",
    "inspect.summary": "**{name}** · {count} records · {size} KB · modified {modified}",

    "overview.readiness": "### Readiness",
    "overview.recheck": "↻ Re-check configuration",
    "overview.check.gen": "Generation model",
    "overview.check.judge": "Judge model",
    "overview.check.sdk": "Anthropic SDK",
    "overview.check.bank": "Tool bank",
    "overview.check.bm25": "BM25 retrieval",
    "overview.check.env": ".env file",
    "overview.check.env.ok": "`{name}` loaded",
    "overview.check.env.missing": "copy `.env.example` to `.env` and fill it in",
    "overview.check.env.unused": "not present — the environment supplies everything",
    "overview.check.installed": "installed",
    "overview.check.needkey": "`{model}` via **{provider}** — set `{variable}`",
    "overview.check.model": "`{model}` via **{provider}**",
    "overview.check.sdk.missing": "`pip install 'toolforge[anthropic]'`",
    "overview.check.bank.detail": "{count} libraries at `{path}`",
    "overview.check.bm25.missing": "`pip install bm25s jieba`",
    "overview.header": "| | check | detail |\n|---|-------|--------|",
    "overview.ready": "\n\n**Ready to run.**",
    "overview.notready": "\n\n**Not ready yet** — fix: {blocked}.",
    "overview.config.accordion": "Resolved configuration",
    "overview.config.label": "Effective settings",
    "overview.panel": """
<div class="tf-panel">
<strong>Configuration</strong> comes from <code>.env</code> at the repo root.<br><br>
Command line equivalent of everything here:<br>
<code>toolforge doctor</code> · <code>toolforge label</code> · <code>toolforge generate</code><br><br>
<span style="color:var(--tf-muted)">ToolForge v{version}</span>
</div>
""",
    "overview.body": """
### What this builds

ToolForge turns raw multi-hop QA into **supervised fine-tuning data that teaches a model
to call tools** — including the parts that are hard to collect from the wild: recovering
from a bad tool choice, retrying with corrected arguments, and falling back to general
search when a specialised tool keeps coming up empty.

Every sample is a full conversation in `<think>` / `<tool_call>` / `<answer>` form, and
every sample is *earned*: it has to survive nine structural checks and an LLM judge before
it is written to the training set. Rejected attempts are kept with their reasons, so the
yield numbers are reproducible rather than asserted.

**{count} dialogue cases** across four families:

- **A** — one turn, one call per attempt ({a_cases})
- **B** — one turn, several calls per attempt ({b_cases})
- **C** — two turns, one call each ({c_count} cases)
- **D** — two turns, several calls per turn ({d_count} cases)

### Where to start

1. **Tool bank** — check the 22 libraries are present and `TOOL_LIST` matches them.
2. **Label** — run stage 2 over a small slice (20 records) and read the output.
3. **Generate** — pick one case, ask for a handful of samples, inspect what comes out.
4. Scale up once the shape looks right.
""",

    "bank.tab.overview": "Bank overview",
    "bank.tab.toollist": "TOOL_LIST",
    "bank.tab.variants": "Stage 1 · generate variants",
    "bank.note": "One JSONL file per tool, one <em>variant</em> per line. Stage 3 draws a random variant "
                 "for every sample, which is what forces a fine-tuned model to read tool descriptions "
                 "instead of memorising tool names.",
    "bank.rescan": "↻ Rescan the bank",
    "bank.report.head": "`{path}` — {total} libraries, {active} of them offered to stage 2.",
    "bank.report.cols": "| in TOOL_LIST | tool | variants | description |\n|---|------|----------|-------------|",
    "bank.report.none": "**No tool libraries found** at `{path}`.",
    "bank.empty_file": "*(empty file)*",

    "toollist.note": "<strong>TOOL_LIST</strong> is the menu stage 2 chooses from. It must stay in step with "
                     "the bank: a tool stage 2 can pick but the bank cannot supply will break stage 3. "
                     "Saving rewrites <code>toolforge/prompts/tool_selection.py</code> in place.",
    "toollist.label": "Tools offered to stage 2",
    "toollist.info": "Tick the tools stage 2 may select. Descriptions are taken from the bank.",
    "toollist.reload": "↻ Reload",
    "toollist.selectall": "Select all",
    "toollist.clear": "Clear",
    "toollist.save": "💾 Save TOOL_LIST",
    "toollist.unsaved": "_Not saved yet._",
    "toollist.saved": "Saved {count} tools to {file}. Restart the app for stage 2 to pick it up.",
    "toollist.refuse_empty": "Refusing to write an empty TOOL_LIST — stage 2 would have nothing to choose from.",
    "toollist.not_in_bank": "Not in the tool bank: {names}",
    "toollist.no_definition": "No TOOL_LIST definition found in {file}",

    "variants.note": "A candidate is kept only when it is <strong>close in meaning</strong> (cosine above the "
                     "threshold) and <strong>far in wording</strong> (BM25 below it). Without a local embedding "
                     "model the gate is skipped and every candidate is kept — set "
                     "<code>EMBEDDING_MODEL_PATH</code> in <code>.env</code> to enable it "
                     "(<code>python download_data.py --with-model</code>).",
    "variants.tool": "Tool definition to paraphrase",
    "variants.output": "Tool library to append to",
    "variants.output.info": "Existing variants in the file are loaded first, so re-running tops it up.",
    "variants.model": "Generation model",
    "variants.target": "Variants in total",
    "variants.cosine": "Minimum semantic similarity (cosine)",
    "variants.cosine.info": "Higher = variants must mean the same thing.",
    "variants.bm25": "Maximum lexical similarity (BM25)",
    "variants.bm25.info": "Lower = variants must be worded differently.",
    "variants.run": "▶  Run stage 1",
    "variants.status": "Generation summary",
    "variants.result": "### {mark} {produced}/{wanted} variants in `{path}`",
    "variants.shortfall": "Not all were produced — lower the cosine threshold or raise the BM25 one, "
                          "then run again to top the file up.",
    "variants.bad_json": "The tool definition is not valid JSON: {error}",
    "variants.need_fields": "The tool definition needs at least a `name` and a `description`.",
    "variants.need_output": "Choose an output file.",

    "label.note": "Stage 2 reads raw HotpotQA / 2WikiMultihopQA records and adds three fields: "
                  "<strong>reasoning</strong> (the trajectory guidance), <strong>tool_select</strong> "
                  "(which tool library the gold tool comes from) and <strong>route_select</strong> "
                  "(case1–case4, which decides the shape of the stage 3 dialogue).",
    "label.input": "Input JSONL",
    "label.input.info": "Raw multi-hop QA. Run `python download_data.py` if you have not yet.",
    "label.output": "Output JSONL",
    "label.residue": "Residue JSONL (optional)",
    "label.residue.info": "Records beyond the limit are parked here so you can work through a corpus in batches.",
    "label.preview": "_Enter an input path to see what is there._",
    "label.model": "Labelling model",
    "label.limit": "Records to label",
    "label.limit.info": "Start small — 20 or so — to sanity-check the output before a full run.",
    "label.concurrency": "Concurrency",
    "label.concurrency.info": "Parallel in-flight requests. Lower it if you hit rate limits.",
    "label.single": "Force route case1",
    "label.single.info": "Uses the stricter prompt that always answers case1 — for topping up the "
                         "under-represented single-call class.",
    "label.run": "▶  Run stage 2",
    "label.status": "Labelling summary",
    "label.inspect": "Inspect a file",
    "label.inspect.label": "File to inspect",
    "label.result": """### ✅ Labelling complete

| | |
|---|---|
| records read | {total} |
| labelled | **{labelled}** |
| failed | {failed} |
| deferred to residue | {deferred} |
| success rate | {rate}% |

Feed the output file into the **Generate** tab next.""",
    "label.no_input": "Input file not found: `{path}`",
    "label.no_output": "Choose an output path.",

    "gen.note": "Each attempt plans a tool-calling trajectory, retrieves realistic passages, authors the "
                "conversation, then runs <strong>nine rule checks</strong> and an <strong>LLM judge</strong>. "
                "Only a perfect 2/2 reaches the training file — everything else is recorded in the score "
                "file with the reason, so the yield is fully auditable.",
    "gen.input": "Labelled JSONL from stage 2",
    "gen.input.info": "The output of the Label tab.",
    "gen.outdir": "Output directory",
    "gen.cases": "Cases to generate",
    "gen.cases.info": "Pick as many as you like — they run one after another.",
    "gen.target": "Samples per case",
    "gen.target.info": "Kept samples, not attempts. Expect several attempts per keeper.",
    "gen.advanced": "Advanced: per-case configuration (overrides the pickers above)",
    "gen.advanced.note": 'Leave the box empty to use the pickers. To drive output paths per case, paste e.g. '
                         '<code>{{"case_C1": {{"target_count": 100, "data_output": "out/c1.jsonl", '
                         '"score_output": "out/c1_scores.jsonl"}}}}</code>',
    "gen.model": "Generation model",
    "gen.model.info": "Writes the dialogues. Quality matters most here.",
    "gen.judge": "Judge model",
    "gen.judge.info": "Scores think/action consistency. Use your strongest model.",
    "gen.concurrency": "Concurrency",
    "gen.concurrency.info": "Parallel attempts per case.",
    "gen.delay": "Delay between attempts (s)",
    "gen.delay.info": "Gentle pacing for rate-limited endpoints.",
    "gen.vmin": "Distractor tools — minimum",
    "gen.vmax": "Distractor tools — maximum",
    "gen.vmax.info": "How many wrong tools sit alongside the gold tool in the prompt.",
    "gen.strict": "Strictness",
    "gen.strict.refs": "Enable check 7 (reference consistency)",
    "gen.strict.refs.info": "This check read the wrong record slots in the original release and always "
                            "passed. Turning it on enforces it — expect a lower yield.",
    "gen.strict.answer": "Enforce the final <answer> format",
    "gen.strict.answer.info": "The original only warned when the last turn was malformed.",
    "gen.run": "▶  Run stages 3 + 4",
    "gen.status": "Run report",
    "gen.inspect": "Inspect generated data or scores",
    "gen.inspect.label": "File to inspect",
    "gen.cases_accordion": "What are the 29 cases?",
    "gen.no_input": "Labelled file not found: `{path}`. Run stage 2 first.",
    "gen.bad_json": "The per-case configuration is not valid JSON: {error}",
    "gen.unknown_case": "Unknown case id(s): {names}",
    "gen.pick_case": "Pick at least one case.",
    "gen.vrange": "The distractor minimum cannot exceed the maximum.",
    "gen.no_records": "{path} has no records with a 'tool_select' field — did stage 2 finish?",
    "gen.loaded": "{count} labelled records loaded",

    "data.note": "Every stage writes JSONL. Point the browser at any of them: raw input, labelled records, "
                 "generated dialogues, or the score file that explains why an attempt was rejected.",
    "data.tab.browse": "Browse records",
    "data.tab.revalidate": "Re-validate a data file",
    "data.browse.label": "File to browse",
    "data.revalidate.note": "Runs the nine rule checks again over a generated-data file — useful after changing "
                            "prompts, or to see how much a stricter setting would cost you.",
    "data.revalidate.path": "Generated-data JSONL",
    "data.revalidate.run": "▶  Re-validate",
    "data.revalidate.idle": "_Nothing checked yet._",
    "data.revalidate.notfound": "### ⚠️ File not found: `{path}`",
    "data.revalidate.empty": "### ⚠️ The file is empty.",
    "data.revalidate.head": "### {mark} {passed}/{total} records pass all nine checks",
    "data.revalidate.cases": "| case | records |\n|------|---------|",
    "data.revalidate.fails": "| failures | check |\n|---|-------|",
    "data.revalidate.notrecord": "record is not a ToolForge generated-data row",

    "run.working": "⏳ Working…",
    "run.done": "### ✅ Done",
    "run.failed": "### ❌ Failed\n\n**{kind}:** {error}\n\n<details><summary>Traceback</summary>\n\n```\n{traceback}\n```\n\n</details>",
    "run.guard": "### ⚠️ {message}",

    "report.none": "No cases were run.",
    "report.head": "### Run complete",
    "report.summary": "**{kept}** samples kept from **{attempts}** attempts — overall yield **{rate}%**",
    "report.cols": "| case | kept | target | attempts | yield |\n|------|------|--------|----------|-------|",
    "report.reasons": "**Most common rejection reasons**",
}

STRINGS: dict[str, dict[str, str]] = {"zh": ZH, "en": EN}


# --------------------------------------------------------------------------- #
# Translating engine output for display
# --------------------------------------------------------------------------- #

#: The nine rule-check labels, exactly as written to the score files.
CHECK_LABELS_ZH: dict[str, str] = {
    "1. Dialogue format validation failed": "1. 对话格式校验失败",
    "2. Assistant content format validation failed": "2. assistant 内容格式校验失败",
    "3. Non-assistant field empty validation failed": "3. 非 assistant 消息内容为空",
    "4. Answer consistency check failed": "4. 最终答案与标准答案不一致",
    "5. Tool-RAG consistency check failed": "5. tool 消息与检索段落不一致",
    "6. Argument validation failed": "6. 重试时修改了非必填参数",
    "7. Reference error at one or more stages": "7. 引用的支撑段落与原始数据不符",
    "8. Predefined tool count mismatch or inconsistent usage order": "8. 调用的工具与标注不符",
    "9. Mismatch between tool_call names/arguments and tool_bank definitions": "9. 工具名或参数不在提供的工具列表中",
}

#: Progress lines emitted by the pipeline, as ``(pattern, replacement)``.
_LOG_PATTERNS_ZH: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"^▶ (\S+): target (\d+), (\d+) labelled records, (\d+) workers$",
         r"▶ \1：目标 \2 条，可用标注记录 \3 条，并发 \4"),
        (r"^  ✓ (\S+) (\d+)/(\d+)$", r"  ✓ \1 已保留 \2/\3"),
        (r"^  ✗ attempt (\d+): (.*)$", r"  ✗ 第 \1 次尝试：\2"),
        (r"^  ✗ unreadable record: (.*)$", r"  ✗ 记录无法解析：\1"),
        (r"^  · (\S+) quota already met, sample scored but not stored$",
         r"  · \1 已达目标数量，本条已评分但不再保存"),
        (r"^⚠️  (\S+): no records carry a 'tool_select' label.*$",
         r"⚠️  \1：没有任何记录带 tool_select 标注 —— 第 2 阶段跑了吗？"),
        (r"^(✅|⚠️) (\S+): (\d+)/(\d+) kept from (\d+) attempts \(([\d.]+)%\)$",
         r"\1 \2：\5 次尝试保留 \3/\4（产出率 \6%）"),
        (r"^read (\d+) records — labelling (\d+), deferring (\d+)$",
         r"读取 \1 条记录 —— 本次标注 \2 条，顺延 \3 条"),
        (r"^  labelled (\d+)/(\d+)$", r"  已标注 \1/\2"),
        (r"^deferred records written to (.*)$", r"顺延的记录已写入 \1"),
        (r"^wrote (\S+) — (.*)$", r"已写入 \1 —— \2"),
        (r"^nothing to label$", "没有需要标注的记录"),
        (r"^(\d+) labelled · (\d+) failed · (\d+) left for later · (\d+) read$",
         r"标注成功 \1 · 失败 \2 · 顺延 \3 · 共读取 \4"),
        (r"^(\d+) variant\(s\) already in (\S+); target is (\d+)$",
         r"\2 中已有 \1 个变体，目标 \3 个"),
        (r"^  ✓ (\d+)/(\d+) — (.*)$", r"  ✓ \1/\2 —— \3"),
        (r"^  ✗ attempt (\d+): the model returned no usable tool$",
         r"  ✗ 第 \1 次尝试：模型没有返回可用的工具定义"),
        (r"^⚠️  no embedding model.*$",
         "⚠️  未配置向量模型 —— 相似度门控已关闭，所有候选都会被保留"),
        (r"^✅ (\S+) now holds (\d+) variants$", r"✅ \1 现在共有 \2 个变体"),
        (r"^⚠️  stopped after (\d+) attempts with (\d+)/(\d+) variants.*$",
         r"⚠️  尝试 \1 次后停止，只得到 \2/\3 个变体 —— 试试放宽阈值"),
    )
)


def translate_log(line: str) -> str:
    """Render one pipeline progress line in the active language."""
    if _current != "zh":
        return line
    for english, chinese in CHECK_LABELS_ZH.items():
        line = line.replace(english, chinese)
    for pattern, replacement in _LOG_PATTERNS_ZH:
        translated, count = pattern.subn(replacement, line)
        if count:
            return translated
    return line


def translate_report(markdown: str) -> str:
    """Render a finished run report in the active language."""
    if _current != "zh":
        return markdown
    replacements = {
        "### Run complete": t("report.head"),
        "samples kept from": "条保留，共尝试",
        "attempts — overall yield": "次 —— 总产出率",
        "| case | kept | target | attempts | yield |": "| 形态 | 保留 | 目标 | 尝试 | 产出率 |",
        "|------|------|--------|----------|-------|": "|------|------|------|------|--------|",
        "**Most common rejection reasons**": t("report.reasons"),
        "No cases were run.": t("report.none"),
    }
    for english, chinese in replacements.items():
        markdown = markdown.replace(english, chinese)
    for english, chinese in CHECK_LABELS_ZH.items():
        markdown = markdown.replace(english, chinese)
    return markdown
