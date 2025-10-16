# ToolForge

[English](README.md) | 简体中文

一个全面的多阶段流水线系统，用于生成、标注和验证多跳推理对话AI训练数据。

## 📋 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [配置说明](#配置说明)


## 🎯 项目概述

ToolForge 是一个精心设计的流水线系统，用于自动化创建高质量的多跳推理任务训练数据。系统通过四个不同的阶段处理原始问答数据：

1. **工具的构建与多样化（Stage 1）**：利用一个基础工具大规模生成虚拟的多样化工具
2. **工具调用范式的选择（Stage 2）**：为每个query提供Optimal tools、Optimal Tool-calling paradigm、Optimal reasoning rationale
3. **数据生成 + 验证评分（Stage 3 & Stage 4）**：生成具有反思和多跳推理能力的多轮工具调用对话数据，并使用基于规则和基于LLM的方法验证生成数据的质量

可以通过启动webui页面，轻松实现上述三个步骤的功能。


**核心特性**：
- 🔧 四阶段自动化处理流水线
- 🛠️ 可扩展的工具库管理系统
- 🎯 双重验证机制（规则+LLM）
- 🖥️ 交互式Gradio Web界面
- 🚀 并发处理与API密钥轮换

## 🏗 系统架构

```
ToolForge/
├── Stage_1/                  # 工具的构建与多样化
│   ├── generate_tool.py
│   └── tool_prompts.py
├── Stage_2/                  # 工具调用范式的选择
│   └── code/
│       ├── llm_generate_label.py
│       └── tool_prompts.py
├── Stage_3/                  # 数据生成
│   ├── config/              # 配置文件
│   ├── core/                # API客户端
│   ├── services/            # 对话生成、工具管理
│   ├── tool_bank/tools/     # 工具定义（JSONL）
│   └── prompts/             # 提示词模板
├── Stage_4/                  # 质量验证
│   ├── config/              # 配置文件    
│   ├── core/                # LLM评估客户端
│   ├── prompts/             # 评估提示词
│   ├── utils/               # 辅助函数
│   └── validators/          # 验证引擎
└── gradio_webui/            # Web界面
    ├── quick_fast.py        # 主应用
    ├── feature_tool_variant_generator.py
    ├── feature_generate_judge.py
    └── feature_tool_list_manager.py
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ToolForge/gradio_webui
pip install -r requirements.txt
```

### 2. 配置API密钥

**方式A - 环境变量（推荐）**：
```bash
export API_KEYS="key1,key2,key3"
export API_BASE_URL="https://api.openai.com/v1"
```

**方式B - 配置文件**：
编辑以下文件中的 `api_keys` 列表：
- `Stage_2/code/llm_generate_label.py`
- `Stage_3/config/api_keys.py`
- `Stage_4/config/api_keys.py`

### 3. 启动Web界面

```bash
python quick_fast.py
# 访问 http://localhost:7860
```

## 🎯 核心功能

### 功能一：工具构建与多样化

基于原始工具生成语义相似的变体，扩充工具库。

#### 1.1 工具变体生成

**输入**：单个工具的JSON定义
```json
{
  "name": "example_search",
  "description": "示例搜索工具",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "搜索内容"}
    },
    "required": ["query"]
  }
}
```

**配置参数**：
- 目标生成数量：10-50（推荐）
- 余弦相似度阈值：0.7（控制语义相似度）
- BM25相似度阈值：0.6（避免过度相似）
- 模型：gpt-4.1 / claude系列
- Temperature：1.0

**输出**：JSONL文件，每行一个工具变体

#### 1.2 工具列表管理

管理 `Stage_2/code/tool_prompts.py` 中的 TOOL_LIST 配置。

**操作流程**：
1. 点击"刷新工具列表"扫描工具库
2. 勾选工具 → 点击"添加到TOOL_LIST"
3. 移除不需要的工具
4. 点击"保存"更新配置文件

#### 1.3 文件查看器

查看生成的工具变体，支持逐行浏览和JSON格式化。

---

### 功能二：工具调用范式的选择

分析多跳问题，自动标注合适的工具和执行路径。

#### 2.1 标注流程

**输入数据格式**（JSONL）：
```json
{
  "question": "谁是《哈利·波特》的作者？她的第一本书是什么时候出版的？",
  "answer": "J.K.罗琳，1997年",
  "type": "bridge",
  "supporting_facts": [...],
  "context": [...]
}
```

**输出格式**（添加三个字段）：
```json
{
  "question": "...",
  "answer": "...",
  "reasoning": "这是一个两跳问题：1) 找到作者 2) 查询出版时间...",
  "tool_select": "person_information_search, creation_information_search",
  "route_select": "case_C2"
}
```

**路径类型**：
- `SRST`：Single-Round Single-Tool (对应case_A)
- `SRMT`：Single-Round Multi-Tool (对应case_B)
- `MRST`：Multi-Round Single-Tool (对应case_D)
- `MRMT`：Multi-Round Multi-Tool (对应case_C)

#### 2.2 实时监控

**LLM实时输出**窗口显示：
- 当前处理的问题
- LLM的推理过程
- 工具选择和路径规划依据

#### 2.3 文件查看器

查看输入文件、输出文件或自定义JSONL文件。

---

### 功能三：多轮工具调用对话数据的生成与检验

整合Stage 3和Stage 4，生成并验证高质量多轮对话数据。



### 工具定义格式

工具定义位于 `Stage_3/tool_bank/tools/`，采用JSONL格式：

```json
{
  "name": "example_search",
  "description": "用于查找特定信息的搜索工具",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "搜索查询内容"},
      "category": {"type": "string", "description": "搜索类别（可选）"}
    },
    "required": ["query"]
  }
}
```

### 添加自定义工具

1. 在 `Stage_3/tool_bank/tools/` 创建JSONL文件
2. 按上述格式定义工具
3. 使用Web界面的工具列表管理器添加到TOOL_LIST



### 参数推荐

| 场景 | MAX_LINES | CONCURRENCY | API密钥数 |
|------|-----------|-------------|----------|
| 开发测试 | 10 | 2 | 1 |
| 小批量 | 1000 | 5-10 | 2-3 |
| 大批量 | 10000+ | 15-20 | 5+ |



## 📄 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件

## 📞 支持

- 📖 查看 [完整文档](README.md)
- 🐛 提交 [Issue](https://github.com/yourusername/ToolForge/issues)
- 💬 参与 [Discussions](https://github.com/yourusername/ToolForge/discussions)

---

**版本**: 1.0.0  
**最后更新**: 2025-01-15  
**维护者**: ToolForge Team

