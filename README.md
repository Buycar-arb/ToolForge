# ToolForge

English | [简体中文](README-zh.md)

A comprehensive multi-stage pipeline system for generating, labeling, and validating multi-hop reasoning conversational AI training data.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [Configuration](#configuration)

## 🎯 Project Overview

ToolForge is a carefully designed pipeline system for automating the creation of high-quality multi-hop reasoning task training data. The system processes raw question-answer data through four distinct stages:

1. **Tool Construction and Diversification (Stage 1)**: Leverages a base tool to generate diverse virtual tools at scale
2. **Selection of Tool-Calling Paradigm (Stage 2)**: Provides optimal tools, optimal tool-calling paradigm, and optimal reasoning rationale for each query
3. **Data Generation + Validation & Scoring (Stage 3 & Stage 4)**: Generates multi-turn tool-calling conversation data with reflection and multi-hop reasoning capabilities, and validates the quality of generated data using rule-based and LLM-based methods

All three stages can be easily implemented through the WebUI interface.

**Core Features**:
- 🔧 Four-stage automated processing pipeline
- 🛠️ Extensible tool library management system
- 🎯 Dual validation mechanism (rules + LLM)
- 🖥️ Interactive Gradio Web UI
- 🚀 Concurrent processing with API key rotation

## 🏗 System Architecture

```
ToolForge/
├── Stage_1/                  # Tool Construction and Diversification
│   ├── generate_tool.py
│   └── tool_prompts.py
├── Stage_2/                  # Selection of Tool-Calling Paradigm
│   └── code/
│       ├── llm_generate_label.py
│       └── tool_prompts.py
├── Stage_3/                  # Data Generation
│   ├── config/              # Configuration files
│   ├── core/                # API client
│   ├── services/            # Conversation generation, tool management
│   ├── tool_bank/tools/     # Tool definitions (JSONL)
│   └── prompts/             # Prompt templates
├── Stage_4/                  # Quality Validation
│   ├── config/              # Configuration files    
│   ├── core/                # LLM evaluation client
│   ├── prompts/             # Evaluation prompts
│   ├── utils/               # Helper functions
│   └── validators/          # Validation engine
└── gradio_webui/            # Web Interface
    ├── quick_fast.py        # Main application
    ├── feature_tool_variant_generator.py
    ├── feature_generate_judge.py
    └── feature_tool_list_manager.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ToolForge/gradio_webui
pip install -r requirements.txt
```

### 2. Configure API Keys

**Option A - Environment Variables (Recommended)**:
```bash
export API_KEYS="key1,key2,key3"
export API_BASE_URL="https://api.openai.com/v1"
```

**Option B - Configuration Files**:
Edit the `api_keys` list in the following files:
- `Stage_2/code/llm_generate_label.py`
- `Stage_3/config/api_keys.py`
- `Stage_4/config/api_keys.py`

### 3. Launch Web Interface

```bash
python quick_fast.py
# Visit http://localhost:7860
```

## 🎯 Core Features

### Feature 1: Tool Construction and Diversification

Generate semantic variants of original tools to expand the tool library.

#### 1.1 Tool Variant Generation

**Input**: JSON definition of a single tool
```json
{
  "name": "example_search",
  "description": "Example search tool",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search content"}
    },
    "required": ["query"]
  }
}
```

**Configuration Parameters**:
- Target generation count: 10-50 (recommended)
- Cosine similarity threshold: 0.7 (controls semantic similarity)
- BM25 similarity threshold: 0.6 (avoids excessive similarity)
- Model: gpt-4.1 / Claude series
- Temperature: 1.0

**Output**: JSONL file with one tool variant per line

#### 1.2 Tool List Management

Manage the TOOL_LIST configuration in `Stage_2/code/tool_prompts.py`.

**Operation Flow**:
1. Click "Refresh Tool List" to scan the tool library
2. Select tools → Click "Add to TOOL_LIST"
3. Remove unwanted tools
4. Click "Save" to update configuration file

#### 1.3 File Viewer

View generated tool variants with line-by-line browsing and JSON formatting support.

---

### Feature 2: Selection of Tool-Calling Paradigm

Analyze multi-hop questions and automatically label appropriate tools and execution paths.

#### 2.1 Labeling Process

**Input Data Format** (JSONL):
```json
{
  "question": "Who is the author of Harry Potter? When was her first book published?",
  "answer": "J.K. Rowling, 1997",
  "type": "bridge",
  "supporting_facts": [...],
  "context": [...]
}
```

**Output Format** (with three additional fields):
```json
{
  "question": "...",
  "answer": "...",
  "reasoning": "This is a two-hop question: 1) Find the author 2) Query publication date...",
  "tool_select": "person_information_search, creation_information_search",
  "route_select": "case_C2"
}
```

**Route Types**:
- `SRST`: Single-Round Single-Tool (corresponds to case_A)
- `SRMT`: Single-Round Multi-Tool (corresponds to case_B)
- `MRST`: Multi-Round Single-Tool (corresponds to case_D)
- `MRMT`: Multi-Round Multi-Tool (corresponds to case_C)

#### 2.2 Real-time Monitoring

**LLM Real-time Output** window displays:
- Current question being processed
- LLM's reasoning process
- Tool selection and path planning rationale

#### 2.3 File Viewer

View input files, output files, or custom JSONL files.

---

### Feature 3: Multi-turn Tool-Calling Conversation Data Generation and Validation

Integrates Stage 3 and Stage 4 to generate and validate high-quality multi-turn conversation data.

### Tool Definition Format

Tool definitions are located in `Stage_3/tool_bank/tools/` in JSONL format:

```json
{
  "name": "example_search",
  "description": "Search tool for finding specific information",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query content"},
      "category": {"type": "string", "description": "Search category (optional)"}
    },
    "required": ["query"]
  }
}
```

### Adding Custom Tools

1. Create a JSONL file in `Stage_3/tool_bank/tools/`
2. Define tools following the format above
3. Use the Tool List Manager in the Web UI to add to TOOL_LIST

### Parameter Recommendations

| Scenario | MAX_LINES | CONCURRENCY | API Keys |
|----------|-----------|-------------|----------|
| Development Testing | 10 | 2 | 1 |
| Small Batch | 1000 | 5-10 | 2-3 |
| Large Batch | 10000+ | 15-20 | 5+ |

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 📞 Support

- 📖 Check [Complete Documentation](README.md)
- 🐛 Submit [Issues](https://github.com/yourusername/ToolForge/issues)
- 💬 Join [Discussions](https://github.com/yourusername/ToolForge/discussions)

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-15  
**Maintainers**: ToolForge Team

