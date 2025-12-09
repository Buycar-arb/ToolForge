# Evaluation Framework

Evaluate LLMs with two search methods: tag-based (`<search>` tags) and function-based (direct function calls).

## Setup

### 1. Install Dependencies

```bash
pip install -r evaluations/requirements.txt
```

### 2. Configuration

All configs in `evaluations/config/`:
- **models.yaml**: Set `active_model` and API keys
- **datasets.yaml**: Set `active_datasets`
- **search_engines.yaml**: Set `search_method` (tag/function)
- **prompts.yaml**: Customize prompts

### 3. Start RAG Server

```bash
cd current_repo
pip3 install cachebox
pip3 install accelerate bitsandbytes datasets deepspeed==0.16.4 einops flash-attn==2.7.0.post2 isort jsonlines loralib optimum packaging peft pynvml>=12.0.0 ray[default]==2.46.0 tensorboard torch==2.6.0 torchmetrics tqdm transformers==4.51.3 transformers_stream_generator wandb wheel
pip3 install vllm==0.8.5      # Mainly for Qwen3 model support
pip3 install "qwen-agent[code_interpreter]"
pip3 install llama_index bs4 pymilvus infinity_client codetiming tensordict==0.6 omegaconf torchdata==0.10.0 hydra-core easydict dill python-multipart mcp==1.9.3
pip3 install -e . --no-deps
pip3 install faiss-gpu-cu12   # Optional, needed for end-to-end search model training with rag_server
pip3 install nvidia-cublas-cu12==12.4.5.8  # Optional, needed while encountering ray worker died issue during training


# 指向你自己的目录
export HF_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yanghaocheng04/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yanghaocheng04/tmp

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

bash rag_server/launch.sh
```

### 4. (Optional) Start vLLM Server

For open-source models:
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B \
    --port 8000 \
    --served-model-name qwen3-8b # 需要与models.yaml中注册的名字一致
```

## Run Evaluation

From the **root directory**:

```bash
# Tag-based search
python evaluations/run_evaluation.py 
```

## Recalculate Metrics

After evaluation completes, you can recalculate metrics independently:

```bash
# Recalculate and update metrics in-place
python evaluations/src/metrics/metrics.py evaluations/results/gpt-4_function_20250918_104723/bamboogle_results.json

# Print metrics without updating file
python evaluations/src/metrics/metrics.py evaluations/results/gpt-4_function_20250918_104723/bamboogle_results.json --print
```


## Datasets

- **NQ**: Natural Questions
- **PopQA**: Popular QA
- **Musique**: Multi-hop reasoning
- **Bamboogle**: Complex reasoning

Data cached in `evaluations/data/` after first download.

## Results
```bash
Tag-based方式：
  {
      'id': str,           # 示例ID
      'question': str,     # 原始问题
      'gold_answer': str,  # 标准答案
      'prediction': str,   # 模型预测
      'response': str      # 完整响应文本
  }

  Function-based方式：
  {
      'id': str,           # 示例ID
      'question': str,     # 原始问题
      'gold_answer': str,  # 标准答案
      'prediction': str,   # 模型预测
      'messages': List     # 完整对话历史
  }
```

## Search Methods

### Tag-based
- For Search-R1 trained models
- Uses `<search>query</search>` tags
- Direct text generation

### Function-based
- For any model supporting function/tool calling
- Specialized search functions (food, cloth, people, location, general)
- Direct execution without subprocess overhead

## Metrics

- **Exact Match (EM)**: Exact answer matching
- **F1 Score**: Token-level overlap
- **Search Stats**: Queries per question, iterations

