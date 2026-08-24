#!/usr/bin/env bash
# Serve a ToolForge-fine-tuned checkpoint with Swift + vLLM, ready for
# benchmark/run_benchmark.py.
#
#   bash deploy_model.sh /path/to/checkpoint [served-name] [gpus]
#
# Then, in another shell:
#   python run_benchmark.py output/data/case_C1.jsonl ours.jsonl \
#       --model toolforge-8b --base-url http://0.0.0.0:8000/v1 --api-key EMPTY

set -euo pipefail

CHECKPOINT="${1:?usage: deploy_model.sh <checkpoint-path> [served-name] [gpus]}"
SERVED_NAME="${2:-toolforge-8b}"
GPUS="${3:-0,1,2,3}"
TENSOR_PARALLEL="$(awk -F, '{print NF}' <<< "$GPUS")"

echo "checkpoint : $CHECKPOINT"
echo "served as  : $SERVED_NAME"
echo "GPUs       : $GPUS  (tensor_parallel_size=$TENSOR_PARALLEL)"
echo

CUDA_VISIBLE_DEVICES="$GPUS" swift deploy \
    --model "$CHECKPOINT" \
    --infer_backend vllm \
    --tensor_parallel_size "$TENSOR_PARALLEL" \
    --max_new_tokens 8192 \
    --served_model_name "$SERVED_NAME"
