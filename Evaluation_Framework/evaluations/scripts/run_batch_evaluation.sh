#!/bin/bash

# 批量运行评估脚本

echo "========================================="
echo "开始批量评估"
echo "========================================="

# 1. GPT-4 with function search
echo ""
echo "1. 运行 GPT-4 + function search + NQ dataset"
python3 evaluations/run_evaluation.py --model gpt-4 --method function --datasets nq

echo ""
echo "2. 运行 GPT-4 + function search + PopQA dataset"
python3 evaluations/run_evaluation.py --model gpt-4 --method function --datasets popqa

# 2. Qwen3-8B with function search
echo ""
echo "3. 运行 Qwen3-8B + function search + NQ dataset"
python3 evaluations/run_evaluation.py --model qwen3-8b --method function --datasets nq

echo ""
echo "4. 运行 Qwen3-8B + function search + PopQA dataset"
python3 evaluations/run_evaluation.py --model qwen3-8b --method function --datasets popqa

# 3. GPT-4 with tag search
echo ""
echo "5. 运行 GPT-4 + tag search + NQ dataset"
python3 evaluations/run_evaluation.py --model gpt-4 --method tag --datasets nq

echo ""
echo "6. 运行 GPT-4 + tag search + PopQA dataset"
python3 evaluations/run_evaluation.py --model gpt-4 --method tag --datasets popqa

# 4. Qwen3-8B with tag search
echo ""
echo "7. 运行 Qwen3-8B + tag search + NQ dataset"
python3 evaluations/run_evaluation.py --model qwen3-8b --method tag --datasets nq

echo ""
echo "8. 运行 Qwen3-8B + tag search + PopQA dataset"
python3 evaluations/run_evaluation.py --model qwen3-8b --method tag --datasets popqa

echo ""
echo "========================================="
echo "所有评估完成！结果保存在 evaluations/results/ 目录"
echo "========================================="