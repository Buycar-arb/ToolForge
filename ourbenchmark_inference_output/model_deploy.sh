source activate swift
CUDA_VISIBLE_DEVICES="0,1,2,3" swift deploy \
    --model your checkpoint path \
    --infer_backend vllm \
    --tensor_parallel_size 4 \
    --max_new_tokens 8192 \
    --served_model_name history-8B
