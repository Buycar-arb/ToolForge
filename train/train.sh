PATH=/usr/local/conda/envs/swift/bin:$PATH
CUDA_VISIBLE_DEVICES="0,1,2,3" \
NPROC_PER_NODE=4 \
swift sft \
    --model your base model path \
    --dataset 'your train dataset path' \
    --val_dataset 'your eval dataset path' \
    --output_dir your output path \
    --packing_cache your cache path \
    --custom_register_path qwen3_mix/qwen3_think.py \
    --template mymymy \
    --split_dataset_ratio 0 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --greater_is_better False \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 20 \
    --max_length 12000 \
    --attn_impl flash_attn \
    --packing false \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 1 \
    --report_to all