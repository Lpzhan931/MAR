#!/bin/bash

# cuda 12.8.1
# transformers==5.3.0
# torch==2.8.0

BASE_MODEL="/home/share/models/Qwen3-8B/"
SMALL_MODEL="/home/pzli/Project/Spec/SpS/models/qwen3-tiny-ep3/"
DATASET="/home/pzli/Project/Spec/medusa/dataset/perfectblend_qwen3-8b_regen_20k.json"
LR=1e-3
FREEZE_SMALL_MODEL=False    # 小模型参与训练

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/output_qwen3_${TIMESTAMP}"

# # [Option 1] Default
# TOKEN_MIX_SIZE=1
# FUSION_LAYERS="-1"

# [Option 2] Token Mix Only
TOKEN_MIX_SIZE=3
FUSION_LAYERS="-1"

# # [Option 3] Layer Fusion Only
# TOKEN_MIX_SIZE=1
# FUSION_LAYERS="-1,-8,-16"

# # [Option 4] Both Token Mix and Layer Fusion
# TOKEN_MIX_SIZE=3
# FUSION_LAYERS="-1,-8,-16"


torchrun --master_port=29507 \
    --nproc_per_node=1 mar_train.py \
    --model_name_or_path "$BASE_MODEL" \
    --small_model_name_or_path "$SMALL_MODEL" \
    --data_path "$DATASET" \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 32 \
    --save_total_limit 1 \
    --learning_rate "$LR" \
    --weight_decay 0.0 \
    --warmup_steps 60 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 4 \
    --medusa_num_layers 1 \
    --freeze_small_model "$FREEZE_SMALL_MODEL" \
    --conv_kernel_size "$TOKEN_MIX_SIZE" \
    --extract_layers="$FUSION_LAYERS"
