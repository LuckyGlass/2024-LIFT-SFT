#!/bin/bash
#SBATCH -J LIFTSFT
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:8
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-LIFTSFT-refine-1205.out.log
#SBATCH -e logs/%j-LIFTSFT-refine-1205.err.log

BASE_MODEL="models/Meta-Llama-3-8B-Instruct"
OUTPUT_PATH="models/LIFTSFT-refine-1205"
DATA_PATH="data/timeline-train.json"
PORT=$((1024 + RANDOM % (65535 - 1024 + 1)))

wandb login --relogin 7cb3dada5935174b3d1b35a051f0e5cabc2d7be1
wandb login

deepspeed --master_port=${PORT} --include=localhost:0,1,2,3,4,5 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path ${BASE_MODEL} \
    --full_finetune True \
    --data_path ${DATA_PATH} \
    --len_segment 8 \
    --len_offset 3 \
    --block_size 256 \
    --input_cache_path ${DATA_PATH}.cache.pkl \
    --output_dir ${OUTPUT_PATH} \
    --num_train_epochs 3 \
    --model_max_length 8000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to wandb \
    --run_name LIFTSFT-refine-1205
