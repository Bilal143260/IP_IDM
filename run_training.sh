#!/bin/bash

# Activate virtual environment
# source /home/bilal/IP-Adapter-Controlnet/.venv/bin/activate

# Run training command in background
nohup accelerate launch  \
        train.py   \
        --pretrained_model_name_or_path="yisol/IDM-VTON"  \
        --data_json_file="/home/bilal/datasets/tested_data/data_2.json" \
        --data_root_path="/home/bilal/datasets/tested_data"  \
        --mixed_precision="fp16"  \
        --resolution=1024  \
        --train_batch_size=3  \
        --dataloader_num_workers=4  \
        --num_train_epochs=100 \
        --learning_rate=1e-05   \
        --weight_decay=0.01 \
        --output_dir="output"  \
        --save_steps=50000 > training_log.txt 2>&1 &