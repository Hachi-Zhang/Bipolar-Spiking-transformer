#!/usr/bin/env bash
set -euo pipefail

# Change this to the directory containing the ImageNet-style train/val folders.
DATA_PATH="/data/maluzhang-folder/datasets/imagenet-1k-timm"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python finetune_191.py \
  --data_path "$DATA_PATH" \
  --batch_size 500 \
  --num_workers 4 \
  --print_freq 100
