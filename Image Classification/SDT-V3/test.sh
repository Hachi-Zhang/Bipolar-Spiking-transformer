#!/usr/bin/env bash
set -e

# Run fast numerical and shape checks before loading a large checkpoint.
python test_models.py

CUDA_VISIBLE_DEVICES=7 python main_finetune.py \
  --batch_size 32 \
  --model Efficient_Spiking_Transformer_l \
  --data_path /mnt/hdd1/zhanghaonan/code/ws-former/dataset/ImageNet-1K \
  --eval \
  --resume /mnt/hdd1/zhanghaonan/code/ws-former/metaformer-main/pami/outputs/TT-5/checkpoint-191.pth
