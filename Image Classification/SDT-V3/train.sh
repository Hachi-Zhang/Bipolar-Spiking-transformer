CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  main_finetune.py \
  --batch_size 400 \
  --blr 6e-4 \
  --warmup_epochs 5 \
  --epochs 200 \
  --model Efficient_Spiking_Transformer_l \
  --data_path /data/maluzhang-folder/datasets/imagenet-1k-timm\
  --output_dir outputs/TT-6 \
  --log_dir outputs/TT-6 \
  --model_mode ms \







