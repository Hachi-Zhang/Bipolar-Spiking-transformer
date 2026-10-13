# Seq-CIFAR 独立训练项目

本目录现在是一个可安装、可训练、可测试的独立 PyTorch 项目，不依赖 `/data/260010055/IJCV`、BrainCog 或 SpikingJelly。静态 CIFAR-10 图像经 `SeqCIFAR` 转为 `[3, 1024]`，模型在内部复制为 `step=4` 个时间步。

## 安装与运行

```bash
cd /data/260010055/IJCV-Code/seq-cifar
python -m pip install -e '.[test]'
python tools/train.py --config configs/main/teformer_scifar.yml \
  --data-dir /path/to/CIFAR10 --output ./outputs/teformer --device cuda:0 --download
```

默认不下载数据；显式使用 `--download`。训练集是 `torchvision.datasets.CIFAR10(train=True)`，验证集是 `train=False`。正式配置包括：

- `main/teformer_scifar.yml`：`TIMv2_e3_seq`（binary alias）
- `ablations/{binary_ssa,ternary_ssa,ternary_shiftmax}_scifar.yml`
- `baselines/{qkformer,sdt,spikformer}_scifar.yml`

快速无网络 CPU 检查（正式 384 维 smoke YAML 仍较慢，建议 CLI 覆盖限制 batch，或使用测试中的 tiny 配置）：

```bash
python tools/train.py --config configs/smoke/teformer_scifar_smoke.yml \
  --synthetic-data --device cpu --workers 0 --max-train-batches 1 \
  --max-val-batches 1 --output /tmp/seq-cifar-smoke
```

恢复训练：

```bash
python tools/train.py --config configs/main/teformer_scifar.yml \
  --resume ./outputs/teformer/checkpoint-last.pt --output ./outputs/teformer
```

优先级为内置 defaults < YAML < CLI。未知 YAML 键直接报错。每次运行保存 `resolved_config.yml`、`metrics.csv`、`checkpoint-last.pt` 和最佳 top-1 的 `checkpoint-best.pt`。checkpoint 采用临时文件加原子替换写入，包含格式版本、模型、优化器、scheduler、scaler、`epoch`/`next_epoch`、`global_step`、最佳指标、resolved config，以及 Python/Torch/可用时 NumPy 和 CUDA RNG。

## 训练语义

训练支持 CPU/单 GPU、AdamW、label smoothing、batch Mixup、top-1/top-5、梯度裁剪、CUDA AMP、warmup + cosine + cooldown。正式 YAML 的 `epochs=400` 是主调度长度，`cooldown_epochs=10` 是额外阶段，因此总计 410 epoch。总轮数始终严格定义为 `epochs + cooldown_epochs`，不含隐藏特判。smoke YAML 显式设置 `epochs=1`、`warmup_epochs=0`、`cooldown_epochs=0`，因此明确只运行 1 轮。当前实现明确是单进程，不声称支持 torchrun/DDP。

增强包括 random crop、horizontal flip、torchvision RandAugment、normalization 和 random erasing。当前 `rand-m9-n1-mstd0.4-inc1` 保留 magnitude=9 / num_ops=1；torchvision API 没有 timm 的 `mstd` 与 `inc` 等价参数，因此不是逐随机数精确复现。Mixup 在 batch 上实现。

## 模型适配说明

主 TIMv2 路径保留三级 1D patch stages、两级 token blocks、stage3 SSA 和时间平均分类头。stage1/stage2 的 V 使用 TIMv2 mixing，MLP 使用源 `T_MLP` 的反向时间 forget-gate 递归；stage3 SSA 的 V 不做 TIMv2 mixing。LIF 与 reset 为纯 PyTorch surrogate，ternary 使用 STE，ShiftMax 使用稳定 `exp2` 且不做 row-sum normalization。QKFormer 保留普通 MLP，并将输入改为复制时间步且保留 RGB 通道，修复源实现将 `[B,3,N]` 变为 `[3B,1,N]` 的问题。

两个 baseline 使用不同本地路径：Spikformer 是 sequence embedding 后的 token-major Linear SSA/MLP；SDT 将 64 token reshape 为 `8×8`，使用 Conv2d SDSA（elementwise `k*v` 聚合、talking-head LIF）、Conv2d residual MLP 和 head LIF。SDT 构造使用有效的 `sequence_length` 和显式 `step`，修复源代码错传 `img_h/img_w` 且 block 漏传 step 的问题。部分实现采用数学等价的通道/Token 维重排，但本地 surrogate、初始化和框架状态管理与 BrainCog 不同，因此不是 bit-exact 重写，也不保证旧 checkpoint 可直接加载。

尚未进行 400+10 epoch 全量 GPU 训练或指标核验，不声明论文精度已复现。

## 验证

```bash
python tools/validate_configs.py
pytest -q
python -m compileall -q src tools tests
```
