# Bipolar Self-Attention

Bipolar Self-Attention (BSA) improves fine-grained token correlation modeling and attention allocation in spiking Transformers. This repository contains task-specific implementations for image classification, semantic segmentation, language modeling, long-sequence visual classification, and event-based tracking. Each task has its own setup; please refer to the corresponding directory for configuration and execution instructions.

## Method overview

BSA addresses three limitations of spiking self-attention (SSA) through Shift-DyT, signed correlation computation, and ShiftMax.

### 1. Reduced Q/K/V channel diversity

**Problem.** BatchNorm in SSA suppresses differences in membrane-potential distributions across Q/K/V channels, limiting channel-specific representations.

**Solution.** Apply Shift-DyT to the attention Q/K/V normalization paths to preserve channel-specific responses. The image classification implementations first use LayerNorm, followed by a scheduled switch to Shift-DyT or Shift-DyT fine-tuning from a LayerNorm checkpoint. The replacement is limited to the attention Q/K/V paths; other BatchNorm layers retain their original settings.

### 2. Loss of polarity information

**Problem.** Binary Q–K products capture positive-positive matches but discard other polarity interactions.

**Solution.** Use signed Q–K products to retain negative, zero, and positive correlation values. QKFormer and Spikingformer use ternary Q/K spikes, while SDT-V3 uses signed multi-level spikes.

### 3. Inconsistent attention-score scales

**Problem.** Without Softmax normalization, attention-score rows in SSA can differ in scale, affecting attention allocation.

**Solution.** ShiftMax applies row-wise base-2 exponential weighting to obtain low-entropy attention allocation similar to Softmax. It uses approximate row normalization, so the weights in each row do not necessarily sum to one.

## Code map

| Task | Implementation |
| --- | --- |
| Image classification: SDT-V3 | Signed attention, base-2 attention weighting, and a scheduled switch from LayerNorm to Shift-DyT for Q/K/V. |
| Image classification: QKFormer | Ternary Q/K, base-2 attention weighting, and Shift-DyT fine-tuning from a LayerNorm checkpoint. |
| Image classification: Spikingformer | Ternary Q/K, base-2 attention weighting, and Shift-DyT fine-tuning from a LayerNorm checkpoint. |
| Semantic segmentation | An MMSegmentation plugin for ADE20K, with signed SSA and configurable ShiftMax variants. |
| NLP tasks | SpikeBERT attention and normalization ablations, MLM/NSP pretraining, and GLUE fine-tuning. |
| Long-sequence visual classification | Model implementations, training and evaluation code, and task-specific configurations for seq-CIFAR. |
| Event-based tracking | SDTrack-based experiments on FE108, VisEvent, and FELT, with dataset setup and training/testing scripts. |

The task directories are independent implementations. The BSA components enabled by default depend on the task configuration. Consult the corresponding guides for dependencies, datasets, configurations, and evaluation details.

## Upstream repositories and acknowledgments

The image classification, semantic segmentation, and event-based tracking implementations build on the following open-source models and frameworks, with BSA integrated into the corresponding task code.

| Task | Base model or framework |
| --- | --- |
| Image classification: SDT-V3 | Spike-Driven Transformer V3 |
| Image classification: QKFormer | QKFormer |
| Image classification: Spikingformer | Spikingformer |
| Semantic segmentation | SDT-V3 and MMSegmentation |
| Event-based tracking | SDTrack |

We thank the authors of these projects for their open-source implementations. Please also cite the corresponding original papers when using their code.

## Image classification examples

### SDT-V3

Prepare an ImageNet-style dataset directory containing `train/` and `val/` subdirectories. Run the following command from `Image Classification/SDT-V3`:

```bash
python main_finetune.py --model Efficient_Spiking_Transformer_s \
  --data_path /path/to/imagenet --norm_mode auto --norm_switch_epoch 100 \
  --output_dir ./outputs/sdt-v3
```

This configuration first trains with LayerNorm, then switches the attention Q/K/V normalization paths to Shift-DyT at the zero-based epoch specified by `--norm_switch_epoch`.

### QKFormer and Spikingformer

These implementations use a completed LayerNorm checkpoint for Shift-DyT fine-tuning. Run the following commands from the respective model directories:

```bash
# Image Classification/QKformer
python train.py --model QKFormer_8_512 --data_path /path/to/imagenet \
  --finetune /path/to/layernorm-checkpoint.pth --shift-dyt \
  --output_dir ./outputs/qkformer-shiftdyt

# Image Classification/Spikingformer
python train.py --config imagenet.yml -data-dir /path/to/imagenet \
  --initial-checkpoint /path/to/layernorm-checkpoint.pth.tar --shift-dyt \
  --output ./outputs/spikingformer-shiftdyt
```

To resume a Shift-DyT checkpoint, use `--resume` together with `--shift-dyt`. The checkpoint must match the selected architecture and number of classes.

### Dependencies

The image classification code uses PyTorch, torchvision, timm, einops, and TensorBoard. QKFormer and Spikingformer also use SpikingJelly’s `clock_driven` API, while SDT-V3 imports torchinfo. The three directories do not share a pinned environment; match package versions to your PyTorch and CUDA setup.
