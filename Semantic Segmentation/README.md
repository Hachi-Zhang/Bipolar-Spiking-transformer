# EFSDT Segmentation

A lightweight MMSegmentation plugin for studying signed spike-driven attention and ShiftMax attention in semantic segmentation.

## Abstract

This repository extracts the model-specific segmentation components from a larger MMSegmentation fork into a compact plugin. It implements a four-stage SDTv3-style backbone, a quantized feature pyramid neck, and a quantized Semantic FPN head. Two attention families are exposed through configuration: a signed nine-value SSA operator without softmax normalization, and a non-negative multi-value ShiftMax operator with token-wise LayerNorm and base-2 exponentiation.

The repository is intended as a transparent implementation and ablation scaffold. It does not currently include trained checkpoints, ADE20K mIoU results, or evidence supporting accuracy, latency, or energy-efficiency claims.

## Highlights

- MMSegmentation registry integration without replacing the upstream `mmseg` package.
- Signed SSA quantization for Q, K, V, and aggregated context.
- ShiftMax with stable base-2 exponentiation and configurable `max8`/`max16` clipping.
- Four-scale SDTv3 backbone connected to `QFPN` and `QFPNHead`.
- Pure PyTorch quantizers; no runtime dependency on SpikingJelly.
- Unit tests for quantization range, STE gradients, attention shapes, and numerical finiteness.

## Repository layout

```text
seg/
├── efsdt_seg/
│   └── models/
│       ├── quantization.py   # STE, signed SSA, non-negative MultiSpike
│       ├── attention.py      # SSA and ShiftMax
│       ├── sdtv3.py          # four-stage backbone
│       ├── qfpn.py           # quantized FPN neck
│       └── qfpn_head.py      # quantized decode head
├── configs/
│   ├── _base_/               # ADE20K, runtime, 160k schedule
│   └── efsdtv3/              # attention ablations
├── tools/                    # train/test entry points
└── tests/
```

## Method



### ShiftMax

ShiftMax first applies non-negative input quantization, then uses bias-free Linear projections and token-wise normalization. Q and K are quantized, while V remains the continuous normalized projection:

\[
Q'=Q_L^+(\operatorname{LN}(Q)),\quad
K'=Q_L^+(\operatorname{LN}(K)),\quad
V'=\operatorname{LN}(V).
\]

The integer-domain score and ShiftMax weight are

\[
Z=(LQ')(LK')^\top,
\]

\[
S_{ij}=2^{\operatorname{clip}(Z_{ij}-\max_k Z_{ik},-60,60)}.
\]

The context is

\[
C=Q_L^+(SV').
\]

ShiftMax is not softmax: its rows do not sum to one. The subtraction of the row maximum only stabilizes exponentiation and maps a maximum score to one.

## Architecture

The default backbone uses channel widths `[32, 64, 128, 192]` and returns four feature maps:

| Output | Channels | Spatial stride |
|---|---:|---:|
| `x1` | 16 | 2 |
| `x2` | 32 | 4 |
| `x3` | 64 | 8 |
| `x4` | 192 | 16 |

The final two stages contain six and two attention blocks, respectively. Each attention block combines spike separable convolution, attention, an MLP, and three LayerScale residual branches initialized to \(10^{-6}\).

`QFPN` projects all four scales to 128 channels and performs top-down nearest-neighbor fusion. `QFPNHead` aligns features to stride 2, sums them, and predicts 150 ADE20K classes.

Approximate parameter counts for the default implementation are:

| Component | Parameters |
|---|---:|
| Backbone | 4.896 M |
| QFPN | 0.632 M |
| QFPNHead | 1.053 M |
| Complete segmentation model | 6.581 M |

The common “5M” label refers to the backbone, not the complete segmentor.

## Configuration selection

| Configuration | Use it when | Important behavior |
|---|---|---|
| `sdtv3_5m_ssa_signed4_ade20k.py` | Starting from the extracted default or studying signed attention | Signed Q/K/V/context; no softmax |
| `sdtv3_5m_shiftmax_ln_max8_ade20k.py` | Studying coarser non-negative ShiftMax | 9-value Q/K and context quantization |
| `sdtv3_5m_shiftmax_ln_max16_ade20k.py` | Studying denser ShiftMax discretization | 17-value Q/K and context quantization |

There are currently no repository-local mIoU results demonstrating that one configuration is more accurate. Use SSA as the implementation baseline; compare max8 and max16 only as controlled ShiftMax ablations.

## Installation

Python 3.10 or newer is recommended because the source uses PEP 604 union types.

```bash
cd /data/260010055/IJCV-Code/seg

# Install a PyTorch build matching the local CUDA toolkit first.
python -m pip install -U openmim
mim install "mmcv>=2.0,<2.2"
python -m pip install -e ".[mmlab,test]"
```

Declared OpenMMLab ranges:

```text
MMEngine >= 0.7, < 1.0
MMCV     >= 2.0, < 2.2
MMSegmentation >= 1.1, < 1.3
```

MMCV wheel availability depends on the PyTorch/CUDA combination. The complete installation has not been validated in the current environment.

## ADE20K preparation

Expected layout:

```text
ADEChallengeData2016/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

The default path is:

```text
/data/260010055/dataset/ade20k/ADEChallengeData2016
```

To use another path, explicitly override every dataloader:

```bash
python tools/train.py \
  configs/efsdtv3/sdtv3_5m_ssa_signed4_ade20k.py \
  --cfg-options \
  train_dataloader.dataset.data_root=/path/to/ADEChallengeData2016 \
  val_dataloader.dataset.data_root=/path/to/ADEChallengeData2016 \
  test_dataloader.dataset.data_root=/path/to/ADEChallengeData2016
```

## Training and evaluation

Single GPU:

```bash
python tools/train.py \
  configs/efsdtv3/sdtv3_5m_ssa_signed4_ade20k.py \
  --work-dir work_dirs/ssa_signed4
```

ShiftMax max8:

```bash
python tools/train.py \
  configs/efsdtv3/sdtv3_5m_shiftmax_ln_max8_ade20k.py \
  --work-dir work_dirs/shiftmax_max8
```

Four GPUs:

```bash
torchrun --nproc-per-node=4 tools/train.py \
  configs/efsdtv3/sdtv3_5m_shiftmax_ln_max16_ade20k.py \
  --launcher pytorch \
  --work-dir work_dirs/shiftmax_max16
```

The configured batch size is per process. With four GPUs, the default global batch is 64; learning rate is not automatically rescaled.

Resume:

```bash
python tools/train.py \
  configs/efsdtv3/sdtv3_5m_ssa_signed4_ade20k.py \
  --work-dir work_dirs/ssa_signed4 \
  --resume
```

Evaluate a checkpoint produced by training:

```bash
python tools/test.py \
  configs/efsdtv3/sdtv3_5m_ssa_signed4_ade20k.py \
  /absolute/path/to/checkpoint.pth
```

The repository does not ship a checkpoint; paths above are examples.

## Pretrained backbone initialization

Prefer a derived config instead of deeply overriding `init_cfg=None` from the command line:

```python
_base_ = ["./sdtv3_5m_shiftmax_ln_max8_ade20k.py"]

model = dict(
    backbone=dict(
        init_cfg=dict(checkpoint="/absolute/path/to/checkpoint.pth"),
    ),
)
```

The loader strips a leading `backbone.` prefix and can squeeze legacy `1x1 Conv2d` Q/K/V weights into Linear weights for ShiftMax. Loading is non-strict; always inspect missing and unexpected keys. This conversion is not a general guarantee of semantic checkpoint compatibility.

## Default optimization protocol

- ADE20K, 150 classes, crop size 512×512.
- AdamW, learning rate `1e-3`, weight decay `5e-3`.
- Dynamic AMP loss scaling.
- 1,500-iteration linear warmup followed by polynomial decay.
- 160,000 total iterations; validation/checkpoint interval 2,500.
- Whole-image inference; mIoU evaluation.

## Verification status

Verified in the current environment:

```text
9 unit tests passed
1 OpenMMLab integration test skipped
Python compileall passed
```

The tests cover quantization values, STE gradients, attention shape/finiteness, ShiftMax max8/max16, signed SSA module assignment, and LayerNorm use.

Not yet verified here:

- real MMCV/MMEngine/MMSegmentation installation and registry integration;
- full EncoderDecoder construction with the actual OpenMMLab runtime;
- ADE20K sample loading, single-/multi-GPU training, SyncBN, or AMP stability;
- checkpoint resume and legacy checkpoint conversion coverage;
- mIoU, FLOPs, latency, memory, energy, or multi-seed reproducibility.

## Academic reporting checklist

For a paper-quality experiment, report at least: exact config, dependency versions, GPU count, global batch size, seed, checkpoint initialization, mIoU, parameter count, FLOPs, throughput, peak memory, and mean/std over repeated runs. Do not compare max8/max16 based only on their quantization resolution.

## Citation and license

A project paper, BibTeX entry, and standalone license have not yet been added. Before public release, add the original paper citation, author information, upstream MMSegmentation attribution, and an explicit license.
