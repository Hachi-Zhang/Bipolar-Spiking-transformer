# SpikeBERT: Spiking BERT with Signed SSA, ShiftMax, and Dynamic Tanh

A research-oriented implementation of a temporally unfolded BERT encoder with spike-driven linear layers, configurable attention, temporal feedback, and normalization ablations.

## Abstract

SpikeBERT adapts a Hugging Face BERT encoder to shared-weight temporal computation. Most encoder projections use `SpikeLinear`, which maintains membrane state and quantizes activations into learned-scale ternary spikes while retaining FP32 weights. The repository exposes three self-attention operators—legacy spike-driven softmax, signed SSA without softmax, and base-2 ShiftMax—and three normalization families—LayerNorm, Dynamic Tanh, and a pulse-oriented Shifted Dynamic Tanh. It supports MLM+NSP pretraining and GLUE-style sequence classification.

This repository is an implementation and ablation scaffold. The included MRPC runs are two-step smoke tests that validate the pipeline; they are not benchmark results and do not establish superiority of any attention or normalization choice.

## Contributions implemented in code

- Shared-weight temporal BERT with `network` and `layerwise` execution.
- FP32 linear weights with learned-scale ternary activation spikes.
- Softmax, signed SSA, and ShiftMax attention under a common interface.
- LayerNorm, DyT, and Shifted-DyT as global normalization choices.
- Optional direct and spike-linear feedback across complete network loops.
- Mean/last temporal readout.
- MLM+NSP and sequence-classification heads.
- Local/Hugging Face GLUE loading with MRPC accuracy and F1 evaluation.

No energy, latency, sparsity, neuromorphic-hardware, or paper-level accuracy claim is made by the current code and outputs.

## Repository layout

```text
spikebert/
├── src/spikebert/
│   ├── modeling_spikebert.py  # BERT, temporal encoder, attention, task heads
│   ├── spiking.py             # SpikeLinear and ElasticBiSpiking
│   ├── quantization.py        # SSA and ShiftMax quantizers
│   ├── normalization.py       # LayerNorm, DyT, Shifted-DyT factory
│   ├── attention.py           # ShiftMax and mask helpers
│   └── configuration.py       # config augmentation
├── scripts/
│   ├── pretrain.py            # MLM+NSP training
│   ├── finetune_glue.py       # GLUE training and validation
│   └── evaluate.py            # logits-only evaluation for tokenized data
├── configs/ablations/         # documented parameter templates
├── tests/
└── outputs/                   # ignored checkpoints and smoke metrics
```

The JSON files under `configs/ablations/` are reference templates; the current scripts do not automatically parse them.

## Spike-driven computation

### SpikeLinear

`SpikeLinear` keeps FP32 weights and applies learned-scale ternary quantization to activations. A simplified temporal recurrence is

\[
m_0=x_0,
\]

\[
m_t=x_t+0.25\,m_{t-1}(c_{t-1}-s_{t-1}),
\]

\[
s_t=Q_{\mathrm{ternary}}(m_t;c_t),\qquad y_t=Ws_t+b.
\]

The clipping scale \(c_t\) is initialized from activation statistics. `input_bits=2` produces ternary spikes; `weight_bits=32` means this implementation is not a weight-quantized BERT.

In `network` mode, membrane state is carried across calls for a full temporal loop and reset after the encoder forward. In `layerwise` mode, a layer processes the complete temporal tensor at once.

## Attention mechanisms

Let \(\widetilde Q,\widetilde K,\widetilde V\) denote SpikeLinear projections and \(d_h\) the head dimension.

### Legacy spike-driven softmax

\[
Q=\widetilde Q,\quad
K=Q_{\mathrm{EB}}(\widetilde K),\quad
V=Q_{\mathrm{EB}}(\widetilde V),
\]

\[
A=\operatorname{Dropout}\left(
\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)
\right),\qquad C=AV.
\]

Only K and V receive the additional legacy `ElasticBiSpiking` operation. Returned attentions are dropout-affected softmax weights.

### Signed SSA

SSA uses

\[
q_s(x)=\frac{\operatorname{round}(\operatorname{clip}(x,-4,4))}{4},
\]

whose nine values are

\[
\{-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1\}.
\]

The attention is

\[
Q=q_s(\widetilde Q),\quad K=q_s(\widetilde K),\quad V=q_s(\widetilde V),
\]

\[
A_{ij}=
\begin{cases}
2Q_iK_j^\top/\sqrt{d_h},&j\text{ is valid},\\
0,&j\text{ is padding},
\end{cases}
\qquad C=q_s(AV).
\]

SSA does not use softmax, row normalization, or attention dropout. Its returned attention tensor is a signed, unnormalized weight matrix—not a probability distribution.

### ShiftMax

For clipping upper bound \(L\in\{8,16\}\),

\[
q_L(x)=\frac{\operatorname{round}(\operatorname{clip}(x,0,L))}{L}.
\]

After the selected normalization \(N\),

\[
Q=q_L(N_Q(\widetilde Q)),\quad
K=q_L(N_K(\widetilde K)),\quad
V=N_V(\widetilde V).
\]

V is normalized but not quantized by `MultiSpikeNorm`. Scores and weights are

\[
S=(LQ)(LK)^\top,
\]

\[
A_{ij}=2^{\operatorname{clip}(S_{ij}-\max_kS_{ik}+M_{ij},-60,60)}.
\]

Padding locations are explicitly zeroed after exponentiation, attention dropout is applied, and

\[
C=q_L(AV).
\]

ShiftMax uses base-2 exponentiation, does not divide scores by \(\sqrt{d_h}\), and does not normalize rows to sum to one. `max8` and `max16` produce 9 and 17 non-negative values, respectively.

The historical mode name is `shiftmax_ln`, but Q/K/V use the globally selected normalization. With `normalization_mode=dyt`, for example, this mode is ShiftMax + DyT rather than ShiftMax + LayerNorm.

## Normalization

`normalization_mode` globally controls embedding, residual, MLM-head, optional feedback, and ShiftMax Q/K/V normalization.

### LayerNorm

\[
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
\]

### Dynamic Tanh

\[
\operatorname{DyT}(x)=\gamma\odot\tanh(\alpha x)+\beta,
\]

where \(\alpha\) is a learned scalar and \(\gamma,\beta\) are learned channel parameters. The default initialization is \(\alpha_0=0.5\).

### Shifted Dynamic Tanh

\[
\operatorname{ShiftedDyT}(x)=
\gamma\odot\tanh\left(\alpha(x+\delta)\right).
\]

The learned channel shift \(\delta\) acts before `tanh`; there is no post-activation bias. This changes the location of features entering the subsequent computation and is intended as a pulse-oriented alternative to an output bias. Not every normalization site is immediately followed by SpikeLinear, so this should not be interpreted as a hardware-equivalent bit shift.

## Temporal execution and feedback

### Network mode

Each time step traverses the complete shared-weight encoder. The previous complete encoder output can be injected into the next step.

- No explicit feedback: `--no use_history` (default).
- Direct feedback:
  \[
  x_t=E+\alpha h_{t-1}.
  \]
- Spike-linear feedback:
  \[
  x_t=E+W_fq_{\{-1,0,1\}}(N(h_{t-1})).
  \]

The feedback projection is bias-free and identity-initialized. Spike-linear feedback requires `network` mode and `use_history=True`.

### Layerwise mode

Embeddings are replicated to `[T,B,L,D]`; each layer processes all time steps before execution advances to the next layer. This is useful for legacy-path compatibility but does not support spike-linear feedback or temporal-state output.

### Temporal readout

- `mean`: average all complete-loop states.
- `last`: use only the final state.

Even with `use_history=False`, SpikeLinear still carries membrane state across temporal steps.

## Choosing a configuration

| Goal | Attention | Normalization | Notes |
|---|---|---|---|
| Maximum compatibility with legacy SpikeLM | `softmax` | `layernorm` | Best starting point for old checkpoints |
| Isolate signed, non-softmax attention | `ssa` | `layernorm` | Signed, unnormalized weights |
| Isolate ShiftMax | `shiftmax_ln`, max8 | `layernorm` | Start with coarser 9-value Q/K/context |
| Study ShiftMax resolution | `shiftmax_ln`, max16 | `layernorm` | 17-value Q/K/context |
| Remove variance normalization | any | `dyt` | Global model-wide replacement |
| Move features before bounded activation | any | `shifted_dyt` | Global model-wide replacement |

Recommended experimental order:

1. Softmax + LayerNorm compatibility baseline.
2. ShiftMax max8 + LayerNorm.
3. Change only normalization to Shifted-DyT.
4. Compare max8/max16 or add SSA while keeping all other settings fixed.

No current result demonstrates that one choice is superior.

## Installation

The project targets Python 3.10.

```bash
cd /data/260010055/IJCV-Code/spikebert
python -m pip install -e ".[train,test]"
```

For the current local environment, the previously prepared SpikeLM virtual environment can also be used:

```bash
export PYTHONPATH=/data/260010055/IJCV-Code/spikebert/src
PY=/data/260010055/LoopTransformer/spikelm/spikeLM-BERT/.venv/bin/python
```

## Local model assets

Training scripts load config and tokenizer from `--model_name_or_path`. The finetuning script sets `local_files_only=True`, so a local BERT directory is the most reliable input:

```text
/data/260010055/LoopTransformer/spikelm/spikeLM-BERT/huggingface/bert-base-uncased
```

Important: `model_name_or_path` initializes config and tokenizer only. It does not automatically load BERT model weights. Use `--checkpoint` to initialize SpikeBERT weights.

Checkpoint loading filters by key and shape, then uses non-strict loading. Always inspect the reported matched/missing/unexpected keys; successful loading does not guarantee semantic equivalence across attention or normalization modes.

## MRPC data

MRPC has been downloaded through a Hugging Face mirror and saved as a `DatasetDict` at:

```text
/data/260010055/dataset/glue/mrpc
```

| Split | Examples |
|---|---:|
| train | 3,668 |
| validation | 408 |
| test | 1,725 |

Fields are `sentence1`, `sentence2`, `label`, and `idx`.

## MRPC smoke tests

The following commands reproduce the pipeline checks. They use only 32 training examples, 32 validation examples, and two optimization steps. They verify data loading, pair tokenization, forward/backward execution, checkpoint saving, and metric calculation—not model quality.

### ShiftMax max8 + LayerNorm

```bash
$PY scripts/finetune_glue.py \
  --model_name_or_path /data/260010055/LoopTransformer/spikelm/spikeLM-BERT/huggingface/bert-base-uncased \
  --dataset_path /data/260010055/dataset/glue/mrpc \
  --task_name mrpc \
  --output_dir outputs/mrpc-shiftmax8-layernorm-smoke \
  --attention_mode shiftmax_ln \
  --attention_max_value 8 \
  --normalization_mode layernorm \
  --time_steps 1 \
  --batch_size 4 \
  --max_length 32 \
  --max_train_samples 32 \
  --max_eval_samples 32 \
  --max_train_steps 2 \
  --learning_rate 2e-5 \
  --seed 42
```

### ShiftMax max8 + Shifted-DyT

```bash
$PY scripts/finetune_glue.py \
  --model_name_or_path /data/260010055/LoopTransformer/spikelm/spikeLM-BERT/huggingface/bert-base-uncased \
  --dataset_path /data/260010055/dataset/glue/mrpc \
  --task_name mrpc \
  --output_dir outputs/mrpc-shiftmax8-shifted-dyt-smoke \
  --attention_mode shiftmax_ln \
  --attention_max_value 8 \
  --normalization_mode shifted_dyt \
  --dyt_alpha_init 0.5 \
  --time_steps 1 \
  --batch_size 4 \
  --max_length 32 \
  --max_train_samples 32 \
  --max_eval_samples 32 \
  --max_train_steps 2 \
  --learning_rate 2e-5 \
  --seed 42
```

Observed smoke metrics:

| Configuration | Steps | Train/eval examples | Mean train loss | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| ShiftMax max8 + LayerNorm | 2 | 32 / 32 | 0.7474 | 0.375 | 0.0 |
| ShiftMax max8 + Shifted-DyT | 2 | 32 / 32 | 0.6931 | 0.375 | 0.0 |

Both models obtain identical smoke accuracy/F1. The loss difference is not evidence that Shifted-DyT is better. The model body was not initialized from a task-appropriate pretrained SpikeBERT checkpoint, and the run is intentionally too small for scientific comparison.

Machine-readable results:

```text
outputs/mrpc-shiftmax8-comparison.json
```

## Full-data MRPC template

A more meaningful run should include a pretrained SpikeBERT checkpoint and multiple seeds:

```bash
$PY scripts/finetune_glue.py \
  --model_name_or_path /path/to/local/bert-base-uncased \
  --checkpoint /path/to/pretrained-spikebert/model.safetensors \
  --dataset_path /data/260010055/dataset/glue/mrpc \
  --task_name mrpc \
  --output_dir outputs/mrpc-shiftmax8-shifted-dyt-full \
  --attention_mode shiftmax_ln \
  --attention_max_value 8 \
  --normalization_mode shifted_dyt \
  --dyt_alpha_init 0.5 \
  --time_steps 4 \
  --temporal_mode network \
  --temporal_readout mean \
  --batch_size 16 \
  --epochs 3 \
  --learning_rate 2e-5 \
  --max_length 128 \
  --seed 42
```

This is a full-data template, not a complete paper protocol: the script currently lacks warmup/scheduling, best-checkpoint selection, resume, AMP, distributed training, and multi-seed aggregation.

## Pretraining

The pretraining input must be a tokenized `datasets.save_to_disk()` dataset containing at least `input_ids`, `labels`, and `next_sentence_label`.

```bash
$PY scripts/pretrain.py \
  --model_name_or_path /path/to/local/bert-base-uncased \
  --dataset_path /path/to/tokenized_mlm_nsp_dataset \
  --output_dir outputs/pretrain-ssa \
  --attention_mode ssa \
  --normalization_mode layernorm \
  --time_steps 4 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --max_steps 1000
```

`pretrain.py` currently performs single-process random initialization and has no checkpoint input, validation, scheduler, warmup, AMP, resume, or DDP. Do not use `torchrun` with it without first adding proper distributed initialization and sharding.

## Evaluation

`finetune_glue.py` performs validation and writes `metrics.json` after training. This is the preferred MRPC evaluation path at present.

`scripts/evaluate.py` is a logits-only utility for datasets that are already tokenized and have raw text columns removed. It does not tokenize raw MRPC and does not compute accuracy/F1.

## Verification status

Validated previously in the prepared environment:

```text
19 tests passed
Python compileall passed
three CLI help paths passed
Hugging Face config/checkpoint round-trip passed
```

Tests cover:

- SSA values and STE gradients;
- ShiftMax max8/max16 and padding masks;
- three attention modes and finite backward passes;
- network/layerwise tiny MLM+NSP paths;
- temporal classification logits;
- LayerNorm, DyT, and Shifted-DyT replacement;
- pre-tanh channel shift and absence of output bias;
- CUDA smoke execution when available.

Not yet established:

- complete pretraining or full MRPC convergence;
- attention/normalization superiority;
- multi-seed reproducibility;
- direct/spike-linear history benchmark quality;
- AMP, DDP, gradient checkpointing, long-sequence stability;
- decoder/cross-attention/cached decoding;
- energy, sparsity, latency, memory, or neuromorphic deployment.

## Reporting recommendations

For academic comparisons, keep checkpoint, data split, tokenizer, temporal mode, readout, history, seed, optimizer, scheduler, and training budget fixed. Report mean and standard deviation across seeds, matched checkpoint coverage, accuracy/F1, trainable parameters, spike rate, throughput, peak memory, and hardware. Treat attention and normalization as separate experimental axes.

## Citation and license

A final paper citation, BibTeX entry, and standalone license are not yet included. Before public release, add the original SpikeLM/BERT attribution, the Dynamic Tanh reference, author information, and explicit licensing for the adapted Hugging Face code.
