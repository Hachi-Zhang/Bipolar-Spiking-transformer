#!/usr/bin/env python3
"""Local/Hugging Face GLUE finetuning and validation for SpikeBERT."""
import argparse
import json
import logging
import random
from pathlib import Path

TASK_KEYS = {'cola': ('sentence', None), 'sst2': ('sentence', None), 'mrpc': ('sentence1', 'sentence2'),
             'qqp': ('question1', 'question2'), 'qnli': ('question', 'sentence'),
             'rte': ('sentence1', 'sentence2'), 'wnli': ('sentence1', 'sentence2'),
             'mnli': ('premise', 'hypothesis'), 'stsb': ('sentence1', 'sentence2')}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name_or_path', required=True); p.add_argument('--task_name', choices=TASK_KEYS, required=True)
    p.add_argument('--dataset_path'); p.add_argument('--checkpoint'); p.add_argument('--output_dir', required=True)
    p.add_argument('--attention_mode', choices=['softmax', 'ssa', 'shiftmax_ln'], default='softmax')
    p.add_argument('--attention_max_value', type=int, choices=[8, 16], default=16)
    p.add_argument('--normalization_mode', choices=['layernorm', 'dyt', 'shifted_dyt'], default='layernorm')
    p.add_argument('--dyt_alpha_init', type=float, default=0.5)
    p.add_argument('--time_steps', type=int, default=4); p.add_argument('--temporal_mode', choices=['network', 'layerwise'], default='network')
    p.add_argument('--temporal_readout', choices=['mean', 'last'], default='mean'); p.add_argument('--use_history', action='store_true')
    p.add_argument('--history_type', choices=['direct', 'spike_linear'], default='direct'); p.add_argument('--history_alpha_init', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=16); p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--learning_rate', type=float, default=2e-5); p.add_argument('--max_train_steps', type=int)
    p.add_argument('--max_train_samples', type=int); p.add_argument('--max_eval_samples', type=int)
    p.add_argument('--max_length', type=int, default=128); p.add_argument('--seed', type=int, default=42); p.add_argument('--device')
    return p.parse_args()

def load_data(args, load_dataset, load_from_disk):
    if not args.dataset_path:
        return load_dataset('glue', args.task_name)
    path = Path(args.dataset_path)
    if (path / 'dataset_dict.json').exists():
        return load_from_disk(str(path))
    parquet = path if path.name.endswith('parquet') else path / 'mrpc-parquet'
    files = {name: str(parquet / f'{name}.parquet') for name in ('train', 'validation', 'test')}
    if all(Path(value).exists() for value in files.values()):
        return load_dataset('parquet', data_files=files)
    raise FileNotFoundError(f'No DatasetDict or MRPC parquet files under {path}')

def load_checkpoint(model, checkpoint, torch):
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if path.is_dir():
        for name in ('model.safetensors', 'pytorch_model.bin'):
            if (path / name).exists(): path = path / name; break
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
    compatible = {k: v for k, v in state.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    result = model.load_state_dict(compatible, strict=False)
    logging.info('checkpoint matched=%d missing=%d unexpected=%d', len(compatible), len(result.missing_keys), len(result.unexpected_keys))
    return result

def main():
    args = parse_args()
    import torch
    from datasets import load_dataset, load_from_disk
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
    from spikebert import SpikeBertForSequenceClassification, configure_spikebert
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    raw = load_data(args, load_dataset, load_from_disk)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    first, second = TASK_KEYS[args.task_name]
    def tokenize(examples):
        return tokenizer(examples[first], examples[second] if second else None, truncation=True, max_length=args.max_length)
    encoded = raw.map(tokenize, batched=True, remove_columns=raw['train'].column_names)
    # Restore labels only; raw string/id columns must not reach the tokenizer collator.
    for split in encoded:
        encoded[split] = encoded[split].add_column('label', raw[split]['label'])
    if args.max_train_samples: encoded['train'] = encoded['train'].select(range(min(args.max_train_samples, len(encoded['train']))))
    eval_split = 'validation_matched' if args.task_name == 'mnli' else 'validation'
    if args.max_eval_samples: encoded[eval_split] = encoded[eval_split].select(range(min(args.max_eval_samples, len(encoded[eval_split]))))
    num_labels = 1 if args.task_name == 'stsb' else len(set(encoded['train']['label']))
    config = configure_spikebert(AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, local_files_only=True),
        time_steps=args.time_steps, temporal_mode=args.temporal_mode, temporal_readout=args.temporal_readout,
        use_history=args.use_history, history_type=args.history_type, history_alpha_init=args.history_alpha_init,
        attention_mode=args.attention_mode, attention_max_value=args.attention_max_value,
        normalization_mode=args.normalization_mode, dyt_alpha_init=args.dyt_alpha_init)
    model = SpikeBertForSequenceClassification(config); load_checkpoint(model, args.checkpoint, torch); model.to(device)
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(encoded['train'], batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(encoded[eval_split], batch_size=args.batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate); step = 0; losses = []
    model.train()
    for _ in range(args.epochs):
        for batch in train_loader:
            labels = batch.pop('labels'); batch = {k: v.to(device) for k, v in batch.items()}; labels = labels.to(device)
            loss = model(**batch, labels=labels).loss; loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach())); step += 1
            if args.max_train_steps and step >= args.max_train_steps: break
        if args.max_train_steps and step >= args.max_train_steps: break
    model.eval(); predictions = []; references = []
    with torch.no_grad():
        for batch in eval_loader:
            labels = batch.pop('labels'); batch = {k: v.to(device) for k, v in batch.items()}
            predictions.extend(model(**batch).logits.argmax(-1).cpu().tolist()); references.extend(labels.tolist())
    metrics = {'task': args.task_name, 'attention_mode': args.attention_mode, 'attention_max_value': args.attention_max_value,
        'normalization_mode': args.normalization_mode, 'steps': step, 'train_samples': len(encoded['train']),
        'eval_samples': len(encoded[eval_split]), 'mean_train_loss': sum(losses) / len(losses),
        'accuracy': accuracy_score(references, predictions), 'f1': f1_score(references, predictions), 'seed': args.seed}
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output); tokenizer.save_pretrained(output); (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__': main()
