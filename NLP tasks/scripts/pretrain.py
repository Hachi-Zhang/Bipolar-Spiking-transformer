#!/usr/bin/env python3
"""Minimal MLM+NSP training entry using a save_to_disk dataset."""
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--attention_mode', choices=['softmax', 'ssa', 'shiftmax_ln'], default='softmax')
    parser.add_argument('--attention_max_value', type=int, choices=[8, 16], default=16)
    parser.add_argument('--normalization_mode', choices=['layernorm', 'dyt', 'shifted_dyt'], default='layernorm')
    parser.add_argument('--dyt_alpha_init', type=float, default=0.5)
    parser.add_argument('--time_steps', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--device')
    return parser.parse_args()

def main():
    args = parse_args()
    import torch
    from datasets import load_from_disk
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoTokenizer, default_data_collator
    from spikebert import SpikeBertForPreTraining, configure_spikebert
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    config = configure_spikebert(AutoConfig.from_pretrained(args.model_name_or_path),
        time_steps=args.time_steps, attention_mode=args.attention_mode,
        attention_max_value=args.attention_max_value, normalization_mode=args.normalization_mode,
        dyt_alpha_init=args.dyt_alpha_init)
    model = SpikeBertForPreTraining(config).to(device)
    dataset = load_from_disk(args.dataset_path)
    train = dataset['train'] if hasattr(dataset, 'keys') and 'train' in dataset else dataset
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train(); step = 0
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            if loss is None:
                raise ValueError('dataset must provide labels and next_sentence_label')
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if step >= args.max_steps:
                break
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output); AutoTokenizer.from_pretrained(args.model_name_or_path).save_pretrained(output)

if __name__ == '__main__':
    main()
