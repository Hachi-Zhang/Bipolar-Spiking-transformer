#!/usr/bin/env python3
"""Evaluate a saved SpikeBERT sequence classifier on a local dataset split."""
import argparse, json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--dataset_path',required=True);p.add_argument('--split',default='validation');p.add_argument('--batch_size',type=int,default=32);p.add_argument('--max_examples',type=int);p.add_argument('--output');a=p.parse_args()
 import torch
 from torch.utils.data import DataLoader
 from transformers import AutoTokenizer, DataCollatorWithPadding
 from spikebert import SpikeBertForSequenceClassification
 from datasets import load_from_disk
 device='cuda' if torch.cuda.is_available() else 'cpu';model=SpikeBertForSequenceClassification.from_pretrained(a.checkpoint).to(device).eval();tok=AutoTokenizer.from_pretrained(a.checkpoint)
 ds=load_from_disk(a.dataset_path)[a.split];ds=ds.select(range(min(a.max_examples,len(ds)))) if a.max_examples else ds
 loader=DataLoader(ds,batch_size=a.batch_size,collate_fn=DataCollatorWithPadding(tok));pred=[]
 with torch.no_grad():
  for b in loader:
   b.pop('labels',b.pop('label',None));b={k:v.to(device) for k,v in b.items()};pred.extend(model(**b).logits.cpu().tolist())
 result={'count':len(pred),'logits':pred};print(json.dumps(result));
 if a.output: Path(a.output).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
