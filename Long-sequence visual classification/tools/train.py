#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from seq_cifar.config import load_config
from seq_cifar.training import run_training

def parser():
 p=argparse.ArgumentParser(description="Standalone seq-CIFAR trainer");p.add_argument("--config",required=True)
 p.add_argument("--data-dir");p.add_argument("--output");p.add_argument("--device");p.add_argument("--epochs",type=int);p.add_argument("--batch-size",type=int);p.add_argument("--workers",type=int);p.add_argument("--resume");p.add_argument("--seed",type=int);p.add_argument("--max-train-batches",type=int);p.add_argument("--max-val-batches",type=int);p.add_argument("--clip-grad",type=float)
 p.add_argument("--download",action=argparse.BooleanOptionalAction,default=None);p.add_argument("--synthetic-data",action=argparse.BooleanOptionalAction,default=None);p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=None);return p
if __name__=="__main__":
 a=parser().parse_args(); overrides={k:v for k,v in vars(a).items() if k!="config" and v is not None};run_training(load_config(a.config,overrides))
