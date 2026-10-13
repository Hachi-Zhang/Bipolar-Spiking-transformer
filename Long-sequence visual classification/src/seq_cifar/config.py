from __future__ import annotations
from pathlib import Path
import yaml

MODEL_NAMES={"TIMv2_e3_seq","TIMv2_e3_seq_binary","TIMv2_e3_seq_ternary","TIMv2_e3_seq_ternary_shiftmax","TIMv2_bl_qk_seq","TIMv2_bl_sdt_seq","TIMv2_bl_spikf_seq"}
HIERARCHICAL_NAMES={"TIMv2_e3_seq","TIMv2_e3_seq_binary","TIMv2_e3_seq_ternary","TIMv2_e3_seq_ternary_shiftmax","TIMv2_bl_qk_seq"}
DEFAULTS={
"data_dir":"./data/CIFAR10","dataset":"scifar","num_classes":10,"img_size":32,"mean":[.4914,.4822,.4465],"std":[.247,.2435,.2616],"crop_pct":1.0,"scale":[1.,1.],"ratio":[1.,1.],"color_jitter":0.,"interpolation":"bicubic","train_interpolation":"bicubic","aa":None,"epochs":400,"mixup":0.,"mixup_off_epoch":0,"mixup_prob":1.,"mixup_mode":"batch","mixup_switch_prob":.5,"cutmix":0.,"reprob":0.,"remode":"const","model":"TIMv2_e3_seq","step":4,"patch_size":4,"in_channels":3,"embed_dim":384,"num_heads":12,"mlp_ratio":4.,"attn_scale":.125,"mlp_drop":0.,"attn_drop":0.,"depths":4,"sequence_length":1024,"amp":False,"batch_size":128,"val_batch_size":128,"lr":5e-4,"min_lr":1e-5,"sched":"cosine","weight_decay":.06,"cooldown_epochs":10,"warmup_epochs":20,"warmup_lr":1e-5,"opt":"adamw","smoothing":.1,"workers":4,"seed":42,"log_interval":100,"output":"./outputs/run","device":"auto","download":False,"synthetic_data":False,"max_train_batches":None,"max_val_batches":None,"resume":"","clip_grad":None,
}

def load_config(path,overrides=None):
    raw=yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw,dict): raise ValueError("YAML root must be a mapping")
    unknown=set(raw)-set(DEFAULTS)
    if unknown: raise ValueError(f"unknown YAML keys: {', '.join(sorted(unknown))}")
    cfg={**DEFAULTS,**raw}
    for key,value in (overrides or {}).items():
        if value is not None: cfg[key]=value
    validate_config(cfg); return cfg

def validate_config(c):
    if c["model"] not in MODEL_NAMES: raise ValueError(f"unknown model {c['model']!r}")
    if c["dataset"]!="scifar": raise ValueError("only dataset: scifar is supported")
    if c["opt"].lower()!="adamw" or c["sched"].lower()!="cosine": raise ValueError("only AdamW with cosine scheduling is supported")
    if c["img_size"]!=32 or c["sequence_length"]!=c["img_size"]**2: raise ValueError("seq-CIFAR requires img_size=32 and sequence_length=1024")
    if c["in_channels"]!=3 or c["num_classes"]<=0: raise ValueError("seq-CIFAR requires three channels and positive num_classes")
    for key in ("step","depths","epochs","batch_size","val_batch_size","embed_dim","num_heads"):
        if not isinstance(c[key],int) or c[key]<1: raise ValueError(f"{key} must be a positive integer")
    if not isinstance(c["workers"],int) or c["workers"]<0: raise ValueError("workers must be non-negative")
    if c["warmup_epochs"]<0 or c["cooldown_epochs"]<0 or c["warmup_epochs"]>c["epochs"]: raise ValueError("require 0 <= warmup_epochs <= epochs and cooldown_epochs >= 0")
    if c["max_train_batches"] is not None and c["max_train_batches"]<1: raise ValueError("max_train_batches must be positive")
    if c["max_val_batches"] is not None and c["max_val_batches"]<1: raise ValueError("max_val_batches must be positive")
    if c["lr"]<=0 or c["min_lr"]<0 or c["warmup_lr"]<0 or c["weight_decay"]<0: raise ValueError("learning rates and weight decay must be non-negative, with lr > 0")
    if len(c["mean"])!=3 or len(c["std"])!=3: raise ValueError("mean/std must have three values")
    if c["model"] in HIERARCHICAL_NAMES and c["depths"]<3: raise ValueError("hierarchical model depths must be at least 3")
    channels=(c["embed_dim"]//4,c["embed_dim"]//2,c["embed_dim"]) if c["model"] in HIERARCHICAL_NAMES else (c["embed_dim"],)
    if c["model"] in HIERARCHICAL_NAMES and c["embed_dim"]%4: raise ValueError("hierarchical model embed_dim must be divisible by 4")
    if any(channel%c["num_heads"] for channel in channels): raise ValueError("all attention stage channels must be divisible by num_heads")

def total_epochs(c): return int(c["epochs"])+int(c["cooldown_epochs"])
