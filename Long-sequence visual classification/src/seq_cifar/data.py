from __future__ import annotations
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class SeqCIFAR(Dataset):
    """Converts a transformed CIFAR image [3,32,32] to [3,1024]."""
    def __init__(self, dataset): self.dataset=dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self,i):
        x,y=self.dataset[i]
        if x.shape != (3,32,32): raise ValueError(f"expected [3,32,32], got {tuple(x.shape)}")
        return x.reshape(3,1024), y

class SyntheticCIFAR(Dataset):
    def __init__(self,size=32,classes=10,seed=0): self.size=size; self.classes=classes; self.seed=seed
    def __len__(self): return self.size
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(self.seed+i); return torch.rand(3,1024,generator=g), i%self.classes

def _randaugment(spec):
    if not spec: return None
    m=re.fullmatch(r"rand-m(\d+)-n(\d+)(?:-mstd[\d.]+)?(?:-inc\d+)?",spec)
    if not m: raise ValueError(f"unsupported aa policy {spec!r}")
    # torchvision has no mstd/inc knobs; magnitude and operation count are preserved.
    return transforms.RandAugment(num_ops=int(m.group(2)),magnitude=int(m.group(1)))

def make_transforms(c):
    train=[transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()]
    ra=_randaugment(c.get("aa"))
    if ra: train.append(ra)
    train += [transforms.ToTensor(),transforms.Normalize(c["mean"],c["std"])]
    if c["reprob"]>0: train.append(transforms.RandomErasing(p=c["reprob"],value=0 if c["remode"]=="const" else "random"))
    return transforms.Compose(train),transforms.Compose([transforms.ToTensor(),transforms.Normalize(c["mean"],c["std"])])

def build_loaders(c):
    if c["synthetic_data"]:
        train=SyntheticCIFAR(max(c["batch_size"]*4,16),c["num_classes"],c["seed"]); val=SyntheticCIFAR(max(c["val_batch_size"]*2,10),c["num_classes"],c["seed"]+10000)
    else:
        tr,te=make_transforms(c)
        train=SeqCIFAR(datasets.CIFAR10(c["data_dir"],train=True,transform=tr,download=c["download"]))
        val=SeqCIFAR(datasets.CIFAR10(c["data_dir"],train=False,transform=te,download=c["download"]))
    kw=dict(num_workers=c["workers"],pin_memory=False)
    return DataLoader(train,batch_size=c["batch_size"],shuffle=True,**kw),DataLoader(val,batch_size=c["val_batch_size"],shuffle=False,**kw)
