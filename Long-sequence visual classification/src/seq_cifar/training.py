from __future__ import annotations
import csv, math, os, random, tempfile
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from .config import total_epochs
from .data import build_loaders
from .models import build_model, reset_net

class WarmupCosineCooldown:
    def __init__(self,opt,c): self.opt=opt; self.c=c; self.base=c["lr"]; self.epoch=-1
    def step(self,epoch):
        self.epoch=epoch; warm=self.c["warmup_epochs"]; e=epoch
        if warm and e<warm:
            progress=e/max(1,warm-1); lr=self.c["warmup_lr"]+(self.base-self.c["warmup_lr"])*progress
        elif e<self.c["epochs"]:
            cosine_epochs=self.c["epochs"]-warm; progress=1.0 if cosine_epochs<=1 else (e-warm)/(cosine_epochs-1)
            lr=self.c["min_lr"]+.5*(self.base-self.c["min_lr"])*(1+math.cos(math.pi*progress))
        else: lr=self.c["min_lr"]
        for g in self.opt.param_groups:g["lr"]=lr
        return lr
    def state_dict(self): return {"epoch":self.epoch}
    def load_state_dict(self,s): self.epoch=s["epoch"]

def resolve_device(value):
    s=str(value).lower()
    if s=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s.isdigit(): s=f"cuda:{s}" if torch.cuda.is_available() else "cpu"
    d=torch.device(s)
    if d.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return d

def mixup(x,y,alpha,classes):
    if alpha<=0:return x,y
    lam=torch.distributions.Beta(alpha,alpha).sample().item(); idx=torch.randperm(x.size(0),device=x.device)
    one=F.one_hot(y,classes).float(); return lam*x+(1-lam)*x[idx],lam*one+(1-lam)*one[idx]
def loss_fn(logits,target,smoothing=0.):
    if target.ndim==2:return -(target*F.log_softmax(logits,dim=1)).sum(1).mean()
    return F.cross_entropy(logits,target,label_smoothing=smoothing)
def accuracy(out,y):
    k=min(5,out.shape[1]); pred=out.topk(k,1).indices; hit=pred.eq(y[:,None]); return hit[:,:1].any(1).float().sum().item(),hit.any(1).float().sum().item()

def _rng():
    state={"python":random.getstate(),"torch":torch.get_rng_state()}
    try:
        import numpy as np; state["numpy"]=np.random.get_state()
    except ImportError: pass
    if torch.cuda.is_available():state["cuda"]=torch.cuda.get_rng_state_all()
    return state
def _restore_rng(s):
    random.setstate(s["python"]);torch.set_rng_state(s["torch"])
    try:
        import numpy as np
        if "numpy" in s:np.random.set_state(s["numpy"])
    except ImportError:pass
    if "cuda" in s and torch.cuda.is_available():torch.cuda.set_rng_state_all(s["cuda"])

def _atomic_torch_save(state,path):
    path=Path(path); fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent); os.close(fd)
    try:
        torch.save(state,tmp); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def run_training(c):
    out=Path(c["output"]);out.mkdir(parents=True,exist_ok=True);(out/"resolved_config.yml").write_text(yaml.safe_dump(c,sort_keys=True))
    random.seed(c["seed"]);torch.manual_seed(c["seed"])
    try:
        import numpy as np;np.random.seed(c["seed"])
    except ImportError:pass
    device=resolve_device(c["device"]); model=build_model(c).to(device); train,val=build_loaders(c)
    opt=torch.optim.AdamW(model.parameters(),lr=c["lr"],weight_decay=c["weight_decay"]); sched=WarmupCosineCooldown(opt,c)
    use_amp=bool(c["amp"] and device.type=="cuda")
    try: scaler=torch.amp.GradScaler("cuda",enabled=use_amp)
    except (AttributeError,TypeError): scaler=torch.cuda.amp.GradScaler(enabled=use_amp)
    start=0;best=-1.;global_step=0
    if c["resume"]:
        ck=torch.load(c["resume"],map_location=device,weights_only=False);model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);sched.load_state_dict(ck["scheduler"]);scaler.load_state_dict(ck["scaler"]);start=ck.get("next_epoch",ck["epoch"]+1);global_step=ck.get("global_step",0);best=ck["best_metric"];_restore_rng(ck["rng"])
    metrics=out/"metrics.csv"; write_header=not metrics.exists() or start==0
    with metrics.open("a",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["epoch","lr","train_loss","val_loss","top1","top5"])
        if write_header:writer.writeheader()
        for epoch in range(start,total_epochs(c)):
            lr=sched.step(epoch);model.train();tl=0.;tn=0
            for bi,(x,y) in enumerate(train):
                if c["max_train_batches"] is not None and bi>=c["max_train_batches"]:break
                x,y=x.to(device),y.to(device); x,target=mixup(x,y,c["mixup"] if epoch<c["mixup_off_epoch"] or not c["mixup_off_epoch"] else 0,c["num_classes"])
                opt.zero_grad(set_to_none=True)
                ctx=torch.amp.autocast("cuda",enabled=use_amp) if hasattr(torch,"amp") else torch.cuda.amp.autocast(enabled=use_amp)
                with ctx: pred=model(x);loss=loss_fn(pred,target,c["smoothing"])
                scaler.scale(loss).backward()
                if c["clip_grad"] is not None:scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),c["clip_grad"])
                scaler.step(opt);scaler.update();global_step+=1;reset_net(model);tl+=loss.item()*x.size(0);tn+=x.size(0)
            model.eval();vl=top1=top5=vn=0.
            with torch.no_grad():
                for bi,(x,y) in enumerate(val):
                    if c["max_val_batches"] is not None and bi>=c["max_val_batches"]:break
                    x,y=x.to(device),y.to(device); pred=model(x);loss=F.cross_entropy(pred,y);a1,a5=accuracy(pred,y);reset_net(model);n=x.size(0);vl+=loss.item()*n;top1+=a1;top5+=a5;vn+=n
            row={"epoch":epoch,"lr":lr,"train_loss":tl/max(tn,1),"val_loss":vl/max(vn,1),"top1":100*top1/max(vn,1),"top5":100*top5/max(vn,1)};writer.writerow(row);f.flush();print(" ".join(f"{k}={v:.4g}" if isinstance(v,float) else f"{k}={v}" for k,v in row.items()))
            improved=row["top1"]>best;best=max(best,row["top1"])
            state={"format_version":2,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"next_epoch":epoch+1,"global_step":global_step,"best_metric":best,"resolved_config":c,"rng":_rng()}
            _atomic_torch_save(state,out/"checkpoint-last.pt")
            if improved:_atomic_torch_save(state,out/"checkpoint-best.pt")
    return {"output":out,"best_metric":best,"start_epoch":start,"global_step":global_step}
