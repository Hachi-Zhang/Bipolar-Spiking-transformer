"""Standalone PyTorch seq-CIFAR models adapted from IJCV/cls.

Tensor shapes, temporal order, and architecture-specific paths are retained.
Local surrogate neurons are trainable approximations, not BrainCog bit-exact nodes.
"""
from __future__ import annotations
import torch
from torch import nn


def _spike(x: torch.Tensor, threshold: float) -> torch.Tensor:
    hard = (x >= threshold).to(x.dtype)
    soft = torch.sigmoid(4.0 * (x - threshold))
    return soft + (hard - soft).detach()


class LIF(nn.Module):
    """Layer-by-layer LIF for time-major flattened [T*B, ...] tensors."""
    def __init__(self, step=4, tau=2.0, threshold=1.0):
        super().__init__(); self.step=step; self.tau=tau; self.threshold=threshold; self.mem=None
    def reset(self): self.mem=None
    def forward(self, x):
        if x.shape[0] % self.step: raise ValueError("leading dimension must be divisible by step")
        xs=x.reshape(self.step,-1,*x.shape[1:]); mem=torch.zeros_like(xs[0]); out=[]
        for xt in xs:
            mem=mem+(xt-mem)/self.tau; spike=_spike(mem,self.threshold); mem=mem*(1-spike.detach()); out.append(spike)
        self.mem=mem.detach(); return torch.stack(out).flatten(0,1)


class TernarySTE(nn.Module):
    def __init__(self, threshold=.5): super().__init__(); self.threshold=threshold
    def forward(self,x):
        hard=torch.sign(x)*(x.abs()>=self.threshold).to(x.dtype); soft=x.clamp(-1,1)
        return soft+(hard-soft).detach()


def reset_net(module: nn.Module):
    for child in module.modules():
        if isinstance(child,LIF): child.reset()


def shiftmax(x,scale=1.0):
    x=x*scale; x=x-x.amax(dim=-1,keepdim=True)
    return torch.exp2(x.clamp(-60.0,0.0))  # intentionally no row-sum normalization


def spike_factory(mode,step,threshold=1.0): return TernarySTE() if mode=="ternary" else LIF(step,threshold=threshold)


class ConvMLP(nn.Module):
    def __init__(self,c,step,ratio=4):
        super().__init__(); h=int(c*ratio)
        self.net=nn.Sequential(nn.Conv1d(c,h,1),nn.BatchNorm1d(h),LIF(step),nn.Conv1d(h,c,1),nn.BatchNorm1d(c),LIF(step))
    def forward(self,x): return self.net(x)


class TemporalMLP(nn.Module):
    """Source T_MLP: reverse-time gated recurrence, restored to original order."""
    def __init__(self,c,step,ratio=4):
        super().__init__(); self.step=step; self.hidden=int(c*ratio)
        self.forget_x=nn.Conv1d(c,self.hidden,1); self.forget_h=nn.Conv1d(self.hidden,self.hidden,1)
        self.input_x=nn.Conv1d(c,self.hidden,1); self.fc_conv2=nn.Conv1d(self.hidden,c,1)
        self.fc_bn2=nn.BatchNorm1d(c); self.fc_lif2=LIF(step)
    def forward(self,x):
        if x.shape[0]%self.step: raise ValueError("leading dimension must be divisible by step")
        xs=x.reshape(self.step,-1,*x.shape[1:]); ht=None; gate_mem=None; reverse_states=[]
        for index in range(self.step-1,-1,-1):
            xt=xs[index]
            if ht is None: candidate=self.input_x(xt)
            else:
                forget=torch.sigmoid(self.forget_x(xt)+self.forget_h(ht)); candidate=forget*ht+(1-forget)*self.input_x(xt)
            gate_mem=torch.zeros_like(candidate) if gate_mem is None else gate_mem
            gate_mem=gate_mem+(candidate-gate_mem)/2.0; ht=_spike(gate_mem,.5); gate_mem=gate_mem*(1-ht.detach())
            reverse_states.append(ht)
        ordered=torch.stack(list(reversed(reverse_states))).flatten(0,1)
        return self.fc_lif2(self.fc_bn2(self.fc_conv2(ordered)))


class TIMv2(nn.Module):
    def __init__(self,step): super().__init__(); self.step=step; self.theta=nn.Parameter(torch.logit(torch.tensor(.5)))
    def forward(self,x):
        t=self.step; z=x.reshape(t,-1); a=.5+.5*torch.sigmoid(self.theta)
        i=torch.arange(t,device=x.device)[:,None]; j=torch.arange(t,device=x.device)[None,:]
        k=a*(1-a).pow((i-j).clamp_min(0))*(i>=j); k[:,0]=(1-a).pow(torch.arange(t,device=x.device,dtype=x.dtype))
        return (k@z).reshape_as(x)


class TokenAttention(nn.Module):
    def __init__(self,c,step,heads,temporal=True):
        super().__init__(); self.heads=heads
        self.q=nn.Sequential(nn.Conv1d(c,c,1,bias=False),nn.BatchNorm1d(c),LIF(step)); self.k=nn.Sequential(nn.Conv1d(c,c,1,bias=False),nn.BatchNorm1d(c),LIF(step))
        self.v=nn.Sequential(nn.Conv1d(c,c,3,padding=1,bias=False),nn.BatchNorm1d(c)); self.tm=TIMv2(step) if temporal else nn.Identity(); self.vlif=LIF(step,threshold=.5)
        self.a=LIF(step,threshold=.5); self.p=nn.Sequential(nn.Conv1d(c,c,1),nn.BatchNorm1d(c),LIF(step))
    def forward(self,x):
        tb,c,n=x.shape; q=self.q(x).reshape(tb,self.heads,c//self.heads,n).sum(2,keepdim=True); k=self.k(x).reshape(tb,self.heads,c//self.heads,n)
        return self.p((self.a(q)*k).flatten(1,2)*self.vlif(self.tm(self.v(x))))


class SSA(nn.Module):
    def __init__(self,c,step,heads,scale=.125,mode="binary",shift=False):
        super().__init__(); self.h=heads; self.scale=scale; self.shift=shift; self.tm=nn.Identity()
        self.q=nn.Sequential(nn.Conv1d(c,c,1,bias=False),nn.BatchNorm1d(c),spike_factory(mode,step)); self.k=nn.Sequential(nn.Conv1d(c,c,1,bias=False),nn.BatchNorm1d(c),spike_factory(mode,step))
        self.v=nn.Sequential(nn.Conv1d(c,c,1,bias=False),nn.BatchNorm1d(c),spike_factory(mode,step)); self.a=spike_factory(mode,step,.5)
        self.p=nn.Sequential(nn.Conv1d(c,c,1),nn.BatchNorm1d(c),spike_factory(mode,step))
    def forward(self,x):
        tb,c,n=x.shape
        def heads(z): return z.transpose(-1,-2).reshape(tb,n,self.h,c//self.h).permute(0,2,1,3)
        q,k,v=heads(self.q(x)),heads(self.k(x)),heads(self.v(x)); y=shiftmax(q@k.transpose(-2,-1),self.scale)@v if self.shift else (q@(k.transpose(-2,-1)@v))*self.scale
        return self.p(self.a(y.transpose(2,3)).reshape(tb,c,n))


class Block(nn.Module):
    def __init__(self,c,step,heads,ratio=4,kind="ssa",scale=.125,mode="binary",shift=False,temporal=False,temporal_mlp=False):
        super().__init__(); self.attn=TokenAttention(c,step,heads,temporal) if kind=="token" else SSA(c,step,heads,scale,mode,shift)
        self.mlp=TemporalMLP(c,step,ratio) if temporal_mlp else ConvMLP(c,step,ratio)
    def forward(self,x):
        x=x+self.attn(x)
        return x+self.mlp(x)


class PatchInit(nn.Module):
    def __init__(self,inc,c,step):
        super().__init__(); h=c//2; self.body=nn.Sequential(nn.Conv1d(inc,h,3,padding=1,bias=False),nn.BatchNorm1d(h),LIF(step),nn.Conv1d(h,c,3,padding=1,bias=False),nn.BatchNorm1d(c),LIF(step)); self.pe=nn.Sequential(nn.Conv1d(c,c,3,padding=1),nn.BatchNorm1d(c),LIF(step))
    def forward(self,x): x=self.body(x); return x+self.pe(x)


class Downsample(nn.Module):
    def __init__(self,inc,c,step):
        super().__init__(); self.main=nn.Sequential(nn.Conv1d(inc,c,3,padding=1,bias=False),nn.BatchNorm1d(c),LIF(step),nn.Conv1d(c,c,3,padding=1,bias=False),nn.BatchNorm1d(c),nn.MaxPool1d(3,2,1),LIF(step)); self.skip=nn.Sequential(nn.Conv1d(inc,c,1,stride=2),nn.BatchNorm1d(c),LIF(step))
    def forward(self,x): return self.main(x)+self.skip(x)


class HierarchicalModel(nn.Module):
    def __init__(self,step=4,in_channels=3,num_classes=10,embed_dim=384,num_heads=12,mlp_ratio=4,attn_scale=.125,depths=4,sequence_length=1024,mode="binary",shift=False,qk=False,**_):
        super().__init__(); self.step=step; self.in_channels=in_channels; self.sequence_length=sequence_length
        if depths<3: raise ValueError("depths must be >= 3")
        for c in (embed_dim//4,embed_dim//2,embed_dim):
            if c%num_heads: raise ValueError(f"stage channel {c} must be divisible by num_heads {num_heads}")
        self.p1=PatchInit(in_channels,embed_dim//4,step); self.b1=Block(embed_dim//4,step,num_heads,mlp_ratio,"token",temporal=not qk,temporal_mlp=not qk)
        self.p2=Downsample(embed_dim//4,embed_dim//2,step); self.b2=Block(embed_dim//2,step,num_heads,mlp_ratio,"token",temporal=not qk,temporal_mlp=not qk)
        self.p3=Downsample(embed_dim//2,embed_dim,step); self.blocks=nn.ModuleList([Block(embed_dim,step,num_heads,mlp_ratio,"ssa",attn_scale,mode,shift) for _ in range(depths-2)]); self.head=nn.Linear(embed_dim,num_classes)
    def _validate_input(self,x):
        expected=(self.in_channels,self.sequence_length)
        if x.ndim!=3 or tuple(x.shape[1:])!=expected: raise ValueError(f"expected [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}")
    def forward(self,x):
        self._validate_input(x)
        x=x.unsqueeze(0).repeat(self.step,1,1,1).flatten(0,1); x=self.b1(self.p1(x)); x=self.b2(self.p2(x)); x=self.p3(x)
        for block in self.blocks: x=block(x)
        return self.head(x.mean(-1).reshape(self.step,-1,x.shape[1]).mean(0))


class SequenceEmbed(nn.Module):
    def __init__(self,inc,c,step,sequence_length=1024,spike_last=True):
        super().__init__(); chans=[inc,c//8,c//4,c//2,c]; strides=[2,2,2,2] if sequence_length>=1024 else [1,2,2,2]; layers=[]
        for i,s in enumerate(strides):
            layers += [nn.Conv1d(chans[i],chans[i+1],5,stride=s,padding=2),nn.BatchNorm1d(chans[i+1])]
            if i<3 or spike_last: layers += [LIF(step)]
        self.net=nn.Sequential(*layers); self.pool=nn.AdaptiveAvgPool1d(64); self.pe=nn.Sequential(nn.Conv1d(c,c,3,padding=1),nn.BatchNorm1d(c))
    def forward(self,x): x=self.pool(self.net(x)); return x+self.pe(x)


class TokenMajorSSA(nn.Module):
    def __init__(self,c,step,heads,scale):
        super().__init__(); self.h=heads; self.scale=scale
        def branch(): return nn.Sequential(nn.Linear(c,c),TransposeBN(c),LIF(step))
        self.q=branch(); self.k=branch(); self.v=branch(); self.a=LIF(step,threshold=.5); self.p=nn.Sequential(nn.Linear(c,c),TransposeBN(c),LIF(step))
    def forward(self,x):
        tb,n,c=x.shape
        def heads(z): return z.reshape(tb,n,self.h,c//self.h).permute(0,2,1,3)
        q,k,v=heads(self.q(x)),heads(self.k(x)),heads(self.v(x)); y=(q@k.transpose(-2,-1))*self.scale@v
        return self.p(self.a(y.transpose(1,2).reshape(tb,n,c)))


class TransposeBN(nn.Module):
    def __init__(self,c): super().__init__(); self.bn=nn.BatchNorm1d(c)
    def forward(self,x): return self.bn(x.transpose(1,2)).transpose(1,2)


class TokenMajorMLP(nn.Module):
    def __init__(self,c,step,ratio):
        super().__init__(); h=int(c*ratio); self.net=nn.Sequential(nn.Linear(c,h),TransposeBN(h),LIF(step),nn.Linear(h,c),TransposeBN(c),LIF(step))
    def forward(self,x): return self.net(x)


class SpikformerBlock(nn.Module):
    def __init__(self,c,step,heads,ratio,scale): super().__init__(); self.attn=TokenMajorSSA(c,step,heads,scale); self.mlp=TokenMajorMLP(c,step,ratio)
    def forward(self,x): x=x+self.attn(x); return x+self.mlp(x)


def _validate_sequence_input(x,in_channels,sequence_length):
    expected=(in_channels,sequence_length)
    if x.ndim!=3 or tuple(x.shape[1:])!=expected: raise ValueError(f"expected [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}")


class SpikformerModel(nn.Module):
    def __init__(self,step=4,in_channels=3,num_classes=10,embed_dim=384,num_heads=12,mlp_ratio=4,attn_scale=.125,depths=4,sequence_length=1024,**_):
        super().__init__(); self.step=step; self.in_channels=in_channels; self.sequence_length=sequence_length
        if embed_dim%num_heads: raise ValueError("embed_dim must be divisible by num_heads")
        self.embed=SequenceEmbed(in_channels,embed_dim,step,sequence_length); self.blocks=nn.ModuleList([SpikformerBlock(embed_dim,step,num_heads,mlp_ratio,attn_scale) for _ in range(depths)]); self.head=nn.Linear(embed_dim,num_classes)
    def forward(self,x):
        _validate_sequence_input(x,self.in_channels,self.sequence_length)
        x=x.unsqueeze(0).repeat(self.step,1,1,1).flatten(0,1); x=self.embed(x).transpose(1,2)
        for block in self.blocks: x=block(x)
        return self.head(x.mean(1).reshape(self.step,-1,x.shape[-1]).mean(0))


class SDSA(nn.Module):
    def __init__(self,c,step,heads):
        super().__init__(); self.h=heads; self.shortcut=LIF(step)
        def branch(): return nn.Sequential(nn.Conv2d(c,c,1,bias=False),nn.BatchNorm2d(c),LIF(step))
        self.q=branch(); self.k=branch(); self.v=branch(); self.talking=nn.Conv1d(heads,heads,1,bias=False); self.talking_lif=LIF(step); self.proj=nn.Sequential(nn.Conv2d(c,c,1),nn.BatchNorm2d(c))
    def forward(self,x):
        identity=x; x=self.shortcut(x); tb,c,h,w=x.shape; n=h*w
        def heads(z): return z.flatten(2).transpose(1,2).reshape(tb,n,self.h,c//self.h).permute(0,2,1,3)
        q,k,v=heads(self.q(x)),heads(self.k(x)),heads(self.v(x)); kv=(k*v).sum(2,keepdim=True)
        kv=self.talking_lif(self.talking(kv.squeeze(2))).unsqueeze(2); y=(q*kv).transpose(1,2).reshape(tb,n,c).transpose(1,2).reshape(tb,c,h,w)
        return self.proj(y)+identity


class Conv2dResidualMLP(nn.Module):
    def __init__(self,c,step,ratio):
        super().__init__(); h=int(c*ratio); self.l1=LIF(step); self.c1=nn.Conv2d(c,h,1); self.b1=nn.BatchNorm2d(h); self.l2=LIF(step); self.c2=nn.Conv2d(h,c,1); self.b2=nn.BatchNorm2d(c)
    def forward(self,x): return x+self.b2(self.c2(self.l2(self.b1(self.c1(self.l1(x))))))


class SDTBlock(nn.Module):
    def __init__(self,c,step,heads,ratio): super().__init__(); self.attn=SDSA(c,step,heads); self.mlp=Conv2dResidualMLP(c,step,ratio)
    def forward(self,x): return self.mlp(self.attn(x))


class SDTModel(nn.Module):
    def __init__(self,step=4,in_channels=3,num_classes=10,embed_dim=384,num_heads=12,mlp_ratio=4,depths=4,sequence_length=1024,**_):
        super().__init__(); self.step=step; self.in_channels=in_channels; self.sequence_length=sequence_length
        if embed_dim%num_heads: raise ValueError("embed_dim must be divisible by num_heads")
        self.embed=SequenceEmbed(in_channels,embed_dim,step,sequence_length,spike_last=False); self.blocks=nn.ModuleList([SDTBlock(embed_dim,step,num_heads,mlp_ratio) for _ in range(depths)]); self.head_lif=LIF(step); self.head=nn.Linear(embed_dim,num_classes)
    def forward(self,x):
        _validate_sequence_input(x,self.in_channels,self.sequence_length)
        x=x.unsqueeze(0).repeat(self.step,1,1,1).flatten(0,1); x=self.embed(x).reshape(x.shape[0],-1,8,8)
        for block in self.blocks: x=block(x)
        z=x.flatten(2).mean(2); z=self.head_lif(z).reshape(self.step,-1,z.shape[-1]); return self.head(z).mean(0)


MODEL_NAMES=("TIMv2_e3_seq","TIMv2_e3_seq_binary","TIMv2_e3_seq_ternary","TIMv2_e3_seq_ternary_shiftmax","TIMv2_bl_qk_seq","TIMv2_bl_sdt_seq","TIMv2_bl_spikf_seq")
def build_model(config):
    cfg=dict(config); name=cfg.pop("model")
    if name not in MODEL_NAMES: raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    allowed={"step","img_size","patch_size","in_channels","num_classes","embed_dim","num_heads","mlp_ratio","attn_scale","depths","sequence_length","mlp_drop","attn_drop"}; kw={k:v for k,v in cfg.items() if k in allowed}
    if name.startswith("TIMv2_e3"): return HierarchicalModel(**kw,mode="ternary" if "ternary" in name else "binary",shift=name.endswith("shiftmax"))
    if name=="TIMv2_bl_qk_seq": return HierarchicalModel(**kw,qk=True)
    if name=="TIMv2_bl_sdt_seq": return SDTModel(**kw)
    return SpikformerModel(**kw)
