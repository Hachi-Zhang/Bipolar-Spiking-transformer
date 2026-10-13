"""SSA and ShiftMax attention implementations."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from .quantization import MultiSpikeNorm, SignedSSAQuantizer

def shiftmax_exp2(logits: Tensor) -> Tensor:
    """Stable ShiftMax exp2 activation (intentionally not sum-normalized)."""
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    return torch.exp2(shifted.clamp(min=-60.0, max=60.0))

class SegmentationAttention(nn.Module):
    """SDTv3 attention with signed SSA or ShiftMax plus LayerNorm."""
    MODES = ("ssa", "shiftmax_ln")

    def __init__(self, dim: int, num_heads: int = 8, value_ratio: int = 4,
                 mode: str = "ssa", shiftmax_levels: int = 16) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if shiftmax_levels not in (8, 16):
            raise ValueError("shiftmax_levels must be 8 or 16")
        self.dim, self.num_heads, self.value_ratio = dim, num_heads, value_ratio
        self.mode, self.shiftmax_levels = mode, shiftmax_levels
        self.scale = (dim // num_heads) ** -0.5
        value_dim = dim * value_ratio
        self.input_spike = MultiSpikeNorm(shiftmax_levels)
        if mode == "shiftmax_ln":
            self.q_proj, self.k_proj = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, value_dim, bias=False)
            self.q_norm, self.k_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.v_norm = nn.LayerNorm(value_dim)
            self.q_quant, self.k_quant = MultiSpikeNorm(shiftmax_levels), MultiSpikeNorm(shiftmax_levels)
            self.v_quant, self.output_quant = nn.Identity(), MultiSpikeNorm(shiftmax_levels)
        else:
            self.q_proj, self.k_proj = self._conv_bn(dim, dim), self._conv_bn(dim, dim)
            self.v_proj = self._conv_bn(dim, value_dim)
            self.q_quant = SignedSSAQuantizer()
            self.k_quant = SignedSSAQuantizer()
            self.v_quant = SignedSSAQuantizer()
            self.output_quant = SignedSSAQuantizer()
        self.output_proj = self._conv_bn(value_dim, dim)

    @staticmethod
    def _conv_bn(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))

    @staticmethod
    def _linear_project(inputs: Tensor, projection: nn.Linear, norm: nn.LayerNorm) -> Tensor:
        batch, _, height, width = inputs.shape
        tokens = norm(projection(inputs.flatten(2).transpose(1, 2)))
        return tokens.transpose(1, 2).reshape(batch, -1, height, width)

    def _heads(self, inputs: Tensor, channels: int) -> Tensor:
        batch, _, height, width = inputs.shape
        return (inputs.flatten(2).transpose(1, 2)
                .reshape(batch, height * width, self.num_heads, channels // self.num_heads)
                .permute(0, 2, 1, 3).contiguous())

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        value_channels = channels * self.value_ratio
        inputs = self.input_spike(inputs)
        if self.mode == "shiftmax_ln":
            q = self._linear_project(inputs, self.q_proj, self.q_norm)
            k = self._linear_project(inputs, self.k_proj, self.k_norm)
            v = self._linear_project(inputs, self.v_proj, self.v_norm)
        else:
            q, k, v = self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)
        q = self._heads(self.q_quant(q), channels)
        k = self._heads(self.k_quant(k), channels)
        v = self._heads(self.v_quant(v), value_channels)
        if self.mode == "shiftmax_ln":
            attended = shiftmax_exp2((q * self.shiftmax_levels) @ (k * self.shiftmax_levels).transpose(-2, -1)) @ v
        else:
            attended = (q @ k.transpose(-2, -1) @ v) * (self.scale * 2)
        attended = attended.transpose(2, 3).reshape(batch, value_channels, height * width)
        attended = self.output_quant(attended).reshape(batch, value_channels, height, width)
        return self.output_proj(attended)
