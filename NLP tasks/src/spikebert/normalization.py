"""Configurable LayerNorm and Dynamic Tanh replacements."""
from __future__ import annotations
import torch
from torch import Tensor, nn

class DynamicTanh(nn.Module):
    """Dynamic Tanh: gamma * tanh(alpha * x) + beta."""
    def __init__(self, hidden_size: int, alpha_init: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, inputs: Tensor) -> Tensor:
        return self.weight * torch.tanh(self.alpha * inputs) + self.bias

class ShiftedDynamicTanh(nn.Module):
    """Pulse-oriented DyT with learnable channel shift before tanh.

    gamma * tanh(alpha * (x + shift)). There is no output bias: the learned
    shift changes the signal before the next SpikeLinear performs spiking.
    """
    def __init__(self, hidden_size: int, alpha_init: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.shift = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, inputs: Tensor) -> Tensor:
        return self.weight * torch.tanh(self.alpha * (inputs + self.shift))

NORMALIZATION_MODES = ("layernorm", "dyt", "shifted_dyt")

def build_normalization(config, hidden_size: int | None = None, eps: float | None = None) -> nn.Module:
    hidden_size = config.hidden_size if hidden_size is None else hidden_size
    mode = getattr(config, "normalization_mode", "layernorm")
    if mode == "layernorm":
        return nn.LayerNorm(hidden_size, eps=config.layer_norm_eps if eps is None else eps)
    alpha_init = float(getattr(config, "dyt_alpha_init", 0.5))
    if mode == "dyt":
        return DynamicTanh(hidden_size, alpha_init)
    if mode == "shifted_dyt":
        return ShiftedDynamicTanh(hidden_size, alpha_init)
    raise ValueError(f"unknown normalization_mode {mode!r}; expected one of {NORMALIZATION_MODES}")
