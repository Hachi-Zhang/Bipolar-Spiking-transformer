"""Shift-DyT conversion for post-training QKformer fine-tuning."""

import torch
from torch import nn


class TanhShiftWeighted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        abs_x = torch.abs(x)
        floor_x = torch.floor(abs_x)
        frac = abs_x - floor_x
        power_floor = torch.pow(2.0, -torch.clamp(floor_x, 0, 15))
        power_ceil = torch.pow(2.0, -torch.clamp(torch.ceil(abs_x), 0, 15))
        power_weighted = (1.0 - frac) * power_floor + frac * power_ceil
        large_signal = torch.sign(x) * (1.0 - power_weighted)
        return torch.where(abs_x < 1.0, 0.5 * x, large_signal)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        abs_x = torch.abs(x)
        n = torch.clamp(torch.floor(abs_x), 0, 15)
        large_grad = torch.pow(2.0, -(n + 1.0))
        grad = torch.where(abs_x < 1.0, torch.full_like(x, 0.5), large_grad)
        return grad_output * grad


class ShiftDyT(nn.Module):
    """Per-channel Shift-DyT for attention features shaped [N, C, L]."""

    def __init__(self, channels, alpha_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((channels,), float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.alpha.numel():
            raise ValueError("ShiftDyT expects [batch, channels, length]")
        shape = (1, -1, 1)
        return self.weight.view(shape) * TanhShiftWeighted.apply(
            self.alpha.view(shape) * x
        ) + self.bias.view(shape)


def convert_attention_norms(model, alpha_init=1.0):
    """Replace attention Q/K/V LayerNorm after loading the trained weights."""
    count = 0
    for parent in model.modules():
        for name in ("q_bn", "k_bn", "v_bn"):
            norm = getattr(parent, name, None)
            if not isinstance(norm, nn.LayerNorm) or not getattr(norm, "channels_first", False):
                continue
            replacement = ShiftDyT(norm.normalized_shape[0], alpha_init)
            replacement.to(device=norm.weight.device, dtype=norm.weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(norm.weight)
                replacement.bias.copy_(norm.bias)
            replacement.train(norm.training)
            setattr(parent, name, replacement)
            count += 1
    if count == 0:
        raise ValueError("No attention Q/K/V LayerNorm layers found")
    return count
