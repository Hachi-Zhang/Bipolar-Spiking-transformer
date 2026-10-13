"""Attention quantizers shared by SSA and ShiftMax."""
from __future__ import annotations
import torch
from torch import Tensor, nn

class RoundClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, minimum: float, maximum: float) -> Tensor:
        if minimum >= maximum:
            raise ValueError("minimum must be smaller than maximum")
        ctx.minimum, ctx.maximum = minimum, maximum
        ctx.save_for_backward(inputs)
        return inputs.clamp(minimum, maximum).round()

    @staticmethod
    def backward(ctx, gradient: Tensor):
        (inputs,) = ctx.saved_tensors
        keep = (inputs >= ctx.minimum) & (inputs <= ctx.maximum)
        return gradient * keep.to(gradient.dtype), None, None

class SpikeQuantizer(nn.Module):
    def __init__(self, minimum: float, maximum: float, divisor: float) -> None:
        super().__init__()
        if divisor == 0:
            raise ValueError("divisor cannot be zero")
        self.minimum, self.maximum, self.divisor = minimum, maximum, divisor
    def forward(self, inputs: Tensor) -> Tensor:
        return RoundClampSTE.apply(inputs, self.minimum, self.maximum) / self.divisor

class SignedSSAQuantizer(SpikeQuantizer):
    """round(clamp(x, -4, 4)) / 4 (nine signed levels)."""
    def __init__(self) -> None:
        super().__init__(-4, 4, 4)

class MultiSpikeNorm(SpikeQuantizer):
    def __init__(self, levels: int) -> None:
        if levels not in (8, 16):
            raise ValueError("ShiftMax levels must be 8 or 16")
        super().__init__(0, levels, levels)
