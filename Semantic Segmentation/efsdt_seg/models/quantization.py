"""Straight-through quantizers used by the segmentation models."""
from __future__ import annotations
import torch
from torch import Tensor, nn

class RoundClampSTE(torch.autograd.Function):
    """Round/clamp forward; pass gradients only inside the clamp range."""
    @staticmethod
    def forward(ctx, inputs: Tensor, min_value: float, max_value: float) -> Tensor:
        if min_value >= max_value:
            raise ValueError("min_value must be smaller than max_value")
        ctx.min_value, ctx.max_value = min_value, max_value
        ctx.save_for_backward(inputs)
        return torch.round(torch.clamp(inputs, min=min_value, max=max_value))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (inputs,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(inputs < ctx.min_value) | (inputs > ctx.max_value)] = 0
        return grad_input, None, None

class SpikeQuantizer(nn.Module):
    """Compute ``round(clamp(x, min, max)) / divisor`` with an STE."""
    def __init__(self, min_value: float = 0, max_value: float = 4,
                 divisor: float | None = None) -> None:
        super().__init__()
        if min_value >= max_value:
            raise ValueError("min_value must be smaller than max_value")
        self.min_value, self.max_value = float(min_value), float(max_value)
        self.divisor = float(max_value if divisor is None else divisor)
        if self.divisor == 0:
            raise ValueError("divisor cannot be zero")

    def forward(self, inputs: Tensor) -> Tensor:
        return RoundClampSTE.apply(inputs, self.min_value, self.max_value) / self.divisor

    def extra_repr(self) -> str:
        return f"min_value={self.min_value:g}, max_value={self.max_value:g}, divisor={self.divisor:g}"

class MultiSpikeNorm(SpikeQuantizer):
    """Original non-negative multi-level spike quantizer."""
    def __init__(self, levels: int = 4) -> None:
        if levels <= 0:
            raise ValueError("levels must be positive")
        super().__init__(0, levels, levels)

class SignedSSAQuantizer(SpikeQuantizer):
    """Nine-level SSA quantizer with output values from -1 to 1."""
    def __init__(self) -> None:
        super().__init__(-4, 4, 4)

MultiSpike_norm = MultiSpikeNorm
