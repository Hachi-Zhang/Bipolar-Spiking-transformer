"""Switchable LayerNorm and Shift-DyT for SDT-V3 attention."""

import torch
import torch.nn as nn


class TanhShiftWeighted(torch.autograd.Function):
    """Shift-DyT nonlinearity with a piecewise surrogate gradient."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        abs_x = torch.abs(x)
        floor_x = torch.floor(abs_x)
        frac = abs_x - floor_x
        small_signal = 0.5 * x
        power_floor = torch.pow(2.0, -torch.clamp(floor_x, 0, 15))
        power_ceil = torch.pow(2.0, -torch.clamp(torch.ceil(abs_x), 0, 15))
        power_weighted = (1.0 - frac) * power_floor + frac * power_ceil
        large_signal = torch.sign(x) * (1.0 - power_weighted)
        return torch.where(abs_x < 1.0, small_signal, large_signal)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        abs_x = torch.abs(x)
        n = torch.clamp(torch.floor(abs_x), 0, 15)
        large_grad = torch.pow(2.0, -(n + 1.0))
        grad = torch.where(abs_x < 1.0, torch.full_like(x, 0.5), large_grad)
        return grad_output * grad


class LayerNormShiftDyT(nn.LayerNorm):
    """Share affine parameters between LayerNorm and Shift-DyT.

    In auto mode, epochs before norm_switch_epoch use LayerNorm; later epochs
    use weight * TanhShiftWeighted(alpha * x) + bias. The schedule is saved in
    the state_dict. DDP requires find_unused_parameters=True in the LN phase.
    """

    def __init__(
            self, normalized_shape, eps=1e-5, norm_mode="auto",
            norm_switch_epoch=100, alpha_init=1.0,
    ):
        super().__init__(normalized_shape, eps=eps)
        if not isinstance(norm_switch_epoch, int) or norm_switch_epoch < 0:
            raise ValueError("norm_switch_epoch must be a nonnegative integer")
        self.norm_switch_epoch = norm_switch_epoch
        self.alpha = nn.Parameter(torch.full(self.normalized_shape, float(alpha_init)))
        self.register_buffer(
            "_norm_schedule", torch.tensor([0, norm_switch_epoch, 0], dtype=torch.long)
        )
        self.current_epoch = 0
        self.set_mode(norm_mode)

    def set_mode(self, norm_mode):
        if norm_mode not in ("auto", "layernorm", "shiftdyt"):
            raise ValueError("norm_mode must be 'auto', 'layernorm', or 'shiftdyt'")
        self.norm_mode = norm_mode
        self._norm_schedule[0] = ("auto", "layernorm", "shiftdyt").index(norm_mode)
        self._update_branch()

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative, zero-based integer")
        self.current_epoch = epoch
        self._norm_schedule[2] = epoch
        self._update_branch()

    def _update_branch(self):
        self.use_shift_dyt = self.norm_mode == "shiftdyt" or (
            self.norm_mode == "auto" and self.current_epoch >= self.norm_switch_epoch
        )

    def forward(self, x):
        if not self.use_shift_dyt:
            return super().forward(x)
        if tuple(x.shape[-len(self.normalized_shape):]) != self.normalized_shape:
            raise RuntimeError(
                f"Expected trailing dimensions {self.normalized_shape}, got {tuple(x.shape)}"
            )
        return self.weight * TanhShiftWeighted.apply(self.alpha * x) + self.bias

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        if prefix + "_norm_schedule" in state_dict:
            mode_index, switch_epoch, epoch = self._norm_schedule.tolist()
            if mode_index not in (0, 1, 2) or switch_epoch < 0 or epoch < 0:
                raise ValueError("Invalid normalization schedule in checkpoint")
            self.norm_switch_epoch = switch_epoch
            self.set_mode(("auto", "layernorm", "shiftdyt")[mode_index])
            self.set_epoch(epoch)

    def extra_repr(self):
        active_norm = "shiftdyt" if self.use_shift_dyt else "layernorm"
        return (
            f"{super().extra_repr()}, norm_mode={self.norm_mode}, "
            f"norm_switch_epoch={self.norm_switch_epoch}, active_norm={active_norm}"
        )


def set_norm_epoch(model, epoch):
    """Update all switchable norms; works with plain models and DDP wrappers."""
    for module in model.modules():
        if isinstance(module, LayerNormShiftDyT):
            module.set_epoch(epoch)


def set_norm_mode(model, norm_mode):
    """Select auto, layernorm, or shiftdyt for every switchable norm."""
    for module in model.modules():
        if isinstance(module, LayerNormShiftDyT):
            module.set_mode(norm_mode)
