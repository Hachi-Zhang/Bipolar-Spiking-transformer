"""Attention helpers independent from the BERT wrapper."""
import torch
from torch import Tensor

def shiftmax_exp2(scores: Tensor) -> Tensor:
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    return torch.exp2(shifted.clamp(-60.0, 60.0))

def valid_key_mask(additive_mask: Tensor | None, scores: Tensor) -> Tensor | None:
    """Convert Hugging Face additive mask to a key-valid boolean mask."""
    if additive_mask is None:
        return None
    mask = additive_mask
    while mask.ndim < scores.ndim:
        mask = mask.unsqueeze(0)
    return mask >= 0
