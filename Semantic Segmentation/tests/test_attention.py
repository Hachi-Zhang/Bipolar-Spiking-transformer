import pytest
import torch
from torch import nn
from efsdt_seg.models.attention import SegmentationAttention, shiftmax_exp2
from efsdt_seg.models.quantization import SignedSSAQuantizer

@pytest.mark.parametrize("mode,levels", [("ssa", 16), ("shiftmax_ln", 8), ("shiftmax_ln", 16)])
def test_attention_shape_and_finiteness(mode, levels):
    module = SegmentationAttention(dim=8, num_heads=2, value_ratio=2, mode=mode, shiftmax_levels=levels).eval()
    output = module(torch.randn(2, 8, 4, 4))
    assert output.shape == (2, 8, 4, 4)
    assert torch.isfinite(output).all()

def test_ssa_quantizes_qkv_and_output_signed():
    module = SegmentationAttention(8, 2, mode="ssa")
    assert isinstance(module.q_quant, SignedSSAQuantizer)
    assert isinstance(module.k_quant, SignedSSAQuantizer)
    assert isinstance(module.v_quant, SignedSSAQuantizer)
    assert isinstance(module.output_quant, SignedSSAQuantizer)

def test_shiftmax_uses_layernorm():
    module = SegmentationAttention(8, 2, mode="shiftmax_ln", shiftmax_levels=8)
    assert isinstance(module.q_norm, nn.LayerNorm)
    assert isinstance(module.k_norm, nn.LayerNorm)
    assert isinstance(module.v_norm, nn.LayerNorm)

def test_shiftmax_has_unit_row_maximum():
    output = shiftmax_exp2(torch.randn(2, 3, 5))
    assert torch.allclose(output.amax(dim=-1), torch.ones(2, 3))
