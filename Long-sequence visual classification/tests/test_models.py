import pytest
import torch
from torch import nn
from seq_cifar.models import (
    MODEL_NAMES, ConvMLP, HierarchicalModel, SDTModel, SDSA,
    SpikformerModel, TemporalMLP, TokenMajorSSA, build_model,
)

TINY = dict(step=2, in_channels=3, num_classes=10, embed_dim=48,
            num_heads=4, mlp_ratio=2, attn_scale=.125, depths=3,
            sequence_length=1024)

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_tiny_forward_backward(name):
    model = build_model({"model": name, **TINY})
    output = model(torch.randn(1, 3, 1024))
    assert output.shape == (1, 10)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_main_and_qk_structure():
    main = build_model({"model": "TIMv2_e3_seq", **TINY})
    qk = build_model({"model": "TIMv2_bl_qk_seq", **TINY})
    assert isinstance(main, HierarchicalModel)
    assert isinstance(main.b1.mlp, TemporalMLP)
    assert isinstance(main.b2.mlp, TemporalMLP)
    assert isinstance(qk.b1.mlp, ConvMLP)
    assert isinstance(qk.b2.mlp, ConvMLP)
    assert all(isinstance(block.attn.tm, nn.Identity) for block in main.blocks)


def test_distinct_baseline_paths():
    sdt = build_model({"model": "TIMv2_bl_sdt_seq", **TINY})
    spikf = build_model({"model": "TIMv2_bl_spikf_seq", **TINY})
    assert isinstance(sdt, SDTModel)
    assert isinstance(spikf, SpikformerModel)
    assert isinstance(sdt.blocks[0].attn, SDSA)
    assert isinstance(spikf.blocks[0].attn, TokenMajorSSA)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_eval_has_no_cross_batch_state_leak(name):
    model=build_model({"model":name,**TINY}).eval()
    x=torch.randn(1,3,1024)
    with torch.no_grad():
        first=model(x); second=model(x)
    assert torch.equal(first,second)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_rejects_wrong_input_shape(name):
    model=build_model({"model":name,**TINY})
    with pytest.raises(ValueError,match="expected"):
        model(torch.randn(1,1,1024))
    with pytest.raises(ValueError,match="expected"):
        model(torch.randn(1,3,128))


def test_unknown():
    with pytest.raises(ValueError):
        build_model({"model": "nope"})
