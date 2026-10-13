import torch
from efsdt_seg.models.quantization import MultiSpikeNorm, SignedSSAQuantizer

def test_signed_ssa_levels_and_range():
    inputs = torch.tensor([-9.0, -3.6, -0.4, 0.4, 2.6, 9.0])
    output = SignedSSAQuantizer()(inputs)
    assert torch.equal(output, torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.75, 1.0]))
    assert output.min() >= -1 and output.max() <= 1
    assert torch.allclose(output * 4, torch.round(output * 4))

def test_ste_blocks_gradients_outside_range():
    inputs = torch.tensor([-5.0, -4.0, 0.0, 4.0, 5.0], requires_grad=True)
    SignedSSAQuantizer()(inputs).sum().backward()
    assert torch.equal(inputs.grad, torch.tensor([0.0, 0.25, 0.25, 0.25, 0.0]))

def test_nonnegative_quantizer():
    output = MultiSpikeNorm(4)(torch.tensor([-1.0, 0.6, 3.6, 8.0]))
    assert torch.equal(output, torch.tensor([0.0, 0.25, 1.0, 1.0]))
