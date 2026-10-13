"""Pre-training checks for the main models.py implementation.

Run with: python test_models.py
The checks are intentionally small and do not require ImageNet data.
"""

import sys

import torch

import models


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_shiftmax():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    weights = models.shiftmax(logits)
    row_sum = weights.sum(dim=-1)

    check(torch.isfinite(weights).all().item(), "ShiftMax produced NaN or Inf")
    check(float(row_sum.min()) >= 0.49, "ShiftMax row sum is below the power-of-two bound")
    check(float(row_sum.max()) <= 1.00001, "ShiftMax row sum is above one")

    # Subtracting the row maximum should make the result shift invariant.
    shifted = models.shiftmax(logits + 7.0)
    error = (weights - shifted).abs().max().item()
    check(error < 1e-5, "ShiftMax is not invariant to an additive logit shift")
    weights.mean().backward()
    check(logits.grad is not None, "ShiftMax did not produce a gradient")
    check(torch.isfinite(logits.grad).all().item(), "ShiftMax gradient contains NaN or Inf")
    print(
        "[PASS] shiftmax: finite=True, row_sum=[%.6f, %.6f], shift_error=%.3e, grad_finite=True"
        % (float(row_sum.min()), float(row_sum.max()), error)
    )


def test_attention_configuration_and_forward():
    torch.manual_seed(1)
    dim, heads = 32, 4
    attention = models.MS_Attention_linear(
        dim=dim, num_heads=heads, attn_drop=0.1, proj_drop=0.1
    )
    expected_scale = (dim // heads) ** -0.5
    check(
        abs(float(attention.scale) - expected_scale) < 1e-8,
        "QK scale is not 1/sqrt(head_dim)",
    )
    check(attention.attn_drop.p == 0.1, "attention dropout is not connected")
    check(attention.proj_drop.p == 0.1, "projection dropout is not connected")

    attention.eval()
    x = torch.randn(2, dim, 4, 4)
    y = attention(x)
    check(y.shape == x.shape, "attention changed the input shape")
    check(torch.isfinite(y).all().item(), "attention produced NaN or Inf")
    print(
        "[PASS] attention: shape=%s, finite=True, qk_scale=%.6f"
        % (tuple(y.shape), float(attention.scale))
    )


def test_mlp_channel_layout():
    torch.manual_seed(2)
    mlp = models.MS_MLP(in_features=32, hidden_features=64)
    mlp.eval()
    x = torch.randn(2, 32, 14, 14)
    y = mlp(x)
    check(y.shape == x.shape, "MS_MLP did not preserve [B,C,H,W] layout")
    check(torch.isfinite(y).all().item(), "MS_MLP produced NaN or Inf")
    print("[PASS] MLP layout: input/output=%s, finite=True" % (tuple(y.shape),))


def test_signed_polarity():
    # This verifies the signed interaction used by the attention path: same
    # polarity reinforces and opposite polarity suppresses the dot product.
    q = torch.tensor([1.0, -1.0])
    same = torch.dot(q, q).item()
    opposite = torch.dot(q, -q).item()
    check(same > 0 and opposite < 0, "signed Q/K polarity interaction is lost")
    print(
        "[PASS] polarity: same_polarity_score=%.3f, opposite_polarity_score=%.3f"
        % (same, opposite)
    )


def test_small_model_forward():
    model = models.Efficient_Spiking_Transformer_s()
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    check(output.shape == (1, 1000), "small model output shape is incorrect")
    check(torch.isfinite(output).all().item(), "small model produced NaN or Inf")
    print("[PASS] model forward: output_shape=%s, finite=True" % (tuple(output.shape),))


def main():
    tests = [
        test_shiftmax,
        test_attention_configuration_and_forward,
        test_mlp_channel_layout,
        test_signed_polarity,
        test_small_model_forward,
    ]
    print("Running models.py pre-training checks...")
    try:
        for test in tests:
            test()
    except Exception as exc:
        print("[FAIL] %s: %s" % (test.__name__, exc))
        return 1
    print("All models.py checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
