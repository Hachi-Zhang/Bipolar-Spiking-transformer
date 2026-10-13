import importlib.util
import pytest
import torch

HAS_MMLAB = all(importlib.util.find_spec(name) for name in ("mmcv", "mmengine", "mmseg"))
pytestmark = pytest.mark.skipif(not HAS_MMLAB, reason="OpenMMLab dependencies are not installed")

def test_backbone_neck_head_and_encoder_decoder_build():
    import efsdt_seg  # noqa: F401 - triggers registry imports
    from mmseg.registry import MODELS
    backbone_cfg = dict(type="Spiking_vit_MetaFormerv2", embed_dim=[16, 32, 64, 96],
                        num_heads=8, depths=8, init_cfg=None)
    neck_cfg = dict(type="QFPN", in_channels=[8, 16, 32, 96], out_channels=32, num_outs=4)
    head_cfg = dict(type="QFPNHead", in_channels=[32] * 4, in_index=[0, 1, 2, 3],
                    feature_strides=[2, 4, 8, 16], channels=32, num_classes=3,
                    norm_cfg=dict(type="BN"), align_corners=False,
                    loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0))
    backbone = MODELS.build(backbone_cfg).eval()
    neck = MODELS.build(neck_cfg).eval()
    head = MODELS.build(head_cfg).eval()
    with torch.no_grad():
        features = backbone(torch.randn(1, 3, 64, 64))
        assert [tuple(x.shape) for x in features] == [(1, 8, 32, 32), (1, 16, 16, 16), (1, 32, 8, 8), (1, 96, 4, 4)]
        pyramid = neck(features)
        logits = head(pyramid)
        assert logits.shape == (1, 3, 32, 32)
    segmentor = MODELS.build(dict(type="EncoderDecoder", backbone=backbone_cfg,
        neck=neck_cfg, decode_head=head_cfg, train_cfg=dict(), test_cfg=dict(mode="whole")))
    assert segmentor.backbone.__class__.__name__ == "Spiking_vit_MetaFormerv2"
    assert segmentor.neck.__class__.__name__ == "QFPN"
    assert segmentor.decode_head.__class__.__name__ == "QFPNHead"
