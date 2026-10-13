_base_ = ["../_base_/ade20k.py", "../_base_/runtime.py", "../_base_/schedule_160k.py"]
custom_imports = dict(imports=["efsdt_seg"], allow_failed_imports=False)
crop_size = (512, 512)
data_preprocessor = dict(type="SegDataPreProcessor", mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=crop_size)
model = dict(type="EncoderDecoder", data_preprocessor=data_preprocessor,
    backbone=dict(type="Spiking_vit_MetaFormerv2", init_cfg=None, in_channels=3,
                  embed_dim=[32, 64, 128, 192], num_heads=8, mlp_ratios=4, depths=8,
                  T=4, decode_mode="QTrick", attention_mode="ssa", attention_max_value=16),
    neck=dict(type="QFPN", in_channels=[16, 32, 64, 192], out_channels=128, num_outs=4, T=4),
    decode_head=dict(type="QFPNHead", in_channels=[128, 128, 128, 128], in_index=[0, 1, 2, 3],
        feature_strides=[2, 4, 8, 16], channels=128, dropout_ratio=0.1, num_classes=150, T=4,
        norm_cfg=dict(type="SyncBN", requires_grad=True), align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(), test_cfg=dict(mode="whole"))
