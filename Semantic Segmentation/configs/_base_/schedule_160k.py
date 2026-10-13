optim_wrapper = dict(type="AmpOptimWrapper", loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=0.001, betas=(0.9, 0.999), weight_decay=0.005))
param_scheduler = [dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
                   dict(type="PolyLR", eta_min=0.0, power=1.0, begin=1500, end=160000, by_epoch=False)]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=160000, val_interval=2500)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
