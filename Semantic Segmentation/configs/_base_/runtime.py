default_scope = "mmseg"
env_cfg = dict(cudnn_benchmark=True, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="nccl"))
default_hooks = dict(timer=dict(type="IterTimerHook"), logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"), checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=2500),
    sampler_seed=dict(type="DistSamplerSeedHook"), visualization=dict(type="SegVisualizationHook"))
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False
