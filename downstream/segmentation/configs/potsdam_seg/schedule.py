_base_ = [
    "../_base_/default_runtime.py",
]
optimizer = dict(type="AdamW", lr=0.001, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            ".ln": dict(decay_mult=0.0),
            ".bias": dict(decay_mult=0.0),
            ".cls_token": dict(decay_mult=0.0),
            ".pos_embed": dict(decay_mult=0.0),
            "backbone.": dict(lr_mult=0.1),
        },
    ),
)

# learning rate scheduler
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=80000, val_interval=8000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=8000, max_keep_ckpts=1
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
