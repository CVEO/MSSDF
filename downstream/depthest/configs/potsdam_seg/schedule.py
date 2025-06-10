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
    dict(
        type="LinearLR",
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=25,
        by_epoch=True,
        begin=5,
        end=30,
        eta_min=1e-6,
        convert_to_iter_based=True,
    ),
]

# training schedule for 80k
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        save_best="rmse",
        rule="less",
        max_keep_ckpts=1,
    ),
    # checkpoint=dict(save_best='rmse', rule='less', max_keep_ckpts=1)
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
