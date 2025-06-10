_base_ = ["./vit-nwpu-random-ft.py"]


# model settings
model = dict(
    backbone=dict(
        frozen_stages=12,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="../checkpoints/vit_base_500.pth",
            prefix=None,
        ),
    ),
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999)),
    constructor="LearningRateDecayOptimWrapperConstructor",
    paramwise_cfg=dict(
        layer_decay_rate=0.75,
        custom_keys={
            ".ln": dict(decay_mult=0.0),
            ".bias": dict(decay_mult=0.0),
            ".cls_token": dict(decay_mult=0.0),
            ".pos_embed": dict(decay_mult=0.0),
            "backbone.": dict(lr_mult=0.1),
        },
    ),
)
