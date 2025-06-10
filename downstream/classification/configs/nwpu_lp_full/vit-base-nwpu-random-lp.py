_base_ = [
    "./dataset.py",
    "../_base_/default_runtime.py",
]


train_dataloader = dict(
    batch_size=32,
)
val_dataloader = dict(
    batch_size=32,
)
test_dataloader = val_dataloader

# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="VisionTransformer",
        arch="base",
        img_size=224,
        patch_size=16,
        drop_path_rate=0.2,  # set to 0.2
        out_type="cls_token",
        final_norm=False,
    ),
    neck=None,
    head=dict(
        type="LinearClsHead",
        num_classes=30,
        in_channels=768,
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, mode="original"),
        init_cfg=[dict(type="TruncNormal", layer="Linear", std=2e-5)],
    ),
    train_cfg=dict(
        augments=[dict(type="Mixup", alpha=0.8), dict(type="CutMix", alpha=1.0)]
    ),
    data_preprocessor=dict(
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True,
        num_classes=30,
    ),
)

# optimizer wrapper
# learning rate and layer decay rate are set to 0.001 and 0.75 respectively
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
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True,
    ),
]

# runtime settings
val_cfg = dict()
test_cfg = dict()
train_cfg = dict(by_epoch=True, max_epochs=100)
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type="CheckpointHook", interval=2, max_keep_ckpts=1)
)

randomness = dict(seed=0, diff_rank_seed=True)
