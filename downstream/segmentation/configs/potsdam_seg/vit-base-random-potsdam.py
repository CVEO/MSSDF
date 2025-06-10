_base_ = [
    "./schedule.py",
    "./dataset.py",
]
model = dict(
    backbone=dict(
        type="mmpretrain.VisionTransformer",
        arch="base",
        img_size=512,
        patch_size=16,
        out_indices=(2, 5, 8, 11),
        drop_path_rate=0.2,  # set to 0.2
        out_type="featmap",
        final_norm=False,
    ),
    neck=dict(
        type="MultiLevelNeck",
        in_channels=[768, 768, 768, 768],
        out_channels=768,
        scales=[4, 2, 1, 0.5],
    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
    ),
    auxiliary_head=dict(
        in_channels=768,
    ),
)

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    ),
]
