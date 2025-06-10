_base_ = ["./vit-base-random-potsdam.py"]


model = dict(
    backbone=dict(
        frozen_stages=-1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="../checkpoints/vit_base_500.pth",
            prefix=None,
        ),
    ),
)
