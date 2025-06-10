# dataset settings
dataset_type = "NYUDataset"
data_root = "/scratch/wangtong/PotsdamClip50/"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadDepthAnnotation", depth_rescale_factor=1.0 / 255.0),
    dict(type="RandomDepthMix", prob=0.25),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomResize", scale=(768, 512), ratio_range=(0.8, 1.5), keep_ratio=True
    ),
    dict(type="RandomCrop", crop_size=(512, 512)),
    dict(
        type="Albu",
        transforms=[
            dict(type="RandomBrightnessContrast"),
            dict(type="RandomGamma"),
            dict(type="HueSaturationValue"),
        ],
        keymap={"img": "image", "gt_depth_map": "mask"},
    ),
    dict(
        type="PackSegInputs",
        meta_keys=(
            "img_path",
            "depth_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadDepthAnnotation", depth_rescale_factor=1.0 / 255.0),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(
        type="PackSegInputs",
        meta_keys=(
            "img_path",
            "depth_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=".png",
        data_prefix=dict(img_path="img_dir/train", depth_map_path="dep_dir/train"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        img_suffix=".png",
        data_prefix=dict(img_path="img_dir/val", depth_map_path="dep_dir/val"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="DepthMetric", min_depth_eval=0.001, max_depth_eval=1.0, crop_type="nyu_crop"
)
test_evaluator = val_evaluator

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    size=(512, 512),
    seg_pad_val=0,
)
model = dict(
    type="DepthEstimator",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(
        type="VPDDepthHead",
        max_depth=1,
        fmap_border=(1, 1),
    ),
    test_cfg=dict(mode="whole"),
)
