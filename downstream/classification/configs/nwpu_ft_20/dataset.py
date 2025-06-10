data_preprocessor = dict(
    num_classes=45,
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RandomResizedCrop", scale=224, backend="pillow", interpolation="bicubic"
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(
        type="RandAugment",
        policies="timm_increasing",
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation="bicubic"),
    ),
    dict(
        type="RandomErasing",
        erase_prob=0.25,
        mode="rand",
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395],
    ),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Resize", scale=(224, 224), interpolation="bicubic", backend="pillow"),
    dict(type="PackInputs"),
]
data_root = "/scratch/wangtong/NWPU-RESISC45"
dataset = dict(
    type="CustomDataset",
    with_label=True,
    classes=[
        "airplane",
        "airport",
        "baseball_diamond",
        "basketball_court",
        "beach",
        "bridge",
        "chaparral",
        "church",
        "circular_farmland",
        "cloud",
        "commercial_area",
        "dense_residential",
        "desert",
        "forest",
        "freeway",
        "golf_course",
        "ground_track_field",
        "harbor",
        "industrial_area",
        "intersection",
        "island",
        "lake",
        "meadow",
        "medium_residential",
        "mobile_home_park",
        "mountain",
        "overpass",
        "palace",
        "parking_lot",
        "railway",
        "railway_station",
        "rectangular_farmland",
        "river",
        "roundabout",
        "runway",
        "sea_ice",
        "ship",
        "snowberg",
        "sparse_residential",
        "stadium",
        "storage_tank",
        "tennis_court",
        "terrace",
        "thermal_power_station",
        "wetland",
    ],
)

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        **dataset,
        data_root=data_root + "/train/train",
        data_prefix="",
        ann_file=data_root + "/train20.txt",
        pipeline=train_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        **dataset,
        data_root=data_root + "/test/test",
        data_prefix="",
        ann_file=data_root + "/test.txt",
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
    persistent_workers=True,
)

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        **dataset,
        data_root=data_root + "/test/test",
        data_prefix="",
        ann_file=data_root + "/test.txt",
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
    persistent_workers=True,
)

val_evaluator = [
    dict(type="Accuracy", topk=(1, 5)),
    dict(type="ConfusionMatrix", num_classes=45),
    dict(type="SingleLabelMetric", num_classes=45),
]

test_evaluator = val_evaluator
