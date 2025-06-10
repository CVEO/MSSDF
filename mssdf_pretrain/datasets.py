# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.pytorch.transforms as T
import numpy as np
from osgeo import gdal
from masking_generator import InfoAwareMaskingGenerator, CrossModalMasking

MAX_PIXELS = 512 * 512


def get_patches_info_from_one_file(data: dict, meta=False):
    img_path = data["img_file"]
    patch_size = data["patch_size"]
    stride = data["stride"]
    # print(img_path)
    src_ds = gdal.Open(img_path)
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize

    if width <= patch_size and height <= patch_size:
        return [
            data,
        ]
    number_of_patch_x = (width - patch_size) // stride + 1
    stride_new_x = (width - patch_size) // number_of_patch_x
    width_list = [item * stride_new_x for item in range(number_of_patch_x)]
    number_of_patch_y = (height - patch_size) // stride + 1
    stride_new_y = (height - patch_size) // number_of_patch_y
    height_list = [item * stride_new_y for item in range(number_of_patch_y)]
    patch_info = []
    for y_off in height_list:
        for x_off in width_list:
            patch_info.append(
                dict(
                    x=x_off,
                    y=y_off,
                    patch=patch_size,
                    stride_x=stride_new_x,
                    stride_y=stride_new_y,
                    **data,
                )
            )
    for x_off in width_list:
        y_off = height - patch_size
        patch_info.append(
            dict(
                x=x_off,
                y=y_off,
                patch=patch_size,
                stride_x=stride_new_x,
                stride_y=stride_new_y,
                **data,
            )
        )

    for y_off in height_list:
        x_off = width - patch_size
        patch_info.append(
            dict(
                x=x_off,
                y=y_off,
                patch=patch_size,
                stride_x=stride_new_x,
                stride_y=stride_new_y,
                **data,
            )
        )

    patch_info.append(
        dict(
            x=width - patch_size,
            y=height - patch_size,
            patch=patch_size,
            stride_x=stride_new_x,
            stride_y=stride_new_y,
            **data,
        )
    )
    return patch_info


class DomDsmDataset(Dataset):
    def __init__(
        self,
        root,
        img_size=224,
        cmap="gray",
        mask_ratio=0.75,
        patch_size=16,
    ):
        self.root = root
        self.transform = A.Compose(
            [
                A.Flip(p=0.5),
                A.ColorJitter(p=0.5),
                A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.5), p=0.5),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                T.ToTensorV2(),
            ],
            additional_targets={"dsm": "image"},
        )
        domlist = [
            fn
            for fn in os.listdir(os.path.join(root, "patch"))
            if fn.endswith("_dom.tif")
        ]
        checked_dsm_list = []
        checked_dom_list = []
        for fn in domlist:
            dsm_fn = fn.replace("_dom.tif", "_dsm.tif")
            if os.path.exists(os.path.join(root, "patch", dsm_fn)):
                checked_dsm_list.append(os.path.join(root, "patch", dsm_fn))
                checked_dom_list.append(os.path.join(root, "patch", fn))
        self.domlist = checked_dom_list
        self.dsmlist = checked_dsm_list

        self.patches = get_patches_info_from_one_file(
            {
                "img_file": self.domlist[0],
                "patch_size": img_size,
                "stride": img_size,
            }
        )

        self.mask_generator = InfoAwareMaskingGenerator(
            input_size=img_size, mask_ratio=mask_ratio, patch_size=patch_size
        )
        self.cross_mask = CrossModalMasking()
        self.p = 0.1
        self.cmap = plt.get_cmap(cmap)

    def __len__(self):
        return len(self.domlist) * len(self.patches)

    def set_p(self, p):
        self.p = p

    def read_patch(self, patch_info, img_file):
        src_ds = gdal.Open(img_file)
        x = patch_info["x"]
        y = patch_info["y"]
        patch_size = patch_info["patch"]
        img = src_ds.ReadAsArray(x, y, patch_size, patch_size)
        return img

    def __getitem__(self, idx):
        image_i = idx // len(self.patches)
        patch_i = idx % len(self.patches)

        dom = self.read_patch(self.patches[patch_i], self.domlist[image_i])
        mask = np.sum(dom == 0, axis=0) == 3

        if np.sum(mask) / mask.shape[0] / mask.shape[1] > 0.5:
            return self.__getitem__(
                np.random.randint(0, len(self.domlist) * len(self.patches))
            )
        dom = np.transpose(dom, (1, 2, 0))

        dsm = self.read_patch(self.patches[patch_i], self.dsmlist[image_i])
        dsm = self.cmap(dsm, bytes=True)[..., :3]

        if self.transform:
            data = self.transform(**{"image": dom, "dsm": dsm})
            dom, dsm = data["image"], data["dsm"]

        mask1, mask2 = self.mask_generator(dom, dsm)
        mask1, mask2 = self.cross_mask(mask1, mask2, self.p)

        return (dom, dsm, mask1, mask2)


def build_pretraining_dataset_mm(args):
    return DomDsmDataset(
        args.data_path, args.input_size, args.cmap, args.mask_ratio, args.window_size
    )
