import numpy as np
import cv2
from typing import Tuple, Optional
import torch


def compute_gradient_magnitude(patch: np.ndarray) -> float:
    """计算patch的梯度幅值"""
    if patch.ndim == 3:
        # 如果是多通道图像，转成灰度图处理
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(grad_x, grad_y)
    return np.mean(grad_mag)


def compute_local_variance(patch: np.ndarray) -> float:
    """计算patch的局部方差"""
    mean = np.mean(patch)
    return np.mean((patch - mean) ** 2)


def compute_information_score(
    patch_rgb: np.ndarray, patch_other: np.ndarray = None
) -> float:
    """
    计算patch的信息得分 S(p) = S_grad + S_var
    如果有第二个模态，则融合两个模态的信息
    """
    score_rgb_grad = compute_gradient_magnitude(patch_rgb)
    score_rgb_var = compute_local_variance(patch_rgb)
    score_rgb = score_rgb_grad + score_rgb_var

    if patch_other is not None:
        score_other_grad = compute_gradient_magnitude(patch_other)
        score_other_var = compute_local_variance(patch_other)
        score_other = score_other_grad + score_other_var
        return score_rgb + score_other
    else:
        return score_rgb


class InfoAwareMaskingGenerator:
    def __init__(self, input_size, mask_ratio, patch_size=16):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.patch_size = patch_size
        self.num_patches = (self.height // patch_size) * (self.width // patch_size)
        self.mask_ratio = mask_ratio
        self.num_mask = int(self.num_patches * self.mask_ratio)

    def __call__(
        self, image_rgb: np.ndarray, image_other: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        输入：
            image_rgb: [H, W, C] RGB图像
            image_other: [H, W, C] 其他模态图像（可选）

        输出：
            mask: [num_patches]
        """
        if isinstance(image_rgb, torch.Tensor):
            image_rgb = image_rgb.cpu().numpy()
            image_rgb = (image_rgb * 255).astype(np.uint8)
            image_rgb = image_rgb.transpose(1, 2, 0)
        if image_other is not None and isinstance(image_other, torch.Tensor):
            image_other = image_other.cpu().numpy()
            image_other = (image_other * 255).astype(np.uint8)
            image_other = image_other.transpose(1, 2, 0)
        h_patches = self.height // self.patch_size
        w_patches = self.width // self.patch_size

        info_scores = []

        for i in range(h_patches):
            for j in range(w_patches):
                y_start = i * self.patch_size
                x_start = j * self.patch_size
                patch_rgb = image_rgb[
                    y_start : y_start + self.patch_size,
                    x_start : x_start + self.patch_size,
                ]

                if image_other is not None:
                    patch_other = image_other[
                        y_start : y_start + self.patch_size,
                        x_start : x_start + self.patch_size,
                    ]
                else:
                    patch_other = None

                score = self.compute_information_score(patch_rgb, patch_other)
                info_scores.append(score)

        info_scores = np.array(info_scores)

        # 计算 Q20 和 Q80
        q20 = np.percentile(info_scores, 20)
        q80 = np.percentile(info_scores, 80)

        # 分配掩码概率
        p_mask = np.zeros_like(info_scores)
        p_mask[info_scores < q20] = 0.8
        p_mask[info_scores > q80] = 0.3
        p_mask[(info_scores >= q20) & (info_scores <= q80)] = 0.5

        # 归一化概率用于采样
        p_mask /= p_mask.sum()

        # 固定数量加权采样
        ids = np.random.choice(
            a=np.arange(self.num_patches), size=self.num_mask, replace=False, p=p_mask
        )

        mask = np.zeros(self.num_patches, dtype=np.uint8)
        mask[ids] = 1

        return mask

    def compute_information_score(
        self, patch_rgb: np.ndarray, patch_other: Optional[np.ndarray]
    ) -> float:
        score_rgb_grad = compute_gradient_magnitude(patch_rgb)
        score_rgb_var = compute_local_variance(patch_rgb)
        score_rgb = score_rgb_grad + score_rgb_var

        if patch_other is not None:
            score_other_grad = compute_gradient_magnitude(patch_other)
            score_other_var = compute_local_variance(patch_other)
            score_other = score_other_grad + score_other_var
            return score_rgb + score_other
        else:
            return score_rgb


class CrossModalMasking:
    def __call__(
        self, mask_mod1: np.ndarray, mask_mod2: np.ndarray, substitution_prob=0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        输入：
            mask_mod1: 第一模态的mask [num_patches]
            mask_mod2: 第二模态的mask [num_patches]

        输出：
            new_mask_mod2: 替换后的第二模态mask
        """
        new_mask_mod2 = mask_mod2.copy()

        # 找出mod2中被mask的位置
        masked_indices = np.where(mask_mod2 == 1)[0]

        for idx in range(len(new_mask_mod2)):
            if np.random.rand() < substitution_prob:
                # 随机从mod2的masked patches中采样一个位置替换到当前idx
                sampled_idx = np.random.choice(masked_indices)
                new_mask_mod2[idx] = mask_mod2[sampled_idx]

        return mask_mod1, new_mask_mod2
