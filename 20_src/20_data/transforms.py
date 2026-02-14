"""
Vesuvius Challenge - 3D 数据增强变换 (Phase 5)

用于 Patch-based 训练：对 (image, label) 配对同步执行变换。

组件:
- RandomCrop3D:      从大体积中随机裁剪固定大小的 3D patch
- RandomFlipRotate3D: 随机翻转（3 轴）+ 90° 旋转增强
- Compose3D:         组合多个变换
"""

import numpy as np
from typing import Tuple, List, Optional


class RandomCrop3D:
    """
    从 (image, label) 配对中随机裁剪 3D patch

    Args:
        crop_size: 裁剪尺寸，int 或 (D, H, W) tuple
    """

    def __init__(self, crop_size=64):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        else:
            self.crop_size = tuple(crop_size)

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: (D, H, W) float32
            label: (D, H, W) float32 或 uint8

        Returns:
            image_crop, label_crop: 裁剪后的配对
        """
        D, H, W = image.shape
        cD, cH, cW = self.crop_size

        # 确保体积足够大
        if D < cD or H < cH or W < cW:
            # 体积不够大，pad 到合适尺寸
            pad_d = max(cD - D, 0)
            pad_h = max(cH - H, 0)
            pad_w = max(cW - W, 0)
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            D, H, W = image.shape

        # 随机起始位置
        d0 = np.random.randint(0, D - cD + 1)
        h0 = np.random.randint(0, H - cH + 1)
        w0 = np.random.randint(0, W - cW + 1)

        image_crop = image[d0:d0+cD, h0:h0+cH, w0:w0+cW]
        label_crop = label[d0:d0+cD, h0:h0+cH, w0:w0+cW]

        return image_crop, label_crop


class RandomFlipRotate3D:
    """
    随机 3D 翻转 + 90° 旋转增强

    对 (image, label) 配对同步操作，保证空间一致性。

    Args:
        flip_prob: 每个轴翻转的概率，默认 0.5
        rotate_prob: 执行 90° 旋转的概率，默认 0.5
    """

    def __init__(self, flip_prob: float = 0.5, rotate_prob: float = 0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: (D, H, W)
            label: (D, H, W)

        Returns:
            image_aug, label_aug: 增强后的配对
        """
        # 确保 contiguous (避免负 stride 问题)
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        # 随机翻转 3 个轴
        for axis in range(3):
            if np.random.random() < self.flip_prob:
                image = np.flip(image, axis=axis)
                label = np.flip(label, axis=axis)

        # 随机 90° 旋转 (在 H-W 平面)
        if np.random.random() < self.rotate_prob:
            k = np.random.randint(1, 4)  # 旋转 90°, 180°, 270°
            image = np.rot90(image, k=k, axes=(1, 2))
            label = np.rot90(label, k=k, axes=(1, 2))

        # 确保 contiguous
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        return image, label


class Compose3D:
    """
    组合多个 (image, label) 配对变换

    Args:
        transforms: 变换列表，每个变换接受 (image, label) 返回 (image, label)
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


if __name__ == "__main__":
    print("=== 3D Transforms 自测 ===")

    # 合成数据
    image = np.random.rand(64, 128, 128).astype(np.float32)
    label = (image > 0.5).astype(np.float32)

    # 测试 RandomCrop3D
    crop = RandomCrop3D(32)
    img_c, lbl_c = crop(image, label)
    assert img_c.shape == (32, 32, 32), f"Crop 形状错误: {img_c.shape}"
    assert lbl_c.shape == (32, 32, 32), f"Label Crop 形状错误: {lbl_c.shape}"
    print(f"  RandomCrop3D: {image.shape} → {img_c.shape} ✓")

    # 测试 RandomFlipRotate3D
    aug = RandomFlipRotate3D()
    img_a, lbl_a = aug(img_c, lbl_c)
    assert img_a.shape == (32, 32, 32), f"Aug 形状错误: {img_a.shape}"
    print(f"  RandomFlipRotate3D: {img_c.shape} → {img_a.shape} ✓")

    # 测试 Compose3D
    pipeline = Compose3D([
        RandomCrop3D(32),
        RandomFlipRotate3D(),
    ])
    img_p, lbl_p = pipeline(image, label)
    assert img_p.shape == (32, 32, 32), f"Pipeline 形状错误: {img_p.shape}"
    print(f"  Compose3D: {image.shape} → {img_p.shape} ✓")

    # 测试小体积 padding
    small_img = np.random.rand(16, 16, 16).astype(np.float32)
    small_lbl = (small_img > 0.5).astype(np.float32)
    crop64 = RandomCrop3D(32)
    img_s, lbl_s = crop64(small_img, small_lbl)
    assert img_s.shape == (32, 32, 32), f"Small vol crop 错误: {img_s.shape}"
    print(f"  Small volume padding: (16,16,16) → {img_s.shape} ✓")

    print("✓ 所有测试通过！")
