"""
Vesuvius Challenge - 3D TIF Chunk 数据加载器

包含两个 Dataset：
- TifChunkDataset:       推理用，整体加载
- VesuviusTrainDataset:  训练用，NPY mmap 零拷贝 / TIF LRU 缓存

核心优化（NPY 模式）：
  np.load(mmap_mode='r') 将文件映射到虚拟内存，
  只有在 slice 时才触发缺页中断读取对应的字节。
  96³ uint8 patch ≈ 0.88MB，IO 时间 < 1ms。
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Union
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile


class TifChunkDataset(Dataset):
    """
    3D TIF Chunk 数据集（推理用）

    从指定目录中扫描所有 .tif/.tiff 文件，逐个整体加载为 3D tensor。
    """

    def __init__(
        self,
        data_source: Union[str, Path, List[str]],
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.transform = transform
        self.normalize = normalize

        if isinstance(data_source, (str, Path)):
            data_source = Path(data_source)
            if data_source.is_dir():
                self.file_paths = sorted([
                    str(p) for p in data_source.iterdir()
                    if p.suffix.lower() in ('.tif', '.tiff')
                ])
            elif data_source.is_file():
                self.file_paths = [str(data_source)]
            else:
                raise FileNotFoundError(f"数据源不存在: {data_source}")
        elif isinstance(data_source, list):
            self.file_paths = [str(p) for p in data_source]
        else:
            raise TypeError(f"不支持的数据源类型: {type(data_source)}")

        if len(self.file_paths) == 0:
            raise ValueError("未找到任何 .tif 文件")

        print(f"[TifChunkDataset] 加载了 {len(self.file_paths)} 个 chunk 文件")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path = self.file_paths[idx]
        volume = tifffile.imread(file_path)

        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        elif volume.ndim != 3:
            raise ValueError(f"不支持的数据维度 {volume.ndim}D，文件: {file_path}")

        volume = volume.astype(np.float32)
        if self.normalize:
            volume = self._normalize(volume)
        if self.transform is not None:
            volume = self.transform(volume)

        return torch.from_numpy(volume).unsqueeze(0)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        max_val = data.max()
        if max_val <= 1.0 + 1e-6:
            return np.clip(data, 0.0, 1.0)
        elif max_val <= 255.0 + 1e-6:
            return data / 255.0
        else:
            return data / 65535.0

    def get_file_path(self, idx: int) -> str:
        return self.file_paths[idx]


# ===================================================================
#  高性能训练 Dataset
# ===================================================================

class VesuviusTrainDataset(Dataset):
    """
    Vesuvius 训练数据集（高性能版）

    自动检测预处理的 NPY 文件，优先使用 mmap 零拷贝模式。
    若 NPY 不存在则回退到 TIF + LRU 缓存。

    性能对比：
      NPY mmap: ~0.1ms/sample (零拷贝切片)
      TIF cache hit: ~0.1ms/sample (内存缓存)
      TIF cache miss: ~100ms/sample (解压 LZW)

    Args:
        image_dir:  train_images 目录
        label_dir:  train_labels 目录
        crop_size:  3D 随机裁剪尺寸
        transform:  裁剪后的增强变换 (接受 (image, label) 返回 (image, label))
        samples_per_volume: 每个体积每 epoch 采几个 patch
        cache_size: TIF 模式下的 LRU 缓存体积数量
        max_files:  最多使用多少个文件（None=全部）
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        crop_size: int = 96,
        transform: Optional[Callable] = None,
        samples_per_volume: int = 16,
        cache_size: int = 32,
        max_files: Optional[int] = None,
        pos_ratio: float = 0.5,
    ):
        super().__init__()
        self.transform = transform
        self.samples_per_volume = samples_per_volume
        self.cache_size = cache_size
        self.pos_ratio = pos_ratio  # 正样本强制采样比例（拒绝采样）

        # 裁剪尺寸
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        else:
            self.crop_size = tuple(crop_size)

        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        # ====== 优先检测 NPY 目录 ======
        npy_image_dir = image_dir.parent / (image_dir.name + "_npy")
        npy_label_dir = label_dir.parent / (label_dir.name + "_npy")

        self.use_npy = False
        if npy_image_dir.exists() and npy_label_dir.exists():
            image_files = {p.stem: p for p in sorted(npy_image_dir.glob("*.npy"))}
            label_files = {p.stem: p for p in sorted(npy_label_dir.glob("*.npy"))}
            if len(image_files) > 0 and len(label_files) > 0:
                self.use_npy = True
                print(f"[Dataset] 🚀 发现预处理 NPY 数据 "
                      f"({len(image_files)} img + {len(label_files)} lbl)，"
                      f"启用极速 mmap 模式！")
            else:
                print("[Dataset] NPY 目录为空，回退到 TIF 模式")

        if not self.use_npy:
            # 回退到 TIF
            image_files = {
                p.stem: p for p in sorted(image_dir.iterdir())
                if p.suffix.lower() in ('.tif', '.tiff')
            }
            label_files = {
                p.stem: p for p in sorted(label_dir.iterdir())
                if p.suffix.lower() in ('.tif', '.tiff')
            }
            print(f"[Dataset] 使用 TIF 模式 + LRU 缓存 (cache_size={cache_size})")

        # 配对
        common_ids = sorted(set(image_files.keys()) & set(label_files.keys()))
        if max_files is not None:
            common_ids = common_ids[:max_files]

        self.pairs = [
            (str(image_files[cid]), str(label_files[cid]))
            for cid in common_ids
        ]

        if len(self.pairs) == 0:
            raise ValueError(
                f"未找到 image-label 配对！\n"
                f"  image_dir: {image_dir}\n"
                f"  label_dir: {label_dir}"
            )

        # 预扫描体积 shape
        self._shapes = []
        if self.use_npy:
            # NPY: 读 header 获取 shape（极快）
            for img_path, _ in self.pairs:
                arr = np.load(img_path, mmap_mode='r')
                self._shapes.append(arr.shape)
        else:
            # TIF: 读文件头
            for img_path, _ in self.pairs:
                with tifffile.TiffFile(img_path) as tif:
                    self._shapes.append(tif.series[0].shape)

        # TIF 模式下的 LRU 缓存
        self._cache = OrderedDict()

        print(f"[VesuviusTrainDataset] {len(self.pairs)} 个配对, "
              f"crop={self.crop_size}, {samples_per_volume} samples/vol, "
              f"总计 {len(self)} 个训练样本/epoch")

    def __len__(self) -> int:
        return len(self.pairs) * self.samples_per_volume

    def _load_volume(self, vol_idx: int):
        """
        加载体积数据

        NPY 模式: np.load(mmap_mode='r') → 返回 mmap 对象，零 RAM 开销
        TIF 模式: imread + LRU 缓存
        """
        if self.use_npy:
            img_path, lbl_path = self.pairs[vol_idx]
            image = np.load(img_path, mmap_mode='r')
            label = np.load(lbl_path, mmap_mode='r')
            return image, label

        # TIF LRU 缓存
        if vol_idx in self._cache:
            self._cache.move_to_end(vol_idx)
            return self._cache[vol_idx]

        img_path, lbl_path = self.pairs[vol_idx]
        image = tifffile.imread(img_path)
        label = tifffile.imread(lbl_path)

        self._cache[vol_idx] = (image, label)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return image, label

    def _random_crop_coords(self, vol_shape):
        """计算随机裁剪坐标，处理小体积 padding"""
        coords = []
        pads = []
        for dim_size, crop_dim in zip(vol_shape, self.crop_size):
            if dim_size >= crop_dim:
                start = np.random.randint(0, dim_size - crop_dim + 1)
                coords.append((start, start + crop_dim))
                pads.append((0, 0))
            else:
                coords.append((0, dim_size))
                pads.append((0, crop_dim - dim_size))
        return coords, pads

    def _surface_biased_crop(self, label_vol, vol_shape):
        """
        Surface-Biased Rejection Sampling（表面偏向拒绝采样）

        以 pos_ratio 的概率强制采样到含正样本（墨水/纸张）的区域，
        剩余概率进行纯随机采样（负样本挖掘）。

        IO 安全保证：
          - label_vol 是 np.memmap 对象（mmap_mode='r'）
          - label_vol[d0:d1, h0:h1, w0:w1] 切片仅触发缺页中断，
            操作系统只读取对应页面（~0.25MB），不加载整个体积
          - np.any() 短路求值，命中首个非零元素即返回

        Args:
            label_vol: label 体积（mmap 对象或 ndarray）
            vol_shape: 体积的形状 (D, H, W)

        Returns:
            coords, pads: 与 _random_crop_coords 格式一致
        """
        force_positive = (np.random.rand() < self.pos_ratio)

        if force_positive:
            # 拒绝采样：最多重试 10 次寻找含正样本的 patch
            for _attempt in range(10):
                coords, pads = self._random_crop_coords(vol_shape)
                (d0, d1), (h0, h1), (w0, w1) = coords
                # 关键: 仅对 label mmap 做切片 peek，不加载整个体积
                label_patch = label_vol[d0:d1, h0:h1, w0:w1]
                if np.any(label_patch == 1):
                    return coords, pads
            # 10 次全失败（极罕见），接受最后一次的随机坐标
            return coords, pads
        else:
            # 负样本挖掘：纯随机裁剪
            return self._random_crop_coords(vol_shape)

    def __getitem__(self, idx: int):
        """
        加载体积 → 随机裁剪 → 归一化 → 增强 → Tensor

        Returns:
            image: (1, cD, cH, cW) float32 [0, 1]
            label: (1, cD, cH, cW) float32 {0, 1}
        """
        vol_idx = idx // self.samples_per_volume
        image_vol, label_vol = self._load_volume(vol_idx)

        # 随机裁剪（image 和 label 用相同坐标）
        # 使用 Surface-Biased Rejection Sampling 替代纯随机裁剪
        coords, pads = self._surface_biased_crop(label_vol, image_vol.shape)
        (d0, d1), (h0, h1), (w0, w1) = coords

        # 切片 + 转 float32（NPY mmap 此时才触发真正的磁盘读取）
        image = np.array(image_vol[d0:d1, h0:h1, w0:w1], dtype=np.float32)
        label = np.array(label_vol[d0:d1, h0:h1, w0:w1], dtype=np.float32)

        # Padding（体积小于 crop_size 时）
        need_pad = any(p != (0, 0) for p in pads)
        if need_pad:
            image = np.pad(image, pads, mode='constant', constant_values=0)
            label = np.pad(label, pads, mode='constant', constant_values=0)

        # 归一化 image → [0, 1]
        max_val = image.max()
        if max_val > 1.0:
            if max_val <= 255.0:
                image = image / 255.0
            else:
                image = image / 65535.0
        image = np.clip(image, 0.0, 1.0)

        # Label 二值化（方案 A：只认 val=1 为纸草表面，val=2 为忽略区域当作背景）
        # 竞赛标签：0=背景, 1=纸草表面(目标), 2=忽略/填充
        label = (label == 1).astype(np.float32)

        # 增强变换（仅 FlipRotate，Crop 已完成）
        if self.transform is not None:
            image, label = self.transform(image, label)

        # 转 Tensor
        image_t = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        label_t = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        return image_t, label_t


if __name__ == "__main__":
    import time

    print("=== Dataset 自测 ===")
    test_dir = Path("__test_tif_chunks__")
    test_dir.mkdir(exist_ok=True)

    try:
        # TifChunkDataset 测试
        for i in range(3):
            tifffile.imwrite(
                str(test_dir / f"chunk_{i:03d}.tif"),
                np.random.randint(0, 65535, (16, 32, 32), dtype=np.uint16)
            )
        ds = TifChunkDataset(test_dir)
        sample = ds[0]
        assert sample.shape == (1, 16, 32, 32)
        print("✓ TifChunkDataset 通过！")

        # VesuviusTrainDataset 测试 (TIF 模式)
        img_dir = test_dir / "images"
        lbl_dir = test_dir / "labels"
        img_dir.mkdir(exist_ok=True)
        lbl_dir.mkdir(exist_ok=True)
        for i in range(5):
            tifffile.imwrite(str(img_dir / f"vol_{i:03d}.tif"),
                             np.random.randint(0, 255, (64, 64, 64), dtype=np.uint8))
            tifffile.imwrite(str(lbl_dir / f"vol_{i:03d}.tif"),
                             np.random.choice([0, 1, 2], (64, 64, 64)).astype(np.uint8))

        train_ds = VesuviusTrainDataset(
            img_dir, lbl_dir, crop_size=32, samples_per_volume=4, cache_size=5
        )

        t0 = time.time()
        for i in range(20):
            img, lbl = train_ds[i % len(train_ds)]
        t1 = time.time()
        print(f"TIF 模式: {(t1-t0)/20*1000:.1f}ms/sample, "
              f"image={img.shape}, label={lbl.shape}")

        # VesuviusTrainDataset 测试 (NPY 模式)
        npy_img = test_dir / "images_npy"
        npy_lbl = test_dir / "labels_npy"
        npy_img.mkdir(exist_ok=True)
        npy_lbl.mkdir(exist_ok=True)
        for i in range(5):
            np.save(str(npy_img / f"vol_{i:03d}.npy"),
                    np.random.randint(0, 255, (64, 64, 64), dtype=np.uint8))
            np.save(str(npy_lbl / f"vol_{i:03d}.npy"),
                    np.random.choice([0, 1, 2], (64, 64, 64)).astype(np.uint8))

        npy_ds = VesuviusTrainDataset(
            img_dir, lbl_dir, crop_size=32, samples_per_volume=4
        )

        t0 = time.time()
        for i in range(20):
            img, lbl = npy_ds[i % len(npy_ds)]
        t1 = time.time()
        print(f"NPY 模式: {(t1-t0)/20*1000:.1f}ms/sample, "
              f"image={img.shape}, label={lbl.shape}")

        print("\n✓ 所有测试通过！")

    finally:
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
