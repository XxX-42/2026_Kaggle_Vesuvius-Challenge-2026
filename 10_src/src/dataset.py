"""
Vesuvius Challenge 2026 - 3D 数据加载器
用于处理 3D TIF 卷轴数据的 PyTorch Dataset 类

核心功能：
- 使用 tifffile 读取 3D TIF volume
- RandomCrop 提取固定大小的 Patch
- uint16 → float32 归一化到 [0, 1]
- 输出格式：(C=1, D, H, W)
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tifffile


class VesuviusDataset(Dataset):
    """
    Vesuvius 3D 卷轴数据集
    
    从大型 3D TIF Volume 中随机裁剪固定大小的 Patch 用于训练。
    
    Args:
        csv_path: train.csv 文件路径，包含 id 和 scroll_id 列
        image_root: 图像文件根目录（包含 {id}.tif 文件）
        label_root: 标签文件根目录（包含 {id}.tif 文件）
        patch_size: 3D Patch 尺寸 (Depth, Height, Width)，默认 (64, 128, 128)
        transform: 可选的数据增强函数
    """
    
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        label_root: str,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        transform: Optional[Callable] = None,
        mode: str = 'train',
        pos_ratio: float = 0.5  # [NEW] 正样本采样比例
    ):
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.patch_size = patch_size  # (D, H, W)
        self.transform = transform
        self.mode = mode
        self.pos_ratio = pos_ratio
        
        # [CRITICAL UPDATE] 阈值从1500降到300以解除IO瓶颈 (速度优先)
        self.rejection_threshold = 300
        self.max_retries = 20
        
        # ====== 自动检测 NPY 目录 ======
        # 假设 NPY 目录名为 {root}_npy，例如 train_images_npy
        self.npy_image_root = self.image_root.parent / (self.image_root.name + "_npy")
        self.npy_label_root = self.label_root.parent / (self.label_root.name + "_npy")
        
        self.use_npy = False
        if self.npy_image_root.exists() and self.npy_label_root.exists():
            self.use_npy = True
            print(f"[Dataset] 🚀 发现预处理 NPY 数据，启用极速 mmap 模式！")
            print(f"          Image: {self.npy_image_root}")
            print(f"          Label: {self.npy_label_root}")
        else:
            print(f"[Dataset] 未找到 NPY 目录，回退到 TIF 模式 (IO 较慢)")
            print(f"          Image: {self.image_root}")
            print(f"          Label: {self.label_root}")
        
        # 读取 CSV 并验证文件存在性
        self.df = pd.read_csv(csv_path)
        self._validate_files()
        
        print(f"[VesuviusDataset] 初始化完成：共 {len(self.df)} 个样本 (mode={mode})")
        print(f"[VesuviusDataset] Patch 尺寸：{patch_size} (D, H, W)")
        print(f"[VesuviusDataset] Sampling Strategy: Surface-Biased (pos_ratio={pos_ratio})")
    
    def _validate_files(self) -> None:
        """验证所有样本的图像和标签文件是否存在，过滤掉缺失的样本"""
        valid_indices = []
        missing_count = 0
        
        for idx, row in self.df.iterrows():
            sample_id = row['id']
            image_path = self.image_root / f"{sample_id}.tif"
            label_path = self.label_root / f"{sample_id}.tif"
            
            if image_path.exists() and label_path.exists():
                valid_indices.append(idx)
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"[VesuviusDataset] 警告：发现 {missing_count} 个缺失文件，已自动过滤")
        
        # 只保留有效样本
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个训练样本
        
        NPY 模式优化:
        - _load_data 返回 mmap 对象 (零拷贝)
        - _surface_biased_crop 进行切片 (只读取 Patch 数据)
        - 转换为 float32 并归一化
        """
        sample_id = self.df.iloc[idx]['id']
        
        # 1. 加载数据 (NPY mmap 或 TIF)
        image_vol, label_vol = self._load_data(str(sample_id))
        
        # 2. 采样裁切 (Surface-Biased)
        # 传入的是整个 Volume (或 mmap)，在内部进行切片读取
        image, label = self._surface_biased_crop(image_vol, label_vol)
        
        # 3. 转换为 float32 并归一化 (image: [0, 1])
        # 注意：mmap 切片后得到的是 numpy array，此时数据已在于内存中
        image = self._normalize(image)
        # label = self._normalize(label) # 废弃旧逻辑, label 保持原始值后续做二值化
        
        # [P0 FIX] 竞赛标签定义：0=背景, 1=纸草表面(目标), 2=忽略区域
        # 1. 生成有效性 Mask (Valid Mask)：val != 2 的区域为有效 (1.0)，val == 2 为无效 (0.0)
        valid_mask = (label != 2).astype(np.float32)
        
        # 2. 生成二值 Label：只认 val=1 为正样本，val=2 为忽略区域（当作背景处理）
        label = (label == 1).astype(np.float32)
        
        # [P3] 形态学闭运算清洗（可选，默认关闭以保性能）
        # 启用方式: VesuviusDataset(..., clean_labels=True)
        if getattr(self, 'clean_labels', False):
            from src.mask_cleaning import clean_mask
            label = clean_mask(label, closing_radius=1, anisotropic=True).astype(np.float32)
        
        # 转换为 PyTorch 张量并添加 Channel 维度
        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, D, H, W)
        label = torch.from_numpy(label).float().unsqueeze(0)  # (1, D, H, W)
        valid_mask = torch.from_numpy(valid_mask).float().unsqueeze(0) # (1, D, H, W)
        
        # [CRITICAL DECISION] 将 label 和 valid_mask 拼接，传递给 transform
        # label shape: (2, D, H, W) -> Channel 0: Label, Channel 1: ValidMask
        label = torch.cat([label, valid_mask], dim=0)
        
        # 应用数据增强 (transform 通常能处理多通道 label)
        if self.transform is not None:
            image, label = self.transform(image, label)
        
        return image, label
    
    def _load_data(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """统一加载数据 (NPY mmap 或 TIF 读取)"""
        if self.use_npy:
            img_path = self.npy_image_root / f"{sample_id}.npy"
            lbl_path = self.npy_label_root / f"{sample_id}.npy"
            # mmap_mode='r' 零拷贝，极速打开
            image = np.load(img_path, mmap_mode='r')
            label = np.load(lbl_path, mmap_mode='r')
        else:
            img_path = self.image_root / f"{sample_id}.tif"
            lbl_path = self.label_root / f"{sample_id}.tif"
            image = tifffile.imread(str(img_path))
            label = tifffile.imread(str(lbl_path))
            
        return image, label

    def _surface_biased_crop(
        self, 
        image: np.ndarray, 
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Surface-Biased Sampling (表面偏置采样)
        
        策略:
        1. 以 pos_ratio 的概率强制采样包含正样本 (val=1) 的 Patch。
        2. 剩余概率进行随机采样 (负样本挖掘)。
        
        优化:
        - 针对 mmap 数组，先切片再检查，避免全量 I/O。
        - 使用 buffer 避免边界溢出。
        """
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # 填充 (如果 volume 小于 patch)
        if d < pd or h < ph or w < pw:
            # 对于 mmap，我们需要先读出来再 pad (会触发 IO)
            # 但通常 volume 很大。这里为了安全，如果是 mmap 且这就发生了，
            # 可能需要特殊处理。简单起见，转为 numpy array (IO cost incurred)
            if isinstance(image, np.memmap):
                image = np.array(image)
                label = np.array(label)
                
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            d, h, w = image.shape
            
        # 决定是否强制采样正样本
        force_positive = (random.random() < self.pos_ratio)
        
        for attempt in range(self.max_retries):
            # 随机坐标
            sd = random.randint(0, d - pd)
            sh = random.randint(0, h - ph)
            sw = random.randint(0, w - pw)
            
            # [Optimization] 如果是强制正样本模式，先只切 label 检查
            if force_positive:
                # 切片 (若为 mmap，此时仅读取 meta info，不读取数据?)
                # 实际上 np.sum 或 np.any 会触发读取。
                # 切取小块 patch 读取开销很小 (~0.5MB)。
                lbl_patch = label[sd:sd+pd, sh:sh+ph, sw:sw+pw]
                
                # 检查是否包含正样本 (val=1)
                # 注意：原始 label 可能包含 0, 1, 2
                # 我们寻找 val=1 的区域
                if np.any(lbl_patch == 1):
                    # 找到了！读取 image 并返回
                    img_patch = image[sd:sd+pd, sh:sh+ph, sw:sw+pw]
                    return img_patch, lbl_patch
            else:
                # 随机模式 (负样本挖掘)，直接接受
                img_patch = image[sd:sd+pd, sh:sh+ph, sw:sw+pw]
                lbl_patch = label[sd:sd+pd, sh:sh+ph, sw:sw+pw]
                return img_patch, lbl_patch
        
        # 如果重试多次仍未找到正样本，退化为随机采样
        img_patch = image[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        lbl_patch = label[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        return img_patch, lbl_patch
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        将数据归一化到 [0, 1] 范围
        
        [P0 FIX] 不再依赖 max_val 启发式判断 dtype，
        直接根据原始 dtype 确定归一化系数，避免边界情况。
        """
        original_dtype = data.dtype
        data = data.astype(np.float32)
        
        # 根据原始 dtype 选择归一化系数
        if original_dtype == np.uint16:
            data = data / 65535.0
        elif original_dtype == np.uint8:
            data = data / 255.0
        # float32 / float64: 假设已经归一化，不处理
        
        return data



def create_dataloader(
    csv_path: str,
    image_root: str,
    label_root: str,
    batch_size: int = 4,
    patch_size: Tuple[int, int, int] = (64, 128, 128),
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[Callable] = None
) -> torch.utils.data.DataLoader:
    """
    创建 DataLoader 的便捷函数
    
    Args:
        csv_path: train.csv 文件路径
        image_root: 图像文件根目录
        label_root: 标签文件根目录
        batch_size: 批次大小
        patch_size: 3D Patch 尺寸 (D, H, W)
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        transform: 可选的数据增强函数
    
    Returns:
        DataLoader 实例
    """
    dataset = VesuviusDataset(
        csv_path=csv_path,
        image_root=image_root,
        label_root=label_root,
        patch_size=patch_size,
        transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # 简单测试
    print("测试 VesuviusDataset...")
    
    # 使用相对路径（从项目根目录运行）
    csv_path = "data/vesuvius-challenge-surface-detection/train.csv"
    image_root = "data/vesuvius-challenge-surface-detection/train_images"
    label_root = "data/vesuvius-challenge-surface-detection/train_labels"
    
    try:
        dataset = VesuviusDataset(
            csv_path=csv_path,
            image_root=image_root,
            label_root=label_root,
            patch_size=(64, 128, 128)
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 获取一个样本
        image, label = dataset[0]
        print(f"图像 shape: {image.shape}, dtype: {image.dtype}")
        print(f"标签 shape: {label.shape}, dtype: {label.dtype}")
        print(f"图像值范围: [{image.min():.4f}, {image.max():.4f}]")
        print(f"标签值范围: [{label.min():.4f}, {label.max():.4f}]")
        
    except Exception as e:
        print(f"测试失败: {e}")
