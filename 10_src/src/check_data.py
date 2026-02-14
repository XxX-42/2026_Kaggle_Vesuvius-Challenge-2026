"""
Vesuvius Challenge 2026 - 数据加载验证脚本

功能：
1. 初始化 VesuviusDataset 并加载一个样本
2. 打印 Tensor 信息（shape, dtype, min/max）
3. 可视化中间切片：CT 图像 + Label Mask 叠加
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import tifffile

from src.dataset import VesuviusDataset


def visualize_sample(
    image: torch.Tensor, 
    label: torch.Tensor, 
    save_path: str = None
) -> None:
    """
    可视化 3D Patch 的中间切片
    
    Args:
        image: 图像张量，形状 (1, D, H, W)
        label: 标签张量，形状 (1, D, H, W)
        save_path: 图片保存路径（可选）
    """
    # 获取中间切片索引
    depth = image.shape[1]
    mid_slice = depth // 2
    
    # 提取中间切片
    img_slice = image[0, mid_slice].numpy()  # (H, W)
    lbl_slice = label[0, mid_slice].numpy()  # (H, W)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 左图：CT 图像
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title(f'CT 图像 (切片 {mid_slice}/{depth})', fontsize=12)
    axes[0].axis('off')
    
    # 中图：Label Mask
    axes[1].imshow(lbl_slice, cmap='hot')
    axes[1].set_title('Label Mask', fontsize=12)
    axes[1].axis('off')
    
    # 右图：叠加显示
    axes[2].imshow(img_slice, cmap='gray')
    # 创建红色透明遮罩
    mask_overlay = np.zeros((*lbl_slice.shape, 4))  # RGBA
    mask_overlay[lbl_slice > 0.5, 0] = 1.0  # Red
    mask_overlay[lbl_slice > 0.5, 3] = 0.5  # Alpha
    axes[2].imshow(mask_overlay)
    axes[2].set_title('CT + Mask 叠加', fontsize=12)
    axes[2].axis('off')
    
    # 添加图例
    red_patch = mpatches.Patch(color='red', alpha=0.5, label='表面标注')
    axes[2].legend(handles=[red_patch], loc='upper right')
    
    plt.suptitle('Vesuvius 3D Patch 可视化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] 图片已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """打印张量详细信息"""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Shape : {tuple(tensor.shape)}")
    print(f"  Dtype : {tensor.dtype}")
    print(f"  Min   : {tensor.min().item():.6f}")
    print(f"  Max   : {tensor.max().item():.6f}")
    print(f"  Mean  : {tensor.mean().item():.6f}")
    print(f"  Std   : {tensor.std().item():.6f}")


def main():
    """主函数：加载数据并验证"""
    print("\n" + "="*60)
    print("  Vesuvius Challenge 2026 - 数据加载验证")
    print("="*60)
    
    # 配置路径
    # data 目录在项目根目录的上级 (与 10_src 平级)
    data_root = project_root.parent / "data" / "vesuvius-challenge-surface-detection"
    csv_path = data_root / "train.csv"
    image_root = data_root / "train_images"
    label_root = data_root / "train_labels"
    output_path = project_root / "outputs" / "data_check.png"
    
    # 检查路径
    print(f"\n[配置信息]")
    print(f"  CSV 路径    : {csv_path}")
    print(f"  图像根目录  : {image_root}")
    print(f"  标签根目录  : {label_root}")
    print(f"  输出路径    : {output_path}")
    
    # 初始化 Dataset
    print(f"\n[初始化 Dataset]")
    patch_size = (64, 128, 128)
    
    try:
        dataset = VesuviusDataset(
            csv_path=str(csv_path),
            image_root=str(image_root),
            label_root=str(label_root),
            patch_size=patch_size
        )
    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到:\n{e}")
        return
    except Exception as e:
        print(f"\n[错误] 初始化失败:\n{e}")
        return
    
    # 获取一个样本
    print(f"\n[加载样本] 正在加载第一个样本...")
    image, label = dataset[0]
    
    # 打印张量信息
    print_tensor_info("图像张量 (Image)", image)
    print_tensor_info("标签张量 (Label)", label)
    
    # 验证格式
    print(f"\n[格式验证]")
    expected_shape = (1, *patch_size)
    
    if image.shape == expected_shape:
        print(f"  ✓ 图像形状正确: {tuple(image.shape)}")
    else:
        print(f"  ✗ 图像形状错误: 期望 {expected_shape}，实际 {tuple(image.shape)}")
    
    # 标签形状可能是 1 (Label) 或 2 (Label + ValidMask)
    if label.shape[0] in [1, 2] and label.shape[1:] == expected_shape[1:]:
        print(f"  ✓ 标签形状正确: {tuple(label.shape)} (Channels: {label.shape[0]})")
        if label.shape[0] == 2:
            print("    (包含 ValidMask 通道)")
    else:
        print(f"  ✗ 标签形状错误: 期望 (1/2, {patch_size})，实际 {tuple(label.shape)}")
    
    if image.dtype == torch.float32:
        print(f"  ✓ 图像数据类型正确: {image.dtype}")
    else:
        print(f"  ✗ 图像数据类型错误: 期望 float32，实际 {image.dtype}")
    
    if 0 <= image.min() and image.max() <= 1:
        print(f"  ✓ 图像值范围正确: [{image.min():.4f}, {image.max():.4f}]")
    else:
        print(f"  ✗ 图像值范围错误: 期望 [0, 1]，实际 [{image.min():.4f}, {image.max():.4f}]")
    
    # 可视化
    print(f"\n[生成可视化]")
    visualize_sample(image, label, save_path=str(output_path))
    
    # ========== 标签稀疏性验证 ==========
    # 参考 20_src/verify_data.py：正确标签应非常稀疏 (val=1 < 10%)
    print(f"\n[标签稀疏性验证]")
    print(f"  参考标准：val=0 背景, val=1 纸草表面(目标), val=2 忽略区域")
    
    num_samples_to_check = min(5, len(dataset))
    fatal_errors = []
    
    for check_idx in range(num_samples_to_check):
        sample_id = dataset.df.iloc[check_idx]['id']
        lbl_path = dataset.label_root / f"{sample_id}.tif"
        raw_label = tifffile.imread(str(lbl_path))
        
        total = raw_label.size
        count_0 = np.sum(raw_label == 0)
        count_1 = np.sum(raw_label == 1)
        count_2 = np.sum(raw_label == 2)
        
        surface_ratio = count_1 / total
        ignore_ratio = count_2 / total
        bg_ratio = count_0 / total
        
        # 状态判断 (与 20_src/verify_data.py 一致)
        status = "✓"
        if surface_ratio > 0.30:
            status = "⚠ WARNING"
        if surface_ratio > 0.90:
            status = "✗ FATAL"
            fatal_errors.append(sample_id)
        
        print(f"  {status} {sample_id}: "
              f"背景(0)={bg_ratio*100:.1f}% | "
              f"表面(1)={surface_ratio*100:.1f}% | "
              f"忽略(2)={ignore_ratio*100:.1f}%")
    
    if fatal_errors:
        print(f"\n  ✗ 致命错误！以下样本正样本比例 > 90%: {fatal_errors}")
    else:
        print(f"  ✓ 标签稀疏性检查通过")
    
    # 测试 DataLoader
    print(f"\n[测试 DataLoader]")
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0  # Windows 需要设为 0
    )
    
    batch_image, batch_label = next(iter(dataloader))
    print(f"  Batch 图像形状: {tuple(batch_image.shape)}")
    print(f"  Batch 标签形状: {tuple(batch_label.shape)}")
    
    print(f"\n{'='*60}")
    print("  ✓ 数据加载验证完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
