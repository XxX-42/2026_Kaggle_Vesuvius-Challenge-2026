"""
Vesuvius Challenge - 数据验证脚本

功能：检查预处理后的 NPY 标签文件的稀疏性，确保模型训练在正确的目标上。
目的：防止"训练在实心 Mask 上"的致命错误再次发生。

正确的标签应该非常稀疏（纸草表面 ~5%），而不是实心块。

用法:
    python 20_src/verify_data.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def verify_labels(label_dir: str, num_samples: int = 5, save_png: bool = True):
    """
    验证标签文件的内容和稀疏性

    Args:
        label_dir: NPY 标签目录
        num_samples: 随机抽样检查的文件数量
        save_png: 是否保存切片可视化 PNG
    """
    label_path = Path(label_dir)
    if not label_path.exists():
        print(f"❌ 标签目录不存在: {label_path}")
        sys.exit(1)

    npy_files = sorted(label_path.glob("*.npy"))
    if len(npy_files) == 0:
        print(f"❌ 未找到 .npy 文件: {label_path}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  🔬 Vesuvius 标签数据验证")
    print(f"  目录: {label_path}")
    print(f"  文件数: {len(npy_files)}")
    print(f"{'='*60}\n")

    # 随机抽样
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(npy_files), size=min(num_samples, len(npy_files)), replace=False)
    sample_files = [npy_files[i] for i in sorted(sample_indices)]

    all_sparsities = []
    fatal_errors = []

    for f in sample_files:
        vol = np.load(str(f), mmap_mode='r')
        total = vol.size
        unique_vals = np.unique(vol)

        # 统计各值占比
        count_0 = np.sum(vol == 0)
        count_1 = np.sum(vol == 1)
        count_2 = np.sum(vol == 2)

        # 计算稀疏度（val=1 是目标）
        surface_ratio = count_1 / total
        ignore_ratio = count_2 / total
        bg_ratio = count_0 / total

        all_sparsities.append(surface_ratio)

        # 状态判断
        status = "✅"
        if surface_ratio > 0.30:
            status = "⚠️ WARNING"
        if surface_ratio > 0.90:
            status = "❌ FATAL"
            fatal_errors.append(f.name)

        print(f"{status} {f.name}:")
        print(f"    shape={vol.shape}, dtype={vol.dtype}, unique={unique_vals}")
        print(f"    背景(0): {bg_ratio*100:.1f}% | "
              f"表面(1): {surface_ratio*100:.1f}% | "
              f"忽略(2): {ignore_ratio*100:.1f}%")
        print()

    # 总结
    avg_sparsity = np.mean(all_sparsities)
    print(f"{'='*60}")
    print(f"  📊 汇总统计")
    print(f"  平均表面占比 (val=1): {avg_sparsity*100:.2f}%")
    print(f"{'='*60}")

    if avg_sparsity < 0.10:
        print(f"\n  ✅ 数据正常！表面标签稀疏度合理 ({avg_sparsity*100:.1f}% < 10%)")
        print(f"  → 模型应该学习画'稀疏的线条'，而非'实心方块'")
    elif avg_sparsity < 0.30:
        print(f"\n  ⚠️ 注意：表面占比偏高 ({avg_sparsity*100:.1f}%)，但可能仍然合理")
    else:
        print(f"\n  ❌ 严重问题！表面占比过高 ({avg_sparsity*100:.1f}%)，"
              f"可能仍在使用错误的标签！")

    if fatal_errors:
        print(f"\n  ❌ FATAL: 以下文件的正样本比例 > 90%: {fatal_errors}")
        sys.exit(1)

    # 生成可视化
    if save_png:
        sample_vol = np.load(str(sample_files[0]))
        mid_z = sample_vol.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始标签 (0, 1, 2)
        axes[0].imshow(sample_vol[mid_z], cmap='tab10', vmin=0, vmax=2)
        axes[0].set_title(f'原始标签 (z={mid_z})\n0=背景, 1=表面, 2=忽略')

        # 只看 val=1 (表面)
        surface = (sample_vol[mid_z] == 1).astype(np.float32)
        axes[1].imshow(surface, cmap='hot', vmin=0, vmax=1)
        sr = np.sum(surface) / surface.size * 100
        axes[1].set_title(f'表面 (val=1)\n稀疏度: {sr:.1f}%')

        # 只看 val=2 (忽略区域)
        ignore = (sample_vol[mid_z] == 2).astype(np.float32)
        axes[2].imshow(ignore, cmap='Blues', vmin=0, vmax=1)
        ir = np.sum(ignore) / ignore.size * 100
        axes[2].set_title(f'忽略区域 (val=2)\n占比: {ir:.1f}%')

        for ax in axes:
            ax.axis('off')

        fig.suptitle(f'文件: {sample_files[0].name}', fontsize=12, fontweight='bold')
        plt.tight_layout()

        out_path = Path("20_src/output/verification_slice.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  📸 可视化已保存: {out_path}")


if __name__ == "__main__":
    verify_labels("data/vesuvius-challenge-surface-detection/train_labels_npy")
