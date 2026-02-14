"""
Vesuvius Challenge 2026 - GT 标签清洗工具

修复训练标签中的"假性孔洞"(Porous Labels)：
- 3D 形态学闭运算: 先膨胀后腐蚀，填补微小内部孔洞
- 保持整体几何形状不变，不会连接本应分开的层

用法:
  - 在线模式: 在 Dataset.__getitem__ 中调用 clean_mask(label_patch)
  - 离线模式: 运行本文件预处理所有标签并保存
"""

import numpy as np
from scipy import ndimage


def clean_mask(
    mask_volume: np.ndarray,
    closing_radius: int = 1,
    anisotropic: bool = True
) -> np.ndarray:
    """
    对 3D mask 执行形态学闭运算，填补假性孔洞

    Args:
        mask_volume: (D, H, W) 二值掩码 (0/1 或 bool)
        closing_radius: 闭运算结构元素半径
            - 2: 保守，只填 1-2 体素的小孔 (推荐)
            - 3: 激进，填更大的孔但有连接风险
        anisotropic: 是否使用各向异性结构元素
            - True: Z 轴半径 = closing_radius // 2 (保护薄层分离)
            - False: 各向同性球形结构元素

    Returns:
        cleaned: (D, H, W) 修复后的二值掩码 (uint8, 0/1)
    """
    binary = (mask_volume > 0).astype(np.uint8)

    # 构建结构元素
    if anisotropic:
        # 各向异性：Z 轴用更小的半径，防止跨层连接
        z_radius = max(1, closing_radius // 2)
        struct = np.zeros((z_radius * 2 + 1, closing_radius * 2 + 1, closing_radius * 2 + 1), dtype=bool)

        # 在每个 Z 切片上创建 2D 圆盘
        for z in range(struct.shape[0]):
            dz = abs(z - z_radius)
            for y in range(struct.shape[1]):
                for x in range(struct.shape[2]):
                    dy = abs(y - closing_radius)
                    dx = abs(x - closing_radius)
                    if (dz / max(z_radius, 1)) ** 2 + (dy / closing_radius) ** 2 + (dx / closing_radius) ** 2 <= 1:
                        struct[z, y, x] = True
    else:
        # 各向同性球形
        struct = ndimage.generate_binary_structure(3, 1)
        struct = ndimage.iterate_structure(struct, closing_radius)

    # 闭运算 = 膨胀 + 腐蚀
    # 膨胀：填充孔洞边缘
    dilated = ndimage.binary_dilation(binary, structure=struct).astype(np.uint8)
    # 腐蚀：恢复原始轮廓
    cleaned = ndimage.binary_erosion(dilated, structure=struct).astype(np.uint8)

    return cleaned


# ========== 可视化验证 ==========

if __name__ == "__main__":
    import os
    import tifffile
    import matplotlib
    matplotlib.use('Agg')  # 无 GUI 后端
    import matplotlib.pyplot as plt

    print("=" * 50)
    print("  GT 标签清洗验证")
    print("=" * 50)

    # 加载一个训练标签
    label_path = "data/vesuvius-challenge-surface-detection/train_labels/1004283650.tif"
    if not os.path.exists(label_path):
        print(f"标签文件不存在: {label_path}")
        exit()

    label = tifffile.imread(label_path)
    print(f"原始标签: shape={label.shape}, dtype={label.dtype}, range=[{label.min()},{label.max()}]")
    binary = (label > 0).astype(np.uint8)
    print(f"二值化后: 正样本比例={binary.mean()*100:.1f}%")

    # 执行清洗
    cleaned = clean_mask(binary, closing_radius=1, anisotropic=True)
    print(f"清洗后:   正样本比例={cleaned.mean()*100:.1f}%")

    # 统计变化 (用 int() 避免 uint8 溢出)
    added = int(((cleaned == 1) & (binary == 0)).sum())
    removed = int(((cleaned == 0) & (binary == 1)).sum())
    print(f"新增体素: {added:,} (填补的孔洞)")
    print(f"移除体素: {removed:,} (闭运算收缩)")
    print(f"净变化:   {added - removed:+,}")

    # 可视化：对比修复前后的中间切片
    mid_z = label.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(binary[mid_z], cmap='gray')
    axes[0].set_title(f'原始 (z={mid_z})')
    axes[0].axis('off')

    axes[1].imshow(cleaned[mid_z], cmap='gray')
    axes[1].set_title(f'清洗后 (z={mid_z})')
    axes[1].axis('off')

    # 差异图：红色=新增, 蓝色=移除
    diff = np.zeros((*binary[mid_z].shape, 3), dtype=np.float32)
    diff[..., 0] = ((cleaned[mid_z] == 1) & (binary[mid_z] == 0)).astype(np.float32)  # 红: 新增
    diff[..., 2] = ((cleaned[mid_z] == 0) & (binary[mid_z] == 1)).astype(np.float32)  # 蓝: 移除
    diff[..., 1] = (binary[mid_z] * 0.3)  # 绿: 原始轮廓

    axes[2].imshow(diff)
    axes[2].set_title(f'差异 (红=填补, 蓝=收缩)')
    axes[2].axis('off')

    output_path = "outputs/mask_cleaning_comparison.png"
    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n可视化已保存: {output_path}")

    # 层间一致性检查
    print(f"\n[层间一致性检查]")
    for z in [mid_z - 2, mid_z - 1, mid_z, mid_z + 1, mid_z + 2]:
        orig_sum = int(binary[z].sum())
        clean_sum = int(cleaned[z].sum())
        diff_pct = (clean_sum - orig_sum) / max(orig_sum, 1) * 100
        print(f"  z={z}: 原始={orig_sum:,}, 清洗={clean_sum:,}, 变化={diff_pct:+.1f}%")
