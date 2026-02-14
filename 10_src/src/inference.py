"""
Vesuvius Challenge 2026 - 竞赛提交推理脚本
===========================================

功能特性:
  1. 双模型集成 (Pixel 0.6 + Topo 0.4 加权平均)
  2. TTA 测试时增强 (Z 轴翻转 + XY 旋转 90°)
  3. 高斯加权滑动窗口 (50% 重叠，消除边缘伪影)
  4. 后处理: 阈值二值化 + 最大连通分量 (LCC)
  5. FP16 推理 (torch.cuda.amp)
  6. Memory Map 大文件支持
  7. 输出: RLE CSV 或 多页 TIF

用法:
  python submission.py \
    --model_pixel outputs/BACKUP_best_model_Ep15_Dice0.7080.pth \
    --model_topo  outputs/Phase9_clDice_W8_.../checkpoints/best_model.pth \
    --data_dir data/vesuvius-challenge-surface-detection \
    --output_dir outputs/submission
"""

import os
import sys
import argparse
import time
import gc
import csv
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model import ResUNet3D


# ============================================================================
#  工具函数
# ============================================================================

def get_gaussian_kernel(patch_size: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """
    生成 3D 高斯权重核，用于滑动窗口融合时消除边缘伪影。
    中心权重高、边缘权重低，确保重叠区域平滑过渡。
    """
    d, h, w = patch_size

    def _1d_gaussian(size: int) -> torch.Tensor:
        sigma = size / 6.0  # 约 ±3σ 覆盖
        x = torch.arange(size, device=device, dtype=torch.float32) - size / 2.0 + 0.5
        return torch.exp(-x ** 2 / (2 * sigma ** 2))

    g_d = _1d_gaussian(d)
    g_h = _1d_gaussian(h)
    g_w = _1d_gaussian(w)

    # 外积生成 3D 核: (D,1,1) * (1,H,1) * (1,1,W) -> (D,H,W)
    kernel = g_d.view(-1, 1, 1) * g_h.view(1, -1, 1) * g_w.view(1, 1, -1)
    return kernel


def load_model(model_path: str, device: torch.device) -> ResUNet3D:
    """
    加载单个 ResUNet3D 模型权重。
    自动处理 strict=False 以兼容不同训练阶段的 checkpoint。
    """
    model = ResUNet3D(in_channels=1, out_channels=1)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 兼容不同格式的 checkpoint (可能带 'model_state_dict' 或直接是 state_dict)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # strict=False: 忽略多余的 key (如 Affinity Head)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        print(f"  [Info] 忽略多余的 key: {load_result.unexpected_keys}")

    model.to(device)
    model.eval()
    return model


# ============================================================================
#  TTA (测试时增强)
# ============================================================================

def tta_forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    TTA 前向推理: 对输入进行多种增强，将所有预测取平均。

    增强策略:
      1. 原始输入
      2. Z 轴翻转 (depth flip)
      3. XY 平面旋转 90°
      4. Z 轴翻转 + XY 旋转 90° (组合)

    Args:
        model: 已加载的模型
        inputs: (B, 1, D, H, W) 输入张量

    Returns:
        平均概率图: (B, 1, D, H, W)，范围 [0, 1]
    """
    probs_list = []

    # 1. 原始输入
    with torch.amp.autocast('cuda'):
        logits = model(inputs)
        probs_list.append(torch.sigmoid(logits))

    # 2. Z 轴翻转 (dim=2 对应 D 维度)
    flipped_z = torch.flip(inputs, dims=[2])
    with torch.amp.autocast('cuda'):
        logits_fz = model(flipped_z)
        # 将预测结果翻转回来再平均
        probs_fz = torch.sigmoid(logits_fz)
        probs_fz = torch.flip(probs_fz, dims=[2])
        probs_list.append(probs_fz)

    # 3. XY 平面旋转 90° (对 H, W 维度转置, dim=3 和 dim=4)
    # 仅当 H == W 时才安全执行 (避免尺寸不匹配)
    _, _, _, h, w = inputs.shape
    if h == w:
        rotated = torch.rot90(inputs, k=1, dims=[3, 4])
        with torch.amp.autocast('cuda'):
            logits_r = model(rotated)
            probs_r = torch.sigmoid(logits_r)
            # 旋转回来: rot90 k=3 等同于逆时针 270° = 顺时针 90° 的逆操作
            probs_r = torch.rot90(probs_r, k=3, dims=[3, 4])
            probs_list.append(probs_r)

        # 4. Z 翻转 + XY 旋转 90° (组合增强)
        flipped_rotated = torch.flip(rotated, dims=[2])
        with torch.amp.autocast('cuda'):
            logits_fr = model(flipped_rotated)
            probs_fr = torch.sigmoid(logits_fr)
            probs_fr = torch.flip(probs_fr, dims=[2])
            probs_fr = torch.rot90(probs_fr, k=3, dims=[3, 4])
            probs_list.append(probs_fr)

    # 对所有增强结果取平均
    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)

    # 显式释放中间结果
    del probs_list
    return avg_probs


# ============================================================================
#  滑动窗口推理引擎
# ============================================================================

def compute_sliding_window_coords(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """
    计算滑动窗口的起始坐标列表。
    确保覆盖整个 Volume（包括边缘不完整的部分）。
    """
    depth, height, width = volume_shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    coords = []

    z_starts = list(range(0, max(depth - pd + 1, 1), sd))
    if len(z_starts) == 0 or z_starts[-1] + pd < depth:
        z_starts.append(max(depth - pd, 0))

    y_starts = list(range(0, max(height - ph + 1, 1), sh))
    if len(y_starts) == 0 or y_starts[-1] + ph < height:
        y_starts.append(max(height - ph, 0))

    x_starts = list(range(0, max(width - pw + 1, 1), sw))
    if len(x_starts) == 0 or x_starts[-1] + pw < width:
        x_starts.append(max(width - pw, 0))

    # 去重并排序
    z_starts = sorted(set(z_starts))
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                coords.append((z, y, x))

    return coords


def predict_volume_ensemble(
    models: List[torch.nn.Module],
    weights: List[float],
    layer_files: Optional[List[Path]],
    volume_data: Optional[np.ndarray],
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device: torch.device,
    batch_size: int = 4,
    use_tta: bool = True,
    temp_dir: Optional[Path] = None,
    max_patches: Optional[int] = None
) -> np.ndarray:
    """
    双模型集成 + TTA + 高斯加权滑动窗口推理。

    流程:
      1. 对每个 Patch，用所有模型分别推理（可选 TTA）
      2. 按权重加权平均各模型的概率输出
      3. 高斯核加权累积到全局 Volume

    Args:
        models: 模型列表 [model_pixel, model_topo]
        weights: 对应权重 [0.6, 0.4]
        layer_files: 排序好的 layer TIF 文件列表
        volume_shape: (D, H, W) 完整 Volume 尺寸
        patch_size: (pD, pH, pW) 窗口大小
        stride: (sD, sH, sW) 步长
        device: GPU 设备
        batch_size: 推理 Batch Size
        use_tta: 是否启用 TTA
        temp_dir: 临时文件目录

    Returns:
        概率图 (D, H, W)，范围 [0, 1]
    """
    depth, height, width = volume_shape
    pd, ph, pw = patch_size

    # 初始化结果容器 (Memory Map 避免内存溢出)
    if temp_dir is None:
        temp_dir = Path("temp_inference")
    temp_dir.mkdir(parents=True, exist_ok=True)

    pred_sum = np.memmap(
        temp_dir / "pred_sum.dat", dtype='float32', mode='w+', shape=volume_shape
    )
    weight_sum = np.memmap(
        temp_dir / "weight_sum.dat", dtype='float32', mode='w+', shape=volume_shape
    )

    # 高斯权重核
    gaussian_kernel = get_gaussian_kernel(patch_size, device)
    gaussian_np = gaussian_kernel.cpu().numpy()

    # 滑动窗口坐标
    coords = compute_sliding_window_coords(volume_shape, patch_size, stride)
    total_patches = len(coords)
    print(f"  滑动窗口配置: Patch {patch_size}, Stride {stride}")
    print(f"  总 Patch 数: {total_patches}")
    print(f"  模型集成: {len(models)} 个模型, 权重 = {weights}")
    print(f"  TTA: {'ON (4x 增强)' if use_tta else 'OFF'}")

    # Debug 模式: 限制 Patch 数量
    if max_patches is not None and max_patches > 0:
        coords = coords[:max_patches]
        total_patches = len(coords)
        print(f"  [DEBUG] 限制为 {total_patches} 个 Patch")

    # ---- Patch 加载函数 ----
    def load_patch(z: int, y: int, x: int) -> np.ndarray:
        """从 volume_data (预加载的 3D 数组) 或 layer_files 中加载指定 Patch"""
        patch = np.zeros((pd, ph, pw), dtype=np.float32)

        if volume_data is not None:
            # 模式 A: 从预加载的 3D numpy 数组中裁剪
            raw = volume_data[z:z + pd, y:y + ph, x:x + pw]
            # 处理边界情况
            if raw.shape != (pd, ph, pw):
                patch[:raw.shape[0], :raw.shape[1], :raw.shape[2]] = raw.astype(np.float32)
            else:
                patch = raw.astype(np.float32)
            # 归一化
            if volume_data.dtype == np.uint16:
                patch /= 65535.0
            elif volume_data.dtype == np.uint8:
                patch /= 255.0
        elif layer_files is not None:
            # 模式 B: 从分层 TIF 文件逐层读取
            for i, z_idx in enumerate(range(z, z + pd)):
                if z_idx < len(layer_files):
                    img = tifffile.imread(layer_files[z_idx])
                    crop = img[y:y + ph, x:x + pw]
                    if crop.shape != (ph, pw):
                        padded = np.zeros((ph, pw), dtype=img.dtype)
                        padded[:crop.shape[0], :crop.shape[1]] = crop
                        crop = padded
                    if crop.dtype == np.uint16:
                        patch[i] = crop.astype(np.float32) / 65535.0
                    elif crop.dtype == np.uint8:
                        patch[i] = crop.astype(np.float32) / 255.0
                    else:
                        patch[i] = crop.astype(np.float32)

        return patch

    # ---- 批量推理 ----
    pbar = tqdm(total=total_patches, desc="  推理中", unit="patch")

    batch_patches = []
    batch_coords = []

    with torch.no_grad():
        for idx, (z, y, x) in enumerate(coords):
            patch = load_patch(z, y, x)
            batch_patches.append(patch[np.newaxis, ...])  # (1, D, H, W)
            batch_coords.append((z, y, x))

            if len(batch_patches) >= batch_size or idx == total_patches - 1:
                # 构建输入张量: (B, 1, D, H, W)
                inputs = torch.from_numpy(np.array(batch_patches)).to(device)

                # 集成推理: 对每个模型进行 TTA 推理，加权平均
                ensemble_probs = torch.zeros_like(inputs)

                for model, w in zip(models, weights):
                    if use_tta:
                        model_probs = tta_forward(model, inputs)
                    else:
                        with torch.amp.autocast('cuda'):
                            logits = model(inputs)
                            model_probs = torch.sigmoid(logits)

                    ensemble_probs += w * model_probs

                # 提取概率图: (B, D, H, W)
                probs_np = ensemble_probs.squeeze(1).cpu().numpy().astype(np.float32)

                # 高斯加权累积到全局 Volume
                for j, (bz, by, bx) in enumerate(batch_coords):
                    pred_sum[bz:bz + pd, by:by + ph, bx:bx + pw] += probs_np[j] * gaussian_np
                    weight_sum[bz:bz + pd, by:by + ph, bx:bx + pw] += gaussian_np

                # 清理
                batch_patches = []
                batch_coords = []
                pbar.update(len(probs_np))

                del inputs, ensemble_probs, probs_np
                torch.cuda.empty_cache()

    pbar.close()

    # ---- 归一化: 加权平均 ----
    print("  归一化概率图...")
    np.seterr(divide='ignore', invalid='ignore')

    prob_volume = np.zeros(volume_shape, dtype=np.float32)
    chunk_z = 16
    for z in range(0, depth, chunk_z):
        ez = min(z + chunk_z, depth)
        p = pred_sum[z:ez]
        w = weight_sum[z:ez]
        result = np.divide(p, w, out=np.zeros_like(p), where=w > 0)
        prob_volume[z:ez] = result

    # 清理临时文件
    del pred_sum, weight_sum

    return prob_volume


# ============================================================================
#  后处理
# ============================================================================

def postprocess_lcc(
    prob_volume: np.ndarray,
    threshold: float = 0.5,
    keep_n_components: int = 2,
    min_component_size: int = 500
) -> np.ndarray:
    """
    后处理流水线: 阈值二值化 + 最大连通分量 (LCC)。

    步骤:
      1. 概率图 > 阈值 -> 二值掩码
      2. 3D 连通分量标记 (26-连接)
      3. 仅保留最大的 N 个连通分量 (去除悬浮噪点)
      4. 移除小于 min_component_size 的碎片

    Args:
        prob_volume: 概率图 (D, H, W)，范围 [0, 1]
        threshold: 二值化阈值
        keep_n_components: 保留最大的 N 个连通分量
        min_component_size: 最小连通分量体素数

    Returns:
        二值掩码 (D, H, W)，值为 0 或 1
    """
    print(f"  后处理: 阈值={threshold}, 保留最大 {keep_n_components} 个连通分量")

    # 1. 二值化
    binary = (prob_volume > threshold).astype(np.uint8)
    total_voxels = binary.sum()
    print(f"    阈值化后体素数: {total_voxels:,}")

    if total_voxels == 0:
        print("    警告: 二值化后无有效体素！请检查阈值或模型质量。")
        return binary

    # 2. 3D 连通分量标记 (使用 26-连接, 即 structure=np.ones((3,3,3)))
    structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-连接
    labeled, num_features = ndimage.label(binary, structure=structure)
    print(f"    检测到 {num_features} 个连通分量")

    if num_features <= keep_n_components:
        # 连通分量数已经足够少，仅移除小碎片
        for i in range(1, num_features + 1):
            component_size = (labeled == i).sum()
            if component_size < min_component_size:
                binary[labeled == i] = 0
                print(f"    移除碎片 #{i}: {component_size:,} 体素 (< {min_component_size})")
        return binary

    # 3. 保留最大的 N 个连通分量
    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    # 按大小降序排列
    sorted_indices = np.argsort(-component_sizes)

    # 构建新的掩码
    result = np.zeros_like(binary)
    for rank, comp_idx in enumerate(sorted_indices[:keep_n_components]):
        comp_label = comp_idx + 1  # ndimage.label 从 1 开始
        comp_size = int(component_sizes[comp_idx])
        if comp_size >= min_component_size:
            result[labeled == comp_label] = 1
            print(f"    保留分量 #{rank + 1}: {comp_size:,} 体素")
        else:
            print(f"    跳过分量 #{rank + 1}: {comp_size:,} 体素 (太小)")

    final_voxels = result.sum()
    print(f"    后处理后体素数: {final_voxels:,} (移除了 {total_voxels - final_voxels:,} 噪点)")

    return result


# ============================================================================
#  RLE 编码
# ============================================================================

def rle_encode(binary_mask: np.ndarray) -> str:
    """
    将 3D 二值掩码编码为 Run-Length Encoding (RLE) 字符串。
    Kaggle 标准格式: 展平后按列主序 (Fortran order)。

    Args:
        binary_mask: (D, H, W) 二值掩码

    Returns:
        RLE 字符串 (空格分隔的 start length 对)
    """
    # 展平 (列主序, Fortran order)
    flat = binary_mask.flatten(order='F')

    # 找到起止位置
    padded = np.concatenate([[0], flat, [0]])
    diffs = np.diff(padded)

    starts = np.where(diffs == 1)[0] + 1  # 1-indexed
    ends = np.where(diffs == -1)[0] + 1

    lengths = ends - starts

    # 格式化
    rle_pairs = []
    for s, l in zip(starts, lengths):
        rle_pairs.append(f"{s} {l}")

    return " ".join(rle_pairs)


# ============================================================================
#  主函数
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description='Vesuvius Challenge 2026 - 竞赛提交推理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型推理 (仅 Pixel 模型)
  python submission.py --model_pixel best_model.pth

  # 双模型集成 + TTA
  python submission.py \\
    --model_pixel outputs/model_pixel.pth \\
    --model_topo  outputs/model_topo.pth \\
    --ensemble_weights 0.6 0.4 \\
    --use_tta
        """
    )

    # 模型路径
    parser.add_argument('--model_pixel', type=str, required=True, help='Pixel 模型权重路径 (高 Dice)')
    parser.add_argument('--model_topo', type=str, default=None, help='Topo 模型权重路径 (高 clDice)')
    parser.add_argument('--ensemble_weights', type=float, nargs=2, default=[0.6, 0.4],
                        help='集成权重: [Pixel, Topo] (默认: 0.6 0.4)')

    # 数据路径
    parser.add_argument('--data_dir', type=str,
                        default='data/vesuvius-challenge-surface-detection', help='数据根目录')
    parser.add_argument('--output_dir', type=str, default='outputs/submission', help='输出目录')

    # 推理参数
    parser.add_argument('--batch_size', type=int, default=4, help='推理 Batch Size')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 128, 128],
                        help='Patch 尺寸 (D, H, W)')
    parser.add_argument('--overlap', type=float, default=0.5, help='滑动窗口重叠率 (0.5 = 50%%)')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')

    # TTA 和后处理
    parser.add_argument('--use_tta', action='store_true', help='启用 TTA (测试时增强)')
    parser.add_argument('--keep_components', type=int, default=2, help='LCC: 保留最大的 N 个连通分量')
    parser.add_argument('--min_component_size', type=int, default=500, help='LCC: 最小连通分量体素数')

    # 输出格式
    parser.add_argument('--output_format', type=str, default='both', choices=['tif', 'rle', 'both'],
                        help='输出格式: tif / rle / both')

    # Debug 模式
    parser.add_argument('--debug', action='store_true',
                        help='Debug 模式: 仅处理少量 Patch 用于快速验证流程')
    parser.add_argument('--max_patches', type=int, default=8,
                        help='Debug 模式下最大 Patch 数 (默认: 8)')

    return parser.parse_args()


def main():
    args = get_args()
    start_time = time.time()

    print("=" * 60)
    print("  Vesuvius Challenge 2026 - 推理提交脚本")
    print("=" * 60)

    # ---- 设备配置 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[设备] {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024 ** 3
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ---- 加载模型 ----
    print(f"\n[模型加载]")
    models = []
    weights = []

    # Pixel 模型 (必须)
    print(f"  加载 Pixel 模型: {args.model_pixel}")
    model_pixel = load_model(args.model_pixel, device)
    models.append(model_pixel)

    if args.model_topo and os.path.exists(args.model_topo):
        # 双模型集成
        print(f"  加载 Topo 模型: {args.model_topo}")
        model_topo = load_model(args.model_topo, device)
        models.append(model_topo)
        weights = args.ensemble_weights
        print(f"  集成权重: Pixel={weights[0]}, Topo={weights[1]}")
    else:
        # 单模型模式
        weights = [1.0]
        print(f"  单模型模式 (无集成)")

    # ---- 计算 Stride ----
    patch_size = tuple(args.patch_size)
    stride = tuple(int(p * (1.0 - args.overlap)) for p in patch_size)

    # ---- 准备输出目录 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"

    # ---- 获取测试卷轴 ----
    test_images_dir = Path(args.data_dir) / "test_images"
    if not test_images_dir.exists():
        print(f"\n错误: 测试数据目录不存在: {test_images_dir}")
        print("请确认 --data_dir 路径正确。")
        return

    # 自动检测数据格式:
    #   格式 A: test_images/scroll_id/layers/*.tif (分层 TIF 目录)
    #   格式 B: test_images/*.tif (单个 3D TIF 文件)
    scroll_dirs = sorted([d for d in test_images_dir.iterdir() if d.is_dir()])
    tif_files = sorted([f for f in test_images_dir.iterdir() if f.suffix == '.tif'])

    # 构建卷轴任务列表: (scroll_id, layer_files_or_None, volume_data_or_None)
    scroll_tasks = []

    if scroll_dirs:
        # 格式 A: 子目录结构
        for d in scroll_dirs:
            layer_dir = d / "layers"
            if layer_dir.exists():
                lf = sorted(list(layer_dir.glob("*.tif")))
                if lf:
                    scroll_tasks.append((d.name, lf, None))
            else:
                # 可能直接包含 tif 文件
                lf = sorted(list(d.glob("*.tif")))
                if lf:
                    scroll_tasks.append((d.name, lf, None))

    if tif_files:
        # 格式 B: 直接 3D TIF 文件
        for tf in tif_files:
            scroll_tasks.append((tf.stem, None, tf))

    if not scroll_tasks:
        print(f"  未找到测试卷轴数据。")
        return

    print(f"\n[测试数据] 发现 {len(scroll_tasks)} 个卷轴: {[t[0] for t in scroll_tasks]}")

    # ---- RLE 结果收集 ----
    rle_results = []

    # ---- 逐卷轴推理 ----
    for scroll_id, layer_files, tif_path in scroll_tasks:
        print(f"\n{'=' * 40}")
        print(f"  处理卷轴: {scroll_id}")
        print(f"{'=' * 40}")

        volume_np = None

        if tif_path is not None:
            # 格式 B: 加载 3D TIF 文件
            print(f"  加载 3D TIF: {tif_path}")
            volume_np = tifffile.imread(str(tif_path))
            volume_shape = volume_np.shape
            print(f"  Volume 形状: {volume_shape}, dtype: {volume_np.dtype}")
        else:
            # 格式 A: 分层 TIF
            depth = len(layer_files)
            first_img = tifffile.imread(layer_files[0])
            height, width = first_img.shape[:2]
            volume_shape = (depth, height, width)
            print(f"  Volume 形状: {volume_shape}, dtype: {first_img.dtype}")
            del first_img

        # ---- 推理 ----
        scroll_temp = temp_dir / scroll_id
        debug_max = args.max_patches if args.debug else None
        if args.debug:
            print(f"  [DEBUG 模式] 仅处理前 {args.max_patches} 个 Patch")

        prob_volume = predict_volume_ensemble(
            models=models,
            weights=weights,
            layer_files=layer_files,
            volume_data=volume_np,
            volume_shape=volume_shape,
            patch_size=patch_size,
            stride=stride,
            device=device,
            batch_size=args.batch_size,
            use_tta=args.use_tta,
            temp_dir=scroll_temp,
            max_patches=debug_max
        )

        # 释放原始 volume
        del volume_np

        # ---- 后处理 ----
        print(f"\n  [后处理]")
        binary_mask = postprocess_lcc(
            prob_volume,
            threshold=args.threshold,
            keep_n_components=args.keep_components,
            min_component_size=args.min_component_size
        )
        del prob_volume
        gc.collect()

        # ---- 保存结果 ----
        if args.output_format in ('tif', 'both'):
            tif_path = output_dir / f"{scroll_id}.tif"
            print(f"  保存 TIF: {tif_path}")
            # 保存为 uint8 多页 TIFF (0/255)
            tifffile.imwrite(
                str(tif_path),
                (binary_mask * 255).astype(np.uint8),
                compression='zlib'
            )

        if args.output_format in ('rle', 'both'):
            rle_string = rle_encode(binary_mask)
            rle_results.append({
                'id': scroll_id,
                'rle': rle_string
            })
            print(f"  RLE 编码长度: {len(rle_string)} 字符")

        del binary_mask
        gc.collect()
        torch.cuda.empty_cache()

        # 清理临时文件
        if scroll_temp.exists():
            import shutil
            shutil.rmtree(scroll_temp, ignore_errors=True)

    # ---- 保存 RLE CSV ----
    if rle_results and args.output_format in ('rle', 'both'):
        csv_path = output_dir / "submission.csv"
        print(f"\n  保存 RLE CSV: {csv_path}")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'rle'])
            writer.writeheader()
            writer.writerows(rle_results)

    # ---- 打包提交 ----
    zip_path = output_dir / "submission.zip"
    print(f"\n  打包提交: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.iterdir():
            if file.suffix in ('.tif', '.csv') and file.name != 'submission.zip':
                zf.write(file, arcname=file.name)
                print(f"    + {file.name}")

    # ---- 总结 ----
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'=' * 60}")
    print(f"  推理完成!")
    print(f"  总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"  输出目录: {output_dir}")
    print(f"  提交文件: {zip_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
