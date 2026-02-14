"""
Vesuvius Challenge - GPU 推理脚本 v2

核心改进：
- Sliding Window 分块推理：避免 320³ 体积直接送入 GPU 导致 OOM
- 实时进度监控：GPU 显存、处理速度、ETA
- 规范化输出目录命名：chimera_run_{日期}_{模式}

用法:
    python 20_src/run_inference.py --max_chunks 5
    python 20_src/run_inference.py --checkpoint path/to/model.pth
"""

import os
import sys
import time
import argparse
import psutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
import tifffile

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from importlib import import_module

model_mod = import_module("20_src.20_model.dual_unet")
DualHeadResUNet3D = model_mod.DualHeadResUNet3D


# ===== 监控工具 =====

def get_gpu_stats():
    """获取 GPU 显存使用信息"""
    if not torch.cuda.is_available():
        return "CPU 模式"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"GPU: {allocated:.1f}G/{total:.1f}G (reserved {reserved:.1f}G)"


def get_ram_stats():
    """获取 RAM 使用信息"""
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1024**3
    total_gb = mem.total / 1024**3
    return f"RAM: {used_gb:.1f}G/{total_gb:.1f}G ({mem.percent}%)"


def format_eta(seconds):
    """格式化剩余时间"""
    if seconds < 0:
        return "计算中..."
    td = timedelta(seconds=int(seconds))
    return str(td)


def print_progress(current, total, chunk_id, stage, extra=""):
    """打印实时进度"""
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{bar}] {current}/{total} ({pct:.0f}%) | ID:{chunk_id} | {stage}"
    if extra:
        line += f" | {extra}"
    print(line, flush=True)


# ===== 分块推理引擎 =====

def sliding_window_inference(
    model,
    volume: torch.Tensor,
    patch_size: tuple = (64, 128, 128),
    overlap: int = 8,
    device: torch.device = None,
):
    """
    Sliding Window 分块推理

    将大体积切成小 patch，逐个送入 GPU 推理，再拼回来。
    重叠区域使用平均融合。

    Args:
        model: DualHeadResUNet3D
        volume: (1, 1, D, H, W) 完整输入体积
        patch_size: 每个 patch 的大小 (pD, pH, pW)
        overlap: patch 之间的重叠体素数
        device: 推理设备

    Returns:
        seg_prob: (D, H, W) 分割概率图
        normal_map: (3, D, H, W) 法线图
    """
    _, _, D, H, W = volume.shape
    pD, pH, pW = patch_size
    stride_d = pD - overlap
    stride_h = pH - overlap
    stride_w = pW - overlap

    # 输出累积器
    seg_sum = torch.zeros(1, 1, D, H, W, dtype=torch.float32)
    normal_sum = torch.zeros(1, 3, D, H, W, dtype=torch.float32)
    count = torch.zeros(1, 1, D, H, W, dtype=torch.float32)

    # 计算所有 patch 的起始位置
    d_starts = list(range(0, max(D - pD + 1, 1), stride_d))
    h_starts = list(range(0, max(H - pH + 1, 1), stride_h))
    w_starts = list(range(0, max(W - pW + 1, 1), stride_w))

    # 确保覆盖边界
    if d_starts[-1] + pD < D:
        d_starts.append(D - pD)
    if h_starts[-1] + pH < H:
        h_starts.append(H - pH)
    if w_starts[-1] + pW < W:
        w_starts.append(W - pW)

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    patch_idx = 0

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                patch_idx += 1

                # 提取 patch
                d1 = min(d0 + pD, D)
                h1 = min(h0 + pH, H)
                w1 = min(w0 + pW, W)

                patch = volume[:, :, d0:d1, h0:h1, w0:w1]

                # Pad 如果尺寸不够
                actual_d, actual_h, actual_w = d1 - d0, h1 - h0, w1 - w0
                if actual_d < pD or actual_h < pH or actual_w < pW:
                    pad_d = pD - actual_d
                    pad_h = pH - actual_h
                    pad_w = pW - actual_w
                    patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d))

                # GPU 推理
                patch_gpu = patch.to(device)
                seg_logits, normals = model(patch_gpu)
                seg_prob = torch.sigmoid(seg_logits)

                # 裁剪回实际尺寸
                seg_prob = seg_prob[:, :, :actual_d, :actual_h, :actual_w].cpu()
                normals = normals[:, :, :actual_d, :actual_h, :actual_w].cpu()

                # 累加
                seg_sum[:, :, d0:d1, h0:h1, w0:w1] += seg_prob
                normal_sum[:, :, d0:d1, h0:h1, w0:w1] += normals
                count[:, :, d0:d1, h0:h1, w0:w1] += 1.0

                # 清理 GPU 显存
                del patch_gpu, seg_logits, seg_prob, normals
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # 每 5 个 patch 打印一次进度
                if patch_idx % 5 == 0 or patch_idx == total_patches:
                    print(f"    patch {patch_idx}/{total_patches} | {get_gpu_stats()}", flush=True)

    # 平均融合
    count = count.clamp(min=1.0)
    seg_avg = (seg_sum / count).squeeze().numpy()       # (D, H, W)
    normal_avg = (normal_sum / count).squeeze(0).numpy() # (3, D, H, W)

    return seg_avg, normal_avg


# ===== 主推理函数 =====

def run_inference(args):
    """主推理流程"""

    # 设备
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    # 规范化输出目录命名: chimera_run_{日期}_{模式}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "gpu" if dev.type == "cuda" else "cpu"
    run_name = f"chimera_run_{timestamp}_{mode_tag}_nf{args.n_filters}"
    run_output_dir = Path(args.output_dir) / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  🔬 Hybrid Chimera MVP - 推理")
    print(f"  设备: {dev} | {get_gpu_stats()}")
    print(f"  {get_ram_stats()}")
    print(f"  Patch 大小: {args.patch_size}")
    print(f"  输入: {args.input_dir}")
    print(f"  输出: {run_output_dir}")
    print(f"{'='*70}\n")

    # 加载模型
    model = DualHeadResUNet3D(in_channels=1, n_filters=args.n_filters)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=dev, weights_only=True)
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"[Model] 已加载权重: {args.checkpoint}")
    else:
        print("[Model] 使用随机初始化权重（MVP 演示模式）")

    model.to(dev)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] 参数量: {params:,} | {get_gpu_stats()}\n")

    # 扫描输入
    input_path = Path(args.input_dir)
    tif_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in ('.tif', '.tiff')
    ])
    if args.max_chunks:
        tif_files = tif_files[:args.max_chunks]

    total = len(tif_files)
    print(f"[Data] {total} 个 .tif 文件待处理\n")

    # 推理循环
    all_times = []
    normal_stats = []

    with torch.no_grad():
        for idx, tif_file in enumerate(tif_files):
            chunk_id = tif_file.stem
            print_progress(idx + 1, total, chunk_id, "开始加载")

            t_start = time.time()

            # 1. 加载
            t0 = time.time()
            volume_raw = tifffile.imread(str(tif_file))
            vol_shape = volume_raw.shape
            volume = volume_raw.astype(np.float32)
            if volume.max() > 1.0:
                volume = volume / 255.0 if volume.max() <= 255.0 else volume / 65535.0
            volume = np.clip(volume, 0.0, 1.0)
            x = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
            t_load = time.time() - t0

            print(f"    形状: {vol_shape} | 加载: {t_load:.2f}s | {get_ram_stats()}", flush=True)

            # 2. Sliding Window 推理
            t0 = time.time()
            patch_size = tuple(args.patch_size)
            seg_prob, normal_map = sliding_window_inference(
                model, x, patch_size=patch_size,
                overlap=args.overlap, device=dev,
            )
            t_unet = time.time() - t0

            # 3. 法向量质量诊断
            # 计算法线模长 (应接近 1.0) 和方向一致性
            norm_magnitude = np.linalg.norm(normal_map, axis=0)  # (D,H,W)
            mask_region = seg_prob > 0.3  # 只在有意义的区域统计
            if mask_region.sum() > 0:
                avg_norm = norm_magnitude[mask_region].mean()
                std_norm = norm_magnitude[mask_region].std()
                # 方向一致性: 邻居法线点积的均值
                normal_stats.append({
                    "id": chunk_id,
                    "avg_norm_magnitude": float(avg_norm),
                    "std_norm_magnitude": float(std_norm),
                })
                norm_diag = f"法线模长: {avg_norm:.3f}±{std_norm:.3f}"
            else:
                norm_diag = "法线: 无有效区域"

            # 4. 阈值化生成 mask
            t0 = time.time()
            final_mask = (seg_prob > 0.5).astype(np.uint8)
            t_post = time.time() - t0

            t_total = time.time() - t_start
            all_times.append(t_total)

            # 5. 保存
            output_filename = f"{chunk_id}_mask.tif"
            output_path = run_output_dir / output_filename
            tifffile.imwrite(str(output_path), final_mask)

            # 计算 ETA
            avg_time = sum(all_times) / len(all_times)
            eta = avg_time * (total - idx - 1)
            mask_pct = final_mask.sum() / final_mask.size * 100

            print(f"    U-Net: {t_unet:.1f}s | 后处理: {t_post:.3f}s | "
                  f"总计: {t_total:.1f}s", flush=True)
            print(f"    Mask: {mask_pct:.1f}% | {norm_diag}", flush=True)
            print(f"    {get_gpu_stats()} | {get_ram_stats()}", flush=True)
            print(f"    ETA: {format_eta(eta)} | 平均: {avg_time:.1f}s/chunk", flush=True)
            print(f"    → {output_path.name}", flush=True)
            print("", flush=True)

            # 清理
            del x, volume, volume_raw
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    # ===== 最终报告 =====
    total_time = sum(all_times)
    avg_time = total_time / max(len(all_times), 1)

    print(f"\n{'='*70}")
    print(f"  推理完成!")
    print(f"  处理: {len(all_times)} chunks")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  平均: {avg_time:.1f}s/chunk")
    print(f"  输出: {run_output_dir}")

    # 法向量质量诊断汇总
    if normal_stats:
        avg_mag = np.mean([s['avg_norm_magnitude'] for s in normal_stats])
        avg_std = np.mean([s['std_norm_magnitude'] for s in normal_stats])
        print(f"\n  📐 法向量诊断:")
        print(f"    平均模长: {avg_mag:.3f} (理想=1.0)")
        print(f"    模长标准差: {avg_std:.3f} (越低越好)")
        if avg_mag < 0.5:
            print(f"    ⚠️ 法线方向杂乱，建议加强 L_Cosine 权重 (当前 0.1 → 0.5)")

    # 内存诊断
    print(f"\n  💾 资源诊断:")
    print(f"    {get_ram_stats()}")
    print(f"    {get_gpu_stats()}")
    ram_gb = psutil.virtual_memory().used / 1024**3
    if ram_gb > 16:
        print(f"    ⚠️ RAM 使用超过 16GB，建议加入 supervoxel 降采样")

    # 时间诊断
    print(f"\n  ⏱️ 性能诊断:")
    print(f"    单 chunk 平均: {avg_time:.1f}s")
    if avg_time > 900:  # 15 分钟
        print(f"    ⚠️ 单 chunk > 15 min，必须切换到 CuPy GPU 求解器")
    else:
        print(f"    ✓ 单 chunk 时间可接受")

    estimated_full = avg_time * 786 / 60
    print(f"    预估全量推理: {estimated_full:.0f} min ({estimated_full/60:.1f} h)")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Chimera - GPU 推理 v2")
    parser.add_argument("--input_dir", type=str,
                        default="data/vesuvius-challenge-surface-detection/train_images")
    parser.add_argument("--output_dir", type=str, default="20_src/output")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_filters", type=int, default=16)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128],
                        help="Sliding window patch 大小 (D H W)")
    parser.add_argument("--overlap", type=int, default=8,
                        help="Patch 重叠体素数")

    args = parser.parse_args()
    run_inference(args)
