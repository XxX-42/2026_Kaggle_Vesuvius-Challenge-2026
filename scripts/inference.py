"""
Vesuvius Inference Script (Stage 4)

主推理入口点。自动加载最佳 Checkpoint，执行滑窗预测，生成可视化和提交文件。

Usage:
    python scripts/inference.py
    
    # 手动指定 Checkpoint:
    python scripts/inference.py --checkpoint checkpoints/MiniUNETR_20260208_001652/best_model.pth

Output:
    - output/inference/prediction_raw.png: 概率图
    - output/inference/overlay.png: 预测叠加可视化
    - output/inference/submission.csv: RLE 编码提交文件
"""

import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 确保 src 在路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.mini_unetr import MiniUNETR
from src.inference.predictor import VesuviusPredictor

# ============================================================================
# 配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(checkpoint_dir="checkpoints"):
    """
    自动扫描 checkpoints 目录，找到最新的 best_model.pth
    
    搜索策略:
    1. 优先查找包含 'BEST' 的目录
    2. 其次按时间戳排序，选择最新的
    3. 在目录中查找 best_model.pth
    
    Returns:
        str: 最佳 Checkpoint 路径
        
    如果自动加载失败，请手动指定:
        python scripts/inference.py --checkpoint <路径>
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # 获取所有子目录
    subdirs = [d for d in checkpoint_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint subdirectories found in {checkpoint_dir}")
    
    # 策略 1: 查找包含 'BEST' 的目录
    best_dirs = [d for d in subdirs if 'BEST' in d.name.upper()]
    if best_dirs:
        best_dir = sorted(best_dirs)[-1]  # 取最新的
        best_path = best_dir / "best_model.pth"
        if best_path.exists():
            logger.info(f"Found BEST checkpoint: {best_path}")
            return str(best_path)
    
    # 策略 2: 按时间戳排序 (假设目录名包含时间戳 YYYYMMDD_HHMMSS)
    dated_dirs = []
    for d in subdirs:
        parts = d.name.split('_')
        if len(parts) >= 3:
            try:
                # 尝试解析时间戳
                date_str = '_'.join(parts[-2:])
                datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                dated_dirs.append(d)
            except ValueError:
                continue
    
    if dated_dirs:
        # 按目录名排序 (时间戳在名称中)
        dated_dirs.sort(key=lambda x: x.name, reverse=True)
        
        for d in dated_dirs:
            best_path = d / "best_model.pth"
            if best_path.exists():
                logger.info(f"Found latest checkpoint: {best_path}")
                return str(best_path)
            
            # 如果没有 best_model.pth，尝试 last_model.pth
            last_path = d / "last_model.pth"
            if last_path.exists():
                logger.info(f"Found latest checkpoint (last): {last_path}")
                return str(last_path)
    
    # 策略 3: 遍历所有目录找 best_model.pth
    for d in subdirs:
        best_path = d / "best_model.pth"
        if best_path.exists():
            logger.info(f"Found checkpoint: {best_path}")
            return str(best_path)
    
    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}. "
        "Please specify manually with --checkpoint <path>"
    )


def load_model(checkpoint_path, config, device='cuda'):
    """
    加载模型和权重
    
    Args:
        checkpoint_path: Checkpoint 文件路径
        config: 模型配置字典
        device: 计算设备
        
    Returns:
        model: 加载好权重的模型
    """
    logger.info(f"Loading model from: {checkpoint_path}")
    
    model_cfg = config['model']
    model = MiniUNETR(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        feature_size=model_cfg['feature_size'],
        hidden_size=model_cfg['hidden_size'],
        num_heads=model_cfg.get('num_heads', 8)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的 Checkpoint 格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 打印 Checkpoint 信息
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        logger.info(f"Checkpoint Metrics: {metrics}")
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint Epoch: {checkpoint['epoch']}")
    
    return model


def apply_morphology(binary_mask, config):
    """
    对二值化掩码应用形态学后处理
    
    Args:
        binary_mask: 二值化掩码 (H, W)，dtype=np.uint8，值为 0 或 255
        config: 配置字典，包含 inference.morphology 设置
        
    Returns:
        np.ndarray: 处理后的二值化掩码
        
    支持的操作：
        - opening: 先腐蚀后膨胀，剔除细小孤立噪点
        - closing: 先膨胀后腐蚀，填充细小空洞
        - dilate: 膨胀操作
        - erode: 腐蚀操作
    """
    morph_cfg = config.get('inference', {}).get('morphology', {})
    
    if not morph_cfg.get('enabled', False):
        return binary_mask
    
    operation = morph_cfg.get('operation', 'opening')
    kernel_size = morph_cfg.get('kernel_size', 3)
    
    # 创建结构元素 (椭圆形通常效果更好)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    
    logger.info(f"Applying morphology: {operation} with kernel {kernel_size}x{kernel_size}")
    
    if operation == 'opening':
        # Opening = Erosion followed by Dilation
        # 效果：剔除细小的孤立噪点，保持较大区域的形状
        result = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        # Closing = Dilation followed by Erosion
        # 效果：填充细小的空洞，连接相邻区域
        result = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'dilate':
        result = cv2.dilate(binary_mask, kernel)
    elif operation == 'erode':
        result = cv2.erode(binary_mask, kernel)
    else:
        logger.warning(f"Unknown morphology operation: {operation}, skipping...")
        result = binary_mask
    
    return result


def rle_encode(mask, threshold=0.5):
    """
    Run-Length 编码 (用于 Kaggle 提交)
    
    Args:
        mask: 2D NumPy 数组 (概率图)
        threshold: 二值化阈值
        
    Returns:
        str: RLE 编码字符串
    """
    binary = (mask > threshold).astype(np.uint8)
    pixels = binary.flatten()
    
    # 添加首尾 0 以处理边界
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)


def rle_encode_binary(binary_mask):
    """
    对已二值化的掩码进行 Run-Length 编码
    
    Args:
        binary_mask: 2D NumPy 数组 (二值化掩码，值为 0 或 255)
        
    Returns:
        str: RLE 编码字符串
    """
    # 将 0/255 转换为 0/1
    pixels = (binary_mask > 0).astype(np.uint8).flatten()
    
    # 添加首尾 0 以处理边界
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)


def save_visualization(prediction, fragment_path, output_dir, threshold=0.5, config=None):
    """
    保存可视化结果
    
    Args:
        prediction: 概率图 (H, W)
        fragment_path: Fragment 路径
        output_dir: 输出目录
        threshold: 二值化阈值
        config: 配置字典（用于形态学后处理）
        
    Returns:
        np.ndarray: 处理后的二值化掩码 (H, W)，值为 0 或 255
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    H, W = prediction.shape
    
    # 1. 保存原始概率图
    logger.info("Saving prediction_raw.png...")
    pred_img = (prediction * 255).astype(np.uint8)
    Image.fromarray(pred_img).save(output_dir / "prediction_raw.png")
    
    # 2. 二值化
    logger.info(f"Thresholding at {threshold}...")
    binary = (prediction > threshold).astype(np.uint8) * 255
    
    # 2.5 保存原始二值化（形态学处理前）
    Image.fromarray(binary).save(output_dir / "prediction_binary_raw.png")
    
    # 3. 应用形态学后处理
    if config is not None:
        binary = apply_morphology(binary, config)
    
    # 4. 保存处理后的二值化预测
    logger.info("Saving prediction_binary.png (after morphology)...")
    Image.fromarray(binary).save(output_dir / "prediction_binary.png")
    
    # 3. 创建 Overlay
    logger.info("Creating overlay...")
    
    # 尝试加载 IR 图像或 Mask
    ir_path = os.path.join(fragment_path, "ir.png")
    mask_path = os.path.join(fragment_path, "mask.png")
    
    if os.path.exists(ir_path):
        base_img = np.array(Image.open(ir_path).convert('L'))
        logger.info("Using ir.png as base")
    elif os.path.exists(mask_path):
        base_img = np.array(Image.open(mask_path).convert('L'))
        logger.info("Using mask.png as base")
    else:
        # 使用灰色背景
        base_img = np.full((H, W), 128, dtype=np.uint8)
        logger.info("Using gray background")
    
    # 确保尺寸匹配
    if base_img.shape != (H, W):
        base_img = cv2.resize(base_img, (W, H))
    
    # 创建 RGB Overlay
    overlay = np.stack([base_img, base_img, base_img], axis=-1)
    
    # 将预测叠加为红色通道（使用形态学处理后的二值图）
    pred_mask = binary > 0  # 使用处理后的二值图
    overlay[pred_mask, 0] = 255  # Red channel
    overlay[pred_mask, 1] = 0
    overlay[pred_mask, 2] = 0
    
    # 保存
    Image.fromarray(overlay).save(output_dir / "overlay.png")
    logger.info(f"Saved overlay to {output_dir / 'overlay.png'}")
    
    # 4. 保存带有半透明叠加的版本
    alpha = 0.5
    overlay_blend = base_img.astype(np.float32)
    overlay_blend = np.stack([overlay_blend, overlay_blend, overlay_blend], axis=-1)
    
    pred_color = np.zeros((H, W, 3), dtype=np.float32)
    pred_color[:, :, 0] = prediction * 255  # Red = probability
    
    blended = overlay_blend * (1 - alpha) + pred_color * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    Image.fromarray(blended).save(output_dir / "overlay_blend.png")
    
    return binary


def main():
    parser = argparse.ArgumentParser(description="Vesuvius Inference Pipeline")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to model checkpoint. If not specified, auto-detect best checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config"
    )
    parser.add_argument(
        "--fragment",
        type=str,
        default="1",
        help="Fragment ID to predict (1, 2, or 3)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/inference",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # 1. 加载配置
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 补充默认数据配置
    if 'data' not in config:
        config['data'] = {}
    config['data'].setdefault('z_start', 29)
    config['data'].setdefault('z_end', 44)
    
    # 2. 确定 Checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_best_checkpoint()
    
    # 3. 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    model = load_model(checkpoint_path, config, device)
    
    # 4. 创建 Predictor
    predictor = VesuviusPredictor(model, config, device)
    
    # 5. 推理
    fragment_path = os.path.join(args.data_path, "native", "train", args.fragment)
    if not os.path.exists(fragment_path):
        # 尝试其他路径格式
        fragment_path = os.path.join(args.data_path, "train", args.fragment)
    
    if not os.path.exists(fragment_path):
        raise FileNotFoundError(f"Fragment not found: {fragment_path}")
    
    logger.info(f"Predicting fragment {args.fragment} at {fragment_path}")
    prediction = predictor.predict_fragment(fragment_path, args.fragment)
    
    # 6. 保存结果
    output_dir = Path(args.output) / f"fragment_{args.fragment}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    binary_result = save_visualization(
        prediction, 
        fragment_path, 
        output_dir, 
        threshold=config['inference']['threshold'],
        config=config
    )
    
    # 7. 生成 RLE 提交（使用形态学处理后的二值图）
    logger.info("Generating submission.csv...")
    # 直接使用处理后的二值图，避免重复二值化
    rle = rle_encode_binary(binary_result)
    
    submission_path = output_dir / "submission.csv"
    with open(submission_path, 'w') as f:
        f.write("Id,Predicted\n")
        f.write(f"{args.fragment},{rle}\n")
    
    logger.info(f"Saved submission to {submission_path}")
    logger.info("Inference complete!")
    
    # 打印统计
    pred_binary = prediction > config['inference']['threshold']
    logger.info(f"Prediction Stats:")
    logger.info(f"  - Shape: {prediction.shape}")
    logger.info(f"  - Min/Max: {prediction.min():.4f} / {prediction.max():.4f}")
    logger.info(f"  - Mean: {prediction.mean():.4f}")
    logger.info(f"  - Positive Pixels: {pred_binary.sum()} ({pred_binary.mean()*100:.2f}%)")
    
    # ========================================================================
    # 8. 生成概率分布直方图 (Stage 6 诊断功能)
    # ========================================================================
    logger.info("Generating probability histogram...")
    
    # 只统计有效区域的概率（排除全黑背景）
    mask_path = os.path.join(fragment_path, "mask.png")
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32) / 255.0
        valid_probs = prediction[mask > 0.5]
    else:
        valid_probs = prediction.flatten()
    
    # 创建直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 完整直方图 (0-1)
    ax = axes[0, 0]
    ax.hist(valid_probs, bins=100, range=(0, 1), color='steelblue', edgecolor='none', alpha=0.7)
    ax.axvline(x=config['inference']['threshold'], color='red', linestyle='--', linewidth=2, 
               label=f'Threshold={config["inference"]["threshold"]:.2f}')
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title('Full Probability Distribution', fontsize=14)
    ax.legend()
    ax.set_yscale('log')  # 对数刻度便于观察
    ax.grid(True, alpha=0.3)
    
    # 2. 高概率区间 (0.4-1.0) 放大
    ax = axes[0, 1]
    high_probs = valid_probs[valid_probs > 0.4]
    if len(high_probs) > 0:
        ax.hist(high_probs, bins=60, range=(0.4, 1.0), color='darkgreen', edgecolor='none', alpha=0.7)
        ax.axvline(x=config['inference']['threshold'], color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_ylabel('Pixel Count', fontsize=12)
        ax.set_title('High Probability Zone (0.4-1.0)', fontsize=14)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No pixels > 0.4', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('High Probability Zone (0.4-1.0)', fontsize=14)
    
    # 3. 概率密度曲线 (KDE 近似)
    ax = axes[1, 0]
    hist_counts, bin_edges = np.histogram(valid_probs, bins=100, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.fill_between(bin_centers, hist_counts / hist_counts.sum(), alpha=0.5, color='purple')
    ax.plot(bin_centers, hist_counts / hist_counts.sum(), color='purple', linewidth=2)
    ax.axvline(x=config['inference']['threshold'], color='red', linestyle='--', linewidth=2,
               label=f'Threshold={config["inference"]["threshold"]:.2f}')
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Probability Density Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 阈值敏感度分析
    ax = axes[1, 1]
    thresholds = np.arange(0.5, 1.0, 0.05)
    positive_ratios = []
    for t in thresholds:
        ratio = (valid_probs > t).sum() / len(valid_probs) * 100
        positive_ratios.append(ratio)
    ax.plot(thresholds, positive_ratios, 'b-o', linewidth=2, markersize=8)
    ax.axvline(x=config['inference']['threshold'], color='red', linestyle='--', linewidth=2,
               label=f'Current={config["inference"]["threshold"]:.2f}')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Positive Ratio (%)', fontsize=12)
    ax.set_title('Threshold Sensitivity Analysis', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Probability Distribution Analysis - Fragment {args.fragment}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    hist_path = output_dir / "prob_histogram.png"
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved probability histogram to {hist_path}")
    
    # 打印诊断摘要
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    p50 = np.percentile(valid_probs, 50)
    p75 = np.percentile(valid_probs, 75)
    p90 = np.percentile(valid_probs, 90)
    p95 = np.percentile(valid_probs, 95)
    logger.info(f"  Percentiles: P50={p50:.4f}, P75={p75:.4f}, P90={p90:.4f}, P95={p95:.4f}")
    logger.info(f"  Pixels > 0.5: {(valid_probs > 0.5).sum()} ({(valid_probs > 0.5).mean()*100:.2f}%)")
    logger.info(f"  Pixels > 0.7: {(valid_probs > 0.7).sum()} ({(valid_probs > 0.7).mean()*100:.2f}%)")
    logger.info(f"  Pixels > 0.8: {(valid_probs > 0.8).sum()} ({(valid_probs > 0.8).mean()*100:.2f}%)")
    logger.info(f"  Pixels > 0.9: {(valid_probs > 0.9).sum()} ({(valid_probs > 0.9).mean()*100:.2f}%)")
    logger.info("=" * 60)
    
    # 根据分布给出建议
    if p90 < 0.6:
        logger.warning("⚠️ 模型输出概率普遍偏低 (P90 < 0.6)，可能需要更多训练或调整模型")
    elif p75 > 0.7:
        logger.info("✅ 模型输出有明显的高置信区域，阈值 0.7+ 应能有效分离")
    else:
        logger.info("📊 模型输出中等偏上，建议尝试 0.6-0.75 的阈值范围")


if __name__ == "__main__":
    main()
