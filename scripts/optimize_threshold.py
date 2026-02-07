"""
动态阈值优化脚本 (Stage 6: Precision & Artifact Suppression)

扫描不同阈值下的 Precision/Recall，找到让"文字线条最细且不断裂"的黄金阈值。

Usage:
    python scripts/optimize_threshold.py
    
    # 指定概率图路径:
    python scripts/optimize_threshold.py --pred output/inference/fragment_1/prediction_raw.png
    
    # 指定搜索范围:
    python scripts/optimize_threshold.py --start 0.5 --end 0.95 --step 0.05

Output:
    - output/threshold_search/threshold_comparison.png: 多阈值对比图
    - output/threshold_search/metrics.csv: 各阈值的指标统计
    - output/threshold_search/best_threshold.txt: 推荐阈值
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging

# 确保 src 在路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prediction(pred_path):
    """加载概率图"""
    logger.info(f"Loading prediction from: {pred_path}")
    img = Image.open(pred_path).convert('L')
    pred = np.array(img).astype(np.float32) / 255.0
    logger.info(f"Prediction shape: {pred.shape}, range: [{pred.min():.4f}, {pred.max():.4f}]")
    return pred


def load_ground_truth(gt_path):
    """加载 Ground Truth (inklabels.png)"""
    if not os.path.exists(gt_path):
        logger.warning(f"Ground truth not found: {gt_path}")
        return None
    
    logger.info(f"Loading ground truth from: {gt_path}")
    img = Image.open(gt_path).convert('L')
    gt = np.array(img).astype(np.float32) / 255.0
    gt = (gt > 0.5).astype(np.float32)  # 二值化
    logger.info(f"Ground truth shape: {gt.shape}, positive ratio: {gt.mean()*100:.2f}%")
    return gt


def apply_morphology(binary, kernel_size=3, operation='opening'):
    """
    应用形态学后处理
    
    Args:
        binary: 二值化图像 (0-255 或 0-1)
        kernel_size: 结构元素大小
        operation: 'opening' (去噪) 或 'closing' (填充)
    
    Returns:
        处理后的二值化图像
    """
    # 确保是 0-255 范围
    if binary.max() <= 1:
        binary = (binary * 255).astype(np.uint8)
    else:
        binary = binary.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'opening':
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        result = binary
    
    return result


def compute_metrics(pred_binary, gt, mask=None):
    """
    计算 Precision, Recall, F1, Dice, IoU
    
    Args:
        pred_binary: 二值化预测 (0-1 或 0-255)
        gt: Ground Truth (0-1)
        mask: 可选的有效区域 mask
        
    Returns:
        dict: 各项指标
    """
    # 归一化到 0-1
    if pred_binary.max() > 1:
        pred = (pred_binary > 127).astype(np.float32)
    else:
        pred = (pred_binary > 0.5).astype(np.float32)
    
    # 应用 mask
    if mask is not None:
        pred = pred * mask
        gt = gt * mask
    
    # 计算 TP, FP, FN
    intersection = np.sum(pred * gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)
    union = pred_sum + gt_sum - intersection
    
    # 避免除零
    eps = 1e-6
    
    precision = intersection / (pred_sum + eps)
    recall = intersection / (gt_sum + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    dice = 2 * intersection / (pred_sum + gt_sum + eps)
    iou = intersection / (union + eps)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'dice': dice,
        'iou': iou,
        'positive_pixels': int(pred_sum),
        'positive_ratio': pred_sum / (pred.size + eps) * 100
    }


def analyze_threshold(pred, threshold, kernel_size=3, gt=None, mask=None):
    """
    分析单个阈值的效果
    
    Args:
        pred: 概率图 (0-1)
        threshold: 阈值
        kernel_size: 形态学核大小
        gt: 可选的 Ground Truth
        mask: 可选的有效区域 mask
        
    Returns:
        dict: 分析结果
    """
    # 二值化
    binary = (pred > threshold).astype(np.uint8) * 255
    
    # 形态学清洗
    cleaned = apply_morphology(binary, kernel_size, 'opening')
    
    # 计算统计信息
    result = {
        'threshold': threshold,
        'raw_positive_pixels': int(np.sum(binary > 0)),
        'raw_positive_ratio': np.sum(binary > 0) / binary.size * 100,
        'cleaned_positive_pixels': int(np.sum(cleaned > 0)),
        'cleaned_positive_ratio': np.sum(cleaned > 0) / cleaned.size * 100,
        'noise_removed': int(np.sum(binary > 0) - np.sum(cleaned > 0)),
    }
    
    # 如果有 Ground Truth，计算指标
    if gt is not None:
        raw_metrics = compute_metrics(binary, gt, mask)
        cleaned_metrics = compute_metrics(cleaned, gt, mask)
        
        result.update({
            'raw_precision': raw_metrics['precision'],
            'raw_recall': raw_metrics['recall'],
            'raw_f1': raw_metrics['f1'],
            'raw_dice': raw_metrics['dice'],
            'cleaned_precision': cleaned_metrics['precision'],
            'cleaned_recall': cleaned_metrics['recall'],
            'cleaned_f1': cleaned_metrics['f1'],
            'cleaned_dice': cleaned_metrics['dice'],
        })
    
    return result, binary, cleaned


def plot_threshold_comparison(pred, thresholds, output_dir, gt=None):
    """
    生成多阈值对比可视化
    """
    n_cols = 4
    n_rows = (len(thresholds) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for i, thresh in enumerate(thresholds):
        if i >= len(axes):
            break
            
        binary = (pred > thresh).astype(np.uint8) * 255
        cleaned = apply_morphology(binary, 3, 'opening')
        
        axes[i].imshow(cleaned, cmap='gray')
        ratio = np.sum(cleaned > 0) / cleaned.size * 100
        axes[i].set_title(f'Threshold={thresh:.2f}\nPositive={ratio:.2f}%')
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(thresholds), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_comparison.png', dpi=150)
    plt.close()
    logger.info(f"Saved threshold comparison to {output_dir / 'threshold_comparison.png'}")


def plot_metrics_curve(results_df, output_dir):
    """
    绘制阈值-指标曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Positive Ratio vs Threshold
    ax = axes[0, 0]
    ax.plot(results_df['threshold'], results_df['raw_positive_ratio'], 'b-o', label='Raw')
    ax.plot(results_df['threshold'], results_df['cleaned_positive_ratio'], 'g-o', label='After Morphology')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Positive Ratio (%)')
    ax.set_title('Positive Pixel Ratio vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Noise Removed
    ax = axes[0, 1]
    ax.bar(results_df['threshold'].astype(str), results_df['noise_removed'], color='orange')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Noise Pixels Removed')
    ax.set_title('Noise Removal by Morphology')
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Precision/Recall (如果有 GT)
    ax = axes[1, 0]
    if 'cleaned_precision' in results_df.columns:
        ax.plot(results_df['threshold'], results_df['cleaned_precision'], 'b-o', label='Precision')
        ax.plot(results_df['threshold'], results_df['cleaned_recall'], 'r-o', label='Recall')
        ax.plot(results_df['threshold'], results_df['cleaned_f1'], 'g-o', label='F1')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision/Recall/F1 vs Threshold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No Ground Truth\nAvailable', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title('(No GT for Metrics)')
    ax.grid(True, alpha=0.3)
    
    # 4. Dice Score (如果有 GT)
    ax = axes[1, 1]
    if 'cleaned_dice' in results_df.columns:
        ax.plot(results_df['threshold'], results_df['cleaned_dice'], 'purple', marker='o', linewidth=2)
        
        # 标记最佳阈值
        best_idx = results_df['cleaned_dice'].idxmax()
        best_thresh = results_df.loc[best_idx, 'threshold']
        best_dice = results_df.loc[best_idx, 'cleaned_dice']
        ax.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.7)
        ax.annotate(f'Best: {best_thresh:.2f}\nDice: {best_dice:.4f}', 
                   xy=(best_thresh, best_dice), xytext=(10, -30),
                   textcoords='offset points', fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Dice Score')
        ax.set_title('Dice Score vs Threshold')
    else:
        # 没有 GT 时，使用 heuristic 找最佳阈值
        # 策略：找到正样本比例的"拐点"（变化率最大的点）
        ratios = results_df['cleaned_positive_ratio'].values
        gradients = np.abs(np.gradient(ratios))
        
        ax.plot(results_df['threshold'], gradients, 'purple', marker='o', linewidth=2)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('|d(Positive Ratio)/d(Threshold)|')
        ax.set_title('Rate of Change (for Elbow Detection)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_curve.png', dpi=150)
    plt.close()
    logger.info(f"Saved metrics curve to {output_dir / 'metrics_curve.png'}")


def find_best_threshold(results_df):
    """
    找到最佳阈值
    
    策略：
    1. 如果有 GT: 选择 Dice 最高的阈值
    2. 如果没有 GT: 使用"肘部法则"找正样本比例的拐点
    """
    if 'cleaned_dice' in results_df.columns:
        best_idx = results_df['cleaned_dice'].idxmax()
        best_thresh = results_df.loc[best_idx, 'threshold']
        best_metric = results_df.loc[best_idx, 'cleaned_dice']
        method = 'Dice Score'
    else:
        # 肘部法则：找变化率最大的点
        ratios = results_df['cleaned_positive_ratio'].values
        thresholds = results_df['threshold'].values
        
        # 计算二阶导数（曲率）
        second_derivative = np.abs(np.gradient(np.gradient(ratios)))
        best_idx = np.argmax(second_derivative[1:-1]) + 1  # 避开边界
        
        best_thresh = thresholds[best_idx]
        best_metric = ratios[best_idx]
        method = 'Elbow Detection (no GT)'
    
    return best_thresh, best_metric, method


def main():
    parser = argparse.ArgumentParser(description="Dynamic Threshold Optimization")
    parser.add_argument(
        "--pred",
        type=str,
        default="output/inference/fragment_1/prediction_raw.png",
        help="Path to prediction probability map"
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Path to ground truth (inklabels.png). If not specified, will try to auto-detect."
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to mask.png for valid region"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.5,
        help="Start threshold for search"
    )
    parser.add_argument(
        "--end",
        type=float,
        default=0.95,
        help="End threshold for search"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Step size for threshold search"
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=3,
        help="Morphology kernel size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/threshold_search",
        help="Output directory"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/vesuvius-challenge-ink-detection",
        help="Path to data directory (for auto-detecting GT)"
    )
    parser.add_argument(
        "--fragment",
        type=str,
        default="1",
        help="Fragment ID (for auto-detecting GT)"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载预测
    pred = load_prediction(args.pred)
    
    # 尝试加载 Ground Truth
    gt = None
    mask = None
    
    if args.gt:
        gt = load_ground_truth(args.gt)
    else:
        # 自动检测
        auto_gt_path = os.path.join(args.data_path, "train", args.fragment, "inklabels.png")
        if os.path.exists(auto_gt_path):
            gt = load_ground_truth(auto_gt_path)
            logger.info(f"Auto-detected ground truth at: {auto_gt_path}")
    
    if args.mask:
        mask_img = Image.open(args.mask).convert('L')
        mask = (np.array(mask_img).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        logger.info(f"Loaded mask from: {args.mask}")
    
    # 生成阈值列表
    thresholds = np.arange(args.start, args.end + args.step/2, args.step)
    logger.info(f"Searching thresholds: {thresholds}")
    
    # 分析每个阈值
    results = []
    for thresh in thresholds:
        result, _, _ = analyze_threshold(pred, thresh, args.kernel, gt, mask)
        results.append(result)
        logger.info(f"Threshold {thresh:.2f}: Positive={result['cleaned_positive_ratio']:.2f}%")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics.csv', index=False)
    logger.info(f"Saved metrics to {output_dir / 'metrics.csv'}")
    
    # 生成可视化
    plot_threshold_comparison(pred, thresholds, output_dir, gt)
    plot_metrics_curve(results_df, output_dir)
    
    # 找到最佳阈值
    best_thresh, best_metric, method = find_best_threshold(results_df)
    
    # 保存最佳阈值
    with open(output_dir / 'best_threshold.txt', 'w') as f:
        f.write(f"Best Threshold: {best_thresh:.2f}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Metric Value: {best_metric:.4f}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best Threshold: {best_thresh:.2f}")
    logger.info(f"Method: {method}")
    logger.info(f"Metric Value: {best_metric:.4f}")
    logger.info(f"{'='*60}")
    
    # 生成最佳阈值的二值化图
    binary = (pred > best_thresh).astype(np.uint8) * 255
    cleaned = apply_morphology(binary, args.kernel, 'opening')
    Image.fromarray(cleaned).save(output_dir / 'best_binary.png')
    logger.info(f"Saved best binary result to {output_dir / 'best_binary.png'}")
    
    print(f"\n推荐阈值: {best_thresh:.2f}")
    print(f"请将此阈值更新到 configs/inference.yaml 的 threshold 字段")


if __name__ == "__main__":
    main()
