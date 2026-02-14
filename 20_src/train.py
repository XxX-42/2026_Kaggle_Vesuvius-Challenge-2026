"""
Vesuvius Challenge - Hybrid Chimera 训练引擎 (Phase 5)

Patch-based 3D 训练循环，支持：
- AMP 混合精度 (FP16)
- Random Crop 3D + 数据增强
- ChimeraLoss (Dice + Normal Cosine)
- tqdm 进度条
- 每 epoch 可视化输出 (PNG 对比图 + TIF mask)
- 验证 Dice 监控 + Best Model 保存

用法:
    # 快速测试 (5 个 chunk, 2 个 epoch)
    python 20_src/train.py --max_chunks 5 --epochs 2

    # 完整训练
    python 20_src/train.py --epochs 50 --batch_size 4 --lr 1e-3

    # 从 checkpoint 恢复
    python 20_src/train.py --resume 20_src/output/best_model.pth --epochs 50
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import tifffile

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from importlib import import_module

# 模型
model_mod = import_module("20_src.20_model.dual_unet")
DualHeadResUNet3D = model_mod.DualHeadResUNet3D

# 损失函数
loss_mod = import_module("20_src.20_model.chimera_loss")
ChimeraLoss = loss_mod.ChimeraLoss

# 数据集
dataset_mod = import_module("20_src.20_data.dataset")
VesuviusTrainDataset = dataset_mod.VesuviusTrainDataset

# 变换
transforms_mod = import_module("20_src.20_data.transforms")
RandomCrop3D = transforms_mod.RandomCrop3D
RandomFlipRotate3D = transforms_mod.RandomFlipRotate3D
Compose3D = transforms_mod.Compose3D


# ===== 工具函数 =====

def get_gpu_stats():
    """获取 GPU 显存使用信息"""
    if not torch.cuda.is_available():
        return "CPU"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"GPU:{reserved:.1f}G/{total:.1f}G"


def compute_dice(pred_logits, targets, threshold=0.5):
    """计算 Dice 系数（评估指标，per-sample 平均）"""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    smooth = 1e-6
    # per-sample 计算，避免空 patch 稀释全局 Dice
    pred_flat = pred.view(pred.size(0), -1)
    tgt_flat = targets.view(targets.size(0), -1)
    intersection = (pred_flat * tgt_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + tgt_flat.sum(dim=1) + smooth)
    return dice.mean().item()


def format_time(seconds):
    """格式化时间"""
    td = timedelta(seconds=int(seconds))
    return str(td)


# ===== 可视化工具 =====

def save_epoch_visualization(
    model, val_loader, device, run_dir, epoch, criterion
):
    """
    每个 epoch 结束后保存可视化对比：
    1. PNG 对比图：中间 slice 的 image / GT / prediction 三列对比
    2. TIF mask：预测结果 3D volume
    """
    model.eval()

    # 从验证集寻找一个包含正样本的 batch 进行可视化
    target_images = None
    target_labels = None
    
    try:
        # 要求中间切片上有足够的 GT 像素，否则可视化看不到任何东西
        for images, labels in val_loader:
            mid_z = images.shape[2] // 2
            mid_slice_gt = labels[:, :, mid_z, :, :].sum()
            if mid_slice_gt > 50:  # 至少 50 个正样本像素
                target_images = images
                target_labels = labels
                break
        
        # 如果没找到（极其罕见），就退回到第一个 batch
        if target_images is None:
            target_images, target_labels = next(iter(val_loader))
            
    except StopIteration:
        return

    images = target_images.to(device)
    labels = target_labels.to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            seg_logits, normals = model(images)
    
    # 取第一个样本
    pred_prob = torch.sigmoid(seg_logits[0, 0]).cpu().numpy()    # (D, H, W)
    pred_mask = (pred_prob > 0.5).astype(np.uint8)               # 二值 mask
    gt_mask = labels[0, 0].cpu().numpy()                          # (D, H, W)
    img_vol = images[0, 0].cpu().numpy()                          # (D, H, W)

    D, H, W = img_vol.shape

    # === 1. 保存 TIF mask ===
    tif_dir = run_dir / "epoch_masks"
    tif_dir.mkdir(exist_ok=True)
    tif_path = tif_dir / f"epoch{epoch+1:03d}_pred_mask.tif"
    tifffile.imwrite(str(tif_path), pred_mask)

    # === 2. 保存 PNG 对比图 ===
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt

        vis_dir = run_dir / "epoch_vis"
        vis_dir.mkdir(exist_ok=True)

        # 取 3 个正交 slice（中间位置）
        slices = {
            'Axial (z-mid)': (img_vol[D//2], gt_mask[D//2], pred_prob[D//2], pred_mask[D//2]),
            'Coronal (y-mid)': (img_vol[:, H//2], gt_mask[:, H//2], pred_prob[:, H//2], pred_mask[:, H//2]),
            'Sagittal (x-mid)': (img_vol[:, :, W//2], gt_mask[:, :, W//2], pred_prob[:, :, W//2], pred_mask[:, :, W//2]),
        }

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Epoch {epoch+1} | Dice: {compute_dice(seg_logits[0:1], labels[0:1]):.4f}',
                     fontsize=16, fontweight='bold')

        for row_idx, (plane_name, (img_s, gt_s, prob_s, mask_s)) in enumerate(slices.items()):
            # 列 1: Input CT
            axes[row_idx, 0].imshow(img_s, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 0].set_title(f'{plane_name}\nInput CT')
            axes[row_idx, 0].axis('off')

            # 列 2: Ground Truth
            axes[row_idx, 1].imshow(gt_s, cmap='Reds', vmin=0, vmax=1, alpha=0.8)
            axes[row_idx, 1].set_title('Ground Truth')
            axes[row_idx, 1].axis('off')

            # 列 3: Prediction (概率图)
            axes[row_idx, 2].imshow(prob_s, cmap='hot', vmin=0, vmax=1)
            axes[row_idx, 2].set_title('Pred Prob')
            axes[row_idx, 2].axis('off')

            # 列 4: Overlay (CT + Prediction 叠加)
            axes[row_idx, 3].imshow(img_s, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 3].imshow(mask_s, cmap='Reds', alpha=0.4)
            axes[row_idx, 3].set_title('Overlay')
            axes[row_idx, 3].axis('off')

        plt.tight_layout()
        png_path = vis_dir / f"epoch{epoch+1:03d}_comparison.png"
        plt.savefig(str(png_path), dpi=120, bbox_inches='tight')
        plt.close(fig)

        print(f"  📸 2D 对比: {png_path.name} | 🗂️ Mask: {tif_path.name}")

    except ImportError:
        print(f"  🗂️ Mask TIF: {tif_path.name} (matplotlib 不可用，跳过 PNG)")

    # === 3. 导出独立 3D 预览文件 (双击直接打开) ===
    try:
        import pyvista as pv

        vis_dir = run_dir / "epoch_vis"
        vis_dir.mkdir(exist_ok=True)

        if pred_mask.sum() > 0:
            # 1. 包装体素并提取几何网格
            grid = pv.wrap(pred_mask.astype(np.float32))
            mesh = grid.contour([0.5])

            # 2. 构建离屏渲染场景
            p = pv.Plotter(off_screen=True)
            p.add_mesh(mesh, color='red', opacity=0.8, show_edges=False)

            # 垫入 GT 作为幽灵轮廓对比
            if gt_mask.sum() > 0:
                gt_grid = pv.wrap(gt_mask.astype(np.float32))
                gt_mesh = gt_grid.contour([0.5])
                p.add_mesh(gt_mesh, color='green', opacity=0.15, show_edges=False)

            # --- 核心导出逻辑 ---

            # 方案 A: 独立 HTML (双击用浏览器打开，纯净底色)
            html_path = vis_dir / f"epoch{epoch+1:03d}_pred.html"
            p.export_html(str(html_path))
            print(f"  🌐 独立 3D HTML 已导出: {html_path.name}")

            # 方案 B: GLTF 模型 (双击用 Windows/Mac 自带 3D 软件打开)
            gltf_path = vis_dir / f"epoch{epoch+1:03d}_pred.gltf"
            p.export_gltf(str(gltf_path))
            print(f"  🧊 独立 3D GLTF 已导出: {gltf_path.name}")

            p.close()
        else:
            print(f"  ⚠️ Epoch {epoch+1} 预测为全 0，无物理实体可导出。")

    except Exception as e:
        print(f"  ⚠️ 3D 导出失败: {e}")



# ===== 训练循环 =====

def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs
):
    """训练一个 epoch（带 tqdm 进度条）"""
    model.train()

    total_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    total_normal_loss = 0.0
    total_dice_score = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Train E{epoch+1}/{total_epochs}",
        ncols=120,
        leave=True,
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)  # (B, 1, D, H, W)
        labels = labels.to(device, non_blocking=True)  # (B, 1, D, H, W)

        optimizer.zero_grad(set_to_none=True)

        # AMP 前向传播
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            seg_logits, normals = model(images)
            loss_total, loss_tversky, loss_bce, loss_normal = criterion(seg_logits, normals, labels)

        # AMP 反向传播
        if device.type == 'cuda':
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 统计
        dice_score = compute_dice(seg_logits.detach(), labels)
        total_loss += loss_total.item()
        total_dice_loss += loss_tversky.item()
        total_bce_loss += loss_bce.item()
        total_normal_loss += loss_normal.item()
        total_dice_score += dice_score
        num_batches += 1

        # Debug "Playing Dead"
        if num_batches % 50 == 0:
            pred_sum = (torch.sigmoid(seg_logits) > 0.5).float().sum()
            target_sum = labels.sum()
            print(f"\n[DEBUG] Batch {num_batches}: Pred_Pixels={pred_sum.item()}, GT_Pixels={target_sum.item()}")

        # 更新 tqdm
        avg_loss = total_loss / num_batches
        avg_dice = total_dice_score / num_batches
        pbar.set_postfix({
            'loss': f'{avg_loss:.2f}',
            'tvsk': f'{total_dice_loss/num_batches:.2f}',
            'dice': f'{avg_dice:.2f}',
            'norm': f'{total_normal_loss/num_batches:.2f}',
            'gpu': get_gpu_stats(),
        })

    pbar.close()

    # epoch 统计
    avg_loss = total_loss / max(num_batches, 1)
    avg_tversky_loss = total_dice_loss / max(num_batches, 1)
    avg_bce_loss = total_bce_loss / max(num_batches, 1)
    avg_normal_loss = total_normal_loss / max(num_batches, 1)
    avg_dice_score = total_dice_score / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "tversky_loss": avg_tversky_loss,
        "bce_loss": avg_bce_loss,
        "normal_loss": avg_normal_loss,
        "dice_score": avg_dice_score,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """验证一个 epoch（带 tqdm）"""
    model.eval()

    total_loss = 0.0
    total_dice_score = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Val   E{epoch+1}/{total_epochs}",
        ncols=120,
        leave=True,
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            seg_logits, normals = model(images)
            loss_total, loss_tversky, loss_bce, loss_normal = criterion(seg_logits, normals, labels)

        dice_score = compute_dice(seg_logits, labels)
        total_loss += loss_total.item()
        total_dice_score += dice_score
        num_batches += 1

        pbar.set_postfix({
            'val_loss': f'{total_loss/num_batches:.4f}',
            'val_dice': f'{total_dice_score/num_batches:.4f}',
        })

    pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice_score / max(num_batches, 1)

    return {"val_loss": avg_loss, "val_dice": avg_dice}


# ===== 主训练函数 =====

def main(args):
    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"train_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  🔥 Hybrid Chimera 训练引擎")
    print(f"  设备: {device}")
    print(f"  Patch 大小: {args.crop_size}³")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  λ_normal: {args.lambda_normal}")
    print(f"  输出: {run_dir}")
    print(f"{'='*70}\n")

    # ===== 数据 =====
    # 增强变换：仅 FlipRotate，Crop 已内置到 Dataset 的 memmap __getitem__
    aug_transform = RandomFlipRotate3D(flip_prob=0.5, rotate_prob=0.5)

    full_dataset = VesuviusTrainDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        crop_size=args.crop_size,
        transform=aug_transform,
        samples_per_volume=args.samples_per_volume,
        cache_size=args.cache_size,
        max_files=args.max_chunks,
    )


    # 按 8:2 拆分 train/val
    total_len = len(full_dataset)
    val_len = max(1, int(total_len * 0.2))
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"[Data] Train: {train_len} samples, Val: {val_len} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ===== 模型 =====
    model = DualHeadResUNet3D(in_channels=1, n_filters=args.n_filters).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] 参数量: {params:,}")

    # 恢复 checkpoint
    start_epoch = 0
    best_dice = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            best_dice = ckpt.get("best_dice", 0.0)
            print(f"[Resume] 从 epoch {start_epoch} 恢复, best_dice={best_dice:.4f}")
        else:
            if any(k.startswith("model.") for k in ckpt.keys()):
                ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt, strict=False)
            print(f"[Resume] 加载权重（无 epoch 信息）")

    # ===== 优化器 & 调度器 =====
    criterion = ChimeraLoss(
        lambda_normal=args.lambda_normal,
        lambda_bce=args.lambda_bce,
        pos_weight=args.pos_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
    ).to(device)
    print(f"[Loss] Tversky(a={args.tversky_alpha}, b={args.tversky_beta}) + BCE(pw={args.pos_weight}, lam={args.lambda_bce}) + Normal(lam={args.lambda_normal})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ===== 训练循环 =====
    print(f"\n{'='*70}")
    print(f"  开始训练 (epoch {start_epoch+1} → {args.epochs})")
    print(f"{'='*70}\n")

    history = []
    t_total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t_ep_start = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"\n--- Epoch {epoch+1}/{args.epochs} | LR: {lr_now:.6f} ---")

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.epochs,
        )

        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, args.epochs)

        # 调度器
        scheduler.step()

        # 每 epoch 可视化输出
        save_epoch_visualization(
            model, val_loader, device, run_dir, epoch, criterion
        )

        # 记录
        ep_time = time.time() - t_ep_start
        epoch_info = {
            "epoch": epoch + 1,
            "lr": lr_now,
            "time": ep_time,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_info)

        # 打印 epoch 总结
        print(
            f"\n  📊 Epoch {epoch+1} 总结:"
            f"\n     Train - Loss: {train_metrics['loss']:.4f} | "
            f"Tversky: {train_metrics['tversky_loss']:.4f} | "
            f"Dice: {train_metrics['dice_score']:.4f} | "
            f"Normal: {train_metrics['normal_loss']:.4f}"
            f"\n     Val   - Loss: {val_metrics['val_loss']:.4f} | "
            f"Dice: {val_metrics['val_dice']:.4f}"
            f"\n     Time: {format_time(ep_time)} | LR: {lr_now:.6f}"
        )

        # 保存 best model
        if val_metrics["val_dice"] > best_dice:
            best_dice = val_metrics["val_dice"]
            best_path = run_dir / "best_model.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "args": vars(args),
            }, str(best_path))
            print(f"  🏆 New Best Dice: {best_dice:.4f} → {best_path.name}")

        # 定期保存 checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = run_dir / f"checkpoint_epoch{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "args": vars(args),
            }, str(ckpt_path))
            print(f"  💾 Checkpoint: {ckpt_path.name}")

    # ===== 最终报告 =====
    total_time = time.time() - t_total_start

    print(f"\n{'='*70}")
    print(f"  训练完成!")
    print(f"  总 Epochs: {args.epochs - start_epoch}")
    print(f"  总耗时: {format_time(total_time)}")
    print(f"  最佳 Val Dice: {best_dice:.4f}")
    print(f"  输出目录: {run_dir}")
    print(f"{'='*70}\n")

    # 保存训练历史
    history_path = run_dir / "training_history.json"
    with open(str(history_path), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"  📈 训练历史: {history_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Chimera 训练引擎")

    # 数据参数
    parser.add_argument("--image_dir", type=str,
                        default="data/vesuvius-challenge-surface-detection/train_images")
    parser.add_argument("--label_dir", type=str,
                        default="data/vesuvius-challenge-surface-detection/train_labels")
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="最多使用多少个 chunk（调试用）")
    parser.add_argument("--samples_per_volume", type=int, default=4,
                        help="每个体积每 epoch 采集几个 patch")
    parser.add_argument("--cache_size", type=int, default=8,
                        help="LRU 缓存体积数量")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=64,
                        help="Random Crop 3D 尺寸")
    parser.add_argument("--lambda_normal", type=float, default=0.0,
                        help="法线损失权重 (分割未收敛前先关闭)")
    parser.add_argument("--lambda_bce", type=float, default=0.3,
                        help="BCE 损失权重")
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="BCE 正样本权重 (1.0=标准 BCE)")
    parser.add_argument("--tversky_alpha", type=float, default=0.3,
                        help="Tversky FP 惩罚系数")
    parser.add_argument("--tversky_beta", type=float, default=0.7,
                        help="Tversky FN 惩罚系数 (鼓励召回)")
    parser.add_argument("--n_filters", type=int, default=32,
                        help="模型基础滤波器数")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 工作进程数")

    # 保存参数
    parser.add_argument("--output_dir", type=str, default="20_src/output")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每几个 epoch 保存 checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")

    # 设备
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    main(args)
