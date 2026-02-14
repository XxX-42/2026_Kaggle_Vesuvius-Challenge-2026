"""
Vesuvius Challenge 2026 - 3D 模型训练脚本 (升级版 + 动态策略)
支持 AMP 训练、实验管理、实时监控、自动绘图和动态 GPU 策略
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
# [FIX] 使用新版 scaler，或者继续使用 GradScaler (它通常兼容)
# torch.amp.GradScaler is available in newer versions, but torch.cuda.amp.GradScaler is alias.
# We will use torch.cuda.amp.GradScaler for compatibility but autocast will be updated.
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset import create_dataloader, VesuviusDataset
from src.model import ResUNet3D, ResUNet3DWithAffinity, compute_affinity_target, MAMBA_AVAILABLE
from src.loss import CombinedLoss, CombinedLossWithAffinity
from src.surface_loss import CompoundLoss
from src.utils import ExperimentManager, plot_training_curves, plot_multi_patch_comparison, plot_3d_comparison, format_time, DynamicGPUManager, generate_high_res_sample




def get_args():
    parser = argparse.ArgumentParser(description='Train 3D ResU-Net for Vesuvius Challenge')
    
    # 路径参数
    parser.add_argument('--data_dir', type=str, default='data/vesuvius-challenge-surface-detection', help='数据根目录')
    parser.add_argument('--output_dir', type=str, default='10_src/outputs', help='实验输出根目录')
    parser.add_argument('--exp_name', type=str, default='ResUNet', help='实验名称前缀')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=6, help='批次大小')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # [FIX] 强制使用正方体 Patch 尺寸 (D, H, W) = (64, 64, 64)
    # 这确保了输入和输出都是正方体，符合 TIF 原生结构，且可视化结果也是正方体
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64], help='训练 Patch 尺寸 (D, H, W)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # Loss 参数
    parser.add_argument('--alpha', type=float, default=0.3, help='BCE Loss 权重')
    parser.add_argument('--beta', type=float, default=0.5, help='Tversky Loss 权重')
    parser.add_argument('--gamma', type=float, default=0.15, help='clDice Loss 权重')
    parser.add_argument('--dynamic_beta', type=float, default=1.0, help='动态 BCE 权重因子')
    parser.add_argument('--tversky_alpha', type=float, default=0.3, help='Tversky FP 权重 (惩罚厚度/假阳性)')
    parser.add_argument('--tversky_beta', type=float, default=0.7, help='Tversky FN 权重 (惩罚召回/假阴性)')
    parser.add_argument('--cldice_warmup', type=int, default=5, help='clDice 预热轮数 (此期间 gamma=0)')
    
    # Affinity 拓扑感知分支
    parser.add_argument('--use_affinity', action='store_true', help='启用 Affinity 拓扑感知分支 + Mamba 全局上下文')
    parser.add_argument('--affinity_weight', type=float, default=0.0, help='Affinity Loss 权重')
    parser.add_argument('--dilation_iters', type=int, default=1, help='训练集标签膨胀迭代次数 (0=关闭, 1-2=建议值)')
    
    # CompoundLoss (表面感知损失)
    parser.add_argument('--use_compound', action='store_true', help='使用 CompoundLoss (BCE + SurfaceDice + Boundary) 替代 CombinedLoss')
    parser.add_argument('--w_bce', type=float, default=1.0, help='CompoundLoss: BCE 权重')
    parser.add_argument('--w_surface', type=float, default=1.0, help='CompoundLoss: SurfaceDice 权重')
    parser.add_argument('--w_boundary', type=float, default=0.5, help='CompoundLoss: Boundary 权重')
    parser.add_argument('--tau', type=float, default=2.0, help='CompoundLoss: SurfaceDice 容差半径 (体素)')
    parser.add_argument('--boundary_warmup', type=int, default=5, help='CompoundLoss: Boundary Loss 开始生效的 Epoch')
    
    # 断点续训
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--debug', action='store_true', help='调试模式 (只使用少量数据)')
    parser.add_argument('--resume', type=str, default=None, help='断点续训: checkpoint 路径')
    parser.add_argument('--reset_optimizer', action='store_true', help='重置优化器状态 (与新损失函数配合使用)')
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def gpu_dilation(mask, iterations=1):
    """
    使用 GPU 上的 3D MaxPool 模拟形态学膨胀
    比 CPU 上的 scipy.ndimage.binary_dilation 快 10-100 倍
    """
    if iterations <= 0: return mask
    for _ in range(iterations):
        mask = torch.nn.functional.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    return mask


def train_epoch(model, loader, optimizer, criterion, scaler, device, accumulation_steps=1, epoch=1, args=None):
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_cldice = 0.0
    running_aff_loss = 0.0
    
    use_affinity = args.use_affinity
    
    # 动态调整 gamma / epoch (区分 CompoundLoss 和 CombinedLoss)
    use_compound = getattr(args, 'use_compound', False)
    if use_compound:
        # CompoundLoss: 设置 epoch 用于 boundary warmup
        criterion.set_epoch(epoch)
    else:
        active_criterion = criterion.seg_loss if use_affinity else criterion
        if epoch <= args.cldice_warmup:
            active_criterion.gamma = 0.0
        else:
            active_criterion.gamma = args.gamma
        
    optimizer.zero_grad()
    
    # 构建进度条描述
    current_lr = optimizer.param_groups[0]['lr']
    if use_compound:
        desc_info = f"Train E{epoch} (LR={current_lr:.6f})"
    else:
        gamma_val = active_criterion.gamma
        desc_info = f"Train E{epoch} (Gamma={gamma_val}, LR={current_lr:.6f})"
    
    # [FIX] 使用 dynamic_ncols=True 和 ascii=False 尝试修复换行问题
    # 如果 VSCode 终端仍有问题，可以尝试 ascii=True
    pbar = tqdm(loader, desc=desc_info, leave=True, dynamic_ncols=True, mininterval=0.5)
    
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # [P0 FIX] 处理 2-channel Labels (Ch0: Label, Ch1: ValidMask)
        if labels.shape[1] == 2:
            valid_mask = labels[:, 1:2, ...]
            labels = labels[:, 0:1, ...]
        else:
            # Fallback for old data or if dataset not updated (safety)
            valid_mask = torch.ones_like(labels)
            
        # [GPU 加速] 在 GPU 上进行标签膨胀 (仅训练时)
        # 注意：仅膨胀 label，不膨胀 valid_mask (保持原始有效区域)
        if args.dilation_iters > 0:
            with torch.no_grad():
                labels = gpu_dilation(labels, iterations=args.dilation_iters)
        
        # [Sanity Check] 首 epoch 首 batch 检查标签有效性 (稀疏性验证)
        if epoch == 1 and i == 0:
            label_sum = labels.float().sum().item()
            label_total = labels.numel()
            label_ratio = label_sum / label_total
            label_max = labels.max().item()
            
            print(f"\n[Sanity Check] Batch 0 Label Stats:")
            print(f"  Max Value: {label_max}")
            print(f"  Positive Ratio: {label_ratio*100:.2f}% (Should be < 10%, Warning if > 30%)")
            
            if label_max == 0:
                print("⚠️ WARNING: Label is all black (0). Check dataset path or IDs!")
            if label_ratio > 0.30:
                print(f"⚠️ CRTICAL WARNING: Label sparsity is suspect ({label_ratio*100:.1f}%)!")
                print("  POSSIBLE CAUSE: 'Ignore' regions (val=2) are being treated as targets.")
                print("  ACTION: Check src/dataset.py label binarization logic.")
        
        # 前向传播 (自动区分 Affinity / 标准模式)
        with torch.amp.autocast('cuda'):
            if use_affinity:
                seg_logits, aff_logits = model(images)
                aff_targets = compute_affinity_target(labels)
                # 传递 valid_mask 给 Loss
                loss, bce, dice, cldice, aff_loss = criterion(seg_logits, aff_logits, labels, aff_targets, valid_mask=valid_mask)
            else:
                outputs = model(images)
                # CompoundLoss 和 CombinedLoss 都返回 4 元组
                # 传递 valid_mask 给 Loss
                loss, bce, dice, cldice = criterion(outputs, labels, valid_mask=valid_mask)
            loss = loss / accumulation_steps
            
        # 混合精度反向传播
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 记录数据
        running_loss += loss.item() * accumulation_steps
        running_bce += bce.item()
        running_dice += dice.item()
        running_cldice += cldice.item()
        
        # [DEBUG] 每 50 个 batch 打印像素级统计 (对标 20_src)
        if (i + 1) % 50 == 0:
            with torch.no_grad():
                if use_affinity:
                    pred_prob = torch.sigmoid(seg_logits)
                else:
                    pred_prob = torch.sigmoid(outputs)
                
                pred_pixels = (pred_prob > 0.5).float().sum().item()
                gt_pixels = labels.sum().item()
                mean_prob = pred_prob.mean().item()
                
                # 使用 \r 覆盖当前行或换行
                tqdm.write(f"\n[DEBUG] Batch {i+1}: Pred_Pixels={int(pred_pixels)}, GT_Pixels={int(gt_pixels)}, Mean_Prob={mean_prob:.4f}")

        # 更新 TQDM Postfix (显示 Dice Score 而非 Loss)
        current_lr = optimizer.param_groups[0]['lr']
        # 注意: criterion 返回的是 dice loss (1-dice)
        # 即使使用了 Tversky, 也可以近似显示 1-loss 作为 soft score
        soft_dice_score = 1.0 - dice.item()
        
        pbar.set_postfix({
            'loss': f"{loss.item() * accumulation_steps:.3f}",
            'dice': f"{soft_dice_score:.3f}", # Training Dice (Score, higher is better)
            'lr': f"{current_lr:.1e}",
        })
        
    avg_loss = running_loss / len(loader)
    avg_bce = running_bce / len(loader)
    avg_dice = running_dice / len(loader) # Loss
    avg_cldice = running_cldice / len(loader)
    
    return avg_loss, avg_bce, avg_dice, avg_cldice


@torch.no_grad()
def validate(model, loader, criterion, device, use_affinity=False):
    model.eval()
    total_loss = 0
    total_bce = 0
    total_dice_loss = 0
    total_cldice = 0
    total_val_dice_score = 0
    
    # 用于可视化的样本列表 (捕获前 4 个 batch 的第一个样本)
    vis_samples = []
    
    # 用于可视化的样本列表 (捕获前 4 个 batch 的第一个样本)
    vis_samples = []
    
    pbar = tqdm(loader, desc="Validating", leave=False, dynamic_ncols=True, mininterval=0.5)
    
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # [P0 FIX] 处理 2-channel Labels
        if labels.shape[1] == 2:
            valid_mask = labels[:, 1:2, ...]
            labels = labels[:, 0:1, ...]
        else:
            valid_mask = torch.ones_like(labels)
            
        with torch.amp.autocast('cuda'):
            if use_affinity:
                seg_logits, aff_logits = model(images)
                aff_targets = compute_affinity_target(labels)
                loss, bce, dice, cldice, aff_loss = criterion(seg_logits, aff_logits, labels, aff_targets, valid_mask=valid_mask)
                outputs = seg_logits  # 后续可视化和 Dice 计算使用分割头输出
            else:
                outputs = model(images)
                loss, bce, dice, cldice = criterion(outputs, labels, valid_mask=valid_mask)
        
        # 捕获前 4 个 batch 的样本用于可视化
        if len(vis_samples) < 4:
            probs_vis = torch.sigmoid(outputs)
            vis_samples.append({
                'raw': images[0, 0].cpu().numpy(),  # (D, H, W)
                'gt': labels[0, 0].cpu().numpy(),   # (D, H, W)
                'pred': probs_vis[0, 0].cpu().numpy()  # (D, H, W)
            })
            
        total_loss += loss.item()
        total_bce += bce.item()
        total_dice_loss += dice.item()
        total_cldice += cldice.item()
        
        # 计算 Dice Score (阈值 0.5)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        intersection = (preds * labels).sum()
        union = preds.sum() + labels.sum()
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        total_val_dice_score += dice_score.item()
        
        # [Output] 实时更新验证进度
        pbar.set_postfix({
            'loss': f"{total_loss / (i+1):.4f}",
            'dice': f"{total_val_dice_score / (i+1):.4f}"
        })
        
    avg_loss = total_loss / len(loader)
    avg_bce = total_bce / len(loader)
    avg_dice = total_dice_loss / len(loader)
    avg_cldice = total_cldice / len(loader)
    avg_val_dice_score = total_val_dice_score / len(loader)
    
    return avg_loss, avg_bce, avg_dice, avg_cldice, avg_val_dice_score, vis_samples


def main():
    # [CRITICAL] 强制检查 GPU
    assert torch.cuda.is_available(), "CRITICAL: No GPU found! Check CUDA installation."
    
    args = get_args()
    set_seed(args.seed)
    
    # 1. 实验初始化
    exp = ExperimentManager(args, experiment_name=args.exp_name)
    
    # 2. GPU 管理器初始化
    gpu_manager = DynamicGPUManager()
    
    # 设备配置
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # 路径配置
    csv_path = os.path.join(args.data_dir, 'train.csv')
    image_root = os.path.join(args.data_dir, 'train_images')
    label_root = os.path.join(args.data_dir, 'train_labels')
    
    # 数据集准备 (训练集带标签膨胀，验证集不带)
    print("Initializing dataset...")
    full_dataset = VesuviusDataset(
        csv_path=csv_path,
        image_root=image_root,
        label_root=label_root,
        patch_size=tuple(args.patch_size),
        mode='train'
    )
    
    # 调试模式
    if args.debug:
        print("DEBUG MODE: Using only 10 samples")
        indices = list(range(min(10, len(full_dataset))))
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # 数据划分
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # 验证集使用独立的 dataset（不膨胀）来保持指标真实性
    val_dataset_clean = VesuviusDataset(
        csv_path=csv_path,
        image_root=image_root,
        label_root=label_root,
        patch_size=tuple(args.patch_size),
        mode='val'
    )
    # 取相同的验证集索引
    val_indices = val_set.indices
    val_set_clean = torch.utils.data.Subset(val_dataset_clean, val_indices)
    
    print(f"Train samples: {len(train_set)} (dilation={args.dilation_iters} on GPU), Val samples: {len(val_set_clean)} (no dilation)")
    
    # DataLoader (优化: persistent_workers 避免重建进程开销, prefetch_factor 预加载数据)
    loader_kwargs = {
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_set_clean, batch_size=args.batch_size, shuffle=False,
        **loader_kwargs
    )
    
    # 模型、损失与优化器
    print("Initializing model...")
    if args.use_affinity:
        model = ResUNet3DWithAffinity(in_channels=1, out_channels=1).to(device)
        mamba_status = "Mamba SSM (状态空间模型)" if MAMBA_AVAILABLE else "大核卷积回退 (5x5x5 等效感受野)"
        print(f"[Model] 使用 ResUNet3DWithAffinity | Bottleneck: {mamba_status} | Affinity Head: ON")
        criterion = CombinedLossWithAffinity(
            alpha=args.alpha, 
            beta=args.beta, 
            gamma=args.gamma,
            affinity_weight=args.affinity_weight,
            dynamic_beta=args.dynamic_beta,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta
        ).to(device)
    else:
        model = ResUNet3D(in_channels=1, out_channels=1).to(device)
        print("[Model] 使用标准 ResUNet3D")
        criterion = CombinedLoss(
            alpha=args.alpha, 
            beta=args.beta, 
            gamma=args.gamma,
            dynamic_beta=args.dynamic_beta,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta
        ).to(device)
    
    # [P1/P2] CompoundLoss 覆盖 (最高优先级)
    if args.use_compound:
        criterion = CompoundLoss(
            w_bce=args.w_bce,
            w_surface=args.w_surface,
            w_boundary=args.w_boundary,
            tau=args.tau,
            boundary_warmup=args.boundary_warmup
        ).to(device)
        print(f"[Loss] 使用 CompoundLoss: BCE({args.w_bce}) + SurfaceDice({args.w_surface}, tau={args.tau}) + Boundary({args.w_boundary}, warmup={args.boundary_warmup})")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    # LR 调度器: 当 Val Dice 停滞时自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )
    
    # 断点续训逻辑 (改进版: 支持 strict=False 部分加载 + --reset_optimizer)
    start_epoch = 1
    best_dice = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[Resume] 加载 checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 支持 strict=False 部分加载（从旧 ResUNet3D 迁移到 Affinity 模型）
            load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if load_result.missing_keys:
                print(f"[Resume] 未加载的 key (新增层, 随机初始化): {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"[Resume] 意外的 key (已忽略): {load_result.unexpected_keys}")
            
            if args.reset_optimizer:
                print(f"[Resume] --reset_optimizer: 重置优化器/调度器/Epoch (手术式微调模式)")
                # 不加载 optimizer, 不加载 epoch, 不加载 best_dice
                # Epoch 重置为 1, best_dice 重置为 0 (新阶段重新开始)
                start_epoch = 1
                best_dice = 0.0
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"[Resume] 加载优化器状态")
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_dice = checkpoint.get('best_dice', 0.0)
            
            print(f"[Resume] 从 Epoch {start_epoch} 开始训练, Best Dice: {best_dice:.4f}")
        else:
            print(f"[Resume] 警告: checkpoint 文件不存在: {args.resume}")
    
    # 训练循环
    start_time = time.time()
    GREEN = '\033[92m'
    RESET = '\033[0m'
    current_accumulation_steps = args.accumulation_steps
    
    print(f"\n{'='*60}")
    print(f"Start training: Epoch {start_epoch} to {args.epochs}")
    if args.use_compound:
        print(f"Loss Config: CompoundLoss [BCE={args.w_bce}, SurfaceDice={args.w_surface}, Boundary={args.w_boundary}]")
        print(f"SurfaceDice tau={args.tau}, Boundary warmup={args.boundary_warmup} epochs")
    else:
        print(f"Loss Config: Alpha={args.alpha}, Beta={args.beta}, Gamma={args.gamma}")
        print(f"Tversky Config: alpha={args.tversky_alpha} (FP), beta={args.tversky_beta} (FN)")
    print(f"Affinity: {'ON (weight=' + str(args.affinity_weight) + ')' if args.use_affinity else 'OFF'}")
    print(f"Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    if not args.use_compound:
        print(f"Warmup: {args.cldice_warmup} epochs (Gamma=0)")
    print(f"{'='*60}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # [Output] Print LR (Decimal Format)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch}/{args.epochs} | Learning Rate: {current_lr:.6f} ---")
        
        # Train & Val (Updated unpacking)
        (train_loss, t_bce, t_dice_loss, t_cldice) = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, 
            accumulation_steps=current_accumulation_steps,
            epoch=epoch,
            args=args
        )
        (val_loss, v_bce, v_dice_loss, v_cldice, val_dice_score, vis_samples) = validate(
            model, val_loader, criterion, device, use_affinity=args.use_affinity
        )
        
        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # [Dynamic Strategy] 获取 GPU 状态并建议策略
        gpu_status = gpu_manager.get_status()
        current_accumulation_steps = gpu_manager.suggest_accumulation_steps(
            current_accumulation_steps, gpu_status['memory_util']
        )
        
        # 记录日志 (扩展字段)
        exp.log_epoch({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice_score,
            "clDice": v_cldice, # 记录验证集 clDice
            "lr": current_lr,
            "time": epoch_duration
        })
        
        # 实时绘图 (带 Patch 可视化)
        patch_data = vis_samples[0] if vis_samples else None
        plot_training_curves(exp.get_log_path(), exp.get_exp_dir(), patch_data=patch_data)
        # 多 Patch 对比图
        plot_multi_patch_comparison(vis_samples, exp.get_exp_dir(), epoch=epoch)
        
        # [Output] 3D Voxel Visualization (Raw/Label/Pred)
        if len(vis_samples) > 0:
             plot_3d_comparison(vis_samples[0], exp.get_exp_dir(), epoch=epoch, sample_idx=0)
             
             # [NEW] 生成 256x256x256 高清样本 (滑动窗口)
             highres_save_dir = Path(exp.get_exp_dir()) / "epoch_vis"
             highres_save_dir.mkdir(exist_ok=True)
             highres_path = highres_save_dir / f"epoch{epoch:03d}_highres_256.tif"
             
             # 注意：patch_size 应该使用训练时的 patch_size (args.patch_size[0])
             generate_high_res_sample(
                 model=model,
                 dataset=val_dataset_clean, # 使用验证集数据
                 save_path=highres_path,
                 device=device,
                 patch_size=args.patch_size[0], # Assuming cube
                 roi_size=256,
                 stride=32
             )

        
        # 命令行监控输出 (优化版)
        dice_str = f"{val_dice_score:.2f}" if val_dice_score > 0 else "0.00"
        if val_dice_score > best_dice:
            dice_str = f"{GREEN}{val_dice_score:.2f} (New Best!){RESET}"
            
            # 保存最佳模型
            save_path = exp.get_checkpoint_path("best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': val_dice_score,
                'args': vars(args)
            }, save_path)
            best_dice = val_dice_score

        # 每个 Epoch 都保存 checkpoint (用于后续集成选择)
        epoch_save_path = exp.get_checkpoint_path(f"epoch_{epoch:03d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'val_dice': val_dice_score,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'args': vars(args)
        }, epoch_save_path)
            
        # 调度器更新
        scheduler.step(val_dice_score)
        
        # [Output] Epoch 总结 (对标 20_src)
        # 将 Train Dice Loss 转换为近似 Score 以便于直观对比
        train_dice_score_est = 1.0 - t_dice_loss
        
        print(
            f"\n  📊 Epoch {epoch} 总结:"
            f"\n     Train - Loss: {train_loss:.4f} | BCE: {t_bce:.4f} | Dice(Soft): {train_dice_score_est:.4f} | clDice: {t_cldice:.4f}"
            f"\n     Val   - Loss: {val_loss:.4f} | BCE: {v_bce:.4f} | Dice(Hard): {dice_str}"
            f"\n     Time: {format_time(epoch_duration)} | LR: {current_lr:.2e} | GPU Mem: {gpu_status['memory_util']:.1f}%"
        )
            
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {format_time(total_time)}")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Experiment saved to: {exp.get_exp_dir()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
