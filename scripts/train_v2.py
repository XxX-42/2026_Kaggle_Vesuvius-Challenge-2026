"""
Vesuvius Challenge 2026 - 训练脚本 V2
=====================================

改进点：
1. 从统一配置文件读取参数 (Single Source of Truth)
2. 物理隔离的验证集划分 (基于空间坐标，非随机)
3. 强制 Hard Dice 指标监控
4. 清理旧的硬编码配置

使用方法：
    python scripts/train_v2.py --config configs/base_config.yaml
"""

import sys
import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset_v2 import VesuviusDatasetV2  # 新版 Dataset
from src.models.mini_unetr import MiniUNETR
from src.losses.masked_loss import MaskedComboLoss
from src.training.trainer import VesuviusTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载统一配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"已加载配置文件: {config_path}")
    return config


def check_gpu_resources() -> dict:
    """
    检查当前 GPU 资源占用情况
    
    Returns:
        dict: GPU 资源信息，包括总显存、已用显存、可用显存、建议 batch size
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，将使用 CPU 训练")
        return {
            "available": False,
            "suggested_batch_size": 2
        }
    
    # 获取 GPU 信息
    gpu_id = 0
    gpu_name = torch.cuda.get_device_name(gpu_id)
    
    # 显存信息 (字节)
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    reserved_memory = torch.cuda.memory_reserved(gpu_id)
    allocated_memory = torch.cuda.memory_allocated(gpu_id)
    
    # 转换为 GB
    total_gb = total_memory / 1e9
    reserved_gb = reserved_memory / 1e9
    allocated_gb = allocated_memory / 1e9
    free_gb = (total_memory - reserved_memory) / 1e9
    
    # 计算建议的 batch size
    # 经验公式：每个样本约占用 0.3-0.5 GB 显存 (256x256x16 输入 + 模型 + 梯度)
    # 预留 1.5 GB 给系统和 CUDA 开销
    usable_gb = max(0, free_gb - 1.5)
    memory_per_sample = 0.35  # GB，保守估计
    
    suggested_batch = max(2, int(usable_gb / memory_per_sample))
    suggested_batch = min(suggested_batch, 32)  # 上限 32
    
    logger.info("=" * 60)
    logger.info("GPU 资源预检 (Pre-flight Check)")
    logger.info("=" * 60)
    logger.info(f"  GPU: {gpu_name}")
    logger.info(f"  总显存: {total_gb:.2f} GB")
    logger.info(f"  已预留: {reserved_gb:.2f} GB")
    logger.info(f"  已分配: {allocated_gb:.2f} GB")
    logger.info(f"  可用: {free_gb:.2f} GB")
    logger.info(f"  建议 Batch Size: {suggested_batch}")
    logger.info("=" * 60)
    
    # 检查是否有其他进程占用显存
    if reserved_gb > 0.5:
        logger.warning(f"⚠️ 检测到已有 {reserved_gb:.2f} GB 显存被占用")
        logger.warning("   建议关闭其他 GPU 程序以获得最佳性能")
    
    return {
        "available": True,
        "gpu_name": gpu_name,
        "total_gb": total_gb,
        "free_gb": free_gb,
        "suggested_batch_size": suggested_batch
    }


def create_dataloaders(config: dict):
    """
    创建训练和验证 DataLoader
    
    关键改进：使用物理隔离而非随机划分
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # 创建训练集（前 80% 区域）
    train_dataset = VesuviusDatasetV2(
        data_path=data_cfg['data_path'],
        mode='train',
        split_region='train',  # 物理隔离：前 80%
        train_ratio=data_cfg.get('train_ratio', 0.8),
        tile_size=data_cfg['tile_size'],
        z_start=data_cfg['z_start'],
        z_end=data_cfg['z_end'],
        cache_data=data_cfg.get('cache_data', True),
        samples_per_epoch=train_cfg.get('samples_per_epoch', 10000)
    )
    
    # 创建验证集（后 20% 区域）
    val_dataset = VesuviusDatasetV2(
        data_path=data_cfg['data_path'],
        mode='val',  # 无增强
        split_region='val',  # 物理隔离：后 20%
        train_ratio=data_cfg.get('train_ratio', 0.8),
        tile_size=data_cfg['tile_size'],
        z_start=data_cfg['z_start'],
        z_end=data_cfg['z_end'],
        cache_data=data_cfg.get('cache_data', True),
        # 验证集使用默认值（将由内部逻辑决定，通常覆盖主要验证点）
        samples_per_epoch=2000 
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=data_cfg.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=data_cfg.get('pin_memory', True)
    )
    
    logger.info(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_model(config: dict, device: torch.device) -> torch.nn.Module:
    """创建模型"""
    model_cfg = config['model']
    
    model = MiniUNETR(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        feature_size=model_cfg['feature_size'],
        hidden_size=model_cfg['hidden_size'],
        num_heads=model_cfg['num_heads']
    )
    
    model = model.to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    return model


def create_optimizer_and_scheduler(model, config: dict):
    """创建优化器和学习率调度器"""
    train_cfg = config['training']
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg['lr']),
        weight_decay=float(train_cfg['weight_decay'])  # 显式转换
    )
    
    # 学习率调度器
    scheduler_type = train_cfg.get('scheduler', 'CosineAnnealingLR')
    
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get('scheduler_T_max', train_cfg['epochs'])),
            eta_min=float(train_cfg.get('scheduler_eta_min', 1e-6))
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=8,
            min_lr=1e-6
        )
    else:
        scheduler = None
        logger.warning(f"未知的调度器类型: {scheduler_type}")
    
    return optimizer, scheduler


def create_criterion(config: dict, device: torch.device):
    """创建损失函数"""
    loss_cfg = config['loss']
    
    pos_weight = torch.tensor([loss_cfg['pos_weight']]).to(device)
    
    criterion = MaskedComboLoss(
        bce_weight=loss_cfg['bce_weight'],
        dice_weight=loss_cfg['dice_weight'],
        use_ignore_mask=loss_cfg['use_ignore_mask'],
        pos_weight=pos_weight
    )
    
    logger.info(f"损失函数: BCE({loss_cfg['bce_weight']}) + Dice({loss_cfg['dice_weight']}), pos_weight={loss_cfg['pos_weight']}")
    
    return criterion


def main():
    parser = argparse.ArgumentParser(description="Vesuvius Training V2")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 Checkpoint 恢复训练"
    )
    args = parser.parse_args()
    
    # 1. 加载配置
    config = load_config(args.config)
    
    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 3. GPU 资源预检（在加载数据前执行）
    gpu_info = check_gpu_resources()
    
    # 根据实际可用显存调整 batch size
    config_batch_size = config['training']['batch_size']
    suggested_batch_size = gpu_info.get('suggested_batch_size', config_batch_size)
    
    if suggested_batch_size < config_batch_size:
        logger.warning(f"⚠️ 显存不足，Batch Size 从 {config_batch_size} 调整为 {suggested_batch_size}")
        config['training']['batch_size'] = suggested_batch_size
    elif suggested_batch_size > config_batch_size:
        logger.info(f"✅ 显存充足，可使用更大 Batch Size: {suggested_batch_size} (当前配置: {config_batch_size})")
        # 可选：自动使用建议值
        # config['training']['batch_size'] = suggested_batch_size
    
    # 4. 创建数据加载器
    train_loader, val_loader = create_dataloaders(config)
    
    # 4. 创建模型
    model = create_model(config, device)
    
    # 5. 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # 6. 创建损失函数
    criterion = create_criterion(config, device)
    
    # 7. 恢复训练（可选）
    start_epoch = 0
    if args.resume:
        logger.info(f"从 Checkpoint 恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"恢复到 Epoch {start_epoch}")
    
    # 8. 创建 Trainer
    # 需要将 config 转换为旧格式以兼容现有 Trainer
    legacy_config = {
        'training': config['training'],
        'data': config['data'],
        'model': config['model'],
        'loss': config['loss'],
        'output': config.get('output', {})
    }
    
    trainer = VesuviusTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=legacy_config,
        scheduler=scheduler,
        device=str(device)
    )
    
    # 9. 打印配置摘要
    logger.info("=" * 60)
    logger.info("训练配置摘要")
    logger.info("=" * 60)
    logger.info(f"  模型: {config['model']['name']}")
    logger.info(f"  hidden_size: {config['model']['hidden_size']}")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch Size: {config['training']['batch_size']}")
    logger.info(f"  Learning Rate: {config['training']['lr']}")
    logger.info(f"  AMP: {config['training']['amp']}")
    logger.info(f"  验证划分: 物理隔离 (train_ratio={config['data'].get('train_ratio', 0.8)})")
    logger.info(f"  主要指标: {config['metrics']['primary_metric']}")
    logger.info("=" * 60)
    
    # 10. 开始训练
    logger.info("开始训练...")
    trainer.fit()
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
