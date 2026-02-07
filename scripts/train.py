import sys
import os
print("[DEBUG] Import: yaml", flush=True)
import yaml
print("[DEBUG] Import: torch", flush=True)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print("[DEBUG] Import: VesuviusDataset", flush=True)
from src.data.dataset import VesuviusDataset
print("[DEBUG] Import: MiniUNETR", flush=True)
from src.models.mini_unetr import MiniUNETR
print("[DEBUG] Import: MaskedComboLoss", flush=True)
from src.losses.masked_loss import MaskedComboLoss
print("[DEBUG] Import: VesuviusTrainer", flush=True)
from src.training.trainer import VesuviusTrainer
print("[DEBUG] Imports Done", flush=True)


# ==================================================================================================
# 配置区域 (Configuration Area)
# ==================================================================================================
CONFIG = {
    # ----------------------------------------------------------------------------------------------
    # 训练控制参数 (Training Control Parameters)
    # ----------------------------------------------------------------------------------------------
    'training': {
        'epochs': 60,                       # Stage 5: 延长训练周期 (Extended training epochs)
        'batch_size': 4,                    # 批次大小 (Batch size) - 如果显存不够请调小
        'lr': 3e-4,                         # Stage 5.1: 提升学习率匹配大Batch (Initial learning rate)
        'val_interval': 1,                  # 验证频率 (Validation interval in epochs)
        'amp': True,                        # 是否启用混合精度训练 (Enable Automatic Mixed Precision)
        'compile': False,                   # 是否启用 torch.compile (Windows上建议关闭以避免兼容性问题)
        'grad_accumulation_steps': 1,       # 梯度累积步数 (Gradient accumulation steps)
        'weight_decay': 1e-2,               # 权重衰减 (Weight decay)
    },
    
    # ----------------------------------------------------------------------------------------------
    # 数据加载参数 (Data Loading Parameters)
    # ----------------------------------------------------------------------------------------------
    'data': {
        'data_path': "./data",              # 数据集根目录 (Root directory of the dataset)
        'num_workers': 0,                   # 数据加载线程数 (Number of workers) - Windows下建议设为0-4
        'pin_memory': True,                 # 是否锁页内存 (Pin memory) - 加速数据传输
        'cache_data': True,                 # 是否将数据缓存到RAM (Cache data to RAM) - 需约5GB内存
    },

    # ----------------------------------------------------------------------------------------------
    # 模型架构参数 (Model Architecture Parameters)
    # ----------------------------------------------------------------------------------------------
    'model': {
        'name': 'MiniUNETR',                # 模型名称 (Model name)
        'in_channels': 1,                   # 输入通道数 (Input channels)
        'out_channels': 1,                  # 输出通道数 (Output channels)
        'feature_size': 16,                 # 基础特征维度 (Base feature size)
        'hidden_size': 256,                 # 隐藏层维度 (Hidden layer size)
        'num_heads': 8,                     # 注意力头数 (Number of attention heads) - 必须能整除 hidden_size (256/8=32)
    },

    # ----------------------------------------------------------------------------------------------
    # 损失函数配置 (Loss Function Configuration)
    # ----------------------------------------------------------------------------------------------
    'loss': {
        'use_ignore_mask': True,            # 是否使用忽略掩码 (Use ignore mask)
        'bce_weight': 0.5,                  # BCE 损失权重 (Weight for BCE Loss)
        'dice_weight': 0.5,                 # Dice 损失权重 (Weight for Dice Loss)
        'pos_weight': 2.0,                  # Stage 5.1: 降低权重避免误报 (Positive sample weight)
    }
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True # Ensure we override any previous config
)
logger = logging.getLogger(__name__)

def main():
    # 1. Load Config
    # config_path = "configs/train_c001.yaml"
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # logger.info(f"Loaded config from {config_path}")
    
    # 使用内部配置覆盖 (Use internal configuration)
    config = CONFIG
    logger.info("已加载内部配置 (Loaded internal configuration).")
    
    # 2. Setup Data
    # For now, we split the training set into Train/Val since we don't have separate folders yet in our mock setup.
    # In real pipeline: use K-Fold or strict split files.
    # Enable RAM Caching for Speed (Uses ~5GB RAM)
    full_dataset = VesuviusDataset(
        data_path=config['data']['data_path'], 
        mode='train', 
        cache_data=config['data']['cache_data']
    )
    
    # Simple 80/20 split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Validation dataset should strictly NOT have augmentations (handled in Dataset class by 'mode', 
    # but since we split from one object, it keeps the robust transform. 
    # TODO: In production, instantiate two VesuviusDataset objects with different modes.)
    # For prototype: We accept this or force mode change if easy. 
    # Let's keep it simple: Real training uses 'train/' and 'val/' folders or indices list.
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # 3. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniUNETR(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        feature_size=config['model']['feature_size'],
        hidden_size=config['model']['hidden_size'],
        num_heads=config['model']['num_heads']
    )
    
    # Optimization: Torch Compile
    # Windows support is experimental, wrapping in try-except
    if config['training'].get('compile', False):
        logger.info("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    # 4. Setup Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['lr']), 
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler: ReduceLROnPlateau (Stage 5.1: patience=8 给 AutoScaler 更多时间)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
    )
    
    # Positive Weight (Force ink prediction)
    pos_weight = torch.tensor([config['loss']['pos_weight']]).to(device)
    
    criterion = MaskedComboLoss(
        bce_weight=config['loss']['bce_weight'], 
        dice_weight=config['loss']['dice_weight'], 
        use_ignore_mask=config['loss']['use_ignore_mask'],
        pos_weight=pos_weight
    )
    
    # 5. Start Trainer
    trainer = VesuviusTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        scheduler=scheduler,
        device=str(device)
    )
    
    logger.info("Starting Training...")
    trainer.fit()

if __name__ == "__main__":
    main()
