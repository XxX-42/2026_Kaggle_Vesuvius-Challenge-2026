import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import VesuviusDataset, BalancedBatchSampler
from src.models import get_unet, load_checkpoint
from src.losses import TverskyLoss
from src.utils import visualize_prediction, seed_everything

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    "layer_dir": PROJECT_ROOT / "data/flattened_layers",
    "mask_path": PROJECT_ROOT / "data/native/train/1/inklabels.png",
    "frag_mask_path": PROJECT_ROOT / "data/native/train/1/mask.png",
    
    "start_layer": 5,
    "num_layers": 16,
    
    "patch_size": 224,
    "batch_size": 8,
    "epochs": 30,
    "lr": 5e-5,
    "wd": 1e-2,
    
    "output_dir": PROJECT_ROOT / "outputs",
    "save_path": PROJECT_ROOT / "models/vesuvius_rgt_v2.pth",
}

def get_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

def train():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIG["output_dir"].mkdir(exist_ok=True, parents=True)
    
    # 1. Dataset
    ds = VesuviusDataset(
        volume_path=CONFIG["layer_dir"],
        ink_mask_path=CONFIG["mask_path"],
        fragment_mask_path=CONFIG["frag_mask_path"],
        is_rgt=True,
        z_start=CONFIG["start_layer"],
        n_channels=CONFIG["num_layers"],
        patch_size=CONFIG["patch_size"],
        augmentations=get_transforms()
    )
    
    sampler = BalancedBatchSampler(ds, CONFIG["batch_size"])
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
    
    # 2. Model
    model = get_unet(in_channels=CONFIG["num_layers"]).to(device)
    
    # 3. Opt/Loss
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["wd"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    scaler = torch.amp.GradScaler('cuda')
    
    # 4. Loop
    for epoch in range(1, CONFIG["epochs"]+1):
        model.train()
        epoch_loss = 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch}/{CONFIG['epochs']} - Loss: {epoch_loss/len(loader):.4f}")
        scheduler.step()
        
    torch.save(model.state_dict(), CONFIG["save_path"])

if __name__ == "__main__":
    train()
