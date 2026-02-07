import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
import cv2
import numpy as np
import random
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import VesuviusDataset, BalancedBatchSampler
from src.models import get_unet, load_checkpoint
from src.losses import DiceBCELoss
from src.utils import visualize_prediction, seed_everything

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    # Data - Native surface volume
    "volume_dir": PROJECT_ROOT / "data/native/train/1/surface_volume",
    "mask_path": PROJECT_ROOT / "data/native/train/1/inklabels.png",
    "papyrus_mask_path": PROJECT_ROOT / "data/native/train/1/mask.png",
    
    # Volume slicing
    "in_channels": 14,
    "z_start": 18,
    
    # Training
    "patch_size": 224,
    "batch_size": 8,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    
    # Output
    "output_dir": PROJECT_ROOT / "outputs",
    "save_path": PROJECT_ROOT / "models/vesuvius_native_finetuned.pth",
    "pretrained_path": PROJECT_ROOT / "models/vesuvius_best.pth",
    
    # Windowing
    "window_min": 18000,
    "window_max": 28000,
}

def get_train_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_REFLECT_101, rotate_limit=30, scale_limit=0.1),
        A.RandomBrightnessContrast(p=0.5),
    ])

def train():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create timestamped run folder
    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = CONFIG["output_dir"] / f"run_native_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory: {run_dir}")
    
    # 1. Dataset
    dataset = VesuviusDataset(
        volume_path=CONFIG["volume_dir"],
        ink_mask_path=CONFIG["mask_path"],
        fragment_mask_path=CONFIG["papyrus_mask_path"],
        is_rgt=False,
        z_start=CONFIG["z_start"],
        n_channels=CONFIG["in_channels"],
        patch_size=CONFIG["patch_size"],
        window_range=(CONFIG["window_min"], CONFIG["window_max"]),
        augmentations=get_train_augmentations(),
        min_ink_threshold=1e-5  # Extremely sensitive threshold for sparse strokes
    )
    
    sampler = BalancedBatchSampler(dataset, CONFIG["batch_size"], ink_ratio=0.5)
    train_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    
    # 2. Model
    model = get_unet(in_channels=CONFIG["in_channels"]).to(device)
    model = load_checkpoint(model, CONFIG["pretrained_path"], device)
    
    # 3. Setup
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = DiceBCELoss(dice_weight=0.5)
    
    # 2026 Standard: torch.amp
    scaler = torch.amp.GradScaler('cuda')
    
    # 4. Loop
    best_loss = float('inf')
    
    # Ensure we use the new Sampler wrapper correctly
    # Note: BalancedBatchSampler yields *indices* into dataset.all_coords
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
        scheduler.step()
        
        # Forensic Visualization
        model.eval()
        with torch.no_grad():
            if dataset.ink_coords:
                # Pick a random ink patch
                # ink_coords stored as (y, x, val)
                y, x, density = random.choice(dataset.ink_coords)
                
                img_t, mask_t = dataset.get_patch(y, x, use_augmentation=False)
                inp = img_t.unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda'):
                    pred = torch.sigmoid(model(inp))[0, 0].cpu().numpy()
                
                # Use middle slice for input viz
                mid_val = img_t[img_t.shape[0]//2].numpy()
                gt_val = mask_t[0].numpy()
                
                # Create detailed debug strip
                vis = visualize_prediction(mid_val, gt_val, pred)
                
                # Overlay stats on image using CV2
                stats_text = f"Ink: {density*100:.2f}%"
                cv2.putText(vis, stats_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv2.imwrite(str(run_dir / f"debug_native_epoch_{epoch:02d}.png"), vis)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best_loss, 
            }, CONFIG["save_path"])
            print("   💾 Saved best model!")

if __name__ == "__main__":
    train()
