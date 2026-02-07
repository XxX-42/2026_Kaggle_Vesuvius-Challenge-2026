"""
Vesuvius Challenge 2026 - RGT Full Scale Training (V2)
======================================================
Strict Balanced Sampling & True 3D Volumetric Training on RGT Layers.
Refactored to use shared modules.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import segmentation_models_pytorch as smp
import albumentations as A
import cv2
import numpy as np
import random
from tqdm import tqdm

# Modular imports
from src.data.rgt_dataset import RGTVolumeDataset
from src.data.samplers import ForcedBalancedBatchSampler
from src.core.metrics import TverskyLoss

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    # Data
    "layer_dir": PROJECT_ROOT / "flattened_layers",
    "mask_path": PROJECT_ROOT / "data/native/train/1/inklabels.png",
    "frag_mask_path": PROJECT_ROOT / "data/native/train/1/mask.png",
    
    # 3D Stack
    "start_layer": 5,    # Start from layer 5
    "num_layers": 16,    # Read 16 layers (5-20)
    
    # Training
    "patch_size": 224,
    "batch_size": 8,     # total batch size
    "epochs": 30,
    "lr": 5e-5,          # User requested 5e-5
    "wd": 1e-2,
    
    # Output
    "output_dir": PROJECT_ROOT / "output",
    "save_path": PROJECT_ROOT / "models/vesuvius_rgt_v2.pth",
}

def collate_fn_balanced(batch):
    """
    Batch is a list of ('type', (y, x)) tuples.
    """
    images = []
    masks = []
    
    for img, mask in batch:
        images.append(img)
        masks.append(mask)
        
    return torch.stack(images), torch.stack(masks)

# Wrapper for the dataset to handle tuple inputs
class WrapperDataset(torch.utils.data.Dataset):
    def __init__(self, real_dataset, transforms=None):
        self.real = real_dataset
        self.transforms = transforms
        
    def __len__(self):
        return len(self.real)
        
    def __getitem__(self, item):
        # item is ('type', (y, x))
        label_type, (y, x) = item
        return self.real.get_patch(y, x, self.transforms)

# =============================================================================
# 🔧 Augmentations
# =============================================================================
def get_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GridDistortion(p=0.2), # Elastic might be too slow for 16ch
        A.RandomBrightnessContrast(p=0.2),
    ])

# =============================================================================
# 🔧 Training Loop
# =============================================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="   Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Modern AMP
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(loader)

def debug_viz(model, dataset, device, epoch, output_dir):
    model.eval()
    
    # 1. Pick one ink, one bg
    y_ink, x_ink = random.choice(dataset.ink_indices)
    y_bg, x_bg = random.choice(dataset.bg_indices)
    
    patches = [('Ink', y_ink, x_ink), ('Bg', y_bg, x_bg)]
    
    rows = []
    
    for label, y, x in patches:
        img_t, mask_t = dataset.get_patch(y, x) # (16, H, W)
        
        # Inference
        inp = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)
            pred = torch.sigmoid(pred)[0,0].cpu().numpy()
            
        # Vis
        mid_idx = img_t.shape[0] // 2
        img_vis = (img_t[mid_idx].numpy() * 255).astype(np.uint8)
        mask_vis = (mask_t[0].numpy() * 255).astype(np.uint8)
        pred_vis = (pred * 255).astype(np.uint8)
        
        row = np.hstack([img_vis, mask_vis, pred_vis])
        row = cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(row, f"{label} GT:{mask_vis.mean():.1f}", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        rows.append(row)
        
    combined = np.vstack(rows)
    cv2.imwrite(str(output_dir / f"debug_rgt_v2_epoch_{epoch:02d}.png"), combined)


# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("🚀 Vesuvius RGT V2 Training Initiated (Refactored)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    CONFIG["output_dir"].mkdir(exist_ok=True, parents=True)
    
    # 1. Dataset
    core_dataset = RGTVolumeDataset(
        layer_dir=CONFIG["layer_dir"],
        mask_path=CONFIG["mask_path"],
        fragment_mask_path=CONFIG["frag_mask_path"],
        start_layer=CONFIG["start_layer"],
        num_layers=CONFIG["num_layers"],
        patch_size=CONFIG["patch_size"]
    )
    
    # 2. Wrapper & Loader
    wrapper = WrapperDataset(core_dataset, transforms=get_transforms())
    
    # Note: ForcedBalancedBatchSampler was moved to src.data.samplers. It iterates list of coords (y, x).
    # Its logic was to iterate self.dataset.ink_indices.
    # We must ensure core_dataset has these attributes. (It did in the original script)
    
    sampler = ForcedBalancedBatchSampler(core_dataset, CONFIG["batch_size"])
    
    loader = DataLoader(
        wrapper, 
        batch_sampler=sampler,
        collate_fn=collate_fn_balanced,
        num_workers=0 # Win safe
    )
    
    # 3. Model
    model = smp.Unet(
        encoder_name='resnet18',
        in_channels=CONFIG["num_layers"],
        classes=1,
        encoder_weights=None # Train from scratch or load carefully
    ).to(device)
    
    # 4. Opt/Loss
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["wd"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    scaler = torch.cuda.amp.GradScaler()
    
    # 5. Loop
    for epoch in range(1, CONFIG["epochs"]+1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        loss = train_one_epoch(model, loader, optimizer, loss_fn, device, scaler)
        print(f"   Loss: {loss:.4f}")
        
        debug_viz(model, core_dataset, device, epoch, CONFIG["output_dir"])
        scheduler.step()
        
        # Save
        if epoch % 5 == 0:
            torch.save(model.state_dict(), CONFIG["save_path"])
            
if __name__ == "__main__":
    main()
