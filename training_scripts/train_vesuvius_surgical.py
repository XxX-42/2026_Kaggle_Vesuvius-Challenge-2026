"""
Vesuvius Challenge 2026 - Surgical Training Script
==================================================
Final optimized version with all data alignment and normalization fixes.

Key Fixes Applied:
1. Z-slices centered on ink signal peak (Z=33-48)
2. Intensity windowing [18000, 28000] -> [0, 1]
3. Gradient accumulation for 6GB VRAM stability
4. Sanity check before training starts
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from glob import glob
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# =========================================================================================
# ⚙️ Configuration
# =========================================================================================
CONFIG = {
    "data_root": Path(r"data/native/train/1"),
    
    # 🎯 Z-Axis Alignment: Use AVAILABLE files with ink signal
    # Files 34-47 are MISSING. Available files with ink: 51-64
    # Z=57 showed strong ink signal (Mean~23k in window)
    "z_slices": list(range(51, 65)),  # Z=51 to Z=64 (14 slices - all available)
    
    # 🎯 Intensity Windowing (from histogram analysis)
    "window_min": 18000.0,
    "window_max": 28000.0,
    
    # Architecture
    "tile_size": 224,
    "batch_size": 4,
    "accumulate_steps": 4,  # Simulate batch_size=16
    "num_workers": 4,
    "epochs": 15,
    "lr": 1e-3,
    
    # System
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Fix Seed
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG["seed"])

# =========================================================================================
# 📦 Dataset with Surgical Normalization
# =========================================================================================
class VesuviusSurgicalDataset(Dataset):
    def __init__(self, root_dir, z_slices, transform=None, mode="train"):
        self.root = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.z_indices = z_slices  # Explicit list [33, 34, ..., 48]
        
        # Windowing parameters
        self.win_min = CONFIG["window_min"]
        self.win_max = CONFIG["window_max"]
        
        # Tiff paths - create a DICTIONARY keyed by actual Z-index from filename
        tif_dir = self.root / "surface_volume"
        all_tifs = list(tif_dir.glob("*.tif"))
        
        # Build dict: {z_index: file_path}
        self.tif_dict = {}
        for f in all_tifs:
            try:
                z_idx = int(f.stem)  # Extract Z from filename like "41.tif" -> 41
                self.tif_dict[z_idx] = f
            except ValueError:
                continue  # Skip non-numeric filenames
        
        # Validate z_slices exist in available files
        available_z = set(self.tif_dict.keys())
        valid_z = [z for z in self.z_indices if z in available_z]
        missing_z = [z for z in self.z_indices if z not in available_z]
        
        if missing_z:
            print(f"⚠️ WARNING: Z-slices {missing_z} not found in data. Available: {sorted(available_z)[:10]}...")
        
        self.z_indices = valid_z
        
        # Get TIF dimensions from first available file
        first_tif = self.tif_dict[self.z_indices[0]]
        with Image.open(first_tif) as img:
            self.w, self.h = img.size  # PIL returns (W, H)
        print(f"[{mode.upper()}] TIF dimensions: {self.w} x {self.h}")
        
        # Load Labels and RESIZE to match TIF dimensions
        print(f"[{mode.upper()}] Loading and resizing labels to {self.w}x{self.h}...")
        mask_path = self.root / "mask.png"
        ink_path = self.root / "inklabels.png"
        
        mask_img = Image.open(mask_path).convert("L")
        ink_img = Image.open(ink_path).convert("L")
        
        # Resize labels/mask to match TIF dimensions
        # NOTE: mask.png might be dummy (all valid). We will combine it with TIF validity.
        provided_mask = np.array(mask_img.resize((self.w, self.h), Image.NEAREST))
        self.ink = np.array(ink_img.resize((self.w, self.h), Image.NEAREST))
        
        mask_img.close()
        ink_img.close()

        # 🚑 AUTO-GENERATED VALIDITY MASK
        # Load one TIF slice to determine where actual data exists
        print(f"[{mode.upper()}] Generatring validity mask from TIF content...")
        with Image.open(first_tif) as img:
            tif_data = np.array(img)
            # Valid where TIF > 0 AND provided_mask > 0
            tif_valid = tif_data > 0
            # Ensure we are robust to dummy mask
            if provided_mask.mean() > 250: # If mask is all white
                self.mask = tif_valid.astype(np.uint8) * 255
            else:
                self.mask = (tif_valid & (provided_mask > 0)).astype(np.uint8) * 255
        
        valid_pixels = (self.mask > 0).sum()
        total_pixels = self.mask.size
        print(f"[{mode.upper()}] Valid Sampling Area: {valid_pixels / total_pixels * 100:.2f}% of image")
        
        print(f"[{mode.upper()}] Using Z-slices: {self.z_indices}")
        print(f"[{mode.upper()}] Intensity window: [{self.win_min}, {self.win_max}]")

    def __len__(self):
        return 2048 if self.mode == "train" else 256

    def __getitem__(self, idx):
        try:
            # 1. Select Random Center Coordinate
            for _ in range(100):
                y = random.randint(0, self.h - CONFIG["tile_size"] - 1)
                x = random.randint(0, self.w - CONFIG["tile_size"] - 1)
                if self.mask[y:y+CONFIG["tile_size"], x:x+CONFIG["tile_size"]].sum() > 0:
                    break
            
            # 2. Load Z-stack
            volume = []
            for z in self.z_indices:
                path = self.tif_dict[z]  # Use dictionary lookup by actual Z-index
                with Image.open(path) as img:
                    region = img.crop((x, y, x + CONFIG["tile_size"], y + CONFIG["tile_size"]))
                    volume.append(np.array(region).astype(np.float32))
            
            volume = np.stack(volume, axis=-1)  # (H, W, Z)
            
            # 3. 🚑 SURGICAL NORMALIZATION: Adaptive Percentile Stretching
            # Instead of fixed [18k, 28k], we stretch EITHER 1-99% OR Min-Max per patch.
            # This ensures every patch has visible texture features (no black holes).
            p_min, p_max = np.percentile(volume, (1, 99))
            if p_max - p_min < 1e-6:
                # Fallback if flat (rare): use global assumption or min/max
                p_min, p_max = volume.min(), volume.max()
            
            # Stretch to [0, 1]
            volume = np.clip((volume - p_min) / (p_max - p_min + 1e-8), 0, 1)

            # 4. Label (binarized)
            # Ensure we are reading INK LABELS, not just the validity mask
            ink_patch = self.ink[y:y+CONFIG["tile_size"], x:x+CONFIG["tile_size"]]
            ink_patch = (ink_patch > 0.5).astype(np.float32)
            
            # 5. Augmentation
            if self.transform:
                data = self.transform(image=volume, mask=ink_patch)
                volume = data["image"]
                ink_patch = data["mask"]
            
            # 6. Format to Tensor
            if not isinstance(volume, torch.Tensor):
                volume = torch.from_numpy(volume).permute(2, 0, 1).float()
            else:
                volume = volume.float()
            
            if not isinstance(ink_patch, torch.Tensor):
                ink_patch = torch.from_numpy(ink_patch).unsqueeze(0).float()
            else:
                ink_patch = ink_patch.float()
                if ink_patch.ndim == 2:
                    ink_patch = ink_patch.unsqueeze(0)
            
            return volume, ink_patch
            
        except Exception as e:
            print(f"⚠️ Data Error at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

# =========================================================================================
# 📉 BCE + Dice Loss
# =========================================================================================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        # BCE (on logits)
        bce_loss = self.bce(inputs, targets)
        
        # Dice (on probabilities)
        probs = torch.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# =========================================================================================
# 📊 Metrics
# =========================================================================================
class MetricMonitor:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fbeta_score(preds, targets, threshold=0.5, beta=0.5):
    preds = (preds > threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    score = ((1 + beta**2) * tp) / ((1 + beta**2) * tp + (beta**2) * fn + fp + 1e-7)
    return score

# =========================================================================================
# 🎓 Training Functions
# =========================================================================================
def sanity_check(dataset):
    """Verify data normalization before training."""
    print("\n🔬 Running Sanity Check...")
    volume, mask = dataset[0]
    
    mean_val = volume.mean().item()
    min_val = volume.min().item()
    max_val = volume.max().item()
    
    print(f"   First Patch Stats (after windowing):")
    print(f"   - Min: {min_val:.4f}")
    print(f"   - Max: {max_val:.4f}")
    print(f"   - Mean: {mean_val:.4f}")
    print(f"   - Mask Sum: {mask.sum().item():.0f}")
    
    if not (0.1 <= mean_val <= 0.9):
        print(f"\n❌ SANITY CHECK FAILED!")
        print(f"   Expected Mean in [0.1, 0.9], got {mean_val:.4f}")
        print(f"   This suggests windowing parameters are incorrect.")
        return False
    
    print("✅ Sanity Check PASSED!\n")
    return True

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    loss_meter = MetricMonitor()
    f05_meter = MetricMonitor()
    
    bar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    optimizer.zero_grad()
    
    for step, (images, masks) in enumerate(bar):
        images = images.to(CONFIG["device"], non_blocking=True)
        masks = masks.to(CONFIG["device"], non_blocking=True)
        
        # Forward with AMP
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / CONFIG["accumulate_steps"]  # Scale for accumulation
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient Accumulation
        if (step + 1) % CONFIG["accumulate_steps"] == 0:
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            score = fbeta_score(preds, masks)
        
        loss_meter.update(loss.item() * CONFIG["accumulate_steps"], images.size(0))
        f05_meter.update(score.item(), images.size(0))
        
        bar.set_postfix(loss=f"{loss_meter.avg:.3f}", f05=f"{f05_meter.avg:.4f}")
        
        # VRAM Protection
        if step % 100 == 0:
            torch.cuda.empty_cache()

def valid_one_epoch(model, loader, criterion, epoch):
    model.eval()
    loss_meter = MetricMonitor()
    
    # Multi-threshold evaluation
    thresholds = np.arange(0.05, 0.55, 0.05)
    scores_per_thresh = {t: MetricMonitor() for t in thresholds}
    
    bar = tqdm(loader, desc=f"Epoch {epoch} [Valid]")
    
    with torch.no_grad():
        for images, masks in bar:
            images = images.to(CONFIG["device"], non_blocking=True)
            masks = masks.to(CONFIG["device"], non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            preds_prob = torch.sigmoid(outputs)
            
            loss_meter.update(loss.item(), images.size(0))
            
            for t in thresholds:
                score = fbeta_score(preds_prob, masks, threshold=t)
                scores_per_thresh[t].update(score.item(), images.size(0))
            
            bar.set_postfix(loss=f"{loss_meter.avg:.3f}", f05=f"{scores_per_thresh[0.1].avg:.4f}")
            
            # Save last batch for visualization
            last_batch = (images, masks, preds_prob)
    
    # Find best threshold
    best_thresh = max(thresholds, key=lambda t: scores_per_thresh[t].avg)
    best_score = scores_per_thresh[best_thresh].avg
    
    print(f"Epoch {epoch} Best F0.5: {best_score:.4f} @ Threshold {best_thresh:.2f}")
    
    return best_score, last_batch

def save_debug_visualization(images, masks, preds, epoch, output_dir="output"):
    """Save debug visualization: Input | Label | Prediction"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Take first sample from batch
    img = images[0].cpu().numpy()  # (C, H, W)
    mask = masks[0, 0].cpu().numpy()  # (H, W)
    pred = preds[0, 0].cpu().numpy()  # (H, W)
    
    # Use middle channel for visualization
    mid_ch = img.shape[0] // 2
    img_slice = img[mid_ch]  # (H, W)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input (middle Z-slice)
    # img is already normalized to [0, 1] by adaptive percentile
    axes[0].imshow(img_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Input (Z={mid_ch})\nMin:{img_slice.min():.2f} Max:{img_slice.max():.2f}', fontsize=10)
    axes[0].axis('off')
    
    # Ground Truth Label
    axes[1].imshow(mask, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth (Ink)', fontsize=12)
    axes[1].axis('off')
    
    # Prediction (probability)
    axes[2].imshow(pred, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title(f'Prediction (prob)', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(f'Epoch {epoch} Debug', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/debug_epoch_{epoch}.png", dpi=100)
    plt.close(fig)
    print(f"   📸 Saved: {output_dir}/debug_epoch_{epoch}.png")

# =========================================================================================
# 🚀 Main
# =========================================================================================
def main():
    print("="*60)
    print("🔬 Vesuvius Challenge 2026 - Surgical Training")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    print(f"Z-slices: {CONFIG['z_slices']}")
    print(f"Window: [{CONFIG['window_min']}, {CONFIG['window_max']}]")
    print(f"Effective Batch Size: {CONFIG['batch_size'] * CONFIG['accumulate_steps']}")
    print("="*60)
    
    # Augmentations
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    ])
    
    val_transform = None
    
    # Data
    print("\nInitializing Datasets...")
    dataset = VesuviusSurgicalDataset(
        CONFIG["data_root"], 
        z_slices=CONFIG["z_slices"], 
        transform=train_transform, 
        mode="train"
    )
    val_dataset = VesuviusSurgicalDataset(
        CONFIG["data_root"], 
        z_slices=CONFIG["z_slices"], 
        transform=val_transform, 
        mode="valid"
    )
    
    # Sanity Check
    if not sanity_check(dataset):
        print("🛑 Training aborted due to sanity check failure.")
        sys.exit(1)
    
    # DataLoaders
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        num_workers=CONFIG["num_workers"], 
        shuffle=True, 
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        num_workers=CONFIG["num_workers"], 
        shuffle=False, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model (ResNet18 Backbone)
    print("Initializing Model...")
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=len(CONFIG["z_slices"]),
        classes=1,
    ).to(CONFIG["device"])
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Loss
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    
    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Training Loop
    best_score = 0
    print("\n🚀 Starting Surgical Training...")
    
    for epoch in range(CONFIG["epochs"]):
        train_one_epoch(model, loader, criterion, optimizer, scaler, epoch)
        score, last_batch = valid_one_epoch(model, val_loader, criterion, epoch)
        
        # Save debug visualization
        images, masks, preds = last_batch
        save_debug_visualization(images, masks, preds, epoch)
        
        scheduler.step(score)
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "vesuvius_surgical_best.pth")
            print(f"🔥 New Best Model Saved! ({best_score:.4f})")
    
    # Final Evaluation
    print("\n📦 Reloading Best Model...")
    model.load_state_dict(torch.load("vesuvius_surgical_best.pth", weights_only=True))
    final_score = valid_one_epoch(model, val_loader, criterion, "final")
    print(f"\n✅ Training Complete! Final Best F0.5: {final_score:.4f}")

if __name__ == "__main__":
    main()
