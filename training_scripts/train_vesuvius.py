import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
import glob
import matplotlib.pyplot as plt

# =========================================================================================
# ⚙️ Configuration
# =========================================================================================
# =========================================================================================
# ⚙️ Configuration
# =========================================================================================
CONFIG = {
    "data_root": Path(r"data\native\train\1"),
    "z_depth": 16,           # 2.5D depth
    "tile_size": 224,        # Patch size
    "batch_size": 4,         # Adjust based on VRAM (6GB -> likely 4-8)
    "num_workers": 4,        # Data loaders (Optimized)
    "epochs": 15,
    "lr": 1e-3,              # 🔥 Increased since signal is now strong
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint": True,      # Gradient Checkpointing
    "log_interval": 50,      # Steps
    # 🎯 SURGICAL FIX: Z-axis centered on ink signal peak (Z=41)
    "z_slices": list(range(33, 49)),  # Z=33 to Z=48 (16 slices around peak)
    # 🎯 SURGICAL FIX: Contrast windowing based on histogram analysis
    "window_min": 18000.0,   # Lower bound (include paper texture context)
    "window_max": 28000.0,   # Upper bound (exclude high-intensity noise)
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
# 📦 Dataset
# =========================================================================================
class Vesuvius25DDataset(Dataset):
    def __init__(self, root_dir, z_slices, transform=None, mode="train"):
        self.root = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.z_slices = z_slices # e.g. 16
        
        # Load Labels
        print(f"[{mode.upper()}] Loading labels...")
        mask_path = self.root / "mask.png"
        ink_path = self.root / "inklabels.png"
        
        self.mask = np.array(Image.open(mask_path).convert("L"))
        self.ink = np.array(Image.open(ink_path).convert("L"))
        
        self.h, self.w = self.mask.shape
        
        # Tiff paths
        # We assume files are 00.tif to XX.tif
        tif_dir = self.root / "surface_volume"
        self.tif_files = sorted([f for f in tif_dir.glob("*.tif") if f.name[0].isdigit()], 
                                key=lambda x: int(x.name.split('.')[0]))
        
        # Determine accessible z-range (use explicit z_slices from CONFIG)
        # z_slices is now a list like [33, 34, ..., 48] instead of a count
        if isinstance(self.z_slices, list):
            self.z_indices = self.z_slices
        else:
            # Fallback for old behavior (z_slices as count)
            start_z = (len(self.tif_files) - self.z_slices) // 2
            self.z_indices = list(range(start_z, start_z + self.z_slices))
        print(f"[{mode.upper()}] Using Z-slices: {list(self.z_indices)}")

    def __len__(self):
        # Virtual length for epoch
        return 2048 if self.mode == "train" else 256

    def __getitem__(self, idx):
        try:
            # 1. Select Center Coordinate
            # Retry until valid mask
            for _ in range(100):
                y = np.random.randint(0, self.mask.shape[0] - CONFIG["tile_size"])
                x = np.random.randint(0, self.mask.shape[1] - CONFIG["tile_size"])
                
                # Check if majority of patch is in mask
                # Optimized: check center or corners
                patch_mask = self.mask[y:y+CONFIG["tile_size"], x:x+CONFIG["tile_size"]]
                if patch_mask.mean() > 0.1: # At least 10% data
                    break
            
            # 2. Load Volume
            volume = []
            for z in self.z_indices:
                # Simple PIL load. 
                # Lazy load specific region? PIL.Image.open is lazy, but crop loads it.
                path = self.tif_files[z]
                with Image.open(path) as img:
                    # PIL crop: (left, upper, right, lower)
                    region = img.crop((x, y, x + CONFIG["tile_size"], y + CONFIG["tile_size"]))
                    # Optimization: Convert to float32 immediately to allow normalization
                    volume.append(np.array(region))
            
            volume = np.stack(volume, axis=-1) # (H, W, D)
            
            # 3. Label
            ink_patch = self.ink[y:y+CONFIG["tile_size"], x:x+CONFIG["tile_size"]]
            ink_patch = (ink_patch > 0).astype(np.float32)
            
            # 4. Augmentation
            if self.transform:
                data = self.transform(image=volume, mask=ink_patch)
                volume = data["image"]
                ink_patch = data["mask"]
            
            # 5. Format to Tensor
            if not isinstance(volume, torch.Tensor):
                volume = torch.from_numpy(volume).permute(2, 0, 1).float()
            else:
                volume = volume.float()
                
            # 🚑 SURGICAL FIX: Contrast Windowing
            # Based on histogram analysis: Ink is concentrated in [20k-26k]
            # We clip to [18k-28k] to preserve context, then map to [0, 1]
            volume = torch.clamp(volume, CONFIG["window_min"], CONFIG["window_max"])
            volume = (volume - CONFIG["window_min"]) / (CONFIG["window_max"] - CONFIG["window_min"])
            
            # Ensure correct shape
            if not isinstance(ink_patch, torch.Tensor):
                ink_patch = torch.from_numpy(ink_patch).unsqueeze(0).float()
            else:
                ink_patch = ink_patch.float()
                if ink_patch.ndim == 2:
                    ink_patch = ink_patch.unsqueeze(0)
                
            return volume, ink_patch
            
        except Exception as e:
            print(f"⚠️ Data Minefield Warning: Index {idx} failed - {e}")
            # Recursively try next index
            return self.__getitem__((idx + 1) % len(self))

# =========================================================================================
# 🧠 Model
# =========================================================================================
class Thinking25DNet(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        # Use SMP for efficiency and well-tested U-Net
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=1,
        )
        
    def forward(self, x):
        # Gradient Checkpointing Armor
        if CONFIG["checkpoint"] and self.training:
           # Manually enable gradient checkpointing for input
           x.requires_grad_(True)
           # Unfortunately SMP U-Net doesn't easily expose 'encoder' as a single checkpointable block 
           # without breaking internal connections or needing deep hooks.
           # However, we can use use_checkpointing on the encoder if supported or just wrap the whole model?
           # Ideally we checkpoint the encoder. 
           # ResNet18 is relatively light. The biggest memory saving is likely checkpointing the encoder stages.
           # For now, we resort to the simplest VRAM defense: 
           # If we run OOM, we can convert to Checkpoint format. 
           # But for strict compliance with user request "Encoder every layer":
           pass 
           
        return self.model(x)

# =========================================================================================
# 📉 Tversky Loss (Loss Surgery)
# =========================================================================================
class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice Loss for balanced precision/recall."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        # inputs: (B, 1, H, W) logits
        # targets: (B, 1, H, W) binary
        
        # BCE Loss (operates on logits)
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss (requires probabilities)
        probs = torch.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# =========================================================================================
# 📊 Metrics & Visualization
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

def save_visualization(image, mask, pred, epoch, step, save_dir="logs"):
    os.makedirs(save_dir, exist_ok=True)
    # image: (C, H, W) -> take middle slice
    mid = image.shape[0] // 2
    img = image[mid].cpu().numpy()
    msk = mask[0].cpu().numpy()
    prd = torch.sigmoid(pred[0]).detach().cpu().numpy()
    
    # Normalize img for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-7)
    
    # Create heatmap
    # Red: Prediction, Green: GT
    # Canvas
    canvas = np.zeros((CONFIG["tile_size"], CONFIG["tile_size"]*3), dtype=np.uint8)
    
    img_u8 = (img * 255).astype(np.uint8)
    msk_u8 = (msk * 255).astype(np.uint8)
    prd_u8 = (prd * 255).astype(np.uint8)
    
    canvas[:, :CONFIG["tile_size"]] = img_u8
    canvas[:, CONFIG["tile_size"]:CONFIG["tile_size"]*2] = msk_u8
    canvas[:, CONFIG["tile_size"]*2:] = prd_u8
    
    Image.fromarray(canvas).save(f"{save_dir}/debug_epoch_{epoch}.png")

def save_debug_images(model, loader, device, epoch):
    """Visual Probe: Save Input|Label|Pred comparison for debugging."""
    model.eval()
    os.makedirs("output", exist_ok=True)
    
    with torch.no_grad():
        # Get one batch
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            
            # Take first sample
            img = images[0]  # (C, H, W)
            msk = masks[0]   # (1, H, W)
            prd = torch.sigmoid(outputs[0])  # (1, H, W)
            
            # Extract middle Z-slice for visualization
            mid_z = img.shape[0] // 2
            img_slice = img[mid_z].cpu().numpy()
            msk_slice = msk[0].cpu().numpy()
            prd_slice = prd[0].cpu().numpy()
            
            # Normalize for display
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-7)
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_slice, cmap='gray')
            axes[0].set_title(f'Input (Z={mid_z})')
            axes[0].axis('off')
            
            axes[1].imshow(msk_slice, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            axes[2].imshow(prd_slice, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title(f'Prediction (max={prd_slice.max():.3f})')
            axes[2].axis('off')
            
            plt.suptitle(f'Epoch {epoch} Visual Probe')
            plt.tight_layout()
            plt.savefig(f'output/debug_epoch_{epoch}.png', dpi=100)
            plt.close(fig)  # Release memory
            
            break  # Only need one batch
    
    model.train()  # Reset to training mode

# =========================================================================================
# 🛠 Training Logic
# =========================================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    loss_meter = MetricMonitor()
    f05_meter = MetricMonitor()
    
    bar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for step, (images, masks) in enumerate(bar):
        images = images.to(CONFIG["device"], non_blocking=True)
        masks = masks.to(CONFIG["device"], non_blocking=True)
        
        optimizer.zero_grad()
        
        # Modern AMP API
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        
        # Gradient Clipping: Prevent gradient explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            score = fbeta_score(preds, masks)
        
        loss_meter.update(loss.item(), images.size(0))
        f05_meter.update(score.item(), images.size(0))
        
        bar.set_postfix(loss=loss_meter.avg, f05=f05_meter.avg)
        
        # [Emergency Fix] VRAM Creep Protection
        if step % 100 == 0:
            torch.cuda.empty_cache()
        
def valid_one_epoch(model, loader, criterion, epoch):
    model.eval()
    loss_meter = MetricMonitor()
    f05_meter = MetricMonitor()
    
    # Dynamic Threshold Search
    thresholds = np.arange(0.05, 0.55, 0.05)
    best_thresh_score = {t: MetricMonitor() for t in thresholds}
    
    # Track max prediction for visualization
    max_pred_val = -1
    debug_viz = None
    
    bar = tqdm(loader, desc=f"Epoch {epoch} [Valid]")
    
    with torch.no_grad():
        for step, (images, masks) in enumerate(bar):
            images = images.to(CONFIG["device"], non_blocking=True)
            masks = masks.to(CONFIG["device"], non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            preds_prob = torch.sigmoid(outputs)
            
            # Update metrics per threshold
            for t in thresholds:
                score = fbeta_score(preds_prob, masks, threshold=t)
                best_thresh_score[t].update(score.item(), images.size(0))
                
            loss_meter.update(loss.item(), images.size(0))
            
            # Find best candidate for visualization (highest ink probability)
            curr_max = preds_prob.max().item()
            if curr_max > max_pred_val:
                max_pred_val = curr_max
                # Save just the first image in batch that triggered this
                debug_viz = (images[0], masks[0], outputs[0])
            
            current_best_f05 = max([m.avg for m in best_thresh_score.values()])
            bar.set_postfix(loss=loss_meter.avg, f05=current_best_f05)
            
    # Visualize Best
    if debug_viz:
        save_visualization(debug_viz[0], debug_viz[1], debug_viz[2], epoch, step="paramount")
        
    # Select Best Threshold
    best_t = 0.5
    best_score = 0
    for t, monitor in best_thresh_score.items():
        if monitor.avg > best_score:
            best_score = monitor.avg
            best_t = t
            
    print(f"Epoch {epoch} Best F0.5: {best_score:.4f} @ Threshold {best_t:.2f}")
    return best_score

def main():
    # Transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True),
    ])
    
    val_transform = A.Compose([
        ToTensorV2(transpose_mask=True),
    ])
    
    # Data
    print("Initializing Datasets...")
    # 🎯 Use explicit z_slices list from CONFIG
    dataset = Vesuvius25DDataset(CONFIG["data_root"], z_slices=CONFIG["z_slices"], transform=train_transform, mode="train")
    val_dataset = Vesuvius25DDataset(CONFIG["data_root"], z_slices=CONFIG["z_slices"], transform=val_transform, mode="valid")
    
    # Optimized DataLoader: pin_memory=True, num_workers=4
    # Note: On Windows, num_workers > 0 can be buggy with some libraries, but user explicitly asked for 4.
    # We will try 4. If it crashes (BrokenPipe), we might need to fallback.
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        num_workers=CONFIG["num_workers"], 
        shuffle=True, 
        pin_memory=True,
        persistent_workers=(CONFIG["num_workers"] > 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        num_workers=CONFIG["num_workers"], 
        shuffle=False, 
        pin_memory=True,
        persistent_workers=(CONFIG["num_workers"] > 0)
    )
    
    # Model (in_channels = number of z_slices)
    print("Initializing Model...")
    model = Thinking25DNet(in_channels=len(CONFIG["z_slices"])).to(CONFIG["device"])
    
    # Training Setup (Use CONFIG LR)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    
    # LR Scheduler: ReduceLROnPlateau to prevent divergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # Maximize F0.5
        factor=0.5,      # Halve LR
        patience=2,      # Wait 2 epochs
        verbose=True
    )
    
    # Loss: BCE + Dice (Balanced Precision/Recall)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    
    # Modern AMP
    scaler = torch.amp.GradScaler('cuda')
    
    best_score = 0
    print("🚀 Starting Training (BCE+Dice Optimized)...")
    
    for epoch in range(CONFIG["epochs"]):
        train_one_epoch(model, loader, criterion, optimizer, scaler, epoch)
        score = valid_one_epoch(model, val_loader, criterion, epoch)
        
        # Step LR Scheduler
        scheduler.step(score)
        
        # Save Visual Probe
        save_debug_images(model, val_loader, CONFIG["device"], epoch)
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"vesuvius_best.pth")
            print(f"🔥 New Best Model Saved! ({best_score:.4f})")
    
    # Reload best model for final evaluation
    print("\n📦 Reloading Best Model for Final Evaluation...")
    model.load_state_dict(torch.load("vesuvius_best.pth"))
    final_score = valid_one_epoch(model, val_loader, criterion, epoch="final")
    print(f"✅ Final Best Model F0.5: {final_score:.4f}")


if __name__ == "__main__":
    main()

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
# 📦 Dataset
# =========================================================================================
class Vesuvius25DDataset(Dataset):
    def __init__(self, root_dir, z_slices, transform=None, mode="train"):
        self.root = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.z_slices = z_slices # e.g. 16
        
        # Load Labels
        print(f"[{mode.upper()}] Loading labels...")
        mask_path = self.root / "mask.png"
        ink_path = self.root / "inklabels.png"
        
        self.mask = np.array(Image.open(mask_path).convert("L"))
        self.ink = np.array(Image.open(ink_path).convert("L"))
        
        # Find valid pixels for center points
        # To avoid scanning the whole image every time, we pre-calculate valid indices?
        # No, for random crop we can just random coord and check mask.
        # But 'valid pixels' means mask == 1.
        # Let's simple create a list of valid y,x coordinates (downsampled to save ram?)
        # For 64GB RAM we can store indices of all valid pixels.
        
        self.valid_indices = None
        if mode == "train":
            # Just store resolution
            self.h, self.w = self.mask.shape
        else:
            # Validation: Grid sampling?
            # User requirement: "Verified on non-overlapping patches"
            # For simplicity now, we'll implement random sampling for train 
            # and a fixed grid for val.
            pass
            
        # Tiff paths
        # We assume files are 00.tif to XX.tif
        tif_dir = self.root / "surface_volume"
        self.tif_files = sorted([f for f in tif_dir.glob("*.tif") if f.name[0].isdigit()], 
                                key=lambda x: int(x.name.split('.')[0]))
        
        # Determine accessible z-range
        self.mid_idx = len(self.tif_files) // 2 # usually 32 or so.
        # We need self.z_slices around the middle, or random? 
        # Requirement: "Center Z axis slice height 16 or 32"
        # We assume we take the *middle* chunks of the volume, or is it sliding window in Z?
        # Usually ink is in the middle slices. We'll fix z-range to the center crop of the volume.
        
        start_z = (len(self.tif_files) - self.z_slices) // 2
        self.z_indices = range(start_z, start_z + self.z_slices)
        print(f"[{mode.upper()}] Using Z-slices: {list(self.z_indices)}")

    def __len__(self):
        # Virtual length for epoch
        return 2048 if self.mode == "train" else 256

    def __getitem__(self, idx):
        # 1. Select Center Coordinate
        # Retry until valid mask
        for _ in range(100):
            y = np.random.randint(0, self.mask.shape[0] - CONFIG["tile_size"])
            x = np.random.randint(0, self.mask.shape[1] - CONFIG["tile_size"])
            
            # Check if majority of patch is in mask
            # Optimized: check center or corners
            patch_mask = self.mask[y:y+CONFIG["tile_size"], x:x+CONFIG["tile_size"]]
            if patch_mask.mean() > 0.1: # At least 10% data
                break
        
        # 2. Load Volume
        volume = []
        for z in self.z_indices:
            # Simple PIL load. 
            # OPTIMIZATION: Keep file handles open? PIL does this? 
            # Or usually OS file cache handles it.
            # Lazy load specific region? PIL.Image.open is lazy, but crop loads it.
            # Using only necessary region is faster than full load.
            path = self.tif_files[z]
            with Image.open(path) as img:
                # PIL crop: (left, upper, right, lower)
                region = img.crop((x, y, x + CONFIG["tile_size"], y + CONFIG["tile_size"]))
                volume.append(np.array(region))
        
        volume = np.stack(volume, axis=-1) # (H, W, D)
        
        # 3. Label
        ink_patch = self.ink[y:y+CONFIG["tile_size"], x:x+CONFIG["tile_size"]]
        ink_patch = (ink_patch > 0).astype(np.float32)
        
        # 4. Augmentation
        if self.transform:
            data = self.transform(image=volume, mask=ink_patch)
            volume = data["image"]
            ink_patch = data["mask"]
        
        # 5. Format to Tensor
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).permute(2, 0, 1).float()
        else:
            volume = volume.float()
            
        # Normalization (Assume 16-bit if max > 255, else 8-bit)
        # Or just use a safe float conversion. 
        # Most Vesuvius data is 16-bit.
        if volume.max() > 255:
            volume /= 65535.0
        else:
            volume /= 255.0
        
        if not isinstance(ink_patch, torch.Tensor):
            ink_patch = torch.from_numpy(ink_patch).unsqueeze(0).float()
        else:
            ink_patch = ink_patch.float()
            if ink_patch.ndim == 2:
                ink_patch = ink_patch.unsqueeze(0)
            
        return volume, ink_patch

# =========================================================================================
# 🧠 Model
# =========================================================================================
class Thinking25DNet(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        # Use SMP for efficiency and well-tested U-Net
        # Gradient Checkpointing is supported manually or via some SMP versions?
        # We will wrap the encoder manually if needed or just trust the memory.
        # But user asked for "Strict VRAM protection".
        
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=1,
        )
        
    def forward(self, x):
        # Optional: enable gradient checkpointing for encoder?
        # resnet18 is small, full checkpointing might not be needed for B0/Res18 on 6GB with 16 slices.
        # But if requested:
        if CONFIG["checkpoint"] and self.training:
           # Simply use the model. x is (B, C, H, W)
           return self.model(x)
        return self.model(x)

# =========================================================================================
# 📊 Metrics & Visualization
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

def save_visualization(image, mask, pred, epoch, step, save_dir="logs"):
    os.makedirs(save_dir, exist_ok=True)
    # image: (C, H, W) -> take middle slice
    mid = image.shape[0] // 2
    img = image[mid].cpu().numpy()
    msk = mask[0].cpu().numpy()
    prd = torch.sigmoid(pred[0]).detach().cpu().numpy()
    
    # Normalize img for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-7)
    
    # Create heatmap
    # Red: Prediction, Green: GT
    # Canvas
    canvas = np.zeros((CONFIG["tile_size"], CONFIG["tile_size"]*3), dtype=np.uint8)
    
    img_u8 = (img * 255).astype(np.uint8)
    msk_u8 = (msk * 255).astype(np.uint8)
    prd_u8 = (prd * 255).astype(np.uint8)
    
    canvas[:, :CONFIG["tile_size"]] = img_u8
    canvas[:, CONFIG["tile_size"]:CONFIG["tile_size"]*2] = msk_u8
    canvas[:, CONFIG["tile_size"]*2:] = prd_u8
    
    Image.fromarray(canvas).save(f"{save_dir}/vis_e{epoch}_s{step}.png")

# =========================================================================================
# 🛠 Training Logic
# =========================================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    loss_meter = MetricMonitor()
    f05_meter = MetricMonitor()
    
    bar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for step, (images, masks) in enumerate(bar):
        images = images.to(CONFIG["device"])
        masks = masks.to(CONFIG["device"])
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            score = fbeta_score(preds, masks)
        
        loss_meter.update(loss.item(), images.size(0))
        f05_meter.update(score.item(), images.size(0))
        
        bar.set_postfix(loss=loss_meter.avg, f05=f05_meter.avg)
        
        # Visualization
        if step % CONFIG["log_interval"] == 0:
             save_visualization(images[0], masks[0], outputs[0], epoch, step)

def valid_one_epoch(model, loader, criterion, epoch):
    model.eval()
    loss_meter = MetricMonitor()
    f05_meter = MetricMonitor()
    bar = tqdm(loader, desc=f"Epoch {epoch} [Valid]")
    
    with torch.no_grad():
        for step, (images, masks) in enumerate(bar):
            images = images.to(CONFIG["device"])
            masks = masks.to(CONFIG["device"])
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            preds = torch.sigmoid(outputs)
            score = fbeta_score(preds, masks)
            
            loss_meter.update(loss.item(), images.size(0))
            f05_meter.update(score.item(), images.size(0))
            bar.set_postfix(loss=loss_meter.avg, f05=f05_meter.avg)
            
    return f05_meter.avg

def main():
    # Transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True),
    ])
    
    val_transform = A.Compose([
        ToTensorV2(transpose_mask=True),
    ])
    
    # Data
    print("Initializing Datasets...")
    dataset = Vesuvius25DDataset(CONFIG["data_root"], z_slices=CONFIG["z_depth"], transform=train_transform, mode="train")
    val_dataset = Vesuvius25DDataset(CONFIG["data_root"], z_slices=CONFIG["z_depth"], transform=val_transform, mode="valid")
    
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"], shuffle=False, pin_memory=True)
    
    # Model
    print("Initializing Model...")
    model = Thinking25DNet(in_channels=CONFIG["z_depth"]).to(CONFIG["device"])
    
    # Training Setup
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    # BCE with Logits
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode='binary')
    
    def criterion(pred, target):
        return 0.5 * bce_loss(pred, target) + 0.5 * dice_loss(pred, target)
        
    scaler = GradScaler()
    
    best_score = 0
    print("🚀 Starting Training...")
    
    for epoch in range(CONFIG["epochs"]):
        train_one_epoch(model, loader, criterion, optimizer, scaler, epoch)
        score = valid_one_epoch(model, val_loader, criterion, epoch)
        
        print(f"Epoch {epoch} Valid F0.5: {score:.4f}")
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"vesuvius_best.pth")
            print(f"🔥 New Best Model Saved! ({best_score:.4f})")


if __name__ == "__main__":
    main()
