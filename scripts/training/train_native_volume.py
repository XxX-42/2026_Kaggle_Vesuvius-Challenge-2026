"""
Vesuvius Challenge 2026 - Native Volume Training (Refactored)
============================================================
Train on ORIGINAL surface_volume data with CORRECT ink labels.
Refactored to use centralized config and modular components.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import cv2
from tqdm import tqdm
import albumentations as A
import gc
from PIL import Image

# Modular imports
from src.core.config import Config
from src.data.native_dataset import NativeVolumeDataset
from src.data.samplers import BalancedSampler
from src.models.segmentation import load_model
from src.core.metrics import DiceBCELoss

# =============================================================================
# 🔧 Data Loading
# =============================================================================
def load_native_volume(volume_dir, z_start, n_channels, window_min, window_max):
    """Load native TIF slices as 3D volume."""
    print(f"📥 Loading native volume from {volume_dir}...")
    
    tif_files = sorted(volume_dir.glob("*.tif"))
    print(f"   Found {len(tif_files)} TIF files")
    
    if not tif_files:
        raise FileNotFoundError(f"No TIF files in {volume_dir}")
    
    # Select slices
    z_end = z_start + n_channels
    selected_files = []
    
    tif_files.sort(key=lambda p: int(p.stem))
    
    for f in tif_files:
        try:
            z_idx = int(f.stem)
            if z_start <= z_idx < z_end:
                selected_files.append((z_idx, f))
        except ValueError:
            continue
            
    print(f"   Requested Z range: {z_start}-{z_end}")
    print(f"   Actual files found: {len(selected_files)}")
    
    if len(selected_files) < n_channels:
        print(f"   ⚠️ Warning: Only found {len(selected_files)}/{n_channels} files")
        # Padding logic could be added here if strictly needed, or fail.
        # For now, we proceed with what we have and let the user know, 
        # or pad. Original script had complex fallback logic.
        # We will implement simple padding to avoid shape mismatch.
    
    # Load first to get dimensions
    first_img = np.array(Image.open(selected_files[0][1]))
    H, W = first_img.shape
    D = len(selected_files)
    
    volume = np.zeros((D, H, W), dtype=np.float32)
    
    for i, (z_idx, path) in enumerate(tqdm(selected_files, desc="   Loading")):
        img = np.array(Image.open(path)).astype(np.float32)
        # Apply windowing
        img = np.clip(img, window_min, window_max)
        img = (img - window_min) / (window_max - window_min)
        volume[i] = img
    
    if D < n_channels:
        print(f"   Padding from {D} to {n_channels} channels")
        pad_total = n_channels - D
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        volume = np.pad(volume, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='reflect')

    return volume

# =============================================================================
# 🔧 Augmentations
# =============================================================================
def get_train_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.5),
    ])

# =============================================================================
# 🔧 Training
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="   Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        del images, masks, outputs, loss
        torch.cuda.empty_cache()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="   Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius - Native Volume Training (Refactored)")
    print("=" * 60)
    
    device = torch.device(Config.DEVICE)
    print(f"\n🖥️ Device: {device}")
    
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(Config.MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load native volume
    volume = load_native_volume(
        Config.VOLUME_DIR,
        Config.Z_START,
        Config.IN_CHANNELS,
        Config.WINDOW_MIN,
        Config.WINDOW_MAX
    )
    
    D, H, W = volume.shape
    print(f"   Final volume: {volume.shape}")
    
    # 2. Load ink mask
    print(f"\n📥 Loading ink mask: {Config.MASK_PATH}")
    ink_mask = cv2.imread(str(Config.MASK_PATH), cv2.IMREAD_GRAYSCALE)
    ink_mask = ink_mask.astype(np.float32) / 255.0
    
    if ink_mask.shape != (H, W):
        print(f"   Resizing mask from {ink_mask.shape} to ({H}, {W})")
        ink_mask = cv2.resize(ink_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # 3. Load paper mask
    if Config.PAPYRUS_MASK_PATH.exists():
        print(f"\n📥 Loading paper mask: {Config.PAPYRUS_MASK_PATH}")
        paper_mask = cv2.imread(str(Config.PAPYRUS_MASK_PATH), cv2.IMREAD_GRAYSCALE)
        paper_mask = paper_mask.astype(np.float32) / 255.0
        if paper_mask.shape != (H, W):
            paper_mask = cv2.resize(paper_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        print("   No paper mask, using full image")
        paper_mask = np.ones((H, W), dtype=np.float32)
    
    # 4. Create dataset
    print("\n📦 Creating dataset...")
    full_dataset = NativeVolumeDataset(
        volume, ink_mask, paper_mask,
        patch_size=Config.PATCH_SIZE,
        min_ink_threshold=Config.MIN_INK_THRESHOLD,
        augmentations=get_train_augmentations()
    )
    
    # 5. Split (Uses Fixed Shuffle to prevent leakage)
    print("   Splitting dataset...")
    train_dataset, val_dataset = full_dataset.split(
        val_split=Config.VAL_SPLIT, 
        shuffle=True
    )
    
    print(f"   Train patches: {len(train_dataset.all_patches)}")
    print(f"   Val patches: {len(val_dataset.all_patches)}")
    
    # 6. Loaders
    train_sampler = BalancedSampler(
        train_dataset,
        Config.BATCH_SIZE,
        Config.INK_RATIO
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 7. Model
    model = load_model(
        Config.PRETRAINED_PATH,
        Config.ENCODER,
        Config.IN_CHANNELS,
        device
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.T_0,
        T_mult=Config.T_MULT
    )
    
    criterion = DiceBCELoss(dice_weight=0.5)
    scaler = torch.cuda.amp.GradScaler()
    
    # 8. Loop
    print(f"\n🏋️ Training for {Config.EPOCHS} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS} (LR: {scheduler.get_last_lr()[0]:.2e})")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, Config.MODEL_SAVE_PATH)
            print(f"   💾 Saved best model!")
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
