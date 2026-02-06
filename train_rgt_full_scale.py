"""
Vesuvius Challenge 2026 - Full-Scale RGT Training
==================================================
Intensive training on RGT-flattened data with 3x data expansion and heavy augmentation.

Strategy:
- Load 3 adjacent layers (layer_06, 07, 08) as independent samples
- Heavy augmentation with Albumentations
- 30 epochs with CosineAnnealing scheduler
- Continue from fine-tuned model

Target: RTX 3060 6GB VRAM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import cv2
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import gc

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    # Data
    "layer_dir": Path("flattened_layers"),
    "layers": ["layer_06.png", "layer_07.png", "layer_08.png"],  # 3x expansion
    "mask_path": "data/native/train/1/inklabels.png",
    
    # Model
    "pretrained_path": "models/vesuvius_rgt_finetuned.pth",
    "encoder": "resnet18",
    "in_channels": 14,
    
    # Training
    "patch_size": 224,
    "batch_size": 8,  # Safe for 6GB VRAM
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    
    # Scheduler
    "T_0": 10,
    "T_mult": 2,
    
    # Filtering
    "min_ink_ratio": 0.005,  # Keep patches with >0.5% ink
    "min_texture_ratio": 0.1,  # Or patches with >10% paper texture
    
    # Output
    "output_dir": Path("output"),
    "save_path": "models/vesuvius_rgt_fullscale.pth",
    
    # Validation
    "val_split": 0.1,
}

# =============================================================================
# 🔧 Heavy Augmentation Pipeline
# =============================================================================
def get_train_augmentations():
    """Heavy augmentation for RGT domain."""
    return A.Compose([
        # Geometric
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
        
        # Non-linear distortion (simulate scroll curvature)
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=120, sigma=6, p=1.0),
        ], p=0.3),
        
        # Intensity
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(5, 25), p=0.3),
        
        # Blur (simulate focus variation)
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
    ])

def get_val_augmentations():
    """No augmentation for validation."""
    return A.Compose([])

# =============================================================================
# 🔧 Dataset with Multi-Layer Support
# =============================================================================
class RGTMultiLayerDataset(Dataset):
    """
    Dataset that loads multiple RGT layers as independent samples.
    All layers share the same ground truth mask.
    """
    
    def __init__(self, images, mask, patch_size=224, in_channels=14,
                 min_ink_ratio=0.005, min_texture_ratio=0.1,
                 augmentations=None, is_train=True):
        """
        Args:
            images: list of (H, W) normalized images
            mask: (H, W) binary mask
        """
        self.images = images
        self.mask = mask
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.augmentations = augmentations
        self.is_train = is_train
        
        H, W = mask.shape
        self.patches = []  # (layer_idx, y, x)
        
        print("   Scanning for valid patches...")
        
        for layer_idx, img in enumerate(images):
            for y in range(0, H - patch_size, patch_size // 2):
                for x in range(0, W - patch_size, patch_size // 2):
                    mask_patch = mask[y:y+patch_size, x:x+patch_size]
                    img_patch = img[y:y+patch_size, x:x+patch_size]
                    
                    ink_ratio = mask_patch.mean()
                    texture_ratio = (img_patch > 0.1).mean()  # Paper texture
                    
                    # Keep if has ink OR has paper texture (not blank)
                    if ink_ratio > min_ink_ratio or texture_ratio > min_texture_ratio:
                        self.patches.append((layer_idx, y, x))
        
        print(f"   Found {len(self.patches)} valid patches from {len(images)} layers")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        layer_idx, y, x = self.patches[idx]
        ps = self.patch_size
        
        # Extract patches
        img = self.images[layer_idx]
        img_patch = img[y:y+ps, x:x+ps].copy()
        mask_patch = self.mask[y:y+ps, x:x+ps].copy()
        
        # Apply augmentations
        if self.augmentations is not None:
            transformed = self.augmentations(image=img_patch, mask=mask_patch)
            img_patch = transformed['image']
            mask_patch = transformed['mask']
        
        # Replicate to in_channels
        img_multi = np.repeat(img_patch[np.newaxis, :, :], self.in_channels, axis=0)
        
        return (
            torch.from_numpy(img_multi.astype(np.float32)),
            torch.from_numpy(mask_patch[np.newaxis, :, :].astype(np.float32))
        )

# =============================================================================
# 🔧 Loss Function
# =============================================================================
class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss."""
    
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        
        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss

# =============================================================================
# 🔧 Model Loading
# =============================================================================
def load_model(pretrained_path, encoder, in_channels, device):
    """Load pretrained model."""
    print(f"\n📥 Loading model: {pretrained_path}")
    
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None
    )
    
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    return model

# =============================================================================
# 🔧 Training Functions
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="   Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
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
        
        # Memory management
        del images, masks, outputs, loss
        torch.cuda.empty_cache()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate model."""
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
# 🔧 Visualization
# =============================================================================
def visualize_epoch(model, images, mask, device, epoch, output_dir, in_channels):
    """Save visualization for epoch."""
    model.eval()
    
    # Find best ink region
    H, W = mask.shape
    ps = 224
    best_y, best_x, best_ink = 0, 0, 0
    
    for y in range(0, H - ps, ps):
        for x in range(0, W - ps, ps):
            ink = mask[y:y+ps, x:x+ps].mean()
            if ink > best_ink:
                best_ink = ink
                best_y, best_x = y, x
    
    # Use middle layer
    img = images[len(images) // 2]
    img_patch = img[best_y:best_y+ps, best_x:best_x+ps]
    mask_patch = mask[best_y:best_y+ps, best_x:best_x+ps]
    
    # Predict
    img_multi = np.repeat(img_patch[np.newaxis, np.newaxis, :, :], in_channels, axis=1)
    img_tensor = torch.from_numpy(img_multi.astype(np.float32)).to(device)
    
    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred_np = pred.cpu().numpy()[0, 0]
    
    # Visualize
    img_vis = (img_patch * 255).astype(np.uint8)
    mask_vis = (mask_patch * 255).astype(np.uint8)
    pred_vis = (pred_np * 255).astype(np.uint8)
    
    combined = np.hstack([img_vis, mask_vis, pred_vis])
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(combined_bgr, f"Epoch {epoch}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(combined_bgr, f"Max: {pred_np.max():.3f}", (ps + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    output_path = output_dir / f"debug_fullscale_epoch_{epoch:02d}.png"
    cv2.imwrite(str(output_path), combined_bgr)
    
    return pred_np.max()

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius - Full-Scale RGT Training")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create directories
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    Path(CONFIG["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load multiple layers (3x data expansion)
    print(f"\n📥 Loading {len(CONFIG['layers'])} layers (3x expansion)...")
    
    images = []
    for layer_name in CONFIG["layers"]:
        path = CONFIG["layer_dir"] / layer_name
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        print(f"   {layer_name}: {img.shape}")
    
    # 2. Load mask
    print(f"\n📥 Loading mask: {CONFIG['mask_path']}")
    mask = cv2.imread(CONFIG["mask_path"], cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    
    # Resize mask to match images
    if mask.shape != images[0].shape:
        print(f"   Resizing mask from {mask.shape} to {images[0].shape}")
        mask = cv2.resize(mask, (images[0].shape[1], images[0].shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    print(f"   Ink ratio: {mask.mean()*100:.2f}%")
    
    # 3. Create datasets
    print("\n📦 Creating datasets...")
    
    # Split patches for train/val
    full_dataset = RGTMultiLayerDataset(
        images, mask,
        patch_size=CONFIG["patch_size"],
        in_channels=CONFIG["in_channels"],
        min_ink_ratio=CONFIG["min_ink_ratio"],
        min_texture_ratio=CONFIG["min_texture_ratio"],
        augmentations=None,
        is_train=True
    )
    
    # Split
    n_total = len(full_dataset.patches)
    n_val = int(n_total * CONFIG["val_split"])
    n_train = n_total - n_val
    
    random.shuffle(full_dataset.patches)
    train_patches = full_dataset.patches[:n_train]
    val_patches = full_dataset.patches[n_train:]
    
    print(f"   Train patches: {n_train}")
    print(f"   Val patches: {n_val}")
    
    # Create separate datasets
    train_dataset = RGTMultiLayerDataset(
        images, mask,
        patch_size=CONFIG["patch_size"],
        in_channels=CONFIG["in_channels"],
        min_ink_ratio=0,  # Already filtered
        min_texture_ratio=0,
        augmentations=get_train_augmentations(),
        is_train=True
    )
    train_dataset.patches = train_patches
    
    val_dataset = RGTMultiLayerDataset(
        images, mask,
        patch_size=CONFIG["patch_size"],
        in_channels=CONFIG["in_channels"],
        min_ink_ratio=0,
        min_texture_ratio=0,
        augmentations=get_val_augmentations(),
        is_train=False
    )
    val_dataset.patches = val_patches
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 4. Load model
    model = load_model(
        CONFIG["pretrained_path"],
        CONFIG["encoder"],
        CONFIG["in_channels"],
        device
    )
    
    # 5. Setup training
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG["T_0"],
        T_mult=CONFIG["T_mult"]
    )
    
    criterion = DiceBCELoss(dice_weight=0.5)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # 6. Training loop
    print(f"\n🏋️ Training for {CONFIG['epochs']} epochs...")
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{CONFIG['epochs']} (LR: {scheduler.get_last_lr()[0]:.2e})")
        print('='*50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        
        # Visualize
        max_pred = visualize_epoch(
            model, images, mask, device, epoch, 
            CONFIG["output_dir"], CONFIG["in_channels"]
        )
        print(f"   Max Pred: {max_pred:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG,
            }, CONFIG["save_path"])
            
            print(f"   💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("📊 Training Complete!")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Model: {CONFIG['save_path']}")
    print(f"   Debug images: {CONFIG['output_dir']}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
