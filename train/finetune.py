"""
Vesuvius Challenge 2026 - RGT Domain Fine-tuning
=================================================
Transfer learning from Raw CT domain to RGT-flattened domain.

Problem: Model trained on raw CT performs poorly on RGT-flattened images.
Solution: Fine-tune decoder only on RGT data to adapt feature space.

Strategy:
- Freeze encoder (preserve feature extraction)
- Train decoder only (adapt to new texture distribution)
- Very low LR (1e-5) for 5 epochs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
import random

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    # Input data
    "image_path": "data/flattened_layers/layer_07.png",
    "mask_path": "data/native/train/1/inklabels.png",
    
    # Pretrained model
    "pretrained_path": "models/vesuvius_surgical_best.pth",
    "encoder": "resnet18",
    "in_channels": 14,  # Will replicate single channel 14x
    
    # Training params
    "patch_size": 224,
    "batch_size": 8,
    "epochs": 5,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    
    # Filtering
    "min_ink_ratio": 0.01,  # Only patches with >1% ink
    
    # Output
    "output_dir": Path("finetune_rgt_results"),
    "save_path": "models/vesuvius_rgt_finetuned.pth",
}

# =============================================================================
# 🔧 Dataset
# =============================================================================
class RGTPatchDataset(Dataset):
    """
    Dataset for RGT fine-tuning.
    Extracts patches containing ink from the flattened layer.
    """
    
    def __init__(self, image, mask, patch_size=224, min_ink_ratio=0.01, in_channels=14):
        self.image = image  # (H, W) normalized 0-1
        self.mask = mask    # (H, W) binary 0-1
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        H, W = image.shape
        self.patches = []
        
        # Find patches containing ink
        print("   Scanning for ink-containing patches...")
        for y in range(0, H - patch_size, patch_size // 2):
            for x in range(0, W - patch_size, patch_size // 2):
                mask_patch = mask[y:y+patch_size, x:x+patch_size]
                ink_ratio = mask_patch.mean()
                
                if ink_ratio > min_ink_ratio:
                    self.patches.append((y, x))
        
        print(f"   Found {len(self.patches)} ink-containing patches")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        y, x = self.patches[idx]
        ps = self.patch_size
        
        # Extract patch
        img_patch = self.image[y:y+ps, x:x+ps]
        mask_patch = self.mask[y:y+ps, x:x+ps]
        
        # Replicate to in_channels (model expects 14 channels)
        img_multi = np.repeat(img_patch[np.newaxis, :, :], self.in_channels, axis=0)
        
        # Add random augmentation
        if random.random() > 0.5:
            img_multi = np.flip(img_multi, axis=2).copy()
            mask_patch = np.flip(mask_patch, axis=1).copy()
        if random.random() > 0.5:
            img_multi = np.flip(img_multi, axis=1).copy()
            mask_patch = np.flip(mask_patch, axis=0).copy()
        
        return (
            torch.from_numpy(img_multi).float(),
            torch.from_numpy(mask_patch[np.newaxis, :, :]).float()
        )

# =============================================================================
# 🔧 Model Loading with Encoder Freezing
# =============================================================================
def load_and_prepare_model(pretrained_path, encoder, in_channels, device):
    """
    Load pretrained model and freeze encoder for fine-tuning.
    """
    print(f"\n📥 Loading pretrained model: {pretrained_path}")
    
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
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # ========================================
    # FREEZE ENCODER (Critical for fine-tuning)
    # ========================================
    print("🔒 Freezing encoder layers...")
    encoder_params = 0
    decoder_params = 0
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
            encoder_params += param.numel()
        else:
            param.requires_grad = True
            decoder_params += param.numel()
    
    print(f"   Encoder params (frozen): {encoder_params:,}")
    print(f"   Decoder params (trainable): {decoder_params:,}")
    
    return model

# =============================================================================
# 🔧 Training Loop
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="   Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# =============================================================================
# 🔧 Validation Visualization
# =============================================================================
def visualize_progress(model, image, mask, device, epoch, output_dir, in_channels):
    """
    Visualize prediction on a fixed region each epoch.
    Shows: Input | Ground Truth | Prediction
    """
    model.eval()
    
    # Find a region with good ink coverage
    H, W = image.shape
    best_y, best_x, best_ink = 0, 0, 0
    ps = 224
    
    for y in range(0, H - ps, ps):
        for x in range(0, W - ps, ps):
            ink = mask[y:y+ps, x:x+ps].mean()
            if ink > best_ink:
                best_ink = ink
                best_y, best_x = y, x
    
    # Extract patch
    img_patch = image[best_y:best_y+ps, best_x:best_x+ps]
    mask_patch = mask[best_y:best_y+ps, best_x:best_x+ps]
    
    # Prepare for model
    img_multi = np.repeat(img_patch[np.newaxis, np.newaxis, :, :], in_channels, axis=1)
    img_tensor = torch.from_numpy(img_multi).float().to(device)
    
    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred_np = pred.cpu().numpy()[0, 0]
    
    # Create visualization
    img_vis = (img_patch * 255).astype(np.uint8)
    mask_vis = (mask_patch * 255).astype(np.uint8)
    pred_vis = (pred_np * 255).astype(np.uint8)
    
    # Stack horizontally
    combined = np.hstack([img_vis, mask_vis, pred_vis])
    
    # Add labels
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    cv2.putText(combined_bgr, "Input", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(combined_bgr, "GT", (ps + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(combined_bgr, "Pred", (2*ps + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    output_path = output_dir / f"debug_finetune_epoch_{epoch:02d}.png"
    cv2.imwrite(str(output_path), combined_bgr)
    
    # Print stats
    print(f"   📸 Epoch {epoch}: pred_max={pred_np.max():.4f}, pred_mean={pred_np.mean():.4f}")
    
    return pred_np.max()

# =============================================================================
# 🔧 Loss Function
# =============================================================================
class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation."""
    
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius - RGT Domain Fine-tuning")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print(f"\n📥 Loading image: {CONFIG['image_path']}")
    image = cv2.imread(CONFIG["image_path"], cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255.0
    print(f"   Image shape: {image.shape}")
    
    print(f"\n📥 Loading mask: {CONFIG['mask_path']}")
    mask = cv2.imread(CONFIG["mask_path"], cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    print(f"   Mask shape: {mask.shape}")
    print(f"   Ink ratio: {mask.mean()*100:.2f}%")
    
    # Ensure same size
    if image.shape != mask.shape:
        print(f"\n⚠️ Size mismatch! Resizing mask...")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(f"   Resized mask: {mask.shape}")
    
    # 2. Create dataset
    print("\n📦 Creating patch dataset...")
    dataset = RGTPatchDataset(
        image, mask,
        patch_size=CONFIG["patch_size"],
        min_ink_ratio=CONFIG["min_ink_ratio"],
        in_channels=CONFIG["in_channels"]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # 3. Load and prepare model
    model = load_and_prepare_model(
        CONFIG["pretrained_path"],
        CONFIG["encoder"],
        CONFIG["in_channels"],
        device
    )
    
    # 4. Optimizer (only for decoder params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    
    criterion = DiceBCELoss()
    
    # 5. Initial visualization (before training)
    print("\n📸 Initial prediction (before fine-tuning):")
    visualize_progress(model, image, mask, device, 0, CONFIG["output_dir"], CONFIG["in_channels"])
    
    # 6. Training loop
    print(f"\n🏋️ Fine-tuning for {CONFIG['epochs']} epochs...")
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{CONFIG['epochs']} ---")
        
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"   Loss: {loss:.4f}")
        
        # Visualize
        max_pred = visualize_progress(model, image, mask, device, epoch, CONFIG["output_dir"], CONFIG["in_channels"])
    
    # 7. Save fine-tuned model
    print(f"\n💾 Saving fine-tuned model: {CONFIG['save_path']}")
    Path(CONFIG["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': CONFIG["epochs"],
        'config': CONFIG,
    }, CONFIG["save_path"])
    
    print(f"   ✅ Saved!")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("📊 Fine-tuning Complete!")
    print(f"   Model: {CONFIG['save_path']}")
    print(f"   Debug images: {CONFIG['output_dir']}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
