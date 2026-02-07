import sys
import os
import torch
import logging
from torch.cuda.amp import autocast

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.mini_unetr import MiniUNETR
from src.losses.masked_loss import MaskedBCELoss
from src.data.dataset import VesuviusDataset

def verify_phase3():
    print("=== Vesuvius Phase 3 Verification / 第三阶段验证 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset Verification
    print("\n--- 1. Dataset Check (Triple Return) ---")
    ds = VesuviusDataset(data_path="./data")
    img, lbl, ign = ds[0]
    print(f"Image Shape : {img.shape}")
    print(f"Label Shape : {lbl.shape}")
    print(f"Ignore Shape: {ign.shape}")
    assert len(ds[0]) == 3, "Dataset must return (img, label, ignore)"
    
    # 2. Model Forward (MiniUNETR)
    print("\n--- 2. MiniUNETR Forward (with AMP) ---")
    model = MiniUNETR().to(device)
    
    # Batch = 2
    input_tensor = torch.randn(2, 1, 16, 256, 256).to(device)
    
    try:
        with autocast(): # Test Mixed Precision
            pred = model(input_tensor)
        print(f"✅ Forward Pass Success. Output: {pred.shape}")
    except Exception as e:
        print(f"❌ Forward Failed: {e}")
        raise e
        
    # 3. Masked Loss Verification
    print("\n--- 3. Masked Loss Logic ---")
    criterion = MaskedBCELoss(use_ignore_mask=True)
    
    # Mock data
    # Pred: (B, 1, H, W)
    # Target: (B, 1, H, W)
    # Mask: (B, 1, H, W) -> 1 means IGNORE
    pred_logits = torch.randn(2, 1, 256, 256).to(device)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    
    # Case A: Full Ignore (Loss should be 0)
    full_ignore = torch.ones_like(target)
    loss_zero = criterion(pred_logits, target, full_ignore)
    print(f"Loss (Full Ignore): {loss_zero.item()} (Expected ~0.0)")
    
    # Case B: No Ignore
    no_ignore = torch.zeros_like(target)
    loss_full = criterion(pred_logits, target, no_ignore)
    print(f"Loss (No Ignore)  : {loss_full.item()}")
    
    assert loss_zero.item() < 1e-5, "Full ignore mask should result in 0 loss."
    assert loss_full.item() > 0, "Normal loss should be positive."
    
    print("\n✅ Phase 3 All Systems Go!")

if __name__ == "__main__":
    verify_phase3()
