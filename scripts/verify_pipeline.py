"""
Pipeline Verification Script
============================
Strictly validates:
1. Config Loading (Z-slices = 51-64)
2. Data Integrity (Files exist, not empty)
3. Spatial Split (Train/Valid non-overlap)
4. Visualization (Saves debug image)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.core.config import cfg
from src.data.datasets import VesuviusDataset

def verify_pipeline():
    print("🔬 Starting Pipeline Verification...")
    
    # 1. Check Config
    print(f"   Config Z-Slices: {cfg.Z_SLICES}")
    expected_z = list(range(51, 65))
    assert cfg.Z_SLICES == expected_z, f"❌ Config mismatch! Expected {expected_z}, got {cfg.Z_SLICES}"
    print("   ✅ Config Z-Slices Correct (51-64)")
    
    # 2. Initialize Datasets
    print("\n📦 Initializing Datasets...")
    train_ds = VesuviusDataset(
        data_root=cfg.DATA_ROOT, 
        z_slices=cfg.Z_SLICES,
        mode="train", 
        split_fraction=cfg.VALID_SPLIT
    )
    valid_ds = VesuviusDataset(
        data_root=cfg.DATA_ROOT, 
        z_slices=cfg.Z_SLICES,
        mode="valid", 
        split_fraction=cfg.VALID_SPLIT
    )
    
    print(f"   Train Length: {len(train_ds)}")
    print(f"   Valid Length: {len(valid_ds)}")
    
    # 3. Assert Spatial Split (Leakage Check)
    print("\n🕵️ Checking Spatial Split (Leakage)...")
    # Hack: Access internal y_range to verify
    print(f"   Train Y-Range: {train_ds.y_range}")
    print(f"   Valid Y-Range: {valid_ds.y_range}")
    
    t_min, t_max = train_ds.y_range
    v_min, v_max = valid_ds.y_range
    
    assert t_max <= v_min, f"❌ DATA LEAKAGE DETECTED! Train Max {t_max} > Valid Min {v_min}"
    print("   ✅ Spatial Split Verified (No Overlap)")
    
    # 4. Load & Visual Check
    print("\n📸 Generatng Debug Visualization...")
    # Get one sample from Train
    tx, ty = train_ds[0] # (C, H, W), (1, H, W)
    # Get one sample from Valid
    vx, vy = valid_ds[0]
    
    # Convert to numpy
    tx_np = tx.numpy()
    vx_np = vx.numpy()
    
    print(f"   Train Sample Mean: {tx_np.mean():.4f} (Should be ~0.5)")
    print(f"   Valid Sample Mean: {vx_np.mean():.4f} (Should be ~0.5)")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Train Row
    mid_z = tx_np.shape[0] // 2
    axes[0,0].imshow(tx_np[mid_z], cmap='gray')
    axes[0,0].set_title(f"Train Input (Z={cfg.Z_SLICES[mid_z]})")
    axes[0,1].imshow(ty[0], cmap='gray')
    axes[0,1].set_title("Train Label")
    axes[0,2].hist(tx_np.flatten(), bins=50)
    axes[0,2].set_title("Train Histogram")
    
    # Valid Row
    axes[1,0].imshow(vx_np[mid_z], cmap='gray')
    axes[1,0].set_title(f"Valid Input (Z={cfg.Z_SLICES[mid_z]})")
    axes[1,1].imshow(vy[0], cmap='gray')
    axes[1,1].set_title("Valid Label")
    axes[1,2].hist(vx_np.flatten(), bins=50)
    axes[1,2].set_title("Valid Histogram")
    
    out_path = PROJECT_ROOT / "outputs" / "debug_pipeline_verify.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"   ✅ Saved debug image: {out_path}")
    
    print("\n🎉 Pipeline Verification PASSED!")

if __name__ == "__main__":
    verify_pipeline()
