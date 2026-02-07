
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# CONFIG (Replicated from train_vesuvius_surgical.py)
CONFIG = {
    "tile_size": 224,
    "z_slices": [57], # Test on the peak signal layer
    "data_root": "data/native/train/1"
}

def check_normalization_logic():
    print("🔬 Auditing Normalization Logic...")
    
    # 1. Load Raw TIF
    tif_path = os.path.join(CONFIG["data_root"], "surface_volume", "57.tif")
    if not os.path.exists(tif_path):
        print(f"❌ Error: {tif_path} not found.")
        return
        
    print(f"   Loading {tif_path}...")
    with Image.open(tif_path) as img:
        raw_full = np.array(img).astype(np.float32)
        
    # Crop a center patch (simulating __getitem__)
    h, w = raw_full.shape
    cy, cx = h // 2, w // 2
    raw_patch = raw_full[cy:cy+CONFIG["tile_size"], cx:cx+CONFIG["tile_size"]]
    
    print(f"   Raw Patch Stats: Min={raw_patch.min():.1f}, Max={raw_patch.max():.1f}, Mean={raw_patch.mean():.1f}")
    
    # 2. Apply Surgical Logic (Adaptive Percentile)
    # ---------------------------------------------------------
    # Logic in train_vesuvius_surgical.py:
    # p_min, p_max = np.percentile(volume, (1, 99))
    # volume = np.clip((volume - p_min) / (p_max - p_min + 1e-8), 0, 1)
    # ---------------------------------------------------------
    
    p_min, p_max = np.percentile(raw_patch, (1, 99))
    print(f"   Calculated Percentiles: p1={p_min:.1f}, p99={p_max:.1f}")
    
    norm_patch = np.clip((raw_patch - p_min) / (p_max - p_min + 1e-8), 0, 1)
    
    print(f"   Normalized Stats: Min={norm_patch.min():.4f}, Max={norm_patch.max():.4f}, Mean={norm_patch.mean():.4f}")
    
    # 3. Verification
    if norm_patch.mean() < 0.1:
        print("\n❌ CRITICAL: Result is too dark (Black Hole).")
    elif norm_patch.mean() > 0.9:
        print("\n❌ CRITICAL: Result is washed out (White).")
    else:
        print("\n✅ PASS: Result has good dynamic range.")
        
    # 4. Save visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_patch, cmap='gray')
    plt.title(f"Raw (Min:{raw_patch.min():.0f} Max:{raw_patch.max():.0f})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(norm_patch, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Normalized (Mean:{norm_patch.mean():.2f})")
    plt.axis('off')
    
    plt.savefig("norm_check.png")
    print("\n📸 Saved visual proof to 'norm_check.png'")

if __name__ == "__main__":
    check_normalization_logic()
