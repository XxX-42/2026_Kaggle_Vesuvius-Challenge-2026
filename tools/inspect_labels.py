import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def inspect_labels(mask_path):
    mask_path = Path(mask_path)
    if not mask_path.exists():
        print(f"❌ Mask not found: {mask_path}")
        return
        
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    print(f"📋 Mask: {mask_path.name}")
    print(f"   Shape: {mask.shape}")
    print(f"   Unique values: {np.unique(mask)}")
    print(f"   Ink pixels: {np.sum(mask > 127)}")
    print(f"   Coverage: {np.mean(mask > 127) * 100:.2f}%")
    
    # Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(mask.flatten(), bins=50)
    plt.title(f"Histogram of {mask_path.name}")
    plt.savefig(PROJECT_ROOT / "outputs" / f"hist_{mask_path.stem}.png")
    print(f"✅ Saved histogram to outputs/hist_{mask_path.stem}.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_labels(sys.argv[1])
    else:
        # Default target
        inspect_labels(PROJECT_ROOT / "data/native/train/1/inklabels.png")
