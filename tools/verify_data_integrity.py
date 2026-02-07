import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def verify_integrity(base_path):
    base_path = Path(base_path)
    print(f"🕵️ Scanning {base_path} for PNG files...")
    
    png_files = list(base_path.glob("**/*.png"))
    
    candidates = []
    
    print(f"{'File':<40} | {'Shape':<15} | {'Density':<10} | {'Status'}")
    print("-" * 80)
    
    for f in png_files:
        if "debug" in f.name or "hist" in f.name:
            continue
            
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        non_zero = np.sum(img > 0)
        total = img.size
        density = non_zero / total
        
        status = "Unknown"
        if density > 0.20:
            status = "❌ Invalid (Geometric?)"
        elif 0.001 < density < 0.10:
            status = "✅ Potential Ink"
            candidates.append(f)
        elif density <= 0.001:
            status = "⚠️ Too Sparse"
            
        print(f"{f.name:<40} | {str(img.shape):<15} | {density*100:6.2f}%    | {status}")
        
        # Save histogram for candidates
        if status == "✅ Potential Ink":
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title("Visual Content")
            plt.subplot(1, 2, 2)
            plt.hist(img.flatten(), bins=50, log=True)
            plt.title("Pixel Value Dist (Log)")
            plt.savefig(PROJECT_ROOT / "outputs" / f"audit_{f.stem}.png")
            plt.close()

    print("\n" + "="*80)
    if candidates:
        print(f"🎉 Found {len(candidates)} potential real ink files:")
        for c in candidates:
            print(f"   -> {c}")
    else:
        print("😱 CRITICAL: No sparse ink files found! You may need to re-download the dataset.")

if __name__ == "__main__":
    verify_integrity(sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "data")
