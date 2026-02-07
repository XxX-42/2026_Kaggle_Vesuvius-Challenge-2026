
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

CONFIG = {
    "data_root": "data/native/train/1/surface_volume",
    "tif_name": "57.tif"
}

def check_coverage():
    tif_path = os.path.join(CONFIG["data_root"], CONFIG["tif_name"])
    if not os.path.exists(tif_path):
        print(f"❌ {tif_path} not found")
        return

    print(f"📦 Loading {tif_path}...")
    with Image.open(tif_path) as img:
        # Load low-res resize to check global coverage map
        w, h = img.size
        img_small = img.resize((w//10, h//10), Image.NEAREST)
        data = np.array(img_small)
    
    print(f"   Original Size: {w}x{h}")
    print(f"   Analysis Size: {data.shape}")
    
    # Check non-zero pixels
    non_zero = data > 0
    coverage = non_zero.sum() / non_zero.size
    
    print(f"   Non-zero Coverage: {coverage*100:.2f}%")
    print(f"   Black Pixels: {(1-coverage)*100:.2f}%")
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.imshow(non_zero, cmap='gray')
    plt.title(f"TIF Data Coverage (White=Data, Black=Empty)\nCoverage: {coverage*100:.2f}%")
    plt.savefig("data_coverage.png")
    print("📸 Saved coverage map to 'data_coverage.png'")
    
    # Compare with Mask
    mask_path = "data/native/train/1/mask.png"
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        mask_coverage = (mask > 0).sum() / mask.size
        print(f"\n🎭 Mask File Checking:")
        print(f"   Mask Coverage: {mask_coverage*100:.2f}%")
        
        if mask_coverage > 99 and coverage < 50:
            print("\n🚨 CRITICAL MISMATCH:")
            print("   Mask says 'EVERYTHING IS VALID' (100%)")
            print(f"   But TIF only has data in {coverage*100:.0f}% of area.")
            print("   -> Random crops will be BLACK {(1-coverage)*100:.0f}% of the time!")

if __name__ == "__main__":
    check_coverage()
