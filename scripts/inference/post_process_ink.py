"""
Vesuvius Challenge 2026 - Ink Prediction Post-Processing
=========================================================
Enhance and purify weak ink predictions using morphological operations.

Pipeline:
1. Contrast stretching (histogram normalization)
2. Gaussian blur for stroke connection
3. Multi-level thresholding for comparison
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    "input_path": "inference_pred.png",
    "output_path": "inference_post_process.png",
    
    # Gaussian blur kernel
    "blur_kernel": (3, 3),
    
    # Multi-level thresholds
    "thresholds": {
        "loose": 0.10,
        "balanced": 0.15,
        "strict": 0.20,
    }
}

# =============================================================================
# 🔧 Processing Functions
# =============================================================================
def load_and_normalize(path):
    """Load grayscale image and normalize to [0, 1]."""
    print(f"📥 Loading {path}...")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    
    # Convert to float32 [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    print(f"   Shape: {img.shape}")
    print(f"   Original range: [{img_float.min():.4f}, {img_float.max():.4f}]")
    
    return img_float

def contrast_stretch(img):
    """Stretch intensity range from [min, max] to [0, 1]."""
    print("🔧 Applying contrast stretching...")
    
    p_min, p_max = img.min(), img.max()
    
    if p_max - p_min < 1e-6:
        print("   ⚠️ Image is flat, skipping stretching.")
        return img
    
    stretched = (img - p_min) / (p_max - p_min)
    
    print(f"   Stretched from [{p_min:.4f}, {p_max:.4f}] to [0, 1]")
    print(f"   New mean: {stretched.mean():.4f}")
    
    return stretched

def apply_gaussian_blur(img, kernel_size=(3, 3)):
    """Apply Gaussian blur for stroke connection."""
    print(f"🔧 Applying Gaussian blur (kernel={kernel_size})...")
    blurred = cv2.GaussianBlur(img, kernel_size, 0)
    return blurred

def multi_threshold(img, thresholds):
    """Generate binary masks at multiple thresholds."""
    print("🔧 Generating multi-level binary masks...")
    
    results = {}
    for name, thresh in thresholds.items():
        binary = (img >= thresh).astype(np.float32)
        coverage = binary.sum() / binary.size * 100
        results[name] = {
            "threshold": thresh,
            "binary": binary,
            "coverage": coverage,
        }
        print(f"   {name.capitalize()} (t={thresh:.2f}): {coverage:.2f}% pixels retained")
    
    return results

# =============================================================================
# 🎨 Visualization
# =============================================================================
def create_visualization(stretched, blurred, thresh_results, output_path):
    """Create 4-panel comparison visualization."""
    print("📊 Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle("Ink Prediction Post-Processing", fontsize=16, fontweight='bold')
    
    # Panel 1: Contrast Stretched (after blur)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(blurred, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f"Contrast Stretched + Blur\nMean: {blurred.mean():.3f}", fontsize=12)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Panel 2: Loose threshold
    loose = thresh_results["loose"]
    ax2 = axes[0, 1]
    ax2.imshow(loose["binary"], cmap='hot', vmin=0, vmax=1)
    ax2.set_title(f"Threshold = {loose['threshold']:.2f} (Loose)\n{loose['coverage']:.2f}% retained", fontsize=12)
    ax2.axis('off')
    
    # Panel 3: Balanced threshold
    balanced = thresh_results["balanced"]
    ax3 = axes[1, 0]
    ax3.imshow(balanced["binary"], cmap='hot', vmin=0, vmax=1)
    ax3.set_title(f"Threshold = {balanced['threshold']:.2f} (Balanced)\n{balanced['coverage']:.2f}% retained", fontsize=12)
    ax3.axis('off')
    
    # Panel 4: Strict threshold
    strict = thresh_results["strict"]
    ax4 = axes[1, 1]
    ax4.imshow(strict["binary"], cmap='hot', vmin=0, vmax=1)
    ax4.set_title(f"Threshold = {strict['threshold']:.2f} (Strict)\n{strict['coverage']:.2f}% retained", fontsize=12)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   ✅ Saved: {output_path}")

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius Ink Prediction Post-Processing")
    print("=" * 60)
    
    # 1. Load and normalize
    img = load_and_normalize(CONFIG["input_path"])
    
    # 2. Contrast stretching
    stretched = contrast_stretch(img)
    
    # 3. Gaussian blur
    blurred = apply_gaussian_blur(stretched, CONFIG["blur_kernel"])
    
    # 4. Multi-level thresholding
    thresh_results = multi_threshold(blurred, CONFIG["thresholds"])
    
    # 5. Visualization
    create_visualization(stretched, blurred, thresh_results, CONFIG["output_path"])
    
    # 6. Save individual binary masks
    print("\n💾 Saving individual binary masks...")
    for name, data in thresh_results.items():
        mask_path = f"inference_binary_{name}.png"
        cv2.imwrite(mask_path, (data["binary"] * 255).astype(np.uint8))
        print(f"   ✅ Saved: {mask_path}")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("📊 Post-Processing Complete!")
    print(f"   Input: {CONFIG['input_path']}")
    print(f"   Output: {CONFIG['output_path']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
