"""
Vesuvius Challenge 2026 - Final Ink Structure Cleanup
======================================================
Remove fine noise from binary predictions using morphological operations.

Pipeline:
1. Load binary prediction (or regenerate from raw)
2. Morphological opening (erosion + dilation)
3. Connected component filtering (remove small blobs)
4. Save cleaned structure
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    "input_pred": "inference_pred.png",  # Raw prediction
    "output_path": "final_ink_structure.png",
    "comparison_path": "cleanup_comparison.png",
    
    # Threshold for binarization
    "threshold": 0.20,
    
    # Morphological opening kernel
    "opening_kernel": (5, 5),
    
    # Connected component minimum area (pixels)
    "min_area": 100,
}

# =============================================================================
# 🔧 Processing Functions
# =============================================================================
def load_and_binarize(path, threshold):
    """Load prediction and apply threshold."""
    print(f"📥 Loading {path}...")
    
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    
    # Normalize to [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # Contrast stretch
    p_min, p_max = img_float.min(), img_float.max()
    if p_max - p_min > 1e-6:
        img_float = (img_float - p_min) / (p_max - p_min)
    
    # Binarize
    binary = (img_float >= threshold).astype(np.uint8) * 255
    
    original_pixels = (binary > 0).sum()
    print(f"   Original binary: {original_pixels} pixels ({original_pixels / binary.size * 100:.2f}%)")
    
    return binary

def morphological_opening(binary, kernel_size):
    """Apply morphological opening to remove small noise."""
    print(f"🔧 Applying morphological opening (kernel={kernel_size})...")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    remaining_pixels = (opened > 0).sum()
    print(f"   After opening: {remaining_pixels} pixels ({remaining_pixels / opened.size * 100:.2f}%)")
    
    return opened

def remove_small_components(binary, min_area):
    """Remove connected components smaller than min_area."""
    print(f"🔧 Removing components < {min_area} pixels...")
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create output mask
    cleaned = np.zeros_like(binary)
    
    kept_count = 0
    removed_count = 0
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
            kept_count += 1
        else:
            removed_count += 1
    
    remaining_pixels = (cleaned > 0).sum()
    print(f"   Components: {kept_count} kept, {removed_count} removed")
    print(f"   After filtering: {remaining_pixels} pixels ({remaining_pixels / cleaned.size * 100:.2f}%)")
    
    return cleaned

def create_comparison(before, after, output_path):
    """Create side-by-side comparison visualization."""
    print("📊 Creating comparison visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Ink Structure Cleanup", fontsize=16, fontweight='bold')
    
    # Before
    before_pixels = (before > 0).sum()
    axes[0].imshow(before, cmap='hot')
    axes[0].set_title(f"Before Cleanup\n{before_pixels:,} pixels ({before_pixels / before.size * 100:.2f}%)", fontsize=14)
    axes[0].axis('off')
    
    # After
    after_pixels = (after > 0).sum()
    axes[1].imshow(after, cmap='hot')
    axes[1].set_title(f"After Cleanup\n{after_pixels:,} pixels ({after_pixels / after.size * 100:.2f}%)", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   ✅ Saved: {output_path}")

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius Ink Structure Cleanup")
    print("=" * 60)
    
    # 1. Load and binarize
    binary = load_and_binarize(CONFIG["input_pred"], CONFIG["threshold"])
    original = binary.copy()
    
    # 2. Morphological opening
    opened = morphological_opening(binary, CONFIG["opening_kernel"])
    
    # 3. Remove small connected components
    cleaned = remove_small_components(opened, CONFIG["min_area"])
    
    # 4. Save final result
    print(f"\n💾 Saving final structure...")
    cv2.imwrite(CONFIG["output_path"], cleaned)
    print(f"   ✅ Saved: {CONFIG['output_path']}")
    
    # 5. Create comparison
    create_comparison(original, cleaned, CONFIG["comparison_path"])
    
    # 6. Summary
    original_pct = (original > 0).sum() / original.size * 100
    final_pct = (cleaned > 0).sum() / cleaned.size * 100
    reduction = original_pct - final_pct
    
    print("\n" + "=" * 60)
    print("📊 Cleanup Complete!")
    print(f"   Original: {original_pct:.2f}%")
    print(f"   Final: {final_pct:.2f}%")
    print(f"   Noise removed: {reduction:.2f}% of image")
    print("=" * 60)

if __name__ == "__main__":
    main()
