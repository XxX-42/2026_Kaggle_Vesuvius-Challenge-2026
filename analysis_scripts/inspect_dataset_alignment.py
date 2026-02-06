"""
Dataset Alignment Inspector for Vesuvius Challenge 2026
Verifies that surface_volume slices and ink labels are correctly aligned.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

def inspect_dataset():
    print("🔍 Starting Dataset Alignment Inspection...")
    
    # Path Configuration
    data_dir = Path("data/native/train/1")
    volume_dir = data_dir / "surface_volume"
    label_path = data_dir / "inklabels.png"
    
    # Check if paths exist
    if not volume_dir.exists():
        print(f"❌ Error: Volume directory not found: {volume_dir}")
        return
    if not label_path.exists():
        print(f"❌ Error: Label file not found: {label_path}")
        return
    
    # 1. Load 3D Volume (Middle Z-slices)
    print("\n📦 Loading Surface Volume...")
    tif_files = sorted(glob(str(volume_dir / "*.tif")))
    print(f"   Total TIF files found: {len(tif_files)}")
    
    if len(tif_files) == 0:
        print("❌ Error: No TIF files found!")
        return
    
    # Select middle Z-slices (e.g., 25-35)
    mid_start = max(0, len(tif_files) // 2 - 5)
    mid_end = min(len(tif_files), len(tif_files) // 2 + 5)
    selected_files = tif_files[mid_start:mid_end]
    print(f"   Using Z-slices: {mid_start} to {mid_end-1} (Total: {len(selected_files)})")
    
    # Load and stack slices (DOWNSAMPLED to 1/4 resolution for memory safety)
    DOWNSAMPLE = 4  # Reduce by factor of 4
    slices = []
    for f in selected_files:
        img = Image.open(f)
        # Downsample to prevent memory crash
        new_size = (img.width // DOWNSAMPLE, img.height // DOWNSAMPLE)
        img_small = img.resize(new_size, Image.LANCZOS)
        slices.append(np.array(img_small))
        img.close()
    
    volume = np.stack(slices, axis=0)  # (Z, H, W)
    
    # Volume Statistics
    print("\n📊 Volume Statistics:")
    print(f"   Shape: {volume.shape}")
    print(f"   Dtype: {volume.dtype}")
    print(f"   Min: {volume.min()}")
    print(f"   Max: {volume.max()}")
    print(f"   Mean: {volume.mean():.2f}")
    
    if volume.max() > 255:
        print("   ⚠️ NOTE: Data is 16-bit. Training code MUST divide by 65535!")
    else:
        print("   ℹ️ Data is 8-bit. Divide by 255 for normalization.")
    
    # Compute Mean Projection
    mean_proj = volume.mean(axis=0)  # (H, W)
    print(f"\n   Mean Projection Shape: {mean_proj.shape}")
    
    # 2. Load Labels (RESIZED to match volume dimensions)
    print("\n🏷️ Loading Ink Labels...")
    labels_full = Image.open(label_path)
    # Resize to exactly match mean_proj dimensions (width, height for PIL)
    target_size = (mean_proj.shape[1], mean_proj.shape[0])  # PIL uses (W, H)
    labels_small = labels_full.resize(target_size, Image.NEAREST)
    labels = np.array(labels_small)
    labels_full.close()
    
    print(f"   Shape: {labels.shape}")
    print(f"   Dtype: {labels.dtype}")
    print(f"   Unique values: {np.unique(labels)}")
    print(f"   Min: {labels.min()}")
    print(f"   Max: {labels.max()}")
    print(f"   Non-zero pixels: {(labels > 0).sum()} / {labels.size} ({100*(labels > 0).sum()/labels.size:.2f}%)")
    
    if labels.max() == 0:
        print("   ❌ CRITICAL: Label is ALL ZEROS! No ink annotation found!")
    elif labels.min() == labels.max():
        print(f"   ❌ CRITICAL: Label is UNIFORM (value={labels.max()})! Invalid mask!")
    else:
        print("   ✅ Label has variation (good).")
    
    # 3. Check alignment (dimensions)
    print("\n🔗 Checking Alignment...")
    if mean_proj.shape[:2] == labels.shape[:2]:
        print(f"   ✅ Dimensions match: Volume {mean_proj.shape[:2]} == Labels {labels.shape[:2]}")
    else:
        print(f"   ❌ DIMENSION MISMATCH: Volume {mean_proj.shape[:2]} != Labels {labels.shape[:2]}")
    
    # 4. Generate Visualization
    print("\n🎨 Generating Overlay Visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Normalize mean projection for display
    mean_proj_norm = (mean_proj - mean_proj.min()) / (mean_proj.max() - mean_proj.min() + 1e-7)
    
    # Enhance contrast (percentile stretch)
    p2, p98 = np.percentile(mean_proj_norm, (2, 98))
    mean_proj_enhanced = np.clip((mean_proj_norm - p2) / (p98 - p2 + 1e-7), 0, 1)
    
    # Left: Surface Texture
    axes[0].imshow(mean_proj_enhanced, cmap='gray')
    axes[0].set_title('Surface Texture (Mean Z-Projection)')
    axes[0].axis('off')
    
    # Middle: Ink Labels
    label_disp = labels.astype(float)
    if labels.max() > 0:
        label_disp = label_disp / labels.max()
    axes[1].imshow(label_disp, cmap='gray')
    axes[1].set_title('Ink Labels (Ground Truth)')
    axes[1].axis('off')
    
    # Right: Red Overlay
    # Create RGB image from grayscale
    rgb = np.stack([mean_proj_enhanced, mean_proj_enhanced, mean_proj_enhanced], axis=-1)
    
    # Create red overlay where label > 0
    if len(labels.shape) == 3:
        label_binary = labels[:, :, 0] > 0  # Handle RGB labels
    else:
        label_binary = labels > 0
    
    # Apply 50% transparent red overlay
    overlay = rgb.copy()
    overlay[label_binary, 0] = 0.5 * overlay[label_binary, 0] + 0.5 * 1.0  # Red channel
    overlay[label_binary, 1] = 0.5 * overlay[label_binary, 1]  # Green channel
    overlay[label_binary, 2] = 0.5 * overlay[label_binary, 2]  # Blue channel
    
    axes[2].imshow(overlay)
    axes[2].set_title('Red Overlay (Ink on Surface)')
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = "dataset_check.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"   ✅ Saved visualization to: {output_path}")
    print("\n🏁 Inspection Complete!")
    print("   → Open 'dataset_check.png' to visually verify ink alignment.")

if __name__ == "__main__":
    inspect_dataset()
