"""
Z-Scan Forensics for Vesuvius Challenge 2026
Locates the physical Z-layer where ink signal is strongest.
Detects Z-axis misalignment between labels and volume data.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from tqdm import tqdm

def run_z_scan():
    print("🔬 Starting Z-Scan Forensics...")
    
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
    
    # 1. Load all TIF files (sorted by numeric index)
    print("\n📦 Loading Surface Volume...")
    tif_files = sorted(glob(str(volume_dir / "*.tif")), 
                       key=lambda x: int(Path(x).stem.split('_')[-1]) if '_' in Path(x).stem else int(Path(x).stem))
    print(f"   Total TIF files found: {len(tif_files)}")
    
    if len(tif_files) == 0:
        print("❌ Error: No TIF files found!")
        return
    
    # 2. Load Labels
    print("\n🏷️ Loading Ink Labels...")
    labels_full = Image.open(label_path)
    labels = np.array(labels_full).astype(np.float32)
    labels_full.close()
    
    # Normalize labels to 0-1
    if labels.max() > 1:
        labels = labels / 255.0
    
    # Handle RGB labels
    if len(labels.shape) == 3:
        labels = labels[:, :, 0]
    
    # Binarize
    labels = (labels > 0.5).astype(np.float32)
    
    print(f"   Label shape: {labels.shape}")
    print(f"   Label sum (ink pixels): {labels.sum():.0f}")
    print(f"   Ink coverage: {100 * labels.sum() / labels.size:.2f}%")
    
    if labels.sum() == 0:
        print("   ❌ CRITICAL: Labels are ALL ZEROS! Cannot perform Z-scan.")
        return
    if labels.sum() == labels.size:
        print("   ❌ CRITICAL: Labels are ALL ONES! Invalid mask.")
        return
    
    print("   ✅ Labels have valid ink regions.")
    
    # 3. Z-Scan: Compute signal for each layer
    print("\n🔍 Running Z-Scan (this may take a while)...")
    
    z_signals = []
    DOWNSAMPLE = 4  # Speed up by downsampling
    
    # Get target size from first image
    first_img = Image.open(tif_files[0])
    target_size = (first_img.width // DOWNSAMPLE, first_img.height // DOWNSAMPLE)
    first_img.close()
    
    # Resize labels to match
    labels_small = np.array(Image.fromarray((labels * 255).astype(np.uint8)).resize(target_size, Image.NEAREST)).astype(np.float32) / 255.0
    labels_small = (labels_small > 0.5).astype(np.float32)
    
    ink_mask = labels_small > 0.5
    bg_mask = labels_small < 0.5
    
    print(f"   Downsampled to: {target_size}")
    print(f"   Ink pixels: {ink_mask.sum()}, BG pixels: {bg_mask.sum()}")
    
    for z_idx, tif_path in enumerate(tqdm(tif_files, desc="   Z-Scan")):
        # Load and downsample
        img = Image.open(tif_path)
        img_small = img.resize(target_size, Image.LANCZOS)
        data = np.array(img_small).astype(np.float32) / 65535.0  # Normalize uint16
        img.close()
        
        # Compute signal: (Mean Ink Brightness) - (Mean BG Brightness)
        if ink_mask.sum() > 0 and bg_mask.sum() > 0:
            mean_ink = data[ink_mask].mean()
            mean_bg = data[bg_mask].mean()
            signal = mean_ink - mean_bg
        else:
            signal = 0
        
        z_signals.append({
            'z': z_idx,
            'signal': signal,
            'mean_ink': mean_ink,
            'mean_bg': mean_bg,
            'path': tif_path
        })
    
    # Convert to arrays
    z_indices = [s['z'] for s in z_signals]
    signals = [s['signal'] for s in z_signals]
    
    # Find peak
    peak_idx = np.argmax(np.abs(signals))
    peak_z = z_indices[peak_idx]
    peak_signal = signals[peak_idx]
    
    print(f"\n📊 Z-Scan Results:")
    print(f"   Peak signal at Z={peak_z}: {peak_signal:.6f}")
    print(f"   Signal range: [{min(signals):.6f}, {max(signals):.6f}]")
    
    if abs(peak_signal) < 0.01:
        print("   ⚠️ WARNING: Peak signal is very weak! Possible misalignment.")
    else:
        print("   ✅ Clear signal detected!")
    
    # 4. Plot Z-Signal Profile
    print("\n📈 Generating Z-Signal Profile...")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(z_indices, signals, 'b-', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=peak_z, color='red', linestyle='--', alpha=0.7, label=f'Peak @ Z={peak_z}')
    
    ax.set_xlabel('Z-Index (Depth)', fontsize=12)
    ax.set_ylabel('Signal: Mean(Ink) - Mean(BG)', fontsize=12)
    ax.set_title('Z-Scan Signal Profile: Where is the Ink?', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight current training range
    train_z_start, train_z_end = 16, 31
    ax.axvspan(train_z_start, train_z_end, alpha=0.2, color='green', label=f'Current Training Range: Z={train_z_start}-{train_z_end}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("z_signal_profile.png", dpi=150)
    plt.close(fig)
    print("   ✅ Saved: z_signal_profile.png")
    
    # 5. Generate Layer Gallery
    print("\n🖼️ Generating Layer Gallery...")
    
    # Select layers: peak +/- 2, and the worst layer
    gallery_indices = [
        max(0, peak_z - 2),
        max(0, peak_z - 1),
        peak_z,
        min(len(tif_files) - 1, peak_z + 1),
        min(len(tif_files) - 1, peak_z + 2),
        np.argmin(np.abs(signals))  # Layer with weakest signal
    ]
    gallery_indices = sorted(set(gallery_indices))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, z_idx in enumerate(gallery_indices[:6]):
        if i >= len(axes):
            break
            
        # Load layer
        img = Image.open(tif_files[z_idx])
        img_small = img.resize(target_size, Image.LANCZOS)
        data = np.array(img_small).astype(np.float32) / 65535.0
        img.close()
        
        # Create RGB overlay
        p2, p98 = np.percentile(data, (2, 98))
        data_norm = np.clip((data - p2) / (p98 - p2 + 1e-7), 0, 1)
        rgb = np.stack([data_norm, data_norm, data_norm], axis=-1)
        
        # Red overlay for labels (30% opacity)
        rgb[ink_mask, 0] = 0.7 * rgb[ink_mask, 0] + 0.3 * 1.0
        rgb[ink_mask, 1] = 0.7 * rgb[ink_mask, 1]
        rgb[ink_mask, 2] = 0.7 * rgb[ink_mask, 2]
        
        axes[i].imshow(rgb)
        signal_val = z_signals[z_idx]['signal']
        title = f"Z={z_idx} | Signal={signal_val:.4f}"
        if z_idx == peak_z:
            title += " [PEAK]"
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(len(gallery_indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Layer Gallery: Red = Ink Labels Overlay', fontsize=14)
    plt.tight_layout()
    plt.savefig("layer_gallery.png", dpi=100)
    plt.close(fig)
    print("   ✅ Saved: layer_gallery.png")
    
    # 6. Final Report
    print("\n" + "="*50)
    print("🔬 Z-SCAN FORENSICS REPORT")
    print("="*50)
    print(f"   Peak Signal Layer: Z={peak_z}")
    print(f"   Peak Signal Value: {peak_signal:.6f}")
    print(f"   Current Training Range: Z=16-31")
    
    if train_z_start <= peak_z <= train_z_end:
        print("   ✅ Peak is WITHIN training range! Alignment OK.")
    else:
        print(f"   ⚠️ Peak is OUTSIDE training range! Consider using Z={max(0, peak_z-8)}-{min(len(tif_files)-1, peak_z+8)}")
    
    print("="*50)
    print("\nOpen 'z_signal_profile.png' and 'layer_gallery.png' to visually confirm.")

if __name__ == "__main__":
    run_z_scan()
