"""
Signal Histogram Analysis for Vesuvius Challenge 2026
Analyzes pixel distribution of Ink vs Background to determine optimal windowing.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_signal():
    print("📊 Starting Signal Histogram Analysis...")
    
    # Path Configuration
    data_dir = Path("data/native/train/1")
    volume_dir = data_dir / "surface_volume"
    label_path = data_dir / "inklabels.png"
    
    # Find the signal layer (should be around Z=41 based on Z-scan)
    # First, list available files to find correct naming
    tif_files = sorted(list(volume_dir.glob("*.tif")))
    print(f"\n📦 Found {len(tif_files)} TIF files")
    
    if len(tif_files) == 0:
        print("❌ No TIF files found!")
        return
    
    # Print first few filenames to understand naming
    print(f"   Sample filenames: {[f.name for f in tif_files[:3]]}")
    
    # Try to load Z=41 (signal) and Z=20 (noise)
    signal_idx = min(41, len(tif_files) - 1)
    noise_idx = min(20, len(tif_files) - 1)
    
    signal_path = tif_files[signal_idx]
    noise_path = tif_files[noise_idx]
    
    print(f"\n🎯 Signal Layer: {signal_path.name} (Z={signal_idx})")
    print(f"🔇 Noise Layer: {noise_path.name} (Z={noise_idx})")
    
    # 1. Load images as uint16 (RAW values)
    print("\n📥 Loading images (keeping raw uint16)...")
    
    signal_img = np.array(Image.open(signal_path))
    noise_img = np.array(Image.open(noise_path))
    
    print(f"   Signal shape: {signal_img.shape}, dtype: {signal_img.dtype}")
    print(f"   Noise shape: {noise_img.shape}, dtype: {noise_img.dtype}")
    
    # 2. Load Labels
    print("\n🏷️ Loading Ink Labels...")
    labels_full = np.array(Image.open(label_path))
    
    # Handle RGB labels
    if len(labels_full.shape) == 3:
        labels_full = labels_full[:, :, 0]
    
    # Resize to match signal image
    labels_resized = np.array(
        Image.fromarray(labels_full).resize(
            (signal_img.shape[1], signal_img.shape[0]), 
            Image.NEAREST
        )
    )
    
    # Binarize
    ink_mask = labels_resized > 127
    bg_mask = labels_resized <= 127
    
    print(f"   Label shape (resized): {labels_resized.shape}")
    print(f"   Ink pixels: {ink_mask.sum():,}")
    print(f"   BG pixels: {bg_mask.sum():,}")
    
    # 3. Extract pixel values
    print("\n🔬 Extracting Pixel Distributions...")
    
    ink_pixels = signal_img[ink_mask].astype(np.float64)
    bg_pixels = signal_img[bg_mask].astype(np.float64)
    noise_all = noise_img.flatten().astype(np.float64)
    
    # 4. Compute Statistics
    print("\n📈 Computing Statistics...")
    
    def compute_stats(arr, name):
        stats = {
            'name': name,
            'mean': np.mean(arr),
            'std': np.std(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'p1': np.percentile(arr, 1),
            'p99': np.percentile(arr, 99),
            'median': np.median(arr)
        }
        return stats
    
    ink_stats = compute_stats(ink_pixels, "Ink (Z=41)")
    bg_stats = compute_stats(bg_pixels, "Background (Z=41)")
    noise_stats = compute_stats(noise_all, "Noise Layer (Z=20)")
    
    # Print stats table
    print("\n" + "="*70)
    print(f"{'Metric':<15} {'Ink (Z=41)':<20} {'Background (Z=41)':<20} {'Noise (Z=20)':<15}")
    print("="*70)
    print(f"{'Mean':<15} {ink_stats['mean']:<20.2f} {bg_stats['mean']:<20.2f} {noise_stats['mean']:<15.2f}")
    print(f"{'Std':<15} {ink_stats['std']:<20.2f} {bg_stats['std']:<20.2f} {noise_stats['std']:<15.2f}")
    print(f"{'Min':<15} {ink_stats['min']:<20.0f} {bg_stats['min']:<20.0f} {noise_stats['min']:<15.0f}")
    print(f"{'Max':<15} {ink_stats['max']:<20.0f} {bg_stats['max']:<20.0f} {noise_stats['max']:<15.0f}")
    print(f"{'P1':<15} {ink_stats['p1']:<20.2f} {bg_stats['p1']:<20.2f} {noise_stats['p1']:<15.2f}")
    print(f"{'P99':<15} {ink_stats['p99']:<20.2f} {bg_stats['p99']:<20.2f} {noise_stats['p99']:<15.2f}")
    print(f"{'Median':<15} {ink_stats['median']:<20.2f} {bg_stats['median']:<20.2f} {noise_stats['median']:<15.2f}")
    print("="*70)
    
    # 5. Compute Separation Metrics
    print("\n🎯 Signal Separation Analysis:")
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((ink_stats['std']**2 + bg_stats['std']**2) / 2)
    cohens_d = (ink_stats['mean'] - bg_stats['mean']) / pooled_std
    print(f"   Cohen's d (effect size): {cohens_d:.4f}")
    
    if abs(cohens_d) < 0.2:
        print("   ⚠️ Very weak separation (|d| < 0.2)")
    elif abs(cohens_d) < 0.5:
        print("   ⚠️ Small separation (0.2 < |d| < 0.5)")
    elif abs(cohens_d) < 0.8:
        print("   ✅ Medium separation (0.5 < |d| < 0.8)")
    else:
        print("   ✅ Large separation (|d| > 0.8)")
    
    # Suggest windowing
    # Use the range that captures most of the ink distribution
    suggested_min = max(0, ink_stats['p1'] - 0.1 * (ink_stats['p99'] - ink_stats['p1']))
    suggested_max = ink_stats['p99'] + 0.1 * (ink_stats['p99'] - ink_stats['p1'])
    
    print(f"\n💡 Suggested Intensity Window:")
    print(f"   min_val = {suggested_min:.0f}")
    print(f"   max_val = {suggested_max:.0f}")
    print(f"   (Normalized: [{suggested_min/65535:.4f}, {suggested_max/65535:.4f}])")
    
    # 6. Generate Histogram Visualization
    print("\n🎨 Generating Histogram Visualization...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define bin range based on actual data
    global_min = min(noise_all.min(), ink_pixels.min(), bg_pixels.min())
    global_max = max(noise_all.max(), ink_pixels.max(), bg_pixels.max())
    bins = np.linspace(global_min, global_max, 200)
    
    # Subplot 1: Noise Layer (Z=20)
    axes[0].hist(noise_all, bins=bins, alpha=0.7, color='gray', density=True, label=f'Z=20 (All pixels)')
    axes[0].set_title(f'Noise Layer (Z={noise_idx}) - Full Histogram', fontsize=12)
    axes[0].set_xlabel('Pixel Intensity (uint16)', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Signal Layer with Ink/BG separation
    axes[1].hist(bg_pixels, bins=bins, alpha=0.6, color='blue', density=True, label=f'Background (N={len(bg_pixels):,})')
    axes[1].hist(ink_pixels, bins=bins, alpha=0.6, color='red', density=True, label=f'Ink (N={len(ink_pixels):,})')
    
    # Mark suggested window
    axes[1].axvline(x=suggested_min, color='green', linestyle='--', linewidth=2, label=f'Window Min: {suggested_min:.0f}')
    axes[1].axvline(x=suggested_max, color='green', linestyle='--', linewidth=2, label=f'Window Max: {suggested_max:.0f}')
    
    # Mark means
    axes[1].axvline(x=ink_stats['mean'], color='darkred', linestyle='-', linewidth=1.5, alpha=0.8)
    axes[1].axvline(x=bg_stats['mean'], color='darkblue', linestyle='-', linewidth=1.5, alpha=0.8)
    
    axes[1].set_title(f'Signal Layer (Z={signal_idx}) - Ink vs Background Distribution', fontsize=12)
    axes[1].set_xlabel('Pixel Intensity (uint16)', fontsize=10)
    axes[1].set_ylabel('Density', fontsize=10)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Add text annotation
    textstr = f"Ink Mean: {ink_stats['mean']:.0f}\nBG Mean: {bg_stats['mean']:.0f}\nCohen's d: {cohens_d:.3f}"
    axes[1].text(0.02, 0.98, textstr, transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("signal_distribution.png", dpi=150)
    plt.close(fig)
    
    print("   ✅ Saved: signal_distribution.png")
    
    # 7. Final Report
    print("\n" + "="*50)
    print("📊 SIGNAL HISTOGRAM ANALYSIS COMPLETE")
    print("="*50)
    print(f"   Ink Mean: {ink_stats['mean']:.2f}")
    print(f"   BG Mean: {bg_stats['mean']:.2f}")
    print(f"   Difference: {ink_stats['mean'] - bg_stats['mean']:.2f}")
    print(f"   Cohen's d: {cohens_d:.4f}")
    print("="*50)

if __name__ == "__main__":
    analyze_signal()
