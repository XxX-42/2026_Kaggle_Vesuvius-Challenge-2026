"""
Vesuvius Challenge 2026 - RGT Scroll Flattening
================================================
Final Phase: Virtual unrolling based on scalar field iso-surfaces.

Algorithm:
1. Load scalar field φ and original CT volume
2. For each iso-value c (layer), find z where φ(x,y,z) = c
3. Sample CT intensity at those z-coordinates
4. Generate flattened 2D texture images

Output: flattened_layers/layer_XX.png
"""

import numpy as np
from scipy.ndimage import zoom, map_coordinates
from pathlib import Path
from tqdm import tqdm
import cv2
import gc
from PIL import Image

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    # Input
    "phi_path": "scalar_field.npz",
    "ct_dir": Path("data/native/train/1/surface_volume"),
    "z_range": (51, 65),  # Same Z range as orientation field
    
    # Output
    "output_dir": Path("flattened_layers"),
    
    # Windowing (Vesuvius-specific)
    "window_min": 18000,
    "window_max": 28000,
    
    # Layer extraction
    "num_layers": 20,  # Need more layers for true 3D volume
    "layer_margin": 0.05,  # Less margin to get more layers
}

# =============================================================================
# 🔧 Data Loading
# =============================================================================
def load_ct_volume(ct_dir, z_range):
    """Load CT volume from TIF slices."""
    print(f"📥 Loading CT volume from {ct_dir}...")
    
    # Find all TIF files
    tif_files = sorted(ct_dir.glob("*.tif"))
    
    if not tif_files:
        raise FileNotFoundError(f"No TIF files in {ct_dir}")
    
    # Filter by Z range
    z_start, z_end = z_range
    selected_files = []
    for f in tif_files:
        try:
            z_idx = int(f.stem)
            if z_start <= z_idx < z_end:
                selected_files.append((z_idx, f))
        except ValueError:
            continue
    
    selected_files.sort(key=lambda x: x[0])
    
    if not selected_files:
        raise ValueError(f"No files found in Z range {z_range}")
    
    print(f"   Loading {len(selected_files)} slices...")
    
    # Load first to get dimensions
    first_img = np.array(Image.open(selected_files[0][1]))
    H, W = first_img.shape
    D = len(selected_files)
    
    volume = np.zeros((D, H, W), dtype=np.float32)
    
    for i, (z_idx, path) in enumerate(tqdm(selected_files, desc="   Loading")):
        img = np.array(Image.open(path))
        volume[i] = img.astype(np.float32)
    
    print(f"   Volume shape: {volume.shape}")
    print(f"   Value range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    return volume

def apply_windowing(volume, win_min, win_max):
    """Apply contrast windowing."""
    volume = np.clip(volume, win_min, win_max)
    volume = (volume - win_min) / (win_max - win_min)
    return volume

# =============================================================================
# 🔧 Scalar Field Processing
# =============================================================================
def load_and_resize_phi(phi_path, target_shape):
    """Load phi and resize to match CT volume."""
    print(f"📥 Loading scalar field from {phi_path}...")
    
    data = np.load(phi_path)
    phi = data['phi']
    
    print(f"   Original phi shape: {phi.shape}")
    print(f"   Target shape: {target_shape}")
    
    # Calculate zoom factors
    factors = tuple(t / s for t, s in zip(target_shape, phi.shape))
    
    print(f"   Zoom factors: {factors}")
    
    if factors != (1.0, 1.0, 1.0):
        print("   Resizing phi...")
        phi_resized = zoom(phi, factors, order=1)
    else:
        phi_resized = phi
    
    # Ensure exact shape match
    phi_resized = phi_resized[:target_shape[0], :target_shape[1], :target_shape[2]]
    
    print(f"   Resized phi shape: {phi_resized.shape}")
    print(f"   Phi range: [{phi_resized.min():.4f}, {phi_resized.max():.4f}]")
    
    return phi_resized

# =============================================================================
# 🔧 Z-Flattening (Core Algorithm)
# =============================================================================
def find_z_for_layer(phi, target_value):
    """
    For each (x, y), find z where phi(x,y,z) ≈ target_value.
    
    Uses linear interpolation between phi values.
    
    Returns:
        z_map: (H, W) array of z-coordinates where phi = target_value
        mask: (H, W) boolean array, True where valid z was found
    """
    D, H, W = phi.shape
    
    z_map = np.full((H, W), np.nan, dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    
    # For each (y, x), search along z axis
    for z in range(D - 1):
        phi_curr = phi[z]      # (H, W)
        phi_next = phi[z + 1]  # (H, W)
        
        # Find where target crosses between z and z+1
        # Case 1: phi_curr <= target < phi_next (increasing)
        # Case 2: phi_curr >= target > phi_next (decreasing)
        
        cross_up = (phi_curr <= target_value) & (phi_next > target_value)
        cross_down = (phi_curr >= target_value) & (phi_next < target_value)
        crosses = cross_up | cross_down
        
        if not crosses.any():
            continue
        
        # Linear interpolation to find exact z
        # z_exact = z + (target - phi_curr) / (phi_next - phi_curr)
        denom = phi_next - phi_curr
        denom[denom == 0] = 1e-8  # Avoid division by zero
        
        t = (target_value - phi_curr) / denom
        z_exact = z + t
        
        # Only update where we cross and haven't already found a value
        update = crosses & ~mask
        z_map[update] = z_exact[update]
        mask[update] = True
    
    return z_map, mask

def sample_volume_at_z(volume, z_map, mask):
    """
    Sample the volume at fractional z coordinates using interpolation.
    
    Args:
        volume: (D, H, W) CT volume
        z_map: (H, W) z-coordinates to sample
        mask: (H, W) valid mask
    
    Returns:
        sampled: (H, W) sampled intensities
    """
    D, H, W = volume.shape
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Flatten for map_coordinates
    z_flat = z_map.ravel()
    y_flat = y_coords.ravel()
    x_flat = x_coords.ravel()
    
    # Stack coordinates: (3, N) array
    coords = np.array([z_flat, y_flat, x_flat])
    
    # Sample with linear interpolation
    sampled_flat = map_coordinates(volume, coords, order=1, mode='constant', cval=0)
    
    sampled = sampled_flat.reshape(H, W)
    
    # Zero out invalid regions
    sampled[~mask] = 0
    
    return sampled

# =============================================================================
# 🔧 Layer Extraction
# =============================================================================
def extract_layers(volume, phi, num_layers, margin=0.1):
    """
    Extract flattened layers from the volume.
    
    Args:
        volume: (D, H, W) windowed CT volume [0, 1]
        phi: (D, H, W) scalar field
        num_layers: number of layers to extract
        margin: fraction of phi range to skip at extremes
    
    Returns:
        layers: list of (H, W) arrays
        layer_values: corresponding phi values
    """
    phi_min, phi_max = phi.min(), phi.max()
    phi_range = phi_max - phi_min
    
    # Skip extreme values
    layer_min = phi_min + margin * phi_range
    layer_max = phi_max - margin * phi_range
    
    layer_values = np.linspace(layer_min, layer_max, num_layers)
    
    print(f"   Phi range: [{phi_min:.4f}, {phi_max:.4f}]")
    print(f"   Layer range: [{layer_min:.4f}, {layer_max:.4f}]")
    print(f"   Extracting {num_layers} layers...")
    
    layers = []
    
    for i, target in enumerate(tqdm(layer_values, desc="   Extracting")):
        z_map, mask = find_z_for_layer(phi, target)
        
        # Sample volume
        layer = sample_volume_at_z(volume, z_map, mask)
        
        layers.append(layer)
    
    return layers, layer_values

# =============================================================================
# 🔧 Output
# =============================================================================
def save_layers(layers, layer_values, output_dir):
    """Save layers as PNG images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving {len(layers)} layers to {output_dir}/...")
    
    for i, (layer, value) in enumerate(zip(layers, layer_values)):
        # Normalize to 0-255
        if layer.max() > 0:
            layer_norm = (layer * 255).clip(0, 255).astype(np.uint8)
        else:
            layer_norm = np.zeros_like(layer, dtype=np.uint8)
        
        # Save
        filename = output_dir / f"layer_{i+1:02d}.png"
        cv2.imwrite(str(filename), layer_norm)
    
    print(f"   ✅ Saved {len(layers)} layers")

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius RGT - Scroll Flattening")
    print("=" * 60)
    
    # 1. Load CT volume
    volume = load_ct_volume(CONFIG["ct_dir"], CONFIG["z_range"])
    
    # 2. Apply windowing
    print("\n🔧 Applying contrast windowing...")
    volume = apply_windowing(volume, CONFIG["window_min"], CONFIG["window_max"])
    print(f"   Windowed range: [{volume.min():.4f}, {volume.max():.4f}]")
    
    # 3. Load and resize phi
    print()
    phi = load_and_resize_phi(CONFIG["phi_path"], volume.shape)
    
    # 4. Extract layers
    print(f"\n🔪 Extracting flattened layers...")
    layers, layer_values = extract_layers(
        volume, phi, 
        CONFIG["num_layers"], 
        CONFIG["layer_margin"]
    )
    
    # 5. Save layers
    save_layers(layers, layer_values, CONFIG["output_dir"])
    
    # 6. Create summary image
    print("\n📊 Creating summary montage...")
    
    # Create grid of all layers
    n = len(layers)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    
    h, w = layers[0].shape
    scale = min(1.0, 2000 / w)  # Scale down if too large
    new_h, new_w = int(h * scale), int(w * scale)
    
    montage = np.zeros((rows * new_h, cols * new_w), dtype=np.uint8)
    
    for i, layer in enumerate(layers):
        row, col = i // cols, i % cols
        
        # Resize for montage
        layer_resized = cv2.resize(
            (layer * 255).clip(0, 255).astype(np.uint8),
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )
        
        y_start = row * new_h
        x_start = col * new_w
        montage[y_start:y_start+new_h, x_start:x_start+new_w] = layer_resized
    
    cv2.imwrite(str(CONFIG["output_dir"] / "montage.png"), montage)
    print(f"   ✅ Saved: {CONFIG['output_dir']}/montage.png")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("📊 Scroll Flattening Complete!")
    print(f"   Output: {CONFIG['output_dir']}/")
    print(f"   Layers: {len(layers)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
