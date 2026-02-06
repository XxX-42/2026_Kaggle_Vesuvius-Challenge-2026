"""
Vesuvius Challenge 2026 - Raw CT Inference Verification
========================================================
Direct inference on original CT data (no RGT flattening).

Purpose: Verify if model works on raw data to rule out:
1. Domain shift from RGT flattening
2. Blank scroll region (no ink)

Input: data/native/train/1/surface_volume/33.tif to 48.tif (16 slices)
Output: raw_inference_pred.png, raw_inference_overlay.png
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import segmentation_models_pytorch as smp
from scipy.ndimage import gaussian_filter

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    # Input (raw CT data)
    "ct_dir": Path("data/native/train/1/surface_volume"),
    "z_start": 51,
    "z_end": 65,  # Exclusive, 51-64 = 14 slices (matches training)
    
    # Vesuvius normalization (same as training)
    "window_min": 18000,
    "window_max": 28000,
    
    # Model
    "model_path": "models/vesuvius_surgical_best.pth",
    "encoder": "resnet18",
    "in_channels": 14,  # Training used 14 slices
    
    # Inference
    "tile_size": 224,
    "stride": 112,
    "batch_size": 8,
    
    # Output
    "output_dir": Path("raw_inference_results"),
    
    # Post-processing
    "gaussian_sigma": 1.0,
    "binary_threshold": 0.10,
}

# =============================================================================
# 🔧 Data Loading (Training-Identical Normalization)
# =============================================================================
def load_raw_ct_volume(ct_dir, z_start, z_end, window_min, window_max):
    """
    Load raw CT slices with TRAINING-IDENTICAL normalization.
    
    This exactly replicates what the model saw during training.
    """
    print(f"📥 Loading raw CT data from {ct_dir}...")
    print(f"   Z range: [{z_start}, {z_end})")
    
    slices = []
    
    for z in tqdm(range(z_start, z_end), desc="   Loading"):
        path = ct_dir / f"{z}.tif"
        
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        
        # Load as 16-bit
        img = np.array(Image.open(path))
        slices.append(img)
    
    volume = np.stack(slices, axis=0).astype(np.float32)  # (D, H, W)
    
    print(f"   Raw volume shape: {volume.shape}")
    print(f"   Raw value range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # ============================================
    # CRITICAL: Training-identical normalization
    # ============================================
    print(f"\n🔧 Applying training normalization...")
    print(f"   Window: [{window_min}, {window_max}]")
    
    # Step 1: Clip to window
    volume = np.clip(volume, window_min, window_max)
    
    # Step 2: Normalize to [0, 1]
    volume = (volume - window_min) / (window_max - window_min)
    
    print(f"   Normalized range: [{volume.min():.4f}, {volume.max():.4f}]")
    
    return volume

# =============================================================================
# 🔧 Model Loading
# =============================================================================
def load_model(model_path, encoder, in_channels):
    """Load trained U-Net model."""
    print(f"\n📥 Loading model from {model_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded")
    
    return model, device

# =============================================================================
# 🔧 Sliding Window Inference
# =============================================================================
def sliding_window_inference(model, volume, tile_size, stride, device, batch_size=8):
    """
    Sliding window inference on 3D volume.
    
    Args:
        volume: (C, H, W) numpy array, normalized to [0, 1]
    """
    C, H, W = volume.shape
    
    print(f"\n🧮 Running inference on raw CT data...")
    print(f"   Volume: {volume.shape}")
    print(f"   Tile: {tile_size}x{tile_size}, Stride: {stride}")
    
    # Output arrays
    prediction = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    
    # Gaussian blending
    gaussian_weight = np.zeros((tile_size, tile_size), dtype=np.float32)
    center = tile_size // 2
    for i in range(tile_size):
        for j in range(tile_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            gaussian_weight[i, j] = np.exp(-dist**2 / (2 * (tile_size/4)**2))
    
    # Tile positions
    positions = []
    for y in range(0, max(1, H - tile_size + 1), stride):
        for x in range(0, max(1, W - tile_size + 1), stride):
            positions.append((y, x))
    
    # Edge tiles
    if H > tile_size and (H - tile_size) % stride != 0:
        for x in range(0, max(1, W - tile_size + 1), stride):
            positions.append((H - tile_size, x))
    if W > tile_size and (W - tile_size) % stride != 0:
        for y in range(0, max(1, H - tile_size + 1), stride):
            positions.append((y, W - tile_size))
    if H > tile_size and W > tile_size:
        if (H - tile_size) % stride != 0 and (W - tile_size) % stride != 0:
            positions.append((H - tile_size, W - tile_size))
    
    positions = list(set(positions))
    print(f"   Tiles: {len(positions)}")
    
    max_pred = 0.0
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(positions), batch_size), desc="   Inferring"):
            batch_pos = positions[batch_start:batch_start + batch_size]
            
            # Extract tiles
            tiles = []
            for y, x in batch_pos:
                tile = volume[:, y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
            
            tiles = np.stack(tiles, axis=0)  # (B, C, H, W)
            tiles_tensor = torch.from_numpy(tiles).float().to(device)
            
            # Forward pass
            outputs = model(tiles_tensor)
            outputs = torch.sigmoid(outputs)
            outputs_np = outputs.cpu().numpy()[:, 0, :, :]
            
            # Track max
            batch_max = outputs_np.max()
            if batch_max > max_pred:
                max_pred = batch_max
            
            # Accumulate
            for i, (y, x) in enumerate(batch_pos):
                prediction[y:y+tile_size, x:x+tile_size] += outputs_np[i] * gaussian_weight
                weight[y:y+tile_size, x:x+tile_size] += gaussian_weight
            
            if batch_start % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
    
    # Normalize
    weight = np.maximum(weight, 1e-8)
    prediction = prediction / weight
    
    print(f"\n" + "=" * 50)
    print(f"📊 RAW INFERENCE RESULTS:")
    print(f"   🎯 MAX PREDICTION VALUE: {max_pred:.4f}")
    print(f"   Mean: {prediction.mean():.4f}")
    print(f"   Range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    print("=" * 50)
    
    # Interpretation
    if max_pred > 0.4:
        print("\n✅ INK DETECTED! RGT flattening likely caused domain shift.")
    elif max_pred > 0.25:
        print("\n⚠️ Weak ink signal. Region may have sparse ink.")
    else:
        print("\n❌ NO SIGNIFICANT INK. This appears to be a blank scroll region.")
    
    return prediction, max_pred

# =============================================================================
# 🔧 Visualization
# =============================================================================
def save_results(prediction, volume, output_dir, sigma, threshold):
    """Save prediction and overlay images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_dir}/...")
    
    # Smooth prediction
    smoothed = gaussian_filter(prediction, sigma=sigma)
    
    # 1. Prediction map
    pred_img = (np.clip(smoothed, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "raw_inference_pred.png"), pred_img)
    print(f"   ✅ raw_inference_pred.png")
    
    # 2. Binary
    binary = (smoothed > threshold).astype(np.uint8) * 255
    cv2.imwrite(str(output_dir / "raw_inference_binary.png"), binary)
    print(f"   ✅ raw_inference_binary.png")
    
    # 3. Overlay on middle slice
    mid_idx = volume.shape[0] // 2
    background = (volume[mid_idx] * 255).astype(np.uint8)
    background_bgr = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    
    heatmap = cv2.applyColorMap(pred_img, cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(background_bgr, 0.5, heatmap, 0.5, 0)
    
    cv2.imwrite(str(output_dir / "raw_inference_overlay.png"), overlay)
    print(f"   ✅ raw_inference_overlay.png")
    
    # Stats
    ink_pixels = np.sum(binary > 0)
    total = binary.size
    print(f"\n📊 Binary stats (threshold={threshold}):")
    print(f"   Ink pixels: {ink_pixels:,} / {total:,} ({ink_pixels/total*100:.2f}%)")

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius - Raw CT Inference Verification")
    print("=" * 60)
    
    # 1. Load raw CT data with training normalization
    volume = load_raw_ct_volume(
        CONFIG["ct_dir"],
        CONFIG["z_start"],
        CONFIG["z_end"],
        CONFIG["window_min"],
        CONFIG["window_max"]
    )
    
    # Adjust to model channels if needed
    actual_slices = volume.shape[0]
    expected_channels = CONFIG["in_channels"]
    
    if actual_slices != expected_channels:
        print(f"\n⚠️ Slice count mismatch: got {actual_slices}, model expects {expected_channels}")
        if actual_slices > expected_channels:
            # Take center slices
            start = (actual_slices - expected_channels) // 2
            volume = volume[start:start + expected_channels]
            print(f"   Taking center {expected_channels} slices")
        else:
            # Reflect pad
            pad_total = expected_channels - actual_slices
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            volume = np.pad(volume, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='reflect')
            print(f"   Reflect padded to {expected_channels} slices")
    
    print(f"   Final volume: {volume.shape}")
    
    # 2. Load model
    model, device = load_model(
        CONFIG["model_path"],
        CONFIG["encoder"],
        CONFIG["in_channels"]
    )
    
    # 3. Run inference
    prediction, max_pred = sliding_window_inference(
        model, volume,
        CONFIG["tile_size"], CONFIG["stride"],
        device, CONFIG["batch_size"]
    )
    
    # 4. Save results
    save_results(
        prediction, volume,
        CONFIG["output_dir"],
        CONFIG["gaussian_sigma"],
        CONFIG["binary_threshold"]
    )
    
    print("\n" + "=" * 60)
    print("📊 Raw Inference Verification Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
