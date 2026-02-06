"""
Vesuvius Challenge 2026 - Full Scroll Inference
================================================
Sliding window inference on full scroll to generate ink prediction map.

Critical Parameters (Must Match Training):
- Z-slices: range(33, 49) - 16 layers centered on Z=41
- Normalization: clip [18000, 28000] -> [0, 1]
- Window: 224, Stride: 112 (50% overlap with average blending)
"""

import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# =============================================================================
# ⚙️ Configuration (MUST MATCH TRAINING)
# =============================================================================
CONFIG = {
    "data_root": Path("data/native/train/1"),
    "model_path": "vesuvius_surgical_best.pth",
    
    # Z-axis alignment (actual training used Z=51-64, 14 slices)
    "z_slices": list(range(51, 65)),  # 14 slices
    
    # Normalization window
    "window_min": 18000.0,
    "window_max": 28000.0,
    
    # Sliding window
    "tile_size": 224,
    "stride": 112,  # 50% overlap
    
    # Model architecture (MUST MATCH TRAINING)
    "encoder": "resnet18",
    "in_channels": 14,  # len(range(51, 65))
    
    # Output
    "output_pred": "inference_pred.png",
    "output_overlay": "inference_overlay.png",
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# =============================================================================
# 📦 Data Loading
# =============================================================================
def load_volume(data_root, z_slices):
    """Load Z-stack volume and apply windowing normalization."""
    print("📦 Loading volume...")
    
    tif_dir = data_root / "surface_volume"
    
    # Build file dictionary by Z-index
    tif_dict = {}
    for f in tif_dir.glob("*.tif"):
        try:
            z_idx = int(f.stem)
            tif_dict[z_idx] = f
        except ValueError:
            continue
    
    # Filter to available z_slices
    available = [z for z in z_slices if z in tif_dict]
    if len(available) < len(z_slices):
        print(f"⚠️ Only {len(available)}/{len(z_slices)} Z-slices available")
    
    # Load first slice to get dimensions
    with Image.open(tif_dict[available[0]]) as img:
        w, h = img.size
    print(f"   Volume dimensions: {w} x {h} x {len(available)}")
    
    # Load all slices
    volume = np.zeros((h, w, len(available)), dtype=np.float32)
    for i, z in enumerate(tqdm(available, desc="   Loading Z-slices")):
        with Image.open(tif_dict[z]) as img:
            volume[:, :, i] = np.array(img).astype(np.float32)
    
    # Apply windowing normalization
    print("🔧 Applying windowing normalization...")
    volume = np.clip(volume, CONFIG["window_min"], CONFIG["window_max"])
    volume = (volume - CONFIG["window_min"]) / (CONFIG["window_max"] - CONFIG["window_min"])
    
    print(f"   Normalized range: [{volume.min():.4f}, {volume.max():.4f}]")
    print(f"   Mean: {volume.mean():.4f}")
    
    return volume, available

# =============================================================================
# 🧠 Sliding Window Inference
# =============================================================================
def sliding_window_inference(model, volume, tile_size=224, stride=112):
    """
    Perform sliding window inference with average blending.
    
    Args:
        model: Trained segmentation model
        volume: (H, W, C) normalized volume
        tile_size: Window size
        stride: Step size (overlap = tile_size - stride)
    
    Returns:
        pred_map: (H, W) probability map
    """
    model.eval()
    h, w, c = volume.shape
    
    # Initialize accumulator and counter
    pred_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    # Calculate grid
    y_steps = list(range(0, h - tile_size + 1, stride))
    x_steps = list(range(0, w - tile_size + 1, stride))
    
    # Add edge tiles if needed
    if y_steps[-1] + tile_size < h:
        y_steps.append(h - tile_size)
    if x_steps[-1] + tile_size < w:
        x_steps.append(w - tile_size)
    
    total_tiles = len(y_steps) * len(x_steps)
    print(f"🔍 Sliding window: {len(y_steps)} x {len(x_steps)} = {total_tiles} tiles")
    
    with torch.no_grad():
        bar = tqdm(total=total_tiles, desc="   Inference")
        
        for y in y_steps:
            for x in x_steps:
                # Extract tile
                tile = volume[y:y+tile_size, x:x+tile_size, :]  # (H, W, C)
                
                # Convert to tensor (N, C, H, W)
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float()
                tile_tensor = tile_tensor.to(CONFIG["device"])
                
                # Forward pass
                with torch.amp.autocast('cuda'):
                    output = model(tile_tensor)
                    prob = torch.sigmoid(output).cpu().numpy()[0, 0]  # (H, W)
                
                # Accumulate
                pred_sum[y:y+tile_size, x:x+tile_size] += prob
                count_map[y:y+tile_size, x:x+tile_size] += 1
                
                bar.update(1)
                
                # VRAM cleanup every 100 tiles
                if bar.n % 100 == 0:
                    torch.cuda.empty_cache()
        
        bar.close()
    
    # Average blending
    pred_map = np.divide(pred_sum, count_map, where=count_map > 0)
    
    print(f"   Prediction range: [{pred_map.min():.4f}, {pred_map.max():.4f}]")
    print(f"   Mean confidence: {pred_map.mean():.4f}")
    
    return pred_map

# =============================================================================
# 🎨 Visualization
# =============================================================================
def create_overlay(pred_map, texture, alpha=0.5):
    """Create red heatmap overlay on grayscale texture."""
    # Normalize texture to [0, 1]
    texture_norm = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    
    # Create RGB image
    h, w = texture_norm.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    
    # Grayscale background
    overlay[:, :, 0] = texture_norm
    overlay[:, :, 1] = texture_norm
    overlay[:, :, 2] = texture_norm
    
    # Red heatmap for prediction
    red_channel = pred_map * alpha
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + red_channel, 0, 1)
    
    return (overlay * 255).astype(np.uint8)

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius Challenge 2026 - Full Scroll Inference")
    print("=" * 60)
    
    # 1. Load Model
    print("\n📂 Loading model...")
    model = smp.Unet(
        encoder_name=CONFIG["encoder"],
        encoder_weights=None,  # Load from checkpoint
        in_channels=CONFIG["in_channels"],
        classes=1,
    )
    
    if os.path.exists(CONFIG["model_path"]):
        model.load_state_dict(torch.load(CONFIG["model_path"], weights_only=True))
        print(f"   ✅ Loaded: {CONFIG['model_path']}")
    else:
        print(f"   ❌ Model not found: {CONFIG['model_path']}")
        return
    
    model = model.to(CONFIG["device"])
    model.eval()
    
    # 2. Load Volume
    volume, z_slices = load_volume(CONFIG["data_root"], CONFIG["z_slices"])
    
    # 3. Sliding Window Inference
    print("\n🧠 Running inference...")
    pred_map = sliding_window_inference(
        model, volume,
        tile_size=CONFIG["tile_size"],
        stride=CONFIG["stride"]
    )
    
    # 4. Save Prediction
    print("\n💾 Saving outputs...")
    
    # Pure prediction map
    pred_uint8 = (pred_map * 255).astype(np.uint8)
    Image.fromarray(pred_uint8).save(CONFIG["output_pred"])
    print(f"   ✅ Saved: {CONFIG['output_pred']}")
    
    # Overlay on Z=41 texture
    z41_idx = z_slices.index(41) if 41 in z_slices else len(z_slices) // 2
    texture = volume[:, :, z41_idx]
    overlay = create_overlay(pred_map, texture, alpha=0.6)
    Image.fromarray(overlay).save(CONFIG["output_overlay"])
    print(f"   ✅ Saved: {CONFIG['output_overlay']}")
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("📊 Inference Complete!")
    print(f"   Prediction Mean: {pred_map.mean():.4f}")
    print(f"   Prediction Max: {pred_map.max():.4f}")
    print(f"   Non-zero pixels: {(pred_map > 0.1).sum()} ({(pred_map > 0.1).sum() / pred_map.size * 100:.2f}%)")
    print("=" * 60)
    
    # Cleanup
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
