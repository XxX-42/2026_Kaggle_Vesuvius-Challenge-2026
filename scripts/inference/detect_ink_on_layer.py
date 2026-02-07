"""
Vesuvius Challenge 2026 - Ink Detection on RGT Flattened Layer
===============================================================
Apply trained U-Net model to detect ink on RGT-flattened texture.

Input: flattened_layers/layer_XX.png (already windowed 0-255)
Output: final_ink_raw.png, final_ink_overlay.png, final_ink_binary.png
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
from scipy.ndimage import gaussian_filter

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    # Input/Output
    "input_path": "flattened_layers/layer_07.png",
    "model_path": "models/vesuvius_surgical_best.pth",
    "output_dir": Path("ink_detection_results"),
    
    # Model architecture (must match training)
    "encoder": "resnet18",
    "in_channels": 14,  # Model trained on 14 Z-slices, will replicate 2D layer
    
    # Inference parameters
    "tile_size": 224,
    "stride": 112,  # 50% overlap
    "batch_size": 16,
    
    # Post-processing
    "gaussian_sigma": 1.0,
    "binary_threshold": 0.15,
}

# =============================================================================
# 🔧 Model Loading
# =============================================================================
def load_model(model_path, encoder, in_channels):
    """Load trained U-Net model."""
    print(f"📥 Loading model from {model_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    
    return model, device

# =============================================================================
# 🔧 Sliding Window Inference
# =============================================================================
def sliding_window_inference(model, image, tile_size, stride, device, batch_size=16):
    """
    Perform sliding window inference on a 2D image.
    
    Args:
        model: PyTorch model
        image: (H, W) numpy array, normalized to [0, 1]
        tile_size: Size of each tile
        stride: Stride between tiles
        device: torch device
        batch_size: Number of tiles to process at once
    
    Returns:
        prediction: (H, W) numpy array of probabilities
    """
    H, W = image.shape
    
    # Initialize output arrays
    prediction = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    
    # Create Gaussian weight for blending
    gaussian_weight = np.zeros((tile_size, tile_size), dtype=np.float32)
    center = tile_size // 2
    for i in range(tile_size):
        for j in range(tile_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            gaussian_weight[i, j] = np.exp(-dist**2 / (2 * (tile_size/4)**2))
    
    # Generate tile positions
    positions = []
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            positions.append((y, x))
    
    # Add edge tiles
    if (H - tile_size) % stride != 0:
        for x in range(0, W - tile_size + 1, stride):
            positions.append((H - tile_size, x))
    if (W - tile_size) % stride != 0:
        for y in range(0, H - tile_size + 1, stride):
            positions.append((y, W - tile_size))
    if (H - tile_size) % stride != 0 and (W - tile_size) % stride != 0:
        positions.append((H - tile_size, W - tile_size))
    
    positions = list(set(positions))  # Remove duplicates
    
    print(f"   Processing {len(positions)} tiles...")
    
    # Process in batches
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(positions), batch_size), desc="   Inferring"):
            batch_positions = positions[batch_start:batch_start + batch_size]
            
            # Extract tiles
            tiles = []
            for y, x in batch_positions:
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
            
            # Stack and convert to tensor
            tiles = np.stack(tiles, axis=0)  # (B, H, W)
            # Replicate to 14 channels to match model trained on Z-stacks
            tiles = np.repeat(tiles[:, np.newaxis, :, :], 14, axis=1)  # (B, 14, H, W)
            tiles_tensor = torch.from_numpy(tiles).float().to(device)
            
            # Inference
            outputs = model(tiles_tensor)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.cpu().numpy()[:, 0, :, :]  # (B, H, W)
            
            # Accumulate predictions
            for i, (y, x) in enumerate(batch_positions):
                prediction[y:y+tile_size, x:x+tile_size] += outputs[i] * gaussian_weight
                weight[y:y+tile_size, x:x+tile_size] += gaussian_weight
            
            # Clear CUDA cache periodically
            if batch_start % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
    
    # Normalize by weight
    weight = np.maximum(weight, 1e-8)
    prediction = prediction / weight
    
    return prediction

# =============================================================================
# 🔧 Post-Processing
# =============================================================================
def post_process(prediction, sigma, threshold):
    """Apply Gaussian smoothing and thresholding."""
    
    # Gaussian blur
    smoothed = gaussian_filter(prediction, sigma=sigma)
    
    # Binary threshold
    binary = (smoothed > threshold).astype(np.uint8) * 255
    
    return smoothed, binary

def create_overlay(image, prediction, alpha=0.5):
    """Create red heatmap overlay on original image."""
    
    # Normalize prediction to 0-255
    pred_norm = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)
    
    # Create heatmap (red channel)
    heatmap = cv2.applyColorMap(pred_norm, cv2.COLORMAP_HOT)
    
    # Convert grayscale to BGR
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image
    
    # Blend
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
    
    return overlay

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius - Ink Detection on RGT Layer")
    print("=" * 60)
    
    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # 1. Load model
    model, device = load_model(
        CONFIG["model_path"],
        CONFIG["encoder"],
        CONFIG["in_channels"]
    )
    
    # 2. Load input image
    print(f"\n📥 Loading {CONFIG['input_path']}...")
    image = cv2.imread(str(CONFIG["input_path"]), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"Could not load {CONFIG['input_path']}")
    
    print(f"   Image shape: {image.shape}")
    print(f"   Value range: [{image.min()}, {image.max()}]")
    
    # 3. Normalize to [0, 1]
    image_norm = image.astype(np.float32) / 255.0
    print(f"   Normalized range: [{image_norm.min():.4f}, {image_norm.max():.4f}]")
    
    # 4. Run inference
    print(f"\n🧮 Running sliding window inference...")
    print(f"   Tile size: {CONFIG['tile_size']}, Stride: {CONFIG['stride']}")
    
    prediction = sliding_window_inference(
        model, image_norm,
        CONFIG["tile_size"], CONFIG["stride"],
        device, CONFIG["batch_size"]
    )
    
    print(f"   Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    
    # 5. Post-process
    print(f"\n🔧 Post-processing...")
    print(f"   Gaussian sigma: {CONFIG['gaussian_sigma']}")
    print(f"   Binary threshold: {CONFIG['binary_threshold']}")
    
    smoothed, binary = post_process(
        prediction,
        CONFIG["gaussian_sigma"],
        CONFIG["binary_threshold"]
    )
    
    # 6. Create overlay
    overlay = create_overlay(image, smoothed, alpha=0.5)
    
    # 7. Save results
    print(f"\n💾 Saving results to {CONFIG['output_dir']}/...")
    
    # Raw probability map
    raw_path = CONFIG["output_dir"] / "final_ink_raw.png"
    raw_img = (np.clip(smoothed, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(raw_path), raw_img)
    print(f"   ✅ {raw_path}")
    
    # Overlay
    overlay_path = CONFIG["output_dir"] / "final_ink_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"   ✅ {overlay_path}")
    
    # Binary
    binary_path = CONFIG["output_dir"] / "final_ink_binary.png"
    cv2.imwrite(str(binary_path), binary)
    print(f"   ✅ {binary_path}")
    
    # 8. Statistics
    ink_pixels = np.sum(binary > 0)
    total_pixels = binary.size
    ink_ratio = ink_pixels / total_pixels * 100
    
    print(f"\n📊 Results:")
    print(f"   Ink pixels: {ink_pixels:,} / {total_pixels:,} ({ink_ratio:.2f}%)")
    print(f"   Mean prediction: {prediction.mean():.4f}")
    print(f"   Max prediction: {prediction.max():.4f}")
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("📊 Ink Detection Complete!")
    print(f"   Output: {CONFIG['output_dir']}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
