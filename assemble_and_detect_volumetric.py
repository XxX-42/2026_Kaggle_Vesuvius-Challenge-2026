"""
Vesuvius Challenge 2026 - Volumetric Ink Detection
===================================================
Stack RGT-flattened layers into 3D volume for proper Z-stack inference.

Problem: Single-layer detection failed (max pred 0.13) due to missing Z-axis features.
Solution: Stack 10 flattened layers + reflect padding to 16 channels.

Input: flattened_layers/layer_01.png to layer_10.png
Output: volumetric_prediction.png, volumetric_overlay.png
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
from scipy.ndimage import gaussian_filter

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    # Input
    "layer_dir": Path("flattened_layers"),
    "num_layers": 10,
    "model_path": "models/vesuvius_surgical_best.pth",
    
    # Model architecture (must match training)
    "encoder": "resnet18",
    "in_channels": 14,  # Model trained on 14 Z-slices
    
    # Inference parameters
    "tile_size": 224,
    "stride": 112,
    "batch_size": 8,
    
    # Output
    "output_dir": Path("volumetric_detection_results"),
    
    # Post-processing
    "gaussian_sigma": 1.0,
    "binary_threshold": 0.10,  # Lower threshold
}

# =============================================================================
# 🔧 Data Loading
# =============================================================================
def load_layer_stack(layer_dir, num_layers):
    """
    Load flattened layers and stack into volume.
    
    Returns:
        volume: (num_layers, H, W) normalized to [0, 1]
    """
    print(f"📥 Loading {num_layers} layers from {layer_dir}/...")
    
    layers = []
    for i in range(1, num_layers + 1):
        path = layer_dir / f"layer_{i:02d}.png"
        
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        layers.append(img)
    
    volume = np.stack(layers, axis=0)  # (num_layers, H, W)
    
    print(f"   Stacked volume shape: {volume.shape}")
    print(f"   Value range: [{volume.min():.4f}, {volume.max():.4f}]")
    
    return volume

def pad_volume_to_channels(volume, target_channels):
    """
    Pad volume to target number of channels using reflect padding.
    
    This allows the model to see the full density transition:
    air -> surface -> interior
    """
    current_channels = volume.shape[0]
    
    if current_channels >= target_channels:
        # Take center slice
        start = (current_channels - target_channels) // 2
        return volume[start:start + target_channels]
    
    # Need to pad
    pad_total = target_channels - current_channels
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    
    print(f"   Padding: {current_channels} -> {target_channels} channels (reflect)")
    print(f"   Pad top: {pad_top}, Pad bottom: {pad_bottom}")
    
    # Reflect padding
    # For numpy: use np.pad with 'reflect' mode
    padded = np.pad(volume, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='reflect')
    
    return padded

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
    
    print(f"   ✅ Model loaded successfully")
    
    return model, device

# =============================================================================
# 🔧 Volumetric Sliding Window Inference
# =============================================================================
def sliding_window_inference_3d(model, volume, tile_size, stride, device, batch_size=8):
    """
    3D sliding window inference on volumetric data.
    
    Args:
        model: PyTorch model expecting (B, C, H, W) input
        volume: (C, H, W) numpy array
        tile_size: Spatial tile size
        stride: Stride between tiles
        device: torch device
        batch_size: Batch size for inference
    
    Returns:
        prediction: (H, W) numpy array of probabilities
    """
    C, H, W = volume.shape
    
    print(f"\n🧮 Running 3D sliding window inference...")
    print(f"   Volume: {volume.shape}")
    print(f"   Tile size: {tile_size}, Stride: {stride}")
    
    # Initialize output
    prediction = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    
    # Gaussian blending weight
    gaussian_weight = np.zeros((tile_size, tile_size), dtype=np.float32)
    center = tile_size // 2
    for i in range(tile_size):
        for j in range(tile_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            gaussian_weight[i, j] = np.exp(-dist**2 / (2 * (tile_size/4)**2))
    
    # Generate tile positions
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
    print(f"   Processing {len(positions)} tiles...")
    
    # Track statistics
    max_pred = 0.0
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(positions), batch_size), desc="   Inferring"):
            batch_positions = positions[batch_start:batch_start + batch_size]
            
            # Extract 3D tiles
            tiles = []
            for y, x in batch_positions:
                tile = volume[:, y:y+tile_size, x:x+tile_size]  # (C, tile, tile)
                tiles.append(tile)
            
            # Stack: (B, C, H, W)
            tiles = np.stack(tiles, axis=0)
            tiles_tensor = torch.from_numpy(tiles).float().to(device)
            
            # Inference
            outputs = model(tiles_tensor)
            outputs = torch.sigmoid(outputs)
            outputs_np = outputs.cpu().numpy()[:, 0, :, :]  # (B, H, W)
            
            # Track max
            batch_max = outputs_np.max()
            if batch_max > max_pred:
                max_pred = batch_max
            
            # Accumulate
            for i, (y, x) in enumerate(batch_positions):
                prediction[y:y+tile_size, x:x+tile_size] += outputs_np[i] * gaussian_weight
                weight[y:y+tile_size, x:x+tile_size] += gaussian_weight
            
            # Clear cache
            if batch_start % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
    
    # Normalize
    weight = np.maximum(weight, 1e-8)
    prediction = prediction / weight
    
    print(f"\n📊 Inference Statistics:")
    print(f"   🎯 Max Prediction Value: {max_pred:.4f}")
    print(f"   Mean Prediction: {prediction.mean():.4f}")
    print(f"   Prediction Range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    
    return prediction

# =============================================================================
# 🔧 Post-Processing and Visualization
# =============================================================================
def post_process(prediction, sigma, threshold):
    """Apply Gaussian smoothing and thresholding."""
    smoothed = gaussian_filter(prediction, sigma=sigma)
    binary = (smoothed > threshold).astype(np.uint8) * 255
    return smoothed, binary

def create_overlay(background, prediction, alpha=0.5):
    """Create heatmap overlay."""
    pred_norm = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(pred_norm, cv2.COLORMAP_HOT)
    
    if len(background.shape) == 2:
        background_bgr = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    else:
        background_bgr = background
    
    overlay = cv2.addWeighted(background_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius - Volumetric Ink Detection")
    print("=" * 60)
    
    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # 1. Load and stack layers
    volume = load_layer_stack(CONFIG["layer_dir"], CONFIG["num_layers"])
    
    # 2. Pad to match model channels
    volume_padded = pad_volume_to_channels(volume, CONFIG["in_channels"])
    print(f"   Final volume shape: {volume_padded.shape}")
    
    # 3. Load model
    model, device = load_model(
        CONFIG["model_path"],
        CONFIG["encoder"],
        CONFIG["in_channels"]
    )
    
    # 4. Run inference
    prediction = sliding_window_inference_3d(
        model, volume_padded,
        CONFIG["tile_size"], CONFIG["stride"],
        device, CONFIG["batch_size"]
    )
    
    # 5. Post-process
    print(f"\n🔧 Post-processing...")
    smoothed, binary = post_process(
        prediction,
        CONFIG["gaussian_sigma"],
        CONFIG["binary_threshold"]
    )
    
    # 6. Load middle layer for overlay
    mid_layer_path = CONFIG["layer_dir"] / f"layer_{CONFIG['num_layers']//2 + 1:02d}.png"
    background = cv2.imread(str(mid_layer_path), cv2.IMREAD_GRAYSCALE)
    
    overlay = create_overlay(background, smoothed, alpha=0.5)
    
    # 7. Save results
    print(f"\n💾 Saving results to {CONFIG['output_dir']}/...")
    
    # Probability map
    pred_path = CONFIG["output_dir"] / "volumetric_prediction.png"
    pred_img = (np.clip(smoothed, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(pred_path), pred_img)
    print(f"   ✅ {pred_path}")
    
    # Overlay
    overlay_path = CONFIG["output_dir"] / "volumetric_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"   ✅ {overlay_path}")
    
    # Binary
    binary_path = CONFIG["output_dir"] / "volumetric_binary.png"
    cv2.imwrite(str(binary_path), binary)
    print(f"   ✅ {binary_path}")
    
    # 8. Statistics
    ink_pixels = np.sum(binary > 0)
    total_pixels = binary.size
    ink_ratio = ink_pixels / total_pixels * 100
    
    print(f"\n📊 Final Results:")
    print(f"   Ink pixels: {ink_pixels:,} / {total_pixels:,} ({ink_ratio:.2f}%)")
    print(f"   Max prediction: {prediction.max():.4f}")
    print(f"   Binary threshold: {CONFIG['binary_threshold']}")
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("📊 Volumetric Ink Detection Complete!")
    print(f"   Output: {CONFIG['output_dir']}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
