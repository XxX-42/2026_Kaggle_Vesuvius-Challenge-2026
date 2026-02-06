"""
Vesuvius Challenge 2026 - Final RGT Inference (Fine-tuned Model)
=================================================================
Apply fine-tuned model to RGT-flattened layer for final ink detection.

Input: flattened_layers/layer_07.png
Model: models/vesuvius_rgt_finetuned.pth
Output: final_rgt_finetuned_pred.png, final_rgt_finetuned_overlay.png
"""

import numpy as np
import torch
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
    "image_path": "flattened_layers/layer_07.png",
    
    # Model
    "model_path": "models/vesuvius_rgt_finetuned.pth",
    "encoder": "resnet18",
    "in_channels": 14,
    
    # Inference
    "tile_size": 224,
    "stride": 112,
    "batch_size": 16,
    
    # Post-processing
    "gaussian_sigma": 1.0,
    "binary_threshold": 0.10,
    
    # Output
    "output_dir": Path("final_rgt_results"),
}

# =============================================================================
# 🔧 Model Loading
# =============================================================================
def load_model(model_path, encoder, in_channels, device):
    """Load fine-tuned model."""
    print(f"📥 Loading fine-tuned model: {model_path}")
    
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
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded")
    
    return model

# =============================================================================
# 🔧 Sliding Window Inference
# =============================================================================
def sliding_window_inference(model, image, in_channels, tile_size, stride, device, batch_size):
    """
    Sliding window inference with channel replication.
    
    Args:
        image: (H, W) grayscale image, normalized 0-1
        in_channels: number of channels to replicate to
    """
    H, W = image.shape
    
    print(f"\n🧮 Running sliding window inference...")
    print(f"   Image: {H} x {W}")
    print(f"   Tile: {tile_size}, Stride: {stride}")
    
    # Output arrays
    prediction = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    
    # Gaussian blending weight
    gaussian_weight = np.zeros((tile_size, tile_size), dtype=np.float32)
    center = tile_size // 2
    for i in range(tile_size):
        for j in range(tile_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            gaussian_weight[i, j] = np.exp(-dist**2 / (2 * (tile_size/4)**2))
    
    # Generate positions
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
            
            # Extract and replicate tiles
            tiles = []
            for y, x in batch_pos:
                tile = image[y:y+tile_size, x:x+tile_size]
                # Replicate to in_channels
                tile_multi = np.repeat(tile[np.newaxis, :, :], in_channels, axis=0)
                tiles.append(tile_multi)
            
            tiles = np.stack(tiles, axis=0)  # (B, C, H, W)
            tiles_tensor = torch.from_numpy(tiles).float().to(device)
            
            # Forward
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
    
    print(f"\n📊 Inference Results:")
    print(f"   🎯 Max Prediction: {max_pred:.4f}")
    print(f"   Mean: {prediction.mean():.4f}")
    print(f"   Range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    
    return prediction

# =============================================================================
# 🔧 Post-processing
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
    print("🔬 Vesuvius - Final RGT Inference (Fine-tuned)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    # Create output dir
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # 1. Load image
    print(f"\n📥 Loading: {CONFIG['image_path']}")
    image = cv2.imread(CONFIG["image_path"], cv2.IMREAD_GRAYSCALE)
    image_norm = image.astype(np.float32) / 255.0
    print(f"   Shape: {image.shape}")
    
    # 2. Load model
    model = load_model(
        CONFIG["model_path"],
        CONFIG["encoder"],
        CONFIG["in_channels"],
        device
    )
    
    # 3. Inference
    prediction = sliding_window_inference(
        model, image_norm,
        CONFIG["in_channels"],
        CONFIG["tile_size"], CONFIG["stride"],
        device, CONFIG["batch_size"]
    )
    
    # 4. Post-process
    print(f"\n🔧 Post-processing (sigma={CONFIG['gaussian_sigma']})...")
    smoothed, binary = post_process(
        prediction,
        CONFIG["gaussian_sigma"],
        CONFIG["binary_threshold"]
    )
    
    # 5. Create overlay
    overlay = create_overlay(image, smoothed, alpha=0.5)
    
    # 6. Save results
    print(f"\n💾 Saving to {CONFIG['output_dir']}/...")
    
    # Prediction map
    pred_img = (np.clip(smoothed, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(CONFIG["output_dir"] / "final_rgt_finetuned_pred.png"), pred_img)
    print(f"   ✅ final_rgt_finetuned_pred.png")
    
    # Overlay
    cv2.imwrite(str(CONFIG["output_dir"] / "final_rgt_finetuned_overlay.png"), overlay)
    print(f"   ✅ final_rgt_finetuned_overlay.png")
    
    # Binary
    cv2.imwrite(str(CONFIG["output_dir"] / "final_rgt_finetuned_binary.png"), binary)
    print(f"   ✅ final_rgt_finetuned_binary.png")
    
    # 7. Stats
    ink_pixels = np.sum(binary > 0)
    total = binary.size
    
    print(f"\n📊 Final Statistics:")
    print(f"   Ink pixels: {ink_pixels:,} / {total:,} ({ink_pixels/total*100:.2f}%)")
    print(f"   Max prediction: {prediction.max():.4f}")
    print(f"   Binary threshold: {CONFIG['binary_threshold']}")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("📊 Final RGT Inference Complete!")
    print(f"   Output: {CONFIG['output_dir']}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
