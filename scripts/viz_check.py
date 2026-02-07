import sys
import os
import yaml
import torch
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.mini_unetr import MiniUNETR
from src.inference.predictor import VesuviusPredictor

def main():
    # Load Config
    with open("configs/train_c001.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load BEST Model
    model_path = Path("checkpoints/MiniUNETR_BEST_EPOCH3/best_model.pth")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading checkpoint: {model_path}")
    model = MiniUNETR(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        feature_size=config['model']['feature_size'],
        hidden_size=config['model']['hidden_size']
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Init Predictor
    predictor = VesuviusPredictor(model, device=device, config={'threshold': 0.4}) # Default threshold

    # Data Path (Train set for validation visually)
    data_root = Path("data/vesuvius-challenge-ink-detection/train")
    if not data_root.exists():
        data_root = Path("data") # Fallback for flat structure
    
    # Fragments to test
    fragment_ids = ['1', '2', '3'] 
    
    output_dir = Path("output/viz_epoch3")
    output_dir.mkdir(parents=True, exist_ok=True)

    for fid in fragment_ids:
        frag_path = data_root / fid
        if not frag_path.exists():
            print(f"Fragment {fid} not found, skipping.")
            continue
            
        print(f"Visualizing Fragment {fid}...")
        
        # Load Mask (Ground Truth)
        mask_path = frag_path / "inklabels.png"
        if not mask_path.exists():
             mask_path = frag_path / "mask.png" # Fallback
             
        if mask_path.exists():
            mask_gt = np.array(Image.open(mask_path).convert('1'))
        else:
            mask_gt = None
            print("No ground truth mask found.")

        # Load Volume (Surface Volume, slice 5-20)
        surface_dir = frag_path / "surface_volume"
        layers = []
        # Load a small chunk for quick viz (center of volume)
        # We need roughly 16 slices for the model input
        z_start = 28 # Middle of 65
        z_end = z_start + 16 # 44
        
        # Actually, let's just load what the predictor expects. 
        # The predictor usually takes the full volume and handles slicing, 
        # or we feed it exactly what it needs.
        # Let's load 0-64 to be safe like inference.py
        h, w = 0, 0
        
        # Optimized loading: only load if needed (inference.py style but simpler)
        # We will just load z=20 to z=40 (20 slices)
        # NOTE: Model input is 16 channels usually? 
        # Config says in_channels: 1? No, MiniUNETR usually takes a block
        # Let's check model config in yaml... in_channels: 1. 
        # Wait, if in_channels is 1, does it take a 3D block or 2D image?
        # MiniUNETR is usually 3D or 2D. 
        # If in_channels=1, it might be 2D on a single slice or 3D on 1 channel.
        # Let's assume the standard Vesuvius approach: 3D volume -> 2D Prediction
        # If in_channels=1, the input shape is (B, 1, D, H, W).
        
        # Let's look at config again.
        # Reading `train_c001.yaml`...
        # in_channels: 1.
        
        # Let's just try to match inference.py logic:
        # It stacks layers.
        
        # Just load a small patch to verify "Quality"
        # Loading full 8k x 6k image might OOM or take too long for a quick check.
        # Let's crop a 1024x1024 region where we know there is ink.
        
        try:
             # Find a region with ink
             if mask_gt is not None:
                 y_indices, x_indices = np.where(mask_gt > 0)
                 if len(y_indices) > 0:
                     # Center on some ink
                     yc, xc = y_indices[len(y_indices)//2], x_indices[len(x_indices)//2]
                     crop_size = 1024
                     y1 = max(0, yc - crop_size//2)
                     x1 = max(0, xc - crop_size//2)
                     # Load volume crop
                     
                     vol_crop = []
                     # Load z 20-36 (16 slices)
                     for z in range(20, 36):
                         z_path = surface_dir / f"{z:02d}.tif"
                         if z_path.exists():
                             img = tifffile.imread(str(z_path))
                             vol_crop.append(img[y1:y1+crop_size, x1:x1+crop_size])
                         else:
                             print(f"Missing slice {z}")
                             
                     volume_stack = np.stack(vol_crop, axis=0) # (16, H, W)
                     # Add batch and channel dims: (1, 1, 16, H, W)
                     input_tensor = torch.from_numpy(volume_stack).unsqueeze(0).unsqueeze(0).float().to(device)
                     input_tensor /= 65535.0
                     
                     with torch.no_grad():
                         # Model forward
                         # If model expects (B, 1, 16, H, W)
                         # Simple inference without sliding window for this crop
                         raw_pred = model(input_tensor)
                         pred_sigmoid = torch.sigmoid(raw_pred).squeeze().cpu().numpy()
                         
                     # Plot
                     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                     # 1. Surface (middle slice)
                     axes[0].imshow(volume_stack[8], cmap='gray')
                     axes[0].set_title(f"Surface (z=28) - {fid}")
                     
                     # 2. GT
                     axes[1].imshow(mask_gt[y1:y1+crop_size, x1:x1+crop_size], cmap='gray')
                     axes[1].set_title("Ground Truth")
                     
                     # 3. Pred
                     axes[2].imshow(pred_sigmoid, cmap='jet', vmin=0, vmax=1)
                     axes[2].set_title("Prediction (Prob)")

                     # 4. Pred Thresholded
                     axes[3].imshow(pred_sigmoid > 0.4, cmap='gray')
                     axes[3].set_title("Prediction (> 0.4)")
                     
                     plt.savefig(output_dir / f"viz_{fid}_crop.png")
                     plt.close()
                     print(f"Saved visualization to {output_dir / f'viz_{fid}_crop.png'}")
                     
        except Exception as e:
            print(f"Failed to visualize {fid}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
