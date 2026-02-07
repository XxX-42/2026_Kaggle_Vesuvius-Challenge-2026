import sys
import os
import torch
import numpy as np
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.mini_unetr import MiniUNETR
from src.inference.predictor import VesuviusPredictor
from scripts.inference import rle_encode

def test_inference_pipeline():
    print("=== Inference Pipeline Verification ===")
    
    # 1. Mock Config & Model
    config = {
        'tile_size': 256,
        'stride': 128,
        'batch_size': 4,
        'tta_shifts': [-1, 0, 1]
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = MiniUNETR(hidden_size=384).to(device).eval()
    predictor = VesuviusPredictor(model, device=device, config=config)
    
    # 2. Mock Data (1024x1024)
    H, W = 1024, 1024
    D = 30 # Load enough layers
    volume = np.random.randint(0, 65535, (D, H, W), dtype=np.uint16)
    
    # Mask (Scout Test: 50% valid)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[256:768, 256:768] = 1 # Center box valid
    
    # 3. Run Prediction
    print("Running Predictor...")
    prob_map = predictor.predict_fragment(volume, mask, z_start=5, z_end=20)
    
    print(f"Output Shape: {prob_map.shape} (Expected {H}, {W})")
    assert prob_map.shape == (H, W)
    
    # 4. Test RLE
    print("Running RLE Encoding...")
    rle = rle_encode(prob_map, threshold=0.5)
    print(f"RLE Length: {len(rle)}")
    
    print("✅ Inference Pipeline Verified!")

if __name__ == "__main__":
    test_inference_pipeline()
