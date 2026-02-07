import sys
import os
import torch
import logging

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.timesformer import TimeSformer
from src.models.mini_unetr import MiniUNETR

def verify_models():
    print("=== Vesuvius Model Verification / 模型验证 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Input Tensor: (Batch=2, Channel=1, Depth=16, Height=256, Width=256)
    input_tensor = torch.randn(2, 1, 16, 256, 256).to(device)
    print(f"Input Shape: {input_tensor.shape}")

    # --- 1. Verify TimeSformer ---
    print("\n--- Testing TimeSformer ---")
    try:
        model_ts = TimeSformer(
            img_size=256, 
            patch_size=16, 
            in_channels=1, 
            num_frames=16, 
            embed_dim=256, 
            depth=2, # Small depth for quick test
            num_heads=4
        ).to(device)
        
        out_ts = model_ts(input_tensor)
        print(f"✅ TimeSformer Output Shape: {out_ts.shape}")
        
        expected_shape = (2, 1, 256, 256)
        assert out_ts.shape == expected_shape, f"Expected {expected_shape}, got {out_ts.shape}"
        
    except Exception as e:
        print(f"❌ TimeSformer Failed: {e}")
        # Print full trace for debugging if needed
        import traceback
        traceback.print_exc()

    # --- 2. Verify MiniUNETR ---
    print("\n--- Testing MiniUNETR ---")
    try:
        model_unetr = MiniUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=16
        ).to(device)
        
        out_unetr = model_unetr(input_tensor)
        print(f"✅ MiniUNETR Output Shape: {out_unetr.shape}")
        
        expected_shape = (2, 1, 256, 256)
        assert out_unetr.shape == expected_shape, f"Expected {expected_shape}, got {out_unetr.shape}"
        
    except Exception as e:
        print(f"❌ MiniUNETR Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_models()
