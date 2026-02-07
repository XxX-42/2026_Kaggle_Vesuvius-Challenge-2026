import torch
import unittest
import sys
import os
import yaml
import time

# Add src to path
sys.path.append(os.getcwd())

from src.models.mini_unetr import MiniUNETR
from src.inference.predictor import VesuviusPredictor

class TestInferenceMemory(unittest.TestCase):
    def setUp(self):
        # Load Config
        with open("configs/inference_3060.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("Warning: CUDA not available. Skipping exact memory test.")
            
    def test_peak_memory(self):
        if self.device == "cpu":
            return 
            
        print("\n[Test] Starting Memory Stress Test...")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        baseline = torch.cuda.memory_allocated()
        print(f"Baseline Memory: {baseline / 1e6:.2f} MB")
        
        # 1. Load Model
        model_args = self.config['model']['args']
        model = MiniUNETR(**model_args)
        model.to(self.device).eval()
        
        model_mem = torch.cuda.memory_allocated()
        print(f"Model Memory: {(model_mem - baseline) / 1e6:.2f} MB")
        
        # 2. Simulate 1 Batch Inference with Sequential TTA
        # Batch Size 4, TTA 3 shifts
        # But we process shifts sequentially, so max tensor is Batch Size 4.
        # Input Shape: (4, 1, 16, 256, 256). Float32.
        # Wait, inside predictor we convert to FP16 via AMP? No, input tensor is Float32 usually?
        # Predictor logic: 
        # crop = ... (float32)
        # t = torch.from_numpy(c) ...
        # stack -> GPU.
        
        B = self.config['inference']['batch_size']
        C = 1
        D = 16
        H = 256
        W = 256
        
        # Create dummy input batch
        # We need to simulate the loop inside predictor._process_batch
        
        print(f"Simulating Batch Size {B} inference...")
        
        # Loop for TTA shifts
        shifts = [0, -1, 1]
        
        for shift in shifts:
            # Create input tensor
            input_tensor = torch.randn(B, C, D, H, W, device=self.device)
            # FP16 AMP context
            with torch.cuda.amp.autocast():
                # Forward
                output = model(input_tensor)
                preds = torch.sigmoid(output).squeeze(1)
                
            # Cleanup
            del input_tensor, output, preds
            torch.cuda.empty_cache()
            
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"Peak Memory usage: {peak_mem / 1e9:.2f} GB")
        
        # Assertion: < 4.5 GB
        # 4.5 * 1024^3 bytes
        limit = 4.5 * 1024**3
        
        if peak_mem > limit:
            self.fail(f"Memory Limit Exceeded! {peak_mem / 1e9:.2f} GB > 4.5 GB")
        else:
            print(f"✅ Memory Test Passed: {peak_mem / 1e9:.2f} GB < 4.5 GB")

if __name__ == '__main__':
    unittest.main()
