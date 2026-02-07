import sys
import os
import torch
import logging

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import VesuviusDataset

def verify_setup():
    print("=== Vesuvius Pipeline Verification / 流水线验证 ===")
    
    # 1. Check CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # 2. Instantiate Dataset
    print("\n--- 初始化数据集 / Initializing Dataset ---")
    dataset = VesuviusDataset(data_path="./data")
    
    # 3. Simulate Data Fetch
    print("--- 模拟数据读取 / Simulating Data Fetch ---")
    try:
        sample = dataset[0]
        print(f"Sample Shape: {sample.shape}")
        
        # 4. Verify Shape
        expected_shape = (1, 16, 256, 256) # Based on dataset.py hardcoded logic for now
        # Note: In dataset.py we mocked H, W as 256. 
        # Ideally it matches the input, but let's verify dimensions C and D.
        
        assert sample.shape[0] == 1, "Channel dim should be 1"
        assert sample.shape[1] == 16, "Depth dim should be 16 (Layers 5-20)"
        
        print(f"✅ Shape Check Passed! / 形状验证通过!")
        print(f"   Expected: (1, 16, H, W)")
        print(f"   Actual:   {sample.shape}")
        
    except Exception as e:
        print(f"❌ Verification Failed / 验证失败: {e}")
        raise e

if __name__ == "__main__":
    verify_setup()
