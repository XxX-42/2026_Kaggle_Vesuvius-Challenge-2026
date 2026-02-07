import sys
import os
import torch
import shutil
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.training.trainer import VesuviusTrainer

def test_visualization():
    print("Testing Visualization Logic...")
    
    # Mock Trainer
    class MockConfig:
        def __getitem__(self, key): return {}
        def get(self, key, default): return default
        
    class MockTrainer(VesuviusTrainer):
        def __init__(self):
            self.model_name = "TestVis"
            self.save_dir = Path("checkpoints/Test_Run")
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
    trainer = MockTrainer()
    
    # Mock Data: (1, 1, 16, 256, 256)
    images = torch.randn(1, 1, 16, 256, 256)
    labels = torch.randint(0, 2, (1, 1, 256, 256)).float()
    preds = torch.rand(1, 1, 256, 256)
    ignore_masks = torch.randint(0, 2, (1, 1, 256, 256)).float()
    
    try:
        trainer.save_debug_images(0, images, labels, preds, ignore_masks)
        
        # Check output
        out_dir = Path("output") / trainer.save_dir.name
        expected_file = out_dir / "epoch_001.png"
        
        if expected_file.exists():
            print(f"✅ Visualization saved to: {expected_file}")
            print(f"File size: {expected_file.stat().st_size} bytes")
        else:
            print(f"❌ File not found at: {expected_file}")
            # Check if dir exists
            if out_dir.exists():
                print(f"Directory {out_dir} exists.")
                print(list(out_dir.glob("*")))
            else:
                print(f"Directory {out_dir} does not exist.")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    test_visualization()
