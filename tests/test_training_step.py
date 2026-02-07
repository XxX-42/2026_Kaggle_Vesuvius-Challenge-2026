import sys
import os
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
import shutil

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.mini_unetr import MiniUNETR
from src.losses.masked_loss import MaskedBCELoss
from src.training.trainer import VesuviusTrainer

# Define Mock Dataset to rely on nothing external
class MockDataset(Dataset):
    def __init__(self, length=10):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # (1, 16, 256, 256)
        return torch.randn(1, 16, 256, 256), torch.randint(0, 2, (1, 256, 256)).float(), torch.randint(0, 2, (1, 256, 256)).float()

def test_training_dry_run():
    print("=== Training Dry Run Test / 训练试运行测试 ===")
    
    # 1. Load minimal config
    config = {
        'model': {'name': 'MiniUNETR_Test', 'in_channels': 1, 'out_channels': 1, 'feature_size': 16, 'hidden_size': 384},
        'training': {'epochs': 1, 'grad_accumulation_steps': 2, 'amp': True, 'val_interval': 1, 'lr': 1e-3, 'batch_size': 2},
        'loss': {'name': 'MaskedBCE', 'use_ignore_mask': True}
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 2. Setup Mock Data
    train_ds = MockDataset(length=4)
    val_ds = MockDataset(length=2)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)
    
    # 3. Setup Model
    model = MiniUNETR(hidden_size=384).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = MaskedBCELoss(use_ignore_mask=True)
    
    # 4. Init Trainer
    trainer = VesuviusTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device
    )
    
    # 5. Run 1 Epoch
    try:
        print("Running Trainer.fit()...")
        trainer.fit()
        print("✅ Trainer ran successfully without crashing.")
        
        # Check generated checkpoint
        checkpoints = list(trainer.save_dir.glob("*.pth"))
        if len(checkpoints) > 0:
             print(f"✅ Checkpoints found: {[f.name for f in checkpoints]}")
        else:
             print("❌ No checkpoints found.")
             
        # Cleanup
        shutil.rmtree("checkpoints")
        print("Cleanup done.")
        
    except Exception as e:
        print(f"❌ Training Crash: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_dry_run()
