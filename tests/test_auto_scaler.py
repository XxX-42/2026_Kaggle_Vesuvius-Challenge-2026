import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.training.trainer import VesuviusTrainer

class TestAutoScaler(unittest.TestCase):
    def setUp(self):
        # Mock dependencies for Trainer init
        self.model = MagicMock()
        self.train_loader = MagicMock()
        self.train_loader.batch_size = 4
        self.val_loader = MagicMock()
        self.optimizer = MagicMock()
        self.criterion = MagicMock()
        
        self.config = {
            'training': {
                'epochs': 10,
                'grad_accumulation_steps': 1,
                'amp': True,
                'val_interval': 1
            },
            'model': {'name': 'TestModel'}
        }
        
    @patch('src.training.trainer.torch.cuda.is_available', return_value=True)
    @patch('src.training.trainer.torch.cuda.current_device', return_value=0)
    @patch('src.training.trainer.torch.cuda.synchronize')
    @patch('src.training.trainer.torch.cuda.get_device_properties')
    @patch('src.training.trainer.torch.cuda.max_memory_allocated')
    @patch('src.training.trainer.torch.cuda.reset_peak_memory_stats')
    def test_logic(self, mock_reset, mock_max_mem, mock_props, mock_sync, mock_dev, mock_avail):
        # Setup Trainer
        trainer = VesuviusTrainer(
            self.model, self.train_loader, self.val_loader, 
            self.optimizer, self.criterion, self.config
        )
        
        # Test Case 1: Low Memory (Utilization 50%) -> Comfort Zone
        # Total 6GB, Used 3GB
        mock_props.return_value.total_memory = 6 * 1024**3
        mock_max_mem.return_value = 3 * 1024**3
        
        print("\n[Test] Case 1: Utilization 50% (Comfort Zone)")
        trainer.auto_scale_batch_size(1)
        self.assertEqual(trainer.batch_size, 6) # 4 + 2
        print(f"Batch Size: 4 -> {trainer.batch_size} (Expected 6)")
        
        # Test Case 2: Moderate (Utilization 80%) -> Safe Zone
        # Used 4.8GB / 6GB
        mock_max_mem.return_value = 4.8 * 1024**3
        
        print("\n[Test] Case 2: Utilization 80% (Safe Zone)")
        trainer.auto_scale_batch_size(2)
        # Previous was 6. +1 -> 7.
        self.assertEqual(trainer.batch_size, 7)
        print(f"Batch Size: 6 -> {trainer.batch_size} (Expected 7)")
        
        # Test Case 3: High (Utilization 90%) -> Hold Zone
        # Used 5.4GB / 6GB
        mock_max_mem.return_value = 5.4 * 1024**3
        
        print("\n[Test] Case 3: Utilization 90% (Hold Zone)")
        trainer.auto_scale_batch_size(3)
        self.assertEqual(trainer.batch_size, 7) # Unchanged
        print(f"Batch Size: 7 -> {trainer.batch_size} (Expected 7)")
        
        # Test Case 4: Critical (Utilization 95%) -> Danger Zone
        # Used 5.7GB / 6GB
        mock_max_mem.return_value = 5.7 * 1024**3
        
        print("\n[Test] Case 4: Utilization 95% (Danger Zone)")
        trainer.auto_scale_batch_size(4)
        self.assertEqual(trainer.batch_size, 5) # 7 - 2
        print(f"Batch Size: 7 -> {trainer.batch_size} (Expected 5)")
        
        # Test Case 5: OOM Simulation (Not covered by auto_scale, but OOM logic logic is inside scaler?)
        # Panic Retreat check
        # If we hit OOM limit logic check in auto_scaler is implicit by high usage.
        # But real OOM happens in training loop.
        
if __name__ == '__main__':
    unittest.main()
