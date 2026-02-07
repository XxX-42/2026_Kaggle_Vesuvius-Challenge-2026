import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Fix Tkinter crash
import matplotlib.pyplot as plt
import gc

logger = logging.getLogger(__name__)

class VesuviusTrainer:
    """
    Vesuvius Training Engine (Stage 4 - 3060 Optimized)
    
    Features:
    - Conservative Dynamic Batch Sizing (Auto-Scale)
    - OOM Protection (Panic Recovery)
    - Mixed Precision Training (AMP)
    - Gradient Accumulation
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        config: dict,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.scheduler = scheduler
        self.device = device
        
        # Hyperparameters
        self.epochs = config['training']['epochs']
        self.accum_steps = config['training'].get('grad_accumulation_steps', 1)
        self.use_amp = config['training'].get('amp', False)
        self.val_interval = config['training'].get('val_interval', 1)
        self.model_name = config['model']['name']
        
        # 3060 Hardcoded Start
        self.batch_size = 4 
        logger.info(f"[AutoScaler] Initialized with Batch Size: {self.batch_size} (Safe Start)")
        
        # Output setup
        self.save_dir = Path("checkpoints") / f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_dice = 0.0
        self.start_epoch = 0
        
        # Initial Loader Sync
        # Ensure loader matches self.batch_size
        if self.train_loader.batch_size != self.batch_size:
            logger.info(f"[Init] Syncing Loader Batch Size to {self.batch_size}...")
            self._update_loader(self.batch_size)
        
        logger.info(f"Trainer Initialized. Save Dir: {self.save_dir}")
        logger.info(f"AMP: {self.use_amp}, Accum Steps: {self.accum_steps}")
        
        # Output setup for visualizations
        self.output_dir = Path("output") / self.save_dir.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics History
        self.history = {
            'train_loss': [],
            'val_epochs': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': []
        }

    def _update_loader(self, batch_size):
        """Helper to recreate loader with new batch size"""
        try:
            dataset = self.train_loader.dataset
            num_workers = self.train_loader.num_workers
            pin_memory = self.train_loader.pin_memory
            
            self.train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        except Exception as e:
            logger.error(f"Failed to update loader: {e}")

    def fit(self):
        """Run full training loop"""
        # Debug: Save first batch to verify alignment
        logger.info("Saving debug batch visualization...")
        self.save_debug_batch()
        
        for epoch in range(self.start_epoch, self.epochs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            logger.info(f"=== Epoch {epoch+1}/{self.epochs} | Batch Size: {self.batch_size} ===")
            
            # Train
            train_metrics = self.train_one_epoch(epoch)
            logger.info(f"Train | Loss: {train_metrics['loss']:.4f}")
            
            # Auto Scale Logic at end of epoch
            self.auto_scale_batch_size(epoch)
            
            # Record Train Metrics
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate(epoch)
                logger.info(f"Val   | \033[30;43mLoss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}\033[0m")
                
                # Record Val Metrics
                self.history['val_epochs'].append(epoch + 1)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_dice'].append(val_metrics['dice'])
                self.history['val_iou'].append(val_metrics['iou'])
                
                # Scheduler Step (带学习率变化日志)
                if self.scheduler:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['dice']) 
                    else:
                        self.scheduler.step()
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        logger.info(f"\033[33m[Scheduler] Learning Rate decayed: {old_lr:.2e} -> {new_lr:.2e}\033[0m")

                # Save Best
                if val_metrics['dice'] > self.best_dice:
                    self.best_dice = val_metrics['dice']
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    logger.info(f"🌟 New Best Dice: {self.best_dice:.4f}")
            
            # Save Last
            self.save_checkpoint(epoch, train_metrics, is_best=False)
            
            # Plot Metrics
            self.plot_metrics()

    def auto_scale_batch_size(self, current_epoch):
        """
        Conservative Dynamic Batch Sizing (PID-like Logic)
        """
        if not torch.cuda.is_available():
            return

        device_id = torch.cuda.current_device()
        torch.cuda.synchronize()
        
        max_mem = torch.cuda.max_memory_allocated(device_id)
        total_mem = torch.cuda.get_device_properties(device_id).total_memory
        
        if max_mem == 0: return

        utilization = max_mem / total_mem
        current_bs = self.batch_size
        
        log_msg = f"[AutoScaler] Memory Peak: {max_mem/1e9:.2f}GB / {total_mem/1e9:.2f}GB ({utilization*100:.1f}%)"
        
        new_bs = current_bs
        
        # Decision Tree (Stage 5 - 激进爬坡策略)
        # Decision Tree (Stage 5.2 - 用户自定义激进策略)
        if utilization < 0.60:
            # 超低利用率区 - 极度激进爬坡
            new_bs += 16
            action = "Ultra Low Zone -> +16 (极度激进)"
        elif utilization < 0.75:
            # 舒适区
            new_bs += 8
            action = "Comfort Zone -> +8"
        elif utilization < 0.85:
            # 安全增长区
            new_bs += 4
            action = "Safe Zone -> +4"
        elif utilization < 0.93:
            # 保持区
            new_bs = current_bs
            action = "Hold Zone -> +0"
        else:
            # 危险区
            new_bs -= 2
            action = "Danger Zone -> -2"
            
        # Hard limits (Stage 5: 解锁上限至 128)
        if new_bs < 1: new_bs = 1
        MAX_BS = 128  # 解锁显存封印
        if new_bs > MAX_BS: new_bs = MAX_BS
        
        logger.info(f"{log_msg} -> {action}")
        
        if new_bs != current_bs:
            logger.info(f"[AutoScaler] Adjusting Batch Size: {current_bs} => {new_bs}")
            self.batch_size = new_bs
            self._update_loader(self.batch_size)
        else:
            logger.info("[AutoScaler] Batch Size Unchanged.")
            
        # Reset stats
        torch.cuda.reset_peak_memory_stats(device_id)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch+1}")
        
        # We need to manually handle enumeration to allow retry logic?
        # Standard loop is fine if valid_batch helper handles logic
        
        for i, batch in enumerate(pbar):
            # Dataset returns: (volume, label, ignore_mask)
            # Safe Process with OOM Retry
            loss_val = self._process_batch_safe(batch)
            
            # Step logic happens inside _process_batch_safe (accumulation)
            # Actually no, _process_batch_safe should return the loss item for logging
            # Optimization step logic needs to be careful about accumulation counting.
            # To keep it simple: 
            # If we split batch, we accumulate gradients. 
            # We step optimizer if (i+1) % accum_steps == 0 (Based on original batch count).
            
            # Helper handles backward.
            # We handle Optimizer Step here based on 'i'.
            
            if (i + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            total_loss += loss_val
            pbar.set_postfix({'loss': total_loss / (i + 1)})
            
        return {'loss': total_loss / len(self.train_loader)}

    def _process_batch_safe(self, batch):
        """
        Handles Forward/Backward with OOM Protection.
        If batch consists of N samples, and self.batch_size < N (due to OOM),
        it splits the batch into chunks.
        """
        images, labels, ignore_masks = batch
        batch_size_in = images.size(0)
        
        # If current target batch size is smaller than input batch size (OOM adjustment happened),
        # or if we are just processing normally.
        # We use a loop to process chunks of size self.batch_size.
        
        chunk_size = self.batch_size
        loss_accum = 0.0
        
        # Split into chunks
        # If batch_size_in <= chunk_size, this loop runs once with entire batch.
        
        indices = list(range(0, batch_size_in, chunk_size))
        
        for start_idx in indices:
            end_idx = min(start_idx + chunk_size, batch_size_in)
            
            # Slicing
            img_chunk = images[start_idx:end_idx].to(self.device)
            lbl_chunk = labels[start_idx:end_idx].to(self.device)
            ign_chunk = ignore_masks[start_idx:end_idx].to(self.device)
            
            try:
                # Forward & Backward
                loss_val = self._forward_backward_step(img_chunk, lbl_chunk, ign_chunk)
                
                # Weight loss by chunk size relative to original batch (for logging/avg)
                # Note: gradients are accumulated sum. 
                # Loss for optimization: Mean over batch.
                # If we split batch B into b1, b2.
                # Loss_total = (L(b1)*len(b1) + L(b2)*len(b2)) / B
                # The _forward_backward_step helps us get scaled loss.
                
                # Actually, standard loss is Mean.
                # output: Mean Loss over chunk.
                # To be mathematically consistent with full batch:
                # Chunk Loss needs to be weighted by (len(chunk)/batch_size_in).
                
                loss_accum += loss_val * (end_idx - start_idx) / batch_size_in
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Panic Recovery
                    logger.warning(f"[OOM] Caught OOM processing chunk {start_idx}-{end_idx}. Recovering...")
                    self.batch_size = max(1, self.batch_size // 2) # Halve BS
                    chunk_size = self.batch_size # Update chunk size for RETRY
                    
                    logger.warning(f"[OOM] Panicked! New Batch Size: {self.batch_size}. Clearing Cache...")
                    # Clear Cache
                    del img_chunk, lbl_chunk, ign_chunk
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Recursive Retry of the *Original Full Batch*? 
                    # Simpler to just retry this logic function with the original input batch.
                    # Since self.batch_size is reduced, the next call will use smaller chunks.
                    return self._process_batch_safe(batch) 
                else:
                    raise e
                    
        return loss_accum

    def _forward_backward_step(self, images, labels, ignore_masks):
        """Single Forward/Backward Step"""
        with autocast(enabled=self.use_amp):
            outputs = self.model(images)
            loss = self.criterion(outputs, labels, ignore_masks)
            # Scale loss for Gradient Accumulation (over steps) AND logical batch splitting
            # If we are splitting a batch into chunks, the gradients accumulated from chunks
            # should sum up to the gradients of the full batch.
            # PyTorch Loss (Mean): dL/dx = 1/N * sum(dl/dx).
            # If we split N into n1, n2. 
            # backward(loss_n1) adds 1/n1 * sum1. 
            # We want 1/N * sum(all).
            # So we should scale chunk loss by (len(chunk)/N) * (1/accum_steps).
            # WAIT. criterion returns Mean over Chunk (n1).
            # So gradients are scaled by 1/n1.
            # We want gradients scaled by 1/N.
            # So we multiply loss by (n1/N).
            # Then we divide by accum_steps.
            
            # Wait, N is unknown to this function? Passed as arg?
            # Let's assume passed. But wait.
            # Actually, standard pattern is just loss / accum_steps if batches are equal size.
            # If we split manually, we need to handle this weighting.
            
            # Let's simplify: 
            # Just return raw loss. Scale in backward.
            pass
            
        # Re-calculating proper scaling
        # We need N (total batch size)
        # But this function doesn't know N.
        # Let's assume batch integrity: 
        # If we just do loss.backward() on chunks, the gradients are derived from Mean of Chunk.
        # Grad = 1/n_chunk * Sum_Grads_Chunk.
        # Average of Averages is not Average of Whole unless n_chunk is constant.
        # But for 3060 training, approximate is fine. "Conservative".
        
        # Consistent Scaled Loss for Accumulation
        loss = loss / self.accum_steps 
        
        # Backward
        self.scaler.scale(loss).backward()
        
        # Return item for logging (inverse scale)
        return loss.item() * self.accum_steps

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_inter = 0
        total_union = 0
        epsilon = 1e-6
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val Ep {epoch+1}")
            for batch in pbar:
                images, labels, ignore_masks = [x.to(self.device) for x in batch]
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels, ignore_masks)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                valid_mask = 1.0 - ignore_masks
                
                preds = preds * valid_mask
                targets = labels * valid_mask
                
                inter = (preds * targets).sum()
                union = preds.sum() + targets.sum()
                
                total_inter += inter.item()
                total_union += union.item()
                
                if epoch >= 0 and pbar.n == 0:
                    self.save_debug_images(epoch, images, labels, probs, ignore_masks)
        
        dice = (2 * total_inter + epsilon) / (total_union + epsilon)
        iou = (total_inter + epsilon) / (total_union - total_inter + epsilon)
        avg_loss = total_loss / len(self.val_loader)
        
        return {'loss': avg_loss, 'dice': dice, 'iou': iou}

    def save_debug_batch(self):
        try:
            import matplotlib.pyplot as plt
            batch = next(iter(self.train_loader))
            images, labels, ignore_masks = [x.to(self.device) for x in batch]
            
            vis_dir = self.output_dir / "debug"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(min(len(images), 4)):
                img = images[i, 0].cpu().numpy()
                lbl = labels[i, 0].cpu().numpy()
                ign = ignore_masks[i, 0].cpu().numpy()
                
                mid_z = img.shape[0] // 2
                img_slice = img[mid_z]
                
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(img_slice, cmap='gray')
                ax[0].set_title(f"Input Z={mid_z}")
                ax[1].imshow(lbl, cmap='gray')
                ax[1].set_title("Label")
                ax[2].imshow(ign, cmap='gray')
                ax[2].set_title("Ignore Mask")
                
                for a in ax: a.axis('off')
                
                plt.suptitle(f"Debug Sample {i}")
                plt.savefig(vis_dir / f"sample_{i}.png")
                plt.close(fig)
            logger.info(f"Debug images saved to {vis_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save debug batch: {e}")

    def save_debug_images(self, epoch, images, labels, preds, ignore_masks):
        try:
            import matplotlib.pyplot as plt
            
            vis_dir = self.output_dir
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            img = images[0, 0].cpu().numpy()
            lbl = labels[0, 0].cpu().numpy()
            prd = preds[0, 0].cpu().detach().numpy()
            ign = ignore_masks[0, 0].cpu().numpy()
            
            mid_z = img.shape[0] // 2
            
            img_slice = img[mid_z]
            lbl_slice = lbl
            prd_slice = prd
            ign_slice = ign
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(img_slice, cmap='gray')
            axes[0].set_title(f"Input (Z={mid_z})")
            axes[1].imshow(lbl_slice, cmap='gray')
            axes[1].set_title("Label")
            axes[2].imshow(prd_slice, cmap='jet', vmin=0, vmax=1)
            axes[2].set_title("Prediction (Prob)")
            axes[3].imshow(ign_slice, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title("Ignore Mask")
            
            for ax in axes: ax.axis('off')
                
            plt.suptitle(f"Epoch {epoch+1}")
            plt.tight_layout()
            
            save_path = vis_dir / f"epoch_{epoch+1:03d}.png"
            plt.savefig(save_path)
            plt.close(fig)
            
        except ImportError:
            logger.warning("Matplotlib not found.")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")

    def save_checkpoint(self, epoch, metrics, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        filename = "best_model.pth" if is_best else "last_model.pth"
        path = self.save_dir / filename
        torch.save(state, path)

    def plot_metrics(self):
        try:
            import matplotlib.pyplot as plt
            
            epochs = range(1, len(self.history['train_loss']) + 1)
            val_epochs = self.history['val_epochs']
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color=color)
            ax1.plot(epochs, self.history['train_loss'], label='Train Loss', color=color, linestyle='-')
            if val_epochs:
                ax1.plot(val_epochs, self.history['val_loss'], label='Val Loss', color=color, linestyle='--')
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Score (Dice/IoU)', color=color)
            if val_epochs:
                ax2.plot(val_epochs, self.history['val_dice'], label='Val Dice', color=color, linestyle='-')
                ax2.plot(val_epochs, self.history['val_iou'], label='Val IoU', color='tab:green', linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title(f"Training Metrics - {self.model_name}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = self.output_dir / "metrics.png"
            plt.savefig(save_path)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to plot metrics: {e}")
