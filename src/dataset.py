import numpy as np
import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from PIL import Image
from tqdm import tqdm
import random

class VesuviusDataset(Dataset):
    """
    Vesuvius Challenge 2026 Core Dataset
    
    Features:
    - Strict Resolution Guard: Raises Error if Mask != InkLabels != Volume
    - 3D Sandwich Loader: Loads consecutives Z-slices centered around target
    - Physical Normalization: Linear clip [window_min, window_max] -> [0, 1]
    - RGT & Native Support: Auto-detects mode based on directory structure
    """
    def __init__(self, 
                 volume_path, 
                 ink_mask_path, 
                 fragment_mask_path=None,
                 is_rgt=False,
                 z_start=0, 
                 n_channels=16,
                 patch_size=224,
                 window_range=(18000, 28000),
                 augmentations=None,
                 min_ink_threshold=1e-5): # Ultra-sensitive for sparse ink strokes
        
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.augmentations = augmentations
        self.is_rgt = is_rgt
        self.min_ink_threshold = min_ink_threshold
        
        # 1. Load Labels First (The Source of Truth)
        print(f"\n🧱 Initializing VesuviusDataset (RGT={is_rgt})")
        self.ink_mask, self.H, self.W = self._load_secure_label(ink_mask_path)
        
        if fragment_mask_path and Path(fragment_mask_path).exists():
            self.frag_mask, h_f, w_f = self._load_secure_label(fragment_mask_path)
            if (h_f, w_f) != (self.H, self.W):
                 raise RuntimeError(f"🚨 Fragment Mask Mismatch: {w_f}x{h_f} vs Ink {self.W}x{self.H}")
        else:
            self.frag_mask = np.ones((self.H, self.W), dtype=np.float32)
            
        # 2. Volume Loading with Strict Alignment
        self.volume = self._load_volume(volume_path, z_start, n_channels, window_range, (self.W, self.H))
        
        # 3. Patch Mining (Balanced Sampling Prep)
        self._mine_patches()

    def _load_secure_label(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing label file: {path}")
            
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
            
        h, w = img.shape
        
        # Forensics: Check for "Thumbnail" trap
        if w < 2000 or h < 2000:
             print(f"⚠️ WARNING: Label {path.name} is oddly small ({w}x{h}). Is this a preview image?")
             
        # Forensics: Density Check
        density = (img > 0).mean()
        if density > 0.30:
             print(f"⚠️ WARNING: Label Density {density:.1%} is suspiciously high. Potential Geometric artifact?")
             
        return (img > 127).astype(np.float32), h, w

    def _load_volume(self, volume_path, z_start, n_channels, window_range, target_hw):
        target_w, target_h = target_hw
        volume_path = Path(volume_path)
        
        if self.is_rgt:
            files = sorted(volume_path.glob("layer_*.png"), key=lambda x: int(x.stem.split('_')[1]))
        else:
            files = sorted(volume_path.glob("*.tif"), key=lambda x: int(x.stem) if x.stem.isdigit() else 9999)
            
        # Select Slices
        if len(files) < n_channels:
             msg = f"Requested {n_channels} channels but only found {len(files)} files in {volume_path}"
             print(f"⚠️ {msg}. Padding will be used.")
             
        # Safe slice indexing
        start_idx = max(0, min(z_start, len(files) - n_channels))
        selected_files = files[start_idx : start_idx + n_channels]
        
        print(f"   Using Z-Range: {selected_files[0].name} ... {selected_files[-1].name}")
        
        # Load Volume
        D = len(selected_files)
        volume = np.zeros((D, target_h, target_w), dtype=np.float32)
        w_min, w_max = window_range
        
        for i, f in enumerate(tqdm(selected_files, desc="   Loading 3D Volume", leave=False)):
            img = np.array(Image.open(f)).astype(np.float32)
            
            # 🚨 STRICT RESOLUTION CHECK
            if img.shape != (target_h, target_w):
                raise RuntimeError(
                    f"\n🛑 CRITICAL MISMATCH on slice {f.name}:\n"
                    f"   Slice Shape: {img.shape}\n"
                    f"   Label Shape: ({target_h}, {target_w})\n"
                    f"   Action: Training aborted to protect data integrity."
                )
            
            # Physical Normalization
            img = np.clip(img, w_min, w_max)
            img = (img - w_min) / (w_max - w_min)
            volume[i] = img
            
        # Pad Z-axis if insufficient depth
        if D < n_channels:
            pad_amt = n_channels - D
            volume = np.pad(volume, ((0, pad_amt), (0,0), (0,0)), mode='edge')
            
        return volume

    def _mine_patches(self):
        self.ink_coords = []
        self.bg_coords = []
        step = self.patch_size // 2
        
        print("   Mining patches...")
        
        # Iterate over valid fragment area
        # Optimization: Only scan where frag_mask has content? 
        # For now, stride over full image, check valid later
        
        y_steps = range(0, self.H - self.patch_size, step)
        x_steps = range(0, self.W - self.patch_size, step)
        
        for y in y_steps:
             for x in x_steps:
                 # 1. Check Fragment (Paper) validity
                 paper_val = self.frag_mask[y:y+self.patch_size, x:x+self.patch_size].mean()
                 if paper_val < 0.1: # At least 10% paper
                     continue
                     
                 # 2. Check Ink
                 ink_val = self.ink_mask[y:y+self.patch_size, x:x+self.patch_size].mean()
                 
                 if ink_val > self.min_ink_threshold:
                     self.ink_coords.append((y, x, ink_val))
                 else:
                     self.bg_coords.append((y, x, ink_val))
                     
        print(f"✅ Minerals Found: {len(self.ink_coords)} Ink Patches, {len(self.bg_coords)} Paper Patches.")
        if not self.ink_coords:
             print("⚠️ WARNING: ZERO ink patches found. Check min_ink_threshold or data source.")

    def get_patch(self, y, x, use_augmentation=True):
        ps = self.patch_size
        vol_patch = self.volume[:, y:y+ps, x:x+ps].copy()
        mask_patch = self.ink_mask[y:y+ps, x:x+ps].copy()
        
        if use_augmentation and self.augmentations:
            # Albumentations standard H,W,C
            vol_trans = vol_patch.transpose(1, 2, 0)
            data = self.augmentations(image=vol_trans, mask=mask_patch)
            vol_patch = data['image'].transpose(2, 0, 1)
            mask_patch = data['mask']
            
        return (
            torch.from_numpy(vol_patch.astype(np.float32)),
            torch.from_numpy(mask_patch[np.newaxis, :, :].astype(np.float32))
        )

    def __getitem__(self, item):
        if isinstance(item, tuple):
            y, x = item
        else:
            y, x, _ = self.all_coords[item] # Access via Combined List in Sampler
            
        return self.get_patch(y, x, use_augmentation=True)
    
    @property
    def all_coords(self):
         return self.ink_coords + self.bg_coords
    
    def __len__(self):
        return len(self.ink_coords) + len(self.bg_coords)


class BalancedBatchSampler(Sampler):
    """
    Forced 50/50 Sampling Strategy.
    Ensures model sees ink in every batch to prevent 'all-zero' collapse.
    """
    def __init__(self, dataset, batch_size, ink_ratio=0.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ink_per_batch = int(batch_size * ink_ratio)
        self.bg_per_batch = batch_size - self.ink_per_batch
        
    def __iter__(self):
        ink_pool = list(range(len(self.dataset.ink_coords))) 
        bg_pool = list(range(len(self.dataset.ink_coords), len(self.dataset.all_coords))) # Offset indices
        
        random.shuffle(ink_pool)
        random.shuffle(bg_pool)
        
        # Calculate max batches supported by ink count (limiter)
        if not ink_pool:
             raise RuntimeError("Cannot use BalancedSampler with 0 ink patches.")
             
        n_batches = len(ink_pool) // max(1, self.ink_per_batch)
        
        for _ in range(n_batches):
            batch = []
            # Fill Ink
            for _ in range(self.ink_per_batch):
                batch.append(ink_pool.pop())
            
            # Fill BG (Resample if needed)
            for _ in range(self.bg_per_batch):
                if not bg_pool:
                     # Refill if exhausted
                     bg_pool = list(range(len(self.dataset.ink_coords), len(self.dataset.all_coords)))
                     random.shuffle(bg_pool)
                batch.append(bg_pool.pop())
                
            random.shuffle(batch)
            yield batch

    def __len__(self):
         return len(self.dataset.ink_coords) // max(1, self.ink_per_batch)
