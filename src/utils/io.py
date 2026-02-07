import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

def get_tif_paths(data_root: Path, z_slices: List[int]) -> dict:
    """
    Locate TIF files for requested Z-slices.
    Raises FileNotFoundError if ANY slice is missing.
    """
    tif_dir = Path(data_root) / "surface_volume"
    if not tif_dir.exists():
         raise FileNotFoundError(f"Surface volume directory not found: {tif_dir}")

    tif_dict = {}
    
    # 1. Scan available files
    available_files = {int(f.stem): f for f in tif_dir.glob("*.tif") if f.stem.isdigit()}
    
    # 2. Strict Check
    for z in z_slices:
        if z not in available_files:
            raise FileNotFoundError(f"❌ CRITICAL: Requested Z-slice {z} is missing in {tif_dir}")
        tif_dict[z] = available_files[z]
        
    return tif_dict

def load_volume_from_files(
    data_root: Path, 
    z_slices: List[int], 
    region_of_interest: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    Load volume from disk.
    
    Args:
        data_root: Path to data directory.
        z_slices: List of Z-indices to load.
        region_of_interest: Optional tuple (x, y, w, h) to load a specific crop.
                            If None, loads the entire volume (Careful with RAM!).
    
    Returns:
        np.ndarray: Volume of shape (H, W, D) or (h, w, D)
    """
    tif_dict = get_tif_paths(data_root, z_slices)
    
    volume = []
    
    for z in z_slices:
        path = tif_dict[z]
        with Image.open(path) as img:
            if region_of_interest:
                x, y, w, h = region_of_interest
                # PIL Crop: (left, upper, right, lower)
                img_data = img.crop((x, y, x + w, y + h))
            else:
                img_data = img
                
            volume.append(np.array(img_data).astype(np.float32))
            
    # Stack along last axis -> (H, W, D)
    return np.stack(volume, axis=-1)

def load_mask_and_labels(data_root: Path, size: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load valid mask and ink labels. Resizes to 'size' if provided.
    """
    root = Path(data_root)
    mask_path = root / "mask.png"
    ink_path = root / "inklabels.png"
    
    if size is None:
        # Determine size from first available TIF if not provided?
        pass # Assume caller knows or we load raw

    # Helper to load and resize
    def _load(path):
        if not path.exists():
            return None
        with Image.open(path) as img:
            if size and img.size != size:
                return np.array(img.resize(size, Image.NEAREST)).astype(np.uint8)
            return np.array(img).astype(np.uint8)

    mask = _load(mask_path)
    ink = _load(ink_path)
    
    # Defaults
    if mask is None and size:
        mask = np.ones((size[1], size[0]), dtype=np.uint8) * 255
        
    if ink is None and size:
        ink = np.zeros((size[1], size[0]), dtype=np.uint8)
        
    return mask, ink
