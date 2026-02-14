
import os
import sys
import glob
import numpy as np
import tifffile
import pyvista as pv
from pathlib import Path

# 配置路径
PROJECT_ROOT = r"d:\Documents\Codes\2026_Kaggle_Vesuvius Challenge 2026 v2 - 副本"
MASK_DIR = os.path.join(PROJECT_ROOT, r"20_src\output\chimera_run_20260214_032021_gpu_nf16")
RAW_DIR = os.path.join(PROJECT_ROOT, r"data\vesuvius-challenge-surface-detection\train_images")

def find_mask_files(mask_dir):
    """查找所有 _mask.tif 文件"""
    search_path = os.path.join(mask_dir, "*_mask.tif")
    files = sorted(glob.glob(search_path))
    return files

def get_raw_path(mask_path):
    """根据 mask 文件名推断原始 CT 路径"""
    filename = os.path.basename(mask_path)
    # 格式: {ID}_mask.tif -> {ID}.tif
    chunk_id = filename.replace("_mask.tif", "")
    raw_path = os.path.join(RAW_DIR, f"{chunk_id}.tif")
    return raw_path, chunk_id

def visualize_chunk(mask_path):
    raw_path, chunk_id = get_raw_path(mask_path)
    
    print(f"\n[{chunk_id}]")
    print(f"  Mask: {mask_path}")
    print(f"  Raw : {raw_path}")

    if not os.path.exists(raw_path):
        print(f"  Error: Raw file not found: {raw_path}")
        return

    # 1. Load Data
    print("  Loading data...")
    vol_raw = tifffile.imread(raw_path)  # (D, H, W)
    vol_mask = tifffile.imread(mask_path) # (D, H, W) 0/1

    # Normalize Raw if needed (uint16 -> uint8 for display or keep as is)
    # PyVista volume rendering works well with float or normalized data
    if vol_raw.dtype == np.uint8:
        pass # OK
    else:
        # Simple normalization for display
        vol_raw = (vol_raw / vol_raw.max() * 255).astype(np.uint8)

    # 2. Create PyVista Grids
    grid_raw = pv.wrap(vol_raw)
    grid_mask = pv.wrap(vol_mask)
    
    # 3. Setup Plotter
    p = pv.Plotter(shape=(1, 2), window_size=(1600, 800), title=f"Hybrid Chimera Result: {chunk_id}")
    
    # --- Left View: Raw CT ---
    p.subplot(0, 0)
    p.add_text(f"Raw CT: {chunk_id}", font_size=12)
    # Opacity for CT (Air is transparent)
    opacity_ct = [0, 0.0, 20, 0.0, 50, 0.3, 255, 0.8] 
    p.add_volume(grid_raw, cmap="gray", opacity=opacity_ct, blending="composite", show_scalar_bar=False)
    p.add_bounding_box()
    p.show_grid()

    # --- Right View: Mask (Red) ---
    p.subplot(0, 1)
    p.add_text("Prediction Mask (Red=Surface)", font_size=12)
    
    # Opacity for Mask (0=Transparent, 1=Opaque Red)
    # Scalar values are 0 and 1
    opacity_mask = [0, 0.0, 0.1, 0.0, 0.9, 0.5, 1.0, 0.5]
    
    # Colormap: 0->Black(Trans), 1->Red
    # We can use a custom cmap
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("binary_red", ["black", "red"])

    p.add_volume(grid_mask, cmap=cmap, opacity=opacity_mask, blending="composite", show_scalar_bar=False)
    p.add_bounding_box()
    p.show_grid()
    
    # Link cameras
    p.link_views()
    
    print("  Visualization Ready! Close window to continue (if in loop).")
    p.show()

def main():
    if not os.path.exists(MASK_DIR):
        print(f"Error: Output directory not found: {MASK_DIR}")
        return

    mask_files = find_mask_files(MASK_DIR)
    if not mask_files:
        print(f"No mask files found in {MASK_DIR}")
        return

    print(f"Found {len(mask_files)} mask files.")
    
    # Visualizing the first one by default, or loop?
    # Let's visualize all sequentially (user can close window to see next)
    for f in mask_files:
        visualize_chunk(f)
        
        # Simple interactive control in console
        resp = input("Show next? [y/n] (default y): ").strip().lower()
        if resp == 'n':
            break

if __name__ == "__main__":
    main()
