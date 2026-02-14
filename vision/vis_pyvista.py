
import os
import time
import numpy as np
import tifffile
import pyvista as pv

# 文件路径
root = r"d:\Documents\Codes\2026_Kaggle_Vesuvius Challenge 2026 v2 - 副本"
image_path = os.path.join(root, r"data\vesuvius-challenge-surface-detection\train_images\2290837.tif")
label_path = os.path.join(root, r"data\vesuvius-challenge-surface-detection\train_labels_visualized_uint8\2290837.tif")

def visualize():
    print("--- Vesuvius Challenge 3D Visualization (PyVista) ---")
    
    # Check paths
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    if not os.path.exists(label_path):
        print(f"Error: Label file not found: {label_path}")
        print("  (Maybe the conversion script is still running? Please wait a moment)")
        return

    # 1. Load Data
    print(f"Loading Image: {os.path.basename(image_path)} ...")
    img_vol = tifffile.imread(image_path)  # (D, H, W)
    
    print(f"Loading Label: {os.path.basename(label_path)} ...")
    lbl_vol = tifffile.imread(label_path)  # (D, H, W) - values: 0, 100, 255

    # 2. Create PyVista Grids
    # pyvista.wrap() takes numpy array and creates a UniformGrid
    # Note: PyVista expects (X, Y, Z) usually, but wrapping numpy (Z, Y, X) works if we treat Z as depth
    # Keep it simple: direct wrap
    grid_img = pv.wrap(img_vol)
    grid_lbl = pv.wrap(lbl_vol)
    
    # 3. Setup Plotter
    p = pv.Plotter(shape=(1, 2), window_size=(1600, 800))
    
    # --- Left View: CT Image Volume ---
    p.subplot(0, 0)
    p.add_text("CT Scan (Volume Rendering)", font_size=12)
    # Opacity: map low values (air) to transparent, high values (material) to opaque
    # CT usually has dark background and bright material
    opacity_img = [0, 0.0, 0.1, 0.5, 0.8] 
    p.add_volume(grid_img, cmap="gray", opacity=opacity_img, blending="composite", show_scalar_bar=False)
    p.add_bounding_box()

    # --- Right View: Label Volume ---
    p.subplot(0, 1)
    p.add_text("Label (Red=Ink, Gray=Paper)", font_size=12)
    
    # Custom Opacity for Label (0, 100, 255)
    # The list is [x0, opacity0, x1, opacity1, ...] or just a list of opacities mapped linearly
    # PyVista add_volume opacity mapping is tricky. Best way is to map scalar values directly.
    # We want: 0 -> 0.0, 100 -> 0.2, 255 -> 0.3 (Red & Transparent)
    opacity_lbl = [0, 0.0, 10, 0.0, 90, 0.2, 110, 0.2, 250, 0.3, 255, 0.3]

    # Custom Colormap: Transparent-Gray-Red
    import matplotlib.colors as mcolors
    # 0 -> Black/Transparent
    # 100 -> Gray (Paper)
    # 255 -> Red (Ink)
    cmap = mcolors.LinearSegmentedColormap.from_list("bw_custom", ["black", "gray", "red"])
    
    p.add_volume(grid_lbl, cmap=cmap, opacity=opacity_lbl, blending="composite", show_scalar_bar=False)
    p.add_bounding_box()

    # Link cameras
    p.link_views()
    
    print("\nVisualization Ready!")
    print("Controls:")
    print("  - Left Mouse: Rotate")
    print("  - Shift + Left Mouse: Pan")
    print("  - Ctrl + Left Mouse: Rotate")
    print("  - Scroll: Zoom")
    
    p.show()

if __name__ == "__main__":
    visualize()
