import os
import glob
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configuration
DATA_DIR = r"d:\Documents\Codes\2026_Kaggle_Vesuvius Challenge 2026 v2 - 副本\data\vesuvius-challenge-surface-detection"
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_LABELS_DIR = os.path.join(DATA_DIR, "train_labels")
OUTPUT_PATH = r"d:\Documents\Codes\2026_Kaggle_Vesuvius Challenge 2026 v2 - 副本\all_train_data_visualization.png"

# Visualization Parameters
GRID_COLS = 28  # Number of pairs per row
# GRID_ROWS will be calculated dynamically
SLICE_DEPTH_RATIO = 0.5  # Take the middle slice (0.5)

def normalize_image(img_slice):
    """Normalize image slice to 0-255 uint8."""
    img_min = img_slice.min()
    img_max = img_slice.max()
    if img_max > img_min:
        img_norm = (img_slice - img_min) / (img_max - img_min) * 255.0
    else:
        img_norm = img_slice * 0
    return img_norm.astype(np.uint8)

def create_visualization():
    # Get file list
    image_files = sorted(glob.glob(os.path.join(TRAIN_IMAGES_DIR, "*.tif")))
    label_files = sorted(glob.glob(os.path.join(TRAIN_LABELS_DIR, "*.tif")))
    
    if not image_files:
        print("No training images found!")
        return

    # Filter to common files (just in case)
    image_basenames = {os.path.basename(f) for f in image_files}
    label_basenames = {os.path.basename(f) for f in label_files}
    common_basenames = sorted(list(image_basenames.intersection(label_basenames)))
    
    total_images = len(common_basenames)
    print(f"Found {total_images} matching images/labels.")
    
    if total_images == 0:
        return

    # Determine layout
    grid_rows = (total_images + GRID_COLS - 1) // GRID_COLS
    
    # Get dimensions from the first image
    first_img_path = os.path.join(TRAIN_IMAGES_DIR, common_basenames[0])
    first_img = tifffile.imread(first_img_path)
    # Shape is likely (D, H, W) or (H, W, D). Based on previous check: (320, 320, 320)
    # Usually Z, Y, X.
    # We will take slice at Z = Shape[0] // 2
    # Resulting slice is (H, W) -> (320, 320)
    
    slice_h, slice_w = first_img.shape[1], first_img.shape[2]
    
    # Dimensions for one cell (Image | Label)
    cell_w = slice_w * 2
    cell_h = slice_h
    
    # Total canvas size
    canvas_w = GRID_COLS * cell_w
    canvas_h = grid_rows * cell_h
    
    print(f"Canvas Size: {canvas_w}x{canvas_h}")
    
    # Create canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    
    for idx, basename in enumerate(tqdm(common_basenames, desc="Processing images")):
        # Load image
        img_path = os.path.join(TRAIN_IMAGES_DIR, basename)
        lbl_path = os.path.join(TRAIN_LABELS_DIR, basename)
        
        try:
            # Load specific slice to save memory? 
            # tifffile.imread loads whole file. These are small enough (35MB).
            img_vol = tifffile.imread(img_path)
            lbl_vol = tifffile.imread(lbl_path)
            
            # Select middle slice
            z_idx = int(img_vol.shape[0] * SLICE_DEPTH_RATIO)
            img_slice = img_vol[z_idx]
            lbl_slice = lbl_vol[z_idx]
            
            # Normalize Image
            img_vis = normalize_image(img_slice)
            img_pil = Image.fromarray(img_vis, mode='L').convert("RGB")
            
            # Prepare Label (0 or 1 -> 0 or 255)
            lbl_vis = (lbl_slice * 255).astype(np.uint8)
            lbl_pil = Image.fromarray(lbl_vis, mode='L').convert("RGB")
            
            # Draw Label overlay or side-by-side? user said "side-by-side" (一对一并排)
            # Combine
            pair_img = Image.new("RGB", (cell_w, cell_h))
            pair_img.paste(img_pil, (0, 0))
            pair_img.paste(lbl_pil, (slice_w, 0))
            
            # Compute position
            row = idx // GRID_COLS
            col = idx % GRID_COLS
            x = col * cell_w
            y = row * cell_h
            
            canvas.paste(pair_img, (x, y))
            
            # Optional: Add text
            # draw = ImageDraw.Draw(canvas)
            # draw.text((x, y), basename, fill="red")
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            continue

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    canvas.save(OUTPUT_PATH)
    print("Done!")

if __name__ == "__main__":
    create_visualization()
