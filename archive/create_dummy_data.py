from PIL import Image
import numpy as np
import os
from pathlib import Path

BASE_PATH = Path(r"d:\Documents\Codes\2026_Kaggle_Vesuvius Challenge 2026\data\native\train\1")
BASE_PATH.mkdir(parents=True, exist_ok=True)

# Create a dummy mask.png (usually very large, but for testing we just need it to exist)
# Let's assume 1000x1000 for a dry run if the real one isn't there
mask_path = BASE_PATH / "mask.png"
ink_path = BASE_PATH / "inklabels.png"

if not mask_path.exists():
    print("Creating dummy mask.png...")
    mask = np.ones((1000, 1000), dtype=np.uint8) * 255
    Image.fromarray(mask).save(mask_path)

if not ink_path.exists():
    print("Creating dummy inklabels.png...")
    ink = np.zeros((1000, 1000), dtype=np.uint8)
    ink[400:600, 400:600] = 255 # Some "ink" in the middle
    Image.fromarray(ink).save(ink_path)
