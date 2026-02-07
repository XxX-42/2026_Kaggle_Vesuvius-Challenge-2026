
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

img_path = "output/debug_epoch_2.png"
if not os.path.exists(img_path):
    print(f"❌ Error: {img_path} not found.")
    exit(1)

# Load image
img = Image.open(img_path)
img_arr = np.array(img)
h, w, c = img_arr.shape

print(f"🖼️ Analyzing {img_path}")
print(f"   Size: {w}x{h}, Channels: {c}")

# Split into 3 parts (approximate due to borders/titles)
# The image has 3 subplots horizontally.
# Let's verify the content of the LEFT part (Input).
w_third = w // 3
input_region = img_arr[100:-50, 0:w_third] # Crop margins to avoid title/axes

# Calculate stats
mean_val = input_region.mean()
std_val = input_region.std()
min_val = input_region.min()
max_val = input_region.max()

print("\n🔍 Input Region Analysis (Left Panel):")
print(f"   Mean Intensity: {mean_val:.2f} (0-255)")
print(f"   Contrast (Std): {std_val:.2f}")
print(f"   Range: [{min_val}, {max_val}]")

# Check if black
if mean_val < 5:
    print("\n❌ FAILED: Input is essentially BLACK.")
elif std_val < 2:
    print("\n❌ FAILED: Input is FLAT (no texture).")
else:
    print("\n✅ PASS: Input shows texture/content.")

# Check Middle (Label)
label_region = img_arr[100:-50, w_third:2*w_third]
print("\n🔍 Label Region Analysis (Middle Panel):")
print(f"   Mean Intensity: {label_region.mean():.2f}")
print(f"   Unique Values (approx): {len(np.unique(label_region))}")

