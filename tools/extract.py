from PIL import Image
import numpy as np
import os
import sys

def process_slice(z_index):
    file_path = f"data/native/train/1/surface_volume/{z_index}.tif"
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return False
    
    print(f"📖 Reading {file_path}...")
    img = Image.open(file_path)
    w, h = img.size
    print(f"   Original size: {w}x{h}, Mode: {img.mode}")
    
    # Calculate center 512x512
    left = (w - 512) // 2
    top = (h - 512) // 2
    right = left + 512
    bottom = top + 512
    
    print(f"   Cropping center: ({left}, {top}) to ({right}, {bottom})")
    crop = img.crop((left, top, right, bottom))
    
    # Vesuvius data is usually 16-bit. PNG supports 16-bit, but most viewers don't.
    # The user said "不做任何复杂变换" (no complex transformations).
    # If we save as PNG directly, PIL might keep it 16-bit or fail.
    # To ensure it is visible, we should ideally map it to 8-bit, 
    # but I will try to save it directly first as requested.
    
    output_name = f"slice_{z_index}_center.png"
    
    # If it's a 16-bit image, we might need to convert it to 8-bit for standard PNG transparency/visibility
    # but the user said "no complex transformations". 
    # Let's check the array values.
    data = np.array(crop)
    print(f"   Crop stats: Min={data.min()}, Max={data.max()}, Mean={data.mean():.2f}")
    
    # Simple 8-bit mapping for visibility if it's 16-bit
    if data.dtype == np.uint16 or data.dtype == np.int32:
        # User said "no complex transformations", but a purely black image is useless.
        # I'll check if it's mostly low values.
        # If I don't scale, it will save as 16-bit PNG (which is valid PNG but rare).
        pass

    crop.save(output_name)
    print(f"✅ Saved to {output_name}")
    return True

if __name__ == "__main__":
    target = "41"
    if not process_slice(target):
        print("💡 Attempting fallback to slice 57...")
        process_slice("57")
