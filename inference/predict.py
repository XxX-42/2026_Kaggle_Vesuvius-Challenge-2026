import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import get_unet, load_checkpoint

CONFIG = {
    "volume_dir": PROJECT_ROOT / "data/native/train/1/surface_volume",
    "model_path": PROJECT_ROOT / "models/vesuvius_native_finetuned.pth",
    "z_slices": list(range(18, 32)),
    "window_min": 18000.0,
    "window_max": 28000.0,
    "tile_size": 224,
    "stride": 112,
    "encoder": "resnet18",
    "output_dir": PROJECT_ROOT / "outputs",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def load_inference_volume():
    print("📦 Loading volume for inference...")
    tif_files = sorted(CONFIG["volume_dir"].glob("*.tif"), key=lambda x: int(x.stem))
    
    selected = [f for f in tif_files if int(f.stem) in CONFIG["z_slices"]]
    
    first = np.array(Image.open(selected[0]))
    h, w = first.shape
    volume = np.zeros((h, w, len(selected)), dtype=np.float32)
    
    for i, f in enumerate(tqdm(selected, desc="Loading")):
        img = np.array(Image.open(f)).astype(np.float32)
        img = np.clip(img, CONFIG["window_min"], CONFIG["window_max"])
        img = (img - CONFIG["window_min"]) / (CONFIG["window_max"] - CONFIG["window_min"])
        volume[:, :, i] = img
        
    return volume

def predict():
    CONFIG["output_dir"].mkdir(exist_ok=True)
    device = torch.device(CONFIG["device"])
    
    # 1. Model
    model = get_unet(in_channels=len(CONFIG["z_slices"])).to(device)
    model = load_checkpoint(model, CONFIG["model_path"], device)
    model.eval()
    
    # 2. Volume
    volume = load_inference_volume()
    h, w, c = volume.shape
    
    # 3. Sliding Window
    pred_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    y_steps = list(range(0, h - CONFIG["tile_size"] + 1, CONFIG["stride"])) + [h - CONFIG["tile_size"]]
    x_steps = list(range(0, w - CONFIG["tile_size"] + 1, CONFIG["stride"])) + [w - CONFIG["tile_size"]]
    y_steps, x_steps = sorted(list(set(y_steps))), sorted(list(set(x_steps)))
    
    with torch.no_grad():
        for y in tqdm(y_steps, desc="Y-Inference"):
            for x in x_steps:
                tile = volume[y:y+CONFIG["tile_size"], x:x+CONFIG["tile_size"], :]
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda'):
                    output = model(tile_tensor)
                    prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                
                pred_sum[y:y+CONFIG["tile_size"], x:x+CONFIG["tile_size"]] += prob
                count_map[y:y+CONFIG["tile_size"], x:x+CONFIG["tile_size"]] += 1
                
    pred_map = np.divide(pred_sum, count_map, where=count_map > 0)
    
    # Save
    out_path = CONFIG["output_dir"] / "inference_full.png"
    cv2.imwrite(str(out_path), (pred_map * 255).astype(np.uint8))
    print(f"✅ Saved prediction to {out_path}")

if __name__ == "__main__":
    predict()
