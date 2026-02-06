import torch
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Import components from training script
try:
    from train_vesuvius import Thinking25DNet, Vesuvius25DDataset, CONFIG
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
except ImportError:
    print("❌ Error: Could not import from train_vesuvius.py. Make sure it's in the same directory.")
    sys.exit(1)

def audit_model():
    print("🔍 Initializing Audit Protocol...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # 1. Load Model
    model_path = "vesuvius_best.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found. Has training started/saved yet?")
        sys.exit(1)
        
    print(f"   Loading model from {model_path}...")
    model = Thinking25DNet(in_channels=CONFIG["z_depth"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Setup Data
    print("   Setting up Validation Data...")
    val_transform = A.Compose([
        ToTensorV2(transpose_mask=True),
    ])
    
    # Force single batch for audit
    val_dataset = Vesuvius25DDataset(CONFIG["data_root"], z_slices=CONFIG["z_depth"], transform=val_transform, mode="valid")
    # Use shuffle=True to find ink faster if start is empty
    loader = DataLoader(val_dataset, batch_size=4, num_workers=0, shuffle=True) 
    
    print("\n🧪 Searching for ink-containing samples...")
    
    found_ink = False
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for step, (images, masks) in enumerate(tqdm(loader)):
                images = images.to(device)
                masks = masks.to(device)
                
                # Check if batch has ink
                if masks.sum() == 0:
                    continue
                
                found_ink = True
                print(f"   ✅ Found ink in batch {step}!")
                
                # 3. Inference
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                
                # 4. Signal-to-Noise Analysis
                # Flatten everything to analyze pixels
                probs = preds.cpu().numpy().flatten()
                targets = masks.cpu().numpy().flatten()
                
                ink_pixels = probs[targets == 1]
                bg_pixels = probs[targets == 0]
                
                if len(ink_pixels) == 0:
                    print("   ⚠️ Warning: Batch sum > 0 but no ink pixels found in flatten? Should not happen.")
                    continue
                
                mean_ink = np.mean(ink_pixels)
                mean_bg = np.mean(bg_pixels)
                
                # Avoid division by zero
                contrast = mean_ink / (mean_bg + 1e-9)
                
                # Report
                report = []
                report.append("="*40)
                report.append("📊 审计报告 (Audit Report)")
                report.append("="*40)
                report.append(f"   - 📦 样本数 (Pixels): {len(probs)}")
                report.append(f"   - ✒️ 墨水像素数: {len(ink_pixels)}")
                report.append(f"   - ⬜ 背景像素数: {len(bg_pixels)}")
                report.append("-" * 40)
                report.append(f"   - 🟢 墨水区平均置信度 (Mean Ink Prob)  : {mean_ink:.6f}")
                report.append(f"   - 🔴 背景区平均置信度 (Mean BG Prob)   : {mean_bg:.6f}")
                report.append("-" * 40)
                report.append(f"   - 🚀 对比度 (Signal/Noise Ratio)      : {contrast:.2f} 倍")
                report.append("="*40)
                
                if contrast > 1.5:
                    report.append("✅ 结论: 模型已学会区分墨水与背景！")
                elif contrast > 1.0:
                    report.append("⚠️ 结论: 信号极其微弱，仅略高于随机猜测。")
                else:
                    report.append("❌ 结论: 模型尚未区分（或产生了反向预测）。")
                
                report_str = "\n".join(report)
                print(report_str)
                with open("audit_result.log", "w", encoding="utf-8") as f:
                    f.write(report_str)
                
                break # Ensure we only do one batch
    
    if not found_ink:
        print("\n❌ Failed: Scanned validation set but found NO ink pixels. Is the label loading correct?")

if __name__ == "__main__":
    audit_model()
