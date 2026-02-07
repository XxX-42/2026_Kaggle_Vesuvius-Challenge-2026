print("Importing torch...")
import torch
print("Importing matplotlib...")
import matplotlib.pyplot as plt
print("Importing numpy...")
import numpy as np
import sys
import os
print("Importing pathlib...")
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

print("Importing custom modules...")
from src.models.mini_unetr import MiniUNETR
from src.data.dataset import VesuviusDataset
print("Imports done.")

# 1. 配置
# Update to latest checkpoint from 23:26 run
CHECKPOINT_PATH = "checkpoints/MiniUNETR_20260207_232607/best_model.pth"
DATA_PATH = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Loading checkpoint: {CHECKPOINT_PATH}")

# 2. 加载模型
# Updated config: hidden_size=256 based on 92MB file size
# Must set num_heads=8 to ensure 256 % heads == 0
model = MiniUNETR(
    in_channels=1,
    out_channels=1,
    img_size=256,
    patch_size=16,
    hidden_size=256,  # 92MB checkpoint -> 256
    num_heads=8,      # Fix for ValueError
    feature_size=16
)

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Remove 'module.' prefix if saved with DataParallel
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.to(DEVICE)
model.eval()
print("Model loaded successfully.")

# 3. 加载验证集的一张图
print("Loading dataset...")
val_dataset = VesuviusDataset(data_path=DATA_PATH, mode='train', cache_data=False) 

# Try to find a sample WITH ink for better visualization
print("Searching for a sample with ink...")
sample = None
found_ink = False

# Use positive indices from dataset if available
if hasattr(val_dataset, 'positive_indices') and len(val_dataset.positive_indices) > 0:
    import random
    # Pick a random ink location
    print(f"Found {len(val_dataset.positive_indices)} fragments with ink.")
    
    # Try multiple times to get a good crop
    for _ in range(10):
        # We manually force the logic from __getitem__ to get an ink crop
        # Or just rely on random access hoping for ink? 
        # Actually dataset[i] logic has 50% chance if mode is train.
        # Let's just iterate a few random samples until we find one with label > 0
        idx = random.randint(0, len(val_dataset)-1)
        s = val_dataset[idx]
        if s[1].max() > 0: # Label has 1
            sample = s
            found_ink = True
            print(f"Found sample with ink at index {idx}")
            break

if not found_ink:
    print("Could not quickly find ink sample, using random index 0")
    sample = val_dataset[0]

image, label, ignore_mask = sample
# image: (1, 16, 256, 256)
# label: (1, 256, 256)

# 4. 预测
# Model expects (B, 1, 16, H, W)
input_tensor = image.unsqueeze(0).to(DEVICE) # [1, 1, 16, H, W]

print("Running inference...")
with torch.no_grad():
    output = model(input_tensor)
    pred_prob = torch.sigmoid(output).squeeze().cpu().numpy() # (H, W)

# 5. 可视化 & 分析
print("Analyzing predictions...")
label_np = label.squeeze().numpy()
mid_slice = image[0, 8, :, :].numpy() # Middle slice of Z-stack

# Stats
max_prob = pred_prob.max()
mean_prob = pred_prob.mean()
std_prob = pred_prob.std()

pred_binary = (pred_prob > 0.5).astype(np.float32)
intersection = (pred_binary * label_np).sum()
union = pred_binary.sum() + label_np.sum()
dice = (2.0 * intersection) / (union + 1e-6)
iou = intersection / (pred_binary.sum() + label_np.sum() - intersection + 1e-6)

print(f"\n===== [Analysis Report] =====")
print(f"Max Probability: {max_prob:.4f}")
print(f"Mean Probability: {mean_prob:.4f}")
print(f"Std Dev: {std_prob:.4f}")
print(f"Dice Score: {dice:.4f}")
print(f"IoU Score: {iou:.4f}")
print("=============================\n")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(mid_slice, cmap='gray') 
axs[0].set_title("Input (Middle Slice)")
axs[0].axis('off')

axs[1].imshow(label_np, cmap='gray')
axs[1].set_title("Ground Truth (Label)")
axs[1].axis('off')

im = axs[2].imshow(pred_prob, cmap='jet', vmin=0, vmax=1) 
axs[2].set_title(f"Prediction (Dice: {dice:.2f})")
axs[2].axis('off')

plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("viz_epoch9_output.png")
print("Saved visualization to viz_epoch9_output.png")
# plt.show() # Disabled to return control immediately
