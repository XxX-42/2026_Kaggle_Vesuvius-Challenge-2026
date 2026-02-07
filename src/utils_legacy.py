import cv2
import numpy as np
import torch
from pathlib import Path

def visualize_prediction(image, mask, pred, ps=224):
    """
    image: (H, W) normalized middle slice
    mask: (H, W) mask
    pred: (H, W) prediction probabilities
    """
    img_vis = (image * 255).astype(np.uint8)
    mask_vis = (mask * 255).astype(np.uint8)
    pred_vis = (pred * 255).astype(np.uint8)
    
    row = np.hstack([img_vis, mask_vis, pred_vis])
    row_bgr = cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(row_bgr, "Input", (10, 20), font, 0.5, (0, 255, 0), 1)
    cv2.putText(row_bgr, "GT", (ps + 10, 20), font, 0.5, (0, 255, 0), 1)
    cv2.putText(row_bgr, "Pred", (2*ps + 10, 20), font, 0.5, (0, 255, 0), 1)
    
    return row_bgr

def seed_everything(seed=42):
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
