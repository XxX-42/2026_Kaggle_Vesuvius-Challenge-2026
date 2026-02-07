import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedComboLoss(nn.Module):
    """
    Masked Combo Loss (BCE + Dice)
    
    Designed for Class Imbalance (Ink < 1%).
    - BCE: Pixel-wise accuracy.
    - Dice: Overlap metric (Robust to imbalance).
    - Masking: Ignores uncertain regions (ignore.png).
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, use_ignore_mask=True, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_ignore_mask = use_ignore_mask
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, pred, target, ignore_mask=None):
        """
        pred: (B, 1, H, W) logits
        target: (B, 1, H, W) 0/1
        ignore_mask: (B, 1, H, W) 1=Ignore, 0=Valid
        """
        # 1. Calc Weights
        if self.use_ignore_mask and ignore_mask is not None:
            weight = 1.0 - ignore_mask
        else:
            weight = torch.ones_like(pred)
            
        # 2. BCE Loss
        bce_loss = self.bce(pred, target)
        bce_loss = (bce_loss * weight).sum() / (weight.sum() + 1e-6)
        
        # 3. Dice Loss
        # Sigmoid for probabilities
        probs = torch.sigmoid(pred)
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        target_flat = target.view(-1)
        weight_flat = weight.view(-1)
        
        # Apply mask to vectors
        probs_flat = probs_flat * weight_flat
        target_flat = target_flat * weight_flat
        
        intersection = (probs_flat * target_flat).sum()
        union = probs_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice
        
        # 4. Combo
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss

# Alias for compatibility if needed, but we should update train.py
MaskedBCELoss = MaskedComboLoss
