import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction='none'): # 'none' to handle weighting
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class HallucinationKillerLoss(nn.Module):
    """
    Suppresses 'High Score Hallucinations' on background patches.
    Strategy:
    1. Focal Loss (Hard Negative Mining).
    2. Dice Loss (Structural).
    3. Empty Patch Penalty: If GT is empty, multiply loss by 1.5.
    """
    def __init__(self, focal_weight=0.7, dice_weight=0.3, alpha=0.8, gamma=2, empty_weight=1.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.empty_weight = empty_weight
        self.smooth = 1e-6
        
    def forward(self, inputs, targets):
        # inputs: (B, 1, H, W) logits
        # targets: (B, 1, H, W) binary
        
        # 1. Focal Loss (Pixel-wise)
        focal_pixel = self.focal(inputs, targets) # (B, 1, H, W)
        
        # 2. Compute weighting per sample based on whether it's empty
        # Check if patch is empty (sum of target < 1 pixel essentially)
        # targets shape: (B, C, H, W)
        patch_sums = targets.view(targets.size(0), -1).sum(dim=1) # (B,)
        
        # Weight vector: 1.5 if empty, 1.0 if not
        # We need to broadcast this to (B, 1, H, W)
        sample_weights = torch.ones_like(patch_sums)
        sample_weights[patch_sums == 0] = self.empty_weight
        
        # Reshape for broadcasting
        # (B,) -> (B, 1, 1, 1)
        sample_weights = sample_weights.view(-1, 1, 1, 1)
        
        # Weighted Focal Mean
        # Apply sample weights to pixel losses
        weighted_focal = focal_pixel * sample_weights
        focal_term = weighted_focal.mean()
        
        # 3. Dice Loss (Batch-wise or Sample-wise?)
        # Standard Dice is usually batch-wise or sample-wise mean
        # We'll do sample-wise to apply weights? 
        # For simplicity/stability, standard Dice on batch is often fine, 
        # but let's do sample-mean to be consistent with weighting strategy.
        
        probs = torch.sigmoid(inputs)
        
        # Flatten per sample: (B, -1)
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.view(targets.size(0), -1)
        
        intersection = (p_flat * t_flat).sum(dim=1)
        union = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        
        # Apply same weights to Dice? Yes, punish empty patches harder if they fail dice (though empty patch dice is weird)
        # Verify: If target empty, intersection=0. Dice = 0/(union). 
        # If model predicts 0, union=0. Dice=1. Loss=0. Good.
        # If model predicts stuff, union>0. Dice=0. Loss=1. Bad.
        # So standard Dice already handles empty patches logic (1 vs 0), but we add extra weight.
        
        weighted_dice = dice_loss * sample_weights.view(-1)
        dice_term = weighted_dice.mean()
        
        return self.focal_weight * focal_term + self.dice_weight * dice_term
