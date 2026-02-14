"""
Vesuvius Challenge 2026 - 损失函数
包含 DiceLoss 和 BCELoss 的混合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    Formula: 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask=None):
        """
        Args:
            logits: (B, 1, D, H, W) - Unnormalized logits
            targets: (B, 1, D, H, W) - Binary masks (0 or 1)
            valid_mask: (B, 1, D, H, W) - 1 for valid pixels, 0 for ignore
        """
        probs = torch.sigmoid(logits)
        
        if valid_mask is not None:
            # Mask out invalid regions
            probs = probs * valid_mask
            targets = targets * valid_mask
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


from src.cldice import soft_cldice


class DynamicWeightedBCELoss(nn.Module):
    """
    Dynamic Weighted Cross Entropy Loss
    
    参考文献: Dynamic Weighted Cross Entropy (MICCAI 2025)
    
    在每个 Batch 中根据正负样本比例动态计算 pos_weight，
    解决极度不平衡数据集中的梯度消失问题。
    
    公式: pos_weight = beta * N_total / (N_positive + epsilon)
    """
    def __init__(self, beta=1.0, epsilon=1e-6):
        super(DynamicWeightedBCELoss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon
        
    def forward(self, logits, targets, valid_mask=None):
        """
        Args:
            logits: (B, 1, D, H, W) - 未归一化的 logits
            targets: (B, 1, D, H, W) - 二值掩码 (0 或 1)
            valid_mask: (B, 1, D, H, W) - 1 for valid, 0 for ignore
        """
        # 动态计算正样本权重 (仅考虑有效区域)
        if valid_mask is not None:
             # 对于 weight calculation，只统计 valid 中的 pos/total
             # 但 BCEWithLogitsLoss 的 reduction 需要小心
             masked_targets = targets * valid_mask
             n_total = valid_mask.sum().clamp(min=1)
             n_positive = masked_targets.sum().clamp(min=1)
        else:
             n_total = targets.numel()
             n_positive = targets.sum().clamp(min=1)
        
        # 动态权重公式
        pos_weight = (self.beta * n_total) / (n_positive + self.epsilon)
        
        # 使用动态权重计算 BCE
        # 如果有 mask，需要使用 reduction='none' 然后手动 mean
        if valid_mask is not None:
            bce_loss_raw = F.binary_cross_entropy_with_logits(
                logits, targets, 
                pos_weight=pos_weight.detach(),
                reduction='none'
            )
            # Apply mask
            bce_loss = (bce_loss_raw * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, 
                pos_weight=pos_weight.detach()
            )
        
        return bce_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation
    
    针对厚度过大问题（高假阳性）设计。
    通过调整 alpha 和 beta 参数，明确控制对 FP 和 FN 的惩罚力度。
    
    公式: 1 - (TP + ε) / (TP + α*FP + β*FN + ε)
    
    参数说明:
    - alpha > beta: 惩罚假阳性（减少厚度/膨胀）
    - alpha < beta: 惩罚假阴性（提高召回率）
    - alpha = beta = 0.5: 等同于 Dice Loss
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # FP 权重 (假阳性/厚度惩罚)
        self.beta = beta    # FN 权重 (假阴性/召回惩罚)
        self.smooth = smooth
        
    def forward(self, logits, targets, valid_mask=None):
        """
        Args:
            logits: (B, 1, D, H, W) - 未归一化的 logits
            targets: (B, 1, D, H, W) - 二值掩码 (0 或 1)
        """
        probs = torch.sigmoid(logits)
        
        # Masking if provided
        if valid_mask is not None:
            probs = probs * valid_mask
            targets = targets * valid_mask
            # Note: FP calculation needs care. 
            # FP = probs * (1-targets). If pixel is ignored (mask=0), probs becomes 0, so FP=0 correctly.
            # FN = (1-probs) * targets. If pixel is ignored, targets=0, so FN=0 correctly.
            # But wait: (1-probs) where probs=0 is 1. 
            # However targets=0 helps.
            # Let's verify:
            # Mask=0 -> probs=0, targets=0.
            # TP = 0*0 = 0.
            # FP = 0*(1-0) = 0.
            # FN = (1-0)*0 = 0.
            # This logic works because we zero out both inputs.
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # 计算 TP, FP, FN
        TP = (probs_flat * targets_flat).sum()
        FP = (probs_flat * (1 - targets_flat)).sum()
        FN = ((1 - probs_flat) * targets_flat).sum()
        
        # Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1.0 - tversky_index


class CombinedLoss(nn.Module):
    """
    Combined Loss: alpha * BCE + beta * Tversky + gamma * clDice
    
    升级版 v2：引入 Tversky Loss 替代 Dice Loss，
    明确控制假阳性（厚度）和假阴性（召回）的平衡。
    
    默认配置 (瘦身策略):
    - alpha = 0.3 (BCE)
    - beta = 0.5 (Tversky, 主导)
    - gamma = 0.2 (clDice, 结构连续性)
    - tversky_alpha = 0.7 (惩罚 FP/厚度)
    - tversky_beta = 0.3 (惩罚 FN)
    """
    def __init__(self, alpha=0.3, beta=0.5, gamma=0.2, 
                 dynamic_beta=1.0, tversky_alpha=0.7, tversky_beta=0.3, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # BCE 权重
        self.beta = beta    # Tversky 权重
        self.gamma = gamma  # clDice 权重
        
        # [P0 FIX] 标准 BCE，不再使用 DynamicWeighted 的 pos_weight 放大
        # 在正样本 ~50% 的数据上，Dynamic pos_weight ≈ 2.0 会导致全正预测崩溃
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Tversky Loss (替代 Dice，用于控制厚度)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=smooth)
        
    def forward(self, logits, targets, valid_mask=None):
        # 1. Dynamic BCE Loss
        if valid_mask is not None:
            bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            bce = (bce_raw * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        else:
            bce = self.bce_loss(logits, targets)
        
        # 2. Tversky Loss (替代 Dice)
        tversky = self.tversky_loss(logits, targets, valid_mask=valid_mask)
        
        # 3. clDice Loss (Optional)
        cldice = torch.tensor(0.0, device=logits.device)
        if self.gamma > 0:
            probs = torch.sigmoid(logits)
            if valid_mask is not None:
                probs = probs * valid_mask
                # clDice logic is complex with soft skeleton.
                # Simply zeroing out might break topology? 
                # If we zero out connection, it splits topology.
                # However, for 'ignore' regions, we probably don't want to enforce connectivity through them.
                # So zeroing out seems safer than computing gradients on garbage.
            
            # soft_cldice returns the score (higher is better), so loss is 1 - score
            cldice_score = soft_cldice(probs, targets)
            cldice = 1.0 - cldice_score
            
        total_loss = self.alpha * bce + self.beta * tversky + self.gamma * cldice
        
        return total_loss, bce, tversky, cldice



class CombinedLossWithAffinity(nn.Module):
    """
    带 Affinity 监督的组合损失
    
    Total Loss = seg_loss + affinity_weight * affinity_loss
    
    其中:
    - seg_loss = CombinedLoss(分割输出, 分割标签)
    - affinity_loss = BCE(Affinity输出, Affinity标签)
    """
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2, affinity_weight=0.5, 
                 dynamic_beta=1.0, tversky_alpha=0.7, tversky_beta=0.3, smooth=1e-6):
        super(CombinedLossWithAffinity, self).__init__()
        self.affinity_weight = affinity_weight
        self.seg_loss = CombinedLoss(
            alpha=alpha, beta=beta, gamma=gamma, 
            dynamic_beta=dynamic_beta,
            tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
            smooth=smooth
        )
        self.affinity_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, seg_logits, affinity_logits, seg_targets, affinity_targets, valid_mask=None):
        """
        Args:
            seg_logits: (B, 1, D, H, W) 分割输出
            affinity_logits: (B, 3, D, H, W) Affinity 输出
            seg_targets: (B, 1, D, H, W) 分割标签
            affinity_targets: (B, 3, D, H, W) Affinity 标签
            valid_mask: (B, 1, D, H, W)
        """
        # 分割损失
        total_seg, bce, dice, cldice = self.seg_loss(seg_logits, seg_targets, valid_mask=valid_mask)
        
        # Affinity 损失
        # Affinity valid mask needs to support 3 channels?
        # Typically if a pixel is ignored, its affinity to neighbors is also ignored?
        # Yes. We can expand valid_mask to 3 channels.
        if valid_mask is not None:
            aff_mask = valid_mask.expand_as(affinity_logits)
            aff_loss_raw = F.binary_cross_entropy_with_logits(affinity_logits, affinity_targets, reduction='none')
            aff_loss = (aff_loss_raw * aff_mask).sum() / aff_mask.sum().clamp(min=1)
        else:
            aff_loss = self.affinity_loss(affinity_logits, affinity_targets)
        
        # 总损失
        total_loss = total_seg + self.affinity_weight * aff_loss
        
        return total_loss, bce, dice, cldice, aff_loss



if __name__ == "__main__":
    # 简单的损失函数测试
    print("测试 CombinedLoss...")
    
    loss_fn = CombinedLoss(alpha=0.5)
    
    # 模拟输入
    logits = torch.randn(2, 1, 64, 128, 128)
    targets = torch.randint(0, 2, (2, 1, 64, 128, 128)).float()
    
    try:
        loss = loss_fn(logits, targets)
        print(f"Loss Value: {loss[0].item():.6f}")
        print("✓ 损失函数测试通过")
    except Exception as e:
        print(f"损失函数运行出错: {e}")

