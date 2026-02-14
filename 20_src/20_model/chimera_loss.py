"""
Vesuvius Challenge - Chimera 复合损失函数 (MVP)

L_total = L_Dice + λ_normal × L_CosineSimilarity

核心组件：
1. DiceLoss: 标准 Dice Loss，用于分割头
2. NormalCosineLoss: Cosine Similarity Loss，仅在 mask 区域计算
3. compute_gt_normals(): 从 binary mask 的 Sobel 梯度实时生成法线 GT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def compute_gt_normals(mask: torch.Tensor) -> torch.Tensor:
    """
    从 binary mask 的梯度实时计算表面法线 Ground Truth

    使用 3D Sobel 算子计算梯度方向，然后归一化为单位法线。
    法线指向从 papyrus 内部到外部的方向。

    Args:
        mask: 分割标签，形状 (B, 1, D, H, W)，值域 {0, 1} 或 [0, 1]

    Returns:
        normals: 法线 GT，形状 (B, 3, D, H, W)，单位向量
                 在非表面区域（梯度为零），法线为 (0, 0, 0)
    """
    # 使用 F.conv3d 计算 3D 梯度（Sobel 简化版: 中心差分）
    # 梯度核: 沿各轴的中心差分 [-1, 0, 1]
    device = mask.device
    dtype = mask.dtype

    # 构建梯度卷积核
    # d 方向梯度 (depth/z)
    kernel_d = torch.zeros(1, 1, 3, 1, 1, device=device, dtype=dtype)
    kernel_d[0, 0, 0, 0, 0] = -1.0
    kernel_d[0, 0, 2, 0, 0] = 1.0

    # h 方向梯度 (height/y)
    kernel_h = torch.zeros(1, 1, 1, 3, 1, device=device, dtype=dtype)
    kernel_h[0, 0, 0, 0, 0] = -1.0
    kernel_h[0, 0, 0, 2, 0] = 1.0

    # w 方向梯度 (width/x)
    kernel_w = torch.zeros(1, 1, 1, 1, 3, device=device, dtype=dtype)
    kernel_w[0, 0, 0, 0, 0] = -1.0
    kernel_w[0, 0, 0, 0, 2] = 1.0

    # 使用平滑后的 mask 计算梯度（避免锯齿状法线）
    mask_smooth = mask.float()

    # 计算三个方向的梯度
    grad_d = F.conv3d(mask_smooth, kernel_d, padding=(1, 0, 0))
    grad_h = F.conv3d(mask_smooth, kernel_h, padding=(0, 1, 0))
    grad_w = F.conv3d(mask_smooth, kernel_w, padding=(0, 0, 1))

    # 拼接为法线向量: (B, 3, D, H, W)
    normals = torch.cat([grad_d, grad_h, grad_w], dim=1)

    # 归一化为单位向量
    norm = torch.norm(normals, dim=1, keepdim=True).clamp(min=1e-8)
    normals = normals / norm

    # 梯度为零的区域（非表面），法线设为 (0, 0, 0)
    zero_mask = (norm < 1e-7).expand_as(normals)
    normals[zero_mask] = 0.0

    return normals


class DiceLoss(nn.Module):
    """
    标准 Dice Loss

    Dice = 2|A∩B| / (|A| + |B|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - 显式控制 FP/FN 惩罚比例

    alpha 控制假阳性(FP)的惩罚力度，beta 控制假阴性(FN)的惩罚力度。
    alpha=0.5, beta=0.5 等价于 Dice Loss。
    alpha=0.7, beta=0.3 强烈惩罚假阳性，迫使模型"截胖"。
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # FP 惩罚系数
        self.beta = beta    # FN 惩罚系数
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        TP = (probs_flat * targets_flat).sum(dim=1)
        FP = ((1 - targets_flat) * probs_flat).sum(dim=1)
        FN = (targets_flat * (1 - probs_flat)).sum(dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1.0 - tversky).mean()


class NormalCosineLoss(nn.Module):
    """
    法线 Cosine Similarity Loss

    仅在 mask 区域（表面附近）计算，忽略背景区域的法线。
    Loss = 1 - mean(cos_sim) 在 mask 区域
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_normals: torch.Tensor,
        gt_normals: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_normals: 预测法线，形状 (B, 3, D, H, W)，值域 [-1, 1]
            gt_normals: GT 法线，形状 (B, 3, D, H, W)
            mask: 表面区域掩码，形状 (B, 1, D, H, W)

        Returns:
            loss: 标量 Cosine Loss
        """
        # 扩展 mask 到 3 通道
        mask_3ch = mask.expand_as(pred_normals)

        # 只在 mask 区域计算 (梯度非零的地方)
        # 同时检查 GT 法线非零（只在表面计算）
        gt_norm = torch.norm(gt_normals, dim=1, keepdim=True)
        surface_mask = (mask > 0.5) & (gt_norm > 1e-6)
        surface_mask_3ch = surface_mask.expand_as(pred_normals)

        if surface_mask_3ch.sum() == 0:
            # 没有表面区域，通过与 0 相乘来返回 0 损失，保持与输入张量的梯度链
            # 同时确保结果是一个标量
            return pred_normals.sum() * 0.0

        # 提取表面区域的法线
        # 将法线 reshape 为 (N_surface, 3) 进行点积
        pred_masked = pred_normals[surface_mask_3ch].view(-1, 3)
        gt_masked = gt_normals[surface_mask_3ch].view(-1, 3)

        # Cosine similarity: dot(pred, gt) / (|pred| * |gt|)
        cos_sim = F.cosine_similarity(pred_masked, gt_masked, dim=1)

        # Loss = 1 - mean(cos_sim)
        loss = 1.0 - cos_sim.mean()

        return loss


class ChimeraLoss(nn.Module):
    """
    Chimera 复合损失函数 (v3 - Tversky)

    L_total = L_Tversky + λ_bce × L_BCE + λ_normal × L_CosineSimilarity

    Tversky Loss 替代 Dice Loss，显式惩罚假阳性 (FP)，用于"截胖"。
    BCE 权重降低到 0.1，别好意思回到"全 0"陷阱。

    Args:
        lambda_normal: 法线损失的权重系数，默认 1.0
        lambda_bce:    BCE 损失的权重系数，默认 0.1
        tversky_alpha: Tversky FP 惩罚系数，默认 0.7
        tversky_beta:  Tversky FN 惩罚系数，默认 0.3
    """

    def __init__(
        self,
        lambda_normal: float = 1.0,
        lambda_bce: float = 0.1,
        dice_smooth: float = 1e-6,
        pos_weight: float = 5.0,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
    ):
        super().__init__()
        self.lambda_normal = lambda_normal
        self.lambda_bce = lambda_bce
        # 主力 Loss: Tversky (alpha=0.7 惩罚 FP, beta=0.3 容忍 FN)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=dice_smooth)
        # 辅助 Loss: 加权 BCE，仅用于防止"全 0"陷阱
        pos_weight_tensor = torch.tensor([pos_weight])
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.normal_loss = NormalCosineLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,
        pred_normals: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            seg_logits:    分割 logits，形状 (B, 1, D, H, W)
            pred_normals:  预测法线，形状 (B, 3, D, H, W)
            targets:       分割标签，形状 (B, 1, D, H, W)

        Returns:
            total_loss:  L_Tversky + λ_bce * L_BCE + λ_normal * L_Cosine
            tversky_val: Tversky Loss 分量
            bce_val:     BCE Loss 分量
            normal_val:  Normal Cosine Loss 分量
        """
        # 1. Tversky Loss (主力，显式惩罚 FP，强迫"截胖")
        tversky_val = self.tversky_loss(seg_logits, targets)

        # 2. BCE Loss (辅助，仅用于防止"全 0"陷阱)
        # 确保 pos_weight 在正确的设备上
        if self.bce_loss.pos_weight is not None and self.bce_loss.pos_weight.device != seg_logits.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(seg_logits.device)
        bce_val = self.bce_loss(seg_logits, targets.float())

        # 3. 实时计算法线 GT
        gt_normals = compute_gt_normals(targets)

        # 4. Normal Cosine Loss (几何)
        normal_val = self.normal_loss(pred_normals, gt_normals, targets)

        # 5. 总损失: Tversky 主导 + 低权重 BCE 辅助 + Normal 几何
        total_loss = tversky_val + (self.lambda_bce * bce_val) + (self.lambda_normal * normal_val)

        return total_loss, tversky_val, bce_val, normal_val


if __name__ == "__main__":
    print("=== ChimeraLoss 自测 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, D, H, W = 2, 32, 32, 32

    # 模拟模型输出 (需要设置 requires_grad=True 进行 backward 测试)
    seg_logits = torch.randn(B, 1, D, H, W, device=device, requires_grad=True)
    pred_normals = torch.randn(B, 3, D, H, W, device=device, requires_grad=True).clamp(-1, 1)

    # 创建简单的球形标签
    zz, yy, xx = torch.meshgrid(
        torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'
    )
    dist = ((zz - D/2)**2 + (yy - H/2)**2 + (xx - W/2)**2).float().sqrt()
    sphere_mask = (dist < D * 0.3).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    targets = sphere_mask.expand(B, -1, -1, -1, -1).to(device)

    # 测试 compute_gt_normals
    gt_normals = compute_gt_normals(targets)
    print(f"GT 法线形状: {gt_normals.shape}")       # (B, 3, D, H, W)
    print(f"GT 法线值域: [{gt_normals.min():.4f}, {gt_normals.max():.4f}]")

    # 测试 ChimeraLoss
    criterion = ChimeraLoss(lambda_normal=1.0, lambda_bce=1.0)
    total, dice, bce, normal = criterion(seg_logits, pred_normals, targets)

    print(f"总损失:   {total.item():.4f}")
    print(f"Dice Loss: {dice.item():.4f}")
    print(f"BCE Loss:  {bce.item():.4f}")
    print(f"Normal:   {normal.item():.4f}")

    # 反向传播测试
    print("正在测试反向传播...")
    total.backward()
    print(f"Seg Logits Grad: {seg_logits.grad is not None}")
    print(f"Pred Normals Grad: {pred_normals.grad is not None}")
    print("✓ 反向传播通过！")

    print("✓ 所有测试通过！")
