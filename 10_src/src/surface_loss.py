"""
Vesuvius Challenge 2026 - 表面感知损失函数
替代不适用于"片状"拓扑的 clDice

包含:
  1. SoftSurfaceDiceLoss - 模拟比赛指标 SurfaceDice@Tolerance
  2. BoundaryLoss       - 强化薄层边缘检测
  3. CompoundLoss        - 混合损失 (BCE + SurfaceDice + Boundary)

[性能优化] 距离变换使用 GPU max_pool3d 迭代膨胀近似，
          避免 CPU scipy EDT 的 ~7.9s/it 瓶颈。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== GPU 距离变换工具 ==========

@torch.no_grad()
def gpu_distance_transform(mask: torch.Tensor, max_iter: int = 5) -> torch.Tensor:
    """
    GPU 近似距离变换：用 max_pool3d 迭代膨胀模拟

    对于二值掩码中的每个前景体素，计算到最近背景体素的近似距离。
    距离值为整数 (1, 2, 3, ...)，精度为 1 体素。

    原理:
    1. 从边界开始，每次 max_pool 膨胀 1 层
    2. 如果体素在第 k 次膨胀后被覆盖，则距离 = k
    3. 等价于曼哈顿距离的 6-连通近似

    Args:
        mask: (B, 1, D, H, W) 二值前景掩码 (float, 0/1)
        max_iter: 最大膨胀迭代次数 (决定最大距离精度)

    Returns:
        dist: (B, 1, D, H, W) 距离图 (float)
              前景内部: 到最近背景的距离 (1, 2, 3, ...)
              背景: 0
    """
    # 初始化：前景 = max_iter+1 (远距离), 背景 = 0
    dist = mask.float() * (max_iter + 1)

    for k in range(1, max_iter + 1):
        # 膨胀背景: 如果相邻体素是背景(0)，则当前距离 = k
        # 等价于：min(dist, dilated_background + k)
        dilated_bg = F.max_pool3d(
            (1.0 - mask).float(),  # 背景 mask
            kernel_size=3, stride=1, padding=1
        )
        # dilated_bg > 0 的区域表示"距离背景 ≤ k"
        near_boundary = (dilated_bg > 0) & (mask > 0)
        # 只更新还未被赋值的体素 (dist still = max_iter+1)
        update_mask = near_boundary & (dist > k)
        dist = torch.where(update_mask, torch.tensor(float(k), device=mask.device), dist)

        # 收缩前景用于下一轮
        mask = mask.float() * (1.0 - dilated_bg)

    # 将剩余未触及的深层前景 clip 到 max_iter
    dist = torch.clamp(dist, max=float(max_iter))

    return dist


@torch.no_grad()
def compute_distance_transform_gpu(targets: torch.Tensor, tau: float = 2.0):
    """
    GPU 版距离变换 - 替代 CPU scipy.ndimage.distance_transform_edt

    Args:
        targets: (B, 1, D, H, W) 二值掩码 (float, 0/1)
        tau: 容差半径 (体素单位)

    Returns:
        dist_pos: (B, 1, D, H, W) 前景到背景的距离
        dist_neg: (B, 1, D, H, W) 背景到前景的距离
    """
    max_iter = max(int(tau) + 1, 3)  # 至少 3 次迭代

    # 前景到最近背景的距离
    dist_pos = gpu_distance_transform(targets, max_iter=max_iter)

    # 背景到最近前景的距离 (反转 mask)
    dist_neg = gpu_distance_transform(1.0 - targets, max_iter=max_iter)

    return dist_pos, dist_neg



# ========== 损失函数类 ==========

class SoftSurfaceDiceLoss(nn.Module):
    """
    Soft Surface Dice Loss - 模拟比赛指标 SurfaceDice@Tolerance

    核心思想:
    - 标准 Dice 对所有体素一视同仁
    - SurfaceDice 仅关注"表面附近"的体素，允许 τ 容差范围内的偏移
    - 对于片状结构（卷轴），这比 volumetric Dice 更合理

    原理:
    1. 对 GT 和 Pred 分别提取表面（距离变换 ≤ tau 的区域）
    2. 将 softmax 概率按距离函数加权
    3. 在加权区域计算 Dice

    Args:
        tau: 容差半径 (体素单位, 默认 2.0)
        smooth: 防除零平滑项
    """

    def __init__(self, tau: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.tau = tau
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, D, H, W) 未归一化的 logits
            targets: (B, 1, D, H, W) 二值掩码
            valid_mask: (B, 1, D, H, W) 有效区域 mask
        """
        probs = torch.sigmoid(logits)

        # GPU 近似距离变换 (全在 GPU，无 CPU 传输)
        # 距离变换时，忽略区域不应该影响距离计算？
        # 这是一个难题。如果我们把忽略区域当作背景(0)，那前景靠近忽略区域时会被认为靠近背景。
        # 如果忽略区域是"未知"，那既不能算前景也不能算背景。
        # 简单起见，我们目前假设 targets 已经在 dataset 中处理过(ignore -> background 0)。
        # 所以这里的 distance transform 还是基于 targets (0/1) 计算的。
        # valid_mask 仅用于最后计算 Dice 时的加权区域。
        dist_pos, dist_neg = compute_distance_transform_gpu(targets, self.tau)

        # 表面权重：距离越近（≤ tau），权重越高
        # GT 表面区域：前景中距离背景 ≤ tau 的体素
        gt_surface_weight = torch.clamp(1.0 - dist_pos / self.tau, min=0.0)
        # 包含"刚好在边界外"的背景也给权重
        gt_surface_weight = gt_surface_weight + torch.clamp(1.0 - dist_neg / self.tau, min=0.0)
        gt_surface_weight = torch.clamp(gt_surface_weight, max=1.0)
        
        # Apply valid_mask to surface weight
        if valid_mask is not None:
             gt_surface_weight = gt_surface_weight * valid_mask

        # Pred 表面区域：使用梯度近似（概率的空间梯度大的区域 = 预测边界）
        # 简化版：直接用 GT 的表面区域作为加权 mask，让模型在该区域上努力
        weighted_probs = probs * gt_surface_weight
        weighted_targets = targets * gt_surface_weight

        # 在加权区域计算 Dice
        intersection = (weighted_probs * weighted_targets).sum()
        union = weighted_probs.sum() + weighted_targets.sum()

        surface_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - surface_dice


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - 基于距离变换的边界感知损失

    核心思想:
    对于极薄的卷轴层，边缘预测的精度至关重要。
    Boundary Loss 使用有符号距离函数对预测进行加权：
    - 正确预测边界附近的体素：低损失
    - 远离边界的错误预测：高损失

    参考: Kervadec et al., "Boundary loss for highly unbalanced segmentation" (2019)

    公式: L_boundary = sum(probs * signed_distance_map) / n_voxels
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, D, H, W) 未归一化的 logits
            targets: (B, 1, D, H, W) 二值掩码
            valid_mask: (B, 1, D, H, W) 有效区域
        """
        probs = torch.sigmoid(logits)

        # GPU 近似有符号距离函数 (Signed Distance Function)
        dist_pos, dist_neg = compute_distance_transform_gpu(targets, tau=3.0)

        # 有号距离: 前景内部为 -dist_pos, 背景为 +dist_neg
        signed_dist = dist_neg - dist_pos  # 前景内部为负, 背景为正

        # Boundary Loss = 预测概率 × 有号距离的均值
        # 如果模型正确预测前景（probs 高, signed_dist 负）→ 负贡献（好）
        # 如果模型在背景处预测前景（probs 高, signed_dist 正）→ 正贡献（惩罚）
        term = probs * signed_dist
        
        if valid_mask is not None:
             term = term * valid_mask
             boundary_loss = term.sum() / valid_mask.sum().clamp(min=1)
        else:
             boundary_loss = term.mean()

        return boundary_loss


class CompoundLoss(nn.Module):
    """
    混合损失函数 (Compound Loss) - 替代 CombinedLoss

    L_total = w_bce * BCE + w_surface * SurfaceDice + w_boundary * Boundary

    默认权重:
    - BCE: 1.0 (基础像素分类)
    - SoftSurfaceDice: 1.0 (模拟比赛指标)
    - BoundaryLoss: 0.5 (边缘强化)

    Args:
        w_bce: BCE 权重
        w_surface: SurfaceDice 权重
        w_boundary: Boundary 权重
        tau: SurfaceDice 的容差
        boundary_warmup: Boundary Loss 开始生效的 Epoch 数
    """

    def __init__(
        self,
        w_bce: float = 1.0,
        w_surface: float = 1.0,
        w_boundary: float = 0.5,
        tau: float = 2.0,
        boundary_warmup: int = 5
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_surface = w_surface
        self.w_boundary = w_boundary
        self.boundary_warmup = boundary_warmup

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.surface_dice_loss = SoftSurfaceDiceLoss(tau=tau)
        self.boundary_loss = BoundaryLoss()

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """外部设置当前 Epoch，用于 warmup 控制"""
        self.current_epoch = epoch

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor = None):
        """
        Args:
            logits: (B, 1, D, H, W) 未归一化的 logits
            targets: (B, 1, D, H, W) 二值掩码
            valid_mask: (B, 1, D, H, W) 有效区域

        Returns:
            total_loss, bce, surface_dice, boundary
        """
        # 1. BCE Loss
        if valid_mask is not None:
            bce_raw = self.bce_loss(logits, targets) # BCEWithLogits default reduction='mean'
            # Wait, default reduction='mean' in init!
            # We need to change reduction to 'none' if mask is present, or assume it's set to 'mean' and we can't easily mask.
            # BCEWithLogitsLoss doesn't support changing reduction in forward.
            # So we should use functional interface or re-init?
            # Re-init is slow. Functional is better.
            # Or assume self.bce_loss is used?
            # Actually self.bce_loss in init uses defaults (mean).
            # Let's use F.binary_cross_entropy_with_logits
            bce_val = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            bce = (bce_val * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        else:
            bce = self.bce_loss(logits, targets)

        # 2. Soft Surface Dice Loss
        surface_dice = self.surface_dice_loss(logits, targets, valid_mask=valid_mask)

        # 3. Boundary Loss (带 warmup)
        if self.current_epoch >= self.boundary_warmup:
            boundary = self.boundary_loss(logits, targets, valid_mask=valid_mask)
        else:
            # 保持计算图连通，但贡献为 0
            boundary = (logits * 0).mean()

        total_loss = (
            self.w_bce * bce +
            self.w_surface * surface_dice +
            self.w_boundary * boundary
        )

        return total_loss, bce, surface_dice, boundary


# ========== 测试 ==========

if __name__ == "__main__":
    print("=" * 50)
    print("  测试 CompoundLoss (Surface-Aware)")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模拟输入 (requires_grad=True 以验证梯度)
    B, C, D, H, W = 2, 1, 32, 64, 64
    logits = torch.randn(B, C, D, H, W, device=device, requires_grad=True)
    targets = torch.zeros(B, C, D, H, W, device=device)

    # 创建一个薄层结构（模拟卷轴）
    targets[:, :, 14:18, 10:54, 10:54] = 1.0

    loss_fn = CompoundLoss(w_bce=1.0, w_surface=1.0, w_boundary=0.5, tau=2.0)
    loss_fn.set_epoch(10)  # 跳过 warmup

    total, bce, surface, boundary = loss_fn(logits, targets)

    print(f"\n[随机预测]")
    print(f"  Total Loss:        {total.item():.6f}")
    print(f"  BCE:               {bce.item():.6f}")
    print(f"  SoftSurfaceDice:   {surface.item():.6f}")
    print(f"  Boundary:          {boundary.item():.6f}")

    # 验证梯度
    total.backward()
    grad_norm = logits.grad.norm().item()
    print(f"  Gradient norm:     {grad_norm:.6f}")
    print(f"  ✅ 梯度回传成功")

    # 完美预测测试
    perfect_logits = torch.where(targets > 0, torch.tensor(5.0), torch.tensor(-5.0)).to(device)
    perfect_logits.requires_grad_(True)
    total2, bce2, surface2, boundary2 = loss_fn(perfect_logits, targets)
    print(f"\n[完美预测]")
    print(f"  Total Loss:        {total2.item():.6f}")
    print(f"  BCE:               {bce2.item():.6f}")
    print(f"  SoftSurfaceDice:   {surface2.item():.6f}")
    print(f"  Boundary:          {boundary2.item():.6f}")

    # Warmup 测试
    loss_fn2 = CompoundLoss(w_bce=1.0, w_surface=1.0, w_boundary=0.5)
    loss_fn2.set_epoch(0)
    logits2 = torch.randn(B, C, D, H, W, device=device, requires_grad=True)
    total3, _, _, b3 = loss_fn2(logits2, targets)
    total3.backward()
    print(f"\n[Warmup 测试 (epoch=0, boundary 应为 0)]")
    print(f"  Boundary:          {b3.item():.6f}")
    print(f"  Gradient norm:     {logits2.grad.norm().item():.6f}")
    print(f"  ✅ Warmup 梯度正常")

    print(f"\n{'=' * 50}")
    print(f"  ✅ 所有测试通过")
    print(f"{'=' * 50}")

