# 2026 Kaggle Vesuvius Challenge - 深度代码审计报告

**审计对象**: 当前工作区 (Version 20_src)
**审计专家**: Antigravity (Google Deepmind Agent)
**日期**: 2026-02-14

---

## 🛑 审计结论总览

经过对 `20_model/chimera_loss.py`, `20_model/dual_unet.py`, `20_src/winding_solver.py`, `20_src/run_inference.py` 的深度审查，发现以下 **致命隐患 (Critical Issues)**：

1.  **[CRITICAL] 虚无主义陷阱 (All-Ones/Zeros Trap)**
    *   **现象**: Batch 100/150 出现 `Pred_Pixels=0.0`，验证集 Dice 接近 0。
    *   **根源**: `ChimeraLoss` 中的 BCE Loss 未加权。由于正样本稀疏度 < 1%，标准 BCE 导致背景（0）的梯度压倒了前景（1）。模型学会“全预测 0”即可获得极低的 Loss (0.01 左右)。
    *   **修复**: 必须引入 `pos_weight` (建议 100.0) 或切换为 Focal Loss。

2.  **[CRITICAL] Z轴分辨率崩溃 (Sensor Resolution Collapse)**
    *   **现象**: 纸草厚度仅 1-3 体素，但网络有 4 层 `MaxPool3d(2)`。
    *   **根源**: 4 层下采样将 Z 轴分辨率降低 16 倍 ($2^4$)。对于 64 层的 Chunk，Bottom 层只有 4 层特征。对于 30 层的 Chunk，特征图在 Z 轴上仅剩 1-2 像素，这在物理上抹除了纸草和法线的任何微观结构。
    *   **修复**: 这里的 MaxPool 必须改为各向异性 (Anisotropic)：`kernel_size=(1, 2, 2)`，仅在 XY 平面下采样，保留 Z 轴分辨率。

3.  **[HIGH] Winding Solver 拓扑断裂风险**
    *   **现象**: `build_sparse_graph` 仅依赖阈值后的 6-邻域连接。
    *   **风险**: 若 U-Net 输出断裂 (Dice < 0.15)，图会分裂成无数孤立子图。`auto_assign_seeds` 仅在 Volume 边界和中心分配种子。孤立的中间碎片将无法接收到正确的边界条件，导致求解出的 Winding Number 为 0 或随机值。
    *   **建议**: 在 Graph 构建前引入形态学闭运算 (Closing) 或在 Solver 中增加连通分量分析。

4.  **[MEDIUM] 推理资源与性能**
    *   **分析**: 单 Chunk ($512^3$) 推理显存约 4GB，RAM 约 5GB，时间约 25s。虽然不会立即 OOM，但若扩展到 $8000^3$ 全图则必死无疑。
    *   **建议**: 保持分块 (Sliding Window) 策略，并严格监控重叠区域的处理。

---

## 🛠️ 代码修正方案 (Actionable Fixes)

### 1. 修正 `chimera_loss.py` (引入 Focal Loss & Weighted BCE)

**文件**: `20_src/20_model/chimera_loss.py`

```python
# 修改 Class: ChimeraLoss

class ChimeraLoss(nn.Module):
    def __init__(
        self,
        lambda_normal: float = 1.0,
        lambda_bce: float = 1.0,
        dice_smooth: float = 1e-6,
        pos_weight: float = 100.0,  # 新增: 正样本权重
    ):
        super().__init__()
        self.lambda_normal = lambda_normal
        self.lambda_bce = lambda_bce
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        
        # 核心修复: 引入 pos_weight 惩罚背景预测
        # pos_weight > 1 增加 Recall，< 1 增加 Precision
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        
        self.normal_loss = NormalCosineLoss()

    def forward(self, seg_logits, pred_normals, targets):
        # ... (同前)
        # 确保 pos_weight 在正确的设备上
        if self.bce_loss.pos_weight.device != seg_logits.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(seg_logits.device)
            
        bce_val = self.bce_loss(seg_logits, targets.float())
        # ...
```

### 2. 修正 `dual_unet.py` (保护 Z 轴分辨率)

**文件**: `20_src/20_model/dual_unet.py`

**原理**: 将 `MaxPool3d(2)` 改为 `MaxPool3d(kernel_size=(1, 2, 2))`。这样 Z 轴保持不变，XY 轴降采样。这对于切片数据至关重要。

```python
# 修改 Class: DualHeadResUNet3D

def __init__(self, in_channels: int = 1, n_filters: int = 16):
    super().__init__()

    # ===== Encoder (各向异性下采样) =====
    self.enc1 = DoubleConv3D(in_channels, n_filters)
    # 第一层可以做全向降采样 (64 -> 32)
    self.pool1 = nn.MaxPool3d(2)  

    self.enc2 = DoubleConv3D(n_filters, n_filters * 2)
    # 第二层开始保护 Z 轴 (32 -> 32)
    self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    self.enc3 = DoubleConv3D(n_filters * 2, n_filters * 4)
    # 第三层继续保护 Z 轴 (32 -> 32)
    self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    self.enc4 = DoubleConv3D(n_filters * 4, n_filters * 8)
    # 第四层继续保护 Z 轴 (32 -> 32)
    self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    # 结果: Z 轴只在第一层降采样了一次 (64 -> 32)，保留了足够的厚度信息。
    # 相应的 Decoder 上采样层 (ConvTranspose3d) 也必须修改 kernel/stride。
```

---

## 🔮 梯度分析结论

*   **当前梯度**: 由于正样本极少，`L_BCE` 的梯度主要由负样本贡献。负样本告诉网络：“降低 logits 值！”。网络照做，将所有 logits 推向 -10，导致 Sigmoid 输出全是 0。`L_Dice` 在预测全 0 时梯度消失或不稳定。
*   **修正后梯度**: `pos_weight=100` 将强制网络关注那 1% 的正样本。正样本的梯度将放大 100 倍，告诉网络：“这里必须是 1！”。这将平衡负样本的压制，打破虚无主义陷阱。

**下一步行动**:
我将直接应用上述代码修改。
