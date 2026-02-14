# Project Architecture: 20_src

## Directory Tree (Filtered)
```text
20_src/
├── 20_data
│   ├── __init__.py
│   ├── dataset.py
│   └── transforms.py
├── 20_model
│   ├── __init__.py
│   ├── chimera_loss.py
│   └── dual_unet.py
├── 20_src_code_only.md
├── AUDIT_REPORT_2026.md
├── __init__.py
├── aggregate.py
├── graph_builder.py
├── output
│   ├── chimera_run_20260214_031947_gpu_nf16
│   ├── chimera_run_20260214_032021_gpu_nf16
│   │   ├── 1004283650_mask.tif
│   │   ├── 1006462223_mask.tif
│   │   ├── 1013184726_mask.tif
│   │   ├── 102536988_mask.tif
│   │   └── 1029212680_mask.tif
│   ├── inference_20260214_031445
│   ├── train_20260214_033543
│   │   ├── best_model.pth
│   │   └── training_history.json
│   ├── train_20260214_034217
│   ├── train_20260214_034358
│   ├── train_20260214_034543
│   ├── train_20260214_034653
│   ├── train_20260214_034843
│   ├── train_20260214_035035
│   ├── train_20260214_035705
│   ├── train_20260214_035932
│   ├── train_20260214_040039
│   ├── train_20260214_040137
│   ├── train_20260214_041118
│   ├── train_20260214_041247
│   ├── train_20260214_041622
│   │   ├── best_model.pth
│   │   ├── epoch_masks
│   │   │   ├── epoch001_pred_mask.tif
│   │   │   ├── epoch002_pred_mask.tif
│   │   │   ├── epoch003_pred_mask.tif
│   │   │   └── epoch004_pred_mask.tif
│   │   └── epoch_vis
│   │       ├── epoch001_3d_comparison.png
│   │       ├── epoch001_comparison.png
│   │       ├── epoch002_3d_comparison.png
│   │       ├── epoch002_comparison.png
│   │       ├── epoch003_3d_comparison.png
│   │       ├── epoch003_comparison.png
│   │       ├── epoch004_3d_comparison.png
│   │       └── epoch004_comparison.png
│   ├── train_20260214_044912
│   ├── train_20260214_050229
│   │   ├── best_model.pth
│   │   ├── epoch_masks
│   │   │   └── epoch001_pred_mask.tif
│   │   └── epoch_vis
│   │       ├── epoch001_3d_comparison.png
│   │       └── epoch001_comparison.png
│   ├── train_20260214_050630
│   │   ├── best_model.pth
│   │   ├── epoch_masks
│   │   │   ├── epoch001_pred_mask.tif
│   │   │   ├── epoch002_pred_mask.tif
│   │   │   ├── epoch003_pred_mask.tif
│   │   │   ├── epoch004_pred_mask.tif
│   │   │   └── epoch005_pred_mask.tif
│   │   └── epoch_vis
│   │       ├── epoch001_3d_comparison.png
│   │       ├── epoch001_comparison.png
│   │       ├── epoch002_3d_comparison.png
│   │       ├── epoch002_comparison.png
│   │       ├── epoch003_3d_comparison.png
│   │       ├── epoch003_comparison.png
│   │       ├── epoch004_3d_comparison.png
│   │       ├── epoch004_comparison.png
│   │       ├── epoch005_3d_comparison.png
│   │       └── epoch005_comparison.png
│   ├── train_20260214_052733
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch010.pth
│   │   ├── epoch_masks
│   │   │   ├── epoch001_pred_mask.tif
│   │   │   ├── epoch002_pred_mask.tif
│   │   │   ├── epoch003_pred_mask.tif
│   │   │   ├── epoch004_pred_mask.tif
│   │   │   ├── epoch005_pred_mask.tif
│   │   │   ├── epoch006_pred_mask.tif
│   │   │   ├── epoch007_pred_mask.tif
│   │   │   ├── epoch008_pred_mask.tif
│   │   │   ├── epoch009_pred_mask.tif
│   │   │   ├── epoch010_pred_mask.tif
│   │   │   ├── epoch011_pred_mask.tif
│   │   │   ├── epoch012_pred_mask.tif
│   │   │   ├── epoch013_pred_mask.tif
│   │   │   ├── epoch014_pred_mask.tif
│   │   │   └── epoch015_pred_mask.tif
│   │   └── epoch_vis
│   │       ├── epoch001_3d_comparison.png
│   │       ├── epoch001_comparison.png
│   │       ├── epoch002_3d_comparison.png
│   │       ├── epoch002_comparison.png
│   │       ├── epoch003_3d_comparison.png
│   │       ├── epoch003_comparison.png
│   │       ├── epoch004_3d_comparison.png
│   │       ├── epoch004_comparison.png
│   │       ├── epoch005_3d_comparison.png
│   │       ├── epoch005_comparison.png
│   │       ├── epoch006_3d_comparison.png
│   │       ├── epoch006_comparison.png
│   │       ├── epoch007_3d_comparison.png
│   │       ├── epoch007_comparison.png
│   │       ├── epoch008_3d_comparison.png
│   │       ├── epoch008_comparison.png
│   │       ├── epoch009_3d_comparison.png
│   │       ├── epoch009_comparison.png
│   │       ├── epoch010_3d_comparison.png
│   │       ├── epoch010_comparison.png
│   │       ├── epoch011_3d_comparison.png
│   │       ├── epoch011_comparison.png
│   │       ├── epoch012_3d_comparison.png
│   │       ├── epoch012_comparison.png
│   │       ├── epoch013_3d_comparison.png
│   │       ├── epoch013_comparison.png
│   │       ├── epoch014_3d_comparison.png
│   │       ├── epoch014_comparison.png
│   │       ├── epoch015_3d_comparison.png
│   │       └── epoch015_comparison.png
│   ├── train_20260214_061413
│   │   ├── best_model.pth
│   │   ├── epoch_masks
│   │   │   ├── epoch001_pred_mask.tif
│   │   │   ├── epoch002_pred_mask.tif
│   │   │   └── epoch003_pred_mask.tif
│   │   └── epoch_vis
│   │       ├── epoch001_3d_comparison.png
│   │       ├── epoch001_comparison.png
│   │       ├── epoch002_3d_comparison.png
│   │       ├── epoch002_comparison.png
│   │       ├── epoch003_3d_comparison.png
│   │       └── epoch003_comparison.png
│   ├── train_20260214_063047
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch010.pth
│   │   ├── epoch_masks
│   │   │   ├── epoch001_pred_mask.tif
│   │   │   ├── epoch002_pred_mask.tif
│   │   │   ├── epoch003_pred_mask.tif
│   │   │   ├── epoch004_pred_mask.tif
│   │   │   ├── epoch005_pred_mask.tif
│   │   │   ├── epoch006_pred_mask.tif
│   │   │   ├── epoch007_pred_mask.tif
│   │   │   ├── epoch008_pred_mask.tif
│   │   │   ├── epoch009_pred_mask.tif
│   │   │   ├── epoch010_pred_mask.tif
│   │   │   ├── epoch011_pred_mask.tif
│   │   │   ├── epoch012_pred_mask.tif
│   │   │   └── epoch013_pred_mask.tif
│   │   └── epoch_vis
│   │       ├── epoch001_3d_comparison.png
│   │       ├── epoch001_comparison.png
│   │       ├── epoch002_3d_comparison.png
│   │       ├── epoch002_comparison.png
│   │       ├── epoch003_3d_comparison.png
│   │       ├── epoch003_comparison.png
│   │       ├── epoch004_3d_comparison.png
│   │       ├── epoch004_comparison.png
│   │       ├── epoch005_3d_comparison.png
│   │       ├── epoch005_comparison.png
│   │       ├── epoch006_3d_comparison.png
│   │       ├── epoch006_comparison.png
│   │       ├── epoch007_3d_comparison.png
│   │       ├── epoch007_comparison.png
│   │       ├── epoch008_3d_comparison.png
│   │       ├── epoch008_comparison.png
│   │       ├── epoch009_3d_comparison.png
│   │       ├── epoch009_comparison.png
│   │       ├── epoch010_3d_comparison.png
│   │       ├── epoch010_comparison.png
│   │       ├── epoch011_3d_comparison.png
│   │       ├── epoch011_comparison.png
│   │       ├── epoch012_3d_comparison.png
│   │       ├── epoch012_comparison.png
│   │       ├── epoch013_3d_comparison.png
│   │       └── epoch013_comparison.png
│   ├── train_20260214_071306
│   ├── train_20260214_071429
│   ├── train_20260214_071458
│   ├── train_20260214_071514
│   │   ├── best_model.pth
│   │   ├── epoch_masks
│   │   │   ├── epoch001_pred_mask.tif
│   │   │   ├── epoch002_pred_mask.tif
│   │   │   └── epoch003_pred_mask.tif
│   │   └── epoch_vis
│   │       ├── epoch001_3d_comparison.png
│   │       ├── epoch001_comparison.png
│   │       ├── epoch002_3d_comparison.png
│   │       ├── epoch002_comparison.png
│   │       ├── epoch003_3d_comparison.png
│   │       └── epoch003_comparison.png
│   ├── train_20260214_072332
│   │   ├── best_model.pth
│   │   ├── epoch_masks
│   │   │   ├── epoch001_pred_mask.tif
│   │   │   ├── epoch002_pred_mask.tif
│   │   │   ├── epoch003_pred_mask.tif
│   │   │   ├── epoch004_pred_mask.tif
│   │   │   ├── epoch005_pred_mask.tif
│   │   │   ├── epoch006_pred_mask.tif
│   │   │   ├── epoch007_pred_mask.tif
│   │   │   ├── epoch008_pred_mask.tif
│   │   │   └── epoch009_pred_mask.tif
│   │   └── epoch_vis
│   │       ├── epoch001_3d_comparison.png
│   │       ├── epoch001_comparison.png
│   │       ├── epoch002_3d_comparison.png
│   │       ├── epoch002_comparison.png
│   │       ├── epoch003_3d_comparison.png
│   │       ├── epoch003_comparison.png
│   │       ├── epoch004_3d_comparison.png
│   │       ├── epoch004_comparison.png
│   │       ├── epoch005_3d_comparison.png
│   │       ├── epoch005_comparison.png
│   │       ├── epoch006_3d_comparison.png
│   │       ├── epoch006_comparison.png
│   │       ├── epoch007_3d_comparison.png
│   │       ├── epoch007_comparison.png
│   │       ├── epoch008_3d_comparison.png
│   │       ├── epoch008_comparison.png
│   │       ├── epoch009_3d_comparison.png
│   │       └── epoch009_comparison.png
│   ├── train_20260214_074649
│   ├── train_20260214_074853
│   │   ├── best_model.pth
│   │   ├── epoch_masks
│   │   │   ├── epoch001_pred_mask.tif
│   │   │   ├── epoch002_pred_mask.tif
│   │   │   ├── epoch003_pred_mask.tif
│   │   │   ├── epoch004_pred_mask.tif
│   │   │   ├── epoch005_pred_mask.tif
│   │   │   └── epoch006_pred_mask.tif
│   │   └── epoch_vis
│   │       ├── epoch001_3d_comparison.png
│   │       ├── epoch001_comparison.png
│   │       ├── epoch002_3d_comparison.png
│   │       ├── epoch002_comparison.png
│   │       ├── epoch003_3d_comparison.png
│   │       ├── epoch003_comparison.png
│   │       ├── epoch004_3d_comparison.png
│   │       ├── epoch004_comparison.png
│   │       ├── epoch005_3d_comparison.png
│   │       ├── epoch005_comparison.png
│   │       ├── epoch006_3d_comparison.png
│   │       └── epoch006_comparison.png
│   └── verification_slice.png
├── preprocess.py
├── run_inference.py
├── submission.py
├── train.py
├── verify_data.py
└── winding_solver.py
```

---

## File: AUDIT_REPORT_2026.md
```md
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

```

---
## File: __init__.py
```py
# 20_src: Hybrid Chimera MVP 模块
# 阶段一: 数据层 + 图构建
# 阶段二: 双头 U-Net 感知层
# 阶段三: Winding Number 求解器

```

---
## File: graph_builder.py
```py
"""
Vesuvius Challenge - 稀疏图构建模块 (MVP)

从 U-Net 输出的概率图和法线图构建 voxel-level 稀疏连接图。
参考 ThaumatoAnakalyptor instances_to_graph.py 的核心概念，纯 Python 重写。

核心逻辑：
1. 阈值化概率图，提取有效 voxel 作为节点
2. 6-邻域连接，边权重 = 法线向量点积（高对齐 = 强连接）
3. 输出 scipy.sparse.csr_matrix 邻接矩阵
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict, Optional


def build_sparse_graph(
    prob_map: np.ndarray,
    normal_map: np.ndarray,
    threshold: float = 0.5,
    use_cupy: bool = False,
) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[tuple, int]]:
    """
    从概率图和法线图构建稀疏邻接图

    Args:
        prob_map: 形状 (D, H, W)，U-Net 输出的概率图，值域 [0, 1]
        normal_map: 形状 (3, D, H, W)，预测的法线图，(nx, ny, nz)
        threshold: 概率阈值，高于此值的 voxel 成为节点，默认 0.5
        use_cupy: 是否使用 CuPy 进行 GPU 加速（预留接口），默认 False

    Returns:
        adjacency: scipy.sparse.csr_matrix，稀疏邻接矩阵，形状 (N, N)
        node_coords: np.ndarray，形状 (N, 3)，每个节点的 (d, h, w) 坐标
        node_index_map: dict，(d, h, w) → 节点索引的映射

    Raises:
        ValueError: 输入形状不匹配时抛出
    """
    # --- 输入校验 ---
    if prob_map.ndim != 3:
        raise ValueError(f"prob_map 必须是 3D 数组，实际: {prob_map.ndim}D")
    if normal_map.ndim != 4 or normal_map.shape[0] != 3:
        raise ValueError(
            f"normal_map 必须是 (3, D, H, W) 形状，实际: {normal_map.shape}"
        )
    if prob_map.shape != normal_map.shape[1:]:
        raise ValueError(
            f"prob_map {prob_map.shape} 和 normal_map {normal_map.shape[1:]} "
            f"空间尺寸不匹配"
        )

    D, H, W = prob_map.shape

    # --- 步骤 1: 阈值化，提取有效 voxel 坐标 ---
    mask = prob_map > threshold
    coords = np.argwhere(mask)  # (N, 3)，每行是 (d, h, w)
    num_nodes = len(coords)

    if num_nodes == 0:
        # 没有有效节点，返回空图
        empty_adj = sparse.csr_matrix((0, 0), dtype=np.float32)
        return empty_adj, np.empty((0, 3), dtype=np.int64), {}

    print(f"[build_sparse_graph] 有效节点数: {num_nodes} / {D*H*W} "
          f"(占比 {num_nodes / (D*H*W) * 100:.1f}%)")

    # --- 步骤 2: 建立坐标 → 索引映射 ---
    node_index_map: Dict[tuple, int] = {}
    for i, (d, h, w) in enumerate(coords):
        node_index_map[(int(d), int(h), int(w))] = i

    # --- 步骤 3: 构建边（6-邻域） ---
    # 6 个邻域方向: ±d, ±h, ±w
    neighbors_offsets = np.array([
        [-1, 0, 0], [1, 0, 0],   # d 方向
        [0, -1, 0], [0, 1, 0],   # h 方向
        [0, 0, -1], [0, 0, 1],   # w 方向
    ], dtype=np.int64)

    # 使用向量化操作加速边构建
    row_indices = []
    col_indices = []
    weights = []

    # 提取每个节点的法线向量 (N, 3)
    node_normals = np.stack([
        normal_map[c, coords[:, 0], coords[:, 1], coords[:, 2]]
        for c in range(3)
    ], axis=1)  # (N, 3)

    # 遍历每个邻域方向，批量处理
    for offset in neighbors_offsets:
        # 计算所有节点的邻居坐标
        neighbor_coords = coords + offset  # (N, 3)

        # 边界检查
        valid_mask = (
            (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 0] < D) &
            (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < H) &
            (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < W)
        )

        # 遍历有效的邻居（需要查字典确认是节点）
        valid_indices = np.where(valid_mask)[0]

        for i in valid_indices:
            nb_key = tuple(neighbor_coords[i])
            if nb_key in node_index_map:
                j = node_index_map[nb_key]

                # 计算边权重: 两个节点法线的点积
                dot = np.dot(node_normals[i], node_normals[j])

                # 只保留正对齐（法线方向一致 = 属于同一表面）
                weight = max(float(dot), 0.0)

                if weight > 0:
                    row_indices.append(i)
                    col_indices.append(j)
                    weights.append(weight)

    # --- 步骤 4: 构建稀疏矩阵 ---
    if len(row_indices) == 0:
        adjacency = sparse.csr_matrix(
            (num_nodes, num_nodes), dtype=np.float32
        )
    else:
        row_indices = np.array(row_indices, dtype=np.int64)
        col_indices = np.array(col_indices, dtype=np.int64)
        weights = np.array(weights, dtype=np.float32)

        adjacency = sparse.csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(num_nodes, num_nodes),
            dtype=np.float32,
        )

    print(f"[build_sparse_graph] 边数: {adjacency.nnz} "
          f"(平均度: {adjacency.nnz / max(num_nodes, 1):.2f})")

    return adjacency, coords, node_index_map


def build_graph_laplacian(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    从邻接矩阵构建 Graph Laplacian: L = D - A

    Args:
        adjacency: 稀疏邻接矩阵 (N, N)

    Returns:
        laplacian: 稀疏 Laplacian 矩阵 (N, N)
    """
    # 度矩阵: 每行权重之和
    degree = np.array(adjacency.sum(axis=1)).flatten()
    D = sparse.diags(degree, format='csr')

    # Laplacian = D - A
    laplacian = D - adjacency

    return laplacian


if __name__ == "__main__":
    print("=== build_sparse_graph 自测 ===")

    # 创建合成数据: 8x8x8 体积
    D, H, W = 8, 8, 8
    prob_map = np.zeros((D, H, W), dtype=np.float32)

    # 中心 4x4x4 区域设为高概率
    prob_map[2:6, 2:6, 2:6] = 0.8

    # 所有法线指向 z 方向 (完美对齐)
    normal_map = np.zeros((3, D, H, W), dtype=np.float32)
    normal_map[2, :, :, :] = 1.0  # nz = 1.0

    # 构建图
    adj, coords, idx_map = build_sparse_graph(prob_map, normal_map)

    print(f"节点数: {len(coords)}")        # 预期: 4^3 = 64
    print(f"边数: {adj.nnz}")              # 预期: 每个内部节点 6 条边
    print(f"邻接矩阵形状: {adj.shape}")

    assert len(coords) == 64, f"预期 64 个节点，实际 {len(coords)}"
    assert adj.nnz > 0, "邻接矩阵应有非零元素"

    # 测试 Laplacian
    L = build_graph_laplacian(adj)
    print(f"Laplacian 形状: {L.shape}")

    # Laplacian 的每行之和应为 0
    row_sums = np.abs(np.array(L.sum(axis=1)).flatten())
    assert row_sums.max() < 1e-6, f"Laplacian 行和不为零: {row_sums.max()}"

    print("✓ 所有测试通过！")

```

---
## File: preprocess.py
```py
"""
Vesuvius Challenge - 数据预处理脚本

功能：将 LZW 压缩的 TIF 转换为未压缩的 NumPy 格式 (.npy)。
目的：彻底解决训练时的 IO 瓶颈，支持 Memory-Mapped (mmap) 零拷贝读取。
性能提升：预计训练 IO 速度提升 100 倍。
硬盘占用：约 25GB (786 个 volumes)。

用法:
    python 20_src/preprocess.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm
import multiprocessing

# 配置
# 原始数据目录
SRC_IMG_DIR = Path("data/vesuvius-challenge-surface-detection/train_images")
SRC_LBL_DIR = Path("data/vesuvius-challenge-surface-detection/train_labels")

# 输出目录
DST_IMG_DIR = Path("data/vesuvius-challenge-surface-detection/train_images_npy")
DST_LBL_DIR = Path("data/vesuvius-challenge-surface-detection/train_labels_npy")


def convert_file(args):
    """
    单个文件转换任务
    """
    src_path, dst_path, is_label = args
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    
    if dst_path.exists():
        return  # 跳过已存在的

    try:
        # 读取 TIF
        volume = tifffile.imread(src_path)
        
        # 确保是 3D
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        
        # 转为 uint8 (节省空间，训练时再转 float32)
        # 原始数据通常就是 uint8，这里确保类型一致
        if volume.dtype != np.uint8:
            if volume.max() <= 1.0:
                volume = (volume * 255).astype(np.uint8)
            elif volume.max() <= 255:
                volume = volume.astype(np.uint8)
            # 如果是 label 且 max > 1 (e.g. 255)，也可以保持 uint8
        
        # 保存为 .npy (未压缩)
        np.save(dst_path, volume)
        
    except Exception as e:
        print(f"\nError converting {src_path}: {e}")


def main():
    print(f"{'='*50}")
    print(f"  🚀 Vesuvius 数据预处理 (TIF -> NPY)")
    print(f"  源目录: {SRC_IMG_DIR}")
    print(f"  目标目录: {DST_IMG_DIR}")
    print(f"{'='*50}\n")
    
    # 创建输出目录
    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DST_LBL_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    
    # 扫描 Image
    img_files = sorted(list(SRC_IMG_DIR.glob("*.tif")))
    for p in img_files:
        dst = DST_IMG_DIR / (p.stem + ".npy")
        tasks.append((str(p), str(dst), False))
        
    # 扫描 Label
    lbl_files = sorted(list(SRC_LBL_DIR.glob("*.tif")))
    for p in lbl_files:
        dst = DST_LBL_DIR / (p.stem + ".npy")
        tasks.append((str(p), str(dst), True))
        
    print(f"找到 {len(img_files)} 个 image 文件, {len(lbl_files)} 个 label 文件。")
    print(f"总任务数: {len(tasks)}")
    
    # 并行处理 (根据 CPU 核数)
    # Windows 下多进程要注意 if __name__ == '__main__':
    num_workers = min(8, os.cpu_count() or 4)
    print(f"使用 {num_workers} 个进程并行转换...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(convert_file, tasks), total=len(tasks), unit="file"))
        
    print("\n✅ 转换完成！")
    print(f"输出大小检查: {DST_IMG_DIR}")


if __name__ == "__main__":
    # Windows 必须
    multiprocessing.freeze_support()
    main()

```

---
## File: run_inference.py
```py
"""
Vesuvius Challenge - GPU 推理脚本 v2

核心改进：
- Sliding Window 分块推理：避免 320³ 体积直接送入 GPU 导致 OOM
- 实时进度监控：GPU 显存、处理速度、ETA
- 规范化输出目录命名：chimera_run_{日期}_{模式}

用法:
    python 20_src/run_inference.py --max_chunks 5
    python 20_src/run_inference.py --checkpoint path/to/model.pth
"""

import os
import sys
import time
import argparse
import psutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
import tifffile

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from importlib import import_module

model_mod = import_module("20_src.20_model.dual_unet")
DualHeadResUNet3D = model_mod.DualHeadResUNet3D


# ===== 监控工具 =====

def get_gpu_stats():
    """获取 GPU 显存使用信息"""
    if not torch.cuda.is_available():
        return "CPU 模式"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"GPU: {allocated:.1f}G/{total:.1f}G (reserved {reserved:.1f}G)"


def get_ram_stats():
    """获取 RAM 使用信息"""
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1024**3
    total_gb = mem.total / 1024**3
    return f"RAM: {used_gb:.1f}G/{total_gb:.1f}G ({mem.percent}%)"


def format_eta(seconds):
    """格式化剩余时间"""
    if seconds < 0:
        return "计算中..."
    td = timedelta(seconds=int(seconds))
    return str(td)


def print_progress(current, total, chunk_id, stage, extra=""):
    """打印实时进度"""
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{bar}] {current}/{total} ({pct:.0f}%) | ID:{chunk_id} | {stage}"
    if extra:
        line += f" | {extra}"
    print(line, flush=True)


# ===== 分块推理引擎 =====

def sliding_window_inference(
    model,
    volume: torch.Tensor,
    patch_size: tuple = (64, 128, 128),
    overlap: int = 8,
    device: torch.device = None,
):
    """
    Sliding Window 分块推理

    将大体积切成小 patch，逐个送入 GPU 推理，再拼回来。
    重叠区域使用平均融合。

    Args:
        model: DualHeadResUNet3D
        volume: (1, 1, D, H, W) 完整输入体积
        patch_size: 每个 patch 的大小 (pD, pH, pW)
        overlap: patch 之间的重叠体素数
        device: 推理设备

    Returns:
        seg_prob: (D, H, W) 分割概率图
        normal_map: (3, D, H, W) 法线图
    """
    _, _, D, H, W = volume.shape
    pD, pH, pW = patch_size
    stride_d = pD - overlap
    stride_h = pH - overlap
    stride_w = pW - overlap

    # 输出累积器
    seg_sum = torch.zeros(1, 1, D, H, W, dtype=torch.float32)
    normal_sum = torch.zeros(1, 3, D, H, W, dtype=torch.float32)
    count = torch.zeros(1, 1, D, H, W, dtype=torch.float32)

    # 计算所有 patch 的起始位置
    d_starts = list(range(0, max(D - pD + 1, 1), stride_d))
    h_starts = list(range(0, max(H - pH + 1, 1), stride_h))
    w_starts = list(range(0, max(W - pW + 1, 1), stride_w))

    # 确保覆盖边界
    if d_starts[-1] + pD < D:
        d_starts.append(D - pD)
    if h_starts[-1] + pH < H:
        h_starts.append(H - pH)
    if w_starts[-1] + pW < W:
        w_starts.append(W - pW)

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    patch_idx = 0

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                patch_idx += 1

                # 提取 patch
                d1 = min(d0 + pD, D)
                h1 = min(h0 + pH, H)
                w1 = min(w0 + pW, W)

                patch = volume[:, :, d0:d1, h0:h1, w0:w1]

                # Pad 如果尺寸不够
                actual_d, actual_h, actual_w = d1 - d0, h1 - h0, w1 - w0
                if actual_d < pD or actual_h < pH or actual_w < pW:
                    pad_d = pD - actual_d
                    pad_h = pH - actual_h
                    pad_w = pW - actual_w
                    patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d))

                # GPU 推理
                patch_gpu = patch.to(device)
                seg_logits, normals = model(patch_gpu)
                seg_prob = torch.sigmoid(seg_logits)

                # 裁剪回实际尺寸
                seg_prob = seg_prob[:, :, :actual_d, :actual_h, :actual_w].cpu()
                normals = normals[:, :, :actual_d, :actual_h, :actual_w].cpu()

                # 累加
                seg_sum[:, :, d0:d1, h0:h1, w0:w1] += seg_prob
                normal_sum[:, :, d0:d1, h0:h1, w0:w1] += normals
                count[:, :, d0:d1, h0:h1, w0:w1] += 1.0

                # 清理 GPU 显存
                del patch_gpu, seg_logits, seg_prob, normals
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # 每 5 个 patch 打印一次进度
                if patch_idx % 5 == 0 or patch_idx == total_patches:
                    print(f"    patch {patch_idx}/{total_patches} | {get_gpu_stats()}", flush=True)

    # 平均融合
    count = count.clamp(min=1.0)
    seg_avg = (seg_sum / count).squeeze().numpy()       # (D, H, W)
    normal_avg = (normal_sum / count).squeeze(0).numpy() # (3, D, H, W)

    return seg_avg, normal_avg


# ===== 主推理函数 =====

def run_inference(args):
    """主推理流程"""

    # 设备
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    # 规范化输出目录命名: chimera_run_{日期}_{模式}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "gpu" if dev.type == "cuda" else "cpu"
    run_name = f"chimera_run_{timestamp}_{mode_tag}_nf{args.n_filters}"
    run_output_dir = Path(args.output_dir) / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  🔬 Hybrid Chimera MVP - 推理")
    print(f"  设备: {dev} | {get_gpu_stats()}")
    print(f"  {get_ram_stats()}")
    print(f"  Patch 大小: {args.patch_size}")
    print(f"  输入: {args.input_dir}")
    print(f"  输出: {run_output_dir}")
    print(f"{'='*70}\n")

    # 加载模型
    model = DualHeadResUNet3D(in_channels=1, n_filters=args.n_filters)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=dev, weights_only=True)
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"[Model] 已加载权重: {args.checkpoint}")
    else:
        print("[Model] 使用随机初始化权重（MVP 演示模式）")

    model.to(dev)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] 参数量: {params:,} | {get_gpu_stats()}\n")

    # 扫描输入
    input_path = Path(args.input_dir)
    tif_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in ('.tif', '.tiff')
    ])
    if args.max_chunks:
        tif_files = tif_files[:args.max_chunks]

    total = len(tif_files)
    print(f"[Data] {total} 个 .tif 文件待处理\n")

    # 推理循环
    all_times = []
    normal_stats = []

    with torch.no_grad():
        for idx, tif_file in enumerate(tif_files):
            chunk_id = tif_file.stem
            print_progress(idx + 1, total, chunk_id, "开始加载")

            t_start = time.time()

            # 1. 加载
            t0 = time.time()
            volume_raw = tifffile.imread(str(tif_file))
            vol_shape = volume_raw.shape
            volume = volume_raw.astype(np.float32)
            if volume.max() > 1.0:
                volume = volume / 255.0 if volume.max() <= 255.0 else volume / 65535.0
            volume = np.clip(volume, 0.0, 1.0)
            x = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
            t_load = time.time() - t0

            print(f"    形状: {vol_shape} | 加载: {t_load:.2f}s | {get_ram_stats()}", flush=True)

            # 2. Sliding Window 推理
            t0 = time.time()
            patch_size = tuple(args.patch_size)
            seg_prob, normal_map = sliding_window_inference(
                model, x, patch_size=patch_size,
                overlap=args.overlap, device=dev,
            )
            t_unet = time.time() - t0

            # 3. 法向量质量诊断
            # 计算法线模长 (应接近 1.0) 和方向一致性
            norm_magnitude = np.linalg.norm(normal_map, axis=0)  # (D,H,W)
            mask_region = seg_prob > 0.3  # 只在有意义的区域统计
            if mask_region.sum() > 0:
                avg_norm = norm_magnitude[mask_region].mean()
                std_norm = norm_magnitude[mask_region].std()
                # 方向一致性: 邻居法线点积的均值
                normal_stats.append({
                    "id": chunk_id,
                    "avg_norm_magnitude": float(avg_norm),
                    "std_norm_magnitude": float(std_norm),
                })
                norm_diag = f"法线模长: {avg_norm:.3f}±{std_norm:.3f}"
            else:
                norm_diag = "法线: 无有效区域"

            # 4. 阈值化生成 mask
            t0 = time.time()
            final_mask = (seg_prob > 0.5).astype(np.uint8)
            t_post = time.time() - t0

            t_total = time.time() - t_start
            all_times.append(t_total)

            # 5. 保存
            output_filename = f"{chunk_id}_mask.tif"
            output_path = run_output_dir / output_filename
            tifffile.imwrite(str(output_path), final_mask)

            # 计算 ETA
            avg_time = sum(all_times) / len(all_times)
            eta = avg_time * (total - idx - 1)
            mask_pct = final_mask.sum() / final_mask.size * 100

            print(f"    U-Net: {t_unet:.1f}s | 后处理: {t_post:.3f}s | "
                  f"总计: {t_total:.1f}s", flush=True)
            print(f"    Mask: {mask_pct:.1f}% | {norm_diag}", flush=True)
            print(f"    {get_gpu_stats()} | {get_ram_stats()}", flush=True)
            print(f"    ETA: {format_eta(eta)} | 平均: {avg_time:.1f}s/chunk", flush=True)
            print(f"    → {output_path.name}", flush=True)
            print("", flush=True)

            # 清理
            del x, volume, volume_raw
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    # ===== 最终报告 =====
    total_time = sum(all_times)
    avg_time = total_time / max(len(all_times), 1)

    print(f"\n{'='*70}")
    print(f"  推理完成!")
    print(f"  处理: {len(all_times)} chunks")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  平均: {avg_time:.1f}s/chunk")
    print(f"  输出: {run_output_dir}")

    # 法向量质量诊断汇总
    if normal_stats:
        avg_mag = np.mean([s['avg_norm_magnitude'] for s in normal_stats])
        avg_std = np.mean([s['std_norm_magnitude'] for s in normal_stats])
        print(f"\n  📐 法向量诊断:")
        print(f"    平均模长: {avg_mag:.3f} (理想=1.0)")
        print(f"    模长标准差: {avg_std:.3f} (越低越好)")
        if avg_mag < 0.5:
            print(f"    ⚠️ 法线方向杂乱，建议加强 L_Cosine 权重 (当前 0.1 → 0.5)")

    # 内存诊断
    print(f"\n  💾 资源诊断:")
    print(f"    {get_ram_stats()}")
    print(f"    {get_gpu_stats()}")
    ram_gb = psutil.virtual_memory().used / 1024**3
    if ram_gb > 16:
        print(f"    ⚠️ RAM 使用超过 16GB，建议加入 supervoxel 降采样")

    # 时间诊断
    print(f"\n  ⏱️ 性能诊断:")
    print(f"    单 chunk 平均: {avg_time:.1f}s")
    if avg_time > 900:  # 15 分钟
        print(f"    ⚠️ 单 chunk > 15 min，必须切换到 CuPy GPU 求解器")
    else:
        print(f"    ✓ 单 chunk 时间可接受")

    estimated_full = avg_time * 786 / 60
    print(f"    预估全量推理: {estimated_full:.0f} min ({estimated_full/60:.1f} h)")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Chimera - GPU 推理 v2")
    parser.add_argument("--input_dir", type=str,
                        default="data/vesuvius-challenge-surface-detection/train_images")
    parser.add_argument("--output_dir", type=str, default="20_src/output")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_filters", type=int, default=16)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128],
                        help="Sliding window patch 大小 (D H W)")
    parser.add_argument("--overlap", type=int, default=8,
                        help="Patch 重叠体素数")

    args = parser.parse_args()
    run_inference(args)

```

---
## File: submission.py
```py
"""
Vesuvius Challenge - Hybrid Chimera MVP 推理流水线 (submission.py)

端到端推理流程：
1. 加载测试 3D TIF Chunk
2. 运行 DualHead U-Net → 概率图 + 法线图
3. build_sparse_graph → 稀疏邻接图
4. solve_winding_number → Winding Number 场
5. cut_mesh → Winding Mask
6. Porosity Injection: Final = Winding_Mask & (Prob > 0.4)
7. 输出最终 Binary Mask

包含性能计时，确保单 chunk < 10 分钟。
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tifffile

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 使用 importlib 导入数字开头的模块
from importlib import import_module

# 阶段一: 数据加载 + 图构建
dataset_mod = import_module("20_src.20_data.dataset")
graph_mod = import_module("20_src.graph_builder")

# 阶段二: 双头 U-Net
model_mod = import_module("20_src.20_model.dual_unet")

# 阶段三: Winding Number 求解器
solver_mod = import_module("20_src.winding_solver")

TifChunkDataset = dataset_mod.TifChunkDataset
DualHeadResUNet3D = model_mod.DualHeadResUNet3D
build_sparse_graph = graph_mod.build_sparse_graph
solve_winding_number = solver_mod.solve_winding_number
cut_mesh = solver_mod.cut_mesh
auto_assign_seeds = solver_mod.auto_assign_seeds


class HybridChimeraPipeline:
    """
    Hybrid Chimera MVP 推理流水线

    将 DualHead U-Net 的神经感知输出
    与 Winding Number 几何求解器的逻辑推理结合，
    通过 Porosity Injection 恢复拓扑细节。
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        prob_threshold: float = 0.5,
        porosity_threshold: float = 0.4,
        winding_threshold: float = 0.5,
        use_cupy: bool = False,
        n_filters: int = 16,
    ):
        """
        Args:
            checkpoint_path: 模型权重路径（.pth），None 则使用随机权重
            device: 推理设备，"auto" 自动选择
            prob_threshold: 图构建时的概率阈值
            porosity_threshold: Porosity Injection 的概率阈值
            winding_threshold: Winding Number 阈值化
            use_cupy: 是否使用 CuPy GPU 加速求解器
            n_filters: 模型基础通道数
        """
        # 设备选择
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Pipeline] 设备: {self.device}")

        self.prob_threshold = prob_threshold
        self.porosity_threshold = porosity_threshold
        self.winding_threshold = winding_threshold
        self.use_cupy = use_cupy

        # 加载模型
        self.model = DualHeadResUNet3D(in_channels=1, n_filters=n_filters)
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # 兼容带 "model." 前缀的 checkpoint
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[Pipeline] 已加载权重: {checkpoint_path}")
        else:
            print("[Pipeline] 使用随机初始化权重（未提供 checkpoint）")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _run_unet(self, volume: torch.Tensor):
        """
        步骤 1-2: 运行 DualHead U-Net

        Args:
            volume: (1, 1, D, H, W) 输入体积

        Returns:
            prob_map: (D, H, W) numpy，概率图
            normal_map: (3, D, H, W) numpy，法线图
        """
        x = volume.to(self.device)
        seg_logits, normals = self.model(x)

        # Sigmoid → 概率图
        prob_map = torch.sigmoid(seg_logits).squeeze().cpu().numpy()  # (D, H, W)

        # 法线已经是 Tanh 归一化后的结果
        normal_map = normals.squeeze(0).cpu().numpy()  # (3, D, H, W)

        return prob_map, normal_map

    def _run_graph_solver(self, prob_map, normal_map):
        """
        步骤 3-5: 图构建 → Winding Number 求解 → 阈值化

        Returns:
            winding_mask: (D, H, W) numpy，binary mask
        """
        D, H, W = prob_map.shape

        # 步骤 3: 构建稀疏图
        adjacency, node_coords, node_index_map = build_sparse_graph(
            prob_map, normal_map, threshold=self.prob_threshold
        )

        if adjacency.shape[0] == 0:
            print("[Pipeline] 警告: 无有效节点，返回空 mask")
            return np.zeros((D, H, W), dtype=np.float32)

        # 步骤 4: 自动分配种子 + 求解
        seeds = auto_assign_seeds(node_coords, (D, H, W))

        if len(seeds) == 0:
            print("[Pipeline] 警告: 无种子节点，回退到概率阈值化")
            return (prob_map > self.prob_threshold).astype(np.float32)

        winding_field = solve_winding_number(
            adjacency, seeds, use_cupy=self.use_cupy
        )

        # 步骤 5: 阈值化
        winding_mask = cut_mesh(
            winding_field, node_coords, (D, H, W),
            threshold=self.winding_threshold
        )

        return winding_mask

    def _porosity_injection(self, winding_mask, prob_map):
        """
        步骤 6: Porosity Injection

        恢复 Winding Mask 可能遗漏的微小孔洞和薄结构。
        Final_Mask = Winding_Mask & (Prob_Map > porosity_threshold)

        在 Winding Mask 的基础上，用更宽松的概率阈值
        补回被几何求解器平滑掉的拓扑细节（对 TopoScore 至关重要）。
        """
        prob_mask = (prob_map > self.porosity_threshold).astype(np.float32)

        # 交集: 保留 Winding 认为的"内部" 且 概率支持的区域
        # 并集补充: 在 Winding 外但概率高的区域也保留（恢复孔洞）
        final_mask = np.maximum(winding_mask, prob_mask)

        # 更保守的版本（纯交集）：
        # final_mask = winding_mask * prob_mask

        winding_only = (winding_mask > 0).sum()
        prob_only = (prob_mask > 0).sum()
        final_count = (final_mask > 0).sum()

        print(f"[Porosity] Winding: {winding_only}, "
              f"Prob(>{self.porosity_threshold}): {prob_only}, "
              f"Final: {final_count}")

        return final_mask

    def process_chunk(self, volume: torch.Tensor):
        """
        处理单个 chunk 的完整推理流程

        Args:
            volume: (1, 1, D, H, W) 或 (1, D, H, W) 输入体积

        Returns:
            final_mask: (D, H, W) numpy，最终 binary mask
            timings: dict，各步骤耗时
        """
        # 确保形状正确
        if volume.dim() == 4:
            volume = volume.unsqueeze(0)  # (1, D, H, W) → (1, 1, D, H, W)

        timings = {}
        total_start = time.time()

        # 步骤 1-2: U-Net 推理
        t0 = time.time()
        prob_map, normal_map = self._run_unet(volume)
        timings["unet_inference"] = time.time() - t0
        print(f"[Timer] U-Net 推理: {timings['unet_inference']:.2f}s")

        # 步骤 3-5: 图构建 + Winding Number 求解
        t0 = time.time()
        winding_mask = self._run_graph_solver(prob_map, normal_map)
        timings["graph_solver"] = time.time() - t0
        print(f"[Timer] 图+求解: {timings['graph_solver']:.2f}s")

        # 步骤 6: Porosity Injection
        t0 = time.time()
        final_mask = self._porosity_injection(winding_mask, prob_map)
        timings["porosity"] = time.time() - t0
        print(f"[Timer] Porosity: {timings['porosity']:.4f}s")

        timings["total"] = time.time() - total_start
        print(f"[Timer] 总耗时: {timings['total']:.2f}s")

        # 性能检查: < 10 分钟
        if timings["total"] > 600:
            print(f"⚠️ 警告: 单 chunk 耗时 {timings['total']:.0f}s > 600s，"
                  f"可能超时！")

        return final_mask, timings

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
    ):
        """
        批量处理目录中的所有 .tif chunk

        Args:
            input_dir: 输入 .tif 文件目录
            output_dir: 输出 mask 保存目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset = TifChunkDataset(input_dir, normalize=True)
        total_chunks = len(dataset)
        all_timings = []

        print(f"\n{'='*60}")
        print(f"  Hybrid Chimera MVP - 批量推理")
        print(f"  Chunks: {total_chunks}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        for i in range(total_chunks):
            print(f"\n--- Chunk {i+1}/{total_chunks}: "
                  f"{dataset.get_file_path(i)} ---")

            volume = dataset[i]  # (1, D, H, W)
            mask, timings = self.process_chunk(volume)
            all_timings.append(timings)

            # 保存结果
            input_name = Path(dataset.get_file_path(i)).stem
            output_file = output_path / f"{input_name}_mask.tif"
            tifffile.imwrite(str(output_file), mask.astype(np.uint8))
            print(f"[Save] → {output_file}")

        # 总结
        total_time = sum(t["total"] for t in all_timings)
        avg_time = total_time / max(len(all_timings), 1)

        print(f"\n{'='*60}")
        print(f"  推理完成!")
        print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
        print(f"  平均每 chunk: {avg_time:.1f}s")
        print(f"  预计全量推理 (假设 50 chunks): {avg_time * 50 / 60:.1f} 分钟")
        print(f"  9 小时限制内可处理: {int(9 * 3600 / max(avg_time, 0.1))} chunks")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Chimera MVP - Vesuvius 推理流水线"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="输入 .tif 文件目录"
    )
    parser.add_argument(
        "--output_dir", type=str, default="20_src/20_outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型权重文件路径 (.pth)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="推理设备"
    )
    parser.add_argument(
        "--prob_threshold", type=float, default=0.5,
        help="图构建概率阈值"
    )
    parser.add_argument(
        "--porosity_threshold", type=float, default=0.4,
        help="Porosity Injection 概率阈值"
    )
    parser.add_argument(
        "--winding_threshold", type=float, default=0.5,
        help="Winding Number 阈值"
    )
    parser.add_argument(
        "--use_cupy", action="store_true",
        help="使用 CuPy GPU 加速求解器"
    )
    parser.add_argument(
        "--n_filters", type=int, default=16,
        help="模型基础通道数"
    )

    args = parser.parse_args()

    pipeline = HybridChimeraPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
        prob_threshold=args.prob_threshold,
        porosity_threshold=args.porosity_threshold,
        winding_threshold=args.winding_threshold,
        use_cupy=args.use_cupy,
        n_filters=args.n_filters,
    )

    pipeline.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

```

---
## File: train.py
```py
"""
Vesuvius Challenge - Hybrid Chimera 训练引擎 (Phase 5)

Patch-based 3D 训练循环，支持：
- AMP 混合精度 (FP16)
- Random Crop 3D + 数据增强
- ChimeraLoss (Dice + Normal Cosine)
- tqdm 进度条
- 每 epoch 可视化输出 (PNG 对比图 + TIF mask)
- 验证 Dice 监控 + Best Model 保存

用法:
    # 快速测试 (5 个 chunk, 2 个 epoch)
    python 20_src/train.py --max_chunks 5 --epochs 2

    # 完整训练
    python 20_src/train.py --epochs 50 --batch_size 4 --lr 1e-3

    # 从 checkpoint 恢复
    python 20_src/train.py --resume 20_src/output/best_model.pth --epochs 50
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import tifffile

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from importlib import import_module

# 模型
model_mod = import_module("20_src.20_model.dual_unet")
DualHeadResUNet3D = model_mod.DualHeadResUNet3D

# 损失函数
loss_mod = import_module("20_src.20_model.chimera_loss")
ChimeraLoss = loss_mod.ChimeraLoss

# 数据集
dataset_mod = import_module("20_src.20_data.dataset")
VesuviusTrainDataset = dataset_mod.VesuviusTrainDataset

# 变换
transforms_mod = import_module("20_src.20_data.transforms")
RandomCrop3D = transforms_mod.RandomCrop3D
RandomFlipRotate3D = transforms_mod.RandomFlipRotate3D
Compose3D = transforms_mod.Compose3D


# ===== 工具函数 =====

def get_gpu_stats():
    """获取 GPU 显存使用信息"""
    if not torch.cuda.is_available():
        return "CPU"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"GPU:{reserved:.1f}G/{total:.1f}G"


def compute_dice(pred_logits, targets, threshold=0.5):
    """计算 Dice 系数（评估指标）"""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    smooth = 1e-6
    intersection = (pred * targets).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + targets.sum() + smooth)
    return dice.item()


def format_time(seconds):
    """格式化时间"""
    td = timedelta(seconds=int(seconds))
    return str(td)


# ===== 可视化工具 =====

def save_epoch_visualization(
    model, val_loader, device, run_dir, epoch, criterion
):
    """
    每个 epoch 结束后保存可视化对比：
    1. PNG 对比图：中间 slice 的 image / GT / prediction 三列对比
    2. TIF mask：预测结果 3D volume
    """
    model.eval()

    # 从验证集寻找一个包含正样本的 batch 进行可视化
    target_images = None
    target_labels = None
    
    try:
        # 尝试遍历验证集寻找有墨水的 patch
        for images, labels in val_loader:
            if labels.sum() > 0:
                target_images = images
                target_labels = labels
                break
        
        # 如果没找到（极其罕见），就退回到第一个 batch
        if target_images is None:
            target_images, target_labels = next(iter(val_loader))
            
    except StopIteration:
        return

    images = target_images.to(device)
    labels = target_labels.to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            seg_logits, normals = model(images)
    
    # 取第一个样本
    pred_prob = torch.sigmoid(seg_logits[0, 0]).cpu().numpy()    # (D, H, W)
    pred_mask = (pred_prob > 0.5).astype(np.uint8)               # 二值 mask
    gt_mask = labels[0, 0].cpu().numpy()                          # (D, H, W)
    img_vol = images[0, 0].cpu().numpy()                          # (D, H, W)

    D, H, W = img_vol.shape

    # === 1. 保存 TIF mask ===
    tif_dir = run_dir / "epoch_masks"
    tif_dir.mkdir(exist_ok=True)
    tif_path = tif_dir / f"epoch{epoch+1:03d}_pred_mask.tif"
    tifffile.imwrite(str(tif_path), pred_mask)

    # === 2. 保存 PNG 对比图 ===
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt

        vis_dir = run_dir / "epoch_vis"
        vis_dir.mkdir(exist_ok=True)

        # 取 3 个正交 slice（中间位置）
        slices = {
            'Axial (z-mid)': (img_vol[D//2], gt_mask[D//2], pred_prob[D//2], pred_mask[D//2]),
            'Coronal (y-mid)': (img_vol[:, H//2], gt_mask[:, H//2], pred_prob[:, H//2], pred_mask[:, H//2]),
            'Sagittal (x-mid)': (img_vol[:, :, W//2], gt_mask[:, :, W//2], pred_prob[:, :, W//2], pred_mask[:, :, W//2]),
        }

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Epoch {epoch+1} | Dice: {compute_dice(seg_logits[0:1], labels[0:1]):.4f}',
                     fontsize=16, fontweight='bold')

        for row_idx, (plane_name, (img_s, gt_s, prob_s, mask_s)) in enumerate(slices.items()):
            # 列 1: Input CT
            axes[row_idx, 0].imshow(img_s, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 0].set_title(f'{plane_name}\nInput CT')
            axes[row_idx, 0].axis('off')

            # 列 2: Ground Truth
            axes[row_idx, 1].imshow(gt_s, cmap='Reds', vmin=0, vmax=1, alpha=0.8)
            axes[row_idx, 1].set_title('Ground Truth')
            axes[row_idx, 1].axis('off')

            # 列 3: Prediction (概率图)
            axes[row_idx, 2].imshow(prob_s, cmap='hot', vmin=0, vmax=1)
            axes[row_idx, 2].set_title('Pred Prob')
            axes[row_idx, 2].axis('off')

            # 列 4: Overlay (CT + Prediction 叠加)
            axes[row_idx, 3].imshow(img_s, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 3].imshow(mask_s, cmap='Reds', alpha=0.4)
            axes[row_idx, 3].set_title('Overlay')
            axes[row_idx, 3].axis('off')

        plt.tight_layout()
        png_path = vis_dir / f"epoch{epoch+1:03d}_comparison.png"
        plt.savefig(str(png_path), dpi=120, bbox_inches='tight')
        plt.close(fig)

        print(f"  📸 2D 对比: {png_path.name} | 🗂️ Mask: {tif_path.name}")

    except ImportError:
        print(f"  🗂️ Mask TIF: {tif_path.name} (matplotlib 不可用，跳过 PNG)")

    # === 3. 3D Volume Rendering (PyVista offscreen) ===
    try:
        import pyvista as pv
        import matplotlib.colors as mcolors

        pv.OFF_SCREEN = True
        vis_dir = run_dir / "epoch_vis"
        vis_dir.mkdir(exist_ok=True)

        p = pv.Plotter(shape=(1, 2), window_size=(1200, 600), off_screen=True)

        # 左: GT mask (绿色)
        p.subplot(0, 0)
        p.add_text(f"GT Mask (Epoch {epoch+1})", font_size=10)
        if gt_mask.sum() > 0:
            gt_grid = pv.wrap(gt_mask.astype(np.float32))
            gt_cmap = mcolors.LinearSegmentedColormap.from_list("gt", ["black", "lime"])
            p.add_volume(gt_grid, cmap=gt_cmap,
                         opacity=[0, 0.0, 0.1, 0.0, 0.9, 0.5, 1.0, 0.5],
                         blending="composite", show_scalar_bar=False)
        p.add_bounding_box()

        # 右: Prediction mask (红色)
        p.subplot(0, 1)
        p.add_text(f"Pred Mask (Epoch {epoch+1})", font_size=10)
        if pred_mask.sum() > 0:
            pred_grid = pv.wrap(pred_mask.astype(np.float32))
            pred_cmap = mcolors.LinearSegmentedColormap.from_list("pred", ["black", "red"])
            p.add_volume(pred_grid, cmap=pred_cmap,
                         opacity=[0, 0.0, 0.1, 0.0, 0.9, 0.5, 1.0, 0.5],
                         blending="composite", show_scalar_bar=False)
        p.add_bounding_box()

        p.link_views()

        png_3d_path = vis_dir / f"epoch{epoch+1:03d}_3d_comparison.png"
        p.screenshot(str(png_3d_path))
        p.close()

        print(f"  🧊 3D 对比: {png_3d_path.name}")

    except Exception as e:
        print(f"  ⚠️ 3D 渲染跳过: {e}")



# ===== 训练循环 =====

def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs
):
    """训练一个 epoch（带 tqdm 进度条）"""
    model.train()

    total_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    total_normal_loss = 0.0
    total_dice_score = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Train E{epoch+1}/{total_epochs}",
        ncols=120,
        leave=True,
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)  # (B, 1, D, H, W)
        labels = labels.to(device, non_blocking=True)  # (B, 1, D, H, W)

        optimizer.zero_grad(set_to_none=True)

        # AMP 前向传播
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            seg_logits, normals = model(images)
            loss_total, loss_dice, loss_bce, loss_normal = criterion(seg_logits, normals, labels)

        # AMP 反向传播
        if device.type == 'cuda':
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 统计
        dice_score = compute_dice(seg_logits.detach(), labels)
        total_loss += loss_total.item()
        total_dice_loss += loss_dice.item()
        total_bce_loss += loss_bce.item()
        total_normal_loss += loss_normal.item()
        total_dice_score += dice_score
        num_batches += 1

        # Debug "Playing Dead"
        if num_batches % 50 == 0:
            pred_sum = (torch.sigmoid(seg_logits) > 0.5).float().sum()
            target_sum = labels.sum()
            print(f"\n[DEBUG] Batch {num_batches}: Pred_Pixels={pred_sum.item()}, GT_Pixels={target_sum.item()}")

        # 更新 tqdm
        avg_loss = total_loss / num_batches
        avg_dice = total_dice_score / num_batches
        pbar.set_postfix({
            'loss': f'{avg_loss:.2f}',
            'bce': f'{total_bce_loss/num_batches:.2f}',
            'dice': f'{avg_dice:.2f}',
            'norm': f'{total_normal_loss/num_batches:.2f}',
            'gpu': get_gpu_stats(),
        })

    pbar.close()

    # epoch 统计
    avg_loss = total_loss / max(num_batches, 1)
    avg_dice_loss = total_dice_loss / max(num_batches, 1)
    avg_bce_loss = total_bce_loss / max(num_batches, 1)
    avg_normal_loss = total_normal_loss / max(num_batches, 1)
    avg_dice_score = total_dice_score / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "dice_loss": avg_dice_loss,
        "bce_loss": avg_bce_loss,
        "normal_loss": avg_normal_loss,
        "dice_score": avg_dice_score,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """验证一个 epoch（带 tqdm）"""
    model.eval()

    total_loss = 0.0
    total_dice_score = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Val   E{epoch+1}/{total_epochs}",
        ncols=120,
        leave=True,
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            seg_logits, normals = model(images)
            loss_total, loss_dice, loss_bce, loss_normal = criterion(seg_logits, normals, labels)

        dice_score = compute_dice(seg_logits, labels)
        total_loss += loss_total.item()
        total_dice_score += dice_score
        num_batches += 1

        pbar.set_postfix({
            'val_loss': f'{total_loss/num_batches:.4f}',
            'val_dice': f'{total_dice_score/num_batches:.4f}',
        })

    pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice_score / max(num_batches, 1)

    return {"val_loss": avg_loss, "val_dice": avg_dice}


# ===== 主训练函数 =====

def main(args):
    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"train_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  🔥 Hybrid Chimera 训练引擎")
    print(f"  设备: {device}")
    print(f"  Patch 大小: {args.crop_size}³")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  λ_normal: {args.lambda_normal}")
    print(f"  输出: {run_dir}")
    print(f"{'='*70}\n")

    # ===== 数据 =====
    # 增强变换：仅 FlipRotate，Crop 已内置到 Dataset 的 memmap __getitem__
    aug_transform = RandomFlipRotate3D(flip_prob=0.5, rotate_prob=0.5)

    full_dataset = VesuviusTrainDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        crop_size=args.crop_size,
        transform=aug_transform,
        samples_per_volume=args.samples_per_volume,
        cache_size=args.cache_size,
        max_files=args.max_chunks,
    )


    # 按 8:2 拆分 train/val
    total_len = len(full_dataset)
    val_len = max(1, int(total_len * 0.2))
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"[Data] Train: {train_len} samples, Val: {val_len} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ===== 模型 =====
    model = DualHeadResUNet3D(in_channels=1, n_filters=args.n_filters).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] 参数量: {params:,}")

    # 恢复 checkpoint
    start_epoch = 0
    best_dice = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            best_dice = ckpt.get("best_dice", 0.0)
            print(f"[Resume] 从 epoch {start_epoch} 恢复, best_dice={best_dice:.4f}")
        else:
            if any(k.startswith("model.") for k in ckpt.keys()):
                ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt, strict=False)
            print(f"[Resume] 加载权重（无 epoch 信息）")

    # ===== 优化器 & 调度器 =====
    criterion = ChimeraLoss(
        lambda_normal=args.lambda_normal,
        pos_weight=args.pos_weight,
    ).to(device)
    print(f"[Loss] pos_weight={args.pos_weight}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ===== 训练循环 =====
    print(f"\n{'='*70}")
    print(f"  开始训练 (epoch {start_epoch+1} → {args.epochs})")
    print(f"{'='*70}\n")

    history = []
    t_total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t_ep_start = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"\n--- Epoch {epoch+1}/{args.epochs} | LR: {lr_now:.6f} ---")

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.epochs,
        )

        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, args.epochs)

        # 调度器
        scheduler.step()

        # 每 epoch 可视化输出
        save_epoch_visualization(
            model, val_loader, device, run_dir, epoch, criterion
        )

        # 记录
        ep_time = time.time() - t_ep_start
        epoch_info = {
            "epoch": epoch + 1,
            "lr": lr_now,
            "time": ep_time,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_info)

        # 打印 epoch 总结
        print(
            f"\n  📊 Epoch {epoch+1} 总结:"
            f"\n     Train - Loss: {train_metrics['loss']:.4f} | "
            f"BCE: {train_metrics['bce_loss']:.4f} | "
            f"Dice: {train_metrics['dice_score']:.4f} | "
            f"Normal: {train_metrics['normal_loss']:.4f}"
            f"\n     Val   - Loss: {val_metrics['val_loss']:.4f} | "
            f"Dice: {val_metrics['val_dice']:.4f}"
            f"\n     Time: {format_time(ep_time)} | LR: {lr_now:.6f}"
        )

        # 保存 best model
        if val_metrics["val_dice"] > best_dice:
            best_dice = val_metrics["val_dice"]
            best_path = run_dir / "best_model.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "args": vars(args),
            }, str(best_path))
            print(f"  🏆 New Best Dice: {best_dice:.4f} → {best_path.name}")

        # 定期保存 checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = run_dir / f"checkpoint_epoch{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "args": vars(args),
            }, str(ckpt_path))
            print(f"  💾 Checkpoint: {ckpt_path.name}")

    # ===== 最终报告 =====
    total_time = time.time() - t_total_start

    print(f"\n{'='*70}")
    print(f"  训练完成!")
    print(f"  总 Epochs: {args.epochs - start_epoch}")
    print(f"  总耗时: {format_time(total_time)}")
    print(f"  最佳 Val Dice: {best_dice:.4f}")
    print(f"  输出目录: {run_dir}")
    print(f"{'='*70}\n")

    # 保存训练历史
    history_path = run_dir / "training_history.json"
    with open(str(history_path), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"  📈 训练历史: {history_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Chimera 训练引擎")

    # 数据参数
    parser.add_argument("--image_dir", type=str,
                        default="data/vesuvius-challenge-surface-detection/train_images")
    parser.add_argument("--label_dir", type=str,
                        default="data/vesuvius-challenge-surface-detection/train_labels")
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="最多使用多少个 chunk（调试用）")
    parser.add_argument("--samples_per_volume", type=int, default=4,
                        help="每个体积每 epoch 采集几个 patch")
    parser.add_argument("--cache_size", type=int, default=8,
                        help="LRU 缓存体积数量")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=64,
                        help="Random Crop 3D 尺寸")
    parser.add_argument("--lambda_normal", type=float, default=1.0,
                        help="法线损失权重")
    parser.add_argument("--pos_weight", type=float, default=10.0,
                        help="BCE 正样本权重 (越大越强调 Recall，越小越强调 Precision)")
    parser.add_argument("--n_filters", type=int, default=32,
                        help="模型基础滤波器数")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 工作进程数")

    # 保存参数
    parser.add_argument("--output_dir", type=str, default="20_src/output")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每几个 epoch 保存 checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")

    # 设备
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    main(args)

```

---
## File: verify_data.py
```py
"""
Vesuvius Challenge - 数据验证脚本

功能：检查预处理后的 NPY 标签文件的稀疏性，确保模型训练在正确的目标上。
目的：防止"训练在实心 Mask 上"的致命错误再次发生。

正确的标签应该非常稀疏（纸草表面 ~5%），而不是实心块。

用法:
    python 20_src/verify_data.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def verify_labels(label_dir: str, num_samples: int = 5, save_png: bool = True):
    """
    验证标签文件的内容和稀疏性

    Args:
        label_dir: NPY 标签目录
        num_samples: 随机抽样检查的文件数量
        save_png: 是否保存切片可视化 PNG
    """
    label_path = Path(label_dir)
    if not label_path.exists():
        print(f"❌ 标签目录不存在: {label_path}")
        sys.exit(1)

    npy_files = sorted(label_path.glob("*.npy"))
    if len(npy_files) == 0:
        print(f"❌ 未找到 .npy 文件: {label_path}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  🔬 Vesuvius 标签数据验证")
    print(f"  目录: {label_path}")
    print(f"  文件数: {len(npy_files)}")
    print(f"{'='*60}\n")

    # 随机抽样
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(npy_files), size=min(num_samples, len(npy_files)), replace=False)
    sample_files = [npy_files[i] for i in sorted(sample_indices)]

    all_sparsities = []
    fatal_errors = []

    for f in sample_files:
        vol = np.load(str(f), mmap_mode='r')
        total = vol.size
        unique_vals = np.unique(vol)

        # 统计各值占比
        count_0 = np.sum(vol == 0)
        count_1 = np.sum(vol == 1)
        count_2 = np.sum(vol == 2)

        # 计算稀疏度（val=1 是目标）
        surface_ratio = count_1 / total
        ignore_ratio = count_2 / total
        bg_ratio = count_0 / total

        all_sparsities.append(surface_ratio)

        # 状态判断
        status = "✅"
        if surface_ratio > 0.30:
            status = "⚠️ WARNING"
        if surface_ratio > 0.90:
            status = "❌ FATAL"
            fatal_errors.append(f.name)

        print(f"{status} {f.name}:")
        print(f"    shape={vol.shape}, dtype={vol.dtype}, unique={unique_vals}")
        print(f"    背景(0): {bg_ratio*100:.1f}% | "
              f"表面(1): {surface_ratio*100:.1f}% | "
              f"忽略(2): {ignore_ratio*100:.1f}%")
        print()

    # 总结
    avg_sparsity = np.mean(all_sparsities)
    print(f"{'='*60}")
    print(f"  📊 汇总统计")
    print(f"  平均表面占比 (val=1): {avg_sparsity*100:.2f}%")
    print(f"{'='*60}")

    if avg_sparsity < 0.10:
        print(f"\n  ✅ 数据正常！表面标签稀疏度合理 ({avg_sparsity*100:.1f}% < 10%)")
        print(f"  → 模型应该学习画'稀疏的线条'，而非'实心方块'")
    elif avg_sparsity < 0.30:
        print(f"\n  ⚠️ 注意：表面占比偏高 ({avg_sparsity*100:.1f}%)，但可能仍然合理")
    else:
        print(f"\n  ❌ 严重问题！表面占比过高 ({avg_sparsity*100:.1f}%)，"
              f"可能仍在使用错误的标签！")

    if fatal_errors:
        print(f"\n  ❌ FATAL: 以下文件的正样本比例 > 90%: {fatal_errors}")
        sys.exit(1)

    # 生成可视化
    if save_png:
        sample_vol = np.load(str(sample_files[0]))
        mid_z = sample_vol.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始标签 (0, 1, 2)
        axes[0].imshow(sample_vol[mid_z], cmap='tab10', vmin=0, vmax=2)
        axes[0].set_title(f'原始标签 (z={mid_z})\n0=背景, 1=表面, 2=忽略')

        # 只看 val=1 (表面)
        surface = (sample_vol[mid_z] == 1).astype(np.float32)
        axes[1].imshow(surface, cmap='hot', vmin=0, vmax=1)
        sr = np.sum(surface) / surface.size * 100
        axes[1].set_title(f'表面 (val=1)\n稀疏度: {sr:.1f}%')

        # 只看 val=2 (忽略区域)
        ignore = (sample_vol[mid_z] == 2).astype(np.float32)
        axes[2].imshow(ignore, cmap='Blues', vmin=0, vmax=1)
        ir = np.sum(ignore) / ignore.size * 100
        axes[2].set_title(f'忽略区域 (val=2)\n占比: {ir:.1f}%')

        for ax in axes:
            ax.axis('off')

        fig.suptitle(f'文件: {sample_files[0].name}', fontsize=12, fontweight='bold')
        plt.tight_layout()

        out_path = Path("20_src/output/verification_slice.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  📸 可视化已保存: {out_path}")


if __name__ == "__main__":
    verify_labels("data/vesuvius-challenge-surface-detection/train_labels_npy")

```

---
## File: winding_solver.py
```py
"""
Vesuvius Challenge - Winding Number Solver (MVP)

替代 ThaumatoAnakalyptor C++ 求解器的纯 Python 实现。

核心思想：
1. 从稀疏邻接图构建 Graph Laplacian L = D - A
2. 设定 Dirichlet 边界条件（seed 节点: 内部=1, 外部=0）
3. 求解 L_ff * u_f = -L_fs * u_s（热扩散问题）
4. 阈值化 winding number 场生成最终 binary mask

支持 GPU (CuPy) 和 CPU (SciPy) 双路径。
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from typing import Dict, Optional, Tuple


def solve_winding_number(
    adjacency: sparse.csr_matrix,
    seeds: Dict[int, float],
    use_cupy: bool = False,
    tol: float = 1e-6,
    maxiter: int = 5000,
) -> np.ndarray:
    """
    求解 Winding Number 标量场

    通过求解 Laplacian 线性系统 + Dirichlet 边界条件，
    将 seed 节点的标量值扩散到整个连通图上。

    Args:
        adjacency: 稀疏邻接矩阵 (N, N)，来自 build_sparse_graph
        seeds: 边界条件字典 {节点索引: 值}
               例如 {0: 0.0, 10: 1.0} → 节点 0 是外部，节点 10 是内部
        use_cupy: 是否使用 CuPy GPU 加速，默认 False
        tol: 求解器收敛容差
        maxiter: 最大迭代次数

    Returns:
        u: np.ndarray，形状 (N,)，每个节点的 winding number 值
           u ≈ 1.0 → 内部，u ≈ 0.0 → 外部

    Raises:
        ValueError: 无效输入时抛出
    """
    N = adjacency.shape[0]

    if N == 0:
        return np.array([], dtype=np.float64)

    if len(seeds) == 0:
        print("[solve_winding_number] 警告: 无 seed 节点，返回全零解")
        return np.zeros(N, dtype=np.float64)

    # --- 连通性预检 (Diagnostics) ---
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(adjacency, connection='strong', directed=False)
    
    if n_components > 1:
        print(f"[solve_winding_number] ⚠️ 警告: 图包含 {n_components} 个不连通的子图 (拓扑断裂风险)")
        
        # 检查是否有子图完全没有种子
        seed_mask = np.zeros(N, dtype=bool)
        for idx in seeds.keys():
            seed_mask[idx] = True
            
        # 统计每个 component 是否有 seed
        components_with_seeds = 0
        for k in range(n_components):
            comp_nodes = np.where(labels == k)[0]
            if np.any(seed_mask[comp_nodes]):
                components_with_seeds += 1
                
        if components_with_seeds < n_components:
            print(f"  🛑 致命: {n_components - components_with_seeds} 个子图完全没有 Seed，将导致无解或全0！")
            print("  建议: 检查 U-Net 预测是否过度破碎，或改进 Seed 分配策略")

    # --- 步骤 1: 构建 Graph Laplacian ---
    degree = np.array(adjacency.sum(axis=1)).flatten()
    # 防止孤立节点（度为 0）导致奇异矩阵
    degree = np.maximum(degree, 1e-10)
    D = sparse.diags(degree, format='csr')
    L = D - adjacency  # Laplacian = D - A

    # --- 步骤 2: 分离 seed (s) 和 free (f) 节点 ---
    seed_indices = sorted(seeds.keys())
    seed_values = np.array([seeds[i] for i in seed_indices], dtype=np.float64)

    all_indices = np.arange(N)
    seed_set = set(seed_indices)
    free_indices = np.array([i for i in all_indices if i not in seed_set])

    if len(free_indices) == 0:
        # 所有节点都是 seed，直接赋值
        u = np.zeros(N, dtype=np.float64)
        for idx, val in seeds.items():
            u[idx] = val
        return u

    # --- 步骤 3: 提取子矩阵 ---
    # L_ff: free-free 子矩阵
    # L_fs: free-seed 子矩阵
    seed_arr = np.array(seed_indices)

    L_ff = L[np.ix_(free_indices, free_indices)]
    L_fs = L[np.ix_(free_indices, seed_arr)]

    # 右端项: b = -L_fs * u_s
    rhs = -L_fs.dot(seed_values)

    print(f"[solve_winding_number] 求解线性系统: "
          f"{len(free_indices)} 个自由节点, {len(seed_indices)} 个种子节点")

    # --- 步骤 4: 求解 L_ff * u_f = rhs ---
    if use_cupy:
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse
            import cupyx.scipy.sparse.linalg as cp_linalg

            # 转移到 GPU
            L_ff_gpu = cp_sparse.csr_matrix(L_ff)
            rhs_gpu = cp.array(rhs)

            # 共轭梯度求解
            u_f_gpu, info = cp_linalg.cg(L_ff_gpu, rhs_gpu, atol=tol, maxiter=maxiter)

            if info != 0:
                print(f"[solve_winding_number] CuPy CG 未收敛 (info={info})，"
                      f"回退到 CPU")
                raise RuntimeError("CuPy CG 未收敛")

            u_f = cp.asnumpy(u_f_gpu)
            print("[solve_winding_number] 使用 CuPy GPU 求解完成")

        except (ImportError, RuntimeError) as e:
            print(f"[solve_winding_number] CuPy 不可用或失败: {e}，回退到 CPU")
            use_cupy = False

    if not use_cupy:
        # CPU 路径: SciPy 共轭梯度
        u_f, info = sp_linalg.cg(L_ff, rhs, atol=tol, maxiter=maxiter)

        if info != 0:
            print(f"[solve_winding_number] SciPy CG 收敛状态: info={info}")
            if info > 0:
                print("  → 未在最大迭代次数内收敛，结果可能不精确")
            else:
                print("  → 输入矩阵存在问题")

        print("[solve_winding_number] 使用 SciPy CPU 求解完成")

    # --- 步骤 5: 组装完整解向量 ---
    u = np.zeros(N, dtype=np.float64)

    # 填入 seed 值
    for idx, val in seeds.items():
        u[idx] = val

    # 填入自由节点的解
    u[free_indices] = u_f

    # Clip 到合理范围
    u = np.clip(u, 0.0, 1.0)

    print(f"[solve_winding_number] 解的范围: [{u.min():.4f}, {u.max():.4f}]")

    return u


def cut_mesh(
    winding_field: np.ndarray,
    node_coords: np.ndarray,
    volume_shape: Tuple[int, int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    将 winding number 标量场映射回体积空间，生成 binary mask

    Args:
        winding_field: 每个节点的 winding number，形状 (N,)
        node_coords: 节点坐标，形状 (N, 3)，每行 (d, h, w)
        volume_shape: 输出体积的形状 (D, H, W)
        threshold: 阈值，u >= threshold → 1（内部），默认 0.5

    Returns:
        mask: binary mask，形状 (D, H, W)，dtype=float32
    """
    D, H, W = volume_shape
    mask = np.zeros((D, H, W), dtype=np.float32)

    if len(winding_field) == 0:
        return mask

    # 将每个节点的 winding number 写入对应位置
    for i, (d, h, w) in enumerate(node_coords):
        d, h, w = int(d), int(h), int(w)
        if 0 <= d < D and 0 <= h < H and 0 <= w < W:
            mask[d, h, w] = winding_field[i]

    # 阈值化
    binary_mask = (mask >= threshold).astype(np.float32)

    num_inside = int(binary_mask.sum())
    total = D * H * W
    print(f"[cut_mesh] 内部体素: {num_inside} / {total} "
          f"(占比 {num_inside / total * 100:.1f}%)")

    return binary_mask


def auto_assign_seeds(
    node_coords: np.ndarray,
    volume_shape: Tuple[int, int, int],
    boundary_thickness: int = 2,
) -> Dict[int, float]:
    """
    自动分配 seed 节点的辅助函数

    策略：
    - 靠近体积边界的节点 → 外部 (u=0)
    - 靠近体积中心的节点 → 内部 (u=1)

    Args:
        node_coords: 节点坐标，形状 (N, 3)
        volume_shape: 体积形状 (D, H, W)
        boundary_thickness: 边界层厚度（体素数），默认 2

    Returns:
        seeds: {节点索引: 值} 字典
    """
    D, H, W = volume_shape
    seeds = {}

    center = np.array([D / 2, H / 2, W / 2])

    for i, coord in enumerate(node_coords):
        d, h, w = coord

        # 判断是否在边界层
        is_boundary = (
            d < boundary_thickness or d >= D - boundary_thickness or
            h < boundary_thickness or h >= H - boundary_thickness or
            w < boundary_thickness or w >= W - boundary_thickness
        )

        if is_boundary:
            seeds[i] = 0.0  # 外部

    # 找到最靠近中心的节点作为内部 seed
    if len(node_coords) > 0:
        distances = np.linalg.norm(node_coords - center, axis=1)
        center_node = int(np.argmin(distances))
        if center_node not in seeds:
            seeds[center_node] = 1.0  # 内部

    print(f"[auto_assign_seeds] 自动分配了 {len(seeds)} 个种子节点 "
          f"(外部: {sum(1 for v in seeds.values() if v == 0.0)}, "
          f"内部: {sum(1 for v in seeds.values() if v == 1.0)})")

    return seeds


if __name__ == "__main__":
    print("=== Winding Number Solver 自测 ===")

    # 导入 graph_builder（使用 importlib 处理数字开头的模块名）
    import sys
    import os
    # 添加项目根目录到 sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from importlib import import_module
    gb = import_module("20_src.graph_builder")

    # --- 创建合成数据 ---
    D, H, W = 8, 8, 8
    prob_map = np.zeros((D, H, W), dtype=np.float32)
    prob_map[1:7, 1:7, 1:7] = 0.8  # 中心 6x6x6 区域有效

    normal_map = np.zeros((3, D, H, W), dtype=np.float32)
    normal_map[2, :, :, :] = 1.0  # 法线全部指向 z

    # --- 构建图 ---
    adj, coords, idx_map = gb.build_sparse_graph(prob_map, normal_map)
    print(f"图: {len(coords)} 节点, {adj.nnz} 边")

    # --- 自动分配 seeds ---
    seeds = auto_assign_seeds(coords, (D, H, W))
    print(f"Seeds: {len(seeds)} 个")

    # --- 求解 winding number ---
    u = solve_winding_number(adj, seeds)
    print(f"解向量长度: {len(u)}")
    print(f"解的范围: [{u.min():.4f}, {u.max():.4f}]")

    # --- 生成 mask ---
    mask = cut_mesh(u, coords, (D, H, W), threshold=0.5)
    print(f"Mask 形状: {mask.shape}")
    print(f"Mask 非零: {mask.sum():.0f}")

    # 基本断言
    assert len(u) == len(coords), "解向量长度应等于节点数"
    assert mask.shape == (D, H, W), f"Mask 形状错误: {mask.shape}"
    assert u.min() >= 0.0 and u.max() <= 1.0, "解应在 [0, 1] 范围内"

    print("✓ 所有测试通过！")

```

---
## File: 20_data\__init__.py
```py
# 20_data: 数据加载模块
from .dataset import TifChunkDataset

```

---
## File: 20_data\dataset.py
```py
"""
Vesuvius Challenge - 3D TIF Chunk 数据加载器

包含两个 Dataset：
- TifChunkDataset:       推理用，整体加载
- VesuviusTrainDataset:  训练用，NPY mmap 零拷贝 / TIF LRU 缓存

核心优化（NPY 模式）：
  np.load(mmap_mode='r') 将文件映射到虚拟内存，
  只有在 slice 时才触发缺页中断读取对应的字节。
  96³ uint8 patch ≈ 0.88MB，IO 时间 < 1ms。
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Union
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile


class TifChunkDataset(Dataset):
    """
    3D TIF Chunk 数据集（推理用）

    从指定目录中扫描所有 .tif/.tiff 文件，逐个整体加载为 3D tensor。
    """

    def __init__(
        self,
        data_source: Union[str, Path, List[str]],
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.transform = transform
        self.normalize = normalize

        if isinstance(data_source, (str, Path)):
            data_source = Path(data_source)
            if data_source.is_dir():
                self.file_paths = sorted([
                    str(p) for p in data_source.iterdir()
                    if p.suffix.lower() in ('.tif', '.tiff')
                ])
            elif data_source.is_file():
                self.file_paths = [str(data_source)]
            else:
                raise FileNotFoundError(f"数据源不存在: {data_source}")
        elif isinstance(data_source, list):
            self.file_paths = [str(p) for p in data_source]
        else:
            raise TypeError(f"不支持的数据源类型: {type(data_source)}")

        if len(self.file_paths) == 0:
            raise ValueError("未找到任何 .tif 文件")

        print(f"[TifChunkDataset] 加载了 {len(self.file_paths)} 个 chunk 文件")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path = self.file_paths[idx]
        volume = tifffile.imread(file_path)

        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        elif volume.ndim != 3:
            raise ValueError(f"不支持的数据维度 {volume.ndim}D，文件: {file_path}")

        volume = volume.astype(np.float32)
        if self.normalize:
            volume = self._normalize(volume)
        if self.transform is not None:
            volume = self.transform(volume)

        return torch.from_numpy(volume).unsqueeze(0)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        max_val = data.max()
        if max_val <= 1.0 + 1e-6:
            return np.clip(data, 0.0, 1.0)
        elif max_val <= 255.0 + 1e-6:
            return data / 255.0
        else:
            return data / 65535.0

    def get_file_path(self, idx: int) -> str:
        return self.file_paths[idx]


# ===================================================================
#  高性能训练 Dataset
# ===================================================================

class VesuviusTrainDataset(Dataset):
    """
    Vesuvius 训练数据集（高性能版）

    自动检测预处理的 NPY 文件，优先使用 mmap 零拷贝模式。
    若 NPY 不存在则回退到 TIF + LRU 缓存。

    性能对比：
      NPY mmap: ~0.1ms/sample (零拷贝切片)
      TIF cache hit: ~0.1ms/sample (内存缓存)
      TIF cache miss: ~100ms/sample (解压 LZW)

    Args:
        image_dir:  train_images 目录
        label_dir:  train_labels 目录
        crop_size:  3D 随机裁剪尺寸
        transform:  裁剪后的增强变换 (接受 (image, label) 返回 (image, label))
        samples_per_volume: 每个体积每 epoch 采几个 patch
        cache_size: TIF 模式下的 LRU 缓存体积数量
        max_files:  最多使用多少个文件（None=全部）
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        crop_size: int = 96,
        transform: Optional[Callable] = None,
        samples_per_volume: int = 16,
        cache_size: int = 32,
        max_files: Optional[int] = None,
        pos_ratio: float = 0.7,
    ):
        super().__init__()
        self.transform = transform
        self.samples_per_volume = samples_per_volume
        self.cache_size = cache_size
        self.pos_ratio = pos_ratio  # 正样本强制采样比例（拒绝采样）

        # 裁剪尺寸
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        else:
            self.crop_size = tuple(crop_size)

        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        # ====== 优先检测 NPY 目录 ======
        npy_image_dir = image_dir.parent / (image_dir.name + "_npy")
        npy_label_dir = label_dir.parent / (label_dir.name + "_npy")

        self.use_npy = False
        if npy_image_dir.exists() and npy_label_dir.exists():
            image_files = {p.stem: p for p in sorted(npy_image_dir.glob("*.npy"))}
            label_files = {p.stem: p for p in sorted(npy_label_dir.glob("*.npy"))}
            if len(image_files) > 0 and len(label_files) > 0:
                self.use_npy = True
                print(f"[Dataset] 🚀 发现预处理 NPY 数据 "
                      f"({len(image_files)} img + {len(label_files)} lbl)，"
                      f"启用极速 mmap 模式！")
            else:
                print("[Dataset] NPY 目录为空，回退到 TIF 模式")

        if not self.use_npy:
            # 回退到 TIF
            image_files = {
                p.stem: p for p in sorted(image_dir.iterdir())
                if p.suffix.lower() in ('.tif', '.tiff')
            }
            label_files = {
                p.stem: p for p in sorted(label_dir.iterdir())
                if p.suffix.lower() in ('.tif', '.tiff')
            }
            print(f"[Dataset] 使用 TIF 模式 + LRU 缓存 (cache_size={cache_size})")

        # 配对
        common_ids = sorted(set(image_files.keys()) & set(label_files.keys()))
        if max_files is not None:
            common_ids = common_ids[:max_files]

        self.pairs = [
            (str(image_files[cid]), str(label_files[cid]))
            for cid in common_ids
        ]

        if len(self.pairs) == 0:
            raise ValueError(
                f"未找到 image-label 配对！\n"
                f"  image_dir: {image_dir}\n"
                f"  label_dir: {label_dir}"
            )

        # 预扫描体积 shape
        self._shapes = []
        if self.use_npy:
            # NPY: 读 header 获取 shape（极快）
            for img_path, _ in self.pairs:
                arr = np.load(img_path, mmap_mode='r')
                self._shapes.append(arr.shape)
        else:
            # TIF: 读文件头
            for img_path, _ in self.pairs:
                with tifffile.TiffFile(img_path) as tif:
                    self._shapes.append(tif.series[0].shape)

        # TIF 模式下的 LRU 缓存
        self._cache = OrderedDict()

        print(f"[VesuviusTrainDataset] {len(self.pairs)} 个配对, "
              f"crop={self.crop_size}, {samples_per_volume} samples/vol, "
              f"总计 {len(self)} 个训练样本/epoch")

    def __len__(self) -> int:
        return len(self.pairs) * self.samples_per_volume

    def _load_volume(self, vol_idx: int):
        """
        加载体积数据

        NPY 模式: np.load(mmap_mode='r') → 返回 mmap 对象，零 RAM 开销
        TIF 模式: imread + LRU 缓存
        """
        if self.use_npy:
            img_path, lbl_path = self.pairs[vol_idx]
            image = np.load(img_path, mmap_mode='r')
            label = np.load(lbl_path, mmap_mode='r')
            return image, label

        # TIF LRU 缓存
        if vol_idx in self._cache:
            self._cache.move_to_end(vol_idx)
            return self._cache[vol_idx]

        img_path, lbl_path = self.pairs[vol_idx]
        image = tifffile.imread(img_path)
        label = tifffile.imread(lbl_path)

        self._cache[vol_idx] = (image, label)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return image, label

    def _random_crop_coords(self, vol_shape):
        """计算随机裁剪坐标，处理小体积 padding"""
        coords = []
        pads = []
        for dim_size, crop_dim in zip(vol_shape, self.crop_size):
            if dim_size >= crop_dim:
                start = np.random.randint(0, dim_size - crop_dim + 1)
                coords.append((start, start + crop_dim))
                pads.append((0, 0))
            else:
                coords.append((0, dim_size))
                pads.append((0, crop_dim - dim_size))
        return coords, pads

    def _surface_biased_crop(self, label_vol, vol_shape):
        """
        Surface-Biased Rejection Sampling（表面偏向拒绝采样）

        以 pos_ratio 的概率强制采样到含正样本（墨水/纸张）的区域，
        剩余概率进行纯随机采样（负样本挖掘）。

        IO 安全保证：
          - label_vol 是 np.memmap 对象（mmap_mode='r'）
          - label_vol[d0:d1, h0:h1, w0:w1] 切片仅触发缺页中断，
            操作系统只读取对应页面（~0.25MB），不加载整个体积
          - np.any() 短路求值，命中首个非零元素即返回

        Args:
            label_vol: label 体积（mmap 对象或 ndarray）
            vol_shape: 体积的形状 (D, H, W)

        Returns:
            coords, pads: 与 _random_crop_coords 格式一致
        """
        force_positive = (np.random.rand() < self.pos_ratio)

        if force_positive:
            # 拒绝采样：最多重试 10 次寻找含正样本的 patch
            for _attempt in range(10):
                coords, pads = self._random_crop_coords(vol_shape)
                (d0, d1), (h0, h1), (w0, w1) = coords
                # 关键: 仅对 label mmap 做切片 peek，不加载整个体积
                label_patch = label_vol[d0:d1, h0:h1, w0:w1]
                if np.any(label_patch == 1):
                    return coords, pads
            # 10 次全失败（极罕见），接受最后一次的随机坐标
            return coords, pads
        else:
            # 负样本挖掘：纯随机裁剪
            return self._random_crop_coords(vol_shape)

    def __getitem__(self, idx: int):
        """
        加载体积 → 随机裁剪 → 归一化 → 增强 → Tensor

        Returns:
            image: (1, cD, cH, cW) float32 [0, 1]
            label: (1, cD, cH, cW) float32 {0, 1}
        """
        vol_idx = idx // self.samples_per_volume
        image_vol, label_vol = self._load_volume(vol_idx)

        # 随机裁剪（image 和 label 用相同坐标）
        # 使用 Surface-Biased Rejection Sampling 替代纯随机裁剪
        coords, pads = self._surface_biased_crop(label_vol, image_vol.shape)
        (d0, d1), (h0, h1), (w0, w1) = coords

        # 切片 + 转 float32（NPY mmap 此时才触发真正的磁盘读取）
        image = np.array(image_vol[d0:d1, h0:h1, w0:w1], dtype=np.float32)
        label = np.array(label_vol[d0:d1, h0:h1, w0:w1], dtype=np.float32)

        # Padding（体积小于 crop_size 时）
        need_pad = any(p != (0, 0) for p in pads)
        if need_pad:
            image = np.pad(image, pads, mode='constant', constant_values=0)
            label = np.pad(label, pads, mode='constant', constant_values=0)

        # 归一化 image → [0, 1]
        max_val = image.max()
        if max_val > 1.0:
            if max_val <= 255.0:
                image = image / 255.0
            else:
                image = image / 65535.0
        image = np.clip(image, 0.0, 1.0)

        # Label 二值化（方案 A：只认 val=1 为纸草表面，val=2 为忽略区域当作背景）
        # 竞赛标签：0=背景, 1=纸草表面(目标), 2=忽略/填充
        label = (label == 1).astype(np.float32)

        # 增强变换（仅 FlipRotate，Crop 已完成）
        if self.transform is not None:
            image, label = self.transform(image, label)

        # 转 Tensor
        image_t = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        label_t = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        return image_t, label_t


if __name__ == "__main__":
    import time

    print("=== Dataset 自测 ===")
    test_dir = Path("__test_tif_chunks__")
    test_dir.mkdir(exist_ok=True)

    try:
        # TifChunkDataset 测试
        for i in range(3):
            tifffile.imwrite(
                str(test_dir / f"chunk_{i:03d}.tif"),
                np.random.randint(0, 65535, (16, 32, 32), dtype=np.uint16)
            )
        ds = TifChunkDataset(test_dir)
        sample = ds[0]
        assert sample.shape == (1, 16, 32, 32)
        print("✓ TifChunkDataset 通过！")

        # VesuviusTrainDataset 测试 (TIF 模式)
        img_dir = test_dir / "images"
        lbl_dir = test_dir / "labels"
        img_dir.mkdir(exist_ok=True)
        lbl_dir.mkdir(exist_ok=True)
        for i in range(5):
            tifffile.imwrite(str(img_dir / f"vol_{i:03d}.tif"),
                             np.random.randint(0, 255, (64, 64, 64), dtype=np.uint8))
            tifffile.imwrite(str(lbl_dir / f"vol_{i:03d}.tif"),
                             np.random.choice([0, 1, 2], (64, 64, 64)).astype(np.uint8))

        train_ds = VesuviusTrainDataset(
            img_dir, lbl_dir, crop_size=32, samples_per_volume=4, cache_size=5
        )

        t0 = time.time()
        for i in range(20):
            img, lbl = train_ds[i % len(train_ds)]
        t1 = time.time()
        print(f"TIF 模式: {(t1-t0)/20*1000:.1f}ms/sample, "
              f"image={img.shape}, label={lbl.shape}")

        # VesuviusTrainDataset 测试 (NPY 模式)
        npy_img = test_dir / "images_npy"
        npy_lbl = test_dir / "labels_npy"
        npy_img.mkdir(exist_ok=True)
        npy_lbl.mkdir(exist_ok=True)
        for i in range(5):
            np.save(str(npy_img / f"vol_{i:03d}.npy"),
                    np.random.randint(0, 255, (64, 64, 64), dtype=np.uint8))
            np.save(str(npy_lbl / f"vol_{i:03d}.npy"),
                    np.random.choice([0, 1, 2], (64, 64, 64)).astype(np.uint8))

        npy_ds = VesuviusTrainDataset(
            img_dir, lbl_dir, crop_size=32, samples_per_volume=4
        )

        t0 = time.time()
        for i in range(20):
            img, lbl = npy_ds[i % len(npy_ds)]
        t1 = time.time()
        print(f"NPY 模式: {(t1-t0)/20*1000:.1f}ms/sample, "
              f"image={img.shape}, label={lbl.shape}")

        print("\n✓ 所有测试通过！")

    finally:
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

```

---
## File: 20_data\transforms.py
```py
"""
Vesuvius Challenge - 3D 数据增强变换 (Phase 5)

用于 Patch-based 训练：对 (image, label) 配对同步执行变换。

组件:
- RandomCrop3D:      从大体积中随机裁剪固定大小的 3D patch
- RandomFlipRotate3D: 随机翻转（3 轴）+ 90° 旋转增强
- Compose3D:         组合多个变换
"""

import numpy as np
from typing import Tuple, List, Optional


class RandomCrop3D:
    """
    从 (image, label) 配对中随机裁剪 3D patch

    Args:
        crop_size: 裁剪尺寸，int 或 (D, H, W) tuple
    """

    def __init__(self, crop_size=64):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        else:
            self.crop_size = tuple(crop_size)

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: (D, H, W) float32
            label: (D, H, W) float32 或 uint8

        Returns:
            image_crop, label_crop: 裁剪后的配对
        """
        D, H, W = image.shape
        cD, cH, cW = self.crop_size

        # 确保体积足够大
        if D < cD or H < cH or W < cW:
            # 体积不够大，pad 到合适尺寸
            pad_d = max(cD - D, 0)
            pad_h = max(cH - H, 0)
            pad_w = max(cW - W, 0)
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            D, H, W = image.shape

        # 随机起始位置
        d0 = np.random.randint(0, D - cD + 1)
        h0 = np.random.randint(0, H - cH + 1)
        w0 = np.random.randint(0, W - cW + 1)

        image_crop = image[d0:d0+cD, h0:h0+cH, w0:w0+cW]
        label_crop = label[d0:d0+cD, h0:h0+cH, w0:w0+cW]

        return image_crop, label_crop


class RandomFlipRotate3D:
    """
    随机 3D 翻转 + 90° 旋转增强

    对 (image, label) 配对同步操作，保证空间一致性。

    Args:
        flip_prob: 每个轴翻转的概率，默认 0.5
        rotate_prob: 执行 90° 旋转的概率，默认 0.5
    """

    def __init__(self, flip_prob: float = 0.5, rotate_prob: float = 0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: (D, H, W)
            label: (D, H, W)

        Returns:
            image_aug, label_aug: 增强后的配对
        """
        # 确保 contiguous (避免负 stride 问题)
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        # 随机翻转 3 个轴
        for axis in range(3):
            if np.random.random() < self.flip_prob:
                image = np.flip(image, axis=axis)
                label = np.flip(label, axis=axis)

        # 随机 90° 旋转 (在 H-W 平面)
        if np.random.random() < self.rotate_prob:
            k = np.random.randint(1, 4)  # 旋转 90°, 180°, 270°
            image = np.rot90(image, k=k, axes=(1, 2))
            label = np.rot90(label, k=k, axes=(1, 2))

        # 确保 contiguous
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        return image, label


class Compose3D:
    """
    组合多个 (image, label) 配对变换

    Args:
        transforms: 变换列表，每个变换接受 (image, label) 返回 (image, label)
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


if __name__ == "__main__":
    print("=== 3D Transforms 自测 ===")

    # 合成数据
    image = np.random.rand(64, 128, 128).astype(np.float32)
    label = (image > 0.5).astype(np.float32)

    # 测试 RandomCrop3D
    crop = RandomCrop3D(32)
    img_c, lbl_c = crop(image, label)
    assert img_c.shape == (32, 32, 32), f"Crop 形状错误: {img_c.shape}"
    assert lbl_c.shape == (32, 32, 32), f"Label Crop 形状错误: {lbl_c.shape}"
    print(f"  RandomCrop3D: {image.shape} → {img_c.shape} ✓")

    # 测试 RandomFlipRotate3D
    aug = RandomFlipRotate3D()
    img_a, lbl_a = aug(img_c, lbl_c)
    assert img_a.shape == (32, 32, 32), f"Aug 形状错误: {img_a.shape}"
    print(f"  RandomFlipRotate3D: {img_c.shape} → {img_a.shape} ✓")

    # 测试 Compose3D
    pipeline = Compose3D([
        RandomCrop3D(32),
        RandomFlipRotate3D(),
    ])
    img_p, lbl_p = pipeline(image, label)
    assert img_p.shape == (32, 32, 32), f"Pipeline 形状错误: {img_p.shape}"
    print(f"  Compose3D: {image.shape} → {img_p.shape} ✓")

    # 测试小体积 padding
    small_img = np.random.rand(16, 16, 16).astype(np.float32)
    small_lbl = (small_img > 0.5).astype(np.float32)
    crop64 = RandomCrop3D(32)
    img_s, lbl_s = crop64(small_img, small_lbl)
    assert img_s.shape == (32, 32, 32), f"Small vol crop 错误: {img_s.shape}"
    print(f"  Small volume padding: (16,16,16) → {img_s.shape} ✓")

    print("✓ 所有测试通过！")

```

---
## File: 20_model\__init__.py
```py
# 20_model: 双头 U-Net 模型 + 复合损失函数
from .dual_unet import DualHeadResUNet3D
from .chimera_loss import ChimeraLoss, compute_gt_normals

```

---
## File: 20_model\chimera_loss.py
```py
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
        """
        Args:
            logits: 分割 logits，形状 (B, 1, D, H, W)
            targets: 分割标签，形状 (B, 1, D, H, W)

        Returns:
            loss: 标量 Dice Loss
        """
        probs = torch.sigmoid(logits)

        # 展平为 (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Dice 系数
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return (1.0 - dice).mean()


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
    Chimera 复合损失函数 (Updated with BCE)

    L_total = L_Dice + λ_bce × L_BCE + λ_normal × L_CosineSimilarity

    同时监督像素分类、重叠区域和法线预测。
    BCE 是打破“全1预测”死循环的关键。

    Args:
        lambda_normal: 法线损失的权重系数，默认 1.0
        lambda_bce:    BCE 损失的权重系数，默认 1.0
        dice_smooth:   Dice Loss 的平滑系数
    """

    def __init__(
        self,
        lambda_normal: float = 1.0,
        lambda_bce: float = 1.0,
        dice_smooth: float = 1e-6,
        pos_weight: float = 10.0,
    ):
        super().__init__()
        self.lambda_normal = lambda_normal
        self.lambda_bce = lambda_bce
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        # 正样本加权 (pos_weight)，打破"全0"陷阱
        # pos_weight=10.0: 适度强调正样本，平衡 Precision/Recall
        # pos_weight=100.0: 极端强调正样本（会导致预测膨胀）
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
            total_loss:  L_Dice + λ_bce * L_BCE + λ_normal * L_Cosine
            dice_val:    Dice Loss 分量
            bce_val:     BCE Loss 分量
            normal_val:  Normal Cosine Loss 分量
        """
        # 1. Dice Loss (重叠度)
        dice_val = self.dice_loss(seg_logits, targets)

        # 2. BCE Loss (像素级分类 - 严惩背景误报)
        # 确保 pos_weight 在正确的设备上
        if self.bce_loss.pos_weight is not None and self.bce_loss.pos_weight.device != seg_logits.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(seg_logits.device)
            
        bce_val = self.bce_loss(seg_logits, targets.float())

        # 3. 实时计算法线 GT
        gt_normals = compute_gt_normals(targets)

        # 4. Normal Cosine Loss (几何)
        normal_val = self.normal_loss(pred_normals, gt_normals, targets)

        # 5. 总损失
        total_loss = dice_val + (self.lambda_bce * bce_val) + (self.lambda_normal * normal_val)

        return total_loss, dice_val, bce_val, normal_val


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

```

---
## File: 20_model\dual_unet.py
```py
"""
Vesuvius Challenge - 双头 Residual 3D U-Net (MVP)

Hybrid Chimera 架构的感知模块：
- Head A (Segmentation): 输出 (B,1,D,H,W)，Sigmoid 激活，概率图
- Head B (Geometry):      输出 (B,3,D,H,W)，Tanh 激活，法线向量 (nx,ny,nz)

设计风格沿用 src/model.py 中 ResUNet3DWithAffinity 的 Encoder-Decoder 模式，
使用 DoubleConv3D (Residual) + 轻量化通道数 (base=16)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """
    3D 双卷积块 + Residual Connection

    结构: Input → (Conv3d→BN→ReLU) × 2 → + Residual → Output
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # 残差连接：通道数不同时用 1x1 卷积对齐
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.residual(x))


class DualHeadResUNet3D(nn.Module):
    """
    双头 Residual 3D U-Net

    Encoder: 4 层下采样
    Decoder: 4 层上采样 + Skip Connections
    输出头:
        - seg_head:    (B, 1, D, H, W) 分割概率图
        - normal_head: (B, 3, D, H, W) 表面法线向量

    Args:
        in_channels: 输入通道数，默认 1（灰度 CT）
        n_filters: 基础滤波器数量，默认 16（轻量化设计）
    """

    def __init__(self, in_channels: int = 1, n_filters: int = 16):
        super().__init__()

        # ===== Encoder (下采样) =====
        self.enc1 = DoubleConv3D(in_channels, n_filters)         # → n
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(n_filters, n_filters * 2)       # → 2n
        # Anisotropic Pooling: 保护 Z 轴分辨率 (D 保持不变)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc3 = DoubleConv3D(n_filters * 2, n_filters * 4)   # → 4n
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc4 = DoubleConv3D(n_filters * 4, n_filters * 8)   # → 8n
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ===== Bottleneck =====
        self.bottleneck = DoubleConv3D(n_filters * 8, n_filters * 16)  # → 16n

        # ===== Decoder (上采样) =====
        # 对应 pool4: (1, 2, 2)
        self.up4 = nn.ConvTranspose3d(n_filters * 16, n_filters * 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec4 = DoubleConv3D(n_filters * 16, n_filters * 8)   # concat: 8n + 8n

        # 对应 pool3: (1, 2, 2)
        self.up3 = nn.ConvTranspose3d(n_filters * 8, n_filters * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = DoubleConv3D(n_filters * 8, n_filters * 4)    # concat: 4n + 4n

        # 对应 pool2: (1, 2, 2)
        self.up2 = nn.ConvTranspose3d(n_filters * 4, n_filters * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = DoubleConv3D(n_filters * 4, n_filters * 2)    # concat: 2n + 2n

        # 对应 pool1: (2, 2, 2) -> 保持原样
        self.up1 = nn.ConvTranspose3d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(n_filters * 2, n_filters)        # concat: n + n

        # ===== 双输出头 =====
        # Head A: 分割概率图 (Sigmoid)
        self.seg_head = nn.Conv3d(n_filters, 1, kernel_size=1)

        # Head B: 表面法线向量 (Tanh → [-1, 1])
        self.normal_head = nn.Conv3d(n_filters, 3, kernel_size=1)

        # ===== 负偏置初始化 =====
        # Sigmoid(-2.0) ≈ 0.12，强制模型初始状态预测"背景"
        # 防止模型一开始就陷入 Logits≈0 (Sigmoid≈0.5) 的舒适区
        nn.init.constant_(self.seg_head.bias, -2.0)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: 输入张量，形状 (B, 1, D, H, W)

        Returns:
            seg_logits: 分割 logits，形状 (B, 1, D, H, W)
                        下游使用 Sigmoid 或 BCEWithLogits 处理
            normals:    法线预测，形状 (B, 3, D, H, W)，值域 [-1, 1]
        """
        # --- Encoder ---
        e1 = self.enc1(x)       # (B, n, D, H, W)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)      # (B, 2n, D/2, H/2, W/2)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)      # (B, 4n, D/4, H/4, W/4)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)      # (B, 8n, D/8, H/8, W/8)
        p4 = self.pool4(e4)

        # --- Bottleneck ---
        b = self.bottleneck(p4)  # (B, 16n, D/16, H/16, W/16)

        # --- Decoder ---
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)      # (B, n, D, H, W) — 共享特征

        # --- 双头输出 ---
        seg_logits = self.seg_head(d1)      # (B, 1, D, H, W) — raw logits
        normals = torch.tanh(self.normal_head(d1))  # (B, 3, D, H, W) — [-1, 1]

        return seg_logits, normals


if __name__ == "__main__":
    print("=== DualHeadResUNet3D 自测 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadResUNet3D(in_channels=1, n_filters=16).to(device)

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 前向传播测试
    # 输入尺寸必须能被 16 整除 (4 层下采样，每层 /2)
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    seg_logits, normals = model(x)

    print(f"输入形状:     {x.shape}")
    print(f"分割输出形状: {seg_logits.shape}")   # 期望 (1, 1, 64, 64, 64)
    print(f"法线输出形状: {normals.shape}")       # 期望 (1, 3, 64, 64, 64)
    print(f"法线值域:     [{normals.min():.4f}, {normals.max():.4f}]")

    assert seg_logits.shape == (1, 1, 64, 64, 64), f"分割输出形状错误: {seg_logits.shape}"
    assert normals.shape == (1, 3, 64, 64, 64), f"法线输出形状错误: {normals.shape}"
    assert normals.min() >= -1.0 and normals.max() <= 1.0, "法线值域应在 [-1, 1]"

    print("✓ 所有测试通过！")

```

---
## File: output\train_20260214_033543\training_history.json
```json
[
  {
    "epoch": 1,
    "lr": 0.001,
    "loss": 0.6997804939746857,
    "dice_loss": 0.6052835807204247,
    "normal_loss": 0.9449691474437714,
    "dice_score": 0.34815455228090286,
    "time": 2.958019256591797,
    "val_loss": 0.6592543125152588,
    "val_dice": 0.022845519706606865
  },
  {
    "epoch": 2,
    "lr": 0.000505,
    "loss": 0.5744479671120644,
    "dice_loss": 0.4918648824095726,
    "normal_loss": 0.8258308321237564,
    "dice_score": 0.44266364723443985,
    "time": 0.24978041648864746,
    "val_loss": 0.6520134210586548,
    "val_dice": 0.010394347831606865
  }
]
```

---
