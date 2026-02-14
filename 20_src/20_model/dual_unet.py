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

        # ===== Encoder =====
        self.enc1 = DoubleConv3D(in_channels, n_filters)         # → n
        # GT 是厚壳结构 (Z 轴 314/320 切片有正样本)，各向同性池化即可
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

        # 对应 pool1: 各向同性上采样
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
