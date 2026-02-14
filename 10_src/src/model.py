"""
Vesuvius Challenge 2026 - 3D ResU-Net 模型
支持 Mamba SSM 全局上下文增强 + Affinity 拓扑感知分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 Mamba SSM 库 (静默检测，避免 num_workers 重复打印)
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class LargeKernelConv3D(nn.Module):
    """
    全局上下文回退方案 (Mamba 不可用时)
    使用两层 3x3x3 深度可分离卷积 + SE 注意力捕获上下文
    等效感受野 5x5x5，但计算量远低于 7x7x7
    """
    def __init__(self, dim):
        super().__init__()
        # 两层 3x3x3 深度可分离卷积 (等效 5x5x5 感受野)
        self.dw_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm3d(dim),
            nn.GELU(),
        )
        # SE 注意力: 全局池化 -> 压缩 -> 激发
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        # 逐点卷积混合通道
        self.pw_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):
        out = self.dw_conv(x)
        out = out * self.se(out)  # SE 注意力加权
        out = self.pw_conv(out)
        return out


class MambaBottleneck(nn.Module):
    """
    Mamba 状态空间模型瓶颈模块
    
    将 3D 特征图展平为序列 -> 运行 Mamba 算子 -> 还原为 3D
    通过残差连接保持梯度稳定性
    
    回退方案: 当 mamba_ssm 库不可用时使用大核卷积
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.use_mamba = MAMBA_AVAILABLE

        if self.use_mamba:
            # Mamba 状态空间模型
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.norm = nn.LayerNorm(dim)
        else:
            # 回退: 大核卷积
            self.fallback = LargeKernelConv3D(dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) 3D 特征图
        Returns:
            out: (B, C, D, H, W) 增强后的特征图
        """
        residual = x

        if self.use_mamba:
            B, C, D, H, W = x.shape
            # 展平: (B, C, D, H, W) -> (B, D*H*W, C)
            x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)
            x_flat = self.norm(x_flat)
            # Mamba 推理
            x_flat = self.mamba(x_flat)
            # 还原: (B, D*H*W, C) -> (B, C, D, H, W)
            x = x_flat.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        else:
            x = self.fallback(x)

        # 残差连接
        return x + residual


class DoubleConv3D(nn.Module):
    """
    3D 双卷积块，包含 Residual Connection
    Structure: Input -> (Conv3d->BN->ReLU) * 2 -> + Input -> Output
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 如果输入输出通道不一致，或者需要调整尺寸，则在残差连接上对输入进行 1x1 卷积
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class ResUNet3D(nn.Module):
    """
    轻量级 3D ResU-Net 用于表面检测
    Encoder: 4 层下采样
    Decoder: 4 层上采样 + Skip Connections
    """
    def __init__(self, in_channels=1, out_channels=1, n_filters=16):
        super().__init__()
        
        # Encoder (Downsampling)
        # Level 1
        self.enc1 = DoubleConv3D(in_channels, n_filters)
        self.pool1 = nn.MaxPool3d(2)
        
        # Level 2
        self.enc2 = DoubleConv3D(n_filters, n_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        # Level 3
        self.enc3 = DoubleConv3D(n_filters * 2, n_filters * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        # Level 4
        self.enc4 = DoubleConv3D(n_filters * 4, n_filters * 8)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck (Level 5)
        self.bottleneck = DoubleConv3D(n_filters * 8, n_filters * 16)
        
        # Decoder (Upsampling)
        # Level 4
        self.up4 = nn.ConvTranspose3d(n_filters * 16, n_filters * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3D(n_filters * 16, n_filters * 8) # concat inputs: 8 + 8 = 16
        
        # Level 3
        self.up3 = nn.ConvTranspose3d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(n_filters * 8, n_filters * 4) # concat inputs: 4 + 4 = 8
        
        # Level 2
        self.up2 = nn.ConvTranspose3d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(n_filters * 4, n_filters * 2) # concat inputs: 2 + 2 = 4
        
        # Level 1
        self.up1 = nn.ConvTranspose3d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(n_filters * 2, n_filters) # concat inputs: 1 + 1 = 2
        
        # Final Output
        self.out_conv = nn.Conv3d(n_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.up4(b)
        # Skip connection: concat with e4
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
        d1 = self.dec1(d1)
        
        # Output Logits
        out = self.out_conv(d1)
        
        return out


class ResUNet3DWithAffinity(nn.Module):
    """
    带 Affinity Head 的 3D ResU-Net
    
    除了输出分割掩码外，还输出 3 通道的 Affinity Map，
    预测当前像素与 X/Y/Z 方向邻居的连通性。
    
    参考文献: Neuron Segmentation with Affinity Learning
    """
    def __init__(self, in_channels=1, out_channels=1, n_filters=16):
        super().__init__()
        
        # 复用 ResUNet3D 的 Encoder-Decoder 结构
        # Encoder (Downsampling)
        self.enc1 = DoubleConv3D(in_channels, n_filters)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = DoubleConv3D(n_filters, n_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = DoubleConv3D(n_filters * 2, n_filters * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = DoubleConv3D(n_filters * 4, n_filters * 8)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(n_filters * 8, n_filters * 16)
        
        # Mamba 全局上下文模块 (插在 bottleneck 后)
        self.mamba = MambaBottleneck(dim=n_filters * 16)
        
        # Decoder (Upsampling)
        self.up4 = nn.ConvTranspose3d(n_filters * 16, n_filters * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3D(n_filters * 16, n_filters * 8)
        
        self.up3 = nn.ConvTranspose3d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(n_filters * 8, n_filters * 4)
        
        self.up2 = nn.ConvTranspose3d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(n_filters * 4, n_filters * 2)
        
        self.up1 = nn.ConvTranspose3d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(n_filters * 2, n_filters)
        
        # 主输出头: Segmentation Mask
        self.seg_head = nn.Conv3d(n_filters, out_channels, kernel_size=1)
        
        # 辅助输出头: Affinity Map (3 通道: X/Y/Z 方向连通性)
        self.affinity_head = nn.Conv3d(n_filters, 3, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck + Mamba 全局上下文
        b = self.bottleneck(p4)
        b = self.mamba(b)  # 全局上下文增强
        
        # Decoder
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
        d1 = self.dec1(d1)
        
        # 双头输出
        seg_out = self.seg_head(d1)  # (B, 1, D, H, W)
        affinity_out = self.affinity_head(d1)  # (B, 3, D, H, W)
        
        return seg_out, affinity_out


def compute_affinity_target(mask: torch.Tensor) -> torch.Tensor:
    """
    根据分割掩码计算 Affinity Ground Truth
    
    Affinity[c] 表示当前像素与 c 方向邻居是否属于同一区域。
    
    Args:
        mask: (B, 1, D, H, W) 二值分割掩码
        
    Returns:
        affinity: (B, 3, D, H, W) Affinity 真值
            - Channel 0: X 方向 (右邻居)
            - Channel 1: Y 方向 (下邻居)
            - Channel 2: Z 方向 (深邻居)
    """
    B, C, D, H, W = mask.shape
    
    # 计算各方向的 Affinity
    # Affinity = 1 当且仅当 mask[i] == mask[i+1] == 1
    affinity_x = (mask[:, :, :, :, :-1] * mask[:, :, :, :, 1:])  # (B, 1, D, H, W-1)
    affinity_y = (mask[:, :, :, :-1, :] * mask[:, :, :, 1:, :])  # (B, 1, D, H-1, W)
    affinity_z = (mask[:, :, :-1, :, :] * mask[:, :, 1:, :, :])  # (B, 1, D-1, H, W)
    
    # 填充到原始尺寸 (边界处填 0)
    affinity_x = F.pad(affinity_x, (0, 1, 0, 0, 0, 0), value=0)  # 右边填充
    affinity_y = F.pad(affinity_y, (0, 0, 0, 1, 0, 0), value=0)  # 下边填充
    affinity_z = F.pad(affinity_z, (0, 0, 0, 0, 0, 1), value=0)  # 深边填充
    
    # 合并为 3 通道
    affinity = torch.cat([affinity_x, affinity_y, affinity_z], dim=1)  # (B, 3, D, H, W)
    
    return affinity


if __name__ == "__main__":
    # 简单的模型测试
    print("测试 ResUNet3D 模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUNet3D(in_channels=1, out_channels=1, n_filters=16).to(device)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    
    # 测试前向传播
    batch_size = 2
    patch_size = (64, 128, 128)
    dummy_input = torch.randn(batch_size, 1, *patch_size).to(device)
    
    try:
        output = model(dummy_input)
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        
        if output.shape == dummy_input.shape:
            print("✓ 模型测试通过：输入输出形状一致")
        else:
            print("✗ 模型测试失败：输入输出形状不一致")
            
    except Exception as e:
        print(f"模型运行出错: {e}")
    
    # 测试 Affinity 模型
    print("\n测试 ResUNet3DWithAffinity 模型...")
    model_aff = ResUNet3DWithAffinity(in_channels=1, out_channels=1, n_filters=16).to(device)
    total_params_aff = sum(p.numel() for p in model_aff.parameters())
    print(f"Affinity 模型总参数量: {total_params_aff / 1e6:.2f} M")
    
    try:
        seg_out, aff_out = model_aff(dummy_input)
        print(f"Seg 输出形状: {seg_out.shape}")
        print(f"Affinity 输出形状: {aff_out.shape}")
        print("✓ Affinity 模型测试通过")
    except Exception as e:
        print(f"Affinity 模型运行出错: {e}")

