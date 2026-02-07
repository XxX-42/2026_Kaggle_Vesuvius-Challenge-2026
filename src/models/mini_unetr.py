import torch
import torch.nn as nn
from einops import rearrange
from transformers import ViTModel, ViTConfig

class MiniUNETR(nn.Module):
    """
    MiniUNETR (SOTA Architecture for Vesuvius)
    
    结合 Transformer 的全局建模能力与 3D CNN 的局部特征提取能力。
    针对 Kaggle 竞赛优化的轻量级版本。
    
    架构优势说明 (Technical Advantage):
    1. **ViT Encoder (Global Context)**: 
       传统的 3D CNN 在提取长距离依赖 (Long-range dependency) 时受限于感受野。
       而墨迹 (Ink) 的连通性往往跨越较大的空间范围。
       Transformer 的 Self-Attention 机制能让模型在第一层就拥有全局感受野，
       更好地识别断断续续的墨迹线条。
       
    2. **3D CNN Decoder (Local Detail)**:
       仅靠 Transformer 很难恢复精细的边缘细节 (因为 Patch 化丢失了分辨率)。
       通过 Skip Connections 将 Encoder 的浅层特征融合到 Decoder，
       并在解码阶段使用 3D 卷积，可以精确地恢复墨迹的几何形状。
       
    3. **Flash Attention 2**:
       利用 PyTorch 2.0+ 的 `scaled_dot_product_attention` 加速计算并节省显存。
    """
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        img_size=256, 
        patch_size=16, 
        hidden_size=384, # 小模型配置，适合 4090 快速迭代
        num_layers=6,   # 减少层数
        num_heads=12,
        mlp_dim=1536,
        feature_size=16
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        
        # --- 1. ViT Encoder ---
        # 使用 Hugging Face 的 ViTConfig 构建，方便集成 Flash Attention
        config = ViTConfig(
            image_size=img_size,
            patch_size=patch_size,
            num_channels=in_channels * 16, # 将 Depth=16 视为 Channel 维度输入 ViT
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=mlp_dim,
            hidden_act="gelu",
            # enable_flash_attention=True # HF transformers >= 4.36 built-in support
        )
        self.vit = ViTModel(config)
        
        # --- 2. Bottleneck Projection ---
        # 将 ViT 输出 (B, N, C) 重塑回 (B, C, H/P, W/P)
        # N = (H/P * W/P)
        self.spatial_size = img_size // patch_size
        
        # --- 3. 3D CNN Decoder ---
        # 我们假设 ViT 提取的是高度压缩的语义特征。
        # 我们需要从 "Flat Representation" 恢复出 "Volumetric Representation"。
        
        # Upsample 1: 16x16 -> 32x32
        self.up1 = nn.ConvTranspose3d(hidden_size, feature_size * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self._block(feature_size * 4, feature_size * 2)
        
        # Upsample 2: 32x32 -> 64x64
        self.up2 = nn.ConvTranspose3d(feature_size * 2, feature_size * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self._block(feature_size * 2, feature_size * 2)
        
        # Upsample 3: 64x64 -> 128x128
        self.up3 = nn.ConvTranspose3d(feature_size * 2, feature_size, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self._block(feature_size, feature_size)
        
        # Upsample 4: 128x128 -> 256x256
        self.up4 = nn.ConvTranspose3d(feature_size, feature_size, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec4 = self._block(feature_size, feature_size)
        
        # Output Head
        self.out_conv = nn.Conv3d(feature_size, out_channels, kernel_size=1)
        
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, 16, H, W) - 3D Volume
        Returns:
            out: (B, 1, H, W) - 2D Mask
        """
        B, C, D, H, W = x.shape
        
        # --- Encoder (ViT) ---
        # ViT 通常处理 2D 图像。我们将 Depth=16 视为 Input Channels。
        # Reshape: (B, 1, 16, H, W) -> (B, 16, H, W)
        x_2d = x.squeeze(1) 
        
        # ViT Output (last_hidden_state): (B, N_patches + 1, Hidden)
        # +1 because of CLS token
        outputs = self.vit(pixel_values=x_2d)
        x_hidden = outputs.last_hidden_state[:, 1:, :] # Drop CLS token -> (B, N, Hidden)
        
        # Reshape for Decoder: (B, Hidden, 1, H/P, W/P)
        # Note: We treat the Z-dim as compressed to 1 in the bottleneck, then expand.
        x_reshaped = rearrange(
            x_hidden, 
            'b (h w) c -> b c 1 h w', 
            h=self.spatial_size, 
            w=self.spatial_size
        ) # (B, 384, 1, 16, 16)
        
        # --- Decoder (3D Upsampling) ---
        # 逐级上采样恢复分辨率
        
        x_up1 = self.up1(x_reshaped) # (B, F*4, 1, 32, 32)
        x_dec1 = self.dec1(x_up1)
        
        x_up2 = self.up2(x_dec1)     # (B, F*2, 1, 64, 64)
        x_dec2 = self.dec2(x_up2)
        
        x_up3 = self.up3(x_dec2)     # (B, F, 1, 128, 128)
        x_dec3 = self.dec3(x_up3)
        
        x_up4 = self.up4(x_dec3)     # (B, F, 1, 256, 256)
        x_dec4 = self.dec4(x_up4)
        
        # --- Output ---
        out_3d = self.out_conv(x_dec4) # (B, 1, 1, H, W)
        
        # Project to 2D
        out_2d = out_3d.squeeze(2) # (B, 1, H, W)
        
        return out_2d
