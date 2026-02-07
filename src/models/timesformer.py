import torch
import torch.nn as nn
from einops import rearrange, repeat
import logging

logger = logging.getLogger(__name__)

class TimeSformer(nn.Module):
    """
    TimeSformer for Vesuvius 3D Ink Detection / 墨迹检测专用 TimeSformer
    
    基于 "Is Space-Time Attention All You Need for Video Understanding?" 论文。
    针对 Vesuvius 墨迹检测任务进行了适配：
    - "Time" 维度对应 Z 轴深度 (Critical Depth = 16)。
    - 输入: (Batch, Channels=1, Depth=16, Height, Width)
    - 输出: (Batch, 1, Height, Width) - 预测墨迹概率图 (Projection)
    
    架构特点:
    - Divided Space-Time Attention (分离的空-深注意力机制)
    - Patch Partitioning
    - Global Average Pooling (Optional) -> Linear Head
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=1,
        num_frames=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        # Patch Dimensions
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        
        # 1. Patch Embeddings / 图像块编码
        # 使用 2D 卷积模拟 Patch 提取 (kernel=stride=patch_size)
        self.patch_embed = nn.Conv2D(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        ) if num_frames == 1 else nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=(1, patch_size, patch_size), 
            stride=(1, patch_size, patch_size)
        )

        # 2. Positional Embeddings / 位置编码
        # 空间位置编码 (Spatial Positional Embedding)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches_per_frame, embed_dim)
        )
        # 时间(深度)位置编码 (Temporal/Depth Positional Embedding)
        self.time_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 3. Transformer Blocks / 编码器堆叠
        # 使用 nn.ModuleList 堆叠 Block
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. Segmentation Head / 分割头
        # 简单的线性投影层 + 上采样，将 Patch 特征恢复为像素级预测
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(patch_size), # Upsample by patch_size (feature -> pixels)
            # PixelShuffle outputs: channel = 256 / (patch_size^2). 
            # If patch_size=16, 256/256=1 channel. Perfect.
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            out: (B, 1, H, W)
        """
        B, C, D, H, W = x.shape
        
        # --- Embedding ---
        # (B, C, D, H, W) -> (B, Embed, D, H/P, W/P)
        x = self.patch_embed(x)
        # Flatten: (B, Embed, D, N_patches) -> (B, D, N_patches, Embed)
        x = x.flatten(3).transpose(1, 3).transpose(1, 2)
        
        # Add Positional Embeddings
        # (1, N_patches, Embed) broadcast over B, D
        x = x + self.pos_embed.unsqueeze(0) 
        # (1, D, Embed) broadcast over B, N_patches
        x = x + self.time_embed.unsqueeze(1) 
        
        x = self.pos_drop(x)
        
        # --- Blocks ---
        # Input to blocks: (B, D, N_patches, Embed)
        for blk in self.blocks:
            x = blk(x)
            
        # --- Head ---
        x = self.norm(x)
        
        # Pooling over Time(Depth) dimension? Average pooling for simplicity.
        # (B, D, N_p, E) -> (B, N_p, E)
        x = x.mean(dim=1) 
        
        # Reshape to (B, E, H/P, W/P) for Conv Head
        h_p, w_p = H // self.patch_size, W // self.patch_size
        x = rearrange(x, 'b (h w) e -> b e h w', h=h_p, w=w_p)
        
        # Upsample to original resolution
        out = self.head(x) # (B, 1, H, W)
        
        return out

class SpaceTimeBlock(nn.Module):
    """
    分离的空-深注意力块 (Divided Space-Time Attention Block)
    先做 Temporal Attention (Z-axis)，再做 Spatial Attention (XY-plane)。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        
        # Temporal Attention (Depth)
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        
        # Spatial Attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        """
        x: (B, D, N, E)
        """
        B, D, N, E = x.shape
        
        # --- Temporal Attention (over D) ---
        # Reshape to (B*N, D, E) to apply attention over D
        xt = x.permute(0, 2, 1, 3).reshape(B*N, D, E)
        res_t = xt
        xt = self.norm1(xt)
        xt, _ = self.attn1(xt, xt, xt)
        xt = res_t + xt
        # Reshape back to (B, D, N, E)
        x = xt.reshape(B, N, D, E).permute(0, 2, 1, 3)
        
        # --- Spatial Attention (over N) ---
        # Reshape to (B*D, N, E) to apply attention over N
        xs = x.reshape(B*D, N, E)
        res_s = xs
        xs = self.norm2(xs)
        xs, _ = self.attn2(xs, xs, xs)
        xs = res_s + xs
        # Reshape back to (B, D, N, E)
        x = xs.reshape(B, D, N, E)
        
        # --- MLP ---
        x = x + self.mlp(self.norm3(x))
        
        return x
