"""
Vesuvius Predictor (Stage 6 - Gaussian Weighted Blending)

针对 RTX 3060 Laptop (6GB VRAM) 优化的推理管道。
新增：高斯加权融合 (Gaussian Weighted Blending) 消除棋盘格纹路。

=============================================================================
为什么高斯权重融合能解决边缘不连续问题？
=============================================================================

在滑窗推理中，相邻 Tile 会在边缘产生重叠区域。由于：
1. 卷积神经网络的"边缘效应"：边缘像素的感受野不完整，预测可信度较低
2. 不同 Tile 在同一位置可能产生不一致的预测值
3. 简单平均法对所有位置一视同仁，无法抑制边缘噪声

高斯权重融合的物理意义：
- 中心权重 = 1.0：完全相信窗口中心的预测（感受野完整）
- 边缘权重 → 0：抑制边缘预测（感受野不完整）
- 在重叠区域，高权重区域的预测会主导最终结果
- 最终效果：像丝绸一样平滑的概率图，无棋盘格纹路

Technical Highlights:
1. Safe Batch Size: 硬编码 batch_size=16 (安全区)
2. Windowing: 256x256 tiles, stride=128 (50% overlap)
3. Gaussian Weighted Blending: 中心权重高，边缘权重低
4. Z-Shift TTA: [-1, 0, +1] 三次预测取平均
5. Memory Safety: torch.no_grad() + torch.amp.autocast
6. Scout Strategy: 跳过 >98% 背景的 tile
=============================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
import tifffile
from PIL import Image
import os
import gc
import logging
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def create_gaussian_weight(tile_size, sigma_ratio=0.25):
    """
    创建 2D 高斯权重矩阵
    
    Args:
        tile_size: Tile 边长 (例如 256)
        sigma_ratio: sigma 与 tile_size 的比例（默认 0.25，即 sigma=64 for 256x256）
                     较小的 ratio 产生更集中的权重（边缘衰减更快）
                     较大的 ratio 产生更平坦的权重
    
    Returns:
        np.ndarray: (tile_size, tile_size) 的权重矩阵，中心=1.0，边缘→0
        
    物理意义：
        - 中心区域：感受野完整，预测可信 → 高权重
        - 边缘区域：感受野不完整，预测不稳定 → 低权重
    """
    sigma = tile_size * sigma_ratio
    
    # 创建坐标网格
    center = tile_size // 2
    y, x = np.ogrid[:tile_size, :tile_size]
    
    # 计算到中心的距离
    dist_sq = (x - center) ** 2 + (y - center) ** 2
    
    # 高斯函数
    weight = np.exp(-dist_sq / (2 * sigma ** 2))
    
    # 归一化到 [0, 1]，确保中心为 1.0
    weight = weight / weight.max()
    
    # 防止权重过小（避免除零问题）
    weight = np.maximum(weight, 0.01)
    
    return weight.astype(np.float32)


class VesuviusPredictor:
    """
    高斯加权推理器 (Gaussian Weighted Predictor)
    
    使用高斯权重融合解决滑窗推理的边缘不连续问题。
    
    Attributes:
        model: PyTorch 模型
        config: 配置字典
        device: 计算设备
        tile_size: Tile 大小 (默认 256)
        stride: 滑动步长 (默认 128, 50% overlap)
        batch_size: 安全批次大小 (硬编码 16)
        tta_enabled: 是否启用 TTA
        gaussian_weight: 高斯权重矩阵
    """
    
    # 安全区 Batch Size (避免 3060 性能悬崖)
    SAFE_BATCH_SIZE = 16
    
    def __init__(self, model, config, device='cuda'):
        """
        初始化预测器
        
        Args:
            model: PyTorch 模型 (已加载权重)
            config: 配置字典，包含 inference 和 data 配置
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        
        # 推理参数
        inf_cfg = config.get('inference', {})
        self.tta_shifts = inf_cfg.get('tta_shifts', [-1, 0, 1])
        self.tile_size = inf_cfg.get('tile_size', 256)
        self.stride = inf_cfg.get('stride', 128)  # 50% overlap (消除网格感)
        self.batch_size = self.SAFE_BATCH_SIZE  # 硬编码安全值
        self.tta_enabled = inf_cfg.get('tta', True)
        self.threshold = inf_cfg.get('threshold', 0.5)
        
        # 高斯权重参数
        self.use_gaussian_weight = inf_cfg.get('gaussian_blend', True)
        self.gaussian_sigma_ratio = inf_cfg.get('gaussian_sigma_ratio', 0.25)
        
        # 创建高斯权重矩阵（仅生成一次）
        if self.use_gaussian_weight:
            self.gaussian_weight = create_gaussian_weight(
                self.tile_size, 
                self.gaussian_sigma_ratio
            )
            logger.info(f"[Predictor] Gaussian Weight: Enabled (sigma_ratio={self.gaussian_sigma_ratio})")
        else:
            self.gaussian_weight = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
            logger.info("[Predictor] Gaussian Weight: Disabled (uniform blending)")
        
        # 数据参数 (Z 轴范围)
        data_cfg = config.get('data', {})
        self.z_start = data_cfg.get('z_start', 29)  # 对应 slice 29 (index 0)
        self.z_end = data_cfg.get('z_end', 44)      # 对应 slice 44 (index 15)
        self.num_z_slices = self.z_end - self.z_start + 1
        
        logger.info(f"[Predictor] Initialized | Tile: {self.tile_size}x{self.tile_size} | "
                   f"Stride: {self.stride} | Batch: {self.batch_size} | TTA: {self.tta_enabled}")
        logger.info(f"[Predictor] Z-Range: {self.z_start}-{self.z_end} ({self.num_z_slices} slices)")

    def _read_z_slice(self, fragment_path, z_idx):
        """读取单个 Z 层 TIFF 文件"""
        path = os.path.join(fragment_path, "surface_volume", f"{z_idx:02d}.tif")
        if not os.path.exists(path):
            logger.warning(f"Missing slice: {path}")
            return None
        return tifffile.imread(path)

    def _load_volume(self, fragment_path, fragment_id):
        """
        加载完整的 Z-Stack Volume 到 RAM
        
        为 TTA 预留额外的 Z 层：加载 [z_start-1, z_end+1] 范围
        """
        # TTA 需要 z-1 和 z+1，所以扩展加载范围
        z_load_start = max(0, self.z_start - 1)
        z_load_end = min(64, self.z_end + 1)  # 假设最大 65 层
        
        # 先读取一层获取尺寸
        sample = self._read_z_slice(fragment_path, self.z_start)
        if sample is None:
            raise FileNotFoundError(f"Cannot read sample slice at z={self.z_start}")
        H, W = sample.shape
        
        logger.info(f"[{fragment_id}] Loading Volume Z={z_load_start}-{z_load_end} into RAM...")
        volume = np.zeros((z_load_end - z_load_start + 1, H, W), dtype=np.uint16)
        
        for i, z in enumerate(tqdm(range(z_load_start, z_load_end + 1), desc="Loading Z-Stack")):
            img = self._read_z_slice(fragment_path, z)
            if img is not None:
                volume[i] = img
                
        logger.info(f"[{fragment_id}] Volume Shape: {volume.shape} | "
                   f"RAM: {volume.nbytes / 1e9:.2f} GB")
        return volume, H, W, z_load_start

    def predict_fragment(self, fragment_path, fragment_id="1"):
        """
        预测整个 Fragment，返回完整的概率图
        
        使用高斯加权融合实现平滑无缝的预测结果。
        
        Args:
            fragment_path: Fragment 路径 (包含 surface_volume/, mask.png 等)
            fragment_id: Fragment ID (用于日志)
            
        Returns:
            np.ndarray: 完整概率图 (H, W), float32, 值域 [0, 1]
        """
        # 1. 加载 Mask (Scout Strategy)
        logger.info(f"[{fragment_id}] Loading Mask...")
        mask_path = os.path.join(fragment_path, "mask.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32) / 255.0
        H, W = mask.shape
        
        # 2. 加载 Volume
        volume, vol_H, vol_W, z_offset = self._load_volume(fragment_path, fragment_id)
        assert (H, W) == (vol_H, vol_W), f"Mask/Volume size mismatch: {(H,W)} vs {(vol_H, vol_W)}"
        
        # 相对 Z 索引计算
        # volume[0] = z_offset, volume[i] = z_offset + i
        # 我们需要 z_start 到 z_end 的范围
        # 索引映射: volume_idx = z - z_offset
        
        # 3. 生成 Grid
        x_points = self._generate_grid_points(W, self.tile_size, self.stride)
        y_points = self._generate_grid_points(H, self.tile_size, self.stride)
        total_tiles = len(x_points) * len(y_points)
        
        logger.info(f"[{fragment_id}] Grid: {len(y_points)}x{len(x_points)} = {total_tiles} tiles")
        
        # 4. 初始化加权输出缓冲
        # 使用高斯加权累加，而非简单计数
        pred_weighted_sum = np.zeros((H, W), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)
        
        # 5. Batched Inference
        all_coords = [(y, x) for y in y_points for x in x_points]
        batch_coords = []
        skipped = 0
        
        pbar = tqdm(all_coords, desc=f"Inferencing {fragment_id}")
        for (y, x) in pbar:
            # Scout Strategy: 跳过背景区域
            mask_tile = mask[y:y+self.tile_size, x:x+self.tile_size]
            if mask_tile.mean() < 0.02:  # < 2% 有效像素
                skipped += 1
                continue
            
            batch_coords.append((y, x))
            
            # Batch 满了，执行推理
            if len(batch_coords) >= self.batch_size:
                self._process_batch_gaussian(
                    batch_coords, volume, z_offset, 
                    pred_weighted_sum, weight_sum
                )
                batch_coords = []
                gc.collect()
                
        # 处理剩余
        if batch_coords:
            self._process_batch_gaussian(
                batch_coords, volume, z_offset, 
                pred_weighted_sum, weight_sum
            )
            
        logger.info(f"[{fragment_id}] Skipped {skipped}/{total_tiles} background tiles")
        
        # 6. 计算加权平均 (避免除零)
        weight_sum = np.maximum(weight_sum, 1e-6)
        prediction = pred_weighted_sum / weight_sum
        
        # 7. Cleanup
        del volume, pred_weighted_sum, weight_sum
        gc.collect()
        torch.cuda.empty_cache()
        
        return prediction

    def _generate_grid_points(self, size, tile_size, stride):
        """生成滑窗坐标点"""
        points = list(range(0, size - tile_size + 1, stride))
        if not points:
            points = [0]
        elif points[-1] != size - tile_size and size > tile_size:
            points.append(size - tile_size)
        return points

    def _process_batch_gaussian(self, batch_coords, volume, z_offset, 
                                 pred_weighted_sum, weight_sum):
        """
        处理一个 Batch 并进行高斯加权累加
        
        关键改进：使用高斯权重矩阵进行加权累加，而非简单的 +1 计数
        
        Args:
            batch_coords: 批次坐标列表 [(y, x), ...]
            volume: 完整 Z-Stack 体积
            z_offset: Volume 的 Z 轴起始偏移
            pred_weighted_sum: 加权预测累加缓冲
            weight_sum: 权重累加缓冲
        """
        B = len(batch_coords)
        
        # TTA 累加器 (CPU)
        tta_sum = np.zeros((B, self.tile_size, self.tile_size), dtype=np.float32)
        tta_count = 0
        
        # 计算 Z 索引映射
        base_start_idx = self.z_start - z_offset
        
        # TTA 偏移
        shifts = self.tta_shifts if self.tta_enabled else [0]
        
        for shift in shifts:
            start_idx = base_start_idx + shift
            end_idx = start_idx + self.num_z_slices
            
            # 边界检查
            if start_idx < 0 or end_idx > volume.shape[0]:
                logger.warning(f"TTA shift {shift} out of bounds, skipping...")
                continue
            
            # 准备 Batch 输入
            tensors = []
            input_stats = []  # 收集输入统计
            for (y, x) in batch_coords:
                crop = volume[start_idx:end_idx, y:y+self.tile_size, x:x+self.tile_size]
                
                # 归一化 (CPU)
                c = crop.astype(np.float32) / 65535.0
                mean, std = c.mean(), c.std()
                c = (c - mean) / (std + 1e-6)
                
                # 收集归一化前后的统计信息
                input_stats.append({
                    'raw_mean': crop.mean(),
                    'raw_std': crop.std(),
                    'norm_mean': c.mean(),
                    'norm_std': c.std(),
                    'norm_min': c.min(),
                    'norm_max': c.max()
                })
                
                t = torch.from_numpy(c).unsqueeze(0)  # (1, Z, H, W)
                tensors.append(t)
            
            # === 输入分布诊断 (仅第一次) ===
            if not hasattr(self, '_input_stats_logged'):
                self._input_stats_logged = True
                avg_raw_mean = np.mean([s['raw_mean'] for s in input_stats])
                avg_raw_std = np.mean([s['raw_std'] for s in input_stats])
                avg_norm_mean = np.mean([s['norm_mean'] for s in input_stats])
                avg_norm_std = np.mean([s['norm_std'] for s in input_stats])
                avg_norm_min = np.mean([s['norm_min'] for s in input_stats])
                avg_norm_max = np.mean([s['norm_max'] for s in input_stats])
                
                logger.info("=" * 60)
                logger.info("INPUT DISTRIBUTION DIAGNOSTIC (First Batch)")
                logger.info("=" * 60)
                logger.info(f"  Raw (0-65535):  Mean={avg_raw_mean:.2f}, Std={avg_raw_std:.2f}")
                logger.info(f"  Normalized:     Mean={avg_norm_mean:.6f}, Std={avg_norm_std:.6f}")
                logger.info(f"  Normalized:     Min={avg_norm_min:.4f}, Max={avg_norm_max:.4f}")
                logger.info("=" * 60)
                
                # 检查是否异常
                if abs(avg_norm_mean) > 0.1:
                    logger.warning(f"⚠️ 归一化后 Mean 不为 0 (实际={avg_norm_mean:.4f})，检查归一化逻辑！")
                if avg_norm_std < 0.5 or avg_norm_std > 2.0:
                    logger.warning(f"⚠️ 归一化后 Std 异常 (实际={avg_norm_std:.4f})，预期接近 1.0")
            
            # Stack -> GPU
            input_tensor = torch.stack(tensors).to(self.device)  # (B, 1, Z, H, W)
            
            # 推理 (FP16)
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    output = self.model(input_tensor)  # (B, 1, H, W)
                    preds = torch.sigmoid(output).squeeze(1).float().cpu().numpy()
                    
                    # 处理 B=1 的 squeeze 问题
                    if B == 1 and len(preds.shape) == 2:
                        preds = np.expand_dims(preds, 0)
            
            # 累加 TTA
            tta_sum += preds
            tta_count += 1
            
            # 清理 GPU
            del input_tensor, output, preds
            torch.cuda.empty_cache()
        
        # TTA 平均
        if tta_count > 0:
            final_preds = tta_sum / tta_count
        else:
            final_preds = tta_sum
        
        # === 高斯加权累加 ===
        # 核心改进：用高斯权重矩阵代替简单的 +1 计数
        for i, (y, x) in enumerate(batch_coords):
            # 预测值 × 高斯权重 → 加权预测
            pred_weighted_sum[y:y+self.tile_size, x:x+self.tile_size] += \
                final_preds[i] * self.gaussian_weight
            
            # 累加权重（而非简单 +1）
            weight_sum[y:y+self.tile_size, x:x+self.tile_size] += \
                self.gaussian_weight
        
        del tta_sum, final_preds
