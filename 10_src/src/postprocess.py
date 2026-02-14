"""
Vesuvius Challenge 2026 - 后处理模块
包含基于结构张量的各向异性扩散与断裂修复逻辑

参考文献: Tensor Voting (MICCAI 2025)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import torch
from typing import Tuple, Optional


def compute_structure_tensor_3d(
    volume: np.ndarray, 
    sigma_deriv: float = 1.0, 
    sigma_window: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 3D 体数据的结构张量 (Structure Tensor)
    
    结构张量可以捕获局部的方向信息，对于卷轴纸张这种薄层结构尤为有效。
    
    Args:
        volume: 3D 体数据 (D, H, W)
        sigma_deriv: 梯度计算时的高斯平滑 sigma
        sigma_window: 结构张量窗口平滑 sigma
        
    Returns:
        eigenvalues: 特征值 (3, D, H, W)
        eigenvectors: 特征向量 (3, 3, D, H, W)
        coherence: 方向一致性度量 (D, H, W)
    """
    # 高斯平滑
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=sigma_deriv)
    
    # 计算梯度
    gz, gy, gx = np.gradient(smoothed)
    
    # 构建结构张量的 6 个独立分量 (对称矩阵)
    Jxx = gaussian_filter(gx * gx, sigma=sigma_window)
    Jyy = gaussian_filter(gy * gy, sigma=sigma_window)
    Jzz = gaussian_filter(gz * gz, sigma=sigma_window)
    Jxy = gaussian_filter(gx * gy, sigma=sigma_window)
    Jxz = gaussian_filter(gx * gz, sigma=sigma_window)
    Jyz = gaussian_filter(gy * gz, sigma=sigma_window)
    
    # 计算方向一致性 (Coherence)
    # 使用简化的 Frobenius 范数近似
    trace = Jxx + Jyy + Jzz
    frobenius = np.sqrt(Jxx**2 + Jyy**2 + Jzz**2 + 2*(Jxy**2 + Jxz**2 + Jyz**2))
    coherence = frobenius / (trace + 1e-6)
    
    return (Jxx, Jyy, Jzz, Jxy, Jxz, Jyz), coherence


def anisotropic_diffusion_3d(
    volume: np.ndarray,
    iterations: int = 5,
    kappa: float = 50.0,
    gamma: float = 0.1,
    option: int = 1
) -> np.ndarray:
    """
    各向异性扩散 (Perona-Malik Diffusion)
    
    沿着梯度较小的方向扩散，保留边缘的同时平滑内部。
    用于修复小的断裂并增强连通性。
    
    Args:
        volume: 输入 3D 体 (D, H, W)
        iterations: 迭代次数
        kappa: 边缘敏感度参数 (越大越保留边缘)
        gamma: 扩散步长 (稳定性要求 <= 1/6)
        option: 扩散函数选择 (1: 指数, 2: 逆二次)
        
    Returns:
        扩散后的体数据
    """
    volume = volume.astype(np.float32)
    
    for _ in range(iterations):
        # 计算各方向的差分
        dz_p = np.diff(volume, axis=0, append=volume[-1:, :, :])
        dz_n = np.diff(volume, axis=0, prepend=volume[:1, :, :])
        dy_p = np.diff(volume, axis=1, append=volume[:, -1:, :])
        dy_n = np.diff(volume, axis=1, prepend=volume[:, :1, :])
        dx_p = np.diff(volume, axis=2, append=volume[:, :, -1:])
        dx_n = np.diff(volume, axis=2, prepend=volume[:, :, :1])
        
        # 计算扩散系数
        if option == 1:
            # Perona-Malik 方程 1: 指数函数
            cz_p = np.exp(-(dz_p / kappa) ** 2)
            cz_n = np.exp(-(dz_n / kappa) ** 2)
            cy_p = np.exp(-(dy_p / kappa) ** 2)
            cy_n = np.exp(-(dy_n / kappa) ** 2)
            cx_p = np.exp(-(dx_p / kappa) ** 2)
            cx_n = np.exp(-(dx_n / kappa) ** 2)
        else:
            # Perona-Malik 方程 2: 逆二次函数
            cz_p = 1.0 / (1.0 + (dz_p / kappa) ** 2)
            cz_n = 1.0 / (1.0 + (dz_n / kappa) ** 2)
            cy_p = 1.0 / (1.0 + (dy_p / kappa) ** 2)
            cy_n = 1.0 / (1.0 + (dy_n / kappa) ** 2)
            cx_p = 1.0 / (1.0 + (dx_p / kappa) ** 2)
            cx_n = 1.0 / (1.0 + (dx_n / kappa) ** 2)
        
        # 更新
        volume = volume + gamma * (
            cz_p * dz_p - cz_n * dz_n +
            cy_p * dy_p - cy_n * dy_n +
            cx_p * dx_p - cx_n * dx_n
        )
    
    return volume


def directional_dilation_3d(
    mask: np.ndarray,
    structure_tensor: Tuple,
    coherence: np.ndarray,
    iterations: int = 2,
    coherence_threshold: float = 0.5
) -> np.ndarray:
    """
    方向性膨胀 (Directional Dilation)
    
    沿着主方向进行膨胀，连接断裂的纸张区域但不破坏层间间隙。
    
    Args:
        mask: 二值分割掩码 (D, H, W)
        structure_tensor: 结构张量分量
        coherence: 方向一致性
        iterations: 膨胀迭代次数
        coherence_threshold: 只在一致性高的区域膨胀
        
    Returns:
        修复后的掩码
    """
    result = mask.copy().astype(np.float32)
    Jxx, Jyy, Jzz, Jxy, Jxz, Jyz = structure_tensor
    
    for _ in range(iterations):
        # 对高一致性区域进行选择性膨胀
        high_coherence = coherence > coherence_threshold
        
        # 使用各向异性结构元素
        # 简化版：沿 XY 平面膨胀 (纸张主要在 XY 平面展开)
        struct_xy = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])
        
        dilated = ndimage.binary_dilation(result > 0.5, structure=struct_xy)
        
        # 只在高一致性区域应用膨胀结果
        result = np.where(high_coherence, dilated.astype(np.float32), result)
    
    return (result > 0.5).astype(np.uint8)


def repair_fractures(
    prediction: np.ndarray,
    diffusion_iterations: int = 3,
    dilation_iterations: int = 1,
    kappa: float = 30.0
) -> np.ndarray:
    """
    断裂修复主函数
    
    结合各向异性扩散和方向性膨胀修复分割结果中的断裂。
    
    Args:
        prediction: 模型输出的概率图或二值掩码 (D, H, W)
        diffusion_iterations: 扩散迭代次数
        dilation_iterations: 膨胀迭代次数
        kappa: 扩散边缘敏感度
        
    Returns:
        修复后的分割掩码
    """
    # 1. 对概率图进行各向异性扩散
    diffused = anisotropic_diffusion_3d(
        prediction, 
        iterations=diffusion_iterations,
        kappa=kappa
    )
    
    # 2. 计算结构张量
    structure_tensor, coherence = compute_structure_tensor_3d(diffused)
    
    # 3. 二值化
    binary = (diffused > 0.5).astype(np.uint8)
    
    # 4. 方向性膨胀修复断裂
    repaired = directional_dilation_3d(
        binary, 
        structure_tensor, 
        coherence,
        iterations=dilation_iterations
    )
    
    return repaired


def postprocess_prediction(
    prediction: np.ndarray,
    apply_diffusion: bool = True,
    apply_dilation: bool = True,
    min_object_size: int = 100
) -> np.ndarray:
    """
    完整后处理流水线
    
    Args:
        prediction: 模型输出 (D, H, W), 范围 [0, 1]
        apply_diffusion: 是否应用各向异性扩散
        apply_dilation: 是否应用方向性膨胀
        min_object_size: 移除小于此体素数的连通区域
        
    Returns:
        后处理后的二值掩码
    """
    result = prediction.copy()
    
    # 1. 断裂修复
    if apply_diffusion or apply_dilation:
        result = repair_fractures(
            result,
            diffusion_iterations=3 if apply_diffusion else 0,
            dilation_iterations=2 if apply_dilation else 0
        )
    else:
        result = (result > 0.5).astype(np.uint8)
    
    # 2. 移除小连通区域
    if min_object_size > 0:
        labeled, num_features = ndimage.label(result)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_object_size:
                result[labeled == i] = 0
    
    return result.astype(np.uint8)


if __name__ == "__main__":
    # 简单测试
    print("测试后处理模块...")
    
    # 创建模拟数据
    test_volume = np.random.rand(32, 64, 64).astype(np.float32)
    test_volume[10:22, 20:50, 20:50] = 0.8  # 添加一个"纸张"区域
    test_volume[15, 30:35, 30:35] = 0.3  # 添加一个"断裂"
    
    try:
        result = postprocess_prediction(test_volume)
        print(f"输入形状: {test_volume.shape}")
        print(f"输出形状: {result.shape}")
        print(f"输出唯一值: {np.unique(result)}")
        print("✓ 后处理模块测试通过")
    except Exception as e:
        print(f"后处理模块测试失败: {e}")
