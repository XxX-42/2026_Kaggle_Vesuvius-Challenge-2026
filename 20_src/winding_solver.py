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
