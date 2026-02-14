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
