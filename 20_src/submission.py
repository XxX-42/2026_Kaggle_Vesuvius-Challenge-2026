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
