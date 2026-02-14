"""
Vesuvius Challenge 2026 - 实验管理与通用工具库
包含实验目录管理、代码备份、日志记录和可视化绘图功能。
"""

import os
import sys
import shutil
import json
import time
import datetime
from pathlib import Path
import csv
import torch

# [Environment Config] 精确指向项目根目录下的 .venv
# 确保在任何 Python 环境下运行时，都能优先加载项目内 .venv 的库 (如 pyvista)
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent # 10_src -> root
_venv_site = _project_root / ".venv" / "Lib" / "site-packages"

if _venv_site.exists():
    _venv_site_str = str(_venv_site)
    if _venv_site_str not in sys.path:
        # 插在 sys.path[1] (仅次于脚本当前目录)，优先于系统库
        sys.path.insert(1, _venv_site_str)
        print(f"[Env] Loading dependencies from: {_venv_site}")

import matplotlib
matplotlib.use('Agg')  # [Fix] 强制使用无头后端，避免 Qt 初始化报错
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
try:
    import pynvml
except ImportError:
    pynvml = None


class ExperimentManager:
    """
    实验管理器
    负责创建实验目录、备份代码、保存配置和初始化日志
    """
    def __init__(self, args, experiment_name="ResUNet", code_dir="src"):
        self.args = args
        self.code_dir = Path(code_dir).resolve()
        
        # 生成带时间戳的实验名称
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{experiment_name}_{timestamp}"
        
        # 创建输出目录
        # 如果 args.output_dir 是 outputs/models 这种子目录，我们取其父目录或直接用 outputs
        # 这里为了符合用户要求的 outputs/ResUNet... 结构，我们假设 args.output_dir 是基础输出目录
        # 如果 args.output_dir 是默认的 outputs/models，我们向上取一级
        base_output_dir = Path(args.output_dir)
        if base_output_dir.name == "models":
            base_output_dir = base_output_dir.parent
            
        self.exp_dir = base_output_dir / self.exp_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.code_backup_dir = self.exp_dir / "code"
        
        self._setup_directories()
        self._backup_code()
        self._save_config()
        self._init_log()
        
        print(f"\n[Experiment] 实验初始化完成")
        print(f"  实验目录: {self.exp_dir}")
        
    def _setup_directories(self):
        """创建目录结构"""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.code_backup_dir.mkdir(exist_ok=True)
        
    def _backup_code(self):
        """备份源代码"""
        if not self.code_dir.exists():
            print(f"[Warning] 代码目录 {self.code_dir} 不存在，跳过备份")
            return
            
        # 复制所有 .py 文件
        for file in self.code_dir.glob("*.py"):
            shutil.copy2(file, self.code_backup_dir)
            
    def _save_config(self):
        """保存训练参数"""
        config_path = self.exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(vars(self.args), f, indent=4)
            
    def _init_log(self):
        """初始化 CSV 日志"""
        self.log_path = self.exp_dir / "training_log.csv"
        # 只有文件不存在时才写入 header
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "clDice", "lr", "time"])
    
    def log_epoch(self, epoch_data):
        """
        记录一个 Epoch 的数据
        epoch_data: dict, 必须包含 header 中的字段
        """
        try:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch_data.get("epoch"),
                    f"{epoch_data.get('train_loss'):.6f}",
                    f"{epoch_data.get('val_loss'):.6f}",
                    f"{epoch_data.get('val_dice'):.6f}",
                    f"{epoch_data.get('clDice', 0):.6f}", # Default to 0 if not present
                    f"{epoch_data.get('lr'):.2e}",
                    f"{epoch_data.get('time'):.2f}"
                ])
        except Exception as e:
            print(f"[Warning] Failed to write log: {e}")
            
    def get_checkpoint_path(self, name="best_model.pth"):
        return str(self.checkpoint_dir / name)
    
    def get_log_path(self):
        return str(self.log_path)
    
    def get_exp_dir(self):
        return str(self.exp_dir)


def plot_training_curves(log_path, save_dir, patch_data=None):
    """
    绘制训练曲线 (Loss 和 Dice) + 可选的 Patch 可视化
    
    Args:
        log_path: training_log.csv 的路径
        save_dir: 图片保存目录
        patch_data: 可选的 Patch 可视化数据 dict，包含:
            - 'raw': (D, H, W) 原始 CT
            - 'gt': (D, H, W) Ground Truth
            - 'pred': (D, H, W) 预测概率图 (0-1)
    """
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"[Warning] 无法读取日志文件进行绘图: {e}")
        return

    if len(df) < 1:
        return

    epochs = df["epoch"]
    
    # 设置绘图风格
    plt.style.use('ggplot')
    
    # 根据是否有 patch_data 决定布局
    if patch_data is not None:
        fig = plt.figure(figsize=(16, 12))
        # 上半部分：折线图 (2 列)
        ax1 = fig.add_subplot(2, 4, (1, 2))
        ax2 = fig.add_subplot(2, 4, (3, 4))
        # 下半部分：Patch 可视化 (4 列)
        ax_raw = fig.add_subplot(2, 4, 5)
        ax_gt = fig.add_subplot(2, 4, 6)
        ax_pred = fig.add_subplot(2, 4, 7)
        ax_overlay = fig.add_subplot(2, 4, 8)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图 1: Loss 曲线
    ax1.plot(epochs, df["train_loss"], label="Train Loss", marker='.', color='tab:blue')
    ax1.plot(epochs, df["val_loss"], label="Val Loss", marker='.', color='tab:orange')
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # 子图 2: Dice Score 曲线
    ax2.plot(epochs, df["val_dice"], label="Val Dice", marker='o', color='tab:green', linewidth=2)
    ax2.set_title("Validation Dice Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.legend()
    ax2.grid(True)
    
    # 标注最佳点
    if len(df) > 0:
        best_idx = df["val_dice"].idxmax()
        best_epoch = df.iloc[best_idx]["epoch"]
        best_dice = df.iloc[best_idx]["val_dice"]
        ax2.annotate(f"Best: {best_dice:.4f} (Ep {best_epoch})", 
                     xy=(best_epoch, best_dice), 
                     xytext=(best_epoch, best_dice - 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Patch 可视化
    if patch_data is not None:
        try:
            import numpy as np
            raw = patch_data['raw']
            gt = patch_data['gt']
            pred = patch_data['pred']
            
            # 取中间切片
            mid_slice = raw.shape[0] // 2
            raw_slice = raw[mid_slice]
            gt_slice = gt[mid_slice]
            pred_slice = pred[mid_slice]
            
            # 自定义颜色映射: 0=空(透明), 1=黑(纸张), 2=噪声(透明)
            from matplotlib.colors import ListedColormap, BoundaryNorm
            vesuvius_cmap = ListedColormap([
                (1, 1, 1, 0.0),       # val=0: 完全透明
                (0, 0, 0, 1.0),       # val=1: 黑色, 不透明
                (0.5, 0.5, 0.5, 0.0), # val=2: 灰色, 完全透明 (噪声隐藏)
            ])
            vesuvius_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)
            
            # 1. Raw CT (0=白/空气, 1=黑/高密度, 2=灰/噪声)
            ax_raw.imshow(raw_slice, cmap=vesuvius_cmap, norm=vesuvius_norm, interpolation='nearest')
            ax_raw.set_title("Raw CT")
            ax_raw.axis('off')
            
            # 2. Ground Truth (0=白/背景, 1=黑/表面, 2=灰/噪声)
            ax_gt.imshow(gt_slice, cmap=vesuvius_cmap, norm=vesuvius_norm, interpolation='nearest')
            ax_gt.set_title("Ground Truth")
            ax_gt.axis('off')
            
            # 3. Prediction Heatmap (低置信透明)
            # 创建带 Alpha 通道的热力图：prob < 0.2 完全透明
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=0, vmax=1)
            cmap_hot = plt.cm.get_cmap('hot_r')
            pred_rgba = cmap_hot(norm(pred_slice))  # (H, W, 4) RGBA
            # 设置 Alpha: prob < 0.2 -> 透明, prob > 0.5 -> 不透明
            pred_rgba[:, :, 3] = np.clip((pred_slice - 0.2) / 0.3, 0, 1)
            
            ax_pred.set_facecolor('white')
            im = ax_pred.imshow(pred_rgba)
            ax_pred.set_title("Prediction Heatmap")
            ax_pred.axis('off')
            # 手动添加 colorbar (因为 imshow RGBA 不自动关联 norm)
            sm = plt.cm.ScalarMappable(cmap='hot_r', norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax_pred, fraction=0.046, pad=0.04)
            
            # 4. Overlay (预测 Mask 叠加在 CT 上)
            mask = (pred_slice > 0.5).astype(np.float32)
            overlay = np.stack([raw_slice] * 3, axis=-1)
            # 归一化到 0-1
            overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
            # 红色叠加预测区域
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask * 0.5, 0, 1)
            overlay[:, :, 1] = overlay[:, :, 1] * (1 - mask * 0.3)
            overlay[:, :, 2] = overlay[:, :, 2] * (1 - mask * 0.3)
            ax_overlay.imshow(overlay)
            ax_overlay.set_title("Overlay (Pred > 0.5)")
            ax_overlay.axis('off')
            
        except Exception as e:
            print(f"[Warning] Patch 可视化失败: {e}")

    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_dir) / "metrics.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    # print(f"[Plot] 曲线图已更新: {save_path}")



def plot_multi_patch_comparison(patch_list, save_dir, epoch=0):
    """
    绘制多 Patch 对比网格 (4x4 布局)
    
    Layout: [Raw CT | Ground Truth | Prediction Heatmap | Overlay]
    
    Aesthetics:
    - Heatmap: Magma (更高级的感知色图)
    - Overlay: 红色半透明预测 + 绿色轮廓 GT
    - Remove axis ticks
    """
    import numpy as np
    
    n_patches = min(4, len(patch_list))  # 最多展示 4 个 Patch
    
    if n_patches == 0:
        return
    
    # Grid: n_patches x 4 columns
    fig, axes = plt.subplots(n_patches, 4, figsize=(16, 4 * n_patches))
    
    # 如果只有一个 Patch，确保 axes 是二维的
    if n_patches == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_patches):
        patch = patch_list[i]
        raw = patch['raw']
        gt = patch['gt']
        pred = patch['pred']
        
        # 取中间切片
        mid_slice = raw.shape[0] // 2
        raw_slice = raw[mid_slice]
        gt_slice = gt[mid_slice]
        pred_slice = pred[mid_slice]
        
        # [Column 1] Raw CT
        axes[i, 0].imshow(raw_slice, cmap='gray')
        if i == 0: axes[i, 0].set_title("Input CT\n(Raw)", fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # [Column 2] Ground Truth
        axes[i, 1].imshow(gt_slice, cmap='gray')
        if i == 0: axes[i, 1].set_title("Ground Truth\n(Label)", fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # [Column 3] Prediction Heatmap (Magma)
        im = axes[i, 2].imshow(pred_slice, cmap='magma', vmin=0, vmax=1)
        if i == 0: axes[i, 2].set_title("Prediction\n(Confidence)", fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # 添加颜色条 (仅第一行，或者更小)
        # if i == 0:
        #     cbar = plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # [Column 4] Overlay (Premium Look)
        # Background: Raw CT (Darkened)
        bg = (raw_slice - raw_slice.min()) / (raw_slice.max() - raw_slice.min() + 1e-8)
        bg = np.stack([bg]*3, axis=-1) * 0.8 
        
        # Pred Mask (Red)
        pred_mask = (pred_slice > 0.5).astype(np.float32)
        
        # GT Contour (Green) or Fill? Let's use GT as Green Mask
        # But if GT is sparse, maybe Green Fill is better.
        gt_mask = (gt_slice > 0.5).astype(np.float32)
        
        # Compose
        overlay = bg.copy()
        # Red channel for Pred
        overlay[..., 0] = np.clip(overlay[..., 0] + pred_mask * 0.5, 0, 1)
        # Green channel for GT
        overlay[..., 1] = np.clip(overlay[..., 1] + gt_mask * 0.3, 0, 1)
        
        axes[i, 3].imshow(overlay)
        if i == 0: axes[i, 3].set_title("Overlay\n(Red=Pred, Green=GT)", fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')

    plt.suptitle(f"Epoch {epoch} - Patch Analysis (20_src Style)", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_dir) / f"patch_comparison_ep{epoch:03d}.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight') # High DPI
    plt.close()


def format_time(seconds):
    """将秒数格式化为 MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class DynamicGPUManager:
    """
    动态 GPU 资源管理器
    监控显存和算力占用，并提供动态训练策略建议
    """
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.nvml_available = False
        try:
            if pynvml:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.nvml_available = True
                print(f"[GPU Manager] NVML initialized for device {device_index}")
            else:
                print("[GPU Manager] pynvml library not found. dynamic strategy disabled.")
        except Exception as e:
            print(f"[GPU Manager] Failed to initialize NVML: {e}")

    def get_status(self):
        """获取当前 GPU 状态"""
        if self.nvml_available:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                
                return {
                    "memory_util": mem_info.used / mem_info.total * 100,
                    "compute_util": util_info.gpu,
                    "memory_used": mem_info.used / 1024**2, # MB
                    "memory_total": mem_info.total / 1024**2 # MB
                }
            except Exception as e:
                # print(f"[GPU Manager] Error getting status: {e}")
                pass

            
        # Fallback: 使用 PyTorch API (仅反映当前进程占用)
        if torch.cuda.is_available():
            try:
                dev = torch.device(f"cuda:{self.device_index}")
                # memory_reserved 是 PyTorch 缓存管理器持有的显存
                mem_used = torch.cuda.memory_reserved(dev) 
                mem_total = torch.cuda.get_device_properties(dev).total_memory
                return {
                    "memory_util": mem_used / mem_total * 100,
                    "compute_util": 0, # torch 无法获取计算利用率
                    "memory_used": mem_used / 1024**2,
                    "memory_total": mem_total / 1024**2
                }
            except:
                pass

        return {"memory_util": 0, "compute_util": 0, "memory_used": 0, "memory_total": 0}

    def suggest_accumulation_steps(self, current_steps, memory_util):
        """
        根据显存占用建议梯度累积步数
        策略：
        - 显存占用 < 70%: 建议增加 Batch Size (通过控制台提示)
        - 显存占用 > 90%: 负载较高，保持现状
        - 算力低 (implied by low compute_util monitored externally): 可以尝试增加 accumulation steps 来模拟更大 batch
        """
        # 简单策略：如果显存占用过低，提示用户
        if memory_util < 70:
            print(f"  [Suggestion] GPU Memory Util is low ({memory_util:.1f}%). Consider increasing Batch Size.")
            
        # 这里暂时不动态调整 steps，因为改变 accumulation steps 会影响训练的等效 batch size，
        # 需要配合学习率调整，比较复杂。主要作为监控和提示工具。
        return current_steps

    def __del__(self):
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass




def plot_3d_comparison(patch_data, save_dir, epoch, sample_idx):
    """
    绘制 3D Voxel 可视化 (导出 TIF，无 GUI)
    """
    # 直接使用优化的 TIF 导出函数，不再依赖 PyVista/Qt
    try:
        _plot_3d_pyvista(patch_data, save_dir, epoch, sample_idx)
    except Exception as e:
        print(f"[Warning] 3D TIF Export error: {e}. Falling back to Matplotlib slice.")
        _plot_3d_matplotlib(patch_data, save_dir, epoch, sample_idx)


def _plot_3d_pyvista(patch_data, save_dir, epoch, sample_idx):
    """
    导出 3D 对比图 PNG (off_screen) + 三合一 TIF。
    使用 PYVISTA_OFF_SCREEN 环境变量阻止 Qt 初始化。
    """
    import numpy as np
    import tifffile
    
    # [Fix] 在 import pyvista 之前强制设置 off_screen，避免 Qt 初始化
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    import pyvista as pv
    
    raw = patch_data['raw'] # (D, H, W)
    gt = patch_data['gt']
    pred = patch_data['pred']
    
    vis_dir = Path(save_dir) / "epoch_vis"
    vis_dir.mkdir(exist_ok=True)
    
    print(f"\n[3D Export] Processing Epoch {epoch} Sample {sample_idx}...")

    # === 1. 生成 3D 对比图 PNG (Off-Screen) ===
    try:
        # Robust Normalization for Raw CT
        p01, p99 = np.percentile(raw, [1, 99])
        raw_norm = np.clip((raw - p01) / (p99 - p01 + 1e-8), 0, 1)
        raw_u8 = (raw_norm * 255).astype(np.uint8)
        
        # 创建 PyVista 网格
        grid_raw = pv.wrap(raw_u8)
        grid_gt = pv.wrap(gt.astype(np.float32))
        grid_pred = pv.wrap(pred.astype(np.float32))
        
        # Setup Plotter (完全 off_screen，不涉及任何 GUI)
        pv.set_plot_theme("document")
        p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 600))
        p.set_background('white')
        
        # --- Panel 1: Raw CT (Vesuvius 颜色规则) ---
        # val=0(空气)→白+透明, val≈128(纸张)→黑+不透明, val=255(噪声)→灰+30%透明
        p.subplot(0, 0)
        p.add_text(f"Raw CT (Ep {epoch})", font_size=12, color='black')
        
        from matplotlib.colors import LinearSegmentedColormap
        raw_cmap = LinearSegmentedColormap.from_list('vesuvius_raw', [
            (0.0, 'white'),   # val=0: 白色 (空气)
            (0.5, 'black'),   # val≈128: 黑色 (纸张/墨迹)
            (1.0, 'gray'),    # val=255: 灰色 (噪声)
        ])
        # 三段式 Opacity: 0-80(透明), 100-140(不透明), 160-255(透明/噪声)
        opacity_ct = [0, 0.0, 80, 0.0, 100, 0.8, 128, 1.0, 140, 0.8, 160, 0.0, 255, 0.0]
        p.add_volume(grid_raw, cmap=raw_cmap, opacity=opacity_ct, show_scalar_bar=False, blending="composite")
        p.add_bounding_box(color='black')
        
        # --- Panel 2: GT (Green) ---
        p.subplot(0, 1)
        p.add_text("Ground Truth", font_size=12, color='black')
        opacity_gt = [0.0, 0.0, 0.1, 0.0, 0.9, 0.6, 1.0, 0.8]
        p.add_volume(grid_gt, cmap=["black", "green"], opacity=opacity_gt, show_scalar_bar=False, blending="composite")
        p.add_bounding_box(color='black')
        
        # --- Panel 3: Prediction (Hot) ---
        p.subplot(0, 2)
        p.add_text("Prediction", font_size=12, color='black')
        opacity_pred = [0.0, 0.0, 0.2, 0.0, 0.5, 0.2, 0.8, 0.5, 1.0, 0.8]
        p.add_volume(grid_pred, cmap="hot", opacity=opacity_pred, show_scalar_bar=True, blending="composite")
        p.add_bounding_box(color='black')
        
        p.link_views()
        
        # 保存 PNG 截图
        png_path = Path(save_dir) / f"3d_vis_ep{epoch:03d}_{sample_idx}.png"
        p.screenshot(str(png_path))
        p.close()
        print(f"  📸 3D 对比图已导出: {png_path.name}")
        
    except Exception as pe:
        print(f"  ⚠️ 3D PNG Render Error: {pe}")

    # === 2. 导出三合一 3D TIFF (Raw | GT | Pred) ===
    try:
        # 统一转换到 [0, 255] uint8
        if raw.max() > 0:
            disp_raw = (raw / raw.max() * 255).astype(np.uint8)
        else:
            disp_raw = raw.astype(np.uint8)
        disp_gt  = (gt * 255).astype(np.uint8)
        disp_pred = (pred * 255).astype(np.uint8)
        
        # 水平拼接 (Depth, Height, 3*Width)
        combined_tif = np.concatenate([disp_raw, disp_gt, disp_pred], axis=2)
        
        tif_path = vis_dir / f"epoch{epoch:03d}_vols_combined.tif"
        tifffile.imwrite(str(tif_path), combined_tif, compression='zlib')
        print(f"  🎞️  三合一 3D TIFF 已导出: {tif_path.name} (形状: {combined_tif.shape})")
        
    except Exception as te:
        print(f"  ⚠️ TIFF Export Error: {te}")


def _plot_3d_matplotlib(patch_data, save_dir, epoch=0, sample_idx=0):
    """
    绘制 3D Voxel 可视化对比 (Matplotlib Fallback)
    """
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    
    raw = patch_data['raw']
    gt = patch_data['gt']
    pred = patch_data['pred']
    
    # 下采样以提高绘图速度
    stride = 2
    raw_s = raw[::stride, ::stride, ::stride]
    gt_s = gt[::stride, ::stride, ::stride]
    pred_s = pred[::stride, ::stride, ::stride]
    
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Raw CT (Point Cloud based on intensity)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    # 选取高亮区域绘制
    threshold = np.percentile(raw_s, 90)
    mask = raw_s > threshold
    x, y, z = np.where(mask)
    vals = raw_s[mask]
    
    # limit points
    if len(x) > 10000:
        choice_idx = np.random.choice(len(x), 10000, replace=False)
        x, y, z = x[choice_idx], y[choice_idx], z[choice_idx]
        vals = vals[choice_idx]
        
    ax1.scatter(x, y, z, c=vals, cmap='gray', s=1, alpha=0.3)
    ax1.set_title(f"3D Raw CT (Top 10%)\nEp {epoch}")
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # 2. Ground Truth (Label)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    mask = gt_s > 0.5
    x, y, z = np.where(mask)
    # GT 通常点较少，但也可能很多
    if len(x) > 0:
        if len(x) > 10000:
             choice_idx = np.random.choice(len(x), 10000, replace=False)
             x, y, z = x[choice_idx], y[choice_idx], z[choice_idx]
        ax2.scatter(x, y, z, c='green', s=1, alpha=0.5)
    ax2.set_title("3D Label (Green)")
    ax2.set_xlim(0, raw_s.shape[0]); ax2.set_ylim(0, raw_s.shape[1]); ax2.set_zlim(0, raw_s.shape[2])
    
    # 3. Prediction
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    mask = pred_s > 0.5
    x, y, z = np.where(mask)
    vals = pred_s[mask]
    
    if len(x) > 0:
        if len(x) > 10000:
             choice_idx = np.random.choice(len(x), 10000, replace=False)
             x, y, z = x[choice_idx], y[choice_idx], z[choice_idx]
             vals = vals[choice_idx]
        
        ax3.scatter(x, y, z, c=vals, cmap='magma', s=1, alpha=0.5)
    ax3.set_title("3D Prediction (Magma)")
    ax3.set_xlim(0, raw_s.shape[0]); ax3.set_ylim(0, raw_s.shape[1]); ax3.set_zlim(0, raw_s.shape[2])
    
    plt.tight_layout()
    save_path = Path(save_dir) / f"3d_vis_ep{epoch:03d}_{sample_idx}.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    

def generate_high_res_sample(model, dataset, save_path, device, patch_size=64, roi_size=256, stride=32):
    """
    生成高清 (256x256x256) 预测样本
    使用滑动窗口推理拼接大图，用于观察模型在更大视野下的表现。
    
    Args:
        model: 训练中的模型
        dataset: VesuviusDataset 实例 (用于访问底层大图)
        save_path: TIF 保存路径
        device:计算设备
        patch_size: 模型训练时的 patch 尺寸 (默认 64)
        roi_size: 目标生成的区域大小 (默认 256)
        stride: 滑动步长 (默认 32, 即 50% 重叠)
    """
    import numpy as np
    import tifffile
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    
    # 1. 寻找一个包含丰富细节的区域 (正样本区域)
    # 我们尝试从 dataset 的 csv 中随机选一个 ID，然后尝试定位正样本中心
    # 由于 dataset 已经封装好了复杂逻辑，我们这里直接利用 dataset 的内部方法稍微 hack 一下
    # 或者简单点：随机从 dataset 中取一个样本，获取其坐标，然后扩展
    
    # 为了简化，我们直接遍历 dataset 直到找到一个 positive ratio 较高的样本作为中心
    # 限制尝试次数避免死循环
    center_sample = None
    for _ in range(20):
        idx = np.random.randint(0, len(dataset))
        # 这里的 dataset[idx] 返回的是 crop 后的，无法得知原始坐标
        # 我们需要访问 dataset 的底层 volume。
        # VesuviusDataset 没有公开坐标信息。
        # 妥协方案：重新读取一个固定的测试区域，或者让 Dataset 暴露 volume。
        pass
        
    # [Better Approach] 从 dataset.image_root 随机读取一个文件，随机裁剪 256x256x256
    # 假设 dataset 已经加载了数据列表
    try:
        sample_id = dataset.df.iloc[np.random.randint(0, len(dataset.df))]['id']
        image_vol, label_vol = dataset._load_data(str(sample_id))
        
        # 寻找包含 mask 的区域
        d, h, w = label_vol.shape
        found = False
        for _ in range(50): # 尝试 50 次寻找有内容的区域
            lz = np.random.randint(0, max(1, d - roi_size))
            ly = np.random.randint(0, max(1, h - roi_size))
            lx = np.random.randint(0, max(1, w - roi_size))
            
            # 检查 label 是否有内容
            roi_label = label_vol[lz:lz+roi_size, ly:ly+roi_size, lx:lx+roi_size]
            if np.any(roi_label == 1):
                found = True
                break
        
        if not found:
            print("[HighRes] ⚠️ 未找到包含正样本的 256x256x256 区域，使用最后一次随机位置")
            
        roi_raw = image_vol[lz:lz+roi_size, ly:ly+roi_size, lx:lx+roi_size]
        roi_label = label_vol[lz:lz+roi_size, ly:ly+roi_size, lx:lx+roi_size]
        
        # 转换为 float32 并归一化 (模拟 Dataset 行为)
        if roi_raw.dtype == np.uint16:
            roi_raw_norm = roi_raw.astype(np.float32) / 65535.0
        else:
            roi_raw_norm = roi_raw.astype(np.float32) / 255.0
            
    except Exception as e:
        print(f"[HighRes] 数据加载失败: {e}")
        return

    # 2. 初始化结果容器 (加权平均)
    prob_map = torch.zeros((roi_size, roi_size, roi_size), device=device, dtype=torch.float16)
    weight_map = torch.zeros((roi_size, roi_size, roi_size), device=device, dtype=torch.float16)
    
    # [Fix] 使用全 1 权重 (平均拼接) 替代高斯权重
    # 高斯权重在边缘衰减太快 (exp(-8) ~ 0.0003)，导致边缘区域归一化时数值爆炸 (Artifacts)
    # 平均拼接虽然可能有一点块效应，但数值绝对稳定，不会出现空心或边缘红点。
    patch_weight = torch.ones((patch_size, patch_size, patch_size), device=device, dtype=torch.float16)

    # 3. 滑动窗口推理
    # 坐标范围
    z_steps = list(range(0, roi_size - patch_size + 1, stride))
    y_steps = list(range(0, roi_size - patch_size + 1, stride))
    x_steps = list(range(0, roi_size - patch_size + 1, stride))
    
    # 确保覆盖边缘
    if z_steps[-1] + patch_size < roi_size: z_steps.append(roi_size - patch_size)
    if y_steps[-1] + patch_size < roi_size: y_steps.append(roi_size - patch_size)
    if x_steps[-1] + patch_size < roi_size: x_steps.append(roi_size - patch_size)
    
    total_steps = len(z_steps) * len(y_steps) * len(x_steps)
    print(f"\n[HighRes] 生成 256³ 高清样本... (Patches: {total_steps}, ROI: {roi_size}³, Stride: {stride})")
    
    batch_patches = []
    batch_coords = []
    BATCH_SIZE = 16 # 推理 Batch Size
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for z in z_steps:
                for y in y_steps:
                    for x in x_steps:
                        # 提取 patch
                        patch = roi_raw_norm[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                        # 填充 (如果不足) - 其实上面的逻辑保证了不会不足，除非 roi_size < patch_size
                        if patch.shape != (patch_size, patch_size, patch_size):
                            continue # Should not happen
                            
                        # 转 Tensor
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0) # (1, D, H, W)
                        batch_patches.append(patch_tensor)
                        batch_coords.append((z, y, x))
                        
                        if len(batch_patches) >= BATCH_SIZE:
                            # 推理
                            inputs = torch.stack(batch_patches).to(device) # (B, 1, D, H, W)
                            
                            # 兼容不同模型输出
                            outputs = model(inputs)
                            if isinstance(outputs, tuple):
                                logits = outputs[0]
                            else:
                                logits = outputs
                            
                            probs = torch.sigmoid(logits).squeeze(1) # (B, D, H, W)
                            
                            # 累积
                            for i, (bz, by, bx) in enumerate(batch_coords):
                                prob_map[bz:bz+patch_size, by:by+patch_size, bx:bx+patch_size] += probs[i] * patch_weight
                                weight_map[bz:bz+patch_size, by:by+patch_size, bx:bx+patch_size] += patch_weight
                                
                            batch_patches = []
                            batch_coords = []
                            
            # 处理剩余 batch
            if len(batch_patches) > 0:
                inputs = torch.stack(batch_patches).to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple): logits = outputs[0]
                else: logits = outputs
                probs = torch.sigmoid(logits).squeeze(1)
                for i, (bz, by, bx) in enumerate(batch_coords):
                    prob_map[bz:bz+patch_size, by:by+patch_size, bx:bx+patch_size] += probs[i] * patch_weight
                    weight_map[bz:bz+patch_size, by:by+patch_size, bx:bx+patch_size] += patch_weight

    # 4. 归一化与保存
    # 避免除以零
    # 之前如果weight_map很小，prob_map如果也小就会得到巨大的噪声
    final_prob = (prob_map / (weight_map + 1e-6)).cpu().numpy().astype(np.float32)
    
    # 准备 exporting layers
    # Raw CT (uint8)
    if roi_raw.dtype != np.uint8:
        raw_u8 = (roi_raw_norm * 255).astype(np.uint8)
    else:
        raw_u8 = roi_raw
        
    # GT (uint8 0/255)
    # Label 中 1 是目标，2 是 ignore。这里只显示目标。
    gt_u8 = ((roi_label == 1) * 255).astype(np.uint8)
    
    # Pred (uint8 0-255)
    pred_u8 = (final_prob * 255).astype(np.uint8)
    
    # 拼接 (Depth, Height, 3*Width)
    combined_tif = np.concatenate([raw_u8, gt_u8, pred_u8], axis=2)
    
    try:
        tifffile.imwrite(str(save_path), combined_tif, compression='zlib')
        print(f"  ✅ 高清样本已保存: {save_path}")
    except Exception as e:
        print(f"  ❌ 保存失败: {e}")

    # 清理显存
    del prob_map, weight_map, final_prob, batch_patches
    torch.cuda.empty_cache()


def _generate_pointcloud_viewer_html(html_path, raw, gt, pred, epoch):
    """
    生成三视图点云 HTML 查看器 (完全自包含)
    - 三个并排视图: Raw CT / Ground Truth / Prediction
    - 只显示点 (Point Cloud), 不显示面 (Mesh)
    - 默认自动旋转, 中键点击切换旋转开关
    - 数据 JSON 内嵌, 可在 VS Code / 浏览器直接查看
    """
    import json as _json
    import numpy as np

    max_points = 150000

    def _extract_points(vol, threshold, max_n):
        mask = vol > threshold
        coords = np.argwhere(mask)
        values = vol[mask]
        if len(coords) > max_n:
            idx = np.random.choice(len(coords), max_n, replace=False)
            coords = coords[idx]
            values = values[idx]
        return coords.tolist(), values.tolist()

    # Raw CT
    raw_norm = raw.astype(np.float32)
    if raw_norm.max() > 0:
        raw_norm = raw_norm / raw_norm.max()
    raw_thresh = max(0.15, float(np.percentile(raw_norm[raw_norm > 0], 70)) if (raw_norm > 0).sum() > 0 else 0.15)
    raw_pts, raw_vals = _extract_points(raw_norm, raw_thresh, max_points)

    # GT
    gt_pts, gt_vals = _extract_points(gt.astype(np.float32), 0.5, max_points)

    # Pred
    pred_f = pred.astype(np.float32)
    pred_max_val = float(pred_f.max())
    pred_thresh = 0.5 if pred_max_val >= 0.5 else max(0.05, pred_max_val * 0.5)
    pred_pts, pred_vals = _extract_points(pred_f, pred_thresh, max_points)

    data_json = _json.dumps({
        "raw": {"pts": raw_pts, "vals": raw_vals},
        "gt":  {"pts": gt_pts,  "vals": gt_vals},
        "pred":{"pts": pred_pts,"vals": pred_vals},
        "shape": list(raw.shape),
    }, separators=(',', ':'))

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Epoch {epoch} - 3D Point Cloud</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f0f1a;font-family:'Segoe UI',sans-serif;overflow:hidden}}
#container{{display:flex;width:100vw;height:100vh}}
.panel{{flex:1;position:relative;border-right:1px solid #222}}
.panel:last-child{{border-right:none}}
.label{{position:absolute;top:8px;left:50%;transform:translateX(-50%);color:#ddd;font-size:13px;font-weight:600;background:rgba(0,0,0,.55);padding:4px 14px;border-radius:6px;pointer-events:none;white-space:nowrap}}
.label .s{{font-size:10px;font-weight:400;color:#999}}
#hud{{position:fixed;bottom:10px;left:50%;transform:translateX(-50%);color:#aaa;font-size:11px;background:rgba(0,0,0,.5);padding:5px 16px;border-radius:6px;pointer-events:none}}
#rb{{position:fixed;top:10px;right:14px;color:#fff;font-size:12px;background:rgba(80,200,120,.8);padding:3px 10px;border-radius:4px;pointer-events:none;transition:background .3s}}
#rb.off{{background:rgba(200,80,80,.8)}}
</style>
</head>
<body>
<div id="container">
<div class="panel" id="p0"><div class="label">Raw CT <span class="s">(亮度)</span></div></div>
<div class="panel" id="p1"><div class="label">Ground Truth <span class="s">(绿色)</span></div></div>
<div class="panel" id="p2"><div class="label">Prediction <span class="s">(Epoch {epoch})</span></div></div>
</div>
<div id="hud">左键旋转 | 右键平移 | 滚轮缩放 | <b>中键</b>切换自动旋转</div>
<div id="rb">🔄 旋转: ON</div>
<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
</script>
<script type="module">
import*as THREE from'three';
import{{OrbitControls}}from'three/addons/controls/OrbitControls.js';
const D={data_json};
let ar=true;
function magma(t){{t=Math.max(0,Math.min(1,t));return new THREE.Color(Math.min(1,t*3),Math.max(0,Math.min(1,(t-.3)*2.5)),Math.max(0,Math.min(1,(t-.6)*3.5)))}}
function mkPC(pts,vals,cm){{
    const g=new THREE.BufferGeometry();
    const pos=new Float32Array(pts.length*3),col=new Float32Array(pts.length*3);
    const s=D.shape,cx=s[0]/2,cy=s[1]/2,cz=s[2]/2;
    for(let i=0;i<pts.length;i++){{
        pos[i*3]=pts[i][0]-cx;pos[i*3+1]=pts[i][1]-cy;pos[i*3+2]=pts[i][2]-cz;
        let c;
        if(cm==='g'){{const v=vals[i];c=new THREE.Color(v*.9+.1,v*.9+.1,v*.95+.1)}}
        else if(cm==='gr')c=new THREE.Color(.2,.9,.3);
        else c=magma(vals[i]);
        col[i*3]=c.r;col[i*3+1]=c.g;col[i*3+2]=c.b;
    }}
    g.setAttribute('position',new THREE.BufferAttribute(pos,3));
    g.setAttribute('color',new THREE.BufferAttribute(col,3));
    return new THREE.Points(g,new THREE.PointsMaterial({{size:0.8,vertexColors:true,sizeAttenuation:true,transparent:true,opacity:.85}}));
}}
function mkView(id,pts,vals,cm){{
    const ct=document.getElementById(id);
    const sc=new THREE.Scene();sc.background=new THREE.Color(0x0f0f1a);
    const w=ct.clientWidth,h=ct.clientHeight;
    
    // 基于数据尺寸设置相机
    const s=D.shape;
    const maxDim = Math.max(s[0], s[1], s[2]);
    const cam=new THREE.PerspectiveCamera(50,w/h,.1,1000);
    cam.position.set(maxDim*1.5, maxDim*1.2, maxDim*1.5); // 确保能看到全貌
    
    const r=new THREE.WebGLRenderer({{antialias:true}});r.setSize(w,h);r.setPixelRatio(Math.min(devicePixelRatio,2));ct.appendChild(r.domElement);
    const oc=new OrbitControls(cam,r.domElement);oc.enableDamping=true;oc.dampingFactor=.08;oc.autoRotate=true;oc.autoRotateSpeed=2;
    
    // 1. 添加全尺寸包围盒 (透明线框)，强制统一视觉参照
    const boxGeo = new THREE.BoxGeometry(s[0], s[1], s[2]);
    const boxMat = new THREE.LineBasicMaterial({{ color: 0x444466, transparent: true, opacity: 0.3 }});
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo), boxMat);
    sc.add(box);

    sc.add(new THREE.AxesHelper(maxDim * 0.5));
    sc.add(new THREE.AmbientLight(0xffffff,.5));
    if(pts.length>0)sc.add(mkPC(pts,vals,cm));
    return{{sc,cam,r,oc,ct}};
}}
const V=[mkView('p0',D.raw.pts,D.raw.vals,'g'),mkView('p1',D.gt.pts,D.gt.vals,'gr'),mkView('p2',D.pred.pts,D.pred.vals,'m')];

// 视图同步逻辑
let masterIdx = 0; // 当前主控视图索引
V.forEach((v, i) => {{
    // 鼠标按下或滚动时，将当前视图设为主控
    v.ct.addEventListener('pointerdown', () => {{ masterIdx = i; }});
    v.ct.addEventListener('wheel', () => {{ masterIdx = i; }});
}});

const badge=document.getElementById('rb');
window.addEventListener('mousedown',e=>{{
    if(e.button===1){{ // 中键切换自动旋转
        e.preventDefault();
        ar=!ar;
        V.forEach(v => v.oc.autoRotate = ar);
        badge.textContent=ar?'🔄 旋转: ON':'⏸ 旋转: OFF';
        badge.className=ar?'':'off';
    }}
}});
window.addEventListener('auxclick',e=>{{if(e.button===1)e.preventDefault()}});

window.addEventListener('resize',()=>{{
    V.forEach(v=>{{
        const w=v.ct.clientWidth,h=v.ct.clientHeight;
        v.cam.aspect=w/h;
        v.cam.updateProjectionMatrix();
        v.r.setSize(w,h);
    }})
}});

(function animate(){{
    requestAnimationFrame(animate);
    
    // 1. 仅更新主控视图的控制器
    V[masterIdx].oc.update();
    
    // 2. 将主控的状态同步给其他视图
    const master = V[masterIdx];
    V.forEach((v, i) => {{
        if (i !== masterIdx) {{
            // 同步相机位置(包含缩放/旋转)
            v.cam.position.copy(master.cam.position);
            v.cam.quaternion.copy(master.cam.quaternion);
            // 同步控制器中心点(平移)
            v.oc.target.copy(master.oc.target);
        }}
        v.r.render(v.sc, v.cam);
    }});
}})();
</script>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def _generate_standalone_viewer_html(html_path, gltf_filename, epoch):
    """
    生成独立的 Three.js HTML 查看器 (完全自包含)
    - GLTF 数据 base64 内嵌到 HTML 中，无需外部文件引用
    - 使用 CDN 加载 Three.js
    - 支持鼠标旋转/缩放/平移
    - 可在 VS Code webview、本地浏览器等任何环境下直接查看
    """
    import base64

    # 读取 GLTF 文件并 base64 编码
    gltf_path = html_path.parent / gltf_filename
    with open(gltf_path, 'rb') as f:
        gltf_b64 = base64.b64encode(f.read()).decode('ascii')

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Epoch {epoch} - 3D Prediction Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #1a1a2e; overflow: hidden; font-family: 'Segoe UI', sans-serif; }}
        canvas {{ display: block; }}
        #info {{
            position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
            color: #e0e0e0; font-size: 14px; text-align: center;
            background: rgba(0,0,0,0.6); padding: 8px 20px; border-radius: 8px;
            backdrop-filter: blur(8px); pointer-events: none;
        }}
        #info h2 {{ font-size: 16px; margin-bottom: 4px; color: #ff6b6b; }}
        #legend {{
            position: absolute; bottom: 16px; left: 16px;
            color: #ccc; font-size: 12px;
            background: rgba(0,0,0,0.5); padding: 10px 14px; border-radius: 8px;
        }}
        #legend span {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
        #loading {{
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            color: #fff; font-size: 18px;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h2>Epoch {epoch} Prediction</h2>
        <span>鼠标左键: 旋转 | 右键: 平移 | 滚轮: 缩放</span>
    </div>
    <div id="legend">
        <span style="background:#ff4444;"></span>Prediction &nbsp;
        <span style="background:#44ff44; opacity:0.4;"></span>Ground Truth (Ghost)
    </div>
    <div id="loading">⏳ 加载 3D 模型中...</div>

    <script type="importmap">
    {{
        "imports": {{
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

        // 场景初始化
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        scene.fog = new THREE.FogExp2(0x1a1a2e, 0.002);

        const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 2000);
        camera.position.set(80, 60, 80);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(innerWidth, innerHeight);
        renderer.setPixelRatio(devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.2;
        document.body.appendChild(renderer.domElement);

        // 轨道控制器
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1.5;

        // 光照
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
        dirLight.position.set(50, 80, 50);
        dirLight.castShadow = true;
        scene.add(dirLight);
        const hemiLight = new THREE.HemisphereLight(0x4488ff, 0x002244, 0.4);
        scene.add(hemiLight);

        // 网格地面
        const gridHelper = new THREE.GridHelper(200, 20, 0x333355, 0x222244);
        gridHelper.position.y = -1;
        scene.add(gridHelper);

        // 坐标轴
        const axesHelper = new THREE.AxesHelper(30);
        scene.add(axesHelper);

        // 从内嵌 base64 数据加载 GLTF
        const gltfBase64 = "{gltf_b64}";
        const binaryStr = atob(gltfBase64);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {{
            bytes[i] = binaryStr.charCodeAt(i);
        }}
        const blob = new Blob([bytes], {{ type: 'model/gltf+json' }});
        const blobUrl = URL.createObjectURL(blob);

        const loader = new GLTFLoader();
        loader.load(
            blobUrl,
            (gltf) => {{
                URL.revokeObjectURL(blobUrl);
                const model = gltf.scene;

                // 自动居中和缩放
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 60 / maxDim;
                model.scale.setScalar(scale);
                model.position.sub(center.multiplyScalar(scale));

                scene.add(model);

                // 调整相机
                controls.target.set(0, 0, 0);
                camera.position.set(maxDim * scale * 0.8, maxDim * scale * 0.6, maxDim * scale * 0.8);
                controls.update();

                document.getElementById('loading').style.display = 'none';
            }},
            undefined,
            (error) => {{
                document.getElementById('loading').textContent = '❌ 加载失败: ' + error.message;
                console.error('GLTF Load Error:', error);
            }}
        );

        // 窗口自适应
        window.addEventListener('resize', () => {{
            camera.aspect = innerWidth / innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(innerWidth, innerHeight);
        }});

        // 渲染循环
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

