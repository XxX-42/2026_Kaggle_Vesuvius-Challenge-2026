import os
import sys
import numpy as np
import tifffile
import pyvista as pv
from pathlib import Path
import json
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# [CONFIG] 渲染配色与参数配置 (在此处统一修改)
# ==========================================
VIS_CONFIG = {
    "background_color": "white",
    "window_size": (1800, 600),
    
    # 1. Raw CT 配置 (复刻 Debug 模式：全量显示)
    "raw": {
        "title": "Raw CT (Full Volume)",
        "show_scalar_bar": False,
        "cmap_colors": [
            (0.0, 'white'),   # 原始值 0
            (0.5, 'black'),   # 原始值 1 (典型值 128)
            (1.0, 'gray'),    # 原始值 2 (典型值 255)
        ],
        "opacity": 1.0,  # [DEBUG 风格] 全实心显示，不剔除背景和噪声
    },
    
    # 2. Ground Truth 配置 (增强显示模式)
    "gt": {
        "title": "Ground Truth (Premium Solid)",
        "color": "lime",
        "mesh_opacity": 1.0, 
        "threshold": 128,
        "smooth_shading": True, # 平滑着色
        "specular": 0.5,        # 增加镜面反射，使其具有金属/塑料质感
        "show_edges": True,     # 显示网格边缘，加强轮廓感
        "point_size": 2.0,      # 点的大小
        "edge_color": "green",  # 边缘颜色设为深绿
    },
    
    # 3. Prediction 配置 (正常外壳模式)
    "pred": {
        "title": "Prediction (Expert Surface)",
        "cmap": "Reds",
        "show_scalar_bar": False,
        # [DEBUG 模式建议]: 1.0 为全实心; 
        # 边缘提取模式建议: [0, 0.0, 95, 0.0, 96, 1.0, 192, 1.0, 255, 0.0]
        "opacity": [0, 0.0, 95, 0.0, 96, 1.0, 192, 1.0, 255, 0.0], # 使用正常模式的阶梯曲线：提取中高置信度外壳
    }
}

# 自动处理 Qt 插件路径问题
def fix_qt_plugin_path():
    try:
        import PyQt5
        qt_path = os.path.dirname(PyQt5.__file__)
        plugin_path = os.path.join(qt_path, "Qt5", "plugins", "platforms")
        if os.path.exists(plugin_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qt_path, "Qt5", "plugins")
    except Exception:
        pass

fix_qt_plugin_path()

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from PyQt5.QtCore import QTimer
    from pyvistaqt import BackgroundPlotter
    HAS_GUI_LIBS = True
except ImportError:
    HAS_GUI_LIBS = False

# 基础路径
PROJECT_ROOT = r"d:\Documents\Codes\2026_Kaggle_Vesuvius Challenge 2026 v2 - 副本"
RAW_DIR = os.path.join(PROJECT_ROOT, r"data\vesuvius-challenge-surface-detection\train_images")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "vis_history.json")

class VesuviusVisualizer:
    def __init__(self):
        if not HAS_GUI_LIBS:
            print("\n" + "!"*60)
            sys.exit(1)

        # 初始化 Plotter
        self.plotter = BackgroundPlotter(
            shape=(1, 3), 
            window_size=VIS_CONFIG["window_size"], 
            title="Vesuvius Expert Viewer (Drag-and-Drop)"
        )
        
        self.plotter.app_window.setAcceptDrops(True)
        self.plotter.app_window.dragEnterEvent = self._drag_enter_event
        self.plotter.app_window.dropEvent = self._drop_event
        self.plotter.set_background(VIS_CONFIG["background_color"])

        self.last_mask_path = None
        self._load_history()

    def _save_history(self, file_path):
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump({"last_path": file_path}, f, ensure_ascii=False, indent=4)
        except Exception: pass

    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    last_path = data.get("last_path")
                    if last_path and os.path.exists(last_path):
                        print(f"发现上次查看的文件: {last_path}")
                        QTimer.singleShot(500, lambda: self.load_and_display(last_path))
            except Exception: pass

    def _drag_enter_event(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def _drop_event(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = str(Path(urls[0].toLocalFile()))
            if file_path.endswith(".tif"): self.load_and_display(file_path)

    def get_raw_path(self, mask_path):
        filename = os.path.basename(mask_path)
        chunk_id = filename.replace("_mask.tif", "").replace(".tif", "")
        raw_path = os.path.join(RAW_DIR, f"{chunk_id}.tif")
        if not os.path.exists(raw_path):
            parent = os.path.dirname(mask_path)
            potential = os.path.join(parent, f"{chunk_id}.tif")
            if os.path.exists(potential): raw_path = potential
        return raw_path, chunk_id

    def _auto_stretch(self, vol, name="Data"):
        v_max = vol.max()
        if v_max > 0:
            if name == "Pred":
                return (vol.astype(np.float32) / v_max * 255).astype(np.uint8)
            if name == "GT" and v_max <= 1.0: 
                return (vol > 0).astype(np.uint8) * 255
            if v_max < 128: 
                return (vol.astype(np.float32) / v_max * 255).astype(np.uint8)
        return vol.astype(np.uint8)

    def load_and_display(self, mask_path):
        raw_path, chunk_id = self.get_raw_path(mask_path)
        print(f"\n正在加载: {mask_path}")

        try:
            data = tifffile.imread(mask_path)
            if len(data.shape) == 3:
                D, H, W_total = data.shape
                if W_total % 3 == 0:
                    W = W_total // 3
                    vol_raw, vol_gt, vol_pred = data[:, :, :W], data[:, :, W:2*W], data[:, :, 2*W:]
                else:
                    vol_raw = tifffile.imread(raw_path) if os.path.exists(raw_path) else np.zeros_like(data)
                    vol_gt, vol_pred = np.zeros_like(data), data
            else: return

            # 并行拉伸
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                vol_raw = executor.submit(self._auto_stretch, vol_raw, "Raw").result()
                vol_gt = executor.submit(self._auto_stretch, vol_gt, "GT").result()
                vol_pred = executor.submit(self._auto_stretch, vol_pred, "Pred").result()

            # 重启 Plotter (保持窗口不跳动)
            self.plotter.clear()
            self.plotter.app_window.setWindowTitle(f"Vesuvius 3D Expert Viewer - {chunk_id}")
            
            # 1. Raw CT
            self.plotter.subplot(0, 0)
            self.plotter.add_text(VIS_CONFIG["raw"]["title"], font_size=10, color="black")
            raw_cmap = LinearSegmentedColormap.from_list('v_raw', VIS_CONFIG["raw"]["cmap_colors"])
            self.plotter.add_volume(pv.wrap(vol_raw), cmap=raw_cmap, opacity=VIS_CONFIG["raw"]["opacity"], show_scalar_bar=VIS_CONFIG["raw"]["show_scalar_bar"])
            self.plotter.add_bounding_box(color="black")

            # 2. GT
            self.plotter.subplot(0, 1)
            self.plotter.add_text(VIS_CONFIG["gt"]["title"], font_size=10, color="black")
            try:
                mesh = pv.wrap(vol_gt).threshold(VIS_CONFIG["gt"]["threshold"])
                if mesh.n_points > 0:
                    self.plotter.add_mesh(
                        mesh, 
                        color=VIS_CONFIG["gt"]["color"], 
                        opacity=VIS_CONFIG["gt"]["mesh_opacity"],
                        smooth_shading=VIS_CONFIG["gt"]["smooth_shading"],
                        specular=VIS_CONFIG["gt"]["specular"],
                        show_edges=VIS_CONFIG["gt"]["show_edges"],
                        edge_color=VIS_CONFIG["gt"]["edge_color"],
                        line_width=1,
                        lighting=True  # 显式开启光照
                    )
            except Exception: pass
            self.plotter.add_bounding_box(color="black")

            # 3. Prediction
            self.plotter.subplot(0, 2)
            self.plotter.add_text(VIS_CONFIG["pred"]["title"], font_size=10, color="black")
            self.plotter.add_volume(pv.wrap(vol_pred), cmap=VIS_CONFIG["pred"]["cmap"], opacity=VIS_CONFIG["pred"]["opacity"], show_scalar_bar=VIS_CONFIG["pred"]["show_scalar_bar"])
            self.plotter.add_bounding_box(color="black")

            self.plotter.link_views()
            self.plotter.camera_position = 'iso'
            self.plotter.reset_camera()
            
            # [NEW] 锁定交互：不仅是 Z 轴向上，更通过禁用自由旋转来模拟转盘
            # PyVista/VTK 的 "Trackball" 模式配合 Z 轴锁定，可以实现类似绕 Z 轴旋转的效果
            # 注意：完全禁止 XY 轴翻转比较复杂，这里我们强制 Z 轴始终向上
            for i in range(3):
                self.plotter.subplot(0, i)
                # 强制 Z 轴向上，禁止视角“翻跟头”
                self.plotter.camera.SetViewUp(0, 0, 1) 
                
            self._save_history(mask_path)
            print(">>> 加载完成。")
        except Exception as e: print(f"加载失败: {e}")

def main():
    app = QApplication(sys.argv)
    viz = VesuviusVisualizer()
    sys.exit(app.exec_())

if __name__ == "__main__": main()
