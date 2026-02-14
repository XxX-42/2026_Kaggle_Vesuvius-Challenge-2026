"""
Vesuvius Challenge - 数据预处理脚本

功能：将 LZW 压缩的 TIF 转换为未压缩的 NumPy 格式 (.npy)。
目的：彻底解决训练时的 IO 瓶颈，支持 Memory-Mapped (mmap) 零拷贝读取。
性能提升：预计训练 IO 速度提升 100 倍。
硬盘占用：约 25GB (786 个 volumes)。

用法:
    python 20_src/preprocess.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm
import multiprocessing

# 配置
# 原始数据目录
SRC_IMG_DIR = Path("data/vesuvius-challenge-surface-detection/train_images")
SRC_LBL_DIR = Path("data/vesuvius-challenge-surface-detection/train_labels")

# 输出目录
DST_IMG_DIR = Path("data/vesuvius-challenge-surface-detection/train_images_npy")
DST_LBL_DIR = Path("data/vesuvius-challenge-surface-detection/train_labels_npy")


def convert_file(args):
    """
    单个文件转换任务
    """
    src_path, dst_path, is_label = args
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    
    if dst_path.exists():
        return  # 跳过已存在的

    try:
        # 读取 TIF
        volume = tifffile.imread(src_path)
        
        # 确保是 3D
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        
        # 转为 uint8 (节省空间，训练时再转 float32)
        # 原始数据通常就是 uint8，这里确保类型一致
        if volume.dtype != np.uint8:
            if volume.max() <= 1.0:
                volume = (volume * 255).astype(np.uint8)
            elif volume.max() <= 255:
                volume = volume.astype(np.uint8)
            # 如果是 label 且 max > 1 (e.g. 255)，也可以保持 uint8
        
        # 保存为 .npy (未压缩)
        np.save(dst_path, volume)
        
    except Exception as e:
        print(f"\nError converting {src_path}: {e}")


def main():
    print(f"{'='*50}")
    print(f"  🚀 Vesuvius 数据预处理 (TIF -> NPY)")
    print(f"  源目录: {SRC_IMG_DIR}")
    print(f"  目标目录: {DST_IMG_DIR}")
    print(f"{'='*50}\n")
    
    # 创建输出目录
    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DST_LBL_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    
    # 扫描 Image
    img_files = sorted(list(SRC_IMG_DIR.glob("*.tif")))
    for p in img_files:
        dst = DST_IMG_DIR / (p.stem + ".npy")
        tasks.append((str(p), str(dst), False))
        
    # 扫描 Label
    lbl_files = sorted(list(SRC_LBL_DIR.glob("*.tif")))
    for p in lbl_files:
        dst = DST_LBL_DIR / (p.stem + ".npy")
        tasks.append((str(p), str(dst), True))
        
    print(f"找到 {len(img_files)} 个 image 文件, {len(lbl_files)} 个 label 文件。")
    print(f"总任务数: {len(tasks)}")
    
    # 并行处理 (根据 CPU 核数)
    # Windows 下多进程要注意 if __name__ == '__main__':
    num_workers = min(8, os.cpu_count() or 4)
    print(f"使用 {num_workers} 个进程并行转换...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(convert_file, tasks), total=len(tasks), unit="file"))
        
    print("\n✅ 转换完成！")
    print(f"输出大小检查: {DST_IMG_DIR}")


if __name__ == "__main__":
    # Windows 必须
    multiprocessing.freeze_support()
    main()
