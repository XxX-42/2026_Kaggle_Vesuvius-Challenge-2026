import os
from pathlib import Path
from PIL import Image
import sys

# Update to absolute path based on my exploration
BASE_PATH = Path(r"d:\Documents\Codes\2026_Kaggle_Vesuvius Challenge 2026\data\native\train\1")

def audit_war_chest():
    print("📋 正在进行数据完整性终审...")
    
    # 1. 检查关键标签
    mask_path = BASE_PATH / "mask.png"
    ink_path = BASE_PATH / "inklabels.png"
    
    for label in [mask_path, ink_path]:
        if not label.exists():
            print(f"❌ 致命缺失: {label.name} 未找到！")
        else:
            try:
                with Image.open(label) as img:
                    print(f"✅ {label.name} 已就绪 | 尺寸: {img.size}")
            except Exception as e:
                print(f"❌ 读取错误: {label.name} - {e}")

    # 2. 检查切片连续性
    tif_dir = BASE_PATH / "surface_volume"
    if not tif_dir.exists():
        print(f"❌ 目录不存在: {tif_dir}")
        return

    tifs = sorted([f for f in tif_dir.glob("*.tif") if f.name.replace('.tif','').isdigit()], 
                  key=lambda x: int(x.name.split('.')[0]))
    
    if len(tifs) < 10:
        print(f"❌ 弹药不足: 仅找到 {len(tifs)} 张切片，无法构建 16 层深度的 2.5D 训练块。")
    else:
        print(f"📊 已就绪切片: {len(tifs)} 张 (范围: {tifs[0].name} 到 {tifs[-1].name})")

    # 3. 检查是否有损坏 (Check a few random files)
    import random
    check_files = [tifs[0], tifs[len(tifs)//2], tifs[-1]]
    
    for f in check_files:
        try:
            with Image.open(f) as img:
                pass
            print(f"✅ 样本切片解析成功: {f.name}")
        except Exception as e:
            print(f"❌ 数据损坏: 无法解析 TIF 文件 {f.name} - {e}")

if __name__ == "__main__":
    audit_war_chest()
