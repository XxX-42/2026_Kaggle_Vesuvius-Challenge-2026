import os
from pathlib import Path
from PIL import Image

# Modified to match our actual path
BASE_PATH = Path(r"data/native/train/1")

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
                Image.MAX_IMAGE_PIXELS = None # Handle large images
                with Image.open(label) as img:
                    print(f"✅ {label.name} 已就绪 | 尺寸: {img.size}")
            except Exception as e:
                print(f"❌ 读取失败: {label.name} - {e}")

    # 2. 检查切片连续性
    tif_dir = BASE_PATH / "surface_volume"
    if not tif_dir.exists():
        print(f"❌ 目录缺失: {tif_dir} 不存在")
        return

    tifs = sorted([f for f in tif_dir.glob("*.tif") if f.name.replace('.tif','').isdigit()], 
                  key=lambda x: int(x.name.split('.')[0]))
    
    if len(tifs) < 10:
        print(f"❌ 弹药不足: 仅找到 {len(tifs)} 张切片，无法构建 16 层深度的 2.5D 训练块。")
    else:
        # Check for gaps
        indices = [int(f.name.split('.')[0]) for f in tifs]
        missing = []
        for i in range(indices[0], indices[-1] + 1):
             if i not in indices:
                 missing.append(i)
        
        if missing:
             print(f"❌ 切片断裂: 缺失层 {missing}")
        else:
             print(f"📊 已就绪切片: {len(tifs)} 张 (范围: {tifs[0].name} 到 {tifs[-1].name})")

    # 3. 检查是否有损坏 (简单的头部检查)
    print("🔍 抽样检查 TIF 文件...")
    valid_count = 0
    for t_file in tifs:
        try:
            with Image.open(t_file) as img:
                pass
            valid_count += 1
        except Exception as e:
            print(f"❌ 数据损坏: 无法解析 {t_file.name} - {e}")
            
    print(f"✅ 完成检查: {valid_count}/{len(tifs)} 文件有效")

if __name__ == "__main__":
    audit_war_chest()
