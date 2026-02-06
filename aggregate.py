import os

# ================= 配置区域 =================

# 1. 目录黑名单：只要包含这些名字的文件夹，其下所有内容直接跳过
IGNORE_DIRS = {
    # 核心构建与缓存 (Flutter/Dart/Mobile)
    '.dart_tool',   # [重灾区] 包含大量构建缓存
    '.gradle',      # [重灾区] Android构建缓存
    'build',        # 构建产物
    'Pods',         # iOS 依赖
    'ios',          # (可选) 如果只看Dart逻辑，建议连原生层都屏蔽，根据需要注释掉此行
    'android',      # (同上)
    'web',          # (同上)
    'linux', 'windows', 'macos', # 桌面端平台目录

    # 通用/IDE/版本控制
    '.git', '.github', '.idea', '.vscode', 
    '__pycache__', 'node_modules', '.venv', 'venv', 'env',
    '.DS_Store', 'dist', 'obsidian', 'coverage'
}

# 2. 文件黑名单：特定的无用文件
IGNORE_FILES = {
    '.DS_Store', 'thumbs.db', 
    'pubspec.lock', 'Podfile.lock', 'yarn.lock', 'package-lock.json', # 锁文件通常不仅浪费token还没用
    'gradlew', 'gradlew.bat', 'local.properties'
}

# 3. 扩展名黑名单：过滤非代码文件
IGNORE_EXTENSIONS = {
    # 图片/媒体
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp', '.mp3', '.mp4',
    # 二进制/编译产物
    '.so', '.dll', '.exe', '.class', '.o', '.a', '.apk', '.jar',
    # 日志/临时/其他
    '.log', '.cache', '.temp', '.tmp', '.zip', '.tar', '.gz'
}

# ===========================================

def should_ignore(name):
    """判断文件或目录名是否在黑名单中"""
    return name in IGNORE_DIRS or name in IGNORE_FILES

def is_text_file(filename):
    """通过扩展名简单判断是否为需要读取的文本文件"""
    # 如果没有扩展名（如LICENSE），通常默认为文本
    if '.' not in filename:
        return True
    ext = os.path.splitext(filename)[1].lower()
    return ext not in IGNORE_EXTENSIONS

def generate_tree(root_dir, current_dir, prefix=""):
    """递归生成项目目录结构树 (带过滤)"""
    tree_str = ""
    try:
        items = sorted(os.listdir(current_dir))
    except PermissionError:
        return ""

    # 过滤忽略项
    items = [i for i in items if not should_ignore(i)]
    
    for i, item in enumerate(items):
        path = os.path.join(current_dir, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        tree_str += f"{prefix}{connector}{item}\n"
        
        if os.path.isdir(path):
            extension_prefix = "    " if is_last else "│   "
            tree_str += generate_tree(root_dir, path, prefix + extension_prefix)
    return tree_str

def aggregate_project():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.basename(current_dir)
    output_filename = f"{folder_name}_code_only.md" # 改名以区分
    output_path = os.path.join(current_dir, output_filename)

    print(f"[*] 开始扫描项目: {folder_name}")
    print(f"[*] 已启用强力过滤模式，忽略 .dart_tool, .gradle 等目录")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        # 1. 写入标题和精简后的目录树
        outfile.write(f"# Project Architecture: {folder_name}\n\n")
        outfile.write("## Directory Tree (Filtered)\n```text\n")
        outfile.write(f"{folder_name}/\n")
        outfile.write(generate_tree(current_dir, current_dir))
        outfile.write("```\n\n---\n\n")

        # 2. 递归遍历文件内容
        file_count = 0
        for root, dirs, files in os.walk(current_dir):
            # 核心逻辑：原地修改 dirs 列表，阻止 os.walk 进入黑名单目录
            # 这步最关键，直接切断对 .dart_tool 等目录的扫描
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in sorted(files):
                # 过滤文件名和扩展名
                if file == os.path.basename(__file__) or file == output_filename or should_ignore(file):
                    continue
                
                if not is_text_file(file):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, current_dir)

                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        
                        # 简单的防错：如果文件太大（例如超过1MB），跳过，防止卡死
                        if len(content) > 1024 * 1024: 
                            print(f"[SKIPPED] File too large: {relative_path}")
                            continue

                        ext = os.path.splitext(file)[1][1:] or "text"
                        outfile.write(f"## File: {relative_path}\n")
                        outfile.write(f"```{ext}\n")
                        outfile.write(content)
                        outfile.write(f"\n```\n\n---\n")
                        
                        file_count += 1
                        print(f"[SUCCESS] Aggregated: {relative_path}")
                except (UnicodeDecodeError):
                    print(f"[SKIPPED] Binary/Non-utf8: {relative_path}")
                except PermissionError:
                    print(f"[SKIPPED] Permission denied: {relative_path}")

    print(f"\n[DONE] 聚合完成。共处理 {file_count} 个核心文件。")
    print(f"结果已保存至: {output_filename}")

if __name__ == "__main__":
    aggregate_project()