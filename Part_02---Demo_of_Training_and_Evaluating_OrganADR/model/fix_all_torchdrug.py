import os
import glob

extension_dir = r"C:\Users\tangw\.conda\envs\organadr\lib\site-packages\torchdrug\layers\functional\extension"

# 查找所有需要修补的文件
files = glob.glob(os.path.join(extension_dir, "*.h")) + glob.glob(os.path.join(extension_dir, "*.cpp"))

print(f"正在检查 {len(files)} 个文件...\n")

for file_path in files:
    filename = os.path.basename(file_path)
    modified = False
    
    try:
        # 备份
        backup_path = file_path + ".original"
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy(file_path, backup_path)
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # 替换所有 SparseTensorUtils.h 引用
        if 'SparseTensorUtils.h' in content:
            print(f"修补: {filename}")
            
            # 注释掉旧的 include
            content = content.replace(
                '#include <ATen/SparseTensorUtils.h>',
                '// #include <ATen/SparseTensorUtils.h>  // Not available in PyTorch 2.2+'
            )
            content = content.replace(
                '#include "ATen/SparseTensorUtils.h"',
                '// #include "ATen/SparseTensorUtils.h"  // Not available in PyTorch 2.2+'
            )
            
            # 确保包含了替代的头文件
            if 'SparseCsrTensorUtils.h' not in content:
                # 在第一个 #include 后添加
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include') and 'pragma' not in line.lower():
                        lines.insert(i + 1, '#include "ATen/SparseCsrTensorUtils.h"')
                        break
                content = '\n'.join(lines)
            
            modified = True
        
        if modified and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ 已修补\n")
        elif 'SparseTensor' in content:
            print(f"检查: {filename} - 已包含 Sparse 相关代码但无需修补\n")
            
    except Exception as e:
        print(f"  ✗ 错误: {e}\n")

print("\n修补完成！")
print("\n请运行以下命令清除缓存并重试：")
print("rmdir /s /q \"%LOCALAPPDATA%\\torch_extensions\"")
print("python train_and_evaluate_demo.py --config .\\config\\demo.json")