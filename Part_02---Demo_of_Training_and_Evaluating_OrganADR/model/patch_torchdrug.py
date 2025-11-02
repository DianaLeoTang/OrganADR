import os

file_path = r"C:\Users\tangw\.conda\envs\organadr\lib\site-packages\torchdrug\layers\functional\extension\spmm.h"

print(f"修补 {file_path}...")

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换头文件
    if 'ATen/SparseTensorUtils.h' in content:
        # 尝试多种替换方案
        new_content = content.replace(
            '#include "ATen/SparseTensorUtils.h"',
            '// #include "ATen/SparseTensorUtils.h"\n#include "ATen/SparseCsrTensorUtils.h"'
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✓ 修补成功！")
    else:
        print("文件已修补或不需要修补")
        
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()