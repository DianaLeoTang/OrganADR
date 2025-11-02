import os

files_to_patch = [
    r"C:\Users\tangw\.conda\envs\organadr39\lib\site-packages\torch\include\c10\util\safe_numerics.h",
    r"C:\Users\tangw\.conda\envs\organadr39\lib\site-packages\torch\include\c10\util\BFloat16-math.h",
    r"C:\Users\tangw\.conda\envs\organadr39\lib\site-packages\torch\include\c10\util\Half-inl.h",
]

patch = """#ifdef _MSC_VER
#include <intrin.h>
#ifndef __builtin_clzll
#define __builtin_clzll(x) _lzcnt_u64(x)
#endif
#endif

"""

for file_path in files_to_patch:
    if not os.path.exists(file_path):
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '<intrin.h>' in content:
        print(f"Already patched: {file_path}")
        continue
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(patch + content)
    
    print(f"✓ Patched: {file_path}")

print("\nDone! Clear cache and retry.")