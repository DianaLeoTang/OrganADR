# test_env1.py
import sys
print(f"Python版本: {sys.version}")
print("\n检查主要包...")

packages = {
    'numpy': 'numpy',
    'pandas': 'pandas', 
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'plotly': 'plotly',
    'networkx': 'networkx',
    'anndata': 'anndata',
    'biopython': 'Bio',
    'rdkit': 'rdkit',
}

success = 0
failed = []

for name, import_name in packages.items():
    try:
        exec(f"import {import_name}")
        print(f"✓ {name}")
        success += 1
    except ImportError:
        print(f"✗ {name} - 未安装")
        failed.append(name)

print(f"\n成功: {success}/{len(packages)}")
if failed:
    print(f"失败: {', '.join(failed)}")
else:
    print("🎉 所有重要包都已正确安装！")