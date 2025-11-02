Windows 环境安装指南
====================

Environment1 (数据分析环境) 安装
--------------------------------

方式一：使用批处理脚本（推荐）
1. 双击运行 environment1.bat
   
   或者在命令行执行：
   ```cmd
   environment1.bat
   ```

方式二：使用PowerShell脚本
```powershell
# 右键以管理员身份运行 PowerShell，然后执行：
.\environment1.ps1
```

方式三：手动安装

使用yaml文件（推荐）：
```cmd
conda env create -f environment1.yaml
conda activate environment1
```

或使用txt文件：
```cmd
conda create -n environment1 python=3.10
conda activate environment1
conda install --file environment1.txt
```

如果以上方式失败，可以手动安装核心包：
```cmd
conda create -n environment1 python=3.10
conda activate environment1
conda install numpy pandas scipy matplotlib seaborn jupyter scikit-learn ipykernel
pip install tqdm openpyxl
```

验证安装
--------
```cmd
conda activate environment1
python -c "import numpy, pandas, matplotlib, seaborn; print('✅ 环境1安装成功！')"
```

Environment2 (深度学习训练环境)
-------------------------------
环境2主要用于GPU训练，需要在Linux环境下运行。
如需在Windows运行，请参考项目根目录下的其他安装文档。

注意事项
--------
1. 确保已安装 Anaconda 或 Miniconda
2. 如果使用yaml或txt文件安装失败，可能是conda源或包版本问题，建议使用手动安装方式
3. 环境1主要用于数据分析，不包含深度学习相关包（PyTorch等）

