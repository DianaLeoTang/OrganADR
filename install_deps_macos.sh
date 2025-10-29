#!/bin/bash

# macOS环境安装脚本 - OrganADR
echo "===================="
echo "安装OrganADR所需依赖"
echo "===================="

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate organADR

echo ""
echo "步骤1: 安装PyTorch (适用于Apple Silicon)"
pip install torch torchvision torchaudio

echo ""
echo "步骤2: 安装torchdrug"
pip install torchdrug

echo ""
echo "步骤3: 安装科学计算包"
pip install numpy pandas scipy

echo ""
echo "步骤4: 安装scikit-learn"
pip install scikit-learn

echo ""
echo "步骤5: 安装其他工具包"
pip install tqdm

echo ""
echo "===================="
echo "安装完成！"
echo "===================="
echo "验证安装："
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
python -c "import sklearn; print(f'scikit-learn版本: {sklearn.__version__}')"
python -c "import torchdrug; print('torchdrug安装成功')"

