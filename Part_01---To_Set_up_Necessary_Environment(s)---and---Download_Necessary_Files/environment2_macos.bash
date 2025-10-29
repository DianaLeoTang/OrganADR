#!/bin/bash

# macOS版本的environment2安装脚本
# 适配Apple Silicon (M1/M2/M3)

source $(conda info --base)/etc/profile.d/conda.sh

ENV_NAME="organADR"
PYTHON_VERSION="3.9"

echo "========================================="
echo "macOS版 OrganADR 环境安装"
echo "========================================="

# 环境已经创建，跳过这一步
# echo "Creating Conda environment: $ENV_NAME"
# conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo "Activating the environment: $ENV_NAME"
conda activate $ENV_NAME

echo ""
echo "Installing ipykernel"
conda install ipykernel -y
echo "Installation complete."

echo ""
echo "Installing PyTorch (macOS版本，支持Apple Silicon MPS)"
# macOS不使用CUDA，使用MPS (Metal Performance Shaders)
pip install torch torchvision torchaudio
echo "Installation complete."

echo ""
echo "Installing PyG (PyTorch Geometric) 相关包"
# macOS版本直接用pip安装
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
echo "Installation complete."

echo ""
echo "Installing torchdrug"
pip install torchdrug
echo "Installation complete."

echo ""
echo "Installing scikit-learn"
conda install scikit-learn -y
echo "Installation complete."

echo ""
echo "Installing pandas and openpyxl"
conda install pandas -y
conda install openpyxl -y
echo "Installation complete."

echo ""
echo "Installing tqdm (进度条)"
pip install tqdm
echo "Installation complete."

echo ""
echo "========================================="
echo "Environment $ENV_NAME has been successfully configured!"
echo "========================================="

echo ""
echo "验证安装："
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import torch; print(f'✓ MPS可用: {torch.backends.mps.is_available()}')" 2>/dev/null || echo "  (MPS不可用，将使用CPU)"
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import sklearn; print(f'✓ scikit-learn: {sklearn.__version__}')"
python -c "import pandas; print(f'✓ pandas: {pandas.__version__}')"
python -c "import torchdrug; print('✓ torchdrug: 安装成功')"

echo ""
conda env list

