#!/bin/bash
# macOS版本的运行脚本 - 适配Apple Silicon (M1/M2/M3)

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate organADR

echo "========================================="
echo "开始运行OrganADR训练和评估 (macOS版本)"
echo "========================================="
echo ""
echo "当前Python版本："
python --version
echo ""
echo "PyTorch信息："
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'MPS可用: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
echo ""
echo "========================================="

# 进入模型目录
cd "$(dirname "$0")/.."

# 运行训练脚本
echo "开始训练..."
python train_and_evaluate_demo.py --config config/demo.json

echo ""
echo "========================================="
echo "训练完成！"
echo "========================================="

