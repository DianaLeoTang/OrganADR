# OrganADR 依赖一键安装脚本 (PowerShell)
# 支持 CUDA 12.8 (使用 CUDA 12.1 兼容版本)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OrganADR 依赖库一键安装脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python环境
Write-Host "[1/6] 检查Python环境..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 未找到Python，请先安装Python 3.9+" -ForegroundColor Red
    exit 1
}
Write-Host "Python版本: $pythonVersion" -ForegroundColor Green

# 检查CUDA
Write-Host ""
Write-Host "[2/6] 检查CUDA..." -ForegroundColor Yellow
$cudaCheck = nvcc --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: 未找到nvcc命令，但将继续安装（如果已安装CUDA 12.x）" -ForegroundColor Yellow
} else {
    Write-Host "CUDA检测成功" -ForegroundColor Green
}

# 升级pip
Write-Host ""
Write-Host "[3/6] 升级pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "pip升级完成" -ForegroundColor Green

# 安装基础依赖
Write-Host ""
Write-Host "[4/6] 安装基础Python依赖..." -ForegroundColor Yellow
pip install "numpy<2.0" scikit-learn>=1.0.0 tqdm>=4.60.0 scipy>=1.9.0 pandas>=1.3.0
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 基础依赖安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "基础依赖安装完成" -ForegroundColor Green

# 安装PyTorch (CUDA 12.1版本，兼容CUDA 12.8)
Write-Host ""
Write-Host "[5/6] 安装PyTorch 2.4.0 (CUDA 12.1)..." -ForegroundColor Yellow
Write-Host "注意: 这将卸载现有的PyTorch版本（如果存在）" -ForegroundColor Yellow

# 卸载旧版本（如果存在）
pip uninstall torch torchvision torchaudio -y

# 安装新版本
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: PyTorch安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "PyTorch安装完成" -ForegroundColor Green

# 安装PyTorch Geometric扩展
Write-Host ""
Write-Host "[6/6] 安装PyTorch Geometric扩展..." -ForegroundColor Yellow

Write-Host "  安装 torch-scatter..." -ForegroundColor Cyan
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
if ($LASTEXITCODE -ne 0) {
    Write-Host "  警告: torch-scatter安装失败，尝试从源码编译（需要C++ Build Tools）" -ForegroundColor Yellow
    Write-Host "  如果失败，请安装Microsoft C++ Build Tools后重试" -ForegroundColor Yellow
}

Write-Host "  安装 torch-cluster..." -ForegroundColor Cyan
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
if ($LASTEXITCODE -ne 0) {
    Write-Host "  警告: torch-cluster安装失败，尝试从源码编译（需要C++ Build Tools）" -ForegroundColor Yellow
    Write-Host "  如果失败，请安装Microsoft C++ Build Tools后重试" -ForegroundColor Yellow
}

Write-Host "  安装 torchdrug..." -ForegroundColor Cyan
pip install torchdrug
if ($LASTEXITCODE -ne 0) {
    Write-Host "  错误: torchdrug安装失败" -ForegroundColor Red
    exit 1
}

Write-Host "PyTorch Geometric扩展安装完成" -ForegroundColor Green

# 验证安装
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "验证安装..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
python -c "import torch_scatter; print('torch-scatter:', torch_scatter.__version__)" 2>&1
python -c "import torch_cluster; print('torch-cluster: 已安装')" 2>&1
python -c "from torchdrug.layers import functional; print('torchdrug: 已安装')" 2>&1
python -c "import numpy; print('NumPy:', numpy.__version__)"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

