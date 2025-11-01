@echo off
REM OrganADR 依赖一键安装脚本 (Windows批处理)
REM 支持 CUDA 12.8 (使用 CUDA 12.1 兼容版本)

echo ========================================
echo OrganADR 依赖库一键安装脚本
echo ========================================
echo.

REM 检查Python环境
echo [1/6] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.9+
    pause
    exit /b 1
)
python --version
echo.

REM 升级pip
echo [2/6] 升级pip...
python -m pip install --upgrade pip
echo.

REM 安装基础依赖
echo [3/6] 安装基础Python依赖...
pip install "numpy<2.0" scikit-learn>=1.0.0 tqdm>=4.60.0 scipy>=1.9.0 pandas>=1.3.0
if errorlevel 1 (
    echo 错误: 基础依赖安装失败
    pause
    exit /b 1
)
echo.

REM 安装PyTorch
echo [4/6] 安装PyTorch 2.4.0 (CUDA 12.1)...
echo 注意: 这将卸载现有的PyTorch版本（如果存在）
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo 错误: PyTorch安装失败
    pause
    exit /b 1
)
echo.

REM 安装PyTorch Geometric扩展
echo [5/6] 安装PyTorch Geometric扩展...
echo   安装 torch-scatter...
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
echo   安装 torch-cluster...
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
echo   安装 torchdrug...
pip install torchdrug
if errorlevel 1 (
    echo 错误: torchdrug安装失败
    pause
    exit /b 1
)
echo.

REM 验证安装
echo [6/6] 验证安装...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
python -c "import numpy; print('NumPy:', numpy.__version__)"

echo.
echo ========================================
echo 安装完成！
echo ========================================
pause

