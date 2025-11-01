@echo off
REM OrganADR 训练脚本（简化版，直接设置路径）

setlocal enabledelayedexpansion

echo ========================================
echo OrganADR 训练启动
echo ========================================
echo.

cd /d "%~dp0"
echo 当前目录: %CD%
echo.

REM 激活conda环境
call conda activate organadr

REM 设置CUDA_HOME
set CUDA_HOME=%CONDA_PREFIX%

REM 添加PyTorch库路径（解决DLL加载问题）
set "TORCH_LIB=C:\Users\tangw\.conda\envs\organadr\lib\site-packages\torch\lib"
set "PATH=%TORCH_LIB%;%PATH%"

REM 添加编译器路径
set "CL_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
if exist "%CL_PATH%\cl.exe" (
    set "PATH=%CL_PATH%;%PATH%"
    echo 已添加编译器: %CL_PATH%
)

echo CUDA_HOME: %CUDA_HOME%
echo.

REM 切换到训练目录
cd "Part_02---Demo_of_Training_and_Evaluating_OrganADR\model"
if errorlevel 1 (
    echo 错误: 找不到训练目录
    pause
    exit /b 1
)

echo 开始训练...
echo ========================================
echo.

python train_and_evaluate_demo.py --config config/demo.json

pause

