@echo off
REM 修复DLL问题并运行训练

setlocal enabledelayedexpansion

echo ========================================
echo 修复DLL问题并运行训练
echo ========================================
echo.

call conda activate organadr

REM 清理编译缓存
echo [1/3] 清理编译缓存...
python -c "import os, shutil; cache = os.path.expanduser('~/.cache/torch_extensions'); shutil.rmtree(cache, ignore_errors=True); print('缓存已清理')" 2>nul
echo.

REM 设置环境变量
echo [2/3] 设置环境变量...
set CUDA_HOME=%CONDA_PREFIX%
set "TORCH_LIB=C:\Users\tangw\.conda\envs\organadr\lib\site-packages\torch\lib"
set "PATH=%TORCH_LIB%;%PATH%"

set "CL_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
if exist "%CL_PATH%\cl.exe" (
    set "PATH=%CL_PATH%;%PATH%"
)

echo CUDA_HOME: %CUDA_HOME%
echo PyTorch库路径: %TORCH_LIB%
echo.

REM 预编译torchdrug扩展（确保环境正确）
echo [3/3] 预编译torchdrug扩展（可能需要几分钟）...
python -c "import os; torch_lib = r'C:\Users\tangw\.conda\envs\organadr\lib\site-packages\torch\lib'; os.add_dll_directory(torch_lib); os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', ''); from torchdrug.layers import functional; print('扩展编译成功！')"
if errorlevel 1 (
    echo.
    echo 预编译失败，但将继续尝试运行训练...
)
echo.

REM 运行训练
cd "Part_02---Demo_of_Training_and_Evaluating_OrganADR\model"
echo ========================================
echo 开始训练...
echo ========================================
echo.

python train_and_evaluate_demo.py --config config/demo.json

pause

