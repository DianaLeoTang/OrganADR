@echo off
REM OrganADR 训练脚本
setlocal enabledelayedexpansion

echo ========================================
echo OrganADR 训练启动
echo ========================================
echo.

REM 切换到项目根目录
cd /d "%~dp0"
echo 当前目录: %CD%
echo.

REM 检查并设置编译器路径
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo 未找到编译器，正在查找...
    REM 查找MSVC版本目录（BuildTools）
    set "VS_BASE=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
    if exist "!VS_BASE!" (
        for /f "delims=" %%V in ('dir /b /ad "!VS_BASE!" 2^>nul') do (
            set "CL_PATH=!VS_BASE!\%%V\bin\Hostx64\x64"
            if exist "!CL_PATH!\cl.exe" (
                set "PATH=!CL_PATH!;!PATH!"
                echo 找到并添加编译器: !CL_PATH!
                goto compiler_found
            )
        )
    )
    REM 如果BuildTools没找到，尝试Community版本
    set "VS_BASE=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
    if exist "!VS_BASE!" (
        for /f "delims=" %%V in ('dir /b /ad "!VS_BASE!" 2^>nul') do (
            set "CL_PATH=!VS_BASE!\%%V\bin\Hostx64\x64"
            if exist "!CL_PATH!\cl.exe" (
                set "PATH=!CL_PATH!;!PATH!"
                echo 找到并添加编译器: !CL_PATH!
                goto compiler_found
            )
        )
    )
    echo.
    echo 错误: 未找到C++编译器
    echo 请确认已安装 Microsoft C++ Build Tools 或 Visual Studio
    pause
    exit /b 1
)

:compiler_found
echo 编译器已就绪
echo.

REM 激活conda环境
call conda activate organadr

REM 设置CUDA_HOME为conda环境路径
set CUDA_HOME=%CONDA_PREFIX%

REM 添加PyTorch库路径到PATH（解决DLL加载问题）- 必须放在最前面
set "TORCH_LIB=C:\Users\tangw\.conda\envs\organadr\lib\site-packages\torch\lib"
set "PATH=%TORCH_LIB%;%PATH%"
echo 已添加PyTorch库路径: %TORCH_LIB%

REM 验证PATH
echo 验证DLL路径...
where cudart64_12.dll >nul 2>&1
if errorlevel 1 (
    echo 警告: CUDA DLL可能不在PATH中
) else (
    echo CUDA DLL路径正确
)
echo.

REM 切换到训练目录
cd "Part_02---Demo_of_Training_and_Evaluating_OrganADR\model"
if errorlevel 1 (
    echo 错误: 找不到训练目录
    echo 请确认在项目根目录运行此脚本
    pause
    exit /b 1
)

echo CUDA_HOME: %CUDA_HOME%
echo 训练目录: %CD%
echo 开始训练...
echo.

REM 运行训练
python train_and_evaluate_demo.py --config config/demo.json

pause
