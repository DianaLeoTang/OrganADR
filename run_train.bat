@echo off
REM OrganADR 训练脚本

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
    REM 自动添加编译器路径（基于你的Build Tools安装位置）
    set "CL_PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
    if exist "%CL_PATH%\cl.exe" (
        set "PATH=%CL_PATH%;%PATH%"
        echo 已添加编译器路径: %CL_PATH%
    ) else (
        REM 如果默认路径不存在，尝试查找
        call setup_compiler.bat
        if errorlevel 1 (
            echo.
            echo 错误: 需要C++编译器才能运行训练
            echo 请确认已安装 Microsoft C++ Build Tools
            pause
            exit /b 1
        )
    )
) else (
    echo 编译器已可用
)
echo.

REM 激活conda环境
call conda activate organadr

REM 设置CUDA_HOME为conda环境路径
set CUDA_HOME=%CONDA_PREFIX%

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

