@echo off
REM OrganADR Environment1 安装脚本 (Windows)
REM 用于数据分析和结果可视化

echo ========================================
echo OrganADR Environment1 安装
echo ========================================
echo.
echo 环境用途: 数据分析和结果可视化
echo.

REM 检查conda是否安装
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 conda 命令
    echo 请先安装 Anaconda 或 Miniconda
    pause
    exit /b 1
)

echo [1/3] 检查conda...
conda --version
echo.

REM 设置环境名称和Python版本
set ENV_NAME=environment1
set PYTHON_VERSION=3.10

echo [2/3] 创建Conda环境: %ENV_NAME%
echo Python版本: %PYTHON_VERSION%
echo.

REM 尝试从yaml文件创建环境（推荐方式）
if exist "%~dp0environment1.yaml" (
    echo 使用 environment1.yaml 创建环境...
    conda env create -f "%~dp0environment1.yaml"
    if %errorlevel% equ 0 (
        echo 环境创建成功！
        goto activate_env
    ) else (
        echo 从yaml文件创建失败，尝试其他方式...
        echo.
    )
)

REM 如果yaml方式失败，尝试从txt文件创建
if exist "%~dp0environment1.txt" (
    echo 使用 environment1.txt 创建环境...
    conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%
    if %errorlevel% equ 0 (
        call conda activate %ENV_NAME%
        conda install -y --file "%~dp0environment1.txt"
        if %errorlevel% equ 0 (
            echo 环境创建成功！
            goto activate_env
        )
    )
)

REM 如果都失败，使用核心包手动安装
echo.
echo 使用核心包手动创建环境...
conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%
if %errorlevel% neq 0 (
    echo 错误: 无法创建conda环境
    pause
    exit /b 1
)

call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo 错误: 无法激活conda环境
    pause
    exit /b 1
)

echo.
echo [3/3] 安装核心数据分析包...
echo.

REM 安装核心包
conda install -y numpy pandas scipy matplotlib seaborn jupyter scikit-learn ipykernel
if %errorlevel% neq 0 (
    echo 警告: 部分包安装可能失败
)

pip install tqdm openpyxl
if %errorlevel% neq 0 (
    echo 警告: 部分pip包安装可能失败
)

:activate_env
echo.
echo ========================================
echo 环境安装完成！
echo ========================================
echo.
echo 激活环境命令:
echo   conda activate %ENV_NAME%
echo.
echo 查看已安装的包:
echo   conda list
echo.
echo 列出所有环境:
conda env list
echo.
echo 提示: 如果使用yaml或txt文件安装失败，可以手动安装所需的包
echo.
pause

