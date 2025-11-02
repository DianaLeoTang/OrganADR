# OrganADR Environment1 安装脚本 (Windows PowerShell)
# 用于数据分析和结果可视化

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OrganADR Environment1 安装" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "环境用途: 数据分析和结果可视化" -ForegroundColor Yellow
Write-Host ""

# 检查conda是否安装
$condaCheck = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCheck) {
    Write-Host "错误: 未找到 conda 命令" -ForegroundColor Red
    Write-Host "请先安装 Anaconda 或 Miniconda" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "[1/3] 检查conda..." -ForegroundColor Yellow
conda --version
Write-Host ""

# 设置环境名称和Python版本
$ENV_NAME = "environment1"
$PYTHON_VERSION = "3.10"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[2/3] 创建Conda环境: $ENV_NAME" -ForegroundColor Yellow
Write-Host "Python版本: $PYTHON_VERSION" -ForegroundColor Yellow
Write-Host ""

# 尝试从yaml文件创建环境（推荐方式）
$yamlFile = Join-Path $scriptDir "environment1.yaml"
if (Test-Path $yamlFile) {
    Write-Host "使用 environment1.yaml 创建环境..." -ForegroundColor Cyan
    conda env create -f $yamlFile
    if ($LASTEXITCODE -eq 0) {
        Write-Host "环境创建成功！" -ForegroundColor Green
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "环境安装完成！" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "激活环境命令:" -ForegroundColor Yellow
        Write-Host "  conda activate $ENV_NAME" -ForegroundColor Cyan
        Write-Host ""
        conda env list
        Write-Host ""
        pause
        exit 0
    } else {
        Write-Host "从yaml文件创建失败，尝试其他方式..." -ForegroundColor Yellow
        Write-Host ""
    }
}

# 如果yaml方式失败，尝试从txt文件创建
$txtFile = Join-Path $scriptDir "environment1.txt"
if (Test-Path $txtFile) {
    Write-Host "使用 environment1.txt 创建环境..." -ForegroundColor Cyan
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    if ($LASTEXITCODE -eq 0) {
        conda activate $ENV_NAME
        conda install -y --file $txtFile
        if ($LASTEXITCODE -eq 0) {
            Write-Host "环境创建成功！" -ForegroundColor Green
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Cyan
            Write-Host "环境安装完成！" -ForegroundColor Green
            Write-Host "========================================" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "激活环境命令:" -ForegroundColor Yellow
            Write-Host "  conda activate $ENV_NAME" -ForegroundColor Cyan
            Write-Host ""
            conda env list
            Write-Host ""
            pause
            exit 0
        }
    }
}

# 如果都失败，使用核心包手动安装
Write-Host ""
Write-Host "使用核心包手动创建环境..." -ForegroundColor Cyan
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 无法创建conda环境" -ForegroundColor Red
    pause
    exit 1
}

conda activate $ENV_NAME
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 无法激活conda环境" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "[3/3] 安装核心数据分析包..." -ForegroundColor Yellow
Write-Host ""

# 安装核心包
Write-Host "安装核心conda包..." -ForegroundColor Cyan
conda install -y numpy pandas scipy matplotlib seaborn jupyter scikit-learn ipykernel
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: 部分conda包安装可能失败" -ForegroundColor Yellow
}

Write-Host "安装pip包..." -ForegroundColor Cyan
pip install tqdm openpyxl
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: 部分pip包安装可能失败" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "环境安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "激活环境命令:" -ForegroundColor Yellow
Write-Host "  conda activate $ENV_NAME" -ForegroundColor Cyan
Write-Host ""
Write-Host "查看已安装的包:" -ForegroundColor Yellow
Write-Host "  conda list" -ForegroundColor Cyan
Write-Host ""
Write-Host "列出所有环境:" -ForegroundColor Yellow
conda env list
Write-Host ""
Write-Host "提示: 如果使用yaml或txt文件安装失败，可以手动安装所需的包" -ForegroundColor Cyan
Write-Host ""
pause

