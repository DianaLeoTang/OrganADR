# OrganADR 训练脚本
# 自动设置环境并运行训练

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OrganADR 训练启动" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 激活conda环境
Write-Host "[1/3] 激活conda环境..." -ForegroundColor Yellow
conda activate organadr

# 查找并设置 CUDA_HOME
Write-Host ""
Write-Host "[2/3] 设置 CUDA_HOME..." -ForegroundColor Yellow

$cudaPaths = @(
    "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    "${env:ProgramFiles(x86)}\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)

$cudaPath = $null
foreach ($path in $cudaPaths) {
    if (Test-Path $path) {
        $cudaPath = $path
        Write-Host "找到 CUDA: $path" -ForegroundColor Green
        break
    }
}

if (-not $cudaPath) {
    Write-Host "未在标准路径找到CUDA Toolkit" -ForegroundColor Yellow
    Write-Host "使用PyTorch自带的CUDA库..." -ForegroundColor Yellow
    
    # 获取PyTorch的CUDA库路径
    $torchCudaPath = python -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>$null
    if ($torchCudaPath) {
        $env:CUDA_HOME = $torchCudaPath.Trim()
        Write-Host "设置 CUDA_HOME 为 PyTorch 路径: $env:CUDA_HOME" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "警告: 无法自动设置 CUDA_HOME" -ForegroundColor Red
        Write-Host "如果训练失败，请手动设置：" -ForegroundColor Yellow
        Write-Host '  $env:CUDA_HOME = "你的CUDA安装路径"' -ForegroundColor Cyan
        Write-Host ""
    }
} else {
    $env:CUDA_HOME = $cudaPath
    $env:PATH = "$cudaPath\bin;$env:PATH"
    Write-Host "CUDA_HOME: $env:CUDA_HOME" -ForegroundColor Green
}

# 切换到训练目录
Write-Host ""
Write-Host "[3/3] 运行训练..." -ForegroundColor Yellow
$trainDir = "Part_02---Demo_of_Training_and_Evaluating_OrganADR\model"
if (Test-Path $trainDir) {
    Set-Location $trainDir
    Write-Host "当前目录: $(Get-Location)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "开始训练..." -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    python train_and_evaluate_demo.py --config config/demo.json
} else {
    Write-Host "错误: 找不到训练目录: $trainDir" -ForegroundColor Red
    exit 1
}

