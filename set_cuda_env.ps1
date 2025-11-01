# 设置 CUDA_HOME 环境变量脚本
# 在运行训练脚本前执行此脚本

Write-Host "正在查找 CUDA 安装路径..." -ForegroundColor Yellow

# 常见的 CUDA 安装路径
$possiblePaths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
    "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
)

$cudaPath = $null
foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $cudaPath = $path
        Write-Host "找到 CUDA: $path" -ForegroundColor Green
        break
    }
}

if (-not $cudaPath) {
    Write-Host "未在标准路径找到 CUDA，请手动指定 CUDA_HOME" -ForegroundColor Red
    Write-Host ""
    Write-Host "请运行以下命令设置 CUDA_HOME（替换为实际路径）：" -ForegroundColor Yellow
    Write-Host '  $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"' -ForegroundColor Cyan
    Write-Host ""
    Write-Host "或者查找 CUDA 安装位置：" -ForegroundColor Yellow
    Write-Host '  Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\"' -ForegroundColor Cyan
    exit 1
}

# 设置环境变量（当前会话）
$env:CUDA_HOME = $cudaPath
$env:PATH = "$cudaPath\bin;$env:PATH"

Write-Host ""
Write-Host "CUDA_HOME 已设置为: $env:CUDA_HOME" -ForegroundColor Green
Write-Host "PATH 已更新" -ForegroundColor Green
Write-Host ""
Write-Host "现在可以运行训练脚本了！" -ForegroundColor Green

