# 检查C++编译器是否可用

Write-Host "检查C++编译器..." -ForegroundColor Yellow
Write-Host ""

$clPath = where.exe cl 2>$null
if ($clPath) {
    Write-Host "找到编译器: $clPath" -ForegroundColor Green
    
    # 检查编译器版本
    $version = & cl 2>&1 | Select-String "Version"
    if ($version) {
        Write-Host $version -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "编译器已就绪，可以运行训练脚本！" -ForegroundColor Green
} else {
    Write-Host "未找到编译器 (cl.exe)" -ForegroundColor Red
    Write-Host ""
    Write-Host "如果已安装 C++ Build Tools，可能需要：" -ForegroundColor Yellow
    Write-Host "1. 重启PowerShell/终端" -ForegroundColor Cyan
    Write-Host "2. 或者运行 Visual Studio Developer Command Prompt" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "查找编译器路径：" -ForegroundColor Yellow
    $vsPaths = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
    )
    
    foreach ($path in $vsPaths) {
        if (Test-Path $path) {
            $clExe = Get-ChildItem $path -Recurse -Filter "cl.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($clExe) {
                Write-Host "找到编译器: $($clExe.FullName)" -ForegroundColor Green
                Write-Host "添加到PATH: `$env:PATH = `"$($clExe.DirectoryName);`$env:PATH`"" -ForegroundColor Cyan
            }
        }
    }
}


