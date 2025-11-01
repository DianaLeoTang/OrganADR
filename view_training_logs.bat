@echo off
REM 查看训练日志脚本

echo ========================================
echo 查看训练日志
echo ========================================
echo.

cd /d "%~dp0"

REM 检查Python进程
echo [1] 检查训练进程...
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
if %errorlevel% equ 0 (
    echo 发现正在运行的Python进程
    tasklist /FI "IMAGENAME eq python.exe" /FO LIST | findstr /I "PID"
) else (
    echo 未发现Python进程（训练可能未运行或已结束）
)
echo.

REM 查找最新日志
echo [2] 查找最新训练日志...
cd "Part_02---Demo_of_Training_and_Evaluating_OrganADR\results\demo"

if exist "*.txt" (
    for /f "delims=" %%i in ('dir /b /s *.txt /o-d 2^>nul') do (
        echo.
        echo 最新日志文件: %%i
        echo 最后修改时间: 
        for %%j in ("%%i") do echo   %%~tj
        echo.
        echo 日志内容 (最后20行):
        echo ========================================
        powershell -Command "Get-Content '%%i' -Tail 20"
        goto :found
    )
)

REM 如果没有txt文件，显示最新目录
for /f "delims=" %%i in ('dir /b /ad /o-d 2^>nul') do (
    echo 最新训练目录: %%i
    echo.
    dir "%%i" /b
    goto :found
)

echo 未找到训练日志
goto :end

:found
echo.
echo ========================================
echo 提示: 使用以下命令实时查看日志更新:
echo   powershell -Command "Get-Content '日志文件路径' -Wait -Tail 5"
echo ========================================

:end
pause

