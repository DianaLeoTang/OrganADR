@echo off
REM 查找并设置C++编译器路径

echo 正在查找C++编译器 (cl.exe)...

REM 常见的Visual Studio路径
set "VS2022_BUILDTOOLS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
set "VS2022_COMMUNITY=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community"
set "VS2022_PRO=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Professional"
set "VS2022_ENTERPRISE=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Enterprise"

REM 查找cl.exe
where cl.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo 编译器已在PATH中
    where cl.exe
    goto :found
)

REM 在常见路径中查找
for %%P in (
    "%VS2022_BUILDTOOLS%\VC\Tools\MSVC"
    "%VS2022_COMMUNITY%\VC\Tools\MSVC"
    "%VS2022_PRO%\VC\Tools\MSVC"
    "%VS2022_ENTERPRISE%\VC\Tools\MSVC"
) do (
    if exist "%%P" (
        for /d %%D in ("%%P\*") do (
            if exist "%%D\bin\Hostx64\x64\cl.exe" (
                echo 找到编译器: %%D\bin\Hostx64\x64\cl.exe
                set "CL_PATH=%%D\bin\Hostx64\x64"
                set "PATH=%CL_PATH%;%PATH%"
                goto :found
            )
        )
    )
)

echo 错误: 未找到C++编译器
echo.
echo 请确认已安装 Microsoft C++ Build Tools 或 Visual Studio
echo 下载地址: https://visualstudio.microsoft.com/visual-cpp-build-tools/
pause
exit /b 1

:found
echo.
echo 编译器路径已添加到当前会话的PATH
echo 现在可以运行训练脚本了
echo.
pause

