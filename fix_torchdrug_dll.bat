@echo off
REM 修复torchdrug DLL加载问题

echo ========================================
echo 修复torchdrug DLL加载问题
echo ========================================
echo.

call conda activate organadr

REM 清理torchdrug扩展缓存
echo [1/3] 清理torchdrug扩展缓存...
python -c "import torchdrug; import os; import shutil; cache_dirs = [os.path.expanduser('~/.cache'), os.path.join(os.path.dirname(torchdrug.__file__), '..')]; [shutil.rmtree(d, ignore_errors=True) for d in cache_dirs if os.path.exists(d)]" 2>nul
echo 缓存已清理
echo.

REM 设置环境变量
echo [2/3] 设置环境变量...
set CUDA_HOME=%CONDA_PREFIX%
set "TORCH_LIB=C:\Users\tangw\.conda\envs\organadr\lib\site-packages\torch\lib"
set "PATH=%TORCH_LIB%;%PATH%"
echo 环境变量已设置
echo.

REM 预编译torchdrug扩展（强制重新编译）
echo [3/3] 预编译torchdrug扩展...
python -c "import os; os.environ['PATH'] = r'C:\Users\tangw\.conda\envs\organadr\lib\site-packages\torch\lib;' + os.environ.get('PATH', ''); from torchdrug.layers import functional; print('torchdrug扩展编译成功！')" 2>&1
if errorlevel 1 (
    echo.
    echo 扩展编译可能需要一些时间，请等待...
    echo 如果长时间无响应，请按Ctrl+C中断后重新运行训练脚本
) else (
    echo.
    echo ========================================
    echo 修复完成！
    echo 现在可以运行训练脚本了
    echo ========================================
)

pause

