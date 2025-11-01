# 🚀 启动训练指南

## 方式一：使用批处理脚本（推荐）

1. **双击运行** `run_train.bat`
   
   或者在命令行执行：
   ```cmd
   run_train.bat
   ```

## 方式二：PowerShell 命令

**重要：如果刚安装了 C++ Build Tools，请先重启 PowerShell！**

### 重启 PowerShell 方法：
1. 关闭当前的 PowerShell 窗口
2. 重新打开 PowerShell
3. 或者输入 `exit` 然后重新打开

### 然后运行：

```powershell
cd C:\DianaFile\tangCode\OrganADR\Part_02---Demo_of_Training_and_Evaluating_OrganADR\model
conda activate organadr
$env:CUDA_HOME = $env:CONDA_PREFIX
python train_and_evaluate_demo.py --config config/demo.json
```

## 方式三：如果编译器仍有问题

如果重启后还是找不到 `cl.exe`，可以手动查找并添加到 PATH：

```powershell
# 查找编译器
Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022" -Recurse -Filter "cl.exe" -ErrorAction SilentlyContinue | Select-Object -First 1

# 找到后，添加到PATH（替换为实际路径）
$env:PATH = "编译器目录;$env:PATH"
```

## 验证环境

运行前可以验证：
```powershell
conda activate organadr
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
where cl  # 检查编译器
```

