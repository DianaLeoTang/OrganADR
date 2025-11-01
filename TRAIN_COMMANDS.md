# 🚀 训练脚本执行方法

## 方式一：CMD（命令提示符）

1. 打开 **命令提示符**（CMD）
2. 运行：
```cmd
cd C:\DianaFile\tangCode\OrganADR
run_train.bat
```

## 方式二：PowerShell

1. 打开 **PowerShell**
2. 运行：
```powershell
cd C:\DianaFile\tangCode\OrganADR
.\run_train.bat
```

或者：
```powershell
cd C:\DianaFile\tangCode\OrganADR
cmd /c run_train.bat
```

## 方式三：直接双击（最简单）

直接双击项目根目录下的 `run_train.bat` 文件

## 方式四：在当前目录执行

如果已经在项目目录下：
```cmd
run_train.bat
```

或在PowerShell中：
```powershell
.\run_train.bat
```

## 完整的手动命令（如果脚本有问题）

```cmd
cd C:\DianaFile\tangCode\OrganADR\Part_02---Demo_of_Training_and_Evaluating_OrganADR\model
conda activate organadr
set CUDA_HOME=%CONDA_PREFIX%
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;%PATH%
python train_and_evaluate_demo.py --config config/demo.json
```

## 注意事项

- 如果遇到"找不到编译器"错误，请确认已安装 Microsoft C++ Build Tools
- 如果遇到 CUDA 相关错误，脚本会自动设置 CUDA_HOME
- 训练过程可能需要较长时间，请耐心等待

