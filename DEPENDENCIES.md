# OrganADR 项目依赖库文档

## 📋 目录

- [系统要求](#系统要求)
- [核心依赖](#核心依赖)
- [安装方式](#安装方式)
- [依赖说明](#依赖说明)
- [故障排除](#故障排除)

## 🔧 系统要求

- **操作系统**: Windows 10/11 (64-bit)
- **Python**: 3.9 或更高版本
- **CUDA**: 12.8 (兼容 CUDA 12.1/12.4 版本的 PyTorch)
- **GPU**: NVIDIA GPU (推荐)

## 📦 核心依赖

### 深度学习框架
- **PyTorch**: 2.4.0 (CUDA 12.1版本，兼容 CUDA 12.8)
- **torchvision**: 0.19.0+
- **torchaudio**: 2.4.0+

### PyTorch Geometric 扩展
- **torch-scatter**: 2.1.2+pt24cu121
- **torch-cluster**: 1.6.3+pt24cu121

### 深度学习库
- **torchdrug**: 0.2.1+

### 科学计算
- **NumPy**: <2.0 (1.26.4 推荐，兼容 rdkit)
- **SciPy**: >=1.9.0
- **scikit-learn**: >=1.0.0
- **pandas**: >=1.3.0

### 工具库
- **tqdm**: >=4.60.0
- **rdkit**: >=2023.0.0 (由 torchdrug 依赖)

## 🚀 安装方式

### 方式一：一键安装脚本（推荐）

#### PowerShell 脚本
```powershell
# 以管理员身份运行 PowerShell
.\install_dependencies.ps1
```

#### 批处理脚本
```cmd
# 双击运行或在命令行执行
install_dependencies.bat
```

### 方式二：手动安装

#### 1. 安装基础依赖
```bash
pip install "numpy<2.0" scikit-learn>=1.0.0 tqdm>=4.60.0 scipy>=1.9.0 pandas>=1.3.0
```

#### 2. 安装 PyTorch (CUDA 12.1)
```bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. 安装 PyTorch Geometric 扩展
```bash
# torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# torch-cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

#### 4. 安装 torchdrug
```bash
pip install torchdrug
```

### 方式三：使用 requirements.txt
```bash
# 注意：PyTorch 和 PyG 扩展需要单独安装（见上方）
pip install -r requirements.txt
```

## 📚 依赖说明

### PyTorch 版本选择
- **CUDA 12.8**: 项目系统安装的 CUDA 版本
- **PyTorch CUDA 12.1**: PyTorch 官方提供的兼容版本，向后兼容 CUDA 12.8
- 实际使用中，PyTorch 2.4.0+cu121 可以在 CUDA 12.8 环境下正常工作

### NumPy 版本限制
- **NumPy < 2.0**: 必需，因为 rdkit 和其他一些库基于 NumPy 1.x 编译
- **推荐版本**: NumPy 1.26.4
- NumPy 2.0+ 会导致兼容性错误

### torch-scatter 和 torch-cluster
- 这些包在 Windows 上通常需要从源码编译（需要 C++ 编译器）
- **推荐**: 使用预编译的 wheel 文件（脚本中已配置）
- **备选**: 如果预编译版本不可用，需要安装 Microsoft C++ Build Tools

## ⚠️ 故障排除

### 问题1: torch-scatter/torch-cluster 编译失败

**错误信息**:
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**解决方案**:
1. 安装 Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 选择 "C++ build tools" 工作负载
3. 重启终端后重新运行安装脚本

### 问题2: NumPy 版本冲突

**错误信息**:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.1
```

**解决方案**:
```bash
pip install "numpy<2.0"
```

### 问题3: CUDA 版本不匹配

**错误信息**:
```
CUDA runtime version mismatch
```

**解决方案**:
1. 确认已安装 CUDA 12.x
2. 使用匹配的 PyTorch 版本（CUDA 12.1 兼容版本）
3. 验证 CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### 问题4: torchdrug 导入错误

**错误信息**:
```
AttributeError: _ARRAY_API not found
```

**解决方案**:
这通常是由 NumPy 版本不兼容引起的，降级 NumPy:
```bash
pip install "numpy<2.0"
```

## ✅ 验证安装

运行以下命令验证所有依赖是否正确安装：

```python
import torch
import torch_scatter
import torch_cluster
from torchdrug.layers import functional
import numpy as np

print("PyTorch:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("NumPy:", np.__version__)
print("所有依赖安装成功！")
```

## 📝 版本历史

### 2025-01-XX
- PyTorch 2.4.0 + CUDA 12.1
- torch-scatter 2.1.2+pt24cu121
- torch-cluster 1.6.3+pt24cu121
- torchdrug 0.2.1
- NumPy 1.26.4

## 🔗 相关链接

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric 文档](https://pytorch-geometric.readthedocs.io/)
- [torchdrug GitHub](https://github.com/DeepGraphLearning/torchdrug)
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

