# 🚀 OrganADR 快速安装指南

## 一键安装（推荐）

### Windows PowerShell
```powershell
# 右键以管理员身份运行 PowerShell，然后执行：
.\install_dependencies.ps1
```

### Windows 命令提示符
```cmd
# 双击运行或在命令行执行：
install_dependencies.bat
```

## 手动安装（3步）

### 1. 基础依赖
```bash
pip install "numpy<2.0" scikit-learn tqdm scipy pandas
```

### 2. PyTorch (CUDA 12.1)
```bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. PyG扩展 + torchdrug
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torchdrug
```

## 验证安装

```python
python -c "import torch; import torch_scatter; from torchdrug.layers import functional; print('✅ 所有依赖安装成功！')"
```

## 系统要求

- Python 3.9+
- CUDA 12.8 (或 12.1/12.4)
- Windows 10/11

## 遇到问题？

查看 [DEPENDENCIES.md](./DEPENDENCIES.md) 获取详细故障排除指南。

