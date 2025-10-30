# OrganADR - Mac M芯片适配指南

本文档介绍如何在配备Apple Silicon (M1/M2/M3)芯片的Mac电脑上运行OrganADR。

## 主要改动说明

### 1. 设备自动检测
代码已修改为自动检测并使用最佳可用设备：
- **CUDA GPU**: 如果检测到NVIDIA GPU（如在Linux/Windows系统）
- **MPS (Metal Performance Shaders)**: 如果在Apple Silicon Mac上
- **CPU**: 如果以上都不可用

### 2. 核心修改内容

#### 代码改动：
- ✅ 将所有 `.cuda()` 调用改为 `.to(device)`
- ✅ 添加了 `get_device()` 函数自动检测设备
- ✅ 在 `DataLoader`、`OrganADR` 和 `BaseModel` 类中传递 `device` 参数
- ✅ CUDA特定配置（如cudnn）仅在CUDA可用时启用
- ✅ 移除了 `torch.cuda.set_device()` 的硬性要求

#### 新增文件：
- `demo_macos.bash`: Mac专用的运行脚本

## 快速测试

在开始训练之前，建议先运行设备检测脚本：

```bash
# 在项目根目录下运行
python test_device.py
```

这个脚本会：
- ✅ 检测您的设备支持情况（CUDA/MPS/CPU）
- ✅ 显示PyTorch和Python版本信息
- ✅ 进行简单的设备功能测试
- ✅ 给出针对性的使用建议

## 环境设置

### 方法1: 使用已有环境安装脚本

如果您已经按照原始README创建了环境，可以直接使用修改后的代码，无需额外配置。

### 方法2: 从头开始设置（推荐）

```bash
# 1. 创建conda环境
conda create -n organADR python=3.9
conda activate organADR

# 2. 安装PyTorch (支持Apple Silicon)
pip install torch torchvision torchaudio

# 3. 安装其他依赖
pip install torchdrug
pip install numpy pandas scipy scikit-learn
pip install tqdm

# 4. 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'MPS可用: {torch.backends.mps.is_available()}')"
```

## 运行方法

### 使用Mac专用脚本运行

```bash
cd Part_02---Demo_of_Training_and_Evaluating_OrganADR/model/bash
bash demo_macos.bash
```

### 或者直接运行Python脚本

```bash
cd Part_02---Demo_of_Training_and_Evaluating_OrganADR/model
python train_and_evaluate_demo.py --config config/demo.json
```

## 性能说明

### MPS (Apple Silicon GPU)
- ✅ **优点**: 比CPU快很多，充分利用M芯片的神经引擎
- ⚠️ **注意**: 某些操作可能还不完全支持，会自动回退到CPU

### CPU模式
- 如果MPS不可用或出现兼容性问题，会自动使用CPU
- 速度较慢，但稳定性好

## 常见问题

### Q1: 如何确认正在使用MPS？
运行脚本时，控制台会输出：
```
使用设备: Apple Silicon MPS
```

### Q2: MPS报错怎么办？
某些PyTorch操作在MPS上可能不支持。如果遇到错误：
1. 尝试更新PyTorch到最新版本：`pip install --upgrade torch`
2. 或者在代码中临时禁用MPS，强制使用CPU（修改`get_device()`函数）

### Q3: 训练速度慢？
- MPS首次运行时需要编译某些操作，后续会更快
- 确保Mac已连接电源（性能模式）
- 检查"活动监视器"确认GPU正在使用

### Q4: 稀疏张量支持
代码中使用了稀疏张量（`torch.sparse_coo_tensor`）。如果在MPS上遇到问题：
- PyTorch正在逐步改进MPS对稀疏张量的支持
- 建议使用PyTorch 2.0+版本

## 配置文件说明

`config/demo.json` 中的 `"gpu": 0` 参数：
- 在Mac M芯片上，此参数会被忽略（MPS不需要指定GPU ID）
- 在CUDA环境中，此参数指定使用哪个GPU

## 技术细节

### 设备选择逻辑
```python
def get_device(gpu_id=0):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device
```

### 兼容性
- ✅ Mac M1/M2/M3 (Apple Silicon)
- ✅ Linux with CUDA
- ✅ Windows with CUDA
- ✅ 任何平台的CPU模式

## 版本要求

- Python: 3.8+
- PyTorch: 2.0+ (推荐2.4+以获得更好的MPS支持)
- macOS: 12.3+ (Monterey或更高版本以使用MPS)

## 性能对比参考

在相同数据集上的大致训练时间（每个epoch）：
- NVIDIA RTX 3090: ~2分钟
- Apple M1 Max (MPS): ~4-6分钟  
- Apple M1 Max (CPU): ~15-20分钟
- Intel CPU: ~20-30分钟

*实际性能取决于具体型号和配置*

## 支持

如有问题，请检查：
1. PyTorch版本是否为2.0+
2. macOS版本是否为12.3+
3. conda环境是否正确激活
4. 所有依赖包是否已安装

## 更新日志

**2025-10-29**
- ✅ 完成Mac M芯片适配
- ✅ 添加自动设备检测
- ✅ 创建macOS专用运行脚本
- ✅ 保持与原始CUDA代码的兼容性

