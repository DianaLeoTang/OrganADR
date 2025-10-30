# 快速开始 - Mac M芯片版

这是一个简化的快速开始指南，帮助Mac M芯片用户快速运行OrganADR。

## 第一步：激活环境

```bash
# 激活conda环境
conda activate organADR
```

如果还没有创建环境，请运行：
```bash
conda create -n organADR python=3.9
conda activate organADR
pip install torch torchvision torchaudio
pip install torchdrug numpy pandas scipy scikit-learn tqdm
```

## 第二步：测试设备

在项目根目录运行：

```bash
python test_device.py
```

你应该看到类似的输出：
```
============================================================
OrganADR - 设备检测测试
============================================================

✓ Python版本: 3.9.x
✓ PyTorch版本: 2.x.x

【CUDA检测】
  ✗ CUDA不可用

【MPS检测 (Apple Silicon)】
  ✓ MPS可用 (Apple Silicon GPU加速)
  - 建议使用MPS进行训练
  ✓ MPS测试通过

【推荐使用的设备】
  → MPS (Apple Silicon)
  → 设备对象: mps
```

✅ 看到"MPS可用"表示Mac M芯片加速已启用！

## 第三步：运行Demo

### 方法A: 使用Mac专用脚本（推荐）

```bash
cd Part_02---Demo_of_Training_and_Evaluating_OrganADR/model/bash
bash demo_macos.bash
```

### 方法B: 直接运行Python

```bash
cd Part_02---Demo_of_Training_and_Evaluating_OrganADR/model
python train_and_evaluate_demo.py --config config/demo.json
```

## 训练开始时的预期输出

```
使用设备: Apple Silicon MPS
```

这表明正在使用Mac M芯片的GPU加速！🚀

## 查看结果

训练完成后，结果保存在：
```
Part_02---Demo_of_Training_and_Evaluating_OrganADR/results/demo/
```

## 性能预期

在Mac M1 Max上：
- **每个epoch**: 约4-6分钟
- **总训练时间** (25 epochs): 约2-2.5小时

*实际时间取决于具体的Mac型号*

## 常见问题

### Q: 为什么没有看到"MPS可用"？

**可能原因**:
1. macOS版本 < 12.3 → 升级系统
2. PyTorch版本 < 2.0 → 运行 `pip install --upgrade torch`
3. 不是Apple Silicon Mac → 会自动使用CPU

### Q: 训练速度慢？

**检查项**:
- ✅ 确认看到"使用设备: Apple Silicon MPS"
- ✅ Mac是否连接电源（性能模式）
- ✅ 关闭其他占用内存的应用

### Q: 遇到MPS错误？

**解决方案**:
1. 更新PyTorch: `pip install --upgrade torch`
2. 如果问题持续，可以暂时使用CPU（速度会慢一些，但稳定）

### Q: 如何强制使用CPU？

修改 `train_and_evaluate_demo.py` 中的 `get_device` 函数：
```python
def get_device(gpu_id=0):
    return torch.device('cpu')  # 强制使用CPU
```

## 获取帮助

详细文档请查看：
- 📖 [Mac M芯片完整指南](README_MacOS_M_Chip.md)
- 📊 [适配总结文档](Mac_M_Chip_Adaptation_Summary.md)
- 📝 [主README](README.md)

## 一切就绪！🎉

现在你可以在Mac M芯片上享受OrganADR的训练加速了！

---

*提示：首次运行时MPS需要编译某些操作，可能会慢一些。后续运行会更快。*

