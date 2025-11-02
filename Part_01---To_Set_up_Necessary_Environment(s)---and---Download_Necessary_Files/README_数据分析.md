# 数据分析指南

## 概述

环境1 (environment1) 用于分析环境2生成的训练结果。

## 训练结果位置

训练结果保存在以下目录：
```
Part_02---Demo_of_Training_and_Evaluating_OrganADR/results/demo/
```

每个训练实验会在 `results/demo/` 下创建一个目录，包含：
- `*.txt` - 训练日志文件（包含每个epoch的指标）
- `v_result_*.pkl` - 验证集预测结果（pickle格式）
- `t_result_*.pkl` - 测试集预测结果（pickle格式）
- `model_*.pth` - 模型权重文件

## 结果文件格式

### 日志文件格式 (.txt)
每行包含一个epoch的完整指标：
```
epoch:  1	 [Valid] PR-AUC: 0.524600 ROC-AUC: 0.526888 ACC: 0.526106 PREC: 0.533651 REC: 0.609371 HL: 0.473894	 [Test] PR-AUC: 0.568672 ROC-AUC: 0.564426 ACC: 0.551310 PREC: 0.562063 REC: 0.635103 HL: 0.448690
```

指标说明：
- **PR-AUC**: Precision-Recall曲线下面积
- **ROC-AUC**: ROC曲线下面积
- **ACC**: Accuracy（准确率）
- **PREC**: Precision（精确率）
- **REC**: Recall（召回率）
- **HL**: Hamming Loss（汉明损失）

### Pickle文件格式 (.pkl)
包含详细的预测结果，格式为：
```python
{
    relation_id: {
        'score': [预测分数列表],
        'preds': [预测类别列表],
        'label': [真实标签列表]
    }
}
```

## 使用方法

### 1. 激活环境1
```cmd
conda activate environment1
```

### 2. 运行分析脚本

#### 方式一：使用提供的分析脚本（推荐）
```cmd
cd "Part_01---To_Set_up_Necessary_Environment(s)---and---Download_Necessary_Files"
python analyze_results.py
```

脚本会自动：
- 扫描所有训练结果目录
- 解析日志文件
- 生成训练曲线图
- 生成实验对比图
- 保存结果到 CSV 文件

#### 方式二：使用Jupyter Notebook
```cmd
jupyter notebook
```
然后创建新的notebook进行自定义分析。

### 3. 自定义分析

可以参考 `analyze_results.py` 编写自己的分析脚本，例如：

```python
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# 读取日志文件
df = parse_log_file('path/to/log.txt')

# 绘制训练曲线
plt.plot(df['epoch'], df['test_roc_auc'])
plt.xlabel('Epoch')
plt.ylabel('Test ROC-AUC')
plt.show()

# 读取pickle结果
with open('path/to/t_result_10.pkl', 'rb') as f:
    results = pickle.load(f)
```

## 分析输出

运行 `analyze_results.py` 后，会在 `analysis_output/` 目录生成：

1. **all_results.csv** - 所有epoch的完整数据
2. **best_results.csv** - 每个实验的最佳结果（基于Test PR-AUC）
3. **training_curves_*.png** - 每个实验的训练曲线图
4. **experiment_comparison.png** - 所有实验的对比图

## 常用分析任务

### 1. 找到最佳模型
查看 `best_results.csv`，找出Test PR-AUC最高的实验。

### 2. 分析训练过程
查看训练曲线图，观察：
- 模型是否过拟合（验证集和测试集差距）
- 最佳epoch位置
- 指标收敛情况

### 3. 比较不同配置
使用对比图比较不同超参数配置的效果。

### 4. 详细预测分析
读取pickle文件进行更细致的分析，例如：
- 特定关系的预测准确率
- 混淆矩阵
- 错误案例分析

## 注意事项

1. **Python版本兼容性**: 如果pickle文件是在不同Python版本下生成的，可能需要兼容处理
2. **文件路径**: 确保路径正确，脚本会自动查找results目录
3. **内存使用**: 大型pickle文件可能占用较多内存

## 需要帮助？

如果分析脚本有问题，可以：
1. 检查环境1是否正确安装（运行 `test_env1.py`）
2. 确认训练结果目录存在
3. 查看脚本输出的错误信息

