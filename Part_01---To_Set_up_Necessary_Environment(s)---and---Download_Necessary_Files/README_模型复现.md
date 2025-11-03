# OrganADR 模型架构复现指南

## 概述

本文档提供基于架构图的OrganADR模型完整复现代码和可视化方法。

## 文件说明

- `organadr_model_visualization.py`: 完整的模型实现和可视化代码

## 模型架构

OrganADR模型包含三个主要模块：

### 1. GNN模块 (e) - 整合信息(1)和(2)

**输入**:
- 药物p和q的分子特征
- 知识图谱(KG)

**处理**:
- 通过线性层生成初始嵌入
- 在整合的KG上进行GNN传播
- 分别处理p->q和q->p两个方向

**输出**: `h_out1^(p,q)` ∈ R^(2×d1)

### 2. GCN模块 (f) - 整合信息(3)和(4)

**输入**:
- `h_out1^(p,q)`: GNN模块输出
- `M`: ADR关联矩阵 (15×15)
- `L'`: 潜在标签 (15维)

**处理**:
- 从h_out1生成潜在标签L'
- 根据L'和M生成节点嵌入
- 在15个器官节点图上进行GCN传播
- 自适应最大池化

**输出**: `h_out2^(p,q)` ∈ R^(d2)

### 3. 注意力模块 (g)

**输入**:
- `h_out1^(p,q)`: GNN模块输出
- `h_out2^(p,q)`: GCN模块输出

**处理**:
- 注意力机制整合两个模块的输出
- 线性层生成最终预测

**输出**: `L` ∈ R^15 (15种ADR的预测分数)

## 使用方法

### 1. 基本使用

```python
from organadr_model_visualization import OrganADR
import torch

# 设置参数
class Args:
    all_ent = 639
    all_rel = 15
    length = 3
    feature = 'F'  # 'F' (特征融合) 或 'M' (Morgan指纹)
    n_dim = 32
    g_dim = 16

args = Args()

# 创建模型
model = OrganADR(
    eval_rel=15,
    args=args,
    drug_features_path='../preprocessed_data/id2drugfeature_new_639.pkl',
    adj_matrix_path='../datasets/dataset3/D/15/train_mat.npy'
)

# 前向传播
head = torch.LongTensor([0, 1])  # 药物p的ID
tail = torch.LongTensor([2, 3])  # 药物q的ID
KG = ...  # 知识图谱（torchdrug格式）

scores = model(head, tail, KG, use_prior_knowledge=True)
# scores: (batch_size, 15) - 15种ADR的预测分数
```

### 2. 生成架构图

```python
from organadr_model_visualization import visualize_organadr_architecture

# 生成完整架构图
visualize_organadr_architecture('organadr_architecture.png')

# 生成数据流图
from organadr_model_visualization import visualize_data_flow
visualize_data_flow({}, 'organadr_dataflow.png')
```

### 3. 运行完整演示

```bash
cd "Part_01---To_Set_up_Necessary_Environment(s)---and---Download_Necessary_Files"
python organadr_model_visualization.py
```

这将：
1. 创建模型实例
2. 运行前向传播演示
3. 生成架构图和数据流图

## 数据文件要求

### 必需文件

1. **药物特征文件**: `id2drugfeature_new_639.pkl`
   - 位置: `Part_02---Demo_of_Training_and_Evaluating_OrganADR/preprocessed_data/`
   - 格式: pickle字典，包含每种药物的特征向量

2. **ADR关联矩阵**: `train_mat.npy`
   - 位置: `Part_02---Demo_of_Training_and_Evaluating_OrganADR/datasets/dataset3/{datasource}/{dataseed}/`
   - 格式: NumPy数组 (15×15)，器官间的ADR关联矩阵

3. **知识图谱**: 
   - 需要torchdrug格式的图结构
   - 在训练代码中通过`DataLoader`加载

### 文件结构示例

```
项目根目录/
├── Part_02---Demo_of_Training_and_Evaluating_OrganADR/
│   ├── preprocessed_data/
│   │   └── id2drugfeature_new_639.pkl  ← 药物特征
│   └── datasets/
│       └── dataset3/
│           └── D/15/
│               └── train_mat.npy  ← ADR关联矩阵
└── Part_01---To_Set_up_Necessary_Environment(s)---and---Download_Necessary_Files/
    └── organadr_model_visualization.py  ← 模型代码
```

## 模块详细说明

### GNN模块 (`enc_ht`)

```python
def enc_ht(self, head, tail, KG):
    """
    输入:
        head: 药物p的ID (batch_size,)
        tail: 药物q的ID (batch_size,)
        KG: 知识图谱
    
    输出:
        embeddings: (batch_size, 2*n_dim)
    """
    # 1. 获取分子特征嵌入
    head_embed = self.enc_head_or_tail(head)  # (batch, n_dim)
    tail_embed = self.enc_head_or_tail(tail)  # (batch, n_dim)
    
    # 2. GNN传播（双向）
    # head -> tail
    head_hid = self.gnn_propagate(head, tail, KG)
    # tail -> head
    tail_hid = self.gnn_propagate(tail, head, KG)
    
    # 3. 拼接
    return torch.cat([head_hid, tail_hid], dim=1)  # (batch, 2*n_dim)
```

### GCN模块 (`get_label_embedding`)

```python
def get_label_embedding(self, labels):
    """
    输入:
        labels: 潜在标签L' (batch_size, 15)
    
    输出:
        pooled_output: h_out2^(p,q) (batch_size, g_dim)
    """
    # 1. 根据标签选择嵌入
    embeddings = self.select_embeddings(labels)  # (batch, 15, g_dim)
    
    # 2. GCN传播
    for gcn_layer in self.gcn_layers:
        embeddings = gcn_layer.apply_gcn(self.adjacency_matrix, embeddings)
    
    # 3. 池化
    return self.pool(embeddings.transpose(1, 2)).squeeze(-1)  # (batch, g_dim)
```

### 注意力模块 (`attention`)

```python
def attention(self, embed_drug, embed_label):
    """
    输入:
        embed_drug: h_out1^(p,q) (batch, 2*n_dim)
        embed_label: h_out2^(p,q) (batch, g_dim)
    
    输出:
        weighted_H: 加权特征 (batch, 2*n_dim)
    """
    # 1. 转换label embedding维度
    LI_transformed = self.transfer_matrix(embed_label)
    
    # 2. 计算注意力分数
    attention_scores = torch.mul(embed_drug, LI_transformed)
    attention_weights = F.softmax(attention_scores, dim=1)
    
    # 3. 加权
    return attention_weights * embed_drug
```

## 与原始代码的对应关系

| 架构图模块 | 原始代码函数 | 说明 |
|-----------|------------|------|
| GNN模块 (e) | `enc_ht()` | 整合分子特征和KG |
| GCN模块 (f) | `get_label_embedding()` | 使用ADR关联矩阵 |
| 注意力模块 (g) | `attention()` | 整合两个模块输出 |
| 预测层 | `enc_r1()`, `enc_r2()` | 生成15维ADR预测 |

## 可视化输出

运行代码后会生成：

1. **organadr_architecture.png**: 
   - 完整架构图
   - 展示三个模块及其连接
   - 包含输入输出维度标注

2. **organadr_dataflow.png**:
   - 数据流图
   - 展示各模块的数据形状变化
   - 便于理解模型工作流程

## 注意事项

1. **知识图谱格式**: 
   - 需要torchdrug的Graph对象
   - 如果无法使用torchdrug，代码提供了简化版本

2. **数据路径**:
   - 确保数据文件路径正确
   - 如果文件不存在，代码会使用随机数据用于演示

3. **设备**:
   - 模型会自动使用CUDA（如果可用）
   - 确保数据也在相同设备上

4. **依赖**:
   ```bash
   pip install torch matplotlib seaborn numpy
   ```

## 扩展使用

### 加载真实数据

```python
import pickle
import numpy as np

# 加载药物特征
with open('../preprocessed_data/id2drugfeature_new_639.pkl', 'rb') as f:
    drug_features = pickle.load(f)

# 加载ADR矩阵
adj_matrix = np.load('../datasets/dataset3/D/15/train_mat.npy')

# 创建模型（会自动加载这些数据）
model = OrganADR(eval_rel=15, args=args,
                 drug_features_path='../preprocessed_data/id2drugfeature_new_639.pkl',
                 adj_matrix_path='../datasets/dataset3/D/15/train_mat.npy')
```

### 训练模式

```python
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    # ... 获取batch数据 ...
    scores = model(head, tail, KG, use_prior_knowledge=True)
    loss = criterion(scores, labels)
    loss.backward()
    optimizer.step()
```

## 问题排查

1. **ImportError**: 确保安装了所需依赖
2. **文件未找到**: 检查数据文件路径
3. **CUDA错误**: 如果没有GPU，代码会自动使用CPU
4. **维度不匹配**: 检查输入数据形状是否匹配

## 参考

- 原始训练代码: `Part_02---Demo_of_Training_and_Evaluating_OrganADR/model/train_and_evaluate_demo.py`
- 数据文件清单: `数据文件清单.md`

