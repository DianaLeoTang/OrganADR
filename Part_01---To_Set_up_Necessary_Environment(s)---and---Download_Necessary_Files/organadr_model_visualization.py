"""
OrganADR 模型架构可视化与复现代码
基于项目实际代码结构和架构图

此脚本包含：
1. 完整的OrganADR模型实现
2. 架构可视化代码
3. 数据加载示例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import seaborn as sns
from pathlib import Path
import pickle

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")

# ============================================================================
# 1. 模型实现
# ============================================================================

class GCNLayer(nn.Module):
    """GCN层 - 用于器官图上的信息传播"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def apply_gcn(self, adjacency_matrix, features):
        """应用GCN操作: H' = ReLU(A @ H @ W)"""
        output = torch.matmul(adjacency_matrix, torch.matmul(features, self.weight))
        return F.relu(output)


class OrganADR(nn.Module):
    """
    OrganADR模型 - 预测药物不良反应
    
    包含三个主要模块：
    1. GNN模块 (enc_ht): 整合分子特征和知识图谱信息
    2. GCN模块 (get_label_embedding): 整合器官层面ADR关联信息
    3. 注意力模块 (attention): 最终整合和预测
    """
    
    def __init__(self, eval_rel, args, drug_features_path=None, adj_matrix_path=None):
        super(OrganADR, self).__init__()
        self.eval_rel = eval_rel  # 评估的关系数量（15个ADR）
        self.all_ent = args.all_ent
        self.all_rel = args.all_rel
        self.L = args.length  # GNN层数
        self.feature = args.feature  # 'F' (特征融合) 或 'M' (Morgan)
        self.n_dim = args.n_dim  # GNN输出维度
        self.g_dim = args.g_dim  # GCN输出维度
        
        all_rel = 2 * args.all_rel - args.eval_rel + 1

        # ========== GCN Module ==========
        # 用于器官图的嵌入层（15个器官）
        self.embedding_layer_in = nn.Embedding(num_embeddings=15, embedding_dim=self.g_dim)
        self.embedding_layer_ab = nn.Embedding(num_embeddings=15, embedding_dim=self.g_dim)
        self.gcn_layers = nn.ModuleList([GCNLayer(self.g_dim, self.g_dim) for _ in range(self.L)])
        self.pool = nn.AdaptiveMaxPool1d(1)

        # ========== 加载药物特征 ==========
        if drug_features_path and Path(drug_features_path).exists():
            with open(drug_features_path, 'rb') as f:
                x = pickle.load(f)
                feature_01 = []  # RDKit Descriptor (210维)
                feature_02 = []  # RDKit Fingerprint (136维)
                feature_03 = []  # MACCS (166维)
                feature_04 = []  # Morgan_512 (512维)
                feature_M = []  # Morgan_1024_raw (1024维)
                
                for k in sorted(x.keys()):  # 确保顺序一致
                    feature_01.append(x[k]['RDKit Descriptor'])
                    feature_02.append(x[k]['RDKit Fingerprint'])
                    feature_03.append(x[k]['MACCS'])
                    feature_04.append(x[k]['Morgan_512'])
                    feature_M.append(x[k]['Morgan_1024_raw'])
        else:
            # 如果没有数据文件，创建随机特征用于演示
            print("警告: 未找到药物特征文件，使用随机特征")
            num_drugs = 639
            feature_01 = np.random.rand(num_drugs, 210).astype(np.float32)
            feature_02 = np.random.rand(num_drugs, 136).astype(np.float32)
            feature_03 = np.random.rand(num_drugs, 166).astype(np.float32)
            feature_04 = np.random.rand(num_drugs, 512).astype(np.float32)
            feature_M = np.random.rand(num_drugs, 1024).astype(np.float32)

        feature_01 = np.array(feature_01)
        feature_02 = np.array(feature_02)
        feature_03 = np.array(feature_03)
        feature_04 = np.array(feature_04)
        feature_M = np.array(feature_M)

        # ========== GNN Module ==========
        if self.feature == 'F':
            # 特征融合模式
            self.feature_01 = nn.Parameter(torch.FloatTensor(feature_01), requires_grad=False)
            self.feature_02 = nn.Parameter(torch.FloatTensor(feature_02), requires_grad=False)
            self.feature_03 = nn.Parameter(torch.FloatTensor(feature_03), requires_grad=False)
            self.feature_04 = nn.Parameter(torch.FloatTensor(feature_04), requires_grad=False)
            self.attention_189 = nn.Linear(210, 210)  # 对特征01的注意力
            self.attention_156 = nn.Linear(166, 166)  # 对特征03的注意力
            self.Went = nn.Linear(210+136+166+512, args.n_dim)  # 融合特征到n_dim
        elif self.feature == 'M':
            # Morgan指纹模式
            self.ent_kg = nn.Parameter(torch.FloatTensor(feature_M), requires_grad=False)
            self.Went = nn.Linear(1024, args.n_dim)

        # 关系嵌入和GNN层
        self.rel_kg = nn.ModuleList([nn.Embedding(all_rel, args.n_dim) for i in range(self.L)])
        self.linear = nn.ModuleList([nn.Linear(args.n_dim, args.n_dim) for i in range(self.L)])
        self.act = nn.ReLU()
        self.relation_linear = nn.ModuleList([nn.Linear(2*args.n_dim, 5) for i in range(self.L)])
        self.attn_relation = nn.ModuleList([nn.Linear(5, all_rel) for i in range(self.L)])

        # ADR关联矩阵（15x15的器官关联图）
        if adj_matrix_path and Path(adj_matrix_path).exists():
            self.adjacency_matrix = torch.from_numpy(
                np.load(adj_matrix_path).astype(np.float32)
            )
        else:
            # 如果没有，创建随机对称矩阵
            print("警告: 未找到ADR关联矩阵，使用随机矩阵")
            adj = np.random.rand(15, 15)
            adj = (adj + adj.T) / 2  # 对称化
            adj = (adj > 0.5).astype(np.float32)  # 二值化
            np.fill_diagonal(adj, 1)  # 对角线为1
            self.adjacency_matrix = torch.from_numpy(adj)
        
        # 预测层
        self.Wr1 = nn.Linear(2 * args.n_dim, eval_rel)  # 第一层预测（仅GNN输出）
        self.Wr2 = nn.Linear(4 * args.n_dim + 16, eval_rel)  # 第二层预测（GNN+GCN+Attention）
        self.transfer_matrix = nn.Linear(16, 64)  # 注意力转移矩阵

    def enc_head_or_tail(self, head_or_tail):
        """编码药物特征 - 对应图中的分子特征p和q的处理"""
        # 使用注意力机制融合多种特征
        weight_210 = self.attention_189(self.feature_01[head_or_tail])
        weight_210 = F.softmax(weight_210, dim=-1)
        embed_210 = self.feature_01[head_or_tail] * weight_210

        weight_166 = self.attention_156(self.feature_03[head_or_tail])
        weight_166 = F.softmax(weight_166, dim=-1)
        embed_166 = self.feature_03[head_or_tail] * weight_166

        embed_136 = self.feature_02[head_or_tail]
        embed_512 = self.feature_04[head_or_tail]

        # 拼接所有特征
        combined_embed = torch.cat((embed_210, embed_166, embed_136, embed_512), dim=1)
        return self.Went(combined_embed)

    def enc_ht(self, head, tail, KG):
        """
        GNN模块 - 整合分子特征和知识图谱信息
        对应架构图的模块(e): GNN on integrated KG
        
        输入:
            head: 药物p的ID列表
            tail: 药物q的ID列表
            KG: 知识图谱（torchdrug格式）
        
        输出:
            embeddings: [head_embedding, tail_embedding] 拼接，维度 (batch, 2*n_dim)
        """
        # 获取药物初始嵌入（分子特征）
        if self.feature == 'F':
            head_embed = self.enc_head_or_tail(head)
            tail_embed = self.enc_head_or_tail(tail)
        elif self.feature == 'M':
            head_embed = self.Went(self.ent_kg[head])
            tail_embed = self.Went(self.ent_kg[tail])

        n_ent = self.all_ent
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)

        # 从head到tail的GNN传播
        hiddens = torch.zeros((n_ent, len(head), self.n_dim), device=head_embed.device)
        hiddens[head, torch.arange(len(head), device=head.device)] = head_embed
        
        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)
            # 关系注意力权重
            relation_weight = self.attn_relation[l](F.relu(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight).unsqueeze(2)
            rel_embed = self.rel_kg[l].weight
            relation_input = relation_weight * rel_embed
            relation_input = relation_input.view(head_embed.size(0), -1, self.n_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)
            
            # 使用torchdrug的generalized_rspmm进行KG上的GNN传播
            try:
                from torchdrug.layers import functional as tdF
                hiddens = tdF.generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
            except:
                # 如果没有torchdrug，使用简化的矩阵乘法
                # 这是一个简化实现，实际应该使用torchdrug的GNN
                hiddens = torch.matmul(KG.to_dense() if hasattr(KG, 'to_dense') else KG, 
                                     hiddens) + relation_input
            
            hiddens = hiddens.view(n_ent * len(head), -1)
            hiddens = self.linear[l](hiddens)
            hiddens = self.act(hiddens)
        
        tail_hid = hiddens.view(n_ent, len(tail), -1)[tail, torch.arange(len(tail), device=tail.device)]

        # 从tail到head的GNN传播（对称处理）
        hiddens = torch.zeros((n_ent, len(head), self.n_dim), device=head_embed.device)
        hiddens[tail, torch.arange(len(tail), device=tail.device)] = tail_embed
        
        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)
            relation_weight = self.attn_relation[l](F.relu(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight).unsqueeze(2)
            rel_embed = self.rel_kg[l].weight
            relation_input = relation_weight * rel_embed
            relation_input = relation_input.view(head_embed.size(0), -1, self.n_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)
            
            try:
                from torchdrug.layers import functional as tdF
                hiddens = tdF.generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
            except:
                hiddens = torch.matmul(KG.to_dense() if hasattr(KG, 'to_dense') else KG, 
                                     hiddens) + relation_input
            
            hiddens = hiddens.view(n_ent * len(head), -1)
            hiddens = self.linear[l](hiddens)
            hiddens = self.act(hiddens)
        
        head_hid = hiddens.view(n_ent, len(head), -1)[head, torch.arange(len(head), device=head.device)]

        # 拼接head和tail的嵌入 -> h_out1^(p,q) 维度 (batch, 2*n_dim)
        embeddings = torch.cat([head_hid, tail_hid], dim=1)
        return embeddings

    def get_label_embedding(self, labels):
        """
        GCN模块 - 整合器官层面ADR关联信息
        对应架构图的模块(f): GCN on organ graph
        
        输入:
            labels: 潜在标签 L'，维度 (batch, 15)，每个位置对应一个器官的ADR
        
        输出:
            pooled_output: h_out2^(p,q)，维度 (batch, g_dim)
        """
        batch_size = labels.shape[0]
        device = labels.device
        
        # 生成15个器官的嵌入
        indices = torch.arange(0, 15, device=device).unsqueeze(0).repeat(batch_size, 1)
        selected_embeddings_in = self.embedding_layer_in(indices)  # E_L
        selected_embeddings_ab = self.embedding_layer_ab(indices)  # E_M
        
        # 根据标签选择嵌入（如果标签为1用in，否则用ab）
        embeddings = torch.where(labels.unsqueeze(-1) == 1, 
                                selected_embeddings_in, 
                                selected_embeddings_ab)
        
        # GCN传播
        gcn_output = embeddings
        adj_matrix = self.adjacency_matrix.to(device)
        
        for gcn in self.gcn_layers:
            gcn_output = gcn.apply_gcn(adj_matrix, gcn_output)
        
        # 自适应最大池化 -> h_out2^(p,q)
        pooled_output = self.pool(gcn_output.transpose(1, 2)).squeeze(-1)
        
        return pooled_output

    def attention(self, embed_drug, embed_label):
        """
        注意力模块 - 整合GNN和GCN的输出
        对应架构图的模块(g): Attention mechanism
        
        输入:
            embed_drug: h_out1^(p,q)，来自GNN模块，维度 (batch, 2*n_dim)
            embed_label: h_out2^(p,q)，来自GCN模块，维度 (batch, g_dim)
        
        输出:
            weighted_H: 加权后的特征，维度 (batch, 2*n_dim)
        """
        # 将label embedding转换维度
        LI_transformed = self.transfer_matrix(embed_label.unsqueeze(1))  # (batch, 1, 64)
        # 这里需要调整维度匹配
        if embed_drug.size(1) != LI_transformed.size(-1):
            # 如果维度不匹配，使用线性层调整
            attention_scores = torch.matmul(embed_drug.unsqueeze(1), 
                                          LI_transformed.transpose(1, 2))
            attention_scores = attention_scores.squeeze(1)
        else:
            attention_scores = torch.mul(embed_drug, LI_transformed.squeeze(1))
        
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_H = attention_weights * embed_drug
        
        return weighted_H

    def forward(self, head, tail, KG, labels=None, use_prior_knowledge=False):
        """
        完整的前向传播
        
        输入:
            head: 药物p的ID
            tail: 药物q的ID
            KG: 知识图谱
            labels: 可选的潜在标签（如果None，会从第一次预测生成）
            use_prior_knowledge: 是否使用先验知识（GCN模块）
        
        输出:
            scores: 15维ADR预测分数
        """
        # ========== GNN模块 ==========
        # h_out1^(p,q) = GNN(p, q, KG)
        h_out1 = self.enc_ht(head, tail, KG)  # (batch, 2*n_dim)
        
        # 第一次预测（仅使用GNN输出）
        scores1 = torch.sigmoid(self.Wr1(h_out1))  # (batch, 15)
        
        if use_prior_knowledge and labels is None:
            # 从第一次预测生成潜在标签
            labels = (scores1 > 0.5).int()  # L'
        
        if use_prior_knowledge and labels is not None:
            # ========== GCN模块 ==========
            # h_out2^(p,q) = GCN(L', M)
            h_out2 = self.get_label_embedding(labels)  # (batch, g_dim)
            
            # ========== 注意力模块 ==========
            # 整合h_out1和h_out2
            weighted_drug = self.attention(h_out1, h_out2)  # (batch, 2*n_dim)
            
            # 拼接所有特征
            concatenated = torch.cat((h_out1, h_out2, weighted_drug), dim=1)  # (batch, 4*n_dim + g_dim)
            
            # 第二次预测（使用完整特征）
            scores2 = torch.sigmoid(self.Wr2(concatenated))  # (batch, 15)
            return scores2
        else:
            # 仅使用GNN输出
            return scores1


# ============================================================================
# 2. 架构可视化
# ============================================================================

def visualize_organadr_architecture(output_path='organadr_architecture.png'):
    """
    可视化OrganADR模型架构
    生成类似论文中的架构图
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'input': '#E8F4F8',
        'gnn': '#FFE5E5',
        'gcn': '#E5F5E5',
        'attention': '#FFF5E5',
        'output': '#F0E8FF',
        'arrow': '#666666'
    }
    
    # ========== 模块(e): GNN模块 ==========
    gnn_box = FancyBboxPatch((1, 7), 4, 4, 
                             boxstyle="round,pad=0.3", 
                             facecolor=colors['gnn'],
                             edgecolor='black',
                             linewidth=2)
    ax.add_patch(gnn_box)
    ax.text(3, 10.5, 'GNN Module\n(e)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # 输入：分子特征p和q
    drug_p_box = FancyBboxPatch((0.5, 8.5), 1, 1, 
                                boxstyle="round,pad=0.1",
                                facecolor='lightgreen',
                                edgecolor='darkgreen',
                                linewidth=1.5)
    drug_q_box = FancyBboxPatch((4.5, 8.5), 1, 1,
                                boxstyle="round,pad=0.1",
                                facecolor='lightblue',
                                edgecolor='darkblue',
                                linewidth=1.5)
    ax.add_patch(drug_p_box)
    ax.add_patch(drug_q_box)
    ax.text(1, 9, 'Drug p\n(分子特征)', ha='center', va='center', fontsize=8)
    ax.text(5, 9, 'Drug q\n(分子特征)', ha='center', va='center', fontsize=8)
    
    # GNN处理
    gnn_process = FancyBboxPatch((2, 8), 2, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='white',
                                 edgecolor='black')
    ax.add_patch(gnn_process)
    ax.text(3, 8.5, 'GNN on\nIntegrated KG', ha='center', va='center', fontsize=9)
    
    # 输出 h_out1
    h_out1_box = FancyBboxPatch((2.5, 7), 1, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='yellow',
                                edgecolor='orange',
                                linewidth=1.5)
    ax.add_patch(h_out1_box)
    ax.text(3, 7.4, 'h_out1^(p,q)\nR^(2d1)', ha='center', va='center', fontsize=8)
    
    # 箭头
    ax.arrow(1, 9.5, 0.8, -0.3, head_width=0.15, head_length=0.1, 
             fc=colors['arrow'], ec=colors['arrow'])
    ax.arrow(5, 9.5, -0.8, -0.3, head_width=0.15, head_length=0.1,
             fc=colors['arrow'], ec=colors['arrow'])
    ax.arrow(3, 7.8, 0, -0.5, head_width=0.15, head_length=0.1,
             fc=colors['arrow'], ec=colors['arrow'])
    
    # ========== 模块(f): GCN模块 ==========
    gcn_box = FancyBboxPatch((7, 7), 4, 4,
                             boxstyle="round,pad=0.3",
                             facecolor=colors['gcn'],
                             edgecolor='black',
                             linewidth=2)
    ax.add_patch(gcn_box)
    ax.text(9, 10.5, 'GCN Module\n(f)', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # ADR关联矩阵M
    M_box = FancyBboxPatch((7.5, 9.5), 1.5, 1.5,
                           boxstyle="round,pad=0.1",
                           facecolor='red',
                           edgecolor='darkred',
                           alpha=0.7,
                           linewidth=1.5)
    ax.add_patch(M_box)
    ax.text(8.25, 10.25, 'M\n15×15', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')
    
    # 潜在标签L'
    L_prime_box = FancyBboxPatch((9.5, 9.5), 1.5, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightcyan',
                                 edgecolor='darkcyan',
                                 linewidth=1.5)
    ax.add_patch(L_prime_box)
    ax.text(10.25, 10.25, "L'\nR^15", ha='center', va='center', fontsize=9)
    
    # GCN处理
    gcn_process = FancyBboxPatch((8, 8), 2, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor='white',
                                 edgecolor='black')
    ax.add_patch(gcn_process)
    ax.text(9, 8.4, 'GCN Propagation', ha='center', va='center', fontsize=9)
    
    # 输出 h_out2
    h_out2_box = FancyBboxPatch((8.5, 7), 1, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor='yellow',
                                 edgecolor='orange',
                                 linewidth=1.5)
    ax.add_patch(h_out2_box)
    ax.text(9, 7.4, 'h_out2^(p,q)\nR^(d2)', ha='center', va='center', fontsize=8)
    
    # 箭头
    ax.arrow(3, 7.4, 3.8, 0.2, head_width=0.15, head_length=0.1,
             fc=colors['arrow'], ec=colors['arrow'])
    ax.arrow(9, 8.8, 0, -0.5, head_width=0.15, head_length=0.1,
             fc=colors['arrow'], ec=colors['arrow'])
    
    # ========== 模块(g): 注意力模块 ==========
    attn_box = FancyBboxPatch((12.5, 7), 3, 4,
                              boxstyle="round,pad=0.3",
                              facecolor=colors['attention'],
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(attn_box)
    ax.text(14, 10.5, 'Attention Module\n(g)', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # 注意力机制
    attn_process = FancyBboxPatch((13, 8.5), 2, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='white',
                                  edgecolor='black')
    ax.add_patch(attn_process)
    ax.text(14, 9.25, 'Attention\nMechanism', ha='center', va='center', fontsize=9)
    
    # 输出：预测L
    output_box = FancyBboxPatch((13.5, 7), 1, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='lightpink',
                                edgecolor='red',
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(14, 7.4, 'L\nR^15', ha='center', va='center',
            fontsize=9, fontweight='bold')
    
    # 箭头
    ax.arrow(9, 7.4, 3.2, 0.2, head_width=0.15, head_length=0.1,
             fc=colors['arrow'], ec=colors['arrow'])
    ax.arrow(14, 8.5, 0, -1, head_width=0.15, head_length=0.1,
             fc=colors['arrow'], ec=colors['arrow'])
    
    # ========== 底部说明 ==========
    info_y = 2
    ax.text(8, info_y, 'OrganADR Model Architecture', ha='center', va='center',
            fontsize=16, fontweight='bold')
    ax.text(8, info_y - 0.8, 
            '(e) GNN Module: 整合分子特征和知识图谱信息', 
            ha='center', va='center', fontsize=10)
    ax.text(8, info_y - 1.4,
            '(f) GCN Module: 整合器官层面ADR关联信息',
            ha='center', va='center', fontsize=10)
    ax.text(8, info_y - 2.0,
            '(g) Attention Module: 最终整合和预测15种ADR',
            ha='center', va='center', fontsize=10)
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=colors['gnn'], label='GNN Module'),
        mpatches.Patch(facecolor=colors['gcn'], label='GCN Module'),
        mpatches.Patch(facecolor=colors['attention'], label='Attention Module'),
        mpatches.Patch(facecolor='yellow', label='Intermediate Output'),
        mpatches.Patch(facecolor='lightpink', label='Final Prediction')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 架构图已保存到: {output_path}")
    plt.close()


# ============================================================================
# 3. 数据流可视化
# ============================================================================

def visualize_data_flow(model_outputs, output_path='organadr_dataflow.png'):
    """
    可视化数据流：展示各模块的输入输出形状
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # GNN模块数据流
    ax1 = axes[0]
    ax1.set_title('GNN Module 数据流', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # 输入
    ax1.add_patch(Rectangle((1, 7), 2, 1, facecolor='lightgreen', edgecolor='black'))
    ax1.text(2, 7.5, 'Drug p\n(batch, n_dim)', ha='center', va='center', fontsize=8)
    
    ax1.add_patch(Rectangle((7, 7), 2, 1, facecolor='lightblue', edgecolor='black'))
    ax1.text(8, 7.5, 'Drug q\n(batch, n_dim)', ha='center', va='center', fontsize=8)
    
    # GNN处理
    ax1.add_patch(Rectangle((3.5, 5), 3, 1, facecolor='lightcoral', edgecolor='black'))
    ax1.text(5, 5.5, 'GNN on KG', ha='center', va='center', fontsize=9)
    
    # 输出
    ax1.add_patch(Rectangle((3.5, 2), 3, 1, facecolor='yellow', edgecolor='black'))
    ax1.text(5, 2.5, 'h_out1^(p,q)\n(batch, 2*n_dim)', ha='center', va='center', fontsize=8)
    
    # 箭头
    ax1.arrow(2, 7, 1, -1.5, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax1.arrow(8, 7, -1, -1.5, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax1.arrow(5, 5, 0, -1.5, head_width=0.2, head_length=0.15, fc='black', ec='black')
    
    # GCN模块数据流
    ax2 = axes[1]
    ax2.set_title('GCN Module 数据流', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # 输入
    ax2.add_patch(Rectangle((1, 7), 2, 1, facecolor='lightcyan', edgecolor='black'))
    ax2.text(2, 7.5, "L' (潜在标签)\n(batch, 15)", ha='center', va='center', fontsize=8)
    
    ax2.add_patch(Rectangle((7, 7), 2, 1, facecolor='lightpink', edgecolor='black'))
    ax2.text(8, 7.5, 'M (关联矩阵)\n(15, 15)', ha='center', va='center', fontsize=8)
    
    # GCN处理
    ax2.add_patch(Rectangle((3.5, 5), 3, 1, facecolor='lightgreen', edgecolor='black'))
    ax2.text(5, 5.5, 'GCN Propagation', ha='center', va='center', fontsize=9)
    
    # 输出
    ax2.add_patch(Rectangle((3.5, 2), 3, 1, facecolor='yellow', edgecolor='black'))
    ax2.text(5, 2.5, 'h_out2^(p,q)\n(batch, g_dim)', ha='center', va='center', fontsize=8)
    
    # 箭头
    ax2.arrow(2, 7, 1, -1.5, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax2.arrow(8, 7, -1, -1.5, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax2.arrow(5, 5, 0, -1.5, head_width=0.2, head_length=0.15, fc='black', ec='black')
    
    # 注意力模块数据流
    ax3 = axes[2]
    ax3.set_title('Attention Module 数据流', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    # 输入
    ax3.add_patch(Rectangle((1, 7), 2, 1, facecolor='yellow', edgecolor='black'))
    ax3.text(2, 7.5, 'h_out1^(p,q)\n(batch, 2*n_dim)', ha='center', va='center', fontsize=8)
    
    ax3.add_patch(Rectangle((7, 7), 2, 1, facecolor='yellow', edgecolor='black'))
    ax3.text(8, 7.5, 'h_out2^(p,q)\n(batch, g_dim)', ha='center', va='center', fontsize=8)
    
    # 注意力处理
    ax3.add_patch(Rectangle((3.5, 5), 3, 1, facecolor='orange', edgecolor='black'))
    ax3.text(5, 5.5, 'Attention', ha='center', va='center', fontsize=9)
    
    # 输出
    ax3.add_patch(Rectangle((3.5, 2), 3, 1, facecolor='lightpink', edgecolor='red', linewidth=2))
    ax3.text(5, 2.5, 'L (预测)\n(batch, 15)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 箭头
    ax3.arrow(2, 7, 1, -1.5, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax3.arrow(8, 7, -1, -1.5, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax3.arrow(5, 5, 0, -1.5, head_width=0.2, head_length=0.15, fc='black', ec='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 数据流图已保存到: {output_path}")
    plt.close()


# ============================================================================
# 4. 示例使用
# ============================================================================

def demo_usage():
    """演示如何使用模型"""
    
    print("="*60)
    print("OrganADR 模型演示")
    print("="*60)
    
    # 创建模拟参数
    class Args:
        all_ent = 639  # 实体数量
        all_rel = 15   # 关系数量（15个ADR）
        length = 3     # GNN层数
        feature = 'F'  # 特征模式：'F' (融合) 或 'M' (Morgan)
        n_dim = 32     # GNN输出维度
        g_dim = 16     # GCN输出维度
    
    args = Args()
    
    # 创建模型
    print("\n1. 创建模型...")
    model = OrganADR(
        eval_rel=15,
        args=args,
        drug_features_path=None,  # 使用随机特征
        adj_matrix_path=None      # 使用随机矩阵
    )
    
    print(f"   - GNN输出维度: 2 * {args.n_dim} = {2 * args.n_dim}")
    print(f"   - GCN输出维度: {args.g_dim}")
    print(f"   - 最终预测维度: 15 (ADR数量)")
    
    # 创建模拟输入
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"\n2. 准备输入数据 (batch_size={batch_size})...")
    head = torch.randint(0, args.all_ent, (batch_size,)).to(device)
    tail = torch.randint(0, args.all_ent, (batch_size,)).to(device)
    
    # 创建模拟的知识图谱（简化）
    # 实际应该使用torchdrug的数据结构
    print("   注意: 使用简化的KG表示（实际应使用torchdrug.Graph）")
    KG = torch.rand(100, 100).to(device)  # 简化的邻接矩阵
    
    print(f"   - Head药物ID: {head.cpu().numpy()}")
    print(f"   - Tail药物ID: {tail.cpu().numpy()}")
    
    # 前向传播
    print("\n3. 前向传播...")
    model.eval()
    with torch.no_grad():
        # 仅使用GNN（不使用先验知识）
        scores_simple = model(head, tail, KG, use_prior_knowledge=False)
        print(f"   - 仅GNN输出形状: {scores_simple.shape}")
        print(f"   - 仅GNN输出示例: {scores_simple[0].cpu().numpy()}")
        
        # 使用完整流程（包含GCN和注意力）
        scores_full = model(head, tail, KG, use_prior_knowledge=True)
        print(f"   - 完整模型输出形状: {scores_full.shape}")
        print(f"   - 完整模型输出示例: {scores_full[0].cpu().numpy()}")
    
    # 可视化
    print("\n4. 生成可视化...")
    visualize_organadr_architecture('organadr_architecture.png')
    visualize_data_flow({}, 'organadr_dataflow.png')
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - organadr_architecture.png: 完整架构图")
    print("  - organadr_dataflow.png: 数据流图")
    print("\n使用项目数据:")
    print("  1. 设置 drug_features_path 指向 id2drugfeature_new_639.pkl")
    print("  2. 设置 adj_matrix_path 指向 train_mat.npy")
    print("  3. 使用真实的KG数据（torchdrug格式）")


if __name__ == '__main__':
    # 生成可视化
    print("正在生成架构图...")
    visualize_organadr_architecture()
    visualize_data_flow({})
    
    # 运行演示
    demo_usage()

