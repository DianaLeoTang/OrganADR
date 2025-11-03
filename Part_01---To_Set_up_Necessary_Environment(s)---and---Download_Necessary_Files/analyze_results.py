"""
OrganADR 训练结果分析脚本
用于在环境1中分析环境2生成的训练结果

使用方法:
1. 激活环境1: conda activate environment1
2. 运行脚本: python analyze_results.py
"""

import os
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def parse_log_file(log_file):
    """解析训练日志文件，提取指标"""
    data = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析格式: epoch: 1 [Valid] PR-AUC: 0.524600 ROC-AUC: 0.526888 ...
            match = re.match(r'epoch:\s*(\d+)\s+\[Valid\]\s+PR-AUC:\s*([\d.]+)\s+ROC-AUC:\s*([\d.]+)\s+ACC:\s*([\d.]+)\s+PREC:\s*([\d.]+)\s+REC:\s*([\d.]+)\s+HL:\s*([\d.]+)\s+\[Test\]\s+PR-AUC:\s*([\d.]+)\s+ROC-AUC:\s*([\d.]+)\s+ACC:\s*([\d.]+)\s+PREC:\s*([\d.]+)\s+REC:\s*([\d.]+)\s+HL:\s*([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                data.append({
                    'epoch': epoch,
                    'valid_pr_auc': float(match.group(2)),
                    'valid_roc_auc': float(match.group(3)),
                    'valid_acc': float(match.group(4)),
                    'valid_prec': float(match.group(5)),
                    'valid_rec': float(match.group(6)),
                    'valid_hl': float(match.group(7)),
                    'test_pr_auc': float(match.group(8)),
                    'test_roc_auc': float(match.group(9)),
                    'test_acc': float(match.group(10)),
                    'test_prec': float(match.group(11)),
                    'test_rec': float(match.group(12)),
                    'test_hl': float(match.group(13))
                })
    return pd.DataFrame(data)

def load_all_results(results_dir):
    """加载所有训练结果"""
    results_dir = Path(results_dir)
    all_data = []
    
    # 遍历所有结果目录
    for result_dir in results_dir.glob('*'):
        if not result_dir.is_dir():
            continue
        
        # 查找日志文件
        log_files = list(result_dir.glob('*.txt'))
        if not log_files:
            continue
        
        for log_file in log_files:
            try:
                df = parse_log_file(log_file)
                if not df.empty:
                    # 从目录名提取实验信息
                    dir_name = result_dir.name
                    parts = dir_name.split('_')
                    df['experiment'] = dir_name
                    df['dataset'] = parts[0] if len(parts) > 0 else 'unknown'
                    all_data.append(df)
                    print(f"✓ 加载: {log_file}")
            except Exception as e:
                print(f"✗ 加载失败 {log_file}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def plot_training_curves(df, output_dir='analysis_output'):
    """绘制训练曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个实验绘制训练曲线
    for exp_name in df['experiment'].unique():
        exp_df = df[df['experiment'] == exp_name].copy()
        exp_df = exp_df.sort_values('epoch')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves: {exp_name}', fontsize=14, fontweight='bold')
        
        # PR-AUC
        axes[0, 0].plot(exp_df['epoch'], exp_df['valid_pr_auc'], label='Valid PR-AUC', marker='o')
        axes[0, 0].plot(exp_df['epoch'], exp_df['test_pr_auc'], label='Test PR-AUC', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('PR-AUC')
        axes[0, 0].set_title('PR-AUC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ROC-AUC
        axes[0, 1].plot(exp_df['epoch'], exp_df['valid_roc_auc'], label='Valid ROC-AUC', marker='o')
        axes[0, 1].plot(exp_df['epoch'], exp_df['test_roc_auc'], label='Test ROC-AUC', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].set_title('ROC-AUC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1, 0].plot(exp_df['epoch'], exp_df['valid_acc'], label='Valid ACC', marker='o')
        axes[1, 0].plot(exp_df['epoch'], exp_df['test_acc'], label='Test ACC', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision & Recall
        axes[1, 1].plot(exp_df['epoch'], exp_df['valid_prec'], label='Valid Precision', marker='o')
        axes[1, 1].plot(exp_df['epoch'], exp_df['valid_rec'], label='Valid Recall', marker='^')
        axes[1, 1].plot(exp_df['epoch'], exp_df['test_prec'], label='Test Precision', marker='s')
        axes[1, 1].plot(exp_df['epoch'], exp_df['test_rec'], label='Test Recall', marker='D')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_name = exp_name.replace('/', '_').replace('\\', '_')
        plt.savefig(os.path.join(output_dir, f'training_curves_{safe_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存训练曲线: {safe_name}")

def compare_experiments(df, output_dir='analysis_output'):
    """比较不同实验的结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 找到每个实验的最佳epoch（基于test PR-AUC）
    best_results = []
    for exp_name in df['experiment'].unique():
        exp_df = df[df['experiment'] == exp_name].copy()
        best_idx = exp_df['test_pr_auc'].idxmax()
        best_results.append(exp_df.loc[best_idx])
    
    best_df = pd.DataFrame(best_results)
    
    # 保存最佳结果表格
    best_df.to_csv(os.path.join(output_dir, 'best_results.csv'), index=False)
    print(f"✓ 保存最佳结果表格: {len(best_df)} 个实验")
    
    # 绘制对比图
    if len(best_df) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Experiment Comparison (Best Epoch)', fontsize=14, fontweight='bold')
        
        # 准备数据
        x_pos = range(len(best_df))
        exp_names = [name[:30] + '...' if len(name) > 30 else name for name in best_df['experiment']]
        
        # PR-AUC对比
        axes[0, 0].barh(x_pos, best_df['test_pr_auc'], alpha=0.7)
        axes[0, 0].set_yticks(x_pos)
        axes[0, 0].set_yticklabels(exp_names, fontsize=8)
        axes[0, 0].set_xlabel('Test PR-AUC')
        axes[0, 0].set_title('PR-AUC Comparison')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # ROC-AUC对比
        axes[0, 1].barh(x_pos, best_df['test_roc_auc'], alpha=0.7, color='orange')
        axes[0, 1].set_yticks(x_pos)
        axes[0, 1].set_yticklabels(exp_names, fontsize=8)
        axes[0, 1].set_xlabel('Test ROC-AUC')
        axes[0, 1].set_title('ROC-AUC Comparison')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Accuracy对比
        axes[1, 0].barh(x_pos, best_df['test_acc'], alpha=0.7, color='green')
        axes[1, 0].set_yticks(x_pos)
        axes[1, 0].set_yticklabels(exp_names, fontsize=8)
        axes[1, 0].set_xlabel('Test Accuracy')
        axes[1, 0].set_title('Accuracy Comparison')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 综合指标雷达图（简化为条形图）
        metrics = ['test_pr_auc', 'test_roc_auc', 'test_acc', 'test_prec', 'test_rec']
        mean_scores = best_df[metrics].mean()
        axes[1, 1].bar(range(len(mean_scores)), mean_scores.values, alpha=0.7)
        axes[1, 1].set_xticks(range(len(mean_scores)))
        axes[1, 1].set_xticklabels(['PR-AUC', 'ROC-AUC', 'ACC', 'Precision', 'Recall'], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Mean Score')
        axes[1, 1].set_title('Average Metrics Across Experiments')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'experiment_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 保存实验对比图")

def print_summary(df):
    """打印结果摘要"""
    print("\n" + "="*60)
    print("训练结果摘要")
    print("="*60)
    
    if df.empty:
        print("未找到训练结果文件")
        return
    
    print(f"\n总共找到 {len(df['experiment'].unique())} 个实验")
    print(f"总共 {len(df)} 条记录")
    
    # 每个实验的最佳结果
    print("\n各实验最佳结果 (基于 Test PR-AUC):")
    print("-"*60)
    for exp_name in df['experiment'].unique():
        exp_df = df[df['experiment'] == exp_name].copy()
        best_idx = exp_df['test_pr_auc'].idxmax()
        best = exp_df.loc[best_idx]
        print(f"\n实验: {exp_name}")
        print(f"  最佳Epoch: {int(best['epoch'])}")
        print(f"  Test PR-AUC:  {best['test_pr_auc']:.6f}")
        print(f"  Test ROC-AUC: {best['test_roc_auc']:.6f}")
        print(f"  Test Accuracy: {best['test_acc']:.6f}")
        print(f"  Test Precision: {best['test_prec']:.6f}")
        print(f"  Test Recall: {best['test_rec']:.6f}")

def main():
    """主函数"""
    # 结果目录路径（相对于脚本位置）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "Part_02---Demo_of_Training_and_Evaluating_OrganADR" / "results" / "demo"
    
    print("="*60)
    print("OrganADR 训练结果分析")
    print("="*60)
    print(f"\n查找结果目录: {results_dir}")
    
    if not results_dir.exists():
        print(f"✗ 结果目录不存在: {results_dir}")
        print("\n提示:")
        print("1. 确保已运行训练脚本生成结果")
        print("2. 结果应该保存在: Part_02---Demo_of_Training_and_Evaluating_OrganADR/results/demo/")
        return
    
    # 加载结果
    print("\n正在加载训练结果...")
    df = load_all_results(results_dir)
    
    if df.empty:
        print("\n✗ 未找到任何训练结果文件")
        return
    
    # 打印摘要
    print_summary(df)
    
    # 生成分析图表
    print("\n正在生成分析图表...")
    output_dir = script_dir / "analysis_output"
    plot_training_curves(df, output_dir)
    compare_experiments(df, output_dir)
    
    # 保存完整数据
    df.to_csv(output_dir / 'all_results.csv', index=False)
    print(f"\n✓ 所有结果已保存到: {output_dir}")
    print(f"  - all_results.csv: 完整数据")
    print(f"  - best_results.csv: 最佳结果")
    print(f"  - training_curves_*.png: 训练曲线图")
    print(f"  - experiment_comparison.png: 实验对比图")

if __name__ == '__main__':
    main()

