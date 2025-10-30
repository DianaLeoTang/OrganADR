#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
设备检测测试脚本 - 用于验证Mac M芯片适配
"""

import torch
import sys

def main():
    print("="*60)
    print("OrganADR - 设备检测测试")
    print("="*60)
    print()
    
    # Python版本
    print(f"✓ Python版本: {sys.version.split()[0]}")
    
    # PyTorch版本
    print(f"✓ PyTorch版本: {torch.__version__}")
    print()
    
    # 检测CUDA
    print("【CUDA检测】")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA可用")
        print(f"  - CUDA版本: {torch.version.cuda}")
        print(f"  - GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"  ✗ CUDA不可用")
    print()
    
    # 检测MPS (Apple Silicon)
    print("【MPS检测 (Apple Silicon)】")
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            print(f"  ✓ MPS可用 (Apple Silicon GPU加速)")
            print(f"  - 建议使用MPS进行训练")
            # 测试MPS是否真的可用
            try:
                test_tensor = torch.randn(3, 3).to('mps')
                print(f"  ✓ MPS测试通过")
            except Exception as e:
                print(f"  ⚠ MPS不稳定: {e}")
        else:
            print(f"  ✗ MPS不可用")
    else:
        print(f"  ✗ MPS不支持 (PyTorch版本可能过低)")
    print()
    
    # CPU信息
    print("【CPU信息】")
    print(f"  ✓ CPU始终可用")
    print()
    
    # 推荐设备
    print("【推荐使用的设备】")
    if torch.cuda.is_available():
        recommended = "CUDA GPU"
        device = torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        recommended = "MPS (Apple Silicon)"
        device = torch.device('mps')
    else:
        recommended = "CPU"
        device = torch.device('cpu')
    
    print(f"  → {recommended}")
    print(f"  → 设备对象: {device}")
    print()
    
    # 进行简单的张量运算测试
    print("【设备测试】")
    try:
        # 创建测试张量
        a = torch.randn(100, 100).to(device)
        b = torch.randn(100, 100).to(device)
        c = torch.matmul(a, b)
        print(f"  ✓ 矩阵运算测试通过")
        print(f"  - 测试张量设备: {c.device}")
        
        # 测试稀疏张量（OrganADR中使用）
        indices = torch.LongTensor([[0, 1, 2], [1, 2, 0]])
        values = torch.FloatTensor([1.0, 2.0, 3.0])
        sparse_tensor = torch.sparse_coo_tensor(indices, values, (3, 3)).to(device)
        print(f"  ✓ 稀疏张量测试通过")
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        print(f"  → 建议使用CPU模式")
    
    print()
    print("="*60)
    print("测试完成！")
    print("="*60)
    print()
    
    # 使用建议
    if recommended == "MPS (Apple Silicon)":
        print("💡 使用建议：")
        print("  - 您的Mac支持MPS加速")
        print("  - 训练速度会比CPU快3-5倍")
        print("  - 请确保Mac已连接电源以获得最佳性能")
    elif recommended == "CPU":
        print("💡 使用建议：")
        print("  - 当前将使用CPU进行训练")
        print("  - 如果训练速度较慢，考虑：")
        print("    1. 升级PyTorch到2.0+以支持MPS")
        print("    2. 更新macOS到12.3+")
        print("    3. 减小batch size")

if __name__ == "__main__":
    main()

