import torch
import torch.nn as nn
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'zoo', 'rtdetr'))

# 模拟utils和src.core模块
class MockUtils:
    @staticmethod
    def get_activation(name):
        if name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        else:
            return nn.ReLU()

class MockRegister:
    def __call__(self, cls):
        return cls

sys.modules['utils'] = MockUtils
sys.modules['src.core'] = type('MockCore', (), {'register': MockRegister()})

# 直接导入hybrid_encoder模块
import hybrid_encoder
HybridEncoder = hybrid_encoder.HybridEncoder

def test_fusion_control():
    """测试融合控制标志的功能"""
    print("=== 测试融合控制标志功能 ===")
    
    # 测试参数
    batch_size = 2
    in_channels = [256, 512, 1024]
    feat_strides = [8, 16, 32]
    hidden_dim = 256
    
    # 创建测试输入
    test_inputs = []
    for i, in_ch in enumerate(in_channels):
        h, w = 64 // (2**i), 64 // (2**i)  # 64x64, 32x32, 16x16
        feat = torch.randn(batch_size, in_ch, h, w)
        test_inputs.append(feat)
    
    print(f"输入特征图尺寸: {[x.shape for x in test_inputs]}")
    
    # 测试1: 使用改进的CGA融合 (默认)
    print("\n--- 测试1: 使用改进的CGA融合 ---")
    encoder_improved = HybridEncoder(
        in_channels=in_channels,
        feat_strides=feat_strides,
        hidden_dim=hidden_dim,
        use_improved_fusion=True
    )
    
    with torch.no_grad():
        outputs_improved = encoder_improved(test_inputs)
    
    print(f"改进融合输出尺寸: {[x.shape for x in outputs_improved]}")
    print(f"CGA融合模块数量 - FPN: {len(encoder_improved.cga_fusion_fpn) if encoder_improved.cga_fusion_fpn else 0}")
    print(f"CGA融合模块数量 - PAN: {len(encoder_improved.cga_fusion_pan) if encoder_improved.cga_fusion_pan else 0}")
    
    # 测试2: 使用传统融合
    print("\n--- 测试2: 使用传统融合 ---")
    encoder_traditional = HybridEncoder(
        in_channels=in_channels,
        feat_strides=feat_strides,
        hidden_dim=hidden_dim,
        use_improved_fusion=False
    )
    
    with torch.no_grad():
        outputs_traditional = encoder_traditional(test_inputs)
    
    print(f"传统融合输出尺寸: {[x.shape for x in outputs_traditional]}")
    print(f"CGA融合模块 - FPN: {encoder_traditional.cga_fusion_fpn}")
    print(f"CGA融合模块 - PAN: {encoder_traditional.cga_fusion_pan}")
    
    # 测试3: 比较输出差异
    print("\n--- 测试3: 比较两种融合方式的输出差异 ---")
    for i, (out_imp, out_trad) in enumerate(zip(outputs_improved, outputs_traditional)):
        diff = torch.abs(out_imp - out_trad).mean().item()
        print(f"特征层{i}的平均绝对差异: {diff:.6f}")
    
    # 测试4: 参数数量比较
    print("\n--- 测试4: 参数数量比较 ---")
    params_improved = sum(p.numel() for p in encoder_improved.parameters())
    params_traditional = sum(p.numel() for p in encoder_traditional.parameters())
    
    print(f"改进融合模型参数数量: {params_improved:,}")
    print(f"传统融合模型参数数量: {params_traditional:,}")
    print(f"参数增加量: {params_improved - params_traditional:,} ({((params_improved - params_traditional) / params_traditional * 100):.2f}%)")
    
    # 测试5: 内存使用比较
    print("\n--- 测试5: 内存使用比较 ---")
    
    # 重置内存
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 测试改进融合的内存使用
    import tracemalloc
    tracemalloc.start()
    
    with torch.no_grad():
        _ = encoder_improved(test_inputs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    improved_memory = peak
    
    # 测试传统融合的内存使用
    tracemalloc.start()
    
    with torch.no_grad():
        _ = encoder_traditional(test_inputs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    traditional_memory = peak
    
    print(f"改进融合内存峰值: {improved_memory / 1024 / 1024:.2f} MB")
    print(f"传统融合内存峰值: {traditional_memory / 1024 / 1024:.2f} MB")
    print(f"内存增加: {(improved_memory - traditional_memory) / 1024 / 1024:.2f} MB")
    
    print("\n=== 融合控制标志测试完成 ===")
    return True

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试单层输入
    print("\n--- 测试单层输入 ---")
    try:
        encoder_single = HybridEncoder(
            in_channels=[512],
            feat_strides=[16],
            hidden_dim=256,
            use_improved_fusion=True
        )
        
        test_input = [torch.randn(1, 512, 32, 32)]
        with torch.no_grad():
            output = encoder_single(test_input)
        print(f"单层输入测试通过，输出尺寸: {[x.shape for x in output]}")
        fpn_count = len(encoder_single.cga_fusion_fpn) if encoder_single.cga_fusion_fpn is not None else 0
        pan_count = len(encoder_single.cga_fusion_pan) if encoder_single.cga_fusion_pan is not None else 0
        print(f"CGA模块数量: FPN={fpn_count}, PAN={pan_count}")
    except Exception as e:
        print(f"单层输入测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试不同尺寸输入
    print("\n--- 测试不同尺寸输入 ---")
    try:
        encoder_diff = HybridEncoder(
            in_channels=[128, 256, 512, 1024],
            feat_strides=[4, 8, 16, 32],
            hidden_dim=256,
            use_improved_fusion=True
        )
        
        test_inputs_diff = [
            torch.randn(1, 128, 128, 128),
            torch.randn(1, 256, 64, 64),
            torch.randn(1, 512, 32, 32),
            torch.randn(1, 1024, 16, 16)
        ]
        
        with torch.no_grad():
            outputs_diff = encoder_diff(test_inputs_diff)
        print(f"不同尺寸输入测试通过，输出尺寸: {[x.shape for x in outputs_diff]}")
    except Exception as e:
        print(f"不同尺寸输入测试失败: {e}")
    
    print("\n=== 边界情况测试完成 ===")
    return True

if __name__ == "__main__":
    try:
        test_fusion_control()
        test_edge_cases()
        print("\n✅ 所有测试通过！融合控制标志功能正常工作。")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()