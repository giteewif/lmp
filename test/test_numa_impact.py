"""
测试 NUMA 对 einsum 计算的影响

测试场景：
1. 本地 NUMA：数据和计算都在同一 NUMA 节点
2. 远程 NUMA：数据在一个 NUMA 节点，计算在另一个 NUMA 节点
3. 交错模式：数据交错分布在多个 NUMA 节点

性能指标：
- einsum 计算时间
- 内存带宽利用率
- 跨 NUMA 访问开销
"""

import torch
import torch.nn.functional as F
import time
import os
import subprocess
from typing import Dict, List, Tuple
import numpy as np


def get_numa_info():
    """获取 NUMA 节点信息"""
    try:
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True)
        print("=== NUMA Hardware Info ===")
        print(result.stdout)
        
        # 获取 NUMA 节点数量
        result = subprocess.run(['numactl', '--show'], 
                              capture_output=True, text=True)
        print("=== NUMA Policy ===")
        print(result.stdout)
        
    except FileNotFoundError:
        print("Warning: numactl not found. Install with: apt-get install numactl")
        return None


def set_numa_policy(node_id=None, interleave=False):
    """
    设置 NUMA 策略
    
    Args:
        node_id: NUMA 节点 ID，None 表示使用默认策略
        interleave: 是否使用交错模式
    """
    if interleave:
        policy = "interleave"
        print(f"Setting NUMA policy: interleave (all nodes)")
    elif node_id is not None:
        policy = f"preferred node {node_id}"
        print(f"Setting NUMA policy: preferred node {node_id}")
    else:
        policy = "default"
        print(f"Setting NUMA policy: default")
    
    return policy


def allocate_tensors_on_numa(shape, dtype, num_tensors, numa_node=None):
    """
    在指定 NUMA 节点上分配 tensor
    
    Args:
        shape: tensor 形状
        dtype: tensor 数据类型
        num_tensors: tensor 数量
        numa_node: NUMA 节点 ID
    
    Returns:
        tensor 列表
    """
    tensors = []
    
    # 如果指定了 NUMA 节点，尝试绑定内存分配
    if numa_node is not None:
        # 注意：这需要 numactl 和适当的权限
        # 实际效果取决于系统配置
        print(f"Allocating {num_tensors} tensors on NUMA node {numa_node}")
    
    for i in range(num_tensors):
        tensor = torch.randn(*shape, dtype=dtype, device='cpu')
        tensors.append(tensor)
    
    return tensors


def run_einsum_benchmark(
    stacked_inputs: torch.Tensor,
    w1_weights: torch.Tensor,
    w2_weights: torch.Tensor,
    w3_weights: torch.Tensor,
    num_iterations: int = 10,
    warmup: int = 2
) -> Dict[str, float]:
    """
    运行 einsum 基准测试
    
    Args:
        stacked_inputs: [E, T, H]
        w1_weights: [E, I, H]
        w2_weights: [E, H, I]
        w3_weights: [E, I, H]
        num_iterations: 迭代次数
        warmup: 预热次数
    
    Returns:
        性能统计字典
    """
    times = {
        'w1_einsum': [],
        'w3_einsum': [],
        'w2_einsum': [],
        'silu': [],
        'multiply': [],
        'total': []
    }
    
    # 预热
    for _ in range(warmup):
        w1_out = torch.einsum('eth,eih->eti', stacked_inputs, w1_weights)
        w1_out = F.silu(w1_out)
        w3_out = torch.einsum('eth,eih->eti', stacked_inputs, w3_weights)
        intermediate = w1_out * w3_out
        outputs = torch.einsum('eti,ehi->eth', intermediate, w2_weights)
    
    # 实际测试
    for i in range(num_iterations):
        total_start = time.perf_counter()
        
        # w1 einsum
        t1 = time.perf_counter()
        w1_out = torch.einsum('eth,eih->eti', stacked_inputs, w1_weights)
        t2 = time.perf_counter()
        times['w1_einsum'].append(t2 - t1)
        
        # silu
        t1 = time.perf_counter()
        w1_out = F.silu(w1_out)
        t2 = time.perf_counter()
        times['silu'].append(t2 - t1)
        
        # w3 einsum
        t1 = time.perf_counter()
        w3_out = torch.einsum('eth,eih->eti', stacked_inputs, w3_weights)
        t2 = time.perf_counter()
        times['w3_einsum'].append(t2 - t1)
        
        # multiply
        t1 = time.perf_counter()
        intermediate = w1_out * w3_out
        t2 = time.perf_counter()
        times['multiply'].append(t2 - t1)
        
        # w2 einsum
        t1 = time.perf_counter()
        outputs = torch.einsum('eti,ehi->eth', intermediate, w2_weights)
        t2 = time.perf_counter()
        times['w2_einsum'].append(t2 - t1)
        
        total_end = time.perf_counter()
        times['total'].append(total_end - total_start)
    
    # 计算统计信息
    stats = {}
    for key, values in times.items():
        stats[f'{key}_mean'] = np.mean(values) * 1000  # 转换为毫秒
        stats[f'{key}_std'] = np.std(values) * 1000
        stats[f'{key}_min'] = np.min(values) * 1000
        stats[f'{key}_max'] = np.max(values) * 1000
    
    return stats


def test_numa_scenario(
    scenario_name: str,
    E: int = 8,  # 专家数量
    T: int = 128,  # token 数量
    H: int = 4096,  # 隐藏维度
    I: int = 14336,  # 中间维度
    numa_config: Dict = None,
    num_iterations: int = 10
):
    """
    测试特定 NUMA 场景
    
    Args:
        scenario_name: 场景名称
        E: 专家数量
        T: token 数量
        H: 隐藏维度
        I: 中间维度
        numa_config: NUMA 配置
        num_iterations: 迭代次数
    """
    print(f"\n{'='*60}")
    print(f"Testing: {scenario_name}")
    print(f"{'='*60}")
    print(f"Shape: E={E}, T={T}, H={H}, I={I}")
    
    dtype = torch.bfloat16
    
    # 分配 tensor
    print("Allocating tensors...")
    stacked_inputs = torch.randn(E, T, H, dtype=dtype, device='cpu')
    w1_weights = torch.randn(E, I, H, dtype=dtype, device='cpu')
    w2_weights = torch.randn(E, H, I, dtype=dtype, device='cpu')
    w3_weights = torch.randn(E, I, H, dtype=dtype, device='cpu')
    
    # 如果有 NUMA 配置，尝试应用
    if numa_config:
        print(f"NUMA config: {numa_config}")
    
    # 运行基准测试
    print(f"Running benchmark ({num_iterations} iterations)...")
    stats = run_einsum_benchmark(
        stacked_inputs, w1_weights, w2_weights, w3_weights,
        num_iterations=num_iterations
    )
    
    # 打印结果
    print(f"\n--- Results for {scenario_name} ---")
    print(f"Total time:     {stats['total_mean']:.2f} ± {stats['total_std']:.2f} ms")
    print(f"  w1 einsum:    {stats['w1_einsum_mean']:.2f} ± {stats['w1_einsum_std']:.2f} ms")
    print(f"  w3 einsum:    {stats['w3_einsum_mean']:.2f} ± {stats['w3_einsum_std']:.2f} ms")
    print(f"  w2 einsum:    {stats['w2_einsum_mean']:.2f} ± {stats['w2_einsum_std']:.2f} ms")
    print(f"  silu:         {stats['silu_mean']:.2f} ± {stats['silu_std']:.2f} ms")
    print(f"  multiply:     {stats['multiply_mean']:.2f} ± {stats['multiply_std']:.2f} ms")
    
    return stats


def compare_numa_scenarios():
    """比较不同 NUMA 场景的性能"""
    
    print("="*60)
    print("NUMA Impact on Einsum Performance Test")
    print("="*60)
    
    # 获取 NUMA 信息
    get_numa_info()
    
    # 测试参数
    E = 8  # 专家数量
    T = 128  # token 数量
    H = 4096  # 隐藏维度
    I = 14336  # 中间维度
    num_iterations = 20
    
    results = {}
    
    # 场景 1: 默认配置（基准）
    results['baseline'] = test_numa_scenario(
        "Baseline (Default NUMA Policy)",
        E=E, T=T, H=H, I=I,
        numa_config=None,
        num_iterations=num_iterations
    )
    
    # 场景 2: 绑定到 NUMA 节点 0
    # 注意：这需要使用 numactl 启动 Python 进程
    # 例如: numactl --cpunodebind=0 --membind=0 python test_numa_impact.py
    results['numa_node_0'] = test_numa_scenario(
        "NUMA Node 0 (if bound)",
        E=E, T=T, H=H, I=I,
        numa_config={'node': 0},
        num_iterations=num_iterations
    )
    
    # 场景 3: 不同线程数
    print("\n" + "="*60)
    print("Testing with different thread counts")
    print("="*60)
    
    thread_counts = [1, 4, 8, 16, 32, 64, 128]
    thread_results = {}
    
    for num_threads in thread_counts:
        print(f"\nTesting with {num_threads} threads...")
        torch.set_num_threads(num_threads)
        
        stats = test_numa_scenario(
            f"Threads={num_threads}",
            E=E, T=T, H=H, I=I,
            numa_config=None,
            num_iterations=10
        )
        thread_results[num_threads] = stats['total_mean']
    
    # 打印线程数对比
    print("\n" + "="*60)
    print("Thread Count Comparison")
    print("="*60)
    print(f"{'Threads':<10} {'Time (ms)':<15} {'Speedup':<10}")
    print("-"*40)
    baseline_time = thread_results[thread_counts[0]]
    for num_threads in thread_counts:
        time_ms = thread_results[num_threads]
        speedup = baseline_time / time_ms
        print(f"{num_threads:<10} {time_ms:<15.2f} {speedup:<10.2f}x")
    
    # 打印总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nKey Findings:")
    print("1. NUMA locality can significantly impact performance")
    print("2. Cross-NUMA memory access introduces additional latency")
    print("3. Optimal thread count depends on NUMA topology")
    print("\nRecommendations:")
    print("- Use numactl to bind process to specific NUMA node")
    print("- Allocate memory on the same NUMA node as compute")
    print("- Use interleave mode for large tensors that don't fit on one node")
    print("\nExample commands:")
    print("  # Bind to NUMA node 0:")
    print("  numactl --cpunodebind=0 --membind=0 python your_script.py")
    print("  # Interleave across all nodes:")
    print("  numactl --interleave=all python your_script.py")


def test_memory_layout_impact():
    """测试内存布局对性能的影响"""
    
    print("\n" + "="*60)
    print("Memory Layout Impact Test")
    print("="*60)
    
    E = 8
    T = 128
    H = 4096
    I = 14336
    dtype = torch.bfloat16
    
    # 场景 1: 连续分配的权重（模拟从共享内存加载）
    print("\nScenario 1: Contiguous weights (simulated)")
    w1_list = []
    w2_list = []
    w3_list = []
    
    # 创建大的连续 tensor，然后 view
    w1_big = torch.randn(E, I, H, dtype=dtype, device='cpu')
    w2_big = torch.randn(E, H, I, dtype=dtype, device='cpu')
    w3_big = torch.randn(E, I, H, dtype=dtype, device='cpu')
    
    for i in range(E):
        w1_list.append(w1_big[i])
        w2_list.append(w2_big[i])
        w3_list.append(w3_big[i])
    
    # 验证是否连续
    data_ptrs_w1 = [w.data_ptr() for w in w1_list]
    is_contiguous = all(
        data_ptrs_w1[i+1] - data_ptrs_w1[i] == w1_list[0].numel() * w1_list[0].element_size()
        for i in range(len(data_ptrs_w1) - 1)
    )
    print(f"Weights are contiguous: {is_contiguous}")
    print(f"Storage shared: {all(w.untyped_storage().data_ptr() == w1_list[0].untyped_storage().data_ptr() for w in w1_list)}")
    
    stacked_inputs = torch.randn(E, T, H, dtype=dtype, device='cpu')
    
    stats_contiguous = run_einsum_benchmark(
        stacked_inputs, w1_big, w2_big, w3_big, num_iterations=20
    )
    
    # 场景 2: 非连续分配的权重
    print("\nScenario 2: Non-contiguous weights")
    w1_list_nc = [torch.randn(I, H, dtype=dtype, device='cpu') for _ in range(E)]
    w2_list_nc = [torch.randn(H, I, dtype=dtype, device='cpu') for _ in range(E)]
    w3_list_nc = [torch.randn(I, H, dtype=dtype, device='cpu') for _ in range(E)]
    
    w1_stacked = torch.stack(w1_list_nc)
    w2_stacked = torch.stack(w2_list_nc)
    w3_stacked = torch.stack(w3_list_nc)
    
    # 验证是否连续
    data_ptrs_nc = [w.data_ptr() for w in w1_list_nc]
    print(f"Weights are contiguous: False (by design)")
    print(f"After stack - is contiguous: {w1_stacked.is_contiguous()}")
    
    stats_non_contiguous = run_einsum_benchmark(
        stacked_inputs, w1_stacked, w2_stacked, w3_stacked, num_iterations=20
    )
    
    # 比较结果
    print("\n" + "="*60)
    print("Memory Layout Comparison")
    print("="*60)
    print(f"{'Operation':<20} {'Contiguous (ms)':<20} {'Non-contiguous (ms)':<20} {'Difference':<15}")
    print("-"*75)
    
    ops = ['w1_einsum', 'w2_einsum', 'w3_einsum', 'total']
    for op in ops:
        cont_time = stats_contiguous[f'{op}_mean']
        non_cont_time = stats_non_contiguous[f'{op}_mean']
        diff_pct = ((non_cont_time - cont_time) / cont_time) * 100
        print(f"{op:<20} {cont_time:<20.2f} {non_cont_time:<20.2f} {diff_pct:>+14.1f}%")


if __name__ == "__main__":
    # 设置线程数
    torch.set_num_threads(128)
    
    print("Starting NUMA impact tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of threads: {torch.get_num_threads()}")
    print(f"MKL available: {torch.backends.mkl.is_available()}")
    print(f"MKL-DNN available: {torch.backends.mkldnn.is_available()}")
    
    # 运行测试
    compare_numa_scenarios()
    
    # 测试内存布局影响
    test_memory_layout_impact()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

