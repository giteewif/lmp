"""
使用NUMA特性优化GQA (Grouped Query Attention) 的CPU执行

基于analyze_numa_cpu_attention.py的分析结果，实现NUMA优化策略：
1. 自动选择最优NUMA节点
2. 内存分配和计算绑定到同一NUMA节点
3. 多线程场景下的NUMA感知调度
4. 使用numactl进行进程级优化
"""

import torch
import time
import os
import subprocess
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F
import threading
from concurrent.futures import ThreadPoolExecutor
import json

# ==================== NUMA工具函数 ====================

def check_numactl():
    """检查numactl是否可用"""
    try:
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0, result.stdout
    except:
        return False, ""

def get_numa_cpus(node_id: int) -> List[int]:
    """获取指定NUMA节点的CPU列表"""
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if f'NUMA node{node_id} CPU(s):' in line:
                cpu_str = line.split(':')[1].strip()
                cpus = []
                for part in cpu_str.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        cpus.extend(range(start, end + 1))
                    else:
                        cpus.append(int(part))
                return cpus
    except:
        pass
    return list(range(os.cpu_count()))

def get_num_numa_nodes() -> int:
    """获取NUMA节点数量"""
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'NUMA node(s):' in line:
                return int(line.split(':')[1].strip())
    except:
        pass
    return 1

def set_cpu_affinity(cpus: List[int]):
    """设置当前进程的CPU亲和性"""
    try:
        os.sched_setaffinity(0, cpus)
        return True
    except Exception as e:
        print(f"设置CPU亲和性失败: {e}")
        return False

def get_optimal_numa_node() -> int:
    """
    通过快速基准测试找到最优NUMA节点
    返回性能最好的NUMA节点ID
    """
    print("正在检测最优NUMA节点...")
    
    batch_size = 256  # 使用较小的batch size进行快速测试
    num_heads = 32
    num_kv_heads = 8
    seq_len = 512
    q_seq_len = 1
    head_dim = 128
    dtype = torch.bfloat16
    
    num_numa_nodes = get_num_numa_nodes()
    results = {}
    
    for numa_node in range(num_numa_nodes):
        old_affinity = os.sched_getaffinity(0)
        try:
            cpus = get_numa_cpus(numa_node)
            os.sched_setaffinity(0, cpus)
            
            query = torch.randn(batch_size, num_heads, q_seq_len, head_dim, 
                              dtype=dtype, device='cpu')
            key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, 
                             dtype=dtype, device='cpu')
            value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, 
                              dtype=dtype, device='cpu')
            
            # 快速测试
            times = []
            for _ in range(3):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = scaled_dot_product_attention_gqa(query, key, value)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            results[numa_node] = avg_time
            print(f"  NUMA Node {numa_node}: {avg_time*1000:.3f}ms")
            
        finally:
            os.sched_setaffinity(0, old_affinity)
    
    if results:
        best_node = min(results.items(), key=lambda x: x[1])[0]
        print(f"最优NUMA节点: Node {best_node}")
        return best_node
    
    return 0

# ==================== GQA实现 ====================

def scaled_dot_product_attention_gqa(
    query_states, 
    key_states, 
    value_states, 
    attn_mask=None, 
    dropout_p=0.0, 
    enable_gqa=True, 
    is_causal=False, 
    output_tensor=None):
    """
    优化的GQA实现，支持NUMA感知
    基于test_decode_thread.py中的实现
    """
    num_query_heads = query_states.shape[1]     # e.g. 32
    num_key_heads = key_states.shape[1]
    num_groups = int(num_query_heads // num_key_heads)   # 4 组
    
    query_states = query_states.contiguous()
   
    if output_tensor is None:
        output_tensor = torch.zeros(
            query_states.shape, dtype=query_states.dtype, device=query_states.device, pin_memory=False
        )
    else:
        output_tensor = output_tensor.contiguous()
    
    query_groups = []
    query_indices_list = []
    
    for group_idx in range(num_groups):
        query_indices = torch.arange(group_idx, num_query_heads, num_groups)
        query_group = query_states[:, query_indices, :, :].contiguous()
        query_groups.append(query_group)
        query_indices_list.append(query_indices)
    
    for group_idx in range(num_groups):
        query_group = query_groups[group_idx]
        query_indices = query_indices_list[group_idx]
        
        key_group = key_states
        value_group = value_states

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                query_group, key_group, value_group,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                enable_gqa=enable_gqa,
                is_causal=is_causal
            )
        
        output_tensor[:, query_indices, :, :] = attn_out

    return output_tensor

# ==================== NUMA优化的GQA类 ====================

class NUMAOptimizedGQA:
    """
    NUMA优化的GQA计算类
    自动管理NUMA节点分配和线程绑定
    """
    
    def __init__(self, numa_node: Optional[int] = None, auto_detect: bool = True):
        """
        初始化NUMA优化的GQA计算器
        
        Args:
            numa_node: 指定的NUMA节点ID，如果为None则自动检测
            auto_detect: 是否自动检测最优NUMA节点
        """
        self.num_numa_nodes = get_num_numa_nodes()
        
        if numa_node is None and auto_detect:
            self.numa_node = get_optimal_numa_node()
        elif numa_node is not None:
            self.numa_node = numa_node
        else:
            self.numa_node = 0
        
        self.cpus = get_numa_cpus(self.numa_node)
        self.old_affinity = None
        
        print(f"NUMAOptimizedGQA初始化: 使用NUMA Node {self.numa_node}, CPUs: {self.cpus[:10]}...")
    
    def __enter__(self):
        """上下文管理器入口：绑定到NUMA节点"""
        self.old_affinity = os.sched_getaffinity(0)
        set_cpu_affinity(self.cpus)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口：恢复CPU亲和性"""
        if self.old_affinity:
            os.sched_setaffinity(0, self.old_affinity)
    
    def create_tensors_on_numa(self, shapes: Dict[str, Tuple], dtype=torch.bfloat16):
        """
        在指定NUMA节点上创建tensor
        
        Args:
            shapes: 字典，key为tensor名称，value为shape元组
            dtype: tensor数据类型
        
        Returns:
            包含所有tensor的字典
        """
        old_affinity = os.sched_getaffinity(0)
        try:
            set_cpu_affinity(self.cpus)
            tensors = {}
            for name, shape in shapes.items():
                tensors[name] = torch.randn(shape, dtype=dtype, device='cpu')
            return tensors
        finally:
            os.sched_setaffinity(0, old_affinity)
    
    def compute(self, query, key, value, **kwargs):
        """
        在NUMA节点上执行GQA计算
        
        Args:
            query: query tensor
            key: key tensor
            value: value tensor
            **kwargs: 其他参数传递给scaled_dot_product_attention_gqa
        
        Returns:
            attention输出
        """
        old_affinity = os.sched_getaffinity(0)
        try:
            set_cpu_affinity(self.cpus)
            return scaled_dot_product_attention_gqa(query, key, value, **kwargs)
        finally:
            os.sched_setaffinity(0, old_affinity)

# ==================== 多线程NUMA优化 ====================

class NUMAThreadPool:
    """
    NUMA感知的线程池
    将不同线程分配到不同的NUMA节点，实现负载均衡
    """
    
    def __init__(self, num_threads_per_node: int = 1):
        """
        初始化NUMA线程池
        
        Args:
            num_threads_per_node: 每个NUMA节点的线程数
        """
        self.num_numa_nodes = get_num_numa_nodes()
        self.num_threads_per_node = num_threads_per_node
        self.total_threads = self.num_numa_nodes * num_threads_per_node
        
        # 为每个线程分配NUMA节点
        self.thread_numa_map = {}
        for thread_id in range(self.total_threads):
            numa_node = thread_id % self.num_numa_nodes
            self.thread_numa_map[thread_id] = numa_node
        
        print(f"NUMAThreadPool初始化: {self.total_threads}个线程分布在{self.num_numa_nodes}个NUMA节点")
    
    def execute_on_numa(self, func, *args, numa_node: int, **kwargs):
        """在指定NUMA节点上执行函数"""
        old_affinity = os.sched_getaffinity(0)
        try:
            cpus = get_numa_cpus(numa_node)
            set_cpu_affinity(cpus)
            return func(*args, **kwargs)
        finally:
            os.sched_setaffinity(0, old_affinity)
    
    def parallel_gqa_compute(self, queries, keys, values, num_runs: int = 1):
        """
        并行执行多个GQA计算，每个分配到不同NUMA节点
        
        Args:
            queries: query tensor列表
            keys: key tensor列表
            values: value tensor列表
            num_runs: 每个计算的运行次数
        
        Returns:
            结果列表
        """
        def compute_single(query, key, value, numa_node, num_runs):
            """在指定NUMA节点上执行单个GQA计算"""
            old_affinity = os.sched_getaffinity(0)
            try:
                cpus = get_numa_cpus(numa_node)
                set_cpu_affinity(cpus)
                
                results = []
                for _ in range(num_runs):
                    output = scaled_dot_product_attention_gqa(query, key, value)
                    results.append(output)
                return results
            finally:
                os.sched_setaffinity(0, old_affinity)
        
        num_tasks = len(queries)
        results = []
        
        with ThreadPoolExecutor(max_workers=self.total_threads) as executor:
            futures = []
            for i in range(num_tasks):
                numa_node = i % self.num_numa_nodes
                future = executor.submit(
                    compute_single, 
                    queries[i], keys[i], values[i], 
                    numa_node, num_runs
                )
                futures.append(future)
            
            for future in futures:
                results.append(future.result())
        
        return results

# ==================== 使用示例和基准测试 ====================

def benchmark_optimized_vs_default():
    """对比NUMA优化版本和默认版本的性能"""
    print("=" * 80)
    print("NUMA优化 vs 默认实现性能对比")
    print("=" * 80)
    
    batch_size = 512
    num_heads = 32
    num_kv_heads = 8
    seq_len = 512
    q_seq_len = 1
    head_dim = 128
    dtype = torch.bfloat16
    num_runs = 10
    
    # 准备数据
    query = torch.randn(batch_size, num_heads, q_seq_len, head_dim, dtype=dtype, device='cpu')
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu')
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu')
    
    # 测试1: 默认实现（不绑定NUMA）
    print("\n1. 默认实现（不绑定NUMA）:")
    times_default = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = scaled_dot_product_attention_gqa(query, key, value)
        end = time.perf_counter()
        times_default.append(end - start)
    
    avg_default = np.mean(times_default)
    print(f"   平均时间: {avg_default*1000:.3f}ms")
    
    # 测试2: NUMA优化版本
    print("\n2. NUMA优化版本（绑定到最优节点）:")
    numa_gqa = NUMAOptimizedGQA(auto_detect=True)
    with numa_gqa:
        times_optimized = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = scaled_dot_product_attention_gqa(query, key, value)
            end = time.perf_counter()
            times_optimized.append(end - start)
    
    avg_optimized = np.mean(times_optimized)
    print(f"   平均时间: {avg_optimized*1000:.3f}ms")
    
    # 性能提升
    speedup = (avg_default / avg_optimized - 1) * 100
    print(f"\n性能提升: {speedup:.2f}%")
    
    return {
        'default': avg_default,
        'optimized': avg_optimized,
        'speedup': speedup
    }

def example_usage():
    """使用示例"""
    print("=" * 80)
    print("NUMA优化GQA使用示例")
    print("=" * 80)
    
    batch_size = 512
    num_heads = 32
    num_kv_heads = 8
    seq_len = 512
    q_seq_len = 1
    head_dim = 128
    dtype = torch.bfloat16
    
    # 方法1: 使用上下文管理器（推荐）
    print("\n方法1: 使用上下文管理器")
    numa_gqa = NUMAOptimizedGQA(auto_detect=True)
    
    with numa_gqa:
        # 在NUMA节点上创建tensor
        query = torch.randn(batch_size, num_heads, q_seq_len, head_dim, dtype=dtype, device='cpu')
        key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu')
        value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu')
        
        # 执行GQA计算
        start = time.perf_counter()
        output = numa_gqa.compute(query, key, value)
        end = time.perf_counter()
        
        print(f"  计算时间: {(end-start)*1000:.3f}ms")
        print(f"  输出shape: {output.shape}")
    
    # 方法2: 手动指定NUMA节点
    print("\n方法2: 手动指定NUMA节点")
    numa_gqa_node3 = NUMAOptimizedGQA(numa_node=3, auto_detect=False)
    
    with numa_gqa_node3:
        query = torch.randn(batch_size, num_heads, q_seq_len, head_dim, dtype=dtype, device='cpu')
        key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu')
        value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu')
        
        start = time.perf_counter()
        output = numa_gqa_node3.compute(query, key, value)
        end = time.perf_counter()
        
        print(f"  计算时间: {(end-start)*1000:.3f}ms")
    
    # 方法3: 多线程并行（不同NUMA节点）
    print("\n方法3: 多线程并行（不同NUMA节点）")
    thread_pool = NUMAThreadPool(num_threads_per_node=1)
    
    queries = [torch.randn(batch_size, num_heads, q_seq_len, head_dim, dtype=dtype, device='cpu') 
               for _ in range(4)]
    keys = [torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu') 
            for _ in range(4)]
    values = [torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu') 
              for _ in range(4)]
    
    start = time.perf_counter()
    results = thread_pool.parallel_gqa_compute(queries, keys, values, num_runs=1)
    end = time.perf_counter()
    
    print(f"  并行计算时间: {(end-start)*1000:.3f}ms")
    print(f"  完成 {len(results)} 个任务")

def print_numactl_usage():
    """打印numactl使用说明"""
    print("\n" + "=" * 80)
    print("使用numactl进行进程级NUMA优化")
    print("=" * 80)
    
    num_numa_nodes = get_num_numa_nodes()
    
    print("\n1. 绑定到特定NUMA节点运行:")
    for node in range(num_numa_nodes):
        print(f"   numactl --cpunodebind={node} --membind={node} python your_script.py")
    
    print("\n2. 绑定CPU到节点0，内存可以使用所有节点:")
    print("   numactl --cpunodebind=0 python your_script.py")
    
    print("\n3. 查看NUMA硬件信息:")
    print("   numactl --hardware")
    
    print("\n4. 查看当前进程的NUMA策略:")
    print("   numactl --show")
    
    print("\n5. 在代码中使用（推荐结合NUMAOptimizedGQA）:")
    print("""
   # 在脚本开头添加
   import os
   import subprocess
   
   # 设置最优NUMA节点（例如节点3）
   os.environ['NUMA_NODE'] = '3'
   
   # 或者使用numactl启动
   # numactl --cpunodebind=3 --membind=3 python your_script.py
   """)

def main():
    """主函数"""
    print("NUMA优化GQA CPU执行")
    print("=" * 80)
    
    # 显示系统信息
    num_numa_nodes = get_num_numa_nodes()
    print(f"\n系统信息:")
    print(f"  NUMA节点数: {num_numa_nodes}")
    print(f"  CPU核心数: {os.cpu_count()}")
    print(f"  PyTorch线程数: {torch.get_num_threads()}")
    
    # 运行示例
    example_usage()
    
    # 性能对比
    benchmark_results = benchmark_optimized_vs_default()
    
    # 打印numactl使用说明
    print_numactl_usage()
    
    # 总结
    print("\n" + "=" * 80)
    print("优化建议总结")
    print("=" * 80)
    print("""
1. 使用NUMAOptimizedGQA类自动管理NUMA节点绑定
2. 通过auto_detect=True自动选择最优NUMA节点
3. 对于批量处理，使用NUMAThreadPool实现多节点并行
4. 在启动脚本时使用numactl进行进程级绑定
5. 确保tensor分配和计算在同一NUMA节点上
6. 根据实际工作负载调整线程数
    """)
    
    if benchmark_results['speedup'] > 0:
        print(f"✓ NUMA优化带来 {benchmark_results['speedup']:.2f}% 的性能提升")
    else:
        print("⚠ 当前环境下NUMA优化效果不明显，可能需要调整配置")

if __name__ == "__main__":
    main()



