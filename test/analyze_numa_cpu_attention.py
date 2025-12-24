"""
分析NUMA对CPU Attention计算的影响

基于test_decode_thread.py中的实际attention实现，测试：
1. 不同NUMA节点内存分配对CPU attention计算性能的影响
2. 线程绑定到不同NUMA节点的影响
3. GQA (Grouped Query Attention) 场景下的NUMA影响
4. 不同batch size和sequence length下的NUMA影响
"""

import torch
import time
import os
import subprocess
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F
import json

# 检查numactl是否可用
def check_numactl():
    try:
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0, result.stdout
    except:
        return False, ""

def get_numa_info():
    """获取NUMA节点信息"""
    numa_available, numa_output = check_numactl()
    numa_nodes = []
    
    if numa_available:
        lines = numa_output.strip().split('\n')
        for line in lines:
            if 'node ' in line.lower():
                numa_nodes.append(line.strip())
    else:
        # 从lscpu获取
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split('\n'):
                if 'NUMA node' in line and 'CPU' in line:
                    numa_nodes.append(line.strip())
        except:
            pass
    
    return numa_nodes

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

def scaled_dot_product_attention_help(
    query_states, 
    key_states, 
    value_states, 
    attn_mask=None, dropout_p=0.0, enable_gqa=False, is_causal=False, output_tensor=None):
    """
    基于test_decode_thread.py中的实际attention实现
    支持GQA (Grouped Query Attention)
    """
    time_start = time.time()
   
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
        query_group = query_states[:, query_indices, :, :].contiguous()  # 确保连续内存
        query_groups.append(query_group)
        query_indices_list.append(query_indices)
    
    for group_idx in range(num_groups):
        query_group = query_groups[group_idx]
        query_indices = query_indices_list[group_idx]
        
        key_group = key_states    # (batch, 8, seq_len, head_dim)
        value_group = value_states # (batch, 8, seq_len, head_dim)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                query_group, key_group, value_group,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                enable_gqa=enable_gqa,
                is_causal=is_causal
            )
        
        output_tensor[:, query_indices, :, :] = attn_out

    return output_tensor, time.time() - time_start

def benchmark_attention_cpu_numa(
    query, key, value, 
    numa_node: Optional[int] = None,
    num_runs: int = 10, 
    warmup: int = 3,
    use_gqa: bool = True):
    """在指定NUMA节点上benchmark CPU attention计算"""
    
    old_affinity = os.sched_getaffinity(0)
    
    try:
        if numa_node is not None:
            cpus = get_numa_cpus(numa_node)
            os.sched_setaffinity(0, cpus)
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                if use_gqa:
                    _, _ = scaled_dot_product_attention_help(
                        query, key, value, enable_gqa=True, is_causal=False
                    )
                else:
                    _ = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                if use_gqa:
                    output, _ = scaled_dot_product_attention_help(
                        query, key, value, enable_gqa=True, is_causal=False
                    )
                else:
                    output = F.scaled_dot_product_attention(query, key, value, is_causal=False)
            end = time.perf_counter()
            times.append(end - start)
        
        return times, output
        
    finally:
        os.sched_setaffinity(0, old_affinity)

def create_tensor_on_numa(shape, dtype, numa_node: int = None, pin_memory: bool = False):
    """在指定NUMA节点上创建tensor"""
    old_affinity = os.sched_getaffinity(0)
    try:
        if numa_node is not None:
            cpus = get_numa_cpus(numa_node)
            os.sched_setaffinity(0, cpus)
        
        tensor = torch.randn(shape, dtype=dtype, device='cpu')
        if pin_memory:
            tensor = tensor.pin_memory()
    finally:
        os.sched_setaffinity(0, old_affinity)
    
    return tensor

def test_numa_memory_allocation_cpu_attention():
    """测试不同NUMA节点内存分配对CPU attention的影响"""
    print("=" * 80)
    print("测试1: NUMA节点内存分配对CPU Attention计算的影响")
    print("=" * 80)
    
    batch_size = 512
    num_heads = 32
    num_kv_heads = 8  # GQA: 32 query heads, 8 key-value heads
    seq_len = 512
    q_seq_len = 1  # decode阶段
    head_dim = 128
    dtype = torch.bfloat16
    
    num_numa_nodes = get_num_numa_nodes()
    print(f"\n检测到 {num_numa_nodes} 个NUMA节点")
    
    results = {}
    
    # 测试不同NUMA节点
    for numa_node in range(num_numa_nodes):
        print(f"\n测试 NUMA Node {numa_node}...")
        
        # 在当前NUMA节点创建tensor
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
            
            # Benchmark CPU attention (使用GQA)
            times_gqa, _ = benchmark_attention_cpu_numa(
                query, key, value, numa_node=numa_node, num_runs=10, use_gqa=True
            )
            
            # Benchmark 标准attention (不使用GQA，需要key/value reshape)
            key_full = key.repeat_interleave(num_heads // num_kv_heads, dim=1)
            value_full = value.repeat_interleave(num_heads // num_kv_heads, dim=1)
            times_std, _ = benchmark_attention_cpu_numa(
                query, key_full, value_full, numa_node=numa_node, num_runs=10, use_gqa=False
            )
            
            avg_time_gqa = np.mean(times_gqa)
            avg_time_std = np.mean(times_std)
            
            results[numa_node] = {
                'avg_time_gqa': avg_time_gqa,
                'avg_time_std': avg_time_std,
                'std_time_gqa': np.std(times_gqa),
                'std_time_std': np.std(times_std),
                'min_time_gqa': np.min(times_gqa),
                'max_time_gqa': np.max(times_gqa),
                'times_gqa': times_gqa,
                'times_std': times_std
            }
            
            print(f"  NUMA Node {numa_node}:")
            print(f"    GQA Attention: 平均时间 {avg_time_gqa*1000:.3f}ms "
                  f"(std: {np.std(times_gqa)*1000:.3f}ms, "
                  f"min: {np.min(times_gqa)*1000:.3f}ms, "
                  f"max: {np.max(times_gqa)*1000:.3f}ms)")
            print(f"    标准Attention: 平均时间 {avg_time_std*1000:.3f}ms "
                  f"(std: {np.std(times_std)*1000:.3f}ms)")
            
        finally:
            os.sched_setaffinity(0, old_affinity)
    
    # 分析结果
    print("\n" + "=" * 80)
    print("结果分析:")
    print("=" * 80)
    
    if results:
        best_node_gqa = min(results.items(), key=lambda x: x[1]['avg_time_gqa'])
        worst_node_gqa = max(results.items(), key=lambda x: x[1]['avg_time_gqa'])
        
        print(f"\nGQA Attention:")
        print(f"  最快NUMA节点: Node {best_node_gqa[0]} ({best_node_gqa[1]['avg_time_gqa']*1000:.3f}ms)")
        print(f"  最慢NUMA节点: Node {worst_node_gqa[0]} ({worst_node_gqa[1]['avg_time_gqa']*1000:.3f}ms)")
        print(f"  性能差异: {(worst_node_gqa[1]['avg_time_gqa'] / best_node_gqa[1]['avg_time_gqa'] - 1) * 100:.2f}%")
        
        best_node_std = min(results.items(), key=lambda x: x[1]['avg_time_std'])
        worst_node_std = max(results.items(), key=lambda x: x[1]['avg_time_std'])
        
        print(f"\n标准Attention:")
        print(f"  最快NUMA节点: Node {best_node_std[0]} ({best_node_std[1]['avg_time_std']*1000:.3f}ms)")
        print(f"  最慢NUMA节点: Node {worst_node_std[0]} ({worst_node_std[1]['avg_time_std']*1000:.3f}ms)")
        print(f"  性能差异: {(worst_node_std[1]['avg_time_std'] / best_node_std[1]['avg_time_std'] - 1) * 100:.2f}%")
    
    return results

def test_numa_different_batch_sizes():
    """测试不同batch size下的NUMA影响"""
    print("\n" + "=" * 80)
    print("测试2: 不同Batch Size下的NUMA影响")
    print("=" * 80)
    
    num_heads = 32
    num_kv_heads = 8
    seq_len = 512
    q_seq_len = 1
    head_dim = 128
    dtype = torch.bfloat16
    
    batch_sizes = [128, 256, 512, 1024]
    num_numa_nodes = get_num_numa_nodes()
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n测试 Batch Size = {batch_size}...")
        results[batch_size] = {}
        
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
                
                times, _ = benchmark_attention_cpu_numa(
                    query, key, value, numa_node=numa_node, num_runs=5, use_gqa=True
                )
                
                avg_time = np.mean(times)
                results[batch_size][numa_node] = avg_time
                
                print(f"  NUMA Node {numa_node}: {avg_time*1000:.3f}ms")
                
            finally:
                os.sched_setaffinity(0, old_affinity)
    
    # 分析结果
    print("\n" + "=" * 80)
    print("结果分析:")
    print("=" * 80)
    
    for batch_size in batch_sizes:
        if batch_size in results:
            node_times = results[batch_size]
            best_node = min(node_times.items(), key=lambda x: x[1])
            worst_node = max(node_times.items(), key=lambda x: x[1])
            
            print(f"\nBatch Size {batch_size}:")
            print(f"  最快: Node {best_node[0]} ({best_node[1]*1000:.3f}ms)")
            print(f"  最慢: Node {worst_node[0]} ({worst_node[1]*1000:.3f}ms)")
            print(f"  差异: {(worst_node[1] / best_node[1] - 1) * 100:.2f}%")
    
    return results

def test_numa_thread_affinity():
    """测试线程亲和性设置对CPU attention的影响"""
    print("\n" + "=" * 80)
    print("测试3: 线程亲和性对CPU Attention计算的影响")
    print("=" * 80)
    
    batch_size = 512
    num_heads = 32
    num_kv_heads = 8
    seq_len = 512
    q_seq_len = 1
    head_dim = 128
    dtype = torch.bfloat16
    
    num_numa_nodes = get_num_numa_nodes()
    
    # 测试不同线程数设置
    thread_counts = [1, 4, 8, 16, 32, 64, 128]
    
    print("\n测试不同线程数设置:")
    print(f"测试 {num_numa_nodes} 个NUMA节点")
    
    results = {}
    
    for num_threads in thread_counts:
        torch.set_num_threads(num_threads)
        results[num_threads] = {}
        
        print(f"\n线程数: {num_threads}")
        
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
                
                times, _ = benchmark_attention_cpu_numa(
                    query, key, value, numa_node=numa_node, num_runs=5, use_gqa=True
                )
                avg_time = np.mean(times)
                results[num_threads][numa_node] = avg_time
                
                print(f"  NUMA Node {numa_node}: {avg_time*1000:.3f}ms")
                
            finally:
                os.sched_setaffinity(0, old_affinity)
    
    # 分析结果
    print("\n" + "=" * 80)
    print("结果分析:")
    print("=" * 80)
    
    for num_threads in thread_counts:
        if num_threads in results:
            node_times = results[num_threads]
            best_node = min(node_times.items(), key=lambda x: x[1])
            avg_time = np.mean(list(node_times.values()))
            
            print(f"\n线程数 {num_threads:3d}:")
            print(f"  最快NUMA节点: Node {best_node[0]} ({best_node[1]*1000:.3f}ms)")
            print(f"  平均时间: {avg_time*1000:.3f}ms")
    
    return results

def test_cross_numa_access():
    """测试跨NUMA节点访问的影响"""
    print("\n" + "=" * 80)
    print("测试4: 跨NUMA节点访问的影响")
    print("=" * 80)
    
    batch_size = 512
    num_heads = 32
    num_kv_heads = 8
    seq_len = 512
    q_seq_len = 1
    head_dim = 128
    dtype = torch.bfloat16
    
    num_numa_nodes = get_num_numa_nodes()
    
    print("\n测试场景: 在不同NUMA节点创建tensor，在另一个NUMA节点计算")
    
    results = {}
    
    for compute_node in range(num_numa_nodes):
        for memory_node in range(num_numa_nodes):
            print(f"\n内存分配在 Node {memory_node}, 计算在 Node {compute_node}...")
            
            # 在memory_node创建tensor
            old_affinity = os.sched_getaffinity(0)
            try:
                cpus = get_numa_cpus(memory_node)
                os.sched_setaffinity(0, cpus)
                
                query = torch.randn(batch_size, num_heads, q_seq_len, head_dim, 
                                  dtype=dtype, device='cpu')
                key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, 
                                 dtype=dtype, device='cpu')
                value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, 
                                  dtype=dtype, device='cpu')
            finally:
                os.sched_setaffinity(0, old_affinity)
            
            # 在compute_node计算
            times, _ = benchmark_attention_cpu_numa(
                query, key, value, numa_node=compute_node, num_runs=5, use_gqa=True
            )
            avg_time = np.mean(times)
            
            key = f"mem_{memory_node}_compute_{compute_node}"
            results[key] = {
                'memory_node': memory_node,
                'compute_node': compute_node,
                'avg_time': avg_time,
                'is_local': memory_node == compute_node
            }
            
            local_str = "(本地)" if memory_node == compute_node else "(跨NUMA)"
            print(f"  时间: {avg_time*1000:.3f}ms {local_str}")
    
    # 分析结果
    print("\n" + "=" * 80)
    print("结果分析:")
    print("=" * 80)
    
    local_times = [r['avg_time'] for r in results.values() if r['is_local']]
    cross_times = [r['avg_time'] for r in results.values() if not r['is_local']]
    
    if local_times and cross_times:
        avg_local = np.mean(local_times)
        avg_cross = np.mean(cross_times)
        
        print(f"\n本地NUMA访问 (内存和计算在同一节点):")
        print(f"  平均时间: {avg_local*1000:.3f}ms")
        
        print(f"\n跨NUMA访问 (内存和计算在不同节点):")
        print(f"  平均时间: {avg_cross*1000:.3f}ms")
        
        print(f"\n性能差异: {(avg_cross / avg_local - 1) * 100:.2f}%")
    
    return results

def main():
    """主函数"""
    print("NUMA对CPU Attention计算影响分析")
    print("=" * 80)
    
    # 显示NUMA信息
    numa_info = get_numa_info()
    print("\n系统NUMA信息:")
    for info in numa_info:
        print(f"  {info}")
    
    num_numa_nodes = get_num_numa_nodes()
    print(f"\nNUMA节点数: {num_numa_nodes}")
    print(f"CPU核心数: {os.cpu_count()}")
    print(f"PyTorch线程数: {torch.get_num_threads()}")
    
    # 运行测试
    all_results = {}
    
    try:
        # 测试1: NUMA内存分配影响
        results1 = test_numa_memory_allocation_cpu_attention()
        all_results['memory_allocation'] = results1
        
        # 测试2: 不同batch size
        results2 = test_numa_different_batch_sizes()
        all_results['batch_sizes'] = results2
        
        # 测试3: 线程亲和性
        results3 = test_numa_thread_affinity()
        all_results['thread_affinity'] = results3
        
        # 测试4: 跨NUMA访问
        results4 = test_cross_numa_access()
        all_results['cross_numa'] = results4
        
        print("\n" + "=" * 80)
        print("分析完成!")
        print("=" * 80)
        
        # 总结建议
        print("\n优化建议:")
        if results1:
            best_node_gqa = min(results1.items(), key=lambda x: x[1]['avg_time_gqa'])[0]
            print(f"1. 建议将CPU attention计算绑定到NUMA Node {best_node_gqa}")
        
        if results4:
            local_times = [r['avg_time'] for r in results4.values() if r['is_local']]
            cross_times = [r['avg_time'] for r in results4.values() if not r['is_local']]
            if local_times and cross_times:
                avg_local = np.mean(local_times)
                avg_cross = np.mean(cross_times)
                overhead = (avg_cross / avg_local - 1) * 100
                print(f"2. 跨NUMA访问性能开销: {overhead:.2f}%，建议避免跨NUMA访问")
        
        if results3:
            # 找到最佳线程数
            best_thread_config = None
            best_time = float('inf')
            for num_threads, node_times in results3.items():
                avg_time = np.mean(list(node_times.values()))
                if avg_time < best_time:
                    best_time = avg_time
                    best_thread_config = num_threads
            if best_thread_config:
                print(f"3. 建议设置线程数: {best_thread_config}")
        
        print("4. 考虑使用numactl来设置进程的NUMA亲和性:")
        if results1:
            best_node = min(results1.items(), key=lambda x: x[1]['avg_time_gqa'])[0]
            print(f"   numactl --cpunodebind={best_node} --membind={best_node} python your_script.py")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存结果到JSON文件
    try:
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(all_results)
        with open('numa_attention_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print("\n结果已保存到 numa_attention_results.json")
    except Exception as e:
        print(f"\n保存结果时出错: {e}")

if __name__ == "__main__":
    main()

