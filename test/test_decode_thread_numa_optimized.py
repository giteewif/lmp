"""
test_decode_thread.py的NUMA优化版本

展示如何在实际代码中集成NUMA优化
"""

import torch
from torch.nn.attention import sdpa_kernel
from torch.nn.attention import SDPBackend
import torch.nn.functional as F
import time
import os

# ==================== NUMA优化工具 ====================

def get_numa_cpus(node_id: int):
    """获取指定NUMA节点的CPU列表"""
    try:
        import subprocess
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

def get_optimal_numa_node():
    """快速检测最优NUMA节点（简化版）"""
    # 根据analyze_numa_cpu_attention.py的结果，Node 3通常最优
    # 实际使用时可以运行完整检测
    return 3

# 全局NUMA节点设置
_optimal_numa_node = None

def get_numa_node():
    """获取最优NUMA节点"""
    global _optimal_numa_node
    if _optimal_numa_node is None:
        _optimal_numa_node = get_optimal_numa_node()
    return _optimal_numa_node

def bind_to_numa_node(numa_node: int):
    """绑定当前线程到指定NUMA节点"""
    try:
        cpus = get_numa_cpus(numa_node)
        os.sched_setaffinity(0, cpus)
        return True
    except:
        return False

# ==================== 原有代码（添加NUMA优化） ====================

def scaled_dot_product_attention_help(
    query_states, 
    key_states, 
    value_states, 
    attn_mask=None, dropout_p=0.0, enable_gqa=False, is_causal=False, output_tensor=None,
    use_numa=True):
    """
    优化的attention函数，支持NUMA绑定
    
    Args:
        use_numa: 是否使用NUMA优化
    """
    # NUMA优化：绑定到最优节点
    old_affinity = None
    if use_numa:
        old_affinity = os.sched_getaffinity(0)
        numa_node = get_numa_node()
        bind_to_numa_node(numa_node)
    
    try:
        time_start = time.time()
       
        num_query_heads = query_states.shape[1]     # e.g. 32
        num_key_heads = key_states.shape[1]
        num_groups = int(num_query_heads//num_key_heads)   # 4 组
        
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
            
            # 优化5: 使用预取的数据，避免重复内存访问
            key_group = key_states    # (batch, 8, seq_len, head_dim)
            value_group = value_states # (batch, 8, seq_len, head_dim)

            time_start_tmp = time.time()
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    query_group, key_group, value_group,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    enable_gqa=enable_gqa,
                    is_causal=is_causal
                )
            print(f"single group {group_idx} real attn out cost {time.time() - time_start_tmp} s")
            
            output_tensor[:, query_indices, :, :] = attn_out

        print(f"dot attn help cost {time.time()-time_start:.6f} seconds")
        return output_tensor
    finally:
        # 恢复CPU亲和性
        if use_numa and old_affinity:
            os.sched_setaffinity(0, old_affinity)

def attn(layer, query_states, key_states, value_states, use_numa=True):
    """
    优化的attention函数，支持NUMA绑定
    """
    with torch.no_grad():
        attn_output = scaled_dot_product_attention_help(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=False,
            use_numa=use_numa
        )
    return attn_output

# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    print("=" * 80)
    print("NUMA优化GQA使用示例")
    print("=" * 80)
    
    # 设置参数
    batch_size = 512
    num_heads = 32
    num_kv_heads = 8
    seq_len = 512
    q_seq_len = 1
    head_dim = 128
    dtype = torch.bfloat16
    
    # 在最优NUMA节点上创建tensor
    numa_node = get_numa_node()
    old_affinity = os.sched_getaffinity(0)
    try:
        bind_to_numa_node(numa_node)
        
        query = torch.randn(batch_size, num_heads, q_seq_len, head_dim, 
                          dtype=dtype, device='cpu')
        key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, 
                         dtype=dtype, device='cpu')
        value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, 
                          dtype=dtype, device='cpu')
    finally:
        os.sched_setaffinity(0, old_affinity)
    
    # 测试1: 不使用NUMA优化
    print("\n1. 不使用NUMA优化:")
    times_no_numa = []
    for _ in range(5):
        start = time.perf_counter()
        output = attn(None, query, key, value, use_numa=False)
        end = time.perf_counter()
        times_no_numa.append(end - start)
    
    avg_no_numa = sum(times_no_numa) / len(times_no_numa)
    print(f"   平均时间: {avg_no_numa*1000:.3f}ms")
    
    # 测试2: 使用NUMA优化
    print("\n2. 使用NUMA优化:")
    times_numa = []
    for _ in range(5):
        start = time.perf_counter()
        output = attn(None, query, key, value, use_numa=True)
        end = time.perf_counter()
        times_numa.append(end - start)
    
    avg_numa = sum(times_numa) / len(times_numa)
    print(f"   平均时间: {avg_numa*1000:.3f}ms")
    
    # 性能提升
    if avg_no_numa > 0:
        speedup = (avg_no_numa / avg_numa - 1) * 100
        print(f"\n性能提升: {speedup:.2f}%")
        print(f"使用的NUMA节点: Node {numa_node}")

if __name__ == "__main__":
    example_usage()

