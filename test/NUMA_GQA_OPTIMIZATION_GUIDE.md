# NUMA优化GQA CPU执行指南

基于`analyze_numa_cpu_attention.py`的分析结果，本文档介绍如何使用NUMA特性优化GQA（Grouped Query Attention）的CPU执行。

## 分析结果总结

根据实际测试，NUMA对CPU attention有显著影响：

- **GQA Attention性能差异**: 最快节点(Node 3) vs 最慢节点(Node 0) = **17.07%**
- **标准Attention性能差异**: 最快节点(Node 3) vs 最慢节点(Node 1) = **4.37%**
- **跨NUMA访问开销**: 相比本地NUMA访问有明显性能损失

## 优化策略

### 1. 自动选择最优NUMA节点

系统会自动检测并选择性能最好的NUMA节点：

```python
from optimize_gqa_with_numa import NUMAOptimizedGQA

# 自动检测最优NUMA节点
numa_gqa = NUMAOptimizedGQA(auto_detect=True)

with numa_gqa:
    # 在最优NUMA节点上执行GQA计算
    output = scaled_dot_product_attention_gqa(query, key, value)
```

### 2. 手动指定NUMA节点

如果已知最优节点，可以手动指定：

```python
# 根据分析结果，Node 3通常性能最好
numa_gqa = NUMAOptimizedGQA(numa_node=3, auto_detect=False)

with numa_gqa:
    output = scaled_dot_product_attention_gqa(query, key, value)
```

### 3. 内存分配和计算绑定

确保tensor分配和计算在同一NUMA节点：

```python
numa_gqa = NUMAOptimizedGQA(auto_detect=True)

with numa_gqa:
    # 在同一NUMA节点上创建tensor
    query = torch.randn(batch_size, num_heads, q_seq_len, head_dim, 
                       dtype=dtype, device='cpu')
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, 
                     dtype=dtype, device='cpu')
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, 
                       dtype=dtype, device='cpu')
    
    # 在同一NUMA节点上计算
    output = numa_gqa.compute(query, key, value)
```

### 4. 多线程并行优化

对于批量处理，使用NUMA感知的线程池：

```python
from optimize_gqa_with_numa import NUMAThreadPool

thread_pool = NUMAThreadPool(num_threads_per_node=1)

# 准备多个任务
queries = [query1, query2, query3, query4]
keys = [key1, key2, key3, key4]
values = [value1, value2, value3, value4]

# 并行执行，自动分配到不同NUMA节点
results = thread_pool.parallel_gqa_compute(queries, keys, values)
```

### 5. 进程级NUMA绑定（使用numactl）

在启动脚本时使用numactl进行进程级绑定：

```bash
# 绑定到最优NUMA节点（例如Node 3）
numactl --cpunodebind=3 --membind=3 python test_decode_thread.py

# 或者只绑定CPU，内存可以使用所有节点
numactl --cpunodebind=3 python test_decode_thread.py
```

## 集成到现有代码

### 修改test_decode_thread.py中的attention函数

在`test_decode_thread.py`中，可以这样优化`scaled_dot_product_attention_help`函数：

```python
from optimize_gqa_with_numa import NUMAOptimizedGQA

# 在文件开头创建全局NUMA优化器
_global_numa_gqa = None

def get_numa_gqa():
    global _global_numa_gqa
    if _global_numa_gqa is None:
        _global_numa_gqa = NUMAOptimizedGQA(auto_detect=True)
    return _global_numa_gqa

def scaled_dot_product_attention_help(
    query_states, 
    key_states, 
    value_states, 
    attn_mask=None, dropout_p=0.0, enable_gqa=False, is_causal=False, output_tensor=None):
    
    # 使用NUMA优化
    numa_gqa = get_numa_gqa()
    with numa_gqa:
        # 原有实现...
        # 确保tensor在NUMA节点上
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        
        # 执行GQA计算
        # ... 原有代码 ...
```

### 在attn函数中使用

```python
def attn(layer, query_states, key_states, value_states):
    numa_gqa = get_numa_gqa()
    with numa_gqa:
        attn_output = scaled_dot_product_attention_help(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
    return attn_output
```

## 性能优化建议

### 1. 根据工作负载选择策略

- **单次计算**: 使用`NUMAOptimizedGQA`自动选择最优节点
- **批量处理**: 使用`NUMAThreadPool`实现多节点并行
- **长期运行**: 使用`numactl`进行进程级绑定

### 2. 线程数配置

根据NUMA节点数和CPU核心数调整：

```python
# 每个NUMA节点使用其CPU核心数
num_numa_nodes = 4
cpus_per_node = 24  # 根据实际情况调整
torch.set_num_threads(cpus_per_node)
```

### 3. 内存预分配

在NUMA节点上预分配内存池：

```python
numa_gqa = NUMAOptimizedGQA(auto_detect=True)

with numa_gqa:
    # 预分配内存池
    memory_pool = {
        'query': torch.empty(batch_size, num_heads, q_seq_len, head_dim, 
                            dtype=dtype, device='cpu'),
        'key': torch.empty(batch_size, num_kv_heads, seq_len, head_dim, 
                          dtype=dtype, device='cpu'),
        'value': torch.empty(batch_size, num_kv_heads, seq_len, head_dim, 
                            dtype=dtype, device='cpu'),
    }
```

## 验证优化效果

运行基准测试：

```python
python optimize_gqa_with_numa.py
```

这会输出：
- 最优NUMA节点检测结果
- 优化前后性能对比
- 使用示例和最佳实践

## 常见问题

### Q1: 如何知道哪个NUMA节点最优？

运行`analyze_numa_cpu_attention.py`或使用`get_optimal_numa_node()`函数自动检测。

### Q2: 多进程场景如何处理？

每个进程应该绑定到不同的NUMA节点，避免竞争：

```python
import multiprocessing as mp

def worker_process(numa_node):
    numa_gqa = NUMAOptimizedGQA(numa_node=numa_node, auto_detect=False)
    # 执行计算...

# 启动多个进程，每个绑定到不同NUMA节点
processes = []
for node in range(4):
    p = mp.Process(target=worker_process, args=(node,))
    p.start()
    processes.append(p)
```

### Q3: 与GPU混合使用时的注意事项？

- CPU attention计算绑定到与GPU最近的NUMA节点
- 使用`numactl --preferred=节点ID`允许内存跨节点但优先使用指定节点

## 性能预期

根据实际测试，NUMA优化可以带来：

- **GQA Attention**: 10-20% 性能提升
- **标准Attention**: 3-5% 性能提升
- **批量处理**: 通过多节点并行，接近线性加速

## 参考资料

- `analyze_numa_cpu_attention.py`: NUMA影响分析脚本
- `optimize_gqa_with_numa.py`: NUMA优化实现
- `numa_attention_results.json`: 详细测试结果



