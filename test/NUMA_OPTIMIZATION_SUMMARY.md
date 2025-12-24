# NUMA优化GQA CPU执行 - 快速参考

## 核心发现

根据`analyze_numa_cpu_attention.py`的分析：

- **GQA Attention性能差异**: 最快节点 vs 最慢节点 = **17.07%**
- **最优NUMA节点**: 通常是 Node 2 或 Node 3（需根据实际测试确定）
- **跨NUMA访问开销**: 明显性能损失，应避免

## 三种优化方法

### 方法1: 使用NUMAOptimizedGQA类（推荐）

```python
from optimize_gqa_with_numa import NUMAOptimizedGQA

# 自动检测最优节点
numa_gqa = NUMAOptimizedGQA(auto_detect=True)

with numa_gqa:
    # 在同一NUMA节点创建tensor和计算
    query = torch.randn(batch_size, num_heads, q_seq_len, head_dim, dtype=dtype, device='cpu')
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu')
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device='cpu')
    
    output = scaled_dot_product_attention_gqa(query, key, value)
```

### 方法2: 在现有代码中添加NUMA绑定

修改`scaled_dot_product_attention_help`函数：

```python
def scaled_dot_product_attention_help(..., use_numa=True):
    old_affinity = None
    if use_numa:
        old_affinity = os.sched_getaffinity(0)
        numa_node = get_optimal_numa_node()  # 或使用固定值如3
        bind_to_numa_node(numa_node)
    
    try:
        # 原有计算代码...
        return output_tensor
    finally:
        if use_numa and old_affinity:
            os.sched_setaffinity(0, old_affinity)
```

### 方法3: 使用numactl启动脚本（最简单）

```bash
# 绑定到最优NUMA节点（根据测试结果选择，通常是Node 2或3）
numactl --cpunodebind=2 --membind=2 python test_decode_thread.py

# 或者只绑定CPU，允许内存跨节点
numactl --cpunodebind=2 python test_decode_thread.py
```

## 快速集成步骤

### 步骤1: 检测最优NUMA节点

```bash
python analyze_numa_cpu_attention.py
```

查看输出中的"最快NUMA节点"。

### 步骤2: 选择集成方式

**选项A - 最小改动**（推荐）:
```bash
numactl --cpunodebind=2 --membind=2 python test_decode_thread.py
```

**选项B - 代码集成**:
```python
# 在test_decode_thread.py开头添加
from optimize_gqa_with_numa import get_optimal_numa_node, bind_to_numa_node
import os

# 在attn函数调用前绑定
numa_node = get_optimal_numa_node()  # 或使用固定值
bind_to_numa_node(numa_node)
```

**选项C - 完整优化**:
使用`NUMAOptimizedGQA`类包装所有attention计算。

### 步骤3: 验证效果

对比优化前后的性能：
- 运行`python optimize_gqa_with_numa.py`查看基准测试
- 或在实际代码中对比`use_numa=True/False`的性能

## 性能预期

- **单次计算**: 5-20% 性能提升（取决于工作负载大小）
- **批量处理**: 通过多节点并行，接近线性加速
- **长期运行**: 更稳定的性能表现

**注意**: 
- 对于小规模计算，频繁切换CPU亲和性可能带来开销
- 建议在批量处理或长期运行场景中使用NUMA优化
- 使用`numactl`进行进程级绑定通常比代码级绑定更高效

## 最佳实践

1. **内存和计算在同一NUMA节点**: 避免跨NUMA访问
2. **批量处理使用多节点**: 使用`NUMAThreadPool`实现并行
3. **长期运行使用numactl**: 进程级绑定更稳定
4. **定期重新检测**: NUMA性能可能因系统负载变化

## 文件说明

- `analyze_numa_cpu_attention.py`: NUMA影响分析工具
- `optimize_gqa_with_numa.py`: NUMA优化实现和工具类
- `test_decode_thread_numa_optimized.py`: 集成示例
- `NUMA_GQA_OPTIMIZATION_GUIDE.md`: 详细使用指南
- `numa_attention_results.json`: 测试结果数据

## 常见问题

**Q: 如何确定最优NUMA节点？**
A: 运行`analyze_numa_cpu_attention.py`或`optimize_gqa_with_numa.py`，查看自动检测结果。

**Q: 多进程场景如何处理？**
A: 每个进程绑定到不同NUMA节点，避免竞争。

**Q: 与GPU混合使用？**
A: 绑定到与GPU最近的NUMA节点（通常通过PCIe拓扑确定）。

**Q: 性能提升不明显？**
A: 检查系统负载，确保NUMA节点未被其他进程占用；尝试不同的NUMA节点。

## 命令速查

```bash
# 查看NUMA信息
numactl --hardware
lscpu | grep -i numa

# 绑定运行
numactl --cpunodebind=2 --membind=2 python script.py

# 查看当前进程NUMA策略
numactl --show

# 运行分析
python analyze_numa_cpu_attention.py

# 运行优化示例
python optimize_gqa_with_numa.py
```

