# NUMA优化GQA - 最佳实践

## 何时使用NUMA优化

### ✅ 适合使用的场景

1. **批量处理大量数据**
   - Batch size >= 512
   - 序列长度 >= 512
   - 多次连续计算

2. **长期运行的服务**
   - 推理服务
   - 批处理任务
   - 持续计算任务

3. **多进程/多线程并行**
   - 每个进程/线程绑定到不同NUMA节点
   - 避免NUMA节点竞争

### ❌ 不适合使用的场景

1. **单次小规模计算**
   - CPU亲和性切换开销可能超过收益
   - Batch size < 128

2. **频繁切换NUMA节点**
   - 每次计算都切换会带来开销
   - 应该保持在同一节点上

## 推荐方案

### 方案1: 进程级绑定（最推荐）

使用`numactl`在启动时绑定，避免代码中的切换开销：

```bash
# 检测最优节点后
numactl --cpunodebind=2 --membind=2 python test_decode_thread.py
```

**优点**:
- 无代码修改
- 无运行时开销
- 适用于所有计算

### 方案2: 初始化时绑定（推荐）

在程序启动时绑定一次，后续所有计算都在该节点：

```python
import os

# 在程序开头，只绑定一次
optimal_numa_node = 2  # 根据测试结果确定
cpus = get_numa_cpus(optimal_numa_node)
os.sched_setaffinity(0, cpus)

# 后续所有计算自动在该节点上
```

**优点**:
- 只需绑定一次
- 无后续切换开销
- 代码改动小

### 方案3: 批量计算时绑定

只在批量处理时使用NUMA优化：

```python
def process_batch(queries, keys, values, use_numa=True):
    """批量处理时使用NUMA优化"""
    if use_numa and len(queries) > 10:  # 批量处理才优化
        numa_node = get_optimal_numa_node()
        old_affinity = os.sched_getaffinity(0)
        bind_to_numa_node(numa_node)
        try:
            results = [compute_attention(q, k, v) for q, k, v in zip(queries, keys, values)]
        finally:
            os.sched_setaffinity(0, old_affinity)
        return results
    else:
        # 小批量不使用NUMA优化
        return [compute_attention(q, k, v) for q, k, v in zip(queries, keys, values)]
```

## 性能调优建议

### 1. 检测最优节点

```bash
# 运行完整分析
python analyze_numa_cpu_attention.py

# 或快速检测
python -c "from optimize_gqa_with_numa import get_optimal_numa_node; print(get_optimal_numa_node())"
```

### 2. 验证优化效果

```python
# 对比测试
import time

# 不使用NUMA
start = time.perf_counter()
for _ in range(100):
    compute_attention(query, key, value)
time_no_numa = time.perf_counter() - start

# 使用NUMA（进程级绑定）
# numactl --cpunodebind=2 python test.py
start = time.perf_counter()
for _ in range(100):
    compute_attention(query, key, value)
time_numa = time.perf_counter() - start

print(f"性能提升: {(time_no_numa/time_numa - 1)*100:.2f}%")
```

### 3. 监控NUMA性能

```bash
# 查看NUMA统计
numastat

# 查看进程NUMA使用
numastat -p <PID>
```

## 常见问题解决

### Q: 优化后性能反而下降？

**可能原因**:
1. 频繁切换CPU亲和性带来开销
2. 工作负载太小，无法体现NUMA优势
3. 系统负载高，NUMA节点被占用

**解决方案**:
- 使用进程级绑定（numactl）而不是代码级绑定
- 只在批量处理时使用
- 检查系统负载，选择空闲的NUMA节点

### Q: 如何确定最优NUMA节点？

```python
# 方法1: 运行完整分析
python analyze_numa_cpu_attention.py

# 方法2: 快速检测
from optimize_gqa_with_numa import get_optimal_numa_node
best_node = get_optimal_numa_node()

# 方法3: 手动测试
for node in range(4):
    numactl --cpunodebind=$node python benchmark.py
```

### Q: 多进程场景如何处理？

```python
import multiprocessing as mp

def worker(numa_node, data):
    # 每个进程绑定到不同NUMA节点
    bind_to_numa_node(numa_node)
    return process(data)

# 启动多个进程
processes = []
for i, numa_node in enumerate([0, 1, 2, 3]):
    p = mp.Process(target=worker, args=(numa_node, data[i]))
    p.start()
    processes.append(p)
```

### Q: 与GPU混合使用？

```bash
# 查看GPU与NUMA节点的关系
nvidia-smi topo -m

# 绑定到与GPU最近的NUMA节点
# 例如GPU在PCIe bus上连接到Node 2
numactl --cpunodebind=2 --membind=2 python script.py
```

## 实际应用示例

### 示例1: 推理服务

```bash
# 启动服务时绑定
numactl --cpunodebind=2 --membind=2 python inference_server.py
```

### 示例2: 批处理脚本

```python
# 在脚本开头
import os
optimal_node = 2  # 根据测试确定
cpus = get_numa_cpus(optimal_node)
os.sched_setaffinity(0, cpus)

# 后续所有计算自动在该节点
```

### 示例3: 多线程并行

```python
from optimize_gqa_with_numa import NUMAThreadPool

# 每个线程分配到不同NUMA节点
pool = NUMAThreadPool(num_threads_per_node=1)
results = pool.parallel_gqa_compute(queries, keys, values)
```

## 总结

1. **优先使用进程级绑定**（numactl），避免代码级切换开销
2. **批量处理场景**更适合NUMA优化
3. **检测最优节点**并固定使用，不要频繁切换
4. **监控性能**，根据实际效果调整策略



