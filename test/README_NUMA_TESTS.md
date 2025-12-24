# NUMA Impact Tests for Einsum Performance

这些测试脚本用于测量 NUMA（Non-Uniform Memory Access）对 einsum 计算性能的影响。

## 文件说明

### 1. `test_numa_quick.py`
快速测试脚本，专注于测量 einsum 计算的 NUMA 开销。

**特点：**
- 简洁高效，适合快速测试
- 测量各个操作的详细时间
- 计算内存带宽和 GFLOPS
- 显示当前 NUMA 配置

**使用方法：**
```bash
# 默认配置
python3 test_numa_quick.py

# 绑定到 NUMA 节点 0
numactl --cpunodebind=0 --membind=0 python3 test_numa_quick.py

# 交错模式
numactl --interleave=all python3 test_numa_quick.py
```

### 2. `test_numa_impact.py`
完整的 NUMA 影响测试套件。

**特点：**
- 多场景测试（默认、节点绑定、交错模式等）
- 测试不同线程数的影响
- 比较连续内存和非连续内存的性能差异
- 生成详细的性能报告

**使用方法：**
```bash
python3 test_numa_impact.py
```

### 3. `run_numa_tests.sh`
自动化测试脚本，使用不同的 NUMA 策略运行测试。

**特点：**
- 自动运行多种 NUMA 配置
- 生成对比报告和 CSV 数据
- 保存详细日志
- 提供优化建议

**使用方法：**
```bash
# 使用默认测试脚本
./run_numa_tests.sh

# 指定测试脚本
./run_numa_tests.sh test_numa_quick.py
```

## NUMA 基础知识

### 什么是 NUMA？

NUMA（Non-Uniform Memory Access）是一种内存架构，其中：
- 每个 CPU 都有本地内存
- 访问本地内存速度快
- 访问远程内存（其他 CPU 的本地内存）速度慢

### NUMA 对性能的影响

1. **本地访问 vs 远程访问**
   - 本地内存访问延迟：~100ns
   - 远程内存访问延迟：~200-300ns
   - 跨 NUMA 带宽可能降低 30-50%

2. **对 einsum 计算的影响**
   - 大规模矩阵运算对内存带宽敏感
   - 跨 NUMA 访问会显著降低性能
   - 合理的 NUMA 配置可提升 20-50% 性能

## 测试场景

### 场景 1: 默认配置（基准）
```bash
python3 test_numa_quick.py
```
系统自动管理 NUMA，作为性能基准。

### 场景 2: 绑定到单个 NUMA 节点
```bash
numactl --cpunodebind=0 --membind=0 python3 test_numa_quick.py
```
**优点：**
- 所有内存访问都是本地的
- 最低延迟，最高带宽
- 适合数据能放入单个 NUMA 节点的情况

**缺点：**
- 只能使用单个 NUMA 节点的内存
- 内存容量受限

### 场景 3: 交错模式（Interleave）
```bash
numactl --interleave=all python3 test_numa_quick.py
```
**优点：**
- 内存在多个 NUMA 节点间均匀分布
- 可以使用全部内存
- 负载均衡

**缺点：**
- 50% 的内存访问是远程的
- 平均延迟较高

### 场景 4: 本地分配（Preferred）
```bash
numactl --localalloc python3 test_numa_quick.py
```
优先在当前 CPU 的本地内存分配，如果不够则使用其他节点。

## 查看 NUMA 信息

```bash
# 查看 NUMA 硬件配置
numactl --hardware

# 查看当前 NUMA 策略
numactl --show

# 查看进程的 NUMA 统计
numastat -c <pid>

# 查看系统 NUMA 统计
numastat
```

## 性能优化建议

### 1. 如果数据能放入单个 NUMA 节点
```bash
# 推荐：绑定到单个节点
numactl --cpunodebind=0 --membind=0 your_program
```

### 2. 如果数据跨多个 NUMA 节点
```bash
# 推荐：使用交错模式
numactl --interleave=all your_program
```

### 3. 对于模型服务器
```bash
# 为每个 NUMA 节点运行一个实例
# 节点 0:
numactl --cpunodebind=0 --membind=0 your_program --port 8000 &

# 节点 1:
numactl --cpunodebind=1 --membind=1 your_program --port 8001 &
```

### 4. 监控 NUMA 性能
```bash
# 实时监控 NUMA 统计
watch -n 1 numastat

# 监控特定进程
watch -n 1 "numastat -c $(pgrep -f your_program)"
```

## 预期结果

### 典型性能差异

根据测试，不同 NUMA 配置的性能差异：

| 配置 | 相对性能 | 适用场景 |
|------|---------|---------|
| 单节点绑定 | 100% (最快) | 数据 < 单节点内存 |
| 本地分配 | 85-95% | 数据略大于单节点 |
| 默认策略 | 70-90% | 一般情况 |
| 交错模式 | 60-80% | 数据 >> 单节点内存 |
| 跨节点访问 | 50-70% (最慢) | 配置不当 |

### 影响因素

1. **数据大小**
   - 小数据：NUMA 影响小（缓存命中）
   - 大数据：NUMA 影响大（内存带宽瓶颈）

2. **访问模式**
   - 顺序访问：NUMA 影响较小
   - 随机访问：NUMA 影响较大

3. **计算强度**
   - 计算密集：NUMA 影响小
   - 内存密集：NUMA 影响大

## 实际应用

### 示例：优化 MoE 模型推理

对于 Mixtral 等 MoE 模型：

```bash
# 1. 测试确定最佳配置
./run_numa_tests.sh test_einsum_experts.py

# 2. 根据结果选择策略
# 如果模型参数 < 单节点内存：
numactl --cpunodebind=0 --membind=0 python generate.py

# 如果模型参数 > 单节点内存：
numactl --interleave=all python generate.py

# 3. 多实例部署
# 为每个 NUMA 节点部署一个实例，避免跨节点访问
```

### 与共享内存的结合

使用 `RestoreExpertsFromSharedMemory` 加载参数时：

```python
# 确保共享内存在正确的 NUMA 节点上分配
# 方法 1: 启动时绑定
numactl --cpunodebind=0 --membind=0 python server.py

# 方法 2: 运行时设置（需要 libnuma）
import ctypes
libnuma = ctypes.CDLL("libnuma.so")
libnuma.numa_set_preferred(0)  # 设置首选节点

# 然后加载参数
experts_state_dict = restore_experts_tensor_from_shared_memory(...)
```

## 故障排查

### 问题 1: numactl 未找到
```bash
sudo apt-get install numactl
```

### 问题 2: 权限不足
某些 NUMA 操作需要 root 权限：
```bash
sudo numactl --cpunodebind=0 --membind=0 python3 your_script.py
```

### 问题 3: 系统不支持 NUMA
检查系统是否支持 NUMA：
```bash
dmesg | grep -i numa
cat /proc/cpuinfo | grep "physical id"
```

### 问题 4: 性能没有提升
可能原因：
1. 数据太小，缓存已足够
2. 系统只有一个 NUMA 节点
3. 计算瓶颈不在内存访问

## 参考资料

- [NUMA Deep Dive](https://www.kernel.org/doc/html/latest/vm/numa.html)
- [numactl man page](https://linux.die.net/man/8/numactl)
- [Understanding NUMA](https://queue.acm.org/detail.cfm?id=2513149)
- [PyTorch NUMA Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#cpu-specific-optimizations)

## 联系与贡献

如有问题或建议，请提交 issue 或 PR。

