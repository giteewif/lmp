#!/bin/bash

# NUMA 测试演示脚本
# 快速展示 NUMA 对性能的影响

echo "=========================================="
echo "NUMA Performance Impact Demo"
echo "=========================================="
echo ""

# 检查依赖
if ! command -v numactl &> /dev/null; then
    echo "Error: numactl not found"
    echo "Please install: sudo apt-get install numactl"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# 检查测试脚本
if [ ! -f "test_numa_quick.py" ]; then
    echo "Error: test_numa_quick.py not found"
    echo "Please run this script in the test directory"
    exit 1
fi

# 显示 NUMA 配置
echo "System NUMA Configuration:"
echo "------------------------------------------"
numactl --hardware | grep -E "available|node.*cpus|node.*size"
echo ""

# 获取 NUMA 节点数量
NUM_NODES=$(numactl --hardware | grep "available:" | awk '{print $2}')
echo "Number of NUMA nodes: $NUM_NODES"
echo ""

# 快速测试
echo "Running quick comparison test..."
echo "(This will take ~2 minutes)"
echo ""

# Test 1: 默认
echo "Test 1/3: Default policy..."
DEFAULT_TIME=$(python3 -c "
import torch
import torch.nn.functional as F
import time
import numpy as np

torch.set_num_threads(128)
E, T, H, I = 8, 128, 4096, 14336
dtype = torch.bfloat16

# 分配
inputs = torch.randn(E, T, H, dtype=dtype)
w1 = torch.randn(E, I, H, dtype=dtype)
w2 = torch.randn(E, H, I, dtype=dtype)
w3 = torch.randn(E, I, H, dtype=dtype)

# 预热
for _ in range(3):
    o1 = torch.einsum('eth,eih->eti', inputs, w1)
    o1 = F.silu(o1)
    o3 = torch.einsum('eth,eih->eti', inputs, w3)
    o = torch.einsum('eti,ehi->eth', o1 * o3, w2)

# 测试
times = []
for _ in range(10):
    t = time.perf_counter()
    o1 = torch.einsum('eth,eih->eti', inputs, w1)
    o1 = F.silu(o1)
    o3 = torch.einsum('eth,eih->eti', inputs, w3)
    o = torch.einsum('eti,ehi->eth', o1 * o3, w2)
    times.append((time.perf_counter() - t) * 1000)

print(f'{np.mean(times):.2f}')
" 2>/dev/null)
echo "  Time: ${DEFAULT_TIME} ms"

# Test 2: 绑定到 NUMA 节点 0
echo "Test 2/3: Bound to NUMA node 0..."
NODE0_TIME=$(numactl --cpunodebind=0 --membind=0 python3 -c "
import torch
import torch.nn.functional as F
import time
import numpy as np

torch.set_num_threads(128)
E, T, H, I = 8, 128, 4096, 14336
dtype = torch.bfloat16

inputs = torch.randn(E, T, H, dtype=dtype)
w1 = torch.randn(E, I, H, dtype=dtype)
w2 = torch.randn(E, H, I, dtype=dtype)
w3 = torch.randn(E, I, H, dtype=dtype)

for _ in range(3):
    o1 = torch.einsum('eth,eih->eti', inputs, w1)
    o1 = F.silu(o1)
    o3 = torch.einsum('eth,eih->eti', inputs, w3)
    o = torch.einsum('eti,ehi->eth', o1 * o3, w2)

times = []
for _ in range(10):
    t = time.perf_counter()
    o1 = torch.einsum('eth,eih->eti', inputs, w1)
    o1 = F.silu(o1)
    o3 = torch.einsum('eth,eih->eti', inputs, w3)
    o = torch.einsum('eti,ehi->eth', o1 * o3, w2)
    times.append((time.perf_counter() - t) * 1000)

print(f'{np.mean(times):.2f}')
" 2>/dev/null)
echo "  Time: ${NODE0_TIME} ms"

# Test 3: 交错模式
echo "Test 3/3: Interleave mode..."
INTERLEAVE_TIME=$(numactl --interleave=all python3 -c "
import torch
import torch.nn.functional as F
import time
import numpy as np

torch.set_num_threads(128)
E, T, H, I = 8, 128, 4096, 14336
dtype = torch.bfloat16

inputs = torch.randn(E, T, H, dtype=dtype)
w1 = torch.randn(E, I, H, dtype=dtype)
w2 = torch.randn(E, H, I, dtype=dtype)
w3 = torch.randn(E, I, H, dtype=dtype)

for _ in range(3):
    o1 = torch.einsum('eth,eih->eti', inputs, w1)
    o1 = F.silu(o1)
    o3 = torch.einsum('eth,eih->eti', inputs, w3)
    o = torch.einsum('eti,ehi->eth', o1 * o3, w2)

times = []
for _ in range(10):
    t = time.perf_counter()
    o1 = torch.einsum('eth,eih->eti', inputs, w1)
    o1 = F.silu(o1)
    o3 = torch.einsum('eth,eih->eti', inputs, w3)
    o = torch.einsum('eti,ehi->eth', o1 * o3, w2)
    times.append((time.perf_counter() - t) * 1000)

print(f'{np.mean(times):.2f}')
" 2>/dev/null)
echo "  Time: ${INTERLEAVE_TIME} ms"

# 计算提升
echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
printf "%-25s %12s %12s\n" "Configuration" "Time (ms)" "Speedup"
echo "------------------------------------------"

if [ -n "$DEFAULT_TIME" ]; then
    printf "%-25s %12s %12s\n" "Default" "$DEFAULT_TIME" "1.00x"
fi

if [ -n "$NODE0_TIME" ] && [ -n "$DEFAULT_TIME" ]; then
    SPEEDUP=$(echo "scale=2; $DEFAULT_TIME / $NODE0_TIME" | bc)
    IMPROVEMENT=$(echo "scale=1; ($DEFAULT_TIME - $NODE0_TIME) / $DEFAULT_TIME * 100" | bc)
    printf "%-25s %12s %12s (+${IMPROVEMENT}%%)\n" "NUMA Node 0" "$NODE0_TIME" "${SPEEDUP}x"
fi

if [ -n "$INTERLEAVE_TIME" ] && [ -n "$DEFAULT_TIME" ]; then
    SPEEDUP=$(echo "scale=2; $DEFAULT_TIME / $INTERLEAVE_TIME" | bc)
    IMPROVEMENT=$(echo "scale=1; ($DEFAULT_TIME - $INTERLEAVE_TIME) / $DEFAULT_TIME * 100" | bc)
    printf "%-25s %12s %12s (${IMPROVEMENT}%%)\n" "Interleave" "$INTERLEAVE_TIME" "${SPEEDUP}x"
fi

echo ""
echo "=========================================="
echo "Interpretation"
echo "=========================================="

if [ -n "$NODE0_TIME" ] && [ -n "$DEFAULT_TIME" ]; then
    DIFF=$(echo "scale=0; ($DEFAULT_TIME - $NODE0_TIME) / $DEFAULT_TIME * 100" | bc)
    
    if [ "$DIFF" -gt 20 ]; then
        echo "✓ NUMA has SIGNIFICANT impact on your system"
        echo "  Recommendation: Use NUMA binding for production"
        echo "  Command: numactl --cpunodebind=0 --membind=0 your_program"
    elif [ "$DIFF" -gt 10 ]; then
        echo "✓ NUMA has MODERATE impact on your system"
        echo "  Recommendation: Consider NUMA binding for performance"
    else
        echo "✓ NUMA has MINIMAL impact on your system"
        echo "  Your system may have:"
        echo "    - Single NUMA node"
        echo "    - Fast interconnect between nodes"
        echo "    - Data fits in cache"
    fi
fi

echo ""
echo "For detailed analysis, run:"
echo "  ./run_numa_tests.sh test_numa_quick.py"
echo ""
echo "Or read: README_NUMA_TESTS.md"

