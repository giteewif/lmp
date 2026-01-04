#!/bin/bash

# 调试脚本：运行 test_load.py 并捕获错误

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Debug Script for test_load.py"
echo "=========================================="
echo ""

# 激活 lmp 环境
echo "Activating lmp environment..."
source /mnt/zhengcf3/env/lmp/bin/activate

# 设置环境变量
export PYTHONPATH="/mnt/zhengcf3/lmp/src:/mnt/zhengcf3/lmp/src/sllm_store:$PYTHONPATH"

# 切换到脚本目录
cd /mnt/zhengcf3/lmp/examples

echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"
echo ""

# 检查 Python 版本
echo "Python version:"
python --version
echo ""

# 检查 CUDA 可用性
echo "CUDA availability:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')" 2>&1 || echo "Failed to check CUDA"
echo ""

# 运行测试
echo "=========================================="
echo "Running test_load.py..."
echo "=========================================="
echo ""

python test_load.py 2>&1 | tee test_load_debug.log

echo ""
echo "=========================================="
echo "Test completed. Check test_load_debug.log for details."
echo "=========================================="

