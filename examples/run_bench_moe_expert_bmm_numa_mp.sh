#!/usr/bin/env bash
# 使用 fslmp 虚拟环境运行 MoE fused BMM NUMA 多进程基准（与 Readme / run_generate_numa.sh 一致）。
#
# Usage:
#   ./examples/run_bench_moe_expert_bmm_numa_mp.sh
#   ./examples/run_bench_moe_expert_bmm_numa_mp.sh -- --tokens 4096 --iters 10
#
# Env:
#   LMP_PYTHON   解释器路径（默认 /mnt/zhengcf3/lmp_env/fslmp/bin/python）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${LMP_PYTHON:-/mnt/zhengcf3/lmp_env/fslmp/bin/python}"
BENCH="${ROOT}/examples/bench_moe_expert_bmm_numa_mp.py"
exec "${PY}" "${BENCH}" "$@"
