#!/usr/bin/env bash
# =============================================================================
# NUMA 下跑 examples/generate.py，并对比配置（需 numactl）。
#
# SLLM 张量地址从哪来？
# - 权重在 **POSIX 共享内存**（/dev/shm 下形如 /dev/shm/<name> 的 shm 对象）里；
#   Python 侧通过 `get_model_shared_memory_names` 拿到 chunk 名，C++ 里 `shm_open` + `mmap(MAP_SHARED)`
#   映射到本进程虚拟地址（见 `sllm_store/csrc/checkpoint/checkpoint.cpp`）。
# - **物理页**属于该 shm 对象；内核在 **首次缺页/写入（first-touch）** 时把页落到某个 NUMA node，
#   通常由 **写入权重的 sllm-store / server 进程当时跑在哪个 node** 决定。仓库里未见 `mbind`/`numa_*`
#   调用，因此没有强制「权重在 node K」的 API。
# - 客户端 mmap 后读到的 **PFN 与 server 填 shm 时相同**；若 **算子跑在别的 node**，就是 **远端访存**，
#   一般会慢于「CPU 与 shm 物理页同 node」。
#
# 怎样测「更优」？
# - 用 `NUMA_MODE=scan` 跑多组，看日志里 `fused_group_bmm` / `total`。
# - 若 server 在 node0 填的权重：客户端用 `numactl --cpunodebind=0 --membind=0` 往往最好。
# - 确认 server 实际 node：`numastat -p <server_pid>` 或看首次加载权重时的 numactl 策略。
#
# Usage:
#   ./examples/run_generate_numa.sh
#   NUMA_MODE=local ./examples/run_generate_numa.sh
#   NUMA_MODE=scan OMP_NUM_THREADS=24 ./examples/run_generate_numa.sh
#   NUMA_MODE=nodes ./examples/run_generate_numa.sh   # node 0..3 各跑一遍 generate.py（比性能）
#
# Env:
#   LMP_PYTHON   Python 解释器（默认 fslmp）
#   OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS
#   LMP_MP_CPU_INTRAOP_THREADS  CPU experts 子进程里 torch.set_num_threads（优先于 OMP_NUM_THREADS）；
#                               不设时默认 min(32, cpu_count)，避免子进程里 64 线程抢 BMM。
#   LMP_MP_SELFTEST_DIAG        可选：把子进程自检 print 追加写入该路径（与 ``>log`` 互补，便于排查缺行）。
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${LMP_PYTHON:-/mnt/zhengcf3/lmp_env/fslmp/bin/python}"
GEN="${ROOT}/examples/generate.py"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export LMP_MP_CPU_INTRAOP_THREADS="${LMP_MP_CPU_INTRAOP_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
# 子进程 print 与主进程一致尽快落盘；自检还可追加到 LMP_MP_SELFTEST_DIAG（见 cpu_thread_manager_mp._mp_worker_diag_line）
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

extract_bmm_lines() {
  grep -E 'fused_group_bmm:|test_group_bmm_fused_experts\]\[half=.*\] total:' || true
}

run_one() {
  local label="$1"
  shift
  echo "======== ${label} ========"
  local log
  log="$(mktemp)"
  set +e
  "$@" "$PY" "$GEN" >"$log" 2>&1
  local rc=$?
  set -e
  tail -8 "$log"
  echo "--- metrics (grep) ---"
  extract_bmm_lines <"$log" || true
  rm -f "$log"
  if [[ "$rc" != 0 ]]; then
    echo "(exit code $rc)"
    return "$rc"
  fi
  echo
}

MODE="${NUMA_MODE:-all}"
cd "$ROOT"

if [[ "$MODE" == "nodes" ]]; then
  echo ">>> Per-NUMA-node: cpunodebind=N membind=N (each run reloads model)"
  max_node="$(numactl --hardware | sed -n 's/^available: \([0-9]*\) nodes.*/\1/p')"
  max_node=$((max_node - 1))
  for n in $(seq 0 "${max_node}"); do
    run_one "NUMA node ${n}: numactl --cpunodebind=${n} --membind=${n}" \
      numactl --cpunodebind="${n}" --membind="${n}" || true
  done
  exit 0
fi

if [[ "$MODE" == "scan" ]]; then
  echo ">>> NUMA scan (each run reloads model — may take minutes total)"
  run_one "A: cpunodebind=0 membind=0 (local DRAM node0)" numactl --cpunodebind=0 --membind=0
  run_one "B: cpunodebind=1 membind=1 (local DRAM node1)" numactl --cpunodebind=1 --membind=1
  run_one "C: interleave=all (DRAM striped)" numactl --interleave=all
  run_one "D: taskset 0-23 + membind=0 (node0 physical cores, mem local)" \
    taskset -c 0-23 numactl --membind=0
  exit 0
fi

if [[ "$MODE" == "local" || "$MODE" == "all" ]]; then
  run_one "numactl --cpunodebind=0 --membind=0" numactl --cpunodebind=0 --membind=0
fi

if [[ "$MODE" == "interleave" || "$MODE" == "all" ]]; then
  run_one "numactl --interleave=all" numactl --interleave=all
fi

if [[ "$MODE" == "bind0_interleave_mem" ]]; then
  run_one "numactl --cpunodebind=0 --interleave=all" numactl --cpunodebind=0 --interleave=all
fi
