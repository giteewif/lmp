#!/usr/bin/env python3
"""
验证：多进程下 CPU 计算是否真正并行（墙钟时间 ≈ max(各进程耗时)，而非求和）。

用法（建议与 numactl 无关，进程内用 sched_setaffinity 绑核；失败则仍跑并行逻辑）:
  /path/to/python examples/validate_mp_cpu_parallel.py
  WORKERS=4 MAT_N=1024 ITERS=80 /path/to/python examples/validate_mp_cpu_parallel.py

可选：按 NUMA 拆 CPU 列表（逗号分隔的区间，管道分隔进程）:
  CPU_SETS='0-11|12-23|24-35|36-47' WORKERS=4 python examples/validate_mp_cpu_parallel.py
未设置 CPU_SETS 时，不绑核，仅验证多进程并行度。

按 NUMA：每个 node 一个进程（从 /sys/.../node*/cpulist 读 CPU，与 numactl --hardware 一致）:
  python validate_mp_cpu_parallel.py --gemma-moe --numa-one-per-node \\
    --config /path/to/Gemma/config.json --batch 64 --seq-len 128 --iters 3
  # 只跑部分 node: --numa-nodes 0,2
  # 多进程后再跑单 worker 全量 token 对比: 加 --also-single-worker [--single-worker-numa-node 0]

Gemma（text_config）MoE 专家 CPU 并行（bf16，仅专家 MLP 形状，不含 attention/vision）:
  python validate_mp_cpu_parallel.py --gemma-moe \\
    --config /path/to/Gemma/config.json --batch 64 --seq-len 128 --workers 4 --iters 3
  # 专家维不堆在 bmm batch 上：bmm 只在 token 行分块（--token-bmm-chunks 0=自动≈线程数）

环境变量 MP_START_METHOD:
  - Linux 默认 ``fork``（父进程已 import torch，子进程并行墙钟≈最慢子进程纯算时间）。
  - 设为 ``spawn`` 可复现「每进程冷启动」下的较大 parallel wall（不推荐用于本验证）。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _parse_cpu_list(spec: str) -> List[int]:
    """'0-3,8,10-12' -> [0,1,2,3,8,10,11,12]"""
    out: List[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_cpu_sets(env: str) -> List[List[int]]:
    """'0-11|12-23' -> [[0..11],[12..23]]"""
    return [_parse_cpu_list(s) for s in env.split("|") if s.strip()]


def _discover_numa_cpu_sets(
    only_nodes: Optional[set[int]] = None,
) -> List[Tuple[int, List[int]]]:
    """
    Return [(node_id, cpus), ...] sorted by node_id from sysfs cpulist.
    `only_nodes` if set: keep only these node ids.
    """
    base = Path("/sys/devices/system/node")
    if not base.is_dir():
        raise RuntimeError("NUMA sysfs not found (/sys/devices/system/node).")

    out: List[Tuple[int, List[int]]] = []
    for cpulist_path in sorted(base.glob("node*/cpulist")):
        node_dir = cpulist_path.parent.name
        if not node_dir.startswith("node"):
            continue
        node_id = int(node_dir[len("node") :])
        if only_nodes is not None and node_id not in only_nodes:
            continue
        text = cpulist_path.read_text(encoding="utf-8").strip()
        cpus = _parse_cpu_list(text)
        if not cpus:
            continue
        out.append((node_id, cpus))
    if not out:
        raise RuntimeError("No NUMA node cpulist found under /sys/devices/system/node.")
    return out


def _cpus_for_numa_node(node_id: int) -> List[int]:
    pairs = _discover_numa_cpu_sets(only_nodes={int(node_id)})
    if not pairs:
        raise SystemExit(f"NUMA node {node_id} not found or has no cpulist.")
    return pairs[0][1]


def _numa_sets_to_worker_cpu_lists(
    node_cpu_pairs: List[Tuple[int, List[int]]],
) -> Tuple[List[int], List[List[int]]]:
    """(node_ids in worker order, cpu_sets per worker)."""
    node_ids = [nid for nid, _ in node_cpu_pairs]
    cpu_sets = [cpus for _, cpus in node_cpu_pairs]
    return node_ids, cpu_sets


def _parse_numa_nodes_filter(spec: Optional[str]) -> Optional[set[int]]:
    if not spec or not str(spec).strip():
        return None
    out: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        out.add(int(part))
    return out


def _resolve_parallel_cpu_binding(
    *,
    numa_one_per_node: bool,
    numa_nodes: Optional[str],
    cpu_sets_env: str,
    explicit_workers: Optional[int],
    env_workers_default: int,
) -> Tuple[int, Optional[List[List[int]]], Optional[List[int]]]:
    """
    Returns (workers, cpu_sets_or_none, numa_node_id_per_worker_or_none).

    cpu_sets: one list of CPU ids per worker for sched_setaffinity (same as CPU_SETS).
    """
    if numa_one_per_node and cpu_sets_env.strip():
        raise SystemExit("不要同时使用 --numa-one-per-node 与 CPU_SETS；请二选一。")

    if numa_one_per_node:
        only = _parse_numa_nodes_filter(numa_nodes)
        pairs = _discover_numa_cpu_sets(only_nodes=only)
        numa_ids, cpu_sets = _numa_sets_to_worker_cpu_lists(pairs)
        workers = len(cpu_sets)
        if explicit_workers is not None and explicit_workers != workers:
            raise SystemExit(
                f"--numa-one-per-node 时 worker 数必须等于 NUMA node 数 ({workers})，"
                f"与 --workers={explicit_workers} 冲突"
            )
        return workers, cpu_sets, numa_ids

    cpu_sets: Optional[List[List[int]]] = None
    if cpu_sets_env.strip():
        cpu_sets = _parse_cpu_sets(cpu_sets_env)
        workers = explicit_workers if explicit_workers is not None else env_workers_default
        if len(cpu_sets) != workers:
            raise SystemExit(
                f"CPU_SETS 管道段数 ({len(cpu_sets)}) 必须等于 workers ({workers})"
            )
        return workers, cpu_sets, None

    workers = explicit_workers if explicit_workers is not None else env_workers_default
    return workers, None, None


@dataclass
class WorkerArgs:
    rank: int
    cpus: Optional[Sequence[int]]
    mat_n: int
    iters: int


@dataclass
class GemmaMoeWorkerArgs:
    rank: int
    cpus: Optional[Sequence[int]]
    token_rows: int
    hidden: int
    moe_int: int
    top_k: int
    iters: int
    # bmm 在 token 行上分块：batch 维 = token 块数；0 表示按线程数自动取 min(token_rows, threads)
    token_bmm_chunks: int = 0


def _effective_token_bmm_chunks(token_rows: int, requested: int, num_threads: int) -> int:
    """bmm 的 batch 维 = token 分块数 B；每块约 ceil(rows/B) 行。"""
    if token_rows < 1:
        return 1
    if requested > 0:
        return min(token_rows, max(1, requested))
    return min(token_rows, max(1, num_threads))


def _moe_block_token_bmm(
    x,
    wg,
    wu,
    wd,
    *,
    num_experts: int,
    token_bmm_chunks: int,
    num_threads: int,
):
    """
    MoE 一层：专家维循环；每个专家内用 bmm 在 **token 行** 上分块计算（(B, chunk, h) @ (B, h, m)）。
    """
    import torch
    import torch.nn.functional as F

    rows_orig, h = x.shape
    k = num_experts
    m = wg.shape[2]
    B = _effective_token_bmm_chunks(rows_orig, token_bmm_chunks, num_threads)
    chunk = (rows_orig + B - 1) // B
    pad_r = B * chunk - rows_orig
    if pad_r:
        x_work = F.pad(x, (0, 0, 0, pad_r))
    else:
        x_work = x
    xb = x_work.view(B, chunk, h)

    y_acc = torch.zeros(B * chunk, h, dtype=x.dtype, device=x.device)
    for ei in range(k):
        Wg = wg[ei].unsqueeze(0).expand(B, h, m).contiguous()
        Wu = wu[ei].unsqueeze(0).expand(B, h, m).contiguous()
        Wd = wd[ei].unsqueeze(0).expand(B, m, h).contiguous()
        gate = torch.bmm(xb, Wg)
        up = torch.bmm(xb, Wu)
        act = F.gelu(gate, approximate="tanh") * up
        y_acc = y_acc + torch.bmm(act, Wd).reshape(B * chunk, h)
    return y_acc[:rows_orig]


def _load_gemma_text_moe_config(config_path: str) -> Tuple[int, int, int]:
    """(hidden_size, moe_intermediate_size, top_k_experts) from text_config only."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config") or {}
    hidden = int(tc["hidden_size"])
    moe_int = int(tc.get("moe_intermediate_size") or tc["intermediate_size"])
    top_k = int(tc["top_k_experts"])
    return hidden, moe_int, top_k


def _worker_gemma_moe(args: GemmaMoeWorkerArgs) -> Tuple[int, float, Optional[str]]:
    """
    One MoE block (sum over top_k experts) on a CPU slice of tokens [token_rows, hidden], bf16.
    bmm 在 **token 行** 上分块（batch=token 块）；专家维 k 用循环，不把 expert 堆在 bmm batch 维上。
    SwiGLU-style: gelu_pytorch_tanh(gate) * up, then down.
    """
    aff_err = None
    if args.cpus:
        try:
            os.sched_setaffinity(0, set(args.cpus))
        except OSError as e:
            aff_err = str(e)

    import torch

    if args.cpus:
        nthreads = max(1, len(args.cpus))
    else:
        nthreads = max(1, int(os.environ.get("OMP_NUM_THREADS", str(os.cpu_count() or 1))))
    torch.set_num_threads(nthreads)

    rows = args.token_rows
    h = args.hidden
    m = args.moe_int
    k = args.top_k

    g = torch.Generator()
    g.manual_seed(42_000 + args.rank)
    x = torch.randn(rows, h, dtype=torch.bfloat16, generator=g)
    # Per-expert weights (same layout as typical gate_proj / up_proj / down_proj)
    wg = torch.randn(k, h, m, dtype=torch.bfloat16, generator=g)
    wu = torch.randn(k, h, m, dtype=torch.bfloat16, generator=g)
    wd = torch.randn(k, m, h, dtype=torch.bfloat16, generator=g)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        x = _moe_block_token_bmm(
            x,
            wg,
            wu,
            wd,
            num_experts=k,
            token_bmm_chunks=args.token_bmm_chunks,
            num_threads=nthreads,
        )
    t1 = time.perf_counter()
    _ = float(x[0, 0].item())
    return args.rank, t1 - t0, aff_err


def _worker_run(args: WorkerArgs) -> Tuple[int, float, Optional[str]]:
    """返回 (rank, 本进程计算秒数, affinity_err_msg)。"""
    aff_err = None
    if args.cpus:
        try:
            os.sched_setaffinity(0, set(args.cpus))
        except OSError as e:
            aff_err = str(e)

    import torch

    if args.cpus:
        torch.set_num_threads(max(1, len(args.cpus)))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    n = args.mat_n
    # 每进程独立张量，避免 false sharing；bf16 贴近 MoE 测试
    g = torch.Generator()
    g.manual_seed(12345 + args.rank)
    a = torch.randn(n, n, dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, n, dtype=torch.bfloat16, generator=g)

    t0 = time.perf_counter()
    c = a
    for _ in range(args.iters):
        c = c @ b
    t1 = time.perf_counter()
    # 防止被 DCE
    _ = float(c[0, 0].item())
    return args.rank, t1 - t0, aff_err


def _worker_entry(wa: WorkerArgs, q: mp.Queue) -> None:
    r = _worker_run(wa)
    q.put(r)


def _worker_entry_gemma(wa: GemmaMoeWorkerArgs, q: mp.Queue) -> None:
    r = _worker_gemma_moe(wa)
    q.put(r)


def _token_rows_for_rank(total_rows: int, workers: int, rank: int) -> int:
    return total_rows // workers + (1 if rank < total_rows % workers else 0)


def main_matmul_legacy(
    *,
    workers: int,
    mat_n: int,
    iters: int,
    cpu_sets: Optional[List[List[int]]],
    numa_node_per_worker: Optional[List[int]],
    mp_method: Optional[str],
) -> None:
    # 父进程先加载 torch，Linux 下用 fork 时子进程不再承担完整 torch 冷启动，并行墙钟才接近 max(纯算)。
    import torch  # noqa: F401

    # MP_START_METHOD=spawn|fork（默认 Linux 用 fork 做本验证；CUDA 多进程请用 spawn）
    _meth = mp_method or os.environ.get(
        "MP_START_METHOD", "fork" if sys.platform == "linux" else "spawn"
    )
    ctx = mp.get_context(_meth)
    worker_args: List[WorkerArgs] = [
        WorkerArgs(
            rank=i,
            cpus=cpu_sets[i] if cpu_sets else None,
            mat_n=mat_n,
            iters=iters,
        )
        for i in range(workers)
    ]

    # --- 并行：同时启动，再 join ---
    t_wall0 = time.perf_counter()
    procs: List[mp.Process] = []
    queues: List[mp.Queue] = []
    for wa in worker_args:
        q = ctx.Queue()
        p = ctx.Process(target=_worker_entry, args=(wa, q))
        queues.append(q)
        procs.append(p)
        p.start()
    results: List[Tuple[int, float, Optional[str]]] = []
    for p, q in zip(procs, queues):
        p.join()
        results.append(q.get())
    t_wall_parallel = time.perf_counter() - t_wall0

    per_rank = {r[0]: r[1] for r in results}
    max_child = max(per_rank.values())
    sum_child = sum(per_rank.values())

    # --- 串行：子进程一个接一个 ---
    t_wall1 = time.perf_counter()
    for wa in worker_args:
        q = ctx.Queue()
        p = ctx.Process(target=_worker_entry, args=(wa, q))
        p.start()
        p.join()
        q.get()
    t_wall_serial = time.perf_counter() - t_wall1

    print("=== validate_mp_cpu_parallel ===")
    print(f"WORKERS={workers} MAT_N={mat_n} ITERS={iters} mp_context={_meth}")
    if numa_node_per_worker is not None:
        for i, nid in enumerate(numa_node_per_worker):
            print(f"  worker{i} NUMA node={nid}")
    if cpu_sets:
        for i, cpus in enumerate(cpu_sets):
            print(f"  worker{i} cpus: {cpus[0]}..{cpus[-1]} ({len(cpus)} cpus)")
    for r, sec, err in sorted(results, key=lambda x: x[0]):
        extra = f" affinity_err={err}" if err else ""
        print(f"  worker{r} compute_time={sec*1e3:.2f} ms{extra}")

    print(f"parallel wall (all start->all join): {t_wall_parallel*1e3:.2f} ms")
    print(f"serial wall   (one after another):   {t_wall_serial*1e3:.2f} ms")
    print(f"sum of per-worker compute times:   {sum_child*1e3:.2f} ms (ideal upper bound for parallel wall if perfectly overlapped)")
    print(f"max per-worker compute time:       {max_child*1e3:.2f} ms (ideal lower bound for parallel wall)")

    # 判据：同一总工作量下，并行墙钟应明显小于「子进程一个接一个」的墙钟。
    if t_wall_parallel + 1e-9 < 0.85 * t_wall_serial:
        print("结论: 并行墙钟明显低于串行墙钟 —— 多进程在时间上重叠执行（真并行）。")
    elif t_wall_parallel < t_wall_serial:
        print("结论: 并行墙钟低于串行墙钟 —— 有并行重叠，可再结合 top 判断是否吃满预期核。")
    else:
        print(
            "结论: 并行墙钟未低于串行，可能 CPU 不足/被抢占/绑核失败；请检查 top 与 CPU_SETS。"
        )

    ratio = t_wall_parallel / max(max_child, 1e-9)
    print(
        f"parallel_wall / max_child_compute ≈ {ratio:.2f} "
        f"(fork 且父已 import torch 时应接近 1；spawn 时因每进程冷启动会远大于 1)"
    )


def main_gemma_moe(
    *,
    config_path: str,
    batch: int,
    seq_len: int,
    workers: int,
    iters: int,
    mp_method: Optional[str],
    cpu_sets: Optional[List[List[int]]],
    numa_node_per_worker: Optional[List[int]],
    also_single_worker: bool,
    single_worker_numa_node: Optional[int],
    token_bmm_chunks: int,
) -> None:
    import torch  # noqa: F401

    hidden, moe_int, top_k = _load_gemma_text_moe_config(config_path)
    total_rows = batch * seq_len

    _meth = mp_method or os.environ.get(
        "MP_START_METHOD", "fork" if sys.platform == "linux" else "spawn"
    )
    ctx = mp.get_context(_meth)

    worker_args: List[GemmaMoeWorkerArgs] = []
    for i in range(workers):
        rows = _token_rows_for_rank(total_rows, workers, i)
        worker_args.append(
            GemmaMoeWorkerArgs(
                rank=i,
                cpus=cpu_sets[i] if cpu_sets else None,
                token_rows=rows,
                hidden=hidden,
                moe_int=moe_int,
                top_k=top_k,
                iters=iters,
                token_bmm_chunks=token_bmm_chunks,
            )
        )

    wa0 = worker_args[0]
    nt0 = len(cpu_sets[0]) if cpu_sets and cpu_sets[0] else max(1, os.cpu_count() or 1)
    B0 = _effective_token_bmm_chunks(wa0.token_rows, token_bmm_chunks, nt0)

    t_wall0 = time.perf_counter()
    procs: List[mp.Process] = []
    queues: List[mp.Queue] = []
    for wa in worker_args:
        q = ctx.Queue()
        p = ctx.Process(target=_worker_entry_gemma, args=(wa, q))
        queues.append(q)
        procs.append(p)
        p.start()
    results: List[Tuple[int, float, Optional[str]]] = []
    for p, q in zip(procs, queues):
        p.join()
        results.append(q.get())
    t_wall_parallel = time.perf_counter() - t_wall0

    per_rank = {r[0]: r[1] for r in results}
    max_child = max(per_rank.values())
    sum_child = sum(per_rank.values())

    t_wall1 = time.perf_counter()
    for wa in worker_args:
        q = ctx.Queue()
        p = ctx.Process(target=_worker_entry_gemma, args=(wa, q))
        p.start()
        p.join()
        q.get()
    t_wall_serial = time.perf_counter() - t_wall1

    print("=== validate_mp_cpu_parallel (Gemma text MoE experts, bf16) ===")
    print(
        f"config={config_path} batch={batch} seq_len={seq_len} "
        f"token_rows={total_rows} hidden={hidden} moe_int={moe_int} top_k={top_k}"
    )
    print(
        f"WORKERS={workers} ITERS={iters} mp_context={_meth} dtype=bfloat16 "
        f"| bmm 按 token 分块: --token-bmm-chunks={token_bmm_chunks} "
        f"(worker0 上自动 B={B0}, threads≈{nt0})"
    )
    if numa_node_per_worker is not None:
        for i, nid in enumerate(numa_node_per_worker):
            print(f"  worker{i} NUMA node={nid}")
    if cpu_sets:
        for i, cpus in enumerate(cpu_sets):
            print(f"  worker{i} cpus: {cpus[0]}..{cpus[-1]} ({len(cpus)} cpus)")
    for wa, (r, sec, err) in zip(worker_args, sorted(results, key=lambda x: x[0])):
        extra = f" affinity_err={err}" if err else ""
        print(
            f"  worker{r} token_rows={wa.token_rows} compute_time={sec*1e3:.2f} ms{extra}"
        )

    # compute_time 仅统计子进程内 MoE（token 分块 bmm + gelu），不含 fork/join/Queue。
    print(
        "专家计算(仅计时区内): "
        f"max={max_child*1e3:.2f} ms（多进程并行时完成全部片的理想墙钟下界≈最慢 worker）; "
        f"sum={sum_child*1e3:.2f} ms（各进程算时相加≈总 CPU 时间，非墙钟）"
    )

    print(f"parallel wall (all start->all join): {t_wall_parallel*1e3:.2f} ms")
    print(f"serial wall   (one after another):   {t_wall_serial*1e3:.2f} ms")
    print(f"sum of per-worker compute times:   {sum_child*1e3:.2f} ms")
    print(f"max per-worker compute time:       {max_child*1e3:.2f} ms")

    if t_wall_parallel + 1e-9 < 0.85 * t_wall_serial:
        print("结论: 并行墙钟明显低于串行墙钟 —— 多进程在时间上重叠执行（真并行）。")
    elif t_wall_parallel < t_wall_serial:
        print("结论: 并行墙钟低于串行墙钟 —— 有并行重叠。")
    else:
        print(
            "结论: 并行墙钟未低于串行，可能 CPU 不足/被抢占/绑核失败；请检查 top 与 CPU_SETS。"
        )

    ratio = t_wall_parallel / max(max_child, 1e-9)
    print(f"parallel_wall / max_child_compute ≈ {ratio:.2f}")

    if also_single_worker:
        cpus_single: Optional[List[int]] = None
        bind_note = "不绑核"
        if single_worker_numa_node is not None:
            cpus_single = _cpus_for_numa_node(single_worker_numa_node)
            bind_note = f"绑 NUMA node={single_worker_numa_node} ({len(cpus_single)} cpus)"
        wa1 = GemmaMoeWorkerArgs(
            rank=0,
            cpus=cpus_single,
            token_rows=total_rows,
            hidden=hidden,
            moe_int=moe_int,
            top_k=top_k,
            iters=iters,
            token_bmm_chunks=token_bmm_chunks,
        )
        print()
        print("=== single-worker baseline（同一 token_rows 总量、同一 ITERS、token 分块 bmm）===")
        print(f"{bind_note}")
        q1 = ctx.Queue()
        t_s0 = time.perf_counter()
        p1 = ctx.Process(target=_worker_entry_gemma, args=(wa1, q1))
        p1.start()
        p1.join()
        r0, sec1, err1 = q1.get()
        t_s1 = time.perf_counter()
        extra = f" affinity_err={err1}" if err1 else ""
        print(
            f"  worker0 token_rows={total_rows} compute_time={sec1*1e3:.2f} ms（仅专家计时区）{extra}"
        )
        print(f"  single parallel wall (1 process start->join): {(t_s1 - t_s0)*1e3:.2f} ms")
        per_token_us = (sec1 * 1e6) / max(total_rows * iters, 1)
        print(
            f"  粗略: 每 token×每轮 MoE 块专家计时 ≈ {per_token_us:.3f} µs "
            f"(= compute_time / (token_rows×ITERS)))"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate multiprocess CPU parallelism.")
    ap.add_argument(
        "--gemma-moe",
        action="store_true",
        help="Gemma text_config MoE expert MLP only (bf16), split batch*seq rows across workers.",
    )
    ap.add_argument("--config", type=str, default=None, help="Path to Gemma config.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count. With --numa-one-per-node, must match NUMA node count (auto).",
    )
    ap.add_argument("--iters", type=int, default=3, help="Repeat full MoE block (sum top_k experts) this many times.")
    ap.add_argument(
        "--mp-start-method",
        type=str,
        default=None,
        choices=("fork", "spawn"),
        help="Override MP_START_METHOD for this run.",
    )
    ap.add_argument(
        "--numa-one-per-node",
        action="store_true",
        help="One worker per NUMA node; CPU affinity from /sys/devices/system/node/node*/cpulist.",
    )
    ap.add_argument(
        "--numa-nodes",
        type=str,
        default=None,
        help="Comma-separated NUMA node ids to use with --numa-one-per-node (default: all nodes).",
    )
    ap.add_argument(
        "--also-single-worker",
        action="store_true",
        help="After multi-worker Gemma run, also run 1 worker on full batch*seq (same ITERS).",
    )
    ap.add_argument(
        "--single-worker-numa-node",
        type=int,
        default=None,
        metavar="N",
        help="With --also-single-worker: bind that process to NUMA node N's cpulist (optional).",
    )
    ap.add_argument(
        "--token-bmm-chunks",
        type=int,
        default=0,
        metavar="B",
        help="Token-axis bmm batch count (0=auto min(token_rows, torch threads)); experts not batched on bmm dim.",
    )
    args = ap.parse_args()

    cpu_sets_env = os.environ.get("CPU_SETS", "").strip()
    env_workers = int(os.environ.get("WORKERS", "4"))
    explicit_workers = args.workers if args.workers is not None else None

    workers, cpu_sets, numa_ids = _resolve_parallel_cpu_binding(
        numa_one_per_node=args.numa_one_per_node,
        numa_nodes=args.numa_nodes,
        cpu_sets_env=cpu_sets_env,
        explicit_workers=explicit_workers,
        env_workers_default=env_workers,
    )

    if args.gemma_moe:
        if not args.config:
            raise SystemExit("--config is required with --gemma-moe")
        main_gemma_moe(
            config_path=args.config,
            batch=args.batch,
            seq_len=args.seq_len,
            workers=workers,
            iters=args.iters,
            mp_method=args.mp_start_method,
            cpu_sets=cpu_sets,
            numa_node_per_worker=numa_ids,
            also_single_worker=args.also_single_worker,
            single_worker_numa_node=args.single_worker_numa_node,
            token_bmm_chunks=args.token_bmm_chunks,
        )
        return

    mat_n = int(os.environ.get("MAT_N", "1024"))
    iters = int(os.environ.get("ITERS", "60"))
    main_matmul_legacy(
        workers=workers,
        mat_n=mat_n,
        iters=iters,
        cpu_sets=cpu_sets,
        numa_node_per_worker=numa_ids,
        mp_method=args.mp_start_method,
    )


if __name__ == "__main__":
    main()
