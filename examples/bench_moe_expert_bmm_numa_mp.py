#!/usr/bin/env python3
"""
多进程 + NUMA CPU 亲和性：将 fused MoE 专家沿 E 维切分到 4 个守护进程并行做 BMM，
与单进程整批 BMM 对比耗时（逻辑与 cuda_memory_view._test_group_bmm_fused_experts_impl 一致）。

依赖 Linux fork：子进程继承父进程已构造的 CPU 张量，避免通过 Queue 整表 pickle。
若本机 NUMA 节点少于 4，则将所有可用 CPU 均分为 4 组亲和掩码。

推荐使用仓库约定的 **fslmp** 解释器（与 ``Readme.md`` / ``run_generate_numa.sh`` 一致）::

    source /mnt/zhengcf3/lmp_env/fslmp/bin/activate
    python examples/bench_moe_expert_bmm_numa_mp.py --warmup 2 --iters 5

或::

    LMP_PYTHON=/mnt/zhengcf3/lmp_env/fslmp/bin/python ./examples/run_bench_moe_expert_bmm_numa_mp.sh

可选：限制单进程/子进程内 OpenMP 线程数，避免过订阅::

    python examples/bench_moe_expert_bmm_numa_mp.py --serial-threads 1 --worker-threads 1
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from multiprocessing import get_context
from typing import Any

import torch
import torch.nn.functional as F

# ---- path：与 generate.py 一致（可选使用 transformers ACT2FN）----
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from transformers.activations import ACT2FN
except ImportError:  # 最小环境仅有 torch 时
    ACT2FN = None


def _batched_pad_inputs_presorted(
    x_sorted: torch.Tensor,
    expert_ids_sorted: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """与 ``MLPModuleWrapper._batched_pad_inputs_presorted`` 相同逻辑（避免强依赖整包 models）。"""
    counts = torch.bincount(expert_ids_sorted, minlength=num_experts)
    max_tokens = int(counts.max().item()) if counts.numel() else 0
    e = num_experts
    h = x_sorted.size(1)
    stacked_inputs = torch.zeros((e, max_tokens, h), device=x_sorted.device, dtype=x_sorted.dtype)
    start = 0
    for expert_idx in range(e):
        c = int(counts[expert_idx].item())
        if c:
            stacked_inputs[expert_idx, :c].copy_(x_sorted[start : start + c], non_blocking=False)
        start += c
    return stacked_inputs, counts


def _batched_unpad_outputs(y_stacked: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """与 ``MLPModuleWrapper._batched_unpad_outputs`` 相同。"""
    outs: list[torch.Tensor] = []
    e = int(counts.numel())
    for expert_idx in range(e):
        c = int(counts[expert_idx].item())
        if c:
            outs.append(y_stacked[expert_idx, :c])
    if not outs:
        return y_stacked.reshape(0, y_stacked.size(-1))
    return torch.cat(outs, dim=0)


def _resolve_act_fn(name: str):
    if ACT2FN is not None and name in ACT2FN:
        fn = ACT2FN[name]
        if isinstance(fn, torch.nn.Module):
            return lambda x: fn.forward(x)
        return fn
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    raise RuntimeError(
        f"激活 {name!r} 需要安装 transformers（activations.ACT2FN），或改用 silu/gelu。"
    )


def _parse_cpulist(s: str) -> list[int]:
    """Parse sysfs cpulist like '0-3,8,10-15'."""
    out: list[int] = []
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def discover_numa_cpu_groups(num_groups: int = 4) -> list[list[int]]:
    """
    返回 ``num_groups`` 组 CPU 列表，尽量按 NUMA node 划分；节点不足时按 CPU 序号均分。
    """
    node_paths = sorted(glob.glob("/sys/devices/system/node/node*/cpulist"))
    per_node: list[list[int]] = []
    for p in node_paths:
        try:
            with open(p, encoding="utf-8") as f:
                per_node.append(_parse_cpulist(f.read()))
        except OSError:
            continue
    per_node = [cpus for cpus in per_node if cpus]

    if len(per_node) >= num_groups:
        return [per_node[i] for i in range(num_groups)]

    if per_node:
        flat = sorted({c for cpus in per_node for c in cpus})
    else:
        n = os.cpu_count() or 1
        flat = list(range(n))

    if len(flat) < num_groups:
        # 极端小机器：允许重复绑核
        return [flat for _ in range(num_groups)]

    chunk = int(math.ceil(len(flat) / num_groups))
    groups: list[list[int]] = []
    for g in range(num_groups):
        lo = g * chunk
        hi = min(lo + chunk, len(flat))
        groups.append(flat[lo:hi] if lo < hi else flat[-1:])
    return groups


# fork 子进程通过模块级 dict 读取大张量（避免 pickle）
_MP_SHARED: dict = {}


def _fused_group_bmm_segment(
    stacked_e: torch.Tensor,
    counts_e: torch.Tensor,
    gu_eh2i: torch.Tensor,
    dn_eih: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    """对 expert 子块 ``[Ei, Tmax, H]`` 做两段 bmm + SwiGLU，返回 ``[Ei, Tmax, H]`` stacked 输出。"""
    gu = torch.bmm(stacked_e, gu_eh2i)
    half = gu.size(-1) // 2
    g, u = gu.split(half, dim=-1)
    mid = act_fn(g) * u
    return torch.bmm(mid, dn_eih)


def _fused_group_bmm_full(
    stacked: torch.Tensor,
    counts: torch.Tensor,
    gu_eh2i: torch.Tensor,
    dn_eih: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    return _fused_group_bmm_segment(stacked, counts, gu_eh2i, dn_eih, act_fn)


def _numa_worker_entry(
    rank: int,
    e0: int,
    e1: int,
    cpus: list[int],
    worker_threads: int,
    act_name: str,
    out_q: Any,
) -> None:
    os.sched_setaffinity(0, set(cpus))
    torch.set_num_threads(max(1, int(worker_threads)))
    # fork 子进程继承父进程 ATLAS 状态，此处再调 set_num_interop_threads 会报错

    stacked = _MP_SHARED["stacked"]
    counts = _MP_SHARED["counts"]
    gu_eh2i = _MP_SHARED["gu_eh2i"]
    dn_eih = _MP_SHARED["dn_eih"]
    act_fn = _resolve_act_fn(act_name)

    s = stacked[e0:e1].contiguous()
    c = counts[e0:e1].contiguous()
    gu = gu_eh2i[e0:e1].contiguous()
    dn = dn_eih[e0:e1].contiguous()

    t0 = time.perf_counter()
    y_seg = _fused_group_bmm_segment(s, c, gu, dn, act_fn)
    dt = time.perf_counter() - t0
    # 避免 torch.multiprocessing 走 storage fd 还原（在部分环境下会 Connection reset）
    y_np = y_seg.detach().float().cpu().numpy().copy()
    out_q.put((rank, e0, e1, y_np, dt))


def _split_ranges(num_experts: int, parts: int) -> list[tuple[int, int]]:
    base = num_experts // parts
    rem = num_experts % parts
    ranges: list[tuple[int, int]] = []
    cur = 0
    for i in range(parts):
        n = base + (1 if i < rem else 0)
        if n <= 0:
            ranges.append((cur, cur))
            continue
        ranges.append((cur, cur + n))
        cur += n
    return ranges


def _build_presorted_inputs(
    num_experts: int,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_ids = torch.randint(
        0, num_experts, (num_tokens,), device=device, generator=gen, dtype=torch.int64
    )
    expert_ids_sorted, perm = torch.sort(expert_ids)
    x_sorted = torch.randn((num_tokens, hidden), device=device, dtype=dtype, generator=gen)
    x_sorted = x_sorted[perm]
    return x_sorted, expert_ids_sorted


def main() -> None:
    p = argparse.ArgumentParser(description="NUMA 多进程 MoE fused BMM benchmark")
    p.add_argument("--experts", type=int, default=32, help="专家数 E")
    p.add_argument("--tokens", type=int, default=8192, help="总 token 数（展平后排序前）")
    p.add_argument("--hidden", type=int, default=2048, help="H")
    p.add_argument("--intermediate", type=int, default=768, help="I（fused gate_up 宽度为 2I）")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--act", type=str, default="silu", help="transformers ACT2FN 名称，如 silu")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--workers", type=int, default=4, help="并行进程数（默认 4）")
    p.add_argument("--serial-threads", type=int, default=0, help="0 表示不改动默认线程数")
    p.add_argument("--worker-threads", type=int, default=1, help="每个子进程 torch.set_num_threads")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if sys.platform != "linux":
        print("警告: 非 Linux 环境将退回默认 multiprocessing 上下文，可能无法 fork 共享大张量。", file=sys.stderr)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    E, T, H, I = args.experts, args.tokens, args.hidden, args.intermediate
    if E < args.workers:
        raise SystemExit(f"experts ({E}) 应 >= workers ({args.workers})")

    # 权重形状与 cuda_memory_view 中 gate_up / down 一致再转置
    group_gate_up = torch.randn((E, 2 * I, H), device=device, dtype=dtype, generator=gen)
    group_down = torch.randn((E, H, I), device=device, dtype=dtype, generator=gen)
    gu_eh2i = group_gate_up.transpose(1, 2).contiguous()
    dn_eih = group_down.transpose(1, 2).contiguous()

    x_sorted, eid_sorted = _build_presorted_inputs(E, T, H, dtype, device, gen)
    stacked, counts = _batched_pad_inputs_presorted(x_sorted, eid_sorted, E)

    act_name = args.act
    act_fn = _resolve_act_fn(act_name)

    # 正确性：整段 vs 分片拼接
    y_full = _fused_group_bmm_full(stacked, counts, gu_eh2i, dn_eih, act_fn)
    ranges = _split_ranges(E, args.workers)
    chunks = []
    for e0, e1 in ranges:
        if e1 > e0:
            chunks.append(
                _fused_group_bmm_segment(
                    stacked[e0:e1], counts[e0:e1], gu_eh2i[e0:e1], dn_eih[e0:e1], act_fn
                )
            )
    y_cat = torch.cat(chunks, dim=0)
    torch.testing.assert_close(y_full, y_cat, rtol=2e-2, atol=2e-2)

    cpu_groups = discover_numa_cpu_groups(args.workers)
    print("NUMA / CPU 分组（每进程一组）:")
    for i, cpus in enumerate(cpu_groups):
        print(f"  worker{i}: {cpus[:8]}{'...' if len(cpus) > 8 else ''} ({len(cpus)} cpus)")

    _MP_SHARED.clear()
    _MP_SHARED["stacked"] = stacked
    _MP_SHARED["counts"] = counts
    _MP_SHARED["gu_eh2i"] = gu_eh2i
    _MP_SHARED["dn_eih"] = dn_eih

    ctx = get_context("fork") if sys.platform == "linux" else get_context()
    if args.serial_threads > 0:
        torch.set_num_threads(args.serial_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    # warmup
    for _ in range(args.warmup):
        _ = _fused_group_bmm_full(stacked, counts, gu_eh2i, dn_eih, act_fn)

    serial_times: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _ = _fused_group_bmm_full(stacked, counts, gu_eh2i, dn_eih, act_fn)
        serial_times.append(time.perf_counter() - t0)

    mp_times: list[float] = []
    # 各 worker 内仅包住两段 bmm+激活（不含 numpy 回传、不含起进程）
    mp_compute_max_times: list[float] = []
    worker_self_times: list[list[float]] = [[] for _ in range(args.workers)]

    for it in range(args.warmup + args.iters):
        out_q = ctx.Queue()
        t_wall0 = time.perf_counter()
        procs = []
        for w, (e0, e1) in enumerate(ranges):
            if e1 <= e0:
                continue
            proc = ctx.Process(
                target=_numa_worker_entry,
                args=(w, e0, e1, cpu_groups[w], args.worker_threads, act_name, out_q),
                daemon=True,
            )
            proc.start()
            procs.append(proc)
        pieces: dict[int, torch.Tensor] = {}
        child_times: dict[int, float] = {}
        for _ in procs:
            rank, e0, e1, y_np, dt = out_q.get()
            pieces[rank] = torch.from_numpy(y_np).to(device=device, dtype=dtype)
            child_times[rank] = dt
        for proc in procs:
            proc.join()
        wall = time.perf_counter() - t_wall0

        order = sorted(pieces.keys())
        y_mp = torch.cat([pieces[r] for r in order], dim=0)
        torch.testing.assert_close(y_full, y_mp, rtol=2e-2, atol=2e-2)

        if it >= args.warmup:
            mp_times.append(wall)
            mp_compute_max_times.append(max(child_times.values()))
            for r in order:
                worker_self_times[r].append(child_times[r])

    ser = sum(serial_times) / len(serial_times)
    par = sum(mp_times) / len(mp_times)
    par_cmp = sum(mp_compute_max_times) / len(mp_compute_max_times)
    n_cmp = len(mp_compute_max_times)
    cpu_sec = (
        sum(sum(worker_self_times[w]) for w in range(args.workers)) / n_cmp
        if n_cmp
        else 0.0
    )

    print("\n【仅计算耗时】单进程：整段 `_fused_group_bmm_*`；多进程：各子进程内 `torch.bmm`×2 + 激活")
    print(f"  单进程（均值）: {ser * 1e3:.3f} ms")
    print(
        f"  多进程理想并行墙 max(worker 内耗时)（均值）: {par_cmp * 1e3:.3f} ms, "
        f"相对加速 {ser / par_cmp:.3f}x"
    )
    print(
        f"  多进程 CPU 时间之和 Σworker（均值，供参考）: {cpu_sec * 1e3:.3f} ms "
        f"（≈ {cpu_sec / ser:.2f}× 单进程算量）"
    )

    print("\n【含调度】父进程 wall（起进程 + Queue + join + 校验），通常远大于纯算子：")
    print(f"  单进程（同上）: {ser * 1e3:.3f} ms")
    print(f"  {args.workers} 进程 wall（均值）: {par * 1e3:.3f} ms, 相对加速 {ser / par:.3f}x")
    for w in range(args.workers):
        if worker_self_times[w]:
            m = sum(worker_self_times[w]) / len(worker_self_times[w])
            print(f"    worker{w} 内算子耗时（均值）: {m * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
