#!/usr/bin/env python3
"""
Benchmark: copy weight-sized tensors from pinned CPU to GPU(s), using
pin_memory + non_blocking H2D.

- Single GPU: --num-gpus 1 --device cuda:0
- Four GPUs in parallel (each ~13 GiB by default): --num-gpus 4 --gib-per-gpu 13

Parallel mode uses one thread per GPU (each calls torch.cuda.set_device in that
thread). All threads synchronize on a Barrier before starting the timed H2D region
(alloc + warmup are outside the per-thread H2D timer). Parallel wall time
for the copy phase is max(per-GPU H2D times) when transfers overlap; a
separate line reports full executor join time (includes allocation).

Requires CUDA. Binary GiB (1024**3) unless --decimal-gb.
"""

from __future__ import annotations

import argparse
import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch


def _dtype_bytes(dt: torch.dtype) -> int:
    if dt == torch.float32:
        return 4
    if dt in (torch.bfloat16, torch.float16):
        return 2
    raise ValueError(f"unsupported dtype {dt}")


def _numel_for_bytes(target_bytes: int, bpe: int) -> int:
    numel = target_bytes // bpe
    if numel < 1:
        raise SystemExit("numel < 1; increase --gib / --gib-per-gpu")
    return numel


def run_single_gpu(
    *,
    target_bytes: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    compare_unpinned: bool,
) -> None:
    bpe = _dtype_bytes(dtype)
    numel = _numel_for_bytes(target_bytes, bpe)
    torch.cuda.set_device(device)

    print(
        f"[single] Allocating pinned CPU tensor: numel={numel:,} dtype={dtype} "
        f"bytes={target_bytes:,}"
    )

    t_cpu = torch.empty(numel, dtype=dtype, pin_memory=True)
    t_cpu[0] = 0
    t_cpu[-1] = 0

    for _ in range(max(0, warmup)):
        w = torch.empty(1024 * 1024, dtype=dtype, pin_memory=True)
        w[0] = 0
        _ = w.to(device, non_blocking=True)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_gpu = t_cpu.to(device, non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    gib_s = (target_bytes / (1024**3)) / elapsed
    print(f"[single] device={device} pin_memory=True non_blocking=True")
    print(f"[single] H2D time: {elapsed:.4f} s")
    print(f"[single] Effective bandwidth: {gib_s:.2f} GiB/s (binary)")

    _ = t_gpu

    if compare_unpinned:
        gc.collect()
        t_slow = torch.empty(numel, dtype=dtype, pin_memory=False)
        t_slow[0] = 0
        t_slow[-1] = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = t_slow.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[single] unpinned baseline H2D time: {t1 - t0:.4f} s")

    del t_cpu, t_gpu
    gc.collect()
    torch.cuda.empty_cache()


def run_multi_gpu_parallel(
    *,
    num_gpus: int,
    gib_per_gpu: float,
    decimal_gb: bool,
    dtype: torch.dtype,
    warmup: int,
) -> None:
    if torch.cuda.device_count() < num_gpus:
        raise SystemExit(
            f"Need at least {num_gpus} CUDA devices, have {torch.cuda.device_count()}"
        )

    unit = 1_000_000_000 if decimal_gb else 1024**3
    target_bytes = int(gib_per_gpu * unit)
    bpe = _dtype_bytes(dtype)
    numel = _numel_for_bytes(target_bytes, bpe)
    total_bytes = target_bytes * num_gpus

    print(
        f"[parallel x{num_gpus}] per-GPU bytes={target_bytes:,} (~{gib_per_gpu} "
        f"{'GB' if decimal_gb else 'GiB'}), dtype={dtype}, numel_each={numel:,}"
    )
    print(f"[parallel x{num_gpus}] total host pinned RAM ~{total_bytes / unit:.3f} {'GB' if decimal_gb else 'GiB'}")

    # Per-thread pinned buffers (allocated inside workers after set_device for locality)
    barrier = threading.Barrier(num_gpus)
    results: dict[int, float] = {}
    errors: list[BaseException] = []

    def worker(gpu_id: int) -> None:
        try:
            torch.cuda.set_device(gpu_id)
            t_cpu = torch.empty(numel, dtype=dtype, pin_memory=True)
            t_cpu[0] = 0
            t_cpu[-1] = 0

            for _ in range(max(0, warmup)):
                w = torch.empty(256 * 1024, dtype=dtype, pin_memory=True)
                w[0] = 0
                _ = w.to(torch.device(f"cuda:{gpu_id}"), non_blocking=True)
            torch.cuda.synchronize()

            barrier.wait()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _t_gpu = t_cpu.to(torch.device(f"cuda:{gpu_id}"), non_blocking=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            results[gpu_id] = elapsed
            _ = _t_gpu, t_cpu
        except BaseException as e:
            errors.append(e)

    # Includes per-thread pinned alloc (~13 GiB each) + warmup + H2D — not comparable to single-GPU H2D-only timing.
    t_join0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_gpus) as ex:
        futs = [ex.submit(worker, gid) for gid in range(num_gpus)]
        for f in as_completed(futs):
            f.result()
    t_join1 = time.perf_counter()

    if errors:
        raise errors[0]

    join_wall = t_join1 - t_join0
    per_max = max(results.values())
    per_min = min(results.values())
    # Overlapping copies: all finish within ~max(per-GPU H2D), same semantics as single-GPU timed region.
    h2d_wall = per_max
    agg_gib_s = (total_bytes / (1024**3)) / h2d_wall

    print(f"[parallel x{num_gpus}] per-GPU H2D time (timed region only, sync own device): min={per_min:.4f}s max={per_max:.4f}s")
    for gid in sorted(results):
        print(f"  cuda:{gid}: {results[gid]:.4f} s")
    print(
        f"[parallel x{num_gpus}] parallel H2D wall ≈ max(per-GPU): {h2d_wall:.4f} s "
        f"(compare single-GPU H2D; not executor join)"
    )
    print(
        f"[parallel x{num_gpus}] aggregate bandwidth (sum_bytes / max_H2D): {agg_gib_s:.2f} GiB/s (binary)"
    )
    print(
        f"[parallel x{num_gpus}] executor join (alloc+warmup+H2D, for diagnostics only): {join_wall:.4f} s"
    )

    gc.collect()
    for gid in range(num_gpus):
        torch.cuda.synchronize(gid)
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use in parallel (1 = single-GPU mode).",
    )
    ap.add_argument(
        "--gib-per-gpu",
        type=float,
        default=6.5,
        help="Bytes per GPU in parallel mode (binary GiB unless --decimal-gb).",
    )
    ap.add_argument(
        "--gib",
        type=float,
        default=None,
        help="Single-GPU mode only: same meaning as before (defaults to --gib-per-gpu if unset).",
    )
    ap.add_argument("--decimal-gb", action="store_true")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    ap.add_argument("--compare-unpinned", action="store_true")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test.")

    dtype = getattr(torch, args.dtype)

    if args.num_gpus < 1:
        raise SystemExit("--num-gpus must be >= 1")

    if args.num_gpus == 1:
        unit = 1_000_000_000 if args.decimal_gb else 1024**3
        gib = args.gib if args.gib is not None else args.gib_per_gpu
        target_bytes = int(gib * unit)
        run_single_gpu(
            target_bytes=target_bytes,
            dtype=dtype,
            device=torch.device(args.device),
            warmup=args.warmup,
            compare_unpinned=args.compare_unpinned,
        )
    else:
        run_multi_gpu_parallel(
            num_gpus=args.num_gpus,
            gib_per_gpu=args.gib_per_gpu,
            decimal_gb=args.decimal_gb,
            dtype=dtype,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # 四卡并行时默认暴露 0–3；若只想测两张卡可 export CUDA_VISIBLE_DEVICES=0,1
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    main()