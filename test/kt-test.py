#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def add_kt_kernel_to_path(root: Path) -> None:
    kt_kernel_dir = root / "ktransformers" / "kt-kernel"
    python_dir = kt_kernel_dir / "python"
    for path in (str(kt_kernel_dir), str(python_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_text_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def detect_numa_nodes() -> list[int]:
    node_root = Path("/sys/devices/system/node")
    nodes: list[int] = []
    if node_root.exists():
        for child in node_root.iterdir():
            name = child.name
            if name.startswith("node") and name[4:].isdigit():
                nodes.append(int(name[4:]))
    return sorted(nodes) or [0]


def default_thread_count(numa_nodes: list[int]) -> int:
    cpu_root = Path("/sys/devices/system/cpu")
    physical_cores: set[tuple[str, str]] = set()
    if cpu_root.exists():
        for cpu_dir in cpu_root.iterdir():
            name = cpu_dir.name
            if not name.startswith("cpu") or not name[3:].isdigit():
                continue
            package_file = cpu_dir / "topology" / "physical_package_id"
            core_file = cpu_dir / "topology" / "core_id"
            try:
                physical_cores.add((package_file.read_text().strip(), core_file.read_text().strip()))
            except OSError:
                pass
    if physical_cores:
        return len(physical_cores)
    return max(1, (os.cpu_count() or 1) // 2)


def make_worker_config(ext, threads: int, numa_nodes: list[int]):
    threadpool_count = max(1, min(len(numa_nodes), threads))
    base = threads // threadpool_count
    extra = threads % threadpool_count

    worker_config = ext.WorkerPoolConfig()
    worker_config.subpool_count = threadpool_count
    worker_config.subpool_numa_map = numa_nodes[:threadpool_count]
    worker_config.subpool_thread_count = [base + (1 if i < extra else 0) for i in range(threadpool_count)]
    return worker_config

def make_bf16_expert_weights(num_experts: int, hidden_size: int, intermediate_size: int, seed: int):
    torch.manual_seed(seed)
    
    # 向量化生成（与 bench_bf16_moe.py 一致）
    gate_proj = (torch.randn((num_experts, intermediate_size, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
    up_proj = (torch.randn((num_experts, intermediate_size, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
    down_proj = (torch.randn((num_experts, hidden_size, intermediate_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
    
    return gate_proj, up_proj, down_proj

def make_inputs(tokens: int, hidden_size: int, num_experts: int, top_k: int, seed: int, uniform: bool = False):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    hidden_states = torch.randn((tokens, hidden_size), generator=generator, dtype=torch.float32).to(torch.bfloat16).contiguous()

    if uniform:
        # ===== 均匀分配策略：每个专家获得相同数量的token =====
        tokens_per_expert = tokens * top_k // num_experts
        remainder = tokens * top_k % num_experts
        
        expert_assignments = []
        for expert_id in range(num_experts):
            count = tokens_per_expert + (1 if expert_id < remainder else 0)
            expert_assignments.extend([expert_id] * count)
        
        expert_assignments = torch.tensor(expert_assignments, dtype=torch.int64)
        expert_assignments = expert_assignments[torch.randperm(len(expert_assignments), generator=generator)]
        topk_ids = expert_assignments.view(tokens, top_k).contiguous()
    else:
        # ===== 随机分配策略：模拟真实MoE路由 =====
        scores = torch.rand((tokens, num_experts), generator=generator, dtype=torch.float32)
        _, topk_ids = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=False)
        topk_ids = topk_ids.to(torch.int64).contiguous()

    topk_weights = torch.rand((tokens, top_k), generator=generator, dtype=torch.float32).contiguous()
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).contiguous()

    output = torch.empty((tokens, hidden_size), dtype=torch.bfloat16).contiguous()
    batch_tensor = torch.tensor([tokens], dtype=torch.int32)
    return hidden_states, topk_ids, topk_weights, output, batch_tensor


def submit_forward(cpu_infer, moe, batch_tensor, topk_ids, topk_weights, hidden_states, output, top_k: int) -> None:
    cpu_infer.submit(
        moe.forward_task(
            batch_tensor.data_ptr(),
            top_k,
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            hidden_states.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run kt-kernel CPU MoE expert benchmark using Gemma config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root_from_file() / "lmp" / "src" / "models" / "Gemma" / "config.json",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0, help="CPU worker threads; default uses physical core count")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--uniform", action="store_true", help="Use uniform token distribution across experts (default: random)")
    args = parser.parse_args()

    root = repo_root_from_file()
    add_kt_kernel_to_path(root)

    import kt_kernel

    ext = kt_kernel.kt_kernel_ext
    text_cfg = load_text_config(args.config)

    hidden_size = int(text_cfg["hidden_size"])
    intermediate_size = int(text_cfg["moe_intermediate_size"])
    num_experts = int(text_cfg["num_experts"])
    top_k = int(text_cfg["top_k_experts"])
    tokens = args.batch * args.seq_len

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    numa_nodes = detect_numa_nodes()
    threads = args.threads if args.threads > 0 else default_thread_count(numa_nodes)
    worker_config = make_worker_config(ext, threads, numa_nodes)

    print(f"kt-kernel version      : {kt_kernel.__version__}")
    print(f"kt-kernel CPU variant : {kt_kernel.__cpu_variant__}")
    print(f"backend               : AVX2BF16_MOE, CPU-only, no CUDA/pinned memory")
    print(f"config                : {args.config}")
    print(f"batch x seq_len       : {args.batch} x {args.seq_len} = {tokens} tokens")
    print(f"experts/top_k         : {num_experts}/{top_k}")
    print(f"hidden/intermediate   : {hidden_size}/{intermediate_size}")
    print(f"threads/NUMA nodes    : {threads}/{worker_config.subpool_numa_map}")
    print(f"token distribution    : {'uniform' if args.uniform else 'random'}")

    cpu_infer = ext.CPUInfer(worker_config)
    gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool).contiguous()

    cfg = ext.moe.MOEConfig(num_experts, top_k, hidden_size, intermediate_size, 0)
    cfg.layer_idx = 0
    cfg.pool = cpu_infer.backend_


    cfg.max_len = tokens
    # cfg.load = True
    # cfg.save = True

    cfg.path = tempfile.mkdtemp(prefix="kt-gemma-bf16-")

    print("allocating BF16 expert weights...")
    gate_proj, up_proj, down_proj = make_bf16_expert_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=args.seed,
    )
    cfg.gate_proj = gate_proj.data_ptr()
    cfg.up_proj = up_proj.data_ptr()
    cfg.down_proj = down_proj.data_ptr()

    # No scales for BF16
    cfg.gate_scale = 0
    cfg.up_scale = 0
    cfg.down_scale = 0

    moe = ext.moe.AVX512BF16_MOE(cfg)
    physical_to_logical = torch.arange(num_experts, dtype=torch.int64).contiguous()

    print("loading weights into kt-kernel...")
    t0 = time.perf_counter()
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    load_s = time.perf_counter() - t0
    print(f"load time             : {load_s:.3f} s")

    hidden_states, topk_ids, topk_weights, output, batch_tensor = make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        seed=args.seed + 1,
        uniform=False,
    )

    print("warming up...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            submit_forward(cpu_infer, moe, batch_tensor, topk_ids, topk_weights, hidden_states, output, top_k)

        print("benchmarking...")
        times_ms: list[float] = []
        for i in range(args.iters):
            t0 = time.perf_counter()
            submit_forward(cpu_infer, moe, batch_tensor, topk_ids, topk_weights, hidden_states, output, top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times_ms.append(elapsed_ms)
            print(f"iter {i:02d}              : {elapsed_ms:.3f} ms")

    avg_ms = sum(times_ms) / len(times_ms)
    best_ms = min(times_ms)
    print(f"avg latency           : {avg_ms:.3f} ms")
    print(f"best latency          : {best_ms:.3f} ms")
    print(f"avg tokens/s          : {tokens / (avg_ms / 1000.0):.2f}")
    print(f"best tokens/s         : {tokens / (best_ms / 1000.0):.2f}")
    print(f"output checksum       : {float(output.float().sum()):.6f}")

    keep_alive = (gate_proj, up_proj, down_proj)
    _ = keep_alive


if __name__ == "__main__":
    main()
