#!/usr/bin/env python3
"""
Benchmark: sequential expert matmul vs grouped_mm vs batched_mm (per-token gathered weights).

Inspired by lmp/test/test_einsum_experts.py (synthetic tensors, timing loops), but targets the
same dispatch pattern as transformers MoE: many token–expert pairs, one shared expert weight bank.

- **sequential**: for each expert e, rows where expert_ids==e, torch.mm
- **grouped**: argsort by expert, cumsum offsets, transformers `_grouped_mm`, inverse permute
- **batched**: weight[expert_ids] -> (S,H,N), torch.bmm(x.unsqueeze(1), Ws).squeeze(1)

Allocations:
- **uniform**: expert ids round-robin so each expert gets ~S/E rows
- **skewed**: power-law-ish counts (few experts dominate)

Run: ``python lmp/test/benchmark_expert_mm_dispatch.py`` (uses CPU + CUDA if available).

**Note:** ``batched_mm`` materializes ``weight[expert_ids]`` with shape ``(S, H, N)``; for large ``S*H*N`` this dominates time and memory. Increase sizes gradually (see ``--pairs`` / ``--hidden`` / ``--intermediate``).

**``--moe-multi-model`` CUDA prefill:** CUDA can use a **denser** ``--moe-prefill-seq-cuda`` grid (default ``32,48,64,80,96,112,128``) while CPU keeps ``--moe-prefill-seq``; optional ``--moe-max-gather-gb-cuda``, ``--moe-batches-cuda`` (decode), ``--moe-prefill-batches-cuda`` (GPU prefill-only B), and ``--moe-prefill-skip-cpu`` to avoid VRAM OOM on large prefill.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import torch

try:
    from transformers.integrations.moe import _grouped_mm
except ImportError:  # minimal fallback if transformers layout changes
    _grouped_mm = None  # type: ignore[misc, assignment]


AllocName = Literal["uniform", "skewed"]


def make_expert_ids_cpu(
    num_pairs: int,
    num_experts: int,
    allocation: AllocName,
    generator: torch.Generator,
) -> torch.Tensor:
    """Long tensor (S,) on CPU so the same routing is used when copied to CPU/GPU."""
    if allocation == "uniform":
        return torch.arange(num_pairs, dtype=torch.int64) % num_experts

    logits = torch.linspace(3.0, 0.0, num_experts)
    probs = torch.softmax(logits, dim=0)
    return torch.multinomial(probs, num_pairs, replacement=True, generator=generator)


def sequential_expert_mm(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """x: (S,H), expert_ids: (S,), weight: (E,H,N). Same row order as x."""
    e_count = weight.size(0)
    out = torch.empty(x.size(0), weight.size(2), device=x.device, dtype=x.dtype)
    for e in range(e_count):
        mask = expert_ids == e
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        out[idx] = torch.mm(x[idx], weight[e])
    return out


def grouped_prepare_sorted(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x.device
    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device, dtype=perm.dtype)
    x_g = x[perm]
    e_sorted = expert_ids[perm]
    offs = torch.cumsum(torch.bincount(e_sorted, minlength=num_experts), dim=0).to(torch.int32)
    return x_g, offs, inv_perm


def grouped_mm_dispatch(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Sort by expert, offsets, _grouped_mm, restore order (matches MoE grouped path)."""
    if _grouped_mm is None:
        raise RuntimeError("transformers.integrations.moe._grouped_mm is required")
    x_g, offs, inv_perm = grouped_prepare_sorted(x, expert_ids, weight.size(0))
    y_g = _grouped_mm(x_g, weight, offs=offs)
    return y_g[inv_perm]


def batched_mm_dispatch(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Batched expert matmul using **grouped/padded inputs** (mirrors `mlpmodule.py` idea):

    - Build `stacked_inputs`: `(E, max_tokens, H)` by grouping tokens per expert (padding within each expert).
    - Run one `torch.bmm(stacked_inputs, weight)` where `weight` is `(E, H, N)`.
    - Unpad + restore original token order.

    This avoids materializing `weight[expert_ids]` with shape `(S, H, N)` (the classic gather-heavy batched path).
    """
    stacked_inputs, counts, perm, inv_perm = batched_pad_inputs(x, expert_ids, weight.size(0))
    y_stacked = batched_bmm_only_from_stacked(stacked_inputs, weight)
    y_sorted = batched_unpad_outputs(y_stacked, counts)
    return y_sorted[inv_perm]


def batched_pad_inputs(
    x: torch.Tensor, expert_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return `(stacked_inputs, counts, perm, inv_perm)`:

    - `stacked_inputs`: `(E, max_tokens, H)` where each expert's tokens are packed contiguously and padded.
    - `counts`: `(E,)` number of tokens for each expert (int64, same device as `x`).
    - `perm`: permutation that sorts `expert_ids` (for `x_sorted = x[perm]`).
    - `inv_perm`: inverse permutation to restore original order (`y = y_sorted[inv_perm]`).
    """
    device = x.device
    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device, dtype=perm.dtype)

    x_sorted = x[perm]
    e_sorted = expert_ids[perm]
    counts = torch.bincount(e_sorted, minlength=num_experts)
    max_tokens = int(counts.max().item()) if counts.numel() else 0
    e = num_experts
    h = x.size(1)
    stacked_inputs = torch.zeros((e, max_tokens, h), device=device, dtype=x.dtype)

    # Fill per expert (loop matches mlpmodule's approach; keeps it simple and stable).
    start = 0
    for expert_idx in range(e):
        c = int(counts[expert_idx].item())
        if c:
            stacked_inputs[expert_idx, :c].copy_(x_sorted[start : start + c], non_blocking=False)
        start += c
    return stacked_inputs, counts, perm, inv_perm


def batched_bmm_only_from_stacked(stacked_inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Pure compute: `(E, T, H) @ (E, H, N)` -> `(E, T, N)`."""
    return torch.bmm(stacked_inputs, weight)


def batched_unpad_outputs(y_stacked: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Inverse of padding: `(E, max_tokens, N)` -> `(S, N)` in expert-sorted order."""
    outs: list[torch.Tensor] = []
    e = int(counts.numel())
    for expert_idx in range(e):
        c = int(counts[expert_idx].item())
        if c:
            outs.append(y_stacked[expert_idx, :c])
    if not outs:
        return y_stacked.reshape(0, y_stacked.size(-1))
    return torch.cat(outs, dim=0)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    for _ in range(warmup):
        _ = fn()
    sync_if_cuda(device)
    t0 = time.perf_counter()
    last = None
    for _ in range(iters):
        last = fn()
    sync_if_cuda(device)
    t1 = time.perf_counter()
    return (t1 - t0) / iters, last  # type: ignore[return-value]


def token_histogram(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(expert_ids, minlength=num_experts)


@dataclass(frozen=True)
class MoeSpec:
    model_id: str
    config_path: str
    hidden_size: int
    expert_out_dim: int
    num_experts: int
    top_k: int
    notes: str


def models_root_default() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "models"


def parse_int_grid(spec: str) -> list[int]:
    s = spec.strip()
    if "-" in s and ":" in s:
        left, right = s.rsplit(":", 1)
        a, b = left.split("-", 1)
        return list(range(int(a), int(b) + 1, int(right)))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_moe_spec(config_path: Path) -> MoeSpec:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    mid = config_path.parent.name
    p = str(config_path)

    def tc() -> dict:
        return data.get("text_config") or {}

    mt = data.get("model_type") or ""

    if mt == "gemma4" or tc().get("model_type") == "gemma4_text":
        t = tc() or data
        mi = int(t["moe_intermediate_size"])
        return MoeSpec(
            mid,
            p,
            int(t["hidden_size"]),
            2 * mi,
            int(t["num_experts"]),
            int(t["top_k_experts"]),
            f"gemma4 text gate+up N=2×{mi}",
        )
    if mt == "qwen3_moe":
        return MoeSpec(
            mid,
            p,
            int(data["hidden_size"]),
            int(data["moe_intermediate_size"]),
            int(data["num_experts"]),
            int(data["num_experts_per_tok"]),
            "qwen3_moe N=moe_intermediate_size",
        )
    if mt == "qwen2_moe":
        return MoeSpec(
            mid,
            p,
            int(data["hidden_size"]),
            int(data["moe_intermediate_size"]),
            int(data["num_experts"]),
            int(data["num_experts_per_tok"]),
            "qwen2_moe",
        )
    if mt == "qwen3_5_moe":
        t = tc()
        return MoeSpec(
            mid,
            p,
            int(t["hidden_size"]),
            int(t["moe_intermediate_size"]),
            int(t["num_experts"]),
            int(t["num_experts_per_tok"]),
            "qwen3_5_moe text_config",
        )
    if mt in ("deepseek_v2", "deepseek"):
        return MoeSpec(
            mid,
            p,
            int(data["hidden_size"]),
            int(data["moe_intermediate_size"]),
            int(data["n_routed_experts"]),
            int(data["num_experts_per_tok"]),
            "deepseek n_routed_experts",
        )
    if mt == "ernie4_5_moe":
        return MoeSpec(
            mid,
            p,
            int(data["hidden_size"]),
            int(data["moe_intermediate_size"]),
            int(data["moe_num_experts"]),
            int(data["moe_k"]),
            "ernie4_5_moe",
        )
    if mt == "ernie4_5_moe_vl":
        moe_i = data["moe_intermediate_size"]
        mi = int(moe_i[0] if isinstance(moe_i, list) else moe_i)
        moe_e = data["moe_num_experts"]
        ne = int(moe_e[0] if isinstance(moe_e, list) else moe_e)
        return MoeSpec(mid, p, int(data["hidden_size"]), mi, ne, int(data["moe_k"]), "ernie4_5_moe_vl text slice")
    raise ValueError(f"unsupported model_type={mt!r} in {config_path}")


DEFAULT_MULTI_MODEL_CONFIG_NAMES = (
    "gemma4-26B-A4B",
    "ERNIE-4.5-VL-28B-A3B-Thinking",
    "ERNIE-4.5-21B-A3B-Thinking",
    "DeepSeek-V2-Lite",
    "deepseek-moe-16b-base",
    "Qwen1.5-MoE-A2.7B",
    "Qwen3-30B-A3B",
    "Qwen3.5-35B",
)


def measure_moe_dispatch_row(
    spec: MoeSpec,
    device: torch.device,
    dtype: torch.dtype,
    seq_tokens: int,
    batch: int,
    phase: Literal["prefill", "decode"],
    allocation: AllocName,
    generator: torch.Generator,
    warmup: int,
    iters: int,
    max_gather_gb: float,
    max_s_sequential: int,
) -> dict | None:
    assert _grouped_mm is not None
    e, h, n, top_k = spec.num_experts, spec.hidden_size, spec.expert_out_dim, spec.top_k
    s = batch * seq_tokens * top_k
    elem = 4 if dtype == torch.float32 else 2
    gather_gb = s * h * n * elem / (1024**3)
    if gather_gb > max_gather_gb:
        return None

    expert_ids = make_expert_ids_cpu(s, e, allocation, generator).to(device)
    x = torch.randn(s, h, device=device, dtype=dtype)
    weight = torch.randn(e, h, n, device=device, dtype=dtype)

    y_seq = sequential_expert_mm(x, expert_ids, weight)
    y_grp = grouped_mm_dispatch(x, expert_ids, weight)
    y_bat = batched_mm_dispatch(x, expert_ids, weight)
    rtol, atol = (0.2, 4.0) if dtype in (torch.bfloat16, torch.float16) else (1e-2, 1e-2)
    if not torch.allclose(y_seq.float(), y_grp.float(), rtol=rtol, atol=atol):
        raise AssertionError(f"{spec.model_id} seq vs grouped")
    if not torch.allclose(y_grp.float(), y_bat.float(), rtol=rtol, atol=atol):
        raise AssertionError(f"{spec.model_id} grouped vs batched")

    x_g, offs, _inv = grouped_prepare_sorted(x, expert_ids, e)
    stacked_inputs, counts, _perm, _invp = batched_pad_inputs(x, expert_ids, e)

    def wrap_grp() -> torch.Tensor:
        return grouped_mm_dispatch(x, expert_ids, weight)

    def wrap_bat() -> torch.Tensor:
        return batched_mm_dispatch(x, expert_ids, weight)

    def wrap_gath() -> torch.Tensor:
        # "gather" replacement: pad/pack inputs into `(E, max_tokens, H)`
        return batched_pad_inputs(x, expert_ids, e)[0]

    def wrap_bmm() -> torch.Tensor:
        return batched_bmm_only_from_stacked(stacked_inputs, weight)

    def wrap_mm() -> torch.Tensor:
        return _grouped_mm(x_g, weight, offs=offs)

    t_seq = None
    if s <= max_s_sequential:

        def wrap_seq() -> torch.Tensor:
            return sequential_expert_mm(x, expert_ids, weight)

        t_seq, _ = bench(wrap_seq, device, warmup, iters)
    t_grp, _ = bench(wrap_grp, device, warmup, iters)
    t_bat, _ = bench(wrap_bat, device, warmup, iters)
    t_gath, _ = bench(wrap_gath, device, warmup, iters)
    t_bmm, _ = bench(wrap_bmm, device, warmup, iters)
    t_gmm, _ = bench(wrap_mm, device, warmup, iters)

    hist = token_histogram(expert_ids, e)
    del x, weight, expert_ids, x_g, offs, stacked_inputs, counts, y_seq, y_grp, y_bat
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_id": spec.model_id,
        "config_path": spec.config_path,
        "phase": phase,
        "batch": batch,
        "seq_tokens": seq_tokens,
        "S": s,
        "alloc": allocation,
        "gather_gb": round(gather_gb, 4),
        "min_exp": int(hist.min().item()),
        "max_exp": int(hist.max().item()),
        "ms_seq": round(t_seq * 1000, 4) if t_seq is not None else None,
        "ms_grp_full": round(t_grp * 1000, 4),
        "ms_bat_full": round(t_bat * 1000, 4),
        "ms_bat_gather": round(t_gath * 1000, 4),
        "ms_bat_bmm": round(t_bmm * 1000, 4),
        "ms_grp_mm_only": round(t_gmm * 1000, 4),
        "seq_skipped": t_seq is None,
        "bench_warmup": warmup,
        "bench_iters": iters,
        "bat_full_over_grp_full": round(t_bat / t_grp, 4) if t_grp > 0 else None,
        "bmm_over_grp_mm": round(t_bmm / t_gmm, 4) if t_gmm > 0 else None,
        "grp_over_seq": round(t_grp / t_seq, 4) if t_seq is not None and t_seq > 0 else None,
        "bat_over_seq": round(t_bat / t_seq, 4) if t_seq is not None and t_seq > 0 else None,
        "notes": spec.notes,
        "H": h,
        "N": n,
        "E": e,
        "top_k": top_k,
    }


def run_multi_model_moe_profile(
    config_paths: list[Path],
    devices: list[torch.device],
    dtype: torch.dtype,
    generator: torch.Generator,
    warmup: int,
    iters: int,
    batches: list[int],
    prefill_seqs: list[int],
    allocs: tuple[AllocName, ...],
    max_gather_gb: float,
    max_s_sequential: int,
    warmup_cpu: int,
    iters_cpu: int,
    *,
    batches_cuda: list[int] | None = None,
    prefill_seqs_cuda: list[int] | None = None,
    max_gather_gb_cuda: float | None = None,
    prefill_skip_cpu: bool = False,
    prefill_batches_cuda: list[int] | None = None,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0

    def wi(dev: torch.device) -> tuple[int, int]:
        return (warmup_cpu, iters_cpu) if dev.type == "cpu" else (warmup, iters)

    def grid_for(dev: torch.device) -> tuple[list[int], list[int], float]:
        bt = batches_cuda if dev.type == "cuda" and batches_cuda is not None else batches
        if dev.type == "cuda" and prefill_seqs_cuda is not None:
            pf = prefill_seqs_cuda
        else:
            pf = prefill_seqs
        cap = max_gather_gb_cuda if dev.type == "cuda" and max_gather_gb_cuda is not None else max_gather_gb
        return bt, pf, cap

    for cfg_path in config_paths:
        try:
            spec = load_moe_spec(cfg_path)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as ex:
            rows.append({"model_id": cfg_path.parent.name, "config_path": str(cfg_path), "error": str(ex)})
            continue
        for device in devices:
            bt, pf, cap = grid_for(device)
            bt_prefill = (
                prefill_batches_cuda if device.type == "cuda" and prefill_batches_cuda is not None else bt
            )
            for alloc in allocs:
                for batch in bt_prefill:
                    if not (prefill_skip_cpu and device.type == "cpu"):
                        for seq_t in pf:
                            w_r, i_r = wi(device)
                            try:
                                row = measure_moe_dispatch_row(
                                    spec,
                                    device,
                                    dtype,
                                    seq_t,
                                    batch,
                                    "prefill",
                                    alloc,
                                    generator,
                                    w_r,
                                    i_r,
                                    cap,
                                    max_s_sequential,
                                )
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                    rows.append(
                                        {
                                            "model_id": spec.model_id,
                                            "oom": True,
                                            "device": device.type,
                                            "phase": "prefill",
                                            "batch": batch,
                                            "seq_tokens": seq_t,
                                            "alloc": alloc,
                                        }
                                    )
                                    continue
                                raise
                            if row is None:
                                skipped += 1
                                continue
                            row["device"] = device.type
                            rows.append(row)
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                for batch in bt:
                    try:
                        w_d, i_d = wi(device)
                        row_d = measure_moe_dispatch_row(
                            spec,
                            device,
                            dtype,
                            1,
                            batch,
                            "decode",
                            alloc,
                            generator,
                            w_d,
                            i_d,
                            cap,
                            max_s_sequential,
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            rows.append(
                                {
                                    "model_id": spec.model_id,
                                    "oom": True,
                                    "device": device.type,
                                    "phase": "decode",
                                    "batch": batch,
                                    "seq_tokens": 1,
                                    "alloc": alloc,
                                }
                            )
                            continue
                        raise
                    if row_d is None:
                        skipped += 1
                    else:
                        row_d["device"] = device.type
                        rows.append(row_d)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
    return rows, skipped


def _write_multi_model_moe_markdown(
    path: Path, args: argparse.Namespace, rows: list[dict], skipped_gather: int
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ok = [r for r in rows if "ms_grp_full" in r]
    err = [r for r in rows if "error" in r]
    oom = [r for r in rows if r.get("oom")]
    devs_measured = sorted({r["device"] for r in ok if "device" in r}, key=lambda d: (d != "cpu", d))
    lines = [
        "# 多模型 MoE dispatch（prefill vs decode）",
        "",
        f"- **Generated:** {ts}",
        f"- **models root:** `{Path(args.moe_models_root)}`",
        f"- **devices measured:** {', '.join(devs_measured) or '(none)'}",
        f"- **dtype:** {args.dtype}",
        f"- **GPU timing:** warmup={args.warmup} iters={args.iters}",
        f"- **CPU timing:** warmup={args.moe_cpu_warmup} iters={args.moe_cpu_iters}（CPU 默认较少次迭代以缩短总耗时；可与 GPU 不可直接比绝对 ms）",
        f"- **batches (CPU / default):** {args.moe_batches} **prefill T (CPU):** {args.moe_prefill_seq} **alloc:** {args.moe_alloc}",
        f"- **CUDA prefill T:** {(args.moe_prefill_seq_cuda or '').strip() or '(同 CPU)'}",
        f"- **CUDA batches (decode / shared):** {(args.moe_batches_cuda or '').strip() or '(同 CPU)'}",
        f"- **CUDA prefill B-only:** {(args.moe_prefill_batches_cuda or '').strip() or '(同 CUDA batches)'}",
        f"- **max_gather_gb (CPU/default):** {args.moe_max_gather_gb} **CUDA only:** {args.moe_max_gather_gb_cuda if args.moe_max_gather_gb_cuda is not None else '(同左)'}",
        f"- **prefill_skip_cpu:** {args.moe_prefill_skip_cpu}",
        f"- **max_S_sequential:** {args.moe_max_s_sequential}",
        f"- **rows OK:** {len(ok)} **skipped (gather cap):** {skipped_gather} **parse err:** {len(err)} **OOM:** {len(oom)}",
        "",
        "| 列 | 含义 |",
        "|----|------|",
        "| bat_ms | gather+bmm 整段 |",
        "| gath_ms | 仅 gather（无 bmm） |",
        "| bmm_ms | 仅 bmm（已预 gather） |",
        "| gmm_ms | 仅 `_grouped_mm` |",
        "",
    ]
    by_model: dict[str, list[dict]] = {}
    for r in ok:
        by_model.setdefault(r["model_id"], []).append(r)
    for mk in sorted(by_model.keys()):
        sub = sorted(
            by_model[mk],
            key=lambda x: (0 if x.get("device") == "cpu" else 1, x["alloc"], x["phase"], x["batch"], x["seq_tokens"]),
        )
        first = sub[0]
        lines += [
            f"## `{mk}`",
            "",
            f"- `{first.get('config_path','')}`",
            f"- **H×N×E×k:** {first['H']}×{first['N']}×{first['E']}×{first['top_k']} — {first.get('notes','')}",
            "",
        ]
        for phase in ("prefill", "decode"):
            chunk = [r for r in sub if r["phase"] == phase]
            if not chunk:
                continue
            lines.append(f"### {phase}")
            lines.append("")
            hdr = (
                "| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | "
                "bat/grp | bmm/gmm | grp/seq | bat/seq |"
            )
            sep = (
                "|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|"
                "--------:|--------:|--------:|--------:|"
            )
            lines += [hdr, sep]
            for r in chunk:
                seq_s = "—" if r.get("seq_skipped") else str(r["ms_seq"])
                grs = "—" if r.get("grp_over_seq") is None else str(r["grp_over_seq"])
                bas = "—" if r.get("bat_over_seq") is None else str(r["bat_over_seq"])
                br = r.get("bat_full_over_grp_full")
                mr = r.get("bmm_over_grp_mm")
                lines.append(
                    f"| {r['device']} | {r['alloc']} | {r['batch']} | {r['seq_tokens']} | {r['S']} | {r['gather_gb']} | "
                    f"{seq_s} | {r['ms_grp_full']} | {r['ms_bat_full']} | {r['ms_bat_gather']} | {r['ms_bat_bmm']} | "
                    f"{r['ms_grp_mm_only']} | {br if br is not None else ''} | {mr if mr is not None else ''} | "
                    f"{grs} | {bas} |"
                )
            lines.append("")
    if err:
        lines += ["## parse errors", ""]
        for r in err:
            lines.append(f"- `{r['model_id']}`: {r.get('error','')}")
        lines.append("")
    if oom:
        lines += ["## OOM", ""]
        for r in oom:
            lines.append(
                f"- {r.get('model_id')} {r.get('device')} {r.get('phase')} B={r.get('batch')} T={r.get('seq_tokens')} {r.get('alloc')}"
            )
        lines.append("")
    repro_parts = [
        f"python {Path(__file__).resolve()} --moe-multi-model",
        f"--moe-models-root {Path(args.moe_models_root)}",
        f'--moe-batches "{args.moe_batches}"',
        f'--moe-prefill-seq "{args.moe_prefill_seq}"',
        f'--moe-prefill-seq-cuda "{(args.moe_prefill_seq_cuda or "").strip()}"',
    ]
    if (args.moe_batches_cuda or "").strip():
        repro_parts.append(f'--moe-batches-cuda "{(args.moe_batches_cuda or "").strip()}"')
    if (args.moe_prefill_batches_cuda or "").strip():
        repro_parts.append(f'--moe-prefill-batches-cuda "{(args.moe_prefill_batches_cuda or "").strip()}"')
    repro_parts.append(f"--moe-max-gather-gb {args.moe_max_gather_gb}")
    if args.moe_max_gather_gb_cuda is not None:
        repro_parts.append(f"--moe-max-gather-gb-cuda {args.moe_max_gather_gb_cuda}")
    if args.moe_prefill_skip_cpu:
        repro_parts.append("--moe-prefill-skip-cpu")
    repro_parts += [
        f"--moe-alloc {args.moe_alloc}",
        f"--moe-max-s-sequential {args.moe_max_s_sequential}",
        f"--moe-cpu-warmup {args.moe_cpu_warmup}",
        f"--moe-cpu-iters {args.moe_cpu_iters}",
        f"--dtype {args.dtype}",
        f"--warmup {args.warmup}",
        f"--iters {args.iters}",
    ]
    lines += ["## 复现", "", "```bash", " \\\n  ".join(repro_parts), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-multi-model", action="store_true", help="Benchmark 8 MoE configs from models/*/config.json")
    parser.add_argument("--moe-models-root", type=Path, default=None)
    parser.add_argument("--moe-model-names", type=str, default=",".join(DEFAULT_MULTI_MODEL_CONFIG_NAMES))
    parser.add_argument("--moe-batches", type=str, default="32-128:32")
    parser.add_argument("--moe-prefill-seq", type=str, default="32-128:32")
    parser.add_argument(
        "--moe-prefill-seq-cuda",
        type=str,
        default="32,48,64,80,96,112,128",
        help="CUDA-only prefill seq_T grid (comma or start-end:step). Empty '' = same as --moe-prefill-seq.",
    )
    parser.add_argument(
        "--moe-batches-cuda",
        type=str,
        default=None,
        help="CUDA-only batch grid for decode (and for prefill if --moe-prefill-batches-cuda unset); default same as --moe-batches.",
    )
    parser.add_argument(
        "--moe-prefill-batches-cuda",
        type=str,
        default="8,16,32",
        help="CUDA-only batch sizes for **prefill** only (decode still uses --moe-batches-cuda or --moe-batches). "
        "Smaller B reduces VRAM vs using the same grid as decode. Empty '' = same as CUDA/decode batch grid.",
    )
    parser.add_argument(
        "--moe-max-gather-gb-cuda",
        type=float,
        default=None,
        help="Gather cap (GiB) on CUDA only; default same as --moe-max-gather-gb. Set higher to keep large prefill on GPU.",
    )
    parser.add_argument(
        "--moe-prefill-skip-cpu",
        action="store_true",
        help="Do not run prefill on CPU (only decode on CPU); frees time/RAM for more CUDA prefill.",
    )
    parser.add_argument("--moe-alloc", choices=["uniform", "skewed", "both"], default="uniform")
    parser.add_argument("--moe-max-gather-gb", type=float, default=48.0)
    parser.add_argument("--moe-max-s-sequential", type=int, default=8192)
    parser.add_argument("--moe-multi-model-gpu-only", action="store_true")
    parser.add_argument(
        "--moe-cpu-warmup",
        type=int,
        default=0,
        help="Warmup iterations on CPU in --moe-multi-model (CPU is slow; default 0).",
    )
    parser.add_argument(
        "--moe-cpu-iters",
        type=int,
        default=1,
        help="Timed iterations on CPU in --moe-multi-model (default 1; GPU uses --iters).",
    )
    parser.add_argument("--moe-output-md", type=Path, default=None)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument(
        "--pairs",
        type=int,
        default=2048,
        help="S = num token–expert pairs (batched path allocates S×H×N gathered weights).",
    )
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Do not run CUDA benchmarks (useful when GPU init hangs or no GPU).",
    )
    args = parser.parse_args()
    if args.moe_models_root is None:
        args.moe_models_root = models_root_default()

    if args.moe_multi_model:
        if _grouped_mm is None:
            raise RuntimeError("Need transformers with integrations.moe._grouped_mm (pip install transformers).")
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        dtype = getattr(torch, args.dtype)
        devs: list[torch.device] = []
        if not args.moe_multi_model_gpu_only:
            devs.append(torch.device("cpu"))
        if not args.cpu_only and torch.cuda.is_available():
            devs.append(torch.device("cuda:0"))
        if not devs:
            raise RuntimeError("No devices (try without --cpu-only / --moe-multi-model-gpu-only).")
        names = tuple(n.strip() for n in args.moe_model_names.split(",") if n.strip())
        paths = [Path(args.moe_models_root) / n / "config.json" for n in names]
        present = [p for p in paths if p.is_file()]
        for p in paths:
            if not p.is_file():
                print(f"[moe-multi-model] missing {p}")
        if not present:
            raise RuntimeError("No config.json found for --moe-model-names.")
        batches = parse_int_grid(args.moe_batches)
        prefill = parse_int_grid(args.moe_prefill_seq)
        raw_pc = (args.moe_prefill_seq_cuda or "").strip()
        prefill_cuda = parse_int_grid(raw_pc) if raw_pc else None
        batches_cuda = parse_int_grid(args.moe_batches_cuda) if (args.moe_batches_cuda or "").strip() else None
        raw_pbc = (args.moe_prefill_batches_cuda or "").strip()
        prefill_batches_cuda = parse_int_grid(raw_pbc) if raw_pbc else None
        allocs: tuple[AllocName, ...] = ("uniform", "skewed") if args.moe_alloc == "both" else (args.moe_alloc,)  # type: ignore[assignment]
        rows_mm, sk = run_multi_model_moe_profile(
            present,
            devs,
            dtype,
            gen,
            args.warmup,
            args.iters,
            batches,
            prefill,
            allocs,
            args.moe_max_gather_gb,
            args.moe_max_s_sequential,
            args.moe_cpu_warmup,
            args.moe_cpu_iters,
            batches_cuda=batches_cuda,
            prefill_seqs_cuda=prefill_cuda,
            max_gather_gb_cuda=args.moe_max_gather_gb_cuda,
            prefill_skip_cpu=args.moe_prefill_skip_cpu,
            prefill_batches_cuda=prefill_batches_cuda,
        )
        out_md = args.moe_output_md or (
            Path(__file__).resolve().parent / "benchmark_multi_model_moe_dispatch_report.md"
        )
        _write_multi_model_moe_markdown(out_md, args, rows_mm, sk)
        print("=== moe-multi-model ===")
        print(f"Wrote {out_md} skipped_gather_cap={sk}")
        return

    dtype = getattr(torch, args.dtype)
    devices: list[torch.device] = [torch.device("cpu")]
    if not args.cpu_only and torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    gen = torch.Generator()
    gen.manual_seed(args.seed)

    if _grouped_mm is None:
        raise RuntimeError("Need transformers with integrations.moe._grouped_mm (pip install transformers).")

    rows = []
    for device in devices:
        for alloc in ("uniform", "skewed"):
            expert_ids = make_expert_ids_cpu(args.pairs, args.experts, alloc, gen).to(device)
            hist = token_histogram(expert_ids, args.experts)
            min_tok = int(hist.min().item())
            max_tok = int(hist.max().item())
            std_tok = float(hist.float().std().item())

            x = torch.randn(args.pairs, args.hidden, device=device, dtype=dtype)
            weight = torch.randn(args.experts, args.hidden, args.intermediate, device=device, dtype=dtype)

            y_seq = sequential_expert_mm(x, expert_ids, weight)
            y_grp = grouped_mm_dispatch(x, expert_ids, weight)
            y_bat = batched_mm_dispatch(x, expert_ids, weight)

            def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
                return (a.float() - b.float()).abs().max().item()

            # Low-precision matmuls differ across kernel shapes (many small mm vs one bmm); check in fp32 with loose bounds.
            rtol, atol = (0.15, 2.0) if dtype in (torch.bfloat16, torch.float16) else (1e-2, 1e-2)
            for a, b, tag in (
                (y_seq, y_grp, "seq vs grouped"),
                (y_seq, y_bat, "seq vs batched"),
                (y_grp, y_bat, "grouped vs batched"),
            ):
                if not torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol):
                    raise AssertionError(
                        f"{tag} mismatch max_abs={max_abs(a, b)} device={device} alloc={alloc} dtype={dtype}"
                    )

            def wrap_seq() -> torch.Tensor:
                return sequential_expert_mm(x, expert_ids, weight)

            def wrap_grp() -> torch.Tensor:
                return grouped_mm_dispatch(x, expert_ids, weight)

            def wrap_bat() -> torch.Tensor:
                return batched_mm_dispatch(x, expert_ids, weight)

            t_seq, _ = bench(wrap_seq, device, args.warmup, args.iters)
            t_grp, _ = bench(wrap_grp, device, args.warmup, args.iters)
            t_bat, _ = bench(wrap_bat, device, args.warmup, args.iters)

            rows.append(
                {
                    "device": device.type,
                    "alloc": alloc,
                    "min_tokens_per_expert": min_tok,
                    "max_tokens_per_expert": max_tok,
                    "std_tokens_per_expert": round(std_tok, 2),
                    "ms_seq": round(t_seq * 1000, 4),
                    "ms_grp": round(t_grp * 1000, 4),
                    "ms_bat": round(t_bat * 1000, 4),
                    "grp_vs_seq": round(t_grp / t_seq, 3),
                    "bat_vs_seq": round(t_bat / t_seq, 3),
                }
            )

    print("=== Expert matmul dispatch benchmark ===")
    print(
        f"config: E={args.experts} S={args.pairs} H={args.hidden} N={args.intermediate} "
        f"dtype={args.dtype} warmup={args.warmup} iters={args.iters}"
    )
    print()
    hdr = (
        f"{'device':<6} {'alloc':<8} {'min/exp':>8} {'max/exp':>8} {'std/exp':>8} "
        f"{'seq_ms':>10} {'grp_ms':>10} {'bat_ms':>10} {'grp/seq':>8} {'bat/seq':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['device']:<6} {r['alloc']:<8} {r['min_tokens_per_expert']:>8} {r['max_tokens_per_expert']:>8} "
            f"{r['std_tokens_per_expert']:>8} {r['ms_seq']:>10} {r['ms_grp']:>10} {r['ms_bat']:>10} "
            f"{r['grp_vs_seq']:>8} {r['bat_vs_seq']:>8}"
        )

    # Also print a second micro-benchmark: **core-only** (exclude sort/gather) to separate algorithm vs memory traffic
    print()
    print("=== Core-only (matmul / bmm only; same tensors, no sort, no index gather) ===")
    core_rows = []
    for device in devices:
        x = torch.randn(args.pairs, args.hidden, device=device, dtype=dtype)
        weight = torch.randn(args.experts, args.hidden, args.intermediate, device=device, dtype=dtype)
        # Pre-sorted uniform expert ids -> contiguous groups
        expert_ids = make_expert_ids_cpu(args.pairs, args.experts, "uniform", gen).to(device)
        perm = torch.argsort(expert_ids)
        x_g = x[perm]
        e_sorted = expert_ids[perm]
        offs = torch.cumsum(torch.bincount(e_sorted, minlength=args.experts), dim=0).to(torch.int32)
        w_bat = weight[expert_ids]  # pre-gather for batched core

        def core_grouped() -> torch.Tensor:
            assert _grouped_mm is not None
            return _grouped_mm(x_g, weight, offs=offs)

        def core_batched() -> torch.Tensor:
            return torch.bmm(x.unsqueeze(1), w_bat).squeeze(1)

        t_cg, _ = bench(core_grouped, device, args.warmup, args.iters)
        t_cb, _ = bench(core_batched, device, args.warmup, args.iters)
        core_rows.append((device.type, round(t_cg * 1000, 4), round(t_cb * 1000, 4), round(t_cb / t_cg, 3)))

    print(f"{'device':<6} {'grouped_core_ms':>16} {'bmm_core_ms':>14} {'bmm/grouped':>12}")
    for d, cg, cb, ratio in core_rows:
        print(f"{d:<6} {cg:>16} {cb:>14} {ratio:>12}")


if __name__ == "__main__":
    main()
