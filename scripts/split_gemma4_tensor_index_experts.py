#!/usr/bin/env python3
"""
Split fused Gemma4 MoE expert entries in a tensor index JSON into per-expert rows.

Each fused block like::
  model.language_model.layers.L.experts.gate_up_proj
  model.language_model.layers.L.experts.down_proj

becomes ``num_experts`` tensors named (default)::
  model.language_model.layers.L.experts.{e}.gate_up_proj.weight
  model.language_model.layers.L.experts.{e}.down_proj.weight

Offsets and strides follow the leading expert dimension of the packed tensor.
Shapes and dtype are checked against ``config.json`` (``text_config`` when present).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

FUSED_GATE = "gate_up_proj"
FUSED_DOWN = "down_proj"

_RE_EXPERTS = re.compile(
    r"^(.+\.layers\.(?P<layer>\d+)\.)experts\.(?P<which>gate_up_proj|down_proj)$"
)


def _dtype_str_from_config(cfg: dict[str, Any]) -> str:
    text_cfg = cfg.get("text_config") or {}
    name = text_cfg.get("dtype") or cfg.get("dtype") or "bfloat16"
    if isinstance(name, str) and name.startswith("torch."):
        return name
    mapping = {
        "bfloat16": "torch.bfloat16",
        "float16": "torch.float16",
        "float32": "torch.float32",
        "float64": "torch.float64",
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype in config: {name!r} (extend mapping if needed)")
    return mapping[name]


def _expected_shapes(*, hidden: int, moe_inter: int, num_experts: int) -> dict[str, tuple[int, ...]]:
    gate_rows = 2 * moe_inter
    return {
        FUSED_GATE: (num_experts, gate_rows, hidden),
        FUSED_DOWN: (num_experts, hidden, moe_inter),
    }


def _per_expert_meta(
    fused: list[Any], *, which: str
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    """Return (byte_offset, byte_size, shape2d, stride2d) for one expert slice."""
    off_b, size_b, shape, stride, _dtype = fused
    if len(shape) != 3 or len(stride) != 3:
        raise ValueError(f"Expected rank-3 tensor metadata, got shape={shape}, stride={stride}")
    elem_size = size_b // (shape[0] * shape[1] * shape[2])
    if elem_size * shape[0] * shape[1] * shape[2] != size_b:
        raise ValueError(f"Inconsistent byte size vs shape: size={size_b}, shape={shape}")
    e0 = shape[0]
    slice_shape = (shape[1], shape[2])
    # Stride of slice [s1, s2] within dim0 index i: offset_elements = i * stride[0]
    stride2 = (stride[1], stride[2])
    slice_bytes = elem_size * slice_shape[0] * slice_shape[1]
    return int(off_b), int(slice_bytes), slice_shape, stride2


def split_tensor_index(
    index: dict[str, Any],
    *,
    num_experts: int,
    hidden: int,
    moe_inter: int,
    dtype_str: str,
    name_template: str,
) -> dict[str, Any]:
    expected = _expected_shapes(hidden=hidden, moe_inter=moe_inter, num_experts=num_experts)
    out: dict[str, Any] = {}
    removed = 0
    added = 0

    for key, fused in index.items():
        m = _RE_EXPERTS.match(key)
        if not m:
            out[key] = fused
            continue
        which = m.group("which")
        layer = m.group("layer")

        exp_shape = expected[which]
        if tuple(fused[2]) != exp_shape:
            raise ValueError(
                f"{key}: shape {fused[2]} does not match config expectation {exp_shape} "
                f"(hidden_size={hidden}, moe_intermediate_size={moe_inter}, num_experts={num_experts})"
            )
        if fused[4] != dtype_str:
            raise ValueError(f"{key}: dtype {fused[4]!r} != config dtype {dtype_str!r}")

        off0_b, slice_bytes, shape2, stride2 = _per_expert_meta(fused, which=which)
        elem_size = slice_bytes // (shape2[0] * shape2[1])
        stride0_elems = fused[3][0]

        for e in range(num_experts):
            new_key = name_template.format(layer=layer, expert=e, which=which)
            off_e = off0_b + e * stride0_elems * elem_size
            out[new_key] = [
                off_e,
                slice_bytes,
                list(shape2),
                list(stride2),
                dtype_str,
            ]
            added += 1
        removed += 1

    if removed == 0:
        raise ValueError("No fused expert keys matched pattern *.layers.N.experts.(gate_up_proj|down_proj)")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--index-in",
        type=Path,
        required=True,
        help="Input tensor index JSON (e.g. tensor_index.json or tensor_index_resize.json).",
    )
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        help="config.json (uses text_config for MoE fields when present).",
    )
    ap.add_argument(
        "--index-out",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    ap.add_argument(
        "--name-template",
        default="model.language_model.layers.{layer}.experts.{expert}.{which}.weight",
        help="Format string for new keys; fields: layer, expert, which (gate_up_proj|down_proj).",
    )
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    text_cfg = cfg.get("text_config") or {}
    num_experts = int(text_cfg.get("num_experts", cfg.get("num_experts")))
    hidden = int(text_cfg.get("hidden_size", cfg.get("hidden_size")))
    moe_inter = int(text_cfg.get("moe_intermediate_size", cfg.get("moe_intermediate_size")))
    dtype_str = _dtype_str_from_config(cfg)

    index = json.loads(args.index_in.read_text())
    if not isinstance(index, dict):
        raise SystemExit("Top-level JSON must be an object (name -> metadata list).")

    out = split_tensor_index(
        index,
        num_experts=num_experts,
        hidden=hidden,
        moe_inter=moe_inter,
        dtype_str=dtype_str,
        name_template=args.name_template,
    )

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    args.index_out.write_text(json.dumps(out, indent=4) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.index_out} ({len(out)} keys). "
        f"Replaced fused expert blocks with {num_experts * 2} tensors per MoE layer."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
