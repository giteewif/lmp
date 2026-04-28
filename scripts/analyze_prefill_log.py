#!/usr/bin/env python3
"""
Parse LMP debug logs: multiple prefill runs with ``end prefill_layer cost`` and ``prefill time``.

For each run, optional outlier replacement on per-layer times, then scaled ``prefill time`` estimate.

Example::

    python scripts/analyze_prefill_log.py examples/generate_cpu_sanityspread0.6_11.log
    python scripts/analyze_prefill_log.py /path/to.log --threshold-ms 200 --keep-layer0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_log(text: str) -> tuple[list[float], list[float], list[float]]:
    """Returns (layer_ms_in_order, prefill_time_seconds, prefill_wall_ms)."""
    layer_pat = re.compile(r"end prefill_layer cost ([\d.]+) ms")
    time_pat = re.compile(r"prefill time: ([\d.]+) seconds")
    wall_pat = re.compile(r"end prefill cost ([\d.]+) ms")
    layers = [float(m.group(1)) for m in layer_pat.finditer(text)]
    times = [float(m.group(1)) for m in time_pat.finditer(text)]
    walls = [float(m.group(1)) for m in wall_pat.finditer(text)]
    return layers, times, walls


def chunk_layers(layers: list[float], n_runs: int) -> list[list[float]]:
    if len(layers) % n_runs != 0:
        raise ValueError(
            f"layer count {len(layers)} not divisible by run count {n_runs}; "
            "pass --num-layers explicitly if the log mixes other hooks."
        )
    n_layers = len(layers) // n_runs
    return [layers[i * n_layers : (i + 1) * n_layers] for i in range(n_runs)]


def fix_layers(
    layer_ms: list[float],
    *,
    threshold_ms: float,
    keep_layer0: bool,
) -> tuple[list[float], list[int], float]:
    """
    Replace layers with cost > threshold_ms by the mean of ``good`` layers.

    ``good`` = indices not forced to keep and with time <= threshold.
    Returns (fixed_layers, outlier_indices, replacement_mean).
    """
    n = len(layer_ms)
    outlier_idx: list[int] = []
    for i, t in enumerate(layer_ms):
        if keep_layer0 and i == 0:
            continue
        if t > threshold_ms:
            outlier_idx.append(i)
    good_vals = [
        layer_ms[i]
        for i in range(n)
        if i not in outlier_idx and (not (keep_layer0 and i == 0) or True)
        and layer_ms[i] <= threshold_ms
    ]
    if not good_vals:
        good_vals = list(layer_ms)
    mu = sum(good_vals) / len(good_vals)
    fixed = [layer_ms[i] if i not in outlier_idx else mu for i in range(n)]
    return fixed, outlier_idx, mu


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze prefill_layer / prefill time from LMP logs.")
    ap.add_argument("log_path", type=Path, help="Path to .log file")
    ap.add_argument(
        "--threshold-ms",
        type=float,
        default=150.0,
        help="Layers above this (ms) are replaced by the mean of non-outlier layers (default 150).",
    )
    ap.add_argument(
        "--keep-layer0",
        action="store_true",
        help="Never treat layer 0 as an outlier (keep cold-start cost).",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Layers per prefill; default inferred as len(layers)//len(runs).",
    )
    args = ap.parse_args()

    path: Path = args.log_path
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8", errors="replace")
    layers, prefill_s, wall_ms = parse_log(text)
    if not prefill_s:
        print("error: no 'prefill time:' lines found", file=sys.stderr)
        return 1
    if len(wall_ms) != len(prefill_s):
        print(
            f"warning: prefill time lines ({len(prefill_s)}) != end prefill cost lines ({len(wall_ms)})",
            file=sys.stderr,
        )

    n_runs = len(prefill_s)
    if args.num_layers is not None:
        n_layers = int(args.num_layers)
        if len(layers) != n_runs * n_layers:
            raise SystemExit(
                f"error: expected {n_runs * n_layers} layer lines, got {len(layers)}"
            )
        runs = [layers[i * n_layers : (i + 1) * n_layers] for i in range(n_runs)]
    else:
        if len(layers) % n_runs != 0:
            raise SystemExit(
                f"error: {len(layers)} layer timings / {n_runs} runs not an integer; "
                "set --num-layers explicitly."
            )
        n_layers = len(layers) // n_runs
        runs = chunk_layers(layers, n_runs)

    print(f"file: {path}")
    print(f"runs: {n_runs}, layers per run: {n_layers}, threshold: {args.threshold_ms} ms")
    print(f"keep_layer0: {args.keep_layer0}")
    print()

    corrected_scaled: list[float] = []
    for i, run_layers in enumerate(runs, start=1):
        s_orig = sum(run_layers)
        fixed, bad_idx, mu = fix_layers(
            run_layers,
            threshold_ms=args.threshold_ms,
            keep_layer0=args.keep_layer0,
        )
        s_fix = sum(fixed)
        raw_t = prefill_s[i - 1] if i - 1 < len(prefill_s) else float("nan")
        wall = wall_ms[i - 1] if i - 1 < len(wall_ms) else float("nan")
        scaled = raw_t * (s_fix / s_orig) if s_orig > 0 else raw_t

        print(f"--- run {i} ---")
        print(f"  prefill time (log):     {raw_t:.6f} s")
        print(f"  end prefill cost:       {wall:.3f} ms")
        print(f"  sum(prefill_layer):     {s_orig:.3f} ms")
        if bad_idx:
            print(f"  outliers (> {args.threshold_ms} ms, 0-based idx): {bad_idx}")
            print(f"    values (ms):         {[run_layers[j] for j in bad_idx]}")
            print(f"  replacement mean:       {mu:.3f} ms")
        else:
            print("  outliers:               (none)")
        print(f"  sum after replace:      {s_fix:.3f} ms")
        print(f"  prefill (scaled):       {scaled:.6f} s   (= raw * sum_fix/sum_orig)")
        print()
        corrected_scaled.append(scaled)

    m_raw = sum(prefill_s) / len(prefill_s)
    m_cor = sum(corrected_scaled) / len(corrected_scaled)
    print("=== summary ===")
    print(f"mean prefill time (raw):     {m_raw:.6f} s")
    print(f"mean prefill time (scaled):  {m_cor:.6f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
