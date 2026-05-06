"""vLLM Triton fused-MoE decode with per-device CUDA-graph capture.

Design
------
``layer_moe_fused_decode_gpu`` currently has three GPU-blocking bottlenecks:

  1. ``bc.cumsum(0).cpu().tolist()``   – D→H transfer, serialises the step.
  2. Python loops building routing maps – cannot reside in a CUDA graph.
  3. BMM pad→GEMM→unpad per expert    – suboptimal GPU utilisation.

This module replaces all three with vLLM's Triton ``fused_experts`` kernel,
which does sorting / routing / GEMM entirely on-device.  When
``LMP_VLLM_MOE_CG=1`` the kernel is additionally captured in a per-device
``torch.cuda.CUDAGraph`` so that replay overhead is minimal.

Multi-GPU support
-----------------
Expert weights are spread across multiple GPUs.  We call ``fused_experts``
once per GPU with that GPU's local weight slice and an ``expert_map`` tensor
that maps global expert IDs → local indices (-1 = absent on this device).
The partial outputs are summed on the primary device.

Weight format (consistent with _prepare_fused_expert_work_items)
-----------------------------------------------------------------
  gate_up_packed : [local_E, 2*ffn_dim, H]  == vLLM w1
  down_packed    : [local_E,   H, ffn_dim]  == vLLM w2

Environment variables
---------------------
  LMP_VLLM_MOE=1      Use this path in layer_moe_fused_decode_gpu.
  LMP_VLLM_MOE_CG=1   Also capture in CUDA graph (requires LMP_VLLM_MOE=1).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# vLLM import helpers
# ---------------------------------------------------------------------------

def _import_fused_experts() -> Optional[Callable]:
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import (  # type: ignore[import]
            fused_experts,
        )
        return fused_experts
    except Exception as exc:
        logger.debug("vllm fused_experts unavailable: %s", exc)
        return None


def _import_moe_activation() -> Any:
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import (  # type: ignore[import]
            MoEActivation,
        )
        return MoEActivation
    except Exception:
        return None


def resolve_moe_activation(act_fn: Callable) -> Any:
    """Map a transformers ACT2FN callable → vLLM MoEActivation (default SILU)."""
    MoEActivation = _import_moe_activation()
    if MoEActivation is None:
        return None
    name = (
        getattr(act_fn, "__name__", "")
        or getattr(type(act_fn), "__name__", "")
        or ""
    ).lower()
    if "silu" in name or "swish" in name:
        return MoEActivation.SILU
    if "gelu" in name:
        return MoEActivation.GELU
    if "relu" in name and "gelu" not in name:
        return MoEActivation.RELU
    logger.warning("Cannot map act_fn %r to MoEActivation; defaulting to SILU", act_fn)
    return MoEActivation.SILU


def build_expert_map(
    global_num_experts: int,
    local_expert_global_ids: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Return a [global_E] int32 tensor: entry i = local index, or -1."""
    em = torch.full((global_num_experts,), -1, dtype=torch.int32, device=device)
    for local_idx, global_id in enumerate(local_expert_global_ids):
        em[int(global_id)] = local_idx
    return em


# ---------------------------------------------------------------------------
# CUDA-graph bundle for one device
# ---------------------------------------------------------------------------

class _DeviceMoeCGBundle:
    """Captures ``fused_experts(...)`` for one device in a single CUDAGraph.

    Static buffers
    --------------
      s_hidden   [T, H]   input  (filled before each replay)
      s_topk_w   [T, k]   topk weights
      s_topk_ids [T, k]   topk expert IDs (global)
      s_out      [T, H]   partial output from this device's experts

    After replay ``s_out`` holds the partial contribution; caller sums across
    all devices on the primary device.
    """

    __slots__ = (
        "graph", "stream", "device",
        "s_hidden", "s_topk_w", "s_topk_ids", "s_out",
    )

    @classmethod
    def try_capture(
        cls,
        stream: torch.cuda.Stream,
        hidden_tmpl: torch.Tensor,      # [T, H] on target device
        topk_w_tmpl: torch.Tensor,      # [T, k]
        topk_ids_tmpl: torch.Tensor,    # [T, k]
        w1: torch.Tensor,               # [local_E, 2*ffn, H]
        w2: torch.Tensor,               # [local_E, H, ffn]
        global_num_experts: int,
        expert_map: torch.Tensor,       # [global_E] int32
        moe_act: Any,
    ) -> Optional["_DeviceMoeCGBundle"]:
        fused_experts = _import_fused_experts()
        if fused_experts is None or not hidden_tmpl.is_cuda:
            return None

        device = hidden_tmpl.device
        torch.cuda.set_device(device)

        sh = torch.zeros_like(hidden_tmpl)
        sw = torch.zeros_like(topk_w_tmpl)
        si = torch.zeros_like(topk_ids_tmpl)
        so = torch.zeros_like(hidden_tmpl)

        graph = torch.cuda.CUDAGraph()
        try:
            # ── warmup on capture stream (satisfies CUDAGraph pre-conditions) ──
            with torch.cuda.stream(stream):
                _partial = fused_experts(
                    sh, w1, w2, sw, si,
                    inplace=False,
                    activation=moe_act,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                )
                so.copy_(_partial)
            stream.synchronize()

            # ── capture ──
            so.zero_()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph):
                    _partial = fused_experts(
                        sh, w1, w2, sw, si,
                        inplace=False,
                        activation=moe_act,
                        global_num_experts=global_num_experts,
                        expert_map=expert_map,
                    )
                    so.copy_(_partial)
            stream.synchronize()

            bundle = cls.__new__(cls)
            bundle.graph = graph
            bundle.stream = stream
            bundle.device = device
            bundle.s_hidden = sh
            bundle.s_topk_w = sw
            bundle.s_topk_ids = si
            bundle.s_out = so
            logger.debug("_DeviceMoeCGBundle captured on %s", device)
            return bundle

        except Exception as exc:
            logger.warning("_DeviceMoeCGBundle capture failed on %s: %s", device, exc)
            try:
                graph.reset()
            except Exception:
                pass
            return None

    def replay(
        self,
        hidden: torch.Tensor,    # [T, H] – already on self.device
        topk_w: torch.Tensor,    # [T, k]
        topk_ids: torch.Tensor,  # [T, k]
    ) -> torch.Tensor:
        """Fill static buffers, replay graph, return partial output [T, H]."""
        with torch.cuda.stream(self.stream):
            self.s_hidden.copy_(hidden, non_blocking=True)
            self.s_topk_w.copy_(topk_w, non_blocking=True)
            self.s_topk_ids.copy_(topk_ids, non_blocking=True)
            self.graph.replay()
        return self.s_out  # [T, H] on self.device


# ---------------------------------------------------------------------------
# Per-device state for one layer
# ---------------------------------------------------------------------------

class _DeviceState:
    """All static data for one device × one layer."""

    __slots__ = (
        "dev_idx", "device", "w1", "w2", "expert_map",
        "stream", "bundle",
        "transfer_event", "replay_event",
    )

    def __init__(
        self,
        dev_idx: int,
        device: torch.device,
        w1: torch.Tensor,
        w2: torch.Tensor,
        expert_map: torch.Tensor,
        stream: torch.cuda.Stream,
        bundle: Optional[_DeviceMoeCGBundle] = None,
    ):
        self.dev_idx = dev_idx
        self.device = device
        self.w1 = w1
        self.w2 = w2
        self.expert_map = expert_map
        self.stream = stream
        self.bundle = bundle
        # Pre-allocated CUDA events for fine-grained stream synchronisation
        # (avoids full device synchronise in the CUDAGraph replay path).
        #   transfer_event: recorded on primary stream after H2D input copies;
        #                   ds.stream waits on it before replay.
        #   replay_event:   recorded on ds.stream after graph replay;
        #                   primary stream waits on it before reading s_out.
        self.transfer_event: torch.cuda.Event = torch.cuda.Event()
        self.replay_event:   torch.cuda.Event = torch.cuda.Event()

    def try_build_graph(
        self,
        hidden_tmpl: torch.Tensor,   # on primary device – will be moved to self.device
        topk_w_tmpl: torch.Tensor,
        topk_ids_tmpl: torch.Tensor,
        moe_act: Any,
        global_num_experts: int,
    ) -> bool:
        if self.bundle is not None:
            return True
        h = hidden_tmpl.to(self.device)
        tw = topk_w_tmpl.to(self.device)
        ti = topk_ids_tmpl.to(self.device)
        self.bundle = _DeviceMoeCGBundle.try_capture(
            self.stream, h, tw, ti,
            self.w1, self.w2,
            global_num_experts, self.expert_map, moe_act,
        )
        return self.bundle is not None


# ---------------------------------------------------------------------------
# VllmMoeDecodeManager – one instance per MLPLLM
# ---------------------------------------------------------------------------

class VllmMoeDecodeManager:
    """Per-(layer, device) state manager for vLLM fused_experts decode.

    Lifecycle
    ---------
    1. Constructed once in ``MLPLLM.__init__``.
    2. On first decode call for layer L: ``forward()`` calls ``_build_layer``
       to extract w1/w2/expert_map from ``experts_state_dict_slices_packed``.
    3. On subsequent calls: optionally captures / replays CUDAGraph.

    Thread-safety
    -------------
    Not thread-safe; designed to be called from the main inference thread only.
    """

    def __init__(self, cuda_graph: bool = False):
        self._cg_enabled = cuda_graph
        self._device_states: Dict[int, List[_DeviceState]] = {}
        self._streams: Dict[int, torch.cuda.Stream] = {}
        self._fused_experts: Optional[Callable] = _import_fused_experts()
        # Tracks which (dev_idx, w1_shape, dtype) combos already had Triton compiled.
        # Avoids redundant warmup calls across layers with identical weight shapes.
        self._warmed_up_shapes: set = set()
        if self._fused_experts is None:
            raise RuntimeError(
                "vllm.fused_experts not importable – ensure vllm is installed."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_stream(self, dev_idx: int) -> torch.cuda.Stream:
        s = self._streams.get(dev_idx)
        if s is None:
            s = torch.cuda.Stream(device=f"cuda:{dev_idx}")
            self._streams[dev_idx] = s
        return s

    def _build_layer(
        self,
        layer_idx: int,
        experts_state_dict_slices_packed: dict,
        full_device_expert_map: Dict[int, List[int]],
        global_num_experts: int,
    ) -> None:
        """Build _DeviceState list for *all* devices (not just activated ones).

        Only called once per layer; subsequent calls use ``_refresh_layer_weights``
        to update w1/w2 without discarding the CUDA-graph bundle.
        """
        packed_by_ld = (
            experts_state_dict_slices_packed.get("packed_by_layer_device", {}) or {}
        )
        layer_packed = packed_by_ld.get(layer_idx, {})
        states: List[_DeviceState] = []

        for dev_idx, global_expert_ids in sorted(full_device_expert_map.items()):
            packed_ent = layer_packed.get(int(dev_idx))
            if not isinstance(packed_ent, dict):
                continue
            gate_up = packed_ent.get("gate_up_packed")
            down = packed_ent.get("down_packed")
            if gate_up is None or down is None:
                continue

            # Use the target CUDA device from full_device_expert_map, NOT gate_up.device,
            # because weights may still be on CPU here (not yet paged in).
            target_device = torch.device(f"cuda:{int(dev_idx)}")
            e_rows = int(gate_up.size(0))
            exp_rows = packed_ent.get("experts")
            if isinstance(exp_rows, list) and len(exp_rows) == e_rows:
                local_order = [int(x) for x in exp_rows]
            else:
                local_order = sorted(int(e) for e in global_expert_ids)

            expert_map = build_expert_map(global_num_experts, local_order, target_device)
            stream = self._get_stream(int(dev_idx))

            # w1 = gate_up_packed [local_E, 2*ffn, H]  ← vLLM w1
            # w2 = down_packed    [local_E,   H, ffn]  ← vLLM w2
            # Ensure weights are on CUDA; if paging hasn't finished yet they will be
            # refreshed by _refresh_layer_weights on the first actual forward call.
            states.append(
                _DeviceState(
                    dev_idx=int(dev_idx),
                    device=target_device,
                    w1=gate_up.contiguous(),
                    w2=down.contiguous(),
                    expert_map=expert_map,
                    stream=stream,
                )
            )

        self._device_states[layer_idx] = states

        # ── Auto-warmup: trigger Triton JIT compilation now, not on first inference ──
        # We fire a tiny dummy forward to compile the Triton fused_moe_kernel while
        # _build_layer runs, so the first real _fused_experts call hits the cache.
        # We skip any (dev_idx, shape, dtype) that has already been compiled to avoid
        # redundant work across layers with identical weight shapes.
        MoEActivation = _import_moe_activation()
        silu_act = MoEActivation.SILU if MoEActivation is not None else None
        for ds in states:
            if not ds.w1.is_cuda or silu_act is None:
                continue  # weights not yet on GPU, or vLLM activation enum unavailable
            local_e, dim2h, H = ds.w1.shape
            ffn = dim2h // 2
            warmup_key = (ds.dev_idx, local_e, H, ffn, ds.w1.dtype)
            if warmup_key in self._warmed_up_shapes:
                continue  # kernel already compiled for this shape on this device
            dummy_h   = torch.zeros(1, H, dtype=ds.w1.dtype, device=ds.device)
            dummy_ids = torch.zeros(1, 1, dtype=torch.int64, device=ds.device)
            dummy_w   = torch.ones(1, 1, dtype=ds.w1.dtype, device=ds.device)
            try:
                with torch.cuda.stream(ds.stream):
                    self._fused_experts(
                        dummy_h, ds.w1, ds.w2, dummy_w, dummy_ids,
                        inplace=False,
                        activation=silu_act,
                        global_num_experts=global_num_experts,
                        expert_map=ds.expert_map,
                    )
                torch.cuda.synchronize(ds.device)
                self._warmed_up_shapes.add(warmup_key)
                logger.info(
                    "VllmMoeDecodeManager: auto-warmup compiled Triton kernel "
                    "for layer=%s dev=%s (E=%d H=%d FFN=%d dtype=%s).",
                    layer_idx, ds.dev_idx, local_e, H, ffn, ds.w1.dtype,
                )
            except Exception as exc:
                logger.warning(
                    "VllmMoeDecodeManager: auto-warmup failed for layer=%s dev=%s "
                    "(non-fatal, will compile on first inference): %s",
                    layer_idx, ds.dev_idx, exc,
                )

    def _refresh_layer_weights(
        self,
        layer_idx: int,
        experts_state_dict_slices_packed: dict,
    ) -> None:
        """Refresh w1/w2 tensors for every cached _DeviceState from the latest
        ``experts_state_dict_slices_packed``.

        Called on every ``forward`` to handle the case where the weight-paging
        system has moved weights back to CPU (eviction) and then reloaded them
        to a new GPU allocation.  The CUDA-graph bundle is invalidated when the
        weight data pointer changes so a new capture will be triggered on the
        next decode step.
        """
        states = self._device_states.get(layer_idx)
        if not states:
            return
        packed_by_ld = (
            experts_state_dict_slices_packed.get("packed_by_layer_device", {}) or {}
        )
        layer_packed = packed_by_ld.get(layer_idx, {})
        for ds in states:
            packed_ent = layer_packed.get(ds.dev_idx)
            if not isinstance(packed_ent, dict):
                continue
            gate_up = packed_ent.get("gate_up_packed")
            down = packed_ent.get("down_packed")
            if gate_up is None or down is None:
                continue
            new_w1 = gate_up.contiguous()
            new_w2 = down.contiguous()
            # Invalidate CUDA-graph bundle if the weight pointer changed
            # (weights were evicted and reloaded to a different GPU address).
            if ds.bundle is not None and (
                new_w1.data_ptr() != ds.w1.data_ptr()
                or new_w2.data_ptr() != ds.w2.data_ptr()
            ):
                logger.debug(
                    "VllmMoeDecodeManager: weight pointer changed for layer=%s dev=%s, "
                    "invalidating CUDA graph bundle.",
                    layer_idx,
                    ds.dev_idx,
                )
                ds.bundle = None
            ds.w1 = new_w1
            ds.w2 = new_w2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        layer_idx: int,
        hidden: torch.Tensor,                    # [T, H] on primary_device
        topk_w: torch.Tensor,                    # [T, k]
        topk_ids: torch.Tensor,                  # [T, k] – global expert IDs
        experts_state_dict_slices_packed: dict,
        full_device_expert_map: Dict[int, List[int]],
        global_num_experts: int,
        moe_act: Any,
        primary_device: torch.device,
        skip_cuda_graph: bool = False,
    ) -> torch.Tensor:
        """Return routed expert output [T, H] on primary_device.

        Replaces the entire routing-loop + BMM block in
        ``layer_moe_fused_decode_gpu`` with one Triton kernel per device.

        Args:
            skip_cuda_graph: When ``True`` the CUDA-graph replay is bypassed
                even if ``LMP_VLLM_MOE_CG=1``.  Pass ``True`` for prefill
                (variable token count) and ``False`` for decode (fixed T=batch).
        """
        # ── Lazy build device states ──
        if layer_idx not in self._device_states:
            self._build_layer(
                layer_idx,
                experts_state_dict_slices_packed,
                full_device_expert_map,
                global_num_experts,
            )
        # Always refresh w1/w2 – weights may have been paged back to CPU and
        # reloaded to a new GPU address since the last forward call.
        self._refresh_layer_weights(layer_idx, experts_state_dict_slices_packed)
        states = self._device_states.get(layer_idx, [])
        if not states:
            raise RuntimeError(
                f"VllmMoeDecodeManager: no device state for layer {layer_idx}. "
                "Check expert allocation / experts_state_dict_slices_packed."
            )

        # Normalise topk_w/topk_ids to [T, k] and correct dtype
        T = hidden.shape[0]
        k = topk_ids.numel() // T
        topk_ids_2d = topk_ids.view(T, k).to(dtype=torch.int64, device=primary_device)
        topk_w_2d  = topk_w.view(T, k).to(dtype=hidden.dtype, device=primary_device)

        out = torch.zeros_like(hidden)

        use_cg = self._cg_enabled and not skip_cuda_graph

        for ds in states:
            # Move inputs to device
            h_dev  = hidden.to(ds.device, non_blocking=True)
            tw_dev = topk_w_2d.to(ds.device, non_blocking=True)
            ti_dev = topk_ids_2d.to(ds.device, non_blocking=True)

            if use_cg:
                # ── Try to capture on first call, replay on subsequent ──
                if ds.bundle is None:
                    ds.try_build_graph(
                        hidden, topk_w_2d, topk_ids_2d, moe_act, global_num_experts
                    )
                if ds.bundle is not None:
                    # ── Event-based synchronisation (no CPU stall) ──────────
                    # 1. Record on primary stream after the non_blocking H2D
                    #    transfers above; ds.stream waits before touching inputs.
                    ds.transfer_event.record(torch.cuda.current_stream(primary_device))
                    ds.stream.wait_event(ds.transfer_event)
                    # 2. Replay graph on ds.stream (inputs now guaranteed ready).
                    partial = ds.bundle.replay(h_dev, tw_dev, ti_dev)
                    # 3. Record on ds.stream after replay; primary stream waits
                    #    before reading s_out / transferring back.
                    ds.replay_event.record(ds.stream)
                    torch.cuda.current_stream(primary_device).wait_event(ds.replay_event)
                    out.add_(partial.to(primary_device, non_blocking=True))
                    continue

            # ── Eager Triton path (prefill, or when graph capture failed) ──
            if not ds.w1.is_cuda:
                logger.warning(
                    "VllmMoeDecodeManager: w1 for layer=%s dev=%s is on %s "
                    "(paging not complete?); skipping device.",
                    layer_idx, ds.dev_idx, ds.w1.device,
                )
                continue
            with torch.cuda.stream(ds.stream):
                partial = self._fused_experts(
                    h_dev, ds.w1, ds.w2, tw_dev, ti_dev,
                    inplace=False,
                    activation=moe_act,
                    global_num_experts=global_num_experts,
                    expert_map=ds.expert_map,
                )
                # Non-blocking transfer back to primary device
                out.add_(partial.to(primary_device, non_blocking=True))

        # Single synchronisation point across all streams
        # torch.cuda.synchronize()
        return out

    def warmup_triton(
        self,
        layer_idx: int,
        hidden_size: int,
        ffn_size: int,
        global_num_experts: int,
        topk: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int = 1,
    ) -> None:
        """Pre-compile the vLLM Triton fused_moe kernel for a given shape.

        Call this once after weights are on GPU (e.g. after model loading) to
        pay the ~0.5–2 s Triton JIT cost upfront so that real inference
        requests see sub-10 ms kernel latency from the first token.

        Args:
            layer_idx:          layer index whose device state to use.  If the
                                layer state is not yet built, a temporary dummy
                                state is used so no ``experts_state_dict`` is
                                required.
            hidden_size:        model hidden dimension H.
            ffn_size:           intermediate (FFN) dimension I.
            global_num_experts: total number of experts in the model.
            topk:               number of active experts per token.
            dtype:              weight/activation dtype (e.g. ``torch.bfloat16``).
            device:             CUDA device to compile for.
            batch_size:         number of tokens in the dummy batch (1 is enough
                                for decode compilation; use a larger value if you
                                also want prefill shapes pre-compiled).
        """
        logger.info(
            "VllmMoeDecodeManager.warmup_triton: layer=%s H=%d FFN=%d E=%d k=%d "
            "dtype=%s device=%s batch=%d — compiling Triton fused_moe_kernel …",
            layer_idx, hidden_size, ffn_size, global_num_experts, topk,
            dtype, device, batch_size,
        )
        t0 = time.perf_counter()
        dev = torch.device(device)

        # Use cached device state if available, else build a temporary dummy.
        states = self._device_states.get(layer_idx)
        if states:
            ds_list = [ds for ds in states if ds.device == dev and ds.w1.is_cuda]
        else:
            ds_list = []

        if ds_list:
            ds = ds_list[0]
            local_e = int(ds.w1.size(0))
            expert_map = ds.expert_map
            w1 = ds.w1
            w2 = ds.w2
            stream = ds.stream
        else:
            # Build minimal dummy weights for compilation trigger only.
            local_e = max(1, global_num_experts)
            w1 = torch.zeros(local_e, ffn_size * 2, hidden_size, dtype=dtype, device=dev)
            w2 = torch.zeros(local_e, hidden_size, ffn_size,     dtype=dtype, device=dev)
            expert_map = build_expert_map(
                global_num_experts,
                list(range(local_e)),
                dev,
            )
            stream = self._get_stream(int(dev.index) if dev.index is not None else 0)

        T = batch_size
        dummy_hidden = torch.zeros(T, hidden_size, dtype=dtype, device=dev)
        # Route each token to expert 0 with equal weights.
        dummy_topk_ids = torch.zeros(T, topk, dtype=torch.int64, device=dev)
        dummy_topk_w   = torch.full((T, topk), 1.0 / topk, dtype=dtype, device=dev)

        MoEActivation = _import_moe_activation()
        silu_act = MoEActivation.SILU if MoEActivation is not None else None
        if silu_act is None:
            logger.warning("VllmMoeDecodeManager.warmup_triton: MoEActivation unavailable; skipping.")
            return
        try:
            with torch.cuda.stream(stream):
                self._fused_experts(
                    dummy_hidden, w1, w2, dummy_topk_w, dummy_topk_ids,
                    inplace=False,
                    activation=silu_act,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                )
            torch.cuda.synchronize(dev)
        except Exception as exc:
            logger.warning(
                "VllmMoeDecodeManager.warmup_triton: kernel warm-up failed "
                "(non-fatal): %s", exc,
            )
            return

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        logger.info(
            "VllmMoeDecodeManager.warmup_triton: done in %.1f ms", elapsed_ms
        )

    def invalidate_layer(self, layer_idx: int) -> None:
        """Discard cached state for layer_idx (e.g. after weight reload)."""
        self._device_states.pop(layer_idx, None)

    def invalidate_all(self) -> None:
        self._device_states.clear()


# ---------------------------------------------------------------------------
# Convenience: full-layer static-KV attention graph
# ---------------------------------------------------------------------------

class _StaticKVDecodeBundle:
    """CUDA-graph bundle for one decoder layer's attention + post-LN step.

    Captures (for decode, seq_len=1):
        input_LN → QKV_proj → RoPE → flash_attn_with_kvcache
        → O_proj → residual_add → post_attn_LN

    Uses ``flash_attn.flash_attn_with_kvcache`` which updates the static KV
    cache in-place and computes attention in a single kernel – no shape change.

    Static tensors
    --------------
      s_hidden      [B, 1, H]
      s_pos_ids     [B, 1]   (position of the current decode token)
      kv_k_cache    [B, num_kv_heads, max_seq, head_dim]
      kv_v_cache    [B, num_kv_heads, max_seq, head_dim]
      s_cache_seqlens  [B]  int32 – current position in KV cache
      s_ffn_skip    [B, 1, H]  output: residual + attn
      s_gate_in     [B, 1, H]  output: post_LN(residual + attn)
    """

    __slots__ = (
        "graph", "stream", "device",
        "s_hidden", "s_pos_ids", "s_cache_seqlens",
        "kv_k_cache", "kv_v_cache",
        "s_ffn_skip", "s_gate_in",
    )

    @classmethod
    def try_capture(
        cls,
        stream: torch.cuda.Stream,
        attn_layer,           # HF model layer (has q/k/v/o_proj, rotary_emb)
        hidden_tmpl: torch.Tensor,   # [B, 1, H]
        pos_ids_tmpl: torch.Tensor,  # [B, 1]
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        iln_fn: Callable,     # input_layernorm callable
        ffn_skip_fn: Callable,
        gate_in_fn: Callable,
    ) -> Optional["_StaticKVDecodeBundle"]:
        try:
            from flash_attn import flash_attn_with_kvcache  # type: ignore
        except Exception:
            return None

        if not hidden_tmpl.is_cuda:
            return None

        device = hidden_tmpl.device
        B = hidden_tmpl.shape[0]
        H = hidden_tmpl.shape[-1]
        dtype = hidden_tmpl.dtype
        torch.cuda.set_device(device)

        sh = torch.zeros_like(hidden_tmpl)
        sp = torch.zeros_like(pos_ids_tmpl)
        kk = torch.zeros(B, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        kv = torch.zeros(B, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        sc = torch.zeros(B, dtype=torch.int32, device=device)
        s_ffn = torch.zeros_like(hidden_tmpl)
        s_gin = torch.zeros_like(hidden_tmpl)

        graph = torch.cuda.CUDAGraph()
        try:
            def _run():
                h = iln_fn(sh)
                # QKV
                q = attn_layer.q_proj(h)
                k = attn_layer.k_proj(h)
                v = attn_layer.v_proj(h)
                # reshape to [B, 1, num_heads, head_dim]
                num_q_heads = q.shape[-1] // head_dim
                q = q.view(B, 1, num_q_heads, head_dim)
                k = k.view(B, 1, num_kv_heads, head_dim)
                v = v.view(B, 1, num_kv_heads, head_dim)
                # RoPE (rotary_emb expects [B, 1, H] input)
                cos, sin = attn_layer.rotary_emb(sh, sp)
                # flash_attn rotary-in-kvcache variant
                attn_out = flash_attn_with_kvcache(
                    q, kk, kv, k, v,
                    rotary_cos=cos,
                    rotary_sin=sin,
                    cache_seqlens=sc,
                    causal=True,
                )  # [B, 1, num_q_heads, head_dim]
                attn_out = attn_out.view(B, 1, -1)
                h_attn = attn_layer.o_proj(attn_out)
                s_ffn.copy_(ffn_skip_fn(sh, h_attn))
                s_gin.copy_(gate_in_fn(sh, h_attn))

            # warmup
            with torch.cuda.stream(stream):
                _run()
            stream.synchronize()

            s_ffn.zero_()
            s_gin.zero_()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph):
                    _run()
            stream.synchronize()

            bundle = cls.__new__(cls)
            bundle.graph = graph
            bundle.stream = stream
            bundle.device = device
            bundle.s_hidden = sh
            bundle.s_pos_ids = sp
            bundle.s_cache_seqlens = sc
            bundle.kv_k_cache = kk
            bundle.kv_v_cache = kv
            bundle.s_ffn_skip = s_ffn
            bundle.s_gate_in = s_gin
            logger.debug("_StaticKVDecodeBundle captured on %s", device)
            return bundle

        except Exception as exc:
            logger.warning("_StaticKVDecodeBundle capture failed on %s: %s", device, exc)
            try:
                graph.reset()
            except Exception:
                pass
            return None

    def step(
        self,
        hidden: torch.Tensor,    # [B, 1, H]
        pos_ids: torch.Tensor,   # [B, 1]
        cache_seqlens: torch.Tensor,  # [B] int32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replay graph; return (ffn_skip, gate_in)."""
        with torch.cuda.stream(self.stream):
            self.s_hidden.copy_(hidden, non_blocking=True)
            self.s_pos_ids.copy_(pos_ids, non_blocking=True)
            self.s_cache_seqlens.copy_(cache_seqlens, non_blocking=True)
            self.graph.replay()
        self.stream.synchronize()
        return self.s_ffn_skip, self.s_gate_in
