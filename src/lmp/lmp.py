from importlib import import_module
import atexit
import os
import re
import warnings
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional, Tuple
import copy
from lmp.pinpool import gpinpool
import torch
import time

# Transformers v5 起 ``modeling_attn_mask_utils`` 会发 FutureWarning；本仓库仍用旧辅助函数直至迁移到 ``masking_utils``。
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.modeling_attn_mask_utils",
)

from transformers import AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask
)

# sllm_store
from sllm_store._C import (
    allocate_cuda_memory,
    free_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors_from_shared_memory_names,
    restore_tensors2,
)
from sllm_store.client import SllmStoreClient

# lmp
from lmp.sllm_store_c import (
    SLLM_ADDRESS,
    STORAGE_PATH,
    TENSOR_INDEX_RESIZE_PATH,
    load_into_cpu,
    load_into_gpu_async,
)
from utils import cuda_h
from utils.cuda_h import cuda_hook, cuda_hook_end, cuda_hook_time, cuda_hook_time_end
from utils.logger import init_logger
from utils.helper import *
from models.mlpmodule import MLPModuleWrapper, ExpertEinsumTask, WeightType
from lmp.cuda_memory_view import (
    CudaMemoryView,
    HostMemoryView,
    _group_fused_select_half_experts,
    _group_fused_test_parse_half_modes,
)

logger = init_logger(__name__)


def _parse_positive_int_list_csv(raw: Optional[str]) -> Optional[list[int]]:
    """解析逗号/空格分隔的正整数列表，升序去重；空串返回 None。"""
    if raw is None or not str(raw).strip():
        return None
    vals: list[int] = []
    for p in str(raw).replace(",", " ").split():
        if not p.strip():
            continue
        v = int(p)
        if v <= 0:
            raise ValueError(f"LMP_MOE_CUDA_GRAPH_BUCKETS_MAX_T invalid bucket {v} (must be positive)")
        vals.append(v)
    return sorted(set(vals))


def _recover_cuda_rng_after_failed_cudagraph_capture() -> None:
    """捕获失败后尝试恢复默认 CUDA generator，防止后续 torch.randn 触发：
    "Offset increment outside graph capture encountered unexpectedly."（pytorch/pytorch#171263）。

    ``torch.cuda.manual_seed_all`` 不能直接重置 C++ 侧 ``capturing_`` flag，但
    可通过创建新 generator 并强制与默认 stream 同步，让 PyTorch 内部状态重新对齐。
    最终兜底：把每个设备的默认 generator seed 重新设置一次，触发 set_current_seed
    路径，部分 PyTorch 版本会在此时清理 capturing 状态。
    """
    try:
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.synchronize(f"cuda:{i}")
            except Exception:
                pass
        seed = (time.time_ns() ^ (os.getpid() << 16)) & 0x7FFF_FFFF
        torch.cuda.manual_seed_all(max(seed, 1))
    except Exception:
        pass


class _MoeBmmCoreCudaGraphBundle:
    """MoE BMM 路径中仅捕获 ``bmm → act × gate → bmm``；unpad/scatter 仍在 graph 外。

    缓存为多 bucket：`dict`` key 区分 layer / device / (E, max_t_bucket, H) / dtype / 权重指针。
    可选环境变量 ``LMP_MOE_CUDA_GRAPH_BUCKETS_MAX_T`` 将 ``max_t`` 向上取整到离散桶，减少捕获次数。
    """

    __slots__ = ("device", "stream", "graph", "stacked_static", "gate", "down", "gu", "mid", "y_st")

    def __init__(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
        graph: torch.cuda.CUDAGraph,
        stacked_static: torch.Tensor,
        gate: torch.Tensor,
        down: torch.Tensor,
        gu: torch.Tensor,
        mid: torch.Tensor,
        y_st: torch.Tensor,
    ):
        self.device = device
        self.stream = stream
        self.graph = graph
        self.stacked_static = stacked_static
        self.gate = gate
        self.down = down
        self.gu = gu
        self.mid = mid
        self.y_st = y_st

    @classmethod
    def try_capture(
        cls,
        stream: torch.cuda.Stream,
        stacked_template: torch.Tensor,
        gate: torch.Tensor,
        down: torch.Tensor,
        act_fn: Callable[..., torch.Tensor],
    ) -> Optional["_MoeBmmCoreCudaGraphBundle"]:
        if not stacked_template.is_cuda:
            return None
        device = stacked_template.device
        torch.cuda.set_device(device)
        stacked_static = torch.empty_like(stacked_template)
        e, max_t, _ = stacked_static.shape
        gu_dim = gate.shape[-1]
        half = gu_dim // 2
        out_h = down.shape[-1]
        dtype = stacked_static.dtype
        gu = torch.empty((e, max_t, gu_dim), device=device, dtype=dtype)
        mid = torch.empty((e, max_t, half), device=device, dtype=dtype)
        y_st = torch.empty((e, max_t, out_h), device=device, dtype=dtype)

        graph: Optional["torch.cuda.CUDAGraph"] = None
        try:
            stacked_static.copy_(stacked_template)
            torch.cuda.synchronize()
            # allocator / cudnn 热身，满足 CUDAGraph 捕获前置条件
            with torch.cuda.stream(stream):
                torch.bmm(stacked_static, gate, out=gu)
                g_half = gu[..., :half]
                u_half = gu[..., half:]
                activated = act_fn(g_half)
                torch.mul(activated, u_half, out=mid)
                torch.bmm(mid, down, out=y_st)
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            # 捕获前保存各 GPU 默认 generator 状态，以便失败时恢复。
            # torch.cuda.set_rng_state 不能重置 capturing_ flag，因此只在捕获
            # 后检查，出现异常则立即调用 _recover_cuda_rng_after_failed_cudagraph_capture()。
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph):
                    torch.bmm(stacked_static, gate, out=gu)
                    g_half = gu[..., :half]
                    u_half = gu[..., half:]
                    activated = act_fn(g_half)
                    torch.mul(activated, u_half, out=mid)
                    torch.bmm(mid, down, out=y_st)
            stream.synchronize()

            # 检测空 graph（wrong-stream 时 capture_end() 不抛异常，只发 UserWarning）：
            # 用一次 replay 后验证输出张量中是否有非零值作为代理。
            # 不能直接读 graph 内部节点数，改用 y_st 哨兵检测。
            y_st.zero_()
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            if not y_st.any():
                raise RuntimeError(
                    "MoE BMM CUDAGraph replay produced all-zero output; "
                    "graph likely empty (captured on wrong stream/device)."
                )

            return cls(device, stream, graph, stacked_static, gate, down, gu, mid, y_st)
        except Exception as exc:
            logger.warning("MoE BMM CUDAGraph capture failed on %s: %s", device, exc)
            if graph is not None:
                try:
                    graph.reset()
                except Exception:
                    pass
            _recover_cuda_rng_after_failed_cudagraph_capture()
            return None

    def copy_inputs_and_replay(self, stacked_inputs: torch.Tensor) -> None:
        torch.cuda.set_device(self.device)
        mt_in = int(stacked_inputs.size(1))
        bt = int(self.stacked_static.size(1))
        if int(stacked_inputs.size(0)) != int(self.stacked_static.size(0)) or int(stacked_inputs.size(2)) != int(
            self.stacked_static.size(2)
        ):
            raise RuntimeError(
                f"CUDAGraph stacked mismatch: in={tuple(stacked_inputs.shape)} bundle={tuple(self.stacked_static.shape)}"
            )
        if mt_in > bt:
            raise RuntimeError(
                f"CUDAGraph stacked max_t in={mt_in} exceeds bundle max_t={bt} (wrong bucket or cache key)"
            )
        with torch.cuda.stream(self.stream):
            self.stacked_static[:, :mt_in].copy_(stacked_inputs, non_blocking=True)
            if bt > mt_in:
                self.stacked_static[:, mt_in:].zero_()
            self.graph.replay()
        self.stream.synchronize()


class MLPLLM:
    def __init__(
        self,
        model_name_type: str,
        model_path: str,
        device_num: int = 4
    ):
        self.model_path = model_path
        self.model_name_type = model_name_type

        client = SllmStoreClient(SLLM_ADDRESS)
        self.client = client
        ret = load_into_cpu(client, model_path)
        if not ret:
            raise ValueError(f"Failed to load model {model_path} into CPU")
        device0 = "cuda:0"
        device1 = "cuda:1"
        device2 = "cuda:2"
        device3 = "cuda:3"
        device_list = [device0, device1, device2, device3]
        # device_list = [device0, device1]
        # device_list = [device0]
        device_list = device_list[:device_num]
        self.device1 = device_list[0]
        self.device_list = device_list

        # layer_moe_fused 里按 GPU 分 work_item，线程数上界为 len(device_list)；复用池避免每层创建 Executor。
        _nw = max(1, len(self.device_list))
        self._fused_expert_pool = ThreadPoolExecutor(
            max_workers=_nw,
            thread_name_prefix="lmp_fused_expert",
        )
        atexit.register(self._fused_expert_pool.shutdown, wait=True)

        mlpm = MLPModuleWrapper(model_name_type, model_path)
        self.mlpm  = mlpm
        
        cuda_hook_time("init_cmv_hmv")
        gpu_meta_model = self.mlpm.init_chmv_meta_model(device=self.device_list[0])
        cpu_meta_model = copy.deepcopy(gpu_meta_model)
        cuda_hook_time_end("init_cmv_hmv")


        tensor_index_resize_path = os.path.join(self.mlpm.model_abs_path, TENSOR_INDEX_RESIZE_PATH)
        tensor_index_resize_json = load_json(tensor_index_resize_path)
        self.tensor_index_json = tensor_index_resize_json
        
        self.cmv = CudaMemoryView(
            mlpm=self.mlpm, 
            meta_model=gpu_meta_model, 
            device_list=self.device_list, 
            client=self.client, 
            tensor_index_resize_json=tensor_index_resize_json
        )
        self.hmv = HostMemoryView(
            mlpm=self.mlpm, 
            meta_model=cpu_meta_model, 
            client=self.client, 
            tensor_index_resize_json=tensor_index_resize_json
        )
        
        self.moes, self.CPUInfer = self.init_kt_kernel()

        # Inverted index: (layer_id, expert_id) -> list[task_id].
        # Built eagerly once after generate_chunk() via _build_expert_task_id_index().
        self._expert_task_id_index: dict[tuple[int, int], list[int]] = {}

        # Per-layer expert allocation cache: layer_idx -> result of get_layer_experts_device_allocation.
        # Built eagerly once after predo_tensor_index_locate() via _build_layer_experts_alloc_index().
        self._layer_experts_alloc_cache: dict[int, dict] = {}

        # MoE fused BMM CUDAGraph：首次命中形状/权重指针时在主推理线程捕获，后续 ``replay``。
        self._moe_cuda_graph_enabled = os.environ.get("LMP_MOE_CUDA_GRAPH", "0").strip() == "1"
        self._moe_cuda_graph_buckets_max_t: Optional[list[int]] = _parse_positive_int_list_csv(
            os.environ.get("LMP_MOE_CUDA_GRAPH_BUCKETS_MAX_T")
        )
        self._moe_cuda_streams: dict[int, torch.cuda.Stream] = {}
        self._moe_bmm_cuda_graph_cache: dict[Tuple[Any, ...], _MoeBmmCoreCudaGraphBundle] = {}
        if self._moe_cuda_graph_enabled:
            for dev_str in self.device_list:
                d = torch.device(dev_str)
                self._moe_cuda_streams[d.index] = torch.cuda.Stream(device=d)
            logger.info(
                "LMP_MOE_CUDA_GRAPH=1: MoE BMM CUDAGraph streams ready; capture on first forward per cache key "
                "(multi-bucket dict); fused experts run sequentially (no thread pool)."
            )
            if self._moe_cuda_graph_buckets_max_t:
                logger.info(
                    "LMP_MOE_CUDA_GRAPH_BUCKETS_MAX_T=%s: max_t quantized upward for graph dedup",
                    self._moe_cuda_graph_buckets_max_t,
                )

    def init_kt_kernel(self, max_len: int = 1024):
        import kt_kernel_ext
        import kt_kernel
        
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print(f"kt-kernel version      : {kt_kernel.__version__}")
        print(f"kt-kernel CPU variant : {kt_kernel.__cpu_variant__}")
        
        CPUINFER_PARAM = 96
        CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
        expert_num = self.mlpm.get_experts_num()
        num_experts_per_tok = self.mlpm.get_experts_per_tok()
        hidden_size = self.mlpm.config.hidden_size
        moe_intermediate_size = self.mlpm.get_moe_intermediate_size()
        
        physical_to_logical_map = torch.tensor(range(expert_num), device="cpu", dtype=torch.int64).contiguous()
        
        moes = []
        for i in range(self.mlpm.get_num_hidden_layers()):
            #TODO TESTING
            # if i > 10:
            #     continue
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, moe_intermediate_size, 0)
            config.max_len = max_len
            config.layer_idx = i
            gate_up_proj, down_proj, _ = self.mlpm.get_fused_experts_gate_up_down_act_fn(self.hmv.mlpm_hi, i)

            gate_up_proj_cpu = gate_up_proj.cpu().contiguous()
            down_proj_cpu = down_proj.cpu().contiguous()
            gate_up_reshaped = gate_up_proj_cpu.view(expert_num, 2, moe_intermediate_size, hidden_size)
            gate_view = gate_up_reshaped[:, 0, :, :]  # [num_experts, intermediate_size, hidden_dim]
            up_view = gate_up_reshaped[:, 1, :, :]    # [num_experts, intermediate_size, hidden_dim]
            gate_view = gate_view.cpu().contiguous()
            up_view = up_view.cpu().contiguous()

            config.gate_proj = gate_view.data_ptr()
            config.up_proj = up_view.data_ptr()
            config.down_proj = down_proj_cpu.data_ptr()

            config.gate_scale = 0
            config.up_scale = 0
            config.down_scale = 0
            config.pool = CPUInfer.backend_

            moe = kt_kernel_ext.moe.AVX512BF16_MOE(config)
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()
            logger.info(f"init kt-kernel layer {i} ok")
            moes.append(moe)
        return moes, CPUInfer

    def _make_kt_kernel_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        cpu_expert_ids: list[int] | None = None,
    ):
        """
        为 kt-kernel ``forward_task`` 生成标准输入（与 ``cuda_memory_view._make_inputs_from_layer_prefill`` 对齐）。
        返回:
        ``(batch_tensor, tok_k, topk_ids_cpu, topk_weights_cpu, hidden_states_cpu, output_cpu_pin)``
        最后一项为 ``gpinpool`` 中分配的 **pinned** 输出缓冲（形状 ``[tokens_num, hidden_size]``，bf16），供 ``CPUInfer`` 写入。
        """
        if hidden_states.ndim != 2:
            raise ValueError(f"hidden_states must be 2D [tokens, hidden], got shape={tuple(hidden_states.shape)}")
        if topk_idx.ndim != 2 or topk_weight.ndim != 2:
            raise ValueError(
                f"topk_idx/topk_weight must be 2D, got idx={tuple(topk_idx.shape)}, w={tuple(topk_weight.shape)}"
            )
        if hidden_states.device.type != "cpu" or topk_idx.device.type != "cpu" or topk_weight.device.type != "cpu":
            raise ValueError("hidden_states/topk_idx/topk_weight must be on CPU")
        tokens_num = int(topk_idx.shape[0])
        tok_k = int(self.mlpm.get_experts_per_tok())
        if int(topk_idx.shape[1]) != tok_k or int(topk_weight.shape[1]) != tok_k:
            raise ValueError(
                f"topk second dim must equal experts_per_tok={tok_k}, "
                f"got idx={tuple(topk_idx.shape)}, w={tuple(topk_weight.shape)}"
            )

        hidden_size = int(hidden_states.shape[-1])
        if int(hidden_states.shape[0]) != tokens_num:
            raise ValueError(
                f"hidden_states tokens mismatch: hidden={tuple(hidden_states.shape)}, topk={tuple(topk_idx.shape)}"
            )

        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous torch.bfloat16 on CPU")
        hidden_states_cpu = hidden_states
        topk_ids_cpu = topk_idx.to(torch.int64).contiguous()
        topk_weights_cpu = topk_weight
        # topk_weights_cpu = topk_weight.float().contiguous()
        if cpu_expert_ids is not None:
            keep_ids = sorted(set(int(x) for x in cpu_expert_ids))
            if not keep_ids:
                raise ValueError("cpu_expert_ids must be non-empty when provided")
            keep_t = torch.tensor(keep_ids, dtype=torch.int64, device=topk_ids_cpu.device)
            in_keep = torch.isin(topk_ids_cpu, keep_t)
            skip_id = torch.tensor(-1, dtype=topk_ids_cpu.dtype, device=topk_ids_cpu.device)
            topk_ids_cpu = torch.where(in_keep, topk_ids_cpu, skip_id).contiguous()
            topk_weights_cpu = (topk_weights_cpu * in_keep.to(dtype=topk_weights_cpu.dtype)).contiguous()
        output_proto = torch.empty(
            (tokens_num, hidden_size), dtype=torch.bfloat16, device="cpu"
        )
        batch_tensor = torch.tensor([tokens_num], dtype=torch.int32, device="cpu").contiguous()
        return batch_tensor, tok_k, topk_ids_cpu, topk_weights_cpu, hidden_states_cpu, output_proto

    def _prepare_fused_expert_work_items(
        self,
        layer_idx: int,
        experts_state_dict_slices_packed: dict,
        device_expert_map: dict[int, list[int]],
        expert_indices_map: Dict[int, tuple[int, int]],
        flat_hidden_states: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_experts_weight: torch.Tensor,
        idxs: torch.Tensor,
        token_idxs: torch.Tensor,
        expert_cache: torch.Tensor,
        mm_backend: str = "bmm",
    ) -> list[dict]:
        mm_backend = (mm_backend or "gmm").strip().lower()
        if mm_backend not in ("gmm", "bmm"):
            raise ValueError(f"Unsupported mm_backend: {mm_backend}")

        packed_by_layer_device = {}
        if isinstance(experts_state_dict_slices_packed, dict):
            packed_by_layer_device = experts_state_dict_slices_packed.get("packed_by_layer_device", {}) or {}
        layer_packed_map = packed_by_layer_device.get(layer_idx, {}) if isinstance(packed_by_layer_device, dict) else {}

        x_slots_all = flat_hidden_states[token_idxs]  # [slots,H] presorted
        expert_ids_all = flat_expert_indices[idxs].to(dtype=torch.int64)  # [slots]
        slot_w_all = flat_experts_weight[idxs]  # [slots,1]

        work_items: list[dict] = []
        for device_id, device_experts in sorted(device_expert_map.items(), key=lambda x: int(x[0])):
            device_expert_ids = sorted(int(eid) for eid in device_experts)
            if not device_expert_ids:
                continue
            packed_ent = layer_packed_map.get(int(device_id)) if isinstance(layer_packed_map, dict) else None
            if not isinstance(packed_ent, dict):
                continue
            gate_up_packed = packed_ent.get("gate_up_packed")
            down_packed = packed_ent.get("down_packed")
            if gate_up_packed is None or down_packed is None:
                continue

            e_rows = int(gate_up_packed.size(0))
            if int(down_packed.size(0)) != e_rows:
                raise RuntimeError(
                    "gate_up_packed / down_packed E mismatch: %s vs %s"
                    % (e_rows, int(down_packed.size(0)))
                )
            exp_rows = packed_ent.get("experts")
            if isinstance(exp_rows, list) and len(exp_rows) == e_rows:
                expert_id_order = [int(x) for x in exp_rows]
            else:
                expert_id_order = sorted(int(e) for e in device_expert_ids)
                if len(expert_id_order) != e_rows:
                    raise RuntimeError(
                        "fused expert pack / device map mismatch: gate_up_packed dim0=%s, "
                        "packed experts=%r, device_expert_ids=%r (layer=%s dev=%s)"
                        % (e_rows, exp_rows, device_expert_ids, layer_idx, device_id)
                    )

            x_parts, eid_parts, w_parts, tok_parts = [], [], [], []
            for eid in expert_id_order:
                if eid not in expert_indices_map:
                    continue
                start_idx, end_idx = expert_indices_map[eid]
                if end_idx <= start_idx:
                    continue
                x_parts.append(x_slots_all[start_idx:end_idx])
                eid_parts.append(expert_ids_all[start_idx:end_idx])
                w_parts.append(slot_w_all[start_idx:end_idx])
                tok_parts.append(token_idxs[start_idx:end_idx])
            if not x_parts:
                continue

            # Avoid unnecessary cat copy for single-expert case.
            x_dev_sorted = x_parts[0] if len(x_parts) == 1 else torch.cat(x_parts, dim=0)
            eid_dev_sorted = eid_parts[0] if len(eid_parts) == 1 else torch.cat(eid_parts, dim=0)
            w_dev_sorted = w_parts[0] if len(w_parts) == 1 else torch.cat(w_parts, dim=0)
            token_ids_for_slots = (tok_parts[0] if len(tok_parts) == 1 else torch.cat(tok_parts, dim=0)).to(expert_cache.device)

            compute_device = gate_up_packed.device
            x_dev_sorted = x_dev_sorted.to(compute_device, non_blocking=True)
            eid_dev_sorted = eid_dev_sorted.to(compute_device, non_blocking=True)
            w_dev_sorted = w_dev_sorted.to(compute_device, non_blocking=True)

            ids_tensor = torch.tensor(expert_id_order, device=compute_device, dtype=torch.int64)
            remap = torch.full((int(self.mlpm.get_experts_num()),), -1, dtype=torch.int64, device=compute_device)
            remap[ids_tensor] = torch.arange(ids_tensor.numel(), device=compute_device, dtype=torch.int64)
            eid_dev_sub = remap[eid_dev_sorted]

            item: dict = {
                "device_id": int(device_id),
                "x_dev_sorted": x_dev_sorted,
                "eid_dev_sub": eid_dev_sub,
                "w_dev_sorted": w_dev_sorted,
                "token_ids_for_slots": token_ids_for_slots,
                "gate_up_packed": gate_up_packed,
                "down_packed": down_packed,
            }
            # Cache transposed expert weights once per packed entry.
            # GMM path also consumes these tensors directly.
            if "gate_up_eh2i" not in packed_ent:
                packed_ent["gate_up_eh2i"] = gate_up_packed.transpose(1, 2)
            if "down_eih" not in packed_ent:
                packed_ent["down_eih"] = down_packed.transpose(1, 2)
            item["gate_up_eh2i"] = packed_ent["gate_up_eh2i"]
            item["down_eih"] = packed_ent["down_eih"]

            # BMM：在提交前完成 pad。权重转置用 .transpose(1,2) 零拷贝 strided view；
            # torch.bmm 底层走 cuBLAS strided-batched GEMM，原生支持非连续张量，无需 .contiguous()。
            # 转置结果缓存在 packed_ent 中，同一 prefill 内各层只算一次。
            if (mm_backend or "bmm").strip().lower() == "bmm":
                stacked, counts = self.mlpm._batched_pad_inputs_presorted(
                    x_dev_sorted, eid_dev_sub, e_rows
                )
                item["stacked_inputs"] = stacked
                item["counts"] = counts
                item["y_bmm_out"] = torch.empty(
                    (x_dev_sorted.size(0), x_dev_sorted.size(1)),
                    device=compute_device,
                    dtype=x_dev_sorted.dtype,
                )
            work_items.append(item)
        return work_items


    @torch.no_grad()
    def test_mp_basic_load(self):
        
        cuda_hook_time("load weights")
        self.cmv.load_general_and_init()
        self.cmv.load_qkvgon_weight_onetime()
        cuda_hook_time_end("load weights")

    def _decoder_layer_forward_bench_path(
        self,
        layer_idx: int,
        ghidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Cache,
        *,
        replica_uuid: str,
        experts_state_dict_slices_packed: dict,
        tensor_to_device: dict,
        tensor_task_queue_index,
        tensor_copy_chunks,
        tensor_device_offsets,
        moe_decode_expert_backend: str = "fused",
    ) -> torch.Tensor:
        """``test_mp_generate_multi_device_layer`` 单层前向：attention 与 FFN/MoE 均经 ``mlpm`` 通用 bench 接口按模型类型分发。

        ``moe_decode_expert_backend``: ``"fused"`` 完整 ``layer_moe_fused``；``"gpu"`` 纯 GPU
        ``layer_moe_fused_decode_gpu``；``"cpu"`` 纯 CPU kt-kernel ``layer_moe_fused_decode_cpu``。
        """
        mi = self.cmv.mlpm_ci
        _be = (moe_decode_expert_backend or "fused").strip().lower()
        if _be == "gpu":
            _moe_route = self.layer_moe_fused_decode_gpu
        elif _be == "cpu":
            _moe_route = self.layer_moe_fused_decode_cpu
        else:
            _moe_route = self.layer_moe_fused
        cuda_hook_time("*sagl")
        residual_in = ghidden_states
        h = self.mlpm.iln_func(mi, layer_idx=layer_idx, hidden_states=ghidden_states)
        h = self.mlpm.self_attn_func(
            mi,
            layer_idx=layer_idx,
            hidden_states=h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        ffn_skip = self.mlpm.bench_ffn_skip_hidden(mi, layer_idx, residual_in, h)
        gate_in = self.mlpm.bench_gate_moe_hidden(mi, layer_idx, residual_in, h)
        cuda_hook_time_end("*sagl")

        if layer_idx < self.mlpm.get_first_k_dense_replace():
            cuda_hook_time("dense_mlp")
            out = self.mlpm.dense_mlp_func(mi, layer_idx=layer_idx, hidden_states=gate_in)
            cuda_hook_time_end("dense_mlp")
            return self.mlpm.apply_decoder_layer_scale(mi, layer_idx, out + ffn_skip)

        dense_prefix = self.mlpm.ffn_dense_prefix_before_route(mi, layer_idx, ffn_skip)
        if dense_prefix is not None:
            topk_idx, topk_weight, _ = self.mlpm.gate_func(mi, layer_idx, ffn_skip)
            expert_in = self.mlpm.moe_experts_input_hidden(mi, layer_idx, ffn_skip)
            cuda_hook_time("*layer_moe_fused")
            routed = _moe_route(
                layer_idx=layer_idx,
                hidden_states=expert_in,
                topk_idx=topk_idx,
                topk_weight=topk_weight,
                replica_uuid=replica_uuid,
                experts_state_dict_slices_packed=experts_state_dict_slices_packed,
                tensor_to_device=tensor_to_device,
                tensor_task_queue_index=tensor_task_queue_index,
                tensor_copy_chunks=tensor_copy_chunks,
                tensor_device_offsets=tensor_device_offsets,
            )
            cuda_hook_time_end("*layer_moe_fused")
            merged = self.mlpm.ffn_merge_dense_and_routed(
                mi, layer_idx, ffn_skip, dense_prefix, routed
            )
            return self.mlpm.apply_decoder_layer_scale(mi, layer_idx, merged)

        if self.mlpm.ffn_skip_routed_moe_use_standalone_dense(mi, layer_idx):
            out = self.mlpm.ffn_dense_non_routed_after_attn(mi, layer_idx, ffn_skip)
            return self.mlpm.apply_decoder_layer_scale(mi, layer_idx, out)

        topk_idx, topk_weight, _ = self.mlpm.gate_func(mi, layer_idx, gate_in)
        cuda_hook_time("*layer_moe_fused")
        routed = _moe_route(
            layer_idx=layer_idx,
            hidden_states=gate_in,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            replica_uuid=replica_uuid,
            experts_state_dict_slices_packed=experts_state_dict_slices_packed,
            tensor_to_device=tensor_to_device,
            tensor_task_queue_index=tensor_task_queue_index,
            tensor_copy_chunks=tensor_copy_chunks,
            tensor_device_offsets=tensor_device_offsets,
        )
        cuda_hook_time_end("*layer_moe_fused")
        merged = self.mlpm.ffn_merge_dense_and_routed(mi, layer_idx, ffn_skip, None, routed)
        return self.mlpm.apply_decoder_layer_scale(mi, layer_idx, merged)

    @torch.no_grad()
    def test_mp_generate_multi_device_layer(self):
        
        # 32, 64, 128
        # Keep this lightweight to avoid OOM on shared GPUs.
        batch_size = 4
        seq_len = 128
        dtype = self.mlpm.config.torch_dtype
        hidden_size = self.mlpm.config.hidden_size
        
        device_list = self.device_list
        device1 = device_list[0]

        cuda_hook_time("generate_input_ids")
        # 使用局部 generator 而非全局默认 CUDA generator：若上一轮 CUDAGraph 捕获失败
        # 导致默认 generator 的 capturing_ flag 卡死，torch.randn 会报
        # "Offset increment outside graph capture encountered unexpectedly"。
        # 局部 generator 不受全局 flag 影响，完全绕过该问题。(pytorch/pytorch#171263)
        _local_gen = torch.Generator(device=device1)
        _local_gen.manual_seed(int(torch.randint(0, 2**31, (1,)).item()))
        inputs_tokens = torch.randn(
            batch_size, seq_len, hidden_size, dtype=dtype, device=device1, generator=_local_gen
        )
        tokenizer=AutoTokenizer.from_pretrained(self.mlpm.model_abs_path, trust_remote_code=True)
        inputs_ids = generate_input_ids(tokenizer, batch_size, seq_len, device1)
        cuda_hook_time_end("generate_input_ids")

        cuda_hook_time("init_cache")
        past_key_value = DynamicCache(config=self.mlpm.config)
        past_key_values_length = past_key_value.get_seq_length()
        cuda_hook_time_end("init_cache")
    

        cuda_hook_time("init_loading_placement")
        tensor_to_device, device_expected_used_bytes, expert_id_device_map = self.predo_tensor_index_locate(
            tensor_index_json=self.tensor_index_json)
        self._layer_experts_alloc_cache = self._build_layer_experts_alloc_index(tensor_to_device)
        device_memory_used = self.calculate_device_memory_from_tensor_to_device(
            tensor_to_device=tensor_to_device, tensor_index_json=self.tensor_index_json)
        device_memory_used_int = {
            int(str(dev).split(":")[-1]): int(bytes_used)
            for dev, bytes_used in device_memory_used.items()
        }
        cuda_memory_ptrs = allocate_cuda_memory(device_memory_used_int)
        all_cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
        device_uuid_map = get_device_uuid_map()

        tensor_device_offsets, tensor_copy_chunks, device_next_offset, tensor_task_queue_index = self.generate_chunk(
            tensor_to_device=tensor_to_device,
            tensor_index_json=self.tensor_index_json,
            device_memory_used=device_memory_used,
        )
        self._expert_task_id_index = self._build_expert_task_id_index(tensor_task_queue_index)
        extracted_general_tensor_device_offsets, extracted_general_tensor_copy_chunks, tensor_device_offsets, tensor_copy_chunks = self.general_chunk_from_chunks(
            tensor_device_offsets=tensor_device_offsets,
            tensor_copy_chunks=tensor_copy_chunks,
            tensor_task_queue_index=tensor_task_queue_index,
        )

        (
            sagl_tensor_device_offsets,
            sagl_tensor_copy_chunks,
            tensor_device_offsets,
            tensor_copy_chunks,
        ) = self.nonexpert_chunk_from_chunks(
            tensor_device_offsets=tensor_device_offsets,
            tensor_copy_chunks=tensor_copy_chunks,
            tensor_task_queue_index=tensor_task_queue_index,
        )
        general_sagl_device_offsets = self._merge_tensor_device_offsets(
            extracted_general_tensor_device_offsets,
            sagl_tensor_device_offsets,
        )
        general_sagl_copy_chunks = self._merge_tensor_copy_chunks(
            extracted_general_tensor_copy_chunks,
            sagl_tensor_copy_chunks,
        )
        general_sagl_cuda_memory_handles = self.get_cuda_memory_handles_by_copy_chunks(
            all_cuda_memory_handles=all_cuda_memory_handles,
            tensor_copy_chunks=general_sagl_copy_chunks,
        )
        cuda_hook_time_end("init_loading_placement")

        cuda_hook_time("init_general_sagl_loading_async")
        _, replica_uuid1 = load_into_gpu_async(
            client=self.client,
            device_uuid_map=device_uuid_map,
            model_path=self.mlpm.model_path,
            tensor_copy_chunks=general_sagl_copy_chunks,
            cuda_memory_handles=general_sagl_cuda_memory_handles,
        )
        general_sagl_state_dict = self.restore_state_dict(
            tensor_index_json=self.tensor_index_json,
            cuda_memory_ptrs=cuda_memory_ptrs,
            tensor_device_offsets=general_sagl_device_offsets,
        )
        self.cmv.restore2model_strict(general_sagl_state_dict, self.cmv.mlpm_ci)
        self.client.confirm_model_loaded(self.mlpm.model_path, replica_uuid1)
        cuda_hook_time_end("init_general_sagl_loading_async")

        cuda_hook_time("init_experts_loading_async")
        experts_tensor_copy_chunks = tensor_copy_chunks
        experts_tensor_device_offsets = MLPLLM._prune_device_offsets_to_copy_chunks_by_task_id(
            tensor_device_offsets,
            experts_tensor_copy_chunks,
            tensor_task_queue_index,
        )
        cuda_memory_handles = all_cuda_memory_handles
        _, replica_uuid2 = load_into_gpu_async(
            client=self.client,
            device_uuid_map=device_uuid_map,
            model_path=self.mlpm.model_path,
            tensor_copy_chunks=experts_tensor_copy_chunks,
            cuda_memory_handles=cuda_memory_handles,
        )
        cuda_hook_time_end("init_experts_loading_async")

        self._log_tensor_split_integrity_check(
            tensor_task_queue_index=tensor_task_queue_index,
            general_sagl_copy_chunks=general_sagl_copy_chunks,
            experts_tensor_copy_chunks=experts_tensor_copy_chunks,
            experts_tensor_device_offsets=experts_tensor_device_offsets,
        )

        cuda_hook_time("restore_state_dict")
        experts_state_dict_slices_packed = self.restore_experts_state_dict(
            tensor_index_json=self.tensor_index_json,
            cuda_memory_ptrs=cuda_memory_ptrs,
            tensor_device_offsets=experts_tensor_device_offsets,
        )
        cuda_hook_time_end("restore_state_dict")

        cuda_hook_time("init_inputs_tokens")
        embed_tokens = self.mlpm.get_embed_tokens(self.cmv.mlpm_ci)
        inputs_tokens = embed_tokens(inputs_ids)
        position_ids = torch.arange(
            past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device1
        )
        position_ids = position_ids.unsqueeze(0)
        # sdpa flash attention
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            None,
            (batch_size, seq_len),
            inputs_tokens,
            past_key_values_length=past_key_values_length,
        )
        if self.mlpm.config._attn_implementation == "eager":
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                None,
                (batch_size, seq_len),
                inputs_tokens,
                past_key_values_length,
            )
        cuda_hook_time_end("init_inputs_tokens")

        cuda_hook_time("prefill_step")
        time_start_prefill = time.time()

        self.num_experts_on_cpu_ratio = 0.5
        
        ghidden_states = inputs_tokens
        for layer_idx in range(self.mlpm.get_num_hidden_layers()):
            #TODO TESTING
            # if layer_idx > 10:
            #     continue
            cuda_hook_time("prefill_layer")
            logger.debug(f"-------------------------------- start prefill layer {layer_idx} --------------------------------")

            ghidden_states = self._decoder_layer_forward_bench_path(
                layer_idx,
                ghidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                replica_uuid=replica_uuid2,
                experts_state_dict_slices_packed=experts_state_dict_slices_packed,
                tensor_to_device=tensor_to_device,
                tensor_task_queue_index=tensor_task_queue_index,
                tensor_copy_chunks=tensor_copy_chunks,
                tensor_device_offsets=tensor_device_offsets,
            )

            cuda_hook_time_end("prefill_layer")
            logger.debug(f"-------------------------------- end prefill layer {layer_idx} --------------------------------")            
        cuda_hook_time_end("prefill_step")
        logger.info(f"prefill time: {time.time() - time_start_prefill} seconds")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # free_cuda_memory(cuda_memory_ptrs)

        # return
        self.num_experts_on_cpu_ratio = 0.0
        num_step = 31
        time_decode_list = []
        for i in range(num_step):
            time_start_decode = time.time()
            cuda_hook_time("decode_step")

            # ── 准备 decode 单步的输入 ──────────────────────────────────────
            cuda_hook_time("init_inputs_tokens")
            # past_key_values_length 在每步后由 DynamicCache 自动增长
            past_key_values_length = past_key_value.get_seq_length()

            norm = self.mlpm.get_final_norm(self.cmv.mlpm_ci)
            lm_head = self.mlpm.get_lm_head(self.cmv.mlpm_ci)
            embed_tokens = self.mlpm.get_embed_tokens(self.cmv.mlpm_ci)
            next_token_ids = get_next_token_helper(norm, lm_head, ghidden_states, self.device1)
            next_inputs_tokens = embed_tokens(next_token_ids)

            query_length = next_inputs_tokens.shape[1]  # decode 阶段固定为 1
            input_shape = (batch_size, query_length)

            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + query_length,
                dtype=torch.long,
                device=device1,
            ).unsqueeze(0)

            if self.mlpm.config._attn_implementation == "eager":
                attention_mask = _prepare_4d_causal_attention_mask(
                    None, input_shape, next_inputs_tokens, past_key_values_length,
                )
            elif self.mlpm.config._attn_implementation == "sdpa":
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    None, input_shape, next_inputs_tokens,
                    past_key_values_length=past_key_values_length,
                )
            else:
                attention_mask = None
            if attention_mask is None:
                bsz, qlen = input_shape
                attention_mask = torch.zeros(
                    (bsz, 1, qlen, past_key_values_length + query_length),
                    dtype=next_inputs_tokens.dtype,
                    device=next_inputs_tokens.device,
                )
            cuda_hook_time_end("init_inputs_tokens")

            ghidden_states = next_inputs_tokens
            logger.debug("decode step %s next_inputs_tokens shape=%s", i, tuple(next_inputs_tokens.shape))

            # ── 逐层推理（与 prefill 结构一致）───────────────────────────────
            for layer_idx in range(self.mlpm.get_num_hidden_layers()):
                cuda_hook_time("decode_layer")
                logger.debug("---- decode step %s layer %s ----", i, layer_idx)

                if i < 1:
                    _decode_moe_backend = "fused"
                else:
                    _raw = os.environ.get("LMP_MOE_DECODE_EXPERT_BACKEND", "cpu").strip().lower()
                    _decode_moe_backend = _raw if _raw in ("fused", "gpu", "cpu") else "gpu"

                ghidden_states = self._decoder_layer_forward_bench_path(
                    layer_idx,
                    ghidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    replica_uuid=replica_uuid2,
                    experts_state_dict_slices_packed=experts_state_dict_slices_packed,
                    tensor_to_device=tensor_to_device,
                    tensor_task_queue_index=tensor_task_queue_index,
                    tensor_copy_chunks=tensor_copy_chunks,
                    tensor_device_offsets=tensor_device_offsets,
                    moe_decode_expert_backend=_decode_moe_backend,
                )
                cuda_hook_time_end("decode_layer")

            cuda_hook_time_end("decode_step")
            decode_time_cost = time.time() - time_start_decode
            time_decode_list.append(decode_time_cost)
            logger.info("decode step %s time: %s seconds", i, decode_time_cost)
            torch.cuda.synchronize()

        if len(time_decode_list) >= 2:
            time_decode_list = time_decode_list[5:]
        if time_decode_list:
            logger.info(
                "average decode time from step 5: %s seconds",
                sum(time_decode_list) / len(time_decode_list),
            )
        free_cuda_memory(cuda_memory_ptrs)

    def _quantize_max_t_for_cuda_graph(self, max_t: int) -> int:
        """将 packed ``max_t`` 量化到桶上界；无桶配置时返回原值。"""
        buckets = self._moe_cuda_graph_buckets_max_t
        if not buckets:
            return max_t
        for b in buckets:
            if b >= max_t:
                return b
        return max_t

    def _stacked_tensor_for_moe_cuda_graph_capture(self, st: torch.Tensor) -> torch.Tensor:
        """构造捕获用的 stacked：若在 ``LMP_MOE_CUDA_GRAPH_BUCKETS_MAX_T`` 中向上量化 ``max_t``，右侧 zero-pad。"""
        if not st.is_cuda:
            return st
        e, mt, h = st.shape
        mt_i = int(mt)
        qmt = self._quantize_max_t_for_cuda_graph(mt_i)
        if qmt == mt_i:
            return st
        tmpl = torch.zeros((int(e), qmt, int(h)), device=st.device, dtype=st.dtype)
        tmpl[:, :mt_i].copy_(st[:, :mt_i])
        return tmpl

    def _moe_bmm_cuda_graph_cache_key(self, layer_idx: int, item: dict) -> Tuple[Any, ...]:
        st = item["stacked_inputs"]
        gu = item["gate_up_eh2i"]
        dn = item["down_eih"]
        e, mt, h = st.shape
        q_mt = self._quantize_max_t_for_cuda_graph(int(mt))
        return (
            int(layer_idx),
            int(st.device.index),
            int(e),
            int(q_mt),
            int(h),
            st.dtype,
            int(gu.data_ptr()),
            int(dn.data_ptr()),
        )

    def _get_or_capture_moe_bmm_cuda_graph(
        self, layer_idx: int, item: dict, act_fn: Callable[..., torch.Tensor]
    ) -> Optional[_MoeBmmCoreCudaGraphBundle]:
        if not self._moe_cuda_graph_enabled:
            return None
        st = item["stacked_inputs"]
        if st is None or not st.is_cuda:
            return None
        key = self._moe_bmm_cuda_graph_cache_key(layer_idx, item)
        bundle = self._moe_bmm_cuda_graph_cache.get(key)
        if bundle is not None:
            return bundle
        capture_template = self._stacked_tensor_for_moe_cuda_graph_capture(st)
        idx = int(st.device.index)
        stream = self._moe_cuda_streams.get(idx)
        if stream is None:
            stream = torch.cuda.Stream(device=st.device)
            self._moe_cuda_streams[idx] = stream
        bundle = _MoeBmmCoreCudaGraphBundle.try_capture(
            stream,
            capture_template,
            item["gate_up_eh2i"],
            item["down_eih"],
            act_fn,
        )
        if bundle is not None:
            self._moe_bmm_cuda_graph_cache[key] = bundle
            logger.info("MoE BMM CUDAGraph captured layer=%s key=%s", layer_idx, key)
        return bundle
    def layer_moe_fused_decode_cpu(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        replica_uuid: str,
        experts_state_dict_slices_packed: dict,
        tensor_to_device,
        tensor_task_queue_index,
        tensor_copy_chunks,
        tensor_device_offsets,
    ):
        """Decode / 纯 CPU MoE（kt-kernel）：路由后的专家计算仅在 CPU 上提交，不经 GPU fused work_items。

        需已 ``init_kt_kernel`` 且 ``self.moes[layer_idx]`` 权重可用。输入仍可在 GPU，经 pin 内存拷到 CPU 再 ``CPUInfer``。
        下列参数为与 ``layer_moe_fused`` 对齐而保留，本函数不使用：
        ``replica_uuid`` / ``experts_state_dict_slices_packed`` / ``tensor_to_device`` /
        ``tensor_task_queue_index`` / ``tensor_copy_chunks`` / ``tensor_device_offsets``。
        """
        _ = (
            replica_uuid,
            experts_state_dict_slices_packed,
            tensor_to_device,
            tensor_task_queue_index,
            tensor_copy_chunks,
            tensor_device_offsets,
        )
        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape

        flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)
        num_slots = int(topk_idx.numel())
        expert_cache = torch.zeros_like(flat_hidden_states)

        flat_hidden_states_on_cpu_pin = gpinpool.alloc_same_pin_tensor(flat_hidden_states)
        flat_hidden_states_on_cpu_pin.copy_(flat_hidden_states, non_blocking=True)
        topk_weight_on_cpu_pin = gpinpool.alloc_same_pin_tensor(topk_weight)
        topk_weight_on_cpu_pin.copy_(topk_weight, non_blocking=True)
        topk_idx_on_cpu_pin = topk_idx.to(dtype=torch.int64, device="cpu", non_blocking=True)

        if num_slots == 0:
            gpinpool.free(flat_hidden_states_on_cpu_pin)
            gpinpool.free(topk_weight_on_cpu_pin)
            y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
            return expert_cache.view(*orig_shape) + y

        batch_tensor, tok_k, topk_ids_cpu, topk_weights_cpu, hidden_states_cpu, output_proto = (
            self._make_kt_kernel_forward_inputs(
                hidden_states=flat_hidden_states_on_cpu_pin,
                topk_idx=topk_idx_on_cpu_pin,
                topk_weight=topk_weight_on_cpu_pin,
                cpu_expert_ids=None,
            )
        )
        output_cpu_pin = gpinpool.alloc_same_pin_tensor(output_proto)
        self.CPUInfer.submit(
            self.moes[layer_idx].forward_task(
                batch_tensor.data_ptr(),
                tok_k,
                topk_ids_cpu.data_ptr(),
                topk_weights_cpu.data_ptr(),
                hidden_states_cpu.data_ptr(),
                output_cpu_pin.data_ptr(),
                False,
            )
        )
        self.CPUInfer.sync()

        output_gpu = output_cpu_pin.to(expert_cache.device, dtype=expert_cache.dtype, non_blocking=True)
        expert_cache.add_(output_gpu)

        y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
        out = expert_cache.view(*orig_shape) + y

        gpinpool.free(flat_hidden_states_on_cpu_pin)
        gpinpool.free(topk_weight_on_cpu_pin)
        gpinpool.free(output_cpu_pin)
        # topk_idx_on_cpu_pin 为普通 CPU tensor，非 gpinpool
        return out

    def layer_moe_fused_decode_gpu(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        replica_uuid: str,
        experts_state_dict_slices_packed: dict,
        tensor_to_device,
        tensor_task_queue_index,
        tensor_copy_chunks,
        tensor_device_offsets,
    ):
        """Decode / 纯 GPU 推理路径：不做 CPU MoE、不提交/等待权重拷贝，仅 GPU 上路由与专家计算。

        调用方需保证本步所需专家权重已在对应 GPU（与 ``experts_state_dict_slices_packed`` 一致）。
        ``replica_uuid`` / ``tensor_task_queue_index`` / ``tensor_copy_chunks`` / ``tensor_device_offsets``
        为与 ``layer_moe_fused`` 签名对齐而保留，本函数不使用。
        """
        _ = (replica_uuid, tensor_task_queue_index, tensor_copy_chunks, tensor_device_offsets)
        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape

        flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)
        flat_expert_indices = topk_idx.view(-1).to(dtype=torch.int64)
        flat_experts_weight = topk_weight.view(-1, 1)
        idxs = flat_expert_indices.argsort()
        num_experts = self.mlpm.get_experts_num()
        bc = torch.bincount(flat_expert_indices, minlength=num_experts)
        tokens_per_expert = bc.cumsum(0).cpu().tolist()
        token_idxs = idxs // self.mlpm.get_experts_per_tok()
        expert_cache = torch.zeros_like(flat_hidden_states)

        num_slots = int(idxs.numel())
        if num_slots == 0:
            y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
            return expert_cache.view(*orig_shape) + y

        expert_indices_map: Dict[int, tuple[int, int]] = {}
        expert_token_counts_list: list[tuple[int, int]] = []
        prev_end = 0
        for expert_id in range(num_experts):
            if expert_id >= len(tokens_per_expert):
                break
            end_idx = int(tokens_per_expert[expert_id])
            if end_idx == prev_end:
                continue
            start_idx = prev_end
            expert_indices_map[expert_id] = (start_idx, end_idx)
            expert_token_counts_list.append((expert_id, end_idx - start_idx))
            prev_end = end_idx

        experts_alloc = self._layer_experts_alloc_cache.get(layer_idx)
        if experts_alloc is None:
            experts_alloc = self.get_layer_experts_device_allocation(
                layer_idx=layer_idx,
                tensor_to_device=tensor_to_device,
            )
        e2d = experts_alloc.get("expert_id_to_device", {})
        if not isinstance(e2d, dict) or not e2d:
            raise ValueError("layer_moe_fused_decode_gpu requires experts_alloc.expert_id_to_device")

        device_expert_map: dict[int, list[int]] = {}
        for eid in expert_indices_map:
            dev = e2d.get(int(eid))
            if dev is None:
                raise RuntimeError(
                    f"layer_moe_fused_decode_gpu: expert {eid} missing from allocation (layer={layer_idx})"
                )
            dev_s = str(dev).lower()
            if not dev_s.startswith("cuda"):
                raise RuntimeError(
                    f"layer_moe_fused_decode_gpu: expert {eid} on {dev!r}, only cuda experts supported (layer={layer_idx})"
                )
            dev_idx = int(str(dev).split(":")[-1]) if ":" in str(dev) else 0
            device_expert_map.setdefault(dev_idx, []).append(int(eid))
        for k in list(device_expert_map.keys()):
            device_expert_map[k] = sorted(device_expert_map[k])

        mm_backend = os.environ.get("LMP_FUSED_EXPERT_MM", "gmm").strip().lower()
        _, _, act_fn = self.mlpm.get_fused_experts_gate_up_down_act_fn(self.cmv.mlpm_ci, layer_idx)
        work_items = self._prepare_fused_expert_work_items(
            layer_idx=layer_idx,
            experts_state_dict_slices_packed=experts_state_dict_slices_packed,
            device_expert_map=device_expert_map,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            flat_expert_indices=flat_expert_indices,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            token_idxs=token_idxs,
            expert_cache=expert_cache,
            mm_backend=mm_backend,
        )

        def _run_one_work_item(item: dict, backend: str):
            w_dev_sorted = item["w_dev_sorted"]
            token_ids_for_slots = item["token_ids_for_slots"]
            if backend == "bmm" and item.get("stacked_inputs") is not None:
                cg_bundle = self._get_or_capture_moe_bmm_cuda_graph(layer_idx, item, act_fn)
                if cg_bundle is not None:
                    cg_bundle.copy_inputs_and_replay(item["stacked_inputs"])
                    self.mlpm._batched_unpad_outputs_into(
                        item["y_bmm_out"], cg_bundle.y_st, item["counts"]
                    )
                    y_dev_sorted = item["y_bmm_out"]
                else:
                    self.mlpm.fused_experts_gate_up_down_bmm_from_padded_into(
                        item["y_bmm_out"],
                        item["stacked_inputs"],
                        item["counts"],
                        item["gate_up_eh2i"],
                        item["down_eih"],
                        act_fn,
                    )
                    y_dev_sorted = item["y_bmm_out"]
            else:
                x_dev_sorted = item["x_dev_sorted"]
                eid_dev_sub = item["eid_dev_sub"]
                gate_up_eh2i = item.get("gate_up_eh2i")
                down_eih = item.get("down_eih")
                if gate_up_eh2i is None or down_eih is None:
                    gate_up_eh2i = item["gate_up_packed"].transpose(1, 2)
                    down_eih = item["down_packed"].transpose(1, 2)
                y_dev_sorted = self.mlpm.fused_experts_gate_up_down_mm_presorted(
                    x_slots_sorted=x_dev_sorted,
                    expert_ids_sorted=eid_dev_sub,
                    gate_up_w_eh2i=gate_up_eh2i,
                    down_w_eih=down_eih,
                    act_fn=act_fn,
                    mm_backend=backend,
                )
            y_slots = y_dev_sorted * w_dev_sorted
            y_slots_main = y_slots.to(expert_cache.device, non_blocking=True)
            expert_cache.scatter_reduce_(
                dim=0,
                index=token_ids_for_slots.view(-1, 1).repeat(1, expert_cache.shape[-1]),
                src=y_slots_main,
                reduce="sum",
            )

        if mm_backend not in ("gmm", "bmm"):
            raise ValueError(f"Unsupported LMP_FUSED_EXPERT_MM backend: {mm_backend}")
        use_thread_pool = len(work_items) > 1 and not self._moe_cuda_graph_enabled
        if use_thread_pool:
            futs = [
                self._fused_expert_pool.submit(_run_one_work_item, item, mm_backend)
                for item in work_items
            ]
            for fut in as_completed(futs):
                fut.result()
        else:
            for item in work_items:
                _run_one_work_item(item, backend=mm_backend)

        y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
        return expert_cache.view(*orig_shape) + y

    def layer_moe_fused(self, 
                        layer_idx: int, 
                        hidden_states: torch.Tensor,
                        topk_idx: torch.Tensor,
                        topk_weight: torch.Tensor,
                        replica_uuid: str,
                        experts_state_dict_slices_packed: dict,
                        tensor_to_device,
                        tensor_task_queue_index,
                        tensor_copy_chunks,
                        tensor_device_offsets,
        ):

        time_start = time.time()
        time_perf_layer0 = time.perf_counter()
        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape

        flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)
        flat_expert_indices = topk_idx.view(-1).to(dtype=torch.int64)  # [slots]
        flat_experts_weight = topk_weight.view(-1, 1)  # [slots,1]
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.mlpm.get_experts_per_tok()
        num_experts = self.mlpm.get_experts_num()
        expert_cache = torch.zeros_like(flat_hidden_states)
        

        flat_hidden_states_on_cpu_pin = gpinpool.alloc_same_pin_tensor(flat_hidden_states)
        flat_hidden_states_on_cpu_pin.copy_(flat_hidden_states, non_blocking=True)
        topk_weight_on_cpu_pin = gpinpool.alloc_same_pin_tensor(topk_weight)
        topk_weight_on_cpu_pin.copy_(topk_weight, non_blocking=True)
        topk_idx_on_cpu_pin = topk_idx.to(dtype=torch.int64, device="cpu", non_blocking=True)

        num_slots = int(idxs.numel())
        if num_slots == 0:
            y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
            return expert_cache.view(*orig_shape) + y


        # 获取 expert_indices_map, expert_token_counts_list
        expert_indices_map: Dict[int, tuple[int, int]] = {}
        expert_token_counts_list = []
        prev_end = 0
        for expert_id in range(num_experts):
            if expert_id >= len(tokens_per_expert):
                break
            end_idx = int(tokens_per_expert[expert_id])
            if end_idx == prev_end:
                continue
            start_idx = prev_end
            expert_indices_map[expert_id] = (start_idx, end_idx)
            expert_token_counts_list.append((expert_id, end_idx - start_idx))
            prev_end = end_idx
        
        

        experts_alloc = self._layer_experts_alloc_cache.get(layer_idx)
        if experts_alloc is None:
            experts_alloc = self.get_layer_experts_device_allocation(
                layer_idx=layer_idx,
                tensor_to_device=tensor_to_device,
            )

        """
        写一个函数，根据 experts_alloc 和 expert_indices_map 和 expert_token_counts_list 分配专家到不同的设备, experts_alloc 已经包含了各专家在设备的位置
        1) 首先 CPU 分配到 self.num_experts_on_cpu_ratio 比例的专家, 剩余的专家分配到 GPU
        2) 其次 GPU 各设备分配到均匀的专家数
        3) 其次 CPU 尽量分配到少量的 token
        4) 打印 CPU 分配到的专家和各设备分配到的专及其树木家，以及各自的 token 总数和每个专家的 token 数
        5) 在下面调用函数，获得 cpu的 专家列表, 和 各设备的专家列表，以及GPU的专家总列表
        6) 保持最优，即CPU 分配到了尽量的少token，同时GPU 各设备分配到了均匀的专家数，GPU 分配到最大程度均匀的专家数
        """
        _t_alloc0 = time.perf_counter()
        experts_partition = self.allocate_experts_across_cpu_gpu(
            experts_alloc=experts_alloc,
            expert_indices_map=expert_indices_map,
            expert_token_counts_list=expert_token_counts_list,
        )
        _alloc_ms = (time.perf_counter() - _t_alloc0) * 1e3
        cpu_expert_ids = experts_partition["cpu_expert_ids"]
        device_expert_map = experts_partition["device_expert_map"]
        gpu_expert_ids = experts_partition["gpu_expert_ids"]
        gpu_expert_ids_by_device = experts_partition["gpu_expert_ids_by_device"]
        _ = (cpu_expert_ids, device_expert_map, gpu_expert_ids, gpu_expert_ids_by_device)

        _prefix_before_alloc_ms = (_t_alloc0 - time_perf_layer0) * 1e3
        logger.info(
            "[layer_moe_fused] layer=%s moe_prefix_before_alloc: %.3fms | allocate_experts_across_cpu_gpu: %.3fms",
            layer_idx,
            _prefix_before_alloc_ms,
            _alloc_ms,
        )


        time_start = time.time()
        layer_gpu_expert_task_ids = self.get_experts_task_ids(
            tensor_task_queue_index=tensor_task_queue_index,
            layer_idx=layer_idx,
            expert_ids=gpu_expert_ids,
        )
        logger.info(f"[layer_moe_fused] get_experts_task_ids time: {time.time() - time_start} seconds")
        if layer_gpu_expert_task_ids:
            time_start = time.time()
            ok_submit, pending_submit = self.client.submit_high_priority_copy_tasks(
                self.mlpm.model_path,
                replica_uuid,
                layer_gpu_expert_task_ids,
            )
            logger.info(
                f"[layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok={ok_submit} pending_count={len(pending_submit)} time: {time.time() - time_start} seconds"
            )

        time_start = time.time()
        if len(cpu_expert_ids) > 0:
            # submit kt-kernel experts（此前未单独打点，会压低分项之和 vs ``*layer_moe_fused`` hook）
            batch_tensor, tok_k, topk_ids_cpu, topk_weights_cpu, hidden_states_cpu, output_proto = (
                self._make_kt_kernel_forward_inputs(
                    hidden_states=flat_hidden_states_on_cpu_pin,
                    topk_idx=topk_idx_on_cpu_pin,
                    topk_weight=topk_weight_on_cpu_pin,
                    cpu_expert_ids=cpu_expert_ids,
                )
            )
            output_cpu_pin = gpinpool.alloc_same_pin_tensor(output_proto)
            self.CPUInfer.submit(
                self.moes[layer_idx].forward_task(
                    batch_tensor.data_ptr(),
                    tok_k,
                    topk_ids_cpu.data_ptr(),
                    topk_weights_cpu.data_ptr(),
                    hidden_states_cpu.data_ptr(),
                    output_cpu_pin.data_ptr(),
                    False,
                )
            )
            logger.info(
                "[layer_moe_fused] kt_kernel_prep_submit time: %s seconds",
                time.time() - time_start,
            )

        time_start = time.time()
        # GPU experts compute from restored packed experts weights.
        mm_backend = os.environ.get("LMP_FUSED_EXPERT_MM", "gmm").strip().lower()
        _, _, act_fn = self.mlpm.get_fused_experts_gate_up_down_act_fn(self.cmv.mlpm_ci, layer_idx)
        work_items = self._prepare_fused_expert_work_items(
            layer_idx=layer_idx,
            experts_state_dict_slices_packed=experts_state_dict_slices_packed,
            device_expert_map=device_expert_map,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            flat_expert_indices=flat_expert_indices,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            token_idxs=token_idxs,
            expert_cache=expert_cache,
            mm_backend=mm_backend,
        )
        logger.info(f"[layer_moe_fused] prepare_fused_expert_work_items time: {time.time() - time_start} seconds")

        if layer_gpu_expert_task_ids:
            time_start = time.time()
            ok_wait, pending_wait = self.client.wait_copy_tasks(
                self.mlpm.model_path,
                replica_uuid,
                layer_gpu_expert_task_ids,
                timeout_ms=60000,
            )
            logger.info(f"[layer_moe_fused] wait_copy_tasks(gpu experts) ok={ok_wait} pending_count={len(pending_wait)} time: {time.time() - time_start} seconds")
        
        def _run_one_work_item(item: dict, backend: str):
            w_dev_sorted = item["w_dev_sorted"]
            token_ids_for_slots = item["token_ids_for_slots"]
            if backend == "bmm" and item.get("stacked_inputs") is not None:
                cg_bundle = self._get_or_capture_moe_bmm_cuda_graph(layer_idx, item, act_fn)
                if cg_bundle is not None:
                    cg_bundle.copy_inputs_and_replay(item["stacked_inputs"])
                    self.mlpm._batched_unpad_outputs_into(
                        item["y_bmm_out"], cg_bundle.y_st, item["counts"]
                    )
                    y_dev_sorted = item["y_bmm_out"]
                else:
                    self.mlpm.fused_experts_gate_up_down_bmm_from_padded_into(
                        item["y_bmm_out"],
                        item["stacked_inputs"],
                        item["counts"],
                        item["gate_up_eh2i"],
                        item["down_eih"],
                        act_fn,
                    )
                    y_dev_sorted = item["y_bmm_out"]
            else:
                x_dev_sorted = item["x_dev_sorted"]
                eid_dev_sub = item["eid_dev_sub"]
                gate_up_eh2i = item.get("gate_up_eh2i")
                down_eih = item.get("down_eih")
                if gate_up_eh2i is None or down_eih is None:
                    gate_up_eh2i = item["gate_up_packed"].transpose(1, 2)
                    down_eih = item["down_packed"].transpose(1, 2)
                y_dev_sorted = self.mlpm.fused_experts_gate_up_down_mm_presorted(
                    x_slots_sorted=x_dev_sorted,
                    expert_ids_sorted=eid_dev_sub,
                    gate_up_w_eh2i=gate_up_eh2i,
                    down_w_eih=down_eih,
                    act_fn=act_fn,
                    mm_backend=backend,
                )
            y_slots = y_dev_sorted * w_dev_sorted
            y_slots_main = y_slots.to(expert_cache.device, non_blocking=True)
            expert_cache.scatter_reduce_(
                dim=0,
                index=token_ids_for_slots.view(-1, 1).repeat(1, expert_cache.shape[-1]),
                src=y_slots_main,
                reduce="sum",
            )

        time_start = time.time()
        # Phase 2: execute experts compute in parallel across devices.
        if mm_backend not in ("gmm", "bmm"):
            raise ValueError(f"Unsupported LMP_FUSED_EXPERT_MM backend: {mm_backend}")
        # CUDAGraph 捕获/重放必须在固定线程与专用 stream 上；多 GPU item 仍顺序执行以免跨线程 CUDA。
        use_thread_pool = len(work_items) > 1 and not self._moe_cuda_graph_enabled
        if use_thread_pool:
            futs = [
                self._fused_expert_pool.submit(_run_one_work_item, item, mm_backend)
                for item in work_items
            ]
            for fut in as_completed(futs):
                fut.result()  # re-raise any exception
        else:
            for item in work_items:
                _run_one_work_item(item, backend=mm_backend)

        logger.info(f"[layer_moe_fused] experts compute time: {time.time() - time_start} seconds")

        

        time_start_s = time.time()
        # CPUInfer output contains masked CPU experts only; merge into the same cache.

        if len(cpu_expert_ids) > 0:
            self.CPUInfer.sync()

            time_start = time.time()
            output_cpu2gpu = output_cpu_pin.to(expert_cache.device, dtype=expert_cache.dtype, non_blocking=True)
            logger.info(f"[layer_moe_fused] to time: {time.time() - time_start} seconds")
            expert_cache.add_(output_cpu2gpu)

        y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
        out = expert_cache.view(*orig_shape) + y
        logger.info(f"[layer_moe_fused] scatter_reduce_ time: {time.time() - time_start_s} seconds")

        gpinpool.free(flat_hidden_states_on_cpu_pin)
        if len(cpu_expert_ids) > 0:
            gpinpool.free(output_cpu_pin)
        return out



    
    def extract_gate_up_down_locate_info(self, layer_idx: int, tensor_to_device: dict):
        """
        1) 从 tensor_to_device 提取 某层layer 的 gate_up_proj, down_proj 的信息
        """
        if not isinstance(tensor_to_device, dict):
            raise TypeError("tensor_to_device must be a dict")
        if layer_idx < 0:
            raise ValueError("layer_idx must be >= 0")

        get_experts_num = getattr(self.mlpm, "get_experts_num", None)
        if not callable(get_experts_num):
            raise RuntimeError("mlpm must provide get_experts_num")
        experts_num = int(get_experts_num())
        expert_ids = list(range(experts_num))

        gate_names = self.mlpm.get_experts_names_w(layer_idx, expert_ids, WeightType.W1)
        down_names = self.mlpm.get_experts_names_w(layer_idx, expert_ids, WeightType.W2)

        def _collect(names: list[str]):
            tensor_map: dict[str, object] = {}
            expert_id_to_device: dict[int, str] = {}
            device_to_expert_ids: dict[str, list[int]] = {}
            for n in names:
                if n not in tensor_to_device:
                    continue
                v = tensor_to_device[n]
                tensor_map[n] = v
                # fused: {"expert_id_to_device": {...}, ...}
                if isinstance(v, dict) and "expert_id_to_device" in v:
                    raw = v.get("expert_id_to_device", {})
                    normalized = {int(k): str(dev) for k, dev in raw.items()}
                    for eid, dev in normalized.items():
                        expert_id_to_device[eid] = dev
                        device_to_expert_ids.setdefault(dev, []).append(eid)
                    continue
                # non-fused: parse (layer, expert) from tensor name
                key = self.mlpm.get_tensor_expert_group_key(n)
                if key is not None and key[0] == layer_idx and key[1] >= 0:
                    eid = int(key[1])
                    dev = str(v)
                    expert_id_to_device[eid] = dev
                    device_to_expert_ids.setdefault(dev, []).append(eid)

            for dev in device_to_expert_ids:
                device_to_expert_ids[dev].sort()
            return tensor_map, expert_id_to_device, device_to_expert_ids

        gate_tensor_map, gate_e2d, gate_d2e = _collect(gate_names)
        down_tensor_map, down_e2d, down_d2e = _collect(down_names)

        common_eids = sorted(set(gate_e2d.keys()) & set(down_e2d.keys()))
        for eid in common_eids:
            if gate_e2d[eid] != down_e2d[eid]:
                raise RuntimeError(
                    f"gate/down locate mismatch at layer={layer_idx}, expert_id={eid}, "
                    f"gate={gate_e2d[eid]}, down={down_e2d[eid]}"
                )

        return {
            "layer_idx": layer_idx,
            "gate_up_proj": {
                "tensor_to_device": gate_tensor_map,
                "expert_id_to_device": gate_e2d,
                "device_to_expert_ids": gate_d2e,
            },
            "down_proj": {
                "tensor_to_device": down_tensor_map,
                "expert_id_to_device": down_e2d,
                "device_to_expert_ids": down_d2e,
            },
        }

    def get_layer_experts_device_allocation(self, layer_idx: int, tensor_to_device: dict):
        """
        获取某一层 experts 到设备上的分配情况。

        返回:
        {
          "layer_idx": int,
          "expert_id_to_device": {expert_id: "cuda:x"},
          "device_to_expert_ids": {"cuda:x": [expert_id, ...]},
          "source_tensors": [tensor_name, ...],
        }
        """
        if not isinstance(tensor_to_device, dict):
            raise TypeError("tensor_to_device must be a dict")
        _ = int(layer_idx)  # kept for API compatibility with existing call sites
        if layer_idx < 0:
            raise ValueError("layer_idx must be >= 0")

        get_tensor_expert_group_key = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        if not callable(get_tensor_expert_group_key):
            raise RuntimeError("mlpm must provide get_tensor_expert_group_key")

        expert_id_to_device: dict[int, str] = {}
        source_tensors: list[str] = []

        def _bind_expert(eid: int, dev: str, src_name: str):
            prev = expert_id_to_device.get(eid)
            if prev is not None and prev != dev:
                raise RuntimeError(
                    f"expert device mismatch at layer={layer_idx}, expert_id={eid}: "
                    f"{prev} vs {dev} (tensor={src_name})"
                )
            expert_id_to_device[eid] = dev

        for tensor_name, locate in tensor_to_device.items():
            tname = str(tensor_name)
            group_key = get_tensor_expert_group_key(tname)
            if group_key is None or int(group_key[0]) != layer_idx:
                continue

            # non-fused experts: tensor_name already includes expert_id
            if int(group_key[1]) >= 0:
                if isinstance(locate, str):
                    _bind_expert(int(group_key[1]), str(locate), tname)
                    source_tensors.append(tname)
                continue

            # fused experts: read expert_id_to_device from locate dict
            if isinstance(locate, dict):
                e2d = locate.get("expert_id_to_device")
                if isinstance(e2d, dict):
                    for raw_eid, raw_dev in e2d.items():
                        _bind_expert(int(raw_eid), str(raw_dev), tname)
                    source_tensors.append(tname)

        device_to_expert_ids: dict[str, list[int]] = {}
        for eid, dev in expert_id_to_device.items():
            device_to_expert_ids.setdefault(dev, []).append(eid)
        for dev in device_to_expert_ids:
            device_to_expert_ids[dev].sort()

        return {
            "layer_idx": layer_idx,
            "expert_id_to_device": dict(sorted(expert_id_to_device.items())),
            "device_to_expert_ids": device_to_expert_ids,
            "source_tensors": sorted(set(source_tensors)),
        }

    def allocate_experts_across_cpu_gpu(
        self,
        experts_alloc: dict,
        expert_indices_map: Dict[int, tuple[int, int]],
        expert_token_counts_list: list[tuple[int, int]],
    ) -> dict:
        """
        根据 experts 分配信息与 token 负载，将命中的 experts 分配到 CPU/GPU。
        约束：experts_alloc 已给出 expert 的设备位置，GPU 侧保持该位置，不做跨卡重排。
        规则：
        1) CPU 按 `num_experts_on_cpu_ratio` 选取一定比例 experts。
        2) 为了让 GPU 各设备 experts 数尽量均匀，优先从“当前专家更多的设备”挑 CPU experts。
        3) 在同设备候选里优先挑 token 更少的 experts 给 CPU。
        """
        if not isinstance(experts_alloc, dict):
            raise TypeError("experts_alloc must be a dict")
        if not isinstance(expert_indices_map, dict):
            raise TypeError("expert_indices_map must be a dict[int, tuple[int, int]]")
        if not isinstance(expert_token_counts_list, list):
            raise TypeError("expert_token_counts_list must be a list[tuple[int, int]]")

        hit_expert_ids = set(int(eid) for eid in expert_indices_map.keys())
        token_counts = [(int(eid), int(cnt)) for eid, cnt in expert_token_counts_list if int(eid) in hit_expert_ids]
        if not token_counts:
            return {
                "cpu_expert_ids": [],
                "device_expert_map": {},
                "gpu_expert_ids": [],
                "gpu_expert_ids_by_device": {},
                "cpu_token_total": 0,
                "device_token_counts": {},
            }

        e2d = experts_alloc.get("expert_id_to_device", {})
        if not isinstance(e2d, dict) or not e2d:
            raise ValueError("experts_alloc.expert_id_to_device is required")

        hit_expert_to_device: dict[int, int] = {}
        for eid in hit_expert_ids:
            if eid not in e2d:
                continue
            dev = str(e2d[eid])
            dev_idx = int(dev.split(":")[-1]) if ":" in dev else int(dev)
            hit_expert_to_device[eid] = dev_idx
        if not hit_expert_to_device:
            raise ValueError("no hit experts found in experts_alloc.expert_id_to_device")

        gpu_device_ids = sorted(set(hit_expert_to_device.values()))

        sorted_experts_by_load = sorted(token_counts, key=lambda x: x[1])  # 小 token 在前
        num_experts_total = len(sorted_experts_by_load)
        num_experts_on_cpu = int(num_experts_total * float(self.num_experts_on_cpu_ratio))
        num_experts_on_cpu = max(0, min(num_experts_total, num_experts_on_cpu))

        token_count_map = {eid: cnt for eid, cnt in token_counts}
        experts_by_device: dict[int, list[int]] = {d: [] for d in gpu_device_ids}
        for eid in hit_expert_ids:
            dev = hit_expert_to_device.get(eid)
            if dev is not None:
                experts_by_device[dev].append(eid)

        # 软约束：GPU 尽量均匀，但每卡保留数不能超过本步路由命中且驻留在该卡的专家数（无跨卡 remap）。
        num_devices = len(gpu_device_ids)
        num_gpu_experts = num_experts_total - num_experts_on_cpu
        if num_devices <= 0 or num_gpu_experts < 0:
            raise ValueError("invalid device/expert counts for allocation")
        base_cnt, rem = divmod(num_gpu_experts, num_devices)
        ideal_gpu_count: dict[int, int] = {}
        for i, d in enumerate(gpu_device_ids):
            ideal_gpu_count[d] = base_cnt + (1 if i < rem else 0)

        current_cnt_by_device = {d: len(experts_by_device[d]) for d in gpu_device_ids}
        keep_on_gpu: dict[int, int] = {}
        for d in gpu_device_ids:
            keep_on_gpu[d] = min(ideal_gpu_count[d], current_cnt_by_device[d])
        remaining = num_gpu_experts - sum(keep_on_gpu.values())
        rr = 0
        device_list = list(gpu_device_ids)
        while remaining > 0:
            progressed = False
            for _ in range(len(device_list)):
                d = device_list[rr % len(device_list)]
                rr += 1
                if keep_on_gpu[d] < current_cnt_by_device[d]:
                    keep_on_gpu[d] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                raise ValueError(
                    "cannot place num_gpu_experts on GPU without cross-device remap: "
                    f"num_gpu_experts={num_gpu_experts}, "
                    f"per_device_hit_counts={dict(current_cnt_by_device)}"
                )

        remove_quota_by_device: dict[int, int] = {
            d: current_cnt_by_device[d] - keep_on_gpu[d] for d in gpu_device_ids
        }

        if sum(remove_quota_by_device.values()) != num_experts_on_cpu:
            raise ValueError(
                "internal mismatch in cpu allocation quota: "
                f"sum(remove_quota)={sum(remove_quota_by_device.values())}, "
                f"num_experts_on_cpu={num_experts_on_cpu}"
            )

        # 在每台设备的移除配额下，选择 token 最少的专家给 CPU（满足硬约束时的最优）。
        cpu_expert_ids: list[int] = []
        for d in gpu_device_ids:
            candidates = sorted(experts_by_device[d], key=lambda eid: (token_count_map.get(eid, 0), eid))
            quota = remove_quota_by_device[d]
            cpu_expert_ids.extend(candidates[:quota])

        cpu_expert_id_set = set(cpu_expert_ids)
        cpu_token_total = sum(cnt for eid, cnt in sorted_experts_by_load if eid in cpu_expert_id_set)

        gpu_experts = [(eid, cnt) for eid, cnt in sorted_experts_by_load if eid not in cpu_expert_id_set]
        device_expert_map: dict[int, list[int]] = {d: [] for d in gpu_device_ids}
        device_token_counts: dict[int, int] = {d: 0 for d in gpu_device_ids}
        for eid, cnt in gpu_experts:
            dev = hit_expert_to_device.get(eid)
            if dev is None:
                continue
            device_expert_map[dev].append(eid)
            device_token_counts[dev] += cnt
        for d in device_expert_map:
            device_expert_map[d].sort()

        print(
            "experts_cpu_alloc",
            {
                "expert_ids": cpu_expert_ids,
                "token_total": cpu_token_total,
                "token_per_expert": {eid: token_count_map[eid] for eid in cpu_expert_ids},
            },
        )
        for d in gpu_device_ids:
            eids = device_expert_map[d]
            print(
                f"experts_gpu_alloc_device_{d}",
                {
                    "expert_ids": eids,
                    "expert_count": len(eids),
                    "ideal_gpu_count": ideal_gpu_count[d],
                    "keep_on_gpu": keep_on_gpu[d],
                    "hit_count_on_device": current_cnt_by_device[d],
                    "token_total": device_token_counts[d],
                    "token_per_expert": {eid: token_count_map[eid] for eid in eids},
                },
            )

        gpu_expert_ids = sorted([eid for eid, _ in gpu_experts])
        gpu_expert_ids_by_device = {d: set(v) for d, v in device_expert_map.items()}
        return {
            "cpu_expert_ids": cpu_expert_ids,
            "device_expert_map": device_expert_map,
            "gpu_expert_ids": gpu_expert_ids,
            "gpu_expert_ids_by_device": gpu_expert_ids_by_device,
            "cpu_token_total": cpu_token_total,
            "device_token_counts": device_token_counts,
        }

    def extract_attn_locate_info(self, layer_idx: int, tensor_to_device: dict):
        """
        1) 从 tensor_to_device 提取 某层attn类 的信息
        """
        if not isinstance(tensor_to_device, dict):
            raise TypeError("tensor_to_device must be a dict")
        if layer_idx < 0:
            raise ValueError("layer_idx must be >= 0")

        get_tensor_index_layer_names = getattr(self.mlpm, "get_tensor_index_layer_names", None)
        if not callable(get_tensor_index_layer_names):
            raise RuntimeError("mlpm must provide get_tensor_index_layer_names")

        layer_names = list(get_tensor_index_layer_names(layer_idx))
        out: dict[str, object] = {}
        for n in layer_names:
            if n in tensor_to_device:
                out[n] = tensor_to_device[n]
        return out

    def extract_generate_locate_info(self,tensor_to_device: dict):
        """
        1) 从 tensor_to_device 提取 generate类 的信息
        """
        if not isinstance(tensor_to_device, dict):
            raise TypeError("tensor_to_device must be a dict")

        get_tensor_index_general_names = getattr(self.mlpm, "get_tensor_index_general_names", None)
        if not callable(get_tensor_index_general_names):
            raise RuntimeError("mlpm must provide get_tensor_index_general_names")

        general_names = list(get_tensor_index_general_names())
        out: dict[str, object] = {}
        for n in general_names:
            if n in tensor_to_device:
                out[n] = tensor_to_device[n]
        return out

    def calculate_device_memory_from_tensor_to_device(self, tensor_to_device: dict, tensor_index_json: dict) -> dict[str, int]:
        """
        1) 从 tensor_to_device 计算每个设备的显存使用情况，依赖 tensor_index_json 中的 tensor_meta 信息
        Args:
            tensor_to_device (dict): 张量名称到设备的映射。如 { "layer.0.weight": "cuda:0", ... }
            tensor_index_json (dict): tensor_index.json 中的信息
        Returns:
            dict: 设备 -> 显存占用（字节）
        """
        if not isinstance(tensor_to_device, dict):
            raise TypeError("tensor_to_device must be a dict")
        if not isinstance(tensor_index_json, dict):
            raise TypeError("tensor_index_json must be a dict")

        def _tensor_size_bytes(meta) -> int:
            if not isinstance(meta, (list, tuple)) or len(meta) < 2:
                return 0
            try:
                return int(meta[1])
            except Exception:
                return 0

        def _tensor_num_experts(meta) -> int | None:
            if not isinstance(meta, (list, tuple)) or len(meta) < 3:
                return None
            shape = meta[2]
            if not isinstance(shape, (list, tuple)) or not shape:
                return None
            try:
                n = int(shape[0])
            except Exception:
                return None
            return n if n > 0 else None

        device_memory: dict[str, int] = {}
        for tensor_name, locate in tensor_to_device.items():
            if tensor_name not in tensor_index_json:
                continue
            total_bytes = _tensor_size_bytes(tensor_index_json[tensor_name])
            if total_bytes <= 0:
                continue

            # 常规结构: tensor_name -> "cuda:x"
            if isinstance(locate, str):
                device_memory[locate] = device_memory.get(locate, 0) + total_bytes
                continue

            # fused 专家结构:
            # {
            #   "default_device": "cuda:x",
            #   "expert_id_to_device": {...},
            #   "device_to_expert_ids": {...},
            # }
            if isinstance(locate, dict):
                e2d = locate.get("expert_id_to_device")
                if isinstance(e2d, dict) and e2d:
                    # fused 专家张量：按“单专家显存 * 每设备专家数”严格计费。
                    normalized_e2d: dict[int, str] = {}
                    for raw_eid, raw_dev in e2d.items():
                        try:
                            eid = int(raw_eid)
                        except Exception as e:
                            raise TypeError(
                                f"invalid expert_id in expert_id_to_device for {tensor_name}: {raw_eid}"
                            ) from e
                        normalized_e2d[eid] = str(raw_dev)

                    if not normalized_e2d:
                        continue
                    sorted_eids = sorted(normalized_e2d.keys())

                    # 优先用 shape[0] 作为专家总数（对 Gemma fused 权重最准确）。
                    # 若缺失 shape 信息，则退化为映射里出现的 expert 数。
                    experts_num_from_shape = _tensor_num_experts(tensor_index_json[tensor_name])
                    experts_num = experts_num_from_shape if experts_num_from_shape is not None else len(sorted_eids)
                    if experts_num <= 0:
                        raise RuntimeError(f"invalid experts_num for {tensor_name}: {experts_num}")
                    if len(sorted_eids) > experts_num:
                        raise RuntimeError(
                            f"expert mapping overflow for {tensor_name}: "
                            f"mapped={len(sorted_eids)}, experts_num={experts_num}"
                        )

                    per_expert_bytes, remainder = divmod(total_bytes, experts_num)
                    if remainder != 0:
                        raise RuntimeError(
                            f"tensor bytes not divisible by experts_num for {tensor_name}: "
                            f"total_bytes={total_bytes}, experts_num={experts_num}, remainder={remainder}"
                        )
                    if per_expert_bytes <= 0:
                        raise RuntimeError(
                            f"invalid per_expert_bytes for {tensor_name}: "
                            f"total_bytes={total_bytes}, experts_num={experts_num}"
                        )

                    device_to_expert_count: dict[str, int] = {}
                    for eid in sorted_eids:
                        dev = normalized_e2d[eid]
                        device_to_expert_count[dev] = device_to_expert_count.get(dev, 0) + 1

                    assigned_total = 0
                    for dev, count in device_to_expert_count.items():
                        dev_bytes = per_expert_bytes * count
                        device_memory[dev] = device_memory.get(dev, 0) + dev_bytes
                        assigned_total += dev_bytes

                    # 当前映射覆盖的 expert 字节必须严格守恒。
                    expected_total = per_expert_bytes * len(sorted_eids)
                    if assigned_total != expected_total:
                        raise RuntimeError(
                            f"expert byte allocation mismatch for {tensor_name}: "
                            f"assigned_total={assigned_total}, expected_total={expected_total}"
                        )
                    continue

                default_device = locate.get("default_device")
                if isinstance(default_device, str):
                    device_memory[default_device] = device_memory.get(default_device, 0) + total_bytes
                continue

            raise TypeError(
                f"invalid tensor_to_device value for {tensor_name}: expected str/dict, got {type(locate)}"
            )
        return device_memory

    def read_tensor_index_json(self, model_path: str):
        tensor_index_json_path = os.path.join(model_path, "tensor_index.json")
        with open(tensor_index_json_path, "r") as f:
            tensor_index_json = json.load(f)
        return tensor_index_json

    def predo_tensor_index_locate(self, tensor_index_json: dict):
        """
        根据当前 GPU 数量与可用显存，为 tensor_index 中的权重分配设备。

        约束：
        1) 除 experts 外的所有参数（包含 lm_head/norm/embed_tokens 与 layer 内 self_attn/gate/layernorm）统一放在第一个设备。
        2) 同一 (layer, expert) 下的所有参数必须放在同一个设备，且同层 experts 尽量均匀分配到设备。
        3) 参数名解析优先使用 mlpm 接口。
        4) 先分配 experts，再放置非 experts 参数到第一张卡。
        5) 不允许存在未处理参数；vision_tower 等跳过项由 mlpm 接口指定。
        7) 由于layer的专家 按照 gate_up_proj, down_proj (num_experts, num_experts, 2 * intermediate_dim, hidden_size) (num_experts, num_experts, 2 * intermediate_dim, hidden_size)
        8) 请你为gate_up_proj , down_proj 按experts 均匀分配到 设备中，给出 gate_up_proj的 哪个专家分配到的设备，down_proj的 哪个专家分配到的设备，即相应的expert_id

        

        Returns:
            tuple[dict[str, object], dict[str, int]]:
                - ``tensor_to_device``:
                  - 常规参数: ``tensor_name -> "cuda:<id>"``
                  - fused 专家参数（gate_up_proj/down_proj）:
                    ``tensor_name -> {"default_device": "...", "expert_id_to_device": {...}, "device_to_expert_ids": {...}}``
                - ``device_expected_used_bytes``: 每个设备预期使用显存（bytes）
        """
        if not isinstance(tensor_index_json, dict) or not tensor_index_json:
            return {}

        from collections import defaultdict

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot locate tensor_index on GPUs")

        # 优先使用实例中的设备列表，避免越界到未启用设备。
        device_names = list(getattr(self, "device_list", []))
        if not device_names:
            device_names = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not device_names:
            raise RuntimeError("No CUDA devices found")

        # 使用 pynvml 读取设备剩余显存（bytes）。
        # 按需求：若无法读取或显存不足，直接报错，不做静默降级。
        device_remaining: dict[str, int] = {}
        try:
            import pynvml
        except ImportError as e:
            raise RuntimeError(
                "pynvml is required for predo_tensor_index_locate but is not installed"
            ) from e

        try:
            pynvml.nvmlInit()
            for dev in device_names:
                device_idx = int(str(dev).split(":")[-1])
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                device_remaining[dev] = int(memory_info.free)
        except Exception as e:
            raise RuntimeError(f"failed to read gpu memory via pynvml: {e}") from e
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        # 记录初始可用显存，用于后续“显存使用均衡”目标。
        device_initial_free = dict(device_remaining)

        # tensor_name -> (offset, size, shape, stride, dtype)
        # 这里我们只依赖 size（bytes）做分配。
        def _tensor_size_bytes(meta) -> int:
            if not isinstance(meta, (list, tuple)) or len(meta) < 2:
                return 0
            try:
                return int(meta[1])
            except Exception:
                return 0

        # 分组策略：
        # 1) 专家参数：同 layer 同 expert 的所有参数归为一组，同层尽量均衡分配；
        # 2) 非专家参数（general + layer_sagln）统一放在第一张卡；
        # 3) 不允许存在未分类参数；若存在，直接报错。
        expert_group_tensors: dict[tuple[int, int], list[str]] = defaultdict(list)
        expert_group_sizes: dict[tuple[int, int], int] = defaultdict(int)
        layer_sagln_tensors: dict[int, list[str]] = defaultdict(list)
        layer_sagln_sizes: dict[int, int] = defaultdict(int)
        general_tensors: list[tuple[str, int]] = []
        other_tensors: list[tuple[str, int]] = []

        # 通过 mlpm 接口获取每层 self_attn/gate 参数名，再映射到 tensor_index。
        # 配置层数缺失时按要求直接报错，不做任何 fallback。
        get_num_hidden_layers = getattr(self.mlpm, "get_num_hidden_layers", None)
        if not callable(get_num_hidden_layers):
            raise RuntimeError("[predo_tensor_index_locate] mlpm must provide get_num_hidden_layers")
        layer_num = int(get_num_hidden_layers())
        if layer_num <= 0:
            raise RuntimeError(
                "[predo_tensor_index_locate] invalid mlpm config: num_hidden_layers is missing or <= 0"
            )
        layer_ids = range(layer_num)

        get_tensor_index_layer_names = getattr(self.mlpm, "get_tensor_index_layer_names", None)
        get_tensor_index_general_names = getattr(self.mlpm, "get_tensor_index_general_names", None)
        get_tensor_index_skip_prefixes = getattr(self.mlpm, "get_tensor_index_skip_prefixes", None)
        get_tensor_expert_group_key = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        if (
            not callable(get_tensor_index_layer_names)
        ):
            raise RuntimeError(
                "[predo_tensor_index_locate] mlpm must provide get_tensor_index_layer_names"
            )
        if not callable(get_tensor_index_general_names):
            raise RuntimeError(
                "[predo_tensor_index_locate] mlpm must provide get_tensor_index_general_names"
            )
        if not callable(get_tensor_index_skip_prefixes):
            raise RuntimeError(
                "[predo_tensor_index_locate] mlpm must provide get_tensor_index_skip_prefixes"
            )
        if not callable(get_tensor_expert_group_key):
            raise RuntimeError(
                "[predo_tensor_index_locate] mlpm must provide get_tensor_expert_group_key"
            )
        layer_sagln_by_name: dict[str, int] = {}
        for layer_idx in layer_ids:
            names = list(get_tensor_index_layer_names(layer_idx))
            for n in names:
                if n in tensor_index_json:
                    layer_sagln_by_name[n] = layer_idx
        general_name_set = set(get_tensor_index_general_names())
        skip_prefixes = tuple(get_tensor_index_skip_prefixes())
        skipped_tensors: list[str] = []

        for tname, tmeta in tensor_index_json.items():
            if skip_prefixes and str(tname).startswith(skip_prefixes):
                skipped_tensors.append(tname)
                continue
            tsize = _tensor_size_bytes(tmeta)
            layer_idx = layer_sagln_by_name.get(tname)
            if layer_idx is not None:
                layer_sagln_tensors[layer_idx].append(tname)
                layer_sagln_sizes[layer_idx] += tsize
                continue

            expert_key = get_tensor_expert_group_key(tname)
            if expert_key is not None:
                expert_group_tensors[expert_key].append(tname)
                expert_group_sizes[expert_key] += tsize
            elif tname in general_name_set:
                general_tensors.append((tname, tsize))
            else:
                other_tensors.append((tname, tsize))

        tensor_to_device: dict[str, object] = {}
        first_device = device_names[0]
        get_experts_num = getattr(self.mlpm, "get_experts_num", None)
        if not callable(get_experts_num):
            raise RuntimeError("[predo_tensor_index_locate] mlpm must provide get_experts_num")
        experts_num = int(get_experts_num())
        if experts_num <= 0:
            raise RuntimeError("[predo_tensor_index_locate] invalid experts_num <= 0")

        # 第一步：专家逐个贪心放置，且同层 experts 尽量均匀分配到设备。
        # 同个 (layer, expert) 组内参数必须放在同一设备。
        # 额外产出：gate_up_proj/down_proj 的 expert_id -> device 映射（用于按专家均匀分配视图）。
        expert_id_device_map: dict[int, dict[str, dict[int, str]]] = {}
        layer_to_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for g in expert_group_tensors:
            layer_to_groups[g[0]].append(g)

        for layer_id, groups in sorted(layer_to_groups.items(), key=lambda x: x[0]):
            per_layer_device_counts = {d: 0 for d in device_names}
            groups_sorted = sorted(groups, key=lambda g: expert_group_sizes[g], reverse=True)
            gate_map: dict[int, str] = {}
            down_map: dict[int, str] = {}
            for g in groups_sorted:
                gsize = expert_group_sizes[g]
                group_tensors = expert_group_tensors[g]
                group_tensor_name = group_tensors[0] if group_tensors else f"layer={layer_id},expert={g[1]}"

                def _expert_cand_key(dev_name: str):
                    rem = device_remaining[dev_name]
                    fits = 0 if rem >= gsize else 1
                    initial = max(1, device_initial_free[dev_name])
                    after_used_ratio = (initial - (rem - gsize)) / initial if rem >= gsize else float("inf")
                    return (
                        fits,
                        per_layer_device_counts[dev_name],
                        after_used_ratio,
                        -rem,
                        dev_name,
                    )

                chosen = min(device_names, key=_expert_cand_key)
                if device_remaining[chosen] < gsize:
                    raise RuntimeError(
                        f"[predo_tensor_index_locate] insufficient GPU memory for "
                        f"layer={layer_id} expert={g[1]} size={gsize} bytes, "
                        f"remaining={device_remaining}"
                    )
                per_layer_device_counts[chosen] += 1
                if g[1] >= 0:
                    device_remaining[chosen] -= gsize
                    for tname in group_tensors:
                        tensor_to_device[tname] = chosen
                    gate_map[g[1]] = chosen
                    down_map[g[1]] = chosen
                else:
                    # fused expert bank（如 Gemma4）在 tensor_index 中是整层参数，无法按 tensor 级别拆分。
                    # 这里额外给出“按 expert_id 硬配额均匀分配”的逻辑分配表，供后续调度/统计使用。
                    virtual_remaining = dict(device_remaining)
                    if gsize % experts_num != 0:
                        raise RuntimeError(
                            f"[predo_tensor_index_locate] fused expert group bytes not divisible by experts_num: "
                            f"group={group_tensor_name}, gsize={gsize}, experts_num={experts_num}"
                        )
                    per_expert_bytes = gsize // experts_num
                    if per_expert_bytes <= 0:
                        raise RuntimeError(
                            f"[predo_tensor_index_locate] invalid per_expert_bytes for fused group: "
                            f"group={group_tensor_name}, gsize={gsize}, experts_num={experts_num}"
                        )
                    num_devices = max(1, len(device_names))
                    base_quota, rem_quota = divmod(experts_num, num_devices)
                    target_counts = {
                        d: base_quota + (1 if i < rem_quota else 0)
                        for i, d in enumerate(device_names)
                    }
                    assigned_counts = {d: 0 for d in device_names}
                    for expert_id in range(experts_num):
                        candidate_devices = [
                            d for d in device_names if assigned_counts[d] < target_counts[d]
                        ]
                        if not candidate_devices:
                            raise RuntimeError(
                                f"[predo_tensor_index_locate] no candidate device for layer={layer_id} "
                                f"expert_id={expert_id} under quota targets={target_counts}"
                            )

                        def _expert_id_cand_key(dev_name: str):
                            rem = virtual_remaining[dev_name]
                            fits = 0 if rem >= per_expert_bytes else 1
                            initial = max(1, device_initial_free[dev_name])
                            after_used_ratio = (
                                (initial - (rem - per_expert_bytes)) / initial
                                if rem >= per_expert_bytes
                                else float("inf")
                            )
                            return (
                                fits,
                                assigned_counts[dev_name],
                                after_used_ratio,
                                -rem,
                                dev_name,
                            )

                        e_chosen = min(candidate_devices, key=_expert_id_cand_key)
                        if virtual_remaining[e_chosen] < per_expert_bytes:
                            raise RuntimeError(
                                f"[predo_tensor_index_locate] insufficient virtual memory for "
                                f"layer={layer_id} expert_id={expert_id} split bytes={per_expert_bytes}"
                            )
                        assigned_counts[e_chosen] += 1
                        virtual_remaining[e_chosen] -= per_expert_bytes
                        gate_map[expert_id] = e_chosen
                        down_map[expert_id] = e_chosen

                    assigned_total = per_expert_bytes * sum(assigned_counts.values())
                    if assigned_total != gsize:
                        raise RuntimeError(
                            f"[predo_tensor_index_locate] fused split byte mismatch: "
                            f"group={group_tensor_name}, assigned_total={assigned_total}, gsize={gsize}"
                        )

                    # 使用按专家拆分后的真实分配结果更新设备剩余显存，确保 expected_used_bytes 精确。
                    for dev_name, count in assigned_counts.items():
                        split_bytes = per_expert_bytes * count
                        if split_bytes <= 0:
                            continue
                        if device_remaining[dev_name] < split_bytes:
                            raise RuntimeError(
                                f"[predo_tensor_index_locate] insufficient GPU memory after fused split: "
                                f"group={group_tensor_name}, device={dev_name}, "
                                f"need={split_bytes}, remaining={device_remaining[dev_name]}"
                            )
                        device_remaining[dev_name] -= split_bytes

                    def _device_to_expert_ids(eid_map: dict[int, str]) -> dict[str, list[int]]:
                        out = {d: [] for d in device_names}
                        for eid, dev in sorted(eid_map.items(), key=lambda x: x[0]):
                            out[dev].append(int(eid))
                        return out

                    gate_device_to_ids = _device_to_expert_ids(gate_map)
                    down_device_to_ids = _device_to_expert_ids(down_map)
                    for tname in group_tensors:
                        if ".experts.gate_up_proj" in tname:
                            tensor_to_device[tname] = {
                                "default_device": chosen,
                                "expert_id_to_device": dict(gate_map),
                                "device_to_expert_ids": gate_device_to_ids,
                            }
                        elif ".experts.down_proj" in tname:
                            tensor_to_device[tname] = {
                                "default_device": chosen,
                                "expert_id_to_device": dict(down_map),
                                "device_to_expert_ids": down_device_to_ids,
                            }
                        else:
                            tensor_to_device[tname] = chosen
            if gate_map or down_map:
                # 约束校验：同一(layer, expert_id)下 gate_up_proj / down_proj 必须同卡。
                shared_eids = set(gate_map.keys()) & set(down_map.keys())
                for eid in shared_eids:
                    if gate_map[eid] != down_map[eid]:
                        raise RuntimeError(
                            f"[predo_tensor_index_locate] gate/down device mismatch "
                            f"at layer={layer_id}, expert_id={eid}, "
                            f"gate={gate_map[eid]}, down={down_map[eid]}"
                        )
                expert_id_device_map[layer_id] = {
                    "gate_up_proj": gate_map,
                    "down_proj": down_map,
                }

        # 第二步：非专家参数（general + layer_sagln）统一放在第一张卡。
        non_expert_total_size = sum(tsize for _, tsize in general_tensors) + sum(layer_sagln_sizes.values())
        if device_remaining[first_device] < non_expert_total_size:
            raise RuntimeError(
                f"[predo_tensor_index_locate] insufficient GPU memory on first device "
                f"{first_device} for non-expert tensors total={non_expert_total_size} bytes, "
                f"remaining={device_remaining[first_device]} bytes"
            )
        for tname, tsize in general_tensors:
            tensor_to_device[tname] = first_device
            device_remaining[first_device] -= tsize
        for layer_id, names in sorted(layer_sagln_tensors.items(), key=lambda x: x[0]):
            for tname in names:
                tensor_to_device[tname] = first_device
            device_remaining[first_device] -= int(layer_sagln_sizes[layer_id])

        # 规则5：不允许有未处理参数（other_tensors）。
        if other_tensors:
            sample_names = [name for name, _ in other_tensors[:10]]
            raise RuntimeError(
                "[predo_tensor_index_locate] found unhandled tensors not in "
                "expert/general/layer_sagln groups, "
                f"count={len(other_tensors)}, sample={sample_names}"
            )

        # 第三步：无（layer_sagln 已在第二步统一放到 first_device）。

        device_usage_ratio = {}
        device_expected_used_bytes = {}
        device_expected_used_mib = {}
        for d in device_names:
            initial = max(1, device_initial_free[d])
            used_bytes = initial - device_remaining[d]
            device_usage_ratio[d] = used_bytes / initial
            device_expected_used_bytes[d] = used_bytes
            device_expected_used_mib[d] = used_bytes / (1024.0 * 1024.0)
        logger.info(
            "[predo_tensor_index_locate] tensors=%d skipped_tensors=%d sagln_layers=%d expert_groups=%d general_tensors=%d first_device=%s "
            "remaining=%s expected_used_bytes=%s expected_used_mib=%s usage_ratio=%s",
            len(tensor_to_device),
            len(skipped_tensors),
            len(layer_sagln_tensors),
            len(expert_group_tensors),
            len(general_tensors),
            first_device,
            {d: device_remaining[d] for d in device_names},
            device_expected_used_bytes,
            device_expected_used_mib,
            device_usage_ratio,
        )
        for layer_id, layer_plan in sorted(expert_id_device_map.items(), key=lambda x: x[0]):
            counts = {d: 0 for d in device_names}
            for _expert_id, dev in sorted(layer_plan["gate_up_proj"].items(), key=lambda x: x[0]):
                counts[dev] += 1
            logger.info(
                "[predo_tensor_index_locate] layer=%d gate_up/down expert_id_device_counts=%s",
                layer_id,
                counts,
            )
        return tensor_to_device, device_expected_used_bytes, expert_id_device_map

    def generate_chunk_task_id(self, chunk_items: list[dict]) -> list[dict]:
        """
        1) 为生成的 chunk 填充task_id, 按照 layer 顺序, layer 中按照 generate -> self_attn, gate, layernorm -> experts 顺序.
        2) 请参考 predo_tensor_index_locate 中的逻辑，生成 task_id
        3) 需正确处理专家的情况，专家tensor需要按照专家数量拆分，每个专家一个chunk。
        """
        if not isinstance(chunk_items, list):
            raise TypeError("chunk_items must be a list")
        if not chunk_items:
            return []

        get_tensor_expert_group_key = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        get_tensor_index_general_names = getattr(self.mlpm, "get_tensor_index_general_names", None)
        get_tensor_index_layer_names = getattr(self.mlpm, "get_tensor_index_layer_names", None)
        if not callable(get_tensor_expert_group_key):
            raise RuntimeError("mlpm must provide get_tensor_expert_group_key")
        if not callable(get_tensor_index_general_names):
            raise RuntimeError("mlpm must provide get_tensor_index_general_names")
        if not callable(get_tensor_index_layer_names):
            raise RuntimeError("mlpm must provide get_tensor_index_layer_names")

        general_names = list(get_tensor_index_general_names())
        general_name_set = set(general_names)
        general_name_order = {name: i for i, name in enumerate(general_names)}

        layer_ids: set[int] = set()
        for item in chunk_items:
            name = str(item["tensor_name"])
            m = re.search(r"layers\.(\d+)\.", name)
            if m:
                layer_ids.add(int(m.group(1)))
            else:
                expert_key = get_tensor_expert_group_key(name)
                if expert_key is not None:
                    layer_ids.add(int(expert_key[0]))

        # Keep layer-order names as list order from mlpm instead of a set, so
        # task_id can follow self_attn -> gate -> layernorm ordering precisely.
        layer_name_order: dict[tuple[int, str], int] = {}
        for layer_idx in sorted(layer_ids):
            names = list(get_tensor_index_layer_names(layer_idx))
            for i, n in enumerate(names):
                layer_name_order[(int(layer_idx), str(n))] = i

        # For fused expert banks (expert_key=(layer,-1)), infer an expert rank by
        # sorted src_offset within each tensor name group.
        fused_expert_src_rank: dict[tuple[str, int], int] = {}
        fused_src_offsets_by_name: dict[str, set[int]] = {}
        for it in chunk_items:
            name = str(it["tensor_name"])
            expert_key = get_tensor_expert_group_key(name)
            if expert_key is None or int(expert_key[1]) >= 0:
                continue
            fused_src_offsets_by_name.setdefault(name, set()).add(int(it["src_offset"]))
        for name, offsets in fused_src_offsets_by_name.items():
            for rank, src in enumerate(sorted(offsets)):
                fused_expert_src_rank[(name, int(src))] = rank

        def _parse_layer_idx(name: str) -> int:
            # General/generate tensors are placed before layer-scoped tensors.
            if name in general_name_set:
                return -1
            expert_key = get_tensor_expert_group_key(name)
            if expert_key is not None:
                return int(expert_key[0])
            m = re.search(r"layers\.(\d+)\.", name)
            if m:
                return int(m.group(1))
            return 10**9

        def _stage(name: str, layer_idx: int) -> int:
            # 0: generate/general -> 1: layer tensors -> 2: experts -> 3: others
            if name in general_name_set:
                return 0
            if (layer_idx, name) in layer_name_order:
                return 1
            expert_key = get_tensor_expert_group_key(name)
            if expert_key is not None:
                return 2
            return 3

        def _layer_local_order(name: str, layer_idx: int) -> int:
            # Respect mlpm-defined per-layer order (self_attn -> gate -> layernorm ...).
            return int(layer_name_order.get((layer_idx, name), 10**9))

        def _expert_order(item: dict) -> tuple[int, int]:
            name = str(item["tensor_name"])
            expert_key = get_tensor_expert_group_key(name)
            if expert_key is None:
                return (10**9, 10**9)
            expert_id = int(expert_key[1])
            if expert_id >= 0:
                rank = expert_id
            else:
                rank = fused_expert_src_rank.get((name, int(item["src_offset"])), 10**9)
            # Keep gate_up before down for same expert if both exist.
            if "gate_up_proj" in name:
                tensor_kind = 0
            elif "down_proj" in name:
                tensor_kind = 1
            else:
                tensor_kind = 2
            return (rank, tensor_kind)

        def _sort_key(it: dict) -> tuple[int, int, int, int, int, int, str, int, int, int]:
            name = str(it["tensor_name"])
            layer_idx = _parse_layer_idx(name)
            stage = _stage(name, layer_idx)
            general_rank = int(general_name_order.get(name, 10**9))
            layer_rank = _layer_local_order(name, layer_idx)
            expert_rank, expert_kind = _expert_order(it)
            return (
                layer_idx,
                stage,
                general_rank,
                layer_rank,
                expert_rank,
                expert_kind,
                name,
                int(it["device_idx"]),
                int(it["src_offset"]),
                int(it["dst_offset"]),
            )

        sorted_items = sorted(chunk_items, key=_sort_key)

        for task_id, item in enumerate(sorted_items, start=1):
            item["task_id"] = task_id
        return sorted_items

    def _copy_chunks_as_flat_rows(self, tensor_copy_chunks, tensor_task_queue_index):
        """list[dict] 直接展平；dict[device,tuples] 需 index。返回 (rows, 是否 dict 输入)。

        查找策略：按 **task_id** 反查 tensor_task_queue_index，不依赖 queue_pos。
        这使得函数在 ``tensor_copy_chunks`` 被 split/重建（位置重新编号）后仍然正确，
        避免了旧 queue_pos 与新位置序号不一致导致的 name→task_id 错位。
        """
        if isinstance(tensor_copy_chunks, list):
            rows = []
            for x in tensor_copy_chunks:
                if not isinstance(x, dict):
                    raise TypeError("tensor_copy_chunks list items must be dict")
                rows.append(dict(x))
            return rows, False
        if not isinstance(tensor_copy_chunks, dict):
            raise TypeError("tensor_copy_chunks must be list[dict] or dict[device, list[tuple]]")
        if not isinstance(tensor_task_queue_index, dict):
            raise TypeError("tensor_task_queue_index required when tensor_copy_chunks is dict")

        # Build (device_idx, task_id) → [tensor_name, ...] for O(1) lookups.
        # task_id is stable across all splits/rebuilds; queue_pos is not.
        tid_dev_to_names: dict[tuple[int, int], list[str]] = {}
        for tname, entries in tensor_task_queue_index.items():
            if not isinstance(entries, list):
                continue
            for rec in entries:
                if isinstance(rec, dict):
                    tid = int(rec.get("task_id", 0))
                    dev = int(rec.get("device_idx", 0))
                    if tid > 0:
                        tid_dev_to_names.setdefault((dev, tid), []).append(str(tname))

        rows = []
        for dev in sorted(tensor_copy_chunks.keys(), key=lambda x: int(x)):
            d = int(dev)
            lst = tensor_copy_chunks[dev]
            if not isinstance(lst, list):
                raise TypeError(f"tensor_copy_chunks[{dev!r}] must be a list")
            for tup in lst:
                if not isinstance(tup, (tuple, list)) or len(tup) < 5:
                    raise TypeError("chunk tuple needs at least 5 fields")
                tid = int(tup[4])
                names = tid_dev_to_names.get((d, tid))
                if not names:
                    raise ValueError(
                        f"missing tensor_task_queue_index for device={d} task_id={tid}"
                    )
                for nm in sorted(set(names)):
                    rows.append(
                        {
                            "tensor_name": nm,
                            "device_idx": d,
                            "src_offset": int(tup[0]),
                            "size": int(tup[1]),
                            "dst_offset": int(tup[2]),
                            "handle_idx": int(tup[3]),
                            "task_id": tid,
                            "priority": int(tup[5]) if len(tup) > 5 else 0,
                            "reorder_hint": bool(tup[6]) if len(tup) > 6 else False,
                        }
                    )
        return rows, True

    @staticmethod
    def _rows_to_device_chunk_dict(rows):
        by = {}
        for it in rows:
            d = int(it["device_idx"])
            by.setdefault(d, []).append(
                (
                    int(it["src_offset"]),
                    int(it["size"]),
                    int(it["dst_offset"]),
                    int(it["handle_idx"]),
                    int(it["task_id"]),
                    int(it.get("priority", 0)),
                    bool(it.get("reorder_hint", False)),
                )
            )
        return {k: by[k] for k in sorted(by.keys())}

    @staticmethod
    def _pop_offsets_for_extracted(tensor_device_offsets, extracted):
        out = {}
        if not isinstance(tensor_device_offsets, dict) or not extracted:
            return out
        seen = set()
        for it in extracted:
            d, nm = int(it["device_idx"]), str(it["tensor_name"])
            k = (d, nm)
            if k in seen:
                continue
            seen.add(k)
            dm = tensor_device_offsets.get(d)
            if isinstance(dm, dict) and nm in dm:
                out.setdefault(d, {})[nm] = int(dm[nm])
        for it in extracted:
            d, nm = int(it["device_idx"]), str(it["tensor_name"])
            dm = tensor_device_offsets.get(d)
            if isinstance(dm, dict) and nm in dm:
                del dm[nm]
        return out

    def _prune_tensor_device_offsets_to_copy_chunks(
        self,
        tensor_device_offsets: dict,
        tensor_copy_chunks,
        tensor_task_queue_index,
    ) -> None:
        """
        删除 ``tensor_device_offsets`` 中已无任何 remainder copy 行的张量键。

        ``_pop_offsets_for_extracted`` 依赖 ext 行里的 ``tensor_name`` 与 offsets 键一致；
        若某层权重在 ``tensor_index`` 里存在、却未列入 ``get_tensor_index_layer_names``，
        则不会被抽进 SAGL，offsets 也不会被 pop，会误留在 experts 侧。此处以当前
        ``tensor_copy_chunks`` 展平后的 (device, name) 为权威集合做一次对齐。
        """
        if not isinstance(tensor_device_offsets, dict) or not tensor_device_offsets:
            return
        if tensor_task_queue_index is None or not isinstance(
            tensor_task_queue_index, dict
        ):
            return
        try:
            rows, _ = self._copy_chunks_as_flat_rows(
                tensor_copy_chunks, tensor_task_queue_index
            )
        except (TypeError, ValueError):
            return
        keep: set[tuple[int, str]] = set()
        for it in rows:
            keep.add((int(it["device_idx"]), str(it["tensor_name"])))
        for dev_idx, name_map in list(tensor_device_offsets.items()):
            if not isinstance(name_map, dict):
                continue
            d = int(dev_idx)
            for nm in list(name_map.keys()):
                if (d, str(nm)) not in keep:
                    del name_map[nm]
            dm = tensor_device_offsets.get(dev_idx)
            if isinstance(dm, dict) and not dm:
                del tensor_device_offsets[dev_idx]

    def _tensor_device_offsets_expert_tensors_only(
        self, tensor_device_offsets: dict
    ) -> dict:
        """
        仅保留专家相关张量的 offset 项（``get_tensor_expert_group_key(name)`` 非空）。

        remainder 的 ``tensor_device_offsets`` 可能与 ``tensor_copy_chunks`` 因命名/抽取
        边界残留 self_attn 等键；第二次 GPU load 与 ``restore_experts_state_dict`` 只应对
        专家权重视图使用该字典，故在此显式过滤。
        """
        gk_fn = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        if not callable(gk_fn):
            return {int(d): dict(m) for d, m in tensor_device_offsets.items() if isinstance(m, dict)}
        out: dict[int, dict[str, int]] = {}
        for dev_idx, name_map in tensor_device_offsets.items():
            if not isinstance(name_map, dict):
                continue
            d = int(dev_idx)
            for nm, off in name_map.items():
                if gk_fn(str(nm)) is None:
                    continue
                out.setdefault(d, {})[str(nm)] = int(off)
        return out

    @staticmethod
    def _prune_device_offsets_to_copy_chunks_by_task_id(
        tensor_device_offsets: dict,
        tensor_copy_chunks: dict,
        tensor_task_queue_index: dict,
    ) -> dict:
        """
        根据 ``tensor_copy_chunks`` 中实际存在的 task_id 集合裁剪 ``tensor_device_offsets``。

        ``_prune_tensor_device_offsets_to_copy_chunks`` 依赖 queue_pos 位置对齐，split/重建后
        位置序号变化会误删有效条目。本函数改用 task_id 匹配：

        1. 从 ``tensor_copy_chunks`` 提取所有 task_id（tuple 第 5 字段，索引 4）。
        2. 在全局 ``tensor_task_queue_index`` 中反查哪些 ``(device_idx, tensor_name)`` 对应
           这些 task_id，构成合法集 ``keep``。
        3. 删除 ``tensor_device_offsets`` 中不在 ``keep`` 里的条目。

        返回裁剪后的新字典（不修改原 ``tensor_device_offsets``）。
        """
        if not isinstance(tensor_copy_chunks, dict) or not isinstance(tensor_task_queue_index, dict):
            return {int(d): dict(m) for d, m in tensor_device_offsets.items() if isinstance(m, dict)}

        # step 1 – collect task_ids present in the copy chunks
        chunk_task_ids: set[int] = set()
        for rows in tensor_copy_chunks.values():
            if not isinstance(rows, list):
                continue
            for tup in rows:
                if isinstance(tup, (tuple, list)) and len(tup) > 4:
                    chunk_task_ids.add(int(tup[4]))

        # step 2 – reverse-map task_id → (device_idx, tensor_name)
        keep: set[tuple[int, str]] = set()
        for tname, entries in tensor_task_queue_index.items():
            if not isinstance(entries, list):
                continue
            for rec in entries:
                if isinstance(rec, dict):
                    tid = int(rec.get("task_id", 0))
                    if tid in chunk_task_ids:
                        keep.add((int(rec["device_idx"]), str(tname)))

        # step 3 – build filtered offsets dict
        out: dict[int, dict[str, int]] = {}
        for dev_idx, name_map in tensor_device_offsets.items():
            if not isinstance(name_map, dict):
                continue
            d = int(dev_idx)
            for nm, off in name_map.items():
                if (d, str(nm)) in keep:
                    out.setdefault(d, {})[str(nm)] = int(off)
        return out

    @staticmethod
    def _dedup_rows_by_physical_copy(rows):
        """按物理拷贝区间去重（忽略 task_id/priority/reorder），用于抽取 ext 时消除同区间多名引用。"""
        dedup = {}
        for it in rows:
            key = (
                int(it["device_idx"]),
                int(it["src_offset"]),
                int(it["size"]),
                int(it["dst_offset"]),
                int(it["handle_idx"]),
            )
            if key not in dedup:
                dedup[key] = dict(it)
        return list(dedup.values())

    @staticmethod
    def _dedup_rows_by_copy_task(rows):
        """
        按完整 copy task 去重（含 task_id / priority / reorder_hint）。

        ``_copy_chunks_as_flat_rows`` 在 dict 输入时会对同一 queue_pos 按每个
        tensor_name 展开一行；若直接写回 chunk dict，会产生重复条目。
        此函数以 ``(device_idx, src_offset, size, dst_offset, handle_idx,
        task_id, priority, reorder_hint)`` 为 key，保留第一条，去除重复。
        """
        dedup: dict[tuple, dict] = {}
        for it in rows:
            key = (
                int(it["device_idx"]),
                int(it["src_offset"]),
                int(it["size"]),
                int(it["dst_offset"]),
                int(it["handle_idx"]),
                int(it.get("task_id", 0)),
                int(it.get("priority", 0)),
                bool(it.get("reorder_hint", False)),
            )
            if key not in dedup:
                dedup[key] = dict(it)
        return list(dedup.values())

    @staticmethod
    def _merge_tensor_device_offsets(*offset_maps):
        merged = {}
        for src in offset_maps:
            if not isinstance(src, dict):
                continue
            for dev_idx, name_to_offset in src.items():
                if not isinstance(name_to_offset, dict):
                    continue
                merged.setdefault(int(dev_idx), {}).update(name_to_offset)
        return merged

    @staticmethod
    def _merge_tensor_copy_chunks(*chunk_maps):
        merged = {}
        for src in chunk_maps:
            if not isinstance(src, dict):
                continue
            for dev_idx, chunks in src.items():
                if not isinstance(chunks, list):
                    continue
                merged.setdefault(int(dev_idx), []).extend(chunks)
        return merged

    def _expert_task_id_set_from_queue(self, tensor_task_queue_index) -> set[int]:
        """
        由 `generate_chunk` 写入的 `tensor_task_queue_index` 推导「专家」task_id 集合：
        凡 `get_tensor_expert_group_key(tensor_name)` 非空的张量所登记的 task_id。
        用于拆 general / SAGL 时强制保留在 remainder，保证第二次 `load_into_gpu_async`
        仍包含全部专家拷贝且无需改动全局 task_id。
        """
        if not isinstance(tensor_task_queue_index, dict):
            return set()
        gk = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        if not callable(gk):
            return set()
        out: set[int] = set()
        for tname, entries in tensor_task_queue_index.items():
            if gk(str(tname)) is None:
                continue
            if not isinstance(entries, list):
                continue
            for rec in entries:
                if isinstance(rec, dict):
                    tid = int(rec.get("task_id", 0))
                    if tid > 0:
                        out.add(tid)
        return out

    def _log_tensor_split_integrity_check(
        self,
        tensor_task_queue_index,
        general_sagl_copy_chunks: dict,
        experts_tensor_copy_chunks: dict,
        experts_tensor_device_offsets: dict,
    ) -> None:
        """Log split-integrity between general/SAGL and experts copy buckets.

        Verifies (via logs):
        1. general_sagl_copy_chunks ∪ experts_tensor_copy_chunks = all original task_ids
        2. The two sets are disjoint
        3. experts bucket has all expert task_ids and only expert task_ids
        4. experts_tensor_device_offsets has no non-expert tensor names
        """
        _expert_tids_queue: set[int] = self._expert_task_id_set_from_queue(
            tensor_task_queue_index
        )
        _all_queue_tids: set[int] = set()
        for _entries in tensor_task_queue_index.values():
            if isinstance(_entries, list):
                for _rec in _entries:
                    if isinstance(_rec, dict) and _rec.get("task_id"):
                        _all_queue_tids.add(int(_rec["task_id"]))

        def _tids_from_chunks(chunks: dict) -> set[int]:
            out: set[int] = set()
            for _rows in chunks.values():
                for _tup in _rows:
                    if isinstance(_tup, (tuple, list)) and len(_tup) > 4:
                        out.add(int(_tup[4]))
            return out

        _sagl_tids = _tids_from_chunks(general_sagl_copy_chunks)
        _exp_tids = _tids_from_chunks(experts_tensor_copy_chunks)
        _overlap = _sagl_tids & _exp_tids
        _lost = _all_queue_tids - (_sagl_tids | _exp_tids)
        _extra = (_sagl_tids | _exp_tids) - _all_queue_tids
        _missing_experts = _expert_tids_queue - _exp_tids
        _nonexpert_in_exp = _exp_tids - _expert_tids_queue

        _exp_rows = {int(d): len(r) for d, r in experts_tensor_copy_chunks.items()}
        _sagl_rows = {int(d): len(r) for d, r in general_sagl_copy_chunks.items()}

        _gk_fn = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        _off_nonexpert: list[str] = []
        _off_expert_count = 0
        for _d_off, _nm_map in experts_tensor_device_offsets.items():
            if isinstance(_nm_map, dict):
                for _nm in _nm_map:
                    if callable(_gk_fn) and _gk_fn(str(_nm)) is None:
                        _off_nonexpert.append(str(_nm))
                    else:
                        _off_expert_count += 1

        logger.info(
            "[split-check] queue_tids=%d  sagl=%d  experts=%d"
            " | overlap=%d  lost=%d  extra=%d"
            " | missing_experts=%d  nonexpert_in_experts=%d"
            " | sagl_rows=%s  exp_rows=%s"
            " | off_expert=%d  off_nonexpert=%d",
            len(_all_queue_tids), len(_sagl_tids), len(_exp_tids),
            len(_overlap), len(_lost), len(_extra),
            len(_missing_experts), len(_nonexpert_in_exp),
            _sagl_rows, _exp_rows,
            _off_expert_count, len(_off_nonexpert),
        )
        if _overlap:
            logger.error("[split-check] OVERLAP task_ids (in both buckets): %s", sorted(_overlap)[:16])
        if _lost:
            logger.error("[split-check] LOST task_ids (in neither bucket): %s", sorted(_lost)[:16])
        if _missing_experts:
            logger.error("[split-check] MISSING expert task_ids: %s", sorted(_missing_experts)[:16])
        if _nonexpert_in_exp:
            logger.error("[split-check] NON-EXPERT task_ids leaked into experts bucket: %s",
                         sorted(_nonexpert_in_exp)[:16])
        if _off_nonexpert:
            logger.error("[split-check] NON-EXPERT names in experts_tensor_device_offsets: %s",
                         _off_nonexpert[:8])

    def general_chunk_from_chunks(
        self, tensor_device_offsets, tensor_copy_chunks, tensor_task_queue_index=None
    ):
        """抽出 general；原地改 chunks/offsets。返回 (提取offsets, 提取chunks, 余下offsets, 余下chunks)。

        凡在 `tensor_task_queue_index` 中登记为专家张量的 task_id 一律留在 remainder，
        不参与 general 抽取，以保持与 MoE / replica2 提交的全局 task_id 一致。
        """
        rows, as_dict = self._copy_chunks_as_flat_rows(tensor_copy_chunks, tensor_task_queue_index)
        expert_tids = self._expert_task_id_set_from_queue(tensor_task_queue_index)
        gn = getattr(self.mlpm, "get_tensor_index_general_names", None)
        if not callable(gn):
            raise RuntimeError("mlpm must provide get_tensor_index_general_names")
        names = set(str(x) for x in gn())
        ext, rem = [], []
        for it in rows:
            tid = int(it.get("task_id", 0))
            if tid in expert_tids:
                rem.append(dict(it))
                continue
            if str(it["tensor_name"]) in names:
                ext.append(dict(it))
            else:
                rem.append(dict(it))
        eo = self._pop_offsets_for_extracted(tensor_device_offsets, ext)
        if as_dict:
            tensor_copy_chunks.clear()
            tensor_copy_chunks.update(
                self._rows_to_device_chunk_dict(self._dedup_rows_by_copy_task(rem))
            )
        else:
            tensor_copy_chunks[:] = self._dedup_rows_by_copy_task(rem)
        ext_copy_rows = self._dedup_rows_by_physical_copy(ext) if ext else []
        ec = self._rows_to_device_chunk_dict(ext_copy_rows) if as_dict else ext_copy_rows
        self._prune_tensor_device_offsets_to_copy_chunks(
            tensor_device_offsets, tensor_copy_chunks, tensor_task_queue_index
        )
        return eo, ec, tensor_device_offsets, tensor_copy_chunks

    def self_attn_gate_layernorm_tasks_ids_chunk_from_chunks(
        self, layer_idx, tensor_device_offsets, tensor_copy_chunks, tensor_task_queue_index=None
    ):
        """抽取当前 ``layer_idx`` 的非专家张量；返回四元组：

        ``(eo, ec, tensor_device_offsets, tensor_copy_chunks)`` — 与 ``general_chunk_from_chunks`` 相同结构：
        - ``eo`` / ``ec``：从 remainder **抽出**到 SAGL 侧的 device_offsets 与 copy_chunks；
        - 后两个：原地更新后的 **remainder**（未抽走、仍留在第二次 GPU load 的张量）。

        专家张量一律留在 remainder；
        非专家仅在其层号等于 ``layer_idx`` 时进入 SAGL，其他层保持在 remainder，
        由后续层循环继续抽取。
        """
        layer_idx = int(layer_idx)
        rows, as_dict = self._copy_chunks_as_flat_rows(tensor_copy_chunks, tensor_task_queue_index)
        gk = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        glayer = getattr(self.mlpm, "get_tensor_layer_idx", None)
        if not callable(gk) or not callable(glayer):
            raise RuntimeError(
                "mlpm must provide get_tensor_expert_group_key and get_tensor_layer_idx"
            )
        ext, rem = [], []
        for it in rows:
            tname = str(it["tensor_name"])
            if gk(tname) is not None:
                rem.append(dict(it))
                continue
            t_layer = glayer(tname)
            if t_layer is not None and int(t_layer) == layer_idx:
                ext.append(dict(it))
            else:
                rem.append(dict(it))
        eo = self._pop_offsets_for_extracted(tensor_device_offsets, ext)
        if as_dict:
            tensor_copy_chunks.clear()
            tensor_copy_chunks.update(
                self._rows_to_device_chunk_dict(self._dedup_rows_by_copy_task(rem))
            )
        else:
            tensor_copy_chunks[:] = self._dedup_rows_by_copy_task(rem)
        ext_copy_rows = self._dedup_rows_by_physical_copy(ext) if ext else []
        ec = self._rows_to_device_chunk_dict(ext_copy_rows) if as_dict else ext_copy_rows
        self._prune_tensor_device_offsets_to_copy_chunks(
            tensor_device_offsets, tensor_copy_chunks, tensor_task_queue_index
        )
        return eo, ec, tensor_device_offsets, tensor_copy_chunks

    def nonexpert_chunk_from_chunks(
        self, tensor_device_offsets, tensor_copy_chunks, tensor_task_queue_index=None
    ):
        """
        一次性抽取全部非专家张量（跨所有 layer），将专家与非专家清晰二分。

        返回:
        - ``eo`` / ``ec``: 非专家侧（用于 general+sagl 加载）
        - 后两个: 专家侧 remainder（用于 experts 加载）
        """
        rows, as_dict = self._copy_chunks_as_flat_rows(
            tensor_copy_chunks, tensor_task_queue_index
        )
        expert_tids = self._expert_task_id_set_from_queue(tensor_task_queue_index)
        gk = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        if not callable(gk):
            raise RuntimeError("mlpm must provide get_tensor_expert_group_key")

        ext, rem = [], []
        for it in rows:
            tname = str(it["tensor_name"])
            tid = int(it.get("task_id", 0))
            if gk(tname) is not None or tid in expert_tids:
                rem.append(dict(it))
            else:
                ext.append(dict(it))

        eo = self._pop_offsets_for_extracted(tensor_device_offsets, ext)
        rem_dedup = MLPLLM._dedup_rows_by_copy_task(rem)
        if as_dict:
            tensor_copy_chunks.clear()
            tensor_copy_chunks.update(self._rows_to_device_chunk_dict(rem_dedup))
        else:
            tensor_copy_chunks[:] = rem_dedup

        ext_copy_rows = MLPLLM._dedup_rows_by_physical_copy(ext) if ext else []
        ec = self._rows_to_device_chunk_dict(ext_copy_rows) if as_dict else ext_copy_rows
        # NOTE: _prune_tensor_device_offsets_to_copy_chunks is NOT called here.
        # After the split, tensor_copy_chunks is rebuilt with new sequential positions,
        # while tensor_task_queue_index still holds the original queue_pos values.
        # Calling _copy_chunks_as_flat_rows on the post-split chunks would produce
        # wrong (device, name) → position mappings, causing valid expert entries
        # to be incorrectly pruned.  The caller (test_mp_generate_multi_device_layer)
        # applies _tensor_device_offsets_expert_tensors_only for the final filter.
        return eo, ec, tensor_device_offsets, tensor_copy_chunks

    def generate_chunk(self, tensor_to_device: dict, tensor_index_json: dict, device_memory_used: dict):
        """
        作用：根据 `tensor_to_device` 生成可直接提交给 sllm_store 的 copy chunks，
        并产出 `tensor_device_offsets`（用于 restore）与每卡最终分配大小。

        规则：
        1) 非 fused tensor：按 tensor 生成单个 chunk。
        2) fused expert tensor（`...experts.gate_up_proj/down_proj`）：
           按专家维拆分为 `...experts.{eid}.gate_up_proj/down_proj`，每个专家一个 chunk。
        3) 单专家 chunk 大小必须正确：
           先由 `total_size / experts_num` 计算，再与 `shape[1:] * dtype_bytes` 交叉校验。
        4) 所有 chunk 最终调用 `generate_chunk_task_id` 生成稳定 `task_id`；
           `priority` 固定 low，`reorder_hint` 固定 False。
        5) `tensor_device_offsets` 需包含拆分后的专家名（含 `expert_id`），
           使 `restore_state_dict` 可按专家视图恢复。
        6) 实现应保持通用性与可读性：命名规则稳定、校验路径明确，
           便于后续扩展到其它专家命名或 dtype 场景。
        7) 同时产出每个 `tensor_name` 对应的 `task_id` 及其在 copy 队列中的位置。
        8) 需处理 同个拷贝的情况，例如 (0, 1476395008, 11418992640, 0, 1, 0, False), (0, 1476395008, 11418992640, 0, 2, 0, False) , copy_chunks 只生成一个拷贝任务，但tensor_device_offsets 和 tensor_task_queue_index  需要生成所有对应的相关tensor_name的内容
        9) 确保 同层 同设备 的 专家 gate_up_proj 和 down_proj 的 拷贝的放置 是连续的，不要交叉拷贝，例如 0.gate_up_proj 和 3.gate_up_proj 是连续的都在同个设备, 0.down_proj 和 1.down_proj 是连续的，以此类推, 这样可以将其聚合为连续的 张量，方便后续 restore_state_dict 为packed 的恢复
        """
        if not isinstance(tensor_to_device, dict):
            raise TypeError("tensor_to_device must be a dict")
        if not isinstance(tensor_index_json, dict):
            raise TypeError("tensor_index_json must be a dict")
        if not isinstance(device_memory_used, dict):
            raise TypeError("device_memory_used must be a dict")

        def _device_to_idx(dev: str) -> int:
            if not isinstance(dev, str) or ":" not in dev:
                raise ValueError(f"invalid device string: {dev}")
            try:
                return int(dev.split(":")[-1])
            except Exception as e:
                raise ValueError(f"invalid device string: {dev}") from e

        def _meta_offset_size(meta) -> tuple[int, int]:
            if not isinstance(meta, (list, tuple)) or len(meta) < 2:
                raise ValueError(f"invalid tensor meta: {meta}")
            return int(meta[0]), int(meta[1])

        def _dtype_nbytes(dtype_obj) -> int:
            # Support torch dtype objects and common serialized dtype strings.
            if isinstance(dtype_obj, torch.dtype):
                return int(torch.empty([], dtype=dtype_obj).element_size())
            ds = str(dtype_obj).strip().lower()
            dtype_size_map = {
                "torch.float16": 2,
                "float16": 2,
                "half": 2,
                "torch.bfloat16": 2,
                "bfloat16": 2,
                "bf16": 2,
                "torch.float32": 4,
                "float32": 4,
                "fp32": 4,
                "torch.float64": 8,
                "float64": 8,
                "double": 8,
                "torch.int8": 1,
                "int8": 1,
                "torch.uint8": 1,
                "uint8": 1,
                "torch.int16": 2,
                "int16": 2,
                "short": 2,
                "torch.int32": 4,
                "int32": 4,
                "torch.int64": 8,
                "int64": 8,
                "long": 8,
                "torch.bool": 1,
                "bool": 1,
            }
            if ds in dtype_size_map:
                return int(dtype_size_map[ds])
            raise ValueError(f"unsupported dtype for byte-size inference: {dtype_obj}")

        def _shape_numel(shape_obj) -> int:
            if not isinstance(shape_obj, (list, tuple)):
                raise ValueError(f"invalid shape: {shape_obj}")
            n = 1
            for dim in shape_obj:
                d = int(dim)
                if d <= 0:
                    raise ValueError(f"invalid shape dim: {shape_obj}")
                n *= d
            return int(n)

        def _logical_expert_tensor_name(base_tensor_name: str, expert_id: int) -> str:
            if ".experts.gate_up_proj" in base_tensor_name:
                return base_tensor_name.replace(
                    ".experts.gate_up_proj",
                    f".experts.{expert_id}.gate_up_proj",
                )
            if ".experts.down_proj" in base_tensor_name:
                return base_tensor_name.replace(
                    ".experts.down_proj",
                    f".experts.{expert_id}.down_proj",
                )
            # Fallback for unexpected fused expert naming.
            return f"{base_tensor_name}.expert_{expert_id}"

        def _resolve_fused_expert_chunk_size(
            tensor_name: str,
            meta: tuple | list,
            total_size: int,
            experts_num: int,
        ) -> int:
            per_expert_size, remainder = divmod(total_size, experts_num)
            if remainder != 0:
                raise RuntimeError(
                    f"tensor bytes not divisible by experts_num for {tensor_name}: "
                    f"total_size={total_size}, experts_num={experts_num}, remainder={remainder}"
                )

            # For gate_up/down fused experts, require shape+dtype validation.
            is_fused_gate_or_down = (
                ".experts.gate_up_proj" in tensor_name
                or ".experts.down_proj" in tensor_name
            )
            shape = meta[2] if len(meta) > 2 else None
            if is_fused_gate_or_down:
                if not (isinstance(shape, (list, tuple)) and len(shape) >= 2 and int(shape[0]) == experts_num):
                    raise RuntimeError(
                        f"invalid fused expert shape for {tensor_name}: "
                        f"shape={shape}, experts_num={experts_num}"
                    )
                if len(meta) <= 4:
                    raise RuntimeError(
                        f"missing dtype for fused expert tensor {tensor_name}, "
                        "cannot validate per-expert chunk size"
                    )
                dtype_bytes = _dtype_nbytes(meta[4])
                expected_per_expert = _shape_numel(shape[1:]) * dtype_bytes
                if expected_per_expert != per_expert_size:
                    raise RuntimeError(
                        f"per-expert chunk size mismatch for {tensor_name}: "
                        f"per_expert_size={per_expert_size}, expected={expected_per_expert}, "
                        f"shape={shape}, dtype={meta[4]}"
                    )
                return int(expected_per_expert)

            # Non gate_up/down fused variants: keep divisible bytes path.
            return int(per_expert_size)

        # Per-device linear allocator for dst_offset.
        device_next_offset: dict[int, int] = {}
        for dev_name, bytes_used in device_memory_used.items():
            dev_idx = _device_to_idx(str(dev_name))
            _ = int(bytes_used)  # keep type check strict, value validated at the end.
            device_next_offset.setdefault(dev_idx, 0)

        chunk_items: list[dict] = []
        # Map each physical copy slice to all logical tensor names that reference it.
        copy_key_to_tensor_names: dict[tuple[int, int, int, int, int], list[str]] = {}
        # Used for de-duplicating exact same host slice copies.
        seen_src_slice: dict[tuple[int, int, int], int] = {}
        get_tensor_expert_group_key = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        get_tensor_layer_idx = getattr(self.mlpm, "get_tensor_layer_idx", None)
        if not callable(get_tensor_expert_group_key):
            raise RuntimeError("mlpm must provide get_tensor_expert_group_key")
        if not callable(get_tensor_layer_idx):
            raise RuntimeError("mlpm must provide get_tensor_layer_idx")

        def _tensor_order_key(item: tuple[str, object]):
            tname = str(item[0])
            layer_idx = get_tensor_layer_idx(tname)
            layer_rank = int(layer_idx) if layer_idx is not None else 10**9
            expert_key = get_tensor_expert_group_key(tname)
            # Experts first (do not change chunk granularity, only copy placement order).
            # Keep gate_up before down within same layer/device to maximize contiguous placement.
            if expert_key is not None and ".experts.gate_up_proj" in tname:
                return (0, layer_rank, 0, tname)
            if expert_key is not None and ".experts.down_proj" in tname:
                return (0, layer_rank, 1, tname)
            if expert_key is not None:
                return (0, layer_rank, 2, tname)
            return (1, layer_rank, 2, tname)

        ordered_items = sorted(tensor_to_device.items(), key=_tensor_order_key)
        for tensor_name, locate in ordered_items:
            if tensor_name not in tensor_index_json:
                continue
            src_offset, total_size = _meta_offset_size(tensor_index_json[tensor_name])
            if total_size <= 0:
                continue

            # Non-fused tensor -> one chunk.
            if isinstance(locate, str):
                dev_idx = _device_to_idx(locate)
                key = (dev_idx, src_offset, total_size)
                if key in seen_src_slice:
                    dst_offset = seen_src_slice[key]
                else:
                    dst_offset = device_next_offset.get(dev_idx, 0)
                    device_next_offset[dev_idx] = dst_offset + total_size
                    seen_src_slice[key] = dst_offset
                chunk_items.append(
                    {
                        "tensor_name": tensor_name,
                        "device_idx": dev_idx,
                        "src_offset": src_offset,
                        "size": total_size,
                        "dst_offset": dst_offset,
                        "handle_idx": 0,
                    }
                )
                copy_key = (dev_idx, src_offset, total_size, dst_offset, 0)
                copy_key_to_tensor_names.setdefault(copy_key, []).append(str(tensor_name))
                continue

            # Fused expert tensor -> one chunk per expert (or contiguous expert slices by mapping).
            if isinstance(locate, dict):
                e2d = locate.get("expert_id_to_device")
                if isinstance(e2d, dict) and e2d:
                    normalized_e2d: dict[int, str] = {}
                    for raw_eid, raw_dev in e2d.items():
                        normalized_e2d[int(raw_eid)] = str(raw_dev)
                    sorted_eids = sorted(normalized_e2d.keys())
                    if not sorted_eids:
                        continue

                    # Prefer shape[0] as expert count for fused expert banks.
                    meta = tensor_index_json[tensor_name]
                    shape = meta[2] if len(meta) > 2 else None
                    experts_num = int(shape[0]) if isinstance(shape, (list, tuple)) and len(shape) > 0 else len(sorted_eids)
                    if experts_num <= 0:
                        raise RuntimeError(f"invalid experts_num for {tensor_name}: {experts_num}")
                    chunk_size = _resolve_fused_expert_chunk_size(
                        tensor_name=tensor_name,
                        meta=meta,
                        total_size=total_size,
                        experts_num=experts_num,
                    )
                    for eid in sorted_eids:
                        if eid < 0 or eid >= experts_num:
                            raise RuntimeError(
                                f"invalid expert_id for {tensor_name}: eid={eid}, experts_num={experts_num}"
                            )
                        dev_idx = _device_to_idx(normalized_e2d[eid])
                        chunk_src_offset = src_offset + eid * chunk_size
                        key = (dev_idx, chunk_src_offset, chunk_size)
                        if key in seen_src_slice:
                            dst_offset = seen_src_slice[key]
                        else:
                            dst_offset = device_next_offset.get(dev_idx, 0)
                            device_next_offset[dev_idx] = dst_offset + chunk_size
                            seen_src_slice[key] = dst_offset
                        logical_tensor_name = _logical_expert_tensor_name(
                            base_tensor_name=tensor_name,
                            expert_id=eid,
                        )
                        chunk_items.append(
                            {
                                "tensor_name": logical_tensor_name,
                                "device_idx": dev_idx,
                                "src_offset": chunk_src_offset,
                                "size": chunk_size,
                                "dst_offset": dst_offset,
                                "handle_idx": 0,
                            }
                        )
                        copy_key = (dev_idx, chunk_src_offset, chunk_size, dst_offset, 0)
                        copy_key_to_tensor_names.setdefault(copy_key, []).append(str(logical_tensor_name))
                    continue

                default_device = locate.get("default_device")
                if isinstance(default_device, str):
                    dev_idx = _device_to_idx(default_device)
                    key = (dev_idx, src_offset, total_size)
                    if key in seen_src_slice:
                        dst_offset = seen_src_slice[key]
                    else:
                        dst_offset = device_next_offset.get(dev_idx, 0)
                        device_next_offset[dev_idx] = dst_offset + total_size
                        seen_src_slice[key] = dst_offset
                    chunk_items.append(
                        {
                            "tensor_name": tensor_name,
                            "device_idx": dev_idx,
                            "src_offset": src_offset,
                            "size": total_size,
                            "dst_offset": dst_offset,
                            "handle_idx": 0,
                        }
                    )
                    copy_key = (dev_idx, src_offset, total_size, dst_offset, 0)
                    copy_key_to_tensor_names.setdefault(copy_key, []).append(str(tensor_name))
                    continue

            raise TypeError(
                f"invalid tensor_to_device value for {tensor_name}: expected str/dict, got {type(locate)}"
            )

        # Keep one copy task for each physical slice; preserve all logical tensor-name bindings.
        dedup_copy_items: list[dict] = []
        for copy_key, tensor_names in copy_key_to_tensor_names.items():
            dev_idx, src_offset, size, dst_offset, handle_idx = copy_key
            dedup_copy_items.append(
                {
                    "tensor_name": sorted(set(tensor_names))[0],
                    "device_idx": int(dev_idx),
                    "src_offset": int(src_offset),
                    "size": int(size),
                    "dst_offset": int(dst_offset),
                    "handle_idx": int(handle_idx),
                }
            )
        dedup_copy_items = self.generate_chunk_task_id(dedup_copy_items)

        # Build output format consumed by sllm_store client:
        # {device_idx: [(src_offset, size, dst_offset, handle_idx, task_id, priority, reorder_hint), ...]}
        tensor_copy_chunks: dict[int, list[tuple[int, int, int, int, int, int, bool]]] = {}
        tensor_task_queue_index: dict[str, list[dict[str, int]]] = {}
        copy_task_info_by_key: dict[tuple[int, int, int, int, int], tuple[int, int, int]] = {}
        for item in dedup_copy_items:
            dev_idx = int(item["device_idx"])
            chunk_list = tensor_copy_chunks.setdefault(dev_idx, [])
            queue_pos = len(chunk_list)
            chunk_list.append(
                (
                    int(item["src_offset"]),
                    int(item["size"]),
                    int(item["dst_offset"]),
                    int(item["handle_idx"]),
                    int(item["task_id"]),
                    0,      # COPY_PRIORITY_LOW
                    False,  # reorder_hint
                )
            )
            copy_key = (
                int(item["device_idx"]),
                int(item["src_offset"]),
                int(item["size"]),
                int(item["dst_offset"]),
                int(item["handle_idx"]),
            )
            copy_task_info_by_key[copy_key] = (int(item["task_id"]), dev_idx, queue_pos)

        for copy_key, tensor_names in copy_key_to_tensor_names.items():
            task_info = copy_task_info_by_key.get(copy_key)
            if task_info is None:
                continue
            task_id, dev_idx, queue_pos = task_info
            for tname in sorted(set(tensor_names)):
                tensor_task_queue_index.setdefault(tname, []).append(
                    {
                        "task_id": task_id,
                        "device_idx": dev_idx,
                        "queue_pos": queue_pos,
                    }
                )

        # Keep simple name->offset map for tensors that map to a single dst slice.
        tensor_device_offsets: dict[int, dict[str, int]] = {}
        single_tensor_seen: dict[tuple[int, str], int] = {}
        single_tensor_multi: set[tuple[int, str]] = set()
        for item in chunk_items:
            key = (int(item["device_idx"]), str(item["tensor_name"]))
            if key in single_tensor_seen and single_tensor_seen[key] != int(item["dst_offset"]):
                single_tensor_multi.add(key)
            single_tensor_seen[key] = int(item["dst_offset"])
        for (dev_idx, tname), dst in single_tensor_seen.items():
            if (dev_idx, tname) in single_tensor_multi:
                continue
            tensor_device_offsets.setdefault(dev_idx, {})[tname] = dst

        # Validate per-device planned size does not exceed expected usage.
        for dev_name, expected_bytes in device_memory_used.items():
            dev_idx = _device_to_idx(str(dev_name))
            used = int(device_next_offset.get(dev_idx, 0))
            if used > int(expected_bytes):
                raise RuntimeError(
                    f"generated chunk bytes exceed expected device memory on {dev_name}: "
                    f"used={used}, expected={int(expected_bytes)}"
                )

        return tensor_device_offsets, tensor_copy_chunks, dict(device_next_offset), tensor_task_queue_index

    def restore_experts_state_dict(self, tensor_index_json: dict, cuda_memory_ptrs: dict, tensor_device_offsets: dict):
        """
        1) 返回 每个 experts 的 gate_up_proj 和 down_proj 的权重
        2) 由于 每层每设备的专家是连续放置的, 同时返回每个设备，每层layer的聚合 gate_up_proj 和 down_proj 的权重
        3) 可参照  allocate_cuda_memory_fused_experts_dual_restore 的实现
        """
        if not isinstance(tensor_index_json, dict):
            raise TypeError("tensor_index_json must be a dict")
        if not isinstance(cuda_memory_ptrs, dict):
            raise TypeError("cuda_memory_ptrs must be a dict")
        if not isinstance(tensor_device_offsets, dict):
            raise TypeError("tensor_device_offsets must be a dict")

        get_tensor_expert_group_key = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        get_fused_expert_tensor_for_restore = getattr(self.mlpm, "get_fused_expert_tensor_for_restore", None)
        if not callable(get_tensor_expert_group_key):
            raise RuntimeError("mlpm must provide get_tensor_expert_group_key")
        if not callable(get_fused_expert_tensor_for_restore):
            raise RuntimeError("mlpm must provide get_fused_expert_tensor_for_restore")

        # Per-expert 2D views (name -> tensor)
        state_dict_slices = self.restore_state_dict(
            tensor_index_json=tensor_index_json,
            cuda_memory_ptrs=cuda_memory_ptrs,
            tensor_device_offsets=tensor_device_offsets,
        )

        # Build per-(layer,device) packed tensors from contiguous expert slices.
        grouped: dict[tuple[int, int], dict[str, dict[int, str] | str]] = {}
        for dev_idx, name_to_offset in tensor_device_offsets.items():
            d = int(dev_idx)
            if not isinstance(name_to_offset, dict):
                continue
            for name in name_to_offset.keys():
                tname = str(name)
                gk = get_tensor_expert_group_key(tname)
                if gk is None or int(gk[1]) < 0:
                    continue
                layer_id, expert_id = int(gk[0]), int(gk[1])
                split_info = get_fused_expert_tensor_for_restore(tname)
                fused_name = split_info[0] if split_info is not None else ""
                key = (layer_id, d)
                ent = grouped.setdefault(
                    key,
                    {
                        "gate_up": {},
                        "down": {},
                        "gate_up_fused_name": "",
                        "down_fused_name": "",
                    },
                )
                if "gate_up_proj" in tname:
                    gate_map = ent["gate_up"]
                    gate_map[int(expert_id)] = tname
                    if fused_name:
                        ent["gate_up_fused_name"] = fused_name
                elif "down_proj" in tname:
                    down_map = ent["down"]
                    down_map[int(expert_id)] = tname
                    if fused_name:
                        ent["down_fused_name"] = fused_name

        packed_by_layer_device: dict[int, dict[int, dict[str, object]]] = {}
        tensor_meta_index_packed: dict[str, tuple[object, object, object]] = {}
        tensor_device_offsets_packed: dict[int, dict[str, int]] = {}
        for (layer_id, d), ent in grouped.items():
            gate_map: dict[int, str] = ent["gate_up"]
            down_map: dict[int, str] = ent["down"]
            common_eids = sorted(set(gate_map.keys()) & set(down_map.keys()))
            if not common_eids:
                continue

            gate_names = [gate_map[eid] for eid in common_eids]
            down_names = [down_map[eid] for eid in common_eids]
            if not all(n in state_dict_slices for n in gate_names + down_names):
                continue

            gate_fused = str(ent.get("gate_up_fused_name", "")).strip() or f"layer_{layer_id}.gate_up_proj"
            down_fused = str(ent.get("down_fused_name", "")).strip() or f"layer_{layer_id}.down_proj"
            if gate_fused not in tensor_index_json or down_fused not in tensor_index_json:
                continue
            gate_meta = tensor_index_json[gate_fused]
            down_meta = tensor_index_json[down_fused]
            if not (isinstance(gate_meta, (list, tuple)) and len(gate_meta) >= 5):
                continue
            if not (isinstance(down_meta, (list, tuple)) and len(down_meta) >= 5):
                continue
            _goff, gate_total_size, gate_shape, _gstride, gate_dtype = gate_meta[:5]
            _doff, down_total_size, down_shape, _dstride, down_dtype = down_meta[:5]
            if not (isinstance(gate_shape, (list, tuple)) and len(gate_shape) >= 3):
                continue
            if not (isinstance(down_shape, (list, tuple)) and len(down_shape) >= 3):
                continue
            total_experts = int(gate_shape[0])
            if total_experts <= 0 or int(down_shape[0]) != total_experts:
                continue
            gate_row_bytes = int(gate_total_size) // total_experts
            down_row_bytes = int(down_total_size) // total_experts

            gate_offsets = [int(tensor_device_offsets[int(d)][nm]) for nm in gate_names]
            down_offsets = [int(tensor_device_offsets[int(d)][nm]) for nm in down_names]
            if any(gate_offsets[i] != gate_offsets[0] + i * gate_row_bytes for i in range(len(gate_offsets))):
                continue
            if any(down_offsets[i] != down_offsets[0] + i * down_row_bytes for i in range(len(down_offsets))):
                continue

            gate_up_packed_name = f"{gate_fused}.packed.dev_{d}"
            down_packed_name = f"{down_fused}.packed.dev_{d}"
            e_dev = len(common_eids)
            gate_up_packed_shape = (e_dev, int(gate_shape[1]), int(gate_shape[2]))
            down_packed_shape = (e_dev, int(down_shape[1]), int(down_shape[2]))
            gate_up_packed_stride = (
                gate_up_packed_shape[1] * gate_up_packed_shape[2],
                gate_up_packed_shape[2],
                1,
            )
            down_packed_stride = (
                down_packed_shape[1] * down_packed_shape[2],
                down_packed_shape[2],
                1,
            )
            tensor_meta_index_packed[gate_up_packed_name] = (
                gate_up_packed_shape,
                gate_up_packed_stride,
                gate_dtype,
            )
            tensor_meta_index_packed[down_packed_name] = (
                down_packed_shape,
                down_packed_stride,
                down_dtype,
            )
            tensor_device_offsets_packed.setdefault(int(d), {})[gate_up_packed_name] = gate_offsets[0]
            tensor_device_offsets_packed.setdefault(int(d), {})[down_packed_name] = down_offsets[0]

            packed_by_layer_device.setdefault(layer_id, {})[d] = {
                "experts": list(common_eids),
                "gate_up_packed_name": gate_up_packed_name,
                "down_packed_name": down_packed_name,
            }

        state_dict_packed: dict[str, torch.Tensor] = {}
        if tensor_meta_index_packed:
            state_dict_packed = restore_tensors2(
                tensor_meta_index_packed, cuda_memory_ptrs, tensor_device_offsets_packed
            )
            for layer_id, dev_map in packed_by_layer_device.items():
                for d, ent in dev_map.items():
                    gname = ent["gate_up_packed_name"]
                    dname = ent["down_packed_name"]
                    if gname in state_dict_packed:
                        ent["gate_up_packed"] = state_dict_packed[gname]
                    if dname in state_dict_packed:
                        ent["down_packed"] = state_dict_packed[dname]

        return {
            "state_dict_slices": state_dict_slices,
            "state_dict_packed": state_dict_packed,
            "packed_by_layer_device": packed_by_layer_device,
        }

    def restore_state_dict(self, tensor_index_json: dict, cuda_memory_ptrs: dict, tensor_device_offsets: dict):
        """
        作用：根据 `tensor_device_offsets` 生成 `restore_tensors2` 所需的
        `tensor_meta_index`，并恢复 GPU state dict 视图。

        规则：
        1) 若名称直接存在于 `tensor_index_json`，直接使用其 `(shape, stride, dtype)`。
        2) 若名称是拆分后的专家名（如 `...experts.{eid}.gate_up_proj/down_proj`），
           则回推到 fused tensor（`...experts.gate_up_proj/down_proj`）并构造单专家视图 meta。
        3) 兼容 fallback 命名（`...experts.gate_up_proj.expert_{eid}` /
           `...experts.down_proj.expert_{eid}`）。
        4) 专家视图恢复会校验 experts 维度、expert_id 边界以及单专家字节数一致性，
           保证恢复视图与 `generate_chunk` 的拆分规则一致。
        5) 专家名称解析优先使用 `mlpm` 接口（如
           `get_fused_expert_tensor_for_restore` / `get_tensor_expert_group_key`），
           便于不同模型命名在 `mlpm` 层统一扩展。
        """
        if not isinstance(tensor_index_json, dict):
            raise TypeError("tensor_index_json must be a dict")
        if not isinstance(cuda_memory_ptrs, dict):
            raise TypeError("cuda_memory_ptrs must be a dict")
        if not isinstance(tensor_device_offsets, dict):
            raise TypeError("tensor_device_offsets must be a dict")
        get_fused_expert_tensor_for_restore = getattr(
            self.mlpm, "get_fused_expert_tensor_for_restore", None
        )
        get_tensor_expert_group_key = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        if not callable(get_fused_expert_tensor_for_restore):
            raise RuntimeError("mlpm must provide get_fused_expert_tensor_for_restore")
        if not callable(get_tensor_expert_group_key):
            raise RuntimeError("mlpm must provide get_tensor_expert_group_key")

        def _dtype_nbytes(dtype_obj) -> int:
            if isinstance(dtype_obj, torch.dtype):
                return int(torch.empty([], dtype=dtype_obj).element_size())
            ds = str(dtype_obj).strip().lower()
            dtype_size_map = {
                "torch.float16": 2,
                "float16": 2,
                "half": 2,
                "torch.bfloat16": 2,
                "bfloat16": 2,
                "bf16": 2,
                "torch.float32": 4,
                "float32": 4,
                "fp32": 4,
                "torch.float64": 8,
                "float64": 8,
                "double": 8,
                "torch.int8": 1,
                "int8": 1,
                "torch.uint8": 1,
                "uint8": 1,
                "torch.int16": 2,
                "int16": 2,
                "short": 2,
                "torch.int32": 4,
                "int32": 4,
                "torch.int64": 8,
                "int64": 8,
                "long": 8,
                "torch.bool": 1,
                "bool": 1,
            }
            if ds in dtype_size_map:
                return int(dtype_size_map[ds])
            raise ValueError(f"unsupported dtype for byte-size inference: {dtype_obj}")

        def _shape_numel(shape_obj) -> int:
            if not isinstance(shape_obj, (list, tuple)):
                raise ValueError(f"invalid shape: {shape_obj}")
            n = 1
            for dim in shape_obj:
                d = int(dim)
                if d <= 0:
                    raise ValueError(f"invalid shape dim: {shape_obj}")
                n *= d
            return int(n)

        # Collect only tensors that are actually mapped to some device offset.
        required_names: set[str] = set()
        for _device_idx, name_to_offset in tensor_device_offsets.items():
            if not isinstance(name_to_offset, dict):
                raise TypeError("tensor_device_offsets values must be dict[str, int]")
            for name, dst in name_to_offset.items():
                _ = int(dst)
                required_names.add(str(name))

        tensor_meta_index: dict[str, tuple[object, object, object]] = {}
        for name in required_names:
            if name in tensor_index_json:
                meta = tensor_index_json[name]
                if not isinstance(meta, (list, tuple)) or len(meta) < 5:
                    raise ValueError(
                        f"invalid tensor meta for {name}, expected [offset,size,shape,stride,dtype], got: {meta}"
                    )
                # restore_tensors2 consumes (shape, stride, dtype)
                _offset, _size, shape, stride, dtype = meta[:5]
                tensor_meta_index[name] = (shape, stride, dtype)
                continue

            split_info = get_fused_expert_tensor_for_restore(name)
            if split_info is None:
                raise KeyError(f"tensor not found in tensor_index_json: {name}")
            fused_name, expert_id = split_info
            group_key = get_tensor_expert_group_key(fused_name)
            if group_key is None or int(group_key[1]) != -1:
                raise ValueError(
                    f"resolved fused tensor is not a fused expert bank: "
                    f"name={name}, fused_name={fused_name}, group_key={group_key}"
                )
            if fused_name not in tensor_index_json:
                raise KeyError(f"fused tensor not found for split expert tensor {name}: {fused_name}")
            fused_meta = tensor_index_json[fused_name]
            if not isinstance(fused_meta, (list, tuple)) or len(fused_meta) < 5:
                raise ValueError(
                    f"invalid tensor meta for {fused_name}, expected [offset,size,shape,stride,dtype], got: {fused_meta}"
                )
            _offset, total_size, shape, stride, dtype = fused_meta[:5]
            if not isinstance(shape, (list, tuple)) or len(shape) < 2:
                raise ValueError(f"invalid fused expert shape for {fused_name}: {shape}")
            experts_num = int(shape[0])
            if experts_num <= 0:
                raise ValueError(f"invalid experts_num from fused shape for {fused_name}: {shape}")
            if expert_id < 0 or expert_id >= experts_num:
                raise ValueError(
                    f"expert_id out of range for {name}: expert_id={expert_id}, experts_num={experts_num}"
                )
            total_size_int = int(total_size)
            per_expert_size, remainder = divmod(total_size_int, experts_num)
            if remainder != 0 or per_expert_size <= 0:
                raise ValueError(
                    f"fused expert bytes not divisible for {fused_name}: "
                    f"total_size={total_size_int}, experts_num={experts_num}, remainder={remainder}"
                )
            dtype_bytes = _dtype_nbytes(dtype)
            expected_per_expert = _shape_numel(shape[1:]) * dtype_bytes
            if expected_per_expert != per_expert_size:
                raise ValueError(
                    f"per-expert bytes mismatch for {fused_name}: "
                    f"per_expert_size={per_expert_size}, expected={expected_per_expert}, "
                    f"shape={shape}, dtype={dtype}"
                )
            slice_shape = tuple(shape[1:])
            if len(slice_shape) == 0:
                raise ValueError(f"invalid slice_shape for {fused_name}: {shape}")
            if len(slice_shape) == 2:
                # Keep contiguous 2D slice view as done in cuda_memory_view.
                slice_stride = (int(slice_shape[1]), 1)
            elif isinstance(stride, (list, tuple)) and len(stride) >= len(shape):
                slice_stride = tuple(int(x) for x in stride[1:])
            else:
                raise ValueError(f"invalid fused stride for {fused_name}: stride={stride}, shape={shape}")
            tensor_meta_index[name] = (slice_shape, slice_stride, dtype)

        state_dict = restore_tensors2(
            tensor_meta_index, cuda_memory_ptrs, tensor_device_offsets
        )
        return state_dict
    
    def get_generate_name_task_ids(self, tensor_task_queue_index: dict[str, list[dict[str, int]]], layer_idx: int):
        """
        1) 根据 `tensor_task_queue_index` 获取 `layer_idx` 层中 generate 对应的 task_id
        2) 使用mlpm接口获取 layer_idx 需要的 tensor_name
        """
        if not isinstance(tensor_task_queue_index, dict):
            raise TypeError("tensor_task_queue_index must be a dict")
        layer_idx = int(layer_idx)
        get_tensor_index_general_names = getattr(self.mlpm, "get_tensor_index_general_names", None)
        if not callable(get_tensor_index_general_names):
            raise RuntimeError("mlpm must provide get_tensor_index_general_names")

        # General/generate tensors are global parameters (embedding/lm_head/final norm),
        # and are ordered before layer-scoped tensors in task_id generation.
        # To avoid duplicate dispatch in per-layer scheduling, only bind them to layer 0.
        if layer_idx != 0:
            return []

        general_name_set = set(str(n) for n in get_tensor_index_general_names())
        seen_task_ids: set[int] = set()
        generate_task_ids: list[int] = []
        for tensor_name, entries in tensor_task_queue_index.items():
            if str(tensor_name) not in general_name_set:
                continue
            if not isinstance(entries, list):
                continue
            for rec in entries:
                if not isinstance(rec, dict):
                    continue
                task_id = int(rec.get("task_id", 0))
                if task_id <= 0 or task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
                generate_task_ids.append(task_id)
        return generate_task_ids

    def get_self_attn_gate_layernorm_tasks_ids(self, tensor_task_queue_index: dict[str, list[dict[str, int]]], layer_idx: int):
        """
        1) 根据 `tensor_task_queue_index` 获取 `layer_idx` 层中 self_attn_gate_layernorm 对应的 task_id
        2) 使用mlpm接口获取 layer_idx 需要的 tensor_name
        """
        if not isinstance(tensor_task_queue_index, dict):
            raise TypeError("tensor_task_queue_index must be a dict")
        layer_idx = int(layer_idx)
        get_tensor_index_layer_names = getattr(self.mlpm, "get_tensor_index_layer_names", None)
        get_tensor_expert_group_key = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        if not callable(get_tensor_index_layer_names):
            raise RuntimeError("mlpm must provide get_tensor_index_layer_names")
        if not callable(get_tensor_expert_group_key):
            raise RuntimeError("mlpm must provide get_tensor_expert_group_key")

        # mlpm 返回该层需同层同卡处理的非专家参数名（self_attn/gate/layernorm 等）。
        layer_name_set = set(str(n) for n in get_tensor_index_layer_names(layer_idx))
        seen_task_ids: set[int] = set()
        self_attn_gate_layernorm_task_ids: list[int] = []
        for tensor_name, entries in tensor_task_queue_index.items():
            tname = str(tensor_name)
            if tname not in layer_name_set:
                continue
            # 额外过滤专家名，避免 split expert 误入。
            if get_tensor_expert_group_key(tname) is not None:
                continue
            if not isinstance(entries, list):
                continue
            for rec in entries:
                if not isinstance(rec, dict):
                    continue
                task_id = int(rec.get("task_id", 0))
                if task_id <= 0 or task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
                self_attn_gate_layernorm_task_ids.append(task_id)
        return self_attn_gate_layernorm_task_ids

    def _build_expert_task_id_index(
        self,
        tensor_task_queue_index: dict[str, list[dict[str, int]]],
    ) -> dict[tuple[int, int], list[int]]:
        """One-time O(N) scan of tensor_task_queue_index to build an inverted index:
        (layer_id, expert_id) -> deduplicated list[task_id].
        The result is cached on self._expert_task_id_index_cache keyed by id(queue).
        """
        get_tensor_expert_group_key = getattr(self.mlpm, "get_tensor_expert_group_key", None)
        if not callable(get_tensor_expert_group_key):
            raise RuntimeError("mlpm must provide get_tensor_expert_group_key")

        index: dict[tuple[int, int], list[int]] = {}
        for tensor_name, entries in tensor_task_queue_index.items():
            group_key = get_tensor_expert_group_key(str(tensor_name))
            if group_key is None:
                continue
            if not isinstance(entries, list):
                continue
            key = (int(group_key[0]), int(group_key[1]))
            seen: set[int] = set()
            task_ids: list[int] = []
            for rec in entries:
                if not isinstance(rec, dict):
                    continue
                task_id = int(rec.get("task_id", 0))
                if task_id <= 0 or task_id in seen:
                    continue
                seen.add(task_id)
                task_ids.append(task_id)
            if task_ids:
                existing = index.get(key)
                if existing is None:
                    index[key] = task_ids
                else:
                    seen_existing = set(existing)
                    for tid in task_ids:
                        if tid not in seen_existing:
                            existing.append(tid)
                            seen_existing.add(tid)
        return index

    def _build_layer_experts_alloc_index(self, tensor_to_device: dict) -> dict[int, dict]:
        """One-time build of per-layer expert allocation cache: layer_idx -> allocation dict.
        Iterates all layers once so layer_moe_fused can do O(1) lookup instead of scanning
        tensor_to_device on every forward pass.
        """
        num_layers = int(self.mlpm.get_num_hidden_layers())
        cache: dict[int, dict] = {}
        for layer_idx in range(num_layers):
            cache[layer_idx] = self.get_layer_experts_device_allocation(
                layer_idx=layer_idx,
                tensor_to_device=tensor_to_device,
            )
        return cache

    def get_experts_task_ids(self, tensor_task_queue_index: dict[str, list[dict[str, int]]], layer_idx: int, expert_ids: list[int]):
        """
        1) 根据预建的倒排索引 self._expert_task_id_index，O(E) 查询 layer_idx 层中
           expert_ids 对应的 task_id 列表（E = len(expert_ids)）。
        2) 索引在 generate_chunk 后由 _build_expert_task_id_index 一次性构建，无需遍历全表。

        tensor_task_queue_index 参数保留仅用于外部调用的兼容性，不再迭代。
        """
        layer_idx = int(layer_idx)
        if not isinstance(expert_ids, list):
            raise TypeError("expert_ids must be a list[int]")

        if not expert_ids:
            return []

        index = self._expert_task_id_index
        seen_task_ids: set[int] = set()
        result: list[int] = []
        for eid in expert_ids:
            task_ids = index.get((layer_idx, int(eid)))
            if task_ids is None:
                continue
            for tid in task_ids:
                if tid not in seen_task_ids:
                    seen_task_ids.add(tid)
                    result.append(tid)
        return result

    def get_copy_chunks_total_size_by_task_ids(self, tensor_copy_chunks, task_ids: list[int]) -> int:
        """
        根据 task_ids 统计对应 copy_chunks 的总 size（bytes）。
        - 支持 dict[device_idx -> list[tuple]] 结构，tuple 至少包含 (src, size, dst, handle, task_id, ...)
        - 支持 list[dict] 结构，dict 需包含 `size` 与 `task_id`
        - 同一 task_id 只累计一次
        """
        if not isinstance(task_ids, list):
            raise TypeError("task_ids must be a list[int]")
        target_task_ids = set(int(tid) for tid in task_ids if int(tid) > 0)
        if not target_task_ids:
            return 0

        total_size = 0
        seen_task_ids: set[int] = set()

        if isinstance(tensor_copy_chunks, dict):
            for _dev, chunks in tensor_copy_chunks.items():
                if not isinstance(chunks, list):
                    continue
                for chunk in chunks:
                    if not isinstance(chunk, (tuple, list)) or len(chunk) < 5:
                        continue
                    task_id = int(chunk[4])
                    if task_id not in target_task_ids or task_id in seen_task_ids:
                        continue
                    seen_task_ids.add(task_id)
                    total_size += int(chunk[1])
            return int(total_size)

        if isinstance(tensor_copy_chunks, list):
            for chunk in tensor_copy_chunks:
                if not isinstance(chunk, dict):
                    continue
                task_id = int(chunk.get("task_id", 0))
                if task_id not in target_task_ids or task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
                total_size += int(chunk.get("size", 0))
            return int(total_size)

        raise TypeError("tensor_copy_chunks must be list[dict] or dict[device, list[tuple]]")

    def get_copy_chunks_size_by_task_ids_per_device(self, tensor_copy_chunks, task_ids: list[int]) -> dict[int, int]:
        """
        根据 task_ids 统计各设备对应 copy_chunks 的总 size（bytes）。
        返回: {device_idx: total_bytes}
        """
        if not isinstance(task_ids, list):
            raise TypeError("task_ids must be a list[int]")
        target_task_ids = set(int(tid) for tid in task_ids if int(tid) > 0)
        if not target_task_ids:
            return {}

        size_by_device: dict[int, int] = {}
        seen_task_ids: set[int] = set()

        if isinstance(tensor_copy_chunks, dict):
            for dev, chunks in tensor_copy_chunks.items():
                if not isinstance(chunks, list):
                    continue
                dev_idx = int(dev)
                for chunk in chunks:
                    if not isinstance(chunk, (tuple, list)) or len(chunk) < 5:
                        continue
                    task_id = int(chunk[4])
                    if task_id not in target_task_ids or task_id in seen_task_ids:
                        continue
                    seen_task_ids.add(task_id)
                    size_by_device[dev_idx] = size_by_device.get(dev_idx, 0) + int(chunk[1])
            return size_by_device

        if isinstance(tensor_copy_chunks, list):
            for chunk in tensor_copy_chunks:
                if not isinstance(chunk, dict):
                    continue
                task_id = int(chunk.get("task_id", 0))
                if task_id not in target_task_ids or task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
                dev_idx = int(chunk.get("device_idx", 0))
                size_by_device[dev_idx] = size_by_device.get(dev_idx, 0) + int(chunk.get("size", 0))
            return size_by_device

        raise TypeError("tensor_copy_chunks must be list[dict] or dict[device, list[tuple]]")

    def get_cuda_memory_handles_by_copy_chunks(self, all_cuda_memory_handles: dict, tensor_copy_chunks) -> dict:
        """
        根据 tensor_copy_chunks 中出现的设备，筛选对应 cuda_memory_handles。
        - all_cuda_memory_handles: {device_idx: handle_bytes}
        - tensor_copy_chunks 支持 dict[device_idx -> list[tuple]] 或 list[dict]
        """
        if not isinstance(all_cuda_memory_handles, dict):
            raise TypeError("all_cuda_memory_handles must be a dict")

        device_ids: set[int] = set()
        if isinstance(tensor_copy_chunks, dict):
            for dev in tensor_copy_chunks.keys():
                device_ids.add(int(dev))
        elif isinstance(tensor_copy_chunks, list):
            for item in tensor_copy_chunks:
                if not isinstance(item, dict):
                    continue
                if "device_idx" in item:
                    device_ids.add(int(item["device_idx"]))
        else:
            raise TypeError("tensor_copy_chunks must be list[dict] or dict[device, list[tuple]]")

        selected = {}
        for dev_idx in sorted(device_ids):
            if dev_idx in all_cuda_memory_handles:
                selected[dev_idx] = all_cuda_memory_handles[dev_idx]
        return selected


    def test_gpuloader(self):
        mlpllm = self
        tensor_index_json = mlpllm.read_tensor_index_json(model_path=mlpllm.mlpm.model_abs_path)
        tensor_to_device, device_expected_used_bytes, expert_id_device_map = mlpllm.predo_tensor_index_locate(tensor_index_json=tensor_index_json)
        # print(tensor_to_device)
        gate_up_down_locate_info = mlpllm.extract_gate_up_down_locate_info(layer_idx=0, tensor_to_device=tensor_to_device)
        # print(gate_up_down_locate_info)
        attn_locate_info = mlpllm.extract_attn_locate_info(layer_idx=0, tensor_to_device=tensor_to_device)
        # print(attn_locate_info)
        generate_locate_info = mlpllm.extract_generate_locate_info(tensor_to_device=tensor_to_device)
        # print(generate_locate_info)

        device_memory_used = mlpllm.calculate_device_memory_from_tensor_to_device(tensor_to_device=tensor_to_device, tensor_index_json=tensor_index_json)
        print(device_memory_used)
        print(device_expected_used_bytes)

        device_memory_used_int = {
            int(str(dev).split(":")[-1]): int(bytes_used)
            for dev, bytes_used in device_memory_used.items()
        }

        cuda_memory_ptrs = allocate_cuda_memory(device_memory_used_int)
        all_cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)

        device_uuid_map = get_device_uuid_map()

        tensor_device_offsets, tensor_copy_chunks, device_next_offset, tensor_task_queue_index = mlpllm.generate_chunk(
            tensor_to_device=tensor_to_device,
            tensor_index_json=tensor_index_json,
            device_memory_used=device_memory_used,
        )
        # print("tensor_device_offsets", tensor_copy_chunks)

        extracted_general_tensor_device_offsets, extracted_general_tensor_copy_chunks, modified_tensor_device_offsets, modified_tensor_copy_chunks = mlpllm.general_chunk_from_chunks(
            tensor_device_offsets=tensor_device_offsets,
            tensor_copy_chunks=tensor_copy_chunks,
            tensor_task_queue_index=tensor_task_queue_index,
        )
        extracted_tensor_device_offsets, extracted_tensor_copy_chunks, modified_tensor_device_offsets, modified_tensor_copy_chunks = mlpllm.self_attn_gate_layernorm_tasks_ids_chunk_from_chunks(
            layer_idx=0,
            tensor_device_offsets=modified_tensor_device_offsets,
            tensor_copy_chunks=modified_tensor_copy_chunks,
            tensor_task_queue_index=tensor_task_queue_index,
        )
        extracted_tensor_device_offsets = self._merge_tensor_device_offsets(
            extracted_general_tensor_device_offsets,
            extracted_tensor_device_offsets,
        )
        extracted_tensor_copy_chunks = self._merge_tensor_copy_chunks(
            extracted_general_tensor_copy_chunks,
            extracted_tensor_copy_chunks,
        )
        
        print("extracted_tensor_device_offsets", extracted_tensor_device_offsets)
        print("extracted_tensor_copy_chunks", extracted_tensor_copy_chunks)

        general_cuda_memory_handles = self.get_cuda_memory_handles_by_copy_chunks(
            all_cuda_memory_handles=all_cuda_memory_handles,
            tensor_copy_chunks=extracted_tensor_copy_chunks,
        )
        print("general_cuda_memory_handles", general_cuda_memory_handles)
    
        cuda_memory_handles = all_cuda_memory_handles

        # return
        time_start_load_general = time.time()
        _, replica_uuid1 = load_into_gpu_async(
            client=self.client,
            device_uuid_map=device_uuid_map,
            model_path=self.mlpm.model_path,
            tensor_copy_chunks=extracted_tensor_copy_chunks,
            cuda_memory_handles=general_cuda_memory_handles,
        )

        general_state_dict = self.restore_state_dict(
            tensor_index_json=tensor_index_json,
            cuda_memory_ptrs=cuda_memory_ptrs,
            tensor_device_offsets=extracted_tensor_device_offsets,
        )
        self.cmv.restore2model_strict(general_state_dict, self.cmv.mlpm_ci)
        self.client.confirm_model_loaded(self.mlpm.model_path, replica_uuid1)
        time_end_load_general = time.time()
        print("load_into_gpu_async time", time_end_load_general - time_start_load_general, "seconds")

        tensor_device_offsets = modified_tensor_device_offsets
        tensor_copy_chunks = modified_tensor_copy_chunks
        
        # print("tensor_device_offsets", tensor_device_offsets)
        # print("tensor_copy_chunks", tensor_copy_chunks)
        # print("device_next_offset", device_next_offset)
        # print("tensor_task_queue_index", tensor_task_queue_index)

        time_start_load = time.time()
        _, replica_uuid1 = load_into_gpu_async(
            client=self.client,
            device_uuid_map=device_uuid_map,
            model_path=self.mlpm.model_path,
            tensor_copy_chunks=tensor_copy_chunks,
            cuda_memory_handles=cuda_memory_handles,
        )

        alloc = self.get_layer_experts_device_allocation(layer_idx=0, tensor_to_device=tensor_to_device)
        print(alloc)
        # 测试：将 layer0 中 [0, 63] 的专家任务提优，并等待其完成。
        layer0_expert_task_ids = self.get_experts_task_ids(
            tensor_task_queue_index=tensor_task_queue_index,
            layer_idx=0,
            expert_ids=list(range(64)),
        )
        bytes_per_device = self.get_copy_chunks_size_by_task_ids_per_device(
            tensor_copy_chunks=tensor_copy_chunks,
            task_ids=layer0_self_attn_gate_layernorm_task_ids,
        )
        total_bytes = sum(bytes_per_device.values())
        print("bytes_per_device", bytes_per_device)
        print("total_bytes", total_bytes)

        layer0_self_attn_gate_layernorm_task_ids = self.get_self_attn_gate_layernorm_tasks_ids(
            tensor_task_queue_index=tensor_task_queue_index,
            layer_idx=1,
        )
        layer2_self_attn_gate_layernorm_task_ids = self.get_self_attn_gate_layernorm_tasks_ids(
            tensor_task_queue_index=tensor_task_queue_index,
            layer_idx=2,
        )
        
        bytes_per_device = self.get_copy_chunks_size_by_task_ids_per_device(
            tensor_copy_chunks=tensor_copy_chunks,
            task_ids=layer0_self_attn_gate_layernorm_task_ids,
        )
        total_bytes = sum(bytes_per_device.values())
        print("bytes_per_device", bytes_per_device)
        print("total_bytes", total_bytes)
        layer0_generate_task_ids = self.get_generate_name_task_ids(
            tensor_task_queue_index=tensor_task_queue_index,
            layer_idx=0,
        )
        
        # time.sleep(0.2)

        layer0_task_ids = layer0_expert_task_ids
        print("layer0_task_ids", layer0_task_ids)
        print("layer0_task_ids_count", len(layer0_task_ids))
        if layer0_task_ids:
            time_start = time.time()
            ok_submit, pending_submit = self.client.submit_high_priority_copy_tasks(
                self.mlpm.model_path,
                replica_uuid1,
                layer0_task_ids,
            )
            print(
                "submit_high_priority_copy_tasks",
                "ok=",
                ok_submit,
                "pending_count=",
                len(pending_submit),
            )
            time_end = time.time()
            print("submit_high_priority_copy_tasks time", time_end - time_start)
            time_start = time.time()
            ok_wait, pending_wait = self.client.wait_copy_tasks(
                self.mlpm.model_path,
                replica_uuid1,
                layer0_task_ids,
                timeout_ms=60000,
            )
            time_end = time.time()
            print("wait_copy_tasks time", time_end - time_start)
            print(
                "wait_copy_tasks(layer0 task_ids)",
                "ok=",
                ok_wait,
                "pending_count=",
                len(pending_wait),
            )

        if layer2_self_attn_gate_layernorm_task_ids:
            time_start = time.time()
            ok_submit, pending_submit = self.client.submit_high_priority_copy_tasks(
                self.mlpm.model_path,
                replica_uuid1,
                layer2_self_attn_gate_layernorm_task_ids,
            )
            print("submit_high_priority_copy_tasks", "ok=", ok_submit, "pending_count=", len(pending_submit))
            time_end = time.time()
            print("submit_high_priority_copy_tasks time", time_end - time_start)
            time_start = time.time()
            ok_wait, pending_wait = self.client.wait_copy_tasks(
                self.mlpm.model_path,
                replica_uuid1,
                layer2_self_attn_gate_layernorm_task_ids,
            )
            time_end = time.time()
            print("wait_copy_tasks time", time_end - time_start)
            print("wait_copy_tasks(layer2 task_ids)", "ok=", ok_wait, "pending_count=", len(pending_wait))
            print(
                "wait_copy_tasks(layer2 task_ids)",
                "ok=",
                ok_wait,
                "pending_count=",
                len(pending_wait),
            )

        self.client.confirm_model_loaded(self.mlpm.model_path, replica_uuid1)
        time_end = time.time()
        print("load_into_gpu_async time", time_end - time_start_load, "seconds")
   

        state_dict = self.restore_state_dict(
            tensor_index_json=tensor_index_json,
            cuda_memory_ptrs=cuda_memory_ptrs,
            tensor_device_offsets=tensor_device_offsets,
        )
        
        print("======= \n\n")
        # print("state_dict", state_dict)



    def test_gate_experts(self):
        device = self.device1
        bsz, seq_len = 32, 256
        layer_idx = 0
        hidden_size = int(getattr(self.mlpm.config, "hidden_size"))
        dtype = getattr(self.mlpm.config, "torch_dtype", torch.bfloat16)
        if dtype is None:
            dtype = torch.bfloat16

        # Ensure embedding/gate related weights are ready before probing activation.
        self.cmv.load_general_and_init()
        self.cmv.init_load_qkvogn_es_weight(layer_idx=layer_idx)

        tokenizer = AutoTokenizer.from_pretrained(self.mlpm.model_abs_path, trust_remote_code=True)
        input_ids = generate_input_ids(tokenizer, bsz, seq_len, device)
        embed_tokens = self.mlpm.get_embed_tokens(self.cmv.mlpm_ci)

        print_layer_parameters(self.cmv.mlpm_ci.model.language_model.layers[layer_idx])
        x = embed_tokens(input_ids).to(dtype=dtype)

        # Use the same path as real prefill before MoE routing:
        # iln -> self_attn -> residual add -> paln, then feed gate.
        past_key_value = DynamicCache(config=self.mlpm.config)
        past_key_values_length = past_key_value.get_seq_length()
        position_ids = torch.arange(
            past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
        ).unsqueeze(0)
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            None,
            (bsz, seq_len),
            x,
            past_key_values_length=past_key_values_length,
        )
        if self.mlpm.config._attn_implementation == "eager":
            attention_mask = _prepare_4d_causal_attention_mask(
                None,
                (bsz, seq_len),
                x,
                past_key_values_length,
            )

        residual = x
        x = self.mlpm.iln_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=x)
        x = self.mlpm.self_attn_func(
            self.cmv.mlpm_ci,
            layer_idx=layer_idx,
            hidden_states=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        
        x = self.mlpm.paln_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=x)
        x = residual + x


        
        # topk_idx, topk_weight, _ = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_idx, x)
        topk_idx, topk_weight, _ = self.mlpm.gate_func(self.hmv.mlpm_hi, layer_idx, x.to("cpu"))
        _ = topk_weight

        # 同时打印两套对象同名参数，确认是否真不一致
        r_c = self.cmv.mlpm_ci.model.language_model.layers[layer_idx].router
        r_h = self.hmv.mlpm_hi.model.language_model.layers[layer_idx].router
        print("cmv per_expert_scale", r_c.per_expert_scale.device, r_c.per_expert_scale.dtype)
        print("hmv per_expert_scale", r_h.per_expert_scale.device, r_h.per_expert_scale.dtype)
        print("max diff per_expert_scale:",
            (r_c.per_expert_scale.detach().float().cpu() - r_h.per_expert_scale.detach().float().cpu()).abs().max())
        print("max diff proj.weight:",
            (r_c.proj.weight.detach().float().cpu()))

        num_experts = int(self.mlpm.get_experts_num())
        top_k = int(topk_idx.shape[-1])

        # 按 top-k 槽位统计每个专家被路由到的总激活次数（同一 token 不同槽位会重复计数）
        flat_expert_indices = topk_idx.reshape(-1).to(torch.int64).cpu()
        counts = torch.bincount(flat_expert_indices, minlength=num_experts)
        total_activations = int(counts.sum().item())
        active_experts = int((counts > 0).sum().item())

        stats: list[tuple[int, int, float]] = []
        for expert_id in range(num_experts):
            cnt = int(counts[expert_id].item())
            pct = (cnt / total_activations * 100.0) if total_activations > 0 else 0.0
            stats.append((expert_id, cnt, pct))
        stats.sort(key=lambda item: (item[1], item[0]))

        logger.info(
            "[test_gate_experts] bsz=%d seq_len=%d top_k=%d experts=%d active_experts=%d total_activations=%d",
            bsz,
            seq_len,
            top_k,
            num_experts,
            active_experts,
            total_activations,
        )
        logger.info("[test_gate_experts] expert activation distribution (sorted by token count asc):")
        for expert_id, cnt, pct in stats:
            logger.info(
                "[test_gate_experts] expert=%d tokens=%d pct=%.4f%%",
                expert_id,
                cnt,
                pct,
            )
        # 从低激活专家开始做累计，观察“覆盖前 k 个专家”时累计 token 占比。
        cumulative_tokens = 0
        logger.info(
            "[test_gate_experts] cumulative token ratio by ascending experts "
            "(k/%d experts -> cumulative_tokens/total_activations)",
            num_experts,
        )
        for k, (_expert_id, cnt, _pct) in enumerate(stats, start=1):
            cumulative_tokens += cnt
            cumulative_pct = (cumulative_tokens / total_activations * 100.0) if total_activations > 0 else 0.0
            expert_pct = k / num_experts * 100.0
            logger.info(
                "[test_gate_experts] k=%d expert_pct=%.4f%% cumulative_tokens=%d cumulative_pct=%.4f%%",
                k,
                expert_pct,
                cumulative_tokens,
                cumulative_pct,
            )