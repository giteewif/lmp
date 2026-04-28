from importlib import import_module
import os
from threading import get_ident
from turtle import position
from typing import Dict
import copy
from lmp.pinpool import gpinpool
import torch
import time
from transformers import AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask
)

# sllm_store
from sllm_store._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors_from_shared_memory_names,
    restore_experts_tensor_from_shared_memory,
    restore_tensors2,
    free_cuda_memory,
)
from sllm_store.client import SllmStoreClient

# lmp
from lmp.sllm_store_c import SLLM_ADDRESS, load_into_cpu
from utils import cuda_h
from utils.cuda_h import cuda_hook, cuda_hook_end, cuda_hook_time, cuda_hook_time_end
from utils.logger import init_logger
from utils.helper import *
from models.mlpmodule import MLPModuleWrapper, ExpertEinsumTask
from lmp.cuda_memory_view import (
    CudaMemoryView,
    HostMemoryView,
    _group_fused_select_half_experts,
    _group_fused_test_parse_half_modes,
)
from lmp.sllm_thread_manager import SLLMTM

logger = init_logger(__name__)


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

        mlpm = MLPModuleWrapper(model_name_type, model_path)
        self.mlpm  = mlpm
        
        cuda_hook_time("init_cmv_hmv")
        self.cmv = CudaMemoryView(self.mlpm, device_list)
        self.cmv.start_init_meta_model()

        empty_model = copy.deepcopy(self.cmv.mlpm_ci)
        self.hmv = HostMemoryView(self.mlpm, empty_model=empty_model)
        cuda_hook_time_end("init_cmv_hmv")

        sllmtm = SLLMTM(num_workers=1)  # SLLM 线程管理器，用于异步加载
        self.sllmtm = sllmtm
        self.sllmtm.start()  # 启动工作线程

        self.cmv.sllmtm = sllmtm     # 将sllmtm绑定到cmv中  
    

    def _mp_next_cpu_request_id(self) -> int:
        if not hasattr(self, "_cpu_mp_request_id"):
            self._cpu_mp_request_id = 0
        self._cpu_mp_request_id += 1
        return int(self._cpu_mp_request_id)

    @staticmethod
    def _mp_partition_experts_across_cpu_workers(
        cpu_expert_idx_list: list[int],
        expert_indices_map: Dict[int, tuple[int, int]],
        num_workers: int,
    ) -> list[list[int]]:
        """
        Partition CPU experts across MP workers in ``O(E * N)`` (``E`` experts, ``N`` workers).

        - **Equal counts:** each worker gets ``floor(E/N)`` or ``ceil(E/N)`` experts (at most one apart).
        - **Max token:** process experts in descending routed slot count; assign each to a worker
          that still has quota and minimizes ``max(bucket_max_tokens, this_expert_tokens)``, then
          lower current bucket max, then fill order — aligns with fused BMM ``[E, max_tokens, H]``
          padding without scanning all buckets' full max each step.
        """
        nw = max(1, int(num_workers))
        buckets: list[list[int]] = [[] for _ in range(nw)]
        eids = list(cpu_expert_idx_list)
        if not eids:
            return buckets

        def _tok(eid: int) -> int:
            s, e_idx = expert_indices_map[eid]
            return int(e_idx - s)

        e_total = len(eids)
        base, rem = divmod(e_total, nw)
        target_counts = [base + (1 if i < rem else 0) for i in range(nw)]
        cur_max = [0 for _ in range(nw)]

        for eid in sorted(eids, key=_tok, reverse=True):
            cand = [i for i in range(nw) if len(buckets[i]) < target_counts[i]]
            if not cand:
                bi = min(range(nw), key=lambda j: (len(buckets[j]), j))
                buckets[bi].append(eid)
                cur_max[bi] = max(cur_max[bi], _tok(eid))
                continue
            t_e = _tok(eid)
            best_i = min(
                cand,
                key=lambda i: (max(cur_max[i], t_e), cur_max[i], len(buckets[i]), i),
            )
            buckets[best_i].append(eid)
            cur_max[best_i] = max(cur_max[best_i], t_e)
        return buckets
        
    def free_cmv(self):
        """
        Make the instance re-entrant for repeated `test_mp_generate_multi_device_layer()` runs.

        We must:
        - stop/drain async SLLM loader tasks (to avoid restoring into a soon-to-be-reset model)
        - free allocated CUDA memory handles
        - recreate the meta model so all parameters are back on `meta` (no dangling freed storages)
        - clear per-run caches on `MLPLLM` and `CudaMemoryView`
        """

        # 1) Best-effort clear CPU MP workers' cached group lists/unmaps.
        if hasattr(self, "cpu_thread_manager_mp") and self.cpu_thread_manager_mp is not None:
            try:
                dummy = torch.empty(0, device="cpu")
                n_workers = int(getattr(self.cpu_thread_manager_mp, "num_workers", 1))
                pending = set()
                for wid in range(n_workers):
                    rid = self._mp_next_cpu_request_id()
                    pending.add(rid)
                    self.cpu_thread_manager_mp.submit_worker(
                        worker_idx=wid,
                        layer_idx=-1,
                        expert_idx_list=[],
                        expert_indices_map={},
                        flat_hidden_states=dummy,
                        idxs=dummy.to(dtype=torch.int64, copy=False),
                        request_id=rid,
                    )
                while pending:
                    res = self.cpu_thread_manager_mp.wait()
                    if getattr(res, "request_id", -1) in pending:
                        pending.remove(res.request_id)
            except Exception:
                pass

        # 2) Free previously allocated CUDA memory blocks + reset CMV internal maps.
        if hasattr(self, "cmv") and self.cmv is not None:
            self.cmv.free_allocated()

            # 3) Recreate meta model for a clean restore target.
            self.cmv.start_init_meta_model()

            # Re-bind loader if present.
            if hasattr(self, "sllmtm"):
                self.cmv.sllmtm = self.sllmtm

        # 4) Clear fused experts caches that live on MLPLLM.
        if hasattr(self, "_fused_experts_state_dict_cache"):
            self._fused_experts_state_dict_cache = {}

    def init_mp_process(self):
        from lmp.cpu_thread_manager_mp import CPUExpertsManagerMP
        from lmp.device_mp import DeviceMP
        cuda_hook_time("init_mp_process")
        self.cpu_thread_manager_mp = CPUExpertsManagerMP(num_workers=1, model_path=self.mlpm.model_path, model_name_type=self.mlpm.model_name_type)
        self.cpu_thread_manager_mp.start()
        self.cpu_thread_manager_mp.wait_worker_bootstrap_ready()
        # self.dp = DeviceMP(num_processes=len(self.device_list))
        # self.dp.start()
        cuda_hook_time_end("init_mp_process")

        # cuda_hook_time("warm_up_mp_process")
        # self.test_mp_process()
        # cuda_hook_time_end("warm_up_mp_process")



    @torch.no_grad()
    def _warm_up_prefill_compute_kernels(self, hidden_states: torch.Tensor, layer_idx: int = 0):
        raw = os.environ.get("LMP_COMPUTE_WARMUP", "1").strip()
        if raw in ("0", "false", "False", "no", "No", "off", "OFF"):
            return
        if not hidden_states.is_cuda:
            return

        device = hidden_states.device
        cuda_hook_time("compute_kernel_warmup")
        try:
            flat = hidden_states.view(-1, hidden_states.size(-1))
            t = int(flat.size(0))
            h = int(flat.size(1))
            k = int(self.mlpm.get_experts_per_tok())
            if t <= 0 or h <= 0 or k <= 0:
                cuda_hook_time_end("compute_kernel_warmup")
                return

            topk_idx, topk_weight, _ = self.mlpm.gate_func(
                self.cmv.mlpm_ci, layer_idx, hidden_states
            )
            flat_expert_indices = topk_idx.reshape(-1).to(dtype=torch.int64)
            idxs = flat_expert_indices.argsort()
            token_idxs = idxs // k

            if hasattr(self, "cpu_thread_manager_mp") and self.cpu_thread_manager_mp is not None:
                dummy_flat = torch.empty((0, h), device="cpu", dtype=hidden_states.dtype)
                dummy_idxs = torch.empty((0,), device="cpu", dtype=torch.int64)
                n_workers = int(getattr(self.cpu_thread_manager_mp, "num_workers", 1))
                pending = set()
                for wid in range(n_workers):
                    rid = self._mp_next_cpu_request_id()
                    pending.add(rid)
                    self.cpu_thread_manager_mp.submit_worker(
                        worker_idx=wid,
                        layer_idx=-1,
                        expert_idx_list=[],
                        expert_indices_map={},
                        flat_hidden_states=dummy_flat,
                        idxs=dummy_idxs,
                        request_id=rid,
                    )
                while pending:
                    res = self.cpu_thread_manager_mp.wait()
                    if getattr(res, "request_id", -1) in pending:
                        pending.remove(res.request_id)

            x_slots = flat[token_idxs]
            slot_w = topk_weight.reshape(-1, 1).to(device=device)
            weighted = x_slots * slot_w[idxs].to(device=device)
            dst = torch.zeros_like(flat)
            dst.scatter_reduce_(
                0,
                token_idxs[:, None].expand(-1, h),
                weighted,
                reduce="sum",
                include_self=False,
            )

            e = min(4, max(1, int(self.mlpm.get_experts_num())))
            m = min(32, max(1, t))
            n = min(256, max(1, h))
            a = torch.empty((e, m, n), device=device, dtype=hidden_states.dtype)
            b = torch.empty((e, n, n), device=device, dtype=hidden_states.dtype)
            torch.bmm(a, b)

            del topk_idx, topk_weight, flat_expert_indices, idxs, token_idxs
            del x_slots, slot_w, weighted, dst, a, b
            torch.cuda.synchronize(device)
        except Exception as e:
            logger.warning("compute kernel warmup skipped/failed: %s", e)
        finally:
            cuda_hook_time_end("compute_kernel_warmup")

    def mp_stop(self):
        self.cpu_thread_manager_mp.stop()


    def test_mp_cpu_tensor(self):

        cuda_hook_time("test_mp_cpu_tensor")
        device = "cpu"
        bsz, seq_len = 1, 128

        # print_layer_parameters(self.hmv.mlpm_hi)

        hidden_size = int(getattr(self.mlpm.config, "hidden_size"))
        dtype = getattr(self.mlpm.config, "torch_dtype", torch.bfloat16)
        if dtype is None:
            dtype = torch.bfloat16

        x = torch.randn((bsz, seq_len, hidden_size), device=device, dtype=dtype)
        
        layer_idx = 0

        cuda_hook_time("cpu_gate")
        topk_idx, topk_weight, _ = self.mlpm.gate_func(self.hmv.mlpm_hi, layer_idx, x)
        cuda_hook_time_end("cpu_gate")

        cuda_hook_time("before bmm_cpu_experts")
        flat_expert_indices = topk_idx.reshape(-1).to(torch.int64)
        idxs = flat_expert_indices.argsort().to(device)
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        flat_hidden_states = x.reshape(bsz * seq_len, hidden_size)

        num_experts = int(self.mlpm.get_experts_num())
        expert_indices_map: dict[int, tuple[int, int]] = {}
        prev_end = 0
        for expert_id in range(num_experts):
            if expert_id >= len(tokens_per_expert):
                break
            end_idx = int(tokens_per_expert[expert_id])
            if end_idx == prev_end:
                continue
            start_idx = prev_end
            expert_indices_map[expert_id] = (start_idx, end_idx)
            prev_end = end_idx

        # -----------------------
        # CPU Prefill: 使用 CPU 线程管理器进行专家计算
        # -----------------------
        if not getattr(self, "cpu_thread_manager_mp", None):
            raise RuntimeError("cpu_thread_manager_mp is not initialized; call init_mp_process() first.")

        layer_idx = 0
        expert_idx_list = sorted(expert_indices_map.keys())
        if not expert_idx_list:
            logger.warning("[test_mp_cpu_tensor] no experts activated")
            return
        cuda_hook_time_end("before bmm_cpu_experts")

        cuda_hook_time("bmm_cpu_experts")

        # 准备数据（已经在 CPU 上）
        flat_hidden_states_cpu = flat_hidden_states
        idxs_cpu = idxs.to(torch.int64)

        # 获取 worker 数量并将专家平均分配到各个 worker
        n_workers = int(getattr(self.cpu_thread_manager_mp, "num_workers", 1))
        buckets = self._mp_partition_experts_across_cpu_workers(
            cpu_expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            num_workers=n_workers,
        )

        # 提交任务到各个 worker（使用 BMM 模式）
        cpu_submit_meta = []
        for wid, eids_sub in enumerate(buckets):
            if not eids_sub:
                continue
            rid = self._mp_next_cpu_request_id()
            self.cpu_thread_manager_mp.submit_worker(
                worker_idx=wid,
                layer_idx=layer_idx,
                expert_idx_list=sorted(eids_sub),
                expert_indices_map={eid: expert_indices_map[eid] for eid in eids_sub},
                flat_hidden_states=flat_hidden_states_cpu,
                idxs=idxs_cpu,
                use_bmm=True,
                request_id=rid,
                device=flat_hidden_states.device,
            )
            cpu_submit_meta.append((rid, sorted(eids_sub)))

        # 等待所有 worker 完成并收集结果
        pending = {rid for rid, _eids in cpu_submit_meta}
        results: dict[int, torch.Tensor] = {}
        while pending:
            res = self.cpu_thread_manager_mp.wait()
            if getattr(res, "request_id", -1) in pending:
                pending.remove(res.request_id)
                results[res.request_id] = res.final_hidden_states

        cuda_hook_time_end("bmm_cpu_experts")

        cuda_hook_time("scatter_reduce_")
        # 初始化 expert_cache，用于聚合所有专家的输出
        expert_cache = torch.zeros_like(flat_hidden_states_cpu)
        token_idxs = idxs_cpu // int(self.mlpm.get_experts_per_tok())
        
        # 处理每个 worker 的结果，使用 scatter_reduce_ 进行加权聚合
        for rid, eids_sub in cpu_submit_meta:
            out_sub = results.get(rid)
            if out_sub is None:
                continue
            
            # 对当前 worker 处理的专家进行加权
            w_parts = []
            token_parts = []
            for eid in eids_sub:
                start_idx, end_idx = expert_indices_map[eid]
                w_parts.append(topk_weight.reshape(-1, 1)[start_idx:end_idx])
                token_parts.append(token_idxs[start_idx:end_idx])
            
            if w_parts:
                w = torch.cat(w_parts, dim=0)
                tok = torch.cat(token_parts, dim=0)
                out_weighted = out_sub * w
                
                # 使用 scatter_reduce_ 将结果聚合到 expert_cache
                expert_cache.scatter_reduce_(
                    dim=0,
                    index=tok.view(-1, 1).expand(-1, expert_cache.size(-1)),
                    src=out_weighted,
                    reduce="sum",
                )

        cuda_hook_time_end("scatter_reduce_")
        
        # 恢复原始形状
        output = expert_cache.view(bsz, seq_len, hidden_size)
        logger.info(f"[test_mp_cpu_tensor] CPU prefill completed, output shape: {output.shape}")
        cuda_hook_time_end("test_mp_cpu_tensor")

    @torch.no_grad()
    def test_mp_basic_load(self):
        
        cuda_hook_time("load weights")
        self.cmv.load_general_and_init()
        self.cmv.load_qkvgon_weight_onetime()
        cuda_hook_time_end("load weights")


    @torch.no_grad()
    def test_mp_generate_multi_device_layer(self):
        cuda_hook_time("generate_input_ids")
        # 32, 64, 128
        # Keep this lightweight to avoid OOM on shared GPUs.
        batch_size = 2
        seq_len = 128
        dtype = self.mlpm.config.torch_dtype
        hidden_size = self.mlpm.config.hidden_size
        
        device_list = self.device_list
        device1 = device_list[0]
        inputs_tokens = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device1)

        tokenizer=AutoTokenizer.from_pretrained(self.mlpm.model_abs_path, trust_remote_code=True)
        inputs_ids = generate_input_ids(tokenizer, batch_size, seq_len, device1)
        cuda_hook_time_end("generate_input_ids")

        cuda_hook_time("init_cache")
        past_key_value = DynamicCache(config=self.mlpm.config)
        past_key_values_length = past_key_value.get_seq_length()
        cuda_hook_time_end("init_cache")
    

        cuda_hook_time("init_general_weights")
        self.cmv.load_general_and_init()
        # only support one first replace dense layer
        self.cmv.init_load_qkvogn_es_weight(layer_idx=0)
        cuda_hook_time_end("init_general_weights")

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

        self._warm_up_prefill_compute_kernels(inputs_tokens, layer_idx=0)

        cuda_hook_time("prefill")
        time_start_prefill = time.time()
        
        if len(self.device_list) == 4:
            # self.num_experts_on_cpu_ratio = 0.2
            self.num_experts_on_cpu_ratio = 0.2
            self.num_experts_on_cpu_ratio = 0.5
        elif len(self.device_list) == 3:
            self.num_experts_on_cpu_ratio = 0.25
            self.num_experts_on_cpu_ratio = 0.5
        else:
            self.num_experts_on_cpu_ratio = 0.5
        from models.mlpmodule import (
            QWEN2_MODEL_NAME_TYPE,
            MIXTRAL_MODEL_NAME_TYPE,
            QWEN3_MODEL_NAME_TYPE,
            GEMMA4_MODEL_NAME_TYPE,
        )
        if self.mlpm.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            if len(self.device_list) == 4:
                self.num_experts_on_cpu_ratio = 0.5
            elif len(self.device_list) == 2:
                self.num_experts_on_cpu_ratio = 0.8
            elif len(self.device_list) == 1:
                self.num_experts_on_cpu_ratio = 0.9
        elif self.mlpm.model_name_type == QWEN3_MODEL_NAME_TYPE:
            if len(self.device_list) == 1:
                self.num_experts_on_cpu_ratio = 0.7

        self.num_experts_on_cpu_ratio = 1
        
        ghidden_states = inputs_tokens
        for layer_idx in range(self.mlpm.config.num_hidden_layers):
            cuda_hook_time("prefill_layer")
            logger.debug(f"-------------------------------- start prefill layer {layer_idx} --------------------------------")
            
            cuda_hook_time("*iln_self_attn_paln")
            if layer_idx < self.mlpm.config.num_hidden_layers-1:
                self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1, device=self.device1)

            residual = ghidden_states
            ghidden_states = self.mlpm.iln_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=ghidden_states)
            cuda_hook_time("**self_attn")
            ghidden_states = self.mlpm.self_attn_func(
                self.cmv.mlpm_ci, layer_idx=layer_idx,
                hidden_states=ghidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            cuda_hook_time_end("**self_attn")

            ghidden_states = residual + ghidden_states
            residual = ghidden_states
            ghidden_states = self.mlpm.paln_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=ghidden_states)
            cuda_hook_time_end("*iln_self_attn_paln")

            cuda_hook_time("*mlp")
            if layer_idx < self.mlpm.get_first_k_dense_replace():
                cuda_hook_time("dense_mlp")
                # self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1,  device=device1)
                ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=ghidden_states)
                self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
                cuda_hook_time_end("dense_mlp")
            else:
                # Gemma4 routed experts are fused (gate_up_proj + down_proj); use fused multi-device path.
                if self.mlpm.model_name_type == GEMMA4_MODEL_NAME_TYPE:
                    ghidden_states = self.layer_moe_generate_mp_multi_device_fused(
                        layer_idx=layer_idx, hidden_states=ghidden_states
                    )
                else:
                    ghidden_states = self.layer_moe_generate_mp_multi_device(
                        layer_idx=layer_idx, hidden_states=ghidden_states
                    )
                # ghidden_states = self.layer_moe_dgenerate_mp_multi_device(layer_idx=layer_idx, hidden_states=ghidden_states)
            ghidden_states = ghidden_states + residual
            cuda_hook_time_end("*mlp")
            # if check_nan_inf(ghidden_states):
            #     logger.warning(f"ghidden_states is nan or inf at layer {layer_idx}")

            cuda_hook_time_end("prefill_layer")
            logger.debug(f"-------------------------------- end prefill layer {layer_idx} --------------------------------")            
        cuda_hook_time_end("prefill")
        logger.info(f"prefill time: {time.time() - time_start_prefill} seconds")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return
    
        cuda_hook_time("async_load_ce")
        if len(self.device_list) > 1:
            self.cmv.async_load_experts_decode_cpu_weight_multi_device()
        else:
            self.cmv.async_load_experts_decode_cpu_weight()
        cuda_hook_time_end("async_load_ce")

        num_step=31
        time_decode_list = []
        for i in range(num_step):
            time_start_decode=time.time()
            cuda_hook_time("decode_step")
            cuda_hook_time("init_inputs_tokens")
            # 更新 past_key_values_length：每次 decode step 后，past_key_value 会增加一个 token
            # get_seq_length() 返回当前 cache 的逻辑序列长度（与 Transformers Cache 接口一致）
            # 第一次 decode step 时，past_key_values_length = seq_len（prefill 的长度）
            # 之后每次 decode step 后，past_key_value 会被更新，长度自动增加 1
            past_key_values_length = past_key_value.get_seq_length()
            
            next_token_ids = get_next_token_helper(self.cmv.mlpm_ci, ghidden_states, self.device1)
            next_inputs_tokens = self.cmv.mlpm_ci.model.embed_tokens(next_token_ids)
            
            # 确保 next_inputs_tokens 的形状正确：(batch_size, 1, hidden_size)
            # input_shape 应该是 (batch_size, query_length)，其中 query_length = 1（decode 阶段每次只处理一个 token）
            query_length = next_inputs_tokens.shape[1]  # 应该是 1
            input_shape = (batch_size, query_length)
            
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + query_length, dtype=torch.long, device=device1
            )
            position_ids = position_ids.unsqueeze(0)
            
            if self.mlpm.config._attn_implementation == "eager":
                # 4d mask is passed through the layers
                # eager 实现需要显式的 4D mask
                attention_mask = _prepare_4d_causal_attention_mask(
                    None,
                    input_shape,
                    next_inputs_tokens,
                    past_key_values_length,
                )
            elif self.mlpm.config._attn_implementation == "sdpa":
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    None,
                    input_shape,
                    next_inputs_tokens,
                    past_key_values_length=past_key_values_length,
                )
            if attention_mask is None:
                # 如果 _prepare_4d_causal_attention_mask 返回 None（通常发生在 query_length == 1 时），
                # 手动创建一个 causal attention mask
                # 对于 decode 阶段：query_length = 1, key_value_length = past_key_values_length + 1
                key_value_length = past_key_values_length + query_length
                batch_size, query_length = input_shape
                
                # 创建一个 causal mask: (batch_size, 1, query_length, key_value_length)
                # 对于 decode 阶段，当前 token 应该能看到所有 past tokens 和自己
                # mask 值为 0 表示可以 attend，负无穷表示不能 attend
                # 对于 causal mask，下三角（包括对角线）应该是 0，上三角应该是负无穷
                # 但在 decode 阶段，query_length=1，所以只需要一行：[0, 0, ..., 0]（全0，表示都可以看到）
                attention_mask = torch.zeros(
                    (batch_size, 1, query_length, key_value_length),
                    dtype=next_inputs_tokens.dtype,
                    device=next_inputs_tokens.device
                )
            cuda_hook_time_end("init_inputs_tokens")

            ghidden_states=next_inputs_tokens
            logger.debug(f"next_inputs_tokens shape: {next_inputs_tokens.shape}")
            # decode
            for layer_idx in range(self.mlpm.config.num_hidden_layers):
                logger.debug(f"-------------------------------- start decode layer {layer_idx} --------------------------------")

                cuda_hook_time("iln_self_attn_paln")
                residual = ghidden_states
                ghidden_states = self.mlpm.iln_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=ghidden_states)
                cuda_hook_time("self_attn")
                ghidden_states = self.mlpm.self_attn_func(
                    self.cmv.mlpm_ci, layer_idx=layer_idx,
                    hidden_states=ghidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                )
                cuda_hook_time_end("self_attn")
                ghidden_states = residual + ghidden_states
                residual = ghidden_states
                ghidden_states = self.mlpm.paln_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=ghidden_states)
                cuda_hook_time_end("iln_self_attn_paln")

                if layer_idx < self.mlpm.get_first_k_dense_replace():
                    cuda_hook_time("dense_mlp")
                    ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=0, hidden_states=ghidden_states)
                    logger.debug(f"ghidden_states after dense_mlp_func shape: {ghidden_states.shape}")
                    cuda_hook_time_end("dense_mlp")
                else:
                    ghidden_states = self.layer_moe_dgenerate_mp_multi_device(layer_idx=layer_idx, hidden_states=ghidden_states)
                ghidden_states = ghidden_states + residual
                logger.debug(f"-------------------------------- end decode layer {layer_idx} --------------------------------")
            cuda_hook_time_end("decode_step")
            decode_time_cost= time.time() - time_start_decode
            time_decode_list.append(decode_time_cost)
            logger.info(f"decode step {i} time: {decode_time_cost} seconds")
            torch.cuda.synchronize()

            # 清空 group_list
            dummy = torch.empty(0, device="cpu")
            n_workers = int(getattr(self.cpu_thread_manager_mp, "num_workers", 1))
            pending = set()
            for wid in range(n_workers):
                rid = self._mp_next_cpu_request_id()
                pending.add(rid)
                self.cpu_thread_manager_mp.submit_worker(
                    worker_idx=wid,
                    layer_idx=-1,
                    expert_idx_list=[],
                    expert_indices_map={},
                    flat_hidden_states=dummy,
                    idxs=dummy.to(dtype=torch.int64, copy=False),
                    request_id=rid,
                )
            while pending:
                res = self.cpu_thread_manager_mp.wait()
                if getattr(res, "request_id", -1) in pending:
                    pending.remove(res.request_id)

        if len(time_decode_list) >= 5:
            time_decode_list = time_decode_list[5:]
        logger.info(f"average decode time from decode step 5: {sum(time_decode_list) / len(time_decode_list)} seconds")
        cuda_hook_time("async_wait_layer_loaded_to_gpu")
        if len(self.device_list) > 1:
            self.cmv.async_wait_layer_loaded_to_gpu_multi_device()
        else:
            self.cmv.async_wait_layer_loaded_to_gpu()
        cuda_hook_time_end("async_wait_layer_loaded_to_gpu")
    
    

    def layer_moe_generate_mp_multi_device_fused(self, layer_idx: int, hidden_states: torch.Tensor):
        """
        Multi-GPU fused experts version (Gemma4-style ``gate_up_proj`` + ``down_proj``).

        - Gate once (on current device) to build expert-sorted slot stream via ``idxs``.
        - Partition hit experts across GPUs (greedy by token count, same as mp_multi_device).
        - Load experts to multiple GPUs via ``allocate_cuda_memory_and_load_into_gpu_multi_device``.
        - For each GPU, run fused gate+up then down using either:
          - ``bmm``: pad-to-``[E_dev, max_tokens, H]`` then 2x ``torch.bmm``;
          - ``gmm``: grouped-mm via transformers `_grouped_mm` (no padding).
        - Aggregate weighted slot outputs back to token space with ``scatter_reduce_``.
        """
        cuda_hook_time(f"layer_moe_generate_mp_multi_device_fused_l_{layer_idx+1}")

        cuda_hook_time(f"*before experts start*")

        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape

        # -----------------------
        # Step 1: gate + sort slots by expert (presorted stream)
        # -----------------------
        cuda_hook_time("gate")
        topk_idx, topk_weight, aux_loss = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_idx, hidden_states)
        flat_expert_indices = topk_idx.view(-1).to(dtype=torch.int64)  # [slots]
        flat_experts_weight = topk_weight.view(-1, 1)  # [slots,1]
        idxs = flat_expert_indices.argsort()
        token_idxs = idxs // self.mlpm.get_experts_per_tok()
        flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)

        flat_hidden_states_on_cpu_pin = gpinpool.alloc_same_pin_tensor(flat_hidden_states)
        flat_hidden_states_on_cpu_pin.copy_(flat_hidden_states, non_blocking=True)
        cuda_hook_time("idxs2cpu")
        idxs_int64_on_cpu_pin = idxs.to(dtype=torch.int64, device="cpu", non_blocking=True)
        cuda_hook_time_end("idxs2cpu")

        cuda_hook_time_end("gate")

        num_slots = int(idxs.numel())
        expert_cache = torch.zeros_like(flat_hidden_states)
        if num_slots == 0:
            cuda_hook_time("gpu_sexperts")
            y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
            cuda_hook_time_end("gpu_sexperts")
            cuda_hook_time_end(f"layer_moe_generate_mp_multi_device_fused_l_{layer_idx+1}")
            return expert_cache.view(*orig_shape) + y

        # -----------------------
        # Step 2: build expert ranges + partition experts across devices
        # -----------------------
        cuda_hook_time("experts_map_get")
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        expert_indices_map: Dict[int, tuple[int, int]] = {}
        expert_token_counts_list = []
        num_experts = self.mlpm.get_experts_num()
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

        num_device = len(self.device_list)
        device_ids = [int(device.split(":")[1]) for device in self.device_list]
        # Partition hit experts: smallest load -> CPU, remaining -> GPUs (greedy balance by token count).
        sorted_experts_by_load = sorted(expert_token_counts_list, key=lambda x: x[1])
        num_experts_total = len(sorted_experts_by_load)
        num_experts_on_cpu = int(num_experts_total * self.num_experts_on_cpu_ratio)
        cpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[:num_experts_on_cpu])
        gpu_experts_list = sorted_experts_by_load[num_experts_on_cpu:]

        device_expert_map = {device_id: [] for device_id in device_ids}
        device_token_counts = {device_id: 0 for device_id in device_ids}
        gpu_experts_sorted = sorted(gpu_experts_list, key=lambda x: x[1], reverse=True)
        for expert_id, token_count in gpu_experts_sorted:
            min_device_id = min(device_ids, key=lambda d: device_token_counts[d])
            device_expert_map[min_device_id].append(expert_id)
            device_token_counts[min_device_id] += token_count

        gpu_expert_ids_by_device = {
            device_id: set(expert_ids) for device_id, expert_ids in device_expert_map.items()
        }
        cuda_hook_time_end("experts_map_get")

        cuda_hook_time_end(f"*before experts start*")

        cuda_hook_time("*single cpu experts*")
        # -----------------------
        # Step 3: load experts multi-device + shared experts
        # -----------------------
        # -----------------------
        # Step 3: submit CPU experts (MP) + load fused experts multi-device + shared experts
        # -----------------------
        cpu_submit_meta: list[tuple[int, list[int]]] = []
        cuda_hook_time("cpu_experts_submit_fused_mp")
        if cpu_expert_ids:
            cpu_expert_idx_list = sorted(cpu_expert_ids)

            n_workers = int(getattr(self.cpu_thread_manager_mp, "num_workers", 1))
            buckets = self._mp_partition_experts_across_cpu_workers(
                cpu_expert_idx_list=cpu_expert_idx_list,
                expert_indices_map=expert_indices_map,
                num_workers=n_workers,
            )
            for wid, eids_sub in enumerate(buckets):
                if not eids_sub:
                    continue
                rid = self._mp_next_cpu_request_id()
                self.cpu_thread_manager_mp.submit_worker(
                    worker_idx=wid,
                    layer_idx=layer_idx,
                    expert_idx_list=sorted(eids_sub),
                    expert_indices_map={eid: expert_indices_map[eid] for eid in eids_sub},
                    flat_hidden_states=flat_hidden_states_on_cpu_pin,
                    idxs=idxs_int64_on_cpu_pin,
                    use_bmm=True,
                    request_id=rid,
                    device=flat_hidden_states.device,
                )
                cpu_submit_meta.append((rid, sorted(eids_sub)))
        cuda_hook_time_end("cpu_experts_submit_fused_mp")


        # Activation for fused gate half.
        _, _, act_fn = self.mlpm.get_fused_experts_gate_up_down_act_fn(self.cmv.mlpm_ci, layer_idx)
        mm_backend = os.environ.get("LMP_FUSED_EXPERT_MM", "bmm").strip().lower()

        # Presorted slot tensors on "main" device (where hidden_states live)
        main_device = flat_hidden_states.device
        x_slots_all = flat_hidden_states[token_idxs]  # [slots,H] presorted
        expert_ids_all = flat_expert_indices[idxs].to(device=main_device, dtype=torch.int64)  # [slots]
        slot_w_all = flat_experts_weight[idxs].to(device=main_device)  # [slots,1]

        

        cuda_hook_time_end("*single cpu experts*")

        cuda_hook_time("*gpu experts*")

        cuda_hook_time("*before fused experts*")
        any_gpu_experts = any(bool(gpu_expert_ids_by_device.get(d)) for d in device_ids)
        state_dict_packed = {}
        replica_uuid = None
        if any_gpu_experts:
            cuda_hook_time("allocate_experts_cuda_memory_and_restore_model_multi_device_fused")
            _ret, replica_uuid, state_dict_packed, _state_dict_slices = self.cmv.allocate_cuda_memory_fused_experts_dual_restore(
                layer_idx=layer_idx,
                gpu_expert_ids_by_device=gpu_expert_ids_by_device,
            )
            # Keep a strong reference to the returned tensors, since they are views into
            # allocated GPU memory and are required for subsequent fused expert compute.
            if not hasattr(self, "_fused_experts_state_dict_cache"):
                self._fused_experts_state_dict_cache = {}
            # Prefer packed banks for compute (2 tensors per device).
            self._fused_experts_state_dict_cache[layer_idx] = state_dict_packed
            # self.cmv.restore2model(state_dict, self.cmv.mlpm_ci)
            cuda_hook_time_end("allocate_experts_cuda_memory_and_restore_model_multi_device_fused")

        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
        cuda_hook_time_end("gpu_sexperts")

        # 等待 qkvogn_s_weight 加载完成
        if layer_idx < self.mlpm.config.num_hidden_layers-1:
            cuda_hook_time("wait_load_qkvogn_s_weight")
            self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
            cuda_hook_time_end("wait_load_qkvogn_s_weight")

    

        cuda_hook_time("prepare_per_device_parameters")
        # Packed banks per device are named like:
        #   "<...gate_up_proj>.packed.dev_<device_id>" -> [E_dev, 2I, H]
        #   "<...down_proj>.packed.dev_<device_id>"    -> [E_dev, H, I]
        # and the corresponding expert id order is recorded in `cmv._layer_experts_map_by_device[layer_idx]`.
        layer_map = self.cmv._layer_experts_map_by_device.get(layer_idx) or {}

        # Step 5.1: prepare per-device parameters
        work_items = []
        for device_id in device_ids:
            device_expert_ids = sorted(gpu_expert_ids_by_device[device_id])
            if not device_expert_ids:
                continue

            # Build contiguous expert-sorted slices for this device.
            x_parts = []
            eid_parts = []
            w_parts = []
            tok_parts = []
            for eid in device_expert_ids:
                start_idx, end_idx = expert_indices_map[eid]
                x_parts.append(x_slots_all[start_idx:end_idx])
                eid_parts.append(expert_ids_all[start_idx:end_idx])
                w_parts.append(slot_w_all[start_idx:end_idx])
                tok_parts.append(token_idxs[start_idx:end_idx])

            x_dev_sorted = torch.cat(x_parts, dim=0)
            eid_dev_sorted = torch.cat(eid_parts, dim=0)
            w_dev_sorted = torch.cat(w_parts, dim=0)
            token_ids_for_slots = torch.cat(tok_parts, dim=0).to(main_device)
            if x_dev_sorted.numel() == 0:
                continue

            device = torch.device(f"cuda:{device_id}")
            x_dev_sorted = x_dev_sorted.to(device, non_blocking=True)
            eid_dev_sorted = eid_dev_sorted.to(device, non_blocking=True)

            # Remap global expert ids -> local [0..E_dev-1] consistent with packed bank order.
            dev_info = layer_map.get(device_id) or {}
            packed_experts = dev_info.get("experts") or []
            packed_experts_int = [int(e) for e in packed_experts]
            if packed_experts_int != device_expert_ids:
                raise RuntimeError(
                    f"Packed expert order mismatch for cuda:{device_id}: "
                    f"packed={packed_experts_int[:16]}... vs expected={device_expert_ids[:16]}..."
                )
            ids_tensor = torch.tensor(packed_experts_int, device=device, dtype=torch.int64)
            remap = torch.full((num_experts,), -1, dtype=torch.int64, device=device)
            remap[ids_tensor] = torch.arange(ids_tensor.numel(), device=device, dtype=torch.int64)
            eid_dev_sub = remap[eid_dev_sorted]

            # Use packed weight banks directly (no per-expert stacking).
            gate_up_packed_name = (dev_info.get("gate_up_packed") or [None])[0]
            down_packed_name = (dev_info.get("down_packed") or [None])[0]
            if not gate_up_packed_name or not down_packed_name:
                raise RuntimeError(f"Missing packed names for device {device_id}: keys={list(dev_info.keys())}")
            gate_up_sub = state_dict_packed[gate_up_packed_name]
            down_sub = state_dict_packed[down_packed_name]
            w_gu = gate_up_sub.transpose(1, 2).contiguous()  # [E_dev, H, 2I]
            w_down = down_sub.transpose(1, 2).contiguous()  # [E_dev, I, H]

            work_items.append(
                (
                    device_id,
                    x_dev_sorted,
                    eid_dev_sub,
                    w_gu,
                    w_down,
                    w_dev_sorted,
                    token_ids_for_slots,
                )
            )
        cuda_hook_time_end("prepare_per_device_parameters")



        if replica_uuid is not None:
            cuda_hook_time("wait_experts_multi_device")
            self.cmv.wait_load_into_gpu(replica_uuid)
            cuda_hook_time_end("wait_experts_multi_device")

        cuda_hook_time_end("*before fused experts*")

         # -----------------------
        # Step 5: per-device fused compute + aggregate to main expert_cache
        # -----------------------
        cuda_hook_time("gpu_experts_multi_device_fused")

        # Step 5.2: compute and aggregate
        for (
            device_id,
            x_dev_sorted,
            eid_dev_sub,
            w_gu,
            w_down,
            w_dev_sorted,
            token_ids_for_slots,
        ) in work_items:
            if mm_backend == "bmm":
                # Pre-pad once outside of `fused_experts_gate_up_down_mm_presorted` to avoid redundant pad work.
                e_dev = int(w_gu.size(0))
                stacked_inputs, counts = self.mlpm._batched_pad_inputs_presorted(
                    x_dev_sorted, eid_dev_sub, e_dev
                )
                counts_cpu = counts.to(device="cpu", non_blocking=False)
                y_dev_sorted = self.mlpm.fused_experts_gate_up_down_bmm_from_padded(
                    stacked_inputs=stacked_inputs,
                    counts=counts_cpu,
                    gate_up_w_eh2i=w_gu,
                    down_w_eih=w_down,
                    act_fn=act_fn,
                )
            else:
                y_dev_sorted = self.mlpm.fused_experts_gate_up_down_mm_presorted(
                    x_slots_sorted=x_dev_sorted,
                    expert_ids_sorted=eid_dev_sub,
                    gate_up_w_eh2i=w_gu,
                    down_w_eih=w_down,
                    act_fn=act_fn,
                    mm_backend="gmm",
                )

            y_main = y_dev_sorted.to(main_device, non_blocking=True)
            out_weighted = y_main * w_dev_sorted
            expert_cache.scatter_reduce_(
                dim=0,
                index=token_ids_for_slots.view(-1, 1).expand(-1, expert_cache.size(-1)),
                src=out_weighted,
                reduce="sum",
            )

        cuda_hook_time_end("gpu_experts_multi_device_fused")

        cuda_hook_time_end("*gpu experts*")

        # -----------------------
        # Step 4: wait CPU + scatter CPU outputs (weighted)
        # -----------------------
        cuda_hook_time("cpu_thread_manager_mp_wait_fused")
        if cpu_expert_ids:
            # Collect results from whichever worker finishes first, keyed by request_id.
            pending = {rid for rid, _eids in cpu_submit_meta}
            results: dict[int, torch.Tensor] = {}
            while pending:
                res = self.cpu_thread_manager_mp.wait()
                if getattr(res, "request_id", -1) in pending:
                    pending.remove(res.request_id)
                    results[res.request_id] = res.final_hidden_states
            cuda_hook_time("cpu_experts_scatter_reduce")
            for rid, eids_sub in cpu_submit_meta:
                out_sub = results.get(rid)
                if out_sub is None:
                    continue
                w_parts_cpu = []
                token_parts_cpu = []
                for eid in eids_sub:
                    start_idx, end_idx = expert_indices_map[eid]
                    w_parts_cpu.append(slot_w_all[start_idx:end_idx])
                    token_parts_cpu.append(token_idxs[start_idx:end_idx])
                w_cpu = torch.cat(w_parts_cpu, dim=0) if w_parts_cpu else None
                tok_cpu = torch.cat(token_parts_cpu, dim=0) if token_parts_cpu else None
                if w_cpu is not None and tok_cpu is not None and int(tok_cpu.numel()) > 0:
                    out_weighted_cpu = out_sub * w_cpu
                    expert_cache.scatter_reduce_(
                        dim=0,
                        index=tok_cpu.view(-1, 1).expand(-1, expert_cache.size(-1)),
                        src=out_weighted_cpu,
                        reduce="sum",
                    )
            cuda_hook_time_end("cpu_experts_scatter_reduce")
        cuda_hook_time_end("cpu_thread_manager_mp_wait_fused")

        layer_output = expert_cache.view(*orig_shape) + y

        gpinpool.free(flat_hidden_states_on_cpu_pin)
        cuda_hook_time_end(f"layer_moe_generate_mp_multi_device_fused_l_{layer_idx+1}")
        return layer_output

    def layer_moe_dgenerate_mp_multi_device_fused(self, layer_idx: int, hidden_states: torch.Tensor):
        """
        Decode 阶段（通常 seq_len=1）的 fused 多卡 MoE。

        与 prefill/generate 的差异：decode 侧 **按专家当前所在 device**（CPU / 各 GPU）决定执行位置，
        避免沿用 generate 的“按 ratio 划分 + 重新分配”策略导致的频繁迁移/不一致。

        - 先 gate 一次拿到 presorted slot stream（``idxs`` / ``token_idxs``）。
        - 构建命中 experts 的 ``expert_indices_map``。
        - decode 侧 fused experts 的 device 分布从 ``CudaMemoryView`` 的 decode 预加载/映射信息获取，
          不依赖模型层内部 module 的 ``get_expert_device_distribution``（Gemma4 fused bank 结构可能不适用）。
        - 聚合：按 slot weight * 输出，再在主 device 上 ``scatter_reduce_`` 回 token 空间。
        """
        cuda_hook_time(f"layer_moe_dgenerate_mp_multi_device_fused_l_{layer_idx+1}")
        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape

        # -----------------------
        # Step 1: gate + presort slots by expert
        # -----------------------
        cuda_hook_time("gate")
        topk_idx, topk_weight, _aux_loss = self.mlpm.gate_func(
            self.cmv.mlpm_ci, layer_idx, hidden_states
        )
        flat_expert_indices = topk_idx.view(-1).to(dtype=torch.int64)  # [S]
        flat_experts_weight = topk_weight.view(-1, 1)  # [S,1]
        idxs = flat_expert_indices.argsort()
        token_idxs = idxs // self.mlpm.get_experts_per_tok()
        flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)
        cuda_hook_time_end("gate")

        num_slots = int(idxs.numel())
        expert_cache = torch.zeros_like(flat_hidden_states)
        if num_slots == 0:
            cuda_hook_time("gpu_sexperts")
            y = self.mlpm.shared_experts_func(
                self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states
            )
            cuda_hook_time_end("gpu_sexperts")
            out = expert_cache.view(*orig_shape) + y
            cuda_hook_time_end(f"layer_moe_dgenerate_mp_multi_device_fused_l_{layer_idx+1}")
            return out

        # -----------------------
        # Step 2: build expert ranges for presorted slots
        # -----------------------
        cuda_hook_time("experts_map_get")
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        expert_indices_map: Dict[int, tuple[int, int]] = {}
        expert_token_counts_list = []
        num_experts = int(self.mlpm.get_experts_num())
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

        # -----------------------
        # Step 3: decode 分配：按 expert 实际 device（来自 cmv 的映射信息）
        # -----------------------
        device_ids = [int(device.split(":")[1]) for device in self.device_list]
        experts_cpu_list: list[int] = []
        gpu_expert_ids_by_device: dict[int, set[int]] = {d: set() for d in device_ids}

        # Prefer cmv-provided map: {device_id(int): {"experts": ["0","1",...] , ...}}
        layer_map = getattr(self.cmv, "_layer_experts_map_by_device", {}).get(layer_idx) or {}
        if layer_map:
            for d in device_ids:
                dev_info = layer_map.get(d) or {}
                exp = dev_info.get("experts") or []
                for eid in exp:
                    try:
                        gpu_expert_ids_by_device[d].add(int(eid))
                    except (TypeError, ValueError):
                        continue

        active_eids = [eid for eid, _tc in expert_token_counts_list]
        active_set = set(active_eids)
        if any(gpu_expert_ids_by_device[d] for d in device_ids):
            # Use cmv mapping (intersect with active experts)
            for d in device_ids:
                gpu_expert_ids_by_device[d].intersection_update(active_set)
            gpu_union = set().union(*(gpu_expert_ids_by_device[d] for d in device_ids))
            experts_cpu_list = sorted(active_set - gpu_union)
        else:
            # Fallback: all active experts on CPU
            experts_cpu_list = sorted(active_eids)

        cuda_hook_time_end("experts_map_get")

        # -----------------------
        # Step 4: submit CPU experts (MP BMM)
        # -----------------------
        cpu_submit_meta: list[tuple[int, list[int]]] = []
        cuda_hook_time("cpu_experts_submit_fused_mp")
        if experts_cpu_list:
            experts_cpu_list = sorted(experts_cpu_list)
            n_workers = int(getattr(self.cpu_thread_manager_mp, "num_workers", 1))
            buckets = self._mp_partition_experts_across_cpu_workers(
                cpu_expert_idx_list=experts_cpu_list,
                expert_indices_map=expert_indices_map,
                num_workers=n_workers,
            )
            for wid, eids_sub in enumerate(buckets):
                if not eids_sub:
                    continue
                rid = self._mp_next_cpu_request_id()
                self.cpu_thread_manager_mp.submit_worker(
                    worker_idx=wid,
                    layer_idx=layer_idx,
                    expert_idx_list=sorted(eids_sub),
                    expert_indices_map={eid: expert_indices_map[eid] for eid in eids_sub},
                    flat_hidden_states=flat_hidden_states,
                    idxs=idxs,
                    use_bmm=True,
                    request_id=rid,
                )
                cpu_submit_meta.append((rid, sorted(eids_sub)))
        cuda_hook_time_end("cpu_experts_submit_fused_mp")

        # -----------------------
        # Step 5: load GPU fused experts via dual-restore packed banks (per-device big tensors)
        # -----------------------
        cuda_hook_time("allocate_experts_cuda_memory_and_restore_model_multi_device_fused")
        _ret, replica_uuid, state_dict_packed, state_dict_slices = (
            self.cmv.allocate_cuda_memory_fused_experts_dual_restore(
                layer_idx=layer_idx,
                gpu_expert_ids_by_device=gpu_expert_ids_by_device,
            )
        )
        if not hasattr(self, "_fused_experts_state_dict_cache"):
            self._fused_experts_state_dict_cache = {}
        # generate/dgenerate 可能逐步为同一层加载部分 experts；这里把 packed+slices 都 cache，
        # 且用“只补缺不覆盖”的 merge，避免覆盖已存在条目。
        prev = self._fused_experts_state_dict_cache.get(layer_idx)
        if not isinstance(prev, dict):
            prev = {}
            self._fused_experts_state_dict_cache[layer_idx] = prev
        for _sd in (state_dict_packed, state_dict_slices):
            for _k, _v in _sd.items():
                if _k not in prev:
                    prev[_k] = _v
        cuda_hook_time_end("allocate_experts_cuda_memory_and_restore_model_multi_device_fused")

        # shared experts
        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states
        )
        cuda_hook_time_end("gpu_sexperts")

        cuda_hook_time("wait_experts_multi_device")
        self.cmv.wait_load_into_gpu(replica_uuid)
        cuda_hook_time_end("wait_experts_multi_device")

        # Activation + backend
        _, _, act_fn = self.mlpm.get_fused_experts_gate_up_down_act_fn(self.cmv.mlpm_ci, layer_idx)
        mm_backend = os.environ.get("LMP_FUSED_EXPERT_MM", "bmm").strip().lower()

        # Presorted slot tensors on main device
        main_device = flat_hidden_states.device
        x_slots_all = flat_hidden_states[token_idxs]  # [S,H] presorted
        expert_ids_all = flat_expert_indices[idxs].to(device=main_device, dtype=torch.int64)  # [S]
        slot_w_all = flat_experts_weight[idxs].to(device=main_device)  # [S,1]

        # -----------------------
        # Step 6: per-device fused compute (packed banks) + scatter_reduce back to main
        # -----------------------
        cuda_hook_time("gpu_experts_multi_device_fused")
        layer_map = self.cmv._layer_experts_map_by_device.get(layer_idx) or {}

        # Step 6.1: prepare per-device parameters (including stacking non-contiguous experts into contiguous banks)
        work_items = []
        for device_id in device_ids:
            device_expert_ids = sorted(gpu_expert_ids_by_device.get(device_id) or [])
            if not device_expert_ids:
                continue

            x_parts = []
            eid_parts = []
            w_parts = []
            token_parts = []
            for eid in device_expert_ids:
                start_idx, end_idx = expert_indices_map[eid]
                x_parts.append(x_slots_all[start_idx:end_idx])
                eid_parts.append(expert_ids_all[start_idx:end_idx])
                w_parts.append(slot_w_all[start_idx:end_idx])
                token_parts.append(token_idxs[start_idx:end_idx])

            x_dev_sorted = torch.cat(x_parts, dim=0)
            eid_dev_sorted = torch.cat(eid_parts, dim=0)
            w_dev_sorted = torch.cat(w_parts, dim=0)
            tok_dev_sorted = torch.cat(token_parts, dim=0).to(main_device)
            if x_dev_sorted.numel() == 0:
                continue

            device = torch.device(f"cuda:{device_id}")
            x_dev_sorted = x_dev_sorted.to(device, non_blocking=True)
            eid_dev_sorted = eid_dev_sorted.to(device, non_blocking=True)

            # Build contiguous local-id mapping for non-contiguous global expert ids
            ids_tensor = torch.tensor(device_expert_ids, device=device, dtype=torch.int64)
            remap = torch.full((num_experts,), -1, dtype=torch.int64, device=device)
            remap[ids_tensor] = torch.arange(ids_tensor.numel(), device=device, dtype=torch.int64)
            eid_dev_sub = remap[eid_dev_sorted]

            # Stack slice tensors into contiguous banks on this device (order == device_expert_ids)
            dev_info = layer_map.get(device_id) or {}
            gate_slice_names = dev_info.get("gate_up") or []
            down_slice_names = dev_info.get("down") or []
            if len(gate_slice_names) != len(device_expert_ids) or len(down_slice_names) != len(device_expert_ids):
                raise RuntimeError(
                    f"[dgenerate_fused] slice names mismatch for cuda:{device_id}: "
                    f"gate={len(gate_slice_names)} down={len(down_slice_names)} expected={len(device_expert_ids)}"
                )
            gate_up_sub = torch.stack([state_dict_slices[n] for n in gate_slice_names], dim=0)  # [E_dev,2I,H]
            down_sub = torch.stack([state_dict_slices[n] for n in down_slice_names], dim=0)     # [E_dev,H,I]
            w_gu = gate_up_sub.transpose(1, 2).contiguous()  # [E_dev,H,2I]
            w_down = down_sub.transpose(1, 2).contiguous()   # [E_dev,I,H]

            work_items.append((x_dev_sorted, eid_dev_sub, w_gu, w_down, w_dev_sorted, tok_dev_sorted))

        # Step 6.2: compute + scatter
        for x_dev_sorted, eid_dev_sub, w_gu, w_down, w_dev_sorted, tok_dev_sorted in work_items:
            if mm_backend == "bmm":
                e_dev = int(w_gu.size(0))
                stacked_inputs, counts = self.mlpm._batched_pad_inputs_presorted(
                    x_dev_sorted, eid_dev_sub, e_dev
                )
                counts_cpu = counts.to(device="cpu", non_blocking=False)
                y_dev_sorted = self.mlpm.fused_experts_gate_up_down_bmm_from_padded(
                    stacked_inputs=stacked_inputs,
                    counts=counts_cpu,
                    gate_up_w_eh2i=w_gu,
                    down_w_eih=w_down,
                    act_fn=act_fn,
                )
            else:
                y_dev_sorted = self.mlpm.fused_experts_gate_up_down_mm_presorted(
                    x_slots_sorted=x_dev_sorted,
                    expert_ids_sorted=eid_dev_sub,
                    gate_up_w_eh2i=w_gu,
                    down_w_eih=w_down,
                    act_fn=act_fn,
                    mm_backend="gmm",
                )

            y_main = y_dev_sorted.to(main_device, non_blocking=True)
            out_weighted = y_main * w_dev_sorted
            expert_cache.scatter_reduce_(
                dim=0,
                index=tok_dev_sorted.view(-1, 1).expand(-1, expert_cache.size(-1)),
                src=out_weighted,
                reduce="sum",
            )

        cuda_hook_time_end("gpu_experts_multi_device_fused")

        # -----------------------
        # Step 7: wait CPU + scatter CPU outputs (weighted)
        # -----------------------
        cuda_hook_time("cpu_thread_manager_mp_wait_fused")
        if experts_cpu_list:
            pending = {rid for rid, _eids in cpu_submit_meta}
            results: dict[int, torch.Tensor] = {}
            while pending:
                res = self.cpu_thread_manager_mp.wait()
                if getattr(res, "request_id", -1) in pending:
                    pending.remove(res.request_id)
                    results[res.request_id] = res.final_hidden_states

            for rid, eids_sub in cpu_submit_meta:
                out_sub = results.get(rid)
                if out_sub is None:
                    continue
                w_parts_cpu = []
                token_parts_cpu = []
                for eid in eids_sub:
                    start_idx, end_idx = expert_indices_map[eid]
                    w_parts_cpu.append(slot_w_all[start_idx:end_idx])
                    token_parts_cpu.append(token_idxs[start_idx:end_idx])
                w_cpu = torch.cat(w_parts_cpu, dim=0) if w_parts_cpu else None
                tok_cpu = torch.cat(token_parts_cpu, dim=0) if token_parts_cpu else None
                if w_cpu is not None and tok_cpu is not None and int(tok_cpu.numel()) > 0:
                    out_weighted_cpu = out_sub * w_cpu
                    expert_cache.scatter_reduce_(
                        dim=0,
                        index=tok_cpu.view(-1, 1).expand(-1, expert_cache.size(-1)),
                        src=out_weighted_cpu,
                        reduce="sum",
                    )
        cuda_hook_time_end("cpu_thread_manager_mp_wait_fused")

        out = expert_cache.view(*orig_shape) + y
        cuda_hook_time_end(f"layer_moe_dgenerate_mp_multi_device_fused_l_{layer_idx+1}")
        return out


    def layer_moe_generate_mp_multi_device(self, layer_idx: int, hidden_states: torch.Tensor):
        cuda_hook_time(f"layer_moe_generate_mp_multi_device_l_{layer_idx+1}")
        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape

        cuda_hook_time("gate")
        topk_idx, topk_weight, aux_loss = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_idx, hidden_states)
        flat_expert_indices = topk_idx.view(-1)      # [batch_size * seq_len * num_experts_per_tok]
        flat_experts_weight = topk_weight.view(-1, 1)  # [batch_size * seq_len * num_experts_per_tok, 1]
        idxs = flat_expert_indices.argsort()         # 排序后的索引
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # [num_experts]
        token_idxs = idxs // self.mlpm.get_experts_per_tok()  # 恢复到原始 token 索引
        flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)  # [batch_size * seq_len, hidden_dim]
        cuda_hook_time_end("gate")

        num_device = len(self.device_list)
        
        cuda_hook_time("experts_map_get")
        # Step 7: 构建每个 expert 的索引信息（避免 tensor 计算和拷贝）
        expert_indices_map = {}  # {expert_id: (start_idx, end_idx)} 保存索引范围
        expert_token_indices_map = {}  # {expert_id: token_ids} 保存 token 索引
        expert_token_counts_list = []  # 用于 CPU/GPU 分配：[(expert_id, token_count), ...]
        
        num_experts = self.mlpm.get_experts_num()
        prev_end = 0  # 前一个 expert 的结束位置
        
        for expert_id in range(num_experts):
            if expert_id >= len(tokens_per_expert):
                break
            
            # tokens_per_expert[expert_id] 是累积和，表示到 expert_id 为止的总 token 数
            end_idx = int(tokens_per_expert[expert_id])
            
            # 如果 end_idx 等于 prev_end，说明该 expert 没有 token
            if end_idx == prev_end:
                continue
            
            start_idx = prev_end
            token_count = end_idx - start_idx  # 该 expert 的实际 token 数量
            
            expert_indices_map[expert_id] = (start_idx, end_idx)
            expert_token_indices_map[expert_id] = token_idxs[start_idx:end_idx]
            expert_token_counts_list.append((expert_id, token_count))
            
            prev_end = end_idx
        
        # Step 8: 根据token数量分配CPU/GPU experts，并将GPU专家平均分配到多个设备
        sorted_experts_by_load = sorted(expert_token_counts_list, key=lambda x: x[1])
        num_experts_total = len(sorted_experts_by_load)
        
        # 使用固定值
        num_experts_on_cpu = int(num_experts_total * self.num_experts_on_cpu_ratio)
        
        cpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[:num_experts_on_cpu])
        gpu_experts_list = sorted_experts_by_load[num_experts_on_cpu:]  # GPU 专家列表（按 token 数量排序）

        # cpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[num_experts_on_cpu:])
        # gpu_experts_list = sorted_experts_by_load[:num_experts_on_cpu]  # GPU 专家列表（按 token 数量排序）
        
        # 将 GPU 专家分配到多个设备
        # 使用 flag 控制是否基于显存分配，否则使用原来的按 token 数量分配逻辑
        use_memory_based_allocation = False  # 设置为 True 启用基于显存的分配
        
        device_ids = [int(device.split(":")[1]) for device in self.device_list]
        device_expert_map = {device_id: [] for device_id in device_ids}  # {device_id: [expert_id, ...]}
        device_token_counts = {device_id: 0 for device_id in device_ids}  # {device_id: token_count}
        
        if use_memory_based_allocation:
            # 基于显存的分配逻辑
            device_memory_used = {device_id: 0 for device_id in device_ids}  # {device_id: memory_used_in_bytes}
            
            # 先获取每个GPU的剩余显存
            device_free_memory = {}  # {device_id: free_memory_in_bytes}
            try:
                import pynvml
                pynvml.nvmlInit()
                
                for device_id in device_ids:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        device_free_memory[device_id] = memory_info.free
                        logger.debug(f"Device {device_id} free memory: {memory_info.free / (1024**3):.2f} GB")
                    except Exception as e:
                        logger.warning(f"Failed to get memory info for device {device_id} using nvml: {e}")
                        # 如果无法获取，使用一个默认大值，避免分配失败
                        device_free_memory[device_id] = 10 * 1024**3  # 默认10GB
            except ImportError:
                logger.warning("pynvml not available, using default memory allocation")
                # 如果nvml不可用，给每个设备分配默认值
                for device_id in device_ids:
                    device_free_memory[device_id] = 10 * 1024**3  # 默认10GB
            except Exception as e:
                logger.warning(f"Failed to initialize nvml: {e}")
                for device_id in device_ids:
                    device_free_memory[device_id] = 10 * 1024**3  # 默认10GB
            
            # 计算每个专家的大小
            from utils.helper import calculate_expert_memory_size
            expert_memory_map = {}  # {expert_id: memory_size_in_bytes}
            for expert_id, token_count in gpu_experts_list:
                try:
                    memory_size = calculate_expert_memory_size(
                        self.mlpm, self.cmv.tensor_index_resize_json, layer_idx, expert_id
                    )
                    expert_memory_map[expert_id] = memory_size
                except Exception as e:
                    logger.warning(f"Failed to calculate memory size for expert {expert_id}: {e}")
                    # 如果无法计算，使用一个估计值（例如100MB）
                    expert_memory_map[expert_id] = 100 * 1024 * 1024  # 默认100MB
            
            # 按显存大小从大到小排序，优先分配大显存的专家
            gpu_experts_with_memory = [
                (expert_id, token_count, expert_memory_map.get(expert_id, 0))
                for expert_id, token_count in gpu_experts_list
            ]
            gpu_experts_sorted = sorted(gpu_experts_with_memory, key=lambda x: x[2], reverse=True)
            
            # 使用贪心算法：按显存大小分配，每次分配给当前显存使用量最少的设备
            for expert_id, token_count, expert_memory in gpu_experts_sorted:
                # 找到当前显存使用量最少的设备（考虑剩余显存）
                # 使用剩余显存最多的设备，或者显存使用量最少的设备
                min_device_id = min(
                    device_ids, 
                    key=lambda d: (
                        device_memory_used[d],  # 优先选择显存使用量少的
                        -device_free_memory.get(d, 0)  # 其次选择剩余显存多的
                    )
                )
                
                # 检查是否有足够的显存
                if device_free_memory.get(min_device_id, 0) < expert_memory:
                    # 如果当前设备显存不足，尝试其他设备
                    available_devices = [
                        d for d in device_ids 
                        if device_free_memory.get(d, 0) >= expert_memory
                    ]
                    if available_devices:
                        min_device_id = min(
                            available_devices,
                            key=lambda d: device_memory_used[d]
                        )
                    else:
                        # 如果所有设备都没有足够显存，仍然分配到显存使用量最少的设备
                        logger.warning(
                            f"Expert {expert_id} requires {expert_memory / (1024**3):.2f} GB, "
                            f"but no device has enough free memory. Allocating to device {min_device_id} anyway."
                        )
                
                device_expert_map[min_device_id].append(expert_id)
                device_token_counts[min_device_id] += token_count
                device_memory_used[min_device_id] += expert_memory
                # 更新剩余显存（如果之前获取过）
                if min_device_id in device_free_memory:
                    device_free_memory[min_device_id] -= expert_memory
        else:
            # 原来的分配逻辑：按 token 数量分配
            # 使用贪心算法：按 token 数量从大到小排序，每次分配给当前 token 数量最少的设备
            gpu_experts_sorted = sorted(gpu_experts_list, key=lambda x: x[1], reverse=True)
            
            for expert_id, token_count in gpu_experts_sorted:
                # 找到当前 token 数量最少的设备（使用设备ID）
                min_device_id = min(device_ids, key=lambda device_id: device_token_counts[device_id])
                device_expert_map[min_device_id].append(expert_id)
                device_token_counts[min_device_id] += token_count
        
        # 构建每个设备的 expert ID 集合，使用设备ID作为键
        gpu_expert_ids_by_device = {
            device_id: set(expert_ids) 
            for device_id, expert_ids in device_expert_map.items()
        }
        
        # 打印调试信息
        cpu_ratio = num_experts_on_cpu / num_experts_total if num_experts_total > 0 else 0
        logger.debug(f"\nExpert Token Distribution & Multi-Device Allocation (MP):")
        logger.debug(f"  Total experts: {num_experts_total}")
        logger.debug(f"  CPU experts: {num_experts_on_cpu} ({cpu_ratio*100:.0f}%)")
        logger.debug(f"  GPU experts: {num_experts_total - num_experts_on_cpu} ({(1-cpu_ratio)*100:.0f}%)")
        logger.debug(f"  Number of GPU devices: {num_device}")
        logger.debug(f"\n  Expert ID | Tokens | Device")
        logger.debug(f"  {'-'*35}")
        
        total_tokens_cpu = sum(count for _, count in sorted_experts_by_load[:num_experts_on_cpu])
        total_tokens_gpu = sum(count for _, count in sorted_experts_by_load[num_experts_on_cpu:])
        
        for expert_id, token_count in sorted_experts_by_load:
            if expert_id in cpu_expert_ids:
                device = "CPU"
            else:
                # 找到该 expert 所在的设备
                device = None
                for device_id, expert_set in gpu_expert_ids_by_device.items():
                    if expert_id in expert_set:
                        device = f"GPU{device_id}({self.device_list[device_ids.index(device_id)]})"
                        break
                if device is None:
                    device = "Unknown"
            logger.debug(f"  Expert {expert_id:2d} | {token_count:6d} | {device}")
        
        logger.debug(f"\n  Device Token Distribution:")
        logger.debug(f"  CPU: {total_tokens_cpu:6d} tokens")
        for device_id in device_ids:
            device_tokens = device_token_counts[device_id]
            device_name = self.device_list[device_ids.index(device_id)]
            if use_memory_based_allocation:
                # 显示显存信息
                memory_used_gb = device_memory_used[device_id] / (1024**3)
                memory_free_gb = device_free_memory.get(device_id, 0) / (1024**3)
                logger.debug(
                    f"  {device_name}: {device_tokens:6d} tokens ({len(device_expert_map[device_id])} experts), "
                    f"Memory: {memory_used_gb:.2f} GB used, {memory_free_gb:.2f} GB free"
                )
            else:
                # 只显示token和expert数量
                logger.debug(
                    f"  {device_name}: {device_tokens:6d} tokens ({len(device_expert_map[device_id])} experts)"
                )
        logger.debug(f"  Total GPU: {total_tokens_gpu:6d} tokens")
        logger.debug(f"{'='*60}\n")
        
        cuda_hook_time_end("experts_map_get")

        expert_cache = torch.zeros_like(flat_hidden_states)
        
        # Step 9: 提交CPU专家执行
        cuda_hook_time("cpu_experts_submit")
        if cpu_expert_ids:
            cpu_expert_idx_list = sorted(cpu_expert_ids)
            logger.debug(f"\n  Computing {len(cpu_expert_ids)} experts on CPU MP...")
            self.cpu_thread_manager_mp.submit_worker(
                worker_idx=0,
                layer_idx=layer_idx,
                expert_idx_list=cpu_expert_idx_list,
                expert_indices_map={eid: expert_indices_map[eid] for eid in cpu_expert_idx_list},
                flat_hidden_states=flat_hidden_states,
                idxs=idxs,
            )
        cuda_hook_time_end("cpu_experts_submit")

        # Step 10: 为每个GPU设备分配和加载专家
        cuda_hook_time("allocate_experts_cuda_memory_and_restore_model_multi_device")
        
        # 构建 tensor_index_names_device_map: {device_id: [expert_names, ...]}
        tensor_index_names_device_map = {}
        for device_id in device_ids:
            device_expert_ids = gpu_expert_ids_by_device[device_id]
            if device_expert_ids:
                gpu_expert_names = self.mlpm.get_experts_names(
                    layer_idx=layer_idx, 
                    expert_idx_list=list(device_expert_ids)
                )
                tensor_index_names_device_map[device_id] = gpu_expert_names
        
        # 一次性为所有设备分配内存和加载专家
        ret, replica_uuid, state_dict = \
            self.cmv.allocate_cuda_memory_and_load_into_gpu_multi_device(
                tensor_index_names_device_map=tensor_index_names_device_map
            )
        # 恢复模型状态到对应设备
        self.cmv.restore2model(state_dict, self.cmv.mlpm_ci)
        
        cuda_hook_time_end("allocate_experts_cuda_memory_and_restore_model_multi_device")

        # Step 11: 执行shared experts（在第一个设备上）
        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            hidden_states=hidden_states,
        )
        cuda_hook_time_end("gpu_sexperts")

        # Step 12: 等待load_qkvogn_s加载完成
        if layer_idx < self.mlpm.config.num_hidden_layers-1:
            cuda_hook_time("wait_load_qkvogn_s_weight")
            self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
            cuda_hook_time_end("wait_load_qkvogn_s_weight")

        # Step 16: 为每个设备准备数据并执行GPU experts（使用 multi_list 版本）
        cuda_hook_time("gpu_experts_multi_device_prepare")
        
        # 为每个设备准备数据
        group_w1_list_map = {}
        group_w2_list_map = {}
        group_w3_list_map = {}
        stacked_inputs_map = {}
        expert_idx_list_map = {}
        flat_hidden_states_map = {}
        flat_experts_weight_map = {}
        idxs_map = {}
        all_expert_weights_map = {}
        all_token_ids_map = {}
        
        for device_id in device_ids:
            device = self.device_list[device_ids.index(device_id)]
            device_expert_ids = gpu_expert_ids_by_device[device_id]
            
            if device_expert_ids:
                device_expert_idx_list = list(device_expert_ids)
                
                # 准备设备上的数据
                if device_id != flat_hidden_states.device.index:
                    device_idxs = idxs.to(device_id)
                    device_flat_experts_weight = flat_experts_weight.to(device_id)
                    device_flat_hidden_states = flat_hidden_states.to(device_id)
                else:
                    device_idxs = idxs
                    device_flat_experts_weight = flat_experts_weight
                    device_flat_hidden_states = flat_hidden_states
                
                # 准备 stacked_inputs
                device_stacked_inputs = self.mlpm.experts_func_mgpu_group_pad(
                    expert_idx_list=device_expert_idx_list,
                    expert_indices_map=expert_indices_map,
                    device_flat_hidden_states=device_flat_hidden_states,
                    device_idxs=device_idxs,
                )
                
                # 准备 group weights
                group_w1_list, group_w2_list, group_w3_list = self.mlpm.experts_func_mgpu_group_list(
                    mi=self.cmv.mlpm_ci,
                    layer_idx=layer_idx,
                    expert_idx_list=device_expert_idx_list
                )
                
                # 提前收集 weights 和 token_ids
                device_token_idxs = device_idxs // self.mlpm.get_experts_per_tok()
                device_all_expert_weights_list = []
                device_all_token_ids_list = []
                for i, expert_idx in enumerate(device_expert_idx_list):
                    start_idx, end_idx = expert_indices_map[expert_idx]
                    token_ids = device_token_idxs[start_idx:end_idx]
                    expert_weights = device_flat_experts_weight[device_idxs[start_idx:end_idx]]
                    device_all_expert_weights_list.append(expert_weights)
                    device_all_token_ids_list.append(token_ids)
                
                # 将列表 concat 为 tensor
                device_all_expert_weights_concat = torch.cat(device_all_expert_weights_list, dim=0) if device_all_expert_weights_list else None
                device_all_token_ids_concat = torch.cat(device_all_token_ids_list, dim=0) if device_all_token_ids_list else None
                
                # 存储到 map 中
                group_w1_list_map[device_id] = group_w1_list
                group_w2_list_map[device_id] = group_w2_list
                group_w3_list_map[device_id] = group_w3_list
                stacked_inputs_map[device_id] = device_stacked_inputs
                expert_idx_list_map[device_id] = device_expert_idx_list
                flat_hidden_states_map[device_id] = device_flat_hidden_states
                flat_experts_weight_map[device_id] = device_flat_experts_weight
                idxs_map[device_id] = device_idxs
                all_expert_weights_map[device_id] = device_all_expert_weights_concat
                all_token_ids_map[device_id] = device_all_token_ids_concat
        cuda_hook_time_end("gpu_experts_multi_device_prepare")
        # Step 15: 等待所有设备的专家加载完成
        cuda_hook_time("wait_experts_multi_device")
        self.cmv.wait_load_into_gpu(replica_uuid)
        cuda_hook_time_end("wait_experts_multi_device")

        # Step 13: 等待CPU experts完成
        cuda_hook_time("cpu_thread_manager_mp_wait")
        output_cpu2gpu = None
        if cpu_expert_ids:
            _res = self.cpu_thread_manager_mp.wait()
            output_cpu2gpu = _res.final_hidden_states
        cuda_hook_time_end("cpu_thread_manager_mp_wait")
        
        # Step 14: 处理CPU experts的输出
        if output_cpu2gpu is not None:
            cuda_hook_time("cpuoutputsdeal")
            acpu_expert_outs_slices = []
            acpu_expert_weights = []
            acpu_token_ids = []
            cpu_expert_idx_list = sorted(cpu_expert_ids)
            for i, expert_idx in enumerate(cpu_expert_idx_list):
                token_ids = expert_token_indices_map[expert_idx]
                num_tokens = token_ids.shape[0]
                
                # 收集对应的 weights
                start_idx, end_idx = expert_indices_map[expert_idx]
                expert_weights = flat_experts_weight[idxs[start_idx:end_idx]]
                acpu_expert_weights.append(expert_weights)
                acpu_token_ids.append(token_ids)
            
            # for i, expert_idx in enumerate(list(cpu_expert_ids)):
            #     token_ids = expert_token_indices_map[expert_idx]
            #     num_tokens = token_ids.shape[0]
            #     expert_out = output_cpu2gpu[i][:num_tokens]
            #     acpu_expert_outs_slices.append(expert_out)
            # concat_expert_out = torch.cat(acpu_expert_outs_slices, dim=0)
            concat_expert_out = output_cpu2gpu
            concat_expert_weights = torch.cat(acpu_expert_weights, dim=0)  # [total_tokens, 1]
            concat_token_ids = torch.cat(acpu_token_ids, dim=0)  # [total_tokens]
            concat_expert_out = concat_expert_out.mul_(concat_expert_weights)
            index = concat_token_ids.view(-1, 1).expand(-1, expert_cache.shape[-1])
            expert_cache.scatter_reduce_(
                dim=0,
                index=index,
                src=concat_expert_out,
                reduce='sum',
            )
            del output_cpu2gpu, concat_expert_out
            cuda_hook_time_end("cpuoutputsdeal")

        # # Step 15: 等待所有设备的专家加载完成
        # cuda_hook_time("wait_experts_multi_device")
        # self.cmv.wait_load_into_gpu(replica_uuid)
        # cuda_hook_time_end("wait_experts_multi_device")

        # 调用 multi_list 版本执行多GPU并行计算
        _ = self.mlpm.experts_func_mgpu_einsum_mp_multi_list(
            layer_idx=layer_idx,
            group_w1_list_map=group_w1_list_map,
            group_w2_list_map=group_w2_list_map,
            group_w3_list_map=group_w3_list_map,
            stacked_inputs_map=stacked_inputs_map,
            expert_idx_list_map=expert_idx_list_map,
            expert_indices_map=expert_indices_map,
            flat_hidden_states_map=flat_hidden_states_map,
            flat_experts_weight_map=flat_experts_weight_map,
            idxs_map=idxs_map,
            final_hidden_states=expert_cache,
            all_expert_weights_map=all_expert_weights_map,
            all_token_ids_map=all_token_ids_map,
            expert_token_indices_map=expert_token_indices_map,
        )

        # Step 17: 合并结果
        layer_output = expert_cache.view(*orig_shape) + y

        cuda_hook_time_end(f"layer_moe_generate_mp_multi_device_l_{layer_idx+1}")
        return layer_output

    def layer_moe_dgenerate_mp_multi_device(self, layer_idx: int, hidden_states: torch.Tensor):
        """多进程多设备版本的decode阶段MoE层生成，基于实际设备位置分配专家，支持多GPU"""
        cuda_hook_time(f"layer_moe_dgenerate_mp_multi_device_l_{layer_idx+1}")
        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape

        cuda_hook_time("gate")
        topk_idx, topk_weight, aux_loss = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_idx, hidden_states)

        flat_expert_indices = topk_idx.view(-1)      # [batch_size * seq_len * num_experts_per_tok]
        flat_experts_weight = topk_weight.view(-1, 1)  # [batch_size * seq_len * num_experts_per_tok, 1]
        idxs = flat_expert_indices.argsort()         # 排序后的索引
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # [num_experts]
        token_idxs = idxs // self.mlpm.get_experts_per_tok()  # 恢复到原始 token 索引
        flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)  # [batch_size * seq_len, hidden_dim]
        cuda_hook_time_end("gate")

        num_device = len(self.device_list)
        
        cuda_hook_time("experts_map_get")
        # Step 7: 构建每个 expert 的索引信息（避免 tensor 计算和拷贝）
        expert_indices_map = {}  # {expert_id: (start_idx, end_idx)} 保存索引范围
        expert_token_indices_map = {}  # {expert_id: token_ids} 保存 token 索引
        expert_token_counts_list = []  # 用于 CPU/GPU 分配：[(expert_id, token_count), ...]
        
        num_experts = self.mlpm.get_experts_num()
        prev_end = 0  # 前一个 expert 的结束位置
        
        for expert_id in range(num_experts):
            if expert_id >= len(tokens_per_expert):
                break
            
            # tokens_per_expert[expert_id] 是累积和，表示到 expert_id 为止的总 token 数
            end_idx = int(tokens_per_expert[expert_id])
            
            # 如果 end_idx 等于 prev_end，说明该 expert 没有 token
            if end_idx == prev_end:
                continue
            
            start_idx = prev_end
            token_count = end_idx - start_idx  # 该 expert 的实际 token 数量
            
            expert_indices_map[expert_id] = (start_idx, end_idx)
            expert_token_indices_map[expert_id] = token_idxs[start_idx:end_idx]
            expert_token_counts_list.append((expert_id, token_count))
            
            prev_end = end_idx
        
        # Step 8: 根据实际设备位置分配CPU/GPU experts，支持多GPU
        sorted_experts_by_load = sorted(expert_token_counts_list, key=lambda x: x[1])
        num_experts_total = len(sorted_experts_by_load)
        
        # 获取每个 expert 的实际设备位置
        layer = self.cmv.mlpm_ci.model.layers[layer_idx]
        expert_actual_device_map = get_expert_device_distribution(layer)
        
        # 按实际设备位置分组：CPU专家和各个GPU设备的专家
        experts_cpu_list = []
        gpu_expert_ids_by_device = {i: [] for i in range(num_device)}  # {device_idx: [expert_id, ...]}
        
        for expert_id, _ in sorted_experts_by_load:
            actual_device = expert_actual_device_map.get(expert_id, "unknown")
            
            # 检查是否在某个GPU设备上
            found_gpu_device = False
            for device_idx in range(num_device):
                device_str = str(self.device_list[device_idx])
                if actual_device == device_str:
                    gpu_expert_ids_by_device[device_idx].append(expert_id)
                    found_gpu_device = True
                    break
            
            # 如果不在任何GPU设备上，则认为是CPU专家
            if not found_gpu_device:
                experts_cpu_list.append(expert_id)
        
        logger.debug(f"\nLayer {layer_idx} Expert Device Distribution (Multi-Device MP):")
        logger.debug(f"  Active experts: {num_experts_total} (out of {num_experts} total)")
        logger.debug(f"\n  Detailed Expert Distribution:")
        logger.debug(f"  {'Expert ID':<10} | {'Tokens':<10} | {'Actual Device':<15}")
        logger.debug(f"  {'-'*70}")
        for expert_id, token_count in sorted_experts_by_load:
            actual_device = expert_actual_device_map.get(expert_id, "unknown")
            logger.debug(f"  {expert_id:<10} | {token_count:<10} |  {actual_device:<15}")
        logger.debug(f"{'='*60}\n")
        
        logger.debug(f"experts_cpu_list: {experts_cpu_list} num: {len(experts_cpu_list)}")
        for device_idx in range(num_device):
            device_expert_ids = gpu_expert_ids_by_device[device_idx]
            logger.debug(f"experts_gpu_list[{device_idx}]({self.device_list[device_idx]}): {device_expert_ids} num: {len(device_expert_ids)}")
        logger.debug(f"expert_actual_device_map {expert_actual_device_map}")
        
        cuda_hook_time_end("experts_map_get")

        expert_cache = torch.zeros_like(flat_hidden_states)
        
        # Step 9: 提交CPU专家执行（使用多进程版本）
        cuda_hook_time("cpu_experts_submit")
        if experts_cpu_list:
            logger.debug(f"\n  Computing {len(experts_cpu_list)} experts on CPU MP...")
            self.cpu_thread_manager_mp.submit_worker(
                worker_idx=0,
                layer_idx=layer_idx,
                expert_idx_list=experts_cpu_list,
                expert_indices_map={eid: expert_indices_map[eid] for eid in experts_cpu_list},
                flat_hidden_states=flat_hidden_states,
                idxs=idxs,
            )
        cuda_hook_time_end("cpu_experts_submit")

        # Step 11: 执行shared experts（在第一个设备上）
        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            hidden_states=hidden_states,
        )
        cuda_hook_time_end("gpu_sexperts")

        # Step 13: 为每个设备准备数据并执行GPU experts（使用 multi_list 版本）
        cuda_hook_time("gpu_experts_multi_device_prepare")
        
        # 为每个设备准备数据（仿照 1760-1785 行的方式）
        group_w1_list_map = {}
        group_w2_list_map = {}
        group_w3_list_map = {}
        stacked_inputs_map = {}
        expert_idx_list_map = {}
        flat_hidden_states_map = {}
        flat_experts_weight_map = {}
        idxs_map = {}
        all_expert_weights_map = {}
        all_token_ids_map = {}
        
        device_ids = [int(device.split(":")[1]) for device in self.device_list]
        for device_id in device_ids:
            device_idx = device_ids.index(device_id)
            device_expert_ids = gpu_expert_ids_by_device[device_idx]
            
            if device_expert_ids:
                device_expert_idx_list = list(device_expert_ids)
                
                # 准备设备上的数据（类似 2087-2096 行）
                if device_id != flat_hidden_states.device.index:
                    device_idxs = idxs.to(device_id)
                    device_flat_experts_weight = flat_experts_weight.to(device_id)
                    device_flat_hidden_states = flat_hidden_states.to(device_id)
                else:
                    device_idxs = idxs
                    device_flat_experts_weight = flat_experts_weight
                    device_flat_hidden_states = flat_hidden_states
                
                # 准备 stacked_inputs
                device_stacked_inputs = self.mlpm.experts_func_mgpu_group_pad(
                    expert_idx_list=device_expert_idx_list,
                    expert_indices_map=expert_indices_map,
                    device_flat_hidden_states=device_flat_hidden_states,
                    device_idxs=device_idxs,
                )
                
                 # 准备 group weights
                group_w1_list, group_w2_list, group_w3_list = self.mlpm.experts_func_mgpu_group_list(
                    mi=self.cmv.mlpm_ci,
                    layer_idx=layer_idx,
                    expert_idx_list=device_expert_idx_list
                )
                
                # 提前收集 weights 和 token_ids（使用设备上的数据）
                device_token_idxs = device_idxs // self.mlpm.get_experts_per_tok()
                device_expert_weights_list = []
                device_token_ids_list = []
                for i, expert_idx in enumerate(device_expert_idx_list):
                    start_idx, end_idx = expert_indices_map[expert_idx]
                    token_ids = device_token_idxs[start_idx:end_idx]
                    expert_weights = device_flat_experts_weight[device_idxs[start_idx:end_idx]]
                    device_expert_weights_list.append(expert_weights)
                    device_token_ids_list.append(token_ids)
                
               
                
                # 将列表 concat 为 tensor
                device_expert_weights_concat = torch.cat(device_expert_weights_list, dim=0) if device_expert_weights_list else None
                device_token_ids_concat = torch.cat(device_token_ids_list, dim=0) if device_token_ids_list else None
                
                # 存储到 map 中
                group_w1_list_map[device_id] = group_w1_list
                group_w2_list_map[device_id] = group_w2_list
                group_w3_list_map[device_id] = group_w3_list
                stacked_inputs_map[device_id] = device_stacked_inputs
                expert_idx_list_map[device_id] = device_expert_idx_list
                flat_hidden_states_map[device_id] = device_flat_hidden_states
                flat_experts_weight_map[device_id] = device_flat_experts_weight
                idxs_map[device_id] = device_idxs
                all_expert_weights_map[device_id] = device_expert_weights_concat
                all_token_ids_map[device_id] = device_token_ids_concat
        cuda_hook_time_end("gpu_experts_multi_device_prepare")

        # Step 14: 等待CPU experts完成
        cuda_hook_time("cpu_thread_manager_mp_wait")
        output_cpu2gpu = None
        if experts_cpu_list:
            _res = self.cpu_thread_manager_mp.wait()
            output_cpu2gpu = _res.final_hidden_states
        cuda_hook_time_end("cpu_thread_manager_mp_wait")
        
        # Step 16: 处理CPU experts的输出
        if output_cpu2gpu is not None:
            cuda_hook_time("cpuoutputsdeal")
            acpu_expert_weights = []
            acpu_token_ids = []
            for i, expert_idx in enumerate(experts_cpu_list):
                token_ids = expert_token_indices_map[expert_idx]
                
                # 收集对应的 weights
                start_idx, end_idx = expert_indices_map[expert_idx]
                expert_weights = flat_experts_weight[idxs[start_idx:end_idx]]
                acpu_expert_weights.append(expert_weights)
                acpu_token_ids.append(token_ids)
            
            concat_expert_out = output_cpu2gpu
            concat_expert_weights = torch.cat(acpu_expert_weights, dim=0)  # [total_tokens, 1]
            concat_token_ids = torch.cat(acpu_token_ids, dim=0)  # [total_tokens]
            concat_expert_out = concat_expert_out.mul_(concat_expert_weights)
            cuda_hook_time("index_scatter")
            index = concat_token_ids.view(-1, 1).expand(-1, expert_cache.shape[-1])
            expert_cache.scatter_reduce_(
                dim=0,
                index=index,
                src=concat_expert_out,
                reduce='sum',
            )
            cuda_hook_time_end("index_scatter")
            del output_cpu2gpu, concat_expert_out
            cuda_hook_time_end("cpuoutputsdeal")

        cuda_hook_time("gpu_experts_multi_device_submit")
        # 调用 multi_list 版本执行多GPU并行计算
        _ = self.mlpm.experts_func_mgpu_einsum_mp_multi_list(
            layer_idx=layer_idx,
            group_w1_list_map=group_w1_list_map,
            group_w2_list_map=group_w2_list_map,
            group_w3_list_map=group_w3_list_map,
            stacked_inputs_map=stacked_inputs_map,
            expert_idx_list_map=expert_idx_list_map,
            expert_indices_map=expert_indices_map,
            flat_hidden_states_map=flat_hidden_states_map,
            flat_experts_weight_map=flat_experts_weight_map,
            idxs_map=idxs_map,
            final_hidden_states=expert_cache,
            all_expert_weights_map=all_expert_weights_map,
            all_token_ids_map=all_token_ids_map,
            expert_token_indices_map=expert_token_indices_map,
        )
        cuda_hook_time_end("gpu_experts_multi_device_submit")

        # Step 17: 合并结果
        layer_output = expert_cache.view(*orig_shape) + y

        cuda_hook_time_end(f"layer_moe_dgenerate_mp_multi_device_l_{layer_idx+1}")
        return layer_output
    

    def read_tensor_index_json(self, model_path: str):
        tensor_index_json_path = os.path.join(model_path, "tensor_index.json")
        with open(tensor_index_json_path, "r") as f:
            tensor_index_json = json.load(f)
        return tensor_index_json

    def test_tensor_index_locate(self, tensor_index_json: dict):
        """
        根据当前 GPU 数量与可用显存，为 tensor_index 中的权重分配设备。

        约束：
        1) 同一 (layer, expert) 下的所有参数必须放在同一个设备。
        2) 同一 layer 的 experts 在设备间尽量均匀（按 expert 组数量）。
        3) 分配时尽量不超过设备可用显存；若都放不下则退化为剩余显存最多设备。

        Returns:
            dict[str, str]: tensor_name -> "cuda:<id>"
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
                "pynvml is required for test_tensor_index_locate but is not installed"
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
        # 1) 专家参数：同 layer 同 expert 的所有参数归为一组，跨设备均衡分配；
        # 2) 非专家参数（包括 self_attn 等）：全部固定放在第一个设备。
        expert_group_tensors: dict[tuple[int, int], list[str]] = defaultdict(list)
        expert_group_sizes: dict[tuple[int, int], int] = defaultdict(int)
        non_expert_tensors: list[tuple[str, int]] = []

        for tname, tmeta in tensor_index_json.items():
            tsize = _tensor_size_bytes(tmeta)
            key = self.mlpm.get_tensor_expert_group_key(tname)
            if key is not None:
                expert_group_tensors[key].append(tname)
                expert_group_sizes[key] += tsize
            else:
                non_expert_tensors.append((tname, tsize))

        tensor_to_device: dict[str, str] = {}
        first_device = device_names[0]

        # 先预留：非专家参数全部放在第一个设备（含 self_attn / embed / norm 等）。
        non_expert_total_size = sum(tsize for _, tsize in non_expert_tensors)
        if device_remaining[first_device] < non_expert_total_size:
            raise RuntimeError(
                f"[test_tensor_index_locate] insufficient GPU memory on first device "
                f"{first_device} for non-expert tensors total={non_expert_total_size} bytes, "
                f"remaining={device_remaining[first_device]} bytes"
            )
        for tname, tsize in non_expert_tensors:
            tensor_to_device[tname] = first_device
            device_remaining[first_device] -= tsize

        # 再按 layer 分配专家组：
        # 在满足可容纳前提下，优先最小化各卡“已用显存占初始可用显存”的离散度，
        # 使整体 GPU 显存使用尽可能均匀。
        layer_to_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for key in expert_group_tensors:
            layer_to_groups[key[0]].append(key)

        for layer_id, groups in sorted(layer_to_groups.items(), key=lambda x: x[0]):
            # 保留同层专家计数作为次级 tie-break（主目标是显存均衡）。
            per_layer_device_counts = {d: 0 for d in device_names}
            # 大组先放，降低后续碎片
            groups_sorted = sorted(groups, key=lambda g: expert_group_sizes[g], reverse=True)

            for g in groups_sorted:
                gsize = expert_group_sizes[g]

                def _cand_key(dev_name: str):
                    rem = device_remaining[dev_name]
                    fits = 0 if rem >= gsize else 1  # 必须可放下
                    initial = max(1, device_initial_free[dev_name])
                    # 放置前后该卡使用率
                    before_used_ratio = (initial - rem) / initial
                    after_used_ratio = (initial - (rem - gsize)) / initial if rem >= gsize else float("inf")
                    # 主目标：让“放置后”的使用率尽量低，从而拉齐各卡水位。
                    return (
                        fits,
                        after_used_ratio,
                        before_used_ratio,
                        per_layer_device_counts[dev_name],  # 次级：同层专家数尽量均衡
                        -rem,
                        dev_name,
                    )

                chosen = min(device_names, key=_cand_key)
                if device_remaining[chosen] < gsize:
                    raise RuntimeError(
                        f"[test_tensor_index_locate] insufficient GPU memory for "
                        f"layer={layer_id} expert={g[1]} size={gsize} bytes, "
                        f"remaining={device_remaining}"
                    )
                per_layer_device_counts[chosen] += 1
                device_remaining[chosen] -= gsize
                for tname in expert_group_tensors[g]:
                    tensor_to_device[tname] = chosen

        device_usage_ratio = {}
        for d in device_names:
            initial = max(1, device_initial_free[d])
            device_usage_ratio[d] = (initial - device_remaining[d]) / initial
        logger.info(
            "[test_tensor_index_locate] tensors=%d expert_groups=%d first_device=%s "
            "remaining=%s usage_ratio=%s",
            len(tensor_to_device),
            len(expert_group_tensors),
            first_device,
            {d: device_remaining[d] for d in device_names},
            device_usage_ratio,
        )
        return tensor_to_device

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

    @torch.no_grad()
    def test_mp_process(self, num_iters: int = 3, warmup: int = 1):
        """
        性能探针：在 ``HostMemoryView._test_group_bmm_fused_experts_impl`` 同款设定下（bsz/seq、半套专家），
        通过 ``cpu_thread_manager_mp.submit_worker(..., use_bmm=True)`` + ``wait()`` 测量子进程
        ``bmm_with_group_tensors_mp_1`` 的端到端耗时（不含 gate 与构图，仅 MP 往返 + worker 内计算）。

        须先调用 ``init_mp_process()``。
        """
        import time as time_mod

        if not getattr(self, "cpu_thread_manager_mp", None):
            raise RuntimeError("cpu_thread_manager_mp is not initialized; call init_mp_process() first.")

        device = self.device1
        bsz, seq_len = 1, 128
        layer_idx = int(self.mlpm.get_first_k_dense_replace())
        hidden_size = int(getattr(self.mlpm.config, "hidden_size"))
        dtype = getattr(self.mlpm.config, "torch_dtype", torch.bfloat16)
        if dtype is None:
            dtype = torch.bfloat16

        # 加载 general weights 和 gate weights
        cuda_hook_time("init_general_weights")
        self.cmv.load_general_and_init()
        # only support one first replace dense layer
        self.cmv.init_load_qkvogn_es_weight(layer_idx=0)
        cuda_hook_time_end("init_general_weights")

        x = torch.randn((bsz, seq_len, hidden_size), device=device, dtype=dtype)
        topk_idx, topk_weight, _ = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_idx, x)
        flat_expert_indices = topk_idx.reshape(-1).to(torch.int64)
        idxs = flat_expert_indices.argsort().to(device)
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        flat_hidden_states = x.reshape(bsz * seq_len, hidden_size)

        num_experts = int(self.mlpm.get_experts_num())
        expert_indices_map: dict[int, tuple[int, int]] = {}
        prev_end = 0
        for expert_id in range(num_experts):
            if expert_id >= len(tokens_per_expert):
                break
            end_idx = int(tokens_per_expert[expert_id])
            if end_idx == prev_end:
                continue
            start_idx = prev_end
            expert_indices_map[expert_id] = (start_idx, end_idx)
            prev_end = end_idx

        expert_ids_top1 = topk_idx[:, 0].to(torch.int64)
        num_experts_total = int(self.mlpm.get_experts_num())
        half_e = max(1, num_experts_total // 2)
        activated_expert_ids = sorted(expert_indices_map.keys())
        n_activated = len(activated_expert_ids)
        if n_activated <= 128:
            ids_repr = str(activated_expert_ids)
        else:
            ids_repr = f"{activated_expert_ids[:64]} ... (+{n_activated - 64} more)"
        logger.info(
            "[test_mp_process] experts: total=%d activated=%d (this batch, any top-k slot) half_e=%d ids=%s",
            num_experts_total,
            n_activated,
            half_e,
            ids_repr,
        )

        for half_mode in _group_fused_test_parse_half_modes():
            expert_half_list, window_tokens, max_tok = _group_fused_select_half_experts(
                expert_ids_top1, num_experts_total, half_e, half_mode
            )
            expert_idx_list = [eid for eid in expert_half_list if eid in expert_indices_map]
            if not expert_idx_list:
                logger.warning(
                    "[test_mp_process] skip half_mode=%s: no experts in half have routed tokens.",
                    half_mode,
                )
                continue
            e_map = {eid: expert_indices_map[eid] for eid in expert_idx_list}

            flat_hidden_states_cpu_pin = gpinpool.alloc_same_pin_tensor(flat_hidden_states)
            flat_hidden_states_cpu_pin.copy_(flat_hidden_states, non_blocking=False)
            idxs_cpu = idxs.to(dtype=torch.int64, device="cpu", non_blocking=False)

            times_ms: list[float] = []
            n_workers = int(getattr(self.cpu_thread_manager_mp, "num_workers", 1))
            buckets = self._mp_partition_experts_across_cpu_workers(
                cpu_expert_idx_list=expert_idx_list,
                expert_indices_map=e_map,
                num_workers=n_workers,
            )
            for _ in range(num_iters):
                torch.cuda.synchronize(device)
                t0 = time_mod.perf_counter()
                pending = set()
                outs = []
                for wid, eids_sub in enumerate(buckets):
                    if not eids_sub:
                        continue
                    rid = self._mp_next_cpu_request_id()
                    pending.add(rid)
                    self.cpu_thread_manager_mp.submit_worker(
                        worker_idx=wid,
                        layer_idx=layer_idx,
                        expert_idx_list=sorted(eids_sub),
                        expert_indices_map={eid: e_map[eid] for eid in eids_sub},
                        flat_hidden_states=flat_hidden_states_cpu_pin,
                        idxs=idxs_cpu,
                        use_bmm=True,
                        request_id=rid,
                        device="cpu",
                    )
                while pending:
                    res = self.cpu_thread_manager_mp.wait()
                    rid = getattr(res, "request_id", -1)
                    if rid in pending:
                        pending.remove(rid)
                        if res.final_hidden_states is not None:
                            outs.append(res.final_hidden_states)
                torch.cuda.synchronize(device)
                t1 = time_mod.perf_counter()
                times_ms.append((t1 - t0) * 1e3)
                for out in outs:
                    del out

            mean_ms = sum(times_ms) / max(len(times_ms), 1)
            logger.info(
                "[test_mp_process] half_mode=%s layer=%d E_half=%d experts=%s "
                "window_tokens=%d max_tok_pad=%d submit+wait mean=%.3f ms (iters=%d) "
                "flat=(%d,%d) device=%s",
                half_mode,
                layer_idx,
                half_e,
                expert_idx_list,
                window_tokens,
                max_tok,
                mean_ms,
                num_iters,
                flat_hidden_states.shape[0],
                flat_hidden_states.shape[1],
                device,
            )