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
        
        

        cuda_hook_time("init_meta")
        self.cmv = CudaMemoryView(self.mlpm, device_list)
        self.cmv.start_init_meta_model()

        empty_model = copy.deepcopy(self.cmv.mlpm_ci)
        self.hmv = HostMemoryView(self.mlpm, empty_model=empty_model)
        cuda_hook_time_end("init_meta")

    
        # cetm = CETM(self.mlpm, self.hmv, num_workers=1)  # 默认1个worker，可通过参数调整
        # self.cetm = cetm
        # self.cetm.start()  # 启动工作线程

        sllmtm = SLLMTM(num_workers=1)  # SLLM 线程管理器，用于异步加载
        self.sllmtm = sllmtm
        self.sllmtm.start()  # 启动工作线程

        # imm = InitMetaManager()
        # imm.start()
        # self.imm = imm

        self.cmv.sllmtm = sllmtm     # 将sllmtm绑定到cmv中  
        # self.cmv.imm = imm
        
        # CPU专家数量：使用固定值 0.5 * total
        # stream = torch.cuda.Stream(device=device1)
        # with torch.cuda.stream(stream):
    def free_cmv(self):
        # 释放gpu分配的资源
        self.cmv.free_allocated()
        if hasattr(self, "cpu_thread_manager_mp"): 
            if self.cpu_thread_manager_mp is not None:
                self.cpu_thread_manager_mp.submit(-1, [], [], [], [])
                self.cpu_thread_manager_mp.wait()

    def init_mp_process(self):
        from lmp.cpu_thread_manager_mp import CPUExpertsManagerMP
        from lmp.device_mp import DeviceMP
        cuda_hook_time("init_mp_process")
        self.cpu_thread_manager_mp = CPUExpertsManagerMP(num_workers=1, model_path=self.mlpm.model_path, model_name_type=self.mlpm.model_name_type)
        self.cpu_thread_manager_mp.start()
        self.cpu_thread_manager_mp.wait_worker_bootstrap_ready()

        # self.imm_mp = InitMetaManagerMPShared(num_processes=1)
        # self.imm_mp.start()
        # self.dp = DeviceMP(num_processes=len(self.device_list))
        # self.dp.start()
        cuda_hook_time_end("init_mp_process")
    def mp_stop(self):
        self.cpu_thread_manager_mp.stop()

    @torch.no_grad()
    def test_mp_generate_multi_device_layer(self):
        cuda_hook_time("generate_input_ids")
        # 32, 64, 128
        batch_size = 128
        seq_len = 64
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

        # cuda_hook_time("copy_emodel")
        # model_cpy = copy.deepcopy(self.cmv.mlpm_ci)
        # cuda_hook_time_end("copy_emodel")

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

        # for bmm test
        # cuda_hook_time("load_all_qkvogn_s")
        # for layer_idx in range(0, self.mlpm.config.num_hidden_layers):
        #     if layer_idx < self.mlpm.config.num_hidden_layers-1:

        #         cuda_hook_time(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")
        #         self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1, device=self.device1)
        #         cuda_hook_time_end(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")

        #         cuda_hook_time("wait_load_qkvogn_s_weight")
        #         self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
        #         cuda_hook_time_end("wait_load_qkvogn_s_weight")
        # cuda_hook_time_end("load_all_qkvogn_s")


        # self.cmv.async_load_experts_decode_cpu_weight_multi_device()
        # self.cmv.async_wait_layer_loaded_to_gpu_multi_device()

        # print_layer_parameters(self.cmv.mlpm_ci)
        # time.sleep(10)
        # raise ValueError("stop here")

        cuda_hook_time("prefill")
        time_start_prefill = time.time()
        
        if len(self.device_list) == 4:
            # self.num_experts_on_cpu_ratio = 0.2
            self.num_experts_on_cpu_ratio = 0.2
        elif len(self.device_list) == 3:
            self.num_experts_on_cpu_ratio = 0.25
        else:
            self.num_experts_on_cpu_ratio = 0.5
        from models.mlpmodule import QWEN2_MODEL_NAME_TYPE, MIXTRAL_MODEL_NAME_TYPE, QWEN3_MODEL_NAME_TYPE
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
        
        ghidden_states = inputs_tokens
        for layer_idx in range(self.mlpm.config.num_hidden_layers):
            cuda_hook_time("prefill_layer")
            logger.debug(f"-------------------------------- start prefill layer {layer_idx} --------------------------------")
            
            if layer_idx < self.mlpm.config.num_hidden_layers-1:
                cuda_hook_time(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")
                self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1, device=self.device1)
                cuda_hook_time_end(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")

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
                # self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1,  device=device1)
                ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=ghidden_states)
                self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
                cuda_hook_time_end("dense_mlp")
            else:
                ghidden_states = self.layer_moe_generate_mp_multi_device(layer_idx=layer_idx, hidden_states=ghidden_states)
                # ghidden_states = self.layer_moe_dgenerate_mp_multi_device(layer_idx=layer_idx, hidden_states=ghidden_states)
            ghidden_states = ghidden_states + residual

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
            self.cpu_thread_manager_mp.submit(-1, [], [], [], [])
            self.cpu_thread_manager_mp.wait()

        if len(time_decode_list) >= 5:
            time_decode_list = time_decode_list[5:]
        logger.info(f"average decode time from decode step 5: {sum(time_decode_list) / len(time_decode_list)} seconds")
        cuda_hook_time("async_wait_layer_loaded_to_gpu")
        if len(self.device_list) > 1:
            self.cmv.async_wait_layer_loaded_to_gpu_multi_device()
        else:
            self.cmv.async_wait_layer_loaded_to_gpu()
        cuda_hook_time_end("async_wait_layer_loaded_to_gpu")
    @torch.no_grad()
    def layer_moe_generate_mp_single_device(self, layer_idx: int, hidden_states: torch.Tensor):
        cuda_hook_time(f"layer_moe_generate_mp_l_{layer_idx+1}")
        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape

        cuda_hook_time("gate")
        topk_idx, topk_weight, aux_loss = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_idx, hidden_states)
        flat_expert_indices = topk_idx.view(-1)      # [batch_size * seq_len * num_experts_per_tok]
        flat_experts_weight = topk_weight.view(-1, 1)  # [batch_size * seq_len * num_experts_per_tok, 1]
        idxs = flat_expert_indices.argsort()         # 排序后的索引
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # [num_experts]
        token_idxs = idxs // self.mlpm.config.num_experts_per_tok  # 恢复到原始 token 索引
        flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)  # [batch_size * seq_len, hidden_dim]
        cuda_hook_time_end("gate")

        cuda_hook_time("experts_map_get")
        # Step 7: 构建每个 expert 的索引信息（避免 tensor 计算和拷贝）
        # tokens_per_expert 现在是累积和，可以直接用于计算 start_idx 和 end_idx
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
        
        # Step 8: 根据token数量分配CPU/GPU experts
        sorted_experts_by_load = sorted(expert_token_counts_list, key=lambda x: x[1])
        num_experts_total = len(sorted_experts_by_load)
        
        # 使用固定值
        num_experts_on_cpu = int(num_experts_total * self.num_experts_on_cpu_ratio)
        
        cpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[:num_experts_on_cpu])
        gpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[num_experts_on_cpu:])
        total_tokens_cpu = sum(count for _, count in sorted_experts_by_load[:num_experts_on_cpu])
        total_tokens_gpu = sum(count for _, count in sorted_experts_by_load[num_experts_on_cpu:])

        # 打印调试信息
        cpu_ratio = num_experts_on_cpu / num_experts_total if num_experts_total > 0 else 0
        logger.debug(f"\nExpert Token Distribution & Device Allocation:")
        logger.debug(f"  Total experts: {num_experts_total}")
        logger.debug(f"  CPU experts: {num_experts_on_cpu} ({cpu_ratio*100:.0f}%)")
        logger.debug(f"  GPU experts: {num_experts_total - num_experts_on_cpu} ({(1-cpu_ratio)*100:.0f}%)")
        logger.debug(f"\n  Expert ID | Tokens | Device")
        logger.debug(f"  {'-'*35}")
            
        
        for expert_id, token_count in sorted_experts_by_load:
            device = "CPU" if expert_id in cpu_expert_ids else "GPU"
            logger.debug(f"  Expert {expert_id:2d} | {token_count:6d} | {device}")
        logger.debug(f"\n  CPU total tokens: {total_tokens_cpu} ({total_tokens_cpu/(total_tokens_cpu+total_tokens_gpu)*100:.1f}%)")
        logger.debug(f"  GPU total tokens: {total_tokens_gpu} ({total_tokens_gpu/(total_tokens_cpu+total_tokens_gpu)*100:.1f}%)")
        
        cuda_hook_time_end("experts_map_get")

        expert_cache = torch.zeros_like(flat_hidden_states)
        cuda_hook_time("cpu_experts_submit")
        # # CPU experts - 传递索引信息，延迟创建 tensor maps
        if cpu_expert_ids:
            logger.debug(f"\n  Computing {len(cpu_expert_ids)} experts on CPU MP...")
            self.cpu_thread_manager_mp.submit_worker(
                worker_idx=0,
                layer_idx=layer_idx,
                expert_idx_list=list(cpu_expert_ids),
                expert_indices_map={eid: expert_indices_map[eid] for eid in cpu_expert_ids},
                flat_hidden_states=flat_hidden_states,
                idxs=idxs,
            )
        cuda_hook_time_end("cpu_experts_submit")

        cuda_hook_time("allocate_experts_cuda_memory_and_restore_model")
        # 在上层提前加载
        # gpu_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_idx)
        gpu_shared_expert_names = []
        gpu_expert_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=list(gpu_expert_ids))
        gpu_expert_names = gpu_expert_names + gpu_shared_expert_names
        ret1, replica_uuid1, state_dict1 = \
            self.cmv.allocate_cuda_memory_and_load_into_gpu(
                gpu_expert_names, device_index_int=int(self.device1.split(":")[1]))
        self.cmv.restore2model(state_dict1, self.cmv.mlpm_ci)
        cuda_hook_time_end("allocate_experts_cuda_memory_and_restore_model")

        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            hidden_states=hidden_states,
        )
        cuda_hook_time_end("gpu_sexperts")

        # expert_cache = torch.zeros_like(flat_hidden_states)
        # cuda_hook_time("cpu_experts_submit")
        # # # CPU experts - 传递索引信息，延迟创建 tensor maps
        # if cpu_expert_ids:
        #     logger.debug(f"\n  Computing {len(cpu_expert_ids)} experts on CPU MP...")
        #     self.cpu_thread_manager_mp.submit_worker(
        #         worker_idx=0,
        #         layer_idx=layer_idx,
        #         expert_idx_list=list(cpu_expert_ids),
        #         expert_indices_map={eid: expert_indices_map[eid] for eid in cpu_expert_ids},
        #         flat_hidden_states=flat_hidden_states,
        #         idxs=idxs,
        #     )
        # if cpu_expert_ids:
        #     logger.debug(f"\n  Computing {len(cpu_expert_ids)} experts on CPU...")
        #     # 使用 CETM 在后台线程执行
        #     task = ExpertEinsumTask(
        #         layer_idx=layer_idx,
        #             expert_idx_list=list(cpu_expert_ids),
        #             expert_indices_map={eid: expert_indices_map[eid] for eid in cpu_expert_ids},
        #             expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in cpu_expert_ids},
        #             flat_hidden_states=flat_hidden_states,
        #             flat_experts_weight=flat_experts_weight,
        #             idxs=idxs,
        #             final_hidden_states=expert_cache
        #         )
        #     self.cetm.submit(task)
        # cuda_hook_time_end("cpu_experts_submit")
        

        device_expert_idx_list = list(gpu_expert_ids)
        device_stacked_inputs = self.mlpm.experts_func_mgpu_group_pad(
            expert_idx_list=device_expert_idx_list,
            expert_indices_map=expert_indices_map,
            device_flat_hidden_states=flat_hidden_states,
            device_idxs=idxs,
        )
        device_expert_weights_list = []
        device_token_ids_list = []
        for i, expert_idx in enumerate(device_expert_idx_list):
            start_idx, end_idx = expert_indices_map[expert_idx]
            token_ids = expert_token_indices_map[expert_idx]
            expert_weights = flat_experts_weight[idxs[start_idx:end_idx]]
            device_expert_weights_list.append(expert_weights)
            device_token_ids_list.append(token_ids)

        group_w1_list, group_w2_list, group_w3_list = self.mlpm.experts_func_mgpu_group_list(
            mi=self.cmv.mlpm_ci,
            layer_idx=layer_idx,
            expert_idx_list=list(gpu_expert_ids)
        )

        # 将列表 concat 为 tensor
        device_expert_weights_concat = torch.cat(device_expert_weights_list, dim=0) if device_expert_weights_list else None
        device_token_ids_concat = torch.cat(device_token_ids_list, dim=0) if device_token_ids_list else None
        # if layer_idx < self.mlpm.config.num_hidden_layers-2:
        #     cuda_hook_time("init_set_layer_func")
        #     self.mlpm.init_set_layer_func(layer_idx=layer_idx+2, config=self.mlpm.config, model=self.cmv.mlpm_ci)
        #     cuda_hook_time_end("init_set_layer_func")
         
        cuda_hook_time("acpu_expert_weight_slices")
        acpu_expert_outs_slices = []
        acpu_expert_weights = []
        acpu_token_ids = []
        for i, expert_idx in enumerate(list(cpu_expert_ids)):
            token_ids = expert_token_indices_map[expert_idx]
            num_tokens = token_ids.shape[0]

            # 收集 outputs 切片（不应用 weights）
            # expert_out_slice = outputs[i, :num_tokens]
            # all_expert_outs_slices.append(expert_out_slice)

            # 收集对应的 weights
            start_idx, end_idx = expert_indices_map[expert_idx]
            expert_weights = flat_experts_weight[idxs[start_idx:end_idx]]
            acpu_expert_weights.append(expert_weights)
            acpu_token_ids.append(token_ids)
        
        cuda_hook_time_end("acpu_expert_weight_slices")


        if layer_idx < self.mlpm.config.num_hidden_layers-1:
            cuda_hook_time("wait_load_qkvogn_s_weight")
            self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
            cuda_hook_time_end("wait_load_qkvogn_s_weight")

        cuda_hook_time("cpu_thread_manager_mp_wait")
        output_cpu2gpu = self.cpu_thread_manager_mp.wait()
        cuda_hook_time_end("cpu_thread_manager_mp_wait")
        cuda_hook_time("cpuoutputsdeal")
        # for i, expert_idx in enumerate(list(cpu_expert_ids)):
        #     token_ids = expert_token_indices_map[expert_idx]
        #     num_tokens = token_ids.shape[0]
        #     expert_out_slice = output_cpu2gpu[i, :num_tokens]
        #     acpu_expert_outs_slices.append(expert_out_slice)
        # concat_expert_out = torch.cat(acpu_expert_outs_slices, dim=0)  # [total_tokens, H]
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


        cuda_hook_time("wait_experts")
        self.cmv.wait_load_into_gpu(replica_uuid1)
        cuda_hook_time_end("wait_experts")

        

        cuda_hook_time("gpu_experts")
        # _ = self.mlpm.experts_func_gpu_einsum(
        #     self.cmv.mlpm_ci, layer_idx=layer_idx,
        #     expert_idx_list=list(gpu_expert_ids),
        #     expert_indices_map={eid: expert_indices_map[eid] for eid in gpu_expert_ids},
        #     expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in gpu_expert_ids},
        #     flat_hidden_states=flat_hidden_states,
        #     flat_experts_weight=flat_experts_weight,
        #     idxs=idxs,
        #     final_hidden_states=expert_cache,
        # )

        device1_id = flat_hidden_states.device.index
        

        # cuda_hook_time("wait_cetm_experts")
        # result = self.cetm.get_result()
        # outputs_cpu = result.final_hidden_states
        # output_cpu2gpu = outputs_cpu.to(device1_id, non_blocking=True)
        # cuda_hook_time_end("wait_cetm_experts")
        
        _ = self.mlpm.experts_func_mgpu_einsum_mp_multi_list(
            layer_idx=layer_idx,
            group_w1_list_map={device1_id: group_w1_list},
            group_w2_list_map={device1_id: group_w2_list},
            group_w3_list_map={device1_id: group_w3_list},
            stacked_inputs_map={device1_id: device_stacked_inputs},
            expert_idx_list_map={device1_id: device_expert_idx_list},
            expert_indices_map=expert_indices_map,
            flat_hidden_states_map={device1_id: flat_hidden_states},
            flat_experts_weight_map={device1_id: flat_experts_weight},
            idxs_map={device1_id: idxs},
            final_hidden_states=expert_cache,
            all_expert_weights_map={device1_id: device_expert_weights_concat},
            all_token_ids_map={device1_id: device_token_ids_concat},
            expert_token_indices_map=expert_token_indices_map,
        )
        cuda_hook_time_end("gpu_experts")

        layer_output = expert_cache.view(*orig_shape) + y

        cuda_hook_time_end(f"layer_moe_generate_mp_l_{layer_idx+1}")
        return layer_output
    
    def layer_moe_generate_mp_single_device_fused(self, layer_idx: int, hidden_states: torch.Tensor):
        """Gemma4-style fused experts: ``gate_up_proj`` + ``down_proj``. Matmul backend: ``bmm`` (pad) or ``gmm`` (``_grouped_mm``), see env ``LMP_FUSED_EXPERT_MM``."""
        cuda_hook_time(f"layer_moe_generate_mp_fused_l_{layer_idx+1}")

        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape
        k = int(self.mlpm.get_experts_per_tok())

        # Step 1 — gate (+ layout). 将 (B,S,K) 视为线性槽位不额外拷贝：view 与 reshape 等价于扁平化索引，
        # 便于一次 argsort、与 flat_hidden_states[B*S,H] 用 token_idx = slot//K 对齐；非必须，也可用二维索引实现同样语义。
        cuda_hook_time("gate")
        topk_idx, topk_weight, aux_loss = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_idx, hidden_states)
        flat_expert_indices = topk_idx.reshape(-1).to(dtype=torch.int64)
        flat_experts_weight = topk_weight.reshape(-1, 1)
        idxs = flat_expert_indices.argsort()
        token_idxs = idxs // k
        flat_hidden_states = hidden_states.reshape(batch_size * seq_len, -1)
        cuda_hook_time_end("gate")

        num_slots = int(idxs.numel())
        expert_cache = torch.zeros_like(flat_hidden_states)

        if num_slots == 0:
            cuda_hook_time("gpu_sexperts")
            y = self.mlpm.shared_experts_func(
                self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states
            )
            cuda_hook_time_end("gpu_sexperts")
            cuda_hook_time_end(f"layer_moe_generate_mp_fused_l_{layer_idx+1}")
            return expert_cache.view(*orig_shape) + y

        # Step 2 — build expert ranges + partition experts across CPU/GPU (same as non-fused single-device)
        cuda_hook_time("experts_map_get")
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        expert_indices_map = {}
        expert_token_indices_map = {}
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
            expert_token_indices_map[expert_id] = token_idxs[start_idx:end_idx]
            expert_token_counts_list.append((expert_id, end_idx - start_idx))
            prev_end = end_idx

        sorted_experts_by_load = sorted(expert_token_counts_list, key=lambda x: x[1])
        num_experts_total = len(sorted_experts_by_load)
        num_experts_on_cpu = int(num_experts_total * self.num_experts_on_cpu_ratio)
        cpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[:num_experts_on_cpu])
        gpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[num_experts_on_cpu:])
        cuda_hook_time_end("experts_map_get")

        # Step 3 — submit CPU fused bmm async (MP)
        cuda_hook_time("cpu_experts_submit_fused")
        if cpu_expert_ids:
            self.cpu_thread_manager_mp.submit_worker(
                worker_idx=0,
                layer_idx=layer_idx,
                expert_idx_list=list(cpu_expert_ids),
                expert_indices_map={eid: expert_indices_map[eid] for eid in cpu_expert_ids},
                flat_hidden_states=flat_hidden_states,
                idxs=idxs,
                use_bmm=True,
            )
        cuda_hook_time_end("cpu_experts_submit_fused")

        # Step 4 — load GPU experts + shared branch
        cuda_hook_time("allocate_experts_cuda_memory_and_restore_model_fused")
        gpu_expert_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=list(gpu_expert_ids))
        _ret, replica_uuid1, state_dict1 = self.cmv.allocate_cuda_memory_and_load_into_gpu(
            gpu_expert_names, device_index_int=int(self.device1.split(":")[1])
        )
        self.cmv.restore2model(state_dict1, self.cmv.mlpm_ci)
        cuda_hook_time_end("allocate_experts_cuda_memory_and_restore_model_fused")

        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
        cuda_hook_time_end("gpu_sexperts")

        # Step 5 — wait CPU and scatter CPU outputs (weighted)
        cuda_hook_time("cpu_thread_manager_mp_wait_fused")
        output_cpu2gpu = self.cpu_thread_manager_mp.wait() if cpu_expert_ids else None
        cuda_hook_time_end("cpu_thread_manager_mp_wait_fused")

        if cpu_expert_ids:
            cuda_hook_time("cpuoutputsdeal_fused")
            acpu_expert_weights = []
            acpu_token_ids = []
            for expert_idx in list(cpu_expert_ids):
                start_idx, end_idx = expert_indices_map[expert_idx]
                acpu_expert_weights.append(flat_experts_weight[idxs[start_idx:end_idx]])
                acpu_token_ids.append(expert_token_indices_map[expert_idx])
            concat_expert_out = output_cpu2gpu
            concat_expert_weights = torch.cat(acpu_expert_weights, dim=0)
            concat_token_ids = torch.cat(acpu_token_ids, dim=0)
            concat_expert_out = concat_expert_out.mul_(concat_expert_weights)
            expert_cache.scatter_reduce_(
                dim=0,
                index=concat_token_ids.view(-1, 1).expand(-1, expert_cache.shape[-1]),
                src=concat_expert_out,
                reduce="sum",
            )
            del output_cpu2gpu, concat_expert_out
            cuda_hook_time_end("cpuoutputsdeal_fused")

        # Step 6 — wait GPU expert weights ready then run fused compute for GPU experts only
        cuda_hook_time("wait_experts_fused")
        self.cmv.wait_load_into_gpu(replica_uuid1)
        cuda_hook_time_end("wait_experts_fused")

        cuda_hook_time("gpu_experts_fused")
        if gpu_expert_ids:
            # Build expert-sorted idx stream restricted to GPU experts to avoid touching missing CPU expert weights.
            idxs_gpu_chunks = []
            for eid in sorted(gpu_expert_ids):
                s, e = expert_indices_map[eid]
                idxs_gpu_chunks.append(idxs[s:e])
            idxs_gpu = torch.cat(idxs_gpu_chunks, dim=0) if idxs_gpu_chunks else idxs[:0]

            token_idxs_gpu = idxs_gpu // k
            x_slots_gpu = flat_hidden_states[token_idxs_gpu]
            expert_ids_gpu = flat_expert_indices[idxs_gpu].to(dtype=torch.int64, device=flat_hidden_states.device)
            slot_w_gpu = flat_experts_weight[idxs_gpu]

            gate_up, down, act_fn = self.mlpm.get_fused_experts_gate_up_down_act_fn(self.cmv.mlpm_ci, layer_idx)
            mm_backend = os.environ.get("LMP_FUSED_EXPERT_MM", "bmm").strip().lower()
            w_gu = gate_up.transpose(1, 2).contiguous()
            w_down = down.transpose(1, 2).contiguous()
            y_slots_gpu = self.mlpm.fused_experts_gate_up_down_mm_presorted(
                x_slots_sorted=x_slots_gpu,
                expert_ids_sorted=expert_ids_gpu,
                gate_up_w_eh2i=w_gu,
                down_w_eih=w_down,
                act_fn=act_fn,
                mm_backend=mm_backend,
            )

            expert_cache.scatter_reduce_(
                dim=0,
                index=token_idxs_gpu.view(-1, 1).expand(-1, expert_cache.size(-1)),
                src=y_slots_gpu * slot_w_gpu,
                reduce="sum",
            )
        cuda_hook_time_end("gpu_experts_fused")

        cuda_hook_time_end(f"layer_moe_generate_mp_fused_l_{layer_idx+1}")
        return expert_cache.view(*orig_shape) + y

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
        device_expert_map = {device_id: [] for device_id in device_ids}
        device_token_counts = {device_id: 0 for device_id in device_ids}

        gpu_experts_sorted = sorted(expert_token_counts_list, key=lambda x: x[1], reverse=True)
        for expert_id, token_count in gpu_experts_sorted:
            min_device_id = min(device_ids, key=lambda d: device_token_counts[d])
            device_expert_map[min_device_id].append(expert_id)
            device_token_counts[min_device_id] += token_count

        gpu_expert_ids_by_device = {device_id: set(expert_ids) for device_id, expert_ids in device_expert_map.items()}
        cuda_hook_time_end("experts_map_get")

        # -----------------------
        # Step 3: load experts multi-device + shared experts
        # -----------------------
        cuda_hook_time("allocate_experts_cuda_memory_and_restore_model_multi_device_fused")
        _ret, replica_uuid, state_dict = self.cmv.allocate_cuda_memory_fused_experts(
            layer_idx=layer_idx,
            gpu_expert_ids_by_device=gpu_expert_ids_by_device
        )
        # Keep a strong reference to the returned tensors, since they are views into
        # allocated GPU memory and are required for subsequent fused expert compute.
        if not hasattr(self, "_fused_experts_state_dict_cache"):
            self._fused_experts_state_dict_cache = {}
        self._fused_experts_state_dict_cache[layer_idx] = state_dict
        # self.cmv.restore2model(state_dict, self.cmv.mlpm_ci)
        cuda_hook_time_end("allocate_experts_cuda_memory_and_restore_model_multi_device_fused")

        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=hidden_states)
        cuda_hook_time_end("gpu_sexperts")

        cuda_hook_time("wait_experts_multi_device")
        self.cmv.wait_load_into_gpu(replica_uuid)
        cuda_hook_time_end("wait_experts_multi_device")

        # Activation for fused gate half (weights are taken from `state_dict` per-device).
        _, _, act_fn = self.mlpm.get_fused_experts_gate_up_down_act_fn(self.cmv.mlpm_ci, layer_idx)
        mm_backend = os.environ.get("LMP_FUSED_EXPERT_MM", "bmm").strip().lower()

        # Presorted slot tensors on "main" device (where hidden_states live)
        main_device = flat_hidden_states.device
        x_slots_all = flat_hidden_states[token_idxs]  # [slots,H] presorted
        expert_ids_all = flat_expert_indices[idxs].to(device=main_device, dtype=torch.int64)  # [slots]
        slot_w_all = flat_experts_weight[idxs].to(device=main_device)  # [slots,1]

        # -----------------------
        # Step 4: per-device fused compute + aggregate to main expert_cache
        # -----------------------
        cuda_hook_time("gpu_experts_multi_device_fused")
        import re

        # Build global-id -> tensor maps from the loaded `state_dict`.
        # Expect keys to encode expert id, e.g. "...experts.<id>....gate_up_proj..." / "...down_proj...".
        def _parse_expert_id(param_name: str) -> int | None:
            m = re.search(r"(?:\\.experts\\.experts\\.|\\.experts\\.)(\\d+)(?:\\.|$)", param_name)
            if m:
                return int(m.group(1))
            return None

        gate_up_map: dict[int, torch.Tensor] = {}
        down_map: dict[int, torch.Tensor] = {}
        for name, t in state_dict.items():
            eid = _parse_expert_id(name)
            if eid is None:
                continue
            if "gate_up_proj" in name:
                gate_up_map[eid] = t
            elif "down_proj" in name:
                down_map[eid] = t

        for device_id in device_ids:
            device_expert_ids = sorted(gpu_expert_ids_by_device[device_id])
            if not device_expert_ids:
                continue

            # Build contiguous expert-sorted slices for this device.
            x_parts = []
            eid_parts = []
            w_parts = []
            for eid in device_expert_ids:
                start_idx, end_idx = expert_indices_map[eid]
                x_parts.append(x_slots_all[start_idx:end_idx])
                eid_parts.append(expert_ids_all[start_idx:end_idx])
                w_parts.append(slot_w_all[start_idx:end_idx])

            x_dev_sorted = torch.cat(x_parts, dim=0)
            eid_dev_sorted = torch.cat(eid_parts, dim=0)
            w_dev_sorted = torch.cat(w_parts, dim=0)
            if x_dev_sorted.numel() == 0:
                continue

            device = torch.device(f"cuda:{device_id}")
            # Move inputs to device for compute.
            x_dev_sorted = x_dev_sorted.to(device, non_blocking=True)
            eid_dev_sorted = eid_dev_sorted.to(device, non_blocking=True)

            # Build per-device weight bank [E_dev, ...] from `state_dict` and remap ids to [0..E_dev-1].
            ids_tensor = torch.tensor(device_expert_ids, device=device, dtype=torch.int64)
            remap = torch.full((num_experts,), -1, dtype=torch.int64, device=device)
            remap[ids_tensor] = torch.arange(ids_tensor.numel(), device=device, dtype=torch.int64)
            eid_dev_sub = remap[eid_dev_sorted]

            # Stack local expert weights in the same order as `device_expert_ids`.
            try:
                gate_up_sub = torch.stack([gate_up_map[eid].to(device) for eid in device_expert_ids], dim=0)
                down_sub = torch.stack([down_map[eid].to(device) for eid in device_expert_ids], dim=0)
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing fused expert weight in state_dict for expert id {exc}. "
                    f"Parsed gate_up={len(gate_up_map)} down={len(down_map)}"
                ) from exc

            w_gu = gate_up_sub.transpose(1, 2).contiguous()  # [E_dev, H, 2I]
            w_down = down_sub.transpose(1, 2).contiguous()  # [E_dev, I, H]

            y_dev_sorted = self.mlpm.fused_experts_gate_up_down_mm_presorted(
                x_slots_sorted=x_dev_sorted,
                expert_ids_sorted=eid_dev_sub,
                gate_up_w_eh2i=w_gu,
                down_w_eih=w_down,
                act_fn=act_fn,
                mm_backend=mm_backend,
            )

            # Bring back to main device and scatter-reduce into expert_cache.
            y_main = y_dev_sorted.to(main_device, non_blocking=True)
            out_weighted = y_main * w_dev_sorted  # weights already on main

            # token indices for these slots: since we concatenated per-expert contiguous ranges, token_idxs slice accordingly.
            token_parts = []
            for eid in device_expert_ids:
                start_idx, end_idx = expert_indices_map[eid]
                token_parts.append(token_idxs[start_idx:end_idx])
            token_ids_for_slots = torch.cat(token_parts, dim=0).to(main_device)

            expert_cache.scatter_reduce_(
                dim=0,
                index=token_ids_for_slots.view(-1, 1).expand(-1, expert_cache.size(-1)),
                src=out_weighted,
                reduce="sum",
            )

        cuda_hook_time_end("gpu_experts_multi_device_fused")

        layer_output = expert_cache.view(*orig_shape) + y
        cuda_hook_time_end(f"layer_moe_generate_mp_multi_device_fused_l_{layer_idx+1}")
        return layer_output


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
        token_idxs = idxs // self.mlpm.config.num_experts_per_tok  # 恢复到原始 token 索引
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
            logger.debug(f"\n  Computing {len(cpu_expert_ids)} experts on CPU MP...")
            self.cpu_thread_manager_mp.submit_worker(
                worker_idx=0,
                layer_idx=layer_idx,
                expert_idx_list=list(cpu_expert_ids),
                expert_indices_map={eid: expert_indices_map[eid] for eid in cpu_expert_ids},
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
                device_token_idxs = device_idxs // self.mlpm.config.num_experts_per_tok
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
        output_cpu2gpu = self.cpu_thread_manager_mp.wait() if cpu_expert_ids else None
        cuda_hook_time_end("cpu_thread_manager_mp_wait")
        
        # Step 14: 处理CPU experts的输出
        if output_cpu2gpu is not None:
            cuda_hook_time("cpuoutputsdeal")
            acpu_expert_outs_slices = []
            acpu_expert_weights = []
            acpu_token_ids = []
            for i, expert_idx in enumerate(list(cpu_expert_ids)):
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
        token_idxs = idxs // self.mlpm.config.num_experts_per_tok  # 恢复到原始 token 索引
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
                device_token_idxs = device_idxs // self.mlpm.config.num_experts_per_tok
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
        output_cpu2gpu = self.cpu_thread_manager_mp.wait() if experts_cpu_list else None
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
        bsz, seq_len = 4, 64
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

            for _ in range(max(0, warmup)):
                self.cpu_thread_manager_mp.submit_worker(
                    worker_idx=0,
                    layer_idx=layer_idx,
                    expert_idx_list=expert_idx_list,
                    expert_indices_map=e_map,
                    flat_hidden_states=flat_hidden_states,
                    idxs=idxs,
                    use_bmm=True,
                )
                out = self.cpu_thread_manager_mp.wait()
                if out is not None:
                    del out
                torch.cuda.synchronize(device)

            times_ms: list[float] = []
            for _ in range(num_iters):
                torch.cuda.synchronize(device)
                t0 = time_mod.perf_counter()
                self.cpu_thread_manager_mp.submit_worker(
                    worker_idx=0,
                    layer_idx=layer_idx,
                    expert_idx_list=expert_idx_list,
                    expert_indices_map=e_map,
                    flat_hidden_states=flat_hidden_states,
                    idxs=idxs,
                    use_bmm=True,
                )
                out = self.cpu_thread_manager_mp.wait()
                torch.cuda.synchronize(device)
                t1 = time_mod.perf_counter()
                times_ms.append((t1 - t0) * 1e3)
                if out is not None:
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