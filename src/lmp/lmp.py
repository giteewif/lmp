from importlib import import_module
import os
from threading import get_ident
from turtle import position
from typing import Dict
import copy
import torch
import time
from transformers import AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
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
from lmp.cuda_memory_view import CudaMemoryView, HostMemoryView
from lmp.cpu_thread_manager import CETM
from lmp.sllm_thread_manager import SLLMTM
from lmp.cpu_thread_manager import CETM
from lmp.init_meta_manager import InitMetaManager
from lmp.init_meta_manager_mp_shared import InitMetaManagerMPShared

logger = init_logger(__name__)


class MLPLLM:
    def __init__(
        self,
        model_name_type: str,
        model_path: str,
    ):
        self.model_path = model_path
        self.model_name_type = model_name_type

        client = SllmStoreClient(SLLM_ADDRESS)
        ret = load_into_cpu(client, model_path)
        if not ret:
            raise ValueError(f"Failed to load model {model_path} into CPU")

        mlpm = MLPModuleWrapper(model_name_type, model_path)
        self.mlpm  = mlpm
        
        hmv = HostMemoryView(self.mlpm)
        cmv = CudaMemoryView(self.mlpm)
        self.hmv = hmv
        self.cmv = cmv

        cetm = CETM(self.mlpm, self.hmv, num_workers=1)  # 默认1个worker，可通过参数调整
        self.cetm = cetm
        self.cetm.start()  # 启动工作线程

        sllmtm = SLLMTM(num_workers=1)  # SLLM 线程管理器，用于异步加载
        self.sllmtm = sllmtm
        self.sllmtm.start()  # 启动工作线程

        imm = InitMetaManager()
        imm.start()
        self.imm = imm

        self.cmv.sllmtm = sllmtm     # 将sllmtm绑定到cmv中  
        self.cmv.imm = imm

        device1 = "cuda:1"
        self.device1 = device1
        
        # CPU专家数量：使用固定值 0.5 * total
        # stream = torch.cuda.Stream(device=device1)
        # with torch.cuda.stream(stream):
    def free_cmv(self):
        # 释放gpu分配的资源
        self.cmv.free_allocated()
    
    @torch.no_grad()
    def test_generate_multi_layer(self): 
        

        cuda_hook_time("generate_input_ids")
        batch_size = 32
        seq_len = 64
        dtype = self.mlpm.config.torch_dtype
        hidden_size = self.mlpm.config.hidden_size
        device1 = "cuda:1"
        device_idx_int1 = int(device1.split(":")[1])
        inputs_tokens = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device1)

        tokenizer=AutoTokenizer.from_pretrained(self.mlpm.model_abs_path, trust_remote_code=True)
        inputs_ids = generate_input_ids(tokenizer, batch_size, seq_len, device1)
        cuda_hook_time_end("generate_input_ids")

        device2 = "cuda:2"
        cuda_hook_time("init_cache")
        # StaticCache
        # past_key_value = StaticCache(
        #    config=self.mlpm.config, batch_size=batch_size, 
        #     max_cache_len=seq_len+2, 
        #     device=device1, dtype=dtype
        # ) 
        past_key_value = DynamicCache()
        past_key_values_length = past_key_value.get_usable_length(seq_len)
        cuda_hook_time_end("init_cache")

        cuda_hook("init_meta")
        self.cmv.start_init_meta_model(hmv=self.hmv)
        # self.cmv.imm_submit_all()
        # self.cmv.imm_submit_meta_layer(layer_idx=1)
        cuda_hook_end("init_meta")

        cuda_hook_time("init_weights")
        self.cmv.load_general_and_init()
        # self.cmv.imm_wait_meta_layer(layer_idx=0)
        # self.cmv.imm_wait_meta_layer(layer_idx=1)
        # self.cmv.imm_wait_all()
        self.cmv.init_load_qkvogn_es_weight(layer_idx=0)
        cuda_hook_time_end("init_weights")

        cuda_hook_time("copy_emodel")
        model_cpy = copy.deepcopy(self.cmv.mlpm_ci.model)
        cuda_hook_time_end("copy_emodel")
        # cuda_hook_time("wait_all_meta")
        # self.cmv.imm.wait_all()
        # cuda_hook_time_end("wait_all_meta")

        cuda_hook_time("init_inputs_tokens")
        inputs_tokens = self.cmv.mlpm_ci.model.embed_tokens(inputs_ids)
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
        cuda_hook_time_end("init_inputs_tokens")

        # cuda_hook_time("warm_up")
        # num_warm_up = 3
        # for i in range(num_warm_up):
        #     self.layer_warmup_generate(hidden_states=inputs_tokens)
        #     ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=0, hidden_states=inputs_tokens)
        #     ghidden_states = self.mlpm.paln_func(self.cmv.mlpm_ci, layer_idx=0, hidden_states=inputs_tokens)
        # cuda_hook_time_end("warm_up")

        cuda_hook_time("multi_layer")
        # self.load_qkvogn_s_weight(layer_idx=0)
        num_step = 1
        self.num_experts_on_cpu_ratio = 0.5
        # prefill
        for i in range(num_step):
            ghidden_states = inputs_tokens
            for layer_idx in range(self.mlpm.config.num_hidden_layers):
                # 测试Deepseek 跳过dense测试
                logger.debug(f"-------------------------------- start layer {layer_idx} --------------------------------")

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
                if layer_idx == 0:
                    cuda_hook_time("dense_mlp")
                    self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1,  device=device1)
                    ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=ghidden_states)
                    self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
                    cuda_hook_time_end("dense_mlp")

                    # cuda_hook_time(f"waiting_meta_l{layer_idx}")
                    # if layer_idx < self.mlpm.config.num_hidden_layers - 2:
                    #     # 每层计算提前等待初始化好下一层的layer, 第一层已初始化好
                    #     self.cmv.imm_submit_meta_layer(layer_idx=layer_idx+2)
                    #     self.cmv.imm_wait_meta_layer(layer_idx=layer_idx+2)
                    # cuda_hook_time_end(f"waiting_meta_l{layer_idx}")

                else:
                    ghidden_states = self.layer_moe_generate(layer_idx=layer_idx, hidden_states=ghidden_states)
                    # ghidden_states = self.layer_moe_generate(layer_idx=layer_idx, hidden_states=ghidden_states)
                ghidden_states = ghidden_states + residual

                # if check_nan_inf(ghidden_states):
                #     logger.error(f"ERROR: ghidden_states contains NaN!")
                #     raise ValueError("ghidden_states contain NaN")
                logger.debug(f"-------------------------------- end layer {layer_idx} --------------------------------")
                # torch.cuda.synchronize(device=device1)
        cuda_hook_time_end("multi_layer")

        def get_next_token(hidden_states):
            normed_hidden_states = self.cmv.mlpm_ci.model.norm(hidden_states)
            last_hidden_states = normed_hidden_states[:, -1:, :]  # Shape: (batch_size, 1, hidden_size)
            next_token_logits = self.cmv.mlpm_ci.lm_head(last_hidden_states).squeeze(1)
            next_token_logits = next_token_logits.float()
            
            # // Debug: 检查 logits 是否包含 inf/nan
            torch.cuda.synchronize()  # 确保之前的操作完成
            if torch.isnan(next_token_logits).any():
                logger.error(f"ERROR: next_token_logits contains NaN!")
                logger.error(f"  NaN count: {torch.isnan(next_token_logits).sum().item()}")
                logger.error(f"  Logits shape: {next_token_logits.shape}")
                logger.error(f"  Logits stats: min={next_token_logits.min().item():.6f}, max={next_token_logits.max().item():.6f}, mean={next_token_logits.mean().item():.6f}")
                raise ValueError("Logits contain NaN")
            
            if torch.isinf(next_token_logits).any():
                logger.error(f"ERROR: next_token_logits contains Inf!")
                logger.error(f"  Inf count: {torch.isinf(next_token_logits).sum().item()}")
                logger.error(f"  Logits shape: {next_token_logits.shape}")
                logger.error(f"  Logits stats: min={next_token_logits.min().item():.6f}, max={next_token_logits.max().item():.6f}, mean={next_token_logits.mean().item():.6f}")
                # 修复 inf: 将 inf 替换为有限值
                next_token_logits = torch.where(
                    torch.isinf(next_token_logits),
                    torch.tensor(0.0, device=next_token_logits.device, dtype=next_token_logits.dtype),
                    next_token_logits
                )
                logger.warning(f"  Fixed Inf values in logits")
            
            # 检查 logits 范围是否合理
            logits_max = next_token_logits.max().item()
            logits_min = next_token_logits.min().item()
            if abs(logits_max) > 1000 or abs(logits_min) > 1000:
                logger.warning(f"WARNING: Logits have extreme values: min={logits_min:.2f}, max={logits_max:.2f}")
                # 裁剪极端值以避免 softmax 溢出
                next_token_logits = torch.clamp(next_token_logits, min=-100, max=100)
                logger.warning(f"  Clamped logits to [-100, 100]")
            # //

            next_token_ids = process_logits_efficiently(
                logits=next_token_logits,
                temperature=1.0,
                top_p=0.9,
                do_sample=True,
                device=device1
            )
            next_token_ids = next_token_ids.unsqueeze(1) # Shape: (batch_size, 1)
            return next_token_ids
    
        cuda_hook_time("init_inputs_tokens")
        next_token_ids = get_next_token(ghidden_states)
        next_inputs_tokens = self.cmv.mlpm_ci.model.embed_tokens(next_token_ids)
        position_ids = torch.arange(
            past_key_values_length, 1 + past_key_values_length, dtype=torch.long, device=device1
        )
        position_ids = position_ids.unsqueeze(0)
        # sdpa flash attention
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            None,
            (batch_size, 1),
            next_inputs_tokens,
            past_key_values_length=past_key_values_length,
        )
        cuda_hook_time_end("init_inputs_tokens")

        ghidden_states=next_inputs_tokens
        logger.debug(f"next_inputs_tokens shape: {next_inputs_tokens.shape}")
        cuda_hook_time("dense_mlp")
        ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=0, hidden_states=ghidden_states)
        logger.debug(f"ghidden_states after dense_mlp_func shape: {ghidden_states.shape}")
        cuda_hook_time_end("dense_mlp")
        # decode
        self.layer_moe_dgenerate(1, ghidden_states)
        torch.cuda.synchronize()
    @torch.no_grad()
    def layer_moe_generate(
        self, 
        layer_idx: int,
        hidden_states: torch.Tensor,
    ):
        cuda_hook_time(f"layer_moe_generate_{layer_idx}")

        device1_idx_int = int(self.device1.split(":")[1])

        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape
        cuda_hook_time("gate")
        # Step 1: 通过 gate 函数获取每个 token 选择的 top-k 专家和权重
        topk_idx, topk_weight, aux_loss = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_idx, hidden_states)
        # Step 2: 展平 expert indices 和 weights
        flat_expert_indices = topk_idx.view(-1)      # [batch_size * seq_len * num_experts_per_tok]
        flat_experts_weight = topk_weight.view(-1, 1)  # [batch_size * seq_len * num_experts_per_tok, 1]
        # Step 3: 按 expert_id 排序，将相同专家的 token 聚集在一起
        idxs = flat_expert_indices.argsort()         # 排序后的索引
        # Step 4: 统计每个专家被分配的 token 数量
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # [num_experts]
        # Step 5: 计算原始 token 的索引
        token_idxs = idxs // self.mlpm.config.num_experts_per_tok  # 恢复到原始 token 索引
            # Step 6: 展平 inputs_tokens（不复制数据，只是 view）
        # hidden_states: [batch_size, seq_len, hidden_dim]
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
        
        # 打印调试信息
        cpu_ratio = num_experts_on_cpu / num_experts_total if num_experts_total > 0 else 0
        logger.debug(f"\nExpert Token Distribution & Device Allocation:")
        logger.debug(f"  Total experts: {num_experts_total}")
        logger.debug(f"  CPU experts: {num_experts_on_cpu} ({cpu_ratio*100:.0f}%)")
        logger.debug(f"  GPU experts: {num_experts_total - num_experts_on_cpu} ({(1-cpu_ratio)*100:.0f}%)")
        logger.debug(f"\n  Expert ID | Tokens | Device")
        logger.debug(f"  {'-'*35}")
        
        total_tokens_cpu = sum(count for _, count in sorted_experts_by_load[:num_experts_on_cpu])
        total_tokens_gpu = sum(count for _, count in sorted_experts_by_load[num_experts_on_cpu:])
        for expert_id, token_count in sorted_experts_by_load:
            device = "CPU" if expert_id in cpu_expert_ids else "GPU"
            logger.debug(f"  Expert {expert_id:2d} | {token_count:6d} | {device}")
        logger.debug(f"\n  CPU total tokens: {total_tokens_cpu} ({total_tokens_cpu/(total_tokens_cpu+total_tokens_gpu)*100:.1f}%)")
        logger.debug(f"  GPU total tokens: {total_tokens_gpu} ({total_tokens_gpu/(total_tokens_cpu+total_tokens_gpu)*100:.1f}%)")
            
        cuda_hook_time_end("experts_map_get")

        # # 1. 提交cpu 专家执行
        # cuda_hook_time("cpu_experts_submit")
        # expert_cache = torch.zeros_like(flat_hidden_states)
        # # CPU experts - 传递索引信息，延迟创建 tensor maps
        # if cpu_expert_ids:
        #     logger.debug(f"\n  Computing {len(cpu_expert_ids)} experts on CPU...")
        #     # 使用 CETM 在后台线程执行
        #     task = ExpertEinsumTask(
        #         layer_idx=layer_idx,
        #         expert_idx_list=list(cpu_expert_ids),
        #         expert_indices_map={eid: expert_indices_map[eid] for eid in cpu_expert_ids},
        #         expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in cpu_expert_ids},
        #         flat_hidden_states=flat_hidden_states,
        #         flat_experts_weight=flat_experts_weight,
        #         idxs=idxs,
        #         final_hidden_states=expert_cache
        #     )
        #     self.cetm.submit(task)
        # cuda_hook_time_end("cpu_experts_submit")

        cuda_hook_time("allocate_experts_cuda_memory_and_restore_model")
        # 在上层提前加载
        # gpu_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_idx)
        gpu_shared_expert_names = []
        gpu_expert_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=list(gpu_expert_ids))
        gpu_expert_names = gpu_expert_names + gpu_shared_expert_names
        ret1, replica_uuid1, state_dict1 = \
            self.cmv.allocate_cuda_memory_and_load_into_gpu(
            gpu_expert_names, device_index_int=device1_idx_int)
        self.cmv.restore2model(state_dict1, self.cmv.mlpm_ci)
        cuda_hook_time_end("allocate_experts_cuda_memory_and_restore_model")

        # cuda_hook_time(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")
        # self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1, device=self.device1)
        # cuda_hook_time_end(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")

        # 2. 提交cpu 专家执行
        cuda_hook_time("cpu_experts_submit")
        expert_cache = torch.zeros_like(flat_hidden_states)
        # CPU experts - 传递索引信息，延迟创建 tensor maps
        if cpu_expert_ids:
            logger.debug(f"\n  Computing {len(cpu_expert_ids)} experts on CPU...")
            # 使用 CETM 在后台线程执行
            task = ExpertEinsumTask(
                layer_idx=layer_idx,
                    expert_idx_list=list(cpu_expert_ids),
                expert_indices_map={eid: expert_indices_map[eid] for eid in cpu_expert_ids},
                expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in cpu_expert_ids},
                flat_hidden_states=flat_hidden_states,
                flat_experts_weight=flat_experts_weight,
                idxs=idxs,
                    final_hidden_states=expert_cache
                )
            self.cetm.submit(task)
        cuda_hook_time_end("cpu_experts_submit")

        cuda_hook_time("wait_cetm_experts")
        result = self.cetm.get_result()
        cuda_hook_time_end("wait_cetm_experts")

        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            hidden_states=hidden_states,
        )
        cuda_hook_time_end("gpu_sexperts")

        # cuda_hook_time(f"waiting_meta_l{layer_idx}")
        # if layer_idx < self.mlpm.config.num_hidden_layers - 2:
        #     # 每层计算提前等待初始化好下一层的layer, 第一层已初始化好
        #     # self.cmv.imm_submit_meta_layer(layer_idx=layer_idx+1)
        #     self.cmv.imm_submit_meta_layer(layer_idx=layer_idx+2)
        # cuda_hook_time_end(f"waiting_meta_l{layer_idx}")

        # 等待 load_qkvogn_s 加载完成
        cuda_hook_time("wait_load_qkvogn_s_weight")
        self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
        cuda_hook_time_end("wait_load_qkvogn_s_weight")

        cuda_hook_time("wait_experts")
        self.cmv.wait_load_into_gpu(replica_uuid1)
        cuda_hook_time_end("wait_experts")


        cuda_hook_time("gpu_experts")
        if gpu_expert_ids:
            logger.debug(f"  Computing {len(gpu_expert_ids)} experts on GPU...")
            _ = self.mlpm.experts_func(
                self.cmv.mlpm_ci, layer_idx=layer_idx,
                expert_idx_list=list(gpu_expert_ids),
                expert_indices_map={eid: expert_indices_map[eid] for eid in gpu_expert_ids},
                expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in gpu_expert_ids},
                flat_hidden_states=flat_hidden_states,
                flat_experts_weight=flat_experts_weight,
                idxs=idxs,
                final_hidden_states=expert_cache,
                device=device
            )
            # _ = self.mlpm.experts_func_gpu_einsum(
            #     self.cmv.mlpm_ci, layer_idx=layer_idx,
            #     expert_idx_list=list(gpu_expert_ids),
            #     expert_indices_map={eid: expert_indices_map[eid] for eid in gpu_expert_ids},
            #     expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in gpu_expert_ids},
            #     flat_hidden_states=flat_hidden_states,
            #     flat_experts_weight=flat_experts_weight,
            #     idxs=idxs,
            #     final_hidden_states=expert_cache
            # )
            # expert_out_map.update(gpu_expert_out)
        cuda_hook_time_end("gpu_experts")
        time_gpu_end = time.time()

        # cuda_hook_time("wait_cetm_experts")
        # result = self.cetm.get_result()
        # cuda_hook_time_end("wait_cetm_experts")
        # time_einsum_end = result.time_einsum_end
        # logger.debug(f"gpu end - einsum end = {(time_gpu_end - time_einsum_end)*1000:.1f}ms")


        # cuda_hook_time(f"waiting_meta_l{layer_idx}")
        # if layer_idx < self.mlpm.config.num_hidden_layers - 2:
        #     # 每层计算提前等待初始化好下一层的layer, 第一层已初始化好
        #     # self.cmv.imm_submit_meta_layer(layer_idx=layer_idx+1)
        #     self.cmv.imm_wait_meta_layer(layer_idx=layer_idx+2)
        # cuda_hook_time_end(f"waiting_meta_l{layer_idx}")

        layer_output = expert_cache.view(*orig_shape) + y

        cuda_hook_time_end(f"layer_moe_generate_{layer_idx}")
        return layer_output
    
    def layer_moe_dgenerate(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ):
    

        cuda_hook_time(f"layer_moe_dgenerate_{layer_idx}")

        device1_idx_int = int(self.device1.split(":")[1])

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
        
        
        # 获取每个 expert 的实际设备位置
        layer = self.cmv.mlpm_ci.model.layers[layer_idx]
        expert_actual_device_map = get_expert_device_distribution(layer)
        
        experts_gpu_list = [expert_id for expert_id, _ in sorted_experts_by_load if expert_actual_device_map.get(expert_id, "unknown") == str(self.device1)]
        experts_cpu_list = [expert_id for expert_id, _ in sorted_experts_by_load if expert_actual_device_map.get(expert_id, "unknown") != str(self.device1)]
        
        logger.info(f"\nLayer {layer_idx} Expert Device Distribution:")
        logger.info(f"  Active experts: {num_experts_total} (out of {num_experts} total)")
        logger.info(f"\n  Detailed Expert Distribution:")
        logger.info(f"  {'Expert ID':<10} | {'Tokens':<10} | {'Actual Device':<15}")
        logger.info(f"  {'-'*70}")
        for expert_id, token_count in sorted_experts_by_load:
            actual_device = expert_actual_device_map.get(expert_id, "unknown")
            logger.info(f"  {expert_id:<10} | {token_count:<10} |  {actual_device:<15}")
        logger.info(f"{'='*60}\n")
        
        logger.info(f"experts_gpu_list: {experts_gpu_list}")
        logger.info(f"experts_cpu_list: {experts_cpu_list}")
        logger.info(f"expert_actual_device_map {expert_actual_device_map}")

        cuda_hook_time_end("experts_map_get")

        expert_cache = torch.zeros_like(flat_hidden_states)

        cuda_hook_time("cpu_experts_submit")
        expert_cache = torch.zeros_like(flat_hidden_states)
        # CPU experts - 传递索引信息，延迟创建 tensor maps
        if len(experts_cpu_list) > 0:
            logger.debug(f"\n  Computing {len(experts_cpu_list)} experts on CPU...")
            # 使用 CETM 在后台线程执行
            task = ExpertEinsumTask(
                layer_idx=layer_idx,
                    expert_idx_list=experts_cpu_list,
                expert_indices_map={eid: expert_indices_map[eid] for eid in experts_cpu_list},
                expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in experts_cpu_list},
                flat_hidden_states=flat_hidden_states,
                flat_experts_weight=flat_experts_weight,
                idxs=idxs,
                    final_hidden_states=expert_cache
                )
            self.cetm.submit(task)
        cuda_hook_time_end("cpu_experts_submit")

        cuda_hook_time("wait_cetm_experts")
        result = self.cetm.get_result()
        cuda_hook_time_end("wait_cetm_experts")

        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            hidden_states=hidden_states,
        )
        cuda_hook_time_end("gpu_sexperts")

        cuda_hook_time("gpu_experts")
        _ = self.mlpm.experts_func_gpu_einsum(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            expert_idx_list=experts_gpu_list,
            expert_indices_map={eid: expert_indices_map[eid] for eid in experts_gpu_list},
            expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in experts_gpu_list},
            flat_hidden_states=flat_hidden_states,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            final_hidden_states=expert_cache
        )
        cuda_hook_time_end("gpu_experts")

        layer_output = expert_cache.view(*orig_shape) + y

        cuda_hook_time_end(f"layer_moe_dgenerate_{layer_idx}")
        return hidden_states
