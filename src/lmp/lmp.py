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

        device1 = "cuda:1"
        device2 = "cuda:2"
        device_list = [device1, device2]
        self.device1 = device_list[0]
        self.device_list = device_list

        mlpm = MLPModuleWrapper(model_name_type, model_path)
        self.mlpm  = mlpm
        
        hmv = HostMemoryView(self.mlpm)
        cmv = CudaMemoryView(self.mlpm, device_list)
        self.hmv = hmv
        self.cmv = cmv

        cetm = CETM(self.mlpm, self.hmv, num_workers=1)  # 默认1个worker，可通过参数调整
        self.cetm = cetm
        self.cetm.start()  # 启动工作线程

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
    # 多GPU
    @torch.no_grad()
    def test_generate_multi_device_layer(self):
        cuda_hook_time("generate_input_ids")
        batch_size = 32
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
        past_key_value = DynamicCache()
        past_key_values_length = past_key_value.get_usable_length(seq_len)
        cuda_hook_time_end("init_cache")

        cuda_hook("init_meta")
        self.cmv.start_init_meta_model(hmv=self.hmv)
        cuda_hook_end("init_meta")

        cuda_hook_time("init_weights")
        self.cmv.load_general_and_init()
        self.cmv.init_load_qkvogn_es_weight(layer_idx=0)
        cuda_hook_time_end("init_weights")

        cuda_hook_time("copy_emodel")
        model_cpy = copy.deepcopy(self.cmv.mlpm_ci)
        cuda_hook_time_end("copy_emodel")

        cuda_hook_time("init_hmv")
        self.hmv.mlpm_hi = model_cpy
        self.mlpm.restore_hm_state_dict2model(self.hmv.hm_state_dict, self.hmv.mlpm_hi)
        cuda_hook_time_end("init_hmv")

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


        self.num_experts_on_cpu_ratio = 0.3
        cuda_hook_time("prefill_layer")
        ghidden_states = inputs_tokens
        for layer_idx in range(self.mlpm.config.num_hidden_layers):
            logger.debug(f"-------------------------------- start prefill layer {layer_idx} --------------------------------")

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
                # self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1,  device=device1)
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
                ghidden_states = self.layer_moe_generate_multi_device(layer_idx=layer_idx, hidden_states=ghidden_states)
                # ghidden_states = self.layer_moe_generate(layer_idx=layer_idx, hidden_states=ghidden_states)
            ghidden_states = ghidden_states + residual
            logger.debug(f"-------------------------------- end prefill layer {layer_idx} --------------------------------")            
        cuda_hook_time_end("prefill_layer")

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
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # should multi
        cuda_hook_time("async_load_ce")
        self.cmv.async_load_experts_decode_cpu_weight_multi_device()
        cuda_hook_time_end("async_load_ce")

        num_step=5

        for i in range(num_step):
            cuda_hook_time("decode_layer")
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
            # decode
            for layer_idx in range(0, self.mlpm.config.num_hidden_layers):
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

                if layer_idx == 0:
                    cuda_hook_time("dense_mlp")
                    ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=0, hidden_states=ghidden_states)
                    logger.debug(f"ghidden_states after dense_mlp_func shape: {ghidden_states.shape}")
                    cuda_hook_time_end("dense_mlp")
                else:
                    ghidden_states = self.layer_moe_dgenerate_multi_device(layer_idx=layer_idx, hidden_states=ghidden_states)
                ghidden_states = ghidden_states + residual

                logger.debug(f"-------------------------------- end decode layer {layer_idx} --------------------------------")

            torch.cuda.synchronize()
            cuda_hook_time_end("decode_layer")
        cuda_hook_time("async_wait_layer_loaded_to_gpu")
        self.cmv.wait_load_experts_decode_cpu_weight_multi_device()
        cuda_hook_time_end("async_wait_layer_loaded_to_gpu")
    # 单GPU
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
        model_cpy = copy.deepcopy(self.cmv.mlpm_ci)
        cuda_hook_time_end("copy_emodel")
        # cuda_hook_time("wait_all_meta")
        # self.cmv.imm.wait_all()
        # cuda_hook_time_end("wait_all_meta")
        cuda_hook_time("init_hmv")
        self.hmv.mlpm_hi = model_cpy
        self.mlpm.restore_hm_state_dict2model(self.hmv.hm_state_dict, self.hmv.mlpm_hi)
        cuda_hook_time_end("init_hmv")

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
                    # self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1,  device=device1)
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        
        cuda_hook_time("async_load_ce")
        self.cmv.async_load_experts_decode_cpu_weight()
        cuda_hook_time_end("async_load_ce")
        num_step=2
        for i in range(num_step):
            cuda_hook_time("decode_layer")
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
            # decode
            for layer_idx in range(0, self.mlpm.config.num_hidden_layers):
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

                if layer_idx == 0:
                    cuda_hook_time("dense_mlp")
                    ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=0, hidden_states=ghidden_states)
                    logger.debug(f"ghidden_states after dense_mlp_func shape: {ghidden_states.shape}")
                    cuda_hook_time_end("dense_mlp")
                else:
                    ghidden_states = self.layer_moe_dgenerate(layer_idx=layer_idx, hidden_states=ghidden_states)
                ghidden_states = ghidden_states + residual

                logger.debug(f"-------------------------------- end decode layer {layer_idx} --------------------------------")

            torch.cuda.synchronize()
            cuda_hook_time_end("decode_layer")
        cuda_hook_time("async_wait_layer_loaded_to_gpu")
        self.cmv.async_wait_layer_loaded_to_gpu()
        cuda_hook_time_end("async_wait_layer_loaded_to_gpu")
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

        # start load before self_attn
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
            # _ = self.mlpm.experts_func(
            #     self.cmv.mlpm_ci, layer_idx=layer_idx,
            #     expert_idx_list=list(gpu_expert_ids),
            #     expert_indices_map={eid: expert_indices_map[eid] for eid in gpu_expert_ids},
            #     expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in gpu_expert_ids},
            #     flat_hidden_states=flat_hidden_states,
            #     flat_experts_weight=flat_experts_weight,
            #     idxs=idxs,
            #     final_hidden_states=expert_cache,
            #     device=device
            # )
            _ = self.mlpm.experts_func_gpu_einsum(
                self.cmv.mlpm_ci, layer_idx=layer_idx,
                expert_idx_list=list(gpu_expert_ids),
                expert_indices_map={eid: expert_indices_map[eid] for eid in gpu_expert_ids},
                expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in gpu_expert_ids},
                flat_hidden_states=flat_hidden_states,
                flat_experts_weight=flat_experts_weight,
                idxs=idxs,
                final_hidden_states=expert_cache
            )
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

    def layer_moe_generate_multi_device(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ):
        cuda_hook_time(f"layer_moe_generate_multi_device_{layer_idx}")

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
        
        # Step 8: 根据token数量分配CPU/GPU experts，并将GPU专家平均分配到多个设备
        sorted_experts_by_load = sorted(expert_token_counts_list, key=lambda x: x[1])
        num_experts_total = len(sorted_experts_by_load)
        
        # 使用固定值
        num_experts_on_cpu = int(num_experts_total * self.num_experts_on_cpu_ratio)
        
        cpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[:num_experts_on_cpu])
        gpu_experts_list = sorted_experts_by_load[num_experts_on_cpu:]  # GPU 专家列表（按 token 数量排序）
        
        # 将 GPU 专家平均分配到多个设备，同时尽量平衡每个设备的 token 数量
        # 使用贪心算法：按 token 数量从大到小排序，每次分配给当前 token 数量最少的设备
        device_expert_map = {i: [] for i in range(num_device)}  # {device_idx: [expert_id, ...]}
        device_token_counts = [0] * num_device  # 每个设备的 token 总数
        
        # 按 token 数量从大到小排序，优先分配大负载的 expert
        gpu_experts_sorted = sorted(gpu_experts_list, key=lambda x: x[1], reverse=True)
        
        for expert_id, token_count in gpu_experts_sorted:
            # 找到当前 token 数量最少的设备
            min_device_idx = min(range(num_device), key=lambda i: device_token_counts[i])
            device_expert_map[min_device_idx].append(expert_id)
            device_token_counts[min_device_idx] += token_count
        
        # 构建每个设备的 expert ID 集合
        gpu_expert_ids_by_device = {
            device_idx: set(expert_ids) 
            for device_idx, expert_ids in device_expert_map.items()
        }
        
        # 打印调试信息
        cpu_ratio = num_experts_on_cpu / num_experts_total if num_experts_total > 0 else 0
        logger.debug(f"\nExpert Token Distribution & Multi-Device Allocation:")
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
                for device_idx, expert_set in gpu_expert_ids_by_device.items():
                    if expert_id in expert_set:
                        device = f"GPU{device_idx}({self.device_list[device_idx]})"
                        break
                if device is None:
                    device = "Unknown"
            logger.debug(f"  Expert {expert_id:2d} | {token_count:6d} | {device}")
        
        logger.debug(f"\n  Device Token Distribution:")
        logger.debug(f"  CPU: {total_tokens_cpu:6d} tokens")
        for device_idx in range(num_device):
            device_tokens = device_token_counts[device_idx]
            device_name = self.device_list[device_idx]
            logger.debug(f"  {device_name}: {device_tokens:6d} tokens ({len(device_expert_map[device_idx])} experts)")
        logger.debug(f"  Total GPU: {total_tokens_gpu:6d} tokens")
        logger.debug(f"{'='*60}\n")
        
        cuda_hook_time_end("experts_map_get")

        # Step 9: 为每个GPU设备分配和加载专家
        cuda_hook_time("allocate_experts_cuda_memory_and_restore_model_multi_device")
        device_replica_uuids = {}  # {device_idx: replica_uuid}
        device_state_dicts = {}  # {device_idx: state_dict}
        
        # 为每个设备加载对应的专家
        for device_idx in range(num_device):
            device = self.device_list[device_idx]
            device_idx_int = int(device.split(":")[1])
            device_expert_ids = gpu_expert_ids_by_device[device_idx]
            
            if device_expert_ids:
                gpu_expert_names = self.mlpm.get_experts_names(
                    layer_idx=layer_idx, 
                    expert_idx_list=list(device_expert_ids)
                )
                gpu_expert_names = gpu_expert_names
                
                ret, replica_uuid, state_dict = \
                    self.cmv.allocate_cuda_memory_and_load_into_gpu(
                        gpu_expert_names, device_index_int=device_idx_int
                    )
                device_replica_uuids[device_idx] = replica_uuid
                device_state_dicts[device_idx] = state_dict
                
                # 恢复模型状态到对应设备
                # restore2model 会将权重设置到对应的设备上
                self.cmv.restore2model(state_dict, self.cmv.mlpm_ci)
        
        cuda_hook_time_end("allocate_experts_cuda_memory_and_restore_model_multi_device")

        # Step 10: 提交CPU专家执行
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

        # Step 11: 执行shared experts（在第一个设备上）
        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            hidden_states=hidden_states,
        )
        cuda_hook_time_end("gpu_sexperts")

        # Step 12: 等待load_qkvogn_s加载完成
        cuda_hook_time("wait_load_qkvogn_s_weight")
        self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
        cuda_hook_time_end("wait_load_qkvogn_s_weight")

        # Step 13: 等待所有设备的专家加载完成
        cuda_hook_time("wait_experts_multi_device")
        for device_idx, replica_uuid in device_replica_uuids.items():
            self.cmv.wait_load_into_gpu(replica_uuid)
        cuda_hook_time_end("wait_experts_multi_device")

        # Step 14: 在每个GPU设备上执行对应的专家计算
        cuda_hook_time("gpu_experts_multi_device")
        
        # 确定 expert_cache 所在的设备（通常是第一个设备）
        main_device = self.device_list[0]
        expert_cache = expert_cache.to(main_device)
        
        # 为每个设备准备数据并执行
        for device_idx in range(num_device - 1, -1, -1):
            device = self.device_list[device_idx]
            device_expert_ids = gpu_expert_ids_by_device[device_idx]
            
            if device_expert_ids:
                logger.debug(f"  Computing {len(device_expert_ids)} experts on {device}...")
                
                # 使用设备上下文确保在正确的设备上执行
                with torch.cuda.device(device):
                    # 将相关参数移动到指定设备进行计算
                    device_flat_hidden_states = flat_hidden_states.to(device, non_blocking=True)
                    device_flat_experts_weight = flat_experts_weight.to(device, non_blocking=True)
                    device_idxs = idxs.to(device, non_blocking=True)
                    
                    # 不移动 expert_token_indices_map，改为从 idxs 重新获取（在原地计算）
                    # 这样可以避免多次小 tensor 的传输，减少开销
                    device_token_idxs = device_idxs // self.mlpm.config.num_experts_per_tok  # 恢复到原始 token 索引
                    device_expert_token_indices_map = {
                        eid: device_token_idxs[expert_indices_map[eid][0]:expert_indices_map[eid][1]]
                        for eid in device_expert_ids
                    }
                    
                    # 处理 expert_cache：如果是主设备直接使用，否则创建临时cache
                    if device == main_device:
                        device_expert_cache = expert_cache
                    else:
                        # 为其他设备创建临时cache，最后累加到主设备
                        device_expert_cache = torch.zeros_like(device_flat_hidden_states)
                        
                        # 执行专家计算
                        # 注意：mlpm_ci 的权重应该已经通过 restore2model 设置到对应设备上
                        device_expert_cache = self.mlpm.experts_func_gpu_einsum(
                            self.cmv.mlpm_ci, layer_idx=layer_idx,
                            expert_idx_list=list(device_expert_ids),
                            expert_indices_map={eid: expert_indices_map[eid] for eid in device_expert_ids},
                            expert_token_indices_map=device_expert_token_indices_map,
                            flat_hidden_states=device_flat_hidden_states,
                            flat_experts_weight=device_flat_experts_weight,
                            idxs=device_idxs,
                            final_hidden_states=device_expert_cache
                        )
                        
                        # 如果 device_expert_cache 不在主设备上，需要将结果传回主设备并累加
                        if device != main_device:
                            expert_cache.add_(device_expert_cache.to(main_device, non_blocking=True))
                        # 如果 device_expert_cache 就是 expert_cache（同一设备），则已经直接修改了
        
        cuda_hook_time_end("gpu_experts_multi_device")

        # Step 15: 合并结果
        layer_output = expert_cache.view(*orig_shape) + y

        cuda_hook_time_end(f"layer_moe_generate_multi_device_{layer_idx}")
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
        
        if self.cmv.check_async_load_experts_decode_cpu_weight(layer_idx=layer_idx):
            logger.info(f"using loaded check layer: True")
        else:
            logger.info(f"using loaded check layer: False")

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
        
        logger.info(f"experts_gpu_list: {experts_gpu_list} num: {len(experts_gpu_list)}")
        logger.info(f"experts_cpu_list: {experts_cpu_list} num: {len(experts_cpu_list)}")
        logger.info(f"expert_actual_device_map {expert_actual_device_map}")

        cuda_hook_time_end("experts_map_get")

        expert_cache = torch.zeros_like(flat_hidden_states)

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

        cuda_hook_time("cpu_experts_submit")
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

        # cuda_hook_time("cpu_experts")
        # _ = self.mlpm.experts_func(
        #     self.hmv.mlpm_hi, layer_idx=layer_idx,
        #     expert_idx_list=list(experts_cpu_list),
        #     expert_indices_map={eid: expert_indices_map[eid] for eid in experts_cpu_list},
        #     expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in experts_cpu_list},
        #     flat_hidden_states=flat_hidden_states,
        #     flat_experts_weight=flat_experts_weight,
        #     idxs=idxs,
        #     final_hidden_states=expert_cache,
        #     device="cpu"
        # )
        # cuda_hook_time_end("cpu_experts")

        layer_output = expert_cache.view(*orig_shape) + y

        cuda_hook_time_end(f"layer_moe_dgenerate_{layer_idx}")
        return hidden_states

    def layer_moe_dgenerate_multi_device(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ):
        """多设备版本的decode阶段MoE层生成，基于实际设备位置分配专家，支持多GPU"""
        cuda_hook_time(f"layer_moe_dgenerate_multi_device_{layer_idx}")

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
        
        # Step 8: 根据实际设备位置分配CPU/GPU experts，支持多GPU
        sorted_experts_by_load = sorted(expert_token_counts_list, key=lambda x: x[1])
        num_experts_total = len(sorted_experts_by_load)
        
        # if self.cmv.check_async_load_experts_decode_cpu_weight(layer_idx=layer_idx):
        #     logger.info(f"using loaded check layer: True")
        # else:
        #     logger.info(f"using loaded check layer: False")

        # 获取每个 expert 的实际设备位置
        layer = self.cmv.mlpm_ci.model.layers[layer_idx]
        expert_actual_device_map = get_expert_device_distribution(layer)
        
        num_device = len(self.device_list)
        
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
        
        logger.info(f"\nLayer {layer_idx} Expert Device Distribution (Multi-Device):")
        logger.info(f"  Active experts: {num_experts_total} (out of {num_experts} total)")
        logger.info(f"\n  Detailed Expert Distribution:")
        logger.info(f"  {'Expert ID':<10} | {'Tokens':<10} | {'Actual Device':<15}")
        logger.info(f"  {'-'*70}")
        for expert_id, token_count in sorted_experts_by_load:
            actual_device = expert_actual_device_map.get(expert_id, "unknown")
            logger.info(f"  {expert_id:<10} | {token_count:<10} |  {actual_device:<15}")
        logger.info(f"{'='*60}\n")
        
        logger.info(f"experts_cpu_list: {experts_cpu_list} num: {len(experts_cpu_list)}")
        for device_idx in range(num_device):
            device_expert_ids = gpu_expert_ids_by_device[device_idx]
            logger.info(f"experts_gpu_list[{device_idx}]({self.device_list[device_idx]}): {device_expert_ids} num: {len(device_expert_ids)}")
        logger.info(f"expert_actual_device_map {expert_actual_device_map}")

        cuda_hook_time_end("experts_map_get")

        expert_cache = torch.zeros_like(flat_hidden_states)

        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            hidden_states=hidden_states,
        )
        cuda_hook_time_end("gpu_sexperts")

        # Step 10: 提交CPU专家执行
        cuda_hook_time("cpu_experts_submit")
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
        if len(experts_cpu_list) > 0:
            result = self.cetm.get_result()
        cuda_hook_time_end("wait_cetm_experts")

        # Step 12: 在每个GPU设备上执行对应的专家计算
        cuda_hook_time("gpu_experts_multi_device")
        
        # 确定 expert_cache 所在的设备（通常是第一个设备）
        main_device = self.device_list[0]
        expert_cache = expert_cache.to(main_device)
        
        # 为每个设备准备数据并执行
        for device_idx in range(num_device - 1, -1, -1):
            device = self.device_list[device_idx]
            device_expert_ids = gpu_expert_ids_by_device[device_idx]
            
            if device_expert_ids:
                logger.debug(f"  Computing {len(device_expert_ids)} experts on {device}...")
                
                with torch.cuda.device(device):
                    # 将相关参数移动到指定设备进行计算
                    device_flat_hidden_states = flat_hidden_states.to(device, non_blocking=True)
                    device_flat_experts_weight = flat_experts_weight.to(device, non_blocking=True)
                    device_idxs = idxs.to(device, non_blocking=True)
                    
                    # 不移动 expert_token_indices_map，改为从 idxs 重新获取（在原地计算）
                    device_token_idxs = device_idxs // self.mlpm.config.num_experts_per_tok  # 恢复到原始 token 索引
                    device_expert_token_indices_map = {
                        eid: device_token_idxs[expert_indices_map[eid][0]:expert_indices_map[eid][1]]
                        for eid in device_expert_ids
                    }
                    
                    # 处理 expert_cache：如果是主设备直接使用，否则创建临时cache
                    if device == main_device:
                        device_expert_cache = expert_cache
                    else:
                        # 为其他设备创建临时cache，最后累加到主设备
                        device_expert_cache = torch.zeros_like(device_flat_hidden_states)
                    
                    # 执行专家计算
                    device_expert_cache = self.mlpm.experts_func(
                        self.cmv.mlpm_ci, layer_idx=layer_idx,
                        expert_idx_list=list(device_expert_ids),
                        expert_indices_map={eid: expert_indices_map[eid] for eid in device_expert_ids},
                        expert_token_indices_map=device_expert_token_indices_map,
                        flat_hidden_states=device_flat_hidden_states,
                        flat_experts_weight=device_flat_experts_weight,
                        idxs=device_idxs,
                        final_hidden_states=device_expert_cache,
                    device=device
                )
                    # 执行专家计算
                    # device_expert_cache = self.mlpm.experts_func_gpu_einsum(
                    #     self.cmv.mlpm_ci, layer_idx=layer_idx,
                    #     expert_idx_list=list(device_expert_ids),
                    #     expert_indices_map={eid: expert_indices_map[eid] for eid in device_expert_ids},
                    #     expert_token_indices_map=device_expert_token_indices_map,
                    #     flat_hidden_states=device_flat_hidden_states,
                    #     flat_experts_weight=device_flat_experts_weight,
                    #     idxs=device_idxs,
                    #     final_hidden_states=device_expert_cache
                    # )
                    
                    # 如果 device_expert_cache 不在主设备上，需要将结果传回主设备并累加
                    if device != main_device:
                        expert_cache.add_(device_expert_cache.to(main_device, non_blocking=True))
        
        cuda_hook_time_end("gpu_experts_multi_device")

        layer_output = expert_cache.view(*orig_shape) + y

        cuda_hook_time_end(f"layer_moe_dgenerate_multi_device_{layer_idx}")
        return layer_output

    def init_mp_process(self):
        from lmp.cpu_thread_manager_mp import CPUExpertsManagerMP
        from lmp.device_mp import DeviceMP
        from lmp.init_meta_manager_mp_shared import InitMetaManagerMPShared
        cuda_hook_time("init_mp_process")
        self.cpu_thread_manager_mp = CPUExpertsManagerMP(num_workers=1, model_path=self.mlpm.model_path, model_name_type=self.mlpm.model_name_type)
        self.cpu_thread_manager_mp.start()

        # self.imm_mp = InitMetaManagerMPShared(num_processes=1)
        # self.imm_mp.start()
        # self.dp = DeviceMP(num_processes=len(self.device_list))
        # self.dp.start()
        cuda_hook_time_end("init_mp_process")
    def mp_stop(self):
        self.cpu_thread_manager_mp.stop()

    @torch.no_grad()
    def test_mp_prefill_generate(self):
        cuda_hook_time("generate_input_ids")
        batch_size = 32
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
        past_key_value = DynamicCache()
        past_key_values_length = past_key_value.get_usable_length(seq_len)
        cuda_hook_time_end("init_cache")

        cuda_hook("init_meta")
        self.cmv.start_init_meta_model(hmv=self.hmv)
        cuda_hook_end("init_meta")

        cuda_hook_time("init_meta_layer")
        # self.mlpm.init_set_layer_func(layer_idx=0, config=self.mlpm.config, model=self.cmv.mlpm_ci)
        # self.mlpm.init_set_layer_func(layer_idx=1, config=self.mlpm.config, model=self.cmv.mlpm_ci)
        # self.mlpm.init_set_layer_func(layer_idx=2, config=self.mlpm.config, model=self.cmv.mlpm_ci)
        # self.imm_mp.submit_layer(layer_idx=0, init_func=self.mlpm.init_layer_func, config=self.mlpm.config)
        # self.imm_mp.submit_layer(layer_idx=1, init_func=self.mlpm.init_layer_func, config=self.mlpm.config)
        # self.imm_mp.submit_layer(layer_idx=2, init_func=self.mlpm.init_layer_func, config=self.mlpm.config)
        # layer0 = self.imm_mp.wait_layer(layer_idx=0)
        # layer1 = self.imm_mp.wait_layer(layer_idx=1)
        # layer2 = self.imm_mp.wait_layer(layer_idx=2)
        # self.cmv.mlpm_ci.model.layers[0] = layer0
        # self.cmv.mlpm_ci.model.layers[1] = layer1
        # self.cmv.mlpm_ci.model.layers[2] = layer2

        cuda_hook_time_end("init_meta_layer")

        cuda_hook_time("init_weights")
        self.cmv.load_general_and_init()
        self.cmv.init_load_qkvogn_es_weight(layer_idx=0)
        cuda_hook_time_end("init_weights")

        cuda_hook_time("copy_emodel")
        model_cpy = copy.deepcopy(self.cmv.mlpm_ci)
        cuda_hook_time_end("copy_emodel")

        # cuda_hook_time("init_hmv")
        # self.hmv.mlpm_hi = model_cpy
        # self.mlpm.restore_hm_state_dict2model(self.hmv.hm_state_dict, self.hmv.mlpm_hi)
        # cuda_hook_time_end("init_hmv")

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

        # cuda_hook_time("load_all_qkvogn_s")
        # for layer_idx in range(0, self.mlpm.config.num_hidden_layers):
        #     if layer_idx < self.mlpm.config.num_hidden_layers-1:
        #         cuda_hook_time("init_set_layer_func")
        #         self.mlpm.init_set_layer_func(layer_idx=layer_idx+1, config=self.mlpm.config, model=self.cmv.mlpm_ci)
        #         cuda_hook_time_end("init_set_layer_func")

        #         cuda_hook_time(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")
        #         self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1, device=self.device1)
        #         cuda_hook_time_end(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")

        #         cuda_hook_time("wait_load_qkvogn_s_weight")
        #         self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
        #         cuda_hook_time_end("wait_load_qkvogn_s_weight")
        # cuda_hook_time_end("load_all_qkvogn_s")
        cuda_hook_time("prefill")
        self.num_experts_on_cpu_ratio = 0.5
        
        ghidden_states = inputs_tokens
        for layer_idx in range(self.mlpm.config.num_hidden_layers):
            cuda_hook_time("prefill_layer")
            logger.debug(f"-------------------------------- start prefill layer {layer_idx} --------------------------------")
            
            if layer_idx < self.mlpm.config.num_hidden_layers-1:
                cuda_hook_time(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")
                self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1, device=self.device1)
                cuda_hook_time_end(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")

            # if layer_idx < self.mlpm.config.num_hidden_layers-2:
            #     cuda_hook_time(f"init_set_layer_func_l_{layer_idx+2}")
            #     self.imm_mp.submit_layer(layer_idx=layer_idx+2, init_func=self.mlpm.init_layer_func, config=self.mlpm.config)
            #     cuda_hook_time_end(f"init_set_layer_func_l_{layer_idx+2}")
                # cuda_hook_time("wait_load_qkvogn_s_weight")
                # self.cmv.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
                # cuda_hook_time_end("wait_load_qkvogn_s_weight")

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
                # self.cmv.start_load_qkvogn_s_weight(layer_idx=layer_idx+1,  device=device1)
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
                ghidden_states = self.layer_moe_generate_mp(layer_idx=layer_idx, hidden_states=ghidden_states)
                # ghidden_states = self.layer_moe_generate(layer_idx=layer_idx, hidden_states=ghidden_states)
            ghidden_states = ghidden_states + residual

            # if layer_idx < self.mlpm.config.num_hidden_layers-2:
            #     cuda_hook_time("wait_init_set_layer_func")
            #     layer = self.imm_mp.wait_layer(layer_idx=layer_idx+2)
            #     self.cmv.mlpm_ci.model.layers[layer_idx+2] = layer
            #     cuda_hook_time_end("wait_init_set_layer_func")
            cuda_hook_time_end("prefill_layer")
            logger.debug(f"-------------------------------- end prefill layer {layer_idx} --------------------------------")            
        cuda_hook_time_end("prefill")
        torch.cuda.synchronize()

    @torch.no_grad()
    def layer_moe_generate_mp(self, layer_idx: int, hidden_states: torch.Tensor):
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