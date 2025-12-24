import os
from threading import get_ident
from turtle import position
from lmp.cpu_thread_manager import CETM
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

        sllmtm = SLLMTM(self.cmv, num_workers=1)  # SLLM 线程管理器，用于异步加载
        self.sllmtm = sllmtm
        self.sllmtm.start()  # 启动工作线程

        device1 = "cuda:1"
        self.device1 = device1
        
        # CPU专家数量：使用固定值 0.6 * total + 2
        # stream = torch.cuda.Stream(device=device1)
        # with torch.cuda.stream(stream):
        # self.test_generate_single_step()
    # load qkvo gate layer norm weight, except experts and shared experts
    # support deepseek here
    def init_qkvogn_weight(self):
        device1_idx_int = int(self.device1.split(":")[1])
        time_start_init_qkvogn_weight = time.time()
        cuda_hook("init qkvogn weight")
        layer_num = self.mlpm.config.num_hidden_layers
        for layer_idx in range(layer_num):
            tensor_al1_names = self.mlpm.get_attention_names(layer_idx=layer_idx)
            tensor_ln1_names = self.mlpm.get_layernorm_names(layer_idx=layer_idx)
            tensor_gate_names = self.mlpm.get_gate_names(layer_idx=layer_idx)
            tensor_index_names = tensor_al1_names + tensor_ln1_names + tensor_gate_names
            self.cmv.allocate_cuda_memory_load_wait(tensor_index_names, device_index_int=device1_idx_int)
        logger.debug(f"init qkvogn weight time: {time.time() - time_start_init_qkvogn_weight}")
        cuda_hook_end("init qkvogn weight")

    def load_qkvogn_s_weight(self, layer_idx: int):
        device1_idx_int = int(self.device1.split(":")[1])
        cuda_hook_time(f"load_qkvogns_weight_l_{layer_idx}")
        tensor_al1_names = self.mlpm.get_attention_names(layer_idx=layer_idx)
        tensor_ln1_names = self.mlpm.get_layernorm_names(layer_idx=layer_idx)
        tensor_gate_names = self.mlpm.get_gate_names(layer_idx=layer_idx)
        tensor_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_idx)
        tensor_index_names = tensor_al1_names + tensor_ln1_names + tensor_gate_names + tensor_shared_expert_names
        self.cmv.allocate_cuda_memory_load_wait(tensor_index_names, device_index_int=device1_idx_int)
        cuda_hook_time_end(f"load_qkvogns_weight_l_{layer_idx}")
        
    def start_load_qkvogn_s_weight(self, layer_idx: int, device: str):
        """
        异步发起加载请求，使用 SLLMTM 线程管理器
        
        Args:
            layer_idx: 层索引
            
        Returns:
            无（异步执行，通过 get_load_result 获取结果）
        """
        if layer_idx >= self.mlpm.config.num_hidden_layers:
            return
        device_idx_int = int(device.split(":")[1])
        cuda_hook_time(f"start_load_qkvogn_s_weight_l_{layer_idx}")
        tensor_al1_names = self.mlpm.get_attention_names(layer_idx=layer_idx)
        tensor_ln1_names = self.mlpm.get_layernorm_names(layer_idx=layer_idx)
        tensor_gate_names = self.mlpm.get_gate_names(layer_idx=layer_idx)
        tensor_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_idx)
        tensor_index_names = tensor_al1_names + tensor_ln1_names + tensor_gate_names + tensor_shared_expert_names
        
        # 使用 SLLMTM 异步提交加载任务
        self.sllmtm.submit_load(
            layer_idx=layer_idx,
            tensor_index_names=tensor_index_names,
            device_index_int=device_idx_int
        )
        cuda_hook_time_end(f"start_load_qkvogn_s_weight_l_{layer_idx}")
    def wait_load_qkvogn_s_weight(self, layer_idx):
        if layer_idx >= self.mlpm.config.num_hidden_layers:
            return
        return self.sllmtm.get_result_wait()

    def test_generate_single_layer(
        self,
    ):
        batch_size = 64
        seq_len = 64
        dtype = self.mlpm.config.torch_dtype
        hidden_size = self.mlpm.config.hidden_size
        device = "cuda:1"
        device_idx_int = int(device.split(":")[1])
        inputs_tokens = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

        tokenizer=AutoTokenizer.from_pretrained(self.mlpm.model_abs_path, trust_remote_code=True)
        inputs_ids = generate_input_ids(tokenizer, batch_size, seq_len, device)
        inputs_tokens = self.cmv.mlpm_ci.model.embed_tokens(inputs_ids)

        layer_first = 1
        tensor_al1_names = self.mlpm.get_attention_names(layer_idx=layer_first)
        tensor_ln1_names = self.mlpm.get_layernorm_names(layer_idx=layer_first)
        tensor_gate_names = self.mlpm.get_gate_names(layer_idx=layer_first)
        tensor_index_names = tensor_al1_names + tensor_ln1_names + tensor_gate_names
        
        self.cmv.allocate_cuda_memory_load_wait(tensor_index_names, device_index_int=device_idx_int)

        with torch.no_grad():   
            # logger.debug_layer_parameters(self.cmv.mlpm_ci.model.layers[layer_first])

            cuda_hook("prepare")
            time_start_prepare = time.time()
            cuda_hook("gate")
            # Step 1: 通过 gate 函数获取每个 token 选择的 top-k 专家和权重
            topk_idx, topk_weight, aux_loss = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_first, inputs_tokens)
            logger.debug(f"gate time: {time.time() - time_start_prepare}")
            cuda_hook_end("gate")
            # topk_idx: [batch_size, seq_len, num_experts_per_tok]
            # topk_weight: [batch_size, seq_len, num_experts_per_tok]

            # Step 2: 展平 expert indices 和 weights
            flat_expert_indices = topk_idx.view(-1)      # [batch_size * seq_len * num_experts_per_tok]
            flat_experts_weight = topk_weight.view(-1, 1)  # [batch_size * seq_len * num_experts_per_tok, 1]
            
            # Step 3: 按 expert_id 排序，将相同专家的 token 聚集在一起
            idxs = flat_expert_indices.argsort()         # 排序后的索引
            
            # Step 4: 统计每个专家被分配的 token 数量
            tokens_per_expert = flat_expert_indices.bincount()  # [num_experts]
            
            # Step 5: 计算原始 token 的索引
            token_idxs = idxs // self.mlpm.config.num_experts_per_tok  # 恢复到原始 token 索引
            
            # Step 6: 展平 inputs_tokens（不复制数据，只是 view）
            # inputs_tokens: [batch_size, seq_len, hidden_dim]
            orig_shape = inputs_tokens.shape
            hidden_states = inputs_tokens.view(batch_size * seq_len, -1)  # [batch_size * seq_len, hidden_dim]
            
            # Step 7: 构建每个 expert 的 tokens 和 weights
            # 优化：避免预先创建 sorted_hidden_states，直接在循环中索引
            expert_tokens_map = {}
            expert_weights_map = {}
            expert_token_indices_map = {}  # 保存每个 expert 的 token 索引，用于 scatter
            
            start_idx = 0
            
            num_experts = self.mlpm.get_experts_num()
            
            for expert_id in range(num_experts):
                if expert_id >= len(tokens_per_expert) or tokens_per_expert[expert_id] == 0:
                    # 该专家没有被分配任何 token
                    continue
                
                end_idx = start_idx + tokens_per_expert[expert_id].item()
                
                # 获取该 expert 的索引范围
                expert_idxs = idxs[start_idx:end_idx]                      # 在排序后的位置
                expert_token_ids = token_idxs[start_idx:end_idx]           # 原始 token IDs
                
                # 直接从原始 hidden_states 索引，避免预先复制整个 sorted_hidden_states
                expert_tokens = hidden_states[expert_token_ids]            # [num_tokens, hidden_dim]
                expert_weights = flat_experts_weight[expert_idxs]          # [num_tokens, 1]
                
                expert_tokens_map[expert_id] = expert_tokens
                expert_weights_map[expert_id] = expert_weights
                expert_token_indices_map[expert_id] = expert_token_ids     # 保存用于 scatter
                
                start_idx = end_idx
            
            # 确保按 expert_id 排序（虽然上面循环已经是有序的，但为了明确性）
            expert_tokens_map = dict(sorted(expert_tokens_map.items()))
            expert_weights_map = dict(sorted(expert_weights_map.items()))
            expert_token_indices_map = dict(sorted(expert_token_indices_map.items()))
            
            # Step 8: 根据token数量分配CPU/GPU experts
            num_experts_cpu = 0.3  # CPU执行的expert比例
            
            # 按token数量排序expert（从少到多）
            expert_token_counts = {
                expert_id: tokens.shape[0] 
                for expert_id, tokens in expert_tokens_map.items()
            }
            sorted_experts_by_load = sorted(expert_token_counts.items(), key=lambda x: x[1])
            
            # 计算有多少个expert在CPU上执行
            num_experts_total = len(sorted_experts_by_load)
            num_experts_on_cpu = int(num_experts_total * num_experts_cpu)
            
            # 分配：token最少的expert放CPU，token多的放GPU
            cpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[:num_experts_on_cpu])
            gpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[num_experts_on_cpu:])
            
            # 打印调试信息
            logger.debug(f"\nExpert Token Distribution & Device Allocation:")
            logger.debug(f"  Total experts: {num_experts_total}")
            logger.debug(f"  CPU experts: {num_experts_on_cpu} ({num_experts_cpu*100:.0f}%)")
            logger.debug(f"  GPU experts: {num_experts_total - num_experts_on_cpu} ({(1-num_experts_cpu)*100:.0f}%)")
            logger.debug(f"\n  Expert ID | Tokens | Device")
            logger.debug(f"  {'-'*35}")
            
            total_tokens_cpu = 0
            total_tokens_gpu = 0
            for expert_id, token_count in sorted_experts_by_load:
                device = "CPU" if expert_id in cpu_expert_ids else "GPU"
                logger.debug(f"  Expert {expert_id:2d} | {token_count:6d} | {device}")
                if device == "CPU":
                    total_tokens_cpu += token_count
                else:
                    total_tokens_gpu += token_count
            
            logger.debug(f"\n  CPU total tokens: {total_tokens_cpu} ({total_tokens_cpu/(total_tokens_cpu+total_tokens_gpu)*100:.1f}%)")
            logger.debug(f"  GPU total tokens: {total_tokens_gpu} ({total_tokens_gpu/(total_tokens_cpu+total_tokens_gpu)*100:.1f}%)")
            
            # 分离CPU和GPU的expert maps
            cpu_expert_tokens_map = {eid: expert_tokens_map[eid] for eid in cpu_expert_ids}
            cpu_expert_weights_map = {eid: expert_weights_map[eid] for eid in cpu_expert_ids}
            
            gpu_expert_tokens_map = {eid: expert_tokens_map[eid] for eid in gpu_expert_ids}
            gpu_expert_weights_map = {eid: expert_weights_map[eid] for eid in gpu_expert_ids}
            logger.debug(f"prepare time: {time.time() - time_start_prepare}")
            cuda_hook_end("prepare")

            # Step 9: 分别在CPU和GPU上计算experts
            # 异步提交CPU，GPU先加载后执行
            expert_cache = torch.zeros_like(hidden_states)
            time_start_cpu = time.time()
            cuda_hook("cpu experts")
            # CPU experts
            if cpu_expert_tokens_map:
                logger.debug(f"\n  Computing {len(cpu_expert_tokens_map)} experts on CPU...")
                # 使用 CETM 在后台线程执行
                task = ExpertEinsumTask(
                    layer_idx=layer_first,
                    expert_idx_list=list(cpu_expert_ids),
                    expert_tokens_map=cpu_expert_tokens_map,
                    expert_weights_map=cpu_expert_weights_map,
                    expert_token_indices_map=expert_token_indices_map,
                    final_hidden_states=expert_cache
                )
                self.cetm.submit(task)
            cuda_hook_end("cpu experts")
            logger.debug(f"cpu compute time: {time.time() - time_start_cpu}")

            # time_start_allocate_shared_cuda_memory=time.time()
            # cuda_hook("allocate shared experts cuda memory and restore model")
            # gpu_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_first)
            # ret2, replica_uuid2, state_dict2 = \
            #     self.cmv.allocate_cuda_memory_and_load_into_gpu(
            #         gpu_shared_expert_names, device_index_int=device_idx_int)
            # self.cmv.restore2model(state_dict2, self.cmv.mlpm_ci)
            # logger.debug(f"allocate shared experts cuda memory and restore model time: {time.time() - time_start_allocate_shared_cuda_memory}")
            # cuda_hook_end("allocate shared experts cuda memory and restore model")

            time_start_allocate_experts_cuda_memory = time.time()
            cuda_hook("allocate experts cuda memory and restore model")
            gpu_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_first)
            gpu_expert_names = self.mlpm.get_experts_names(layer_idx=layer_first, expert_idx_list=list(gpu_expert_ids))
            gpu_expert_names = gpu_expert_names+gpu_shared_expert_names
            ret1, replica_uuid1, state_dict1 = \
                self.cmv.allocate_cuda_memory_and_load_into_gpu(
                    gpu_expert_names, device_index_int=device_idx_int)
            self.cmv.restore2model(state_dict1, self.cmv.mlpm_ci)
            logger.debug(f"allocate experts cuda memory and restore model time: {time.time() - time_start_allocate_experts_cuda_memory}")
            cuda_hook_end("allocate experts cuda memory and restore model")

            # time_start_wait=time.time()
            # cuda_hook("wait_sexperts")
            # self.cmv.wait_load_into_gpu(replica_uuid2)
            # cuda_hook_end("wait_sexperts")
            # logger.debug(f"wait_shared_experts time: {time.time() - time_start_wait}")

            


            time_start_wait=time.time()
            cuda_hook("wait_experts")
            self.cmv.wait_load_into_gpu(replica_uuid1)
            cuda_hook_end("wait_experts")
            logger.debug(f"wait_experts time: {time.time() - time_start_wait}")

            cuda_hook("gpu_sexperts")
            y = self.mlpm.shared_experts_func(
                self.cmv.mlpm_ci, layer_first,
                hidden_states=inputs_tokens,
            )
            cuda_hook_end("gpu_sexperts")
            
            time_start_gpu = time.time()
            # GPU experts  
            cuda_hook("gpu experts")
            if gpu_expert_tokens_map:
                logger.debug(f"  Computing {len(gpu_expert_tokens_map)} experts on GPU...")
                _ = self.mlpm.experts_func(
                    self.cmv.mlpm_ci, layer_first,
                    experts_tokens_map=gpu_expert_tokens_map, 
                    experts_weights_map=gpu_expert_weights_map,
                    expert_token_indices_map=expert_token_indices_map,
                    final_hidden_states=expert_cache,
                    device=device
                )
                # expert_out_map.update(gpu_expert_out)
            cuda_hook_end("gpu experts")
            logger.debug(f"gpu compute time: {time.time() - time_start_gpu}")

            result = self.cetm.get_result()
            _ = result.final_hidden_states

            layer_output = expert_cache.view(*orig_shape) + y
            logger.debug(f"\nFinal output shape: {layer_output.shape}")
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
        inputs_tokens = self.cmv.mlpm_ci.model.embed_tokens(inputs_ids)
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
        cuda_hook_time_end("init_cache")

        self.load_qkvogn_s_weight(layer_idx=0)
        cuda_hook_time("warm_up")
        num_warm_up = 3
        for i in range(num_warm_up):
            self.layer_warmup_generate(hidden_states=inputs_tokens)
            ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=0, hidden_states=inputs_tokens)
            ghidden_states = self.mlpm.paln_func(self.cmv.mlpm_ci, layer_idx=0, hidden_states=inputs_tokens)
        cuda_hook_time_end("warm_up")

        cuda_hook_time("multi_layer")
        # self.load_qkvogn_s_weight(layer_idx=0)
        num_step = 1
        self.num_experts_on_cpu_ratio = 0.5
        for i in range(num_step):
            ghidden_states = inputs_tokens
            for layer_idx in range(self.mlpm.config.num_hidden_layers):
                # 测试Deepseek 跳过dense测试
                logger.debug(f"-------------------------------- start layer {layer_idx} --------------------------------")
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
                    self.start_load_qkvogn_s_weight(layer_idx=layer_idx+1, device=device1)
                    ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=ghidden_states)
                    self.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
                    cuda_hook_time_end("dense_mlp")
                else:
                    ghidden_states = self.layer_moe_generate(layer_idx=layer_idx, hidden_states=ghidden_states)
                    # ghidden_states = self.layer_moe_generate(layer_idx=layer_idx, hidden_states=ghidden_states)
                ghidden_states = ghidden_states + residual
                logger.debug(f"-------------------------------- end layer {layer_idx} --------------------------------")
                # torch.cuda.synchronize(device=device1)
        cuda_hook_time_end("multi_layer")
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

        # 提交cpu 专家执行
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

        cuda_hook_time(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")
        self.start_load_qkvogn_s_weight(layer_idx=layer_idx+1, device=self.device1)
        cuda_hook_time_end(f"start_load_qkvogn_s_weight_l_{layer_idx+1}")

        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            hidden_states=hidden_states,
        )
        cuda_hook_time_end("gpu_sexperts")

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

        cuda_hook_time("wait_cetm_experts")
        result = self.cetm.get_result()
        cuda_hook_time_end("wait_cetm_experts")
        time_einsum_end = result.time_einsum_end
        logger.debug(f"gpu end - einsum end = {(time_gpu_end - time_einsum_end)*1000:.1f}ms")

        layer_output = expert_cache.view(*orig_shape) + y
        # 等待 load_qkvogn_s 加载完成
        cuda_hook_time("wait_load_qkvogn_s_weight")
        self.wait_load_qkvogn_s_weight(layer_idx=layer_idx+1)
        cuda_hook_time_end("wait_load_qkvogn_s_weight")

        cuda_hook_time_end(f"layer_moe_generate_{layer_idx}")
        return layer_output
    @torch.no_grad()
    def layer_warmup_generate(
        self,
        hidden_states: torch.Tensor,
    ):
        device1_idx_int = int(self.device1.split(":")[1])
        batch_size, seq_len = hidden_states.shape[:2]
        orig_shape = hidden_states.shape
        cuda_hook_time("warm_up_gate")
        # Step 1: 通过 gate 函数获取每个 token 选择的 top-k 专家和权重
        topk_idx, topk_weight, aux_loss = self.mlpm.gate_func(self.cmv.mlpm_ci, layer_idx=1, hidden_states=hidden_states)
        # Step 2: 展平 expert indices 和 weights
        flat_expert_indices = topk_idx.view(-1)
        flat_experts_weight = topk_weight.view(-1, 1)
        # Step 3: 按 expert_id 排序
        idxs = flat_expert_indices.argsort()
        # Step 4: 统计每个专家被分配的 token 数量
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # Step 5: 计算原始 token 的索引
        token_idxs = idxs // self.mlpm.config.num_experts_per_tok
        # Step 6: 展平 hidden_states
        flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)
        cuda_hook_time_end("warm_up_gate")

        # Step 7: 构建每个 expert 的索引信息
        expert_indices_map = {}
        expert_token_indices_map = {}
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
            prev_end = end_idx

        # 只使用有效的 experts（有 token 的）
        cpu_experts_id = [eid for eid in range(32) if eid in expert_indices_map]
        if not cpu_experts_id:
            cuda_hook_time_end("warm_up_cpu")
            return
        
        expert_cache = torch.zeros_like(flat_hidden_states)
        cuda_hook_time("warm_up_cpu")
        task = ExpertEinsumTask(
            layer_idx=1,
            expert_idx_list=cpu_experts_id,
            expert_indices_map={eid: expert_indices_map[eid] for eid in cpu_experts_id},
            expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in cpu_experts_id},
            flat_hidden_states=flat_hidden_states,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            final_hidden_states=expert_cache)
        self.cetm.submit(task)
        # 等待任务完成
        _ = self.cetm.get_result()
        # 等待任务完成
        cuda_hook_time_end("warm_up_cpu")

        cuda_hook_time("warm_up_gpu")
        ghidden_states = self.mlpm.dense_mlp_func(self.cmv.mlpm_ci, layer_idx=0, hidden_states=hidden_states)
        cuda_hook_time_end("warm_up_gpu")

    @torch.no_grad()
    def layer_moe_generate_grouped(
        self, 
        layer_idx: int,
        hidden_states: torch.Tensor,
        max_group_size: int = 16,
        max_token_ratio: float = 2.0,
    ):
        """
        使用分组策略的 layer_moe_generate 函数，将token数量相近的experts分组计算，减少padding浪费
        
        Args:
            layer_idx: 层索引
            hidden_states: 隐藏状态
            max_group_size: 每组最多experts数量
            max_token_ratio: 组内最大token数 / 最小token数的最大比例
        """
        cuda_hook_time(f"layer_moe_generate_grouped_{layer_idx}")

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
        
        # Step 8: 根据token数量分配CPU/GPU experts（使用已计算的 token 数量）
        num_experts_cpu = 0.6
        sorted_experts_by_load = sorted(expert_token_counts_list, key=lambda x: x[1])
        num_experts_total = len(sorted_experts_by_load)
        num_experts_on_cpu = int(num_experts_total * num_experts_cpu)
        
        cpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[:num_experts_on_cpu])
        gpu_expert_ids = set(expert_id for expert_id, _ in sorted_experts_by_load[num_experts_on_cpu:])
        
        # 打印调试信息
        logger.debug(f"\nExpert Token Distribution & Device Allocation (Grouped):")
        logger.debug(f"  Total experts: {num_experts_total}")
        logger.debug(f"  CPU experts: {num_experts_on_cpu} ({num_experts_cpu*100:.0f}%)")
        logger.debug(f"  GPU experts: {num_experts_total - num_experts_on_cpu} ({(1-num_experts_cpu)*100:.0f}%)")
        
        # 显示分组信息
        cpu_expert_token_counts = [(eid, count) for eid, count in expert_token_counts_list if eid in cpu_expert_ids]
        gpu_expert_token_counts = [(eid, count) for eid, count in expert_token_counts_list if eid in gpu_expert_ids]
        cpu_expert_groups = self._group_experts_for_einsum(cpu_expert_token_counts, max_group_size, max_token_ratio) if cpu_expert_token_counts else []
        gpu_expert_groups = self._group_experts_for_einsum(gpu_expert_token_counts, max_group_size, max_token_ratio) if gpu_expert_token_counts else []
        
        logger.debug(f"  CPU expert groups: {len(cpu_expert_groups)}")
        logger.debug(f"  GPU expert groups: {len(gpu_expert_groups)}")
        logger.debug(f"\n  Expert ID | Tokens | Device | Group")
        logger.debug(f"  {'-'*45}")

        total_tokens_cpu = sum(count for _, count in sorted_experts_by_load[:num_experts_on_cpu])
        total_tokens_gpu = sum(count for _, count in sorted_experts_by_load[num_experts_on_cpu:])
        
        # 创建expert_id到group_id的映射
        expert_to_group = {}
        for group_idx, group in enumerate(cpu_expert_groups):
            for expert_id in group:
                expert_to_group[expert_id] = f"CPU-{group_idx}"
        for group_idx, group in enumerate(gpu_expert_groups):
            for expert_id in group:
                expert_to_group[expert_id] = f"GPU-{group_idx}"
        
        for expert_id, token_count in sorted_experts_by_load:
            device = "CPU" if expert_id in cpu_expert_ids else "GPU"
            group_id = expert_to_group.get(expert_id, "N/A")
            logger.debug(f"  Expert {expert_id:2d} | {token_count:6d} | {device:4s} | {group_id}")
        logger.debug(f"\n  CPU total tokens: {total_tokens_cpu} ({total_tokens_cpu/(total_tokens_cpu+total_tokens_gpu)*100:.1f}%)")
        logger.debug(f"  GPU total tokens: {total_tokens_gpu} ({total_tokens_gpu/(total_tokens_cpu+total_tokens_gpu)*100:.1f}%)")
        
        # 打印分组详细信息
        logger.debug(f"\n  Expert Groups for Einsum (to reduce padding waste):")
        for group_idx, group in enumerate(cpu_expert_groups):
            group_tokens = [count for eid, count in cpu_expert_token_counts if eid in group]
            min_tokens = min(group_tokens) if group_tokens else 0
            max_tokens = max(group_tokens) if group_tokens else 0
            total_tokens = sum(group_tokens)
            padding_waste = (max_tokens * len(group) - total_tokens) / total_tokens * 100 if total_tokens > 0 else 0
            logger.debug(f"    CPU Group {group_idx}: experts={group}, tokens=[{min_tokens}, {max_tokens}], "
                        f"total={total_tokens}, padding_waste={padding_waste:.1f}%")
        for group_idx, group in enumerate(gpu_expert_groups):
            group_tokens = [count for eid, count in gpu_expert_token_counts if eid in group]
            min_tokens = min(group_tokens) if group_tokens else 0
            max_tokens = max(group_tokens) if group_tokens else 0
            total_tokens = sum(group_tokens)
            padding_waste = (max_tokens * len(group) - total_tokens) / total_tokens * 100 if total_tokens > 0 else 0
            logger.debug(f"    GPU Group {group_idx}: experts={group}, tokens=[{min_tokens}, {max_tokens}], "
                        f"total={total_tokens}, padding_waste={padding_waste:.1f}%")
        
        cuda_hook_time_end("experts_map_get")

        # 提交cpu 专家执行
        cuda_hook_time("cpu_experts_submit")
        expert_cache = torch.zeros_like(flat_hidden_states)
        # CPU experts - 按token数量分组，减少padding浪费
        if cpu_expert_ids:
            logger.debug(f"\n  Computing {len(cpu_expert_ids)} experts on CPU in {len(cpu_expert_groups)} groups...")
            for group_idx, expert_group in enumerate(cpu_expert_groups):
                logger.debug(f"    CPU Group {group_idx}: {len(expert_group)} experts {expert_group}")
                # 对每个组分别提交任务
                task = ExpertEinsumTask(
                    layer_idx=layer_idx,
                    expert_idx_list=expert_group,
                    expert_indices_map={eid: expert_indices_map[eid] for eid in expert_group},
                    expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in expert_group},
                    flat_hidden_states=flat_hidden_states,
                    flat_experts_weight=flat_experts_weight,
                    idxs=idxs,
                    final_hidden_states=expert_cache
                )
                self.cetm.submit(task)
        cuda_hook_time_end("cpu_experts_submit")

        cuda_hook_time("allocate_experts_cuda_memory_and_restore_model")
        gpu_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_idx)
        gpu_expert_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=list(gpu_expert_ids))
        gpu_expert_names = gpu_expert_names+gpu_shared_expert_names
        ret1, replica_uuid1, state_dict1 = \
            self.cmv.allocate_cuda_memory_and_load_into_gpu(
                gpu_expert_names, device_index_int=device1_idx_int)
        self.cmv.restore2model(state_dict1, self.cmv.mlpm_ci)
        cuda_hook_time_end("allocate_experts_cuda_memory_and_restore_model")

        cuda_hook_time("wait_experts")
        self.cmv.wait_load_into_gpu(replica_uuid1)
        cuda_hook_time_end("wait_experts")

        cuda_hook_time("gpu_sexperts")
        y = self.mlpm.shared_experts_func(
            self.cmv.mlpm_ci, layer_idx=layer_idx,
            hidden_states=hidden_states,
        )
        cuda_hook_time_end("gpu_sexperts")

        cuda_hook_time("gpu_experts")
        if gpu_expert_ids:
            logger.debug(f"  Computing {len(gpu_expert_ids)} experts on GPU in {len(gpu_expert_groups)} groups...")
            for group_idx, expert_group in enumerate(gpu_expert_groups):
                logger.debug(f"    GPU Group {group_idx}: {len(expert_group)} experts {expert_group}")
                # 对每个组分别调用einsum计算
                _ = self.mlpm.experts_func_gpu_einsum(
                    self.cmv.mlpm_ci, layer_idx=layer_idx,
                    expert_idx_list=expert_group,
                    expert_indices_map={eid: expert_indices_map[eid] for eid in expert_group},
                    expert_token_indices_map={eid: expert_token_indices_map[eid] for eid in expert_group},
                    flat_hidden_states=flat_hidden_states,
                    flat_experts_weight=flat_experts_weight,
                    idxs=idxs,
                    final_hidden_states=expert_cache
                )
        cuda_hook_time_end("gpu_experts")

        cuda_hook_time("wait_cetm_experts")
        # 获取所有CPU expert groups的结果（每个组提交了一个任务）
        num_cpu_groups = len(cpu_expert_groups)
        for _ in range(num_cpu_groups):
            _ = self.cetm.get_result()
        cuda_hook_time_end("wait_cetm_experts")

        layer_output = expert_cache.view(*orig_shape) + y

        cuda_hook_time_end(f"layer_moe_generate_grouped_{layer_idx}")
        return layer_output
    
    def _group_experts_for_einsum(self, expert_token_counts_list: list[tuple[int, int]], 
                                   max_group_size: int = 16, max_token_ratio: float = 2.0,
                                   small_token_threshold: int = 100, small_token_max_diff: int = 50,
                                   max_groups: int = 2) -> list[list[int]]:
        """
        根据token数量将experts分组，用于einsum聚合计算，减少padding浪费
        
        策略：
        1. 将token数量相近的experts分在一组
        2. 对于token数量较少的experts，使用绝对差值判断，避免分得太散
        3. 对于token数量较多的experts，使用比例判断
        4. 每组内的max_tokens应该尽量小，减少padding浪费
        5. 最多分成max_groups个组（默认2组）
        
        Args:
            expert_token_counts_list: [(expert_id, token_count), ...] 专家及其token数量列表
            max_group_size: 每组最多experts数量，避免单组过大
            max_token_ratio: 组内最大token数 / 最小token数的最大比例，超过此比例不合并（用于大token数）
            small_token_threshold: token数量阈值，低于此值的experts使用绝对差值判断
            small_token_max_diff: 小token数experts之间的最大绝对差值，超过此差值不合并
            max_groups: 最多分成的组数，默认2组
        
        Returns:
            list[list[int]]: 每个子列表包含应该一起计算的expert_ids，最多max_groups个组
        """
        if not expert_token_counts_list:
            return []
        
        # 按token数量排序
        sorted_experts = sorted(expert_token_counts_list, key=lambda x: x[1])
        
        groups = []
        i = 0
        
        while i < len(sorted_experts) and len(groups) < max_groups:
            current_group = []
            current_group.append(sorted_experts[i][0])  # 添加expert_id
            min_tokens = sorted_experts[i][1]
            max_tokens = sorted_experts[i][1]
            
            # 判断当前组是否属于小token数组（使用第一个expert的token数判断）
            is_small_token_group = min_tokens < small_token_threshold
            
            # 尝试将后续的experts加入当前组
            j = i + 1
            while j < len(sorted_experts) and len(current_group) < max_group_size:
                expert_id, token_count = sorted_experts[j]
                
                # 计算如果加入当前组，新的max_tokens
                new_max_tokens = max(max_tokens, token_count)
                
                # 判断是否可以加入
                can_add = False
                
                if is_small_token_group and token_count < small_token_threshold:
                    # 小token数组：使用绝对差值判断
                    # 如果新expert也是小token数，且差值在允许范围内，可以加入
                    token_diff = new_max_tokens - min_tokens
                    if token_diff <= small_token_max_diff:
                        can_add = True
                else:
                    # 大token数组或混合：使用比例判断
                    new_ratio = new_max_tokens / min_tokens if min_tokens > 0 else float('inf')
                    if new_ratio <= max_token_ratio:
                        can_add = True
                
                # 如果已经有max_groups-1个组了，且这是最后一个组，强制加入所有剩余的experts
                if len(groups) == max_groups - 1:
                    # 最后一个组，强制加入所有剩余的experts
                    can_add = True
                
                if can_add:
                    current_group.append(expert_id)
                    max_tokens = new_max_tokens
                    j += 1
                else:
                    # 不能加入，停止尝试（除非是最后一个组）
                    if len(groups) < max_groups - 1:
                        break
                    else:
                        # 最后一个组，强制加入
                        current_group.append(expert_id)
                        max_tokens = new_max_tokens
                        j += 1
            
            groups.append(current_group)
            i = j
        
        return groups
