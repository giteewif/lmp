import enum
import time
import os
import torch
import queue
import copy
from typing import Optional, Tuple, Dict, List

from typing import Dict, TYPE_CHECKING
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device


from lmp.sllm_store_c import STORAGE_PATH
if TYPE_CHECKING:
    from lmp.cuda_memory_view import HostMemoryView
    from lmp.cuda_memory_view import CudaMemoryView
    from lmp.sllm_thread_manager import SLLMTM

from models.Deepseek.deepseek_moe_16b_base.modeling_deepseek import DeepseekForCausalLM, DeepseekDecoderLayer
from models.Deepseek.mlpmodule import DeepseekModule, DeepseekOCalModel
from models.Mixtral.mlpmodule import MixtralModule
from utils.logger import init_logger
from utils.cuda_h import *
from lmp.pinpool import gpinpool
from dataclasses import dataclass

logger = init_logger(__name__)

@dataclass
class ExpertEinsumTask:
    """CPU expert einsum 任务的输入"""
    layer_idx: int
    expert_idx_list: List[int]
    expert_indices_map: Dict[int, Tuple[int, int]]  # {expert_id: (start_idx, end_idx)}
    expert_token_indices_map: Dict[int, torch.Tensor]  # {expert_id: token_ids}
    flat_hidden_states: torch.Tensor  # 原始展平的 hidden states
    flat_experts_weight: torch.Tensor  # 原始展平的 experts weight
    idxs: torch.Tensor  # 排序后的索引
    final_hidden_states: torch.Tensor
    if_decode: bool = False

@dataclass
class ExpertEinsumResult:
    """CPU expert einsum 任务的结果"""
    final_hidden_states: torch.Tensor
    time_einsum_end: float

DEEPSEEK_MODEL_NAME_TYPE = "Deepseek"
MIXTRAL_MODEL_NAME_TYPE = "Mixtral"

# original_dtype = torch.get_default_dtype()
# torch.set_default_dtype(torch.bfloat16)

class WeightType(enum.Enum):
    W1 = 1
    W2 = 2
    W3 = 3

class MLPModuleWrapper:
    def __init__(self, model_name_type: str, model_path: str):
        self.model_name_type = model_name_type
        self.model_path = model_path
        self.model_abs_path = os.path.join(STORAGE_PATH, model_path)
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            self.model_class = DeepseekModule()
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            self.model_class = MixtralModule()
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

        self.config = self.model_class.get_config(self.model_abs_path)

        
    def init_chmv_meta_model(self, cmv: "CudaMemoryView", hmv: "HostMemoryView"):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            with init_empty_weights():
                # cm = DeepseekOCalModel(self.config)
                cm = DeepseekForCausalLM(self.config)
                cm.to(self.config.torch_dtype)
                cm.eval()
                cmv.mlpm_ci = cm

                # Not need hm, we use einsum to restore experts weights from shared memory to model
                # hm =copy.deepcopy(cm)
                # hmv.mlpm_hi = None 
                # self.layerc = DeepseekDecoderLayer(self.config, 1)
            return
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
    def init_set_layer_func(
        self, layer_idx: int, config, model
    ):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = DeepseekDecoderLayer(config, layer_idx)
                # if layer_idx >= self.config.first_k_dense_replace:
                #         layer = copy.deepcopy(self.layerc)
                #         layer.self_attn.layer_idx = layer_idx
                # else:
                # # with init_empty_weights():
                #     layer = DeepseekDecoderLayer(config, layer_idx)
            # 优化：在 init_empty_weights() 上下文外调用 to() 和 eval()
            # 原因：layer.to() 需要递归遍历所有子模块（attention、MLP/MoE、layernorm）
            # 对于 MoE 层，包含 n_routed_experts 个 experts，每个都需要遍历
            # 在上下文外调用可以减少上下文管理器的一些开销
            layer.eval()
            # 注意：即使参数是空的，to() 仍然需要遍历模块树来设置 dtype 属性
            # 这是 PyTorch 的设计，无法完全避免，但可以减少上下文开销
            layer.to(config.torch_dtype)
            # mlpm_ci DeepseekOCalModel or DeepseekForCausalLM
            # self.cmv.mlpm_ci.model.layers[layer_idx]
            model.model.layers[layer_idx] = layer
            return layer
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def init_layer_func(
        self, layer_idx: int, config):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = DeepseekDecoderLayer(config, layer_idx)
            #     layer.to(config.torch_dtype)
            #     layer.eval()
            #     return layer
                # if layer_idx >= 1:
                #     layer = copy.deepcopy(self.layerc)
                #     layer.self_attn.layer_idx = layer_idx
                # else:
                #     with init_empty_weights():
                #         layer = DeepseekDecoderLayer(config, layer_idx)
                # print(layer)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def create_empty_model(self):
        """
        创建空的模型结构（不加载权重）。
        对于 Deepseek-MoE-16B (28层 x 64专家)，此操作耗时约 3 秒，主要开销：
        1. 实例化所有模块对象（28层 x 多个子模块）
        2. 注册所有参数（即使为空，也需要创建 Parameter 对象）
        3. 构建模块层次结构
        4. 设置 dtype（需要遍历所有参数）
        
        优化建议：
        - 如果多次使用，可以考虑缓存模型结构
        - 或者延迟创建，只在真正需要时创建
        """
        cuda_hook_time("create_empty_model")
        self.config._attn_implementation = "sdpa"
        with init_empty_weights():
            if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
                model = DeepseekForCausalLM(self.config)
            elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
                model = AutoModelForCausalLM.from_config(
                    self.config, trust_remote_code=True
                )
            else:
                raise ValueError(f"Invalid model name type: {self.model_name_type}")
            # model = AutoModelForCausalLM.from_config(
            #     self.config, trust_remote_code=True
            # )
        cuda_hook_time_end("create_empty_model")
        # cuda_hook_time("to_dtype")
        # model.to(dtype=self.config.torch_dtype)
        # cuda_hook_time_end("to_dtype")
        return model

    def restore_hm_state_dict2model(self, hm_state_dict, model):
        """
        Restore expert tensors from CPU shared memory back into the given `model`.
        Supports full-precision Linear modules (HQQLinear support not implemented).
        
        For Deepseek model:
        - Uses model.layers.X.mlp.experts.Y.{gate_proj|up_proj|down_proj}.weight
        - Also supports model.layers.X.mlp.shared_experts.{gate_proj|up_proj|down_proj}.weight
        - gate_proj corresponds to w2
        - up_proj corresponds to w1  
        - down_proj corresponds to w3
        
        For Mixtral model:
        - Uses model.layers.X.block_sparse_moe.experts.Y.{w1|w2|w3}.weight
        """
        if not hm_state_dict:
            logger.warning("restore_hm_state_dict2model received empty state_dict")
            return
        
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            # Deepseek uses:
            # - model.layers.X.mlp.experts.Y.{gate_proj|up_proj|down_proj}.weight (regular experts)
            # - model.layers.X.mlp.shared_experts.{gate_proj|up_proj|down_proj}.weight (shared experts)
            expert_indicators = [".mlp.experts.", ".mlp.shared_experts."]
            target_linears = {"gate_proj", "up_proj", "down_proj"}
            updated_params = 0
            
            with torch.no_grad():
                for name, tensor in hm_state_dict.items():
                    # 检查是否是 expert 或 shared_expert 相关的 tensor
                    is_expert_tensor = any(indicator in name for indicator in expert_indicators)
                    if not is_expert_tensor:
                        continue
                    
                    line_segments = name.split(".")
                    # 获取 gate_proj, up_proj, down_proj 的位置
                    linear_pos = next(
                        (idx for idx, token in enumerate(line_segments) if token in target_linears),
                        -1,
                    )
                    if linear_pos == -1:
                        continue
                    
                    try:
                        # 使用 accelerate 的工具函数设置 tensor
                        set_module_tensor_to_device(
                            model,
                            name,
                            tensor.device,
                            tensor,
                            clear_cache=False,
                        )
                        updated_params += 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to assign tensor %s to module: %s", name, exc, exc_info=True
                        )
            
            logger.debug(
                "restore_hm_state_dict2model loaded %d expert tensors (including shared_experts) for Deepseek model",
                updated_params,
            )
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            # Mixtral uses model.layers.X.block_sparse_moe.experts.Y.{w1|w2|w3}.weight
            expert_indicator = ".block_sparse_moe.experts."
            target_linears = {"w1", "w2", "w3"}
            updated_params = 0
            
            with torch.no_grad():
                for name, tensor in hm_state_dict.items():
                    if expert_indicator not in name:
                        continue
                    
                    line_segments = name.split(".")
                    # 获取 w1, w2, w3 的位置
                    linear_pos = next(
                        (idx for idx, token in enumerate(line_segments) if token in target_linears),
                        -1,
                    )
                    if linear_pos == -1:
                        continue
                    
                    try:
                        # 使用 accelerate 的工具函数设置 tensor
                        set_module_tensor_to_device(
                            model,
                            name,
                            tensor.device,
                            tensor,
                            clear_cache=False,
                        )
                        updated_params += 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to assign tensor %s to module: %s", name, exc, exc_info=True
                        )
            
            logger.debug(
                "restore_hm_state_dict2model loaded %d expert tensors for Mixtral model",
                updated_params,
            )
        else:
            logger.warning(f"restore_hm_state_dict2model not implemented for {self.model_name_type}")
            pass
    def get_experts_num(self):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            return self.config.n_routed_experts
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            return self.config.num_local_experts
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
    def get_experts_names_w(self, layer_idx: int, experts_idx_list, type_idx: WeightType):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_proj", "down_proj", "up_proj"]
            type_str = type_str_list[type_idx.value]
            experts_names = [f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{type_str}.weight" for expert_idx in experts_idx_list]
            return experts_names
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            type_str_list = ["none", "w1", "w2", "w3"]
            type_str = type_str_list[type_idx.value]
            experts_names = [f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{type_str}.weight" for expert_idx in experts_idx_list]
            return experts_names
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_tensor_index_general_names(self):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            return ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
    def get_shared_experts_names(self, layer_idx: int):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            if layer_idx < self.config.first_k_dense_replace:
                return []
            return [
                f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
            ]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
    def get_experts_names(self, layer_idx: int, expert_idx_list: list[int]):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            if layer_idx == 0:
                return [
                    "model.layers.0.mlp.gate_proj.weight",
                    "model.layers.0.mlp.down_proj.weight",
                    "model.layers.0.mlp.up_proj.weight",
                ]
            else:
                names_list = []
                for expert_idx in expert_idx_list:
                    names_list.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
                    names_list.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")
                    names_list.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")
                return names_list
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
    def get_gate_names(self, layer_idx: int):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            # first k dense with dense no gate
            # layer_idx start from 0, first_k_dense_replace start from 1
            if layer_idx < self.config.first_k_dense_replace:
                return []
            return [f"model.layers.{layer_idx}.mlp.gate.weight"]
    def get_layernorm_names(self, layer_idx: int):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.post_attention_layernorm.weight", 
                f"model.layers.{layer_idx}.input_layernorm.weight", ]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
    def get_attention_names(self, layer_idx: int):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.self_attn.q_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.k_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.v_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def gate_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            topk_idx, topk_weight, aux_loss = mi.model.layers[layer_idx].mlp.gate(hidden_states)
            return topk_idx, topk_weight, aux_loss
    
    def shared_experts_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            y = mi.model.layers[layer_idx].mlp.shared_experts(hidden_states)
            return y
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
    def self_attn_func(self, mi, layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Cache,
    ):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            # 类型注解：mi 是 DeepseekModule 实例
            mi: "DeepseekModule"
            hidden_states, _, _ = mi.model.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=True,
            )
            return hidden_states
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def iln_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            # 类型注解：mi 是 DeepseekModule 实例
            # mi: "DeepseekForCausalLM" = mi
            logger.debug(f"{mi.model.layers[layer_idx].input_layernorm.weight.data.device} {hidden_states.device}")
            hidden_states = mi.model.layers[layer_idx].input_layernorm(hidden_states)
            return hidden_states
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def paln_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].post_attention_layernorm(hidden_states)
            return hidden_states

    def dense_mlp_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].mlp(hidden_states)
            return hidden_states
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
    @torch.no_grad()
    def experts_func(self, mi, layer_idx: int, 
        expert_idx_list: list[int], 
        expert_indices_map: Dict[int, Tuple[int, int]],  # {expert_id: (start_idx, end_idx)}
        expert_token_indices_map: Dict[int, torch.Tensor],  # {expert_id: token_ids}
        flat_hidden_states: torch.Tensor,  # 原始展平的 hidden states
        flat_experts_weight: torch.Tensor,  # 原始展平的 experts weight
        idxs: torch.Tensor,  # 排序后的索引
        final_hidden_states: torch.Tensor,
        device='cuda'):
        """
        对每个 expert 处理其分配的 tokens
        
        Args:
            mi: 模型实例
            layer_idx: 层索引
            expert_idx_list: expert 索引列表
            expert_indices_map: {expert_id: (start_idx, end_idx)} 索引范围
            expert_token_indices_map: {expert_id: token_ids} token 索引
            flat_hidden_states: 原始展平的 hidden states
            flat_experts_weight: 原始展平的 experts weight
            idxs: 排序后的索引
            device: 计算设备 ('cuda', 'cpu', 或 'cuda:X')
        
        Returns:
            final_hidden_states: 最终隐藏状态
        """
        # 延迟创建 tensor maps：只在需要时索引 tensor
        for expert_id in expert_idx_list:
            if expert_id not in expert_token_indices_map or expert_id not in expert_indices_map:
                continue
            
            # 只在需要时创建 tensor
            tokens = flat_hidden_states[expert_token_indices_map[expert_id]]
            weights = flat_experts_weight[idxs[expert_indices_map[expert_id][0]:expert_indices_map[expert_id][1]]]
            token_indices = expert_token_indices_map[expert_id]
            
            if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
                if device == "cpu":
                    tokens_on_device = tokens.to("cpu")
                    weights_on_device = weights.to("cpu")
                    DeepseekModule.experts_func(
                        mi, layer_idx, expert_id, tokens_on_device, weights_on_device,
                        token_indices,
                        final_hidden_states=final_hidden_states
                    )
                else:
                    DeepseekModule.experts_func(
                        mi, layer_idx, expert_id, tokens, weights,
                        token_indices,
                        final_hidden_states=final_hidden_states
                )
            elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
                # TODO: 添加 Mixtral 支持
                raise NotImplementedError("Mixtral experts_func not implemented yet")
            else:
                raise ValueError(f"Invalid model name type: {self.model_name_type}")
        
        return final_hidden_states
    
    def scatter(self, expert_cache, expert_out_map, expert_token_indices_map, experts_weights_map):
        """
        将 expert 输出 scatter 回原始 hidden_states 的位置
        
        Args:
            hidden_states: [batch_size * seq_len, hidden_dim] 原始输入（展平后）
            expert_out_map: {expert_id: output} expert 输出
            expert_token_indices_map: {expert_id: token_indices} 每个 expert 处理的 token 索引
            experts_weights_map: {expert_id: weights} 路由权重（可能已经在 expert_out 中应用）
        
        Returns:
            expert_cache: [batch_size * seq_len, hidden_dim] 聚合后的输出
        """
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            expert_cache = DeepseekModule.scatter(
                expert_cache, expert_out_map, expert_token_indices_map
            )
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            # TODO: 添加 Mixtral 支持
            raise NotImplementedError("Mixtral scatter not implemented yet")
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
        
        return expert_cache
    def experts_func_mgpu_group_pad(
        self,
        expert_idx_list: list[int],
        expert_indices_map: Dict[int, Tuple[int, int]],
        device_flat_hidden_states: torch.Tensor,
        device_idxs: torch.Tensor,
    ):
        cuda_hook_time("experts_func_mgpu_group_pad")
        device_token_idxs = device_idxs // self.config.num_experts_per_tok

        expert_token_indices_map = {}
        for i, expert_idx in enumerate(expert_idx_list):
            expert_token_indices_map[expert_idx] = device_token_idxs[expert_indices_map[expert_idx][0]:expert_indices_map[expert_idx][1]]

        max_tokens = max(expert_token_indices_map[expert_idx].shape[0] for expert_idx in expert_idx_list)
        H = device_flat_hidden_states.shape[1]
        E = len(expert_idx_list)
        stacked_inputs = torch.zeros(
            E, max_tokens, H,
            dtype=device_flat_hidden_states.dtype, device=device_flat_hidden_states.device
        )

        for i, expert_idx in enumerate(expert_idx_list):
            token_ids = expert_token_indices_map[expert_idx]
            num_tokens = token_ids.shape[0]
            stacked_inputs[i, :num_tokens].copy_(device_flat_hidden_states[token_ids], non_blocking=True)
        cuda_hook_time_end("experts_func_mgpu_group_pad")
        return stacked_inputs
    def experts_func_mgpu_group_list(
        self,
        mi,
        layer_idx: int,
        expert_idx_list: list[int],
    ):
        cuda_hook_time("gpu_group_list")
        group_w1_list = []
        group_w2_list = []
        group_w3_list = []

        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            for expert_idx in expert_idx_list:
                expert_module = mi.model.layers[layer_idx].mlp.experts[expert_idx]
                group_w1_list.append(expert_module.gate_proj.weight)  # [I, H]
                group_w2_list.append(expert_module.down_proj.weight)   # [H, I]
                group_w3_list.append(expert_module.up_proj.weight)     # [I, H]
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            for expert_idx in expert_idx_list:
                expert_module = mi.model.layers[layer_idx].block_sparse_moe.experts[expert_idx]
                group_w1_list.append(expert_module.w1.weight)  # [I, H]
                group_w2_list.append(expert_module.w2.weight)  # [H, I]
                group_w3_list.append(expert_module.w3.weight)  # [I, H]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
        # 堆叠 weights: [E, I, H], [E, H, I], [E, I, H]
        # group_w1 = torch.stack(group_w1_list)  # [E, I, H]
        # group_w2 = torch.stack(group_w2_list)  # [E, H, I]
        # group_w3 = torch.stack(group_w3_list)  # [E, I, H]
        cuda_hook_time_end("gpu_group_list")
        return group_w1_list, group_w2_list, group_w3_list
    @torch.no_grad()
    def experts_func_mgpu_group_experts(
        self,
        group_w1_list,
        group_w2_list,
        group_w3_list,
    ):
        cuda_hook_time("gpu_group_tensor")
        # 堆叠 weights: [E, I, H], [E, H, I], [E, I, H]
        group_w1 = torch.stack(group_w1_list)  # [E, I, H]
        group_w2 = torch.stack(group_w2_list)  # [E, H, I]
        group_w3 = torch.stack(group_w3_list)  # [E, I, H]
        cuda_hook_time_end("gpu_group_tensor")
        return group_w1, group_w2, group_w3
    @torch.no_grad()
    def experts_func_mgpu_einsum_mp_multi_list(
        self,
        layer_idx: int,
        group_w1_list_map: Dict[int, list[torch.Tensor]],
        group_w2_list_map: Dict[int, list[torch.Tensor]],
        group_w3_list_map: Dict[int, list[torch.Tensor]],
        stacked_inputs_map: Dict[int, torch.Tensor],
        expert_idx_list_map: Dict[int, list[int]],
        # not device_id, but expert_id
        expert_indices_map: Dict[int, Tuple[int, int]],
        flat_hidden_states_map: Dict[int, torch.Tensor],
        flat_experts_weight_map: Dict[int, torch.Tensor],
        idxs_map: Dict[int, torch.Tensor],
        final_hidden_states: torch.Tensor,
        all_expert_weights_map: Optional[Dict[int, torch.Tensor]],
        all_token_ids_map: Optional[Dict[int, torch.Tensor]],
        expert_token_indices_map: Optional[Dict[int, torch.Tensor]],
    ):
        cuda_hook_time("gpu_group_einsum_mp_multi_list")
        group_w1_map = {}
        group_w2_map = {}
        group_w3_map = {}
        
        for device_id in group_w1_list_map.keys():
            cuda_hook_time("gpu_group_tensor")
            group_w1_list = group_w1_list_map[device_id]
            group_w2_list = group_w2_list_map[device_id]
            group_w3_list = group_w3_list_map[device_id]

             # 堆叠 weights: [E, I, H], [E, H, I], [E, I, H]
            group_w1 = torch.stack(group_w1_list)  # [E, I, H]
            group_w2 = torch.stack(group_w2_list)  # [E, H, I]
            group_w3 = torch.stack(group_w3_list)  # [E, I, H]

            group_w1_map[device_id] = group_w1
            group_w2_map[device_id] = group_w2
            group_w3_map[device_id] = group_w3
            cuda_hook_time_end("gpu_group_tensor")

        outputs_map = {}
        for device_id in group_w1_list_map.keys():
            # group_w1_list = group_w1_list_map[device_id]
            # group_w2_list = group_w2_list_map[device_id]
            # group_w3_list = group_w3_list_map[device_id]
            group_w1 = group_w1_map[device_id]
            group_w2 = group_w2_map[device_id]
            group_w3 = group_w3_map[device_id]

            stacked_inputs = stacked_inputs_map[device_id]
            expert_idx_list = expert_idx_list_map[device_id]
            device_flat_hidden_states = flat_hidden_states_map[device_id]
            device_flat_experts_weight = flat_experts_weight_map[device_id]
            device_idxs = idxs_map[device_id]
            device_token_idxs = device_idxs // self.config.num_experts_per_tok

            # cuda_hook_time("gpu_group_tensor")
            # # 堆叠 weights: [E, I, H], [E, H, I], [E, I, H]
            # group_w1 = torch.stack(group_w1_list)  # [E, I, H]
            # group_w2 = torch.stack(group_w2_list)  # [E, H, I]
            # group_w3 = torch.stack(group_w3_list)  # [E, I, H]
            # cuda_hook_time_end("gpu_group_tensor")

            cuda_hook_time("gpu_group_einsum")
            w1_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w1)
            act_fn = ACT2FN[self.config.hidden_act]
            w1_out = act_fn(w1_out)
            w3_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w3)
            intermediate = w1_out * w3_out
            outputs = torch.einsum('eti,ehi->eth', intermediate, group_w2)
            outputs_map[device_id] = outputs
            cuda_hook_time_end("gpu_group_einsum")

        for device_id in reversed(list(group_w1_list_map.keys())):
            outputs = outputs_map[device_id]
            
            stacked_inputs = stacked_inputs_map[device_id]
            expert_idx_list = expert_idx_list_map[device_id]
            device_flat_hidden_states = flat_hidden_states_map[device_id]
            device_flat_experts_weight = flat_experts_weight_map[device_id]
            device_idxs = idxs_map[device_id]
            device_token_idxs = device_idxs // self.config.num_experts_per_tok

            cuda_hook_time("gpu_final_hidden_states_scatter")
            final_device = final_hidden_states.device
            
            #     outputs = outputs.to(final_device, non_blocking=True)
            if outputs.device != final_device:
                device_final_hidden_states = torch.zeros_like(final_hidden_states, device=outputs.device)
            else:
                device_final_hidden_states = final_hidden_states
            
            # 批量收集所有需要 scatter 的数据和 weights
            all_expert_outs_slices = []
            
            # 如果提供了预收集的数据，直接使用；否则在循环中收集
            if all_expert_weights_map is not None and device_id in all_expert_weights_map:
                # 使用已经 concat 好的数据
                concat_expert_weights = all_expert_weights_map[device_id]  # [total_tokens, 1]
                concat_token_ids = all_token_ids_map[device_id] if all_token_ids_map and device_id in all_token_ids_map else None
                # 从 expert_token_indices_map 中提取当前设备的 expert_token_indices_map
                # expert_token_indices_map 是 {expert_id: token_ids}，不按设备分组
                device_expert_token_indices_map = {}
                if expert_token_indices_map is not None:
                    for expert_idx in expert_idx_list:
                        if expert_idx in expert_token_indices_map:
                            expert_token_indices_map[expert_idx] = expert_token_indices_map[expert_idx]
            else:
                all_expert_weights = []
                all_token_ids = []
                expert_token_indices_map = {}
                cuda_hook_time("all_expert_weights_slices")
                for i, expert_idx in enumerate(expert_idx_list):
                    start_idx, end_idx = expert_indices_map[expert_idx]
                    token_ids = device_token_idxs[start_idx:end_idx]
                    expert_token_indices_map[expert_idx] = token_ids
                    expert_weights = device_flat_experts_weight[device_idxs[start_idx:end_idx]]
                    all_expert_weights.append(expert_weights)
                    all_token_ids.append(token_ids)
                cuda_hook_time_end("all_expert_weights_slices")
                # 如果没有提供预收集的数据，需要 concat
                concat_expert_weights = torch.cat(all_expert_weights, dim=0) if all_expert_weights else None
                concat_token_ids = torch.cat(all_token_ids, dim=0) if all_token_ids else None
            
            # 收集 outputs 切片（必须在 outputs 计算后才能收集）
            cuda_hook_time("all_expert_outputs_slices")
            for i, expert_idx in enumerate(expert_idx_list):
                if expert_idx in expert_token_indices_map:
                    token_ids = expert_token_indices_map[expert_idx]
                else:
                    start_idx, end_idx = expert_indices_map[expert_idx]
                    token_ids = device_token_idxs[start_idx:end_idx]
                num_tokens = token_ids.shape[0]
                
                # 收集 outputs 切片（不应用 weights）
                expert_out_slice = outputs[i, :num_tokens]
                all_expert_outs_slices.append(expert_out_slice)
            cuda_hook_time_end("all_expert_outputs_slices")

            # 一次性批量处理：合并所有数据，批量应用 weights，然后 scatter
            if all_expert_outs_slices and concat_expert_weights is not None:
                cuda_hook_time("concat_expert_out")
                # 合并所有 expert_out 切片
                concat_expert_out = torch.cat(all_expert_outs_slices, dim=0)  # [total_tokens, H]
                cuda_hook_time_end("concat_expert_out")
                # 一次性批量应用 weights
                concat_expert_out = concat_expert_out.mul_(concat_expert_weights)
                cuda_hook_time("index_scatter")
                # 扩展 token_ids 以匹配 expert_out 的形状
                if concat_token_ids is None:
                    # 如果没有提供 concat_token_ids，需要从 device_expert_token_indices_map 重新构建
                    all_token_ids_list = []
                    for expert_idx in expert_idx_list:
                        if expert_idx in device_expert_token_indices_map:
                            all_token_ids_list.append(device_expert_token_indices_map[expert_idx])
                    concat_token_ids = torch.cat(all_token_ids_list, dim=0) if all_token_ids_list else None
                
                if concat_token_ids is not None:
                    index = concat_token_ids.view(-1, 1).expand(-1, device_final_hidden_states.shape[-1])
                    
                    # 一次性执行 scatter_reduce_
                    device_final_hidden_states.scatter_reduce_(
                        dim=0,
                        index=index,
                        src=concat_expert_out,
                        reduce='sum',
                    )
                cuda_hook_time_end("index_scatter")
            
            if device_final_hidden_states.device != final_device:
                device_final_hidden_states = device_final_hidden_states.to(final_device, non_blocking=True)
                # 将设备上的结果写回 final_hidden_states
                final_hidden_states.add_(device_final_hidden_states)
            
            cuda_hook_time_end("gpu_final_hidden_states_scatter")

            
    @torch.no_grad()
    def experts_func_mgpu_einsum_mp(
        self,
        layer_idx: int,
        group_w1: torch.Tensor,
        group_w2: torch.Tensor,
        group_w3: torch.Tensor,
        stacked_inputs: torch.Tensor,
        expert_idx_list: list[int],
        expert_indices_map: Dict[int, Tuple[int, int]],
        flat_hidden_states: torch.Tensor,
        flat_experts_weight: torch.Tensor,
        idxs: torch.Tensor,
        final_hidden_states: torch.Tensor
    ):
        """
        使用 einsum 在多 GPU 上并行批量计算 expert outputs
        
        Args:
            layer_idx: 层索引
            group_w1: 组 w1
            group_w2: 组 w2
            group_w3: 组 w3
            expert_idx_list: expert 索引列表
            expert_indices_map: {expert_id: (start_idx, end_idx)} 索引范围
            flat_hidden_states: 原始展平的 hidden states
            flat_experts_weight: 原始展平的 experts weight
            idxs: 排序后的索引
            final_hidden_states: 最终隐藏状态
        """
        if not expert_idx_list:
            return final_hidden_states
        # cuda_hook_time("gpu_group_pad")
        # if group_w1.device != flat_hidden_states.device:
        #     device_flat_hidden_states = flat_hidden_states.to(group_w1.device, non_blocking=True)
        #     device_idxs = idxs.to(group_w1.device, non_blocking=True)
        #     device_token_idxs = device_idxs // self.config.num_experts_per_tok
        #     device_flat_experts_weight = flat_experts_weight.to(group_w1.device, non_blocking=True)
        # else:
        #     device_flat_hidden_states = flat_hidden_states
        #     device_idxs = idxs
        #     device_token_idxs = device_idxs // self.config.num_experts_per_tok
        #     device_flat_experts_weight = flat_experts_weight
        # expert_token_indices_map = {}
        

        # max_tokens = max(expert_token_indices_map[expert_idx].shape[0] for expert_idx in expert_idx_list)
        # H = flat_hidden_states.shape[1]
        # E = len(expert_idx_list)
        # stacked_inputs = torch.zeros(
        #     E, max_tokens, H,
        #     dtype=flat_hidden_states.dtype, device=group_w1.device
        # )

        # for i, expert_idx in enumerate(expert_idx_list):
        #     token_ids = expert_token_indices_map[expert_idx]
        #     num_tokens = token_ids.shape[0]
        #     stacked_inputs[i, :num_tokens].copy_(device_flat_hidden_states[token_ids], non_blocking=True)

        # cuda_hook_time_end("gpu_group_pad")
        
        device_flat_hidden_states = flat_hidden_states
        device_idxs = idxs
        device_token_idxs = device_idxs // self.config.num_experts_per_tok
        device_flat_experts_weight = flat_experts_weight

        cuda_hook_time("gpu_group_einsum")
        w1_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w1)
        act_fn = ACT2FN[self.config.hidden_act]
        w1_out = act_fn(w1_out)
        w3_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w3)
        intermediate = w1_out * w3_out
        outputs = torch.einsum('eti,ehi->eth', intermediate, group_w2)
        cuda_hook_time_end("gpu_group_einsum")

        cuda_hook_time("gpu_final_hidden_states_scatter")
        final_device = final_hidden_states.device
        expert_token_indices_map = {}
        for i, expert_idx in enumerate(expert_idx_list):
            expert_token_indices_map[expert_idx] = device_token_idxs[expert_indices_map[expert_idx][0]:expert_indices_map[expert_idx][1]]
            
        #     outputs = outputs.to(final_device, non_blocking=True)
        if outputs.device != final_device:
            device_final_hidden_states = torch.zeros_like(final_hidden_states, device=outputs.device)
        else:
            device_final_hidden_states = final_hidden_states
        
        # 批量收集所有需要 scatter 的数据和 weights
        all_expert_outs_slices = []
        all_expert_weights = []
        all_token_ids = []
        
        cuda_hook_time("all_expert_weight_slices")
        for i, expert_idx in enumerate(expert_idx_list):
            token_ids = expert_token_indices_map[expert_idx]
            num_tokens = token_ids.shape[0]

            # 收集 outputs 切片（不应用 weights）
            # expert_out_slice = outputs[i, :num_tokens]
            # all_expert_outs_slices.append(expert_out_slice)

            # 收集对应的 weights
            start_idx, end_idx = expert_indices_map[expert_idx]
            expert_weights = device_flat_experts_weight[device_idxs[start_idx:end_idx]]
            all_expert_weights.append(expert_weights)
            all_token_ids.append(token_ids)
        
        cuda_hook_time_end("all_expert_weight_slices")

        cuda_hook_time("all_expert_output_slices")
        for i, expert_idx in enumerate(expert_idx_list):
            token_ids = expert_token_indices_map[expert_idx]
            num_tokens = token_ids.shape[0]

            # 收集 outputs 切片（不应用 weights）
            expert_out_slice = outputs[i, :num_tokens]
            all_expert_outs_slices.append(expert_out_slice)
        
        cuda_hook_time_end("all_expert_output_slices")
        
        # 一次性批量处理：合并所有数据，批量应用 weights
        if all_expert_outs_slices:
            cuda_hook_time("concat_expert_out")
            # 合并所有 expert_out 切片和 weights
            concat_expert_out = torch.cat(all_expert_outs_slices, dim=0)  # [total_tokens, H]
            concat_expert_weights = torch.cat(all_expert_weights, dim=0)  # [total_tokens, 1]
            concat_token_ids = torch.cat(all_token_ids, dim=0)  # [total_tokens]
            cuda_hook_time_end("concat_expert_out")
            # 一次性批量应用 weights
            concat_expert_out = concat_expert_out.mul_(concat_expert_weights)
            cuda_hook_time("index_scatter")
            # 扩展 token_ids 以匹配 expert_out 的形状
            index = concat_token_ids.view(-1, 1).expand(-1, device_final_hidden_states.shape[-1])
            
            # 一次性执行 scatter_reduce_
            device_final_hidden_states.scatter_reduce_(
                dim=0,
                index=index,
                src=concat_expert_out,
                reduce='sum',
            )
            cuda_hook_time_end("index_scatter")
        
        if device_final_hidden_states.device != final_device:
            device_final_hidden_states = device_final_hidden_states.to(final_device, non_blocking=True)
            # 将设备上的结果写回 final_hidden_states
            final_hidden_states.add_(device_final_hidden_states)
        
        cuda_hook_time_end("gpu_final_hidden_states_scatter")

        return final_hidden_states
    @torch.no_grad()
    def experts_func_mgpu_einsum(
        self, 
        mi,
        layer_idx: int,
        expert_idx_list_map: Dict[int, list[int]],  # {device_id: [expert_id]}
        expert_indices_map: Dict[int, Tuple[int, int]],  # {expert_id: (start_idx, end_idx)}
        expert_token_indices_map: Dict[int, torch.Tensor],  # {expert_id: token_ids}
        flat_hidden_states: torch.Tensor,  # 原始展平的 hidden states
        flat_experts_weight: torch.Tensor,  # 原始展平的 experts weight
        idxs: torch.Tensor,  # 排序后的索引
        final_hidden_states: torch.Tensor,
        streams
    ):
        """
        使用 einsum 在多 GPU 上并行批量计算 expert outputs
        
        Args:
            mi: 模型实例（可能包含多个GPU的模型）
            layer_idx: 层索引
            expert_idx_list: {device_id: [expert_id]} 每个设备上的 expert 列表
            expert_indices_map: {expert_id: (start_idx, end_idx)} 索引范围
            expert_token_indices_map: {expert_id: token_ids} token 索引
            flat_hidden_states: 原始展平的 hidden states
            flat_experts_weight: 原始展平的 experts weight
            idxs: 排序后的索引
            final_hidden_states: 最终隐藏状态
        
        Returns:
            final_hidden_states: 最终隐藏状态
        """
        if not expert_idx_list_map:
            return final_hidden_states
        
        cuda_hook("mgpu_einsum_with_group_tensors")
        time_start_group = time.time()
        
        # 获取所有设备ID
        device_ids = list(expert_idx_list_map.keys())
        if not device_ids:
            return final_hidden_states
        
        # 为每个设备创建 CUDA stream，实现并行执行
        # streams = {device_id: torch.cuda.Stream(device=device_id) for device_id in device_ids}
        
        # 存储每个设备的结果
        device_outputs = {}
        device_expert_mappings = {}  # {device_id: [(expert_idx, i)]} 用于后续 scatter
        
        # 步骤1: 并行获取所有设备的 group tensors
        cuda_hook("mgpu_group_tensor")
        device_group_tensors = {}
        for device_id in device_ids:
            with torch.cuda.device(device_id):
                stream = streams[device_id]
                with torch.cuda.stream(stream):
                    expert_indices = [idx for idx in expert_idx_list_map[device_id] if idx in expert_token_indices_map]
                    if not expert_indices:
                        continue
                    
                    group_w1_list = []
                    group_w2_list = []
                    group_w3_list = []
                    
                    if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
                        for expert_idx in expert_indices:
                            expert_module = mi.model.layers[layer_idx].mlp.experts[expert_idx]
                            group_w1_list.append(expert_module.gate_proj.weight)
                            group_w2_list.append(expert_module.down_proj.weight)
                            group_w3_list.append(expert_module.up_proj.weight)
                    elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
                        for expert_idx in expert_indices:
                            expert_module = mi.model.layers[layer_idx].block_sparse_moe.experts[expert_idx]
                            group_w1_list.append(expert_module.w1.weight)
                            group_w2_list.append(expert_module.w2.weight)
                            group_w3_list.append(expert_module.w3.weight)
                    else:
                        raise ValueError(f"Invalid model name type: {self.model_name_type}")
                    
                    device_group_tensors[device_id] = {
                        'w1': torch.stack(group_w1_list),
                        'w2': torch.stack(group_w2_list),
                        'w3': torch.stack(group_w3_list),
                        'expert_indices': expert_indices
                    }
        cuda_hook_end("mgpu_group_tensor")
        
        # 步骤2: 并行填充所有设备的 stacked_inputs
        cuda_hook("mgpu_group_pad")
        device_stacked_inputs = {}
        for device_id in device_ids:
            if device_id not in device_group_tensors:
                continue
            with torch.cuda.device(device_id):
                stream = streams[device_id]
                with torch.cuda.stream(stream):
                    expert_indices = device_group_tensors[device_id]['expert_indices']
                    group_w1 = device_group_tensors[device_id]['w1']
                    
                    # 计算 max_tokens
                    max_tokens = max(
                        expert_token_indices_map[eid].shape[0] 
                        for eid in expert_indices
                    )
                    
                    H = flat_hidden_states.shape[1]
                    E = len(expert_indices)
                    
                    # 在对应设备上创建 stacked_inputs
                    stacked_inputs = torch.zeros(
                        E, max_tokens, H,
                        dtype=flat_hidden_states.dtype, device=f'cuda:{device_id}'
                    )
                
                    # 将 flat_hidden_states 复制到对应设备
                    device_flat_hidden_states = flat_hidden_states.to(f'cuda:{device_id}', non_blocking=True)
                    device_idxs = idxs.to(f'cuda:{device_id}', non_blocking=True)
                    device_token_idxs = device_idxs // self.config.num_experts_per_tok
                    # 填充 stacked_inputs
                    for i, expert_idx in enumerate(expert_indices):
                        token_ids = device_token_idxs[expert_indices_map[expert_idx][0]:expert_indices_map[expert_idx][1]]
                        num_tokens = token_ids.shape[0]
                        # 将 token_ids 也移到对应设备
                        stacked_inputs[i, :num_tokens].copy_(
                            device_flat_hidden_states[token_ids], non_blocking=True
                        )
                    
                    device_stacked_inputs[device_id] = stacked_inputs
                    device_expert_mappings[device_id] = [
                        (expert_idx, i) for i, expert_idx in enumerate(expert_indices)
                    ]
        cuda_hook_end("mgpu_group_pad")
        
        # 步骤3: 并行执行所有设备的 einsum 计算
        cuda_hook("mgpu_group_einsum")
        time_start_einsum = time.time()
        for device_id in device_ids:
            if device_id not in device_group_tensors:
                continue
            with torch.cuda.device(device_id):
                stream = streams[device_id]
                with torch.cuda.stream(stream):
                    stacked_inputs = device_stacked_inputs[device_id]
                    group_w1 = device_group_tensors[device_id]['w1']
                    group_w2 = device_group_tensors[device_id]['w2']
                    group_w3 = device_group_tensors[device_id]['w3']
                    
                    # 并行执行 einsum 计算
                    w1_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w1)
                    act_fn = ACT2FN[self.config.hidden_act]
                    w1_out = act_fn(w1_out)
                    
                    w3_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w3)
                    intermediate = w1_out * w3_out
                    outputs = torch.einsum('eti,ehi->eth', intermediate, group_w2)
                    
                    device_outputs[device_id] = outputs
        cuda_hook_end("mgpu_group_einsum")
        logger.debug(f"mgpu group einsum cost {time.time() - time_start_einsum} s")
        
        # 同步所有 streams，确保所有计算完成
        # for device_id in device_ids:
        #     if device_id in streams:
        #         streams[device_id].synchronize()
        
        # 步骤4: 收集所有设备的结果并 scatter 回 final_hidden_states
        cuda_hook("mgpu_final_hidden_states_scatter")
        final_device = final_hidden_states.device
        
        for device_id in device_ids:
            if device_id not in device_outputs:
                continue
            
            outputs = device_outputs[device_id]
            expert_mappings = device_expert_mappings[device_id]
            
            
            # 将 outputs 移动到 final_device（如果需要）
            if outputs.device != final_device:
                expert_cache = torch.zeros_like(final_hidden_states, device=outputs.device)
                device_flat_experts_weight = flat_experts_weight.to(outputs.device, non_blocking=True)
                device_idxs = idxs.to(group_w1.device, non_blocking=True)
                device_token_idxs = device_idxs // self.config.num_experts_per_tok
            else:
                expert_cache = final_hidden_states
                device_flat_experts_weight = flat_experts_weight
                device_idxs = idxs
                device_token_idxs = device_idxs // self.config.num_experts_per_tok
            # Scatter 每个 expert 的输出
            for expert_idx, i in expert_mappings:
                
                start_idx, end_idx = expert_indices_map[expert_idx]
                token_ids = device_token_idxs[start_idx:end_idx]
                num_tokens = token_ids.shape[0]
                
                expert_out = outputs[i, :num_tokens]  # [num_tokens, H]
                
                expert_weights = device_flat_experts_weight[device_idxs[start_idx:end_idx]]
                
                # 应用 weights
                expert_out = expert_out.mul_(expert_weights)
                
                # Scatter 回 final_hidden_states
                expert_cache.scatter_reduce_(
                    dim=0,
                    index=token_ids.view(-1, 1).expand(-1, final_hidden_states.shape[-1]),
                    src=expert_out,
                    reduce='sum'
                )
            if expert_cache.device != final_device:
                expert_cache = expert_cache.to(final_device, non_blocking=True)
                final_hidden_states.add_(expert_cache)
        cuda_hook_end("mgpu_final_hidden_states_scatter")
        
        logger.debug(f"mgpu experts func einsum cost {time.time() - time_start_group} s")
        cuda_hook_end("mgpu_einsum_with_group_tensors")
        
        return final_hidden_states
    @torch.no_grad()
    def experts_func_gpu_einsum(self,
        mi, layer_idx: int,
        expert_idx_list: list[int],
        expert_indices_map: Dict[int, Tuple[int, int]],  # {expert_id: (start_idx, end_idx)}
        expert_token_indices_map: Dict[int, torch.Tensor],  # {expert_id: token_ids}
        flat_hidden_states: torch.Tensor,  # 原始展平的 hidden states
        flat_experts_weight: torch.Tensor,  # 原始展平的 experts weight
        idxs: torch.Tensor,  # 排序后的索引
        final_hidden_states: torch.Tensor
    ):
        """
        使用 einsum 在 GPU 上批量计算 expert outputs
        
        Args:
            mi: 模型实例（GPU）
            layer_idx: 层索引
            expert_idx_list: expert 索引列表
            expert_indices_map: {expert_id: (start_idx, end_idx)} 索引范围
            expert_token_indices_map: {expert_id: token_ids} token 索引
            flat_hidden_states: 原始展平的 hidden states（GPU）
            flat_experts_weight: 原始展平的 experts weight（GPU）
            idxs: 排序后的索引（GPU）
            final_hidden_states: 最终隐藏状态（GPU）
        
        Returns:
            final_hidden_states: 最终隐藏状态
        """
        if not expert_idx_list:
            return final_hidden_states

        cuda_hook("gpu_einsum_with_group_tensors")
        time_start_group = time.time()
        
        # 过滤有效的 expert indices
        expert_indices = [idx for idx in expert_idx_list if idx in expert_token_indices_map]
        if not expert_indices:
            return final_hidden_states
        
        # 从 GPU 模型获取 group tensors
        cuda_hook("gpu_group_tensor")
        group_w1_list = []
        group_w2_list = []
        group_w3_list = []
        
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            for expert_idx in expert_indices:
                expert_module = mi.model.layers[layer_idx].mlp.experts[expert_idx]
                group_w1_list.append(expert_module.gate_proj.weight)  # [I, H]
                group_w2_list.append(expert_module.down_proj.weight)   # [H, I]
                group_w3_list.append(expert_module.up_proj.weight)     # [I, H]
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            for expert_idx in expert_indices:
                expert_module = mi.model.layers[layer_idx].block_sparse_moe.experts[expert_idx]
                group_w1_list.append(expert_module.w1.weight)  # [I, H]
                group_w2_list.append(expert_module.w2.weight)  # [H, I]
                group_w3_list.append(expert_module.w3.weight)  # [I, H]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
        
        # 堆叠 weights: [E, I, H], [E, H, I], [E, I, H]
        group_w1 = torch.stack(group_w1_list)  # [E, I, H]
        group_w2 = torch.stack(group_w2_list)  # [E, H, I]
        group_w3 = torch.stack(group_w3_list)  # [E, I, H]
        cuda_hook_end("gpu_group_tensor")
        logger.debug(f"gpu group tensors cost {time.time() - time_start_group} s")
        
        time_start_pad = time.time()
        cuda_hook("gpu_group_pad")
        # 计算 max_tokens
        max_tokens = max(
            expert_token_indices_map[eid].shape[0] 
            for eid in expert_indices
        )
        
        # 获取 hidden_dim
        H = flat_hidden_states.shape[1]  # hidden_dim
        E = len(expert_indices)  # expert 数量
        
        # 直接分配整个 stacked_inputs tensor（在 GPU 上）
        # 形状: [E, max_tokens, H]
        stacked_inputs = torch.zeros(
            E, max_tokens, H,
            dtype=flat_hidden_states.dtype, device=group_w1.device
        )
        
        if group_w1.device != flat_hidden_states.device:
            device_flat_hidden_states = flat_hidden_states.to(group_w1.device, non_blocking=True)
            device_idxs = idxs.to(group_w1.device, non_blocking=True)
            device_token_idxs = device_idxs // self.config.num_experts_per_tok
        else:
            device_flat_hidden_states = flat_hidden_states
            device_idxs = idxs
            device_token_idxs = device_idxs // self.config.num_experts_per_tok

        # 直接从 flat_hidden_states 复制需要的 token 到 stacked_inputs 的对应位置
        for i, expert_idx in enumerate(expert_indices):
            token_ids = device_token_idxs[expert_indices_map[expert_idx][0]:expert_indices_map[expert_idx][1]]
            num_tokens = token_ids.shape[0]
            
            # 直接从 flat_hidden_states 复制到 stacked_inputs[i, :num_tokens, :]
            stacked_inputs[i, :num_tokens].copy_(device_flat_hidden_states[token_ids], non_blocking=True)
            # padding 部分保持未初始化（如果后续不需要0值，可以跳过 zero_）
            # 如果需要确保 padding 为0，取消下面的注释
            # stacked_inputs[i, num_tokens:].zero_()
        
        cuda_hook_end("gpu_group_pad")
        logger.debug(f"gpu pad cost {time.time() - time_start_pad} s")

        time_start_einsum = time.time()
        cuda_hook("gpu_group_einsum")
        # 使用 einsum 批量计算（在 GPU 上）
        # w1_out: [E, max_tokens, I]
        logger.debug(f"start_w1")
        w1_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w1)
        act_fn = ACT2FN[self.config.hidden_act]
        w1_out = act_fn(w1_out)
        logger.debug(f"start_w3")
        # w3_out: [E, max_tokens, I]
        w3_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w3)
        
        # intermediate: [E, max_tokens, I]
        intermediate = w1_out * w3_out
        logger.debug(f"start_w2")
        # outputs: [E, max_tokens, H]
        outputs = torch.einsum('eti,ehi->eth', intermediate, group_w2)
        logger.debug(f"gpu group einsum cost {time.time() - time_start_einsum} s")
        cuda_hook_end("gpu_group_einsum")

        cuda_hook("gpu_final_hidden_states_scatter")
        
        # ========== 原始单GPU逻辑（已注释，方便后续调试） ==========
        # # 提取有效结果（去除 padding）并 scatter 回 final_hidden_states
        # for i, expert_idx in enumerate(expert_indices):
        #     # 直接从索引信息获取 token 数量
        #     token_ids = expert_token_indices_map[expert_idx]
        #     num_tokens = token_ids.shape[0]
        #     
        #     # 使用切片 [:num_tokens] 创建 view，避免拷贝
        #     expert_out = outputs[i, :num_tokens]  # [num_tokens, H]
        #     
        #     # 直接从 flat_experts_weight 获取 weights，避免中间拷贝
        #     start_idx, end_idx = expert_indices_map[expert_idx]
        #     expert_weights = flat_experts_weight[idxs[start_idx:end_idx]]  # [num_tokens, 1]
        #     
        #     # 应用 weights
        #     expert_out = expert_out.mul_(expert_weights)
        #     
        #     # 使用 token_ids 进行 scatter
        #     final_hidden_states.scatter_reduce_(
        #         dim=0,
        #         index=token_ids.view(-1, 1).repeat(1, final_hidden_states.shape[-1]),
        #         src=expert_out,
        #         reduce='sum'
        #     )
        # ========== 原始逻辑结束 ==========
        
        # ========== 多GPU适配逻辑 ==========
        # 确保 final_hidden_states 和 outputs 在同一个设备上
        # 如果不在，需要将 outputs 移动到 final_hidden_states 的设备上
        final_device = final_hidden_states.device
        outputs_device = outputs.device

        if outputs.device != final_device:
            expert_cache = torch.zeros_like(final_hidden_states, device=outputs.device)
            device_flat_experts_weight = flat_experts_weight.to(outputs.device, non_blocking=True)
        else:
            expert_cache = final_hidden_states
            device_flat_experts_weight = flat_experts_weight
           
        
        # 提取有效结果（去除 padding）并 scatter 回 final_hidden_states
        for i, expert_idx in enumerate(expert_indices):
            # 直接从索引信息获取 token 数量
            start_idx, end_idx = expert_indices_map[expert_idx]
            token_ids = device_token_idxs[start_idx:end_idx]
            num_tokens = token_ids.shape[0]
            
            # 使用切片 [:num_tokens] 创建 view，避免拷贝
            expert_out = outputs[i, :num_tokens]  # [num_tokens, H]
            
            # 获取 expert weights（在正确的设备上）
            expert_weights = device_flat_experts_weight[device_idxs[start_idx:end_idx]]  # [num_tokens, 1]
            
            # 应用 weights
            expert_out = expert_out.mul_(expert_weights)
            
            # 使用 token_ids 进行 scatter
            # 注意：scatter_reduce_ 要求所有张量在同一个设备上
    
            index = token_ids.view(-1, 1).expand(-1, expert_cache.shape[-1])
            expert_cache.scatter_reduce_(
                dim=0,
                index=index,
                src=expert_out,
                reduce='sum',
            )
        # ========== 多GPU适配逻辑结束 ==========
        if expert_cache.device != final_device:
            expert_cache = expert_cache.to(final_device, non_blocking=True)
            final_hidden_states.add_(expert_cache)
        cuda_hook_end("gpu_final_hidden_states_scatter")
        
        logger.debug(f"gpu experts func einsum cost {time.time()-time_start_group} s")
        cuda_hook_end("gpu_einsum_with_group_tensors")
        
        return final_hidden_states

    def experts_func_einsum(self, 
        hmv: "HostMemoryView", layer_idx: int, 
        expert_idx_list: list[int], 
        expert_indices_map: Dict[int, Tuple[int, int]],  # {expert_id: (start_idx, end_idx)}
        expert_token_indices_map: Dict[int, torch.Tensor],  # {expert_id: token_ids}
        flat_hidden_states: torch.Tensor,  # 原始展平的 hidden states
        flat_experts_weight: torch.Tensor,  # 原始展平的 experts weight
        idxs: torch.Tensor,  # 排序后的索引
        final_hidden_states: torch.Tensor,
        output_queue: queue.Queue
    ):
        """
        使用 einsum 批量计算 expert outputs
        
        Args:
            hmv: HostMemoryView 实例
            layer_idx: 层索引
            expert_idx_list: expert 索引列表
            expert_indices_map: {expert_id: (start_idx, end_idx)} 索引范围
            expert_token_indices_map: {expert_id: token_ids} token 索引
            flat_hidden_states: 原始展平的 hidden states
            flat_experts_weight: 原始展平的 experts weight
            idxs: 排序后的索引
        
        Returns:
            final_hidden_states: 最终隐藏状态
        """
        if not expert_idx_list:
            return final_hidden_states

        cuda_hook("einsum_with_group_tensors")
        time_start_group = time.time()
        # 调用 einsum_with_group_tensors 函数（在 CPU 上计算）
        # 直接传递索引信息，避免创建中间 tensor maps
        final_hidden_states = self.einsum_with_group_tensors(
            hmv=hmv,
            layer_idx=layer_idx,
            expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            expert_token_indices_map=expert_token_indices_map,
            flat_hidden_states=flat_hidden_states,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            final_hidden_states=final_hidden_states,
            output_queue=output_queue
        )
        logger.debug(f" experts func einsum cost {time.time()-time_start_group} s")
        cuda_hook_end("einsum_with_group_tensors")

        return final_hidden_states
    
    def experts_func_einsum_mp(self, 
        hmv: "HostMemoryView", layer_idx: int, 
        expert_idx_list: list[int], 
        expert_indices_map: Dict[int, Tuple[int, int]],  # {expert_id: (start_idx, end_idx)}
        flat_hidden_states: torch.Tensor,  # 原始展平的 hidden states
        idxs: torch.Tensor,  # 排序后的索引
        output_queue
    ):
        """
        使用 einsum 批量计算 expert outputs
        
        Args:
            hmv: HostMemoryView 实例
            layer_idx: 层索引
            expert_idx_list: expert 索引列表
            expert_indices_map: {expert_id: (start_idx, end_idx)} 索引范围
            expert_token_indices_map: {expert_id: token_ids} token 索引
            flat_hidden_states: 原始展平的 hidden states
            flat_experts_weight: 原始展平的 experts weight
            idxs: 排序后的索引
        
        Returns:
            final_hidden_states: 最终隐藏状态
        """
        if not expert_idx_list:
            return None, []
        # 调用 einsum_with_group_tensors 函数（在 CPU 上计算）
        # 直接传递索引信息，避免创建中间 tensor maps
        final_hidden_states, group_list = self.einsum_with_group_tensors_mp(
            hmv=hmv,
            layer_idx=layer_idx,
            expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            idxs=idxs,
            output_queue=output_queue
        )
        return final_hidden_states, group_list

    @torch.no_grad()
    def einsum_with_group_tensors(self, 
        hmv: "HostMemoryView",
        layer_idx: int, expert_idx_list: list[int], 
        expert_indices_map: Dict[int, Tuple[int, int]],  # {expert_id: (start_idx, end_idx)}
        expert_token_indices_map: Dict[int, torch.Tensor],  # {expert_id: token_ids}
        flat_hidden_states: torch.Tensor,  # 原始展平的 hidden states
        flat_experts_weight: torch.Tensor,  # 原始展平的 experts weight
        idxs: torch.Tensor,  # 排序后的索引
        final_hidden_states: torch.Tensor,
        output_queue: queue.Queue
    ):
        """
        使用 group tensors 进行批量 einsum 计算
        
        Args:
            layer_idx: 层索引
            expert_idx_list: expert 索引列表
            expert_indices_map: {expert_id: (start_idx, end_idx)} 索引范围
            expert_token_indices_map: {expert_id: token_ids} token 索引
            flat_hidden_states: 原始展平的 hidden states
            flat_experts_weight: 原始展平的 experts weight
            idxs: 排序后的索引
            expert_token_indices_map_full: 完整的 token indices map（用于 scatter）
        
        Returns:
            final_hidden_states: 最终隐藏状态
        """
        time_start_group = time.time()
        cuda_hook("group tensor")
        # 获取 group tensors (已经是堆叠好的 [E, ...] 形状)
        group_dict = hmv.group_experts_tensor(layer_idx, expert_idx_list)
        group_w1 = group_dict['group_w1']  # [E, I, H]
        group_w2 = group_dict['group_w2']  # [E, H, I]
        group_w3 = group_dict['group_w3']  # [E, I, H]
        
        cuda_hook_end("group tensor")
        logger.debug(f"group tensors cost {time.time() - time_start_group} s")
        
        cuda_hook_time("move_flat_hidden2cpu")
        flat_hidden_states_cpu_pin = gpinpool.alloc_same_pin_tensor(flat_hidden_states)
        flat_hidden_states_cpu_pin.copy_(flat_hidden_states, non_blocking=False)

        token_idxs = idxs // self.config.num_experts_per_tok
        token_idxs_cpu_pin = token_idxs.cpu()
        # token_idxs_cpu_pin = gpinpool.alloc_same_pin_tensor(token_idxs)
        # token_idxs_cpu_pin.copy_(token_idxs, non_blocking=False)
        # token_idxs_cpu_pin = token_idxs_cpu_pin.long()

        H = flat_hidden_states_cpu_pin.shape[1]
        E = len(expert_idx_list)

        max_tokens = max(
            expert_token_indices_map[eid].shape[0] 
            for eid in expert_idx_list
        )
        stacked_inputs_cpu = torch.zeros(
            E, max_tokens, H,
            dtype=flat_hidden_states_cpu_pin.dtype, device="cpu"
        )

        # 直接从 flat_hidden_states 复制需要的 token 到 stacked_inputs 的对应位置
        for i, expert_idx in enumerate(expert_idx_list):
            start_idx, end_idx = expert_indices_map[expert_idx]
            token_ids = token_idxs_cpu_pin[start_idx:end_idx]
            num_tokens = token_ids.shape[0]
            
            # 使用 blocking copy 以确保数据完整性（特别是跨设备复制时）
            stacked_inputs_cpu[i, :num_tokens].copy_(flat_hidden_states_cpu_pin[token_ids], non_blocking=True)
        cuda_hook_time_end("move_flat_hidden2cpu")
        gpinpool.free(flat_hidden_states_cpu_pin)
        # gpinpool.free(token_idxs_cpu_pin)

        time_start_pad = time.time()
        cuda_hook("group pad")
        # 过滤有效的 expert indices
        expert_indices = [idx for idx in expert_idx_list if idx in expert_token_indices_map]
        if not expert_indices:
            return final_hidden_states
        
        # 计算 max_tokens（直接从索引信息计算，避免创建 tensor）
        max_tokens = max(
            expert_token_indices_map[eid].shape[0] 
            for eid in expert_indices
        )
        

        # 获取 hidden_dim
        H = flat_hidden_states.shape[1]  # hidden_dim
        E = len(expert_indices)  # expert 数量
        
        # 优化：直接分配整个 stacked_inputs tensor，避免多次分配和 stack 操作
        # 形状: [E, max_tokens, H]
        stacked_inputs = torch.zeros(
            E, max_tokens, H,
            dtype=flat_hidden_states.dtype, device=flat_hidden_states.device
        )
        
        # 直接从 flat_hidden_states 复制需要的 token 到 stacked_inputs 的对应位置
        for i, expert_idx in enumerate(expert_indices):
            token_ids = expert_token_indices_map[expert_idx]
            num_tokens = token_ids.shape[0]
            
            # 使用 blocking copy 以确保数据完整性（特别是跨设备复制时）
            stacked_inputs[i, :num_tokens].copy_(flat_hidden_states[token_ids], non_blocking=False)
            # padding 部分保持未初始化（如果后续不需要0值，可以跳过 zero_）
            # 如果需要确保 padding 为0，取消下面的注释
            # stacked_inputs[i, num_tokens:].zero_()
        
        cuda_hook_end("group pad")
        logger.debug(f"pad cost {time.time() - time_start_pad} s")

        time_start_create_cpu = time.time()
        cuda_hook("group stack")
        outputs_pin = gpinpool.alloc_same_pin_tensor(stacked_inputs)
        stacked_inputs_cpu_pin = gpinpool.alloc_same_pin_tensor(stacked_inputs)
        logger.debug(f"create cpu tensor cost {time.time() - time_start_create_cpu} s")
        time_start_move2cpu = time.time()
        stacked_inputs_cpu_pin.copy_(stacked_inputs, non_blocking=False)
        stacked_inputs_gpu = stacked_inputs
        stacked_inputs = stacked_inputs_cpu_pin
        logger.debug(f"move to cpu cost {time.time() - time_start_move2cpu} s")
        cuda_hook_end("group stack")

        time_start_einsum = time.time()
        cuda_hook("group_einsum")
        # 使用 einsum 批量计算
        # w1_out: [E, max_tokens, I]
        w1_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w1)
        act_fn = ACT2FN[self.config.hidden_act]
        w1_out = act_fn(w1_out)
        

        # 安全地检查 group_w3，避免 Bus error
        # 先检查基本属性，不访问数据
        logger.debug(f"group_w3: shape={group_w3.shape}, device={group_w3.device}, dtype={group_w3.dtype}, numel={group_w3.numel()}")
        logger.debug(f"group_w3 strides: {group_w3.stride()}, is_contiguous: {group_w3.is_contiguous()}")
        # 只检查第一个元素，避免访问整个 tensor
        try:
            if group_w3.numel() > 0:
                first_elem = group_w3.view(-1)[0]
                logger.debug(f"group_w3 first element: {first_elem.item()}")
                if torch.isnan(first_elem) or torch.isinf(first_elem):
                    logger.warning("group_w3 first element contains inf or nan!")
        except Exception as e:
            logger.error(f"Error accessing group_w3 first element: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # C++ 端已修复映射问题，确保内存映射正确且连续，无需复制
        logger.warning(f"start einsum2")
        # w3_out: [E, max_tokens, I]
        w3_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w3)
        
        # intermediate: [E, max_tokens, I]
        intermediate = w1_out * w3_out
        
        # outputs: [E, max_tokens, H] - 使用预分配的 tensor（先计算再复制）
        # 先计算结果，然后复制到预分配的 tensor（因为某些 PyTorch 版本不支持 einsum 的 out 参数）
        outputs_result = torch.einsum('eti,ehi->eth', intermediate, group_w2)
        logger.debug(f"group einsum cost {time.time() - time_start_einsum} s")
        cuda_hook_end("group_einsum")

        cuda_hook("cpy2cpu2gpu_tensor")
        time_start_cpy = time.time()
        outputs_pin.copy_(outputs_result, non_blocking=False)
        outputs = outputs_pin
        outputs_cpu = outputs
        cuda_hook_end("cpy2cputensor")
        logger.debug(f"cpy2cputensor cost {time.time() - time_start_cpy} s")

        cuda_hook_time("move_outputs")
        outputs = outputs.to(final_hidden_states.device, non_blocking=False)
        cuda_hook_time_end("move_outputs")
        cuda_hook("final_hidden_states scatter")
        # 一次性提交
        
        # 提取有效结果（去除 padding）
        for i, expert_idx in enumerate(expert_indices):
            # 直接从索引信息获取 token 数量，避免访问 tensor
            token_ids = expert_token_indices_map[expert_idx]
            num_tokens = token_ids.shape[0]
            
            # 使用切片 [:num_tokens] 创建 view，避免拷贝
            expert_out = outputs[i][:num_tokens]
            # 多次提交
            # expert_out = expert_out.to(final_hidden_states.device, non_blocking=True)
            
            # 直接从 flat_experts_weight 获取 weights，避免中间拷贝
            start_idx, end_idx = expert_indices_map[expert_idx]
            expert_weights = flat_experts_weight[idxs[start_idx:end_idx]]
            
            expert_out = expert_out.mul_(expert_weights)
            
            # 使用 token_ids 进行 scatter
            final_hidden_states.scatter_reduce_(
                dim=0,
                index=token_ids.view(-1, 1).repeat(1, final_hidden_states.shape[-1]),
                src=expert_out,
                reduce='sum'
            )
        cuda_hook_end("final_hidden_states scatter")
        
        # 在 scatter 操作完成后再释放内存
        # 确保所有对 outputs 的访问都已完成
        gpinpool.free(outputs_pin)
        gpinpool.free(stacked_inputs_cpu_pin)

        time_einsum_end = time.time()
        result = ExpertEinsumResult(final_hidden_states=outputs_cpu, time_einsum_end=time_einsum_end)
        
        output_queue.put(result)
        del group_w1, group_w2, group_w3
        return final_hidden_states

    @torch.no_grad()
    def einsum_with_group_tensors_mp(self, 
        hmv: "HostMemoryView",
        layer_idx: int, expert_idx_list: list[int], 
        expert_indices_map: Dict[int, Tuple[int, int]],  # {expert_id: (start_idx, end_idx)}
        flat_hidden_states: torch.Tensor,  # 原始展平的 hidden states
        idxs: torch.Tensor,
        output_queue
    ):
        """
        使用 group tensors 进行批量 einsum 计算
        
        Args:
            layer_idx: 层索引
            expert_idx_list: expert 索引列表
            expert_indices_map: {expert_id: (start_idx, end_idx)} 索引范围
            expert_token_indices_map: {expert_id: token_ids} token 索引
            flat_hidden_states: 原始展平的 hidden states
            flat_experts_weight: 原始展平的 experts weight
            idxs: 排序后的索引
            expert_token_indices_map_full: 完整的 token indices map（用于 scatter）
        
        Returns:
            final_hidden_states: 最终隐藏状态
        """
        cuda_hook_time("move_flatidxs")
        flat_hidden_states_cpu_pin = gpinpool.alloc_same_pin_tensor(flat_hidden_states)
        flat_hidden_states_cpu_pin.copy_(flat_hidden_states, non_blocking=False)
        token_idxs = idxs // self.config.num_experts_per_tok
        token_idxs_cpu_pin = token_idxs.cpu()
        cuda_hook_time_end("move_flatidxs")

        cuda_hook_time("group group_tensors")
        # 获取 group tensors (已经是堆叠好的 [E, ...] 形状)
        group_dict = hmv.group_experts_tensor(layer_idx, expert_idx_list)
        group_w1 = group_dict['group_w1']  # [E, I, H]
        group_w2 = group_dict['group_w2']  # [E, H, I]
        group_w3 = group_dict['group_w3']  # [E, I, H]
        
        cuda_hook_time_end("group_tensors")
        
        expert_token_indices_map = {
            expert_idx: 
            token_idxs_cpu_pin[expert_indices_map[expert_idx][0]:expert_indices_map[expert_idx][1]] 
            for expert_idx in expert_idx_list
        }

        cuda_hook_time("group pad")
        # 计算 max_tokens（直接从索引信息计算，避免创建 tensor）
        max_tokens = max(
            expert_token_indices_map[expert_idx].shape[0] 
            for expert_idx in expert_idx_list
        )
        
        # 获取 hidden_dim
        H = flat_hidden_states_cpu_pin.shape[1]  # hidden_dim
        E = len(expert_idx_list)  # expert 数量
        
        # 优化：直接分配整个 stacked_inputs tensor，避免多次分配和 stack 操作
        # 形状: [E, max_tokens, H]
        stacked_inputs = torch.zeros(
            E, max_tokens, H,
            dtype=flat_hidden_states_cpu_pin.dtype, device="cpu"
        )
        
        # 直接从 flat_hidden_states 复制需要的 token 到 stacked_inputs 的对应位置
        for i, expert_idx in enumerate(expert_idx_list):
            token_ids = expert_token_indices_map[expert_idx]
            num_tokens = token_ids.shape[0]
            
            # 使用 blocking copy 以确保数据完整性（特别是跨设备复制时）
            stacked_inputs[i, :num_tokens].copy_(flat_hidden_states_cpu_pin[token_ids], non_blocking=True)
            # padding 部分保持未初始化（如果后续不需要0值，可以跳过 zero_）
            # 如果需要确保 padding 为0，取消下面的注释
            # stacked_inputs[i, num_tokens:].zero_()
        
        cuda_hook_time_end("group pad")

        cuda_hook_time("group_einsum")
        # 使用 einsum 批量计算
        # w1_out: [E, max_tokens, I]
        w1_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w1)
        act_fn = ACT2FN[self.config.hidden_act]
        w1_out = act_fn(w1_out)
        # w3_out: [E, max_tokens, I]
        w3_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w3)
        # intermediate: [E, max_tokens, I]
        intermediate = w1_out * w3_out
        # outputs: [E, max_tokens, H] - 使用预分配的 tensor（先计算再复制）
        # 先计算结果，然后复制到预分配的 tensor（因为某些 PyTorch 版本不支持 einsum 的 out 参数）
        outputs_result = torch.einsum('eti,ehi->eth', intermediate, group_w2)
        cuda_hook_time_end("group_einsum")
        

        result = ExpertEinsumResult(final_hidden_states=outputs_result, time_einsum_end=time.time())
        output_queue.put(result)
        del group_w1, group_w2, group_w3
        group_list = []
        # group_list.append(group_w1)
        # group_list.append(group_w2)
        # group_list.append(group_w3)
        return outputs_result, group_list