import enum
import time
import os
import torch
import torch.nn.functional as F
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
from models.Deepseek.deepseek_v2_lite.modeling_deepseek import DeepseekV2ForCausalLM, DeepseekV2DecoderLayer
from models.Deepseek.mlpmodule import DeepseekModule, DeepseekOCalModel
from models.Mixtral.mlpmodule import MixtralModule
from models.Qwen.mlpmodule import Qwen2MoEModule, Qwen3MoEModule, Qwen3_5MoEModule
from models.Gemma.mlpmodule import Gemma4Module
from models.GPT.mlpmodule import GptOssModule
from models.Erine.mlpmodule import ErineMoeModule
# from models.Qwen.Qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeDecoderLayer
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeDecoderLayer, Qwen2MoeRotaryEmbedding
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralDecoderLayer, MixtralRotaryEmbedding
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextRotaryEmbedding,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssDecoderLayer, GptOssRotaryEmbedding
from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import (
    Ernie4_5_MoeForCausalLM,
    Ernie4_5_MoeDecoderLayer,
    Ernie4_5_MoeRotaryEmbedding,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4TextDecoderLayer
)
from utils.logger import init_logger
from utils.cuda_h import *
from lmp.pinpool import gpinpool
from dataclasses import dataclass

logger = init_logger(__name__)

"""
MoE / MLP 推理侧封装：按模型类型（Deepseek、Mixtral、Qwen2/3/3.5 MoE、Gemma4、GPT-OSS、Ernie4.5 MoE）分发子模块调用，
并提供专家权重分组、einsum/bmm 批量计算、多 GPU scatter 等路径。

本文件核心类型为 ``MLPModuleWrapper``：不负责训练，主要服务于 LP/共享内存加载与 CETM 等推理管线。
"""

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
QWEN2_MODEL_NAME_TYPE = "Qwen2_moe"
QWEN3_MODEL_NAME_TYPE = "Qwen3_moe"
QWEN3_5_MODEL_NAME_TYPE = "Qwen3_5_moe"
DEEPSEEK_V2_LITE="DeepSeek-V2-Lite"
GEMMA4_MODEL_NAME_TYPE = "Gemma4"
ERINE_MODEL_NAME_TYPE = "Erine"
GPT_OSS_MODEL_NAME_TYPE = "GPT-OSS"
# original_dtype = torch.get_default_dtype()
# torch.set_default_dtype(torch.bfloat16)


def _normalize_model_name_type(model_name_type: str) -> str:
    """
    Map folder / config style names to internal ``*_MODEL_NAME_TYPE`` constants.

    Accepts e.g. ``qwen3_5_moe``, ``Qwen3_5``, ``gpt_oss``, ``gemma4``, ``ernie4_5_moe``, ``erine4_5_moe``.
    """
    if not model_name_type:
        return model_name_type
    raw = model_name_type.strip()
    key = raw.lower().replace("-", "_").replace(".", "_")
    aliases = {
        "qwen3_5": QWEN3_5_MODEL_NAME_TYPE,
        "qwen3_5_moe": QWEN3_5_MODEL_NAME_TYPE,
        "gpt_oss": GPT_OSS_MODEL_NAME_TYPE,
        "gptoss": GPT_OSS_MODEL_NAME_TYPE,
        "gemma_4": GEMMA4_MODEL_NAME_TYPE,
        "ernie4_5_moe": ERINE_MODEL_NAME_TYPE,
        "ernie_4_5_moe": ERINE_MODEL_NAME_TYPE,
        "erine4_5_moe": ERINE_MODEL_NAME_TYPE,
        "erine": ERINE_MODEL_NAME_TYPE,
    }
    if key in aliases:
        return aliases[key]
    for canon in (
        DEEPSEEK_MODEL_NAME_TYPE,
        MIXTRAL_MODEL_NAME_TYPE,
        QWEN2_MODEL_NAME_TYPE,
        QWEN3_MODEL_NAME_TYPE,
        QWEN3_5_MODEL_NAME_TYPE,
        GEMMA4_MODEL_NAME_TYPE,
        ERINE_MODEL_NAME_TYPE,
        GPT_OSS_MODEL_NAME_TYPE,
    ):
        if key == canon.lower().replace("-", "_"):
            return canon
    return raw


def _fused_expert_gate_up_w123(expert_module):
    """``gate_up_proj`` [2I,H] + ``down_proj`` [H,I] -> (w_gate [I,H], w_down [H,I], w_up [I,H]) for MoE einsum packing."""
    w = expert_module.gate_up_proj.weight
    half = w.shape[0] // 2
    return w[:half], expert_module.down_proj.weight, w[half:]


@torch.no_grad()
def _scatter_fused_moe_expert_output(
    expert_mlp,
    tokens: torch.Tensor,
    weights: torch.Tensor,
    token_indices: torch.Tensor,
    final_hidden_states: torch.Tensor,
):
    """单 expert 前向（``gate_up_proj`` + ``down_proj``）并按 token 聚合到 ``final_hidden_states``。"""
    out = expert_mlp(tokens)
    out.mul_(weights)
    if out.device != final_hidden_states.device:
        out = out.to(final_hidden_states.device, non_blocking=True)
    final_hidden_states.scatter_reduce_(
        dim=0,
        index=token_indices.view(-1, 1).repeat(1, final_hidden_states.shape[-1]),
        src=out,
        reduce="sum",
    )

class WeightType(enum.Enum):
    """专家线性层在 checkpoint / 共享内存中的类型枚举（与各模型 gate/up/down 命名对齐）。"""
    W1 = 1
    W2 = 2
    W3 = 3

class MLPModuleWrapper:
    """
    各 MoE/Dense 模型的 MLP 与专家计算统一入口。

    职责概览：
    - 解析 ``model_path``、加载 ``config``，并选择对应的 ``*Module`` 元类；
    - 提供空模型构建、层模块实例化、Host 权重写回 GPU 模型等初始化能力；
    - 暴露 gate、注意力、LayerNorm、dense MoE、逐 expert / 批量 expert 等前向片段，供上层 LP 调度。
    """
    def __init__(self, model_name_type: str, model_path: str):
        """根据模型类型与相对仓库路径初始化包装器，并解析 ``text_config``（Gemma4 多模态时）。"""
        self.model_name_type = _normalize_model_name_type(model_name_type)
        self.model_path = model_path
        self.model_abs_path = os.path.join(STORAGE_PATH, model_path)
        self._raw_config = None
        self._gemma4_uses_language_model = False
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            self.model_class = DeepseekModule()
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            self.model_class = MixtralModule()
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            self.model_class = Qwen2MoEModule()
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            self.model_class = Qwen3MoEModule()
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            self.model_class = Qwen3_5MoEModule()
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            self.model_class = Gemma4Module()
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            self.model_class = GptOssModule()
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            self.model_class = ErineMoeModule()
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

        # 只支持 text_config：若为 Gemma4 多模态顶层 config，则直接取 text_config。
        cfg = self.model_class.get_config(self.model_abs_path)
        self._raw_config = cfg
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE and hasattr(cfg, "text_config"):
            self._gemma4_uses_language_model = True
            self.config = cfg.text_config
        else:
            self.config = cfg

    def _gemma4_weight_prefix(self) -> str:
        """返回 Gemma4 权重在 state_dict 中的前缀（多模态为 ``model.language_model``）。"""
        # Gemma4ForConditionalGeneration 的 text tower 在 model.language_model.*
        return "model.language_model" if self._gemma4_uses_language_model else "model"

    def _ernie_layer_is_sparse_moe(self, layer_idx: int) -> bool:
        """Ernie4.5：仅部分层为 ``Ernie4_5_MoeSparseMoeBlock``，与 ``modeling_ernie4_5_moe`` 条件一致。"""
        c = self.config
        return (
            ((layer_idx + 1) % c.moe_layer_interval == 0)
            and layer_idx >= c.moe_layer_start_index
            and layer_idx <= c.moe_layer_end_index
        )

    @staticmethod
    def _pack_group_w1_w3_no_stack(
        group_w1_list: list[torch.Tensor],
        group_w3_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        将每个 expert 的 gate（w1）与 up（w3）在中间维拼接为一块连续权重。

        输入为长度 E 的列表，元素形状 ``[I, H]``；输出 ``[E, 2*I, H]``。
        使用 ``empty`` + 循环 ``copy_``，避免 ``torch.stack`` 的额外内核路径。
        """
        E = len(group_w1_list)
        if E == 0:
            return torch.empty(0, 0, 0)
        w1_0 = group_w1_list[0]
        I, H = w1_0.shape[0], w1_0.shape[1]
        out = torch.empty(E, 2 * I, H, device=w1_0.device, dtype=w1_0.dtype)
        for i, (w1, w3) in enumerate(zip(group_w1_list, group_w3_list)):
            out[i, :I, :].copy_(w1)
            out[i, I:, :].copy_(w3)
        return out

    @staticmethod
    def _pack_group_w2_no_stack(group_w2_list: list[torch.Tensor]) -> torch.Tensor:
        """
        将每个 expert 的 down（w2）权重拷贝到批维张量 ``[E, H, I]``。

        使用 ``empty`` + ``copy_``，避免 ``torch.stack``。
        """
        E = len(group_w2_list)
        if E == 0:
            return torch.empty(0, 0, 0)
        w2_0 = group_w2_list[0]
        Hi, Ii = w2_0.shape[0], w2_0.shape[1]
        out = torch.empty(E, Hi, Ii, device=w2_0.device, dtype=w2_0.dtype)
        for i, w2 in enumerate(group_w2_list):
            out[i].copy_(w2)
        return out

    @staticmethod
    def _pack_batched_w1_w3(group_w1: torch.Tensor, group_w3: torch.Tensor) -> torch.Tensor:
        """
        在已为 ``[E, I, H]`` 的批张量上，沿 dim=1 拼接 w1/w3 得到 ``[E, 2*I, H]``。

        用于共享内存恢复后的张量融合 gate+up 一次线性；显式 ``copy_`` 保证连续布局。
        """
        E, I, H = group_w1.shape
        if group_w3.shape != (E, I, H):
            raise ValueError(f"group_w3 shape {group_w3.shape} != group_w1 {(E, I, H)}")
        out = torch.empty(E, 2 * I, H, device=group_w1.device, dtype=group_w1.dtype)
        out[:, :I, :].copy_(group_w1)
        out[:, I:, :].copy_(group_w3)
        return out

    @staticmethod
    def _coerce_expert_group_w123(group_dict: Dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从 ``group_experts_tensor`` 返回的字典中取出 ``(group_w1, group_w2, group_w3)``。

        兼容：
        - 已由 ``HostMemoryView.group_experts_tensor`` 填好的 ``group_w1`` / ``group_w2`` / ``group_w3``；
        - 或 ``group_w1_w3`` + ``group_w2``（gate+up 已拼成 ``[E,2I,H]``）；
        - 或 C++ 旧三 key ``group_{0,1,2}_big_tensor``；
        - 或仅两 key：``group_0_big_tensor`` 为交错 w1/w3 的 ``[2E,I,H]``、``group_1_big_tensor`` 为 w2。
        """
        if "group_w1" in group_dict and "group_w2" in group_dict and "group_w3" in group_dict:
            return group_dict["group_w1"], group_dict["group_w2"], group_dict["group_w3"]
        if "group_w1_w3" in group_dict and "group_w2" in group_dict:
            gw = group_dict["group_w1_w3"]
            half = gw.shape[1] // 2
            return gw[:, :half, :], group_dict["group_w2"], gw[:, half:, :]
        if "group_0_big_tensor" in group_dict and "group_2_big_tensor" in group_dict:
            return (
                group_dict["group_0_big_tensor"],
                group_dict["group_1_big_tensor"],
                group_dict["group_2_big_tensor"],
            )
        if "group_0_big_tensor" in group_dict and "group_1_big_tensor" in group_dict:
            g0 = group_dict["group_0_big_tensor"]
            if g0.dim() != 3 or g0.shape[0] % 2 != 0:
                raise ValueError(
                    f"group_0_big_tensor expected [2*E, I, H] for merged w1+w3, got {tuple(g0.shape)}"
                )
            e = g0.shape[0] // 2
            i_dim, h_dim = g0.shape[1], g0.shape[2]
            if g0.is_contiguous():
                gw = g0.view(e, 2 * i_dim, h_dim)
            else:
                logger.error(
                    f"group_0_big_tensor is not contiguous; reshaping may be inefficient for [2*E, I, H] -> [E, 2*I, H]"
                )
                gw = g0.reshape(e, 2, i_dim, h_dim).flatten(1, 2)
            half = gw.shape[1] // 2
            return gw[:, :half, :], group_dict["group_1_big_tensor"], gw[:, half:, :]
        raise KeyError(
            "group_dict missing expert tensors; expected group_w1/w2/w3 or group_w1_w3+w2 or group_0/1/2"
        )

    def _moe_gate_up_from_stacked(self, stacked_inputs: torch.Tensor, group_w1_w3: torch.Tensor):
        """
        对 padding 后的专家输入做一次 ``einsum`` 完成 gate 与 up 的线性，再对 gate 半幅做激活后与 up 相乘。

        ``stacked_inputs``: ``[E, T, H]``；``group_w1_w3``: ``[E, 2*I, H]``（前半为 gate，后半为 up）。
        返回 SwiGLU 前的逐元素乘结果，形状 ``[E, T, I]``。
        """
        gate_up = torch.einsum("eth,eih->eti", stacked_inputs, group_w1_w3)
        I_half = group_w1_w3.shape[1] // 2
        w1_out, w3_out = gate_up.split(I_half, dim=-1)
        act_fn = ACT2FN[self.config.hidden_act]
        w1_out = act_fn(w1_out)
        return w1_out * w3_out

    def _moe_gate_up_bmm_from_stacked(self, stacked_inputs: torch.Tensor, group_w1_w3: torch.Tensor):
        """
        CPU 侧与 ``_moe_gate_up_from_stacked`` 等价的融合路径，使用 ``torch.bmm``。

        先将 ``group_w1_w3`` 转为 ``[E, H, 2*I]``，再 ``bmm``、切分、激活 gate 后与 up 相乘。
        """
        w_t = group_w1_w3.transpose(1, 2)
        gate_up = torch.bmm(stacked_inputs, w_t)
        I_half = group_w1_w3.shape[1] // 2
        w1_out, w3_out = gate_up.split(I_half, dim=-1)
        act_fn = ACT2FN[self.config.hidden_act]
        w1_out = act_fn(w1_out)
        return w1_out * w3_out

        
    def init_chmv_meta_model(self, cmv: "CudaMemoryView", hmv: "HostMemoryView", device=None):
        """
        在 ``init_empty_weights`` 上下文中构建 GPU 侧元模型 ``cmv.mlpm_ci``，供后续按需加载权重。

        各分支写入 ``cmv.mlpm_ci`` 并 ``eval()``；部分模型会设置 ``rotary_emb`` 等到指定 ``device``。
        """
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            if self.model_path == DEEPSEEK_V2_LITE:
                with init_empty_weights():
                    self.config._attn_implementation = "eager"
                    # cm = DeepseekOCalModel(self.config)
                    cm = DeepseekV2ForCausalLM(self.config)
                    cm.to(self.config.torch_dtype)
                    cm.eval()
                    cmv.mlpm_ci = cm
            else:
                with init_empty_weights():
                    self.config._attn_implementation = "sdpa"
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
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            with init_empty_weights():
                self.config._attn_implementation = "sdpa"
                cm = Qwen2MoeForCausalLM(self.config)
                cm.model.rotary_emb = Qwen2MoeRotaryEmbedding(config=self.config, device=device)
                for i in range(self.config.num_hidden_layers):
                    cm.model.layers[i].self_attn.rotary_emb = \
                        Qwen2MoeRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.torch_dtype)
                cm.eval()
                cmv.mlpm_ci = cm
                return
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            with init_empty_weights():
                cm = Qwen3MoeForCausalLM(self.config)
                cm.model.rotary_emb = Qwen3MoeRotaryEmbedding(config=self.config, device=device)
                # for i in range(self.config.num_hidden_layers):
                #     cm.model.layers[i].self_attn.rotary_emb = \
                #         Qwen3MoeRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.torch_dtype)
                cm.eval()
                cmv.mlpm_ci = cm
                return
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            with init_empty_weights():
                self.config._attn_implementation = "sdpa"
                cm = MixtralForCausalLM(self.config)
                for i in range(self.config.num_hidden_layers):
                    self_attn = cm.model.layers[i].self_attn
                    self_attn.rotary_emb = \
                        MixtralRotaryEmbedding(
                            dim=self_attn.head_dim,
                            max_position_embeddings=self.config.max_position_embeddings,
                            base=self.config.rope_theta,
                            device=device
                        )
                cm.to(self.config.torch_dtype)
                cm.eval()
                cmv.mlpm_ci = cm
                return
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            with init_empty_weights():
                # Gemma4 text attention 需要 position_embeddings；这里启用 sdpa（与现有 mask 构造兼容）。
                self.config._attn_implementation = "sdpa"
                if self._gemma4_uses_language_model:
                    cm = Gemma4ForConditionalGeneration(self._raw_config)
                else:
                    cm = Gemma4ForCausalLM(self.config)
                cm.to(self.config.torch_dtype)
                cm.eval()
                cmv.mlpm_ci = cm
                return
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            with init_empty_weights():
                self.config._attn_implementation = "sdpa"
                cm = Qwen3_5MoeForCausalLM(self.config)
                cm.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.torch_dtype)
                cm.eval()
                cmv.mlpm_ci = cm
                return
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            with init_empty_weights():
                cm = GptOssForCausalLM(self.config)
                cm.model.rotary_emb = GptOssRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.torch_dtype)
                cm.eval()
                cmv.mlpm_ci = cm
                return
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            with init_empty_weights():
                self.config._attn_implementation = "sdpa"
                cm = Ernie4_5_MoeForCausalLM(self.config)
                cm.model.rotary_emb = Ernie4_5_MoeRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.torch_dtype)
                cm.eval()
                cmv.mlpm_ci = cm
                return
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def init_set_layer_func(
        self, layer_idx: int, config, model
    ):
        """
        构造单层 ``DecoderLayer``，设置 dtype 与 ``eval()``，并挂到 ``model.model.layers[layer_idx]``。

        用于 Deepseek 在 ``init_empty_weights`` 外对整层 ``to(dtype)`` 以降低上下文开销。
        """
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            if self.model_path == DEEPSEEK_V2_LITE:
                with init_empty_weights():
                    layer = DeepseekV2DecoderLayer(config, layer_idx)
            else:
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
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen2MoeDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen3MoeDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = MixtralDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Gemma4TextDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen3_5MoeDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = GptOssDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Ernie4_5_MoeDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def init_layer_func(
        self, layer_idx: int, config):
        """
        仅构造并返回单层模块（不挂到 ``model``），用于与 ``init_set_layer_func`` 不同的初始化流程。

        各分支在 ``init_empty_weights`` 内实例化对应 ``DecoderLayer`` 后 ``to(dtype)`` 与 ``eval()``。
        """
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            if self.model_path == DEEPSEEK_V2_LITE:
                with init_empty_weights():
                    layer = DeepseekV2DecoderLayer(config, layer_idx)
            else:
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
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen2MoeDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen3MoeDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = MixtralDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Gemma4TextDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen3_5MoeDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = GptOssDecoderLayer(config, layer_idx)
                layer.to(config.torch_dtype)
                layer.eval()
                return layer
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Ernie4_5_MoeDecoderLayer(config, layer_idx)
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
        
        with init_empty_weights():
            if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
                if self.model_path == DEEPSEEK_V2_LITE:
                    self.config._attn_implementation = "eager"
                    model = DeepseekV2ForCausalLM(self.config)
                else:
                    self.config._attn_implementation = "sdpa"
                    model = DeepseekForCausalLM(self.config)
            elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
                model = Qwen2MoeForCausalLM(self.config)
            elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
                model = Qwen3MoeForCausalLM(self.config)
            elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
                model = AutoModelForCausalLM.from_config(
                    self.config, trust_remote_code=True
                )
            elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
                self.config._attn_implementation = "sdpa"
                if self._gemma4_uses_language_model:
                    model = Gemma4ForConditionalGeneration(self._raw_config)
                else:
                    model = Gemma4ForCausalLM(self.config)
            elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
                self.config._attn_implementation = "sdpa"
                model = Qwen3_5MoeForCausalLM(self.config)
            elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
                model = GptOssForCausalLM(self.config)
            elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
                self.config._attn_implementation = "sdpa"
                model = Ernie4_5_MoeForCausalLM(self.config)
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
        将 Host 侧 ``hm_state_dict`` 中的专家/共享专家权重写回 ``model`` 对应参数（``set_module_tensor_to_device``）。

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
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            expert_indicators = [".mlp.experts.", ".mlp.shared_experts."]
            target_linears = {"gate_proj", "up_proj", "down_proj"}
            updated_params = 0
            with torch.no_grad():
                for name, tensor in hm_state_dict.items():
                    if not any(ind in name for ind in expert_indicators):
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
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            expert_indicators = [".mlp.experts.experts.", ".mlp.shared_expert."]
            target_linears = {"gate_up_proj", "down_proj", "gate_proj", "up_proj"}
            updated_params = 0
            with torch.no_grad():
                for name, tensor in hm_state_dict.items():
                    if not any(ind in name for ind in expert_indicators):
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
                "restore_hm_state_dict2model loaded %d expert tensors for Qwen3 model",
                updated_params,
            )
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            expert_indicators = [".mlp.experts.experts.", ".mlp.shared_expert.", ".mlp.shared_expert_gate."]
            target_linears = {"gate_up_proj", "down_proj", "gate_proj", "up_proj", "shared_expert_gate"}
            updated_params = 0
            with torch.no_grad():
                for name, tensor in hm_state_dict.items():
                    if not any(ind in name for ind in expert_indicators):
                        continue
                    line_segments = name.split(".")
                    linear_pos = next(
                        (idx for idx, token in enumerate(line_segments) if token in target_linears),
                        -1,
                    )
                    if linear_pos == -1:
                        continue
                    try:
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
                "restore_hm_state_dict2model loaded %d expert tensors for Qwen3_5 model",
                updated_params,
            )
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            expert_indicators = [".mlp.experts.experts."]
            target_linears = {"gate_up_proj", "down_proj"}
            updated_params = 0
            with torch.no_grad():
                for name, tensor in hm_state_dict.items():
                    if not any(ind in name for ind in expert_indicators):
                        continue
                    line_segments = name.split(".")
                    linear_pos = next(
                        (idx for idx, token in enumerate(line_segments) if token in target_linears),
                        -1,
                    )
                    if linear_pos == -1:
                        continue
                    try:
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
                "restore_hm_state_dict2model loaded %d expert tensors for GPT-OSS model",
                updated_params,
            )
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            expert_indicators = [".mlp.experts.experts.", ".mlp.shared_experts."]
            target_linears = {"gate_up_proj", "down_proj", "gate_proj", "up_proj"}
            updated_params = 0
            with torch.no_grad():
                for name, tensor in hm_state_dict.items():
                    if not any(ind in name for ind in expert_indicators):
                        continue
                    line_segments = name.split(".")
                    linear_pos = next(
                        (idx for idx, token in enumerate(line_segments) if token in target_linears),
                        -1,
                    )
                    if linear_pos == -1:
                        continue
                    try:
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
                "restore_hm_state_dict2model loaded %d expert tensors for Ernie MoE model",
                updated_params,
            )
        else:
            logger.warning(f"restore_hm_state_dict2model not implemented for {self.model_name_type}")
            pass

    def get_experts_num(self):
        """返回当前配置下每层路由专家数量（不含 shared 语义，具体见各模型 config 字段名）。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            return self.config.n_routed_experts
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            return self.config.num_local_experts
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            return self.config.num_experts
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            return self.config.num_experts
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            return self.config.num_experts
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            return self.config.num_experts
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            return self.config.num_local_experts
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            return self.config.moe_num_experts
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_experts_names_w(self, layer_idx: int, experts_idx_list, type_idx: WeightType):
        """
        按 ``WeightType`` 生成若干 expert 的完整参数名列表，供共享内存按名恢复。

        ``type_idx`` 与 gate/down/up（或 Mixtral w1/w2/w3）的字符串映射因模型而异。
        """
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
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            # 顺序很重要，对齐 w1 w2 w3
            type_str_list = ["none", "gate_proj", "down_proj", "up_proj"]
            type_str = type_str_list[type_idx.value]
            experts_names = [f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{type_str}.weight" for expert_idx in experts_idx_list]
            return experts_names
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_up_proj", "down_proj"]
            type_str = type_str_list[type_idx.value]
            if type_idx.value == 0:
                return []
            experts_names = [
                f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.{type_str}.weight"
                for expert_idx in experts_idx_list
            ]
            return experts_names
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_up_proj", "down_proj"]
            type_str = type_str_list[type_idx.value]
            if type_idx.value == 0:
                return []
            experts_names = [
                f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.{type_str}.weight"
                for expert_idx in experts_idx_list
            ]
            return experts_names
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_up_proj", "down_proj"]
            type_str = type_str_list[type_idx.value]
            if type_idx.value == 0:
                return []
            experts_names = [
                f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.{type_str}.weight"
                for expert_idx in experts_idx_list
            ]
            return experts_names
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_up_proj", "down_proj"]
            type_str = type_str_list[type_idx.value]
            if type_idx.value == 0:
                return []
            experts_names = [
                f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.{type_str}.weight"
                for expert_idx in experts_idx_list
            ]
            return experts_names
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_up_proj", "down_proj"]
            type_str = type_str_list[type_idx.value]
            if type_idx.value == 0:
                return []
            p = self._gemma4_weight_prefix()
            return [
                f"{p}.layers.{layer_idx}.experts.{expert_idx}.{type_str}.weight"
                for expert_idx in experts_idx_list
            ]

        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_tensor_index_general_names(self):
        """返回与专家无关、需在 tensor 索引中一并处理的通用权重名（embedding、lm_head、final norm 等）。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            return ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            return ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            return ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            return ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            return ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            return ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            return ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            p = self._gemma4_weight_prefix()
            return [
                "lm_head.weight",
                f"{p}.embed_tokens.weight",
                f"{p}.norm.weight",
            ]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_shared_experts_names(self, layer_idx: int):
        """返回指定层 shared expert（若存在）相关权重名列表；无则返回空列表。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            if layer_idx < self.config.first_k_dense_replace:
                return []
            return [
                f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
            ]
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight",
            ]
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            # empty shared_experts
            return []
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            return []
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight",
            ]
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            return []
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            if not self._ernie_layer_is_sparse_moe(layer_idx):
                return []
            if getattr(self.config, "moe_num_shared_experts", 0) <= 0:
                return []
            return [
                f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
            ]
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            return []
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_experts_names(self, layer_idx: int, expert_idx_list: list[int]):
        """
        返回某层在 ``expert_idx_list`` 上需要加载/恢复的全部专家权重名（含每层 dense 前几层特例）。

        Gemma4 为逐专家 ``Linear`` 权重名；``expert_idx_list`` 为空时展开为该层全部专家。
        """
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            if layer_idx < self.config.first_k_dense_replace:
                return [
                    f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                    f"model.layers.{layer_idx}.mlp.down_proj.weight",
                    f"model.layers.{layer_idx}.mlp.up_proj.weight",
                ]
            else:
                names_list = []
                for expert_idx in expert_idx_list:
                    names_list.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
                    names_list.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")
                    names_list.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")
                return names_list
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            names_list = []
            for expert_idx in expert_idx_list:
                names_list.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
                names_list.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")
                names_list.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")
            return names_list
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            names_list = []
            for expert_idx in expert_idx_list:
                names_list.append(f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight")
                names_list.append(f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight")
                names_list.append(f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight")
            return names_list
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            names_list = []
            for expert_idx in expert_idx_list:
                names_list.append(
                    f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.gate_up_proj.weight"
                )
                names_list.append(
                    f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.down_proj.weight"
                )
            return names_list
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            names_list = [
                f"model.layers.{layer_idx}.mlp.gate.weight",
                f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight",
            ]
            expert_ids = expert_idx_list if expert_idx_list else list(range(self.get_experts_num()))
            for expert_idx in expert_ids:
                names_list.append(
                    f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.gate_up_proj.weight"
                )
                names_list.append(
                    f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.down_proj.weight"
                )
            return names_list
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            names_list = [
                f"model.layers.{layer_idx}.mlp.router.weight",
                f"model.layers.{layer_idx}.mlp.router.bias",
            ]
            expert_ids = expert_idx_list if expert_idx_list else list(range(self.get_experts_num()))
            for expert_idx in expert_ids:
                names_list.append(
                    f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.gate_up_proj.weight"
                )
                names_list.append(
                    f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.down_proj.weight"
                )
            return names_list
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            if not self._ernie_layer_is_sparse_moe(layer_idx):
                return [
                    f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                    f"model.layers.{layer_idx}.mlp.up_proj.weight",
                    f"model.layers.{layer_idx}.mlp.down_proj.weight",
                ]
            names_list = [
                f"model.layers.{layer_idx}.mlp.gate.weight",
                f"model.layers.{layer_idx}.mlp.gate.moe_statics.e_score_correction_bias",
            ]
            expert_ids = expert_idx_list if expert_idx_list else list(range(self.get_experts_num()))
            for expert_idx in expert_ids:
                names_list.append(
                    f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.gate_up_proj.weight"
                )
                names_list.append(
                    f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.down_proj.weight"
                )
            return names_list
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            p = self._gemma4_weight_prefix()
            names_list = [
                f"{p}.layers.{layer_idx}.router.proj.weight",
                f"{p}.layers.{layer_idx}.router.scale",
                f"{p}.layers.{layer_idx}.router.per_expert_scale",
            ]
            expert_ids = expert_idx_list if expert_idx_list else list(range(self.get_experts_num()))
            for expert_idx in expert_ids:
                names_list.append(f"{p}.layers.{layer_idx}.experts.{expert_idx}.gate_up_proj.weight")
                names_list.append(f"{p}.layers.{layer_idx}.experts.{expert_idx}.down_proj.weight")
            return names_list
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_gate_names(self, layer_idx: int):
        """返回 MoE 路由（gate）权重参数名；dense 前几层无 gate 时返回空列表。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            # first k dense with dense no gate
            # layer_idx start from 0, first_k_dense_replace start from 1
            if layer_idx < self.config.first_k_dense_replace:
                return []
            return [f"model.layers.{layer_idx}.mlp.gate.weight"]
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            return [f"model.layers.{layer_idx}.mlp.gate.weight"]
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            return [f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"]
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            return [f"model.layers.{layer_idx}.mlp.gate.weight"]
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            return [f"model.layers.{layer_idx}.mlp.gate.weight"]
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.mlp.router.weight",
                f"model.layers.{layer_idx}.mlp.router.bias",
            ]
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            if not self._ernie_layer_is_sparse_moe(layer_idx):
                return []
            return [f"model.layers.{layer_idx}.mlp.gate.weight"]
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            p = self._gemma4_weight_prefix()
            return [
                f"{p}.layers.{layer_idx}.router.proj.weight",
                f"{p}.layers.{layer_idx}.router.scale",
                f"{p}.layers.{layer_idx}.router.per_expert_scale",
            ]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_layernorm_names(self, layer_idx: int):
        """返回该层 input / post-attention LayerNorm 权重名（Gemma4 使用 text tower 前缀）。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.post_attention_layernorm.weight", 
                f"model.layers.{layer_idx}.input_layernorm.weight", ]
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.post_attention_layernorm.weight", 
                f"model.layers.{layer_idx}.input_layernorm.weight", ]
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.post_attention_layernorm.weight", 
                f"model.layers.{layer_idx}.input_layernorm.weight", ]
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.post_attention_layernorm.weight", 
                f"model.layers.{layer_idx}.input_layernorm.weight", ]
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.post_attention_layernorm.weight",
                f"model.layers.{layer_idx}.input_layernorm.weight",
            ]
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.post_attention_layernorm.weight",
                f"model.layers.{layer_idx}.input_layernorm.weight",
            ]
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.post_attention_layernorm.weight",
                f"model.layers.{layer_idx}.input_layernorm.weight",
            ]
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            p = self._gemma4_weight_prefix()
            return [
                f"{p}.layers.{layer_idx}.input_layernorm.weight",
                f"{p}.layers.{layer_idx}.post_attention_layernorm.weight",
                f"{p}.layers.{layer_idx}.post_feedforward_layernorm.weight",
                f"{p}.layers.{layer_idx}.post_feedforward_layernorm_1.weight",
                f"{p}.layers.{layer_idx}.post_feedforward_layernorm_2.weight",
                f"{p}.layers.{layer_idx}.pre_feedforward_layernorm.weight",
                f"{p}.layers.{layer_idx}.pre_feedforward_layernorm_1.weight",
                f"{p}.layers.{layer_idx}.pre_feedforward_layernorm_2.weight",
            ]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_attention_names(self, layer_idx: int):
        """返回该层 self-attention 相关权重名（含 Qwen 的 q_norm/k_norm、Deepseek-V2 MQA 等变体）。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            if self.model_path == DEEPSEEK_V2_LITE:
                return [
                    f"model.layers.{layer_idx}.self_attn.q_proj.weight", 
                    f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight", 
                    f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight", 
                    f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", 
                    f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
            else:
                return [
                    f"model.layers.{layer_idx}.self_attn.q_proj.weight", 
                    f"model.layers.{layer_idx}.self_attn.k_proj.weight", 
                    f"model.layers.{layer_idx}.self_attn.v_proj.weight", 
                    f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.self_attn.q_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.k_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.v_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                f"model.layers.{layer_idx}.self_attn.q_proj.bias",
                f"model.layers.{layer_idx}.self_attn.k_proj.bias",
                f"model.layers.{layer_idx}.self_attn.v_proj.bias",
            ]
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.self_attn.q_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.k_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.v_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            ]
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            return [
                f"model.layers.{layer_idx}.self_attn.q_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.k_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.v_proj.weight", 
                f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                f"model.layers.{layer_idx}.self_attn.q_norm.weight",
                f"model.layers.{layer_idx}.self_attn.k_norm.weight",
            ]
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            p = self._gemma4_weight_prefix()
            return [
                f"{p}.layers.{layer_idx}.self_attn.q_proj.weight",
                f"{p}.layers.{layer_idx}.self_attn.k_proj.weight",
                f"{p}.layers.{layer_idx}.self_attn.v_proj.weight",
                f"{p}.layers.{layer_idx}.self_attn.o_proj.weight",
                f"{p}.layers.{layer_idx}.self_attn.q_norm.weight",
                f"{p}.layers.{layer_idx}.self_attn.k_norm.weight",
            ]
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            layer_types = getattr(self.config, "layer_types", None)
            if layer_types is None or layer_types[layer_idx] != "full_attention":
                return []
            p = f"model.layers.{layer_idx}.self_attn"
            names = [
                f"{p}.q_proj.weight",
                f"{p}.k_proj.weight",
                f"{p}.v_proj.weight",
                f"{p}.o_proj.weight",
                f"{p}.q_norm.weight",
                f"{p}.k_norm.weight",
            ]
            if getattr(self.config, "attention_bias", False):
                names.extend(
                    [
                        f"{p}.q_proj.bias",
                        f"{p}.k_proj.bias",
                        f"{p}.v_proj.bias",
                        f"{p}.o_proj.bias",
                    ]
                )
            return names
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            p = f"model.layers.{layer_idx}.self_attn"
            names = [
                f"{p}.q_proj.weight",
                f"{p}.k_proj.weight",
                f"{p}.v_proj.weight",
                f"{p}.o_proj.weight",
                f"{p}.sinks",
            ]
            if getattr(self.config, "attention_bias", False):
                names.extend(
                    [
                        f"{p}.q_proj.bias",
                        f"{p}.k_proj.bias",
                        f"{p}.v_proj.bias",
                        f"{p}.o_proj.bias",
                    ]
                )
            return names
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            p = f"model.layers.{layer_idx}.self_attn"
            names = [
                f"{p}.q_proj.weight",
                f"{p}.k_proj.weight",
                f"{p}.v_proj.weight",
                f"{p}.o_proj.weight",
            ]
            if getattr(self.config, "use_bias", False):
                names.extend(
                    [
                        f"{p}.q_proj.bias",
                        f"{p}.k_proj.bias",
                        f"{p}.v_proj.bias",
                        f"{p}.o_proj.bias",
                    ]
                )
            return names
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_mlp_names(self, layer_idx: int):
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            p = self._gemma4_weight_prefix()
            return [
                f"{p}.layers.{layer_idx}.mlp.down_proj.weight",
                f"{p}.layers.{layer_idx}.mlp.up_proj.weight",
                f"{p}.layers.{layer_idx}.mlp.gate_up_proj.weight",
            ]
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def gate_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        """
        MoE 路由：根据 ``hidden_states`` 计算本层 top-k expert 索引与路由权重。

        返回格式与 Deepseek gate 对齐：``(topk_idx, topk_weight, aux_loss)``，其中 ``aux_loss`` 推理多为 ``None``。
        """
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            topk_idx, topk_weight, aux_loss = mi.model.layers[layer_idx].mlp.gate(hidden_states)
            return topk_idx, topk_weight, aux_loss
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            router_logits = mi.model.layers[layer_idx].mlp.gate(hidden_states)
            
            # Compute softmax scores
            scores = F.softmax(router_logits, dim=-1)
            
            # Get top-k experts
            top_k = mi.model.layers[layer_idx].mlp.top_k
            topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
            
            # Normalize weights if needed
            if mi.model.layers[layer_idx].mlp.norm_topk_prob and top_k > 1:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator
            
            # Return in (batch*seq, top_k) format to match Deepseek gate output
            # Deepseek gate returns (bsz*seq_len, top_k), not (bsz, seq_len, top_k)
            # Aux loss is typically None in inference mode
            aux_loss = None
            
            return topk_idx, topk_weight, aux_loss
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            router_logits = mi.model.layers[layer_idx].block_sparse_moe.gate(hidden_states)
             # Compute softmax scores
            scores = F.softmax(router_logits, dim=-1)
            
            # Get top-k experts
            top_k = mi.model.layers[layer_idx].block_sparse_moe.top_k
            topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
            
            
            # Return in (batch*seq, top_k) format to match Deepseek gate output
            # Deepseek gate returns (bsz*seq_len, top_k), not (bsz, seq_len, top_k)
            # Aux loss is typically None in inference mode
            aux_loss = None
            
            return topk_idx, topk_weight, aux_loss
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            router_logits = mi.model.layers[layer_idx].mlp.gate(hidden_states)
            scores = F.softmax(router_logits, dim=-1)
            top_k = mi.model.layers[layer_idx].mlp.top_k
            topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
            if mi.model.layers[layer_idx].mlp.norm_topk_prob and top_k > 1:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator
            aux_loss = None
            return topk_idx, topk_weight, aux_loss
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            layer = lm.layers[layer_idx]
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            _, top_k_weights, top_k_index = layer.router(flat)
            aux_loss = None
            return top_k_index, top_k_weights, aux_loss
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            _, topk_weight, topk_idx = mi.model.layers[layer_idx].mlp.gate(flat)
            aux_loss = None
            return topk_idx, topk_weight, aux_loss
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            _, topk_weight, topk_idx = mi.model.layers[layer_idx].mlp.router(flat)
            aux_loss = None
            return topk_idx, topk_weight, aux_loss
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            if not self._ernie_layer_is_sparse_moe(layer_idx):
                raise RuntimeError(
                    "gate_func is only valid on Ernie MoE layers; this layer uses dense MLP."
                )
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            _, topk_idx, topk_weight = mi.model.layers[layer_idx].mlp.gate(flat)
            aux_loss = None
            return topk_idx, topk_weight, aux_loss
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def shared_experts_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        """前向调用该层 shared expert（若存在）；无则原样返回 ``hidden_states``。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            y = mi.model.layers[layer_idx].mlp.shared_experts(hidden_states)
            return y
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            y = mi.model.layers[layer_idx].mlp.shared_expert(hidden_states)
            y = F.sigmoid(mi.model.layers[layer_idx].mlp.shared_expert_gate(hidden_states)) * y
            return y
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            return hidden_states
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            # no shared experts in Mixtral
            return hidden_states
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            # no shared experts in Gemma4
            return hidden_states
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            y = mi.model.layers[layer_idx].mlp.shared_expert(flat)
            y = F.sigmoid(mi.model.layers[layer_idx].mlp.shared_expert_gate(flat)) * y
            return y.view_as(hidden_states)
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            return hidden_states
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            mlp = mi.model.layers[layer_idx].mlp
            if not self._ernie_layer_is_sparse_moe(layer_idx) or getattr(mlp, "shared_experts", None) is None:
                return hidden_states
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            y = mlp.shared_experts(flat)
            return y.view_as(hidden_states)
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def self_attn_func(self, mi, layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Cache,
    ):
        """执行该层 self-attention，返回更新后的 ``hidden_states``（``use_cache=True``）。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            hidden_states, _, _ = mi.model.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=True,
            )
            return hidden_states
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            hidden_states, _, _ = mi.model.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=True,
            )
            return hidden_states
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            hidden_states, _, _ = mi.model.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=True,
            )
            return hidden_states
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            position_embeddings = mi.model.rotary_emb(hidden_states, position_ids)
            hidden_states, _ = mi.model.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=True,
            )
            return hidden_states
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            position_embeddings = lm.rotary_emb(hidden_states, position_ids)
            hidden_states, _ = lm.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                output_attentions=False,
                use_cache=True,
            )
            return hidden_states
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            layer = mi.model.layers[layer_idx]
            if layer.layer_type == "linear_attention":
                hidden_states = layer.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=past_key_value,
                    attention_mask=attention_mask,
                )
            else:
                position_embeddings = mi.model.rotary_emb(hidden_states, position_ids)
                hidden_states, _ = layer.self_attn(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_value,
                )
            return hidden_states
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            position_embeddings = mi.model.rotary_emb(hidden_states, position_ids)
            hidden_states, _ = mi.model.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                use_cache=True,
            )
            return hidden_states
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            position_embeddings = mi.model.rotary_emb(hidden_states, position_ids)
            hidden_states, _ = mi.model.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                use_cache=True,
            )
            return hidden_states
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def iln_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        """Input LayerNorm（注意力前）。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            # 类型注解：mi 是 DeepseekModule 实例
            # mi: "DeepseekForCausalLM" = mi
            logger.debug(f"{mi.model.layers[layer_idx].input_layernorm.weight.data.device} {hidden_states.device}")
            hidden_states = mi.model.layers[layer_idx].input_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].input_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].input_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].input_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            hidden_states = lm.layers[layer_idx].input_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].input_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].input_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].input_layernorm(hidden_states)
            return hidden_states
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def paln_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        """Post-attention LayerNorm（FFN / MoE 前）。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].post_attention_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].post_attention_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].post_attention_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].post_attention_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            hidden_states = lm.layers[layer_idx].post_attention_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].post_attention_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].post_attention_layernorm(hidden_states)
            return hidden_states
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].post_attention_layernorm(hidden_states)
            return hidden_states
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def dense_mlp_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        """Dense MLP 或等价 FFN（含 Gemma4 的 layernorm + post_ffn 分支）。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].mlp(hidden_states)
            return hidden_states
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].mlp(hidden_states)
            return hidden_states
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].mlp(hidden_states)
            return hidden_states
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].mlp(hidden_states)
            return hidden_states
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            layer = lm.layers[layer_idx]
            hs = layer.pre_feedforward_layernorm(hidden_states)
            out = layer.mlp(hs)
            # Gemma4 moe block 存在时，dense 分支先经过 post_feedforward_layernorm_1
            if getattr(layer, "enable_moe_block", False) and hasattr(layer, "post_feedforward_layernorm_1"):
                out = layer.post_feedforward_layernorm_1(out)
            else:
                out = layer.post_feedforward_layernorm(out)
            return out
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].mlp(hidden_states)
            return hidden_states
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            hidden_states, _ = mi.model.layers[layer_idx].mlp(hidden_states)
            return hidden_states
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            hidden_states = mi.model.layers[layer_idx].mlp(hidden_states)
            return hidden_states
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    @torch.no_grad()
    def gemma4_moe_only(self, mi, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gemma4 专用：返回 MoE block 的输出（不含 dense MLP），形状与 hidden_states 一致。"""
        if self.model_name_type != GEMMA4_MODEL_NAME_TYPE:
            raise ValueError("gemma4_moe_only called for non-Gemma4 model")
        lm = getattr(mi.model, "language_model", mi.model)
        layer = lm.layers[layer_idx]
        if not getattr(layer, "enable_moe_block", False):
            return torch.zeros_like(hidden_states)
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        _, top_k_weights, top_k_index = layer.router(flat)
        hs2 = layer.pre_feedforward_layernorm_2(flat)
        moe = layer.experts(hs2, top_k_index, top_k_weights).view_as(flat)
        moe = moe.view(*hidden_states.shape)
        moe = layer.post_feedforward_layernorm_2(moe)
        return moe
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
        逐 expert 调用各模型 ``Module.experts_func``，在 ``final_hidden_states`` 上做 scatter 聚合。

        与批量 ``einsum`` 路径不同：此处按 expert 循环，便于 CPU 或小批量场景。

        Args:
            mi: 已加载权重的模型实例。
            layer_idx: Transformer 层索引。
            expert_idx_list: 本步参与的 expert id 列表。
            expert_indices_map: 每个 expert 在 ``idxs`` 展平序列上的 ``[start, end)``。
            expert_token_indices_map: 每个 expert 负责的 token 下标（展平 batch×seq）。
            flat_hidden_states: 展平后的隐状态 ``[N, H]``。
            flat_experts_weight: 与路由对齐的标量权重 ``[N, 1]`` 等。
            idxs: 与 ``flat_experts_weight`` 对齐的排序索引。
            final_hidden_states: 累加输出的缓冲区，原地更新。
            device: ``'cuda'`` / ``'cpu'`` 等，控制 token 张量计算设备。

        Returns:
            更新后的 ``final_hidden_states``。
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
                if device == "cpu":
                    tokens_on_device = tokens.to("cpu")
                    weights_on_device = weights.to("cpu")
                    MixtralModule.experts_func(
                        mi, layer_idx, expert_id, tokens_on_device, weights_on_device,
                        token_indices,
                        final_hidden_states=final_hidden_states
                    )
                else:
                    MixtralModule.experts_func(
                        mi, layer_idx, expert_id, tokens, weights,
                        token_indices,
                        final_hidden_states=final_hidden_states
                    )
            elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
                if device == "cpu":
                    tokens_on_device = tokens.to("cpu")
                    weights_on_device = weights.to("cpu")
                    Qwen2MoEModule.experts_func(
                        mi, layer_idx, expert_id, tokens_on_device, weights_on_device,
                        token_indices,
                        final_hidden_states=final_hidden_states
                    )
                else:
                    Qwen2MoEModule.experts_func(
                        mi, layer_idx, expert_id, tokens, weights,
                        token_indices,
                        final_hidden_states=final_hidden_states
                    )
            elif self.model_name_type in (
                QWEN3_MODEL_NAME_TYPE,
                QWEN3_5_MODEL_NAME_TYPE,
                GPT_OSS_MODEL_NAME_TYPE,
            ):
                expert_mlp = mi.model.layers[layer_idx].mlp.experts.experts[expert_id]
                if device == "cpu":
                    tokens_on_device = tokens.to("cpu")
                    weights_on_device = weights.to("cpu")
                    _scatter_fused_moe_expert_output(
                        expert_mlp,
                        tokens_on_device,
                        weights_on_device,
                        token_indices,
                        final_hidden_states,
                    )
                else:
                    _scatter_fused_moe_expert_output(
                        expert_mlp,
                        tokens,
                        weights,
                        token_indices,
                        final_hidden_states,
                    )
            elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
                if not self._ernie_layer_is_sparse_moe(layer_idx):
                    continue
                expert_mlp = mi.model.layers[layer_idx].mlp.experts.experts[expert_id]
                if device == "cpu":
                    tokens_on_device = tokens.to("cpu")
                    weights_on_device = weights.to("cpu")
                    _scatter_fused_moe_expert_output(
                        expert_mlp,
                        tokens_on_device,
                        weights_on_device,
                        token_indices,
                        final_hidden_states,
                    )
                else:
                    _scatter_fused_moe_expert_output(
                        expert_mlp,
                        tokens,
                        weights,
                        token_indices,
                        final_hidden_states,
                    )
            elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
                if device == "cpu":
                    tokens_on_device = tokens.to("cpu")
                    weights_on_device = weights.to("cpu")
                    Gemma4Module.experts_func(
                        mi,
                        layer_idx,
                        expert_id,
                        tokens_on_device,
                        weights_on_device,
                        token_indices,
                        final_hidden_states=final_hidden_states,
                    )
                else:
                    Gemma4Module.experts_func(
                        mi,
                        layer_idx,
                        expert_id,
                        tokens,
                        weights,
                        token_indices,
                        final_hidden_states=final_hidden_states,
                    )
            else:
                raise ValueError(f"Invalid model name type: {self.model_name_type}")
        
        return final_hidden_states

    def get_experts_per_tok(self):
        """每个 token 路由到的 expert 数量（top-k 的 k，来自 config）。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            return self.config.num_experts_per_tok
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            return self.config.num_experts_per_tok
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            return self.config.num_experts_per_tok
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            return self.config.num_experts_per_tok
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            return self.config.num_experts_per_tok
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            return self.config.num_experts_per_tok
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            return self.config.moe_k
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def get_first_k_dense_replace(self):
        """Deepseek 前几层用 dense MLP 替代 MoE 的层数阈值；其它模型返回 0。"""
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            return self.config.first_k_dense_replace
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            return 0
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            return 0
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            return 0
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            return 0
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            return 0
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            return 0
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            return 0
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")

    def experts_func_mgpu_group_pad(
        self,
        expert_idx_list: list[int],
        expert_indices_map: Dict[int, Tuple[int, int]],
        device_flat_hidden_states: torch.Tensor,
        device_idxs: torch.Tensor,
    ):
        """
        在单设备上为 expert 列表构造 ``stacked_inputs``：形状 ``[E, max_tokens, H]``，有效段 ``copy_``，其余为 0。

        供多 GPU einsum 路径复用，与 ``experts_func_mgpu_group_list`` 输出的权重批维对齐。
        """
        device_token_idxs = device_idxs // self.get_experts_per_tok()

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
        return stacked_inputs
    def experts_func_mgpu_group_list(
        self,
        mi,
        layer_idx: int,
        expert_idx_list: list[int],
    ):
        """
        从 GPU 模型收集若干 expert 的 w1/w2/w3 权重引用列表（不堆叠），供后续 ``_pack_*`` 或测试使用。

        各元素形状：w1/w3 为 ``[I, H]``，w2 为 ``[H, I]``（命名随模型：gate/up/down 或 w1/w3/w2）。
        """
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
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            for expert_idx in expert_idx_list:
                expert_module = mi.model.layers[layer_idx].mlp.experts[expert_idx]
                group_w1_list.append(expert_module.gate_proj.weight)  # [I, H]
                group_w2_list.append(expert_module.down_proj.weight)   # [H, I]
                group_w3_list.append(expert_module.up_proj.weight)     # [I, H]
        elif self.model_name_type in (
            QWEN3_MODEL_NAME_TYPE,
            QWEN3_5_MODEL_NAME_TYPE,
            GPT_OSS_MODEL_NAME_TYPE,
            ERINE_MODEL_NAME_TYPE,
        ):
            if self.model_name_type == ERINE_MODEL_NAME_TYPE and not self._ernie_layer_is_sparse_moe(layer_idx):
                raise ValueError("experts_func_mgpu_group_list called on Ernie dense MLP layer")
            for expert_idx in expert_idx_list:
                expert_module = mi.model.layers[layer_idx].mlp.experts.experts[expert_idx]
                w1, w2, w3 = _fused_expert_gate_up_w123(expert_module)
                group_w1_list.append(w1)
                group_w2_list.append(w2)
                group_w3_list.append(w3)
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            layer = lm.layers[layer_idx]
            for expert_idx in expert_idx_list:
                expert_module = getattr(layer.experts, str(int(expert_idx)))
                w1, w2, w3 = _fused_expert_gate_up_w123(expert_module)
                group_w1_list.append(w1)
                group_w2_list.append(w2)
                group_w3_list.append(w3)
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
        # 堆叠 weights: [E, I, H], [E, H, I], [E, I, H]
        # group_w1 = torch.stack(group_w1_list)  # [E, I, H]
        # group_w2 = torch.stack(group_w2_list)  # [E, H, I]
        # group_w3 = torch.stack(group_w3_list)  # [E, I, H]
        return group_w1_list, group_w2_list, group_w3_list

    @torch.no_grad()
    def experts_func_mgpu_group_experts(
        self,
        group_w1_list,
        group_w2_list,
        group_w3_list,
    ):
        """
        将 ``experts_func_mgpu_group_list`` 得到的列表打包为批张量。

        使用 ``_pack_group_w1_w3_no_stack`` 与 ``_pack_group_w2_no_stack``；返回的 w1/w3 为 ``group_w1_w3`` 的视图切片。
        """
        cuda_hook_time("gpu_group_tensor")
        # 堆叠 weights: [E, I, H], [E, H, I], [E, I, H] — 预分配 + copy_，避免 torch.stack
        group_w1_w3 = self._pack_group_w1_w3_no_stack(group_w1_list, group_w3_list)
        group_w2 = self._pack_group_w2_no_stack(group_w2_list)
        I_gate = group_w1_w3.shape[1] // 2
        group_w1 = group_w1_w3[:, :I_gate, :]
        group_w3 = group_w1_w3[:, I_gate:, :]
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
        """
        多卡多流：按 ``device_id`` 打包权重、对 ``stacked_inputs_map`` 做融合 gate+up einsum 与 down einsum，
        再按设备将 expert 输出 scatter 回 ``final_hidden_states``。

        ``*_map`` 的 key 均为设备 id；可选 ``all_expert_weights_map`` 等用于批量 scatter 优化。
        """
        group_w1_w3_map: Dict[int, torch.Tensor] = {}
        group_w2_map: Dict[int, torch.Tensor] = {}
        
        for device_id in group_w1_list_map.keys():
            cuda_hook_time("gpu_group_tensor")
            group_w1_list = group_w1_list_map[device_id]
            group_w2_list = group_w2_list_map[device_id]
            group_w3_list = group_w3_list_map[device_id]

             # 堆叠 weights: [E, 2I, H], [E, H, I] — 预分配 + copy_，避免 torch.stack
            group_w1_w3_map[device_id] = self._pack_group_w1_w3_no_stack(group_w1_list, group_w3_list)
            group_w2_map[device_id] = self._pack_group_w2_no_stack(group_w2_list)
            cuda_hook_time_end("gpu_group_tensor")

        outputs_map = {}
        for device_id in group_w1_list_map.keys():
            group_w1_w3 = group_w1_w3_map[device_id]
            group_w2 = group_w2_map[device_id]

            stacked_inputs = stacked_inputs_map[device_id]
            expert_idx_list = expert_idx_list_map[device_id]
            device_flat_hidden_states = flat_hidden_states_map[device_id]
            device_flat_experts_weight = flat_experts_weight_map[device_id]
            device_idxs = idxs_map[device_id]
            device_token_idxs = device_idxs // self.get_experts_per_tok()

            # cuda_hook_time("gpu_group_tensor")
            # # 堆叠 weights: [E, I, H], [E, H, I], [E, I, H]
            # group_w1 = torch.stack(group_w1_list)  # [E, I, H]
            # group_w2 = torch.stack(group_w2_list)  # [E, H, I]
            # group_w3 = torch.stack(group_w3_list)  # [E, I, H]
            # cuda_hook_time_end("gpu_group_tensor")

            cuda_hook_time("gpu_group_einsum")
            intermediate = self._moe_gate_up_from_stacked(stacked_inputs, group_w1_w3)
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
            device_token_idxs = device_idxs // self.get_experts_per_tok()

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
        单设备上对已 padding 的 ``stacked_inputs`` 与批权重做 MoE 前向（融合 gate+up 一次 einsum），再 scatter。

        Args:
            layer_idx: 层索引（部分日志/钩子使用）。
            group_w1, group_w2, group_w3: 形状分别为 ``[E,I,H]``、``[E,H,I]``、``[E,I,H]``；内部会先拼为 ``group_w1_w3``。
            stacked_inputs: ``[E, max_tokens, H]``，与 ``group_*`` 同设备。
            expert_idx_list / expert_indices_map: expert 与 ``idxs`` 区间的对应关系。
            flat_hidden_states / flat_experts_weight / idxs: 用于 scatter 阶段取 token 与路由权重。
            final_hidden_states: 输出累加缓冲区。

        Returns:
            更新后的 ``final_hidden_states``。
        """
        if not expert_idx_list:
            return final_hidden_states
        # cuda_hook_time("gpu_group_pad")
        # if group_w1.device != flat_hidden_states.device:
        #     device_flat_hidden_states = flat_hidden_states.to(group_w1.device, non_blocking=True)
        #     device_idxs = idxs.to(group_w1.device, non_blocking=True)
        #     device_token_idxs = device_idxs // self.get_experts_per_tok()
        #     device_flat_experts_weight = flat_experts_weight.to(group_w1.device, non_blocking=True)
        # else:
        #     device_flat_hidden_states = flat_hidden_states
        #     device_idxs = idxs
        #     device_token_idxs = device_idxs // self.get_experts_per_tok()
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
        device_token_idxs = device_idxs // self.get_experts_per_tok()
        device_flat_experts_weight = flat_experts_weight

        cuda_hook_time("gpu_group_einsum")
        group_w1_w3 = self._pack_batched_w1_w3(group_w1, group_w3)
        intermediate = self._moe_gate_up_from_stacked(stacked_inputs, group_w1_w3)
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
        多 GPU：每卡 CUDA stream 上收集专家权重（``w1_w3``/``w2``）、padding 输入、融合 einsum MoE，
        再按 expert 将结果 scatter 回 ``final_hidden_states``。

        Args:
            mi: 已加载到多卡的模型实例。
            layer_idx: 层索引。
            expert_idx_list_map: ``{device_id: [expert_id, ...]}``。
            expert_indices_map / expert_token_indices_map: 与单卡路径相同的索引语义。
            flat_hidden_states / flat_experts_weight / idxs: 主设备展平缓冲与索引。
            final_hidden_states: 聚合输出。
            streams: 每设备一条 ``torch.cuda.Stream``，与 ``expert_idx_list_map`` 的 key 对齐。

        Returns:
            更新后的 ``final_hidden_states``。
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
                    elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
                        for expert_idx in expert_indices:
                            expert_module = mi.model.layers[layer_idx].mlp.experts[expert_idx]
                            group_w1_list.append(expert_module.gate_proj.weight)
                            group_w2_list.append(expert_module.down_proj.weight)
                            group_w3_list.append(expert_module.up_proj.weight)
                    elif self.model_name_type in (
                        QWEN3_MODEL_NAME_TYPE,
                        QWEN3_5_MODEL_NAME_TYPE,
                        GPT_OSS_MODEL_NAME_TYPE,
                        ERINE_MODEL_NAME_TYPE,
                    ):
                        if self.model_name_type == ERINE_MODEL_NAME_TYPE and not self._ernie_layer_is_sparse_moe(
                            layer_idx
                        ):
                            raise ValueError(
                                "experts_func_mgpu_einsum called on Ernie dense MLP layer"
                            )
                        for expert_idx in expert_indices:
                            expert_module = mi.model.layers[layer_idx].mlp.experts.experts[expert_idx]
                            w1, w2, w3 = _fused_expert_gate_up_w123(expert_module)
                            group_w1_list.append(w1)
                            group_w2_list.append(w2)
                            group_w3_list.append(w3)
                    elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
                        lm = getattr(mi.model, "language_model", mi.model)
                        layer = lm.layers[layer_idx]
                        for expert_idx in expert_indices:
                            expert_module = getattr(layer.experts, str(int(expert_idx)))
                            w1, w2, w3 = _fused_expert_gate_up_w123(expert_module)
                            group_w1_list.append(w1)
                            group_w2_list.append(w2)
                            group_w3_list.append(w3)
                    else:
                        raise ValueError(f"Invalid model name type: {self.model_name_type}")
                    
                    device_group_tensors[device_id] = {
                        'w1_w3': self._pack_group_w1_w3_no_stack(group_w1_list, group_w3_list),
                        'w2': self._pack_group_w2_no_stack(group_w2_list),
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
                    group_w1_w3 = device_group_tensors[device_id]['w1_w3']
                    
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
                    device_token_idxs = device_idxs // self.get_experts_per_tok()
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
                    group_w1_w3 = device_group_tensors[device_id]['w1_w3']
                    group_w2 = device_group_tensors[device_id]['w2']
                    
                    # 并行执行 einsum 计算
                    intermediate = self._moe_gate_up_from_stacked(stacked_inputs, group_w1_w3)
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
                device_idxs = idxs.to(outputs.device, non_blocking=True)
                device_token_idxs = device_idxs // self.get_experts_per_tok()
            else:
                expert_cache = final_hidden_states
                device_flat_experts_weight = flat_experts_weight
                device_idxs = idxs
                device_token_idxs = device_idxs // self.get_experts_per_tok()
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
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            for expert_idx in expert_indices:
                expert_module = mi.model.layers[layer_idx].mlp.experts[expert_idx]
                group_w1_list.append(expert_module.gate_proj.weight)
                group_w2_list.append(expert_module.down_proj.weight)
                group_w3_list.append(expert_module.up_proj.weight)
        elif self.model_name_type in (
            QWEN3_MODEL_NAME_TYPE,
            QWEN3_5_MODEL_NAME_TYPE,
            GPT_OSS_MODEL_NAME_TYPE,
            ERINE_MODEL_NAME_TYPE,
        ):
            if self.model_name_type == ERINE_MODEL_NAME_TYPE and not self._ernie_layer_is_sparse_moe(layer_idx):
                raise ValueError("experts_func_gpu_einsum called on Ernie dense MLP layer")
            for expert_idx in expert_indices:
                expert_module = mi.model.layers[layer_idx].mlp.experts.experts[expert_idx]
                w1, w2, w3 = _fused_expert_gate_up_w123(expert_module)
                group_w1_list.append(w1)
                group_w2_list.append(w2)
                group_w3_list.append(w3)
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            layer = lm.layers[layer_idx]
            for expert_idx in expert_indices:
                expert_module = getattr(layer.experts, str(int(expert_idx)))
                w1, w2, w3 = _fused_expert_gate_up_w123(expert_module)
                group_w1_list.append(w1)
                group_w2_list.append(w2)
                group_w3_list.append(w3)
        else:
            raise ValueError(f"Invalid model name type: {self.model_name_type}")
        
        # 堆叠 weights: [E, 2I, H], [E, H, I] — 预分配 + copy_，避免 torch.stack
        group_w1_w3 = self._pack_group_w1_w3_no_stack(group_w1_list, group_w3_list)
        group_w2 = self._pack_group_w2_no_stack(group_w2_list)
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
            dtype=flat_hidden_states.dtype, device=group_w2.device
        )
        
        if group_w2.device != flat_hidden_states.device:
            device_flat_hidden_states = flat_hidden_states.to(group_w2.device, non_blocking=True)
            device_idxs = idxs.to(group_w2.device, non_blocking=True)
            device_token_idxs = device_idxs // self.get_experts_per_tok()
        else:
            device_flat_hidden_states = flat_hidden_states
            device_idxs = idxs
            device_token_idxs = device_idxs // self.get_experts_per_tok()

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
        logger.debug(f"start_gate_up_fused")
        intermediate = self._moe_gate_up_from_stacked(stacked_inputs, group_w1_w3)
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
        委托 ``einsum_with_group_tensors``：在 CPU 上用共享内存恢复的专家组权重做批量 MoE（含 scatter）。

        适用于主进程内同步 einsum 路径；结果写回 ``final_hidden_states`` 并向 ``output_queue`` 投递标记。
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
        子进程 / MP 用入口：当前实现委托 ``bmm_with_group_tensors_mp``（CPU 上融合 bmm gate+up，再回传 GPU）。

        返回 ``(output_gpu, group_list)`` 供调用方持有组张量引用以延迟 unmap；无 expert 时返回 ``(None, [])``。
        """
        if not expert_idx_list:
            return None, []
        # 调用 einsum_with_group_tensors 函数（在 CPU 上计算）
        # 直接传递索引信息，避免创建中间 tensor maps
        # final_hidden_states, group_list = self.einsum_with_group_tensors_mp(
        #     hmv=hmv,
        #     layer_idx=layer_idx,
        #     expert_idx_list=expert_idx_list,
        #     expert_indices_map=expert_indices_map,
        #     flat_hidden_states=flat_hidden_states,
        #     idxs=idxs,
        #     output_queue=output_queue
        # )
        final_hidden_states, group_list = self.bmm_with_group_tensors_mp(
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
    def bmm_with_group_tensors_mp(
        self, 
        hmv: "HostMemoryView",
        layer_idx: int, expert_idx_list: list[int], 
        expert_indices_map: Dict[int, Tuple[int, int]],  # {expert_id: (start_idx, end_idx)}
        flat_hidden_states: torch.Tensor,  # 原始展平的 hidden states
        idxs: torch.Tensor,
        output_queue
    ):
        """
        从 ``hmv.group_experts_tensor`` 取批专家权重，在 CPU pinned 缓冲上做融合 gate+up ``bmm`` 与 down ``bmm``，
        将拼接后的 token 输出拷回 GPU。

        ``group_list`` 顺序：``[w1, w2, w3, w1_w3^T, w2^T, w3^T]``（末项保留以兼容旧调用方槽位）。

        Args:
            hmv: HostMemoryView。
            layer_idx / expert_idx_list / expert_indices_map: 层与 expert 及 idx 区间。
            flat_hidden_states: GPU 展平隐状态（先拷到 CPU pin）。
            idxs: 路由索引（用于构造 token 映射）。
            output_queue: 写入 ``ExpertEinsumResult``。

        Returns:
            ``(output_gpu, group_list)``。
        """
        cuda_hook_time("move_flatidxs")
        flat_hidden_states_cpu_pin = gpinpool.alloc_same_pin_tensor(flat_hidden_states)
        flat_hidden_states_cpu_pin.copy_(flat_hidden_states, non_blocking=False)
        token_idxs = idxs // self.get_experts_per_tok()
        token_idxs_cpu_pin = token_idxs.cpu()
        cuda_hook_time_end("move_flatidxs")

        cuda_hook_time("group_tensors")
        # 获取 group tensors (已经是堆叠好的 [E, ...] 形状)
        group_dict = hmv.group_experts_tensor(layer_idx, expert_idx_list)
        try:
            group_w1, group_w2, group_w3 = self._coerce_expert_group_w123(group_dict)
        except Exception as e:
            logger.error(f"group_dict: {group_dict}")
            logger.error(f"Error getting group tensors: {e}")
            raise e
        group_w1_w3 = (
            group_dict["group_w1_w3"]
            if "group_w1_w3" in group_dict
            else self._pack_batched_w1_w3(group_w1, group_w3)
        )
        
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

        cuda_hook_time("group_bmm")
        # stacked_inputs: [E, max_tokens, H]；一次 bmm 完成 gate+up，再 bmm down
        intermediate = self._moe_gate_up_bmm_from_stacked(stacked_inputs, group_w1_w3)
        group_w2_transposed = group_w2.transpose(1, 2)  # [E, I, H]
        outputs_result = torch.bmm(intermediate, group_w2_transposed)  # [E, max_tokens, H]
        cuda_hook_time_end("group_bmm")
        

        cuda_hook_time("get_outputs_cpu1")
        ce_out_list = []
        for i, expert_idx in enumerate(expert_idx_list):
            token_ids = expert_token_indices_map[expert_idx]
            num_tokens = token_ids.shape[0]
            expert_out = outputs_result[i][:num_tokens]
            ce_out_list.append(expert_out)
        concat_ce_out = torch.cat(ce_out_list, dim=0)
        outputs_result_cpu_pin = gpinpool.alloc_same_pin_tensor(concat_ce_out)
        outputs_result_cpu_pin.copy_(concat_ce_out, non_blocking=False)
        output_gpu = outputs_result_cpu_pin.to(flat_hidden_states.device, non_blocking=False)
        cuda_hook_time_end("get_outputs_cpu1")

        # cuda_hook_time("get_outputs_cpu2")
        # outputs_result_cpu_pin = gpinpool.alloc_same_pin_tensor(outputs_result)
        # outputs_result_cpu_pin.copy_(outputs_result, non_blocking=False)
        # output_gpu = outputs_result_cpu_pin.to(flat_hidden_states.device, non_blocking=False)
        # cuda_hook_time_end("get_outputs_cpu2")

        result = ExpertEinsumResult(final_hidden_states=output_gpu, time_einsum_end=time.time())
        output_queue.put(result)
        gpinpool.free(flat_hidden_states_cpu_pin)
        gpinpool.free(token_idxs_cpu_pin)
        gpinpool.free(outputs_result_cpu_pin)
        group_w1_w3_transposed = group_w1_w3.transpose(1, 2)
        group_w3_transposed = group_w3.transpose(1, 2)
        group_list = [
            group_w1,
            group_w2,
            group_w3,
            group_w1_w3_transposed,
            group_w2_transposed,
            group_w3_transposed,
        ]
        return output_gpu, group_list

    @torch.no_grad()
    def einsum_with_group_tensors(
        self, 
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
        主进程同步路径：从 Host 共享内存拉取 ``[E,*]`` 批专家权重，GPU 上 padding 输入后做融合 gate+up einsum 与 down einsum，
        将结果搬回 CPU pin 再 scatter 到 ``final_hidden_states``，并向 ``output_queue`` 放入结果对象。

        中间含两段 ``stacked_inputs`` 构造（历史路径：先 CPU pin 预取再 GPU 主算），修改时需保持索引与 expert 过滤一致。
        """
        time_start_group = time.time()
        cuda_hook("group tensor")
        # 获取 group tensors (已经是堆叠好的 [E, ...] 形状)
        group_dict = hmv.group_experts_tensor(layer_idx, expert_idx_list)
        group_w1, group_w2, group_w3 = self._coerce_expert_group_w123(group_dict)
        group_w1_w3 = (
            group_dict["group_w1_w3"]
            if "group_w1_w3" in group_dict
            else self._pack_batched_w1_w3(group_w1, group_w3)
        )
        
        cuda_hook_end("group tensor")
        logger.debug(f"group tensors cost {time.time() - time_start_group} s")
        
        cuda_hook_time("move_flat_hidden2cpu")
        flat_hidden_states_cpu_pin = gpinpool.alloc_same_pin_tensor(flat_hidden_states)
        flat_hidden_states_cpu_pin.copy_(flat_hidden_states, non_blocking=False)

        token_idxs = idxs // self.get_experts_per_tok()
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
        # 一次 einsum 完成 gate+up，再 einsum down
        intermediate = self._moe_gate_up_from_stacked(stacked_inputs, group_w1_w3)
        
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
        del group_w1, group_w2, group_w3, group_w1_w3
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
        MP 变体：输入先拷到 CPU pin，在 CPU 上 padding ``stacked_inputs`` 后做融合 gate+up einsum 与 down einsum，
        将有效 expert 行拼成连续张量再拷回 ``flat_hidden_states.device``，结果放入 ``output_queue``。

        与 ``einsum_with_group_tensors`` 相比不做 ``final_hidden_states`` 上的 scatter，适合子进程只负责算子回传。
        """
        cuda_hook_time("move_flatidxs")
        flat_hidden_states_cpu_pin = gpinpool.alloc_same_pin_tensor(flat_hidden_states)
        flat_hidden_states_cpu_pin.copy_(flat_hidden_states, non_blocking=False)
        token_idxs = idxs // self.get_experts_per_tok()
        token_idxs_cpu_pin = token_idxs.cpu()
        cuda_hook_time_end("move_flatidxs")

        cuda_hook_time("group_tensors")
        # 获取 group tensors (已经是堆叠好的 [E, ...] 形状)
        group_dict = hmv.group_experts_tensor(layer_idx, expert_idx_list)
        group_w1, group_w2, group_w3 = self._coerce_expert_group_w123(group_dict)
        group_w1_w3 = (
            group_dict["group_w1_w3"]
            if "group_w1_w3" in group_dict
            else self._pack_batched_w1_w3(group_w1, group_w3)
        )
        
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
        intermediate = self._moe_gate_up_from_stacked(stacked_inputs, group_w1_w3)
        # outputs: [E, max_tokens, H] - 使用预分配的 tensor（先计算再复制）
        # 先计算结果，然后复制到预分配的 tensor（因为某些 PyTorch 版本不支持 einsum 的 out 参数）
        outputs_result = torch.einsum('eti,ehi->eth', intermediate, group_w2)
        cuda_hook_time_end("group_einsum")
        

        cuda_hook_time("get_outputs_cpu1")
        ce_out_list = []
        for i, expert_idx in enumerate(expert_idx_list):
            token_ids = expert_token_indices_map[expert_idx]
            num_tokens = token_ids.shape[0]
            expert_out = outputs_result[i][:num_tokens]
            ce_out_list.append(expert_out)
        concat_ce_out = torch.cat(ce_out_list, dim=0)
        outputs_result_cpu_pin = gpinpool.alloc_same_pin_tensor(concat_ce_out)
        outputs_result_cpu_pin.copy_(concat_ce_out, non_blocking=False)
        output_gpu = outputs_result_cpu_pin.to(flat_hidden_states.device, non_blocking=False)
        cuda_hook_time_end("get_outputs_cpu1")

        # cuda_hook_time("get_outputs_cpu2")
        # outputs_result_cpu_pin = gpinpool.alloc_same_pin_tensor(outputs_result)
        # outputs_result_cpu_pin.copy_(outputs_result, non_blocking=False)
        # output_gpu = outputs_result_cpu_pin.to(flat_hidden_states.device, non_blocking=False)
        # cuda_hook_time_end("get_outputs_cpu2")

        result = ExpertEinsumResult(final_hidden_states=output_gpu, time_einsum_end=time.time())
        output_queue.put(result)
        # del group_w1, group_w2, group_w3
        gpinpool.free(flat_hidden_states_cpu_pin)
        gpinpool.free(token_idxs_cpu_pin)
        gpinpool.free(outputs_result_cpu_pin)
        group_list = []
        group_list.append(group_w1)
        group_list.append(group_w2)
        group_list.append(group_w3)
        return output_gpu, group_list