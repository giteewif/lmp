import enum
import time
import os
import re
import torch
import torch.nn.functional as F
import queue
import copy
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
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

# ---------------------------------------------------------------------------
# Flash Attention 可用性与 head_dim 限制
# ---------------------------------------------------------------------------
# flash_attn GPU 内核限制（见 flash_attn_interface 报错信息）。
_FA2_MAX_HEAD_DIM: int = 256


def _flash_attn_import_ok() -> bool:
    try:
        import importlib.util

        if importlib.util.find_spec("flash_attn") is None:
            return False
        import flash_attn  # noqa: F401 – 仅做存在性验证
        return True
    except Exception:
        return False


_FLASH_ATTN_AVAILABLE: bool = _flash_attn_import_ok()


def _config_max_attention_head_dim(config) -> Optional[int]:
    """取 config 里可能参与 FlashAttention 的**最大**单头维度。

    Gemma4 等模型同时有 ``head_dim``（如 sliding）与 ``global_head_dim``（如 full attention）；
    仅读 ``head_dim`` 会误判为 ≤256，而 full 层实际 Q/K/V head 维为 ``global_head_dim``（如 512）。
    """
    dims: List[int] = []
    hd = getattr(config, "head_dim", None)
    if hd is not None:
        dims.append(int(hd))
    ghd = getattr(config, "global_head_dim", None)
    if ghd is not None:
        dims.append(int(ghd))
    hs = getattr(config, "hidden_size", None)
    nah = getattr(config, "num_attention_heads", None)
    if hs is not None and nah is not None and int(nah) > 0:
        dims.append(int(hs) // int(nah))
    if not dims:
        return None
    return max(dims)


def resolve_attn_implementation(config) -> str:
    """按环境与 config 选择 ``_attn_implementation``。

    FlashAttention 2 仅支持实际 head 维 ≤256；超出则退回 ``sdpa``，避免运行时崩溃。
    """
    if not _FLASH_ATTN_AVAILABLE:
        return "sdpa"
    max_hd = _config_max_attention_head_dim(config)
    if max_hd is not None and max_hd > _FA2_MAX_HEAD_DIM:
        logger.info(
            "max attention head_dim=%d > %d (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa",
            max_hd,
            _FA2_MAX_HEAD_DIM,
        )
        return "sdpa"
    return "flash_attention_2"

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
    # Optional: correlate MP results with submits (multi-worker).
    request_id: int = -1


def _print_group_bmm_self_test(msg: str) -> None:
    """与 ``lmp.cuda_memory_view._print_group_bmm_self_test`` 一致：``print``+flush；可选 ``LMP_MP_SELFTEST_DIAG`` 追加。"""
    print(msg, flush=True)
    path = (os.environ.get("LMP_MP_SELFTEST_DIAG") or "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


DEEPSEEK_MODEL_NAME_TYPE = "Deepseek"
MIXTRAL_MODEL_NAME_TYPE = "Mixtral"
QWEN2_MODEL_NAME_TYPE = "Qwen2_moe"
QWEN3_MODEL_NAME_TYPE = "Qwen3_moe"
QWEN3_5_MODEL_NAME_TYPE = "Qwen3_5_moe"
DEEPSEEK_V2_LITE="DeepSeek-V2-Lite"
GEMMA4_MODEL_NAME_TYPE = "Gemma4"
ERINE_MODEL_NAME_TYPE = "Erine"
GPT_OSS_MODEL_NAME_TYPE = "GPT-OSS"


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
        return "model.language_model"

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

    def _moe_gate_act_fn(self):
        """Gemma4 等用 ``hidden_activation``；多数 MoE 用 ``hidden_act``。"""
        act_name = getattr(self.config, "hidden_activation", None) or getattr(
            self.config, "hidden_act", None
        )
        if act_name is None:
            raise RuntimeError(
                "Cannot determine gate activation from config (expected hidden_activation or hidden_act)."
            )
        return ACT2FN[act_name]

    def _moe_gate_up_from_stacked(self, stacked_inputs: torch.Tensor, group_w1_w3: torch.Tensor):
        """
        对 padding 后的专家输入做一次 ``einsum`` 完成 gate 与 up 的线性，再对 gate 半幅做激活后与 up 相乘。

        ``stacked_inputs``: ``[E, T, H]``；``group_w1_w3``: ``[E, 2*I, H]``（前半为 gate，后半为 up）。
        返回 SwiGLU 前的逐元素乘结果，形状 ``[E, T, I]``。
        """
        gate_up = torch.einsum("eth,eih->eti", stacked_inputs, group_w1_w3)
        I_half = group_w1_w3.shape[1] // 2
        w1_out, w3_out = gate_up.split(I_half, dim=-1)
        act_fn = self._moe_gate_act_fn()
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
        act_fn = self._moe_gate_act_fn()
        w1_out = act_fn(w1_out)
        return w1_out * w3_out

        
    def init_chmv_meta_model(self, device=None):
        """
        在 ``init_empty_weights`` 上下文中构建 GPU 侧元模型 ``cmv.mlpm_ci``，供后续按需加载权重。

        各分支写入 ``cmv.mlpm_ci`` 并 ``eval()``；部分模型会设置 ``rotary_emb`` 等到指定 ``device``。
        """
        if self.model_name_type == DEEPSEEK_MODEL_NAME_TYPE:
            if self.model_path == DEEPSEEK_V2_LITE:
                with init_empty_weights():
                    # DeepSeek V2 Lite 的自定义 MLA attention 暂不支持 flash_attention_2，
                    # 保持 eager 以避免形状不匹配。
                    self.config._attn_implementation = "eager"
                    cm = DeepseekV2ForCausalLM(self.config)
                    cm.to(self.config.dtype)
                    cm.eval()
                    return cm
            else:
                with init_empty_weights():
                    self.config._attn_implementation = resolve_attn_implementation(self.config)
                    cm = DeepseekForCausalLM(self.config)
                    cm.to(self.config.dtype)
                    cm.eval()
                    return cm

                # Not need hm, we use einsum to restore experts weights from shared memory to model
                # hm =copy.deepcopy(cm)
                # hmv.mlpm_hi = None 
                # self.layerc = DeepseekDecoderLayer(self.config, 1)
            return cm
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            with init_empty_weights():
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                cm = Qwen2MoeForCausalLM(self.config)
                cm.model.rotary_emb = Qwen2MoeRotaryEmbedding(config=self.config, device=device)
                for i in range(self.config.num_hidden_layers):
                    cm.model.layers[i].self_attn.rotary_emb = \
                        Qwen2MoeRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.dtype)
                cm.eval()
                return cm
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            with init_empty_weights():
                cm = Qwen3MoeForCausalLM(self.config)
                cm.model.rotary_emb = Qwen3MoeRotaryEmbedding(config=self.config, device=device)
                # for i in range(self.config.num_hidden_layers):
                #     cm.model.layers[i].self_attn.rotary_emb = \
                #         Qwen3MoeRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.dtype)
                cm.eval()
                return cm
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            with init_empty_weights():
                self.config._attn_implementation = resolve_attn_implementation(self.config)
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
                cm.to(self.config.dtype)
                cm.eval()
                return cm
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            with init_empty_weights():
                # Gemma4 text attention 需要 position_embeddings。
                # flash_attention_2 与现有 mask 构造兼容；无 flash_attn 时退回 sdpa。
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                if self._gemma4_uses_language_model:
                    cm = Gemma4ForConditionalGeneration(self._raw_config)
                else:
                    cm = Gemma4ForCausalLM(self.config)
                cm.to(self.config.dtype)
                cm.eval()
                return cm
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            with init_empty_weights():
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                cm = Qwen3_5MoeForCausalLM(self.config)
                cm.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.dtype)
                cm.eval()
                return cm
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            with init_empty_weights():
                cm = GptOssForCausalLM(self.config)
                cm.model.rotary_emb = GptOssRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.dtype)
                cm.eval()
                return cm
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            with init_empty_weights():
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                cm = Ernie4_5_MoeForCausalLM(self.config)
                cm.model.rotary_emb = Ernie4_5_MoeRotaryEmbedding(config=self.config, device=device)
                cm.to(self.config.dtype)
                cm.eval()
                return cm
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
            layer.to(config.dtype)
            # mlpm_ci DeepseekOCalModel or DeepseekForCausalLM
            # self.cmv.mlpm_ci.model.layers[layer_idx]
            model.model.layers[layer_idx] = layer
            return layer
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen2MoeDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen3MoeDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = MixtralDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Gemma4TextDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen3_5MoeDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = GptOssDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Ernie4_5_MoeDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
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
            #     layer.to(config.dtype)
            #     layer.eval()
            #     return layer
                # if layer_idx >= 1:
                #     layer = copy.deepcopy(self.layerc)
                #     layer.self_attn.layer_idx = layer_idx
                # else:
                #     with init_empty_weights():
                #         layer = DeepseekDecoderLayer(config, layer_idx)
                # print(layer)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen2MoeDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen3MoeDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = MixtralDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Gemma4TextDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Qwen3_5MoeDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = GptOssDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
                layer.eval()
                return layer
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            with init_empty_weights():
                layer = Ernie4_5_MoeDecoderLayer(config, layer_idx)
                layer.to(config.dtype)
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
                    self.config._attn_implementation = resolve_attn_implementation(self.config)
                    model = DeepseekForCausalLM(self.config)
            elif self.model_name_type == QWEN2_MODEL_NAME_TYPE:
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                model = Qwen2MoeForCausalLM(self.config)
            elif self.model_name_type == QWEN3_MODEL_NAME_TYPE:
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                model = Qwen3MoeForCausalLM(self.config)
            elif self.model_name_type == MIXTRAL_MODEL_NAME_TYPE:
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                model = AutoModelForCausalLM.from_config(
                    self.config, trust_remote_code=True
                )
            elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                if self._gemma4_uses_language_model:
                    model = Gemma4ForConditionalGeneration(self._raw_config)
                else:
                    model = Gemma4ForCausalLM(self.config)
            elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                model = Qwen3_5MoeForCausalLM(self.config)
            elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                model = GptOssForCausalLM(self.config)
            elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
                self.config._attn_implementation = resolve_attn_implementation(self.config)
                model = Ernie4_5_MoeForCausalLM(self.config)
            else:
                raise ValueError(f"Invalid model name type: {self.model_name_type}")
            # model = AutoModelForCausalLM.from_config(
            #     self.config, trust_remote_code=True
            # )
        cuda_hook_time_end("create_empty_model")
        # cuda_hook_time("to_dtype")
        # model.to(dtype=self.config.dtype)
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
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            # Gemma4: 遍历 model.parameters() 获取 language_model 参数，从 state_dict 中查找对应权重
            language_model_indicator = "language_model"
        
            updated_params = 0
            with torch.no_grad():
                # 遍历模型的所有参数，只处理 language_model 相关的
                for param_name, param in model.named_parameters():
                    # 只处理 language_model 相关的参数
                    if language_model_indicator not in param_name:
                        continue
                    
                    # 尝试从 hm_state_dict 中获取对应的权重
                    tensor = None
                    
                    # 首先尝试直接匹配
                    if param_name in hm_state_dict:
                        tensor = hm_state_dict[param_name]
                    if tensor is not None:
                        try:
                            set_module_tensor_to_device(
                                model,
                                param_name,
                                tensor.device,
                                tensor,
                                clear_cache=False,
                            )
                            updated_params += 1
                        except Exception as exc:
                            logger.warning(
                                "Failed to assign tensor %s to module: %s", param_name, exc, exc_info=True
                            )
            logger.debug(
                "restore_hm_state_dict2model loaded %d language_model tensors for Gemma4 model",
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

    def get_embed_tokens(self, model):
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            inner = getattr(model.model, "language_model", model.model)
            return inner.embed_tokens
        else:
            return model.model.embed_tokens

    def get_final_norm(self, model):
        """与 ``iln_func`` 一致：解析文本塔（含 Gemma4 ``model.language_model``），返回最终 norm。"""
        inner = getattr(model.model, "language_model", model.model)
        return inner.norm

    def get_lm_head(self, model):
        """返回顶层 ``lm_head``（各支持的 CausalLM / Gemma4 条件生成类均在根模块上）。"""
        return model.lm_head

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

    def get_tensor_expert_group_key(self, tensor_name: str) -> Optional[Tuple[int, int]]:
        """
        Parse expert tensor name into a stable grouping key ``(layer_idx, expert_idx)``.

        Returns:
            - ``(layer_idx, expert_idx)`` for routed expert tensors
            - ``(layer_idx, -1)`` for fused-per-layer expert banks (e.g. Gemma4)
            - ``None`` for non-expert tensors
        """
        if not tensor_name:
            return None

        patterns: list[re.Pattern] = []
        if self.model_name_type in (
            DEEPSEEK_MODEL_NAME_TYPE,
            QWEN2_MODEL_NAME_TYPE,
            MIXTRAL_MODEL_NAME_TYPE,
        ):
            patterns = [
                re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\."),
                re.compile(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."),
            ]
        elif self.model_name_type in (
            QWEN3_MODEL_NAME_TYPE,
            QWEN3_5_MODEL_NAME_TYPE,
            GPT_OSS_MODEL_NAME_TYPE,
            ERINE_MODEL_NAME_TYPE,
        ):
            patterns = [
                re.compile(r"layers\.(\d+)\.mlp\.experts\.experts\.(\d+)\."),
            ]
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            # Also accept non-fused per-expert naming variants if present in
            # some checkpoints/tools, e.g.:
            #   ...layers.{L}.experts.{E}.gate_up_proj(.weight)
            #   ...layers.{L}.experts.{E}.down_proj(.weight)
            m = re.search(
                r"layers\.(\d+)\.experts\.(\d+)\.(gate_up_proj|down_proj)(?:\.|$)",
                tensor_name,
            )
            if m:
                return (int(m.group(1)), int(m.group(2)))
            # Gemma4 expert weights are fused banks at layer scope.
            m = re.search(r"layers\.(\d+)\.experts\.(gate_up_proj|down_proj)", tensor_name)
            if m:
                return (int(m.group(1)), -1)
            return None
        else:
            patterns = [
                re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\."),
            ]

        for pat in patterns:
            m = pat.search(tensor_name)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        return None

    def get_tensor_layer_idx(self, tensor_name: str) -> Optional[int]:
        """
        Parse tensor name and return layer index when available.

        Returns:
            - ``layer_idx`` for layer-scoped tensors
            - ``None`` for non-layer tensors
        """
        if not tensor_name:
            return None
        expert_key = self.get_tensor_expert_group_key(tensor_name)
        if expert_key is not None:
            return int(expert_key[0])
        m = re.search(r"layers\.(\d+)\.", str(tensor_name))
        if m:
            return int(m.group(1))
        return None

    def get_fused_expert_tensor_for_restore(self, tensor_name: str) -> Optional[Tuple[str, int]]:
        """
        Parse split expert tensor names and return fused restore source.

        Returns:
            - ``(fused_tensor_name, expert_id)`` for split names
            - ``None`` for non-split names

        Supported split naming:
            1) ``...layers.{L}.experts.{E}.gate_up_proj(.weight)?``
               ``...layers.{L}.experts.{E}.down_proj(.weight)?``
            2) ``...layers.{L}.experts.gate_up_proj(.weight)?.expert_{E}``
               ``...layers.{L}.experts.down_proj(.weight)?.expert_{E}``
        """
        if not tensor_name:
            return None

        m = re.search(
            r"^(.*\.layers\.\d+)\.experts\.(\d+)\.(gate_up_proj|down_proj)(\.weight)?$",
            tensor_name,
        )
        if m:
            prefix = m.group(1)
            expert_id = int(m.group(2))
            proj_name = m.group(3)
            weight_suffix = m.group(4) or ""
            return (f"{prefix}.experts.{proj_name}{weight_suffix}", expert_id)

        m = re.search(
            r"^(.*\.layers\.\d+)\.experts\.(gate_up_proj|down_proj)(\.weight)?\.expert_(\d+)$",
            tensor_name,
        )
        if m:
            prefix = m.group(1)
            proj_name = m.group(2)
            weight_suffix = m.group(3) or ""
            expert_id = int(m.group(4))
            return (f"{prefix}.experts.{proj_name}{weight_suffix}", expert_id)

        return None

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
            if type_idx.value == 0 or type_idx.value >= len(type_str_list):
                return []
            type_str = type_str_list[type_idx.value]
            experts_names = [
                f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.{type_str}.weight"
                for expert_idx in experts_idx_list
            ]
            return experts_names
        elif self.model_name_type == QWEN3_5_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_up_proj", "down_proj"]
            if type_idx.value == 0 or type_idx.value >= len(type_str_list):
                return []
            type_str = type_str_list[type_idx.value]
            experts_names = [
                f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.{type_str}.weight"
                for expert_idx in experts_idx_list
            ]
            return experts_names
        elif self.model_name_type == GPT_OSS_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_up_proj", "down_proj"]
            if type_idx.value == 0 or type_idx.value >= len(type_str_list):
                return []
            type_str = type_str_list[type_idx.value]
            experts_names = [
                f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.{type_str}.weight"
                for expert_idx in experts_idx_list
            ]
            return experts_names
        elif self.model_name_type == ERINE_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_up_proj", "down_proj"]
            if type_idx.value == 0 or type_idx.value >= len(type_str_list):
                return []
            type_str = type_str_list[type_idx.value]
            experts_names = [
                f"model.layers.{layer_idx}.mlp.experts.experts.{expert_idx}.{type_str}.weight"
                for expert_idx in experts_idx_list
            ]
            return experts_names
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            type_str_list = ["none", "gate_up_proj", "down_proj"]
            if type_idx.value == 0 or type_idx.value >= len(type_str_list):
                return []
            type_str = type_str_list[type_idx.value]
            p = self._gemma4_weight_prefix()
            # Gemma4 routed experts are stored as fused Parameter banks on the layer:
            #   `{p}.layers.{layer_idx}.experts.gate_up_proj`  shape [E, 2I, H]
            #   `{p}.layers.{layer_idx}.experts.down_proj`     shape [E, H, I]
            # The tensor index uses these names WITHOUT a trailing `.weight`.
            return [f"{p}.layers.{layer_idx}.experts.{type_str}"]

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

    def get_tensor_index_skip_prefixes(self):
        """
        返回 tensor_index 中需要跳过（不参与设备放置）的参数名前缀列表。
        """
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            # Gemma4 多模态场景下跳过 vision tower 权重。
            return ["model.vision_tower.", "model.embed_vision."]
        return []

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
            names_list = []
            expert_ids = expert_idx_list if expert_idx_list else list(range(self.get_experts_num()))
            for expert_idx in expert_ids:
                names_list.append(f"{p}.layers.{layer_idx}.experts.gate_up_proj")
                names_list.append(f"{p}.layers.{layer_idx}.experts.down_proj")
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
            names = [
                f"{p}.layers.{layer_idx}.self_attn.q_proj.weight",
                f"{p}.layers.{layer_idx}.self_attn.k_proj.weight",
                f"{p}.layers.{layer_idx}.self_attn.o_proj.weight",
                f"{p}.layers.{layer_idx}.self_attn.q_norm.weight",
                f"{p}.layers.{layer_idx}.self_attn.k_norm.weight",
            ]
            # Gemma4 may share V==K on some layers (attention_k_eq_v=True); those layers have no v_proj weights
            # in the checkpoint (`tensor_index.json`), so don't request them from sllm.
            layer_types = getattr(self.config, "layer_types", None)
            k_eq_v = bool(getattr(self.config, "attention_k_eq_v", False))
            if not (k_eq_v and layer_types is not None and layer_types[layer_idx] == "full_attention"):
                names.insert(2, f"{p}.layers.{layer_idx}.self_attn.v_proj.weight")
            return names
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

    def get_tensor_index_layer_names(self, layer_idx: int):
        """
        返回 tensor_index 定位中该层需要“同层同卡”放置的参数名：
        ``self_attn + gate + layernorm``。
        """
        names = []
        names.extend(self.get_attention_names(layer_idx))
        names.extend(self.get_gate_names(layer_idx))
        names.extend(self.get_layernorm_names(layer_idx))
        names.extend(self.get_layer_general_names(layer_idx))
        names.extend(self.get_mlp_names(layer_idx))
        return names

    def get_layer_general_names(self, layer_idx: int):
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            p = self._gemma4_weight_prefix()
            return [
                f"{p}.layers.{layer_idx}.layer_scalar",
            ]
        else:
            return []

    def get_mlp_names(self, layer_idx: int):
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            p = self._gemma4_weight_prefix()
            return [
                f"{p}.layers.{layer_idx}.mlp.down_proj.weight",
                f"{p}.layers.{layer_idx}.mlp.up_proj.weight",
                f"{p}.layers.{layer_idx}.mlp.gate_proj.weight",
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
            # Gemma4 RoPE needs `layer_type` when config.layer_types is set; otherwise forward uses None -> "None_inv_freq".
            layer_type = getattr(getattr(lm.layers[layer_idx], "self_attn", None), "layer_type", None)
            position_embeddings = lm.rotary_emb(hidden_states, position_ids, layer_type=layer_type)
            # Gemma4 attention requires `shared_kv_states` for KV-sharing layers. This dict must be shared across
            # layers within the same forward step; we attach it to the cache object and reset at layer 0.
            if layer_idx == 0 or not hasattr(past_key_value, "_gemma4_shared_kv_states"):
                setattr(past_key_value, "_gemma4_shared_kv_states", {})
            shared_kv_states = getattr(past_key_value, "_gemma4_shared_kv_states")
            hidden_states, _ = lm.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                shared_kv_states=shared_kv_states,
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
    def bench_ffn_skip_hidden(
        self, mi, layer_idx: int, residual_in: torch.Tensor, h_attn: torch.Tensor
    ) -> torch.Tensor:
        """Attention 子块后、与 FFN 相加的残差侧 hidden（非 Gemma4：``residual + attn``；Gemma4：``residual + paln(attn)``）。"""
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            return residual_in + self.paln_func(mi, layer_idx, h_attn)
        return residual_in + h_attn

    def bench_gate_moe_hidden(
        self, mi, layer_idx: int, residual_in: torch.Tensor, h_attn: torch.Tensor
    ) -> torch.Tensor:
        """路由 / fused MoE 的输入 hidden（非 Gemma4：``paln(residual+attn)``；Gemma4：与 ``bench_ffn_skip_hidden`` 相同）。"""
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            return self.bench_ffn_skip_hidden(mi, layer_idx, residual_in, h_attn)
        return self.paln_func(mi, layer_idx, residual_in + h_attn)

    def layer_uses_routed_moe(self, mi, layer_idx: int) -> bool:
        """本层是否走分路由专家（Gemma4 看 ``enable_moe_block``；其它模型看 ``first_k_dense_replace`` 等）。"""
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            return bool(getattr(lm.layers[layer_idx], "enable_moe_block", False))
        return layer_idx >= self.get_first_k_dense_replace()

    def ffn_dense_prefix_before_route(self, mi, layer_idx: int, skip_after_attn: torch.Tensor):
        """若存在「路由前的 dense 支路前缀」（当前为 Gemma4 MoE），返回该张量；否则 ``None``。"""
        if self.model_name_type != GEMMA4_MODEL_NAME_TYPE:
            return None
        lm = getattr(mi.model, "language_model", mi.model)
        layer = lm.layers[layer_idx]
        if not getattr(layer, "enable_moe_block", False):
            return None
        h = layer.pre_feedforward_layernorm(skip_after_attn)
        h = layer.mlp(h)
        return layer.post_feedforward_layernorm_1(h)

    def feedforward_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        """兼容旧名：等价于 ``ffn_dense_prefix_before_route``（无 dense 前缀时抛错）。"""
        out = self.ffn_dense_prefix_before_route(mi, layer_idx, hidden_states)
        if out is None:
            raise RuntimeError(
                "feedforward_func: no dense prefix for this model/layer; use ffn_dense_prefix_before_route"
            )
        return out

    def moe_experts_input_hidden(self, mi, layer_idx: int, hidden_for_gate: torch.Tensor) -> torch.Tensor:
        """进入 fused/CPU MoE 专家线性前的 hidden（Gemma4 MoE 为 ``pre_feedforward_layernorm_2``；否则与 gate 输入相同）。"""
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            layer = lm.layers[layer_idx]
            if getattr(layer, "enable_moe_block", False):
                return layer.pre_feedforward_layernorm_2(hidden_for_gate)
        return hidden_for_gate

    def ffn_merge_dense_and_routed(
        self,
        mi,
        layer_idx: int,
        ffn_skip: torch.Tensor,
        dense_prefix: Optional[torch.Tensor],
        routed_moe: torch.Tensor,
    ) -> torch.Tensor:
        """合并 routed MoE 输出与残差 / dense 支路（``dense_prefix`` 非空时为 Gemma4 MoE 完整后段；否则 ``routed_moe + ffn_skip``）。"""
        if dense_prefix is not None:
            lm = getattr(mi.model, "language_model", mi.model)
            layer = lm.layers[layer_idx]
            h2 = layer.post_feedforward_layernorm_2(routed_moe)
            h_comb = dense_prefix + h2
            h_out = layer.post_feedforward_layernorm(h_comb)
            return ffn_skip + h_out
        return routed_moe + ffn_skip

    def ffn_dense_non_routed_after_attn(self, mi, layer_idx: int, skip_after_attn: torch.Tensor) -> torch.Tensor:
        """Gemma4 无路由 MoE 时的整段 dense FFN（``pre_ffn → mlp → post_ffn`` 后与 ``skip`` 相加）。"""
        if self.model_name_type != GEMMA4_MODEL_NAME_TYPE:
            raise ValueError("ffn_dense_non_routed_after_attn is only defined for Gemma4")
        lm = getattr(mi.model, "language_model", mi.model)
        layer = lm.layers[layer_idx]
        h_ffn = layer.pre_feedforward_layernorm(skip_after_attn)
        h_ffn = layer.mlp(h_ffn)
        h_ffn = layer.post_feedforward_layernorm(h_ffn)
        return skip_after_attn + h_ffn

    def apply_decoder_layer_scale(self, mi, layer_idx: int, out: torch.Tensor) -> torch.Tensor:
        """Decoder 层输出尺度（Gemma4 为 ``layer_scalar``；其它模型恒等）。"""
        if self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            lm = getattr(mi.model, "language_model", mi.model)
            layer = lm.layers[layer_idx]
            return out * layer.layer_scalar.to(device=out.device, dtype=out.dtype)
        return out

    def ffn_skip_routed_moe_use_standalone_dense(self, mi, layer_idx: int) -> bool:
        """Gemma4 且无 ``enable_moe_block`` 时走独立 dense 块（与 bench 中 ``dense_prefix is None`` 且需 dense-only 一致）。"""
        if self.model_name_type != GEMMA4_MODEL_NAME_TYPE:
            return False
        return not self.layer_uses_routed_moe(mi, layer_idx)

    def paln_func(self, mi, layer_idx: int, hidden_states: torch.Tensor):
        """Post-attention 阶段处理（Gemma4 含 FFN dense 分支前半段）。"""
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
            layer = lm.layers[layer_idx]
            hidden_states = layer.post_attention_layernorm(hidden_states)
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
            # Gemma4 的前半段（pre_ffn + mlp + 可选 post_ffn_1）已在 paln_func 执行。
            if getattr(layer, "enable_moe_block", False) and hasattr(layer, "post_feedforward_layernorm_1"):
                return hidden_states
            return layer.post_feedforward_layernorm(hidden_states)
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
        elif self.model_name_type == GEMMA4_MODEL_NAME_TYPE:
            # Gemma4 text_config 使用 ``top_k_experts``（与 transformers Gemma4 一致），未必有 ``num_experts_per_tok``。
            if hasattr(self.config, "num_experts_per_tok"):
                return int(self.config.num_experts_per_tok)
            k = getattr(self.config, "top_k_experts", None)
            if k is None:
                raise RuntimeError(
                    "Gemma4 config must define top_k_experts or num_experts_per_tok for routed experts k."
                )
            return int(k)
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

    def get_moe_intermediate_size(self) -> int:
        """返回 MoE expert 的 intermediate size。"""
        v = getattr(self.config, "moe_intermediate_size", None)
        if v is None:
            raise RuntimeError("config must define moe_intermediate_size")
        return int(v)

    def get_num_hidden_layers(self) -> int:
        """返回模型层数。"""
        v = getattr(self.config, "num_hidden_layers", None)
        if v is None:
            raise RuntimeError("config must define num_hidden_layers")
        return int(v)

    def get_fused_experts_gate_up_down_act_fn(self, model, layer_idx: int):
        """
        从 ``model``（通常为 ``mlpm_ci``）解析 ``layers[layer_idx]``（兼容 Gemma4 多模态 ``language_model`` 路径），
        返回 fused routed 块的 ``gate_up_proj``、``down_proj`` 与 ``hidden_activation`` 对应的 ``act_fn``。
        """
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None and hasattr(getattr(model, "model", None), "language_model"):
            layers = model.model.language_model.layers
        if layers is None:
            raise RuntimeError("Cannot locate model.layers for fused expert execution")
        layer = layers[layer_idx]
        experts_mod = getattr(layer, "experts", None)
        if experts_mod is None:
            raise RuntimeError("Cannot locate experts module on this layer for fused path")
        gate_up = getattr(experts_mod, "gate_up_proj", None)
        down = getattr(experts_mod, "down_proj", None)
        if gate_up is None or down is None:
            raise RuntimeError("Fused path expects `gate_up_proj` and `down_proj` on experts")
        act_name = getattr(self.config, "hidden_activation", None) or getattr(self.config, "hidden_act", None)
        if act_name is None:
            raise RuntimeError("Cannot determine activation name from config for fused experts")
        return gate_up, down, ACT2FN[act_name]

    @staticmethod
    def _gmm_slots_presorted(
        x_sorted: torch.Tensor,
        eid_sorted: torch.Tensor,
        w_ehin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Grouped expert matmul on presorted inputs.

        Args:
            x_sorted: ``[S, H]`` already grouped by ``eid_sorted`` (non-decreasing).
            eid_sorted: ``[S]`` expert ids in ``[0, E)``.
            w_ehin: ``[E, H, N]`` expert weight bank (already transposed for matmul).
        """
        try:
            from transformers.integrations.moe import _grouped_mm as _tf_grouped_mm
        except Exception as exc:
            raise RuntimeError(
                "Grouped-mm requires `transformers.integrations.moe._grouped_mm` to be available."
            ) from exc

        offs = torch.cumsum(
            torch.bincount(eid_sorted, minlength=w_ehin.size(0)), dim=0
        ).to(torch.int32)
        return _tf_grouped_mm(x_sorted, w_ehin, offs=offs)

    @staticmethod
    def _batched_pad_inputs_presorted(
        x_sorted: torch.Tensor,
        expert_ids_sorted: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad+pack presorted rows to ``stacked_inputs`` for BMM.

        Returns:
            stacked_inputs: ``[E, max_tokens, H]``.
            counts: ``[E]`` tokens per expert.
        """
        counts = torch.bincount(expert_ids_sorted, minlength=num_experts)
        max_tokens = int(counts.max().item()) if counts.numel() else 0
        e = num_experts
        h = x_sorted.size(1)
        stacked_inputs = torch.zeros((e, max_tokens, h), device=x_sorted.device, dtype=x_sorted.dtype)

        co = counts.detach().view(-1).tolist()
        start = 0
        nb = x_sorted.is_cuda
        for expert_idx in range(e):
            c = int(co[expert_idx])
            if c:
                stacked_inputs[expert_idx, :c].copy_(x_sorted[start : start + c], non_blocking=nb)
            start += c
        return stacked_inputs, counts
    @staticmethod
    def _gather_sort_and_pad_presorted(
        flat_hidden_states: torch.Tensor,
        slot_token_row: torch.Tensor,
        slot_expert_ids: torch.Tensor,
        global_to_local: torch.Tensor,
        slot_ids: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        统一的 gather + sort + pad 操作，避免中间张量拷贝。
        
        Args:
            flat_hidden_states: [T, H] 原始展平的 hidden states
            slot_token_row: [K*T] 槽位到 token 行的映射
            slot_expert_ids: [K*T] 槽位的专家 ID（全局）
            global_to_local: [num_global_experts] 全局到局部专家 ID 的映射
            slot_ids: 选中的槽位索引
            num_experts: 专家数量（局部）
            
        Returns:
            stacked: [E, max_tokens, H] pad 后的堆叠张量
            counts: [E] 每个专家的 token 数量
            perm: 排序后的索引（用于后续 unpad）
        """
        # Step 1: 获取选中槽位的相关数据
        tok_rows = slot_token_row.index_select(0, slot_ids)
        expert_ids_global = slot_expert_ids.index_select(0, slot_ids)
        expert_ids_local = global_to_local.index_select(0, expert_ids_global)
        
        # Step 2: 排序并获取排序索引
        expert_ids_sorted, perm = torch.sort(expert_ids_local)
        
        # Step 3: 计算 counts（避免 bincount 的额外开销）
        counts = torch.bincount(expert_ids_sorted, minlength=num_experts)
        max_tokens = int(counts.max().item()) if counts.numel() else 0
        
        # Step 4: 直接构建 stacked，避免中间 x_sorted 张量
        h = flat_hidden_states.size(1)
        stacked = torch.zeros((num_experts, max_tokens, h), 
                            device=flat_hidden_states.device, 
                            dtype=flat_hidden_states.dtype)
        
        # 使用 scatter 直接写入，避免循环
        # 获取排序后的 token 行索引
        tok_rows_sorted = tok_rows.index_select(0, perm)
        
        # 计算每个专家在 stacked 中的位置
        cumsum_counts = torch.cumsum(counts, dim=0)
        cumsum_counts = torch.nn.functional.pad(cumsum_counts[:-1], (1, 0), value=0)
        
        # 构建目标索引
        expert_indices = expert_ids_sorted
        position_indices = torch.arange(len(expert_ids_sorted), device=flat_hidden_states.device)
        position_indices = position_indices - cumsum_counts[expert_indices]
        
        # 使用 scatter 写入
        stacked[expert_indices, position_indices] = flat_hidden_states[tok_rows_sorted]
        
        return stacked, counts, perm    
        
    @staticmethod
    def _batched_unpad_outputs(y_stacked: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Inverse of padding: ``[E, max_tokens, N]`` -> ``[S, N]`` in expert-sorted order.

        ``y_stacked`` 可在任意 device；``counts`` 通常在 CPU（仅读标量 ``.item()``）。
        """
        outs: list[torch.Tensor] = []
        e = int(counts.numel())
        for expert_idx in range(e):
            c = int(counts[expert_idx].item())
            if c:
                outs.append(y_stacked[expert_idx, :c])
        if not outs:
            return y_stacked.reshape(0, y_stacked.size(-1))
        return torch.cat(outs, dim=0)


    def _batched_pad_inputs_presorted_on_cpu(
        self,
        flat_hidden_states: torch.Tensor,
        idxs: torch.Tensor,
        expert_idx_list: list[int],
        expert_indices_map: Dict[int, Tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CPU 版本的 batched pad：从 CPU 上的 flat hidden states 按专家分组并 pad。
        
        与 ``_gather_cuda_flat_hidden_to_stacked_cpu_pin`` 逻辑一致，但输入数据已在 CPU 上。
        
        Args:
            flat_hidden_states: CPU 上的展平 hidden states，形状为 ``[T, H]``。
            idxs: CPU 上的槽位索引数组。
            expert_idx_list: 专家索引列表，用于分组。
            expert_indices_map: 专家索引到槽位范围的映射，expert_indices_map[eid] = (start_slot_idx, end_slot_idx)。
            k_per_tok: 每个 token 的专家数，默认 8。
            
        Returns:
            stacked: ``[E, max_tokens, H]``，pad 后的堆叠张量。
            counts: ``[E]``，每个专家的 token 数量。
        """
        k_per_tok = int(self.get_experts_per_tok())
        half_e = len(expert_idx_list)
        if half_e == 0:
            return None, None
        
        # 计算每个专家的 token 数量
        n_tokens_per_expert = [0] * half_e
        for li, eid in enumerate(expert_idx_list):
            start_idx, end_idx = expert_indices_map[eid]
            if end_idx > start_idx:
                n_tokens_per_expert[li] = end_idx - start_idx
        
        total_tokens = sum(n_tokens_per_expert)
        if total_tokens == 0:
            return None, None
        
        max_tokens = max(n_tokens_per_expert)
        h = flat_hidden_states.size(1)
        
        # 创建 stacked 张量
        stacked = torch.zeros((half_e, max_tokens, h), device='cpu', dtype=flat_hidden_states.dtype)
        counts = torch.tensor(n_tokens_per_expert, dtype=torch.int64, device='cpu')
        
        token_idxs = idxs // int(k_per_tok)

        for li, eid in enumerate(expert_idx_list):
            n = int(n_tokens_per_expert[li])
            if n <= 0:
                continue
            start_slot_idx, end_slot_idx = expert_indices_map[eid]
            token_indices = token_idxs[start_slot_idx:end_slot_idx]
            stacked[li, :n].copy_(flat_hidden_states.index_select(0, token_indices), non_blocking=False)
        
        return stacked, counts

    @staticmethod
    def _batched_unpad_outputs_on_cpu(y_stacked: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        CPU 版本的 unpad：将 ``[E, max_tokens, N]`` 转换为 ``[S, N]``。
        
        Args:
            y_stacked: CPU 上的堆叠输出张量。
            counts: CPU 上的每个专家 token 数量。
            
        Returns:
            展平后的输出张量，形状为 ``[S, N]``。
        """
        outs: list[torch.Tensor] = []
        e = int(counts.numel())
        for expert_idx in range(e):
            c = int(counts[expert_idx].item())
            if c:
                outs.append(y_stacked[expert_idx, :c])
        if not outs:
            return y_stacked.reshape(0, y_stacked.size(-1))
        return torch.cat(outs, dim=0)

    @staticmethod
    def _batched_unpad_outputs_into(
        dst: torch.Tensor,
        y_stacked: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """
        与 ``_batched_unpad_outputs`` 相同的行拼接顺序，但直接写入已分配好的 ``dst``（``[sum(counts), N]``），
        避免 ``torch.cat`` 产生中间张量及额外整块 ``copy_``。
        """
        e = int(counts.numel())
        # 一次把 counts 拉到 CPU，避免每个 expert 在 GPU 上 .item() 同步。
        co = counts.detach().view(-1).tolist()
        offset = 0
        nb = bool(dst.is_cuda and y_stacked.is_cuda)
        for expert_idx in range(e):
            c = int(co[expert_idx])
            if c:
                dst[offset : offset + c].copy_(y_stacked[expert_idx, :c], non_blocking=nb)
                offset += c
        if offset != dst.size(0):
            raise RuntimeError(
                f"_batched_unpad_outputs_into: wrote {offset} rows but dst has {dst.size(0)} rows."
            )

    @torch.no_grad()
    def _gather_cuda_flat_hidden_to_stacked_cpu_pin(
        self,
        flat_hidden_states: torch.Tensor,
        idxs: torch.Tensor,
        expert_idx_list: list[int],
        expert_indices_map: Dict[int, Tuple[int, int]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        从 CUDA 上的 ``flat_hidden_states``（``[T, H]``）按 MoE 槽位图取出各专家 token 行，
        写入 CPU pinned 的 ``stacked``（``[E, max_tokens, H]``），布局与 ``_batched_pad_inputs_presorted``
        输出一致；``expert_idx_list`` 的下标 ``li`` 即组内局部专家维。

        Returns:
            ``(stacked_pin, counts)``：``counts`` 为 ``[E]`` int64 CPU，第 ``li`` 维为该专家 token 数。
            若 ``len(expert_idx_list)==0`` 或总 token 数为 0，返回 ``(None, None)``（调用方应提前判空时可跳过）。

        Raises:
            RuntimeError: ``flat_hidden_states`` 不在 CUDA 上。
        """
        half_e = len(expert_idx_list)
        if half_e == 0:
            return None, None
        if not flat_hidden_states.is_cuda:
            raise RuntimeError(
                "_gather_cuda_flat_hidden_to_stacked_cpu_pin requires flat_hidden_states on CUDA, "
                f"got device={flat_hidden_states.device}."
            )

        cuda_hook_time("prepare_group_stack")
        x_dev = flat_hidden_states.device
        k_per_tok = int(self.get_experts_per_tok())
        t = int(flat_hidden_states.size(0))
        s_total = int(idxs.numel())
        if t <= 0 or s_total <= 0:
            return None, None
        if s_total % t != 0:
            raise RuntimeError(
                f"Invalid routed slot shape: slots={s_total} is not divisible by tokens={t}."
            )
        k_actual = s_total // t
        if k_actual < 1:
            k_actual = k_per_tok
        token_idxs = (idxs.to(device=x_dev, dtype=torch.int64)) // int(k_actual)

        n_tokens_per_expert = [0] * half_e
        for li, eid in enumerate(expert_idx_list):
            start_idx, end_idx = expert_indices_map[eid]
            if end_idx > start_idx:
                n_tokens_per_expert[li] = end_idx - start_idx
        if sum(n_tokens_per_expert) == 0:
            return None, None

        max_tokens = max(n_tokens_per_expert)
        h = int(flat_hidden_states.size(-1))
        counts = torch.tensor(n_tokens_per_expert, dtype=torch.int64, device="cpu")

        stacked_proto = torch.empty(
            (half_e, max_tokens, h),
            dtype=flat_hidden_states.dtype,
            device=torch.device("cpu"),
        )
        stacked_pin = gpinpool.alloc_same_pin_tensor(stacked_proto)
        stacked_pin.zero_()
        cuda_hook_time_end("prepare_group_stack")

        cuda_hook_time("group stack")
        # 默认使用紧排 D2H（S*H）+ CPU pack，避免在 GPU 上分配 padded 的 [E*max_tokens,H] 导致 OOM。
        # 若确实需要“一次性 D2H 到 stacked_pin”，可设环境变量 `LMP_GATHER_STACKED_GPU_PAD=1`。
        s = int(sum(n_tokens_per_expert))
        tok_list: list[torch.Tensor] = []
        for li, eid in enumerate(expert_idx_list):
            n = n_tokens_per_expert[li]
            if n == 0:
                continue
            start_idx, end_idx = expert_indices_map[eid]
            tok_list.append(token_idxs[start_idx:end_idx])
        tok_all = torch.cat(tok_list, dim=0)  # [S] on CUDA

        x_sorted_gpu = torch.index_select(flat_hidden_states, 0, tok_all)  # [S, H] on CUDA

        use_gpu_padded = (os.environ.get("LMP_GATHER_STACKED_GPU_PAD", "").strip() == "1")
        if use_gpu_padded:
            # Build destination indices for padded layout [E,max_tokens,H] flattened to [E*max_tokens, H].
            dst_rows_cpu: list[torch.Tensor] = []
            for li in range(half_e):
                n = int(n_tokens_per_expert[li])
                if n:
                    base = li * max_tokens
                    dst_rows_cpu.append(
                        torch.arange(base, base + n, dtype=torch.int64, device="cpu")
                    )
            dst_rows = torch.cat(dst_rows_cpu, dim=0).to(device=x_dev, non_blocking=True)  # [S] on CUDA

            stacked_gpu_flat = torch.zeros(
                (half_e * max_tokens, h),
                dtype=flat_hidden_states.dtype,
                device=x_dev,
            )
            stacked_gpu_flat.index_copy_(0, dst_rows, x_sorted_gpu)
            stacked_gpu = stacked_gpu_flat.view(half_e, max_tokens, h)
            # single D2H copy into the final pinned buffer
            stacked_pin.copy_(stacked_gpu, non_blocking=True)
        else:
            # Single D2H copy of the compact [S,H] buffer, then CPU->CPU pack into stacked_pin.
            x_sorted_proto = torch.empty(
                (s, h),
                dtype=flat_hidden_states.dtype,
                device=torch.device("cpu"),
            )
            x_sorted_pin = gpinpool.alloc_same_pin_tensor(x_sorted_proto)
            x_sorted_pin.copy_(x_sorted_gpu, non_blocking=False)

            start = 0
            for li in range(half_e):
                n = int(n_tokens_per_expert[li])
                if n:
                    stacked_pin[li, :n].copy_(x_sorted_pin[start : start + n], non_blocking=False)
                start += n
            gpinpool.free(x_sorted_pin)
        cuda_hook_time_end("group stack")

        return stacked_pin, counts

    @torch.no_grad()
    def fused_experts_gate_up_down_mm_presorted(
        self,
        x_slots_sorted: torch.Tensor,
        expert_ids_sorted: torch.Tensor,
        gate_up_w_eh2i: torch.Tensor,
        down_w_eih: torch.Tensor,
        act_fn,
        mm_backend: str,
    ) -> torch.Tensor:
        """
        Fused routed experts (gate+up then down) compute for presorted slot rows.

        Args:
            x_slots_sorted: ``[S, H]`` rows grouped by expert id (non-decreasing).
            expert_ids_sorted: ``[S]`` expert id for each row.
            gate_up_w_eh2i: ``[E, H, 2I]``.
            down_w_eih: ``[E, I, H]``.
            act_fn: activation for gate half.
            mm_backend: ``"bmm"`` or ``"gmm"``.

        Returns:
            y_slots_sorted: ``[S, H]`` aligned with input order.
        """
        mm_backend = (mm_backend or "bmm").strip().lower()
        if mm_backend == "gmm":
            t0 = time.perf_counter()
            gate_up_slots = self._gmm_slots_presorted(x_slots_sorted, expert_ids_sorted, gate_up_w_eh2i)
            half = gate_up_slots.size(-1) // 2
            gate, up = gate_up_slots.split(half, dim=-1)
            mid_slots = act_fn(gate) * up
            y = self._gmm_slots_presorted(mid_slots, expert_ids_sorted, down_w_eih)
            t_ms = (time.perf_counter() - t0) * 1e3
            e = int(gate_up_w_eh2i.size(0))
            s = int(x_slots_sorted.size(0))
            h = int(x_slots_sorted.size(1))
            logger.info(
                "[fused_experts] gmm total=%.3fms E=%d S=%d H=%d dtype=%s",
                t_ms,
                e,
                s,
                h,
                str(x_slots_sorted.dtype),
            )
            return y

        # bmm：输入仍是 ``[S,H]`` + 专家 id，必须先 pad；matmul/unpad 与 ``fused_experts_gate_up_down_bmm_from_padded`` 共用。
        # 若调用方已 pad，请直接调 ``fused_experts_gate_up_down_bmm_from_padded`` / ``..._into``，勿再经本函数 bmm 分支。

        e = int(gate_up_w_eh2i.size(0))
        s = int(x_slots_sorted.size(0))
        h = int(x_slots_sorted.size(1))
        n2 = int(gate_up_w_eh2i.size(-1))

        t0 = time.perf_counter()
        stacked, counts = self._batched_pad_inputs_presorted(x_slots_sorted, expert_ids_sorted, e)
        t_pad = (time.perf_counter() - t0) * 1e3
        max_tokens = int(stacked.size(1)) if stacked.dim() == 3 else 0

        t0 = time.perf_counter()
        y = self.fused_experts_gate_up_down_bmm_from_padded(
            stacked, counts, gate_up_w_eh2i, down_w_eih, act_fn
        )
        t_mm = (time.perf_counter() - t0) * 1e3

        logger.info(
            "[fused_experts] bmm_presorted pad=%.3fms matmul+act+unpad=%.3fms total=%.3fms "
            "E=%d S=%d H=%d 2I=%d maxT=%d dtype=%s",
            t_pad,
            t_mm,
            t_pad + t_mm,
            e,
            s,
            h,
            n2,
            max_tokens,
            str(x_slots_sorted.dtype),
        )
        return y

    @torch.no_grad()
    def fused_experts_gate_up_down_bmm_from_padded(
        self,
        stacked_inputs: torch.Tensor,
        counts: torch.Tensor,
        gate_up_w_eh2i: torch.Tensor,
        down_w_eih: torch.Tensor,
        act_fn,
    ) -> torch.Tensor:
        """
        BMM 路径（已 pad/pack 好输入）：避免在函数内重复 `pad`。

        Args:
            stacked_inputs: ``[E, max_tokens, H]`` (padded).
            counts: ``[E]`` tokens per expert (建议在 CPU；GPU 也可但 `.item()` 会触发同步).
            gate_up_w_eh2i: ``[E, H, 2I]``.
            down_w_eih: ``[E, I, H]``.

        Returns:
            y_slots_sorted: ``[S, H]`` (expert-sorted order).
        """
        s = int(counts.sum().item())
        out = torch.empty(
            (s, stacked_inputs.size(-1)),
            device=stacked_inputs.device,
            dtype=stacked_inputs.dtype,
        )
        self.fused_experts_gate_up_down_bmm_from_padded_into(
            out, stacked_inputs, counts, gate_up_w_eh2i, down_w_eih, act_fn
        )
        return out

    @torch.no_grad()
    def fused_experts_gate_up_down_bmm_from_padded_into(
        self,
        dst: torch.Tensor,
        stacked_inputs: torch.Tensor,
        counts: torch.Tensor,
        gate_up_w_eh2i: torch.Tensor,
        down_w_eih: torch.Tensor,
        act_fn,
    ) -> None:
        """BMM 路径：输入已为 ``[E,maxT,H]`` pad 布局；仅 matmul+act+unpad 写入 ``dst``（``[S,H]``）。"""
        e, max_t, h = stacked_inputs.shape[0], stacked_inputs.shape[1], stacked_inputs.shape[2]
        t0 = time.perf_counter()
        torch.cuda.set_device(stacked_inputs.device)  # 确保当前线程持有正确的 CUDA context，避免 cuBLAS 懒初始化警告
        gu = torch.bmm(stacked_inputs, gate_up_w_eh2i)
        t_bmm1 = (time.perf_counter() - t0) * 1e3
        half = gu.size(-1) // 2
        t0 = time.perf_counter()
        g, u = gu.split(half, dim=-1)
        mid = act_fn(g) * u
        t_act = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        y_st = torch.bmm(mid, down_w_eih)
        t_bmm2 = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        self._batched_unpad_outputs_into(dst, y_st, counts)
        t_unpad = (time.perf_counter() - t0) * 1e3
        t_all = t_bmm1 + t_act + t_bmm2 + t_unpad
        logger.info(
            "[fused_experts] bmm_from_padded bmm1=%.3fms act=%.3fms bmm2=%.3fms unpad=%.3fms total=%.3fms "
            "E=%d maxT=%d S=%d H=%d dev=%s dtype=%s",
            t_bmm1,
            t_act,
            t_bmm2,
            t_unpad,
            t_all,
            int(e),
            int(max_t),
            int(dst.size(0)),
            int(h),
            str(stacked_inputs.device),
            str(stacked_inputs.dtype),
        )

    

    

    

            
    
    
    
    

    
        

    

    

    