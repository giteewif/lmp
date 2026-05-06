import os
import time
import warnings
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Dict, Optional
from transformers import AutoConfig
from transformers.activations import ACT2FN

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.modeling_attn_mask_utils",
)
from accelerate.utils import set_module_tensor_to_device

from sllm_store.client import SllmStoreClient
from sllm_store._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors_from_shared_memory_names,
    restore_tensors2,
)

from lmp.sllm_store_c import load_into_gpu_async

if TYPE_CHECKING:
    from lmp.sllm_thread_manager import SLLMTM

from models.mlpmodule import QWEN3_MODEL_NAME_TYPE, MLPModuleWrapper, WeightType
from utils.helper import (
    load_json, 
    calculate_device_offset, 
    get_expert_device_distribution,
    calculate_expert_memory_size,
    filter_experts_by_memory
)
from utils.cuda_h import *
from utils.logger import init_logger
logger = init_logger(__name__)

# ``_test_group_*_fused_experts``: pick half the experts to **minimize total top-1 routed tokens** into that set.
# ``noncontiguous`` = global ``half_e`` smallest per-expert counts; ``contiguous`` = best length-``half_e`` window.
# Env: ``noncontiguous`` | ``contiguous`` | ``both`` (default ``noncontiguous``).
_GROUP_FUSED_TEST_HALF_ENV = "LMP_TEST_GROUP_FUSED_EXPERTS_HALF"


def _print_group_bmm_self_test(msg: str) -> None:
    """Half-expert fused BMM 自检进度：``print``+flush；若设 ``LMP_MP_SELFTEST_DIAG`` 则追加到该文件。"""
    print(msg, flush=True)
    path = (os.environ.get("LMP_MP_SELFTEST_DIAG") or "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


def _group_fused_test_parse_half_modes() -> list[str]:
    raw = os.environ.get(_GROUP_FUSED_TEST_HALF_ENV, "noncontiguous").strip().lower()
    if raw == "both":
        return ["contiguous", "noncontiguous"]
    if raw in ("contiguous", "noncontiguous"):
        return [raw]
    logger.warning(
        "%s=%r is not contiguous|noncontiguous|both; using noncontiguous",
        _GROUP_FUSED_TEST_HALF_ENV,
        raw,
    )
    return ["noncontiguous"]


def _group_fused_select_half_experts(
    expert_ids: torch.Tensor,
    num_experts_total: int,
    half_e: int,
    half_mode: str,
) -> tuple[list[int], int, int]:
    """
    Pick ``half_e`` experts for group fused tests.

    ``expert_ids`` 为 **一维** 专家 id：可以是每个 token 的 top‑1（长度 ``T``），也可以是展平后的 **top‑k
    槽位**（长度 ``T*K``，每槽一路专家）。``torch.bincount`` 按「槽位次数」统计，目标仍是使落入所选半套
    的 **总槽位数** 之和尽量小。

    - ``noncontiguous``: pick ``half_e`` distinct experts with globally minimum token sum (take ``half_e``
      smallest per-expert counts; ties broken stably by smaller expert id). This is optimal for the sum.
    - ``contiguous``: among windows ``[s, s+half_e)``, minimize the same sum; ties broken by smaller
      ``max`` count in the window (tighter BMM padding), then smaller ``s``.
    """
    counts_all = torch.bincount(expert_ids, minlength=num_experts_total)
    device = expert_ids.device
    if half_mode == "contiguous":
        best_start = 0
        best_key: tuple[int, int, int] | None = None  # (sum_tokens, max_tokens, start) lexicographic min
        for start_e in range(0, num_experts_total - half_e + 1):
            slice_c = counts_all[start_e : start_e + half_e]
            window_sum = int(slice_c.sum().item())
            window_max = int(slice_c.max().item())
            key = (window_sum, window_max, start_e)
            if best_key is None or key < best_key:
                best_key = key
                best_start = start_e
        expert_half_list = list(range(best_start, best_start + half_e))
        window_tokens = best_key[0] if best_key is not None else 0
        idx_t = torch.tensor(expert_half_list, dtype=torch.int64, device=device)
        max_tokens = int(counts_all[idx_t].max().item())
    else:
        expert_order = torch.argsort(counts_all, descending=False, stable=True)
        expert_half_list = [int(expert_order[i].item()) for i in range(half_e)]
        window_tokens = int(counts_all[expert_order[:half_e]].sum().item())
        idx_t = torch.tensor(expert_half_list, dtype=torch.int64, device=device)
        max_tokens = int(counts_all[idx_t].max().item())
    return expert_half_list, window_tokens, max_tokens


def _mask_topk_weights_outside_expert_half(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_half_list: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Half-set 内的槽位保持原 ``topk_ids`` / ``topk_weights``；半套外槽位：

    - ``topk_ids`` 置为 **-1**（与 ``kt_kernel_ext`` 中 ``should_skip_expert`` 一致：``id < 0`` 整槽不参与
      gate/up/down，不占用该专家的激活计数）。勿用 **0**：0 是合法专家下标，会误路由到 0 号专家。
    - ``topk_weights`` 置 0，行不做归一化。

    返回的 ``topk_ids`` 为 **新张量**（clone 后改槽位），shape 与输入相同。
    """
    device = topk_ids.device
    half_t = torch.tensor(expert_half_list, dtype=torch.int64, device=device)
    if int(half_t.numel()) == 0:
        raise ValueError("expert_half_list must be non-empty")
    in_half = torch.isin(topk_ids, half_t)
    skip_id = torch.tensor(-1, dtype=topk_ids.dtype, device=device)
    new_ids = torch.where(in_half, topk_ids, skip_id).contiguous()
    new_w = (topk_weights * in_half.to(dtype=topk_weights.dtype)).contiguous()
    return new_ids, new_w


def _log_kt_kernel_topk_activation(
    topk_ids: torch.Tensor,
    num_experts_total: int,
    expert_half_list: list[int],
    tokens_num: int,
    tok_k: int,
    *,
    log_prefix: str = "kt-kernel infer",
) -> None:
    """掩码后 top-k 槽位统计：``id >= 0`` 计入激活，``id < 0`` 为禁用槽位。"""
    slots_i = topk_ids.reshape(-1).to(torch.int64)
    active_m = slots_i >= 0
    n_slots = int(slots_i.numel())
    n_active = int(active_m.sum().item())
    n_masked = n_slots - n_active
    if n_active > 0:
        act_counts = torch.bincount(slots_i[active_m], minlength=num_experts_total)
    else:
        act_counts = torch.zeros(num_experts_total, dtype=torch.int64, device=slots_i.device)
    experts_used = int((act_counts > 0).sum().item())
    busiest = int(torch.argmax(act_counts).item()) if n_active > 0 else -1
    half_t = torch.tensor(expert_half_list, dtype=torch.int64, device=act_counts.device)
    half_counts = act_counts[half_t]
    logger.info(
        "%s: topk activation (post-mask): T=%d K=%d slots=%d active=%d masked=%d "
        "experts_hit=%d/%d per_expert_slots min/mean/max(in_half)=%d/%.2f/%d busiest_e=%d "
        "half_expert_slot_sum=%d",
        log_prefix,
        tokens_num,
        tok_k,
        n_slots,
        n_active,
        n_masked,
        experts_used,
        num_experts_total,
        int(half_counts.min().item()) if half_counts.numel() else 0,
        float(half_counts.float().mean().item()) if half_counts.numel() else 0.0,
        int(half_counts.max().item()) if half_counts.numel() else 0,
        busiest,
        int(half_counts.sum().item()),
    )


def _log_topk_activation_distribution(
    topk_ids: torch.Tensor,
    num_experts_total: int,
    *,
    log_prefix: str,
    ignore_negative: bool = False,
    log_cumulative: bool = True,
) -> None:
    """
    通用 top-k 激活统计：
    - 按槽位统计每个专家激活次数（支持 ``ignore_negative`` 跳过 ``id < 0``）
    - 输出按激活次数升序的专家分布
    - 可选输出累计覆盖占比（与 ``test_gate_experts`` 风格一致）
    """
    flat_ids = topk_ids.reshape(-1).to(torch.int64).cpu()
    if ignore_negative:
        flat_ids = flat_ids[flat_ids >= 0]

    counts = torch.bincount(flat_ids, minlength=num_experts_total)
    total_activations = int(counts.sum().item())
    active_experts = int((counts > 0).sum().item())
    top_k = int(topk_ids.shape[-1]) if topk_ids.dim() > 1 else 1

    stats: list[tuple[int, int, float]] = []
    for expert_id in range(num_experts_total):
        cnt = int(counts[expert_id].item())
        pct = (cnt / total_activations * 100.0) if total_activations > 0 else 0.0
        stats.append((expert_id, cnt, pct))
    stats.sort(key=lambda item: (item[1], item[0]))

    logger.info(
        "%s: top_k=%d experts=%d active_experts=%d total_activations=%d",
        log_prefix,
        top_k,
        num_experts_total,
        active_experts,
        total_activations,
    )
    logger.info("%s: expert activation distribution (sorted by token count asc):", log_prefix)
    for expert_id, cnt, pct in stats:
        logger.info(
            "%s: expert=%d tokens=%d pct=%.4f%%",
            log_prefix,
            expert_id,
            cnt,
            pct,
        )

    if not log_cumulative:
        return

    cumulative_tokens = 0
    logger.info(
        "%s: cumulative token ratio by ascending experts "
        "(k/%d experts -> cumulative_tokens/total_activations)",
        log_prefix,
        num_experts_total,
    )
    for k, (_expert_id, cnt, _pct) in enumerate(stats, start=1):
        cumulative_tokens += cnt
        cumulative_pct = (cumulative_tokens / total_activations * 100.0) if total_activations > 0 else 0.0
        expert_pct = k / num_experts_total * 100.0
        logger.info(
            "%s: k=%d expert_pct=%.4f%% cumulative_tokens=%d cumulative_pct=%.4f%%",
            log_prefix,
            k,
            expert_pct,
            cumulative_tokens,
            cumulative_pct,
        )


class CudaMemoryView:
    def __init__(
        self,
        mlpm: MLPModuleWrapper,
        client: SllmStoreClient,
        tensor_index_resize_json: dict,
        meta_model,
        device_list: list[str]
    ):
        self.mlpm = mlpm
        self.client = client
        self.tensor_index_resize_json = tensor_index_resize_json
        self.mlpm_ci = meta_model

        mshm_names, chunk_size = self.client.get_model_shared_memory_names(self.mlpm.model_path)
        if len(mshm_names) <= 0:
            raise ValueError(f"Only Support shared memory, But sllm not shared")     
        self.mchunk_size = chunk_size
                
        self.device_list = device_list
        self.device1_str = device_list[0]
        self.device1 = int(self.device1_str.split(":")[1])
        
        self.device_uuid_map = get_device_uuid_map()

        self.cuda_memory_ptrs_allocated = []

        self.sllmtm: Optional["SLLMTM"] = None

    def restore2model(self, model_state_dict, model):
        with torch.no_grad():
            for name, param in model_state_dict.items():
                set_module_tensor_to_device(model, name, param.device, param, clear_cache=False)
        
    def restore2model_strict(self, model_state_dict, model):
        model_param_names = set(dict(model.named_parameters()).keys())
        with torch.no_grad():
            for name, param in model_state_dict.items():
                if name not in model_param_names:
                    continue
                set_module_tensor_to_device(model, name, param.device, param, clear_cache=False)


    def load_general_and_init(self):     
        cuda_hook_time("load_general")
        tensor_index_general_names = self.mlpm.get_tensor_index_general_names()
        tensor_index_init_names = tensor_index_general_names

        ret1, replica_uuid1, state_dict1 = \
            self.allocate_cuda_memory_and_load_into_gpu(tensor_index_init_names, device_index_int=self.device1)

        self.restore2model(state_dict1, self.mlpm_ci)
        self.wait_load_into_gpu(replica_uuid1)
        cuda_hook_time_end("load_general")


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
        device_idx_int = self.device1
        cuda_hook_time(f"start_load_qkvogn_s_weight_l_{layer_idx}")
        tensor_al1_names = self.mlpm.get_attention_names(layer_idx=layer_idx)
        tensor_ln1_names = self.mlpm.get_layernorm_names(layer_idx=layer_idx)
        tensor_gate_names = self.mlpm.get_gate_names(layer_idx=layer_idx)
        if layer_idx < self.mlpm.get_first_k_dense_replace():
            tensor_mlp_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=[])
        else:
            tensor_mlp_names = []
        tensor_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_idx)
        tensor_index_names = tensor_al1_names + tensor_ln1_names + tensor_gate_names + tensor_shared_expert_names + tensor_mlp_names
        
        # 使用 SLLMTM 异步提交加载任务
        self.sllmtm.submit_load(
            layer_idx=layer_idx,
            tensor_index_names=tensor_index_names,
            device_index_int=device_idx_int,
            cmv=self
        )
        cuda_hook_time_end(f"start_load_qkvogn_s_weight_l_{layer_idx}")
        
    def wait_load_qkvogn_s_weight(self, layer_idx: int):
        if layer_idx >= self.mlpm.config.num_hidden_layers:
            return
        self.sllmtm.get_result_wait()
    def async_load_experts_decode_cpu_weight_multi_device(self):
        """
        多设备版本：从最后一层往前加载 CPU 上的 expert weights
        串行提交到多个GPU设备，支持多GPU
        """
        if self.mlpm_ci is None:
            logger.warning("mlpm_ci is None, cannot check expert device distribution. Loading all experts.")
            # 如果模型未初始化，按原逻辑加载所有 expert 到所有设备
            num_device = len(self.device_list)
            for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, -1, -1):
                for device_idx in range(num_device):
                    device = self.device_list[device_idx]
                    self.start_load_experts_decode_cpu_weight(layer_idx=layer_idx, device=device, expert_idx_list=[])
            return
        
        # Step 1: 先获取所有层的 expert 分布情况
        layer_cpu_experts_map = {}  # {layer_idx: [expert_id, ...]}
        logger.debug("Collecting expert device distribution for all layers (multi-device)...")
        
        for layer_idx in range(0, self.mlpm.config.num_hidden_layers):
            # 跳过dense 层
            if layer_idx < self.mlpm.get_first_k_dense_replace():
                continue
            # 获取该层的 expert 设备分布
            layer = self.mlpm_ci.model.layers[layer_idx]
            expert_device_map = get_expert_device_distribution(layer)
            
            # 筛选出 CPU 上的 expert（只加载明确在 CPU 上的 expert）
            cpu_expert_list = []
            for expert_id, device in expert_device_map.items():
                # 只加载明确在 CPU 上的 expert
                # 'cuda:X' 表示在 GPU 上，不需要加载
                # 'meta' 表示未初始化，需要加载
                # 'unknown' 表示未知设备，需要加载
                # 'cpu' 表示在 CPU 上，需要加载
                if device == 'meta' or device == 'unknown' or device == 'cpu':
                    cpu_expert_list.append(expert_id)
                else:
                    # 如果expert已经在GPU上，记录日志但不加载
                    logger.debug(f"Layer {layer_idx} Expert {expert_id} already on {device}, skipping")
            
            layer_cpu_experts_map[layer_idx] = cpu_expert_list
            logger.debug(f"Layer {layer_idx}: CPU experts = {cpu_expert_list} (total: {len(cpu_expert_list)}, device_map: {expert_device_map})")
        
        # Step 1.5: 检查多GPU显存并计算所需显存，如果不足则报错，放得下则按剩余显存比例分配
        num_device = len(self.device_list)
        
        # 计算所有 CPU expert 的总显存需求
        from utils.helper import calculate_expert_memory_size
        total_required_memory = 0
        expert_memory_map = {}  # {(layer_idx, expert_idx): memory_size}
        layer_memory_map = {}  # {layer_idx: total_memory_for_layer}
        
        for layer_idx, expert_list in layer_cpu_experts_map.items():
            layer_total = 0
            for expert_idx in expert_list:
                memory_size = calculate_expert_memory_size(
                    self.mlpm, self.tensor_index_resize_json, layer_idx, expert_idx
                )
                expert_memory_map[(layer_idx, expert_idx)] = memory_size
                layer_total += memory_size
            layer_memory_map[layer_idx] = layer_total
            total_required_memory += layer_total
        
        logger.debug(f"Total required memory for all CPU experts: {total_required_memory / (1024**3):.2f} GB")
        
        # 检查所有设备的可用显存总和
        import pynvml
        pynvml.nvmlInit()
        
        total_available_memory = 0
        device_free_memory = {}  # {device_idx: free_memory}
        
        for device_idx in range(num_device):
            device = self.device_list[device_idx]
            device_idx_int = int(device.split(":")[1])
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx_int)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory = memory_info.free  # 空闲显存(字节)

                #==========================================
                if self.mlpm.model_name_type == QWEN3_MODEL_NAME_TYPE:
                    if num_device <= 2:
                        redisdent_memory = 3
                    else:
                        redisdent_memory = 1
                else: 
                    redisdent_memory = 0
                    
                device_free_memory[device_idx] = free_memory - redisdent_memory * 1024**3
                total_available_memory += free_memory
                
                logger.debug(f"GPU {device} (device {device_idx_int}) memory status:")
                logger.debug(f"  Total: {memory_info.total / (1024**3):.2f} GB")
                logger.debug(f"  Used: {memory_info.used / (1024**3):.2f} GB")
                logger.debug(f"  Free: {free_memory / (1024**3):.2f} GB")
            except Exception as e:
                raise ValueError(f"Failed to get memory info for device {device}: {e}")
        
        logger.debug(f"Total available memory across all devices: {total_available_memory / (1024**3):.2f} GB")
        logger.debug(f"Total required memory: {total_required_memory / (1024**3):.2f} GB")
        
        # 检查是否能放下，如果不足则警告并尽量分配
        memory_insufficient = total_required_memory > total_available_memory
        if memory_insufficient:
            logger.warning(
                f"Insufficient GPU memory across all devices! "
                f"Required: {total_required_memory / (1024**3):.2f} GB, "
                f"Available: {total_available_memory / (1024**3):.2f} GB. "
                f"Will try to allocate as many experts as possible to GPU, remaining experts will stay on CPU."
            )
        else:
            logger.debug("GPU memory is sufficient, will distribute experts based on available memory per device.")
        
        # 为每个设备分配expert（根据剩余显存空间按比例分配）
        layer_experts_map_by_device = {device_idx: {} for device_idx in range(num_device)}  # {device_idx: {layer_idx: [expert_id, ...]}}
        
        # 记录保留在CPU的expert（无法放入GPU的）
        layer_cpu_experts_stay_on_cpu = {}  # {layer_idx: [expert_id, ...]}
        
        # 维护每个设备的剩余显存（动态更新）
        device_remaining_memory = device_free_memory.copy()
        
        for layer_idx, expert_list in layer_cpu_experts_map.items():
            if not expert_list:
                continue
            
            # 计算该层每个expert的内存大小，并按内存大小排序（从大到小，优先分配大expert）
            expert_memory_list = [
                (expert_idx, expert_memory_map[(layer_idx, expert_idx)])
                for expert_idx in expert_list
            ]
            expert_memory_list.sort(key=lambda x: x[1], reverse=True)  # 按内存大小降序排序
            
            # 计算该层总内存需求
            layer_total_memory = layer_memory_map[layer_idx]
            
            # 计算每个设备应该分配的内存比例（基于剩余显存）
            total_remaining_memory = sum(device_remaining_memory.values())
            if total_remaining_memory == 0:
                # 如果所有设备都没有剩余显存，回退到平均分配
                logger.warning(f"Layer {layer_idx}: All devices have no remaining memory, falling back to even distribution.")
                experts_per_device = len(expert_list) // num_device
                remaining_experts = len(expert_list) % num_device
                expert_idx = 0
                for device_idx in range(num_device):
                    count = experts_per_device + (1 if device_idx < remaining_experts else 0)
                    if count > 0:
                        device_experts = expert_list[expert_idx:expert_idx + count]
                        layer_experts_map_by_device[device_idx][layer_idx] = device_experts
                        # 更新剩余显存（即使为0也要记录）
                        allocated_memory = sum(
                            expert_memory_map[(layer_idx, e_idx)] 
                            for e_idx in device_experts
                        )
                        device_remaining_memory[device_idx] -= allocated_memory
                        expert_idx += count
                        logger.debug(f"  Device {device_idx} ({self.device_list[device_idx]}): {len(device_experts)} experts (fallback)")
                continue
            
            # 计算每个设备的目标内存分配量（按剩余显存比例）
            device_target_memory = {}
            for device_idx in range(num_device):
                if total_remaining_memory > 0:
                    # 按剩余显存比例分配该层的内存
                    target_memory = (device_remaining_memory[device_idx] / total_remaining_memory) * layer_total_memory
                    device_target_memory[device_idx] = target_memory
                else:
                    device_target_memory[device_idx] = 0
            
            # 使用贪心算法分配expert到各个设备
            device_allocated_memory = {device_idx: 0 for device_idx in range(num_device)}
            device_experts_allocated = {device_idx: [] for device_idx in range(num_device)}
            
            for expert_idx, expert_memory in expert_memory_list:
                # 找到最适合的设备（剩余显存最多且未超过目标分配量）
                best_device = None
                best_score = -1
                
                for device_idx in range(num_device):
                    # 检查该设备是否有足够剩余显存
                    if device_remaining_memory[device_idx] < expert_memory:
                        continue
                    
                    # 计算得分：优先选择剩余显存多且未超过目标分配量的设备
                    # 如果已超过目标，得分会降低
                    current_allocated = device_allocated_memory[device_idx]
                    target = device_target_memory[device_idx]
                    
                    if current_allocated < target:
                        # 未超过目标，优先选择剩余显存多的
                        score = device_remaining_memory[device_idx]
                    else:
                        # 已超过目标，降低优先级（但仍允许分配，只要不超过剩余显存）
                        score = device_remaining_memory[device_idx] * 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_device = device_idx
                
                if best_device is not None:
                    # 分配到最佳设备
                    device_experts_allocated[best_device].append(expert_idx)
                    device_allocated_memory[best_device] += expert_memory
                    device_remaining_memory[best_device] -= expert_memory
                else:
                    # 如果所有设备都没有足够显存，尝试强制分配到剩余显存最多的设备
                    best_device = max(range(num_device), key=lambda idx: device_remaining_memory[idx])
                    if device_remaining_memory[best_device] >= expert_memory:
                        # 如果剩余显存最多的设备能放下，就分配
                        device_experts_allocated[best_device].append(expert_idx)
                        device_allocated_memory[best_device] += expert_memory
                        device_remaining_memory[best_device] -= expert_memory
                        logger.debug(
                            f"Layer {layer_idx} Expert {expert_idx}: Allocated to device {best_device} "
                            f"({expert_memory / (1024**3):.2f} GB) despite tight memory."
                        )
                    else:
                        # 无法放入任何GPU，保留在CPU
                        if layer_idx not in layer_cpu_experts_stay_on_cpu:
                            layer_cpu_experts_stay_on_cpu[layer_idx] = []
                        layer_cpu_experts_stay_on_cpu[layer_idx].append(expert_idx)
                        logger.debug(
                            f"Layer {layer_idx} Expert {expert_idx}: Cannot allocate {expert_memory / (1024**3):.2f} GB to any GPU. "
                            f"Device {best_device} only has {device_remaining_memory[best_device] / (1024**3):.2f} GB remaining. "
                            f"Expert will stay on CPU."
                        )
            
            # 保存分配结果
            for device_idx in range(num_device):
                if device_experts_allocated[device_idx]:
                    layer_experts_map_by_device[device_idx][layer_idx] = device_experts_allocated[device_idx]
                    allocated_memory_gb = device_allocated_memory[device_idx] / (1024**3)
                    remaining_memory_gb = device_remaining_memory[device_idx] / (1024**3)
                    logger.debug(
                        f"  Device {device_idx} ({self.device_list[device_idx]}): "
                        f"{len(device_experts_allocated[device_idx])} experts, "
                        f"allocated {allocated_memory_gb:.2f} GB, "
                        f"remaining {remaining_memory_gb:.2f} GB"
                    )
        
        # 保存分配结果，供等待时使用
        self._layer_experts_map_by_device = layer_experts_map_by_device
        
        # 统计并记录保留在CPU的expert
        if layer_cpu_experts_stay_on_cpu:
            total_cpu_experts = sum(len(expert_list) for expert_list in layer_cpu_experts_stay_on_cpu.values())
            total_cpu_memory = sum(
                expert_memory_map[(layer_idx, expert_idx)]
                for layer_idx, expert_list in layer_cpu_experts_stay_on_cpu.items()
                for expert_idx in expert_list
            )
            logger.warning(
                f"Total {total_cpu_experts} experts ({total_cpu_memory / (1024**3):.2f} GB) will stay on CPU "
                f"due to insufficient GPU memory. Details: {layer_cpu_experts_stay_on_cpu}"
            )
            # 保存保留在CPU的expert信息，供后续使用
            self._layer_cpu_experts_stay_on_cpu = layer_cpu_experts_stay_on_cpu
        else:
            self._layer_cpu_experts_stay_on_cpu = {}
        
        # 验证分配结果：检查是否有重复分配
        all_experts_allocated = {}  # {(layer_idx, expert_idx): [device_idx, ...]}
        for device_idx, device_layer_map in layer_experts_map_by_device.items():
            for layer_idx, expert_list in device_layer_map.items():
                for expert_idx in expert_list:
                    key = (layer_idx, expert_idx)
                    if key not in all_experts_allocated:
                        all_experts_allocated[key] = []
                    all_experts_allocated[key].append(device_idx)
        
        # 检查重复分配
        duplicate_allocations = {k: v for k, v in all_experts_allocated.items() if len(v) > 1}
        if duplicate_allocations:
            logger.warning(f"发现重复分配的experts: {duplicate_allocations}")
            raise RuntimeError(f"检测到重复分配：某些experts被分配到多个设备上！重复分配详情: {duplicate_allocations}")
        
        # Step 2: 从最后一层往前逐层加载，串行提交到多个GPU设备（基于剩余显存比例分配）
        logger.debug("Starting to load CPU experts from last layer to first layer (multi-device serial mode, memory-proportional distribution)...")
        
        rend_layer_idx = self.mlpm.get_first_k_dense_replace() - 1
        for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, rend_layer_idx, -1):
            # 检查是否有任何设备需要加载这一层
            has_experts = False
            for device_idx in range(num_device):
                if layer_idx in layer_experts_map_by_device[device_idx]:
                    has_experts = True
                    break
            
            if has_experts:
                logger.debug(f"Loading Layer {layer_idx} across {num_device} devices...")
                
                # 串行提交到每个GPU设备（每个设备加载分配给它的expert）
                for device_idx in range(num_device):
                    device = self.device_list[device_idx]
                    device_expert_list = layer_experts_map_by_device[device_idx].get(layer_idx, [])
                    
                    if device_expert_list:
                        logger.debug(f"  Device {device_idx} ({device}): {len(device_expert_list)} experts: {device_expert_list}")
                        self.start_load_experts_decode_cpu_weight(
                            layer_idx=layer_idx,
                            device=device,
                            expert_idx_list=device_expert_list
                        )
            else:
                logger.debug(f"Layer {layer_idx}: No CPU experts to load, skipping.")
                # 即使没有 CPU expert 需要加载，也标记为已加载
                self._layer_loaded_to_gpu[layer_idx] = True
    
    def async_wait_layer_loaded_to_gpu_multi_device(self):
        """
        多设备版本：等待所有层的CPU专家加载完成
        串行等待每个设备的结果，只等待实际提交了任务的设备
        """
        rend_layer_idx = self.mlpm.get_first_k_dense_replace() - 1
        num_device = len(self.device_list)
        
        # 获取分配结果，确保只等待实际提交了任务的设备
        layer_experts_map_by_device = getattr(self, '_layer_experts_map_by_device', {})
        
        # 串行等待每个层的加载完成（每个层可能有多个设备的任务）
        for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, rend_layer_idx, -1):
            # 只等待实际提交了任务的设备
            # 统计该层有多少个设备提交了任务
            tasks_submitted = 0
            for device_idx in range(num_device):
                # 检查该设备在该层是否有expert需要等待
                if layer_idx in layer_experts_map_by_device.get(device_idx, {}):
                    device_expert_list = layer_experts_map_by_device[device_idx][layer_idx]
                    if device_expert_list:  # 确保列表不为空
                        tasks_submitted += 1
            
            # 等待该层所有提交的任务完成（每个设备一个任务）
            for _ in range(tasks_submitted):
                self.wait_load_experts_decode_cpu_weight(layer_idx=layer_idx)

    def async_load_experts_decode_cpu_weight(self):
        """
        从最后一层往前加载 CPU 上的 expert weights
        先获取所有层的 expert 分布情况，然后反向逐层加载
        """
        if self.mlpm_ci is None:
            logger.warning("mlpm_ci is None, cannot check expert device distribution. Loading all experts.")
            # 如果模型未初始化，按原逻辑加载所有 expert
            for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, -1, -1):
                self.start_load_experts_decode_cpu_weight(layer_idx=layer_idx, device=self.device1, expert_idx_list=[])
            return
        
        # Step 1: 先获取所有层的 expert 分布情况
        layer_cpu_experts_map = {}  # {layer_idx: [expert_id, ...]}
        logger.debug("Collecting expert device distribution for all layers...")
        

        for layer_idx in range(0, self.mlpm.config.num_hidden_layers):
            # 跳过dense 层
            if layer_idx < self.mlpm.get_first_k_dense_replace():
                continue
            # 获取该层的 expert 设备分布
            layer = self.mlpm_ci.model.layers[layer_idx]
            expert_device_map = get_expert_device_distribution(layer)
            
            # 筛选出 CPU 上的 expert（只加载明确在 CPU 上的 expert）
            cpu_expert_list = []
            for expert_id, device in expert_device_map.items():
                # 只加载明确在 CPU 上的 expert
                # 'cuda:X' 表示在 GPU 上，不需要加载
                # 'meta' 表示未初始化，不需要加载
                # 'unknown' 表示未知设备，不加载
                if device == 'meta' or device == 'unknown':
                    cpu_expert_list.append(expert_id)
            
            layer_cpu_experts_map[layer_idx] = cpu_expert_list
            logger.debug(f"Layer {layer_idx}: CPU experts = {cpu_expert_list} (total: {len(cpu_expert_list)}, device_map: {expert_device_map})")
        
        # Step 1.5: 检查 GPU 显存并计算所需显存，如果不足则按比例选择 expert
        layer_cpu_experts_map = filter_experts_by_memory(
            mlpm=self.mlpm,
            tensor_index_resize_json=self.tensor_index_resize_json,
            config=self.mlpm.config,
            device1=self.device1,
            layer_cpu_experts_map=layer_cpu_experts_map
        )
        
        self._layer_cpu_experts_need_load_map = layer_cpu_experts_map
        # Step 2: 从最后一层往前逐层加载
        logger.debug("Starting to load CPU experts from last layer to first layer...")
        rend_layer_idx = self.mlpm.get_first_k_dense_replace() - 1
        for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, rend_layer_idx, -1):
            if layer_idx not in layer_cpu_experts_map:
                continue
            cpu_expert_list = layer_cpu_experts_map[layer_idx]
            
            
            # 只加载有 CPU expert 的层
            if cpu_expert_list:
                logger.debug(f"Loading Layer {layer_idx}: {len(cpu_expert_list)} CPU experts: {cpu_expert_list}")
                self.start_load_experts_decode_cpu_weight(
                    layer_idx=layer_idx, 
                    device=self.device1_str, 
                    expert_idx_list=cpu_expert_list
                )
                # self.wait_load_experts_decode_cpu_weight(layer_idx=layer_idx)
                # 标记该层参数已全部加载到 GPU
                # self._layer_loaded_to_gpu[layer_idx] = True

                # logger.debug(f"Layer {layer_idx}: All parameters loaded to GPU, marked as complete.")
                
            else:
                logger.debug(f"Layer {layer_idx}: No CPU experts to load, skipping.")
                # 即使没有 CPU expert 需要加载，也标记为已加载（可能该层所有 expert 都在 GPU 上）
                self._layer_loaded_to_gpu[layer_idx] = True
                
    def async_wait_layer_loaded_to_gpu(self):
        rend_layer_idx = self.mlpm.get_first_k_dense_replace() - 1
        for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, rend_layer_idx, -1):
            # 只等待实际提交了任务的层（如果层已经标记为已加载，则跳过）
            if layer_idx not in self._layer_cpu_experts_need_load_map:
                continue
            self.wait_load_experts_decode_cpu_weight(layer_idx=layer_idx)

    # part load here
    def check_async_load_experts_decode_cpu_weight(self, layer_idx: int):
        if layer_idx >= self.mlpm.config.num_hidden_layers:
            raise ValueError(f"layer_idx must be less than {self.mlpm.config.num_hidden_layers}")
        if self._layer_loaded_to_gpu.get(layer_idx, False):
            return True
        return False

    def start_load_experts_decode_cpu_weight(self, layer_idx: int, device: str, expert_idx_list: list[int]):
        if layer_idx >= self.mlpm.config.num_hidden_layers:
            return
        device_idx_int = int(device.split(":")[1])

        # notify and set
        def set_label_layer_loaded_to_gpu(layer_idx: int):
            self._layer_loaded_to_gpu[layer_idx] = True

        cuda_hook_time(f"start_load_experts_decode_cpu_weight_l_{layer_idx}")
        tensor_index_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=expert_idx_list)
        self.sllmtm.submit_load(
            layer_idx=layer_idx,
            tensor_index_names=tensor_index_names,
            device_index_int=device_idx_int,
            cmv=self,
            set_label_func=set_label_layer_loaded_to_gpu
        )
        cuda_hook_time_end(f"start_load_experts_decode_cpu_weight_l_{layer_idx}")

    def wait_load_experts_decode_cpu_weight(self, layer_idx: int):
        if layer_idx >= self.mlpm.config.num_hidden_layers:
            return
        self.sllmtm.get_result_wait()

    def load_qkvgon_weight_onetime(self):
        device1_idx_int = self.device1
        time_start_init_qkvogn_weight = time.time()
        cuda_hook_time("init qkvogn weight one time")
        layer_num = self.mlpm.config.num_hidden_layers
        tensor_names = []
        for layer_idx in range(layer_num):
            tensor_al1_names = self.mlpm.get_attention_names(layer_idx=layer_idx)
            tensor_ln1_names = self.mlpm.get_layernorm_names(layer_idx=layer_idx)
            tensor_gate_names = self.mlpm.get_gate_names(layer_idx=layer_idx)
            tensor_index_names = tensor_al1_names + tensor_ln1_names + tensor_gate_names
            tensor_names = tensor_names + tensor_index_names
        self.allocate_cuda_memory_load_wait(tensor_names, device_index_int=device1_idx_int)
        cuda_hook_time_end("init qkvogn weight one time")

    def load_all_qkvogn_weight(self):
        device1_idx_int = self.device1
        time_start_init_qkvogn_weight = time.time()
        cuda_hook("init qkvogn weight")
        layer_num = self.mlpm.config.num_hidden_layers
        for layer_idx in range(layer_num):
            tensor_al1_names = self.mlpm.get_attention_names(layer_idx=layer_idx)
            tensor_ln1_names = self.mlpm.get_layernorm_names(layer_idx=layer_idx)
            tensor_gate_names = self.mlpm.get_gate_names(layer_idx=layer_idx)
            tensor_index_names = tensor_al1_names + tensor_ln1_names + tensor_gate_names
            self.allocate_cuda_memory_load_wait(tensor_index_names, device_index_int=device1_idx_int)
        logger.debug(f"init qkvogn weight time: {time.time() - time_start_init_qkvogn_weight}")
        cuda_hook_end("init qkvogn weight")

    

    def init_load_qkvogn_es_weight(self, layer_idx: int = 0):
        layer_idx = layer_idx
        # if layer_idx != 0:
        #     raise ValueError(f"layer_idx must be 0")
        device1_idx_int = self.device1
        cuda_hook_time(f"load_qkvogns_weight_l_{layer_idx}")
        tensor_al1_names = self.mlpm.get_attention_names(layer_idx=layer_idx)
        tensor_ln1_names = self.mlpm.get_layernorm_names(layer_idx=layer_idx)
        tensor_gate_names = self.mlpm.get_gate_names(layer_idx=layer_idx)
        if layer_idx < self.mlpm.get_first_k_dense_replace():
            tensor_mlp_names = []
        else:
            tensor_mlp_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=[])
        tensor_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_idx)
        tensor_index_names = tensor_al1_names + tensor_ln1_names + tensor_gate_names + tensor_shared_expert_names + tensor_mlp_names
        self.allocate_cuda_memory_load_wait(tensor_index_names, device_index_int=device1_idx_int)
        cuda_hook_time_end(f"load_qkvogns_weight_l_{layer_idx}")

    def allocate_cuda_memory(self, tensor_index_names: list[str], device_index_int: int):
        tensor_meta_index, tensor_data_index, tensor_device_offsets, tensor_copy_chunks, tensor_device_size = \
            self.get_meta_data_offsets_and_copy_chunks(tensor_index_names, device_index_int)
        device_memory = {
            device_index_int: tensor_device_size
        }
        cuda_memory_ptrs = allocate_cuda_memory(device_memory)
        self.cuda_memory_ptrs_allocated.append(cuda_memory_ptrs)
        return cuda_memory_ptrs

    def wait_load_into_gpu(self, replica_uuid: str):
        self.client.confirm_model_loaded(self.mlpm.model_path, replica_uuid)

    

    def prepare_cuda_memory_fused_experts(self, layer_idx: int, gpu_expert_ids_by_device: dict[int, set[int]]):
        fused_tensor_index_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=[])
        """
        为 fused experts（如 Gemma4/Qwen3/GPT-OSS 的 ``gate_up_proj`` + ``down_proj``）按设备分配显存，
        并生成按 expert 行切片的 ``tensor_device_offsets`` / ``tensor_copy_chunks``，供后续异步拷贝使用。

        设计目标：
        - 源端共享内存里 `gate_up_proj`/`down_proj` 是整表大张量（第一维为 num_experts）。
        - 这里按 ``gpu_expert_ids_by_device`` 只拷贝每个 device 需要的 expert 行（每行对应连续 byte range）。
        - 在目标 GPU 上将这些行按 device 的 expert 顺序紧凑打包成新的「局部连续表」，后续可直接做 fused BMM/GMM。

        Returns:
            (cuda_memory_ptrs_device_map, tensor_meta_index, tensor_device_offsets_device_map, tensor_copy_chunks_device_map)
            其中 `tensor_meta_index`/`tensor_device_offsets_device_map` 可直接喂给 `restore_tensors2`。
        """
        if not fused_tensor_index_names:
            raise ValueError("fused_tensor_index_names is empty; cannot allocate fused experts")

        # Identify fused weight names in tensor index.
        gate_up_name = next((n for n in fused_tensor_index_names if "gate_up_proj" in n), None)
        down_name = next((n for n in fused_tensor_index_names if "down_proj" in n), None)
        if gate_up_name is None or down_name is None:
            raise ValueError(f"Cannot find gate_up/down in fused_tensor_index_names: {fused_tensor_index_names}")
        if gate_up_name not in self.tensor_index_resize_json or down_name not in self.tensor_index_resize_json:
            raise KeyError(
                f"Fused tensor names not in tensor_index_resize_json: gate_up={gate_up_name in self.tensor_index_resize_json}, "
                f"down={down_name in self.tensor_index_resize_json}"
            )

        gate_up_offset, gate_up_size, gate_up_shape, gate_up_stride, gate_up_dtype = self.tensor_index_resize_json[gate_up_name]
        down_offset, down_size, down_shape, down_stride, down_dtype = self.tensor_index_resize_json[down_name]

        # Expect 3D layout: [E, ...]
        if len(gate_up_shape) < 3 or len(down_shape) < 3:
            raise ValueError(f"Expected fused weights to be 3D with expert dim: gate_up_shape={gate_up_shape}, down_shape={down_shape}")
        num_experts_total = int(gate_up_shape[0])
        if int(down_shape[0]) != num_experts_total:
            raise ValueError(f"gate_up/down expert dim mismatch: {gate_up_shape[0]} vs {down_shape[0]}")
        if num_experts_total <= 0:
            raise ValueError(f"Invalid num_experts_total={num_experts_total}")

        # Bytes per expert row (assume contiguous by expert in storage).
        if gate_up_size % num_experts_total != 0 or down_size % num_experts_total != 0:
            raise ValueError(
                f"Fused tensor size not divisible by num_experts: gate_up_size={gate_up_size}, down_size={down_size}, E={num_experts_total}"
            )
        gate_up_row_bytes = int(gate_up_size // num_experts_total)
        down_row_bytes = int(down_size // num_experts_total)

        # Slice meta (2D) for each expert row. Stride here is in elements; we set standard contiguous 2D strides.
        gate_up_slice_shape = tuple(gate_up_shape[1:])
        down_slice_shape = tuple(down_shape[1:])
        gate_up_slice_stride = (int(gate_up_slice_shape[1]), 1) if len(gate_up_slice_shape) == 2 else tuple(gate_up_stride[1:])
        down_slice_stride = (int(down_slice_shape[1]), 1) if len(down_slice_shape) == 2 else tuple(down_stride[1:])

        tensor_meta_index: dict[str, tuple] = {}
        tensor_device_offsets_device_map: dict[int, dict[str, int]] = {}
        tensor_copy_chunks_device_map: dict[int, list[tuple[int, int, int, int]]] = {}
        tensor_device_size_device_map: dict[int, int] = {}

        # Helpful for callers: map device -> list of slice names in packed order.
        tensor_slice_names_by_device: dict[int, dict[str, list[str]]] = {}

        for device_index_int, expert_ids in gpu_expert_ids_by_device.items():
            expert_list = sorted(int(e) for e in expert_ids)
            if not expert_list:
                continue

            # Build per-device "virtual tensors" (one per expert row) so restore_tensors2 yields packed [E_dev, ...] slices.
            tensor_data_index_device: dict[str, tuple[int, int]] = {}
            gate_up_slice_names: list[str] = []
            down_slice_names: list[str] = []

            for eid in expert_list:
                if eid < 0 or eid >= num_experts_total:
                    raise ValueError(f"Expert id out of range: {eid} (E={num_experts_total})")

                n_gu = f"{gate_up_name}.expert_{eid}"
                n_dn = f"{down_name}.expert_{eid}"

                tensor_meta_index[n_gu] = (gate_up_slice_shape, gate_up_slice_stride, gate_up_dtype)
                tensor_meta_index[n_dn] = (down_slice_shape, down_slice_stride, down_dtype)

                tensor_data_index_device[n_gu] = (int(gate_up_offset + eid * gate_up_row_bytes), gate_up_row_bytes)
                tensor_data_index_device[n_dn] = (int(down_offset + eid * down_row_bytes), down_row_bytes)

                gate_up_slice_names.append(n_gu)
                down_slice_names.append(n_dn)

            tensor_slice_names_by_device[device_index_int] = {
                "gate_up": gate_up_slice_names,
                "down": down_slice_names,
                "experts": [str(e) for e in expert_list],
            }

            tensor_device_offsets, tensor_copy_chunks, tensor_device_size = calculate_device_offset(
                tensor_index=tensor_data_index_device, device_idx=device_index_int
            )
            tensor_device_offsets_device_map.update(tensor_device_offsets)
            tensor_copy_chunks_device_map.update(tensor_copy_chunks)
            tensor_device_size_device_map[device_index_int] = tensor_device_size

        if not tensor_device_size_device_map:
            raise ValueError("No experts to allocate (gpu_expert_ids_by_device empty)")

        device_memory = {
            device_index_int: tensor_device_size
            for device_index_int, tensor_device_size in tensor_device_size_device_map.items()
        }
        cuda_memory_ptrs_device_map = allocate_cuda_memory(device_memory)
        self.cuda_memory_ptrs_allocated.append(cuda_memory_ptrs_device_map)

        logger.debug(
            f"allocate_cuda_memory_fused_experts layer={layer_idx} "
            f"devices={list(tensor_device_size_device_map.keys())} "
            f"gate_up={gate_up_name} down={down_name}"
        )

        # Caller can use `tensor_slice_names_by_device` via this attribute if needed.
        self._layer_experts_map_by_device[layer_idx] = tensor_slice_names_by_device

        return (
            cuda_memory_ptrs_device_map,
            tensor_meta_index,
            tensor_device_offsets_device_map,
            tensor_copy_chunks_device_map,
        )
    def allocate_cuda_memory_fused_experts(self, layer_idx: int, gpu_expert_ids_by_device: dict[int, set[int]]):
        """
        一站式：为 fused experts 准备 offsets/chunks，调用 ``load_into_gpu_async``，并用 ``restore_tensors2``
        返回本次加载得到的 `state_dict`（条目 tensor 位于对应 device 上）。

        Returns:
            (ret, replica_uuid, state_dict)
        """
        (
            cuda_memory_ptrs_device_map,
            tensor_meta_index,
            tensor_device_offsets_device_map,
            tensor_copy_chunks_device_map,
        ) = self.prepare_cuda_memory_fused_experts(layer_idx, gpu_expert_ids_by_device)

        cuda_memory_handles_device_map = get_cuda_memory_handles(cuda_memory_ptrs_device_map)

        cuda_hook_time("load_into_gpu_async_fused_experts")
        ret, replica_uuid = load_into_gpu_async(
            client=self.client,
            device_uuid_map=self.device_uuid_map,
            model_path=self.mlpm.model_path,
            tensor_copy_chunks=tensor_copy_chunks_device_map,
            cuda_memory_handles=cuda_memory_handles_device_map,
        )
        cuda_hook_time_end("load_into_gpu_async_fused_experts")

        cuda_hook_time("restore_tensors2_fused_experts")
        state_dict = restore_tensors2(
            tensor_meta_index, cuda_memory_ptrs_device_map, tensor_device_offsets_device_map
        )
        cuda_hook_time_end("restore_tensors2_fused_experts")
        return ret, replica_uuid, state_dict

    def allocate_cuda_memory_fused_experts_dual_restore(
        self, layer_idx: int, gpu_expert_ids_by_device: dict[int, set[int]]
    ):
        """
        一次 load_into_gpu_async，复用同一份 cuda memory view，做两次 restore：

        - `state_dict_packed`: 每个 device 两个连续大张量（gate_up_packed: [E_dev,2I,H], down_packed: [E_dev,H,I]）
        - `state_dict_slices`: 每个 expert 一个 2D 张量（gate_up_name.expert_<eid>, down_name.expert_<eid>）

        要点：目标 GPU 内存布局需保证 gate_up 行连续、down 行连续，因此这里按顺序写入：
        先写 gate_up 的所有 expert 行，再写 down 的所有 expert 行。

        Returns:
            (ret, replica_uuid, state_dict_packed, state_dict_slices)
        """
        fused_tensor_index_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=[])
        if not fused_tensor_index_names:
            raise ValueError("fused_tensor_index_names is empty; cannot allocate fused experts")

        gate_up_name = next((n for n in fused_tensor_index_names if "gate_up_proj" in n), None)
        down_name = next((n for n in fused_tensor_index_names if "down_proj" in n), None)
        if gate_up_name is None or down_name is None:
            raise ValueError(f"Cannot find gate_up/down in fused_tensor_index_names: {fused_tensor_index_names}")
        if gate_up_name not in self.tensor_index_resize_json or down_name not in self.tensor_index_resize_json:
            raise KeyError("Fused tensor names not in tensor_index_resize_json")

        gate_up_offset, gate_up_size, gate_up_shape, gate_up_stride, gate_up_dtype = self.tensor_index_resize_json[gate_up_name]
        down_offset, down_size, down_shape, down_stride, down_dtype = self.tensor_index_resize_json[down_name]
        if len(gate_up_shape) < 3 or len(down_shape) < 3:
            raise ValueError(f"Expected 3D fused weights: gate_up_shape={gate_up_shape}, down_shape={down_shape}")
        num_experts_total = int(gate_up_shape[0])
        if int(down_shape[0]) != num_experts_total:
            raise ValueError("gate_up/down expert dim mismatch")
        if gate_up_size % num_experts_total != 0 or down_size % num_experts_total != 0:
            raise ValueError("Fused tensor size not divisible by num_experts")
        gate_up_row_bytes = int(gate_up_size // num_experts_total)
        down_row_bytes = int(down_size // num_experts_total)

        gate_up_slice_shape = tuple(gate_up_shape[1:])
        down_slice_shape = tuple(down_shape[1:])
        gate_up_slice_stride = (int(gate_up_slice_shape[1]), 1) if len(gate_up_slice_shape) == 2 else tuple(gate_up_stride[1:])
        down_slice_stride = (int(down_slice_shape[1]), 1) if len(down_slice_shape) == 2 else tuple(down_stride[1:])

        # Shared allocation inputs
        tensor_meta_index_slices: dict[str, tuple] = {}
        tensor_device_offsets_slices: dict[int, dict[str, int]] = {}
        tensor_copy_chunks_device_map: dict[int, list[tuple[int, int, int, int]]] = {}
        tensor_device_size_device_map: dict[int, int] = {}

        # Packed views (no additional allocation/copy)
        tensor_meta_index_packed: dict[str, tuple] = {}
        tensor_device_offsets_packed: dict[int, dict[str, int]] = {}

        tensor_slice_names_by_device: dict[int, dict[str, list[str]]] = {}

        for device_index_int, expert_ids in gpu_expert_ids_by_device.items():
            expert_list = sorted(int(e) for e in expert_ids)
            if not expert_list:
                continue

            e_dev = len(expert_list)
            gate_block_bytes = e_dev * gate_up_row_bytes
            down_block_bytes = e_dev * down_row_bytes
            device_size = gate_block_bytes + down_block_bytes
            tensor_device_size_device_map[device_index_int] = device_size

            # Build slice tensors in the same memory layout (gate rows first, then down rows)
            device_offsets: dict[str, int] = {}
            copy_chunks: list[tuple[int, int, int, int]] = []
            gate_slice_names: list[str] = []
            down_slice_names: list[str] = []

            # Gate slices occupy [0, gate_block_bytes)
            for i, eid in enumerate(expert_list):
                if eid < 0 or eid >= num_experts_total:
                    raise ValueError(f"Expert id out of range: {eid} (E={num_experts_total})")
                n_gu = f"{gate_up_name}.expert_{eid}"
                tensor_meta_index_slices[n_gu] = (gate_up_slice_shape, gate_up_slice_stride, gate_up_dtype)
                dst_off = i * gate_up_row_bytes
                device_offsets[n_gu] = dst_off
                src_off = int(gate_up_offset + eid * gate_up_row_bytes)
                copy_chunks.append((src_off, gate_up_row_bytes, dst_off, 0))
                gate_slice_names.append(n_gu)

            # Down slices occupy [gate_block_bytes, gate_block_bytes+down_block_bytes)
            for i, eid in enumerate(expert_list):
                n_dn = f"{down_name}.expert_{eid}"
                tensor_meta_index_slices[n_dn] = (down_slice_shape, down_slice_stride, down_dtype)
                dst_off = gate_block_bytes + i * down_row_bytes
                device_offsets[n_dn] = dst_off
                src_off = int(down_offset + eid * down_row_bytes)
                copy_chunks.append((src_off, down_row_bytes, dst_off, 0))
                down_slice_names.append(n_dn)

            tensor_device_offsets_slices[device_index_int] = device_offsets
            tensor_copy_chunks_device_map[device_index_int] = copy_chunks

            # Packed meta: two 3D tensors pointing into same memory.
            gate_up_packed_name = f"{gate_up_name}.packed.dev_{device_index_int}"
            down_packed_name = f"{down_name}.packed.dev_{device_index_int}"
            gate_up_packed_shape = (e_dev, int(gate_up_shape[1]), int(gate_up_shape[2]))
            down_packed_shape = (e_dev, int(down_shape[1]), int(down_shape[2]))
            gate_up_packed_stride = (gate_up_packed_shape[1] * gate_up_packed_shape[2], gate_up_packed_shape[2], 1)
            down_packed_stride = (down_packed_shape[1] * down_packed_shape[2], down_packed_shape[2], 1)
            tensor_meta_index_packed[gate_up_packed_name] = (gate_up_packed_shape, gate_up_packed_stride, gate_up_dtype)
            tensor_meta_index_packed[down_packed_name] = (down_packed_shape, down_packed_stride, down_dtype)
            tensor_device_offsets_packed[device_index_int] = {
                gate_up_packed_name: 0,
                down_packed_name: gate_block_bytes,
            }

            tensor_slice_names_by_device[device_index_int] = {
                "gate_up": gate_slice_names,
                "down": down_slice_names,
                "gate_up_packed": [gate_up_packed_name],
                "down_packed": [down_packed_name],
                "experts": [str(e) for e in expert_list],
            }

        if not tensor_device_size_device_map:
            raise ValueError("No experts to allocate (gpu_expert_ids_by_device empty)")

        device_memory = {d: sz for d, sz in tensor_device_size_device_map.items()}
        cuda_memory_ptrs_device_map = allocate_cuda_memory(device_memory)
        self.cuda_memory_ptrs_allocated.append(cuda_memory_ptrs_device_map)
        cuda_memory_handles_device_map = get_cuda_memory_handles(cuda_memory_ptrs_device_map)

        # Save naming info for callers (incremental merge; do not overwrite other devices/layers).
        prev_layer_map = self._layer_experts_map_by_device.get(layer_idx)
        if isinstance(prev_layer_map, dict):
            # per-device dict merge; new entries override per-device keys (expected: newer mapping for same device)
            prev_layer_map.update(tensor_slice_names_by_device)
        else:
            self._layer_experts_map_by_device[layer_idx] = tensor_slice_names_by_device

        cuda_hook_time("load_into_gpu_async_fused_experts_dual")
        ret, replica_uuid = load_into_gpu_async(
            client=self.client,
            device_uuid_map=self.device_uuid_map,
            model_path=self.mlpm.model_path,
            tensor_copy_chunks=tensor_copy_chunks_device_map,
            cuda_memory_handles=cuda_memory_handles_device_map,
        )
        cuda_hook_time_end("load_into_gpu_async_fused_experts_dual")

        cuda_hook_time("restore_tensors2_fused_experts_packed")
        state_dict_packed = restore_tensors2(
            tensor_meta_index_packed, cuda_memory_ptrs_device_map, tensor_device_offsets_packed
        )
        cuda_hook_time_end("restore_tensors2_fused_experts_packed")

        cuda_hook_time("restore_tensors2_fused_experts_slices")
        state_dict_slices = restore_tensors2(
            tensor_meta_index_slices, cuda_memory_ptrs_device_map, tensor_device_offsets_slices
        )
        cuda_hook_time_end("restore_tensors2_fused_experts_slices")

        return ret, replica_uuid, state_dict_packed, state_dict_slices
    def allocate_cuda_memory_and_load_into_gpu_multi_device(
        self, 
        tensor_index_names_device_map: dict[int, list[str]]
    ):
        cuda_hook_time("allocate_cuda_memory_and_load_into_gpu_multi_device")
        tensor_meta_index = {}
        tensor_data_index = {}
        tensor_device_offsets_device_map = {}
        tensor_copy_chunks_device_map = {}
        tensor_device_size_device_map = {}
        for device_index_int, tensor_index_names in tensor_index_names_device_map.items():
            tensor_meta_index_device, tensor_data_index_device, tensor_device_offsets, tensor_copy_chunks, tensor_device_size = \
                self.get_meta_data_offsets_and_copy_chunks(tensor_index_names, device_index_int)
            tensor_meta_index.update(tensor_meta_index_device)
            tensor_data_index.update(tensor_data_index_device)
            tensor_device_offsets_device_map.update(tensor_device_offsets)
            tensor_copy_chunks_device_map.update(tensor_copy_chunks)
            tensor_device_size_device_map[device_index_int] = tensor_device_size
        
        device_memory = {
            device_index_int: tensor_device_size
            for device_index_int, tensor_device_size in tensor_device_size_device_map.items()
        }
        cuda_memory_ptrs_device_map = allocate_cuda_memory(device_memory)
        self.cuda_memory_ptrs_allocated.append(cuda_memory_ptrs_device_map)

        logger.debug(
            f"tensor_device_offsets_device_map {tensor_device_offsets_device_map}"
            f"tensor_copy_chunks_device_map {tensor_copy_chunks_device_map}"
            f"cuda_memory_handles_device_map {cuda_memory_ptrs_device_map}"
        )
        cuda_memory_handles_device_map = get_cuda_memory_handles(cuda_memory_ptrs_device_map)

        ret1, replica_uuid1 = load_into_gpu_async(
            client=self.client,
            device_uuid_map=self.device_uuid_map,
            model_path=self.mlpm.model_path,
            tensor_copy_chunks=tensor_copy_chunks_device_map,
            cuda_memory_handles=cuda_memory_handles_device_map,
        )
        state_dict = restore_tensors2(
            tensor_meta_index, cuda_memory_ptrs_device_map, tensor_device_offsets_device_map
        )
        cuda_hook_time_end("allocate_cuda_memory_and_load_into_gpu_multi_device")
        return ret1, replica_uuid1, state_dict
    def allocate_cuda_memory_and_load_into_gpu(self, tensor_index_names: list[str], device_index_int: int):
        cuda_hook_time("allocate_cuda_memory_and_load_into_gpu")
        tensor_meta_index, tensor_data_index, tensor_device_offsets, tensor_copy_chunks, tensor_device_size = \
            self.get_meta_data_offsets_and_copy_chunks(tensor_index_names, device_index_int)
        device_memory = {
            device_index_int: tensor_device_size
        }
        cuda_hook_time("allocate_cuda_memory")
        logger.debug(f"allocate cuda memory {device_memory}")
        cuda_memory_ptrs = allocate_cuda_memory(device_memory)
        cuda_hook_time_end("allocate_cuda_memory")
        self.cuda_memory_ptrs_allocated.append(cuda_memory_ptrs)
        cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
        cuda_hook_time("load_into_gpu_async")
        ret1, replica_uuid1 = load_into_gpu_async(
            client=self.client,
            device_uuid_map=self.device_uuid_map,
            model_path=self.mlpm.model_path,
            tensor_copy_chunks=tensor_copy_chunks,
            cuda_memory_handles=cuda_memory_handles,
        )
        cuda_hook_time_end("load_into_gpu_async")
        cuda_hook_time("restore_tensors2")
        state_dict = restore_tensors2(
            tensor_meta_index, cuda_memory_ptrs, tensor_device_offsets
        )
        cuda_hook_time_end("restore_tensors2")
        cuda_hook_time_end("allocate_cuda_memory_and_load_into_gpu")
        return ret1, replica_uuid1, state_dict
    def allocate_cuda_memory_load_wait(self, tensor_index_names: list[str], device_index_int: int):
        ret1, replica_uuid1, state_dict1 = \
            self.allocate_cuda_memory_and_load_into_gpu(tensor_index_names, device_index_int)
        self.wait_load_into_gpu(replica_uuid1)
        self.restore2model(state_dict1, self.mlpm_ci)
        return state_dict1
    def get_meta_data_offsets_and_copy_chunks(self, tensor_index_names: list[str], device_index_int: int):
        tensor_meta_index = {}
        tensor_data_index = {}
        for name in tensor_index_names:
            offset, size, shape, stride, dtype = self.tensor_index_resize_json[name]
            tensor_meta_index[name] = (shape, stride, dtype)
            tensor_data_index[name] = (offset, size)
        tensor_device_offsets, tensor_copy_chunks, tensor_device_size = \
            calculate_device_offset(tensor_index=tensor_data_index, device_idx=device_index_int)
        
        return tensor_meta_index, tensor_data_index, tensor_device_offsets, tensor_copy_chunks, tensor_device_size



    def test_init_onelayer_experts_weight(self, layer_idx: int = 0):
        cuda_hook_time("test_init_onelayer_experts_weight")
        if layer_idx >= self.mlpm.config.num_hidden_layers:
            return
        device1_idx_int = self.device1
        tensor_index_names = self.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=[])
        self.allocate_cuda_memory_load_wait(tensor_index_names, device_index_int=device1_idx_int)
        cuda_hook_time_end("test_init_onelayer_experts_weight")

from pathlib import Path
# 添加 kt-kernel 路径 - 使用固定路径 /mnt/zhengcf3/ktransformers/kt-kernel
from kt_kernel import kt_kernel_ext

def detect_numa_nodes() -> list[int]:
    node_root = Path("/sys/devices/system/node")
    nodes: list[int] = []
    if node_root.exists():
        for child in node_root.iterdir():
            name = child.name
            if name.startswith("node") and name[4:].isdigit():
                nodes.append(int(name[4:]))
    return sorted(nodes) or [0]


def default_thread_count(numa_nodes: list[int]) -> int:
    cpu_root = Path("/sys/devices/system/cpu")
    physical_cores: set[tuple[str, str]] = set()
    if cpu_root.exists():
        for cpu_dir in cpu_root.iterdir():
            name = cpu_dir.name
            if not name.startswith("cpu") or not name[3:].isdigit():
                continue
            package_file = cpu_dir / "topology" / "physical_package_id"
            core_file = cpu_dir / "topology" / "core_id"
            try:
                physical_cores.add((package_file.read_text().strip(), core_file.read_text().strip()))
            except OSError:
                pass
    if physical_cores:
        return len(physical_cores)
    return max(1, (os.cpu_count() or 1) // 2)


def make_worker_config(ext, threads: int, numa_nodes: list[int]):
    threadpool_count = max(1, min(len(numa_nodes), threads))
    base = threads // threadpool_count
    extra = threads % threadpool_count

    worker_config = ext.WorkerPoolConfig()
    worker_config.subpool_count = threadpool_count
    worker_config.subpool_numa_map = numa_nodes[:threadpool_count]
    worker_config.subpool_thread_count = [base + (1 if i < extra else 0) for i in range(threadpool_count)]
    return worker_config


numa_nodes = detect_numa_nodes()
threads = default_thread_count(numa_nodes)
worker_config = make_worker_config(kt_kernel_ext, threads, numa_nodes)
CPUInfer = kt_kernel_ext.CPUInfer(worker_config)

class HostMemoryView:
    def __init__(
        self, 
        mlpm: MLPModuleWrapper,
        meta_model,
        client,
        tensor_index_resize_json,
    ):
        self.client = client
        self.mlpm = mlpm
        self.mlpm_hi = meta_model
        self.tensor_index_resize_json = tensor_index_resize_json
        
        mshm_names, chunk_size = self.client.get_model_shared_memory_names(self.mlpm.model_path)
        if len(mshm_names) <= 0:
            raise ValueError(f"Only Support shared memory, But sllm not shared")
        self.mshm_names = mshm_names
        self.mchunk_size = chunk_size

        time_start_restore = time.time()
        self.hm_state_dict = restore_tensors_from_shared_memory_names(
                                self.mshm_names, self.tensor_index_resize_json, self.mchunk_size)
        logger.debug(f"\nrestore_tensors_from_shared_memory_names time: {time.time() - time_start_restore}")

        time_start_restore = time.time()
        self.mlpm.restore_hm_state_dict2model(self.hm_state_dict, self.mlpm_hi)
        logger.debug(f"\nrestore_hm_state_dict2model time: {time.time() - time_start_restore}")

    def _test_kt_kernel_init(self, layer_idx: int, max_len: int = 1024):
        """测试 kt-kernel MOE 初始化，仿照 bench_bf16_moe.py 的模式"""
        import kt_kernel
        import kt_kernel_ext

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        print(f"kt-kernel version      : {kt_kernel.__version__}")
        print(f"kt-kernel CPU variant : {kt_kernel.__cpu_variant__}")
        # 获取模型配置参数
        # 修复: 使用 self.mlpm.get_ 相关方法获取参数
        expert_num = self.mlpm.get_experts_num()
        num_experts_per_tok = self.mlpm.get_experts_per_tok()
        hidden_size = self.mlpm.config.hidden_size
        moe_intermediate_size = self.mlpm.config.moe_intermediate_size
 
        # 物理到逻辑专家映射
        physical_to_logical_map = torch.tensor(range(expert_num), device="cpu", dtype=torch.int64).contiguous()
        
        # 为每个隐藏层创建 MOE 配置
        moes = []
        # layers_num = self.mlpm.config.num_hidden_layers
        layers_num = layer_idx + 1
        for i in range(layers_num):
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, moe_intermediate_size, 0)
            config.max_len = max_len
            config.layer_idx = i
            # 通过 mlpm 的通用方法获取 fused experts 线性层（兼容 language_model.layers 等路径）
            gate_up_proj_mod, down_proj_mod, _ = self.mlpm.get_fused_experts_gate_up_down_act_fn(self.mlpm_hi, i)

            # nn paramter
            gate_up_proj = gate_up_proj_mod
            down_proj = down_proj_mod
            
            # 确保张量在CPU上且连续
            gate_up_proj_cpu = gate_up_proj.cpu().contiguous()
            down_proj_cpu = down_proj.cpu().contiguous()
            
            # 通过切片直接获取指针（不拷贝数据）
            # gate_up_proj形状: [num_experts, 2*intermediate_dim, hidden_dim]
            # 通过视图获取gate和up部分的指针
            gate_up_reshaped = gate_up_proj_cpu.view(expert_num, 2, moe_intermediate_size, hidden_size)
            gate_view = gate_up_reshaped[:, 0, :, :]  # [num_experts, intermediate_size, hidden_dim]
            up_view = gate_up_reshaped[:, 1, :, :]    # [num_experts, intermediate_size, hidden_dim]

            gate_view = gate_view.cpu().contiguous()
            up_view = up_view.cpu().contiguous()
            
            # 设置 BF16 权重指针（不需要 scales）
            config.gate_proj = gate_view.data_ptr()
            config.up_proj = up_view.data_ptr()
            config.down_proj = down_proj_cpu.data_ptr()
            
            # BF16 不需要 scales
            config.gate_scale = 0
            config.up_scale = 0
            config.down_scale = 0
            config.pool = CPUInfer.backend_
            
            # 使用可用的 AVX2BF16_MOE 替代 AMXBF16_MOE
            moe = kt_kernel_ext.moe.AVX512BF16_MOE(config)
            # moe = kt_kernel_ext.moe.AMXBF16_MOE(config)  # 如果编译支持 AMX，可以启用
            
            # 提交权重加载任务
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()
            moes.append(moe)
            
            logger.info(f"初始化第 {i} 层 MOE，专家数: {expert_num}, 隐藏层大小: {hidden_size}")
        
        self.moes = moes
        return moes

    def _make_inputs(self, tokens: int, hidden_size: int, num_experts: int, top_k: int, seed: int = 42, uniform: bool = False):
        """
        生成测试输入数据，确保数据格式正确
        
        Args:
            tokens: 输入token数量
            hidden_size: 隐藏层大小
            num_experts: 专家数量
            top_k: 每个token选择的前k个专家
            seed: 随机种子
            uniform: 是否使用均匀分配策略
            
        Returns:
            hidden_states: 输入隐藏状态 [tokens, hidden_size]
            topk_ids: 专家ID [tokens, top_k]
            topk_weights: 专家权重 [tokens, top_k]
            output: 输出缓冲区 [tokens, hidden_size]
            batch_tensor: batch大小张量 [1]
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        
        # 生成输入隐藏状态
        hidden_states = torch.randn((tokens, hidden_size), generator=generator, dtype=torch.float32)
        hidden_states = hidden_states.to(torch.bfloat16).contiguous()
        
        # 生成专家ID - 确保与 kt-kernel 接口兼容
        if uniform:
            # ===== 均匀分配策略：每个专家获得相同数量的token =====
            tokens_per_expert = tokens * top_k // num_experts
            remainder = tokens * top_k % num_experts
            
            expert_assignments = []
            for expert_id in range(num_experts):
                count = tokens_per_expert + (1 if expert_id < remainder else 0)
                expert_assignments.extend([expert_id] * count)
            
            expert_assignments = torch.tensor(expert_assignments, dtype=torch.int64)
            expert_assignments = expert_assignments[torch.randperm(len(expert_assignments), generator=generator)]
            topk_ids = expert_assignments.view(tokens, top_k).contiguous()
        else:
            # ===== 随机分配策略：模拟真实MoE路由 =====
            scores = torch.rand((tokens, num_experts), generator=generator, dtype=torch.float32)
            _, topk_ids = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=False)
            topk_ids = topk_ids.to(torch.int64).contiguous()
        
        # 生成专家权重（归一化）
        topk_weights = torch.rand((tokens, top_k), generator=generator, dtype=torch.float32).contiguous()
        topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).contiguous()
        
        # 准备输出缓冲区
        output = torch.empty((tokens, hidden_size), dtype=torch.bfloat16).contiguous()
        batch_tensor = torch.tensor([tokens], dtype=torch.int32)
        
        return hidden_states, topk_ids, topk_weights, output, batch_tensor

    def prefill_hidden_states_before_gate(
        self,
        cmv: "CudaMemoryView",
        layer_idx: int,
        bsz: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        在真实 MoE ``gate`` 之前跑完单层 prefill：tokenizer + embed -> iln -> self_attn -> paln -> residual，
        与 ``MLPLLM.test_gate_experts`` 中调用 ``gate_func`` 之前的路径一致。

        Returns:
            ``[bsz, seq_len, hidden_size]``，与 ``cmv.mlpm_ci`` 同 device / dtype（config.torch_dtype）。
        """
        from transformers import AutoTokenizer
        from transformers.cache_utils import StaticCache
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask,
            _prepare_4d_causal_attention_mask_for_sdpa,
        )

        from utils.helper import generate_input_ids

        if cmv.mlpm_ci is None:
            raise RuntimeError("cmv.mlpm_ci is None; cannot run prefill_hidden_states_before_gate")

        _bsz = int(bsz)
        _seq = int(seq_len)
        if _bsz <= 0 or _seq <= 0:
            raise ValueError(f"bsz and seq_len must be positive, got bsz={_bsz}, seq_len={_seq}")

        device = f"cuda:{cmv.device1}"
        dtype = getattr(self.mlpm.config, "torch_dtype", torch.bfloat16)
        if dtype is None:
            dtype = torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(self.mlpm.model_abs_path, trust_remote_code=True)
        input_ids = generate_input_ids(tokenizer, _bsz, _seq, device)
        embed_tokens = self.mlpm.get_embed_tokens(cmv.mlpm_ci)

        with torch.inference_mode():
            x = embed_tokens(input_ids).to(dtype=dtype)
            _max_kv = int(_seq) + 64
            mpe = getattr(self.mlpm.config, "max_position_embeddings", None)
            if mpe is not None:
                _max_kv = min(_max_kv, int(mpe))
            _max_kv = max(_max_kv, int(_seq) + 1)
            past_key_value = StaticCache(config=self.mlpm.config, max_cache_len=_max_kv)
            past_key_values_length = int(past_key_value.get_seq_length())
            position_ids = torch.arange(
                past_key_values_length,
                _seq + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                None,
                (_bsz, _seq),
                x,
                past_key_values_length=past_key_values_length,
            )
            if getattr(self.mlpm.config, "_attn_implementation", None) == "eager":
                attention_mask = _prepare_4d_causal_attention_mask(
                    None,
                    (_bsz, _seq),
                    x,
                    past_key_values_length,
                )

            residual = x
            x = self.mlpm.iln_func(cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=x)
            x = self.mlpm.self_attn_func(
                cmv.mlpm_ci,
                layer_idx=layer_idx,
                hidden_states=x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            x = self.mlpm.paln_func(cmv.mlpm_ci, layer_idx=layer_idx, hidden_states=x)
            x = residual + x

        return x

    def _make_inputs_from_layer_prefill(
        self,
        cmv: "CudaMemoryView",
        tokens_num: int,
        layer_idx: int,
        bsz: Optional[int],
        seq_len: Optional[int],
    ):
        """
        与 ``MLPLLM.test_gate_experts`` 一致：``prefill_hidden_states_before_gate`` 得到隐状态，
        再用 ``gate_func(mlpm_hi)`` 得到 top-k；供 kt-kernel 使用 CPU 上的 bf16 隐藏态与路由张量。
        """
        _bsz = int(bsz) if bsz is not None else 1
        _seq = int(seq_len) if seq_len is not None else int(tokens_num)
        if _bsz * _seq != int(tokens_num):
            raise ValueError(
                f"bsz*seq_len must equal tokens_num for kt infer: bsz={_bsz}, seq_len={_seq}, tokens_num={tokens_num}"
            )

        hidden_size = int(self.mlpm.config.hidden_size)
        x = self.prefill_hidden_states_before_gate(cmv, layer_idx, _bsz, _seq)

        with torch.inference_mode():
            topk_idx, topk_weight, _ = self.mlpm.gate_func(self.mlpm_hi, layer_idx, x.detach().to("cpu"))

        hidden_states = (
            x.reshape(-1, hidden_size).detach().cpu().to(torch.bfloat16).contiguous()
        )
        topk_ids = topk_idx.to(torch.int64).contiguous().cpu()
        topk_weights = topk_weight.float().contiguous().cpu()
        output = torch.empty((tokens_num, hidden_size), dtype=torch.bfloat16).contiguous()
        batch_tensor = torch.tensor([tokens_num], dtype=torch.int32)
        return hidden_states, topk_ids, topk_weights, output, batch_tensor

    def _test_kt_kernel_infer(
        self,
        max_len: int = 1024,
        cmv: Optional["CudaMemoryView"] = None,
        layer_idx: int = 0,
        bsz: Optional[int] = None,
        seq_len: Optional[int] = None,
    ):
        """
        测试 kt-kernel MoE 推理功能，确保数据格式与 bench_bf16_moe.py 对齐
        
        Args:
            moes: MOE层实例列表
            batch_tensor: batch大小张量 [1]，包含token数量（对应qlen_tensor）
            topk_ids: 专家ID张量 [tokens, top_k]；半套外槽位会被置为 **-1**（kt_kernel 跳过，勿用 0）
            topk_weights: 专家权重张量 [tokens, top_k]；半套外槽位置 0，**不做行归一化**
            hidden_states: 输入隐藏状态 [tokens, hidden_size]，BF16格式
            output: 输出缓冲区 [tokens, hidden_size]，BF16格式
            top_k: 每个token选择的前k个专家（对应num_experts_per_tok）

            cmv: 若传入 ``CudaMemoryView``，则隐藏态与路由来自真实 prefill（与 ``lmp.test_gate_experts`` 中
                embed -> iln -> self_attn -> paln -> residual 及 ``gate_func(mlpm_hi)`` 一致），否则沿用随机 ``_make_inputs``。
            layer_idx / bsz / seq_len: 仅当 ``cmv`` 非空时生效；默认 ``bsz=1``、``seq_len=max_len``，且须满足 ``bsz*seq_len==max_len``。
        """
        import kt_kernel_ext
        
        tokens_num = max_len
        # if cmv is not None:
        hidden_states, topk_ids, topk_weights, output, batch_tensor = self._make_inputs_from_layer_prefill(
            cmv,
            tokens_num=tokens_num,
            layer_idx=layer_idx,
            bsz=bsz,
            seq_len=seq_len,
        )
        # else:
        #     hidden_states, topk_ids, topk_weights, output, batch_tensor = self._make_inputs(
        #         tokens=tokens_num,
        #         hidden_size=self.mlpm.config.hidden_size,
        #         num_experts=self.mlpm.get_experts_num(),
        #         top_k=self.mlpm.get_experts_per_tok(),
        #         seed=42,
        #         uniform=False,
        #     )
        tok_k = self.mlpm.get_experts_per_tok()

        # 选半套专家（与 _group_fused_select_half_experts 一致）；半套外槽位 topk_ids=-1、权重=0（kt_kernel 跳过 -1；
        # 勿用 0 作占位），行不归一化。
        num_experts_total = int(self.mlpm.get_experts_num())
        ratio = 0.5
        half_e = max(1, int(num_experts_total * ratio))
        half_mode = "noncontiguous"
        slot_expert_ids = topk_ids.reshape(-1).to(torch.int64)
        _log_topk_activation_distribution(
            topk_ids,
            num_experts_total,
            log_prefix="kt-kernel infer: native topk_ids",
        )
        expert_half_list, window_tokens, max_tokens = _group_fused_select_half_experts(
            slot_expert_ids, num_experts_total, half_e, half_mode
        )
        
        topk_ids, topk_weights = _mask_topk_weights_outside_expert_half(
            topk_ids, topk_weights, expert_half_list
        )
        total_slots = int(slot_expert_ids.numel())
        chosen_half_pct = (window_tokens / total_slots * 100.0) if total_slots > 0 else 0.0
        logger.info(
            "kt-kernel infer: half_expert subset (%s): E_half=%d experts=%s "
            "slots_in_chosen_half=%d slots_in_chosen_half_pct=%.4f%% max_count_among_chosen=%d (total E=%d, T*K=%d; "
            "outside half: topk_id=-1, weights=0, no row renorm)",
            half_mode,
            half_e,
            expert_half_list,
            window_tokens,
            chosen_half_pct,
            max_tokens,
            num_experts_total,
            total_slots,
        )
        _log_kt_kernel_topk_activation(
            topk_ids,
            num_experts_total,
            expert_half_list,
            tokens_num,
            tok_k,
        )

        moes = self.moes
        # 使用第一个MOE层进行测试
        layer_idx = 0
        
        time0 = time.perf_counter()
        # 提交前向推理任务 - 参数格式与 bench_bf16_moe.py 完全对齐
        CPUInfer.submit(
            moes[layer_idx].forward_task(
                batch_tensor.data_ptr(),        # 参数1: batch大小指针，对应qlen_tensor.data_ptr()
                tok_k,                          # 参数2: 每个token选择的专家数，对应num_experts_per_tok
                topk_ids.data_ptr(),            # 参数3: 专家ID指针，内存布局为连续专家ID数组
                topk_weights.data_ptr(),        # 参数4: 专家权重指针（可为部分和，行和未必为 1）
                hidden_states.data_ptr(),       # 参数5: 输入隐藏状态指针，BF16格式
                output.data_ptr(),              # 参数6: 输出缓冲区指针，BF16格式
                False,                          # 参数7: 是否使用调试模式
            )
        )
        
        # 等待推理任务完成
        CPUInfer.sync()
        logger.info(f"kt-kernel MoE推理完成，时间: {time.perf_counter() - time0:.6f}秒")        
        # 计算输出校验和，用于验证结果正确性
        output_checksum = float(output.float().sum())
        logger.info(f"kt-kernel MoE推理完成，输出checksum: {output_checksum:.6f}")
        
        return output_checksum


    def _test_group_bmm_fused_experts(self):
        """Half-expert set minimizes total top-1 routed tokens; mode from env ``LMP_TEST_GROUP_FUSED_EXPERTS_HALF``."""
        _print_group_bmm_self_test(
            "[test_group_bmm_fused_experts] start (print+flush; bypasses logging)."
        )
        for half_mode in _group_fused_test_parse_half_modes():
            self._test_group_bmm_fused_experts_impl(half_mode)

    def _test_group_bmm_fused_experts_impl(self, half_mode: str):
        import time

        t_all0 = time.perf_counter()
        # batch=64, seq_len=128
        bsz, seq_len = 4, 128
        layer_idx = self.mlpm.get_first_k_dense_replace()
        hidden_size = int(getattr(self.mlpm.config, "hidden_size"))

        t0 = time.perf_counter()
        x = torch.randn((bsz, seq_len, hidden_size), device="cpu", dtype=torch.bfloat16)
        _print_group_bmm_self_test(
            f"[test_group_bmm_fused_experts][half={half_mode}] input_gen: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )

        # Gate -> experts：展平 **top‑k 各路**（每 token K 槽），半套专家统计与 BMM 均按槽位计。
        t0 = time.perf_counter()
        topk_idx, topk_weight, _ = self.mlpm.gate_func(self.mlpm_hi, layer_idx, x)
        _print_group_bmm_self_test(
            f"[test_group_bmm_fused_experts][half={half_mode}] gate_func: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )
        flat = x.view(-1, x.size(-1))  # [T, H]
        topk_idx = topk_idx.to(flat.device)
        topk_weight = topk_weight.to(flat.device)
        t_tokens = int(flat.size(0))
        k = int(topk_idx.size(1))
        if k <= 0:
            raise RuntimeError(f"gate_func returned invalid top-k dim: {tuple(topk_idx.shape)}")

        slot_expert_ids = topk_idx.reshape(-1).to(torch.int64)  # [T*K]
        slot_weights = topk_weight.reshape(-1).to(torch.bfloat16)  # [T*K]
        slot_token_row = (
            torch.arange(t_tokens, device=flat.device, dtype=torch.int64)
            .unsqueeze(1)
            .expand(t_tokens, k)
            .reshape(-1)
        )  # [T*K] 每槽对应 flat 的行号

        n_unique_experts = int(torch.unique(slot_expert_ids).numel())
        n_routed_slots = int(t_tokens * k)
        
        t0 = time.perf_counter()
        # Total routed experts for this model/config — minimize **total activated slots** in the half-set.
        num_experts_total = int(self.mlpm.get_experts_num())
        half_e = max(1, num_experts_total // 2)
        expert_half_list, window_tokens, max_tokens = _group_fused_select_half_experts(
            slot_expert_ids, num_experts_total, half_e, half_mode
        )
        expert_half_tensor = torch.tensor(
            expert_half_list, dtype=torch.int64, device=slot_expert_ids.device
        )

        _print_group_bmm_self_test(
            f"[test_group_bmm_fused_experts][half={half_mode}] "
            f"activated_experts_unique={n_unique_experts}, routed_slots={n_routed_slots} total experts = {num_experts_total} "
            f"(T={t_tokens}, K={k})"
        )


        # Build global->local mapping for presorted bmm path: local ids in [0, half_e)
        global_to_local = torch.full(
            (num_experts_total,), -1, dtype=torch.int64, device=slot_expert_ids.device
        )
        for li, eid in enumerate(expert_half_list):
            global_to_local[eid] = li

        _print_group_bmm_self_test(
            f"[test_group_bmm_fused_experts][half={half_mode}] choose_half_experts: {(time.perf_counter() - t0) * 1e3:.3f} ms "
            f"(E_total={num_experts_total}, E_half={half_e}, experts={expert_half_list}, "
            f"total_tokens={window_tokens}, max_tokens={max_tokens})"
        )

        # 只保留「槽位上的专家 id ∈ 半套专家」的各路（共 T*K 槽中一个子集）
        t0 = time.perf_counter()
        in_half = torch.isin(slot_expert_ids, expert_half_tensor)
        if not bool(in_half.any().item()):
            raise RuntimeError("No top-k slots routed into the selected half expert subset.")
        slot_ids = torch.nonzero(in_half, as_tuple=False).flatten()
        _print_group_bmm_self_test(
            f"[test_group_bmm_fused_experts][half={half_mode}] filter_slots_in_half: {(time.perf_counter() - t0) * 1e3:.3f} ms "
            f"(slots_in_half={int(slot_ids.numel())}, T={t_tokens}, K={k})"
        )

        # Restore fused weights for the selected half experts via group_fused_experts_tensor
        t0 = time.perf_counter()
        group = self.group_fused_experts_tensor(layer_idx, expert_half_list)
        _print_group_bmm_self_test(
            f"[test_group_bmm_fused_experts][half={half_mode}] restore_group_fused_weights: "
            f"{(time.perf_counter() - t0) * 1e3:.3f} ms"
        )
        group_gate_up = group["group_gate_up"]
        group_down = group["group_down"]
        if group_gate_up.size(0) != half_e or group_down.size(0) != half_e:
            raise RuntimeError(
                f"group_fused_experts_tensor returned wrong E: gate_up={group_gate_up.size(0)} down={group_down.size(0)} expected={half_e}"
            )

        t0 = time.perf_counter()
        group_gate_up = group_gate_up  # [E, 2I, H]
        group_down = group_down        # [E, H, I]
        gu_eh2i = group_gate_up.transpose(1, 2) # [E, H, 2I]
        dn_eih = group_down.transpose(1, 2)     # [E, I, H]
        _print_group_bmm_self_test(
            f"[test_group_bmm_fused_experts][half={half_mode}] transpose_contig: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )


        t0 = time.perf_counter()
        stacked, counts, perm = self.mlpm._gather_sort_and_pad_presorted(
            flat_hidden_states=flat,
            slot_token_row=slot_token_row,
            slot_expert_ids=slot_expert_ids,
            global_to_local=global_to_local,
            slot_ids=slot_ids,
            num_experts=half_e,
        )
        _print_group_bmm_self_test(
            f"[test_group_bmm_fused_experts][half={half_mode}] gather_sort_pad: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )

        t0 = time.perf_counter()
        gu = torch.bmm(stacked, gu_eh2i)
        t_bmm1 = time.perf_counter()
        half = gu.size(-1) // 2
        g, u = gu.split(half, dim=-1)
        act_name = getattr(self.mlpm.config, "hidden_activation", None) or getattr(self.mlpm.config, "hidden_act", None)
        if act_name is None:
            raise RuntimeError("Cannot determine activation name for fused experts test.")
        act_fn = ACT2FN[act_name]
        mid = act_fn(g) * u
        t_act = time.perf_counter()
        y_stacked = torch.bmm(mid, dn_eih)
        t_bmm2 = time.perf_counter()
        y_sorted_bmm = self.mlpm._batched_unpad_outputs(y_stacked, counts)
        t_unpad = time.perf_counter()
        _print_group_bmm_self_test(
            f"[test_group_bmm_fused_experts][half={half_mode}] fused_group_bmm: "
            f"{(t_unpad - t0) * 1e3:.3f} ms "
            f"(bmm1={(t_bmm1 - t0) * 1e3:.3f} act+split={(t_act - t_bmm1) * 1e3:.3f} "
            f"bmm2={(t_bmm2 - t_act) * 1e3:.3f} unpad={(t_unpad - t_bmm2) * 1e3:.3f})"
        )

       

    def _test_group_gmm_fused_experts(self):
        """
        GMM version for fused experts group compute (no explicit padding to max_tokens).

        Mirrors ``_test_group_bmm_fused_experts`` expert selection (min total top-1 routed tokens; see
        ``LMP_TEST_GROUP_FUSED_EXPERTS_HALF``). Uses
        ``MLPModuleWrapper.fused_experts_gate_up_down_mm_presorted(..., mm_backend="gmm")``
        and checks consistency against the bmm backend on the same presorted slots.
        """
        for half_mode in _group_fused_test_parse_half_modes():
            self._test_group_gmm_fused_experts_impl(half_mode)

    def _test_group_gmm_fused_experts_impl(self, half_mode: str):
        import time

        t_all0 = time.perf_counter()
        bsz, seq_len = 64, 128
        layer_idx = self.mlpm.get_first_k_dense_replace()
        hidden_size = int(getattr(self.mlpm.config, "hidden_size"))

        t0 = time.perf_counter()
        x = torch.randn((bsz, seq_len, hidden_size), device="cpu", dtype=torch.bfloat16)
        logger.debug(
            f"[test_group_gmm_fused_experts][half={half_mode}] input_gen: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )

        t0 = time.perf_counter()
        topk_idx, topk_weight, _ = self.mlpm.gate_func(self.mlpm_hi, layer_idx, x)
        logger.debug(
            f"[test_group_gmm_fused_experts][half={half_mode}] gate_func: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )
        expert_ids = topk_idx[:, 0].to(torch.int64)      # [T]
        expert_w = topk_weight[:, 0].to(torch.bfloat16)  # [T]
        flat = x.view(-1, x.size(-1))                    # [T, H]

        num_experts_total = int(self.mlpm.get_experts_num())
        half_e = max(1, num_experts_total // 2)

        t0 = time.perf_counter()
        expert_half_list, window_tokens, max_tokens = _group_fused_select_half_experts(
            expert_ids, num_experts_total, half_e, half_mode
        )
        expert_half_tensor = torch.tensor(expert_half_list, dtype=torch.int64, device=expert_ids.device)

        global_to_local = torch.full((num_experts_total,), -1, dtype=torch.int64, device=expert_ids.device)
        for li, eid in enumerate(expert_half_list):
            global_to_local[eid] = li

        logger.debug(
            f"[test_group_gmm_fused_experts][half={half_mode}] choose_half_experts: {(time.perf_counter() - t0) * 1e3:.3f} ms "
            f"(E_total={num_experts_total}, E_half={half_e}, experts={expert_half_list}, "
            f"total_tokens={window_tokens}, max_tokens={max_tokens})"
        )

        # Only compute tokens routed to the selected half subset.
        in_half = torch.isin(expert_ids, expert_half_tensor)
        if not bool(in_half.any().item()):
            raise RuntimeError("No tokens routed into the selected half expert subset.")
        token_ids = torch.nonzero(in_half, as_tuple=False).flatten()

        x_half = flat.index_select(0, token_ids)
        expert_ids_half_global = expert_ids.index_select(0, token_ids)
        expert_ids_half_local = global_to_local.index_select(0, expert_ids_half_global)  # [S]

        # Presort by expert id for grouped-mm
        t0 = time.perf_counter()
        expert_ids_sorted, perm = torch.sort(expert_ids_half_local)
        x_sorted = x_half.index_select(0, perm)  # [S, H]
        logger.debug(
            f"[test_group_gmm_fused_experts][half={half_mode}] sort_by_expert: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )

        # Restore fused weights for the selected half experts
        t0 = time.perf_counter()
        group = self.group_fused_experts_tensor(layer_idx, expert_half_list)
        logger.debug(
            f"[test_group_gmm_fused_experts][half={half_mode}] restore_group_fused_weights: "
            f"{(time.perf_counter() - t0) * 1e3:.3f} ms"
        )
        group_gate_up = group["group_gate_up"]  # [E, 2I, H]
        group_down = group["group_down"]        # [E, H, I]
        if group_gate_up.size(0) != half_e or group_down.size(0) != half_e:
            raise RuntimeError(
                f"group_fused_experts_tensor returned wrong E: gate_up={group_gate_up.size(0)} down={group_down.size(0)} expected={half_e}"
            )

        # Prepare weights for mm backends
        t0 = time.perf_counter()
        gu_eh2i = group_gate_up.transpose(1, 2)  # [E, H, 2I]
        dn_eih = group_down.transpose(1, 2)      # [E, I, H]
        logger.debug(
            f"[test_group_gmm_fused_experts][half={half_mode}] transpose: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )

        act_name = getattr(self.mlpm.config, "hidden_activation", None) or getattr(self.mlpm.config, "hidden_act", None)
        if act_name is None:
            raise RuntimeError("Cannot determine activation name for fused experts test.")
        act_fn = ACT2FN[act_name]

        # Run gmm and bmm on the same presorted slots and compare.
        t0 = time.perf_counter()
        y_sorted_gmm = self.mlpm.fused_experts_gate_up_down_mm_presorted(
            x_slots_sorted=x_sorted,
            expert_ids_sorted=expert_ids_sorted,
            gate_up_w_eh2i=gu_eh2i,
            down_w_eih=dn_eih,
            act_fn=act_fn,
            mm_backend="gmm",
        )
        logger.debug(
            f"[test_group_gmm_fused_experts][half={half_mode}] fused_group_gmm: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )

        t0 = time.perf_counter()
        y_sorted_bmm = self.mlpm.fused_experts_gate_up_down_mm_presorted(
            x_slots_sorted=x_sorted,
            expert_ids_sorted=expert_ids_sorted,
            gate_up_w_eh2i=gu_eh2i,
            down_w_eih=dn_eih,
            act_fn=act_fn,
            mm_backend="bmm",
        )
        logger.debug(
            f"[test_group_gmm_fused_experts][half={half_mode}] fused_group_bmm_ref: {(time.perf_counter() - t0) * 1e3:.3f} ms"
        )

        torch.testing.assert_close(y_sorted_gmm, y_sorted_bmm, rtol=2e-2, atol=2e-2)

        # Unsort to token order and apply weights (sanity)
        y_half = torch.empty_like(x_half)
        y_half.index_copy_(0, perm, y_sorted_gmm)
        y_half = y_half * expert_w.index_select(0, token_ids).unsqueeze(-1)

        logger.debug(
            f"[test_group_gmm_fused_experts][half={half_mode}] total: {(time.perf_counter() - t_all0) * 1e3:.3f} ms "
            f"(T={int(flat.size(0))}, tokens_in_half={int(token_ids.numel())}, E_half={half_e})"
        )
