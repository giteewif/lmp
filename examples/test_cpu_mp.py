"""
测试 CPUMP - 执行CPU einsum推理
"""
import sys
import os
import time
import torch
import copy

# ===== 关键：必须在导入任何使用 multiprocessing 的模块之前设置启动方法 =====
# 这对于 CUDA 多进程是必需的
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    # 已经设置过了，检查是否是 'spawn'
    current_method = mp.get_start_method(allow_none=True)
    if current_method != 'spawn':
        # 如果不是 'spawn'，尝试强制设置
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError as e:
            print(f"Warning: Cannot set start method to 'spawn': {e}")
            print(f"Current method: {current_method}")
            print("This may cause CUDA initialization errors in subprocesses.")

# 获取项目根目录和必要的路径
project_root = "/mnt/zhengcf3/lmp"
src_dir = os.path.join(project_root, 'src')
sllm_store_dir = os.path.join(project_root, 'src', 'sllm_store')

# 添加必要的目录到 Python 路径
sys.path.insert(0, sllm_store_dir)
sys.path.insert(0, src_dir)

from transformers import AutoConfig
from models.Deepseek.deepseek_moe_16b_base.modeling_deepseek import DeepseekDecoderLayer
from models.Deepseek.mlpmodule import DeepseekOCalModel
from accelerate import init_empty_weights
from lmp.device_mp import DeviceMP
from lmp.cpu_thread_manager_mp import CPUExpertsManagerMP
from utils.cuda_h import cuda_hook, cuda_hook_end, cuda_hook_time, cuda_hook_time_end

from lmp.lmp import MLPLLM

@torch.no_grad()
def main():
    model_path = "deepseek-moe-16b-base-bfloat16"
    model_name_type = "Deepseek"

    cpu_thread_manager_mp = CPUExpertsManagerMP(num_workers=2, model_path=model_path, model_name_type=model_name_type)
    cpu_thread_manager_mp.start()

    mlpllm =MLPLLM( model_name_type=model_name_type, model_path=model_path )

    mlpllm.cmv.start_init_meta_model(hmv=mlpllm.hmv)
    mlpllm.cmv.load_general_and_init()
    mlpllm.cmv.init_load_qkvogn_es_weight(layer_idx=0)
    model_cpy = copy.deepcopy(mlpllm.cmv.mlpm_ci)
    mlpllm.mlpm.restore_hm_state_dict2model(mlpllm.hmv.hm_state_dict, model_cpy)
    
    inputs_tokens = torch.randn(32, 64, 2048, dtype=torch.bfloat16, device="cuda:1")
    
    mlpllm.cmv.start_load_qkvogn_s_weight(layer_idx=1, device=mlpllm.device1)
    mlpllm.cmv.wait_load_qkvogn_s_weight(layer_idx=1)

    hidden_states = inputs_tokens
    batch_size, seq_len = hidden_states.shape[:2]
    layer_idx=1
    cuda_hook_time("gate")
    topk_idx, topk_weight, aux_loss = mlpllm.mlpm.gate_func(mlpllm.cmv.mlpm_ci, layer_idx, hidden_states)
    flat_expert_indices = topk_idx.view(-1)      # [batch_size * seq_len * num_experts_per_tok]
    flat_experts_weight = topk_weight.view(-1, 1)  # [batch_size * seq_len * num_experts_per_tok, 1]
    idxs = flat_expert_indices.argsort()         # 排序后的索引
    tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # [num_experts]
    token_idxs = idxs // mlpllm.mlpm.config.num_experts_per_tok  # 恢复到原始 token 索引
    flat_hidden_states = hidden_states.view(batch_size * seq_len, -1)  # [batch_size * seq_len, hidden_dim]
    cuda_hook_time_end("gate")

    cuda_hook_time("experts_map_get")
    expert_indices_map = {}  # {expert_id: (start_idx, end_idx)} 保存索引范围
    expert_token_indices_map = {}  # {expert_id: token_ids} 保存 token 索引
    expert_token_counts_list = []  # 用于 CPU/GPU 分配：[(expert_id, token_count), ...]
    num_experts = mlpllm.mlpm.get_experts_num()
    prev_end = 0  # 前一个 expert 的结束位置
    for expert_id in range(num_experts):
        if expert_id >= len(tokens_per_expert):
            break
        end_idx = int(tokens_per_expert[expert_id])
        if end_idx == prev_end:
            continue
        start_idx = prev_end
        token_count = end_idx - start_idx  # 该 expert 的实际 token 数量
        expert_indices_map[expert_id] = (start_idx, end_idx)
        expert_token_indices_map[expert_id] = token_idxs[start_idx:end_idx]
        expert_token_counts_list.append((expert_id, token_count))
        prev_end = end_idx
    sorted_experts_by_load = sorted(expert_token_counts_list, key=lambda x: x[1])
    num_experts_total = len(sorted_experts_by_load)
    gpu_experts_list = sorted_experts_by_load[:num_experts_total]

    cpu_experts_list = [ i[0] for i in sorted_experts_by_load][:num_experts_total//2]

    num_device = len(mlpllm.device_list)
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
    
    device_index = int(mlpllm.device1.split(":")[1])
    device1_expert_ids = gpu_expert_ids_by_device[device_index]
    cuda_hook_time_end("experts_map_get")

    # gpu_experts_names = mlpllm.mlpm.get_experts_names(layer_idx=1, expert_idx_list=[i for i in range(0, 27)])
    # ret, replica_uuid, state_dict = \
    #     mlpllm.cmv.allocate_cuda_memory_and_load_into_gpu(
    #         gpu_experts_names, device_index_int=int(mlpllm.device1.split(":")[1])
    #     )
    # mlpllm.cmv.wait_load_into_gpu(replica_uuid=replica_uuid)

    # mlpllm.cmv.restore2model(state_dict, mlpllm.cmv.mlpm_ci)
    final_hidden_states = torch.zeros_like(flat_hidden_states)
    for i in range(8):
        cuda_hook_time("task_processing_mp")
        
        cpu_thread_manager_mp.submit_worker(
            worker_idx=0,
            layer_idx=layer_idx,
            expert_idx_list=list(cpu_experts_list),
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            final_hidden_states=final_hidden_states,
        )

        output_tensor = cpu_thread_manager_mp.wait()
        # print(output_tensor.shape)
        cuda_hook_time_end("task_processing_mp")
    
    time.sleep(1)
    cpu_thread_manager_mp.stop()

if __name__ == '__main__':
    main()