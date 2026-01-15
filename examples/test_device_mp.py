"""
测试 DeviceMP - 共享 DecoderLayer 对象版本
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
from lmp.device_mp_init import DeviceMP2
from utils.cuda_h import cuda_hook, cuda_hook_end, cuda_hook_time, cuda_hook_time_end
from utils.helper import print_layer_parameters
from lmp.lmp import MLPLLM

@torch.no_grad()
def main():
    model_path = "deepseek-moe-16b-base-bfloat16"
    model_name_type = "Deepseek"
    

    mlpllm = MLPLLM( model_name_type=model_name_type, model_path=model_path )

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

    # 将 GPU 专家平均分配到多个设备，同时尽量平衡每个设备的 token 数量
    # 使用贪心算法：按 token 数量从大到小排序，每次分配给当前 token 数量最少的设备
    device_ids = [int(device.split(":")[1]) for device in mlpllm.device_list]
    device_expert_map = {device_id: [] for device_id in device_ids}  # {device_id: [expert_id, ...]}
    device_token_counts = {device_id: 0 for device_id in device_ids}  # {device_id: token_count}
    
    # 按 token 数量从大到小排序，优先分配大负载的 expert
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
    
    cuda_hook_time_end("experts_map_get")

    device1_id = device_ids[0]
    device2_id = device_ids[1] if len(device_ids) > 1 else None

    expert_cache = torch.zeros_like(flat_hidden_states)
    gpu_experts_idx_list_map = {
        device1_id: list(gpu_expert_ids_by_device[device1_id]),
        device2_id: list(gpu_expert_ids_by_device[device2_id])
    }
    streams = { device_id: torch.cuda.Stream(device=device_id) for device_id in device_ids }
    for i in range(8):
        cuda_hook_time("task_processing_mp_load")
        device_id1_experts_names = mlpllm.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=list(gpu_expert_ids_by_device[device1_id]))
        device_id2_experts_names = mlpllm.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=list(gpu_expert_ids_by_device[device2_id]))
        tensor_index_names_device_map = {
            device1_id: device_id1_experts_names, device2_id: device_id2_experts_names
        }
        ret, replica_uuid, state_dict = \
            mlpllm.cmv.allocate_cuda_memory_and_load_into_gpu_multi_device(
                tensor_index_names_device_map=tensor_index_names_device_map
            )
        mlpllm.cmv.restore2model(state_dict, mlpllm.cmv.mlpm_ci)

        device1_expert_idx_list = list(gpu_expert_ids_by_device[device1_id])
        device2_expert_idx_list = list(gpu_expert_ids_by_device[device2_id])

        if device1_id != flat_hidden_states.device.index:
            device1_idxs = idxs.to(device1_id)
            device1_flat_experts_weight = flat_experts_weight.to(device1_id)
            device1_flat_hidden_states = flat_hidden_states.to(device1_id)
        else:
            device1_idxs = idxs
            device1_flat_experts_weight = flat_experts_weight
            device1_flat_hidden_states = flat_hidden_states

        if device2_id != flat_hidden_states.device.index:
            device2_idxs = idxs.to(device2_id)
            device2_flat_experts_weight = flat_experts_weight.to(device2_id)
            device2_flat_hidden_states = flat_hidden_states.to(device2_id)
        else:
            device2_idxs = idxs
            device2_flat_experts_weight = flat_experts_weight
            device2_flat_hidden_states = flat_hidden_states

        # mlpllm.cmv.wait_load_into_gpu(replica_uuid=replica_uuid)

        device1_stacked_inputs = mlpllm.mlpm.experts_func_mgpu_group_pad(
            expert_idx_list=device1_expert_idx_list,
            expert_indices_map=expert_indices_map,
            device_flat_hidden_states=device1_flat_hidden_states,
            device_idxs=device1_idxs,
        )
        
        device2_stacked_inputs = mlpllm.mlpm.experts_func_mgpu_group_pad(
            expert_idx_list=device2_expert_idx_list,
            expert_indices_map=expert_indices_map,
            device_flat_hidden_states=device2_flat_hidden_states,
            device_idxs=device2_idxs,
        )
        group_w1_list, group_w2_list, group_w3_list = mlpllm.mlpm.experts_func_mgpu_group_list(
            mi=mlpllm.cmv.mlpm_ci,
            layer_idx=layer_idx,
            expert_idx_list=list(gpu_expert_ids_by_device[device1_id])
        )
        dgroup_w1_list, dgroup_w2_list, dgroup_w3_list = mlpllm.mlpm.experts_func_mgpu_group_list(
            mi=mlpllm.cmv.mlpm_ci,
            layer_idx=layer_idx,
            expert_idx_list=list(gpu_expert_ids_by_device[device2_id])
        )

        # 提前收集 weights 和 token_ids，用于后续一次性批量处理
        device1_token_idxs = device1_idxs // mlpllm.mlpm.config.num_experts_per_tok
        device2_token_idxs = device2_idxs // mlpllm.mlpm.config.num_experts_per_tok
        
        # 合并所有设备的 expert_token_indices_map（key 是 expert_id，value 是 token_ids）
        all_expert_token_indices_map = {}
        device1_all_expert_weights = []
        device1_all_token_ids = []
        for i, expert_idx in enumerate(device1_expert_idx_list):
            start_idx, end_idx = expert_indices_map[expert_idx]
            token_ids = device1_token_idxs[start_idx:end_idx]
            all_expert_token_indices_map[expert_idx] = token_ids
            expert_weights = device1_flat_experts_weight[device1_idxs[start_idx:end_idx]]
            device1_all_expert_weights.append(expert_weights)
            device1_all_token_ids.append(token_ids)
        
        device2_all_expert_weights = []
        device2_all_token_ids = []
        for i, expert_idx in enumerate(device2_expert_idx_list):
            start_idx, end_idx = expert_indices_map[expert_idx]
            token_ids = device2_token_idxs[start_idx:end_idx]
            all_expert_token_indices_map[expert_idx] = token_ids
            expert_weights = device2_flat_experts_weight[device2_idxs[start_idx:end_idx]]
            device2_all_expert_weights.append(expert_weights)
            device2_all_token_ids.append(token_ids)
        
        # 将列表 concat 为 tensor
        device1_all_expert_weights_concat = torch.cat(device1_all_expert_weights, dim=0) if device1_all_expert_weights else None
        device1_all_token_ids_concat = torch.cat(device1_all_token_ids, dim=0) if device1_all_token_ids else None
        device2_all_expert_weights_concat = torch.cat(device2_all_expert_weights, dim=0) if device2_all_expert_weights else None
        device2_all_token_ids_concat = torch.cat(device2_all_token_ids, dim=0) if device2_all_token_ids else None

        mlpllm.cmv.wait_load_into_gpu(replica_uuid=replica_uuid)

        cuda_hook_time_end("task_processing_mp_load")

        cuda_hook_time("exec_together")
        mlpllm.mlpm.experts_func_mgpu_einsum(
            mi=mlpllm.cmv.mlpm_ci,
            layer_idx=layer_idx,
            expert_idx_list_map=gpu_experts_idx_list_map,
            expert_indices_map=expert_indices_map,
            expert_token_indices_map=expert_token_indices_map,
            flat_hidden_states=flat_hidden_states,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            final_hidden_states=expert_cache,
            streams =streams
        )
        cuda_hook_time_end("exec_together")
        # torch.cuda.synchronize()
        cuda_hook_time("exec_one_by_one")
        for i in gpu_experts_idx_list_map.keys():
            mlpllm.mlpm.experts_func_gpu_einsum(
                mi=mlpllm.cmv.mlpm_ci,
                layer_idx=layer_idx,
                expert_idx_list=gpu_experts_idx_list_map[i],
                expert_indices_map=expert_indices_map,
                expert_token_indices_map=expert_token_indices_map,
                flat_hidden_states=flat_hidden_states,
                flat_experts_weight=flat_experts_weight,
                idxs=idxs,
                final_hidden_states=expert_cache,
            )
        cuda_hook_time_end("exec_one_by_one")
        # torch.cuda.synchronize()
        cuda_hook_time("exec_one_by_one_end_new")
        group_w1, group_w2, group_w3 = mlpllm.mlpm.experts_func_mgpu_group_experts(
            group_w1_list=group_w1_list,
            group_w2_list=group_w2_list,
            group_w3_list=group_w3_list,
        )

        dgroup_w1, dgroup_w2, dgroup_w3 = mlpllm.mlpm.experts_func_mgpu_group_experts(
            group_w1_list=dgroup_w1_list,
            group_w2_list=dgroup_w2_list,
            group_w3_list=dgroup_w3_list,
        )

        _ = mlpllm.mlpm.experts_func_mgpu_einsum_mp(
            layer_idx=layer_idx,
            group_w1=group_w1,
            group_w2=group_w2,
            group_w3=group_w3,
            stacked_inputs=device1_stacked_inputs,
            expert_idx_list=device1_expert_idx_list,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=device1_flat_hidden_states,
            flat_experts_weight=device1_flat_experts_weight,
            idxs=device1_idxs,
            final_hidden_states=expert_cache,
        )

        _ = mlpllm.mlpm.experts_func_mgpu_einsum_mp(
            layer_idx=layer_idx,
            group_w1=dgroup_w1,
            group_w2=dgroup_w2,
            group_w3=dgroup_w3,
            stacked_inputs=device2_stacked_inputs,
            expert_idx_list=device2_expert_idx_list,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=device2_flat_hidden_states,
            flat_experts_weight=device2_flat_experts_weight,
            idxs=device2_idxs,
            final_hidden_states=expert_cache,
        )


        cuda_hook_time_end("exec_one_by_one_end_new")
        # torch.cuda.synchronize()

        cuda_hook_time("exec_one_by_one_end_new2")
        _ = mlpllm.mlpm.experts_func_mgpu_einsum_mp_multi_list(
            layer_idx=layer_idx,
            group_w1_list_map={device1_id: group_w1_list, device2_id: dgroup_w1_list},
            group_w2_list_map={device1_id: group_w2_list, device2_id: dgroup_w2_list},
            group_w3_list_map={device1_id: group_w3_list, device2_id: dgroup_w3_list},
            stacked_inputs_map={device1_id: device1_stacked_inputs, device2_id: device2_stacked_inputs},
            expert_idx_list_map={device1_id: device1_expert_idx_list, device2_id: device2_expert_idx_list},
            expert_indices_map=expert_indices_map,
            flat_hidden_states_map={device1_id: device1_flat_hidden_states, device2_id: device2_flat_hidden_states},
            flat_experts_weight_map={device1_id: device1_flat_experts_weight, device2_id: device2_flat_experts_weight},
            idxs_map={device1_id: device1_idxs, device2_id: device2_idxs},
            final_hidden_states=expert_cache,
            all_expert_weights_map={device1_id: device1_all_expert_weights_concat, device2_id: device2_all_expert_weights_concat} if device1_all_expert_weights_concat is not None else None,
            all_token_ids_map={device1_id: device1_all_token_ids_concat, device2_id: device2_all_token_ids_concat} if device1_all_token_ids_concat is not None else None,
            expert_token_indices_map=all_expert_token_indices_map,
        )
        cuda_hook_time_end("exec_one_by_one_end_new2")

        mlpllm.cmv.free_allocated()
    
    


    # dp = DeviceMP2(num_processes=2)
    # dp.start()

    # expert_cache = torch.zeros_like(flat_hidden_states)
    # for i in range(8):

    #     cuda_hook_time("task_processing_mp_load")
    #     device_id1_experts_names = mlpllm.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=list(gpu_expert_ids_by_device[device1_id]))
    #     device_id2_experts_names = mlpllm.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=list(gpu_expert_ids_by_device[device2_id]))
    #     tensor_index_names_device_map = {
    #         device1_id: device_id1_experts_names, device2_id: device_id2_experts_names
    #     }
    #     ret, replica_uuid, state_dict = \
    #         mlpllm.cmv.allocate_cuda_memory_and_load_into_gpu_multi_device(
    #             tensor_index_names_device_map=tensor_index_names_device_map
    #         )
    #     mlpllm.cmv.restore2model(state_dict, mlpllm.cmv.mlpm_ci)

    #     device1_expert_idx_list = list(gpu_expert_ids_by_device[device1_id])
    #     device2_expert_idx_list = list(gpu_expert_ids_by_device[device2_id])

        
    #     if device1_id != flat_hidden_states.device.index:
    #         device1_idxs = idxs.to(device1_id)
    #         device1_flat_experts_weight = flat_experts_weight.to(device1_id)
    #         device1_flat_hidden_states = flat_hidden_states.to(device1_id)
    #     else:
    #         device1_idxs = idxs
    #         device1_flat_experts_weight = flat_experts_weight
    #         device1_flat_hidden_states = flat_hidden_states

    #     if device2_id != flat_hidden_states.device.index:
    #         device2_idxs = idxs.to(device2_id)
    #         device2_flat_experts_weight = flat_experts_weight.to(device2_id)
    #         device2_flat_hidden_states = flat_hidden_states.to(device2_id)
    #     else:
    #         device2_idxs = idxs
    #         device2_flat_experts_weight = flat_experts_weight
    #         device2_flat_hidden_states = flat_hidden_states

    #     device1_stacked_inputs = mlpllm.mlpm.experts_func_mgpu_group_pad(
    #         expert_idx_list=device1_expert_idx_list,
    #         expert_indices_map=expert_indices_map,
    #         device_flat_hidden_states=device1_flat_hidden_states,
    #         device_idxs=device1_idxs,
    #     )
        
    #     device2_stacked_inputs = mlpllm.mlpm.experts_func_mgpu_group_pad(
    #         expert_idx_list=device2_expert_idx_list,
    #         expert_indices_map=expert_indices_map,
    #         device_flat_hidden_states=device2_flat_hidden_states,
    #         device_idxs=device2_idxs,
    #     )
    #     group_w1_list, group_w2_list, group_w3_list = mlpllm.mlpm.experts_func_mgpu_group_list(
    #         mi=mlpllm.cmv.mlpm_ci,
    #         layer_idx=layer_idx,
    #         expert_idx_list=list(gpu_expert_ids_by_device[device1_id])
    #     )
    #     dgroup_w1_list, dgroup_w2_list, dgroup_w3_list = mlpllm.mlpm.experts_func_mgpu_group_list(
    #         mi=mlpllm.cmv.mlpm_ci,
    #         layer_idx=layer_idx,
    #         expert_idx_list=list(gpu_expert_ids_by_device[device2_id])
    #     )

    #     mlpllm.cmv.wait_load_into_gpu(replica_uuid=replica_uuid)

    #     cuda_hook_time_end("task_processing_mp_load")

    #     cuda_hook_time("task_processing_mp")

        


    #     group_w1, group_w2, group_w3 = mlpllm.mlpm.experts_func_mgpu_group_experts(
    #         group_w1_list=group_w1_list,
    #         group_w2_list=group_w2_list,
    #         group_w3_list=group_w3_list,
    #     )
    #     dgroup_w1, dgroup_w2, dgroup_w3 = mlpllm.mlpm.experts_func_mgpu_group_experts(
    #         group_w1_list=dgroup_w1_list,
    #         group_w2_list=dgroup_w2_list,
    #         group_w3_list=dgroup_w3_list,
    #     )


    #     dp.submit_worker(
    #         worker_idx=0,
    #         layer_idx=layer_idx,
    #         expert_idx_list=list(gpu_expert_ids_by_device[device1_id]),
    #         expert_indices_map=expert_indices_map,
    #         group_w1=group_w1,
    #         group_w2=group_w2,
    #         group_w3=group_w3,
    #         stacked_inputs=device1_stacked_inputs,
    #         flat_hidden_states=device1_flat_hidden_states,
    #         flat_experts_weight=device1_flat_experts_weight,
    #         idxs=device1_idxs,
    #         final_hidden_states=expert_cache,
    #     )

    #     dp.submit_worker(
    #         worker_idx=1,
    #         layer_idx=layer_idx,
    #         expert_idx_list=list(gpu_expert_ids_by_device[device2_id]),
    #         expert_indices_map=expert_indices_map,
    #         group_w1=dgroup_w1,
    #         group_w2=dgroup_w2,
    #         group_w3=dgroup_w3,
    #         stacked_inputs=device2_stacked_inputs,
    #         flat_hidden_states=device2_flat_hidden_states,
    #         flat_experts_weight=device2_flat_experts_weight,
    #         idxs=device2_idxs,
    #         final_hidden_states=expert_cache,
    #     )

    #     _ = dp.wait()
    #     _ = dp.wait()
    #     # print(output_tensor.shape)
    #     cuda_hook_time_end("task_processing_mp")
    #     mlpllm.cmv.free_allocated()
    # time.sleep(1)
    # dp.stop()

if __name__ == '__main__':
    main()