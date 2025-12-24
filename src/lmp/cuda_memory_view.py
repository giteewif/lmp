import os
import time
import torch
import torch.nn.functional as F
from typing import Dict
from transformers import AutoConfig
from transformers.activations import ACT2FN
from accelerate.utils import set_module_tensor_to_device

from sllm_store.client import SllmStoreClient
from sllm_store._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors_from_shared_memory_names,
    restore_experts_tensor_from_shared_memory,
    restore_experts_groups_from_shared_memory,
    restore_experts_groups_from_shared_memory_cached,  # 缓存版本，复用虚拟地址空间
    release_cached_group_memory,  # 释放缓存的虚拟地址空间
    restore_tensors2,
    free_cuda_memory,
)

from lmp.sllm_store_c import *
from lmp.sllm_store_c import TENSOR_INDEX_RESIZE_PATH, SLLM_ADDRESS, STORAGE_PATH
from models.mlpmodule import MLPModuleWrapper, WeightType
from utils.helper import load_json, calculate_device_offset
from utils.cuda_h import *
from utils.logger import init_logger
from lmp.pinpool import gpinpool
logger = init_logger(__name__)

class CudaMemoryView:
    def __init__(
        self,
        mlpm: MLPModuleWrapper,
    ):
        self.client = SllmStoreClient(SLLM_ADDRESS)
        self.mlpm = mlpm
        tensor_index_resize_path = os.path.join(self.mlpm.model_abs_path, TENSOR_INDEX_RESIZE_PATH)
        tensor_index_resize_json = load_json(tensor_index_resize_path)

        mshm_names, chunk_size = self.client.get_model_shared_memory_names(self.mlpm.model_path)
        if len(mshm_names) <= 0:
            raise ValueError(f"Only Support shared memory, But sllm not shared")     

        self.tensor_index_resize_json = tensor_index_resize_json
        self.mchunk_size = chunk_size
        
        self.device1 = 1
        self.device2 = 2
        self.device_uuid_map = get_device_uuid_map()

        self.cuda_memory_ptrs_allocated = []
        # for deepseek
        self.load_general_and_init()

    def load_general_and_init(self):     
        tensor_index_general_names = self.mlpm.get_tensor_index_general_names()
        tensor_index_attention_names = self.mlpm.get_attention_names(layer_idx=0)
        tensor_index_layernorm_names = self.mlpm.get_layernorm_names(layer_idx=0)
        # empty expert for 0
        tensor_experts0_names = self.mlpm.get_experts_names(
            layer_idx=0, expert_idx_list=[i for i in range(self.mlpm.config.n_routed_experts)])
        tensor_index_init_names = tensor_index_general_names + tensor_index_attention_names  \
                + tensor_index_layernorm_names + tensor_experts0_names

        ret1, replica_uuid1, state_dict1 = \
            self.allocate_cuda_memory_and_load_into_gpu(tensor_index_init_names, device_index_int=self.device1)

        self.mlpm_ci = self.mlpm.create_empty_model()
        self.mlpm_ci.eval()
        
        self.restore2model(state_dict1, self.mlpm_ci)
        self.wait_load_into_gpu(replica_uuid1)
        
        
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

    def restore2model(self, model_state_dict, model):
        with torch.no_grad():
            for name, param in model_state_dict.items():
                # print(f"{name}: device={param.device}, dtype={param.dtype} shape={param.shape}")
                set_module_tensor_to_device(model, name, param.device, param, clear_cache=False)
        
    def allocate_cuda_memory_and_load_into_gpu(self, tensor_index_names: list[str], device_index_int: int):
        cuda_hook_time("allocate_cuda_memory_and_load_into_gpu")
        tensor_meta_index, tensor_data_index, tensor_device_offsets, tensor_copy_chunks, tensor_device_size = \
            self.get_meta_data_offsets_and_copy_chunks(tensor_index_names, device_index_int)
        device_memory = {
            device_index_int: tensor_device_size
        }
        cuda_hook_time("allocate_cuda_memory")
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
            use_fixed_gpu_ptrs=False
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
        
class HostMemoryView:
    def __init__(
        self, 
        mlpm: MLPModuleWrapper,
    ):
        self.client = SllmStoreClient(SLLM_ADDRESS)

        self.mlpm = mlpm
        tensor_index_resize_path = os.path.join(self.mlpm.model_abs_path, TENSOR_INDEX_RESIZE_PATH)
        tensor_index_resize_json = load_json(tensor_index_resize_path)

        mshm_names, chunk_size = self.client.get_model_shared_memory_names(self.mlpm.model_path)
        if len(mshm_names) <= 0:
            raise ValueError(f"Only Support shared memory, But sllm not shared")

        time_start_restore = time.time()
        self.hm_state_dict = restore_tensors_from_shared_memory_names(
                                mshm_names, tensor_index_resize_json, chunk_size)
        logger.debug(f"\nrestore_tensors_from_shared_memory_names time: {time.time() - time_start_restore}")

        self.mshm_names = mshm_names
        self.tensor_index_resize_json = tensor_index_resize_json
        self.mchunk_size = chunk_size

        self.mlpm_hi = self.mlpm.create_empty_model()
        self.mlpm.restore_hm_state_dict2model(self.hm_state_dict, self.mlpm_hi)

        # self.test_restore_experts_tensor_from_shared_memory()
        # self.test_restore_group_experts_tensor_from_shared_memory()
    
    def group_experts_tensor(self, layer_idx: int, expert_idx_list: list[int]):
        ewnc1 = self.mlpm.get_experts_names_w(layer_idx, expert_idx_list, type_idx=WeightType.W1)
        ewnc2 = self.mlpm.get_experts_names_w(layer_idx, expert_idx_list, type_idx=WeightType.W2)
        ewnc3 = self.mlpm.get_experts_names_w(layer_idx, expert_idx_list, type_idx=WeightType.W3)
        
        # 使用缓存版本，复用虚拟地址空间，避免频繁 mmap/munmap
        # tensor_state_dict = restore_experts_groups_from_shared_memory_cached(
        #     self.mshm_names, self.tensor_index_resize_json,
        #     self.mchunk_size, [ewnc1, ewnc2, ewnc3])
        
        # 一次性调用，不复用
        tensor_state_dict = restore_experts_groups_from_shared_memory(
            self.mshm_names, self.tensor_index_resize_json,
            self.mchunk_size, [ewnc1, ewnc2, ewnc3])

        # 将 key 从 group_0_big_tensor, group_1_big_tensor, group_2_big_tensor 
        # 重命名为 group_w1, group_w2, group_w3
        renamed_dict = {}
        key_mapping = {
            'group_0_big_tensor': 'group_w1',
            'group_1_big_tensor': 'group_w2',
            'group_2_big_tensor': 'group_w3'
        }
        for key, value in tensor_state_dict.items():
            new_key = key_mapping.get(key, key)
            renamed_dict[new_key] = value

        # 单次调用
        # group_w1 = restore_experts_tensor_from_shared_memory(
        #     self.mshm_names, self.tensor_index_resize_json,
        #     self.mchunk_size, ewnc1
        # )
        # group_w2 = restore_experts_tensor_from_shared_memory(
        #     self.mshm_names, self.tensor_index_resize_json,
        #     self.mchunk_size, ewnc2
        # )
        # group_w3 = restore_experts_tensor_from_shared_memory(
        #     self.mshm_names, self.tensor_index_resize_json,
        #     self.mchunk_size, ewnc3
        # )
        
        # renamed_dict = {
        #     'group_w1': group_w1["big_tensor"],
        #     'group_w2': group_w2["big_tensor"],
        #     'group_w3': group_w3["big_tensor"]
        # }
        
        


        return renamed_dict
    
    def test_restore_group_experts_tensor_from_shared_memory(self):
        expert_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        experts_names_continuous1 = self.mlpm.get_experts_names_w(
            1, expert_idx_list, type_idx=WeightType.W1)
        experts_names_continuous2 = self.mlpm.get_experts_names_w(
            1, expert_idx_list, type_idx=WeightType.W2)
        experts_names_continuous3 = self.mlpm.get_experts_names_w(
            1, expert_idx_list, type_idx=WeightType.W3)
        
        time_start_restore = time.time()
        cuda_hook("restore experts")
        tensor_state_dict = restore_experts_groups_from_shared_memory(
            self.mshm_names, self.tensor_index_resize_json,
            self.mchunk_size, [experts_names_continuous1, experts_names_continuous2, experts_names_continuous3])
        cuda_hook_end("restore experts")

        logger.debug(
            f"restore time: {time.time() - time_start_restore}, "
        )

    def test_restore_experts_tensor_from_shared_memory(self):
        layer_idx = 1
        expert_idx_list = [0, 1, 2, 3, 4, 5]
        experts_names_continuous = self.mlpm.get_experts_names_w(
            layer_idx, expert_idx_list, type_idx=WeightType.W1)
        
        time_start_restore = time.time()

        cuda_hook("restore experts")
        experts_state_dict = restore_experts_tensor_from_shared_memory(
            self.mshm_names, self.tensor_index_resize_json,
            self.mchunk_size, experts_names_continuous)
        cuda_hook_end("restore experts")
        # C++ 层已经创建了 big_tensor，直接使用
        # big_tensor 的 key 是第一个 tensor 名称 + "_big_tensor"
        first_tensor_name = experts_names_continuous[0]
        big_tensor_key = first_tensor_name + "_big_tensor"
        
        if big_tensor_key in experts_state_dict:
            # 直接使用 C++ 创建的 big_tensor
            big_tensor = experts_state_dict[big_tensor_key]
            logger.debug(
                f"restore time: {time.time() - time_start_restore}, "
                f"big_tensor shape: {big_tensor.shape}, "
                f"big_tensor stride: {big_tensor.stride()}"
            )
        else:
            # 如果没有 big_tensor，回退到使用 try_stack_without_copy
            experts_tensors_list = []
            for i in experts_names_continuous:
                experts_tensors_list.append(experts_state_dict[i])
            big_tensor = HostMemoryView.try_stack_without_copy(experts_tensors_list)
            logger.debug(
                f"restore time: {time.time() - time_start_restore}, "
                f"fallback to try_stack_without_copy, big_tensor shape: {big_tensor.shape}"
            )

    def try_stack_without_copy(weights_list):
        """
        尝试在不复制数据的情况下创建 stacked tensor。
        要求：tensor 必须在连续地址空间中，且按 weights_list 的顺序排列。
        
        如果无法满足条件，抛出 ValueError。
        """
        if len(weights_list) == 0:
            raise ValueError("weights_list cannot be empty")
        
        first = weights_list[0]
        
        # 检查所有 tensor 的形状和类型是否一致
        if not all(w.shape == first.shape and w.dtype == first.dtype for w in weights_list):
            raise ValueError("All tensors must have the same shape and dtype")
        
        # 获取所有 tensor 的数据指针（保持原始顺序）
        data_ptrs = [w.data_ptr() for w in weights_list]
        
        # 检查地址是否连续（按照 weights_list 的顺序）
        element_size = first.element_size()
        tensor_size = first.numel() * element_size
        
        # 检查原始顺序是否连续
        is_contiguous_in_order = True
        for i in range(len(data_ptrs) - 1):
            expected_next = data_ptrs[i] + tensor_size
            if data_ptrs[i + 1] != expected_next:
                is_contiguous_in_order = False
                break
        
        # 如果原始顺序不连续，检查排序后是否连续（用于诊断）
        if not is_contiguous_in_order:
            data_ptrs_sorted = sorted(data_ptrs)
            is_contiguous_sorted = True
            for i in range(len(data_ptrs_sorted) - 1):
                expected_next = data_ptrs_sorted[i] + tensor_size
                if data_ptrs_sorted[i + 1] != expected_next:
                    is_contiguous_sorted = False
                    break
            
            if is_contiguous_sorted:
                raise ValueError(
                    "Tensors are contiguous in memory but not in the order of weights_list. "
                    "Cannot use as_strided without copying (would cause index misalignment)."
                )
            else:
                raise ValueError(
                    "Tensors are not contiguous in memory. Cannot use as_strided without copying."
                )
        
        # 原始顺序连续，可以使用 as_strided
        first_tensor = weights_list[0]
        E = len(weights_list)
        
        # 只支持 2D tensor
        if len(first_tensor.shape) != 2:
            raise ValueError(
                f"Only 2D tensors are supported, got shape {first_tensor.shape}"
            )
        
        dim0, dim1 = first_tensor.shape[0], first_tensor.shape[1]
        big_shape = (E, dim0, dim1)
        
        # 计算 stride：确保与 torch.stack 创建的 tensor 的 stride 相同
        # torch.stack 创建的 tensor stride 取决于原始 tensor 的形状
        # 对于 [dim0, dim1] 形状的 tensor，stack 后的 stride 是 [dim0*dim1, dim1, 1]
        # 
        # 注意：stride 是以元素为单位的，不是字节
        # stride[0] = dim0*dim1 (从第 i 个 tensor 到第 i+1 个 tensor)
        # stride[1] = dim1 (从第 j 行到第 j+1 行，与原始 tensor 相同)
        # stride[2] = 1 (每个元素之间的 stride)
        big_stride = (dim0 * dim1, dim1, 1)
        
        # 创建大 tensor（共享底层存储）
        # 使用 as_strided 创建，指向第一个 tensor 的地址
        # 这避免了数据复制，但 stride 与 stack 相同，所以计算效率相同
        big_tensor = torch.as_strided(
            first_tensor,
            size=big_shape,
            stride=big_stride
        )
        
        # 验证：检查 big_tensor 的每个切片是否对应原始 tensor（按顺序）
        for i in range(E):
            if big_tensor[i].data_ptr() != weights_list[i].data_ptr():
                raise ValueError(
                    f"Verification failed: big_tensor[{i}] does not match weights_list[{i}]. "
                    f"Expected data_ptr {weights_list[i].data_ptr()}, got {big_tensor[i].data_ptr()}"
                )
        
        # 验证通过，返回 big_tensor（避免复制）
        return big_tensor
