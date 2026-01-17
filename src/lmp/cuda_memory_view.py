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
    restore_experts_groups_from_shared_memory_profiled,
    restore_experts_groups_from_shared_memory_profiled_cached_ptr,  # 使用缓存的 tensor_metadata
    create_tensor_index_cache,  # 创建 tensor_metadata 缓存
    TensorIndexResizeMapCache,  # 缓存类
    restore_experts_groups_from_shared_memory_cached,  # 缓存版本，复用虚拟地址空间
    release_cached_group_memory,  # 释放缓存的虚拟地址空间
    restore_tensors2,
    free_cuda_memory,
)

from lmp.sllm_store_c import *
from lmp.sllm_store_c import TENSOR_INDEX_RESIZE_PATH, SLLM_ADDRESS, STORAGE_PATH
from models.mlpmodule import MLPModuleWrapper, WeightType
from utils.helper import (
    load_json, 
    calculate_device_offset, 
    get_expert_device_distribution,
    calculate_expert_memory_size,
    filter_experts_by_memory
)
from utils.cuda_h import *
from utils.logger import init_logger
from lmp.sllm_thread_manager import SLLMTM
from lmp.init_meta_manager import InitMetaManager
from lmp.init_meta_manager_mp_shared import InitMetaManagerMPShared
logger = init_logger(__name__)

class CudaMemoryView:
    def __init__(
        self,
        mlpm: MLPModuleWrapper,
        device_list: list[str]
    ):
        self.sllmtm: SLLMTM = None
        self.imm: InitMetaManager =None
        self.mlpm = mlpm

        self.client = SllmStoreClient(SLLM_ADDRESS)
        tensor_index_resize_path = os.path.join(self.mlpm.model_abs_path, TENSOR_INDEX_RESIZE_PATH)
        tensor_index_resize_json = load_json(tensor_index_resize_path)

        mshm_names, chunk_size = self.client.get_model_shared_memory_names(self.mlpm.model_path)
        if len(mshm_names) <= 0:
            raise ValueError(f"Only Support shared memory, But sllm not shared")     

        self.tensor_index_resize_json = tensor_index_resize_json
        self.mchunk_size = chunk_size
                
        self.device_list = device_list
        self.device1_str = device_list[0]
        self.device1 = int(self.device1_str.split(":")[1])
        
        self.device_uuid_map = get_device_uuid_map()

        self.cuda_memory_ptrs_allocated = []
        
        # self.mlpm_ci = self.mlpm.create_empty_model()
        # self.mlpm_ci.eval()
        self.mlpm_ci = None
        
        # 跟踪每层参数是否已全部加载到 GPU
        # {layer_idx: bool} - True 表示该层参数已全部加载到 GPU
        self._layer_loaded_to_gpu = {}

    def load_general_and_init(self):     
        tensor_index_general_names = self.mlpm.get_tensor_index_general_names()
        # tensor_index_attention_names = self.mlpm.get_attention_names(layer_idx=0)
        # tensor_index_layernorm_names = self.mlpm.get_layernorm_names(layer_idx=0)
        # empty expert for 0
        # tensor_experts0_names = self.mlpm.get_experts_names(
        #     layer_idx=0, expert_idx_list=[i for i in range(self.mlpm.config.n_routed_experts)])
        tensor_index_init_names = tensor_index_general_names 
            # + tensor_index_attention_names + tensor_index_layernorm_names + tensor_experts0_names

        ret1, replica_uuid1, state_dict1 = \
            self.allocate_cuda_memory_and_load_into_gpu(tensor_index_init_names, device_index_int=self.device1)

        self.restore2model(state_dict1, self.mlpm_ci)
        self.wait_load_into_gpu(replica_uuid1)
    def start_init_meta_model(self, hmv: "HostMemoryView"):
        self.mlpm.init_chmv_meta_model(cmv=self, hmv=hmv)
        # self.imm.submit_all(
        #     init_func=self.mlpm.init_layer_func,
        #     config=self.mlpm.config,
        # )
    def imm_submit_meta_layer(self, layer_idx: int):
        self.imm.submit_layer(
            layer_idx=layer_idx,
            init_func=self.mlpm.init_layer_func,
            config=self.mlpm.config
        )
    def imm_submit_all(self):
        for layer_idx in range(self.mlpm.config.num_hidden_layers):
            self.imm_submit_meta_layer(layer_idx=layer_idx)
    def imm_wait_meta_layer(self, layer_idx: int):
        if layer_idx >= self.mlpm.config.num_hidden_layers:
            return 
        layer = self.imm.wait_layer(layer_idx=layer_idx)
        print(f"{layer}")
        # set layer to model
        self.mlpm_ci.model.layers[layer_idx] = layer
    def imm_wait_all(self):
        for layer_idx in range(self.mlpm.config.num_hidden_layers):
            self.imm_wait_meta_layer(layer_idx=layer_idx)

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
        tensor_shared_expert_names = self.mlpm.get_shared_experts_names(layer_idx=layer_idx)
        tensor_index_names = tensor_al1_names + tensor_ln1_names + tensor_gate_names + tensor_shared_expert_names
        
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
        
        rend_layer_idx = self.mlpm.config.first_k_dense_replace - 1
        
        for layer_idx in range(rend_layer_idx, self.mlpm.config.num_hidden_layers):
            # 跳过dense 层
            if layer_idx < self.mlpm.config.first_k_dense_replace:
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
                if device == 'meta' or device == 'unknown':
                    cpu_expert_list.append(expert_id)
            
            layer_cpu_experts_map[layer_idx] = cpu_expert_list
            logger.debug(f"Layer {layer_idx}: CPU experts = {cpu_expert_list} (total: {len(cpu_expert_list)}, device_map: {expert_device_map})")
        
        # Step 1.5: 检查多GPU显存并计算所需显存，如果不足则报错，放得下则均匀分配
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
                device_free_memory[device_idx] = free_memory
                total_available_memory += free_memory
                
                logger.debug(f"GPU {device} (device {device_idx_int}) memory status:")
                logger.debug(f"  Total: {memory_info.total / (1024**3):.2f} GB")
                logger.debug(f"  Used: {memory_info.used / (1024**3):.2f} GB")
                logger.debug(f"  Free: {free_memory / (1024**3):.2f} GB")
            except Exception as e:
                raise ValueError(f"Failed to get memory info for device {device}: {e}")
        
        logger.debug(f"Total available memory across all devices: {total_available_memory / (1024**3):.2f} GB")
        logger.debug(f"Total required memory: {total_required_memory / (1024**3):.2f} GB")
        
        # 检查是否能放下
        if total_required_memory > total_available_memory:
            raise RuntimeError(
                f"Insufficient GPU memory across all devices! "
                f"Required: {total_required_memory / (1024**3):.2f} GB, "
                f"Available: {total_available_memory / (1024**3):.2f} GB"
            )
        
        logger.debug("GPU memory is sufficient, will distribute experts evenly across devices.")
        
        # 为每个设备分配expert（每层平均分配）
        layer_cpu_experts_map_by_device = {device_idx: {} for device_idx in range(num_device)}  # {device_idx: {layer_idx: [expert_id, ...]}}
        
        for layer_idx, expert_list in layer_cpu_experts_map.items():
            if not expert_list:
                continue
            
            # 平均分配到各个设备
            total_experts = len(expert_list)
            experts_per_device = total_experts // num_device  # 每个设备的基础expert数量
            remaining_experts = total_experts % num_device  # 剩余的expert数量
            
            # 分配expert到各个设备
            expert_idx = 0
            for device_idx in range(num_device):
                # 前 remaining_experts 个设备多分配一个expert
                count = experts_per_device + (1 if device_idx < remaining_experts else 0)
                
                if count > 0:
                    device_experts = expert_list[expert_idx:expert_idx + count]
                    layer_cpu_experts_map_by_device[device_idx][layer_idx] = device_experts
                    expert_idx += count
                    logger.debug(f"  Device {device_idx} ({self.device_list[device_idx]}): {len(device_experts)} experts")
        
        # 保存分配结果，供等待时使用
        self._layer_cpu_experts_map_by_device = layer_cpu_experts_map_by_device
        
        # Step 2: 从最后一层往前逐层加载，串行提交到多个GPU设备（均匀分配）
        logger.debug("Starting to load CPU experts from last layer to first layer (multi-device serial mode, evenly distributed)...")
        
        for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, rend_layer_idx, -1):
            # 检查是否有任何设备需要加载这一层
            has_experts = False
            for device_idx in range(num_device):
                if layer_idx in layer_cpu_experts_map_by_device[device_idx]:
                    has_experts = True
                    break
            
            if has_experts:
                logger.debug(f"Loading Layer {layer_idx} across {num_device} devices...")
                
                # 串行提交到每个GPU设备（每个设备加载分配给它的expert）
                for device_idx in range(num_device):
                    device = self.device_list[device_idx]
                    device_expert_list = layer_cpu_experts_map_by_device[device_idx].get(layer_idx, [])
                    
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
        rend_layer_idx = self.mlpm.config.first_k_dense_replace - 1
        num_device = len(self.device_list)
        
        # 获取分配结果，确保只等待实际提交了任务的设备
        layer_cpu_experts_map_by_device = getattr(self, '_layer_cpu_experts_map_by_device', {})
        
        # 串行等待每个层的加载完成（每个层可能有多个设备的任务）
        for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, rend_layer_idx, -1):
            # 只等待实际提交了任务的设备
            # 统计该层有多少个设备提交了任务
            tasks_submitted = 0
            for device_idx in range(num_device):
                # 检查该设备在该层是否有expert需要等待
                if layer_idx in layer_cpu_experts_map_by_device.get(device_idx, {}):
                    device_expert_list = layer_cpu_experts_map_by_device[device_idx][layer_idx]
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
        
        # first_k_dense_replace 1, rend_layer_idx 0
        rend_layer_idx = self.mlpm.config.first_k_dense_replace - 1

        for layer_idx in range(rend_layer_idx, self.mlpm.config.num_hidden_layers):
            # 跳过dense 层
            if layer_idx < self.mlpm.config.first_k_dense_replace:
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
        
        # Step 2: 从最后一层往前逐层加载
        logger.debug("Starting to load CPU experts from last layer to first layer...")
        for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, rend_layer_idx, -1):
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
        rend_layer_idx = self.mlpm.config.first_k_dense_replace - 1
        for layer_idx in range(self.mlpm.config.num_hidden_layers - 1, rend_layer_idx, -1):
            # 只等待实际提交了任务的层（如果层已经标记为已加载，则跳过）
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
        if layer_idx != 0:
            raise ValueError(f"layer_idx must be 0")
        device1_idx_int = self.device1
        cuda_hook_time(f"load_qkvogns_weight_l_{layer_idx}")
        tensor_al1_names = self.mlpm.get_attention_names(layer_idx=layer_idx)
        tensor_ln1_names = self.mlpm.get_layernorm_names(layer_idx=layer_idx)
        tensor_gate_names = self.mlpm.get_gate_names(layer_idx=layer_idx)
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

    def restore2model(self, model_state_dict, model):
        cuda_hook_time("restore2model")
        with torch.no_grad():
            for name, param in model_state_dict.items():
                # print(f"{name}: device={param.device}, dtype={param.dtype} shape={param.shape}")
                set_module_tensor_to_device(model, name, param.device, param, clear_cache=False)
        cuda_hook_time_end("restore2model")
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
            use_fixed_gpu_ptrs=False
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

    def free_allocated(self):
        for cuda_memory_ptrs in self.cuda_memory_ptrs_allocated:
            free_cuda_memory(cuda_memory_ptrs)
        self.cuda_memory_ptrs_allocated = []
        # to empty, 需重置以能够重入
        self._layer_loaded_to_gpu = {}
        self._layer_cpu_experts_map_by_device = {}

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

        # 创建 tensor_metadata 缓存，避免每次调用都转换 Python dict -> C++ map
        # 这个转换对于大型字典（5466 个条目，1.3MB）可能很耗时（约 2-3ms）
        # 通过缓存，只在初始化时转换一次，后续调用直接使用缓存的 C++ 对象
        self.tensor_index_cache = create_tensor_index_cache(tensor_index_resize_json)

        # self.mlpm_hi = self.mlpm.create_empty_model()
        # self.mlpm.restore_hm_state_dict2model(self.hm_state_dict, self.mlpm_hi)
        self.mlpm_hi = None

    
    def group_experts_tensor(self, layer_idx: int, expert_idx_list: list[int]):
        cuda_hook_time("get_experts_names_w")
        ewnc1 = self.mlpm.get_experts_names_w(layer_idx, expert_idx_list, type_idx=WeightType.W1)
        ewnc2 = self.mlpm.get_experts_names_w(layer_idx, expert_idx_list, type_idx=WeightType.W2)
        ewnc3 = self.mlpm.get_experts_names_w(layer_idx, expert_idx_list, type_idx=WeightType.W3)
        cuda_hook_time_end("get_experts_names_w")
        # 使用缓存版本，复用虚拟地址空间，避免频繁 mmap/munmap
        # tensor_state_dict = restore_experts_groups_from_shared_memory_cached(
        #     self.mshm_names, self.tensor_index_resize_json,
        #     self.mchunk_size, [ewnc1, ewnc2, ewnc3])
        
        # 一次性调用，不复用
        # tensor_state_dict = restore_experts_groups_from_shared_memory(
        #     self.mshm_names, self.tensor_index_resize_json,
        #     self.mchunk_size, [ewnc1, ewnc2, ewnc3])
        cuda_hook_time("restore_tensors")
        # 使用缓存的 tensor_metadata，避免每次调用都转换 Python dict -> C++ map
        # 这可以显著减少 Python/C++ 绑定开销（特别是对于大型 tensor_index_resize_json）
        tensor_state_dict = restore_experts_groups_from_shared_memory_profiled_cached_ptr(
            self.mshm_names, self.tensor_index_cache,
            self.mchunk_size, [ewnc1, ewnc2, ewnc3])
        cuda_hook_time_end("restore_tensors")

        # cuda_hook_time("rename_dict")
        # # 将 key 从 group_0_big_tensor, group_1_big_tensor, group_2_big_tensor 
        # # 重命名为 group_w1, group_w2, group_w3
        # renamed_dict = {}
        # key_mapping = {
        #     'group_0_big_tensor': 'group_w1',
        #     'group_1_big_tensor': 'group_w2',
        #     'group_2_big_tensor': 'group_w3'
        # }
        # # print(f"{tensor_state_dict}")
        # for key, value in tensor_state_dict.items():
        #     new_key = key_mapping.get(key, key)
        #     renamed_dict[new_key] = value
        # cuda_hook_time("rename_dict_end")

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

        return tensor_state_dict
    
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
