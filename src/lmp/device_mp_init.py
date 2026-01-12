"""
多进程版本的 InitMetaManager - 独立初始化进程版本
使用 torch.multiprocessing 启动多个独立进程初始化所有 DecoderLayer
主进程从这些进程获取初始化好的 DecoderLayer 对象
torch.multiprocessing 支持 CUDA 张量的共享
支持配置进程数量，多个进程共享任务队列（工作窃取模式）
"""
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
import torch
import queue
from typing import Optional, Dict, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from utils.cuda_h import *

# 在模块级别设置启动方法为 'spawn'
# 这对于 CUDA 多进程是必需的
# 注意：这必须在导入模块时设置，且只能在主进程中设置一次
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    # 已经设置过了，检查是否是 'spawn'
    current_method = mp.get_start_method(allow_none=True)
    if current_method != 'spawn':
        # 如果不是 'spawn'，尝试强制设置
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果强制设置也失败，说明已经有进程在运行
            # 这种情况下，用户需要在主脚本中手动设置
            pass

@dataclass
class DeviceOutput:
    output_tensor: torch.Tensor

@dataclass
class InitRequest:
    """初始化请求"""
    layer_idx: int
    expert_idx_list: list[int]
    expert_indices_map: dict[int, tuple[int, int]]
    # expert_token_indices_map: dict[int, torch.Tensor]
    flat_hidden_states: torch.Tensor
    flat_experts_weight: torch.Tensor
    idxs: torch.Tensor
    final_hidden_states: torch.Tensor
    if_decode: bool = False

def _init_process_func(input_queue: Queue, output_queue: Queue, device_list_idx: int, layer_idx: int, exit_event):
    """
    独立初始化进程主函数
    
    从队列获取初始化请求，初始化 DecoderLayer，然后通过队列发送给主进程
    进程会一直运行直到收到退出信号
    
    Args:
        input_queue: 输入队列（从主进程接收初始化请求，包含函数和参数）
        output_queue: 输出队列（向主进程传递初始化好的 layer）
        exit_event: 退出事件（主进程设置此事件来通知初始化进程退出）
    """
    # 导入必要的模块（在初始化进程中）
    import sys
    import os
    import time
    import queue as queue_module
    import torch
    from utils.logger import init_logger
    from utils.cuda_h import cuda_hook_time, cuda_hook_time_end
    
    from sllm_store._C import (
        allocate_cuda_memory,
    )
    logger = init_logger(__name__)

    # ===== 关键修复：在创建任何可能使用 CUDA 的对象之前，先初始化 CUDA =====
    # 在子进程中，CUDA 需要在使用之前显式初始化
    # 这是因为子进程不会自动继承父进程的 CUDA 上下文
    # 如果父进程在使用 CUDA，子进程可能无法初始化，所以需要先初始化
    
    # 设置环境变量，帮助 CUDA 初始化
    # 这可以避免某些 CUDA 初始化问题
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    
    

    if torch.cuda.is_available():
        # 方法1: 先检查 CUDA 是否已初始化
        # 在子进程中，CUDA 通常还没有初始化
        if not torch.cuda.is_initialized():
            # 尝试初始化 CUDA
            # 注意：如果父进程在使用 CUDA，这里可能会失败
            try:
                logger.debug(f"初始化")
                torch.cuda.init()
            except RuntimeError as e:
                logger.warning(f"Init process {os.getpid()}: torch.cuda.init() failed: {e}")
                # 继续尝试，可能通过创建张量可以初始化
    from lmp.lmp import MLPLLM
    # 现在 CUDA 已经初始化，可以安全地创建 MLPLLM 对象
    model_path = "deepseek-moe-16b-base-bfloat16"
    model_name_type = "Deepseek"

    mlpllm =MLPLLM( model_name_type=model_name_type, model_path=model_path )
    mlpllm.cmv.start_init_meta_model(hmv=mlpllm.hmv)

    # 先初始化 CUDA（在创建 MLPLLM 之前）
    # 默认使用设备 1（根据后续代码中的 device_memory 使用设备 1）
    # 注意：这里我们先使用一个默认设备，稍后再根据 mlpllm.device1 调整
    device_str = mlpllm.device_list[device_list_idx]  # 默认设备，稍后会根据 mlpllm.device1 更新

    # 根据 mlpllm.device1 更新设备索引
    device_index = int(device_str.split(":")[1])
    
    # 如果设备索引改变了，需要重新设置设备
    # if torch.cuda.is_available() and torch.cuda.current_device() != device_index:
    #     torch.cuda.set_device(device_index)
    #     torch.cuda.synchronize(device=device_index)
    #     logger.info(f"Init process {os.getpid()}: Switched to CUDA device {device_index}")
    layer_idx = layer_idx

    gpu_experts_names = mlpllm.mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=[i for i in range(0, 64)])
    ret, replica_uuid, state_dict = \
        mlpllm.cmv.allocate_cuda_memory_and_load_into_gpu(
            gpu_experts_names, device_index_int=device_index
        )
    mlpllm.cmv.wait_load_into_gpu(replica_uuid=replica_uuid)
    mlpllm.cmv.restore2model(state_dict, mlpllm.cmv.mlpm_ci)

    local_list = list()
    # 持续运行，从队列获取初始化请求
    while not exit_event.is_set():
        # try:
            request: Optional[InitRequest] = input_queue.get()
            
            # 处理初始化请求
            # try:    
            cuda_hook_time("experts_func_gpu_einsum_mp")
            
                
            # expert0 = mlpllm.cmv.mlpm_ci.model.layers[1].mlp.experts[0]
            # out = expert0(request.flat_hidden_states)
            # device_flat_hidden_states = request.flat_hidden_states
            # device_idxs = request.idxs // mlpllm.cmv.mlpm_ci.config.num_experts_per_tok
            # device_expert_token_indices_map = {
            #     eid: device_idxs[request.expert_indices_map[eid][0]:request.expert_indices_map[eid][1]]
            #     for eid in request.expert_idx_list
            # }
            # device_flat_hidden_states = device_flat_hidden_states.to(device="cuda:1", non_blocking=True)
            # logger.debug(f"dtype {request.flat_hidden_states.dtype}")
            # for i in request.expert_idx_list:
            #     expert = mlpllm.cmv.mlpm_ci.model.layers[1].mlp.experts[i]
            #     tokens = device_flat_hidden_states[device_expert_token_indices_map[i]]
            #     logger.debug(f"tokens dtype {tokens.dtype} tokens shape {tokens.shape} "
            #         f"tokens device {tokens.device}"
            #     )
            #     logger.debug(f"device_flat_hidden_states dtype {device_flat_hidden_states.dtype} tokens shape {device_flat_hidden_states.shape} "
            #         f"device_flat_hidden_states device {device_flat_hidden_states.device}"
            #     )
            #     out = expert(device_flat_hidden_states)
            #     # out = expert(tokens)
            #     print(f"expert {i} out: {out.shape}")

            with torch.no_grad():
                device = device_str  # 使用 device1，确保与模型权重在同一设备上
                
                device_flat_hidden_states = request.flat_hidden_states
                device_flat_experts_weight = request.flat_experts_weight
                device_idxs = request.idxs // mlpllm.cmv.mlpm_ci.config.num_experts_per_tok
                
                device_flat_hidden_states = request.flat_hidden_states.to(device, non_blocking=True)
                device_flat_experts_weight = request.flat_experts_weight.to(device, non_blocking=True)
                device_idxs = device_idxs.to(device, non_blocking=True)

                # 同步数据传输，确保所有张量都已到达设备
                torch.cuda.synchronize(device=device_index)
                
                device_expert_token_indices_map = {
                    eid: device_idxs[request.expert_indices_map[eid][0]:request.expert_indices_map[eid][1]]
                    for eid in request.expert_idx_list
                }
                device_expert_cache = torch.zeros_like(device_flat_hidden_states, device=device)

                # tokens = flat_hidden_states[expert_token_indices_map[expert_id]]
                

                logger.debug(
                    f"device_idx device: {device_index}, device_idxs.device: {device_idxs.device}"
                    f"device_flat_hidden_states.device: {device_flat_hidden_states.device}"
                    f"device_flat_experts_weight.device: {device_flat_experts_weight.device}"
                    f"device_idxs.device: {device_idxs.device}"
                    f"device_expert_cache.device: {device_expert_cache.device}"
                )
                device_expert_cache = mlpllm.mlpm.experts_func_gpu_einsum(
                    mi=mlpllm.cmv.mlpm_ci,
                    layer_idx=request.layer_idx,
                    expert_idx_list=request.expert_idx_list,
                    expert_indices_map=request.expert_indices_map,
                    expert_token_indices_map=device_expert_token_indices_map,
                    flat_hidden_states=device_flat_hidden_states,
                    flat_experts_weight=device_flat_experts_weight,
                    idxs=device_idxs,
                    final_hidden_states=device_expert_cache
                )
                # device_expert_cache = mlpllm.mlpm.experts_func(
                #     mi=mlpllm.cmv.mlpm_ci,
                #     layer_idx=request.layer_idx,
                #     expert_idx_list=request.expert_idx_list,
                #     expert_indices_map=request.expert_indices_map,
                #     expert_token_indices_map=device_expert_token_indices_map,
                #     flat_hidden_states=device_flat_hidden_states,
                #     flat_experts_weight=device_flat_experts_weight,
                #     idxs=device_idxs,
                #     final_hidden_states=device_expert_cache,
                #     device=device  # 显式传递设备参数
                # )
                
                # 关键：同步所有 CUDA 操作，确保 einsum 计算完成
                # 这对于跨进程传递 CUDA 张量非常重要
                torch.cuda.synchronize(device=device_index)
                
                output_tensor = device_expert_cache


                output_queue.put(DeviceOutput(output_tensor=output_tensor))
            cuda_hook_time_end("experts_func_gpu_einsum_mp")
        #     except Exception as e:
        #         print(f"Init process {os.getpid()}: Failed to write result: {e}")
                
        # except Exception as e:
        #     print(f"Init process {os.getpid()}: Error in main loop: {e}")
                
class DeviceMP:
    """
    多进程版本的 DeviceManager
    使用 torch.multiprocessing 启动多个独立进程初始化所有 DecoderLayer
    主进程从这些进程获取初始化好的 DecoderLayer 对象
    torch.multiprocessing 支持 CUDA 张量的共享
    支持配置进程数量，多个进程共享任务队列（工作窃取模式）
    """
    def __init__(self, num_processes: int = 1):
        self.num_processes = num_processes
        self.input_queue = []
        for i in range(num_processes):
            self.input_queue.append(Queue())
        self.output_queue = Queue()
        self.exit_event = mp.Event()

    def submit(self,
        layer_idx: int,
        expert_idx_list: list[int],
        expert_indices_map: dict[int, tuple[int, int]],
        flat_hidden_states: torch.Tensor,
        flat_experts_weight: torch.Tensor,
        idxs: torch.Tensor,
        final_hidden_states: torch.Tensor,
        if_decode: bool = False
    ):        
        self.input_queue[0].put(
            InitRequest(
                layer_idx=layer_idx, 
                expert_idx_list=expert_idx_list, 
                expert_indices_map=expert_indices_map, 
                flat_hidden_states=flat_hidden_states, 
                flat_experts_weight=flat_experts_weight, idxs=idxs, 
                final_hidden_states=final_hidden_states, if_decode=if_decode
            )
        )
    def submit_worker(self,
        worker_idx: int,
        layer_idx: int,
        expert_idx_list: list[int],
        expert_indices_map: dict[int, tuple[int, int]],
        flat_hidden_states: torch.Tensor,
        flat_experts_weight: torch.Tensor,
        idxs: torch.Tensor,
        final_hidden_states: torch.Tensor,
        if_decode: bool = False
    ):
        self.input_queue[worker_idx].put(
            InitRequest(
                layer_idx=layer_idx, 
                expert_idx_list=expert_idx_list, 
                expert_indices_map=expert_indices_map, 
                flat_hidden_states=flat_hidden_states, 
                flat_experts_weight=flat_experts_weight, idxs=idxs, 
                final_hidden_states=final_hidden_states, if_decode=if_decode
            )
        )

    def wait(self):
        """
        等待初始化完成
        """
        ot =  self.output_queue.get()
        output_tensor = ot.output_tensor
        return output_tensor

    def start(self):
        """
        启动多进程设备管理器
        """
        # 验证启动方法是否为 'spawn'
        # 启动方法应该在模块级别已经设置（见文件顶部）
        # 这里只是验证，确保使用的是 'spawn' 方法
        current_method = mp.get_start_method(allow_none=True)
        if current_method != 'spawn':
            raise RuntimeError(
                f"Multiprocessing start method must be 'spawn' for CUDA, but got '{current_method}'. "
                "Please set it before importing this module or creating DeviceMP instance. "
                "Add this at the beginning of your main script (before any imports):\n"
                "  import torch.multiprocessing as mp\n"
                "  mp.set_start_method('spawn')\n"
            )
        
        # 重要：在创建子进程之前，确保父进程释放所有 CUDA 资源
        # 这可以避免子进程 CUDA 初始化失败
        import torch
        if torch.cuda.is_available():
            # 同步所有 CUDA 操作，确保没有未完成的操作
            torch.cuda.synchronize()
            # 清理 CUDA 缓存（可选，但可以帮助释放资源）
            # torch.cuda.empty_cache()
        
        self.processes = []
        for i in range(self.num_processes):
            process = Process(target=_init_process_func, args=(self.input_queue[i], self.output_queue, i, i+1, self.exit_event))
            process.start()
            self.processes.append(process)

    def stop(self):
        """
        停止多进程设备管理器
        """
        # 设置退出事件，通知所有进程退出
        self.exit_event.set()
        
        # 等待所有进程结束
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                # 如果进程还在运行，强制终止
                process.terminate()
                process.join()
        
        self.processes = []