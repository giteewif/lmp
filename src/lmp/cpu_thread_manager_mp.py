import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
import torch
import queue
from typing import Optional, Dict, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from utils.cuda_h import *
from models.mlpmodule import ExpertEinsumResult

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
class CPUExpertsInput:
    layer_idx: int
    expert_idx_list: list[int]
    expert_indices_map: dict[int, tuple[int, int]]
    flat_hidden_states: torch.Tensor
    flat_experts_weight: torch.Tensor
    idxs: torch.Tensor
    final_hidden_states: torch.Tensor
    if_decode: bool = False

def _cpu_experts_worker(
    input_queue: Queue, 
    output_queue: Queue, 
    model_path: str, 
    model_name_type: str, 
    exit_event
):
    """
    CPU experts worker 进程主函数
    
    从队列获取 CPU experts 计算请求，执行计算，然后通过队列发送结果给主进程
    进程会一直运行直到收到退出信号
    
    Args:
        input_queue: 输入队列（从主进程接收计算请求）
        output_queue: 输出队列（向主进程传递计算结果）
        model_path: 模型路径
        model_name_type: 模型名称类型
        exit_event: 退出事件（主进程设置此事件来通知工作进程退出）
    """
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

    from lmp.lmp import MLPLLM
    mlpllm = MLPLLM(model_name_type=model_name_type, model_path=model_path)
    
    
    while not exit_event.is_set():
      
        input_data: CPUExpertsInput = input_queue.get()

        cuda_hook_time("experts_func_gpu_einsum_mp")

        expert_idx_list = input_data.expert_idx_list
        expert_indices_map = input_data.expert_indices_map
        expert_token_indices_map = {}
        token_idxs = input_data.idxs // mlpllm.mlpm.config.num_experts_per_tok
        for expert_id in expert_idx_list:
            expert_token_indices_map[expert_id] = token_idxs[expert_indices_map[expert_id][0]:expert_indices_map[expert_id][1]]
        idxs = input_data.idxs
        flat_hidden_states = input_data.flat_hidden_states
        flat_experts_weight = input_data.flat_experts_weight
        final_hidden_states = input_data.final_hidden_states
        
        # 使用一个临时的本地队列，因为 experts_func_einsum 需要 output_queue 参数
        # 但结果已经通过 in-place 修改写入到 final_hidden_states 中
        # output_queue_tmp = queue.Queue()
        _ = mlpllm.mlpm.experts_func_einsum_mp(
            hmv=mlpllm.hmv,
            layer_idx=input_data.layer_idx,
            expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            expert_token_indices_map=expert_token_indices_map,
            flat_hidden_states=flat_hidden_states,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            final_hidden_states=final_hidden_states,
            output_queue=output_queue
        )
        
        # 关键：final_hidden_states 是通过共享内存传递的，子进程的 in-place 修改
        # (scatter_reduce_) 已经直接反映到主进程的原始张量中
        # 因此不需要再通过队列返回结果，只需要发送一个完成标记
        
        # 重要：必须从 output_queue_tmp 获取结果，即使不使用它
        # 因为 experts_func_einsum 内部会执行 output_queue.put(result)，
        # 如果不获取，put 操作可能会阻塞（如果队列满了），导致后续代码无法执行
        # einsum_result = output_queue_tmp.get()
        # 不使用 einsum_result，因为结果已经在 final_hidden_states 中
        
        # 只发送完成标记，不发送张量（主进程已经有结果了）
        # 使用 None 或一个简单的标记表示计算完成
        # output_queue.put(None)
        
        cuda_hook_time_end("experts_func_gpu_einsum_mp")
        # 注意：不要 return，继续循环处理下一个任务


class CPUExpertsManagerMP:
    """
    多进程版本的 CPU Experts Manager
    使用 torch.multiprocessing 启动多个独立进程处理 CPU experts 计算
    主进程从这些进程获取计算结果
    torch.multiprocessing 支持 CUDA 张量的共享
    支持配置进程数量，多个进程共享任务队列（工作窃取模式）
    """
    def __init__(self, num_workers: int = 1, model_path: str = "deepseek-moe-16b-base-bfloat16", model_name_type: str = "Deepseek"):
        self.num_workers = num_workers
        self.input_queues = []
        for i in range(num_workers):
            self.input_queues.append(Queue())
        self.output_queue = Queue()
        self.exit_event = mp.Event()
        self.model_path = model_path
        self.model_name_type = model_name_type
        self.processes = []

    def submit(
        self,
        layer_idx: int,
        expert_idx_list: list[int],
        expert_indices_map: dict[int, tuple[int, int]],
        flat_hidden_states: torch.Tensor,
        flat_experts_weight: torch.Tensor,
        idxs: torch.Tensor,
        final_hidden_states: torch.Tensor,
    ):
        """
        提交任务到第一个 worker 的队列
        """
        input_data = CPUExpertsInput(
            layer_idx=layer_idx,
            expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            final_hidden_states=final_hidden_states,
        )
        self.input_queues[0].put(input_data)
    
    def submit_worker(
        self, worker_idx: int, 
        layer_idx: int,
        expert_idx_list: list[int],
        expert_indices_map: dict[int, tuple[int, int]],
        flat_hidden_states: torch.Tensor,
        flat_experts_weight: torch.Tensor,
        idxs: torch.Tensor,
        final_hidden_states: torch.Tensor,
    ):
        """
        提交任务到指定 worker 的队列
        
        Args:
            worker_idx: worker 索引
            input_data: CPU experts 输入数据
        """
        input_data = CPUExpertsInput(
            layer_idx=layer_idx,
            expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            flat_experts_weight=flat_experts_weight,
            idxs=idxs,
            final_hidden_states=final_hidden_states,
        )
        
        if worker_idx >= len(self.input_queues):
            raise ValueError(f"Worker index {worker_idx} out of range [0, {len(self.input_queues)})")
        self.input_queues[worker_idx].put(input_data)
        return input_data
    
    def wait(self):
        """
        等待计算完成
        
        注意：计算结果已经通过 in-place 修改写入到主进程传入的 final_hidden_states 中，
        这里只需要等待子进程完成计算并返回完成标记。
        
        Returns:
            None（计算结果已经在主进程的 final_hidden_states 中）
        """
        result = self.output_queue.get()
        if isinstance(result, Exception):
            raise result
        # result 应该是 None（完成标记），计算结果已经在主进程的 final_hidden_states 中
        return result
    
    def start(self):
        """
        启动多进程 CPU experts 管理器
        """
        # 验证启动方法是否为 'spawn'
        current_method = mp.get_start_method(allow_none=True)
        if current_method != 'spawn':
            raise RuntimeError(
                f"Multiprocessing start method must be 'spawn' for CUDA, but got '{current_method}'. "
                "Please set it before importing this module or creating CPUExpertsManagerMP instance. "
                "Add this at the beginning of your main script (before any imports):\n"
                "  import torch.multiprocessing as mp\n"
                "  mp.set_start_method('spawn')\n"
            )
        
        # 重要：在创建子进程之前，确保父进程释放所有 CUDA 资源
        import torch
        if torch.cuda.is_available():
            # 同步所有 CUDA 操作，确保没有未完成的操作
            torch.cuda.synchronize()
        
        self.processes = []
        for i in range(self.num_workers):
            process = Process(
                target=_cpu_experts_worker, 
                args=(self.input_queues[i], self.output_queue, self.model_path, self.model_name_type, self.exit_event)
            )
            process.start()
            self.processes.append(process)
    
    def stop(self):
        """
        停止多进程 CPU experts 管理器
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