import queue
import threading

import torch
from typing import Optional, List, Any
from dataclasses import dataclass

from lmp.cuda_memory_view import CudaMemoryView
from utils.cuda_h import cuda_hook_time, cuda_hook_time_end

@dataclass
class LoadTask:
    """SLLM 加载任务的输入"""
    layer_idx: int
    tensor_index_names: List[str]
    device_index_int: int


@dataclass
class LoadResult:
    """SLLM 加载任务的结果"""
    layer_idx: int
    ret1: Any
    replica_uuid: str


class SLLMTM:
    """
    SLLM Thread Manager - 专门用于向 client 发起 load 请求和 wait 请求的线程管理器
    
    支持多个 worker 并行处理加载请求，异步执行加载操作。
    """
    def __init__(self, cmv: CudaMemoryView, num_workers: int = 1):
        """
        Args:
            cmv: CudaMemoryView 实例
            num_workers: worker 线程数量，默认为 1
        """
        self.cmv = cmv
        self.num_workers = num_workers
        self.running = False
        self.threads = []
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
    def _worker(self, worker_id: int):
        """工作线程主循环"""
        while self.running:
            # 从队列获取任务
            task: Optional[LoadTask] = self.input_queue.get()
            
            # 如果收到 None，表示停止信号
            if task is None:
                break
            
            try:
                cuda_hook_time("sllm_worker_task")
                # 执行异步加载请求
                ret1, replica_uuid, state_dict = self.cmv.allocate_cuda_memory_and_load_into_gpu(
                    tensor_index_names=task.tensor_index_names,
                    device_index_int=task.device_index_int
                )
                # 将权重恢复到模型中（类似 allocate_cuda_memory_load_wait）
                self.cmv.restore2model(state_dict, self.cmv.mlpm_ci)
                # 将结果放入输出队列
                result = LoadResult(
                    layer_idx=task.layer_idx,
                    ret1=ret1,
                    replica_uuid=replica_uuid,
                )
                self.wait_load(result.replica_uuid)
                cuda_hook_time_end("sllm_worker_task")
                self.output_queue.put(result)
            except Exception as e:
                # 将异常放入输出队列
                self.output_queue.put(e)

    def submit_load(self, layer_idx: int, tensor_index_names: List[str], device_index_int: int):
        """
        提交异步加载任务到队列
        
        Args:
            layer_idx: 层索引
            tensor_index_names: 需要加载的 tensor 名称列表
            device_index_int: 设备索引
            
        Returns:
            任务已提交（异步执行）
        """
        task = LoadTask(
            layer_idx=layer_idx,
            tensor_index_names=tensor_index_names,
            device_index_int=device_index_int
        )
        self.input_queue.put(task)
        
    def get_result_wait(self, timeout: Optional[float] = None) -> LoadResult:
        """
        获取加载结果并等待加载完成，然后将权重恢复到模型中
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
            
        Returns:
            LoadResult: 加载结果
            
        Raises:
            Exception: 如果加载过程中发生异常
        """
        result = self.output_queue.get(timeout=timeout)

        if isinstance(result, Exception):
            raise result

        # 等待加载完成
        # self.wait_load(result.replica_uuid)
        return result 
        
    def has_result(self):
        """检查是否有结果"""
        return not self.output_queue.empty()
    
    def wait_load(self, replica_uuid: str):
        """
        等待加载完成（同步调用）
        
        Args:
            replica_uuid: 从 LoadResult 中获取的 replica_uuid
        """
        self.cmv.wait_load_into_gpu(replica_uuid)

    def start(self):
        """启动所有 worker 线程"""
        if self.running:
            return
        self.running = True
        self.threads = []
        
        # 启动多个 worker 线程
        for worker_id in range(self.num_workers):
            thread = threading.Thread(
                target=self._worker, 
                args=(worker_id,),
                daemon=True,
                name=f"SLLMTM-Worker-{worker_id}"
            )
            thread.start()
            self.threads.append(thread)

    def stop(self):
        """停止所有 worker 线程"""
        if not self.running:
            return
        self.running = False
        
        # 向每个 worker 发送停止信号（发送 num_workers 个 None）
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        
        # 等待所有线程结束
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        self.threads = []
        
        # 清理队列中的剩余 None 信号（防止有残留）
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        # 清理 CUDA 缓存
        torch.cuda.empty_cache()

