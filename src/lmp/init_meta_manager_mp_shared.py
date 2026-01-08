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

# 尝试导入 cloudpickle（torch 推荐用于序列化函数）
try:
    import cloudpickle
    HAS_CLOUDPICKLE = True
except ImportError:
    HAS_CLOUDPICKLE = False
    # 如果没有 cloudpickle，使用标准 pickle（可能无法序列化 __main__ 中的函数）
    import pickle


class LayerStatus(Enum):
    """Layer 初始化状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LayerShareInfo:
    """共享的 DecoderLayer 信息"""
    layer_idx: int
    layer: Any  # DecoderLayer 对象（torch.multiprocessing 支持直接传递）


@dataclass
class InitRequest:
    """初始化请求"""
    layer_idx: int
    init_func: Callable  # 初始化函数
    config: Any  # 模型配置
    batch_finish: bool = False


def _init_process_func(input_queue: Queue, output_queue: Queue, exit_event):
    """
    独立初始化进程主函数
    
    从队列获取初始化请求，初始化 DecoderLayer，然后通过队列发送给主进程
    进程会一直运行直到收到退出信号
    
    Args:
        input_queue: 输入队列（从主进程接收初始化请求，包含函数和参数）
        output_queue: 输出队列（向主进程发送初始化好的 layer）
        exit_event: 退出事件（主进程设置此事件来通知初始化进程退出）
    """
    # 导入必要的模块（在初始化进程中）
    import sys
    import os
    import time
    import queue as queue_module
    # project_root = "/mnt/zhengcf3/lmp"
    # src_dir = os.path.join(project_root, 'src')
    # sllm_store_dir = os.path.join(project_root, 'src', 'sllm_store')
    # sys.path.insert(0, sllm_store_dir)
    # sys.path.insert(0, src_dir)
    
    from utils.cuda_h import cuda_hook_time, cuda_hook_time_end
    
    # print(f"Init process {os.getpid()}: Started, waiting for initialization requests...")
    local_list = list()
    try:
        # 持续运行，从队列获取初始化请求
        while not exit_event.is_set():
            try:
                # 从输入队列获取初始化请求（带超时，以便定期检查退出信号）
                try:
                    request: Optional[InitRequest] = input_queue.get()
                except queue_module.Empty:
                    # 队列为空，继续检查退出信号
                    continue
                
                # 如果收到 None，表示没有更多请求
                if request is None:
                    continue
                
                # 处理初始化请求
                try:
                    layer_idx = request.layer_idx
                    init_func = request.init_func
                    config = request.config
                    
                    cuda_hook_time(f"init_layer_{layer_idx}")
                    
                    # 调用初始化函数
                    layer = init_func(layer_idx, config)
                    
                    layer_info = LayerShareInfo(layer_idx=layer_idx, layer=layer)
                    # layer_info = LayerShareInfo(layer_idx=layer_idx, layer=layer)
                    # layer_info = LayerShareInfo(layer_idx=layer_idx, layer=None)
                    # output_queue.put(layer_info)
                    cuda_hook_time_end(f"init_layer_{layer_idx}")
                    
                    # 通过队列发送给主进程
                    if_batch_finish = request.batch_finish
                    if not if_batch_finish:
                        local_list.append(layer_info)
                    else:
                        local_list.append(layer_info)
                        # batch_finish为True时，先发送当前layer，然后发送本地队列中的所有layer
                        cuda_hook_time("send_batch_finish")
                        # 然后发送本地队列中的所有layer
                        output_queue.put(local_list)
                        cuda_hook_time_end("send_batch_finish")

                    
                    # print(f"Init process {os.getpid()}: Initialized DecoderLayer {layer_idx}")
                    
                except Exception as e:
                    import traceback
                    print(f"Init process {os.getpid()}: Failed to initialize DecoderLayer {request.layer_idx}: {e}")
                    traceback.print_exc()
                    # 发送错误信息到输出队列
                    error_info = LayerShareInfo(layer_idx=request.layer_idx, layer=None)
                    output_queue.put(error_info)
                    
            except Exception as e:
                # 处理队列操作异常
                import traceback
                print(f"Init process {os.getpid()}: Error processing request: {e}")
                traceback.print_exc()
                # 继续运行，不退出
                time.sleep(0.1)
        
        print(f"Init process {os.getpid()}: Received exit signal, exiting...")
        
    except Exception as e:
        import traceback
        print(f"Init process {os.getpid()}: Fatal error: {e}")
        traceback.print_exc()


class InitMetaManagerMPShared:
    """
    多进程版本的 InitMetaManager - 独立初始化进程版本
    
    使用 torch.multiprocessing 实现：
    1. 启动多个独立进程初始化所有 DecoderLayer（支持配置进程数量）
    2. 主进程从这些进程获取初始化好的 DecoderLayer 对象
    3. torch.multiprocessing 支持 CUDA 张量共享，可以直接传递对象
    4. 多个进程共享同一个任务队列（工作窃取模式），自动负载均衡
    
    注意：
    1. 初始化进程需要独立的 CUDA 上下文
    2. 使用 torch.multiprocessing 而不是标准 multiprocessing
    3. DecoderLayer 对象可以直接传递，无需序列化
    """
    
    def __init__(self, num_processes: int = 1):
        """
        初始化管理器
        
        Args:
            num_processes: 初始化进程数量，默认为1（单进程模式）
        """
        # 设置启动方法（torch.multiprocessing 推荐使用 'spawn'）
        try:
            mp.set_start_method('spawn', force=False)
        except RuntimeError:
            pass  # 已经设置过了
        
        # 注意：使用 spawn 启动方法时，传递给子进程的函数必须可以被 pickle
        # 如果函数定义在 __main__ 中，需要使用 cloudpickle 或将函数移到模块级别
        # 这里我们尝试配置 cloudpickle（如果可用），但更推荐将函数定义在模块级别
        if HAS_CLOUDPICKLE:
            try:
                # 尝试配置 torch.multiprocessing 使用 cloudpickle
                # 注意：这可能需要根据具体的 torch 版本调整
                import multiprocessing
                # 某些情况下，可以通过设置环境变量或使用其他方法
                # 但最可靠的方法是将函数定义在模块级别
                pass
            except Exception:
                pass
        
        self.num_processes = num_processes
        self.running = False
        self.init_processes = []  # 初始化进程列表
        self.input_queues = []  # 输入队列列表（每个进程一个独立的输入队列）
        self.output_queue = None  # 输出队列（所有进程共享，主进程接收结果）
        
        # Manager 和共享状态延迟创建（避免在导入时创建）
        self.manager = None
        self.exit_event = None  # 退出事件（用于通知所有初始化进程退出）
        self._result_cache = {}  # 主进程中的结果缓存（从 Queue 接收）
        
        # 存储任务信息（主进程）
        self.layer_tasks: Dict[int, int] = {}  # layer_idx -> status
        
        # 初始化参数（保存用于启动进程）
        self.num_layers = 0
        self.init_func = None
        self.model = None
    
    def submit_all(self, init_func: Callable[[int, Any], Any], config):
        """
        提交所有 layer 的初始化任务
        
        注意：这个版本会启动一个独立进程来初始化所有 layer
        
        Args:
            num_layers: layer 总数
            init_func: 初始化函数（在初始化进程中调用）
                       init_func(layer_idx, model) -> layer 或 (layer, layer_copy)
            model: 模型对象（需要可序列化，传递给初始化进程）
            config: 模型配置对象（可选，未使用）
        """
        self.num_layers = config.num_hidden_layers
        self.init_func = init_func
        self.config = config
        
        # 初始化所有任务状态为 PENDING
        for layer_idx in range(self.num_layers):
            self.layer_tasks[layer_idx] = LayerStatus.PENDING.value
        
        # 先按layer数量分配好，再统一提交
        # 为每个进程分配layer索引列表
        process_layer_indices = [[] for _ in range(self.num_processes)]
        for layer_idx in range(self.num_layers):
            process_idx = layer_idx % self.num_processes
            process_layer_indices[process_idx].append(layer_idx)
        
        # 为每个进程创建请求并提交
        for process_idx in range(self.num_processes):
            layer_indices = process_layer_indices[process_idx]
            for i, layer_idx in enumerate(layer_indices):
                # 判断是否是该进程的最后一个layer
                is_last_for_process = (i == len(layer_indices) - 1)
                
                request = InitRequest(
                    layer_idx=layer_idx,
                    init_func=init_func,
                    config=config,
                    batch_finish=is_last_for_process
                )
                # 提交到对应进程的输入队列
                self.input_queues[process_idx].put(request)
    def submit_layer(self, layer_idx: int, init_func: Callable[[int, Any], Any], config: Any):
        """
        提交特定 layer 的初始化任务
        """
        request = InitRequest(
            layer_idx=layer_idx,
            init_func=init_func,
            config=config
        )
        self.layer_tasks[layer_idx] = LayerStatus.PENDING.value
        # 将请求分配到对应的进程队列（轮询分配）
        process_idx = layer_idx % self.num_processes
        self.input_queues[process_idx].put(request)

    def start(self):
        """启动初始化进程"""
        if self.running:
            return
        
        if self.num_processes < 1:
            raise ValueError(f"num_processes must be >= 1, got {self.num_processes}")
        
        self.running = True
        
        # 在 start 时初始化和创建 Manager 和 Queue
        if self.manager is None:
            # 使用标准 multiprocessing.Manager（因为 torch.multiprocessing 没有 Manager）
            self.manager = mp.Manager()
            self.exit_event = self.manager.Event()  # 退出事件
            # 使用 torch.multiprocessing.Queue（对 CUDA 张量更友好）
            # 为每个进程创建独立的输入队列
            self.input_queues = [Queue() for _ in range(self.num_processes)]
            self.output_queue = Queue()  # 输出队列（所有进程共享）
        
        # 启动多个初始化进程，每个进程使用自己的输入队列
        self.init_processes = []
        for i in range(self.num_processes):
            process = Process(
                target=_init_process_func,
                args=(self.input_queues[i], self.output_queue, self.exit_event),
                name=f"InitMetaManagerMPShared-InitProcess-{i}"
            )
            process.start()
            self.init_processes.append(process)
            print(f"Started initialization process {i} (PID: {process.pid})")
        
        print(f"Started {self.num_processes} initialization process(es)")
    
    def wait_layer(self, layer_idx: int, timeout: Optional[float] = None) -> Any:
        """
        等待特定 layer 初始化完成
        
        从共享输出队列接收 layer，可能收到任意 layer（因为多进程并行执行）
        如果收到的是其他 layer，缓存起来并继续等待目标 layer
        """
        if layer_idx not in self.layer_tasks:
            raise KeyError(f"Layer {layer_idx} not found in tasks")
        
        if self.output_queue is None:
            raise RuntimeError("Manager not initialized. Call start() first.")
        
        if not self.running:
            raise RuntimeError("Initialization process not started. Call start() first.")
        
        # 如果已经在缓存中，直接返回
        if layer_idx in self._result_cache:
            return self._result_cache[layer_idx]
        
        # 从队列接收结果，直到收到目标 layer
        # 注意：现在收到的是一个列表，包含多个 LayerShareInfo
        while True:
            received_data = self.output_queue.get()
            
            # 判断收到的是列表还是单个 LayerShareInfo（兼容性处理）
            if isinstance(received_data, list):
                # 收到的是列表，遍历处理每个 layer_info
                for layer_info in received_data:
                    result = layer_info.layer
                    received_layer_idx = layer_info.layer_idx
                    
                    # 缓存结果
                    self._result_cache[received_layer_idx] = result
                    
                    # 如果收到的是当前等待的 layer，直接返回
                    if received_layer_idx == layer_idx:
                        return result
            else:
                # 兼容旧格式：单个 LayerShareInfo
                layer_info = received_data
                result = layer_info.layer
                received_layer_idx = layer_info.layer_idx
                
                # 缓存结果
                self._result_cache[received_layer_idx] = result
                
                # 如果收到的是当前等待的 layer，直接返回
                if received_layer_idx == layer_idx:
                    return result
                    
    
    def wait_all(self, timeout: Optional[float] = None) -> Dict[int, Tuple[Any, Any]]:
        """
        等待所有 layer 初始化完成
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
            
        Returns:
            字典：layer_idx -> layer
        """
        if not self.running:
            raise RuntimeError("Initialization process not started. Call start() first.")
        
        # 从队列接收所有 layer，直到收到所有 num_layers 个结果
        # 注意：现在收到的是一个列表，包含多个 LayerShareInfo
        while len(self._result_cache) < self.num_layers:
            received_data = self.output_queue.get()
            
            # 判断收到的是列表还是单个 LayerShareInfo（兼容性处理）
            if isinstance(received_data, list):
                # 收到的是列表，遍历处理每个 layer_info
                for layer_info in received_data:
                    result = layer_info.layer
                    layer_idx = layer_info.layer_idx
                    
                    # 缓存结果（如果已存在则更新）
                    self._result_cache[layer_idx] = result
            else:
                # 兼容旧格式：单个 LayerShareInfo
                layer_info = received_data
                result = layer_info.layer
                layer_idx = layer_info.layer_idx
                
                # 缓存结果（如果已存在则更新）
                self._result_cache[layer_idx] = result

        return self._result_cache
    
    def _reset_cache(self):
        self._result_cache = {}

    def stop(self):
        """停止所有初始化进程"""
        if not self.running:
            return
        
        self.running = False
        
        # 发送退出信号给所有初始化进程
        if self.exit_event is not None:
            self.exit_event.set()
        
        # 等待所有初始化进程结束
        for i, process in enumerate(self.init_processes):
            if process is not None:
                process.join(timeout=10.0)
                if process.is_alive():
                    print(f"Terminating initialization process {i} (PID: {process.pid})...")
                    process.terminate()
                    process.join(timeout=5.0)
                    if process.is_alive():
                        process.kill()
                        process.join()
        
        self.init_processes = []
        
        # 清理队列
        if self.input_queues is not None:
            for input_queue in self.input_queues:
                if input_queue is not None:
                    while not input_queue.empty():
                        try:
                            input_queue.get_nowait()
                        except Exception:
                            break
        
        if self.output_queue is not None:
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except Exception:
                    break
        
        # 清理 CUDA 缓存
        torch.cuda.empty_cache()

