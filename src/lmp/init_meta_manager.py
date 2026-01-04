import queue
import time
import threading
import torch
from typing import Optional, Dict, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from utils.cuda_h import *

class LayerStatus(Enum):
    """Layer 初始化状态"""
    PENDING = "pending"      # 等待初始化
    IN_PROGRESS = "in_progress"  # 正在初始化
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 初始化失败


@dataclass
class InitLayerTask:
    """Layer 初始化任务"""
    layer_idx: int
    init_func: Callable[[int, Any], Any]  # 初始化函数，接受 layer_idx 和 config 作为参数，返回 layer
    config: Any  # 模型配置


@dataclass
class LayerShareInfo:
    """共享的 Layer 信息"""
    layer_idx: int
    layer: Any  # Layer 对象


class InitMetaManager:
    """
    异步初始化模型结构的管理器（单线程版本）
    
    功能：
    1. 一次性提交所有 layer 的初始化任务
    2. 每个 layer 完成后标志该 layer 完成
    3. 可以等待特定 layer 或确认 layer 已初始化
    4. 每次初始化两个 layer（原始 layer 和深拷贝）
    """
    
    def __init__(self):
        """
        单线程版本，不需要 num_workers 参数
        """
        self.running = False
        self.thread = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()  # 输出队列，用于接收初始化结果
        self.init_stream = torch.cuda.Stream()
        # 使用字典存储每个 layer 的任务（单线程，不需要锁）
        self.layer_tasks: Dict[int, InitLayerTask] = {}
        # 结果缓存（从输出队列接收）
        self.result_cache: Dict[int, Any] = {}

        self.num_layers = 0
        
    def _worker(self):
        """工作线程主循环（单线程）"""
        with torch.cuda.stream(self.init_stream):
            while self.running:
                # 从队列获取任务
                task: Optional[InitLayerTask] = self.input_queue.get()
                
                # 如果收到 None，表示停止信号
                if task is None:
                    break
                
                try:
                    cuda_hook_time(f"init_meta_l{task.layer_idx}")
                    
                    # 执行初始化函数（返回 layer 或 (layer, layer_copy)）
                    result = task.init_func(task.layer_idx, task.config)
                    
                    # 通过输出队列发送结果
                    layer_info = LayerShareInfo(layer_idx=task.layer_idx, layer=result)
                    self.output_queue.put(layer_info)
                    
                    cuda_hook_time_end(f"init_meta_l{task.layer_idx}")
                except Exception as e:
                    import traceback
                    print(f"Layer {task.layer_idx} initialization failed: {e}")
                    traceback.print_exc()
                    # 发送错误信息到输出队列（layer 为 None 表示失败）
                    raise ValueError(f"Layer {task.layer_idx} initialization failed: {e}")
    
    def submit_all(
        self, init_func: Callable[[int, Any], Any], 
        config: Any):
        """
        一次性提交所有 layer 的初始化任务
        
        Args:
            num_layers: layer 的总数
            init_func: 初始化函数，接受 layer_idx 作为参数，返回 layer
            config: 模型配置
        """
        self.num_layers = config.num_hidden_layers
        # 创建所有任务
        for layer_idx in range(self.num_layers):
            task = InitLayerTask(
                layer_idx=layer_idx,
                init_func=init_func,
                config=config
            )
            self.layer_tasks[layer_idx] = task
            # 提交到队列
            self.input_queue.put(task)
    def wait_layer(self, layer_idx: int, timeout: Optional[float] = None) -> Any:
        """
        等待特定 layer 初始化完成
        
        只从队列接收 layer，不使用状态信号
        """
        if layer_idx not in self.layer_tasks:
            raise KeyError(f"Layer {layer_idx} not found in tasks")
        
        if self.output_queue is None:
            raise RuntimeError("Manager not initialized. Call start() first.")
        
        if not self.running:
            raise RuntimeError("Initialization thread not started. Call start() first.")
        
        layer_info: Optional[LayerShareInfo] = self.output_queue.get()
        
        result = layer_info.layer
        self.result_cache[layer_info.layer_idx] = result
        
        # 如果收到的是当前等待的 layer，直接返回
        if layer_info.layer_idx == layer_idx:
            return result
    def submit_layer(self, layer_idx: int, init_func: Callable[[int, Any], Any], config: Any):
        """
        提交特定 layer 的初始化任务
        """
        task = InitLayerTask(
            layer_idx=layer_idx,
            init_func=init_func,
            config=config
        )
        self.layer_tasks[layer_idx] = task
        self.input_queue.put(task)
    def wait_layer_spin(self, layer_idx: int, timeout: Optional[float] = None) -> Any:
        """
        等待特定 layer 初始化完成（使用自旋锁轮询等待）
        
        自旋锁版本会持续轮询检查状态，不休眠，完全占用 CPU。
        适用于等待时间很短的情况，响应速度最快。
        
        Args:
            layer_idx: layer 索引
            timeout: 超时时间（秒），None 表示无限等待
            
        Returns:
            layer
            
        Raises:
            KeyError: 如果 layer_idx 不存在
            TimeoutError: 如果超时
            Exception: 如果初始化失败
        """
        if layer_idx not in self.layer_tasks:
            raise KeyError(f"Layer {layer_idx} not found in tasks")
        
        # 如果已经在缓存中，直接返回
        if layer_idx in self.result_cache:
            result = self.result_cache[layer_idx]
            if result is None:
                raise Exception(f"Layer {layer_idx} initialization failed")
            return result
        
        # 使用自旋锁轮询等待（不休眠）
        start_time = time.time()
        
        while True:
            # 检查超时
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Layer {layer_idx} initialization timeout after {timeout}s")
            
            # 尝试从队列获取结果（非阻塞）
            try:
                layer_info: LayerShareInfo = self.output_queue.get_nowait()
                # 将结果存入缓存
                self.result_cache[layer_info.layer_idx] = layer_info.layer
                
                # 如果收到的是当前等待的 layer，检查结果并返回
                if layer_info.layer_idx == layer_idx:
                    if layer_info.layer is None:
                        raise Exception(f"Layer {layer_idx} initialization failed")
                    return layer_info.layer
            except queue.Empty:
                # 队列为空，继续自旋等待
                pass
    
    def is_layer_ready(self, layer_idx: int) -> bool:
        """
        检查 layer 是否已初始化完成
        
        Args:
            layer_idx: layer 索引
            
        Returns:
            True 如果已完成，False 否则
        """
        if layer_idx not in self.layer_tasks:
            return False
        
        return layer_idx in self.result_cache
    
    def get_layer(self, layer_idx: int) -> Optional[Any]:
        """
        获取已初始化的 layer，如果未完成则返回 None
        
        Args:
            layer_idx: layer 索引
            
        Returns:
            layer，如果未完成则返回 None
        """
        if layer_idx not in self.layer_tasks:
            return None
        
        return self.result_cache.get(layer_idx)
    
    def get_layer_status(self, layer_idx: int) -> Optional[LayerStatus]:
        """
        获取 layer 的状态
        
        Args:
            layer_idx: layer 索引
            
        Returns:
            LayerStatus 或 None（如果 layer_idx 不存在）
        """
        if layer_idx not in self.layer_tasks:
            return None
        
        if layer_idx in self.result_cache:
            result = self.result_cache[layer_idx]
            if result is None:
                return LayerStatus.FAILED
            return LayerStatus.COMPLETED
        return LayerStatus.PENDING
    
    def wait_all(self, timeout: Optional[float] = None) -> Dict[int, Any]:
        """
        等待所有 layer 初始化完成（从队列获取结果）
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
            
        Returns:
            字典 {layer_idx: layer}
            
        """
        if not self.running:
            raise RuntimeError("Initialization process not started. Call start() first.")
        
        for i in range(self.num_layers):
            self.wait_layer(i)
        
        return self.result_cache
    
    def start(self):
        """启动工作线程（单线程）"""
        if self.running:
            return
        self.running = True
        
        # 启动单个工作线程
        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="InitMetaManager-Worker"
        )
        self.thread.start()
    
    def stop(self):
        """停止工作线程"""
        if not self.running:
            return
        self.running = False
        
        # 发送停止信号
        self.input_queue.put(None)
        
        # 等待线程结束
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        
        self.thread = None
        
        # 清理队列中的剩余None信号（防止有残留）
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        # 清理CUDA缓存
        torch.cuda.empty_cache()

