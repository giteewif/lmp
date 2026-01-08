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

@dataclass
class DeviceOutput:
    output_tensor: torch.Tensor

@dataclass
class InitRequest:
    """初始化请求"""
    input_tensor: torch.Tensor  # 输入张量

def _init_process_func(input_queue: Queue, output_queue: Queue, exit_event):
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
    
    from utils.cuda_h import cuda_hook_time, cuda_hook_time_end
    
    local_list = list()
    # 持续运行，从队列获取初始化请求
    while not exit_event.is_set():
        try:
            request: Optional[InitRequest] = input_queue.get()
            
            # 处理初始化请求
            try:
                cuda_hook_time("write_result")
                input_tensor = request.input_tensor
                output_tensor = torch.randn(32, 64, 2048, dtype=torch.bfloat16, device="cuda:0")
                output_queue.put(DeviceOutput(output_tensor=output_tensor))
                cuda_hook_time_end("write_result")
            except Exception as e:
                import traceback
                print(f"Init process {os.getpid()}: Failed to write result: {e}")
                print(traceback.format_exc())
                # 发送错误信息到输出队列
                error_info = DeviceOutput(output_tensor=None)
                output_queue.put(error_info)
                
        except Exception as e:
            import traceback
            print(f"Init process {os.getpid()}: Error in main loop: {e}")
            print(traceback.format_exc())
    
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
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.exit_event = mp.Event()

    def submit(self, input_tensor: torch.Tensor):        
        """
        提交初始化请求
        """
        self.input_queue.put(InitRequest(input_tensor=input_tensor))

    def wait(self):
        """
        等待初始化完成
        """
        cuda_hook_time("wait")
        output =  self.output_queue.get()
        output = output.output_tensor
        cuda_hook_time_end("wait")
        return output

    def start(self):
        """
        启动多进程设备管理器
        """
        self.processes = []
        for i in range(self.num_processes):
            process = Process(target=_init_process_func, args=(self.input_queue, self.output_queue, self.exit_event))
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