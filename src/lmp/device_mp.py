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


def _init_process_func(input_queue: Queue, output_queue: Queue, exit_event):
    """
    独立初始化进程主函数
    
    从队列获取初始化请求，初始化 DecoderLayer，然后通过队列发送给主进程
    进程会一直运行直到收到退出信号
    """
    pass

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
        for process in self.processes:
            process.terminate()
            process.join()
        self.processes = []