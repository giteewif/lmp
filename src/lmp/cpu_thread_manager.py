import queue
import time
import threading
import torch
from typing import Optional, Tuple, Dict, List
from models.mlpmodule import MLPModuleWrapper, ExpertEinsumTask, ExpertEinsumResult

class CETM:
    def __init__(self, mlpm: MLPModuleWrapper, hmv, num_workers: int = 1):
        """
        CPU Expert Thread Manager - 支持多个worker并行处理请求
        
        Args:
            mlpm: MLPModuleWrapper 实例
            hmv: HostMemoryView 实例
            num_workers: worker线程数量，默认为1
        """
        self.mlpm = mlpm
        self.hmv = hmv
        self.num_workers = num_workers
        self.running = False
        self.threads = []  # 改为列表，支持多个线程
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
    def _worker(self, worker_id: int):
        """工作线程主循环"""
        while self.running:
            # 从队列获取任务
            task: Optional[ExpertEinsumTask] = self.input_queue.get()
            
            # 如果收到 None，表示停止信号
            if task is None:
                break
            
            try:
                if not task.if_decode:
                    # 执行 experts_func_einsum
                    final_hidden_states = self.mlpm.experts_func_einsum(
                        hmv=self.hmv,
                        layer_idx=task.layer_idx,
                        expert_idx_list=task.expert_idx_list,
                        expert_indices_map=task.expert_indices_map,
                        expert_token_indices_map=task.expert_token_indices_map,
                        flat_hidden_states=task.flat_hidden_states,
                        flat_experts_weight=task.flat_experts_weight,
                        idxs=task.idxs,
                        final_hidden_states=task.final_hidden_states,
                        output_queue=self.output_queue
                    )
                else:
                    final_hidden_states = self.mlpm.experts_func(
                        hmv=self.hmv.mlpm_hi, layer_idx=task.layer_idx,
                        expert_idx_list=list(task.expert_idx_list),
                        expert_indices_map={eid: task.expert_indices_map[eid] for eid in task.expert_idx_list},
                        expert_token_indices_map={eid: task.expert_token_indices_map[eid] for eid in task.expert_idx_list},
                        flat_hidden_states=task.flat_hidden_states,
                        flat_experts_weight=task.flat_experts_weight,
                        idxs=task.idxs,
                        final_hidden_states=task.final_hidden_states,
                        device="cpu"
                    )
                # 在 experts_func_einsum 中pool，避免释放tensor 阻塞
                # result = ExpertEinsumResult(final_hidden_states=final_hidden_states)
                # self.output_queue.put(result)
            except Exception as e:
                # 将异常放入输出队列
                self.output_queue.put(e)

    def submit(self, task: ExpertEinsumTask):
        """提交任务到队列"""
        self.input_queue.put(task)
        
    def get_result(self, timeout: Optional[float] = None) -> ExpertEinsumResult:
        """获取结果"""
        result = self.output_queue.get(timeout=timeout)
        if isinstance(result, Exception):
            raise result
        return result
        
    def has_result(self):
        """检查是否有结果"""
        return not self.output_queue.empty()

    def start(self):
        """启动所有worker线程"""
        if self.running:
            return
        self.running = True
        self.threads = []
        
        # 启动多个worker线程
        for worker_id in range(self.num_workers):
            thread = threading.Thread(
                target=self._worker, 
                args=(worker_id,),
                daemon=True,
                name=f"CETM-Worker-{worker_id}"
            )
            thread.start()
            self.threads.append(thread)

    def stop(self):
        """停止所有worker线程"""
        if not self.running:
            return
        self.running = False
        
        # 向每个worker发送停止信号（发送num_workers个None）
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        
        # 等待所有线程结束
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        self.threads = []
        
        # 清理队列中的剩余None信号（防止有残留）
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        # 清理CUDA缓存
        torch.cuda.empty_cache()
    