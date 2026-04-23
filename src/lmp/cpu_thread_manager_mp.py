import os
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


def _resolve_mp_worker_intraop_threads() -> int:
    """
    CPU experts 子进程内 ``torch.set_num_threads`` 取值（避免硬编码 64 导致 BF16 BMM 过度抢占）。

    优先级（取第一个合法正整数）：
    1. ``LMP_MP_CPU_INTRAOP_THREADS``
    2. ``OMP_NUM_THREADS``
    3. 默认 ``min(32, max(1, os.cpu_count() or 8))``
    """
    for key in ("LMP_MP_CPU_INTRAOP_THREADS", "OMP_NUM_THREADS"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        try:
            n = int(raw, 10)
            return max(1, min(n, 512))
        except ValueError:
            continue
    try:
        c = int(os.cpu_count() or 8)
    except (TypeError, ValueError):
        c = 8
    return max(1, min(c, 32))


def _mp_worker_diag_line(msg: str) -> None:
    """子进程诊断行：``print`` + flush；若设 ``LMP_MP_SELFTEST_DIAG=/path`` 则同时追加到文件（不依赖 shell 重定向）。"""
    print(msg, flush=True)
    path = (os.environ.get("LMP_MP_SELFTEST_DIAG") or "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


@dataclass
class CPUExpertsInput:
    layer_idx: int
    expert_idx_list: list[int]
    expert_indices_map: dict[int, tuple[int, int]]
    flat_hidden_states: torch.Tensor
    idxs: torch.Tensor
    if_decode: bool = False
    fused: bool = False
    # ``True``：``einsum_with_group_tensors_mp(..., fused=True)`` 内用 ``group_fused_experts_tensor`` 取权再 einsum；
    # ``False``：``group_experts_tensor`` + 同一套 einsum。
    use_bmm: bool = False
    # ``True``：走 ``bmm_with_group_tensors_mp_1``（融合专家 + CPU BMM）；与 ``fused`` 独立，BMM 路径固定用 ``group_fused_experts_tensor``。

def _cpu_experts_worker(
    input_queue: Queue,
    output_queue: Queue,
    model_path: str,
    model_name_type: str,
    exit_event,
    bootstrap_ready_queue: Queue,
):
    """
    CPU experts worker 进程主函数

    - ``use_bmm=True``：``bmm_with_group_tensors_mp_1``（融合权重 BMM）。
    - ``use_bmm=False``：``experts_func_einsum_mp`` → ``einsum_with_group_tensors_mp``，其中 ``fused`` 控制
      ``group_fused_experts_tensor`` 与 ``group_experts_tensor`` 取权。结果经 ``output_queue`` 回传 ``ExpertEinsumResult``。

    Args:
        input_queue: 输入队列（从主进程接收计算请求）
        output_queue: 输出队列（向主进程传递计算结果）
        model_path: 模型路径
        model_name_type: 模型名称类型
        exit_event: 退出事件（主进程设置此事件来通知工作进程退出）
        bootstrap_ready_queue: 各 worker 完成自检后 ``put(True)``，供主进程 ``wait_worker_bootstrap_ready`` 同步
    """
    import sys
    import time
    import queue as queue_module
    import torch
    from utils.logger import init_logger
    from utils.cuda_h import cuda_hook_time, cuda_hook_time_end
    
    from sllm_store._C import (
        allocate_cuda_memory,
    )
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    logger = init_logger(__name__)
    _mp_worker_diag_line(
        f"[cpu_experts_worker pid={os.getpid()}] entered (model_path={model_path!r} type={model_name_type!r})"
    )

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
    _intra = _resolve_mp_worker_intraop_threads()
    torch.set_num_threads(_intra)
    logger.debug(
        "CPU experts MP worker: torch.set_num_threads(%d) "
        "(LMP_MP_CPU_INTRAOP_THREADS / OMP_NUM_THREADS / default cap 32)",
        _intra,
    )
    _mp_worker_diag_line(
        f"[cpu_experts_worker pid={os.getpid()}] torch.set_num_threads({_intra}) done"
    )

    _mp_worker_diag_line(
        f"[cpu_experts_worker pid={os.getpid()}] importing MLPModuleWrapper / HostMemoryView ..."
    )
    from models.mlpmodule import MLPModuleWrapper
    from lmp.cuda_memory_view import HostMemoryView
    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] imports done")

    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] MLPModuleWrapper(...) ...")
    mlpm = MLPModuleWrapper(model_name_type=model_name_type, model_path=model_path)
    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] init_chmv_meta_model(...) ...")
    cm = mlpm.init_chmv_meta_model(device="cpu")
    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] HostMemoryView(...) ...")
    hmv = HostMemoryView(mlpm=mlpm, empty_model=cm)
    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] HostMemoryView ready")

    # 子进程启动时做一次 CPU 半专家 fused BMM 自检；细粒度进度见 cuda_memory_view._print_group_bmm_self_test
    _mp_worker_diag_line(
        f"[cpu_experts_worker pid={os.getpid()}] begin _test_group_bmm_fused_experts()"
    )
    try:
        hmv._test_group_bmm_fused_experts()
    except Exception:
        import traceback

        _mp_worker_diag_line(traceback.format_exc())
        raise
    _mp_worker_diag_line(
        f"[cpu_experts_worker pid={os.getpid()}] end _test_group_bmm_fused_experts()"
    )
    logger.info("CPU experts worker: _test_group_bmm_fused_experts() finished.")

    try:
        bootstrap_ready_queue.put(True)
    except Exception:
        pass

    group_list_list = []
    while not exit_event.is_set():
      
        input_data: CPUExpertsInput = input_queue.get()

        # 清空group_list_list, 释放unmap
        if input_data.layer_idx == -1:
            group_list_list.clear()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result = ExpertEinsumResult(
                final_hidden_states=None,
                time_einsum_end=0.0,
            )
            output_queue.put(result)
            continue

        cuda_hook_time("experts_func_gpu_einsum_mp")

        expert_idx_list = input_data.expert_idx_list
        expert_indices_map = input_data.expert_indices_map
        idxs = input_data.idxs
        flat_hidden_states = input_data.flat_hidden_states

        if input_data.use_bmm:
            _, group_list = mlpm.bmm_with_group_tensors_mp(
                hmv=hmv,
                layer_idx=input_data.layer_idx,
                expert_idx_list=expert_idx_list,
                expert_indices_map=expert_indices_map,
                flat_hidden_states=flat_hidden_states,
                idxs=idxs,
                output_queue=output_queue,
            )
        else:
            _, group_list = mlpm.experts_func_einsum_mp(
                hmv=hmv,
                layer_idx=input_data.layer_idx,
                expert_idx_list=expert_idx_list,
                expert_indices_map=expert_indices_map,
                flat_hidden_states=flat_hidden_states,
                idxs=idxs,
                output_queue=output_queue,
                fused=input_data.fused,
            )
        group_list_list.append(group_list)
        # 两条路径均会向 ``output_queue`` 放入 ``ExpertEinsumResult``；``group_list`` 供延迟 unmap。

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
        self._bootstrap_ready_queue: Queue = Queue()
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
        idxs: torch.Tensor,
    ):
        """
        提交任务到第一个 worker 的队列
        """
        input_data = CPUExpertsInput(
            layer_idx=layer_idx,
            expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            idxs=idxs,
        )
        self.input_queues[0].put(input_data)
    
    def submit_worker(
        self,
        worker_idx: int,
        layer_idx: int,
        expert_idx_list: list[int],
        expert_indices_map: dict[int, tuple[int, int]],
        flat_hidden_states: torch.Tensor,
        idxs: torch.Tensor,
        fused: bool = False,
        use_bmm: bool = False,
    ):
        """
        提交任务到指定 worker 的队列
        
        Args:
            worker_idx: worker 索引
            fused: 仅 ``use_bmm=False`` 时生效，传给 ``einsum_with_group_tensors_mp``。
            use_bmm: ``True`` 时走 ``bmm_with_group_tensors_mp_1``，否则走 ``experts_func_einsum_mp``。
        """
        input_data = CPUExpertsInput(
            layer_idx=layer_idx,
            expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            idxs=idxs,
            fused=fused,
            use_bmm=use_bmm,
        )
        
        if worker_idx >= len(self.input_queues):
            raise ValueError(f"Worker index {worker_idx} out of range [0, {len(self.input_queues)})")
        self.input_queues[worker_idx].put(input_data)
        return input_data

    def wait_worker_bootstrap_ready(self, timeout: float | None = None) -> None:
        """
        阻塞直到每个 worker 完成 ``HostMemoryView`` 构建与 ``_test_group_bmm_fused_experts``。

        避免主进程在 ``start()`` 后立即 ``mp_stop()`` 导致子进程尚未自检就被杀掉、重定向日志截断。

        超时秒数：参数 ``timeout``，或环境变量 ``LMP_MP_BOOTSTRAP_TIMEOUT_SEC``（默认 600），按 **每个** worker 一次 ``get`` 使用。
        """
        import queue as queue_module

        if timeout is None:
            raw = os.environ.get("LMP_MP_BOOTSTRAP_TIMEOUT_SEC", "600").strip()
            try:
                timeout = float(raw)
            except ValueError:
                timeout = 600.0
        for _ in range(self.num_workers):
            try:
                self._bootstrap_ready_queue.get(timeout=timeout)
            except queue_module.Empty as e:
                raise RuntimeError(
                    "CPU experts MP worker bootstrap timed out "
                    f"(>{timeout}s per worker). Check model/shm or set LMP_MP_BOOTSTRAP_TIMEOUT_SEC."
                ) from e

    def wait(self):
        """
        阻塞直到 worker 在 ``output_queue`` 上放入 ``ExpertEinsumResult``（由 ``bmm_with_group_tensors_mp`` 写入）。

        Returns:
            ``ExpertEinsumResult.final_hidden_states``（通常为 GPU 张量）；清空任务时为 ``None``。
        """
        result: ExpertEinsumResult = self.output_queue.get()
        return result.final_hidden_states
    
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
                args=(
                    self.input_queues[i],
                    self.output_queue,
                    self.model_path,
                    self.model_name_type,
                    self.exit_event,
                    self._bootstrap_ready_queue,
                ),
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