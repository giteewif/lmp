import torch
import time
def cuda_hook(name):
    # torch.cuda.nvtx.range_push(name)
    pass
def cuda_hook_end(name):
    # torch.cuda.nvtx.range_pop()
    pass

timings = {}
def cuda_hook_time(name):
    logger.debug(f"start {name}")
    time_start = time.time()
    timings[name] = time_start
    cuda_hook(name)
    return time_start  
def cuda_hook_time_end(name):
    time_end = time.time()
    cuda_hook_end(name)
    time_cost = time_end - timings[name]
    logger.debug(f"end {name} cost {time_cost} seconds")
    return time_cost

from utils.logger import init_logger

logger = init_logger(__name__)

def log_cuda_memory_usage(device, step_name="", step_num=""):
    """Log CUDA memory usage for debugging using nvml"""
    # return 0, 0, 0
    try:
        import pynvml
        pynvml.nvmlInit()
        device_idx = (
            device.index if isinstance(device, torch.device) else
            int(device.split(':')[1]) if isinstance(device, str) and ":" in device else
            int(device)
        )
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem = memory_info.total / 1024**3  # 总显存(GB)
        used_mem = memory_info.used / 1024**3    # 已用显存(GB)
        free_mem = memory_info.free / 1024**3    # 空闲显存(GB)
        step_info = f" (step {step_name} {step_num})"
        logger.debug(
            f"NVML Memory {step_info}: Total={total_mem:.2f}GB, Used={used_mem:.2f}GB, Free={free_mem:.6f}GB"
        )
        return used_mem, free_mem, total_mem
    except Exception as e:
        logger.debug(f"NVML unavailable or failed: {e}")