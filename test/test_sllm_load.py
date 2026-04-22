import time
import torch
import os
import gc

# 设置只使用 cuda:1,2,3，不使用 cuda:0
# 通过设置 CUDA_VISIBLE_DEVICES 来重新映射设备
# 这样 cuda:1,2,3 会被映射为 cuda:0,1,2
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch

import pynvml

    
import sys
# 获取项目根目录和必要的路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
sllm_store_dir = os.path.join(project_root, 'src', 'sllm_store')

# 添加必要的目录到 Python 路径
# 1. 添加 sllm_store 目录（必须在 src 之前，这样 sllm_store 可以被找到）
sys.path.insert(0, sllm_store_dir)
# 2. 添加 src 目录（用于导入 lmp, utils 等）
sys.path.insert(0, src_dir)

from sllm_store.transformers import load_model
from lmp.lmp import MLPLLM
from utils.helper import * 
# warm up the GPU
# 由于设置了 CUDA_VISIBLE_DEVICES='1,2,3'，现在 cuda:0,1,2 对应物理的 cuda:1,2,3
def warm_up():
    num_gpus = torch.cuda.device_count()
    print(f"Warming up {num_gpus} GPU(s)...")
    for i in range(num_gpus):
        device = f"cuda:{i}"
        # 预热基本操作
        for _ in range(10):
            # 创建和移动 tensor
            x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
            y = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
            # 矩阵乘法
            z = torch.matmul(x, y)
            # Batch matrix multiplication (bmm)
            x_batch = torch.randn(8, 1024, 512, dtype=torch.bfloat16, device=device)
            y_batch = torch.randn(8, 512, 1024, dtype=torch.bfloat16, device=device)
            z_batch = torch.bmm(x_batch, y_batch)
            # Einsum (常用于 MoE 计算)
            a = torch.randn(8, 1024, 512, dtype=torch.bfloat16, device=device)
            b = torch.randn(8, 512, 1024, dtype=torch.bfloat16, device=device)
            c = torch.einsum('bij,bjk->bik', a, b)
            # 激活函数 (SiLU)
            x_act = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
            x_act = torch.nn.functional.silu(x_act)
            # 转置操作
            x_t = x.transpose(0, 1)
        torch.cuda.synchronize(device)
        print(f"GPU {i} warmed up")
    print("GPU warmup completed")

from transformers import AutoTokenizer


def load_local_tokenizer(path: str):
    """
    Load tokenizer from a local folder. Some checkpoints need sentencepiece/tiktoken
    for fast-tokenizer fallback; if that still fails, use the slow tokenizer.
    Install: pip install sentencepiece tiktoken
    """
    kwargs = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(path, **kwargs)
    except ValueError as e:
        err = str(e).lower()
        if "sentencepiece" in err or "tiktoken" in err or "backend tokenizer" in err:
            return AutoTokenizer.from_pretrained(path, use_fast=False, **kwargs)
        raise


#"deepseek-moe-16b-base" "Mixtral-8x22B" "Mixtral-8x7B" "Mixtral-8x7B-safetensors-hqq_hf"
# Qwen2-57B-A14B Qwen3-Coder-Next
# gemma4-26B-A4B ERNIE-4.5-VL-28B-A3B-Thinking ERNIE-4.5-21B-A3B-Thinking Qwen3.5-35B gpt-oss-20b
# Qwen1.5-MoE-A2.7B Deepseek-V2-Lite
gmodel_path = "gemma4-26B-A4B"
# batch 32 64
# seq_len 64 128
batch_size = 128
seq_len = 64
device1 = "cuda:0"  # 由于 CUDA_VISIBLE_DEVICES，cuda:0 实际对应物理的 cuda:1
token_path = f"/mnt/zhengcf3/models/sllm_models/{gmodel_path}"
# tokenizer = load_local_tokenizer(token_path)
# inputs = generate_input_ids_pad_new(tokenizer, batch_size, seq_len, device1)

def release_model_resources(model):
    """
    完全释放模型占用的资源
    
    Args:
        model: 要释放的模型对象（transformers 模型或 MLPLLM 对象）
    """
    try:
        
        # 2. 删除模型内部的子模块（如果有）
        if hasattr(model, 'model'):
            del model.model
        
        # 3. 删除其他可能的子模块
        if hasattr(model, 'lm_head'):
            del model.lm_head
        
        # 4. 删除模型本身
        del model
        
    except Exception as e:
        print(f"Warning: Error during model cleanup: {e}")
    
    # 5. 强制垃圾回收
    gc.collect()
    
    # 6. 清空所有 GPU 的缓存
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=f"cuda:{i}")
    
    # 7. 再次垃圾回收（确保所有引用都被清理）
    gc.collect()
    
    # 8. 最后再次清空缓存
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()

def get_gpu_memory():
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    total_gpu_mem = 0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i} 当前已分配显存: {mem_info.used / 1024 / 1024:.2f} MB")
        total_gpu_mem += mem_info.used
    print(f"所有GPU当前总显存占用: {total_gpu_mem / 1024 / 1024:.2f} MB")
    pynvml.nvmlShutdown()


def test_load_model(fully_parallel=True):
    model_path = gmodel_path
    storage_path = "/mnt/zhengcf3/models/sllm_models"
    start = time.time()
    model = load_model(model_path, device_map="auto", torch_dtype=torch.bfloat16, storage_path=storage_path, fully_parallel=fully_parallel)
    end = time.time()
    get_gpu_memory()
    print(f"Model load time: {time.time() - start:.2f}s")
    return model



if __name__ == "__main__":
    for i in range(3):
        warm_up()
    fully_parallel = True

    get_gpu_memory()
    # 第一次运行
    model = test_load_model(fully_parallel=fully_parallel)
    # INSERT_YOUR_CODE

    time.sleep(1)

    # INSERT_YOUR_CODE
    # 检查model下各参数所在的设备
    # for name, param in model.named_parameters():
    #     print(f"Param: {name}, Device: {param.device}")
    # for name, buf in model.named_buffers():
    #     print(f"Buffer: {name}, Device: {buf.device}")

    time.sleep(3)
    get_gpu_memory()