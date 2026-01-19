import time
import torch
import os
import gc

# 设置只使用 cuda:1,2,3，不使用 cuda:0
# 通过设置 CUDA_VISIBLE_DEVICES 来重新映射设备
# 这样 cuda:1,2,3 会被映射为 cuda:0,1,2
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

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
# Qwen1.5-MoE-A2.7B or deepseek-moe-16b-base-bfloat16 or Mixtral-8x7B
# DeepSeek-V2-Lite
gmodel_path = "deepseek-moe-16b-base-bfloat16"
batch_size = 32
seq_len = 64
device1 = "cuda:0"  # 由于 CUDA_VISIBLE_DEVICES，cuda:0 实际对应物理的 cuda:1
token_path = f"/mnt/zhengcf3/models/sllm_models/{gmodel_path}"
tokenizer = AutoTokenizer.from_pretrained(token_path)

inputs = generate_input_ids_pad_new(tokenizer, batch_size, seq_len, device1)

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

def test_load_and_generate_model(fully_parallel=True):
    # Qwen1.5-MoE-A2.7B or deepseek-moe-16b-base-bfloat16 or Mixtral-8x7B
    model_path = gmodel_path
    storage_path = "/mnt/zhengcf3/models/sllm_models"
    start = time.time()
    # 使用 cuda:1,2,3（通过 CUDA_VISIBLE_DEVICES 映射为 cuda:0,1,2）
    model = load_model(model_path, device_map="auto", torch_dtype=torch.bfloat16, storage_path=storage_path, fully_parallel=fully_parallel)
    # Please note the loading time depends on the model size and the hardware bandwidth.
    end = time.time()
    print(f"Model loading time: {time.time() - start:.2f}s")

    # inputs = tokenizer('Hello, my dog is cute', return_tensors='pt').to("cuda")

    # 预热：第一次调用时会有 CUDA kernel 编译等开销
    print("=" * 60)
    print("First generate (with warmup overhead):")
    print("=" * 60)
    torch.cuda.synchronize()
    generage_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=True,  # 使用贪心解码
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    torch.cuda.synchronize()
    generage_end = time.time()
    first_time = generage_end - generage_start
    print(f"First generate time: {first_time:.2f}s")

    print("=" * 60)
    print("Prefill generate:")
    print("=" * 60)
    torch.cuda.synchronize()
    generage_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=True,  # 使用贪心解码
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    torch.cuda.synchronize()
    generage_end = time.time()
    second_time = generage_end - generage_start
    print(f"Prefill generate:: {second_time:.2f}s")

    print("=" * 60)
    print("32 output generate (should be faster):")
    print("=" * 60)
    torch.cuda.synchronize()
    generage_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,  # 使用贪心解码
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    torch.cuda.synchronize()
    generage_end = time.time()
    second32_time = generage_end - generage_start

    decode_single_time = (second32_time - second_time) / 31
    print(f"32 output generate time: {second32_time:.2f}s")
    print(f"decode single time: {decode_single_time:.2f}s")

    # 计算加速比
    if second_time > 0:
        speedup = first_time / second_time
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"\n原因分析:")
        print(f"  1. 第一次调用包含 CUDA kernel JIT 编译开销 (~{first_time - second_time:.2f}s)")
        print(f"  2. 第一次调用需要初始化 KV cache (past_key_values)")
        print(f"  3. 第一次调用 cuDNN 需要选择最优算法 (benchmark)")
        print(f"  4. 第一次调用可能需要加载某些权重到 GPU")
        print(f"  5. PyTorch 的 autograd 图构建和优化")
    print(f"Model loading time: {end - start:.2f}s")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    # 释放模型资源
    print("\nReleasing model resources...")
    release_model_resources(model)
    print("Model resources released")

if __name__ == "__main__":
    for i in range(3):
        warm_up()
    fully_parallel = True
    # 第一次运行
    test_load_and_generate_model(fully_parallel=fully_parallel)
    
    # 等待资源完全释放
    print("\nWaiting for resources to be fully released...")
    time.sleep(2)
    
    # 第二次运行（测试重新加载）
    print(f"\n{'=' * 60}")
    print("Second run (reload test):")
    print(f"{'=' * 60}")
    test_load_and_generate_model(fully_parallel=fully_parallel)