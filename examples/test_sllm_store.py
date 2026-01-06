import time
import torch
import os

# 设置只使用 cuda:1,2,3，不使用 cuda:0
# 通过设置 CUDA_VISIBLE_DEVICES 来重新映射设备
# 这样 cuda:1,2,3 会被映射为 cuda:0,1,2
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

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
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    torch.ones(1).to(f"cuda:{i}")
    torch.cuda.synchronize()

model_path = "deepseek-moe-16b-base-bfloat16"
storage_path = "/mnt/zhengcf3/models/sllm_models"
start = time.time()
# 使用 cuda:1,2,3（通过 CUDA_VISIBLE_DEVICES 映射为 cuda:0,1,2）
model = load_model(model_path, device_map="auto", torch_dtype=torch.bfloat16, storage_path=storage_path, fully_parallel=True)
# Please note the loading time depends on the model size and the hardware bandwidth.
end = time.time()
print(f"Model loading time: {time.time() - start:.2f}s")

from transformers import AutoTokenizer


batch_size = 32
seq_len = 64
device1 = "cuda:0"  # 由于 CUDA_VISIBLE_DEVICES，cuda:0 实际对应物理的 cuda:1
token_path = "/mnt/zhengcf3/models/sllm_models/deepseek-moe-16b-base-bfloat16"
tokenizer = AutoTokenizer.from_pretrained(token_path)

inputs = generate_input_ids_pad_new(tokenizer, batch_size, seq_len, device1)
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
    do_sample=False,  # 使用贪心解码
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
)
torch.cuda.synchronize()
generage_end = time.time()
first_time = generage_end - generage_start
print(f"First generate time: {first_time:.2f}s")

print("=" * 60)
print("Second generate (should be faster):")
print("=" * 60)
torch.cuda.synchronize()
generage_start = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=1,
    do_sample=False,  # 使用贪心解码
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
)
torch.cuda.synchronize()
generage_end = time.time()
second_time = generage_end - generage_start
print(f"Second generate time: {second_time:.2f}s")

print("=" * 60)
print("Second generate (should be faster):")
print("=" * 60)
torch.cuda.synchronize()
generage_start = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=2,
    do_sample=False,  # 使用贪心解码
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
)
torch.cuda.synchronize()
generage_end = time.time()
second_time = generage_end - generage_start
print(f"Second generate time: {second_time:.2f}s")

print("=" * 60)
print("Second generate (should be faster):")
print("=" * 60)
torch.cuda.synchronize()
generage_start = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=3,
    do_sample=False,  # 使用贪心解码
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
)
torch.cuda.synchronize()
generage_end = time.time()
second_time = generage_end - generage_start
print(f"Second generate time: {second_time:.2f}s")

print("=" * 60)
print("Second generate (should be faster):")
print("=" * 60)
torch.cuda.synchronize()
generage_start = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=13,
    do_sample=False,  # 使用贪心解码
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
)
torch.cuda.synchronize()
generage_end = time.time()
second_time = generage_end - generage_start
print(f"Second generate time: {second_time:.2f}s")

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