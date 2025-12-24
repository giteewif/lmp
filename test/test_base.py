import torch
import time
from transformers import MixtralForCausalLM, AutoModelForCausalLM


# 加载 Mixtral 模型（只加载一个 expert）
#Mixtral-8x22B-Instruct-v0.1
model_path = "/mnt/zhengcf3/models/Mixtral-8x7B"
print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # 加载到 CPU
)

model.eval()
times = 5
h1 = 14336
h2 = 4096
dtype=torch.bfloat16
device1 = "cuda:1"
device2 = "cuda:2"
device3 = "cuda:3"

layer1 = model.model.layers[1]
layer2 = model.model.layers[2]
layer3 = model.model.layers[3]
layer1 = layer1.to(device1)
layer2 = layer2.to(device2)

layer1.eval()
layer2.eval()
batch_size = 64
seq_len = 512

inputsb0 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)
inputsb1 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)
inputsb2 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)

inputs_cpu = torch.randn(batch_size, seq_len, h2, dtype=dtype, device="cpu", pin_memory=True)
inputs_list = [inputsb0, inputsb1, inputsb2]

w1 = torch.nn.Linear(h2, h1, bias=False, device=device1, dtype=dtype)
w2 = torch.nn.Linear(h1, h2, bias=False, device=device1, dtype=dtype)
w3 = torch.nn.Linear(h2, h1, bias=False, device=device1, dtype=dtype)
act_fn = torch.nn.SiLU()

def experts_cal(layer, inputs):
    expert = layer.block_sparse_moe.experts[0]
    # for expert in layer.block_sparse_moe.experts:
    out = expert(inputs)
    return out
def layer_cal(layer, inputs):
    bmoe = layer.block_sparse_moe
    out, _ = bmoe(inputs)
    return out

def move(tensor, device):
    return tensor.to(device)

times_list = []
time_start = time.time()
io_stream = torch.cuda.Stream(device=device1)
for i in range(times):
    # time_start_once = time.time()
    for input in inputs_list:
        time_start_once = time.time()
        print(f"提交任务到worker1")
        with torch.cuda.stream(io_stream):
            inputs_gpu = inputs_cpu.to(device1, non_blocking=True)
        out = layer_cal(layer1,inputs_gpu)
        # layer = move(layer3, device1)
        torch.cuda.synchronize(device=device1)
        
    
    # 关于 del out 的说明：
    # Python 会自动释放：当 out 被重新赋值时，旧的 out 会失去引用，垃圾回收器会自动释放
    # 但 PyTorch 使用 CUDA 内存池，del 后内存可能仍在池中，需要 empty_cache() 才能真正释放
    # 在这个循环中，理论上不需要显式 del（因为每次迭代都会重新赋值），
    # 但如果有内存压力，del + empty_cache() 可以更及时地释放内存
        # del out  # 可选：显式删除以立即释放引用
        time_end_once = time.time()
    # time_end_once = time.time()
    # torch.cuda.empty_cache()  # 释放 CUDA 内存池中的未使用内存
    times_list.append(round(time_end_once - time_start_once, 6))
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds"
    f"Every time {times_list}"
)
