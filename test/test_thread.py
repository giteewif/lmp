import torch
import time
from transformers import MixtralForCausalLM, AutoModelForCausalLM
import threading
import queue
import time
from typing import Optional, Tuple

def cuda_hook(name):
    torch.cuda.nvtx.range_push(name)
def cuda_hook_end(name):
    torch.cuda.nvtx.range_pop()


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
class LayerComputeThread:
    """独立的layer计算线程，使用独立的CUDA流"""
    
    def __init__(self, layer, device: str, thread_id: int):
        self.layer = layer
        self.device = device
        self.thread_id = thread_id
        self.running = False
        self.thread = None
        
        # 创建独立的CUDA流
        self.stream = torch.cuda.Stream(device=device)
        self.io_stream = torch.cuda.Stream(device=device)
        # 输入队列：接收待处理的任务 (task_id, inputs)
        self.input_queue = queue.Queue()
        
        # 输出队列：返回处理结果 (task_id, outputs)
        self.output_queue = queue.Queue()
        
    def _worker(self):
        """工作线程主循环"""
        print(f"线程 {self.thread_id} (设备 {self.device}) 启动，CUDA流已创建")
        
        while self.running:
            task = None
            task_id = None
            # 从队列获取任务，超时1秒
            task = self.input_queue.get()
            
            if task is None:  # 停止信号
                break
            

            task_id, inputs, next_device = task

            
            # 在独立的CUDA流上执行计算
            outputs = None

            with torch.cuda.stream(self.stream):
                time_start_stream = time.time()
                # 执行layer计算
                cuda_hook("layer_compute")
                outputs, _ = self.layer.block_sparse_moe(inputs)
                cuda_hook_end("layer_compute")
                time_end_stream = time.time()
            with torch.cuda.stream(self.io_stream):
                time_start_move = time.time()
                outputs = outputs.to(next_device, non_blocking=True)
                print(f"outputs.shape: {outputs.device} to {next_device}")
                time_end_move = time.time()
            self.output_queue.put((task_id, outputs))
            print(f"线程 {self.thread_id} (设备 {self.device}) 流计算时间: {time_end_stream - time_start_stream:.6f}s 移动时间: {time_end_move - time_start_move:.6f}s")
            
    def start(self):
        """启动线程"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止线程"""
        if not self.running:
            return
        self.running = False
        self.input_queue.put(None)  # 发送停止信号
        if self.thread:
            self.thread.join(timeout=5.0)
        # 清理CUDA缓存
        torch.cuda.empty_cache()
    
    def submit(self, task_id: int, inputs: torch.Tensor, next_device: str) -> None:
        """提交任务到队列"""
        self.input_queue.put((task_id, inputs, next_device))
    
    def get_result(self, timeout: Optional[float] = None) -> Tuple[int, Optional[torch.Tensor]]:
        """从输出队列获取结果"""
        return self.output_queue.get(timeout=timeout)
    
    def has_result(self) -> bool:
        """检查是否有结果可用"""
        return not self.output_queue.empty()


# 使用示例和测试函数
def create_layer_workers(layer1, layer2, device1: str, device2: str):
    """创建两个layer计算线程"""
    worker1 = LayerComputeThread(layer1, device1, thread_id=1)
    worker2 = LayerComputeThread(layer2, device2, thread_id=2)
    
    worker1.start()
    worker2.start()
    
    return worker1, worker2

def stop_layer_workers(worker1, worker2):
    """停止两个线程"""
    worker1.stop()
    worker2.stop()

print("LayerComputeThread 类已定义，可以使用 create_layer_workers() 创建线程")


times = 5
h1 = 14336
h2 = 4096
dtype=torch.bfloat16
device1 = "cuda:1"
device2 = "cuda:2"
device3 = "cuda:3"

layer1 = model.model.layers[1]
layer2 = model.model.layers[2]
layer1 = layer1.to(device1)
layer2 = layer2.to(device2)

layer1.eval()
layer2.eval()

batch_size = 32
seq_len = 512

inputsb0 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)
inputsb1 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)
inputsb2 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)
inputsb3 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)
inputsb4 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)
inputsb5 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)
inputsb6 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)
inputsb7 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)

hid = model.model.config.hidden_size
inter_size = model.model.config.intermediate_size
expert1 = torch.randn(inter_size, hid, dtype=dtype, device="cpu", pin_memory=True)
expert2 = torch.randn(inter_size, hid, dtype=dtype, device="cpu", pin_memory=True)

expert1_gpu = torch.randn(inter_size, hid, dtype=dtype, device=device1)
expert2_gpu = torch.randn(inter_size, hid, dtype=dtype, device=device2)


# 定义layer计算函数（用于线程中）
def layer_cal(layer, inputs):
    bmoe = layer.block_sparse_moe
    out, _ = bmoe(inputs)
    return out

expert_move_stream = torch.cuda.Stream(device=device1)
expert_move_stream2 = torch.cuda.Stream(device=device2)
def move_layer(stream, expert_gpu, expert_cpu):
    with torch.cuda.stream(stream):
        for _ in range(3*8):
            expert_gpu.copy_(expert_cpu, non_blocking=True)
    return expert_gpu

# 创建包装类，使layer可以被线程调用
class LayerWrapper:
    def __init__(self, layer):
        self.layer = layer
    
    def __call__(self, inputs):
        return layer_cal(self.layer, inputs)

# 创建两个独立的计算线程
print("创建多线程计算系统...")
worker1 = LayerComputeThread(layer1, device1, thread_id=1)
worker2 = LayerComputeThread(layer2, device2, thread_id=2)

worker1.start()
worker2.start()

# 等待线程启动
time.sleep(0.5)

# 测试：提交任务并获取结果
print("\n开始测试多线程计算...")
print(f"GPU内存状态 - Device1: {torch.cuda.memory_allocated(device1)/1024**3:.2f} GB, Device2: {torch.cuda.memory_allocated(device2)/1024**3:.2f} GB")

# 清理CUDA缓存
torch.cuda.empty_cache()

times_list = []
time_start = time.time()

for i in range(times):
    inputs_list = [inputsb0, inputsb1, inputsb2, inputsb3, inputsb4, inputsb5, inputsb6, inputsb7]
    input_queue = queue.Queue()
    for i in range(len(inputs_list)):
        input_queue.put(inputs_list[i])
    worker1_queue = input_queue
    worker2_queue = queue.Queue()

    time_start_once = time.time()

    # move_layer(expert_move_stream, expert1_gpu, expert1)
    # move_layer(expert_move_stream2, expert2_gpu, expert2)

    worker1.submit(task_id=i, inputs=worker1_queue.get(), next_device=device2)
    task_id1, result1 = worker1.get_result()
    worker2_queue.put(result1)

    # 处理每个输入
    results = []
    for idx in range(len(inputs_list)-1):
        # 提交任务到worker1
        print(f"提交任务到worker1")
        inputs1 = worker1_queue.get()
        worker1.submit(task_id=i*len(inputs_list)*2 + idx*2, inputs=inputs1, next_device=device2)
        inputs2 = worker2_queue.get()
        worker2.submit(task_id=i*len(inputs_list)*2 + idx*2 + 1, inputs=inputs2, next_device=device1)
        
        task_id1, result1 = worker1.get_result()
        task_id2, result2 = worker2.get_result()
        worker2_queue.put(result1)

    
    time_end_once = time.time()
    # 同步所有设备
    torch.cuda.synchronize(device=device1)
    torch.cuda.synchronize(device=device2)
 
    # 定期清理CUDA缓存
    torch.cuda.empty_cache()
    times_list.append(round(time_end_once - time_start_once, 6))
    
    print(f"任务 {i}: 完成，耗时 {times_list[-1]:.6f}s")
    print(f"  GPU内存 - Device1: {torch.cuda.memory_allocated(device1)/1024**3:.2f} GB, Device2: {torch.cuda.memory_allocated(device2)/1024**3:.2f} GB")
torch.cuda.empty_cache()
time_end = time.time()
print(f"\n总时间: {time_end - time_start:.6f} 秒")
print(f"每次时间: {times_list}")
print(f"平均时间: {sum(times_list)/len(times_list):.6f} 秒")

# 最终内存状态
print(f"\n最终GPU内存状态:")
print(f"  Device1: {torch.cuda.memory_allocated(device1)/1024**3:.2f} GB / {torch.cuda.max_memory_allocated(device1)/1024**3:.2f} GB (峰值)")
print(f"  Device2: {torch.cuda.memory_allocated(device2)/1024**3:.2f} GB / {torch.cuda.max_memory_allocated(device2)/1024**3:.2f} GB (峰值)")

# 清理所有中间变量
del inputs_list
if 'results' in locals():
    del results
torch.cuda.empty_cache()

# 停止线程
print("\n停止线程...")
worker1.stop()
worker2.stop()
print("所有线程已停止")
torch.cuda.empty_cache()
# 最终清理
torch.cuda.empty_cache()
print("内存清理完成")
