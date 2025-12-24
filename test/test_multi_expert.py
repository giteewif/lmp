
import torch
from torch.nn.attention import  sdpa_kernel
from torch.nn.attention import  SDPBackend
import torch.nn.functional as F
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
        # self.stream = torch.cuda.Stream(device=device)
        # self.io_stream = torch.cuda.Stream(device=device)
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
            

            task_id, layer, expert_id, inputs = task

            out_list = []
            time_start_stream = time.time()
            # with torch.cuda.stream(self.stream):
            for i in range(len(inputs)):
                out = layer.block_sparse_moe.experts[i](inputs[i])
                out_list.append(out)

            cuda_hook_end("layer_experts_gpu2")
            time_end_stream = time.time()
            self.output_queue.put((task_id, out))
            print(f"线程 {self.thread_id} (设备 {self.device}) 流计算时间: {time_end_stream - time_start_stream:.6f}s")
            
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
    
    def submit(self, task_id: int, layer,
               expert_id: int, inputs) -> None:
        """提交任务到队列"""
        self.input_queue.put((task_id, layer,expert_id, inputs))
    
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

layer1.eval()
layer2.eval()

batch_size = 64
seq_len = 1
seq_len_kv = 512

inputsb0 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device="cpu")
inputsb1 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device="cpu")

# torch.set_num_threads(160)
# 创建两个独立的计算线程
print("创建多线程计算系统...")
worker1 = LayerComputeThread(layer1, device1, thread_id=1)
worker2 = LayerComputeThread(layer1, device1, thread_id=2)

worker1.start()
worker2.start()

# 等待线程启动
time.sleep(0.5)


times_list = []
time_start = time.time()

for i in range(times):
    inputs_list = []
    batch_size = 1024
    for i in range(1):
        inputs = torch.randn(batch_size, seq_len, h2, dtype=dtype, device="cpu")
        inputs_list.append(inputs)

    time_start_once = time.time()

    worker1.submit(i, layer1, 0, inputs_list)
    # worker2.submit(i, layer2, 1, inputsb0)

    out1 = worker1.get_result()
    # out2 = worker2.get_result()

    del out1
    # del out2

    time_end_once = time.time()
    times_list.append(round(time_end_once - time_start_once, 6))

print(f"任务 {i}: 完成，耗时 {times_list}s")