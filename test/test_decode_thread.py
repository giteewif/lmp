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
class CPUAttnComputeThread:
    def __init__(self, thread_id: int):
        torch.set_num_threads(128)
        self.thread_id = thread_id
        self.running = False
        self.thread = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
    def _worker(self):
        """工作线程主循环"""
        print(f"线程 {self.thread_id} 启动")
        while self.running:
            task = None
            task_id = None
            # 从队列获取任务，超时1秒
            task = self.input_queue.get()
            
            if task is None:  # 停止信号
                break
            
            task_id, query_states, key_states, value_states = task
            
            time_start_stream = time.time()
            # 在独立的CUDA流上执行计算
            outputs = None
            cuda_hook("cpu attn")
            outputs = scaled_dot_product_attention_help(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            cuda_hook_end("cpu attn")
            time_end_stream = time.time()
            self.output_queue.put((task_id, outputs))
            print(f"线程 {self.thread_id} 流计算时间: {time_end_stream - time_start_stream:.6f}s")
    def submit(self, task_id: int, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor):
        self.input_queue.put((task_id, query_states, key_states, value_states))
    def get_result(self, timeout: Optional[float] = None) -> Tuple[int, Optional[torch.Tensor]]:
        return self.output_queue.get(timeout=timeout)
    def has_result(self) -> bool:
        return not self.output_queue.empty()
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
            

            task_id, layer, expert_loading_list, expert_idx_list, flat_hidden_states, \
                expert_mask, routing_weights, final_hidden_states = task

            time_start_stream = time.time()
            # 在独立的CUDA流上执行计算
            outputs = None
            cuda_hook("layer_experts_gpu1")
            with torch.cuda.stream(self.stream):
                layer_experts_gpu(
                    layer=layer,
                    expert_idx_list=expert_loading_list, 
                    flat_hidden_states=flat_hidden_states, 
                    expert_mask=expert_mask, routing_weights=routing_weights, 
                    final_hidden_states=final_hidden_states, 
                    cuda_device=self.device
                )
            cuda_hook_end("layer_experts_gpu1")
            cuda_hook("layer_experts_gpu2")
            with torch.cuda.stream(self.io_stream):
                for expert_idx in expert_idx_list:
                    load_experts(layer=layer, expert_id=expert_idx, device=self.device)
                # torch.cuda.synchronize(device=self.device)
                print_layer_parameters(layer=layer, device=f"{self.device} after load_experts")
                layer_experts_gpu(
                    layer=layer,
                    expert_idx_list=expert_idx_list, 
                    flat_hidden_states=flat_hidden_states, 
                    expert_mask=expert_mask, routing_weights=routing_weights, 
                    final_hidden_states=final_hidden_states, 
                    cuda_device=self.device
                )
                cuda_hook_end("layer_experts_gpu2")
                time_end_stream = time.time()
            self.output_queue.put((task_id, final_hidden_states))
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
               expert_loading_list: list,
               expert_idx_list: list, 
               flat_hidden_states: torch.Tensor, 
               expert_mask: torch.Tensor, 
               routing_weights: torch.Tensor, 
               final_hidden_states: torch.Tensor) -> None:
        """提交任务到队列"""
        self.input_queue.put((task_id, layer,
            expert_loading_list,
            expert_idx_list, flat_hidden_states, 
            expert_mask, routing_weights, final_hidden_states))
    
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
# layer1 = layer1.to(device1)
# layer2 = layer2.to(device2)

layer1.eval()
layer2.eval()

batch_size = 512
seq_len = 1
seq_len_kv = 512

inputsb0 = torch.randn(batch_size, seq_len, h2, dtype=dtype, device=device1)

num_heads = model.model.config.num_attention_heads
num_kv_heads = model.model.config.num_key_value_heads
head_dim = model.model.config.hidden_size // model.model.config.num_attention_heads

key = torch.randn(batch_size, num_kv_heads, seq_len_kv, head_dim, device="cpu", dtype=dtype)
value = torch.randn(batch_size, num_kv_heads, seq_len_kv, head_dim, device="cpu", dtype=dtype)

hid = model.model.config.hidden_size
inter_size = model.model.config.intermediate_size
expert1 = torch.randn(inter_size, hid, dtype=dtype, device="cpu", pin_memory=True)
expert2 = torch.randn(inter_size, hid, dtype=dtype, device="cpu", pin_memory=True)
expert3 = torch.randn(hid, inter_size, dtype=dtype, device="cpu", pin_memory=True)

expert1_device2 = torch.randn(inter_size, hid, dtype=dtype, device="cpu", pin_memory=True)
expert2_device2 = torch.randn(inter_size, hid, dtype=dtype, device="cpu", pin_memory=True)
expert3_device2 = torch.randn(hid, inter_size, dtype=dtype, device="cpu", pin_memory=True)

expert1_gpu = torch.randn(inter_size, hid, dtype=dtype, device=device1)
expert2_gpu = torch.randn(inter_size, hid, dtype=dtype, device=device2)


# 定义layer计算函数（用于线程中）
def layer_cal(layer, inputs):
    bmoe = layer.block_sparse_moe
    out, _ = bmoe(inputs)
    return out
def qkv(layer, hidden_states):
    bsz, q_len, _ = hidden_states.size()
    print(f"hidden_states device {hidden_states.device}, layer input_layernorm device {layer.input_layernorm.weight.device}")
    hidden_states=layer.input_layernorm(hidden_states)
    query_states = layer.self_attn.q_proj(hidden_states)
    key_states = layer.self_attn.k_proj(hidden_states)
    value_states = layer.self_attn.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2] + 512
    cos, sin = layer.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)

    return query_states, key_states, value_states, cos, sin
@torch.no_grad()
def scaled_dot_product_attention_help(
    query_states, 
    key_states, 
    value_states, 
    attn_mask=None, dropout_p=0.0, enable_gqa=False, is_causal=False, output_tensor=None):

    time_start = time.time()
   
    num_query_heads = query_states.shape[1]     # e.g. 32
    num_key_heads = key_states.shape[1]
    num_groups = int(num_query_heads//num_key_heads)   # 4 组
    
    query_states = query_states
   
    if output_tensor is None:
        output_tensor = torch.zeros(
            query_states.shape, dtype=query_states.dtype, device=query_states.device, pin_memory=False
        )
    else:
        output_tensor = output_tensor.contiguous()
    
    query_groups = []
    query_indices_list = []
    
    for group_idx in range(num_groups):
        query_indices = torch.arange(group_idx, num_query_heads, num_groups)
        query_group = query_states[:, query_indices, :, :]  # 确保连续内存
        query_groups.append(query_group)
        query_indices_list.append(query_indices)
    
    for group_idx in range(num_groups):
        query_group = query_groups[group_idx]
        query_indices = query_indices_list[group_idx]
        
        
        # 优化5: 使用预取的数据，避免重复内存访问
        key_group = key_states    # (batch, 8, seq_len, head_dim)
        value_group = value_states # (batch, 8, seq_len, head_dim)

        time_start_tmp = time.time()
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                query_group, key_group, value_group,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                enable_gqa=enable_gqa,
                is_causal=is_causal
            )
        print(f"single group {group_idx} real attn out cost {time.time() - time_start_tmp} s")
        
        time_start_cpy = time.time()
        output_tensor[:, query_indices, :, :] = attn_out
        print(f"write to output tensor cost {time.time() - time_start_cpy} s")
    print(f"dot attn help cost {time.time()-time_start:.6f} seconds")
    return output_tensor

def attn(layer, query_states, key_states, value_states):
    with torch.no_grad():
        attn_output = scaled_dot_product_attention_help(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=False,
        )
    return attn_output
def moe_infer_prepare(layer, hidden_states):
        # 存在中间值 identity
        identity = hidden_states

        _, _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = layer.block_sparse_moe.gate(hidden_states)


        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, layer.block_sparse_moe.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)


        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=layer.block_sparse_moe.num_experts)
        # 计算激活的expert数量
        # expert_mask的shape为 [batch*seq, top_k, num_experts]，其中每个token的top_k个被选中的expert位置为1，其余为0
        # 对batch*seq和top_k两个维度求和，得到每个expert被选中的总次数
        # 然后判断每个expert是否至少被选中过一次（>0），最后统计被激活的expert数量
        # 统计每个expert被激活（被分配到token）的token数
        tokens_per_expert = expert_mask.sum(dim=(0, 1)).tolist()
        num_active_experts = (expert_mask.sum(dim=(0, 1)) > 0).sum().item()
        print(f"Tokens per expert: {tokens_per_expert}")
        print(f"Number of active experts: {num_active_experts}")
        expert_mask = expert_mask.permute(2, 1, 0)
        return hidden_states, expert_mask, routing_weights, identity
def og(layer, hidden_states, o_hidden_states):
    bsz, q_len, hidden_size = hidden_states.shape
    o_hidden_states = o_hidden_states.transpose(1, 2).contiguous()
    o_hidden_states = o_hidden_states.reshape(
        bsz, #bsz
        q_len, #q_len
        -1
    )
    residual=hidden_states
    hidden_states=layer.self_attn.o_proj(o_hidden_states)
    hidden_states=residual+hidden_states
    residual=hidden_states
    hidden_states=layer.post_attention_layernorm(hidden_states)
    hidden_states, expert_mask, routing_weights, _ = moe_infer_prepare(layer, hidden_states)

    return hidden_states, expert_mask, routing_weights, residual

def layer_experts_gpu(layer, expert_idx_list, flat_hidden_states, 
    expert_mask, routing_weights, final_hidden_states, cuda_device):
    hidden_dim = final_hidden_states.shape[-1]

    expert_tokens_map = {}  # {expert_idx: (top_x_list, idx_list)}
    expert_mask_gpu = expert_mask
    for expert_idx in expert_idx_list:
        idx, top_x = torch.where(expert_mask_gpu[expert_idx])
        if idx.numel() > 0:
            expert_tokens_map[expert_idx] = (top_x, idx)
    if len(expert_tokens_map) == 0:
        return final_hidden_states

    time_start_move2gpu = time.time()
    expert_routing_map = {}  # {expert_idx: routing_weights_tensor}
    expert_states_map = {}
    expert_results_gpu ={}
    for expert_idx, (top_x, idx) in expert_tokens_map.items():
        expert_routing_map[expert_idx] = routing_weights[top_x, idx, None].to(cuda_device)
        expert_states_map[expert_idx] = flat_hidden_states[None, top_x].reshape(-1, hidden_dim).to(cuda_device)
    time_move2gpu = time.time() - time_start_move2gpu
        
    time_start_gpu_compute = time.time()
    for expert_idx, (top_x, idx) in expert_tokens_map.items():
        expert_layer = layer.block_sparse_moe.experts[expert_idx]
        expert_current_states = expert_states_map[expert_idx]
        expert_hidden_states = expert_layer(expert_current_states) * expert_routing_map[expert_idx]
        expert_results_gpu[expert_idx] = (expert_hidden_states, top_x)
    time_gpu_compute = time.time() - time_start_gpu_compute 

    time_start_move2gpu = time.time()
    for expert_idx, (expert_hidden_states, top_x) in expert_results_gpu.items():
        # 先移动到 GPU，再执行 index_add_
        expert_hidden_states_gpu = expert_hidden_states.to(dtype=flat_hidden_states.dtype, device=final_hidden_states.device)
        final_hidden_states.index_add_(0, top_x, expert_hidden_states_gpu)
    time_move2gpu = time.time() - time_start_move2gpu

    time_end = time.time()
    print(
        f"decoder_mlp_expert_ce_gpu experts {expert_idx_list}"
        f"move2gpu {time_move2gpu:.6f}s, gpu_compute {time_gpu_compute:.6f}s, move2gpu {time_move2gpu:.6f}s, "
        f"total {time_end - time_start:.6f}s"
    )
    cuda_hook_end("decoder_mlp_expert_ce_gpu end")
    
    return final_hidden_states
def layer_experts_cpu(layer, expert_idx_list, flat_hidden_states, expert_mask, routing_weights, final_hidden_states):
    hidden_dim = final_hidden_states.shape[-1]
    device = flat_hidden_states.device

    expert_tokens_map = {}  # {expert_idx: (top_x_list, idx_list)}
    expert_mask_cpu = expert_mask
    for expert_idx in expert_idx_list:
        idx, top_x = torch.where(expert_mask_cpu[expert_idx])
        if idx.numel() > 0:
            expert_tokens_map[expert_idx] = (top_x, idx)
    if len(expert_tokens_map) == 0:
        return final_hidden_states

    time_start_move2cpu = time.time()
    expert_routing_map = {}  # {expert_idx: routing_weights_tensor}
    expert_states_map = {}
    expert_results_cpu ={}
    for expert_idx, (top_x, idx) in expert_tokens_map.items():
        expert_routing_map[expert_idx] = routing_weights[top_x, idx, None].to("cpu")
        expert_states_map[expert_idx] = flat_hidden_states[None, top_x].reshape(-1, hidden_dim).to("cpu")
    time_move2cpu = time.time() - time_start_move2cpu
        
    time_start_cpu_compute = time.time()
    for expert_idx, (top_x, idx) in expert_tokens_map.items():
        expert_layer = layer.block_sparse_moe.experts[expert_idx]
        expert_current_states = expert_states_map[expert_idx]
        expert_hidden_states = expert_layer(expert_current_states) * expert_routing_map[expert_idx]
        expert_results_cpu[expert_idx] = (expert_hidden_states, top_x)
    time_cpu_compute = time.time() - time_start_cpu_compute 

    time_start_move2gpu = time.time()
    print(f"flat_hidden_states device {flat_hidden_states.device}, expert_mask device {expert_mask.device} routing_weights device {routing_weights.device}")
    for expert_idx, (expert_hidden_states, top_x) in expert_results_cpu.items():
        # 先移动到 GPU，再执行 index_add_
        expert_hidden_states_gpu = expert_hidden_states.to(dtype=flat_hidden_states.dtype, device=flat_hidden_states.device)
        final_hidden_states.index_add_(0, top_x, expert_hidden_states_gpu)
    time_move2gpu = time.time() - time_start_move2gpu

    time_end = time.time()
    print(
        f"decoder_mlp_expert_ce_cpu experts {expert_idx_list}"
        f"move2cpu {time_move2cpu:.6f}s, cpu_compute {time_cpu_compute:.6f}s, move2gpu {time_move2gpu:.6f}s, "
        f"total {time_end - time_start:.6f}s"
    )
    cuda_hook_end("decoder_mlp_expert_ce_cpu end")
    
    return final_hidden_states
def get_device_expert_list(expert_mask, expert_device1, expert_device2):
    """
    将experts分配到CPU和GPU上
    - GPU总共分配num_new_experts_gpu个专家（激活多的）
    - 其他专家分配给CPU（优先激活token少的）
    - GPU内部：激活多的在GPU2，激活少的在GPU1
    
    Args:
        expert_mask: shape [num_experts, top_k, batch*seq]，表示每个expert的激活情况
        expert_device1: 已经在GPU1上的expert列表
        expert_device2: 已经在GPU2上的expert列表
    
    Returns:
        expert_list_new_cpu: 分配给CPU的expert列表（优先激活少的）
        expert_device1_new: 分配给GPU1的expert列表（激活较少的）
        expert_device2_new: 分配给GPU2的expert列表（激活较多的）
    """
    expert_list_new_cpu = []
    expert_device1_new = []
    expert_device2_new = []
    num_new_experts_gpu = 0

    # 计算每个expert被激活的token总数
    tokens_per_expert = expert_mask.sum(dim=(1, 2))

    # 找到所有候选experts（不在已有设备上的）
    candidate_experts = []
    for i in range(len(tokens_per_expert)):
        if i not in expert_device1 and i not in expert_device2:
            candidate_experts.append((i, tokens_per_expert[i].item()))
    
    if len(candidate_experts) == 0:
        return expert_list_new_cpu, expert_device1_new, expert_device2_new
    
    # 按激活token数从多到少排序
    candidate_experts.sort(key=lambda x: x[1], reverse=True)
    
    # 分配策略：
    # 1. 前num_new_experts_gpu个激活最多的分配给GPU
    # 2. 剩余的分配给CPU（优先激活少的，即从后往前取）
    
    # 分配给GPU的专家（激活最多的前num_new_experts_gpu个）
    gpu_experts = candidate_experts[:num_new_experts_gpu]
    
    # GPU内部分配：激活多的在GPU2，激活少的在GPU1
    # gpu_experts已经是从多到少排序的
    if len(gpu_experts) == 1:
        # 如果只有一个GPU expert，分配给GPU2（激活最多的）
        expert_device2_new.append(gpu_experts[0][0])
    else:
        # 多个GPU experts：激活更多的分配给GPU2，激活较少的分配给GPU1
        # 将gpu_experts分成两部分：前半部分（激活更多）给GPU2，后半部分（激活较少）给GPU1
        mid_point = len(gpu_experts) // 2
        for i, (expert_idx, token_count) in enumerate(gpu_experts):
            if i < mid_point:
                # 激活更多的分配给GPU2
                expert_device2_new.append(expert_idx)
            else:
                # 激活较少的分配给GPU1
                expert_device1_new.append(expert_idx)
    
    # 分配给CPU的专家（剩余的，优先激活少的）
    cpu_experts = candidate_experts[num_new_experts_gpu:]
    # cpu_experts已经是按激活数从多到少排序的，但CPU优先要激活少的，所以反转
    cpu_experts_sorted = sorted(cpu_experts, key=lambda x: x[1])  # 从少到多排序
    for expert_idx, _ in cpu_experts_sorted:
        expert_list_new_cpu.append(expert_idx)
    
    # 创建expert_idx到token_count的映射，用于打印
    expert_token_map = {expert_idx: token_count for expert_idx, token_count in candidate_experts}
    
    # 计算每个设备的总token数
    cpu_total_tokens = sum(expert_token_map[idx] for idx in expert_list_new_cpu)
    gpu1_total_tokens = sum(expert_token_map[idx] for idx in expert_device1_new)
    gpu2_total_tokens = sum(expert_token_map[idx] for idx in expert_device2_new)
    
    # 打印分配结果，包含每个专家的token数和总token数
    print(f"Expert分配结果:")
    print(f"  CPU experts (共{len(expert_list_new_cpu)}个, 优先激活少的):")
    if expert_list_new_cpu:
        cpu_details = [f"expert_{idx}({expert_token_map[idx]} tokens)" for idx in expert_list_new_cpu]
        print(f"    {cpu_details}")
        print(f"    总token数: {cpu_total_tokens}")
    else:
        print(f"    无")
    
    print(f"  GPU1 experts (共{len(expert_device1_new)}个, 激活较少的):")
    if expert_device1_new:
        gpu1_details = [f"expert_{idx}({expert_token_map[idx]} tokens)" for idx in expert_device1_new]
        print(f"    {gpu1_details}")
        print(f"    总token数: {gpu1_total_tokens}")
    else:
        print(f"    无")
    
    print(f"  GPU2 experts (共{len(expert_device2_new)}个, 激活较多的):")
    if expert_device2_new:
        gpu2_details = [f"expert_{idx}({expert_token_map[idx]} tokens)" for idx in expert_device2_new]
        print(f"    {gpu2_details}")
        print(f"    总token数: {gpu2_total_tokens}")
    else:
        print(f"    无")
    
    return expert_list_new_cpu, expert_device1_new, expert_device2_new

def load_experts_device2(layer, expert_id, device):
    w1_gpu = expert1_device2.to(device, non_blocking=True)
    w2_gpu = expert3_device2.to(device, non_blocking=True)
    w3_gpu = expert2_device2.to(device, non_blocking=True)
    layer.block_sparse_moe.experts[expert_id].w1.weight.data = w1_gpu
    layer.block_sparse_moe.experts[expert_id].w2.weight.data = w2_gpu
    layer.block_sparse_moe.experts[expert_id].w3.weight.data = w3_gpu


def load_experts(layer, expert_id, device):
    if device == "cuda:2":
        load_experts_device2(layer=layer, expert_id=expert_id, device=device)
        return
    else:
        w1_gpu = expert1.to(device, non_blocking=True)
        w2_gpu = expert3.to(device, non_blocking=True)
        w3_gpu = expert2.to(device, non_blocking=True)
        layer.block_sparse_moe.experts[expert_id].w1.weight.data = w1_gpu
        layer.block_sparse_moe.experts[expert_id].w2.weight.data = w2_gpu
        layer.block_sparse_moe.experts[expert_id].w3.weight.data = w3_gpu



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

attn_thread = CPUAttnComputeThread(thread_id=3)

worker1.start()
worker2.start()

attn_thread.start()

# 等待线程启动
time.sleep(0.5)

# 测试：提交任务并获取结果
print("\n开始测试多线程计算...")
print(f"GPU内存状态 - Device1: {torch.cuda.memory_allocated(device1)/1024**3:.2f} GB, Device2: {torch.cuda.memory_allocated(device2)/1024**3:.2f} GB")

# 清理CUDA缓存
torch.cuda.empty_cache()

times_list = []
time_start = time.time()

def print_layer_parameters(layer, device):
    return
    print("=" * 60)
    print(f"layer 参数位置 (设备: {device}):")
    print("=" * 60)
    for name, param in layer.named_parameters():
        print(f"  {name}: {param.device} (shape: {param.shape})")
    print("=" * 60)

layer1_gpu = layer1.to(device1)
layer_cpu_device = layer2.to("cpu")

print_layer_parameters(layer=layer1_gpu, device=device1)

torch.cuda.synchronize(device=device1)
move_stream1 = torch.cuda.Stream(device=device1)
move_stream2 = torch.cuda.Stream(device=device2)
move_attn_stream = torch.cuda.Stream(device=device1)
# attn_output = torch.randn()

for i in range(times):
    print_layer_parameters(layer=layer1_gpu, device=device1)
    # print_layer_parameters(layer=layer1_gpu, device=device1)
    time_start_once = time.time()

    inputs = inputsb0
    
    query_states, _, _, cos, sin = qkv(layer1_gpu, inputs)

    load_in_attn = 1
    
    
    # expert_device1 = []
    # expert_device2 = []

    # cuda_hook("attn")
    # print(f"{query_states.shape}, {key.shape}, {value.shape}")
    # attn_output = attn(layer1_gpu, query_states.to("cpu"), key, value)
    # attn_output = attn_output.pin_memory()
    # attn_output = attn_output.to(device1, non_blocking=True)
    # print(f"attnoutput shape {attn_output.shape}")
    # cuda_hook_end("attn")  
    attn_thread.submit(task_id=i, query_states=query_states.to("cpu"), key_states=key, value_states=value)

    with torch.cuda.stream(move_stream1):
        load_experts(layer_cpu_device, 0, device1)
    with torch.cuda.stream(move_stream2):
        load_experts(layer_cpu_device, 3, device2)

    with torch.cuda.stream(move_stream1):
        # load_experts(layer_cpu_device, 0, device1)
        load_experts(layer_cpu_device, 1, device1)
        load_experts(layer_cpu_device, 2, device1)
    with torch.cuda.stream(move_stream2):
        # load_experts(layer_cpu_device, 3, device2)
        load_experts(layer_cpu_device, 4, device2)
        load_experts(layer_cpu_device, 5, device2)
    expert_device1 = [0, 1, 2]
    expert_device2 = [3, 4, 5]

    task_id, attn_output = attn_thread.get_result()
    if attn_output is None:
        print(f"task {i} attn output is None")
        continue
    attn_output = attn_output.pin_memory()
    # 独立避免阻塞
    with torch.cuda.stream(move_attn_stream):
        attn_output = attn_output.to(device1, non_blocking=True)

    flat_hidden_states, expert_mask, routing_weights, residual = og(layer1_gpu, inputs, attn_output)
    
    cpu_experts_list, expert_device1_new, expert_device2_new = get_device_expert_list(
            expert_mask, expert_device1, expert_device2)
    
    bsz, sequence_length, hidden_dim = residual.shape
    final_hidden_states = torch.zeros(
        (bsz * sequence_length, hidden_dim), dtype=flat_hidden_states.dtype, device=flat_hidden_states.device
    )
    worker2.submit(
        task_id=i, layer=layer_cpu_device, 
        expert_loading_list=expert_device2,
        expert_idx_list=expert_device2_new, 
        flat_hidden_states=flat_hidden_states,
        expert_mask=expert_mask,
        routing_weights=routing_weights,
        final_hidden_states=final_hidden_states
    )
    worker1.submit(
        task_id=i, layer=layer_cpu_device, 
        expert_loading_list=expert_device1,
        expert_idx_list=expert_device1_new, 
        flat_hidden_states=flat_hidden_states,
        expert_mask=expert_mask,
        routing_weights=routing_weights,
        final_hidden_states=final_hidden_states
    )
    
    cuda_hook("cpu experts")
    print_layer_parameters(layer=layer_cpu_device, device="before cpu")
    out = layer_experts_cpu(
        layer=layer_cpu_device, expert_idx_list=cpu_experts_list, 
        flat_hidden_states=flat_hidden_states, expert_mask=expert_mask, 
        routing_weights=routing_weights, final_hidden_states=final_hidden_states)
    cuda_hook_end("cpu experts")
    final_hidden_states = final_hidden_states + residual

    worker1.get_result()
    worker2.get_result()
    
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


torch.cuda.empty_cache()

# 停止线程
print("\n停止线程...")
worker1.stop()
worker2.stop()
attn_thread.stop()
print("所有线程已停止")
torch.cuda.empty_cache()
# 最终清理
torch.cuda.empty_cache()
print("内存清理完成")
