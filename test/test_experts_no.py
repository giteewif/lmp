import torch
from torch.nn.attention import  sdpa_kernel
from torch.nn.attention import  SDPBackend
import torch.nn.functional as F
from transformers import MixtralForCausalLM, AutoModelForCausalLM
import threading
import queue
import time
from typing import Optional, Tuple, Dict

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


def einsum_experts(
    layer, 
    expert_tokens_map: Dict[int, torch.Tensor],
):
    time_all = time.time()
    expert_indices = list(expert_tokens_map.keys())
    max_tokens = max(tokens.shape[0] for tokens in expert_tokens_map.values())

    # Padding并堆叠
    stacked_inputs = []
    masks = []
    w1_list, w2_list, w3_list = [], [], []

    cuda_hook("pad token")
    time_start_pad = time.time()
    for expert_idx in expert_indices:
        tokens = expert_tokens_map[expert_idx]
        num_tokens = tokens.shape[0]
        
        # Padding
        if num_tokens < max_tokens:
            padding = torch.zeros(
                max_tokens - num_tokens, tokens.shape[1],
                dtype=tokens.dtype, device=tokens.device
            )
            padded_tokens = torch.cat([tokens, padding], dim=0)
        else:
            padded_tokens = tokens
        
        stacked_inputs.append(padded_tokens)
        
        # 创建mask
        mask = torch.zeros(max_tokens, dtype=torch.bool, device=tokens.device)
        mask[:num_tokens] = True
        masks.append(mask)
        
        # 收集权重
        expert = layer.block_sparse_moe.experts[expert_idx]
        w1_list.append(expert.w1.weight)
        w2_list.append(expert.w2.weight)
        w3_list.append(expert.w3.weight)
    time_end_pad = time.time()
    cuda_hook_end("pad token")

    cuda_hook("weights stacked")
    
    # 堆叠
    time_start_stack_inputs = time.time()
    stacked_inputs = torch.stack(stacked_inputs)  # [E, max_tokens, H]
    print(f"stack inputs cost {time.time() - time_start_stack_inputs} s")
    time_start_stacked = time.time()
    
    # 注意：torch.stack 会复制数据，即使输入 tensor 在连续地址空间
    # 对于 expert 权重（来自不同的 expert 对象），它们通常不在连续地址空间
    # 所以必须使用 stack 来创建新的维度并复制数据
    # 
    # 如果权重确实在连续地址空间且共享底层存储，可以考虑：
    # 1. 使用 torch.cat + view（如果维度允许）
    # 2. 直接使用 view/reshape（如果它们共享存储）
    # 但这种情况很少见，因为不同 expert 的权重是独立分配的
    
    # 检查是否可以避免复制（可选优化）
    def try_stack_without_copy(weight_list):
        """
        如果权重在连续地址空间，尝试避免复制
        
        计算效率分析：
        1. 内存复制开销：
           - torch.stack: O(E*I*H) 的数据复制
           - as_strided: O(1) 操作，无数据复制，节省内存和时间
        
        2. 后续 einsum 计算效率：
           - 如果 stride 相同（都是 [I*H, H, 1]），计算效率相同
           - as_strided 创建的 tensor 与 stack 创建的 tensor 在 einsum 中的表现应该相同
           - 因为内存布局相同，缓存友好性相同
        
        3. 潜在问题：
           - as_strided 创建的 tensor 可能不是"连续"的（从 PyTorch 的角度）
           - 某些操作（如某些 CUDA kernel）可能需要连续化，这会触发复制
           - 但对于 einsum，通常不需要连续化，所以效率应该相同
        """
        if len(weight_list) <= 1:
            return torch.stack(weight_list)
        
        first = weight_list[0]
        
        # 检查1：是否所有权重共享同一个底层存储（很少见）
        if all(w.data_ptr() == first.data_ptr() and w.shape == first.shape for w in weight_list):
            # 如果共享存储，可以使用 view（但这种情况几乎不存在）
            E = len(weight_list)
            if len(first.shape) == 2:
                dim0, dim1 = first.shape[0], first.shape[1]
                return first.view(1, dim0, dim1).expand(E, dim0, dim1).contiguous()
            else:
                # 对于其他维度，回退到 stack
                return torch.stack(weight_list)
        
        # 检查2：是否所有权重在连续地址空间中（来自 RestoreExpertsFromSharedMemory）
        # 检查：所有 tensor 形状相同，且地址连续
        if all(w.shape == first.shape and w.dtype == first.dtype for w in weight_list):
            # 获取所有 tensor 的数据指针
            data_ptrs = [w.data_ptr() for w in weight_list]
            data_ptrs_sorted = sorted(data_ptrs)
            
            # 检查地址是否连续
            element_size = first.element_size()
            tensor_size = first.numel() * element_size
            
            is_contiguous = True
            for i in range(len(data_ptrs_sorted) - 1):
                expected_next = data_ptrs_sorted[i] + tensor_size
                if data_ptrs_sorted[i + 1] != expected_next:
                    is_contiguous = False
                    break
            
            if is_contiguous:
                # 在连续地址空间中，创建一个大的 tensor 然后用 view 分割
                # 找到第一个 tensor 的地址
                first_ptr = min(data_ptrs)
                first_idx = data_ptrs.index(first_ptr)
                first_tensor = weight_list[first_idx]
                
                # 创建 [E, ...] 形状的大 tensor
                E = len(weight_list)
                
                # 支持任意 2D 形状的 tensor
                if len(first_tensor.shape) == 2:
                    dim0, dim1 = first_tensor.shape[0], first_tensor.shape[1]
                    big_shape = (E, dim0, dim1)
                    
                    # 计算 stride：确保与 torch.stack 创建的 tensor 的 stride 相同
                    # torch.stack 创建的 tensor stride 取决于原始 tensor 的形状
                    # 对于 [dim0, dim1] 形状的 tensor，stack 后的 stride 是 [dim0*dim1, dim1, 1]
                    # 
                    # 注意：stride 是以元素为单位的，不是字节
                    # stride[0] = dim0*dim1 (从第 i 个 tensor 到第 i+1 个 tensor)
                    # stride[1] = dim1 (从第 j 行到第 j+1 行，与原始 tensor 相同)
                    # stride[2] = 1 (每个元素之间的 stride)
                    big_stride = (dim0 * dim1, dim1, 1)
                    
                    # 创建大 tensor（共享底层存储）
                    # 使用 as_strided 创建，指向第一个 tensor 的地址
                    # 这避免了数据复制，但 stride 与 stack 相同，所以计算效率相同
                    big_tensor = torch.as_strided(
                        first_tensor,
                        size=big_shape,
                        stride=big_stride
                    )
                    
                    # 验证：检查 big_tensor 的每个切片是否对应原始 tensor
                    # 如果验证通过，返回 big_tensor（避免复制）
                    try:
                        # 验证第一个和最后一个 tensor
                        if (big_tensor[0].data_ptr() == first_tensor.data_ptr()):
                            last_idx = data_ptrs.index(max(data_ptrs))
                            if big_tensor[-1].data_ptr() == weight_list[last_idx].data_ptr():
                                # 验证 stride 是否正确（应该与 stack 创建的 tensor 相同）
                                # 如果 stride 相同，计算效率应该是一样的
                                return big_tensor
                    except:
                        pass
                else:
                    # 对于非 2D tensor，回退到 stack
                    pass
        
        # 标准情况：使用 stack（会复制）
        return torch.stack(weight_list)
    
    # 对于 expert 权重，尝试使用优化方法避免复制
    w1_weights = try_stack_without_copy(w1_list)  # [E, I, H] - 如果连续则避免复制
    w2_weights = try_stack_without_copy(w2_list)  # [E, H, I] - 如果连续则避免复制
    w3_weights = try_stack_without_copy(w3_list)  # [E, I, H] - 如果连续则避免复制
    time_end_stacked = time.time()
    cuda_hook_end("weights stacked")

    cuda_hook("einsum compute")
    time_start_exec = time.time()
    # 使用einsum批量计算
    w1_out = torch.einsum('eth,eih->eti', stacked_inputs, w1_weights)
    w1_out = F.silu(w1_out)
    w3_out = torch.einsum('eth,eih->eti', stacked_inputs, w3_weights)
    intermediate = w1_out * w3_out
    outputs = torch.einsum('eti,ehi->eth', intermediate, w2_weights)
    time_end_exec = time.time()
    print(f"    once einsum {round(time_end_exec-time_start_exec, 6)}s")
    cuda_hook_end("einsum compute end")

    cuda_hook("mask results")
    # 提取有效结果
    results = {}
    time_start_result = time.time()
    for i, expert_idx in enumerate(expert_indices):
        expert_outputs = outputs[i][masks[i]]
        # if expert_idx in routing_weights:
        #     expert_outputs = expert_outputs * routing_weights[expert_idx]
        results[expert_idx] = expert_outputs
    cuda_hook_end("mask results")
    print(f"pad {round(time_end_pad-time_start_pad, 6)} "
    f"stacked {round(time_end_stacked-time_start_stacked,6)} this {time.time()-time_all} s"
    f"resutl {round(time.time()-time_start_result,6)} "
    )
    return results, round(time_end_exec-time_start_exec, 6)
times = 5
time_list = []
time_once_lists = []
time_start_all = time.time()
inputs_tokens = {}
dtype=torch.bfloat16
h2 = 4096
layer1 = model.model.layers[1]
layer2 = model.model.layers[2]
device = "cuda:1"
layer1 = layer1.to("cuda:1")
torch.set_num_threads(128)

inputs = torch.randn(64, 64, 4096, dtype=dtype, device=device)
inputs_cpu = torch.randn(64, 64, 4096, dtype=dtype, device="cpu")
for i in range(times):
    time_start_once = time.time()
    out = layer1(inputs)
    time_end_once = time.time() - time_start_once
    time_list.append(round(time_end_once, 6))

    # time_start_once = time.time()
    # out = layer2(inputs_cpu)
    # time_end_once = time.time() - time_start_once
    # time_once_lists.append(round(time_end_once, 6))
print(f"{time_list} {time_once_lists}")