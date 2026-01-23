import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union

# 配置参数
hidden_size = 2048
moe_intermediate_size = 1408
num_tokens_per_expert: Union[int, List[int]] = [32, 64, 128]
expert_counts = [4, 8, 16, 32, 64, 128]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 激活函数（SwiGLU: gate * silu(up)）
def swiglu_activation(gate, up):
    """SwiGLU activation: gate * silu(up)"""
    return gate * torch.nn.functional.silu(up)

# 方法1: BMM 批量计算（使用 einsum）
def bmm_method(
    group_w1: torch.Tensor,  # [E, I, H]
    group_w2: torch.Tensor,  # [E, H, I]
    group_w3: torch.Tensor,  # [E, I, H]
    stacked_inputs: torch.Tensor,  # [E, T, H] where T is max_tokens
    expert_token_counts: List[int],  # 每个 expert 的实际 token 数量
    expert_weights: torch.Tensor,  # [total_tokens] 路由权重
    expert_token_indices: List[torch.Tensor],  # 每个 expert 的 token 索引
    final_hidden_states: torch.Tensor
):
    """
    使用批量矩阵乘法计算所有 experts
    """
    # w1: [E, T, H] @ [E, I, H] -> [E, T, I]
    w1_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w1)
    
    # w3: [E, T, H] @ [E, I, H] -> [E, T, I]
    w3_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w3)
    
    # SwiGLU activation: w1_out * silu(w3_out)
    intermediate = swiglu_activation(w1_out, w3_out)
    
    # w2: [E, T, I] @ [E, H, I] -> [E, T, H]
    outputs = torch.einsum('eti,ehi->eth', intermediate, group_w2)
    
    # Scatter 回 final_hidden_states
    for i, (expert_out, token_indices, num_tokens) in enumerate(
        zip(outputs, expert_token_indices, expert_token_counts)
    ):
        expert_out_slice = expert_out[:num_tokens]  # [num_tokens, H]
        # 获取对应的权重
        start_idx = sum(expert_token_counts[:i])
        end_idx = start_idx + num_tokens
        weights = expert_weights[start_idx:end_idx]
        
        # 应用权重
        expert_out_slice = expert_out_slice.mul_(weights.view(-1, 1))
        
        # Scatter 回 final_hidden_states
        final_hidden_states.scatter_reduce_(
            dim=0,
            index=token_indices.view(-1, 1).expand(-1, hidden_size),
            src=expert_out_slice,
            reduce='sum'
        )
    
    return final_hidden_states

# 方法2: 逐个 expert 计算
def sequential_method(
    w1_list: List[torch.Tensor],  # List of [I, H]
    w2_list: List[torch.Tensor],  # List of [H, I]
    w3_list: List[torch.Tensor],  # List of [I, H]
    expert_inputs: List[torch.Tensor],  # List of [T_i, H]
    expert_weights: torch.Tensor,  # [total_tokens] 路由权重
    expert_token_indices: List[torch.Tensor],  # 每个 expert 的 token 索引
    final_hidden_states: torch.Tensor
):
    """
    逐个 expert 计算
    """
    weight_idx = 0
    for w1, w2, w3, tokens, token_indices in zip(
        w1_list, w2_list, w3_list, expert_inputs, expert_token_indices
    ):
        num_tokens = tokens.shape[0]
        
        # w1: [T_i, H] @ [I, H]^T -> [T_i, I]
        w1_out = torch.matmul(tokens, w1.t())
        
        # w3: [T_i, H] @ [I, H]^T -> [T_i, I]
        w3_out = torch.matmul(tokens, w3.t())
        
        # SwiGLU activation: w1_out * silu(w3_out)
        intermediate = swiglu_activation(w1_out, w3_out)
        
        # w2: [T_i, I] @ [H, I]^T -> [T_i, H]
        expert_out = torch.matmul(intermediate, w2.t())
        
        # 应用权重
        weights = expert_weights[weight_idx:weight_idx + num_tokens]
        expert_out = expert_out.mul_(weights.view(-1, 1))
        weight_idx += num_tokens
        
        # Scatter 回 final_hidden_states
        final_hidden_states.scatter_reduce_(
            dim=0,
            index=token_indices.view(-1, 1).expand(-1, hidden_size),
            src=expert_out,
            reduce='sum'
        )
    
    return final_hidden_states

# 生成测试数据
def generate_test_data(num_experts: int, num_tokens_per_expert: Union[int, List[int]]):
    """生成测试数据
    Args:
        num_experts: expert 数量
        num_tokens_per_expert: 可以是 int 或 List[int]。如果是 int，所有 expert 使用相同的 token 数；
                               如果是 List[int]，每个 expert 使用对应的 token 数
    """
    # 处理 num_tokens_per_expert 参数
    if isinstance(num_tokens_per_expert, int):
        expert_token_counts = [num_tokens_per_expert] * num_experts
    elif isinstance(num_tokens_per_expert, list):
        if len(num_tokens_per_expert) != num_experts:
            raise ValueError(f"num_tokens_per_expert 列表长度 ({len(num_tokens_per_expert)}) 必须等于 num_experts ({num_experts})")
        expert_token_counts = num_tokens_per_expert
    else:
        raise TypeError(f"num_tokens_per_expert 必须是 int 或 List[int]，当前类型: {type(num_tokens_per_expert)}")
    
    total_tokens = sum(expert_token_counts)
    max_tokens = max(expert_token_counts)
    
    # 生成权重
    w1_list = [torch.randn(moe_intermediate_size, hidden_size, device=device, dtype=torch.bfloat16) 
                for _ in range(num_experts)]
    w2_list = [torch.randn(hidden_size, moe_intermediate_size, device=device, dtype=torch.bfloat16) 
                for _ in range(num_experts)]
    w3_list = [torch.randn(moe_intermediate_size, hidden_size, device=device, dtype=torch.bfloat16) 
                for _ in range(num_experts)]
    
    # 堆叠为 group tensors (BMM 方法使用)
    group_w1 = torch.stack(w1_list)  # [E, I, H]
    group_w2 = torch.stack(w2_list)  # [E, H, I]
    group_w3 = torch.stack(w3_list)  # [E, I, H]
    
    # 生成输入 tokens
    hidden_states = torch.randn(total_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    
    # 为每个 expert 分配 tokens
    expert_token_indices = []
    expert_inputs = []
    stacked_inputs = torch.zeros(num_experts, max_tokens, hidden_size, 
                                  device=device, dtype=torch.bfloat16)
    
    token_idx = 0
    for i in range(num_experts):
        num_tokens = expert_token_counts[i]
        token_indices = torch.arange(token_idx, token_idx + num_tokens, device=device)
        expert_token_indices.append(token_indices)
        
        expert_tokens = hidden_states[token_idx:token_idx + num_tokens]
        expert_inputs.append(expert_tokens)
        stacked_inputs[i, :num_tokens] = expert_tokens
        
        token_idx += num_tokens
    
    # 生成路由权重
    expert_weights = torch.rand(total_tokens, device=device, dtype=torch.bfloat16)
    
    return {
        'w1_list': w1_list,
        'w2_list': w2_list,
        'w3_list': w3_list,
        'group_w1': group_w1,
        'group_w2': group_w2,
        'group_w3': group_w3,
        'expert_inputs': expert_inputs,
        'stacked_inputs': stacked_inputs,
        'expert_token_indices': expert_token_indices,
        'expert_token_counts': expert_token_counts,
        'expert_weights': expert_weights,
        'hidden_states': hidden_states
    }

# 性能测试
def benchmark_methods(num_experts: int, num_tokens_per_expert: Union[int, List[int]], num_warmup: int = 5, num_runs: int = 20):
    """测试两种方法的性能"""
    data = generate_test_data(num_experts, num_tokens_per_expert)
    
    # Warmup
    for _ in range(num_warmup):
        final_hidden_states_bmm = torch.zeros_like(data['hidden_states'])
        final_hidden_states_seq = torch.zeros_like(data['hidden_states'])
        
        bmm_method(
            data['group_w1'], data['group_w2'], data['group_w3'],
            data['stacked_inputs'], data['expert_token_counts'],
            data['expert_weights'], data['expert_token_indices'],
            final_hidden_states_bmm.clone()
        )
        
        sequential_method(
            data['w1_list'], data['w2_list'], data['w3_list'],
            data['expert_inputs'], data['expert_weights'],
            data['expert_token_indices'],
            final_hidden_states_seq.clone()
        )
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # BMM 方法测试
    bmm_times = []
    for _ in range(num_runs):
        final_hidden_states_bmm = torch.zeros_like(data['hidden_states'])
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        bmm_method(
            data['group_w1'], data['group_w2'], data['group_w3'],
            data['stacked_inputs'], data['expert_token_counts'],
            data['expert_weights'], data['expert_token_indices'],
            final_hidden_states_bmm
        )
        if device == 'cuda':
            torch.cuda.synchronize()
        bmm_times.append(time.time() - start)
    
    # Sequential 方法测试
    seq_times = []
    for _ in range(num_runs):
        final_hidden_states_seq = torch.zeros_like(data['hidden_states'])
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        sequential_method(
            data['w1_list'], data['w2_list'], data['w3_list'],
            data['expert_inputs'], data['expert_weights'],
            data['expert_token_indices'],
            final_hidden_states_seq
        )
        if device == 'cuda':
            torch.cuda.synchronize()
        seq_times.append(time.time() - start)
    
    # 验证结果一致性（允许小的数值误差）
    max_diff = torch.max(torch.abs(final_hidden_states_bmm - final_hidden_states_seq)).item()
    
    return {
        'bmm_mean': np.mean(bmm_times) * 1000,  # 转换为毫秒
        'bmm_std': np.std(bmm_times) * 1000,
        'seq_mean': np.mean(seq_times) * 1000,
        'seq_std': np.std(seq_times) * 1000,
        'speedup': np.mean(seq_times) / np.mean(bmm_times),
        'max_diff': max_diff
    }

print("开始性能测试...")
print(f"设备: {device}")
print(f"hidden_size: {hidden_size}, moe_intermediate_size: {moe_intermediate_size}")
print("-" * 80)

results = {}

# 处理 num_tokens_per_expert 参数
if isinstance(num_tokens_per_expert, int):
    # 如果是单个整数，对所有 expert 使用相同的 token 数
    token_configs = [(num_tokens_per_expert, f"{num_tokens_per_expert} tokens/expert")]
elif isinstance(num_tokens_per_expert, list):
    # 如果是列表，遍历每个配置
    token_configs = []
    for tokens in num_tokens_per_expert:
        token_configs.append((tokens, f"{tokens} tokens/expert"))
else:
    raise TypeError(f"num_tokens_per_expert 必须是 int 或 List[int]，当前类型: {type(num_tokens_per_expert)}")

# 遍历不同的 token 配置
for tokens_per_expert, config_name in token_configs:
    print(f"\n{'=' * 80}")
    print(f"配置: {config_name}")
    print(f"{'=' * 80}")
    
    for num_experts in expert_counts:
        print(f"\n测试 {num_experts} experts...")
        result = benchmark_methods(num_experts, tokens_per_expert)
        key = (num_experts, tokens_per_expert)
        results[key] = result
        print(f"  BMM 方法: {result['bmm_mean']:.3f} ± {result['bmm_std']:.3f} ms")
        print(f"  Sequential 方法: {result['seq_mean']:.3f} ± {result['seq_std']:.3f} ms")
        print(f"  加速比: {result['speedup']:.2f}x")
        print(f"  最大差异: {result['max_diff']:.6f}")

print("\n" + "=" * 80)
print("测试完成！")

# 打印出每个 tokens_per_expert 下的 bmm 和 seq 的 list
print("\n" + "=" * 80)
print("数据汇总：每个 Token 配置下的 BMM 和 Sequential 时间列表")
print("=" * 80)

for tokens_per_expert, config_name in token_configs:
    print(f"\n配置: {config_name}")
    print("-" * 80)
    
    # 收集该配置下的所有数据
    bmm_list = []
    seq_list = []
    expert_nums_list = []
    
    for num_experts in expert_counts:
        key = (num_experts, tokens_per_expert)
        if key in results:
            expert_nums_list.append(num_experts)
            bmm_list.append(results[key]['bmm_mean'])
            seq_list.append(results[key]['seq_mean'])
    
    # 打印列表
    print(f"Expert 数量列表: {expert_nums_list}")
    print(f"BMM 时间列表 (ms): {[f'{x:.3f}' for x in bmm_list]}")
    print(f"Sequential 时间列表 (ms): {[f'{x:.3f}' for x in seq_list]}")
    
    # 也可以打印为 Python 可用的格式
    print(f"\nPython 格式:")
    print(f"bmm_times_{tokens_per_expert} = {bmm_list}")
    print(f"seq_times_{tokens_per_expert} = {seq_list}")
    print(f"expert_counts_{tokens_per_expert} = {expert_nums_list}")

print("\n" + "=" * 80)
print("所有数据汇总完成！")
