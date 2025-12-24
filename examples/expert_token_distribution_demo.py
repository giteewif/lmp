"""
演示 MoE 模型中 expert token 分配的完整流程
展示如何从 gate 输出构建每个 expert 的 tokens 和 weights
"""

import torch
import numpy as np


def demo_expert_token_distribution():
    """演示专家 token 分配流程"""
    
    print("="*60)
    print("MoE Expert Token Distribution Demo")
    print("="*60)
    
    # 配置参数
    batch_size = 2
    seq_len = 3
    hidden_dim = 8
    num_experts = 4
    num_experts_per_tok = 2
    
    print(f"\n配置:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_experts: {num_experts}")
    print(f"  num_experts_per_tok: {num_experts_per_tok}")
    print(f"  total_tokens: {batch_size * seq_len} = {batch_size} * {seq_len}")
    
    # 模拟输入
    inputs_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"\ninputs_tokens shape: {inputs_tokens.shape}")
    
    # 模拟 gate 函数输出
    # 为简化演示，手动构造 topk_idx 和 topk_weight
    topk_idx = torch.tensor([
        [[3, 1],   # token 0 选择 expert 3 和 1
         [1, 2],   # token 1 选择 expert 1 和 2
         [3, 0]],  # token 2 选择 expert 3 和 0
        [[0, 2],   # token 3 选择 expert 0 和 2
         [2, 1],   # token 4 选择 expert 2 和 1
         [3, 0]]   # token 5 选择 expert 3 和 0
    ])
    
    topk_weight = torch.rand(batch_size, seq_len, num_experts_per_tok)
    # 归一化权重
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    
    print(f"\ntopk_idx shape: {topk_idx.shape}")
    print(f"topk_weight shape: {topk_weight.shape}")
    
    print("\nGate 输出详情:")
    for b in range(batch_size):
        for s in range(seq_len):
            token_id = b * seq_len + s
            experts = topk_idx[b, s].tolist()
            weights = topk_weight[b, s].tolist()
            print(f"  Token {token_id}: experts={experts}, weights={[f'{w:.3f}' for w in weights]}")
    
    # ========== 开始处理流程 ==========
    
    # Step 1: 展平
    print("\n" + "="*60)
    print("Step 1: 展平 expert indices 和 weights")
    print("="*60)
    
    flat_expert_indices = topk_idx.view(-1)
    flat_experts_weight = topk_weight.view(-1, 1)
    
    print(f"flat_expert_indices: {flat_expert_indices.tolist()}")
    print(f"  shape: {flat_expert_indices.shape}")
    print(f"flat_experts_weight shape: {flat_experts_weight.shape}")
    
    # Step 2: 排序
    print("\n" + "="*60)
    print("Step 2: 按 expert_id 排序")
    print("="*60)
    
    idxs = flat_expert_indices.argsort()
    print(f"排序索引 idxs: {idxs.tolist()}")
    
    # 显示排序结果
    print("\n排序后的 expert_id 和对应的原始位置:")
    for i, idx in enumerate(idxs.tolist()):
        expert_id = flat_expert_indices[idx].item()
        print(f"  Position {i}: expert_id={expert_id} (来自原始位置 {idx})")
    
    # Step 3: 统计每个专家的 token 数量
    print("\n" + "="*60)
    print("Step 3: 统计每个专家的 token 数量")
    print("="*60)
    
    tokens_per_expert = flat_expert_indices.bincount(minlength=num_experts)
    print(f"tokens_per_expert: {tokens_per_expert.tolist()}")
    
    for expert_id in range(num_experts):
        count = tokens_per_expert[expert_id].item()
        print(f"  Expert {expert_id}: {count} tokens")
    
    # Step 4: 计算原始 token 索引
    print("\n" + "="*60)
    print("Step 4: 恢复原始 token 索引")
    print("="*60)
    
    token_idxs = idxs // num_experts_per_tok
    print(f"token_idxs: {token_idxs.tolist()}")
    
    print("\n排序后每个位置对应的原始 token:")
    for i, (idx, token_id) in enumerate(zip(idxs.tolist(), token_idxs.tolist())):
        expert_id = flat_expert_indices[idx].item()
        print(f"  Position {i}: token {token_id} -> expert {expert_id}")
    
    # Step 5: 重排 hidden_states 和 weights
    print("\n" + "="*60)
    print("Step 5: 重排 hidden_states 和 weights")
    print("="*60)
    
    hidden_states = inputs_tokens.view(batch_size * seq_len, -1)
    sorted_hidden_states = hidden_states[token_idxs]
    sorted_weights = flat_experts_weight[idxs]
    
    print(f"hidden_states shape: {hidden_states.shape}")
    print(f"sorted_hidden_states shape: {sorted_hidden_states.shape}")
    print(f"sorted_weights shape: {sorted_weights.shape}")
    
    # Step 6: 构建每个 expert 的 tokens 和 weights
    print("\n" + "="*60)
    print("Step 6: 构建每个 expert 的 tokens 和 weights")
    print("="*60)
    
    expert_tokens_map = {}
    expert_weights_map = {}
    
    start_idx = 0
    for expert_id in range(num_experts):
        if tokens_per_expert[expert_id] == 0:
            print(f"\nExpert {expert_id}: 没有分配 tokens")
            continue
        
        end_idx = start_idx + tokens_per_expert[expert_id].item()
        
        expert_tokens = sorted_hidden_states[start_idx:end_idx]
        expert_weights = sorted_weights[start_idx:end_idx]
        
        expert_tokens_map[expert_id] = expert_tokens
        expert_weights_map[expert_id] = expert_weights
        
        print(f"\nExpert {expert_id}:")
        print(f"  Tokens shape: {expert_tokens.shape}")
        print(f"  Weights shape: {expert_weights.shape}")
        print(f"  处理的原始 token IDs: {token_idxs[start_idx:end_idx].tolist()}")
        print(f"  权重值: {[f'{w.item():.3f}' for w in expert_weights]}")
        
        start_idx = end_idx
    
    # 验证
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    
    total_processed = sum(tokens.shape[0] for tokens in expert_tokens_map.values())
    expected_total = batch_size * seq_len * num_experts_per_tok
    
    print(f"总共处理的 token-expert 对: {total_processed}")
    print(f"预期的 token-expert 对: {expected_total}")
    print(f"验证通过: {total_processed == expected_total}")
    
    # 可视化分配
    print("\n" + "="*60)
    print("Token -> Expert 分配可视化")
    print("="*60)
    
    print("\n原始分配 (按 token 组织):")
    for b in range(batch_size):
        for s in range(seq_len):
            token_id = b * seq_len + s
            experts = topk_idx[b, s].tolist()
            print(f"  Token {token_id} -> Experts {experts}")
    
    print("\n排序后分配 (按 expert 组织):")
    for expert_id in sorted(expert_tokens_map.keys()):
        start_idx = sum(tokens_per_expert[:expert_id]).item()
        end_idx = start_idx + tokens_per_expert[expert_id].item()
        token_list = token_idxs[start_idx:end_idx].tolist()
        print(f"  Expert {expert_id} <- Tokens {token_list}")
    
    return expert_tokens_map, expert_weights_map


def demo_einsum_computation():
    """演示如何使用分配好的 tokens 进行 einsum 计算"""
    
    print("\n\n" + "="*60)
    print("Einsum 批量计算演示")
    print("="*60)
    
    # 获取分配结果
    expert_tokens_map, expert_weights_map = demo_expert_token_distribution()
    
    # 模拟 expert 权重
    num_experts = 4
    hidden_dim = 8
    intermediate_dim = 16
    
    print("\n模拟 expert 权重:")
    w1_weights = torch.randn(num_experts, intermediate_dim, hidden_dim)
    w2_weights = torch.randn(num_experts, hidden_dim, intermediate_dim)
    w3_weights = torch.randn(num_experts, intermediate_dim, hidden_dim)
    
    print(f"  w1_weights: {w1_weights.shape}")
    print(f"  w2_weights: {w2_weights.shape}")
    print(f"  w3_weights: {w3_weights.shape}")
    
    # 为每个 expert 计算（单独处理）
    print("\n方法 1: 逐个 expert 计算 (传统方法)")
    results_individual = {}
    
    for expert_id in expert_tokens_map.keys():
        tokens = expert_tokens_map[expert_id]  # [num_tokens, hidden_dim]
        weights = expert_weights_map[expert_id]  # [num_tokens, 1]
        
        # MLP 计算: w1 * x, silu, w3 * x, multiply, w2 * result
        w1_out = torch.matmul(tokens, w1_weights[expert_id].T)  # [num_tokens, intermediate_dim]
        w1_out = torch.nn.functional.silu(w1_out)
        w3_out = torch.matmul(tokens, w3_weights[expert_id].T)
        intermediate = w1_out * w3_out
        output = torch.matmul(intermediate, w2_weights[expert_id].T)  # [num_tokens, hidden_dim]
        
        # 应用路由权重
        output = output * weights
        
        results_individual[expert_id] = output
        print(f"  Expert {expert_id}: input {tokens.shape} -> output {output.shape}")
    
    print("\n方法 2: Einsum 批量计算 (优化方法)")
    print("  (参考 test_einsum_experts.py 中的实现)")
    print("  - 优点: 可以批量处理多个 expert")
    print("  - 适合: expert 权重已加载到连续内存")
    print("  - 性能: 显著快于逐个处理")


if __name__ == "__main__":
    # 运行演示
    demo_einsum_computation()
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)
    print("\n关键要点:")
    print("1. argsort() 将相同 expert 的 token 聚集在一起")
    print("2. bincount() 统计每个 expert 的 token 数量")
    print("3. token_idxs 恢复原始 token 索引")
    print("4. 重排 hidden_states 和 weights 以匹配排序顺序")
    print("5. 按 expert 分组构建 expert_tokens_map")
    print("\n这样就可以批量处理同一 expert 的所有 tokens，提高效率！")

