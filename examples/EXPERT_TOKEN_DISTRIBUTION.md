# MoE Expert Token Distribution 详解

## 📊 完整流程图

```
输入 inputs_tokens [batch_size, seq_len, hidden_dim]
           ↓
    Gate Function
           ↓
topk_idx [batch_size, seq_len, num_experts_per_tok]
topk_weight [batch_size, seq_len, num_experts_per_tok]
           ↓
    ┌──────┴──────┐
    │   Step 1:   │  展平
    │   Flatten   │
    └──────┬──────┘
           ↓
flat_expert_indices [total_selections]
flat_experts_weight [total_selections, 1]
           ↓
    ┌──────┴──────┐
    │   Step 2:   │  按 expert_id 排序
    │   argsort   │
    └──────┬──────┘
           ↓
idxs [total_selections]  (排序后的索引)
           ↓
    ┌──────┴──────┐
    │   Step 3:   │  统计每个 expert 的 token 数
    │  bincount   │
    └──────┬──────┘
           ↓
tokens_per_expert [num_experts]
           ↓
    ┌──────┴──────┐
    │   Step 4:   │  恢复原始 token 索引
    │  token_idxs │
    └──────┬──────┘
           ↓
token_idxs [total_selections]
           ↓
    ┌──────┴──────┐
    │   Step 5:   │  重排 hidden_states 和 weights
    │  Reorder    │
    └──────┬──────┘
           ↓
sorted_hidden_states [total_selections, hidden_dim]
sorted_weights [total_selections, 1]
           ↓
    ┌──────┴──────┐
    │   Step 6:   │  按 expert 分组
    │   Group     │
    └──────┬──────┘
           ↓
expert_tokens_map: {expert_id: tokens}
expert_weights_map: {expert_id: weights}
```

## 🎯 示例说明

### 假设配置
```python
batch_size = 2
seq_len = 3
num_experts_per_tok = 2
num_experts = 4
total_tokens = 6  (2 * 3)
total_selections = 12  (6 * 2)
```

### Step 1: Gate 输出

```
Token 0 -> [Expert 3 (w:0.6), Expert 1 (w:0.4)]
Token 1 -> [Expert 1 (w:0.5), Expert 2 (w:0.5)]
Token 2 -> [Expert 3 (w:0.7), Expert 0 (w:0.3)]
Token 3 -> [Expert 0 (w:0.6), Expert 2 (w:0.4)]
Token 4 -> [Expert 2 (w:0.8), Expert 1 (w:0.2)]
Token 5 -> [Expert 3 (w:0.5), Expert 0 (w:0.5)]

topk_idx = [[3,1], [1,2], [3,0], [0,2], [2,1], [3,0]]
```

### Step 2: 展平

```python
flat_expert_indices = [3, 1, 1, 2, 3, 0, 0, 2, 2, 1, 3, 0]
                       ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
Position:              0  1  2  3  4  5  6  7  8  9 10 11
Token:                 0  0  1  1  2  2  3  3  4  4  5  5
```

### Step 3: 排序（argsort）

```python
# 按 expert_id 排序
idxs = [5, 6, 11, 1, 2, 9, 3, 7, 8, 0, 4, 10]

排序后的 expert_id:
Position: 0  1  2   3  4  5   6  7  8   9 10 11
Expert:   0  0  0 | 1  1  1 | 2  2  2 | 3  3  3
```

### Step 4: 统计（bincount）

```python
tokens_per_expert = [3, 3, 3, 3]
# Expert 0: 3 个 token-expert 对
# Expert 1: 3 个 token-expert 对
# Expert 2: 3 个 token-expert 对
# Expert 3: 3 个 token-expert 对
```

### Step 5: 恢复 token 索引

```python
token_idxs = idxs // num_experts_per_tok
           = [5, 6, 11, 1, 2, 9, 3, 7, 8, 0, 4, 10] // 2
           = [2, 3, 5,  0, 1, 4, 1, 3, 4, 0, 2, 5]

解释：
Position 0: idxs[0]=5  -> token_id = 5//2 = 2 (Expert 0 处理 Token 2)
Position 1: idxs[1]=6  -> token_id = 6//2 = 3 (Expert 0 处理 Token 3)
Position 2: idxs[2]=11 -> token_id = 11//2 = 5 (Expert 0 处理 Token 5)
...
```

### Step 6: 构建 expert_tokens_map

```python
expert_tokens_map = {
    0: sorted_hidden_states[0:3],   # Tokens [2, 3, 5]
    1: sorted_hidden_states[3:6],   # Tokens [0, 1, 4]
    2: sorted_hidden_states[6:9],   # Tokens [1, 3, 4]
    3: sorted_hidden_states[9:12],  # Tokens [0, 2, 5]
}

expert_weights_map = {
    0: sorted_weights[0:3],    # [0.3, 0.6, 0.5]
    1: sorted_weights[3:6],    # [0.4, 0.5, 0.2]
    2: sorted_weights[6:9],    # [0.5, 0.4, 0.8]
    3: sorted_weights[9:12],   # [0.6, 0.7, 0.5]
}
```

## 🔍 关键代码解析

### 1. 为什么要 argsort？

```python
idxs = flat_expert_indices.argsort()
```

**目的：** 将相同 expert_id 的 token 聚集在一起

**效果：**
- 排序前：token 按原始顺序排列
- 排序后：token 按 expert_id 分组

**好处：**
- 可以批量处理同一 expert 的所有 tokens
- 减少 expert 切换开销
- 适合 einsum 批量计算

### 2. 为什么要 bincount？

```python
tokens_per_expert = flat_expert_indices.bincount()
```

**目的：** 统计每个 expert 被分配了多少个 token

**输出：** 长度为 `num_experts` 的张量，每个元素表示对应 expert 的 token 数量

**用途：** 确定每个 expert 在 sorted_hidden_states 中的范围

### 3. 为什么要 `// num_experts_per_tok`？

```python
token_idxs = idxs // num_experts_per_tok
```

**目的：** 从展平的位置恢复原始 token 索引

**原理：**
```
展平时：flat_position = token_id * num_experts_per_tok + expert_slot
恢复时：token_id = flat_position // num_experts_per_tok
```

**示例：**
```
Token 0 的第 1 个 expert: position 0 -> 0 // 2 = 0
Token 0 的第 2 个 expert: position 1 -> 1 // 2 = 0
Token 1 的第 1 个 expert: position 2 -> 2 // 2 = 1
Token 1 的第 2 个 expert: position 3 -> 3 // 2 = 1
```

## 💻 代码实现

完整实现参见：
- `lmp/src/lmp/lmp.py` (lines 67-110)
- `lmp/examples/expert_token_distribution_demo.py`

## 🚀 性能优化

### 批量计算的优势

传统方法（逐个处理）：
```python
for token in all_tokens:
    expert_id = select_expert(token)
    result = experts[expert_id](token)
```

优化方法（批量处理）：
```python
for expert_id, tokens in expert_tokens_map.items():
    results[expert_id] = experts[expert_id](tokens)  # 批量计算
```

### Einsum 批量计算

参考 `test_einsum_experts.py`:
```python
# 将多个 expert 的权重 stack 成 [E, I, H]
w1_weights = torch.stack([expert.w1.weight for expert in experts])

# 批量计算所有 expert
results = torch.einsum('eth,eih->eti', stacked_inputs, w1_weights)
```

**性能提升：**
- 减少 Python 循环开销
- 更好的缓存利用
- 可以利用 SIMD 指令
- NUMA 优化更有效

## 📚 相关资源

- `test_einsum_experts.py`: Einsum 批量计算实现
- `test_numa_impact.py`: NUMA 对性能的影响测试
- `checkpoint.cpp`: 从共享内存加载 expert 权重
- `cuda_memory_view.py`: GPU 内存管理

## 🎓 总结

这套流程的核心思想是：

1. **分组（argsort）**：将 tokens 按 expert 分组
2. **统计（bincount）**：确定每组的大小
3. **映射（token_idxs）**：保持与原始 tokens 的对应关系
4. **批量计算**：每个 expert 一次性处理所有分配给它的 tokens

这样可以显著提升 MoE 模型的推理效率！

