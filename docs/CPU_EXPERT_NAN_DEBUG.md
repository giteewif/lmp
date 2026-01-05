# CPU 专家计算后 expert_cache 出现 NaN 值调试指南

## 问题描述

在 CPU 执行专家计算后，`expert_cache` 出现了 NaN 值。错误发生在：
```python
if check_nan_inf(expert_cache):
    raise ValueError("after cetm expert_cache contain NaN")
```

## 可能的原因分析

### 1. 从共享内存加载的权重包含 NaN/Inf

**位置**：`group_experts_tensor()` 函数
- `group_w1`, `group_w2`, `group_w3` 从共享内存加载
- 如果共享内存中的数据本身包含 NaN/Inf，会导致后续计算产生 NaN

**检查点**：已在 `einsum_with_group_tensors` 中添加检查

### 2. 输入数据包含 NaN/Inf

**位置**：`flat_hidden_states` 传入 CPU 计算
- 如果 `flat_hidden_states` 在传入前就包含 NaN/Inf
- 会导致所有后续计算都产生 NaN

**检查点**：已在 `layer_moe_generate` 中添加检查

### 3. Einsum 计算溢出

**位置**：三个 einsum 操作
- `w1_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w1)`
- `w3_out = torch.einsum('eth,eih->eti', stacked_inputs, group_w3)`
- `outputs_result = torch.einsum('eti,ehi->eth', intermediate, group_w2)`

**可能原因**：
- 输入值过大导致计算溢出
- 矩阵维度不匹配
- 数据类型不匹配（如 bfloat16 精度问题）

**检查点**：已在每个 einsum 前后添加检查

### 4. 激活函数产生 NaN/Inf

**位置**：`w1_out = act_fn(w1_out)`
- 某些激活函数（如 GELU、Swish）在输入值过大时可能产生 Inf
- 如果输入包含 NaN，激活函数输出也会是 NaN

**检查点**：已在激活函数前后添加检查

### 5. 逐元素相乘产生 NaN/Inf

**位置**：`intermediate = w1_out * w3_out`
- 如果 `w1_out` 或 `w3_out` 包含 NaN/Inf
- 相乘结果也会是 NaN/Inf

**检查点**：已在相乘前后添加检查

### 6. 设备间数据传输问题

**位置**：CPU 到 GPU 的数据传输
- `expert_out = expert_out.to(final_hidden_states.device, non_blocking=True)`
- 如果使用 `non_blocking=True`，可能在数据未完全传输时就使用
- 需要确保同步完成

**检查点**：已在传输前后添加检查

### 7. Scatter 操作问题

**位置**：`final_hidden_states.scatter_reduce_()`
- 如果 `expert_out` 或 `token_ids` 包含无效值
- scatter 操作可能产生 NaN

**检查点**：已在 scatter 前后添加检查

### 8. 多线程竞争

**位置**：CETM 多线程执行
- 如果多个线程同时访问共享资源
- 可能导致数据竞争和 NaN

**检查点**：当前使用单线程（num_workers=1），应该不是问题

## 调试步骤

### 步骤 1: 运行程序并查看日志

运行程序后，查看详细的错误日志，定位 NaN 首次出现的位置：

```bash
python generate.py 2>&1 | tee debug.log
```

### 步骤 2: 检查各个检查点的输出

根据日志中的错误信息，确定 NaN 首次出现的位置：

1. **如果出现在 "group_w1/2/3 from shared memory"**：
   - 问题在权重加载阶段
   - 检查共享内存中的数据是否正确
   - 检查 `restore_experts_groups_from_shared_memory` 函数

2. **如果出现在 "stacked_inputs after copying"**：
   - 问题在输入数据复制阶段
   - 检查 `flat_hidden_states` 的来源
   - 检查 GPU 到 CPU 的数据传输

3. **如果出现在 "w1_out after einsum"**：
   - 问题在第一个 einsum 计算
   - 检查 `stacked_inputs` 和 `group_w1` 的值
   - 检查矩阵维度是否匹配

4. **如果出现在 "w1_out after activation"**：
   - 问题在激活函数
   - 检查激活函数的输入值范围
   - 考虑使用数值稳定的激活函数实现

5. **如果出现在 "intermediate after multiplication"**：
   - 问题在逐元素相乘
   - 检查 `w1_out` 和 `w3_out` 的值

6. **如果出现在 "outputs_result after final einsum"**：
   - 问题在最后一个 einsum
   - 检查 `intermediate` 和 `group_w2` 的值

7. **如果出现在 "expert_out after GPU transfer"**：
   - 问题在设备间传输
   - 确保使用 `torch.cuda.synchronize()` 同步

8. **如果出现在 "final_hidden_states after scatter"**：
   - 问题在 scatter 操作
   - 检查 `token_ids` 是否有效
   - 检查 `expert_out` 的值

### 步骤 3: 添加同步点

在关键操作后添加同步，确保数据完全传输：

```python
# 在 CPU 到 GPU 传输后
expert_out = expert_out.to(final_hidden_states.device, non_blocking=True)
torch.cuda.synchronize()  # 确保传输完成
```

### 步骤 4: 检查权重加载

验证从共享内存加载的权重是否正确：

```python
# 在 group_experts_tensor 后
for name, tensor in group_dict.items():
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.error(f"{name} contains NaN/Inf!")
        # 检查共享内存中的数据
```

## 常见修复方法

### 1. 修复权重加载

如果权重包含 NaN/Inf，检查：
- 模型文件是否正确
- 共享内存映射是否正确
- 数据类型转换是否正确

### 2. 修复数值溢出

如果计算溢出，可以：
- 裁剪输入值范围
- 使用更高精度的数据类型（如 float32 而不是 bfloat16）
- 使用数值稳定的计算方式

### 3. 修复激活函数

如果激活函数产生 NaN/Inf：
- 检查输入值范围
- 使用数值稳定的激活函数实现
- 裁剪激活函数输入

### 4. 修复设备同步

确保所有异步操作完成：
```python
torch.cuda.synchronize()
```

## 当前添加的调试代码

已在以下位置添加了详细的调试检查：

1. **权重加载后**：检查 `group_w1`, `group_w2`, `group_w3`
2. **输入数据复制后**：检查 `stacked_inputs`
3. **每个 einsum 前后**：检查计算结果
4. **激活函数后**：检查 `w1_out`
5. **逐元素相乘后**：检查 `intermediate`
6. **GPU 传输后**：检查 `expert_out`
7. **Scatter 后**：检查 `final_hidden_states`

运行程序后，根据日志输出可以精确定位 NaN 首次出现的位置。

