# NaN 和 Inf 值调试指南

## 结论：NaN 和 Inf 值**不正常**

在模型推理过程中，出现 NaN（Not a Number）和 Inf（Infinity）值通常表示：

1. **数值不稳定**：计算过程中出现了数值溢出或下溢
2. **模型权重问题**：某些权重未正确初始化或加载
3. **输入数据问题**：输入包含异常值
4. **计算错误**：某些操作（如除零、log(0) 等）产生了无效值

## 常见原因分析

### 1. 模型权重问题

**症状**：
- 某些层的权重包含 NaN/Inf
- 模型加载不完整

**检查方法**：
```python
# 检查模型权重
for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"ERROR: {name} contains NaN/Inf")
```

### 2. 输入数据问题

**症状**：
- 输入 embedding 包含异常值
- Token IDs 超出范围

**检查方法**：
```python
# 检查输入
if torch.isnan(inputs).any() or torch.isinf(inputs).any():
    print("ERROR: Input contains NaN/Inf")
```

### 3. 数值溢出

**症状**：
- Logits 值过大（> 1000）
- Softmax 计算溢出

**检查方法**：
```python
# 检查 logits 范围
if logits.abs().max() > 1000:
    print("WARNING: Logits too large, may cause overflow")
    logits = torch.clamp(logits, min=-100, max=100)
```

### 4. 除零操作

**症状**：
- LayerNorm 中分母为 0
- 归一化操作失败

**检查方法**：
```python
# 检查 LayerNorm 输入
if (hidden_states.std() == 0).any():
    print("WARNING: Zero variance in LayerNorm input")
```

## 调试步骤

### 步骤 1: 定位问题源头

在代码中添加检查点，逐层追踪：

```python
# 在每个 layer 后检查
if check_nan_inf(ghidden_states):
    logger.error(f"ERROR: NaN/Inf detected at layer {layer_idx}")
    # 检查该层的输入
    logger.error(f"  Input stats: min={inputs.min()}, max={inputs.max()}, mean={inputs.mean()}")
    raise ValueError("NaN/Inf detected")
```

### 步骤 2: 检查模型权重

```python
def check_model_weights(model):
    """检查模型权重是否包含 NaN/Inf"""
    issues = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"{name}: NaN")
        if torch.isinf(param).any():
            issues.append(f"{name}: Inf")
    return issues
```

### 步骤 3: 检查计算过程

在关键计算点添加检查：

```python
# 在 attention 计算后
attention_output = self_attn_func(...)
if check_nan_inf(attention_output):
    logger.error("NaN/Inf in attention output")

# 在 MLP 计算后
mlp_output = mlp_func(...)
if check_nan_inf(mlp_output):
    logger.error("NaN/Inf in MLP output")
```

### 步骤 4: 使用 CUDA 调试工具

```bash
# 启用同步执行，获得准确的错误位置
CUDA_LAUNCH_BLOCKING=1 python your_script.py

# 启用设备端断言
TORCH_USE_CUDA_DSA=1 python your_script.py
```

## 修复建议

### 1. 临时修复（不推荐，仅用于调试）

```python
# 替换 NaN/Inf 为 0
tensor = torch.where(torch.isnan(tensor), torch.tensor(0.0), tensor)
tensor = torch.where(torch.isinf(tensor), torch.tensor(0.0), tensor)
```

### 2. 裁剪极端值

```python
# 裁剪 logits 避免溢出
logits = torch.clamp(logits, min=-100, max=100)
```

### 3. 添加数值稳定性检查

```python
# 在 LayerNorm 中添加 epsilon
output = input / (std + 1e-8)
```

### 4. 检查模型加载

确保所有权重都正确加载：

```python
# 检查模型状态
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if missing_keys:
    logger.warning(f"Missing keys: {missing_keys}")
if unexpected_keys:
    logger.warning(f"Unexpected keys: {unexpected_keys}")
```

## 当前代码中的检查点

1. **Layer 输出检查**（`lmp.py:211`）：
   - 在每个 layer 后检查 `ghidden_states`

2. **Logits 检查**（`lmp.py:227-254`）：
   - 检查 logits 是否包含 NaN/Inf
   - 裁剪极端值

3. **概率检查**（`helper.py:105-151`）：
   - 在 softmax 后检查概率
   - 修复无效概率

## 建议

1. **不要忽略 NaN/Inf**：它们会导致错误的推理结果
2. **找到根本原因**：临时修复只是权宜之计
3. **添加预防性检查**：在关键计算点添加检查
4. **使用混合精度时注意**：bfloat16 可能更容易出现数值问题

## 常见场景

### 场景 1: 模型权重未加载

**症状**：所有输出都是 NaN

**解决**：检查模型加载代码，确保权重正确加载

### 场景 2: 输入异常

**症状**：特定输入导致 NaN

**解决**：检查输入数据，确保 token IDs 在有效范围内

### 场景 3: 数值溢出

**症状**：Logits 值过大导致 softmax 溢出

**解决**：裁剪 logits 或使用数值稳定的 softmax 实现

### 场景 4: 空模型（meta tensor）

**症状**：使用 `init_empty_weights()` 创建的模型，权重在 meta 设备上

**解决**：确保在计算前将权重加载到正确的设备

