# Dtype 问题分析

## 🔍 问题描述

Expert 权重从共享内存加载后显示为 `torch.float32`，但 config 中配置的是 `bfloat16`。

## ✅ 已确认正确的部分

1. **tensor_index_resize.json 格式正确**
   - dtype 字段格式：`"torch.bfloat16"` ✓
   - C++ `stringToScalarType` 支持此格式 ✓

2. **C++ 代码逻辑正确**
   - `RestoreTensorsFromSharedMemoryNames` 正确读取 dtype ✓
   - `torch::from_blob` 使用正确的 dtype ✓

## ⚠️ 可能的问题点

### 1. 模型创建时未使用正确的 dtype

**位置**: `DeepseekModule.create_empty_model()`

```python
def create_empty_model(self, config: AutoConfig):
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = DeepseekForCausalLM(config)  # ⚠️ 可能未使用 config.torch_dtype
        return model
```

**问题**: `init_empty_weights()` 创建的模型可能使用默认 dtype (float32)，而不是 config.torch_dtype。

**解决方案**: 需要在创建模型时显式指定 dtype：

```python
def create_empty_model(self, config: AutoConfig):
    config._attn_implementation = "sdpa"
    
    # 获取目标 dtype
    target_dtype = config.torch_dtype
    if isinstance(target_dtype, str):
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        target_dtype = dtype_map.get(target_dtype, torch.float32)
    
    with init_empty_weights():
        model = DeepseekForCausalLM(config)
        # 将模型转换为目标 dtype
        model = model.to(dtype=target_dtype)
        return model
```

### 2. set_module_tensor_to_device 可能覆盖 dtype

**位置**: `restore2model()` 和 `restore_hm_state_dict2model()`

**问题**: `set_module_tensor_to_device` 可能根据目标模块的 dtype 自动转换，而不是保持 tensor 的原始 dtype。

**检查方法**: 在设置前打印 tensor 的 dtype：

```python
def restore2model(self, model_state_dict, model):
    with torch.no_grad():
        for name, param in model_state_dict.items():
            print(f"Setting {name}: tensor dtype={param.dtype}, model dtype={getattr(model, name.split('.')[0], None)}")
            set_module_tensor_to_device(model, name, param.device, param, clear_cache=False)
```

### 3. 共享内存中的数据格式问题

**问题**: 即使 tensor_index_resize.json 标记为 bfloat16，共享内存中的实际数据可能是 float32。

**检查方法**: 在 C++ 代码中添加调试输出：

```cpp
at::ScalarType dtype = stringToScalarType(dtype_str);
std::cerr << "Tensor " << name << ": dtype_str=" << dtype_str 
          << ", ScalarType=" << dtype << std::endl;
```

## 🔧 调试步骤

### Step 1: 检查从共享内存加载的 tensor dtype

在 `cuda_memory_view.py` 中添加：

```python
self.hm_state_dict = restore_tensors_from_shared_memory_names(...)

# 检查 dtype
for name, tensor in list(self.hm_state_dict.items())[:5]:
    if 'expert' in name:
        print(f"{name}: dtype={tensor.dtype}, shape={tensor.shape}")
```

### Step 2: 检查模型创建时的 dtype

在 `mlpmodule.py` 中添加：

```python
def create_empty_model(self):
    model = self.model_class.create_empty_model(self.config)
    
    # 检查第一个 expert 的 dtype
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer0 = model.model.layers[0]
        if hasattr(layer0, 'mlp') and hasattr(layer0.mlp, 'experts'):
            expert0 = layer0.mlp.experts[0]
            if hasattr(expert0, 'gate_proj'):
                print(f"Model expert dtype: {expert0.gate_proj.weight.dtype}")
    
    return model
```

### Step 3: 检查 restore 后的 dtype

在 `restore2model` 中添加：

```python
def restore2model(self, model_state_dict, model):
    with torch.no_grad():
        for name, param in model_state_dict.items():
            if 'expert' in name and 'weight' in name:
                print(f"Before restore: {name} dtype={param.dtype}")
                set_module_tensor_to_device(model, name, param.device, param, clear_cache=False)
                
                # 检查 restore 后
                module_param = get_module_tensor(model, name)
                if module_param is not None:
                    print(f"After restore: {name} dtype={module_param.dtype}")
```

## 📋 检查清单

- [ ] tensor_index_resize.json 中 dtype 格式正确 ✓ (已确认)
- [ ] C++ 代码正确读取 dtype ✓ (已确认)
- [ ] 从共享内存加载的 tensor dtype 正确？
- [ ] 模型创建时使用正确的 dtype？
- [ ] restore 后模型中的 dtype 正确？

## 🎯 最可能的原因

根据诊断结果，**最可能的问题是模型创建时未使用 config.torch_dtype**。

`init_empty_weights()` 创建的模型默认使用 float32，即使 config 中指定了 bfloat16。

## 💡 解决方案

修改 `DeepseekModule.create_empty_model()`:

```python
def create_empty_model(self, config: AutoConfig):
    config._attn_implementation = "sdpa"
    
    # 获取目标 dtype
    target_dtype = getattr(config, 'torch_dtype', None)
    if target_dtype is None:
        target_dtype = torch.float32
    elif isinstance(target_dtype, str):
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        target_dtype = dtype_map.get(target_dtype, torch.float32)
    
    with init_empty_weights():
        model = DeepseekForCausalLM(config)
        # 将模型转换为目标 dtype
        model = model.to(dtype=target_dtype)
        return model
```

这样创建的模型就是 bfloat16，后续从共享内存加载的 bfloat16 tensor 就能正确匹配。

