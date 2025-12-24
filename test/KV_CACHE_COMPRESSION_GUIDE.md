# KV Cache 压缩方案指南

## 概述

在LLM推理中，KV cache会随着序列长度增长而线性增长，占用大量内存。本指南提供了多种KV cache压缩方案。

## 压缩方法对比

| 方法 | 压缩比 | 精度损失 | 计算开销 | 适用场景 |
|------|--------|----------|----------|----------|
| **量化 (INT8)** | 2x | 低 | 低 | 通用，平衡精度和速度 |
| **量化 (INT4)** | 4x | 中 | 低 | 内存受限场景 |
| **稀疏化 (50%)** | 2x | 中 | 中 | 可以容忍精度损失 |
| **滑动窗口** | 可变 | 低 | 极低 | 长序列，只关注最近tokens |
| **低秩分解** | 2-4x | 中-高 | 高 | 离线压缩，不推荐实时使用 |
| **混合方案** | 4-8x | 中 | 中 | 极致压缩需求 |

## 使用方法

### 1. 量化压缩（推荐）

```python
from kv_cache_compression import QuantizedKVCache

# 初始化量化器
quantizer = QuantizedKVCache(bits=8, symmetric=True)

# 压缩KV cache
key_compressed, scale_k, zp_k = quantizer.quantize(key)
value_compressed, scale_v, zp_v = quantizer.quantize(value)

# 使用时反量化
key_decompressed = quantizer.dequantize(key_compressed, scale_k, zp_k)
value_decompressed = quantizer.dequantize(value_compressed, scale_v, zp_v)
```

**优点：**
- 压缩比固定（2x for INT8, 4x for INT4）
- 精度损失小
- 实现简单

**缺点：**
- 需要存储scale和zero_point
- 反量化有少量计算开销

### 2. 稀疏化压缩

```python
from kv_cache_compression import SparseKVCache

# 初始化稀疏化器
sparsifier = SparseKVCache(sparsity_ratio=0.5, method="magnitude")

# 压缩（需要attention scores用于更好的选择）
compressed_key, compressed_value, mask = sparsifier.compress(key, value, attention_scores)

# 使用时需要根据mask重建
```

**优点：**
- 可以保留最重要的KV对
- 压缩比可调

**缺点：**
- 需要额外的mask存储
- 重建时需要处理稀疏结构

### 3. 滑动窗口压缩（最简单）

```python
from kv_cache_compression import SlidingWindowKVCache

# 初始化滑动窗口
sliding = SlidingWindowKVCache(window_size=256)

# 压缩（只保留最近256个tokens）
compressed_key, compressed_value = sliding.compress(key, value)
```

**优点：**
- 实现最简单
- 无精度损失（只是丢弃旧tokens）
- 计算开销极低

**缺点：**
- 丢失历史信息
- 不适合需要长上下文的任务

### 4. 混合压缩（推荐用于极致压缩）

```python
from kv_cache_compression import HybridKVCacheCompressor

# 初始化混合压缩器
compressor = HybridKVCacheCompressor(
    quantize_bits=8,           # INT8量化
    sparsity_ratio=0.3,        # 30%稀疏化
    use_sliding_window=True,   # 使用滑动窗口
    window_size=256            # 窗口大小256
)

# 压缩
compressed_data = compressor.compress(key, value)

# 反压缩
key_decompressed, value_decompressed = compressor.decompress(compressed_data)
```

## 集成到现有代码

### 修改 test_decode_thread.py

```python
from kv_cache_compression import HybridKVCacheCompressor

# 在初始化时创建压缩器
kv_compressor = HybridKVCacheCompressor(
    quantize_bits=8,
    use_sliding_window=True,
    window_size=256
)

# 在存储KV cache时压缩
def store_kv_cache(key, value):
    compressed = kv_compressor.compress(key, value)
    # 存储压缩后的数据
    return compressed

# 在使用KV cache时反压缩
def load_kv_cache(compressed_data):
    key, value = kv_compressor.decompress(compressed_data)
    return key, value
```

## 性能建议

### 场景1: 内存充足，追求精度
- **方案**: INT8量化
- **压缩比**: 2x
- **精度损失**: <1%

### 场景2: 内存受限，需要长上下文
- **方案**: INT8量化 + 滑动窗口(512)
- **压缩比**: 2-4x（取决于窗口大小）
- **精度损失**: <1%

### 场景3: 极致压缩需求
- **方案**: INT4量化 + 稀疏化(50%) + 滑动窗口(256)
- **压缩比**: 8-16x
- **精度损失**: 2-5%

## 内存节省计算

假设：
- Batch size: 512
- KV heads: 8
- Sequence length: 512
- Head dim: 128
- Dtype: bfloat16 (2 bytes)

**原始KV cache大小:**
```
Key: 512 × 8 × 512 × 128 × 2 = 536 MB
Value: 512 × 8 × 512 × 128 × 2 = 536 MB
总计: 1072 MB
```

**使用INT8量化后:**
```
Key: 512 × 8 × 512 × 128 × 1 = 268 MB (+ scale: ~0.001 MB)
Value: 512 × 8 × 512 × 128 × 1 = 268 MB (+ scale: ~0.001 MB)
总计: ~536 MB (压缩比: 2x)
```

**使用INT8 + 滑动窗口256后:**
```
Key: 512 × 8 × 256 × 128 × 1 = 134 MB
Value: 512 × 8 × 256 × 128 × 1 = 134 MB
总计: ~268 MB (压缩比: 4x)
```

## 注意事项

1. **精度vs压缩比权衡**: 压缩比越高，精度损失越大
2. **计算开销**: 量化/反量化有少量开销，但通常可以忽略
3. **滑动窗口**: 适合decoding阶段，不适合prefill阶段
4. **稀疏化**: 需要额外的mask存储，可能抵消部分压缩收益
5. **低秩分解**: 计算开销大，不推荐实时使用

## 实验建议

1. 先用INT8量化测试，这是最安全的方案
2. 如果内存仍然不足，添加滑动窗口
3. 如果还需要压缩，考虑INT4或稀疏化
4. 监控精度指标（perplexity, accuracy等）

## 参考实现

参考 `kv_cache_compression.py` 中的完整实现。


