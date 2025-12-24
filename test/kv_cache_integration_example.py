"""
KV Cache压缩集成示例
展示如何在实际的attention代码中使用KV cache压缩
"""

import torch
from kv_cache_compression import HybridKVCacheCompressor, QuantizedKVCache
from typing import Dict, Optional, Tuple


class CompressedKVCacheManager:
    """KV Cache管理器，支持压缩存储"""
    
    def __init__(self, compression_config: Dict):
        """
        Args:
            compression_config: 压缩配置
                {
                    'quantize_bits': 8,
                    'use_sliding_window': True,
                    'window_size': 256,
                    'sparsity_ratio': 0.0
                }
        """
        self.compressor = HybridKVCacheCompressor(**compression_config)
        self.cache = {}  # 存储压缩后的KV cache
        
    def update_cache(self, layer_id: int, key: torch.Tensor, value: torch.Tensor):
        """更新指定层的KV cache（压缩存储）"""
        compressed = self.compressor.compress(key, value)
        self.cache[layer_id] = compressed
        
    def get_cache(self, layer_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取指定层的KV cache（自动反压缩）"""
        if layer_id not in self.cache:
            return None
        
        compressed_data = self.cache[layer_id]
        key, value = self.compressor.decompress(compressed_data)
        return key, value
    
    def clear_cache(self, layer_id: Optional[int] = None):
        """清空指定层或所有层的cache"""
        if layer_id is None:
            self.cache.clear()
        else:
            self.cache.pop(layer_id, None)


def attention_with_compressed_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_manager: CompressedKVCacheManager,
    layer_id: int,
    is_prefill: bool = False,
    enable_gqa: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用压缩KV cache的attention计算
    
    Args:
        query: [B, num_heads, q_len, head_dim]
        key: [B, num_kv_heads, kv_len, head_dim] (当前step的key)
        value: [B, num_kv_heads, kv_len, head_dim] (当前step的value)
        kv_cache_manager: KV cache管理器
        layer_id: 层ID
        is_prefill: 是否是prefill阶段
        enable_gqa: 是否启用GQA（Grouped Query Attention）
    
    Returns:
        output, new_key, new_value
    """
    # Prefill阶段：不压缩，直接计算
    if is_prefill:
        # 计算attention（支持GQA）
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=enable_gqa
        )
        # 存储KV cache（压缩）
        kv_cache_manager.update_cache(layer_id, key, value)
        return output, key, value
    
    # Decode阶段：从cache中恢复历史KV
    cached_kv = kv_cache_manager.get_cache(layer_id)
    
    if cached_kv is not None:
        cached_key, cached_value = cached_kv
        # 拼接历史KV和当前KV
        # 注意：需要处理GQA的情况
        new_key = torch.cat([cached_key, key], dim=-2)
        new_value = torch.cat([cached_value, value], dim=-2)
    else:
        new_key = key
        new_value = value
    
    # 计算attention（支持GQA）
    output = torch.nn.functional.scaled_dot_product_attention(
        query, new_key, new_value, is_causal=False, enable_gqa=enable_gqa
    )
    
    # 更新cache（压缩存储）
    kv_cache_manager.update_cache(layer_id, new_key, new_value)
    
    return output, new_key, new_value


# 使用示例
if __name__ == "__main__":
    # 配置压缩参数
    compression_config = {
        'quantize_bits': 8,           # INT8量化
        'use_sliding_window': True,   # 使用滑动窗口
        'window_size': 256,           # 窗口大小256
        'sparsity_ratio': 0.0         # 不使用稀疏化
    }
    
    # 创建KV cache管理器
    kv_manager = CompressedKVCacheManager(compression_config)
    
    # 模拟数据
    batch_size = 512
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    dtype = torch.bfloat16
    
    # Prefill阶段
    print("Prefill阶段:")
    q_len = 512
    query_prefill = torch.randn(batch_size, num_heads, q_len, head_dim, dtype=dtype)
    key_prefill = torch.randn(batch_size, num_kv_heads, q_len, head_dim, dtype=dtype)
    value_prefill = torch.randn(batch_size, num_kv_heads, q_len, head_dim, dtype=dtype)
    
    output_prefill, _, _ = attention_with_compressed_kv(
        query_prefill, key_prefill, value_prefill, kv_manager, layer_id=0, 
        is_prefill=True, enable_gqa=True
    )
    print(f"  Output shape: {output_prefill.shape}")
    
    # Decode阶段（多个token）
    print("\nDecode阶段:")
    for step in range(5):
        q_len = 1
        query_decode = torch.randn(batch_size, num_heads, q_len, head_dim, dtype=dtype)
        key_decode = torch.randn(batch_size, num_kv_heads, q_len, head_dim, dtype=dtype)
        value_decode = torch.randn(batch_size, num_kv_heads, q_len, head_dim, dtype=dtype)
        
        output_decode, new_key, new_value = attention_with_compressed_kv(
            query_decode, key_decode, value_decode, kv_manager, layer_id=0, 
            is_prefill=False, enable_gqa=True
        )
        print(f"  Step {step+1}: KV cache length = {new_key.shape[-2]}")
    
    # 检查内存使用
    cached_kv = kv_manager.get_cache(0)
    if cached_kv:
        cached_key, cached_value = cached_kv
        print(f"\n最终KV cache长度: {cached_key.shape[-2]}")
        print(f"压缩后内存: {cached_key.numel() * cached_key.element_size() / 1024 / 1024:.2f} MB")
        print(f"(原始512 tokens需要: {512 * num_kv_heads * head_dim * 2 / 1024 / 1024:.2f} MB)")

