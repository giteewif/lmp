"""
KV Cache 压缩方案
支持多种压缩方法：量化、稀疏化、低秩分解等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class QuantizedKVCache:
    """量化压缩的KV Cache"""
    
    def __init__(self, bits: int = 8, symmetric: bool = True):
        """
        Args:
            bits: 量化位数 (4, 8, 16)
            symmetric: 是否使用对称量化
        """
        self.bits = bits
        self.symmetric = symmetric
        self.scale = None
        self.zero_point = None
        
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """量化KV cache"""
        if self.bits == 16:
            return tensor, None, None  # 不压缩
        
        # 计算量化参数
        if self.symmetric:
            # 对称量化: [-max, max] -> [-2^(bits-1), 2^(bits-1)-1]
            max_val = tensor.abs().max()
            self.scale = max_val / (2 ** (self.bits - 1) - 1)
            self.zero_point = None
            quantized = torch.round(tensor / self.scale).clamp(
                -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
            ).to(torch.int8 if self.bits == 8 else torch.int4)
        else:
            # 非对称量化
            min_val = tensor.min()
            max_val = tensor.max()
            self.scale = (max_val - min_val) / (2 ** self.bits - 1)
            self.zero_point = torch.round(-min_val / self.scale).to(torch.int8)
            quantized = torch.round((tensor - min_val) / self.scale).clamp(0, 2 ** self.bits - 1)
            quantized = quantized.to(torch.int8 if self.bits == 8 else torch.int4)
        
        return quantized, self.scale, self.zero_point
    
    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor, 
                   zero_point: Optional[torch.Tensor] = None, 
                   dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """反量化"""
        if zero_point is None:
            result = quantized.float() * scale
        else:
            result = (quantized.float() - zero_point.float()) * scale
        
        # 转换回原始数据类型
        if dtype is not None:
            result = result.to(dtype)
        
        return result


class SparseKVCache:
    """稀疏化压缩的KV Cache - 只保留重要的KV对"""
    
    def __init__(self, sparsity_ratio: float = 0.5, method: str = "magnitude"):
        """
        Args:
            sparsity_ratio: 稀疏化比例 (0.0-1.0)
            method: 稀疏化方法 ("magnitude", "attention_score")
        """
        self.sparsity_ratio = sparsity_ratio
        self.method = method
        
    def compress(self, key: torch.Tensor, value: torch.Tensor, 
                attention_scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        压缩KV cache
        Returns:
            compressed_key, compressed_value, mask
        """
        batch_size, num_heads, seq_len, head_dim = key.shape
        
        if self.method == "magnitude":
            # 基于幅度的稀疏化
            key_norm = key.norm(dim=-1)  # [B, H, S]
            value_norm = value.norm(dim=-1)
            importance = (key_norm + value_norm) / 2
            
        elif self.method == "attention_score" and attention_scores is not None:
            # 基于attention score的稀疏化
            importance = attention_scores.mean(dim=1)  # [B, S]
            importance = importance.unsqueeze(1).expand(-1, num_heads, -1)
        else:
            # 默认使用magnitude
            key_norm = key.norm(dim=-1)
            value_norm = value.norm(dim=-1)
            importance = (key_norm + value_norm) / 2
        
        # 选择top-k重要的位置
        k = int(seq_len * (1 - self.sparsity_ratio))
        _, top_indices = torch.topk(importance, k, dim=-1)  # [B, H, k]
        
        # 创建mask
        mask = torch.zeros(batch_size, num_heads, seq_len, dtype=torch.bool, device=key.device)
        mask.scatter_(-1, top_indices, True)
        
        # 压缩
        compressed_key = key[mask.unsqueeze(-1).expand(-1, -1, -1, head_dim)]
        compressed_value = value[mask.unsqueeze(-1).expand(-1, -1, -1, head_dim)]
        
        return compressed_key, compressed_value, mask


class LowRankKVCache:
    """低秩分解压缩 - 将KV cache分解为低秩矩阵"""
    
    def __init__(self, rank: int = 64):
        """
        Args:
            rank: 低秩分解的秩 (rank << head_dim)
        """
        self.rank = rank
        
    def compress(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用SVD分解压缩
        Returns:
            key_U, key_S, value_U, value_S
        """
        batch_size, num_heads, seq_len, head_dim = key.shape
        
        # 重塑为 [B*H, S, D]
        key_reshaped = key.view(batch_size * num_heads, seq_len, head_dim)
        value_reshaped = value.view(batch_size * num_heads, seq_len, head_dim)
        
        # SVD分解
        key_U, key_S, key_Vh = torch.linalg.svd(key_reshaped, full_matrices=False)
        value_U, value_S, value_Vh = torch.linalg.svd(value_reshaped, full_matrices=False)
        
        # 只保留前rank个奇异值
        key_U = key_U[:, :, :self.rank]  # [B*H, S, rank]
        key_S = key_S[:, :self.rank]  # [B*H, rank]
        key_Vh = key_Vh[:, :self.rank, :]  # [B*H, rank, D]
        
        value_U = value_U[:, :, :self.rank]
        value_S = value_S[:, :self.rank]
        value_Vh = value_Vh[:, :self.rank, :]
        
        return key_U, key_S, key_Vh, value_U, value_S, value_Vh
    
    def decompress(self, key_U, key_S, key_Vh, value_U, value_S, value_Vh) -> Tuple[torch.Tensor, torch.Tensor]:
        """反压缩"""
        key = (key_U * key_S.unsqueeze(1)) @ key_Vh
        value = (value_U * value_S.unsqueeze(1)) @ value_Vh
        
        batch_size, num_heads = key_U.shape[0] // key_U.shape[0], key_U.shape[0] // key_U.shape[0]
        # 这里需要根据实际情况调整batch_size和num_heads
        return key, value


class SlidingWindowKVCache:
    """滑动窗口KV Cache - 只保留最近的N个token"""
    
    def __init__(self, window_size: int = 256):
        """
        Args:
            window_size: 窗口大小
        """
        self.window_size = window_size
        
    def compress(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """只保留最近的window_size个token"""
        seq_len = key.shape[-2]
        if seq_len <= self.window_size:
            return key, value
        
        # 保留最近的window_size个token
        compressed_key = key[:, :, -self.window_size:, :]
        compressed_value = value[:, :, -self.window_size:, :]
        
        return compressed_key, compressed_value


class HybridKVCacheCompressor:
    """混合压缩方案 - 结合多种压缩方法"""
    
    def __init__(self, 
                 quantize_bits: int = 8,
                 sparsity_ratio: float = 0.0,
                 use_sliding_window: bool = False,
                 window_size: int = 256):
        """
        Args:
            quantize_bits: 量化位数
            sparsity_ratio: 稀疏化比例
            use_sliding_window: 是否使用滑动窗口
            window_size: 滑动窗口大小
        """
        self.quantizer = QuantizedKVCache(bits=quantize_bits) if quantize_bits < 16 else None
        self.sparsifier = SparseKVCache(sparsity_ratio=sparsity_ratio) if sparsity_ratio > 0 else None
        self.sliding_window = SlidingWindowKVCache(window_size=window_size) if use_sliding_window else None
        
    def compress(self, key: torch.Tensor, value: torch.Tensor) -> Dict:
        """压缩KV cache"""
        compressed_key, compressed_value = key, value
        metadata = {'dtype': key.dtype}  # 保存原始数据类型
        
        # 1. 滑动窗口压缩
        if self.sliding_window:
            compressed_key, compressed_value = self.sliding_window.compress(compressed_key, compressed_value)
            metadata['window_size'] = compressed_key.shape[-2]
        
        # 2. 稀疏化
        if self.sparsifier:
            compressed_key, compressed_value, mask = self.sparsifier.compress(compressed_key, compressed_value)
            metadata['mask'] = mask
        
        # 3. 量化
        if self.quantizer:
            compressed_key, scale_k, zp_k = self.quantizer.quantize(compressed_key)
            compressed_value, scale_v, zp_v = self.quantizer.quantize(compressed_value)
            metadata['scale_k'] = scale_k
            metadata['scale_v'] = scale_v
            metadata['zp_k'] = zp_k
            metadata['zp_v'] = zp_v
        
        return {
            'key': compressed_key,
            'value': compressed_value,
            'metadata': metadata
        }
    
    def decompress(self, compressed_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """反压缩"""
        key = compressed_data['key']
        value = compressed_data['value']
        metadata = compressed_data['metadata']
        original_dtype = metadata.get('dtype', torch.bfloat16)
        
        # 1. 反量化
        if 'scale_k' in metadata:
            quantizer = self.quantizer
            key = quantizer.dequantize(key, metadata['scale_k'], metadata.get('zp_k'), dtype=original_dtype)
            value = quantizer.dequantize(value, metadata['scale_v'], metadata.get('zp_v'), dtype=original_dtype)
        
        # 2. 反稀疏化（如果需要）
        if 'mask' in metadata:
            # 这里需要根据mask重建完整形状
            # 简化实现，实际需要保存原始shape
            pass
        
        return key, value


def estimate_compression_ratio(original_key: torch.Tensor, compressed_data: Dict) -> float:
    """估算压缩比"""
    original_size = original_key.numel() * original_key.element_size()
    
    compressed_key = compressed_data['key']
    compressed_size = compressed_key.numel() * compressed_key.element_size()
    
    # 加上metadata大小
    metadata = compressed_data['metadata']
    if 'scale_k' in metadata:
        compressed_size += metadata['scale_k'].numel() * metadata['scale_k'].element_size()
    if 'scale_v' in metadata:
        compressed_size += metadata['scale_v'].numel() * metadata['scale_v'].element_size()
    
    return original_size / compressed_size


# 使用示例
if __name__ == "__main__":
    # 模拟KV cache
    batch_size = 512
    num_kv_heads = 8
    seq_len = 512
    head_dim = 128
    dtype = torch.bfloat16
    
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype)
    
    print(f"原始KV cache大小: {key.numel() * key.element_size() / 1024 / 1024:.2f} MB")
    
    # 测试量化压缩
    print("\n1. 量化压缩 (INT8):")
    quantizer = QuantizedKVCache(bits=8)
    key_q, scale_k, zp_k = quantizer.quantize(key)
    value_q, scale_v, zp_v = quantizer.quantize(value)
    print(f"  压缩后大小: {key_q.numel() * key_q.element_size() / 1024 / 1024:.2f} MB")
    print(f"  压缩比: {estimate_compression_ratio(key, {'key': key_q, 'value': value_q, 'metadata': {'scale_k': scale_k, 'scale_v': scale_v}}):.2f}x")
    
    # 测试稀疏化压缩
    print("\n2. 稀疏化压缩 (50%):")
    sparsifier = SparseKVCache(sparsity_ratio=0.5)
    key_s, value_s, mask = sparsifier.compress(key, value)
    print(f"  压缩后大小: {key_s.numel() * key_s.element_size() / 1024 / 1024:.2f} MB")
    
    # 测试滑动窗口
    print("\n3. 滑动窗口压缩 (256 tokens):")
    sliding = SlidingWindowKVCache(window_size=256)
    key_w, value_w = sliding.compress(key, value)
    print(f"  压缩后大小: {key_w.numel() * key_w.element_size() / 1024 / 1024:.2f} MB")
    
    # 测试混合压缩
    print("\n4. 混合压缩 (INT8 + 滑动窗口256):")
    hybrid = HybridKVCacheCompressor(quantize_bits=8, use_sliding_window=True, window_size=256)
    compressed = hybrid.compress(key, value)
    print(f"  压缩后大小: {compressed['key'].numel() * compressed['key'].element_size() / 1024 / 1024:.2f} MB")
    print(f"  压缩比: {estimate_compression_ratio(key, compressed):.2f}x")

