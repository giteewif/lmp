import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights

from models.Deepseek.deepseek_moe_16b_base.modeling_deepseek import DeepseekForCausalLM, DeepseekRMSNorm

class _PlaceholderLayer(nn.Module):
    """占位符 layer，用于预分配 ModuleList"""
    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise RuntimeError("Placeholder layer should not be called. Please set the actual layer first.")

class DeepseekOModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 先创建空的 ModuleList，使用占位符模块，后续再填充
        self.layers = nn.ModuleList([_PlaceholderLayer() for _ in range(config.num_hidden_layers)])
    
    def set_layer(self, layer_idx: int, layer: nn.Module):
        """
        设置指定索引的 layer
        
        Args:
            layer_idx: layer 索引
            layer: DeepseekDecoderLayer 实例
        """
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {len(self.layers)})")
        self.layers[layer_idx] = layer
class DeepseekOCalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = DeepseekOModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
class DeepseekModule():
    def create_empty_model(self, config: AutoConfig):
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            model = DeepseekForCausalLM(config)
            return model

    def get_config(self, path: str):
        config = AutoConfig.from_pretrained(path , trust_remote_code=True)
        return config

    def gate_func(mi: DeepseekForCausalLM, layer_idx: int, hidden_states: torch.Tensor):
        topk_idx, topk_weight, aux_loss = mi.model.layers[layer_idx].mlp.gate(hidden_states)
        return topk_idx, topk_weight, aux_loss

    @torch.no_grad()
    def experts_func(
        mi: DeepseekForCausalLM, 
        layer_idx: int, 
        expert_idx: int,
        tokens: torch.Tensor, 
        weights: torch.Tensor,
        token_indices: torch.Tensor,
        final_hidden_states: torch.Tensor,
    ):
        """
        在指定设备上执行expert计算
        
        Args:
            mi: 模型实例
            layer_idx: 层索引
            expert_idx: expert索引
            tokens: 输入tokens
            weights: 路由权重
        
        Returns:
            final_hidden_states: 最终隐藏状态
        """
        expert = mi.model.layers[layer_idx].mlp.experts[expert_idx]
        expert_out = expert(tokens)
        expert_out.mul_(weights)
        # support cpu
        if expert_out.device != final_hidden_states.device:
            expert_out = expert_out.to(final_hidden_states.device, non_blocking=True)
        final_hidden_states.scatter_reduce_(
            dim=0,
            index=token_indices.view(-1, 1).repeat(1, final_hidden_states.shape[-1]),
            src=expert_out,
            reduce='sum'
        )
        return final_hidden_states
        
    def scatter(
        expert_cache, 
        expert_out_map, 
        expert_token_indices_map
    ):
        """
        将 expert 输出 scatter 回原始 hidden_states 的位置
        
        Args:
            expert_cache: [num_tokens, hidden_dim] 聚合后的输出
            expert_out_map: {expert_id: output} expert 的输出，已应用权重
            expert_token_indices_map: {expert_id: token_indices} 每个 expert 处理的 token 索引
        
        Returns:
            expert_cache: [num_tokens, hidden_dim] 聚合后的输出
        """
        # 初始化输出张量
        
        # 按 expert_id 顺序处理（确保可重复性）
        for expert_id in sorted(expert_out_map.keys()):
            expert_out = expert_out_map[expert_id]
            token_indices = expert_token_indices_map[expert_id]
            
            # 将 token_indices 扩展到匹配 hidden_dim
            # token_indices: [num_expert_tokens]
            # 需要变成: [num_expert_tokens, hidden_dim]
            indices_expanded = token_indices.view(-1, 1).repeat(1, expert_cache.shape[-1])
            
            # 使用 scatter_reduce 将 expert 输出加到对应位置
            # dim=0 表示在第一个维度（token 维度）上 scatter
            # reduce='sum' 表示多个 expert 的输出相加
            expert_cache.scatter_reduce_(
                dim=0,
                index=indices_expanded,
                src=expert_out,
                reduce='sum'
            )
        
        return expert_cache