import torch
import torch.nn.functional as F
from transformers import AutoConfig


class Gemma4Module:
    """Gemma4 text MoE helpers for `models/mlpmodule.py`.

    Notes:
    - Gemma4 的 MoE experts 为数字子模块名 ``0``..``N-1``（``add_module``），每项含 ``gate_up_proj`` / ``down_proj`` 两个 ``Linear``，与 ``tensor_index`` 中 ``...experts.{i}.*.weight`` 一致。
    - 路由器是 `layer.router`，直接返回 top-k weights / indices。
    """

    def get_config(self, path: str):
        return AutoConfig.from_pretrained(path, trust_remote_code=True)

    @torch.no_grad()
    def experts_func(
        mi,
        layer_idx: int,
        expert_idx: int,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        token_indices: torch.Tensor,
        final_hidden_states: torch.Tensor,
    ):
        """Run one expert FFN and scatter-add into `final_hidden_states`.

        This is a best-effort compatibility path with the existing per-expert scheduling code. For Gemma4, the preferred
        path is to call `layer.experts(hidden_states_flat, top_k_index, top_k_weights)` directly (vectorized top-k).
        """
        lm = getattr(mi.model, "language_model", mi.model)
        layer = lm.layers[layer_idx]
        expert_mlp = getattr(layer.experts, str(int(expert_idx)))
        out = expert_mlp(tokens)
        out.mul_(weights)

        if out.device != final_hidden_states.device:
            out = out.to(final_hidden_states.device, non_blocking=True)

        final_hidden_states.scatter_reduce_(
            dim=0,
            index=token_indices.view(-1, 1).repeat(1, final_hidden_states.shape[-1]),
            src=out,
            reduce="sum",
        )
        return final_hidden_states

    def scatter(expert_cache, expert_out_map, expert_token_indices_map):
        for expert_id in sorted(expert_out_map.keys()):
            expert_out = expert_out_map[expert_id]
            token_indices = expert_token_indices_map[expert_id]
            indices_expanded = token_indices.view(-1, 1).repeat(1, expert_cache.shape[-1])
            expert_cache.scatter_reduce_(
                dim=0,
                index=indices_expanded,
                src=expert_out,
                reduce="sum",
            )
        return expert_cache

