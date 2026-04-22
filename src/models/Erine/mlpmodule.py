from transformers import AutoConfig
from accelerate import init_empty_weights
from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeForCausalLM


class ErineMoeModule:
    """Host-side config / empty-model helper for Ernie 4.5 MoE (``ERINE_MODEL_NAME_TYPE``)."""

    def create_empty_model(self, config):
        with init_empty_weights():
            return Ernie4_5_MoeForCausalLM(config)

    def get_config(self, path):
        return AutoConfig.from_pretrained(path, trust_remote_code=True)
