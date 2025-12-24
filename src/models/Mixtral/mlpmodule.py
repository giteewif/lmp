from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

class MixtralModule:
    def create_empty_model(self, config):
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            model = MixtralForCausalLM(config)
            return model
    
    def get_config(self, path):
        pass