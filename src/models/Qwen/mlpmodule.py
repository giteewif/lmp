import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeDecoderLayer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeDecoderLayer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeDecoderLayer

class Qwen2MoEModule:
    def create_empty_model(self, config):
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            model = Qwen2MoeForCausalLM(config)
            return model
    
    def get_config(self, path):
        config = AutoConfig.from_pretrained(path , trust_remote_code=True)
        return config

class Qwen3MoEModule:
    def create_empty_model(self, config):
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            model = Qwen3MoeForCausalLM(config)
            return model
    def get_config(self, path):
        config = AutoConfig.from_pretrained(path , trust_remote_code=True)
        return config

class Qwen3_5MoEModule:
    def create_empty_model(self, config):
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            model = Qwen3_5MoeForCausalLM(config)
            return model
            
    def get_config(self, path):
        config = AutoConfig.from_pretrained(path , trust_remote_code=True)
        return config