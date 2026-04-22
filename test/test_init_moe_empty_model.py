# from sllm_store.transformers import save_model

# # Load a model from HuggingFace model hub.
# import torch
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained('/mnt/zhengcf3/models/deepseek-moe-16b-base', torch_dtype=torch.float16, trust_remote_code=True)

# # Replace './models' with your local path.
# save_model(model, '/mnt/zhengcf3/models/models/deepseek-moe-16b-base')
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '/mnt/zhengcf3/lmp/src/sllm_store')

import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import json

import time
storage_path = "/mnt/zhengcf3/models/sllm_models"
tdevice = "cuda:0"

#"deepseek-moe-16b-base" "Mixtral-8x22B" "Mixtral-8x7B" "Mixtral-8x7B-safetensors-hqq_hf"
# Qwen2-57B-A14B Qwen3-Coder-Next
# gemma4-26B-A4B ERNIE-4.5-VL-28B-A3B-Thinking ERNIE-4.5-21B-A3B-Thinking Qwen3.5-35B gpt-oss-20b

# Qwen3_next Qwen2_moe

from accelerate import init_empty_weights
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeDecoderLayer, Qwen2MoeRotaryEmbedding
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4TextDecoderLayer, Gemma4ForConditionalGeneration

model_path = "/mnt/zhengcf3/models/gemma4-26B-A4B"
config = AutoConfig.from_pretrained(model_path)

start_time = time.time()
with init_empty_weights():
    # config = config.text_config
    config._attn_implementation = "sdpa"
    cm = Gemma4ForConditionalGeneration(config)
print(f"init empty model cost {time.time() - start_time} s")


# start_time = time.time()
# cm = Qwen2MoeForCausalLM(config)
# print(f"init model cost {time.time() - start_time} s")