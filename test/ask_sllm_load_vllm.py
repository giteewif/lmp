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
storage_path = "/mnt/zhengcf3/models/vllm_sllm_models"
tdevice = "cuda:0"

#"deepseek-moe-16b-base" "Mixtral-8x22B" "Mixtral-8x7B" "Mixtral-8x7B-safetensors-hqq_hf"
# Qwen2-57B-A14B Qwen3-Coder-Next
# gemma4-26B-A4B ERNIE-4.5-VL-28B-A3B-Thinking ERNIE-4.5-21B-A3B-Thinking Qwen3.5-35B gpt-oss-20b
# Qwen1.5-MoE-A2.7B
model_name = "gemma4-26B-A4B"
# Qwen3_next Qwen2_moe
# Gemma4
model_name_type = "Gemma4"
model_path = f"/mnt/zhengcf3/models/vllm_sllm_models/{model_name}"


from sllm_store.client import SllmStoreClient
from sllm_store._C import (
    restore_tensors_from_shared_memory_names
)

# 8074 Qwen1.5-MoE-A2.7B 8073 Gemma4
client = SllmStoreClient("127.0.0.1:8074")
time_start_load_cpu = time.time()
ret = client.load_into_cpu(model_name)
if not ret:
    raise ValueError(f"Failed to load model {model_name} into CPU")
print(f"load into cpu cost {time.time() - time_start_load_cpu} s")
client.release_all_registered_fixed_gpu_ptrs()
# time_start = time.time()
# shm_names, chunk_size = client.get_model_shared_memory_names(model_name)
# print(f"get shm names {time.time()-time_start}")
# print(shm_names, chunk_size)

# tensor_resize_map = json.load(open(os.path.join(storage_path, model_name, "tensor_index_resize.json"), "r"))

# state_dict = restore_tensors_from_shared_memory_names(shm_names, tensor_resize_map,chunk_size)
# print(state_dict["model.layers.0.block_sparse_moe.experts.0.w1.weight"])
