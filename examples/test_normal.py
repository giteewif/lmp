import torch
import time
from transformers import MixtralForCausalLM, AutoModelForCausalLM
import copy
from accelerate import init_empty_weights

import sys, os
timings = {}
def time_hook(name):
    start = time.time()
    print(f"start {name}")
    timings[name] = start

def time_hook_end(name):
    end = time.time()
    
    cost = end - timings[name]
    print(f"end {name} cost {cost}")
# 获取项目根目录和必要的路径
project_root = "/mnt/zhengcf3/lmp"
src_dir = os.path.join(project_root, 'src')
sllm_store_dir = os.path.join(project_root, 'src', 'sllm_store')

# 添加必要的目录到 Python 路径
# 1. 添加 sllm_store 目录（必须在 src 之前，这样 sllm_store 可以被找到）
sys.path.insert(0, sllm_store_dir)
# 2. 添加 src 目录（用于导入 lmp, utils 等）
sys.path.insert(0, src_dir)

from models.Deepseek.deepseek_moe_16b_base.modeling_deepseek import DeepseekForCausalLM, DeepseekRMSNorm
from lmp.lmp import MLPLLM
from utils.cuda_h import *

# 加载 Mixtral 模型（只加载一个 expert）
#Mixtral-8x22B-Instruct-v0.1
model_path = "deepseek-moe-16b-base-bfloat16"
model_name_type = "Deepseek"
mlpllm = MLPLLM( model_name_type=model_name_type, model_path=model_path )
device1 = "cuda:1"
device1_index = int(device1.split(":")[1])

mlpllm.cmv.start_init_meta_model(hmv=mlpllm.hmv)
mlpllm.cmv.load_general_and_init()
mlpllm.cmv.init_load_qkvogn_es_weight(layer_idx=0)

layer_idx = 1
mlpllm.cmv.start_load_qkvogn_s_weight(layer_idx=1, device=device1)
mlpllm.cmv.wait_load_qkvogn_s_weight(layer_idx=1)

gpu_expert_ids = [i for i in range(mlpllm.mlpm.config.num_hidden_layers)]
gpu_expert_names = mlpllm.mlpm.get_experts_names(layer_idx=1, expert_idx_list=list(gpu_expert_ids))
gpu_expert_names = gpu_expert_names
ret1, replica_uuid1, state_dict1 = \
    mlpllm.cmv.allocate_cuda_memory_and_load_into_gpu(
    gpu_expert_names, device_index_int=device1_index)
mlpllm.cmv.restore2model(state_dict1, mlpllm.cmv.mlpm_ci)
mlpllm.cmv.wait_load_into_gpu(replica_uuid1)

input_hidden_states = torch.randn(32, 1, 2048, dtype=torch.bfloat16, device=device1)

layer = mlpllm.cmv.mlpm_ci.model.layers[layer_idx]
cuda_hook_time("one")
out = layer.mlp(input_hidden_states)
cuda_hook_time_end("one")

cuda_hook_time("two")
out = layer.mlp(input_hidden_states)
cuda_hook_time_end("two")

cuda_hook_time("two")
out = layer.mlp(input_hidden_states)
cuda_hook_time_end("two")