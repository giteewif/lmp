import sys
import os
import time
import torch
import copy

# 获取项目根目录和必要的路径
project_root = "/mnt/zhengcf3/lmp"
src_dir = os.path.join(project_root, 'src')
sllm_store_dir = os.path.join(project_root, 'src', 'sllm_store')

# 添加必要的目录到 Python 路径
sys.path.insert(0, sllm_store_dir)
sys.path.insert(0, src_dir)

from utils.cuda_h import cuda_hook, cuda_hook_end, cuda_hook_time, cuda_hook_time_end

from transformers import AutoConfig
from models.Deepseek.deepseek_moe_16b_base.modeling_deepseek import DeepseekDecoderLayer
from accelerate import init_empty_weights

def init_layer_func(layer_idx: int, config):
    """初始化单个 layer 的函数，返回原始 layer 和深拷贝"""
    with init_empty_weights():
        layer_h = DeepseekDecoderLayer(config, layer_idx)
        layer_h = layer_h.to(torch.bfloat16)
        layer_c = None
        print(f"Init process {os.getpid()}: Layer {layer_idx} initialized, dtype={layer_h.input_layernorm.weight.dtype}")
    return layer_h, layer_c
