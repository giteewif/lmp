"""
测试 InitMetaManagerMPShared - 共享 DecoderLayer 对象版本
"""
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

from transformers import AutoConfig
from models.Deepseek.deepseek_moe_16b_base.modeling_deepseek import DeepseekDecoderLayer
from models.Deepseek.mlpmodule import DeepseekOCalModel
from accelerate import init_empty_weights
from lmp.init_meta_manager_mp_shared import InitMetaManagerMPShared
from utils.cuda_h import cuda_hook, cuda_hook_end, cuda_hook_time, cuda_hook_time_end
# 定义初始化函数（必须在模块级别定义，以便可以被 pickle）
# 注意：使用 spawn 启动方法时，函数必须在模块级别定义，不能在 __main__ 中定义
# config 将通过 model.config 传递
def init_layer_func(layer_idx: int, config):
    """初始化单个 layer 的函数，返回原始 layer 和深拷贝"""
    
    with init_empty_weights():
        layer_h = DeepseekDecoderLayer(config, layer_idx)
        layer_h = layer_h.to(torch.bfloat16)
        layer_c = None
        print(f"Init process {os.getpid()}: Layer {layer_idx} initialized, dtype={layer_h.input_layernorm.weight.dtype}")
    return layer_h, layer_c

if __name__ == '__main__':
    path = "/mnt/zhengcf3/lmp/src/models/Deepseek/deepseek_moe_16b_base"
    cuda_hook_time("load_config")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    cuda_hook_time_end("load_config")

    cuda_hook_time("init_meta")
    with init_empty_weights():
        module = DeepseekOCalModel(config=config)
        module2 = copy.deepcopy(module)
    cuda_hook_time_end("init_meta")
    
    # 将 config 添加到 model 中，以便 init_layer_func 可以访问
    # 注意：这是必需的，因为 init_layer_func 在独立进程中运行，无法访问 __main__ 中的变量
    module.config = config

    # 使用共享 DecoderLayer 版本 InitMetaManagerMPShared
    imm = InitMetaManagerMPShared(num_processes=2)

    cuda_hook_time("start_workers")
    imm.start()
    cuda_hook_time_end("start_workers")

    time.sleep(3)
    for i in range(3):
        cuda_hook_time("one_submit_all")
        cuda_hook_time("submit_all")
        # 一次性提交所有 layer 的初始化任务
        # 注意：这个版本在主进程中初始化所有 layer，然后共享给工作进程
        imm.submit_all(init_layer_func, config=config)
        cuda_hook_time_end("submit_all")
        
        
        cuda_hook_time("wait_all")
        imm.wait_all()
        cuda_hook_time_end("wait_all")

        cuda_hook_time_end("one_submit_all")
        imm._reset_cache()
        # time.sleep(1)

    # 停止所有工作进程
    cuda_hook_time("stop_workers")
    imm.stop()
    cuda_hook_time_end("stop_workers")
    
    print("Test completed!")

