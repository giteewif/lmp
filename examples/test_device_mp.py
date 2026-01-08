"""
测试 DeviceMP - 共享 DecoderLayer 对象版本
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
from lmp.device_mp import DeviceMP
from utils.cuda_h import cuda_hook, cuda_hook_end, cuda_hook_time, cuda_hook_time_end

if __name__ == '__main__':

    dp = DeviceMP(num_processes=1)
    dp.start()

    input_tensor = torch.randn(32, 64, 2048, dtype=torch.bfloat16, device="cuda:0")
    dp.submit(input_tensor)

    output_tensor_list = []
    for i in range(10):
        cuda_hook_time("one")

        dp.submit(input_tensor)
        output_tensor = dp.wait()
        
        output_tensor_list.append(output_tensor)
        cuda_hook_time_end("one")
    # print(output_tensor)
    time.sleep(2)
    dp.stop()