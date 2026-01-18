import sys
import os
from time import sleep
import torch

# 获取项目根目录和必要的路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
sllm_store_dir = os.path.join(project_root, 'src', 'sllm_store')

# 添加必要的目录到 Python 路径
# 1. 添加 sllm_store 目录（必须在 src 之前，这样 sllm_store 可以被找到）
sys.path.insert(0, sllm_store_dir)
# 2. 添加 src 目录（用于导入 lmp, utils 等）
sys.path.insert(0, src_dir)

from lmp.lmp import MLPLLM
from models.mlpmodule import QWEN2_MODEL_NAME_TYPE

def test():
    # model_path = "deepseek-moe-16b-base-bfloat16"
    model_path = "Qwen1.5-MoE-A2.7B"
    # model_name_type = "Deepseek"
    model_name_type = QWEN2_MODEL_NAME_TYPE
    mlpllm = MLPLLM( model_name_type=model_name_type, model_path=model_path )
    # only for mp process
    mlpllm.init_mp_process()
    for i in range(3):
        mlpllm.free_cmv()
        torch.cuda.empty_cache()
        sleep(2)
        # mlpllm.test_generate_multi_layer()
        mlpllm.test_mp_generate_multi_device_layer()
    sleep(2)
    # mlpllm.imm.stop()
    mlpllm.mp_stop()
if __name__ == "__main__":
    test()