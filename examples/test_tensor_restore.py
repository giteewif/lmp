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
from utils.cuda_h import cuda_hook_time, cuda_hook_time_end
from utils.logger import init_logger

logger = init_logger(__name__)
def test():
    model_path = "deepseek-moe-16b-base-bfloat16"
    model_name_type = "Deepseek"
    mlpllm = MLPLLM( model_name_type=model_name_type, model_path=model_path )
    # only for mp process
    # mlpllm.init_mp_process()
    for i in range(5):
        hmv = mlpllm.hmv
        layer_idx = i+1
        expert_idx_list = [i for i in range(32)]
        cuda_hook_time("group_tensors")
        hmv.group_experts_tensor(layer_idx=1, expert_idx_list=expert_idx_list)
        cuda_hook_time_end("group_tensors")
    # mlpllm.imm.stop()
    # mlpllm.mp_stop()
if __name__ == "__main__":
    test()