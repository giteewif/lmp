import sys
import os
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


def test():
    model_path = "deepseek-moe-16b-base-bfloat16"
    model_name_type = "Deepseek"
    mlpllm = MLPLLM( model_name_type=model_name_type, model_path=model_path )
    # mlpllm.test_generate_single_layer()
    mlpllm.test_generate_multi_layer()


if __name__ == "__main__":
    test()