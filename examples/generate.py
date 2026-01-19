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



def warm_up():
    num_gpus = torch.cuda.device_count()
    print(f"Warming up {num_gpus} GPU(s)...")
    for i in range(num_gpus):
        device = f"cuda:{i}"
        # 预热基本操作
        for _ in range(10):
            # 创建和移动 tensor
            x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
            y = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
            # 矩阵乘法
            z = torch.matmul(x, y)
            # Batch matrix multiplication (bmm)
            x_batch = torch.randn(8, 1024, 512, dtype=torch.bfloat16, device=device)
            y_batch = torch.randn(8, 512, 1024, dtype=torch.bfloat16, device=device)
            z_batch = torch.bmm(x_batch, y_batch)
            # Einsum (常用于 MoE 计算)
            a = torch.randn(8, 1024, 512, dtype=torch.bfloat16, device=device)
            b = torch.randn(8, 512, 1024, dtype=torch.bfloat16, device=device)
            c = torch.einsum('bij,bjk->bik', a, b)
            # 激活函数 (SiLU)
            x_act = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
            x_act = torch.nn.functional.silu(x_act)
            # 转置操作
            x_t = x.transpose(0, 1)
        torch.cuda.synchronize(device)
        print(f"GPU {i} warmed up")
    print("GPU warmup completed")
from lmp.lmp import MLPLLM
from models.mlpmodule import DEEPSEEK_MODEL_NAME_TYPE, QWEN2_MODEL_NAME_TYPE, MIXTRAL_MODEL_NAME_TYPE
def test():
    for i in range(3):
        warm_up()
    # model_path = "deepseek-moe-16b-base-bfloat16" or Qwen1.5-MoE-A2.7B Mixtral-8x7B "DeepSeek-V2-Lite"
    model_path = "deepseek-moe-16b-base-bfloat16"
    # model_name_type = "Deepseek"
    model_name_type = DEEPSEEK_MODEL_NAME_TYPE
    device_num = 3
    mlpllm = MLPLLM(model_name_type=model_name_type, model_path=model_path, device_num=device_num)
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