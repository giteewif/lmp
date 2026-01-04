import sys
import os
import time
import torch
import copy

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
from transformers import AutoModelForCausalLM, AutoConfig

from models.Deepseek.deepseek_moe_16b_base.modeling_deepseek import DeepseekDecoderLayer
from models.Deepseek.mlpmodule import DeepseekOModel, DeepseekModule, DeepseekOCalModel
from accelerate import init_empty_weights
from lmp.init_meta_manager import InitMetaManager
from lmp.init_meta_manager_mp import InitMetaManagerMP

if __name__ == '__main__':
    path = "/mnt/zhengcf3/lmp/src/models/Deepseek/deepseek_moe_16b_base"
    time_hook("load_config")
    config = AutoConfig.from_pretrained(path , trust_remote_code=True)
    config._attn_implementation = "sdpa"
    time_hook_end("load_config")

    time_hook("init_meta")
    with init_empty_weights():
        module = DeepseekOCalModel(config=config)
        module2 = copy.deepcopy(module)
    time_hook_end("init_meta")

    # 使用多进程版本 InitMetaManagerMP（绕过 GIL，真正并行）
    imm = InitMetaManagerMP(num_workers=1)  # 4 个工作进程并行处理

    # 定义初始化函数（用于兼容性，但在多进程版本中会在工作进程中重新定义）
    def init_layer_func(layer_idx: int, model):
        """初始化单个 layer 的函数，返回原始 layer 和深拷贝"""
        with init_empty_weights():
            layer_h = DeepseekDecoderLayer(config, layer_idx)
            layer_h = layer_h.to(torch.bfloat16)
            layer_c = None
            print(f"Main process {os.getpid()}: Layer {layer_idx} initialized, dtype={layer_h.input_layernorm.weight.dtype}")
        return layer_h, layer_c
    

    # 多次提交测试
    num_runs = 3  # 运行次数
    num_layers = 28
    
    # 第一次提交：初始化 Manager 和 Queue，并启动工作进程
    print(f"\n{'='*60}")
    print(f"Submit Run 1/{num_runs} (Initial)")
    print(f"{'='*60}\n")
    
    time_hook("submit_all_run0")
    imm.submit_all(num_layers, init_layer_func, module, config=config, reset=False)
    time_hook_end("submit_all_run0")
    
    # 启动工作进程（必须在第一次 submit_all 之后）
    imm.start()
    
    for run_idx in range(num_runs):
        if run_idx > 0:
            print(f"\n{'='*60}")
            print(f"Submit Run {run_idx + 1}/{num_runs}")
            print(f"{'='*60}\n")
            
            # 每次提交所有 layer 的初始化任务（重置状态）
            time_hook(f"submit_all_run{run_idx}")
            imm.submit_all(num_layers, init_layer_func, module, config=config, reset=True)
            time_hook_end(f"submit_all_run{run_idx}")
        
        # 等待所有 layer 初始化完成
        time_hook(f"wait_layers_run{run_idx}")
        layers = []
        for i in range(num_layers):
            # 等待特定 layer 初始化完成，返回 (layer, layer_copy) 元组
            time_hook(f"layer_{i}_run{run_idx}")
            layer, layer_copy = imm.wait_layer(i)
            
            # 测试传递的 DecodeLayer（只在第一次运行或特定 layer 时打印详细信息）
            if run_idx == 0 or i < 3:
                print(f"\n=== Testing DecodeLayer {i} (Submit Run {run_idx + 1}) ===")
                print(f"Layer type: {type(layer)}")
                print(f"Layer device: {next(layer.parameters()).device if list(layer.parameters()) else 'N/A'}")
                print(f"Layer dtype: {next(layer.parameters()).dtype if list(layer.parameters()) else 'N/A'}")
                if hasattr(layer, 'input_layernorm'):
                    print(f"input_layernorm weight shape: {layer.input_layernorm.weight.shape}")
                    print(f"input_layernorm weight dtype: {layer.input_layernorm.weight.dtype}")
                print(f"Layer is valid: {layer is not None}")
                print(f"===============================\n")
            
            # 在多进程版本中，需要手动设置 layer 到模型中
            # 因为 model 对象无法传递给子进程
            if hasattr(module.model, 'layers') and i < len(module.model.layers):
                module.model.layers[i] = layer
            layers.append(layer)
            if layer_copy is not None:
                layers.append(layer_copy)
            time_hook_end(f"layer_{i}_run{run_idx}")
        
        time_hook_end(f"wait_layers_run{run_idx}")
        print(f"\nSubmit Run {run_idx + 1} completed. Total layers: {len(layers)}\n")
    
    # 停止所有工作进程
    imm.stop()
    
