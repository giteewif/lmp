"""
测量空模型对象（使用 init_empty_weights 创建）的内存占用
"""
import sys
import os
import torch
import gc

# 尝试导入 pympler（可选）
try:
    from pympler import asizeof
    HAS_PYMPLER = True
except ImportError:
    HAS_PYMPLER = False

# 获取项目根目录和必要的路径
project_root = "/mnt/zhengcf3/lmp"
src_dir = os.path.join(project_root, 'src')
sllm_store_dir = os.path.join(project_root, 'src', 'sllm_store')

# 添加必要的目录到 Python 路径
sys.path.insert(0, sllm_store_dir)
sys.path.insert(0, src_dir)

from transformers import AutoConfig
from models.Deepseek.deepseek_moe_16b_base.modeling_deepseek import DeepseekForCausalLM
from accelerate import init_empty_weights

def format_size(size_bytes):
    """格式化字节大小为可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def measure_model_size():
    """测量空模型对象的内存占用"""
    
    # 加载配置
    path = "/mnt/zhengcf3/lmp/src/models/Deepseek/deepseek_moe_16b_base"
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    
    print("=" * 60)
    print("测量空模型对象的内存占用")
    print("=" * 60)
    
    # 方法1: 使用 sys.getsizeof (只测量对象本身，不包括嵌套对象)
    print("\n方法1: sys.getsizeof (浅层大小)")
    gc.collect()
    with init_empty_weights():
        model = DeepseekForCausalLM(config)
    
    size_sys = sys.getsizeof(model)
    print(f"  sys.getsizeof(model): {format_size(size_sys)}")
    
    # 方法2: 使用 pympler.asizeof (递归测量所有嵌套对象)
    if HAS_PYMPLER:
        print("\n方法2: pympler.asizeof (深层大小，包括所有嵌套对象)")
        size_pympler = asizeof.asizeof(model)
        print(f"  asizeof.asizeof(model): {format_size(size_pympler)}")
    else:
        print("\n方法2: pympler.asizeof (跳过)")
        print("  pympler 未安装，跳过此方法")
        print("  安装命令: pip install pympler")
    
    # 方法3: 统计模型参数数量（空权重）
    print("\n方法3: 模型结构统计")
    total_params = 0
    total_buffers = 0
    param_count = 0
    buffer_count = 0
    
    for name, param in model.named_parameters():
        param_count += 1
        if param.numel() > 0:
            total_params += param.numel()
    
    for name, buffer in model.named_buffers():
        buffer_count += 1
        if buffer.numel() > 0:
            total_buffers += buffer.numel()
    
    print(f"  参数数量: {param_count}")
    print(f"  参数元素总数: {total_params:,}")
    print(f"  缓冲区数量: {buffer_count}")
    print(f"  缓冲区元素总数: {total_buffers:,}")
    
    # 方法4: 测量模型状态字典的大小（如果权重为空，应该很小）
    print("\n方法4: 模型状态字典大小")
    state_dict = model.state_dict()
    state_dict_size = sys.getsizeof(state_dict)
    print(f"  state_dict 对象大小: {format_size(state_dict_size)}")
    
    # 统计状态字典中的键值对
    print(f"  状态字典键数量: {len(state_dict)}")
    
    # 方法5: 递归统计所有子模块
    print("\n方法5: 模型结构统计")
    module_count = 0
    for name, module in model.named_modules():
        module_count += 1
    
    print(f"  子模块数量: {module_count}")
    
    # 方法6: 使用 torch 的内存分析（空模型不能移到 CUDA）
    print("\n方法6: PyTorch 内存统计")
    print("  (注意: 空模型使用 meta tensor，不能移到 CUDA)")
    print("  空模型的内存占用主要在 CPU 上，主要是对象结构本身")
    
    # 方法7: 递归统计所有模块对象大小
    print("\n方法7: 递归统计所有模块对象大小")
    model_size_estimate = 0  # 初始化
    try:
        module_sizes = []
        for name, module in model.named_modules():
            try:
                size = sys.getsizeof(module)
                model_size_estimate += size
                module_sizes.append((name, size))
            except:
                pass
        print(f"  所有模块对象大小总和: {format_size(model_size_estimate)}")
        print(f"  模块数量: {len(module_sizes)}")
    except Exception as e:
        print(f"  无法计算: {e}")
        model_size_estimate = 0
    
    # 方法8: 估算空权重张量的元数据大小
    print("\n方法8: 空权重张量的元数据大小估算")
    total_meta_size = 0
    for name, param in model.named_parameters():
        # 每个张量的元数据（shape, dtype, device 等）大约占用一些内存
        # 这里只估算元数据，不包括实际数据
        meta_size = sys.getsizeof(param)  # 张量对象本身
        total_meta_size += meta_size
    
    for name, buffer in model.named_buffers():
        meta_size = sys.getsizeof(buffer)
        total_meta_size += meta_size
    
    print(f"  所有张量对象（元数据）大小: {format_size(total_meta_size)}")
    
    # 计算总计（避免重复计算）
    print("\n" + "=" * 60)
    print("总计计算:")
    print("=" * 60)
    
    # 方法1: 使用递归方式计算（如果可能）
    total_estimated = 0
    
    # 基础对象大小
    total_estimated += size_sys  # 模型对象本身
    
    # 模块对象大小（已经计算过）
    total_estimated += model_size_estimate
    
    # 状态字典大小（已经计算过）
    total_estimated += state_dict_size
    
    # 注意：张量元数据可能已经包含在模块对象中，所以可能重复计算
    # 但为了保守估计，我们加上它
    total_estimated += total_meta_size
    
    print(f"  保守估计总计: {format_size(total_estimated)}")
    print(f"  (注意: 可能存在部分重复计算)")
    
    # 更准确的方法：只计算主要部分
    # 1. 模型对象本身
    # 2. 所有模块对象（包含参数引用）
    # 3. 状态字典（包含参数引用，但字典结构本身）
    # 4. 配置对象
    try:
        config_size = sys.getsizeof(config)
        print(f"\n  配置对象大小: {format_size(config_size)}")
    except:
        config_size = 0
    
    # 更准确的估计：模块对象 + 状态字典结构 + 配置
    accurate_total = model_size_estimate + state_dict_size + config_size
    print(f"\n  更准确的估计: {format_size(accurate_total)}")
    print(f"  (模块对象 + 状态字典结构 + 配置对象)")
    
    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    print("空模型对象主要包含:")
    print("  1. Python 对象本身的开销（类定义、属性等）")
    print("  2. 模型结构定义（所有子模块、层的定义）")
    print("  3. 空的权重张量（shape 信息，但没有实际数据）")
    print("  4. 配置信息")
    print(f"\n保守估计总计: {format_size(total_estimated)}")
    print(f"更准确估计总计: {format_size(accurate_total)}")
    print("\n实际权重数据的内存占用会在加载权重时分配。")
    print(f"如果所有参数都加载为 bfloat16，预计需要:")
    param_size_gb = (total_params * 2) / (1024**3)  # bfloat16 = 2 bytes
    print(f"  {param_size_gb:.2f} GB (仅参数)")
    buffer_size_gb = (total_buffers * 2) / (1024**3)
    print(f"  {buffer_size_gb:.2f} GB (仅缓冲区)")
    print(f"  总计: {param_size_gb + buffer_size_gb:.2f} GB")
    
    return model

if __name__ == '__main__':
    model = measure_model_size()

