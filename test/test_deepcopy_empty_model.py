"""
测试深度拷贝空模型对象的行为和影响
"""
import sys
import os
import copy
import torch

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

def test_deepcopy_empty_model():
    """测试深度拷贝空模型对象"""
    
    print("=" * 60)
    print("测试深度拷贝空模型对象")
    print("=" * 60)
    
    # 加载配置
    path = "/mnt/zhengcf3/lmp/src/models/Deepseek/deepseek_moe_16b_base"
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    
    # 创建原始空模型
    print("\n1. 创建原始空模型...")
    with init_empty_weights():
        model = DeepseekForCausalLM(config)
    
    print(f"   原始模型类型: {type(model)}")
    print(f"   原始模型 ID: {id(model)}")
    
    # 深度拷贝模型
    print("\n2. 深度拷贝模型...")
    import time
    start_time = time.time()
    model_cpy = copy.deepcopy(model)
    copy_time = time.time() - start_time
    print(f"   拷贝耗时: {copy_time:.4f} 秒")
    print(f"   拷贝模型类型: {type(model_cpy)}")
    print(f"   拷贝模型 ID: {id(model_cpy)}")
    print(f"   是否为同一对象: {model is model_cpy}")
    
    # 检查独立性
    print("\n3. 检查独立性...")
    print("   3.1 检查模型对象是否独立:")
    print(f"      model.model ID: {id(model.model)}")
    print(f"      model_cpy.model ID: {id(model_cpy.model)}")
    print(f"      是否独立: {model.model is not model_cpy.model}")
    
    print("\n   3.2 检查配置对象:")
    print(f"      model.config ID: {id(model.config)}")
    print(f"      model_cpy.config ID: {id(model_cpy.config)}")
    print(f"      配置是否共享: {model.config is model_cpy.config}")
    print("      (注意: 配置对象通常会被共享，这是正常的)")
    
    # 检查子模块独立性
    print("\n   3.3 检查子模块独立性 (检查前3个):")
    model_modules = list(model.named_modules())[:3]
    cpy_modules = list(model_cpy.named_modules())[:3]
    for (name1, mod1), (name2, mod2) in zip(model_modules, cpy_modules):
        print(f"      {name1}: 原始 ID={id(mod1)}, 拷贝 ID={id(mod2)}, 独立={mod1 is not mod2}")
    
    # 检查参数独立性
    print("\n   3.4 检查参数独立性 (检查前3个):")
    model_params = list(model.named_parameters())[:3]
    cpy_params = list(model_cpy.named_parameters())[:3]
    for (name1, param1), (name2, param2) in zip(model_params, cpy_params):
        print(f"      {name1}: 原始 ID={id(param1)}, 拷贝 ID={id(param2)}, 独立={param1 is not param2}")
        print(f"        原始 shape: {param1.shape}, 拷贝 shape: {param2.shape}")
        print(f"        原始 dtype: {param1.dtype}, 拷贝 dtype: {param2.dtype}")
        print(f"        原始 device: {param1.device}, 拷贝 device: {param2.device}")
    
    # 测试修改原始模型是否影响拷贝
    print("\n4. 测试修改独立性...")
    original_attr = getattr(model, 'vocab_size', None)
    model.vocab_size = 99999  # 修改原始模型
    cpy_attr = getattr(model_cpy, 'vocab_size', None)
    print(f"   修改原始模型 vocab_size 为 99999")
    print(f"   原始模型 vocab_size: {model.vocab_size}")
    print(f"   拷贝模型 vocab_size: {model_cpy.vocab_size}")
    print(f"   是否独立: {model.vocab_size != model_cpy.vocab_size}")
    model.vocab_size = original_attr  # 恢复
    
    # 测试内存占用
    print("\n5. 内存占用对比...")
    import sys
    def get_size(obj):
        return sys.getsizeof(obj)
    
    model_size = get_size(model)
    cpy_size = get_size(model_cpy)
    print(f"   原始模型对象大小: {model_size} bytes")
    print(f"   拷贝模型对象大小: {cpy_size} bytes")
    
    # 统计模块数量
    print("\n6. 结构统计...")
    model_modules_count = len(list(model.named_modules()))
    cpy_modules_count = len(list(model_cpy.named_modules()))
    model_params_count = len(list(model.named_parameters()))
    cpy_params_count = len(list(model_cpy.named_parameters()))
    
    print(f"   原始模型 - 模块数: {model_modules_count}, 参数数: {model_params_count}")
    print(f"   拷贝模型 - 模块数: {cpy_modules_count}, 参数数: {cpy_params_count}")
    print(f"   结构是否一致: {model_modules_count == cpy_modules_count and model_params_count == cpy_params_count}")
    
    # 测试是否可以独立使用
    print("\n7. 测试是否可以独立使用...")
    print("   7.1 测试状态字典:")
    model_state = model.state_dict()
    cpy_state = model_cpy.state_dict()
    print(f"      原始模型状态字典键数: {len(model_state)}")
    print(f"      拷贝模型状态字典键数: {len(cpy_state)}")
    print(f"      状态字典是否独立: {model_state is not cpy_state}")
    
    print("\n   7.2 测试模型方法:")
    try:
        # 测试一些基本方法
        model_embeddings = model.get_input_embeddings()
        cpy_embeddings = model_cpy.get_input_embeddings()
        print(f"      原始模型 embeddings ID: {id(model_embeddings)}")
        print(f"      拷贝模型 embeddings ID: {id(cpy_embeddings)}")
        print(f"      embeddings 是否独立: {model_embeddings is not cpy_embeddings}")
    except Exception as e:
        print(f"      测试方法时出错: {e}")
    
    # 关于推理的影响
    print("\n" + "=" * 60)
    print("关于推理的影响:")
    print("=" * 60)
    print("""
    1. **空模型的深度拷贝是安全的**:
       - 空模型使用 meta tensor，不包含实际权重数据
       - 深度拷贝会创建完全独立的对象结构
       - 不会影响原始模型
    
    2. **对推理的影响**:
       - 如果两个模型都加载了相同的权重，它们会共享权重数据（如果权重是共享的）
       - 如果权重是独立的，两个模型可以独立推理，互不影响
       - 空模型本身不能直接推理（需要加载权重）
    
    3. **使用场景**:
       - 适合创建多个独立的模型实例
       - 适合在不同进程/线程中使用
       - 适合创建模型的多个副本用于不同目的
    
    4. **注意事项**:
       - 深度拷贝会复制所有对象结构，内存占用会翻倍
       - 配置对象可能被共享（这是正常的，配置通常不需要独立）
       - 如果后续加载权重，需要确保权重加载到正确的模型实例
    """)
    
    return model, model_cpy

if __name__ == '__main__':
    model, model_cpy = test_deepcopy_empty_model()

