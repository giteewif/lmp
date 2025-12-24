"""
诊断 dtype 问题：检查从 tensor_index_resize.json 到 C++ 的 dtype 传递
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.helper import load_json

def check_tensor_index_resize_dtype():
    """检查 tensor_index_resize.json 中的 dtype 信息"""
    
    tensor_index_resize_path = "/mnt/zhengcf3/models/sllm_models/deepseek-moe-16b-base-bfloat16/tensor_index_resize.json"
    
    print("="*60)
    print("检查 tensor_index_resize.json 中的 dtype 信息")
    print("="*60)
    
    tensor_index_resize = load_json(tensor_index_resize_path)
    
    # 查找一个 expert 的 tensor
    expert_keys = [k for k in tensor_index_resize.keys() if 'expert' in k and 'gate_proj' in k]
    
    if not expert_keys:
        print("未找到 expert tensor")
        return
    
    test_key = expert_keys[0]
    print(f"\n测试 tensor: {test_key}")
    
    info = tensor_index_resize[test_key]
    print(f"原始信息: {info}")
    print(f"类型: {type(info)}")
    print(f"长度: {len(info)}")
    
    if len(info) >= 5:
        offset, size, shape, strides, dtype = info[0], info[1], info[2], info[3], info[4]
        print(f"\n解析结果:")
        print(f"  offset: {offset} (type: {type(offset)})")
        print(f"  size: {size} (type: {type(size)})")
        print(f"  shape: {shape} (type: {type(shape)})")
        print(f"  strides: {strides} (type: {type(strides)})")
        print(f"  dtype: {dtype} (type: {type(dtype)})")
        
        # 检查 dtype 格式
        print(f"\nDtype 检查:")
        if dtype == "torch.bfloat16":
            print(f"  ✓ dtype 格式正确: {dtype}")
        elif dtype == "bfloat16":
            print(f"  ⚠ dtype 缺少 'torch.' 前缀: {dtype}")
            print(f"  应该改为: torch.bfloat16")
        else:
            print(f"  ✗ dtype 格式不正确: {dtype}")
            print(f"  期望: torch.bfloat16")
    else:
        print(f"  ✗ 信息不完整，长度只有 {len(info)}，需要 5 个元素")
    
    # 检查多个 expert tensors
    print(f"\n" + "="*60)
    print("检查多个 expert tensors 的 dtype")
    print("="*60)
    
    expert_keys_sample = expert_keys[:5]
    dtype_set = set()
    
    for key in expert_keys_sample:
        info = tensor_index_resize[key]
        if len(info) >= 5:
            dtype = info[4]
            dtype_set.add(dtype)
            print(f"  {key}: dtype = {dtype}")
    
    print(f"\n发现的 dtype 类型: {dtype_set}")
    
    if len(dtype_set) == 1:
        print(f"  ✓ 所有 tensor 的 dtype 一致: {list(dtype_set)[0]}")
    else:
        print(f"  ✗ 发现多种 dtype: {dtype_set}")
    
    # 检查 C++ 期望的格式
    print(f"\n" + "="*60)
    print("C++ 期望的 dtype 格式")
    print("="*60)
    
    cpp_expected = {
        "torch.float16": "torch::kFloat16",
        "torch.float32": "torch::kFloat32",
        "torch.float64": "torch::kFloat64",
        "torch.bfloat16": "torch::kBFloat16",
        "torch.int16": "torch::kInt16",
        "torch.int32": "torch::kInt32",
        "torch.int64": "torch::kInt64",
        "torch.uint8": "torch::kUInt8",
        "torch.int8": "torch::kInt8",
    }
    
    print("C++ stringToScalarType 支持的格式:")
    for py_format, cpp_type in cpp_expected.items():
        print(f"  {py_format:20} -> {cpp_type}")
    
    # 检查实际 dtype 是否匹配
    if dtype_set:
        actual_dtype = list(dtype_set)[0]
        if actual_dtype in cpp_expected:
            print(f"\n✓ 实际 dtype '{actual_dtype}' 在 C++ 支持列表中")
        else:
            print(f"\n✗ 实际 dtype '{actual_dtype}' 不在 C++ 支持列表中")
            print(f"  这可能是问题所在！")
    
    return tensor_index_resize


def check_cpp_loading():
    """检查 C++ 加载时的 dtype 处理"""
    
    print(f"\n" + "="*60)
    print("C++ 代码中的 dtype 处理")
    print("="*60)
    
    print("""
在 checkpoint.cpp 中：

1. stringToScalarType 函数 (line 106-120):
   - 支持 "torch.bfloat16" -> torch::kBFloat16
   - 如果找不到匹配，会抛出异常

2. RestoreTensorsFromSharedMemoryNames (line 453-568):
   - 从 tensor_metadata 读取: auto [offset, size, shape, strides, dtype_str] = info;
   - 转换 dtype: at::ScalarType dtype = stringToScalarType(dtype_str);
   - 创建 tensor: torch::from_blob(..., torch::TensorOptions().device(torch::kCPU).dtype(dtype));

3. 可能的问题点:
   a) dtype_str 格式不正确（缺少 "torch." 前缀）
   b) dtype_str 在 JSON 中缺失或为 None
   c) pybind11 转换时出现问题
   d) 共享内存中的数据本身就是 float32
    """)


def check_config():
    """检查 config.json 中的 torch_dtype"""
    
    print(f"\n" + "="*60)
    print("检查 config.json")
    print("="*60)
    
    config_path = "/mnt/zhengcf3/models/sllm_models/deepseek-moe-16b-base-bfloat16/config.json"
    
    if os.path.exists(config_path):
        config = load_json(config_path)
        torch_dtype = config.get("torch_dtype", "NOT FOUND")
        print(f"config.torch_dtype: {torch_dtype}")
        print(f"类型: {type(torch_dtype)}")
    else:
        print(f"Config 文件不存在: {config_path}")


if __name__ == "__main__":
    print("Dtype 问题诊断工具")
    print("="*60)
    
    # 1. 检查 tensor_index_resize.json
    tensor_index_resize = check_tensor_index_resize_dtype()
    
    # 2. 检查 config.json
    check_config()
    
    # 3. 说明 C++ 处理
    check_cpp_loading()
    
    print(f"\n" + "="*60)
    print("诊断总结")
    print("="*60)
    print("""
可能的问题原因：

1. ✅ tensor_index_resize.json 中 dtype 格式正确
   - 应该是 "torch.bfloat16" 格式
   - 如果只是 "bfloat16"，C++ 无法识别

2. ⚠️ 共享内存中的数据本身是 float32
   - 即使 dtype 标记为 bfloat16，如果共享内存中的数据是 float32
   - torch::from_blob 会按照 dtype 解释数据，但数据本身可能不对

3. ⚠️ pybind11 转换问题
   - Python dict -> C++ std::unordered_map 转换时
   - tuple 的最后一个元素（dtype）可能丢失或格式错误

4. ⚠️ 模型创建时的问题
   - create_empty_model 可能使用默认 dtype (float32)
   - 需要检查 DeepseekModule.create_empty_model 是否正确设置 dtype

建议检查：
1. 打印 C++ 中实际接收到的 dtype_str
2. 检查共享内存中数据的实际格式
3. 验证 create_empty_model 是否正确使用 config.torch_dtype
    """)

