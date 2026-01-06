import json
import torch
import time
from typing import Dict
from datasets import load_from_disk
import pynvml
pynvml.nvmlInit()

from utils.logger import init_logger
logger = init_logger(__name__)
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def print_layer_parameters(layer):
    print("=" * 60)
    for name, param in layer.named_parameters():
        print(f"  {name}: {param.device} (shape: {param.shape}) (dtype: {param.dtype})")
    print("=" * 60)

def check_nan_inf(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return True
    return False

def get_expert_device_distribution(layer) -> Dict[int, str]:
        """
        获取 layer 中每个 expert 的设备分布
        
        Returns:
            Dict[int, str]: {expert_id: device} 映射，device 可能是 'cuda:X', 'meta', 'cpu' 等
        """
        expert_device_map = {}
        
        
        experts = layer.mlp.experts
        num_experts = len(experts)
        
        for expert_id in range(num_experts):
            expert = experts[expert_id]
            # 检查 expert 的第一个参数来确定设备
            # 通常检查 gate_proj.weight 或 up_proj.weight
            device = None
            for name, param in expert.named_parameters():
                device = str(param.device)
                break  # 只检查第一个参数即可
            
            if device is None:
                # 如果没有参数，检查 buffers
                for name, buffer in expert.named_buffers():
                    device = str(buffer.device)
                    break
            
            expert_device_map[expert_id] = device if device else "unknown"
        
        return expert_device_map

def calculate_expert_memory_size(mlpm, tensor_index_resize_json, layer_idx: int, expert_idx: int) -> int:
    """
    计算单个 expert 的显存大小（字节）
    
    Args:
        mlpm: MLPModuleWrapper 实例
        tensor_index_resize_json: tensor 索引 JSON 字典
        layer_idx: 层索引
        expert_idx: expert 索引
        
    Returns:
        int: expert 的显存大小（字节）
    """
    # 获取该 expert 的所有权重矩阵名称
    expert_names = mlpm.get_experts_names(layer_idx=layer_idx, expert_idx_list=[expert_idx])
    
    total_size = 0
    for name in expert_names:
        if name in tensor_index_resize_json:
            # tensor_index_resize_json 格式: (offset, size, shape, stride, dtype)
            _, size, _, _, _ = tensor_index_resize_json[name]
            total_size += size
        else:
            import sys
            print(f"WARNING: Tensor {name} not found in tensor_index_resize_json, cannot calculate size", file=sys.stderr)
    
    return total_size

def filter_experts_by_memory(
    mlpm, 
    tensor_index_resize_json, 
    config,
    device1: int,
    layer_cpu_experts_map: Dict[int, list]
) -> Dict[int, list]:
    """
    根据 GPU 剩余显存过滤 expert，如果显存不足则按比例选择部分 expert
    每个层的 cpu_expert_list 按比例选择
    
    Args:
        mlpm: MLPModuleWrapper 实例
        tensor_index_resize_json: tensor 索引 JSON 字典
        config: 模型配置
        device1: GPU 设备索引
        layer_cpu_experts_map: {layer_idx: [expert_id, ...]} 需要加载的 CPU expert 映射
        
    Returns:
        Dict[int, list]: 过滤后的 expert 映射
    """
    # 计算所有 CPU expert 的总显存需求
    total_required_memory = 0
    expert_memory_map = {}  # {(layer_idx, expert_idx): memory_size}
    layer_memory_map = {}  # {layer_idx: total_memory_for_layer}
    
    for layer_idx, expert_list in layer_cpu_experts_map.items():
        layer_total = 0
        for expert_idx in expert_list:
            memory_size = calculate_expert_memory_size(mlpm, tensor_index_resize_json, layer_idx, expert_idx)
            expert_memory_map[(layer_idx, expert_idx)] = memory_size
            layer_total += memory_size
        layer_memory_map[layer_idx] = layer_total
        total_required_memory += layer_total
    
    import sys
    print(f"Total required memory for all CPU experts: {total_required_memory / (1024**3):.2f} GB", file=sys.stderr)
    
    # 使用 NVML 获取 GPU 显存信息（更准确）
    try:
       
        device_idx = device1
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total  # 总显存(字节)
        used_memory = memory_info.used   # 已用显存(字节)
        free_memory = memory_info.free   # 空闲显存(字节)
        
        print(f"GPU {device1} memory status (NVML):", file=sys.stderr)
        print(f"  Total: {total_memory / (1024**3):.2f} GB", file=sys.stderr)
        print(f"  Used: {used_memory / (1024**3):.2f} GB", file=sys.stderr)
        print(f"  Free: {free_memory / (1024**3):.2f} GB", file=sys.stderr)
        print(f"  Required: {total_required_memory / (1024**3):.2f} GB", file=sys.stderr)
    except Exception as e:
        # 如果 NVML 不可用，回退到 PyTorch 方式
        raise ValueError(f"NVML unavailable or failed: {e}, falling back to PyTorch memory API")
    
    # 如果显存足够，返回原始映射
    if total_required_memory <= free_memory:
        print("GPU memory is sufficient, loading all CPU experts.", file=sys.stderr)
        return layer_cpu_experts_map
    
    # 显存不足，按比例选择 expert
    print(f"WARNING: GPU memory is insufficient! Required: {total_required_memory / (1024**3):.2f} GB, "
          f"Available: {free_memory / (1024**3):.2f} GB", file=sys.stderr)
    
    # 计算每层可以使用的显存比例
    # 至少保留 1GB 显存
    reserved_buffer = 2 * 1024**3  # 1GB 的字节数
    available_memory = max(0, free_memory - reserved_buffer)  # 可用显存（至少保留1GB）
    
    if available_memory <= 0:
        print(f"WARNING: Available memory after reserving 1GB is {available_memory / (1024**3):.2f} GB, cannot load any experts.", file=sys.stderr)
        return {}
    
    memory_ratio = available_memory / total_required_memory
    actual_ratio = memory_ratio
    
    print(f"Memory reservation: Reserved {reserved_buffer / (1024**3):.2f} GB buffer, "
          f"available for loading: {available_memory / (1024**3):.2f} GB", file=sys.stderr)
    
    print(f"Will load {actual_ratio * 100:.1f}% of experts per layer based on available memory.", file=sys.stderr)
    
    # 按层从后往前选择 expert（优先加载后面的层），每层按比例选择
    filtered_map = {}
    accumulated_memory = 0
    
    # 从最后一层往前遍历
    for layer_idx in range(config.num_hidden_layers - 1, -1, -1):
        if layer_idx not in layer_cpu_experts_map:
            continue
        
        expert_list = layer_cpu_experts_map[layer_idx]
        layer_total_memory = layer_memory_map[layer_idx]
        layer_target_memory = int(layer_total_memory * actual_ratio)
        
        # 按 expert 索引排序
        sorted_experts = sorted(expert_list)
        filtered_experts = []
        layer_accumulated = 0
        
        # 按比例选择该层的 expert
        for expert_idx in sorted_experts:
            memory_size = expert_memory_map[(layer_idx, expert_idx)]
            
            if layer_accumulated + memory_size <= layer_target_memory:
                filtered_experts.append(expert_idx)
                layer_accumulated += memory_size
            else:
                # 如果加上这个 expert 会超限，跳过
                pass
        
        if filtered_experts:
            filtered_map[layer_idx] = filtered_experts
            accumulated_memory += layer_accumulated
            print(f"Layer {layer_idx}: Selected {len(filtered_experts)}/{len(expert_list)} experts "
                  f"({len(filtered_experts)/len(expert_list)*100:.1f}%, "
                  f"memory: {layer_accumulated / (1024**3):.2f} GB / {layer_total_memory / (1024**3):.2f} GB)", file=sys.stderr)
    
    print(f"Filtered experts total memory: {accumulated_memory / (1024**3):.2f} GB "
          f"({accumulated_memory / total_required_memory * 100:.1f}% of original)", file=sys.stderr)
    
    return filtered_map

def process_logits_efficiently(logits, temperature=1.0, top_p=0.9, do_sample=True, device=None):
    """
    Efficiently process logits for token generation with proper memory management.
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        temperature: Temperature for scaling logits
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to use sampling or greedy decoding
        device: Device for memory monitoring
    
    Returns:
        next_tokens: Tensor of shape (batch_size,)
    """    
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    

    
    if do_sample:
        # Apply top-p filtering if needed
        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            
            # Compute softmax probabilities
            probs = torch.softmax(sorted_logits, dim=-1)
            
            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(probs, dim=-1)
            
            # Find cutoff point
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Apply mask
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Clean up
            del sorted_logits, sorted_indices, probs, cumulative_probs, sorted_indices_to_remove, indices_to_remove
        
        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        
        # Debug: 检查概率是否有效
        if torch.isnan(probs).any():
            import sys
            print(f"ERROR: probs contains NaN after softmax!", file=sys.stderr)
            print(f"  NaN count: {torch.isnan(probs).sum().item()}", file=sys.stderr)
            print(f"  Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}", file=sys.stderr)
            print(f"  Logits contains NaN: {torch.isnan(logits).any().item()}", file=sys.stderr)
            print(f"  Logits contains Inf: {torch.isinf(logits).any().item()}", file=sys.stderr)
            # 修复: 将 nan 替换为 0
            probs = torch.where(torch.isnan(probs), torch.tensor(0.0, device=probs.device, dtype=probs.dtype), probs)
            print(f"  Fixed NaN values in probs", file=sys.stderr)
        
        if torch.isinf(probs).any():
            import sys
            print(f"ERROR: probs contains Inf after softmax!", file=sys.stderr)
            print(f"  Inf count: {torch.isinf(probs).sum().item()}", file=sys.stderr)
            # 修复: 将 inf 替换为 0
            probs = torch.where(torch.isinf(probs), torch.tensor(0.0, device=probs.device, dtype=probs.dtype), probs)
            print(f"  Fixed Inf values in probs", file=sys.stderr)
        
        # Ensure valid probabilities (non-negative)
        if (probs < 0).any():
            import sys
            print(f"WARNING: probs contains negative values, clamping to 0", file=sys.stderr)
            probs = torch.clamp(probs, min=0.0)
        
        # Renormalize to ensure sum = 1
        probs_sum = probs.sum(dim=-1, keepdim=True)
        if (probs_sum <= 0).any():
            import sys
            print(f"ERROR: probs sum is <= 0 for some samples!", file=sys.stderr)
            print(f"  Invalid sum count: {(probs_sum <= 0).sum().item()}", file=sys.stderr)
            # 修复: 如果 sum <= 0，使用均匀分布
            vocab_size = probs.shape[-1]
            uniform_probs = torch.ones_like(probs) / vocab_size
            probs = torch.where(probs_sum <= 0, uniform_probs, probs)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            print(f"  Fixed invalid probs with uniform distribution", file=sys.stderr)
        
        probs = probs / probs_sum
        
        # Final check before multinomial
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            import sys
            print(f"ERROR: probs still invalid before multinomial!", file=sys.stderr)
            print(f"  NaN: {torch.isnan(probs).any().item()}, Inf: {torch.isinf(probs).any().item()}, <0: {(probs < 0).any().item()}", file=sys.stderr)
            # 使用均匀分布作为后备
            vocab_size = probs.shape[-1]
            probs = torch.ones_like(probs) / vocab_size
            print(f"  Using uniform distribution as fallback", file=sys.stderr)
        
        # Ensure probabilities sum to 1 (with small epsilon tolerance)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Sample
        torch.cuda.synchronize()  # 确保所有操作完成
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        # Clean up
        del probs
    else:
        # Greedy decoding
        next_tokens = torch.argmax(logits, dim=-1)
    
    return next_tokens
    
# only for single device here, multiple device in sllm_store caculate_tensor_device_offsets, for layer and qkv
def calculate_device_offset(tensor_index, device_idx):
    device_offset = 0
    tensor_device_offsets = {}
    tensor_copy_chunks = {}
    tensor_copy_chunks[device_idx] = []
    tensor_device_offsets[device_idx] = {}
    tensor_record = {}
    single_device_offset = tensor_device_offsets[device_idx]
    single_copy_chunks_list = tensor_copy_chunks[device_idx]
    for name, (offset, size) in tensor_index.items():
        if (offset, size) in tensor_record:
            single_device_offset[name] = tensor_record[(offset, size)]
        else:
            tensor_record[(offset, size)] = device_offset
            single_device_offset[name] = device_offset
            single_copy_chunks_list.append(
                (offset, size, device_offset, 0)
            )
            device_offset += size
    return tensor_device_offsets, tensor_copy_chunks, device_offset


def generate_input_ids(tokenizer, batch_size, seq_len, tdevice):
    full = True
    if full:
        return generate_input_ids_full(tokenizer, batch_size, seq_len, tdevice)
    else:
        return generate_input_ids_pad(tokenizer, batch_size, seq_len, tdevice)

def generate_input_ids_full(tokenizer, batch_size, seq_len, tdevice):
    time_start_generate_input_ids = time.time()
        # 设置 pad_token（如果不存在，使用 eos_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 从本地数据集加载数据
    local_dataset = load_from_disk("/mnt/zhengcf3/models/data/wikitext2/wikitext-103-raw-v1")
    test_dataset = local_dataset['test']

    # 从数据集中获取 batch_size 个样本，并用数据集中的文本补充到指定长度
    batch_input_ids = []
    # 220
    dataset_idx = 16  # 从数据集的某个位置开始取样本
    for i in range(batch_size):
        # 收集足够的文本来达到 seq_len 长度
        combined_text = ""
        current_tokens = 0
        
        while current_tokens < seq_len:
            # 循环使用数据集中的样本
            sample_idx = dataset_idx % len(test_dataset)
            text = test_dataset[sample_idx]['text']
            dataset_idx += 1
            
            # 拼接文本
            if combined_text:
                combined_text += " " + text
            else:
                combined_text = text
            
            # 检查当前 token 数量
            encoded = tokenizer(combined_text, return_tensors="pt", truncation=False, add_special_tokens=False)
            current_tokens = encoded.input_ids.shape[1]
            
            # 如果已经达到或超过目标长度，跳出循环
            if current_tokens >= seq_len:
                break
        
        # 对拼接后的文本进行 tokenization 并截断到 seq_len
        encoded = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=seq_len, add_special_tokens=False)
        batch_input_ids.append(encoded.input_ids.squeeze(0))

    # 组合成 (batch_size, seq_len) 的张量
    input_ids = torch.stack(batch_input_ids).to(tdevice)
    print(f"generate input ids cost {time.time() - time_start_generate_input_ids} s")
    return input_ids


def generate_input_ids_pad(tokenizer, batch_size, seq_len, tdevice):
    # 设置 pad_token（如果不存在，使用 eos_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 从本地数据集加载数据
    local_dataset = load_from_disk("/mnt/zhengcf3/models/data/wikitext2/wikitext-103-raw-v1")
    test_dataset = local_dataset['test']

    # 从数据集中获取 batch_size 个样本
    batch_texts = []
    for i in range(batch_size):
        # 循环使用数据集中的样本
        sample_idx = i % len(test_dataset)
        text = test_dataset[sample_idx]['text']
        batch_texts.append(text)

    print(batch_texts[3])
    # 对每个文本进行 tokenization
    batch_input_ids = []
    for text in batch_texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len, padding='max_length')
        batch_input_ids.append(encoded.input_ids.squeeze(0))

    # 组合成 (batch_size, seq_len) 的张量
    input_ids = torch.stack(batch_input_ids).to(tdevice)
    return input_ids

def generate_input_ids_pad_new(tokenizer, batch_size, seq_len, tdevice):
    # 设置 pad_token（如果不存在，使用 eos_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 从本地数据集加载数据
    local_dataset = load_from_disk("/mnt/zhengcf3/models/data/wikitext2/wikitext-103-raw-v1")
    test_dataset = local_dataset['test']

    # 从数据集中获取 batch_size 个样本
    batch_texts = []
    for i in range(batch_size):
        # 循环使用数据集中的样本
        sample_idx = i % len(test_dataset)
        text = test_dataset[sample_idx]['text']
        batch_texts.append(text)

    # 对每个文本进行 tokenization
    batch_encoded = []
    for text in batch_texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len, padding='max_length')
        batch_encoded.append(encoded)

    # 组合成 (batch_size, seq_len) 的张量
    input_ids = torch.stack([e.input_ids.squeeze(0) for e in batch_encoded]).to(tdevice)
    attention_mask = torch.stack([e.attention_mask.squeeze(0) for e in batch_encoded]).to(tdevice) if hasattr(batch_encoded[0], 'attention_mask') and batch_encoded[0].attention_mask is not None else None
    
    # 返回字典格式，可以直接用于 **inputs
    result = {'input_ids': input_ids}
    if attention_mask is not None:
        result['attention_mask'] = attention_mask
    return result