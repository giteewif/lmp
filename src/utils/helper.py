import json
import torch
import time
from datasets import load_from_disk

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
    test_dataset = local_dataset['train']

    # 从数据集中获取 batch_size 个样本，并用数据集中的文本补充到指定长度
    batch_input_ids = []
    dataset_idx = 220  # 从数据集的某个位置开始取样本
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