### tensor_index 分配
1. 请你写个python 函数根据现有GPU设备的内存情况和数量，为 tensor_index json中参数 在设备上指定分配
2. tensor_index json 格式一般如下
```json
    "model.language_model.layers.3.mlp.down_proj.weight": [
        4496498536,
        11894784,
        [
            2816,
            2112
        ],
        [
            2112,
            1
        ],
        "torch.bfloat16"
    ],
    "model.language_model.layers.3.mlp.up_proj.weight": [
        4508393320,
        11894784,
        [
            2112,
            2816
        ],
        [
            2816,
            1
        ],
        "torch.bfloat16"
    ],
```
3. 同个 layer下的专家要求均匀分配到设备上，确保同个专家的参数分配到一个设备中
4. 函数需给出，每个tensor的 设备分配情况，以map的形式，可通过名称查找到设备
5. 使用 pynvml 读gpu设备状态，不够显存则暂时报错
6. self_attn参数则只放在一个GPU上，除专家外的参数，其他参数需一并放在第一个设备内存中，只有专家的参数需要均匀，首先确保GPU显存使用均匀
7. 首先GPU显存使用尽量动态平衡，再同layer 专家尽量均匀


### restore_tensors_from_shared_memory_names
1. 更改 host buffer，使用 共享文件的形式
2. 支持将host buffer 共享给其他进程使用，使用 restore_tensors_from_shared_memory_names




# 最小改动，最大收益（消除 D2H + Python 路由循环）
LMP_VLLM_MOE=1 LMP_MOE_DECODE_EXPERT_BACKEND=gpu python3 ...

# 加 MoE CUDA Graph（每设备首次 forward 自动捕获，之后 replay）
LMP_VLLM_MOE=1 LMP_VLLM_MOE_CG=1 LMP_MOE_DECODE_EXPERT_BACKEND=gpu python3 ...

# 完整优化：flash_attn_with_kvcache + vLLM MoE + 静态 KV
LMP_VLLM_MOE=1 LMP_VLLM_MOE_CG=1 LMP_ATTN_CG=1 LMP_STATIC_KV_MAX_SEQ=4096 \
LMP_MOE_DECODE_EXPERT_BACKEND=gpu python3 ...