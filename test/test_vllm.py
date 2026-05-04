"""
vLLM 测 prefill / decode：与 ``examples/test_sllm_store.py`` 相同思路（墙钟 + cuda 同步），
不用 ``RequestOutput.metrics``。

1) 第一次 ``max_tokens=1``：含 JIT 等预热
2) 第二次 ``max_tokens=1``：记为 Prefill 代理时长（整次 generate：prefill + 首 token）
3) 第三次 ``max_tokens=max_new_tokens``：总长减去第二次，按 ``(t - t_prefill) / (max_new_tokens - 1)``
   得到平均单步 decode（与同文件里 ``decode_single_time`` 公式一致）
"""
from __future__ import annotations

from typing import Any, List, Tuple

from vllm import LLM, SamplingParams

import os
import time

import torch

model_name = "gemma4-26B-A4B"
storage_path = "/mnt/zhengcf3/models/"
model_path = os.path.join(storage_path, model_name)

max_model_len = 256
max_new_tokens = int(os.environ.get("VLLM_TEST_MAX_NEW_TOKENS", "32"))

llm = LLM(
    model=model_path,
    dtype="bfloat16",
    tensor_parallel_size=4,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    max_model_len=max_model_len,
)

prompts = [
    "Hello, my name is John and I am excited to share with you a fascinating story about artificial intelligence and its rapid development in recent years. " * 5 + "This is a test prompt to reach 256 characters for batch processing.",
    "The president of the United States is a position of great responsibility, requiring leadership, vision, and the ability to make difficult decisions that affect millions of lives both at home and abroad. " * 4 + "This prompt explores political leadership and governance.",
    "The capital of France is Paris, a city renowned for its art, culture, cuisine, and iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. " * 4 + "Paris is often called the City of Light.",
    "The future of AI is filled with endless possibilities and potential, from revolutionizing healthcare and education to transforming transportation and communication systems. " * 4 + "AI continues to evolve at an unprecedented pace.",
]

prompts2 = [
    "In the age of artificial intelligence, we witness daily advancements that change the way we interact with technology and one another. " * 5 + "This is a test prompt to reach 256 characters for batch processing.",
    "Leadership, diplomacy, and decision making are the hallmarks of any head of state, especially in complex global environments where choices impact millions. " * 4 + "This prompt explores political leadership and governance.",
    "Known for its remarkable history and beauty, Paris stands as a symbol of French excellence in art, cuisine, architecture, and notable monuments such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. " * 3 + "Paris is often called the City of Light.",
    "Technological innovation is paving the way for the next generation of discoveries in health, mobility, and communication, leading to a future defined by AI and its boundless capacities." * 3 + "AI continues to evolve at an unprecedented pace.",
]

prompts = [prompt[:max_model_len] for prompt in prompts]
prompts2 = [prompt[:max_model_len] for prompt in prompts2]

sampling_params_1 = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)
sampling_params_n = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=max_new_tokens
)


def sync_all_cuda():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=f"cuda:{i}")


def run_timed_generate(
    label: str, params: SamplingParams, prompts: List[str]
) -> Tuple[float, List[Any]]:
    sync_all_cuda()
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=params)
    sync_all_cuda()
    elapsed = time.time() - t0
    print(f"{label}: {elapsed:.4f}s")
    return elapsed, outputs


def main():

    sep = "=" * 60
    print(sep)
    print("First generate (with warmup overhead), max_tokens=1:")
    print(sep)
    first_time, _ = run_timed_generate("First generate time", sampling_params_n, prompts)

    print(sep)
    print("Prefill generate (proxy: full batch max_tokens=1):")
    print(sep)
    second_time, _ = run_timed_generate("Prefill generate", sampling_params_1, prompts)

    print(sep)
    print("Prefill generate (proxy: full batch max_tokens=1):")
    print(sep)
    second_time, _ = run_timed_generate("Prefill generate", sampling_params_1, prompts)

    print(sep)
    print(f"{max_new_tokens} output generate (prefill + decode):")
    print(sep)
    second_n_time, outputs = run_timed_generate(
        f"{max_new_tokens} output generate time", sampling_params_n, prompts2
    )

    if max_new_tokens > 1:
        decode_single_time = (second_n_time - second_time) / (max_new_tokens - 1)
        print(f"decode single time (avg per step): {decode_single_time:.4f}s")
    else:
        print("max_new_tokens<=1, skip decode_single_time")

    if second_time > 0:
        speedup = first_time / second_time
        print(f"\nSpeedup (first / prefill-proxy): {speedup:.2f}x")
        print("原因与 test_sllm_store 类似：首次含 CUDA kernel JIT、引擎初始化等。")

    print(sep)
    for output in outputs:
        co = output.outputs[0]
        print(f"Prompt: {output.prompt[:80]!r}..., text: {co.text!r}")


if __name__ == "__main__":
    main()
