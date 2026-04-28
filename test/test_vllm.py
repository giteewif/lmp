from vllm import LLM, SamplingParams

import os
import time

model_name = "gemma4-26B-A4B"
storage_path = "/mnt/zhengcf3/models/"
model_path = os.path.join(storage_path, model_name)

max_model_len = 256
llm = LLM(
    model=model_path, dtype="bfloat16",
    tensor_parallel_size=4,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    max_model_len=max_model_len,  # 设置最大输入长度为256
    # enforce_eager=True,
)

prompts = [
    "Hello, my name is John and I am excited to share with you a fascinating story about artificial intelligence and its rapid development in recent years. " * 5 + "This is a test prompt to reach 256 characters for batch processing.",
    "The president of the United States is a position of great responsibility, requiring leadership, vision, and the ability to make difficult decisions that affect millions of lives both at home and abroad. " * 4 + "This prompt explores political leadership and governance.",
    "The capital of France is Paris, a city renowned for its art, culture, cuisine, and iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. " * 4 + "Paris is often called the City of Light.",
    "The future of AI is filled with endless possibilities and potential, from revolutionizing healthcare and education to transforming transportation and communication systems. " * 4 + "AI continues to evolve at an unprecedented pace.",
]

# 确保每个 prompt 不超过 max_model_len 字符
max_model_len = 256
prompts = [prompt[:max_model_len] for prompt in prompts]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

t0 = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)

t1 = time.perf_counter()
print(f"generate takes {t1 - t0} seconds")



# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")