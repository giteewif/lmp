## qc
source /mnt/zhengcf3/env/lmp/bin/activate

/mnt/zhengcf3/nvidia/nsight-systems/2025.6.1/bin/nsys profile --force-overwrite true -o report1.nsys-rep python generate.py > generate_multi.log

/mnt/zhengcf3/nvidia/nsight-systems/2025.6.1/bin/nsys profile --force-overwrite true -o report2.nsys-rep python test_normal.py > test_normal.log

/mnt/zhengcf3/nvidia/nsight-systems/2025.6.1/bin/nsys profile --force-overwrite true -o report3.nsys-rep python test_sllm_store.py > test_sllm_store_b2.log 

/mnt/zhengcf3/nvidia/nsight-systems/2025.6.1/bin/nsys profile --force-overwrite true -o report4_testmp.nsys-rep python test_init_meta_manager_mp_shared.py > test_init_meta_manager_mp_shared.log

/mnt/zhengcf3/nvidia/nsight-systems/2025.6.1/bin/nsys profile --force-overwrite true -o report5.nsys-rep python test_device_mp.py > test_device_mp.log

/mnt/zhengcf3/nvidia/nsight-systems/2025.6.1/bin/nsys profile --force-overwrite true -o report6.nsys-rep python test_cpu_mp.py > test_cpu_mp.log
## protoc

cd /mnt/zhengcf3/lmp/src/sllm_store && python -m grpc_tools.protoc --proto_path=proto --python_out=sllm_store --grpc_python_out=sllm_store proto/storage.proto  

## sllm_store
pip install -e .
python setup.py build_ext --inplace

Deepseek
sllm-store start --storage-path /mnt/zhengcf3/models/sllm_models --num-thread 8 --mem-pool-size 32GB --use-shared-memory True --chunk-size 1056MB

QWEN30B
sllm-store start --storage-path /mnt/zhengcf3/models/sllm_models --num-thread 8 --mem-pool-size 64GB --use-shared-memory True --chunk-size 1152MB

Gemma4
sllm-store start --storage-path /mnt/zhengcf3/models/sllm_models --num-thread 8 --mem-pool-size 64GB --use-shared-memory True --chunk-size 1452MB

numactl --cpunodebind=0 --membind=0 sllm-store start --storage-path /mnt/zhengcf3/models/sllm_models --num-thread 8 --mem-pool-size 64GB --use-shared-memory True --chunk-size 1GB

Gemma4 vllm
sllm-store start --storage-path /mnt/zhengcf3/models/vllm_sllm_models --num-thread 8 --mem-pool-size 60GB --use-shared-memory True --chunk-size 1452MB --port 8074


QWEN-1.5
sllm-store start --storage-path /mnt/zhengcf3/models/sllm_models --num-thread 8 --mem-pool-size 32GB --use-shared-memory True --chunk-size 990MB --port 8074

sllm-store start --storage-path /mnt/zhengcf3/models/sllm_models --num-thread 8 --mem-pool-size 95GB --use-shared-memory True --chunk-size 2688MB

python /mnt/zhengcf3/lmp/examples/test_sllm_store.py

python /mnt/zhengcf3/lpllm/lpllm/resize_index.py /mnt/zhengcf3/models/sllm_models/Qwen3-30B-A3B/tensor_index.json

## install lpllm
conda create -n lpllm python=3.10 -y & apt-get install gpustat -y

export PATH=/mnt/huwf5/conda-envs/sida39/bin:$PATH

## gpustat

watch -n 1 gpustat

# resize index file
python /mnt/zhengcf3/lpllm/lpllm/resize_index.py /mnt/zhengcf3/models/sllm_models/Mixtral-8x7B/tensor_index.json > /mnt/zhengcf3/models/sllm_models/Mixtral-8x7B/chunk_size



python setup.py build_ext --inplace
python -m pip install -e . --no-build-isolation

pip install "git+https://github.com/huggingface/transformers.git@v5.5.4"
pip install -i https://pypi.org/simple --upgrade "transformers==5.5.3"


source /mnt/zhengcf3/lmp_env/vllm/bin/activate


### conda

source /root/miniconda3/etc/profile.d/conda.sh
conda create -y -p /mnt/zhengcf3/conda_envs/sllm_vllm python=3.10
conda activate /mnt/zhengcf3/conda_envs/sllm_vllm

#### 使用大盘上的独立 conda base（/mnt/zhengcf3/miniconda3）

# 只需要 source 一次：会临时切换 HOME，避免读取 /root/.condarc 导致 env/pkgs 指回系统盘
source /mnt/zhengcf3/use_mnt_conda.sh

# 创建命名环境（会落到 /mnt/zhengcf3/miniconda3/envs/<name>）
conda create -y -n myenv python=3.10
conda activate myenv

# 或者强制把环境放到你指定的大盘目录（前缀环境）
conda create -y -p /mnt/zhengcf3/conda_envs/myenv python=3.10
conda activate /mnt/zhengcf3/conda_envs/myenv

/mnt/zhengcf3/miniconda3/bin/conda create -y -p /mnt/zhengcf3/conda_envs/myenv -c conda-forge python=3.10
/mnt/zhengcf3/miniconda3/bin/conda run -p /mnt/zhengcf3/conda_envs/myenv python -V
/mnt/zhengcf3/miniconda3/bin/conda run -p /mnt/zhengcf3/conda_envs/myenv pip install -U pip
eval "$(/mnt/zhengcf3/miniconda3/bin/conda shell.bash hook)"
conda activate /mnt/zhengcf3/conda_envs/myenv

pip install -i https://pypi.org/simple datasets


source /mnt/zhengcf3/lmp_env/fslmp/bin/activate


conda create -n lpllm python=3.10 -y


## split experts
python3 /mnt/zhengcf3/lmp/scripts/split_gemma4_tensor_index_experts.py \
  --index-in /mnt/zhengcf3/models/sllm_models/gemma4-26B-A4B/tensor_index.json \
  --config /mnt/zhengcf3/models/sllm_models/gemma4-26B-A4B/config.json \
  --index-out /mnt/zhengcf3/models/sllm_models/gemma4-26B-A4B/tensor_index_resize_per_expert.json

## vllm 
version 0.19.1 patch for serverlessllm 0.8.0


### Serverlessllm usage with vllm 

source /mnt/zhengcf3/lmp_env/lmp/bin/activate

website `https://serverlessllm.github.io/docs/store/quickstart`

save load
`
python3 /mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py --model-name gemma4-26B-A4B --local-model-path /mnt/zhengcf3/models/gemma4-26B-A4B --storage-path /mnt/zhengcf3/models/vllm_sllm_models --tensor-parallel-size 4
`

本机为 3×GPU 时：`tensor_parallel_size=4` 不可用；`tensor_parallel_size=3` 与 ERNIE / Qwen 的 head 数不整除。已用 `TP=2`、`--dtype bfloat16`、`--gpu-memory-utilization 0.95`，并在运行前 `export VLLM_USE_FLASHINFER_MOE_FP16=1 VLLM_FLASHINFER_MOE_BACKEND=latency` 跑通下列四个权重（`save_vllm_model.py` 内已设 `disable_custom_all_reduce=True`）。一键顺序导出：`bash /mnt/zhengcf3/lmp/scripts/run_save_vllm_models_four.sh`（日志在 `lmp/logs/save_vllm_models/`）。

`
python3 /mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py --model-name Qwen3.5-35B --local-model-path /mnt/zhengcf3/models/Qwen3.5-35B --storage-path /mnt/zhengcf3/models/vllm_sllm_models --tensor-parallel-size 4

python3 /mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py --model-name Qwen3-30B-A3B --local-model-path /mnt/zhengcf3/models/Qwen3-30B-A3B --storage-path /mnt/zhengcf3/models/vllm_sllm_models --tensor-parallel-size 4

python3 /mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py --model-name ERNIE-4.5-VL-28B-A3B-Thinking --local-model-path /mnt/zhengcf3/models/ERNIE-4.5-VL-28B-A3B-Thinking --storage-path /mnt/zhengcf3/models/vllm_sllm_models --tensor-parallel-size 4

python3 /mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py --model-name ERNIE-4.5-21B-A3B-Thinking --local-model-path /mnt/zhengcf3/models/ERNIE-4.5-21B-A3B-Thinking --storage-path /mnt/zhengcf3/models/vllm_sllm_models --tensor-parallel-size 4

python3 /mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py --model-name Qwen1.5-MoE-A2.7B --local-model-path /mnt/zhengcf3/models/Qwen1.5-MoE-A2.7B --storage-path /mnt/zhengcf3/models/vllm_sllm_models --tensor-parallel-size 4

python3 /mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py --model-name gpt-oss-20b --local-model-path /mnt/zhengcf3/models/gpt-oss-20b --storage-path /mnt/zhengcf3/models/vllm_sllm_models --tensor-parallel-size 4



python3 /mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py --model-name deepseek-moe-16b-base --local-model-path /mnt/zhengcf3/models/deepseek-moe-16b-base --storage-path /mnt/zhengcf3/models/vllm_sllm_models --tensor-parallel-size 4

python3 /mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py --model-name DeepSeek-V2-Lite --local-model-path /mnt/zhengcf3/models/DeepSeek-V2-Lite --storage-path /mnt/zhengcf3/models/vllm_sllm_models --tensor-parallel-size 4
`