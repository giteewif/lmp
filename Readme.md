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

QWEN-1.5
sllm-store start --storage-path /mnt/zhengcf3/models/sllm_models --num-thread 8 --mem-pool-size 32GB --use-shared-memory True --chunk-size 990MB

sllm-store start --storage-path /mnt/zhengcf3/models/sllm_models --num-thread 8 --mem-pool-size 95GB --use-shared-memory True --chunk-size 2688MB

python /mnt/zhengcf3/lmp/examples/test_sllm_store.py

## install lpllm
conda create -n lpllm python=3.10 -y & apt-get install gpustat -y

export PATH=/mnt/huwf5/conda-envs/sida39/bin:$PATH

## gpustat

watch -n 1 gpustat

# resize index file
python /mnt/zhengcf3/lpllm/lpllm/resize_index.py /mnt/zhengcf3/models/sllm_models/Mixtral-8x7B/tensor_index.json > /mnt/zhengcf3/models/sllm_models/Mixtral-8x7B/chunk_size