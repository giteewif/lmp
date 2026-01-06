## qc
source /mnt/zhengcf3/env/lmp/bin/activate

/mnt/zhengcf3/nvidia/nsight-systems/2025.6.1/bin/nsys profile --force-overwrite true -o report1.nsys-rep python generate.py > generate_multi.log

/mnt/zhengcf3/nvidia/nsight-systems/2025.6.1/bin/nsys profile --force-overwrite true -o report2.nsys-rep python test_normal.py > test_normal.log
## protoc

cd /mnt/zhengcf3/lmp/src/sllm_store && python -m grpc_tools.protoc --proto_path=proto --python_out=sllm_store --grpc_python_out=sllm_store proto/storage.proto  

## sllm_store
pip install -e .
python setup.py build_ext --inplace

sllm-store start --storage-path /mnt/zhengcf3/models/sllm_models --num-thread 8 --mem-pool-size 32GB --use-shared-memory True --chunk-size 1GB

python /mnt/zhengcf3/lmp/examples/test_sllm_store.py

## install lpllm
conda create -n lpllm python=3.10 -y & apt-get install gpustat -y

export PATH=/mnt/huwf5/conda-envs/sida39/bin:$PATH

## gpustat

watch -n 1 gpustat