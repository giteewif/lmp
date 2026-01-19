#!/bin/bash
# python test_sllm_store.py 
for i in {1..3}; do
    # python test_sllm_store.py > test_sllm_store_3g_nometa_${i}.log
    python test_sllm_store.py > test_sllm_store_3g_${i}.log
    # python generate.py > generate_multi_2g_${i}.log
    # python generate.py > generate_multi_3g_${i}.log
    # python generate.py > generate_multi_4g_${i}.log
done