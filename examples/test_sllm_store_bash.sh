#!/bin/bash
# python test_sllm_store.py > test_sllm_store_64.log
for i in {1..3}; do
#     # python test_sllm_store.py > test_sllm_store_4g_nometa_${i}.log
#     # python test_sllm_store.py > test_sllm_store_3g_${i}.log
#     # python test_sllm_store.py > test_sllm_store_3g_nometa_${i}.log
    python test_sllm_store.py > test_sllm_store_4g_${i}_128.log
done