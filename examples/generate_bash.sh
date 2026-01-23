#!/bin/bash
# python generate.py > generate_multi_4g.log
for i in {1..3}; do
#     # python generate.py > generate_multi_3g_${i}.log
#     # python generate.py > generate_multi_1g_${i}.log
#     # python generate.py > generate_multi_3g_${i}.log
    python generate.py > generate_multi_4g_${i}_128.log
    # python generate.py > generate_multi_4g_${i}_nocpubmm.log
done