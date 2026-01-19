#!/bin/bash
python generate.py
for i in {1..3}; do
    python generate.py > generate_multi_3g_${i}.log
    # python generate.py > generate_multi_2g_${i}.log
    # python generate.py > generate_multi_3g_${i}.log
    # python generate.py > generate_multi_4g_${i}.log
done