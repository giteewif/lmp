here pin
INFO 05-06 15:31:12.006780.006780 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-06 15:31:12.587781.587781 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-06 15:31:13.051446.051446 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-06 15:31:13.051368.051368 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 1.045s
Warming up 4 GPU(s)...
GPU 0 warmed up
GPU 1 warmed up
GPU 2 warmed up
GPU 3 warmed up
GPU warmup completed
Warming up 4 GPU(s)...
GPU 0 warmed up
GPU 1 warmed up
GPU 2 warmed up
GPU 3 warmed up
GPU warmup completed
Warming up 4 GPU(s)...
GPU 0 warmed up
GPU 1 warmed up
GPU 2 warmed up
GPU 3 warmed up
GPU warmup completed
INFO 05-06 15:31:20.261186.261186 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-06 15:31:20.691933.691933 cuda_h.py:27] end init_cmv_hmv cost 430.702 ms
DEBUG 05-06 15:31:20.699455.699455 cuda_memory_view.py:1366] 
DEBUG 05-06 15:31:20.699455.699455 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.0026047229766845703
DEBUG 05-06 15:31:20.716391.716391 mlpmodule.py:993] restore_hm_state_dict2model loaded 657 language_model tensors for Gemma4 model
DEBUG 05-06 15:31:20.717245.717245 cuda_memory_view.py:1370] 
DEBUG 05-06 15:31:20.717245.717245 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.017147064208984375
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-06 15:31:22.620937.620937 lmp.py:255] init kt-kernel layer 0 ok
INFO 05-06 15:31:23.463017.463017 lmp.py:255] init kt-kernel layer 1 ok
INFO 05-06 15:31:24.295576.295576 lmp.py:255] init kt-kernel layer 2 ok
INFO 05-06 15:31:25.123228.123228 lmp.py:255] init kt-kernel layer 3 ok
INFO 05-06 15:31:25.962356.962356 lmp.py:255] init kt-kernel layer 4 ok
INFO 05-06 15:31:26.796983.796983 lmp.py:255] init kt-kernel layer 5 ok
INFO 05-06 15:31:27.644840.644840 lmp.py:255] init kt-kernel layer 6 ok
INFO 05-06 15:31:28.481194.481194 lmp.py:255] init kt-kernel layer 7 ok
INFO 05-06 15:31:29.315440.315440 lmp.py:255] init kt-kernel layer 8 ok
INFO 05-06 15:31:30.160636.160636 lmp.py:255] init kt-kernel layer 9 ok
INFO 05-06 15:31:31.013228.013228 lmp.py:255] init kt-kernel layer 10 ok
INFO 05-06 15:31:31.838304.838304 lmp.py:255] init kt-kernel layer 11 ok
INFO 05-06 15:31:32.665702.665702 lmp.py:255] init kt-kernel layer 12 ok
INFO 05-06 15:31:33.484065.484065 lmp.py:255] init kt-kernel layer 13 ok
INFO 05-06 15:31:34.323023.323023 lmp.py:255] init kt-kernel layer 14 ok
INFO 05-06 15:31:35.155619.155619 lmp.py:255] init kt-kernel layer 15 ok
INFO 05-06 15:31:36.017561.017561 lmp.py:255] init kt-kernel layer 16 ok
INFO 05-06 15:31:36.854275.854275 lmp.py:255] init kt-kernel layer 17 ok
INFO 05-06 15:31:37.693807.693807 lmp.py:255] init kt-kernel layer 18 ok
INFO 05-06 15:31:38.528134.528134 lmp.py:255] init kt-kernel layer 19 ok
INFO 05-06 15:31:39.330491.330491 lmp.py:255] init kt-kernel layer 20 ok
INFO 05-06 15:31:40.149402.149402 lmp.py:255] init kt-kernel layer 21 ok
INFO 05-06 15:31:40.983015.983015 lmp.py:255] init kt-kernel layer 22 ok
CPUInfer[0x62ed076c2d70]: Hello
WorkerPool[0x62ed076b2550] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x62ed2d4e7b70]: Hello
WorkerPool[0x62ed243f6260] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVINFO 05-06 15:31:41.811859.811859 lmp.py:255] init kt-kernel layer 23 ok
INFO 05-06 15:31:42.632026.632026 lmp.py:255] init kt-kernel layer 24 ok
INFO 05-06 15:31:43.450222.450222 lmp.py:255] init kt-kernel layer 25 ok
INFO 05-06 15:31:44.269785.269785 lmp.py:255] init kt-kernel layer 26 ok
INFO 05-06 15:31:45.130551.130551 lmp.py:255] init kt-kernel layer 27 ok
INFO 05-06 15:31:45.990998.990998 lmp.py:255] init kt-kernel layer 28 ok
INFO 05-06 15:31:46.833829.833829 lmp.py:255] init kt-kernel layer 29 ok
INFO 05-06 15:31:47.673536.673536 lmp.py:186] vLLM Triton fused-MoE enabled (CUDAGraph=False).
generate input ids cost 0.06088566780090332 s
DEBUG 05-06 15:31:50.569112.569112 cuda_h.py:27] end generate_input_ids cost 2876.870 ms
DEBUG 05-06 15:31:50.569988.569988 cuda_h.py:27] end init_cache cost 0.046 ms
INFO 05-06 15:31:50.579947.579947 lmp.py:367] _ensure_static_kv_cache (Gemma4 list): 30 layers, 1760.0 MiB on cuda:0
INFO 05-06 15:31:50.579003.579003 lmp.py:1162] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 15:31:50.593515.593515 lmp.py:2797] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 4784365508, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.787078263565146, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-06 15:31:50.593565.593565 lmp.py:2815] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.593057.593057 lmp.py:2815] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.593588.593588 lmp.py:2815] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.593212.593212 lmp.py:2815] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.594482.594482 lmp.py:2815] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.594821.594821 lmp.py:2815] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.594776.594776 lmp.py:2815] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.594869.594869 lmp.py:2815] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.594016.594016 lmp.py:2815] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.594334.594334 lmp.py:2815] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.594957.594957 lmp.py:2815] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.594112.594112 lmp.py:2815] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.594497.594497 lmp.py:2815] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595067.595067 lmp.py:2815] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595691.595691 lmp.py:2815] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595194.595194 lmp.py:2815] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595341.595341 lmp.py:2815] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595585.595585 lmp.py:2815] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595448.595448 lmp.py:2815] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595979.595979 lmp.py:2815] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595172.595172 lmp.py:2815] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595039.595039 lmp.py:2815] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.595901.595901 lmp.py:2815] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.596820.596820 lmp.py:2815] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.596967.596967 lmp.py:2815] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.596813.596813 lmp.py:2815] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.596198.596198 lmp.py:2815] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.596257.596257 lmp.py:2815] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.596166.596166 lmp.py:2815] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:50.596258.596258 lmp.py:2815] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 15:31:50.892396.892396 cuda_h.py:27] end init_loading_placement cost 312.903 ms
DEBUG 05-06 15:31:50.892427.892427 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 15:31:50.892867.892867 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 15:31:50 client.py:72] load_into_gpu: gemma4-26B-A4B, cd3f708e-0519-4fab-8851-32969e4a2580
INFO 05-06 15:31:50 client.py:135] Model loaded: gemma4-26B-A4B, cd3f708e-0519-4fab-8851-32969e4a2580
INFO 05-06 15:31:50 client.py:204] confirm_model_loaded: gemma4-26B-A4B, cd3f708e-0519-4fab-8851-32969e4a2580
INFO 05-06 15:31:51 client.py:212] Model loaded
DEBUG 05-06 15:31:51.422001.422001 cuda_h.py:27] end init_general_sagl_loading_async cost 530.366 ms
INFO 05-06 15:31:51.472407.472407 lmp.py:3318] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 15:31:51.578077.578077 cuda_h.py:27] end restore_state_dict cost 105.710 ms
WARNING 05-06 15:31:51 [fused_moe.py:1090] Using default MoE config. Performance might be sub-optimal! Config file not found at /mnt/zhengcf3/lmp_env/fslmp/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=32,N=704,device_name=NVIDIA_GeForce_RTX_4090.json
INFO 05-06 15:31:52.629770.629770 lmp.py:1291] vLLM Triton pre-warmup done in 1050.9 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 15:31:52.629426.629426 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 15:31:52.629190.629190 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 15:31:52 client.py:72] load_into_gpu: gemma4-26B-A4B, b32ee213-cbc1-4efd-a1fe-b0c5a3f06dee
INFO 05-06 15:31:52 client.py:135] Model loaded: gemma4-26B-A4B, b32ee213-cbc1-4efd-a1fe-b0c5a3f06dee
DEBUG 05-06 15:31:52.703640.703640 cuda_h.py:27] end init_experts_loading_async cost 74.183 ms
DEBUG 05-06 15:31:52.731463.731463 cuda_h.py:27] end init_inputs_tokens cost 27.762 ms
DEBUG 05-06 15:31:52.731713.731713 lmp.py:1350] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 15:31:52.800385.800385 cuda_h.py:27] end prefill_ln cost 68.258 ms
DEBUG 05-06 15:31:52.876967.876967 cuda_h.py:27] end prefill_attn cost 76.200 ms
DEBUG 05-06 15:31:52.876374.876374 cuda_h.py:27] end prefill_ffn_prep cost 0.348 ms
DEBUG 05-06 15:31:52.954524.954524 cuda_h.py:27] end prefill_gate cost 72.072 ms
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 75, 115, 83, 127, 31, 3, 67, 51, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 92, 116, 112, 64, 72, 48, 32, 52, 101, 109, 85, 49, 45, 65, 93, 69, 5, 13, 9, 73, 77, 37, 89, 25, 105, 117, 125, 41, 113, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18, 70, 110, 118, 122, 78, 26, 54, 74], 'token_total': 1502, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 75: 29, 115: 29, 83: 33, 127: 33, 31: 34, 3: 46, 67: 47, 51: 48, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 92: 16, 116: 18, 112: 23, 64: 27, 72: 35, 48: 41, 32: 43, 52: 43, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 13: 16, 9: 17, 73: 19, 77: 19, 37: 20, 89: 20, 25: 24, 105: 24, 117: 26, 125: 26, 41: 27, 113: 39, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18, 70: 25, 110: 27, 118: 29, 122: 35, 78: 36, 26: 59, 54: 59, 74: 61}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 39, 47, 55, 91, 103], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 28, 'token_total': 917, 'token_per_expert': {7: 95, 39: 176, 47: 318, 55: 51, 91: 99, 103: 178}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 68, 104, 124], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 26, 'token_total': 512, 'token_per_expert': {0: 73, 16: 48, 68: 170, 104: 43, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 33, 53, 121], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 26, 'token_total': 603, 'token_per_expert': {1: 75, 21: 48, 33: 210, 53: 205, 121: 65}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 46, 50, 90, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 25, 'token_total': 562, 'token_per_expert': {22: 64, 46: 119, 50: 110, 90: 154, 126: 115}}
INFO 05-06 15:31:52.986552.986552 lmp.py:1839] [layer_moe_fused] layer=0 prefix: 30.422ms alloc: 0.279ms
INFO 05-06 15:31:52.986696.986696 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.7418136596679688e-05 seconds
INFO 05-06 15:31:52.988468.988468 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0018582344055175781s
INFO 05-06 15:31:52.988147.988147 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008149147033691406 seconds
DEBUG 05-06 15:31:52.989914.989914 cuda_h.py:27] end moe_cpu_prep_submit cost 1.377 ms
INFO 05-06 15:31:52.991377.991377 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016865730285644531s
DEBUG 05-06 15:31:52.991149.991149 cuda_h.py:27] end moe_wait_copy_tasks cost 1.806 ms
DEBUG 05-06 15:31:53.023250.023250 cuda_h.py:27] end moe_vllm_forward cost 31.065 ms
DEBUG 05-06 15:31:53.187152.187152 cuda_h.py:27] end moe_cpu_merge cost 164.630 ms
DEBUG 05-06 15:31:53.188554.188554 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:53.188266.188266 lmp.py:1953] [layer_moe_fused] vllm triton time: 196.678ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.188267.188267 cuda_h.py:27] end *layer_moe_fused cost 233.250 ms
DEBUG 05-06 15:31:53.190750.190750 cuda_h.py:27] end prefill_merge_scale cost 1.486 ms
DEBUG 05-06 15:31:53.190185.190185 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.057 ms
DEBUG 05-06 15:31:53.190610.190610 cuda_h.py:27] end prefill_layer cost 458.781 ms
DEBUG 05-06 15:31:53.190040.190040 lmp.py:1394] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 15:31:53.190525.190525 lmp.py:1350] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 15:31:53.190057.190057 cuda_h.py:27] end prefill_ln cost 0.209 ms
DEBUG 05-06 15:31:53.194580.194580 cuda_h.py:27] end prefill_attn cost 3.094 ms
DEBUG 05-06 15:31:53.194874.194874 cuda_h.py:27] end prefill_ffn_prep cost 0.411 ms
DEBUG 05-06 15:31:53.195396.195396 cuda_h.py:27] end prefill_gate cost 0.570 ms
experts_cpu_alloc {'expert_ids': [63, 107, 39, 23, 111, 87, 3, 35, 31, 55, 91, 7, 71, 127, 67, 83, 95, 11, 123, 103, 15, 115, 28, 36, 44, 32, 48, 88, 112, 76, 24, 16, 56, 40, 120, 108, 68, 4, 116, 92, 84, 52, 124, 12, 60, 100, 72, 96, 61, 113, 33, 57, 1, 125, 41, 69, 117, 45, 5, 21, 77, 105, 121, 81, 101, 29, 49, 37, 73, 53, 85, 65, 89, 2, 14, 62, 114, 86, 18, 66, 54, 38, 78, 102, 34, 10, 110, 74, 90, 26, 50, 42, 94, 46, 106, 122], 'token_total': 1957, 'token_per_expert': {63: 1, 107: 1, 39: 3, 23: 4, 111: 4, 87: 6, 3: 7, 35: 10, 31: 11, 55: 11, 91: 11, 7: 13, 71: 16, 127: 16, 67: 17, 83: 18, 95: 20, 11: 25, 123: 27, 103: 31, 15: 34, 115: 36, 28: 1, 36: 1, 44: 2, 32: 3, 48: 3, 88: 3, 112: 3, 76: 5, 24: 6, 16: 8, 56: 9, 40: 10, 120: 16, 108: 17, 68: 21, 4: 23, 116: 23, 92: 27, 84: 29, 52: 33, 124: 34, 12: 38, 60: 38, 100: 41, 72: 47, 96: 52, 61: 1, 113: 2, 33: 4, 57: 4, 1: 9, 125: 9, 41: 11, 69: 12, 117: 13, 45: 14, 5: 15, 21: 16, 77: 16, 105: 16, 121: 16, 81: 18, 101: 18, 29: 24, 49: 25, 37: 31, 73: 32, 53: 38, 85: 42, 65: 46, 89: 50, 2: 1, 14: 1, 62: 4, 114: 6, 86: 9, 18: 12, 66: 14, 54: 15, 38: 19, 78: 19, 102: 23, 34: 28, 10: 29, 110: 34, 74: 35, 90: 39, 26: 41, 50: 42, 42: 43, 94: 55, 46: 61, 106: 65, 122: 65}}
experts_gpu_alloc_device_0 {'expert_ids': [27, 47, 51, 59, 79, 99, 119], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 29, 'token_total': 458, 'token_per_expert': {27: 49, 47: 88, 51: 58, 59: 48, 79: 60, 99: 96, 119: 59}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 20, 64, 80, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 32, 'token_total': 662, 'token_per_expert': {0: 56, 8: 159, 20: 151, 64: 86, 80: 147, 104: 63}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 13, 25, 93, 97, 109], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 31, 'token_total': 475, 'token_per_expert': {9: 66, 13: 125, 25: 60, 93: 64, 97: 109, 109: 51}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 22, 30, 82, 98, 118], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 29, 'token_total': 544, 'token_per_expert': {6: 80, 22: 126, 30: 82, 82: 103, 98: 78, 118: 75}}
INFO 05-06 15:31:53.197001.197001 lmp.py:1839] [layer_moe_fused] layer=1 prefix: 0.494ms alloc: 0.435ms
INFO 05-06 15:31:53.197534.197534 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 3.5762786865234375e-05 seconds
INFO 05-06 15:31:53.198082.198082 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0011126995086669922s
INFO 05-06 15:31:53.199079.199079 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006380081176757812 seconds
DEBUG 05-06 15:31:53.199883.199883 cuda_h.py:27] end moe_cpu_prep_submit cost 0.734 ms
INFO 05-06 15:31:53.201211.201211 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017421245574951172s
DEBUG 05-06 15:31:53.201839.201839 cuda_h.py:27] end moe_wait_copy_tasks cost 1.956 ms
DEBUG 05-06 15:31:53.210725.210725 cuda_h.py:27] end moe_vllm_forward cost 8.307 ms
DEBUG 05-06 15:31:53.276164.276164 cuda_h.py:27] end moe_cpu_merge cost 65.097 ms
DEBUG 05-06 15:31:53.276400.276400 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:53.276191.276191 lmp.py:1953] [layer_moe_fused] vllm triton time: 74.515ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.276355.276355 cuda_h.py:27] end *layer_moe_fused cost 80.338 ms
DEBUG 05-06 15:31:53.277908.277908 cuda_h.py:27] end prefill_merge_scale cost 0.459 ms
DEBUG 05-06 15:31:53.277548.277548 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.047 ms
DEBUG 05-06 15:31:53.277381.277381 cuda_h.py:27] end prefill_layer cost 86.691 ms
DEBUG 05-06 15:31:53.277670.277670 lmp.py:1394] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 15:31:53.277432.277432 lmp.py:1350] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 15:31:53.277456.277456 cuda_h.py:27] end prefill_ln cost 0.202 ms
DEBUG 05-06 15:31:53.279376.279376 cuda_h.py:27] end prefill_attn cost 1.913 ms
DEBUG 05-06 15:31:53.280140.280140 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 15:31:53.281777.281777 cuda_h.py:27] end prefill_gate cost 0.425 ms
experts_cpu_alloc {'expert_ids': [31, 27, 67, 47, 63, 51, 55, 119, 123, 91, 103, 107, 115, 127, 71, 99, 23, 87, 95, 15, 83, 59, 12, 92, 36, 116, 16, 104, 8, 24, 40, 120, 68, 44, 84, 100, 56, 52, 88, 96, 48, 20, 28, 64, 60, 76, 57, 105, 21, 53, 61, 93, 117, 33, 97, 113, 45, 73, 77, 125, 17, 49, 41, 85, 69, 13, 37, 65, 10, 38, 70, 74, 98, 66, 82, 54, 26, 118, 58, 126, 14, 18, 122, 46, 90, 50, 114, 106, 42, 110], 'token_total': 1439, 'token_per_expert': {31: 1, 27: 5, 67: 5, 47: 7, 63: 7, 51: 9, 55: 10, 119: 16, 123: 17, 91: 18, 103: 20, 107: 21, 115: 21, 127: 23, 71: 27, 99: 31, 23: 32, 87: 40, 95: 41, 15: 44, 83: 63, 59: 72, 12: 1, 92: 1, 36: 2, 116: 3, 16: 4, 104: 4, 8: 5, 24: 6, 40: 6, 120: 6, 68: 8, 44: 10, 84: 11, 100: 13, 56: 15, 52: 16, 88: 19, 96: 20, 48: 21, 20: 25, 28: 26, 64: 35, 60: 39, 76: 44, 57: 1, 105: 1, 21: 2, 53: 3, 61: 3, 93: 3, 117: 4, 33: 5, 97: 7, 113: 7, 45: 11, 73: 13, 77: 13, 125: 19, 17: 24, 49: 26, 41: 27, 85: 28, 69: 36, 13: 41, 37: 42, 65: 45, 10: 1, 38: 1, 70: 1, 74: 2, 98: 2, 66: 3, 82: 3, 54: 4, 26: 5, 118: 6, 58: 7, 126: 7, 14: 8, 18: 8, 122: 9, 46: 10, 90: 12, 50: 19, 114: 21, 106: 22, 42: 28, 110: 29}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 35, 43], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 28, 'token_total': 620, 'token_per_expert': {3: 161, 7: 132, 11: 75, 19: 89, 35: 78, 43: 85}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 72, 80, 108, 124], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 30, 'token_total': 764, 'token_per_expert': {0: 210, 4: 147, 72: 89, 80: 112, 108: 118, 124: 88}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 29, 81, 109], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 28, 'token_total': 831, 'token_per_expert': {1: 334, 5: 134, 9: 149, 29: 56, 81: 94, 109: 64}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34, 62, 102], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 27, 'token_total': 442, 'token_per_expert': {2: 128, 6: 128, 34: 84, 62: 40, 102: 62}}
INFO 05-06 15:31:53.282787.282787 lmp.py:1839] [layer_moe_fused] layer=2 prefix: 0.432ms alloc: 0.403ms
INFO 05-06 15:31:53.282275.282275 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 3.218650817871094e-05 seconds
INFO 05-06 15:31:53.283870.283870 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010402202606201172s
INFO 05-06 15:31:53.284574.284574 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005979537963867188 seconds
DEBUG 05-06 15:31:53.284684.284684 cuda_h.py:27] end moe_cpu_prep_submit cost 0.842 ms
INFO 05-06 15:31:53.287710.287710 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0019426345825195312s
DEBUG 05-06 15:31:53.287842.287842 cuda_h.py:27] end moe_wait_copy_tasks cost 2.133 ms
DEBUG 05-06 15:31:53.295173.295173 cuda_h.py:27] end moe_vllm_forward cost 7.449 ms
DEBUG 05-06 15:31:53.301415.301415 cuda_h.py:27] end moe_cpu_merge cost 5.512 ms
DEBUG 05-06 15:31:53.301829.301829 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 15:31:53.301514.301514 lmp.py:1953] [layer_moe_fused] vllm triton time: 13.872ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.301952.301952 cuda_h.py:27] end *layer_moe_fused cost 20.191 ms
DEBUG 05-06 15:31:53.302036.302036 cuda_h.py:27] end prefill_merge_scale cost 0.446 ms
DEBUG 05-06 15:31:53.302285.302285 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.047 ms
DEBUG 05-06 15:31:53.302449.302449 cuda_h.py:27] end prefill_layer cost 25.011 ms
DEBUG 05-06 15:31:53.302746.302746 lmp.py:1394] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 15:31:53.302794.302794 lmp.py:1350] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 15:31:53.303825.303825 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 15:31:53.305712.305712 cuda_h.py:27] end prefill_attn cost 2.100 ms
DEBUG 05-06 15:31:53.305523.305523 cuda_h.py:27] end prefill_ffn_prep cost 0.408 ms
DEBUG 05-06 15:31:53.307877.307877 cuda_h.py:27] end prefill_gate cost 0.536 ms
experts_cpu_alloc {'expert_ids': [123, 79, 55, 111, 103, 31, 87, 27, 35, 39, 23, 59, 75, 83, 43, 19, 127, 11, 15, 51, 119, 32, 36, 40, 76, 72, 116, 64, 104, 12, 24, 100, 56, 96, 8, 68, 120, 80, 20, 108, 88, 60, 44, 16, 97, 65, 57, 73, 89, 37, 113, 105, 121, 49, 125, 101, 21, 41, 109, 117, 33, 61, 69, 81, 93, 25, 53, 17, 82, 46, 78, 98, 106, 34, 122, 42, 118, 18, 86, 38, 74, 110, 58, 22, 30, 114, 14, 50, 70, 10, 26, 94], 'token_total': 1457, 'token_per_expert': {123: 3, 79: 4, 55: 5, 111: 5, 103: 7, 31: 8, 87: 12, 27: 13, 35: 15, 39: 16, 23: 19, 59: 19, 75: 21, 83: 21, 43: 22, 19: 25, 127: 25, 11: 28, 15: 30, 51: 33, 119: 33, 32: 1, 36: 1, 40: 1, 76: 1, 72: 2, 116: 2, 64: 5, 104: 6, 12: 7, 24: 8, 100: 11, 56: 12, 96: 12, 8: 13, 68: 14, 120: 14, 80: 25, 20: 27, 108: 30, 88: 43, 60: 46, 44: 48, 16: 65, 97: 1, 65: 2, 57: 3, 73: 3, 89: 3, 37: 4, 113: 4, 105: 6, 121: 6, 49: 8, 125: 9, 101: 11, 21: 12, 41: 13, 109: 13, 117: 17, 33: 18, 61: 18, 69: 21, 81: 25, 93: 37, 25: 40, 53: 40, 17: 41, 82: 1, 46: 3, 78: 4, 98: 4, 106: 5, 34: 6, 122: 6, 42: 7, 118: 7, 18: 9, 86: 9, 38: 11, 74: 12, 110: 13, 58: 17, 22: 18, 30: 19, 114: 22, 14: 23, 50: 24, 70: 26, 10: 28, 26: 31, 94: 39}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 67, 71, 91, 107], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 27, 'token_total': 625, 'token_per_expert': {3: 149, 7: 128, 67: 149, 71: 64, 91: 57, 107: 78}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 52, 84, 92], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 29, 'token_total': 801, 'token_per_expert': {0: 157, 4: 216, 28: 159, 52: 70, 84: 118, 92: 81}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 29, 77, 85], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 30, 'token_total': 639, 'token_per_expert': {1: 155, 5: 150, 9: 88, 29: 44, 77: 84, 85: 118}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 62, 66, 102], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 29, 'token_total': 574, 'token_per_expert': {2: 154, 6: 191, 62: 122, 66: 60, 102: 47}}
INFO 05-06 15:31:53.308573.308573 lmp.py:1839] [layer_moe_fused] layer=3 prefix: 0.459ms alloc: 0.399ms
INFO 05-06 15:31:53.308629.308629 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 3.361701965332031e-05 seconds
INFO 05-06 15:31:53.309861.309861 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009796619415283203s
INFO 05-06 15:31:53.310135.310135 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005979537963867188 seconds
DEBUG 05-06 15:31:53.310269.310269 cuda_h.py:27] end moe_cpu_prep_submit cost 0.804 ms
INFO 05-06 15:31:53.312543.312543 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016472339630126953s
DEBUG 05-06 15:31:53.312093.312093 cuda_h.py:27] end moe_wait_copy_tasks cost 1.867 ms
DEBUG 05-06 15:31:53.320286.320286 cuda_h.py:27] end moe_vllm_forward cost 7.516 ms
DEBUG 05-06 15:31:53.327001.327001 cuda_h.py:27] end moe_cpu_merge cost 6.404 ms
DEBUG 05-06 15:31:53.327091.327091 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 15:31:53.327736.327736 lmp.py:1953] [layer_moe_fused] vllm triton time: 14.770ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.327108.327108 cuda_h.py:27] end *layer_moe_fused cost 20.604 ms
DEBUG 05-06 15:31:53.328548.328548 cuda_h.py:27] end prefill_merge_scale cost 0.447 ms
DEBUG 05-06 15:31:53.328744.328744 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.046 ms
DEBUG 05-06 15:31:53.328006.328006 cuda_h.py:27] end prefill_layer cost 25.928 ms
DEBUG 05-06 15:31:53.328799.328799 lmp.py:1394] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 15:31:53.328846.328846 lmp.py:1350] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 15:31:53.329427.329427 cuda_h.py:27] end prefill_ln cost 0.197 ms
DEBUG 05-06 15:31:53.331675.331675 cuda_h.py:27] end prefill_attn cost 1.806 ms
DEBUG 05-06 15:31:53.331047.331047 cuda_h.py:27] end prefill_ffn_prep cost 0.373 ms
DEBUG 05-06 15:31:53.332539.332539 cuda_h.py:27] end prefill_gate cost 0.414 ms
experts_cpu_alloc {'expert_ids': [39, 79, 127, 123, 115, 43, 107, 15, 87, 19, 59, 103, 111, 83, 75, 35, 51, 47, 71, 31, 63, 67, 12, 44, 52, 96, 108, 68, 112, 24, 64, 76, 84, 20, 40, 28, 88, 36, 72, 92, 100, 116, 56, 120, 33, 81, 121, 65, 69, 117, 45, 57, 85, 97, 101, 109, 17, 21, 29, 93, 105, 113, 49, 61, 77, 89, 125, 50, 66, 110, 18, 82, 62, 106, 122, 54, 86, 38, 34, 98, 78, 126, 30, 94, 22, 70], 'token_total': 816, 'token_per_expert': {39: 1, 79: 1, 127: 1, 123: 3, 115: 5, 43: 6, 107: 9, 15: 10, 87: 12, 19: 13, 59: 14, 103: 14, 111: 15, 83: 16, 75: 17, 35: 19, 51: 21, 47: 24, 71: 25, 31: 26, 63: 42, 67: 43, 12: 1, 44: 1, 52: 1, 96: 1, 108: 1, 68: 2, 112: 2, 24: 3, 64: 3, 76: 3, 84: 6, 20: 7, 40: 10, 28: 13, 88: 13, 36: 14, 72: 15, 92: 16, 100: 19, 116: 20, 56: 23, 120: 23, 33: 1, 81: 1, 121: 1, 65: 2, 69: 2, 117: 2, 45: 3, 57: 3, 85: 3, 97: 3, 101: 3, 109: 4, 17: 6, 21: 8, 29: 11, 93: 12, 105: 12, 113: 12, 49: 13, 61: 13, 77: 16, 89: 17, 125: 17, 50: 1, 66: 1, 110: 1, 18: 2, 82: 3, 62: 4, 106: 4, 122: 4, 54: 5, 86: 5, 38: 6, 34: 7, 98: 7, 78: 9, 126: 9, 30: 10, 94: 10, 22: 12, 70: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 55, 119], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 28, 'token_total': 940, 'token_per_expert': {3: 329, 7: 320, 23: 87, 27: 45, 55: 77, 119: 82}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 32, 80, 124], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 28, 'token_total': 800, 'token_per_expert': {0: 320, 4: 320, 8: 55, 32: 48, 80: 34, 124: 23}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 41, 53], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 28, 'token_total': 763, 'token_per_expert': {1: 377, 5: 328, 37: 19, 41: 20, 53: 19}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 74, 90], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 24, 'token_total': 777, 'token_per_expert': {2: 346, 6: 320, 26: 18, 74: 74, 90: 19}}
INFO 05-06 15:31:53.333177.333177 lmp.py:1839] [layer_moe_fused] layer=4 prefix: 0.423ms alloc: 0.386ms
INFO 05-06 15:31:53.333565.333565 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.9325485229492188e-05 seconds
INFO 05-06 15:31:53.334380.334380 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000972747802734375s
INFO 05-06 15:31:53.335586.335586 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005452632904052734 seconds
DEBUG 05-06 15:31:53.335656.335656 cuda_h.py:27] end moe_cpu_prep_submit cost 0.662 ms
INFO 05-06 15:31:53.338405.338405 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0022039413452148438s
DEBUG 05-06 15:31:53.338166.338166 cuda_h.py:27] end moe_wait_copy_tasks cost 2.409 ms
DEBUG 05-06 15:31:53.346963.346963 cuda_h.py:27] end moe_vllm_forward cost 7.507 ms
DEBUG 05-06 15:31:53.347366.347366 cuda_h.py:27] end moe_cpu_merge cost 1.235 ms
DEBUG 05-06 15:31:53.347349.347349 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 15:31:53.347750.347750 lmp.py:1953] [layer_moe_fused] vllm triton time: 9.472ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.348464.348464 cuda_h.py:27] end *layer_moe_fused cost 15.526 ms
DEBUG 05-06 15:31:53.348296.348296 cuda_h.py:27] end prefill_merge_scale cost 0.442 ms
DEBUG 05-06 15:31:53.348108.348108 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.043 ms
DEBUG 05-06 15:31:53.349152.349152 cuda_h.py:27] end prefill_layer cost 20.203 ms
DEBUG 05-06 15:31:53.349474.349474 lmp.py:1394] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 15:31:53.349522.349522 lmp.py:1350] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 15:31:53.349089.349089 cuda_h.py:27] end prefill_ln cost 0.194 ms
DEBUG 05-06 15:31:53.359548.359548 cuda_h.py:27] end prefill_attn cost 9.507 ms
DEBUG 05-06 15:31:53.359456.359456 cuda_h.py:27] end prefill_ffn_prep cost 0.497 ms
DEBUG 05-06 15:31:53.360775.360775 cuda_h.py:27] end prefill_gate cost 0.411 ms
experts_cpu_alloc {'expert_ids': [19, 31, 87, 107, 43, 39, 99, 55, 67, 15, 83, 51, 119, 127, 123, 52, 64, 96, 8, 20, 92, 36, 88, 84, 24, 16, 112, 116, 44, 48, 104, 32, 100, 68, 28, 120, 29, 69, 93, 37, 41, 97, 61, 57, 9, 45, 13, 101, 17, 89, 38, 46, 78, 102, 10, 82, 106, 70, 22, 98, 86, 118, 42, 14, 74, 126], 'token_total': 551, 'token_per_expert': {19: 1, 31: 1, 87: 1, 107: 1, 43: 2, 39: 3, 99: 4, 55: 5, 67: 5, 15: 6, 83: 6, 51: 10, 119: 11, 127: 17, 123: 19, 52: 1, 64: 1, 96: 1, 8: 2, 20: 2, 92: 3, 36: 4, 88: 4, 84: 5, 24: 6, 16: 7, 112: 7, 116: 7, 44: 8, 48: 8, 104: 10, 32: 11, 100: 11, 68: 14, 28: 16, 120: 18, 29: 1, 69: 1, 93: 1, 37: 3, 41: 5, 97: 7, 61: 9, 57: 16, 9: 17, 45: 19, 13: 23, 101: 23, 17: 24, 89: 25, 38: 1, 46: 1, 78: 1, 102: 1, 10: 2, 82: 2, 106: 3, 70: 4, 22: 5, 98: 6, 86: 10, 118: 15, 42: 19, 14: 20, 74: 22, 126: 27}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 71, 75, 111], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 20, 'token_total': 856, 'token_per_expert': {3: 384, 7: 405, 71: 26, 75: 20, 111: 21}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 56, 72], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 25, 'token_total': 904, 'token_per_expert': {0: 385, 4: 389, 56: 108, 72: 22}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 73, 113], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 895, 'token_per_expert': {1: 384, 5: 391, 73: 88, 113: 32}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 50, 94], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 890, 'token_per_expert': {2: 387, 6: 398, 50: 69, 94: 36}}
INFO 05-06 15:31:53.361365.361365 lmp.py:1839] [layer_moe_fused] layer=5 prefix: 0.408ms alloc: 0.319ms
INFO 05-06 15:31:53.361885.361885 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.6941299438476562e-05 seconds
INFO 05-06 15:31:53.362556.362556 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008594989776611328s
INFO 05-06 15:31:53.363226.363226 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005743503570556641 seconds
DEBUG 05-06 15:31:53.363189.363189 cuda_h.py:27] end moe_cpu_prep_submit cost 0.820 ms
INFO 05-06 15:31:53.365829.365829 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001844644546508789s
DEBUG 05-06 15:31:53.365992.365992 cuda_h.py:27] end moe_wait_copy_tasks cost 1.978 ms
DEBUG 05-06 15:31:53.373537.373537 cuda_h.py:27] end moe_vllm_forward cost 6.927 ms
DEBUG 05-06 15:31:53.373801.373801 cuda_h.py:27] end moe_cpu_merge cost 0.055 ms
DEBUG 05-06 15:31:53.373678.373678 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:53.373811.373811 lmp.py:1953] [layer_moe_fused] vllm triton time: 7.998ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.374442.374442 cuda_h.py:27] end *layer_moe_fused cost 13.112 ms
DEBUG 05-06 15:31:53.374716.374716 cuda_h.py:27] end prefill_merge_scale cost 0.425 ms
DEBUG 05-06 15:31:53.374236.374236 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:31:53.374240.374240 cuda_h.py:27] end prefill_layer cost 25.599 ms
DEBUG 05-06 15:31:53.374475.374475 lmp.py:1394] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 15:31:53.375238.375238 lmp.py:1350] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 15:31:53.375440.375440 cuda_h.py:27] end prefill_ln cost 0.191 ms
DEBUG 05-06 15:31:53.381818.381818 cuda_h.py:27] end prefill_attn cost 5.695 ms
DEBUG 05-06 15:31:53.381899.381899 cuda_h.py:27] end prefill_ffn_prep cost 0.369 ms
DEBUG 05-06 15:31:53.382099.382099 cuda_h.py:27] end prefill_gate cost 0.413 ms
experts_cpu_alloc {'expert_ids': [51, 55, 11, 67, 91, 115, 127, 27, 107, 23, 103, 111, 119, 123, 95, 19, 35, 8, 48, 60, 116, 120, 124, 24, 104, 28, 32, 72, 80, 56, 44, 16, 36, 96, 17, 21, 29, 33, 41, 81, 113, 45, 53, 89, 9, 105, 77, 121, 25, 57, 85, 93, 61, 13, 69, 10, 38, 74, 78, 98, 102, 30, 62, 82, 114, 70, 110, 90, 50, 106, 58, 26, 42, 86, 34, 94], 'token_total': 490, 'token_per_expert': {51: 2, 55: 2, 11: 4, 67: 4, 91: 4, 115: 5, 127: 5, 27: 8, 107: 9, 23: 10, 103: 10, 111: 11, 119: 12, 123: 14, 95: 18, 19: 23, 35: 26, 8: 1, 48: 1, 60: 1, 116: 1, 120: 1, 124: 1, 24: 2, 104: 2, 28: 3, 32: 3, 72: 4, 80: 4, 56: 5, 44: 6, 16: 8, 36: 9, 96: 9, 17: 1, 21: 1, 29: 1, 33: 1, 41: 1, 81: 2, 113: 2, 45: 3, 53: 3, 89: 3, 9: 4, 105: 4, 77: 5, 121: 5, 25: 6, 57: 7, 85: 7, 93: 8, 61: 9, 13: 11, 69: 15, 10: 1, 38: 1, 74: 2, 78: 2, 98: 2, 102: 2, 30: 4, 62: 4, 82: 4, 114: 4, 70: 5, 110: 6, 90: 8, 50: 9, 106: 10, 58: 11, 26: 12, 42: 15, 86: 18, 34: 20, 94: 23}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 75, 87, 99], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 22, 'token_total': 928, 'token_per_expert': {3: 385, 7: 384, 75: 27, 87: 60, 99: 72}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 100, 108, 112], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 22, 'token_total': 946, 'token_per_expert': {0: 459, 4: 385, 100: 34, 108: 56, 112: 12}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 65, 101], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 26, 'token_total': 876, 'token_per_expert': {1: 385, 5: 399, 37: 24, 65: 16, 101: 52}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 126], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 25, 'token_total': 856, 'token_per_expert': {2: 394, 6: 396, 22: 35, 126: 31}}
INFO 05-06 15:31:53.383338.383338 lmp.py:1839] [layer_moe_fused] layer=6 prefix: 0.412ms alloc: 0.346ms
INFO 05-06 15:31:53.383918.383918 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.6464462280273438e-05 seconds
INFO 05-06 15:31:53.384404.384404 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008008480072021484s
INFO 05-06 15:31:53.385929.385929 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005707740783691406 seconds
DEBUG 05-06 15:31:53.385544.385544 cuda_h.py:27] end moe_cpu_prep_submit cost 0.744 ms
INFO 05-06 15:31:53.387881.387881 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0018668174743652344s
DEBUG 05-06 15:31:53.387012.387012 cuda_h.py:27] end moe_wait_copy_tasks cost 2.018 ms
DEBUG 05-06 15:31:53.395738.395738 cuda_h.py:27] end moe_vllm_forward cost 6.989 ms
DEBUG 05-06 15:31:53.395101.395101 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 15:31:53.395768.395768 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 15:31:53.396735.396735 lmp.py:1953] [layer_moe_fused] vllm triton time: 8.312ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.396654.396654 cuda_h.py:27] end *layer_moe_fused cost 13.740 ms
DEBUG 05-06 15:31:53.397743.397743 cuda_h.py:27] end prefill_merge_scale cost 0.433 ms
DEBUG 05-06 15:31:53.397237.397237 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.045 ms
DEBUG 05-06 15:31:53.397109.397109 cuda_h.py:27] end prefill_layer cost 22.271 ms
DEBUG 05-06 15:31:53.397636.397636 lmp.py:1394] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 15:31:53.397445.397445 lmp.py:1350] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 15:31:53.397827.397827 cuda_h.py:27] end prefill_ln cost 0.197 ms
DEBUG 05-06 15:31:53.403022.403022 cuda_h.py:27] end prefill_attn cost 5.767 ms
DEBUG 05-06 15:31:53.404145.404145 cuda_h.py:27] end prefill_ffn_prep cost 0.417 ms
DEBUG 05-06 15:31:53.405029.405029 cuda_h.py:27] end prefill_gate cost 0.539 ms
experts_cpu_alloc {'expert_ids': [39, 19, 27, 63, 67, 111, 115, 51, 71, 23, 31, 79, 99, 55, 47, 83, 91, 107, 16, 84, 12, 20, 24, 8, 56, 64, 112, 32, 80, 104, 44, 28, 92, 68, 9, 13, 45, 113, 17, 25, 77, 65, 57, 89, 101, 105, 61, 53, 125, 18, 26, 30, 38, 42, 58, 66, 106, 74, 118, 126, 10, 86, 122, 82, 14, 54, 70, 90], 'token_total': 331, 'token_per_expert': {39: 1, 19: 2, 27: 2, 63: 2, 67: 2, 111: 2, 115: 3, 51: 4, 71: 5, 23: 6, 31: 6, 79: 6, 99: 7, 55: 8, 47: 10, 83: 11, 91: 14, 107: 14, 16: 1, 84: 1, 12: 2, 20: 2, 24: 2, 8: 3, 56: 3, 64: 3, 112: 3, 32: 4, 80: 4, 104: 4, 44: 5, 28: 6, 92: 6, 68: 9, 9: 1, 13: 1, 45: 2, 113: 2, 17: 3, 25: 3, 77: 3, 65: 4, 57: 5, 89: 5, 101: 5, 105: 5, 61: 8, 53: 9, 125: 11, 18: 1, 26: 1, 30: 1, 38: 1, 42: 2, 58: 2, 66: 2, 106: 4, 74: 5, 118: 5, 126: 5, 10: 6, 86: 7, 122: 7, 82: 9, 14: 10, 54: 11, 70: 11, 90: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 43, 103, 119], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 23, 'token_total': 963, 'token_per_expert': {3: 448, 7: 453, 43: 17, 103: 25, 119: 20}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 72, 120], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 916, 'token_per_expert': {0: 448, 4: 448, 72: 9, 120: 11}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 69], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 954, 'token_per_expert': {1: 448, 5: 449, 29: 13, 69: 44}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 102, 114], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 23, 'token_total': 932, 'token_per_expert': {2: 449, 6: 448, 102: 21, 114: 14}}
INFO 05-06 15:31:53.406860.406860 lmp.py:1839] [layer_moe_fused] layer=7 prefix: 0.431ms alloc: 0.316ms
INFO 05-06 15:31:53.406333.406333 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.5987625122070312e-05 seconds
INFO 05-06 15:31:53.407162.407162 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008144378662109375s
INFO 05-06 15:31:53.408502.408502 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006120204925537109 seconds
DEBUG 05-06 15:31:53.408252.408252 cuda_h.py:27] end moe_cpu_prep_submit cost 0.846 ms
INFO 05-06 15:31:53.410589.410589 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0018124580383300781s
DEBUG 05-06 15:31:53.410752.410752 cuda_h.py:27] end moe_wait_copy_tasks cost 1.949 ms
DEBUG 05-06 15:31:53.417039.417039 cuda_h.py:27] end moe_vllm_forward cost 6.728 ms
DEBUG 05-06 15:31:53.418765.418765 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 15:31:53.418516.418516 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:53.418241.418241 lmp.py:1953] [layer_moe_fused] vllm triton time: 7.614ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.418343.418343 cuda_h.py:27] end *layer_moe_fused cost 13.011 ms
DEBUG 05-06 15:31:53.419511.419511 cuda_h.py:27] end prefill_merge_scale cost 0.424 ms
DEBUG 05-06 15:31:53.419747.419747 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:31:53.419547.419547 cuda_h.py:27] end prefill_layer cost 22.019 ms
DEBUG 05-06 15:31:53.419618.419618 lmp.py:1394] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 15:31:53.419302.419302 lmp.py:1350] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 15:31:53.420051.420051 cuda_h.py:27] end prefill_ln cost 0.198 ms
DEBUG 05-06 15:31:53.425069.425069 cuda_h.py:27] end prefill_attn cost 5.217 ms
DEBUG 05-06 15:31:53.425296.425296 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 15:31:53.426806.426806 cuda_h.py:27] end prefill_gate cost 0.413 ms
experts_cpu_alloc {'expert_ids': [11, 31, 59, 91, 39, 87, 103, 15, 27, 55, 123, 71, 111, 47, 20, 48, 72, 88, 108, 52, 68, 100, 56, 104, 116, 36, 80, 76, 32, 24, 44, 120, 12, 9, 49, 69, 93, 17, 41, 77, 101, 81, 45, 57, 61, 105, 113, 53, 10, 14, 74, 82, 86, 126, 38, 118, 106, 110, 102, 98, 50, 46, 58, 66], 'token_total': 298, 'token_per_expert': {11: 1, 31: 1, 59: 1, 91: 1, 39: 2, 87: 2, 103: 2, 15: 4, 27: 5, 55: 6, 123: 6, 71: 7, 111: 8, 47: 10, 20: 1, 48: 1, 72: 1, 88: 1, 108: 1, 52: 2, 68: 2, 100: 2, 56: 3, 104: 3, 116: 3, 36: 4, 80: 4, 76: 5, 32: 6, 24: 9, 44: 10, 120: 15, 12: 19, 9: 2, 49: 2, 69: 2, 93: 2, 17: 3, 41: 3, 77: 3, 101: 3, 81: 4, 45: 6, 57: 6, 61: 6, 105: 8, 113: 8, 53: 10, 10: 1, 14: 1, 74: 1, 82: 1, 86: 2, 126: 2, 38: 3, 118: 3, 106: 5, 110: 5, 102: 6, 98: 7, 50: 10, 46: 11, 58: 12, 66: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 51, 75], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 19, 'token_total': 950, 'token_per_expert': {3: 454, 7: 448, 19: 12, 51: 17, 75: 19}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 64, 124], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 23, 'token_total': 946, 'token_per_expert': {0: 451, 4: 452, 64: 19, 124: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 73], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 966, 'token_per_expert': {1: 452, 5: 461, 29: 13, 73: 40}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 114, 122], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 936, 'token_per_expert': {2: 453, 6: 449, 114: 12, 122: 22}}
INFO 05-06 15:31:53.427177.427177 lmp.py:1839] [layer_moe_fused] layer=8 prefix: 0.404ms alloc: 0.311ms
INFO 05-06 15:31:53.427313.427313 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.5033950805664062e-05 seconds
INFO 05-06 15:31:53.428046.428046 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007612705230712891s
INFO 05-06 15:31:53.429502.429502 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005214214324951172 seconds
DEBUG 05-06 15:31:53.429706.429706 cuda_h.py:27] end moe_cpu_prep_submit cost 0.649 ms
INFO 05-06 15:31:53.431932.431932 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015838146209716797s
DEBUG 05-06 15:31:53.431379.431379 cuda_h.py:27] end moe_wait_copy_tasks cost 1.719 ms
DEBUG 05-06 15:31:53.438856.438856 cuda_h.py:27] end moe_vllm_forward cost 6.697 ms
DEBUG 05-06 15:31:53.438616.438616 cuda_h.py:27] end moe_cpu_merge cost 0.054 ms
DEBUG 05-06 15:31:53.439516.439516 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:53.439810.439810 lmp.py:1953] [layer_moe_fused] vllm triton time: 7.469ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.439374.439374 cuda_h.py:27] end *layer_moe_fused cost 12.348 ms
DEBUG 05-06 15:31:53.439642.439642 cuda_h.py:27] end prefill_merge_scale cost 0.428 ms
DEBUG 05-06 15:31:53.439308.439308 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:31:53.440203.440203 cuda_h.py:27] end prefill_layer cost 20.572 ms
DEBUG 05-06 15:31:53.440058.440058 lmp.py:1394] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 15:31:53.440820.440820 lmp.py:1350] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 15:31:53.440422.440422 cuda_h.py:27] end prefill_ln cost 0.196 ms
DEBUG 05-06 15:31:53.446076.446076 cuda_h.py:27] end prefill_attn cost 5.405 ms
DEBUG 05-06 15:31:53.446249.446249 cuda_h.py:27] end prefill_ffn_prep cost 0.368 ms
DEBUG 05-06 15:31:53.447688.447688 cuda_h.py:27] end prefill_gate cost 0.432 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.448817.448817 lmp.py:1839] [layer_moe_fused] layer=9 prefix: 0.329ms alloc: 0.102ms
INFO 05-06 15:31:53.448840.448840 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0251998901367188e-05 seconds
INFO 05-06 15:31:53.449399.449399 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007081031799316406s
INFO 05-06 15:31:53.449927.449927 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.00029587745666503906 seconds
DEBUG 05-06 15:31:53.449665.449665 cuda_h.py:27] end moe_cpu_prep_submit cost 0.382 ms
INFO 05-06 15:31:53.450334.450334 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0007731914520263672s
DEBUG 05-06 15:31:53.450331.450331 cuda_h.py:27] end moe_wait_copy_tasks cost 0.882 ms
DEBUG 05-06 15:31:53.464937.464937 cuda_h.py:27] end moe_vllm_forward cost 12.856 ms
DEBUG 05-06 15:31:53.619595.619595 cuda_h.py:27] end moe_cpu_merge cost 155.365 ms
DEBUG 05-06 15:31:53.620482.620482 cuda_h.py:27] end moe_shared_experts cost 0.010 ms
INFO 05-06 15:31:53.620763.620763 lmp.py:1953] [layer_moe_fused] vllm triton time: 168.975ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.620835.620835 cuda_h.py:27] end *layer_moe_fused cost 172.463 ms
DEBUG 05-06 15:31:53.621886.621886 cuda_h.py:27] end prefill_merge_scale cost 0.473 ms
DEBUG 05-06 15:31:53.621711.621711 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.045 ms
DEBUG 05-06 15:31:53.621107.621107 cuda_h.py:27] end prefill_layer cost 180.829 ms
DEBUG 05-06 15:31:53.621112.621112 lmp.py:1394] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 15:31:53.621159.621159 lmp.py:1350] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 15:31:53.621429.621429 cuda_h.py:27] end prefill_ln cost 0.201 ms
DEBUG 05-06 15:31:53.623482.623482 cuda_h.py:27] end prefill_attn cost 1.904 ms
DEBUG 05-06 15:31:53.624914.624914 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 15:31:53.625922.625922 cuda_h.py:27] end prefill_gate cost 0.420 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.625138.625138 lmp.py:1839] [layer_moe_fused] layer=10 prefix: 0.345ms alloc: 0.109ms
INFO 05-06 15:31:53.626413.626413 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 7.62939453125e-06 seconds
INFO 05-06 15:31:53.627915.627915 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009593963623046875s
INFO 05-06 15:31:53.627629.627629 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002903938293457031 seconds
DEBUG 05-06 15:31:53.627396.627396 cuda_h.py:27] end moe_cpu_prep_submit cost 0.446 ms
INFO 05-06 15:31:53.629411.629411 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015208721160888672s
DEBUG 05-06 15:31:53.629787.629787 cuda_h.py:27] end moe_wait_copy_tasks cost 1.700 ms
DEBUG 05-06 15:31:53.642313.642313 cuda_h.py:27] end moe_vllm_forward cost 11.919 ms
DEBUG 05-06 15:31:53.662580.662580 cuda_h.py:27] end moe_cpu_merge cost 19.872 ms
DEBUG 05-06 15:31:53.662161.662161 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:53.662198.662198 lmp.py:1953] [layer_moe_fused] vllm triton time: 32.715ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.663646.663646 cuda_h.py:27] end *layer_moe_fused cost 37.771 ms
DEBUG 05-06 15:31:53.663472.663472 cuda_h.py:27] end prefill_merge_scale cost 0.480 ms
DEBUG 05-06 15:31:53.663827.663827 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.046 ms
DEBUG 05-06 15:31:53.664912.664912 cuda_h.py:27] end prefill_layer cost 42.653 ms
DEBUG 05-06 15:31:53.664752.664752 lmp.py:1394] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 15:31:53.664038.664038 lmp.py:1350] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 15:31:53.664539.664539 cuda_h.py:27] end prefill_ln cost 0.194 ms
DEBUG 05-06 15:31:53.666831.666831 cuda_h.py:27] end prefill_attn cost 2.114 ms
DEBUG 05-06 15:31:53.667556.667556 cuda_h.py:27] end prefill_ffn_prep cost 0.409 ms
DEBUG 05-06 15:31:53.668605.668605 cuda_h.py:27] end prefill_gate cost 0.416 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.669622.669622 lmp.py:1839] [layer_moe_fused] layer=11 prefix: 0.340ms alloc: 0.111ms
INFO 05-06 15:31:53.669128.669128 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0967254638671875e-05 seconds
INFO 05-06 15:31:53.670783.670783 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009946823120117188s
INFO 05-06 15:31:53.670212.670212 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002925395965576172 seconds
DEBUG 05-06 15:31:53.670828.670828 cuda_h.py:27] end moe_cpu_prep_submit cost 0.481 ms
INFO 05-06 15:31:53.672599.672599 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015408992767333984s
DEBUG 05-06 15:31:53.672353.672353 cuda_h.py:27] end moe_wait_copy_tasks cost 1.717 ms
DEBUG 05-06 15:31:53.685609.685609 cuda_h.py:27] end moe_vllm_forward cost 12.074 ms
DEBUG 05-06 15:31:53.705664.705664 cuda_h.py:27] end moe_cpu_merge cost 19.893 ms
DEBUG 05-06 15:31:53.706914.706914 cuda_h.py:27] end moe_shared_experts cost 0.010 ms
INFO 05-06 15:31:53.706527.706527 lmp.py:1953] [layer_moe_fused] vllm triton time: 32.890ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.706918.706918 cuda_h.py:27] end *layer_moe_fused cost 37.963 ms
DEBUG 05-06 15:31:53.707233.707233 cuda_h.py:27] end prefill_merge_scale cost 0.460 ms
DEBUG 05-06 15:31:53.707051.707051 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.043 ms
DEBUG 05-06 15:31:53.707079.707079 cuda_h.py:27] end prefill_layer cost 43.147 ms
DEBUG 05-06 15:31:53.707122.707122 lmp.py:1394] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 15:31:53.707766.707766 lmp.py:1350] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 15:31:53.707551.707551 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 15:31:53.709740.709740 cuda_h.py:27] end prefill_attn cost 1.831 ms
DEBUG 05-06 15:31:53.710927.710927 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 15:31:53.711697.711697 cuda_h.py:27] end prefill_gate cost 0.420 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.712219.712219 lmp.py:1839] [layer_moe_fused] layer=12 prefix: 0.574ms alloc: 0.107ms
INFO 05-06 15:31:53.712248.712248 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 9.5367431640625e-06 seconds
INFO 05-06 15:31:53.713671.713671 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009834766387939453s
INFO 05-06 15:31:53.713378.713378 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002865791320800781 seconds
DEBUG 05-06 15:31:53.714431.714431 cuda_h.py:27] end moe_cpu_prep_submit cost 0.482 ms
INFO 05-06 15:31:53.715436.715436 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014274120330810547s
DEBUG 05-06 15:31:53.715647.715647 cuda_h.py:27] end moe_wait_copy_tasks cost 1.585 ms
DEBUG 05-06 15:31:53.728289.728289 cuda_h.py:27] end moe_vllm_forward cost 12.039 ms
DEBUG 05-06 15:31:53.748029.748029 cuda_h.py:27] end moe_cpu_merge cost 19.947 ms
DEBUG 05-06 15:31:53.749226.749226 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:53.749216.749216 lmp.py:1953] [layer_moe_fused] vllm triton time: 32.893ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.749854.749854 cuda_h.py:27] end *layer_moe_fused cost 38.057 ms
DEBUG 05-06 15:31:53.750613.750613 cuda_h.py:27] end prefill_merge_scale cost 0.469 ms
DEBUG 05-06 15:31:53.750101.750101 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.045 ms
DEBUG 05-06 15:31:53.750226.750226 cuda_h.py:27] end prefill_layer cost 42.825 ms
DEBUG 05-06 15:31:53.750092.750092 lmp.py:1394] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 15:31:53.750855.750855 lmp.py:1350] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 15:31:53.750588.750588 cuda_h.py:27] end prefill_ln cost 0.195 ms
DEBUG 05-06 15:31:53.752408.752408 cuda_h.py:27] end prefill_attn cost 1.876 ms
DEBUG 05-06 15:31:53.753211.753211 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 15:31:53.754119.754119 cuda_h.py:27] end prefill_gate cost 0.417 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.755806.755806 lmp.py:1839] [layer_moe_fused] layer=13 prefix: 0.345ms alloc: 0.109ms
INFO 05-06 15:31:53.755749.755749 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 9.775161743164062e-06 seconds
INFO 05-06 15:31:53.756101.756101 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012042522430419922s
INFO 05-06 15:31:53.756716.756716 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028705596923828125 seconds
DEBUG 05-06 15:31:53.757231.757231 cuda_h.py:27] end moe_cpu_prep_submit cost 0.608 ms
INFO 05-06 15:31:53.758862.758862 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001333475112915039s
DEBUG 05-06 15:31:53.759556.759556 cuda_h.py:27] end moe_wait_copy_tasks cost 1.506 ms
DEBUG 05-06 15:31:53.771766.771766 cuda_h.py:27] end moe_vllm_forward cost 11.370 ms
DEBUG 05-06 15:31:53.790102.790102 cuda_h.py:27] end moe_cpu_merge cost 19.370 ms
DEBUG 05-06 15:31:53.790829.790829 cuda_h.py:27] end moe_shared_experts cost 0.010 ms
INFO 05-06 15:31:53.791150.791150 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.681ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.791685.791685 cuda_h.py:27] end *layer_moe_fused cost 36.815 ms
DEBUG 05-06 15:31:53.791120.791120 cuda_h.py:27] end prefill_merge_scale cost 0.475 ms
DEBUG 05-06 15:31:53.792177.792177 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.043 ms
DEBUG 05-06 15:31:53.792389.792389 cuda_h.py:27] end prefill_layer cost 41.649 ms
DEBUG 05-06 15:31:53.792731.792731 lmp.py:1394] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 15:31:53.792017.792017 lmp.py:1350] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 15:31:53.792009.792009 cuda_h.py:27] end prefill_ln cost 0.192 ms
DEBUG 05-06 15:31:53.794596.794596 cuda_h.py:27] end prefill_attn cost 1.843 ms
DEBUG 05-06 15:31:53.795982.795982 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 15:31:53.796872.796872 cuda_h.py:27] end prefill_gate cost 0.430 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.796869.796869 lmp.py:1839] [layer_moe_fused] layer=14 prefix: 0.329ms alloc: 0.107ms
INFO 05-06 15:31:53.797322.797322 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 8.106231689453125e-06 seconds
INFO 05-06 15:31:53.798803.798803 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009622573852539062s
INFO 05-06 15:31:53.798709.798709 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028896331787109375 seconds
DEBUG 05-06 15:31:53.798792.798792 cuda_h.py:27] end moe_cpu_prep_submit cost 0.568 ms
INFO 05-06 15:31:53.800547.800547 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0013022422790527344s
DEBUG 05-06 15:31:53.800440.800440 cuda_h.py:27] end moe_wait_copy_tasks cost 1.489 ms
DEBUG 05-06 15:31:53.812133.812133 cuda_h.py:27] end moe_vllm_forward cost 11.316 ms
DEBUG 05-06 15:31:53.832190.832190 cuda_h.py:27] end moe_cpu_merge cost 19.561 ms
DEBUG 05-06 15:31:53.832274.832274 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:53.832350.832350 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.781ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.833640.833640 cuda_h.py:27] end *layer_moe_fused cost 36.718 ms
DEBUG 05-06 15:31:53.833186.833186 cuda_h.py:27] end prefill_merge_scale cost 0.455 ms
DEBUG 05-06 15:31:53.833766.833766 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.046 ms
DEBUG 05-06 15:31:53.833904.833904 cuda_h.py:27] end prefill_layer cost 41.523 ms
DEBUG 05-06 15:31:53.834036.834036 lmp.py:1394] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 15:31:53.834276.834276 lmp.py:1350] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 15:31:53.834624.834624 cuda_h.py:27] end prefill_ln cost 0.196 ms
DEBUG 05-06 15:31:53.836461.836461 cuda_h.py:27] end prefill_attn cost 1.780 ms
DEBUG 05-06 15:31:53.836403.836403 cuda_h.py:27] end prefill_ffn_prep cost 0.369 ms
DEBUG 05-06 15:31:53.837795.837795 cuda_h.py:27] end prefill_gate cost 0.414 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.838580.838580 lmp.py:1839] [layer_moe_fused] layer=15 prefix: 0.318ms alloc: 0.105ms
INFO 05-06 15:31:53.838563.838563 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0728836059570312e-05 seconds
INFO 05-06 15:31:53.839885.839885 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009620189666748047s
INFO 05-06 15:31:53.840115.840115 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028514862060546875 seconds
DEBUG 05-06 15:31:53.840569.840569 cuda_h.py:27] end moe_cpu_prep_submit cost 0.528 ms
INFO 05-06 15:31:53.842406.842406 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0012810230255126953s
DEBUG 05-06 15:31:53.842769.842769 cuda_h.py:27] end moe_wait_copy_tasks cost 1.464 ms
DEBUG 05-06 15:31:53.854041.854041 cuda_h.py:27] end moe_vllm_forward cost 11.293 ms
DEBUG 05-06 15:31:53.873918.873918 cuda_h.py:27] end moe_cpu_merge cost 19.324 ms
DEBUG 05-06 15:31:53.874956.874956 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 15:31:53.874647.874647 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.538ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.874001.874001 cuda_h.py:27] end *layer_moe_fused cost 36.552 ms
DEBUG 05-06 15:31:53.875256.875256 cuda_h.py:27] end prefill_merge_scale cost 0.448 ms
DEBUG 05-06 15:31:53.875167.875167 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.043 ms
DEBUG 05-06 15:31:53.875123.875123 cuda_h.py:27] end prefill_layer cost 41.318 ms
DEBUG 05-06 15:31:53.875646.875646 lmp.py:1394] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 15:31:53.875840.875840 lmp.py:1350] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 15:31:53.876063.876063 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 15:31:53.877933.877933 cuda_h.py:27] end prefill_attn cost 1.806 ms
DEBUG 05-06 15:31:53.878598.878598 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 15:31:53.879182.879182 cuda_h.py:27] end prefill_gate cost 0.426 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.880279.880279 lmp.py:1839] [layer_moe_fused] layer=16 prefix: 0.321ms alloc: 0.104ms
INFO 05-06 15:31:53.880786.880786 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0728836059570312e-05 seconds
INFO 05-06 15:31:53.881333.881333 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009336471557617188s
INFO 05-06 15:31:53.881179.881179 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002834796905517578 seconds
DEBUG 05-06 15:31:53.881141.881141 cuda_h.py:27] end moe_cpu_prep_submit cost 0.500 ms
INFO 05-06 15:31:53.883880.883880 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014312267303466797s
DEBUG 05-06 15:31:53.884450.884450 cuda_h.py:27] end moe_wait_copy_tasks cost 1.679 ms
DEBUG 05-06 15:31:53.896746.896746 cuda_h.py:27] end moe_vllm_forward cost 11.332 ms
DEBUG 05-06 15:31:53.916434.916434 cuda_h.py:27] end moe_cpu_merge cost 19.599 ms
DEBUG 05-06 15:31:53.916425.916425 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:53.916593.916593 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.815ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.916967.916967 cuda_h.py:27] end *layer_moe_fused cost 36.997 ms
DEBUG 05-06 15:31:53.917998.917998 cuda_h.py:27] end prefill_merge_scale cost 0.460 ms
DEBUG 05-06 15:31:53.917008.917008 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.046 ms
DEBUG 05-06 15:31:53.917788.917788 cuda_h.py:27] end prefill_layer cost 41.780 ms
DEBUG 05-06 15:31:53.917031.917031 lmp.py:1394] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 15:31:53.917033.917033 lmp.py:1350] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 15:31:53.918714.918714 cuda_h.py:27] end prefill_ln cost 0.199 ms
DEBUG 05-06 15:31:53.920911.920911 cuda_h.py:27] end prefill_attn cost 1.872 ms
DEBUG 05-06 15:31:53.920138.920138 cuda_h.py:27] end prefill_ffn_prep cost 0.369 ms
DEBUG 05-06 15:31:53.921449.921449 cuda_h.py:27] end prefill_gate cost 0.402 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.922519.922519 lmp.py:1839] [layer_moe_fused] layer=17 prefix: 0.319ms alloc: 0.105ms
INFO 05-06 15:31:53.922601.922601 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.049041748046875e-05 seconds
INFO 05-06 15:31:53.923855.923855 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010082721710205078s
INFO 05-06 15:31:53.923390.923390 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.00030040740966796875 seconds
DEBUG 05-06 15:31:53.924028.924028 cuda_h.py:27] end moe_cpu_prep_submit cost 0.514 ms
INFO 05-06 15:31:53.926825.926825 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015048980712890625s
DEBUG 05-06 15:31:53.926429.926429 cuda_h.py:27] end moe_wait_copy_tasks cost 1.788 ms
DEBUG 05-06 15:31:53.938003.938003 cuda_h.py:27] end moe_vllm_forward cost 11.351 ms
DEBUG 05-06 15:31:53.958051.958051 cuda_h.py:27] end moe_cpu_merge cost 19.695 ms
DEBUG 05-06 15:31:53.958374.958374 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:53.958450.958450 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.932ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:53.959271.959271 cuda_h.py:27] end *layer_moe_fused cost 37.325 ms
DEBUG 05-06 15:31:53.959241.959241 cuda_h.py:27] end prefill_merge_scale cost 0.450 ms
DEBUG 05-06 15:31:53.959757.959757 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.046 ms
DEBUG 05-06 15:31:53.959907.959907 cuda_h.py:27] end prefill_layer cost 42.147 ms
DEBUG 05-06 15:31:53.960296.960296 lmp.py:1394] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 15:31:53.960298.960298 lmp.py:1350] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 15:31:53.960687.960687 cuda_h.py:27] end prefill_ln cost 0.194 ms
DEBUG 05-06 15:31:53.962988.962988 cuda_h.py:27] end prefill_attn cost 1.807 ms
DEBUG 05-06 15:31:53.962268.962268 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 15:31:53.963899.963899 cuda_h.py:27] end prefill_gate cost 0.429 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:53.964505.964505 lmp.py:1839] [layer_moe_fused] layer=18 prefix: 0.326ms alloc: 0.104ms
INFO 05-06 15:31:53.964481.964481 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 7.3909759521484375e-06 seconds
INFO 05-06 15:31:53.965631.965631 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009305477142333984s
INFO 05-06 15:31:53.966246.966246 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028896331787109375 seconds
DEBUG 05-06 15:31:53.966803.966803 cuda_h.py:27] end moe_cpu_prep_submit cost 0.504 ms
INFO 05-06 15:31:53.967404.967404 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011718273162841797s
DEBUG 05-06 15:31:53.967874.967874 cuda_h.py:27] end moe_wait_copy_tasks cost 1.369 ms
DEBUG 05-06 15:31:53.980379.980379 cuda_h.py:27] end moe_vllm_forward cost 11.121 ms
DEBUG 05-06 15:31:53.999130.999130 cuda_h.py:27] end moe_cpu_merge cost 19.510 ms
DEBUG 05-06 15:31:53.999267.999267 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:53.999727.999727 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.520ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.000157.000157 cuda_h.py:27] end *layer_moe_fused cost 36.344 ms
DEBUG 05-06 15:31:54.000181.000181 cuda_h.py:27] end prefill_merge_scale cost 0.455 ms
DEBUG 05-06 15:31:54.000708.000708 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.044 ms
DEBUG 05-06 15:31:54.001387.001387 cuda_h.py:27] end prefill_layer cost 41.041 ms
DEBUG 05-06 15:31:54.001722.001722 lmp.py:1394] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 15:31:54.001200.001200 lmp.py:1350] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 15:31:54.001019.001019 cuda_h.py:27] end prefill_ln cost 0.203 ms
DEBUG 05-06 15:31:54.003028.003028 cuda_h.py:27] end prefill_attn cost 1.804 ms
DEBUG 05-06 15:31:54.003083.003083 cuda_h.py:27] end prefill_ffn_prep cost 0.384 ms
DEBUG 05-06 15:31:54.004462.004462 cuda_h.py:27] end prefill_gate cost 0.419 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.005876.005876 lmp.py:1839] [layer_moe_fused] layer=19 prefix: 0.323ms alloc: 0.110ms
INFO 05-06 15:31:54.005713.005713 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 9.775161743164062e-06 seconds
INFO 05-06 15:31:54.006442.006442 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009582042694091797s
INFO 05-06 15:31:54.007010.007010 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002892017364501953 seconds
DEBUG 05-06 15:31:54.007476.007476 cuda_h.py:27] end moe_cpu_prep_submit cost 0.540 ms
INFO 05-06 15:31:54.009919.009919 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015382766723632812s
DEBUG 05-06 15:31:54.009321.009321 cuda_h.py:27] end moe_wait_copy_tasks cost 1.703 ms
DEBUG 05-06 15:31:54.021577.021577 cuda_h.py:27] end moe_vllm_forward cost 11.052 ms
DEBUG 05-06 15:31:54.040531.040531 cuda_h.py:27] end moe_cpu_merge cost 19.214 ms
DEBUG 05-06 15:31:54.041264.041264 cuda_h.py:27] end moe_shared_experts cost 0.011 ms
INFO 05-06 15:31:54.041870.041870 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.208ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.041235.041235 cuda_h.py:27] end *layer_moe_fused cost 36.480 ms
DEBUG 05-06 15:31:54.042312.042312 cuda_h.py:27] end prefill_merge_scale cost 0.456 ms
DEBUG 05-06 15:31:54.042660.042660 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.045 ms
DEBUG 05-06 15:31:54.042786.042786 cuda_h.py:27] end prefill_layer cost 41.232 ms
DEBUG 05-06 15:31:54.042791.042791 lmp.py:1394] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 15:31:54.042077.042077 lmp.py:1350] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 15:31:54.043632.043632 cuda_h.py:27] end prefill_ln cost 0.203 ms
DEBUG 05-06 15:31:54.044365.044365 cuda_h.py:27] end prefill_attn cost 1.845 ms
DEBUG 05-06 15:31:54.045930.045930 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 15:31:54.046632.046632 cuda_h.py:27] end prefill_gate cost 0.414 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.047000.047000 lmp.py:1839] [layer_moe_fused] layer=20 prefix: 0.324ms alloc: 0.108ms
INFO 05-06 15:31:54.047884.047884 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0013580322265625e-05 seconds
INFO 05-06 15:31:54.048870.048870 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010175704956054688s
INFO 05-06 15:31:54.048869.048869 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002887248992919922 seconds
DEBUG 05-06 15:31:54.049583.049583 cuda_h.py:27] end moe_cpu_prep_submit cost 0.608 ms
INFO 05-06 15:31:54.050631.050631 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0012900829315185547s
DEBUG 05-06 15:31:54.051656.051656 cuda_h.py:27] end moe_wait_copy_tasks cost 1.453 ms
DEBUG 05-06 15:31:54.063082.063082 cuda_h.py:27] end moe_vllm_forward cost 11.227 ms
DEBUG 05-06 15:31:54.083947.083947 cuda_h.py:27] end moe_cpu_merge cost 19.729 ms
DEBUG 05-06 15:31:54.083130.083130 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 15:31:54.083252.083252 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.845ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.083367.083367 cuda_h.py:27] end *layer_moe_fused cost 37.119 ms
DEBUG 05-06 15:31:54.084437.084437 cuda_h.py:27] end prefill_merge_scale cost 0.454 ms
DEBUG 05-06 15:31:54.084918.084918 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.045 ms
DEBUG 05-06 15:31:54.084801.084801 cuda_h.py:27] end prefill_layer cost 42.000 ms
DEBUG 05-06 15:31:54.084197.084197 lmp.py:1394] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 15:31:54.084721.084721 lmp.py:1350] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 15:31:54.085593.085593 cuda_h.py:27] end prefill_ln cost 0.195 ms
DEBUG 05-06 15:31:54.086316.086316 cuda_h.py:27] end prefill_attn cost 1.769 ms
DEBUG 05-06 15:31:54.087305.087305 cuda_h.py:27] end prefill_ffn_prep cost 0.367 ms
DEBUG 05-06 15:31:54.088798.088798 cuda_h.py:27] end prefill_gate cost 0.419 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.089589.089589 lmp.py:1839] [layer_moe_fused] layer=21 prefix: 0.320ms alloc: 0.109ms
INFO 05-06 15:31:54.089625.089625 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 9.5367431640625e-06 seconds
INFO 05-06 15:31:54.090142.090142 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009648799896240234s
INFO 05-06 15:31:54.090425.090425 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002899169921875 seconds
DEBUG 05-06 15:31:54.091865.091865 cuda_h.py:27] end moe_cpu_prep_submit cost 0.564 ms
INFO 05-06 15:31:54.092644.092644 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0013241767883300781s
DEBUG 05-06 15:31:54.092789.092789 cuda_h.py:27] end moe_wait_copy_tasks cost 1.508 ms
DEBUG 05-06 15:31:54.105975.105975 cuda_h.py:27] end moe_vllm_forward cost 11.228 ms
DEBUG 05-06 15:31:54.126424.126424 cuda_h.py:27] end moe_cpu_merge cost 21.564 ms
DEBUG 05-06 15:31:54.126528.126528 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:54.127465.127465 lmp.py:1953] [layer_moe_fused] vllm triton time: 33.695ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.127885.127885 cuda_h.py:27] end *layer_moe_fused cost 38.748 ms
DEBUG 05-06 15:31:54.128042.128042 cuda_h.py:27] end prefill_merge_scale cost 0.481 ms
DEBUG 05-06 15:31:54.128099.128099 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.044 ms
DEBUG 05-06 15:31:54.128325.128325 cuda_h.py:27] end prefill_layer cost 43.563 ms
DEBUG 05-06 15:31:54.128331.128331 lmp.py:1394] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 15:31:54.128333.128333 lmp.py:1350] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 15:31:54.128681.128681 cuda_h.py:27] end prefill_ln cost 0.196 ms
DEBUG 05-06 15:31:54.130095.130095 cuda_h.py:27] end prefill_attn cost 1.822 ms
DEBUG 05-06 15:31:54.131375.131375 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 15:31:54.132549.132549 cuda_h.py:27] end prefill_gate cost 0.419 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.133447.133447 lmp.py:1839] [layer_moe_fused] layer=22 prefix: 0.326ms alloc: 0.106ms
INFO 05-06 15:31:54.133377.133377 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 7.152557373046875e-06 seconds
INFO 05-06 15:31:54.134375.134375 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009531974792480469s
INFO 05-06 15:31:54.134665.134665 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002951622009277344 seconds
DEBUG 05-06 15:31:54.134428.134428 cuda_h.py:27] end moe_cpu_prep_submit cost 0.526 ms
INFO 05-06 15:31:54.136806.136806 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0012812614440917969s
DEBUG 05-06 15:31:54.136732.136732 cuda_h.py:27] end moe_wait_copy_tasks cost 1.450 ms
DEBUG 05-06 15:31:54.148759.148759 cuda_h.py:27] end moe_vllm_forward cost 11.240 ms
DEBUG 05-06 15:31:54.168376.168376 cuda_h.py:27] end moe_cpu_merge cost 19.861 ms
DEBUG 05-06 15:31:54.169467.169467 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 15:31:54.169033.169033 lmp.py:1953] [layer_moe_fused] vllm triton time: 32.021ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.169918.169918 cuda_h.py:27] end *layer_moe_fused cost 37.056 ms
DEBUG 05-06 15:31:54.170571.170571 cuda_h.py:27] end prefill_merge_scale cost 0.462 ms
DEBUG 05-06 15:31:54.170005.170005 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.044 ms
DEBUG 05-06 15:31:54.170783.170783 cuda_h.py:27] end prefill_layer cost 41.939 ms
DEBUG 05-06 15:31:54.170198.170198 lmp.py:1394] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 15:31:54.170723.170723 lmp.py:1350] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 15:31:54.171326.171326 cuda_h.py:27] end prefill_ln cost 0.195 ms
DEBUG 05-06 15:31:54.173475.173475 cuda_h.py:27] end prefill_attn cost 2.222 ms
DEBUG 05-06 15:31:54.173869.173869 cuda_h.py:27] end prefill_ffn_prep cost 0.414 ms
DEBUG 05-06 15:31:54.174453.174453 cuda_h.py:27] end prefill_gate cost 0.415 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.175750.175750 lmp.py:1839] [layer_moe_fused] layer=23 prefix: 0.324ms alloc: 0.156ms
INFO 05-06 15:31:54.175210.175210 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0251998901367188e-05 seconds
INFO 05-06 15:31:54.176506.176506 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009675025939941406s
INFO 05-06 15:31:54.177511.177511 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002956390380859375 seconds
DEBUG 05-06 15:31:54.177124.177124 cuda_h.py:27] end moe_cpu_prep_submit cost 0.557 ms
INFO 05-06 15:31:54.179281.179281 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0012006759643554688s
DEBUG 05-06 15:31:54.179014.179014 cuda_h.py:27] end moe_wait_copy_tasks cost 1.360 ms
DEBUG 05-06 15:31:54.191692.191692 cuda_h.py:27] end moe_vllm_forward cost 11.143 ms
DEBUG 05-06 15:31:54.211421.211421 cuda_h.py:27] end moe_cpu_merge cost 19.613 ms
DEBUG 05-06 15:31:54.211009.211009 cuda_h.py:27] end moe_shared_experts cost 0.010 ms
INFO 05-06 15:31:54.211212.211212 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.731ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.212328.212328 cuda_h.py:27] end *layer_moe_fused cost 36.936 ms
DEBUG 05-06 15:31:54.212273.212273 cuda_h.py:27] end prefill_merge_scale cost 0.466 ms
DEBUG 05-06 15:31:54.212468.212468 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.045 ms
DEBUG 05-06 15:31:54.212871.212871 cuda_h.py:27] end prefill_layer cost 42.208 ms
DEBUG 05-06 15:31:54.213922.213922 lmp.py:1394] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 15:31:54.213923.213923 lmp.py:1350] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 15:31:54.213418.213418 cuda_h.py:27] end prefill_ln cost 0.196 ms
DEBUG 05-06 15:31:54.215091.215091 cuda_h.py:27] end prefill_attn cost 1.836 ms
DEBUG 05-06 15:31:54.215439.215439 cuda_h.py:27] end prefill_ffn_prep cost 0.416 ms
DEBUG 05-06 15:31:54.216016.216016 cuda_h.py:27] end prefill_gate cost 0.417 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.217006.217006 lmp.py:1839] [layer_moe_fused] layer=24 prefix: 0.324ms alloc: 0.108ms
INFO 05-06 15:31:54.217804.217804 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0013580322265625e-05 seconds
INFO 05-06 15:31:54.218870.218870 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009815692901611328s
INFO 05-06 15:31:54.219676.219676 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002899169921875 seconds
DEBUG 05-06 15:31:54.219818.219818 cuda_h.py:27] end moe_cpu_prep_submit cost 0.548 ms
INFO 05-06 15:31:54.221280.221280 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0013232231140136719s
DEBUG 05-06 15:31:54.221021.221021 cuda_h.py:27] end moe_wait_copy_tasks cost 1.491 ms
DEBUG 05-06 15:31:54.233697.233697 cuda_h.py:27] end moe_vllm_forward cost 11.261 ms
DEBUG 05-06 15:31:54.253216.253216 cuda_h.py:27] end moe_cpu_merge cost 19.898 ms
DEBUG 05-06 15:31:54.253730.253730 cuda_h.py:27] end moe_shared_experts cost 0.010 ms
INFO 05-06 15:31:54.253760.253760 lmp.py:1953] [layer_moe_fused] vllm triton time: 32.071ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.254467.254467 cuda_h.py:27] end *layer_moe_fused cost 37.153 ms
DEBUG 05-06 15:31:54.254199.254199 cuda_h.py:27] end prefill_merge_scale cost 0.451 ms
DEBUG 05-06 15:31:54.254203.254203 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:54.255784.255784 cuda_h.py:27] end prefill_layer cost 41.925 ms
DEBUG 05-06 15:31:54.255008.255008 lmp.py:1394] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 15:31:54.255009.255009 lmp.py:1350] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 15:31:54.255485.255485 cuda_h.py:27] end prefill_ln cost 0.196 ms
DEBUG 05-06 15:31:54.257163.257163 cuda_h.py:27] end prefill_attn cost 1.793 ms
DEBUG 05-06 15:31:54.257582.257582 cuda_h.py:27] end prefill_ffn_prep cost 0.369 ms
DEBUG 05-06 15:31:54.258146.258146 cuda_h.py:27] end prefill_gate cost 0.416 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.259573.259573 lmp.py:1839] [layer_moe_fused] layer=25 prefix: 0.320ms alloc: 0.119ms
INFO 05-06 15:31:54.259318.259318 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0967254638671875e-05 seconds
INFO 05-06 15:31:54.260949.260949 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009636878967285156s
INFO 05-06 15:31:54.261709.261709 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002849102020263672 seconds
DEBUG 05-06 15:31:54.261026.261026 cuda_h.py:27] end moe_cpu_prep_submit cost 0.571 ms
INFO 05-06 15:31:54.263966.263966 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0012924671173095703s
DEBUG 05-06 15:31:54.263421.263421 cuda_h.py:27] end moe_wait_copy_tasks cost 1.459 ms
DEBUG 05-06 15:31:54.275054.275054 cuda_h.py:27] end moe_vllm_forward cost 11.364 ms
DEBUG 05-06 15:31:54.296861.296861 cuda_h.py:27] end moe_cpu_merge cost 21.123 ms
DEBUG 05-06 15:31:54.297402.297402 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:54.297246.297246 lmp.py:1953] [layer_moe_fused] vllm triton time: 33.448ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.297232.297232 cuda_h.py:27] end *layer_moe_fused cost 38.499 ms
DEBUG 05-06 15:31:54.298646.298646 cuda_h.py:27] end prefill_merge_scale cost 0.460 ms
DEBUG 05-06 15:31:54.298749.298749 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.044 ms
DEBUG 05-06 15:31:54.298152.298152 cuda_h.py:27] end prefill_layer cost 43.222 ms
DEBUG 05-06 15:31:54.298282.298282 lmp.py:1394] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 15:31:54.298283.298283 lmp.py:1350] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 15:31:54.298732.298732 cuda_h.py:27] end prefill_ln cost 0.196 ms
DEBUG 05-06 15:31:54.300689.300689 cuda_h.py:27] end prefill_attn cost 1.835 ms
DEBUG 05-06 15:31:54.301877.301877 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 15:31:54.302162.302162 cuda_h.py:27] end prefill_gate cost 0.421 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.303960.303960 lmp.py:1839] [layer_moe_fused] layer=26 prefix: 0.322ms alloc: 0.113ms
INFO 05-06 15:31:54.303606.303606 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 7.62939453125e-06 seconds
INFO 05-06 15:31:54.304395.304395 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010423660278320312s
INFO 05-06 15:31:54.304446.304446 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.00029349327087402344 seconds
DEBUG 05-06 15:31:54.304284.304284 cuda_h.py:27] end moe_cpu_prep_submit cost 0.532 ms
INFO 05-06 15:31:54.307543.307543 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017273426055908203s
DEBUG 05-06 15:31:54.307390.307390 cuda_h.py:27] end moe_wait_copy_tasks cost 1.923 ms
DEBUG 05-06 15:31:54.319620.319620 cuda_h.py:27] end moe_vllm_forward cost 11.076 ms
DEBUG 05-06 15:31:54.339388.339388 cuda_h.py:27] end moe_cpu_merge cost 19.616 ms
DEBUG 05-06 15:31:54.339531.339531 cuda_h.py:27] end moe_shared_experts cost 0.010 ms
INFO 05-06 15:31:54.339799.339799 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.592ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.339431.339431 cuda_h.py:27] end *layer_moe_fused cost 37.232 ms
DEBUG 05-06 15:31:54.340177.340177 cuda_h.py:27] end prefill_merge_scale cost 0.457 ms
DEBUG 05-06 15:31:54.340134.340134 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.044 ms
DEBUG 05-06 15:31:54.340305.340305 cuda_h.py:27] end prefill_layer cost 41.974 ms
DEBUG 05-06 15:31:54.340701.340701 lmp.py:1394] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 15:31:54.340987.340987 lmp.py:1350] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 15:31:54.341010.341010 cuda_h.py:27] end prefill_ln cost 0.202 ms
DEBUG 05-06 15:31:54.342052.342052 cuda_h.py:27] end prefill_attn cost 1.794 ms
DEBUG 05-06 15:31:54.343571.343571 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 15:31:54.344063.344063 cuda_h.py:27] end prefill_gate cost 0.423 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.345623.345623 lmp.py:1839] [layer_moe_fused] layer=27 prefix: 0.321ms alloc: 0.110ms
INFO 05-06 15:31:54.345937.345937 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0251998901367188e-05 seconds
INFO 05-06 15:31:54.346341.346341 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009753704071044922s
INFO 05-06 15:31:54.346432.346432 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028896331787109375 seconds
DEBUG 05-06 15:31:54.347289.347289 cuda_h.py:27] end moe_cpu_prep_submit cost 0.553 ms
INFO 05-06 15:31:54.348454.348454 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014042854309082031s
DEBUG 05-06 15:31:54.349407.349407 cuda_h.py:27] end moe_wait_copy_tasks cost 1.588 ms
DEBUG 05-06 15:31:54.361500.361500 cuda_h.py:27] end moe_vllm_forward cost 11.262 ms
DEBUG 05-06 15:31:54.381711.381711 cuda_h.py:27] end moe_cpu_merge cost 19.979 ms
DEBUG 05-06 15:31:54.381663.381663 cuda_h.py:27] end moe_shared_experts cost 0.010 ms
INFO 05-06 15:31:54.381838.381838 lmp.py:1953] [layer_moe_fused] vllm triton time: 32.140ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.382699.382699 cuda_h.py:27] end *layer_moe_fused cost 37.325 ms
DEBUG 05-06 15:31:54.382491.382491 cuda_h.py:27] end prefill_merge_scale cost 0.460 ms
DEBUG 05-06 15:31:54.382740.382740 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.045 ms
DEBUG 05-06 15:31:54.382415.382415 cuda_h.py:27] end prefill_layer cost 42.063 ms
DEBUG 05-06 15:31:54.383685.383685 lmp.py:1394] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 15:31:54.383448.383448 lmp.py:1350] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 15:31:54.383035.383035 cuda_h.py:27] end prefill_ln cost 0.194 ms
DEBUG 05-06 15:31:54.385073.385073 cuda_h.py:27] end prefill_attn cost 1.857 ms
DEBUG 05-06 15:31:54.385122.385122 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 15:31:54.386368.386368 cuda_h.py:27] end prefill_gate cost 0.423 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.387457.387457 lmp.py:1839] [layer_moe_fused] layer=28 prefix: 0.324ms alloc: 0.109ms
INFO 05-06 15:31:54.387255.387255 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 9.298324584960938e-06 seconds
INFO 05-06 15:31:54.389457.389457 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010797977447509766s
INFO 05-06 15:31:54.389416.389416 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.00029540061950683594 seconds
DEBUG 05-06 15:31:54.389644.389644 cuda_h.py:27] end moe_cpu_prep_submit cost 0.554 ms
INFO 05-06 15:31:54.391663.391663 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0013325214385986328s
DEBUG 05-06 15:31:54.391642.391642 cuda_h.py:27] end moe_wait_copy_tasks cost 1.510 ms
DEBUG 05-06 15:31:54.403440.403440 cuda_h.py:27] end moe_vllm_forward cost 11.021 ms
DEBUG 05-06 15:31:54.423255.423255 cuda_h.py:27] end moe_cpu_merge cost 19.813 ms
DEBUG 05-06 15:31:54.423831.423831 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:31:54.423251.423251 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.792ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.424241.424241 cuda_h.py:27] end *layer_moe_fused cost 37.286 ms
DEBUG 05-06 15:31:54.424894.424894 cuda_h.py:27] end prefill_merge_scale cost 0.462 ms
DEBUG 05-06 15:31:54.424951.424951 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.044 ms
DEBUG 05-06 15:31:54.425143.425143 cuda_h.py:27] end prefill_layer cost 42.119 ms
DEBUG 05-06 15:31:54.425386.425386 lmp.py:1394] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 15:31:54.425341.425341 lmp.py:1350] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 15:31:54.425975.425975 cuda_h.py:27] end prefill_ln cost 0.194 ms
DEBUG 05-06 15:31:54.428450.428450 cuda_h.py:27] end prefill_attn cost 2.251 ms
DEBUG 05-06 15:31:54.428829.428829 cuda_h.py:27] end prefill_ffn_prep cost 0.373 ms
DEBUG 05-06 15:31:54.429154.429154 cuda_h.py:27] end prefill_gate cost 0.412 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 5, 2, 6], 'token_total': 3072, 'token_per_expert': {3: 512, 0: 512, 1: 512, 5: 512, 2: 512, 6: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 2, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 15:31:54.430237.430237 lmp.py:1839] [layer_moe_fused] layer=29 prefix: 0.326ms alloc: 0.108ms
INFO 05-06 15:31:54.430075.430075 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 1.0967254638671875e-05 seconds
INFO 05-06 15:31:54.431470.431470 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009450912475585938s
INFO 05-06 15:31:54.431290.431290 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002999305725097656 seconds
DEBUG 05-06 15:31:54.432326.432326 cuda_h.py:27] end moe_cpu_prep_submit cost 0.712 ms
INFO 05-06 15:31:54.433725.433725 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0013167858123779297s
DEBUG 05-06 15:31:54.434526.434526 cuda_h.py:27] end moe_wait_copy_tasks cost 1.494 ms
DEBUG 05-06 15:31:54.446905.446905 cuda_h.py:27] end moe_vllm_forward cost 11.290 ms
DEBUG 05-06 15:31:54.466146.466146 cuda_h.py:27] end moe_cpu_merge cost 19.722 ms
DEBUG 05-06 15:31:54.466873.466873 cuda_h.py:27] end moe_shared_experts cost 0.010 ms
INFO 05-06 15:31:54.466717.466717 lmp.py:1953] [layer_moe_fused] vllm triton time: 31.929ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:54.466845.466845 cuda_h.py:27] end *layer_moe_fused cost 36.982 ms
DEBUG 05-06 15:31:54.467783.467783 cuda_h.py:27] end prefill_merge_scale cost 0.461 ms
DEBUG 05-06 15:31:54.467263.467263 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.044 ms
DEBUG 05-06 15:31:54.467330.467330 cuda_h.py:27] end prefill_layer cost 42.196 ms
DEBUG 05-06 15:31:54.467548.467548 lmp.py:1394] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 15:31:54.467456.467456 cuda_h.py:27] end prefill_step cost 1736.039 ms
INFO 05-06 15:31:54.467539.467539 lmp.py:1397] prefill time: 1.8384532928466797 seconds
INFO 05-06 15:31:54.473097.473097 lmp.py:1409] Static-KV prefill complete; seqlens set to 128.
WARNING 05-06 15:31:54.502570.502570 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 15:31:54.503243.503243 helper.py:35]   NaN count (hidden): 1441792
WARNING 05-06 15:31:54.503180.503180 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 15:31:54.504476.504476 helper.py:39]   NaN count (normed): 1441792
WARNING 05-06 15:31:54.510451.510451 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 15:31:54.510145.510145 helper.py:50]   NaN count: 1048576
WARNING 05-06 15:31:54.510177.510177 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 15:31:54.626115.626115 cuda_h.py:27] end init_inputs_tokens cost 153.108 ms
DEBUG 05-06 15:31:54.627861.627861 lmp.py:1510] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 15:31:54.627651.627651 lmp.py:1516] ---- decode step 0 layer 0 ----
DEBUG 05-06 15:31:54.653606.653606 cuda_h.py:27] end decode_layer cost 26.533 ms
DEBUG 05-06 15:31:54.653013.653013 lmp.py:1516] ---- decode step 0 layer 1 ----
DEBUG 05-06 15:31:54.659233.659233 cuda_h.py:27] end decode_layer cost 5.562 ms
DEBUG 05-06 15:31:54.659520.659520 lmp.py:1516] ---- decode step 0 layer 2 ----
DEBUG 05-06 15:31:54.664767.664767 cuda_h.py:27] end decode_layer cost 4.812 ms
DEBUG 05-06 15:31:54.664326.664326 lmp.py:1516] ---- decode step 0 layer 3 ----
DEBUG 05-06 15:31:54.669385.669385 cuda_h.py:27] end decode_layer cost 5.130 ms
DEBUG 05-06 15:31:54.669890.669890 lmp.py:1516] ---- decode step 0 layer 4 ----
DEBUG 05-06 15:31:54.674713.674713 cuda_h.py:27] end decode_layer cost 4.779 ms
DEBUG 05-06 15:31:54.767008.767008 lmp.py:1516] ---- decode step 0 layer 5 ----
DEBUG 05-06 15:31:54.799639.799639 cuda_h.py:27] end decode_layer cost 31.935 ms
DEBUG 05-06 15:31:54.799762.799762 lmp.py:1516] ---- decode step 0 layer 6 ----
DEBUG 05-06 15:31:54.804519.804519 cuda_h.py:27] end decode_layer cost 5.010 ms
DEBUG 05-06 15:31:54.804422.804422 lmp.py:1516] ---- decode step 0 layer 7 ----
DEBUG 05-06 15:31:54.810616.810616 cuda_h.py:27] end decode_layer cost 5.578 ms
DEBUG 05-06 15:31:54.810320.810320 lmp.py:1516] ---- decode step 0 layer 8 ----
DEBUG 05-06 15:31:54.815530.815530 cuda_h.py:27] end decode_layer cost 4.889 ms
DEBUG 05-06 15:31:54.815419.815419 lmp.py:1516] ---- decode step 0 layer 9 ----
DEBUG 05-06 15:31:54.820777.820777 cuda_h.py:27] end decode_layer cost 4.926 ms
DEBUG 05-06 15:31:54.820812.820812 lmp.py:1516] ---- decode step 0 layer 10 ----
DEBUG 05-06 15:31:54.825001.825001 cuda_h.py:27] end decode_layer cost 4.838 ms
DEBUG 05-06 15:31:54.825082.825082 lmp.py:1516] ---- decode step 0 layer 11 ----
DEBUG 05-06 15:31:54.830420.830420 cuda_h.py:27] end decode_layer cost 5.334 ms
DEBUG 05-06 15:31:54.830031.830031 lmp.py:1516] ---- decode step 0 layer 12 ----
DEBUG 05-06 15:31:54.835127.835127 cuda_h.py:27] end decode_layer cost 4.840 ms
DEBUG 05-06 15:31:54.835732.835732 lmp.py:1516] ---- decode step 0 layer 13 ----
DEBUG 05-06 15:31:54.840736.840736 cuda_h.py:27] end decode_layer cost 4.878 ms
DEBUG 05-06 15:31:54.840149.840149 lmp.py:1516] ---- decode step 0 layer 14 ----
DEBUG 05-06 15:31:54.845408.845408 cuda_h.py:27] end decode_layer cost 4.750 ms
DEBUG 05-06 15:31:54.845674.845674 lmp.py:1516] ---- decode step 0 layer 15 ----
DEBUG 05-06 15:31:54.850968.850968 cuda_h.py:27] end decode_layer cost 4.811 ms
DEBUG 05-06 15:31:54.850189.850189 lmp.py:1516] ---- decode step 0 layer 16 ----
DEBUG 05-06 15:31:54.855096.855096 cuda_h.py:27] end decode_layer cost 4.736 ms
DEBUG 05-06 15:31:54.855932.855932 lmp.py:1516] ---- decode step 0 layer 17 ----
DEBUG 05-06 15:31:54.860694.860694 cuda_h.py:27] end decode_layer cost 5.155 ms
DEBUG 05-06 15:31:54.860253.860253 lmp.py:1516] ---- decode step 0 layer 18 ----
DEBUG 05-06 15:31:54.865916.865916 cuda_h.py:27] end decode_layer cost 4.766 ms
DEBUG 05-06 15:31:54.865520.865520 lmp.py:1516] ---- decode step 0 layer 19 ----
DEBUG 05-06 15:31:54.870564.870564 cuda_h.py:27] end decode_layer cost 4.873 ms
DEBUG 05-06 15:31:54.870308.870308 lmp.py:1516] ---- decode step 0 layer 20 ----
DEBUG 05-06 15:31:54.875452.875452 cuda_h.py:27] end decode_layer cost 4.874 ms
DEBUG 05-06 15:31:54.875341.875341 lmp.py:1516] ---- decode step 0 layer 21 ----
DEBUG 05-06 15:31:54.879872.879872 cuda_h.py:27] end decode_layer cost 4.774 ms
DEBUG 05-06 15:31:54.879477.879477 lmp.py:1516] ---- decode step 0 layer 22 ----
DEBUG 05-06 15:31:54.884408.884408 cuda_h.py:27] end decode_layer cost 4.859 ms
DEBUG 05-06 15:31:54.884371.884371 lmp.py:1516] ---- decode step 0 layer 23 ----
DEBUG 05-06 15:31:54.890439.890439 cuda_h.py:27] end decode_layer cost 5.206 ms
DEBUG 05-06 15:31:54.890467.890467 lmp.py:1516] ---- decode step 0 layer 24 ----
DEBUG 05-06 15:31:54.894528.894528 cuda_h.py:27] end decode_layer cost 4.779 ms
DEBUG 05-06 15:31:54.895557.895557 lmp.py:1516] ---- decode step 0 layer 25 ----
DEBUG 05-06 15:31:54.899541.899541 cuda_h.py:27] end decode_layer cost 4.863 ms
DEBUG 05-06 15:31:54.899907.899907 lmp.py:1516] ---- decode step 0 layer 26 ----
DEBUG 05-06 15:31:54.904551.904551 cuda_h.py:27] end decode_layer cost 4.787 ms
DEBUG 05-06 15:31:54.904917.904917 lmp.py:1516] ---- decode step 0 layer 27 ----
DEBUG 05-06 15:31:54.909776.909776 cuda_h.py:27] end decode_layer cost 4.876 ms
DEBUG 05-06 15:31:54.909043.909043 lmp.py:1516] ---- decode step 0 layer 28 ----
DEBUG 05-06 15:31:54.914933.914933 cuda_h.py:27] end decode_layer cost 4.828 ms
DEBUG 05-06 15:31:54.914776.914776 lmp.py:1516] ---- decode step 0 layer 29 ----
DEBUG 05-06 15:31:54.919854.919854 cuda_h.py:27] end decode_layer cost 5.280 ms
DEBUG 05-06 15:31:54.920986.920986 cuda_h.py:27] end decode_step cost 446.330 ms
INFO 05-06 15:31:54.920405.920405 lmp.py:1564] decode step 0 time: 0.4466078281402588 seconds
WARNING 05-06 15:31:54.920519.920519 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 15:31:54.920272.920272 helper.py:35]   NaN count (hidden): 11264
WARNING 05-06 15:31:54.921733.921733 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 15:31:54.921459.921459 helper.py:39]   NaN count (normed): 11264
WARNING 05-06 15:31:54.926927.926927 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 15:31:54.926997.926997 helper.py:50]   NaN count: 1048576
WARNING 05-06 15:31:54.927029.927029 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 15:31:54.928494.928494 cuda_h.py:27] end init_inputs_tokens cost 8.079 ms
DEBUG 05-06 15:31:54.928390.928390 lmp.py:1510] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 15:31:54.928444.928444 lmp.py:1516] ---- decode step 1 layer 0 ----
DEBUG 05-06 15:31:54.933304.933304 cuda_h.py:27] end decode_layer cost 4.910 ms
DEBUG 05-06 15:31:54.933432.933432 lmp.py:1516] ---- decode step 1 layer 1 ----
DEBUG 05-06 15:31:54.938750.938750 cuda_h.py:27] end decode_layer cost 4.933 ms
DEBUG 05-06 15:31:54.938878.938878 lmp.py:1516] ---- decode step 1 layer 2 ----
DEBUG 05-06 15:31:54.943180.943180 cuda_h.py:27] end decode_layer cost 4.677 ms
DEBUG 05-06 15:31:54.943685.943685 lmp.py:1516] ---- decode step 1 layer 3 ----
DEBUG 05-06 15:31:54.948296.948296 cuda_h.py:27] end decode_layer cost 4.798 ms
DEBUG 05-06 15:31:54.948616.948616 lmp.py:1516] ---- decode step 1 layer 4 ----
DEBUG 05-06 15:31:54.953268.953268 cuda_h.py:27] end decode_layer cost 4.829 ms
DEBUG 05-06 15:31:54.953773.953773 lmp.py:1516] ---- decode step 1 layer 5 ----
DEBUG 05-06 15:31:54.958442.958442 cuda_h.py:27] end decode_layer cost 5.122 ms
DEBUG 05-06 15:31:54.958662.958662 lmp.py:1516] ---- decode step 1 layer 6 ----
DEBUG 05-06 15:31:54.963364.963364 cuda_h.py:27] end decode_layer cost 4.725 ms
DEBUG 05-06 15:31:54.963823.963823 lmp.py:1516] ---- decode step 1 layer 7 ----
DEBUG 05-06 15:31:54.967606.967606 cuda_h.py:27] end decode_layer cost 4.785 ms
DEBUG 05-06 15:31:54.967780.967780 lmp.py:1516] ---- decode step 1 layer 8 ----
DEBUG 05-06 15:31:54.972798.972798 cuda_h.py:27] end decode_layer cost 4.887 ms
DEBUG 05-06 15:31:54.972164.972164 lmp.py:1516] ---- decode step 1 layer 9 ----
DEBUG 05-06 15:31:54.977631.977631 cuda_h.py:27] end decode_layer cost 4.833 ms
DEBUG 05-06 15:31:54.977136.977136 lmp.py:1516] ---- decode step 1 layer 10 ----
DEBUG 05-06 15:31:54.982029.982029 cuda_h.py:27] end decode_layer cost 4.726 ms
DEBUG 05-06 15:31:54.982535.982535 lmp.py:1516] ---- decode step 1 layer 11 ----
DEBUG 05-06 15:31:54.987795.987795 cuda_h.py:27] end decode_layer cost 4.996 ms
DEBUG 05-06 15:31:54.987929.987929 lmp.py:1516] ---- decode step 1 layer 12 ----
DEBUG 05-06 15:31:54.992203.992203 cuda_h.py:27] end decode_layer cost 4.795 ms
DEBUG 05-06 15:31:54.992284.992284 lmp.py:1516] ---- decode step 1 layer 13 ----
DEBUG 05-06 15:31:54.997621.997621 cuda_h.py:27] end decode_layer cost 4.913 ms
DEBUG 05-06 15:31:54.997126.997126 lmp.py:1516] ---- decode step 1 layer 14 ----
DEBUG 05-06 15:31:55.002652.002652 cuda_h.py:27] end decode_layer cost 4.805 ms
DEBUG 05-06 15:31:55.002555.002555 lmp.py:1516] ---- decode step 1 layer 15 ----
DEBUG 05-06 15:31:55.007693.007693 cuda_h.py:27] end decode_layer cost 4.906 ms
DEBUG 05-06 15:31:55.007059.007059 lmp.py:1516] ---- decode step 1 layer 16 ----
DEBUG 05-06 15:31:55.012893.012893 cuda_h.py:27] end decode_layer cost 4.717 ms
DEBUG 05-06 15:31:55.012212.012212 lmp.py:1516] ---- decode step 1 layer 17 ----
DEBUG 05-06 15:31:55.017289.017289 cuda_h.py:27] end decode_layer cost 5.036 ms
DEBUG 05-06 15:31:55.017893.017893 lmp.py:1516] ---- decode step 1 layer 18 ----
DEBUG 05-06 15:31:55.021940.021940 cuda_h.py:27] end decode_layer cost 4.769 ms
DEBUG 05-06 15:31:55.022969.022969 lmp.py:1516] ---- decode step 1 layer 19 ----
DEBUG 05-06 15:31:55.026302.026302 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 15:31:55.026092.026092 lmp.py:1516] ---- decode step 1 layer 20 ----
DEBUG 05-06 15:31:55.031467.031467 cuda_h.py:27] end decode_layer cost 4.870 ms
DEBUG 05-06 15:31:55.031310.031310 lmp.py:1516] ---- decode step 1 layer 21 ----
DEBUG 05-06 15:31:55.036606.036606 cuda_h.py:27] end decode_layer cost 4.882 ms
DEBUG 05-06 15:31:55.036211.036211 lmp.py:1516] ---- decode step 1 layer 22 ----
DEBUG 05-06 15:31:55.041908.041908 cuda_h.py:27] end decode_layer cost 4.792 ms
DEBUG 05-06 15:31:55.041082.041082 lmp.py:1516] ---- decode step 1 layer 23 ----
DEBUG 05-06 15:31:55.046075.046075 cuda_h.py:27] end decode_layer cost 5.115 ms
DEBUG 05-06 15:31:55.046819.046819 lmp.py:1516] ---- decode step 1 layer 24 ----
DEBUG 05-06 15:31:55.051832.051832 cuda_h.py:27] end decode_layer cost 4.744 ms
DEBUG 05-06 15:31:55.051610.051610 lmp.py:1516] ---- decode step 1 layer 25 ----
DEBUG 05-06 15:31:55.056462.056462 cuda_h.py:27] end decode_layer cost 4.887 ms
DEBUG 05-06 15:31:55.056027.056027 lmp.py:1516] ---- decode step 1 layer 26 ----
DEBUG 05-06 15:31:55.061485.061485 cuda_h.py:27] end decode_layer cost 4.792 ms
DEBUG 05-06 15:31:55.061706.061706 lmp.py:1516] ---- decode step 1 layer 27 ----
DEBUG 05-06 15:31:55.066730.066730 cuda_h.py:27] end decode_layer cost 4.857 ms
DEBUG 05-06 15:31:55.066334.066334 lmp.py:1516] ---- decode step 1 layer 28 ----
DEBUG 05-06 15:31:55.071827.071827 cuda_h.py:27] end decode_layer cost 4.817 ms
DEBUG 05-06 15:31:55.071047.071047 lmp.py:1516] ---- decode step 1 layer 29 ----
DEBUG 05-06 15:31:55.076309.076309 cuda_h.py:27] end decode_layer cost 5.032 ms
DEBUG 05-06 15:31:55.076616.076616 cuda_h.py:27] end decode_step cost 155.927 ms
INFO 05-06 15:31:55.076809.076809 lmp.py:1564] decode step 1 time: 0.15596652030944824 seconds
WARNING 05-06 15:31:55.076311.076311 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 15:31:55.076571.076571 helper.py:35]   NaN count (hidden): 11264
WARNING 05-06 15:31:55.077384.077384 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 15:31:55.077334.077334 helper.py:39]   NaN count (normed): 11264
WARNING 05-06 15:31:55.082947.082947 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 15:31:55.082249.082249 helper.py:50]   NaN count: 1048576
WARNING 05-06 15:31:55.082487.082487 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 15:31:55.084825.084825 cuda_h.py:27] end init_inputs_tokens cost 7.524 ms
DEBUG 05-06 15:31:55.084907.084907 lmp.py:1510] decode step 2 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 15:31:55.084961.084961 lmp.py:1516] ---- decode step 2 layer 0 ----
DEBUG 05-06 15:31:55.088520.088520 cuda_h.py:27] end decode_layer cost 4.829 ms
DEBUG 05-06 15:31:55.088602.088602 lmp.py:1516] ---- decode step 2 layer 1 ----
DEBUG 05-06 15:31:55.093792.093792 cuda_h.py:27] end decode_layer cost 4.874 ms
DEBUG 05-06 15:31:55.093728.093728 lmp.py:1516] ---- decode step 2 layer 2 ----
DEBUG 05-06 15:31:55.098651.098651 cuda_h.py:27] end decode_layer cost 4.818 ms
DEBUG 05-06 15:31:55.098977.098977 lmp.py:1516] ---- decode step 2 layer 3 ----
DEBUG 05-06 15:31:55.103627.103627 cuda_h.py:27] end decode_layer cost 4.757 ms
DEBUG 05-06 15:31:55.103928.103928 lmp.py:1516] ---- decode step 2 layer 4 ----
DEBUG 05-06 15:31:55.108142.108142 cuda_h.py:27] end decode_layer cost 4.822 ms
DEBUG 05-06 15:31:55.108985.108985 lmp.py:1516] ---- decode step 2 layer 5 ----
DEBUG 05-06 15:31:55.113525.113525 cuda_h.py:27] end decode_layer cost 5.027 ms
DEBUG 05-06 15:31:55.113553.113553 lmp.py:1516] ---- decode step 2 layer 6 ----
DEBUG 05-06 15:31:55.118623.118623 cuda_h.py:27] end decode_layer cost 4.856 ms
DEBUG 05-06 15:31:55.118652.118652 lmp.py:1516] ---- decode step 2 layer 7 ----
DEBUG 05-06 15:31:55.123454.123454 cuda_h.py:27] end decode_layer cost 4.764 ms
DEBUG 05-06 15:31:55.123244.123244 lmp.py:1516] ---- decode step 2 layer 8 ----
DEBUG 05-06 15:31:55.128634.128634 cuda_h.py:27] end decode_layer cost 4.741 ms
DEBUG 05-06 15:31:55.128285.128285 lmp.py:1516] ---- decode step 2 layer 9 ----
DEBUG 05-06 15:31:55.133415.133415 cuda_h.py:27] end decode_layer cost 4.865 ms
DEBUG 05-06 15:31:55.133920.133920 lmp.py:1516] ---- decode step 2 layer 10 ----
DEBUG 05-06 15:31:55.137493.137493 cuda_h.py:27] end decode_layer cost 4.840 ms
DEBUG 05-06 15:31:55.138191.138191 lmp.py:1516] ---- decode step 2 layer 11 ----
DEBUG 05-06 15:31:55.143129.143129 cuda_h.py:27] end decode_layer cost 5.074 ms
DEBUG 05-06 15:31:55.143349.143349 lmp.py:1516] ---- decode step 2 layer 12 ----
DEBUG 05-06 15:31:55.147140.147140 cuda_h.py:27] end decode_layer cost 4.826 ms
DEBUG 05-06 15:31:55.148268.148268 lmp.py:1516] ---- decode step 2 layer 13 ----
DEBUG 05-06 15:31:55.152439.152439 cuda_h.py:27] end decode_layer cost 4.895 ms
DEBUG 05-06 15:31:55.153183.153183 lmp.py:1516] ---- decode step 2 layer 14 ----
DEBUG 05-06 15:31:55.157733.157733 cuda_h.py:27] end decode_layer cost 4.754 ms
DEBUG 05-06 15:31:55.157006.157006 lmp.py:1516] ---- decode step 2 layer 15 ----
DEBUG 05-06 15:31:55.162729.162729 cuda_h.py:27] end decode_layer cost 4.776 ms
DEBUG 05-06 15:31:55.162758.162758 lmp.py:1516] ---- decode step 2 layer 16 ----
DEBUG 05-06 15:31:55.167677.167677 cuda_h.py:27] end decode_layer cost 4.710 ms
DEBUG 05-06 15:31:55.167898.167898 lmp.py:1516] ---- decode step 2 layer 17 ----
DEBUG 05-06 15:31:55.172595.172595 cuda_h.py:27] end decode_layer cost 5.002 ms
DEBUG 05-06 15:31:55.172484.172484 lmp.py:1516] ---- decode step 2 layer 18 ----
DEBUG 05-06 15:31:55.177300.177300 cuda_h.py:27] end decode_layer cost 4.774 ms
DEBUG 05-06 15:31:55.177282.177282 lmp.py:1516] ---- decode step 2 layer 19 ----
DEBUG 05-06 15:31:55.182623.182623 cuda_h.py:27] end decode_layer cost 4.845 ms
DEBUG 05-06 15:31:55.182367.182367 lmp.py:1516] ---- decode step 2 layer 20 ----
DEBUG 05-06 15:31:55.187600.187600 cuda_h.py:27] end decode_layer cost 4.766 ms
DEBUG 05-06 15:31:55.187582.187582 lmp.py:1516] ---- decode step 2 layer 21 ----
DEBUG 05-06 15:31:55.191788.191788 cuda_h.py:27] end decode_layer cost 4.781 ms
DEBUG 05-06 15:31:55.191678.191678 lmp.py:1516] ---- decode step 2 layer 22 ----
DEBUG 05-06 15:31:55.196333.196333 cuda_h.py:27] end decode_layer cost 4.725 ms
DEBUG 05-06 15:31:55.196461.196461 lmp.py:1516] ---- decode step 2 layer 23 ----
DEBUG 05-06 15:31:55.201967.201967 cuda_h.py:27] end decode_layer cost 5.038 ms
DEBUG 05-06 15:31:55.201472.201472 lmp.py:1516] ---- decode step 2 layer 24 ----
DEBUG 05-06 15:31:55.206384.206384 cuda_h.py:27] end decode_layer cost 4.880 ms
DEBUG 05-06 15:31:55.206704.206704 lmp.py:1516] ---- decode step 2 layer 25 ----
DEBUG 05-06 15:31:55.211427.211427 cuda_h.py:27] end decode_layer cost 4.776 ms
DEBUG 05-06 15:31:55.211946.211946 lmp.py:1516] ---- decode step 2 layer 26 ----
DEBUG 05-06 15:31:55.216208.216208 cuda_h.py:27] end decode_layer cost 4.857 ms
DEBUG 05-06 15:31:55.216528.216528 lmp.py:1516] ---- decode step 2 layer 27 ----
DEBUG 05-06 15:31:55.221963.221963 cuda_h.py:27] end decode_layer cost 4.880 ms
DEBUG 05-06 15:31:55.221283.221283 lmp.py:1516] ---- decode step 2 layer 28 ----
DEBUG 05-06 15:31:55.226611.226611 cuda_h.py:27] end decode_layer cost 4.836 ms
DEBUG 05-06 15:31:55.226692.226692 lmp.py:1516] ---- decode step 2 layer 29 ----
DEBUG 05-06 15:31:55.231784.231784 cuda_h.py:27] end decode_layer cost 5.118 ms
DEBUG 05-06 15:31:55.231888.231888 cuda_h.py:27] end decode_step cost 155.151 ms
INFO 05-06 15:31:55.231511.231511 lmp.py:1564] decode step 2 time: 0.15519261360168457 seconds
Time taken: 7.650959748774767 seconds
generate input ids cost 0.03805661201477051 s
DEBUG 05-06 15:31:58.048273.048273 cuda_h.py:27] end generate_input_ids cost 2695.579 ms
DEBUG 05-06 15:31:58.048231.048231 cuda_h.py:27] end init_cache cost 0.032 ms
INFO 05-06 15:31:58.048216.048216 lmp.py:1162] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 15:31:58.061990.061990 lmp.py:2797] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 4740325316, 'cuda:1': 12875595776, 'cuda:2': 12875595776, 'cuda:3': 12875595776} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7886239265315211, 'cuda:1': 0.4700220660037874, 'cuda:2': 0.4700220660037874, 'cuda:3': 0.4700220660037874}
INFO 05-06 15:31:58.061133.061133 lmp.py:2815] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.061055.061055 lmp.py:2815] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.061824.061824 lmp.py:2815] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.061409.061409 lmp.py:2815] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.061770.061770 lmp.py:2815] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.061870.061870 lmp.py:2815] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.061825.061825 lmp.py:2815] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.061611.061611 lmp.py:2815] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.062996.062996 lmp.py:2815] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.062724.062724 lmp.py:2815] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.062871.062871 lmp.py:2815] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.062773.062773 lmp.py:2815] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.062444.062444 lmp.py:2815] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.062947.062947 lmp.py:2815] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.062617.062617 lmp.py:2815] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.062995.062995 lmp.py:2815] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.062665.062665 lmp.py:2815] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.063784.063784 lmp.py:2815] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.063693.063693 lmp.py:2815] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.063235.063235 lmp.py:2815] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.063382.063382 lmp.py:2815] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.063985.063985 lmp.py:2815] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.063370.063370 lmp.py:2815] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.063848.063848 lmp.py:2815] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.063901.063901 lmp.py:2815] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.063525.063525 lmp.py:2815] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.064153.064153 lmp.py:2815] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.064777.064777 lmp.py:2815] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.064644.064644 lmp.py:2815] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:31:58.064029.064029 lmp.py:2815] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 15:31:58.360405.360405 cuda_h.py:27] end init_loading_placement cost 311.039 ms
DEBUG 05-06 15:31:58.360959.360959 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 15:31:58.360934.360934 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 15:31:58 client.py:72] load_into_gpu: gemma4-26B-A4B, 9ba52a9b-8cba-4419-abfa-3b0b2547a07e
INFO 05-06 15:31:58 client.py:135] Model loaded: gemma4-26B-A4B, 9ba52a9b-8cba-4419-abfa-3b0b2547a07e
INFO 05-06 15:31:58 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 9ba52a9b-8cba-4419-abfa-3b0b2547a07e
INFO 05-06 15:31:58 client.py:212] Model loaded
DEBUG 05-06 15:31:58.888771.888771 cuda_h.py:27] end init_general_sagl_loading_async cost 528.603 ms
INFO 05-06 15:31:58.941059.941059 lmp.py:3318] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 15:31:59.042974.042974 cuda_h.py:27] end restore_state_dict cost 100.640 ms
INFO 05-06 15:31:59.045286.045286 lmp.py:1291] vLLM Triton pre-warmup done in 2.2 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 15:31:59.045110.045110 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 15:31:59.045344.045344 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 15:31:59 client.py:72] load_into_gpu: gemma4-26B-A4B, ca3d5fab-d3e4-4eb0-9b56-4acd6a741070
INFO 05-06 15:31:59 client.py:135] Model loaded: gemma4-26B-A4B, ca3d5fab-d3e4-4eb0-9b56-4acd6a741070
DEBUG 05-06 15:31:59.173320.173320 cuda_h.py:27] end init_experts_loading_async cost 128.463 ms
DEBUG 05-06 15:31:59.175116.175116 cuda_h.py:27] end init_inputs_tokens cost 1.279 ms
DEBUG 05-06 15:31:59.175903.175903 lmp.py:1350] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 15:31:59.176229.176229 cuda_h.py:27] end prefill_ln cost 0.733 ms
DEBUG 05-06 15:31:59.181618.181618 cuda_h.py:27] end prefill_attn cost 5.560 ms
DEBUG 05-06 15:31:59.182963.182963 cuda_h.py:27] end prefill_ffn_prep cost 0.825 ms
DEBUG 05-06 15:31:59.185485.185485 cuda_h.py:27] end prefill_gate cost 1.112 ms
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 75, 115, 83, 127, 31, 3, 67, 51, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 92, 116, 112, 64, 72, 48, 32, 52, 101, 109, 85, 49, 45, 65, 93, 69, 5, 13, 9, 73, 77, 37, 89, 25, 105, 117, 125, 41, 113, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18, 70, 110, 118, 122, 78, 26, 54, 74], 'token_total': 1502, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 75: 29, 115: 29, 83: 33, 127: 33, 31: 34, 3: 46, 67: 47, 51: 48, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 92: 16, 116: 18, 112: 23, 64: 27, 72: 35, 48: 41, 32: 43, 52: 43, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 13: 16, 9: 17, 73: 19, 77: 19, 37: 20, 89: 20, 25: 24, 105: 24, 117: 26, 125: 26, 41: 27, 113: 39, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18, 70: 25, 110: 27, 118: 29, 122: 35, 78: 36, 26: 59, 54: 59, 74: 61}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 39, 47, 55, 91, 103], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 28, 'token_total': 917, 'token_per_expert': {7: 95, 39: 176, 47: 318, 55: 51, 91: 99, 103: 178}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 68, 104, 124], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 26, 'token_total': 512, 'token_per_expert': {0: 73, 16: 48, 68: 170, 104: 43, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 33, 53, 121], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 26, 'token_total': 603, 'token_per_expert': {1: 75, 21: 48, 33: 210, 53: 205, 121: 65}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 46, 50, 90, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 25, 'token_total': 562, 'token_per_expert': {22: 64, 46: 119, 50: 110, 90: 154, 126: 115}}
INFO 05-06 15:31:59.187039.187039 lmp.py:1839] [layer_moe_fused] layer=0 prefix: 0.970ms alloc: 0.338ms
INFO 05-06 15:31:59.187294.187294 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.7179718017578125e-05 seconds
INFO 05-06 15:31:59.188887.188887 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=42 time: 0.0014421939849853516s
INFO 05-06 15:31:59.189389.189389 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.000690460205078125 seconds
DEBUG 05-06 15:31:59.189717.189717 cuda_h.py:27] end moe_cpu_prep_submit cost 0.793 ms
INFO 05-06 15:31:59.310784.310784 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.12019014358520508s
DEBUG 05-06 15:31:59.310242.310242 cuda_h.py:27] end moe_wait_copy_tasks cost 120.448 ms
DEBUG 05-06 15:31:59.325157.325157 cuda_h.py:27] end moe_vllm_forward cost 14.071 ms
DEBUG 05-06 15:31:59.325628.325628 cuda_h.py:27] end moe_cpu_merge cost 0.060 ms
DEBUG 05-06 15:31:59.325790.325790 cuda_h.py:27] end moe_shared_experts cost 0.010 ms
INFO 05-06 15:31:59.326857.326857 lmp.py:1953] [layer_moe_fused] vllm triton time: 15.384ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.326221.326221 cuda_h.py:27] end *layer_moe_fused cost 140.481 ms
DEBUG 05-06 15:31:59.326296.326296 cuda_h.py:27] end prefill_merge_scale cost 0.415 ms
DEBUG 05-06 15:31:59.327499.327499 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.048 ms
DEBUG 05-06 15:31:59.327518.327518 cuda_h.py:27] end prefill_layer cost 151.902 ms
DEBUG 05-06 15:31:59.327444.327444 lmp.py:1394] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 15:31:59.327816.327816 lmp.py:1350] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 15:31:59.327265.327265 cuda_h.py:27] end prefill_ln cost 0.212 ms
DEBUG 05-06 15:31:59.330504.330504 cuda_h.py:27] end prefill_attn cost 2.328 ms
DEBUG 05-06 15:31:59.330941.330941 cuda_h.py:27] end prefill_ffn_prep cost 0.325 ms
DEBUG 05-06 15:31:59.331932.331932 cuda_h.py:27] end prefill_gate cost 0.352 ms
experts_cpu_alloc {'expert_ids': [63, 107, 111, 23, 39, 87, 35, 91, 3, 55, 7, 31, 127, 83, 67, 95, 71, 15, 11, 123, 103, 115, 27, 28, 36, 44, 32, 48, 88, 112, 24, 76, 56, 16, 40, 120, 108, 116, 68, 4, 84, 92, 52, 60, 124, 12, 100, 72, 96, 61, 33, 57, 1, 125, 69, 105, 21, 45, 5, 77, 117, 121, 41, 101, 81, 29, 49, 37, 73, 53, 65, 85, 89, 2, 62, 114, 86, 18, 54, 66, 38, 78, 102, 10, 34, 90, 110, 74, 26, 50, 42, 94, 122, 46, 106], 'token_total': 1986, 'token_per_expert': {63: 1, 107: 1, 111: 2, 23: 3, 39: 4, 87: 5, 35: 8, 91: 8, 3: 10, 55: 10, 7: 12, 31: 12, 127: 14, 83: 15, 67: 17, 95: 19, 71: 22, 15: 24, 11: 25, 123: 27, 103: 32, 115: 36, 27: 47, 28: 1, 36: 1, 44: 2, 32: 3, 48: 3, 88: 3, 112: 3, 24: 5, 76: 5, 56: 6, 16: 8, 40: 13, 120: 15, 108: 16, 116: 19, 68: 23, 4: 24, 84: 25, 92: 29, 52: 31, 60: 32, 124: 33, 12: 39, 100: 42, 72: 47, 96: 52, 61: 1, 33: 2, 57: 5, 1: 10, 125: 10, 69: 13, 105: 14, 21: 15, 45: 15, 5: 16, 77: 16, 117: 16, 121: 18, 41: 19, 101: 19, 81: 20, 29: 22, 49: 24, 37: 31, 73: 34, 53: 35, 65: 44, 85: 50, 89: 50, 2: 1, 62: 3, 114: 6, 86: 8, 18: 12, 54: 12, 66: 16, 38: 18, 78: 18, 102: 23, 10: 31, 34: 32, 90: 32, 110: 35, 74: 36, 26: 37, 50: 39, 42: 42, 94: 56, 122: 64, 46: 65, 106: 67}}
experts_gpu_alloc_device_0 {'expert_ids': [47, 51, 59, 79, 99, 119], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 29, 'token_total': 427, 'token_per_expert': {47: 86, 51: 58, 59: 55, 79: 63, 99: 102, 119: 63}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 20, 64, 80, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 32, 'token_total': 677, 'token_per_expert': {0: 54, 8: 160, 20: 155, 64: 87, 80: 162, 104: 59}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 13, 25, 93, 97, 109], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 30, 'token_total': 460, 'token_per_expert': {9: 62, 13: 125, 25: 55, 93: 56, 97: 107, 109: 55}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 22, 30, 82, 98, 118], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 28, 'token_total': 546, 'token_per_expert': {6: 80, 22: 127, 30: 88, 82: 100, 98: 77, 118: 74}}
INFO 05-06 15:31:59.332722.332722 lmp.py:1839] [layer_moe_fused] layer=1 prefix: 0.369ms alloc: 0.256ms
INFO 05-06 15:31:59.332136.332136 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.2649765014648438e-05 seconds
INFO 05-06 15:31:59.333817.333817 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0015020370483398438s
INFO 05-06 15:31:59.335069.335069 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.001230478286743164 seconds
DEBUG 05-06 15:31:59.335049.335049 cuda_h.py:27] end moe_cpu_prep_submit cost 1.415 ms
INFO 05-06 15:31:59.341583.341583 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.005376338958740234s
DEBUG 05-06 15:31:59.341996.341996 cuda_h.py:27] end moe_wait_copy_tasks cost 5.668 ms
DEBUG 05-06 15:31:59.350046.350046 cuda_h.py:27] end moe_vllm_forward cost 8.142 ms
DEBUG 05-06 15:31:59.361190.361190 cuda_h.py:27] end moe_cpu_merge cost 10.899 ms
DEBUG 05-06 15:31:59.361703.361703 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 15:31:59.361103.361103 lmp.py:1953] [layer_moe_fused] vllm triton time: 20.268ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.362571.362571 cuda_h.py:27] end *layer_moe_fused cost 30.712 ms
DEBUG 05-06 15:31:59.362833.362833 cuda_h.py:27] end prefill_merge_scale cost 0.454 ms
DEBUG 05-06 15:31:59.363214.363214 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.043 ms
DEBUG 05-06 15:31:59.363786.363786 cuda_h.py:27] end prefill_layer cost 35.895 ms
DEBUG 05-06 15:31:59.363024.363024 lmp.py:1394] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 15:31:59.363833.363833 lmp.py:1350] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 15:31:59.363826.363826 cuda_h.py:27] end prefill_ln cost 0.212 ms
DEBUG 05-06 15:31:59.365413.365413 cuda_h.py:27] end prefill_attn cost 1.844 ms
DEBUG 05-06 15:31:59.366316.366316 cuda_h.py:27] end prefill_ffn_prep cost 0.378 ms
DEBUG 05-06 15:31:59.367654.367654 cuda_h.py:27] end prefill_gate cost 0.419 ms
experts_cpu_alloc {'expert_ids': [23, 31, 47, 67, 27, 119, 51, 103, 55, 63, 107, 127, 115, 91, 123, 71, 95, 15, 99, 87, 83, 59, 12, 24, 92, 36, 40, 116, 8, 16, 68, 88, 104, 120, 44, 52, 100, 56, 20, 84, 48, 96, 124, 60, 28, 76, 57, 105, 61, 93, 113, 21, 33, 53, 117, 77, 97, 45, 73, 125, 49, 17, 69, 85, 41, 37, 13, 65, 70, 10, 38, 66, 74, 82, 98, 26, 54, 46, 126, 118, 14, 18, 114, 122, 58, 90, 50, 106, 110, 42], 'token_total': 930, 'token_per_expert': {23: 1, 31: 1, 47: 1, 67: 2, 27: 4, 119: 4, 51: 6, 103: 7, 55: 8, 63: 8, 107: 10, 127: 10, 115: 12, 91: 14, 123: 15, 71: 21, 95: 27, 15: 28, 99: 28, 87: 30, 83: 36, 59: 46, 12: 1, 24: 1, 92: 1, 36: 3, 40: 3, 116: 3, 8: 4, 16: 4, 68: 5, 88: 5, 104: 5, 120: 7, 44: 9, 52: 9, 100: 9, 56: 11, 20: 12, 84: 12, 48: 13, 96: 14, 124: 14, 60: 19, 28: 20, 76: 23, 57: 1, 105: 1, 61: 2, 93: 2, 113: 2, 21: 3, 33: 3, 53: 3, 117: 4, 77: 5, 97: 6, 45: 8, 73: 12, 125: 13, 49: 16, 17: 18, 69: 18, 85: 18, 41: 20, 37: 25, 13: 27, 65: 32, 70: 1, 10: 2, 38: 2, 66: 2, 74: 2, 82: 2, 98: 2, 26: 4, 54: 4, 46: 5, 126: 5, 118: 6, 14: 7, 18: 8, 114: 8, 122: 9, 58: 10, 90: 11, 50: 13, 106: 19, 110: 21, 42: 22}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 35, 43], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 28, 'token_total': 774, 'token_per_expert': {3: 272, 7: 260, 11: 70, 19: 51, 35: 70, 43: 51}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 64, 72, 80, 108], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 30, 'token_total': 782, 'token_per_expert': {0: 299, 4: 265, 64: 24, 72: 56, 80: 42, 108: 96}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 29, 81, 109], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 28, 'token_total': 955, 'token_per_expert': {1: 420, 5: 259, 9: 105, 29: 34, 81: 90, 109: 47}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34, 62, 102], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 27, 'token_total': 655, 'token_per_expert': {2: 256, 6: 256, 34: 73, 62: 28, 102: 42}}
INFO 05-06 15:31:59.368447.368447 lmp.py:1839] [layer_moe_fused] layer=2 prefix: 0.452ms alloc: 0.393ms
INFO 05-06 15:31:59.368364.368364 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 3.123283386230469e-05 seconds
INFO 05-06 15:31:59.369210.369210 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009362697601318359s
INFO 05-06 15:31:59.370098.370098 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005602836608886719 seconds
DEBUG 05-06 15:31:59.370862.370862 cuda_h.py:27] end moe_cpu_prep_submit cost 0.817 ms
INFO 05-06 15:31:59.372146.372146 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017635822296142578s
DEBUG 05-06 15:31:59.372941.372941 cuda_h.py:27] end moe_wait_copy_tasks cost 1.976 ms
DEBUG 05-06 15:31:59.377445.377445 cuda_h.py:27] end moe_vllm_forward cost 4.284 ms
DEBUG 05-06 15:31:59.383502.383502 cuda_h.py:27] end moe_cpu_merge cost 5.941 ms
DEBUG 05-06 15:31:59.383524.383524 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 15:31:59.384448.384448 lmp.py:1953] [layer_moe_fused] vllm triton time: 11.107ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.384128.384128 cuda_h.py:27] end *layer_moe_fused cost 17.048 ms
DEBUG 05-06 15:31:59.385569.385569 cuda_h.py:27] end prefill_merge_scale cost 0.445 ms
DEBUG 05-06 15:31:59.385658.385658 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.040 ms
DEBUG 05-06 15:31:59.385227.385227 cuda_h.py:27] end prefill_layer cost 21.848 ms
DEBUG 05-06 15:31:59.385278.385278 lmp.py:1394] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 15:31:59.385849.385849 lmp.py:1350] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 15:31:59.385602.385602 cuda_h.py:27] end prefill_ln cost 0.200 ms
DEBUG 05-06 15:31:59.387448.387448 cuda_h.py:27] end prefill_attn cost 1.862 ms
DEBUG 05-06 15:31:59.388575.388575 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 15:31:59.389868.389868 cuda_h.py:27] end prefill_gate cost 0.424 ms
experts_cpu_alloc {'expert_ids': [63, 31, 123, 111, 39, 79, 55, 103, 127, 59, 119, 35, 23, 87, 75, 43, 51, 19, 83, 11, 107, 32, 40, 116, 80, 12, 64, 104, 8, 24, 100, 68, 120, 108, 56, 20, 96, 44, 60, 88, 16, 57, 97, 105, 73, 89, 37, 101, 113, 121, 41, 49, 125, 21, 33, 109, 81, 9, 61, 117, 69, 25, 53, 17, 82, 106, 46, 74, 70, 78, 86, 98, 18, 34, 42, 118, 122, 38, 58, 110, 14, 30, 114, 10, 22, 50, 26, 66], 'token_total': 1013, 'token_per_expert': {63: 1, 31: 2, 123: 3, 111: 5, 39: 6, 79: 6, 55: 7, 103: 7, 127: 8, 59: 9, 119: 10, 35: 11, 23: 12, 87: 14, 75: 15, 43: 17, 51: 20, 19: 21, 83: 21, 11: 22, 107: 24, 32: 1, 40: 1, 116: 2, 80: 3, 12: 4, 64: 5, 104: 5, 8: 6, 24: 8, 100: 10, 68: 11, 120: 11, 108: 12, 56: 14, 20: 15, 96: 15, 44: 36, 60: 39, 88: 41, 16: 44, 57: 1, 97: 1, 105: 1, 73: 3, 89: 3, 37: 4, 101: 5, 113: 5, 121: 5, 41: 6, 49: 6, 125: 7, 21: 8, 33: 8, 109: 9, 81: 12, 9: 14, 61: 16, 117: 17, 69: 20, 25: 26, 53: 26, 17: 28, 82: 1, 106: 1, 46: 2, 74: 3, 70: 4, 78: 4, 86: 4, 98: 4, 18: 5, 34: 6, 42: 6, 118: 6, 122: 6, 38: 7, 58: 9, 110: 12, 14: 15, 30: 17, 114: 17, 10: 20, 22: 21, 50: 24, 26: 27, 66: 37}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 67, 71, 91], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 27, 'token_total': 699, 'token_per_expert': {3: 263, 7: 256, 15: 27, 67: 83, 71: 32, 91: 38}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 52, 84, 92], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 26, 'token_total': 900, 'token_per_expert': {0: 286, 4: 313, 28: 97, 52: 72, 84: 73, 92: 59}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 77, 85, 93], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 29, 'token_total': 745, 'token_per_expert': {1: 275, 5: 279, 29: 29, 77: 48, 85: 83, 93: 31}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 62, 94, 102], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 29, 'token_total': 739, 'token_per_expert': {2: 273, 6: 278, 62: 102, 94: 38, 102: 48}}
INFO 05-06 15:31:59.390745.390745 lmp.py:1839] [layer_moe_fused] layer=3 prefix: 0.425ms alloc: 0.385ms
INFO 05-06 15:31:59.390093.390093 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 3.170967102050781e-05 seconds
INFO 05-06 15:31:59.391673.391673 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.00092315673828125s
INFO 05-06 15:31:59.392389.392389 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005738735198974609 seconds
DEBUG 05-06 15:31:59.392078.392078 cuda_h.py:27] end moe_cpu_prep_submit cost 0.759 ms
INFO 05-06 15:31:59.400099.400099 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.007869958877563477s
DEBUG 05-06 15:31:59.400044.400044 cuda_h.py:27] end moe_wait_copy_tasks cost 8.021 ms
DEBUG 05-06 15:31:59.405716.405716 cuda_h.py:27] end moe_vllm_forward cost 3.987 ms
DEBUG 05-06 15:31:59.405447.405447 cuda_h.py:27] end moe_cpu_merge cost 0.583 ms
DEBUG 05-06 15:31:59.405887.405887 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 15:31:59.405280.405280 lmp.py:1953] [layer_moe_fused] vllm triton time: 5.222ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.406541.406541 cuda_h.py:27] end *layer_moe_fused cost 16.886 ms
DEBUG 05-06 15:31:59.406955.406955 cuda_h.py:27] end prefill_merge_scale cost 0.426 ms
DEBUG 05-06 15:31:59.406236.406236 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:59.407395.407395 cuda_h.py:27] end prefill_layer cost 21.644 ms
DEBUG 05-06 15:31:59.407540.407540 lmp.py:1394] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 15:31:59.407157.407157 lmp.py:1350] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 15:31:59.407513.407513 cuda_h.py:27] end prefill_ln cost 0.225 ms
DEBUG 05-06 15:31:59.412348.412348 cuda_h.py:27] end prefill_attn cost 4.696 ms
DEBUG 05-06 15:31:59.412721.412721 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 15:31:59.413868.413868 cuda_h.py:27] end prefill_gate cost 0.417 ms
experts_cpu_alloc {'expert_ids': [127, 79, 123, 43, 15, 107, 115, 111, 19, 103, 35, 59, 83, 51, 75, 47, 87, 71, 31, 63, 67, 12, 108, 44, 76, 96, 112, 68, 20, 64, 24, 40, 84, 92, 28, 72, 88, 36, 100, 116, 120, 124, 33, 65, 81, 97, 101, 117, 57, 69, 121, 45, 85, 109, 21, 17, 113, 29, 93, 49, 89, 105, 125, 77, 53, 14, 50, 66, 110, 18, 62, 82, 54, 86, 122, 98, 78, 106, 34, 38, 126, 30, 22, 94, 90], 'token_total': 1077, 'token_per_expert': {127: 1, 79: 2, 123: 3, 43: 5, 15: 9, 107: 9, 115: 10, 111: 15, 19: 16, 103: 20, 35: 21, 59: 21, 83: 22, 51: 23, 75: 24, 47: 25, 87: 26, 71: 27, 31: 34, 63: 53, 67: 55, 12: 1, 108: 1, 44: 2, 76: 2, 96: 2, 112: 3, 68: 5, 20: 6, 64: 6, 24: 7, 40: 10, 84: 10, 92: 17, 28: 18, 72: 18, 88: 19, 36: 23, 100: 26, 116: 26, 120: 27, 124: 33, 33: 2, 65: 2, 81: 2, 97: 2, 101: 2, 117: 2, 57: 3, 69: 3, 121: 3, 45: 4, 85: 5, 109: 5, 21: 8, 17: 10, 113: 15, 29: 16, 93: 16, 49: 20, 89: 20, 105: 20, 125: 20, 77: 21, 53: 22, 14: 1, 50: 1, 66: 1, 110: 1, 18: 3, 62: 4, 82: 4, 54: 6, 86: 6, 122: 8, 98: 9, 78: 10, 106: 10, 34: 11, 38: 11, 126: 13, 30: 15, 22: 16, 94: 19, 90: 22}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 55, 119], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 27, 'token_total': 925, 'token_per_expert': {3: 270, 7: 257, 23: 118, 27: 63, 55: 105, 119: 112}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 32, 56, 80], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 27, 'token_total': 727, 'token_per_expert': {0: 257, 4: 256, 8: 74, 32: 60, 56: 40, 80: 40}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 41, 61], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 28, 'token_total': 669, 'token_per_expert': {1: 321, 5: 273, 37: 23, 41: 30, 61: 22}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 70, 74], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 25, 'token_total': 698, 'token_per_expert': {2: 288, 6: 256, 26: 31, 70: 26, 74: 97}}
INFO 05-06 15:31:59.415638.415638 lmp.py:1839] [layer_moe_fused] layer=4 prefix: 0.425ms alloc: 0.375ms
INFO 05-06 15:31:59.415503.415503 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 3.0040740966796875e-05 seconds
INFO 05-06 15:31:59.416558.416558 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008325576782226562s
INFO 05-06 15:31:59.416162.416162 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005600452423095703 seconds
DEBUG 05-06 15:31:59.417608.417608 cuda_h.py:27] end moe_cpu_prep_submit cost 0.802 ms
INFO 05-06 15:31:59.440701.440701 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.02299022674560547s
DEBUG 05-06 15:31:59.440884.440884 cuda_h.py:27] end moe_wait_copy_tasks cost 23.144 ms
DEBUG 05-06 15:31:59.444433.444433 cuda_h.py:27] end moe_vllm_forward cost 3.719 ms
DEBUG 05-06 15:31:59.444961.444961 cuda_h.py:27] end moe_cpu_merge cost 0.059 ms
DEBUG 05-06 15:31:59.445311.445311 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.445460.445460 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.495ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.445845.445845 cuda_h.py:27] end *layer_moe_fused cost 31.293 ms
DEBUG 05-06 15:31:59.446351.446351 cuda_h.py:27] end prefill_merge_scale cost 0.427 ms
DEBUG 05-06 15:31:59.446070.446070 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:31:59.446806.446806 cuda_h.py:27] end prefill_layer cost 38.960 ms
DEBUG 05-06 15:31:59.446665.446665 lmp.py:1394] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 15:31:59.446520.446520 lmp.py:1350] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 15:31:59.446651.446651 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 15:31:59.455637.455637 cuda_h.py:27] end prefill_attn cost 8.425 ms
DEBUG 05-06 15:31:59.455606.455606 cuda_h.py:27] end prefill_ffn_prep cost 0.389 ms
DEBUG 05-06 15:31:59.456448.456448 cuda_h.py:27] end prefill_gate cost 0.404 ms
experts_cpu_alloc {'expert_ids': [19, 31, 59, 63, 23, 107, 15, 83, 43, 55, 87, 39, 51, 99, 119, 47, 67, 71, 20, 52, 76, 96, 64, 8, 60, 36, 124, 84, 32, 24, 48, 88, 92, 112, 44, 116, 68, 100, 104, 28, 120, 21, 29, 69, 49, 53, 117, 81, 93, 37, 77, 33, 41, 97, 61, 89, 13, 113, 57, 9, 17, 18, 30, 46, 34, 54, 78, 114, 10, 38, 102, 106, 98, 22, 70, 82, 86, 118, 42, 74, 14], 'token_total': 1022, 'token_per_expert': {19: 1, 31: 1, 59: 1, 63: 1, 23: 6, 107: 7, 15: 8, 83: 8, 43: 11, 55: 12, 87: 12, 39: 14, 51: 16, 99: 16, 119: 18, 47: 25, 67: 28, 71: 28, 20: 1, 52: 1, 76: 1, 96: 1, 64: 2, 8: 3, 60: 3, 36: 4, 124: 8, 84: 10, 32: 12, 24: 13, 48: 15, 88: 17, 92: 17, 112: 17, 44: 18, 116: 18, 68: 20, 100: 27, 104: 27, 28: 37, 120: 48, 21: 1, 29: 1, 69: 1, 49: 2, 53: 2, 117: 2, 81: 3, 93: 3, 37: 4, 77: 4, 33: 7, 41: 8, 97: 8, 61: 14, 89: 28, 13: 30, 113: 33, 57: 34, 9: 38, 17: 45, 18: 1, 30: 1, 46: 1, 34: 2, 54: 2, 78: 2, 114: 2, 10: 4, 38: 4, 102: 4, 106: 7, 98: 9, 22: 10, 70: 10, 82: 12, 86: 13, 118: 30, 42: 31, 74: 37, 14: 39}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 75, 111, 123, 127], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 24, 'token_total': 682, 'token_per_expert': {3: 256, 7: 291, 75: 28, 111: 42, 123: 34, 127: 31}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 56, 72], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 28, 'token_total': 853, 'token_per_expert': {0: 261, 4: 269, 16: 78, 56: 182, 72: 63}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 45, 73, 101], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 25, 'token_total': 808, 'token_per_expert': {1: 257, 5: 266, 45: 57, 73: 164, 101: 64}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 50, 94, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 25, 'token_total': 731, 'token_per_expert': {2: 270, 6: 272, 50: 83, 94: 53, 126: 53}}
INFO 05-06 15:31:59.457967.457967 lmp.py:1839] [layer_moe_fused] layer=5 prefix: 0.420ms alloc: 0.365ms
INFO 05-06 15:31:59.457354.457354 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 3.0040740966796875e-05 seconds
INFO 05-06 15:31:59.458213.458213 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006914138793945312s
INFO 05-06 15:31:59.459465.459465 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005469322204589844 seconds
DEBUG 05-06 15:31:59.459314.459314 cuda_h.py:27] end moe_cpu_prep_submit cost 0.780 ms
INFO 05-06 15:31:59.475520.475520 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015598535537719727s
DEBUG 05-06 15:31:59.475094.475094 cuda_h.py:27] end moe_wait_copy_tasks cost 15.759 ms
DEBUG 05-06 15:31:59.479693.479693 cuda_h.py:27] end moe_vllm_forward cost 3.618 ms
DEBUG 05-06 15:31:59.479168.479168 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 15:31:59.480080.480080 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:31:59.480466.480466 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.344ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.480030.480030 cuda_h.py:27] end *layer_moe_fused cost 23.449 ms
DEBUG 05-06 15:31:59.481629.481629 cuda_h.py:27] end prefill_merge_scale cost 0.426 ms
DEBUG 05-06 15:31:59.481924.481924 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.043 ms
DEBUG 05-06 15:31:59.481885.481885 cuda_h.py:27] end prefill_layer cost 34.818 ms
DEBUG 05-06 15:31:59.481605.481605 lmp.py:1394] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 15:31:59.481699.481699 lmp.py:1350] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 15:31:59.481280.481280 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 15:31:59.487661.487661 cuda_h.py:27] end prefill_attn cost 5.591 ms
DEBUG 05-06 15:31:59.487816.487816 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 15:31:59.489049.489049 cuda_h.py:27] end prefill_gate cost 0.414 ms
experts_cpu_alloc {'expert_ids': [39, 47, 63, 71, 15, 55, 11, 51, 91, 115, 127, 111, 27, 107, 23, 103, 123, 67, 95, 19, 119, 48, 116, 124, 8, 28, 24, 72, 104, 120, 88, 56, 60, 80, 32, 44, 16, 112, 96, 17, 33, 109, 117, 81, 113, 21, 45, 77, 57, 105, 9, 53, 89, 85, 121, 13, 61, 93, 25, 69, 102, 10, 38, 46, 78, 122, 82, 98, 114, 74, 70, 62, 30, 90, 110, 106, 50, 26, 42, 58, 18, 94, 22], 'token_total': 935, 'token_per_expert': {39: 1, 47: 1, 63: 1, 71: 1, 15: 2, 55: 3, 11: 5, 51: 5, 91: 6, 115: 7, 127: 9, 111: 13, 27: 14, 107: 16, 23: 24, 103: 24, 123: 24, 67: 30, 95: 31, 19: 34, 119: 38, 48: 1, 116: 1, 124: 2, 8: 3, 28: 3, 24: 4, 72: 4, 104: 4, 120: 4, 88: 5, 56: 8, 60: 8, 80: 9, 32: 11, 44: 11, 16: 12, 112: 12, 96: 17, 17: 1, 33: 1, 109: 1, 117: 1, 81: 3, 113: 3, 21: 4, 45: 4, 77: 6, 57: 7, 105: 7, 9: 10, 53: 10, 89: 10, 85: 11, 121: 11, 13: 19, 61: 19, 93: 23, 25: 27, 69: 37, 102: 1, 10: 2, 38: 2, 46: 2, 78: 2, 122: 2, 82: 3, 98: 4, 114: 4, 74: 6, 70: 9, 62: 12, 30: 13, 90: 13, 110: 13, 106: 18, 50: 20, 26: 21, 42: 21, 58: 25, 18: 26, 94: 42, 22: 51}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 75, 87, 99], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 27, 'token_total': 897, 'token_per_expert': {3: 263, 7: 260, 35: 57, 75: 48, 87: 84, 99: 185}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 100, 108], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 23, 'token_total': 802, 'token_per_expert': {0: 387, 4: 257, 36: 25, 100: 45, 108: 88}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 65, 101], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 26, 'token_total': 693, 'token_per_expert': {1: 259, 5: 287, 37: 46, 65: 38, 101: 63}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34, 86, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 28, 'token_total': 769, 'token_per_expert': {2: 273, 6: 277, 34: 61, 86: 103, 126: 55}}
INFO 05-06 15:31:59.490753.490753 lmp.py:1839] [layer_moe_fused] layer=6 prefix: 0.418ms alloc: 0.368ms
INFO 05-06 15:31:59.490048.490048 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.956390380859375e-05 seconds
INFO 05-06 15:31:59.491489.491489 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006880760192871094s
INFO 05-06 15:31:59.491669.491669 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005633831024169922 seconds
DEBUG 05-06 15:31:59.492553.492553 cuda_h.py:27] end moe_cpu_prep_submit cost 0.825 ms
INFO 05-06 15:31:59.503804.503804 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.011213541030883789s
DEBUG 05-06 15:31:59.503503.503503 cuda_h.py:27] end moe_wait_copy_tasks cost 11.359 ms
DEBUG 05-06 15:31:59.508402.508402 cuda_h.py:27] end moe_vllm_forward cost 3.872 ms
DEBUG 05-06 15:31:59.508976.508976 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 15:31:59.508695.508695 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.508466.508466 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.780ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.509990.509990 cuda_h.py:27] end *layer_moe_fused cost 19.750 ms
DEBUG 05-06 15:31:59.509059.509059 cuda_h.py:27] end prefill_merge_scale cost 0.417 ms
DEBUG 05-06 15:31:59.509817.509817 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:31:59.509738.509738 cuda_h.py:27] end prefill_layer cost 28.257 ms
DEBUG 05-06 15:31:59.509504.509504 lmp.py:1394] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 15:31:59.509836.509836 lmp.py:1350] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 15:31:59.510814.510814 cuda_h.py:27] end prefill_ln cost 0.195 ms
DEBUG 05-06 15:31:59.515797.515797 cuda_h.py:27] end prefill_attn cost 5.370 ms
DEBUG 05-06 15:31:59.516601.516601 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 15:31:59.517960.517960 cuda_h.py:27] end prefill_gate cost 0.426 ms
experts_cpu_alloc {'expert_ids': [75, 123, 63, 19, 115, 55, 67, 31, 99, 51, 27, 23, 47, 71, 111, 79, 107, 43, 36, 40, 48, 84, 124, 96, 108, 16, 116, 76, 8, 88, 20, 24, 12, 112, 32, 64, 44, 104, 56, 68, 120, 28, 97, 45, 9, 41, 21, 101, 13, 33, 25, 77, 17, 89, 105, 57, 61, 121, 113, 65, 125, 30, 38, 50, 78, 66, 126, 110, 18, 58, 26, 106, 42, 122, 118, 54, 74, 90, 14, 10, 86, 70], 'token_total': 1185, 'token_per_expert': {75: 1, 123: 2, 63: 3, 19: 9, 115: 9, 55: 10, 67: 10, 31: 11, 99: 12, 51: 13, 27: 14, 23: 16, 47: 23, 71: 24, 111: 26, 79: 31, 107: 47, 43: 58, 36: 1, 40: 1, 48: 1, 84: 1, 124: 1, 96: 2, 108: 2, 16: 3, 116: 4, 76: 7, 8: 8, 88: 8, 20: 9, 24: 10, 12: 11, 112: 12, 32: 14, 64: 14, 44: 15, 104: 17, 56: 20, 68: 24, 120: 25, 28: 26, 97: 2, 45: 7, 9: 8, 41: 8, 21: 9, 101: 11, 13: 14, 33: 14, 25: 16, 77: 16, 17: 17, 89: 17, 105: 17, 57: 24, 61: 24, 121: 25, 113: 26, 65: 32, 125: 38, 30: 1, 38: 1, 50: 1, 78: 2, 66: 4, 126: 4, 110: 5, 18: 9, 58: 9, 26: 10, 106: 12, 42: 17, 122: 17, 118: 19, 54: 21, 74: 21, 90: 25, 14: 30, 10: 32, 86: 32, 70: 33}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 83, 91, 103, 119], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 24, 'token_total': 810, 'token_per_expert': {3: 263, 7: 268, 83: 69, 91: 72, 103: 75, 119: 63}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 72, 80, 92], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 29, 'token_total': 625, 'token_per_expert': {0: 257, 4: 256, 72: 40, 80: 37, 92: 35}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 53, 69], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 24, 'token_total': 790, 'token_per_expert': {1: 256, 5: 262, 29: 44, 53: 51, 69: 177}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 82, 102, 114], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 26, 'token_total': 686, 'token_per_expert': {2: 257, 6: 260, 82: 43, 102: 87, 114: 39}}
INFO 05-06 15:31:59.518756.518756 lmp.py:1839] [layer_moe_fused] layer=7 prefix: 0.424ms alloc: 0.363ms
INFO 05-06 15:31:59.518475.518475 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.9087066650390625e-05 seconds
INFO 05-06 15:31:59.519950.519950 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007240772247314453s
INFO 05-06 15:31:59.519812.519812 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005757808685302734 seconds
DEBUG 05-06 15:31:59.520269.520269 cuda_h.py:27] end moe_cpu_prep_submit cost 0.753 ms
INFO 05-06 15:31:59.539527.539527 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.019222259521484375s
DEBUG 05-06 15:31:59.540042.540042 cuda_h.py:27] end moe_wait_copy_tasks cost 19.377 ms
DEBUG 05-06 15:31:59.544522.544522 cuda_h.py:27] end moe_vllm_forward cost 3.673 ms
DEBUG 05-06 15:31:59.544189.544189 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 15:31:59.544349.544349 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.544451.544451 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.478ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.545422.545422 cuda_h.py:27] end *layer_moe_fused cost 27.580 ms
DEBUG 05-06 15:31:59.545312.545312 cuda_h.py:27] end prefill_merge_scale cost 0.431 ms
DEBUG 05-06 15:31:59.545415.545415 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.043 ms
DEBUG 05-06 15:31:59.545422.545422 cuda_h.py:27] end prefill_layer cost 35.894 ms
DEBUG 05-06 15:31:59.545050.545050 lmp.py:1394] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 15:31:59.546620.546620 lmp.py:1350] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 15:31:59.546705.546705 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 15:31:59.552410.552410 cuda_h.py:27] end prefill_attn cost 6.357 ms
DEBUG 05-06 15:31:59.553120.553120 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 15:31:59.554843.554843 cuda_h.py:27] end prefill_gate cost 0.421 ms
experts_cpu_alloc {'expert_ids': [107, 11, 23, 43, 59, 31, 39, 127, 95, 119, 35, 91, 87, 103, 99, 71, 27, 123, 47, 111, 19, 15, 100, 96, 28, 108, 20, 68, 88, 104, 48, 52, 80, 116, 24, 36, 76, 56, 64, 12, 44, 25, 125, 65, 85, 121, 93, 81, 49, 41, 77, 69, 45, 113, 101, 9, 17, 29, 57, 61, 10, 74, 82, 34, 42, 62, 126, 14, 38, 102, 118, 106, 66, 70, 86, 98, 110, 46, 114], 'token_total': 1087, 'token_per_expert': {107: 1, 11: 2, 23: 2, 43: 2, 59: 2, 31: 3, 39: 3, 127: 3, 95: 4, 119: 4, 35: 5, 91: 6, 87: 11, 103: 11, 99: 13, 71: 18, 27: 19, 123: 22, 47: 23, 111: 23, 19: 26, 15: 28, 100: 1, 96: 2, 28: 4, 108: 4, 20: 5, 68: 5, 88: 6, 104: 6, 48: 10, 52: 10, 80: 16, 116: 17, 24: 22, 36: 23, 76: 24, 56: 30, 64: 38, 12: 57, 44: 62, 25: 1, 125: 1, 65: 2, 85: 2, 121: 2, 93: 3, 81: 6, 49: 7, 41: 9, 77: 11, 69: 12, 45: 14, 113: 14, 101: 19, 9: 21, 17: 26, 29: 27, 57: 27, 61: 34, 10: 3, 74: 3, 82: 4, 34: 6, 42: 7, 62: 8, 126: 8, 14: 12, 38: 12, 102: 12, 118: 13, 106: 16, 66: 19, 70: 19, 86: 20, 98: 20, 110: 28, 46: 33, 114: 33}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 51, 55, 75], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 27, 'token_total': 750, 'token_per_expert': {3: 290, 7: 258, 51: 76, 55: 33, 75: 93}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 32, 120, 124], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 24, 'token_total': 745, 'token_per_expert': {0: 262, 4: 273, 32: 66, 120: 75, 124: 69}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 53, 73, 105], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 24, 'token_total': 800, 'token_per_expert': {1: 266, 5: 300, 53: 46, 73: 154, 105: 34}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 50, 58, 122], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 24, 'token_total': 714, 'token_per_expert': {2: 280, 6: 260, 50: 43, 58: 65, 122: 66}}
INFO 05-06 15:31:59.555871.555871 lmp.py:1839] [layer_moe_fused] layer=8 prefix: 0.418ms alloc: 0.360ms
INFO 05-06 15:31:59.555067.555067 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.765655517578125e-05 seconds
INFO 05-06 15:31:59.556962.556962 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006897449493408203s
INFO 05-06 15:31:59.557420.557420 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005574226379394531 seconds
DEBUG 05-06 15:31:59.557992.557992 cuda_h.py:27] end moe_cpu_prep_submit cost 0.798 ms
INFO 05-06 15:31:59.569140.569140 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01198577880859375s
DEBUG 05-06 15:31:59.569648.569648 cuda_h.py:27] end moe_wait_copy_tasks cost 12.130 ms
DEBUG 05-06 15:31:59.574035.574035 cuda_h.py:27] end moe_vllm_forward cost 3.851 ms
DEBUG 05-06 15:31:59.574179.574179 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 15:31:59.574358.574358 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.574744.574744 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.633ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.574846.574846 cuda_h.py:27] end *layer_moe_fused cost 20.392 ms
DEBUG 05-06 15:31:59.575491.575491 cuda_h.py:27] end prefill_merge_scale cost 0.424 ms
DEBUG 05-06 15:31:59.575733.575733 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:59.575953.575953 cuda_h.py:27] end prefill_layer cost 29.703 ms
DEBUG 05-06 15:31:59.575958.575958 lmp.py:1394] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 15:31:59.575290.575290 lmp.py:1350] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 15:31:59.576944.576944 cuda_h.py:27] end prefill_ln cost 0.209 ms
DEBUG 05-06 15:31:59.582266.582266 cuda_h.py:27] end prefill_attn cost 6.005 ms
DEBUG 05-06 15:31:59.582347.582347 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 15:31:59.583309.583309 cuda_h.py:27] end prefill_gate cost 0.426 ms
experts_cpu_alloc {'expert_ids': [11, 43, 79, 115, 39, 27, 15, 67, 103, 75, 19, 83, 111, 127, 8, 96, 104, 120, 20, 84, 88, 112, 56, 36, 44, 68, 24, 32, 100, 108, 12, 76, 92, 124, 52, 17, 105, 113, 21, 73, 93, 33, 25, 117, 69, 81, 41, 45, 9, 125, 57, 29, 61, 13, 86, 22, 34, 14, 102, 18, 10, 122, 42, 70, 30, 74, 54, 62, 26, 82, 106, 66], 'token_total': 1198, 'token_per_expert': {11: 1, 43: 1, 79: 1, 115: 4, 39: 6, 27: 8, 15: 11, 67: 12, 103: 12, 75: 19, 19: 28, 83: 28, 111: 28, 127: 30, 8: 1, 96: 1, 104: 1, 120: 1, 20: 2, 84: 2, 88: 2, 112: 3, 56: 4, 36: 7, 44: 9, 68: 10, 24: 11, 32: 13, 100: 13, 108: 16, 12: 22, 76: 26, 92: 31, 124: 32, 52: 39, 17: 1, 105: 2, 113: 2, 21: 3, 73: 3, 93: 4, 33: 5, 25: 6, 117: 11, 69: 16, 81: 21, 41: 23, 45: 29, 9: 33, 125: 39, 57: 45, 29: 50, 61: 57, 13: 73, 86: 1, 22: 2, 34: 2, 14: 4, 102: 4, 18: 8, 10: 9, 122: 10, 42: 11, 70: 16, 30: 17, 74: 21, 54: 24, 62: 26, 26: 32, 82: 38, 106: 44, 66: 71}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 71, 95], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 19, 'token_total': 704, 'token_per_expert': {3: 328, 7: 257, 23: 32, 71: 43, 95: 44}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 48, 72], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 26, 'token_total': 720, 'token_per_expert': {0: 281, 4: 303, 16: 42, 48: 51, 72: 43}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 101], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 23, 'token_total': 802, 'token_per_expert': {1: 273, 5: 271, 37: 140, 101: 118}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 22, 'token_total': 672, 'token_per_expert': {2: 256, 6: 256, 38: 85, 46: 75}}
INFO 05-06 15:31:59.584269.584269 lmp.py:1839] [layer_moe_fused] layer=9 prefix: 0.405ms alloc: 0.325ms
INFO 05-06 15:31:59.584842.584842 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.765655517578125e-05 seconds
INFO 05-06 15:31:59.585879.585879 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007364749908447266s
INFO 05-06 15:31:59.586085.586085 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005483627319335938 seconds
DEBUG 05-06 15:31:59.586840.586840 cuda_h.py:27] end moe_cpu_prep_submit cost 0.740 ms
INFO 05-06 15:31:59.600022.600022 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01350855827331543s
DEBUG 05-06 15:31:59.600921.600921 cuda_h.py:27] end moe_wait_copy_tasks cost 13.662 ms
DEBUG 05-06 15:31:59.604443.604443 cuda_h.py:27] end moe_vllm_forward cost 3.664 ms
DEBUG 05-06 15:31:59.604348.604348 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 15:31:59.604975.604975 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:31:59.604408.604408 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.426ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.605045.605045 cuda_h.py:27] end *layer_moe_fused cost 21.355 ms
DEBUG 05-06 15:31:59.605313.605313 cuda_h.py:27] end prefill_merge_scale cost 0.425 ms
DEBUG 05-06 15:31:59.605370.605370 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:59.606038.606038 cuda_h.py:27] end prefill_layer cost 30.255 ms
DEBUG 05-06 15:31:59.606098.606098 lmp.py:1394] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 15:31:59.606861.606861 lmp.py:1350] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 15:31:59.606482.606482 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 15:31:59.611053.611053 cuda_h.py:27] end prefill_attn cost 4.714 ms
DEBUG 05-06 15:31:59.612923.612923 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 15:31:59.613479.613479 cuda_h.py:27] end prefill_gate cost 0.412 ms
experts_cpu_alloc {'expert_ids': [83, 91, 119, 67, 39, 127, 51, 103, 75, 79, 44, 84, 32, 108, 112, 8, 16, 120, 52, 80, 72, 124, 92, 88, 60, 29, 105, 13, 9, 17, 109, 125, 93, 121, 85, 57, 21, 113, 45, 41, 50, 82, 86, 106, 54, 34, 78, 110, 90, 70, 126, 74, 58, 46, 42, 14, 22, 66], 'token_total': 1199, 'token_per_expert': {83: 4, 91: 4, 119: 4, 67: 7, 39: 8, 127: 13, 51: 17, 103: 19, 75: 36, 79: 40, 44: 1, 84: 1, 32: 3, 108: 5, 112: 5, 8: 6, 16: 6, 120: 7, 52: 8, 80: 14, 72: 18, 124: 20, 92: 40, 88: 50, 60: 71, 29: 1, 105: 1, 13: 2, 9: 4, 17: 4, 109: 5, 125: 5, 93: 6, 121: 7, 85: 9, 57: 11, 21: 39, 113: 102, 45: 111, 41: 117, 50: 1, 82: 1, 86: 2, 106: 2, 54: 4, 34: 5, 78: 5, 110: 7, 90: 9, 70: 11, 126: 17, 74: 21, 58: 22, 46: 32, 42: 35, 14: 47, 22: 68, 66: 79}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 99], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 14, 'token_total': 661, 'token_per_expert': {3: 257, 7: 256, 31: 94, 99: 54}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 56, 100], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 703, 'token_per_expert': {0: 266, 4: 256, 56: 110, 100: 71}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 81], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 867, 'token_per_expert': {1: 263, 5: 256, 37: 176, 81: 172}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 21, 'token_total': 666, 'token_per_expert': {2: 257, 6: 256, 10: 153}}
INFO 05-06 15:31:59.613008.613008 lmp.py:1839] [layer_moe_fused] layer=10 prefix: 0.414ms alloc: 0.281ms
INFO 05-06 15:31:59.614766.614766 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.4318695068359375e-05 seconds
INFO 05-06 15:31:59.614466.614466 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006895065307617188s
INFO 05-06 15:31:59.615598.615598 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005307197570800781 seconds
DEBUG 05-06 15:31:59.615196.615196 cuda_h.py:27] end moe_cpu_prep_submit cost 0.779 ms
INFO 05-06 15:31:59.634924.634924 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.018198251724243164s
DEBUG 05-06 15:31:59.634048.634048 cuda_h.py:27] end moe_wait_copy_tasks cost 18.342 ms
DEBUG 05-06 15:31:59.638350.638350 cuda_h.py:27] end moe_vllm_forward cost 3.683 ms
DEBUG 05-06 15:31:59.638964.638964 cuda_h.py:27] end moe_cpu_merge cost 0.054 ms
DEBUG 05-06 15:31:59.638889.638889 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:31:59.638229.638229 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.379ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.639191.639191 cuda_h.py:27] end *layer_moe_fused cost 26.150 ms
DEBUG 05-06 15:31:59.639042.639042 cuda_h.py:27] end prefill_merge_scale cost 0.437 ms
DEBUG 05-06 15:31:59.640715.640715 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:31:59.640203.640203 cuda_h.py:27] end prefill_layer cost 33.748 ms
DEBUG 05-06 15:31:59.640923.640923 lmp.py:1394] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 15:31:59.640017.640017 lmp.py:1350] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 15:31:59.640658.640658 cuda_h.py:27] end prefill_ln cost 0.198 ms
DEBUG 05-06 15:31:59.647312.647312 cuda_h.py:27] end prefill_attn cost 6.600 ms
DEBUG 05-06 15:31:59.647890.647890 cuda_h.py:27] end prefill_ffn_prep cost 0.384 ms
DEBUG 05-06 15:31:59.648540.648540 cuda_h.py:27] end prefill_gate cost 0.405 ms
experts_cpu_alloc {'expert_ids': [103, 119, 19, 47, 67, 11, 71, 111, 31, 43, 59, 87, 79, 83, 32, 112, 52, 120, 56, 84, 8, 104, 124, 24, 100, 108, 16, 20, 68, 57, 73, 105, 21, 121, 113, 25, 29, 93, 89, 49, 77, 61, 69, 81, 22, 82, 38, 110, 34, 70, 18, 30, 50, 46, 66, 98, 42], 'token_total': 995, 'token_per_expert': {103: 1, 119: 1, 19: 2, 47: 2, 67: 3, 11: 9, 71: 12, 111: 17, 31: 18, 43: 18, 59: 19, 87: 26, 79: 39, 83: 55, 32: 1, 112: 1, 52: 2, 120: 2, 56: 5, 84: 7, 8: 8, 104: 8, 124: 10, 24: 13, 100: 15, 108: 20, 16: 36, 20: 59, 68: 62, 57: 1, 73: 1, 105: 1, 21: 2, 121: 3, 113: 4, 25: 7, 29: 9, 93: 14, 89: 16, 49: 17, 77: 18, 61: 27, 69: 36, 81: 52, 22: 2, 82: 5, 38: 11, 110: 12, 34: 13, 70: 14, 18: 16, 30: 20, 50: 28, 46: 31, 66: 46, 98: 50, 42: 68}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 684, 'token_per_expert': {3: 256, 7: 270, 23: 79, 27: 79}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 92], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 797, 'token_per_expert': {0: 308, 4: 263, 36: 160, 92: 66}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 117], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 898, 'token_per_expert': {1: 256, 5: 256, 17: 192, 117: 194}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 102], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 16, 'token_total': 722, 'token_per_expert': {2: 282, 6: 304, 102: 136}}
INFO 05-06 15:31:59.649393.649393 lmp.py:1839] [layer_moe_fused] layer=11 prefix: 0.409ms alloc: 0.278ms
INFO 05-06 15:31:59.649535.649535 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.5510787963867188e-05 seconds
INFO 05-06 15:31:59.650184.650184 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006871223449707031s
INFO 05-06 15:31:59.651469.651469 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.000537872314453125 seconds
DEBUG 05-06 15:31:59.651245.651245 cuda_h.py:27] end moe_cpu_prep_submit cost 0.778 ms
INFO 05-06 15:31:59.666066.666066 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014996528625488281s
DEBUG 05-06 15:31:59.667143.667143 cuda_h.py:27] end moe_wait_copy_tasks cost 15.139 ms
DEBUG 05-06 15:31:59.671305.671305 cuda_h.py:27] end moe_vllm_forward cost 3.649 ms
DEBUG 05-06 15:31:59.671926.671926 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 15:31:59.671533.671533 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.671966.671966 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.362ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.671213.671213 cuda_h.py:27] end *layer_moe_fused cost 22.818 ms
DEBUG 05-06 15:31:59.672726.672726 cuda_h.py:27] end prefill_merge_scale cost 0.428 ms
DEBUG 05-06 15:31:59.672736.672736 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.044 ms
DEBUG 05-06 15:31:59.672300.672300 cuda_h.py:27] end prefill_layer cost 32.370 ms
DEBUG 05-06 15:31:59.672458.672458 lmp.py:1394] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 15:31:59.672029.672029 lmp.py:1350] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 15:31:59.673935.673935 cuda_h.py:27] end prefill_ln cost 0.203 ms
DEBUG 05-06 15:31:59.679270.679270 cuda_h.py:27] end prefill_attn cost 6.190 ms
DEBUG 05-06 15:31:59.680186.680186 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 15:31:59.681141.681141 cuda_h.py:27] end prefill_gate cost 0.416 ms
experts_cpu_alloc {'expert_ids': [39, 79, 91, 15, 47, 83, 96, 104, 12, 20, 100, 64, 40, 48, 120, 116, 36, 80, 108, 84, 92, 93, 33, 41, 101, 13, 73, 25, 65, 117, 77, 85, 49, 53, 34, 98, 122, 38, 46, 90, 114, 58, 110, 22, 106, 102, 94, 50], 'token_total': 1316, 'token_per_expert': {39: 3, 79: 4, 91: 6, 15: 25, 47: 27, 83: 31, 96: 1, 104: 1, 12: 3, 20: 4, 100: 6, 64: 10, 40: 17, 48: 17, 120: 24, 116: 27, 36: 30, 80: 35, 108: 35, 84: 40, 92: 70, 93: 1, 33: 2, 41: 2, 101: 2, 13: 3, 73: 6, 25: 10, 65: 10, 117: 66, 77: 85, 85: 89, 49: 148, 53: 210, 34: 1, 98: 1, 122: 1, 38: 2, 46: 3, 90: 3, 114: 4, 58: 6, 110: 9, 22: 10, 106: 17, 102: 49, 94: 67, 50: 93}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 71], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 9, 'token_total': 704, 'token_per_expert': {3: 354, 7: 256, 71: 94}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 124], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 18, 'token_total': 583, 'token_per_expert': {0: 256, 4: 256, 124: 71}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 45], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 16, 'token_total': 815, 'token_per_expert': {1: 263, 5: 340, 45: 212}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 82], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 17, 'token_total': 678, 'token_per_expert': {2: 256, 6: 260, 82: 162}}
INFO 05-06 15:31:59.681708.681708 lmp.py:1839] [layer_moe_fused] layer=12 prefix: 0.403ms alloc: 0.244ms
INFO 05-06 15:31:59.682036.682036 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.09808349609375e-05 seconds
INFO 05-06 15:31:59.682251.682251 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006694793701171875s
INFO 05-06 15:31:59.683085.683085 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005211830139160156 seconds
DEBUG 05-06 15:31:59.683952.683952 cuda_h.py:27] end moe_cpu_prep_submit cost 0.663 ms
INFO 05-06 15:31:59.698162.698162 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014710426330566406s
DEBUG 05-06 15:31:59.698537.698537 cuda_h.py:27] end moe_wait_copy_tasks cost 14.864 ms
DEBUG 05-06 15:31:59.703864.703864 cuda_h.py:27] end moe_vllm_forward cost 3.791 ms
DEBUG 05-06 15:31:59.703723.703723 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 15:31:59.703469.703469 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.703955.703955 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.518ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.703877.703877 cuda_h.py:27] end *layer_moe_fused cost 22.510 ms
DEBUG 05-06 15:31:59.704310.704310 cuda_h.py:27] end prefill_merge_scale cost 0.438 ms
DEBUG 05-06 15:31:59.704691.704691 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.044 ms
DEBUG 05-06 15:31:59.704884.704884 cuda_h.py:27] end prefill_layer cost 31.668 ms
DEBUG 05-06 15:31:59.704551.704551 lmp.py:1394] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 15:31:59.704122.704122 lmp.py:1350] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 15:31:59.705875.705875 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 15:31:59.710176.710176 cuda_h.py:27] end prefill_attn cost 5.570 ms
DEBUG 05-06 15:31:59.711787.711787 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 15:31:59.712558.712558 cuda_h.py:27] end prefill_gate cost 0.430 ms
experts_cpu_alloc {'expert_ids': [87, 99, 115, 27, 15, 19, 95, 119, 59, 43, 75, 16, 40, 112, 52, 64, 116, 80, 96, 32, 84, 25, 21, 37, 9, 69, 113, 125, 61, 45, 85, 121, 93, 53, 65, 98, 66, 94, 14, 54, 122, 62, 114, 82, 42, 102, 58, 38], 'token_total': 1154, 'token_per_expert': {87: 1, 99: 1, 115: 1, 27: 3, 15: 5, 19: 6, 95: 7, 119: 7, 59: 21, 43: 27, 75: 43, 16: 1, 40: 1, 112: 1, 52: 5, 64: 14, 116: 17, 80: 19, 96: 19, 32: 28, 84: 29, 25: 1, 21: 2, 37: 5, 9: 8, 69: 8, 113: 11, 125: 17, 61: 18, 45: 19, 85: 46, 121: 46, 93: 55, 53: 62, 65: 124, 98: 1, 66: 2, 94: 3, 14: 6, 54: 6, 122: 6, 62: 9, 114: 23, 82: 30, 42: 36, 102: 66, 58: 90, 38: 198}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 91], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 15, 'token_total': 814, 'token_per_expert': {3: 302, 7: 256, 31: 109, 91: 147}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 104], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 13, 'token_total': 586, 'token_per_expert': {0: 256, 4: 256, 104: 74}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 17, 'token_total': 755, 'token_per_expert': {1: 286, 5: 256, 13: 213}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 78], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 16, 'token_total': 787, 'token_per_expert': {2: 325, 6: 256, 78: 206}}
INFO 05-06 15:31:59.713881.713881 lmp.py:1839] [layer_moe_fused] layer=13 prefix: 0.427ms alloc: 0.254ms
INFO 05-06 15:31:59.713294.713294 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.1219253540039062e-05 seconds
INFO 05-06 15:31:59.714140.714140 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006937980651855469s
INFO 05-06 15:31:59.714684.714684 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005469322204589844 seconds
DEBUG 05-06 15:31:59.715301.715301 cuda_h.py:27] end moe_cpu_prep_submit cost 0.779 ms
INFO 05-06 15:31:59.732700.732700 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.016925573348999023s
DEBUG 05-06 15:31:59.732075.732075 cuda_h.py:27] end moe_wait_copy_tasks cost 17.081 ms
DEBUG 05-06 15:31:59.736359.736359 cuda_h.py:27] end moe_vllm_forward cost 3.725 ms
DEBUG 05-06 15:31:59.736550.736550 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 15:31:59.737447.737447 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.737595.737595 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.417ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.737392.737392 cuda_h.py:27] end *layer_moe_fused cost 25.043 ms
DEBUG 05-06 15:31:59.738904.738904 cuda_h.py:27] end prefill_merge_scale cost 0.428 ms
DEBUG 05-06 15:31:59.738855.738855 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:59.738350.738350 cuda_h.py:27] end prefill_layer cost 33.524 ms
DEBUG 05-06 15:31:59.738784.738784 lmp.py:1394] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 15:31:59.738640.738640 lmp.py:1350] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 15:31:59.738532.738532 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 15:31:59.744488.744488 cuda_h.py:27] end prefill_attn cost 5.947 ms
DEBUG 05-06 15:31:59.745808.745808 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 15:31:59.746147.746147 cuda_h.py:27] end prefill_gate cost 0.419 ms
experts_cpu_alloc {'expert_ids': [39, 115, 123, 35, 127, 91, 47, 71, 107, 119, 31, 83, 95, 12, 108, 20, 80, 68, 44, 24, 48, 16, 60, 112, 52, 29, 73, 77, 81, 45, 33, 65, 117, 125, 89, 121, 34, 42, 102, 54, 26, 98, 74, 86, 50, 18, 38, 110], 'token_total': 1171, 'token_per_expert': {39: 1, 115: 1, 123: 1, 35: 2, 127: 2, 91: 3, 47: 5, 71: 10, 107: 10, 119: 10, 31: 26, 83: 41, 95: 79, 12: 1, 108: 1, 20: 3, 80: 6, 68: 7, 44: 8, 24: 27, 48: 33, 16: 37, 60: 79, 112: 107, 52: 111, 29: 1, 73: 1, 77: 1, 81: 1, 45: 2, 33: 3, 65: 6, 117: 22, 125: 31, 89: 47, 121: 109, 34: 1, 42: 1, 102: 1, 54: 2, 26: 3, 98: 7, 74: 11, 86: 11, 50: 46, 18: 47, 38: 97, 110: 109}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 59, 111], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 17, 'token_total': 748, 'token_per_expert': {3: 256, 7: 263, 59: 138, 111: 91}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 15, 'token_total': 767, 'token_per_expert': {0: 256, 4: 256, 8: 255}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 97], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 14, 'token_total': 707, 'token_per_expert': {1: 256, 5: 256, 97: 195}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 15, 'token_total': 703, 'token_per_expert': {2: 256, 6: 256, 10: 191}}
INFO 05-06 15:31:59.747475.747475 lmp.py:1839] [layer_moe_fused] layer=14 prefix: 0.399ms alloc: 0.250ms
INFO 05-06 15:31:59.747743.747743 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.09808349609375e-05 seconds
INFO 05-06 15:31:59.748188.748188 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006687641143798828s
INFO 05-06 15:31:59.748651.748651 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005276203155517578 seconds
DEBUG 05-06 15:31:59.748241.748241 cuda_h.py:27] end moe_cpu_prep_submit cost 0.743 ms
INFO 05-06 15:31:59.770295.770295 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.02166604995727539s
DEBUG 05-06 15:31:59.771935.771935 cuda_h.py:27] end moe_wait_copy_tasks cost 21.803 ms
DEBUG 05-06 15:31:59.775600.775600 cuda_h.py:27] end moe_vllm_forward cost 3.625 ms
DEBUG 05-06 15:31:59.775598.775598 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 15:31:59.775681.775681 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:31:59.775591.775591 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.298ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.775228.775228 cuda_h.py:27] end *layer_moe_fused cost 29.263 ms
DEBUG 05-06 15:31:59.776449.776449 cuda_h.py:27] end prefill_merge_scale cost 0.424 ms
DEBUG 05-06 15:31:59.776969.776969 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:31:59.776517.776517 cuda_h.py:27] end prefill_layer cost 38.075 ms
DEBUG 05-06 15:31:59.776236.776236 lmp.py:1394] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 15:31:59.776330.776330 lmp.py:1350] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 15:31:59.776168.776168 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 15:31:59.783055.783055 cuda_h.py:27] end prefill_attn cost 6.843 ms
DEBUG 05-06 15:31:59.784712.784712 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 15:31:59.785958.785958 cuda_h.py:27] end prefill_gate cost 0.412 ms
experts_cpu_alloc {'expert_ids': [19, 75, 83, 47, 59, 127, 23, 39, 95, 99, 103, 107, 67, 88, 72, 48, 52, 40, 16, 28, 68, 116, 84, 100, 112, 36, 12, 113, 121, 69, 41, 101, 85, 33, 97, 117, 65, 9, 109, 105, 42, 82, 110, 46, 22, 78, 70], 'token_total': 1490, 'token_per_expert': {19: 1, 75: 1, 83: 1, 47: 2, 59: 2, 127: 2, 23: 4, 39: 5, 95: 6, 99: 9, 103: 14, 107: 45, 67: 57, 88: 1, 72: 2, 48: 3, 52: 4, 40: 5, 16: 19, 28: 41, 68: 44, 116: 67, 84: 68, 100: 103, 112: 199, 36: 212, 12: 223, 113: 1, 121: 1, 69: 3, 41: 7, 101: 7, 85: 13, 33: 15, 97: 23, 117: 26, 65: 40, 9: 49, 109: 58, 105: 70, 42: 1, 82: 1, 110: 1, 46: 5, 22: 8, 78: 8, 70: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 91], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 16, 'token_total': 575, 'token_per_expert': {3: 256, 7: 256, 91: 63}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 76], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 17, 'token_total': 800, 'token_per_expert': {0: 256, 4: 292, 76: 252}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 125], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 16, 'token_total': 666, 'token_per_expert': {1: 256, 5: 256, 125: 154}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 10, 'token_total': 565, 'token_per_expert': {2: 269, 6: 273, 34: 23}}
INFO 05-06 15:31:59.786088.786088 lmp.py:1839] [layer_moe_fused] layer=15 prefix: 0.401ms alloc: 0.240ms
INFO 05-06 15:31:59.786455.786455 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.09808349609375e-05 seconds
INFO 05-06 15:31:59.787329.787329 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.00063323974609375s
INFO 05-06 15:31:59.787091.787091 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005373954772949219 seconds
DEBUG 05-06 15:31:59.787623.787623 cuda_h.py:27] end moe_cpu_prep_submit cost 0.759 ms
INFO 05-06 15:31:59.799439.799439 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.011485099792480469s
DEBUG 05-06 15:31:59.799662.799662 cuda_h.py:27] end moe_wait_copy_tasks cost 11.629 ms
DEBUG 05-06 15:31:59.804535.804535 cuda_h.py:27] end moe_vllm_forward cost 4.063 ms
DEBUG 05-06 15:31:59.804653.804653 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 15:31:59.804937.804937 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.804562.804562 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.868ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.805267.805267 cuda_h.py:27] end *layer_moe_fused cost 19.633 ms
DEBUG 05-06 15:31:59.805250.805250 cuda_h.py:27] end prefill_merge_scale cost 0.424 ms
DEBUG 05-06 15:31:59.805684.805684 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:59.805133.805133 cuda_h.py:27] end prefill_layer cost 29.312 ms
DEBUG 05-06 15:31:59.806269.806269 lmp.py:1394] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 15:31:59.806362.806362 lmp.py:1350] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 15:31:59.806273.806273 cuda_h.py:27] end prefill_ln cost 0.194 ms
DEBUG 05-06 15:31:59.812089.812089 cuda_h.py:27] end prefill_attn cost 5.913 ms
DEBUG 05-06 15:31:59.812269.812269 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 15:31:59.813217.813217 cuda_h.py:27] end prefill_gate cost 0.418 ms
experts_cpu_alloc {'expert_ids': [67, 51, 79, 115, 23, 123, 71, 99, 119, 63, 47, 55, 35, 39, 32, 48, 80, 92, 104, 36, 108, 24, 72, 84, 100, 44, 12, 120, 20, 73, 105, 117, 121, 49, 57, 13, 21, 29, 109, 25, 9, 10, 30, 58, 78, 86, 106, 42, 70, 82, 74, 90, 114, 126, 34, 110, 102, 38, 26, 46], 'token_total': 1007, 'token_per_expert': {67: 1, 51: 3, 79: 3, 115: 5, 23: 6, 123: 9, 71: 10, 99: 11, 119: 14, 63: 20, 47: 34, 55: 48, 35: 65, 39: 93, 32: 1, 48: 1, 80: 1, 92: 2, 104: 2, 36: 6, 108: 6, 24: 7, 72: 7, 84: 14, 100: 16, 44: 27, 12: 32, 120: 35, 20: 58, 73: 1, 105: 1, 117: 1, 121: 1, 49: 2, 57: 2, 13: 4, 21: 30, 29: 49, 109: 53, 25: 78, 9: 132, 10: 1, 30: 1, 58: 1, 78: 1, 86: 1, 106: 1, 42: 2, 70: 3, 82: 3, 74: 4, 90: 4, 114: 4, 126: 5, 34: 6, 110: 8, 102: 9, 38: 15, 26: 20, 46: 27}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 103], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 758, 'token_per_expert': {3: 256, 7: 256, 31: 100, 103: 146}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 56, 68], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 880, 'token_per_expert': {0: 257, 4: 319, 56: 223, 68: 81}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 81], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 16, 'token_total': 882, 'token_per_expert': {1: 257, 5: 256, 17: 138, 81: 231}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 22, 'token_total': 569, 'token_per_expert': {2: 256, 6: 261, 18: 52}}
INFO 05-06 15:31:59.814169.814169 lmp.py:1839] [layer_moe_fused] layer=16 prefix: 0.400ms alloc: 0.290ms
INFO 05-06 15:31:59.814875.814875 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.3365020751953125e-05 seconds
INFO 05-06 15:31:59.815376.815376 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007894039154052734s
INFO 05-06 15:31:59.816760.816760 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005393028259277344 seconds
DEBUG 05-06 15:31:59.816532.816532 cuda_h.py:27] end moe_cpu_prep_submit cost 0.746 ms
INFO 05-06 15:31:59.835950.835950 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.018346786499023438s
DEBUG 05-06 15:31:59.835935.835935 cuda_h.py:27] end moe_wait_copy_tasks cost 18.494 ms
DEBUG 05-06 15:31:59.839725.839725 cuda_h.py:27] end moe_vllm_forward cost 3.607 ms
DEBUG 05-06 15:31:59.839989.839989 cuda_h.py:27] end moe_cpu_merge cost 0.074 ms
DEBUG 05-06 15:31:59.839971.839971 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.840980.840980 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.490ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.840493.840493 cuda_h.py:27] end *layer_moe_fused cost 26.389 ms
DEBUG 05-06 15:31:59.840715.840715 cuda_h.py:27] end prefill_merge_scale cost 0.425 ms
DEBUG 05-06 15:31:59.841619.841619 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:31:59.841258.841258 cuda_h.py:27] end prefill_layer cost 35.244 ms
DEBUG 05-06 15:31:59.841820.841820 lmp.py:1394] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 15:31:59.841589.841589 lmp.py:1350] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 15:31:59.841861.841861 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 15:31:59.846508.846508 cuda_h.py:27] end prefill_attn cost 4.627 ms
DEBUG 05-06 15:31:59.847650.847650 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 15:31:59.848433.848433 cuda_h.py:27] end prefill_gate cost 0.410 ms
experts_cpu_alloc {'expert_ids': [11, 95, 115, 119, 35, 91, 99, 47, 19, 71, 83, 43, 23, 111, 103, 80, 100, 124, 76, 56, 88, 104, 32, 60, 108, 112, 68, 28, 116, 40, 41, 109, 65, 73, 21, 85, 37, 113, 57, 97, 77, 45, 29, 9, 81, 58, 62, 102, 118, 122, 78, 42, 90, 98, 38, 86, 10, 54, 26], 'token_total': 1118, 'token_per_expert': {11: 1, 95: 1, 115: 1, 119: 1, 35: 3, 91: 6, 99: 8, 47: 9, 19: 10, 71: 16, 83: 16, 43: 23, 23: 25, 111: 26, 103: 29, 80: 1, 100: 1, 124: 1, 76: 2, 56: 3, 88: 3, 104: 4, 32: 10, 60: 18, 108: 19, 112: 22, 68: 31, 28: 33, 116: 40, 40: 48, 41: 1, 109: 1, 65: 2, 73: 3, 21: 5, 85: 6, 37: 7, 113: 12, 57: 22, 97: 27, 77: 29, 45: 58, 29: 78, 9: 88, 81: 132, 58: 1, 62: 1, 102: 1, 118: 3, 122: 5, 78: 6, 42: 9, 90: 9, 98: 11, 38: 13, 86: 19, 10: 26, 54: 62, 26: 70}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 55], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 631, 'token_per_expert': {3: 307, 7: 257, 39: 32, 55: 35}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 64], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 728, 'token_per_expert': {0: 267, 4: 341, 20: 62, 64: 58}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 53, 125], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 924, 'token_per_expert': {1: 256, 5: 257, 53: 169, 125: 242}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 110], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 17, 'token_total': 695, 'token_per_expert': {2: 256, 6: 256, 110: 183}}
INFO 05-06 15:31:59.849332.849332 lmp.py:1839] [layer_moe_fused] layer=17 prefix: 0.404ms alloc: 0.281ms
INFO 05-06 15:31:59.849176.849176 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.2649765014648438e-05 seconds
INFO 05-06 15:31:59.850081.850081 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006515979766845703s
INFO 05-06 15:31:59.850412.850412 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005359649658203125 seconds
DEBUG 05-06 15:31:59.851166.851166 cuda_h.py:27] end moe_cpu_prep_submit cost 0.828 ms
INFO 05-06 15:31:59.865818.865818 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013973712921142578s
DEBUG 05-06 15:31:59.865849.865849 cuda_h.py:27] end moe_wait_copy_tasks cost 14.119 ms
DEBUG 05-06 15:31:59.869964.869964 cuda_h.py:27] end moe_vllm_forward cost 3.653 ms
DEBUG 05-06 15:31:59.869962.869962 cuda_h.py:27] end moe_cpu_merge cost 0.055 ms
DEBUG 05-06 15:31:59.869121.869121 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.870985.870985 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.416ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.870955.870955 cuda_h.py:27] end *layer_moe_fused cost 22.076 ms
DEBUG 05-06 15:31:59.870116.870116 cuda_h.py:27] end prefill_merge_scale cost 0.419 ms
DEBUG 05-06 15:31:59.871027.871027 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:59.871087.871087 cuda_h.py:27] end prefill_layer cost 29.684 ms
DEBUG 05-06 15:31:59.871072.871072 lmp.py:1394] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 15:31:59.871166.871166 lmp.py:1350] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 15:31:59.871132.871132 cuda_h.py:27] end prefill_ln cost 0.210 ms
DEBUG 05-06 15:31:59.876768.876768 cuda_h.py:27] end prefill_attn cost 4.691 ms
DEBUG 05-06 15:31:59.877101.877101 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 15:31:59.878334.878334 cuda_h.py:27] end prefill_gate cost 0.420 ms
experts_cpu_alloc {'expert_ids': [15, 11, 55, 67, 79, 27, 127, 47, 87, 91, 35, 103, 111, 63, 83, 39, 31, 75, 19, 112, 116, 20, 24, 96, 80, 56, 120, 8, 32, 72, 12, 124, 68, 52, 40, 100, 84, 57, 77, 89, 37, 41, 49, 53, 93, 25, 73, 21, 69, 97, 81, 105, 109, 33, 98, 66, 78, 58, 90, 26, 18, 30, 110, 14, 114, 38, 50, 94], 'token_total': 1174, 'token_per_expert': {15: 1, 11: 2, 55: 3, 67: 3, 79: 4, 27: 5, 127: 6, 47: 9, 87: 13, 91: 15, 35: 16, 103: 16, 111: 17, 63: 20, 83: 21, 39: 25, 31: 38, 75: 41, 19: 42, 112: 1, 116: 3, 20: 5, 24: 5, 96: 5, 80: 10, 56: 19, 120: 19, 8: 21, 32: 22, 72: 30, 12: 31, 124: 39, 68: 48, 52: 51, 40: 63, 100: 65, 84: 76, 57: 1, 77: 1, 89: 1, 37: 2, 41: 2, 49: 2, 53: 4, 93: 6, 25: 8, 73: 13, 21: 14, 69: 15, 97: 15, 81: 24, 105: 25, 109: 32, 33: 40, 98: 1, 66: 2, 78: 2, 58: 5, 90: 6, 26: 7, 18: 8, 30: 13, 110: 13, 14: 14, 114: 14, 38: 19, 50: 19, 94: 36}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 59, 119, 123], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 24, 'token_total': 894, 'token_per_expert': {3: 326, 7: 256, 59: 77, 119: 44, 123: 191}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 88], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 22, 'token_total': 706, 'token_per_expert': {0: 256, 4: 289, 16: 80, 88: 81}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 85], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 21, 'token_total': 708, 'token_per_expert': {1: 265, 5: 256, 13: 97, 85: 90}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 42, 62], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 614, 'token_per_expert': {2: 267, 6: 256, 42: 51, 62: 40}}
INFO 05-06 15:31:59.879380.879380 lmp.py:1839] [layer_moe_fused] layer=18 prefix: 0.409ms alloc: 0.310ms
INFO 05-06 15:31:59.879184.879184 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.574920654296875e-05 seconds
INFO 05-06 15:31:59.879344.879344 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006747245788574219s
INFO 05-06 15:31:59.880458.880458 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005486011505126953 seconds
DEBUG 05-06 15:31:59.880929.880929 cuda_h.py:27] end moe_cpu_prep_submit cost 0.775 ms
INFO 05-06 15:31:59.906293.906293 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.025521516799926758s
DEBUG 05-06 15:31:59.906947.906947 cuda_h.py:27] end moe_wait_copy_tasks cost 25.670 ms
DEBUG 05-06 15:31:59.910692.910692 cuda_h.py:27] end moe_vllm_forward cost 3.658 ms
DEBUG 05-06 15:31:59.911498.911498 cuda_h.py:27] end moe_cpu_merge cost 0.055 ms
DEBUG 05-06 15:31:59.911993.911993 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:31:59.911572.911572 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.391ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.911761.911761 cuda_h.py:27] end *layer_moe_fused cost 33.483 ms
DEBUG 05-06 15:31:59.912751.912751 cuda_h.py:27] end prefill_merge_scale cost 0.428 ms
DEBUG 05-06 15:31:59.912602.912602 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:59.912581.912581 cuda_h.py:27] end prefill_layer cost 41.070 ms
DEBUG 05-06 15:31:59.912116.912116 lmp.py:1394] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 15:31:59.912448.912448 lmp.py:1350] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 15:31:59.913066.913066 cuda_h.py:27] end prefill_ln cost 0.200 ms
DEBUG 05-06 15:31:59.919374.919374 cuda_h.py:27] end prefill_attn cost 6.380 ms
DEBUG 05-06 15:31:59.920416.920416 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 15:31:59.921463.921463 cuda_h.py:27] end prefill_gate cost 0.416 ms
experts_cpu_alloc {'expert_ids': [59, 103, 99, 111, 47, 39, 87, 83, 23, 35, 63, 19, 51, 119, 96, 112, 124, 104, 56, 20, 92, 32, 64, 100, 120, 88, 84, 12, 80, 28, 24, 13, 53, 101, 61, 89, 121, 25, 9, 109, 73, 125, 41, 69, 14, 70, 110, 78, 126, 118, 58, 46, 106, 50, 102, 122, 90, 94], 'token_total': 1112, 'token_per_expert': {59: 1, 103: 1, 99: 2, 111: 2, 47: 5, 39: 6, 87: 7, 83: 8, 23: 11, 35: 12, 63: 13, 19: 15, 51: 28, 119: 105, 96: 1, 112: 1, 124: 3, 104: 4, 56: 7, 20: 8, 92: 10, 32: 15, 64: 16, 100: 20, 120: 21, 88: 25, 84: 27, 12: 49, 80: 71, 28: 77, 24: 99, 13: 1, 53: 1, 101: 1, 61: 2, 89: 3, 121: 3, 25: 10, 9: 17, 109: 17, 73: 23, 125: 29, 41: 31, 69: 42, 14: 1, 70: 1, 110: 1, 78: 2, 126: 2, 118: 3, 58: 7, 46: 8, 106: 11, 50: 13, 102: 21, 122: 26, 90: 57, 94: 109}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 75], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 766, 'token_per_expert': {3: 256, 7: 256, 27: 144, 75: 110}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 48, 52], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 21, 'token_total': 758, 'token_per_expert': {0: 256, 4: 257, 48: 118, 52: 127}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 81], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 17, 'token_total': 738, 'token_per_expert': {1: 275, 5: 266, 37: 79, 81: 118}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 17, 'token_total': 722, 'token_per_expert': {2: 256, 6: 258, 38: 208}}
INFO 05-06 15:31:59.922707.922707 lmp.py:1839] [layer_moe_fused] layer=19 prefix: 0.409ms alloc: 0.278ms
INFO 05-06 15:31:59.922836.922836 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 15:31:59.922235.922235 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006561279296875s
INFO 05-06 15:31:59.923466.923466 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005328655242919922 seconds
DEBUG 05-06 15:31:59.923841.923841 cuda_h.py:27] end moe_cpu_prep_submit cost 0.627 ms
INFO 05-06 15:31:59.927605.927605 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0036284923553466797s
DEBUG 05-06 15:31:59.927271.927271 cuda_h.py:27] end moe_wait_copy_tasks cost 3.746 ms
DEBUG 05-06 15:31:59.932466.932466 cuda_h.py:27] end moe_vllm_forward cost 4.049 ms
DEBUG 05-06 15:31:59.936018.936018 cuda_h.py:27] end moe_cpu_merge cost 3.922 ms
DEBUG 05-06 15:31:59.936041.936041 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 15:31:59.936296.936296 lmp.py:1953] [layer_moe_fused] vllm triton time: 8.613ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.937146.937146 cuda_h.py:27] end *layer_moe_fused cost 15.761 ms
DEBUG 05-06 15:31:59.937460.937460 cuda_h.py:27] end prefill_merge_scale cost 0.428 ms
DEBUG 05-06 15:31:59.937132.937132 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:59.937631.937631 cuda_h.py:27] end prefill_layer cost 25.239 ms
DEBUG 05-06 15:31:59.938399.938399 lmp.py:1394] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 15:31:59.938492.938492 lmp.py:1350] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 15:31:59.938961.938961 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 15:31:59.940124.940124 cuda_h.py:27] end prefill_attn cost 1.850 ms
DEBUG 05-06 15:31:59.940742.940742 cuda_h.py:27] end prefill_ffn_prep cost 0.378 ms
DEBUG 05-06 15:31:59.941412.941412 cuda_h.py:27] end prefill_gate cost 0.419 ms
experts_cpu_alloc {'expert_ids': [39, 103, 63, 47, 55, 111, 59, 27, 19, 51, 79, 91, 127, 11, 83, 56, 12, 16, 116, 28, 104, 20, 80, 96, 52, 84, 32, 92, 100, 13, 77, 105, 89, 17, 81, 121, 9, 21, 65, 85, 33, 45, 101, 57, 73, 49, 22, 58, 122, 30, 82, 14, 62, 38, 102, 90, 114, 50, 18, 70, 66], 'token_total': 1075, 'token_per_expert': {39: 1, 103: 1, 63: 2, 47: 3, 55: 3, 111: 4, 59: 7, 27: 10, 19: 12, 51: 14, 79: 17, 91: 26, 127: 38, 11: 44, 83: 100, 56: 1, 12: 2, 16: 2, 116: 3, 28: 4, 104: 4, 20: 5, 80: 7, 96: 7, 52: 10, 84: 10, 32: 12, 92: 13, 100: 17, 13: 1, 77: 1, 105: 3, 89: 4, 17: 5, 81: 6, 121: 6, 9: 7, 21: 7, 65: 30, 85: 34, 33: 37, 45: 50, 101: 62, 57: 77, 73: 99, 49: 108, 22: 1, 58: 1, 122: 2, 30: 3, 82: 3, 14: 4, 62: 6, 38: 10, 102: 11, 90: 13, 114: 13, 50: 18, 18: 19, 70: 26, 66: 29}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 115, 123], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 850, 'token_per_expert': {3: 276, 7: 256, 115: 156, 123: 162}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 120], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 584, 'token_per_expert': {0: 259, 4: 257, 8: 23, 120: 45}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 97, 125], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 21, 'token_total': 970, 'token_per_expert': {1: 256, 5: 401, 97: 201, 125: 112}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 106, 110], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 19, 'token_total': 617, 'token_per_expert': {2: 274, 6: 256, 106: 58, 110: 29}}
INFO 05-06 15:31:59.942153.942153 lmp.py:1839] [layer_moe_fused] layer=20 prefix: 0.415ms alloc: 0.290ms
INFO 05-06 15:31:59.942057.942057 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.4557113647460938e-05 seconds
INFO 05-06 15:31:59.943474.943474 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008063316345214844s
INFO 05-06 15:31:59.944414.944414 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005278587341308594 seconds
DEBUG 05-06 15:31:59.944145.944145 cuda_h.py:27] end moe_cpu_prep_submit cost 0.774 ms
INFO 05-06 15:31:59.972553.972553 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.027745962142944336s
DEBUG 05-06 15:31:59.972100.972100 cuda_h.py:27] end moe_wait_copy_tasks cost 27.885 ms
DEBUG 05-06 15:31:59.976109.976109 cuda_h.py:27] end moe_vllm_forward cost 3.641 ms
DEBUG 05-06 15:31:59.977915.977915 cuda_h.py:27] end moe_cpu_merge cost 0.055 ms
DEBUG 05-06 15:31:59.977840.977840 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:31:59.977326.977326 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.343ms (seq_len=128 cg=False)
DEBUG 05-06 15:31:59.977965.977965 cuda_h.py:27] end *layer_moe_fused cost 35.648 ms
DEBUG 05-06 15:31:59.978710.978710 cuda_h.py:27] end prefill_merge_scale cost 0.424 ms
DEBUG 05-06 15:31:59.978991.978991 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:31:59.978380.978380 cuda_h.py:27] end prefill_layer cost 40.366 ms
DEBUG 05-06 15:31:59.978346.978346 lmp.py:1394] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 15:31:59.978201.978201 lmp.py:1350] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 15:31:59.979374.979374 cuda_h.py:27] end prefill_ln cost 0.198 ms
DEBUG 05-06 15:31:59.984536.984536 cuda_h.py:27] end prefill_attn cost 5.571 ms
DEBUG 05-06 15:31:59.985478.985478 cuda_h.py:27] end prefill_ffn_prep cost 0.373 ms
DEBUG 05-06 15:31:59.986134.986134 cuda_h.py:27] end prefill_gate cost 0.412 ms
experts_cpu_alloc {'expert_ids': [91, 19, 27, 47, 79, 59, 103, 75, 71, 35, 119, 55, 39, 99, 83, 67, 11, 52, 56, 108, 28, 36, 96, 72, 92, 68, 32, 120, 24, 12, 100, 88, 104, 80, 21, 93, 85, 81, 97, 17, 13, 41, 49, 61, 9, 53, 73, 121, 109, 29, 14, 78, 106, 38, 10, 66, 110, 118, 26, 82, 62, 22, 58, 94, 34, 30, 98, 122], 'token_total': 946, 'token_per_expert': {91: 1, 19: 2, 27: 2, 47: 6, 79: 6, 59: 7, 103: 7, 75: 8, 71: 9, 35: 10, 119: 11, 55: 12, 39: 16, 99: 18, 83: 25, 67: 37, 11: 65, 52: 1, 56: 1, 108: 2, 28: 4, 36: 4, 96: 4, 72: 5, 92: 5, 68: 8, 32: 11, 120: 11, 24: 22, 12: 24, 100: 24, 88: 38, 104: 39, 80: 109, 21: 1, 93: 1, 85: 3, 81: 7, 97: 8, 17: 10, 13: 12, 41: 13, 49: 15, 61: 15, 9: 17, 53: 18, 73: 24, 121: 24, 109: 38, 29: 61, 14: 2, 78: 2, 106: 2, 38: 3, 10: 4, 66: 4, 110: 4, 118: 4, 26: 5, 82: 5, 62: 7, 22: 8, 58: 9, 94: 9, 34: 10, 30: 12, 98: 17, 122: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 95, 115], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 22, 'token_total': 808, 'token_per_expert': {3: 256, 7: 256, 15: 121, 95: 69, 115: 106}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 76], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 21, 'token_total': 877, 'token_per_expert': {0: 285, 4: 259, 8: 222, 76: 111}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 65, 105], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 720, 'token_per_expert': {1: 270, 5: 289, 65: 98, 105: 63}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 46], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 22, 'token_total': 745, 'token_per_expert': {2: 277, 6: 256, 18: 61, 46: 151}}
INFO 05-06 15:31:59.987220.987220 lmp.py:1839] [layer_moe_fused] layer=21 prefix: 0.404ms alloc: 0.314ms
INFO 05-06 15:31:59.987224.987224 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.7894973754882812e-05 seconds
INFO 05-06 15:31:59.988968.988968 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006842613220214844s
INFO 05-06 15:31:59.988558.988558 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005500316619873047 seconds
DEBUG 05-06 15:31:59.989224.989224 cuda_h.py:27] end moe_cpu_prep_submit cost 0.849 ms
INFO 05-06 15:32:00.004301.004301 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014673709869384766s
DEBUG 05-06 15:32:00.004994.004994 cuda_h.py:27] end moe_wait_copy_tasks cost 14.816 ms
DEBUG 05-06 15:32:00.008030.008030 cuda_h.py:27] end moe_vllm_forward cost 3.656 ms
DEBUG 05-06 15:32:00.008697.008697 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 15:32:00.008909.008909 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:32:00.008731.008731 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.578ms (seq_len=128 cg=False)
DEBUG 05-06 15:32:00.009816.009816 cuda_h.py:27] end *layer_moe_fused cost 22.859 ms
DEBUG 05-06 15:32:00.009474.009474 cuda_h.py:27] end prefill_merge_scale cost 0.430 ms
DEBUG 05-06 15:32:00.009994.009994 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:32:00.010020.010020 cuda_h.py:27] end prefill_layer cost 31.360 ms
DEBUG 05-06 15:32:00.010071.010071 lmp.py:1394] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 15:32:00.010641.010641 lmp.py:1350] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 15:32:00.010534.010534 cuda_h.py:27] end prefill_ln cost 0.203 ms
DEBUG 05-06 15:32:00.016588.016588 cuda_h.py:27] end prefill_attn cost 5.701 ms
DEBUG 05-06 15:32:00.016622.016622 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 15:32:00.017345.017345 cuda_h.py:27] end prefill_gate cost 0.421 ms
experts_cpu_alloc {'expert_ids': [39, 11, 51, 67, 119, 127, 15, 71, 87, 35, 63, 83, 75, 19, 99, 123, 107, 20, 60, 44, 92, 28, 8, 48, 52, 76, 16, 120, 40, 116, 72, 100, 108, 84, 32, 17, 85, 97, 29, 33, 45, 69, 53, 9, 93, 37, 125, 101, 117, 30, 98, 10, 38, 78, 110, 90, 66, 70, 42, 118, 94, 126, 54], 'token_total': 1165, 'token_per_expert': {39: 1, 11: 2, 51: 3, 67: 3, 119: 4, 127: 4, 15: 6, 71: 7, 87: 7, 35: 10, 63: 11, 83: 11, 75: 14, 19: 32, 99: 38, 123: 38, 107: 41, 20: 1, 60: 1, 44: 2, 92: 2, 28: 5, 8: 7, 48: 8, 52: 10, 76: 10, 16: 14, 120: 19, 40: 26, 116: 27, 72: 35, 100: 47, 108: 48, 84: 60, 32: 75, 17: 2, 85: 2, 97: 3, 29: 4, 33: 4, 45: 7, 69: 11, 53: 14, 9: 15, 93: 15, 37: 34, 125: 45, 101: 102, 117: 115, 30: 1, 98: 1, 10: 3, 38: 4, 78: 4, 110: 4, 90: 5, 66: 8, 70: 8, 42: 9, 118: 18, 94: 21, 126: 38, 54: 39}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 103], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 21, 'token_total': 656, 'token_per_expert': {3: 256, 7: 264, 23: 67, 103: 69}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 88, 124], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 22, 'token_total': 834, 'token_per_expert': {0: 267, 4: 256, 88: 111, 124: 200}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 41, 73], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 817, 'token_per_expert': {1: 258, 5: 256, 41: 179, 73: 124}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34, 50], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 624, 'token_per_expert': {2: 263, 6: 257, 34: 63, 50: 41}}
INFO 05-06 15:32:00.018126.018126 lmp.py:1839] [layer_moe_fused] layer=22 prefix: 0.409ms alloc: 0.295ms
INFO 05-06 15:32:00.018838.018838 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.47955322265625e-05 seconds
INFO 05-06 15:32:00.019128.019128 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007305145263671875s
INFO 05-06 15:32:00.020850.020850 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005421638488769531 seconds
DEBUG 05-06 15:32:00.020461.020461 cuda_h.py:27] end moe_cpu_prep_submit cost 0.709 ms
INFO 05-06 15:32:00.034064.034064 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013396978378295898s
DEBUG 05-06 15:32:00.034168.034168 cuda_h.py:27] end moe_wait_copy_tasks cost 13.562 ms
DEBUG 05-06 15:32:00.038301.038301 cuda_h.py:27] end moe_vllm_forward cost 3.737 ms
DEBUG 05-06 15:32:00.038644.038644 cuda_h.py:27] end moe_cpu_merge cost 0.059 ms
DEBUG 05-06 15:32:00.038290.038290 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:32:00.039200.039200 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.477ms (seq_len=128 cg=False)
DEBUG 05-06 15:32:00.039760.039760 cuda_h.py:27] end *layer_moe_fused cost 21.435 ms
DEBUG 05-06 15:32:00.040294.040294 cuda_h.py:27] end prefill_merge_scale cost 0.447 ms
DEBUG 05-06 15:32:00.040013.040013 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.043 ms
DEBUG 05-06 15:32:00.040277.040277 cuda_h.py:27] end prefill_layer cost 30.067 ms
DEBUG 05-06 15:32:00.040448.040448 lmp.py:1394] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 15:32:00.040211.040211 lmp.py:1350] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 15:32:00.040516.040516 cuda_h.py:27] end prefill_ln cost 0.200 ms
DEBUG 05-06 15:32:00.046740.046740 cuda_h.py:27] end prefill_attn cost 5.652 ms
DEBUG 05-06 15:32:00.047444.047444 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 15:32:00.048220.048220 cuda_h.py:27] end prefill_gate cost 0.414 ms
experts_cpu_alloc {'expert_ids': [99, 123, 47, 59, 43, 83, 91, 103, 11, 27, 79, 107, 39, 111, 23, 60, 8, 80, 112, 36, 116, 12, 20, 44, 92, 40, 64, 76, 124, 120, 56, 88, 69, 81, 89, 101, 109, 65, 21, 25, 17, 61, 57, 93, 9, 13, 29, 105, 66, 54, 90, 110, 126, 70, 22, 118, 98, 78, 106, 18, 38, 26, 42, 62], 'token_total': 1092, 'token_per_expert': {99: 2, 123: 2, 47: 3, 59: 3, 43: 5, 83: 6, 91: 9, 103: 10, 11: 11, 27: 14, 79: 15, 107: 15, 39: 19, 111: 22, 23: 23, 60: 1, 8: 2, 80: 2, 112: 2, 36: 3, 116: 3, 12: 11, 20: 14, 44: 22, 92: 24, 40: 26, 64: 33, 76: 36, 124: 47, 120: 48, 56: 51, 88: 61, 69: 1, 81: 1, 89: 1, 101: 2, 109: 2, 65: 3, 21: 4, 25: 5, 17: 14, 61: 16, 57: 21, 93: 30, 9: 44, 13: 50, 29: 59, 105: 71, 66: 1, 54: 2, 90: 2, 110: 3, 126: 3, 70: 5, 22: 7, 118: 7, 98: 10, 78: 12, 106: 12, 18: 21, 38: 22, 26: 26, 42: 43, 62: 47}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 51, 119], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 20, 'token_total': 845, 'token_per_expert': {3: 256, 7: 258, 31: 39, 51: 129, 119: 163}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 104], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 21, 'token_total': 679, 'token_per_expert': {0: 256, 4: 258, 28: 83, 104: 82}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 73], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 757, 'token_per_expert': {1: 284, 5: 256, 33: 74, 73: 143}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 114], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 723, 'token_per_expert': {2: 257, 6: 279, 14: 117, 114: 70}}
INFO 05-06 15:32:00.049054.049054 lmp.py:1839] [layer_moe_fused] layer=23 prefix: 0.404ms alloc: 0.302ms
INFO 05-06 15:32:00.049196.049196 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.6702880859375e-05 seconds
INFO 05-06 15:32:00.050979.050979 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007255077362060547s
INFO 05-06 15:32:00.050516.050516 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005462169647216797 seconds
DEBUG 05-06 15:32:00.051247.051247 cuda_h.py:27] end moe_cpu_prep_submit cost 0.774 ms
INFO 05-06 15:32:00.071568.071568 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01992511749267578s
DEBUG 05-06 15:32:00.071076.071076 cuda_h.py:27] end moe_wait_copy_tasks cost 20.074 ms
DEBUG 05-06 15:32:00.075211.075211 cuda_h.py:27] end moe_vllm_forward cost 3.617 ms
DEBUG 05-06 15:32:00.075999.075999 cuda_h.py:27] end moe_cpu_merge cost 0.061 ms
DEBUG 05-06 15:32:00.075517.075517 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:32:00.076904.076904 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.505ms (seq_len=128 cg=False)
DEBUG 05-06 15:32:00.076290.076290 cuda_h.py:27] end *layer_moe_fused cost 28.341 ms
DEBUG 05-06 15:32:00.077088.077088 cuda_h.py:27] end prefill_merge_scale cost 0.427 ms
DEBUG 05-06 15:32:00.077608.077608 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:32:00.077818.077818 cuda_h.py:27] end prefill_layer cost 36.928 ms
DEBUG 05-06 15:32:00.077678.077678 lmp.py:1394] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 15:32:00.077295.077295 lmp.py:1350] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 15:32:00.077154.077154 cuda_h.py:27] end prefill_ln cost 0.201 ms
DEBUG 05-06 15:32:00.082881.082881 cuda_h.py:27] end prefill_attn cost 4.655 ms
DEBUG 05-06 15:32:00.083201.083201 cuda_h.py:27] end prefill_ffn_prep cost 0.369 ms
DEBUG 05-06 15:32:00.084653.084653 cuda_h.py:27] end prefill_gate cost 0.425 ms
experts_cpu_alloc {'expert_ids': [99, 123, 119, 35, 103, 107, 31, 55, 47, 67, 111, 63, 75, 15, 23, 83, 36, 60, 88, 44, 92, 40, 16, 116, 72, 24, 8, 12, 112, 124, 80, 93, 85, 117, 37, 13, 9, 101, 45, 53, 125, 41, 73, 29, 105, 121, 17, 49, 110, 34, 82, 14, 122, 42, 102, 10, 62, 78, 46, 90, 98, 66, 118, 50, 74, 70, 30, 18], 'token_total': 1185, 'token_per_expert': {99: 1, 123: 1, 119: 2, 35: 3, 103: 3, 107: 3, 31: 4, 55: 5, 47: 8, 67: 8, 111: 8, 63: 11, 75: 20, 15: 28, 23: 29, 83: 52, 36: 1, 60: 1, 88: 1, 44: 2, 92: 2, 40: 3, 16: 4, 116: 6, 72: 12, 24: 13, 8: 16, 12: 20, 112: 21, 124: 23, 80: 24, 93: 2, 85: 3, 117: 3, 37: 4, 13: 5, 9: 6, 101: 7, 45: 9, 53: 10, 125: 10, 41: 12, 73: 13, 29: 17, 105: 17, 121: 28, 17: 36, 49: 103, 110: 2, 34: 3, 82: 4, 14: 6, 122: 6, 42: 9, 102: 9, 10: 13, 62: 16, 78: 16, 46: 28, 90: 28, 98: 33, 66: 35, 118: 35, 50: 38, 74: 61, 70: 64, 30: 71, 18: 88}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 43, 51, 127], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 21, 'token_total': 736, 'token_per_expert': {3: 278, 7: 257, 43: 75, 51: 54, 127: 72}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 76, 100], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 20, 'token_total': 664, 'token_per_expert': {0: 265, 4: 275, 28: 39, 76: 38, 100: 47}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 97, 109], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 21, 'token_total': 786, 'token_per_expert': {1: 256, 5: 256, 97: 120, 109: 154}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 114, 126], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 24, 'token_total': 725, 'token_per_expert': {2: 256, 6: 256, 114: 107, 126: 106}}
INFO 05-06 15:32:00.085659.085659 lmp.py:1839] [layer_moe_fused] layer=24 prefix: 0.411ms alloc: 0.319ms
INFO 05-06 15:32:00.085994.085994 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.8133392333984375e-05 seconds
INFO 05-06 15:32:00.086038.086038 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007166862487792969s
INFO 05-06 15:32:00.086621.086621 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005462169647216797 seconds
DEBUG 05-06 15:32:00.087762.087762 cuda_h.py:27] end moe_cpu_prep_submit cost 0.746 ms
INFO 05-06 15:32:00.106735.106735 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.019153833389282227s
DEBUG 05-06 15:32:00.106574.106574 cuda_h.py:27] end moe_wait_copy_tasks cost 19.297 ms
DEBUG 05-06 15:32:00.110219.110219 cuda_h.py:27] end moe_vllm_forward cost 3.653 ms
DEBUG 05-06 15:32:00.110363.110363 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 15:32:00.111904.111904 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:32:00.111529.111529 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.353ms (seq_len=128 cg=False)
DEBUG 05-06 15:32:00.111607.111607 cuda_h.py:27] end *layer_moe_fused cost 27.238 ms
DEBUG 05-06 15:32:00.112258.112258 cuda_h.py:27] end prefill_merge_scale cost 0.429 ms
DEBUG 05-06 15:32:00.112633.112633 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:32:00.112137.112137 cuda_h.py:27] end prefill_layer cost 34.854 ms
DEBUG 05-06 15:32:00.112937.112937 lmp.py:1394] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 15:32:00.112554.112554 lmp.py:1350] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 15:32:00.113422.113422 cuda_h.py:27] end prefill_ln cost 0.207 ms
DEBUG 05-06 15:32:00.117026.117026 cuda_h.py:27] end prefill_attn cost 4.737 ms
DEBUG 05-06 15:32:00.118823.118823 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 15:32:00.119201.119201 cuda_h.py:27] end prefill_gate cost 0.413 ms
experts_cpu_alloc {'expert_ids': [35, 83, 107, 47, 75, 19, 67, 123, 27, 79, 55, 119, 59, 39, 115, 99, 111, 76, 104, 16, 20, 60, 124, 120, 32, 24, 72, 84, 44, 56, 52, 36, 92, 41, 73, 97, 13, 89, 113, 125, 9, 105, 49, 93, 109, 117, 53, 69, 25, 37, 45, 18, 30, 122, 58, 102, 50, 126, 14, 78, 86, 94, 26, 66, 62, 22, 118, 10, 82], 'token_total': 994, 'token_per_expert': {35: 1, 83: 1, 107: 1, 47: 2, 75: 4, 19: 5, 67: 5, 123: 6, 27: 8, 79: 10, 55: 11, 119: 13, 59: 16, 39: 18, 115: 38, 99: 45, 111: 53, 76: 1, 104: 1, 16: 2, 20: 2, 60: 2, 124: 3, 120: 5, 32: 6, 24: 11, 72: 12, 84: 12, 44: 17, 56: 19, 52: 21, 36: 32, 92: 44, 41: 1, 73: 1, 97: 1, 13: 2, 89: 3, 113: 3, 125: 3, 9: 4, 105: 5, 49: 6, 93: 13, 109: 22, 117: 38, 53: 39, 69: 45, 25: 46, 37: 46, 45: 49, 18: 1, 30: 1, 122: 1, 58: 2, 102: 3, 50: 4, 126: 4, 14: 6, 78: 6, 86: 6, 94: 9, 26: 11, 66: 11, 62: 12, 22: 24, 118: 25, 10: 29, 82: 85}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 63, 95], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 22, 'token_total': 831, 'token_per_expert': {3: 256, 7: 333, 11: 63, 63: 76, 95: 103}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 40, 80, 100], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 21, 'token_total': 738, 'token_per_expert': {0: 256, 4: 256, 40: 112, 80: 45, 100: 69}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 121], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 22, 'token_total': 678, 'token_per_expert': {1: 256, 5: 256, 21: 74, 121: 92}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 42, 74], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 22, 'token_total': 855, 'token_per_expert': {2: 270, 6: 259, 42: 228, 74: 98}}
INFO 05-06 15:32:00.120606.120606 lmp.py:1839] [layer_moe_fused] layer=25 prefix: 0.412ms alloc: 0.317ms
INFO 05-06 15:32:00.120086.120086 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.956390380859375e-05 seconds
INFO 05-06 15:32:00.121646.121646 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006928443908691406s
INFO 05-06 15:32:00.121885.121885 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.000537872314453125 seconds
DEBUG 05-06 15:32:00.122769.122769 cuda_h.py:27] end moe_cpu_prep_submit cost 0.824 ms
INFO 05-06 15:32:00.131469.131469 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.00939488410949707s
DEBUG 05-06 15:32:00.132169.132169 cuda_h.py:27] end moe_wait_copy_tasks cost 9.542 ms
DEBUG 05-06 15:32:00.136382.136382 cuda_h.py:27] end moe_vllm_forward cost 3.768 ms
DEBUG 05-06 15:32:00.136857.136857 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 15:32:00.136684.136684 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:32:00.136593.136593 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.520ms (seq_len=128 cg=False)
DEBUG 05-06 15:32:00.137896.137896 cuda_h.py:27] end *layer_moe_fused cost 17.521 ms
DEBUG 05-06 15:32:00.137356.137356 cuda_h.py:27] end prefill_merge_scale cost 0.425 ms
DEBUG 05-06 15:32:00.137730.137730 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.040 ms
DEBUG 05-06 15:32:00.137788.137788 cuda_h.py:27] end prefill_layer cost 25.171 ms
DEBUG 05-06 15:32:00.138508.138508 lmp.py:1394] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 15:32:00.138125.138125 lmp.py:1350] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 15:32:00.138309.138309 cuda_h.py:27] end prefill_ln cost 0.202 ms
DEBUG 05-06 15:32:00.144068.144068 cuda_h.py:27] end prefill_attn cost 6.399 ms
DEBUG 05-06 15:32:00.145726.145726 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 15:32:00.146819.146819 cuda_h.py:27] end prefill_gate cost 0.416 ms
experts_cpu_alloc {'expert_ids': [31, 59, 71, 83, 27, 11, 99, 107, 67, 115, 111, 87, 51, 32, 124, 16, 88, 92, 112, 36, 68, 24, 12, 20, 60, 57, 121, 93, 113, 125, 89, 65, 25, 37, 45, 61, 33, 13, 85, 94, 90, 126, 18, 22, 62, 102, 30, 98, 110, 82, 26, 66, 10, 122, 46, 78, 86], 'token_total': 830, 'token_per_expert': {31: 1, 59: 1, 71: 1, 83: 1, 27: 2, 11: 3, 99: 3, 107: 7, 67: 14, 115: 15, 111: 24, 87: 41, 51: 42, 32: 1, 124: 1, 16: 2, 88: 2, 92: 3, 112: 3, 36: 6, 68: 11, 24: 15, 12: 16, 20: 20, 60: 20, 57: 1, 121: 1, 93: 2, 113: 2, 125: 2, 89: 6, 65: 9, 25: 15, 37: 16, 45: 16, 61: 16, 33: 21, 13: 34, 85: 98, 94: 2, 90: 3, 126: 5, 18: 7, 22: 7, 62: 7, 102: 8, 30: 9, 98: 9, 110: 9, 82: 10, 26: 11, 66: 11, 10: 19, 122: 26, 46: 35, 78: 66, 86: 92}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 123, 127], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 17, 'token_total': 727, 'token_per_expert': {3: 256, 7: 282, 123: 83, 127: 106}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 96, 120], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 16, 'token_total': 727, 'token_per_expert': {0: 256, 4: 256, 96: 67, 120: 148}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 29], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 1044, 'token_per_expert': {1: 272, 5: 351, 21: 217, 29: 204}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14], 'expert_count': 3, 'ideal_gpu_count': 3, 'keep_on_gpu': 3, 'hit_count_on_device': 21, 'token_total': 768, 'token_per_expert': {2: 256, 6: 297, 14: 215}}
INFO 05-06 15:32:00.147897.147897 lmp.py:1839] [layer_moe_fused] layer=26 prefix: 0.398ms alloc: 0.274ms
INFO 05-06 15:32:00.147324.147324 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.4557113647460938e-05 seconds
INFO 05-06 15:32:00.148354.148354 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007364749908447266s
INFO 05-06 15:32:00.148355.148355 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005364418029785156 seconds
DEBUG 05-06 15:32:00.149078.149078 cuda_h.py:27] end moe_cpu_prep_submit cost 0.772 ms
INFO 05-06 15:32:00.172597.172597 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.02348494529724121s
DEBUG 05-06 15:32:00.173191.173191 cuda_h.py:27] end moe_wait_copy_tasks cost 23.624 ms
DEBUG 05-06 15:32:00.177884.177884 cuda_h.py:27] end moe_vllm_forward cost 3.674 ms
DEBUG 05-06 15:32:00.177359.177359 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 15:32:00.177820.177820 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:32:00.177753.177753 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.484ms (seq_len=128 cg=False)
DEBUG 05-06 15:32:00.178301.178301 cuda_h.py:27] end *layer_moe_fused cost 31.521 ms
DEBUG 05-06 15:32:00.178337.178337 cuda_h.py:27] end prefill_merge_scale cost 0.433 ms
DEBUG 05-06 15:32:00.178141.178141 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:32:00.178913.178913 cuda_h.py:27] end prefill_layer cost 40.748 ms
DEBUG 05-06 15:32:00.178369.178369 lmp.py:1394] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 15:32:00.178178.178178 lmp.py:1350] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 15:32:00.179640.179640 cuda_h.py:27] end prefill_ln cost 0.203 ms
DEBUG 05-06 15:32:00.184257.184257 cuda_h.py:27] end prefill_attn cost 5.523 ms
DEBUG 05-06 15:32:00.185385.185385 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 15:32:00.186048.186048 cuda_h.py:27] end prefill_gate cost 0.415 ms
experts_cpu_alloc {'expert_ids': [11, 19, 39, 43, 67, 91, 119, 63, 15, 75, 107, 103, 71, 127, 123, 51, 47, 87, 27, 92, 48, 12, 28, 40, 44, 104, 76, 68, 20, 56, 16, 36, 8, 124, 96, 53, 37, 29, 97, 69, 101, 77, 113, 89, 109, 9, 61, 13, 81, 57, 93, 34, 86, 94, 118, 126, 74, 82, 110, 14, 62, 50, 114, 54, 22, 18, 66], 'token_total': 1067, 'token_per_expert': {11: 1, 19: 1, 39: 1, 43: 1, 67: 2, 91: 3, 119: 3, 63: 4, 15: 6, 75: 7, 107: 7, 103: 8, 71: 11, 127: 14, 123: 15, 51: 19, 47: 39, 87: 39, 27: 41, 92: 1, 48: 2, 12: 4, 28: 4, 40: 5, 44: 5, 104: 5, 76: 8, 68: 10, 20: 11, 56: 14, 16: 47, 36: 73, 8: 75, 124: 87, 96: 98, 53: 1, 37: 2, 29: 3, 97: 3, 69: 6, 101: 6, 77: 7, 113: 7, 89: 8, 109: 9, 9: 10, 61: 14, 13: 19, 81: 26, 57: 35, 93: 49, 34: 1, 86: 1, 94: 1, 118: 1, 126: 3, 74: 5, 82: 6, 110: 9, 14: 12, 62: 12, 50: 13, 114: 18, 54: 19, 22: 21, 18: 30, 66: 39}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 59, 111], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 24, 'token_total': 822, 'token_per_expert': {3: 258, 7: 374, 23: 42, 59: 104, 111: 44}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 64, 80], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 720, 'token_per_expert': {0: 256, 4: 256, 64: 109, 80: 99}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 25, 85], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 725, 'token_per_expert': {1: 299, 5: 257, 25: 81, 85: 88}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 58, 90], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 762, 'token_per_expert': {2: 315, 6: 256, 58: 66, 90: 125}}
INFO 05-06 15:32:00.187379.187379 lmp.py:1839] [layer_moe_fused] layer=27 prefix: 0.411ms alloc: 0.310ms
INFO 05-06 15:32:00.187283.187283 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 3.0517578125e-05 seconds
INFO 05-06 15:32:00.188821.188821 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007154941558837891s
INFO 05-06 15:32:00.189517.189517 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005543231964111328 seconds
DEBUG 05-06 15:32:00.189349.189349 cuda_h.py:27] end moe_cpu_prep_submit cost 0.647 ms
INFO 05-06 15:32:00.202635.202635 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012723445892333984s
DEBUG 05-06 15:32:00.202501.202501 cuda_h.py:27] end moe_wait_copy_tasks cost 12.891 ms
DEBUG 05-06 15:32:00.206926.206926 cuda_h.py:27] end moe_vllm_forward cost 3.719 ms
DEBUG 05-06 15:32:00.206738.206738 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 15:32:00.206975.206975 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:32:00.207792.207792 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.492ms (seq_len=128 cg=False)
DEBUG 05-06 15:32:00.207160.207160 cuda_h.py:27] end *layer_moe_fused cost 20.894 ms
DEBUG 05-06 15:32:00.208462.208462 cuda_h.py:27] end prefill_merge_scale cost 0.453 ms
DEBUG 05-06 15:32:00.208128.208128 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.043 ms
DEBUG 05-06 15:32:00.208134.208134 cuda_h.py:27] end prefill_layer cost 29.319 ms
DEBUG 05-06 15:32:00.208137.208137 lmp.py:1394] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 15:32:00.208231.208231 lmp.py:1350] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 15:32:00.208931.208931 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 15:32:00.216249.216249 cuda_h.py:27] end prefill_attn cost 7.266 ms
DEBUG 05-06 15:32:00.216053.216053 cuda_h.py:27] end prefill_ffn_prep cost 0.373 ms
DEBUG 05-06 15:32:00.217054.217054 cuda_h.py:27] end prefill_gate cost 0.416 ms
experts_cpu_alloc {'expert_ids': [23, 59, 11, 63, 39, 87, 31, 27, 91, 123, 111, 55, 43, 115, 16, 28, 124, 88, 8, 96, 56, 84, 12, 60, 104, 76, 24, 48, 40, 52, 9, 93, 29, 17, 21, 97, 53, 117, 13, 105, 121, 85, 77, 33, 81, 89, 49, 57, 113, 109, 94, 14, 26, 50, 74, 34, 106, 78, 102, 82, 30, 58, 18, 38, 118, 122], 'token_total': 995, 'token_per_expert': {23: 1, 59: 1, 11: 2, 63: 2, 39: 4, 87: 4, 31: 5, 27: 6, 91: 8, 123: 15, 111: 16, 55: 19, 43: 31, 115: 40, 16: 1, 28: 1, 124: 1, 88: 3, 8: 4, 96: 5, 56: 8, 84: 8, 12: 9, 60: 11, 104: 16, 76: 17, 24: 35, 48: 35, 40: 48, 52: 53, 9: 1, 93: 1, 29: 2, 17: 3, 21: 3, 97: 3, 53: 5, 117: 5, 13: 7, 105: 7, 121: 7, 85: 9, 77: 17, 33: 21, 81: 21, 89: 30, 49: 33, 57: 38, 113: 113, 109: 114, 94: 1, 14: 2, 26: 2, 50: 3, 74: 3, 34: 4, 106: 4, 78: 6, 102: 6, 82: 8, 30: 12, 58: 14, 18: 16, 38: 18, 118: 20, 122: 27}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 35, 79], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 19, 'token_total': 846, 'token_per_expert': {3: 282, 7: 257, 15: 121, 35: 109, 79: 77}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 32, 36], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 695, 'token_per_expert': {0: 269, 4: 257, 32: 81, 36: 88}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 45, 65], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 24, 'token_total': 909, 'token_per_expert': {1: 265, 5: 278, 45: 238, 65: 128}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 70, 90], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 20, 'token_total': 651, 'token_per_expert': {2: 261, 6: 258, 70: 84, 90: 48}}
INFO 05-06 15:32:00.218894.218894 lmp.py:1839] [layer_moe_fused] layer=28 prefix: 0.407ms alloc: 0.307ms
INFO 05-06 15:32:00.218791.218791 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.5987625122070312e-05 seconds
INFO 05-06 15:32:00.219606.219606 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007569789886474609s
INFO 05-06 15:32:00.220004.220004 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005476474761962891 seconds
DEBUG 05-06 15:32:00.220328.220328 cuda_h.py:27] end moe_cpu_prep_submit cost 0.876 ms
INFO 05-06 15:32:00.227492.227492 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.00632023811340332s
DEBUG 05-06 15:32:00.227145.227145 cuda_h.py:27] end moe_wait_copy_tasks cost 6.469 ms
DEBUG 05-06 15:32:00.231586.231586 cuda_h.py:27] end moe_vllm_forward cost 4.008 ms
DEBUG 05-06 15:32:00.233888.233888 cuda_h.py:27] end moe_cpu_merge cost 1.011 ms
DEBUG 05-06 15:32:00.233136.233136 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:32:00.233244.233244 lmp.py:1953] [layer_moe_fused] vllm triton time: 5.683ms (seq_len=128 cg=False)
DEBUG 05-06 15:32:00.233080.233080 cuda_h.py:27] end *layer_moe_fused cost 15.952 ms
DEBUG 05-06 15:32:00.234639.234639 cuda_h.py:27] end prefill_merge_scale cost 0.428 ms
DEBUG 05-06 15:32:00.234159.234159 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:32:00.234323.234323 cuda_h.py:27] end prefill_layer cost 26.085 ms
DEBUG 05-06 15:32:00.234042.234042 lmp.py:1394] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 15:32:00.234374.234374 lmp.py:1350] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 15:32:00.234881.234881 cuda_h.py:27] end prefill_ln cost 0.201 ms
DEBUG 05-06 15:32:00.241703.241703 cuda_h.py:27] end prefill_attn cost 6.303 ms
DEBUG 05-06 15:32:00.241937.241937 cuda_h.py:27] end prefill_ffn_prep cost 0.373 ms
DEBUG 05-06 15:32:00.242918.242918 cuda_h.py:27] end prefill_gate cost 0.406 ms
experts_cpu_alloc {'expert_ids': [15, 91, 47, 95, 99, 31, 35, 43, 55, 107, 11, 59, 115, 83, 79, 119, 111, 51, 28, 104, 76, 80, 92, 108, 44, 56, 48, 60, 96, 124, 40, 72, 73, 113, 109, 69, 45, 21, 33, 9, 41, 13, 101, 57, 85, 89, 93, 121, 105, 46, 82, 62, 90, 58, 126, 94, 54, 38, 10, 78, 114], 'token_total': 876, 'token_per_expert': {15: 1, 91: 1, 47: 2, 95: 2, 99: 2, 31: 3, 35: 3, 43: 3, 55: 4, 107: 5, 11: 7, 59: 7, 115: 9, 83: 10, 79: 21, 119: 21, 111: 26, 51: 29, 28: 1, 104: 1, 76: 2, 80: 2, 92: 2, 108: 11, 44: 13, 56: 26, 48: 27, 60: 28, 96: 28, 124: 32, 40: 35, 72: 35, 73: 1, 113: 1, 109: 2, 69: 3, 45: 4, 21: 5, 33: 5, 9: 7, 41: 9, 13: 10, 101: 10, 57: 17, 85: 21, 89: 26, 93: 37, 121: 63, 105: 92, 46: 1, 82: 1, 62: 2, 90: 3, 58: 4, 126: 4, 94: 6, 54: 8, 38: 11, 10: 37, 78: 42, 114: 45}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 63, 103], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 22, 'token_total': 615, 'token_per_expert': {3: 256, 7: 275, 63: 54, 103: 30}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 112], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 18, 'token_total': 822, 'token_per_expert': {0: 272, 4: 256, 36: 40, 112: 254}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 77], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 21, 'token_total': 820, 'token_per_expert': {1: 260, 5: 256, 17: 206, 77: 98}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 110], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 16, 'token_total': 963, 'token_per_expert': {2: 352, 6: 287, 30: 160, 110: 164}}
INFO 05-06 15:32:00.243805.243805 lmp.py:1839] [layer_moe_fused] layer=29 prefix: 0.407ms alloc: 0.305ms
INFO 05-06 15:32:00.243702.243702 lmp.py:1853] [layer_moe_fused] get_experts_task_ids time: 2.574920654296875e-05 seconds
INFO 05-06 15:32:00.244145.244145 lmp.py:1861] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007851123809814453s
INFO 05-06 15:32:00.245397.245397 lmp.py:1891] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005476474761962891 seconds
DEBUG 05-06 15:32:00.245379.245379 cuda_h.py:27] end moe_cpu_prep_submit cost 0.805 ms
INFO 05-06 15:32:00.266868.266868 lmp.py:1904] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.020483732223510742s
DEBUG 05-06 15:32:00.266123.266123 cuda_h.py:27] end moe_wait_copy_tasks cost 20.619 ms
DEBUG 05-06 15:32:00.270478.270478 cuda_h.py:27] end moe_vllm_forward cost 3.639 ms
DEBUG 05-06 15:32:00.270045.270045 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 15:32:00.270565.270565 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:32:00.270574.270574 lmp.py:1953] [layer_moe_fused] vllm triton time: 4.321ms (seq_len=128 cg=False)
DEBUG 05-06 15:32:00.271989.271989 cuda_h.py:27] end *layer_moe_fused cost 28.381 ms
DEBUG 05-06 15:32:00.271562.271562 cuda_h.py:27] end prefill_merge_scale cost 0.439 ms
DEBUG 05-06 15:32:00.272036.272036 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.041 ms
DEBUG 05-06 15:32:00.272869.272869 cuda_h.py:27] end prefill_layer cost 37.512 ms
DEBUG 05-06 15:32:00.272111.272111 lmp.py:1394] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 15:32:00.272728.272728 cuda_h.py:27] end prefill_step cost 1097.054 ms
INFO 05-06 15:32:00.272665.272665 lmp.py:1397] prefill time: 1.227203369140625 seconds
INFO 05-06 15:32:00.282702.282702 lmp.py:1409] Static-KV prefill complete; seqlens set to 128.
WARNING 05-06 15:32:00.282627.282627 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 15:32:00.283648.283648 helper.py:35]   NaN count (hidden): 720896
WARNING 05-06 15:32:00.283706.283706 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 15:32:00.283220.283220 helper.py:39]   NaN count (normed): 720896
WARNING 05-06 15:32:00.288560.288560 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 15:32:00.288571.288571 helper.py:50]   NaN count: 524288
WARNING 05-06 15:32:00.288237.288237 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 15:32:00.291666.291666 cuda_h.py:27] end init_inputs_tokens cost 8.877 ms
DEBUG 05-06 15:32:00.291960.291960 lmp.py:1510] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 15:32:00.291638.291638 lmp.py:1516] ---- decode step 0 layer 0 ----
DEBUG 05-06 15:32:00.299278.299278 cuda_h.py:27] end decode_layer cost 7.838 ms
DEBUG 05-06 15:32:00.299711.299711 lmp.py:1516] ---- decode step 0 layer 1 ----
DEBUG 05-06 15:32:00.305324.305324 cuda_h.py:27] end decode_layer cost 6.414 ms
DEBUG 05-06 15:32:00.306425.306425 lmp.py:1516] ---- decode step 0 layer 2 ----
DEBUG 05-06 15:32:00.311765.311765 cuda_h.py:27] end decode_layer cost 4.987 ms
DEBUG 05-06 15:32:00.311462.311462 lmp.py:1516] ---- decode step 0 layer 3 ----
DEBUG 05-06 15:32:00.316988.316988 cuda_h.py:27] end decode_layer cost 5.016 ms
DEBUG 05-06 15:32:00.316831.316831 lmp.py:1516] ---- decode step 0 layer 4 ----
DEBUG 05-06 15:32:00.321744.321744 cuda_h.py:27] end decode_layer cost 4.916 ms
DEBUG 05-06 15:32:00.321349.321349 lmp.py:1516] ---- decode step 0 layer 5 ----
DEBUG 05-06 15:32:00.326648.326648 cuda_h.py:27] end decode_layer cost 5.166 ms
DEBUG 05-06 15:32:00.326345.326345 lmp.py:1516] ---- decode step 0 layer 6 ----
DEBUG 05-06 15:32:00.331046.331046 cuda_h.py:27] end decode_layer cost 4.900 ms
DEBUG 05-06 15:32:00.331551.331551 lmp.py:1516] ---- decode step 0 layer 7 ----
DEBUG 05-06 15:32:00.336321.336321 cuda_h.py:27] end decode_layer cost 4.985 ms
DEBUG 05-06 15:32:00.336356.336356 lmp.py:1516] ---- decode step 0 layer 8 ----
DEBUG 05-06 15:32:00.341810.341810 cuda_h.py:27] end decode_layer cost 4.859 ms
DEBUG 05-06 15:32:00.341130.341130 lmp.py:1516] ---- decode step 0 layer 9 ----
DEBUG 05-06 15:32:00.346592.346592 cuda_h.py:27] end decode_layer cost 4.900 ms
DEBUG 05-06 15:32:00.346336.346336 lmp.py:1516] ---- decode step 0 layer 10 ----
DEBUG 05-06 15:32:00.351652.351652 cuda_h.py:27] end decode_layer cost 4.898 ms
DEBUG 05-06 15:32:00.351827.351827 lmp.py:1516] ---- decode step 0 layer 11 ----
DEBUG 05-06 15:32:00.356735.356735 cuda_h.py:27] end decode_layer cost 5.158 ms
DEBUG 05-06 15:32:00.356101.356101 lmp.py:1516] ---- decode step 0 layer 12 ----
DEBUG 05-06 15:32:00.361884.361884 cuda_h.py:27] end decode_layer cost 4.786 ms
DEBUG 05-06 15:32:00.361634.361634 lmp.py:1516] ---- decode step 0 layer 13 ----
DEBUG 05-06 15:32:00.366493.366493 cuda_h.py:27] end decode_layer cost 4.876 ms
DEBUG 05-06 15:32:00.366713.366713 lmp.py:1516] ---- decode step 0 layer 14 ----
DEBUG 05-06 15:32:00.371944.371944 cuda_h.py:27] end decode_layer cost 4.905 ms
DEBUG 05-06 15:32:00.371641.371641 lmp.py:1516] ---- decode step 0 layer 15 ----
DEBUG 05-06 15:32:00.376722.376722 cuda_h.py:27] end decode_layer cost 4.969 ms
DEBUG 05-06 15:32:00.376088.376088 lmp.py:1516] ---- decode step 0 layer 16 ----
DEBUG 05-06 15:32:00.381754.381754 cuda_h.py:27] end decode_layer cost 4.840 ms
DEBUG 05-06 15:32:00.381974.381974 lmp.py:1516] ---- decode step 0 layer 17 ----
DEBUG 05-06 15:32:00.386227.386227 cuda_h.py:27] end decode_layer cost 5.167 ms
DEBUG 05-06 15:32:00.386262.386262 lmp.py:1516] ---- decode step 0 layer 18 ----
DEBUG 05-06 15:32:00.391886.391886 cuda_h.py:27] end decode_layer cost 5.370 ms
DEBUG 05-06 15:32:00.391444.391444 lmp.py:1516] ---- decode step 0 layer 19 ----
DEBUG 05-06 15:32:00.396014.396014 cuda_h.py:27] end decode_layer cost 4.944 ms
DEBUG 05-06 15:32:00.396526.396526 lmp.py:1516] ---- decode step 0 layer 20 ----
DEBUG 05-06 15:32:00.401854.401854 cuda_h.py:27] end decode_layer cost 4.871 ms
DEBUG 05-06 15:32:00.401413.401413 lmp.py:1516] ---- decode step 0 layer 21 ----
DEBUG 05-06 15:32:00.406717.406717 cuda_h.py:27] end decode_layer cost 4.924 ms
DEBUG 05-06 15:32:00.406798.406798 lmp.py:1516] ---- decode step 0 layer 22 ----
DEBUG 05-06 15:32:00.411448.411448 cuda_h.py:27] end decode_layer cost 4.968 ms
DEBUG 05-06 15:32:00.411960.411960 lmp.py:1516] ---- decode step 0 layer 23 ----
DEBUG 05-06 15:32:00.416086.416086 cuda_h.py:27] end decode_layer cost 5.144 ms
DEBUG 05-06 15:32:00.417453.417453 lmp.py:1516] ---- decode step 0 layer 24 ----
DEBUG 05-06 15:32:00.421405.421405 cuda_h.py:27] end decode_layer cost 4.910 ms
DEBUG 05-06 15:32:00.421010.421010 lmp.py:1516] ---- decode step 0 layer 25 ----
DEBUG 05-06 15:32:00.426149.426149 cuda_h.py:27] end decode_layer cost 4.942 ms
DEBUG 05-06 15:32:00.426469.426469 lmp.py:1516] ---- decode step 0 layer 26 ----
DEBUG 05-06 15:32:00.431651.431651 cuda_h.py:27] end decode_layer cost 4.834 ms
DEBUG 05-06 15:32:00.431209.431209 lmp.py:1516] ---- decode step 0 layer 27 ----
DEBUG 05-06 15:32:00.436537.436537 cuda_h.py:27] end decode_layer cost 4.836 ms
DEBUG 05-06 15:32:00.436857.436857 lmp.py:1516] ---- decode step 0 layer 28 ----
DEBUG 05-06 15:32:00.441504.441504 cuda_h.py:27] end decode_layer cost 4.896 ms
DEBUG 05-06 15:32:00.441063.441063 lmp.py:1516] ---- decode step 0 layer 29 ----
DEBUG 05-06 15:32:00.446799.446799 cuda_h.py:27] end decode_layer cost 5.172 ms
DEBUG 05-06 15:32:00.447259.447259 cuda_h.py:27] end decode_step cost 164.659 ms
INFO 05-06 15:32:00.447499.447499 lmp.py:1564] decode step 0 time: 0.16469812393188477 seconds
WARNING 05-06 15:32:00.447988.447988 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 15:32:00.447070.447070 helper.py:35]   NaN count (hidden): 11264
WARNING 05-06 15:32:00.447600.447600 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 15:32:00.448193.448193 helper.py:39]   NaN count (normed): 11264
WARNING 05-06 15:32:00.453822.453822 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 15:32:00.453323.453323 helper.py:50]   NaN count: 1048576
WARNING 05-06 15:32:00.453430.453430 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 15:32:00.454398.454398 cuda_h.py:27] end init_inputs_tokens cost 7.611 ms
DEBUG 05-06 15:32:00.454910.454910 lmp.py:1510] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 15:32:00.454488.454488 lmp.py:1516] ---- decode step 1 layer 0 ----
DEBUG 05-06 15:32:00.459369.459369 cuda_h.py:27] end decode_layer cost 4.962 ms
DEBUG 05-06 15:32:00.459597.459597 lmp.py:1516] ---- decode step 1 layer 1 ----
DEBUG 05-06 15:32:00.464908.464908 cuda_h.py:27] end decode_layer cost 4.928 ms
DEBUG 05-06 15:32:00.464718.464718 lmp.py:1516] ---- decode step 1 layer 2 ----
DEBUG 05-06 15:32:00.469940.469940 cuda_h.py:27] end decode_layer cost 5.048 ms
DEBUG 05-06 15:32:00.469975.469975 lmp.py:1516] ---- decode step 1 layer 3 ----
DEBUG 05-06 15:32:00.474637.474637 cuda_h.py:27] end decode_layer cost 4.942 ms
DEBUG 05-06 15:32:00.475534.475534 lmp.py:1516] ---- decode step 1 layer 4 ----
DEBUG 05-06 15:32:00.479094.479094 cuda_h.py:27] end decode_layer cost 4.868 ms
DEBUG 05-06 15:32:00.479844.479844 lmp.py:1516] ---- decode step 1 layer 5 ----
DEBUG 05-06 15:32:00.485421.485421 cuda_h.py:27] end decode_layer cost 5.160 ms
DEBUG 05-06 15:32:00.485218.485218 lmp.py:1516] ---- decode step 1 layer 6 ----
DEBUG 05-06 15:32:00.490716.490716 cuda_h.py:27] end decode_layer cost 5.382 ms
DEBUG 05-06 15:32:00.490944.490944 lmp.py:1516] ---- decode step 1 layer 7 ----
DEBUG 05-06 15:32:00.495812.495812 cuda_h.py:27] end decode_layer cost 4.952 ms
DEBUG 05-06 15:32:00.495860.495860 lmp.py:1516] ---- decode step 1 layer 8 ----
DEBUG 05-06 15:32:00.500004.500004 cuda_h.py:27] end decode_layer cost 4.875 ms
DEBUG 05-06 15:32:00.500947.500947 lmp.py:1516] ---- decode step 1 layer 9 ----
DEBUG 05-06 15:32:00.505478.505478 cuda_h.py:27] end decode_layer cost 4.985 ms
DEBUG 05-06 15:32:00.505420.505420 lmp.py:1516] ---- decode step 1 layer 10 ----
DEBUG 05-06 15:32:00.510670.510670 cuda_h.py:27] end decode_layer cost 5.269 ms
DEBUG 05-06 15:32:00.510612.510612 lmp.py:1516] ---- decode step 1 layer 11 ----
DEBUG 05-06 15:32:00.516557.516557 cuda_h.py:27] end decode_layer cost 5.255 ms
DEBUG 05-06 15:32:00.516353.516353 lmp.py:1516] ---- decode step 1 layer 12 ----
DEBUG 05-06 15:32:00.521960.521960 cuda_h.py:27] end decode_layer cost 5.631 ms
DEBUG 05-06 15:32:00.522115.522115 lmp.py:1516] ---- decode step 1 layer 13 ----
DEBUG 05-06 15:32:00.527566.527566 cuda_h.py:27] end decode_layer cost 5.174 ms
DEBUG 05-06 15:32:00.527217.527217 lmp.py:1516] ---- decode step 1 layer 14 ----
DEBUG 05-06 15:32:00.532986.532986 cuda_h.py:27] end decode_layer cost 4.950 ms
DEBUG 05-06 15:32:00.532021.532021 lmp.py:1516] ---- decode step 1 layer 15 ----
DEBUG 05-06 15:32:00.537949.537949 cuda_h.py:27] end decode_layer cost 4.963 ms
DEBUG 05-06 15:32:00.537361.537361 lmp.py:1516] ---- decode step 1 layer 16 ----
DEBUG 05-06 15:32:00.542142.542142 cuda_h.py:27] end decode_layer cost 4.924 ms
DEBUG 05-06 15:32:00.542363.542363 lmp.py:1516] ---- decode step 1 layer 17 ----
DEBUG 05-06 15:32:00.547548.547548 cuda_h.py:27] end decode_layer cost 5.116 ms
DEBUG 05-06 15:32:00.547053.547053 lmp.py:1516] ---- decode step 1 layer 18 ----
DEBUG 05-06 15:32:00.552002.552002 cuda_h.py:27] end decode_layer cost 4.802 ms
DEBUG 05-06 15:32:00.552706.552706 lmp.py:1516] ---- decode step 1 layer 19 ----
DEBUG 05-06 15:32:00.557732.557732 cuda_h.py:27] end decode_layer cost 4.930 ms
DEBUG 05-06 15:32:00.557860.557860 lmp.py:1516] ---- decode step 1 layer 20 ----
DEBUG 05-06 15:32:00.562847.562847 cuda_h.py:27] end decode_layer cost 4.935 ms
DEBUG 05-06 15:32:00.562974.562974 lmp.py:1516] ---- decode step 1 layer 21 ----
DEBUG 05-06 15:32:00.567806.567806 cuda_h.py:27] end decode_layer cost 4.857 ms
DEBUG 05-06 15:32:00.567265.567265 lmp.py:1516] ---- decode step 1 layer 22 ----
DEBUG 05-06 15:32:00.572856.572856 cuda_h.py:27] end decode_layer cost 4.784 ms
DEBUG 05-06 15:32:00.572937.572937 lmp.py:1516] ---- decode step 1 layer 23 ----
DEBUG 05-06 15:32:00.577372.577372 cuda_h.py:27] end decode_layer cost 5.056 ms
DEBUG 05-06 15:32:00.577830.577830 lmp.py:1516] ---- decode step 1 layer 24 ----
DEBUG 05-06 15:32:00.582304.582304 cuda_h.py:27] end decode_layer cost 4.837 ms
DEBUG 05-06 15:32:00.582545.582545 lmp.py:1516] ---- decode step 1 layer 25 ----
DEBUG 05-06 15:32:00.586792.586792 cuda_h.py:27] end decode_layer cost 4.812 ms
DEBUG 05-06 15:32:00.587774.587774 lmp.py:1516] ---- decode step 1 layer 26 ----
DEBUG 05-06 15:32:00.591148.591148 cuda_h.py:27] end decode_layer cost 4.830 ms
DEBUG 05-06 15:32:00.591819.591819 lmp.py:1516] ---- decode step 1 layer 27 ----
DEBUG 05-06 15:32:00.596167.596167 cuda_h.py:27] end decode_layer cost 4.852 ms
DEBUG 05-06 15:32:00.596263.596263 lmp.py:1516] ---- decode step 1 layer 28 ----
DEBUG 05-06 15:32:00.601126.601126 cuda_h.py:27] end decode_layer cost 4.810 ms
DEBUG 05-06 15:32:00.601108.601108 lmp.py:1516] ---- decode step 1 layer 29 ----
DEBUG 05-06 15:32:00.606610.606610 cuda_h.py:27] end decode_layer cost 5.105 ms
DEBUG 05-06 15:32:00.606679.606679 cuda_h.py:27] end decode_step cost 159.788 ms
INFO 05-06 15:32:00.606488.606488 lmp.py:1564] decode step 1 time: 0.15982580184936523 seconds
WARNING 05-06 15:32:00.607568.607568 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 15:32:00.607087.607087 helper.py:35]   NaN count (hidden): 11264
WARNING 05-06 15:32:00.607583.607583 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 15:32:00.607156.607156 helper.py:39]   NaN count (normed): 11264
WARNING 05-06 15:32:00.612484.612484 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 15:32:00.613753.613753 helper.py:50]   NaN count: 1048576
WARNING 05-06 15:32:00.613145.613145 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 15:32:00.614096.614096 cuda_h.py:27] end init_inputs_tokens cost 7.608 ms
DEBUG 05-06 15:32:00.614085.614085 lmp.py:1510] decode step 2 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 15:32:00.614981.614981 lmp.py:1516] ---- decode step 2 layer 0 ----
DEBUG 05-06 15:32:00.619691.619691 cuda_h.py:27] end decode_layer cost 4.765 ms
DEBUG 05-06 15:32:00.619196.619196 lmp.py:1516] ---- decode step 2 layer 1 ----
DEBUG 05-06 15:32:00.624466.624466 cuda_h.py:27] end decode_layer cost 4.898 ms
DEBUG 05-06 15:32:00.624025.624025 lmp.py:1516] ---- decode step 2 layer 2 ----
DEBUG 05-06 15:32:00.629895.629895 cuda_h.py:27] end decode_layer cost 4.815 ms
DEBUG 05-06 15:32:00.629830.629830 lmp.py:1516] ---- decode step 2 layer 3 ----
DEBUG 05-06 15:32:00.634411.634411 cuda_h.py:27] end decode_layer cost 4.882 ms
DEBUG 05-06 15:32:00.634870.634870 lmp.py:1516] ---- decode step 2 layer 4 ----
DEBUG 05-06 15:32:00.639792.639792 cuda_h.py:27] end decode_layer cost 4.783 ms
DEBUG 05-06 15:32:00.639297.639297 lmp.py:1516] ---- decode step 2 layer 5 ----
DEBUG 05-06 15:32:00.644814.644814 cuda_h.py:27] end decode_layer cost 5.150 ms
DEBUG 05-06 15:32:00.644180.644180 lmp.py:1516] ---- decode step 2 layer 6 ----
DEBUG 05-06 15:32:00.649648.649648 cuda_h.py:27] end decode_layer cost 4.869 ms
DEBUG 05-06 15:32:00.649776.649776 lmp.py:1516] ---- decode step 2 layer 7 ----
DEBUG 05-06 15:32:00.654507.654507 cuda_h.py:27] end decode_layer cost 4.818 ms
DEBUG 05-06 15:32:00.654489.654489 lmp.py:1516] ---- decode step 2 layer 8 ----
DEBUG 05-06 15:32:00.659669.659669 cuda_h.py:27] end decode_layer cost 4.762 ms
DEBUG 05-06 15:32:00.659459.659459 lmp.py:1516] ---- decode step 2 layer 9 ----
DEBUG 05-06 15:32:00.663702.663702 cuda_h.py:27] end decode_layer cost 4.878 ms
DEBUG 05-06 15:32:00.664306.664306 lmp.py:1516] ---- decode step 2 layer 10 ----
DEBUG 05-06 15:32:00.668702.668702 cuda_h.py:27] end decode_layer cost 4.886 ms
DEBUG 05-06 15:32:00.668830.668830 lmp.py:1516] ---- decode step 2 layer 11 ----
DEBUG 05-06 15:32:00.674702.674702 cuda_h.py:27] end decode_layer cost 5.098 ms
DEBUG 05-06 15:32:00.674923.674923 lmp.py:1516] ---- decode step 2 layer 12 ----
DEBUG 05-06 15:32:00.678064.678064 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 15:32:00.679523.679523 lmp.py:1516] ---- decode step 2 layer 13 ----
DEBUG 05-06 15:32:00.683018.683018 cuda_h.py:27] end decode_layer cost 4.889 ms
DEBUG 05-06 15:32:00.683576.683576 lmp.py:1516] ---- decode step 2 layer 14 ----
DEBUG 05-06 15:32:00.688968.688968 cuda_h.py:27] end decode_layer cost 4.778 ms
DEBUG 05-06 15:32:00.688334.688334 lmp.py:1516] ---- decode step 2 layer 15 ----
DEBUG 05-06 15:32:00.693671.693671 cuda_h.py:27] end decode_layer cost 4.913 ms
DEBUG 05-06 15:32:00.693044.693044 lmp.py:1516] ---- decode step 2 layer 16 ----
DEBUG 05-06 15:32:00.699274.699274 cuda_h.py:27] end decode_layer cost 5.465 ms
DEBUG 05-06 15:32:00.699309.699309 lmp.py:1516] ---- decode step 2 layer 17 ----
DEBUG 05-06 15:32:00.704993.704993 cuda_h.py:27] end decode_layer cost 5.204 ms
DEBUG 05-06 15:32:00.704075.704075 lmp.py:1516] ---- decode step 2 layer 18 ----
DEBUG 05-06 15:32:00.709496.709496 cuda_h.py:27] end decode_layer cost 4.870 ms
DEBUG 05-06 15:32:00.709909.709909 lmp.py:1516] ---- decode step 2 layer 19 ----
DEBUG 05-06 15:32:00.714007.714007 cuda_h.py:27] end decode_layer cost 4.913 ms
DEBUG 05-06 15:32:00.714513.714513 lmp.py:1516] ---- decode step 2 layer 20 ----
DEBUG 05-06 15:32:00.719170.719170 cuda_h.py:27] end decode_layer cost 4.798 ms
DEBUG 05-06 15:32:00.719437.719437 lmp.py:1516] ---- decode step 2 layer 21 ----
DEBUG 05-06 15:32:00.724461.724461 cuda_h.py:27] end decode_layer cost 4.857 ms
DEBUG 05-06 15:32:00.724065.724065 lmp.py:1516] ---- decode step 2 layer 22 ----
DEBUG 05-06 15:32:00.729155.729155 cuda_h.py:27] end decode_layer cost 4.836 ms
DEBUG 05-06 15:32:00.729421.729421 lmp.py:1516] ---- decode step 2 layer 23 ----
DEBUG 05-06 15:32:00.734690.734690 cuda_h.py:27] end decode_layer cost 5.038 ms
DEBUG 05-06 15:32:00.734387.734387 lmp.py:1516] ---- decode step 2 layer 24 ----
DEBUG 05-06 15:32:00.739123.739123 cuda_h.py:27] end decode_layer cost 4.786 ms
DEBUG 05-06 15:32:00.739059.739059 lmp.py:1516] ---- decode step 2 layer 25 ----
DEBUG 05-06 15:32:00.744258.744258 cuda_h.py:27] end decode_layer cost 4.951 ms
DEBUG 05-06 15:32:00.744625.744625 lmp.py:1516] ---- decode step 2 layer 26 ----
DEBUG 05-06 15:32:00.748181.748181 cuda_h.py:27] end decode_layer cost 4.759 ms
DEBUG 05-06 15:32:00.748640.748640 lmp.py:1516] ---- decode step 2 layer 27 ----
DEBUG 05-06 15:32:00.753465.753465 cuda_h.py:27] end decode_layer cost 4.851 ms
DEBUG 05-06 15:32:00.753255.753255 lmp.py:1516] ---- decode step 2 layer 28 ----
DEBUG 05-06 15:32:00.758482.758482 cuda_h.py:27] end decode_layer cost 4.797 ms
DEBUG 05-06 15:32:00.758987.758987 lmp.py:1516] ---- decode step 2 layer 29 ----
DEBUG 05-06 15:32:00.763240.763240 cuda_h.py:27] end decode_layer cost 5.166 ms
DEBUG 05-06 15:32:00.763885.763885 cuda_h.py:27] end decode_step cost 156.965 ms
INFO 05-06 15:32:00.764079.764079 lmp.py:1564] decode step 2 time: 0.15700459480285645 seconds
Time taken: 5.550280410796404 seconds
X512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x62ed243f6260, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
CPUInfer[0x62ed2d4e7b70]: Goodbye
CPUInfer[0x62ed076c2d70]: Goodbye
