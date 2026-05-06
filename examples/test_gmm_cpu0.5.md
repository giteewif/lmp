here pin
INFO 05-06 11:48:16.709082.709082 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-06 11:48:17.267597.267597 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-06 11:48:17.703464.703464 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-06 11:48:17.703933.703933 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 0.994s
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
INFO 05-06 11:48:25.206741.206741 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-06 11:48:25.654666.654666 cuda_h.py:27] end init_cmv_hmv cost 447.685 ms
DEBUG 05-06 11:48:25.662072.662072 cuda_memory_view.py:1366] 
DEBUG 05-06 11:48:25.662072.662072 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.0026674270629882812
DEBUG 05-06 11:48:25.682274.682274 mlpmodule.py:993] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-06 11:48:25.682439.682439 cuda_memory_view.py:1370] 
DEBUG 05-06 11:48:25.682439.682439 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.01947164535522461
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-06 11:48:27.602374.602374 lmp.py:255] init kt-kernel layer 0 ok
INFO 05-06 11:48:28.391706.391706 lmp.py:255] init kt-kernel layer 1 ok
INFO 05-06 11:48:29.203325.203325 lmp.py:255] init kt-kernel layer 2 ok
INFO 05-06 11:48:30.016367.016367 lmp.py:255] init kt-kernel layer 3 ok
INFO 05-06 11:48:30.841035.841035 lmp.py:255] init kt-kernel layer 4 ok
INFO 05-06 11:48:31.671342.671342 lmp.py:255] init kt-kernel layer 5 ok
INFO 05-06 11:48:32.505752.505752 lmp.py:255] init kt-kernel layer 6 ok
INFO 05-06 11:48:33.341915.341915 lmp.py:255] init kt-kernel layer 7 ok
INFO 05-06 11:48:34.162211.162211 lmp.py:255] init kt-kernel layer 8 ok
INFO 05-06 11:48:34.987120.987120 lmp.py:255] init kt-kernel layer 9 ok
INFO 05-06 11:48:35.817501.817501 lmp.py:255] init kt-kernel layer 10 ok
INFO 05-06 11:48:36.641432.641432 lmp.py:255] init kt-kernel layer 11 ok
INFO 05-06 11:48:37.473717.473717 lmp.py:255] init kt-kernel layer 12 ok
INFO 05-06 11:48:38.327920.327920 lmp.py:255] init kt-kernel layer 13 ok
INFO 05-06 11:48:39.163650.163650 lmp.py:255] init kt-kernel layer 14 ok
INFO 05-06 11:48:40.001969.001969 lmp.py:255] init kt-kernel layer 15 ok
INFO 05-06 11:48:40.837890.837890 lmp.py:255] init kt-kernel layer 16 ok
INFO 05-06 11:48:41.672302.672302 lmp.py:255] init kt-kernel layer 17 ok
INFO 05-06 11:48:42.500334.500334 lmp.py:255] init kt-kernel layer 18 ok
INFO 05-06 11:48:43.338010.338010 lmp.py:255] init kt-kernel layer 19 ok
INFO 05-06 11:48:44.154482.154482 lmp.py:255] init kt-kernel layer 20 ok
INFO 05-06 11:48:44.990456.990456 lmp.py:255] init kt-kernel layer 21 ok
INFO 05-06 11:48:45.808387.808387 lmp.py:255] init kt-kernel layer 22 ok
CPUInfer[0x5f378c247d50]: Hello
WorkerPool[0x5f378c247600] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x5f37a6ebdfb0]: Hello
WorkerPool[0x5f37be761e80] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVINFO 05-06 11:48:46.629688.629688 lmp.py:255] init kt-kernel layer 23 ok
INFO 05-06 11:48:47.469580.469580 lmp.py:255] init kt-kernel layer 24 ok
INFO 05-06 11:48:48.295356.295356 lmp.py:255] init kt-kernel layer 25 ok
INFO 05-06 11:48:49.127701.127701 lmp.py:255] init kt-kernel layer 26 ok
INFO 05-06 11:48:49.947728.947728 lmp.py:255] init kt-kernel layer 27 ok
INFO 05-06 11:48:50.757269.757269 lmp.py:255] init kt-kernel layer 28 ok
INFO 05-06 11:48:51.550172.550172 lmp.py:255] init kt-kernel layer 29 ok
INFO 05-06 11:48:52.577811.577811 lmp.py:186] vLLM Triton fused-MoE enabled (CUDAGraph=False).
generate input ids cost 0.05460047721862793 s
DEBUG 05-06 11:48:55.696300.696300 cuda_h.py:27] end generate_input_ids cost 3098.057 ms
DEBUG 05-06 11:48:55.696098.696098 cuda_h.py:27] end init_cache cost 0.048 ms
INFO 05-06 11:48:55.707090.707090 lmp.py:367] _ensure_static_kv_cache (Gemma4 list): 30 layers, 1760.0 MiB on cuda:0
INFO 05-06 11:48:55.707730.707730 lmp.py:1160] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 11:48:55.724114.724114 lmp.py:2794] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 4784365508, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.787078263565146, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-06 11:48:55.724971.724971 lmp.py:2812] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.724755.724755 lmp.py:2812] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.724577.724577 lmp.py:2812] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.724264.724264 lmp.py:2812] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.724848.724848 lmp.py:2812] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725787.725787 lmp.py:2812] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725603.725603 lmp.py:2812] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725125.725125 lmp.py:2812] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725272.725272 lmp.py:2812] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725760.725760 lmp.py:2812] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725431.725431 lmp.py:2812] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725012.725012 lmp.py:2812] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725159.725159 lmp.py:2812] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725330.725330 lmp.py:2812] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.725715.725715 lmp.py:2812] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726882.726882 lmp.py:2812] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726552.726552 lmp.py:2812] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726280.726280 lmp.py:2812] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726712.726712 lmp.py:2812] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726922.726922 lmp.py:2812] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726314.726314 lmp.py:2812] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726578.726578 lmp.py:2812] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726010.726010 lmp.py:2812] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726795.726795 lmp.py:2812] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.726988.726988 lmp.py:2812] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.727151.727151 lmp.py:2812] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.727868.727868 lmp.py:2812] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.727615.727615 lmp.py:2812] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.727570.727570 lmp.py:2812] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:48:55.727856.727856 lmp.py:2812] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 11:48:56.038406.038406 cuda_h.py:27] end init_loading_placement cost 330.839 ms
DEBUG 05-06 11:48:56.038788.038788 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:48:56.038261.038261 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:48:56 client.py:72] load_into_gpu: gemma4-26B-A4B, 02b495be-6b5c-471f-af46-94edede92024
INFO 05-06 11:48:56 client.py:135] Model loaded: gemma4-26B-A4B, 02b495be-6b5c-471f-af46-94edede92024
INFO 05-06 11:48:56 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 02b495be-6b5c-471f-af46-94edede92024
INFO 05-06 11:48:56 client.py:212] Model loaded
DEBUG 05-06 11:48:56.571032.571032 cuda_h.py:27] end init_general_sagl_loading_async cost 533.385 ms
INFO 05-06 11:48:56.625492.625492 lmp.py:3315] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 11:48:56.729268.729268 cuda_h.py:27] end restore_state_dict cost 103.922 ms
WARNING 05-06 11:48:56 [fused_moe.py:1090] Using default MoE config. Performance might be sub-optimal! Config file not found at /mnt/zhengcf3/lmp_env/fslmp/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=32,N=704,device_name=NVIDIA_GeForce_RTX_4090.json
INFO 05-06 11:48:57.824551.824551 lmp.py:1288] vLLM Triton pre-warmup done in 1093.8 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 11:48:57.824670.824670 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:48:57.824388.824388 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:48:57 client.py:72] load_into_gpu: gemma4-26B-A4B, 60b59ffe-c5fa-4dbe-ac8b-d0db846878c1
INFO 05-06 11:48:57 client.py:135] Model loaded: gemma4-26B-A4B, 60b59ffe-c5fa-4dbe-ac8b-d0db846878c1
DEBUG 05-06 11:48:57.900652.900652 cuda_h.py:27] end init_experts_loading_async cost 76.158 ms
DEBUG 05-06 11:48:57.927946.927946 cuda_h.py:27] end init_inputs_tokens cost 26.834 ms
DEBUG 05-06 11:48:57.927327.927327 lmp.py:1347] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 11:48:57.997037.997037 cuda_h.py:27] end prefill_ln cost 70.004 ms
DEBUG 05-06 11:48:58.080138.080138 cuda_h.py:27] end prefill_attn cost 82.341 ms
DEBUG 05-06 11:48:58.080430.080430 cuda_h.py:27] end prefill_ffn_prep cost 0.450 ms
DEBUG 05-06 11:48:58.166523.166523 cuda_h.py:27] end prefill_gate cost 79.318 ms
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 101, 109, 85, 49, 45, 65, 93, 69, 5, 13, 9, 73, 77, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18], 'token_total': 420, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 13: 16, 9: 17, 73: 19, 77: 19, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 51, 55, 67, 75, 83, 91, 103, 115, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1216, 'token_per_expert': {3: 46, 7: 95, 31: 34, 39: 176, 47: 318, 51: 48, 55: 51, 67: 47, 75: 29, 83: 33, 91: 99, 103: 178, 115: 29, 127: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 32, 48, 52, 64, 68, 72, 92, 104, 112, 116, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 758, 'token_per_expert': {0: 73, 16: 48, 32: 43, 48: 41, 52: 43, 64: 27, 68: 170, 72: 35, 92: 16, 104: 43, 112: 23, 116: 18, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 25, 33, 37, 41, 53, 89, 105, 113, 117, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 809, 'token_per_expert': {1: 75, 21: 48, 25: 24, 33: 210, 37: 20, 41: 27, 53: 205, 89: 20, 105: 24, 113: 39, 117: 26, 121: 65, 125: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 46, 50, 54, 70, 74, 78, 90, 110, 118, 122, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 893, 'token_per_expert': {22: 64, 26: 59, 46: 119, 50: 110, 54: 59, 70: 25, 74: 61, 78: 36, 90: 154, 110: 27, 118: 29, 122: 35, 126: 115}}
INFO 05-06 11:48:58.199418.199418 lmp.py:1836] [layer_moe_fused] layer=0 prefix: 32.093ms alloc: 0.293ms
INFO 05-06 11:48:58.199543.199543 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 4.553794860839844e-05 seconds
INFO 05-06 11:48:58.201958.201958 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.002245187759399414s
INFO 05-06 11:48:58.202370.202370 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008938312530517578 seconds
DEBUG 05-06 11:48:58.206738.206738 cuda_h.py:27] end moe_cpu_prep_submit cost 4.532 ms
INFO 05-06 11:48:58.208417.208417 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0021529197692871094s
DEBUG 05-06 11:48:58.209264.209264 cuda_h.py:27] end moe_wait_copy_tasks cost 2.335 ms
DEBUG 05-06 11:48:58.243559.243559 cuda_h.py:27] end moe_vllm_forward cost 33.565 ms
DEBUG 05-06 11:48:58.256212.256212 cuda_h.py:27] end moe_cpu_merge cost 12.649 ms
DEBUG 05-06 11:48:58.256612.256612 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:58.256343.256343 lmp.py:1950] [layer_moe_fused] vllm triton time: 47.365ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.256162.256162 cuda_h.py:27] end *layer_moe_fused cost 90.224 ms
DEBUG 05-06 11:48:58.258949.258949 cuda_h.py:27] end prefill_merge_scale cost 1.679 ms
DEBUG 05-06 11:48:58.258263.258263 cuda_h.py:27] end prefill_layer cost 331.173 ms
DEBUG 05-06 11:48:58.258026.258026 lmp.py:1391] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 11:48:58.258173.258173 lmp.py:1347] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 11:48:58.259591.259591 cuda_h.py:27] end prefill_ln cost 0.212 ms
DEBUG 05-06 11:48:58.262614.262614 cuda_h.py:27] end prefill_attn cost 3.004 ms
DEBUG 05-06 11:48:58.262108.262108 cuda_h.py:27] end prefill_ffn_prep cost 0.414 ms
DEBUG 05-06 11:48:58.264219.264219 cuda_h.py:27] end prefill_gate cost 0.546 ms
experts_cpu_alloc {'expert_ids': [19, 39, 63, 15, 43, 55, 115, 31, 83, 103, 123, 87, 16, 88, 40, 32, 44, 72, 84, 108, 112, 0, 116, 60, 48, 56, 76, 4, 61, 77, 117, 33, 41, 57, 81, 89, 125, 29, 45, 121, 93, 69, 37, 101, 6, 86, 2, 26, 18, 38, 62, 110, 14, 66, 78, 50, 74], 'token_total': 353, 'token_per_expert': {19: 1, 39: 1, 63: 2, 15: 3, 43: 4, 55: 4, 115: 4, 31: 5, 83: 5, 103: 5, 123: 6, 87: 7, 16: 3, 88: 3, 40: 4, 32: 5, 44: 5, 72: 5, 84: 5, 108: 6, 112: 6, 0: 7, 116: 7, 60: 8, 48: 10, 56: 12, 76: 14, 4: 15, 61: 1, 77: 1, 117: 1, 33: 3, 41: 5, 57: 5, 81: 5, 89: 5, 125: 5, 29: 6, 45: 6, 121: 7, 93: 8, 69: 10, 37: 11, 101: 13, 6: 1, 86: 1, 2: 2, 26: 4, 18: 5, 38: 7, 62: 7, 110: 7, 14: 8, 66: 11, 78: 12, 50: 17, 74: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 35, 47, 51, 59, 67, 79, 91, 95, 99, 119, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 27, 'token_total': 627, 'token_per_expert': {3: 26, 7: 33, 11: 11, 27: 15, 35: 24, 47: 40, 51: 61, 59: 39, 67: 120, 79: 26, 91: 7, 95: 22, 99: 123, 119: 27, 127: 53}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 20, 28, 52, 64, 68, 80, 92, 96, 100, 104, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 1016, 'token_per_expert': {8: 114, 12: 55, 20: 64, 28: 67, 52: 272, 64: 25, 68: 179, 80: 50, 92: 24, 96: 34, 100: 50, 104: 22, 120: 30, 124: 30}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 49, 53, 65, 73, 85, 97, 105, 109], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 903, 'token_per_expert': {1: 33, 5: 104, 9: 17, 13: 282, 21: 16, 25: 43, 49: 30, 53: 36, 65: 36, 73: 34, 85: 17, 97: 123, 105: 26, 109: 106}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 22, 30, 34, 42, 46, 54, 82, 90, 94, 98, 106, 118, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1197, 'token_per_expert': {10: 166, 22: 126, 30: 243, 34: 31, 42: 39, 46: 39, 54: 61, 82: 180, 90: 20, 94: 33, 98: 26, 106: 40, 118: 60, 122: 133}}
INFO 05-06 11:48:58.265849.265849 lmp.py:1836] [layer_moe_fused] layer=1 prefix: 0.449ms alloc: 0.427ms
INFO 05-06 11:48:58.265655.265655 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.103515625e-05 seconds
INFO 05-06 11:48:58.266488.266488 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009758472442626953s
INFO 05-06 11:48:58.267941.267941 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006215572357177734 seconds
DEBUG 05-06 11:48:58.268314.268314 cuda_h.py:27] end moe_cpu_prep_submit cost 1.455 ms
INFO 05-06 11:48:58.270008.270008 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0018897056579589844s
DEBUG 05-06 11:48:58.270966.270966 cuda_h.py:27] end moe_wait_copy_tasks cost 2.012 ms
DEBUG 05-06 11:48:58.277242.277242 cuda_h.py:27] end moe_vllm_forward cost 6.856 ms
DEBUG 05-06 11:48:58.277578.277578 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:48:58.278934.278934 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:48:58.278844.278844 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.590ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.278460.278460 cuda_h.py:27] end *layer_moe_fused cost 13.796 ms
DEBUG 05-06 11:48:58.283055.283055 cuda_h.py:27] end prefill_merge_scale cost 5.051 ms
DEBUG 05-06 11:48:58.283700.283700 cuda_h.py:27] end prefill_layer cost 24.650 ms
DEBUG 05-06 11:48:58.283422.283422 lmp.py:1391] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 11:48:58.283198.283198 lmp.py:1347] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 11:48:58.284845.284845 cuda_h.py:27] end prefill_ln cost 0.202 ms
DEBUG 05-06 11:48:58.286941.286941 cuda_h.py:27] end prefill_attn cost 1.796 ms
DEBUG 05-06 11:48:58.286121.286121 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 11:48:58.287089.287089 cuda_h.py:27] end prefill_gate cost 0.421 ms
experts_cpu_alloc {'expert_ids': [75, 67, 99, 27, 111, 115, 95, 71, 63, 23, 35, 43, 123, 3, 68, 12, 120, 40, 116, 0, 96, 64, 24, 44, 72, 56, 100, 88, 25, 45, 21, 5, 113, 121, 61, 85, 17, 57, 105, 69, 77, 49, 66, 26, 50, 6, 42, 114, 82, 58, 46, 126, 70], 'token_total': 620, 'token_per_expert': {75: 1, 67: 2, 99: 4, 27: 6, 111: 6, 115: 7, 95: 13, 71: 15, 63: 17, 23: 18, 35: 19, 43: 20, 123: 21, 3: 22, 68: 1, 12: 2, 120: 4, 40: 6, 116: 8, 0: 9, 96: 9, 64: 14, 24: 15, 44: 16, 72: 16, 56: 17, 100: 17, 88: 18, 25: 2, 45: 3, 21: 5, 5: 7, 113: 8, 121: 10, 61: 12, 85: 17, 17: 18, 57: 18, 105: 19, 69: 20, 77: 20, 49: 24, 66: 2, 26: 4, 50: 4, 6: 5, 42: 9, 114: 9, 82: 13, 58: 14, 46: 17, 126: 18, 70: 19}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 15, 19, 31, 51, 55, 59, 83, 91, 103, 107, 119, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 987, 'token_per_expert': {7: 65, 11: 208, 15: 95, 19: 118, 31: 23, 51: 35, 55: 68, 59: 110, 83: 24, 91: 42, 103: 23, 107: 25, 119: 23, 127: 128}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 20, 28, 36, 48, 52, 60, 76, 80, 84, 104, 108, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 756, 'token_per_expert': {4: 25, 8: 26, 20: 52, 28: 19, 36: 19, 48: 71, 52: 19, 60: 47, 76: 63, 80: 64, 84: 60, 104: 46, 108: 216, 124: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 29, 33, 37, 41, 53, 65, 81, 97, 109, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 974, 'token_per_expert': {1: 103, 9: 111, 13: 106, 29: 84, 33: 27, 37: 76, 41: 142, 53: 40, 65: 41, 81: 73, 97: 40, 109: 44, 125: 87}}
experts_gpu_alloc_device_3 {'expert_ids': [14, 18, 34, 54, 62, 78, 90, 98, 102, 106, 110, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 759, 'token_per_expert': {14: 40, 18: 63, 34: 33, 54: 117, 62: 141, 78: 33, 90: 76, 98: 22, 102: 88, 106: 37, 110: 31, 118: 54, 122: 24}}
INFO 05-06 11:48:58.288485.288485 lmp.py:1836] [layer_moe_fused] layer=2 prefix: 0.420ms alloc: 0.402ms
INFO 05-06 11:48:58.288576.288576 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.936622619628906e-05 seconds
INFO 05-06 11:48:58.289258.289258 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008363723754882812s
INFO 05-06 11:48:58.290684.290684 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006020069122314453 seconds
DEBUG 05-06 11:48:58.290578.290578 cuda_h.py:27] end moe_cpu_prep_submit cost 0.798 ms
INFO 05-06 11:48:58.292463.292463 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014035701751708984s
DEBUG 05-06 11:48:58.292271.292271 cuda_h.py:27] end moe_wait_copy_tasks cost 1.588 ms
DEBUG 05-06 11:48:58.299630.299630 cuda_h.py:27] end moe_vllm_forward cost 6.682 ms
DEBUG 05-06 11:48:58.318376.318376 cuda_h.py:27] end moe_cpu_merge cost 19.034 ms
DEBUG 05-06 11:48:58.319538.319538 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:48:58.319169.319169 lmp.py:1950] [layer_moe_fused] vllm triton time: 26.368ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.319398.319398 cuda_h.py:27] end *layer_moe_fused cost 31.661 ms
DEBUG 05-06 11:48:58.320557.320557 cuda_h.py:27] end prefill_merge_scale cost 0.515 ms
DEBUG 05-06 11:48:58.320487.320487 cuda_h.py:27] end prefill_layer cost 36.307 ms
DEBUG 05-06 11:48:58.320592.320592 lmp.py:1391] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 11:48:58.320415.320415 lmp.py:1347] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 11:48:58.320546.320546 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:48:58.322181.322181 cuda_h.py:27] end prefill_attn cost 2.087 ms
DEBUG 05-06 11:48:58.323383.323383 cuda_h.py:27] end prefill_ffn_prep cost 0.414 ms
DEBUG 05-06 11:48:58.324261.324261 cuda_h.py:27] end prefill_gate cost 0.537 ms
experts_cpu_alloc {'expert_ids': [23, 103, 55, 35, 91, 127, 27, 43, 63, 31, 67, 123, 111, 20, 72, 80, 36, 60, 16, 32, 100, 116, 40, 56, 24, 8, 48, 64, 65, 29, 41, 89, 117, 57, 13, 33, 61, 77, 101, 46, 94, 18, 42, 82, 30, 98, 86, 110, 26, 114, 54, 74, 70, 58], 'token_total': 507, 'token_per_expert': {23: 1, 103: 1, 55: 2, 35: 3, 91: 3, 127: 7, 27: 8, 43: 9, 63: 9, 31: 12, 67: 15, 123: 16, 111: 17, 20: 1, 72: 2, 80: 2, 36: 3, 60: 4, 16: 7, 32: 7, 100: 7, 116: 8, 40: 9, 56: 9, 24: 10, 8: 11, 48: 11, 64: 20, 65: 2, 29: 3, 41: 3, 89: 4, 117: 4, 57: 5, 13: 9, 33: 11, 61: 13, 77: 15, 101: 15, 46: 1, 94: 1, 18: 3, 42: 4, 82: 4, 30: 6, 98: 6, 86: 10, 110: 16, 26: 20, 114: 24, 54: 25, 74: 28, 70: 30, 58: 31}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 39, 51, 59, 71, 75, 83, 95, 107, 119], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 743, 'token_per_expert': {3: 153, 7: 128, 11: 37, 15: 30, 19: 20, 39: 19, 51: 32, 59: 21, 71: 65, 75: 79, 83: 41, 95: 55, 107: 34, 119: 29}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 44, 52, 68, 76, 84, 88, 92, 96, 104, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 883, 'token_per_expert': {0: 158, 4: 178, 28: 79, 44: 26, 52: 43, 68: 49, 76: 34, 84: 58, 88: 50, 92: 56, 96: 41, 104: 36, 108: 35, 120: 40}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 25, 53, 69, 73, 85, 93, 97, 109, 121], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 849, 'token_per_expert': {1: 135, 5: 170, 9: 59, 17: 38, 25: 41, 53: 42, 69: 34, 73: 29, 85: 102, 93: 60, 97: 52, 109: 21, 121: 66}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 34, 50, 62, 66, 78, 102, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 1114, 'token_per_expert': {2: 149, 6: 142, 10: 49, 14: 61, 22: 86, 34: 59, 50: 131, 62: 83, 66: 87, 78: 86, 102: 89, 118: 38, 122: 54}}
INFO 05-06 11:48:58.325049.325049 lmp.py:1836] [layer_moe_fused] layer=3 prefix: 0.444ms alloc: 0.407ms
INFO 05-06 11:48:58.325954.325954 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.079673767089844e-05 seconds
INFO 05-06 11:48:58.326748.326748 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008518695831298828s
INFO 05-06 11:48:58.327187.327187 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005748271942138672 seconds
DEBUG 05-06 11:48:58.327186.327186 cuda_h.py:27] end moe_cpu_prep_submit cost 0.725 ms
INFO 05-06 11:48:58.329894.329894 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001636505126953125s
DEBUG 05-06 11:48:58.329699.329699 cuda_h.py:27] end moe_wait_copy_tasks cost 1.751 ms
DEBUG 05-06 11:48:58.336502.336502 cuda_h.py:27] end moe_vllm_forward cost 6.743 ms
DEBUG 05-06 11:48:58.337030.337030 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:48:58.337696.337696 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:48:58.337751.337751 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.645ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.337998.337998 cuda_h.py:27] end *layer_moe_fused cost 12.793 ms
DEBUG 05-06 11:48:58.342232.342232 cuda_h.py:27] end prefill_merge_scale cost 5.135 ms
DEBUG 05-06 11:48:58.343785.343785 cuda_h.py:27] end prefill_layer cost 22.708 ms
DEBUG 05-06 11:48:58.343300.343300 lmp.py:1391] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 11:48:58.343017.343017 lmp.py:1347] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 11:48:58.343751.343751 cuda_h.py:27] end prefill_ln cost 0.216 ms
DEBUG 05-06 11:48:58.345713.345713 cuda_h.py:27] end prefill_attn cost 1.766 ms
DEBUG 05-06 11:48:58.345132.345132 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 11:48:58.346358.346358 cuda_h.py:27] end prefill_gate cost 0.414 ms
experts_cpu_alloc {'expert_ids': [35, 79, 103, 31, 107, 15, 91, 75, 47, 87, 123, 71, 19, 67, 72, 12, 120, 44, 56, 40, 80, 84, 36, 52, 88, 108, 64, 41, 69, 121, 21, 101, 37, 109, 45, 77, 81, 25, 73, 97, 57, 46, 70, 50, 114, 66, 122, 18, 126, 90, 34, 38, 78, 118], 'token_total': 498, 'token_per_expert': {35: 1, 79: 1, 103: 3, 31: 4, 107: 13, 15: 14, 91: 15, 75: 16, 47: 21, 87: 23, 123: 23, 71: 25, 19: 32, 67: 37, 72: 1, 12: 2, 120: 5, 44: 6, 56: 6, 40: 8, 80: 8, 84: 10, 36: 11, 52: 11, 88: 13, 108: 14, 64: 15, 41: 2, 69: 3, 121: 3, 21: 4, 101: 4, 37: 5, 109: 5, 45: 6, 77: 7, 81: 7, 25: 9, 73: 9, 97: 9, 57: 13, 46: 1, 70: 1, 50: 2, 114: 2, 66: 3, 122: 3, 18: 4, 126: 5, 90: 7, 34: 9, 38: 9, 78: 13, 118: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 39, 43, 51, 55, 59, 63, 83, 111, 115, 119], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1330, 'token_per_expert': {3: 158, 7: 132, 23: 87, 27: 57, 39: 39, 43: 104, 51: 56, 55: 44, 59: 128, 63: 203, 83: 61, 111: 79, 115: 77, 119: 105}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 24, 28, 32, 60, 76, 92, 96, 104, 116, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 751, 'token_per_expert': {0: 128, 4: 144, 8: 131, 20: 21, 24: 72, 28: 28, 32: 43, 60: 25, 76: 37, 92: 23, 96: 26, 104: 28, 116: 15, 124: 30}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 29, 49, 53, 61, 85, 89, 93, 105, 113, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 761, 'token_per_expert': {1: 187, 5: 153, 17: 14, 29: 44, 49: 28, 53: 42, 61: 21, 85: 32, 89: 70, 93: 35, 105: 17, 113: 70, 117: 14, 125: 34}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 30, 54, 62, 74, 82, 86, 94, 98, 106], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 756, 'token_per_expert': {2: 128, 6: 132, 22: 59, 26: 60, 30: 23, 54: 54, 62: 26, 74: 69, 82: 49, 86: 20, 94: 26, 98: 21, 106: 89}}
INFO 05-06 11:48:58.348335.348335 lmp.py:1836] [layer_moe_fused] layer=4 prefix: 0.443ms alloc: 0.402ms
INFO 05-06 11:48:58.348764.348764 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.079673767089844e-05 seconds
INFO 05-06 11:48:58.349168.349168 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007996559143066406s
INFO 05-06 11:48:58.349327.349327 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005447864532470703 seconds
DEBUG 05-06 11:48:58.350114.350114 cuda_h.py:27] end moe_cpu_prep_submit cost 1.176 ms
INFO 05-06 11:48:58.352288.352288 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016269683837890625s
DEBUG 05-06 11:48:58.352384.352384 cuda_h.py:27] end moe_wait_copy_tasks cost 1.744 ms
DEBUG 05-06 11:48:58.359914.359914 cuda_h.py:27] end moe_vllm_forward cost 6.870 ms
DEBUG 05-06 11:48:58.360104.360104 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 11:48:58.360085.360085 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:48:58.360902.360902 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.707ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.360048.360048 cuda_h.py:27] end *layer_moe_fused cost 13.454 ms
DEBUG 05-06 11:48:58.367820.367820 cuda_h.py:27] end prefill_merge_scale cost 6.319 ms
DEBUG 05-06 11:48:58.367611.367611 cuda_h.py:27] end prefill_layer cost 23.892 ms
DEBUG 05-06 11:48:58.367527.367527 lmp.py:1391] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 11:48:58.367303.367303 lmp.py:1347] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 11:48:58.367025.367025 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 11:48:58.373439.373439 cuda_h.py:27] end prefill_attn cost 5.953 ms
DEBUG 05-06 11:48:58.374836.374836 cuda_h.py:27] end prefill_ffn_prep cost 0.473 ms
DEBUG 05-06 11:48:58.375135.375135 cuda_h.py:27] end prefill_gate cost 0.403 ms
experts_cpu_alloc {'expert_ids': [51, 115, 19, 55, 83, 27, 119, 63, 79, 107, 31, 8, 32, 92, 124, 68, 52, 56, 100, 84, 44, 96, 80, 60, 116, 120, 17, 21, 81, 105, 57, 37, 77, 125, 26, 50, 82, 110, 30, 58, 86, 34, 122, 54, 62, 114, 10, 98, 102, 106], 'token_total': 272, 'token_per_expert': {51: 2, 115: 3, 19: 4, 55: 5, 83: 5, 27: 6, 119: 6, 63: 8, 79: 8, 107: 8, 31: 9, 8: 1, 32: 1, 92: 1, 124: 1, 68: 3, 52: 4, 56: 4, 100: 6, 84: 8, 44: 9, 96: 14, 80: 17, 60: 18, 116: 19, 120: 20, 17: 1, 21: 1, 81: 1, 105: 2, 57: 3, 37: 4, 77: 6, 125: 6, 26: 1, 50: 1, 82: 1, 110: 1, 30: 2, 58: 2, 86: 2, 34: 3, 122: 3, 54: 5, 62: 5, 114: 5, 10: 6, 98: 6, 102: 7, 106: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 43, 67, 71, 75, 87, 99, 111, 123, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 997, 'token_per_expert': {3: 256, 7: 259, 23: 21, 39: 88, 43: 14, 67: 14, 71: 135, 75: 11, 87: 26, 99: 28, 111: 47, 123: 39, 127: 59}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 28, 36, 64, 72, 76, 88, 104, 112], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 1021, 'token_per_expert': {0: 262, 4: 284, 16: 56, 20: 73, 24: 32, 28: 30, 36: 38, 64: 50, 72: 43, 76: 24, 88: 29, 104: 26, 112: 74}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 29, 33, 49, 61, 73, 93, 101, 113, 117], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 21, 'token_total': 1032, 'token_per_expert': {1: 256, 5: 267, 9: 26, 13: 32, 29: 13, 33: 58, 49: 82, 61: 39, 73: 32, 93: 24, 101: 166, 113: 8, 117: 29}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 42, 46, 70, 74, 94, 118, 126], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 28, 'token_total': 774, 'token_per_expert': {2: 302, 6: 260, 14: 11, 18: 11, 22: 52, 42: 33, 46: 18, 70: 27, 74: 21, 94: 19, 118: 9, 126: 11}}
INFO 05-06 11:48:58.376980.376980 lmp.py:1836] [layer_moe_fused] layer=5 prefix: 0.422ms alloc: 0.390ms
INFO 05-06 11:48:58.376594.376594 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.7697296142578125e-05 seconds
INFO 05-06 11:48:58.377814.377814 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008041858673095703s
INFO 05-06 11:48:58.378432.378432 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005667209625244141 seconds
DEBUG 05-06 11:48:58.378793.378793 cuda_h.py:27] end moe_cpu_prep_submit cost 0.864 ms
INFO 05-06 11:48:58.380343.380343 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017321109771728516s
DEBUG 05-06 11:48:58.380393.380393 cuda_h.py:27] end moe_wait_copy_tasks cost 1.849 ms
DEBUG 05-06 11:48:58.387202.387202 cuda_h.py:27] end moe_vllm_forward cost 6.534 ms
DEBUG 05-06 11:48:58.387121.387121 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:48:58.387645.387645 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:48:58.388131.388131 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.343ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.388305.388305 cuda_h.py:27] end *layer_moe_fused cost 12.503 ms
DEBUG 05-06 11:48:58.393096.393096 cuda_h.py:27] end prefill_merge_scale cost 5.159 ms
DEBUG 05-06 11:48:58.393218.393218 cuda_h.py:27] end prefill_layer cost 26.129 ms
DEBUG 05-06 11:48:58.393417.393417 lmp.py:1391] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 11:48:58.393611.393611 lmp.py:1347] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 11:48:58.394154.394154 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:48:58.396699.396699 cuda_h.py:27] end prefill_attn cost 1.775 ms
DEBUG 05-06 11:48:58.396310.396310 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 11:48:58.397841.397841 cuda_h.py:27] end prefill_gate cost 0.417 ms
experts_cpu_alloc {'expert_ids': [15, 67, 83, 111, 11, 19, 91, 103, 43, 127, 95, 27, 60, 72, 84, 100, 40, 52, 112, 120, 124, 16, 76, 20, 28, 81, 101, 109, 33, 37, 97, 41, 73, 105, 57, 85, 125, 118, 18, 38, 82, 110, 114, 22, 30, 58, 74, 126, 10, 42, 14, 122, 70, 50], 'token_total': 234, 'token_per_expert': {15: 1, 67: 1, 83: 1, 111: 1, 11: 2, 19: 4, 91: 4, 103: 5, 43: 7, 127: 7, 95: 9, 27: 13, 60: 1, 72: 1, 84: 1, 100: 1, 40: 2, 52: 2, 112: 2, 120: 2, 124: 2, 16: 3, 76: 3, 20: 5, 28: 6, 81: 1, 101: 1, 109: 1, 33: 2, 37: 2, 97: 3, 41: 5, 73: 6, 105: 6, 57: 7, 85: 8, 125: 8, 118: 1, 18: 2, 38: 2, 82: 2, 110: 2, 114: 2, 22: 3, 30: 4, 58: 6, 74: 6, 126: 6, 10: 7, 42: 7, 14: 9, 122: 9, 70: 12, 50: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 35, 51, 71, 75, 79, 87, 99, 107, 115, 119, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 942, 'token_per_expert': {3: 263, 7: 256, 23: 37, 35: 63, 51: 13, 71: 18, 75: 19, 79: 23, 87: 47, 99: 102, 107: 15, 115: 51, 119: 22, 123: 13}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 32, 36, 44, 56, 64, 68, 80, 96, 104, 108, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 947, 'token_per_expert': {0: 260, 4: 256, 24: 26, 32: 18, 36: 13, 44: 12, 56: 14, 64: 80, 68: 150, 80: 7, 96: 21, 104: 26, 108: 55, 116: 9}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 53, 65, 69, 77, 89, 93, 113, 117, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 998, 'token_per_expert': {1: 271, 5: 267, 9: 21, 13: 36, 25: 100, 53: 60, 65: 58, 69: 13, 77: 14, 89: 10, 93: 87, 113: 16, 117: 23, 121: 22}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 34, 46, 62, 78, 86, 90, 94, 98, 102, 106], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 30, 'token_total': 975, 'token_per_expert': {2: 265, 6: 265, 26: 19, 34: 55, 46: 20, 62: 25, 78: 22, 86: 61, 90: 45, 94: 50, 98: 32, 102: 63, 106: 53}}
INFO 05-06 11:48:58.398984.398984 lmp.py:1836] [layer_moe_fused] layer=6 prefix: 0.420ms alloc: 0.406ms
INFO 05-06 11:48:58.398121.398121 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.841255187988281e-05 seconds
INFO 05-06 11:48:58.399343.399343 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007786750793457031s
INFO 05-06 11:48:58.400107.400107 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006041526794433594 seconds
DEBUG 05-06 11:48:58.400160.400160 cuda_h.py:27] end moe_cpu_prep_submit cost 0.813 ms
INFO 05-06 11:48:58.402681.402681 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0018076896667480469s
DEBUG 05-06 11:48:58.402870.402870 cuda_h.py:27] end moe_wait_copy_tasks cost 1.927 ms
DEBUG 05-06 11:48:58.409243.409243 cuda_h.py:27] end moe_vllm_forward cost 6.548 ms
DEBUG 05-06 11:48:58.409578.409578 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 11:48:58.409361.409361 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:48:58.409893.409893 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.378ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.410531.410531 cuda_h.py:27] end *layer_moe_fused cost 12.436 ms
DEBUG 05-06 11:48:58.418758.418758 cuda_h.py:27] end prefill_merge_scale cost 7.904 ms
DEBUG 05-06 11:48:58.418595.418595 cuda_h.py:27] end prefill_layer cost 24.460 ms
DEBUG 05-06 11:48:58.418164.418164 lmp.py:1391] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 11:48:58.418596.418596 lmp.py:1347] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 11:48:58.418508.418508 cuda_h.py:27] end prefill_ln cost 0.209 ms
DEBUG 05-06 11:48:58.421172.421172 cuda_h.py:27] end prefill_attn cost 2.352 ms
DEBUG 05-06 11:48:58.421474.421474 cuda_h.py:27] end prefill_ffn_prep cost 0.419 ms
DEBUG 05-06 11:48:58.423040.423040 cuda_h.py:27] end prefill_gate cost 0.540 ms
experts_cpu_alloc {'expert_ids': [11, 15, 35, 67, 75, 107, 111, 119, 127, 55, 63, 23, 31, 123, 87, 36, 124, 16, 116, 88, 32, 68, 64, 72, 80, 48, 8, 84, 37, 109, 73, 25, 45, 117, 17, 21, 41, 101, 33, 125, 9, 61, 38, 50, 62, 26, 30, 66, 102, 78, 82, 118, 22, 98, 18, 54], 'token_total': 263, 'token_per_expert': {11: 1, 15: 1, 35: 1, 67: 1, 75: 1, 107: 1, 111: 1, 119: 1, 127: 1, 55: 3, 63: 4, 23: 5, 31: 6, 123: 6, 87: 7, 36: 2, 124: 2, 16: 3, 116: 5, 88: 6, 32: 7, 68: 7, 64: 8, 72: 8, 80: 8, 48: 17, 8: 18, 84: 18, 37: 1, 109: 1, 73: 2, 25: 3, 45: 3, 117: 3, 17: 4, 21: 4, 41: 4, 101: 4, 33: 5, 125: 5, 9: 7, 61: 7, 38: 1, 50: 1, 62: 1, 26: 2, 30: 2, 66: 2, 102: 2, 78: 4, 82: 6, 118: 6, 22: 7, 98: 7, 18: 10, 54: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 43, 47, 51, 59, 71, 79, 83, 91, 95, 99, 103, 115], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 880, 'token_per_expert': {3: 321, 7: 330, 19: 8, 43: 10, 47: 12, 51: 23, 59: 14, 71: 11, 79: 20, 83: 13, 91: 70, 95: 7, 99: 11, 103: 7, 115: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 28, 44, 52, 56, 60, 96, 104, 108, 112, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1015, 'token_per_expert': {0: 328, 4: 360, 12: 39, 20: 34, 28: 20, 44: 20, 52: 31, 56: 20, 60: 25, 96: 24, 104: 21, 108: 47, 112: 20, 120: 26}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 53, 57, 65, 69, 77, 85, 97, 105, 113, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1032, 'token_per_expert': {1: 320, 5: 339, 13: 12, 29: 41, 53: 16, 57: 14, 65: 21, 69: 37, 77: 7, 85: 29, 97: 102, 105: 12, 113: 24, 121: 58}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 34, 42, 70, 86, 90, 106, 110, 114, 122, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 906, 'token_per_expert': {2: 320, 6: 325, 10: 25, 14: 17, 34: 29, 42: 12, 70: 31, 86: 21, 90: 27, 106: 37, 110: 21, 114: 20, 122: 10, 126: 11}}
INFO 05-06 11:48:58.424107.424107 lmp.py:1836] [layer_moe_fused] layer=7 prefix: 0.468ms alloc: 0.416ms
INFO 05-06 11:48:58.424966.424966 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.318092346191406e-05 seconds
INFO 05-06 11:48:58.425389.425389 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007655620574951172s
INFO 05-06 11:48:58.425852.425852 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005221366882324219 seconds
DEBUG 05-06 11:48:58.426859.426859 cuda_h.py:27] end moe_cpu_prep_submit cost 0.848 ms
INFO 05-06 11:48:58.429954.429954 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0022742748260498047s
DEBUG 05-06 11:48:58.429567.429567 cuda_h.py:27] end moe_wait_copy_tasks cost 2.390 ms
DEBUG 05-06 11:48:58.436509.436509 cuda_h.py:27] end moe_vllm_forward cost 6.581 ms
DEBUG 05-06 11:48:58.436845.436845 cuda_h.py:27] end moe_cpu_merge cost 0.060 ms
DEBUG 05-06 11:48:58.436592.436592 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:48:58.436124.436124 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.321ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.436411.436411 cuda_h.py:27] end *layer_moe_fused cost 13.508 ms
DEBUG 05-06 11:48:58.442959.442959 cuda_h.py:27] end prefill_merge_scale cost 5.223 ms
DEBUG 05-06 11:48:58.442604.442604 cuda_h.py:27] end prefill_layer cost 23.775 ms
DEBUG 05-06 11:48:58.442682.442682 lmp.py:1391] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 11:48:58.442637.442637 lmp.py:1347] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 11:48:58.442353.442353 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:48:58.444416.444416 cuda_h.py:27] end prefill_attn cost 1.806 ms
DEBUG 05-06 11:48:58.445557.445557 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 11:48:58.446512.446512 cuda_h.py:27] end prefill_gate cost 0.417 ms
experts_cpu_alloc {'expert_ids': [43, 99, 11, 47, 119, 91, 39, 31, 111, 48, 64, 72, 84, 100, 104, 8, 92, 124, 24, 96, 116, 20, 52, 25, 101, 13, 37, 49, 117, 9, 21, 113, 17, 53, 45, 57, 77, 61, 85, 26, 82, 106, 62, 90, 118, 10, 34, 86, 126, 14, 42, 22, 66], 'token_total': 221, 'token_per_expert': {43: 1, 99: 1, 11: 2, 47: 2, 119: 2, 91: 3, 39: 4, 31: 10, 111: 10, 48: 1, 64: 1, 72: 1, 84: 1, 100: 1, 104: 1, 8: 2, 92: 2, 124: 2, 24: 3, 96: 3, 116: 4, 20: 5, 52: 5, 25: 1, 101: 1, 13: 2, 37: 4, 49: 4, 117: 4, 9: 5, 21: 5, 113: 5, 17: 6, 53: 7, 45: 8, 57: 8, 77: 8, 61: 9, 85: 9, 26: 1, 82: 1, 106: 1, 62: 2, 90: 2, 118: 2, 10: 3, 34: 3, 86: 5, 126: 7, 14: 9, 42: 9, 22: 11, 66: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 51, 55, 63, 71, 75, 87, 103, 123, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 23, 'token_total': 993, 'token_per_expert': {3: 335, 7: 321, 15: 15, 19: 26, 27: 19, 51: 65, 55: 27, 63: 20, 71: 24, 75: 34, 87: 33, 103: 45, 123: 18, 127: 11}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 28, 32, 36, 44, 56, 68, 76, 80, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 877, 'token_per_expert': {0: 320, 4: 325, 12: 24, 16: 11, 28: 33, 32: 16, 36: 23, 44: 6, 56: 27, 68: 6, 76: 8, 80: 27, 108: 10, 120: 41}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 41, 65, 69, 73, 81, 89, 93, 105, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 966, 'token_per_expert': {1: 327, 5: 331, 29: 26, 41: 21, 65: 30, 69: 22, 73: 54, 81: 21, 89: 10, 93: 11, 105: 33, 121: 45, 125: 35}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 50, 54, 58, 70, 98, 102, 110, 114, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 1039, 'token_per_expert': {2: 345, 6: 330, 38: 18, 46: 29, 50: 25, 54: 75, 58: 67, 70: 30, 98: 12, 102: 19, 110: 37, 114: 28, 122: 24}}
INFO 05-06 11:48:58.447170.447170 lmp.py:1836] [layer_moe_fused] layer=8 prefix: 0.420ms alloc: 0.399ms
INFO 05-06 11:48:58.447506.447506 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.175041198730469e-05 seconds
INFO 05-06 11:48:58.448114.448114 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007767677307128906s
INFO 05-06 11:48:58.449704.449704 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005393028259277344 seconds
DEBUG 05-06 11:48:58.449946.449946 cuda_h.py:27] end moe_cpu_prep_submit cost 0.854 ms
INFO 05-06 11:48:58.451760.451760 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001621246337890625s
DEBUG 05-06 11:48:58.451412.451412 cuda_h.py:27] end moe_wait_copy_tasks cost 1.726 ms
DEBUG 05-06 11:48:58.458100.458100 cuda_h.py:27] end moe_vllm_forward cost 6.466 ms
DEBUG 05-06 11:48:58.458382.458382 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:48:58.458855.458855 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:48:58.458818.458818 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.344ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.459900.459900 cuda_h.py:27] end *layer_moe_fused cost 12.526 ms
DEBUG 05-06 11:48:58.464118.464118 cuda_h.py:27] end prefill_merge_scale cost 5.473 ms
DEBUG 05-06 11:48:58.464956.464956 cuda_h.py:27] end prefill_layer cost 22.179 ms
DEBUG 05-06 11:48:58.464624.464624 lmp.py:1391] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 11:48:58.464579.464579 lmp.py:1347] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 11:48:58.465850.465850 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:48:58.467586.467586 cuda_h.py:27] end prefill_attn cost 1.739 ms
DEBUG 05-06 11:48:58.467336.467336 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 11:48:58.468973.468973 cuda_h.py:27] end prefill_gate cost 0.410 ms
experts_cpu_alloc {'expert_ids': [11, 119, 15, 67, 79, 99, 83, 39, 55, 115, 20, 84, 96, 100, 108, 120, 44, 80, 112, 124, 8, 28, 52, 40, 104, 68, 17, 73, 77, 97, 29, 45, 33, 41, 113, 105, 9, 37, 10, 14, 50, 110, 34, 58, 90, 38, 82, 26, 42, 54, 86, 98], 'token_total': 152, 'token_per_expert': {11: 2, 119: 2, 15: 3, 67: 3, 79: 3, 99: 4, 83: 5, 39: 6, 55: 6, 115: 8, 20: 1, 84: 1, 96: 1, 100: 1, 108: 1, 120: 1, 44: 2, 80: 2, 112: 2, 124: 2, 8: 3, 28: 3, 52: 3, 40: 5, 104: 5, 68: 6, 17: 1, 73: 1, 77: 1, 97: 1, 29: 2, 45: 2, 33: 3, 41: 3, 113: 3, 105: 4, 9: 5, 37: 5, 10: 1, 14: 1, 50: 1, 110: 1, 34: 2, 58: 2, 90: 2, 38: 3, 82: 3, 26: 4, 42: 5, 54: 5, 86: 5, 98: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 43, 51, 71, 75, 95, 103, 111, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 23, 'token_total': 1018, 'token_per_expert': {3: 386, 7: 386, 19: 16, 23: 9, 27: 9, 43: 46, 51: 14, 71: 12, 75: 21, 95: 60, 103: 40, 111: 11, 127: 8}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 24, 32, 36, 48, 56, 72, 76, 88, 92], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 971, 'token_per_expert': {0: 386, 4: 398, 12: 42, 16: 22, 24: 26, 32: 11, 36: 10, 48: 10, 56: 13, 72: 14, 76: 18, 88: 7, 92: 14}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 57, 61, 69, 81, 89, 93, 101, 117, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 933, 'token_per_expert': {1: 402, 5: 392, 13: 17, 21: 10, 57: 11, 61: 14, 69: 24, 81: 12, 89: 7, 93: 13, 101: 15, 117: 6, 125: 10}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 30, 46, 62, 66, 70, 74, 102, 106, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 1022, 'token_per_expert': {2: 384, 6: 387, 18: 8, 22: 13, 30: 12, 46: 43, 62: 13, 66: 6, 70: 34, 74: 25, 102: 9, 106: 72, 122: 16}}
INFO 05-06 11:48:58.469267.469267 lmp.py:1836] [layer_moe_fused] layer=9 prefix: 0.416ms alloc: 0.382ms
INFO 05-06 11:48:58.469265.469265 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.079673767089844e-05 seconds
INFO 05-06 11:48:58.470817.470817 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007407665252685547s
INFO 05-06 11:48:58.471267.471267 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005140304565429688 seconds
DEBUG 05-06 11:48:58.471889.471889 cuda_h.py:27] end moe_cpu_prep_submit cost 0.901 ms
INFO 05-06 11:48:58.473615.473615 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015952587127685547s
DEBUG 05-06 11:48:58.473406.473406 cuda_h.py:27] end moe_wait_copy_tasks cost 1.697 ms
DEBUG 05-06 11:48:58.480269.480269 cuda_h.py:27] end moe_vllm_forward cost 6.382 ms
DEBUG 05-06 11:48:58.480499.480499 cuda_h.py:27] end moe_cpu_merge cost 0.054 ms
DEBUG 05-06 11:48:58.480625.480625 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:48:58.480488.480488 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.141ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.481642.481642 cuda_h.py:27] end *layer_moe_fused cost 12.211 ms
DEBUG 05-06 11:48:58.486577.486577 cuda_h.py:27] end prefill_merge_scale cost 5.488 ms
DEBUG 05-06 11:48:58.486368.486368 cuda_h.py:27] end prefill_layer cost 21.788 ms
DEBUG 05-06 11:48:58.486395.486395 lmp.py:1391] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 11:48:58.486588.486588 lmp.py:1347] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 11:48:58.487183.487183 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:48:58.489946.489946 cuda_h.py:27] end prefill_attn cost 1.760 ms
DEBUG 05-06 11:48:58.489995.489995 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 11:48:58.490075.490075 cuda_h.py:27] end prefill_gate cost 0.407 ms
experts_cpu_alloc {'expert_ids': [23, 59, 87, 111, 63, 91, 19, 99, 103, 67, 79, 31, 64, 124, 72, 84, 120, 44, 104, 112, 56, 88, 17, 29, 73, 89, 97, 101, 25, 109, 33, 93, 113, 117, 37, 49, 78, 90, 98, 102, 70, 18, 50, 58, 42, 62, 34], 'token_total': 149, 'token_per_expert': {23: 1, 59: 1, 87: 1, 111: 1, 63: 2, 91: 2, 19: 3, 99: 3, 103: 3, 67: 6, 79: 6, 31: 7, 64: 1, 124: 1, 72: 2, 84: 2, 120: 2, 44: 3, 104: 3, 112: 4, 56: 5, 88: 9, 17: 1, 29: 1, 73: 1, 89: 1, 97: 1, 101: 1, 25: 2, 109: 2, 33: 4, 93: 4, 113: 4, 117: 5, 37: 6, 49: 6, 78: 1, 90: 1, 98: 1, 102: 1, 70: 2, 18: 3, 50: 3, 58: 6, 42: 7, 62: 7, 34: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 39, 43, 47, 71, 75, 83, 115, 119, 127], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 24, 'token_total': 929, 'token_per_expert': {3: 385, 7: 387, 11: 11, 39: 42, 43: 10, 47: 9, 71: 14, 75: 12, 83: 14, 115: 22, 119: 8, 127: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 60, 68, 76, 80, 92, 100, 108], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 22, 'token_total': 1028, 'token_per_expert': {0: 390, 4: 385, 8: 36, 16: 18, 20: 11, 60: 35, 68: 15, 76: 32, 80: 55, 92: 9, 100: 11, 108: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 41, 57, 69, 81, 85, 105, 121, 125], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 26, 'token_total': 972, 'token_per_expert': {1: 435, 5: 385, 13: 7, 21: 8, 41: 21, 57: 13, 69: 11, 81: 44, 85: 10, 105: 10, 121: 11, 125: 17}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 46, 54, 74, 82, 86, 94, 106, 126], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 23, 'token_total': 1018, 'token_per_expert': {2: 384, 6: 387, 10: 24, 14: 44, 46: 20, 54: 10, 74: 36, 82: 28, 86: 34, 94: 13, 106: 14, 126: 24}}
INFO 05-06 11:48:58.491547.491547 lmp.py:1836] [layer_moe_fused] layer=10 prefix: 0.405ms alloc: 0.360ms
INFO 05-06 11:48:58.491922.491922 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.5789947509765625e-05 seconds
INFO 05-06 11:48:58.492700.492700 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007331371307373047s
INFO 05-06 11:48:58.493303.493303 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005202293395996094 seconds
DEBUG 05-06 11:48:58.493758.493758 cuda_h.py:27] end moe_cpu_prep_submit cost 0.852 ms
INFO 05-06 11:48:58.495134.495134 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001649618148803711s
DEBUG 05-06 11:48:58.495925.495925 cuda_h.py:27] end moe_wait_copy_tasks cost 1.760 ms
DEBUG 05-06 11:48:58.502813.502813 cuda_h.py:27] end moe_vllm_forward cost 6.333 ms
DEBUG 05-06 11:48:58.502520.502520 cuda_h.py:27] end moe_cpu_merge cost 0.055 ms
DEBUG 05-06 11:48:58.502540.502540 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:48:58.502311.502311 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.119ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.502988.502988 cuda_h.py:27] end *layer_moe_fused cost 12.001 ms
DEBUG 05-06 11:48:58.509395.509395 cuda_h.py:27] end prefill_merge_scale cost 6.736 ms
DEBUG 05-06 11:48:58.509279.509279 cuda_h.py:27] end prefill_layer cost 22.804 ms
DEBUG 05-06 11:48:58.509790.509790 lmp.py:1391] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 11:48:58.509983.509983 lmp.py:1347] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 11:48:58.510174.510174 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 11:48:58.512524.512524 cuda_h.py:27] end prefill_attn cost 2.087 ms
DEBUG 05-06 11:48:58.512812.512812 cuda_h.py:27] end prefill_ffn_prep cost 0.408 ms
DEBUG 05-06 11:48:58.514866.514866 cuda_h.py:27] end prefill_gate cost 0.398 ms
experts_cpu_alloc {'expert_ids': [47, 51, 63, 123, 27, 43, 99, 67, 71, 48, 64, 8, 96, 116, 40, 44, 88, 120, 80, 20, 52, 76, 36, 9, 13, 25, 125, 21, 33, 53, 85, 97, 61, 29, 37, 89, 42, 46, 58, 106, 22, 34, 50, 18, 54, 74], 'token_total': 161, 'token_per_expert': {47: 1, 51: 1, 63: 1, 123: 1, 27: 5, 43: 5, 99: 7, 67: 9, 71: 9, 48: 1, 64: 1, 8: 2, 96: 2, 116: 3, 40: 4, 44: 4, 88: 4, 120: 4, 80: 5, 20: 6, 52: 6, 76: 8, 36: 9, 9: 1, 13: 1, 25: 1, 125: 1, 21: 2, 33: 2, 53: 2, 85: 2, 97: 3, 61: 5, 29: 7, 37: 7, 89: 7, 42: 1, 46: 1, 58: 1, 106: 1, 22: 2, 34: 2, 50: 2, 18: 4, 54: 4, 74: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 31, 59, 79, 83, 87, 91, 111, 119], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 21, 'token_total': 1008, 'token_per_expert': {3: 386, 7: 403, 19: 19, 23: 17, 31: 19, 59: 12, 79: 30, 83: 19, 87: 52, 91: 16, 111: 20, 119: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 24, 28, 32, 56, 68, 100, 108, 112, 124], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 26, 'token_total': 978, 'token_per_expert': {0: 384, 4: 389, 16: 56, 24: 13, 28: 17, 32: 17, 56: 37, 68: 21, 100: 14, 108: 11, 112: 10, 124: 9}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 49, 57, 69, 77, 81, 93, 113, 117], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 24, 'token_total': 1021, 'token_per_expert': {1: 389, 5: 389, 17: 43, 49: 25, 57: 11, 69: 18, 77: 15, 81: 28, 93: 51, 113: 38, 117: 14}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 30, 38, 66, 70, 82, 98, 102, 126], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 21, 'token_total': 928, 'token_per_expert': {2: 398, 6: 395, 10: 19, 30: 16, 38: 10, 66: 8, 70: 22, 82: 6, 98: 13, 102: 30, 126: 11}}
INFO 05-06 11:48:58.515212.515212 lmp.py:1836] [layer_moe_fused] layer=11 prefix: 0.407ms alloc: 0.352ms
INFO 05-06 11:48:58.515442.515442 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.5789947509765625e-05 seconds
INFO 05-06 11:48:58.516947.516947 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007407665252685547s
INFO 05-06 11:48:58.516610.516610 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005285739898681641 seconds
DEBUG 05-06 11:48:58.516243.516243 cuda_h.py:27] end moe_cpu_prep_submit cost 0.853 ms
INFO 05-06 11:48:58.518846.518846 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001493215560913086s
DEBUG 05-06 11:48:58.518750.518750 cuda_h.py:27] end moe_wait_copy_tasks cost 1.610 ms
DEBUG 05-06 11:48:58.525407.525407 cuda_h.py:27] end moe_vllm_forward cost 6.400 ms
DEBUG 05-06 11:48:58.525776.525776 cuda_h.py:27] end moe_cpu_merge cost 0.054 ms
DEBUG 05-06 11:48:58.525900.525900 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:48:58.525764.525764 lmp.py:1950] [layer_moe_fused] vllm triton time: 7.096ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.526209.526209 cuda_h.py:27] end *layer_moe_fused cost 11.963 ms
DEBUG 05-06 11:48:58.531817.531817 cuda_h.py:27] end prefill_merge_scale cost 4.849 ms
DEBUG 05-06 11:48:58.531178.531178 cuda_h.py:27] end prefill_layer cost 21.245 ms
DEBUG 05-06 11:48:58.531203.531203 lmp.py:1391] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 11:48:58.531158.531158 lmp.py:1347] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 11:48:58.531932.531932 cuda_h.py:27] end prefill_ln cost 0.218 ms
DEBUG 05-06 11:48:58.533846.533846 cuda_h.py:27] end prefill_attn cost 1.728 ms
DEBUG 05-06 11:48:58.534411.534411 cuda_h.py:27] end prefill_ffn_prep cost 0.373 ms
DEBUG 05-06 11:48:58.535776.535776 cuda_h.py:27] end prefill_gate cost 0.405 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:58.535894.535894 lmp.py:1836] [layer_moe_fused] layer=12 prefix: 0.384ms alloc: 0.103ms
INFO 05-06 11:48:58.535354.535354 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 9.298324584960938e-06 seconds
INFO 05-06 11:48:58.536690.536690 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006735324859619141s
INFO 05-06 11:48:58.537133.537133 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00030040740966796875 seconds
DEBUG 05-06 11:48:58.537368.537368 cuda_h.py:27] end moe_cpu_prep_submit cost 0.403 ms
INFO 05-06 11:48:58.538777.538777 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011725425720214844s
DEBUG 05-06 11:48:58.538985.538985 cuda_h.py:27] end moe_wait_copy_tasks cost 1.261 ms
DEBUG 05-06 11:48:58.549995.549995 cuda_h.py:27] end moe_vllm_forward cost 10.624 ms
DEBUG 05-06 11:48:58.744090.744090 cuda_h.py:27] end moe_cpu_merge cost 194.935 ms
DEBUG 05-06 11:48:58.745902.745902 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 11:48:58.745886.745886 lmp.py:1950] [layer_moe_fused] vllm triton time: 206.231ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.745712.745712 cuda_h.py:27] end *layer_moe_fused cost 210.259 ms
DEBUG 05-06 11:48:58.746097.746097 cuda_h.py:27] end prefill_merge_scale cost 0.540 ms
DEBUG 05-06 11:48:58.746080.746080 cuda_h.py:27] end prefill_layer cost 214.828 ms
DEBUG 05-06 11:48:58.746767.746767 lmp.py:1391] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 11:48:58.746245.746245 lmp.py:1347] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 11:48:58.746569.746569 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:48:58.748648.748648 cuda_h.py:27] end prefill_attn cost 1.888 ms
DEBUG 05-06 11:48:58.749564.749564 cuda_h.py:27] end prefill_ffn_prep cost 0.381 ms
DEBUG 05-06 11:48:58.750354.750354 cuda_h.py:27] end prefill_gate cost 0.430 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:58.751107.751107 lmp.py:1836] [layer_moe_fused] layer=13 prefix: 0.351ms alloc: 0.115ms
INFO 05-06 11:48:58.751428.751428 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.1682510375976562e-05 seconds
INFO 05-06 11:48:58.752398.752398 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010673999786376953s
INFO 05-06 11:48:58.752549.752549 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002932548522949219 seconds
DEBUG 05-06 11:48:58.753936.753936 cuda_h.py:27] end moe_cpu_prep_submit cost 0.719 ms
INFO 05-06 11:48:58.755709.755709 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016210079193115234s
DEBUG 05-06 11:48:58.755085.755085 cuda_h.py:27] end moe_wait_copy_tasks cost 1.806 ms
DEBUG 05-06 11:48:58.766881.766881 cuda_h.py:27] end moe_vllm_forward cost 10.137 ms
DEBUG 05-06 11:48:58.777430.777430 cuda_h.py:27] end moe_cpu_merge cost 10.929 ms
DEBUG 05-06 11:48:58.777897.777897 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:58.778482.778482 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.854ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.778885.778885 cuda_h.py:27] end *layer_moe_fused cost 27.870 ms
DEBUG 05-06 11:48:58.779587.779587 cuda_h.py:27] end prefill_merge_scale cost 0.530 ms
DEBUG 05-06 11:48:58.779418.779418 cuda_h.py:27] end prefill_layer cost 32.640 ms
DEBUG 05-06 11:48:58.779097.779097 lmp.py:1391] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 11:48:58.779767.779767 lmp.py:1347] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 11:48:58.779678.779678 cuda_h.py:27] end prefill_ln cost 0.200 ms
DEBUG 05-06 11:48:58.781802.781802 cuda_h.py:27] end prefill_attn cost 1.840 ms
DEBUG 05-06 11:48:58.781612.781612 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 11:48:58.782289.782289 cuda_h.py:27] end prefill_gate cost 0.424 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:58.783494.783494 lmp.py:1836] [layer_moe_fused] layer=14 prefix: 0.343ms alloc: 0.173ms
INFO 05-06 11:48:58.783716.783716 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.1920928955078125e-05 seconds
INFO 05-06 11:48:58.785583.785583 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009593963623046875s
INFO 05-06 11:48:58.785998.785998 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.000278472900390625 seconds
DEBUG 05-06 11:48:58.785972.785972 cuda_h.py:27] end moe_cpu_prep_submit cost 0.591 ms
INFO 05-06 11:48:58.787910.787910 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014464855194091797s
DEBUG 05-06 11:48:58.787843.787843 cuda_h.py:27] end moe_wait_copy_tasks cost 1.647 ms
DEBUG 05-06 11:48:58.798298.798298 cuda_h.py:27] end moe_vllm_forward cost 9.957 ms
DEBUG 05-06 11:48:58.809954.809954 cuda_h.py:27] end moe_cpu_merge cost 10.974 ms
DEBUG 05-06 11:48:58.809229.809229 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:58.809006.809006 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.728ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.810946.810946 cuda_h.py:27] end *layer_moe_fused cost 26.873 ms
DEBUG 05-06 11:48:58.810898.810898 cuda_h.py:27] end prefill_merge_scale cost 0.505 ms
DEBUG 05-06 11:48:58.810829.810829 cuda_h.py:27] end prefill_layer cost 31.504 ms
DEBUG 05-06 11:48:58.810562.810562 lmp.py:1391] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 11:48:58.810040.810040 lmp.py:1347] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 11:48:58.811231.811231 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:48:58.813062.813062 cuda_h.py:27] end prefill_attn cost 1.810 ms
DEBUG 05-06 11:48:58.813978.813978 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 11:48:58.814351.814351 cuda_h.py:27] end prefill_gate cost 0.427 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:58.815084.815084 lmp.py:1836] [layer_moe_fused] layer=15 prefix: 0.345ms alloc: 0.106ms
INFO 05-06 11:48:58.815259.815259 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 9.775161743164062e-06 seconds
INFO 05-06 11:48:58.816773.816773 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009860992431640625s
INFO 05-06 11:48:58.816235.816235 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002777576446533203 seconds
DEBUG 05-06 11:48:58.817274.817274 cuda_h.py:27] end moe_cpu_prep_submit cost 0.764 ms
INFO 05-06 11:48:58.819098.819098 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014297962188720703s
DEBUG 05-06 11:48:58.819985.819985 cuda_h.py:27] end moe_wait_copy_tasks cost 1.595 ms
DEBUG 05-06 11:48:58.830194.830194 cuda_h.py:27] end moe_vllm_forward cost 9.946 ms
DEBUG 05-06 11:48:58.841415.841415 cuda_h.py:27] end moe_cpu_merge cost 11.040 ms
DEBUG 05-06 11:48:58.841590.841590 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:48:58.841129.841129 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.780ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.841587.841587 cuda_h.py:27] end *layer_moe_fused cost 26.950 ms
DEBUG 05-06 11:48:58.842453.842453 cuda_h.py:27] end prefill_merge_scale cost 0.510 ms
DEBUG 05-06 11:48:58.842191.842191 cuda_h.py:27] end prefill_layer cost 31.605 ms
DEBUG 05-06 11:48:58.842230.842230 lmp.py:1391] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 11:48:58.842662.842662 lmp.py:1347] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 11:48:58.843177.843177 cuda_h.py:27] end prefill_ln cost 0.202 ms
DEBUG 05-06 11:48:58.845995.845995 cuda_h.py:27] end prefill_attn cost 1.836 ms
DEBUG 05-06 11:48:58.845527.845527 cuda_h.py:27] end prefill_ffn_prep cost 0.378 ms
DEBUG 05-06 11:48:58.846218.846218 cuda_h.py:27] end prefill_gate cost 0.425 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:58.847189.847189 lmp.py:1836] [layer_moe_fused] layer=16 prefix: 0.346ms alloc: 0.104ms
INFO 05-06 11:48:58.847848.847848 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.3828277587890625e-05 seconds
INFO 05-06 11:48:58.848343.848343 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009615421295166016s
INFO 05-06 11:48:58.848189.848189 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002791881561279297 seconds
DEBUG 05-06 11:48:58.849009.849009 cuda_h.py:27] end moe_cpu_prep_submit cost 0.605 ms
INFO 05-06 11:48:58.850295.850295 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014874935150146484s
DEBUG 05-06 11:48:58.851751.851751 cuda_h.py:27] end moe_wait_copy_tasks cost 1.654 ms
DEBUG 05-06 11:48:58.861239.861239 cuda_h.py:27] end moe_vllm_forward cost 10.031 ms
DEBUG 05-06 11:48:58.872776.872776 cuda_h.py:27] end moe_cpu_merge cost 10.777 ms
DEBUG 05-06 11:48:58.872819.872819 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 11:48:58.873265.873265 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.639ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.873833.873833 cuda_h.py:27] end *layer_moe_fused cost 26.633 ms
DEBUG 05-06 11:48:58.873575.873575 cuda_h.py:27] end prefill_merge_scale cost 0.524 ms
DEBUG 05-06 11:48:58.874836.874836 cuda_h.py:27] end prefill_layer cost 31.306 ms
DEBUG 05-06 11:48:58.874849.874849 lmp.py:1391] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 11:48:58.874281.874281 lmp.py:1347] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 11:48:58.874948.874948 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:48:58.876153.876153 cuda_h.py:27] end prefill_attn cost 1.909 ms
DEBUG 05-06 11:48:58.877493.877493 cuda_h.py:27] end prefill_ffn_prep cost 0.377 ms
DEBUG 05-06 11:48:58.878905.878905 cuda_h.py:27] end prefill_gate cost 0.421 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:58.878420.878420 lmp.py:1836] [layer_moe_fused] layer=17 prefix: 0.353ms alloc: 0.111ms
INFO 05-06 11:48:58.878980.878980 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.3589859008789062e-05 seconds
INFO 05-06 11:48:58.880252.880252 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010328292846679688s
INFO 05-06 11:48:58.880105.880105 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002834796905517578 seconds
DEBUG 05-06 11:48:58.881178.881178 cuda_h.py:27] end moe_cpu_prep_submit cost 0.797 ms
INFO 05-06 11:48:58.883460.883460 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016624927520751953s
DEBUG 05-06 11:48:58.883830.883830 cuda_h.py:27] end moe_wait_copy_tasks cost 1.836 ms
DEBUG 05-06 11:48:58.894412.894412 cuda_h.py:27] end moe_vllm_forward cost 10.189 ms
DEBUG 05-06 11:48:58.905715.905715 cuda_h.py:27] end moe_cpu_merge cost 10.706 ms
DEBUG 05-06 11:48:58.905282.905282 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:58.905536.905536 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.732ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.906522.906522 cuda_h.py:27] end *layer_moe_fused cost 27.653 ms
DEBUG 05-06 11:48:58.906264.906264 cuda_h.py:27] end prefill_merge_scale cost 0.524 ms
DEBUG 05-06 11:48:58.906571.906571 cuda_h.py:27] end prefill_layer cost 32.411 ms
DEBUG 05-06 11:48:58.906664.906664 lmp.py:1391] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 11:48:58.906811.906811 lmp.py:1347] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 11:48:58.907143.907143 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 11:48:58.909447.909447 cuda_h.py:27] end prefill_attn cost 1.877 ms
DEBUG 05-06 11:48:58.909310.909310 cuda_h.py:27] end prefill_ffn_prep cost 0.378 ms
DEBUG 05-06 11:48:58.910153.910153 cuda_h.py:27] end prefill_gate cost 0.432 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:58.911012.911012 lmp.py:1836] [layer_moe_fused] layer=18 prefix: 0.358ms alloc: 0.112ms
INFO 05-06 11:48:58.911910.911910 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.5020370483398438e-05 seconds
INFO 05-06 11:48:58.912532.912532 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010001659393310547s
INFO 05-06 11:48:58.913723.913723 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002887248992919922 seconds
DEBUG 05-06 11:48:58.913924.913924 cuda_h.py:27] end moe_cpu_prep_submit cost 0.494 ms
INFO 05-06 11:48:58.916220.916220 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016672611236572266s
DEBUG 05-06 11:48:58.916729.916729 cuda_h.py:27] end moe_wait_copy_tasks cost 1.843 ms
DEBUG 05-06 11:48:58.927323.927323 cuda_h.py:27] end moe_vllm_forward cost 10.205 ms
DEBUG 05-06 11:48:58.938564.938564 cuda_h.py:27] end moe_cpu_merge cost 10.840 ms
DEBUG 05-06 11:48:58.938938.938938 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 11:48:58.938669.938669 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.863ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.938026.938026 cuda_h.py:27] end *layer_moe_fused cost 27.804 ms
DEBUG 05-06 11:48:58.939635.939635 cuda_h.py:27] end prefill_merge_scale cost 0.527 ms
DEBUG 05-06 11:48:58.939234.939234 cuda_h.py:27] end prefill_layer cost 32.614 ms
DEBUG 05-06 11:48:58.939043.939043 lmp.py:1391] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 11:48:58.939859.939859 lmp.py:1347] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 11:48:58.940884.940884 cuda_h.py:27] end prefill_ln cost 0.217 ms
DEBUG 05-06 11:48:58.942220.942220 cuda_h.py:27] end prefill_attn cost 1.866 ms
DEBUG 05-06 11:48:58.942322.942322 cuda_h.py:27] end prefill_ffn_prep cost 0.379 ms
DEBUG 05-06 11:48:58.943716.943716 cuda_h.py:27] end prefill_gate cost 0.482 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:58.944132.944132 lmp.py:1836] [layer_moe_fused] layer=19 prefix: 0.352ms alloc: 0.109ms
INFO 05-06 11:48:58.944936.944936 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.0251998901367188e-05 seconds
INFO 05-06 11:48:58.945544.945544 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009152889251708984s
INFO 05-06 11:48:58.946298.946298 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028252601623535156 seconds
DEBUG 05-06 11:48:58.946105.946105 cuda_h.py:27] end moe_cpu_prep_submit cost 0.605 ms
INFO 05-06 11:48:58.948635.948635 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.002142190933227539s
DEBUG 05-06 11:48:58.949183.949183 cuda_h.py:27] end moe_wait_copy_tasks cost 2.318 ms
DEBUG 05-06 11:48:58.959633.959633 cuda_h.py:27] end moe_vllm_forward cost 9.814 ms
DEBUG 05-06 11:48:58.970254.970254 cuda_h.py:27] end moe_cpu_merge cost 10.524 ms
DEBUG 05-06 11:48:58.970098.970098 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:48:58.970021.970021 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.148ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:58.970516.970516 cuda_h.py:27] end *layer_moe_fused cost 26.979 ms
DEBUG 05-06 11:48:58.971700.971700 cuda_h.py:27] end prefill_merge_scale cost 0.497 ms
DEBUG 05-06 11:48:58.971008.971008 cuda_h.py:27] end prefill_layer cost 31.745 ms
DEBUG 05-06 11:48:58.971725.971725 lmp.py:1391] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 11:48:58.971110.971110 lmp.py:1347] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 11:48:58.972824.972824 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:48:58.974012.974012 cuda_h.py:27] end prefill_attn cost 1.793 ms
DEBUG 05-06 11:48:58.974630.974630 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 11:48:58.975061.975061 cuda_h.py:27] end prefill_gate cost 0.420 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:58.976675.976675 lmp.py:1836] [layer_moe_fused] layer=20 prefix: 0.323ms alloc: 0.110ms
INFO 05-06 11:48:58.976135.976135 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.1920928955078125e-05 seconds
INFO 05-06 11:48:58.990491.990491 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.014208316802978516s
INFO 05-06 11:48:58.991429.991429 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002772808074951172 seconds
DEBUG 05-06 11:48:58.991675.991675 cuda_h.py:27] end moe_cpu_prep_submit cost 0.593 ms
INFO 05-06 11:48:58.993943.993943 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014188289642333984s
DEBUG 05-06 11:48:58.993306.993306 cuda_h.py:27] end moe_wait_copy_tasks cost 1.585 ms
DEBUG 05-06 11:48:59.002364.002364 cuda_h.py:27] end moe_vllm_forward cost 8.857 ms
DEBUG 05-06 11:48:59.014530.014530 cuda_h.py:27] end moe_cpu_merge cost 11.383 ms
DEBUG 05-06 11:48:59.014129.014129 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 11:48:59.014636.014636 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.030ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.015304.015304 cuda_h.py:27] end *layer_moe_fused cost 39.209 ms
DEBUG 05-06 11:48:59.015980.015980 cuda_h.py:27] end prefill_merge_scale cost 0.501 ms
DEBUG 05-06 11:48:59.015818.015818 cuda_h.py:27] end prefill_layer cost 43.883 ms
DEBUG 05-06 11:48:59.016563.016563 lmp.py:1391] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 11:48:59.016995.016995 lmp.py:1347] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 11:48:59.016689.016689 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 11:48:59.018241.018241 cuda_h.py:27] end prefill_attn cost 1.781 ms
DEBUG 05-06 11:48:59.018336.018336 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 11:48:59.019901.019901 cuda_h.py:27] end prefill_gate cost 0.437 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:59.020037.020037 lmp.py:1836] [layer_moe_fused] layer=21 prefix: 0.324ms alloc: 0.111ms
INFO 05-06 11:48:59.020828.020828 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.1682510375976562e-05 seconds
INFO 05-06 11:48:59.021272.021272 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000986337661743164s
INFO 05-06 11:48:59.022595.022595 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028014183044433594 seconds
DEBUG 05-06 11:48:59.022600.022600 cuda_h.py:27] end moe_cpu_prep_submit cost 0.581 ms
INFO 05-06 11:48:59.024127.024127 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001961946487426758s
DEBUG 05-06 11:48:59.024199.024199 cuda_h.py:27] end moe_wait_copy_tasks cost 2.124 ms
DEBUG 05-06 11:48:59.034709.034709 cuda_h.py:27] end moe_vllm_forward cost 8.708 ms
DEBUG 05-06 11:48:59.045185.045185 cuda_h.py:27] end moe_cpu_merge cost 11.337 ms
DEBUG 05-06 11:48:59.045638.045638 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:59.046270.046270 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.819ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.046614.046614 cuda_h.py:27] end *layer_moe_fused cost 26.392 ms
DEBUG 05-06 11:48:59.047660.047660 cuda_h.py:27] end prefill_merge_scale cost 0.502 ms
DEBUG 05-06 11:48:59.047921.047921 cuda_h.py:27] end prefill_layer cost 31.018 ms
DEBUG 05-06 11:48:59.047724.047724 lmp.py:1391] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 11:48:59.047440.047440 lmp.py:1347] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 11:48:59.047711.047711 cuda_h.py:27] end prefill_ln cost 0.214 ms
DEBUG 05-06 11:48:59.049132.049132 cuda_h.py:27] end prefill_attn cost 1.812 ms
DEBUG 05-06 11:48:59.050081.050081 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 11:48:59.051135.051135 cuda_h.py:27] end prefill_gate cost 0.423 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:59.051953.051953 lmp.py:1836] [layer_moe_fused] layer=22 prefix: 0.338ms alloc: 0.108ms
INFO 05-06 11:48:59.051804.051804 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.1920928955078125e-05 seconds
INFO 05-06 11:48:59.053606.053606 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009846687316894531s
INFO 05-06 11:48:59.053213.053213 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028061866760253906 seconds
DEBUG 05-06 11:48:59.053060.053060 cuda_h.py:27] end moe_cpu_prep_submit cost 0.592 ms
INFO 05-06 11:48:59.056920.056920 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001989126205444336s
DEBUG 05-06 11:48:59.056813.056813 cuda_h.py:27] end moe_wait_copy_tasks cost 2.182 ms
DEBUG 05-06 11:48:59.065629.065629 cuda_h.py:27] end moe_vllm_forward cost 8.741 ms
DEBUG 05-06 11:48:59.077019.077019 cuda_h.py:27] end moe_cpu_merge cost 11.341 ms
DEBUG 05-06 11:48:59.077473.077473 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:48:59.077820.077820 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.860ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.077164.077164 cuda_h.py:27] end *layer_moe_fused cost 26.519 ms
DEBUG 05-06 11:48:59.078593.078593 cuda_h.py:27] end prefill_merge_scale cost 0.501 ms
DEBUG 05-06 11:48:59.078570.078570 cuda_h.py:27] end prefill_layer cost 31.169 ms
DEBUG 05-06 11:48:59.078749.078749 lmp.py:1391] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 11:48:59.078373.078373 lmp.py:1347] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 11:48:59.079524.079524 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 11:48:59.081189.081189 cuda_h.py:27] end prefill_attn cost 2.213 ms
DEBUG 05-06 11:48:59.081776.081776 cuda_h.py:27] end prefill_ffn_prep cost 0.415 ms
DEBUG 05-06 11:48:59.082221.082221 cuda_h.py:27] end prefill_gate cost 0.417 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:59.083410.083410 lmp.py:1836] [layer_moe_fused] layer=23 prefix: 0.322ms alloc: 0.117ms
INFO 05-06 11:48:59.083678.083678 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 9.775161743164062e-06 seconds
INFO 05-06 11:48:59.084943.084943 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009796619415283203s
INFO 05-06 11:48:59.085027.085027 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002796649932861328 seconds
DEBUG 05-06 11:48:59.085146.085146 cuda_h.py:27] end moe_cpu_prep_submit cost 0.588 ms
INFO 05-06 11:48:59.087444.087444 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0018148422241210938s
DEBUG 05-06 11:48:59.087867.087867 cuda_h.py:27] end moe_wait_copy_tasks cost 1.993 ms
DEBUG 05-06 11:48:59.097959.097959 cuda_h.py:27] end moe_vllm_forward cost 8.814 ms
DEBUG 05-06 11:48:59.109269.109269 cuda_h.py:27] end moe_cpu_merge cost 11.319 ms
DEBUG 05-06 11:48:59.109391.109391 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 11:48:59.109546.109546 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.917ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.109341.109341 cuda_h.py:27] end *layer_moe_fused cost 26.656 ms
DEBUG 05-06 11:48:59.110147.110147 cuda_h.py:27] end prefill_merge_scale cost 0.502 ms
DEBUG 05-06 11:48:59.110886.110886 cuda_h.py:27] end prefill_layer cost 31.732 ms
DEBUG 05-06 11:48:59.110111.110111 lmp.py:1391] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 11:48:59.110543.110543 lmp.py:1347] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 11:48:59.111835.111835 cuda_h.py:27] end prefill_ln cost 0.203 ms
DEBUG 05-06 11:48:59.113686.113686 cuda_h.py:27] end prefill_attn cost 1.827 ms
DEBUG 05-06 11:48:59.113297.113297 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 11:48:59.114444.114444 cuda_h.py:27] end prefill_gate cost 0.423 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:59.115951.115951 lmp.py:1836] [layer_moe_fused] layer=24 prefix: 0.321ms alloc: 0.104ms
INFO 05-06 11:48:59.115841.115841 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.3828277587890625e-05 seconds
INFO 05-06 11:48:59.116117.116117 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009396076202392578s
INFO 05-06 11:48:59.116870.116870 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002796649932861328 seconds
DEBUG 05-06 11:48:59.117042.117042 cuda_h.py:27] end moe_cpu_prep_submit cost 0.583 ms
INFO 05-06 11:48:59.119243.119243 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0018165111541748047s
DEBUG 05-06 11:48:59.119222.119222 cuda_h.py:27] end moe_wait_copy_tasks cost 1.980 ms
DEBUG 05-06 11:48:59.129262.129262 cuda_h.py:27] end moe_vllm_forward cost 8.894 ms
DEBUG 05-06 11:48:59.140749.140749 cuda_h.py:27] end moe_cpu_merge cost 11.444 ms
DEBUG 05-06 11:48:59.140440.140440 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:59.140787.140787 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.117ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.141125.141125 cuda_h.py:27] end *layer_moe_fused cost 26.401 ms
DEBUG 05-06 11:48:59.141886.141886 cuda_h.py:27] end prefill_merge_scale cost 0.500 ms
DEBUG 05-06 11:48:59.141001.141001 cuda_h.py:27] end prefill_layer cost 31.095 ms
DEBUG 05-06 11:48:59.142981.142981 lmp.py:1391] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 11:48:59.142651.142651 lmp.py:1347] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 11:48:59.142553.142553 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:48:59.144027.144027 cuda_h.py:27] end prefill_attn cost 1.830 ms
DEBUG 05-06 11:48:59.144784.144784 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 11:48:59.145229.145229 cuda_h.py:27] end prefill_gate cost 0.418 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:59.146358.146358 lmp.py:1836] [layer_moe_fused] layer=25 prefix: 0.323ms alloc: 0.105ms
INFO 05-06 11:48:59.146725.146725 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.3589859008789062e-05 seconds
INFO 05-06 11:48:59.147470.147470 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009567737579345703s
INFO 05-06 11:48:59.148554.148554 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002803802490234375 seconds
DEBUG 05-06 11:48:59.148081.148081 cuda_h.py:27] end moe_cpu_prep_submit cost 0.563 ms
INFO 05-06 11:48:59.150926.150926 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001828908920288086s
DEBUG 05-06 11:48:59.150806.150806 cuda_h.py:27] end moe_wait_copy_tasks cost 1.991 ms
DEBUG 05-06 11:48:59.160464.160464 cuda_h.py:27] end moe_vllm_forward cost 8.754 ms
DEBUG 05-06 11:48:59.171271.171271 cuda_h.py:27] end moe_cpu_merge cost 11.331 ms
DEBUG 05-06 11:48:59.171248.171248 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:59.171357.171357 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.858ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.172372.172372 cuda_h.py:27] end *layer_moe_fused cost 26.181 ms
DEBUG 05-06 11:48:59.172026.172026 cuda_h.py:27] end prefill_merge_scale cost 0.495 ms
DEBUG 05-06 11:48:59.172619.172619 cuda_h.py:27] end prefill_layer cost 30.873 ms
DEBUG 05-06 11:48:59.173912.173912 lmp.py:1391] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 11:48:59.173867.173867 lmp.py:1347] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 11:48:59.173779.173779 cuda_h.py:27] end prefill_ln cost 0.202 ms
DEBUG 05-06 11:48:59.175902.175902 cuda_h.py:27] end prefill_attn cost 1.816 ms
DEBUG 05-06 11:48:59.175952.175952 cuda_h.py:27] end prefill_ffn_prep cost 0.413 ms
DEBUG 05-06 11:48:59.177741.177741 cuda_h.py:27] end prefill_gate cost 0.425 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:59.177394.177394 lmp.py:1836] [layer_moe_fused] layer=26 prefix: 0.326ms alloc: 0.103ms
INFO 05-06 11:48:59.177284.177284 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.33514404296875e-05 seconds
INFO 05-06 11:48:59.178223.178223 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009007453918457031s
INFO 05-06 11:48:59.179215.179215 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0002751350402832031 seconds
DEBUG 05-06 11:48:59.179930.179930 cuda_h.py:27] end moe_cpu_prep_submit cost 0.571 ms
INFO 05-06 11:48:59.181140.181140 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014493465423583984s
DEBUG 05-06 11:48:59.181165.181165 cuda_h.py:27] end moe_wait_copy_tasks cost 1.614 ms
DEBUG 05-06 11:48:59.191966.191966 cuda_h.py:27] end moe_vllm_forward cost 9.042 ms
DEBUG 05-06 11:48:59.204388.204388 cuda_h.py:27] end moe_cpu_merge cost 12.481 ms
DEBUG 05-06 11:48:59.204537.204537 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 11:48:59.204844.204844 lmp.py:1950] [layer_moe_fused] vllm triton time: 22.327ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.205944.205944 cuda_h.py:27] end *layer_moe_fused cost 27.735 ms
DEBUG 05-06 11:48:59.205347.205347 cuda_h.py:27] end prefill_merge_scale cost 0.519 ms
DEBUG 05-06 11:48:59.205562.205562 cuda_h.py:27] end prefill_layer cost 32.428 ms
DEBUG 05-06 11:48:59.205559.205559 lmp.py:1391] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 11:48:59.206752.206752 lmp.py:1347] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 11:48:59.206619.206619 cuda_h.py:27] end prefill_ln cost 0.199 ms
DEBUG 05-06 11:48:59.208478.208478 cuda_h.py:27] end prefill_attn cost 1.866 ms
DEBUG 05-06 11:48:59.208904.208904 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 11:48:59.209104.209104 cuda_h.py:27] end prefill_gate cost 0.418 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:59.210823.210823 lmp.py:1836] [layer_moe_fused] layer=27 prefix: 0.332ms alloc: 0.111ms
INFO 05-06 11:48:59.210137.210137 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.049041748046875e-05 seconds
INFO 05-06 11:48:59.212132.212132 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001085519790649414s
INFO 05-06 11:48:59.212555.212555 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028395652770996094 seconds
DEBUG 05-06 11:48:59.212116.212116 cuda_h.py:27] end moe_cpu_prep_submit cost 0.575 ms
INFO 05-06 11:48:59.214350.214350 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.002039670944213867s
DEBUG 05-06 11:48:59.215468.215468 cuda_h.py:27] end moe_wait_copy_tasks cost 2.203 ms
DEBUG 05-06 11:48:59.224949.224949 cuda_h.py:27] end moe_vllm_forward cost 8.763 ms
DEBUG 05-06 11:48:59.236302.236302 cuda_h.py:27] end moe_cpu_merge cost 11.419 ms
DEBUG 05-06 11:48:59.236186.236186 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:59.236817.236817 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.967ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.236189.236189 cuda_h.py:27] end *layer_moe_fused cost 26.807 ms
DEBUG 05-06 11:48:59.237943.237943 cuda_h.py:27] end prefill_merge_scale cost 0.498 ms
DEBUG 05-06 11:48:59.237296.237296 cuda_h.py:27] end prefill_layer cost 31.474 ms
DEBUG 05-06 11:48:59.237606.237606 lmp.py:1391] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 11:48:59.237482.237482 lmp.py:1347] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 11:48:59.237715.237715 cuda_h.py:27] end prefill_ln cost 0.201 ms
DEBUG 05-06 11:48:59.239555.239555 cuda_h.py:27] end prefill_attn cost 1.853 ms
DEBUG 05-06 11:48:59.240325.240325 cuda_h.py:27] end prefill_ffn_prep cost 0.378 ms
DEBUG 05-06 11:48:59.241710.241710 cuda_h.py:27] end prefill_gate cost 0.422 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:59.242846.242846 lmp.py:1836] [layer_moe_fused] layer=28 prefix: 0.325ms alloc: 0.107ms
INFO 05-06 11:48:59.242829.242829 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.2636184692382812e-05 seconds
INFO 05-06 11:48:59.243100.243100 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000980377197265625s
INFO 05-06 11:48:59.243635.243635 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00029730796813964844 seconds
DEBUG 05-06 11:48:59.244093.244093 cuda_h.py:27] end moe_cpu_prep_submit cost 0.653 ms
INFO 05-06 11:48:59.246650.246650 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001626729965209961s
DEBUG 05-06 11:48:59.246152.246152 cuda_h.py:27] end moe_wait_copy_tasks cost 1.791 ms
DEBUG 05-06 11:48:59.256398.256398 cuda_h.py:27] end moe_vllm_forward cost 8.692 ms
DEBUG 05-06 11:48:59.267528.267528 cuda_h.py:27] end moe_cpu_merge cost 11.287 ms
DEBUG 05-06 11:48:59.267319.267319 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:59.267143.267143 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.797ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.268377.268377 cuda_h.py:27] end *layer_moe_fused cost 26.777 ms
DEBUG 05-06 11:48:59.268561.268561 cuda_h.py:27] end prefill_merge_scale cost 0.499 ms
DEBUG 05-06 11:48:59.269869.269869 cuda_h.py:27] end prefill_layer cost 31.323 ms
DEBUG 05-06 11:48:59.269947.269947 lmp.py:1391] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 11:48:59.269902.269902 lmp.py:1347] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 11:48:59.269464.269464 cuda_h.py:27] end prefill_ln cost 0.201 ms
DEBUG 05-06 11:48:59.271577.271577 cuda_h.py:27] end prefill_attn cost 2.299 ms
DEBUG 05-06 11:48:59.272671.272671 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 11:48:59.273249.273249 cuda_h.py:27] end prefill_gate cost 0.420 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:48:59.274000.274000 lmp.py:1836] [layer_moe_fused] layer=29 prefix: 0.325ms alloc: 0.109ms
INFO 05-06 11:48:59.274421.274421 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.1920928955078125e-05 seconds
INFO 05-06 11:48:59.275582.275582 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009839534759521484s
INFO 05-06 11:48:59.275580.275580 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00028824806213378906 seconds
DEBUG 05-06 11:48:59.276758.276758 cuda_h.py:27] end moe_cpu_prep_submit cost 1.037 ms
INFO 05-06 11:48:59.278729.278729 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001644134521484375s
DEBUG 05-06 11:48:59.278331.278331 cuda_h.py:27] end moe_wait_copy_tasks cost 1.815 ms
DEBUG 05-06 11:48:59.288617.288617 cuda_h.py:27] end moe_vllm_forward cost 8.951 ms
DEBUG 05-06 11:48:59.299201.299201 cuda_h.py:27] end moe_cpu_merge cost 11.196 ms
DEBUG 05-06 11:48:59.299846.299846 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:48:59.299147.299147 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.926ms (seq_len=128 cg=False)
DEBUG 05-06 11:48:59.300494.300494 cuda_h.py:27] end *layer_moe_fused cost 26.644 ms
DEBUG 05-06 11:48:59.300062.300062 cuda_h.py:27] end prefill_merge_scale cost 0.503 ms
DEBUG 05-06 11:48:59.301893.301893 cuda_h.py:27] end prefill_layer cost 31.769 ms
DEBUG 05-06 11:48:59.301686.301686 lmp.py:1391] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 11:48:59.301357.301357 cuda_h.py:27] end prefill_step cost 1373.751 ms
INFO 05-06 11:48:59.301229.301229 lmp.py:1394] prefill time: 1.477144718170166 seconds
INFO 05-06 11:48:59.307038.307038 lmp.py:1406] Static-KV prefill complete; seqlens set to 128.
WARNING 05-06 11:48:59.337386.337386 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:48:59.337281.337281 helper.py:35]   NaN count (hidden): 1441792
WARNING 05-06 11:48:59.338883.338883 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:48:59.338503.338503 helper.py:39]   NaN count (normed): 1441792
WARNING 05-06 11:48:59.344326.344326 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:48:59.344828.344828 helper.py:50]   NaN count: 1048576
WARNING 05-06 11:48:59.344017.344017 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:48:59.460716.460716 cuda_h.py:27] end init_inputs_tokens cost 153.011 ms
DEBUG 05-06 11:48:59.460568.460568 lmp.py:1507] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:48:59.460981.460981 lmp.py:1513] ---- decode step 0 layer 0 ----
DEBUG 05-06 11:48:59.490174.490174 cuda_h.py:27] end decode_layer cost 29.684 ms
DEBUG 05-06 11:48:59.490470.490470 lmp.py:1513] ---- decode step 0 layer 1 ----
DEBUG 05-06 11:48:59.496544.496544 cuda_h.py:27] end decode_layer cost 5.770 ms
DEBUG 05-06 11:48:59.496308.496308 lmp.py:1513] ---- decode step 0 layer 2 ----
DEBUG 05-06 11:48:59.501901.501901 cuda_h.py:27] end decode_layer cost 4.855 ms
DEBUG 05-06 11:48:59.501096.501096 lmp.py:1513] ---- decode step 0 layer 3 ----
DEBUG 05-06 11:48:59.506431.506431 cuda_h.py:27] end decode_layer cost 5.262 ms
DEBUG 05-06 11:48:59.506466.506466 lmp.py:1513] ---- decode step 0 layer 4 ----
DEBUG 05-06 11:48:59.511806.511806 cuda_h.py:27] end decode_layer cost 4.808 ms
DEBUG 05-06 11:48:59.511623.511623 lmp.py:1513] ---- decode step 0 layer 5 ----
DEBUG 05-06 11:48:59.532263.532263 cuda_h.py:27] end decode_layer cost 21.673 ms
DEBUG 05-06 11:48:59.533067.533067 lmp.py:1513] ---- decode step 0 layer 6 ----
DEBUG 05-06 11:48:59.537038.537038 cuda_h.py:27] end decode_layer cost 4.678 ms
DEBUG 05-06 11:48:59.537550.537550 lmp.py:1513] ---- decode step 0 layer 7 ----
DEBUG 05-06 11:48:59.543307.543307 cuda_h.py:27] end decode_layer cost 5.398 ms
DEBUG 05-06 11:48:59.543395.543395 lmp.py:1513] ---- decode step 0 layer 8 ----
DEBUG 05-06 11:48:59.547367.547367 cuda_h.py:27] end decode_layer cost 4.678 ms
DEBUG 05-06 11:48:59.548640.548640 lmp.py:1513] ---- decode step 0 layer 9 ----
DEBUG 05-06 11:48:59.552206.552206 cuda_h.py:27] end decode_layer cost 4.835 ms
DEBUG 05-06 11:48:59.552765.552765 lmp.py:1513] ---- decode step 0 layer 10 ----
DEBUG 05-06 11:48:59.557471.557471 cuda_h.py:27] end decode_layer cost 4.658 ms
DEBUG 05-06 11:48:59.557552.557552 lmp.py:1513] ---- decode step 0 layer 11 ----
DEBUG 05-06 11:48:59.562883.562883 cuda_h.py:27] end decode_layer cost 5.330 ms
DEBUG 05-06 11:48:59.563395.563395 lmp.py:1513] ---- decode step 0 layer 12 ----
DEBUG 05-06 11:48:59.567310.567310 cuda_h.py:27] end decode_layer cost 4.776 ms
DEBUG 05-06 11:48:59.567630.567630 lmp.py:1513] ---- decode step 0 layer 13 ----
DEBUG 05-06 11:48:59.572825.572825 cuda_h.py:27] end decode_layer cost 4.842 ms
DEBUG 05-06 11:48:59.572476.572476 lmp.py:1513] ---- decode step 0 layer 14 ----
DEBUG 05-06 11:48:59.577037.577037 cuda_h.py:27] end decode_layer cost 4.655 ms
DEBUG 05-06 11:48:59.577880.577880 lmp.py:1513] ---- decode step 0 layer 15 ----
DEBUG 05-06 11:48:59.582706.582706 cuda_h.py:27] end decode_layer cost 4.886 ms
DEBUG 05-06 11:48:59.582171.582171 lmp.py:1513] ---- decode step 0 layer 16 ----
DEBUG 05-06 11:48:59.587145.587145 cuda_h.py:27] end decode_layer cost 4.750 ms
DEBUG 05-06 11:48:59.587465.587465 lmp.py:1513] ---- decode step 0 layer 17 ----
DEBUG 05-06 11:48:59.592939.592939 cuda_h.py:27] end decode_layer cost 5.049 ms
DEBUG 05-06 11:48:59.592497.592497 lmp.py:1513] ---- decode step 0 layer 18 ----
DEBUG 05-06 11:48:59.597912.597912 cuda_h.py:27] end decode_layer cost 4.653 ms
DEBUG 05-06 11:48:59.597708.597708 lmp.py:1513] ---- decode step 0 layer 19 ----
DEBUG 05-06 11:48:59.601355.601355 cuda_h.py:27] end decode_layer cost 4.860 ms
DEBUG 05-06 11:48:59.602913.602913 lmp.py:1513] ---- decode step 0 layer 20 ----
DEBUG 05-06 11:48:59.606275.606275 cuda_h.py:27] end decode_layer cost 4.686 ms
DEBUG 05-06 11:48:59.606549.606549 lmp.py:1513] ---- decode step 0 layer 21 ----
DEBUG 05-06 11:48:59.611474.611474 cuda_h.py:27] end decode_layer cost 4.890 ms
DEBUG 05-06 11:48:59.611179.611179 lmp.py:1513] ---- decode step 0 layer 22 ----
DEBUG 05-06 11:48:59.616997.616997 cuda_h.py:27] end decode_layer cost 4.846 ms
DEBUG 05-06 11:48:59.616747.616747 lmp.py:1513] ---- decode step 0 layer 23 ----
DEBUG 05-06 11:48:59.621512.621512 cuda_h.py:27] end decode_layer cost 5.228 ms
DEBUG 05-06 11:48:59.621116.621116 lmp.py:1513] ---- decode step 0 layer 24 ----
DEBUG 05-06 11:48:59.626610.626610 cuda_h.py:27] end decode_layer cost 4.641 ms
DEBUG 05-06 11:48:59.626168.626168 lmp.py:1513] ---- decode step 0 layer 25 ----
DEBUG 05-06 11:48:59.631648.631648 cuda_h.py:27] end decode_layer cost 4.842 ms
DEBUG 05-06 11:48:59.631114.631114 lmp.py:1513] ---- decode step 0 layer 26 ----
DEBUG 05-06 11:48:59.636284.636284 cuda_h.py:27] end decode_layer cost 4.684 ms
DEBUG 05-06 11:48:59.636889.636889 lmp.py:1513] ---- decode step 0 layer 27 ----
DEBUG 05-06 11:48:59.641746.641746 cuda_h.py:27] end decode_layer cost 4.839 ms
DEBUG 05-06 11:48:59.641735.641735 lmp.py:1513] ---- decode step 0 layer 28 ----
DEBUG 05-06 11:48:59.646623.646623 cuda_h.py:27] end decode_layer cost 4.757 ms
DEBUG 05-06 11:48:59.646466.646466 lmp.py:1513] ---- decode step 0 layer 29 ----
DEBUG 05-06 11:48:59.651135.651135 cuda_h.py:27] end decode_layer cost 5.122 ms
DEBUG 05-06 11:48:59.651821.651821 cuda_h.py:27] end decode_step cost 344.055 ms
INFO 05-06 11:48:59.651398.651398 lmp.py:1561] decode step 0 time: 0.3440985679626465 seconds
WARNING 05-06 11:48:59.651504.651504 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:48:59.651672.651672 helper.py:35]   NaN count (hidden): 11264
WARNING 05-06 11:48:59.652008.652008 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:48:59.652879.652879 helper.py:39]   NaN count (normed): 11264
WARNING 05-06 11:48:59.657057.657057 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:48:59.657406.657406 helper.py:50]   NaN count: 1048576
WARNING 05-06 11:48:59.657036.657036 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:48:59.658369.658369 cuda_h.py:27] end init_inputs_tokens cost 7.557 ms
DEBUG 05-06 11:48:59.659689.659689 lmp.py:1507] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:48:59.659790.659790 lmp.py:1513] ---- decode step 1 layer 0 ----
DEBUG 05-06 11:48:59.665258.665258 cuda_h.py:27] end decode_layer cost 6.061 ms
DEBUG 05-06 11:48:59.665054.665054 lmp.py:1513] ---- decode step 1 layer 1 ----
DEBUG 05-06 11:48:59.670674.670674 cuda_h.py:27] end decode_layer cost 4.839 ms
DEBUG 05-06 11:48:59.670616.670616 lmp.py:1513] ---- decode step 1 layer 2 ----
DEBUG 05-06 11:48:59.675073.675073 cuda_h.py:27] end decode_layer cost 4.930 ms
DEBUG 05-06 11:48:59.675638.675638 lmp.py:1513] ---- decode step 1 layer 3 ----
DEBUG 05-06 11:48:59.680258.680258 cuda_h.py:27] end decode_layer cost 4.875 ms
DEBUG 05-06 11:48:59.680684.680684 lmp.py:1513] ---- decode step 1 layer 4 ----
DEBUG 05-06 11:48:59.684553.684553 cuda_h.py:27] end decode_layer cost 4.778 ms
DEBUG 05-06 11:48:59.684396.684396 lmp.py:1513] ---- decode step 1 layer 5 ----
DEBUG 05-06 11:48:59.689723.689723 cuda_h.py:27] end decode_layer cost 5.009 ms
DEBUG 05-06 11:48:59.689189.689189 lmp.py:1513] ---- decode step 1 layer 6 ----
DEBUG 05-06 11:48:59.694911.694911 cuda_h.py:27] end decode_layer cost 4.740 ms
DEBUG 05-06 11:48:59.694992.694992 lmp.py:1513] ---- decode step 1 layer 7 ----
DEBUG 05-06 11:48:59.699225.699225 cuda_h.py:27] end decode_layer cost 4.765 ms
DEBUG 05-06 11:48:59.699697.699697 lmp.py:1513] ---- decode step 1 layer 8 ----
DEBUG 05-06 11:48:59.704011.704011 cuda_h.py:27] end decode_layer cost 4.825 ms
DEBUG 05-06 11:48:59.704570.704570 lmp.py:1513] ---- decode step 1 layer 9 ----
DEBUG 05-06 11:48:59.709026.709026 cuda_h.py:27] end decode_layer cost 4.719 ms
DEBUG 05-06 11:48:59.709346.709346 lmp.py:1513] ---- decode step 1 layer 10 ----
DEBUG 05-06 11:48:59.714145.714145 cuda_h.py:27] end decode_layer cost 4.867 ms
DEBUG 05-06 11:48:59.714749.714749 lmp.py:1513] ---- decode step 1 layer 11 ----
DEBUG 05-06 11:48:59.719510.719510 cuda_h.py:27] end decode_layer cost 4.909 ms
DEBUG 05-06 11:48:59.719307.719307 lmp.py:1513] ---- decode step 1 layer 12 ----
DEBUG 05-06 11:48:59.723777.723777 cuda_h.py:27] end decode_layer cost 4.729 ms
DEBUG 05-06 11:48:59.724381.724381 lmp.py:1513] ---- decode step 1 layer 13 ----
DEBUG 05-06 11:48:59.728222.728222 cuda_h.py:27] end decode_layer cost 4.933 ms
DEBUG 05-06 11:48:59.729542.729542 lmp.py:1513] ---- decode step 1 layer 14 ----
DEBUG 05-06 11:48:59.733875.733875 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 11:48:59.733957.733957 lmp.py:1513] ---- decode step 1 layer 15 ----
DEBUG 05-06 11:48:59.738537.738537 cuda_h.py:27] end decode_layer cost 4.671 ms
DEBUG 05-06 11:48:59.738857.738857 lmp.py:1513] ---- decode step 1 layer 16 ----
DEBUG 05-06 11:48:59.743329.743329 cuda_h.py:27] end decode_layer cost 4.801 ms
DEBUG 05-06 11:48:59.743411.743411 lmp.py:1513] ---- decode step 1 layer 17 ----
DEBUG 05-06 11:48:59.748827.748827 cuda_h.py:27] end decode_layer cost 4.900 ms
DEBUG 05-06 11:48:59.748193.748193 lmp.py:1513] ---- decode step 1 layer 18 ----
DEBUG 05-06 11:48:59.753551.753551 cuda_h.py:27] end decode_layer cost 4.752 ms
DEBUG 05-06 11:48:59.753109.753109 lmp.py:1513] ---- decode step 1 layer 19 ----
DEBUG 05-06 11:48:59.757399.757399 cuda_h.py:27] end decode_layer cost 4.702 ms
DEBUG 05-06 11:48:59.758481.758481 lmp.py:1513] ---- decode step 1 layer 20 ----
DEBUG 05-06 11:48:59.762658.762658 cuda_h.py:27] end decode_layer cost 4.900 ms
DEBUG 05-06 11:48:59.763978.763978 lmp.py:1513] ---- decode step 1 layer 21 ----
DEBUG 05-06 11:48:59.767867.767867 cuda_h.py:27] end decode_layer cost 4.793 ms
DEBUG 05-06 11:48:59.767764.767764 lmp.py:1513] ---- decode step 1 layer 22 ----
DEBUG 05-06 11:48:59.772159.772159 cuda_h.py:27] end decode_layer cost 4.885 ms
DEBUG 05-06 11:48:59.772717.772717 lmp.py:1513] ---- decode step 1 layer 23 ----
DEBUG 05-06 11:48:59.777921.777921 cuda_h.py:27] end decode_layer cost 4.884 ms
DEBUG 05-06 11:48:59.777764.777764 lmp.py:1513] ---- decode step 1 layer 24 ----
DEBUG 05-06 11:48:59.782919.782919 cuda_h.py:27] end decode_layer cost 4.812 ms
DEBUG 05-06 11:48:59.782239.782239 lmp.py:1513] ---- decode step 1 layer 25 ----
DEBUG 05-06 11:48:59.787453.787453 cuda_h.py:27] end decode_layer cost 4.821 ms
DEBUG 05-06 11:48:59.787773.787773 lmp.py:1513] ---- decode step 1 layer 26 ----
DEBUG 05-06 11:48:59.792245.792245 cuda_h.py:27] end decode_layer cost 4.801 ms
DEBUG 05-06 11:48:59.792612.792612 lmp.py:1513] ---- decode step 1 layer 27 ----
DEBUG 05-06 11:48:59.797710.797710 cuda_h.py:27] end decode_layer cost 4.701 ms
DEBUG 05-06 11:48:59.797030.797030 lmp.py:1513] ---- decode step 1 layer 28 ----
DEBUG 05-06 11:48:59.802072.802072 cuda_h.py:27] end decode_layer cost 4.836 ms
DEBUG 05-06 11:48:59.802200.802200 lmp.py:1513] ---- decode step 1 layer 29 ----
DEBUG 05-06 11:48:59.807500.807500 cuda_h.py:27] end decode_layer cost 4.990 ms
DEBUG 05-06 11:48:59.807854.807854 cuda_h.py:27] end decode_step cost 155.743 ms
INFO 05-06 11:48:59.807663.807663 lmp.py:1561] decode step 1 time: 0.15578007698059082 seconds
WARNING 05-06 11:48:59.807774.807774 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:48:59.807036.807036 helper.py:35]   NaN count (hidden): 11264
WARNING 05-06 11:48:59.808394.808394 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:48:59.808629.808629 helper.py:39]   NaN count (normed): 11264
WARNING 05-06 11:48:59.813560.813560 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:48:59.813385.813385 helper.py:50]   NaN count: 1048576
WARNING 05-06 11:48:59.813493.813493 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:48:59.814490.814490 cuda_h.py:27] end init_inputs_tokens cost 7.422 ms
DEBUG 05-06 11:48:59.814856.814856 lmp.py:1507] decode step 2 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:48:59.814526.814526 lmp.py:1513] ---- decode step 2 layer 0 ----
DEBUG 05-06 11:48:59.819059.819059 cuda_h.py:27] end decode_layer cost 4.634 ms
DEBUG 05-06 11:48:59.819810.819810 lmp.py:1513] ---- decode step 2 layer 1 ----
DEBUG 05-06 11:48:59.824415.824415 cuda_h.py:27] end decode_layer cost 4.830 ms
DEBUG 05-06 11:48:59.824788.824788 lmp.py:1513] ---- decode step 2 layer 2 ----
DEBUG 05-06 11:48:59.829242.829242 cuda_h.py:27] end decode_layer cost 4.647 ms
DEBUG 05-06 11:48:59.829039.829039 lmp.py:1513] ---- decode step 2 layer 3 ----
DEBUG 05-06 11:48:59.834305.834305 cuda_h.py:27] end decode_layer cost 4.965 ms
DEBUG 05-06 11:48:59.834009.834009 lmp.py:1513] ---- decode step 2 layer 4 ----
DEBUG 05-06 11:48:59.839054.839054 cuda_h.py:27] end decode_layer cost 5.294 ms
DEBUG 05-06 11:48:59.839235.839235 lmp.py:1513] ---- decode step 2 layer 5 ----
DEBUG 05-06 11:48:59.844871.844871 cuda_h.py:27] end decode_layer cost 5.168 ms
DEBUG 05-06 11:48:59.844430.844430 lmp.py:1513] ---- decode step 2 layer 6 ----
DEBUG 05-06 11:48:59.849745.849745 cuda_h.py:27] end decode_layer cost 4.650 ms
DEBUG 05-06 11:48:59.849303.849303 lmp.py:1513] ---- decode step 2 layer 7 ----
DEBUG 05-06 11:48:59.854581.854581 cuda_h.py:27] end decode_layer cost 4.728 ms
DEBUG 05-06 11:48:59.854139.854139 lmp.py:1513] ---- decode step 2 layer 8 ----
DEBUG 05-06 11:48:59.858913.858913 cuda_h.py:27] end decode_layer cost 4.707 ms
DEBUG 05-06 11:48:59.859948.859948 lmp.py:1513] ---- decode step 2 layer 9 ----
DEBUG 05-06 11:48:59.863169.863169 cuda_h.py:27] end decode_layer cost 4.827 ms
DEBUG 05-06 11:48:59.863012.863012 lmp.py:1513] ---- decode step 2 layer 10 ----
DEBUG 05-06 11:48:59.868514.868514 cuda_h.py:27] end decode_layer cost 4.682 ms
DEBUG 05-06 11:48:59.868933.868933 lmp.py:1513] ---- decode step 2 layer 11 ----
DEBUG 05-06 11:48:59.873480.873480 cuda_h.py:27] end decode_layer cost 5.067 ms
DEBUG 05-06 11:48:59.873661.873661 lmp.py:1513] ---- decode step 2 layer 12 ----
DEBUG 05-06 11:48:59.878381.878381 cuda_h.py:27] end decode_layer cost 4.667 ms
DEBUG 05-06 11:48:59.878939.878939 lmp.py:1513] ---- decode step 2 layer 13 ----
DEBUG 05-06 11:48:59.883817.883817 cuda_h.py:27] end decode_layer cost 4.854 ms
DEBUG 05-06 11:48:59.883899.883899 lmp.py:1513] ---- decode step 2 layer 14 ----
DEBUG 05-06 11:48:59.888392.888392 cuda_h.py:27] end decode_layer cost 4.641 ms
DEBUG 05-06 11:48:59.888473.888473 lmp.py:1513] ---- decode step 2 layer 15 ----
DEBUG 05-06 11:48:59.893094.893094 cuda_h.py:27] end decode_layer cost 4.875 ms
DEBUG 05-06 11:48:59.893652.893652 lmp.py:1513] ---- decode step 2 layer 16 ----
DEBUG 05-06 11:48:59.897887.897887 cuda_h.py:27] end decode_layer cost 4.626 ms
DEBUG 05-06 11:48:59.897683.897683 lmp.py:1513] ---- decode step 2 layer 17 ----
DEBUG 05-06 11:48:59.902283.902283 cuda_h.py:27] end decode_layer cost 5.036 ms
DEBUG 05-06 11:48:59.902841.902841 lmp.py:1513] ---- decode step 2 layer 18 ----
DEBUG 05-06 11:48:59.907521.907521 cuda_h.py:27] end decode_layer cost 4.884 ms
DEBUG 05-06 11:48:59.907702.907702 lmp.py:1513] ---- decode step 2 layer 19 ----
DEBUG 05-06 11:48:59.912775.912775 cuda_h.py:27] end decode_layer cost 4.928 ms
DEBUG 05-06 11:48:59.912546.912546 lmp.py:1513] ---- decode step 2 layer 20 ----
DEBUG 05-06 11:48:59.917972.917972 cuda_h.py:27] end decode_layer cost 4.627 ms
DEBUG 05-06 11:48:59.917054.917054 lmp.py:1513] ---- decode step 2 layer 21 ----
DEBUG 05-06 11:48:59.922000.922000 cuda_h.py:27] end decode_layer cost 4.905 ms
DEBUG 05-06 11:48:59.922604.922604 lmp.py:1513] ---- decode step 2 layer 22 ----
DEBUG 05-06 11:48:59.927126.927126 cuda_h.py:27] end decode_layer cost 4.697 ms
DEBUG 05-06 11:48:59.927446.927446 lmp.py:1513] ---- decode step 2 layer 23 ----
DEBUG 05-06 11:48:59.932048.932048 cuda_h.py:27] end decode_layer cost 5.107 ms
DEBUG 05-06 11:48:59.932275.932275 lmp.py:1513] ---- decode step 2 layer 24 ----
DEBUG 05-06 11:48:59.937269.937269 cuda_h.py:27] end decode_layer cost 4.765 ms
DEBUG 05-06 11:48:59.937543.937543 lmp.py:1513] ---- decode step 2 layer 25 ----
DEBUG 05-06 11:48:59.942282.942282 cuda_h.py:27] end decode_layer cost 4.858 ms
DEBUG 05-06 11:48:59.942363.942363 lmp.py:1513] ---- decode step 2 layer 26 ----
DEBUG 05-06 11:48:59.946971.946971 cuda_h.py:27] end decode_layer cost 4.690 ms
DEBUG 05-06 11:48:59.946483.946483 lmp.py:1513] ---- decode step 2 layer 27 ----
DEBUG 05-06 11:48:59.951870.951870 cuda_h.py:27] end decode_layer cost 4.844 ms
DEBUG 05-06 11:48:59.951713.951713 lmp.py:1513] ---- decode step 2 layer 28 ----
DEBUG 05-06 11:48:59.956210.956210 cuda_h.py:27] end decode_layer cost 4.749 ms
DEBUG 05-06 11:48:59.956868.956868 lmp.py:1513] ---- decode step 2 layer 29 ----
DEBUG 05-06 11:48:59.961711.961711 cuda_h.py:27] end decode_layer cost 5.180 ms
DEBUG 05-06 11:48:59.961356.961356 cuda_h.py:27] end decode_step cost 154.729 ms
INFO 05-06 11:48:59.962834.962834 lmp.py:1561] decode step 2 time: 0.1547679901123047 seconds
Time taken: 7.48088251426816 seconds
generate input ids cost 0.04131817817687988 s
DEBUG 05-06 11:49:02.738776.738776 cuda_h.py:27] end generate_input_ids cost 2649.839 ms
DEBUG 05-06 11:49:02.738915.738915 cuda_h.py:27] end init_cache cost 0.043 ms
INFO 05-06 11:49:02.738404.738404 lmp.py:1160] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 11:49:02.751444.751444 lmp.py:2794] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 4740325316, 'cuda:1': 12875595776, 'cuda:2': 12875595776, 'cuda:3': 12875595776} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7886239265315211, 'cuda:1': 0.4700220660037874, 'cuda:2': 0.4700220660037874, 'cuda:3': 0.4700220660037874}
INFO 05-06 11:49:02.751871.751871 lmp.py:2812] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.751409.751409 lmp.py:2812] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.751271.751271 lmp.py:2812] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.751511.751511 lmp.py:2812] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.751227.751227 lmp.py:2812] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.751940.751940 lmp.py:2812] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.751994.751994 lmp.py:2812] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.751357.751357 lmp.py:2812] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.752696.752696 lmp.py:2812] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.752484.752484 lmp.py:2812] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.752062.752062 lmp.py:2812] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.752228.752228 lmp.py:2812] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.752852.752852 lmp.py:2812] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.752057.752057 lmp.py:2812] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.752873.752873 lmp.py:2812] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.752164.752164 lmp.py:2812] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.752880.752880 lmp.py:2812] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.753219.753219 lmp.py:2812] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.753245.753245 lmp.py:2812] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.753723.753723 lmp.py:2812] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.753974.753974 lmp.py:2812] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.753452.753452 lmp.py:2812] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.753559.753559 lmp.py:2812] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.753798.753798 lmp.py:2812] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.753904.753904 lmp.py:2812] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.753859.753859 lmp.py:2812] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.754234.754234 lmp.py:2812] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.754427.754427 lmp.py:2812] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.754585.754585 lmp.py:2812] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:49:02.754970.754970 lmp.py:2812] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 11:49:03.040939.040939 cuda_h.py:27] end init_loading_placement cost 301.214 ms
DEBUG 05-06 11:49:03.040287.040287 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:49:03.040104.040104 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:49:03 client.py:72] load_into_gpu: gemma4-26B-A4B, 74f31f33-7296-45b5-a627-faeff6e9759a
INFO 05-06 11:49:03 client.py:135] Model loaded: gemma4-26B-A4B, 74f31f33-7296-45b5-a627-faeff6e9759a
INFO 05-06 11:49:03 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 74f31f33-7296-45b5-a627-faeff6e9759a
INFO 05-06 11:49:03 client.py:212] Model loaded
DEBUG 05-06 11:49:03.567546.567546 cuda_h.py:27] end init_general_sagl_loading_async cost 527.283 ms
INFO 05-06 11:49:03.614052.614052 lmp.py:3315] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 11:49:03.714516.714516 cuda_h.py:27] end restore_state_dict cost 99.765 ms
INFO 05-06 11:49:03.716768.716768 lmp.py:1288] vLLM Triton pre-warmup done in 2.0 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 11:49:03.716532.716532 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:49:03.716692.716692 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:49:03 client.py:72] load_into_gpu: gemma4-26B-A4B, 8eb641b8-7f4c-44b7-a0af-f4d0bbeb0fb7
INFO 05-06 11:49:03 client.py:135] Model loaded: gemma4-26B-A4B, 8eb641b8-7f4c-44b7-a0af-f4d0bbeb0fb7
DEBUG 05-06 11:49:03.840760.840760 cuda_h.py:27] end init_experts_loading_async cost 123.676 ms
DEBUG 05-06 11:49:03.841362.841362 cuda_h.py:27] end init_inputs_tokens cost 1.039 ms
DEBUG 05-06 11:49:03.841010.841010 lmp.py:1347] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 11:49:03.842717.842717 cuda_h.py:27] end prefill_ln cost 0.805 ms
DEBUG 05-06 11:49:03.848724.848724 cuda_h.py:27] end prefill_attn cost 5.400 ms
DEBUG 05-06 11:49:03.849169.849169 cuda_h.py:27] end prefill_ffn_prep cost 0.505 ms
DEBUG 05-06 11:49:03.850869.850869 cuda_h.py:27] end prefill_gate cost 0.675 ms
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 101, 109, 85, 49, 45, 65, 93, 69, 5, 13, 9, 73, 77, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18], 'token_total': 420, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 13: 16, 9: 17, 73: 19, 77: 19, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 51, 55, 67, 75, 83, 91, 103, 115, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1216, 'token_per_expert': {3: 46, 7: 95, 31: 34, 39: 176, 47: 318, 51: 48, 55: 51, 67: 47, 75: 29, 83: 33, 91: 99, 103: 178, 115: 29, 127: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 32, 48, 52, 64, 68, 72, 92, 104, 112, 116, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 758, 'token_per_expert': {0: 73, 16: 48, 32: 43, 48: 41, 52: 43, 64: 27, 68: 170, 72: 35, 92: 16, 104: 43, 112: 23, 116: 18, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 25, 33, 37, 41, 53, 89, 105, 113, 117, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 809, 'token_per_expert': {1: 75, 21: 48, 25: 24, 33: 210, 37: 20, 41: 27, 53: 205, 89: 20, 105: 24, 113: 39, 117: 26, 121: 65, 125: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 46, 50, 54, 70, 74, 78, 90, 110, 118, 122, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 893, 'token_per_expert': {22: 64, 26: 59, 46: 119, 50: 110, 54: 59, 70: 25, 74: 61, 78: 36, 90: 154, 110: 27, 118: 29, 122: 35, 126: 115}}
INFO 05-06 11:49:03.851240.851240 lmp.py:1836] [layer_moe_fused] layer=0 prefix: 0.610ms alloc: 0.425ms
INFO 05-06 11:49:03.852464.852464 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.626678466796875e-05 seconds
INFO 05-06 11:49:03.853250.853250 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=106 time: 0.001466989517211914s
INFO 05-06 11:49:03.854743.854743 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007963180541992188 seconds
DEBUG 05-06 11:49:03.854985.854985 cuda_h.py:27] end moe_cpu_prep_submit cost 0.905 ms
INFO 05-06 11:49:03.972572.972572 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.11751413345336914s
DEBUG 05-06 11:49:03.972394.972394 cuda_h.py:27] end moe_wait_copy_tasks cost 117.753 ms
DEBUG 05-06 11:49:03.989731.989731 cuda_h.py:27] end moe_vllm_forward cost 16.196 ms
DEBUG 05-06 11:49:03.989684.989684 cuda_h.py:27] end moe_cpu_merge cost 0.064 ms
DEBUG 05-06 11:49:03.990255.990255 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:49:03.990065.990065 lmp.py:1950] [layer_moe_fused] vllm triton time: 17.349ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:03.990323.990323 cuda_h.py:27] end *layer_moe_fused cost 139.807 ms
DEBUG 05-06 11:49:03.991951.991951 cuda_h.py:27] end prefill_merge_scale cost 0.693 ms
DEBUG 05-06 11:49:03.991583.991583 cuda_h.py:27] end prefill_layer cost 149.838 ms
DEBUG 05-06 11:49:03.991582.991582 lmp.py:1391] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 11:49:03.991285.991285 lmp.py:1347] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 11:49:03.992899.992899 cuda_h.py:27] end prefill_ln cost 0.184 ms
DEBUG 05-06 11:49:03.994731.994731 cuda_h.py:27] end prefill_attn cost 2.243 ms
DEBUG 05-06 11:49:03.994532.994532 cuda_h.py:27] end prefill_ffn_prep cost 0.297 ms
DEBUG 05-06 11:49:03.995245.995245 cuda_h.py:27] end prefill_gate cost 0.393 ms
experts_cpu_alloc {'expert_ids': [39, 115, 15, 31, 55, 103, 43, 83, 87, 123, 91, 16, 44, 72, 88, 40, 60, 116, 32, 84, 112, 56, 108, 48, 76, 104, 64, 61, 33, 117, 57, 45, 81, 89, 29, 41, 121, 125, 93, 37, 69, 85, 18, 26, 66, 14, 38, 110, 62, 74, 90, 78, 50, 34, 98], 'token_total': 325, 'token_per_expert': {39: 1, 115: 1, 15: 2, 31: 3, 55: 3, 103: 3, 43: 4, 83: 4, 87: 4, 123: 4, 91: 5, 16: 2, 44: 2, 72: 2, 88: 3, 40: 4, 60: 4, 116: 4, 32: 5, 84: 5, 112: 5, 56: 7, 108: 7, 48: 8, 76: 9, 104: 13, 64: 20, 61: 1, 33: 2, 117: 2, 57: 3, 45: 4, 81: 4, 89: 4, 29: 5, 41: 5, 121: 5, 125: 5, 93: 6, 37: 9, 69: 9, 85: 9, 18: 3, 26: 4, 66: 4, 14: 5, 38: 5, 110: 5, 62: 6, 74: 10, 90: 10, 78: 12, 50: 17, 34: 18, 98: 19}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 35, 47, 51, 59, 67, 79, 95, 99, 119, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 757, 'token_per_expert': {3: 150, 7: 153, 11: 8, 27: 11, 35: 21, 47: 34, 51: 47, 59: 25, 67: 114, 79: 24, 95: 18, 99: 97, 119: 18, 127: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 28, 52, 68, 80, 92, 96, 100, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 998, 'token_per_expert': {0: 135, 4: 134, 8: 87, 12: 39, 20: 47, 28: 49, 52: 205, 68: 129, 80: 34, 92: 21, 96: 29, 100: 35, 120: 25, 124: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 49, 53, 65, 73, 97, 101, 105, 109], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 921, 'token_per_expert': {1: 151, 5: 211, 9: 13, 13: 203, 21: 10, 25: 35, 49: 24, 53: 29, 65: 24, 73: 27, 97: 87, 101: 10, 105: 19, 109: 78}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 30, 42, 46, 54, 82, 94, 106, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 1095, 'token_per_expert': {2: 130, 6: 131, 10: 139, 22: 87, 30: 184, 42: 35, 46: 30, 54: 50, 82: 137, 94: 25, 106: 28, 118: 41, 122: 78}}
INFO 05-06 11:49:03.996107.996107 lmp.py:1836] [layer_moe_fused] layer=1 prefix: 0.370ms alloc: 0.256ms
INFO 05-06 11:49:03.996641.996641 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 4.363059997558594e-05 seconds
INFO 05-06 11:49:03.998699.998699 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001889944076538086s
INFO 05-06 11:49:03.999485.999485 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007891654968261719 seconds
DEBUG 05-06 11:49:03.999719.999719 cuda_h.py:27] end moe_cpu_prep_submit cost 0.856 ms
INFO 05-06 11:49:04.005539.005539 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0058574676513671875s
DEBUG 05-06 11:49:04.005086.005086 cuda_h.py:27] end moe_wait_copy_tasks cost 5.994 ms
DEBUG 05-06 11:49:04.010332.010332 cuda_h.py:27] end moe_vllm_forward cost 4.172 ms
DEBUG 05-06 11:49:04.010728.010728 cuda_h.py:27] end moe_cpu_merge cost 0.061 ms
DEBUG 05-06 11:49:04.010087.010087 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.010674.010674 lmp.py:1950] [layer_moe_fused] vllm triton time: 5.059ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.011607.011607 cuda_h.py:27] end *layer_moe_fused cost 15.358 ms
DEBUG 05-06 11:49:04.016219.016219 cuda_h.py:27] end prefill_merge_scale cost 5.552 ms
DEBUG 05-06 11:49:04.016017.016017 cuda_h.py:27] end prefill_layer cost 25.220 ms
DEBUG 05-06 11:49:04.017964.017964 lmp.py:1391] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 11:49:04.017442.017442 lmp.py:1347] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 11:49:04.017315.017315 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:49:04.019697.019697 cuda_h.py:27] end prefill_attn cost 1.830 ms
DEBUG 05-06 11:49:04.019171.019171 cuda_h.py:27] end prefill_ffn_prep cost 0.443 ms
DEBUG 05-06 11:49:04.021153.021153 cuda_h.py:27] end prefill_gate cost 0.420 ms
experts_cpu_alloc {'expert_ids': [67, 111, 115, 99, 27, 95, 63, 71, 123, 35, 43, 23, 31, 107, 120, 96, 116, 40, 72, 100, 64, 44, 88, 8, 52, 24, 28, 45, 61, 21, 121, 105, 113, 77, 69, 85, 17, 57, 49, 33, 26, 66, 50, 42, 82, 114, 126, 58, 70, 46, 98, 122], 'token_total': 572, 'token_per_expert': {67: 2, 111: 3, 115: 4, 99: 5, 27: 7, 95: 9, 63: 15, 71: 15, 123: 15, 35: 16, 43: 17, 23: 19, 31: 20, 107: 20, 120: 3, 96: 5, 116: 5, 40: 7, 72: 8, 100: 9, 64: 11, 44: 12, 88: 12, 8: 14, 52: 14, 24: 15, 28: 15, 45: 2, 61: 4, 21: 5, 121: 10, 105: 12, 113: 12, 77: 13, 69: 14, 85: 14, 17: 15, 57: 15, 49: 20, 33: 22, 26: 1, 66: 1, 50: 3, 42: 7, 82: 8, 114: 8, 126: 10, 58: 11, 70: 13, 46: 16, 98: 17, 122: 22}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 51, 55, 59, 83, 91, 103, 119, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 984, 'token_per_expert': {3: 144, 7: 168, 11: 149, 15: 80, 19: 85, 51: 28, 55: 52, 59: 81, 83: 23, 91: 36, 103: 24, 119: 21, 127: 93}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 36, 48, 56, 60, 76, 80, 84, 104, 108, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 768, 'token_per_expert': {0: 135, 4: 150, 20: 30, 36: 16, 48: 42, 56: 17, 60: 31, 76: 51, 80: 43, 84: 52, 104: 37, 108: 143, 124: 21}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 29, 37, 41, 53, 65, 81, 97, 109, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 954, 'token_per_expert': {1: 218, 5: 130, 9: 86, 13: 62, 29: 60, 37: 54, 41: 118, 53: 24, 65: 31, 81: 46, 97: 29, 109: 29, 125: 67}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 34, 54, 62, 78, 90, 102, 106, 110, 118], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 818, 'token_per_expert': {2: 128, 6: 129, 14: 28, 18: 49, 34: 29, 54: 93, 62: 122, 78: 29, 90: 51, 102: 63, 106: 25, 110: 24, 118: 48}}
INFO 05-06 11:49:04.022381.022381 lmp.py:1836] [layer_moe_fused] layer=2 prefix: 0.429ms alloc: 0.388ms
INFO 05-06 11:49:04.022956.022956 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.602836608886719e-05 seconds
INFO 05-06 11:49:04.023884.023884 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008153915405273438s
INFO 05-06 11:49:04.023489.023489 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005919933319091797 seconds
DEBUG 05-06 11:49:04.024609.024609 cuda_h.py:27] end moe_cpu_prep_submit cost 0.997 ms
INFO 05-06 11:49:04.038455.038455 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014333486557006836s
DEBUG 05-06 11:49:04.038273.038273 cuda_h.py:27] end moe_wait_copy_tasks cost 14.457 ms
DEBUG 05-06 11:49:04.042304.042304 cuda_h.py:27] end moe_vllm_forward cost 3.518 ms
DEBUG 05-06 11:49:04.042401.042401 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:49:04.043881.043881 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.043321.043321 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.351ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.043958.043958 cuda_h.py:27] end *layer_moe_fused cost 22.384 ms
DEBUG 05-06 11:49:04.048561.048561 cuda_h.py:27] end prefill_merge_scale cost 4.700 ms
DEBUG 05-06 11:49:04.048452.048452 cuda_h.py:27] end prefill_layer cost 31.325 ms
DEBUG 05-06 11:49:04.048487.048487 lmp.py:1391] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 11:49:04.048011.048011 lmp.py:1347] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 11:49:04.049718.049718 cuda_h.py:27] end prefill_ln cost 0.203 ms
DEBUG 05-06 11:49:04.051025.051025 cuda_h.py:27] end prefill_attn cost 1.769 ms
DEBUG 05-06 11:49:04.051550.051550 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 11:49:04.052130.052130 cuda_h.py:27] end prefill_gate cost 0.418 ms
experts_cpu_alloc {'expert_ids': [23, 55, 35, 91, 27, 127, 43, 63, 31, 67, 111, 123, 20, 80, 72, 36, 16, 60, 32, 100, 56, 116, 8, 24, 48, 40, 64, 65, 21, 29, 89, 41, 117, 57, 13, 61, 33, 101, 77, 46, 94, 18, 82, 30, 98, 42, 86, 110, 114, 26, 74, 58, 54, 70], 'token_total': 521, 'token_per_expert': {23: 1, 55: 1, 35: 3, 91: 3, 27: 4, 127: 8, 43: 9, 63: 9, 31: 12, 67: 15, 111: 16, 123: 16, 20: 1, 80: 1, 72: 2, 36: 3, 16: 5, 60: 5, 32: 6, 100: 6, 56: 8, 116: 8, 8: 10, 24: 10, 48: 11, 40: 12, 64: 18, 65: 2, 21: 3, 29: 3, 89: 3, 41: 4, 117: 5, 57: 6, 13: 11, 61: 12, 33: 13, 101: 14, 77: 17, 46: 1, 94: 1, 18: 3, 82: 3, 30: 6, 98: 6, 42: 8, 86: 10, 110: 20, 114: 22, 26: 24, 74: 28, 58: 30, 54: 31, 70: 32}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 39, 51, 59, 71, 75, 83, 95, 107, 119], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 756, 'token_per_expert': {3: 153, 7: 128, 11: 38, 15: 29, 19: 20, 39: 19, 51: 31, 59: 23, 71: 66, 75: 77, 83: 47, 95: 57, 107: 39, 119: 29}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 44, 52, 68, 76, 84, 88, 92, 96, 104, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 864, 'token_per_expert': {0: 158, 4: 176, 28: 72, 44: 24, 52: 37, 68: 47, 76: 34, 84: 54, 88: 51, 92: 58, 96: 41, 104: 36, 108: 36, 120: 40}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 25, 53, 69, 73, 85, 93, 97, 109, 121], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 847, 'token_per_expert': {1: 133, 5: 171, 9: 59, 17: 38, 25: 41, 53: 41, 69: 32, 73: 28, 85: 96, 93: 66, 97: 53, 109: 23, 121: 66}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 34, 50, 62, 66, 78, 102, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 1108, 'token_per_expert': {2: 149, 6: 145, 10: 46, 14: 62, 22: 83, 34: 62, 50: 126, 62: 79, 66: 83, 78: 89, 102: 92, 118: 39, 122: 53}}
INFO 05-06 11:49:04.053067.053067 lmp.py:1836] [layer_moe_fused] layer=3 prefix: 0.423ms alloc: 0.392ms
INFO 05-06 11:49:04.053595.053595 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.841255187988281e-05 seconds
INFO 05-06 11:49:04.054005.054005 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007576942443847656s
INFO 05-06 11:49:04.055833.055833 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005466938018798828 seconds
DEBUG 05-06 11:49:04.056217.056217 cuda_h.py:27] end moe_cpu_prep_submit cost 1.281 ms
INFO 05-06 11:49:04.069434.069434 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013331890106201172s
DEBUG 05-06 11:49:04.070524.070524 cuda_h.py:27] end moe_wait_copy_tasks cost 13.447 ms
DEBUG 05-06 11:49:04.073860.073860 cuda_h.py:27] end moe_vllm_forward cost 3.530 ms
DEBUG 05-06 11:49:04.074288.074288 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 11:49:04.074439.074439 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:49:04.074733.074733 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.224ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.074058.074058 cuda_h.py:27] end *layer_moe_fused cost 21.915 ms
DEBUG 05-06 11:49:04.080764.080764 cuda_h.py:27] end prefill_merge_scale cost 5.587 ms
DEBUG 05-06 11:49:04.080078.080078 cuda_h.py:27] end prefill_layer cost 31.649 ms
DEBUG 05-06 11:49:04.080183.080183 lmp.py:1391] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 11:49:04.080946.080946 lmp.py:1347] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 11:49:04.081296.081296 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:49:04.082311.082311 cuda_h.py:27] end prefill_attn cost 1.772 ms
DEBUG 05-06 11:49:04.083459.083459 cuda_h.py:27] end prefill_ffn_prep cost 0.381 ms
DEBUG 05-06 11:49:04.084085.084085 cuda_h.py:27] end prefill_gate cost 0.422 ms
experts_cpu_alloc {'expert_ids': [35, 79, 31, 103, 107, 15, 75, 91, 47, 71, 123, 87, 19, 39, 12, 72, 56, 44, 120, 40, 80, 84, 36, 88, 52, 108, 64, 41, 121, 69, 101, 21, 37, 45, 77, 109, 81, 25, 73, 97, 17, 58, 70, 114, 18, 46, 66, 122, 126, 90, 34, 38, 78, 98], 'token_total': 510, 'token_per_expert': {35: 1, 79: 1, 31: 3, 103: 3, 107: 13, 15: 14, 75: 15, 91: 16, 47: 23, 71: 23, 123: 23, 87: 25, 19: 34, 39: 35, 12: 1, 72: 1, 56: 5, 44: 6, 120: 7, 40: 8, 80: 8, 84: 10, 36: 11, 88: 12, 52: 14, 108: 14, 64: 15, 41: 1, 121: 1, 69: 2, 101: 3, 21: 4, 37: 6, 45: 6, 77: 6, 109: 7, 81: 8, 25: 10, 73: 11, 97: 11, 17: 13, 58: 1, 70: 1, 114: 2, 18: 3, 46: 3, 66: 4, 122: 4, 126: 5, 90: 8, 34: 9, 38: 10, 78: 13, 98: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 43, 51, 55, 59, 63, 67, 83, 111, 115, 119], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1321, 'token_per_expert': {3: 160, 7: 133, 23: 87, 27: 58, 43: 107, 51: 52, 55: 44, 59: 124, 63: 202, 67: 36, 83: 58, 111: 76, 115: 74, 119: 110}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 24, 28, 32, 60, 76, 92, 96, 104, 116, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 756, 'token_per_expert': {0: 128, 4: 145, 8: 126, 20: 25, 24: 69, 28: 28, 32: 45, 60: 27, 76: 41, 92: 23, 96: 27, 104: 25, 116: 16, 124: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 49, 53, 57, 61, 85, 89, 93, 105, 113, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 758, 'token_per_expert': {1: 186, 5: 149, 29: 42, 49: 27, 53: 41, 57: 14, 61: 21, 85: 37, 89: 77, 93: 36, 105: 16, 113: 64, 117: 14, 125: 34}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 30, 54, 62, 74, 82, 86, 94, 106, 118], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 751, 'token_per_expert': {2: 128, 6: 131, 22: 60, 26: 58, 30: 21, 54: 53, 62: 26, 74: 70, 82: 48, 86: 19, 94: 27, 106: 93, 118: 17}}
INFO 05-06 11:49:04.085658.085658 lmp.py:1836] [layer_moe_fused] layer=4 prefix: 0.426ms alloc: 0.401ms
INFO 05-06 11:49:04.085517.085517 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.817413330078125e-05 seconds
INFO 05-06 11:49:04.086251.086251 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007402896881103516s
INFO 05-06 11:49:04.087517.087517 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005526542663574219 seconds
DEBUG 05-06 11:49:04.087716.087716 cuda_h.py:27] end moe_cpu_prep_submit cost 1.125 ms
INFO 05-06 11:49:04.103047.103047 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015245437622070312s
DEBUG 05-06 11:49:04.103051.103051 cuda_h.py:27] end moe_wait_copy_tasks cost 15.369 ms
DEBUG 05-06 11:49:04.107982.107982 cuda_h.py:27] end moe_vllm_forward cost 3.502 ms
DEBUG 05-06 11:49:04.107982.107982 cuda_h.py:27] end moe_cpu_merge cost 0.127 ms
DEBUG 05-06 11:49:04.107974.107974 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.107552.107552 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.252ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.108585.108585 cuda_h.py:27] end *layer_moe_fused cost 23.566 ms
DEBUG 05-06 11:49:04.113571.113571 cuda_h.py:27] end prefill_merge_scale cost 4.885 ms
DEBUG 05-06 11:49:04.113217.113217 cuda_h.py:27] end prefill_layer cost 32.606 ms
DEBUG 05-06 11:49:04.113799.113799 lmp.py:1391] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 11:49:04.113323.113323 lmp.py:1347] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 11:49:04.113269.113269 cuda_h.py:27] end prefill_ln cost 0.203 ms
DEBUG 05-06 11:49:04.118779.118779 cuda_h.py:27] end prefill_attn cost 4.700 ms
DEBUG 05-06 11:49:04.119161.119161 cuda_h.py:27] end prefill_ffn_prep cost 0.432 ms
DEBUG 05-06 11:49:04.120160.120160 cuda_h.py:27] end prefill_gate cost 0.582 ms
experts_cpu_alloc {'expert_ids': [15, 51, 19, 115, 27, 67, 75, 107, 31, 83, 119, 79, 32, 92, 124, 8, 56, 68, 52, 84, 44, 96, 100, 120, 80, 104, 60, 17, 21, 77, 105, 53, 57, 37, 113, 30, 78, 82, 38, 58, 102, 26, 50, 54, 86, 114, 106, 98, 34, 62], 'token_total': 273, 'token_per_expert': {15: 1, 51: 1, 19: 2, 115: 2, 27: 3, 67: 6, 75: 7, 107: 7, 31: 8, 83: 8, 119: 8, 79: 10, 32: 1, 92: 1, 124: 1, 8: 2, 56: 4, 68: 5, 52: 6, 84: 7, 44: 9, 96: 13, 100: 16, 120: 16, 80: 18, 104: 19, 60: 20, 17: 1, 21: 1, 77: 1, 105: 1, 53: 3, 57: 3, 37: 4, 113: 5, 30: 1, 78: 1, 82: 1, 38: 2, 58: 2, 102: 2, 26: 3, 50: 3, 54: 4, 86: 4, 114: 4, 106: 5, 98: 6, 34: 7, 62: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 43, 55, 63, 71, 87, 99, 111, 123, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 966, 'token_per_expert': {3: 256, 7: 265, 23: 22, 39: 76, 43: 23, 55: 11, 63: 17, 71: 136, 87: 33, 99: 25, 111: 29, 123: 27, 127: 46}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 28, 36, 64, 72, 76, 88, 112, 116], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 1057, 'token_per_expert': {0: 266, 4: 297, 16: 69, 20: 73, 24: 43, 28: 42, 36: 42, 64: 38, 72: 32, 76: 29, 88: 25, 112: 77, 116: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 29, 33, 49, 61, 73, 93, 101, 117, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 21, 'token_total': 1041, 'token_per_expert': {1: 256, 5: 280, 9: 35, 13: 39, 29: 19, 33: 60, 49: 100, 61: 22, 73: 21, 93: 29, 101: 134, 117: 31, 125: 15}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 42, 46, 70, 74, 94, 118, 126], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 27, 'token_total': 759, 'token_per_expert': {2: 305, 6: 263, 14: 10, 18: 9, 22: 20, 42: 32, 46: 17, 70: 25, 74: 20, 94: 25, 118: 12, 126: 21}}
INFO 05-06 11:49:04.121189.121189 lmp.py:1836] [layer_moe_fused] layer=5 prefix: 0.430ms alloc: 0.383ms
INFO 05-06 11:49:04.121571.121571 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.626678466796875e-05 seconds
INFO 05-06 11:49:04.122996.122996 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008118152618408203s
INFO 05-06 11:49:04.123625.123625 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005409717559814453 seconds
DEBUG 05-06 11:49:04.123493.123493 cuda_h.py:27] end moe_cpu_prep_submit cost 0.926 ms
INFO 05-06 11:49:04.138778.138778 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014729738235473633s
DEBUG 05-06 11:49:04.138152.138152 cuda_h.py:27] end moe_wait_copy_tasks cost 14.844 ms
DEBUG 05-06 11:49:04.142074.142074 cuda_h.py:27] end moe_vllm_forward cost 3.611 ms
DEBUG 05-06 11:49:04.142218.142218 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 11:49:04.142064.142064 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.143212.143212 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.326ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.143750.143750 cuda_h.py:27] end *layer_moe_fused cost 22.720 ms
DEBUG 05-06 11:49:04.148518.148518 cuda_h.py:27] end prefill_merge_scale cost 5.459 ms
DEBUG 05-06 11:49:04.149600.149600 cuda_h.py:27] end prefill_layer cost 35.560 ms
DEBUG 05-06 11:49:04.149838.149838 lmp.py:1391] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 11:49:04.149746.149746 lmp.py:1347] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 11:49:04.149076.149076 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 11:49:04.151503.151503 cuda_h.py:27] end prefill_attn cost 1.794 ms
DEBUG 05-06 11:49:04.151240.151240 cuda_h.py:27] end prefill_ffn_prep cost 0.395 ms
DEBUG 05-06 11:49:04.152811.152811 cuda_h.py:27] end prefill_gate cost 0.417 ms
experts_cpu_alloc {'expert_ids': [31, 47, 83, 19, 59, 67, 111, 15, 11, 91, 127, 43, 51, 103, 8, 20, 52, 72, 92, 112, 124, 120, 16, 40, 76, 60, 80, 17, 21, 33, 97, 109, 81, 101, 49, 73, 89, 125, 37, 41, 57, 85, 22, 114, 38, 82, 126, 18, 30, 42, 110, 74, 14, 10, 58, 70, 46], 'token_total': 264, 'token_per_expert': {31: 1, 47: 1, 83: 1, 19: 2, 59: 2, 67: 2, 111: 2, 15: 4, 11: 5, 91: 6, 127: 6, 43: 9, 51: 11, 103: 11, 8: 1, 20: 1, 52: 1, 72: 1, 92: 1, 112: 1, 124: 1, 120: 2, 16: 3, 40: 3, 76: 4, 60: 5, 80: 5, 17: 1, 21: 1, 33: 1, 97: 1, 109: 1, 81: 2, 101: 2, 49: 3, 73: 5, 89: 5, 125: 6, 37: 7, 41: 8, 57: 9, 85: 9, 22: 1, 114: 1, 38: 2, 82: 3, 126: 4, 18: 6, 30: 6, 42: 6, 110: 6, 74: 8, 14: 9, 10: 10, 58: 12, 70: 15, 46: 22}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 35, 71, 75, 79, 87, 95, 99, 107, 115, 119, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 961, 'token_per_expert': {3: 256, 7: 257, 23: 48, 27: 14, 35: 44, 71: 13, 75: 13, 79: 20, 87: 42, 95: 15, 99: 129, 107: 17, 115: 50, 119: 30, 123: 13}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 28, 32, 36, 44, 56, 64, 68, 96, 104, 108, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 916, 'token_per_expert': {0: 265, 4: 258, 24: 17, 28: 12, 32: 14, 36: 26, 44: 14, 56: 11, 64: 50, 68: 141, 96: 21, 104: 17, 108: 60, 116: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 53, 65, 69, 77, 93, 105, 113, 117, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 988, 'token_per_expert': {1: 271, 5: 268, 9: 25, 13: 40, 25: 114, 53: 44, 65: 52, 69: 18, 77: 10, 93: 82, 105: 9, 113: 14, 117: 17, 121: 24}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 34, 50, 62, 78, 86, 90, 94, 98, 102, 106, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 967, 'token_per_expert': {2: 270, 6: 267, 26: 39, 34: 33, 50: 22, 62: 24, 78: 22, 86: 59, 90: 43, 94: 32, 98: 32, 102: 32, 106: 66, 122: 26}}
INFO 05-06 11:49:04.154106.154106 lmp.py:1836] [layer_moe_fused] layer=6 prefix: 0.423ms alloc: 0.411ms
INFO 05-06 11:49:04.154111.154111 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.984306335449219e-05 seconds
INFO 05-06 11:49:04.155996.155996 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007417201995849609s
INFO 05-06 11:49:04.155341.155341 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00054168701171875 seconds
DEBUG 05-06 11:49:04.156325.156325 cuda_h.py:27] end moe_cpu_prep_submit cost 0.833 ms
INFO 05-06 11:49:04.170747.170747 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014312505722045898s
DEBUG 05-06 11:49:04.170658.170658 cuda_h.py:27] end moe_wait_copy_tasks cost 14.438 ms
DEBUG 05-06 11:49:04.174072.174072 cuda_h.py:27] end moe_vllm_forward cost 3.686 ms
DEBUG 05-06 11:49:04.174985.174985 cuda_h.py:27] end moe_cpu_merge cost 0.060 ms
DEBUG 05-06 11:49:04.175996.175996 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:49:04.175721.175721 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.397ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.175345.175345 cuda_h.py:27] end *layer_moe_fused cost 22.433 ms
DEBUG 05-06 11:49:04.180829.180829 cuda_h.py:27] end prefill_merge_scale cost 4.299 ms
DEBUG 05-06 11:49:04.180435.180435 cuda_h.py:27] end prefill_layer cost 30.843 ms
DEBUG 05-06 11:49:04.180990.180990 lmp.py:1391] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 11:49:04.180468.180468 lmp.py:1347] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 11:49:04.180659.180659 cuda_h.py:27] end prefill_ln cost 0.211 ms
DEBUG 05-06 11:49:04.182934.182934 cuda_h.py:27] end prefill_attn cost 1.820 ms
DEBUG 05-06 11:49:04.183221.183221 cuda_h.py:27] end prefill_ffn_prep cost 0.379 ms
DEBUG 05-06 11:49:04.184926.184926 cuda_h.py:27] end prefill_gate cost 0.438 ms
experts_cpu_alloc {'expert_ids': [27, 35, 67, 119, 15, 63, 127, 55, 107, 51, 95, 23, 87, 83, 100, 32, 88, 92, 116, 8, 68, 16, 112, 64, 80, 72, 104, 73, 49, 77, 109, 37, 41, 45, 101, 17, 21, 117, 125, 9, 25, 13, 30, 50, 62, 66, 38, 54, 94, 26, 82, 78, 122, 98, 126, 118], 'token_total': 328, 'token_per_expert': {27: 1, 35: 1, 67: 1, 119: 1, 15: 2, 63: 3, 127: 3, 55: 4, 107: 4, 51: 6, 95: 7, 23: 8, 87: 8, 83: 9, 100: 1, 32: 2, 88: 4, 92: 4, 116: 5, 8: 6, 68: 8, 16: 9, 112: 9, 64: 11, 80: 15, 72: 17, 104: 21, 73: 1, 49: 2, 77: 2, 109: 2, 37: 3, 41: 5, 45: 5, 101: 5, 17: 6, 21: 7, 117: 7, 125: 8, 9: 11, 25: 11, 13: 14, 30: 2, 50: 2, 62: 2, 66: 2, 38: 3, 54: 4, 94: 4, 26: 5, 82: 5, 78: 6, 122: 7, 98: 8, 126: 9, 118: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 43, 47, 59, 71, 79, 91, 99, 103, 111, 115, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 791, 'token_per_expert': {3: 256, 7: 272, 19: 13, 43: 14, 47: 13, 59: 13, 71: 22, 79: 32, 91: 96, 99: 14, 103: 18, 111: 9, 115: 9, 123: 10}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 28, 44, 48, 52, 56, 60, 84, 96, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 988, 'token_per_expert': {0: 258, 4: 286, 12: 82, 20: 22, 28: 34, 44: 30, 48: 23, 52: 34, 56: 32, 60: 26, 84: 47, 96: 24, 108: 49, 120: 41}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 33, 53, 57, 61, 65, 69, 85, 97, 105, 113, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1056, 'token_per_expert': {1: 256, 5: 278, 29: 50, 33: 31, 53: 34, 57: 23, 61: 17, 65: 52, 69: 42, 85: 35, 97: 131, 105: 14, 113: 26, 121: 67}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 34, 42, 70, 86, 90, 106, 110, 114], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 933, 'token_per_expert': {2: 256, 6: 270, 10: 51, 14: 34, 18: 16, 22: 20, 34: 41, 42: 41, 70: 52, 86: 26, 90: 41, 106: 15, 110: 37, 114: 33}}
INFO 05-06 11:49:04.185797.185797 lmp.py:1836] [layer_moe_fused] layer=7 prefix: 0.425ms alloc: 0.410ms
INFO 05-06 11:49:04.185133.185133 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.817413330078125e-05 seconds
INFO 05-06 11:49:04.186286.186286 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007560253143310547s
INFO 05-06 11:49:04.186240.186240 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005354881286621094 seconds
DEBUG 05-06 11:49:04.187204.187204 cuda_h.py:27] end moe_cpu_prep_submit cost 1.237 ms
INFO 05-06 11:49:04.203503.203503 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01593613624572754s
DEBUG 05-06 11:49:04.204844.204844 cuda_h.py:27] end moe_wait_copy_tasks cost 16.060 ms
DEBUG 05-06 11:49:04.207646.207646 cuda_h.py:27] end moe_vllm_forward cost 3.585 ms
DEBUG 05-06 11:49:04.208836.208836 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:49:04.208185.208185 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.208956.208956 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.294ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.208991.208991 cuda_h.py:27] end *layer_moe_fused cost 24.439 ms
DEBUG 05-06 11:49:04.214365.214365 cuda_h.py:27] end prefill_merge_scale cost 5.763 ms
DEBUG 05-06 11:49:04.214448.214448 cuda_h.py:27] end prefill_layer cost 34.393 ms
DEBUG 05-06 11:49:04.214692.214692 lmp.py:1391] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 11:49:04.214932.214932 lmp.py:1347] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 11:49:04.215222.215222 cuda_h.py:27] end prefill_ln cost 0.202 ms
DEBUG 05-06 11:49:04.217596.217596 cuda_h.py:27] end prefill_attn cost 1.827 ms
DEBUG 05-06 11:49:04.217353.217353 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 11:49:04.218208.218208 cuda_h.py:27] end prefill_gate cost 0.417 ms
experts_cpu_alloc {'expert_ids': [23, 39, 43, 35, 91, 99, 119, 31, 127, 55, 63, 60, 72, 8, 48, 84, 104, 96, 124, 92, 116, 68, 64, 44, 101, 13, 89, 33, 85, 117, 37, 49, 29, 57, 17, 21, 41, 113, 26, 90, 18, 82, 86, 106, 118, 34, 22, 62, 74, 66, 42, 98, 10, 14], 'token_total': 290, 'token_per_expert': {23: 1, 39: 2, 43: 2, 35: 3, 91: 3, 99: 5, 119: 6, 31: 8, 127: 10, 55: 11, 63: 11, 60: 1, 72: 1, 8: 2, 48: 2, 84: 2, 104: 3, 96: 4, 124: 4, 92: 5, 116: 5, 68: 6, 64: 7, 44: 8, 101: 2, 13: 3, 89: 3, 33: 5, 85: 5, 117: 5, 37: 6, 49: 6, 29: 7, 57: 7, 17: 8, 21: 8, 41: 9, 113: 11, 26: 1, 90: 1, 18: 2, 82: 2, 86: 2, 106: 2, 118: 2, 34: 4, 22: 5, 62: 6, 74: 6, 66: 8, 42: 10, 98: 12, 10: 15, 14: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 47, 51, 71, 75, 87, 103, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 1024, 'token_per_expert': {3: 279, 7: 256, 11: 14, 15: 22, 19: 43, 27: 18, 47: 12, 51: 96, 71: 38, 75: 51, 87: 63, 103: 92, 111: 15, 123: 25}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 20, 28, 32, 36, 52, 56, 76, 80, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 858, 'token_per_expert': {0: 257, 4: 267, 12: 23, 16: 10, 20: 9, 28: 70, 32: 39, 36: 18, 52: 17, 56: 36, 76: 16, 80: 25, 108: 8, 120: 63}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 45, 53, 61, 65, 69, 73, 77, 81, 93, 105, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 816, 'token_per_expert': {1: 259, 5: 273, 45: 12, 53: 21, 61: 13, 65: 25, 69: 17, 73: 53, 77: 17, 81: 16, 93: 12, 105: 40, 121: 28, 125: 30}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 50, 54, 58, 70, 102, 110, 114, 122, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 1108, 'token_per_expert': {2: 290, 6: 273, 38: 36, 46: 31, 50: 45, 54: 104, 58: 102, 70: 43, 102: 31, 110: 64, 114: 44, 122: 28, 126: 17}}
INFO 05-06 11:49:04.219928.219928 lmp.py:1836] [layer_moe_fused] layer=8 prefix: 0.425ms alloc: 0.442ms
INFO 05-06 11:49:04.219125.219125 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-06 11:49:04.220089.220089 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007166862487792969s
INFO 05-06 11:49:04.221950.221950 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005359649658203125 seconds
DEBUG 05-06 11:49:04.221469.221469 cuda_h.py:27] end moe_cpu_prep_submit cost 0.909 ms
INFO 05-06 11:49:04.237939.237939 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01491236686706543s
DEBUG 05-06 11:49:04.237982.237982 cuda_h.py:27] end moe_wait_copy_tasks cost 15.029 ms
DEBUG 05-06 11:49:04.241263.241263 cuda_h.py:27] end moe_vllm_forward cost 3.494 ms
DEBUG 05-06 11:49:04.241930.241930 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:49:04.241671.241671 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:49:04.241011.241011 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.198ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.241596.241596 cuda_h.py:27] end *layer_moe_fused cost 22.993 ms
DEBUG 05-06 11:49:04.248782.248782 cuda_h.py:27] end prefill_merge_scale cost 6.471 ms
DEBUG 05-06 11:49:04.248818.248818 cuda_h.py:27] end prefill_layer cost 33.587 ms
DEBUG 05-06 11:49:04.248461.248461 lmp.py:1391] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 11:49:04.248224.248224 lmp.py:1347] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 11:49:04.249938.249938 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:49:04.250826.250826 cuda_h.py:27] end prefill_attn cost 1.750 ms
DEBUG 05-06 11:49:04.251491.251491 cuda_h.py:27] end prefill_ffn_prep cost 0.378 ms
DEBUG 05-06 11:49:04.252136.252136 cuda_h.py:27] end prefill_gate cost 0.424 ms
experts_cpu_alloc {'expert_ids': [11, 31, 63, 79, 115, 119, 67, 19, 27, 39, 51, 112, 52, 84, 96, 8, 44, 20, 116, 120, 124, 24, 68, 80, 29, 33, 41, 105, 97, 117, 77, 113, 37, 73, 9, 45, 17, 26, 50, 58, 66, 90, 122, 10, 34, 98, 114, 82, 42, 62], 'token_total': 268, 'token_per_expert': {11: 1, 31: 1, 63: 1, 79: 1, 115: 1, 119: 3, 67: 5, 19: 8, 27: 11, 39: 11, 51: 15, 112: 1, 52: 2, 84: 2, 96: 2, 8: 4, 44: 4, 20: 5, 116: 6, 120: 7, 124: 8, 24: 9, 68: 9, 80: 10, 29: 1, 33: 1, 41: 1, 105: 3, 97: 4, 117: 6, 77: 7, 113: 7, 37: 9, 73: 9, 9: 12, 45: 14, 17: 15, 26: 1, 50: 1, 58: 1, 66: 1, 90: 3, 122: 3, 10: 4, 34: 4, 98: 4, 114: 4, 82: 7, 42: 8, 62: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 23, 43, 71, 75, 83, 95, 99, 103, 111, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 1001, 'token_per_expert': {3: 266, 7: 264, 15: 18, 23: 32, 43: 57, 71: 15, 75: 53, 83: 17, 95: 144, 99: 24, 103: 70, 111: 18, 127: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 32, 36, 40, 48, 56, 72, 76, 88, 92], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 920, 'token_per_expert': {0: 262, 4: 283, 12: 80, 16: 69, 32: 28, 36: 24, 40: 12, 48: 38, 56: 47, 72: 20, 76: 18, 88: 12, 92: 27}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 57, 61, 69, 81, 89, 93, 101, 125], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 25, 'token_total': 883, 'token_per_expert': {1: 270, 5: 261, 13: 19, 21: 16, 57: 26, 61: 15, 69: 35, 81: 53, 89: 18, 93: 71, 101: 79, 125: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 30, 38, 46, 54, 70, 74, 86, 102, 106], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 25, 'token_total': 1024, 'token_per_expert': {2: 257, 6: 257, 22: 17, 30: 17, 38: 20, 46: 101, 54: 25, 70: 102, 74: 66, 86: 16, 102: 42, 106: 104}}
INFO 05-06 11:49:04.253943.253943 lmp.py:1836] [layer_moe_fused] layer=9 prefix: 0.499ms alloc: 0.388ms
INFO 05-06 11:49:04.253657.253657 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.461143493652344e-05 seconds
INFO 05-06 11:49:04.254603.254603 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009801387786865234s
INFO 05-06 11:49:04.255864.255864 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.000743865966796875 seconds
DEBUG 05-06 11:49:04.256571.256571 cuda_h.py:27] end moe_cpu_prep_submit cost 0.913 ms
INFO 05-06 11:49:04.271572.271572 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014992475509643555s
DEBUG 05-06 11:49:04.271424.271424 cuda_h.py:27] end moe_wait_copy_tasks cost 15.104 ms
DEBUG 05-06 11:49:04.275578.275578 cuda_h.py:27] end moe_vllm_forward cost 3.474 ms
DEBUG 05-06 11:49:04.275192.275192 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:49:04.275946.275946 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.275810.275810 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.181ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.275415.275415 cuda_h.py:27] end *layer_moe_fused cost 23.118 ms
DEBUG 05-06 11:49:04.281364.281364 cuda_h.py:27] end prefill_merge_scale cost 5.487 ms
DEBUG 05-06 11:49:04.281208.281208 cuda_h.py:27] end prefill_layer cost 32.746 ms
DEBUG 05-06 11:49:04.281317.281317 lmp.py:1391] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 11:49:04.281318.281318 lmp.py:1347] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 11:49:04.282438.282438 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:49:04.284236.284236 cuda_h.py:27] end prefill_attn cost 1.824 ms
DEBUG 05-06 11:49:04.284099.284099 cuda_h.py:27] end prefill_ffn_prep cost 0.383 ms
DEBUG 05-06 11:49:04.285093.285093 cuda_h.py:27] end prefill_gate cost 0.414 ms
experts_cpu_alloc {'expert_ids': [51, 111, 123, 35, 103, 107, 59, 11, 15, 67, 83, 12, 32, 52, 124, 40, 56, 120, 64, 44, 28, 100, 112, 68, 53, 65, 77, 33, 37, 61, 29, 109, 9, 25, 97, 69, 73, 117, 93, 121, 38, 66, 70, 102, 98, 26, 34, 78, 50, 94, 10, 90, 46], 'token_total': 247, 'token_per_expert': {51: 1, 111: 1, 123: 1, 35: 2, 103: 2, 107: 2, 59: 3, 11: 4, 15: 4, 67: 4, 83: 4, 12: 1, 32: 1, 52: 1, 124: 1, 40: 3, 56: 3, 120: 3, 64: 5, 44: 6, 28: 10, 100: 10, 112: 10, 68: 12, 53: 1, 65: 1, 77: 1, 33: 2, 37: 2, 61: 2, 29: 3, 109: 3, 9: 4, 25: 4, 97: 4, 69: 7, 73: 8, 117: 8, 93: 10, 121: 10, 38: 1, 66: 1, 70: 1, 102: 1, 98: 2, 26: 3, 34: 6, 78: 7, 50: 8, 94: 9, 10: 10, 90: 16, 46: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 39, 43, 47, 63, 71, 75, 79, 99, 115, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 777, 'token_per_expert': {3: 259, 7: 270, 19: 10, 31: 18, 39: 13, 43: 12, 47: 23, 63: 16, 71: 34, 75: 26, 79: 8, 99: 12, 115: 50, 127: 26}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 60, 72, 76, 80, 84, 88, 92, 108], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 1143, 'token_per_expert': {0: 316, 4: 263, 8: 87, 16: 29, 20: 17, 60: 74, 72: 31, 76: 118, 80: 77, 84: 20, 88: 55, 92: 29, 108: 27}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 41, 49, 57, 81, 85, 89, 105, 113, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 951, 'token_per_expert': {1: 345, 5: 269, 13: 22, 21: 34, 41: 32, 49: 23, 57: 42, 81: 69, 85: 31, 89: 12, 105: 16, 113: 26, 125: 30}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 42, 54, 58, 62, 74, 82, 86, 106, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 978, 'token_per_expert': {2: 256, 6: 257, 14: 48, 18: 39, 42: 53, 54: 20, 58: 36, 62: 59, 74: 55, 82: 19, 86: 65, 106: 47, 126: 24}}
INFO 05-06 11:49:04.286725.286725 lmp.py:1836] [layer_moe_fused] layer=10 prefix: 0.416ms alloc: 0.386ms
INFO 05-06 11:49:04.286154.286154 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.698204040527344e-05 seconds
INFO 05-06 11:49:04.287441.287441 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007481575012207031s
INFO 05-06 11:49:04.288288.288288 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005261898040771484 seconds
DEBUG 05-06 11:49:04.288817.288817 cuda_h.py:27] end moe_cpu_prep_submit cost 0.881 ms
INFO 05-06 11:49:04.304812.304812 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015207052230834961s
DEBUG 05-06 11:49:04.304531.304531 cuda_h.py:27] end moe_wait_copy_tasks cost 15.327 ms
DEBUG 05-06 11:49:04.308589.308589 cuda_h.py:27] end moe_vllm_forward cost 3.540 ms
DEBUG 05-06 11:49:04.308826.308826 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 11:49:04.308408.308408 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.308987.308987 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.265ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.309840.309840 cuda_h.py:27] end *layer_moe_fused cost 23.221 ms
DEBUG 05-06 11:49:04.313528.313528 cuda_h.py:27] end prefill_merge_scale cost 4.661 ms
DEBUG 05-06 11:49:04.313240.313240 cuda_h.py:27] end prefill_layer cost 32.068 ms
DEBUG 05-06 11:49:04.314338.314338 lmp.py:1391] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 11:49:04.314578.314578 lmp.py:1347] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 11:49:04.314146.314146 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 11:49:04.316972.316972 cuda_h.py:27] end prefill_attn cost 1.845 ms
DEBUG 05-06 11:49:04.316622.316622 cuda_h.py:27] end prefill_ffn_prep cost 0.369 ms
DEBUG 05-06 11:49:04.317020.317020 cuda_h.py:27] end prefill_gate cost 0.401 ms
experts_cpu_alloc {'expert_ids': [35, 47, 127, 63, 115, 59, 51, 91, 71, 43, 39, 123, 12, 64, 84, 52, 72, 80, 8, 40, 48, 28, 120, 44, 36, 124, 21, 53, 65, 85, 9, 13, 33, 97, 125, 117, 121, 29, 61, 74, 94, 106, 114, 22, 34, 58, 98, 110, 122, 126, 42, 50, 62], 'token_total': 211, 'token_per_expert': {35: 1, 47: 1, 127: 1, 63: 2, 115: 2, 59: 3, 51: 6, 91: 6, 71: 7, 43: 9, 39: 10, 123: 10, 12: 2, 64: 2, 84: 2, 52: 3, 72: 3, 80: 3, 8: 4, 40: 4, 48: 4, 28: 5, 120: 5, 44: 6, 36: 7, 124: 10, 21: 1, 53: 1, 65: 1, 85: 1, 9: 2, 13: 2, 33: 2, 97: 2, 125: 3, 117: 6, 121: 11, 29: 14, 61: 14, 74: 1, 94: 1, 106: 1, 114: 1, 22: 2, 34: 2, 58: 2, 98: 2, 110: 2, 122: 2, 126: 2, 42: 5, 50: 5, 62: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 31, 67, 79, 83, 87, 99, 111, 119], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 1051, 'token_per_expert': {3: 259, 7: 307, 11: 12, 19: 16, 23: 51, 27: 11, 31: 21, 67: 50, 79: 76, 83: 77, 87: 79, 99: 23, 111: 49, 119: 20}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 32, 56, 68, 76, 92, 100, 108, 112, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1057, 'token_per_expert': {0: 259, 4: 258, 16: 81, 20: 29, 24: 30, 32: 36, 56: 102, 68: 47, 76: 30, 92: 86, 100: 23, 108: 32, 112: 33, 116: 11}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 25, 37, 49, 57, 69, 77, 81, 89, 93, 113], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 952, 'token_per_expert': {1: 265, 5: 264, 17: 45, 25: 16, 37: 18, 49: 52, 57: 19, 69: 24, 77: 28, 81: 78, 89: 18, 93: 53, 113: 72}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 30, 38, 46, 54, 66, 70, 82, 102, 118], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 825, 'token_per_expert': {2: 282, 6: 310, 10: 27, 18: 9, 30: 26, 38: 15, 46: 10, 54: 7, 66: 19, 70: 5, 82: 6, 102: 104, 118: 5}}
INFO 05-06 11:49:04.318320.318320 lmp.py:1836] [layer_moe_fused] layer=11 prefix: 0.416ms alloc: 0.391ms
INFO 05-06 11:49:04.319895.319895 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.031990051269531e-05 seconds
INFO 05-06 11:49:04.320259.320259 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007684230804443359s
INFO 05-06 11:49:04.320577.320577 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005238056182861328 seconds
DEBUG 05-06 11:49:04.320810.320810 cuda_h.py:27] end moe_cpu_prep_submit cost 0.761 ms
INFO 05-06 11:49:04.335398.335398 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014271259307861328s
DEBUG 05-06 11:49:04.335825.335825 cuda_h.py:27] end moe_wait_copy_tasks cost 14.399 ms
DEBUG 05-06 11:49:04.339613.339613 cuda_h.py:27] end moe_vllm_forward cost 3.578 ms
DEBUG 05-06 11:49:04.339134.339134 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 11:49:04.339172.339172 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.339433.339433 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.274ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.340564.340564 cuda_h.py:27] end *layer_moe_fused cost 22.127 ms
DEBUG 05-06 11:49:04.343332.343332 cuda_h.py:27] end prefill_merge_scale cost 3.703 ms
DEBUG 05-06 11:49:04.344077.344077 cuda_h.py:27] end prefill_layer cost 29.931 ms
DEBUG 05-06 11:49:04.344734.344734 lmp.py:1391] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 11:49:04.344974.344974 lmp.py:1347] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 11:49:04.344469.344469 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:49:04.346166.346166 cuda_h.py:27] end prefill_attn cost 1.747 ms
DEBUG 05-06 11:49:04.346055.346055 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 11:49:04.348316.348316 cuda_h.py:27] end prefill_gate cost 0.478 ms
experts_cpu_alloc {'expert_ids': [47, 59, 67, 31, 111, 107, 119, 127, 63, 123, 103, 8, 120, 20, 24, 32, 104, 88, 12, 40, 36, 80, 37, 81, 113, 125, 13, 17, 33, 65, 105, 89, 77, 101, 18, 94, 102, 58, 70, 38, 90, 22, 98, 34, 106], 'token_total': 214, 'token_per_expert': {47: 1, 59: 1, 67: 1, 31: 2, 111: 2, 107: 3, 119: 3, 127: 3, 63: 5, 123: 5, 103: 7, 8: 1, 120: 1, 20: 3, 24: 3, 32: 3, 104: 6, 88: 7, 12: 8, 40: 8, 36: 9, 80: 9, 37: 1, 81: 1, 113: 1, 125: 2, 13: 3, 17: 3, 33: 3, 65: 3, 105: 3, 89: 5, 77: 6, 101: 9, 18: 1, 94: 2, 102: 3, 58: 4, 70: 4, 38: 5, 90: 5, 22: 6, 98: 7, 34: 12, 106: 34}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 35, 39, 71, 79, 91, 95, 115], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 23, 'token_total': 943, 'token_per_expert': {3: 273, 7: 256, 15: 72, 19: 36, 23: 25, 35: 15, 39: 88, 71: 89, 79: 8, 91: 32, 95: 20, 115: 29}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 68, 76, 84, 92, 100, 108, 112, 116, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 22, 'token_total': 749, 'token_per_expert': {0: 256, 4: 257, 68: 15, 76: 10, 84: 14, 92: 18, 100: 11, 108: 81, 112: 12, 116: 64, 124: 11}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25, 45, 49, 53, 73, 85, 97, 117], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 23, 'token_total': 996, 'token_per_expert': {1: 262, 5: 291, 21: 103, 25: 32, 45: 63, 49: 29, 53: 114, 73: 37, 85: 15, 97: 33, 117: 17}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 46, 50, 74, 78, 82, 86, 110, 114, 118], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 22, 'token_total': 1194, 'token_per_expert': {2: 256, 6: 294, 46: 44, 50: 72, 74: 80, 78: 164, 82: 45, 86: 72, 110: 56, 114: 62, 118: 49}}
INFO 05-06 11:49:04.349595.349595 lmp.py:1836] [layer_moe_fused] layer=12 prefix: 0.410ms alloc: 0.343ms
INFO 05-06 11:49:04.349632.349632 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.125999450683594e-05 seconds
INFO 05-06 11:49:04.350245.350245 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000774383544921875s
INFO 05-06 11:49:04.350271.350271 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005190372467041016 seconds
DEBUG 05-06 11:49:04.351547.351547 cuda_h.py:27] end moe_cpu_prep_submit cost 1.036 ms
INFO 05-06 11:49:04.367854.367854 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015561580657958984s
DEBUG 05-06 11:49:04.367466.367466 cuda_h.py:27] end moe_wait_copy_tasks cost 15.680 ms
DEBUG 05-06 11:49:04.371590.371590 cuda_h.py:27] end moe_vllm_forward cost 3.494 ms
DEBUG 05-06 11:49:04.371438.371438 cuda_h.py:27] end moe_cpu_merge cost 0.124 ms
DEBUG 05-06 11:49:04.371191.371191 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.371961.371961 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.248ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.372978.372978 cuda_h.py:27] end *layer_moe_fused cost 24.073 ms
DEBUG 05-06 11:49:04.378697.378697 cuda_h.py:27] end prefill_merge_scale cost 5.774 ms
DEBUG 05-06 11:49:04.378965.378965 cuda_h.py:27] end prefill_layer cost 33.945 ms
DEBUG 05-06 11:49:04.378912.378912 lmp.py:1391] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 11:49:04.378628.378628 lmp.py:1347] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 11:49:04.378217.378217 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:49:04.380140.380140 cuda_h.py:27] end prefill_attn cost 1.810 ms
DEBUG 05-06 11:49:04.381652.381652 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 11:49:04.382838.382838 cuda_h.py:27] end prefill_gate cost 0.416 ms
experts_cpu_alloc {'expert_ids': [107, 83, 123, 87, 27, 43, 47, 55, 67, 75, 11, 115, 99, 48, 56, 104, 8, 40, 92, 64, 96, 16, 28, 97, 105, 45, 53, 65, 73, 9, 57, 13, 117, 93, 101, 10, 62, 74, 94, 66, 70, 106, 90, 26, 122, 82, 46, 38, 42, 86], 'token_total': 236, 'token_per_expert': {107: 1, 83: 2, 123: 2, 87: 3, 27: 4, 43: 4, 47: 4, 55: 7, 67: 7, 75: 7, 11: 8, 115: 8, 99: 11, 48: 1, 56: 1, 104: 3, 8: 4, 40: 5, 92: 5, 64: 6, 96: 6, 16: 7, 28: 7, 97: 1, 105: 1, 45: 2, 53: 2, 65: 3, 73: 3, 9: 4, 57: 5, 13: 6, 117: 7, 93: 9, 101: 9, 10: 1, 62: 1, 74: 1, 94: 1, 66: 2, 70: 2, 106: 3, 90: 4, 26: 5, 122: 5, 82: 6, 46: 8, 38: 9, 42: 10, 86: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 31, 39, 51, 59, 63, 71, 79, 91, 103, 119], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 1040, 'token_per_expert': {3: 271, 7: 256, 15: 20, 31: 108, 39: 22, 51: 27, 59: 38, 63: 37, 71: 48, 79: 74, 91: 100, 103: 25, 119: 14}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 32, 52, 60, 68, 84, 100, 108, 116, 120, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 23, 'token_total': 852, 'token_per_expert': {0: 256, 4: 256, 20: 22, 32: 56, 52: 8, 60: 9, 68: 7, 84: 19, 100: 108, 108: 11, 116: 16, 120: 64, 124: 20}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 25, 33, 37, 41, 69, 81, 113, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 925, 'token_per_expert': {1: 281, 5: 256, 17: 50, 21: 27, 25: 38, 33: 24, 37: 49, 41: 13, 69: 23, 81: 51, 113: 25, 121: 69, 125: 19}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 34, 78, 98, 102, 110, 114, 118, 126], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 27, 'token_total': 1043, 'token_per_expert': {2: 263, 6: 302, 14: 49, 22: 29, 34: 22, 78: 54, 98: 33, 102: 35, 110: 89, 114: 103, 118: 32, 126: 32}}
INFO 05-06 11:49:04.383285.383285 lmp.py:1836] [layer_moe_fused] layer=13 prefix: 0.427ms alloc: 0.381ms
INFO 05-06 11:49:04.383375.383375 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.555152893066406e-05 seconds
INFO 05-06 11:49:04.384547.384547 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000782012939453125s
INFO 05-06 11:49:04.385732.385732 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005307197570800781 seconds
DEBUG 05-06 11:49:04.385651.385651 cuda_h.py:27] end moe_cpu_prep_submit cost 1.059 ms
INFO 05-06 11:49:04.402766.402766 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.016365528106689453s
DEBUG 05-06 11:49:04.402869.402869 cuda_h.py:27] end moe_wait_copy_tasks cost 16.492 ms
DEBUG 05-06 11:49:04.406483.406483 cuda_h.py:27] end moe_vllm_forward cost 3.522 ms
DEBUG 05-06 11:49:04.406819.406819 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 11:49:04.406326.406326 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.406759.406759 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.200ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.406723.406723 cuda_h.py:27] end *layer_moe_fused cost 24.379 ms
DEBUG 05-06 11:49:04.412821.412821 cuda_h.py:27] end prefill_merge_scale cost 5.458 ms
DEBUG 05-06 11:49:04.412374.412374 cuda_h.py:27] end prefill_layer cost 33.940 ms
DEBUG 05-06 11:49:04.412121.412121 lmp.py:1391] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 11:49:04.412884.412884 lmp.py:1347] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 11:49:04.413121.413121 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 11:49:04.415502.415502 cuda_h.py:27] end prefill_attn cost 2.216 ms
DEBUG 05-06 11:49:04.416969.416969 cuda_h.py:27] end prefill_ffn_prep cost 0.533 ms
DEBUG 05-06 11:49:04.417880.417880 cuda_h.py:27] end prefill_gate cost 0.499 ms
experts_cpu_alloc {'expert_ids': [51, 63, 91, 111, 15, 19, 71, 35, 23, 67, 43, 107, 83, 48, 56, 68, 96, 40, 108, 36, 44, 64, 116, 60, 8, 120, 16, 29, 33, 37, 41, 61, 73, 85, 17, 77, 101, 9, 81, 109, 125, 93, 21, 14, 46, 54, 70, 118, 98, 106, 126, 10, 58, 78, 102, 90, 110], 'token_total': 248, 'token_per_expert': {51: 1, 63: 1, 91: 1, 111: 1, 15: 2, 19: 2, 71: 3, 35: 4, 23: 5, 67: 5, 43: 7, 107: 7, 83: 11, 48: 2, 56: 2, 68: 3, 96: 3, 40: 5, 108: 5, 36: 6, 44: 6, 64: 6, 116: 7, 60: 8, 8: 9, 120: 9, 16: 10, 29: 1, 33: 1, 37: 1, 41: 1, 61: 1, 73: 1, 85: 1, 17: 2, 77: 2, 101: 2, 9: 4, 81: 4, 109: 7, 125: 8, 93: 9, 21: 10, 14: 1, 46: 1, 54: 1, 70: 2, 118: 3, 98: 4, 106: 4, 126: 4, 10: 5, 58: 7, 78: 7, 102: 7, 90: 8, 110: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 31, 39, 47, 59, 75, 95, 99, 103, 115, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 1127, 'token_per_expert': {3: 260, 7: 269, 11: 17, 31: 29, 39: 54, 47: 47, 59: 32, 75: 49, 95: 33, 99: 36, 103: 42, 115: 151, 119: 64, 123: 26, 127: 18}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 24, 28, 32, 52, 72, 76, 80, 92, 100, 104, 112, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 821, 'token_per_expert': {0: 260, 4: 256, 12: 24, 24: 11, 28: 13, 32: 16, 52: 13, 72: 17, 76: 16, 80: 30, 92: 11, 100: 41, 104: 18, 112: 11, 124: 84}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 45, 53, 57, 65, 89, 97, 105, 113, 117, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 922, 'token_per_expert': {1: 257, 5: 264, 13: 22, 25: 12, 45: 11, 53: 22, 57: 13, 65: 54, 89: 18, 97: 50, 105: 15, 113: 39, 117: 41, 121: 104}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 30, 34, 38, 42, 50, 62, 66, 74, 86, 114, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 978, 'token_per_expert': {2: 290, 6: 257, 26: 65, 30: 27, 34: 16, 38: 13, 42: 28, 50: 64, 62: 33, 66: 56, 74: 17, 86: 70, 114: 11, 122: 31}}
INFO 05-06 11:49:04.418018.418018 lmp.py:1836] [layer_moe_fused] layer=14 prefix: 0.443ms alloc: 0.426ms
INFO 05-06 11:49:04.418115.418115 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.079673767089844e-05 seconds
INFO 05-06 11:49:04.419844.419844 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008444786071777344s
INFO 05-06 11:49:04.420163.420163 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00054168701171875 seconds
DEBUG 05-06 11:49:04.420168.420168 cuda_h.py:27] end moe_cpu_prep_submit cost 1.108 ms
INFO 05-06 11:49:04.437827.437827 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.016623735427856445s
DEBUG 05-06 11:49:04.437613.437613 cuda_h.py:27] end moe_wait_copy_tasks cost 16.765 ms
DEBUG 05-06 11:49:04.441421.441421 cuda_h.py:27] end moe_vllm_forward cost 3.583 ms
DEBUG 05-06 11:49:04.441664.441664 cuda_h.py:27] end moe_cpu_merge cost 0.059 ms
DEBUG 05-06 11:49:04.442417.442417 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.442472.442472 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.283ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.442383.442383 cuda_h.py:27] end *layer_moe_fused cost 24.846 ms
DEBUG 05-06 11:49:04.447210.447210 cuda_h.py:27] end prefill_merge_scale cost 4.660 ms
DEBUG 05-06 11:49:04.447200.447200 cuda_h.py:27] end prefill_layer cost 34.747 ms
DEBUG 05-06 11:49:04.447385.447385 lmp.py:1391] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 11:49:04.447578.447578 lmp.py:1347] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 11:49:04.448490.448490 cuda_h.py:27] end prefill_ln cost 0.217 ms
DEBUG 05-06 11:49:04.449320.449320 cuda_h.py:27] end prefill_attn cost 1.773 ms
DEBUG 05-06 11:49:04.450170.450170 cuda_h.py:27] end prefill_ffn_prep cost 0.373 ms
DEBUG 05-06 11:49:04.451615.451615 cuda_h.py:27] end prefill_gate cost 0.418 ms
experts_cpu_alloc {'expert_ids': [79, 35, 11, 19, 55, 111, 127, 43, 115, 119, 107, 31, 59, 32, 56, 100, 28, 80, 96, 36, 48, 116, 120, 16, 24, 45, 57, 105, 25, 117, 33, 121, 29, 77, 13, 41, 69, 17, 97, 73, 94, 126, 22, 54, 118, 18, 82, 38, 78, 34, 58, 46], 'token_total': 349, 'token_per_expert': {79: 1, 35: 2, 11: 3, 19: 3, 55: 3, 111: 3, 127: 3, 43: 10, 115: 10, 119: 10, 107: 11, 31: 13, 59: 16, 32: 1, 56: 2, 100: 2, 28: 5, 80: 7, 96: 7, 36: 9, 48: 10, 116: 13, 120: 13, 16: 14, 24: 17, 45: 1, 57: 2, 105: 2, 25: 4, 117: 4, 33: 6, 121: 6, 29: 7, 77: 7, 13: 8, 41: 11, 69: 11, 17: 12, 97: 12, 73: 14, 94: 1, 126: 1, 22: 2, 54: 2, 118: 3, 18: 5, 82: 5, 38: 6, 78: 6, 34: 7, 58: 7, 46: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 47, 51, 63, 71, 75, 83, 91, 95, 99, 103], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 987, 'token_per_expert': {3: 257, 7: 276, 23: 34, 39: 32, 47: 17, 51: 28, 63: 19, 71: 38, 75: 49, 83: 76, 91: 93, 95: 19, 99: 30, 103: 19}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 52, 64, 68, 72, 76, 84, 88, 104, 108, 112, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 1015, 'token_per_expert': {0: 264, 4: 261, 52: 35, 64: 26, 68: 86, 72: 20, 76: 93, 84: 22, 88: 21, 104: 28, 108: 50, 112: 84, 124: 25}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 37, 65, 81, 85, 93, 101, 109, 113, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 898, 'token_per_expert': {1: 268, 5: 279, 9: 34, 21: 26, 37: 24, 65: 65, 81: 25, 85: 22, 93: 21, 101: 31, 109: 70, 113: 14, 125: 19}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 30, 42, 66, 70, 86, 90, 98, 102, 114], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 847, 'token_per_expert': {2: 285, 6: 257, 10: 47, 14: 11, 30: 45, 42: 15, 66: 37, 70: 24, 86: 9, 90: 67, 98: 30, 102: 9, 114: 11}}
INFO 05-06 11:49:04.452785.452785 lmp.py:1836] [layer_moe_fused] layer=15 prefix: 0.456ms alloc: 0.388ms
INFO 05-06 11:49:04.452312.452312 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.8650970458984375e-05 seconds
INFO 05-06 11:49:04.453520.453520 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008425712585449219s
INFO 05-06 11:49:04.454527.454527 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005390644073486328 seconds
DEBUG 05-06 11:49:04.454369.454369 cuda_h.py:27] end moe_cpu_prep_submit cost 1.154 ms
INFO 05-06 11:49:04.464780.464780 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.009816646575927734s
DEBUG 05-06 11:49:04.465115.465115 cuda_h.py:27] end moe_wait_copy_tasks cost 9.937 ms
DEBUG 05-06 11:49:04.469690.469690 cuda_h.py:27] end moe_vllm_forward cost 3.573 ms
DEBUG 05-06 11:49:04.469073.469073 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 11:49:04.469681.469681 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.469021.469021 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.283ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.469733.469733 cuda_h.py:27] end *layer_moe_fused cost 18.296 ms
DEBUG 05-06 11:49:04.474648.474648 cuda_h.py:27] end prefill_merge_scale cost 4.725 ms
DEBUG 05-06 11:49:04.474340.474340 cuda_h.py:27] end prefill_layer cost 27.105 ms
DEBUG 05-06 11:49:04.474233.474233 lmp.py:1391] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 11:49:04.474996.474996 lmp.py:1347] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 11:49:04.475239.475239 cuda_h.py:27] end prefill_ln cost 0.210 ms
DEBUG 05-06 11:49:04.477381.477381 cuda_h.py:27] end prefill_attn cost 1.794 ms
DEBUG 05-06 11:49:04.477992.477992 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 11:49:04.478324.478324 cuda_h.py:27] end prefill_gate cost 0.420 ms
experts_cpu_alloc {'expert_ids': [11, 103, 51, 43, 59, 71, 99, 111, 123, 91, 119, 28, 36, 84, 88, 24, 120, 56, 64, 104, 40, 92, 116, 80, 108, 72, 29, 69, 73, 101, 49, 53, 89, 33, 81, 109, 61, 13, 37, 9, 57, 113, 46, 50, 106, 122, 38, 34, 98, 10, 118, 82, 18, 62, 90, 102, 30, 22], 'token_total': 278, 'token_per_expert': {11: 1, 103: 1, 51: 2, 43: 3, 59: 3, 71: 3, 99: 3, 111: 4, 123: 5, 91: 6, 119: 7, 28: 1, 36: 1, 84: 1, 88: 2, 24: 3, 120: 3, 56: 4, 64: 5, 104: 5, 40: 10, 92: 10, 116: 12, 80: 13, 108: 15, 72: 18, 29: 1, 69: 1, 73: 1, 101: 1, 49: 2, 53: 2, 89: 2, 33: 4, 81: 4, 109: 4, 61: 5, 13: 6, 37: 6, 9: 7, 57: 7, 113: 7, 46: 1, 50: 1, 106: 1, 122: 1, 38: 2, 34: 3, 98: 3, 10: 4, 118: 4, 82: 5, 18: 6, 62: 6, 90: 7, 102: 7, 30: 11, 22: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 31, 55, 63, 67, 75, 79, 83, 87, 107, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 26, 'token_total': 930, 'token_per_expert': {3: 270, 7: 260, 15: 9, 19: 15, 23: 19, 31: 32, 55: 22, 63: 25, 67: 63, 75: 21, 79: 13, 83: 32, 87: 95, 107: 39, 127: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 32, 44, 48, 52, 68, 76, 96, 100, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1141, 'token_per_expert': {0: 281, 4: 281, 8: 38, 12: 31, 16: 95, 20: 19, 32: 115, 44: 36, 48: 20, 52: 98, 68: 25, 76: 21, 96: 24, 100: 28, 124: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 45, 65, 77, 85, 93, 97, 105, 117, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 806, 'token_per_expert': {1: 311, 5: 277, 17: 9, 21: 11, 45: 10, 65: 10, 77: 18, 85: 17, 93: 10, 97: 18, 105: 56, 117: 21, 121: 16, 125: 22}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 26, 42, 54, 58, 66, 70, 78, 86, 110, 114, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 941, 'token_per_expert': {2: 267, 6: 257, 14: 30, 26: 23, 42: 20, 54: 25, 58: 18, 66: 41, 70: 17, 78: 18, 86: 64, 110: 18, 114: 21, 126: 122}}
INFO 05-06 11:49:04.479017.479017 lmp.py:1836] [layer_moe_fused] layer=16 prefix: 0.419ms alloc: 0.428ms
INFO 05-06 11:49:04.479691.479691 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.103515625e-05 seconds
INFO 05-06 11:49:04.480942.480942 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008175373077392578s
INFO 05-06 11:49:04.481625.481625 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005300045013427734 seconds
DEBUG 05-06 11:49:04.482862.482862 cuda_h.py:27] end moe_cpu_prep_submit cost 1.081 ms
INFO 05-06 11:49:04.504771.504771 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.02212071418762207s
DEBUG 05-06 11:49:04.504828.504828 cuda_h.py:27] end moe_wait_copy_tasks cost 22.245 ms
DEBUG 05-06 11:49:04.508458.508458 cuda_h.py:27] end moe_vllm_forward cost 3.433 ms
DEBUG 05-06 11:49:04.508602.508602 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:49:04.508434.508434 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.508298.508298 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.101ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.509116.509116 cuda_h.py:27] end *layer_moe_fused cost 30.102 ms
DEBUG 05-06 11:49:04.513083.513083 cuda_h.py:27] end prefill_merge_scale cost 4.906 ms
DEBUG 05-06 11:49:04.514305.514305 cuda_h.py:27] end prefill_layer cost 39.095 ms
DEBUG 05-06 11:49:04.514230.514230 lmp.py:1391] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 11:49:04.514232.514232 lmp.py:1347] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 11:49:04.514131.514131 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:49:04.516816.516816 cuda_h.py:27] end prefill_attn cost 1.809 ms
DEBUG 05-06 11:49:04.516089.516089 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 11:49:04.517533.517533 cuda_h.py:27] end prefill_gate cost 0.399 ms
experts_cpu_alloc {'expert_ids': [11, 51, 79, 83, 111, 91, 87, 123, 15, 59, 19, 31, 99, 119, 55, 88, 92, 112, 36, 96, 8, 80, 44, 124, 16, 60, 104, 108, 116, 100, 120, 84, 77, 117, 65, 85, 93, 97, 13, 113, 9, 29, 33, 109, 125, 66, 102, 122, 14, 38, 42, 34, 62, 90, 114, 118, 126], 'token_total': 279, 'token_per_expert': {11: 1, 51: 1, 79: 1, 83: 1, 111: 1, 91: 2, 87: 3, 123: 3, 15: 4, 59: 4, 19: 6, 31: 8, 99: 8, 119: 8, 55: 9, 88: 1, 92: 1, 112: 2, 36: 4, 96: 4, 8: 5, 80: 5, 44: 6, 124: 7, 16: 8, 60: 8, 104: 8, 108: 9, 116: 9, 100: 13, 120: 15, 84: 17, 77: 1, 117: 1, 65: 2, 85: 2, 93: 2, 97: 4, 13: 5, 113: 5, 9: 6, 29: 6, 33: 6, 109: 6, 125: 8, 66: 2, 102: 2, 122: 2, 14: 3, 38: 3, 42: 3, 34: 4, 62: 4, 90: 4, 114: 4, 118: 4, 126: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 35, 39, 43, 47, 63, 67, 71, 75, 95, 103, 107], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1019, 'token_per_expert': {3: 267, 7: 256, 23: 73, 27: 51, 35: 19, 39: 52, 43: 26, 47: 23, 63: 24, 67: 14, 71: 23, 75: 61, 95: 73, 103: 18, 107: 39}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 28, 40, 48, 52, 56, 64, 68, 72, 76], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 31, 'token_total': 953, 'token_per_expert': {0: 264, 4: 267, 12: 20, 20: 23, 24: 75, 28: 26, 40: 38, 48: 22, 52: 35, 56: 23, 64: 32, 68: 26, 72: 34, 76: 68}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 37, 45, 49, 53, 57, 61, 69, 73, 89, 101], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 993, 'token_per_expert': {1: 258, 5: 274, 17: 33, 21: 54, 37: 80, 45: 13, 49: 30, 53: 27, 57: 21, 61: 32, 69: 82, 73: 14, 89: 41, 101: 34}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 54, 58, 70, 74, 78, 86, 94, 98, 106], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 852, 'token_per_expert': {2: 260, 6: 269, 10: 23, 18: 32, 22: 15, 54: 21, 58: 29, 70: 18, 74: 59, 78: 12, 86: 70, 94: 12, 98: 10, 106: 22}}
INFO 05-06 11:49:04.518516.518516 lmp.py:1836] [layer_moe_fused] layer=17 prefix: 0.413ms alloc: 0.405ms
INFO 05-06 11:49:04.519037.519037 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.6743621826171875e-05 seconds
INFO 05-06 11:49:04.520547.520547 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008137226104736328s
INFO 05-06 11:49:04.520997.520997 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005152225494384766 seconds
DEBUG 05-06 11:49:04.520710.520710 cuda_h.py:27] end moe_cpu_prep_submit cost 0.656 ms
INFO 05-06 11:49:04.532522.532522 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.011509418487548828s
DEBUG 05-06 11:49:04.532896.532896 cuda_h.py:27] end moe_wait_copy_tasks cost 11.623 ms
DEBUG 05-06 11:49:04.536599.536599 cuda_h.py:27] end moe_vllm_forward cost 3.421 ms
DEBUG 05-06 11:49:04.536689.536689 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:49:04.536198.536198 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.536062.536062 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.124ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.537701.537701 cuda_h.py:27] end *layer_moe_fused cost 19.005 ms
DEBUG 05-06 11:49:04.541313.541313 cuda_h.py:27] end prefill_merge_scale cost 4.569 ms
DEBUG 05-06 11:49:04.541919.541919 cuda_h.py:27] end prefill_layer cost 27.647 ms
DEBUG 05-06 11:49:04.542967.542967 lmp.py:1391] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 11:49:04.542445.542445 lmp.py:1347] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 11:49:04.542166.542166 cuda_h.py:27] end prefill_ln cost 0.212 ms
DEBUG 05-06 11:49:04.544220.544220 cuda_h.py:27] end prefill_attn cost 1.765 ms
DEBUG 05-06 11:49:04.544070.544070 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 11:49:04.545527.545527 cuda_h.py:27] end prefill_gate cost 0.410 ms
experts_cpu_alloc {'expert_ids': [11, 19, 63, 27, 23, 115, 59, 67, 39, 51, 55, 103, 107, 15, 35, 95, 44, 96, 112, 24, 16, 52, 124, 116, 56, 68, 80, 12, 48, 108, 92, 25, 41, 113, 45, 9, 109, 21, 29, 97, 89, 13, 37, 73, 125, 69, 102, 18, 94, 98, 62, 66, 74, 82, 114, 26, 30, 42, 46], 'token_total': 349, 'token_per_expert': {11: 1, 19: 1, 63: 1, 27: 2, 23: 3, 115: 3, 59: 4, 67: 4, 39: 5, 51: 5, 55: 5, 103: 6, 107: 8, 15: 9, 35: 11, 95: 12, 44: 1, 96: 1, 112: 1, 24: 3, 16: 4, 52: 4, 124: 5, 116: 7, 56: 9, 68: 9, 80: 10, 12: 11, 48: 12, 108: 12, 92: 16, 25: 1, 41: 1, 113: 1, 45: 5, 9: 6, 109: 6, 21: 7, 29: 7, 97: 7, 89: 8, 13: 9, 37: 10, 73: 11, 125: 12, 69: 16, 102: 1, 18: 2, 94: 2, 98: 3, 62: 4, 66: 4, 74: 4, 82: 5, 114: 5, 26: 6, 30: 6, 42: 7, 46: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 43, 47, 71, 75, 83, 87, 91, 99, 111, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 940, 'token_per_expert': {3: 294, 7: 263, 31: 32, 43: 40, 47: 15, 71: 16, 75: 14, 83: 43, 87: 24, 91: 15, 99: 69, 111: 52, 119: 23, 123: 16, 127: 24}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 32, 36, 40, 60, 64, 72, 76, 84, 88, 100, 104, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 910, 'token_per_expert': {0: 260, 4: 287, 8: 31, 32: 45, 36: 47, 40: 30, 60: 25, 64: 31, 72: 22, 76: 22, 84: 24, 88: 17, 100: 22, 104: 26, 120: 21}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 33, 49, 53, 57, 61, 65, 77, 81, 85, 93, 101, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1016, 'token_per_expert': {1: 282, 5: 260, 17: 30, 33: 32, 49: 39, 53: 32, 57: 17, 61: 35, 65: 25, 77: 60, 81: 22, 85: 51, 93: 22, 101: 38, 121: 71}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 34, 38, 50, 54, 58, 70, 78, 90, 110, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 881, 'token_per_expert': {2: 302, 6: 256, 10: 15, 14: 38, 34: 17, 38: 18, 50: 40, 54: 57, 58: 34, 70: 9, 78: 24, 90: 10, 110: 22, 118: 30, 122: 9}}
INFO 05-06 11:49:04.546153.546153 lmp.py:1836] [layer_moe_fused] layer=18 prefix: 0.414ms alloc: 0.420ms
INFO 05-06 11:49:04.547065.547065 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.246566772460938e-05 seconds
INFO 05-06 11:49:04.548819.548819 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007436275482177734s
INFO 05-06 11:49:04.548236.548236 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005266666412353516 seconds
DEBUG 05-06 11:49:04.549655.549655 cuda_h.py:27] end moe_cpu_prep_submit cost 1.154 ms
INFO 05-06 11:49:04.570692.570692 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.021263599395751953s
DEBUG 05-06 11:49:04.570835.570835 cuda_h.py:27] end moe_wait_copy_tasks cost 21.381 ms
DEBUG 05-06 11:49:04.574540.574540 cuda_h.py:27] end moe_vllm_forward cost 3.491 ms
DEBUG 05-06 11:49:04.574916.574916 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:49:04.574636.574636 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.575738.575738 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.175ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.575146.575146 cuda_h.py:27] end *layer_moe_fused cost 29.484 ms
DEBUG 05-06 11:49:04.579560.579560 cuda_h.py:27] end prefill_merge_scale cost 3.816 ms
DEBUG 05-06 11:49:04.579418.579418 cuda_h.py:27] end prefill_layer cost 37.373 ms
DEBUG 05-06 11:49:04.579412.579412 lmp.py:1391] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 11:49:04.579380.579380 lmp.py:1347] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 11:49:04.580841.580841 cuda_h.py:27] end prefill_ln cost 0.249 ms
DEBUG 05-06 11:49:04.583319.583319 cuda_h.py:27] end prefill_attn cost 2.872 ms
DEBUG 05-06 11:49:04.584622.584622 cuda_h.py:27] end prefill_ffn_prep cost 0.507 ms
DEBUG 05-06 11:49:04.585860.585860 cuda_h.py:27] end prefill_gate cost 0.407 ms
experts_cpu_alloc {'expert_ids': [43, 115, 107, 127, 103, 59, 111, 15, 55, 99, 47, 27, 83, 8, 124, 100, 20, 68, 56, 96, 112, 60, 84, 104, 108, 12, 72, 29, 49, 93, 105, 57, 101, 45, 77, 121, 25, 65, 53, 97, 33, 13, 18, 74, 46, 54, 82, 114, 42, 66, 70, 118, 86, 106], 'token_total': 312, 'token_per_expert': {43: 1, 115: 1, 107: 2, 127: 3, 103: 6, 59: 7, 111: 7, 15: 9, 55: 12, 99: 12, 47: 13, 27: 14, 83: 14, 8: 1, 124: 1, 100: 2, 20: 4, 68: 5, 56: 6, 96: 7, 112: 7, 60: 10, 84: 10, 104: 11, 108: 11, 12: 17, 72: 18, 29: 1, 49: 1, 93: 1, 105: 1, 57: 2, 101: 2, 45: 3, 77: 3, 121: 4, 25: 5, 65: 5, 53: 6, 97: 6, 33: 9, 13: 10, 18: 1, 74: 1, 46: 2, 54: 2, 82: 2, 114: 2, 42: 3, 66: 3, 70: 3, 118: 6, 86: 8, 106: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 39, 51, 63, 75, 79, 119, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 898, 'token_per_expert': {3: 301, 7: 304, 11: 15, 19: 21, 23: 22, 31: 23, 35: 17, 39: 23, 51: 58, 63: 20, 75: 19, 79: 24, 119: 17, 123: 34}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 24, 36, 40, 44, 48, 52, 64, 76, 80, 88, 92], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1040, 'token_per_expert': {0: 265, 4: 258, 16: 22, 24: 40, 36: 19, 40: 35, 44: 74, 48: 21, 52: 109, 64: 65, 76: 27, 80: 22, 88: 33, 92: 50}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 37, 41, 61, 69, 73, 89, 109, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 957, 'token_per_expert': {1: 281, 5: 275, 9: 30, 17: 12, 21: 27, 37: 80, 41: 15, 61: 30, 69: 17, 73: 12, 89: 83, 109: 25, 117: 52, 125: 18}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 26, 38, 50, 58, 90, 98, 102, 122, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 889, 'token_per_expert': {2: 270, 6: 261, 10: 28, 22: 13, 26: 17, 38: 96, 50: 33, 58: 13, 90: 11, 98: 14, 102: 29, 122: 92, 126: 12}}
INFO 05-06 11:49:04.586697.586697 lmp.py:1836] [layer_moe_fused] layer=19 prefix: 0.420ms alloc: 0.396ms
INFO 05-06 11:49:04.586649.586649 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.7220458984375e-05 seconds
INFO 05-06 11:49:04.587076.587076 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007698535919189453s
INFO 05-06 11:49:04.587653.587653 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005385875701904297 seconds
DEBUG 05-06 11:49:04.588262.588262 cuda_h.py:27] end moe_cpu_prep_submit cost 1.021 ms
INFO 05-06 11:49:04.603376.603376 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014270544052124023s
DEBUG 05-06 11:49:04.603042.603042 cuda_h.py:27] end moe_wait_copy_tasks cost 14.392 ms
DEBUG 05-06 11:49:04.606913.606913 cuda_h.py:27] end moe_vllm_forward cost 3.505 ms
DEBUG 05-06 11:49:04.607143.607143 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:49:04.607413.607413 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.607515.607515 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.209ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.607030.607030 cuda_h.py:27] end *layer_moe_fused cost 22.449 ms
DEBUG 05-06 11:49:04.613298.613298 cuda_h.py:27] end prefill_merge_scale cost 5.757 ms
DEBUG 05-06 11:49:04.613374.613374 cuda_h.py:27] end prefill_layer cost 34.017 ms
DEBUG 05-06 11:49:04.614228.614228 lmp.py:1391] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 11:49:04.614230.614230 lmp.py:1347] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 11:49:04.614129.614129 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:49:04.616144.616144 cuda_h.py:27] end prefill_attn cost 1.773 ms
DEBUG 05-06 11:49:04.616033.616033 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 11:49:04.617391.617391 cuda_h.py:27] end prefill_gate cost 0.410 ms
experts_cpu_alloc {'expert_ids': [31, 51, 87, 75, 99, 11, 35, 47, 19, 95, 16, 104, 24, 36, 60, 76, 12, 120, 84, 20, 72, 100, 112, 64, 17, 89, 25, 61, 105, 117, 97, 101, 69, 9, 93, 121, 41, 113, 53, 85, 81, 22, 86, 106, 110, 126, 34, 62, 70, 90, 98, 26, 74, 18, 10, 58, 38], 'token_total': 333, 'token_per_expert': {31: 1, 51: 1, 87: 1, 75: 2, 99: 2, 11: 4, 35: 5, 47: 5, 19: 7, 95: 7, 16: 1, 104: 1, 24: 5, 36: 6, 60: 6, 76: 6, 12: 7, 120: 7, 84: 8, 20: 9, 72: 11, 100: 11, 112: 14, 64: 17, 17: 1, 89: 1, 25: 2, 61: 2, 105: 3, 117: 3, 97: 4, 101: 4, 69: 5, 9: 7, 93: 7, 121: 7, 41: 11, 113: 12, 53: 15, 85: 15, 81: 17, 22: 1, 86: 1, 106: 2, 110: 2, 126: 2, 34: 3, 62: 3, 70: 3, 90: 3, 98: 4, 26: 6, 74: 7, 18: 8, 10: 9, 58: 9, 38: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 43, 55, 59, 63, 71, 79, 83, 103, 107, 111, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 25, 'token_total': 895, 'token_per_expert': {3: 296, 7: 258, 15: 11, 27: 23, 43: 27, 55: 10, 59: 27, 63: 67, 71: 13, 79: 10, 83: 8, 103: 10, 107: 100, 111: 7, 123: 28}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 28, 32, 40, 44, 52, 56, 68, 88, 92, 108, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1038, 'token_per_expert': {0: 265, 4: 301, 8: 30, 28: 36, 32: 22, 40: 36, 44: 30, 52: 19, 56: 44, 68: 139, 88: 24, 92: 37, 108: 28, 116: 27}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 33, 37, 45, 49, 57, 65, 73, 77, 109, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 31, 'token_total': 1024, 'token_per_expert': {1: 265, 5: 302, 13: 27, 21: 35, 33: 20, 37: 37, 45: 77, 49: 74, 57: 33, 65: 28, 73: 39, 77: 38, 109: 27, 125: 22}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 42, 46, 50, 54, 66, 82, 94, 102, 114, 118, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 806, 'token_per_expert': {2: 264, 6: 257, 30: 26, 42: 18, 46: 11, 50: 12, 54: 17, 66: 25, 82: 19, 94: 79, 102: 46, 114: 11, 118: 10, 122: 11}}
INFO 05-06 11:49:04.618997.618997 lmp.py:1836] [layer_moe_fused] layer=20 prefix: 0.412ms alloc: 0.409ms
INFO 05-06 11:49:04.618809.618809 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.936622619628906e-05 seconds
INFO 05-06 11:49:04.619709.619709 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007655620574951172s
INFO 05-06 11:49:04.620894.620894 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005292892456054688 seconds
DEBUG 05-06 11:49:04.620599.620599 cuda_h.py:27] end moe_cpu_prep_submit cost 1.005 ms
INFO 05-06 11:49:04.636820.636820 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015226364135742188s
DEBUG 05-06 11:49:04.636154.636154 cuda_h.py:27] end moe_wait_copy_tasks cost 15.344 ms
DEBUG 05-06 11:49:04.640207.640207 cuda_h.py:27] end moe_vllm_forward cost 3.571 ms
DEBUG 05-06 11:49:04.640582.640582 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:49:04.640935.640935 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.640891.640891 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.368ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.641228.641228 cuda_h.py:27] end *layer_moe_fused cost 23.362 ms
DEBUG 05-06 11:49:04.645384.645384 cuda_h.py:27] end prefill_merge_scale cost 4.589 ms
DEBUG 05-06 11:49:04.646075.646075 cuda_h.py:27] end prefill_layer cost 31.977 ms
DEBUG 05-06 11:49:04.646690.646690 lmp.py:1391] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 11:49:04.646215.646215 lmp.py:1347] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 11:49:04.646683.646683 cuda_h.py:27] end prefill_ln cost 0.201 ms
DEBUG 05-06 11:49:04.648167.648167 cuda_h.py:27] end prefill_attn cost 1.733 ms
DEBUG 05-06 11:49:04.648679.648679 cuda_h.py:27] end prefill_ffn_prep cost 0.373 ms
DEBUG 05-06 11:49:04.649567.649567 cuda_h.py:27] end prefill_gate cost 0.408 ms
experts_cpu_alloc {'expert_ids': [19, 47, 59, 99, 23, 27, 107, 43, 71, 115, 119, 55, 79, 56, 52, 60, 104, 16, 32, 96, 64, 88, 80, 20, 24, 40, 44, 12, 9, 25, 77, 17, 113, 101, 69, 93, 125, 21, 45, 121, 81, 33, 54, 98, 106, 94, 126, 114, 38, 50, 74, 34, 58, 82, 42, 118, 10], 'token_total': 310, 'token_per_expert': {19: 1, 47: 1, 59: 1, 99: 1, 23: 2, 27: 2, 107: 2, 43: 3, 71: 6, 115: 6, 119: 6, 55: 7, 79: 9, 56: 1, 52: 2, 60: 2, 104: 2, 16: 3, 32: 3, 96: 4, 64: 5, 88: 5, 80: 6, 20: 7, 24: 10, 40: 10, 44: 10, 12: 14, 9: 1, 25: 1, 77: 1, 17: 2, 113: 3, 101: 4, 69: 5, 93: 5, 125: 6, 21: 7, 45: 7, 121: 7, 81: 9, 33: 11, 54: 1, 98: 2, 106: 3, 94: 4, 126: 4, 114: 5, 38: 7, 50: 7, 74: 7, 34: 11, 58: 11, 82: 11, 42: 12, 118: 12, 10: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 31, 35, 51, 67, 75, 83, 87, 95, 103, 111, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 800, 'token_per_expert': {3: 260, 7: 281, 11: 34, 31: 13, 35: 27, 51: 34, 67: 12, 75: 12, 83: 28, 87: 11, 95: 11, 103: 23, 111: 27, 123: 10, 127: 17}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 36, 48, 68, 72, 76, 84, 92, 100, 112, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 922, 'token_per_expert': {0: 257, 4: 277, 8: 26, 36: 14, 48: 55, 68: 22, 72: 19, 76: 36, 84: 28, 92: 34, 100: 65, 112: 32, 120: 32, 124: 25}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 37, 41, 53, 57, 61, 65, 73, 97, 105, 109], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1032, 'token_per_expert': {1: 326, 5: 350, 13: 27, 29: 30, 37: 28, 41: 20, 53: 31, 57: 20, 61: 27, 65: 60, 73: 32, 97: 18, 105: 40, 109: 23}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 26, 30, 46, 62, 70, 78, 86, 90, 102, 110, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1032, 'token_per_expert': {2: 276, 6: 334, 18: 39, 26: 56, 30: 20, 46: 40, 62: 24, 70: 13, 78: 99, 86: 15, 90: 26, 102: 21, 110: 31, 122: 38}}
INFO 05-06 11:49:04.650411.650411 lmp.py:1836] [layer_moe_fused] layer=21 prefix: 0.415ms alloc: 0.406ms
INFO 05-06 11:49:04.651701.651701 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.079673767089844e-05 seconds
INFO 05-06 11:49:04.651890.651890 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007419586181640625s
INFO 05-06 11:49:04.652453.652453 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005288124084472656 seconds
DEBUG 05-06 11:49:04.653476.653476 cuda_h.py:27] end moe_cpu_prep_submit cost 0.918 ms
INFO 05-06 11:49:04.670738.670738 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.016973495483398438s
DEBUG 05-06 11:49:04.670920.670920 cuda_h.py:27] end moe_wait_copy_tasks cost 17.088 ms
DEBUG 05-06 11:49:04.674880.674880 cuda_h.py:27] end moe_vllm_forward cost 3.403 ms
DEBUG 05-06 11:49:04.674017.674017 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:49:04.674264.674264 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.674220.674220 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.187ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.675364.675364 cuda_h.py:27] end *layer_moe_fused cost 25.123 ms
DEBUG 05-06 11:49:04.679848.679848 cuda_h.py:27] end prefill_merge_scale cost 4.511 ms
DEBUG 05-06 11:49:04.679639.679639 cuda_h.py:27] end prefill_layer cost 33.633 ms
DEBUG 05-06 11:49:04.680947.680947 lmp.py:1391] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 11:49:04.680710.680710 lmp.py:1347] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 11:49:04.680662.680662 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:49:04.682027.682027 cuda_h.py:27] end prefill_attn cost 1.749 ms
DEBUG 05-06 11:49:04.682062.682062 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 11:49:04.683879.683879 cuda_h.py:27] end prefill_gate cost 0.465 ms
experts_cpu_alloc {'expert_ids': [63, 71, 95, 23, 27, 67, 39, 87, 51, 47, 15, 83, 11, 107, 115, 79, 31, 36, 60, 20, 96, 84, 112, 48, 32, 44, 16, 124, 40, 88, 9, 13, 17, 29, 61, 81, 109, 65, 125, 45, 57, 14, 54, 110, 10, 62, 98, 106, 122, 42, 102, 34, 26, 58, 30], 'token_total': 305, 'token_per_expert': {63: 1, 71: 1, 95: 1, 23: 2, 27: 2, 67: 2, 39: 3, 87: 3, 51: 6, 47: 7, 15: 8, 83: 8, 11: 9, 107: 10, 115: 10, 79: 12, 31: 15, 36: 1, 60: 1, 20: 2, 96: 2, 84: 3, 112: 6, 48: 7, 32: 9, 44: 9, 16: 10, 124: 13, 40: 14, 88: 14, 9: 1, 13: 1, 17: 1, 29: 2, 61: 2, 81: 2, 109: 2, 65: 3, 125: 3, 45: 5, 57: 5, 14: 1, 54: 1, 110: 1, 10: 2, 62: 2, 98: 3, 106: 3, 122: 4, 42: 7, 102: 10, 34: 11, 26: 13, 58: 13, 30: 16}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 35, 43, 55, 59, 75, 99, 103, 111, 119, 123, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 31, 'token_total': 965, 'token_per_expert': {3: 257, 7: 281, 19: 19, 35: 80, 43: 25, 55: 33, 59: 44, 75: 27, 99: 19, 103: 49, 111: 24, 119: 38, 123: 37, 127: 32}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 24, 28, 64, 68, 72, 76, 92, 100, 108, 116, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1168, 'token_per_expert': {0: 260, 4: 256, 8: 34, 24: 64, 28: 17, 64: 100, 68: 43, 72: 97, 76: 22, 92: 33, 100: 156, 108: 25, 116: 31, 120: 30}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 25, 33, 41, 53, 69, 73, 85, 89, 93, 101, 113, 117], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 787, 'token_per_expert': {1: 274, 5: 257, 25: 8, 33: 13, 41: 11, 53: 27, 69: 18, 73: 47, 85: 9, 89: 13, 93: 47, 101: 13, 113: 12, 117: 38}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 66, 70, 74, 82, 86, 90, 94, 118, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 871, 'token_per_expert': {2: 257, 6: 257, 38: 32, 46: 19, 66: 17, 70: 29, 74: 66, 82: 24, 86: 28, 90: 32, 94: 30, 118: 16, 126: 64}}
INFO 05-06 11:49:04.684756.684756 lmp.py:1836] [layer_moe_fused] layer=22 prefix: 0.413ms alloc: 0.395ms
INFO 05-06 11:49:04.685754.685754 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.91278076171875e-05 seconds
INFO 05-06 11:49:04.685486.685486 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007135868072509766s
INFO 05-06 11:49:04.686366.686366 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005178451538085938 seconds
DEBUG 05-06 11:49:04.687677.687677 cuda_h.py:27] end moe_cpu_prep_submit cost 1.091 ms
INFO 05-06 11:49:04.703871.703871 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.016088247299194336s
DEBUG 05-06 11:49:04.703476.703476 cuda_h.py:27] end moe_wait_copy_tasks cost 16.209 ms
DEBUG 05-06 11:49:04.707250.707250 cuda_h.py:27] end moe_vllm_forward cost 3.372 ms
DEBUG 05-06 11:49:04.707195.707195 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:49:04.707651.707651 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.707514.707514 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.064ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.708665.708665 cuda_h.py:27] end *layer_moe_fused cost 24.199 ms
DEBUG 05-06 11:49:04.713842.713842 cuda_h.py:27] end prefill_merge_scale cost 4.637 ms
DEBUG 05-06 11:49:04.713626.713626 cuda_h.py:27] end prefill_layer cost 32.908 ms
DEBUG 05-06 11:49:04.713752.713752 lmp.py:1391] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 11:49:04.713323.713323 lmp.py:1347] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 11:49:04.713745.713745 cuda_h.py:27] end prefill_ln cost 0.198 ms
DEBUG 05-06 11:49:04.715157.715157 cuda_h.py:27] end prefill_attn cost 1.747 ms
DEBUG 05-06 11:49:04.716644.716644 cuda_h.py:27] end prefill_ffn_prep cost 0.423 ms
DEBUG 05-06 11:49:04.717803.717803 cuda_h.py:27] end prefill_gate cost 0.401 ms
experts_cpu_alloc {'expert_ids': [55, 119, 11, 23, 95, 99, 107, 127, 75, 19, 51, 91, 103, 28, 88, 60, 64, 12, 48, 92, 36, 52, 120, 32, 40, 68, 76, 124, 116, 13, 69, 77, 41, 53, 81, 49, 89, 33, 9, 57, 73, 105, 50, 102, 62, 74, 10, 54, 58, 66, 82, 110, 14, 38, 122, 42], 'token_total': 268, 'token_per_expert': {55: 1, 119: 1, 11: 2, 23: 2, 95: 2, 99: 2, 107: 3, 127: 3, 75: 5, 19: 7, 51: 7, 91: 9, 103: 9, 28: 1, 88: 1, 60: 2, 64: 2, 12: 3, 48: 4, 92: 4, 36: 5, 52: 5, 120: 5, 32: 6, 40: 6, 68: 6, 76: 6, 124: 8, 116: 12, 13: 1, 69: 1, 77: 1, 41: 2, 53: 3, 81: 3, 49: 4, 89: 4, 33: 9, 9: 10, 57: 10, 73: 15, 105: 18, 50: 1, 102: 1, 62: 2, 74: 2, 10: 4, 54: 4, 58: 4, 66: 4, 82: 4, 110: 4, 14: 5, 38: 5, 122: 7, 42: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 39, 43, 47, 59, 67, 71, 79, 83, 87, 115, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 967, 'token_per_expert': {3: 284, 7: 260, 31: 15, 35: 28, 39: 59, 43: 56, 47: 30, 59: 14, 67: 66, 71: 11, 79: 46, 83: 30, 87: 17, 115: 15, 123: 36}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 24, 44, 56, 72, 80, 84, 100, 104, 108, 112], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 809, 'token_per_expert': {0: 257, 4: 257, 8: 17, 16: 22, 24: 14, 44: 37, 56: 59, 72: 16, 80: 17, 84: 23, 100: 25, 104: 21, 108: 30, 112: 14}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 25, 29, 37, 61, 65, 85, 97, 109, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1124, 'token_per_expert': {1: 295, 5: 299, 17: 20, 21: 103, 25: 39, 29: 40, 37: 45, 61: 57, 65: 40, 85: 42, 97: 47, 109: 30, 117: 23, 125: 44}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 26, 30, 34, 46, 78, 86, 90, 98, 106, 118], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 928, 'token_per_expert': {2: 275, 6: 275, 18: 28, 22: 19, 26: 14, 30: 14, 34: 13, 46: 56, 78: 20, 86: 83, 90: 36, 98: 40, 106: 15, 118: 40}}
INFO 05-06 11:49:04.718402.718402 lmp.py:1836] [layer_moe_fused] layer=23 prefix: 0.411ms alloc: 0.402ms
INFO 05-06 11:49:04.718214.718214 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.008148193359375e-05 seconds
INFO 05-06 11:49:04.719506.719506 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006721019744873047s
INFO 05-06 11:49:04.719560.719560 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005397796630859375 seconds
DEBUG 05-06 11:49:04.720113.720113 cuda_h.py:27] end moe_cpu_prep_submit cost 0.998 ms
INFO 05-06 11:49:04.735712.735712 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015485286712646484s
DEBUG 05-06 11:49:04.735133.735133 cuda_h.py:27] end moe_wait_copy_tasks cost 15.595 ms
DEBUG 05-06 11:49:04.739283.739283 cuda_h.py:27] end moe_vllm_forward cost 3.542 ms
DEBUG 05-06 11:49:04.740083.740083 cuda_h.py:27] end moe_cpu_merge cost 0.055 ms
DEBUG 05-06 11:49:04.740259.740259 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.740692.740692 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.202ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.740605.740605 cuda_h.py:27] end *layer_moe_fused cost 23.509 ms
DEBUG 05-06 11:49:04.746217.746217 cuda_h.py:27] end prefill_merge_scale cost 5.769 ms
DEBUG 05-06 11:49:04.746724.746724 cuda_h.py:27] end prefill_layer cost 33.334 ms
DEBUG 05-06 11:49:04.747471.747471 lmp.py:1391] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 11:49:04.747996.747996 lmp.py:1347] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 11:49:04.747160.747160 cuda_h.py:27] end prefill_ln cost 0.202 ms
DEBUG 05-06 11:49:04.749745.749745 cuda_h.py:27] end prefill_attn cost 1.771 ms
DEBUG 05-06 11:49:04.749541.749541 cuda_h.py:27] end prefill_ffn_prep cost 0.371 ms
DEBUG 05-06 11:49:04.750608.750608 cuda_h.py:27] end prefill_gate cost 0.404 ms
experts_cpu_alloc {'expert_ids': [15, 47, 95, 123, 31, 87, 107, 99, 115, 55, 119, 75, 79, 127, 24, 28, 80, 112, 40, 76, 84, 104, 68, 32, 20, 92, 96, 124, 120, 93, 101, 113, 25, 65, 105, 53, 61, 57, 81, 9, 10, 22, 78, 106, 102, 126, 38, 42, 66, 46, 62, 118, 50, 74], 'token_total': 239, 'token_per_expert': {15: 1, 47: 1, 95: 1, 123: 1, 31: 2, 87: 2, 107: 2, 99: 3, 115: 3, 55: 4, 119: 6, 75: 7, 79: 7, 127: 7, 24: 2, 28: 2, 80: 2, 112: 2, 40: 3, 76: 3, 84: 3, 104: 3, 68: 4, 32: 5, 20: 7, 92: 7, 96: 7, 124: 7, 120: 8, 93: 1, 101: 1, 113: 1, 25: 2, 65: 2, 105: 4, 53: 6, 61: 6, 57: 7, 81: 8, 9: 10, 10: 1, 22: 1, 78: 1, 106: 1, 102: 2, 126: 3, 38: 4, 42: 5, 66: 5, 46: 6, 62: 12, 118: 12, 50: 13, 74: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 35, 43, 63, 67, 71, 83, 91, 111], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 955, 'token_per_expert': {3: 256, 7: 269, 11: 54, 19: 39, 23: 20, 27: 77, 35: 23, 43: 14, 63: 62, 67: 46, 71: 34, 83: 18, 91: 33, 111: 10}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 36, 44, 48, 52, 56, 60, 64, 100, 108], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 930, 'token_per_expert': {0: 256, 4: 293, 8: 12, 12: 37, 16: 29, 36: 18, 44: 52, 48: 23, 52: 63, 56: 42, 60: 14, 64: 72, 100: 9, 108: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 29, 33, 37, 45, 49, 73, 77, 97, 109, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 997, 'token_per_expert': {1: 281, 5: 270, 13: 14, 17: 28, 29: 18, 33: 58, 37: 19, 45: 35, 49: 10, 73: 23, 77: 23, 97: 73, 109: 23, 121: 122}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 34, 70, 82, 86, 90, 94, 98, 110, 114, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 975, 'token_per_expert': {2: 256, 6: 307, 30: 20, 34: 44, 70: 70, 82: 14, 86: 18, 90: 112, 94: 28, 98: 42, 110: 21, 114: 27, 122: 16}}
INFO 05-06 11:49:04.751677.751677 lmp.py:1836] [layer_moe_fused] layer=24 prefix: 0.410ms alloc: 0.402ms
INFO 05-06 11:49:04.751244.751244 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.650520324707031e-05 seconds
INFO 05-06 11:49:04.752862.752862 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0006756782531738281s
INFO 05-06 11:49:04.753981.753981 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005183219909667969 seconds
DEBUG 05-06 11:49:04.753304.753304 cuda_h.py:27] end moe_cpu_prep_submit cost 1.064 ms
INFO 05-06 11:49:04.766674.766674 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.011928558349609375s
DEBUG 05-06 11:49:04.766002.766002 cuda_h.py:27] end moe_wait_copy_tasks cost 12.044 ms
DEBUG 05-06 11:49:04.770712.770712 cuda_h.py:27] end moe_vllm_forward cost 3.427 ms
DEBUG 05-06 11:49:04.770849.770849 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:49:04.770066.770066 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:49:04.770645.770645 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.122ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.771419.771419 cuda_h.py:27] end *layer_moe_fused cost 20.007 ms
DEBUG 05-06 11:49:04.775967.775967 cuda_h.py:27] end prefill_merge_scale cost 4.843 ms
DEBUG 05-06 11:49:04.776520.776520 cuda_h.py:27] end prefill_layer cost 28.896 ms
DEBUG 05-06 11:49:04.776201.776201 lmp.py:1391] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 11:49:04.776918.776918 lmp.py:1347] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 11:49:04.776479.776479 cuda_h.py:27] end prefill_ln cost 0.202 ms
DEBUG 05-06 11:49:04.778347.778347 cuda_h.py:27] end prefill_attn cost 1.735 ms
DEBUG 05-06 11:49:04.778236.778236 cuda_h.py:27] end prefill_ffn_prep cost 0.370 ms
DEBUG 05-06 11:49:04.779292.779292 cuda_h.py:27] end prefill_gate cost 0.465 ms
experts_cpu_alloc {'expert_ids': [15, 95, 127, 27, 55, 75, 103, 23, 31, 79, 47, 51, 119, 43, 99, 87, 76, 84, 108, 24, 12, 92, 124, 8, 72, 88, 112, 116, 48, 57, 61, 33, 73, 81, 13, 53, 121, 29, 21, 49, 9, 109, 74, 38, 118, 26, 42, 22, 122, 66, 46, 126, 50, 78], 'token_total': 263, 'token_per_expert': {15: 1, 95: 1, 127: 1, 27: 2, 55: 2, 75: 2, 103: 3, 23: 4, 31: 4, 79: 5, 47: 6, 51: 7, 119: 7, 43: 8, 99: 8, 87: 9, 76: 1, 84: 1, 108: 1, 24: 2, 12: 3, 92: 4, 124: 5, 8: 7, 72: 7, 88: 7, 112: 7, 116: 13, 48: 14, 57: 1, 61: 1, 33: 2, 73: 2, 81: 2, 13: 3, 53: 3, 121: 4, 29: 5, 21: 6, 49: 6, 9: 7, 109: 8, 74: 2, 38: 3, 118: 3, 26: 4, 42: 4, 22: 5, 122: 5, 66: 6, 46: 8, 126: 9, 50: 10, 78: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 35, 39, 63, 67, 71, 83, 91, 107, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 919, 'token_per_expert': {3: 285, 7: 274, 11: 21, 19: 11, 35: 56, 39: 14, 63: 16, 67: 21, 71: 22, 83: 25, 91: 11, 107: 74, 111: 25, 123: 64}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 36, 44, 52, 56, 60, 64, 68, 80, 100, 104, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1058, 'token_per_expert': {0: 267, 4: 260, 16: 157, 36: 16, 44: 19, 52: 53, 56: 14, 60: 49, 64: 39, 68: 88, 80: 27, 100: 21, 104: 31, 120: 17}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 25, 41, 45, 69, 77, 85, 89, 93, 97, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 825, 'token_per_expert': {1: 256, 5: 264, 17: 9, 25: 10, 41: 11, 45: 72, 69: 46, 77: 13, 85: 53, 89: 9, 93: 21, 97: 12, 117: 40, 125: 9}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 34, 58, 70, 82, 90, 106, 110, 114], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 1031, 'token_per_expert': {2: 304, 6: 260, 10: 14, 14: 13, 18: 63, 34: 24, 58: 136, 70: 33, 82: 21, 90: 21, 106: 23, 110: 100, 114: 19}}
INFO 05-06 11:49:04.780354.780354 lmp.py:1836] [layer_moe_fused] layer=25 prefix: 0.414ms alloc: 0.393ms
INFO 05-06 11:49:04.781643.781643 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.318092346191406e-05 seconds
INFO 05-06 11:49:04.782353.782353 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007436275482177734s
INFO 05-06 11:49:04.782670.782670 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005230903625488281 seconds
DEBUG 05-06 11:49:04.783725.783725 cuda_h.py:27] end moe_cpu_prep_submit cost 1.132 ms
INFO 05-06 11:49:04.802370.802370 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.018989086151123047s
DEBUG 05-06 11:49:04.802360.802360 cuda_h.py:27] end moe_wait_copy_tasks cost 19.099 ms
DEBUG 05-06 11:49:04.806184.806184 cuda_h.py:27] end moe_vllm_forward cost 3.479 ms
DEBUG 05-06 11:49:04.806943.806943 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:49:04.806260.806260 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.806931.806931 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.177ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.807633.807633 cuda_h.py:27] end *layer_moe_fused cost 27.351 ms
DEBUG 05-06 11:49:04.812623.812623 cuda_h.py:27] end prefill_merge_scale cost 4.779 ms
DEBUG 05-06 11:49:04.812553.812553 cuda_h.py:27] end prefill_layer cost 36.176 ms
DEBUG 05-06 11:49:04.812883.812883 lmp.py:1391] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 11:49:04.812408.812408 lmp.py:1347] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 11:49:04.812446.812446 cuda_h.py:27] end prefill_ln cost 0.199 ms
DEBUG 05-06 11:49:04.814214.814214 cuda_h.py:27] end prefill_attn cost 1.738 ms
DEBUG 05-06 11:49:04.815964.815964 cuda_h.py:27] end prefill_ffn_prep cost 0.372 ms
DEBUG 05-06 11:49:04.816111.816111 cuda_h.py:27] end prefill_gate cost 0.410 ms
experts_cpu_alloc {'expert_ids': [91, 107, 39, 47, 55, 71, 31, 23, 35, 63, 115, 67, 99, 75, 48, 64, 32, 44, 72, 80, 92, 40, 116, 28, 96, 108, 88, 112, 8, 9, 53, 93, 117, 121, 109, 13, 41, 29, 61, 81, 57, 25, 125, 77, 42, 54, 18, 122, 50, 74, 38, 118, 26], 'token_total': 292, 'token_per_expert': {91: 1, 107: 1, 39: 2, 47: 3, 55: 3, 71: 3, 31: 4, 23: 6, 35: 6, 63: 7, 115: 7, 67: 8, 99: 11, 75: 15, 48: 1, 64: 1, 32: 3, 44: 3, 72: 4, 80: 4, 92: 4, 40: 5, 116: 5, 28: 6, 96: 6, 108: 6, 88: 8, 112: 8, 8: 11, 9: 1, 53: 1, 93: 1, 117: 1, 121: 1, 109: 3, 13: 4, 41: 4, 29: 7, 61: 7, 81: 7, 57: 8, 25: 9, 125: 9, 77: 13, 42: 4, 54: 4, 18: 5, 122: 6, 50: 7, 74: 7, 38: 10, 118: 10, 26: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 43, 51, 59, 79, 87, 95, 103, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 986, 'token_per_expert': {3: 268, 7: 256, 15: 22, 19: 21, 27: 50, 43: 49, 51: 19, 59: 17, 79: 17, 87: 85, 95: 64, 103: 19, 111: 77, 123: 22}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 36, 52, 56, 60, 68, 76, 84, 104, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 913, 'token_per_expert': {0: 260, 4: 260, 20: 110, 24: 43, 36: 15, 52: 32, 56: 16, 60: 26, 68: 15, 76: 18, 84: 56, 104: 35, 124: 27}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 37, 45, 49, 65, 73, 85, 89, 97, 105, 113], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 1115, 'token_per_expert': {1: 263, 5: 256, 17: 93, 37: 14, 45: 17, 49: 19, 65: 60, 73: 36, 85: 142, 89: 84, 97: 14, 105: 14, 113: 103}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 30, 66, 70, 78, 86, 90, 102, 114, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 22, 'token_total': 790, 'token_per_expert': {2: 261, 6: 256, 10: 17, 14: 12, 30: 13, 66: 19, 70: 26, 78: 21, 86: 28, 90: 17, 102: 20, 114: 77, 126: 23}}
INFO 05-06 11:49:04.817814.817814 lmp.py:1836] [layer_moe_fused] layer=26 prefix: 0.404ms alloc: 0.384ms
INFO 05-06 11:49:04.817667.817667 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.555152893066406e-05 seconds
INFO 05-06 11:49:04.818347.818347 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007295608520507812s
INFO 05-06 11:49:04.819472.819472 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005228519439697266 seconds
DEBUG 05-06 11:49:04.819432.819432 cuda_h.py:27] end moe_cpu_prep_submit cost 0.695 ms
INFO 05-06 11:49:04.835738.835738 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015772104263305664s
DEBUG 05-06 11:49:04.835258.835258 cuda_h.py:27] end moe_wait_copy_tasks cost 15.888 ms
DEBUG 05-06 11:49:04.839411.839411 cuda_h.py:27] end moe_vllm_forward cost 3.617 ms
DEBUG 05-06 11:49:04.839879.839879 cuda_h.py:27] end moe_cpu_merge cost 0.058 ms
DEBUG 05-06 11:49:04.839877.839877 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.839025.839025 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.285ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.840170.840170 cuda_h.py:27] end *layer_moe_fused cost 23.877 ms
DEBUG 05-06 11:49:04.846877.846877 cuda_h.py:27] end prefill_merge_scale cost 5.625 ms
DEBUG 05-06 11:49:04.846006.846006 cuda_h.py:27] end prefill_layer cost 33.691 ms
DEBUG 05-06 11:49:04.846682.846682 lmp.py:1391] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 11:49:04.846922.846922 lmp.py:1347] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 11:49:04.847267.847267 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 11:49:04.848269.848269 cuda_h.py:27] end prefill_attn cost 1.761 ms
DEBUG 05-06 11:49:04.849549.849549 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 11:49:04.850060.850060 cuda_h.py:27] end prefill_gate cost 0.409 ms
experts_cpu_alloc {'expert_ids': [11, 55, 59, 67, 71, 19, 63, 47, 15, 39, 91, 27, 23, 83, 127, 75, 44, 84, 80, 124, 32, 116, 96, 68, 40, 20, 28, 112, 89, 101, 93, 97, 29, 113, 81, 21, 125, 105, 49, 22, 30, 26, 86, 110, 126, 10, 54, 58, 90, 74, 106, 114, 66, 94, 42], 'token_total': 334, 'token_per_expert': {11: 1, 55: 1, 59: 2, 67: 2, 71: 2, 19: 3, 63: 3, 47: 4, 15: 5, 39: 5, 91: 8, 27: 9, 23: 10, 83: 10, 127: 10, 75: 12, 44: 1, 84: 1, 80: 2, 124: 3, 32: 5, 116: 5, 96: 6, 68: 7, 40: 9, 20: 10, 28: 11, 112: 13, 89: 1, 101: 1, 93: 2, 97: 2, 29: 3, 113: 3, 81: 4, 21: 6, 125: 6, 105: 9, 49: 10, 22: 1, 30: 1, 26: 2, 86: 2, 110: 2, 126: 2, 10: 4, 54: 9, 58: 9, 90: 9, 74: 10, 106: 12, 114: 15, 66: 16, 94: 16, 42: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 43, 51, 79, 87, 95, 103, 111, 115, 119, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 1058, 'token_per_expert': {3: 290, 7: 276, 31: 26, 35: 16, 43: 67, 51: 28, 79: 29, 87: 84, 95: 58, 103: 44, 111: 36, 115: 51, 119: 13, 123: 40}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 24, 36, 48, 56, 64, 76, 88, 100, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 944, 'token_per_expert': {0: 260, 4: 270, 8: 18, 12: 15, 24: 46, 36: 35, 48: 36, 56: 16, 64: 18, 76: 43, 88: 47, 100: 61, 108: 21, 120: 58}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 33, 37, 41, 45, 53, 61, 65, 85, 109, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 897, 'token_per_expert': {1: 294, 5: 257, 13: 28, 25: 39, 33: 40, 37: 25, 41: 21, 45: 51, 53: 23, 61: 20, 65: 43, 85: 12, 109: 24, 121: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 46, 50, 62, 70, 78, 82, 98, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 863, 'token_per_expert': {2: 256, 6: 260, 14: 34, 18: 21, 46: 30, 50: 66, 62: 29, 70: 26, 78: 29, 82: 49, 98: 24, 118: 21, 122: 18}}
INFO 05-06 11:49:04.851149.851149 lmp.py:1836] [layer_moe_fused] layer=27 prefix: 0.424ms alloc: 0.401ms
INFO 05-06 11:49:04.851061.851061 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.413459777832031e-05 seconds
INFO 05-06 11:49:04.852201.852201 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007631778717041016s
INFO 05-06 11:49:04.853903.853903 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005254745483398438 seconds
DEBUG 05-06 11:49:04.853764.853764 cuda_h.py:27] end moe_cpu_prep_submit cost 1.012 ms
INFO 05-06 11:49:04.869283.869283 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01522517204284668s
DEBUG 05-06 11:49:04.869373.869373 cuda_h.py:27] end moe_wait_copy_tasks cost 15.345 ms
DEBUG 05-06 11:49:04.873383.873383 cuda_h.py:27] end moe_vllm_forward cost 3.505 ms
DEBUG 05-06 11:49:04.873858.873858 cuda_h.py:27] end moe_cpu_merge cost 0.056 ms
DEBUG 05-06 11:49:04.873472.873472 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.873335.873335 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.183ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.874792.874792 cuda_h.py:27] end *layer_moe_fused cost 23.600 ms
DEBUG 05-06 11:49:04.879319.879319 cuda_h.py:27] end prefill_merge_scale cost 5.386 ms
DEBUG 05-06 11:49:04.879501.879501 cuda_h.py:27] end prefill_layer cost 33.115 ms
DEBUG 05-06 11:49:04.879985.879985 lmp.py:1391] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 11:49:04.879225.879225 lmp.py:1347] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 11:49:04.880987.880987 cuda_h.py:27] end prefill_ln cost 0.211 ms
DEBUG 05-06 11:49:04.882751.882751 cuda_h.py:27] end prefill_attn cost 1.800 ms
DEBUG 05-06 11:49:04.882031.882031 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 11:49:04.883107.883107 cuda_h.py:27] end prefill_gate cost 0.472 ms
experts_cpu_alloc {'expert_ids': [19, 59, 67, 83, 15, 99, 127, 87, 23, 79, 39, 43, 56, 108, 120, 48, 44, 92, 36, 60, 84, 88, 29, 61, 93, 109, 73, 9, 81, 33, 65, 117, 69, 121, 97, 105, 38, 58, 66, 102, 114, 34, 50, 54, 94, 98, 122, 74, 126], 'token_total': 198, 'token_per_expert': {19: 1, 59: 1, 67: 1, 83: 1, 15: 2, 99: 2, 127: 2, 87: 3, 23: 5, 79: 5, 39: 6, 43: 8, 56: 1, 108: 2, 120: 2, 48: 3, 44: 4, 92: 4, 36: 5, 60: 7, 84: 7, 88: 7, 29: 1, 61: 1, 93: 1, 109: 1, 73: 3, 9: 4, 81: 4, 33: 6, 65: 6, 117: 7, 69: 9, 121: 9, 97: 11, 105: 11, 38: 1, 58: 1, 66: 1, 102: 1, 114: 1, 34: 2, 50: 3, 54: 3, 94: 3, 98: 6, 122: 7, 74: 8, 126: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 47, 55, 71, 75, 91, 95, 111, 115, 119, 123], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 1061, 'token_per_expert': {3: 258, 7: 257, 11: 40, 47: 36, 55: 19, 71: 32, 75: 45, 91: 49, 95: 9, 111: 195, 115: 83, 119: 26, 123: 12}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 32, 40, 52, 68, 76, 100, 104, 112], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 23, 'token_total': 1151, 'token_per_expert': {0: 256, 4: 256, 12: 198, 20: 148, 24: 11, 32: 36, 40: 35, 52: 30, 68: 22, 76: 72, 100: 8, 104: 15, 112: 64}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 37, 49, 53, 57, 77, 85, 89, 101, 113], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 26, 'token_total': 933, 'token_per_expert': {1: 270, 5: 268, 13: 24, 37: 12, 49: 131, 53: 33, 57: 86, 77: 19, 85: 18, 89: 16, 101: 23, 113: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 30, 46, 62, 70, 78, 90, 106, 110], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 25, 'token_total': 753, 'token_per_expert': {2: 256, 6: 257, 18: 10, 22: 33, 30: 18, 46: 22, 62: 9, 70: 32, 78: 19, 90: 48, 106: 12, 110: 37}}
INFO 05-06 11:49:04.884440.884440 lmp.py:1836] [layer_moe_fused] layer=28 prefix: 0.419ms alloc: 0.371ms
INFO 05-06 11:49:04.884637.884637 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.7697296142578125e-05 seconds
INFO 05-06 11:49:04.885669.885669 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.00074005126953125s
INFO 05-06 11:49:04.886609.886609 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005269050598144531 seconds
DEBUG 05-06 11:49:04.886915.886915 cuda_h.py:27] end moe_cpu_prep_submit cost 0.959 ms
INFO 05-06 11:49:04.902744.902744 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015426397323608398s
DEBUG 05-06 11:49:04.902595.902595 cuda_h.py:27] end moe_wait_copy_tasks cost 15.546 ms
DEBUG 05-06 11:49:04.906884.906884 cuda_h.py:27] end moe_vllm_forward cost 3.529 ms
DEBUG 05-06 11:49:04.906359.906359 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:49:04.907834.907834 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.907267.907267 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.216ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.907890.907890 cuda_h.py:27] end *layer_moe_fused cost 23.703 ms
DEBUG 05-06 11:49:04.913376.913376 cuda_h.py:27] end prefill_merge_scale cost 5.354 ms
DEBUG 05-06 11:49:04.913836.913836 cuda_h.py:27] end prefill_layer cost 33.279 ms
DEBUG 05-06 11:49:04.913351.913351 lmp.py:1391] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 11:49:04.913114.913114 lmp.py:1347] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 11:49:04.913100.913100 cuda_h.py:27] end prefill_ln cost 0.200 ms
DEBUG 05-06 11:49:04.915017.915017 cuda_h.py:27] end prefill_attn cost 1.806 ms
DEBUG 05-06 11:49:04.916966.916966 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 11:49:04.917463.917463 cuda_h.py:27] end prefill_gate cost 0.402 ms
experts_cpu_alloc {'expert_ids': [87, 127, 55, 111, 123, 11, 31, 35, 75, 115, 95, 119, 68, 104, 108, 84, 40, 76, 80, 88, 96, 92, 32, 24, 8, 17, 25, 37, 105, 33, 125, 65, 13, 21, 77, 9, 61, 69, 73, 81, 89, 122, 70, 118, 126, 74, 38, 94, 50, 58, 10, 46, 114, 30, 66], 'token_total': 331, 'token_per_expert': {87: 1, 127: 2, 55: 3, 111: 4, 123: 5, 11: 6, 31: 6, 35: 6, 75: 6, 115: 6, 95: 7, 119: 7, 68: 1, 104: 1, 108: 1, 84: 2, 40: 3, 76: 4, 80: 4, 88: 4, 96: 4, 92: 5, 32: 6, 24: 8, 8: 9, 17: 1, 25: 3, 37: 3, 105: 3, 33: 4, 125: 4, 65: 5, 13: 6, 21: 6, 77: 10, 9: 15, 61: 15, 69: 15, 73: 16, 81: 16, 89: 17, 122: 1, 70: 2, 118: 2, 126: 2, 74: 3, 38: 4, 94: 4, 50: 6, 58: 6, 10: 7, 46: 8, 114: 10, 30: 13, 66: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 43, 63, 67, 71, 83, 91, 99, 107], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 1019, 'token_per_expert': {3: 269, 7: 355, 15: 9, 19: 39, 23: 27, 27: 25, 43: 47, 63: 8, 67: 11, 71: 37, 83: 10, 91: 92, 99: 80, 107: 10}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 28, 44, 48, 52, 56, 60, 64, 116, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 993, 'token_per_expert': {0: 256, 4: 300, 16: 30, 20: 53, 28: 43, 44: 14, 48: 26, 52: 75, 56: 43, 60: 24, 64: 77, 116: 13, 120: 13, 124: 26}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 49, 53, 57, 85, 93, 97, 101, 109, 113, 117, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 848, 'token_per_expert': {1: 262, 5: 257, 29: 23, 49: 26, 53: 21, 57: 45, 85: 19, 93: 26, 97: 28, 101: 18, 109: 21, 113: 19, 117: 42, 121: 41}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 42, 54, 62, 78, 82, 86, 90, 106], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 905, 'token_per_expert': {2: 273, 6: 264, 14: 20, 18: 25, 22: 24, 26: 27, 42: 47, 54: 27, 62: 19, 78: 17, 82: 32, 86: 54, 90: 29, 106: 47}}
INFO 05-06 11:49:04.918902.918902 lmp.py:1836] [layer_moe_fused] layer=29 prefix: 0.406ms alloc: 0.398ms
INFO 05-06 11:49:04.918153.918153 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.984306335449219e-05 seconds
INFO 05-06 11:49:04.919819.919819 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007498264312744141s
INFO 05-06 11:49:04.919462.919462 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005538463592529297 seconds
DEBUG 05-06 11:49:04.920147.920147 cuda_h.py:27] end moe_cpu_prep_submit cost 0.946 ms
INFO 05-06 11:49:04.932710.932710 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.011687278747558594s
DEBUG 05-06 11:49:04.932627.932627 cuda_h.py:27] end moe_wait_copy_tasks cost 11.818 ms
DEBUG 05-06 11:49:04.936499.936499 cuda_h.py:27] end moe_vllm_forward cost 3.498 ms
DEBUG 05-06 11:49:04.936649.936649 cuda_h.py:27] end moe_cpu_merge cost 0.057 ms
DEBUG 05-06 11:49:04.936576.936576 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:49:04.936678.936678 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.248ms (seq_len=128 cg=False)
DEBUG 05-06 11:49:04.937560.937560 cuda_h.py:27] end *layer_moe_fused cost 20.043 ms
DEBUG 05-06 11:49:04.938833.938833 cuda_h.py:27] end prefill_merge_scale cost 0.600 ms
DEBUG 05-06 11:49:04.938572.938572 cuda_h.py:27] end prefill_layer cost 24.739 ms
DEBUG 05-06 11:49:04.938990.938990 lmp.py:1391] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 11:49:04.938276.938276 cuda_h.py:27] end prefill_step cost 1096.786 ms
INFO 05-06 11:49:04.938056.938056 lmp.py:1394] prefill time: 1.2219467163085938 seconds
INFO 05-06 11:49:04.944906.944906 lmp.py:1406] Static-KV prefill complete; seqlens set to 128.
WARNING 05-06 11:49:04.945109.945109 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:49:04.945919.945919 helper.py:35]   NaN count (hidden): 720896
WARNING 05-06 11:49:04.946261.946261 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:49:04.946252.946252 helper.py:39]   NaN count (normed): 720896
WARNING 05-06 11:49:04.951410.951410 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:49:04.951753.951753 helper.py:50]   NaN count: 524288
WARNING 05-06 11:49:04.952414.952414 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:49:04.954647.954647 cuda_h.py:27] end init_inputs_tokens cost 8.953 ms
DEBUG 05-06 11:49:04.954941.954941 lmp.py:1507] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:49:04.954653.954653 lmp.py:1513] ---- decode step 0 layer 0 ----
DEBUG 05-06 11:49:04.962339.962339 cuda_h.py:27] end decode_layer cost 8.399 ms
DEBUG 05-06 11:49:04.962202.962202 lmp.py:1513] ---- decode step 0 layer 1 ----
DEBUG 05-06 11:49:04.969986.969986 cuda_h.py:27] end decode_layer cost 6.400 ms
DEBUG 05-06 11:49:04.969333.969333 lmp.py:1513] ---- decode step 0 layer 2 ----
DEBUG 05-06 11:49:04.975980.975980 cuda_h.py:27] end decode_layer cost 6.264 ms
DEBUG 05-06 11:49:04.975651.975651 lmp.py:1513] ---- decode step 0 layer 3 ----
DEBUG 05-06 11:49:04.980502.980502 cuda_h.py:27] end decode_layer cost 5.011 ms
DEBUG 05-06 11:49:04.980868.980868 lmp.py:1513] ---- decode step 0 layer 4 ----
DEBUG 05-06 11:49:04.985856.985856 cuda_h.py:27] end decode_layer cost 4.796 ms
DEBUG 05-06 11:49:04.985123.985123 lmp.py:1513] ---- decode step 0 layer 5 ----
DEBUG 05-06 11:49:04.990564.990564 cuda_h.py:27] end decode_layer cost 5.059 ms
DEBUG 05-06 11:49:04.990361.990361 lmp.py:1513] ---- decode step 0 layer 6 ----
DEBUG 05-06 11:49:04.995929.995929 cuda_h.py:27] end decode_layer cost 4.697 ms
DEBUG 05-06 11:49:04.995911.995911 lmp.py:1513] ---- decode step 0 layer 7 ----
DEBUG 05-06 11:49:05.000107.000107 cuda_h.py:27] end decode_layer cost 4.843 ms
DEBUG 05-06 11:49:05.000711.000711 lmp.py:1513] ---- decode step 0 layer 8 ----
DEBUG 05-06 11:49:05.005859.005859 cuda_h.py:27] end decode_layer cost 4.809 ms
DEBUG 05-06 11:49:05.005510.005510 lmp.py:1513] ---- decode step 0 layer 9 ----
DEBUG 05-06 11:49:05.010680.010680 cuda_h.py:27] end decode_layer cost 4.859 ms
DEBUG 05-06 11:49:05.010430.010430 lmp.py:1513] ---- decode step 0 layer 10 ----
DEBUG 05-06 11:49:05.014772.014772 cuda_h.py:27] end decode_layer cost 4.671 ms
DEBUG 05-06 11:49:05.015900.015900 lmp.py:1513] ---- decode step 0 layer 11 ----
DEBUG 05-06 11:49:05.020794.020794 cuda_h.py:27] end decode_layer cost 5.148 ms
DEBUG 05-06 11:49:05.020591.020591 lmp.py:1513] ---- decode step 0 layer 12 ----
DEBUG 05-06 11:49:05.025733.025733 cuda_h.py:27] end decode_layer cost 5.225 ms
DEBUG 05-06 11:49:05.025338.025338 lmp.py:1513] ---- decode step 0 layer 13 ----
DEBUG 05-06 11:49:05.030348.030348 cuda_h.py:27] end decode_layer cost 4.847 ms
DEBUG 05-06 11:49:05.030383.030383 lmp.py:1513] ---- decode step 0 layer 14 ----
DEBUG 05-06 11:49:05.035140.035140 cuda_h.py:27] end decode_layer cost 4.801 ms
DEBUG 05-06 11:49:05.035553.035553 lmp.py:1513] ---- decode step 0 layer 15 ----
DEBUG 05-06 11:49:05.040797.040797 cuda_h.py:27] end decode_layer cost 4.914 ms
DEBUG 05-06 11:49:05.040640.040640 lmp.py:1513] ---- decode step 0 layer 16 ----
DEBUG 05-06 11:49:05.044784.044784 cuda_h.py:27] end decode_layer cost 4.701 ms
DEBUG 05-06 11:49:05.045766.045766 lmp.py:1513] ---- decode step 0 layer 17 ----
DEBUG 05-06 11:49:05.050277.050277 cuda_h.py:27] end decode_layer cost 4.970 ms
DEBUG 05-06 11:49:05.050836.050836 lmp.py:1513] ---- decode step 0 layer 18 ----
DEBUG 05-06 11:49:05.054179.054179 cuda_h.py:27] end decode_layer cost 4.707 ms
DEBUG 05-06 11:49:05.054115.054115 lmp.py:1513] ---- decode step 0 layer 19 ----
DEBUG 05-06 11:49:05.059352.059352 cuda_h.py:27] end decode_layer cost 4.699 ms
DEBUG 05-06 11:49:05.059222.059222 lmp.py:1513] ---- decode step 0 layer 20 ----
DEBUG 05-06 11:49:05.064581.064581 cuda_h.py:27] end decode_layer cost 4.805 ms
DEBUG 05-06 11:49:05.064232.064232 lmp.py:1513] ---- decode step 0 layer 21 ----
DEBUG 05-06 11:49:05.069816.069816 cuda_h.py:27] end decode_layer cost 4.779 ms
DEBUG 05-06 11:49:05.069798.069798 lmp.py:1513] ---- decode step 0 layer 22 ----
DEBUG 05-06 11:49:05.074220.074220 cuda_h.py:27] end decode_layer cost 4.695 ms
DEBUG 05-06 11:49:05.074964.074964 lmp.py:1513] ---- decode step 0 layer 23 ----
DEBUG 05-06 11:49:05.079917.079917 cuda_h.py:27] end decode_layer cost 4.910 ms
DEBUG 05-06 11:49:05.079660.079660 lmp.py:1513] ---- decode step 0 layer 24 ----
DEBUG 05-06 11:49:05.083745.083745 cuda_h.py:27] end decode_layer cost 4.692 ms
DEBUG 05-06 11:49:05.083250.083250 lmp.py:1513] ---- decode step 0 layer 25 ----
DEBUG 05-06 11:49:05.088022.088022 cuda_h.py:27] end decode_layer cost 4.672 ms
DEBUG 05-06 11:49:05.088528.088528 lmp.py:1513] ---- decode step 0 layer 26 ----
DEBUG 05-06 11:49:05.093904.093904 cuda_h.py:27] end decode_layer cost 4.695 ms
DEBUG 05-06 11:49:05.093409.093409 lmp.py:1513] ---- decode step 0 layer 27 ----
DEBUG 05-06 11:49:05.097590.097590 cuda_h.py:27] end decode_layer cost 4.622 ms
DEBUG 05-06 11:49:05.097334.097334 lmp.py:1513] ---- decode step 0 layer 28 ----
DEBUG 05-06 11:49:05.102084.102084 cuda_h.py:27] end decode_layer cost 4.796 ms
DEBUG 05-06 11:49:05.102973.102973 lmp.py:1513] ---- decode step 0 layer 29 ----
DEBUG 05-06 11:49:05.107474.107474 cuda_h.py:27] end decode_layer cost 5.069 ms
DEBUG 05-06 11:49:05.108742.108742 cuda_h.py:27] end decode_step cost 162.929 ms
INFO 05-06 11:49:05.108697.108697 lmp.py:1561] decode step 0 time: 0.16296887397766113 seconds
WARNING 05-06 11:49:05.108616.108616 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:49:05.108465.108465 helper.py:35]   NaN count (hidden): 5632
WARNING 05-06 11:49:05.108803.108803 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:49:05.108416.108416 helper.py:39]   NaN count (normed): 5632
WARNING 05-06 11:49:05.114148.114148 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:49:05.114311.114311 helper.py:50]   NaN count: 524288
WARNING 05-06 11:49:05.114227.114227 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:49:05.115945.115945 cuda_h.py:27] end init_inputs_tokens cost 7.524 ms
DEBUG 05-06 11:49:05.115842.115842 lmp.py:1507] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:49:05.115016.115016 lmp.py:1513] ---- decode step 1 layer 0 ----
DEBUG 05-06 11:49:05.120374.120374 cuda_h.py:27] end decode_layer cost 5.139 ms
DEBUG 05-06 11:49:05.121456.121456 lmp.py:1513] ---- decode step 1 layer 1 ----
DEBUG 05-06 11:49:05.125827.125827 cuda_h.py:27] end decode_layer cost 4.763 ms
DEBUG 05-06 11:49:05.125763.125763 lmp.py:1513] ---- decode step 1 layer 2 ----
DEBUG 05-06 11:49:05.130534.130534 cuda_h.py:27] end decode_layer cost 4.636 ms
DEBUG 05-06 11:49:05.130377.130377 lmp.py:1513] ---- decode step 1 layer 3 ----
DEBUG 05-06 11:49:05.135796.135796 cuda_h.py:27] end decode_layer cost 4.798 ms
DEBUG 05-06 11:49:05.135447.135447 lmp.py:1513] ---- decode step 1 layer 4 ----
DEBUG 05-06 11:49:05.141961.141961 cuda_h.py:27] end decode_layer cost 5.843 ms
DEBUG 05-06 11:49:05.141917.141917 lmp.py:1513] ---- decode step 1 layer 5 ----
DEBUG 05-06 11:49:05.146635.146635 cuda_h.py:27] end decode_layer cost 5.018 ms
DEBUG 05-06 11:49:05.146140.146140 lmp.py:1513] ---- decode step 1 layer 6 ----
DEBUG 05-06 11:49:05.151181.151181 cuda_h.py:27] end decode_layer cost 4.764 ms
DEBUG 05-06 11:49:05.151137.151137 lmp.py:1513] ---- decode step 1 layer 7 ----
DEBUG 05-06 11:49:05.156018.156018 cuda_h.py:27] end decode_layer cost 4.752 ms
DEBUG 05-06 11:49:05.156384.156384 lmp.py:1513] ---- decode step 1 layer 8 ----
DEBUG 05-06 11:49:05.160244.160244 cuda_h.py:27] end decode_layer cost 4.701 ms
DEBUG 05-06 11:49:05.160657.160657 lmp.py:1513] ---- decode step 1 layer 9 ----
DEBUG 05-06 11:49:05.165868.165868 cuda_h.py:27] end decode_layer cost 4.715 ms
DEBUG 05-06 11:49:05.165326.165326 lmp.py:1513] ---- decode step 1 layer 10 ----
DEBUG 05-06 11:49:05.170809.170809 cuda_h.py:27] end decode_layer cost 4.704 ms
DEBUG 05-06 11:49:05.170175.170175 lmp.py:1513] ---- decode step 1 layer 11 ----
DEBUG 05-06 11:49:05.175814.175814 cuda_h.py:27] end decode_layer cost 5.030 ms
DEBUG 05-06 11:49:05.175465.175465 lmp.py:1513] ---- decode step 1 layer 12 ----
DEBUG 05-06 11:49:05.180369.180369 cuda_h.py:27] end decode_layer cost 4.839 ms
DEBUG 05-06 11:49:05.180212.180212 lmp.py:1513] ---- decode step 1 layer 13 ----
DEBUG 05-06 11:49:05.185317.185317 cuda_h.py:27] end decode_layer cost 4.707 ms
DEBUG 05-06 11:49:05.185106.185106 lmp.py:1513] ---- decode step 1 layer 14 ----
DEBUG 05-06 11:49:05.189533.189533 cuda_h.py:27] end decode_layer cost 4.627 ms
DEBUG 05-06 11:49:05.189376.189376 lmp.py:1513] ---- decode step 1 layer 15 ----
DEBUG 05-06 11:49:05.194451.194451 cuda_h.py:27] end decode_layer cost 4.790 ms
DEBUG 05-06 11:49:05.194717.194717 lmp.py:1513] ---- decode step 1 layer 16 ----
DEBUG 05-06 11:49:05.199711.199711 cuda_h.py:27] end decode_layer cost 4.554 ms
DEBUG 05-06 11:49:05.199667.199667 lmp.py:1513] ---- decode step 1 layer 17 ----
DEBUG 05-06 11:49:05.204227.204227 cuda_h.py:27] end decode_layer cost 5.042 ms
DEBUG 05-06 11:49:05.204878.204878 lmp.py:1513] ---- decode step 1 layer 18 ----
DEBUG 05-06 11:49:05.209272.209272 cuda_h.py:27] end decode_layer cost 4.850 ms
DEBUG 05-06 11:49:05.209162.209162 lmp.py:1513] ---- decode step 1 layer 19 ----
DEBUG 05-06 11:49:05.214138.214138 cuda_h.py:27] end decode_layer cost 4.822 ms
DEBUG 05-06 11:49:05.214405.214405 lmp.py:1513] ---- decode step 1 layer 20 ----
DEBUG 05-06 11:49:05.218353.218353 cuda_h.py:27] end decode_layer cost 4.591 ms
DEBUG 05-06 11:49:05.218859.218859 lmp.py:1513] ---- decode step 1 layer 21 ----
DEBUG 05-06 11:49:05.223645.223645 cuda_h.py:27] end decode_layer cost 4.893 ms
DEBUG 05-06 11:49:05.223058.223058 lmp.py:1513] ---- decode step 1 layer 22 ----
DEBUG 05-06 11:49:05.228470.228470 cuda_h.py:27] end decode_layer cost 4.582 ms
DEBUG 05-06 11:49:05.228975.228975 lmp.py:1513] ---- decode step 1 layer 23 ----
DEBUG 05-06 11:49:05.233626.233626 cuda_h.py:27] end decode_layer cost 5.004 ms
DEBUG 05-06 11:49:05.233846.233846 lmp.py:1513] ---- decode step 1 layer 24 ----
DEBUG 05-06 11:49:05.238500.238500 cuda_h.py:27] end decode_layer cost 4.690 ms
DEBUG 05-06 11:49:05.238920.238920 lmp.py:1513] ---- decode step 1 layer 25 ----
DEBUG 05-06 11:49:05.243507.243507 cuda_h.py:27] end decode_layer cost 5.616 ms
DEBUG 05-06 11:49:05.244488.244488 lmp.py:1513] ---- decode step 1 layer 26 ----
DEBUG 05-06 11:49:05.249632.249632 cuda_h.py:27] end decode_layer cost 5.726 ms
DEBUG 05-06 11:49:05.249475.249475 lmp.py:1513] ---- decode step 1 layer 27 ----
DEBUG 05-06 11:49:05.254665.254665 cuda_h.py:27] end decode_layer cost 4.874 ms
DEBUG 05-06 11:49:05.254005.254005 lmp.py:1513] ---- decode step 1 layer 28 ----
DEBUG 05-06 11:49:05.259449.259449 cuda_h.py:27] end decode_layer cost 4.745 ms
DEBUG 05-06 11:49:05.259339.259339 lmp.py:1513] ---- decode step 1 layer 29 ----
DEBUG 05-06 11:49:05.264701.264701 cuda_h.py:27] end decode_layer cost 5.071 ms
DEBUG 05-06 11:49:05.264267.264267 cuda_h.py:27] end decode_step cost 156.784 ms
INFO 05-06 11:49:05.264983.264983 lmp.py:1561] decode step 1 time: 0.15682315826416016 seconds
WARNING 05-06 11:49:05.265062.265062 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:49:05.265415.265415 helper.py:35]   NaN count (hidden): 5632
WARNING 05-06 11:49:05.265161.265161 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:49:05.265080.265080 helper.py:39]   NaN count (normed): 5632
WARNING 05-06 11:49:05.270177.270177 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:49:05.271094.271094 helper.py:50]   NaN count: 524288
WARNING 05-06 11:49:05.271818.271818 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 11:49:05.271931.271931 helper.py:80] WARNING: Logits have extreme values: min=-788.00, max=1160.00
WARNING 05-06 11:49:05.271570.271570 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 11:49:05.273592.273592 cuda_h.py:27] end init_inputs_tokens cost 8.040 ms
DEBUG 05-06 11:49:05.273720.273720 lmp.py:1507] decode step 2 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:49:05.273390.273390 lmp.py:1513] ---- decode step 2 layer 0 ----
DEBUG 05-06 11:49:05.277181.277181 cuda_h.py:27] end decode_layer cost 4.824 ms
DEBUG 05-06 11:49:05.278263.278263 lmp.py:1513] ---- decode step 2 layer 1 ----
DEBUG 05-06 11:49:05.282511.282511 cuda_h.py:27] end decode_layer cost 4.847 ms
DEBUG 05-06 11:49:05.282593.282593 lmp.py:1513] ---- decode step 2 layer 2 ----
DEBUG 05-06 11:49:05.287538.287538 cuda_h.py:27] end decode_layer cost 4.694 ms
DEBUG 05-06 11:49:05.287474.287474 lmp.py:1513] ---- decode step 2 layer 3 ----
DEBUG 05-06 11:49:05.292035.292035 cuda_h.py:27] end decode_layer cost 4.867 ms
DEBUG 05-06 11:49:05.292408.292408 lmp.py:1513] ---- decode step 2 layer 4 ----
DEBUG 05-06 11:49:05.297654.297654 cuda_h.py:27] end decode_layer cost 4.776 ms
DEBUG 05-06 11:49:05.297020.297020 lmp.py:1513] ---- decode step 2 layer 5 ----
DEBUG 05-06 11:49:05.302960.302960 cuda_h.py:27] end decode_layer cost 5.112 ms
DEBUG 05-06 11:49:05.302564.302564 lmp.py:1513] ---- decode step 2 layer 6 ----
DEBUG 05-06 11:49:05.307697.307697 cuda_h.py:27] end decode_layer cost 4.762 ms
DEBUG 05-06 11:49:05.307587.307587 lmp.py:1513] ---- decode step 2 layer 7 ----
DEBUG 05-06 11:49:05.312548.312548 cuda_h.py:27] end decode_layer cost 5.373 ms
DEBUG 05-06 11:49:05.312968.312968 lmp.py:1513] ---- decode step 2 layer 8 ----
DEBUG 05-06 11:49:05.317210.317210 cuda_h.py:27] end decode_layer cost 5.054 ms
DEBUG 05-06 11:49:05.318099.318099 lmp.py:1513] ---- decode step 2 layer 9 ----
DEBUG 05-06 11:49:05.323531.323531 cuda_h.py:27] end decode_layer cost 4.982 ms
DEBUG 05-06 11:49:05.323282.323282 lmp.py:1513] ---- decode step 2 layer 10 ----
DEBUG 05-06 11:49:05.327451.327451 cuda_h.py:27] end decode_layer cost 4.859 ms
DEBUG 05-06 11:49:05.328102.328102 lmp.py:1513] ---- decode step 2 layer 11 ----
DEBUG 05-06 11:49:05.333266.333266 cuda_h.py:27] end decode_layer cost 5.101 ms
DEBUG 05-06 11:49:05.333633.333633 lmp.py:1513] ---- decode step 2 layer 12 ----
DEBUG 05-06 11:49:05.337679.337679 cuda_h.py:27] end decode_layer cost 4.733 ms
DEBUG 05-06 11:49:05.337661.337661 lmp.py:1513] ---- decode step 2 layer 13 ----
DEBUG 05-06 11:49:05.342251.342251 cuda_h.py:27] end decode_layer cost 4.959 ms
DEBUG 05-06 11:49:05.343379.343379 lmp.py:1513] ---- decode step 2 layer 14 ----
DEBUG 05-06 11:49:05.347850.347850 cuda_h.py:27] end decode_layer cost 4.766 ms
DEBUG 05-06 11:49:05.347501.347501 lmp.py:1513] ---- decode step 2 layer 15 ----
DEBUG 05-06 11:49:05.352750.352750 cuda_h.py:27] end decode_layer cost 5.058 ms
DEBUG 05-06 11:49:05.352685.352685 lmp.py:1513] ---- decode step 2 layer 16 ----
DEBUG 05-06 11:49:05.357275.357275 cuda_h.py:27] end decode_layer cost 4.748 ms
DEBUG 05-06 11:49:05.357496.357496 lmp.py:1513] ---- decode step 2 layer 17 ----
DEBUG 05-06 11:49:05.362021.362021 cuda_h.py:27] end decode_layer cost 5.016 ms
DEBUG 05-06 11:49:05.362911.362911 lmp.py:1513] ---- decode step 2 layer 18 ----
DEBUG 05-06 11:49:05.367329.367329 cuda_h.py:27] end decode_layer cost 4.761 ms
DEBUG 05-06 11:49:05.367669.367669 lmp.py:1513] ---- decode step 2 layer 19 ----
DEBUG 05-06 11:49:05.372503.372503 cuda_h.py:27] end decode_layer cost 4.928 ms
DEBUG 05-06 11:49:05.372392.372392 lmp.py:1513] ---- decode step 2 layer 20 ----
DEBUG 05-06 11:49:05.377480.377480 cuda_h.py:27] end decode_layer cost 4.799 ms
DEBUG 05-06 11:49:05.377562.377562 lmp.py:1513] ---- decode step 2 layer 21 ----
DEBUG 05-06 11:49:05.382753.382753 cuda_h.py:27] end decode_layer cost 4.910 ms
DEBUG 05-06 11:49:05.382172.382172 lmp.py:1513] ---- decode step 2 layer 22 ----
DEBUG 05-06 11:49:05.387685.387685 cuda_h.py:27] end decode_layer cost 4.832 ms
DEBUG 05-06 11:49:05.387383.387383 lmp.py:1513] ---- decode step 2 layer 23 ----
DEBUG 05-06 11:49:05.392509.392509 cuda_h.py:27] end decode_layer cost 5.143 ms
DEBUG 05-06 11:49:05.392544.392544 lmp.py:1513] ---- decode step 2 layer 24 ----
DEBUG 05-06 11:49:05.397669.397669 cuda_h.py:27] end decode_layer cost 4.721 ms
DEBUG 05-06 11:49:05.397843.397843 lmp.py:1513] ---- decode step 2 layer 25 ----
DEBUG 05-06 11:49:05.402230.402230 cuda_h.py:27] end decode_layer cost 4.809 ms
DEBUG 05-06 11:49:05.402496.402496 lmp.py:1513] ---- decode step 2 layer 26 ----
DEBUG 05-06 11:49:05.407828.407828 cuda_h.py:27] end decode_layer cost 4.768 ms
DEBUG 05-06 11:49:05.407334.407334 lmp.py:1513] ---- decode step 2 layer 27 ----
DEBUG 05-06 11:49:05.412491.412491 cuda_h.py:27] end decode_layer cost 4.885 ms
DEBUG 05-06 11:49:05.412572.412572 lmp.py:1513] ---- decode step 2 layer 28 ----
DEBUG 05-06 11:49:05.416512.416512 cuda_h.py:27] end decode_layer cost 4.725 ms
DEBUG 05-06 11:49:05.416355.416355 lmp.py:1513] ---- decode step 2 layer 29 ----
DEBUG 05-06 11:49:05.421381.421381 cuda_h.py:27] end decode_layer cost 5.104 ms
DEBUG 05-06 11:49:05.422211.422211 cuda_h.py:27] end decode_step cost 157.062 ms
INFO 05-06 11:49:05.422259.422259 lmp.py:1561] decode step 2 time: 0.15709948539733887 seconds
Time taken: 5.441812496632338 seconds
X512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x5f37be761e80, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
CPUInfer[0x5f37a6ebdfb0]: Goodbye
CPUInfer[0x5f378c247d50]: Goodbye
