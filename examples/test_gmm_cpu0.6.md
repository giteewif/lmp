here pin
INFO 05-06 11:55:08.039672.039672 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-06 11:55:08.597975.597975 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-06 11:55:09.022836.022836 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-06 11:55:09.022736.022736 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 0.983s
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
INFO 05-06 11:55:16.569312.569312 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-06 11:55:16.976116.976116 cuda_h.py:27] end init_cmv_hmv cost 408.207 ms
DEBUG 05-06 11:55:16.986876.986876 cuda_memory_view.py:1366] 
DEBUG 05-06 11:55:16.986876.986876 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.00266265869140625
DEBUG 05-06 11:55:17.001781.001781 mlpmodule.py:993] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-06 11:55:17.002700.002700 cuda_memory_view.py:1370] 
DEBUG 05-06 11:55:17.002700.002700 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.015807628631591797
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-06 11:55:18.905115.905115 lmp.py:255] init kt-kernel layer 0 ok
INFO 05-06 11:55:19.673708.673708 lmp.py:255] init kt-kernel layer 1 ok
INFO 05-06 11:55:20.470819.470819 lmp.py:255] init kt-kernel layer 2 ok
INFO 05-06 11:55:21.280801.280801 lmp.py:255] init kt-kernel layer 3 ok
INFO 05-06 11:55:22.096385.096385 lmp.py:255] init kt-kernel layer 4 ok
INFO 05-06 11:55:22.901832.901832 lmp.py:255] init kt-kernel layer 5 ok
INFO 05-06 11:55:23.712117.712117 lmp.py:255] init kt-kernel layer 6 ok
INFO 05-06 11:55:24.545198.545198 lmp.py:255] init kt-kernel layer 7 ok
INFO 05-06 11:55:25.370126.370126 lmp.py:255] init kt-kernel layer 8 ok
INFO 05-06 11:55:26.202606.202606 lmp.py:255] init kt-kernel layer 9 ok
INFO 05-06 11:55:27.031347.031347 lmp.py:255] init kt-kernel layer 10 ok
INFO 05-06 11:55:27.860463.860463 lmp.py:255] init kt-kernel layer 11 ok
INFO 05-06 11:55:28.681433.681433 lmp.py:255] init kt-kernel layer 12 ok
INFO 05-06 11:55:29.519399.519399 lmp.py:255] init kt-kernel layer 13 ok
INFO 05-06 11:55:30.332399.332399 lmp.py:255] init kt-kernel layer 14 ok
INFO 05-06 11:55:31.163128.163128 lmp.py:255] init kt-kernel layer 15 ok
INFO 05-06 11:55:31.986028.986028 lmp.py:255] init kt-kernel layer 16 ok
INFO 05-06 11:55:32.804828.804828 lmp.py:255] init kt-kernel layer 17 ok
INFO 05-06 11:55:33.618698.618698 lmp.py:255] init kt-kernel layer 18 ok
INFO 05-06 11:55:34.447732.447732 lmp.py:255] init kt-kernel layer 19 ok
INFO 05-06 11:55:35.277093.277093 lmp.py:255] init kt-kernel layer 20 ok
INFO 05-06 11:55:36.105653.105653 lmp.py:255] init kt-kernel layer 21 ok
INFO 05-06 11:55:36.941003.941003 lmp.py:255] init kt-kernel layer 22 ok
CPUInfer[0x62d8fba272b0]: Hello
WorkerPool[0x62d8fba4f5b0] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x62d912f33e40]: Hello
WorkerPool[0x62d912f5a080] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVINFO 05-06 11:55:37.762278.762278 lmp.py:255] init kt-kernel layer 23 ok
INFO 05-06 11:55:38.611998.611998 lmp.py:255] init kt-kernel layer 24 ok
INFO 05-06 11:55:39.414410.414410 lmp.py:255] init kt-kernel layer 25 ok
INFO 05-06 11:55:40.227875.227875 lmp.py:255] init kt-kernel layer 26 ok
INFO 05-06 11:55:41.038234.038234 lmp.py:255] init kt-kernel layer 27 ok
INFO 05-06 11:55:41.834591.834591 lmp.py:255] init kt-kernel layer 28 ok
INFO 05-06 11:55:42.628835.628835 lmp.py:255] init kt-kernel layer 29 ok
INFO 05-06 11:55:43.503023.503023 lmp.py:186] vLLM Triton fused-MoE enabled (CUDAGraph=False).
generate input ids cost 0.05070781707763672 s
DEBUG 05-06 11:55:46.456126.456126 cuda_h.py:27] end generate_input_ids cost 2933.498 ms
DEBUG 05-06 11:55:46.456575.456575 cuda_h.py:27] end init_cache cost 0.038 ms
INFO 05-06 11:55:46.466693.466693 lmp.py:367] _ensure_static_kv_cache (Gemma4 list): 30 layers, 1760.0 MiB on cuda:0
INFO 05-06 11:55:46.466611.466611 lmp.py:1160] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 11:55:46.479925.479925 lmp.py:2794] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 4784365508, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.787078263565146, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-06 11:55:46.479538.479538 lmp.py:2812] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.479798.479798 lmp.py:2812] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.479190.479190 lmp.py:2812] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.480960.480960 lmp.py:2812] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.480380.480380 lmp.py:2812] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.480818.480818 lmp.py:2812] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.480726.480726 lmp.py:2812] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.480641.480641 lmp.py:2812] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.480403.480403 lmp.py:2812] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.480186.480186 lmp.py:2812] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.481563.481563 lmp.py:2812] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.481856.481856 lmp.py:2812] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.481438.481438 lmp.py:2812] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.481731.481731 lmp.py:2812] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.481784.481784 lmp.py:2812] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.481839.481839 lmp.py:2812] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.481713.481713 lmp.py:2812] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.481621.481621 lmp.py:2812] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.481541.481541 lmp.py:2812] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482788.482788 lmp.py:2812] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482642.482642 lmp.py:2812] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482836.482836 lmp.py:2812] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482410.482410 lmp.py:2812] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482080.482080 lmp.py:2812] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482171.482171 lmp.py:2812] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482795.482795 lmp.py:2812] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482700.482700 lmp.py:2812] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482562.482562 lmp.py:2812] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.482383.482383 lmp.py:2812] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:46.483815.483815 lmp.py:2812] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 11:55:46.780920.780920 cuda_h.py:27] end init_loading_placement cost 314.140 ms
DEBUG 05-06 11:55:46.780950.780950 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:55:46.781263.781263 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:55:46 client.py:72] load_into_gpu: gemma4-26B-A4B, ff04038b-4750-4969-ad01-f662c1fbb325
INFO 05-06 11:55:46 client.py:135] Model loaded: gemma4-26B-A4B, ff04038b-4750-4969-ad01-f662c1fbb325
INFO 05-06 11:55:46 client.py:204] confirm_model_loaded: gemma4-26B-A4B, ff04038b-4750-4969-ad01-f662c1fbb325
INFO 05-06 11:55:47 client.py:212] Model loaded
DEBUG 05-06 11:55:47.309972.309972 cuda_h.py:27] end init_general_sagl_loading_async cost 528.338 ms
INFO 05-06 11:55:47.359456.359456 lmp.py:3315] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 11:55:47.464300.464300 cuda_h.py:27] end restore_state_dict cost 104.726 ms
WARNING 05-06 11:55:47 [fused_moe.py:1090] Using default MoE config. Performance might be sub-optimal! Config file not found at /mnt/zhengcf3/lmp_env/fslmp/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=32,N=704,device_name=NVIDIA_GeForce_RTX_4090.json
INFO 05-06 11:55:48.631179.631179 lmp.py:1288] vLLM Triton pre-warmup done in 1166.4 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 11:55:48.631318.631318 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:55:48.631320.631320 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:55:48 client.py:72] load_into_gpu: gemma4-26B-A4B, 2074fe66-59ba-4ea7-9955-0b2d89d90508
INFO 05-06 11:55:48 client.py:135] Model loaded: gemma4-26B-A4B, 2074fe66-59ba-4ea7-9955-0b2d89d90508
DEBUG 05-06 11:55:48.706415.706415 cuda_h.py:27] end init_experts_loading_async cost 74.494 ms
DEBUG 05-06 11:55:48.736863.736863 cuda_h.py:27] end init_inputs_tokens cost 29.821 ms
DEBUG 05-06 11:55:48.736184.736184 lmp.py:1347] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 11:55:48.804750.804750 cuda_h.py:27] end prefill_ln cost 68.428 ms
DEBUG 05-06 11:55:48.883641.883641 cuda_h.py:27] end prefill_attn cost 78.624 ms
DEBUG 05-06 11:55:48.884436.884436 cuda_h.py:27] end prefill_ffn_prep cost 0.447 ms
DEBUG 05-06 11:55:48.965394.965394 cuda_h.py:27] end prefill_gate cost 74.909 ms
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 75, 115, 83, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 92, 116, 101, 109, 85, 49, 45, 65, 93, 69, 5, 13, 9, 73, 77, 37, 89, 25, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18, 70, 110, 118], 'token_total': 690, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 75: 29, 115: 29, 83: 33, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 92: 16, 116: 18, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 13: 16, 9: 17, 73: 19, 77: 19, 37: 20, 89: 20, 25: 24, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18, 70: 25, 110: 27, 118: 29}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 51, 55, 67, 91, 103, 127], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 1125, 'token_per_expert': {3: 46, 7: 95, 31: 34, 39: 176, 47: 318, 51: 48, 55: 51, 67: 47, 91: 99, 103: 178, 127: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 32, 48, 52, 64, 68, 72, 104, 112, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 724, 'token_per_expert': {0: 73, 16: 48, 32: 43, 48: 41, 52: 43, 64: 27, 68: 170, 72: 35, 104: 43, 112: 23, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 33, 41, 53, 105, 113, 117, 121, 125], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 26, 'token_total': 745, 'token_per_expert': {1: 75, 21: 48, 33: 210, 41: 27, 53: 205, 105: 24, 113: 39, 117: 26, 121: 65, 125: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 46, 50, 54, 74, 78, 90, 122, 126], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 25, 'token_total': 812, 'token_per_expert': {22: 64, 26: 59, 46: 119, 50: 110, 54: 59, 74: 61, 78: 36, 90: 154, 122: 35, 126: 115}}
INFO 05-06 11:55:48.995612.995612 lmp.py:1836] [layer_moe_fused] layer=0 prefix: 29.295ms alloc: 0.298ms
INFO 05-06 11:55:48.995485.995485 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 3.6716461181640625e-05 seconds
INFO 05-06 11:55:48.997001.997001 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001912832260131836s
INFO 05-06 11:55:48.998144.998144 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008218288421630859 seconds
DEBUG 05-06 11:55:49.011925.011925 cuda_h.py:27] end moe_cpu_prep_submit cost 13.839 ms
INFO 05-06 11:55:49.015912.015912 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0034759044647216797s
DEBUG 05-06 11:55:49.015131.015131 cuda_h.py:27] end moe_wait_copy_tasks cost 3.892 ms
DEBUG 05-06 11:55:49.048555.048555 cuda_h.py:27] end moe_vllm_forward cost 32.358 ms
DEBUG 05-06 11:55:49.078570.078570 cuda_h.py:27] end moe_cpu_merge cost 30.202 ms
DEBUG 05-06 11:55:49.079844.079844 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:49.079635.079635 lmp.py:1950] [layer_moe_fused] vllm triton time: 63.254ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.079461.079461 cuda_h.py:27] end *layer_moe_fused cost 113.316 ms
DEBUG 05-06 11:55:49.081987.081987 cuda_h.py:27] end prefill_merge_scale cost 1.801 ms
DEBUG 05-06 11:55:49.081884.081884 cuda_h.py:27] end prefill_layer cost 345.077 ms
DEBUG 05-06 11:55:49.081645.081645 lmp.py:1391] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 11:55:49.081699.081699 lmp.py:1347] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 11:55:49.082594.082594 cuda_h.py:27] end prefill_ln cost 0.264 ms
DEBUG 05-06 11:55:49.085379.085379 cuda_h.py:27] end prefill_attn cost 3.596 ms
DEBUG 05-06 11:55:49.086341.086341 cuda_h.py:27] end prefill_ffn_prep cost 0.545 ms
DEBUG 05-06 11:55:49.087143.087143 cuda_h.py:27] end prefill_gate cost 0.693 ms
experts_cpu_alloc {'expert_ids': [39, 103, 31, 43, 55, 83, 91, 115, 123, 11, 87, 95, 27, 44, 108, 16, 40, 72, 88, 32, 60, 84, 112, 116, 56, 76, 48, 92, 124, 64, 104, 120, 61, 41, 45, 117, 33, 57, 81, 121, 89, 93, 125, 69, 9, 37, 101, 21, 85, 26, 14, 62, 110, 18, 38, 66, 50, 74, 78, 98, 90, 94, 34, 42], 'token_total': 378, 'token_per_expert': {39: 1, 103: 1, 31: 2, 43: 2, 55: 3, 83: 3, 91: 3, 115: 3, 123: 3, 11: 6, 87: 7, 95: 7, 27: 9, 44: 1, 108: 1, 16: 2, 40: 2, 72: 2, 88: 2, 32: 3, 60: 3, 84: 3, 112: 4, 116: 5, 56: 6, 76: 7, 48: 8, 92: 8, 124: 9, 64: 12, 104: 15, 120: 16, 61: 1, 41: 2, 45: 2, 117: 2, 33: 3, 57: 3, 81: 3, 121: 3, 89: 4, 93: 5, 125: 5, 69: 6, 9: 7, 37: 7, 101: 7, 21: 9, 85: 11, 26: 1, 14: 2, 62: 2, 110: 2, 18: 3, 38: 4, 66: 6, 50: 7, 74: 11, 78: 11, 98: 13, 90: 14, 94: 16, 34: 23, 42: 24}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 47, 51, 59, 67, 79, 99, 119, 127], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 24, 'token_total': 745, 'token_per_expert': {3: 200, 7: 209, 35: 16, 47: 36, 51: 43, 59: 17, 67: 87, 79: 14, 99: 78, 119: 15, 127: 30}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 28, 52, 68, 80, 96, 100], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 30, 'token_total': 1012, 'token_per_expert': {0: 198, 4: 200, 8: 78, 12: 33, 20: 43, 28: 31, 52: 203, 68: 139, 80: 32, 96: 27, 100: 28}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 49, 53, 65, 73, 97, 105, 109], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 925, 'token_per_expert': {1: 214, 5: 261, 13: 166, 25: 31, 49: 19, 53: 28, 65: 27, 73: 24, 97: 81, 105: 15, 109: 59}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 30, 46, 54, 82, 106, 118, 122], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 1036, 'token_per_expert': {2: 195, 6: 194, 10: 110, 22: 79, 30: 129, 46: 29, 54: 28, 82: 125, 106: 29, 118: 42, 122: 76}}
INFO 05-06 11:55:49.089636.089636 lmp.py:1836] [layer_moe_fused] layer=1 prefix: 0.502ms alloc: 0.416ms
INFO 05-06 11:55:49.089343.089343 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 4.935264587402344e-05 seconds
INFO 05-06 11:55:49.090620.090620 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0013589859008789062s
INFO 05-06 11:55:49.091394.091394 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007131099700927734 seconds
DEBUG 05-06 11:55:49.092022.092022 cuda_h.py:27] end moe_cpu_prep_submit cost 1.282 ms
INFO 05-06 11:55:49.094435.094435 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0021665096282958984s
DEBUG 05-06 11:55:49.094320.094320 cuda_h.py:27] end moe_wait_copy_tasks cost 2.306 ms
DEBUG 05-06 11:55:49.103942.103942 cuda_h.py:27] end moe_vllm_forward cost 8.206 ms
DEBUG 05-06 11:55:49.103258.103258 cuda_h.py:27] end moe_cpu_merge cost 0.068 ms
DEBUG 05-06 11:55:49.103718.103718 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:49.103527.103527 lmp.py:1950] [layer_moe_fused] vllm triton time: 9.301ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.104827.104827 cuda_h.py:27] end *layer_moe_fused cost 15.914 ms
DEBUG 05-06 11:55:49.109232.109232 cuda_h.py:27] end prefill_merge_scale cost 4.804 ms
DEBUG 05-06 11:55:49.109467.109467 cuda_h.py:27] end prefill_layer cost 27.670 ms
DEBUG 05-06 11:55:49.109363.109363 lmp.py:1391] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 11:55:49.109178.109178 lmp.py:1347] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 11:55:49.110478.110478 cuda_h.py:27] end prefill_ln cost 0.258 ms
DEBUG 05-06 11:55:49.112668.112668 cuda_h.py:27] end prefill_attn cost 2.246 ms
DEBUG 05-06 11:55:49.112257.112257 cuda_h.py:27] end prefill_ffn_prep cost 0.489 ms
DEBUG 05-06 11:55:49.114638.114638 cuda_h.py:27] end prefill_gate cost 0.531 ms
experts_cpu_alloc {'expert_ids': [27, 95, 111, 115, 23, 99, 63, 119, 123, 103, 35, 31, 43, 51, 71, 40, 96, 124, 88, 116, 120, 52, 72, 36, 100, 80, 8, 24, 56, 64, 44, 45, 61, 113, 69, 21, 77, 105, 33, 121, 85, 57, 17, 49, 53, 65, 109, 26, 114, 42, 50, 70, 82, 126, 58, 98, 46, 110, 122, 78, 14], 'token_total': 487, 'token_per_expert': {27: 2, 95: 2, 111: 3, 115: 3, 23: 4, 99: 4, 63: 7, 119: 7, 123: 7, 103: 8, 35: 10, 31: 11, 43: 12, 51: 13, 71: 13, 40: 1, 96: 2, 124: 2, 88: 3, 116: 3, 120: 3, 52: 5, 72: 6, 36: 7, 100: 7, 80: 8, 8: 9, 24: 9, 56: 9, 64: 9, 44: 10, 45: 1, 61: 3, 113: 3, 69: 4, 21: 5, 77: 5, 105: 6, 33: 8, 121: 8, 85: 9, 57: 13, 17: 14, 49: 14, 53: 18, 65: 18, 109: 25, 26: 1, 114: 1, 42: 2, 50: 3, 70: 4, 82: 5, 126: 6, 58: 11, 98: 12, 46: 13, 110: 18, 122: 18, 78: 19, 14: 21}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 55, 59, 83, 91, 107, 127], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 960, 'token_per_expert': {3: 264, 7: 286, 11: 96, 15: 57, 19: 49, 55: 32, 59: 63, 83: 16, 91: 21, 107: 13, 127: 63}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 28, 48, 60, 76, 84, 104, 108], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 26, 'token_total': 802, 'token_per_expert': {0: 262, 4: 262, 20: 13, 28: 12, 48: 22, 60: 23, 76: 44, 84: 38, 104: 22, 108: 104}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 29, 37, 41, 81, 97, 125], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 26, 'token_total': 956, 'token_per_expert': {1: 331, 5: 257, 9: 53, 13: 43, 29: 44, 37: 36, 41: 85, 81: 36, 97: 26, 125: 45}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 34, 54, 62, 90, 102, 106, 118], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 24, 'token_total': 891, 'token_per_expert': {2: 256, 6: 257, 18: 41, 34: 26, 54: 86, 62: 79, 90: 44, 102: 49, 106: 21, 118: 32}}
INFO 05-06 11:55:49.115976.115976 lmp.py:1836] [layer_moe_fused] layer=2 prefix: 0.485ms alloc: 0.394ms
INFO 05-06 11:55:49.115782.115782 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.078315734863281e-05 seconds
INFO 05-06 11:55:49.116243.116243 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008974075317382812s
INFO 05-06 11:55:49.117331.117331 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005927085876464844 seconds
DEBUG 05-06 11:55:49.117016.117016 cuda_h.py:27] end moe_cpu_prep_submit cost 1.053 ms
INFO 05-06 11:55:49.119334.119334 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017616748809814453s
DEBUG 05-06 11:55:49.120768.120768 cuda_h.py:27] end moe_wait_copy_tasks cost 1.884 ms
DEBUG 05-06 11:55:49.128883.128883 cuda_h.py:27] end moe_vllm_forward cost 8.148 ms
DEBUG 05-06 11:55:49.128007.128007 cuda_h.py:27] end moe_cpu_merge cost 0.067 ms
DEBUG 05-06 11:55:49.129254.129254 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:49.129401.129401 lmp.py:1950] [layer_moe_fused] vllm triton time: 9.157ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.129860.129860 cuda_h.py:27] end *layer_moe_fused cost 15.016 ms
DEBUG 05-06 11:55:49.135574.135574 cuda_h.py:27] end prefill_merge_scale cost 5.836 ms
DEBUG 05-06 11:55:49.135425.135425 cuda_h.py:27] end prefill_layer cost 25.952 ms
DEBUG 05-06 11:55:49.135320.135320 lmp.py:1391] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 11:55:49.135758.135758 lmp.py:1347] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 11:55:49.136375.136375 cuda_h.py:27] end prefill_ln cost 0.263 ms
DEBUG 05-06 11:55:49.138236.138236 cuda_h.py:27] end prefill_attn cost 2.486 ms
DEBUG 05-06 11:55:49.139795.139795 cuda_h.py:27] end prefill_ffn_prep cost 0.601 ms
DEBUG 05-06 11:55:49.141444.141444 cuda_h.py:27] end prefill_gate cost 0.663 ms
experts_cpu_alloc {'expert_ids': [23, 35, 55, 91, 27, 31, 63, 43, 67, 119, 59, 107, 111, 19, 20, 72, 32, 36, 16, 60, 100, 8, 40, 48, 24, 64, 116, 56, 44, 88, 108, 65, 29, 57, 89, 117, 41, 13, 33, 101, 61, 77, 109, 9, 73, 46, 94, 82, 18, 42, 98, 30, 86, 74, 110, 26, 114, 54, 58, 70, 122, 118, 10], 'token_total': 582, 'token_per_expert': {23: 1, 35: 1, 55: 1, 91: 1, 27: 4, 31: 5, 63: 5, 43: 7, 67: 7, 119: 9, 59: 11, 107: 11, 111: 12, 19: 15, 20: 1, 72: 1, 32: 2, 36: 2, 16: 3, 60: 4, 100: 5, 8: 6, 40: 6, 48: 6, 24: 8, 64: 8, 116: 9, 56: 10, 44: 22, 88: 24, 108: 24, 65: 1, 29: 2, 57: 2, 89: 2, 117: 3, 41: 4, 13: 5, 33: 7, 101: 9, 61: 11, 77: 12, 109: 17, 9: 21, 73: 25, 46: 1, 94: 1, 82: 2, 18: 3, 42: 3, 98: 3, 30: 4, 86: 8, 74: 9, 110: 14, 26: 16, 114: 16, 54: 19, 58: 19, 70: 20, 122: 27, 118: 30, 10: 35}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 39, 51, 71, 75, 83, 95, 123], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 793, 'token_per_expert': {3: 272, 7: 256, 11: 27, 15: 24, 39: 16, 51: 23, 71: 42, 75: 55, 83: 32, 95: 31, 123: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 52, 68, 76, 84, 92, 96, 104, 120], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 881, 'token_per_expert': {0: 275, 4: 282, 28: 55, 52: 39, 68: 32, 76: 28, 84: 46, 92: 36, 96: 28, 104: 28, 120: 32}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 25, 53, 69, 85, 93, 97, 121], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 24, 'token_total': 854, 'token_per_expert': {1: 262, 5: 293, 17: 29, 25: 27, 53: 30, 69: 34, 85: 66, 93: 31, 97: 38, 121: 44}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 34, 50, 62, 66, 78, 102], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 28, 'token_total': 986, 'token_per_expert': {2: 268, 6: 265, 14: 43, 22: 68, 34: 35, 50: 90, 62: 57, 66: 55, 78: 42, 102: 63}}
INFO 05-06 11:55:49.142314.142314 lmp.py:1836] [layer_moe_fused] layer=3 prefix: 0.483ms alloc: 0.398ms
INFO 05-06 11:55:49.142643.142643 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.125999450683594e-05 seconds
INFO 05-06 11:55:49.143419.143419 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008625984191894531s
INFO 05-06 11:55:49.144396.144396 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006160736083984375 seconds
DEBUG 05-06 11:55:49.144696.144696 cuda_h.py:27] end moe_cpu_prep_submit cost 1.069 ms
INFO 05-06 11:55:49.146683.146683 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017428398132324219s
DEBUG 05-06 11:55:49.147554.147554 cuda_h.py:27] end moe_wait_copy_tasks cost 1.874 ms
DEBUG 05-06 11:55:49.155342.155342 cuda_h.py:27] end moe_vllm_forward cost 8.229 ms
DEBUG 05-06 11:55:49.155089.155089 cuda_h.py:27] end moe_cpu_merge cost 0.071 ms
DEBUG 05-06 11:55:49.156403.156403 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:49.156020.156020 lmp.py:1950] [layer_moe_fused] vllm triton time: 9.314ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.156824.156824 cuda_h.py:27] end *layer_moe_fused cost 15.129 ms
DEBUG 05-06 11:55:49.161232.161232 cuda_h.py:27] end prefill_merge_scale cost 4.205 ms
DEBUG 05-06 11:55:49.161845.161845 cuda_h.py:27] end prefill_layer cost 25.132 ms
DEBUG 05-06 11:55:49.161328.161328 lmp.py:1391] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 11:55:49.161905.161905 lmp.py:1347] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 11:55:49.161978.161978 cuda_h.py:27] end prefill_ln cost 0.260 ms
DEBUG 05-06 11:55:49.164388.164388 cuda_h.py:27] end prefill_attn cost 2.260 ms
DEBUG 05-06 11:55:49.164539.164539 cuda_h.py:27] end prefill_ffn_prep cost 0.482 ms
DEBUG 05-06 11:55:49.165032.165032 cuda_h.py:27] end prefill_gate cost 0.515 ms
experts_cpu_alloc {'expert_ids': [31, 79, 103, 107, 75, 91, 19, 47, 123, 15, 87, 39, 71, 51, 55, 115, 12, 120, 56, 44, 80, 64, 108, 116, 36, 40, 84, 88, 52, 28, 96, 121, 21, 45, 69, 97, 109, 17, 37, 77, 81, 101, 117, 61, 73, 25, 57, 105, 30, 58, 66, 114, 18, 122, 126, 34, 90, 94, 86, 118, 38, 62], 'token_total': 408, 'token_per_expert': {31: 1, 79: 1, 103: 1, 107: 6, 75: 8, 91: 8, 19: 9, 47: 9, 123: 9, 15: 10, 87: 13, 39: 14, 71: 17, 51: 24, 55: 27, 115: 27, 12: 2, 120: 2, 56: 3, 44: 4, 80: 4, 64: 5, 108: 5, 116: 5, 36: 6, 40: 6, 84: 7, 88: 9, 52: 10, 28: 11, 96: 13, 121: 1, 21: 3, 45: 3, 69: 3, 97: 3, 109: 3, 17: 4, 37: 4, 77: 4, 81: 4, 101: 4, 117: 4, 61: 6, 73: 7, 25: 8, 57: 9, 105: 11, 30: 1, 58: 1, 66: 1, 114: 1, 18: 2, 122: 3, 126: 3, 34: 4, 90: 4, 94: 4, 86: 6, 118: 6, 38: 7, 62: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 43, 59, 63, 67, 83, 111, 119], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 1109, 'token_per_expert': {3: 331, 7: 321, 23: 49, 27: 34, 43: 59, 59: 50, 63: 108, 67: 28, 83: 29, 111: 39, 119: 61}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 24, 32, 60, 76, 92, 104, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 889, 'token_per_expert': {0: 320, 4: 331, 8: 67, 20: 17, 24: 53, 32: 20, 60: 18, 76: 15, 92: 16, 104: 18, 124: 14}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 49, 53, 85, 89, 93, 113, 125], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 27, 'token_total': 859, 'token_per_expert': {1: 355, 5: 333, 29: 22, 49: 16, 53: 17, 85: 19, 89: 34, 93: 21, 113: 23, 125: 19}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 54, 74, 78, 82, 98, 106], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 24, 'token_total': 831, 'token_per_expert': {2: 320, 6: 324, 22: 29, 26: 20, 54: 19, 74: 43, 78: 9, 82: 18, 98: 13, 106: 36}}
INFO 05-06 11:55:49.167092.167092 lmp.py:1836] [layer_moe_fused] layer=4 prefix: 0.485ms alloc: 0.399ms
INFO 05-06 11:55:49.167375.167375 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.078315734863281e-05 seconds
INFO 05-06 11:55:49.168502.168502 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008428096771240234s
INFO 05-06 11:55:49.168405.168405 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005960464477539062 seconds
DEBUG 05-06 11:55:49.169917.169917 cuda_h.py:27] end moe_cpu_prep_submit cost 0.868 ms
INFO 05-06 11:55:49.171245.171245 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017864704132080078s
DEBUG 05-06 11:55:49.171441.171441 cuda_h.py:27] end moe_wait_copy_tasks cost 1.919 ms
DEBUG 05-06 11:55:49.179952.179952 cuda_h.py:27] end moe_vllm_forward cost 8.245 ms
DEBUG 05-06 11:55:49.180554.180554 cuda_h.py:27] end moe_cpu_merge cost 0.069 ms
DEBUG 05-06 11:55:49.180707.180707 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:49.180067.180067 lmp.py:1950] [layer_moe_fused] vllm triton time: 9.152ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.180745.180745 cuda_h.py:27] end *layer_moe_fused cost 14.430 ms
DEBUG 05-06 11:55:49.186655.186655 cuda_h.py:27] end prefill_merge_scale cost 5.387 ms
DEBUG 05-06 11:55:49.186791.186791 cuda_h.py:27] end prefill_layer cost 24.879 ms
DEBUG 05-06 11:55:49.186446.186446 lmp.py:1391] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 11:55:49.186308.186308 lmp.py:1347] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 11:55:49.186720.186720 cuda_h.py:27] end prefill_ln cost 0.253 ms
DEBUG 05-06 11:55:49.192572.192572 cuda_h.py:27] end prefill_attn cost 5.785 ms
DEBUG 05-06 11:55:49.193027.193027 cuda_h.py:27] end prefill_ffn_prep cost 0.608 ms
DEBUG 05-06 11:55:49.194321.194321 cuda_h.py:27] end prefill_gate cost 0.507 ms
experts_cpu_alloc {'expert_ids': [51, 19, 107, 115, 27, 119, 31, 67, 83, 55, 75, 79, 63, 43, 8, 48, 56, 92, 124, 52, 68, 44, 60, 84, 100, 120, 96, 104, 80, 88, 116, 36, 21, 77, 37, 81, 57, 125, 113, 29, 61, 73, 30, 58, 86, 102, 114, 26, 50, 54, 106, 62, 98, 18, 46, 118], 'token_total': 232, 'token_per_expert': {51: 1, 19: 2, 107: 2, 115: 2, 27: 3, 119: 3, 31: 4, 67: 4, 83: 4, 55: 5, 75: 5, 79: 6, 63: 7, 43: 8, 8: 1, 48: 1, 56: 1, 92: 1, 124: 1, 52: 2, 68: 3, 44: 4, 60: 4, 84: 4, 100: 5, 120: 6, 96: 7, 104: 8, 80: 10, 88: 11, 116: 13, 36: 16, 21: 1, 77: 1, 37: 2, 81: 2, 57: 3, 125: 3, 113: 4, 29: 5, 61: 12, 73: 12, 30: 1, 58: 1, 86: 1, 102: 1, 114: 1, 26: 2, 50: 2, 54: 2, 106: 2, 62: 3, 98: 3, 18: 4, 46: 5, 118: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 71, 87, 99, 111, 123, 127], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 24, 'token_total': 991, 'token_per_expert': {3: 384, 7: 385, 23: 17, 39: 51, 71: 77, 87: 8, 99: 10, 111: 18, 123: 19, 127: 22}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 28, 64, 72, 76, 112], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 28, 'token_total': 1004, 'token_per_expert': {0: 386, 4: 406, 16: 29, 20: 41, 24: 24, 28: 19, 64: 18, 72: 18, 76: 21, 112: 42}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 33, 49, 93, 101, 117], 'expert_count': 9, 'ideal_gpu_count': 9, 'keep_on_gpu': 9, 'hit_count_on_device': 19, 'token_total': 1003, 'token_per_expert': {1: 384, 5: 394, 9: 15, 13: 24, 33: 38, 49: 41, 93: 16, 101: 73, 117: 18}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 42, 70, 74, 94, 126], 'expert_count': 9, 'ideal_gpu_count': 9, 'keep_on_gpu': 9, 'hit_count_on_device': 23, 'token_total': 866, 'token_per_expert': {2: 400, 6: 388, 14: 7, 22: 14, 42: 15, 70: 9, 74: 13, 94: 15, 126: 5}}
INFO 05-06 11:55:49.195786.195786 lmp.py:1836] [layer_moe_fused] layer=5 prefix: 0.476ms alloc: 0.426ms
INFO 05-06 11:55:49.196770.196770 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 4.458427429199219e-05 seconds
INFO 05-06 11:55:49.197551.197551 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008149147033691406s
INFO 05-06 11:55:49.197030.197030 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006015300750732422 seconds
DEBUG 05-06 11:55:49.198435.198435 cuda_h.py:27] end moe_cpu_prep_submit cost 1.041 ms
INFO 05-06 11:55:49.200403.200403 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017859935760498047s
DEBUG 05-06 11:55:49.200552.200552 cuda_h.py:27] end moe_wait_copy_tasks cost 1.915 ms
DEBUG 05-06 11:55:49.208509.208509 cuda_h.py:27] end moe_vllm_forward cost 7.821 ms
DEBUG 05-06 11:55:49.208620.208620 cuda_h.py:27] end moe_cpu_merge cost 0.066 ms
DEBUG 05-06 11:55:49.209043.209043 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:49.209165.209165 lmp.py:1950] [layer_moe_fused] vllm triton time: 8.762ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.209300.209300 cuda_h.py:27] end *layer_moe_fused cost 14.488 ms
DEBUG 05-06 11:55:49.214197.214197 cuda_h.py:27] end prefill_merge_scale cost 5.375 ms
DEBUG 05-06 11:55:49.215856.215856 cuda_h.py:27] end prefill_layer cost 28.632 ms
DEBUG 05-06 11:55:49.215950.215950 lmp.py:1391] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 11:55:49.215528.215528 lmp.py:1347] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 11:55:49.215621.215621 cuda_h.py:27] end prefill_ln cost 0.274 ms
DEBUG 05-06 11:55:49.218234.218234 cuda_h.py:27] end prefill_attn cost 2.205 ms
DEBUG 05-06 11:55:49.218392.218392 cuda_h.py:27] end prefill_ffn_prep cost 0.487 ms
DEBUG 05-06 11:55:49.219715.219715 cuda_h.py:27] end prefill_gate cost 0.561 ms
experts_cpu_alloc {'expert_ids': [111, 11, 127, 43, 51, 103, 91, 71, 75, 95, 123, 27, 16, 60, 72, 76, 112, 124, 40, 80, 20, 116, 28, 32, 44, 33, 89, 101, 37, 81, 73, 125, 113, 41, 85, 105, 57, 69, 77, 22, 30, 74, 82, 114, 38, 58, 110, 126, 10, 42, 122, 14, 70, 50, 78, 26, 46, 90], 'token_total': 259, 'token_per_expert': {111: 1, 11: 2, 127: 4, 43: 5, 51: 5, 103: 5, 91: 6, 71: 7, 75: 7, 95: 7, 123: 7, 27: 8, 16: 1, 60: 1, 72: 1, 76: 1, 112: 1, 124: 1, 40: 2, 80: 3, 20: 4, 116: 4, 28: 5, 32: 7, 44: 7, 33: 1, 89: 1, 101: 1, 37: 2, 81: 2, 73: 3, 125: 3, 113: 4, 41: 5, 85: 5, 105: 5, 57: 6, 69: 8, 77: 11, 22: 1, 30: 1, 74: 1, 82: 1, 114: 1, 38: 2, 58: 2, 110: 3, 126: 3, 10: 4, 42: 4, 122: 5, 14: 6, 70: 6, 50: 10, 78: 11, 26: 13, 46: 13, 90: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 35, 79, 87, 99, 107, 115, 119], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 22, 'token_total': 961, 'token_per_expert': {3: 384, 7: 384, 23: 24, 35: 20, 79: 12, 87: 22, 99: 55, 107: 10, 115: 35, 119: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 36, 56, 64, 68, 96, 104, 108], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 23, 'token_total': 969, 'token_per_expert': {0: 387, 4: 384, 24: 9, 36: 15, 56: 8, 64: 34, 68: 79, 96: 9, 104: 7, 108: 37}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 53, 65, 93, 117, 121], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 24, 'token_total': 984, 'token_per_expert': {1: 392, 5: 391, 9: 14, 13: 19, 25: 48, 53: 30, 65: 22, 93: 44, 117: 12, 121: 12}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34, 62, 86, 94, 98, 102, 106], 'expert_count': 9, 'ideal_gpu_count': 9, 'keep_on_gpu': 9, 'hit_count_on_device': 28, 'token_total': 923, 'token_per_expert': {2: 392, 6: 388, 34: 16, 62: 14, 86: 28, 94: 13, 98: 16, 102: 20, 106: 36}}
INFO 05-06 11:55:49.221920.221920 lmp.py:1836] [layer_moe_fused] layer=6 prefix: 0.478ms alloc: 0.373ms
INFO 05-06 11:55:49.221587.221587 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 4.839897155761719e-05 seconds
INFO 05-06 11:55:49.222276.222276 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008149147033691406s
INFO 05-06 11:55:49.222796.222796 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006315708160400391 seconds
DEBUG 05-06 11:55:49.223627.223627 cuda_h.py:27] end moe_cpu_prep_submit cost 0.923 ms
INFO 05-06 11:55:49.226793.226793 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.00269317626953125s
DEBUG 05-06 11:55:49.226651.226651 cuda_h.py:27] end moe_wait_copy_tasks cost 2.816 ms
DEBUG 05-06 11:55:49.234409.234409 cuda_h.py:27] end moe_vllm_forward cost 7.784 ms
DEBUG 05-06 11:55:49.234189.234189 cuda_h.py:27] end moe_cpu_merge cost 0.069 ms
DEBUG 05-06 11:55:49.235793.235793 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:49.235854.235854 lmp.py:1950] [layer_moe_fused] vllm triton time: 8.770ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.235056.235056 cuda_h.py:27] end *layer_moe_fused cost 15.421 ms
DEBUG 05-06 11:55:49.240083.240083 cuda_h.py:27] end prefill_merge_scale cost 4.696 ms
DEBUG 05-06 11:55:49.240597.240597 cuda_h.py:27] end prefill_layer cost 25.189 ms
DEBUG 05-06 11:55:49.240056.240056 lmp.py:1391] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 11:55:49.240872.240872 lmp.py:1347] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 11:55:49.241369.241369 cuda_h.py:27] end prefill_ln cost 0.259 ms
DEBUG 05-06 11:55:49.244277.244277 cuda_h.py:27] end prefill_attn cost 2.739 ms
DEBUG 05-06 11:55:49.244179.244179 cuda_h.py:27] end prefill_ffn_prep cost 0.539 ms
DEBUG 05-06 11:55:49.246709.246709 cuda_h.py:27] end prefill_gate cost 0.672 ms
experts_cpu_alloc {'expert_ids': [11, 35, 87, 95, 107, 127, 51, 63, 111, 123, 43, 47, 55, 83, 115, 8, 32, 116, 16, 112, 64, 68, 72, 80, 108, 20, 28, 56, 60, 21, 37, 77, 13, 25, 101, 117, 125, 9, 61, 85, 33, 105, 26, 38, 62, 94, 102, 122, 30, 18, 54, 82, 98, 118, 42, 78, 86, 126, 106], 'token_total': 162, 'token_per_expert': {11: 1, 35: 1, 87: 1, 95: 1, 107: 1, 127: 1, 51: 2, 63: 2, 111: 2, 123: 2, 43: 3, 47: 3, 55: 3, 83: 3, 115: 3, 8: 1, 32: 1, 116: 1, 16: 2, 112: 2, 64: 3, 68: 3, 72: 5, 80: 6, 108: 6, 20: 7, 28: 7, 56: 7, 60: 7, 21: 1, 37: 1, 77: 1, 13: 2, 25: 2, 101: 2, 117: 2, 125: 2, 9: 3, 61: 3, 85: 3, 33: 4, 105: 5, 26: 1, 38: 1, 62: 1, 94: 1, 102: 1, 122: 1, 30: 2, 18: 3, 54: 3, 82: 3, 98: 3, 118: 3, 42: 4, 78: 4, 86: 4, 126: 4, 106: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 59, 71, 79, 91, 99, 103], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 25, 'token_total': 956, 'token_per_expert': {3: 448, 7: 450, 19: 5, 23: 5, 59: 6, 71: 4, 79: 7, 91: 18, 99: 6, 103: 7}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 44, 48, 52, 84, 96, 104, 120], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 24, 'token_total': 994, 'token_per_expert': {0: 449, 4: 458, 12: 17, 44: 10, 48: 8, 52: 9, 84: 10, 96: 14, 104: 7, 120: 12}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 53, 57, 65, 69, 97, 113, 121], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 23, 'token_total': 1007, 'token_per_expert': {1: 448, 5: 451, 29: 6, 53: 6, 57: 6, 65: 12, 69: 12, 97: 34, 113: 8, 121: 24}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 34, 70, 90, 110, 114], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 27, 'token_total': 977, 'token_per_expert': {2: 448, 6: 451, 10: 16, 14: 7, 22: 6, 34: 9, 70: 15, 90: 10, 110: 7, 114: 8}}
INFO 05-06 11:55:49.247731.247731 lmp.py:1836] [layer_moe_fused] layer=7 prefix: 0.473ms alloc: 0.378ms
INFO 05-06 11:55:49.247180.247180 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.341934204101562e-05 seconds
INFO 05-06 11:55:49.248828.248828 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007662773132324219s
INFO 05-06 11:55:49.249809.249809 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005488395690917969 seconds
DEBUG 05-06 11:55:49.249881.249881 cuda_h.py:27] end moe_cpu_prep_submit cost 0.687 ms
INFO 05-06 11:55:49.251569.251569 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001650094985961914s
DEBUG 05-06 11:55:49.251713.251713 cuda_h.py:27] end moe_wait_copy_tasks cost 1.801 ms
DEBUG 05-06 11:55:49.260202.260202 cuda_h.py:27] end moe_vllm_forward cost 7.869 ms
DEBUG 05-06 11:55:49.260406.260406 cuda_h.py:27] end moe_cpu_merge cost 0.066 ms
DEBUG 05-06 11:55:49.260829.260829 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:49.260236.260236 lmp.py:1950] [layer_moe_fused] vllm triton time: 8.648ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.260517.260517 cuda_h.py:27] end *layer_moe_fused cost 14.039 ms
DEBUG 05-06 11:55:49.266214.266214 cuda_h.py:27] end prefill_merge_scale cost 5.929 ms
DEBUG 05-06 11:55:49.266635.266635 cuda_h.py:27] end prefill_layer cost 25.946 ms
DEBUG 05-06 11:55:49.266402.266402 lmp.py:1391] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 11:55:49.267980.267980 lmp.py:1347] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 11:55:49.267398.267398 cuda_h.py:27] end prefill_ln cost 0.258 ms
DEBUG 05-06 11:55:49.269280.269280 cuda_h.py:27] end prefill_attn cost 2.160 ms
DEBUG 05-06 11:55:49.270385.270385 cuda_h.py:27] end prefill_ffn_prep cost 0.484 ms
DEBUG 05-06 11:55:49.271712.271712 cuda_h.py:27] end prefill_gate cost 0.512 ms
experts_cpu_alloc {'expert_ids': [43, 47, 91, 119, 11, 39, 55, 127, 63, 15, 27, 111, 64, 92, 104, 116, 124, 20, 96, 108, 76, 16, 52, 68, 49, 89, 21, 37, 61, 85, 117, 41, 53, 65, 45, 69, 93, 113, 57, 14, 62, 82, 22, 42, 98, 126, 66, 38, 122, 70, 46], 'token_total': 150, 'token_per_expert': {43: 1, 47: 1, 91: 1, 119: 1, 11: 2, 39: 2, 55: 2, 127: 2, 63: 3, 15: 4, 27: 5, 111: 6, 64: 1, 92: 1, 104: 1, 116: 1, 124: 1, 20: 2, 96: 2, 108: 2, 76: 3, 16: 4, 52: 5, 68: 5, 49: 1, 89: 1, 21: 2, 37: 2, 61: 2, 85: 2, 117: 2, 41: 3, 53: 3, 65: 3, 45: 4, 69: 4, 93: 4, 113: 4, 57: 5, 14: 1, 62: 1, 82: 1, 22: 2, 42: 2, 98: 3, 126: 3, 66: 4, 38: 7, 122: 7, 70: 8, 46: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 51, 71, 75, 87, 103, 123], 'expert_count': 9, 'ideal_gpu_count': 9, 'keep_on_gpu': 9, 'hit_count_on_device': 21, 'token_total': 1004, 'token_per_expert': {3: 453, 7: 448, 19: 12, 51: 28, 71: 11, 75: 15, 87: 14, 103: 16, 123: 7}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 28, 32, 36, 56, 80, 120], 'expert_count': 9, 'ideal_gpu_count': 9, 'keep_on_gpu': 9, 'hit_count_on_device': 21, 'token_total': 972, 'token_per_expert': {0: 448, 4: 450, 12: 13, 28: 14, 32: 9, 36: 6, 56: 11, 80: 7, 120: 14}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 73, 77, 81, 105, 121, 125], 'expert_count': 8, 'ideal_gpu_count': 8, 'keep_on_gpu': 8, 'hit_count_on_device': 23, 'token_total': 968, 'token_per_expert': {1: 448, 5: 450, 73: 20, 77: 5, 81: 10, 105: 13, 121: 9, 125: 13}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 50, 54, 58, 102, 110, 114], 'expert_count': 8, 'ideal_gpu_count': 8, 'keep_on_gpu': 8, 'hit_count_on_device': 20, 'token_total': 1002, 'token_per_expert': {2: 454, 6: 452, 50: 11, 54: 28, 58: 21, 102: 11, 110: 13, 114: 12}}
INFO 05-06 11:55:49.272399.272399 lmp.py:1836] [layer_moe_fused] layer=8 prefix: 0.466ms alloc: 0.336ms
INFO 05-06 11:55:49.272106.272106 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 4.363059997558594e-05 seconds
INFO 05-06 11:55:49.273038.273038 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007567405700683594s
INFO 05-06 11:55:49.274205.274205 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005805492401123047 seconds
DEBUG 05-06 11:55:49.274145.274145 cuda_h.py:27] end moe_cpu_prep_submit cost 0.726 ms
INFO 05-06 11:55:49.276447.276447 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016312599182128906s
DEBUG 05-06 11:55:49.276291.276291 cuda_h.py:27] end moe_wait_copy_tasks cost 1.739 ms
DEBUG 05-06 11:55:49.284958.284958 cuda_h.py:27] end moe_vllm_forward cost 7.632 ms
DEBUG 05-06 11:55:49.284261.284261 cuda_h.py:27] end moe_cpu_merge cost 0.070 ms
DEBUG 05-06 11:55:49.284015.284015 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:49.285367.285367 lmp.py:1950] [layer_moe_fused] vllm triton time: 8.535ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.285325.285325 cuda_h.py:27] end *layer_moe_fused cost 13.602 ms
DEBUG 05-06 11:55:49.289696.289696 cuda_h.py:27] end prefill_merge_scale cost 4.497 ms
DEBUG 05-06 11:55:49.290209.290209 cuda_h.py:27] end prefill_layer cost 23.060 ms
DEBUG 05-06 11:55:49.290917.290917 lmp.py:1391] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 11:55:49.290209.290209 lmp.py:1347] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 11:55:49.290011.290011 cuda_h.py:27] end prefill_ln cost 0.255 ms
DEBUG 05-06 11:55:49.292013.292013 cuda_h.py:27] end prefill_attn cost 2.143 ms
DEBUG 05-06 11:55:49.293972.293972 cuda_h.py:27] end prefill_ffn_prep cost 0.481 ms
DEBUG 05-06 11:55:49.294392.294392 cuda_h.py:27] end prefill_gate cost 0.507 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.295107.295107 lmp.py:1836] [layer_moe_fused] layer=9 prefix: 0.386ms alloc: 0.107ms
INFO 05-06 11:55:49.295648.295648 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 9.775161743164062e-06 seconds
INFO 05-06 11:55:49.296409.296409 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007283687591552734s
INFO 05-06 11:55:49.297879.297879 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00031566619873046875 seconds
DEBUG 05-06 11:55:49.297897.297897 cuda_h.py:27] end moe_cpu_prep_submit cost 0.463 ms
INFO 05-06 11:55:49.299159.299159 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017948150634765625s
DEBUG 05-06 11:55:49.299268.299268 cuda_h.py:27] end moe_wait_copy_tasks cost 1.883 ms
DEBUG 05-06 11:55:49.312344.312344 cuda_h.py:27] end moe_vllm_forward cost 13.010 ms
DEBUG 05-06 11:55:49.497804.497804 cuda_h.py:27] end moe_cpu_merge cost 185.054 ms
DEBUG 05-06 11:55:49.497318.497318 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 11:55:49.497262.497262 lmp.py:1950] [layer_moe_fused] vllm triton time: 198.754ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.498752.498752 cuda_h.py:27] end *layer_moe_fused cost 203.289 ms
DEBUG 05-06 11:55:49.499637.499637 cuda_h.py:27] end prefill_merge_scale cost 0.664 ms
DEBUG 05-06 11:55:49.499773.499773 cuda_h.py:27] end prefill_layer cost 208.875 ms
DEBUG 05-06 11:55:49.499734.499734 lmp.py:1391] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 11:55:49.499027.499027 lmp.py:1347] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 11:55:49.499822.499822 cuda_h.py:27] end prefill_ln cost 0.264 ms
DEBUG 05-06 11:55:49.502562.502562 cuda_h.py:27] end prefill_attn cost 2.265 ms
DEBUG 05-06 11:55:49.502244.502244 cuda_h.py:27] end prefill_ffn_prep cost 0.486 ms
DEBUG 05-06 11:55:49.504738.504738 cuda_h.py:27] end prefill_gate cost 0.555 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.504031.504031 lmp.py:1836] [layer_moe_fused] layer=10 prefix: 0.408ms alloc: 0.115ms
INFO 05-06 11:55:49.505028.505028 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 9.775161743164062e-06 seconds
INFO 05-06 11:55:49.506354.506354 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010612010955810547s
INFO 05-06 11:55:49.506115.506115 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00032019615173339844 seconds
DEBUG 05-06 11:55:49.507346.507346 cuda_h.py:27] end moe_cpu_prep_submit cost 0.821 ms
INFO 05-06 11:55:49.509665.509665 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001749277114868164s
DEBUG 05-06 11:55:49.509314.509314 cuda_h.py:27] end moe_wait_copy_tasks cost 1.953 ms
DEBUG 05-06 11:55:49.522192.522192 cuda_h.py:27] end moe_vllm_forward cost 12.128 ms
DEBUG 05-06 11:55:49.530252.530252 cuda_h.py:27] end moe_cpu_merge cost 8.386 ms
DEBUG 05-06 11:55:49.530202.530202 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:55:49.531709.531709 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.368ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.531933.531933 cuda_h.py:27] end *layer_moe_fused cost 27.070 ms
DEBUG 05-06 11:55:49.532506.532506 cuda_h.py:27] end prefill_merge_scale cost 0.645 ms
DEBUG 05-06 11:55:49.532112.532112 cuda_h.py:27] end prefill_layer cost 32.821 ms
DEBUG 05-06 11:55:49.532940.532940 lmp.py:1391] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 11:55:49.532279.532279 lmp.py:1347] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 11:55:49.532418.532418 cuda_h.py:27] end prefill_ln cost 0.252 ms
DEBUG 05-06 11:55:49.535401.535401 cuda_h.py:27] end prefill_attn cost 2.585 ms
DEBUG 05-06 11:55:49.536614.536614 cuda_h.py:27] end prefill_ffn_prep cost 0.558 ms
DEBUG 05-06 11:55:49.537631.537631 cuda_h.py:27] end prefill_gate cost 0.509 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.538572.538572 lmp.py:1836] [layer_moe_fused] layer=11 prefix: 0.405ms alloc: 0.112ms
INFO 05-06 11:55:49.538900.538900 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.3113021850585938e-05 seconds
INFO 05-06 11:55:49.539233.539233 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012083053588867188s
INFO 05-06 11:55:49.540180.540180 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003147125244140625 seconds
DEBUG 05-06 11:55:49.540379.540379 cuda_h.py:27] end moe_cpu_prep_submit cost 0.811 ms
INFO 05-06 11:55:49.542848.542848 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016508102416992188s
DEBUG 05-06 11:55:49.542855.542855 cuda_h.py:27] end moe_wait_copy_tasks cost 1.835 ms
DEBUG 05-06 11:55:49.556769.556769 cuda_h.py:27] end moe_vllm_forward cost 12.146 ms
DEBUG 05-06 11:55:49.564202.564202 cuda_h.py:27] end moe_cpu_merge cost 8.448 ms
DEBUG 05-06 11:55:49.564013.564013 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:55:49.564520.564520 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.475ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.565939.565939 cuda_h.py:27] end *layer_moe_fused cost 27.476 ms
DEBUG 05-06 11:55:49.565545.565545 cuda_h.py:27] end prefill_merge_scale cost 0.634 ms
DEBUG 05-06 11:55:49.566389.566389 cuda_h.py:27] end prefill_layer cost 33.541 ms
DEBUG 05-06 11:55:49.566880.566880 lmp.py:1391] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 11:55:49.566364.566364 lmp.py:1347] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 11:55:49.566431.566431 cuda_h.py:27] end prefill_ln cost 0.257 ms
DEBUG 05-06 11:55:49.569913.569913 cuda_h.py:27] end prefill_attn cost 2.252 ms
DEBUG 05-06 11:55:49.569409.569409 cuda_h.py:27] end prefill_ffn_prep cost 0.491 ms
DEBUG 05-06 11:55:49.570591.570591 cuda_h.py:27] end prefill_gate cost 0.528 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.571593.571593 lmp.py:1836] [layer_moe_fused] layer=12 prefix: 0.406ms alloc: 0.120ms
INFO 05-06 11:55:49.571312.571312 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.2159347534179688e-05 seconds
INFO 05-06 11:55:49.573431.573431 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009744167327880859s
INFO 05-06 11:55:49.573490.573490 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003292560577392578 seconds
DEBUG 05-06 11:55:49.573178.573178 cuda_h.py:27] end moe_cpu_prep_submit cost 0.824 ms
INFO 05-06 11:55:49.575370.575370 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001566171646118164s
DEBUG 05-06 11:55:49.575899.575899 cuda_h.py:27] end moe_wait_copy_tasks cost 1.749 ms
DEBUG 05-06 11:55:49.589298.589298 cuda_h.py:27] end moe_vllm_forward cost 12.197 ms
DEBUG 05-06 11:55:49.597829.597829 cuda_h.py:27] end moe_cpu_merge cost 8.623 ms
DEBUG 05-06 11:55:49.597402.597402 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:55:49.597240.597240 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.684ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.598478.598478 cuda_h.py:27] end *layer_moe_fused cost 27.166 ms
DEBUG 05-06 11:55:49.599561.599561 cuda_h.py:27] end prefill_merge_scale cost 0.634 ms
DEBUG 05-06 11:55:49.599359.599359 cuda_h.py:27] end prefill_layer cost 32.859 ms
DEBUG 05-06 11:55:49.599220.599220 lmp.py:1391] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 11:55:49.599036.599036 lmp.py:1347] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 11:55:49.599329.599329 cuda_h.py:27] end prefill_ln cost 0.286 ms
DEBUG 05-06 11:55:49.602346.602346 cuda_h.py:27] end prefill_attn cost 2.222 ms
DEBUG 05-06 11:55:49.602120.602120 cuda_h.py:27] end prefill_ffn_prep cost 0.485 ms
DEBUG 05-06 11:55:49.604986.604986 cuda_h.py:27] end prefill_gate cost 0.580 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.604789.604789 lmp.py:1836] [layer_moe_fused] layer=13 prefix: 0.409ms alloc: 0.107ms
INFO 05-06 11:55:49.604978.604978 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.33514404296875e-05 seconds
INFO 05-06 11:55:49.606490.606490 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009889602661132812s
INFO 05-06 11:55:49.606483.606483 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003151893615722656 seconds
DEBUG 05-06 11:55:49.607054.607054 cuda_h.py:27] end moe_cpu_prep_submit cost 0.900 ms
INFO 05-06 11:55:49.609105.609105 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001621246337890625s
DEBUG 05-06 11:55:49.609291.609291 cuda_h.py:27] end moe_wait_copy_tasks cost 1.834 ms
DEBUG 05-06 11:55:49.622912.622912 cuda_h.py:27] end moe_vllm_forward cost 12.120 ms
DEBUG 05-06 11:55:49.631147.631147 cuda_h.py:27] end moe_cpu_merge cost 8.897 ms
DEBUG 05-06 11:55:49.631681.631681 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:49.631492.631492 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.887ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.632396.632396 cuda_h.py:27] end *layer_moe_fused cost 27.759 ms
DEBUG 05-06 11:55:49.632055.632055 cuda_h.py:27] end prefill_merge_scale cost 0.639 ms
DEBUG 05-06 11:55:49.632569.632569 cuda_h.py:27] end prefill_layer cost 33.501 ms
DEBUG 05-06 11:55:49.633686.633686 lmp.py:1391] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 11:55:49.633264.633264 lmp.py:1347] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 11:55:49.633495.633495 cuda_h.py:27] end prefill_ln cost 0.254 ms
DEBUG 05-06 11:55:49.635733.635733 cuda_h.py:27] end prefill_attn cost 2.282 ms
DEBUG 05-06 11:55:49.636414.636414 cuda_h.py:27] end prefill_ffn_prep cost 0.486 ms
DEBUG 05-06 11:55:49.637967.637967 cuda_h.py:27] end prefill_gate cost 0.533 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.638088.638088 lmp.py:1836] [layer_moe_fused] layer=14 prefix: 0.396ms alloc: 0.110ms
INFO 05-06 11:55:49.638416.638416 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 9.5367431640625e-06 seconds
INFO 05-06 11:55:49.639638.639638 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010356903076171875s
INFO 05-06 11:55:49.640207.640207 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003170967102050781 seconds
DEBUG 05-06 11:55:49.640798.640798 cuda_h.py:27] end moe_cpu_prep_submit cost 0.873 ms
INFO 05-06 11:55:49.642768.642768 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015728473663330078s
DEBUG 05-06 11:55:49.642290.642290 cuda_h.py:27] end moe_wait_copy_tasks cost 1.753 ms
DEBUG 05-06 11:55:49.656729.656729 cuda_h.py:27] end moe_vllm_forward cost 12.124 ms
DEBUG 05-06 11:55:49.664459.664459 cuda_h.py:27] end moe_cpu_merge cost 8.419 ms
DEBUG 05-06 11:55:49.664449.664449 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:55:49.664002.664002 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.448ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.665523.665523 cuda_h.py:27] end *layer_moe_fused cost 27.340 ms
DEBUG 05-06 11:55:49.666175.666175 cuda_h.py:27] end prefill_merge_scale cost 0.635 ms
DEBUG 05-06 11:55:49.666926.666926 cuda_h.py:27] end prefill_layer cost 33.036 ms
DEBUG 05-06 11:55:49.666202.666202 lmp.py:1391] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 11:55:49.666494.666494 lmp.py:1347] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 11:55:49.666564.666564 cuda_h.py:27] end prefill_ln cost 0.259 ms
DEBUG 05-06 11:55:49.669940.669940 cuda_h.py:27] end prefill_attn cost 2.243 ms
DEBUG 05-06 11:55:49.669244.669244 cuda_h.py:27] end prefill_ffn_prep cost 0.489 ms
DEBUG 05-06 11:55:49.671831.671831 cuda_h.py:27] end prefill_gate cost 0.530 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.672827.672827 lmp.py:1836] [layer_moe_fused] layer=15 prefix: 0.431ms alloc: 0.115ms
INFO 05-06 11:55:49.672016.672016 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.4066696166992188e-05 seconds
INFO 05-06 11:55:49.673162.673162 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010101795196533203s
INFO 05-06 11:55:49.673970.673970 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003185272216796875 seconds
DEBUG 05-06 11:55:49.674804.674804 cuda_h.py:27] end moe_cpu_prep_submit cost 0.794 ms
INFO 05-06 11:55:49.676997.676997 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001795053482055664s
DEBUG 05-06 11:55:49.677877.677877 cuda_h.py:27] end moe_wait_copy_tasks cost 1.994 ms
DEBUG 05-06 11:55:49.690812.690812 cuda_h.py:27] end moe_vllm_forward cost 12.359 ms
DEBUG 05-06 11:55:49.698759.698759 cuda_h.py:27] end moe_cpu_merge cost 8.366 ms
DEBUG 05-06 11:55:49.771532.771532 cuda_h.py:27] end moe_shared_experts cost 0.013 ms
INFO 05-06 11:55:49.771917.771917 lmp.py:1950] [layer_moe_fused] vllm triton time: 94.165ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.772210.772210 cuda_h.py:27] end *layer_moe_fused cost 100.785 ms
DEBUG 05-06 11:55:49.773067.773067 cuda_h.py:27] end prefill_merge_scale cost 1.346 ms
DEBUG 05-06 11:55:49.773034.773034 cuda_h.py:27] end prefill_layer cost 107.477 ms
DEBUG 05-06 11:55:49.774457.774457 lmp.py:1391] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 11:55:49.774451.774451 lmp.py:1347] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 11:55:49.775253.775253 cuda_h.py:27] end prefill_ln cost 0.607 ms
DEBUG 05-06 11:55:49.780536.780536 cuda_h.py:27] end prefill_attn cost 4.558 ms
DEBUG 05-06 11:55:49.781605.781605 cuda_h.py:27] end prefill_ffn_prep cost 1.110 ms
DEBUG 05-06 11:55:49.784875.784875 cuda_h.py:27] end prefill_gate cost 1.041 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.784360.784360 lmp.py:1836] [layer_moe_fused] layer=16 prefix: 0.389ms alloc: 0.096ms
INFO 05-06 11:55:49.784926.784926 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.2636184692382812e-05 seconds
INFO 05-06 11:55:49.795384.795384 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.010469198226928711s
INFO 05-06 11:55:49.796258.796258 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006189346313476562 seconds
DEBUG 05-06 11:55:49.796694.796694 cuda_h.py:27] end moe_cpu_prep_submit cost 0.792 ms
INFO 05-06 11:55:49.799277.799277 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0022935867309570312s
DEBUG 05-06 11:55:49.800710.800710 cuda_h.py:27] end moe_wait_copy_tasks cost 2.587 ms
DEBUG 05-06 11:55:49.816546.816546 cuda_h.py:27] end moe_vllm_forward cost 14.849 ms
DEBUG 05-06 11:55:49.824132.824132 cuda_h.py:27] end moe_cpu_merge cost 7.683 ms
DEBUG 05-06 11:55:49.824261.824261 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:49.824052.824052 lmp.py:1950] [layer_moe_fused] vllm triton time: 23.994ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.824722.824722 cuda_h.py:27] end *layer_moe_fused cost 40.574 ms
DEBUG 05-06 11:55:49.825307.825307 cuda_h.py:27] end prefill_merge_scale cost 0.621 ms
DEBUG 05-06 11:55:49.825244.825244 cuda_h.py:27] end prefill_layer cost 51.188 ms
DEBUG 05-06 11:55:49.825998.825998 lmp.py:1391] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 11:55:49.825337.825337 lmp.py:1347] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 11:55:49.826119.826119 cuda_h.py:27] end prefill_ln cost 0.257 ms
DEBUG 05-06 11:55:49.828535.828535 cuda_h.py:27] end prefill_attn cost 2.273 ms
DEBUG 05-06 11:55:49.829687.829687 cuda_h.py:27] end prefill_ffn_prep cost 0.482 ms
DEBUG 05-06 11:55:49.830405.830405 cuda_h.py:27] end prefill_gate cost 0.512 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.831842.831842 lmp.py:1836] [layer_moe_fused] layer=17 prefix: 0.381ms alloc: 0.115ms
INFO 05-06 11:55:49.831978.831978 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.239776611328125e-05 seconds
INFO 05-06 11:55:49.832001.832001 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009081363677978516s
INFO 05-06 11:55:49.832603.832603 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003056526184082031 seconds
DEBUG 05-06 11:55:49.833729.833729 cuda_h.py:27] end moe_cpu_prep_submit cost 0.664 ms
INFO 05-06 11:55:49.835195.835195 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0013649463653564453s
DEBUG 05-06 11:55:49.835247.835247 cuda_h.py:27] end moe_wait_copy_tasks cost 1.550 ms
DEBUG 05-06 11:55:49.847954.847954 cuda_h.py:27] end moe_vllm_forward cost 11.755 ms
DEBUG 05-06 11:55:49.856791.856791 cuda_h.py:27] end moe_cpu_merge cost 8.465 ms
DEBUG 05-06 11:55:49.856927.856927 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:55:49.856718.856718 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.046ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.857747.857747 cuda_h.py:27] end *layer_moe_fused cost 26.342 ms
DEBUG 05-06 11:55:49.857823.857823 cuda_h.py:27] end prefill_merge_scale cost 0.631 ms
DEBUG 05-06 11:55:49.857906.857906 cuda_h.py:27] end prefill_layer cost 32.018 ms
DEBUG 05-06 11:55:49.858265.858265 lmp.py:1391] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 11:55:49.858525.858525 lmp.py:1347] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 11:55:49.858893.858893 cuda_h.py:27] end prefill_ln cost 0.260 ms
DEBUG 05-06 11:55:49.860699.860699 cuda_h.py:27] end prefill_attn cost 2.247 ms
DEBUG 05-06 11:55:49.861565.861565 cuda_h.py:27] end prefill_ffn_prep cost 0.482 ms
DEBUG 05-06 11:55:49.862065.862065 cuda_h.py:27] end prefill_gate cost 0.527 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.863549.863549 lmp.py:1836] [layer_moe_fused] layer=18 prefix: 0.382ms alloc: 0.111ms
INFO 05-06 11:55:49.863514.863514 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 8.344650268554688e-06 seconds
INFO 05-06 11:55:49.864414.864414 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009577274322509766s
INFO 05-06 11:55:49.865215.865215 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00031256675720214844 seconds
DEBUG 05-06 11:55:49.865258.865258 cuda_h.py:27] end moe_cpu_prep_submit cost 0.752 ms
INFO 05-06 11:55:49.867924.867924 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0019309520721435547s
DEBUG 05-06 11:55:49.868777.868777 cuda_h.py:27] end moe_wait_copy_tasks cost 2.110 ms
DEBUG 05-06 11:55:49.880293.880293 cuda_h.py:27] end moe_vllm_forward cost 11.766 ms
DEBUG 05-06 11:55:49.889305.889305 cuda_h.py:27] end moe_cpu_merge cost 8.343 ms
DEBUG 05-06 11:55:49.889540.889540 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:49.889285.889285 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.951ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.889061.889061 cuda_h.py:27] end *layer_moe_fused cost 26.858 ms
DEBUG 05-06 11:55:49.890436.890436 cuda_h.py:27] end prefill_merge_scale cost 0.639 ms
DEBUG 05-06 11:55:49.890664.890664 cuda_h.py:27] end prefill_layer cost 32.439 ms
DEBUG 05-06 11:55:49.890627.890627 lmp.py:1391] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 11:55:49.891159.891159 lmp.py:1347] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 11:55:49.891695.891695 cuda_h.py:27] end prefill_ln cost 0.251 ms
DEBUG 05-06 11:55:49.893705.893705 cuda_h.py:27] end prefill_attn cost 2.220 ms
DEBUG 05-06 11:55:49.894718.894718 cuda_h.py:27] end prefill_ffn_prep cost 0.484 ms
DEBUG 05-06 11:55:49.895238.895238 cuda_h.py:27] end prefill_gate cost 0.530 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.896493.896493 lmp.py:1836] [layer_moe_fused] layer=19 prefix: 0.393ms alloc: 0.111ms
INFO 05-06 11:55:49.896166.896166 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.1444091796875e-05 seconds
INFO 05-06 11:55:49.897753.897753 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010042190551757812s
INFO 05-06 11:55:49.898130.898130 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003161430358886719 seconds
DEBUG 05-06 11:55:49.898319.898319 cuda_h.py:27] end moe_cpu_prep_submit cost 0.760 ms
INFO 05-06 11:55:49.901801.901801 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014908313751220703s
DEBUG 05-06 11:55:49.901701.901701 cuda_h.py:27] end moe_wait_copy_tasks cost 1.684 ms
DEBUG 05-06 11:55:49.914579.914579 cuda_h.py:27] end moe_vllm_forward cost 11.626 ms
DEBUG 05-06 11:55:49.922715.922715 cuda_h.py:27] end moe_cpu_merge cost 8.299 ms
DEBUG 05-06 11:55:49.922705.922705 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:49.922542.922542 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.769ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.923890.923890 cuda_h.py:27] end *layer_moe_fused cost 27.206 ms
DEBUG 05-06 11:55:49.923649.923649 cuda_h.py:27] end prefill_merge_scale cost 0.642 ms
DEBUG 05-06 11:55:49.923162.923162 cuda_h.py:27] end prefill_layer cost 32.936 ms
DEBUG 05-06 11:55:49.924873.924873 lmp.py:1391] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 11:55:49.924405.924405 lmp.py:1347] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 11:55:49.924610.924610 cuda_h.py:27] end prefill_ln cost 0.254 ms
DEBUG 05-06 11:55:49.926229.926229 cuda_h.py:27] end prefill_attn cost 2.211 ms
DEBUG 05-06 11:55:49.927765.927765 cuda_h.py:27] end prefill_ffn_prep cost 0.486 ms
DEBUG 05-06 11:55:49.928393.928393 cuda_h.py:27] end prefill_gate cost 0.521 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.929870.929870 lmp.py:1836] [layer_moe_fused] layer=20 prefix: 0.381ms alloc: 0.109ms
INFO 05-06 11:55:49.929244.929244 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.2874603271484375e-05 seconds
INFO 05-06 11:55:49.931332.931332 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009937286376953125s
INFO 05-06 11:55:49.931033.931033 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00031065940856933594 seconds
DEBUG 05-06 11:55:49.931204.931204 cuda_h.py:27] end moe_cpu_prep_submit cost 0.601 ms
INFO 05-06 11:55:49.934354.934354 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0019240379333496094s
DEBUG 05-06 11:55:49.934843.934843 cuda_h.py:27] end moe_wait_copy_tasks cost 2.116 ms
DEBUG 05-06 11:55:49.946827.946827 cuda_h.py:27] end moe_vllm_forward cost 11.747 ms
DEBUG 05-06 11:55:49.955319.955319 cuda_h.py:27] end moe_cpu_merge cost 8.445 ms
DEBUG 05-06 11:55:49.955978.955978 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:55:49.955293.955293 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.047ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.956707.956707 cuda_h.py:27] end *layer_moe_fused cost 26.882 ms
DEBUG 05-06 11:55:49.956943.956943 cuda_h.py:27] end prefill_merge_scale cost 0.639 ms
DEBUG 05-06 11:55:49.956595.956595 cuda_h.py:27] end prefill_layer cost 32.582 ms
DEBUG 05-06 11:55:49.957485.957485 lmp.py:1391] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 11:55:49.957254.957254 lmp.py:1347] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 11:55:49.957611.957611 cuda_h.py:27] end prefill_ln cost 0.256 ms
DEBUG 05-06 11:55:49.959762.959762 cuda_h.py:27] end prefill_attn cost 2.251 ms
DEBUG 05-06 11:55:49.960297.960297 cuda_h.py:27] end prefill_ffn_prep cost 0.485 ms
DEBUG 05-06 11:55:49.961135.961135 cuda_h.py:27] end prefill_gate cost 0.522 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.962666.962666 lmp.py:1836] [layer_moe_fused] layer=21 prefix: 0.383ms alloc: 0.109ms
INFO 05-06 11:55:49.962755.962755 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.2159347534179688e-05 seconds
INFO 05-06 11:55:49.963937.963937 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.00106048583984375s
INFO 05-06 11:55:49.964639.964639 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00031065940856933594 seconds
DEBUG 05-06 11:55:49.964632.964632 cuda_h.py:27] end moe_cpu_prep_submit cost 0.659 ms
INFO 05-06 11:55:49.966351.966351 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017158985137939453s
DEBUG 05-06 11:55:49.966073.966073 cuda_h.py:27] end moe_wait_copy_tasks cost 1.902 ms
DEBUG 05-06 11:55:49.979390.979390 cuda_h.py:27] end moe_vllm_forward cost 11.582 ms
DEBUG 05-06 11:55:49.987815.987815 cuda_h.py:27] end moe_cpu_merge cost 8.391 ms
DEBUG 05-06 11:55:49.988673.988673 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:49.988510.988510 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.847ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:49.988110.988110 cuda_h.py:27] end *layer_moe_fused cost 26.632 ms
DEBUG 05-06 11:55:49.989193.989193 cuda_h.py:27] end prefill_merge_scale cost 0.636 ms
DEBUG 05-06 11:55:49.989514.989514 cuda_h.py:27] end prefill_layer cost 32.282 ms
DEBUG 05-06 11:55:49.989904.989904 lmp.py:1391] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 11:55:49.989197.989197 lmp.py:1347] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 11:55:49.990210.990210 cuda_h.py:27] end prefill_ln cost 0.251 ms
DEBUG 05-06 11:55:49.992189.992189 cuda_h.py:27] end prefill_attn cost 2.262 ms
DEBUG 05-06 11:55:49.992486.992486 cuda_h.py:27] end prefill_ffn_prep cost 0.484 ms
DEBUG 05-06 11:55:49.994278.994278 cuda_h.py:27] end prefill_gate cost 0.525 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:49.995755.995755 lmp.py:1836] [layer_moe_fused] layer=22 prefix: 0.385ms alloc: 0.107ms
INFO 05-06 11:55:49.995898.995898 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.0013580322265625e-05 seconds
INFO 05-06 11:55:49.996309.996309 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001024484634399414s
INFO 05-06 11:55:49.996964.996964 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003108978271484375 seconds
DEBUG 05-06 11:55:49.997162.997162 cuda_h.py:27] end moe_cpu_prep_submit cost 0.711 ms
INFO 05-06 11:55:49.999624.999624 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001435995101928711s
DEBUG 05-06 11:55:49.999776.999776 cuda_h.py:27] end moe_wait_copy_tasks cost 1.625 ms
DEBUG 05-06 11:55:50.012965.012965 cuda_h.py:27] end moe_vllm_forward cost 11.712 ms
DEBUG 05-06 11:55:50.020709.020709 cuda_h.py:27] end moe_cpu_merge cost 8.453 ms
DEBUG 05-06 11:55:50.020044.020044 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:50.020981.020981 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.040ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:50.021895.021895 cuda_h.py:27] end *layer_moe_fused cost 26.829 ms
DEBUG 05-06 11:55:50.022594.022594 cuda_h.py:27] end prefill_merge_scale cost 0.633 ms
DEBUG 05-06 11:55:50.022584.022584 cuda_h.py:27] end prefill_layer cost 32.516 ms
DEBUG 05-06 11:55:50.022067.022067 lmp.py:1391] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 11:55:50.022598.022598 lmp.py:1347] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 11:55:50.022273.022273 cuda_h.py:27] end prefill_ln cost 0.250 ms
DEBUG 05-06 11:55:50.025254.025254 cuda_h.py:27] end prefill_attn cost 2.722 ms
DEBUG 05-06 11:55:50.026633.026633 cuda_h.py:27] end prefill_ffn_prep cost 0.536 ms
DEBUG 05-06 11:55:50.027689.027689 cuda_h.py:27] end prefill_gate cost 0.511 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:50.028385.028385 lmp.py:1836] [layer_moe_fused] layer=23 prefix: 0.389ms alloc: 0.108ms
INFO 05-06 11:55:50.028621.028621 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.2636184692382812e-05 seconds
INFO 05-06 11:55:50.029231.029231 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009980201721191406s
INFO 05-06 11:55:50.030747.030747 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003139972686767578 seconds
DEBUG 05-06 11:55:50.030977.030977 cuda_h.py:27] end moe_cpu_prep_submit cost 0.618 ms
INFO 05-06 11:55:50.032618.032618 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015063285827636719s
DEBUG 05-06 11:55:50.032756.032756 cuda_h.py:27] end moe_wait_copy_tasks cost 1.683 ms
DEBUG 05-06 11:55:50.045905.045905 cuda_h.py:27] end moe_vllm_forward cost 11.636 ms
DEBUG 05-06 11:55:50.053126.053126 cuda_h.py:27] end moe_cpu_merge cost 8.459 ms
DEBUG 05-06 11:55:50.053500.053500 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:55:50.054722.054722 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.956ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:50.054555.054555 cuda_h.py:27] end *layer_moe_fused cost 26.715 ms
DEBUG 05-06 11:55:50.055306.055306 cuda_h.py:27] end prefill_merge_scale cost 0.637 ms
DEBUG 05-06 11:55:50.055720.055720 cuda_h.py:27] end prefill_layer cost 32.918 ms
DEBUG 05-06 11:55:50.055346.055346 lmp.py:1391] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 11:55:50.055162.055162 lmp.py:1347] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 11:55:50.055507.055507 cuda_h.py:27] end prefill_ln cost 0.260 ms
DEBUG 05-06 11:55:50.058709.058709 cuda_h.py:27] end prefill_attn cost 2.219 ms
DEBUG 05-06 11:55:50.058576.058576 cuda_h.py:27] end prefill_ffn_prep cost 0.484 ms
DEBUG 05-06 11:55:50.060388.060388 cuda_h.py:27] end prefill_gate cost 0.544 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:50.060587.060587 lmp.py:1836] [layer_moe_fused] layer=24 prefix: 0.383ms alloc: 0.113ms
INFO 05-06 11:55:50.060869.060869 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.239776611328125e-05 seconds
INFO 05-06 11:55:50.062535.062535 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010161399841308594s
INFO 05-06 11:55:50.062951.062951 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003113746643066406 seconds
DEBUG 05-06 11:55:50.063177.063177 cuda_h.py:27] end moe_cpu_prep_submit cost 0.645 ms
INFO 05-06 11:55:50.065147.065147 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014629364013671875s
DEBUG 05-06 11:55:50.065140.065140 cuda_h.py:27] end moe_wait_copy_tasks cost 1.637 ms
DEBUG 05-06 11:55:50.077140.077140 cuda_h.py:27] end moe_vllm_forward cost 11.664 ms
DEBUG 05-06 11:55:50.086359.086359 cuda_h.py:27] end moe_cpu_merge cost 8.596 ms
DEBUG 05-06 11:55:50.086363.086363 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 11:55:50.086154.086154 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.116ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:50.087172.087172 cuda_h.py:27] end *layer_moe_fused cost 26.880 ms
DEBUG 05-06 11:55:50.087561.087561 cuda_h.py:27] end prefill_merge_scale cost 0.685 ms
DEBUG 05-06 11:55:50.088313.088313 cuda_h.py:27] end prefill_layer cost 32.588 ms
DEBUG 05-06 11:55:50.088007.088007 lmp.py:1391] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 11:55:50.088538.088538 lmp.py:1347] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 11:55:50.088889.088889 cuda_h.py:27] end prefill_ln cost 0.255 ms
DEBUG 05-06 11:55:50.090787.090787 cuda_h.py:27] end prefill_attn cost 2.204 ms
DEBUG 05-06 11:55:50.091707.091707 cuda_h.py:27] end prefill_ffn_prep cost 0.487 ms
DEBUG 05-06 11:55:50.092181.092181 cuda_h.py:27] end prefill_gate cost 0.526 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:50.093685.093685 lmp.py:1836] [layer_moe_fused] layer=25 prefix: 0.385ms alloc: 0.116ms
INFO 05-06 11:55:50.093775.093775 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.2636184692382812e-05 seconds
INFO 05-06 11:55:50.094834.094834 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009918212890625s
INFO 05-06 11:55:50.095151.095151 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00030875205993652344 seconds
DEBUG 05-06 11:55:50.095670.095670 cuda_h.py:27] end moe_cpu_prep_submit cost 0.731 ms
INFO 05-06 11:55:50.098394.098394 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016543865203857422s
DEBUG 05-06 11:55:50.098506.098506 cuda_h.py:27] end moe_wait_copy_tasks cost 1.851 ms
DEBUG 05-06 11:55:50.110996.110996 cuda_h.py:27] end moe_vllm_forward cost 11.400 ms
DEBUG 05-06 11:55:50.119064.119064 cuda_h.py:27] end moe_cpu_merge cost 8.414 ms
DEBUG 05-06 11:55:50.119107.119107 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:50.119567.119567 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.657ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:50.119389.119389 cuda_h.py:27] end *layer_moe_fused cost 26.655 ms
DEBUG 05-06 11:55:50.120571.120571 cuda_h.py:27] end prefill_merge_scale cost 0.636 ms
DEBUG 05-06 11:55:50.120746.120746 cuda_h.py:27] end prefill_layer cost 32.314 ms
DEBUG 05-06 11:55:50.120402.120402 lmp.py:1391] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 11:55:50.120503.120503 lmp.py:1347] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 11:55:50.121238.121238 cuda_h.py:27] end prefill_ln cost 0.258 ms
DEBUG 05-06 11:55:50.123341.123341 cuda_h.py:27] end prefill_attn cost 2.217 ms
DEBUG 05-06 11:55:50.124036.124036 cuda_h.py:27] end prefill_ffn_prep cost 0.488 ms
DEBUG 05-06 11:55:50.125536.125536 cuda_h.py:27] end prefill_gate cost 0.525 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:50.126205.126205 lmp.py:1836] [layer_moe_fused] layer=26 prefix: 0.384ms alloc: 0.110ms
INFO 05-06 11:55:50.126864.126864 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.0251998901367188e-05 seconds
INFO 05-06 11:55:50.127752.127752 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009813308715820312s
INFO 05-06 11:55:50.127719.127719 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003228187561035156 seconds
DEBUG 05-06 11:55:50.128166.128166 cuda_h.py:27] end moe_cpu_prep_submit cost 0.795 ms
INFO 05-06 11:55:50.130704.130704 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001871347427368164s
DEBUG 05-06 11:55:50.130127.130127 cuda_h.py:27] end moe_wait_copy_tasks cost 2.047 ms
DEBUG 05-06 11:55:50.143726.143726 cuda_h.py:27] end moe_vllm_forward cost 11.621 ms
DEBUG 05-06 11:55:50.152215.152215 cuda_h.py:27] end moe_cpu_merge cost 8.720 ms
DEBUG 05-06 11:55:50.152549.152549 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:50.152248.152248 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.220ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:50.152168.152168 cuda_h.py:27] end *layer_moe_fused cost 27.260 ms
DEBUG 05-06 11:55:50.153297.153297 cuda_h.py:27] end prefill_merge_scale cost 0.633 ms
DEBUG 05-06 11:55:50.153426.153426 cuda_h.py:27] end prefill_layer cost 32.911 ms
DEBUG 05-06 11:55:50.153090.153090 lmp.py:1391] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 11:55:50.154859.154859 lmp.py:1347] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 11:55:50.154905.154905 cuda_h.py:27] end prefill_ln cost 0.253 ms
DEBUG 05-06 11:55:50.156532.156532 cuda_h.py:27] end prefill_attn cost 2.219 ms
DEBUG 05-06 11:55:50.157491.157491 cuda_h.py:27] end prefill_ffn_prep cost 0.481 ms
DEBUG 05-06 11:55:50.158315.158315 cuda_h.py:27] end prefill_gate cost 0.520 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:50.159852.159852 lmp.py:1836] [layer_moe_fused] layer=27 prefix: 0.385ms alloc: 0.115ms
INFO 05-06 11:55:50.159657.159657 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.33514404296875e-05 seconds
INFO 05-06 11:55:50.160678.160678 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000997781753540039s
INFO 05-06 11:55:50.161002.161002 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00031185150146484375 seconds
DEBUG 05-06 11:55:50.161318.161318 cuda_h.py:27] end moe_cpu_prep_submit cost 0.630 ms
INFO 05-06 11:55:50.163205.163205 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0019440650939941406s
DEBUG 05-06 11:55:50.163456.163456 cuda_h.py:27] end moe_wait_copy_tasks cost 2.137 ms
DEBUG 05-06 11:55:50.176930.176930 cuda_h.py:27] end moe_vllm_forward cost 11.602 ms
DEBUG 05-06 11:55:50.184035.184035 cuda_h.py:27] end moe_cpu_merge cost 8.332 ms
DEBUG 05-06 11:55:50.185840.185840 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:50.185392.185392 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.804ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:50.185274.185274 cuda_h.py:27] end *layer_moe_fused cost 26.939 ms
DEBUG 05-06 11:55:50.186496.186496 cuda_h.py:27] end prefill_merge_scale cost 0.632 ms
DEBUG 05-06 11:55:50.186194.186194 cuda_h.py:27] end prefill_layer cost 32.548 ms
DEBUG 05-06 11:55:50.186578.186578 lmp.py:1391] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 11:55:50.186109.186109 lmp.py:1347] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 11:55:50.187712.187712 cuda_h.py:27] end prefill_ln cost 0.258 ms
DEBUG 05-06 11:55:50.189121.189121 cuda_h.py:27] end prefill_attn cost 2.266 ms
DEBUG 05-06 11:55:50.190472.190472 cuda_h.py:27] end prefill_ffn_prep cost 0.488 ms
DEBUG 05-06 11:55:50.191031.191031 cuda_h.py:27] end prefill_gate cost 0.532 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:50.192252.192252 lmp.py:1836] [layer_moe_fused] layer=28 prefix: 0.388ms alloc: 0.109ms
INFO 05-06 11:55:50.192011.192011 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.3828277587890625e-05 seconds
INFO 05-06 11:55:50.193520.193520 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009675025939941406s
INFO 05-06 11:55:50.193082.193082 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003123283386230469 seconds
DEBUG 05-06 11:55:50.194338.194338 cuda_h.py:27] end moe_cpu_prep_submit cost 0.589 ms
INFO 05-06 11:55:50.196112.196112 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.002069711685180664s
DEBUG 05-06 11:55:50.196727.196727 cuda_h.py:27] end moe_wait_copy_tasks cost 2.245 ms
DEBUG 05-06 11:55:50.209432.209432 cuda_h.py:27] end moe_vllm_forward cost 11.813 ms
DEBUG 05-06 11:55:50.218951.218951 cuda_h.py:27] end moe_cpu_merge cost 8.634 ms
DEBUG 05-06 11:55:50.218399.218399 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 11:55:50.218574.218574 lmp.py:1950] [layer_moe_fused] vllm triton time: 21.340ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:50.219311.219311 cuda_h.py:27] end *layer_moe_fused cost 27.368 ms
DEBUG 05-06 11:55:50.219654.219654 cuda_h.py:27] end prefill_merge_scale cost 0.674 ms
DEBUG 05-06 11:55:50.219313.219313 cuda_h.py:27] end prefill_layer cost 33.188 ms
DEBUG 05-06 11:55:50.220551.220551 lmp.py:1391] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 11:55:50.220797.220797 lmp.py:1347] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 11:55:50.220764.220764 cuda_h.py:27] end prefill_ln cost 0.254 ms
DEBUG 05-06 11:55:50.223009.223009 cuda_h.py:27] end prefill_attn cost 2.706 ms
DEBUG 05-06 11:55:50.224163.224163 cuda_h.py:27] end prefill_ffn_prep cost 0.551 ms
DEBUG 05-06 11:55:50.225987.225987 cuda_h.py:27] end prefill_gate cost 0.516 ms
experts_cpu_alloc {'expert_ids': [3, 0, 1, 2], 'token_total': 2048, 'token_per_expert': {3: 512, 0: 512, 1: 512, 2: 512}}
experts_gpu_alloc_device_0 {'expert_ids': [7], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {7: 512}}
experts_gpu_alloc_device_1 {'expert_ids': [4], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {4: 512}}
experts_gpu_alloc_device_2 {'expert_ids': [5], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {5: 512}}
experts_gpu_alloc_device_3 {'expert_ids': [6], 'expert_count': 1, 'ideal_gpu_count': 1, 'keep_on_gpu': 1, 'hit_count_on_device': 2, 'token_total': 512, 'token_per_expert': {6: 512}}
INFO 05-06 11:55:50.226247.226247 lmp.py:1836] [layer_moe_fused] layer=29 prefix: 0.389ms alloc: 0.111ms
INFO 05-06 11:55:50.226065.226065 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 1.430511474609375e-05 seconds
INFO 05-06 11:55:50.227538.227538 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001070261001586914s
INFO 05-06 11:55:50.227650.227650 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.00033211708068847656 seconds
DEBUG 05-06 11:55:50.228438.228438 cuda_h.py:27] end moe_cpu_prep_submit cost 0.663 ms
INFO 05-06 11:55:50.230508.230508 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.002208709716796875s
DEBUG 05-06 11:55:50.230600.230600 cuda_h.py:27] end moe_wait_copy_tasks cost 2.385 ms
DEBUG 05-06 11:55:50.243005.243005 cuda_h.py:27] end moe_vllm_forward cost 11.513 ms
DEBUG 05-06 11:55:50.252534.252534 cuda_h.py:27] end moe_cpu_merge cost 8.550 ms
DEBUG 05-06 11:55:50.252040.252040 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:50.252070.252070 lmp.py:1950] [layer_moe_fused] vllm triton time: 20.899ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:50.252751.252751 cuda_h.py:27] end *layer_moe_fused cost 27.246 ms
DEBUG 05-06 11:55:50.253728.253728 cuda_h.py:27] end prefill_merge_scale cost 0.628 ms
DEBUG 05-06 11:55:50.253996.253996 cuda_h.py:27] end prefill_layer cost 33.431 ms
DEBUG 05-06 11:55:50.253920.253920 lmp.py:1391] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 11:55:50.253212.253212 cuda_h.py:27] end prefill_step cost 1517.665 ms
INFO 05-06 11:55:50.254595.254595 lmp.py:1394] prefill time: 1.6224017143249512 seconds
INFO 05-06 11:55:50.259656.259656 lmp.py:1406] Static-KV prefill complete; seqlens set to 128.
WARNING 05-06 11:55:50.290102.290102 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:55:50.290713.290713 helper.py:35]   NaN count (hidden): 1441792
WARNING 05-06 11:55:50.291563.291563 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:55:50.291979.291979 helper.py:39]   NaN count (normed): 1441792
WARNING 05-06 11:55:50.297359.297359 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:55:50.297319.297319 helper.py:50]   NaN count: 1048576
WARNING 05-06 11:55:50.297691.297691 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:55:50.415959.415959 cuda_h.py:27] end init_inputs_tokens cost 155.062 ms
DEBUG 05-06 11:55:50.415812.415812 lmp.py:1507] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:55:50.415363.415363 lmp.py:1513] ---- decode step 0 layer 0 ----
DEBUG 05-06 11:55:50.445976.445976 cuda_h.py:27] end decode_layer cost 30.351 ms
DEBUG 05-06 11:55:50.445443.445443 lmp.py:1513] ---- decode step 0 layer 1 ----
DEBUG 05-06 11:55:50.452221.452221 cuda_h.py:27] end decode_layer cost 6.604 ms
DEBUG 05-06 11:55:50.452230.452230 lmp.py:1513] ---- decode step 0 layer 2 ----
DEBUG 05-06 11:55:50.458818.458818 cuda_h.py:27] end decode_layer cost 5.692 ms
DEBUG 05-06 11:55:50.458774.458774 lmp.py:1513] ---- decode step 0 layer 3 ----
DEBUG 05-06 11:55:50.464011.464011 cuda_h.py:27] end decode_layer cost 6.277 ms
DEBUG 05-06 11:55:50.464245.464245 lmp.py:1513] ---- decode step 0 layer 4 ----
DEBUG 05-06 11:55:50.470315.470315 cuda_h.py:27] end decode_layer cost 5.838 ms
DEBUG 05-06 11:55:50.470357.470357 lmp.py:1513] ---- decode step 0 layer 5 ----
DEBUG 05-06 11:55:50.493651.493651 cuda_h.py:27] end decode_layer cost 22.822 ms
DEBUG 05-06 11:55:50.493885.493885 lmp.py:1513] ---- decode step 0 layer 6 ----
DEBUG 05-06 11:55:50.499745.499745 cuda_h.py:27] end decode_layer cost 5.683 ms
DEBUG 05-06 11:55:50.499191.499191 lmp.py:1513] ---- decode step 0 layer 7 ----
DEBUG 05-06 11:55:50.505696.505696 cuda_h.py:27] end decode_layer cost 6.370 ms
DEBUG 05-06 11:55:50.505069.505069 lmp.py:1513] ---- decode step 0 layer 8 ----
DEBUG 05-06 11:55:50.511931.511931 cuda_h.py:27] end decode_layer cost 5.754 ms
DEBUG 05-06 11:55:50.511702.511702 lmp.py:1513] ---- decode step 0 layer 9 ----
DEBUG 05-06 11:55:50.518375.518375 cuda_h.py:27] end decode_layer cost 6.847 ms
DEBUG 05-06 11:55:50.518543.518543 lmp.py:1513] ---- decode step 0 layer 10 ----
DEBUG 05-06 11:55:50.524756.524756 cuda_h.py:27] end decode_layer cost 5.944 ms
DEBUG 05-06 11:55:50.524698.524698 lmp.py:1513] ---- decode step 0 layer 11 ----
DEBUG 05-06 11:55:50.530829.530829 cuda_h.py:27] end decode_layer cost 6.268 ms
DEBUG 05-06 11:55:50.530871.530871 lmp.py:1513] ---- decode step 0 layer 12 ----
DEBUG 05-06 11:55:50.536891.536891 cuda_h.py:27] end decode_layer cost 5.731 ms
DEBUG 05-06 11:55:50.536694.536694 lmp.py:1513] ---- decode step 0 layer 13 ----
DEBUG 05-06 11:55:50.542439.542439 cuda_h.py:27] end decode_layer cost 5.809 ms
DEBUG 05-06 11:55:50.542103.542103 lmp.py:1513] ---- decode step 0 layer 14 ----
DEBUG 05-06 11:55:50.548386.548386 cuda_h.py:27] end decode_layer cost 5.679 ms
DEBUG 05-06 11:55:50.548951.548951 lmp.py:1513] ---- decode step 0 layer 15 ----
DEBUG 05-06 11:55:50.554614.554614 cuda_h.py:27] end decode_layer cost 5.923 ms
DEBUG 05-06 11:55:50.554656.554656 lmp.py:1513] ---- decode step 0 layer 16 ----
DEBUG 05-06 11:55:50.559140.559140 cuda_h.py:27] end decode_layer cost 5.757 ms
DEBUG 05-06 11:55:50.560612.560612 lmp.py:1513] ---- decode step 0 layer 17 ----
DEBUG 05-06 11:55:50.566854.566854 cuda_h.py:27] end decode_layer cost 6.036 ms
DEBUG 05-06 11:55:50.566035.566035 lmp.py:1513] ---- decode step 0 layer 18 ----
DEBUG 05-06 11:55:50.571791.571791 cuda_h.py:27] end decode_layer cost 5.747 ms
DEBUG 05-06 11:55:50.571402.571402 lmp.py:1513] ---- decode step 0 layer 19 ----
DEBUG 05-06 11:55:50.577211.577211 cuda_h.py:27] end decode_layer cost 5.751 ms
DEBUG 05-06 11:55:50.577273.577273 lmp.py:1513] ---- decode step 0 layer 20 ----
DEBUG 05-06 11:55:50.583483.583483 cuda_h.py:27] end decode_layer cost 5.871 ms
DEBUG 05-06 11:55:50.583049.583049 lmp.py:1513] ---- decode step 0 layer 21 ----
DEBUG 05-06 11:55:50.589080.589080 cuda_h.py:27] end decode_layer cost 5.880 ms
DEBUG 05-06 11:55:50.589314.589314 lmp.py:1513] ---- decode step 0 layer 22 ----
DEBUG 05-06 11:55:50.595088.595088 cuda_h.py:27] end decode_layer cost 5.893 ms
DEBUG 05-06 11:55:50.595269.595269 lmp.py:1513] ---- decode step 0 layer 23 ----
DEBUG 05-06 11:55:50.601597.601597 cuda_h.py:27] end decode_layer cost 6.240 ms
DEBUG 05-06 11:55:50.601209.601209 lmp.py:1513] ---- decode step 0 layer 24 ----
DEBUG 05-06 11:55:50.607990.607990 cuda_h.py:27] end decode_layer cost 5.730 ms
DEBUG 05-06 11:55:50.607986.607986 lmp.py:1513] ---- decode step 0 layer 25 ----
DEBUG 05-06 11:55:50.613105.613105 cuda_h.py:27] end decode_layer cost 5.910 ms
DEBUG 05-06 11:55:50.613484.613484 lmp.py:1513] ---- decode step 0 layer 26 ----
DEBUG 05-06 11:55:50.619330.619330 cuda_h.py:27] end decode_layer cost 5.673 ms
DEBUG 05-06 11:55:50.619942.619942 lmp.py:1513] ---- decode step 0 layer 27 ----
DEBUG 05-06 11:55:50.625641.625641 cuda_h.py:27] end decode_layer cost 5.845 ms
DEBUG 05-06 11:55:50.625822.625822 lmp.py:1513] ---- decode step 0 layer 28 ----
DEBUG 05-06 11:55:50.631613.631613 cuda_h.py:27] end decode_layer cost 5.808 ms
DEBUG 05-06 11:55:50.631655.631655 lmp.py:1513] ---- decode step 0 layer 29 ----
DEBUG 05-06 11:55:50.637488.637488 cuda_h.py:27] end decode_layer cost 6.085 ms
DEBUG 05-06 11:55:50.637890.637890 cuda_h.py:27] end decode_step cost 377.514 ms
INFO 05-06 11:55:50.637090.637090 lmp.py:1561] decode step 0 time: 0.3775608539581299 seconds
WARNING 05-06 11:55:50.638395.638395 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:55:50.638991.638991 helper.py:35]   NaN count (hidden): 11264
WARNING 05-06 11:55:50.639858.639858 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:55:50.639597.639597 helper.py:39]   NaN count (normed): 11264
WARNING 05-06 11:55:50.644226.644226 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:55:50.644502.644502 helper.py:50]   NaN count: 1048576
WARNING 05-06 11:55:50.645040.645040 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:55:50.646190.646190 cuda_h.py:27] end init_inputs_tokens cost 8.438 ms
DEBUG 05-06 11:55:50.646563.646563 lmp.py:1507] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:55:50.646240.646240 lmp.py:1513] ---- decode step 1 layer 0 ----
DEBUG 05-06 11:55:50.652915.652915 cuda_h.py:27] end decode_layer cost 5.932 ms
DEBUG 05-06 11:55:50.652527.652527 lmp.py:1513] ---- decode step 1 layer 1 ----
DEBUG 05-06 11:55:50.658637.658637 cuda_h.py:27] end decode_layer cost 5.657 ms
DEBUG 05-06 11:55:50.658778.658778 lmp.py:1513] ---- decode step 1 layer 2 ----
DEBUG 05-06 11:55:50.664107.664107 cuda_h.py:27] end decode_layer cost 5.853 ms
DEBUG 05-06 11:55:50.664957.664957 lmp.py:1513] ---- decode step 1 layer 3 ----
DEBUG 05-06 11:55:50.670482.670482 cuda_h.py:27] end decode_layer cost 5.998 ms
DEBUG 05-06 11:55:50.670524.670524 lmp.py:1513] ---- decode step 1 layer 4 ----
DEBUG 05-06 11:55:50.676815.676815 cuda_h.py:27] end decode_layer cost 5.720 ms
DEBUG 05-06 11:55:50.676355.676355 lmp.py:1513] ---- decode step 1 layer 5 ----
DEBUG 05-06 11:55:50.682985.682985 cuda_h.py:27] end decode_layer cost 6.146 ms
DEBUG 05-06 11:55:50.682881.682881 lmp.py:1513] ---- decode step 1 layer 6 ----
DEBUG 05-06 11:55:50.689263.689263 cuda_h.py:27] end decode_layer cost 6.454 ms
DEBUG 05-06 11:55:50.689828.689828 lmp.py:1513] ---- decode step 1 layer 7 ----
DEBUG 05-06 11:55:50.695474.695474 cuda_h.py:27] end decode_layer cost 6.053 ms
DEBUG 05-06 11:55:50.695132.695132 lmp.py:1513] ---- decode step 1 layer 8 ----
DEBUG 05-06 11:55:50.701108.701108 cuda_h.py:27] end decode_layer cost 5.794 ms
DEBUG 05-06 11:55:50.701587.701587 lmp.py:1513] ---- decode step 1 layer 9 ----
DEBUG 05-06 11:55:50.706573.706573 cuda_h.py:27] end decode_layer cost 5.881 ms
DEBUG 05-06 11:55:50.707350.707350 lmp.py:1513] ---- decode step 1 layer 10 ----
DEBUG 05-06 11:55:50.713106.713106 cuda_h.py:27] end decode_layer cost 5.958 ms
DEBUG 05-06 11:55:50.713241.713241 lmp.py:1513] ---- decode step 1 layer 11 ----
DEBUG 05-06 11:55:50.719388.719388 cuda_h.py:27] end decode_layer cost 5.965 ms
DEBUG 05-06 11:55:50.719046.719046 lmp.py:1513] ---- decode step 1 layer 12 ----
DEBUG 05-06 11:55:50.724696.724696 cuda_h.py:27] end decode_layer cost 5.774 ms
DEBUG 05-06 11:55:50.724307.724307 lmp.py:1513] ---- decode step 1 layer 13 ----
DEBUG 05-06 11:55:50.730026.730026 cuda_h.py:27] end decode_layer cost 5.824 ms
DEBUG 05-06 11:55:50.730876.730876 lmp.py:1513] ---- decode step 1 layer 14 ----
DEBUG 05-06 11:55:50.736876.736876 cuda_h.py:27] end decode_layer cost 5.927 ms
DEBUG 05-06 11:55:50.736964.736964 lmp.py:1513] ---- decode step 1 layer 15 ----
DEBUG 05-06 11:55:50.742964.742964 cuda_h.py:27] end decode_layer cost 5.716 ms
DEBUG 05-06 11:55:50.742145.742145 lmp.py:1513] ---- decode step 1 layer 16 ----
DEBUG 05-06 11:55:50.748997.748997 cuda_h.py:27] end decode_layer cost 5.677 ms
DEBUG 05-06 11:55:50.748847.748847 lmp.py:1513] ---- decode step 1 layer 17 ----
DEBUG 05-06 11:55:50.754108.754108 cuda_h.py:27] end decode_layer cost 6.014 ms
DEBUG 05-06 11:55:50.754528.754528 lmp.py:1513] ---- decode step 1 layer 18 ----
DEBUG 05-06 11:55:50.760804.760804 cuda_h.py:27] end decode_layer cost 5.674 ms
DEBUG 05-06 11:55:50.760568.760568 lmp.py:1513] ---- decode step 1 layer 19 ----
DEBUG 05-06 11:55:50.766822.766822 cuda_h.py:27] end decode_layer cost 5.798 ms
DEBUG 05-06 11:55:50.766288.766288 lmp.py:1513] ---- decode step 1 layer 20 ----
DEBUG 05-06 11:55:50.771953.771953 cuda_h.py:27] end decode_layer cost 5.820 ms
DEBUG 05-06 11:55:50.772372.772372 lmp.py:1513] ---- decode step 1 layer 21 ----
DEBUG 05-06 11:55:50.777284.777284 cuda_h.py:27] end decode_layer cost 5.651 ms
DEBUG 05-06 11:55:50.777464.777464 lmp.py:1513] ---- decode step 1 layer 22 ----
DEBUG 05-06 11:55:50.783464.783464 cuda_h.py:27] end decode_layer cost 5.716 ms
DEBUG 05-06 11:55:50.783221.783221 lmp.py:1513] ---- decode step 1 layer 23 ----
DEBUG 05-06 11:55:50.789962.789962 cuda_h.py:27] end decode_layer cost 5.912 ms
DEBUG 05-06 11:55:50.789859.789859 lmp.py:1513] ---- decode step 1 layer 24 ----
DEBUG 05-06 11:55:50.795546.795546 cuda_h.py:27] end decode_layer cost 5.697 ms
DEBUG 05-06 11:55:50.795158.795158 lmp.py:1513] ---- decode step 1 layer 25 ----
DEBUG 05-06 11:55:50.800018.800018 cuda_h.py:27] end decode_layer cost 5.718 ms
DEBUG 05-06 11:55:50.801776.801776 lmp.py:1513] ---- decode step 1 layer 26 ----
DEBUG 05-06 11:55:50.806364.806364 cuda_h.py:27] end decode_layer cost 5.694 ms
DEBUG 05-06 11:55:50.806260.806260 lmp.py:1513] ---- decode step 1 layer 27 ----
DEBUG 05-06 11:55:50.812193.812193 cuda_h.py:27] end decode_layer cost 5.913 ms
DEBUG 05-06 11:55:50.812341.812341 lmp.py:1513] ---- decode step 1 layer 28 ----
DEBUG 05-06 11:55:50.818702.818702 cuda_h.py:27] end decode_layer cost 5.632 ms
DEBUG 05-06 11:55:50.818599.818599 lmp.py:1513] ---- decode step 1 layer 29 ----
DEBUG 05-06 11:55:50.824286.824286 cuda_h.py:27] end decode_layer cost 6.293 ms
DEBUG 05-06 11:55:50.824383.824383 cuda_h.py:27] end decode_step cost 186.677 ms
INFO 05-06 11:55:50.824437.824437 lmp.py:1561] decode step 1 time: 0.1867213249206543 seconds
WARNING 05-06 11:55:50.825605.825605 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:55:50.825526.825526 helper.py:35]   NaN count (hidden): 11264
WARNING 05-06 11:55:50.825455.825455 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:55:50.826770.826770 helper.py:39]   NaN count (normed): 11264
WARNING 05-06 11:55:50.831075.831075 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:55:50.831112.831112 helper.py:50]   NaN count: 1048576
WARNING 05-06 11:55:50.831366.831366 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:55:50.832013.832013 cuda_h.py:27] end init_inputs_tokens cost 7.735 ms
DEBUG 05-06 11:55:50.832240.832240 lmp.py:1507] decode step 2 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:55:50.832725.832725 lmp.py:1513] ---- decode step 2 layer 0 ----
DEBUG 05-06 11:55:50.838664.838664 cuda_h.py:27] end decode_layer cost 5.880 ms
DEBUG 05-06 11:55:50.838944.838944 lmp.py:1513] ---- decode step 2 layer 1 ----
DEBUG 05-06 11:55:50.844813.844813 cuda_h.py:27] end decode_layer cost 5.971 ms
DEBUG 05-06 11:55:50.844140.844140 lmp.py:1513] ---- decode step 2 layer 2 ----
DEBUG 05-06 11:55:50.850092.850092 cuda_h.py:27] end decode_layer cost 5.892 ms
DEBUG 05-06 11:55:50.850373.850373 lmp.py:1513] ---- decode step 2 layer 3 ----
DEBUG 05-06 11:55:50.856610.856610 cuda_h.py:27] end decode_layer cost 5.891 ms
DEBUG 05-06 11:55:50.856745.856745 lmp.py:1513] ---- decode step 2 layer 4 ----
DEBUG 05-06 11:55:50.862939.862939 cuda_h.py:27] end decode_layer cost 5.789 ms
DEBUG 05-06 11:55:50.862358.862358 lmp.py:1513] ---- decode step 2 layer 5 ----
DEBUG 05-06 11:55:50.869350.869350 cuda_h.py:27] end decode_layer cost 7.079 ms
DEBUG 05-06 11:55:50.869631.869631 lmp.py:1513] ---- decode step 2 layer 6 ----
DEBUG 05-06 11:55:50.875757.875757 cuda_h.py:27] end decode_layer cost 5.739 ms
DEBUG 05-06 11:55:50.875891.875891 lmp.py:1513] ---- decode step 2 layer 7 ----
DEBUG 05-06 11:55:50.881567.881567 cuda_h.py:27] end decode_layer cost 5.723 ms
DEBUG 05-06 11:55:50.881085.881085 lmp.py:1513] ---- decode step 2 layer 8 ----
DEBUG 05-06 11:55:50.887852.887852 cuda_h.py:27] end decode_layer cost 5.685 ms
DEBUG 05-06 11:55:50.887510.887510 lmp.py:1513] ---- decode step 2 layer 9 ----
DEBUG 05-06 11:55:50.893035.893035 cuda_h.py:27] end decode_layer cost 5.787 ms
DEBUG 05-06 11:55:50.893454.893454 lmp.py:1513] ---- decode step 2 layer 10 ----
DEBUG 05-06 11:55:50.898794.898794 cuda_h.py:27] end decode_layer cost 5.791 ms
DEBUG 05-06 11:55:50.898452.898452 lmp.py:1513] ---- decode step 2 layer 11 ----
DEBUG 05-06 11:55:50.905273.905273 cuda_h.py:27] end decode_layer cost 6.111 ms
DEBUG 05-06 11:55:50.905692.905692 lmp.py:1513] ---- decode step 2 layer 12 ----
DEBUG 05-06 11:55:50.910094.910094 cuda_h.py:27] end decode_layer cost 5.661 ms
DEBUG 05-06 11:55:50.910374.910374 lmp.py:1513] ---- decode step 2 layer 13 ----
DEBUG 05-06 11:55:50.916521.916521 cuda_h.py:27] end decode_layer cost 5.965 ms
DEBUG 05-06 11:55:50.916941.916941 lmp.py:1513] ---- decode step 2 layer 14 ----
DEBUG 05-06 11:55:50.922078.922078 cuda_h.py:27] end decode_layer cost 5.677 ms
DEBUG 05-06 11:55:50.922213.922213 lmp.py:1513] ---- decode step 2 layer 15 ----
DEBUG 05-06 11:55:50.928628.928628 cuda_h.py:27] end decode_layer cost 5.671 ms
DEBUG 05-06 11:55:50.928955.928955 lmp.py:1513] ---- decode step 2 layer 16 ----
DEBUG 05-06 11:55:50.934326.934326 cuda_h.py:27] end decode_layer cost 5.745 ms
DEBUG 05-06 11:55:50.934745.934745 lmp.py:1513] ---- decode step 2 layer 17 ----
DEBUG 05-06 11:55:50.940546.940546 cuda_h.py:27] end decode_layer cost 6.095 ms
DEBUG 05-06 11:55:50.940872.940872 lmp.py:1513] ---- decode step 2 layer 18 ----
DEBUG 05-06 11:55:50.946410.946410 cuda_h.py:27] end decode_layer cost 5.762 ms
DEBUG 05-06 11:55:50.946591.946591 lmp.py:1513] ---- decode step 2 layer 19 ----
DEBUG 05-06 11:55:50.952497.952497 cuda_h.py:27] end decode_layer cost 5.892 ms
DEBUG 05-06 11:55:50.952300.952300 lmp.py:1513] ---- decode step 2 layer 20 ----
DEBUG 05-06 11:55:50.957928.957928 cuda_h.py:27] end decode_layer cost 5.688 ms
DEBUG 05-06 11:55:50.957586.957586 lmp.py:1513] ---- decode step 2 layer 21 ----
DEBUG 05-06 11:55:50.963912.963912 cuda_h.py:27] end decode_layer cost 5.781 ms
DEBUG 05-06 11:55:50.963378.963378 lmp.py:1513] ---- decode step 2 layer 22 ----
DEBUG 05-06 11:55:50.969626.969626 cuda_h.py:27] end decode_layer cost 5.829 ms
DEBUG 05-06 11:55:50.969212.969212 lmp.py:1513] ---- decode step 2 layer 23 ----
DEBUG 05-06 11:55:50.975062.975062 cuda_h.py:27] end decode_layer cost 6.008 ms
DEBUG 05-06 11:55:50.975435.975435 lmp.py:1513] ---- decode step 2 layer 24 ----
DEBUG 05-06 11:55:50.981606.981606 cuda_h.py:27] end decode_layer cost 5.702 ms
DEBUG 05-06 11:55:50.981886.981886 lmp.py:1513] ---- decode step 2 layer 25 ----
DEBUG 05-06 11:55:50.987675.987675 cuda_h.py:27] end decode_layer cost 5.736 ms
DEBUG 05-06 11:55:50.987571.987571 lmp.py:1513] ---- decode step 2 layer 26 ----
DEBUG 05-06 11:55:50.993790.993790 cuda_h.py:27] end decode_layer cost 5.736 ms
DEBUG 05-06 11:55:50.993163.993163 lmp.py:1513] ---- decode step 2 layer 27 ----
DEBUG 05-06 11:55:50.998100.998100 cuda_h.py:27] end decode_layer cost 5.636 ms
DEBUG 05-06 11:55:50.998758.998758 lmp.py:1513] ---- decode step 2 layer 28 ----
DEBUG 05-06 11:55:51.004665.004665 cuda_h.py:27] end decode_layer cost 5.718 ms
DEBUG 05-06 11:55:51.004322.004322 lmp.py:1513] ---- decode step 2 layer 29 ----
DEBUG 05-06 11:55:51.010096.010096 cuda_h.py:27] end decode_layer cost 6.075 ms
DEBUG 05-06 11:55:51.010370.010370 cuda_h.py:27] end decode_step cost 185.742 ms
INFO 05-06 11:55:51.010709.010709 lmp.py:1561] decode step 2 time: 0.18578505516052246 seconds
Time taken: 7.600564256310463 seconds
generate input ids cost 0.03987526893615723 s
DEBUG 05-06 11:55:53.741455.741455 cuda_h.py:27] end generate_input_ids cost 2609.156 ms
DEBUG 05-06 11:55:53.742236.742236 cuda_h.py:27] end init_cache cost 0.036 ms
INFO 05-06 11:55:53.742400.742400 lmp.py:1160] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 11:55:53.755510.755510 lmp.py:2794] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 4740325316, 'cuda:1': 12875595776, 'cuda:2': 12875595776, 'cuda:3': 12875595776} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7886239265315211, 'cuda:1': 0.4700220660037874, 'cuda:2': 0.4700220660037874, 'cuda:3': 0.4700220660037874}
INFO 05-06 11:55:53.755805.755805 lmp.py:2812] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.755965.755965 lmp.py:2812] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.755781.755781 lmp.py:2812] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.755213.755213 lmp.py:2812] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.755314.755314 lmp.py:2812] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.755865.755865 lmp.py:2812] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.755350.755350 lmp.py:2812] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.756860.756860 lmp.py:2812] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.756060.756060 lmp.py:2812] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.756438.756438 lmp.py:2812] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.756730.756730 lmp.py:2812] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.756864.756864 lmp.py:2812] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.756442.756442 lmp.py:2812] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.756468.756468 lmp.py:2812] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.756615.756615 lmp.py:2812] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.757019.757019 lmp.py:2812] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.757212.757212 lmp.py:2812] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.757616.757616 lmp.py:2812] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.757286.757286 lmp.py:2812] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.757160.757160 lmp.py:2812] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.757592.757592 lmp.py:2812] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.757644.757644 lmp.py:2812] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.757076.757076 lmp.py:2812] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.758715.758715 lmp.py:2812] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.758891.758891 lmp.py:2812] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.758028.758028 lmp.py:2812] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.758221.758221 lmp.py:2812] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.758689.758689 lmp.py:2812] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.758121.758121 lmp.py:2812] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:55:53.758868.758868 lmp.py:2812] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 11:55:54.048737.048737 cuda_h.py:27] end init_loading_placement cost 305.694 ms
DEBUG 05-06 11:55:54.048166.048166 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:55:54.048281.048281 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:55:54 client.py:72] load_into_gpu: gemma4-26B-A4B, 00c6d345-f66e-4295-bfd6-5874e7b24bc6
INFO 05-06 11:55:54 client.py:135] Model loaded: gemma4-26B-A4B, 00c6d345-f66e-4295-bfd6-5874e7b24bc6
INFO 05-06 11:55:54 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 00c6d345-f66e-4295-bfd6-5874e7b24bc6
INFO 05-06 11:55:54 client.py:212] Model loaded
DEBUG 05-06 11:55:54.574109.574109 cuda_h.py:27] end init_general_sagl_loading_async cost 526.381 ms
INFO 05-06 11:55:54.624455.624455 lmp.py:3315] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 11:55:54.725960.725960 cuda_h.py:27] end restore_state_dict cost 100.386 ms
INFO 05-06 11:55:54.727750.727750 lmp.py:1288] vLLM Triton pre-warmup done in 2.6 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 11:55:54.727428.727428 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:55:54.728496.728496 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:55:54 client.py:72] load_into_gpu: gemma4-26B-A4B, fbee9f9c-d888-4c7d-8ca4-37d71155c112
INFO 05-06 11:55:54 client.py:135] Model loaded: gemma4-26B-A4B, fbee9f9c-d888-4c7d-8ca4-37d71155c112
DEBUG 05-06 11:55:54.860194.860194 cuda_h.py:27] end init_experts_loading_async cost 132.039 ms
DEBUG 05-06 11:55:54.861488.861488 cuda_h.py:27] end init_inputs_tokens cost 1.120 ms
DEBUG 05-06 11:55:54.861050.861050 lmp.py:1347] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 11:55:54.862549.862549 cuda_h.py:27] end prefill_ln cost 0.935 ms
DEBUG 05-06 11:55:54.869876.869876 cuda_h.py:27] end prefill_attn cost 6.438 ms
DEBUG 05-06 11:55:54.870243.870243 cuda_h.py:27] end prefill_ffn_prep cost 1.117 ms
DEBUG 05-06 11:55:54.873945.873945 cuda_h.py:27] end prefill_gate cost 0.982 ms
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 75, 115, 83, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 92, 116, 101, 109, 85, 49, 45, 65, 93, 69, 5, 13, 9, 73, 77, 37, 89, 25, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18, 70, 110, 118], 'token_total': 690, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 75: 29, 115: 29, 83: 33, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 92: 16, 116: 18, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 13: 16, 9: 17, 73: 19, 77: 19, 37: 20, 89: 20, 25: 24, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18, 70: 25, 110: 27, 118: 29}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 51, 55, 67, 91, 103, 127], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 1125, 'token_per_expert': {3: 46, 7: 95, 31: 34, 39: 176, 47: 318, 51: 48, 55: 51, 67: 47, 91: 99, 103: 178, 127: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 32, 48, 52, 64, 68, 72, 104, 112, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 724, 'token_per_expert': {0: 73, 16: 48, 32: 43, 48: 41, 52: 43, 64: 27, 68: 170, 72: 35, 104: 43, 112: 23, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 33, 41, 53, 105, 113, 117, 121, 125], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 26, 'token_total': 745, 'token_per_expert': {1: 75, 21: 48, 33: 210, 41: 27, 53: 205, 105: 24, 113: 39, 117: 26, 121: 65, 125: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 46, 50, 54, 74, 78, 90, 122, 126], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 25, 'token_total': 812, 'token_per_expert': {22: 64, 26: 59, 46: 119, 50: 110, 54: 59, 74: 61, 78: 36, 90: 154, 122: 35, 126: 115}}
INFO 05-06 11:55:54.874252.874252 lmp.py:1836] [layer_moe_fused] layer=0 prefix: 0.575ms alloc: 0.321ms
INFO 05-06 11:55:54.874733.874733 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 4.0531158447265625e-05 seconds
INFO 05-06 11:55:54.876454.876454 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=84 time: 0.0015864372253417969s
INFO 05-06 11:55:54.877923.877923 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006875991821289062 seconds
DEBUG 05-06 11:55:54.877291.877291 cuda_h.py:27] end moe_cpu_prep_submit cost 0.783 ms
INFO 05-06 11:55:54.994509.994509 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.11129021644592285s
DEBUG 05-06 11:55:54.994822.994822 cuda_h.py:27] end moe_wait_copy_tasks cost 111.590 ms
DEBUG 05-06 11:55:55.011000.011000 cuda_h.py:27] end moe_vllm_forward cost 15.393 ms
DEBUG 05-06 11:55:55.011558.011558 cuda_h.py:27] end moe_cpu_merge cost 0.075 ms
DEBUG 05-06 11:55:55.011974.011974 cuda_h.py:27] end moe_shared_experts cost 0.008 ms
INFO 05-06 11:55:55.011375.011375 lmp.py:1950] [layer_moe_fused] vllm triton time: 17.006ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.012187.012187 cuda_h.py:27] end *layer_moe_fused cost 138.607 ms
DEBUG 05-06 11:55:55.012400.012400 cuda_h.py:27] end prefill_merge_scale cost 0.556 ms
DEBUG 05-06 11:55:55.013369.013369 cuda_h.py:27] end prefill_layer cost 151.543 ms
DEBUG 05-06 11:55:55.013556.013556 lmp.py:1391] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 11:55:55.013052.013052 lmp.py:1347] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 11:55:55.013160.013160 cuda_h.py:27] end prefill_ln cost 0.221 ms
DEBUG 05-06 11:55:55.016959.016959 cuda_h.py:27] end prefill_attn cost 2.637 ms
DEBUG 05-06 11:55:55.016697.016697 cuda_h.py:27] end prefill_ffn_prep cost 0.397 ms
DEBUG 05-06 11:55:55.017925.017925 cuda_h.py:27] end prefill_gate cost 0.436 ms
experts_cpu_alloc {'expert_ids': [43, 39, 55, 31, 103, 123, 83, 91, 115, 11, 87, 27, 95, 108, 16, 40, 88, 32, 60, 84, 44, 72, 112, 116, 56, 48, 76, 92, 124, 64, 104, 96, 29, 61, 45, 117, 33, 41, 81, 121, 57, 89, 125, 37, 93, 101, 69, 85, 9, 21, 114, 62, 110, 18, 26, 38, 14, 50, 66, 78, 74, 90, 98, 94, 34, 42], 'token_total': 503, 'token_per_expert': {43: 1, 39: 2, 55: 2, 31: 3, 103: 3, 123: 3, 83: 5, 91: 5, 115: 6, 11: 8, 87: 8, 27: 9, 95: 9, 108: 1, 16: 2, 40: 3, 88: 3, 32: 4, 60: 4, 84: 4, 44: 5, 72: 5, 112: 7, 116: 7, 56: 9, 48: 10, 76: 10, 92: 10, 124: 12, 64: 16, 104: 23, 96: 31, 29: 1, 61: 1, 45: 2, 117: 2, 33: 3, 41: 3, 81: 3, 121: 3, 57: 4, 89: 5, 125: 6, 37: 7, 93: 8, 101: 8, 69: 9, 85: 11, 9: 13, 21: 16, 114: 1, 62: 2, 110: 2, 18: 3, 26: 3, 38: 5, 14: 7, 50: 8, 66: 11, 78: 12, 74: 14, 90: 14, 98: 17, 94: 19, 34: 24, 42: 26}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 47, 51, 59, 67, 79, 99, 119, 127], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 24, 'token_total': 687, 'token_per_expert': {3: 139, 7: 163, 35: 19, 47: 39, 51: 49, 59: 21, 67: 98, 79: 20, 99: 88, 119: 17, 127: 34}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 28, 52, 68, 80, 100, 120], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 30, 'token_total': 993, 'token_per_expert': {0: 134, 4: 136, 8: 92, 12: 39, 20: 53, 28: 50, 52: 228, 68: 154, 80: 38, 100: 38, 120: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 49, 53, 65, 73, 97, 105, 109], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 908, 'token_per_expert': {1: 148, 5: 205, 13: 207, 25: 39, 49: 24, 53: 34, 65: 37, 73: 25, 97: 98, 105: 17, 109: 74}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 30, 46, 54, 82, 106, 118, 122], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 1005, 'token_per_expert': {2: 131, 6: 130, 10: 120, 22: 86, 30: 156, 46: 32, 54: 34, 82: 142, 106: 34, 118: 49, 122: 91}}
INFO 05-06 11:55:55.018571.018571 lmp.py:1836] [layer_moe_fused] layer=1 prefix: 0.398ms alloc: 0.241ms
INFO 05-06 11:55:55.018820.018820 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 3.361701965332031e-05 seconds
INFO 05-06 11:55:55.020839.020839 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0017559528350830078s
INFO 05-06 11:55:55.021813.021813 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006906986236572266 seconds
DEBUG 05-06 11:55:55.023633.023633 cuda_h.py:27] end moe_cpu_prep_submit cost 0.765 ms
INFO 05-06 11:55:55.038848.038848 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014728546142578125s
DEBUG 05-06 11:55:55.038046.038046 cuda_h.py:27] end moe_wait_copy_tasks cost 14.940 ms
DEBUG 05-06 11:55:55.044154.044154 cuda_h.py:27] end moe_vllm_forward cost 4.589 ms
DEBUG 05-06 11:55:55.044199.044199 cuda_h.py:27] end moe_cpu_merge cost 0.079 ms
DEBUG 05-06 11:55:55.044631.044631 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:55:55.044793.044793 lmp.py:1950] [layer_moe_fused] vllm triton time: 5.817ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.045187.045187 cuda_h.py:27] end *layer_moe_fused cost 27.358 ms
DEBUG 05-06 11:55:55.050809.050809 cuda_h.py:27] end prefill_merge_scale cost 5.235 ms
DEBUG 05-06 11:55:55.050388.050388 cuda_h.py:27] end prefill_layer cost 37.533 ms
DEBUG 05-06 11:55:55.051882.051882 lmp.py:1391] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 11:55:55.051506.051506 lmp.py:1347] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 11:55:55.051818.051818 cuda_h.py:27] end prefill_ln cost 0.242 ms
DEBUG 05-06 11:55:55.053190.053190 cuda_h.py:27] end prefill_attn cost 2.120 ms
DEBUG 05-06 11:55:55.054319.054319 cuda_h.py:27] end prefill_ffn_prep cost 0.395 ms
DEBUG 05-06 11:55:55.055435.055435 cuda_h.py:27] end prefill_gate cost 0.479 ms
experts_cpu_alloc {'expert_ids': [75, 79, 27, 95, 115, 23, 99, 111, 63, 119, 35, 71, 43, 103, 51, 83, 107, 68, 12, 40, 120, 116, 88, 96, 64, 72, 44, 52, 56, 124, 36, 24, 100, 8, 25, 45, 21, 113, 61, 121, 69, 77, 33, 85, 105, 17, 57, 49, 65, 97, 86, 26, 42, 50, 114, 70, 82, 126, 98, 46, 58, 122, 78, 110], 'token_total': 678, 'token_per_expert': {75: 1, 79: 1, 27: 2, 95: 3, 115: 3, 23: 4, 99: 4, 111: 4, 63: 10, 119: 11, 35: 12, 71: 13, 43: 15, 103: 16, 51: 17, 83: 19, 107: 19, 68: 1, 12: 2, 40: 2, 120: 5, 116: 6, 88: 7, 96: 8, 64: 10, 72: 10, 44: 11, 52: 11, 56: 11, 124: 11, 36: 12, 24: 14, 100: 15, 8: 19, 25: 3, 45: 4, 21: 5, 113: 6, 61: 7, 121: 8, 69: 9, 77: 11, 33: 12, 85: 13, 105: 13, 17: 15, 57: 15, 49: 18, 65: 28, 97: 34, 86: 1, 26: 3, 42: 3, 50: 4, 114: 4, 70: 9, 82: 10, 126: 12, 98: 15, 46: 17, 58: 18, 122: 22, 78: 24, 110: 26}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 31, 55, 59, 91, 123, 127], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 930, 'token_per_expert': {3: 142, 7: 182, 11: 136, 15: 75, 19: 80, 31: 22, 55: 49, 59: 100, 91: 25, 123: 20, 127: 99}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 28, 48, 60, 76, 80, 84, 104, 108], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 761, 'token_per_expert': {0: 141, 4: 138, 20: 38, 28: 19, 48: 48, 60: 38, 76: 64, 80: 26, 84: 48, 104: 38, 108: 163}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 29, 37, 41, 53, 81, 109, 125], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 923, 'token_per_expert': {1: 213, 5: 132, 9: 78, 13: 81, 29: 68, 37: 55, 41: 111, 53: 35, 81: 54, 109: 38, 125: 58}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 34, 54, 62, 90, 102, 106, 118], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 804, 'token_per_expert': {2: 128, 6: 131, 14: 32, 18: 57, 34: 36, 54: 105, 62: 103, 90: 64, 102: 73, 106: 32, 118: 43}}
INFO 05-06 11:55:55.056211.056211 lmp.py:1836] [layer_moe_fused] layer=2 prefix: 0.471ms alloc: 0.431ms
INFO 05-06 11:55:55.056699.056699 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.175041198730469e-05 seconds
INFO 05-06 11:55:55.057280.057280 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008170604705810547s
INFO 05-06 11:55:55.058840.058840 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006284713745117188 seconds
DEBUG 05-06 11:55:55.058074.058074 cuda_h.py:27] end moe_cpu_prep_submit cost 0.896 ms
INFO 05-06 11:55:55.072053.072053 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013298749923706055s
DEBUG 05-06 11:55:55.072356.072356 cuda_h.py:27] end moe_wait_copy_tasks cost 13.474 ms
DEBUG 05-06 11:55:55.077423.077423 cuda_h.py:27] end moe_vllm_forward cost 4.314 ms
DEBUG 05-06 11:55:55.077594.077594 cuda_h.py:27] end moe_cpu_merge cost 0.066 ms
DEBUG 05-06 11:55:55.077983.077983 cuda_h.py:27] end moe_shared_experts cost 0.007 ms
INFO 05-06 11:55:55.077183.077183 lmp.py:1950] [layer_moe_fused] vllm triton time: 5.273ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.078332.078332 cuda_h.py:27] end *layer_moe_fused cost 22.554 ms
DEBUG 05-06 11:55:55.082198.082198 cuda_h.py:27] end prefill_merge_scale cost 4.436 ms
DEBUG 05-06 11:55:55.082002.082002 cuda_h.py:27] end prefill_layer cost 31.650 ms
DEBUG 05-06 11:55:55.083602.083602 lmp.py:1391] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 11:55:55.083987.083987 lmp.py:1347] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 11:55:55.083491.083491 cuda_h.py:27] end prefill_ln cost 0.206 ms
DEBUG 05-06 11:55:55.085379.085379 cuda_h.py:27] end prefill_attn cost 2.120 ms
DEBUG 05-06 11:55:55.086004.086004 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 11:55:55.087717.087717 cuda_h.py:27] end prefill_gate cost 0.498 ms
experts_cpu_alloc {'expert_ids': [35, 55, 87, 91, 127, 23, 27, 63, 31, 67, 43, 119, 123, 107, 59, 19, 20, 72, 36, 16, 32, 48, 60, 40, 100, 116, 8, 24, 56, 64, 44, 108, 88, 125, 29, 57, 65, 89, 41, 117, 33, 13, 101, 61, 77, 109, 25, 46, 82, 94, 98, 126, 18, 30, 42, 86, 26, 110, 114, 74, 58, 122, 54, 70, 118], 'token_total': 794, 'token_per_expert': {35: 1, 55: 1, 87: 1, 91: 1, 127: 2, 23: 4, 27: 6, 63: 7, 31: 8, 67: 8, 43: 11, 119: 16, 123: 18, 107: 19, 59: 21, 19: 22, 20: 1, 72: 1, 36: 3, 16: 4, 32: 5, 48: 7, 60: 8, 40: 9, 100: 9, 116: 9, 8: 10, 24: 10, 56: 10, 64: 14, 44: 25, 108: 33, 88: 40, 125: 1, 29: 3, 57: 3, 65: 3, 89: 4, 41: 5, 117: 6, 33: 7, 13: 9, 101: 13, 61: 16, 77: 17, 109: 31, 25: 34, 46: 1, 82: 2, 94: 3, 98: 4, 126: 4, 18: 6, 30: 8, 42: 9, 86: 10, 26: 17, 110: 22, 114: 22, 74: 24, 58: 29, 122: 30, 54: 32, 70: 32, 118: 43}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 39, 51, 71, 75, 83, 95, 111], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 652, 'token_per_expert': {3: 155, 7: 128, 11: 34, 15: 27, 39: 28, 51: 34, 71: 61, 75: 74, 83: 42, 95: 43, 111: 26}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 52, 68, 76, 84, 92, 96, 104, 120], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 808, 'token_per_expert': {0: 154, 4: 174, 28: 78, 52: 56, 68: 41, 76: 47, 84: 57, 92: 54, 96: 53, 104: 43, 120: 51}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 53, 69, 73, 85, 93, 97, 121], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 814, 'token_per_expert': {1: 136, 5: 183, 9: 45, 17: 35, 53: 48, 69: 38, 73: 34, 85: 106, 93: 53, 97: 56, 121: 80}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 34, 50, 62, 66, 78, 102], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 1028, 'token_per_expert': {2: 141, 6: 141, 10: 44, 14: 70, 22: 95, 34: 50, 50: 145, 62: 89, 66: 76, 78: 93, 102: 84}}
INFO 05-06 11:55:55.088394.088394 lmp.py:1836] [layer_moe_fused] layer=3 prefix: 0.528ms alloc: 0.430ms
INFO 05-06 11:55:55.088035.088035 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.651878356933594e-05 seconds
INFO 05-06 11:55:55.089627.089627 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008437633514404297s
INFO 05-06 11:55:55.090439.090439 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006389617919921875 seconds
DEBUG 05-06 11:55:55.090638.090638 cuda_h.py:27] end moe_cpu_prep_submit cost 0.759 ms
INFO 05-06 11:55:55.104247.104247 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012722969055175781s
DEBUG 05-06 11:55:55.104597.104597 cuda_h.py:27] end moe_wait_copy_tasks cost 12.899 ms
DEBUG 05-06 11:55:55.108555.108555 cuda_h.py:27] end moe_vllm_forward cost 4.044 ms
DEBUG 05-06 11:55:55.108481.108481 cuda_h.py:27] end moe_cpu_merge cost 0.062 ms
DEBUG 05-06 11:55:55.109031.109031 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.109709.109709 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.888ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.109651.109651 cuda_h.py:27] end *layer_moe_fused cost 22.078 ms
DEBUG 05-06 11:55:55.115527.115527 cuda_h.py:27] end prefill_merge_scale cost 5.534 ms
DEBUG 05-06 11:55:55.115001.115001 cuda_h.py:27] end prefill_layer cost 32.241 ms
DEBUG 05-06 11:55:55.115107.115107 lmp.py:1391] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 11:55:55.115175.115175 lmp.py:1347] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 11:55:55.116943.116943 cuda_h.py:27] end prefill_ln cost 0.210 ms
DEBUG 05-06 11:55:55.118351.118351 cuda_h.py:27] end prefill_attn cost 2.023 ms
DEBUG 05-06 11:55:55.118413.118413 cuda_h.py:27] end prefill_ffn_prep cost 0.381 ms
DEBUG 05-06 11:55:55.119313.119313 cuda_h.py:27] end prefill_gate cost 0.524 ms
experts_cpu_alloc {'expert_ids': [95, 31, 79, 103, 107, 91, 15, 75, 123, 47, 19, 87, 71, 39, 55, 67, 27, 12, 120, 80, 56, 44, 36, 84, 40, 64, 88, 108, 116, 52, 96, 28, 41, 121, 69, 45, 21, 109, 101, 37, 77, 25, 73, 81, 117, 17, 57, 61, 97, 46, 50, 66, 110, 18, 58, 70, 122, 114, 126, 94, 38, 90, 34, 78, 30, 62], 'token_total': 803, 'token_per_expert': {95: 1, 31: 2, 79: 3, 103: 7, 107: 12, 91: 14, 15: 16, 75: 17, 123: 22, 47: 25, 19: 26, 87: 27, 71: 31, 39: 33, 55: 44, 67: 49, 27: 57, 12: 1, 120: 3, 80: 4, 56: 5, 44: 7, 36: 10, 84: 12, 40: 13, 64: 14, 88: 14, 108: 14, 116: 15, 52: 19, 96: 21, 28: 25, 41: 1, 121: 1, 69: 3, 45: 4, 21: 5, 109: 5, 101: 6, 37: 7, 77: 8, 25: 9, 73: 9, 81: 9, 117: 11, 17: 12, 57: 12, 61: 12, 97: 13, 46: 1, 50: 1, 66: 1, 110: 1, 18: 2, 58: 2, 70: 2, 122: 3, 114: 4, 126: 6, 94: 8, 38: 10, 90: 10, 34: 12, 78: 14, 30: 18, 62: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 43, 51, 59, 63, 83, 111, 115, 119], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 1177, 'token_per_expert': {3: 156, 7: 130, 23: 92, 43: 114, 51: 61, 59: 116, 63: 200, 83: 57, 111: 69, 115: 70, 119: 112}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 24, 32, 60, 76, 92, 104, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 700, 'token_per_expert': {0: 128, 4: 155, 8: 132, 20: 34, 24: 79, 32: 29, 60: 27, 76: 36, 92: 25, 104: 27, 124: 28}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 49, 53, 85, 89, 93, 105, 113, 125], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 723, 'token_per_expert': {1: 190, 5: 157, 29: 40, 49: 31, 53: 39, 85: 31, 89: 89, 93: 37, 105: 21, 113: 51, 125: 37}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 54, 74, 82, 86, 98, 106, 118], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 693, 'token_per_expert': {2: 129, 6: 132, 22: 73, 26: 59, 54: 45, 74: 79, 82: 41, 86: 19, 98: 21, 106: 75, 118: 20}}
INFO 05-06 11:55:55.120921.120921 lmp.py:1836] [layer_moe_fused] layer=4 prefix: 0.463ms alloc: 0.425ms
INFO 05-06 11:55:55.121456.121456 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.389617919921875e-05 seconds
INFO 05-06 11:55:55.122636.122636 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008170604705810547s
INFO 05-06 11:55:55.122791.122791 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006115436553955078 seconds
DEBUG 05-06 11:55:55.123201.123201 cuda_h.py:27] end moe_cpu_prep_submit cost 0.947 ms
INFO 05-06 11:55:55.137829.137829 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014696121215820312s
DEBUG 05-06 11:55:55.138986.138986 cuda_h.py:27] end moe_wait_copy_tasks cost 14.867 ms
DEBUG 05-06 11:55:55.142446.142446 cuda_h.py:27] end moe_vllm_forward cost 3.999 ms
DEBUG 05-06 11:55:55.142326.142326 cuda_h.py:27] end moe_cpu_merge cost 0.064 ms
DEBUG 05-06 11:55:55.142442.142442 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.143597.143597 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.940ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.143811.143811 cuda_h.py:27] end *layer_moe_fused cost 23.528 ms
DEBUG 05-06 11:55:55.148463.148463 cuda_h.py:27] end prefill_merge_scale cost 4.772 ms
DEBUG 05-06 11:55:55.148645.148645 cuda_h.py:27] end prefill_layer cost 32.838 ms
DEBUG 05-06 11:55:55.148919.148919 lmp.py:1391] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 11:55:55.148205.148205 lmp.py:1347] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 11:55:55.149788.149788 cuda_h.py:27] end prefill_ln cost 0.210 ms
DEBUG 05-06 11:55:55.154161.154161 cuda_h.py:27] end prefill_attn cost 4.946 ms
DEBUG 05-06 11:55:55.154508.154508 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 11:55:55.155908.155908 cuda_h.py:27] end prefill_gate cost 0.443 ms
experts_cpu_alloc {'expert_ids': [15, 51, 19, 27, 115, 83, 67, 107, 119, 79, 75, 31, 55, 63, 92, 8, 32, 48, 124, 56, 68, 84, 44, 52, 100, 80, 120, 96, 104, 60, 76, 116, 17, 21, 45, 81, 105, 77, 57, 53, 37, 113, 125, 29, 78, 82, 30, 38, 50, 26, 54, 110, 58, 10, 86, 114, 34, 102, 106, 62, 98, 18, 14], 'token_total': 583, 'token_per_expert': {15: 1, 51: 2, 19: 3, 27: 3, 115: 4, 83: 10, 67: 11, 107: 11, 119: 11, 79: 14, 75: 16, 31: 18, 55: 20, 63: 23, 92: 1, 8: 2, 32: 2, 48: 2, 124: 2, 56: 3, 68: 6, 84: 7, 44: 9, 52: 9, 100: 17, 80: 19, 120: 25, 96: 26, 104: 28, 60: 29, 76: 30, 116: 33, 17: 1, 21: 1, 45: 1, 81: 1, 105: 1, 77: 3, 57: 4, 53: 5, 37: 7, 113: 8, 125: 22, 29: 26, 78: 1, 82: 1, 30: 2, 38: 2, 50: 3, 26: 4, 54: 4, 110: 4, 58: 5, 10: 6, 86: 6, 114: 6, 34: 7, 102: 7, 106: 7, 62: 8, 98: 9, 18: 10, 14: 14}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 43, 71, 87, 99, 111, 123, 127], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 925, 'token_per_expert': {3: 128, 7: 138, 23: 29, 39: 116, 43: 26, 71: 214, 87: 50, 99: 45, 111: 54, 123: 46, 127: 79}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 28, 36, 64, 72, 88, 112], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 965, 'token_per_expert': {0: 142, 4: 191, 16: 95, 20: 107, 24: 55, 28: 53, 36: 60, 64: 71, 72: 52, 88: 34, 112: 105}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 33, 49, 61, 73, 93, 101, 117], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 23, 'token_total': 992, 'token_per_expert': {1: 128, 5: 156, 9: 47, 13: 58, 33: 83, 49: 125, 61: 52, 73: 37, 93: 38, 101: 222, 117: 46}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 42, 46, 70, 74, 94, 118, 126], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 29, 'token_total': 631, 'token_per_expert': {2: 208, 6: 140, 22: 60, 42: 52, 46: 24, 70: 33, 74: 28, 94: 34, 118: 21, 126: 31}}
INFO 05-06 11:55:55.156707.156707 lmp.py:1836] [layer_moe_fused] layer=5 prefix: 0.441ms alloc: 0.405ms
INFO 05-06 11:55:55.156434.156434 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.389617919921875e-05 seconds
INFO 05-06 11:55:55.157170.157170 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008218288421630859s
INFO 05-06 11:55:55.158020.158020 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005974769592285156 seconds
DEBUG 05-06 11:55:55.158482.158482 cuda_h.py:27] end moe_cpu_prep_submit cost 0.925 ms
INFO 05-06 11:55:55.170248.170248 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.011501550674438477s
DEBUG 05-06 11:55:55.170577.170577 cuda_h.py:27] end moe_wait_copy_tasks cost 11.658 ms
DEBUG 05-06 11:55:55.175154.175154 cuda_h.py:27] end moe_vllm_forward cost 3.964 ms
DEBUG 05-06 11:55:55.175881.175881 cuda_h.py:27] end moe_cpu_merge cost 0.063 ms
DEBUG 05-06 11:55:55.175033.175033 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.175519.175519 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.764ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.176173.176173 cuda_h.py:27] end *layer_moe_fused cost 20.118 ms
DEBUG 05-06 11:55:55.181278.181278 cuda_h.py:27] end prefill_merge_scale cost 5.036 ms
DEBUG 05-06 11:55:55.181983.181983 cuda_h.py:27] end prefill_layer cost 32.523 ms
DEBUG 05-06 11:55:55.181243.181243 lmp.py:1391] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 11:55:55.181437.181437 lmp.py:1347] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 11:55:55.181469.181469 cuda_h.py:27] end prefill_ln cost 0.207 ms
DEBUG 05-06 11:55:55.183332.183332 cuda_h.py:27] end prefill_attn cost 1.974 ms
DEBUG 05-06 11:55:55.184957.184957 cuda_h.py:27] end prefill_ffn_prep cost 0.378 ms
DEBUG 05-06 11:55:55.185304.185304 cuda_h.py:27] end prefill_gate cost 0.447 ms
experts_cpu_alloc {'expert_ids': [31, 47, 59, 67, 111, 83, 11, 15, 19, 43, 91, 127, 51, 103, 123, 27, 71, 8, 88, 92, 112, 124, 52, 72, 40, 20, 120, 16, 60, 80, 76, 116, 24, 17, 101, 109, 49, 81, 33, 97, 89, 37, 41, 125, 105, 77, 85, 57, 73, 113, 114, 22, 82, 38, 18, 30, 74, 110, 42, 14, 126, 70, 10, 58, 78, 50, 46, 122], 'token_total': 629, 'token_per_expert': {31: 1, 47: 2, 59: 2, 67: 2, 111: 3, 83: 4, 11: 6, 15: 6, 19: 10, 43: 10, 91: 10, 127: 13, 51: 16, 103: 19, 123: 19, 27: 20, 71: 21, 8: 1, 88: 1, 92: 1, 112: 1, 124: 1, 52: 2, 72: 2, 40: 3, 20: 4, 120: 4, 16: 6, 60: 6, 80: 6, 76: 10, 116: 12, 24: 17, 17: 1, 101: 1, 109: 1, 49: 2, 81: 3, 33: 4, 97: 4, 89: 8, 37: 9, 41: 10, 125: 11, 105: 12, 77: 13, 85: 13, 57: 14, 73: 16, 113: 19, 114: 1, 22: 2, 82: 2, 38: 3, 18: 5, 30: 7, 74: 8, 110: 9, 42: 10, 14: 12, 126: 17, 70: 18, 10: 20, 58: 20, 78: 24, 50: 27, 46: 31, 122: 31}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 35, 75, 79, 87, 95, 99, 107, 115, 119], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 29, 'token_total': 886, 'token_per_expert': {3: 130, 7: 129, 23: 70, 35: 76, 75: 32, 79: 35, 87: 71, 95: 21, 99: 173, 107: 28, 115: 78, 119: 43}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 32, 36, 44, 56, 64, 68, 96, 104, 108], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 28, 'token_total': 843, 'token_per_expert': {0: 139, 4: 131, 28: 23, 32: 26, 36: 33, 44: 24, 56: 19, 64: 81, 68: 216, 96: 31, 104: 27, 108: 93}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 53, 65, 69, 93, 117, 121], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 899, 'token_per_expert': {1: 154, 5: 145, 9: 34, 13: 52, 25: 177, 53: 73, 65: 84, 69: 22, 93: 104, 117: 21, 121: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 34, 62, 86, 90, 94, 98, 102, 106], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 839, 'token_per_expert': {2: 150, 6: 141, 26: 43, 34: 43, 62: 33, 86: 82, 90: 72, 94: 54, 98: 53, 102: 73, 106: 95}}
INFO 05-06 11:55:55.186416.186416 lmp.py:1836] [layer_moe_fused] layer=6 prefix: 0.470ms alloc: 0.437ms
INFO 05-06 11:55:55.186143.186143 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.556510925292969e-05 seconds
INFO 05-06 11:55:55.187398.187398 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008029937744140625s
INFO 05-06 11:55:55.188513.188513 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005819797515869141 seconds
DEBUG 05-06 11:55:55.188548.188548 cuda_h.py:27] end moe_cpu_prep_submit cost 0.822 ms
INFO 05-06 11:55:55.202017.202017 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013524055480957031s
DEBUG 05-06 11:55:55.202492.202492 cuda_h.py:27] end moe_wait_copy_tasks cost 13.684 ms
DEBUG 05-06 11:55:55.207271.207271 cuda_h.py:27] end moe_vllm_forward cost 4.025 ms
DEBUG 05-06 11:55:55.207197.207197 cuda_h.py:27] end moe_cpu_merge cost 0.065 ms
DEBUG 05-06 11:55:55.207694.207694 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.207181.207181 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.892ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.208503.208503 cuda_h.py:27] end *layer_moe_fused cost 22.287 ms
DEBUG 05-06 11:55:55.213550.213550 cuda_h.py:27] end prefill_merge_scale cost 5.695 ms
DEBUG 05-06 11:55:55.213771.213771 cuda_h.py:27] end prefill_layer cost 32.338 ms
DEBUG 05-06 11:55:55.214304.214304 lmp.py:1391] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 11:55:55.214067.214067 lmp.py:1347] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 11:55:55.214007.214007 cuda_h.py:27] end prefill_ln cost 0.224 ms
DEBUG 05-06 11:55:55.216027.216027 cuda_h.py:27] end prefill_attn cost 2.123 ms
DEBUG 05-06 11:55:55.217221.217221 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 11:55:55.218485.218485 cuda_h.py:27] end prefill_gate cost 0.494 ms
experts_cpu_alloc {'expert_ids': [27, 31, 67, 75, 119, 35, 15, 55, 107, 23, 63, 127, 95, 111, 83, 87, 115, 51, 100, 32, 92, 16, 88, 112, 8, 80, 64, 116, 68, 72, 48, 96, 20, 49, 45, 109, 37, 21, 77, 17, 101, 41, 25, 117, 9, 125, 13, 61, 33, 113, 30, 50, 62, 38, 94, 26, 78, 66, 54, 82, 126, 98, 118, 122, 18, 22, 106], 'token_total': 811, 'token_per_expert': {27: 1, 31: 1, 67: 1, 75: 1, 119: 1, 35: 3, 15: 4, 55: 4, 107: 4, 23: 8, 63: 8, 127: 8, 95: 10, 111: 11, 83: 13, 87: 15, 115: 15, 51: 16, 100: 1, 32: 3, 92: 5, 16: 10, 88: 10, 112: 13, 8: 14, 80: 15, 64: 16, 116: 16, 68: 20, 72: 23, 48: 34, 96: 37, 20: 39, 49: 2, 45: 6, 109: 6, 37: 7, 21: 8, 77: 8, 17: 9, 101: 9, 41: 10, 25: 14, 117: 15, 9: 16, 125: 17, 13: 18, 61: 23, 33: 33, 113: 34, 30: 2, 50: 2, 62: 2, 38: 4, 94: 5, 26: 7, 78: 7, 66: 9, 54: 11, 82: 11, 126: 11, 98: 14, 118: 16, 122: 17, 18: 23, 22: 27, 106: 28}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 43, 47, 59, 71, 79, 91, 99, 103, 123], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 30, 'token_total': 641, 'token_per_expert': {3: 128, 7: 158, 19: 18, 43: 24, 47: 20, 59: 18, 71: 25, 79: 47, 91: 139, 99: 17, 103: 28, 123: 19}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 28, 44, 52, 56, 60, 84, 104, 108, 120], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 27, 'token_total': 909, 'token_per_expert': {0: 132, 4: 180, 12: 104, 28: 51, 44: 52, 52: 52, 56: 41, 60: 48, 84: 73, 104: 40, 108: 80, 120: 56}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 53, 57, 65, 69, 85, 97, 105, 121], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 920, 'token_per_expert': {1: 128, 5: 156, 29: 68, 53: 43, 57: 40, 65: 69, 69: 67, 85: 45, 97: 173, 105: 35, 121: 96}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 34, 42, 70, 86, 90, 110, 114], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 815, 'token_per_expert': {2: 128, 6: 145, 10: 77, 14: 55, 34: 56, 42: 55, 70: 89, 86: 41, 90: 66, 110: 50, 114: 53}}
INFO 05-06 11:55:55.219119.219119 lmp.py:1836] [layer_moe_fused] layer=7 prefix: 0.444ms alloc: 0.427ms
INFO 05-06 11:55:55.219495.219495 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.747245788574219e-05 seconds
INFO 05-06 11:55:55.220782.220782 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008084774017333984s
INFO 05-06 11:55:55.221327.221327 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005850791931152344 seconds
DEBUG 05-06 11:55:55.221098.221098 cuda_h.py:27] end moe_cpu_prep_submit cost 0.845 ms
INFO 05-06 11:55:55.235149.235149 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013276338577270508s
DEBUG 05-06 11:55:55.235114.235114 cuda_h.py:27] end moe_wait_copy_tasks cost 13.447 ms
DEBUG 05-06 11:55:55.239615.239615 cuda_h.py:27] end moe_vllm_forward cost 4.013 ms
DEBUG 05-06 11:55:55.239640.239640 cuda_h.py:27] end moe_cpu_merge cost 0.066 ms
DEBUG 05-06 11:55:55.240249.240249 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 11:55:55.240549.240549 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.846ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.240824.240824 cuda_h.py:27] end *layer_moe_fused cost 21.936 ms
DEBUG 05-06 11:55:55.246166.246166 cuda_h.py:27] end prefill_merge_scale cost 5.807 ms
DEBUG 05-06 11:55:55.246587.246587 cuda_h.py:27] end prefill_layer cost 32.333 ms
DEBUG 05-06 11:55:55.246900.246900 lmp.py:1391] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 11:55:55.246663.246663 lmp.py:1347] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 11:55:55.247390.247390 cuda_h.py:27] end prefill_ln cost 0.212 ms
DEBUG 05-06 11:55:55.249729.249729 cuda_h.py:27] end prefill_attn cost 1.941 ms
DEBUG 05-06 11:55:55.249155.249155 cuda_h.py:27] end prefill_ffn_prep cost 0.376 ms
DEBUG 05-06 11:55:55.250562.250562 cuda_h.py:27] end prefill_gate cost 0.460 ms
experts_cpu_alloc {'expert_ids': [79, 23, 35, 39, 91, 119, 43, 99, 47, 11, 127, 31, 55, 63, 24, 60, 88, 48, 72, 8, 104, 64, 96, 124, 84, 20, 44, 108, 116, 68, 92, 9, 25, 13, 101, 33, 37, 89, 49, 117, 21, 17, 29, 113, 41, 45, 57, 93, 85, 69, 18, 26, 90, 106, 82, 118, 62, 34, 74, 86, 22, 66, 42, 10, 14, 98, 126, 122], 'token_total': 655, 'token_per_expert': {79: 1, 23: 3, 35: 4, 39: 5, 91: 5, 119: 7, 43: 8, 99: 8, 47: 9, 11: 14, 127: 14, 31: 16, 55: 23, 63: 26, 24: 1, 60: 1, 88: 1, 48: 2, 72: 2, 8: 3, 104: 3, 64: 4, 96: 4, 124: 6, 84: 7, 20: 9, 44: 9, 108: 9, 116: 9, 68: 10, 92: 10, 9: 2, 25: 3, 13: 4, 101: 4, 33: 6, 37: 6, 89: 8, 49: 9, 117: 10, 21: 11, 17: 12, 29: 13, 113: 14, 41: 15, 45: 17, 57: 19, 93: 19, 85: 20, 69: 22, 18: 2, 26: 2, 90: 2, 106: 3, 82: 5, 118: 5, 62: 6, 34: 8, 74: 8, 86: 8, 22: 12, 66: 13, 42: 14, 10: 19, 14: 20, 98: 21, 126: 22, 122: 38}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 51, 71, 75, 87, 103, 111, 123], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 26, 'token_total': 940, 'token_per_expert': {3: 158, 7: 128, 15: 29, 19: 63, 27: 36, 51: 136, 71: 55, 75: 60, 87: 79, 103: 139, 111: 26, 123: 31}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 28, 32, 36, 52, 56, 76, 80, 120], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 29, 'token_total': 709, 'token_per_expert': {0: 129, 4: 142, 12: 44, 16: 25, 28: 80, 32: 44, 36: 27, 52: 20, 56: 52, 76: 25, 80: 40, 120: 81}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 53, 61, 65, 73, 77, 81, 105, 121, 125], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 30, 'token_total': 707, 'token_per_expert': {1: 131, 5: 154, 53: 30, 61: 24, 65: 38, 73: 99, 77: 24, 81: 28, 105: 70, 121: 53, 125: 56}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 50, 54, 58, 70, 102, 110, 114], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 1085, 'token_per_expert': {2: 170, 6: 149, 38: 52, 46: 59, 50: 76, 54: 152, 58: 157, 70: 61, 102: 47, 110: 99, 114: 63}}
INFO 05-06 11:55:55.251302.251302 lmp.py:1836] [layer_moe_fused] layer=8 prefix: 0.438ms alloc: 0.444ms
INFO 05-06 11:55:55.252651.252651 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.914138793945312e-05 seconds
INFO 05-06 11:55:55.253874.253874 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007829666137695312s
INFO 05-06 11:55:55.253167.253167 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005741119384765625 seconds
DEBUG 05-06 11:55:55.254528.254528 cuda_h.py:27] end moe_cpu_prep_submit cost 0.841 ms
INFO 05-06 11:55:55.268680.268680 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014669656753540039s
DEBUG 05-06 11:55:55.269208.269208 cuda_h.py:27] end moe_wait_copy_tasks cost 14.832 ms
DEBUG 05-06 11:55:55.273754.273754 cuda_h.py:27] end moe_vllm_forward cost 3.995 ms
DEBUG 05-06 11:55:55.273004.273004 cuda_h.py:27] end moe_cpu_merge cost 0.060 ms
DEBUG 05-06 11:55:55.273756.273756 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.274911.274911 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.938ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.274360.274360 cuda_h.py:27] end *layer_moe_fused cost 23.575 ms
DEBUG 05-06 11:55:55.280145.280145 cuda_h.py:27] end prefill_merge_scale cost 5.994 ms
DEBUG 05-06 11:55:55.280897.280897 cuda_h.py:27] end prefill_layer cost 33.878 ms
DEBUG 05-06 11:55:55.280208.280208 lmp.py:1391] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 11:55:55.280017.280017 lmp.py:1347] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 11:55:55.281288.281288 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 11:55:55.283399.283399 cuda_h.py:27] end prefill_attn cost 1.878 ms
DEBUG 05-06 11:55:55.283568.283568 cuda_h.py:27] end prefill_ffn_prep cost 0.392 ms
DEBUG 05-06 11:55:55.284808.284808 cuda_h.py:27] end prefill_gate cost 0.438 ms
experts_cpu_alloc {'expert_ids': [11, 31, 63, 123, 79, 115, 119, 67, 19, 27, 15, 83, 39, 71, 28, 84, 100, 112, 64, 96, 44, 52, 104, 20, 116, 8, 120, 124, 80, 68, 24, 40, 88, 121, 33, 41, 29, 105, 77, 113, 117, 73, 37, 97, 17, 9, 45, 61, 18, 50, 94, 14, 10, 58, 90, 66, 26, 34, 114, 98, 122, 82, 42, 62, 86], 'token_total': 587, 'token_per_expert': {11: 1, 31: 1, 63: 1, 123: 1, 79: 3, 115: 4, 119: 5, 67: 6, 19: 12, 27: 20, 15: 21, 83: 25, 39: 27, 71: 27, 28: 1, 84: 1, 100: 1, 112: 1, 64: 2, 96: 2, 44: 4, 52: 4, 104: 4, 20: 5, 116: 6, 8: 7, 120: 10, 124: 12, 80: 14, 68: 17, 24: 18, 40: 18, 88: 25, 121: 1, 33: 2, 41: 2, 29: 3, 105: 6, 77: 7, 113: 8, 117: 9, 73: 11, 37: 14, 97: 15, 17: 18, 9: 21, 45: 23, 61: 27, 18: 1, 50: 1, 94: 1, 14: 2, 10: 3, 58: 3, 90: 3, 66: 4, 26: 5, 34: 5, 114: 5, 98: 7, 122: 11, 82: 13, 42: 14, 62: 16, 86: 20}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 43, 51, 75, 95, 99, 103, 111, 127], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 949, 'token_per_expert': {3: 143, 7: 138, 23: 45, 43: 94, 51: 28, 75: 94, 95: 205, 99: 30, 103: 106, 111: 31, 127: 35}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 32, 36, 48, 56, 72, 76, 92], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 30, 'token_total': 805, 'token_per_expert': {0: 134, 4: 165, 12: 123, 16: 102, 32: 29, 36: 40, 48: 48, 56: 69, 72: 26, 76: 27, 92: 42}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 57, 69, 81, 89, 93, 101, 125], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 795, 'token_per_expert': {1: 158, 5: 136, 13: 33, 21: 32, 57: 41, 69: 58, 81: 68, 89: 30, 93: 94, 101: 103, 125: 42}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 30, 38, 46, 54, 70, 74, 102, 106], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 960, 'token_per_expert': {2: 129, 6: 133, 22: 34, 30: 22, 38: 31, 46: 156, 54: 29, 70: 132, 74: 85, 102: 46, 106: 163}}
INFO 05-06 11:55:55.285322.285322 lmp.py:1836] [layer_moe_fused] layer=9 prefix: 0.444ms alloc: 0.409ms
INFO 05-06 11:55:55.286718.286718 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.437301635742188e-05 seconds
INFO 05-06 11:55:55.287208.287208 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007929801940917969s
INFO 05-06 11:55:55.287992.287992 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005846023559570312 seconds
DEBUG 05-06 11:55:55.288342.288342 cuda_h.py:27] end moe_cpu_prep_submit cost 1.129 ms
INFO 05-06 11:55:55.302691.302691 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01312398910522461s
DEBUG 05-06 11:55:55.302311.302311 cuda_h.py:27] end moe_wait_copy_tasks cost 13.286 ms
DEBUG 05-06 11:55:55.306563.306563 cuda_h.py:27] end moe_vllm_forward cost 3.913 ms
DEBUG 05-06 11:55:55.306098.306098 cuda_h.py:27] end moe_cpu_merge cost 0.062 ms
DEBUG 05-06 11:55:55.306930.306930 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.306509.306509 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.671ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.307619.307619 cuda_h.py:27] end *layer_moe_fused cost 22.390 ms
DEBUG 05-06 11:55:55.312197.312197 cuda_h.py:27] end prefill_merge_scale cost 4.544 ms
DEBUG 05-06 11:55:55.312572.312572 cuda_h.py:27] end prefill_layer cost 31.200 ms
DEBUG 05-06 11:55:55.312686.312686 lmp.py:1391] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 11:55:55.312734.312734 lmp.py:1347] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 11:55:55.312005.312005 cuda_h.py:27] end prefill_ln cost 0.205 ms
DEBUG 05-06 11:55:55.314966.314966 cuda_h.py:27] end prefill_attn cost 1.940 ms
DEBUG 05-06 11:55:55.315313.315313 cuda_h.py:27] end prefill_ffn_prep cost 0.381 ms
DEBUG 05-06 11:55:55.316038.316038 cuda_h.py:27] end prefill_gate cost 0.462 ms
experts_cpu_alloc {'expert_ids': [23, 51, 123, 35, 15, 27, 59, 107, 67, 119, 111, 11, 103, 19, 83, 79, 12, 32, 36, 48, 116, 40, 52, 56, 120, 124, 64, 28, 44, 112, 84, 100, 68, 65, 77, 101, 109, 25, 33, 53, 29, 9, 61, 97, 73, 37, 117, 89, 93, 69, 105, 121, 38, 66, 110, 70, 26, 102, 98, 34, 78, 50, 94, 90, 46, 126, 54, 10], 'token_total': 595, 'token_per_expert': {23: 1, 51: 1, 123: 1, 35: 2, 15: 3, 27: 3, 59: 3, 107: 3, 67: 4, 119: 4, 111: 5, 11: 6, 103: 6, 19: 11, 83: 11, 79: 16, 12: 1, 32: 1, 36: 1, 48: 1, 116: 1, 40: 2, 52: 3, 56: 4, 120: 4, 124: 6, 64: 8, 28: 11, 44: 12, 112: 12, 84: 22, 100: 22, 68: 29, 65: 1, 77: 1, 101: 1, 109: 3, 25: 4, 33: 4, 53: 4, 29: 5, 9: 6, 61: 8, 97: 8, 73: 9, 37: 11, 117: 12, 89: 13, 93: 15, 69: 16, 105: 20, 121: 22, 38: 1, 66: 1, 110: 1, 70: 2, 26: 4, 102: 4, 98: 6, 34: 7, 78: 9, 50: 10, 94: 18, 90: 19, 46: 31, 126: 32, 54: 33, 10: 34}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 43, 47, 63, 71, 75, 99, 115, 127], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 28, 'token_total': 656, 'token_per_expert': {3: 132, 7: 157, 31: 30, 39: 29, 43: 24, 47: 33, 63: 30, 71: 51, 75: 41, 99: 23, 115: 68, 127: 38}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 60, 72, 76, 80, 88, 92, 108], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 29, 'token_total': 1147, 'token_per_expert': {0: 210, 4: 139, 8: 136, 16: 48, 20: 34, 60: 107, 72: 34, 76: 155, 80: 129, 88: 63, 92: 38, 108: 54}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 41, 49, 57, 81, 85, 113, 125], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 30, 'token_total': 866, 'token_per_expert': {1: 238, 5: 143, 13: 34, 21: 50, 41: 48, 49: 26, 57: 57, 81: 126, 85: 47, 113: 35, 125: 62}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 42, 58, 62, 74, 82, 86, 106], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 832, 'token_per_expert': {2: 128, 6: 134, 14: 86, 18: 40, 42: 60, 58: 48, 62: 66, 74: 80, 82: 42, 86: 82, 106: 66}}
INFO 05-06 11:55:55.317971.317971 lmp.py:1836] [layer_moe_fused] layer=10 prefix: 0.438ms alloc: 0.440ms
INFO 05-06 11:55:55.317843.317843 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.532669067382812e-05 seconds
INFO 05-06 11:55:55.318468.318468 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007729530334472656s
INFO 05-06 11:55:55.319927.319927 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005908012390136719 seconds
DEBUG 05-06 11:55:55.319762.319762 cuda_h.py:27] end moe_cpu_prep_submit cost 0.924 ms
INFO 05-06 11:55:55.334384.334384 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014077186584472656s
DEBUG 05-06 11:55:55.334819.334819 cuda_h.py:27] end moe_wait_copy_tasks cost 14.252 ms
DEBUG 05-06 11:55:55.339372.339372 cuda_h.py:27] end moe_vllm_forward cost 4.010 ms
DEBUG 05-06 11:55:55.339529.339529 cuda_h.py:27] end moe_cpu_merge cost 0.062 ms
DEBUG 05-06 11:55:55.339951.339951 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.339398.339398 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.777ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.340514.340514 cuda_h.py:27] end *layer_moe_fused cost 23.438 ms
DEBUG 05-06 11:55:55.345910.345910 cuda_h.py:27] end prefill_merge_scale cost 5.428 ms
DEBUG 05-06 11:55:55.345039.345039 cuda_h.py:27] end prefill_layer cost 33.229 ms
DEBUG 05-06 11:55:55.345045.345045 lmp.py:1391] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 11:55:55.345000.345000 lmp.py:1347] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 11:55:55.346588.346588 cuda_h.py:27] end prefill_ln cost 0.212 ms
DEBUG 05-06 11:55:55.348402.348402 cuda_h.py:27] end prefill_attn cost 1.900 ms
DEBUG 05-06 11:55:55.348312.348312 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 11:55:55.349539.349539 cuda_h.py:27] end prefill_gate cost 0.427 ms
experts_cpu_alloc {'expert_ids': [35, 127, 115, 47, 63, 59, 39, 51, 123, 11, 71, 27, 43, 91, 19, 84, 104, 12, 52, 64, 72, 8, 80, 28, 40, 48, 36, 44, 116, 120, 124, 20, 100, 53, 65, 21, 85, 9, 13, 33, 97, 125, 117, 121, 25, 29, 89, 61, 22, 114, 74, 106, 110, 126, 58, 94, 34, 122, 118, 42, 62, 98, 50, 82], 'token_total': 545, 'token_per_expert': {35: 1, 127: 1, 115: 2, 47: 3, 63: 3, 59: 7, 39: 9, 51: 9, 123: 9, 11: 12, 71: 12, 27: 15, 43: 16, 91: 16, 19: 26, 84: 1, 104: 1, 12: 2, 52: 3, 64: 4, 72: 4, 8: 6, 80: 6, 28: 7, 40: 7, 48: 10, 36: 14, 44: 15, 116: 15, 120: 17, 124: 18, 20: 35, 100: 37, 53: 1, 65: 1, 21: 2, 85: 2, 9: 3, 13: 3, 33: 3, 97: 3, 125: 3, 117: 13, 121: 14, 25: 17, 29: 18, 89: 21, 61: 23, 22: 1, 114: 1, 74: 2, 106: 2, 110: 2, 126: 2, 58: 3, 94: 3, 34: 4, 122: 4, 118: 5, 42: 7, 62: 7, 98: 9, 50: 11, 82: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 31, 67, 79, 83, 87, 99, 111, 119], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 978, 'token_per_expert': {3: 131, 7: 197, 23: 79, 31: 32, 67: 67, 79: 105, 83: 118, 87: 108, 99: 31, 111: 73, 119: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 24, 32, 56, 68, 76, 92, 108, 112], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 956, 'token_per_expert': {0: 131, 4: 131, 16: 131, 24: 41, 32: 56, 56: 148, 68: 61, 76: 50, 92: 112, 108: 45, 112: 50}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 37, 49, 57, 69, 77, 81, 93, 113], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 886, 'token_per_expert': {1: 140, 5: 139, 17: 76, 37: 35, 49: 81, 57: 33, 69: 27, 77: 34, 81: 110, 93: 107, 113: 104}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 30, 38, 46, 54, 66, 70, 102], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 731, 'token_per_expert': {2: 172, 6: 211, 10: 41, 18: 15, 30: 39, 38: 23, 46: 17, 54: 17, 66: 26, 70: 15, 102: 155}}
INFO 05-06 11:55:55.350265.350265 lmp.py:1836] [layer_moe_fused] layer=11 prefix: 0.443ms alloc: 0.424ms
INFO 05-06 11:55:55.351138.351138 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.866455078125e-05 seconds
INFO 05-06 11:55:55.352841.352841 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007619857788085938s
INFO 05-06 11:55:55.352154.352154 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005877017974853516 seconds
DEBUG 05-06 11:55:55.353420.353420 cuda_h.py:27] end moe_cpu_prep_submit cost 1.045 ms
INFO 05-06 11:55:55.364053.364053 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.010925054550170898s
DEBUG 05-06 11:55:55.364143.364143 cuda_h.py:27] end moe_wait_copy_tasks cost 11.083 ms
DEBUG 05-06 11:55:55.369242.369242 cuda_h.py:27] end moe_vllm_forward cost 3.885 ms
DEBUG 05-06 11:55:55.369591.369591 cuda_h.py:27] end moe_cpu_merge cost 0.063 ms
DEBUG 05-06 11:55:55.369323.369323 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.369617.369617 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.629ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.370766.370766 cuda_h.py:27] end *layer_moe_fused cost 19.990 ms
DEBUG 05-06 11:55:55.376829.376829 cuda_h.py:27] end prefill_merge_scale cost 6.548 ms
DEBUG 05-06 11:55:55.376958.376958 cuda_h.py:27] end prefill_layer cost 30.796 ms
DEBUG 05-06 11:55:55.377882.377882 lmp.py:1391] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 11:55:55.377644.377644 lmp.py:1347] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 11:55:55.377272.377272 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 11:55:55.379260.379260 cuda_h.py:27] end prefill_attn cost 1.924 ms
DEBUG 05-06 11:55:55.379673.379673 cuda_h.py:27] end prefill_ffn_prep cost 0.397 ms
DEBUG 05-06 11:55:55.380106.380106 cuda_h.py:27] end prefill_gate cost 0.441 ms
experts_cpu_alloc {'expert_ids': [59, 67, 107, 31, 111, 47, 63, 119, 123, 127, 79, 103, 95, 8, 56, 64, 120, 20, 32, 24, 104, 12, 112, 100, 40, 76, 80, 37, 41, 65, 109, 81, 17, 113, 125, 33, 105, 13, 89, 77, 117, 101, 85, 18, 94, 70, 38, 98, 58, 90, 102, 22, 34, 106, 118, 46], 'token_total': 584, 'token_per_expert': {59: 1, 67: 3, 107: 3, 31: 4, 111: 4, 47: 5, 63: 5, 119: 5, 123: 6, 127: 6, 79: 10, 103: 12, 95: 23, 8: 1, 56: 1, 64: 1, 120: 3, 20: 4, 32: 4, 24: 6, 104: 10, 12: 11, 112: 12, 100: 14, 40: 15, 76: 15, 80: 15, 37: 1, 41: 1, 65: 1, 109: 1, 81: 2, 17: 3, 113: 3, 125: 3, 33: 5, 105: 5, 13: 6, 89: 7, 77: 8, 117: 24, 101: 25, 85: 26, 18: 2, 94: 6, 70: 7, 38: 8, 98: 8, 58: 9, 90: 9, 102: 9, 22: 15, 34: 18, 106: 54, 118: 59, 46: 60}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 35, 39, 71, 91, 115], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 23, 'token_total': 829, 'token_per_expert': {3: 155, 7: 128, 15: 95, 19: 38, 23: 40, 35: 30, 39: 133, 71: 133, 91: 43, 115: 34}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 68, 84, 88, 92, 108, 116, 124], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 24, 'token_total': 588, 'token_per_expert': {0: 128, 4: 129, 36: 19, 68: 18, 84: 20, 88: 15, 92: 24, 108: 119, 116: 99, 124: 17}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25, 45, 49, 53, 73, 97], 'expert_count': 9, 'ideal_gpu_count': 9, 'keep_on_gpu': 9, 'hit_count_on_device': 25, 'token_total': 958, 'token_per_expert': {1: 141, 5: 181, 21: 156, 25: 53, 45: 107, 49: 52, 53: 159, 73: 44, 97: 65}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 50, 74, 78, 82, 86, 110, 114], 'expert_count': 9, 'ideal_gpu_count': 9, 'keep_on_gpu': 9, 'hit_count_on_device': 22, 'token_total': 1137, 'token_per_expert': {2: 128, 6: 177, 50: 109, 74: 108, 78: 218, 82: 85, 86: 110, 110: 104, 114: 98}}
INFO 05-06 11:55:55.382526.382526 lmp.py:1836] [layer_moe_fused] layer=12 prefix: 0.434ms alloc: 0.382ms
INFO 05-06 11:55:55.382763.382763 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.459785461425781e-05 seconds
INFO 05-06 11:55:55.383799.383799 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007662773132324219s
INFO 05-06 11:55:55.383556.383556 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006000995635986328 seconds
DEBUG 05-06 11:55:55.384657.384657 cuda_h.py:27] end moe_cpu_prep_submit cost 0.847 ms
INFO 05-06 11:55:55.397231.397231 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013082265853881836s
DEBUG 05-06 11:55:55.397824.397824 cuda_h.py:27] end moe_wait_copy_tasks cost 13.220 ms
DEBUG 05-06 11:55:55.401542.401542 cuda_h.py:27] end moe_vllm_forward cost 3.832 ms
DEBUG 05-06 11:55:55.401978.401978 cuda_h.py:27] end moe_cpu_merge cost 0.062 ms
DEBUG 05-06 11:55:55.402899.402899 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.402432.402432 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.658ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.402574.402574 cuda_h.py:27] end *layer_moe_fused cost 21.432 ms
DEBUG 05-06 11:55:55.407662.407662 cuda_h.py:27] end prefill_merge_scale cost 4.360 ms
DEBUG 05-06 11:55:55.407930.407930 cuda_h.py:27] end prefill_layer cost 30.090 ms
DEBUG 05-06 11:55:55.407384.407384 lmp.py:1391] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 11:55:55.407670.407670 lmp.py:1347] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 11:55:55.407571.407571 cuda_h.py:27] end prefill_ln cost 0.211 ms
DEBUG 05-06 11:55:55.409153.409153 cuda_h.py:27] end prefill_attn cost 1.871 ms
DEBUG 05-06 11:55:55.410340.410340 cuda_h.py:27] end prefill_ffn_prep cost 0.373 ms
DEBUG 05-06 11:55:55.411342.411342 cuda_h.py:27] end prefill_gate cost 0.441 ms
experts_cpu_alloc {'expert_ids': [19, 83, 95, 107, 47, 87, 27, 43, 123, 55, 11, 67, 99, 75, 115, 119, 15, 12, 36, 48, 56, 112, 80, 104, 96, 8, 16, 64, 28, 52, 68, 92, 108, 77, 97, 53, 61, 105, 45, 57, 9, 65, 73, 101, 93, 13, 41, 117, 69, 66, 10, 62, 74, 94, 70, 106, 122, 26, 90, 82, 42, 46, 38, 34, 86], 'token_total': 596, 'token_per_expert': {19: 1, 83: 3, 95: 3, 107: 3, 47: 4, 87: 4, 27: 5, 43: 5, 123: 7, 55: 9, 11: 11, 67: 11, 99: 16, 75: 17, 115: 23, 119: 26, 15: 28, 12: 1, 36: 1, 48: 1, 56: 1, 112: 1, 80: 2, 104: 5, 96: 6, 8: 9, 16: 9, 64: 9, 28: 10, 52: 11, 68: 11, 92: 11, 108: 13, 77: 1, 97: 1, 53: 2, 61: 2, 105: 2, 45: 3, 57: 4, 9: 5, 65: 6, 73: 9, 101: 9, 93: 12, 13: 14, 41: 20, 117: 24, 69: 29, 66: 1, 10: 2, 62: 2, 74: 2, 94: 3, 70: 4, 106: 4, 122: 7, 26: 8, 90: 8, 82: 12, 42: 14, 46: 16, 38: 29, 34: 31, 86: 33}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 51, 59, 63, 71, 79, 91, 103], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 968, 'token_per_expert': {3: 145, 7: 128, 31: 176, 39: 44, 51: 32, 59: 54, 63: 48, 71: 61, 79: 112, 91: 137, 103: 31}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 32, 40, 60, 84, 100, 116, 120, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 711, 'token_per_expert': {0: 128, 4: 128, 20: 41, 32: 74, 40: 18, 60: 21, 84: 32, 100: 135, 116: 21, 120: 84, 124: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 25, 33, 37, 81, 113, 121, 125], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 846, 'token_per_expert': {1: 176, 5: 128, 17: 69, 21: 32, 25: 57, 33: 45, 37: 86, 81: 76, 113: 34, 121: 110, 125: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 78, 98, 102, 110, 114, 118, 126], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 975, 'token_per_expert': {2: 140, 6: 185, 14: 79, 22: 38, 78: 69, 98: 44, 102: 48, 110: 132, 114: 143, 118: 43, 126: 54}}
INFO 05-06 11:55:55.412268.412268 lmp.py:1836] [layer_moe_fused] layer=13 prefix: 0.434ms alloc: 0.429ms
INFO 05-06 11:55:55.412902.412902 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.747245788574219e-05 seconds
INFO 05-06 11:55:55.413962.413962 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007946491241455078s
INFO 05-06 11:55:55.414553.414553 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005834102630615234 seconds
DEBUG 05-06 11:55:55.414090.414090 cuda_h.py:27] end moe_cpu_prep_submit cost 1.124 ms
INFO 05-06 11:55:55.433406.433406 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.018349409103393555s
DEBUG 05-06 11:55:55.433550.433550 cuda_h.py:27] end moe_wait_copy_tasks cost 18.515 ms
DEBUG 05-06 11:55:55.438396.438396 cuda_h.py:27] end moe_vllm_forward cost 3.878 ms
DEBUG 05-06 11:55:55.438077.438077 cuda_h.py:27] end moe_cpu_merge cost 0.062 ms
DEBUG 05-06 11:55:55.438810.438810 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.438489.438489 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.656ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.439872.439872 cuda_h.py:27] end *layer_moe_fused cost 27.401 ms
DEBUG 05-06 11:55:55.445198.445198 cuda_h.py:27] end prefill_merge_scale cost 6.113 ms
DEBUG 05-06 11:55:55.445519.445519 cuda_h.py:27] end prefill_layer cost 37.807 ms
DEBUG 05-06 11:55:55.445726.445726 lmp.py:1391] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 11:55:55.445727.445727 lmp.py:1347] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 11:55:55.446656.446656 cuda_h.py:27] end prefill_ln cost 0.208 ms
DEBUG 05-06 11:55:55.448490.448490 cuda_h.py:27] end prefill_attn cost 1.921 ms
DEBUG 05-06 11:55:55.448393.448393 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 11:55:55.449886.449886 cuda_h.py:27] end prefill_gate cost 0.456 ms
experts_cpu_alloc {'expert_ids': [27, 63, 91, 111, 51, 15, 19, 71, 23, 67, 43, 35, 107, 83, 11, 127, 123, 56, 36, 68, 96, 44, 64, 116, 48, 120, 40, 60, 28, 92, 52, 112, 108, 24, 33, 37, 61, 85, 29, 77, 101, 17, 41, 109, 9, 73, 81, 21, 125, 25, 45, 93, 14, 46, 54, 70, 106, 126, 118, 102, 78, 10, 58, 98, 90, 114, 110, 34, 74], 'token_total': 533, 'token_per_expert': {27: 1, 63: 1, 91: 1, 111: 1, 51: 2, 15: 3, 19: 3, 71: 4, 23: 5, 67: 5, 43: 7, 35: 8, 107: 8, 83: 16, 11: 17, 127: 24, 123: 36, 56: 2, 36: 4, 68: 4, 96: 4, 44: 6, 64: 6, 116: 6, 48: 8, 120: 10, 40: 11, 60: 11, 28: 12, 92: 12, 52: 13, 112: 13, 108: 14, 24: 16, 33: 1, 37: 1, 61: 1, 85: 1, 29: 2, 77: 2, 101: 2, 17: 3, 41: 4, 109: 4, 9: 5, 73: 5, 81: 7, 21: 9, 125: 11, 25: 12, 45: 12, 93: 12, 14: 1, 46: 1, 54: 1, 70: 2, 106: 4, 126: 4, 118: 5, 102: 6, 78: 7, 10: 9, 58: 9, 98: 10, 90: 11, 114: 15, 110: 16, 34: 21, 74: 23}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 59, 75, 95, 99, 103, 115, 119], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 29, 'token_total': 1081, 'token_per_expert': {3: 197, 7: 209, 31: 42, 39: 71, 47: 67, 59: 45, 75: 67, 95: 46, 99: 44, 103: 50, 115: 166, 119: 77}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 32, 72, 76, 80, 100, 104, 124], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 29, 'token_total': 741, 'token_per_expert': {0: 197, 4: 193, 8: 17, 12: 27, 16: 18, 32: 21, 72: 19, 76: 28, 80: 37, 100: 57, 104: 26, 124: 101}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 53, 57, 65, 89, 97, 105, 113, 117, 121], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 30, 'token_total': 852, 'token_per_expert': {1: 194, 5: 199, 13: 24, 53: 26, 57: 20, 65: 73, 89: 25, 97: 66, 105: 17, 113: 40, 117: 55, 121: 113}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 30, 38, 42, 50, 62, 66, 86, 122], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 889, 'token_per_expert': {2: 233, 6: 193, 26: 80, 30: 29, 38: 26, 42: 30, 50: 65, 62: 41, 66: 60, 86: 94, 122: 38}}
INFO 05-06 11:55:55.450096.450096 lmp.py:1836] [layer_moe_fused] layer=14 prefix: 0.433ms alloc: 0.446ms
INFO 05-06 11:55:55.450161.450161 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.961822509765625e-05 seconds
INFO 05-06 11:55:55.451660.451660 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007822513580322266s
INFO 05-06 11:55:55.452582.452582 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005819797515869141 seconds
DEBUG 05-06 11:55:55.452712.452712 cuda_h.py:27] end moe_cpu_prep_submit cost 0.858 ms
INFO 05-06 11:55:55.465052.465052 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012978553771972656s
DEBUG 05-06 11:55:55.466673.466673 cuda_h.py:27] end moe_wait_copy_tasks cost 13.141 ms
DEBUG 05-06 11:55:55.470309.470309 cuda_h.py:27] end moe_vllm_forward cost 3.929 ms
DEBUG 05-06 11:55:55.470851.470851 cuda_h.py:27] end moe_cpu_merge cost 0.064 ms
DEBUG 05-06 11:55:55.470630.470630 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.470877.470877 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.676ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.471259.471259 cuda_h.py:27] end *layer_moe_fused cost 21.578 ms
DEBUG 05-06 11:55:55.477223.477223 cuda_h.py:27] end prefill_merge_scale cost 6.159 ms
DEBUG 05-06 11:55:55.477974.477974 cuda_h.py:27] end prefill_layer cost 32.097 ms
DEBUG 05-06 11:55:55.477149.477149 lmp.py:1391] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 11:55:55.477627.477627 lmp.py:1347] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 11:55:55.478824.478824 cuda_h.py:27] end prefill_ln cost 0.207 ms
DEBUG 05-06 11:55:55.480652.480652 cuda_h.py:27] end prefill_attn cost 1.914 ms
DEBUG 05-06 11:55:55.480508.480508 cuda_h.py:27] end prefill_ffn_prep cost 0.374 ms
DEBUG 05-06 11:55:55.481094.481094 cuda_h.py:27] end prefill_gate cost 0.438 ms
experts_cpu_alloc {'expert_ids': [67, 87, 35, 79, 11, 111, 19, 127, 55, 43, 115, 107, 119, 31, 63, 103, 47, 8, 32, 60, 100, 40, 56, 80, 28, 96, 48, 36, 116, 120, 16, 24, 124, 84, 89, 105, 45, 57, 25, 117, 77, 29, 17, 33, 13, 41, 69, 97, 121, 113, 73, 125, 22, 54, 126, 94, 18, 118, 34, 38, 82, 46, 86, 102, 78, 58], 'token_total': 611, 'token_per_expert': {67: 1, 87: 1, 35: 2, 79: 2, 11: 3, 111: 4, 19: 6, 127: 6, 55: 8, 43: 10, 115: 11, 107: 12, 119: 12, 31: 17, 63: 19, 103: 19, 47: 20, 8: 1, 32: 1, 60: 1, 100: 1, 40: 4, 56: 4, 80: 6, 28: 8, 96: 8, 48: 10, 36: 12, 116: 18, 120: 21, 16: 23, 24: 24, 124: 24, 84: 25, 89: 1, 105: 1, 45: 2, 57: 2, 25: 4, 117: 5, 77: 7, 29: 8, 17: 9, 33: 9, 13: 11, 41: 11, 69: 13, 97: 13, 121: 13, 113: 17, 73: 18, 125: 23, 22: 2, 54: 2, 126: 2, 94: 3, 18: 7, 118: 7, 34: 8, 38: 8, 82: 8, 46: 10, 86: 10, 102: 10, 78: 11, 58: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 51, 59, 71, 75, 83, 91, 95, 99], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 29, 'token_total': 908, 'token_per_expert': {3: 193, 7: 218, 23: 45, 39: 36, 51: 38, 59: 20, 71: 50, 75: 52, 83: 86, 91: 115, 95: 20, 99: 35}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 52, 64, 68, 72, 76, 88, 104, 108, 112], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 973, 'token_per_expert': {0: 201, 4: 197, 52: 46, 64: 34, 68: 104, 72: 27, 76: 126, 88: 27, 104: 31, 108: 60, 112: 120}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 37, 65, 81, 85, 93, 101, 109], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 840, 'token_per_expert': {1: 207, 5: 217, 9: 42, 21: 26, 37: 33, 65: 93, 81: 31, 85: 24, 93: 29, 101: 41, 109: 97}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 30, 42, 66, 70, 90, 98, 114], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 764, 'token_per_expert': {2: 222, 6: 193, 10: 59, 14: 17, 30: 51, 42: 16, 66: 42, 70: 33, 90: 82, 98: 33, 114: 16}}
INFO 05-06 11:55:55.482403.482403 lmp.py:1836] [layer_moe_fused] layer=15 prefix: 0.443ms alloc: 0.434ms
INFO 05-06 11:55:55.483799.483799 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 6.747245788574219e-05 seconds
INFO 05-06 11:55:55.484439.484439 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008692741394042969s
INFO 05-06 11:55:55.484831.484831 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005755424499511719 seconds
DEBUG 05-06 11:55:55.485493.485493 cuda_h.py:27] end moe_cpu_prep_submit cost 1.215 ms
INFO 05-06 11:55:55.496689.496689 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.010423421859741211s
DEBUG 05-06 11:55:55.496272.496272 cuda_h.py:27] end moe_wait_copy_tasks cost 10.633 ms
DEBUG 05-06 11:55:55.501213.501213 cuda_h.py:27] end moe_vllm_forward cost 4.310 ms
DEBUG 05-06 11:55:55.501967.501967 cuda_h.py:27] end moe_cpu_merge cost 0.076 ms
DEBUG 05-06 11:55:55.501358.501358 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.502472.502472 lmp.py:1950] [layer_moe_fused] vllm triton time: 5.335ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.502179.502179 cuda_h.py:27] end *layer_moe_fused cost 20.694 ms
DEBUG 05-06 11:55:55.506135.506135 cuda_h.py:27] end prefill_merge_scale cost 3.939 ms
DEBUG 05-06 11:55:55.506529.506529 cuda_h.py:27] end prefill_layer cost 28.928 ms
DEBUG 05-06 11:55:55.507816.507816 lmp.py:1391] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 11:55:55.507916.507916 lmp.py:1347] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 11:55:55.507964.507964 cuda_h.py:27] end prefill_ln cost 0.277 ms
DEBUG 05-06 11:55:55.509512.509512 cuda_h.py:27] end prefill_attn cost 2.264 ms
DEBUG 05-06 11:55:55.510485.510485 cuda_h.py:27] end prefill_ffn_prep cost 0.489 ms
DEBUG 05-06 11:55:55.511867.511867 cuda_h.py:27] end prefill_gate cost 0.584 ms
experts_cpu_alloc {'expert_ids': [11, 27, 51, 59, 103, 71, 43, 123, 91, 15, 99, 111, 119, 79, 127, 36, 60, 84, 28, 88, 24, 64, 104, 120, 56, 92, 40, 80, 116, 72, 48, 108, 20, 96, 29, 41, 101, 25, 53, 69, 49, 89, 33, 13, 37, 109, 9, 61, 81, 45, 93, 21, 57, 46, 50, 94, 106, 122, 98, 38, 10, 34, 118, 18, 62, 82, 102, 90, 22, 30, 110, 70], 'token_total': 558, 'token_per_expert': {11: 1, 27: 1, 51: 3, 59: 3, 103: 3, 71: 4, 43: 5, 123: 6, 91: 7, 15: 8, 99: 8, 111: 8, 119: 10, 79: 13, 127: 19, 36: 1, 60: 1, 84: 1, 28: 2, 88: 2, 24: 3, 64: 5, 104: 6, 120: 6, 56: 7, 92: 10, 40: 11, 80: 13, 116: 15, 72: 20, 48: 23, 108: 23, 20: 24, 96: 26, 29: 1, 41: 1, 101: 1, 25: 2, 53: 2, 69: 2, 49: 3, 89: 3, 33: 5, 13: 7, 37: 7, 109: 7, 9: 8, 61: 9, 81: 9, 45: 10, 93: 11, 21: 13, 57: 13, 46: 1, 50: 1, 94: 1, 106: 1, 122: 1, 98: 3, 38: 4, 10: 5, 34: 5, 118: 6, 18: 7, 62: 7, 82: 7, 102: 11, 90: 12, 22: 13, 30: 15, 110: 22, 70: 24}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 31, 55, 63, 67, 75, 83, 87, 107], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 27, 'token_total': 867, 'token_per_expert': {3: 210, 7: 197, 19: 21, 23: 35, 31: 39, 55: 28, 63: 34, 67: 85, 75: 24, 83: 41, 87: 107, 107: 46}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 32, 44, 52, 68, 76, 100, 124], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 31, 'token_total': 1061, 'token_per_expert': {0: 224, 4: 224, 8: 48, 12: 47, 16: 111, 32: 124, 44: 38, 52: 123, 68: 34, 76: 30, 100: 30, 124: 28}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 65, 77, 85, 97, 105, 113, 117, 121, 125], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 31, 'token_total': 730, 'token_per_expert': {1: 254, 5: 219, 17: 15, 65: 19, 77: 18, 85: 29, 97: 18, 105: 68, 113: 13, 117: 29, 121: 20, 125: 28}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 26, 42, 54, 58, 66, 78, 86, 114, 126], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 31, 'token_total': 880, 'token_per_expert': {2: 213, 6: 197, 14: 34, 26: 25, 42: 25, 54: 28, 58: 25, 66: 47, 78: 25, 86: 75, 114: 29, 126: 157}}
INFO 05-06 11:55:55.513765.513765 lmp.py:1836] [layer_moe_fused] layer=16 prefix: 0.513ms alloc: 0.457ms
INFO 05-06 11:55:55.513061.513061 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.555152893066406e-05 seconds
INFO 05-06 11:55:55.514340.514340 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008161067962646484s
INFO 05-06 11:55:55.515840.515840 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006170272827148438 seconds
DEBUG 05-06 11:55:55.515484.515484 cuda_h.py:27] end moe_cpu_prep_submit cost 1.038 ms
INFO 05-06 11:55:55.532608.532608 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01643204689025879s
DEBUG 05-06 11:55:55.532123.532123 cuda_h.py:27] end moe_wait_copy_tasks cost 16.584 ms
DEBUG 05-06 11:55:55.536255.536255 cuda_h.py:27] end moe_vllm_forward cost 4.148 ms
DEBUG 05-06 11:55:55.537519.537519 cuda_h.py:27] end moe_cpu_merge cost 0.071 ms
DEBUG 05-06 11:55:55.537379.537379 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.537356.537356 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.941ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.538327.538327 cuda_h.py:27] end *layer_moe_fused cost 25.860 ms
DEBUG 05-06 11:55:55.541298.541298 cuda_h.py:27] end prefill_merge_scale cost 3.638 ms
DEBUG 05-06 11:55:55.541573.541573 cuda_h.py:27] end prefill_layer cost 34.647 ms
DEBUG 05-06 11:55:55.542780.542780 lmp.py:1391] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 11:55:55.542927.542927 lmp.py:1347] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 11:55:55.542148.542148 cuda_h.py:27] end prefill_ln cost 0.255 ms
DEBUG 05-06 11:55:55.544121.544121 cuda_h.py:27] end prefill_attn cost 2.298 ms
DEBUG 05-06 11:55:55.545796.545796 cuda_h.py:27] end prefill_ffn_prep cost 0.483 ms
DEBUG 05-06 11:55:55.546884.546884 cuda_h.py:27] end prefill_gate cost 0.508 ms
experts_cpu_alloc {'expert_ids': [11, 51, 79, 83, 111, 91, 87, 123, 59, 15, 19, 119, 55, 99, 31, 67, 103, 35, 88, 92, 112, 36, 96, 8, 44, 80, 16, 60, 104, 108, 124, 116, 120, 100, 84, 12, 56, 77, 81, 93, 117, 65, 85, 29, 97, 113, 9, 13, 33, 109, 125, 45, 73, 57, 122, 102, 38, 42, 66, 14, 34, 62, 90, 114, 118, 126, 98, 78, 94], 'token_total': 458, 'token_per_expert': {11: 1, 51: 1, 79: 1, 83: 1, 111: 1, 91: 2, 87: 3, 123: 3, 59: 4, 15: 5, 19: 6, 119: 7, 55: 8, 99: 8, 31: 9, 67: 12, 103: 19, 35: 20, 88: 1, 92: 2, 112: 2, 36: 4, 96: 4, 8: 5, 44: 6, 80: 7, 16: 8, 60: 8, 104: 8, 108: 8, 124: 8, 116: 9, 120: 14, 100: 16, 84: 17, 12: 20, 56: 21, 77: 1, 81: 1, 93: 1, 117: 1, 65: 2, 85: 2, 29: 4, 97: 4, 113: 5, 9: 6, 13: 6, 33: 6, 109: 6, 125: 9, 45: 12, 73: 12, 57: 21, 122: 1, 102: 2, 38: 3, 42: 3, 66: 3, 14: 4, 34: 4, 62: 4, 90: 4, 114: 4, 118: 4, 126: 8, 98: 10, 78: 13, 94: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 39, 43, 47, 63, 71, 75, 95, 107], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 30, 'token_total': 967, 'token_per_expert': {3: 268, 7: 256, 23: 72, 27: 53, 39: 53, 43: 25, 47: 23, 63: 23, 71: 25, 75: 59, 95: 73, 107: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 28, 40, 48, 52, 64, 68, 72, 76], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 31, 'token_total': 911, 'token_per_expert': {0: 264, 4: 266, 20: 23, 24: 74, 28: 28, 40: 34, 48: 24, 52: 40, 64: 35, 68: 22, 72: 33, 76: 68}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 37, 49, 53, 61, 69, 89, 101], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 945, 'token_per_expert': {1: 257, 5: 276, 17: 34, 21: 55, 37: 81, 49: 30, 53: 28, 61: 31, 69: 81, 89: 39, 101: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 54, 58, 70, 74, 86, 106], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 815, 'token_per_expert': {2: 260, 6: 270, 10: 23, 18: 31, 22: 15, 54: 21, 58: 29, 70: 15, 74: 61, 86: 68, 106: 22}}
INFO 05-06 11:55:55.548528.548528 lmp.py:1836] [layer_moe_fused] layer=17 prefix: 0.496ms alloc: 0.435ms
INFO 05-06 11:55:55.548586.548586 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.245208740234375e-05 seconds
INFO 05-06 11:55:55.549470.549470 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007681846618652344s
INFO 05-06 11:55:55.549552.549552 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005908012390136719 seconds
DEBUG 05-06 11:55:55.550529.550529 cuda_h.py:27] end moe_cpu_prep_submit cost 1.089 ms
INFO 05-06 11:55:55.559163.559163 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.00867915153503418s
DEBUG 05-06 11:55:55.559856.559856 cuda_h.py:27] end moe_wait_copy_tasks cost 8.823 ms
DEBUG 05-06 11:55:55.564753.564753 cuda_h.py:27] end moe_vllm_forward cost 4.038 ms
DEBUG 05-06 11:55:55.564533.564533 cuda_h.py:27] end moe_cpu_merge cost 0.068 ms
DEBUG 05-06 11:55:55.564478.564478 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.564978.564978 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.775ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.565147.565147 cuda_h.py:27] end *layer_moe_fused cost 18.034 ms
DEBUG 05-06 11:55:55.570098.570098 cuda_h.py:27] end prefill_merge_scale cost 4.992 ms
DEBUG 05-06 11:55:55.570426.570426 cuda_h.py:27] end prefill_layer cost 28.139 ms
DEBUG 05-06 11:55:55.570353.570353 lmp.py:1391] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 11:55:55.570785.570785 lmp.py:1347] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 11:55:55.570474.570474 cuda_h.py:27] end prefill_ln cost 0.256 ms
DEBUG 05-06 11:55:55.573272.573272 cuda_h.py:27] end prefill_attn cost 2.204 ms
DEBUG 05-06 11:55:55.573615.573615 cuda_h.py:27] end prefill_ffn_prep cost 0.483 ms
DEBUG 05-06 11:55:55.575102.575102 cuda_h.py:27] end prefill_gate cost 0.526 ms
experts_cpu_alloc {'expert_ids': [19, 11, 23, 27, 59, 115, 51, 67, 39, 55, 103, 15, 35, 107, 95, 47, 71, 75, 44, 112, 96, 24, 16, 52, 124, 116, 56, 80, 12, 68, 108, 48, 88, 92, 120, 100, 25, 41, 113, 9, 45, 109, 21, 89, 97, 13, 29, 125, 73, 37, 69, 57, 93, 65, 18, 102, 94, 74, 98, 66, 82, 26, 114, 30, 42, 62, 46, 70, 90, 122], 'token_total': 543, 'token_per_expert': {19: 1, 11: 2, 23: 3, 27: 3, 59: 3, 115: 3, 51: 4, 67: 4, 39: 5, 55: 5, 103: 6, 15: 9, 35: 9, 107: 9, 95: 12, 47: 15, 71: 15, 75: 15, 44: 1, 112: 1, 96: 2, 24: 3, 16: 4, 52: 4, 124: 5, 116: 8, 56: 10, 80: 10, 12: 11, 68: 11, 108: 11, 48: 12, 88: 17, 92: 17, 120: 21, 100: 22, 25: 1, 41: 1, 113: 1, 9: 6, 45: 6, 109: 6, 21: 7, 89: 7, 97: 7, 13: 8, 29: 8, 125: 9, 73: 11, 37: 12, 69: 16, 57: 17, 93: 23, 65: 24, 18: 1, 102: 1, 94: 2, 74: 3, 98: 3, 66: 4, 82: 4, 26: 5, 114: 5, 30: 6, 42: 6, 62: 6, 46: 7, 70: 9, 90: 9, 122: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 43, 83, 87, 91, 99, 111, 119, 123, 127], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 30, 'token_total': 894, 'token_per_expert': {3: 298, 7: 264, 31: 32, 43: 41, 83: 43, 87: 22, 91: 15, 99: 66, 111: 50, 119: 23, 123: 17, 127: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 32, 36, 40, 60, 64, 72, 76, 84, 104], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 30, 'token_total': 855, 'token_per_expert': {0: 260, 4: 287, 8: 30, 32: 45, 36: 51, 40: 31, 60: 26, 64: 30, 72: 23, 76: 23, 84: 24, 104: 25}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 33, 49, 53, 61, 77, 81, 85, 101, 121], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 30, 'token_total': 956, 'token_per_expert': {1: 282, 5: 262, 17: 29, 33: 32, 49: 38, 53: 35, 61: 35, 77: 61, 81: 24, 85: 50, 101: 38, 121: 70}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 34, 38, 50, 54, 58, 78, 110, 118], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 28, 'token_total': 848, 'token_per_expert': {2: 300, 6: 256, 10: 15, 14: 36, 34: 16, 38: 19, 50: 42, 54: 57, 58: 36, 78: 22, 110: 22, 118: 27}}
INFO 05-06 11:55:55.576547.576547 lmp.py:1836] [layer_moe_fused] layer=18 prefix: 0.497ms alloc: 0.427ms
INFO 05-06 11:55:55.576174.576174 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.4836273193359375e-05 seconds
INFO 05-06 11:55:55.577417.577417 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000713348388671875s
INFO 05-06 11:55:55.578267.578267 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005946159362792969 seconds
DEBUG 05-06 11:55:55.578998.578998 cuda_h.py:27] end moe_cpu_prep_submit cost 1.145 ms
INFO 05-06 11:55:55.596557.596557 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.017802715301513672s
DEBUG 05-06 11:55:55.596641.596641 cuda_h.py:27] end moe_wait_copy_tasks cost 17.953 ms
DEBUG 05-06 11:55:55.601247.601247 cuda_h.py:27] end moe_vllm_forward cost 4.045 ms
DEBUG 05-06 11:55:55.601603.601603 cuda_h.py:27] end moe_cpu_merge cost 0.069 ms
DEBUG 05-06 11:55:55.601947.601947 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.601531.601531 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.969ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.602822.602822 cuda_h.py:27] end *layer_moe_fused cost 27.324 ms
DEBUG 05-06 11:55:55.607541.607541 cuda_h.py:27] end prefill_merge_scale cost 4.398 ms
DEBUG 05-06 11:55:55.607345.607345 cuda_h.py:27] end prefill_layer cost 36.717 ms
DEBUG 05-06 11:55:55.607073.607073 lmp.py:1391] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 11:55:55.607889.607889 lmp.py:1347] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 11:55:55.607069.607069 cuda_h.py:27] end prefill_ln cost 0.259 ms
DEBUG 05-06 11:55:55.610786.610786 cuda_h.py:27] end prefill_attn cost 2.180 ms
DEBUG 05-06 11:55:55.610845.610845 cuda_h.py:27] end prefill_ffn_prep cost 0.485 ms
DEBUG 05-06 11:55:55.611120.611120 cuda_h.py:27] end prefill_gate cost 0.543 ms
experts_cpu_alloc {'expert_ids': [115, 67, 107, 43, 127, 103, 59, 111, 15, 99, 55, 27, 47, 83, 35, 11, 75, 8, 116, 124, 100, 20, 68, 56, 96, 112, 108, 84, 60, 104, 12, 36, 72, 48, 80, 29, 49, 105, 57, 101, 45, 77, 25, 65, 121, 53, 33, 97, 13, 17, 73, 41, 18, 34, 42, 46, 54, 82, 66, 70, 114, 118, 86, 106, 22, 90], 'token_total': 498, 'token_per_expert': {115: 1, 67: 2, 107: 2, 43: 3, 127: 4, 103: 6, 59: 8, 111: 8, 15: 10, 99: 10, 55: 12, 27: 13, 47: 13, 83: 14, 35: 17, 11: 18, 75: 18, 8: 1, 116: 1, 124: 1, 100: 2, 20: 4, 68: 4, 56: 7, 96: 7, 112: 7, 108: 9, 84: 10, 60: 11, 104: 12, 12: 16, 36: 17, 72: 18, 48: 22, 80: 22, 29: 1, 49: 1, 105: 1, 57: 2, 101: 2, 45: 3, 77: 3, 25: 5, 65: 5, 121: 5, 53: 7, 33: 8, 97: 8, 13: 11, 17: 11, 73: 13, 41: 16, 18: 1, 34: 1, 42: 2, 46: 2, 54: 2, 82: 2, 66: 3, 70: 3, 114: 4, 118: 5, 86: 8, 106: 9, 22: 12, 90: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 31, 39, 51, 63, 79, 119, 123], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 843, 'token_per_expert': {3: 301, 7: 303, 19: 19, 23: 23, 31: 20, 39: 22, 51: 57, 63: 22, 79: 25, 119: 18, 123: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 24, 40, 44, 52, 64, 76, 88, 92], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 971, 'token_per_expert': {0: 263, 4: 258, 16: 23, 24: 35, 40: 37, 44: 73, 52: 107, 64: 64, 76: 27, 88: 34, 92: 50}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 37, 61, 69, 89, 109, 117, 125], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 917, 'token_per_expert': {1: 282, 5: 274, 9: 28, 21: 23, 37: 82, 61: 30, 69: 17, 89: 82, 109: 27, 117: 52, 125: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 26, 38, 50, 58, 98, 102, 122, 126], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 867, 'token_per_expert': {2: 269, 6: 261, 10: 27, 26: 19, 38: 97, 50: 34, 58: 13, 98: 15, 102: 29, 122: 90, 126: 13}}
INFO 05-06 11:55:55.613571.613571 lmp.py:1836] [layer_moe_fused] layer=19 prefix: 0.489ms alloc: 0.404ms
INFO 05-06 11:55:55.613099.613099 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.245208740234375e-05 seconds
INFO 05-06 11:55:55.614693.614693 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000751495361328125s
INFO 05-06 11:55:55.614622.614622 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005838871002197266 seconds
DEBUG 05-06 11:55:55.615174.615174 cuda_h.py:27] end moe_cpu_prep_submit cost 1.071 ms
INFO 05-06 11:55:55.630984.630984 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014686346054077148s
DEBUG 05-06 11:55:55.630538.630538 cuda_h.py:27] end moe_wait_copy_tasks cost 14.829 ms
DEBUG 05-06 11:55:55.635198.635198 cuda_h.py:27] end moe_vllm_forward cost 4.090 ms
DEBUG 05-06 11:55:55.635601.635601 cuda_h.py:27] end moe_cpu_merge cost 0.069 ms
DEBUG 05-06 11:55:55.635983.635983 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.635959.635959 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.834ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.636978.636978 cuda_h.py:27] end *layer_moe_fused cost 23.912 ms
DEBUG 05-06 11:55:55.640484.640484 cuda_h.py:27] end prefill_merge_scale cost 4.381 ms
DEBUG 05-06 11:55:55.640805.640805 cuda_h.py:27] end prefill_layer cost 33.293 ms
DEBUG 05-06 11:55:55.640885.640885 lmp.py:1391] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 11:55:55.641270.641270 lmp.py:1347] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 11:55:55.641383.641383 cuda_h.py:27] end prefill_ln cost 0.255 ms
DEBUG 05-06 11:55:55.643141.643141 cuda_h.py:27] end prefill_attn cost 2.210 ms
DEBUG 05-06 11:55:55.644114.644114 cuda_h.py:27] end prefill_ffn_prep cost 0.491 ms
DEBUG 05-06 11:55:55.645753.645753 cuda_h.py:27] end prefill_gate cost 0.525 ms
experts_cpu_alloc {'expert_ids': [31, 51, 75, 87, 99, 11, 35, 47, 111, 83, 95, 19, 79, 16, 104, 36, 24, 20, 60, 76, 12, 84, 120, 72, 100, 112, 64, 52, 88, 17, 89, 25, 61, 105, 69, 117, 97, 101, 9, 93, 121, 113, 41, 81, 53, 85, 33, 125, 109, 86, 90, 110, 34, 62, 70, 106, 126, 74, 18, 26, 98, 10, 38, 58, 118, 122, 114], 'token_total': 492, 'token_per_expert': {31: 1, 51: 1, 75: 1, 87: 1, 99: 2, 11: 5, 35: 5, 47: 5, 111: 6, 83: 7, 95: 7, 19: 8, 79: 8, 16: 1, 104: 3, 36: 4, 24: 5, 20: 6, 60: 6, 76: 6, 12: 8, 84: 9, 120: 9, 72: 11, 100: 11, 112: 14, 64: 15, 52: 20, 88: 22, 17: 1, 89: 1, 25: 2, 61: 2, 105: 2, 69: 3, 117: 3, 97: 4, 101: 4, 9: 8, 93: 8, 121: 8, 113: 11, 41: 12, 81: 13, 53: 17, 85: 17, 33: 20, 125: 21, 109: 26, 86: 1, 90: 1, 110: 2, 34: 3, 62: 3, 70: 3, 106: 3, 126: 3, 74: 5, 18: 6, 26: 6, 98: 7, 10: 9, 38: 9, 58: 9, 118: 10, 122: 10, 114: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 43, 55, 59, 63, 71, 103, 107, 123], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 25, 'token_total': 866, 'token_per_expert': {3: 295, 7: 257, 15: 10, 27: 23, 43: 27, 55: 10, 59: 27, 63: 67, 71: 16, 103: 9, 107: 96, 123: 29}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 28, 32, 40, 44, 56, 68, 92, 108, 116], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 28, 'token_total': 990, 'token_per_expert': {0: 266, 4: 300, 8: 29, 28: 35, 32: 23, 40: 38, 44: 29, 56: 45, 68: 140, 92: 35, 108: 28, 116: 22}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 37, 45, 49, 57, 65, 73, 77], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 31, 'token_total': 965, 'token_per_expert': {1: 264, 5: 302, 13: 30, 21: 36, 37: 37, 45: 76, 49: 77, 57: 34, 65: 29, 73: 39, 77: 41}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 42, 46, 50, 54, 66, 82, 94, 102], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 783, 'token_per_expert': {2: 264, 6: 257, 30: 29, 42: 19, 46: 13, 50: 13, 54: 21, 66: 26, 82: 17, 94: 77, 102: 47}}
INFO 05-06 11:55:55.646734.646734 lmp.py:1836] [layer_moe_fused] layer=20 prefix: 0.483ms alloc: 0.413ms
INFO 05-06 11:55:55.646448.646448 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.054473876953125e-05 seconds
INFO 05-06 11:55:55.647381.647381 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007538795471191406s
INFO 05-06 11:55:55.648071.648071 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005819797515869141 seconds
DEBUG 05-06 11:55:55.649895.649895 cuda_h.py:27] end moe_cpu_prep_submit cost 1.164 ms
INFO 05-06 11:55:55.658720.658720 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.008733272552490234s
DEBUG 05-06 11:55:55.658035.658035 cuda_h.py:27] end moe_wait_copy_tasks cost 8.880 ms
DEBUG 05-06 11:55:55.662851.662851 cuda_h.py:27] end moe_vllm_forward cost 3.998 ms
DEBUG 05-06 11:55:55.662731.662731 cuda_h.py:27] end moe_cpu_merge cost 0.069 ms
DEBUG 05-06 11:55:55.663310.663310 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.663664.663664 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.881ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.664703.664703 cuda_h.py:27] end *layer_moe_fused cost 18.151 ms
DEBUG 05-06 11:55:55.668456.668456 cuda_h.py:27] end prefill_merge_scale cost 4.426 ms
DEBUG 05-06 11:55:55.668446.668446 cuda_h.py:27] end prefill_layer cost 27.585 ms
DEBUG 05-06 11:55:55.668841.668841 lmp.py:1391] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 11:55:55.668465.668465 lmp.py:1347] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 11:55:55.669074.669074 cuda_h.py:27] end prefill_ln cost 0.254 ms
DEBUG 05-06 11:55:55.671006.671006 cuda_h.py:27] end prefill_attn cost 2.175 ms
DEBUG 05-06 11:55:55.672078.672078 cuda_h.py:27] end prefill_ffn_prep cost 0.492 ms
DEBUG 05-06 11:55:55.673319.673319 cuda_h.py:27] end prefill_gate cost 0.517 ms
experts_cpu_alloc {'expert_ids': [19, 47, 59, 99, 27, 107, 23, 43, 115, 55, 119, 71, 87, 95, 123, 31, 56, 104, 16, 52, 60, 32, 96, 64, 88, 80, 20, 40, 24, 44, 36, 12, 72, 9, 25, 77, 17, 113, 101, 69, 125, 93, 121, 21, 45, 81, 33, 57, 97, 41, 54, 98, 126, 94, 106, 114, 38, 50, 74, 82, 34, 58, 118, 42, 70, 10, 86, 30], 'token_total': 488, 'token_per_expert': {19: 1, 47: 1, 59: 1, 99: 1, 27: 2, 107: 2, 23: 3, 43: 3, 115: 6, 55: 7, 119: 7, 71: 8, 87: 11, 95: 11, 123: 11, 31: 12, 56: 1, 104: 1, 16: 2, 52: 2, 60: 2, 32: 3, 96: 4, 64: 5, 88: 5, 80: 6, 20: 8, 40: 8, 24: 10, 44: 10, 36: 13, 12: 16, 72: 19, 9: 1, 25: 1, 77: 1, 17: 2, 113: 3, 101: 4, 69: 5, 125: 5, 93: 6, 121: 7, 21: 8, 45: 8, 81: 9, 33: 10, 57: 19, 97: 19, 41: 21, 54: 1, 98: 2, 126: 3, 94: 4, 106: 4, 114: 5, 38: 6, 50: 8, 74: 8, 82: 10, 34: 11, 58: 11, 118: 11, 42: 13, 70: 13, 10: 14, 86: 15, 30: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 35, 51, 67, 75, 79, 83, 103, 111, 127], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 28, 'token_total': 770, 'token_per_expert': {3: 259, 7: 280, 11: 38, 35: 29, 51: 33, 67: 12, 75: 13, 79: 12, 83: 30, 103: 24, 111: 25, 127: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 48, 68, 76, 84, 92, 100, 112, 120, 124], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 29, 'token_total': 888, 'token_per_expert': {0: 257, 4: 276, 8: 29, 48: 54, 68: 23, 76: 34, 84: 28, 92: 37, 100: 63, 112: 33, 120: 30, 124: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 37, 53, 61, 65, 73, 105, 109], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 970, 'token_per_expert': {1: 325, 5: 348, 13: 27, 29: 29, 37: 27, 53: 31, 61: 26, 65: 63, 73: 31, 105: 39, 109: 24}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 26, 46, 62, 78, 90, 102, 110, 122], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 980, 'token_per_expert': {2: 275, 6: 331, 18: 41, 26: 56, 46: 40, 62: 24, 78: 96, 90: 26, 102: 23, 110: 32, 122: 36}}
INFO 05-06 11:55:55.674022.674022 lmp.py:1836] [layer_moe_fused] layer=21 prefix: 0.481ms alloc: 0.419ms
INFO 05-06 11:55:55.674928.674928 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.054473876953125e-05 seconds
INFO 05-06 11:55:55.675888.675888 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007696151733398438s
INFO 05-06 11:55:55.676387.676387 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005807876586914062 seconds
DEBUG 05-06 11:55:55.677261.677261 cuda_h.py:27] end moe_cpu_prep_submit cost 1.193 ms
INFO 05-06 11:55:55.694527.694527 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.017403125762939453s
DEBUG 05-06 11:55:55.695657.695657 cuda_h.py:27] end moe_wait_copy_tasks cost 17.549 ms
DEBUG 05-06 11:55:55.699698.699698 cuda_h.py:27] end moe_vllm_forward cost 3.987 ms
DEBUG 05-06 11:55:55.699306.699306 cuda_h.py:27] end moe_cpu_merge cost 0.081 ms
DEBUG 05-06 11:55:55.699540.699540 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.700931.700931 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.971ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.700388.700388 cuda_h.py:27] end *layer_moe_fused cost 27.051 ms
DEBUG 05-06 11:55:55.705484.705484 cuda_h.py:27] end prefill_merge_scale cost 4.786 ms
DEBUG 05-06 11:55:55.705050.705050 cuda_h.py:27] end prefill_layer cost 36.876 ms
DEBUG 05-06 11:55:55.706962.706962 lmp.py:1391] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 11:55:55.706586.706586 lmp.py:1347] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 11:55:55.706163.706163 cuda_h.py:27] end prefill_ln cost 0.257 ms
DEBUG 05-06 11:55:55.708736.708736 cuda_h.py:27] end prefill_attn cost 2.215 ms
DEBUG 05-06 11:55:55.709272.709272 cuda_h.py:27] end prefill_ffn_prep cost 0.486 ms
DEBUG 05-06 11:55:55.710473.710473 cuda_h.py:27] end prefill_gate cost 0.523 ms
experts_cpu_alloc {'expert_ids': [23, 63, 71, 27, 67, 87, 95, 39, 47, 51, 83, 15, 107, 11, 115, 79, 31, 99, 19, 43, 36, 60, 20, 96, 84, 48, 112, 32, 16, 44, 124, 88, 28, 40, 76, 108, 13, 17, 9, 61, 109, 29, 65, 81, 125, 45, 25, 57, 85, 41, 14, 54, 110, 62, 98, 106, 10, 122, 42, 34, 102, 58, 26, 118, 30, 66], 'token_total': 487, 'token_per_expert': {23: 1, 63: 1, 71: 1, 27: 2, 67: 2, 87: 2, 95: 2, 39: 3, 47: 6, 51: 6, 83: 6, 15: 7, 107: 9, 11: 10, 115: 10, 79: 11, 31: 14, 99: 17, 19: 19, 43: 25, 36: 1, 60: 1, 20: 2, 96: 2, 84: 3, 48: 6, 112: 7, 32: 8, 16: 10, 44: 10, 124: 11, 88: 14, 28: 15, 40: 16, 76: 22, 108: 24, 13: 1, 17: 1, 9: 2, 61: 2, 109: 2, 29: 3, 65: 3, 81: 3, 125: 3, 45: 5, 25: 6, 57: 6, 85: 9, 41: 11, 14: 1, 54: 1, 110: 1, 62: 2, 98: 2, 106: 3, 10: 5, 122: 5, 42: 6, 34: 10, 102: 12, 58: 13, 26: 15, 118: 15, 30: 16, 66: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 55, 59, 75, 103, 111, 119, 123, 127], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 31, 'token_total': 907, 'token_per_expert': {3: 257, 7: 282, 35: 82, 55: 36, 59: 43, 75: 26, 103: 49, 111: 25, 119: 37, 123: 37, 127: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 24, 64, 68, 72, 92, 100, 116, 120], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 1102, 'token_per_expert': {0: 260, 4: 256, 8: 32, 24: 64, 64: 107, 68: 43, 72: 97, 92: 29, 100: 153, 116: 32, 120: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 53, 69, 73, 89, 93, 101, 113, 117], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 762, 'token_per_expert': {1: 274, 5: 257, 33: 12, 53: 27, 69: 16, 73: 49, 89: 17, 93: 50, 101: 14, 113: 12, 117: 34}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 70, 74, 82, 86, 90, 94, 126], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 838, 'token_per_expert': {2: 257, 6: 257, 38: 31, 46: 20, 70: 30, 74: 67, 82: 21, 86: 31, 90: 32, 94: 28, 126: 64}}
INFO 05-06 11:55:55.711547.711547 lmp.py:1836] [layer_moe_fused] layer=22 prefix: 0.480ms alloc: 0.413ms
INFO 05-06 11:55:55.712452.712452 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 4.9591064453125e-05 seconds
INFO 05-06 11:55:55.713854.713854 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007414817810058594s
INFO 05-06 11:55:55.713147.713147 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005705356597900391 seconds
DEBUG 05-06 11:55:55.714806.714806 cuda_h.py:27] end moe_cpu_prep_submit cost 1.170 ms
INFO 05-06 11:55:55.728896.728896 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013880491256713867s
DEBUG 05-06 11:55:55.728496.728496 cuda_h.py:27] end moe_wait_copy_tasks cost 14.025 ms
DEBUG 05-06 11:55:55.733437.733437 cuda_h.py:27] end moe_vllm_forward cost 3.984 ms
DEBUG 05-06 11:55:55.733085.733085 cuda_h.py:27] end moe_cpu_merge cost 0.069 ms
DEBUG 05-06 11:55:55.733219.733219 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.733050.733050 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.827ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.734016.734016 cuda_h.py:27] end *layer_moe_fused cost 23.223 ms
DEBUG 05-06 11:55:55.738097.738097 cuda_h.py:27] end prefill_merge_scale cost 3.718 ms
DEBUG 05-06 11:55:55.738325.738325 cuda_h.py:27] end prefill_layer cost 31.969 ms
DEBUG 05-06 11:55:55.738385.738385 lmp.py:1391] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 11:55:55.738009.738009 lmp.py:1347] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 11:55:55.738791.738791 cuda_h.py:27] end prefill_ln cost 0.257 ms
DEBUG 05-06 11:55:55.741830.741830 cuda_h.py:27] end prefill_attn cost 2.276 ms
DEBUG 05-06 11:55:55.741319.741319 cuda_h.py:27] end prefill_ffn_prep cost 0.485 ms
DEBUG 05-06 11:55:55.743275.743275 cuda_h.py:27] end prefill_gate cost 0.518 ms
experts_cpu_alloc {'expert_ids': [55, 119, 11, 23, 27, 95, 107, 127, 99, 19, 75, 51, 103, 91, 71, 31, 59, 28, 88, 64, 12, 60, 92, 36, 68, 120, 32, 48, 52, 76, 40, 124, 116, 24, 112, 13, 69, 77, 93, 41, 49, 53, 81, 89, 33, 57, 9, 73, 17, 105, 117, 109, 50, 74, 62, 82, 102, 58, 110, 10, 14, 38, 54, 66, 122, 42, 34, 26, 30], 'token_total': 470, 'token_per_expert': {55: 1, 119: 1, 11: 2, 23: 2, 27: 2, 95: 2, 107: 2, 127: 2, 99: 4, 19: 6, 75: 6, 51: 7, 103: 7, 91: 9, 71: 11, 31: 15, 59: 15, 28: 1, 88: 1, 64: 2, 12: 3, 60: 4, 92: 4, 36: 5, 68: 5, 120: 5, 32: 6, 48: 6, 52: 6, 76: 6, 40: 7, 124: 10, 116: 12, 24: 13, 112: 13, 13: 1, 69: 1, 77: 1, 93: 1, 41: 3, 49: 3, 53: 3, 81: 3, 89: 6, 33: 9, 57: 9, 9: 10, 73: 18, 17: 19, 105: 20, 117: 24, 109: 31, 50: 1, 74: 2, 62: 3, 82: 3, 102: 3, 58: 4, 110: 4, 10: 5, 14: 5, 38: 5, 54: 5, 66: 5, 122: 7, 42: 10, 34: 13, 26: 15, 30: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 39, 43, 47, 67, 79, 83, 87, 115, 123], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 29, 'token_total': 926, 'token_per_expert': {3: 286, 7: 260, 35: 27, 39: 58, 43: 55, 47: 29, 67: 69, 79: 44, 83: 30, 87: 17, 115: 15, 123: 36}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 44, 56, 72, 80, 84, 100, 104, 108], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 30, 'token_total': 774, 'token_per_expert': {0: 256, 4: 257, 8: 17, 16: 22, 44: 35, 56: 58, 72: 16, 80: 17, 84: 22, 100: 22, 104: 21, 108: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25, 29, 37, 61, 65, 85, 97, 125], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 1043, 'token_per_expert': {1: 293, 5: 296, 21: 104, 25: 39, 29: 39, 37: 44, 61: 59, 65: 43, 85: 37, 97: 45, 125: 44}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 46, 78, 86, 90, 98, 106, 118], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 883, 'token_per_expert': {2: 274, 6: 273, 18: 25, 22: 18, 46: 59, 78: 21, 86: 83, 90: 34, 98: 43, 106: 15, 118: 38}}
INFO 05-06 11:55:55.744786.744786 lmp.py:1836] [layer_moe_fused] layer=23 prefix: 0.486ms alloc: 0.416ms
INFO 05-06 11:55:55.744474.744474 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.3882598876953125e-05 seconds
INFO 05-06 11:55:55.745569.745569 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007154941558837891s
INFO 05-06 11:55:55.745929.745929 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.000583648681640625 seconds
DEBUG 05-06 11:55:55.746300.746300 cuda_h.py:27] end moe_cpu_prep_submit cost 0.983 ms
INFO 05-06 11:55:55.761720.761720 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014602184295654297s
DEBUG 05-06 11:55:55.761751.761751 cuda_h.py:27] end moe_wait_copy_tasks cost 14.746 ms
DEBUG 05-06 11:55:55.765998.765998 cuda_h.py:27] end moe_vllm_forward cost 3.995 ms
DEBUG 05-06 11:55:55.765208.765208 cuda_h.py:27] end moe_cpu_merge cost 0.069 ms
DEBUG 05-06 11:55:55.766548.766548 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.766608.766608 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.980ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.767483.767483 cuda_h.py:27] end *layer_moe_fused cost 23.890 ms
DEBUG 05-06 11:55:55.770077.770077 cuda_h.py:27] end prefill_merge_scale cost 3.643 ms
DEBUG 05-06 11:55:55.771067.771067 cuda_h.py:27] end prefill_layer cost 32.590 ms
DEBUG 05-06 11:55:55.771708.771708 lmp.py:1391] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 11:55:55.771571.771571 lmp.py:1347] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 11:55:55.771121.771121 cuda_h.py:27] end prefill_ln cost 0.253 ms
DEBUG 05-06 11:55:55.773461.773461 cuda_h.py:27] end prefill_attn cost 2.181 ms
DEBUG 05-06 11:55:55.774480.774480 cuda_h.py:27] end prefill_ffn_prep cost 0.490 ms
DEBUG 05-06 11:55:55.775928.775928 cuda_h.py:27] end prefill_gate cost 0.564 ms
experts_cpu_alloc {'expert_ids': [15, 47, 95, 31, 99, 115, 87, 107, 55, 119, 75, 127, 79, 111, 43, 83, 24, 28, 80, 104, 112, 40, 76, 32, 68, 84, 92, 124, 20, 96, 120, 100, 108, 8, 25, 65, 101, 113, 93, 53, 61, 105, 57, 81, 9, 49, 13, 29, 10, 18, 22, 26, 78, 102, 106, 126, 42, 66, 38, 46, 118, 62, 74, 82, 50, 86], 'token_total': 394, 'token_per_expert': {15: 1, 47: 1, 95: 1, 31: 2, 99: 2, 115: 2, 87: 3, 107: 3, 55: 5, 119: 6, 75: 7, 127: 8, 79: 9, 111: 11, 43: 16, 83: 17, 24: 2, 28: 2, 80: 2, 104: 2, 112: 2, 40: 3, 76: 3, 32: 4, 68: 4, 84: 4, 92: 6, 124: 6, 20: 7, 96: 7, 120: 7, 100: 8, 108: 10, 8: 13, 25: 1, 65: 1, 101: 1, 113: 1, 93: 2, 53: 5, 61: 5, 105: 5, 57: 6, 81: 8, 9: 10, 49: 10, 13: 16, 29: 18, 10: 1, 18: 1, 22: 1, 26: 1, 78: 2, 102: 2, 106: 2, 126: 3, 42: 5, 66: 5, 38: 7, 46: 7, 118: 12, 62: 13, 74: 13, 82: 13, 50: 14, 86: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 35, 63, 67, 71, 91], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 915, 'token_per_expert': {3: 256, 7: 271, 11: 55, 19: 38, 23: 20, 27: 79, 35: 24, 63: 60, 67: 45, 71: 34, 91: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 36, 44, 48, 52, 56, 60, 64], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 903, 'token_per_expert': {0: 256, 4: 290, 12: 40, 16: 28, 36: 17, 44: 50, 48: 22, 52: 69, 56: 40, 60: 18, 64: 73}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 33, 37, 45, 73, 77, 97, 109, 121], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 949, 'token_per_expert': {1: 282, 5: 270, 17: 28, 33: 55, 37: 18, 45: 37, 73: 23, 77: 22, 97: 69, 109: 23, 121: 122}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 34, 70, 90, 94, 98, 110, 114, 122], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 935, 'token_per_expert': {2: 256, 6: 304, 30: 18, 34: 43, 70: 70, 90: 112, 94: 26, 98: 40, 110: 20, 114: 28, 122: 18}}
INFO 05-06 11:55:55.776451.776451 lmp.py:1836] [layer_moe_fused] layer=24 prefix: 0.478ms alloc: 0.399ms
INFO 05-06 11:55:55.777595.777595 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.054473876953125e-05 seconds
INFO 05-06 11:55:55.778481.778481 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007212162017822266s
INFO 05-06 11:55:55.778595.778595 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005779266357421875 seconds
DEBUG 05-06 11:55:55.779329.779329 cuda_h.py:27] end moe_cpu_prep_submit cost 0.927 ms
INFO 05-06 11:55:55.792009.792009 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013147115707397461s
DEBUG 05-06 11:55:55.792125.792125 cuda_h.py:27] end moe_wait_copy_tasks cost 13.284 ms
DEBUG 05-06 11:55:55.797364.797364 cuda_h.py:27] end moe_vllm_forward cost 3.958 ms
DEBUG 05-06 11:55:55.797859.797859 cuda_h.py:27] end moe_cpu_merge cost 0.068 ms
DEBUG 05-06 11:55:55.797085.797085 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.797108.797108 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.792ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.798937.798937 cuda_h.py:27] end *layer_moe_fused cost 22.305 ms
DEBUG 05-06 11:55:55.803138.803138 cuda_h.py:27] end prefill_merge_scale cost 5.355 ms
DEBUG 05-06 11:55:55.803135.803135 cuda_h.py:27] end prefill_layer cost 32.682 ms
DEBUG 05-06 11:55:55.804184.804184 lmp.py:1391] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 11:55:55.804570.804570 lmp.py:1347] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 11:55:55.804756.804756 cuda_h.py:27] end prefill_ln cost 0.257 ms
DEBUG 05-06 11:55:55.806718.806718 cuda_h.py:27] end prefill_attn cost 2.150 ms
DEBUG 05-06 11:55:55.807445.807445 cuda_h.py:27] end prefill_ffn_prep cost 0.487 ms
DEBUG 05-06 11:55:55.808444.808444 cuda_h.py:27] end prefill_gate cost 0.523 ms
experts_cpu_alloc {'expert_ids': [15, 95, 27, 55, 127, 75, 23, 31, 103, 47, 119, 79, 87, 99, 51, 43, 91, 19, 63, 76, 84, 108, 12, 24, 92, 124, 8, 72, 88, 112, 116, 48, 56, 36, 120, 57, 61, 81, 73, 13, 33, 53, 121, 21, 29, 49, 9, 17, 109, 89, 25, 30, 42, 74, 118, 38, 66, 122, 26, 22, 46, 78, 126, 50, 10, 14], 'token_total': 405, 'token_per_expert': {15: 1, 95: 1, 27: 2, 55: 2, 127: 2, 75: 3, 23: 4, 31: 4, 103: 4, 47: 5, 119: 5, 79: 6, 87: 7, 99: 7, 51: 9, 43: 10, 91: 10, 19: 13, 63: 13, 76: 1, 84: 1, 108: 1, 12: 3, 24: 3, 92: 4, 124: 5, 8: 7, 72: 7, 88: 7, 112: 7, 116: 12, 48: 13, 56: 13, 36: 16, 120: 18, 57: 1, 61: 1, 81: 1, 73: 2, 13: 3, 33: 3, 53: 3, 121: 4, 21: 5, 29: 6, 49: 6, 9: 7, 17: 7, 109: 8, 89: 10, 25: 11, 30: 1, 42: 3, 74: 3, 118: 3, 38: 4, 66: 4, 122: 4, 26: 5, 22: 6, 46: 6, 78: 10, 126: 10, 50: 13, 10: 14, 14: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 35, 39, 67, 71, 83, 107, 111, 123], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 30, 'token_total': 873, 'token_per_expert': {3: 284, 7: 274, 11: 17, 35: 58, 39: 14, 67: 21, 71: 19, 83: 27, 107: 70, 111: 25, 123: 64}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 44, 52, 60, 64, 68, 80, 100, 104], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 1020, 'token_per_expert': {0: 268, 4: 260, 16: 157, 44: 20, 52: 58, 60: 48, 64: 41, 68: 87, 80: 27, 100: 22, 104: 32}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 41, 45, 69, 77, 85, 93, 97, 117, 125], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 795, 'token_per_expert': {1: 257, 5: 265, 41: 13, 45: 68, 69: 45, 77: 13, 85: 51, 93: 20, 97: 12, 117: 39, 125: 12}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 34, 58, 70, 82, 90, 106, 110, 114], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 1003, 'token_per_expert': {2: 308, 6: 261, 18: 62, 34: 22, 58: 130, 70: 34, 82: 22, 90: 22, 106: 22, 110: 101, 114: 19}}
INFO 05-06 11:55:55.809305.809305 lmp.py:1836] [layer_moe_fused] layer=25 prefix: 0.477ms alloc: 0.402ms
INFO 05-06 11:55:55.810932.810932 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.6743621826171875e-05 seconds
INFO 05-06 11:55:55.811951.811951 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007166862487792969s
INFO 05-06 11:55:55.811185.811185 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005972385406494141 seconds
DEBUG 05-06 11:55:55.812171.812171 cuda_h.py:27] end moe_cpu_prep_submit cost 1.136 ms
INFO 05-06 11:55:55.826558.826558 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013862848281860352s
DEBUG 05-06 11:55:55.826205.826205 cuda_h.py:27] end moe_wait_copy_tasks cost 14.004 ms
DEBUG 05-06 11:55:55.830179.830179 cuda_h.py:27] end moe_vllm_forward cost 3.974 ms
DEBUG 05-06 11:55:55.830635.830635 cuda_h.py:27] end moe_cpu_merge cost 0.071 ms
DEBUG 05-06 11:55:55.830599.830599 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.831767.831767 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.693ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.831424.831424 cuda_h.py:27] end *layer_moe_fused cost 22.846 ms
DEBUG 05-06 11:55:55.836691.836691 cuda_h.py:27] end prefill_merge_scale cost 4.734 ms
DEBUG 05-06 11:55:55.836496.836496 cuda_h.py:27] end prefill_layer cost 32.598 ms
DEBUG 05-06 11:55:55.837267.837267 lmp.py:1391] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 11:55:55.837652.837652 lmp.py:1347] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 11:55:55.837827.837827 cuda_h.py:27] end prefill_ln cost 0.256 ms
DEBUG 05-06 11:55:55.839441.839441 cuda_h.py:27] end prefill_attn cost 2.244 ms
DEBUG 05-06 11:55:55.840076.840076 cuda_h.py:27] end prefill_ffn_prep cost 0.490 ms
DEBUG 05-06 11:55:55.841748.841748 cuda_h.py:27] end prefill_gate cost 0.519 ms
experts_cpu_alloc {'expert_ids': [55, 47, 71, 31, 91, 23, 63, 35, 67, 115, 99, 75, 79, 59, 19, 48, 64, 32, 40, 44, 92, 72, 80, 96, 108, 116, 28, 112, 88, 8, 68, 36, 9, 93, 117, 121, 41, 109, 13, 81, 57, 61, 25, 125, 29, 37, 77, 105, 45, 54, 42, 122, 18, 74, 50, 118, 14, 26, 38, 30, 10], 'token_total': 453, 'token_per_expert': {55: 2, 47: 3, 71: 3, 31: 4, 91: 4, 23: 6, 63: 6, 35: 7, 67: 8, 115: 8, 99: 11, 75: 15, 79: 16, 59: 17, 19: 19, 48: 1, 64: 1, 32: 2, 40: 3, 44: 3, 92: 3, 72: 4, 80: 4, 96: 5, 108: 5, 116: 5, 28: 6, 112: 8, 88: 9, 8: 11, 68: 15, 36: 16, 9: 1, 93: 1, 117: 1, 121: 1, 41: 3, 109: 3, 13: 4, 81: 6, 57: 8, 61: 8, 25: 9, 125: 9, 29: 10, 37: 13, 77: 13, 105: 13, 45: 14, 54: 4, 42: 5, 122: 5, 18: 6, 74: 7, 50: 8, 118: 9, 14: 10, 26: 11, 38: 11, 30: 12, 10: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 43, 51, 87, 95, 103, 111, 123], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 940, 'token_per_expert': {3: 269, 7: 256, 15: 25, 27: 49, 43: 46, 51: 19, 87: 90, 95: 63, 103: 19, 111: 79, 123: 25}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 52, 56, 60, 76, 84, 104, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 880, 'token_per_expert': {0: 260, 4: 260, 20: 106, 24: 44, 52: 34, 56: 17, 60: 26, 76: 17, 84: 53, 104: 38, 124: 25}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 49, 65, 73, 85, 89, 97, 113], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 27, 'token_total': 1071, 'token_per_expert': {1: 263, 5: 256, 17: 97, 49: 19, 65: 61, 73: 33, 85: 138, 89: 87, 97: 14, 113: 103}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 66, 70, 78, 86, 90, 102, 114, 126], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 22, 'token_total': 752, 'token_per_expert': {2: 261, 6: 256, 66: 19, 70: 25, 78: 21, 86: 29, 90: 19, 102: 21, 114: 77, 126: 24}}
INFO 05-06 11:55:55.842218.842218 lmp.py:1836] [layer_moe_fused] layer=26 prefix: 0.476ms alloc: 0.391ms
INFO 05-06 11:55:55.843077.843077 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.078315734863281e-05 seconds
INFO 05-06 11:55:55.844915.844915 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007512569427490234s
INFO 05-06 11:55:55.844645.844645 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005743503570556641 seconds
DEBUG 05-06 11:55:55.845942.845942 cuda_h.py:27] end moe_cpu_prep_submit cost 1.169 ms
INFO 05-06 11:55:55.858668.858668 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012862443923950195s
DEBUG 05-06 11:55:55.858261.858261 cuda_h.py:27] end moe_wait_copy_tasks cost 12.998 ms
DEBUG 05-06 11:55:55.863022.863022 cuda_h.py:27] end moe_vllm_forward cost 3.926 ms
DEBUG 05-06 11:55:55.863855.863855 cuda_h.py:27] end moe_cpu_merge cost 0.069 ms
DEBUG 05-06 11:55:55.863854.863854 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.863141.863141 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.856ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.864249.864249 cuda_h.py:27] end *layer_moe_fused cost 22.346 ms
DEBUG 05-06 11:55:55.869764.869764 cuda_h.py:27] end prefill_merge_scale cost 5.440 ms
DEBUG 05-06 11:55:55.870377.870377 cuda_h.py:27] end prefill_layer cost 32.905 ms
DEBUG 05-06 11:55:55.870700.870700 lmp.py:1391] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 11:55:55.870370.870370 lmp.py:1347] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 11:55:55.870577.870577 cuda_h.py:27] end prefill_ln cost 0.258 ms
DEBUG 05-06 11:55:55.873443.873443 cuda_h.py:27] end prefill_attn cost 2.254 ms
DEBUG 05-06 11:55:55.873839.873839 cuda_h.py:27] end prefill_ffn_prep cost 0.488 ms
DEBUG 05-06 11:55:55.874293.874293 cuda_h.py:27] end prefill_gate cost 0.526 ms
experts_cpu_alloc {'expert_ids': [11, 55, 71, 59, 19, 63, 67, 47, 15, 39, 27, 91, 23, 83, 127, 75, 119, 35, 44, 72, 84, 80, 124, 32, 116, 68, 96, 40, 20, 28, 112, 12, 56, 64, 89, 101, 29, 93, 97, 81, 113, 21, 125, 105, 49, 85, 41, 121, 22, 30, 86, 110, 126, 26, 10, 58, 74, 54, 90, 106, 42, 94, 114, 66, 122, 118], 'token_total': 505, 'token_per_expert': {11: 1, 55: 1, 71: 1, 59: 2, 19: 3, 63: 3, 67: 3, 47: 4, 15: 5, 39: 5, 27: 8, 91: 9, 23: 10, 83: 10, 127: 11, 75: 12, 119: 14, 35: 17, 44: 1, 72: 1, 84: 1, 80: 2, 124: 2, 32: 4, 116: 5, 68: 6, 96: 8, 40: 9, 20: 11, 28: 12, 112: 13, 12: 14, 56: 17, 64: 17, 89: 1, 101: 1, 29: 2, 93: 2, 97: 2, 81: 3, 113: 3, 21: 6, 125: 6, 105: 9, 49: 10, 85: 12, 41: 20, 121: 20, 22: 1, 30: 1, 86: 2, 110: 2, 126: 2, 26: 3, 10: 4, 58: 8, 74: 8, 54: 9, 90: 11, 106: 11, 42: 16, 94: 16, 114: 16, 66: 17, 122: 18, 118: 21}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 43, 51, 79, 87, 95, 103, 111, 115, 123], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 30, 'token_total': 1022, 'token_per_expert': {3: 291, 7: 276, 31: 26, 43: 67, 51: 27, 79: 28, 87: 84, 95: 59, 103: 43, 111: 36, 115: 49, 123: 36}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 24, 36, 48, 76, 88, 100, 108, 120], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 27, 'token_total': 893, 'token_per_expert': {0: 260, 4: 270, 8: 18, 24: 45, 36: 33, 48: 37, 76: 43, 88: 45, 100: 64, 108: 22, 120: 56}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 33, 37, 45, 53, 61, 65, 109], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 25, 'token_total': 853, 'token_per_expert': {1: 291, 5: 257, 13: 27, 25: 41, 33: 41, 37: 26, 45: 51, 53: 23, 61: 22, 65: 48, 109: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 46, 50, 62, 70, 78, 82, 98], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 29, 'token_total': 823, 'token_per_expert': {2: 256, 6: 259, 14: 34, 18: 23, 46: 31, 50: 64, 62: 29, 70: 25, 78: 28, 82: 50, 98: 24}}
INFO 05-06 11:55:55.876864.876864 lmp.py:1836] [layer_moe_fused] layer=27 prefix: 0.492ms alloc: 0.414ms
INFO 05-06 11:55:55.876491.876491 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.5789947509765625e-05 seconds
INFO 05-06 11:55:55.877688.877688 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007374286651611328s
INFO 05-06 11:55:55.877869.877869 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005941390991210938 seconds
DEBUG 05-06 11:55:55.878104.878104 cuda_h.py:27] end moe_cpu_prep_submit cost 1.023 ms
INFO 05-06 11:55:55.890130.890130 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.011499881744384766s
DEBUG 05-06 11:55:55.890843.890843 cuda_h.py:27] end moe_wait_copy_tasks cost 11.660 ms
DEBUG 05-06 11:55:55.895751.895751 cuda_h.py:27] end moe_vllm_forward cost 4.139 ms
DEBUG 05-06 11:55:55.895161.895161 cuda_h.py:27] end moe_cpu_merge cost 0.071 ms
DEBUG 05-06 11:55:55.895604.895604 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.895969.895969 lmp.py:1950] [layer_moe_fused] vllm triton time: 5.107ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.896906.896906 cuda_h.py:27] end *layer_moe_fused cost 21.339 ms
DEBUG 05-06 11:55:55.901413.901413 cuda_h.py:27] end prefill_merge_scale cost 5.014 ms
DEBUG 05-06 11:55:55.901126.901126 cuda_h.py:27] end prefill_layer cost 31.454 ms
DEBUG 05-06 11:55:55.901576.901576 lmp.py:1391] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 11:55:55.901484.901484 lmp.py:1347] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 11:55:55.902557.902557 cuda_h.py:27] end prefill_ln cost 0.261 ms
DEBUG 05-06 11:55:55.904014.904014 cuda_h.py:27] end prefill_attn cost 2.304 ms
DEBUG 05-06 11:55:55.905564.905564 cuda_h.py:27] end prefill_ffn_prep cost 0.494 ms
DEBUG 05-06 11:55:55.906038.906038 cuda_h.py:27] end prefill_gate cost 0.538 ms
experts_cpu_alloc {'expert_ids': [19, 59, 67, 83, 87, 99, 127, 15, 79, 23, 39, 43, 95, 123, 55, 28, 56, 108, 120, 92, 36, 44, 48, 88, 84, 24, 60, 100, 104, 29, 61, 93, 109, 73, 81, 9, 33, 65, 117, 69, 121, 37, 97, 105, 89, 38, 58, 114, 66, 94, 50, 54, 98, 122, 126, 18, 62, 74], 'token_total': 324, 'token_per_expert': {19: 1, 59: 1, 67: 1, 83: 1, 87: 2, 99: 2, 127: 2, 15: 3, 79: 5, 23: 6, 39: 6, 43: 8, 95: 10, 123: 12, 55: 20, 28: 1, 56: 1, 108: 2, 120: 2, 92: 3, 36: 4, 44: 4, 48: 6, 88: 6, 84: 7, 24: 8, 60: 9, 100: 9, 104: 15, 29: 1, 61: 1, 93: 1, 109: 1, 73: 2, 81: 4, 9: 5, 33: 7, 65: 8, 117: 8, 69: 9, 121: 9, 37: 11, 97: 11, 105: 11, 89: 15, 38: 1, 58: 1, 114: 1, 66: 2, 94: 2, 50: 3, 54: 3, 98: 6, 122: 7, 126: 8, 18: 9, 62: 10, 74: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 47, 71, 75, 91, 111, 115, 119], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 25, 'token_total': 1017, 'token_per_expert': {3: 258, 7: 257, 11: 38, 47: 36, 71: 31, 75: 44, 91: 52, 111: 193, 115: 83, 119: 25}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 32, 40, 52, 68, 76, 112], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 24, 'token_total': 1122, 'token_per_expert': {0: 256, 4: 256, 12: 196, 20: 152, 32: 37, 40: 35, 52: 26, 68: 24, 76: 72, 112: 68}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 49, 53, 57, 77, 85, 101, 113], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 26, 'token_total': 901, 'token_per_expert': {1: 270, 5: 268, 13: 24, 49: 131, 53: 33, 57: 87, 77: 17, 85: 18, 101: 20, 113: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 30, 46, 70, 78, 90, 106, 110], 'expert_count': 10, 'ideal_gpu_count': 10, 'keep_on_gpu': 10, 'hit_count_on_device': 23, 'token_total': 732, 'token_per_expert': {2: 256, 6: 257, 22: 33, 30: 17, 46: 22, 70: 31, 78: 18, 90: 48, 106: 12, 110: 38}}
INFO 05-06 11:55:55.907243.907243 lmp.py:1836] [layer_moe_fused] layer=28 prefix: 0.498ms alloc: 0.384ms
INFO 05-06 11:55:55.907818.907818 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 4.9114227294921875e-05 seconds
INFO 05-06 11:55:55.909662.909662 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007500648498535156s
INFO 05-06 11:55:55.909498.909498 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005843639373779297 seconds
DEBUG 05-06 11:55:55.910259.910259 cuda_h.py:27] end moe_cpu_prep_submit cost 0.858 ms
INFO 05-06 11:55:55.922990.922990 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012106180191040039s
DEBUG 05-06 11:55:55.922484.922484 cuda_h.py:27] end moe_wait_copy_tasks cost 12.239 ms
DEBUG 05-06 11:55:55.927235.927235 cuda_h.py:27] end moe_vllm_forward cost 4.055 ms
DEBUG 05-06 11:55:55.927261.927261 cuda_h.py:27] end moe_cpu_merge cost 0.070 ms
DEBUG 05-06 11:55:55.927908.927908 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 11:55:55.927526.927526 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.976ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.928059.928059 cuda_h.py:27] end *layer_moe_fused cost 21.676 ms
DEBUG 05-06 11:55:55.933752.933752 cuda_h.py:27] end prefill_merge_scale cost 5.222 ms
DEBUG 05-06 11:55:55.933603.933603 cuda_h.py:27] end prefill_layer cost 32.038 ms
DEBUG 05-06 11:55:55.934406.934406 lmp.py:1391] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 11:55:55.934507.934507 lmp.py:1347] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 11:55:55.934450.934450 cuda_h.py:27] end prefill_ln cost 0.257 ms
DEBUG 05-06 11:55:55.937338.937338 cuda_h.py:27] end prefill_attn cost 2.306 ms
DEBUG 05-06 11:55:55.937781.937781 cuda_h.py:27] end prefill_ffn_prep cost 0.487 ms
DEBUG 05-06 11:55:55.938962.938962 cuda_h.py:27] end prefill_gate cost 0.509 ms
experts_cpu_alloc {'expert_ids': [87, 127, 55, 111, 31, 35, 123, 11, 75, 115, 119, 63, 95, 83, 15, 104, 108, 84, 96, 88, 40, 76, 92, 80, 32, 24, 8, 116, 120, 44, 17, 33, 105, 13, 25, 37, 125, 21, 65, 77, 9, 61, 81, 69, 89, 73, 109, 101, 85, 122, 70, 118, 126, 38, 74, 94, 58, 46, 50, 10, 114, 66, 30, 78, 14, 62], 'token_total': 505, 'token_per_expert': {87: 2, 127: 2, 55: 3, 111: 4, 31: 5, 35: 5, 123: 5, 11: 6, 75: 6, 115: 6, 119: 6, 63: 7, 95: 8, 83: 9, 15: 10, 104: 1, 108: 1, 84: 2, 96: 2, 88: 3, 40: 4, 76: 4, 92: 4, 80: 5, 32: 6, 24: 8, 8: 9, 116: 11, 120: 12, 44: 15, 17: 1, 33: 3, 105: 3, 13: 4, 25: 4, 37: 4, 125: 5, 21: 6, 65: 6, 77: 11, 9: 13, 61: 14, 81: 15, 69: 16, 89: 17, 73: 18, 109: 18, 101: 19, 85: 20, 122: 1, 70: 2, 118: 2, 126: 2, 38: 3, 74: 3, 94: 4, 58: 6, 46: 7, 50: 7, 10: 9, 114: 9, 66: 12, 30: 15, 78: 17, 14: 19, 62: 19}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 43, 67, 71, 91, 99, 107], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 997, 'token_per_expert': {3: 269, 7: 357, 19: 39, 23: 29, 27: 27, 43: 49, 67: 12, 71: 37, 91: 92, 99: 76, 107: 10}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 28, 48, 52, 56, 60, 64, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 26, 'token_total': 959, 'token_per_expert': {0: 257, 4: 301, 16: 30, 20: 54, 28: 43, 48: 27, 52: 74, 56: 43, 60: 29, 64: 77, 124: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 49, 53, 57, 93, 97, 113, 117, 121], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 30, 'token_total': 787, 'token_per_expert': {1: 262, 5: 256, 29: 24, 49: 25, 53: 21, 57: 45, 93: 24, 97: 27, 113: 22, 117: 41, 121: 40}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 26, 42, 54, 82, 86, 90, 106], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 28, 'token_total': 848, 'token_per_expert': {2: 273, 6: 263, 18: 21, 22: 24, 26: 26, 42: 48, 54: 28, 82: 32, 86: 52, 90: 31, 106: 50}}
INFO 05-06 11:55:55.940413.940413 lmp.py:1836] [layer_moe_fused] layer=29 prefix: 0.488ms alloc: 0.405ms
INFO 05-06 11:55:55.940948.940948 lmp.py:1850] [layer_moe_fused] get_experts_task_ids time: 5.269050598144531e-05 seconds
INFO 05-06 11:55:55.941900.941900 lmp.py:1858] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007207393646240234s
INFO 05-06 11:55:55.941975.941975 lmp.py:1888] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005834102630615234 seconds
DEBUG 05-06 11:55:55.942443.942443 cuda_h.py:27] end moe_cpu_prep_submit cost 1.039 ms
INFO 05-06 11:55:55.955395.955395 lmp.py:1901] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012671709060668945s
DEBUG 05-06 11:55:55.955049.955049 cuda_h.py:27] end moe_wait_copy_tasks cost 12.818 ms
DEBUG 05-06 11:55:55.959347.959347 cuda_h.py:27] end moe_vllm_forward cost 3.950 ms
DEBUG 05-06 11:55:55.960849.960849 cuda_h.py:27] end moe_cpu_merge cost 0.070 ms
DEBUG 05-06 11:55:55.960325.960325 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 11:55:55.960909.960909 lmp.py:1950] [layer_moe_fused] vllm triton time: 4.866ms (seq_len=128 cg=False)
DEBUG 05-06 11:55:55.961270.961270 cuda_h.py:27] end *layer_moe_fused cost 22.139 ms
DEBUG 05-06 11:55:55.962034.962034 cuda_h.py:27] end prefill_merge_scale cost 0.611 ms
DEBUG 05-06 11:55:55.962971.962971 cuda_h.py:27] end prefill_layer cost 27.904 ms
DEBUG 05-06 11:55:55.962705.962705 lmp.py:1391] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 11:55:55.962521.962521 cuda_h.py:27] end prefill_step cost 1100.901 ms
INFO 05-06 11:55:55.962751.962751 lmp.py:1394] prefill time: 1.234539270401001 seconds
INFO 05-06 11:55:55.971255.971255 lmp.py:1406] Static-KV prefill complete; seqlens set to 128.
WARNING 05-06 11:55:55.972274.972274 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:55:55.972523.972523 helper.py:35]   NaN count (hidden): 720896
WARNING 05-06 11:55:55.972105.972105 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:55:55.973515.973515 helper.py:39]   NaN count (normed): 720896
WARNING 05-06 11:55:55.978936.978936 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:55:55.978750.978750 helper.py:50]   NaN count: 524288
WARNING 05-06 11:55:55.978636.978636 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:55:55.981988.981988 cuda_h.py:27] end init_inputs_tokens cost 9.678 ms
DEBUG 05-06 11:55:55.981004.981004 lmp.py:1507] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:55:55.981022.981022 lmp.py:1513] ---- decode step 0 layer 0 ----
DEBUG 05-06 11:55:55.990824.990824 cuda_h.py:27] end decode_layer cost 9.115 ms
DEBUG 05-06 11:55:55.990079.990079 lmp.py:1513] ---- decode step 0 layer 1 ----
DEBUG 05-06 11:55:55.998018.998018 cuda_h.py:27] end decode_layer cost 7.883 ms
DEBUG 05-06 11:55:55.998796.998796 lmp.py:1513] ---- decode step 0 layer 2 ----
DEBUG 05-06 11:55:56.005862.005862 cuda_h.py:27] end decode_layer cost 6.504 ms
DEBUG 05-06 11:55:56.005473.005473 lmp.py:1513] ---- decode step 0 layer 3 ----
DEBUG 05-06 11:55:56.011856.011856 cuda_h.py:27] end decode_layer cost 6.104 ms
DEBUG 05-06 11:55:56.011932.011932 lmp.py:1513] ---- decode step 0 layer 4 ----
DEBUG 05-06 11:55:56.017994.017994 cuda_h.py:27] end decode_layer cost 5.814 ms
DEBUG 05-06 11:55:56.017890.017890 lmp.py:1513] ---- decode step 0 layer 5 ----
DEBUG 05-06 11:55:56.023870.023870 cuda_h.py:27] end decode_layer cost 6.298 ms
DEBUG 05-06 11:55:56.023673.023673 lmp.py:1513] ---- decode step 0 layer 6 ----
DEBUG 05-06 11:55:56.029176.029176 cuda_h.py:27] end decode_layer cost 5.701 ms
DEBUG 05-06 11:55:56.029118.029118 lmp.py:1513] ---- decode step 0 layer 7 ----
DEBUG 05-06 11:55:56.035118.035118 cuda_h.py:27] end decode_layer cost 5.927 ms
DEBUG 05-06 11:55:56.035028.035028 lmp.py:1513] ---- decode step 0 layer 8 ----
DEBUG 05-06 11:55:56.041949.041949 cuda_h.py:27] end decode_layer cost 5.939 ms
DEBUG 05-06 11:55:56.041083.041083 lmp.py:1513] ---- decode step 0 layer 9 ----
DEBUG 05-06 11:55:56.047108.047108 cuda_h.py:27] end decode_layer cost 5.875 ms
DEBUG 05-06 11:55:56.047323.047323 lmp.py:1513] ---- decode step 0 layer 10 ----
DEBUG 05-06 11:55:56.053036.053036 cuda_h.py:27] end decode_layer cost 6.083 ms
DEBUG 05-06 11:55:56.053363.053363 lmp.py:1513] ---- decode step 0 layer 11 ----
DEBUG 05-06 11:55:56.060446.060446 cuda_h.py:27] end decode_layer cost 6.234 ms
DEBUG 05-06 11:55:56.060727.060727 lmp.py:1513] ---- decode step 0 layer 12 ----
DEBUG 05-06 11:55:56.065438.065438 cuda_h.py:27] end decode_layer cost 5.820 ms
DEBUG 05-06 11:55:56.065573.065573 lmp.py:1513] ---- decode step 0 layer 13 ----
DEBUG 05-06 11:55:56.071330.071330 cuda_h.py:27] end decode_layer cost 5.994 ms
DEBUG 05-06 11:55:56.072372.072372 lmp.py:1513] ---- decode step 0 layer 14 ----
DEBUG 05-06 11:55:56.077314.077314 cuda_h.py:27] end decode_layer cost 5.779 ms
DEBUG 05-06 11:55:56.077972.077972 lmp.py:1513] ---- decode step 0 layer 15 ----
DEBUG 05-06 11:55:56.083025.083025 cuda_h.py:27] end decode_layer cost 5.931 ms
DEBUG 05-06 11:55:56.083305.083305 lmp.py:1513] ---- decode step 0 layer 16 ----
DEBUG 05-06 11:55:56.090985.090985 cuda_h.py:27] end decode_layer cost 6.630 ms
DEBUG 05-06 11:55:56.090623.090623 lmp.py:1513] ---- decode step 0 layer 17 ----
DEBUG 05-06 11:55:56.097174.097174 cuda_h.py:27] end decode_layer cost 6.544 ms
DEBUG 05-06 11:55:56.097170.097170 lmp.py:1513] ---- decode step 0 layer 18 ----
DEBUG 05-06 11:55:56.103782.103782 cuda_h.py:27] end decode_layer cost 5.817 ms
DEBUG 05-06 11:55:56.103247.103247 lmp.py:1513] ---- decode step 0 layer 19 ----
DEBUG 05-06 11:55:56.109661.109661 cuda_h.py:27] end decode_layer cost 6.021 ms
DEBUG 05-06 11:55:56.109749.109749 lmp.py:1513] ---- decode step 0 layer 20 ----
DEBUG 05-06 11:55:56.115694.115694 cuda_h.py:27] end decode_layer cost 6.062 ms
DEBUG 05-06 11:55:56.115067.115067 lmp.py:1513] ---- decode step 0 layer 21 ----
DEBUG 05-06 11:55:56.121994.121994 cuda_h.py:27] end decode_layer cost 5.908 ms
DEBUG 05-06 11:55:56.121844.121844 lmp.py:1513] ---- decode step 0 layer 22 ----
DEBUG 05-06 11:55:56.127474.127474 cuda_h.py:27] end decode_layer cost 5.760 ms
DEBUG 05-06 11:55:56.127370.127370 lmp.py:1513] ---- decode step 0 layer 23 ----
DEBUG 05-06 11:55:56.133849.133849 cuda_h.py:27] end decode_layer cost 6.175 ms
DEBUG 05-06 11:55:56.133030.133030 lmp.py:1513] ---- decode step 0 layer 24 ----
DEBUG 05-06 11:55:56.139865.139865 cuda_h.py:27] end decode_layer cost 5.771 ms
DEBUG 05-06 11:55:56.139907.139907 lmp.py:1513] ---- decode step 0 layer 25 ----
DEBUG 05-06 11:55:56.145212.145212 cuda_h.py:27] end decode_layer cost 5.942 ms
DEBUG 05-06 11:55:56.145109.145109 lmp.py:1513] ---- decode step 0 layer 26 ----
DEBUG 05-06 11:55:56.151235.151235 cuda_h.py:27] end decode_layer cost 5.950 ms
DEBUG 05-06 11:55:56.151608.151608 lmp.py:1513] ---- decode step 0 layer 27 ----
DEBUG 05-06 11:55:56.157669.157669 cuda_h.py:27] end decode_layer cost 5.972 ms
DEBUG 05-06 11:55:56.157427.157427 lmp.py:1513] ---- decode step 0 layer 28 ----
DEBUG 05-06 11:55:56.163683.163683 cuda_h.py:27] end decode_layer cost 5.871 ms
DEBUG 05-06 11:55:56.163771.163771 lmp.py:1513] ---- decode step 0 layer 29 ----
DEBUG 05-06 11:55:56.169811.169811 cuda_h.py:27] end decode_layer cost 6.132 ms
DEBUG 05-06 11:55:56.169523.169523 cuda_h.py:27] end decode_step cost 197.756 ms
INFO 05-06 11:55:56.169478.169478 lmp.py:1561] decode step 0 time: 0.19779706001281738 seconds
WARNING 05-06 11:55:56.169021.169021 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:55:56.169208.169208 helper.py:35]   NaN count (hidden): 5632
WARNING 05-06 11:55:56.170176.170176 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:55:56.170107.170107 helper.py:39]   NaN count (normed): 5632
WARNING 05-06 11:55:56.175135.175135 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:55:56.175173.175173 helper.py:50]   NaN count: 524288
WARNING 05-06 11:55:56.175996.175996 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:55:56.177603.177603 cuda_h.py:27] end init_inputs_tokens cost 7.995 ms
DEBUG 05-06 11:55:56.177353.177353 lmp.py:1507] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:55:56.177500.177500 lmp.py:1513] ---- decode step 1 layer 0 ----
DEBUG 05-06 11:55:56.183251.183251 cuda_h.py:27] end decode_layer cost 5.813 ms
DEBUG 05-06 11:55:56.183253.183253 lmp.py:1513] ---- decode step 1 layer 1 ----
DEBUG 05-06 11:55:56.189641.189641 cuda_h.py:27] end decode_layer cost 5.827 ms
DEBUG 05-06 11:55:56.189252.189252 lmp.py:1513] ---- decode step 1 layer 2 ----
DEBUG 05-06 11:55:56.195732.195732 cuda_h.py:27] end decode_layer cost 5.825 ms
DEBUG 05-06 11:55:56.195675.195675 lmp.py:1513] ---- decode step 1 layer 3 ----
DEBUG 05-06 11:55:56.201621.201621 cuda_h.py:27] end decode_layer cost 5.888 ms
DEBUG 05-06 11:55:56.201801.201801 lmp.py:1513] ---- decode step 1 layer 4 ----
DEBUG 05-06 11:55:56.207828.207828 cuda_h.py:27] end decode_layer cost 5.736 ms
DEBUG 05-06 11:55:56.207532.207532 lmp.py:1513] ---- decode step 1 layer 5 ----
DEBUG 05-06 11:55:56.213641.213641 cuda_h.py:27] end decode_layer cost 6.393 ms
DEBUG 05-06 11:55:56.213683.213683 lmp.py:1513] ---- decode step 1 layer 6 ----
DEBUG 05-06 11:55:56.219320.219320 cuda_h.py:27] end decode_layer cost 5.766 ms
DEBUG 05-06 11:55:56.219216.219216 lmp.py:1513] ---- decode step 1 layer 7 ----
DEBUG 05-06 11:55:56.225513.225513 cuda_h.py:27] end decode_layer cost 5.901 ms
DEBUG 05-06 11:55:56.225455.225455 lmp.py:1513] ---- decode step 1 layer 8 ----
DEBUG 05-06 11:55:56.231947.231947 cuda_h.py:27] end decode_layer cost 5.762 ms
DEBUG 05-06 11:55:56.231843.231843 lmp.py:1513] ---- decode step 1 layer 9 ----
DEBUG 05-06 11:55:56.236345.236345 cuda_h.py:27] end decode_layer cost 5.701 ms
DEBUG 05-06 11:55:56.237526.237526 lmp.py:1513] ---- decode step 1 layer 10 ----
DEBUG 05-06 11:55:56.242789.242789 cuda_h.py:27] end decode_layer cost 5.876 ms
DEBUG 05-06 11:55:56.242924.242924 lmp.py:1513] ---- decode step 1 layer 11 ----
DEBUG 05-06 11:55:56.249465.249465 cuda_h.py:27] end decode_layer cost 6.081 ms
DEBUG 05-06 11:55:56.249461.249461 lmp.py:1513] ---- decode step 1 layer 12 ----
DEBUG 05-06 11:55:56.255709.255709 cuda_h.py:27] end decode_layer cost 6.601 ms
DEBUG 05-06 11:55:56.255850.255850 lmp.py:1513] ---- decode step 1 layer 13 ----
DEBUG 05-06 11:55:56.262514.262514 cuda_h.py:27] end decode_layer cost 6.171 ms
DEBUG 05-06 11:55:56.262271.262271 lmp.py:1513] ---- decode step 1 layer 14 ----
DEBUG 05-06 11:55:56.268413.268413 cuda_h.py:27] end decode_layer cost 6.207 ms
DEBUG 05-06 11:55:56.268594.268594 lmp.py:1513] ---- decode step 1 layer 15 ----
DEBUG 05-06 11:55:56.274246.274246 cuda_h.py:27] end decode_layer cost 6.021 ms
DEBUG 05-06 11:55:56.274858.274858 lmp.py:1513] ---- decode step 1 layer 16 ----
DEBUG 05-06 11:55:56.280223.280223 cuda_h.py:27] end decode_layer cost 5.951 ms
DEBUG 05-06 11:55:56.280503.280503 lmp.py:1513] ---- decode step 1 layer 17 ----
DEBUG 05-06 11:55:56.286149.286149 cuda_h.py:27] end decode_layer cost 6.228 ms
DEBUG 05-06 11:55:56.286475.286475 lmp.py:1513] ---- decode step 1 layer 18 ----
DEBUG 05-06 11:55:56.292134.292134 cuda_h.py:27] end decode_layer cost 6.027 ms
DEBUG 05-06 11:55:56.292653.292653 lmp.py:1513] ---- decode step 1 layer 19 ----
DEBUG 05-06 11:55:56.298399.298399 cuda_h.py:27] end decode_layer cost 6.056 ms
DEBUG 05-06 11:55:56.298772.298772 lmp.py:1513] ---- decode step 1 layer 20 ----
DEBUG 05-06 11:55:56.305539.305539 cuda_h.py:27] end decode_layer cost 6.071 ms
DEBUG 05-06 11:55:56.305627.305627 lmp.py:1513] ---- decode step 1 layer 21 ----
DEBUG 05-06 11:55:56.311504.311504 cuda_h.py:27] end decode_layer cost 6.012 ms
DEBUG 05-06 11:55:56.311738.311738 lmp.py:1513] ---- decode step 1 layer 22 ----
DEBUG 05-06 11:55:56.317030.317030 cuda_h.py:27] end decode_layer cost 5.932 ms
DEBUG 05-06 11:55:56.317687.317687 lmp.py:1513] ---- decode step 1 layer 23 ----
DEBUG 05-06 11:55:56.323334.323334 cuda_h.py:27] end decode_layer cost 6.264 ms
DEBUG 05-06 11:55:56.323515.323515 lmp.py:1513] ---- decode step 1 layer 24 ----
DEBUG 05-06 11:55:56.329566.329566 cuda_h.py:27] end decode_layer cost 5.859 ms
DEBUG 05-06 11:55:56.329654.329654 lmp.py:1513] ---- decode step 1 layer 25 ----
DEBUG 05-06 11:55:56.335517.335517 cuda_h.py:27] end decode_layer cost 5.966 ms
DEBUG 05-06 11:55:56.335082.335082 lmp.py:1513] ---- decode step 1 layer 26 ----
DEBUG 05-06 11:55:56.341891.341891 cuda_h.py:27] end decode_layer cost 5.962 ms
DEBUG 05-06 11:55:56.341218.341218 lmp.py:1513] ---- decode step 1 layer 27 ----
DEBUG 05-06 11:55:56.347091.347091 cuda_h.py:27] end decode_layer cost 5.904 ms
DEBUG 05-06 11:55:56.347272.347272 lmp.py:1513] ---- decode step 1 layer 28 ----
DEBUG 05-06 11:55:56.353557.353557 cuda_h.py:27] end decode_layer cost 5.927 ms
DEBUG 05-06 11:55:56.353738.353738 lmp.py:1513] ---- decode step 1 layer 29 ----
DEBUG 05-06 11:55:56.359911.359911 cuda_h.py:27] end decode_layer cost 6.160 ms
DEBUG 05-06 11:55:56.359431.359431 cuda_h.py:27] end decode_step cost 190.176 ms
INFO 05-06 11:55:56.359532.359532 lmp.py:1561] decode step 1 time: 0.19021844863891602 seconds
WARNING 05-06 11:55:56.360554.360554 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:55:56.360826.360826 helper.py:35]   NaN count (hidden): 5632
WARNING 05-06 11:55:56.360968.360968 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:55:56.360523.360523 helper.py:39]   NaN count (normed): 5632
WARNING 05-06 11:55:56.365012.365012 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:55:56.366103.366103 helper.py:50]   NaN count: 524288
WARNING 05-06 11:55:56.366402.366402 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 11:55:56.366490.366490 helper.py:80] WARNING: Logits have extreme values: min=-788.00, max=1152.00
WARNING 05-06 11:55:56.366322.366322 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 11:55:56.368645.368645 cuda_h.py:27] end init_inputs_tokens cost 8.173 ms
DEBUG 05-06 11:55:56.368594.368594 lmp.py:1507] decode step 2 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:55:56.368510.368510 lmp.py:1513] ---- decode step 2 layer 0 ----
DEBUG 05-06 11:55:56.374214.374214 cuda_h.py:27] end decode_layer cost 5.989 ms
DEBUG 05-06 11:55:56.374064.374064 lmp.py:1513] ---- decode step 2 layer 1 ----
DEBUG 05-06 11:55:56.380417.380417 cuda_h.py:27] end decode_layer cost 5.801 ms
DEBUG 05-06 11:55:56.380459.380459 lmp.py:1513] ---- decode step 2 layer 2 ----
DEBUG 05-06 11:55:56.386476.386476 cuda_h.py:27] end decode_layer cost 5.834 ms
DEBUG 05-06 11:55:56.386041.386041 lmp.py:1513] ---- decode step 2 layer 3 ----
DEBUG 05-06 11:55:56.392588.392588 cuda_h.py:27] end decode_layer cost 6.050 ms
DEBUG 05-06 11:55:56.392484.392484 lmp.py:1513] ---- decode step 2 layer 4 ----
DEBUG 05-06 11:55:56.397007.397007 cuda_h.py:27] end decode_layer cost 5.715 ms
DEBUG 05-06 11:55:56.398188.398188 lmp.py:1513] ---- decode step 2 layer 5 ----
DEBUG 05-06 11:55:56.404963.404963 cuda_h.py:27] end decode_layer cost 6.534 ms
DEBUG 05-06 11:55:56.404767.404767 lmp.py:1513] ---- decode step 2 layer 6 ----
DEBUG 05-06 11:55:56.410832.410832 cuda_h.py:27] end decode_layer cost 5.905 ms
DEBUG 05-06 11:55:56.410444.410444 lmp.py:1513] ---- decode step 2 layer 7 ----
DEBUG 05-06 11:55:56.416468.416468 cuda_h.py:27] end decode_layer cost 5.875 ms
DEBUG 05-06 11:55:56.416080.416080 lmp.py:1513] ---- decode step 2 layer 8 ----
DEBUG 05-06 11:55:56.422234.422234 cuda_h.py:27] end decode_layer cost 5.795 ms
DEBUG 05-06 11:55:56.422223.422223 lmp.py:1513] ---- decode step 2 layer 9 ----
DEBUG 05-06 11:55:56.428383.428383 cuda_h.py:27] end decode_layer cost 5.939 ms
DEBUG 05-06 11:55:56.428994.428994 lmp.py:1513] ---- decode step 2 layer 10 ----
DEBUG 05-06 11:55:56.434679.434679 cuda_h.py:27] end decode_layer cost 5.800 ms
DEBUG 05-06 11:55:56.434383.434383 lmp.py:1513] ---- decode step 2 layer 11 ----
DEBUG 05-06 11:55:56.440040.440040 cuda_h.py:27] end decode_layer cost 6.166 ms
DEBUG 05-06 11:55:56.440605.440605 lmp.py:1513] ---- decode step 2 layer 12 ----
DEBUG 05-06 11:55:56.446034.446034 cuda_h.py:27] end decode_layer cost 5.893 ms
DEBUG 05-06 11:55:56.446599.446599 lmp.py:1513] ---- decode step 2 layer 13 ----
DEBUG 05-06 11:55:56.452415.452415 cuda_h.py:27] end decode_layer cost 5.967 ms
DEBUG 05-06 11:55:56.452550.452550 lmp.py:1513] ---- decode step 2 layer 14 ----
DEBUG 05-06 11:55:56.458453.458453 cuda_h.py:27] end decode_layer cost 5.822 ms
DEBUG 05-06 11:55:56.458158.458158 lmp.py:1513] ---- decode step 2 layer 15 ----
DEBUG 05-06 11:55:56.464227.464227 cuda_h.py:27] end decode_layer cost 5.838 ms
DEBUG 05-06 11:55:56.464932.464932 lmp.py:1513] ---- decode step 2 layer 16 ----
DEBUG 05-06 11:55:56.470473.470473 cuda_h.py:27] end decode_layer cost 5.694 ms
DEBUG 05-06 11:55:56.470562.470562 lmp.py:1513] ---- decode step 2 layer 17 ----
DEBUG 05-06 11:55:56.476585.476585 cuda_h.py:27] end decode_layer cost 6.225 ms
DEBUG 05-06 11:55:56.476196.476196 lmp.py:1513] ---- decode step 2 layer 18 ----
DEBUG 05-06 11:55:56.482213.482213 cuda_h.py:27] end decode_layer cost 5.835 ms
DEBUG 05-06 11:55:56.482679.482679 lmp.py:1513] ---- decode step 2 layer 19 ----
DEBUG 05-06 11:55:56.488791.488791 cuda_h.py:27] end decode_layer cost 5.729 ms
DEBUG 05-06 11:55:56.488595.488595 lmp.py:1513] ---- decode step 2 layer 20 ----
DEBUG 05-06 11:55:56.494568.494568 cuda_h.py:27] end decode_layer cost 5.907 ms
DEBUG 05-06 11:55:56.494702.494702 lmp.py:1513] ---- decode step 2 layer 21 ----
DEBUG 05-06 11:55:56.500326.500326 cuda_h.py:27] end decode_layer cost 5.955 ms
DEBUG 05-06 11:55:56.500653.500653 lmp.py:1513] ---- decode step 2 layer 22 ----
DEBUG 05-06 11:55:56.505563.505563 cuda_h.py:27] end decode_layer cost 5.827 ms
DEBUG 05-06 11:55:56.506983.506983 lmp.py:1513] ---- decode step 2 layer 23 ----
DEBUG 05-06 11:55:56.512629.512629 cuda_h.py:27] end decode_layer cost 6.228 ms
DEBUG 05-06 11:55:56.512194.512194 lmp.py:1513] ---- decode step 2 layer 24 ----
DEBUG 05-06 11:55:56.518591.518591 cuda_h.py:27] end decode_layer cost 5.729 ms
DEBUG 05-06 11:55:56.518494.518494 lmp.py:1513] ---- decode step 2 layer 25 ----
DEBUG 05-06 11:55:56.524899.524899 cuda_h.py:27] end decode_layer cost 5.945 ms
DEBUG 05-06 11:55:56.524987.524987 lmp.py:1513] ---- decode step 2 layer 26 ----
DEBUG 05-06 11:55:56.530033.530033 cuda_h.py:27] end decode_layer cost 5.926 ms
DEBUG 05-06 11:55:56.530698.530698 lmp.py:1513] ---- decode step 2 layer 27 ----
DEBUG 05-06 11:55:56.536725.536725 cuda_h.py:27] end decode_layer cost 5.947 ms
DEBUG 05-06 11:55:56.536244.536244 lmp.py:1513] ---- decode step 2 layer 28 ----
DEBUG 05-06 11:55:56.542883.542883 cuda_h.py:27] end decode_layer cost 5.837 ms
DEBUG 05-06 11:55:56.542587.542587 lmp.py:1513] ---- decode step 2 layer 29 ----
DEBUG 05-06 11:55:56.548544.548544 cuda_h.py:27] end decode_layer cost 6.036 ms
DEBUG 05-06 11:55:56.548627.548627 cuda_h.py:27] end decode_step cost 188.220 ms
INFO 05-06 11:55:56.548913.548913 lmp.py:1561] decode step 2 time: 0.18825960159301758 seconds
Time taken: 5.5208806954324245 seconds
X512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x62d912f5a080, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
CPUInfer[0x62d912f33e40]: Goodbye
CPUInfer[0x62d8fba272b0]: Goodbye
