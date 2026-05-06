here pin
INFO 05-06 10:00:16.148775.148775 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-06 10:00:16.710510.710510 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-06 10:00:17.153705.153705 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-06 10:00:17.153203.153203 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 1.005s
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
INFO 05-06 10:00:24.772827.772827 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-06 10:00:25.209540.209540 cuda_h.py:27] end init_cmv_hmv cost 436.835 ms
DEBUG 05-06 10:00:25.219733.219733 cuda_memory_view.py:1366] 
DEBUG 05-06 10:00:25.219733.219733 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.004125118255615234
DEBUG 05-06 10:00:25.238546.238546 mlpmodule.py:993] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-06 10:00:25.239519.239519 cuda_memory_view.py:1370] 
DEBUG 05-06 10:00:25.239519.239519 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.0193479061126709
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-06 10:00:27.154918.154918 lmp.py:255] init kt-kernel layer 0 ok
INFO 05-06 10:00:27.930659.930659 lmp.py:255] init kt-kernel layer 1 ok
INFO 05-06 10:00:28.746315.746315 lmp.py:255] init kt-kernel layer 2 ok
INFO 05-06 10:00:29.551876.551876 lmp.py:255] init kt-kernel layer 3 ok
INFO 05-06 10:00:30.378858.378858 lmp.py:255] init kt-kernel layer 4 ok
INFO 05-06 10:00:31.199686.199686 lmp.py:255] init kt-kernel layer 5 ok
INFO 05-06 10:00:32.023357.023357 lmp.py:255] init kt-kernel layer 6 ok
INFO 05-06 10:00:32.852241.852241 lmp.py:255] init kt-kernel layer 7 ok
INFO 05-06 10:00:33.678958.678958 lmp.py:255] init kt-kernel layer 8 ok
INFO 05-06 10:00:34.499635.499635 lmp.py:255] init kt-kernel layer 9 ok
INFO 05-06 10:00:35.350418.350418 lmp.py:255] init kt-kernel layer 10 ok
INFO 05-06 10:00:36.179452.179452 lmp.py:255] init kt-kernel layer 11 ok
INFO 05-06 10:00:37.002416.002416 lmp.py:255] init kt-kernel layer 12 ok
INFO 05-06 10:00:37.840450.840450 lmp.py:255] init kt-kernel layer 13 ok
INFO 05-06 10:00:38.688270.688270 lmp.py:255] init kt-kernel layer 14 ok
INFO 05-06 10:00:39.528671.528671 lmp.py:255] init kt-kernel layer 15 ok
INFO 05-06 10:00:40.365180.365180 lmp.py:255] init kt-kernel layer 16 ok
INFO 05-06 10:00:41.181979.181979 lmp.py:255] init kt-kernel layer 17 ok
INFO 05-06 10:00:42.012637.012637 lmp.py:255] init kt-kernel layer 18 ok
INFO 05-06 10:00:42.830764.830764 lmp.py:255] init kt-kernel layer 19 ok
INFO 05-06 10:00:43.644743.644743 lmp.py:255] init kt-kernel layer 20 ok
INFO 05-06 10:00:44.465830.465830 lmp.py:255] init kt-kernel layer 21 ok
INFO 05-06 10:00:45.274480.274480 lmp.py:255] init kt-kernel layer 22 ok
CPUInfer[0x5cb0b98f1c80]: Hello
WorkerPool[0x5cb0b98ee150] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x5cb0d4536a20]: Hello
WorkerPool[0x5cb0d39ff940] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVINFO 05-06 10:00:46.182211.182211 lmp.py:255] init kt-kernel layer 23 ok
INFO 05-06 10:00:47.034784.034784 lmp.py:255] init kt-kernel layer 24 ok
INFO 05-06 10:00:47.869789.869789 lmp.py:255] init kt-kernel layer 25 ok
INFO 05-06 10:00:48.702430.702430 lmp.py:255] init kt-kernel layer 26 ok
INFO 05-06 10:00:49.529621.529621 lmp.py:255] init kt-kernel layer 27 ok
INFO 05-06 10:00:50.329249.329249 lmp.py:255] init kt-kernel layer 28 ok
INFO 05-06 10:00:51.134304.134304 lmp.py:255] init kt-kernel layer 29 ok
INFO 05-06 10:00:52.020324.020324 lmp.py:186] vLLM Triton fused-MoE enabled (CUDAGraph=False).
generate input ids cost 0.05403566360473633 s
DEBUG 05-06 10:00:54.910676.910676 cuda_h.py:27] end generate_input_ids cost 2870.469 ms
DEBUG 05-06 10:00:54.910470.910470 cuda_h.py:27] end init_cache cost 0.050 ms
INFO 05-06 10:00:54.919671.919671 lmp.py:367] _ensure_static_kv_cache (Gemma4 list): 30 layers, 1760.0 MiB on cuda:0
INFO 05-06 10:00:54.920258.920258 lmp.py:1158] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 10:00:54.933582.933582 lmp.py:2782] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 4784365508, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.787078263565146, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-06 10:00:54.933757.933757 lmp.py:2800] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.933010.933010 lmp.py:2800] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.933111.933111 lmp.py:2800] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.933543.933543 lmp.py:2800] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.933133.933133 lmp.py:2800] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.934962.934962 lmp.py:2800] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.934339.934339 lmp.py:2800] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.934961.934961 lmp.py:2800] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.934346.934346 lmp.py:2800] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.934040.934040 lmp.py:2800] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.934233.934233 lmp.py:2800] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.934246.934246 lmp.py:2800] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.935963.935963 lmp.py:2800] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.935068.935068 lmp.py:2800] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.935500.935500 lmp.py:2800] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.935758.935758 lmp.py:2800] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.935236.935236 lmp.py:2800] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.935268.935268 lmp.py:2800] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.935204.935204 lmp.py:2800] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.935575.935575 lmp.py:2800] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.935059.935059 lmp.py:2800] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.936624.936624 lmp.py:2800] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.936917.936917 lmp.py:2800] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.936596.936596 lmp.py:2800] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.936604.936604 lmp.py:2800] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.936336.936336 lmp.py:2800] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.936868.936868 lmp.py:2800] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.936179.936179 lmp.py:2800] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.936471.936471 lmp.py:2800] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:00:54.936390.936390 lmp.py:2800] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 10:00:55.222690.222690 cuda_h.py:27] end init_loading_placement cost 302.785 ms
DEBUG 05-06 10:00:55.223237.223237 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:00:55.223239.223239 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:00:55 client.py:72] load_into_gpu: gemma4-26B-A4B, 085f83ca-317d-4644-9fcb-75e073a594ae
INFO 05-06 10:00:55 client.py:135] Model loaded: gemma4-26B-A4B, 085f83ca-317d-4644-9fcb-75e073a594ae
INFO 05-06 10:00:55 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 085f83ca-317d-4644-9fcb-75e073a594ae
INFO 05-06 10:00:55 client.py:212] Model loaded
DEBUG 05-06 10:00:55.751952.751952 cuda_h.py:27] end init_general_sagl_loading_async cost 528.673 ms
INFO 05-06 10:00:55.799061.799061 lmp.py:3303] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 10:00:55.800185.800185 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:00:55.800511.800511 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:00:55 client.py:72] load_into_gpu: gemma4-26B-A4B, 1719c47c-df73-452c-95ff-59b50d3fec06
INFO 05-06 10:00:55 client.py:135] Model loaded: gemma4-26B-A4B, 1719c47c-df73-452c-95ff-59b50d3fec06
DEBUG 05-06 10:00:55.872252.872252 cuda_h.py:27] end init_experts_loading_async cost 72.613 ms
DEBUG 05-06 10:00:55.985020.985020 cuda_h.py:27] end restore_state_dict cost 112.956 ms
WARNING 05-06 10:00:56 [fused_moe.py:1090] Using default MoE config. Performance might be sub-optimal! Config file not found at /mnt/zhengcf3/lmp_env/fslmp/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=32,N=704,device_name=NVIDIA_GeForce_RTX_4090.json
INFO 05-06 10:00:57.080408.080408 lmp.py:1299] vLLM Triton pre-warmup done in 1094.1 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 10:00:57.100995.100995 cuda_h.py:27] end init_inputs_tokens cost 20.321 ms
DEBUG 05-06 10:00:57.101920.101920 lmp.py:1346] -------------------------------- start prefill layer 0 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 101, 109, 85, 49, 45, 65, 93, 69, 5, 13, 9, 73, 77, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18], 'token_total': 420, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 13: 16, 9: 17, 73: 19, 77: 19, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 51, 55, 67, 75, 83, 91, 103, 115, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1216, 'token_per_expert': {3: 46, 7: 95, 31: 34, 39: 176, 47: 318, 51: 48, 55: 51, 67: 47, 75: 29, 83: 33, 91: 99, 103: 178, 115: 29, 127: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 32, 48, 52, 64, 68, 72, 92, 104, 112, 116, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 758, 'token_per_expert': {0: 73, 16: 48, 32: 43, 48: 41, 52: 43, 64: 27, 68: 170, 72: 35, 92: 16, 104: 43, 112: 23, 116: 18, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 25, 33, 37, 41, 53, 89, 105, 113, 117, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 809, 'token_per_expert': {1: 75, 21: 48, 25: 24, 33: 210, 37: 20, 41: 27, 53: 205, 89: 20, 105: 24, 113: 39, 117: 26, 121: 65, 125: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 46, 50, 54, 70, 74, 78, 90, 110, 118, 122, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 893, 'token_per_expert': {22: 64, 26: 59, 46: 119, 50: 110, 54: 59, 70: 25, 74: 61, 78: 36, 90: 154, 110: 27, 118: 29, 122: 35, 126: 115}}
INFO 05-06 10:00:57.341606.341606 lmp.py:1833] [layer_moe_fused] layer=0 prefix: 21.565ms alloc: 0.293ms
INFO 05-06 10:00:57.342989.342989 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 4.38690185546875e-05 seconds
INFO 05-06 10:00:57.344724.344724 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0023064613342285156s
INFO 05-06 10:00:57.345533.345533 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0012516975402832031 seconds
INFO 05-06 10:00:57.351338.351338 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0026793479919433594s
INFO 05-06 10:00:57.399517.399517 lmp.py:1938] [layer_moe_fused] vllm triton time: 48.690ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.400234.400234 cuda_h.py:27] end *layer_moe_fused cost 80.208 ms
DEBUG 05-06 10:00:57.401937.401937 cuda_h.py:27] end prefill_layer cost 300.361 ms
DEBUG 05-06 10:00:57.401616.401616 lmp.py:1388] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 10:00:57.401059.401059 lmp.py:1346] -------------------------------- start prefill layer 1 --------------------------------
experts_cpu_alloc {'expert_ids': [39, 63, 15, 55, 115, 31, 103, 43, 83, 123, 87, 16, 88, 40, 44, 60, 72, 84, 0, 112, 108, 32, 48, 56, 116, 4, 76, 61, 77, 117, 33, 41, 45, 81, 57, 89, 125, 29, 37, 121, 69, 93, 101, 6, 70, 86, 2, 18, 26, 14, 38, 62, 110, 66, 78, 74, 50], 'token_total': 355, 'token_per_expert': {39: 1, 63: 1, 15: 4, 55: 4, 115: 4, 31: 5, 103: 5, 43: 6, 83: 6, 123: 6, 87: 7, 16: 2, 88: 3, 40: 4, 44: 4, 60: 5, 72: 5, 84: 5, 0: 7, 112: 7, 108: 8, 32: 9, 48: 10, 56: 10, 116: 10, 4: 15, 76: 17, 61: 1, 77: 1, 117: 2, 33: 3, 41: 3, 45: 4, 81: 4, 57: 5, 89: 5, 125: 6, 29: 7, 37: 9, 121: 10, 69: 11, 93: 11, 101: 13, 6: 1, 70: 1, 86: 1, 2: 2, 18: 3, 26: 4, 14: 5, 38: 6, 62: 6, 110: 6, 66: 10, 78: 12, 74: 15, 50: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 35, 47, 51, 59, 67, 79, 91, 95, 99, 119, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 26, 'token_total': 642, 'token_per_expert': {3: 26, 7: 39, 11: 10, 27: 14, 35: 24, 47: 47, 51: 61, 59: 35, 67: 134, 79: 26, 91: 8, 95: 22, 99: 122, 119: 24, 127: 50}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 20, 28, 52, 64, 68, 80, 92, 96, 100, 104, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 1019, 'token_per_expert': {8: 116, 12: 56, 20: 61, 28: 66, 52: 269, 64: 25, 68: 175, 80: 48, 92: 26, 96: 38, 100: 55, 104: 21, 120: 32, 124: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 49, 53, 65, 73, 85, 97, 105, 109], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 898, 'token_per_expert': {1: 31, 5: 104, 9: 15, 13: 275, 21: 19, 25: 47, 49: 33, 53: 43, 65: 36, 73: 30, 85: 15, 97: 122, 105: 25, 109: 103}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 22, 30, 34, 42, 46, 54, 82, 90, 94, 98, 106, 118, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1182, 'token_per_expert': {10: 166, 22: 126, 30: 248, 34: 29, 42: 37, 46: 42, 54: 59, 82: 177, 90: 22, 94: 29, 98: 21, 106: 39, 118: 61, 122: 126}}
INFO 05-06 10:00:57.408585.408585 lmp.py:1833] [layer_moe_fused] layer=1 prefix: 0.486ms alloc: 0.424ms
INFO 05-06 10:00:57.409027.409027 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.008148193359375e-05 seconds
INFO 05-06 10:00:57.410133.410133 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009264945983886719s
INFO 05-06 10:00:57.410886.410886 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006642341613769531 seconds
INFO 05-06 10:00:57.413683.413683 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0022211074829101562s
INFO 05-06 10:00:57.421168.421168 lmp.py:1938] [layer_moe_fused] vllm triton time: 8.032ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.421071.421071 cuda_h.py:27] end *layer_moe_fused cost 13.838 ms
DEBUG 05-06 10:00:57.422792.422792 cuda_h.py:27] end prefill_layer cost 20.898 ms
DEBUG 05-06 10:00:57.422033.422033 lmp.py:1388] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 10:00:57.422602.422602 lmp.py:1346] -------------------------------- start prefill layer 2 --------------------------------
experts_cpu_alloc {'expert_ids': [67, 75, 79, 87, 111, 115, 27, 99, 95, 35, 71, 23, 43, 63, 107, 31, 12, 40, 116, 120, 96, 0, 64, 88, 24, 44, 52, 72, 100, 25, 45, 21, 5, 121, 61, 113, 105, 17, 77, 85, 57, 69, 22, 66, 86, 26, 50, 6, 114, 42, 82, 58, 46, 126, 70], 'token_total': 590, 'token_per_expert': {67: 1, 75: 1, 79: 1, 87: 1, 111: 5, 115: 5, 27: 6, 99: 6, 95: 11, 35: 17, 71: 17, 23: 19, 43: 19, 63: 20, 107: 21, 31: 22, 12: 2, 40: 5, 116: 5, 120: 5, 96: 8, 0: 9, 64: 12, 88: 15, 24: 16, 44: 17, 52: 17, 72: 17, 100: 18, 25: 3, 45: 4, 21: 5, 5: 6, 121: 8, 61: 11, 113: 11, 105: 16, 17: 18, 77: 18, 85: 18, 57: 19, 69: 20, 22: 1, 66: 1, 86: 1, 26: 4, 50: 4, 6: 5, 114: 7, 42: 9, 82: 14, 58: 15, 46: 17, 126: 18, 70: 19}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 51, 55, 59, 83, 91, 103, 119, 123, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 991, 'token_per_expert': {3: 24, 7: 69, 11: 202, 15: 95, 19: 116, 51: 39, 55: 65, 59: 117, 83: 22, 91: 36, 103: 28, 119: 24, 123: 24, 127: 130}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 20, 28, 36, 48, 56, 60, 76, 80, 84, 104, 108, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 771, 'token_per_expert': {4: 25, 8: 27, 20: 53, 28: 20, 36: 21, 48: 66, 56: 20, 60: 50, 76: 66, 80: 66, 84: 62, 104: 48, 108: 216, 124: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 29, 33, 37, 41, 49, 53, 65, 81, 97, 109, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 987, 'token_per_expert': {1: 99, 9: 115, 13: 105, 29: 89, 33: 26, 37: 74, 41: 145, 49: 25, 53: 39, 65: 42, 81: 62, 97: 39, 109: 37, 125: 90}}
experts_gpu_alloc_device_3 {'expert_ids': [14, 18, 34, 54, 62, 78, 90, 98, 102, 106, 110, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 757, 'token_per_expert': {14: 33, 18: 62, 34: 34, 54: 121, 62: 145, 78: 32, 90: 68, 98: 21, 102: 88, 106: 39, 110: 30, 118: 59, 122: 25}}
INFO 05-06 10:00:57.428867.428867 lmp.py:1833] [layer_moe_fused] layer=2 prefix: 0.470ms alloc: 0.409ms
INFO 05-06 10:00:57.428978.428978 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.841255187988281e-05 seconds
INFO 05-06 10:00:57.429376.429376 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008416175842285156s
INFO 05-06 10:00:57.430234.430234 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006380081176757812 seconds
INFO 05-06 10:00:57.431231.431231 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011935234069824219s
INFO 05-06 10:00:57.454225.454225 lmp.py:1938] [layer_moe_fused] vllm triton time: 23.042ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.454134.454134 cuda_h.py:27] end *layer_moe_fused cost 27.447 ms
DEBUG 05-06 10:00:57.455922.455922 cuda_h.py:27] end prefill_layer cost 32.708 ms
DEBUG 05-06 10:00:57.455660.455660 lmp.py:1388] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 10:00:57.455204.455204 lmp.py:1346] -------------------------------- start prefill layer 3 --------------------------------
experts_cpu_alloc {'expert_ids': [23, 35, 55, 87, 91, 127, 27, 63, 31, 43, 67, 119, 111, 20, 72, 32, 36, 16, 100, 60, 8, 40, 116, 24, 48, 56, 64, 44, 65, 29, 57, 89, 117, 13, 41, 33, 101, 61, 77, 94, 18, 82, 98, 30, 42, 86, 74, 26, 110, 114, 58, 70, 54], 'token_total': 345, 'token_per_expert': {23: 1, 35: 1, 55: 1, 87: 1, 91: 1, 127: 2, 27: 3, 63: 3, 31: 5, 43: 9, 67: 9, 119: 9, 111: 10, 20: 1, 72: 1, 32: 2, 36: 2, 16: 3, 100: 4, 60: 5, 8: 6, 40: 7, 116: 7, 24: 8, 48: 8, 56: 8, 64: 8, 44: 19, 65: 1, 29: 2, 57: 2, 89: 2, 117: 3, 13: 4, 41: 4, 33: 6, 101: 9, 61: 11, 77: 13, 94: 1, 18: 3, 82: 3, 98: 3, 30: 4, 42: 7, 86: 8, 74: 9, 26: 14, 110: 16, 114: 17, 58: 18, 70: 20, 54: 21}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 39, 51, 59, 71, 75, 83, 95, 107, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 842, 'token_per_expert': {3: 270, 7: 256, 11: 28, 15: 23, 19: 15, 39: 17, 51: 24, 59: 11, 71: 42, 75: 60, 83: 35, 95: 32, 107: 13, 123: 16}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 52, 68, 76, 84, 88, 92, 96, 104, 108, 120], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 927, 'token_per_expert': {0: 275, 4: 282, 28: 55, 52: 38, 68: 33, 76: 26, 84: 44, 88: 23, 92: 36, 96: 30, 104: 29, 108: 25, 120: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 25, 53, 69, 73, 85, 93, 97, 109, 121], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 910, 'token_per_expert': {1: 262, 5: 293, 9: 18, 17: 28, 25: 27, 53: 31, 69: 34, 73: 25, 85: 64, 93: 35, 97: 34, 109: 16, 121: 43}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 34, 50, 62, 66, 78, 102, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 1072, 'token_per_expert': {2: 269, 6: 265, 10: 35, 14: 43, 22: 70, 34: 36, 50: 87, 62: 59, 66: 53, 78: 40, 102: 61, 118: 28, 122: 26}}
INFO 05-06 10:00:57.461968.461968 lmp.py:1833] [layer_moe_fused] layer=3 prefix: 0.463ms alloc: 0.393ms
INFO 05-06 10:00:57.462788.462788 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.151199340820312e-05 seconds
INFO 05-06 10:00:57.463438.463438 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009229183197021484s
INFO 05-06 10:00:57.463542.463542 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006337165832519531 seconds
INFO 05-06 10:00:57.465251.465251 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017673969268798828s
INFO 05-06 10:00:57.473071.473071 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.767ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.474834.474834 cuda_h.py:27] end *layer_moe_fused cost 13.083 ms
DEBUG 05-06 10:00:57.474478.474478 cuda_h.py:27] end prefill_layer cost 19.050 ms
DEBUG 05-06 10:00:57.475619.475619 lmp.py:1388] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 10:00:57.475352.475352 lmp.py:1346] -------------------------------- start prefill layer 4 --------------------------------
experts_cpu_alloc {'expert_ids': [79, 103, 107, 19, 75, 15, 91, 123, 47, 87, 71, 39, 12, 56, 120, 44, 80, 64, 40, 108, 84, 36, 88, 52, 116, 41, 121, 69, 21, 45, 97, 109, 37, 101, 73, 81, 77, 117, 17, 61, 58, 66, 70, 18, 46, 114, 94, 122, 126, 30, 34, 90, 118], 'token_total': 359, 'token_per_expert': {79: 1, 103: 2, 107: 9, 19: 10, 75: 12, 15: 13, 91: 13, 123: 13, 47: 17, 87: 17, 71: 18, 39: 19, 12: 1, 56: 4, 120: 4, 44: 5, 80: 5, 64: 6, 40: 7, 108: 8, 84: 9, 36: 10, 88: 10, 52: 11, 116: 13, 41: 1, 121: 1, 69: 2, 21: 4, 45: 4, 97: 4, 109: 4, 37: 5, 101: 5, 73: 6, 81: 6, 77: 7, 117: 7, 17: 8, 61: 8, 58: 1, 66: 1, 70: 1, 18: 2, 46: 2, 114: 2, 94: 4, 122: 4, 126: 5, 30: 6, 34: 6, 90: 8, 118: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 43, 51, 55, 59, 63, 67, 83, 111, 115, 119], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 1238, 'token_per_expert': {3: 272, 7: 256, 23: 67, 27: 44, 43: 75, 51: 30, 55: 27, 59: 72, 63: 138, 67: 35, 83: 36, 111: 56, 115: 42, 119: 88}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 24, 28, 32, 60, 76, 92, 96, 104, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 859, 'token_per_expert': {0: 256, 4: 269, 8: 90, 20: 20, 24: 63, 28: 16, 32: 19, 60: 26, 76: 23, 92: 18, 96: 15, 104: 24, 124: 20}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 25, 29, 49, 53, 57, 85, 89, 93, 105, 113, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 838, 'token_per_expert': {1: 304, 5: 270, 25: 9, 29: 32, 49: 22, 53: 22, 57: 9, 85: 26, 89: 50, 93: 27, 105: 15, 113: 26, 125: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 38, 54, 62, 74, 78, 82, 86, 98, 106], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 802, 'token_per_expert': {2: 256, 6: 260, 22: 32, 26: 32, 38: 9, 54: 32, 62: 13, 74: 57, 78: 13, 82: 22, 86: 10, 98: 13, 106: 53}}
INFO 05-06 10:00:57.480689.480689 lmp.py:1833] [layer_moe_fused] layer=4 prefix: 0.453ms alloc: 0.394ms
INFO 05-06 10:00:57.480502.480502 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-06 10:00:57.481272.481272 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008325576782226562s
INFO 05-06 10:00:57.482009.482009 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005834102630615234 seconds
INFO 05-06 10:00:57.484546.484546 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017583370208740234s
INFO 05-06 10:00:57.492434.492434 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.644ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.492397.492397 cuda_h.py:27] end *layer_moe_fused cost 12.888 ms
DEBUG 05-06 10:00:57.493073.493073 cuda_h.py:27] end prefill_layer cost 18.150 ms
DEBUG 05-06 10:00:57.493645.493645 lmp.py:1388] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 10:00:57.493827.493827 lmp.py:1346] -------------------------------- start prefill layer 5 --------------------------------
experts_cpu_alloc {'expert_ids': [19, 27, 51, 67, 115, 107, 119, 75, 79, 83, 31, 8, 32, 92, 124, 56, 44, 68, 52, 84, 80, 96, 100, 120, 60, 104, 17, 21, 121, 77, 81, 105, 37, 57, 53, 113, 30, 38, 78, 82, 58, 26, 50, 54, 102, 34, 86, 114, 62, 98, 106], 'token_total': 269, 'token_per_expert': {19: 2, 27: 2, 51: 2, 67: 2, 115: 3, 107: 7, 119: 7, 75: 8, 79: 8, 83: 8, 31: 9, 8: 1, 32: 1, 92: 1, 124: 1, 56: 4, 44: 5, 68: 6, 52: 7, 84: 7, 80: 15, 96: 15, 100: 16, 120: 16, 60: 21, 104: 22, 17: 1, 21: 1, 121: 1, 77: 2, 81: 2, 105: 3, 37: 4, 57: 4, 53: 5, 113: 5, 30: 1, 38: 1, 78: 1, 82: 1, 58: 2, 26: 3, 50: 3, 54: 3, 102: 3, 34: 4, 86: 4, 114: 4, 62: 5, 98: 5, 106: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 43, 55, 63, 71, 87, 99, 111, 123, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 972, 'token_per_expert': {3: 256, 7: 265, 23: 22, 39: 83, 43: 21, 55: 13, 63: 15, 71: 131, 87: 32, 99: 23, 111: 31, 123: 29, 127: 51}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 28, 36, 64, 72, 76, 88, 112, 116], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 1051, 'token_per_expert': {0: 265, 4: 295, 16: 65, 20: 74, 24: 39, 28: 41, 36: 47, 64: 36, 72: 30, 76: 30, 88: 24, 112: 82, 116: 23}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 29, 33, 49, 61, 73, 93, 101, 117, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 23, 'token_total': 1045, 'token_per_expert': {1: 257, 5: 280, 9: 37, 13: 37, 29: 16, 33: 65, 49: 101, 61: 24, 73: 21, 93: 28, 101: 133, 117: 32, 125: 14}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 42, 46, 70, 74, 94, 118, 126], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 27, 'token_total': 759, 'token_per_expert': {2: 308, 6: 259, 14: 10, 18: 7, 22: 24, 42: 31, 46: 18, 70: 22, 74: 19, 94: 27, 118: 12, 126: 22}}
INFO 05-06 10:00:57.501154.501154 lmp.py:1833] [layer_moe_fused] layer=5 prefix: 0.459ms alloc: 0.388ms
INFO 05-06 10:00:57.501682.501682 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.008148193359375e-05 seconds
INFO 05-06 10:00:57.503572.503572 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008165836334228516s
INFO 05-06 10:00:57.503986.503986 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006265640258789062 seconds
INFO 05-06 10:00:57.505463.505463 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001661539077758789s
INFO 05-06 10:00:57.513476.513476 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.595ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.513498.513498 cuda_h.py:27] end *layer_moe_fused cost 12.900 ms
DEBUG 05-06 10:00:57.514378.514378 cuda_h.py:27] end prefill_layer cost 20.800 ms
DEBUG 05-06 10:00:57.514235.514235 lmp.py:1388] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 10:00:57.515889.515889 lmp.py:1346] -------------------------------- start prefill layer 6 --------------------------------
experts_cpu_alloc {'expert_ids': [31, 83, 47, 59, 67, 15, 111, 91, 11, 127, 43, 103, 51, 71, 52, 72, 92, 112, 124, 16, 20, 40, 120, 60, 76, 80, 21, 109, 49, 81, 97, 101, 73, 89, 125, 37, 57, 85, 105, 22, 82, 114, 38, 110, 126, 42, 18, 30, 74, 14, 10, 58, 70, 78, 46], 'token_total': 282, 'token_per_expert': {31: 1, 83: 1, 47: 2, 59: 2, 67: 2, 15: 3, 111: 3, 91: 4, 11: 5, 127: 6, 43: 9, 103: 10, 51: 11, 71: 11, 52: 1, 72: 1, 92: 1, 112: 1, 124: 1, 16: 2, 20: 2, 40: 2, 120: 2, 60: 5, 76: 5, 80: 5, 21: 1, 109: 1, 49: 2, 81: 2, 97: 2, 101: 2, 73: 4, 89: 5, 125: 8, 37: 9, 57: 9, 85: 9, 105: 9, 22: 1, 82: 1, 114: 1, 38: 3, 110: 4, 126: 4, 42: 5, 18: 6, 30: 6, 74: 7, 14: 9, 10: 10, 58: 11, 70: 13, 78: 18, 46: 22}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 35, 75, 79, 87, 95, 99, 107, 115, 119, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 960, 'token_per_expert': {3: 257, 7: 257, 23: 49, 27: 14, 35: 45, 75: 11, 79: 20, 87: 44, 95: 13, 99: 134, 107: 17, 115: 49, 119: 35, 123: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 28, 32, 36, 44, 56, 64, 68, 96, 104, 108, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 919, 'token_per_expert': {0: 263, 4: 258, 24: 15, 28: 13, 32: 17, 36: 22, 44: 15, 56: 11, 64: 47, 68: 145, 96: 21, 104: 15, 108: 68, 116: 9}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 41, 53, 65, 69, 77, 93, 113, 117, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 993, 'token_per_expert': {1: 272, 5: 267, 9: 24, 13: 37, 25: 113, 41: 10, 53: 45, 65: 51, 69: 18, 77: 14, 93: 82, 113: 13, 117: 22, 121: 25}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 34, 50, 62, 86, 90, 94, 98, 102, 106, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 942, 'token_per_expert': {2: 269, 6: 267, 26: 35, 34: 36, 50: 22, 62: 23, 86: 54, 90: 44, 94: 32, 98: 33, 102: 37, 106: 65, 122: 25}}
INFO 05-06 10:00:57.520388.520388 lmp.py:1833] [layer_moe_fused] layer=6 prefix: 0.453ms alloc: 0.404ms
INFO 05-06 10:00:57.520969.520969 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.841255187988281e-05 seconds
INFO 05-06 10:00:57.521357.521357 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007946491241455078s
INFO 05-06 10:00:57.522340.522340 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006256103515625 seconds
INFO 05-06 10:00:57.524550.524550 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017728805541992188s
INFO 05-06 10:00:57.532975.532975 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.825ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.532527.532527 cuda_h.py:27] end *layer_moe_fused cost 13.110 ms
DEBUG 05-06 10:00:57.533641.533641 cuda_h.py:27] end prefill_layer cost 18.490 ms
DEBUG 05-06 10:00:57.533497.533497 lmp.py:1388] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 10:00:57.533724.533724 lmp.py:1346] -------------------------------- start prefill layer 7 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 27, 35, 67, 75, 119, 63, 15, 55, 107, 127, 51, 23, 95, 87, 36, 32, 100, 88, 92, 116, 8, 16, 112, 68, 64, 72, 80, 49, 73, 109, 37, 77, 41, 45, 101, 17, 117, 21, 125, 9, 25, 105, 50, 62, 66, 102, 30, 38, 94, 82, 26, 54, 126, 78, 122, 98, 118], 'token_total': 307, 'token_per_expert': {11: 1, 27: 1, 35: 1, 67: 1, 75: 1, 119: 1, 63: 2, 15: 3, 55: 4, 107: 4, 127: 4, 51: 6, 23: 7, 95: 7, 87: 8, 36: 1, 32: 2, 100: 2, 88: 3, 92: 4, 116: 5, 8: 6, 16: 8, 112: 9, 68: 10, 64: 11, 72: 15, 80: 16, 49: 1, 73: 1, 109: 2, 37: 4, 77: 4, 41: 5, 45: 5, 101: 5, 17: 7, 117: 7, 21: 8, 125: 9, 9: 10, 25: 10, 105: 14, 50: 2, 62: 2, 66: 2, 102: 2, 30: 3, 38: 3, 94: 3, 82: 4, 26: 5, 54: 5, 126: 6, 78: 7, 122: 8, 98: 9, 118: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 43, 47, 59, 71, 79, 83, 91, 99, 103, 111, 115, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 804, 'token_per_expert': {3: 256, 7: 270, 19: 14, 43: 13, 47: 15, 59: 14, 71: 20, 79: 32, 83: 10, 91: 98, 99: 12, 103: 20, 111: 8, 115: 9, 123: 13}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 28, 44, 48, 52, 56, 60, 84, 96, 104, 108, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 1005, 'token_per_expert': {0: 258, 4: 289, 12: 77, 20: 19, 28: 33, 44: 31, 48: 23, 52: 31, 56: 34, 60: 26, 84: 47, 96: 26, 104: 22, 108: 50, 120: 39}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 33, 53, 57, 61, 65, 69, 85, 97, 113, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1055, 'token_per_expert': {1: 256, 5: 277, 13: 15, 29: 49, 33: 34, 53: 33, 57: 23, 61: 17, 65: 49, 69: 42, 85: 37, 97: 130, 113: 25, 121: 68}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 34, 42, 70, 86, 90, 106, 110, 114], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 925, 'token_per_expert': {2: 256, 6: 267, 10: 48, 14: 33, 18: 17, 22: 17, 34: 41, 42: 41, 70: 51, 86: 25, 90: 44, 106: 15, 110: 37, 114: 33}}
INFO 05-06 10:00:57.540654.540654 lmp.py:1833] [layer_moe_fused] layer=7 prefix: 0.479ms alloc: 0.416ms
INFO 05-06 10:00:57.540011.540011 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.389617919921875e-05 seconds
INFO 05-06 10:00:57.541251.541251 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007605552673339844s
INFO 05-06 10:00:57.542876.542876 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006067752838134766 seconds
INFO 05-06 10:00:57.544262.544262 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001779794692993164s
INFO 05-06 10:00:57.552173.552173 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.722ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.552997.552997 cuda_h.py:27] end *layer_moe_fused cost 13.123 ms
DEBUG 05-06 10:00:57.553016.553016 cuda_h.py:27] end prefill_layer cost 19.204 ms
DEBUG 05-06 10:00:57.553449.553449 lmp.py:1388] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 10:00:57.553273.553273 lmp.py:1346] -------------------------------- start prefill layer 8 --------------------------------
experts_cpu_alloc {'expert_ids': [23, 39, 35, 43, 91, 31, 119, 127, 11, 99, 47, 8, 72, 48, 84, 92, 104, 64, 96, 116, 44, 68, 108, 9, 25, 101, 117, 13, 89, 33, 49, 37, 57, 85, 21, 29, 17, 41, 113, 18, 26, 82, 86, 90, 106, 118, 34, 22, 62, 74, 66, 42, 98, 14, 10], 'token_total': 288, 'token_per_expert': {23: 1, 39: 1, 35: 2, 43: 2, 91: 3, 31: 5, 119: 5, 127: 9, 11: 10, 99: 10, 47: 11, 8: 1, 72: 1, 48: 2, 84: 3, 92: 3, 104: 3, 64: 4, 96: 4, 116: 6, 44: 7, 68: 7, 108: 7, 9: 1, 25: 2, 101: 2, 117: 3, 13: 4, 89: 4, 33: 5, 49: 5, 37: 6, 57: 6, 85: 6, 21: 7, 29: 7, 17: 8, 41: 9, 113: 10, 18: 2, 26: 2, 82: 2, 86: 2, 90: 2, 106: 2, 118: 2, 34: 4, 22: 6, 62: 6, 74: 6, 66: 7, 42: 11, 98: 12, 14: 14, 10: 16}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 51, 55, 63, 71, 75, 87, 103, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 1025, 'token_per_expert': {3: 279, 7: 256, 15: 29, 19: 40, 27: 19, 51: 95, 55: 11, 63: 11, 71: 40, 75: 46, 87: 61, 103: 99, 111: 16, 123: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 20, 28, 32, 36, 52, 56, 76, 80, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 861, 'token_per_expert': {0: 258, 4: 266, 12: 24, 16: 12, 20: 12, 28: 66, 32: 44, 36: 15, 52: 19, 56: 38, 76: 16, 80: 26, 120: 57, 124: 8}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 45, 53, 61, 65, 69, 73, 77, 81, 93, 105, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 817, 'token_per_expert': {1: 258, 5: 272, 45: 11, 53: 22, 61: 15, 65: 24, 69: 15, 73: 57, 77: 17, 81: 19, 93: 14, 105: 39, 121: 26, 125: 28}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 50, 54, 58, 70, 102, 110, 114, 122, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 1105, 'token_per_expert': {2: 289, 6: 271, 38: 37, 46: 30, 50: 45, 54: 112, 58: 102, 70: 45, 102: 29, 110: 63, 114: 43, 122: 21, 126: 18}}
INFO 05-06 10:00:57.558994.558994 lmp.py:1833] [layer_moe_fused] layer=8 prefix: 0.459ms alloc: 0.402ms
INFO 05-06 10:00:57.558621.558621 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.103515625e-05 seconds
INFO 05-06 10:00:57.560182.560182 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008063316345214844s
INFO 05-06 10:00:57.560727.560727 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005807876586914062 seconds
INFO 05-06 10:00:57.562883.562883 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017223358154296875s
INFO 05-06 10:00:57.570194.570194 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.603ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.570263.570263 cuda_h.py:27] end *layer_moe_fused cost 12.856 ms
DEBUG 05-06 10:00:57.571195.571195 cuda_h.py:27] end prefill_layer cost 18.017 ms
DEBUG 05-06 10:00:57.571290.571290 lmp.py:1388] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 10:00:57.572171.572171 lmp.py:1346] -------------------------------- start prefill layer 9 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 63, 79, 115, 119, 67, 19, 27, 39, 83, 64, 84, 112, 44, 96, 52, 8, 20, 116, 120, 24, 124, 68, 40, 29, 33, 121, 41, 105, 37, 97, 77, 113, 117, 73, 17, 9, 26, 50, 66, 94, 58, 10, 90, 98, 114, 122, 34, 82, 42, 62], 'token_total': 258, 'token_per_expert': {11: 1, 63: 1, 79: 1, 115: 1, 119: 2, 67: 5, 19: 9, 27: 12, 39: 12, 83: 12, 64: 1, 84: 1, 112: 1, 44: 2, 96: 2, 52: 3, 8: 6, 20: 6, 116: 6, 120: 6, 24: 9, 124: 9, 68: 10, 40: 11, 29: 1, 33: 1, 121: 1, 41: 2, 105: 3, 37: 6, 97: 6, 77: 7, 113: 7, 117: 8, 73: 10, 17: 11, 9: 12, 26: 1, 50: 1, 66: 1, 94: 1, 58: 2, 10: 3, 90: 3, 98: 3, 114: 3, 122: 3, 34: 4, 82: 7, 42: 11, 62: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 23, 43, 51, 71, 75, 95, 99, 103, 111, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 23, 'token_total': 997, 'token_per_expert': {3: 268, 7: 264, 15: 19, 23: 29, 43: 57, 51: 14, 71: 18, 75: 57, 95: 143, 99: 20, 103: 68, 111: 17, 127: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 32, 36, 48, 56, 72, 76, 80, 88, 92], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 915, 'token_per_expert': {0: 261, 4: 284, 12: 72, 16: 68, 32: 28, 36: 25, 48: 37, 56: 48, 72: 21, 76: 17, 80: 12, 88: 13, 92: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 45, 57, 61, 69, 81, 89, 93, 101, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 906, 'token_per_expert': {1: 269, 5: 260, 13: 17, 21: 17, 45: 16, 57: 28, 61: 15, 69: 31, 81: 53, 89: 22, 93: 72, 101: 85, 125: 21}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 30, 38, 46, 54, 70, 74, 86, 102, 106], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 26, 'token_total': 1020, 'token_per_expert': {2: 257, 6: 258, 22: 19, 30: 19, 38: 18, 46: 97, 54: 27, 70: 103, 74: 65, 86: 12, 102: 43, 106: 102}}
INFO 05-06 10:00:57.577564.577564 lmp.py:1833] [layer_moe_fused] layer=9 prefix: 0.451ms alloc: 0.382ms
INFO 05-06 10:00:57.577808.577808 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-06 10:00:57.578298.578298 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007855892181396484s
INFO 05-06 10:00:57.579267.579267 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005791187286376953 seconds
INFO 05-06 10:00:57.581892.581892 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001580953598022461s
INFO 05-06 10:00:57.588981.588981 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.516ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.589819.589819 cuda_h.py:27] end *layer_moe_fused cost 12.398 ms
DEBUG 05-06 10:00:57.589208.589208 cuda_h.py:27] end prefill_layer cost 17.718 ms
DEBUG 05-06 10:00:57.589542.589542 lmp.py:1388] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 10:00:57.590426.590426 lmp.py:1346] -------------------------------- start prefill layer 10 --------------------------------
experts_cpu_alloc {'expert_ids': [27, 51, 123, 15, 35, 107, 111, 11, 59, 103, 67, 83, 79, 12, 40, 52, 56, 64, 124, 120, 28, 44, 100, 112, 68, 33, 53, 77, 25, 109, 61, 9, 29, 37, 69, 97, 89, 73, 117, 121, 66, 70, 98, 26, 34, 78, 50, 94, 10, 90, 46], 'token_total': 260, 'token_per_expert': {27: 1, 51: 1, 123: 1, 15: 2, 35: 2, 107: 2, 111: 2, 11: 3, 59: 3, 103: 3, 67: 4, 83: 5, 79: 9, 12: 1, 40: 2, 52: 2, 56: 2, 64: 3, 124: 3, 120: 5, 28: 7, 44: 7, 100: 10, 112: 10, 68: 16, 33: 1, 53: 1, 77: 1, 25: 2, 109: 2, 61: 3, 9: 4, 29: 4, 37: 4, 69: 6, 97: 6, 89: 8, 73: 9, 117: 10, 121: 11, 66: 1, 70: 1, 98: 2, 26: 4, 34: 5, 78: 7, 50: 8, 94: 9, 10: 12, 90: 15, 46: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 39, 43, 47, 63, 71, 75, 99, 115, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 776, 'token_per_expert': {3: 257, 7: 271, 19: 12, 31: 19, 39: 11, 43: 12, 47: 27, 63: 17, 71: 26, 75: 27, 99: 13, 115: 54, 127: 30}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 60, 72, 76, 80, 84, 88, 92, 108], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 1146, 'token_per_expert': {0: 317, 4: 269, 8: 87, 16: 34, 20: 19, 60: 73, 72: 31, 76: 121, 80: 79, 84: 17, 88: 50, 92: 25, 108: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 41, 49, 57, 81, 85, 93, 105, 113, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 952, 'token_per_expert': {1: 348, 5: 268, 13: 23, 21: 33, 41: 30, 49: 25, 57: 37, 81: 68, 85: 32, 93: 12, 105: 14, 113: 29, 125: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 42, 54, 58, 62, 74, 82, 86, 106, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 962, 'token_per_expert': {2: 256, 6: 256, 14: 46, 18: 38, 42: 42, 54: 20, 58: 36, 62: 56, 74: 55, 82: 19, 86: 64, 106: 50, 126: 24}}
INFO 05-06 10:00:57.595261.595261 lmp.py:1833] [layer_moe_fused] layer=10 prefix: 0.444ms alloc: 0.395ms
INFO 05-06 10:00:57.595696.595696 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.745887756347656e-05 seconds
INFO 05-06 10:00:57.596951.596951 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007905960083007812s
INFO 05-06 10:00:57.597674.597674 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005738735198974609 seconds
INFO 05-06 10:00:57.599188.599188 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017969608306884766s
INFO 05-06 10:00:57.606529.606529 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.522ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.607712.607712 cuda_h.py:27] end *layer_moe_fused cost 12.513 ms
DEBUG 05-06 10:00:57.608002.608002 cuda_h.py:27] end prefill_layer cost 17.798 ms
DEBUG 05-06 10:00:57.608197.608197 lmp.py:1388] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 10:00:57.608362.608362 lmp.py:1346] -------------------------------- start prefill layer 11 --------------------------------
experts_cpu_alloc {'expert_ids': [15, 47, 127, 35, 115, 59, 63, 91, 51, 71, 39, 11, 43, 64, 12, 72, 28, 40, 52, 80, 84, 48, 120, 8, 36, 44, 116, 124, 21, 53, 97, 13, 33, 125, 9, 117, 121, 25, 29, 94, 106, 114, 34, 122, 126, 22, 58, 110, 118, 42, 98, 50], 'token_total': 219, 'token_per_expert': {15: 1, 47: 1, 127: 1, 35: 2, 115: 2, 59: 3, 63: 3, 91: 6, 51: 7, 71: 7, 39: 8, 11: 9, 43: 9, 64: 1, 12: 2, 72: 2, 28: 3, 40: 3, 52: 3, 80: 3, 84: 3, 48: 4, 120: 5, 8: 6, 36: 6, 44: 7, 116: 8, 124: 11, 21: 1, 53: 1, 97: 1, 13: 2, 33: 2, 125: 2, 9: 3, 117: 6, 121: 13, 25: 14, 29: 14, 94: 1, 106: 1, 114: 1, 34: 2, 122: 2, 126: 2, 22: 3, 58: 3, 110: 3, 118: 3, 42: 4, 98: 4, 50: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 31, 67, 79, 83, 87, 99, 111, 119, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1055, 'token_per_expert': {3: 259, 7: 305, 19: 20, 23: 54, 27: 11, 31: 20, 67: 46, 79: 73, 83: 77, 87: 78, 99: 27, 111: 49, 119: 26, 123: 10}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 32, 56, 68, 76, 92, 100, 108, 112], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 1042, 'token_per_expert': {0: 258, 4: 258, 16: 79, 20: 29, 24: 29, 32: 38, 56: 97, 68: 47, 76: 31, 92: 88, 100: 24, 108: 31, 112: 33}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 37, 49, 57, 61, 69, 77, 81, 89, 93, 113], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 959, 'token_per_expert': {1: 263, 5: 264, 17: 41, 37: 24, 49: 54, 57: 20, 61: 15, 69: 21, 77: 21, 81: 76, 89: 18, 93: 62, 113: 80}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 30, 38, 46, 54, 62, 66, 70, 82, 102], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 821, 'token_per_expert': {2: 285, 6: 305, 10: 27, 18: 9, 30: 26, 38: 13, 46: 10, 54: 5, 62: 5, 66: 17, 70: 5, 82: 9, 102: 105}}
INFO 05-06 10:00:57.614322.614322 lmp.py:1833] [layer_moe_fused] layer=11 prefix: 0.450ms alloc: 0.391ms
INFO 05-06 10:00:57.614864.614864 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.29425048828125e-05 seconds
INFO 05-06 10:00:57.615632.615632 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000759124755859375s
INFO 05-06 10:00:57.615058.615058 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005986690521240234 seconds
INFO 05-06 10:00:57.617488.617488 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016078948974609375s
INFO 05-06 10:00:57.625955.625955 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.508ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.625827.625827 cuda_h.py:27] end *layer_moe_fused cost 12.555 ms
DEBUG 05-06 10:00:57.626632.626632 cuda_h.py:27] end prefill_layer cost 18.135 ms
DEBUG 05-06 10:00:57.626250.626250 lmp.py:1388] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 10:00:57.626636.626636 lmp.py:1346] -------------------------------- start prefill layer 12 --------------------------------
experts_cpu_alloc {'expert_ids': [31, 59, 47, 111, 63, 107, 119, 127, 123, 79, 8, 120, 20, 32, 24, 88, 12, 40, 104, 112, 36, 37, 81, 113, 13, 65, 33, 105, 125, 17, 89, 101, 77, 18, 102, 94, 70, 38, 58, 90, 22, 98, 34, 106], 'token_total': 220, 'token_per_expert': {31: 1, 59: 1, 47: 2, 111: 2, 63: 3, 107: 3, 119: 3, 127: 4, 123: 5, 79: 8, 8: 1, 120: 1, 20: 3, 32: 3, 24: 4, 88: 6, 12: 7, 40: 7, 104: 8, 112: 8, 36: 9, 37: 1, 81: 1, 113: 1, 13: 2, 65: 2, 33: 3, 105: 3, 125: 3, 17: 5, 89: 6, 101: 8, 77: 9, 18: 1, 102: 2, 94: 3, 70: 4, 38: 5, 58: 5, 90: 5, 22: 7, 98: 7, 34: 13, 106: 35}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 35, 39, 71, 91, 95, 103, 115], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 22, 'token_total': 937, 'token_per_expert': {3: 275, 7: 256, 15: 71, 19: 33, 23: 22, 35: 18, 39: 88, 71: 89, 91: 32, 95: 22, 103: 8, 115: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 68, 76, 80, 84, 92, 100, 108, 116, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 22, 'token_total': 753, 'token_per_expert': {0: 256, 4: 258, 68: 13, 76: 9, 80: 10, 84: 13, 92: 21, 100: 12, 108: 83, 116: 67, 124: 11}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25, 45, 49, 53, 73, 85, 97, 117], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 23, 'token_total': 998, 'token_per_expert': {1: 261, 5: 288, 21: 107, 25: 31, 45: 59, 49: 29, 53: 115, 73: 34, 85: 23, 97: 36, 117: 15}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 46, 50, 74, 78, 82, 86, 110, 114, 118], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 22, 'token_total': 1188, 'token_per_expert': {2: 257, 6: 288, 46: 41, 50: 76, 74: 83, 78: 160, 82: 50, 86: 70, 110: 53, 114: 62, 118: 48}}
INFO 05-06 10:00:57.632871.632871 lmp.py:1833] [layer_moe_fused] layer=12 prefix: 0.460ms alloc: 0.350ms
INFO 05-06 10:00:57.632922.632922 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.7697296142578125e-05 seconds
INFO 05-06 10:00:57.633046.633046 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007376670837402344s
INFO 05-06 10:00:57.633014.633014 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005779266357421875 seconds
INFO 05-06 10:00:57.635472.635472 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0014879703521728516s
INFO 05-06 10:00:57.643456.643456 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.335ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.643798.643798 cuda_h.py:27] end *layer_moe_fused cost 12.275 ms
DEBUG 05-06 10:00:57.644736.644736 cuda_h.py:27] end prefill_layer cost 17.412 ms
DEBUG 05-06 10:00:57.644455.644455 lmp.py:1388] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 10:00:57.644324.644324 lmp.py:1346] -------------------------------- start prefill layer 13 --------------------------------
experts_cpu_alloc {'expert_ids': [47, 95, 123, 83, 107, 27, 43, 87, 11, 75, 115, 67, 99, 55, 48, 104, 28, 8, 64, 40, 52, 60, 96, 53, 105, 61, 97, 57, 73, 9, 65, 13, 117, 93, 101, 10, 58, 62, 66, 94, 70, 74, 106, 82, 90, 26, 46, 122, 38, 42, 86], 'token_total': 235, 'token_per_expert': {47: 1, 95: 1, 123: 2, 83: 3, 107: 3, 27: 4, 43: 4, 87: 4, 11: 7, 75: 7, 115: 7, 67: 8, 99: 9, 55: 10, 48: 1, 104: 3, 28: 4, 8: 5, 64: 5, 40: 6, 52: 6, 60: 6, 96: 6, 53: 1, 105: 1, 61: 2, 97: 2, 57: 3, 73: 3, 9: 4, 65: 4, 13: 5, 117: 8, 93: 9, 101: 9, 10: 1, 58: 1, 62: 1, 66: 1, 94: 1, 70: 2, 74: 2, 106: 3, 82: 4, 90: 4, 26: 5, 46: 7, 122: 7, 38: 9, 42: 10, 86: 14}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 31, 39, 51, 59, 63, 71, 79, 91, 103, 119], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 1032, 'token_per_expert': {3: 272, 7: 256, 15: 18, 31: 107, 39: 18, 51: 24, 59: 39, 63: 36, 71: 48, 79: 71, 91: 103, 103: 24, 119: 16}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 32, 68, 84, 92, 100, 108, 116, 120, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 22, 'token_total': 854, 'token_per_expert': {0: 257, 4: 256, 16: 9, 20: 22, 32: 52, 68: 8, 84: 21, 92: 7, 100: 104, 108: 11, 116: 16, 120: 70, 124: 21}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 25, 33, 37, 41, 69, 81, 113, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 925, 'token_per_expert': {1: 280, 5: 256, 17: 55, 21: 25, 25: 38, 33: 23, 37: 54, 41: 15, 69: 21, 81: 47, 113: 21, 121: 71, 125: 19}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 34, 78, 98, 102, 110, 114, 118, 126], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 28, 'token_total': 1050, 'token_per_expert': {2: 265, 6: 299, 14: 50, 22: 27, 34: 22, 78: 48, 98: 33, 102: 39, 110: 95, 114: 108, 118: 31, 126: 33}}
INFO 05-06 10:00:57.649417.649417 lmp.py:1833] [layer_moe_fused] layer=13 prefix: 0.437ms alloc: 0.378ms
INFO 05-06 10:00:57.649806.649806 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.8650970458984375e-05 seconds
INFO 05-06 10:00:57.650346.650346 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007169246673583984s
INFO 05-06 10:00:57.651254.651254 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005676746368408203 seconds
INFO 05-06 10:00:57.653348.653348 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015187263488769531s
INFO 05-06 10:00:57.660086.660086 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.501ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.661939.661939 cuda_h.py:27] end *layer_moe_fused cost 12.436 ms
DEBUG 05-06 10:00:57.661895.661895 cuda_h.py:27] end prefill_layer cost 17.443 ms
DEBUG 05-06 10:00:57.661612.661612 lmp.py:1388] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 10:00:57.662031.662031 lmp.py:1346] -------------------------------- start prefill layer 14 --------------------------------
experts_cpu_alloc {'expert_ids': [19, 63, 79, 91, 111, 15, 43, 51, 71, 35, 67, 107, 23, 83, 56, 48, 68, 96, 36, 64, 40, 44, 108, 116, 120, 8, 24, 28, 17, 85, 9, 33, 37, 77, 73, 101, 109, 81, 21, 93, 125, 14, 46, 54, 94, 18, 70, 98, 10, 118, 126, 102, 106, 78, 58, 90, 110], 'token_total': 243, 'token_per_expert': {19: 1, 63: 1, 79: 1, 91: 1, 111: 1, 15: 2, 43: 2, 51: 2, 71: 3, 35: 4, 67: 5, 107: 6, 23: 7, 83: 9, 56: 1, 48: 3, 68: 3, 96: 3, 36: 4, 64: 4, 40: 6, 44: 6, 108: 6, 116: 7, 120: 7, 8: 9, 24: 9, 28: 10, 17: 1, 85: 1, 9: 2, 33: 2, 37: 2, 77: 2, 73: 3, 101: 3, 109: 5, 81: 6, 21: 9, 93: 10, 125: 10, 14: 1, 46: 1, 54: 1, 94: 1, 18: 2, 70: 3, 98: 3, 10: 4, 118: 4, 126: 4, 102: 5, 106: 5, 78: 6, 58: 7, 90: 8, 110: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 31, 39, 47, 59, 75, 95, 99, 103, 115, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 1131, 'token_per_expert': {3: 260, 7: 268, 11: 20, 31: 28, 39: 61, 47: 49, 59: 30, 75: 39, 95: 31, 99: 38, 103: 44, 115: 147, 119: 64, 123: 28, 127: 24}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 32, 52, 60, 72, 76, 80, 92, 100, 104, 112, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 815, 'token_per_expert': {0: 260, 4: 256, 12: 21, 16: 15, 32: 17, 52: 17, 60: 11, 72: 16, 76: 14, 80: 30, 92: 11, 100: 34, 104: 21, 112: 12, 124: 80}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 45, 53, 57, 65, 89, 97, 105, 113, 117, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 932, 'token_per_expert': {1: 257, 5: 265, 13: 20, 25: 12, 45: 11, 53: 22, 57: 12, 65: 63, 89: 19, 97: 55, 105: 15, 113: 36, 117: 43, 121: 102}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 30, 34, 38, 42, 50, 62, 66, 74, 86, 114, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 975, 'token_per_expert': {2: 289, 6: 257, 26: 76, 30: 30, 34: 13, 38: 13, 42: 26, 50: 60, 62: 36, 66: 54, 74: 16, 86: 70, 114: 9, 122: 26}}
INFO 05-06 10:00:57.667095.667095 lmp.py:1833] [layer_moe_fused] layer=14 prefix: 0.450ms alloc: 0.419ms
INFO 05-06 10:00:57.667531.667531 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.008148193359375e-05 seconds
INFO 05-06 10:00:57.668607.668607 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007417201995849609s
INFO 05-06 10:00:57.669052.669052 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005776882171630859 seconds
INFO 05-06 10:00:57.671410.671410 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001567840576171875s
INFO 05-06 10:00:57.679121.679121 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.681ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.679670.679670 cuda_h.py:27] end *layer_moe_fused cost 12.880 ms
DEBUG 05-06 10:00:57.680482.680482 cuda_h.py:27] end prefill_layer cost 18.021 ms
DEBUG 05-06 10:00:57.680584.680584 lmp.py:1388] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 10:00:57.680140.680140 lmp.py:1346] -------------------------------- start prefill layer 15 --------------------------------
experts_cpu_alloc {'expert_ids': [79, 19, 35, 11, 111, 127, 55, 119, 43, 107, 115, 31, 59, 32, 56, 100, 8, 80, 28, 96, 48, 36, 120, 116, 16, 24, 45, 57, 105, 25, 117, 33, 121, 29, 77, 13, 41, 17, 69, 113, 97, 94, 126, 54, 82, 22, 34, 38, 58, 78, 118, 18, 46], 'token_total': 339, 'token_per_expert': {79: 1, 19: 2, 35: 2, 11: 3, 111: 3, 127: 3, 55: 4, 119: 9, 43: 10, 107: 10, 115: 11, 31: 12, 59: 14, 32: 1, 56: 1, 100: 1, 8: 4, 80: 5, 28: 6, 96: 7, 48: 8, 36: 9, 120: 12, 116: 13, 16: 14, 24: 18, 45: 1, 57: 1, 105: 3, 25: 4, 117: 4, 33: 6, 121: 6, 29: 7, 77: 7, 13: 9, 41: 9, 17: 10, 69: 10, 113: 14, 97: 15, 94: 1, 126: 1, 54: 2, 82: 3, 22: 4, 34: 5, 38: 5, 58: 5, 78: 5, 118: 5, 18: 6, 46: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 47, 51, 63, 71, 75, 83, 91, 95, 99, 103], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 981, 'token_per_expert': {3: 257, 7: 278, 23: 34, 39: 31, 47: 18, 51: 30, 63: 20, 71: 40, 75: 47, 83: 72, 91: 91, 95: 17, 99: 27, 103: 19}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 52, 64, 68, 72, 76, 84, 88, 104, 108, 112, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 1013, 'token_per_expert': {0: 265, 4: 261, 52: 33, 64: 27, 68: 86, 72: 22, 76: 94, 84: 23, 88: 21, 104: 25, 108: 46, 112: 86, 124: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 37, 65, 73, 81, 85, 93, 101, 109, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 914, 'token_per_expert': {1: 270, 5: 277, 9: 38, 21: 26, 37: 22, 65: 70, 73: 18, 81: 29, 85: 22, 93: 21, 101: 30, 109: 74, 125: 17}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 30, 42, 66, 70, 86, 90, 98, 102, 114], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 849, 'token_per_expert': {2: 284, 6: 256, 10: 48, 14: 12, 30: 43, 42: 15, 66: 33, 70: 25, 86: 8, 90: 70, 98: 33, 102: 11, 114: 11}}
INFO 05-06 10:00:57.685265.685265 lmp.py:1833] [layer_moe_fused] layer=15 prefix: 0.448ms alloc: 0.398ms
INFO 05-06 10:00:57.686714.686714 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.127357482910156e-05 seconds
INFO 05-06 10:00:57.687853.687853 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007846355438232422s
INFO 05-06 10:00:57.687862.687862 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006060600280761719 seconds
INFO 05-06 10:00:57.689722.689722 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015921592712402344s
INFO 05-06 10:00:57.697158.697158 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.764ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.698806.698806 cuda_h.py:27] end *layer_moe_fused cost 13.043 ms
DEBUG 05-06 10:00:57.698505.698505 cuda_h.py:27] end prefill_layer cost 18.204 ms
DEBUG 05-06 10:00:57.698984.698984 lmp.py:1388] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 10:00:57.699059.699059 lmp.py:1346] -------------------------------- start prefill layer 16 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 115, 99, 103, 111, 43, 51, 71, 59, 123, 15, 119, 60, 84, 56, 64, 88, 120, 36, 24, 104, 40, 92, 116, 80, 108, 72, 29, 53, 101, 41, 49, 89, 109, 113, 33, 81, 37, 13, 57, 61, 46, 74, 106, 34, 50, 98, 122, 10, 38, 118, 82, 18, 62, 90, 102, 30, 110], 'token_total': 264, 'token_per_expert': {11: 1, 115: 1, 99: 2, 103: 2, 111: 2, 43: 3, 51: 3, 71: 3, 59: 4, 123: 5, 15: 6, 119: 7, 60: 1, 84: 1, 56: 2, 64: 2, 88: 2, 120: 2, 36: 3, 24: 4, 104: 5, 40: 9, 92: 9, 116: 12, 80: 13, 108: 15, 72: 18, 29: 1, 53: 1, 101: 1, 41: 2, 49: 2, 89: 3, 109: 3, 113: 3, 33: 4, 81: 4, 37: 5, 13: 6, 57: 6, 61: 6, 46: 1, 74: 1, 106: 1, 34: 2, 50: 2, 98: 2, 122: 2, 10: 3, 38: 3, 118: 4, 82: 5, 18: 6, 62: 6, 90: 7, 102: 7, 30: 11, 110: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 31, 55, 63, 67, 75, 79, 83, 87, 91, 107, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 27, 'token_total': 938, 'token_per_expert': {3: 268, 7: 261, 19: 16, 23: 24, 31: 31, 55: 24, 63: 21, 67: 65, 75: 20, 79: 12, 83: 34, 87: 99, 91: 8, 107: 37, 127: 18}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 32, 44, 48, 52, 68, 76, 96, 100, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1151, 'token_per_expert': {0: 282, 4: 283, 8: 38, 12: 31, 16: 94, 20: 20, 32: 112, 44: 38, 48: 20, 52: 96, 68: 25, 76: 22, 96: 29, 100: 30, 124: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 45, 65, 77, 85, 93, 97, 105, 117, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 799, 'token_per_expert': {1: 309, 5: 275, 9: 9, 17: 9, 21: 10, 45: 9, 65: 13, 77: 15, 85: 17, 93: 11, 97: 17, 105: 53, 117: 17, 121: 14, 125: 21}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 26, 42, 54, 58, 66, 70, 78, 86, 114, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 31, 'token_total': 944, 'token_per_expert': {2: 267, 6: 257, 14: 32, 22: 19, 26: 24, 42: 20, 54: 24, 58: 18, 66: 45, 70: 18, 78: 18, 86: 60, 114: 21, 126: 121}}
INFO 05-06 10:00:57.704980.704980 lmp.py:1833] [layer_moe_fused] layer=16 prefix: 0.455ms alloc: 0.426ms
INFO 05-06 10:00:57.704243.704243 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.437301635742188e-05 seconds
INFO 05-06 10:00:57.705510.705510 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007870197296142578s
INFO 05-06 10:00:57.706645.706645 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005941390991210938 seconds
INFO 05-06 10:00:57.708309.708309 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0019233226776123047s
INFO 05-06 10:00:57.716478.716478 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.705ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.716591.716591 cuda_h.py:27] end *layer_moe_fused cost 13.199 ms
DEBUG 05-06 10:00:57.717091.717091 cuda_h.py:27] end prefill_layer cost 18.345 ms
DEBUG 05-06 10:00:57.717570.717570 lmp.py:1388] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 10:00:57.717313.717313 lmp.py:1346] -------------------------------- start prefill layer 17 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 51, 111, 127, 87, 79, 91, 123, 15, 59, 31, 19, 99, 119, 67, 88, 36, 112, 8, 92, 96, 16, 44, 104, 60, 80, 108, 116, 124, 100, 120, 77, 81, 85, 117, 29, 65, 93, 97, 9, 33, 113, 13, 109, 125, 66, 14, 42, 62, 102, 34, 38, 122, 90, 118, 114, 126], 'token_total': 262, 'token_per_expert': {11: 1, 51: 1, 111: 1, 127: 1, 87: 2, 79: 3, 91: 3, 123: 4, 15: 5, 59: 5, 31: 6, 19: 7, 99: 7, 119: 9, 67: 11, 88: 1, 36: 2, 112: 2, 8: 3, 92: 3, 96: 4, 16: 6, 44: 6, 104: 7, 60: 8, 80: 9, 108: 9, 116: 9, 124: 9, 100: 15, 120: 16, 77: 1, 81: 1, 85: 1, 117: 1, 29: 2, 65: 2, 93: 2, 97: 4, 9: 5, 33: 5, 113: 5, 13: 6, 109: 6, 125: 8, 66: 1, 14: 2, 42: 2, 62: 2, 102: 2, 34: 3, 38: 3, 122: 3, 90: 4, 118: 4, 114: 5, 126: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 35, 39, 43, 47, 55, 63, 71, 75, 95, 103, 107], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1013, 'token_per_expert': {3: 268, 7: 256, 23: 72, 27: 58, 35: 19, 39: 48, 43: 24, 47: 22, 55: 12, 63: 23, 71: 23, 75: 63, 95: 75, 103: 18, 107: 32}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 28, 40, 48, 52, 56, 64, 68, 72, 76, 84], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 978, 'token_per_expert': {0: 263, 4: 270, 12: 20, 20: 25, 24: 82, 28: 27, 40: 36, 48: 24, 52: 38, 56: 19, 64: 34, 68: 22, 72: 35, 76: 66, 84: 17}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 37, 45, 49, 53, 57, 61, 69, 73, 89, 101], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 986, 'token_per_expert': {1: 258, 5: 272, 17: 32, 21: 57, 37: 79, 45: 11, 49: 29, 53: 27, 57: 22, 61: 31, 69: 80, 73: 15, 89: 44, 101: 29}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 54, 58, 70, 74, 78, 86, 94, 98, 106], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 857, 'token_per_expert': {2: 260, 6: 276, 10: 23, 18: 29, 22: 17, 54: 23, 58: 29, 70: 15, 74: 57, 78: 14, 86: 70, 94: 12, 98: 12, 106: 20}}
INFO 05-06 10:00:57.723849.723849 lmp.py:1833] [layer_moe_fused] layer=17 prefix: 0.460ms alloc: 0.419ms
INFO 05-06 10:00:57.723053.723053 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.0558319091796875e-05 seconds
INFO 05-06 10:00:57.724062.724062 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008027553558349609s
INFO 05-06 10:00:57.725787.725787 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006046295166015625 seconds
INFO 05-06 10:00:57.727997.727997 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017249584197998047s
INFO 05-06 10:00:57.735678.735678 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.766ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.735367.735367 cuda_h.py:27] end *layer_moe_fused cost 13.163 ms
DEBUG 05-06 10:00:57.736940.736940 cuda_h.py:27] end prefill_layer cost 18.328 ms
DEBUG 05-06 10:00:57.736658.736658 lmp.py:1388] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 10:00:57.736014.736014 lmp.py:1346] -------------------------------- start prefill layer 18 --------------------------------
experts_cpu_alloc {'expert_ids': [27, 63, 11, 19, 23, 59, 115, 51, 67, 39, 55, 103, 15, 107, 35, 95, 44, 96, 24, 112, 52, 16, 124, 48, 56, 68, 116, 12, 80, 108, 92, 25, 41, 113, 45, 9, 21, 109, 13, 97, 29, 37, 89, 73, 125, 57, 18, 94, 102, 98, 66, 74, 26, 62, 82, 30, 42, 70, 114], 'token_total': 353, 'token_per_expert': {27: 1, 63: 1, 11: 2, 19: 2, 23: 3, 59: 3, 115: 3, 51: 4, 67: 4, 39: 5, 55: 5, 103: 7, 15: 9, 107: 9, 35: 10, 95: 13, 44: 1, 96: 1, 24: 2, 112: 2, 52: 3, 16: 4, 124: 5, 48: 10, 56: 10, 68: 10, 116: 10, 12: 11, 80: 12, 108: 13, 92: 15, 25: 1, 41: 1, 113: 1, 45: 4, 9: 5, 21: 6, 109: 6, 13: 7, 97: 7, 29: 10, 37: 10, 89: 10, 73: 11, 125: 13, 57: 16, 18: 1, 94: 2, 102: 2, 98: 3, 66: 4, 74: 4, 26: 5, 62: 5, 82: 5, 30: 6, 42: 6, 70: 6, 114: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 43, 47, 71, 75, 83, 87, 91, 99, 111, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 932, 'token_per_expert': {3: 298, 7: 260, 31: 34, 43: 40, 47: 14, 71: 16, 75: 16, 83: 39, 87: 24, 91: 14, 99: 64, 111: 46, 119: 27, 123: 17, 127: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 32, 36, 40, 60, 64, 72, 76, 84, 88, 100, 104, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 919, 'token_per_expert': {0: 262, 4: 283, 8: 34, 32: 48, 36: 49, 40: 25, 60: 30, 64: 32, 72: 23, 76: 21, 84: 22, 88: 20, 100: 22, 104: 27, 120: 21}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 33, 49, 53, 61, 65, 69, 77, 81, 85, 93, 101, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1006, 'token_per_expert': {1: 280, 5: 262, 17: 27, 33: 30, 49: 38, 53: 27, 61: 37, 65: 23, 69: 16, 77: 61, 81: 22, 85: 47, 93: 26, 101: 40, 121: 70}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 34, 38, 46, 50, 54, 58, 78, 90, 110, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 886, 'token_per_expert': {2: 299, 6: 256, 10: 15, 14: 35, 34: 14, 38: 22, 46: 9, 50: 42, 54: 56, 58: 36, 78: 25, 90: 9, 110: 25, 118: 35, 122: 8}}
INFO 05-06 10:00:57.742202.742202 lmp.py:1833] [layer_moe_fused] layer=18 prefix: 0.466ms alloc: 0.427ms
INFO 05-06 10:00:57.742929.742929 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.318092346191406e-05 seconds
INFO 05-06 10:00:57.743001.743001 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007929801940917969s
INFO 05-06 10:00:57.743878.743878 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006134510040283203 seconds
INFO 05-06 10:00:57.745983.745983 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0018565654754638672s
INFO 05-06 10:00:57.753501.753501 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.846ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.754602.754602 cuda_h.py:27] end *layer_moe_fused cost 13.231 ms
DEBUG 05-06 10:00:57.755877.755877 cuda_h.py:27] end prefill_layer cost 18.430 ms
DEBUG 05-06 10:00:57.755025.755025 lmp.py:1388] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 10:00:57.755859.755859 lmp.py:1346] -------------------------------- start prefill layer 19 --------------------------------
experts_cpu_alloc {'expert_ids': [115, 67, 107, 127, 43, 103, 59, 15, 111, 47, 55, 99, 83, 11, 8, 100, 68, 124, 20, 56, 112, 84, 96, 108, 60, 104, 36, 12, 49, 105, 57, 101, 29, 45, 53, 25, 65, 77, 121, 17, 33, 73, 18, 30, 34, 54, 62, 66, 74, 46, 82, 70, 114, 42, 118, 86], 'token_total': 306, 'token_per_expert': {115: 1, 67: 2, 107: 2, 127: 2, 43: 3, 103: 7, 59: 8, 15: 9, 111: 9, 47: 12, 55: 12, 99: 12, 83: 13, 11: 14, 8: 1, 100: 1, 68: 2, 124: 2, 20: 3, 56: 4, 112: 6, 84: 8, 96: 8, 108: 9, 60: 12, 104: 12, 36: 15, 12: 16, 49: 1, 105: 1, 57: 2, 101: 2, 29: 3, 45: 3, 53: 4, 25: 5, 65: 5, 77: 6, 121: 8, 17: 9, 33: 10, 73: 10, 18: 1, 30: 1, 34: 1, 54: 1, 62: 1, 66: 1, 74: 1, 46: 2, 82: 2, 70: 3, 114: 3, 42: 4, 118: 5, 86: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 31, 35, 39, 51, 63, 75, 79, 119, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 890, 'token_per_expert': {3: 300, 7: 303, 19: 15, 23: 24, 27: 14, 31: 23, 35: 16, 39: 19, 51: 58, 63: 22, 75: 19, 79: 24, 119: 18, 123: 35}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 24, 40, 44, 48, 52, 64, 72, 76, 80, 88, 92], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1037, 'token_per_expert': {0: 263, 4: 258, 16: 24, 24: 36, 40: 36, 44: 70, 48: 18, 52: 109, 64: 66, 72: 17, 76: 28, 80: 22, 88: 33, 92: 57}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 37, 41, 61, 69, 89, 97, 109, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 964, 'token_per_expert': {1: 282, 5: 275, 9: 28, 13: 12, 21: 28, 37: 84, 41: 16, 61: 33, 69: 18, 89: 84, 97: 10, 109: 26, 117: 51, 125: 17}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 26, 38, 50, 58, 90, 98, 102, 106, 122, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 899, 'token_per_expert': {2: 271, 6: 260, 10: 28, 22: 13, 26: 21, 38: 97, 50: 33, 58: 11, 90: 12, 98: 15, 102: 30, 106: 9, 122: 88, 126: 11}}
INFO 05-06 10:00:57.760904.760904 lmp.py:1833] [layer_moe_fused] layer=19 prefix: 0.462ms alloc: 0.407ms
INFO 05-06 10:00:57.760631.760631 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.198883056640625e-05 seconds
INFO 05-06 10:00:57.761649.761649 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008015632629394531s
INFO 05-06 10:00:57.762002.762002 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005776882171630859 seconds
INFO 05-06 10:00:57.764078.764078 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016291141510009766s
INFO 05-06 10:00:57.772182.772182 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.724ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.772858.772858 cuda_h.py:27] end *layer_moe_fused cost 12.949 ms
DEBUG 05-06 10:00:57.773411.773411 cuda_h.py:27] end prefill_layer cost 18.061 ms
DEBUG 05-06 10:00:57.773652.773652 lmp.py:1388] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 10:00:57.773420.773420 lmp.py:1346] -------------------------------- start prefill layer 20 --------------------------------
experts_cpu_alloc {'expert_ids': [75, 87, 99, 31, 35, 11, 95, 111, 19, 16, 80, 124, 104, 12, 20, 24, 36, 76, 84, 60, 120, 72, 64, 100, 29, 89, 17, 25, 61, 69, 117, 105, 93, 97, 101, 121, 9, 41, 113, 81, 85, 53, 86, 106, 110, 34, 70, 126, 62, 74, 90, 26, 98, 58, 38, 10, 114], 'token_total': 324, 'token_per_expert': {75: 1, 87: 1, 99: 1, 31: 2, 35: 4, 11: 5, 95: 5, 111: 6, 19: 7, 16: 1, 80: 1, 124: 1, 104: 4, 12: 5, 20: 5, 24: 6, 36: 6, 76: 6, 84: 7, 60: 8, 120: 9, 72: 11, 64: 13, 100: 14, 29: 1, 89: 1, 17: 2, 25: 2, 61: 2, 69: 3, 117: 3, 105: 4, 93: 5, 97: 5, 101: 5, 121: 6, 9: 7, 41: 12, 113: 13, 81: 14, 85: 16, 53: 20, 86: 1, 106: 2, 110: 2, 34: 3, 70: 3, 126: 3, 62: 5, 74: 5, 90: 5, 26: 6, 98: 6, 58: 7, 38: 8, 10: 9, 114: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 43, 47, 55, 59, 63, 71, 79, 83, 103, 107, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 24, 'token_total': 889, 'token_per_expert': {3: 297, 7: 258, 15: 10, 27: 25, 43: 24, 47: 8, 55: 9, 59: 27, 63: 73, 71: 14, 79: 9, 83: 8, 103: 8, 107: 92, 123: 27}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 28, 32, 40, 44, 52, 56, 68, 88, 92, 108, 112, 116], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1065, 'token_per_expert': {0: 265, 4: 302, 8: 29, 28: 40, 32: 24, 40: 42, 44: 26, 52: 19, 56: 46, 68: 143, 88: 21, 92: 36, 108: 27, 112: 15, 116: 30}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 33, 37, 45, 49, 57, 65, 73, 77, 109, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 32, 'token_total': 1017, 'token_per_expert': {1: 264, 5: 303, 13: 27, 21: 33, 33: 21, 37: 36, 45: 76, 49: 78, 57: 30, 65: 27, 73: 37, 77: 36, 109: 28, 125: 21}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 30, 42, 46, 50, 54, 66, 82, 94, 102, 118, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 801, 'token_per_expert': {2: 264, 6: 258, 18: 12, 30: 29, 42: 15, 46: 11, 50: 12, 54: 15, 66: 25, 82: 18, 94: 76, 102: 48, 118: 9, 122: 9}}
INFO 05-06 10:00:57.778737.778737 lmp.py:1833] [layer_moe_fused] layer=20 prefix: 0.465ms alloc: 0.423ms
INFO 05-06 10:00:57.779809.779809 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.4849853515625e-05 seconds
INFO 05-06 10:00:57.780076.780076 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008025169372558594s
INFO 05-06 10:00:57.780482.780482 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005822181701660156 seconds
INFO 05-06 10:00:57.783855.783855 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0018088817596435547s
INFO 05-06 10:00:57.790557.790557 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.773ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.791777.791777 cuda_h.py:27] end *layer_moe_fused cost 13.347 ms
DEBUG 05-06 10:00:57.792211.792211 cuda_h.py:27] end prefill_layer cost 18.482 ms
DEBUG 05-06 10:00:57.792313.792313 lmp.py:1388] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 10:00:57.792448.792448 lmp.py:1346] -------------------------------- start prefill layer 21 --------------------------------
experts_cpu_alloc {'expert_ids': [19, 27, 47, 99, 23, 59, 107, 43, 115, 55, 119, 71, 87, 56, 52, 60, 104, 16, 32, 64, 88, 20, 96, 40, 80, 24, 44, 12, 17, 25, 77, 113, 101, 21, 69, 125, 93, 121, 45, 33, 81, 54, 74, 98, 94, 106, 126, 50, 114, 38, 34, 42, 58, 82, 118, 10], 'token_total': 315, 'token_per_expert': {19: 1, 27: 1, 47: 1, 99: 1, 23: 2, 59: 2, 107: 3, 43: 5, 115: 5, 55: 6, 119: 6, 71: 8, 87: 9, 56: 1, 52: 2, 60: 2, 104: 2, 16: 3, 32: 4, 64: 4, 88: 5, 20: 6, 96: 7, 40: 8, 80: 9, 24: 10, 44: 10, 12: 14, 17: 1, 25: 1, 77: 1, 113: 2, 101: 4, 21: 5, 69: 5, 125: 5, 93: 6, 121: 6, 45: 9, 33: 13, 81: 15, 54: 1, 74: 2, 98: 2, 94: 3, 106: 3, 126: 3, 50: 4, 114: 5, 38: 9, 34: 11, 42: 11, 58: 11, 82: 13, 118: 13, 10: 14}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 31, 35, 51, 67, 75, 79, 83, 95, 103, 111, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 798, 'token_per_expert': {3: 258, 7: 279, 11: 37, 31: 14, 35: 28, 51: 34, 67: 13, 75: 13, 79: 11, 83: 23, 95: 12, 103: 20, 111: 33, 123: 10, 127: 13}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 36, 48, 68, 72, 76, 84, 92, 100, 112, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 936, 'token_per_expert': {0: 258, 4: 278, 8: 26, 36: 15, 48: 52, 68: 24, 72: 18, 76: 39, 84: 27, 92: 37, 100: 69, 112: 37, 120: 31, 124: 25}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 37, 41, 53, 57, 61, 65, 73, 97, 105, 109], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1028, 'token_per_expert': {1: 321, 5: 349, 13: 24, 29: 28, 37: 31, 41: 21, 53: 30, 57: 20, 61: 26, 65: 57, 73: 34, 97: 24, 105: 40, 109: 23}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 26, 30, 46, 62, 70, 78, 86, 90, 102, 110, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1019, 'token_per_expert': {2: 274, 6: 334, 18: 41, 26: 55, 30: 15, 46: 40, 62: 22, 70: 16, 78: 90, 86: 16, 90: 26, 102: 24, 110: 30, 122: 36}}
INFO 05-06 10:00:57.797885.797885 lmp.py:1833] [layer_moe_fused] layer=21 prefix: 0.471ms alloc: 0.414ms
INFO 05-06 10:00:57.797149.797149 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.413459777832031e-05 seconds
INFO 05-06 10:00:57.799978.799978 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007894039154052734s
INFO 05-06 10:00:57.799995.799995 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006470680236816406 seconds
INFO 05-06 10:00:57.801814.801814 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017032623291015625s
INFO 05-06 10:00:57.809931.809931 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.739ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.810059.810059 cuda_h.py:27] end *layer_moe_fused cost 13.198 ms
DEBUG 05-06 10:00:57.810274.810274 cuda_h.py:27] end prefill_layer cost 18.334 ms
DEBUG 05-06 10:00:57.810992.810992 lmp.py:1388] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 10:00:57.811555.811555 lmp.py:1346] -------------------------------- start prefill layer 22 --------------------------------
experts_cpu_alloc {'expert_ids': [67, 71, 95, 23, 63, 27, 39, 87, 47, 51, 83, 79, 15, 107, 115, 11, 31, 36, 60, 104, 20, 96, 84, 32, 48, 112, 16, 44, 124, 28, 40, 9, 17, 29, 77, 61, 109, 65, 81, 45, 57, 125, 14, 54, 110, 114, 98, 106, 62, 10, 122, 42, 34, 30, 102, 58], 'token_total': 299, 'token_per_expert': {67: 1, 71: 1, 95: 1, 23: 2, 63: 2, 27: 3, 39: 3, 87: 4, 47: 6, 51: 8, 83: 8, 79: 9, 15: 10, 107: 11, 115: 11, 11: 12, 31: 16, 36: 1, 60: 1, 104: 1, 20: 2, 96: 2, 84: 3, 32: 5, 48: 7, 112: 7, 16: 9, 44: 10, 124: 10, 28: 15, 40: 15, 9: 1, 17: 1, 29: 1, 77: 1, 61: 2, 109: 2, 65: 3, 81: 3, 45: 4, 57: 4, 125: 4, 14: 1, 54: 1, 110: 1, 114: 1, 98: 3, 106: 3, 62: 4, 10: 5, 122: 5, 42: 8, 34: 10, 30: 11, 102: 11, 58: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 35, 43, 55, 59, 75, 99, 103, 111, 119, 123, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 31, 'token_total': 952, 'token_per_expert': {3: 256, 7: 280, 19: 19, 35: 81, 43: 25, 55: 36, 59: 42, 75: 26, 99: 16, 103: 50, 111: 20, 119: 35, 123: 38, 127: 28}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 24, 64, 68, 72, 76, 88, 92, 100, 108, 116, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1166, 'token_per_expert': {0: 262, 4: 256, 8: 23, 24: 58, 64: 109, 68: 43, 72: 106, 76: 21, 88: 15, 92: 33, 100: 157, 108: 24, 116: 32, 120: 27}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 25, 33, 41, 53, 69, 73, 85, 89, 93, 101, 113, 117], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 792, 'token_per_expert': {1: 272, 5: 257, 25: 7, 33: 11, 41: 11, 53: 26, 69: 15, 73: 49, 85: 9, 89: 13, 93: 48, 101: 13, 113: 18, 117: 43}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 38, 46, 66, 70, 74, 82, 86, 90, 94, 118, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 887, 'token_per_expert': {2: 257, 6: 257, 26: 15, 38: 29, 46: 22, 66: 19, 70: 26, 74: 73, 82: 24, 86: 31, 90: 30, 94: 22, 118: 18, 126: 64}}
INFO 05-06 10:00:57.816250.816250 lmp.py:1833] [layer_moe_fused] layer=22 prefix: 0.473ms alloc: 0.403ms
INFO 05-06 10:00:57.816460.816460 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.031990051269531e-05 seconds
INFO 05-06 10:00:57.817464.817464 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007748603820800781s
INFO 05-06 10:00:57.818572.818572 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005748271942138672 seconds
INFO 05-06 10:00:57.821308.821308 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.003081083297729492s
INFO 05-06 10:00:57.829250.829250 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.469ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.829093.829093 cuda_h.py:27] end *layer_moe_fused cost 13.998 ms
DEBUG 05-06 10:00:57.830686.830686 cuda_h.py:27] end prefill_layer cost 19.156 ms
DEBUG 05-06 10:00:57.830404.830404 lmp.py:1388] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 10:00:57.830016.830016 lmp.py:1346] -------------------------------- start prefill layer 23 --------------------------------
experts_cpu_alloc {'expert_ids': [99, 107, 111, 11, 23, 27, 55, 95, 127, 91, 51, 75, 19, 71, 28, 88, 92, 12, 64, 36, 60, 68, 32, 52, 120, 48, 76, 40, 124, 13, 77, 93, 101, 113, 41, 49, 53, 81, 57, 89, 33, 9, 73, 105, 50, 102, 54, 58, 62, 74, 82, 10, 66, 110, 38, 14, 122, 42], 'token_total': 260, 'token_per_expert': {99: 1, 107: 1, 111: 1, 11: 2, 23: 2, 27: 2, 55: 2, 95: 3, 127: 4, 91: 6, 51: 7, 75: 7, 19: 9, 71: 9, 28: 1, 88: 1, 92: 2, 12: 3, 64: 3, 36: 4, 60: 4, 68: 4, 32: 5, 52: 5, 120: 5, 48: 6, 76: 6, 40: 7, 124: 10, 13: 1, 77: 1, 93: 1, 101: 1, 113: 1, 41: 3, 49: 3, 53: 4, 81: 4, 57: 6, 89: 6, 33: 8, 9: 14, 73: 16, 105: 17, 50: 1, 102: 1, 54: 2, 58: 2, 62: 2, 74: 2, 82: 3, 10: 4, 66: 4, 110: 4, 38: 5, 14: 6, 122: 7, 42: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 39, 43, 47, 59, 67, 79, 83, 87, 103, 115, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 963, 'token_per_expert': {3: 287, 7: 260, 31: 17, 35: 28, 39: 57, 43: 57, 47: 26, 59: 16, 67: 72, 79: 43, 83: 28, 87: 17, 103: 9, 115: 12, 123: 34}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 24, 44, 56, 72, 80, 84, 100, 104, 108, 112, 116], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 829, 'token_per_expert': {0: 257, 4: 258, 8: 17, 16: 22, 24: 16, 44: 34, 56: 61, 72: 16, 80: 21, 84: 26, 100: 19, 104: 22, 108: 31, 112: 16, 116: 13}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 25, 29, 37, 61, 65, 85, 97, 109, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1121, 'token_per_expert': {1: 291, 5: 300, 17: 18, 21: 102, 25: 46, 29: 42, 37: 41, 61: 59, 65: 44, 85: 39, 97: 42, 109: 26, 117: 23, 125: 48}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 26, 30, 34, 46, 78, 86, 90, 98, 106, 118], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 923, 'token_per_expert': {2: 276, 6: 270, 18: 25, 22: 23, 26: 16, 30: 14, 34: 18, 46: 55, 78: 20, 86: 83, 90: 31, 98: 39, 106: 13, 118: 40}}
INFO 05-06 10:00:57.836954.836954 lmp.py:1833] [layer_moe_fused] layer=23 prefix: 0.472ms alloc: 0.413ms
INFO 05-06 10:00:57.836058.836058 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.175041198730469e-05 seconds
INFO 05-06 10:00:57.837366.837366 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007746219635009766s
INFO 05-06 10:00:57.838149.838149 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005805492401123047 seconds
INFO 05-06 10:00:57.840235.840235 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001775503158569336s
INFO 05-06 10:00:57.848083.848083 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.610ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.848092.848092 cuda_h.py:27] end *layer_moe_fused cost 12.925 ms
DEBUG 05-06 10:00:57.849737.849737 cuda_h.py:27] end prefill_layer cost 18.573 ms
DEBUG 05-06 10:00:57.849025.849025 lmp.py:1388] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 10:00:57.849738.849738 lmp.py:1346] -------------------------------- start prefill layer 24 --------------------------------
experts_cpu_alloc {'expert_ids': [15, 51, 95, 47, 55, 87, 115, 31, 107, 119, 75, 127, 111, 28, 80, 112, 116, 104, 40, 76, 68, 32, 84, 120, 20, 92, 96, 100, 8, 25, 65, 61, 53, 9, 49, 105, 57, 81, 10, 18, 22, 26, 42, 38, 126, 46, 66, 102, 62, 82, 110, 118], 'token_total': 205, 'token_per_expert': {15: 1, 51: 1, 95: 1, 47: 2, 55: 2, 87: 2, 115: 2, 31: 3, 107: 3, 119: 3, 75: 4, 127: 4, 111: 6, 28: 1, 80: 1, 112: 1, 116: 1, 104: 2, 40: 3, 76: 3, 68: 4, 32: 5, 84: 5, 120: 5, 20: 6, 92: 6, 96: 7, 100: 7, 8: 8, 25: 1, 65: 1, 61: 4, 53: 5, 9: 6, 49: 6, 105: 6, 57: 7, 81: 8, 10: 1, 18: 1, 22: 2, 26: 2, 42: 2, 38: 3, 126: 3, 46: 4, 66: 4, 102: 4, 62: 6, 82: 8, 110: 11, 118: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 35, 43, 63, 67, 71, 79, 83, 91], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 963, 'token_per_expert': {3: 320, 7: 331, 11: 42, 19: 32, 23: 16, 27: 54, 35: 14, 43: 8, 63: 44, 67: 31, 71: 27, 79: 8, 83: 15, 91: 21}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 36, 44, 48, 52, 56, 60, 64, 108, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 950, 'token_per_expert': {0: 320, 4: 348, 12: 22, 16: 28, 36: 11, 44: 49, 48: 18, 52: 48, 56: 25, 60: 10, 64: 55, 108: 8, 124: 8}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 29, 33, 37, 45, 73, 77, 97, 109, 121], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 22, 'token_total': 996, 'token_per_expert': {1: 343, 5: 333, 13: 10, 17: 23, 29: 9, 33: 47, 37: 10, 45: 27, 73: 18, 77: 21, 97: 55, 109: 18, 121: 82}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 34, 50, 70, 74, 86, 90, 94, 98, 114, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 982, 'token_per_expert': {2: 320, 6: 354, 30: 16, 34: 32, 50: 13, 70: 49, 74: 12, 86: 12, 90: 84, 94: 21, 98: 32, 114: 23, 122: 14}}
INFO 05-06 10:00:57.854974.854974 lmp.py:1833] [layer_moe_fused] layer=24 prefix: 0.449ms alloc: 0.386ms
INFO 05-06 10:00:57.855317.855317 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.8650970458984375e-05 seconds
INFO 05-06 10:00:57.856585.856585 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007953643798828125s
INFO 05-06 10:00:57.856440.856440 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005624294281005859 seconds
INFO 05-06 10:00:57.858516.858516 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016541481018066406s
INFO 05-06 10:00:57.866558.866558 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.475ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.866136.866136 cuda_h.py:27] end *layer_moe_fused cost 12.788 ms
DEBUG 05-06 10:00:57.867476.867476 cuda_h.py:27] end prefill_layer cost 17.883 ms
DEBUG 05-06 10:00:57.867194.867194 lmp.py:1388] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 10:00:57.867768.867768 lmp.py:1346] -------------------------------- start prefill layer 25 --------------------------------
experts_cpu_alloc {'expert_ids': [55, 95, 103, 87, 23, 47, 79, 119, 19, 31, 51, 99, 76, 108, 24, 92, 124, 88, 72, 8, 36, 48, 112, 116, 37, 53, 57, 65, 81, 17, 61, 13, 29, 33, 73, 121, 9, 21, 77, 109, 86, 22, 26, 38, 42, 66, 126, 118, 122, 46, 78, 10], 'token_total': 217, 'token_per_expert': {55: 1, 95: 1, 103: 1, 87: 3, 23: 4, 47: 4, 79: 5, 119: 5, 19: 6, 31: 6, 51: 7, 99: 9, 76: 1, 108: 2, 24: 3, 92: 3, 124: 3, 88: 5, 72: 7, 8: 8, 36: 8, 48: 8, 112: 8, 116: 10, 37: 1, 53: 1, 57: 1, 65: 1, 81: 1, 17: 2, 61: 2, 13: 3, 29: 4, 33: 4, 73: 4, 121: 4, 9: 5, 21: 5, 77: 5, 109: 5, 86: 1, 22: 2, 26: 2, 38: 3, 42: 3, 66: 3, 126: 4, 118: 5, 122: 5, 46: 6, 78: 7, 10: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 35, 39, 43, 63, 67, 71, 83, 91, 107, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 934, 'token_per_expert': {3: 341, 7: 334, 11: 13, 35: 42, 39: 10, 43: 10, 63: 15, 67: 16, 71: 14, 83: 23, 91: 10, 107: 53, 111: 15, 123: 38}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 44, 52, 56, 60, 64, 68, 80, 100, 104, 120], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 1042, 'token_per_expert': {0: 329, 4: 322, 16: 120, 44: 13, 52: 45, 56: 11, 60: 29, 64: 32, 68: 63, 80: 23, 100: 13, 104: 28, 120: 14}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 25, 41, 45, 49, 69, 85, 89, 93, 97, 117, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 873, 'token_per_expert': {1: 320, 5: 329, 25: 8, 41: 8, 45: 59, 49: 8, 69: 28, 85: 29, 89: 8, 93: 15, 97: 10, 117: 41, 125: 10}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 34, 50, 58, 70, 82, 90, 106, 110, 114], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 1030, 'token_per_expert': {2: 360, 6: 322, 14: 12, 18: 50, 34: 22, 50: 11, 58: 99, 70: 20, 82: 13, 90: 16, 106: 15, 110: 78, 114: 12}}
INFO 05-06 10:00:57.873641.873641 lmp.py:1833] [layer_moe_fused] layer=25 prefix: 0.445ms alloc: 0.385ms
INFO 05-06 10:00:57.873719.873719 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.437301635742188e-05 seconds
INFO 05-06 10:00:57.874494.874494 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007443428039550781s
INFO 05-06 10:00:57.874535.874535 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.000560760498046875 seconds
INFO 05-06 10:00:57.876132.876132 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016129016876220703s
INFO 05-06 10:00:57.884354.884354 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.502ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.885960.885960 cuda_h.py:27] end *layer_moe_fused cost 12.722 ms
DEBUG 05-06 10:00:57.885300.885300 cuda_h.py:27] end prefill_layer cost 17.872 ms
DEBUG 05-06 10:00:57.885064.885064 lmp.py:1388] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 10:00:57.886087.886087 lmp.py:1346] -------------------------------- start prefill layer 26 --------------------------------
experts_cpu_alloc {'expert_ids': [31, 119, 127, 47, 55, 91, 107, 71, 23, 35, 63, 115, 99, 67, 79, 48, 64, 32, 116, 80, 92, 96, 108, 28, 44, 72, 40, 8, 88, 112, 9, 93, 117, 121, 109, 13, 29, 81, 125, 41, 45, 77, 57, 25, 54, 122, 18, 42, 74, 38, 14, 50, 26], 'token_total': 227, 'token_per_expert': {31: 1, 119: 1, 127: 1, 47: 2, 55: 2, 91: 2, 107: 2, 71: 3, 23: 6, 35: 6, 63: 6, 115: 6, 99: 8, 67: 9, 79: 12, 48: 1, 64: 1, 32: 2, 116: 2, 80: 3, 92: 3, 96: 3, 108: 3, 28: 4, 44: 4, 72: 4, 40: 5, 8: 6, 88: 7, 112: 8, 9: 1, 93: 1, 117: 1, 121: 1, 109: 2, 13: 5, 29: 5, 81: 5, 125: 5, 41: 6, 45: 6, 77: 6, 57: 7, 25: 8, 54: 2, 122: 3, 18: 4, 42: 4, 74: 4, 38: 6, 14: 7, 50: 7, 26: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 43, 51, 59, 75, 87, 95, 103, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 988, 'token_per_expert': {3: 327, 7: 320, 15: 16, 19: 16, 27: 41, 43: 36, 51: 16, 59: 15, 75: 13, 87: 63, 95: 43, 103: 15, 111: 52, 123: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 36, 52, 56, 60, 68, 76, 84, 104, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 930, 'token_per_expert': {0: 324, 4: 324, 20: 73, 24: 31, 36: 12, 52: 25, 56: 14, 60: 20, 68: 9, 76: 13, 84: 40, 104: 28, 124: 17}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 37, 49, 61, 65, 73, 85, 89, 97, 105, 113], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 1085, 'token_per_expert': {1: 322, 5: 322, 17: 69, 37: 10, 49: 12, 61: 8, 65: 44, 73: 25, 85: 99, 89: 63, 97: 11, 105: 14, 113: 86}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 30, 66, 70, 78, 86, 90, 102, 114, 118, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 22, 'token_total': 866, 'token_per_expert': {2: 323, 6: 320, 10: 13, 30: 10, 66: 14, 70: 16, 78: 16, 86: 24, 90: 13, 102: 16, 114: 69, 118: 10, 126: 22}}
INFO 05-06 10:00:57.891691.891691 lmp.py:1833] [layer_moe_fused] layer=26 prefix: 0.449ms alloc: 0.395ms
INFO 05-06 10:00:57.891947.891947 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.175041198730469e-05 seconds
INFO 05-06 10:00:57.892210.892210 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.00077056884765625s
INFO 05-06 10:00:57.893258.893258 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005650520324707031 seconds
INFO 05-06 10:00:57.895635.895635 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0017199516296386719s
INFO 05-06 10:00:57.902056.902056 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.515ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.903907.903907 cuda_h.py:27] end *layer_moe_fused cost 12.836 ms
DEBUG 05-06 10:00:57.904360.904360 cuda_h.py:27] end prefill_layer cost 18.031 ms
DEBUG 05-06 10:00:57.904363.904363 lmp.py:1388] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 10:00:57.904234.904234 lmp.py:1346] -------------------------------- start prefill layer 27 --------------------------------
experts_cpu_alloc {'expert_ids': [19, 55, 47, 59, 63, 67, 39, 15, 27, 91, 75, 119, 23, 127, 80, 124, 16, 96, 40, 116, 32, 68, 112, 20, 8, 29, 89, 101, 113, 81, 93, 97, 125, 21, 49, 105, 85, 22, 30, 34, 86, 126, 10, 26, 74, 58, 54, 90, 106, 66, 94, 122, 42], 'token_total': 247, 'token_per_expert': {19: 1, 55: 1, 47: 2, 59: 2, 63: 2, 67: 2, 39: 3, 15: 5, 27: 5, 91: 5, 75: 7, 119: 8, 23: 9, 127: 9, 80: 1, 124: 2, 16: 3, 96: 3, 40: 4, 116: 4, 32: 5, 68: 6, 112: 6, 20: 9, 8: 10, 29: 1, 89: 1, 101: 1, 113: 1, 81: 2, 93: 2, 97: 2, 125: 3, 21: 6, 49: 6, 105: 9, 85: 10, 22: 1, 30: 1, 34: 1, 86: 1, 126: 1, 10: 2, 26: 4, 74: 4, 58: 6, 54: 7, 90: 7, 106: 8, 66: 10, 94: 10, 122: 11, 42: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 43, 51, 79, 83, 87, 95, 103, 111, 115, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1043, 'token_per_expert': {3: 350, 7: 338, 31: 17, 35: 13, 43: 62, 51: 15, 79: 23, 83: 11, 87: 63, 95: 38, 103: 31, 111: 20, 115: 32, 123: 30}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 24, 28, 36, 48, 56, 64, 76, 88, 100, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 965, 'token_per_expert': {0: 322, 4: 335, 12: 14, 24: 36, 28: 11, 36: 28, 48: 27, 56: 11, 64: 18, 76: 38, 88: 28, 100: 44, 108: 12, 120: 41}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 33, 37, 41, 45, 53, 61, 65, 109, 121], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 933, 'token_per_expert': {1: 347, 5: 320, 13: 22, 25: 25, 33: 34, 37: 22, 41: 13, 45: 36, 53: 19, 61: 15, 65: 38, 109: 24, 121: 18}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 46, 50, 62, 70, 78, 82, 98, 114, 118], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 908, 'token_per_expert': {2: 320, 6: 324, 14: 26, 18: 19, 46: 17, 50: 45, 62: 20, 70: 27, 78: 16, 82: 45, 98: 15, 114: 17, 118: 17}}
INFO 05-06 10:00:57.909927.909927 lmp.py:1833] [layer_moe_fused] layer=27 prefix: 0.447ms alloc: 0.390ms
INFO 05-06 10:00:57.909600.909600 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.031990051269531e-05 seconds
INFO 05-06 10:00:57.910421.910421 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007467269897460938s
INFO 05-06 10:00:57.911237.911237 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005645751953125 seconds
INFO 05-06 10:00:57.913166.913166 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015714168548583984s
INFO 05-06 10:00:57.922942.922942 lmp.py:1938] [layer_moe_fused] vllm triton time: 8.820ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.922257.922257 cuda_h.py:27] end *layer_moe_fused cost 13.862 ms
DEBUG 05-06 10:00:57.923479.923479 cuda_h.py:27] end prefill_layer cost 18.958 ms
DEBUG 05-06 10:00:57.923218.923218 lmp.py:1388] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 10:00:57.923157.923157 lmp.py:1346] -------------------------------- start prefill layer 28 --------------------------------
experts_cpu_alloc {'expert_ids': [67, 83, 87, 99, 15, 59, 23, 39, 79, 127, 43, 95, 8, 28, 80, 108, 120, 48, 36, 44, 88, 84, 100, 24, 21, 29, 93, 9, 81, 73, 117, 69, 121, 37, 65, 97, 105, 50, 34, 58, 94, 98, 18, 54, 66, 126, 62], 'token_total': 167, 'token_per_expert': {67: 1, 83: 1, 87: 1, 99: 1, 15: 2, 59: 2, 23: 3, 39: 3, 79: 4, 127: 4, 43: 6, 95: 8, 8: 1, 28: 1, 80: 1, 108: 1, 120: 2, 48: 3, 36: 4, 44: 4, 88: 4, 84: 5, 100: 6, 24: 8, 21: 1, 29: 1, 93: 1, 9: 2, 81: 2, 73: 5, 117: 6, 69: 7, 121: 7, 37: 8, 65: 8, 97: 8, 105: 8, 50: 1, 34: 2, 58: 2, 94: 2, 98: 2, 18: 3, 54: 3, 66: 3, 126: 4, 62: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 47, 55, 71, 75, 91, 111, 115, 119, 123], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 24, 'token_total': 1037, 'token_per_expert': {3: 321, 7: 320, 11: 31, 47: 25, 55: 16, 71: 25, 75: 28, 91: 33, 111: 147, 115: 66, 119: 16, 123: 9}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 32, 40, 52, 60, 68, 76, 104, 112], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 24, 'token_total': 1122, 'token_per_expert': {0: 320, 4: 320, 12: 153, 20: 118, 32: 20, 40: 33, 52: 23, 60: 9, 68: 14, 76: 41, 104: 12, 112: 59}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 33, 49, 53, 57, 77, 85, 89, 101, 113], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 25, 'token_total': 947, 'token_per_expert': {1: 329, 5: 329, 13: 14, 33: 9, 49: 102, 53: 26, 57: 59, 77: 15, 85: 11, 89: 14, 101: 20, 113: 19}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 30, 46, 70, 74, 78, 90, 106, 110, 122], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 22, 'token_total': 823, 'token_per_expert': {2: 320, 6: 320, 22: 22, 30: 14, 46: 23, 70: 22, 74: 9, 78: 18, 90: 34, 106: 11, 110: 25, 122: 5}}
INFO 05-06 10:00:57.929512.929512 lmp.py:1833] [layer_moe_fused] layer=28 prefix: 0.447ms alloc: 0.361ms
INFO 05-06 10:00:57.929940.929940 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.507469177246094e-05 seconds
INFO 05-06 10:00:57.930703.930703 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007579326629638672s
INFO 05-06 10:00:57.930294.930294 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005772113800048828 seconds
INFO 05-06 10:00:57.932544.932544 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0018470287322998047s
INFO 05-06 10:00:57.940674.940674 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.329ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.940267.940267 cuda_h.py:27] end *layer_moe_fused cost 12.497 ms
DEBUG 05-06 10:00:57.941833.941833 cuda_h.py:27] end prefill_layer cost 17.590 ms
DEBUG 05-06 10:00:57.941881.941881 lmp.py:1388] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 10:00:57.941021.941021 lmp.py:1346] -------------------------------- start prefill layer 29 --------------------------------
experts_cpu_alloc {'expert_ids': [35, 87, 119, 127, 11, 55, 83, 111, 15, 63, 115, 123, 40, 68, 88, 108, 92, 96, 8, 32, 120, 13, 37, 105, 17, 25, 33, 65, 109, 125, 77, 81, 89, 9, 69, 49, 34, 70, 98, 118, 122, 126, 46, 50, 58, 114, 10, 74, 94, 66, 30, 62], 'token_total': 160, 'token_per_expert': {35: 1, 87: 1, 119: 1, 127: 1, 11: 2, 55: 2, 83: 2, 111: 2, 15: 3, 63: 3, 115: 3, 123: 3, 40: 1, 68: 1, 88: 1, 108: 1, 92: 3, 96: 3, 8: 6, 32: 6, 120: 7, 13: 1, 37: 1, 105: 1, 17: 2, 25: 2, 33: 2, 65: 3, 109: 4, 125: 4, 77: 6, 81: 6, 89: 6, 9: 7, 69: 8, 49: 9, 34: 1, 70: 1, 98: 1, 118: 1, 122: 1, 126: 1, 46: 2, 50: 2, 58: 2, 114: 2, 10: 3, 74: 3, 94: 3, 66: 4, 30: 9, 62: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 31, 43, 67, 71, 75, 91, 95, 99, 107], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 1019, 'token_per_expert': {3: 389, 7: 433, 19: 25, 23: 8, 27: 9, 31: 5, 43: 34, 67: 5, 71: 13, 75: 5, 91: 44, 95: 4, 99: 40, 107: 5}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 28, 44, 48, 52, 56, 60, 64, 116, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 22, 'token_total': 1013, 'token_per_expert': {0: 385, 4: 410, 16: 23, 20: 25, 28: 31, 44: 9, 48: 14, 52: 37, 56: 11, 60: 13, 64: 37, 116: 8, 124: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 53, 57, 73, 85, 93, 97, 101, 113, 117, 121], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 945, 'token_per_expert': {1: 386, 5: 384, 29: 14, 53: 17, 57: 27, 73: 11, 85: 10, 93: 10, 97: 14, 101: 11, 113: 11, 117: 28, 121: 22}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 42, 54, 78, 82, 86, 90, 106], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 959, 'token_per_expert': {2: 390, 6: 389, 14: 10, 18: 16, 22: 10, 26: 18, 42: 22, 54: 13, 78: 9, 82: 18, 86: 33, 90: 10, 106: 21}}
INFO 05-06 10:00:57.947920.947920 lmp.py:1833] [layer_moe_fused] layer=29 prefix: 0.443ms alloc: 0.389ms
INFO 05-06 10:00:57.947176.947176 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.198883056640625e-05 seconds
INFO 05-06 10:00:57.948467.948467 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007755756378173828s
INFO 05-06 10:00:57.949494.949494 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005483627319335938 seconds
INFO 05-06 10:00:57.951101.951101 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0016758441925048828s
INFO 05-06 10:00:57.958902.958902 lmp.py:1938] [layer_moe_fused] vllm triton time: 7.406ms (seq_len=128 cg=False)
DEBUG 05-06 10:00:57.959079.959079 cuda_h.py:27] end *layer_moe_fused cost 12.502 ms
DEBUG 05-06 10:00:57.959963.959963 cuda_h.py:27] end prefill_layer cost 18.033 ms
DEBUG 05-06 10:00:57.959680.959680 lmp.py:1388] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 10:00:57.960855.960855 cuda_h.py:27] end prefill_step cost 859.034 ms
INFO 05-06 10:00:57.960735.960735 lmp.py:1391] prefill time: 2.1602416038513184 seconds
INFO 05-06 10:00:57.966325.966325 lmp.py:1403] Static-KV prefill complete; seqlens set to 128.
WARNING 05-06 10:00:57.995719.995719 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:57.996960.996960 helper.py:35]   NaN count (hidden): 1081344
WARNING 05-06 10:00:57.997768.997768 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:57.997463.997463 helper.py:39]   NaN count (normed): 1081344
WARNING 05-06 10:00:58.003454.003454 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:58.003315.003315 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:58.003036.003036 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:00:58.120016.120016 cuda_h.py:27] end init_inputs_tokens cost 154.022 ms
DEBUG 05-06 10:00:58.120657.120657 lmp.py:1504] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:58.120123.120123 lmp.py:1510] ---- decode step 0 layer 0 ----
DEBUG 05-06 10:00:58.150118.150118 cuda_h.py:27] end decode_layer cost 30.491 ms
DEBUG 05-06 10:00:58.150016.150016 lmp.py:1510] ---- decode step 0 layer 1 ----
DEBUG 05-06 10:00:58.157926.157926 cuda_h.py:27] end decode_layer cost 6.420 ms
DEBUG 05-06 10:00:58.157512.157512 lmp.py:1510] ---- decode step 0 layer 2 ----
DEBUG 05-06 10:00:58.163460.163460 cuda_h.py:27] end decode_layer cost 5.959 ms
DEBUG 05-06 10:00:58.163879.163879 lmp.py:1510] ---- decode step 0 layer 3 ----
DEBUG 05-06 10:00:58.169673.169673 cuda_h.py:27] end decode_layer cost 6.091 ms
DEBUG 05-06 10:00:58.169854.169854 lmp.py:1510] ---- decode step 0 layer 4 ----
DEBUG 05-06 10:00:58.175809.175809 cuda_h.py:27] end decode_layer cost 5.789 ms
DEBUG 05-06 10:00:58.175229.175229 lmp.py:1510] ---- decode step 0 layer 5 ----
DEBUG 05-06 10:00:58.198457.198457 cuda_h.py:27] end decode_layer cost 23.229 ms
DEBUG 05-06 10:00:58.198691.198691 lmp.py:1510] ---- decode step 0 layer 6 ----
DEBUG 05-06 10:00:58.204539.204539 cuda_h.py:27] end decode_layer cost 5.745 ms
DEBUG 05-06 10:00:58.204767.204767 lmp.py:1510] ---- decode step 0 layer 7 ----
DEBUG 05-06 10:00:58.211548.211548 cuda_h.py:27] end decode_layer cost 6.328 ms
DEBUG 05-06 10:00:58.211014.211014 lmp.py:1510] ---- decode step 0 layer 8 ----
DEBUG 05-06 10:00:58.216450.216450 cuda_h.py:27] end decode_layer cost 5.687 ms
DEBUG 05-06 10:00:58.216200.216200 lmp.py:1510] ---- decode step 0 layer 9 ----
DEBUG 05-06 10:00:58.222883.222883 cuda_h.py:27] end decode_layer cost 5.764 ms
DEBUG 05-06 10:00:58.222826.222826 lmp.py:1510] ---- decode step 0 layer 10 ----
DEBUG 05-06 10:00:58.228031.228031 cuda_h.py:27] end decode_layer cost 5.727 ms
DEBUG 05-06 10:00:58.228126.228126 lmp.py:1510] ---- decode step 0 layer 11 ----
DEBUG 05-06 10:00:58.234002.234002 cuda_h.py:27] end decode_layer cost 6.187 ms
DEBUG 05-06 10:00:58.234945.234945 lmp.py:1510] ---- decode step 0 layer 12 ----
DEBUG 05-06 10:00:58.240319.240319 cuda_h.py:27] end decode_layer cost 5.641 ms
DEBUG 05-06 10:00:58.240355.240355 lmp.py:1510] ---- decode step 0 layer 13 ----
DEBUG 05-06 10:00:58.247881.247881 cuda_h.py:27] end decode_layer cost 6.806 ms
DEBUG 05-06 10:00:58.247115.247115 lmp.py:1510] ---- decode step 0 layer 14 ----
DEBUG 05-06 10:00:58.253632.253632 cuda_h.py:27] end decode_layer cost 5.747 ms
DEBUG 05-06 10:00:58.253382.253382 lmp.py:1510] ---- decode step 0 layer 15 ----
DEBUG 05-06 10:00:58.258639.258639 cuda_h.py:27] end decode_layer cost 5.695 ms
DEBUG 05-06 10:00:58.258151.258151 lmp.py:1510] ---- decode step 0 layer 16 ----
DEBUG 05-06 10:00:58.264958.264958 cuda_h.py:27] end decode_layer cost 5.678 ms
DEBUG 05-06 10:00:58.264947.264947 lmp.py:1510] ---- decode step 0 layer 17 ----
DEBUG 05-06 10:00:58.270924.270924 cuda_h.py:27] end decode_layer cost 6.051 ms
DEBUG 05-06 10:00:58.270198.270198 lmp.py:1510] ---- decode step 0 layer 18 ----
DEBUG 05-06 10:00:58.276742.276742 cuda_h.py:27] end decode_layer cost 5.767 ms
DEBUG 05-06 10:00:58.276969.276969 lmp.py:1510] ---- decode step 0 layer 19 ----
DEBUG 05-06 10:00:58.282267.282267 cuda_h.py:27] end decode_layer cost 5.725 ms
DEBUG 05-06 10:00:58.282541.282541 lmp.py:1510] ---- decode step 0 layer 20 ----
DEBUG 05-06 10:00:58.288949.288949 cuda_h.py:27] end decode_layer cost 5.841 ms
DEBUG 05-06 10:00:58.288560.288560 lmp.py:1510] ---- decode step 0 layer 21 ----
DEBUG 05-06 10:00:58.294516.294516 cuda_h.py:27] end decode_layer cost 5.789 ms
DEBUG 05-06 10:00:58.294789.294789 lmp.py:1510] ---- decode step 0 layer 22 ----
DEBUG 05-06 10:00:58.299601.299601 cuda_h.py:27] end decode_layer cost 5.647 ms
DEBUG 05-06 10:00:58.299352.299352 lmp.py:1510] ---- decode step 0 layer 23 ----
DEBUG 05-06 10:00:58.306553.306553 cuda_h.py:27] end decode_layer cost 6.217 ms
DEBUG 05-06 10:00:58.306019.306019 lmp.py:1510] ---- decode step 0 layer 24 ----
DEBUG 05-06 10:00:58.311555.311555 cuda_h.py:27] end decode_layer cost 5.726 ms
DEBUG 05-06 10:00:58.311187.311187 lmp.py:1510] ---- decode step 0 layer 25 ----
DEBUG 05-06 10:00:58.317361.317361 cuda_h.py:27] end decode_layer cost 5.774 ms
DEBUG 05-06 10:00:58.317826.317826 lmp.py:1510] ---- decode step 0 layer 26 ----
DEBUG 05-06 10:00:58.323285.323285 cuda_h.py:27] end decode_layer cost 5.773 ms
DEBUG 05-06 10:00:58.323977.323977 lmp.py:1510] ---- decode step 0 layer 27 ----
DEBUG 05-06 10:00:58.329217.329217 cuda_h.py:27] end decode_layer cost 5.788 ms
DEBUG 05-06 10:00:58.329160.329160 lmp.py:1510] ---- decode step 0 layer 28 ----
DEBUG 05-06 10:00:58.335592.335592 cuda_h.py:27] end decode_layer cost 5.789 ms
DEBUG 05-06 10:00:58.335919.335919 lmp.py:1510] ---- decode step 0 layer 29 ----
DEBUG 05-06 10:00:58.341434.341434 cuda_h.py:27] end decode_layer cost 6.096 ms
DEBUG 05-06 10:00:58.341214.341214 cuda_h.py:27] end decode_step cost 375.545 ms
INFO 05-06 10:00:58.341991.341991 lmp.py:1558] decode step 0 time: 0.3755950927734375 seconds
WARNING 05-06 10:00:58.342525.342525 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:58.342231.342231 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:00:58.343424.343424 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:58.343024.343024 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:00:58.348196.348196 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:58.348717.348717 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:58.348746.348746 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:00:58.350096.350096 cuda_h.py:27] end init_inputs_tokens cost 8.860 ms
DEBUG 05-06 10:00:58.351251.351251 lmp.py:1504] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:58.351259.351259 lmp.py:1510] ---- decode step 1 layer 0 ----
DEBUG 05-06 10:00:58.358856.358856 cuda_h.py:27] end decode_layer cost 7.346 ms
DEBUG 05-06 10:00:58.358124.358124 lmp.py:1510] ---- decode step 1 layer 1 ----
DEBUG 05-06 10:00:58.364188.364188 cuda_h.py:27] end decode_layer cost 6.255 ms
DEBUG 05-06 10:00:58.364827.364827 lmp.py:1510] ---- decode step 1 layer 2 ----
DEBUG 05-06 10:00:58.370088.370088 cuda_h.py:27] end decode_layer cost 5.804 ms
DEBUG 05-06 10:00:58.370792.370792 lmp.py:1510] ---- decode step 1 layer 3 ----
DEBUG 05-06 10:00:58.376757.376757 cuda_h.py:27] end decode_layer cost 6.076 ms
DEBUG 05-06 10:00:58.376607.376607 lmp.py:1510] ---- decode step 1 layer 4 ----
DEBUG 05-06 10:00:58.382065.382065 cuda_h.py:27] end decode_layer cost 5.948 ms
DEBUG 05-06 10:00:58.382153.382153 lmp.py:1510] ---- decode step 1 layer 5 ----
DEBUG 05-06 10:00:58.389426.389426 cuda_h.py:27] end decode_layer cost 6.374 ms
DEBUG 05-06 10:00:58.389991.389991 lmp.py:1510] ---- decode step 1 layer 6 ----
DEBUG 05-06 10:00:58.395151.395151 cuda_h.py:27] end decode_layer cost 5.939 ms
DEBUG 05-06 10:00:58.395716.395716 lmp.py:1510] ---- decode step 1 layer 7 ----
DEBUG 05-06 10:00:58.401493.401493 cuda_h.py:27] end decode_layer cost 6.009 ms
DEBUG 05-06 10:00:58.401251.401251 lmp.py:1510] ---- decode step 1 layer 8 ----
DEBUG 05-06 10:00:58.407914.407914 cuda_h.py:27] end decode_layer cost 5.959 ms
DEBUG 05-06 10:00:58.407002.407002 lmp.py:1510] ---- decode step 1 layer 9 ----
DEBUG 05-06 10:00:58.413695.413695 cuda_h.py:27] end decode_layer cost 6.051 ms
DEBUG 05-06 10:00:58.413022.413022 lmp.py:1510] ---- decode step 1 layer 10 ----
DEBUG 05-06 10:00:58.419764.419764 cuda_h.py:27] end decode_layer cost 5.948 ms
DEBUG 05-06 10:00:58.419787.419787 lmp.py:1510] ---- decode step 1 layer 11 ----
DEBUG 05-06 10:00:58.425706.425706 cuda_h.py:27] end decode_layer cost 6.289 ms
DEBUG 05-06 10:00:58.426841.426841 lmp.py:1510] ---- decode step 1 layer 12 ----
DEBUG 05-06 10:00:58.431039.431039 cuda_h.py:27] end decode_layer cost 5.897 ms
DEBUG 05-06 10:00:58.432842.432842 lmp.py:1510] ---- decode step 1 layer 13 ----
DEBUG 05-06 10:00:58.437312.437312 cuda_h.py:27] end decode_layer cost 5.923 ms
DEBUG 05-06 10:00:58.438586.438586 lmp.py:1510] ---- decode step 1 layer 14 ----
DEBUG 05-06 10:00:58.443750.443750 cuda_h.py:27] end decode_layer cost 5.871 ms
DEBUG 05-06 10:00:58.443407.443407 lmp.py:1510] ---- decode step 1 layer 15 ----
DEBUG 05-06 10:00:58.449451.449451 cuda_h.py:27] end decode_layer cost 5.854 ms
DEBUG 05-06 10:00:58.449440.449440 lmp.py:1510] ---- decode step 1 layer 16 ----
DEBUG 05-06 10:00:58.455514.455514 cuda_h.py:27] end decode_layer cost 5.945 ms
DEBUG 05-06 10:00:58.455363.455363 lmp.py:1510] ---- decode step 1 layer 17 ----
DEBUG 05-06 10:00:58.462640.462640 cuda_h.py:27] end decode_layer cost 6.271 ms
DEBUG 05-06 10:00:58.462013.462013 lmp.py:1510] ---- decode step 1 layer 18 ----
DEBUG 05-06 10:00:58.468343.468343 cuda_h.py:27] end decode_layer cost 6.244 ms
DEBUG 05-06 10:00:58.468969.468969 lmp.py:1510] ---- decode step 1 layer 19 ----
DEBUG 05-06 10:00:58.474845.474845 cuda_h.py:27] end decode_layer cost 5.976 ms
DEBUG 05-06 10:00:58.474595.474595 lmp.py:1510] ---- decode step 1 layer 20 ----
DEBUG 05-06 10:00:58.480796.480796 cuda_h.py:27] end decode_layer cost 6.005 ms
DEBUG 05-06 10:00:58.480169.480169 lmp.py:1510] ---- decode step 1 layer 21 ----
DEBUG 05-06 10:00:58.486482.486482 cuda_h.py:27] end decode_layer cost 6.157 ms
DEBUG 05-06 10:00:58.486808.486808 lmp.py:1510] ---- decode step 1 layer 22 ----
DEBUG 05-06 10:00:58.492737.492737 cuda_h.py:27] end decode_layer cost 5.979 ms
DEBUG 05-06 10:00:58.493587.493587 lmp.py:1510] ---- decode step 1 layer 23 ----
DEBUG 05-06 10:00:58.499841.499841 cuda_h.py:27] end decode_layer cost 6.184 ms
DEBUG 05-06 10:00:58.499830.499830 lmp.py:1510] ---- decode step 1 layer 24 ----
DEBUG 05-06 10:00:58.505866.505866 cuda_h.py:27] end decode_layer cost 5.813 ms
DEBUG 05-06 10:00:58.505854.505854 lmp.py:1510] ---- decode step 1 layer 25 ----
DEBUG 05-06 10:00:58.511574.511574 cuda_h.py:27] end decode_layer cost 5.861 ms
DEBUG 05-06 10:00:58.511848.511848 lmp.py:1510] ---- decode step 1 layer 26 ----
DEBUG 05-06 10:00:58.517285.517285 cuda_h.py:27] end decode_layer cost 5.933 ms
DEBUG 05-06 10:00:58.517797.517797 lmp.py:1510] ---- decode step 1 layer 27 ----
DEBUG 05-06 10:00:58.522874.522874 cuda_h.py:27] end decode_layer cost 5.843 ms
DEBUG 05-06 10:00:58.523862.523862 lmp.py:1510] ---- decode step 1 layer 28 ----
DEBUG 05-06 10:00:58.528932.528932 cuda_h.py:27] end decode_layer cost 5.627 ms
DEBUG 05-06 10:00:58.528444.528444 lmp.py:1510] ---- decode step 1 layer 29 ----
DEBUG 05-06 10:00:58.534018.534018 cuda_h.py:27] end decode_layer cost 6.069 ms
DEBUG 05-06 10:00:58.534200.534200 cuda_h.py:27] end decode_step cost 192.755 ms
INFO 05-06 10:00:58.534824.534824 lmp.py:1558] decode step 1 time: 0.1927957534790039 seconds
WARNING 05-06 10:00:58.535148.535148 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:58.535795.535795 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:00:58.535305.535305 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:58.535224.535224 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:00:58.541440.541440 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:58.541617.541617 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:58.541248.541248 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 10:00:58.541722.541722 helper.py:80] WARNING: Logits have extreme values: min=-652.00, max=1152.00
WARNING 05-06 10:00:58.541769.541769 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 10:00:58.543912.543912 cuda_h.py:27] end init_inputs_tokens cost 8.496 ms
DEBUG 05-06 10:00:58.543093.543093 lmp.py:1504] decode step 2 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:58.543055.543055 lmp.py:1510] ---- decode step 2 layer 0 ----
DEBUG 05-06 10:00:58.549929.549929 cuda_h.py:27] end decode_layer cost 6.114 ms
DEBUG 05-06 10:00:58.549917.549917 lmp.py:1510] ---- decode step 2 layer 1 ----
DEBUG 05-06 10:00:58.555088.555088 cuda_h.py:27] end decode_layer cost 5.860 ms
DEBUG 05-06 10:00:58.555799.555799 lmp.py:1510] ---- decode step 2 layer 2 ----
DEBUG 05-06 10:00:58.561353.561353 cuda_h.py:27] end decode_layer cost 5.669 ms
DEBUG 05-06 10:00:58.561057.561057 lmp.py:1510] ---- decode step 2 layer 3 ----
DEBUG 05-06 10:00:58.567685.567685 cuda_h.py:27] end decode_layer cost 5.687 ms
DEBUG 05-06 10:00:58.567005.567005 lmp.py:1510] ---- decode step 2 layer 4 ----
DEBUG 05-06 10:00:58.573096.573096 cuda_h.py:27] end decode_layer cost 5.889 ms
DEBUG 05-06 10:00:58.573708.573708 lmp.py:1510] ---- decode step 2 layer 5 ----
DEBUG 05-06 10:00:58.579987.579987 cuda_h.py:27] end decode_layer cost 6.168 ms
DEBUG 05-06 10:00:58.579215.579215 lmp.py:1510] ---- decode step 2 layer 6 ----
DEBUG 05-06 10:00:58.585606.585606 cuda_h.py:27] end decode_layer cost 5.760 ms
DEBUG 05-06 10:00:58.585310.585310 lmp.py:1510] ---- decode step 2 layer 7 ----
DEBUG 05-06 10:00:58.590972.590972 cuda_h.py:27] end decode_layer cost 5.713 ms
DEBUG 05-06 10:00:58.591889.591889 lmp.py:1510] ---- decode step 2 layer 8 ----
DEBUG 05-06 10:00:58.596660.596660 cuda_h.py:27] end decode_layer cost 5.829 ms
DEBUG 05-06 10:00:58.596649.596649 lmp.py:1510] ---- decode step 2 layer 9 ----
DEBUG 05-06 10:00:58.602692.602692 cuda_h.py:27] end decode_layer cost 5.818 ms
DEBUG 05-06 10:00:58.602919.602919 lmp.py:1510] ---- decode step 2 layer 10 ----
DEBUG 05-06 10:00:58.609702.609702 cuda_h.py:27] end decode_layer cost 6.954 ms
DEBUG 05-06 10:00:58.609579.609579 lmp.py:1510] ---- decode step 2 layer 11 ----
DEBUG 05-06 10:00:58.616457.616457 cuda_h.py:27] end decode_layer cost 6.223 ms
DEBUG 05-06 10:00:58.616830.616830 lmp.py:1510] ---- decode step 2 layer 12 ----
DEBUG 05-06 10:00:58.622323.622323 cuda_h.py:27] end decode_layer cost 5.835 ms
DEBUG 05-06 10:00:58.622266.622266 lmp.py:1510] ---- decode step 2 layer 13 ----
DEBUG 05-06 10:00:58.627351.627351 cuda_h.py:27] end decode_layer cost 5.709 ms
DEBUG 05-06 10:00:58.627817.627817 lmp.py:1510] ---- decode step 2 layer 14 ----
DEBUG 05-06 10:00:58.633575.633575 cuda_h.py:27] end decode_layer cost 5.818 ms
DEBUG 05-06 10:00:58.633233.633233 lmp.py:1510] ---- decode step 2 layer 15 ----
DEBUG 05-06 10:00:58.639317.639317 cuda_h.py:27] end decode_layer cost 5.884 ms
DEBUG 05-06 10:00:58.639452.639452 lmp.py:1510] ---- decode step 2 layer 16 ----
DEBUG 05-06 10:00:58.645177.645177 cuda_h.py:27] end decode_layer cost 5.829 ms
DEBUG 05-06 10:00:58.645835.645835 lmp.py:1510] ---- decode step 2 layer 17 ----
DEBUG 05-06 10:00:58.651673.651673 cuda_h.py:27] end decode_layer cost 6.229 ms
DEBUG 05-06 10:00:58.651999.651999 lmp.py:1510] ---- decode step 2 layer 18 ----
DEBUG 05-06 10:00:58.657749.657749 cuda_h.py:27] end decode_layer cost 5.778 ms
DEBUG 05-06 10:00:58.657453.657453 lmp.py:1510] ---- decode step 2 layer 19 ----
DEBUG 05-06 10:00:58.663693.663693 cuda_h.py:27] end decode_layer cost 5.963 ms
DEBUG 05-06 10:00:58.663828.663828 lmp.py:1510] ---- decode step 2 layer 20 ----
DEBUG 05-06 10:00:58.669584.669584 cuda_h.py:27] end decode_layer cost 5.783 ms
DEBUG 05-06 10:00:58.669812.669812 lmp.py:1510] ---- decode step 2 layer 21 ----
DEBUG 05-06 10:00:58.675119.675119 cuda_h.py:27] end decode_layer cost 5.802 ms
DEBUG 05-06 10:00:58.675392.675392 lmp.py:1510] ---- decode step 2 layer 22 ----
DEBUG 05-06 10:00:58.681721.681721 cuda_h.py:27] end decode_layer cost 5.853 ms
DEBUG 05-06 10:00:58.681187.681187 lmp.py:1510] ---- decode step 2 layer 23 ----
DEBUG 05-06 10:00:58.687415.687415 cuda_h.py:27] end decode_layer cost 6.025 ms
DEBUG 05-06 10:00:58.687642.687642 lmp.py:1510] ---- decode step 2 layer 24 ----
DEBUG 05-06 10:00:58.693684.693684 cuda_h.py:27] end decode_layer cost 5.782 ms
DEBUG 05-06 10:00:58.693342.693342 lmp.py:1510] ---- decode step 2 layer 25 ----
DEBUG 05-06 10:00:58.699253.699253 cuda_h.py:27] end decode_layer cost 5.862 ms
DEBUG 05-06 10:00:58.699434.699434 lmp.py:1510] ---- decode step 2 layer 26 ----
DEBUG 05-06 10:00:58.705623.705623 cuda_h.py:27] end decode_layer cost 5.820 ms
DEBUG 05-06 10:00:58.705950.705950 lmp.py:1510] ---- decode step 2 layer 27 ----
DEBUG 05-06 10:00:58.711860.711860 cuda_h.py:27] end decode_layer cost 5.826 ms
DEBUG 05-06 10:00:58.711611.711611 lmp.py:1510] ---- decode step 2 layer 28 ----
DEBUG 05-06 10:00:58.716366.716366 cuda_h.py:27] end decode_layer cost 5.746 ms
DEBUG 05-06 10:00:58.716739.716739 lmp.py:1510] ---- decode step 2 layer 29 ----
DEBUG 05-06 10:00:58.723394.723394 cuda_h.py:27] end decode_layer cost 6.094 ms
DEBUG 05-06 10:00:58.723907.723907 cuda_h.py:27] end decode_step cost 188.132 ms
INFO 05-06 10:00:58.723623.723623 lmp.py:1558] decode step 2 time: 0.1881723403930664 seconds
WARNING 05-06 10:00:58.723702.723702 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:58.723641.723641 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:00:58.724588.724588 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:58.724850.724850 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:00:58.729563.729563 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:58.729746.729746 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:58.729761.729761 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:00:58.731871.731871 cuda_h.py:27] end init_inputs_tokens cost 8.124 ms
DEBUG 05-06 10:00:58.731244.731244 lmp.py:1504] decode step 3 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:58.731537.731537 lmp.py:1510] ---- decode step 3 layer 0 ----
DEBUG 05-06 10:00:58.737262.737262 cuda_h.py:27] end decode_layer cost 5.829 ms
DEBUG 05-06 10:00:58.737920.737920 lmp.py:1510] ---- decode step 3 layer 1 ----
DEBUG 05-06 10:00:58.744059.744059 cuda_h.py:27] end decode_layer cost 6.907 ms
DEBUG 05-06 10:00:58.744193.744193 lmp.py:1510] ---- decode step 3 layer 2 ----
DEBUG 05-06 10:00:58.750675.750675 cuda_h.py:27] end decode_layer cost 5.686 ms
DEBUG 05-06 10:00:58.750426.750426 lmp.py:1510] ---- decode step 3 layer 3 ----
DEBUG 05-06 10:00:58.755902.755902 cuda_h.py:27] end decode_layer cost 5.716 ms
DEBUG 05-06 10:00:58.755891.755891 lmp.py:1510] ---- decode step 3 layer 4 ----
DEBUG 05-06 10:00:58.761987.761987 cuda_h.py:27] end decode_layer cost 5.647 ms
DEBUG 05-06 10:00:58.761738.761738 lmp.py:1510] ---- decode step 3 layer 5 ----
DEBUG 05-06 10:00:58.767399.767399 cuda_h.py:27] end decode_layer cost 6.098 ms
DEBUG 05-06 10:00:58.767342.767342 lmp.py:1510] ---- decode step 3 layer 6 ----
DEBUG 05-06 10:00:58.773049.773049 cuda_h.py:27] end decode_layer cost 5.886 ms
DEBUG 05-06 10:00:58.773522.773522 lmp.py:1510] ---- decode step 3 layer 7 ----
DEBUG 05-06 10:00:58.779869.779869 cuda_h.py:27] end decode_layer cost 5.832 ms
DEBUG 05-06 10:00:58.779110.779110 lmp.py:1510] ---- decode step 3 layer 8 ----
DEBUG 05-06 10:00:58.785087.785087 cuda_h.py:27] end decode_layer cost 5.840 ms
DEBUG 05-06 10:00:58.785268.785268 lmp.py:1510] ---- decode step 3 layer 9 ----
DEBUG 05-06 10:00:58.791672.791672 cuda_h.py:27] end decode_layer cost 5.734 ms
DEBUG 05-06 10:00:58.791423.791423 lmp.py:1510] ---- decode step 3 layer 10 ----
DEBUG 05-06 10:00:58.797956.797956 cuda_h.py:27] end decode_layer cost 5.828 ms
DEBUG 05-06 10:00:58.797660.797660 lmp.py:1510] ---- decode step 3 layer 11 ----
DEBUG 05-06 10:00:58.803597.803597 cuda_h.py:27] end decode_layer cost 6.021 ms
DEBUG 05-06 10:00:58.803493.803493 lmp.py:1510] ---- decode step 3 layer 12 ----
DEBUG 05-06 10:00:58.809751.809751 cuda_h.py:27] end decode_layer cost 5.906 ms
DEBUG 05-06 10:00:58.809601.809601 lmp.py:1510] ---- decode step 3 layer 13 ----
DEBUG 05-06 10:00:58.815477.815477 cuda_h.py:27] end decode_layer cost 5.975 ms
DEBUG 05-06 10:00:58.815518.815518 lmp.py:1510] ---- decode step 3 layer 14 ----
DEBUG 05-06 10:00:58.821261.821261 cuda_h.py:27] end decode_layer cost 5.947 ms
DEBUG 05-06 10:00:58.821588.821588 lmp.py:1510] ---- decode step 3 layer 15 ----
DEBUG 05-06 10:00:58.827077.827077 cuda_h.py:27] end decode_layer cost 5.901 ms
DEBUG 05-06 10:00:58.827741.827741 lmp.py:1510] ---- decode step 3 layer 16 ----
DEBUG 05-06 10:00:58.833232.833232 cuda_h.py:27] end decode_layer cost 5.937 ms
DEBUG 05-06 10:00:58.833128.833128 lmp.py:1510] ---- decode step 3 layer 17 ----
DEBUG 05-06 10:00:58.839952.839952 cuda_h.py:27] end decode_layer cost 6.219 ms
DEBUG 05-06 10:00:58.839577.839577 lmp.py:1510] ---- decode step 3 layer 18 ----
DEBUG 05-06 10:00:58.845123.845123 cuda_h.py:27] end decode_layer cost 5.803 ms
DEBUG 05-06 10:00:58.845973.845973 lmp.py:1510] ---- decode step 3 layer 19 ----
DEBUG 05-06 10:00:58.851620.851620 cuda_h.py:27] end decode_layer cost 5.877 ms
DEBUG 05-06 10:00:58.851708.851708 lmp.py:1510] ---- decode step 3 layer 20 ----
DEBUG 05-06 10:00:58.857970.857970 cuda_h.py:27] end decode_layer cost 5.839 ms
DEBUG 05-06 10:00:58.857343.857343 lmp.py:1510] ---- decode step 3 layer 21 ----
DEBUG 05-06 10:00:58.863952.863952 cuda_h.py:27] end decode_layer cost 5.920 ms
DEBUG 05-06 10:00:58.863325.863325 lmp.py:1510] ---- decode step 3 layer 22 ----
DEBUG 05-06 10:00:58.869044.869044 cuda_h.py:27] end decode_layer cost 5.824 ms
DEBUG 05-06 10:00:58.869086.869086 lmp.py:1510] ---- decode step 3 layer 23 ----
DEBUG 05-06 10:00:58.875431.875431 cuda_h.py:27] end decode_layer cost 6.146 ms
DEBUG 05-06 10:00:58.875996.875996 lmp.py:1510] ---- decode step 3 layer 24 ----
DEBUG 05-06 10:00:58.881919.881919 cuda_h.py:27] end decode_layer cost 6.396 ms
DEBUG 05-06 10:00:58.881769.881769 lmp.py:1510] ---- decode step 3 layer 25 ----
DEBUG 05-06 10:00:58.888793.888793 cuda_h.py:27] end decode_layer cost 6.049 ms
DEBUG 05-06 10:00:58.888073.888073 lmp.py:1510] ---- decode step 3 layer 26 ----
DEBUG 05-06 10:00:58.893556.893556 cuda_h.py:27] end decode_layer cost 5.896 ms
DEBUG 05-06 10:00:58.894167.894167 lmp.py:1510] ---- decode step 3 layer 27 ----
DEBUG 05-06 10:00:58.900653.900653 cuda_h.py:27] end decode_layer cost 6.004 ms
DEBUG 05-06 10:00:58.900218.900218 lmp.py:1510] ---- decode step 3 layer 28 ----
DEBUG 05-06 10:00:58.906901.906901 cuda_h.py:27] end decode_layer cost 5.938 ms
DEBUG 05-06 10:00:58.906704.906704 lmp.py:1510] ---- decode step 3 layer 29 ----
DEBUG 05-06 10:00:58.912348.912348 cuda_h.py:27] end decode_layer cost 6.156 ms
DEBUG 05-06 10:00:58.912768.912768 cuda_h.py:27] end decode_step cost 189.199 ms
INFO 05-06 10:00:58.912246.912246 lmp.py:1558] decode step 3 time: 0.1892383098602295 seconds
WARNING 05-06 10:00:58.912319.912319 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:58.912533.912533 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:00:58.913348.913348 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:58.913133.913133 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:00:58.918351.918351 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:58.918766.918766 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:58.918801.918801 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 10:00:58.919335.919335 helper.py:80] WARNING: Logits have extreme values: min=-772.00, max=1128.00
WARNING 05-06 10:00:58.919927.919927 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 10:00:58.921786.921786 cuda_h.py:27] end init_inputs_tokens cost 8.532 ms
DEBUG 05-06 10:00:58.921159.921159 lmp.py:1504] decode step 4 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:58.921975.921975 lmp.py:1510] ---- decode step 4 layer 0 ----
DEBUG 05-06 10:00:58.926313.926313 cuda_h.py:27] end decode_layer cost 5.718 ms
DEBUG 05-06 10:00:58.926540.926540 lmp.py:1510] ---- decode step 4 layer 1 ----
DEBUG 05-06 10:00:58.932866.932866 cuda_h.py:27] end decode_layer cost 5.956 ms
DEBUG 05-06 10:00:58.932570.932570 lmp.py:1510] ---- decode step 4 layer 2 ----
DEBUG 05-06 10:00:58.938877.938877 cuda_h.py:27] end decode_layer cost 5.802 ms
DEBUG 05-06 10:00:58.938819.938819 lmp.py:1510] ---- decode step 4 layer 3 ----
DEBUG 05-06 10:00:58.944391.944391 cuda_h.py:27] end decode_layer cost 5.786 ms
DEBUG 05-06 10:00:58.944048.944048 lmp.py:1510] ---- decode step 4 layer 4 ----
DEBUG 05-06 10:00:58.950899.950899 cuda_h.py:27] end decode_layer cost 5.606 ms
DEBUG 05-06 10:00:58.950603.950603 lmp.py:1510] ---- decode step 4 layer 5 ----
DEBUG 05-06 10:00:58.956009.956009 cuda_h.py:27] end decode_layer cost 5.980 ms
DEBUG 05-06 10:00:58.956905.956905 lmp.py:1510] ---- decode step 4 layer 6 ----
DEBUG 05-06 10:00:58.962213.962213 cuda_h.py:27] end decode_layer cost 5.838 ms
DEBUG 05-06 10:00:58.962348.962348 lmp.py:1510] ---- decode step 4 layer 7 ----
DEBUG 05-06 10:00:58.968926.968926 cuda_h.py:27] end decode_layer cost 5.792 ms
DEBUG 05-06 10:00:58.968345.968345 lmp.py:1510] ---- decode step 4 layer 8 ----
DEBUG 05-06 10:00:58.973326.973326 cuda_h.py:27] end decode_layer cost 5.737 ms
DEBUG 05-06 10:00:58.974268.974268 lmp.py:1510] ---- decode step 4 layer 9 ----
DEBUG 05-06 10:00:58.979187.979187 cuda_h.py:27] end decode_layer cost 5.867 ms
DEBUG 05-06 10:00:58.979083.979083 lmp.py:1510] ---- decode step 4 layer 10 ----
DEBUG 05-06 10:00:58.985010.985010 cuda_h.py:27] end decode_layer cost 5.733 ms
DEBUG 05-06 10:00:58.985953.985953 lmp.py:1510] ---- decode step 4 layer 11 ----
DEBUG 05-06 10:00:58.991586.991586 cuda_h.py:27] end decode_layer cost 6.042 ms
DEBUG 05-06 10:00:58.991720.991720 lmp.py:1510] ---- decode step 4 layer 12 ----
DEBUG 05-06 10:00:58.997521.997521 cuda_h.py:27] end decode_layer cost 5.710 ms
DEBUG 05-06 10:00:58.997987.997987 lmp.py:1510] ---- decode step 4 layer 13 ----
DEBUG 05-06 10:00:59.003723.003723 cuda_h.py:27] end decode_layer cost 5.768 ms
DEBUG 05-06 10:00:59.003142.003142 lmp.py:1510] ---- decode step 4 layer 14 ----
DEBUG 05-06 10:00:59.009158.009158 cuda_h.py:27] end decode_layer cost 5.587 ms
DEBUG 05-06 10:00:59.009862.009862 lmp.py:1510] ---- decode step 4 layer 15 ----
DEBUG 05-06 10:00:59.014226.014226 cuda_h.py:27] end decode_layer cost 5.739 ms
DEBUG 05-06 10:00:59.014553.014553 lmp.py:1510] ---- decode step 4 layer 16 ----
DEBUG 05-06 10:00:59.021871.021871 cuda_h.py:27] end decode_layer cost 6.338 ms
DEBUG 05-06 10:00:59.021668.021668 lmp.py:1510] ---- decode step 4 layer 17 ----
DEBUG 05-06 10:00:59.027545.027545 cuda_h.py:27] end decode_layer cost 6.012 ms
DEBUG 05-06 10:00:59.027249.027249 lmp.py:1510] ---- decode step 4 layer 18 ----
DEBUG 05-06 10:00:59.033743.033743 cuda_h.py:27] end decode_layer cost 5.659 ms
DEBUG 05-06 10:00:59.033401.033401 lmp.py:1510] ---- decode step 4 layer 19 ----
DEBUG 05-06 10:00:59.038199.038199 cuda_h.py:27] end decode_layer cost 5.812 ms
DEBUG 05-06 10:00:59.039380.039380 lmp.py:1510] ---- decode step 4 layer 20 ----
DEBUG 05-06 10:00:59.044182.044182 cuda_h.py:27] end decode_layer cost 5.746 ms
DEBUG 05-06 10:00:59.044886.044886 lmp.py:1510] ---- decode step 4 layer 21 ----
DEBUG 05-06 10:00:59.050334.050334 cuda_h.py:27] end decode_layer cost 5.660 ms
DEBUG 05-06 10:00:59.050276.050276 lmp.py:1510] ---- decode step 4 layer 22 ----
DEBUG 05-06 10:00:59.056580.056580 cuda_h.py:27] end decode_layer cost 5.694 ms
DEBUG 05-06 10:00:59.056522.056522 lmp.py:1510] ---- decode step 4 layer 23 ----
DEBUG 05-06 10:00:59.062827.062827 cuda_h.py:27] end decode_layer cost 5.941 ms
DEBUG 05-06 10:00:59.062008.062008 lmp.py:1510] ---- decode step 4 layer 24 ----
DEBUG 05-06 10:00:59.068741.068741 cuda_h.py:27] end decode_layer cost 5.660 ms
DEBUG 05-06 10:00:59.068730.068730 lmp.py:1510] ---- decode step 4 layer 25 ----
DEBUG 05-06 10:00:59.073245.073245 cuda_h.py:27] end decode_layer cost 5.674 ms
DEBUG 05-06 10:00:59.073094.073094 lmp.py:1510] ---- decode step 4 layer 26 ----
DEBUG 05-06 10:00:59.079552.079552 cuda_h.py:27] end decode_layer cost 5.738 ms
DEBUG 05-06 10:00:59.079005.079005 lmp.py:1510] ---- decode step 4 layer 27 ----
DEBUG 05-06 10:00:59.085741.085741 cuda_h.py:27] end decode_layer cost 5.768 ms
DEBUG 05-06 10:00:59.085969.085969 lmp.py:1510] ---- decode step 4 layer 28 ----
DEBUG 05-06 10:00:59.091058.091058 cuda_h.py:27] end decode_layer cost 5.642 ms
DEBUG 05-06 10:00:59.091008.091008 lmp.py:1510] ---- decode step 4 layer 29 ----
DEBUG 05-06 10:00:59.097525.097525 cuda_h.py:27] end decode_layer cost 6.133 ms
DEBUG 05-06 10:00:59.097031.097031 cuda_h.py:27] end decode_step cost 184.947 ms
INFO 05-06 10:00:59.097079.097079 lmp.py:1558] decode step 4 time: 0.18498468399047852 seconds
WARNING 05-06 10:00:59.097521.097521 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:59.097181.097181 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:00:59.098694.098694 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:59.098519.098519 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:00:59.103212.103212 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:59.103865.103865 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:59.103291.103291 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:00:59.105671.105671 cuda_h.py:27] end init_inputs_tokens cost 7.747 ms
DEBUG 05-06 10:00:59.105468.105468 lmp.py:1504] decode step 5 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:59.105615.105615 lmp.py:1510] ---- decode step 5 layer 0 ----
DEBUG 05-06 10:00:59.111135.111135 cuda_h.py:27] end decode_layer cost 5.642 ms
DEBUG 05-06 10:00:59.111362.111362 lmp.py:1510] ---- decode step 5 layer 1 ----
DEBUG 05-06 10:00:59.116040.116040 cuda_h.py:27] end decode_layer cost 5.795 ms
DEBUG 05-06 10:00:59.116221.116221 lmp.py:1510] ---- decode step 5 layer 2 ----
DEBUG 05-06 10:00:59.122281.122281 cuda_h.py:27] end decode_layer cost 5.725 ms
DEBUG 05-06 10:00:59.122746.122746 lmp.py:1510] ---- decode step 5 layer 3 ----
DEBUG 05-06 10:00:59.128567.128567 cuda_h.py:27] end decode_layer cost 5.899 ms
DEBUG 05-06 10:00:59.128463.128463 lmp.py:1510] ---- decode step 5 layer 4 ----
DEBUG 05-06 10:00:59.134483.134483 cuda_h.py:27] end decode_layer cost 5.731 ms
DEBUG 05-06 10:00:59.134948.134948 lmp.py:1510] ---- decode step 5 layer 5 ----
DEBUG 05-06 10:00:59.140498.140498 cuda_h.py:27] end decode_layer cost 6.121 ms
DEBUG 05-06 10:00:59.140633.140633 lmp.py:1510] ---- decode step 5 layer 6 ----
DEBUG 05-06 10:00:59.146116.146116 cuda_h.py:27] end decode_layer cost 5.722 ms
DEBUG 05-06 10:00:59.146674.146674 lmp.py:1510] ---- decode step 5 layer 7 ----
DEBUG 05-06 10:00:59.152449.152449 cuda_h.py:27] end decode_layer cost 5.725 ms
DEBUG 05-06 10:00:59.152345.152345 lmp.py:1510] ---- decode step 5 layer 8 ----
DEBUG 05-06 10:00:59.158301.158301 cuda_h.py:27] end decode_layer cost 6.386 ms
DEBUG 05-06 10:00:59.158290.158290 lmp.py:1510] ---- decode step 5 layer 9 ----
DEBUG 05-06 10:00:59.164858.164858 cuda_h.py:27] end decode_layer cost 5.890 ms
DEBUG 05-06 10:00:59.164231.164231 lmp.py:1510] ---- decode step 5 layer 10 ----
DEBUG 05-06 10:00:59.170856.170856 cuda_h.py:27] end decode_layer cost 5.615 ms
DEBUG 05-06 10:00:59.170991.170991 lmp.py:1510] ---- decode step 5 layer 11 ----
DEBUG 05-06 10:00:59.176096.176096 cuda_h.py:27] end decode_layer cost 6.110 ms
DEBUG 05-06 10:00:59.176992.176992 lmp.py:1510] ---- decode step 5 layer 12 ----
DEBUG 05-06 10:00:59.182917.182917 cuda_h.py:27] end decode_layer cost 5.661 ms
DEBUG 05-06 10:00:59.182429.182429 lmp.py:1510] ---- decode step 5 layer 13 ----
DEBUG 05-06 10:00:59.188983.188983 cuda_h.py:27] end decode_layer cost 5.843 ms
DEBUG 05-06 10:00:59.188548.188548 lmp.py:1510] ---- decode step 5 layer 14 ----
DEBUG 05-06 10:00:59.193586.193586 cuda_h.py:27] end decode_layer cost 5.674 ms
DEBUG 05-06 10:00:59.193052.193052 lmp.py:1510] ---- decode step 5 layer 15 ----
DEBUG 05-06 10:00:59.199518.199518 cuda_h.py:27] end decode_layer cost 5.815 ms
DEBUG 05-06 10:00:59.199507.199507 lmp.py:1510] ---- decode step 5 layer 16 ----
DEBUG 05-06 10:00:59.205682.205682 cuda_h.py:27] end decode_layer cost 5.635 ms
DEBUG 05-06 10:00:59.205671.205671 lmp.py:1510] ---- decode step 5 layer 17 ----
DEBUG 05-06 10:00:59.211631.211631 cuda_h.py:27] end decode_layer cost 5.897 ms
DEBUG 05-06 10:00:59.211673.211673 lmp.py:1510] ---- decode step 5 layer 18 ----
DEBUG 05-06 10:00:59.217040.217040 cuda_h.py:27] end decode_layer cost 5.636 ms
DEBUG 05-06 10:00:59.217652.217652 lmp.py:1510] ---- decode step 5 layer 19 ----
DEBUG 05-06 10:00:59.223672.223672 cuda_h.py:27] end decode_layer cost 5.731 ms
DEBUG 05-06 10:00:59.223422.223422 lmp.py:1510] ---- decode step 5 layer 20 ----
DEBUG 05-06 10:00:59.228291.228291 cuda_h.py:27] end decode_layer cost 5.585 ms
DEBUG 05-06 10:00:59.228850.228850 lmp.py:1510] ---- decode step 5 layer 21 ----
DEBUG 05-06 10:00:59.234826.234826 cuda_h.py:27] end decode_layer cost 5.803 ms
DEBUG 05-06 10:00:59.234815.234815 lmp.py:1510] ---- decode step 5 layer 22 ----
DEBUG 05-06 10:00:59.240017.240017 cuda_h.py:27] end decode_layer cost 5.656 ms
DEBUG 05-06 10:00:59.240529.240529 lmp.py:1510] ---- decode step 5 layer 23 ----
DEBUG 05-06 10:00:59.246916.246916 cuda_h.py:27] end decode_layer cost 6.001 ms
DEBUG 05-06 10:00:59.246859.246859 lmp.py:1510] ---- decode step 5 layer 24 ----
DEBUG 05-06 10:00:59.251105.251105 cuda_h.py:27] end decode_layer cost 5.582 ms
DEBUG 05-06 10:00:59.252379.252379 lmp.py:1510] ---- decode step 5 layer 25 ----
DEBUG 05-06 10:00:59.257761.257761 cuda_h.py:27] end decode_layer cost 5.682 ms
DEBUG 05-06 10:00:59.257419.257419 lmp.py:1510] ---- decode step 5 layer 26 ----
DEBUG 05-06 10:00:59.263833.263833 cuda_h.py:27] end decode_layer cost 5.635 ms
DEBUG 05-06 10:00:59.263253.263253 lmp.py:1510] ---- decode step 5 layer 27 ----
DEBUG 05-06 10:00:59.269019.269019 cuda_h.py:27] end decode_layer cost 5.685 ms
DEBUG 05-06 10:00:59.269485.269485 lmp.py:1510] ---- decode step 5 layer 28 ----
DEBUG 05-06 10:00:59.274797.274797 cuda_h.py:27] end decode_layer cost 5.560 ms
DEBUG 05-06 10:00:59.274548.274548 lmp.py:1510] ---- decode step 5 layer 29 ----
DEBUG 05-06 10:00:59.281575.281575 cuda_h.py:27] end decode_layer cost 6.158 ms
DEBUG 05-06 10:00:59.281181.281181 cuda_h.py:27] end decode_step cost 183.585 ms
INFO 05-06 10:00:59.281182.281182 lmp.py:1558] decode step 5 time: 0.1836245059967041 seconds
WARNING 05-06 10:00:59.281777.281777 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:59.281694.281694 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:00:59.282060.282060 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:59.282269.282269 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:00:59.287070.287070 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:59.287723.287723 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:59.287354.287354 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:00:59.289922.289922 cuda_h.py:27] end init_inputs_tokens cost 8.172 ms
DEBUG 05-06 10:00:59.289672.289672 lmp.py:1504] decode step 6 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:59.289342.289342 lmp.py:1510] ---- decode step 6 layer 0 ----
DEBUG 05-06 10:00:59.295387.295387 cuda_h.py:27] end decode_layer cost 5.677 ms
DEBUG 05-06 10:00:59.295283.295283 lmp.py:1510] ---- decode step 6 layer 1 ----
DEBUG 05-06 10:00:59.301168.301168 cuda_h.py:27] end decode_layer cost 6.053 ms
DEBUG 05-06 10:00:59.301872.301872 lmp.py:1510] ---- decode step 6 layer 2 ----
DEBUG 05-06 10:00:59.307333.307333 cuda_h.py:27] end decode_layer cost 5.634 ms
DEBUG 05-06 10:00:59.307798.307798 lmp.py:1510] ---- decode step 6 layer 3 ----
DEBUG 05-06 10:00:59.313997.313997 cuda_h.py:27] end decode_layer cost 5.933 ms
DEBUG 05-06 10:00:59.313178.313178 lmp.py:1510] ---- decode step 6 layer 4 ----
DEBUG 05-06 10:00:59.318724.318724 cuda_h.py:27] end decode_layer cost 5.628 ms
DEBUG 05-06 10:00:59.318097.318097 lmp.py:1510] ---- decode step 6 layer 5 ----
DEBUG 05-06 10:00:59.324500.324500 cuda_h.py:27] end decode_layer cost 6.083 ms
DEBUG 05-06 10:00:59.324357.324357 lmp.py:1510] ---- decode step 6 layer 6 ----
DEBUG 05-06 10:00:59.330395.330395 cuda_h.py:27] end decode_layer cost 5.674 ms
DEBUG 05-06 10:00:59.330529.330529 lmp.py:1510] ---- decode step 6 layer 7 ----
DEBUG 05-06 10:00:59.336413.336413 cuda_h.py:27] end decode_layer cost 5.805 ms
DEBUG 05-06 10:00:59.336739.336739 lmp.py:1510] ---- decode step 6 layer 8 ----
DEBUG 05-06 10:00:59.342432.342432 cuda_h.py:27] end decode_layer cost 5.666 ms
DEBUG 05-06 10:00:59.342898.342898 lmp.py:1510] ---- decode step 6 layer 9 ----
DEBUG 05-06 10:00:59.348250.348250 cuda_h.py:27] end decode_layer cost 5.765 ms
DEBUG 05-06 10:00:59.348100.348100 lmp.py:1510] ---- decode step 6 layer 10 ----
DEBUG 05-06 10:00:59.353045.353045 cuda_h.py:27] end decode_layer cost 5.676 ms
DEBUG 05-06 10:00:59.353750.353750 lmp.py:1510] ---- decode step 6 layer 11 ----
DEBUG 05-06 10:00:59.359779.359779 cuda_h.py:27] end decode_layer cost 6.019 ms
DEBUG 05-06 10:00:59.359675.359675 lmp.py:1510] ---- decode step 6 layer 12 ----
DEBUG 05-06 10:00:59.365611.365611 cuda_h.py:27] end decode_layer cost 5.599 ms
DEBUG 05-06 10:00:59.365792.365792 lmp.py:1510] ---- decode step 6 layer 13 ----
DEBUG 05-06 10:00:59.371785.371785 cuda_h.py:27] end decode_layer cost 5.711 ms
DEBUG 05-06 10:00:59.371204.371204 lmp.py:1510] ---- decode step 6 layer 14 ----
DEBUG 05-06 10:00:59.377657.377657 cuda_h.py:27] end decode_layer cost 5.804 ms
DEBUG 05-06 10:00:59.377123.377123 lmp.py:1510] ---- decode step 6 layer 15 ----
DEBUG 05-06 10:00:59.382485.382485 cuda_h.py:27] end decode_layer cost 5.667 ms
DEBUG 05-06 10:00:59.383236.383236 lmp.py:1510] ---- decode step 6 layer 16 ----
DEBUG 05-06 10:00:59.388474.388474 cuda_h.py:27] end decode_layer cost 5.715 ms
DEBUG 05-06 10:00:59.388814.388814 lmp.py:1510] ---- decode step 6 layer 17 ----
DEBUG 05-06 10:00:59.394317.394317 cuda_h.py:27] end decode_layer cost 6.122 ms
DEBUG 05-06 10:00:59.395121.395121 lmp.py:1510] ---- decode step 6 layer 18 ----
DEBUG 05-06 10:00:59.400988.400988 cuda_h.py:27] end decode_layer cost 5.724 ms
DEBUG 05-06 10:00:59.400123.400123 lmp.py:1510] ---- decode step 6 layer 19 ----
DEBUG 05-06 10:00:59.406546.406546 cuda_h.py:27] end decode_layer cost 5.713 ms
DEBUG 05-06 10:00:59.406727.406727 lmp.py:1510] ---- decode step 6 layer 20 ----
DEBUG 05-06 10:00:59.412759.412759 cuda_h.py:27] end decode_layer cost 5.705 ms
DEBUG 05-06 10:00:59.412271.412271 lmp.py:1510] ---- decode step 6 layer 21 ----
DEBUG 05-06 10:00:59.418678.418678 cuda_h.py:27] end decode_layer cost 5.630 ms
DEBUG 05-06 10:00:59.418667.418667 lmp.py:1510] ---- decode step 6 layer 22 ----
DEBUG 05-06 10:00:59.423016.423016 cuda_h.py:27] end decode_layer cost 5.657 ms
DEBUG 05-06 10:00:59.423005.423005 lmp.py:1510] ---- decode step 6 layer 23 ----
DEBUG 05-06 10:00:59.429553.429553 cuda_h.py:27] end decode_layer cost 5.874 ms
DEBUG 05-06 10:00:59.429303.429303 lmp.py:1510] ---- decode step 6 layer 24 ----
DEBUG 05-06 10:00:59.435323.435323 cuda_h.py:27] end decode_layer cost 5.941 ms
DEBUG 05-06 10:00:59.435027.435027 lmp.py:1510] ---- decode step 6 layer 25 ----
DEBUG 05-06 10:00:59.441785.441785 cuda_h.py:27] end decode_layer cost 5.819 ms
DEBUG 05-06 10:00:59.441774.441774 lmp.py:1510] ---- decode step 6 layer 26 ----
DEBUG 05-06 10:00:59.447455.447455 cuda_h.py:27] end decode_layer cost 5.692 ms
DEBUG 05-06 10:00:59.447828.447828 lmp.py:1510] ---- decode step 6 layer 27 ----
DEBUG 05-06 10:00:59.453667.453667 cuda_h.py:27] end decode_layer cost 5.668 ms
DEBUG 05-06 10:00:59.453941.453941 lmp.py:1510] ---- decode step 6 layer 28 ----
DEBUG 05-06 10:00:59.458077.458077 cuda_h.py:27] end decode_layer cost 5.641 ms
DEBUG 05-06 10:00:59.458781.458781 lmp.py:1510] ---- decode step 6 layer 29 ----
DEBUG 05-06 10:00:59.465651.465651 cuda_h.py:27] end decode_layer cost 6.181 ms
DEBUG 05-06 10:00:59.465773.465773 cuda_h.py:27] end decode_step cost 183.909 ms
INFO 05-06 10:00:59.465820.465820 lmp.py:1558] decode step 6 time: 0.18394780158996582 seconds
WARNING 05-06 10:00:59.465640.465640 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:59.465021.465021 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:00:59.466315.466315 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:59.466670.466670 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:00:59.471574.471574 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:59.471559.471559 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:59.471951.471951 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:00:59.473464.473464 cuda_h.py:27] end init_inputs_tokens cost 8.102 ms
DEBUG 05-06 10:00:59.473930.473930 lmp.py:1504] decode step 7 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:59.473461.473461 lmp.py:1510] ---- decode step 7 layer 0 ----
DEBUG 05-06 10:00:59.479228.479228 cuda_h.py:27] end decode_layer cost 6.070 ms
DEBUG 05-06 10:00:59.479316.479316 lmp.py:1510] ---- decode step 7 layer 1 ----
DEBUG 05-06 10:00:59.485073.485073 cuda_h.py:27] end decode_layer cost 5.783 ms
DEBUG 05-06 10:00:59.485207.485207 lmp.py:1510] ---- decode step 7 layer 2 ----
DEBUG 05-06 10:00:59.491274.491274 cuda_h.py:27] end decode_layer cost 5.714 ms
DEBUG 05-06 10:00:59.491931.491931 lmp.py:1510] ---- decode step 7 layer 3 ----
DEBUG 05-06 10:00:59.497141.497141 cuda_h.py:27] end decode_layer cost 5.872 ms
DEBUG 05-06 10:00:59.497230.497230 lmp.py:1510] ---- decode step 7 layer 4 ----
DEBUG 05-06 10:00:59.502817.502817 cuda_h.py:27] end decode_layer cost 5.658 ms
DEBUG 05-06 10:00:59.502759.502759 lmp.py:1510] ---- decode step 7 layer 5 ----
DEBUG 05-06 10:00:59.509829.509829 cuda_h.py:27] end decode_layer cost 6.224 ms
DEBUG 05-06 10:00:59.509917.509917 lmp.py:1510] ---- decode step 7 layer 6 ----
DEBUG 05-06 10:00:59.514996.514996 cuda_h.py:27] end decode_layer cost 5.704 ms
DEBUG 05-06 10:00:59.514177.514177 lmp.py:1510] ---- decode step 7 layer 7 ----
DEBUG 05-06 10:00:59.520320.520320 cuda_h.py:27] end decode_layer cost 5.646 ms
DEBUG 05-06 10:00:59.520693.520693 lmp.py:1510] ---- decode step 7 layer 8 ----
DEBUG 05-06 10:00:59.526032.526032 cuda_h.py:27] end decode_layer cost 5.581 ms
DEBUG 05-06 10:00:59.526498.526498 lmp.py:1510] ---- decode step 7 layer 9 ----
DEBUG 05-06 10:00:59.532590.532590 cuda_h.py:27] end decode_layer cost 5.714 ms
DEBUG 05-06 10:00:59.532009.532009 lmp.py:1510] ---- decode step 7 layer 10 ----
DEBUG 05-06 10:00:59.537670.537670 cuda_h.py:27] end decode_layer cost 5.676 ms
DEBUG 05-06 10:00:59.537281.537281 lmp.py:1510] ---- decode step 7 layer 11 ----
DEBUG 05-06 10:00:59.543259.543259 cuda_h.py:27] end decode_layer cost 6.051 ms
DEBUG 05-06 10:00:59.544393.544393 lmp.py:1510] ---- decode step 7 layer 12 ----
DEBUG 05-06 10:00:59.549714.549714 cuda_h.py:27] end decode_layer cost 5.602 ms
DEBUG 05-06 10:00:59.549848.549848 lmp.py:1510] ---- decode step 7 layer 13 ----
DEBUG 05-06 10:00:59.555935.555935 cuda_h.py:27] end decode_layer cost 5.745 ms
DEBUG 05-06 10:00:59.555639.555639 lmp.py:1510] ---- decode step 7 layer 14 ----
DEBUG 05-06 10:00:59.561450.561450 cuda_h.py:27] end decode_layer cost 5.612 ms
DEBUG 05-06 10:00:59.561915.561915 lmp.py:1510] ---- decode step 7 layer 15 ----
DEBUG 05-06 10:00:59.566641.566641 cuda_h.py:27] end decode_layer cost 5.655 ms
DEBUG 05-06 10:00:59.566061.566061 lmp.py:1510] ---- decode step 7 layer 16 ----
DEBUG 05-06 10:00:59.572633.572633 cuda_h.py:27] end decode_layer cost 5.998 ms
DEBUG 05-06 10:00:59.572390.572390 lmp.py:1510] ---- decode step 7 layer 17 ----
DEBUG 05-06 10:00:59.579635.579635 cuda_h.py:27] end decode_layer cost 6.143 ms
DEBUG 05-06 10:00:59.579863.579863 lmp.py:1510] ---- decode step 7 layer 18 ----
DEBUG 05-06 10:00:59.584254.584254 cuda_h.py:27] end decode_layer cost 5.759 ms
DEBUG 05-06 10:00:59.585766.585766 lmp.py:1510] ---- decode step 7 layer 19 ----
DEBUG 05-06 10:00:59.590617.590617 cuda_h.py:27] end decode_layer cost 5.817 ms
DEBUG 05-06 10:00:59.590467.590467 lmp.py:1510] ---- decode step 7 layer 20 ----
DEBUG 05-06 10:00:59.596540.596540 cuda_h.py:27] end decode_layer cost 5.735 ms
DEBUG 05-06 10:00:59.596536.596536 lmp.py:1510] ---- decode step 7 layer 21 ----
DEBUG 05-06 10:00:59.602456.602456 cuda_h.py:27] end decode_layer cost 5.728 ms
DEBUG 05-06 10:00:59.602637.602637 lmp.py:1510] ---- decode step 7 layer 22 ----
DEBUG 05-06 10:00:59.608844.608844 cuda_h.py:27] end decode_layer cost 5.588 ms
DEBUG 05-06 10:00:59.608548.608548 lmp.py:1510] ---- decode step 7 layer 23 ----
DEBUG 05-06 10:00:59.614051.614051 cuda_h.py:27] end decode_layer cost 5.912 ms
DEBUG 05-06 10:00:59.614755.614755 lmp.py:1510] ---- decode step 7 layer 24 ----
DEBUG 05-06 10:00:59.619738.619738 cuda_h.py:27] end decode_layer cost 5.598 ms
DEBUG 05-06 10:00:59.619965.619965 lmp.py:1510] ---- decode step 7 layer 25 ----
DEBUG 05-06 10:00:59.625362.625362 cuda_h.py:27] end decode_layer cost 5.729 ms
DEBUG 05-06 10:00:59.625994.625994 lmp.py:1510] ---- decode step 7 layer 26 ----
DEBUG 05-06 10:00:59.631337.631337 cuda_h.py:27] end decode_layer cost 5.689 ms
DEBUG 05-06 10:00:59.631379.631379 lmp.py:1510] ---- decode step 7 layer 27 ----
DEBUG 05-06 10:00:59.637825.637825 cuda_h.py:27] end decode_layer cost 5.799 ms
DEBUG 05-06 10:00:59.637198.637198 lmp.py:1510] ---- decode step 7 layer 28 ----
DEBUG 05-06 10:00:59.642337.642337 cuda_h.py:27] end decode_layer cost 5.713 ms
DEBUG 05-06 10:00:59.643710.643710 lmp.py:1510] ---- decode step 7 layer 29 ----
DEBUG 05-06 10:00:59.649873.649873 cuda_h.py:27] end decode_layer cost 6.259 ms
DEBUG 05-06 10:00:59.649426.649426 cuda_h.py:27] end decode_step cost 184.167 ms
INFO 05-06 10:00:59.649189.649189 lmp.py:1558] decode step 7 time: 0.1842057704925537 seconds
WARNING 05-06 10:00:59.649584.649584 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:59.649768.649768 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:00:59.650029.650029 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:59.650644.650644 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:00:59.655784.655784 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:59.655014.655014 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:59.655075.655075 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:00:59.657431.657431 cuda_h.py:27] end init_inputs_tokens cost 7.981 ms
DEBUG 05-06 10:00:59.657612.657612 lmp.py:1504] decode step 8 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:59.657951.657951 lmp.py:1510] ---- decode step 8 layer 0 ----
DEBUG 05-06 10:00:59.663568.663568 cuda_h.py:27] end decode_layer cost 5.749 ms
DEBUG 05-06 10:00:59.729685.729685 lmp.py:1510] ---- decode step 8 layer 1 ----
DEBUG 05-06 10:00:59.741914.741914 cuda_h.py:27] end decode_layer cost 11.794 ms
DEBUG 05-06 10:00:59.741479.741479 lmp.py:1510] ---- decode step 8 layer 2 ----
DEBUG 05-06 10:00:59.747787.747787 cuda_h.py:27] end decode_layer cost 5.802 ms
DEBUG 05-06 10:00:59.747266.747266 lmp.py:1510] ---- decode step 8 layer 3 ----
DEBUG 05-06 10:00:59.753326.753326 cuda_h.py:27] end decode_layer cost 5.762 ms
DEBUG 05-06 10:00:59.753746.753746 lmp.py:1510] ---- decode step 8 layer 4 ----
DEBUG 05-06 10:00:59.759211.759211 cuda_h.py:27] end decode_layer cost 5.568 ms
DEBUG 05-06 10:00:59.759392.759392 lmp.py:1510] ---- decode step 8 layer 5 ----
DEBUG 05-06 10:00:59.765777.765777 cuda_h.py:27] end decode_layer cost 5.965 ms
DEBUG 05-06 10:00:59.765342.765342 lmp.py:1510] ---- decode step 8 layer 6 ----
DEBUG 05-06 10:00:59.770293.770293 cuda_h.py:27] end decode_layer cost 5.645 ms
DEBUG 05-06 10:00:59.770143.770143 lmp.py:1510] ---- decode step 8 layer 7 ----
DEBUG 05-06 10:00:59.778363.778363 cuda_h.py:27] end decode_layer cost 7.175 ms
DEBUG 05-06 10:00:59.778731.778731 lmp.py:1510] ---- decode step 8 layer 8 ----
DEBUG 05-06 10:00:59.784937.784937 cuda_h.py:27] end decode_layer cost 5.939 ms
DEBUG 05-06 10:00:59.784502.784502 lmp.py:1510] ---- decode step 8 layer 9 ----
DEBUG 05-06 10:00:59.789716.789716 cuda_h.py:27] end decode_layer cost 5.805 ms
DEBUG 05-06 10:00:59.790851.790851 lmp.py:1510] ---- decode step 8 layer 10 ----
DEBUG 05-06 10:00:59.795032.795032 cuda_h.py:27] end decode_layer cost 5.604 ms
DEBUG 05-06 10:00:59.795167.795167 lmp.py:1510] ---- decode step 8 layer 11 ----
DEBUG 05-06 10:00:59.801552.801552 cuda_h.py:27] end decode_layer cost 5.966 ms
DEBUG 05-06 10:00:59.801018.801018 lmp.py:1510] ---- decode step 8 layer 12 ----
DEBUG 05-06 10:00:59.807913.807913 cuda_h.py:27] end decode_layer cost 5.569 ms
DEBUG 05-06 10:00:59.807855.807855 lmp.py:1510] ---- decode step 8 layer 13 ----
DEBUG 05-06 10:00:59.813601.813601 cuda_h.py:27] end decode_layer cost 5.634 ms
DEBUG 05-06 10:00:59.813782.813782 lmp.py:1510] ---- decode step 8 layer 14 ----
DEBUG 05-06 10:00:59.818180.818180 cuda_h.py:27] end decode_layer cost 5.554 ms
DEBUG 05-06 10:00:59.818361.818361 lmp.py:1510] ---- decode step 8 layer 15 ----
DEBUG 05-06 10:00:59.824480.824480 cuda_h.py:27] end decode_layer cost 5.734 ms
DEBUG 05-06 10:00:59.824091.824091 lmp.py:1510] ---- decode step 8 layer 16 ----
DEBUG 05-06 10:00:59.830572.830572 cuda_h.py:27] end decode_layer cost 5.650 ms
DEBUG 05-06 10:00:59.830276.830276 lmp.py:1510] ---- decode step 8 layer 17 ----
DEBUG 05-06 10:00:59.836190.836190 cuda_h.py:27] end decode_layer cost 6.320 ms
DEBUG 05-06 10:00:59.836517.836517 lmp.py:1510] ---- decode step 8 layer 18 ----
DEBUG 05-06 10:00:59.842413.842413 cuda_h.py:27] end decode_layer cost 5.605 ms
DEBUG 05-06 10:00:59.842594.842594 lmp.py:1510] ---- decode step 8 layer 19 ----
DEBUG 05-06 10:00:59.848766.848766 cuda_h.py:27] end decode_layer cost 5.703 ms
DEBUG 05-06 10:00:59.848516.848516 lmp.py:1510] ---- decode step 8 layer 20 ----
DEBUG 05-06 10:00:59.853782.853782 cuda_h.py:27] end decode_layer cost 5.561 ms
DEBUG 05-06 10:00:59.853963.853963 lmp.py:1510] ---- decode step 8 layer 21 ----
DEBUG 05-06 10:00:59.859449.859449 cuda_h.py:27] end decode_layer cost 5.619 ms
DEBUG 05-06 10:00:59.859392.859392 lmp.py:1510] ---- decode step 8 layer 22 ----
DEBUG 05-06 10:00:59.865585.865585 cuda_h.py:27] end decode_layer cost 5.574 ms
DEBUG 05-06 10:00:59.865051.865051 lmp.py:1510] ---- decode step 8 layer 23 ----
DEBUG 05-06 10:00:59.870963.870963 cuda_h.py:27] end decode_layer cost 5.862 ms
DEBUG 05-06 10:00:59.870475.870475 lmp.py:1510] ---- decode step 8 layer 24 ----
DEBUG 05-06 10:00:59.876820.876820 cuda_h.py:27] end decode_layer cost 5.549 ms
DEBUG 05-06 10:00:59.876285.876285 lmp.py:1510] ---- decode step 8 layer 25 ----
DEBUG 05-06 10:00:59.882622.882622 cuda_h.py:27] end decode_layer cost 5.684 ms
DEBUG 05-06 10:00:59.882372.882372 lmp.py:1510] ---- decode step 8 layer 26 ----
DEBUG 05-06 10:00:59.888783.888783 cuda_h.py:27] end decode_layer cost 5.738 ms
DEBUG 05-06 10:00:59.888156.888156 lmp.py:1510] ---- decode step 8 layer 27 ----
DEBUG 05-06 10:00:59.893501.893501 cuda_h.py:27] end decode_layer cost 5.761 ms
DEBUG 05-06 10:00:59.894298.894298 lmp.py:1510] ---- decode step 8 layer 28 ----
DEBUG 05-06 10:00:59.899152.899152 cuda_h.py:27] end decode_layer cost 5.714 ms
DEBUG 05-06 10:00:59.899856.899856 lmp.py:1510] ---- decode step 8 layer 29 ----
DEBUG 05-06 10:00:59.905865.905865 cuda_h.py:27] end decode_layer cost 6.004 ms
DEBUG 05-06 10:00:59.905180.905180 cuda_h.py:27] end decode_step cost 256.395 ms
INFO 05-06 10:00:59.905704.905704 lmp.py:1558] decode step 8 time: 0.25643467903137207 seconds
WARNING 05-06 10:00:59.906448.906448 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:00:59.906346.906346 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:00:59.907881.907881 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:00:59.907230.907230 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:00:59.912440.912440 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:00:59.912909.912909 helper.py:50]   NaN count: 786432
WARNING 05-06 10:00:59.912261.912261 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:00:59.914426.914426 cuda_h.py:27] end init_inputs_tokens cost 8.121 ms
DEBUG 05-06 10:00:59.914368.914368 lmp.py:1504] decode step 9 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:00:59.914091.914091 lmp.py:1510] ---- decode step 9 layer 0 ----
DEBUG 05-06 10:00:59.921338.921338 cuda_h.py:27] end decode_layer cost 6.535 ms
DEBUG 05-06 10:00:59.921964.921964 lmp.py:1510] ---- decode step 9 layer 1 ----
DEBUG 05-06 10:00:59.927361.927361 cuda_h.py:27] end decode_layer cost 5.904 ms
DEBUG 05-06 10:00:59.927542.927542 lmp.py:1510] ---- decode step 9 layer 2 ----
DEBUG 05-06 10:00:59.933428.933428 cuda_h.py:27] end decode_layer cost 5.879 ms
DEBUG 05-06 10:00:59.933224.933224 lmp.py:1510] ---- decode step 9 layer 3 ----
DEBUG 05-06 10:00:59.938512.938512 cuda_h.py:27] end decode_layer cost 5.824 ms
DEBUG 05-06 10:00:59.939190.939190 lmp.py:1510] ---- decode step 9 layer 4 ----
DEBUG 05-06 10:00:59.944010.944010 cuda_h.py:27] end decode_layer cost 5.689 ms
DEBUG 05-06 10:00:59.944999.944999 lmp.py:1510] ---- decode step 9 layer 5 ----
DEBUG 05-06 10:00:59.950189.950189 cuda_h.py:27] end decode_layer cost 6.067 ms
DEBUG 05-06 10:00:59.950324.950324 lmp.py:1510] ---- decode step 9 layer 6 ----
DEBUG 05-06 10:00:59.956467.956467 cuda_h.py:27] end decode_layer cost 5.857 ms
DEBUG 05-06 10:00:59.956470.956470 lmp.py:1510] ---- decode step 9 layer 7 ----
DEBUG 05-06 10:00:59.962299.962299 cuda_h.py:27] end decode_layer cost 5.766 ms
DEBUG 05-06 10:00:59.962572.962572 lmp.py:1510] ---- decode step 9 layer 8 ----
DEBUG 05-06 10:00:59.968684.968684 cuda_h.py:27] end decode_layer cost 5.693 ms
DEBUG 05-06 10:00:59.968864.968864 lmp.py:1510] ---- decode step 9 layer 9 ----
DEBUG 05-06 10:00:59.974575.974575 cuda_h.py:27] end decode_layer cost 5.784 ms
DEBUG 05-06 10:00:59.974041.974041 lmp.py:1510] ---- decode step 9 layer 10 ----
DEBUG 05-06 10:00:59.980317.980317 cuda_h.py:27] end decode_layer cost 5.674 ms
DEBUG 05-06 10:00:59.980114.980114 lmp.py:1510] ---- decode step 9 layer 11 ----
DEBUG 05-06 10:00:59.986033.986033 cuda_h.py:27] end decode_layer cost 6.078 ms
DEBUG 05-06 10:00:59.986306.986306 lmp.py:1510] ---- decode step 9 layer 12 ----
DEBUG 05-06 10:00:59.991393.991393 cuda_h.py:27] end decode_layer cost 5.745 ms
DEBUG 05-06 10:00:59.992812.992812 lmp.py:1510] ---- decode step 9 layer 13 ----
DEBUG 05-06 10:00:59.997150.997150 cuda_h.py:27] end decode_layer cost 5.720 ms
DEBUG 05-06 10:00:59.997185.997185 lmp.py:1510] ---- decode step 9 layer 14 ----
DEBUG 05-06 10:01:00.003091.003091 cuda_h.py:27] end decode_layer cost 6.102 ms
DEBUG 05-06 10:01:00.004617.004617 lmp.py:1510] ---- decode step 9 layer 15 ----
DEBUG 05-06 10:01:00.010932.010932 cuda_h.py:27] end decode_layer cost 6.055 ms
DEBUG 05-06 10:01:00.010305.010305 lmp.py:1510] ---- decode step 9 layer 16 ----
DEBUG 05-06 10:01:00.016742.016742 cuda_h.py:27] end decode_layer cost 5.897 ms
DEBUG 05-06 10:01:00.016923.016923 lmp.py:1510] ---- decode step 9 layer 17 ----
DEBUG 05-06 10:01:00.022953.022953 cuda_h.py:27] end decode_layer cost 6.230 ms
DEBUG 05-06 10:01:00.022133.022133 lmp.py:1510] ---- decode step 9 layer 18 ----
DEBUG 05-06 10:01:00.028093.028093 cuda_h.py:27] end decode_layer cost 5.897 ms
DEBUG 05-06 10:01:00.028274.028274 lmp.py:1510] ---- decode step 9 layer 19 ----
DEBUG 05-06 10:01:00.034143.034143 cuda_h.py:27] end decode_layer cost 5.971 ms
DEBUG 05-06 10:01:00.034800.034800 lmp.py:1510] ---- decode step 9 layer 20 ----
DEBUG 05-06 10:01:00.040612.040612 cuda_h.py:27] end decode_layer cost 5.823 ms
DEBUG 05-06 10:01:00.040031.040031 lmp.py:1510] ---- decode step 9 layer 21 ----
DEBUG 05-06 10:01:00.046893.046893 cuda_h.py:27] end decode_layer cost 5.756 ms
DEBUG 05-06 10:01:00.046643.046643 lmp.py:1510] ---- decode step 9 layer 22 ----
DEBUG 05-06 10:01:00.051376.051376 cuda_h.py:27] end decode_layer cost 5.659 ms
DEBUG 05-06 10:01:00.051034.051034 lmp.py:1510] ---- decode step 9 layer 23 ----
DEBUG 05-06 10:01:00.057819.057819 cuda_h.py:27] end decode_layer cost 6.050 ms
DEBUG 05-06 10:01:00.057908.057908 lmp.py:1510] ---- decode step 9 layer 24 ----
DEBUG 05-06 10:01:00.063567.063567 cuda_h.py:27] end decode_layer cost 5.641 ms
DEBUG 05-06 10:01:00.063960.063960 lmp.py:1510] ---- decode step 9 layer 25 ----
DEBUG 05-06 10:01:00.069925.069925 cuda_h.py:27] end decode_layer cost 5.866 ms
DEBUG 05-06 10:01:00.069391.069391 lmp.py:1510] ---- decode step 9 layer 26 ----
DEBUG 05-06 10:01:00.075175.075175 cuda_h.py:27] end decode_layer cost 5.803 ms
DEBUG 05-06 10:01:00.075283.075283 lmp.py:1510] ---- decode step 9 layer 27 ----
DEBUG 05-06 10:01:00.082883.082883 cuda_h.py:27] end decode_layer cost 6.592 ms
DEBUG 05-06 10:01:00.082965.082965 lmp.py:1510] ---- decode step 9 layer 28 ----
DEBUG 05-06 10:01:00.088309.088309 cuda_h.py:27] end decode_layer cost 6.112 ms
DEBUG 05-06 10:01:00.088205.088205 lmp.py:1510] ---- decode step 9 layer 29 ----
DEBUG 05-06 10:01:00.094292.094292 cuda_h.py:27] end decode_layer cost 6.132 ms
DEBUG 05-06 10:01:00.094990.094990 cuda_h.py:27] end decode_step cost 188.352 ms
INFO 05-06 10:01:00.094945.094945 lmp.py:1558] decode step 9 time: 0.18839144706726074 seconds
WARNING 05-06 10:01:00.094878.094878 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:00.095122.095122 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:00.095811.095811 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:00.095212.095212 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:00.101126.101126 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:00.101402.101402 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:00.101794.101794 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:00.103720.103720 cuda_h.py:27] end init_inputs_tokens cost 8.444 ms
DEBUG 05-06 10:01:00.103947.103947 lmp.py:1504] decode step 10 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:00.103571.103571 lmp.py:1510] ---- decode step 10 layer 0 ----
DEBUG 05-06 10:01:00.109406.109406 cuda_h.py:27] end decode_layer cost 5.734 ms
DEBUG 05-06 10:01:00.109487.109487 lmp.py:1510] ---- decode step 10 layer 1 ----
DEBUG 05-06 10:01:00.114661.114661 cuda_h.py:27] end decode_layer cost 5.774 ms
DEBUG 05-06 10:01:00.114458.114458 lmp.py:1510] ---- decode step 10 layer 2 ----
DEBUG 05-06 10:01:00.120684.120684 cuda_h.py:27] end decode_layer cost 5.568 ms
DEBUG 05-06 10:01:00.120957.120957 lmp.py:1510] ---- decode step 10 layer 3 ----
DEBUG 05-06 10:01:00.126811.126811 cuda_h.py:27] end decode_layer cost 5.714 ms
DEBUG 05-06 10:01:00.126045.126045 lmp.py:1510] ---- decode step 10 layer 4 ----
DEBUG 05-06 10:01:00.132847.132847 cuda_h.py:27] end decode_layer cost 5.746 ms
DEBUG 05-06 10:01:00.132074.132074 lmp.py:1510] ---- decode step 10 layer 5 ----
DEBUG 05-06 10:01:00.138512.138512 cuda_h.py:27] end decode_layer cost 6.145 ms
DEBUG 05-06 10:01:00.138714.138714 lmp.py:1510] ---- decode step 10 layer 6 ----
DEBUG 05-06 10:01:00.144112.144112 cuda_h.py:27] end decode_layer cost 5.764 ms
DEBUG 05-06 10:01:00.144293.144293 lmp.py:1510] ---- decode step 10 layer 7 ----
DEBUG 05-06 10:01:00.150333.150333 cuda_h.py:27] end decode_layer cost 5.746 ms
DEBUG 05-06 10:01:00.150084.150084 lmp.py:1510] ---- decode step 10 layer 8 ----
DEBUG 05-06 10:01:00.155228.155228 cuda_h.py:27] end decode_layer cost 5.682 ms
DEBUG 05-06 10:01:00.155932.155932 lmp.py:1510] ---- decode step 10 layer 9 ----
DEBUG 05-06 10:01:00.161250.161250 cuda_h.py:27] end decode_layer cost 5.740 ms
DEBUG 05-06 10:01:00.161431.161431 lmp.py:1510] ---- decode step 10 layer 10 ----
DEBUG 05-06 10:01:00.167189.167189 cuda_h.py:27] end decode_layer cost 5.608 ms
DEBUG 05-06 10:01:00.167747.167747 lmp.py:1510] ---- decode step 10 layer 11 ----
DEBUG 05-06 10:01:00.173498.173498 cuda_h.py:27] end decode_layer cost 6.199 ms
DEBUG 05-06 10:01:00.173679.173679 lmp.py:1510] ---- decode step 10 layer 12 ----
DEBUG 05-06 10:01:00.179243.179243 cuda_h.py:27] end decode_layer cost 5.956 ms
DEBUG 05-06 10:01:00.179484.179484 lmp.py:1510] ---- decode step 10 layer 13 ----
DEBUG 05-06 10:01:00.185093.185093 cuda_h.py:27] end decode_layer cost 5.927 ms
DEBUG 05-06 10:01:00.185989.185989 lmp.py:1510] ---- decode step 10 layer 14 ----
DEBUG 05-06 10:01:00.191721.191721 cuda_h.py:27] end decode_layer cost 5.835 ms
DEBUG 05-06 10:01:00.191948.191948 lmp.py:1510] ---- decode step 10 layer 15 ----
DEBUG 05-06 10:01:00.197813.197813 cuda_h.py:27] end decode_layer cost 5.863 ms
DEBUG 05-06 10:01:00.197233.197233 lmp.py:1510] ---- decode step 10 layer 16 ----
DEBUG 05-06 10:01:00.203081.203081 cuda_h.py:27] end decode_layer cost 5.745 ms
DEBUG 05-06 10:01:00.203308.203308 lmp.py:1510] ---- decode step 10 layer 17 ----
DEBUG 05-06 10:01:00.209253.209253 cuda_h.py:27] end decode_layer cost 6.062 ms
DEBUG 05-06 10:01:00.209811.209811 lmp.py:1510] ---- decode step 10 layer 18 ----
DEBUG 05-06 10:01:00.215457.215457 cuda_h.py:27] end decode_layer cost 5.631 ms
DEBUG 05-06 10:01:00.215208.215208 lmp.py:1510] ---- decode step 10 layer 19 ----
DEBUG 05-06 10:01:00.220150.220150 cuda_h.py:27] end decode_layer cost 5.779 ms
DEBUG 05-06 10:01:00.220708.220708 lmp.py:1510] ---- decode step 10 layer 20 ----
DEBUG 05-06 10:01:00.226016.226016 cuda_h.py:27] end decode_layer cost 5.627 ms
DEBUG 05-06 10:01:00.226243.226243 lmp.py:1510] ---- decode step 10 layer 21 ----
DEBUG 05-06 10:01:00.232007.232007 cuda_h.py:27] end decode_layer cost 5.788 ms
DEBUG 05-06 10:01:00.232664.232664 lmp.py:1510] ---- decode step 10 layer 22 ----
DEBUG 05-06 10:01:00.238298.238298 cuda_h.py:27] end decode_layer cost 5.657 ms
DEBUG 05-06 10:01:00.238194.238194 lmp.py:1510] ---- decode step 10 layer 23 ----
DEBUG 05-06 10:01:00.244753.244753 cuda_h.py:27] end decode_layer cost 6.410 ms
DEBUG 05-06 10:01:00.244173.244173 lmp.py:1510] ---- decode step 10 layer 24 ----
DEBUG 05-06 10:01:00.250512.250512 cuda_h.py:27] end decode_layer cost 5.966 ms
DEBUG 05-06 10:01:00.250547.250547 lmp.py:1510] ---- decode step 10 layer 25 ----
DEBUG 05-06 10:01:00.256910.256910 cuda_h.py:27] end decode_layer cost 5.703 ms
DEBUG 05-06 10:01:00.256945.256945 lmp.py:1510] ---- decode step 10 layer 26 ----
DEBUG 05-06 10:01:00.262860.262860 cuda_h.py:27] end decode_layer cost 5.759 ms
DEBUG 05-06 10:01:00.262134.262134 lmp.py:1510] ---- decode step 10 layer 27 ----
DEBUG 05-06 10:01:00.268147.268147 cuda_h.py:27] end decode_layer cost 5.726 ms
DEBUG 05-06 10:01:00.268421.268421 lmp.py:1510] ---- decode step 10 layer 28 ----
DEBUG 05-06 10:01:00.273558.273558 cuda_h.py:27] end decode_layer cost 5.678 ms
DEBUG 05-06 10:01:00.273593.273593 lmp.py:1510] ---- decode step 10 layer 29 ----
DEBUG 05-06 10:01:00.279858.279858 cuda_h.py:27] end decode_layer cost 6.122 ms
DEBUG 05-06 10:01:00.280457.280457 cuda_h.py:27] end decode_step cost 185.327 ms
INFO 05-06 10:01:00.280220.280220 lmp.py:1558] decode step 10 time: 0.1853656768798828 seconds
WARNING 05-06 10:01:00.280046.280046 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:00.280658.280658 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:00.280538.280538 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:00.281840.281840 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:00.286986.286986 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:00.286163.286163 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:00.286078.286078 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:00.288732.288732 cuda_h.py:27] end init_inputs_tokens cost 8.108 ms
DEBUG 05-06 10:01:00.288436.288436 lmp.py:1504] decode step 11 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:00.288205.288205 lmp.py:1510] ---- decode step 11 layer 0 ----
DEBUG 05-06 10:01:00.294025.294025 cuda_h.py:27] end decode_layer cost 5.689 ms
DEBUG 05-06 10:01:00.294789.294789 lmp.py:1510] ---- decode step 11 layer 1 ----
DEBUG 05-06 10:01:00.299156.299156 cuda_h.py:27] end decode_layer cost 5.811 ms
DEBUG 05-06 10:01:00.300145.300145 lmp.py:1510] ---- decode step 11 layer 2 ----
DEBUG 05-06 10:01:00.305509.305509 cuda_h.py:27] end decode_layer cost 5.704 ms
DEBUG 05-06 10:01:00.305544.305544 lmp.py:1510] ---- decode step 11 layer 3 ----
DEBUG 05-06 10:01:00.311939.311939 cuda_h.py:27] end decode_layer cost 5.656 ms
DEBUG 05-06 10:01:00.311643.311643 lmp.py:1510] ---- decode step 11 layer 4 ----
DEBUG 05-06 10:01:00.317282.317282 cuda_h.py:27] end decode_layer cost 5.626 ms
DEBUG 05-06 10:01:00.317555.317555 lmp.py:1510] ---- decode step 11 layer 5 ----
DEBUG 05-06 10:01:00.323479.323479 cuda_h.py:27] end decode_layer cost 6.011 ms
DEBUG 05-06 10:01:00.323037.323037 lmp.py:1510] ---- decode step 11 layer 6 ----
DEBUG 05-06 10:01:00.328548.328548 cuda_h.py:27] end decode_layer cost 5.567 ms
DEBUG 05-06 10:01:00.328345.328345 lmp.py:1510] ---- decode step 11 layer 7 ----
DEBUG 05-06 10:01:00.334058.334058 cuda_h.py:27] end decode_layer cost 5.681 ms
DEBUG 05-06 10:01:00.334570.334570 lmp.py:1510] ---- decode step 11 layer 8 ----
DEBUG 05-06 10:01:00.340236.340236 cuda_h.py:27] end decode_layer cost 5.821 ms
DEBUG 05-06 10:01:00.340701.340701 lmp.py:1510] ---- decode step 11 layer 9 ----
DEBUG 05-06 10:01:00.346856.346856 cuda_h.py:27] end decode_layer cost 5.795 ms
DEBUG 05-06 10:01:00.346560.346560 lmp.py:1510] ---- decode step 11 layer 10 ----
DEBUG 05-06 10:01:00.352035.352035 cuda_h.py:27] end decode_layer cost 5.680 ms
DEBUG 05-06 10:01:00.352978.352978 lmp.py:1510] ---- decode step 11 layer 11 ----
DEBUG 05-06 10:01:00.358660.358660 cuda_h.py:27] end decode_layer cost 5.939 ms
DEBUG 05-06 10:01:00.358457.358457 lmp.py:1510] ---- decode step 11 layer 12 ----
DEBUG 05-06 10:01:00.363739.363739 cuda_h.py:27] end decode_layer cost 5.644 ms
DEBUG 05-06 10:01:00.363012.363012 lmp.py:1510] ---- decode step 11 layer 13 ----
DEBUG 05-06 10:01:00.369601.369601 cuda_h.py:27] end decode_layer cost 5.694 ms
DEBUG 05-06 10:01:00.369589.369589 lmp.py:1510] ---- decode step 11 layer 14 ----
DEBUG 05-06 10:01:00.375231.375231 cuda_h.py:27] end decode_layer cost 5.698 ms
DEBUG 05-06 10:01:00.375412.375412 lmp.py:1510] ---- decode step 11 layer 15 ----
DEBUG 05-06 10:01:00.381559.381559 cuda_h.py:27] end decode_layer cost 5.790 ms
DEBUG 05-06 10:01:00.381833.381833 lmp.py:1510] ---- decode step 11 layer 16 ----
DEBUG 05-06 10:01:00.386671.386671 cuda_h.py:27] end decode_layer cost 5.632 ms
DEBUG 05-06 10:01:00.387037.387037 lmp.py:1510] ---- decode step 11 layer 17 ----
DEBUG 05-06 10:01:00.393493.393493 cuda_h.py:27] end decode_layer cost 6.088 ms
DEBUG 05-06 10:01:00.393482.393482 lmp.py:1510] ---- decode step 11 layer 18 ----
DEBUG 05-06 10:01:00.398182.398182 cuda_h.py:27] end decode_layer cost 5.671 ms
DEBUG 05-06 10:01:00.398217.398217 lmp.py:1510] ---- decode step 11 layer 19 ----
DEBUG 05-06 10:01:00.404600.404600 cuda_h.py:27] end decode_layer cost 5.893 ms
DEBUG 05-06 10:01:00.404079.404079 lmp.py:1510] ---- decode step 11 layer 20 ----
DEBUG 05-06 10:01:00.410969.410969 cuda_h.py:27] end decode_layer cost 5.811 ms
DEBUG 05-06 10:01:00.410243.410243 lmp.py:1510] ---- decode step 11 layer 21 ----
DEBUG 05-06 10:01:00.416560.416560 cuda_h.py:27] end decode_layer cost 5.915 ms
DEBUG 05-06 10:01:00.416503.416503 lmp.py:1510] ---- decode step 11 layer 22 ----
DEBUG 05-06 10:01:00.422455.422455 cuda_h.py:27] end decode_layer cost 5.681 ms
DEBUG 05-06 10:01:00.422636.422636 lmp.py:1510] ---- decode step 11 layer 23 ----
DEBUG 05-06 10:01:00.428010.428010 cuda_h.py:27] end decode_layer cost 6.027 ms
DEBUG 05-06 10:01:00.428906.428906 lmp.py:1510] ---- decode step 11 layer 24 ----
DEBUG 05-06 10:01:00.434451.434451 cuda_h.py:27] end decode_layer cost 5.767 ms
DEBUG 05-06 10:01:00.434870.434870 lmp.py:1510] ---- decode step 11 layer 25 ----
DEBUG 05-06 10:01:00.440477.440477 cuda_h.py:27] end decode_layer cost 5.847 ms
DEBUG 05-06 10:01:00.440704.440704 lmp.py:1510] ---- decode step 11 layer 26 ----
DEBUG 05-06 10:01:00.446242.446242 cuda_h.py:27] end decode_layer cost 5.798 ms
DEBUG 05-06 10:01:00.446708.446708 lmp.py:1510] ---- decode step 11 layer 27 ----
DEBUG 05-06 10:01:00.451193.451193 cuda_h.py:27] end decode_layer cost 5.758 ms
DEBUG 05-06 10:01:00.452228.452228 lmp.py:1510] ---- decode step 11 layer 28 ----
DEBUG 05-06 10:01:00.457714.457714 cuda_h.py:27] end decode_layer cost 5.619 ms
DEBUG 05-06 10:01:00.457226.457226 lmp.py:1510] ---- decode step 11 layer 29 ----
DEBUG 05-06 10:01:00.463984.463984 cuda_h.py:27] end decode_layer cost 6.204 ms
DEBUG 05-06 10:01:00.464444.464444 cuda_h.py:27] end decode_step cost 183.848 ms
INFO 05-06 10:01:00.464637.464637 lmp.py:1558] decode step 11 time: 0.18388891220092773 seconds
WARNING 05-06 10:01:00.464947.464947 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:00.464734.464734 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:00.464761.464761 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:00.465878.465878 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:00.470260.470260 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:00.470205.470205 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:00.470028.470028 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:00.472365.472365 cuda_h.py:27] end init_inputs_tokens cost 8.231 ms
DEBUG 05-06 10:01:00.472116.472116 lmp.py:1504] decode step 12 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:00.472978.472978 lmp.py:1510] ---- decode step 12 layer 0 ----
DEBUG 05-06 10:01:00.478360.478360 cuda_h.py:27] end decode_layer cost 5.857 ms
DEBUG 05-06 10:01:00.478349.478349 lmp.py:1510] ---- decode step 12 layer 1 ----
DEBUG 05-06 10:01:00.484698.484698 cuda_h.py:27] end decode_layer cost 5.868 ms
DEBUG 05-06 10:01:00.484878.484878 lmp.py:1510] ---- decode step 12 layer 2 ----
DEBUG 05-06 10:01:00.490581.490581 cuda_h.py:27] end decode_layer cost 5.743 ms
DEBUG 05-06 10:01:00.490901.490901 lmp.py:1510] ---- decode step 12 layer 3 ----
DEBUG 05-06 10:01:00.495339.495339 cuda_h.py:27] end decode_layer cost 5.758 ms
DEBUG 05-06 10:01:00.495950.495950 lmp.py:1510] ---- decode step 12 layer 4 ----
DEBUG 05-06 10:01:00.501991.501991 cuda_h.py:27] end decode_layer cost 5.747 ms
DEBUG 05-06 10:01:00.501026.501026 lmp.py:1510] ---- decode step 12 layer 5 ----
DEBUG 05-06 10:01:00.507764.507764 cuda_h.py:27] end decode_layer cost 6.015 ms
DEBUG 05-06 10:01:00.507183.507183 lmp.py:1510] ---- decode step 12 layer 6 ----
DEBUG 05-06 10:01:00.513247.513247 cuda_h.py:27] end decode_layer cost 5.658 ms
DEBUG 05-06 10:01:00.513713.513713 lmp.py:1510] ---- decode step 12 layer 7 ----
DEBUG 05-06 10:01:00.519116.519116 cuda_h.py:27] end decode_layer cost 5.698 ms
DEBUG 05-06 10:01:00.519866.519866 lmp.py:1510] ---- decode step 12 layer 8 ----
DEBUG 05-06 10:01:00.525241.525241 cuda_h.py:27] end decode_layer cost 5.641 ms
DEBUG 05-06 10:01:00.525799.525799 lmp.py:1510] ---- decode step 12 layer 9 ----
DEBUG 05-06 10:01:00.530789.530789 cuda_h.py:27] end decode_layer cost 5.639 ms
DEBUG 05-06 10:01:00.530493.530493 lmp.py:1510] ---- decode step 12 layer 10 ----
DEBUG 05-06 10:01:00.536967.536967 cuda_h.py:27] end decode_layer cost 5.819 ms
DEBUG 05-06 10:01:00.536340.536340 lmp.py:1510] ---- decode step 12 layer 11 ----
DEBUG 05-06 10:01:00.542501.542501 cuda_h.py:27] end decode_layer cost 6.012 ms
DEBUG 05-06 10:01:00.542013.542013 lmp.py:1510] ---- decode step 12 layer 12 ----
DEBUG 05-06 10:01:00.548256.548256 cuda_h.py:27] end decode_layer cost 5.650 ms
DEBUG 05-06 10:01:00.548053.548053 lmp.py:1510] ---- decode step 12 layer 13 ----
DEBUG 05-06 10:01:00.554905.554905 cuda_h.py:27] end decode_layer cost 5.678 ms
DEBUG 05-06 10:01:00.554040.554040 lmp.py:1510] ---- decode step 12 layer 14 ----
DEBUG 05-06 10:01:00.559747.559747 cuda_h.py:27] end decode_layer cost 5.676 ms
DEBUG 05-06 10:01:00.559596.559596 lmp.py:1510] ---- decode step 12 layer 15 ----
DEBUG 05-06 10:01:00.565373.565373 cuda_h.py:27] end decode_layer cost 5.587 ms
DEBUG 05-06 10:01:00.565693.565693 lmp.py:1510] ---- decode step 12 layer 16 ----
DEBUG 05-06 10:01:00.571203.571203 cuda_h.py:27] end decode_layer cost 5.741 ms
DEBUG 05-06 10:01:00.571907.571907 lmp.py:1510] ---- decode step 12 layer 17 ----
DEBUG 05-06 10:01:00.577687.577687 cuda_h.py:27] end decode_layer cost 6.256 ms
DEBUG 05-06 10:01:00.577483.577483 lmp.py:1510] ---- decode step 12 layer 18 ----
DEBUG 05-06 10:01:00.583316.583316 cuda_h.py:27] end decode_layer cost 5.663 ms
DEBUG 05-06 10:01:00.583305.583305 lmp.py:1510] ---- decode step 12 layer 19 ----
DEBUG 05-06 10:01:00.589240.589240 cuda_h.py:27] end decode_layer cost 5.774 ms
DEBUG 05-06 10:01:00.589560.589560 lmp.py:1510] ---- decode step 12 layer 20 ----
DEBUG 05-06 10:01:00.594457.594457 cuda_h.py:27] end decode_layer cost 5.641 ms
DEBUG 05-06 10:01:00.595016.595016 lmp.py:1510] ---- decode step 12 layer 21 ----
DEBUG 05-06 10:01:00.600951.600951 cuda_h.py:27] end decode_layer cost 5.774 ms
DEBUG 05-06 10:01:00.600430.600430 lmp.py:1510] ---- decode step 12 layer 22 ----
DEBUG 05-06 10:01:00.606504.606504 cuda_h.py:27] end decode_layer cost 5.560 ms
DEBUG 05-06 10:01:00.606301.606301 lmp.py:1510] ---- decode step 12 layer 23 ----
DEBUG 05-06 10:01:00.612505.612505 cuda_h.py:27] end decode_layer cost 5.902 ms
DEBUG 05-06 10:01:00.612348.612348 lmp.py:1510] ---- decode step 12 layer 24 ----
DEBUG 05-06 10:01:00.618932.618932 cuda_h.py:27] end decode_layer cost 5.550 ms
DEBUG 05-06 10:01:00.618729.618729 lmp.py:1510] ---- decode step 12 layer 25 ----
DEBUG 05-06 10:01:00.623526.623526 cuda_h.py:27] end decode_layer cost 5.813 ms
DEBUG 05-06 10:01:00.623422.623422 lmp.py:1510] ---- decode step 12 layer 26 ----
DEBUG 05-06 10:01:00.629266.629266 cuda_h.py:27] end decode_layer cost 5.601 ms
DEBUG 05-06 10:01:00.629632.629632 lmp.py:1510] ---- decode step 12 layer 27 ----
DEBUG 05-06 10:01:00.635870.635870 cuda_h.py:27] end decode_layer cost 5.717 ms
DEBUG 05-06 10:01:00.635859.635859 lmp.py:1510] ---- decode step 12 layer 28 ----
DEBUG 05-06 10:01:00.641635.641635 cuda_h.py:27] end decode_layer cost 5.762 ms
DEBUG 05-06 10:01:00.641193.641193 lmp.py:1510] ---- decode step 12 layer 29 ----
DEBUG 05-06 10:01:00.647484.647484 cuda_h.py:27] end decode_layer cost 6.106 ms
DEBUG 05-06 10:01:00.647527.647527 cuda_h.py:27] end decode_step cost 183.355 ms
INFO 05-06 10:01:00.647813.647813 lmp.py:1558] decode step 12 time: 0.1833944320678711 seconds
WARNING 05-06 10:01:00.647488.647488 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:00.648076.648076 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:00.648142.648142 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:00.648120.648120 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:00.653549.653549 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:00.653732.653732 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:00.653078.653078 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:00.655655.655655 cuda_h.py:27] end init_inputs_tokens cost 7.895 ms
DEBUG 05-06 10:01:00.655167.655167 lmp.py:1504] decode step 13 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:00.655553.655553 lmp.py:1510] ---- decode step 13 layer 0 ----
DEBUG 05-06 10:01:00.661797.661797 cuda_h.py:27] end decode_layer cost 5.895 ms
DEBUG 05-06 10:01:00.661263.661263 lmp.py:1510] ---- decode step 13 layer 1 ----
DEBUG 05-06 10:01:00.667666.667666 cuda_h.py:27] end decode_layer cost 5.908 ms
DEBUG 05-06 10:01:00.667132.667132 lmp.py:1510] ---- decode step 13 layer 2 ----
DEBUG 05-06 10:01:00.673885.673885 cuda_h.py:27] end decode_layer cost 5.886 ms
DEBUG 05-06 10:01:00.673828.673828 lmp.py:1510] ---- decode step 13 layer 3 ----
DEBUG 05-06 10:01:00.679056.679056 cuda_h.py:27] end decode_layer cost 5.815 ms
DEBUG 05-06 10:01:00.679906.679906 lmp.py:1510] ---- decode step 13 layer 4 ----
DEBUG 05-06 10:01:00.685391.685391 cuda_h.py:27] end decode_layer cost 5.794 ms
DEBUG 05-06 10:01:00.685380.685380 lmp.py:1510] ---- decode step 13 layer 5 ----
DEBUG 05-06 10:01:00.691495.691495 cuda_h.py:27] end decode_layer cost 6.012 ms
DEBUG 05-06 10:01:00.691723.691723 lmp.py:1510] ---- decode step 13 layer 6 ----
DEBUG 05-06 10:01:00.697718.697718 cuda_h.py:27] end decode_layer cost 5.783 ms
DEBUG 05-06 10:01:00.697991.697991 lmp.py:1510] ---- decode step 13 layer 7 ----
DEBUG 05-06 10:01:00.702913.702913 cuda_h.py:27] end decode_layer cost 5.764 ms
DEBUG 05-06 10:01:00.702948.702948 lmp.py:1510] ---- decode step 13 layer 8 ----
DEBUG 05-06 10:01:00.708186.708186 cuda_h.py:27] end decode_layer cost 5.716 ms
DEBUG 05-06 10:01:00.708083.708083 lmp.py:1510] ---- decode step 13 layer 9 ----
DEBUG 05-06 10:01:00.714936.714936 cuda_h.py:27] end decode_layer cost 5.714 ms
DEBUG 05-06 10:01:00.714971.714971 lmp.py:1510] ---- decode step 13 layer 10 ----
DEBUG 05-06 10:01:00.720851.720851 cuda_h.py:27] end decode_layer cost 5.698 ms
DEBUG 05-06 10:01:00.720317.720317 lmp.py:1510] ---- decode step 13 layer 11 ----
DEBUG 05-06 10:01:00.726558.726558 cuda_h.py:27] end decode_layer cost 5.999 ms
DEBUG 05-06 10:01:00.726639.726639 lmp.py:1510] ---- decode step 13 layer 12 ----
DEBUG 05-06 10:01:00.732008.732008 cuda_h.py:27] end decode_layer cost 5.673 ms
DEBUG 05-06 10:01:00.732282.732282 lmp.py:1510] ---- decode step 13 layer 13 ----
DEBUG 05-06 10:01:00.737997.737997 cuda_h.py:27] end decode_layer cost 5.717 ms
DEBUG 05-06 10:01:00.737793.737793 lmp.py:1510] ---- decode step 13 layer 14 ----
DEBUG 05-06 10:01:00.743356.743356 cuda_h.py:27] end decode_layer cost 5.921 ms
DEBUG 05-06 10:01:00.743914.743914 lmp.py:1510] ---- decode step 13 layer 15 ----
DEBUG 05-06 10:01:00.749138.749138 cuda_h.py:27] end decode_layer cost 5.670 ms
DEBUG 05-06 10:01:00.749650.749650 lmp.py:1510] ---- decode step 13 layer 16 ----
DEBUG 05-06 10:01:00.755130.755130 cuda_h.py:27] end decode_layer cost 5.650 ms
DEBUG 05-06 10:01:00.755404.755404 lmp.py:1510] ---- decode step 13 layer 17 ----
DEBUG 05-06 10:01:00.761330.761330 cuda_h.py:27] end decode_layer cost 5.908 ms
DEBUG 05-06 10:01:00.761889.761889 lmp.py:1510] ---- decode step 13 layer 18 ----
DEBUG 05-06 10:01:00.766356.766356 cuda_h.py:27] end decode_layer cost 5.640 ms
DEBUG 05-06 10:01:00.767914.767914 lmp.py:1510] ---- decode step 13 layer 19 ----
DEBUG 05-06 10:01:00.772375.772375 cuda_h.py:27] end decode_layer cost 5.845 ms
DEBUG 05-06 10:01:00.772523.772523 lmp.py:1510] ---- decode step 13 layer 20 ----
DEBUG 05-06 10:01:00.778253.778253 cuda_h.py:27] end decode_layer cost 5.763 ms
DEBUG 05-06 10:01:00.778765.778765 lmp.py:1510] ---- decode step 13 layer 21 ----
DEBUG 05-06 10:01:00.784570.784570 cuda_h.py:27] end decode_layer cost 5.854 ms
DEBUG 05-06 10:01:00.784944.784944 lmp.py:1510] ---- decode step 13 layer 22 ----
DEBUG 05-06 10:01:00.790623.790623 cuda_h.py:27] end decode_layer cost 5.656 ms
DEBUG 05-06 10:01:00.790181.790181 lmp.py:1510] ---- decode step 13 layer 23 ----
DEBUG 05-06 10:01:00.796356.796356 cuda_h.py:27] end decode_layer cost 5.985 ms
DEBUG 05-06 10:01:00.796537.796537 lmp.py:1510] ---- decode step 13 layer 24 ----
DEBUG 05-06 10:01:00.802967.802967 cuda_h.py:27] end decode_layer cost 5.718 ms
DEBUG 05-06 10:01:00.802240.802240 lmp.py:1510] ---- decode step 13 layer 25 ----
DEBUG 05-06 10:01:00.807987.807987 cuda_h.py:27] end decode_layer cost 5.670 ms
DEBUG 05-06 10:01:00.808883.808883 lmp.py:1510] ---- decode step 13 layer 26 ----
DEBUG 05-06 10:01:00.813481.813481 cuda_h.py:27] end decode_layer cost 5.596 ms
DEBUG 05-06 10:01:00.813039.813039 lmp.py:1510] ---- decode step 13 layer 27 ----
DEBUG 05-06 10:01:00.819095.819095 cuda_h.py:27] end decode_layer cost 5.617 ms
DEBUG 05-06 10:01:00.819369.819369 lmp.py:1510] ---- decode step 13 layer 28 ----
DEBUG 05-06 10:01:00.825141.825141 cuda_h.py:27] end decode_layer cost 5.654 ms
DEBUG 05-06 10:01:00.825090.825090 lmp.py:1510] ---- decode step 13 layer 29 ----
DEBUG 05-06 10:01:00.831886.831886 cuda_h.py:27] end decode_layer cost 5.953 ms
DEBUG 05-06 10:01:00.831584.831584 cuda_h.py:27] end decode_step cost 183.585 ms
INFO 05-06 10:01:00.831393.831393 lmp.py:1558] decode step 13 time: 0.1836235523223877 seconds
WARNING 05-06 10:01:00.831273.831273 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:00.831687.831687 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:00.832875.832875 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:00.832674.832674 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:00.837888.837888 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:00.837495.837495 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:00.837457.837457 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:00.839657.839657 cuda_h.py:27] end init_inputs_tokens cost 7.997 ms
DEBUG 05-06 10:01:00.839599.839599 lmp.py:1504] decode step 14 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:00.839654.839654 lmp.py:1510] ---- decode step 14 layer 0 ----
DEBUG 05-06 10:01:00.844669.844669 cuda_h.py:27] end decode_layer cost 5.586 ms
DEBUG 05-06 10:01:00.845704.845704 lmp.py:1510] ---- decode step 14 layer 1 ----
DEBUG 05-06 10:01:00.850049.850049 cuda_h.py:27] end decode_layer cost 5.761 ms
DEBUG 05-06 10:01:00.850369.850369 lmp.py:1510] ---- decode step 14 layer 2 ----
DEBUG 05-06 10:01:00.856476.856476 cuda_h.py:27] end decode_layer cost 5.549 ms
DEBUG 05-06 10:01:00.856796.856796 lmp.py:1510] ---- decode step 14 layer 3 ----
DEBUG 05-06 10:01:00.862997.862997 cuda_h.py:27] end decode_layer cost 5.619 ms
DEBUG 05-06 10:01:00.862317.862317 lmp.py:1510] ---- decode step 14 layer 4 ----
DEBUG 05-06 10:01:00.867982.867982 cuda_h.py:27] end decode_layer cost 5.610 ms
DEBUG 05-06 10:01:00.867686.867686 lmp.py:1510] ---- decode step 14 layer 5 ----
DEBUG 05-06 10:01:00.873409.873409 cuda_h.py:27] end decode_layer cost 5.969 ms
DEBUG 05-06 10:01:00.873921.873921 lmp.py:1510] ---- decode step 14 layer 6 ----
DEBUG 05-06 10:01:00.879407.879407 cuda_h.py:27] end decode_layer cost 5.583 ms
DEBUG 05-06 10:01:00.879349.879349 lmp.py:1510] ---- decode step 14 layer 7 ----
DEBUG 05-06 10:01:00.885249.885249 cuda_h.py:27] end decode_layer cost 5.713 ms
DEBUG 05-06 10:01:00.885092.885092 lmp.py:1510] ---- decode step 14 layer 8 ----
DEBUG 05-06 10:01:00.890713.890713 cuda_h.py:27] end decode_layer cost 5.683 ms
DEBUG 05-06 10:01:00.891748.891748 lmp.py:1510] ---- decode step 14 layer 9 ----
DEBUG 05-06 10:01:00.896531.896531 cuda_h.py:27] end decode_layer cost 5.767 ms
DEBUG 05-06 10:01:00.896772.896772 lmp.py:1510] ---- decode step 14 layer 10 ----
DEBUG 05-06 10:01:00.902631.902631 cuda_h.py:27] end decode_layer cost 5.683 ms
DEBUG 05-06 10:01:00.902666.902666 lmp.py:1510] ---- decode step 14 layer 11 ----
DEBUG 05-06 10:01:00.908635.908635 cuda_h.py:27] end decode_layer cost 6.174 ms
DEBUG 05-06 10:01:00.908717.908717 lmp.py:1510] ---- decode step 14 layer 12 ----
DEBUG 05-06 10:01:00.914051.914051 cuda_h.py:27] end decode_layer cost 5.612 ms
DEBUG 05-06 10:01:00.914847.914847 lmp.py:1510] ---- decode step 14 layer 13 ----
DEBUG 05-06 10:01:00.920358.920358 cuda_h.py:27] end decode_layer cost 5.741 ms
DEBUG 05-06 10:01:00.920870.920870 lmp.py:1510] ---- decode step 14 layer 14 ----
DEBUG 05-06 10:01:00.926598.926598 cuda_h.py:27] end decode_layer cost 5.727 ms
DEBUG 05-06 10:01:00.926110.926110 lmp.py:1510] ---- decode step 14 layer 15 ----
DEBUG 05-06 10:01:00.931420.931420 cuda_h.py:27] end decode_layer cost 5.699 ms
DEBUG 05-06 10:01:00.931409.931409 lmp.py:1510] ---- decode step 14 layer 16 ----
DEBUG 05-06 10:01:00.937469.937469 cuda_h.py:27] end decode_layer cost 5.726 ms
DEBUG 05-06 10:01:00.937027.937027 lmp.py:1510] ---- decode step 14 layer 17 ----
DEBUG 05-06 10:01:00.943517.943517 cuda_h.py:27] end decode_layer cost 6.112 ms
DEBUG 05-06 10:01:00.943181.943181 lmp.py:1510] ---- decode step 14 layer 18 ----
DEBUG 05-06 10:01:00.949387.949387 cuda_h.py:27] end decode_layer cost 5.728 ms
DEBUG 05-06 10:01:00.949945.949945 lmp.py:1510] ---- decode step 14 layer 19 ----
DEBUG 05-06 10:01:00.955617.955617 cuda_h.py:27] end decode_layer cost 5.615 ms
DEBUG 05-06 10:01:00.955221.955221 lmp.py:1510] ---- decode step 14 layer 20 ----
DEBUG 05-06 10:01:00.961675.961675 cuda_h.py:27] end decode_layer cost 5.630 ms
DEBUG 05-06 10:01:00.961710.961710 lmp.py:1510] ---- decode step 14 layer 21 ----
DEBUG 05-06 10:01:00.966000.966000 cuda_h.py:27] end decode_layer cost 5.684 ms
DEBUG 05-06 10:01:00.966419.966419 lmp.py:1510] ---- decode step 14 layer 22 ----
DEBUG 05-06 10:01:00.972329.972329 cuda_h.py:27] end decode_layer cost 5.615 ms
DEBUG 05-06 10:01:00.972272.972272 lmp.py:1510] ---- decode step 14 layer 23 ----
DEBUG 05-06 10:01:00.978551.978551 cuda_h.py:27] end decode_layer cost 5.958 ms
DEBUG 05-06 10:01:00.978871.978871 lmp.py:1510] ---- decode step 14 layer 24 ----
DEBUG 05-06 10:01:00.984339.984339 cuda_h.py:27] end decode_layer cost 5.675 ms
DEBUG 05-06 10:01:00.984043.984043 lmp.py:1510] ---- decode step 14 layer 25 ----
DEBUG 05-06 10:01:00.990235.990235 cuda_h.py:27] end decode_layer cost 5.718 ms
DEBUG 05-06 10:01:00.990701.990701 lmp.py:1510] ---- decode step 14 layer 26 ----
DEBUG 05-06 10:01:00.995751.995751 cuda_h.py:27] end decode_layer cost 5.648 ms
DEBUG 05-06 10:01:00.995217.995217 lmp.py:1510] ---- decode step 14 layer 27 ----
DEBUG 05-06 10:01:01.001855.001855 cuda_h.py:27] end decode_layer cost 5.801 ms
DEBUG 05-06 10:01:01.001652.001652 lmp.py:1510] ---- decode step 14 layer 28 ----
DEBUG 05-06 10:01:01.007462.007462 cuda_h.py:27] end decode_layer cost 5.612 ms
DEBUG 05-06 10:01:01.007213.007213 lmp.py:1510] ---- decode step 14 layer 29 ----
DEBUG 05-06 10:01:01.013396.013396 cuda_h.py:27] end decode_layer cost 6.062 ms
DEBUG 05-06 10:01:01.013710.013710 cuda_h.py:27] end decode_step cost 182.269 ms
INFO 05-06 10:01:01.013758.013758 lmp.py:1558] decode step 14 time: 0.18230748176574707 seconds
WARNING 05-06 10:01:01.013438.013438 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:01.014611.014611 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:01.014024.014024 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:01.014611.014611 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:01.019961.019961 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:01.019952.019952 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:01.019298.019298 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:01.021288.021288 cuda_h.py:27] end init_inputs_tokens cost 8.263 ms
DEBUG 05-06 10:01:01.021323.021323 lmp.py:1504] decode step 15 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:01.021994.021994 lmp.py:1510] ---- decode step 15 layer 0 ----
DEBUG 05-06 10:01:01.027200.027200 cuda_h.py:27] end decode_layer cost 5.762 ms
DEBUG 05-06 10:01:01.027427.027427 lmp.py:1510] ---- decode step 15 layer 1 ----
DEBUG 05-06 10:01:01.033763.033763 cuda_h.py:27] end decode_layer cost 5.859 ms
DEBUG 05-06 10:01:01.033316.033316 lmp.py:1510] ---- decode step 15 layer 2 ----
DEBUG 05-06 10:01:01.039236.039236 cuda_h.py:27] end decode_layer cost 5.939 ms
DEBUG 05-06 10:01:01.039941.039941 lmp.py:1510] ---- decode step 15 layer 3 ----
DEBUG 05-06 10:01:01.045468.045468 cuda_h.py:27] end decode_layer cost 5.859 ms
DEBUG 05-06 10:01:01.045887.045887 lmp.py:1510] ---- decode step 15 layer 4 ----
DEBUG 05-06 10:01:01.051911.051911 cuda_h.py:27] end decode_layer cost 5.839 ms
DEBUG 05-06 10:01:01.051006.051006 lmp.py:1510] ---- decode step 15 layer 5 ----
DEBUG 05-06 10:01:01.057757.057757 cuda_h.py:27] end decode_layer cost 6.200 ms
DEBUG 05-06 10:01:01.057461.057461 lmp.py:1510] ---- decode step 15 layer 6 ----
DEBUG 05-06 10:01:01.063088.063088 cuda_h.py:27] end decode_layer cost 5.862 ms
DEBUG 05-06 10:01:01.063176.063176 lmp.py:1510] ---- decode step 15 layer 7 ----
DEBUG 05-06 10:01:01.069547.069547 cuda_h.py:27] end decode_layer cost 6.130 ms
DEBUG 05-06 10:01:01.070682.070682 lmp.py:1510] ---- decode step 15 layer 8 ----
DEBUG 05-06 10:01:01.076273.076273 cuda_h.py:27] end decode_layer cost 5.976 ms
DEBUG 05-06 10:01:01.076560.076560 lmp.py:1510] ---- decode step 15 layer 9 ----
DEBUG 05-06 10:01:01.082525.082525 cuda_h.py:27] end decode_layer cost 6.077 ms
DEBUG 05-06 10:01:01.082250.082250 lmp.py:1510] ---- decode step 15 layer 10 ----
DEBUG 05-06 10:01:01.088435.088435 cuda_h.py:27] end decode_layer cost 5.923 ms
DEBUG 05-06 10:01:01.088762.088762 lmp.py:1510] ---- decode step 15 layer 11 ----
DEBUG 05-06 10:01:01.094687.094687 cuda_h.py:27] end decode_layer cost 6.258 ms
DEBUG 05-06 10:01:01.094106.094106 lmp.py:1510] ---- decode step 15 layer 12 ----
DEBUG 05-06 10:01:01.100628.100628 cuda_h.py:27] end decode_layer cost 5.890 ms
DEBUG 05-06 10:01:01.100240.100240 lmp.py:1510] ---- decode step 15 layer 13 ----
DEBUG 05-06 10:01:01.106941.106941 cuda_h.py:27] end decode_layer cost 5.918 ms
DEBUG 05-06 10:01:01.106076.106076 lmp.py:1510] ---- decode step 15 layer 14 ----
DEBUG 05-06 10:01:01.112161.112161 cuda_h.py:27] end decode_layer cost 5.885 ms
DEBUG 05-06 10:01:01.112818.112818 lmp.py:1510] ---- decode step 15 layer 15 ----
DEBUG 05-06 10:01:01.118087.118087 cuda_h.py:27] end decode_layer cost 5.845 ms
DEBUG 05-06 10:01:01.118460.118460 lmp.py:1510] ---- decode step 15 layer 16 ----
DEBUG 05-06 10:01:01.124500.124500 cuda_h.py:27] end decode_layer cost 5.921 ms
DEBUG 05-06 10:01:01.124158.124158 lmp.py:1510] ---- decode step 15 layer 17 ----
DEBUG 05-06 10:01:01.130512.130512 cuda_h.py:27] end decode_layer cost 6.223 ms
DEBUG 05-06 10:01:01.130123.130123 lmp.py:1510] ---- decode step 15 layer 18 ----
DEBUG 05-06 10:01:01.136705.136705 cuda_h.py:27] end decode_layer cost 5.900 ms
DEBUG 05-06 10:01:01.136648.136648 lmp.py:1510] ---- decode step 15 layer 19 ----
DEBUG 05-06 10:01:01.142218.142218 cuda_h.py:27] end decode_layer cost 5.961 ms
DEBUG 05-06 10:01:01.142797.142797 lmp.py:1510] ---- decode step 15 layer 20 ----
DEBUG 05-06 10:01:01.148394.148394 cuda_h.py:27] end decode_layer cost 5.946 ms
DEBUG 05-06 10:01:01.148290.148290 lmp.py:1510] ---- decode step 15 layer 21 ----
DEBUG 05-06 10:01:01.154385.154385 cuda_h.py:27] end decode_layer cost 5.786 ms
DEBUG 05-06 10:01:01.154089.154089 lmp.py:1510] ---- decode step 15 layer 22 ----
DEBUG 05-06 10:01:01.160974.160974 cuda_h.py:27] end decode_layer cost 5.878 ms
DEBUG 05-06 10:01:01.160347.160347 lmp.py:1510] ---- decode step 15 layer 23 ----
DEBUG 05-06 10:01:01.166784.166784 cuda_h.py:27] end decode_layer cost 6.109 ms
DEBUG 05-06 10:01:01.166018.166018 lmp.py:1510] ---- decode step 15 layer 24 ----
DEBUG 05-06 10:01:01.172245.172245 cuda_h.py:27] end decode_layer cost 5.778 ms
DEBUG 05-06 10:01:01.172472.172472 lmp.py:1510] ---- decode step 15 layer 25 ----
DEBUG 05-06 10:01:01.178042.178042 cuda_h.py:27] end decode_layer cost 5.751 ms
DEBUG 05-06 10:01:01.178078.178078 lmp.py:1510] ---- decode step 15 layer 26 ----
DEBUG 05-06 10:01:01.184096.184096 cuda_h.py:27] end decode_layer cost 5.695 ms
DEBUG 05-06 10:01:01.184323.184323 lmp.py:1510] ---- decode step 15 layer 27 ----
DEBUG 05-06 10:01:01.189847.189847 cuda_h.py:27] end decode_layer cost 5.752 ms
DEBUG 05-06 10:01:01.189359.189359 lmp.py:1510] ---- decode step 15 layer 28 ----
DEBUG 05-06 10:01:01.195948.195948 cuda_h.py:27] end decode_layer cost 5.678 ms
DEBUG 05-06 10:01:01.195413.195413 lmp.py:1510] ---- decode step 15 layer 29 ----
DEBUG 05-06 10:01:01.201645.201645 cuda_h.py:27] end decode_layer cost 6.134 ms
DEBUG 05-06 10:01:01.201874.201874 cuda_h.py:27] end decode_step cost 188.334 ms
INFO 05-06 10:01:01.201398.201398 lmp.py:1558] decode step 15 time: 0.18837285041809082 seconds
WARNING 05-06 10:01:01.202060.202060 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:01.202419.202419 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:01.202134.202134 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:01.203197.203197 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:01.208719.208719 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:01.208902.208902 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:01.208725.208725 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:01.210649.210649 cuda_h.py:27] end init_inputs_tokens cost 8.408 ms
DEBUG 05-06 10:01:01.210445.210445 lmp.py:1504] decode step 16 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:01.210785.210785 lmp.py:1510] ---- decode step 16 layer 0 ----
DEBUG 05-06 10:01:01.216003.216003 cuda_h.py:27] end decode_layer cost 5.736 ms
DEBUG 05-06 10:01:01.216707.216707 lmp.py:1510] ---- decode step 16 layer 1 ----
DEBUG 05-06 10:01:01.222146.222146 cuda_h.py:27] end decode_layer cost 5.759 ms
DEBUG 05-06 10:01:01.222134.222134 lmp.py:1510] ---- decode step 16 layer 2 ----
DEBUG 05-06 10:01:01.227109.227109 cuda_h.py:27] end decode_layer cost 5.768 ms
DEBUG 05-06 10:01:01.228052.228052 lmp.py:1510] ---- decode step 16 layer 3 ----
DEBUG 05-06 10:01:01.234257.234257 cuda_h.py:27] end decode_layer cost 6.113 ms
DEBUG 05-06 10:01:01.234676.234676 lmp.py:1510] ---- decode step 16 layer 4 ----
DEBUG 05-06 10:01:01.240694.240694 cuda_h.py:27] end decode_layer cost 5.870 ms
DEBUG 05-06 10:01:01.240636.240636 lmp.py:1510] ---- decode step 16 layer 5 ----
DEBUG 05-06 10:01:01.246450.246450 cuda_h.py:27] end decode_layer cost 6.106 ms
DEBUG 05-06 10:01:01.246724.246724 lmp.py:1510] ---- decode step 16 layer 6 ----
DEBUG 05-06 10:01:01.252255.252255 cuda_h.py:27] end decode_layer cost 5.756 ms
DEBUG 05-06 10:01:01.252720.252720 lmp.py:1510] ---- decode step 16 layer 7 ----
DEBUG 05-06 10:01:01.257184.257184 cuda_h.py:27] end decode_layer cost 5.743 ms
DEBUG 05-06 10:01:01.257266.257266 lmp.py:1510] ---- decode step 16 layer 8 ----
DEBUG 05-06 10:01:01.263479.263479 cuda_h.py:27] end decode_layer cost 5.768 ms
DEBUG 05-06 10:01:01.263706.263706 lmp.py:1510] ---- decode step 16 layer 9 ----
DEBUG 05-06 10:01:01.269752.269752 cuda_h.py:27] end decode_layer cost 5.716 ms
DEBUG 05-06 10:01:01.269834.269834 lmp.py:1510] ---- decode step 16 layer 10 ----
DEBUG 05-06 10:01:01.275675.275675 cuda_h.py:27] end decode_layer cost 5.740 ms
DEBUG 05-06 10:01:01.275379.275379 lmp.py:1510] ---- decode step 16 layer 11 ----
DEBUG 05-06 10:01:01.281919.281919 cuda_h.py:27] end decode_layer cost 6.220 ms
DEBUG 05-06 10:01:01.281861.281861 lmp.py:1510] ---- decode step 16 layer 12 ----
DEBUG 05-06 10:01:01.287050.287050 cuda_h.py:27] end decode_layer cost 5.821 ms
DEBUG 05-06 10:01:01.287231.287231 lmp.py:1510] ---- decode step 16 layer 13 ----
DEBUG 05-06 10:01:01.293849.293849 cuda_h.py:27] end decode_layer cost 5.996 ms
DEBUG 05-06 10:01:01.293268.293268 lmp.py:1510] ---- decode step 16 layer 14 ----
DEBUG 05-06 10:01:01.299635.299635 cuda_h.py:27] end decode_layer cost 5.811 ms
DEBUG 05-06 10:01:01.299770.299770 lmp.py:1510] ---- decode step 16 layer 15 ----
DEBUG 05-06 10:01:01.305758.305758 cuda_h.py:27] end decode_layer cost 5.778 ms
DEBUG 05-06 10:01:01.305986.305986 lmp.py:1510] ---- decode step 16 layer 16 ----
DEBUG 05-06 10:01:01.311487.311487 cuda_h.py:27] end decode_layer cost 5.665 ms
DEBUG 05-06 10:01:01.311668.311668 lmp.py:1510] ---- decode step 16 layer 17 ----
DEBUG 05-06 10:01:01.317513.317513 cuda_h.py:27] end decode_layer cost 6.059 ms
DEBUG 05-06 10:01:01.317979.317979 lmp.py:1510] ---- decode step 16 layer 18 ----
DEBUG 05-06 10:01:01.322526.322526 cuda_h.py:27] end decode_layer cost 5.664 ms
DEBUG 05-06 10:01:01.322038.322038 lmp.py:1510] ---- decode step 16 layer 19 ----
DEBUG 05-06 10:01:01.328570.328570 cuda_h.py:27] end decode_layer cost 5.792 ms
DEBUG 05-06 10:01:01.328798.328798 lmp.py:1510] ---- decode step 16 layer 20 ----
DEBUG 05-06 10:01:01.334843.334843 cuda_h.py:27] end decode_layer cost 5.715 ms
DEBUG 05-06 10:01:01.334309.334309 lmp.py:1510] ---- decode step 16 layer 21 ----
DEBUG 05-06 10:01:01.340934.340934 cuda_h.py:27] end decode_layer cost 6.002 ms
DEBUG 05-06 10:01:01.340069.340069 lmp.py:1510] ---- decode step 16 layer 22 ----
DEBUG 05-06 10:01:01.346397.346397 cuda_h.py:27] end decode_layer cost 5.853 ms
DEBUG 05-06 10:01:01.346863.346863 lmp.py:1510] ---- decode step 16 layer 23 ----
DEBUG 05-06 10:01:01.352012.352012 cuda_h.py:27] end decode_layer cost 6.202 ms
DEBUG 05-06 10:01:01.352193.352193 lmp.py:1510] ---- decode step 16 layer 24 ----
DEBUG 05-06 10:01:01.358978.358978 cuda_h.py:27] end decode_layer cost 5.839 ms
DEBUG 05-06 10:01:01.358920.358920 lmp.py:1510] ---- decode step 16 layer 25 ----
DEBUG 05-06 10:01:01.364285.364285 cuda_h.py:27] end decode_layer cost 5.950 ms
DEBUG 05-06 10:01:01.364910.364910 lmp.py:1510] ---- decode step 16 layer 26 ----
DEBUG 05-06 10:01:01.370190.370190 cuda_h.py:27] end decode_layer cost 5.783 ms
DEBUG 05-06 10:01:01.370987.370987 lmp.py:1510] ---- decode step 16 layer 27 ----
DEBUG 05-06 10:01:01.376929.376929 cuda_h.py:27] end decode_layer cost 5.779 ms
DEBUG 05-06 10:01:01.376587.376587 lmp.py:1510] ---- decode step 16 layer 28 ----
DEBUG 05-06 10:01:01.382572.382572 cuda_h.py:27] end decode_layer cost 5.670 ms
DEBUG 05-06 10:01:01.382560.382560 lmp.py:1510] ---- decode step 16 layer 29 ----
DEBUG 05-06 10:01:01.388974.388974 cuda_h.py:27] end decode_layer cost 6.021 ms
DEBUG 05-06 10:01:01.388958.388958 cuda_h.py:27] end decode_step cost 186.344 ms
INFO 05-06 10:01:01.388959.388959 lmp.py:1558] decode step 16 time: 0.18638277053833008 seconds
WARNING 05-06 10:01:01.388030.388030 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:01.388695.388695 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:01.389992.389992 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:01.389870.389870 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:01.394815.394815 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:01.394469.394469 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:01.394907.394907 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:01.397515.397515 cuda_h.py:27] end init_inputs_tokens cost 9.225 ms
DEBUG 05-06 10:01:01.397683.397683 lmp.py:1504] decode step 17 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:01.397791.397791 lmp.py:1510] ---- decode step 17 layer 0 ----
DEBUG 05-06 10:01:01.403318.403318 cuda_h.py:27] end decode_layer cost 6.034 ms
DEBUG 05-06 10:01:01.403360.403360 lmp.py:1510] ---- decode step 17 layer 1 ----
DEBUG 05-06 10:01:01.409228.409228 cuda_h.py:27] end decode_layer cost 5.760 ms
DEBUG 05-06 10:01:01.409502.409502 lmp.py:1510] ---- decode step 17 layer 2 ----
DEBUG 05-06 10:01:01.415976.415976 cuda_h.py:27] end decode_layer cost 5.645 ms
DEBUG 05-06 10:01:01.415965.415965 lmp.py:1510] ---- decode step 17 layer 3 ----
DEBUG 05-06 10:01:01.421245.421245 cuda_h.py:27] end decode_layer cost 5.783 ms
DEBUG 05-06 10:01:01.421280.421280 lmp.py:1510] ---- decode step 17 layer 4 ----
DEBUG 05-06 10:01:01.427972.427972 cuda_h.py:27] end decode_layer cost 5.841 ms
DEBUG 05-06 10:01:01.427723.427723 lmp.py:1510] ---- decode step 17 layer 5 ----
DEBUG 05-06 10:01:01.433836.433836 cuda_h.py:27] end decode_layer cost 6.151 ms
DEBUG 05-06 10:01:01.433587.433587 lmp.py:1510] ---- decode step 17 layer 6 ----
DEBUG 05-06 10:01:01.439056.439056 cuda_h.py:27] end decode_layer cost 5.712 ms
DEBUG 05-06 10:01:01.439330.439330 lmp.py:1510] ---- decode step 17 layer 7 ----
DEBUG 05-06 10:01:01.444507.444507 cuda_h.py:27] end decode_layer cost 5.672 ms
DEBUG 05-06 10:01:01.444780.444780 lmp.py:1510] ---- decode step 17 layer 8 ----
DEBUG 05-06 10:01:01.450512.450512 cuda_h.py:27] end decode_layer cost 5.624 ms
DEBUG 05-06 10:01:01.450077.450077 lmp.py:1510] ---- decode step 17 layer 9 ----
DEBUG 05-06 10:01:01.456989.456989 cuda_h.py:27] end decode_layer cost 5.687 ms
DEBUG 05-06 10:01:01.456839.456839 lmp.py:1510] ---- decode step 17 layer 10 ----
DEBUG 05-06 10:01:01.462635.462635 cuda_h.py:27] end decode_layer cost 5.952 ms
DEBUG 05-06 10:01:01.462339.462339 lmp.py:1510] ---- decode step 17 layer 11 ----
DEBUG 05-06 10:01:01.468779.468779 cuda_h.py:27] end decode_layer cost 6.216 ms
DEBUG 05-06 10:01:01.468437.468437 lmp.py:1510] ---- decode step 17 layer 12 ----
DEBUG 05-06 10:01:01.474308.474308 cuda_h.py:27] end decode_layer cost 5.832 ms
DEBUG 05-06 10:01:01.474442.474442 lmp.py:1510] ---- decode step 17 layer 13 ----
DEBUG 05-06 10:01:01.480046.480046 cuda_h.py:27] end decode_layer cost 5.951 ms
DEBUG 05-06 10:01:01.480942.480942 lmp.py:1510] ---- decode step 17 layer 14 ----
DEBUG 05-06 10:01:01.486427.486427 cuda_h.py:27] end decode_layer cost 5.969 ms
DEBUG 05-06 10:01:01.486038.486038 lmp.py:1510] ---- decode step 17 layer 15 ----
DEBUG 05-06 10:01:01.492758.492758 cuda_h.py:27] end decode_layer cost 5.861 ms
DEBUG 05-06 10:01:01.492747.492747 lmp.py:1510] ---- decode step 17 layer 16 ----
DEBUG 05-06 10:01:01.498598.498598 cuda_h.py:27] end decode_layer cost 5.642 ms
DEBUG 05-06 10:01:01.498110.498110 lmp.py:1510] ---- decode step 17 layer 17 ----
DEBUG 05-06 10:01:01.504280.504280 cuda_h.py:27] end decode_layer cost 6.052 ms
DEBUG 05-06 10:01:01.504938.504938 lmp.py:1510] ---- decode step 17 layer 18 ----
DEBUG 05-06 10:01:01.510842.510842 cuda_h.py:27] end decode_layer cost 5.647 ms
DEBUG 05-06 10:01:01.510162.510162 lmp.py:1510] ---- decode step 17 layer 19 ----
DEBUG 05-06 10:01:01.516291.516291 cuda_h.py:27] end decode_layer cost 5.811 ms
DEBUG 05-06 10:01:01.516233.516233 lmp.py:1510] ---- decode step 17 layer 20 ----
DEBUG 05-06 10:01:01.521824.521824 cuda_h.py:27] end decode_layer cost 5.766 ms
DEBUG 05-06 10:01:01.521290.521290 lmp.py:1510] ---- decode step 17 layer 21 ----
DEBUG 05-06 10:01:01.527805.527805 cuda_h.py:27] end decode_layer cost 5.886 ms
DEBUG 05-06 10:01:01.527462.527462 lmp.py:1510] ---- decode step 17 layer 22 ----
DEBUG 05-06 10:01:01.533481.533481 cuda_h.py:27] end decode_layer cost 5.696 ms
DEBUG 05-06 10:01:01.533708.533708 lmp.py:1510] ---- decode step 17 layer 23 ----
DEBUG 05-06 10:01:01.539475.539475 cuda_h.py:27] end decode_layer cost 6.070 ms
DEBUG 05-06 10:01:01.539941.539941 lmp.py:1510] ---- decode step 17 layer 24 ----
DEBUG 05-06 10:01:01.545221.545221 cuda_h.py:27] end decode_layer cost 5.608 ms
DEBUG 05-06 10:01:01.545972.545972 lmp.py:1510] ---- decode step 17 layer 25 ----
DEBUG 05-06 10:01:01.551567.551567 cuda_h.py:27] end decode_layer cost 5.699 ms
DEBUG 05-06 10:01:01.551602.551602 lmp.py:1510] ---- decode step 17 layer 26 ----
DEBUG 05-06 10:01:01.557763.557763 cuda_h.py:27] end decode_layer cost 5.800 ms
DEBUG 05-06 10:01:01.557182.557182 lmp.py:1510] ---- decode step 17 layer 27 ----
DEBUG 05-06 10:01:01.563253.563253 cuda_h.py:27] end decode_layer cost 6.020 ms
DEBUG 05-06 10:01:01.563164.563164 lmp.py:1510] ---- decode step 17 layer 28 ----
DEBUG 05-06 10:01:01.569715.569715 cuda_h.py:27] end decode_layer cost 5.982 ms
DEBUG 05-06 10:01:01.569612.569612 lmp.py:1510] ---- decode step 17 layer 29 ----
DEBUG 05-06 10:01:01.575646.575646 cuda_h.py:27] end decode_layer cost 6.164 ms
DEBUG 05-06 10:01:01.575060.575060 cuda_h.py:27] end decode_step cost 187.092 ms
INFO 05-06 10:01:01.575882.575882 lmp.py:1558] decode step 17 time: 0.18713951110839844 seconds
WARNING 05-06 10:01:01.575557.575557 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:01.576247.576247 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:01.576657.576657 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:01.576058.576058 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:01.581924.581924 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:01.581777.581777 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:01.581646.581646 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:01.583239.583239 cuda_h.py:27] end init_inputs_tokens cost 8.185 ms
DEBUG 05-06 10:01:01.583850.583850 lmp.py:1504] decode step 18 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:01.583620.583620 lmp.py:1510] ---- decode step 18 layer 0 ----
DEBUG 05-06 10:01:01.589435.589435 cuda_h.py:27] end decode_layer cost 5.756 ms
DEBUG 05-06 10:01:01.589186.589186 lmp.py:1510] ---- decode step 18 layer 1 ----
DEBUG 05-06 10:01:01.595059.595059 cuda_h.py:27] end decode_layer cost 5.693 ms
DEBUG 05-06 10:01:01.595571.595571 lmp.py:1510] ---- decode step 18 layer 2 ----
DEBUG 05-06 10:01:01.601345.601345 cuda_h.py:27] end decode_layer cost 5.726 ms
DEBUG 05-06 10:01:01.601857.601857 lmp.py:1510] ---- decode step 18 layer 3 ----
DEBUG 05-06 10:01:01.607029.607029 cuda_h.py:27] end decode_layer cost 5.702 ms
DEBUG 05-06 10:01:01.607541.607541 lmp.py:1510] ---- decode step 18 layer 4 ----
DEBUG 05-06 10:01:01.612441.612441 cuda_h.py:27] end decode_layer cost 5.713 ms
DEBUG 05-06 10:01:01.612714.612714 lmp.py:1510] ---- decode step 18 layer 5 ----
DEBUG 05-06 10:01:01.618127.618127 cuda_h.py:27] end decode_layer cost 5.986 ms
DEBUG 05-06 10:01:01.618685.618685 lmp.py:1510] ---- decode step 18 layer 6 ----
DEBUG 05-06 10:01:01.624797.624797 cuda_h.py:27] end decode_layer cost 5.693 ms
DEBUG 05-06 10:01:01.624739.624739 lmp.py:1510] ---- decode step 18 layer 7 ----
DEBUG 05-06 10:01:01.630452.630452 cuda_h.py:27] end decode_layer cost 5.856 ms
DEBUG 05-06 10:01:01.630679.630679 lmp.py:1510] ---- decode step 18 layer 8 ----
DEBUG 05-06 10:01:01.636378.636378 cuda_h.py:27] end decode_layer cost 5.846 ms
DEBUG 05-06 10:01:01.636321.636321 lmp.py:1510] ---- decode step 18 layer 9 ----
DEBUG 05-06 10:01:01.642761.642761 cuda_h.py:27] end decode_layer cost 5.831 ms
DEBUG 05-06 10:01:01.642227.642227 lmp.py:1510] ---- decode step 18 layer 10 ----
DEBUG 05-06 10:01:01.648304.648304 cuda_h.py:27] end decode_layer cost 5.843 ms
DEBUG 05-06 10:01:01.648438.648438 lmp.py:1510] ---- decode step 18 layer 11 ----
DEBUG 05-06 10:01:01.654567.654567 cuda_h.py:27] end decode_layer cost 6.197 ms
DEBUG 05-06 10:01:01.654463.654463 lmp.py:1510] ---- decode step 18 layer 12 ----
DEBUG 05-06 10:01:01.660223.660223 cuda_h.py:27] end decode_layer cost 5.891 ms
DEBUG 05-06 10:01:01.660881.660881 lmp.py:1510] ---- decode step 18 layer 13 ----
DEBUG 05-06 10:01:01.666840.666840 cuda_h.py:27] end decode_layer cost 5.897 ms
DEBUG 05-06 10:01:01.666260.666260 lmp.py:1510] ---- decode step 18 layer 14 ----
DEBUG 05-06 10:01:01.672045.672045 cuda_h.py:27] end decode_layer cost 5.839 ms
DEBUG 05-06 10:01:01.672510.672510 lmp.py:1510] ---- decode step 18 layer 15 ----
DEBUG 05-06 10:01:01.678506.678506 cuda_h.py:27] end decode_layer cost 5.784 ms
DEBUG 05-06 10:01:01.678541.678541 lmp.py:1510] ---- decode step 18 layer 16 ----
DEBUG 05-06 10:01:01.684958.684958 cuda_h.py:27] end decode_layer cost 5.743 ms
DEBUG 05-06 10:01:01.684616.684616 lmp.py:1510] ---- decode step 18 layer 17 ----
DEBUG 05-06 10:01:01.690945.690945 cuda_h.py:27] end decode_layer cost 6.065 ms
DEBUG 05-06 10:01:01.690981.690981 lmp.py:1510] ---- decode step 18 layer 18 ----
DEBUG 05-06 10:01:01.695005.695005 cuda_h.py:27] end decode_layer cost 5.664 ms
DEBUG 05-06 10:01:01.696663.696663 lmp.py:1510] ---- decode step 18 layer 19 ----
DEBUG 05-06 10:01:01.701974.701974 cuda_h.py:27] end decode_layer cost 5.736 ms
DEBUG 05-06 10:01:01.701009.701009 lmp.py:1510] ---- decode step 18 layer 20 ----
DEBUG 05-06 10:01:01.707361.707361 cuda_h.py:27] end decode_layer cost 5.554 ms
DEBUG 05-06 10:01:01.707158.707158 lmp.py:1510] ---- decode step 18 layer 21 ----
DEBUG 05-06 10:01:01.713482.713482 cuda_h.py:27] end decode_layer cost 5.709 ms
DEBUG 05-06 10:01:01.713755.713755 lmp.py:1510] ---- decode step 18 layer 22 ----
DEBUG 05-06 10:01:01.719195.719195 cuda_h.py:27] end decode_layer cost 6.006 ms
DEBUG 05-06 10:01:01.719621.719621 lmp.py:1510] ---- decode step 18 layer 23 ----
DEBUG 05-06 10:01:01.725202.725202 cuda_h.py:27] end decode_layer cost 6.075 ms
DEBUG 05-06 10:01:01.725714.725714 lmp.py:1510] ---- decode step 18 layer 24 ----
DEBUG 05-06 10:01:01.731748.731748 cuda_h.py:27] end decode_layer cost 5.741 ms
DEBUG 05-06 10:01:01.731736.731736 lmp.py:1510] ---- decode step 18 layer 25 ----
DEBUG 05-06 10:01:01.737463.737463 cuda_h.py:27] end decode_layer cost 6.457 ms
DEBUG 05-06 10:01:01.737002.737002 lmp.py:1510] ---- decode step 18 layer 26 ----
DEBUG 05-06 10:01:01.743632.743632 cuda_h.py:27] end decode_layer cost 5.936 ms
DEBUG 05-06 10:01:01.743574.743574 lmp.py:1510] ---- decode step 18 layer 27 ----
DEBUG 05-06 10:01:01.749584.749584 cuda_h.py:27] end decode_layer cost 5.829 ms
DEBUG 05-06 10:01:01.749142.749142 lmp.py:1510] ---- decode step 18 layer 28 ----
DEBUG 05-06 10:01:01.755296.755296 cuda_h.py:27] end decode_layer cost 5.759 ms
DEBUG 05-06 10:01:01.755192.755192 lmp.py:1510] ---- decode step 18 layer 29 ----
DEBUG 05-06 10:01:01.761023.761023 cuda_h.py:27] end decode_layer cost 6.013 ms
DEBUG 05-06 10:01:01.761960.761960 cuda_h.py:27] end decode_step cost 185.969 ms
INFO 05-06 10:01:01.761484.761484 lmp.py:1558] decode step 18 time: 0.18600797653198242 seconds
WARNING 05-06 10:01:01.761370.761370 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:01.762230.762230 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:01.762210.762210 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:01.762797.762797 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:01.767603.767603 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:01.767085.767085 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:01.768430.768430 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:01.769250.769250 cuda_h.py:27] end init_inputs_tokens cost 8.192 ms
DEBUG 05-06 10:01:01.770286.770286 lmp.py:1504] decode step 19 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:01.770194.770194 lmp.py:1510] ---- decode step 19 layer 0 ----
DEBUG 05-06 10:01:01.775458.775458 cuda_h.py:27] end decode_layer cost 5.699 ms
DEBUG 05-06 10:01:01.775208.775208 lmp.py:1510] ---- decode step 19 layer 1 ----
DEBUG 05-06 10:01:01.781450.781450 cuda_h.py:27] end decode_layer cost 5.825 ms
DEBUG 05-06 10:01:01.781439.781439 lmp.py:1510] ---- decode step 19 layer 2 ----
DEBUG 05-06 10:01:01.787577.787577 cuda_h.py:27] end decode_layer cost 5.713 ms
DEBUG 05-06 10:01:01.787851.787851 lmp.py:1510] ---- decode step 19 layer 3 ----
DEBUG 05-06 10:01:01.793416.793416 cuda_h.py:27] end decode_layer cost 5.782 ms
DEBUG 05-06 10:01:01.793451.793451 lmp.py:1510] ---- decode step 19 layer 4 ----
DEBUG 05-06 10:01:01.798434.798434 cuda_h.py:27] end decode_layer cost 5.634 ms
DEBUG 05-06 10:01:01.799946.799946 lmp.py:1510] ---- decode step 19 layer 5 ----
DEBUG 05-06 10:01:01.805435.805435 cuda_h.py:27] end decode_layer cost 6.077 ms
DEBUG 05-06 10:01:01.805424.805424 lmp.py:1510] ---- decode step 19 layer 6 ----
DEBUG 05-06 10:01:01.810315.810315 cuda_h.py:27] end decode_layer cost 5.636 ms
DEBUG 05-06 10:01:01.810827.810827 lmp.py:1510] ---- decode step 19 layer 7 ----
DEBUG 05-06 10:01:01.816675.816675 cuda_h.py:27] end decode_layer cost 5.745 ms
DEBUG 05-06 10:01:01.816141.816141 lmp.py:1510] ---- decode step 19 layer 8 ----
DEBUG 05-06 10:01:01.822369.822369 cuda_h.py:27] end decode_layer cost 5.639 ms
DEBUG 05-06 10:01:01.822451.822451 lmp.py:1510] ---- decode step 19 layer 9 ----
DEBUG 05-06 10:01:01.828351.828351 cuda_h.py:27] end decode_layer cost 5.713 ms
DEBUG 05-06 10:01:01.828148.828148 lmp.py:1510] ---- decode step 19 layer 10 ----
DEBUG 05-06 10:01:01.833913.833913 cuda_h.py:27] end decode_layer cost 5.649 ms
DEBUG 05-06 10:01:01.833472.833472 lmp.py:1510] ---- decode step 19 layer 11 ----
DEBUG 05-06 10:01:01.839910.839910 cuda_h.py:27] end decode_layer cost 5.970 ms
DEBUG 05-06 10:01:01.839468.839468 lmp.py:1510] ---- decode step 19 layer 12 ----
DEBUG 05-06 10:01:01.845399.845399 cuda_h.py:27] end decode_layer cost 5.630 ms
DEBUG 05-06 10:01:01.845149.845149 lmp.py:1510] ---- decode step 19 layer 13 ----
DEBUG 05-06 10:01:01.851178.851178 cuda_h.py:27] end decode_layer cost 5.808 ms
DEBUG 05-06 10:01:01.851313.851313 lmp.py:1510] ---- decode step 19 layer 14 ----
DEBUG 05-06 10:01:01.857011.857011 cuda_h.py:27] end decode_layer cost 5.599 ms
DEBUG 05-06 10:01:01.857046.857046 lmp.py:1510] ---- decode step 19 layer 15 ----
DEBUG 05-06 10:01:01.862918.862918 cuda_h.py:27] end decode_layer cost 5.657 ms
DEBUG 05-06 10:01:01.862714.862714 lmp.py:1510] ---- decode step 19 layer 16 ----
DEBUG 05-06 10:01:01.868592.868592 cuda_h.py:27] end decode_layer cost 5.626 ms
DEBUG 05-06 10:01:01.868296.868296 lmp.py:1510] ---- decode step 19 layer 17 ----
DEBUG 05-06 10:01:01.874237.874237 cuda_h.py:27] end decode_layer cost 5.954 ms
DEBUG 05-06 10:01:01.874272.874272 lmp.py:1510] ---- decode step 19 layer 18 ----
DEBUG 05-06 10:01:01.880671.880671 cuda_h.py:27] end decode_layer cost 5.589 ms
DEBUG 05-06 10:01:01.880329.880329 lmp.py:1510] ---- decode step 19 layer 19 ----
DEBUG 05-06 10:01:01.886727.886727 cuda_h.py:27] end decode_layer cost 5.729 ms
DEBUG 05-06 10:01:01.886954.886954 lmp.py:1510] ---- decode step 19 layer 20 ----
DEBUG 05-06 10:01:01.891622.891622 cuda_h.py:27] end decode_layer cost 5.717 ms
DEBUG 05-06 10:01:01.891373.891373 lmp.py:1510] ---- decode step 19 layer 21 ----
DEBUG 05-06 10:01:01.897929.897929 cuda_h.py:27] end decode_layer cost 5.741 ms
DEBUG 05-06 10:01:01.897726.897726 lmp.py:1510] ---- decode step 19 layer 22 ----
DEBUG 05-06 10:01:01.903195.903195 cuda_h.py:27] end decode_layer cost 5.676 ms
DEBUG 05-06 10:01:01.903753.903753 lmp.py:1510] ---- decode step 19 layer 23 ----
DEBUG 05-06 10:01:01.909531.909531 cuda_h.py:27] end decode_layer cost 6.009 ms
DEBUG 05-06 10:01:01.909380.909380 lmp.py:1510] ---- decode step 19 layer 24 ----
DEBUG 05-06 10:01:01.915327.915327 cuda_h.py:27] end decode_layer cost 6.308 ms
DEBUG 05-06 10:01:01.915700.915700 lmp.py:1510] ---- decode step 19 layer 25 ----
DEBUG 05-06 10:01:01.921825.921825 cuda_h.py:27] end decode_layer cost 5.915 ms
DEBUG 05-06 10:01:01.921861.921861 lmp.py:1510] ---- decode step 19 layer 26 ----
DEBUG 05-06 10:01:01.927418.927418 cuda_h.py:27] end decode_layer cost 5.776 ms
DEBUG 05-06 10:01:01.927838.927838 lmp.py:1510] ---- decode step 19 layer 27 ----
DEBUG 05-06 10:01:01.933906.933906 cuda_h.py:27] end decode_layer cost 5.803 ms
DEBUG 05-06 10:01:01.933750.933750 lmp.py:1510] ---- decode step 19 layer 28 ----
DEBUG 05-06 10:01:01.939577.939577 cuda_h.py:27] end decode_layer cost 5.730 ms
DEBUG 05-06 10:01:01.939420.939420 lmp.py:1510] ---- decode step 19 layer 29 ----
DEBUG 05-06 10:01:01.945944.945944 cuda_h.py:27] end decode_layer cost 6.138 ms
DEBUG 05-06 10:01:01.945920.945920 cuda_h.py:27] end decode_step cost 183.867 ms
INFO 05-06 10:01:01.945253.945253 lmp.py:1558] decode step 19 time: 0.18390417098999023 seconds
WARNING 05-06 10:01:01.945595.945595 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:01.946262.946262 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:01.946189.946189 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:01.946822.946822 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:01.951158.951158 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:01.951626.951626 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:01.951734.951734 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:01.953644.953644 cuda_h.py:27] end init_inputs_tokens cost 8.056 ms
DEBUG 05-06 10:01:01.953487.953487 lmp.py:1504] decode step 20 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:01.953879.953879 lmp.py:1510] ---- decode step 20 layer 0 ----
DEBUG 05-06 10:01:01.959732.959732 cuda_h.py:27] end decode_layer cost 5.678 ms
DEBUG 05-06 10:01:01.959350.959350 lmp.py:1510] ---- decode step 20 layer 1 ----
DEBUG 05-06 10:01:01.965778.965778 cuda_h.py:27] end decode_layer cost 5.857 ms
DEBUG 05-06 10:01:01.965005.965005 lmp.py:1510] ---- decode step 20 layer 2 ----
DEBUG 05-06 10:01:01.971796.971796 cuda_h.py:27] end decode_layer cost 5.808 ms
DEBUG 05-06 10:01:01.971308.971308 lmp.py:1510] ---- decode step 20 layer 3 ----
DEBUG 05-06 10:01:01.977054.977054 cuda_h.py:27] end decode_layer cost 5.845 ms
DEBUG 05-06 10:01:01.977519.977519 lmp.py:1510] ---- decode step 20 layer 4 ----
DEBUG 05-06 10:01:01.983077.983077 cuda_h.py:27] end decode_layer cost 5.777 ms
DEBUG 05-06 10:01:01.983543.983543 lmp.py:1510] ---- decode step 20 layer 5 ----
DEBUG 05-06 10:01:01.989358.989358 cuda_h.py:27] end decode_layer cost 6.142 ms
DEBUG 05-06 10:01:01.989777.989777 lmp.py:1510] ---- decode step 20 layer 6 ----
DEBUG 05-06 10:01:01.995643.995643 cuda_h.py:27] end decode_layer cost 5.689 ms
DEBUG 05-06 10:01:01.995155.995155 lmp.py:1510] ---- decode step 20 layer 7 ----
DEBUG 05-06 10:01:02.000282.000282 cuda_h.py:27] end decode_layer cost 5.739 ms
DEBUG 05-06 10:01:02.000747.000747 lmp.py:1510] ---- decode step 20 layer 8 ----
DEBUG 05-06 10:01:02.006552.006552 cuda_h.py:27] end decode_layer cost 5.643 ms
DEBUG 05-06 10:01:02.006634.006634 lmp.py:1510] ---- decode step 20 layer 9 ----
DEBUG 05-06 10:01:02.012582.012582 cuda_h.py:27] end decode_layer cost 5.748 ms
DEBUG 05-06 10:01:02.012809.012809 lmp.py:1510] ---- decode step 20 layer 10 ----
DEBUG 05-06 10:01:02.018995.018995 cuda_h.py:27] end decode_layer cost 5.749 ms
DEBUG 05-06 10:01:02.018553.018553 lmp.py:1510] ---- decode step 20 layer 11 ----
DEBUG 05-06 10:01:02.024497.024497 cuda_h.py:27] end decode_layer cost 6.026 ms
DEBUG 05-06 10:01:02.024724.024724 lmp.py:1510] ---- decode step 20 layer 12 ----
DEBUG 05-06 10:01:02.030346.030346 cuda_h.py:27] end decode_layer cost 5.719 ms
DEBUG 05-06 10:01:02.030858.030858 lmp.py:1510] ---- decode step 20 layer 13 ----
DEBUG 05-06 10:01:02.036365.036365 cuda_h.py:27] end decode_layer cost 5.845 ms
DEBUG 05-06 10:01:02.036308.036308 lmp.py:1510] ---- decode step 20 layer 14 ----
DEBUG 05-06 10:01:02.041912.041912 cuda_h.py:27] end decode_layer cost 5.776 ms
DEBUG 05-06 10:01:02.041424.041424 lmp.py:1510] ---- decode step 20 layer 15 ----
DEBUG 05-06 10:01:02.047974.047974 cuda_h.py:27] end decode_layer cost 5.719 ms
DEBUG 05-06 10:01:02.047155.047155 lmp.py:1510] ---- decode step 20 layer 16 ----
DEBUG 05-06 10:01:02.053137.053137 cuda_h.py:27] end decode_layer cost 5.773 ms
DEBUG 05-06 10:01:02.053079.053079 lmp.py:1510] ---- decode step 20 layer 17 ----
DEBUG 05-06 10:01:02.059302.059302 cuda_h.py:27] end decode_layer cost 6.056 ms
DEBUG 05-06 10:01:02.059622.059622 lmp.py:1510] ---- decode step 20 layer 18 ----
DEBUG 05-06 10:01:02.065234.065234 cuda_h.py:27] end decode_layer cost 5.816 ms
DEBUG 05-06 10:01:02.065938.065938 lmp.py:1510] ---- decode step 20 layer 19 ----
DEBUG 05-06 10:01:02.071293.071293 cuda_h.py:27] end decode_layer cost 5.838 ms
DEBUG 05-06 10:01:02.071474.071474 lmp.py:1510] ---- decode step 20 layer 20 ----
DEBUG 05-06 10:01:02.077758.077758 cuda_h.py:27] end decode_layer cost 5.716 ms
DEBUG 05-06 10:01:02.077509.077509 lmp.py:1510] ---- decode step 20 layer 21 ----
DEBUG 05-06 10:01:02.083237.083237 cuda_h.py:27] end decode_layer cost 5.727 ms
DEBUG 05-06 10:01:02.083034.083034 lmp.py:1510] ---- decode step 20 layer 22 ----
DEBUG 05-06 10:01:02.088170.088170 cuda_h.py:27] end decode_layer cost 5.852 ms
DEBUG 05-06 10:01:02.089205.089205 lmp.py:1510] ---- decode step 20 layer 23 ----
DEBUG 05-06 10:01:02.095727.095727 cuda_h.py:27] end decode_layer cost 6.066 ms
DEBUG 05-06 10:01:02.095762.095762 lmp.py:1510] ---- decode step 20 layer 24 ----
DEBUG 05-06 10:01:02.100305.100305 cuda_h.py:27] end decode_layer cost 5.731 ms
DEBUG 05-06 10:01:02.100963.100963 lmp.py:1510] ---- decode step 20 layer 25 ----
DEBUG 05-06 10:01:02.106201.106201 cuda_h.py:27] end decode_layer cost 5.717 ms
DEBUG 05-06 10:01:02.106951.106951 lmp.py:1510] ---- decode step 20 layer 26 ----
DEBUG 05-06 10:01:02.112495.112495 cuda_h.py:27] end decode_layer cost 5.767 ms
DEBUG 05-06 10:01:02.112153.112153 lmp.py:1510] ---- decode step 20 layer 27 ----
DEBUG 05-06 10:01:02.118345.118345 cuda_h.py:27] end decode_layer cost 5.717 ms
DEBUG 05-06 10:01:02.118380.118380 lmp.py:1510] ---- decode step 20 layer 28 ----
DEBUG 05-06 10:01:02.124976.124976 cuda_h.py:27] end decode_layer cost 5.734 ms
DEBUG 05-06 10:01:02.124395.124395 lmp.py:1510] ---- decode step 20 layer 29 ----
DEBUG 05-06 10:01:02.130558.130558 cuda_h.py:27] end decode_layer cost 6.222 ms
DEBUG 05-06 10:01:02.130826.130826 cuda_h.py:27] end decode_step cost 184.774 ms
INFO 05-06 10:01:02.130065.130065 lmp.py:1558] decode step 20 time: 0.1848139762878418 seconds
WARNING 05-06 10:01:02.130588.130588 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:02.130061.130061 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:02.131047.131047 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:02.131163.131163 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:02.136738.136738 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:02.136968.136968 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:02.136075.136075 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:02.138348.138348 cuda_h.py:27] end init_inputs_tokens cost 7.958 ms
DEBUG 05-06 10:01:02.138290.138290 lmp.py:1504] decode step 21 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:02.138391.138391 lmp.py:1510] ---- decode step 21 layer 0 ----
DEBUG 05-06 10:01:02.144453.144453 cuda_h.py:27] end decode_layer cost 5.796 ms
DEBUG 05-06 10:01:02.144157.144157 lmp.py:1510] ---- decode step 21 layer 1 ----
DEBUG 05-06 10:01:02.150208.150208 cuda_h.py:27] end decode_layer cost 5.860 ms
DEBUG 05-06 10:01:02.150627.150627 lmp.py:1510] ---- decode step 21 layer 2 ----
DEBUG 05-06 10:01:02.156134.156134 cuda_h.py:27] end decode_layer cost 5.845 ms
DEBUG 05-06 10:01:02.156838.156838 lmp.py:1510] ---- decode step 21 layer 3 ----
DEBUG 05-06 10:01:02.162116.162116 cuda_h.py:27] end decode_layer cost 5.922 ms
DEBUG 05-06 10:01:02.162913.162913 lmp.py:1510] ---- decode step 21 layer 4 ----
DEBUG 05-06 10:01:02.168202.168202 cuda_h.py:27] end decode_layer cost 5.649 ms
DEBUG 05-06 10:01:02.168521.168521 lmp.py:1510] ---- decode step 21 layer 5 ----
DEBUG 05-06 10:01:02.174982.174982 cuda_h.py:27] end decode_layer cost 6.020 ms
DEBUG 05-06 10:01:02.174163.174163 lmp.py:1510] ---- decode step 21 layer 6 ----
DEBUG 05-06 10:01:02.179686.179686 cuda_h.py:27] end decode_layer cost 5.751 ms
DEBUG 05-06 10:01:02.179391.179391 lmp.py:1510] ---- decode step 21 layer 7 ----
DEBUG 05-06 10:01:02.185279.185279 cuda_h.py:27] end decode_layer cost 5.776 ms
DEBUG 05-06 10:01:02.185553.185553 lmp.py:1510] ---- decode step 21 layer 8 ----
DEBUG 05-06 10:01:02.191557.191557 cuda_h.py:27] end decode_layer cost 5.649 ms
DEBUG 05-06 10:01:02.191738.191738 lmp.py:1510] ---- decode step 21 layer 9 ----
DEBUG 05-06 10:01:02.197564.197564 cuda_h.py:27] end decode_layer cost 5.869 ms
DEBUG 05-06 10:01:02.197122.197122 lmp.py:1510] ---- decode step 21 layer 10 ----
DEBUG 05-06 10:01:02.203575.203575 cuda_h.py:27] end decode_layer cost 5.805 ms
DEBUG 05-06 10:01:02.203233.203233 lmp.py:1510] ---- decode step 21 layer 11 ----
DEBUG 05-06 10:01:02.209643.209643 cuda_h.py:27] end decode_layer cost 6.125 ms
DEBUG 05-06 10:01:02.209109.209109 lmp.py:1510] ---- decode step 21 layer 12 ----
DEBUG 05-06 10:01:02.215059.215059 cuda_h.py:27] end decode_layer cost 5.821 ms
DEBUG 05-06 10:01:02.215525.215525 lmp.py:1510] ---- decode step 21 layer 13 ----
DEBUG 05-06 10:01:02.221740.221740 cuda_h.py:27] end decode_layer cost 5.840 ms
DEBUG 05-06 10:01:02.221021.221021 lmp.py:1510] ---- decode step 21 layer 14 ----
DEBUG 05-06 10:01:02.227526.227526 cuda_h.py:27] end decode_layer cost 5.773 ms
DEBUG 05-06 10:01:02.227322.227322 lmp.py:1510] ---- decode step 21 layer 15 ----
DEBUG 05-06 10:01:02.233611.233611 cuda_h.py:27] end decode_layer cost 5.859 ms
DEBUG 05-06 10:01:02.233031.233031 lmp.py:1510] ---- decode step 21 layer 16 ----
DEBUG 05-06 10:01:02.238962.238962 cuda_h.py:27] end decode_layer cost 5.666 ms
DEBUG 05-06 10:01:02.238713.238713 lmp.py:1510] ---- decode step 21 layer 17 ----
DEBUG 05-06 10:01:02.245641.245641 cuda_h.py:27] end decode_layer cost 6.155 ms
DEBUG 05-06 10:01:02.245537.245537 lmp.py:1510] ---- decode step 21 layer 18 ----
DEBUG 05-06 10:01:02.250211.250211 cuda_h.py:27] end decode_layer cost 5.687 ms
DEBUG 05-06 10:01:02.250008.250008 lmp.py:1510] ---- decode step 21 layer 19 ----
DEBUG 05-06 10:01:02.256945.256945 cuda_h.py:27] end decode_layer cost 5.810 ms
DEBUG 05-06 10:01:02.256172.256172 lmp.py:1510] ---- decode step 21 layer 20 ----
DEBUG 05-06 10:01:02.262346.262346 cuda_h.py:27] end decode_layer cost 5.600 ms
DEBUG 05-06 10:01:02.262812.262812 lmp.py:1510] ---- decode step 21 layer 21 ----
DEBUG 05-06 10:01:02.268355.268355 cuda_h.py:27] end decode_layer cost 5.942 ms
DEBUG 05-06 10:01:02.268867.268867 lmp.py:1510] ---- decode step 21 layer 22 ----
DEBUG 05-06 10:01:02.273372.273372 cuda_h.py:27] end decode_layer cost 5.562 ms
DEBUG 05-06 10:01:02.274407.274407 lmp.py:1510] ---- decode step 21 layer 23 ----
DEBUG 05-06 10:01:02.279128.279128 cuda_h.py:27] end decode_layer cost 5.896 ms
DEBUG 05-06 10:01:02.280593.280593 lmp.py:1510] ---- decode step 21 layer 24 ----
DEBUG 05-06 10:01:02.285938.285938 cuda_h.py:27] end decode_layer cost 5.725 ms
DEBUG 05-06 10:01:02.285165.285165 lmp.py:1510] ---- decode step 21 layer 25 ----
DEBUG 05-06 10:01:02.291978.291978 cuda_h.py:27] end decode_layer cost 5.684 ms
DEBUG 05-06 10:01:02.291252.291252 lmp.py:1510] ---- decode step 21 layer 26 ----
DEBUG 05-06 10:01:02.297847.297847 cuda_h.py:27] end decode_layer cost 5.699 ms
DEBUG 05-06 10:01:02.297643.297643 lmp.py:1510] ---- decode step 21 layer 27 ----
DEBUG 05-06 10:01:02.303550.303550 cuda_h.py:27] end decode_layer cost 5.718 ms
DEBUG 05-06 10:01:02.303870.303870 lmp.py:1510] ---- decode step 21 layer 28 ----
DEBUG 05-06 10:01:02.308061.308061 cuda_h.py:27] end decode_layer cost 5.681 ms
DEBUG 05-06 10:01:02.308778.308778 lmp.py:1510] ---- decode step 21 layer 29 ----
DEBUG 05-06 10:01:02.314037.314037 cuda_h.py:27] end decode_layer cost 5.943 ms
DEBUG 05-06 10:01:02.314490.314490 cuda_h.py:27] end decode_step cost 184.297 ms
INFO 05-06 10:01:02.314584.314584 lmp.py:1558] decode step 21 time: 0.18433451652526855 seconds
WARNING 05-06 10:01:02.315973.315973 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:02.315938.315938 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:02.315044.315044 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:02.316436.316436 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:02.321661.321661 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:02.321029.321029 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:02.321945.321945 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:02.323901.323901 cuda_h.py:27] end init_inputs_tokens cost 8.212 ms
DEBUG 05-06 10:01:02.323698.323698 lmp.py:1504] decode step 22 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:02.323891.323891 lmp.py:1510] ---- decode step 22 layer 0 ----
DEBUG 05-06 10:01:02.329886.329886 cuda_h.py:27] end decode_layer cost 5.782 ms
DEBUG 05-06 10:01:02.329398.329398 lmp.py:1510] ---- decode step 22 layer 1 ----
DEBUG 05-06 10:01:02.334385.334385 cuda_h.py:27] end decode_layer cost 5.742 ms
DEBUG 05-06 10:01:02.334897.334897 lmp.py:1510] ---- decode step 22 layer 2 ----
DEBUG 05-06 10:01:02.340481.340481 cuda_h.py:27] end decode_layer cost 5.936 ms
DEBUG 05-06 10:01:02.340715.340715 lmp.py:1510] ---- decode step 22 layer 3 ----
DEBUG 05-06 10:01:02.346360.346360 cuda_h.py:27] end decode_layer cost 5.806 ms
DEBUG 05-06 10:01:02.346679.346679 lmp.py:1510] ---- decode step 22 layer 4 ----
DEBUG 05-06 10:01:02.352979.352979 cuda_h.py:27] end decode_layer cost 5.761 ms
DEBUG 05-06 10:01:02.352159.352159 lmp.py:1510] ---- decode step 22 layer 5 ----
DEBUG 05-06 10:01:02.358808.358808 cuda_h.py:27] end decode_layer cost 5.914 ms
DEBUG 05-06 10:01:02.358327.358327 lmp.py:1510] ---- decode step 22 layer 6 ----
DEBUG 05-06 10:01:02.364188.364188 cuda_h.py:27] end decode_layer cost 5.544 ms
DEBUG 05-06 10:01:02.364747.364747 lmp.py:1510] ---- decode step 22 layer 7 ----
DEBUG 05-06 10:01:02.369119.369119 cuda_h.py:27] end decode_layer cost 5.569 ms
DEBUG 05-06 10:01:02.369154.369154 lmp.py:1510] ---- decode step 22 layer 8 ----
DEBUG 05-06 10:01:02.375300.375300 cuda_h.py:27] end decode_layer cost 5.543 ms
DEBUG 05-06 10:01:02.375858.375858 lmp.py:1510] ---- decode step 22 layer 9 ----
DEBUG 05-06 10:01:02.381387.381387 cuda_h.py:27] end decode_layer cost 5.685 ms
DEBUG 05-06 10:01:02.381375.381375 lmp.py:1510] ---- decode step 22 layer 10 ----
DEBUG 05-06 10:01:02.386323.386323 cuda_h.py:27] end decode_layer cost 5.537 ms
DEBUG 05-06 10:01:02.386643.386643 lmp.py:1510] ---- decode step 22 layer 11 ----
DEBUG 05-06 10:01:02.392996.392996 cuda_h.py:27] end decode_layer cost 6.013 ms
DEBUG 05-06 10:01:02.392601.392601 lmp.py:1510] ---- decode step 22 layer 12 ----
DEBUG 05-06 10:01:02.398696.398696 cuda_h.py:27] end decode_layer cost 5.611 ms
DEBUG 05-06 10:01:02.398493.398493 lmp.py:1510] ---- decode step 22 layer 13 ----
DEBUG 05-06 10:01:02.404348.404348 cuda_h.py:27] end decode_layer cost 5.750 ms
DEBUG 05-06 10:01:02.404768.404768 lmp.py:1510] ---- decode step 22 layer 14 ----
DEBUG 05-06 10:01:02.410374.410374 cuda_h.py:27] end decode_layer cost 5.637 ms
DEBUG 05-06 10:01:02.410124.410124 lmp.py:1510] ---- decode step 22 layer 15 ----
DEBUG 05-06 10:01:02.415195.415195 cuda_h.py:27] end decode_layer cost 5.663 ms
DEBUG 05-06 10:01:02.415753.415753 lmp.py:1510] ---- decode step 22 layer 16 ----
DEBUG 05-06 10:01:02.421422.421422 cuda_h.py:27] end decode_layer cost 5.543 ms
DEBUG 05-06 10:01:02.421219.421219 lmp.py:1510] ---- decode step 22 layer 17 ----
DEBUG 05-06 10:01:02.427542.427542 cuda_h.py:27] end decode_layer cost 5.885 ms
DEBUG 05-06 10:01:02.427577.427577 lmp.py:1510] ---- decode step 22 layer 18 ----
DEBUG 05-06 10:01:02.433338.433338 cuda_h.py:27] end decode_layer cost 5.716 ms
DEBUG 05-06 10:01:02.433374.433374 lmp.py:1510] ---- decode step 22 layer 19 ----
DEBUG 05-06 10:01:02.438417.438417 cuda_h.py:27] end decode_layer cost 5.643 ms
DEBUG 05-06 10:01:02.438214.438214 lmp.py:1510] ---- decode step 22 layer 20 ----
DEBUG 05-06 10:01:02.444162.444162 cuda_h.py:27] end decode_layer cost 5.959 ms
DEBUG 05-06 10:01:02.444005.444005 lmp.py:1510] ---- decode step 22 layer 21 ----
DEBUG 05-06 10:01:02.450671.450671 cuda_h.py:27] end decode_layer cost 5.646 ms
DEBUG 05-06 10:01:02.450945.450945 lmp.py:1510] ---- decode step 22 layer 22 ----
DEBUG 05-06 10:01:02.456072.456072 cuda_h.py:27] end decode_layer cost 5.564 ms
DEBUG 05-06 10:01:02.456398.456398 lmp.py:1510] ---- decode step 22 layer 23 ----
DEBUG 05-06 10:01:02.462598.462598 cuda_h.py:27] end decode_layer cost 5.970 ms
DEBUG 05-06 10:01:02.462349.462349 lmp.py:1510] ---- decode step 22 layer 24 ----
DEBUG 05-06 10:01:02.467284.467284 cuda_h.py:27] end decode_layer cost 5.563 ms
DEBUG 05-06 10:01:02.468796.468796 lmp.py:1510] ---- decode step 22 layer 25 ----
DEBUG 05-06 10:01:02.473044.473044 cuda_h.py:27] end decode_layer cost 5.619 ms
DEBUG 05-06 10:01:02.473556.473556 lmp.py:1510] ---- decode step 22 layer 26 ----
DEBUG 05-06 10:01:02.479861.479861 cuda_h.py:27] end decode_layer cost 5.556 ms
DEBUG 05-06 10:01:02.479896.479896 lmp.py:1510] ---- decode step 22 layer 27 ----
DEBUG 05-06 10:01:02.485076.485076 cuda_h.py:27] end decode_layer cost 5.743 ms
DEBUG 05-06 10:01:02.485402.485402 lmp.py:1510] ---- decode step 22 layer 28 ----
DEBUG 05-06 10:01:02.490694.490694 cuda_h.py:27] end decode_layer cost 5.546 ms
DEBUG 05-06 10:01:02.490014.490014 lmp.py:1510] ---- decode step 22 layer 29 ----
DEBUG 05-06 10:01:02.496344.496344 cuda_h.py:27] end decode_layer cost 5.890 ms
DEBUG 05-06 10:01:02.496281.496281 cuda_h.py:27] end decode_step cost 181.742 ms
INFO 05-06 10:01:02.496044.496044 lmp.py:1558] decode step 22 time: 0.18178009986877441 seconds
WARNING 05-06 10:01:02.496400.496400 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:02.497117.497117 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:02.498202.498202 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:02.498421.498421 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:02.503593.503593 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:02.503353.503353 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:02.503937.503937 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:02.505381.505381 cuda_h.py:27] end init_inputs_tokens cost 8.438 ms
DEBUG 05-06 10:01:02.505509.505509 lmp.py:1504] decode step 23 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:02.505418.505418 lmp.py:1510] ---- decode step 23 layer 0 ----
DEBUG 05-06 10:01:02.510108.510108 cuda_h.py:27] end decode_layer cost 5.592 ms
DEBUG 05-06 10:01:02.511289.511289 lmp.py:1510] ---- decode step 23 layer 1 ----
DEBUG 05-06 10:01:02.516388.516388 cuda_h.py:27] end decode_layer cost 5.720 ms
DEBUG 05-06 10:01:02.516231.516231 lmp.py:1510] ---- decode step 23 layer 2 ----
DEBUG 05-06 10:01:02.522748.522748 cuda_h.py:27] end decode_layer cost 5.536 ms
DEBUG 05-06 10:01:02.522498.522498 lmp.py:1510] ---- decode step 23 layer 3 ----
DEBUG 05-06 10:01:02.528131.528131 cuda_h.py:27] end decode_layer cost 5.621 ms
DEBUG 05-06 10:01:02.528735.528735 lmp.py:1510] ---- decode step 23 layer 4 ----
DEBUG 05-06 10:01:02.533129.533129 cuda_h.py:27] end decode_layer cost 5.621 ms
DEBUG 05-06 10:01:02.533118.533118 lmp.py:1510] ---- decode step 23 layer 5 ----
DEBUG 05-06 10:01:02.540341.540341 cuda_h.py:27] end decode_layer cost 6.267 ms
DEBUG 05-06 10:01:02.540191.540191 lmp.py:1510] ---- decode step 23 layer 6 ----
DEBUG 05-06 10:01:02.546560.546560 cuda_h.py:27] end decode_layer cost 5.884 ms
DEBUG 05-06 10:01:02.546357.546357 lmp.py:1510] ---- decode step 23 layer 7 ----
DEBUG 05-06 10:01:02.551976.551976 cuda_h.py:27] end decode_layer cost 5.610 ms
DEBUG 05-06 10:01:02.551057.551057 lmp.py:1510] ---- decode step 23 layer 8 ----
DEBUG 05-06 10:01:02.557442.557442 cuda_h.py:27] end decode_layer cost 5.544 ms
DEBUG 05-06 10:01:02.557238.557238 lmp.py:1510] ---- decode step 23 layer 9 ----
DEBUG 05-06 10:01:02.563394.563394 cuda_h.py:27] end decode_layer cost 5.621 ms
DEBUG 05-06 10:01:02.563429.563429 lmp.py:1510] ---- decode step 23 layer 10 ----
DEBUG 05-06 10:01:02.568204.568204 cuda_h.py:27] end decode_layer cost 5.551 ms
DEBUG 05-06 10:01:02.568240.568240 lmp.py:1510] ---- decode step 23 layer 11 ----
DEBUG 05-06 10:01:02.574358.574358 cuda_h.py:27] end decode_layer cost 5.909 ms
DEBUG 05-06 10:01:02.574201.574201 lmp.py:1510] ---- decode step 23 layer 12 ----
DEBUG 05-06 10:01:02.580131.580131 cuda_h.py:27] end decode_layer cost 5.805 ms
DEBUG 05-06 10:01:02.580073.580073 lmp.py:1510] ---- decode step 23 layer 13 ----
DEBUG 05-06 10:01:02.586483.586483 cuda_h.py:27] end decode_layer cost 5.703 ms
DEBUG 05-06 10:01:02.586472.586472 lmp.py:1510] ---- decode step 23 layer 14 ----
DEBUG 05-06 10:01:02.591442.591442 cuda_h.py:27] end decode_layer cost 5.624 ms
DEBUG 05-06 10:01:02.591192.591192 lmp.py:1510] ---- decode step 23 layer 15 ----
DEBUG 05-06 10:01:02.597999.597999 cuda_h.py:27] end decode_layer cost 5.679 ms
DEBUG 05-06 10:01:02.597080.597080 lmp.py:1510] ---- decode step 23 layer 16 ----
DEBUG 05-06 10:01:02.603562.603562 cuda_h.py:27] end decode_layer cost 5.686 ms
DEBUG 05-06 10:01:02.603458.603458 lmp.py:1510] ---- decode step 23 layer 17 ----
DEBUG 05-06 10:01:02.609198.609198 cuda_h.py:27] end decode_layer cost 6.051 ms
DEBUG 05-06 10:01:02.609994.609994 lmp.py:1510] ---- decode step 23 layer 18 ----
DEBUG 05-06 10:01:02.615896.615896 cuda_h.py:27] end decode_layer cost 5.574 ms
DEBUG 05-06 10:01:02.615839.615839 lmp.py:1510] ---- decode step 23 layer 19 ----
DEBUG 05-06 10:01:02.621898.621898 cuda_h.py:27] end decode_layer cost 5.900 ms
DEBUG 05-06 10:01:02.621648.621648 lmp.py:1510] ---- decode step 23 layer 20 ----
DEBUG 05-06 10:01:02.626816.626816 cuda_h.py:27] end decode_layer cost 5.594 ms
DEBUG 05-06 10:01:02.626997.626997 lmp.py:1510] ---- decode step 23 layer 21 ----
DEBUG 05-06 10:01:02.632170.632170 cuda_h.py:27] end decode_layer cost 5.774 ms
DEBUG 05-06 10:01:02.632636.632636 lmp.py:1510] ---- decode step 23 layer 22 ----
DEBUG 05-06 10:01:02.638204.638204 cuda_h.py:27] end decode_layer cost 5.889 ms
DEBUG 05-06 10:01:02.638591.638591 lmp.py:1510] ---- decode step 23 layer 23 ----
DEBUG 05-06 10:01:02.644609.644609 cuda_h.py:27] end decode_layer cost 6.256 ms
DEBUG 05-06 10:01:02.645220.645220 lmp.py:1510] ---- decode step 23 layer 24 ----
DEBUG 05-06 10:01:02.650760.650760 cuda_h.py:27] end decode_layer cost 5.834 ms
DEBUG 05-06 10:01:02.650226.650226 lmp.py:1510] ---- decode step 23 layer 25 ----
DEBUG 05-06 10:01:02.656752.656752 cuda_h.py:27] end decode_layer cost 5.824 ms
DEBUG 05-06 10:01:02.656662.656662 lmp.py:1510] ---- decode step 23 layer 26 ----
DEBUG 05-06 10:01:02.662708.662708 cuda_h.py:27] end decode_layer cost 5.715 ms
DEBUG 05-06 10:01:02.662842.662842 lmp.py:1510] ---- decode step 23 layer 27 ----
DEBUG 05-06 10:01:02.668327.668327 cuda_h.py:27] end decode_layer cost 5.968 ms
DEBUG 05-06 10:01:02.668700.668700 lmp.py:1510] ---- decode step 23 layer 28 ----
DEBUG 05-06 10:01:02.674537.674537 cuda_h.py:27] end decode_layer cost 5.807 ms
DEBUG 05-06 10:01:02.674095.674095 lmp.py:1510] ---- decode step 23 layer 29 ----
DEBUG 05-06 10:01:02.680239.680239 cuda_h.py:27] end decode_layer cost 6.068 ms
DEBUG 05-06 10:01:02.680746.680746 cuda_h.py:27] end decode_step cost 183.887 ms
INFO 05-06 10:01:02.680416.680416 lmp.py:1558] decode step 23 time: 0.1839277744293213 seconds
WARNING 05-06 10:01:02.680415.680415 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:02.681067.681067 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:02.681497.681497 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:02.681991.681991 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:02.686009.686009 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:02.686186.686186 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:02.686863.686863 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 10:01:02.687148.687148 helper.py:80] WARNING: Logits have extreme values: min=-688.00, max=1336.00
WARNING 05-06 10:01:02.687363.687363 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 10:01:02.688596.688596 cuda_h.py:27] end init_inputs_tokens cost 8.063 ms
DEBUG 05-06 10:01:02.688392.688392 lmp.py:1504] decode step 24 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:02.688063.688063 lmp.py:1510] ---- decode step 24 layer 0 ----
DEBUG 05-06 10:01:02.694569.694569 cuda_h.py:27] end decode_layer cost 5.633 ms
DEBUG 05-06 10:01:02.694081.694081 lmp.py:1510] ---- decode step 24 layer 1 ----
DEBUG 05-06 10:01:02.700261.700261 cuda_h.py:27] end decode_layer cost 5.743 ms
DEBUG 05-06 10:01:02.700965.700965 lmp.py:1510] ---- decode step 24 layer 2 ----
DEBUG 05-06 10:01:02.706297.706297 cuda_h.py:27] end decode_layer cost 5.576 ms
DEBUG 05-06 10:01:02.706856.706856 lmp.py:1510] ---- decode step 24 layer 3 ----
DEBUG 05-06 10:01:02.711971.711971 cuda_h.py:27] end decode_layer cost 5.627 ms
DEBUG 05-06 10:01:02.711444.711444 lmp.py:1510] ---- decode step 24 layer 4 ----
DEBUG 05-06 10:01:02.717657.717657 cuda_h.py:27] end decode_layer cost 5.558 ms
DEBUG 05-06 10:01:02.717692.717692 lmp.py:1510] ---- decode step 24 layer 5 ----
DEBUG 05-06 10:01:02.723718.723718 cuda_h.py:27] end decode_layer cost 5.911 ms
DEBUG 05-06 10:01:02.723276.723276 lmp.py:1510] ---- decode step 24 layer 6 ----
DEBUG 05-06 10:01:02.729058.729058 cuda_h.py:27] end decode_layer cost 5.556 ms
DEBUG 05-06 10:01:02.729047.729047 lmp.py:1510] ---- decode step 24 layer 7 ----
DEBUG 05-06 10:01:02.734803.734803 cuda_h.py:27] end decode_layer cost 5.572 ms
DEBUG 05-06 10:01:02.734839.734839 lmp.py:1510] ---- decode step 24 layer 8 ----
DEBUG 05-06 10:01:02.740062.740062 cuda_h.py:27] end decode_layer cost 5.671 ms
DEBUG 05-06 10:01:02.740004.740004 lmp.py:1510] ---- decode step 24 layer 9 ----
DEBUG 05-06 10:01:02.746491.746491 cuda_h.py:27] end decode_layer cost 5.619 ms
DEBUG 05-06 10:01:02.746764.746764 lmp.py:1510] ---- decode step 24 layer 10 ----
DEBUG 05-06 10:01:02.751222.751222 cuda_h.py:27] end decode_layer cost 5.563 ms
DEBUG 05-06 10:01:02.751211.751211 lmp.py:1510] ---- decode step 24 layer 11 ----
DEBUG 05-06 10:01:02.757640.757640 cuda_h.py:27] end decode_layer cost 5.893 ms
DEBUG 05-06 10:01:02.757437.757437 lmp.py:1510] ---- decode step 24 layer 12 ----
DEBUG 05-06 10:01:02.763352.763352 cuda_h.py:27] end decode_layer cost 5.548 ms
DEBUG 05-06 10:01:02.763625.763625 lmp.py:1510] ---- decode step 24 layer 13 ----
DEBUG 05-06 10:01:02.769213.769213 cuda_h.py:27] end decode_layer cost 5.694 ms
DEBUG 05-06 10:01:02.769441.769441 lmp.py:1510] ---- decode step 24 layer 14 ----
DEBUG 05-06 10:01:02.774806.774806 cuda_h.py:27] end decode_layer cost 5.564 ms
DEBUG 05-06 10:01:02.774364.774364 lmp.py:1510] ---- decode step 24 layer 15 ----
DEBUG 05-06 10:01:02.780898.780898 cuda_h.py:27] end decode_layer cost 5.654 ms
DEBUG 05-06 10:01:02.780410.780410 lmp.py:1510] ---- decode step 24 layer 16 ----
DEBUG 05-06 10:01:02.786965.786965 cuda_h.py:27] end decode_layer cost 5.669 ms
DEBUG 05-06 10:01:02.786761.786761 lmp.py:1510] ---- decode step 24 layer 17 ----
DEBUG 05-06 10:01:02.792142.792142 cuda_h.py:27] end decode_layer cost 6.031 ms
DEBUG 05-06 10:01:02.792085.792085 lmp.py:1510] ---- decode step 24 layer 18 ----
DEBUG 05-06 10:01:02.797010.797010 cuda_h.py:27] end decode_layer cost 5.661 ms
DEBUG 05-06 10:01:02.798853.798853 lmp.py:1510] ---- decode step 24 layer 19 ----
DEBUG 05-06 10:01:02.803774.803774 cuda_h.py:27] end decode_layer cost 5.938 ms
DEBUG 05-06 10:01:02.804670.804670 lmp.py:1510] ---- decode step 24 layer 20 ----
DEBUG 05-06 10:01:02.809278.809278 cuda_h.py:27] end decode_layer cost 5.884 ms
DEBUG 05-06 10:01:02.809889.809889 lmp.py:1510] ---- decode step 24 layer 21 ----
DEBUG 05-06 10:01:02.815689.815689 cuda_h.py:27] end decode_layer cost 5.675 ms
DEBUG 05-06 10:01:02.815962.815962 lmp.py:1510] ---- decode step 24 layer 22 ----
DEBUG 05-06 10:01:02.821354.821354 cuda_h.py:27] end decode_layer cost 5.549 ms
DEBUG 05-06 10:01:02.821912.821912 lmp.py:1510] ---- decode step 24 layer 23 ----
DEBUG 05-06 10:01:02.827354.827354 cuda_h.py:27] end decode_layer cost 5.867 ms
DEBUG 05-06 10:01:02.827342.827342 lmp.py:1510] ---- decode step 24 layer 24 ----
DEBUG 05-06 10:01:02.832683.832683 cuda_h.py:27] end decode_layer cost 5.616 ms
DEBUG 05-06 10:01:02.832387.832387 lmp.py:1510] ---- decode step 24 layer 25 ----
DEBUG 05-06 10:01:02.838261.838261 cuda_h.py:27] end decode_layer cost 5.729 ms
DEBUG 05-06 10:01:02.838773.838773 lmp.py:1510] ---- decode step 24 layer 26 ----
DEBUG 05-06 10:01:02.844167.844167 cuda_h.py:27] end decode_layer cost 5.621 ms
DEBUG 05-06 10:01:02.844679.844679 lmp.py:1510] ---- decode step 24 layer 27 ----
DEBUG 05-06 10:01:02.850735.850735 cuda_h.py:27] end decode_layer cost 5.617 ms
DEBUG 05-06 10:01:02.850293.850293 lmp.py:1510] ---- decode step 24 layer 28 ----
DEBUG 05-06 10:01:02.855791.855791 cuda_h.py:27] end decode_layer cost 5.557 ms
DEBUG 05-06 10:01:02.855349.855349 lmp.py:1510] ---- decode step 24 layer 29 ----
DEBUG 05-06 10:01:02.861871.861871 cuda_h.py:27] end decode_layer cost 5.891 ms
DEBUG 05-06 10:01:02.861563.861563 cuda_h.py:27] end decode_step cost 180.937 ms
INFO 05-06 10:01:02.861133.861133 lmp.py:1558] decode step 24 time: 0.18097376823425293 seconds
WARNING 05-06 10:01:02.861191.861191 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:02.862849.862849 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:02.863651.863651 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:02.863046.863046 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:02.868372.868372 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:02.868310.868310 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:02.868702.868702 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:02.870156.870156 cuda_h.py:27] end init_inputs_tokens cost 8.521 ms
DEBUG 05-06 10:01:02.870429.870429 lmp.py:1504] decode step 25 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:02.870815.870815 lmp.py:1510] ---- decode step 25 layer 0 ----
DEBUG 05-06 10:01:02.876082.876082 cuda_h.py:27] end decode_layer cost 5.597 ms
DEBUG 05-06 10:01:02.876832.876832 lmp.py:1510] ---- decode step 25 layer 1 ----
DEBUG 05-06 10:01:02.881206.881206 cuda_h.py:27] end decode_layer cost 5.606 ms
DEBUG 05-06 10:01:02.881241.881241 lmp.py:1510] ---- decode step 25 layer 2 ----
DEBUG 05-06 10:01:02.887462.887462 cuda_h.py:27] end decode_layer cost 5.598 ms
DEBUG 05-06 10:01:02.887735.887735 lmp.py:1510] ---- decode step 25 layer 3 ----
DEBUG 05-06 10:01:02.893412.893412 cuda_h.py:27] end decode_layer cost 5.759 ms
DEBUG 05-06 10:01:02.893745.893745 lmp.py:1510] ---- decode step 25 layer 4 ----
DEBUG 05-06 10:01:02.898808.898808 cuda_h.py:27] end decode_layer cost 5.622 ms
DEBUG 05-06 10:01:02.899558.899558 lmp.py:1510] ---- decode step 25 layer 5 ----
DEBUG 05-06 10:01:02.904951.904951 cuda_h.py:27] end decode_layer cost 5.970 ms
DEBUG 05-06 10:01:02.905463.905463 lmp.py:1510] ---- decode step 25 layer 6 ----
DEBUG 05-06 10:01:02.910900.910900 cuda_h.py:27] end decode_layer cost 5.548 ms
DEBUG 05-06 10:01:02.910174.910174 lmp.py:1510] ---- decode step 25 layer 7 ----
DEBUG 05-06 10:01:02.916983.916983 cuda_h.py:27] end decode_layer cost 5.576 ms
DEBUG 05-06 10:01:02.916495.916495 lmp.py:1510] ---- decode step 25 layer 8 ----
DEBUG 05-06 10:01:02.921979.921979 cuda_h.py:27] end decode_layer cost 5.547 ms
DEBUG 05-06 10:01:02.921491.921491 lmp.py:1510] ---- decode step 25 layer 9 ----
DEBUG 05-06 10:01:02.927408.927408 cuda_h.py:27] end decode_layer cost 5.620 ms
DEBUG 05-06 10:01:02.927642.927642 lmp.py:1510] ---- decode step 25 layer 10 ----
DEBUG 05-06 10:01:02.933274.933274 cuda_h.py:27] end decode_layer cost 5.621 ms
DEBUG 05-06 10:01:02.933694.933694 lmp.py:1510] ---- decode step 25 layer 11 ----
DEBUG 05-06 10:01:02.939978.939978 cuda_h.py:27] end decode_layer cost 5.927 ms
DEBUG 05-06 10:01:02.939729.939729 lmp.py:1510] ---- decode step 25 layer 12 ----
DEBUG 05-06 10:01:02.945930.945930 cuda_h.py:27] end decode_layer cost 5.794 ms
DEBUG 05-06 10:01:02.945349.945349 lmp.py:1510] ---- decode step 25 layer 13 ----
DEBUG 05-06 10:01:02.951165.951165 cuda_h.py:27] end decode_layer cost 5.967 ms
DEBUG 05-06 10:01:02.951346.951346 lmp.py:1510] ---- decode step 25 layer 14 ----
DEBUG 05-06 10:01:02.957521.957521 cuda_h.py:27] end decode_layer cost 5.811 ms
DEBUG 05-06 10:01:02.957794.957794 lmp.py:1510] ---- decode step 25 layer 15 ----
DEBUG 05-06 10:01:02.962223.962223 cuda_h.py:27] end decode_layer cost 5.681 ms
DEBUG 05-06 10:01:02.962881.962881 lmp.py:1510] ---- decode step 25 layer 16 ----
DEBUG 05-06 10:01:02.968978.968978 cuda_h.py:27] end decode_layer cost 5.858 ms
DEBUG 05-06 10:01:02.968298.968298 lmp.py:1510] ---- decode step 25 layer 17 ----
DEBUG 05-06 10:01:02.974910.974910 cuda_h.py:27] end decode_layer cost 6.028 ms
DEBUG 05-06 10:01:02.974476.974476 lmp.py:1510] ---- decode step 25 layer 18 ----
DEBUG 05-06 10:01:02.980986.980986 cuda_h.py:27] end decode_layer cost 5.741 ms
DEBUG 05-06 10:01:02.980213.980213 lmp.py:1510] ---- decode step 25 layer 19 ----
DEBUG 05-06 10:01:02.986657.986657 cuda_h.py:27] end decode_layer cost 5.728 ms
DEBUG 05-06 10:01:02.986599.986599 lmp.py:1510] ---- decode step 25 layer 20 ----
DEBUG 05-06 10:01:02.992500.992500 cuda_h.py:27] end decode_layer cost 5.713 ms
DEBUG 05-06 10:01:02.992488.992488 lmp.py:1510] ---- decode step 25 layer 21 ----
DEBUG 05-06 10:01:02.997063.997063 cuda_h.py:27] end decode_layer cost 5.684 ms
DEBUG 05-06 10:01:02.998337.998337 lmp.py:1510] ---- decode step 25 layer 22 ----
DEBUG 05-06 10:01:03.003577.003577 cuda_h.py:27] end decode_layer cost 5.577 ms
DEBUG 05-06 10:01:03.003526.003526 lmp.py:1510] ---- decode step 25 layer 23 ----
DEBUG 05-06 10:01:03.009194.009194 cuda_h.py:27] end decode_layer cost 5.893 ms
DEBUG 05-06 10:01:03.009944.009944 lmp.py:1510] ---- decode step 25 layer 24 ----
DEBUG 05-06 10:01:03.015807.015807 cuda_h.py:27] end decode_layer cost 5.581 ms
DEBUG 05-06 10:01:03.015365.015365 lmp.py:1510] ---- decode step 25 layer 25 ----
DEBUG 05-06 10:01:03.020959.020959 cuda_h.py:27] end decode_layer cost 5.662 ms
DEBUG 05-06 10:01:03.020040.020040 lmp.py:1510] ---- decode step 25 layer 26 ----
DEBUG 05-06 10:01:03.026241.026241 cuda_h.py:27] end decode_layer cost 5.584 ms
DEBUG 05-06 10:01:03.026991.026991 lmp.py:1510] ---- decode step 25 layer 27 ----
DEBUG 05-06 10:01:03.032606.032606 cuda_h.py:27] end decode_layer cost 5.714 ms
DEBUG 05-06 10:01:03.032595.032595 lmp.py:1510] ---- decode step 25 layer 28 ----
DEBUG 05-06 10:01:03.038128.038128 cuda_h.py:27] end decode_layer cost 5.617 ms
DEBUG 05-06 10:01:03.038786.038786 lmp.py:1510] ---- decode step 25 layer 29 ----
DEBUG 05-06 10:01:03.044230.044230 cuda_h.py:27] end decode_layer cost 5.939 ms
DEBUG 05-06 10:01:03.044160.044160 cuda_h.py:27] end decode_step cost 182.301 ms
INFO 05-06 10:01:03.044207.044207 lmp.py:1558] decode step 25 time: 0.18233847618103027 seconds
WARNING 05-06 10:01:03.044365.044365 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:03.044936.044936 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:03.044855.044855 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:03.045157.045157 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:03.050526.050526 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:03.050511.050511 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:03.050472.050472 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:03.052473.052473 cuda_h.py:27] end init_inputs_tokens cost 7.858 ms
DEBUG 05-06 10:01:03.052130.052130 lmp.py:1504] decode step 26 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:03.052993.052993 lmp.py:1510] ---- decode step 26 layer 0 ----
DEBUG 05-06 10:01:03.058438.058438 cuda_h.py:27] end decode_layer cost 5.974 ms
DEBUG 05-06 10:01:03.058235.058235 lmp.py:1510] ---- decode step 26 layer 1 ----
DEBUG 05-06 10:01:03.063894.063894 cuda_h.py:27] end decode_layer cost 5.641 ms
DEBUG 05-06 10:01:03.063168.063168 lmp.py:1510] ---- decode step 26 layer 2 ----
DEBUG 05-06 10:01:03.069612.069612 cuda_h.py:27] end decode_layer cost 5.552 ms
DEBUG 05-06 10:01:03.069409.069409 lmp.py:1510] ---- decode step 26 layer 3 ----
DEBUG 05-06 10:01:03.075219.075219 cuda_h.py:27] end decode_layer cost 5.612 ms
DEBUG 05-06 10:01:03.075062.075062 lmp.py:1510] ---- decode step 26 layer 4 ----
DEBUG 05-06 10:01:03.080050.080050 cuda_h.py:27] end decode_layer cost 5.567 ms
DEBUG 05-06 10:01:03.080609.080609 lmp.py:1510] ---- decode step 26 layer 5 ----
DEBUG 05-06 10:01:03.086410.086410 cuda_h.py:27] end decode_layer cost 5.921 ms
DEBUG 05-06 10:01:03.086445.086445 lmp.py:1510] ---- decode step 26 layer 6 ----
DEBUG 05-06 10:01:03.092820.092820 cuda_h.py:27] end decode_layer cost 5.641 ms
DEBUG 05-06 10:01:03.092193.092193 lmp.py:1510] ---- decode step 26 layer 7 ----
DEBUG 05-06 10:01:03.098634.098634 cuda_h.py:27] end decode_layer cost 5.655 ms
DEBUG 05-06 10:01:03.098907.098907 lmp.py:1510] ---- decode step 26 layer 8 ----
DEBUG 05-06 10:01:03.103207.103207 cuda_h.py:27] end decode_layer cost 5.587 ms
DEBUG 05-06 10:01:03.103766.103766 lmp.py:1510] ---- decode step 26 layer 9 ----
DEBUG 05-06 10:01:03.109914.109914 cuda_h.py:27] end decode_layer cost 5.615 ms
DEBUG 05-06 10:01:03.109188.109188 lmp.py:1510] ---- decode step 26 layer 10 ----
DEBUG 05-06 10:01:03.115977.115977 cuda_h.py:27] end decode_layer cost 5.561 ms
DEBUG 05-06 10:01:03.115773.115773 lmp.py:1510] ---- decode step 26 layer 11 ----
DEBUG 05-06 10:01:03.121920.121920 cuda_h.py:27] end decode_layer cost 5.965 ms
DEBUG 05-06 10:01:03.121479.121479 lmp.py:1510] ---- decode step 26 layer 12 ----
DEBUG 05-06 10:01:03.126164.126164 cuda_h.py:27] end decode_layer cost 5.625 ms
DEBUG 05-06 10:01:03.127438.127438 lmp.py:1510] ---- decode step 26 layer 13 ----
DEBUG 05-06 10:01:03.132191.132191 cuda_h.py:27] end decode_layer cost 5.886 ms
DEBUG 05-06 10:01:03.132849.132849 lmp.py:1510] ---- decode step 26 layer 14 ----
DEBUG 05-06 10:01:03.138756.138756 cuda_h.py:27] end decode_layer cost 5.929 ms
DEBUG 05-06 10:01:03.138845.138845 lmp.py:1510] ---- decode step 26 layer 15 ----
DEBUG 05-06 10:01:03.146176.146176 cuda_h.py:27] end decode_layer cost 7.327 ms
DEBUG 05-06 10:01:03.146968.146968 lmp.py:1510] ---- decode step 26 layer 16 ----
DEBUG 05-06 10:01:03.152165.152165 cuda_h.py:27] end decode_layer cost 5.898 ms
DEBUG 05-06 10:01:03.152823.152823 lmp.py:1510] ---- decode step 26 layer 17 ----
DEBUG 05-06 10:01:03.158302.158302 cuda_h.py:27] end decode_layer cost 6.000 ms
DEBUG 05-06 10:01:03.158053.158053 lmp.py:1510] ---- decode step 26 layer 18 ----
DEBUG 05-06 10:01:03.164659.164659 cuda_h.py:27] end decode_layer cost 5.637 ms
DEBUG 05-06 10:01:03.164456.164456 lmp.py:1510] ---- decode step 26 layer 19 ----
DEBUG 05-06 10:01:03.169644.169644 cuda_h.py:27] end decode_layer cost 5.609 ms
DEBUG 05-06 10:01:03.169679.169679 lmp.py:1510] ---- decode step 26 layer 20 ----
DEBUG 05-06 10:01:03.175746.175746 cuda_h.py:27] end decode_layer cost 5.555 ms
DEBUG 05-06 10:01:03.175735.175735 lmp.py:1510] ---- decode step 26 layer 21 ----
DEBUG 05-06 10:01:03.181624.181624 cuda_h.py:27] end decode_layer cost 5.775 ms
DEBUG 05-06 10:01:03.181659.181659 lmp.py:1510] ---- decode step 26 layer 22 ----
DEBUG 05-06 10:01:03.186860.186860 cuda_h.py:27] end decode_layer cost 5.619 ms
DEBUG 05-06 10:01:03.186134.186134 lmp.py:1510] ---- decode step 26 layer 23 ----
DEBUG 05-06 10:01:03.193469.193469 cuda_h.py:27] end decode_layer cost 6.033 ms
DEBUG 05-06 10:01:03.193742.193742 lmp.py:1510] ---- decode step 26 layer 24 ----
DEBUG 05-06 10:01:03.198926.198926 cuda_h.py:27] end decode_layer cost 5.676 ms
DEBUG 05-06 10:01:03.198584.198584 lmp.py:1510] ---- decode step 26 layer 25 ----
DEBUG 05-06 10:01:03.204468.204468 cuda_h.py:27] end decode_layer cost 5.842 ms
DEBUG 05-06 10:01:03.204649.204649 lmp.py:1510] ---- decode step 26 layer 26 ----
DEBUG 05-06 10:01:03.210114.210114 cuda_h.py:27] end decode_layer cost 5.568 ms
DEBUG 05-06 10:01:03.210388.210388 lmp.py:1510] ---- decode step 26 layer 27 ----
DEBUG 05-06 10:01:03.216081.216081 cuda_h.py:27] end decode_layer cost 5.666 ms
DEBUG 05-06 10:01:03.216262.216262 lmp.py:1510] ---- decode step 26 layer 28 ----
DEBUG 05-06 10:01:03.221899.221899 cuda_h.py:27] end decode_layer cost 5.590 ms
DEBUG 05-06 10:01:03.221935.221935 lmp.py:1510] ---- decode step 26 layer 29 ----
DEBUG 05-06 10:01:03.227189.227189 cuda_h.py:27] end decode_layer cost 6.009 ms
DEBUG 05-06 10:01:03.227272.227272 cuda_h.py:27] end decode_step cost 183.627 ms
INFO 05-06 10:01:03.227657.227657 lmp.py:1558] decode step 26 time: 0.1836683750152588 seconds
WARNING 05-06 10:01:03.228795.228795 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:03.228124.228124 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:03.228336.228336 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:03.228830.228830 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:03.234505.234505 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:03.234443.234443 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:03.234882.234882 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:03.236795.236795 cuda_h.py:27] end init_inputs_tokens cost 8.217 ms
DEBUG 05-06 10:01:03.236162.236162 lmp.py:1504] decode step 27 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:03.236501.236501 lmp.py:1510] ---- decode step 27 layer 0 ----
DEBUG 05-06 10:01:03.241704.241704 cuda_h.py:27] end decode_layer cost 5.655 ms
DEBUG 05-06 10:01:03.242216.242216 lmp.py:1510] ---- decode step 27 layer 1 ----
DEBUG 05-06 10:01:03.247706.247706 cuda_h.py:27] end decode_layer cost 5.727 ms
DEBUG 05-06 10:01:03.247502.247502 lmp.py:1510] ---- decode step 27 layer 2 ----
DEBUG 05-06 10:01:03.253370.253370 cuda_h.py:27] end decode_layer cost 5.724 ms
DEBUG 05-06 10:01:03.253789.253789 lmp.py:1510] ---- decode step 27 layer 3 ----
DEBUG 05-06 10:01:03.259537.259537 cuda_h.py:27] end decode_layer cost 5.917 ms
DEBUG 05-06 10:01:03.259003.259003 lmp.py:1510] ---- decode step 27 layer 4 ----
DEBUG 05-06 10:01:03.265204.265204 cuda_h.py:27] end decode_layer cost 5.794 ms
DEBUG 05-06 10:01:03.265384.265384 lmp.py:1510] ---- decode step 27 layer 5 ----
DEBUG 05-06 10:01:03.271512.271512 cuda_h.py:27] end decode_layer cost 6.161 ms
DEBUG 05-06 10:01:03.271408.271408 lmp.py:1510] ---- decode step 27 layer 6 ----
DEBUG 05-06 10:01:03.277887.277887 cuda_h.py:27] end decode_layer cost 5.789 ms
DEBUG 05-06 10:01:03.277067.277067 lmp.py:1510] ---- decode step 27 layer 7 ----
DEBUG 05-06 10:01:03.283918.283918 cuda_h.py:27] end decode_layer cost 5.817 ms
DEBUG 05-06 10:01:03.283384.283384 lmp.py:1510] ---- decode step 27 layer 8 ----
DEBUG 05-06 10:01:03.289685.289685 cuda_h.py:27] end decode_layer cost 5.833 ms
DEBUG 05-06 10:01:03.289151.289151 lmp.py:1510] ---- decode step 27 layer 9 ----
DEBUG 05-06 10:01:03.295947.295947 cuda_h.py:27] end decode_layer cost 5.778 ms
DEBUG 05-06 10:01:03.295936.295936 lmp.py:1510] ---- decode step 27 layer 10 ----
DEBUG 05-06 10:01:03.301363.301363 cuda_h.py:27] end decode_layer cost 5.820 ms
DEBUG 05-06 10:01:03.301352.301352 lmp.py:1510] ---- decode step 27 layer 11 ----
DEBUG 05-06 10:01:03.307696.307696 cuda_h.py:27] end decode_layer cost 6.110 ms
DEBUG 05-06 10:01:03.307446.307446 lmp.py:1510] ---- decode step 27 layer 12 ----
DEBUG 05-06 10:01:03.313062.313062 cuda_h.py:27] end decode_layer cost 5.924 ms
DEBUG 05-06 10:01:03.313157.313157 lmp.py:1510] ---- decode step 27 layer 13 ----
DEBUG 05-06 10:01:03.319544.319544 cuda_h.py:27] end decode_layer cost 6.002 ms
DEBUG 05-06 10:01:03.319963.319963 lmp.py:1510] ---- decode step 27 layer 14 ----
DEBUG 05-06 10:01:03.325753.325753 cuda_h.py:27] end decode_layer cost 6.339 ms
DEBUG 05-06 10:01:03.325617.325617 lmp.py:1510] ---- decode step 27 layer 15 ----
DEBUG 05-06 10:01:03.331573.331573 cuda_h.py:27] end decode_layer cost 6.000 ms
DEBUG 05-06 10:01:03.331946.331946 lmp.py:1510] ---- decode step 27 layer 16 ----
DEBUG 05-06 10:01:03.337941.337941 cuda_h.py:27] end decode_layer cost 5.958 ms
DEBUG 05-06 10:01:03.337029.337029 lmp.py:1510] ---- decode step 27 layer 17 ----
DEBUG 05-06 10:01:03.344071.344071 cuda_h.py:27] end decode_layer cost 6.379 ms
DEBUG 05-06 10:01:03.344444.344444 lmp.py:1510] ---- decode step 27 layer 18 ----
DEBUG 05-06 10:01:03.350383.350383 cuda_h.py:27] end decode_layer cost 6.093 ms
DEBUG 05-06 10:01:03.350518.350518 lmp.py:1510] ---- decode step 27 layer 19 ----
DEBUG 05-06 10:01:03.356784.356784 cuda_h.py:27] end decode_layer cost 5.983 ms
DEBUG 05-06 10:01:03.356873.356873 lmp.py:1510] ---- decode step 27 layer 20 ----
DEBUG 05-06 10:01:03.362038.362038 cuda_h.py:27] end decode_layer cost 5.908 ms
DEBUG 05-06 10:01:03.362225.362225 lmp.py:1510] ---- decode step 27 layer 21 ----
DEBUG 05-06 10:01:03.368639.368639 cuda_h.py:27] end decode_layer cost 6.021 ms
DEBUG 05-06 10:01:03.368774.368774 lmp.py:1510] ---- decode step 27 layer 22 ----
DEBUG 05-06 10:01:03.374866.374866 cuda_h.py:27] end decode_layer cost 5.925 ms
DEBUG 05-06 10:01:03.374571.374571 lmp.py:1510] ---- decode step 27 layer 23 ----
DEBUG 05-06 10:01:03.380663.380663 cuda_h.py:27] end decode_layer cost 6.311 ms
DEBUG 05-06 10:01:03.381082.381082 lmp.py:1510] ---- decode step 27 layer 24 ----
DEBUG 05-06 10:01:03.386151.386151 cuda_h.py:27] end decode_layer cost 5.802 ms
DEBUG 05-06 10:01:03.386140.386140 lmp.py:1510] ---- decode step 27 layer 25 ----
DEBUG 05-06 10:01:03.392287.392287 cuda_h.py:27] end decode_layer cost 5.955 ms
DEBUG 05-06 10:01:03.392322.392322 lmp.py:1510] ---- decode step 27 layer 26 ----
DEBUG 05-06 10:01:03.398032.398032 cuda_h.py:27] end decode_layer cost 5.783 ms
DEBUG 05-06 10:01:03.398644.398644 lmp.py:1510] ---- decode step 27 layer 27 ----
DEBUG 05-06 10:01:03.404420.404420 cuda_h.py:27] end decode_layer cost 5.973 ms
DEBUG 05-06 10:01:03.404575.404575 lmp.py:1510] ---- decode step 27 layer 28 ----
DEBUG 05-06 10:01:03.410360.410360 cuda_h.py:27] end decode_layer cost 5.849 ms
DEBUG 05-06 10:01:03.410064.410064 lmp.py:1510] ---- decode step 27 layer 29 ----
DEBUG 05-06 10:01:03.416509.416509 cuda_h.py:27] end decode_layer cost 6.150 ms
DEBUG 05-06 10:01:03.417254.417254 cuda_h.py:27] end decode_step cost 189.038 ms
INFO 05-06 10:01:03.417878.417878 lmp.py:1558] decode step 27 time: 0.18907594680786133 seconds
WARNING 05-06 10:01:03.417691.417691 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:03.417384.417384 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:03.417357.417357 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:03.418043.418043 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:03.423228.423228 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:03.423935.423935 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:03.423519.423519 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:03.425606.425606 cuda_h.py:27] end init_inputs_tokens cost 8.231 ms
DEBUG 05-06 10:01:03.425025.425025 lmp.py:1504] decode step 28 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:03.425172.425172 lmp.py:1510] ---- decode step 28 layer 0 ----
DEBUG 05-06 10:01:03.431933.431933 cuda_h.py:27] end decode_layer cost 5.715 ms
DEBUG 05-06 10:01:03.431968.431968 lmp.py:1510] ---- decode step 28 layer 1 ----
DEBUG 05-06 10:01:03.437740.437740 cuda_h.py:27] end decode_layer cost 5.829 ms
DEBUG 05-06 10:01:03.437252.437252 lmp.py:1510] ---- decode step 28 layer 2 ----
DEBUG 05-06 10:01:03.442295.442295 cuda_h.py:27] end decode_layer cost 5.643 ms
DEBUG 05-06 10:01:03.442046.442046 lmp.py:1510] ---- decode step 28 layer 3 ----
DEBUG 05-06 10:01:03.448418.448418 cuda_h.py:27] end decode_layer cost 5.780 ms
DEBUG 05-06 10:01:03.448646.448646 lmp.py:1510] ---- decode step 28 layer 4 ----
DEBUG 05-06 10:01:03.454281.454281 cuda_h.py:27] end decode_layer cost 5.903 ms
DEBUG 05-06 10:01:03.454177.454177 lmp.py:1510] ---- decode step 28 layer 5 ----
DEBUG 05-06 10:01:03.460528.460528 cuda_h.py:27] end decode_layer cost 6.327 ms
DEBUG 05-06 10:01:03.461186.461186 lmp.py:1510] ---- decode step 28 layer 6 ----
DEBUG 05-06 10:01:03.466992.466992 cuda_h.py:27] end decode_layer cost 5.890 ms
DEBUG 05-06 10:01:03.467412.467412 lmp.py:1510] ---- decode step 28 layer 7 ----
DEBUG 05-06 10:01:03.472045.472045 cuda_h.py:27] end decode_layer cost 5.868 ms
DEBUG 05-06 10:01:03.472372.472372 lmp.py:1510] ---- decode step 28 layer 8 ----
DEBUG 05-06 10:01:03.478900.478900 cuda_h.py:27] end decode_layer cost 5.860 ms
DEBUG 05-06 10:01:03.478365.478365 lmp.py:1510] ---- decode step 28 layer 9 ----
DEBUG 05-06 10:01:03.484036.484036 cuda_h.py:27] end decode_layer cost 5.965 ms
DEBUG 05-06 10:01:03.484170.484170 lmp.py:1510] ---- decode step 28 layer 10 ----
DEBUG 05-06 10:01:03.490190.490190 cuda_h.py:27] end decode_layer cost 5.731 ms
DEBUG 05-06 10:01:03.490464.490464 lmp.py:1510] ---- decode step 28 layer 11 ----
DEBUG 05-06 10:01:03.497037.497037 cuda_h.py:27] end decode_layer cost 6.835 ms
DEBUG 05-06 10:01:03.497384.497384 lmp.py:1510] ---- decode step 28 layer 12 ----
DEBUG 05-06 10:01:03.503640.503640 cuda_h.py:27] end decode_layer cost 5.661 ms
DEBUG 05-06 10:01:03.503676.503676 lmp.py:1510] ---- decode step 28 layer 13 ----
DEBUG 05-06 10:01:03.509035.509035 cuda_h.py:27] end decode_layer cost 5.770 ms
DEBUG 05-06 10:01:03.509547.509547 lmp.py:1510] ---- decode step 28 layer 14 ----
DEBUG 05-06 10:01:03.514571.514571 cuda_h.py:27] end decode_layer cost 5.665 ms
DEBUG 05-06 10:01:03.514891.514891 lmp.py:1510] ---- decode step 28 layer 15 ----
DEBUG 05-06 10:01:03.520974.520974 cuda_h.py:27] end decode_layer cost 5.637 ms
DEBUG 05-06 10:01:03.520294.520294 lmp.py:1510] ---- decode step 28 layer 16 ----
DEBUG 05-06 10:01:03.526953.526953 cuda_h.py:27] end decode_layer cost 5.641 ms
DEBUG 05-06 10:01:03.526511.526511 lmp.py:1510] ---- decode step 28 layer 17 ----
DEBUG 05-06 10:01:03.532558.532558 cuda_h.py:27] end decode_layer cost 5.926 ms
DEBUG 05-06 10:01:03.532546.532546 lmp.py:1510] ---- decode step 28 layer 18 ----
DEBUG 05-06 10:01:03.538240.538240 cuda_h.py:27] end decode_layer cost 5.666 ms
DEBUG 05-06 10:01:03.538752.538752 lmp.py:1510] ---- decode step 28 layer 19 ----
DEBUG 05-06 10:01:03.543897.543897 cuda_h.py:27] end decode_layer cost 5.718 ms
DEBUG 05-06 10:01:03.543217.543217 lmp.py:1510] ---- decode step 28 layer 20 ----
DEBUG 05-06 10:01:03.549982.549982 cuda_h.py:27] end decode_layer cost 5.648 ms
DEBUG 05-06 10:01:03.549594.549594 lmp.py:1510] ---- decode step 28 layer 21 ----
DEBUG 05-06 10:01:03.555737.555737 cuda_h.py:27] end decode_layer cost 5.858 ms
DEBUG 05-06 10:01:03.555488.555488 lmp.py:1510] ---- decode step 28 layer 22 ----
DEBUG 05-06 10:01:03.561731.561731 cuda_h.py:27] end decode_layer cost 5.685 ms
DEBUG 05-06 10:01:03.561528.561528 lmp.py:1510] ---- decode step 28 layer 23 ----
DEBUG 05-06 10:01:03.567758.567758 cuda_h.py:27] end decode_layer cost 5.887 ms
DEBUG 05-06 10:01:03.567039.567039 lmp.py:1510] ---- decode step 28 layer 24 ----
DEBUG 05-06 10:01:03.573038.573038 cuda_h.py:27] end decode_layer cost 5.891 ms
DEBUG 05-06 10:01:03.573172.573172 lmp.py:1510] ---- decode step 28 layer 25 ----
DEBUG 05-06 10:01:03.579068.579068 cuda_h.py:27] end decode_layer cost 5.991 ms
DEBUG 05-06 10:01:03.579918.579918 lmp.py:1510] ---- decode step 28 layer 26 ----
DEBUG 05-06 10:01:03.585808.585808 cuda_h.py:27] end decode_layer cost 5.812 ms
DEBUG 05-06 10:01:03.585320.585320 lmp.py:1510] ---- decode step 28 layer 27 ----
DEBUG 05-06 10:01:03.590022.590022 cuda_h.py:27] end decode_layer cost 5.707 ms
DEBUG 05-06 10:01:03.590057.590057 lmp.py:1510] ---- decode step 28 layer 28 ----
DEBUG 05-06 10:01:03.596798.596798 cuda_h.py:27] end decode_layer cost 5.701 ms
DEBUG 05-06 10:01:03.596979.596979 lmp.py:1510] ---- decode step 28 layer 29 ----
DEBUG 05-06 10:01:03.602833.602833 cuda_h.py:27] end decode_layer cost 5.925 ms
DEBUG 05-06 10:01:03.602631.602631 cuda_h.py:27] end decode_step cost 185.607 ms
INFO 05-06 10:01:03.602539.602539 lmp.py:1558] decode step 28 time: 0.18564748764038086 seconds
WARNING 05-06 10:01:03.602240.602240 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:03.603327.603327 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:03.603249.603249 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:03.603724.603724 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:03.609241.609241 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:03.609683.609683 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:03.609552.609552 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:03.611140.611140 cuda_h.py:27] end init_inputs_tokens cost 8.384 ms
DEBUG 05-06 10:01:03.611890.611890 lmp.py:1504] decode step 29 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:03.611753.611753 lmp.py:1510] ---- decode step 29 layer 0 ----
DEBUG 05-06 10:01:03.616960.616960 cuda_h.py:27] end decode_layer cost 5.588 ms
DEBUG 05-06 10:01:03.616233.616233 lmp.py:1510] ---- decode step 29 layer 1 ----
DEBUG 05-06 10:01:03.622604.622604 cuda_h.py:27] end decode_layer cost 5.709 ms
DEBUG 05-06 10:01:03.622354.622354 lmp.py:1510] ---- decode step 29 layer 2 ----
DEBUG 05-06 10:01:03.628614.628614 cuda_h.py:27] end decode_layer cost 5.592 ms
DEBUG 05-06 10:01:03.628934.628934 lmp.py:1510] ---- decode step 29 layer 3 ----
DEBUG 05-06 10:01:03.634309.634309 cuda_h.py:27] end decode_layer cost 5.642 ms
DEBUG 05-06 10:01:03.634582.634582 lmp.py:1510] ---- decode step 29 layer 4 ----
DEBUG 05-06 10:01:03.639137.639137 cuda_h.py:27] end decode_layer cost 5.668 ms
DEBUG 05-06 10:01:03.639556.639556 lmp.py:1510] ---- decode step 29 layer 5 ----
DEBUG 05-06 10:01:03.645693.645693 cuda_h.py:27] end decode_layer cost 6.063 ms
DEBUG 05-06 10:01:03.645920.645920 lmp.py:1510] ---- decode step 29 layer 6 ----
DEBUG 05-06 10:01:03.651258.651258 cuda_h.py:27] end decode_layer cost 5.719 ms
DEBUG 05-06 10:01:03.651485.651485 lmp.py:1510] ---- decode step 29 layer 7 ----
DEBUG 05-06 10:01:03.657061.657061 cuda_h.py:27] end decode_layer cost 5.720 ms
DEBUG 05-06 10:01:03.657527.657527 lmp.py:1510] ---- decode step 29 layer 8 ----
DEBUG 05-06 10:01:03.663445.663445 cuda_h.py:27] end decode_layer cost 5.656 ms
DEBUG 05-06 10:01:03.663241.663241 lmp.py:1510] ---- decode step 29 layer 9 ----
DEBUG 05-06 10:01:03.669211.669211 cuda_h.py:27] end decode_layer cost 6.221 ms
DEBUG 05-06 10:01:03.669439.669439 lmp.py:1510] ---- decode step 29 layer 10 ----
DEBUG 05-06 10:01:03.675491.675491 cuda_h.py:27] end decode_layer cost 5.720 ms
DEBUG 05-06 10:01:03.675050.675050 lmp.py:1510] ---- decode step 29 layer 11 ----
DEBUG 05-06 10:01:03.681085.681085 cuda_h.py:27] end decode_layer cost 5.988 ms
DEBUG 05-06 10:01:03.681405.681405 lmp.py:1510] ---- decode step 29 layer 12 ----
DEBUG 05-06 10:01:03.687248.687248 cuda_h.py:27] end decode_layer cost 5.601 ms
DEBUG 05-06 10:01:03.687045.687045 lmp.py:1510] ---- decode step 29 layer 13 ----
DEBUG 05-06 10:01:03.692608.692608 cuda_h.py:27] end decode_layer cost 5.746 ms
DEBUG 05-06 10:01:03.692120.692120 lmp.py:1510] ---- decode step 29 layer 14 ----
DEBUG 05-06 10:01:03.698929.698929 cuda_h.py:27] end decode_layer cost 5.751 ms
DEBUG 05-06 10:01:03.698156.698156 lmp.py:1510] ---- decode step 29 layer 15 ----
DEBUG 05-06 10:01:03.704523.704523 cuda_h.py:27] end decode_layer cost 5.811 ms
DEBUG 05-06 10:01:03.704658.704658 lmp.py:1510] ---- decode step 29 layer 16 ----
DEBUG 05-06 10:01:03.710910.710910 cuda_h.py:27] end decode_layer cost 5.938 ms
DEBUG 05-06 10:01:03.710806.710806 lmp.py:1510] ---- decode step 29 layer 17 ----
DEBUG 05-06 10:01:03.716833.716833 cuda_h.py:27] end decode_layer cost 6.122 ms
DEBUG 05-06 10:01:03.716252.716252 lmp.py:1510] ---- decode step 29 layer 18 ----
DEBUG 05-06 10:01:03.722823.722823 cuda_h.py:27] end decode_layer cost 5.787 ms
DEBUG 05-06 10:01:03.722574.722574 lmp.py:1510] ---- decode step 29 layer 19 ----
DEBUG 05-06 10:01:03.728186.728186 cuda_h.py:27] end decode_layer cost 5.816 ms
DEBUG 05-06 10:01:03.728797.728797 lmp.py:1510] ---- decode step 29 layer 20 ----
DEBUG 05-06 10:01:03.734585.734585 cuda_h.py:27] end decode_layer cost 5.701 ms
DEBUG 05-06 10:01:03.734573.734573 lmp.py:1510] ---- decode step 29 layer 21 ----
DEBUG 05-06 10:01:03.740974.740974 cuda_h.py:27] end decode_layer cost 5.836 ms
DEBUG 05-06 10:01:03.740023.740023 lmp.py:1510] ---- decode step 29 layer 22 ----
DEBUG 05-06 10:01:03.745895.745895 cuda_h.py:27] end decode_layer cost 5.657 ms
DEBUG 05-06 10:01:03.745884.745884 lmp.py:1510] ---- decode step 29 layer 23 ----
DEBUG 05-06 10:01:03.751096.751096 cuda_h.py:27] end decode_layer cost 5.943 ms
DEBUG 05-06 10:01:03.751184.751184 lmp.py:1510] ---- decode step 29 layer 24 ----
DEBUG 05-06 10:01:03.757930.757930 cuda_h.py:27] end decode_layer cost 5.635 ms
DEBUG 05-06 10:01:03.757965.757965 lmp.py:1510] ---- decode step 29 layer 25 ----
DEBUG 05-06 10:01:03.763313.763313 cuda_h.py:27] end decode_layer cost 5.657 ms
DEBUG 05-06 10:01:03.763064.763064 lmp.py:1510] ---- decode step 29 layer 26 ----
DEBUG 05-06 10:01:03.769351.769351 cuda_h.py:27] end decode_layer cost 5.613 ms
DEBUG 05-06 10:01:03.769386.769386 lmp.py:1510] ---- decode step 29 layer 27 ----
DEBUG 05-06 10:01:03.774562.774562 cuda_h.py:27] end decode_layer cost 5.636 ms
DEBUG 05-06 10:01:03.774597.774597 lmp.py:1510] ---- decode step 29 layer 28 ----
DEBUG 05-06 10:01:03.780313.780313 cuda_h.py:27] end decode_layer cost 5.753 ms
DEBUG 05-06 10:01:03.780223.780223 lmp.py:1510] ---- decode step 29 layer 29 ----
DEBUG 05-06 10:01:03.786333.786333 cuda_h.py:27] end decode_layer cost 6.043 ms
DEBUG 05-06 10:01:03.786045.786045 cuda_h.py:27] end decode_step cost 183.952 ms
INFO 05-06 10:01:03.786092.786092 lmp.py:1558] decode step 29 time: 0.1839900016784668 seconds
WARNING 05-06 10:01:03.786104.786104 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:03.787162.787162 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:03.787819.787819 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:03.787597.787597 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:03.793005.793005 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:03.793513.793513 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:03.793190.793190 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:03.794907.794907 cuda_h.py:27] end init_inputs_tokens cost 8.053 ms
DEBUG 05-06 10:01:03.794797.794797 lmp.py:1504] decode step 30 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:03.795374.795374 lmp.py:1510] ---- decode step 30 layer 0 ----
DEBUG 05-06 10:01:03.800515.800515 cuda_h.py:27] end decode_layer cost 5.785 ms
DEBUG 05-06 10:01:03.800219.800219 lmp.py:1510] ---- decode step 30 layer 1 ----
DEBUG 05-06 10:01:03.806696.806696 cuda_h.py:27] end decode_layer cost 5.717 ms
DEBUG 05-06 10:01:03.806446.806446 lmp.py:1510] ---- decode step 30 layer 2 ----
DEBUG 05-06 10:01:03.812984.812984 cuda_h.py:27] end decode_layer cost 5.587 ms
DEBUG 05-06 10:01:03.812543.812543 lmp.py:1510] ---- decode step 30 layer 3 ----
DEBUG 05-06 10:01:03.818005.818005 cuda_h.py:27] end decode_layer cost 5.707 ms
DEBUG 05-06 10:01:03.818186.818186 lmp.py:1510] ---- decode step 30 layer 4 ----
DEBUG 05-06 10:01:03.823560.823560 cuda_h.py:27] end decode_layer cost 5.605 ms
DEBUG 05-06 10:01:03.823217.823217 lmp.py:1510] ---- decode step 30 layer 5 ----
DEBUG 05-06 10:01:03.829090.829090 cuda_h.py:27] end decode_layer cost 6.079 ms
DEBUG 05-06 10:01:03.829702.829702 lmp.py:1510] ---- decode step 30 layer 6 ----
DEBUG 05-06 10:01:03.835864.835864 cuda_h.py:27] end decode_layer cost 5.626 ms
DEBUG 05-06 10:01:03.835185.835185 lmp.py:1510] ---- decode step 30 layer 7 ----
DEBUG 05-06 10:01:03.841626.841626 cuda_h.py:27] end decode_layer cost 5.690 ms
DEBUG 05-06 10:01:03.841376.841376 lmp.py:1510] ---- decode step 30 layer 8 ----
DEBUG 05-06 10:01:03.847393.847393 cuda_h.py:27] end decode_layer cost 5.835 ms
DEBUG 05-06 10:01:03.847382.847382 lmp.py:1510] ---- decode step 30 layer 9 ----
DEBUG 05-06 10:01:03.852540.852540 cuda_h.py:27] end decode_layer cost 5.693 ms
DEBUG 05-06 10:01:03.853575.853575 lmp.py:1510] ---- decode step 30 layer 10 ----
DEBUG 05-06 10:01:03.858523.858523 cuda_h.py:27] end decode_layer cost 5.573 ms
DEBUG 05-06 10:01:03.858035.858035 lmp.py:1510] ---- decode step 30 layer 11 ----
DEBUG 05-06 10:01:03.864982.864982 cuda_h.py:27] end decode_layer cost 5.923 ms
DEBUG 05-06 10:01:03.864832.864832 lmp.py:1510] ---- decode step 30 layer 12 ----
DEBUG 05-06 10:01:03.870218.870218 cuda_h.py:27] end decode_layer cost 5.580 ms
DEBUG 05-06 10:01:03.870730.870730 lmp.py:1510] ---- decode step 30 layer 13 ----
DEBUG 05-06 10:01:03.876169.876169 cuda_h.py:27] end decode_layer cost 5.795 ms
DEBUG 05-06 10:01:03.876827.876827 lmp.py:1510] ---- decode step 30 layer 14 ----
DEBUG 05-06 10:01:03.881683.881683 cuda_h.py:27] end decode_layer cost 5.575 ms
DEBUG 05-06 10:01:03.881433.881433 lmp.py:1510] ---- decode step 30 layer 15 ----
DEBUG 05-06 10:01:03.887821.887821 cuda_h.py:27] end decode_layer cost 5.652 ms
DEBUG 05-06 10:01:03.887333.887333 lmp.py:1510] ---- decode step 30 layer 16 ----
DEBUG 05-06 10:01:03.893829.893829 cuda_h.py:27] end decode_layer cost 5.696 ms
DEBUG 05-06 10:01:03.893387.893387 lmp.py:1510] ---- decode step 30 layer 17 ----
DEBUG 05-06 10:01:03.899397.899397 cuda_h.py:27] end decode_layer cost 6.040 ms
DEBUG 05-06 10:01:03.899148.899148 lmp.py:1510] ---- decode step 30 layer 18 ----
DEBUG 05-06 10:01:03.905454.905454 cuda_h.py:27] end decode_layer cost 5.767 ms
DEBUG 05-06 10:01:03.905250.905250 lmp.py:1510] ---- decode step 30 layer 19 ----
DEBUG 05-06 10:01:03.910051.910051 cuda_h.py:27] end decode_layer cost 5.710 ms
DEBUG 05-06 10:01:03.911325.911325 lmp.py:1510] ---- decode step 30 layer 20 ----
DEBUG 05-06 10:01:03.916102.916102 cuda_h.py:27] end decode_layer cost 5.788 ms
DEBUG 05-06 10:01:03.916614.916614 lmp.py:1510] ---- decode step 30 layer 21 ----
DEBUG 05-06 10:01:03.922494.922494 cuda_h.py:27] end decode_layer cost 5.698 ms
DEBUG 05-06 10:01:03.922529.922529 lmp.py:1510] ---- decode step 30 layer 22 ----
DEBUG 05-06 10:01:03.928684.928684 cuda_h.py:27] end decode_layer cost 5.620 ms
DEBUG 05-06 10:01:03.928865.928865 lmp.py:1510] ---- decode step 30 layer 23 ----
DEBUG 05-06 10:01:03.934540.934540 cuda_h.py:27] end decode_layer cost 5.898 ms
DEBUG 05-06 10:01:03.934052.934052 lmp.py:1510] ---- decode step 30 layer 24 ----
DEBUG 05-06 10:01:03.939419.939419 cuda_h.py:27] end decode_layer cost 5.636 ms
DEBUG 05-06 10:01:03.940031.940031 lmp.py:1510] ---- decode step 30 layer 25 ----
DEBUG 05-06 10:01:03.945386.945386 cuda_h.py:27] end decode_layer cost 5.663 ms
DEBUG 05-06 10:01:03.945137.945137 lmp.py:1510] ---- decode step 30 layer 26 ----
DEBUG 05-06 10:01:03.951180.951180 cuda_h.py:27] end decode_layer cost 5.643 ms
DEBUG 05-06 10:01:03.951507.951507 lmp.py:1510] ---- decode step 30 layer 27 ----
DEBUG 05-06 10:01:03.957104.957104 cuda_h.py:27] end decode_layer cost 5.772 ms
DEBUG 05-06 10:01:03.957139.957139 lmp.py:1510] ---- decode step 30 layer 28 ----
DEBUG 05-06 10:01:03.962690.962690 cuda_h.py:27] end decode_layer cost 5.561 ms
DEBUG 05-06 10:01:03.962202.962202 lmp.py:1510] ---- decode step 30 layer 29 ----
DEBUG 05-06 10:01:03.968539.968539 cuda_h.py:27] end decode_layer cost 5.895 ms
DEBUG 05-06 10:01:03.968422.968422 cuda_h.py:27] end decode_step cost 182.068 ms
INFO 05-06 10:01:03.968993.968993 lmp.py:1558] decode step 30 time: 0.18210482597351074 seconds
INFO 05-06 10:01:03.969075.969075 lmp.py:1564] average decode time from step 5: 0.1872941072170551 seconds
Time taken: 12.041474375873804 seconds
generate input ids cost 0.046903371810913086 s
DEBUG 05-06 10:01:06.680035.680035 cuda_h.py:27] end generate_input_ids cost 2589.720 ms
DEBUG 05-06 10:01:06.681499.681499 cuda_h.py:27] end init_cache cost 0.039 ms
INFO 05-06 10:01:06.681378.681378 lmp.py:1158] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 10:01:06.694752.694752 lmp.py:2782] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 4740325316, 'cuda:1': 12875595776, 'cuda:2': 12875595776, 'cuda:3': 12875595776} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7886239265315211, 'cuda:1': 0.4700220660037874, 'cuda:2': 0.4700220660037874, 'cuda:3': 0.4700220660037874}
INFO 05-06 10:01:06.694524.694524 lmp.py:2800] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.694254.694254 lmp.py:2800] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.694308.694308 lmp.py:2800] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.694740.694740 lmp.py:2800] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.694646.694646 lmp.py:2800] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.694793.694793 lmp.py:2800] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.695469.695469 lmp.py:2800] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.695947.695947 lmp.py:2800] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.695742.695742 lmp.py:2800] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.695935.695935 lmp.py:2800] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.695351.695351 lmp.py:2800] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.695498.695498 lmp.py:2800] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.695022.695022 lmp.py:2800] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.695764.695764 lmp.py:2800] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.695957.695957 lmp.py:2800] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696480.696480 lmp.py:2800] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696150.696150 lmp.py:2800] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696209.696209 lmp.py:2800] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696402.696402 lmp.py:2800] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696573.696573 lmp.py:2800] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696528.696528 lmp.py:2800] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696759.696759 lmp.py:2800] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696237.696237 lmp.py:2800] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696581.696581 lmp.py:2800] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.696297.696297 lmp.py:2800] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.697483.697483 lmp.py:2800] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.697961.697961 lmp.py:2800] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.697597.697597 lmp.py:2800] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.697313.697313 lmp.py:2800] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:01:06.697006.697006 lmp.py:2800] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 10:01:06.982384.982384 cuda_h.py:27] end init_loading_placement cost 301.138 ms
DEBUG 05-06 10:01:06.982058.982058 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:01:06.983497.983497 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:01:06 client.py:72] load_into_gpu: gemma4-26B-A4B, bfb2298a-d1f4-428e-911a-c4011365e9cf
INFO 05-06 10:01:07 client.py:135] Model loaded: gemma4-26B-A4B, bfb2298a-d1f4-428e-911a-c4011365e9cf
INFO 05-06 10:01:07 client.py:204] confirm_model_loaded: gemma4-26B-A4B, bfb2298a-d1f4-428e-911a-c4011365e9cf
INFO 05-06 10:01:07 client.py:212] Model loaded
DEBUG 05-06 10:01:07.512550.512550 cuda_h.py:27] end init_general_sagl_loading_async cost 529.126 ms
INFO 05-06 10:01:07.559260.559260 lmp.py:3303] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 10:01:07.560180.560180 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:01:07.560744.560744 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:01:07 client.py:72] load_into_gpu: gemma4-26B-A4B, 6e6e8245-a391-4bd2-9595-58627bf88ceb
INFO 05-06 10:01:07 client.py:135] Model loaded: gemma4-26B-A4B, 6e6e8245-a391-4bd2-9595-58627bf88ceb
DEBUG 05-06 10:01:07.688126.688126 cuda_h.py:27] end init_experts_loading_async cost 128.707 ms
DEBUG 05-06 10:01:07.803069.803069 cuda_h.py:27] end restore_state_dict cost 113.763 ms
INFO 05-06 10:01:07.807273.807273 lmp.py:1299] vLLM Triton pre-warmup done in 3.6 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 10:01:07.807718.807718 cuda_h.py:27] end init_inputs_tokens cost 0.531 ms
DEBUG 05-06 10:01:07.807635.807635 lmp.py:1346] -------------------------------- start prefill layer 0 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 101, 109, 85, 49, 45, 65, 93, 69, 5, 13, 9, 73, 77, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18], 'token_total': 420, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 13: 16, 9: 17, 73: 19, 77: 19, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 51, 55, 67, 75, 83, 91, 103, 115, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1216, 'token_per_expert': {3: 46, 7: 95, 31: 34, 39: 176, 47: 318, 51: 48, 55: 51, 67: 47, 75: 29, 83: 33, 91: 99, 103: 178, 115: 29, 127: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 32, 48, 52, 64, 68, 72, 92, 104, 112, 116, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 758, 'token_per_expert': {0: 73, 16: 48, 32: 43, 48: 41, 52: 43, 64: 27, 68: 170, 72: 35, 92: 16, 104: 43, 112: 23, 116: 18, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 25, 33, 37, 41, 53, 89, 105, 113, 117, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 809, 'token_per_expert': {1: 75, 21: 48, 25: 24, 33: 210, 37: 20, 41: 27, 53: 205, 89: 20, 105: 24, 113: 39, 117: 26, 121: 65, 125: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 46, 50, 54, 70, 74, 78, 90, 110, 118, 122, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 893, 'token_per_expert': {22: 64, 26: 59, 46: 119, 50: 110, 54: 59, 70: 25, 74: 61, 78: 36, 90: 154, 110: 27, 118: 29, 122: 35, 126: 115}}
INFO 05-06 10:01:07.813753.813753 lmp.py:1833] [layer_moe_fused] layer=0 prefix: 0.508ms alloc: 0.250ms
INFO 05-06 10:01:07.813884.813884 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 4.00543212890625e-05 seconds
INFO 05-06 10:01:07.816044.816044 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0021855831146240234s
INFO 05-06 10:01:07.816814.816814 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005688667297363281 seconds
INFO 05-06 10:01:07.818545.818545 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001786947250366211s
INFO 05-06 10:01:07.848030.848030 lmp.py:1938] [layer_moe_fused] vllm triton time: 29.410ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:07.848887.848887 cuda_h.py:27] end *layer_moe_fused cost 35.798 ms
DEBUG 05-06 10:01:07.849118.849118 cuda_h.py:27] end prefill_layer cost 41.709 ms
DEBUG 05-06 10:01:07.849279.849279 lmp.py:1388] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 10:01:07.849148.849148 lmp.py:1346] -------------------------------- start prefill layer 1 --------------------------------
experts_cpu_alloc {'expert_ids': [39, 43, 55, 31, 83, 103, 115, 123, 11, 87, 16, 44, 60, 72, 88, 108, 40, 84, 32, 112, 56, 48, 92, 76, 124, 61, 41, 117, 29, 33, 45, 121, 57, 89, 81, 125, 85, 21, 37, 93, 69, 14, 62, 110, 18, 38, 26, 66, 50, 74, 78, 90, 98, 34], 'token_total': 260, 'token_per_expert': {39: 1, 43: 2, 55: 2, 31: 3, 83: 3, 103: 3, 115: 3, 123: 3, 11: 5, 87: 5, 16: 1, 44: 2, 60: 2, 72: 2, 88: 2, 108: 2, 40: 3, 84: 3, 32: 4, 112: 6, 56: 7, 48: 8, 92: 8, 76: 10, 124: 12, 61: 1, 41: 2, 117: 2, 29: 3, 33: 3, 45: 3, 121: 3, 57: 4, 89: 4, 81: 5, 125: 5, 85: 6, 21: 7, 37: 7, 93: 7, 69: 8, 14: 1, 62: 1, 110: 1, 18: 3, 38: 3, 26: 4, 66: 4, 50: 10, 74: 11, 78: 11, 90: 11, 98: 12, 34: 16}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 35, 47, 51, 59, 67, 79, 91, 95, 99, 119, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 24, 'token_total': 784, 'token_per_expert': {3: 200, 7: 215, 27: 10, 35: 20, 47: 31, 51: 43, 59: 15, 67: 99, 79: 18, 91: 5, 95: 7, 99: 79, 119: 14, 127: 28}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 28, 52, 64, 68, 80, 96, 100, 104, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1054, 'token_per_expert': {0: 199, 4: 196, 8: 79, 12: 35, 20: 47, 28: 45, 52: 188, 64: 16, 68: 124, 80: 33, 96: 27, 100: 33, 104: 13, 120: 19}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 49, 53, 65, 73, 97, 101, 105, 109], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 926, 'token_per_expert': {1: 212, 5: 257, 9: 10, 13: 169, 25: 30, 49: 20, 53: 28, 65: 24, 73: 22, 97: 78, 101: 10, 105: 13, 109: 53}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 30, 42, 46, 54, 82, 94, 106, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 1072, 'token_per_expert': {2: 194, 6: 193, 10: 121, 22: 70, 30: 146, 42: 23, 46: 26, 54: 36, 82: 117, 94: 20, 106: 27, 118: 36, 122: 63}}
INFO 05-06 10:01:07.856812.856812 lmp.py:1833] [layer_moe_fused] layer=1 prefix: 0.546ms alloc: 0.415ms
INFO 05-06 10:01:07.856652.856652 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.9604644775390625e-05 seconds
INFO 05-06 10:01:07.857029.857029 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0010938644409179688s
INFO 05-06 10:01:07.858304.858304 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006229877471923828 seconds
INFO 05-06 10:01:07.866940.866940 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.007190704345703125s
INFO 05-06 10:01:07.879338.879338 lmp.py:1938] [layer_moe_fused] vllm triton time: 13.551ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:07.880781.880781 cuda_h.py:27] end *layer_moe_fused cost 24.956 ms
DEBUG 05-06 10:01:07.881885.881885 cuda_h.py:27] end prefill_layer cost 31.470 ms
DEBUG 05-06 10:01:07.881794.881794 lmp.py:1388] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 10:01:07.881944.881944 lmp.py:1346] -------------------------------- start prefill layer 2 --------------------------------
experts_cpu_alloc {'expert_ids': [67, 27, 111, 115, 99, 95, 63, 119, 43, 103, 71, 35, 123, 23, 40, 120, 96, 88, 116, 72, 36, 44, 8, 100, 64, 52, 56, 45, 61, 113, 21, 77, 121, 69, 105, 33, 85, 57, 17, 49, 26, 66, 114, 42, 50, 82, 70, 126, 58, 98, 46, 78], 'token_total': 427, 'token_per_expert': {67: 1, 27: 2, 111: 3, 115: 3, 99: 4, 95: 6, 63: 9, 119: 10, 43: 12, 103: 12, 71: 13, 35: 15, 123: 15, 23: 16, 40: 3, 120: 3, 96: 4, 88: 5, 116: 5, 72: 6, 36: 8, 44: 9, 8: 10, 100: 10, 64: 11, 52: 12, 56: 13, 45: 2, 61: 4, 113: 4, 21: 5, 77: 7, 121: 8, 69: 9, 105: 9, 33: 11, 85: 12, 57: 13, 17: 15, 49: 16, 26: 1, 66: 1, 114: 2, 42: 3, 50: 3, 82: 7, 70: 8, 126: 9, 58: 12, 98: 12, 46: 14, 78: 20}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 31, 51, 55, 59, 83, 91, 107, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 991, 'token_per_expert': {3: 206, 7: 226, 11: 115, 15: 69, 19: 69, 31: 19, 51: 21, 55: 43, 59: 77, 83: 22, 91: 25, 107: 16, 127: 83}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 28, 48, 60, 76, 80, 84, 104, 108, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 799, 'token_per_expert': {0: 201, 4: 200, 20: 22, 24: 14, 28: 14, 48: 30, 60: 30, 76: 47, 80: 26, 84: 48, 104: 31, 108: 120, 124: 16}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 29, 37, 41, 53, 65, 81, 97, 109, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 994, 'token_per_expert': {1: 271, 5: 194, 9: 71, 13: 58, 29: 56, 37: 49, 41: 102, 53: 22, 65: 20, 81: 41, 97: 28, 109: 31, 125: 51}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 34, 54, 62, 90, 102, 106, 110, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 885, 'token_per_expert': {2: 192, 6: 193, 14: 23, 18: 46, 34: 29, 54: 86, 62: 103, 90: 48, 102: 61, 106: 24, 110: 22, 118: 38, 122: 20}}
INFO 05-06 10:01:07.886817.886817 lmp.py:1833] [layer_moe_fused] layer=2 prefix: 0.491ms alloc: 0.392ms
INFO 05-06 10:01:07.887676.887676 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.459785461425781e-05 seconds
INFO 05-06 10:01:07.888201.888201 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008347034454345703s
INFO 05-06 10:01:07.888734.888734 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006067752838134766 seconds
INFO 05-06 10:01:07.899547.899547 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.010031938552856445s
INFO 05-06 10:01:07.904321.904321 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.580ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:07.904551.904551 cuda_h.py:27] end *layer_moe_fused cost 18.666 ms
DEBUG 05-06 10:01:07.911193.911193 cuda_h.py:27] end prefill_layer cost 29.774 ms
DEBUG 05-06 10:01:07.911401.911401 lmp.py:1388] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 10:01:07.911898.911898 lmp.py:1346] -------------------------------- start prefill layer 3 --------------------------------
experts_cpu_alloc {'expert_ids': [23, 35, 55, 91, 127, 27, 31, 63, 43, 67, 119, 59, 20, 72, 32, 36, 16, 60, 8, 100, 40, 48, 24, 116, 56, 64, 44, 29, 65, 13, 41, 57, 89, 117, 33, 101, 61, 77, 46, 94, 82, 18, 98, 30, 42, 86, 74, 26, 110, 114, 58, 70, 54], 'token_total': 352, 'token_per_expert': {23: 1, 35: 1, 55: 1, 91: 1, 127: 1, 27: 3, 31: 5, 63: 5, 43: 8, 67: 8, 119: 9, 59: 11, 20: 1, 72: 1, 32: 2, 36: 2, 16: 3, 60: 4, 8: 5, 100: 5, 40: 7, 48: 7, 24: 8, 116: 8, 56: 9, 64: 9, 44: 23, 29: 2, 65: 2, 13: 3, 41: 3, 57: 3, 89: 3, 117: 4, 33: 6, 101: 9, 61: 11, 77: 12, 46: 1, 94: 1, 82: 2, 18: 3, 98: 3, 30: 4, 42: 7, 86: 8, 74: 9, 26: 16, 110: 16, 114: 16, 58: 19, 70: 20, 54: 21}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 39, 51, 71, 75, 83, 95, 107, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 839, 'token_per_expert': {3: 272, 7: 256, 11: 28, 15: 24, 19: 15, 39: 18, 51: 23, 71: 42, 75: 56, 83: 34, 95: 31, 107: 12, 111: 12, 123: 16}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 52, 68, 76, 84, 88, 92, 96, 104, 108, 120], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 924, 'token_per_expert': {0: 275, 4: 280, 28: 56, 52: 32, 68: 32, 76: 27, 84: 48, 88: 24, 92: 36, 96: 28, 104: 30, 108: 24, 120: 32}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 25, 53, 69, 73, 85, 93, 97, 109, 121], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 913, 'token_per_expert': {1: 262, 5: 293, 9: 20, 17: 29, 25: 27, 53: 30, 69: 32, 73: 25, 85: 65, 93: 33, 97: 36, 109: 18, 121: 43}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 34, 50, 62, 66, 78, 102, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 1068, 'token_per_expert': {2: 267, 6: 265, 10: 31, 14: 45, 22: 65, 34: 37, 50: 87, 62: 57, 66: 55, 78: 43, 102: 62, 118: 28, 122: 26}}
INFO 05-06 10:01:07.916359.916359 lmp.py:1833] [layer_moe_fused] layer=3 prefix: 0.496ms alloc: 0.403ms
INFO 05-06 10:01:07.917125.917125 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.7220458984375e-05 seconds
INFO 05-06 10:01:07.918246.918246 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007758140563964844s
INFO 05-06 10:01:07.918202.918202 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.00060272216796875 seconds
INFO 05-06 10:01:07.931645.931645 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012488842010498047s
INFO 05-06 10:01:07.936027.936027 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.549ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:07.937278.937278 cuda_h.py:27] end *layer_moe_fused cost 21.030 ms
DEBUG 05-06 10:01:07.942946.942946 cuda_h.py:27] end prefill_layer cost 30.714 ms
DEBUG 05-06 10:01:07.942247.942247 lmp.py:1388] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 10:01:07.942175.942175 lmp.py:1346] -------------------------------- start prefill layer 4 --------------------------------
experts_cpu_alloc {'expert_ids': [31, 79, 103, 107, 75, 91, 15, 19, 123, 47, 39, 71, 87, 56, 80, 120, 36, 44, 40, 108, 64, 84, 88, 52, 116, 41, 121, 69, 21, 45, 97, 101, 109, 37, 77, 81, 73, 117, 17, 25, 46, 58, 66, 70, 18, 114, 122, 94, 30, 126, 38, 90, 34], 'token_total': 351, 'token_per_expert': {31: 1, 79: 1, 103: 2, 107: 8, 75: 11, 91: 11, 15: 12, 19: 12, 123: 13, 47: 15, 39: 18, 71: 18, 87: 20, 56: 4, 80: 4, 120: 4, 36: 6, 44: 6, 40: 7, 108: 7, 64: 8, 84: 10, 88: 11, 52: 12, 116: 13, 41: 1, 121: 1, 69: 2, 21: 3, 45: 4, 97: 4, 101: 4, 109: 4, 37: 5, 77: 5, 81: 5, 73: 7, 117: 7, 17: 9, 25: 9, 46: 1, 58: 1, 66: 1, 70: 1, 18: 2, 114: 2, 122: 3, 94: 4, 30: 5, 126: 5, 38: 7, 90: 7, 34: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 43, 51, 55, 59, 63, 67, 83, 111, 115, 119], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1231, 'token_per_expert': {3: 270, 7: 256, 23: 67, 27: 45, 43: 67, 51: 29, 55: 32, 59: 72, 63: 137, 67: 37, 83: 36, 111: 54, 115: 42, 119: 87}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 24, 28, 32, 60, 76, 92, 96, 104, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 855, 'token_per_expert': {0: 256, 4: 269, 8: 88, 20: 19, 24: 65, 28: 16, 32: 21, 60: 24, 76: 22, 92: 18, 96: 17, 104: 21, 124: 19}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 49, 53, 57, 61, 85, 89, 93, 105, 113, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 842, 'token_per_expert': {1: 303, 5: 269, 29: 33, 49: 25, 53: 21, 57: 9, 61: 10, 85: 27, 89: 50, 93: 28, 105: 15, 113: 26, 125: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 54, 62, 74, 78, 82, 86, 98, 106, 118], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 817, 'token_per_expert': {2: 256, 6: 259, 22: 38, 26: 32, 54: 33, 62: 14, 74: 60, 78: 12, 82: 25, 86: 12, 98: 14, 106: 53, 118: 9}}
INFO 05-06 10:01:07.947984.947984 lmp.py:1833] [layer_moe_fused] layer=4 prefix: 0.497ms alloc: 0.402ms
INFO 05-06 10:01:07.948141.948141 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.793571472167969e-05 seconds
INFO 05-06 10:01:07.949593.949593 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007698535919189453s
INFO 05-06 10:01:07.949596.949596 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006017684936523438 seconds
INFO 05-06 10:01:07.963526.963526 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013215065002441406s
INFO 05-06 10:01:07.968133.968133 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.576ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:07.968974.968974 cuda_h.py:27] end *layer_moe_fused cost 21.961 ms
DEBUG 05-06 10:01:07.974781.974781 cuda_h.py:27] end prefill_layer cost 31.934 ms
DEBUG 05-06 10:01:07.974605.974605 lmp.py:1388] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 10:01:07.974786.974786 lmp.py:1346] -------------------------------- start prefill layer 5 --------------------------------
experts_cpu_alloc {'expert_ids': [15, 19, 51, 27, 115, 67, 75, 107, 119, 31, 83, 79, 92, 8, 124, 56, 52, 68, 84, 44, 100, 96, 80, 120, 60, 104, 17, 21, 77, 105, 53, 57, 37, 113, 125, 30, 78, 82, 38, 50, 58, 26, 102, 54, 86, 114, 106, 62, 98, 34], 'token_total': 285, 'token_per_expert': {15: 1, 19: 2, 51: 2, 27: 3, 115: 3, 67: 6, 75: 7, 107: 7, 119: 7, 31: 8, 83: 8, 79: 10, 92: 1, 8: 2, 124: 2, 56: 4, 52: 5, 68: 6, 84: 7, 44: 9, 100: 13, 96: 15, 80: 16, 120: 16, 60: 20, 104: 20, 17: 1, 21: 1, 77: 1, 105: 2, 53: 3, 57: 3, 37: 4, 113: 5, 125: 14, 30: 1, 78: 1, 82: 1, 38: 2, 50: 2, 58: 2, 26: 3, 102: 3, 54: 4, 86: 4, 114: 4, 106: 5, 62: 6, 98: 6, 34: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 43, 55, 63, 71, 87, 99, 111, 123, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 964, 'token_per_expert': {3: 256, 7: 266, 23: 20, 39: 76, 43: 21, 55: 13, 63: 17, 71: 134, 87: 32, 99: 23, 111: 30, 123: 28, 127: 48}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 28, 36, 64, 72, 76, 88, 112, 116], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 1057, 'token_per_expert': {0: 267, 4: 295, 16: 67, 20: 74, 24: 41, 28: 43, 36: 42, 64: 38, 72: 32, 76: 27, 88: 27, 112: 80, 116: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 29, 33, 49, 61, 73, 93, 101, 117], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 21, 'token_total': 1029, 'token_per_expert': {1: 256, 5: 279, 9: 35, 13: 39, 29: 20, 33: 62, 49: 99, 61: 23, 73: 20, 93: 29, 101: 136, 117: 31}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 42, 46, 70, 74, 94, 118, 126], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 27, 'token_total': 761, 'token_per_expert': {2: 309, 6: 262, 14: 10, 18: 8, 22: 23, 42: 31, 46: 17, 70: 22, 74: 17, 94: 27, 118: 13, 126: 22}}
INFO 05-06 10:01:07.982839.982839 lmp.py:1833] [layer_moe_fused] layer=5 prefix: 0.481ms alloc: 0.377ms
INFO 05-06 10:01:07.983182.983182 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.5789947509765625e-05 seconds
INFO 05-06 10:01:07.984134.984134 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007481575012207031s
INFO 05-06 10:01:07.984401.984401 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005857944488525391 seconds
INFO 05-06 10:01:07.998985.998985 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013178348541259766s
INFO 05-06 10:01:08.003981.003981 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.725ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.003836.003836 cuda_h.py:27] end *layer_moe_fused cost 21.713 ms
DEBUG 05-06 10:01:08.009116.009116 cuda_h.py:27] end prefill_layer cost 34.749 ms
DEBUG 05-06 10:01:08.009701.009701 lmp.py:1388] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 10:01:08.009384.009384 lmp.py:1346] -------------------------------- start prefill layer 6 --------------------------------
experts_cpu_alloc {'expert_ids': [31, 47, 83, 19, 59, 67, 111, 15, 11, 91, 127, 43, 103, 51, 71, 8, 52, 72, 92, 112, 124, 16, 20, 40, 120, 76, 80, 60, 33, 97, 101, 109, 49, 81, 73, 89, 41, 37, 57, 125, 85, 22, 82, 114, 38, 18, 110, 30, 42, 74, 126, 14, 10, 58, 70, 78], 'token_total': 270, 'token_per_expert': {31: 1, 47: 1, 83: 1, 19: 2, 59: 2, 67: 2, 111: 2, 15: 4, 11: 5, 91: 6, 127: 6, 43: 10, 103: 11, 51: 12, 71: 12, 8: 1, 52: 1, 72: 1, 92: 1, 112: 1, 124: 1, 16: 2, 20: 3, 40: 3, 120: 3, 76: 4, 80: 4, 60: 5, 33: 1, 97: 1, 101: 1, 109: 1, 49: 2, 81: 2, 73: 5, 89: 5, 41: 7, 37: 8, 57: 8, 125: 8, 85: 9, 22: 1, 82: 1, 114: 1, 38: 3, 18: 5, 110: 5, 30: 6, 42: 7, 74: 7, 126: 7, 14: 8, 10: 10, 58: 11, 70: 16, 78: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 35, 75, 79, 87, 95, 99, 107, 115, 119, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 961, 'token_per_expert': {3: 256, 7: 257, 23: 49, 27: 14, 35: 42, 75: 14, 79: 21, 87: 45, 95: 15, 99: 131, 107: 17, 115: 49, 119: 36, 123: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 28, 32, 36, 44, 56, 64, 68, 96, 104, 108, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 913, 'token_per_expert': {0: 264, 4: 258, 24: 14, 28: 14, 32: 16, 36: 22, 44: 15, 56: 10, 64: 49, 68: 143, 96: 21, 104: 16, 108: 62, 116: 9}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 53, 65, 69, 77, 93, 105, 113, 117, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 984, 'token_per_expert': {1: 272, 5: 268, 9: 27, 13: 38, 25: 112, 53: 44, 65: 52, 69: 18, 77: 10, 93: 80, 105: 9, 113: 13, 117: 17, 121: 24}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 34, 46, 50, 62, 86, 90, 94, 98, 102, 106, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 968, 'token_per_expert': {2: 270, 6: 268, 26: 37, 34: 32, 46: 22, 50: 22, 62: 23, 86: 58, 90: 45, 94: 30, 98: 32, 102: 33, 106: 67, 122: 29}}
INFO 05-06 10:01:08.015472.015472 lmp.py:1833] [layer_moe_fused] layer=6 prefix: 0.483ms alloc: 0.412ms
INFO 05-06 10:01:08.015968.015968 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.151199340820312e-05 seconds
INFO 05-06 10:01:08.016998.016998 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007195472717285156s
INFO 05-06 10:01:08.017795.017795 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005908012390136719 seconds
INFO 05-06 10:01:08.030165.030165 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012487411499023438s
INFO 05-06 10:01:08.034576.034576 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.434ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.035761.035761 cuda_h.py:27] end *layer_moe_fused cost 20.809 ms
DEBUG 05-06 10:01:08.040064.040064 cuda_h.py:27] end prefill_layer cost 30.197 ms
DEBUG 05-06 10:01:08.040557.040557 lmp.py:1388] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 10:01:08.040752.040752 lmp.py:1346] -------------------------------- start prefill layer 7 --------------------------------
experts_cpu_alloc {'expert_ids': [27, 35, 67, 119, 15, 127, 55, 63, 107, 51, 95, 23, 87, 36, 100, 32, 88, 92, 116, 8, 68, 112, 16, 64, 80, 72, 20, 49, 73, 109, 37, 77, 41, 101, 17, 21, 45, 125, 117, 9, 25, 13, 30, 102, 50, 62, 66, 38, 94, 26, 54, 82, 78, 122, 98, 118, 126], 'token_total': 319, 'token_per_expert': {27: 1, 35: 1, 67: 1, 119: 1, 15: 2, 127: 3, 55: 4, 63: 4, 107: 4, 51: 6, 95: 7, 23: 8, 87: 8, 36: 1, 100: 1, 32: 2, 88: 4, 92: 5, 116: 5, 8: 6, 68: 8, 112: 9, 16: 10, 64: 12, 80: 14, 72: 18, 20: 21, 49: 1, 73: 1, 109: 2, 37: 3, 77: 4, 41: 5, 101: 5, 17: 6, 21: 6, 45: 6, 125: 7, 117: 8, 9: 9, 25: 10, 13: 14, 30: 1, 102: 1, 50: 2, 62: 2, 66: 2, 38: 3, 94: 3, 26: 4, 54: 4, 82: 6, 78: 7, 122: 7, 98: 8, 118: 8, 126: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 43, 47, 59, 71, 79, 83, 91, 99, 103, 111, 115, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 808, 'token_per_expert': {3: 256, 7: 272, 19: 14, 43: 14, 47: 14, 59: 12, 71: 21, 79: 32, 83: 11, 91: 98, 99: 12, 103: 21, 111: 9, 115: 9, 123: 13}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 28, 44, 48, 52, 56, 60, 84, 96, 104, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 981, 'token_per_expert': {0: 258, 4: 285, 12: 82, 28: 34, 44: 30, 48: 24, 52: 29, 56: 33, 60: 25, 84: 46, 96: 24, 104: 21, 108: 50, 120: 40}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 33, 53, 57, 61, 65, 69, 85, 97, 105, 113, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1055, 'token_per_expert': {1: 256, 5: 275, 29: 46, 33: 32, 53: 35, 57: 25, 61: 17, 65: 53, 69: 44, 85: 36, 97: 132, 105: 15, 113: 25, 121: 64}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 34, 42, 70, 86, 90, 106, 110, 114], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 933, 'token_per_expert': {2: 256, 6: 268, 10: 53, 14: 34, 18: 15, 22: 19, 34: 37, 42: 41, 70: 53, 86: 28, 90: 42, 106: 15, 110: 37, 114: 35}}
INFO 05-06 10:01:08.045782.045782 lmp.py:1833] [layer_moe_fused] layer=7 prefix: 0.487ms alloc: 0.416ms
INFO 05-06 10:01:08.046124.046124 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.151199340820312e-05 seconds
INFO 05-06 10:01:08.046977.046977 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007336139678955078s
INFO 05-06 10:01:08.047132.047132 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006084442138671875 seconds
INFO 05-06 10:01:08.060395.060395 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012308835983276367s
INFO 05-06 10:01:08.065324.065324 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.460ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.065052.065052 cuda_h.py:27] end *layer_moe_fused cost 20.662 ms
DEBUG 05-06 10:01:08.070423.070423 cuda_h.py:27] end prefill_layer cost 30.296 ms
DEBUG 05-06 10:01:08.070531.070531 lmp.py:1388] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 10:01:08.071871.071871 lmp.py:1346] -------------------------------- start prefill layer 8 --------------------------------
experts_cpu_alloc {'expert_ids': [23, 43, 35, 39, 91, 99, 119, 31, 47, 127, 55, 60, 72, 8, 48, 84, 104, 92, 96, 64, 116, 124, 68, 44, 25, 101, 13, 89, 85, 33, 37, 49, 117, 29, 17, 21, 57, 41, 113, 26, 90, 18, 82, 118, 86, 106, 34, 74, 22, 62, 66, 42, 98, 10, 14], 'token_total': 292, 'token_per_expert': {23: 2, 43: 2, 35: 3, 39: 3, 91: 3, 99: 5, 119: 6, 31: 7, 47: 9, 127: 10, 55: 11, 60: 1, 72: 1, 8: 2, 48: 2, 84: 2, 104: 3, 92: 4, 96: 4, 64: 5, 116: 5, 124: 5, 68: 6, 44: 8, 25: 2, 101: 2, 13: 3, 89: 3, 85: 4, 33: 5, 37: 6, 49: 6, 117: 6, 29: 7, 17: 8, 21: 8, 57: 8, 41: 9, 113: 11, 26: 1, 90: 1, 18: 2, 82: 2, 118: 2, 86: 3, 106: 3, 34: 4, 74: 4, 22: 6, 62: 6, 66: 8, 42: 11, 98: 12, 10: 15, 14: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 51, 63, 71, 75, 87, 103, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 1020, 'token_per_expert': {3: 280, 7: 256, 11: 13, 15: 22, 19: 42, 27: 19, 51: 95, 63: 12, 71: 39, 75: 45, 87: 63, 103: 95, 111: 15, 123: 24}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 20, 28, 32, 36, 52, 56, 76, 80, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 861, 'token_per_expert': {0: 257, 4: 268, 12: 23, 16: 10, 20: 9, 28: 67, 32: 39, 36: 18, 52: 17, 56: 38, 76: 16, 80: 26, 108: 8, 120: 65}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 45, 53, 61, 65, 69, 73, 77, 81, 93, 105, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 821, 'token_per_expert': {1: 257, 5: 274, 45: 12, 53: 22, 61: 14, 65: 25, 69: 17, 73: 54, 77: 17, 81: 16, 93: 12, 105: 40, 121: 30, 125: 31}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 50, 54, 58, 70, 102, 110, 114, 122, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 1102, 'token_per_expert': {2: 290, 6: 273, 38: 36, 46: 30, 50: 48, 54: 106, 58: 103, 70: 42, 102: 29, 110: 61, 114: 45, 122: 22, 126: 17}}
INFO 05-06 10:01:08.076932.076932 lmp.py:1833] [layer_moe_fused] layer=8 prefix: 0.531ms alloc: 0.402ms
INFO 05-06 10:01:08.076082.076082 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.841255187988281e-05 seconds
INFO 05-06 10:01:08.077578.077578 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007543563842773438s
INFO 05-06 10:01:08.078984.078984 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.000583648681640625 seconds
INFO 05-06 10:01:08.096938.096938 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.017258405685424805s
INFO 05-06 10:01:08.100800.100800 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.488ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.101728.101728 cuda_h.py:27] end *layer_moe_fused cost 25.715 ms
DEBUG 05-06 10:01:08.106923.106923 cuda_h.py:27] end prefill_layer cost 35.774 ms
DEBUG 05-06 10:01:08.106131.106131 lmp.py:1388] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 10:01:08.107201.107201 lmp.py:1346] -------------------------------- start prefill layer 9 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 31, 63, 79, 119, 67, 19, 27, 39, 71, 28, 64, 84, 112, 96, 52, 20, 44, 8, 116, 120, 24, 68, 124, 40, 29, 33, 41, 105, 97, 117, 77, 113, 73, 9, 37, 17, 45, 26, 58, 66, 10, 90, 122, 34, 98, 114, 82, 42, 62], 'token_total': 273, 'token_per_expert': {11: 1, 31: 1, 63: 1, 79: 1, 119: 3, 67: 6, 19: 8, 27: 11, 39: 12, 71: 13, 28: 1, 64: 1, 84: 1, 112: 1, 96: 2, 52: 3, 20: 4, 44: 4, 8: 6, 116: 6, 120: 8, 24: 9, 68: 10, 124: 10, 40: 11, 29: 1, 33: 1, 41: 1, 105: 4, 97: 6, 117: 6, 77: 7, 113: 7, 73: 9, 9: 10, 37: 10, 17: 13, 45: 14, 26: 1, 58: 1, 66: 1, 10: 2, 90: 3, 122: 3, 34: 4, 98: 4, 114: 5, 82: 7, 42: 8, 62: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 23, 43, 51, 75, 83, 95, 99, 103, 111, 127], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 23, 'token_total': 998, 'token_per_expert': {3: 267, 7: 264, 15: 19, 23: 33, 43: 56, 51: 17, 75: 51, 83: 17, 95: 143, 99: 24, 103: 65, 111: 19, 127: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 32, 36, 48, 56, 72, 76, 80, 88, 92], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 917, 'token_per_expert': {0: 262, 4: 283, 12: 78, 16: 69, 32: 26, 36: 24, 48: 39, 56: 51, 72: 20, 76: 17, 80: 11, 88: 11, 92: 26}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 57, 61, 69, 81, 89, 93, 101, 125], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 25, 'token_total': 889, 'token_per_expert': {1: 270, 5: 261, 13: 19, 21: 20, 57: 26, 61: 16, 69: 36, 81: 52, 89: 19, 93: 71, 101: 78, 125: 21}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 30, 38, 46, 54, 70, 74, 86, 102, 106], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 24, 'token_total': 1019, 'token_per_expert': {2: 257, 6: 257, 22: 16, 30: 18, 38: 19, 46: 103, 54: 22, 70: 103, 74: 64, 86: 14, 102: 39, 106: 107}}
INFO 05-06 10:01:08.112513.112513 lmp.py:1833] [layer_moe_fused] layer=9 prefix: 0.480ms alloc: 0.373ms
INFO 05-06 10:01:08.112617.112617 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.412101745605469e-05 seconds
INFO 05-06 10:01:08.113066.113066 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007145404815673828s
INFO 05-06 10:01:08.114181.114181 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005784034729003906 seconds
INFO 05-06 10:01:08.123684.123684 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.008732080459594727s
INFO 05-06 10:01:08.128747.128747 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.366ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.128967.128967 cuda_h.py:27] end *layer_moe_fused cost 16.990 ms
DEBUG 05-06 10:01:08.134799.134799 cuda_h.py:27] end prefill_layer cost 26.766 ms
DEBUG 05-06 10:01:08.134193.134193 lmp.py:1388] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 10:01:08.134229.134229 lmp.py:1346] -------------------------------- start prefill layer 10 --------------------------------
experts_cpu_alloc {'expert_ids': [51, 123, 35, 107, 111, 11, 59, 103, 15, 67, 83, 12, 32, 124, 40, 52, 56, 120, 64, 44, 28, 100, 112, 68, 53, 77, 101, 33, 37, 61, 109, 9, 25, 29, 97, 73, 69, 117, 89, 93, 38, 66, 70, 102, 26, 98, 34, 78, 50, 94, 10, 90, 46], 'token_total': 253, 'token_per_expert': {51: 1, 123: 1, 35: 2, 107: 2, 111: 2, 11: 3, 59: 3, 103: 3, 15: 4, 67: 4, 83: 6, 12: 1, 32: 1, 124: 1, 40: 2, 52: 2, 56: 3, 120: 3, 64: 5, 44: 6, 28: 9, 100: 11, 112: 11, 68: 13, 53: 1, 77: 1, 101: 1, 33: 2, 37: 2, 61: 2, 109: 3, 9: 4, 25: 4, 29: 4, 97: 4, 73: 7, 69: 8, 117: 8, 89: 10, 93: 10, 38: 1, 66: 1, 70: 1, 102: 1, 26: 3, 98: 3, 34: 6, 78: 6, 50: 8, 94: 9, 10: 10, 90: 16, 46: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 39, 43, 47, 63, 71, 75, 79, 99, 115, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 777, 'token_per_expert': {3: 259, 7: 270, 19: 11, 31: 18, 39: 14, 43: 9, 47: 25, 63: 17, 71: 31, 75: 27, 79: 8, 99: 12, 115: 50, 127: 26}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 60, 72, 76, 80, 84, 88, 92, 108], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 1145, 'token_per_expert': {0: 316, 4: 261, 8: 87, 16: 31, 20: 16, 60: 75, 72: 29, 76: 118, 80: 78, 84: 20, 88: 56, 92: 29, 108: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 41, 49, 57, 81, 85, 105, 113, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 952, 'token_per_expert': {1: 344, 5: 269, 13: 21, 21: 34, 41: 32, 49: 23, 57: 38, 81: 74, 85: 30, 105: 15, 113: 27, 121: 10, 125: 35}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 42, 54, 58, 62, 74, 82, 86, 106, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 969, 'token_per_expert': {2: 256, 6: 256, 14: 45, 18: 39, 42: 48, 54: 18, 58: 36, 62: 58, 74: 55, 82: 19, 86: 65, 106: 50, 126: 24}}
INFO 05-06 10:01:08.139415.139415 lmp.py:1833] [layer_moe_fused] layer=10 prefix: 0.475ms alloc: 0.396ms
INFO 05-06 10:01:08.139625.139625 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.841255187988281e-05 seconds
INFO 05-06 10:01:08.140976.140976 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007295608520507812s
INFO 05-06 10:01:08.141574.141574 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005848407745361328 seconds
INFO 05-06 10:01:08.161606.161606 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01911330223083496s
INFO 05-06 10:01:08.165846.165846 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.483ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.166186.166186 cuda_h.py:27] end *layer_moe_fused cost 27.461 ms
DEBUG 05-06 10:01:08.171057.171057 cuda_h.py:27] end prefill_layer cost 36.978 ms
DEBUG 05-06 10:01:08.171358.171358 lmp.py:1388] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 10:01:08.171320.171320 lmp.py:1346] -------------------------------- start prefill layer 11 --------------------------------
experts_cpu_alloc {'expert_ids': [35, 47, 127, 115, 59, 63, 51, 71, 91, 11, 43, 123, 12, 64, 52, 72, 80, 84, 28, 48, 8, 44, 120, 40, 36, 116, 124, 21, 53, 65, 9, 13, 33, 97, 125, 117, 121, 25, 29, 22, 74, 106, 114, 34, 58, 94, 110, 122, 126, 98, 50, 62, 70], 'token_total': 223, 'token_per_expert': {35: 1, 47: 1, 127: 1, 115: 2, 59: 3, 63: 3, 51: 6, 71: 6, 91: 7, 11: 9, 43: 9, 123: 9, 12: 2, 64: 2, 52: 3, 72: 3, 80: 3, 84: 3, 28: 4, 48: 4, 8: 5, 44: 5, 120: 5, 40: 6, 36: 7, 116: 10, 124: 11, 21: 1, 53: 1, 65: 1, 9: 2, 13: 2, 33: 2, 97: 2, 125: 3, 117: 6, 121: 11, 25: 14, 29: 14, 22: 1, 74: 1, 106: 1, 114: 1, 34: 2, 58: 2, 94: 2, 110: 2, 122: 2, 126: 2, 98: 3, 50: 5, 62: 5, 70: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 31, 39, 67, 79, 83, 87, 99, 111, 119], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 1046, 'token_per_expert': {3: 258, 7: 305, 19: 15, 23: 55, 27: 11, 31: 21, 39: 10, 67: 51, 79: 73, 83: 78, 87: 77, 99: 21, 111: 52, 119: 19}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 32, 56, 68, 76, 92, 100, 108, 112], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 1049, 'token_per_expert': {0: 259, 4: 258, 16: 82, 20: 29, 24: 30, 32: 38, 56: 103, 68: 48, 76: 29, 92: 85, 100: 24, 108: 30, 112: 34}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 37, 49, 57, 61, 69, 77, 81, 89, 93, 113], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 954, 'token_per_expert': {1: 265, 5: 265, 17: 44, 37: 19, 49: 52, 57: 21, 61: 14, 69: 23, 77: 29, 81: 74, 89: 18, 93: 56, 113: 74}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 30, 38, 42, 46, 54, 66, 82, 102, 118], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 824, 'token_per_expert': {2: 284, 6: 308, 10: 28, 18: 8, 30: 26, 38: 14, 42: 6, 46: 9, 54: 7, 66: 18, 82: 6, 102: 105, 118: 5}}
INFO 05-06 10:01:08.177976.177976 lmp.py:1833] [layer_moe_fused] layer=11 prefix: 0.476ms alloc: 0.396ms
INFO 05-06 10:01:08.177650.177650 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.079673767089844e-05 seconds
INFO 05-06 10:01:08.178031.178031 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007145404815673828s
INFO 05-06 10:01:08.179689.179689 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005877017974853516 seconds
INFO 05-06 10:01:08.191049.191049 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012093544006347656s
INFO 05-06 10:01:08.196217.196217 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.490ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.196377.196377 cuda_h.py:27] end *layer_moe_fused cost 20.376 ms
DEBUG 05-06 10:01:08.201331.201331 cuda_h.py:27] end prefill_layer cost 29.773 ms
DEBUG 05-06 10:01:08.201294.201294 lmp.py:1388] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 10:01:08.201326.201326 lmp.py:1346] -------------------------------- start prefill layer 12 --------------------------------
experts_cpu_alloc {'expert_ids': [59, 67, 31, 111, 107, 119, 127, 123, 63, 103, 8, 120, 20, 24, 32, 88, 12, 40, 104, 36, 80, 37, 41, 65, 81, 113, 33, 125, 13, 17, 105, 89, 77, 101, 18, 94, 102, 58, 70, 38, 90, 22, 98, 34, 106], 'token_total': 209, 'token_per_expert': {59: 1, 67: 1, 31: 2, 111: 2, 107: 3, 119: 3, 127: 3, 123: 5, 63: 6, 103: 7, 8: 1, 120: 1, 20: 3, 24: 3, 32: 3, 88: 6, 12: 7, 40: 7, 104: 7, 36: 9, 80: 9, 37: 1, 41: 1, 65: 1, 81: 1, 113: 1, 33: 2, 125: 2, 13: 3, 17: 3, 105: 3, 89: 4, 77: 6, 101: 9, 18: 1, 94: 2, 102: 3, 58: 4, 70: 4, 38: 5, 90: 5, 22: 6, 98: 7, 34: 12, 106: 34}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 35, 39, 71, 79, 91, 95, 115], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 22, 'token_total': 948, 'token_per_expert': {3: 274, 7: 256, 15: 76, 19: 32, 23: 28, 35: 17, 39: 87, 71: 90, 79: 8, 91: 31, 95: 21, 115: 28}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 68, 76, 84, 92, 100, 108, 112, 116, 124], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 22, 'token_total': 747, 'token_per_expert': {0: 256, 4: 257, 68: 13, 76: 11, 84: 12, 92: 18, 100: 11, 108: 77, 112: 12, 116: 67, 124: 13}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25, 45, 49, 53, 73, 85, 97, 117], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 24, 'token_total': 993, 'token_per_expert': {1: 263, 5: 291, 21: 104, 25: 31, 45: 61, 49: 27, 53: 112, 73: 34, 85: 18, 97: 34, 117: 18}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 46, 50, 74, 78, 82, 86, 110, 114, 118], 'expert_count': 11, 'ideal_gpu_count': 11, 'keep_on_gpu': 11, 'hit_count_on_device': 22, 'token_total': 1199, 'token_per_expert': {2: 257, 6: 293, 46: 44, 50: 74, 74: 82, 78: 162, 82: 47, 86: 72, 110: 56, 114: 61, 118: 51}}
INFO 05-06 10:01:08.207072.207072 lmp.py:1833] [layer_moe_fused] layer=12 prefix: 0.475ms alloc: 0.348ms
INFO 05-06 10:01:08.207878.207878 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.1975250244140625e-05 seconds
INFO 05-06 10:01:08.208367.208367 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014030933380126953s
INFO 05-06 10:01:08.209627.209627 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005805492401123047 seconds
INFO 05-06 10:01:08.223956.223956 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014097929000854492s
INFO 05-06 10:01:08.227203.227203 lmp.py:1938] [layer_moe_fused] vllm triton time: 3.904ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.228946.228946 cuda_h.py:27] end *layer_moe_fused cost 22.187 ms
DEBUG 05-06 10:01:08.233820.233820 cuda_h.py:27] end prefill_layer cost 31.957 ms
DEBUG 05-06 10:01:08.233067.233067 lmp.py:1388] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 10:01:08.234058.234058 lmp.py:1346] -------------------------------- start prefill layer 13 --------------------------------
experts_cpu_alloc {'expert_ids': [107, 83, 27, 43, 47, 87, 123, 11, 67, 75, 55, 115, 99, 12, 56, 8, 104, 40, 16, 64, 68, 96, 28, 97, 105, 45, 53, 65, 9, 57, 73, 13, 117, 93, 101, 10, 62, 66, 74, 94, 70, 106, 90, 26, 82, 122, 46, 42, 38, 86], 'token_total': 234, 'token_per_expert': {107: 1, 83: 2, 27: 3, 43: 3, 47: 3, 87: 3, 123: 3, 11: 7, 67: 7, 75: 7, 55: 9, 115: 9, 99: 10, 12: 1, 56: 1, 8: 3, 104: 3, 40: 5, 16: 6, 64: 6, 68: 6, 96: 6, 28: 7, 97: 1, 105: 1, 45: 2, 53: 2, 65: 3, 9: 4, 57: 4, 73: 4, 13: 6, 117: 7, 93: 9, 101: 9, 10: 1, 62: 1, 66: 1, 74: 1, 94: 1, 70: 2, 106: 3, 90: 4, 26: 5, 82: 5, 122: 5, 46: 7, 42: 9, 38: 10, 86: 16}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 31, 39, 51, 59, 63, 71, 79, 91, 103, 119], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 1038, 'token_per_expert': {3: 272, 7: 256, 15: 20, 31: 109, 39: 22, 51: 26, 59: 38, 63: 35, 71: 48, 79: 74, 91: 101, 103: 23, 119: 14}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 32, 52, 60, 84, 92, 100, 108, 116, 120, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 23, 'token_total': 846, 'token_per_expert': {0: 256, 4: 256, 20: 22, 32: 53, 52: 8, 60: 9, 84: 19, 92: 7, 100: 108, 108: 11, 116: 14, 120: 63, 124: 20}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 25, 33, 37, 41, 69, 81, 113, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 931, 'token_per_expert': {1: 280, 5: 256, 17: 49, 21: 27, 25: 39, 33: 25, 37: 49, 41: 14, 69: 23, 81: 51, 113: 27, 121: 68, 125: 23}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 34, 78, 98, 102, 110, 114, 118, 126], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 27, 'token_total': 1047, 'token_per_expert': {2: 264, 6: 301, 14: 49, 22: 31, 34: 22, 78: 50, 98: 36, 102: 36, 110: 94, 114: 100, 118: 32, 126: 32}}
INFO 05-06 10:01:08.238603.238603 lmp.py:1833] [layer_moe_fused] layer=13 prefix: 0.412ms alloc: 0.372ms
INFO 05-06 10:01:08.238892.238892 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.507469177246094e-05 seconds
INFO 05-06 10:01:08.239437.239437 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007121562957763672s
INFO 05-06 10:01:08.240377.240377 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005242824554443359 seconds
INFO 05-06 10:01:08.258587.258587 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01778125762939453s
INFO 05-06 10:01:08.262423.262423 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.310ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.263988.263988 cuda_h.py:27] end *layer_moe_fused cost 25.544 ms
DEBUG 05-06 10:01:08.268150.268150 cuda_h.py:27] end prefill_layer cost 34.786 ms
DEBUG 05-06 10:01:08.269060.269060 lmp.py:1388] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 10:01:08.269512.269512 lmp.py:1346] -------------------------------- start prefill layer 14 --------------------------------
experts_cpu_alloc {'expert_ids': [51, 63, 91, 111, 15, 19, 71, 23, 35, 43, 67, 107, 83, 56, 48, 68, 96, 40, 64, 108, 36, 44, 16, 116, 120, 60, 8, 24, 33, 37, 61, 73, 85, 77, 101, 17, 9, 81, 109, 125, 21, 93, 14, 46, 54, 70, 118, 106, 126, 10, 98, 102, 58, 78, 110, 114], 'token_total': 254, 'token_per_expert': {51: 1, 63: 1, 91: 1, 111: 1, 15: 2, 19: 2, 71: 3, 23: 5, 35: 5, 43: 5, 67: 5, 107: 6, 83: 12, 56: 2, 48: 3, 68: 3, 96: 3, 40: 5, 64: 5, 108: 5, 36: 6, 44: 6, 16: 7, 116: 7, 120: 7, 60: 8, 8: 9, 24: 9, 33: 1, 37: 1, 61: 1, 73: 1, 85: 1, 77: 2, 101: 2, 17: 3, 9: 4, 81: 5, 109: 7, 125: 8, 21: 10, 93: 11, 14: 1, 46: 1, 54: 1, 70: 3, 118: 3, 106: 4, 126: 4, 10: 5, 98: 5, 102: 6, 58: 7, 78: 7, 110: 8, 114: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 31, 39, 47, 59, 75, 95, 99, 103, 115, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 1130, 'token_per_expert': {3: 260, 7: 271, 11: 18, 31: 27, 39: 59, 47: 50, 59: 34, 75: 47, 95: 34, 99: 35, 103: 44, 115: 144, 119: 63, 123: 26, 127: 18}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 28, 32, 52, 72, 76, 80, 92, 100, 104, 112, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 812, 'token_per_expert': {0: 260, 4: 257, 12: 24, 28: 11, 32: 16, 52: 14, 72: 17, 76: 14, 80: 30, 92: 13, 100: 43, 104: 20, 112: 11, 124: 82}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 45, 53, 57, 65, 89, 97, 105, 113, 117, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 926, 'token_per_expert': {1: 257, 5: 264, 13: 21, 25: 13, 45: 12, 53: 22, 57: 15, 65: 56, 89: 17, 97: 52, 105: 15, 113: 38, 117: 41, 121: 103}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 30, 34, 38, 42, 50, 62, 66, 74, 86, 90, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 974, 'token_per_expert': {2: 290, 6: 257, 26: 68, 30: 28, 34: 14, 38: 14, 42: 27, 50: 60, 62: 33, 66: 54, 74: 17, 86: 72, 90: 9, 122: 31}}
INFO 05-06 10:01:08.273914.273914 lmp.py:1833] [layer_moe_fused] layer=14 prefix: 0.414ms alloc: 0.408ms
INFO 05-06 10:01:08.273019.273019 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.031990051269531e-05 seconds
INFO 05-06 10:01:08.274621.274621 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007576942443847656s
INFO 05-06 10:01:08.275098.275098 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005338191986083984 seconds
INFO 05-06 10:01:08.289084.289084 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014191627502441406s
INFO 05-06 10:01:08.294848.294848 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.317ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.294533.294533 cuda_h.py:27] end *layer_moe_fused cost 22.066 ms
DEBUG 05-06 10:01:08.300378.300378 cuda_h.py:27] end prefill_layer cost 30.792 ms
DEBUG 05-06 10:01:08.300195.300195 lmp.py:1388] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 10:01:08.300530.300530 lmp.py:1346] -------------------------------- start prefill layer 15 --------------------------------
experts_cpu_alloc {'expert_ids': [67, 79, 35, 11, 19, 55, 111, 127, 43, 115, 107, 119, 31, 59, 32, 56, 100, 8, 28, 80, 96, 36, 48, 116, 120, 16, 45, 105, 57, 25, 117, 33, 29, 77, 121, 13, 17, 41, 69, 97, 73, 94, 126, 22, 54, 118, 34, 82, 18, 38, 58, 78, 46], 'token_total': 329, 'token_per_expert': {67: 1, 79: 1, 35: 2, 11: 3, 19: 3, 55: 3, 111: 3, 127: 3, 43: 10, 115: 10, 107: 11, 119: 11, 31: 16, 59: 16, 32: 1, 56: 1, 100: 1, 8: 2, 28: 6, 80: 7, 96: 7, 36: 9, 48: 10, 116: 13, 120: 13, 16: 14, 45: 1, 105: 1, 57: 2, 25: 4, 117: 4, 33: 6, 29: 7, 77: 7, 121: 7, 13: 9, 17: 9, 41: 9, 69: 9, 97: 11, 73: 14, 94: 1, 126: 1, 22: 2, 54: 2, 118: 3, 34: 5, 82: 5, 18: 6, 38: 6, 58: 6, 78: 6, 46: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 47, 51, 63, 71, 75, 83, 91, 95, 99, 103], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 984, 'token_per_expert': {3: 257, 7: 277, 23: 35, 39: 32, 47: 19, 51: 30, 63: 18, 71: 37, 75: 48, 83: 72, 91: 93, 95: 18, 99: 29, 103: 19}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 52, 64, 68, 72, 76, 84, 88, 104, 108, 112, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 1039, 'token_per_expert': {0: 264, 4: 263, 24: 18, 52: 37, 64: 24, 68: 88, 72: 24, 76: 89, 84: 21, 88: 21, 104: 28, 108: 49, 112: 90, 124: 23}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 37, 65, 81, 85, 93, 101, 109, 113, 125], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 901, 'token_per_expert': {1: 269, 5: 279, 9: 34, 21: 25, 37: 24, 65: 62, 81: 26, 85: 22, 93: 22, 101: 31, 109: 74, 113: 15, 125: 18}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 30, 42, 66, 70, 86, 90, 98, 102, 114], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 843, 'token_per_expert': {2: 283, 6: 257, 10: 44, 14: 14, 30: 41, 42: 15, 66: 34, 70: 25, 86: 9, 90: 68, 98: 32, 102: 9, 114: 12}}
INFO 05-06 10:01:08.304565.304565 lmp.py:1833] [layer_moe_fused] layer=15 prefix: 0.408ms alloc: 0.392ms
INFO 05-06 10:01:08.305670.305670 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.817413330078125e-05 seconds
INFO 05-06 10:01:08.306709.306709 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008032321929931641s
INFO 05-06 10:01:08.306411.306411 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005245208740234375 seconds
INFO 05-06 10:01:08.320826.320826 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013302326202392578s
INFO 05-06 10:01:08.324615.324615 lmp.py:1938] [layer_moe_fused] vllm triton time: 3.883ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.324247.324247 cuda_h.py:27] end *layer_moe_fused cost 20.757 ms
DEBUG 05-06 10:01:08.330192.330192 cuda_h.py:27] end prefill_layer cost 29.834 ms
DEBUG 05-06 10:01:08.330155.330155 lmp.py:1388] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 10:01:08.330493.330493 lmp.py:1346] -------------------------------- start prefill layer 16 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 27, 103, 43, 51, 59, 71, 99, 111, 91, 123, 119, 28, 36, 84, 88, 64, 120, 24, 56, 104, 92, 40, 80, 116, 108, 72, 29, 41, 69, 73, 49, 53, 81, 89, 33, 109, 13, 61, 57, 113, 37, 46, 50, 106, 122, 38, 98, 34, 10, 82, 118, 18, 62, 90, 102, 30, 22], 'token_total': 277, 'token_per_expert': {11: 1, 27: 1, 103: 2, 43: 3, 51: 3, 59: 3, 71: 3, 99: 3, 111: 4, 91: 5, 123: 5, 119: 7, 28: 1, 36: 1, 84: 1, 88: 2, 64: 3, 120: 3, 24: 4, 56: 4, 104: 6, 92: 9, 40: 11, 80: 13, 116: 13, 108: 14, 72: 18, 29: 1, 41: 1, 69: 1, 73: 1, 49: 2, 53: 2, 81: 3, 89: 3, 33: 4, 109: 5, 13: 6, 61: 6, 57: 7, 113: 7, 37: 8, 46: 1, 50: 1, 106: 1, 122: 1, 38: 2, 98: 2, 34: 3, 10: 4, 82: 4, 118: 5, 18: 6, 62: 7, 90: 7, 102: 7, 30: 11, 22: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 31, 55, 63, 67, 75, 79, 83, 87, 107, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 27, 'token_total': 930, 'token_per_expert': {3: 270, 7: 260, 15: 8, 19: 17, 23: 19, 31: 33, 55: 22, 63: 25, 67: 62, 75: 20, 79: 11, 83: 31, 87: 98, 107: 39, 127: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 32, 44, 48, 52, 68, 76, 96, 100, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1135, 'token_per_expert': {0: 283, 4: 281, 8: 39, 12: 30, 16: 93, 20: 19, 32: 110, 44: 39, 48: 19, 52: 98, 68: 24, 76: 23, 96: 22, 100: 28, 124: 27}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 45, 65, 77, 85, 93, 97, 105, 117, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 818, 'token_per_expert': {1: 309, 5: 278, 9: 9, 17: 9, 21: 12, 45: 8, 65: 13, 77: 17, 85: 19, 93: 11, 97: 17, 105: 57, 117: 22, 121: 13, 125: 24}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 26, 42, 54, 58, 66, 70, 78, 86, 110, 114, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 936, 'token_per_expert': {2: 267, 6: 257, 14: 31, 26: 22, 42: 20, 54: 22, 58: 18, 66: 43, 70: 17, 78: 16, 86: 64, 110: 17, 114: 20, 126: 122}}
INFO 05-06 10:01:08.334594.334594 lmp.py:1833] [layer_moe_fused] layer=16 prefix: 0.407ms alloc: 0.415ms
INFO 05-06 10:01:08.335499.335499 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.0558319091796875e-05 seconds
INFO 05-06 10:01:08.336520.336520 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007700920104980469s
INFO 05-06 10:01:08.336891.336891 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005252361297607422 seconds
INFO 05-06 10:01:08.356056.356056 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.018883705139160156s
INFO 05-06 10:01:08.360488.360488 lmp.py:1938] [layer_moe_fused] vllm triton time: 3.884ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.360524.360524 cuda_h.py:27] end *layer_moe_fused cost 26.440 ms
DEBUG 05-06 10:01:08.366765.366765 cuda_h.py:27] end prefill_layer cost 35.796 ms
DEBUG 05-06 10:01:08.366727.366727 lmp.py:1388] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 10:01:08.366521.366521 lmp.py:1346] -------------------------------- start prefill layer 17 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 51, 79, 83, 111, 91, 87, 123, 59, 15, 19, 99, 119, 31, 55, 88, 92, 112, 36, 96, 8, 44, 80, 16, 60, 108, 124, 104, 116, 120, 100, 77, 81, 93, 117, 65, 85, 97, 9, 29, 113, 13, 33, 109, 125, 66, 102, 122, 38, 42, 14, 34, 62, 90, 114, 118, 126], 'token_total': 268, 'token_per_expert': {11: 1, 51: 1, 79: 1, 83: 1, 111: 1, 91: 2, 87: 3, 123: 3, 59: 4, 15: 5, 19: 6, 99: 7, 119: 7, 31: 9, 55: 9, 88: 1, 92: 2, 112: 2, 36: 4, 96: 4, 8: 5, 44: 6, 80: 7, 16: 8, 60: 8, 108: 8, 124: 8, 104: 9, 116: 9, 120: 14, 100: 16, 77: 1, 81: 1, 93: 1, 117: 1, 65: 2, 85: 2, 97: 4, 9: 5, 29: 5, 113: 5, 13: 6, 33: 6, 109: 6, 125: 8, 66: 2, 102: 2, 122: 2, 38: 3, 42: 3, 14: 4, 34: 4, 62: 4, 90: 4, 114: 4, 118: 4, 126: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 35, 39, 43, 47, 63, 67, 71, 75, 95, 103, 107], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1019, 'token_per_expert': {3: 268, 7: 256, 23: 73, 27: 52, 35: 20, 39: 52, 43: 25, 47: 23, 63: 24, 67: 13, 71: 24, 75: 60, 95: 72, 103: 19, 107: 38}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 28, 40, 48, 52, 56, 64, 68, 72, 76, 84], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 963, 'token_per_expert': {0: 264, 4: 266, 12: 20, 20: 20, 24: 72, 28: 27, 40: 36, 48: 22, 52: 38, 56: 22, 64: 34, 68: 24, 72: 33, 76: 67, 84: 18}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 37, 45, 49, 53, 57, 61, 69, 73, 89, 101], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 996, 'token_per_expert': {1: 258, 5: 277, 17: 33, 21: 57, 37: 81, 45: 12, 49: 30, 53: 28, 57: 21, 61: 31, 69: 83, 73: 12, 89: 41, 101: 32}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 54, 58, 70, 74, 78, 86, 94, 98, 106], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 850, 'token_per_expert': {2: 260, 6: 270, 10: 23, 18: 30, 22: 14, 54: 21, 58: 28, 70: 16, 74: 61, 78: 12, 86: 69, 94: 13, 98: 10, 106: 23}}
INFO 05-06 10:01:08.371517.371517 lmp.py:1833] [layer_moe_fused] layer=17 prefix: 0.414ms alloc: 0.419ms
INFO 05-06 10:01:08.371137.371137 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.008148193359375e-05 seconds
INFO 05-06 10:01:08.372836.372836 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008380413055419922s
INFO 05-06 10:01:08.372637.372637 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005252361297607422 seconds
INFO 05-06 10:01:08.383717.383717 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01057744026184082s
INFO 05-06 10:01:08.388735.388735 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.005ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.388116.388116 cuda_h.py:27] end *layer_moe_fused cost 18.341 ms
DEBUG 05-06 10:01:08.395173.395173 cuda_h.py:27] end prefill_layer cost 29.158 ms
DEBUG 05-06 10:01:08.395613.395613 lmp.py:1388] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 10:01:08.396590.396590 lmp.py:1346] -------------------------------- start prefill layer 18 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 19, 63, 23, 59, 115, 27, 67, 39, 51, 55, 103, 15, 107, 35, 95, 44, 112, 96, 24, 52, 16, 124, 116, 56, 48, 68, 80, 12, 108, 92, 25, 41, 113, 109, 9, 45, 29, 89, 97, 13, 21, 73, 125, 37, 69, 102, 18, 94, 98, 66, 74, 82, 114, 26, 30, 62, 42, 46], 'token_total': 349, 'token_per_expert': {11: 1, 19: 1, 63: 1, 23: 3, 59: 3, 115: 3, 27: 4, 67: 4, 39: 5, 51: 5, 55: 5, 103: 6, 15: 9, 107: 9, 35: 10, 95: 12, 44: 1, 112: 1, 96: 2, 24: 3, 52: 3, 16: 4, 124: 5, 116: 7, 56: 9, 48: 10, 68: 10, 80: 10, 12: 11, 108: 11, 92: 15, 25: 1, 41: 1, 113: 1, 109: 5, 9: 6, 45: 6, 29: 7, 89: 7, 97: 7, 13: 8, 21: 8, 73: 11, 125: 11, 37: 12, 69: 16, 102: 1, 18: 2, 94: 2, 98: 3, 66: 4, 74: 4, 82: 5, 114: 5, 26: 6, 30: 6, 62: 6, 42: 7, 46: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 43, 47, 71, 75, 83, 87, 91, 99, 111, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 944, 'token_per_expert': {3: 294, 7: 264, 31: 33, 43: 40, 47: 15, 71: 16, 75: 14, 83: 43, 87: 25, 91: 15, 99: 67, 111: 53, 119: 24, 123: 18, 127: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 32, 36, 40, 60, 64, 72, 76, 84, 88, 100, 104, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 915, 'token_per_expert': {0: 260, 4: 285, 8: 29, 32: 45, 36: 50, 40: 31, 60: 26, 64: 31, 72: 22, 76: 24, 84: 25, 88: 18, 100: 22, 104: 25, 120: 22}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 33, 49, 53, 57, 61, 65, 77, 81, 85, 93, 101, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 1012, 'token_per_expert': {1: 282, 5: 261, 17: 28, 33: 31, 49: 38, 53: 33, 57: 18, 61: 34, 65: 26, 77: 61, 81: 22, 85: 51, 93: 21, 101: 38, 121: 68}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 34, 38, 50, 54, 58, 70, 78, 90, 110, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 876, 'token_per_expert': {2: 298, 6: 256, 10: 14, 14: 38, 34: 16, 38: 18, 50: 43, 54: 55, 58: 35, 70: 9, 78: 23, 90: 8, 110: 24, 118: 30, 122: 9}}
INFO 05-06 10:01:08.400125.400125 lmp.py:1833] [layer_moe_fused] layer=18 prefix: 0.421ms alloc: 0.425ms
INFO 05-06 10:01:08.400851.400851 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.222724914550781e-05 seconds
INFO 05-06 10:01:08.401755.401755 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008053779602050781s
INFO 05-06 10:01:08.402047.402047 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005381107330322266 seconds
INFO 05-06 10:01:08.421287.421287 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.018393278121948242s
INFO 05-06 10:01:08.425988.425988 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.006ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.425389.425389 cuda_h.py:27] end *layer_moe_fused cost 26.148 ms
DEBUG 05-06 10:01:08.430632.430632 cuda_h.py:27] end prefill_layer cost 34.679 ms
DEBUG 05-06 10:01:08.430933.430933 lmp.py:1388] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 10:01:08.430318.430318 lmp.py:1346] -------------------------------- start prefill layer 19 --------------------------------
experts_cpu_alloc {'expert_ids': [107, 115, 43, 67, 127, 103, 59, 111, 15, 99, 55, 27, 47, 83, 8, 124, 100, 20, 68, 96, 112, 108, 56, 60, 84, 104, 36, 12, 29, 49, 57, 105, 77, 101, 45, 25, 53, 65, 121, 97, 13, 33, 34, 54, 74, 18, 42, 46, 82, 66, 70, 114, 118, 86, 106], 'token_total': 319, 'token_per_expert': {107: 2, 115: 2, 43: 3, 67: 3, 127: 3, 103: 6, 59: 7, 111: 8, 15: 9, 99: 10, 55: 12, 27: 13, 47: 13, 83: 14, 8: 1, 124: 1, 100: 2, 20: 4, 68: 5, 96: 7, 112: 7, 108: 9, 56: 10, 60: 10, 84: 10, 104: 12, 36: 17, 12: 18, 29: 1, 49: 1, 57: 1, 105: 1, 77: 2, 101: 2, 45: 3, 25: 5, 53: 5, 65: 5, 121: 5, 97: 8, 13: 9, 33: 9, 34: 1, 54: 1, 74: 1, 18: 2, 42: 2, 46: 2, 82: 2, 66: 3, 70: 3, 114: 4, 118: 5, 86: 9, 106: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 39, 51, 63, 75, 79, 119, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 896, 'token_per_expert': {3: 301, 7: 304, 11: 18, 19: 20, 23: 22, 31: 21, 35: 17, 39: 21, 51: 55, 63: 22, 75: 18, 79: 25, 119: 17, 123: 35}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 24, 40, 44, 48, 52, 64, 72, 76, 80, 88, 92], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1035, 'token_per_expert': {0: 263, 4: 258, 16: 23, 24: 37, 40: 37, 44: 74, 48: 21, 52: 109, 64: 64, 72: 18, 76: 26, 80: 22, 88: 32, 92: 51}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 37, 41, 61, 69, 73, 89, 109, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 955, 'token_per_expert': {1: 282, 5: 274, 9: 29, 17: 12, 21: 24, 37: 81, 41: 15, 61: 31, 69: 17, 73: 14, 89: 77, 109: 27, 117: 53, 125: 19}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 26, 38, 50, 58, 90, 98, 102, 122, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 891, 'token_per_expert': {2: 270, 6: 260, 10: 28, 22: 12, 26: 19, 38: 97, 50: 34, 58: 13, 90: 12, 98: 14, 102: 30, 122: 90, 126: 12}}
INFO 05-06 10:01:08.435113.435113 lmp.py:1833] [layer_moe_fused] layer=19 prefix: 0.421ms alloc: 0.401ms
INFO 05-06 10:01:08.435510.435510 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-06 10:01:08.436889.436889 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008916854858398438s
INFO 05-06 10:01:08.437498.437498 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005269050598144531 seconds
INFO 05-06 10:01:08.454523.454523 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01677083969116211s
INFO 05-06 10:01:08.458904.458904 lmp.py:1938] [layer_moe_fused] vllm triton time: 3.956ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.459974.459974 cuda_h.py:27] end *layer_moe_fused cost 24.774 ms
DEBUG 05-06 10:01:08.466253.466253 cuda_h.py:27] end prefill_layer cost 35.020 ms
DEBUG 05-06 10:01:08.466408.466408 lmp.py:1388] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 10:01:08.466357.466357 lmp.py:1346] -------------------------------- start prefill layer 20 --------------------------------
experts_cpu_alloc {'expert_ids': [31, 51, 87, 99, 11, 35, 47, 111, 83, 16, 80, 104, 24, 36, 60, 76, 12, 20, 84, 120, 100, 72, 64, 112, 17, 89, 25, 61, 117, 69, 105, 97, 101, 9, 93, 121, 113, 41, 81, 85, 53, 86, 110, 126, 34, 62, 70, 90, 106, 74, 18, 26, 98, 38, 10, 58], 'token_total': 331, 'token_per_expert': {31: 1, 51: 1, 87: 1, 99: 2, 11: 4, 35: 5, 47: 6, 111: 6, 83: 7, 16: 1, 80: 1, 104: 2, 24: 5, 36: 5, 60: 6, 76: 6, 12: 8, 20: 8, 84: 9, 120: 9, 100: 11, 72: 12, 64: 13, 112: 14, 17: 1, 89: 1, 25: 2, 61: 2, 117: 2, 69: 3, 105: 3, 97: 4, 101: 4, 9: 8, 93: 8, 121: 8, 113: 11, 41: 13, 81: 14, 85: 16, 53: 17, 86: 1, 110: 2, 126: 2, 34: 3, 62: 3, 70: 3, 90: 3, 106: 3, 74: 4, 18: 6, 26: 6, 98: 6, 38: 9, 10: 10, 58: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 43, 55, 59, 63, 71, 79, 95, 103, 107, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 24, 'token_total': 892, 'token_per_expert': {3: 295, 7: 257, 15: 9, 19: 8, 27: 25, 43: 28, 55: 9, 59: 26, 63: 67, 71: 14, 79: 10, 95: 8, 103: 10, 107: 98, 123: 28}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 28, 32, 40, 44, 52, 56, 68, 88, 92, 108, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1044, 'token_per_expert': {0: 266, 4: 302, 8: 29, 28: 36, 32: 23, 40: 38, 44: 30, 52: 20, 56: 43, 68: 146, 88: 23, 92: 37, 108: 27, 116: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 33, 37, 45, 49, 57, 65, 73, 77, 109, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 31, 'token_total': 1021, 'token_per_expert': {1: 264, 5: 301, 13: 25, 21: 36, 33: 19, 37: 35, 45: 77, 49: 77, 57: 34, 65: 28, 73: 36, 77: 42, 109: 27, 125: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 42, 46, 50, 54, 66, 82, 94, 102, 114, 118, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 808, 'token_per_expert': {2: 264, 6: 257, 30: 28, 42: 19, 46: 11, 50: 12, 54: 19, 66: 26, 82: 17, 94: 77, 102: 46, 114: 12, 118: 10, 122: 10}}
INFO 05-06 10:01:08.470660.470660 lmp.py:1833] [layer_moe_fused] layer=20 prefix: 0.426ms alloc: 0.413ms
INFO 05-06 10:01:08.471765.471765 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.9604644775390625e-05 seconds
INFO 05-06 10:01:08.472997.472997 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0009236335754394531s
INFO 05-06 10:01:08.472395.472395 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005450248718261719 seconds
INFO 05-06 10:01:08.486971.486971 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013385534286499023s
INFO 05-06 10:01:08.490088.490088 lmp.py:1938] [layer_moe_fused] vllm triton time: 3.966ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.491125.491125 cuda_h.py:27] end *layer_moe_fused cost 21.443 ms
DEBUG 05-06 10:01:08.497552.497552 cuda_h.py:27] end prefill_layer cost 30.929 ms
DEBUG 05-06 10:01:08.497376.497376 lmp.py:1388] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 10:01:08.497489.497489 lmp.py:1346] -------------------------------- start prefill layer 21 --------------------------------
experts_cpu_alloc {'expert_ids': [19, 47, 99, 27, 43, 59, 107, 23, 71, 115, 55, 119, 31, 56, 16, 52, 60, 104, 32, 64, 88, 96, 20, 40, 80, 24, 44, 36, 25, 77, 17, 113, 101, 69, 125, 93, 45, 121, 21, 81, 33, 54, 98, 106, 94, 126, 114, 50, 38, 74, 82, 118, 34, 58, 70, 42], 'token_total': 314, 'token_per_expert': {19: 1, 47: 1, 99: 1, 27: 2, 43: 2, 59: 2, 107: 2, 23: 3, 71: 6, 115: 6, 55: 7, 119: 8, 31: 10, 56: 1, 16: 2, 52: 2, 60: 2, 104: 2, 32: 3, 64: 3, 88: 5, 96: 5, 20: 8, 40: 8, 80: 8, 24: 10, 44: 10, 36: 14, 25: 1, 77: 1, 17: 2, 113: 3, 101: 4, 69: 5, 125: 5, 93: 6, 45: 7, 121: 7, 21: 8, 81: 9, 33: 12, 54: 2, 98: 2, 106: 3, 94: 4, 126: 4, 114: 6, 50: 7, 38: 8, 74: 9, 82: 10, 118: 10, 34: 11, 58: 11, 70: 11, 42: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 35, 51, 67, 75, 79, 83, 87, 95, 103, 111, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 804, 'token_per_expert': {3: 259, 7: 281, 11: 38, 35: 28, 51: 35, 67: 12, 75: 13, 79: 13, 83: 28, 87: 11, 95: 11, 103: 23, 111: 27, 123: 10, 127: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 48, 68, 72, 76, 84, 92, 100, 112, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 921, 'token_per_expert': {0: 257, 4: 277, 8: 27, 12: 15, 48: 54, 68: 23, 72: 19, 76: 33, 84: 28, 92: 38, 100: 62, 112: 35, 120: 29, 124: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 37, 41, 53, 57, 61, 65, 73, 97, 105, 109], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1024, 'token_per_expert': {1: 324, 5: 349, 13: 27, 29: 30, 37: 24, 41: 20, 53: 29, 57: 19, 61: 25, 65: 62, 73: 32, 97: 20, 105: 40, 109: 23}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 26, 30, 46, 62, 78, 86, 90, 102, 110, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1033, 'token_per_expert': {2: 274, 6: 333, 10: 14, 18: 40, 26: 57, 30: 19, 46: 41, 62: 25, 78: 99, 86: 14, 90: 26, 102: 22, 110: 33, 122: 36}}
INFO 05-06 10:01:08.502215.502215 lmp.py:1833] [layer_moe_fused] layer=21 prefix: 0.418ms alloc: 0.410ms
INFO 05-06 10:01:08.502266.502266 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.246566772460938e-05 seconds
INFO 05-06 10:01:08.503348.503348 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007901191711425781s
INFO 05-06 10:01:08.503308.503308 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005388259887695312 seconds
INFO 05-06 10:01:08.519154.519154 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015163660049438477s
INFO 05-06 10:01:08.523601.523601 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.145ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.524460.524460 cuda_h.py:27] end *layer_moe_fused cost 23.166 ms
DEBUG 05-06 10:01:08.530846.530846 cuda_h.py:27] end prefill_layer cost 32.979 ms
DEBUG 05-06 10:01:08.530616.530616 lmp.py:1388] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 10:01:08.530388.530388 lmp.py:1346] -------------------------------- start prefill layer 22 --------------------------------
experts_cpu_alloc {'expert_ids': [23, 63, 71, 27, 67, 95, 39, 87, 47, 51, 83, 11, 15, 107, 79, 115, 31, 36, 60, 104, 20, 96, 84, 112, 48, 32, 44, 16, 124, 88, 28, 9, 13, 17, 29, 61, 109, 65, 81, 125, 45, 25, 14, 54, 110, 62, 98, 106, 122, 10, 42, 34, 102, 26, 58], 'token_total': 292, 'token_per_expert': {23: 1, 63: 1, 71: 1, 27: 2, 67: 2, 95: 2, 39: 3, 87: 4, 47: 6, 51: 6, 83: 6, 11: 8, 15: 8, 107: 9, 79: 10, 115: 10, 31: 14, 36: 1, 60: 1, 104: 1, 20: 2, 96: 2, 84: 3, 112: 5, 48: 6, 32: 8, 44: 9, 16: 10, 124: 12, 88: 14, 28: 15, 9: 1, 13: 1, 17: 1, 29: 2, 61: 2, 109: 2, 65: 3, 81: 3, 125: 4, 45: 5, 25: 6, 14: 1, 54: 1, 110: 1, 62: 2, 98: 2, 106: 3, 122: 5, 10: 7, 42: 7, 34: 11, 102: 12, 26: 14, 58: 14}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 35, 43, 55, 59, 75, 99, 103, 111, 119, 123, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 31, 'token_total': 962, 'token_per_expert': {3: 257, 7: 281, 19: 19, 35: 79, 43: 26, 55: 34, 59: 42, 75: 27, 99: 18, 103: 49, 111: 25, 119: 38, 123: 36, 127: 31}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 24, 40, 64, 68, 72, 76, 92, 100, 108, 116, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 1170, 'token_per_expert': {0: 260, 4: 256, 8: 32, 24: 66, 40: 15, 64: 107, 68: 43, 72: 98, 76: 22, 92: 30, 100: 155, 108: 25, 116: 32, 120: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 41, 53, 57, 69, 73, 85, 89, 93, 101, 113, 117], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 786, 'token_per_expert': {1: 274, 5: 257, 33: 12, 41: 10, 53: 28, 57: 6, 69: 15, 73: 48, 85: 9, 89: 16, 93: 49, 101: 14, 113: 14, 117: 34}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 38, 46, 66, 70, 74, 82, 86, 90, 94, 118, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 886, 'token_per_expert': {2: 257, 6: 257, 30: 15, 38: 29, 46: 19, 66: 18, 70: 31, 74: 68, 82: 23, 86: 31, 90: 32, 94: 25, 118: 15, 126: 66}}
INFO 05-06 10:01:08.535510.535510 lmp.py:1833] [layer_moe_fused] layer=22 prefix: 0.421ms alloc: 0.400ms
INFO 05-06 10:01:08.535607.535607 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.031990051269531e-05 seconds
INFO 05-06 10:01:08.536442.536442 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008077621459960938s
INFO 05-06 10:01:08.537283.537283 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005209445953369141 seconds
INFO 05-06 10:01:08.553798.553798 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.01560664176940918s
INFO 05-06 10:01:08.557109.557109 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.011ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.557583.557583 cuda_h.py:27] end *layer_moe_fused cost 23.383 ms
DEBUG 05-06 10:01:08.563526.563526 cuda_h.py:27] end prefill_layer cost 32.221 ms
DEBUG 05-06 10:01:08.563397.563397 lmp.py:1388] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 10:01:08.563020.563020 lmp.py:1346] -------------------------------- start prefill layer 23 --------------------------------
experts_cpu_alloc {'expert_ids': [55, 119, 11, 23, 27, 95, 99, 107, 127, 19, 75, 51, 103, 91, 28, 88, 60, 64, 12, 92, 36, 52, 120, 32, 40, 48, 68, 76, 124, 112, 13, 69, 77, 41, 53, 81, 49, 89, 57, 33, 9, 73, 17, 50, 62, 74, 66, 82, 102, 54, 58, 110, 10, 14, 38, 122, 42], 'token_total': 276, 'token_per_expert': {55: 1, 119: 1, 11: 2, 23: 2, 27: 2, 95: 2, 99: 2, 107: 3, 127: 3, 19: 6, 75: 6, 51: 7, 103: 9, 91: 10, 28: 1, 88: 1, 60: 2, 64: 2, 12: 3, 92: 4, 36: 5, 52: 5, 120: 5, 32: 6, 40: 6, 48: 6, 68: 6, 76: 6, 124: 10, 112: 12, 13: 1, 69: 1, 77: 1, 41: 3, 53: 3, 81: 3, 49: 4, 89: 5, 57: 7, 33: 9, 9: 10, 73: 17, 17: 18, 50: 1, 62: 2, 74: 2, 66: 3, 82: 3, 102: 3, 54: 4, 58: 4, 110: 4, 10: 5, 14: 5, 38: 5, 122: 7, 42: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 39, 43, 47, 59, 67, 71, 79, 83, 87, 115, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 969, 'token_per_expert': {3: 285, 7: 260, 31: 15, 35: 30, 39: 59, 43: 53, 47: 28, 59: 15, 67: 68, 71: 11, 79: 45, 83: 30, 87: 18, 115: 16, 123: 36}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 24, 44, 56, 72, 80, 84, 100, 104, 108, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 801, 'token_per_expert': {0: 256, 4: 257, 8: 16, 16: 22, 24: 14, 44: 35, 56: 60, 72: 16, 80: 17, 84: 22, 100: 22, 104: 22, 108: 30, 116: 12}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25, 29, 37, 61, 65, 85, 97, 105, 109, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1128, 'token_per_expert': {1: 294, 5: 300, 21: 104, 25: 40, 29: 40, 37: 44, 61: 59, 65: 43, 85: 37, 97: 48, 105: 18, 109: 32, 117: 24, 125: 45}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 26, 30, 34, 46, 78, 86, 90, 98, 106, 118], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 922, 'token_per_expert': {2: 272, 6: 273, 18: 28, 22: 18, 26: 15, 30: 14, 34: 12, 46: 57, 78: 21, 86: 84, 90: 33, 98: 43, 106: 14, 118: 38}}
INFO 05-06 10:01:08.567568.567568 lmp.py:1833] [layer_moe_fused] layer=23 prefix: 0.415ms alloc: 0.429ms
INFO 05-06 10:01:08.567864.567864 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.9604644775390625e-05 seconds
INFO 05-06 10:01:08.568614.568614 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007569789886474609s
INFO 05-06 10:01:08.569230.569230 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005311965942382812 seconds
INFO 05-06 10:01:08.584727.584727 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013987064361572266s
INFO 05-06 10:01:08.588651.588651 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.325ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.589178.589178 cuda_h.py:27] end *layer_moe_fused cost 22.047 ms
DEBUG 05-06 10:01:08.595450.595450 cuda_h.py:27] end prefill_layer cost 32.187 ms
DEBUG 05-06 10:01:08.595168.595168 lmp.py:1388] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 10:01:08.595984.595984 lmp.py:1346] -------------------------------- start prefill layer 24 --------------------------------
experts_cpu_alloc {'expert_ids': [15, 47, 87, 95, 99, 31, 107, 115, 55, 119, 75, 79, 127, 24, 28, 80, 112, 40, 76, 104, 32, 68, 84, 96, 92, 124, 20, 100, 25, 41, 93, 113, 65, 61, 105, 53, 57, 9, 49, 10, 26, 58, 106, 22, 78, 102, 126, 38, 42, 66, 46, 62, 74, 82, 118], 'token_total': 245, 'token_per_expert': {15: 1, 47: 1, 87: 1, 95: 1, 99: 2, 31: 3, 107: 3, 115: 3, 55: 5, 119: 6, 75: 7, 79: 7, 127: 8, 24: 2, 28: 2, 80: 2, 112: 2, 40: 3, 76: 3, 104: 3, 32: 4, 68: 4, 84: 4, 96: 4, 92: 6, 124: 6, 20: 7, 100: 9, 25: 1, 41: 1, 93: 1, 113: 1, 65: 2, 61: 4, 105: 5, 53: 6, 57: 9, 9: 10, 49: 10, 10: 1, 26: 1, 58: 1, 106: 1, 22: 2, 78: 2, 102: 3, 126: 3, 38: 5, 42: 5, 66: 5, 46: 6, 62: 12, 74: 13, 82: 13, 118: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 35, 43, 63, 67, 71, 83, 91, 111], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 952, 'token_per_expert': {3: 256, 7: 269, 11: 53, 19: 38, 23: 20, 27: 78, 35: 23, 43: 15, 63: 61, 67: 45, 71: 34, 83: 17, 91: 32, 111: 11}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 36, 44, 48, 52, 56, 60, 64, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 928, 'token_per_expert': {0: 256, 4: 293, 8: 11, 12: 37, 16: 27, 36: 19, 44: 50, 48: 22, 52: 66, 56: 39, 60: 17, 64: 71, 108: 11, 120: 9}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 29, 33, 37, 45, 73, 77, 81, 97, 109, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 1001, 'token_per_expert': {1: 283, 5: 269, 13: 14, 17: 28, 29: 18, 33: 58, 37: 18, 45: 38, 73: 22, 77: 22, 81: 10, 97: 75, 109: 23, 121: 123}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 34, 50, 70, 86, 90, 94, 98, 110, 114, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 970, 'token_per_expert': {2: 256, 6: 306, 30: 19, 34: 43, 50: 14, 70: 71, 86: 17, 90: 111, 94: 26, 98: 39, 110: 21, 114: 29, 122: 18}}
INFO 05-06 10:01:08.599188.599188 lmp.py:1833] [layer_moe_fused] layer=24 prefix: 0.417ms alloc: 0.404ms
INFO 05-06 10:01:08.600669.600669 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.7697296142578125e-05 seconds
INFO 05-06 10:01:08.601022.601022 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007584095001220703s
INFO 05-06 10:01:08.601492.601492 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005285739898681641 seconds
INFO 05-06 10:01:08.617435.617435 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.015557527542114258s
INFO 05-06 10:01:08.622542.622542 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.283ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.622653.622653 cuda_h.py:27] end *layer_moe_fused cost 23.428 ms
DEBUG 05-06 10:01:08.629465.629465 cuda_h.py:27] end prefill_layer cost 33.737 ms
DEBUG 05-06 10:01:08.629282.629282 lmp.py:1388] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 10:01:08.629575.629575 lmp.py:1346] -------------------------------- start prefill layer 25 --------------------------------
experts_cpu_alloc {'expert_ids': [15, 95, 127, 55, 27, 23, 31, 75, 103, 119, 47, 79, 87, 99, 51, 91, 76, 84, 108, 24, 12, 92, 124, 112, 72, 88, 8, 48, 116, 57, 61, 33, 73, 81, 13, 53, 121, 29, 21, 49, 9, 109, 30, 38, 74, 42, 22, 26, 66, 118, 122, 46, 126, 50, 78], 'token_total': 266, 'token_per_expert': {15: 1, 95: 1, 127: 1, 55: 2, 27: 3, 23: 4, 31: 4, 75: 4, 103: 5, 119: 5, 47: 6, 79: 6, 87: 7, 99: 7, 51: 9, 91: 10, 76: 1, 84: 1, 108: 1, 24: 2, 12: 3, 92: 4, 124: 5, 112: 6, 72: 7, 88: 7, 8: 8, 48: 12, 116: 12, 57: 1, 61: 1, 33: 2, 73: 2, 81: 2, 13: 3, 53: 3, 121: 4, 29: 5, 21: 6, 49: 6, 9: 7, 109: 8, 30: 1, 38: 3, 74: 3, 42: 4, 22: 5, 26: 5, 66: 5, 118: 5, 122: 5, 46: 6, 126: 8, 50: 10, 78: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 35, 39, 43, 63, 67, 71, 83, 107, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 919, 'token_per_expert': {3: 284, 7: 273, 11: 19, 19: 11, 35: 62, 39: 14, 43: 11, 63: 13, 67: 20, 71: 20, 83: 27, 107: 75, 111: 25, 123: 65}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 36, 44, 52, 56, 60, 64, 68, 80, 100, 104, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 1059, 'token_per_expert': {0: 268, 4: 259, 16: 156, 36: 18, 44: 18, 52: 53, 56: 14, 60: 47, 64: 40, 68: 85, 80: 28, 100: 22, 104: 32, 120: 19}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 25, 41, 45, 69, 77, 85, 89, 93, 97, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 825, 'token_per_expert': {1: 256, 5: 264, 17: 9, 25: 11, 41: 11, 45: 71, 69: 44, 77: 12, 85: 53, 89: 10, 93: 21, 97: 12, 117: 42, 125: 9}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 34, 58, 70, 82, 90, 106, 110, 114], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 1027, 'token_per_expert': {2: 307, 6: 259, 10: 14, 14: 13, 18: 61, 34: 23, 58: 132, 70: 34, 82: 21, 90: 22, 106: 22, 110: 100, 114: 19}}
INFO 05-06 10:01:08.634970.634970 lmp.py:1833] [layer_moe_fused] layer=25 prefix: 0.414ms alloc: 0.395ms
INFO 05-06 10:01:08.634511.634511 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.198883056640625e-05 seconds
INFO 05-06 10:01:08.635271.635271 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0008203983306884766s
INFO 05-06 10:01:08.635993.635993 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005381107330322266 seconds
INFO 05-06 10:01:08.648747.648747 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012098312377929688s
INFO 05-06 10:01:08.652407.652407 lmp.py:1938] [layer_moe_fused] vllm triton time: 3.957ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.652617.652617 cuda_h.py:27] end *layer_moe_fused cost 19.547 ms
DEBUG 05-06 10:01:08.656409.656409 cuda_h.py:27] end prefill_layer cost 27.257 ms
DEBUG 05-06 10:01:08.656795.656795 lmp.py:1388] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 10:01:08.657527.657527 lmp.py:1346] -------------------------------- start prefill layer 26 --------------------------------
experts_cpu_alloc {'expert_ids': [107, 127, 47, 71, 91, 31, 55, 23, 63, 35, 67, 115, 99, 75, 48, 64, 32, 44, 40, 80, 92, 116, 28, 72, 108, 96, 112, 88, 8, 53, 117, 121, 9, 109, 13, 41, 81, 57, 61, 25, 125, 29, 77, 54, 18, 42, 122, 50, 74, 118, 14, 26], 'token_total': 290, 'token_per_expert': {107: 1, 127: 1, 47: 3, 71: 3, 91: 3, 31: 4, 55: 4, 23: 6, 63: 6, 35: 7, 67: 8, 115: 8, 99: 10, 75: 15, 48: 1, 64: 1, 32: 3, 44: 3, 40: 4, 80: 4, 92: 4, 116: 4, 28: 5, 72: 5, 108: 5, 96: 6, 112: 7, 88: 8, 8: 9, 53: 1, 117: 1, 121: 1, 9: 2, 109: 3, 13: 4, 41: 4, 81: 6, 57: 8, 61: 8, 25: 9, 125: 9, 29: 10, 77: 11, 54: 4, 18: 5, 42: 5, 122: 5, 50: 7, 74: 7, 118: 10, 14: 11, 26: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 43, 51, 59, 79, 87, 95, 103, 111, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 993, 'token_per_expert': {3: 269, 7: 256, 15: 24, 19: 21, 27: 49, 43: 48, 51: 19, 59: 18, 79: 17, 87: 88, 95: 63, 103: 18, 111: 78, 123: 25}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 36, 52, 56, 60, 68, 76, 84, 104, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 906, 'token_per_expert': {0: 260, 4: 260, 20: 107, 24: 47, 36: 14, 52: 32, 56: 15, 60: 25, 68: 14, 76: 17, 84: 55, 104: 36, 124: 24}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 37, 45, 49, 65, 73, 85, 89, 97, 105, 113], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 1110, 'token_per_expert': {1: 262, 5: 256, 17: 96, 37: 14, 45: 14, 49: 19, 65: 60, 73: 32, 85: 139, 89: 86, 97: 14, 105: 14, 113: 104}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 30, 38, 66, 70, 78, 86, 90, 102, 114, 126], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 22, 'token_total': 797, 'token_per_expert': {2: 261, 6: 256, 10: 19, 30: 12, 38: 11, 66: 20, 70: 25, 78: 21, 86: 28, 90: 18, 102: 20, 114: 82, 126: 24}}
INFO 05-06 10:01:08.661075.661075 lmp.py:1833] [layer_moe_fused] layer=26 prefix: 0.414ms alloc: 0.386ms
INFO 05-06 10:01:08.661312.661312 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.8650970458984375e-05 seconds
INFO 05-06 10:01:08.662286.662286 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007774829864501953s
INFO 05-06 10:01:08.663140.663140 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005304813385009766 seconds
INFO 05-06 10:01:08.683766.683766 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.019680500030517578s
INFO 05-06 10:01:08.687399.687399 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.343ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.688193.688193 cuda_h.py:27] end *layer_moe_fused cost 27.645 ms
DEBUG 05-06 10:01:08.694242.694242 cuda_h.py:27] end prefill_layer cost 37.176 ms
DEBUG 05-06 10:01:08.694159.694159 lmp.py:1388] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 10:01:08.694715.694715 lmp.py:1346] -------------------------------- start prefill layer 27 --------------------------------
experts_cpu_alloc {'expert_ids': [11, 55, 59, 19, 63, 67, 15, 47, 39, 83, 91, 27, 23, 75, 127, 44, 80, 84, 124, 32, 116, 68, 96, 20, 40, 28, 112, 89, 101, 29, 93, 97, 81, 113, 21, 125, 105, 49, 22, 30, 86, 110, 26, 126, 10, 58, 54, 74, 90, 106, 114, 42, 66, 94], 'token_total': 331, 'token_per_expert': {11: 1, 55: 1, 59: 2, 19: 3, 63: 3, 67: 4, 15: 5, 47: 5, 39: 6, 83: 8, 91: 8, 27: 9, 23: 10, 75: 11, 127: 11, 44: 1, 80: 1, 84: 1, 124: 3, 32: 4, 116: 5, 68: 7, 96: 7, 20: 9, 40: 9, 28: 12, 112: 13, 89: 1, 101: 1, 29: 2, 93: 2, 97: 2, 81: 3, 113: 3, 21: 6, 125: 6, 105: 8, 49: 10, 22: 1, 30: 1, 86: 2, 110: 2, 26: 3, 126: 3, 10: 4, 58: 8, 54: 9, 74: 9, 90: 10, 106: 10, 114: 15, 42: 17, 66: 17, 94: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 43, 51, 79, 87, 95, 103, 111, 115, 119, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 1054, 'token_per_expert': {3: 290, 7: 275, 31: 27, 35: 17, 43: 67, 51: 28, 79: 32, 87: 83, 95: 58, 103: 43, 111: 34, 115: 49, 119: 13, 123: 38}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 24, 36, 48, 56, 64, 76, 88, 100, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 954, 'token_per_expert': {0: 260, 4: 270, 8: 20, 12: 14, 24: 45, 36: 34, 48: 38, 56: 17, 64: 19, 76: 48, 88: 46, 100: 61, 108: 22, 120: 60}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 33, 37, 41, 45, 53, 61, 65, 85, 109, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 25, 'token_total': 897, 'token_per_expert': {1: 293, 5: 257, 13: 27, 25: 38, 33: 40, 37: 25, 41: 20, 45: 51, 53: 23, 61: 19, 65: 43, 85: 14, 109: 26, 121: 21}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 46, 50, 62, 70, 78, 82, 98, 118, 122], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 29, 'token_total': 860, 'token_per_expert': {2: 256, 6: 259, 14: 33, 18: 25, 46: 30, 50: 64, 62: 29, 70: 25, 78: 28, 82: 50, 98: 22, 118: 22, 122: 17}}
INFO 05-06 10:01:08.699758.699758 lmp.py:1833] [layer_moe_fused] layer=27 prefix: 0.424ms alloc: 0.404ms
INFO 05-06 10:01:08.699525.699525 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.246566772460938e-05 seconds
INFO 05-06 10:01:08.700704.700704 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007717609405517578s
INFO 05-06 10:01:08.700831.700831 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005555152893066406 seconds
INFO 05-06 10:01:08.714129.714129 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.013644218444824219s
INFO 05-06 10:01:08.719455.719455 lmp.py:1938] [layer_moe_fused] vllm triton time: 4.474ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.720634.720634 cuda_h.py:27] end *layer_moe_fused cost 21.841 ms
DEBUG 05-06 10:01:08.725827.725827 cuda_h.py:27] end prefill_layer cost 30.621 ms
DEBUG 05-06 10:01:08.725843.725843 lmp.py:1388] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 10:01:08.725445.725445 lmp.py:1346] -------------------------------- start prefill layer 28 --------------------------------
experts_cpu_alloc {'expert_ids': [19, 59, 67, 83, 127, 87, 99, 15, 79, 23, 39, 43, 28, 56, 108, 120, 92, 36, 44, 48, 88, 60, 84, 29, 61, 93, 109, 73, 81, 9, 33, 65, 117, 121, 69, 97, 105, 34, 38, 58, 102, 66, 50, 54, 94, 98, 122, 126, 62], 'token_total': 202, 'token_per_expert': {19: 1, 59: 1, 67: 1, 83: 1, 127: 1, 87: 2, 99: 2, 15: 3, 79: 5, 23: 6, 39: 6, 43: 8, 28: 1, 56: 1, 108: 2, 120: 2, 92: 3, 36: 4, 44: 4, 48: 5, 88: 6, 60: 7, 84: 7, 29: 1, 61: 1, 93: 1, 109: 1, 73: 3, 81: 4, 9: 5, 33: 7, 65: 7, 117: 7, 121: 9, 69: 10, 97: 11, 105: 11, 34: 1, 38: 1, 58: 1, 102: 1, 66: 2, 50: 3, 54: 3, 94: 4, 98: 5, 122: 7, 126: 8, 62: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 47, 55, 71, 75, 91, 95, 111, 115, 119, 123], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 1061, 'token_per_expert': {3: 258, 7: 257, 11: 39, 47: 36, 55: 19, 71: 34, 75: 44, 91: 49, 95: 9, 111: 197, 115: 81, 119: 26, 123: 12}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 32, 40, 52, 68, 76, 100, 104, 112], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 24, 'token_total': 1148, 'token_per_expert': {0: 256, 4: 256, 12: 198, 20: 151, 24: 8, 32: 35, 40: 34, 52: 28, 68: 23, 76: 70, 100: 8, 104: 15, 112: 66}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 37, 49, 53, 57, 77, 85, 89, 101, 113], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 26, 'token_total': 931, 'token_per_expert': {1: 269, 5: 268, 13: 24, 37: 13, 49: 131, 53: 31, 57: 88, 77: 17, 85: 18, 89: 16, 101: 22, 113: 34}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 30, 46, 70, 74, 78, 90, 106, 110], 'expert_count': 12, 'ideal_gpu_count': 12, 'keep_on_gpu': 12, 'hit_count_on_device': 24, 'token_total': 754, 'token_per_expert': {2: 256, 6: 257, 18: 10, 22: 33, 30: 19, 46: 23, 70: 32, 74: 10, 78: 18, 90: 45, 106: 12, 110: 39}}
INFO 05-06 10:01:08.730968.730968 lmp.py:1833] [layer_moe_fused] layer=28 prefix: 0.418ms alloc: 0.372ms
INFO 05-06 10:01:08.730357.730357 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-06 10:01:08.731200.731200 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0007863044738769531s
INFO 05-06 10:01:08.731637.731637 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005400180816650391 seconds
INFO 05-06 10:01:08.744916.744916 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.012395143508911133s
INFO 05-06 10:01:08.748397.748397 lmp.py:1938] [layer_moe_fused] vllm triton time: 3.964ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.748204.748204 cuda_h.py:27] end *layer_moe_fused cost 19.717 ms
DEBUG 05-06 10:01:08.754714.754714 cuda_h.py:27] end prefill_layer cost 28.981 ms
DEBUG 05-06 10:01:08.754631.754631 lmp.py:1388] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 10:01:08.754811.754811 lmp.py:1346] -------------------------------- start prefill layer 29 --------------------------------
experts_cpu_alloc {'expert_ids': [87, 127, 55, 111, 123, 11, 35, 63, 75, 95, 115, 119, 104, 108, 84, 96, 76, 88, 40, 80, 92, 32, 24, 8, 17, 33, 105, 25, 37, 125, 13, 21, 65, 77, 61, 81, 9, 69, 73, 89, 122, 126, 70, 118, 74, 94, 38, 50, 58, 46, 10, 114, 66, 30, 78], 'token_total': 343, 'token_per_expert': {87: 2, 127: 2, 55: 3, 111: 4, 123: 5, 11: 6, 35: 6, 63: 6, 75: 6, 95: 6, 115: 6, 119: 6, 104: 1, 108: 1, 84: 2, 96: 2, 76: 3, 88: 3, 40: 4, 80: 4, 92: 5, 32: 6, 24: 8, 8: 9, 17: 1, 33: 3, 105: 3, 25: 4, 37: 4, 125: 4, 13: 5, 21: 6, 65: 6, 77: 11, 61: 14, 81: 14, 9: 15, 69: 15, 73: 18, 89: 19, 122: 1, 126: 1, 70: 2, 118: 2, 74: 3, 94: 3, 38: 4, 50: 6, 58: 6, 46: 7, 10: 8, 114: 10, 66: 12, 30: 13, 78: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 43, 67, 71, 83, 91, 99, 107], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 1019, 'token_per_expert': {3: 269, 7: 358, 15: 9, 19: 38, 23: 26, 27: 27, 31: 7, 43: 48, 67: 11, 71: 38, 83: 10, 91: 90, 99: 78, 107: 10}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 28, 44, 48, 52, 56, 60, 64, 116, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 998, 'token_per_expert': {0: 257, 4: 301, 16: 33, 20: 53, 28: 43, 44: 14, 48: 27, 52: 74, 56: 40, 60: 28, 64: 76, 116: 11, 120: 14, 124: 27}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 49, 53, 57, 85, 93, 97, 101, 109, 113, 117, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 847, 'token_per_expert': {1: 261, 5: 256, 29: 23, 49: 25, 53: 25, 57: 45, 85: 21, 93: 24, 97: 28, 101: 20, 109: 19, 113: 20, 117: 39, 121: 41}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 42, 54, 62, 82, 86, 90, 106], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 28, 'token_total': 889, 'token_per_expert': {2: 276, 6: 264, 14: 20, 18: 22, 22: 25, 26: 28, 42: 47, 54: 28, 62: 20, 82: 32, 86: 51, 90: 30, 106: 46}}
INFO 05-06 10:01:08.759373.759373 lmp.py:1833] [layer_moe_fused] layer=29 prefix: 0.417ms alloc: 0.404ms
INFO 05-06 10:01:08.759113.759113 lmp.py:1847] [layer_moe_fused] get_experts_task_ids time: 6.246566772460938e-05 seconds
INFO 05-06 10:01:08.760970.760970 lmp.py:1855] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.000759124755859375s
INFO 05-06 10:01:08.761201.761201 lmp.py:1884] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005271434783935547 seconds
INFO 05-06 10:01:08.775412.775412 lmp.py:1895] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.014185667037963867s
INFO 05-06 10:01:08.779548.779548 lmp.py:1938] [layer_moe_fused] vllm triton time: 3.953ms (seq_len=128 cg=False)
DEBUG 05-06 10:01:08.780945.780945 cuda_h.py:27] end *layer_moe_fused cost 21.857 ms
DEBUG 05-06 10:01:08.781014.781014 cuda_h.py:27] end prefill_layer cost 26.443 ms
DEBUG 05-06 10:01:08.781778.781778 lmp.py:1388] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 10:01:08.781343.781343 cuda_h.py:27] end prefill_step cost 973.881 ms
INFO 05-06 10:01:08.781117.781117 lmp.py:1391] prefill time: 1.2216274738311768 seconds
INFO 05-06 10:01:08.789447.789447 lmp.py:1403] Static-KV prefill complete; seqlens set to 128.
WARNING 05-06 10:01:08.789411.789411 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:08.790299.790299 helper.py:35]   NaN count (hidden): 720896
WARNING 05-06 10:01:08.790979.790979 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:08.790831.790831 helper.py:39]   NaN count (normed): 720896
WARNING 05-06 10:01:08.795336.795336 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:08.795586.795586 helper.py:50]   NaN count: 524288
WARNING 05-06 10:01:08.796799.796799 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:08.798947.798947 cuda_h.py:27] end init_inputs_tokens cost 8.884 ms
DEBUG 05-06 10:01:08.798817.798817 lmp.py:1504] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:08.798402.798402 lmp.py:1510] ---- decode step 0 layer 0 ----
DEBUG 05-06 10:01:08.806358.806358 cuda_h.py:27] end decode_layer cost 7.965 ms
DEBUG 05-06 10:01:08.806559.806559 lmp.py:1510] ---- decode step 0 layer 1 ----
DEBUG 05-06 10:01:08.813366.813366 cuda_h.py:27] end decode_layer cost 6.698 ms
DEBUG 05-06 10:01:08.813660.813660 lmp.py:1510] ---- decode step 0 layer 2 ----
DEBUG 05-06 10:01:08.819041.819041 cuda_h.py:27] end decode_layer cost 6.210 ms
DEBUG 05-06 10:01:08.819652.819652 lmp.py:1510] ---- decode step 0 layer 3 ----
DEBUG 05-06 10:01:08.824107.824107 cuda_h.py:27] end decode_layer cost 5.059 ms
DEBUG 05-06 10:01:08.824764.824764 lmp.py:1510] ---- decode step 0 layer 4 ----
DEBUG 05-06 10:01:08.829912.829912 cuda_h.py:27] end decode_layer cost 4.982 ms
DEBUG 05-06 10:01:08.829093.829093 lmp.py:1510] ---- decode step 0 layer 5 ----
DEBUG 05-06 10:01:08.835116.835116 cuda_h.py:27] end decode_layer cost 5.243 ms
DEBUG 05-06 10:01:08.835112.835112 lmp.py:1510] ---- decode step 0 layer 6 ----
DEBUG 05-06 10:01:08.840363.840363 cuda_h.py:27] end decode_layer cost 4.919 ms
DEBUG 05-06 10:01:08.840497.840497 lmp.py:1510] ---- decode step 0 layer 7 ----
DEBUG 05-06 10:01:08.845882.845882 cuda_h.py:27] end decode_layer cost 4.948 ms
DEBUG 05-06 10:01:08.845539.845539 lmp.py:1510] ---- decode step 0 layer 8 ----
DEBUG 05-06 10:01:08.850439.850439 cuda_h.py:27] end decode_layer cost 4.906 ms
DEBUG 05-06 10:01:08.850097.850097 lmp.py:1510] ---- decode step 0 layer 9 ----
DEBUG 05-06 10:01:08.855826.855826 cuda_h.py:27] end decode_layer cost 4.956 ms
DEBUG 05-06 10:01:08.855007.855007 lmp.py:1510] ---- decode step 0 layer 10 ----
DEBUG 05-06 10:01:08.860896.860896 cuda_h.py:27] end decode_layer cost 4.793 ms
DEBUG 05-06 10:01:08.860885.860885 lmp.py:1510] ---- decode step 0 layer 11 ----
DEBUG 05-06 10:01:08.865462.865462 cuda_h.py:27] end decode_layer cost 5.159 ms
DEBUG 05-06 10:01:08.865166.865166 lmp.py:1510] ---- decode step 0 layer 12 ----
DEBUG 05-06 10:01:08.870062.870062 cuda_h.py:27] end decode_layer cost 4.798 ms
DEBUG 05-06 10:01:08.870719.870719 lmp.py:1510] ---- decode step 0 layer 13 ----
DEBUG 05-06 10:01:08.875318.875318 cuda_h.py:27] end decode_layer cost 4.825 ms
DEBUG 05-06 10:01:08.875877.875877 lmp.py:1510] ---- decode step 0 layer 14 ----
DEBUG 05-06 10:01:08.879139.879139 cuda_h.py:27] end decode_layer cost 4.857 ms
DEBUG 05-06 10:01:08.880081.880081 lmp.py:1510] ---- decode step 0 layer 15 ----
DEBUG 05-06 10:01:08.884314.884314 cuda_h.py:27] end decode_layer cost 4.976 ms
DEBUG 05-06 10:01:08.885257.885257 lmp.py:1510] ---- decode step 0 layer 16 ----
DEBUG 05-06 10:01:08.889883.889883 cuda_h.py:27] end decode_layer cost 4.845 ms
DEBUG 05-06 10:01:08.889256.889256 lmp.py:1510] ---- decode step 0 layer 17 ----
DEBUG 05-06 10:01:08.895500.895500 cuda_h.py:27] end decode_layer cost 5.125 ms
DEBUG 05-06 10:01:08.895489.895489 lmp.py:1510] ---- decode step 0 layer 18 ----
DEBUG 05-06 10:01:08.900468.900468 cuda_h.py:27] end decode_layer cost 4.894 ms
DEBUG 05-06 10:01:08.900649.900649 lmp.py:1510] ---- decode step 0 layer 19 ----
DEBUG 05-06 10:01:08.905556.905556 cuda_h.py:27] end decode_layer cost 4.947 ms
DEBUG 05-06 10:01:08.905737.905737 lmp.py:1510] ---- decode step 0 layer 20 ----
DEBUG 05-06 10:01:08.910897.910897 cuda_h.py:27] end decode_layer cost 4.957 ms
DEBUG 05-06 10:01:08.910839.910839 lmp.py:1510] ---- decode step 0 layer 21 ----
DEBUG 05-06 10:01:08.915776.915776 cuda_h.py:27] end decode_layer cost 5.039 ms
DEBUG 05-06 10:01:08.915242.915242 lmp.py:1510] ---- decode step 0 layer 22 ----
DEBUG 05-06 10:01:08.920335.920335 cuda_h.py:27] end decode_layer cost 4.943 ms
DEBUG 05-06 10:01:08.920324.920324 lmp.py:1510] ---- decode step 0 layer 23 ----
DEBUG 05-06 10:01:08.925517.925517 cuda_h.py:27] end decode_layer cost 5.157 ms
DEBUG 05-06 10:01:08.925221.925221 lmp.py:1510] ---- decode step 0 layer 24 ----
DEBUG 05-06 10:01:08.930714.930714 cuda_h.py:27] end decode_layer cost 4.853 ms
DEBUG 05-06 10:01:08.930465.930465 lmp.py:1510] ---- decode step 0 layer 25 ----
DEBUG 05-06 10:01:08.935406.935406 cuda_h.py:27] end decode_layer cost 4.972 ms
DEBUG 05-06 10:01:08.935395.935395 lmp.py:1510] ---- decode step 0 layer 26 ----
DEBUG 05-06 10:01:08.940116.940116 cuda_h.py:27] end decode_layer cost 4.914 ms
DEBUG 05-06 10:01:08.940297.940297 lmp.py:1510] ---- decode step 0 layer 27 ----
DEBUG 05-06 10:01:08.945457.945457 cuda_h.py:27] end decode_layer cost 4.957 ms
DEBUG 05-06 10:01:08.945492.945492 lmp.py:1510] ---- decode step 0 layer 28 ----
DEBUG 05-06 10:01:08.950741.950741 cuda_h.py:27] end decode_layer cost 4.847 ms
DEBUG 05-06 10:01:08.950921.950921 lmp.py:1510] ---- decode step 0 layer 29 ----
DEBUG 05-06 10:01:08.955940.955940 cuda_h.py:27] end decode_layer cost 5.099 ms
DEBUG 05-06 10:01:08.955606.955606 cuda_h.py:27] end decode_step cost 166.218 ms
INFO 05-06 10:01:08.955012.955012 lmp.py:1558] decode step 0 time: 0.16628122329711914 seconds
WARNING 05-06 10:01:08.955540.955540 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:08.956624.956624 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:08.956347.956347 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:08.956774.956774 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:08.961193.961193 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:08.961918.961918 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:08.961641.961641 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:08.963052.963052 cuda_h.py:27] end init_inputs_tokens cost 7.483 ms
DEBUG 05-06 10:01:08.963326.963326 lmp.py:1504] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:08.963996.963996 lmp.py:1510] ---- decode step 1 layer 0 ----
DEBUG 05-06 10:01:08.968912.968912 cuda_h.py:27] end decode_layer cost 4.812 ms
DEBUG 05-06 10:01:08.968947.968947 lmp.py:1510] ---- decode step 1 layer 1 ----
DEBUG 05-06 10:01:08.973097.973097 cuda_h.py:27] end decode_layer cost 4.845 ms
DEBUG 05-06 10:01:08.973655.973655 lmp.py:1510] ---- decode step 1 layer 2 ----
DEBUG 05-06 10:01:08.977264.977264 cuda_h.py:27] end decode_layer cost 4.727 ms
DEBUG 05-06 10:01:08.977729.977729 lmp.py:1510] ---- decode step 1 layer 3 ----
DEBUG 05-06 10:01:08.982811.982811 cuda_h.py:27] end decode_layer cost 4.794 ms
DEBUG 05-06 10:01:08.982415.982415 lmp.py:1510] ---- decode step 1 layer 4 ----
DEBUG 05-06 10:01:08.987739.987739 cuda_h.py:27] end decode_layer cost 4.728 ms
DEBUG 05-06 10:01:08.987821.987821 lmp.py:1510] ---- decode step 1 layer 5 ----
DEBUG 05-06 10:01:08.992415.992415 cuda_h.py:27] end decode_layer cost 5.066 ms
DEBUG 05-06 10:01:08.992403.992403 lmp.py:1510] ---- decode step 1 layer 6 ----
DEBUG 05-06 10:01:08.997742.997742 cuda_h.py:27] end decode_layer cost 4.773 ms
DEBUG 05-06 10:01:08.997824.997824 lmp.py:1510] ---- decode step 1 layer 7 ----
DEBUG 05-06 10:01:09.002984.002984 cuda_h.py:27] end decode_layer cost 4.782 ms
DEBUG 05-06 10:01:09.002688.002688 lmp.py:1510] ---- decode step 1 layer 8 ----
DEBUG 05-06 10:01:09.007535.007535 cuda_h.py:27] end decode_layer cost 4.727 ms
DEBUG 05-06 10:01:09.007332.007332 lmp.py:1510] ---- decode step 1 layer 9 ----
DEBUG 05-06 10:01:09.012655.012655 cuda_h.py:27] end decode_layer cost 4.902 ms
DEBUG 05-06 10:01:09.012598.012598 lmp.py:1510] ---- decode step 1 layer 10 ----
DEBUG 05-06 10:01:09.016454.016454 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 10:01:09.017158.017158 lmp.py:1510] ---- decode step 1 layer 11 ----
DEBUG 05-06 10:01:09.022367.022367 cuda_h.py:27] end decode_layer cost 5.029 ms
DEBUG 05-06 10:01:09.022879.022879 lmp.py:1510] ---- decode step 1 layer 12 ----
DEBUG 05-06 10:01:09.026648.026648 cuda_h.py:27] end decode_layer cost 4.775 ms
DEBUG 05-06 10:01:09.026445.026445 lmp.py:1510] ---- decode step 1 layer 13 ----
DEBUG 05-06 10:01:09.032347.032347 cuda_h.py:27] end decode_layer cost 5.174 ms
DEBUG 05-06 10:01:09.032509.032509 lmp.py:1510] ---- decode step 1 layer 14 ----
DEBUG 05-06 10:01:09.037338.037338 cuda_h.py:27] end decode_layer cost 5.171 ms
DEBUG 05-06 10:01:09.037373.037373 lmp.py:1510] ---- decode step 1 layer 15 ----
DEBUG 05-06 10:01:09.042180.042180 cuda_h.py:27] end decode_layer cost 4.906 ms
DEBUG 05-06 10:01:09.042023.042023 lmp.py:1510] ---- decode step 1 layer 16 ----
DEBUG 05-06 10:01:09.047592.047592 cuda_h.py:27] end decode_layer cost 4.733 ms
DEBUG 05-06 10:01:09.047197.047197 lmp.py:1510] ---- decode step 1 layer 17 ----
DEBUG 05-06 10:01:09.052169.052169 cuda_h.py:27] end decode_layer cost 5.100 ms
DEBUG 05-06 10:01:09.052158.052158 lmp.py:1510] ---- decode step 1 layer 18 ----
DEBUG 05-06 10:01:09.057379.057379 cuda_h.py:27] end decode_layer cost 4.827 ms
DEBUG 05-06 10:01:09.057415.057415 lmp.py:1510] ---- decode step 1 layer 19 ----
DEBUG 05-06 10:01:09.062686.062686 cuda_h.py:27] end decode_layer cost 4.724 ms
DEBUG 05-06 10:01:09.062767.062767 lmp.py:1510] ---- decode step 1 layer 20 ----
DEBUG 05-06 10:01:09.066966.066966 cuda_h.py:27] end decode_layer cost 4.740 ms
DEBUG 05-06 10:01:09.066762.066762 lmp.py:1510] ---- decode step 1 layer 21 ----
DEBUG 05-06 10:01:09.071508.071508 cuda_h.py:27] end decode_layer cost 4.863 ms
DEBUG 05-06 10:01:09.071543.071543 lmp.py:1510] ---- decode step 1 layer 22 ----
DEBUG 05-06 10:01:09.076761.076761 cuda_h.py:27] end decode_layer cost 4.709 ms
DEBUG 05-06 10:01:09.076750.076750 lmp.py:1510] ---- decode step 1 layer 23 ----
DEBUG 05-06 10:01:09.081016.081016 cuda_h.py:27] end decode_layer cost 4.965 ms
DEBUG 05-06 10:01:09.081336.081336 lmp.py:1510] ---- decode step 1 layer 24 ----
DEBUG 05-06 10:01:09.086765.086765 cuda_h.py:27] end decode_layer cost 4.699 ms
DEBUG 05-06 10:01:09.086138.086138 lmp.py:1510] ---- decode step 1 layer 25 ----
DEBUG 05-06 10:01:09.091054.091054 cuda_h.py:27] end decode_layer cost 4.814 ms
DEBUG 05-06 10:01:09.091328.091328 lmp.py:1510] ---- decode step 1 layer 26 ----
DEBUG 05-06 10:01:09.096468.096468 cuda_h.py:27] end decode_layer cost 4.767 ms
DEBUG 05-06 10:01:09.096410.096410 lmp.py:1510] ---- decode step 1 layer 27 ----
DEBUG 05-06 10:01:09.101496.101496 cuda_h.py:27] end decode_layer cost 4.938 ms
DEBUG 05-06 10:01:09.101531.101531 lmp.py:1510] ---- decode step 1 layer 28 ----
DEBUG 05-06 10:01:09.105790.105790 cuda_h.py:27] end decode_layer cost 4.750 ms
DEBUG 05-06 10:01:09.105110.105110 lmp.py:1510] ---- decode step 1 layer 29 ----
DEBUG 05-06 10:01:09.110520.110520 cuda_h.py:27] end decode_layer cost 4.932 ms
DEBUG 05-06 10:01:09.110735.110735 cuda_h.py:27] end decode_step cost 155.252 ms
INFO 05-06 10:01:09.111544.111544 lmp.py:1558] decode step 1 time: 0.1552891731262207 seconds
WARNING 05-06 10:01:09.111933.111933 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:09.111688.111688 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:09.111118.111118 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:09.112691.112691 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:09.117298.117298 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:09.117674.117674 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:09.117735.117735 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:09.118460.118460 cuda_h.py:27] end init_inputs_tokens cost 7.920 ms
DEBUG 05-06 10:01:09.119588.119588 lmp.py:1504] decode step 2 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:09.119351.119351 lmp.py:1510] ---- decode step 2 layer 0 ----
DEBUG 05-06 10:01:09.123251.123251 cuda_h.py:27] end decode_layer cost 4.730 ms
DEBUG 05-06 10:01:09.123810.123810 lmp.py:1510] ---- decode step 2 layer 1 ----
DEBUG 05-06 10:01:09.128917.128917 cuda_h.py:27] end decode_layer cost 4.778 ms
DEBUG 05-06 10:01:09.128475.128475 lmp.py:1510] ---- decode step 2 layer 2 ----
DEBUG 05-06 10:01:09.133313.133313 cuda_h.py:27] end decode_layer cost 4.651 ms
DEBUG 05-06 10:01:09.133156.133156 lmp.py:1510] ---- decode step 2 layer 3 ----
DEBUG 05-06 10:01:09.138386.138386 cuda_h.py:27] end decode_layer cost 4.693 ms
DEBUG 05-06 10:01:09.138945.138945 lmp.py:1510] ---- decode step 2 layer 4 ----
DEBUG 05-06 10:01:09.142972.142972 cuda_h.py:27] end decode_layer cost 4.755 ms
DEBUG 05-06 10:01:09.143007.143007 lmp.py:1510] ---- decode step 2 layer 5 ----
DEBUG 05-06 10:01:09.148824.148824 cuda_h.py:27] end decode_layer cost 5.021 ms
DEBUG 05-06 10:01:09.148005.148005 lmp.py:1510] ---- decode step 2 layer 6 ----
DEBUG 05-06 10:01:09.152506.152506 cuda_h.py:27] end decode_layer cost 4.857 ms
DEBUG 05-06 10:01:09.153779.153779 lmp.py:1510] ---- decode step 2 layer 7 ----
DEBUG 05-06 10:01:09.157867.157867 cuda_h.py:27] end decode_layer cost 4.800 ms
DEBUG 05-06 10:01:09.157187.157187 lmp.py:1510] ---- decode step 2 layer 8 ----
DEBUG 05-06 10:01:09.162007.162007 cuda_h.py:27] end decode_layer cost 4.707 ms
DEBUG 05-06 10:01:09.162931.162931 lmp.py:1510] ---- decode step 2 layer 9 ----
DEBUG 05-06 10:01:09.167411.167411 cuda_h.py:27] end decode_layer cost 4.843 ms
DEBUG 05-06 10:01:09.167883.167883 lmp.py:1510] ---- decode step 2 layer 10 ----
DEBUG 05-06 10:01:09.172681.172681 cuda_h.py:27] end decode_layer cost 4.831 ms
DEBUG 05-06 10:01:09.172862.172862 lmp.py:1510] ---- decode step 2 layer 11 ----
DEBUG 05-06 10:01:09.177784.177784 cuda_h.py:27] end decode_layer cost 4.993 ms
DEBUG 05-06 10:01:09.177058.177058 lmp.py:1510] ---- decode step 2 layer 12 ----
DEBUG 05-06 10:01:09.182522.182522 cuda_h.py:27] end decode_layer cost 4.761 ms
DEBUG 05-06 10:01:09.182511.182511 lmp.py:1510] ---- decode step 2 layer 13 ----
DEBUG 05-06 10:01:09.187465.187465 cuda_h.py:27] end decode_layer cost 4.771 ms
DEBUG 05-06 10:01:09.187785.187785 lmp.py:1510] ---- decode step 2 layer 14 ----
DEBUG 05-06 10:01:09.192788.192788 cuda_h.py:27] end decode_layer cost 4.806 ms
DEBUG 05-06 10:01:09.192730.192730 lmp.py:1510] ---- decode step 2 layer 15 ----
DEBUG 05-06 10:01:09.196674.196674 cuda_h.py:27] end decode_layer cost 4.834 ms
DEBUG 05-06 10:01:09.197232.197232 lmp.py:1510] ---- decode step 2 layer 16 ----
DEBUG 05-06 10:01:09.201286.201286 cuda_h.py:27] end decode_layer cost 4.774 ms
DEBUG 05-06 10:01:09.201129.201129 lmp.py:1510] ---- decode step 2 layer 17 ----
DEBUG 05-06 10:01:09.206216.206216 cuda_h.py:27] end decode_layer cost 4.974 ms
DEBUG 05-06 10:01:09.206775.206775 lmp.py:1510] ---- decode step 2 layer 18 ----
DEBUG 05-06 10:01:09.212588.212588 cuda_h.py:27] end decode_layer cost 5.685 ms
DEBUG 05-06 10:01:09.212868.212868 lmp.py:1510] ---- decode step 2 layer 19 ----
DEBUG 05-06 10:01:09.217937.217937 cuda_h.py:27] end decode_layer cost 5.031 ms
DEBUG 05-06 10:01:09.217741.217741 lmp.py:1510] ---- decode step 2 layer 20 ----
DEBUG 05-06 10:01:09.222817.222817 cuda_h.py:27] end decode_layer cost 4.825 ms
DEBUG 05-06 10:01:09.222421.222421 lmp.py:1510] ---- decode step 2 layer 21 ----
DEBUG 05-06 10:01:09.227141.227141 cuda_h.py:27] end decode_layer cost 4.879 ms
DEBUG 05-06 10:01:09.227699.227699 lmp.py:1510] ---- decode step 2 layer 22 ----
DEBUG 05-06 10:01:09.232468.232468 cuda_h.py:27] end decode_layer cost 4.739 ms
DEBUG 05-06 10:01:09.232933.232933 lmp.py:1510] ---- decode step 2 layer 23 ----
DEBUG 05-06 10:01:09.237935.237935 cuda_h.py:27] end decode_layer cost 4.981 ms
DEBUG 05-06 10:01:09.237255.237255 lmp.py:1510] ---- decode step 2 layer 24 ----
DEBUG 05-06 10:01:09.242990.242990 cuda_h.py:27] end decode_layer cost 4.750 ms
DEBUG 05-06 10:01:09.242787.242787 lmp.py:1510] ---- decode step 2 layer 25 ----
DEBUG 05-06 10:01:09.247992.247992 cuda_h.py:27] end decode_layer cost 4.745 ms
DEBUG 05-06 10:01:09.247743.247743 lmp.py:1510] ---- decode step 2 layer 26 ----
DEBUG 05-06 10:01:09.251249.251249 cuda_h.py:27] end decode_layer cost 4.826 ms
DEBUG 05-06 10:01:09.251575.251575 lmp.py:1510] ---- decode step 2 layer 27 ----
DEBUG 05-06 10:01:09.256797.256797 cuda_h.py:27] end decode_layer cost 4.828 ms
DEBUG 05-06 10:01:09.256309.256309 lmp.py:1510] ---- decode step 2 layer 28 ----
DEBUG 05-06 10:01:09.261055.261055 cuda_h.py:27] end decode_layer cost 4.688 ms
DEBUG 05-06 10:01:09.261759.261759 lmp.py:1510] ---- decode step 2 layer 29 ----
DEBUG 05-06 10:01:09.266901.266901 cuda_h.py:27] end decode_layer cost 5.014 ms
DEBUG 05-06 10:01:09.266871.266871 cuda_h.py:27] end decode_step cost 155.664 ms
INFO 05-06 10:01:09.266726.266726 lmp.py:1558] decode step 2 time: 0.1557002067565918 seconds
WARNING 05-06 10:01:09.266764.266764 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:09.267321.267321 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:09.267014.267014 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:09.267872.267872 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:09.272997.272997 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:09.272921.272921 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:09.273121.273121 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:09.274166.274166 cuda_h.py:27] end init_inputs_tokens cost 7.534 ms
DEBUG 05-06 10:01:09.274055.274055 lmp.py:1504] decode step 3 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:09.274487.274487 lmp.py:1510] ---- decode step 3 layer 0 ----
DEBUG 05-06 10:01:09.279725.279725 cuda_h.py:27] end decode_layer cost 4.733 ms
DEBUG 05-06 10:01:09.279807.279807 lmp.py:1510] ---- decode step 3 layer 1 ----
DEBUG 05-06 10:01:09.283932.283932 cuda_h.py:27] end decode_layer cost 4.721 ms
DEBUG 05-06 10:01:09.284013.284013 lmp.py:1510] ---- decode step 3 layer 2 ----
DEBUG 05-06 10:01:09.288889.288889 cuda_h.py:27] end decode_layer cost 4.772 ms
DEBUG 05-06 10:01:09.288970.288970 lmp.py:1510] ---- decode step 3 layer 3 ----
DEBUG 05-06 10:01:09.293594.293594 cuda_h.py:27] end decode_layer cost 4.773 ms
DEBUG 05-06 10:01:09.293629.293629 lmp.py:1510] ---- decode step 3 layer 4 ----
DEBUG 05-06 10:01:09.298676.298676 cuda_h.py:27] end decode_layer cost 4.769 ms
DEBUG 05-06 10:01:09.298473.298473 lmp.py:1510] ---- decode step 3 layer 5 ----
DEBUG 05-06 10:01:09.303633.303633 cuda_h.py:27] end decode_layer cost 4.957 ms
DEBUG 05-06 10:01:09.303714.303714 lmp.py:1510] ---- decode step 3 layer 6 ----
DEBUG 05-06 10:01:09.308801.308801 cuda_h.py:27] end decode_layer cost 4.763 ms
DEBUG 05-06 10:01:09.308028.308028 lmp.py:1510] ---- decode step 3 layer 7 ----
DEBUG 05-06 10:01:09.313706.313706 cuda_h.py:27] end decode_layer cost 4.813 ms
DEBUG 05-06 10:01:09.313264.313264 lmp.py:1510] ---- decode step 3 layer 8 ----
DEBUG 05-06 10:01:09.318010.318010 cuda_h.py:27] end decode_layer cost 4.862 ms
DEBUG 05-06 10:01:09.318205.318205 lmp.py:1510] ---- decode step 3 layer 9 ----
DEBUG 05-06 10:01:09.323981.323981 cuda_h.py:27] end decode_layer cost 4.780 ms
DEBUG 05-06 10:01:09.323824.323824 lmp.py:1510] ---- decode step 3 layer 10 ----
DEBUG 05-06 10:01:09.327856.327856 cuda_h.py:27] end decode_layer cost 4.723 ms
DEBUG 05-06 10:01:09.327084.327084 lmp.py:1510] ---- decode step 3 layer 11 ----
DEBUG 05-06 10:01:09.332635.332635 cuda_h.py:27] end decode_layer cost 5.001 ms
DEBUG 05-06 10:01:09.332624.332624 lmp.py:1510] ---- decode step 3 layer 12 ----
DEBUG 05-06 10:01:09.337601.337601 cuda_h.py:27] end decode_layer cost 4.647 ms
DEBUG 05-06 10:01:09.337683.337683 lmp.py:1510] ---- decode step 3 layer 13 ----
DEBUG 05-06 10:01:09.342393.342393 cuda_h.py:27] end decode_layer cost 4.801 ms
DEBUG 05-06 10:01:09.342336.342336 lmp.py:1510] ---- decode step 3 layer 14 ----
DEBUG 05-06 10:01:09.347074.347074 cuda_h.py:27] end decode_layer cost 4.822 ms
DEBUG 05-06 10:01:09.347539.347539 lmp.py:1510] ---- decode step 3 layer 15 ----
DEBUG 05-06 10:01:09.352238.352238 cuda_h.py:27] end decode_layer cost 4.828 ms
DEBUG 05-06 10:01:09.352180.352180 lmp.py:1510] ---- decode step 3 layer 16 ----
DEBUG 05-06 10:01:09.357812.357812 cuda_h.py:27] end decode_layer cost 4.814 ms
DEBUG 05-06 10:01:09.357893.357893 lmp.py:1510] ---- decode step 3 layer 17 ----
DEBUG 05-06 10:01:09.362672.362672 cuda_h.py:27] end decode_layer cost 5.063 ms
DEBUG 05-06 10:01:09.362469.362469 lmp.py:1510] ---- decode step 3 layer 18 ----
DEBUG 05-06 10:01:09.367680.367680 cuda_h.py:27] end decode_layer cost 4.715 ms
DEBUG 05-06 10:01:09.367000.367000 lmp.py:1510] ---- decode step 3 layer 19 ----
DEBUG 05-06 10:01:09.372217.372217 cuda_h.py:27] end decode_layer cost 4.894 ms
DEBUG 05-06 10:01:09.372206.372206 lmp.py:1510] ---- decode step 3 layer 20 ----
DEBUG 05-06 10:01:09.376879.376879 cuda_h.py:27] end decode_layer cost 4.669 ms
DEBUG 05-06 10:01:09.376483.376483 lmp.py:1510] ---- decode step 3 layer 21 ----
DEBUG 05-06 10:01:09.381404.381404 cuda_h.py:27] end decode_layer cost 4.746 ms
DEBUG 05-06 10:01:09.381247.381247 lmp.py:1510] ---- decode step 3 layer 22 ----
DEBUG 05-06 10:01:09.386400.386400 cuda_h.py:27] end decode_layer cost 4.952 ms
DEBUG 05-06 10:01:09.386634.386634 lmp.py:1510] ---- decode step 3 layer 23 ----
DEBUG 05-06 10:01:09.391827.391827 cuda_h.py:27] end decode_layer cost 5.157 ms
DEBUG 05-06 10:01:09.391677.391677 lmp.py:1510] ---- decode step 3 layer 24 ----
DEBUG 05-06 10:01:09.396226.396226 cuda_h.py:27] end decode_layer cost 4.717 ms
DEBUG 05-06 10:01:09.396453.396453 lmp.py:1510] ---- decode step 3 layer 25 ----
DEBUG 05-06 10:01:09.401376.401376 cuda_h.py:27] end decode_layer cost 4.819 ms
DEBUG 05-06 10:01:09.401557.401557 lmp.py:1510] ---- decode step 3 layer 26 ----
DEBUG 05-06 10:01:09.406093.406093 cuda_h.py:27] end decode_layer cost 4.744 ms
DEBUG 05-06 10:01:09.406652.406652 lmp.py:1510] ---- decode step 3 layer 27 ----
DEBUG 05-06 10:01:09.411342.411342 cuda_h.py:27] end decode_layer cost 4.787 ms
DEBUG 05-06 10:01:09.411569.411569 lmp.py:1510] ---- decode step 3 layer 28 ----
DEBUG 05-06 10:01:09.416465.416465 cuda_h.py:27] end decode_layer cost 4.798 ms
DEBUG 05-06 10:01:09.416977.416977 lmp.py:1510] ---- decode step 3 layer 29 ----
DEBUG 05-06 10:01:09.421625.421625 cuda_h.py:27] end decode_layer cost 5.107 ms
DEBUG 05-06 10:01:09.421072.421072 cuda_h.py:27] end decode_step cost 154.450 ms
INFO 05-06 10:01:09.421927.421927 lmp.py:1558] decode step 3 time: 0.15448594093322754 seconds
WARNING 05-06 10:01:09.421442.421442 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:09.421924.421924 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:09.422726.422726 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:09.422292.422292 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:09.427844.427844 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:09.427331.427331 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:09.427816.427816 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:09.428472.428472 cuda_h.py:27] end init_inputs_tokens cost 7.506 ms
DEBUG 05-06 10:01:09.428361.428361 lmp.py:1504] decode step 4 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:09.428886.428886 lmp.py:1510] ---- decode step 4 layer 0 ----
DEBUG 05-06 10:01:09.433417.433417 cuda_h.py:27] end decode_layer cost 4.985 ms
DEBUG 05-06 10:01:09.434737.434737 lmp.py:1510] ---- decode step 4 layer 1 ----
DEBUG 05-06 10:01:09.438254.438254 cuda_h.py:27] end decode_layer cost 4.765 ms
DEBUG 05-06 10:01:09.438620.438620 lmp.py:1510] ---- decode step 4 layer 2 ----
DEBUG 05-06 10:01:09.443381.443381 cuda_h.py:27] end decode_layer cost 4.698 ms
DEBUG 05-06 10:01:09.443946.443946 lmp.py:1510] ---- decode step 4 layer 3 ----
DEBUG 05-06 10:01:09.448189.448189 cuda_h.py:27] end decode_layer cost 4.709 ms
DEBUG 05-06 10:01:09.448509.448509 lmp.py:1510] ---- decode step 4 layer 4 ----
DEBUG 05-06 10:01:09.453344.453344 cuda_h.py:27] end decode_layer cost 4.753 ms
DEBUG 05-06 10:01:09.453141.453141 lmp.py:1510] ---- decode step 4 layer 5 ----
DEBUG 05-06 10:01:09.458600.458600 cuda_h.py:27] end decode_layer cost 5.003 ms
DEBUG 05-06 10:01:09.458158.458158 lmp.py:1510] ---- decode step 4 layer 6 ----
DEBUG 05-06 10:01:09.463903.463903 cuda_h.py:27] end decode_layer cost 4.827 ms
DEBUG 05-06 10:01:09.463700.463700 lmp.py:1510] ---- decode step 4 layer 7 ----
DEBUG 05-06 10:01:09.467972.467972 cuda_h.py:27] end decode_layer cost 4.759 ms
DEBUG 05-06 10:01:09.467530.467530 lmp.py:1510] ---- decode step 4 layer 8 ----
DEBUG 05-06 10:01:09.472803.472803 cuda_h.py:27] end decode_layer cost 4.795 ms
DEBUG 05-06 10:01:09.472077.472077 lmp.py:1510] ---- decode step 4 layer 9 ----
DEBUG 05-06 10:01:09.477997.477997 cuda_h.py:27] end decode_layer cost 4.710 ms
DEBUG 05-06 10:01:09.477555.477555 lmp.py:1510] ---- decode step 4 layer 10 ----
DEBUG 05-06 10:01:09.482574.482574 cuda_h.py:27] end decode_layer cost 4.713 ms
DEBUG 05-06 10:01:09.482371.482371 lmp.py:1510] ---- decode step 4 layer 11 ----
DEBUG 05-06 10:01:09.487713.487713 cuda_h.py:27] end decode_layer cost 5.092 ms
DEBUG 05-06 10:01:09.487232.487232 lmp.py:1510] ---- decode step 4 layer 12 ----
DEBUG 05-06 10:01:09.492766.492766 cuda_h.py:27] end decode_layer cost 4.847 ms
DEBUG 05-06 10:01:09.492801.492801 lmp.py:1510] ---- decode step 4 layer 13 ----
DEBUG 05-06 10:01:09.497282.497282 cuda_h.py:27] end decode_layer cost 4.878 ms
DEBUG 05-06 10:01:09.497556.497556 lmp.py:1510] ---- decode step 4 layer 14 ----
DEBUG 05-06 10:01:09.502041.502041 cuda_h.py:27] end decode_layer cost 4.811 ms
DEBUG 05-06 10:01:09.502077.502077 lmp.py:1510] ---- decode step 4 layer 15 ----
DEBUG 05-06 10:01:09.507375.507375 cuda_h.py:27] end decode_layer cost 4.743 ms
DEBUG 05-06 10:01:09.507695.507695 lmp.py:1510] ---- decode step 4 layer 16 ----
DEBUG 05-06 10:01:09.511550.511550 cuda_h.py:27] end decode_layer cost 4.767 ms
DEBUG 05-06 10:01:09.511016.511016 lmp.py:1510] ---- decode step 4 layer 17 ----
DEBUG 05-06 10:01:09.517413.517413 cuda_h.py:27] end decode_layer cost 5.133 ms
DEBUG 05-06 10:01:09.517117.517117 lmp.py:1510] ---- decode step 4 layer 18 ----
DEBUG 05-06 10:01:09.521075.521075 cuda_h.py:27] end decode_layer cost 4.668 ms
DEBUG 05-06 10:01:09.521395.521395 lmp.py:1510] ---- decode step 4 layer 19 ----
DEBUG 05-06 10:01:09.526678.526678 cuda_h.py:27] end decode_layer cost 4.697 ms
DEBUG 05-06 10:01:09.526759.526759 lmp.py:1510] ---- decode step 4 layer 20 ----
DEBUG 05-06 10:01:09.531361.531361 cuda_h.py:27] end decode_layer cost 4.722 ms
DEBUG 05-06 10:01:09.531681.531681 lmp.py:1510] ---- decode step 4 layer 21 ----
DEBUG 05-06 10:01:09.536389.536389 cuda_h.py:27] end decode_layer cost 4.730 ms
DEBUG 05-06 10:01:09.536233.536233 lmp.py:1510] ---- decode step 4 layer 22 ----
DEBUG 05-06 10:01:09.540821.540821 cuda_h.py:27] end decode_layer cost 4.712 ms
DEBUG 05-06 10:01:09.541141.541141 lmp.py:1510] ---- decode step 4 layer 23 ----
DEBUG 05-06 10:01:09.546725.546725 cuda_h.py:27] end decode_layer cost 4.990 ms
DEBUG 05-06 10:01:09.546522.546522 lmp.py:1510] ---- decode step 4 layer 24 ----
DEBUG 05-06 10:01:09.550489.550489 cuda_h.py:27] end decode_layer cost 4.745 ms
DEBUG 05-06 10:01:09.550716.550716 lmp.py:1510] ---- decode step 4 layer 25 ----
DEBUG 05-06 10:01:09.555694.555694 cuda_h.py:27] end decode_layer cost 4.858 ms
DEBUG 05-06 10:01:09.555159.555159 lmp.py:1510] ---- decode step 4 layer 26 ----
DEBUG 05-06 10:01:09.561414.561414 cuda_h.py:27] end decode_layer cost 5.414 ms
DEBUG 05-06 10:01:09.561879.561879 lmp.py:1510] ---- decode step 4 layer 27 ----
DEBUG 05-06 10:01:09.566307.566307 cuda_h.py:27] end decode_layer cost 4.839 ms
DEBUG 05-06 10:01:09.566626.566626 lmp.py:1510] ---- decode step 4 layer 28 ----
DEBUG 05-06 10:01:09.570448.570448 cuda_h.py:27] end decode_layer cost 4.743 ms
DEBUG 05-06 10:01:09.571960.571960 lmp.py:1510] ---- decode step 4 layer 29 ----
DEBUG 05-06 10:01:09.576691.576691 cuda_h.py:27] end decode_layer cost 5.028 ms
DEBUG 05-06 10:01:09.576615.576615 cuda_h.py:27] end decode_step cost 154.772 ms
INFO 05-06 10:01:09.576708.576708 lmp.py:1558] decode step 4 time: 0.15480804443359375 seconds
WARNING 05-06 10:01:09.576938.576938 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:09.576307.576307 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:09.577367.577367 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:09.577364.577364 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:09.582454.582454 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:09.582372.582372 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:09.582619.582619 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:09.583591.583591 cuda_h.py:27] end init_inputs_tokens cost 7.508 ms
DEBUG 05-06 10:01:09.583626.583626 lmp.py:1504] decode step 5 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:09.583343.583343 lmp.py:1510] ---- decode step 5 layer 0 ----
DEBUG 05-06 10:01:09.588753.588753 cuda_h.py:27] end decode_layer cost 5.141 ms
DEBUG 05-06 10:01:09.589126.589126 lmp.py:1510] ---- decode step 5 layer 1 ----
DEBUG 05-06 10:01:09.593712.593712 cuda_h.py:27] end decode_layer cost 4.815 ms
DEBUG 05-06 10:01:09.593270.593270 lmp.py:1510] ---- decode step 5 layer 2 ----
DEBUG 05-06 10:01:09.598812.598812 cuda_h.py:27] end decode_layer cost 4.713 ms
DEBUG 05-06 10:01:09.598609.598609 lmp.py:1510] ---- decode step 5 layer 3 ----
DEBUG 05-06 10:01:09.603103.603103 cuda_h.py:27] end decode_layer cost 4.853 ms
DEBUG 05-06 10:01:09.603337.603337 lmp.py:1510] ---- decode step 5 layer 4 ----
DEBUG 05-06 10:01:09.608199.608199 cuda_h.py:27] end decode_layer cost 4.773 ms
DEBUG 05-06 10:01:09.608234.608234 lmp.py:1510] ---- decode step 5 layer 5 ----
DEBUG 05-06 10:01:09.613746.613746 cuda_h.py:27] end decode_layer cost 5.007 ms
DEBUG 05-06 10:01:09.613351.613351 lmp.py:1510] ---- decode step 5 layer 6 ----
DEBUG 05-06 10:01:09.618647.618647 cuda_h.py:27] end decode_layer cost 4.882 ms
DEBUG 05-06 10:01:09.618920.618920 lmp.py:1510] ---- decode step 5 layer 7 ----
DEBUG 05-06 10:01:09.623698.623698 cuda_h.py:27] end decode_layer cost 4.816 ms
DEBUG 05-06 10:01:09.623448.623448 lmp.py:1510] ---- decode step 5 layer 8 ----
DEBUG 05-06 10:01:09.628272.628272 cuda_h.py:27] end decode_layer cost 4.815 ms
DEBUG 05-06 10:01:09.628181.628181 lmp.py:1510] ---- decode step 5 layer 9 ----
DEBUG 05-06 10:01:09.633429.633429 cuda_h.py:27] end decode_layer cost 4.812 ms
DEBUG 05-06 10:01:09.633226.633226 lmp.py:1510] ---- decode step 5 layer 10 ----
DEBUG 05-06 10:01:09.637491.637491 cuda_h.py:27] end decode_layer cost 4.754 ms
DEBUG 05-06 10:01:09.637572.637572 lmp.py:1510] ---- decode step 5 layer 11 ----
DEBUG 05-06 10:01:09.643869.643869 cuda_h.py:27] end decode_layer cost 5.093 ms
DEBUG 05-06 10:01:09.643904.643904 lmp.py:1510] ---- decode step 5 layer 12 ----
DEBUG 05-06 10:01:09.648765.648765 cuda_h.py:27] end decode_layer cost 4.947 ms
DEBUG 05-06 10:01:09.648469.648469 lmp.py:1510] ---- decode step 5 layer 13 ----
DEBUG 05-06 10:01:09.653599.653599 cuda_h.py:27] end decode_layer cost 4.865 ms
DEBUG 05-06 10:01:09.653396.653396 lmp.py:1510] ---- decode step 5 layer 14 ----
DEBUG 05-06 10:01:09.657282.657282 cuda_h.py:27] end decode_layer cost 4.896 ms
DEBUG 05-06 10:01:09.658364.658364 lmp.py:1510] ---- decode step 5 layer 15 ----
DEBUG 05-06 10:01:09.662867.662867 cuda_h.py:27] end decode_layer cost 4.755 ms
DEBUG 05-06 10:01:09.662903.662903 lmp.py:1510] ---- decode step 5 layer 16 ----
DEBUG 05-06 10:01:09.667493.667493 cuda_h.py:27] end decode_layer cost 4.784 ms
DEBUG 05-06 10:01:09.667098.667098 lmp.py:1510] ---- decode step 5 layer 17 ----
DEBUG 05-06 10:01:09.672930.672930 cuda_h.py:27] end decode_layer cost 5.066 ms
DEBUG 05-06 10:01:09.672204.672204 lmp.py:1510] ---- decode step 5 layer 18 ----
DEBUG 05-06 10:01:09.677449.677449 cuda_h.py:27] end decode_layer cost 4.740 ms
DEBUG 05-06 10:01:09.677199.677199 lmp.py:1510] ---- decode step 5 layer 19 ----
DEBUG 05-06 10:01:09.682943.682943 cuda_h.py:27] end decode_layer cost 4.791 ms
DEBUG 05-06 10:01:09.682024.682024 lmp.py:1510] ---- decode step 5 layer 20 ----
DEBUG 05-06 10:01:09.687170.687170 cuda_h.py:27] end decode_layer cost 4.736 ms
DEBUG 05-06 10:01:09.687205.687205 lmp.py:1510] ---- decode step 5 layer 21 ----
DEBUG 05-06 10:01:09.692197.692197 cuda_h.py:27] end decode_layer cost 4.904 ms
DEBUG 05-06 10:01:09.692994.692994 lmp.py:1510] ---- decode step 5 layer 22 ----
DEBUG 05-06 10:01:09.697722.697722 cuda_h.py:27] end decode_layer cost 4.745 ms
DEBUG 05-06 10:01:09.697042.697042 lmp.py:1510] ---- decode step 5 layer 23 ----
DEBUG 05-06 10:01:09.702222.702222 cuda_h.py:27] end decode_layer cost 4.972 ms
DEBUG 05-06 10:01:09.702827.702827 lmp.py:1510] ---- decode step 5 layer 24 ----
DEBUG 05-06 10:01:09.706533.706533 cuda_h.py:27] end decode_layer cost 4.658 ms
DEBUG 05-06 10:01:09.706614.706614 lmp.py:1510] ---- decode step 5 layer 25 ----
DEBUG 05-06 10:01:09.711064.711064 cuda_h.py:27] end decode_layer cost 4.714 ms
DEBUG 05-06 10:01:09.711272.711272 lmp.py:1510] ---- decode step 5 layer 26 ----
DEBUG 05-06 10:01:09.716291.716291 cuda_h.py:27] end decode_layer cost 4.924 ms
DEBUG 05-06 10:01:09.716995.716995 lmp.py:1510] ---- decode step 5 layer 27 ----
DEBUG 05-06 10:01:09.721699.721699 cuda_h.py:27] end decode_layer cost 4.797 ms
DEBUG 05-06 10:01:09.721019.721019 lmp.py:1510] ---- decode step 5 layer 28 ----
DEBUG 05-06 10:01:09.726932.726932 cuda_h.py:27] end decode_layer cost 4.705 ms
DEBUG 05-06 10:01:09.726251.726251 lmp.py:1510] ---- decode step 5 layer 29 ----
DEBUG 05-06 10:01:09.731933.731933 cuda_h.py:27] end decode_layer cost 4.921 ms
DEBUG 05-06 10:01:09.731810.731810 cuda_h.py:27] end decode_step cost 155.076 ms
INFO 05-06 10:01:09.731712.731712 lmp.py:1558] decode step 5 time: 0.15511226654052734 seconds
WARNING 05-06 10:01:09.731465.731465 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:09.731320.731320 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:09.732850.732850 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:09.732339.732339 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:09.737726.737726 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:09.737227.737227 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:09.737711.737711 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:09.738300.738300 cuda_h.py:27] end init_inputs_tokens cost 7.608 ms
DEBUG 05-06 10:01:09.739143.739143 lmp.py:1504] decode step 6 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:09.739382.739382 lmp.py:1510] ---- decode step 6 layer 0 ----
DEBUG 05-06 10:01:09.743585.743585 cuda_h.py:27] end decode_layer cost 4.847 ms
DEBUG 05-06 10:01:09.743050.743050 lmp.py:1510] ---- decode step 6 layer 1 ----
DEBUG 05-06 10:01:09.748642.748642 cuda_h.py:27] end decode_layer cost 4.820 ms
DEBUG 05-06 10:01:09.748485.748485 lmp.py:1510] ---- decode step 6 layer 2 ----
DEBUG 05-06 10:01:09.753207.753207 cuda_h.py:27] end decode_layer cost 4.740 ms
DEBUG 05-06 10:01:09.753097.753097 lmp.py:1510] ---- decode step 6 layer 3 ----
DEBUG 05-06 10:01:09.758766.758766 cuda_h.py:27] end decode_layer cost 4.772 ms
DEBUG 05-06 10:01:09.758133.758133 lmp.py:1510] ---- decode step 6 layer 4 ----
DEBUG 05-06 10:01:09.763893.763893 cuda_h.py:27] end decode_layer cost 4.698 ms
DEBUG 05-06 10:01:09.763974.763974 lmp.py:1510] ---- decode step 6 layer 5 ----
DEBUG 05-06 10:01:09.768117.768117 cuda_h.py:27] end decode_layer cost 5.050 ms
DEBUG 05-06 10:01:09.768344.768344 lmp.py:1510] ---- decode step 6 layer 6 ----
DEBUG 05-06 10:01:09.773419.773419 cuda_h.py:27] end decode_layer cost 4.789 ms
DEBUG 05-06 10:01:09.773500.773500 lmp.py:1510] ---- decode step 6 layer 7 ----
DEBUG 05-06 10:01:09.777685.777685 cuda_h.py:27] end decode_layer cost 4.730 ms
DEBUG 05-06 10:01:09.778767.778767 lmp.py:1510] ---- decode step 6 layer 8 ----
DEBUG 05-06 10:01:09.782234.782234 cuda_h.py:27] end decode_layer cost 4.657 ms
DEBUG 05-06 10:01:09.782461.782461 lmp.py:1510] ---- decode step 6 layer 9 ----
DEBUG 05-06 10:01:09.787229.787229 cuda_h.py:27] end decode_layer cost 4.914 ms
DEBUG 05-06 10:01:09.787695.787695 lmp.py:1510] ---- decode step 6 layer 10 ----
DEBUG 05-06 10:01:09.792664.792664 cuda_h.py:27] end decode_layer cost 4.817 ms
DEBUG 05-06 10:01:09.792130.792130 lmp.py:1510] ---- decode step 6 layer 11 ----
DEBUG 05-06 10:01:09.797827.797827 cuda_h.py:27] end decode_layer cost 5.003 ms
DEBUG 05-06 10:01:09.797385.797385 lmp.py:1510] ---- decode step 6 layer 12 ----
DEBUG 05-06 10:01:09.802715.802715 cuda_h.py:27] end decode_layer cost 4.697 ms
DEBUG 05-06 10:01:09.802035.802035 lmp.py:1510] ---- decode step 6 layer 13 ----
DEBUG 05-06 10:01:09.807016.807016 cuda_h.py:27] end decode_layer cost 4.755 ms
DEBUG 05-06 10:01:09.807812.807812 lmp.py:1510] ---- decode step 6 layer 14 ----
DEBUG 05-06 10:01:09.812902.812902 cuda_h.py:27] end decode_layer cost 4.835 ms
DEBUG 05-06 10:01:09.812228.812228 lmp.py:1510] ---- decode step 6 layer 15 ----
DEBUG 05-06 10:01:09.817707.817707 cuda_h.py:27] end decode_layer cost 4.807 ms
DEBUG 05-06 10:01:09.817935.817935 lmp.py:1510] ---- decode step 6 layer 16 ----
DEBUG 05-06 10:01:09.821404.821404 cuda_h.py:27] end decode_layer cost 4.730 ms
DEBUG 05-06 10:01:09.821201.821201 lmp.py:1510] ---- decode step 6 layer 17 ----
DEBUG 05-06 10:01:09.826557.826557 cuda_h.py:27] end decode_layer cost 4.891 ms
DEBUG 05-06 10:01:09.826115.826115 lmp.py:1510] ---- decode step 6 layer 18 ----
DEBUG 05-06 10:01:09.831820.831820 cuda_h.py:27] end decode_layer cost 4.622 ms
DEBUG 05-06 10:01:09.831140.831140 lmp.py:1510] ---- decode step 6 layer 19 ----
DEBUG 05-06 10:01:09.836551.836551 cuda_h.py:27] end decode_layer cost 4.756 ms
DEBUG 05-06 10:01:09.836348.836348 lmp.py:1510] ---- decode step 6 layer 20 ----
DEBUG 05-06 10:01:09.841681.841681 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 10:01:09.841147.841147 lmp.py:1510] ---- decode step 6 layer 21 ----
DEBUG 05-06 10:01:09.845583.845583 cuda_h.py:27] end decode_layer cost 4.705 ms
DEBUG 05-06 10:01:09.845949.845949 lmp.py:1510] ---- decode step 6 layer 22 ----
DEBUG 05-06 10:01:09.850799.850799 cuda_h.py:27] end decode_layer cost 4.624 ms
DEBUG 05-06 10:01:09.850881.850881 lmp.py:1510] ---- decode step 6 layer 23 ----
DEBUG 05-06 10:01:09.855035.855035 cuda_h.py:27] end decode_layer cost 4.988 ms
DEBUG 05-06 10:01:09.855308.855308 lmp.py:1510] ---- decode step 6 layer 24 ----
DEBUG 05-06 10:01:09.860855.860855 cuda_h.py:27] end decode_layer cost 4.646 ms
DEBUG 05-06 10:01:09.860413.860413 lmp.py:1510] ---- decode step 6 layer 25 ----
DEBUG 05-06 10:01:09.865864.865864 cuda_h.py:27] end decode_layer cost 4.750 ms
DEBUG 05-06 10:01:09.865912.865912 lmp.py:1510] ---- decode step 6 layer 26 ----
DEBUG 05-06 10:01:09.869764.869764 cuda_h.py:27] end decode_layer cost 4.660 ms
DEBUG 05-06 10:01:09.870753.870753 lmp.py:1510] ---- decode step 6 layer 27 ----
DEBUG 05-06 10:01:09.874059.874059 cuda_h.py:27] end decode_layer cost 4.785 ms
DEBUG 05-06 10:01:09.874094.874094 lmp.py:1510] ---- decode step 6 layer 28 ----
DEBUG 05-06 10:01:09.879176.879176 cuda_h.py:27] end decode_layer cost 4.619 ms
DEBUG 05-06 10:01:09.879258.879258 lmp.py:1510] ---- decode step 6 layer 29 ----
DEBUG 05-06 10:01:09.884488.884488 cuda_h.py:27] end decode_layer cost 4.904 ms
DEBUG 05-06 10:01:09.884418.884418 cuda_h.py:27] end decode_step cost 153.171 ms
INFO 05-06 10:01:09.884512.884512 lmp.py:1558] decode step 6 time: 0.15320777893066406 seconds
WARNING 05-06 10:01:09.884795.884795 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:09.885474.885474 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:09.885857.885857 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:09.885331.885331 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:09.890733.890733 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:09.890127.890127 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:09.890089.890089 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:09.892929.892929 cuda_h.py:27] end init_inputs_tokens cost 7.447 ms
DEBUG 05-06 10:01:09.892441.892441 lmp.py:1504] decode step 7 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:09.892111.892111 lmp.py:1510] ---- decode step 7 layer 0 ----
DEBUG 05-06 10:01:09.897383.897383 cuda_h.py:27] end decode_layer cost 4.934 ms
DEBUG 05-06 10:01:09.897941.897941 lmp.py:1510] ---- decode step 7 layer 1 ----
DEBUG 05-06 10:01:09.901954.901954 cuda_h.py:27] end decode_layer cost 4.745 ms
DEBUG 05-06 10:01:09.901751.901751 lmp.py:1510] ---- decode step 7 layer 2 ----
DEBUG 05-06 10:01:09.906885.906885 cuda_h.py:27] end decode_layer cost 4.787 ms
DEBUG 05-06 10:01:09.906967.906967 lmp.py:1510] ---- decode step 7 layer 3 ----
DEBUG 05-06 10:01:09.912300.912300 cuda_h.py:27] end decode_layer cost 5.401 ms
DEBUG 05-06 10:01:09.912057.912057 lmp.py:1510] ---- decode step 7 layer 4 ----
DEBUG 05-06 10:01:09.917754.917754 cuda_h.py:27] end decode_layer cost 4.967 ms
DEBUG 05-06 10:01:09.917458.917458 lmp.py:1510] ---- decode step 7 layer 5 ----
DEBUG 05-06 10:01:09.922470.922470 cuda_h.py:27] end decode_layer cost 5.094 ms
DEBUG 05-06 10:01:09.922743.922743 lmp.py:1510] ---- decode step 7 layer 6 ----
DEBUG 05-06 10:01:09.927083.927083 cuda_h.py:27] end decode_layer cost 4.809 ms
DEBUG 05-06 10:01:09.927787.927787 lmp.py:1510] ---- decode step 7 layer 7 ----
DEBUG 05-06 10:01:09.932935.932935 cuda_h.py:27] end decode_layer cost 4.808 ms
DEBUG 05-06 10:01:09.932401.932401 lmp.py:1510] ---- decode step 7 layer 8 ----
DEBUG 05-06 10:01:09.937422.937422 cuda_h.py:27] end decode_layer cost 4.785 ms
DEBUG 05-06 10:01:09.937696.937696 lmp.py:1510] ---- decode step 7 layer 9 ----
DEBUG 05-06 10:01:09.941088.941088 cuda_h.py:27] end decode_layer cost 4.777 ms
DEBUG 05-06 10:01:09.941123.941123 lmp.py:1510] ---- decode step 7 layer 10 ----
DEBUG 05-06 10:01:09.946758.946758 cuda_h.py:27] end decode_layer cost 4.711 ms
DEBUG 05-06 10:01:09.946839.946839 lmp.py:1510] ---- decode step 7 layer 11 ----
DEBUG 05-06 10:01:09.951699.951699 cuda_h.py:27] end decode_layer cost 4.911 ms
DEBUG 05-06 10:01:09.951258.951258 lmp.py:1510] ---- decode step 7 layer 12 ----
DEBUG 05-06 10:01:09.956814.956814 cuda_h.py:27] end decode_layer cost 4.758 ms
DEBUG 05-06 10:01:09.956280.956280 lmp.py:1510] ---- decode step 7 layer 13 ----
DEBUG 05-06 10:01:09.961778.961778 cuda_h.py:27] end decode_layer cost 4.786 ms
DEBUG 05-06 10:01:09.961145.961145 lmp.py:1510] ---- decode step 7 layer 14 ----
DEBUG 05-06 10:01:09.966507.966507 cuda_h.py:27] end decode_layer cost 4.686 ms
DEBUG 05-06 10:01:09.966065.966065 lmp.py:1510] ---- decode step 7 layer 15 ----
DEBUG 05-06 10:01:09.970807.970807 cuda_h.py:27] end decode_layer cost 4.755 ms
DEBUG 05-06 10:01:09.970843.970843 lmp.py:1510] ---- decode step 7 layer 16 ----
DEBUG 05-06 10:01:09.975259.975259 cuda_h.py:27] end decode_layer cost 4.725 ms
DEBUG 05-06 10:01:09.975010.975010 lmp.py:1510] ---- decode step 7 layer 17 ----
DEBUG 05-06 10:01:09.980518.980518 cuda_h.py:27] end decode_layer cost 4.899 ms
DEBUG 05-06 10:01:09.980361.980361 lmp.py:1510] ---- decode step 7 layer 18 ----
DEBUG 05-06 10:01:09.985925.985925 cuda_h.py:27] end decode_layer cost 4.763 ms
DEBUG 05-06 10:01:09.985391.985391 lmp.py:1510] ---- decode step 7 layer 19 ----
DEBUG 05-06 10:01:09.990008.990008 cuda_h.py:27] end decode_layer cost 4.768 ms
DEBUG 05-06 10:01:09.990043.990043 lmp.py:1510] ---- decode step 7 layer 20 ----
DEBUG 05-06 10:01:09.995112.995112 cuda_h.py:27] end decode_layer cost 4.820 ms
DEBUG 05-06 10:01:09.995816.995816 lmp.py:1510] ---- decode step 7 layer 21 ----
DEBUG 05-06 10:01:10.000499.000499 cuda_h.py:27] end decode_layer cost 4.782 ms
DEBUG 05-06 10:01:10.000965.000965 lmp.py:1510] ---- decode step 7 layer 22 ----
DEBUG 05-06 10:01:10.004136.004136 cuda_h.py:27] end decode_layer cost 4.720 ms
DEBUG 05-06 10:01:10.004456.004456 lmp.py:1510] ---- decode step 7 layer 23 ----
DEBUG 05-06 10:01:10.009452.009452 cuda_h.py:27] end decode_layer cost 5.012 ms
DEBUG 05-06 10:01:10.010295.010295 lmp.py:1510] ---- decode step 7 layer 24 ----
DEBUG 05-06 10:01:10.014512.014512 cuda_h.py:27] end decode_layer cost 4.683 ms
DEBUG 05-06 10:01:10.014070.014070 lmp.py:1510] ---- decode step 7 layer 25 ----
DEBUG 05-06 10:01:10.019956.019956 cuda_h.py:27] end decode_layer cost 4.895 ms
DEBUG 05-06 10:01:10.019945.019945 lmp.py:1510] ---- decode step 7 layer 26 ----
DEBUG 05-06 10:01:10.024871.024871 cuda_h.py:27] end decode_layer cost 4.715 ms
DEBUG 05-06 10:01:10.024337.024337 lmp.py:1510] ---- decode step 7 layer 27 ----
DEBUG 05-06 10:01:10.029192.029192 cuda_h.py:27] end decode_layer cost 4.768 ms
DEBUG 05-06 10:01:10.029989.029989 lmp.py:1510] ---- decode step 7 layer 28 ----
DEBUG 05-06 10:01:10.034821.034821 cuda_h.py:27] end decode_layer cost 4.681 ms
DEBUG 05-06 10:01:10.034379.034379 lmp.py:1510] ---- decode step 7 layer 29 ----
DEBUG 05-06 10:01:10.039441.039441 cuda_h.py:27] end decode_layer cost 4.990 ms
DEBUG 05-06 10:01:10.039603.039603 cuda_h.py:27] end decode_step cost 154.542 ms
INFO 05-06 10:01:10.039220.039220 lmp.py:1558] decode step 7 time: 0.15457892417907715 seconds
WARNING 05-06 10:01:10.039303.039303 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:10.039046.039046 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:10.039139.039139 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:10.040904.040904 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:10.045807.045807 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:10.045294.045294 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:10.045541.045541 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:10.046803.046803 cuda_h.py:27] end init_inputs_tokens cost 7.289 ms
DEBUG 05-06 10:01:10.046931.046931 lmp.py:1504] decode step 8 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:10.046171.046171 lmp.py:1510] ---- decode step 8 layer 0 ----
DEBUG 05-06 10:01:10.051865.051865 cuda_h.py:27] end decode_layer cost 4.894 ms
DEBUG 05-06 10:01:10.051814.051814 lmp.py:1510] ---- decode step 8 layer 1 ----
DEBUG 05-06 10:01:10.056321.056321 cuda_h.py:27] end decode_layer cost 4.863 ms
DEBUG 05-06 10:01:10.056741.056741 lmp.py:1510] ---- decode step 8 layer 2 ----
DEBUG 05-06 10:01:10.061170.061170 cuda_h.py:27] end decode_layer cost 4.910 ms
DEBUG 05-06 10:01:10.061828.061828 lmp.py:1510] ---- decode step 8 layer 3 ----
DEBUG 05-06 10:01:10.066650.066650 cuda_h.py:27] end decode_layer cost 4.954 ms
DEBUG 05-06 10:01:10.066923.066923 lmp.py:1510] ---- decode step 8 layer 4 ----
DEBUG 05-06 10:01:10.071563.071563 cuda_h.py:27] end decode_layer cost 4.855 ms
DEBUG 05-06 10:01:10.071028.071028 lmp.py:1510] ---- decode step 8 layer 5 ----
DEBUG 05-06 10:01:10.076689.076689 cuda_h.py:27] end decode_layer cost 5.081 ms
DEBUG 05-06 10:01:10.076916.076916 lmp.py:1510] ---- decode step 8 layer 6 ----
DEBUG 05-06 10:01:10.081011.081011 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 10:01:10.081000.081000 lmp.py:1510] ---- decode step 8 layer 7 ----
DEBUG 05-06 10:01:10.086043.086043 cuda_h.py:27] end decode_layer cost 4.836 ms
DEBUG 05-06 10:01:10.086270.086270 lmp.py:1510] ---- decode step 8 layer 8 ----
DEBUG 05-06 10:01:10.091508.091508 cuda_h.py:27] end decode_layer cost 5.119 ms
DEBUG 05-06 10:01:10.091689.091689 lmp.py:1510] ---- decode step 8 layer 9 ----
DEBUG 05-06 10:01:10.096081.096081 cuda_h.py:27] end decode_layer cost 4.988 ms
DEBUG 05-06 10:01:10.096693.096693 lmp.py:1510] ---- decode step 8 layer 10 ----
DEBUG 05-06 10:01:10.101985.101985 cuda_h.py:27] end decode_layer cost 4.775 ms
DEBUG 05-06 10:01:10.101212.101212 lmp.py:1510] ---- decode step 8 layer 11 ----
DEBUG 05-06 10:01:10.106456.106456 cuda_h.py:27] end decode_layer cost 5.089 ms
DEBUG 05-06 10:01:10.106127.106127 lmp.py:1510] ---- decode step 8 layer 12 ----
DEBUG 05-06 10:01:10.111205.111205 cuda_h.py:27] end decode_layer cost 4.897 ms
DEBUG 05-06 10:01:10.111585.111585 lmp.py:1510] ---- decode step 8 layer 13 ----
DEBUG 05-06 10:01:10.116846.116846 cuda_h.py:27] end decode_layer cost 4.821 ms
DEBUG 05-06 10:01:10.116550.116550 lmp.py:1510] ---- decode step 8 layer 14 ----
DEBUG 05-06 10:01:10.121814.121814 cuda_h.py:27] end decode_layer cost 4.719 ms
DEBUG 05-06 10:01:10.121300.121300 lmp.py:1510] ---- decode step 8 layer 15 ----
DEBUG 05-06 10:01:10.126849.126849 cuda_h.py:27] end decode_layer cost 4.734 ms
DEBUG 05-06 10:01:10.126407.126407 lmp.py:1510] ---- decode step 8 layer 16 ----
DEBUG 05-06 10:01:10.130797.130797 cuda_h.py:27] end decode_layer cost 4.705 ms
DEBUG 05-06 10:01:10.130309.130309 lmp.py:1510] ---- decode step 8 layer 17 ----
DEBUG 05-06 10:01:10.135609.135609 cuda_h.py:27] end decode_layer cost 4.990 ms
DEBUG 05-06 10:01:10.135889.135889 lmp.py:1510] ---- decode step 8 layer 18 ----
DEBUG 05-06 10:01:10.140381.140381 cuda_h.py:27] end decode_layer cost 4.781 ms
DEBUG 05-06 10:01:10.140177.140177 lmp.py:1510] ---- decode step 8 layer 19 ----
DEBUG 05-06 10:01:10.145207.145207 cuda_h.py:27] end decode_layer cost 4.826 ms
DEBUG 05-06 10:01:10.145288.145288 lmp.py:1510] ---- decode step 8 layer 20 ----
DEBUG 05-06 10:01:10.150843.150843 cuda_h.py:27] end decode_layer cost 4.687 ms
DEBUG 05-06 10:01:10.150924.150924 lmp.py:1510] ---- decode step 8 layer 21 ----
DEBUG 05-06 10:01:10.155171.155171 cuda_h.py:27] end decode_layer cost 4.811 ms
DEBUG 05-06 10:01:10.155683.155683 lmp.py:1510] ---- decode step 8 layer 22 ----
DEBUG 05-06 10:01:10.160366.160366 cuda_h.py:27] end decode_layer cost 4.745 ms
DEBUG 05-06 10:01:10.160831.160831 lmp.py:1510] ---- decode step 8 layer 23 ----
DEBUG 05-06 10:01:10.165669.165669 cuda_h.py:27] end decode_layer cost 5.036 ms
DEBUG 05-06 10:01:10.165373.165373 lmp.py:1510] ---- decode step 8 layer 24 ----
DEBUG 05-06 10:01:10.170527.170527 cuda_h.py:27] end decode_layer cost 4.767 ms
DEBUG 05-06 10:01:10.170131.170131 lmp.py:1510] ---- decode step 8 layer 25 ----
DEBUG 05-06 10:01:10.174297.174297 cuda_h.py:27] end decode_layer cost 4.751 ms
DEBUG 05-06 10:01:10.174617.174617 lmp.py:1510] ---- decode step 8 layer 26 ----
DEBUG 05-06 10:01:10.179267.179267 cuda_h.py:27] end decode_layer cost 4.756 ms
DEBUG 05-06 10:01:10.179732.179732 lmp.py:1510] ---- decode step 8 layer 27 ----
DEBUG 05-06 10:01:10.184261.184261 cuda_h.py:27] end decode_layer cost 4.913 ms
DEBUG 05-06 10:01:10.184489.184489 lmp.py:1510] ---- decode step 8 layer 28 ----
DEBUG 05-06 10:01:10.189482.189482 cuda_h.py:27] end decode_layer cost 4.729 ms
DEBUG 05-06 10:01:10.189325.189325 lmp.py:1510] ---- decode step 8 layer 29 ----
DEBUG 05-06 10:01:10.194033.194033 cuda_h.py:27] end decode_layer cost 5.115 ms
DEBUG 05-06 10:01:10.194486.194486 cuda_h.py:27] end decode_step cost 155.444 ms
INFO 05-06 10:01:10.194487.194487 lmp.py:1558] decode step 8 time: 0.15548253059387207 seconds
WARNING 05-06 10:01:10.194485.194485 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:10.195921.195921 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:10.195622.195622 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:10.195361.195361 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:10.200305.200305 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:10.200098.200098 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:10.200490.200490 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:10.202604.202604 cuda_h.py:27] end init_inputs_tokens cost 7.574 ms
DEBUG 05-06 10:01:10.202162.202162 lmp.py:1504] decode step 9 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:10.202640.202640 lmp.py:1510] ---- decode step 9 layer 0 ----
DEBUG 05-06 10:01:10.207766.207766 cuda_h.py:27] end decode_layer cost 4.931 ms
DEBUG 05-06 10:01:10.207754.207754 lmp.py:1510] ---- decode step 9 layer 1 ----
DEBUG 05-06 10:01:10.212838.212838 cuda_h.py:27] end decode_layer cost 4.866 ms
DEBUG 05-06 10:01:10.212588.212588 lmp.py:1510] ---- decode step 9 layer 2 ----
DEBUG 05-06 10:01:10.217143.217143 cuda_h.py:27] end decode_layer cost 4.897 ms
DEBUG 05-06 10:01:10.217609.217609 lmp.py:1510] ---- decode step 9 layer 3 ----
DEBUG 05-06 10:01:10.222116.222116 cuda_h.py:27] end decode_layer cost 4.863 ms
DEBUG 05-06 10:01:10.222105.222105 lmp.py:1510] ---- decode step 9 layer 4 ----
DEBUG 05-06 10:01:10.227764.227764 cuda_h.py:27] end decode_layer cost 4.834 ms
DEBUG 05-06 10:01:10.227561.227561 lmp.py:1510] ---- decode step 9 layer 5 ----
DEBUG 05-06 10:01:10.232214.232214 cuda_h.py:27] end decode_layer cost 5.076 ms
DEBUG 05-06 10:01:10.232442.232442 lmp.py:1510] ---- decode step 9 layer 6 ----
DEBUG 05-06 10:01:10.237536.237536 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 10:01:10.237770.237770 lmp.py:1510] ---- decode step 9 layer 7 ----
DEBUG 05-06 10:01:10.242152.242152 cuda_h.py:27] end decode_layer cost 4.875 ms
DEBUG 05-06 10:01:10.242857.242857 lmp.py:1510] ---- decode step 9 layer 8 ----
DEBUG 05-06 10:01:10.246256.246256 cuda_h.py:27] end decode_layer cost 4.818 ms
DEBUG 05-06 10:01:10.247722.247722 lmp.py:1510] ---- decode step 9 layer 9 ----
DEBUG 05-06 10:01:10.252976.252976 cuda_h.py:27] end decode_layer cost 4.991 ms
DEBUG 05-06 10:01:10.252680.252680 lmp.py:1510] ---- decode step 9 layer 10 ----
DEBUG 05-06 10:01:10.257188.257188 cuda_h.py:27] end decode_layer cost 4.899 ms
DEBUG 05-06 10:01:10.257508.257508 lmp.py:1510] ---- decode step 9 layer 11 ----
DEBUG 05-06 10:01:10.262963.262963 cuda_h.py:27] end decode_layer cost 5.069 ms
DEBUG 05-06 10:01:10.262475.262475 lmp.py:1510] ---- decode step 9 layer 12 ----
DEBUG 05-06 10:01:10.268309.268309 cuda_h.py:27] end decode_layer cost 6.120 ms
DEBUG 05-06 10:01:10.268279.268279 lmp.py:1510] ---- decode step 9 layer 13 ----
DEBUG 05-06 10:01:10.273275.273275 cuda_h.py:27] end decode_layer cost 5.012 ms
DEBUG 05-06 10:01:10.273548.273548 lmp.py:1510] ---- decode step 9 layer 14 ----
DEBUG 05-06 10:01:10.278129.278129 cuda_h.py:27] end decode_layer cost 4.882 ms
DEBUG 05-06 10:01:10.278880.278880 lmp.py:1510] ---- decode step 9 layer 15 ----
DEBUG 05-06 10:01:10.283725.283725 cuda_h.py:27] end decode_layer cost 4.866 ms
DEBUG 05-06 10:01:10.283190.283190 lmp.py:1510] ---- decode step 9 layer 16 ----
DEBUG 05-06 10:01:10.288600.288600 cuda_h.py:27] end decode_layer cost 4.895 ms
DEBUG 05-06 10:01:10.288350.288350 lmp.py:1510] ---- decode step 9 layer 17 ----
DEBUG 05-06 10:01:10.293224.293224 cuda_h.py:27] end decode_layer cost 5.133 ms
DEBUG 05-06 10:01:10.293167.293167 lmp.py:1510] ---- decode step 9 layer 18 ----
DEBUG 05-06 10:01:10.298652.298652 cuda_h.py:27] end decode_layer cost 4.812 ms
DEBUG 05-06 10:01:10.298734.298734 lmp.py:1510] ---- decode step 9 layer 19 ----
DEBUG 05-06 10:01:10.303071.303071 cuda_h.py:27] end decode_layer cost 4.912 ms
DEBUG 05-06 10:01:10.303775.303775 lmp.py:1510] ---- decode step 9 layer 20 ----
DEBUG 05-06 10:01:10.308724.308724 cuda_h.py:27] end decode_layer cost 4.802 ms
DEBUG 05-06 10:01:10.308335.308335 lmp.py:1510] ---- decode step 9 layer 21 ----
DEBUG 05-06 10:01:10.313181.313181 cuda_h.py:27] end decode_layer cost 4.866 ms
DEBUG 05-06 10:01:10.313693.313693 lmp.py:1510] ---- decode step 9 layer 22 ----
DEBUG 05-06 10:01:10.318795.318795 cuda_h.py:27] end decode_layer cost 4.845 ms
DEBUG 05-06 10:01:10.318638.318638 lmp.py:1510] ---- decode step 9 layer 23 ----
DEBUG 05-06 10:01:10.323032.323032 cuda_h.py:27] end decode_layer cost 5.024 ms
DEBUG 05-06 10:01:10.323975.323975 lmp.py:1510] ---- decode step 9 layer 24 ----
DEBUG 05-06 10:01:10.328268.328268 cuda_h.py:27] end decode_layer cost 4.810 ms
DEBUG 05-06 10:01:10.328972.328972 lmp.py:1510] ---- decode step 9 layer 25 ----
DEBUG 05-06 10:01:10.332262.332262 cuda_h.py:27] end decode_layer cost 4.878 ms
DEBUG 05-06 10:01:10.333774.333774 lmp.py:1510] ---- decode step 9 layer 26 ----
DEBUG 05-06 10:01:10.337179.337179 cuda_h.py:27] end decode_layer cost 4.787 ms
DEBUG 05-06 10:01:10.337645.337645 lmp.py:1510] ---- decode step 9 layer 27 ----
DEBUG 05-06 10:01:10.342046.342046 cuda_h.py:27] end decode_layer cost 4.855 ms
DEBUG 05-06 10:01:10.342181.342181 lmp.py:1510] ---- decode step 9 layer 28 ----
DEBUG 05-06 10:01:10.347769.347769 cuda_h.py:27] end decode_layer cost 4.922 ms
DEBUG 05-06 10:01:10.347235.347235 lmp.py:1510] ---- decode step 9 layer 29 ----
DEBUG 05-06 10:01:10.352480.352480 cuda_h.py:27] end decode_layer cost 5.125 ms
DEBUG 05-06 10:01:10.353940.353940 cuda_h.py:27] end decode_step cost 158.248 ms
INFO 05-06 10:01:10.353147.353147 lmp.py:1558] decode step 9 time: 0.15829801559448242 seconds
WARNING 05-06 10:01:10.353006.353006 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:10.353503.353503 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:10.354088.354088 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:10.354469.354469 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:10.359379.359379 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:10.359442.359442 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:10.359119.359119 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:10.360431.360431 cuda_h.py:27] end init_inputs_tokens cost 7.627 ms
DEBUG 05-06 10:01:10.360751.360751 lmp.py:1504] decode step 10 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:10.360805.360805 lmp.py:1510] ---- decode step 10 layer 0 ----
DEBUG 05-06 10:01:10.366807.366807 cuda_h.py:27] end decode_layer cost 5.191 ms
DEBUG 05-06 10:01:10.366034.366034 lmp.py:1510] ---- decode step 10 layer 1 ----
DEBUG 05-06 10:01:10.371233.371233 cuda_h.py:27] end decode_layer cost 4.951 ms
DEBUG 05-06 10:01:10.371460.371460 lmp.py:1510] ---- decode step 10 layer 2 ----
DEBUG 05-06 10:01:10.375078.375078 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 10:01:10.376544.376544 lmp.py:1510] ---- decode step 10 layer 3 ----
DEBUG 05-06 10:01:10.380740.380740 cuda_h.py:27] end decode_layer cost 4.843 ms
DEBUG 05-06 10:01:10.380536.380536 lmp.py:1510] ---- decode step 10 layer 4 ----
DEBUG 05-06 10:01:10.385891.385891 cuda_h.py:27] end decode_layer cost 4.855 ms
DEBUG 05-06 10:01:10.385926.385926 lmp.py:1510] ---- decode step 10 layer 5 ----
DEBUG 05-06 10:01:10.391708.391708 cuda_h.py:27] end decode_layer cost 5.135 ms
DEBUG 05-06 10:01:10.391127.391127 lmp.py:1510] ---- decode step 10 layer 6 ----
DEBUG 05-06 10:01:10.395464.395464 cuda_h.py:27] end decode_layer cost 4.912 ms
DEBUG 05-06 10:01:10.396883.396883 lmp.py:1510] ---- decode step 10 layer 7 ----
DEBUG 05-06 10:01:10.400351.400351 cuda_h.py:27] end decode_layer cost 4.869 ms
DEBUG 05-06 10:01:10.400532.400532 lmp.py:1510] ---- decode step 10 layer 8 ----
DEBUG 05-06 10:01:10.405143.405143 cuda_h.py:27] end decode_layer cost 4.799 ms
DEBUG 05-06 10:01:10.405178.405178 lmp.py:1510] ---- decode step 10 layer 9 ----
DEBUG 05-06 10:01:10.410854.410854 cuda_h.py:27] end decode_layer cost 4.952 ms
DEBUG 05-06 10:01:10.410558.410558 lmp.py:1510] ---- decode step 10 layer 10 ----
DEBUG 05-06 10:01:10.415071.415071 cuda_h.py:27] end decode_layer cost 4.832 ms
DEBUG 05-06 10:01:10.415775.415775 lmp.py:1510] ---- decode step 10 layer 11 ----
DEBUG 05-06 10:01:10.420723.420723 cuda_h.py:27] end decode_layer cost 5.152 ms
DEBUG 05-06 10:01:10.420758.420758 lmp.py:1510] ---- decode step 10 layer 12 ----
DEBUG 05-06 10:01:10.425448.425448 cuda_h.py:27] end decode_layer cost 4.787 ms
DEBUG 05-06 10:01:10.425768.425768 lmp.py:1510] ---- decode step 10 layer 13 ----
DEBUG 05-06 10:01:10.430661.430661 cuda_h.py:27] end decode_layer cost 4.901 ms
DEBUG 05-06 10:01:10.430650.430650 lmp.py:1510] ---- decode step 10 layer 14 ----
DEBUG 05-06 10:01:10.435136.435136 cuda_h.py:27] end decode_layer cost 4.811 ms
DEBUG 05-06 10:01:10.435409.435409 lmp.py:1510] ---- decode step 10 layer 15 ----
DEBUG 05-06 10:01:10.441629.441629 cuda_h.py:27] end decode_layer cost 5.918 ms
DEBUG 05-06 10:01:10.441222.441222 lmp.py:1510] ---- decode step 10 layer 16 ----
DEBUG 05-06 10:01:10.446410.446410 cuda_h.py:27] end decode_layer cost 5.015 ms
DEBUG 05-06 10:01:10.446399.446399 lmp.py:1510] ---- decode step 10 layer 17 ----
DEBUG 05-06 10:01:10.452286.452286 cuda_h.py:27] end decode_layer cost 5.143 ms
DEBUG 05-06 10:01:10.452514.452514 lmp.py:1510] ---- decode step 10 layer 18 ----
DEBUG 05-06 10:01:10.456835.456835 cuda_h.py:27] end decode_layer cost 4.830 ms
DEBUG 05-06 10:01:10.456539.456539 lmp.py:1510] ---- decode step 10 layer 19 ----
DEBUG 05-06 10:01:10.461459.461459 cuda_h.py:27] end decode_layer cost 4.921 ms
DEBUG 05-06 10:01:10.461878.461878 lmp.py:1510] ---- decode step 10 layer 20 ----
DEBUG 05-06 10:01:10.466206.466206 cuda_h.py:27] end decode_layer cost 4.835 ms
DEBUG 05-06 10:01:10.466248.466248 lmp.py:1510] ---- decode step 10 layer 21 ----
DEBUG 05-06 10:01:10.471902.471902 cuda_h.py:27] end decode_layer cost 4.900 ms
DEBUG 05-06 10:01:10.471414.471414 lmp.py:1510] ---- decode step 10 layer 22 ----
DEBUG 05-06 10:01:10.476748.476748 cuda_h.py:27] end decode_layer cost 4.840 ms
DEBUG 05-06 10:01:10.476452.476452 lmp.py:1510] ---- decode step 10 layer 23 ----
DEBUG 05-06 10:01:10.481002.481002 cuda_h.py:27] end decode_layer cost 5.139 ms
DEBUG 05-06 10:01:10.481706.481706 lmp.py:1510] ---- decode step 10 layer 24 ----
DEBUG 05-06 10:01:10.486209.486209 cuda_h.py:27] end decode_layer cost 4.930 ms
DEBUG 05-06 10:01:10.486059.486059 lmp.py:1510] ---- decode step 10 layer 25 ----
DEBUG 05-06 10:01:10.491445.491445 cuda_h.py:27] end decode_layer cost 4.983 ms
DEBUG 05-06 10:01:10.492102.492102 lmp.py:1510] ---- decode step 10 layer 26 ----
DEBUG 05-06 10:01:10.496141.496141 cuda_h.py:27] end decode_layer cost 4.903 ms
DEBUG 05-06 10:01:10.496607.496607 lmp.py:1510] ---- decode step 10 layer 27 ----
DEBUG 05-06 10:01:10.501987.501987 cuda_h.py:27] end decode_layer cost 5.015 ms
DEBUG 05-06 10:01:10.502883.502883 lmp.py:1510] ---- decode step 10 layer 28 ----
DEBUG 05-06 10:01:10.506285.506285 cuda_h.py:27] end decode_layer cost 4.890 ms
DEBUG 05-06 10:01:10.507135.507135 lmp.py:1510] ---- decode step 10 layer 29 ----
DEBUG 05-06 10:01:10.512542.512542 cuda_h.py:27] end decode_layer cost 5.209 ms
DEBUG 05-06 10:01:10.512949.512949 cuda_h.py:27] end decode_step cost 159.172 ms
INFO 05-06 10:01:10.512288.512288 lmp.py:1558] decode step 10 time: 0.1592104434967041 seconds
WARNING 05-06 10:01:10.512505.512505 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:10.512396.512396 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:10.513622.513622 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:10.513864.513864 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:10.518418.518418 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:10.518581.518581 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:10.518211.518211 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:10.520572.520572 cuda_h.py:27] end init_inputs_tokens cost 7.836 ms
DEBUG 05-06 10:01:10.520322.520322 lmp.py:1504] decode step 11 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:10.520708.520708 lmp.py:1510] ---- decode step 11 layer 0 ----
DEBUG 05-06 10:01:10.525333.525333 cuda_h.py:27] end decode_layer cost 5.018 ms
DEBUG 05-06 10:01:10.525944.525944 lmp.py:1510] ---- decode step 11 layer 1 ----
DEBUG 05-06 10:01:10.530808.530808 cuda_h.py:27] end decode_layer cost 5.021 ms
DEBUG 05-06 10:01:10.530512.530512 lmp.py:1510] ---- decode step 11 layer 2 ----
DEBUG 05-06 10:01:10.535246.535246 cuda_h.py:27] end decode_layer cost 4.889 ms
DEBUG 05-06 10:01:10.535711.535711 lmp.py:1510] ---- decode step 11 layer 3 ----
DEBUG 05-06 10:01:10.540401.540401 cuda_h.py:27] end decode_layer cost 4.961 ms
DEBUG 05-06 10:01:10.540012.540012 lmp.py:1510] ---- decode step 11 layer 4 ----
DEBUG 05-06 10:01:10.545833.545833 cuda_h.py:27] end decode_layer cost 4.918 ms
DEBUG 05-06 10:01:10.545391.545391 lmp.py:1510] ---- decode step 11 layer 5 ----
DEBUG 05-06 10:01:10.550499.550499 cuda_h.py:27] end decode_layer cost 5.200 ms
DEBUG 05-06 10:01:10.550819.550819 lmp.py:1510] ---- decode step 11 layer 6 ----
DEBUG 05-06 10:01:10.555916.555916 cuda_h.py:27] end decode_layer cost 4.876 ms
DEBUG 05-06 10:01:10.555859.555859 lmp.py:1510] ---- decode step 11 layer 7 ----
DEBUG 05-06 10:01:10.560388.560388 cuda_h.py:27] end decode_layer cost 4.913 ms
DEBUG 05-06 10:01:10.560807.560807 lmp.py:1510] ---- decode step 11 layer 8 ----
DEBUG 05-06 10:01:10.565413.565413 cuda_h.py:27] end decode_layer cost 4.830 ms
DEBUG 05-06 10:01:10.565117.565117 lmp.py:1510] ---- decode step 11 layer 9 ----
DEBUG 05-06 10:01:10.570275.570275 cuda_h.py:27] end decode_layer cost 4.921 ms
DEBUG 05-06 10:01:10.570502.570502 lmp.py:1510] ---- decode step 11 layer 10 ----
DEBUG 05-06 10:01:10.575487.575487 cuda_h.py:27] end decode_layer cost 4.863 ms
DEBUG 05-06 10:01:10.575191.575191 lmp.py:1510] ---- decode step 11 layer 11 ----
DEBUG 05-06 10:01:10.580007.580007 cuda_h.py:27] end decode_layer cost 5.195 ms
DEBUG 05-06 10:01:10.580427.580427 lmp.py:1510] ---- decode step 11 layer 12 ----
DEBUG 05-06 10:01:10.585919.585919 cuda_h.py:27] end decode_layer cost 4.817 ms
DEBUG 05-06 10:01:10.585623.585623 lmp.py:1510] ---- decode step 11 layer 13 ----
DEBUG 05-06 10:01:10.590407.590407 cuda_h.py:27] end decode_layer cost 4.821 ms
DEBUG 05-06 10:01:10.590204.590204 lmp.py:1510] ---- decode step 11 layer 14 ----
DEBUG 05-06 10:01:10.595257.595257 cuda_h.py:27] end decode_layer cost 4.738 ms
DEBUG 05-06 10:01:10.595815.595815 lmp.py:1510] ---- decode step 11 layer 15 ----
DEBUG 05-06 10:01:10.600098.600098 cuda_h.py:27] end decode_layer cost 4.872 ms
DEBUG 05-06 10:01:10.600802.600802 lmp.py:1510] ---- decode step 11 layer 16 ----
DEBUG 05-06 10:01:10.605842.605842 cuda_h.py:27] end decode_layer cost 4.765 ms
DEBUG 05-06 10:01:10.605878.605878 lmp.py:1510] ---- decode step 11 layer 17 ----
DEBUG 05-06 10:01:10.610323.610323 cuda_h.py:27] end decode_layer cost 4.992 ms
DEBUG 05-06 10:01:10.610835.610835 lmp.py:1510] ---- decode step 11 layer 18 ----
DEBUG 05-06 10:01:10.615393.615393 cuda_h.py:27] end decode_layer cost 4.795 ms
DEBUG 05-06 10:01:10.615952.615952 lmp.py:1510] ---- decode step 11 layer 19 ----
DEBUG 05-06 10:01:10.620152.620152 cuda_h.py:27] end decode_layer cost 5.789 ms
DEBUG 05-06 10:01:10.620738.620738 lmp.py:1510] ---- decode step 11 layer 20 ----
DEBUG 05-06 10:01:10.625912.625912 cuda_h.py:27] end decode_layer cost 4.790 ms
DEBUG 05-06 10:01:10.625470.625470 lmp.py:1510] ---- decode step 11 layer 21 ----
DEBUG 05-06 10:01:10.630349.630349 cuda_h.py:27] end decode_layer cost 4.891 ms
DEBUG 05-06 10:01:10.630815.630815 lmp.py:1510] ---- decode step 11 layer 22 ----
DEBUG 05-06 10:01:10.635117.635117 cuda_h.py:27] end decode_layer cost 4.852 ms
DEBUG 05-06 10:01:10.635774.635774 lmp.py:1510] ---- decode step 11 layer 23 ----
DEBUG 05-06 10:01:10.640966.640966 cuda_h.py:27] end decode_layer cost 5.121 ms
DEBUG 05-06 10:01:10.640193.640193 lmp.py:1510] ---- decode step 11 layer 24 ----
DEBUG 05-06 10:01:10.645703.645703 cuda_h.py:27] end decode_layer cost 4.934 ms
DEBUG 05-06 10:01:10.645076.645076 lmp.py:1510] ---- decode step 11 layer 25 ----
DEBUG 05-06 10:01:10.650414.650414 cuda_h.py:27] end decode_layer cost 4.948 ms
DEBUG 05-06 10:01:10.650880.650880 lmp.py:1510] ---- decode step 11 layer 26 ----
DEBUG 05-06 10:01:10.655860.655860 cuda_h.py:27] end decode_layer cost 4.930 ms
DEBUG 05-06 10:01:10.655279.655279 lmp.py:1510] ---- decode step 11 layer 27 ----
DEBUG 05-06 10:01:10.660241.660241 cuda_h.py:27] end decode_layer cost 4.988 ms
DEBUG 05-06 10:01:10.660183.660183 lmp.py:1510] ---- decode step 11 layer 28 ----
DEBUG 05-06 10:01:10.665797.665797 cuda_h.py:27] end decode_layer cost 4.870 ms
DEBUG 05-06 10:01:10.665024.665024 lmp.py:1510] ---- decode step 11 layer 29 ----
DEBUG 05-06 10:01:10.671077.671077 cuda_h.py:27] end decode_layer cost 5.124 ms
DEBUG 05-06 10:01:10.671273.671273 cuda_h.py:27] end decode_step cost 158.708 ms
INFO 05-06 10:01:10.671751.671751 lmp.py:1558] decode step 11 time: 0.15874719619750977 seconds
WARNING 05-06 10:01:10.671338.671338 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:10.671430.671430 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:10.672515.672515 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:10.672373.672373 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:10.677179.677179 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:10.677958.677958 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:10.677357.677357 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:10.678798.678798 cuda_h.py:27] end init_inputs_tokens cost 7.699 ms
DEBUG 05-06 10:01:10.678549.678549 lmp.py:1504] decode step 12 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:10.679457.679457 lmp.py:1510] ---- decode step 12 layer 0 ----
DEBUG 05-06 10:01:10.683494.683494 cuda_h.py:27] end decode_layer cost 4.830 ms
DEBUG 05-06 10:01:10.683482.683482 lmp.py:1510] ---- decode step 12 layer 1 ----
DEBUG 05-06 10:01:10.688509.688509 cuda_h.py:27] end decode_layer cost 4.929 ms
DEBUG 05-06 10:01:10.688259.688259 lmp.py:1510] ---- decode step 12 layer 2 ----
DEBUG 05-06 10:01:10.693937.693937 cuda_h.py:27] end decode_layer cost 4.813 ms
DEBUG 05-06 10:01:10.693972.693972 lmp.py:1510] ---- decode step 12 layer 3 ----
DEBUG 05-06 10:01:10.698898.698898 cuda_h.py:27] end decode_layer cost 4.890 ms
DEBUG 05-06 10:01:10.698317.698317 lmp.py:1510] ---- decode step 12 layer 4 ----
DEBUG 05-06 10:01:10.703988.703988 cuda_h.py:27] end decode_layer cost 4.808 ms
DEBUG 05-06 10:01:10.703653.703653 lmp.py:1510] ---- decode step 12 layer 5 ----
DEBUG 05-06 10:01:10.708811.708811 cuda_h.py:27] end decode_layer cost 5.132 ms
DEBUG 05-06 10:01:10.708800.708800 lmp.py:1510] ---- decode step 12 layer 6 ----
DEBUG 05-06 10:01:10.713617.713617 cuda_h.py:27] end decode_layer cost 4.810 ms
DEBUG 05-06 10:01:10.713997.713997 lmp.py:1510] ---- decode step 12 layer 7 ----
DEBUG 05-06 10:01:10.718142.718142 cuda_h.py:27] end decode_layer cost 4.911 ms
DEBUG 05-06 10:01:10.718084.718084 lmp.py:1510] ---- decode step 12 layer 8 ----
DEBUG 05-06 10:01:10.723153.723153 cuda_h.py:27] end decode_layer cost 4.820 ms
DEBUG 05-06 10:01:10.723665.723665 lmp.py:1510] ---- decode step 12 layer 9 ----
DEBUG 05-06 10:01:10.728213.728213 cuda_h.py:27] end decode_layer cost 4.893 ms
DEBUG 05-06 10:01:10.728725.728725 lmp.py:1510] ---- decode step 12 layer 10 ----
DEBUG 05-06 10:01:10.733117.733117 cuda_h.py:27] end decode_layer cost 4.778 ms
DEBUG 05-06 10:01:10.733960.733960 lmp.py:1510] ---- decode step 12 layer 11 ----
DEBUG 05-06 10:01:10.738800.738800 cuda_h.py:27] end decode_layer cost 5.108 ms
DEBUG 05-06 10:01:10.738412.738412 lmp.py:1510] ---- decode step 12 layer 12 ----
DEBUG 05-06 10:01:10.743142.743142 cuda_h.py:27] end decode_layer cost 4.781 ms
DEBUG 05-06 10:01:10.743369.743369 lmp.py:1510] ---- decode step 12 layer 13 ----
DEBUG 05-06 10:01:10.748537.748537 cuda_h.py:27] end decode_layer cost 4.823 ms
DEBUG 05-06 10:01:10.748811.748811 lmp.py:1510] ---- decode step 12 layer 14 ----
DEBUG 05-06 10:01:10.753542.753542 cuda_h.py:27] end decode_layer cost 4.817 ms
DEBUG 05-06 10:01:10.753484.753484 lmp.py:1510] ---- decode step 12 layer 15 ----
DEBUG 05-06 10:01:10.758656.758656 cuda_h.py:27] end decode_layer cost 4.932 ms
DEBUG 05-06 10:01:10.758883.758883 lmp.py:1510] ---- decode step 12 layer 16 ----
DEBUG 05-06 10:01:10.763654.763654 cuda_h.py:27] end decode_layer cost 4.810 ms
DEBUG 05-06 10:01:10.763927.763927 lmp.py:1510] ---- decode step 12 layer 17 ----
DEBUG 05-06 10:01:10.768510.768510 cuda_h.py:27] end decode_layer cost 5.129 ms
DEBUG 05-06 10:01:10.768545.768545 lmp.py:1510] ---- decode step 12 layer 18 ----
DEBUG 05-06 10:01:10.773468.773468 cuda_h.py:27] end decode_layer cost 4.818 ms
DEBUG 05-06 10:01:10.773457.773457 lmp.py:1510] ---- decode step 12 layer 19 ----
DEBUG 05-06 10:01:10.778615.778615 cuda_h.py:27] end decode_layer cost 4.921 ms
DEBUG 05-06 10:01:10.778127.778127 lmp.py:1510] ---- decode step 12 layer 20 ----
DEBUG 05-06 10:01:10.783959.783959 cuda_h.py:27] end decode_layer cost 4.856 ms
DEBUG 05-06 10:01:10.783948.783948 lmp.py:1510] ---- decode step 12 layer 21 ----
DEBUG 05-06 10:01:10.788122.788122 cuda_h.py:27] end decode_layer cost 5.003 ms
DEBUG 05-06 10:01:10.788826.788826 lmp.py:1510] ---- decode step 12 layer 22 ----
DEBUG 05-06 10:01:10.793204.793204 cuda_h.py:27] end decode_layer cost 5.539 ms
DEBUG 05-06 10:01:10.793100.793100 lmp.py:1510] ---- decode step 12 layer 23 ----
DEBUG 05-06 10:01:10.799628.799628 cuda_h.py:27] end decode_layer cost 5.263 ms
DEBUG 05-06 10:01:10.799901.799901 lmp.py:1510] ---- decode step 12 layer 24 ----
DEBUG 05-06 10:01:10.803178.803178 cuda_h.py:27] end decode_layer cost 4.903 ms
DEBUG 05-06 10:01:10.804975.804975 lmp.py:1510] ---- decode step 12 layer 25 ----
DEBUG 05-06 10:01:10.809393.809393 cuda_h.py:27] end decode_layer cost 4.972 ms
DEBUG 05-06 10:01:10.809144.809144 lmp.py:1510] ---- decode step 12 layer 26 ----
DEBUG 05-06 10:01:10.813709.813709 cuda_h.py:27] end decode_layer cost 4.800 ms
DEBUG 05-06 10:01:10.813029.813029 lmp.py:1510] ---- decode step 12 layer 27 ----
DEBUG 05-06 10:01:10.818672.818672 cuda_h.py:27] end decode_layer cost 4.962 ms
DEBUG 05-06 10:01:10.818422.818422 lmp.py:1510] ---- decode step 12 layer 28 ----
DEBUG 05-06 10:01:10.823479.823479 cuda_h.py:27] end decode_layer cost 4.846 ms
DEBUG 05-06 10:01:10.823375.823375 lmp.py:1510] ---- decode step 12 layer 29 ----
DEBUG 05-06 10:01:10.829798.829798 cuda_h.py:27] end decode_layer cost 5.117 ms
DEBUG 05-06 10:01:10.829828.829828 cuda_h.py:27] end decode_step cost 157.900 ms
INFO 05-06 10:01:10.829829.829829 lmp.py:1558] decode step 12 time: 0.15793919563293457 seconds
WARNING 05-06 10:01:10.829112.829112 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:10.829805.829805 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:10.830606.830606 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:10.830517.830517 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:10.835309.835309 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:10.835134.835134 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:10.835527.835527 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:10.836078.836078 cuda_h.py:27] end init_inputs_tokens cost 7.566 ms
DEBUG 05-06 10:01:10.836351.836351 lmp.py:1504] decode step 13 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:10.836452.836452 lmp.py:1510] ---- decode step 13 layer 0 ----
DEBUG 05-06 10:01:10.841534.841534 cuda_h.py:27] end decode_layer cost 4.829 ms
DEBUG 05-06 10:01:10.841285.841285 lmp.py:1510] ---- decode step 13 layer 1 ----
DEBUG 05-06 10:01:10.846223.846223 cuda_h.py:27] end decode_layer cost 4.864 ms
DEBUG 05-06 10:01:10.846973.846973 lmp.py:1510] ---- decode step 13 layer 2 ----
DEBUG 05-06 10:01:10.851273.851273 cuda_h.py:27] end decode_layer cost 4.815 ms
DEBUG 05-06 10:01:10.851938.851938 lmp.py:1510] ---- decode step 13 layer 3 ----
DEBUG 05-06 10:01:10.856771.856771 cuda_h.py:27] end decode_layer cost 4.898 ms
DEBUG 05-06 10:01:10.856760.856760 lmp.py:1510] ---- decode step 13 layer 4 ----
DEBUG 05-06 10:01:10.861611.861611 cuda_h.py:27] end decode_layer cost 4.835 ms
DEBUG 05-06 10:01:10.861884.861884 lmp.py:1510] ---- decode step 13 layer 5 ----
DEBUG 05-06 10:01:10.866017.866017 cuda_h.py:27] end decode_layer cost 5.148 ms
DEBUG 05-06 10:01:10.866483.866483 lmp.py:1510] ---- decode step 13 layer 6 ----
DEBUG 05-06 10:01:10.871809.871809 cuda_h.py:27] end decode_layer cost 4.800 ms
DEBUG 05-06 10:01:10.871652.871652 lmp.py:1510] ---- decode step 13 layer 7 ----
DEBUG 05-06 10:01:10.876192.876192 cuda_h.py:27] end decode_layer cost 4.852 ms
DEBUG 05-06 10:01:10.876704.876704 lmp.py:1510] ---- decode step 13 layer 8 ----
DEBUG 05-06 10:01:10.881430.881430 cuda_h.py:27] end decode_layer cost 4.848 ms
DEBUG 05-06 10:01:10.881180.881180 lmp.py:1510] ---- decode step 13 layer 9 ----
DEBUG 05-06 10:01:10.886257.886257 cuda_h.py:27] end decode_layer cost 4.861 ms
DEBUG 05-06 10:01:10.886531.886531 lmp.py:1510] ---- decode step 13 layer 10 ----
DEBUG 05-06 10:01:10.891748.891748 cuda_h.py:27] end decode_layer cost 4.894 ms
DEBUG 05-06 10:01:10.891783.891783 lmp.py:1510] ---- decode step 13 layer 11 ----
DEBUG 05-06 10:01:10.896294.896294 cuda_h.py:27] end decode_layer cost 5.181 ms
DEBUG 05-06 10:01:10.896714.896714 lmp.py:1510] ---- decode step 13 layer 12 ----
DEBUG 05-06 10:01:10.901010.901010 cuda_h.py:27] end decode_layer cost 4.882 ms
DEBUG 05-06 10:01:10.901191.901191 lmp.py:1510] ---- decode step 13 layer 13 ----
DEBUG 05-06 10:01:10.906263.906263 cuda_h.py:27] end decode_layer cost 4.928 ms
DEBUG 05-06 10:01:10.906490.906490 lmp.py:1510] ---- decode step 13 layer 14 ----
DEBUG 05-06 10:01:10.911336.911336 cuda_h.py:27] end decode_layer cost 4.866 ms
DEBUG 05-06 10:01:10.911563.911563 lmp.py:1510] ---- decode step 13 layer 15 ----
DEBUG 05-06 10:01:10.916511.916511 cuda_h.py:27] end decode_layer cost 4.977 ms
DEBUG 05-06 10:01:10.916215.916215 lmp.py:1510] ---- decode step 13 layer 16 ----
DEBUG 05-06 10:01:10.921455.921455 cuda_h.py:27] end decode_layer cost 4.981 ms
DEBUG 05-06 10:01:10.921682.921682 lmp.py:1510] ---- decode step 13 layer 17 ----
DEBUG 05-06 10:01:10.926537.926537 cuda_h.py:27] end decode_layer cost 5.154 ms
DEBUG 05-06 10:01:10.926526.926526 lmp.py:1510] ---- decode step 13 layer 18 ----
DEBUG 05-06 10:01:10.931142.931142 cuda_h.py:27] end decode_layer cost 4.942 ms
DEBUG 05-06 10:01:10.931561.931561 lmp.py:1510] ---- decode step 13 layer 19 ----
DEBUG 05-06 10:01:10.936853.936853 cuda_h.py:27] end decode_layer cost 4.949 ms
DEBUG 05-06 10:01:10.936511.936511 lmp.py:1510] ---- decode step 13 layer 20 ----
DEBUG 05-06 10:01:10.941158.941158 cuda_h.py:27] end decode_layer cost 4.896 ms
DEBUG 05-06 10:01:10.941147.941147 lmp.py:1510] ---- decode step 13 layer 21 ----
DEBUG 05-06 10:01:10.946883.946883 cuda_h.py:27] end decode_layer cost 4.960 ms
DEBUG 05-06 10:01:10.946634.946634 lmp.py:1510] ---- decode step 13 layer 22 ----
DEBUG 05-06 10:01:10.951401.951401 cuda_h.py:27] end decode_layer cost 4.914 ms
DEBUG 05-06 10:01:10.951867.951867 lmp.py:1510] ---- decode step 13 layer 23 ----
DEBUG 05-06 10:01:10.956344.956344 cuda_h.py:27] end decode_layer cost 5.156 ms
DEBUG 05-06 10:01:10.956048.956048 lmp.py:1510] ---- decode step 13 layer 24 ----
DEBUG 05-06 10:01:10.961517.961517 cuda_h.py:27] end decode_layer cost 4.904 ms
DEBUG 05-06 10:01:10.961983.961983 lmp.py:1510] ---- decode step 13 layer 25 ----
DEBUG 05-06 10:01:10.966989.966989 cuda_h.py:27] end decode_layer cost 4.914 ms
DEBUG 05-06 10:01:10.966408.966408 lmp.py:1510] ---- decode step 13 layer 26 ----
DEBUG 05-06 10:01:10.972758.972758 cuda_h.py:27] end decode_layer cost 5.097 ms
DEBUG 05-06 10:01:10.972131.972131 lmp.py:1510] ---- decode step 13 layer 27 ----
DEBUG 05-06 10:01:10.977522.977522 cuda_h.py:27] end decode_layer cost 4.953 ms
DEBUG 05-06 10:01:10.977080.977080 lmp.py:1510] ---- decode step 13 layer 28 ----
DEBUG 05-06 10:01:10.981375.981375 cuda_h.py:27] end decode_layer cost 4.846 ms
DEBUG 05-06 10:01:10.982364.982364 lmp.py:1510] ---- decode step 13 layer 29 ----
DEBUG 05-06 10:01:10.987307.987307 cuda_h.py:27] end decode_layer cost 5.219 ms
DEBUG 05-06 10:01:10.987853.987853 cuda_h.py:27] end decode_step cost 158.108 ms
INFO 05-06 10:01:10.987139.987139 lmp.py:1558] decode step 13 time: 0.15814566612243652 seconds
WARNING 05-06 10:01:10.987137.987137 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:10.987469.987469 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:10.988747.988747 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:10.988088.988088 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:10.993160.993160 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:10.993515.993515 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:10.993861.993861 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:10.995094.995094 cuda_h.py:27] end init_inputs_tokens cost 7.701 ms
DEBUG 05-06 10:01:10.995752.995752 lmp.py:1504] decode step 14 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:10.995138.995138 lmp.py:1510] ---- decode step 14 layer 0 ----
DEBUG 05-06 10:01:11.000895.000895 cuda_h.py:27] end decode_layer cost 5.222 ms
DEBUG 05-06 10:01:11.000891.000891 lmp.py:1510] ---- decode step 14 layer 1 ----
DEBUG 05-06 10:01:11.005985.005985 cuda_h.py:27] end decode_layer cost 4.979 ms
DEBUG 05-06 10:01:11.005881.005881 lmp.py:1510] ---- decode step 14 layer 2 ----
DEBUG 05-06 10:01:11.010177.010177 cuda_h.py:27] end decode_layer cost 4.883 ms
DEBUG 05-06 10:01:11.010736.010736 lmp.py:1510] ---- decode step 14 layer 3 ----
DEBUG 05-06 10:01:11.015657.015657 cuda_h.py:27] end decode_layer cost 4.957 ms
DEBUG 05-06 10:01:11.015930.015930 lmp.py:1510] ---- decode step 14 layer 4 ----
DEBUG 05-06 10:01:11.020131.020131 cuda_h.py:27] end decode_layer cost 4.987 ms
DEBUG 05-06 10:01:11.020742.020742 lmp.py:1510] ---- decode step 14 layer 5 ----
DEBUG 05-06 10:01:11.025405.025405 cuda_h.py:27] end decode_layer cost 5.153 ms
DEBUG 05-06 10:01:11.025970.025970 lmp.py:1510] ---- decode step 14 layer 6 ----
DEBUG 05-06 10:01:11.030251.030251 cuda_h.py:27] end decode_layer cost 4.836 ms
DEBUG 05-06 10:01:11.030955.030955 lmp.py:1510] ---- decode step 14 layer 7 ----
DEBUG 05-06 10:01:11.035683.035683 cuda_h.py:27] end decode_layer cost 4.920 ms
DEBUG 05-06 10:01:11.035626.035626 lmp.py:1510] ---- decode step 14 layer 8 ----
DEBUG 05-06 10:01:11.040477.040477 cuda_h.py:27] end decode_layer cost 4.835 ms
DEBUG 05-06 10:01:11.040419.040419 lmp.py:1510] ---- decode step 14 layer 9 ----
DEBUG 05-06 10:01:11.045019.045019 cuda_h.py:27] end decode_layer cost 4.861 ms
DEBUG 05-06 10:01:11.045816.045816 lmp.py:1510] ---- decode step 14 layer 10 ----
DEBUG 05-06 10:01:11.050175.050175 cuda_h.py:27] end decode_layer cost 4.788 ms
DEBUG 05-06 10:01:11.050833.050833 lmp.py:1510] ---- decode step 14 layer 11 ----
DEBUG 05-06 10:01:11.055032.055032 cuda_h.py:27] end decode_layer cost 5.162 ms
DEBUG 05-06 10:01:11.055651.055651 lmp.py:1510] ---- decode step 14 layer 12 ----
DEBUG 05-06 10:01:11.060819.060819 cuda_h.py:27] end decode_layer cost 4.823 ms
DEBUG 05-06 10:01:11.060523.060523 lmp.py:1510] ---- decode step 14 layer 13 ----
DEBUG 05-06 10:01:11.065971.065971 cuda_h.py:27] end decode_layer cost 4.854 ms
DEBUG 05-06 10:01:11.065767.065767 lmp.py:1510] ---- decode step 14 layer 14 ----
DEBUG 05-06 10:01:11.070635.070635 cuda_h.py:27] end decode_layer cost 4.742 ms
DEBUG 05-06 10:01:11.070955.070955 lmp.py:1510] ---- decode step 14 layer 15 ----
DEBUG 05-06 10:01:11.074383.074383 cuda_h.py:27] end decode_layer cost 4.663 ms
DEBUG 05-06 10:01:11.074179.074179 lmp.py:1510] ---- decode step 14 layer 16 ----
DEBUG 05-06 10:01:11.079101.079101 cuda_h.py:27] end decode_layer cost 4.782 ms
DEBUG 05-06 10:01:11.079090.079090 lmp.py:1510] ---- decode step 14 layer 17 ----
DEBUG 05-06 10:01:11.084779.084779 cuda_h.py:27] end decode_layer cost 5.137 ms
DEBUG 05-06 10:01:11.085245.085245 lmp.py:1510] ---- decode step 14 layer 18 ----
DEBUG 05-06 10:01:11.089103.089103 cuda_h.py:27] end decode_layer cost 4.876 ms
DEBUG 05-06 10:01:11.089139.089139 lmp.py:1510] ---- decode step 14 layer 19 ----
DEBUG 05-06 10:01:11.094966.094966 cuda_h.py:27] end decode_layer cost 4.923 ms
DEBUG 05-06 10:01:11.094239.094239 lmp.py:1510] ---- decode step 14 layer 20 ----
DEBUG 05-06 10:01:11.099534.099534 cuda_h.py:27] end decode_layer cost 4.846 ms
DEBUG 05-06 10:01:11.099477.099477 lmp.py:1510] ---- decode step 14 layer 21 ----
DEBUG 05-06 10:01:11.104694.104694 cuda_h.py:27] end decode_layer cost 4.894 ms
DEBUG 05-06 10:01:11.104683.104683 lmp.py:1510] ---- decode step 14 layer 22 ----
DEBUG 05-06 10:01:11.109660.109660 cuda_h.py:27] end decode_layer cost 4.859 ms
DEBUG 05-06 10:01:11.109695.109695 lmp.py:1510] ---- decode step 14 layer 23 ----
DEBUG 05-06 10:01:11.114588.114588 cuda_h.py:27] end decode_layer cost 5.112 ms
DEBUG 05-06 10:01:11.114862.114862 lmp.py:1510] ---- decode step 14 layer 24 ----
DEBUG 05-06 10:01:11.119650.119650 cuda_h.py:27] end decode_layer cost 4.928 ms
DEBUG 05-06 10:01:11.119354.119354 lmp.py:1510] ---- decode step 14 layer 25 ----
DEBUG 05-06 10:01:11.124122.124122 cuda_h.py:27] end decode_layer cost 4.950 ms
DEBUG 05-06 10:01:11.124926.124926 lmp.py:1510] ---- decode step 14 layer 26 ----
DEBUG 05-06 10:01:11.129440.129440 cuda_h.py:27] end decode_layer cost 4.867 ms
DEBUG 05-06 10:01:11.129191.129191 lmp.py:1510] ---- decode step 14 layer 27 ----
DEBUG 05-06 10:01:11.134652.134652 cuda_h.py:27] end decode_layer cost 4.864 ms
DEBUG 05-06 10:01:11.134594.134594 lmp.py:1510] ---- decode step 14 layer 28 ----
DEBUG 05-06 10:01:11.139040.139040 cuda_h.py:27] end decode_layer cost 4.817 ms
DEBUG 05-06 10:01:11.139029.139029 lmp.py:1510] ---- decode step 14 layer 29 ----
DEBUG 05-06 10:01:11.145024.145024 cuda_h.py:27] end decode_layer cost 5.362 ms
DEBUG 05-06 10:01:11.145384.145384 cuda_h.py:27] end decode_step cost 157.779 ms
INFO 05-06 10:01:11.145670.145670 lmp.py:1558] decode step 14 time: 0.15781664848327637 seconds
WARNING 05-06 10:01:11.145430.145430 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:11.145035.145035 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:11.145489.145489 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:11.146022.146022 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:11.151284.151284 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:11.151539.151539 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:11.151170.151170 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:11.152899.152899 cuda_h.py:27] end init_inputs_tokens cost 7.376 ms
DEBUG 05-06 10:01:11.152173.152173 lmp.py:1504] decode step 15 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:11.152843.152843 lmp.py:1510] ---- decode step 15 layer 0 ----
DEBUG 05-06 10:01:11.157783.157783 cuda_h.py:27] end decode_layer cost 4.935 ms
DEBUG 05-06 10:01:11.157487.157487 lmp.py:1510] ---- decode step 15 layer 1 ----
DEBUG 05-06 10:01:11.162420.162420 cuda_h.py:27] end decode_layer cost 4.895 ms
DEBUG 05-06 10:01:11.162839.162839 lmp.py:1510] ---- decode step 15 layer 2 ----
DEBUG 05-06 10:01:11.167022.167022 cuda_h.py:27] end decode_layer cost 4.861 ms
DEBUG 05-06 10:01:11.167488.167488 lmp.py:1510] ---- decode step 15 layer 3 ----
DEBUG 05-06 10:01:11.172274.172274 cuda_h.py:27] end decode_layer cost 4.893 ms
DEBUG 05-06 10:01:11.172025.172025 lmp.py:1510] ---- decode step 15 layer 4 ----
DEBUG 05-06 10:01:11.177995.177995 cuda_h.py:27] end decode_layer cost 4.853 ms
DEBUG 05-06 10:01:11.177223.177223 lmp.py:1510] ---- decode step 15 layer 5 ----
DEBUG 05-06 10:01:11.182395.182395 cuda_h.py:27] end decode_layer cost 5.142 ms
DEBUG 05-06 10:01:11.182814.182814 lmp.py:1510] ---- decode step 15 layer 6 ----
DEBUG 05-06 10:01:11.187589.187589 cuda_h.py:27] end decode_layer cost 4.919 ms
DEBUG 05-06 10:01:11.187008.187008 lmp.py:1510] ---- decode step 15 layer 7 ----
DEBUG 05-06 10:01:11.192398.192398 cuda_h.py:27] end decode_layer cost 4.916 ms
DEBUG 05-06 10:01:11.192771.192771 lmp.py:1510] ---- decode step 15 layer 8 ----
DEBUG 05-06 10:01:11.197286.197286 cuda_h.py:27] end decode_layer cost 4.904 ms
DEBUG 05-06 10:01:11.197766.197766 lmp.py:1510] ---- decode step 15 layer 9 ----
DEBUG 05-06 10:01:11.202732.202732 cuda_h.py:27] end decode_layer cost 4.920 ms
DEBUG 05-06 10:01:11.202913.202913 lmp.py:1510] ---- decode step 15 layer 10 ----
DEBUG 05-06 10:01:11.207461.207461 cuda_h.py:27] end decode_layer cost 4.885 ms
DEBUG 05-06 10:01:11.207093.207093 lmp.py:1510] ---- decode step 15 layer 11 ----
DEBUG 05-06 10:01:11.212730.212730 cuda_h.py:27] end decode_layer cost 5.169 ms
DEBUG 05-06 10:01:11.212672.212672 lmp.py:1510] ---- decode step 15 layer 12 ----
DEBUG 05-06 10:01:11.217895.217895 cuda_h.py:27] end decode_layer cost 4.863 ms
DEBUG 05-06 10:01:11.217003.217003 lmp.py:1510] ---- decode step 15 layer 13 ----
DEBUG 05-06 10:01:11.222434.222434 cuda_h.py:27] end decode_layer cost 4.947 ms
DEBUG 05-06 10:01:11.222900.222900 lmp.py:1510] ---- decode step 15 layer 14 ----
DEBUG 05-06 10:01:11.227903.227903 cuda_h.py:27] end decode_layer cost 4.842 ms
DEBUG 05-06 10:01:11.227515.227515 lmp.py:1510] ---- decode step 15 layer 15 ----
DEBUG 05-06 10:01:11.232981.232981 cuda_h.py:27] end decode_layer cost 4.832 ms
DEBUG 05-06 10:01:11.232447.232447 lmp.py:1510] ---- decode step 15 layer 16 ----
DEBUG 05-06 10:01:11.237337.237337 cuda_h.py:27] end decode_layer cost 4.829 ms
DEBUG 05-06 10:01:11.237803.237803 lmp.py:1510] ---- decode step 15 layer 17 ----
DEBUG 05-06 10:01:11.242756.242756 cuda_h.py:27] end decode_layer cost 5.121 ms
DEBUG 05-06 10:01:11.242268.242268 lmp.py:1510] ---- decode step 15 layer 18 ----
DEBUG 05-06 10:01:11.247198.247198 cuda_h.py:27] end decode_layer cost 4.822 ms
DEBUG 05-06 10:01:11.247147.247147 lmp.py:1510] ---- decode step 15 layer 19 ----
DEBUG 05-06 10:01:11.252889.252889 cuda_h.py:27] end decode_layer cost 4.929 ms
DEBUG 05-06 10:01:11.252070.252070 lmp.py:1510] ---- decode step 15 layer 20 ----
DEBUG 05-06 10:01:11.257162.257162 cuda_h.py:27] end decode_layer cost 4.943 ms
DEBUG 05-06 10:01:11.257105.257105 lmp.py:1510] ---- decode step 15 layer 21 ----
DEBUG 05-06 10:01:11.262634.262634 cuda_h.py:27] end decode_layer cost 4.914 ms
DEBUG 05-06 10:01:11.262954.262954 lmp.py:1510] ---- decode step 15 layer 22 ----
DEBUG 05-06 10:01:11.267592.267592 cuda_h.py:27] end decode_layer cost 4.818 ms
DEBUG 05-06 10:01:11.267773.267773 lmp.py:1510] ---- decode step 15 layer 23 ----
DEBUG 05-06 10:01:11.272812.272812 cuda_h.py:27] end decode_layer cost 5.114 ms
DEBUG 05-06 10:01:11.272039.272039 lmp.py:1510] ---- decode step 15 layer 24 ----
DEBUG 05-06 10:01:11.277917.277917 cuda_h.py:27] end decode_layer cost 4.855 ms
DEBUG 05-06 10:01:11.277383.277383 lmp.py:1510] ---- decode step 15 layer 25 ----
DEBUG 05-06 10:01:11.282268.282268 cuda_h.py:27] end decode_layer cost 4.860 ms
DEBUG 05-06 10:01:11.282733.282733 lmp.py:1510] ---- decode step 15 layer 26 ----
DEBUG 05-06 10:01:11.287201.287201 cuda_h.py:27] end decode_layer cost 4.868 ms
DEBUG 05-06 10:01:11.287429.287429 lmp.py:1510] ---- decode step 15 layer 27 ----
DEBUG 05-06 10:01:11.292768.292768 cuda_h.py:27] end decode_layer cost 4.984 ms
DEBUG 05-06 10:01:11.292618.292618 lmp.py:1510] ---- decode step 15 layer 28 ----
DEBUG 05-06 10:01:11.297128.297128 cuda_h.py:27] end decode_layer cost 4.759 ms
DEBUG 05-06 10:01:11.297309.297309 lmp.py:1510] ---- decode step 15 layer 29 ----
DEBUG 05-06 10:01:11.302620.302620 cuda_h.py:27] end decode_layer cost 5.139 ms
DEBUG 05-06 10:01:11.302690.302690 cuda_h.py:27] end decode_step cost 157.367 ms
INFO 05-06 10:01:11.302499.302499 lmp.py:1558] decode step 15 time: 0.15740418434143066 seconds
WARNING 05-06 10:01:11.302020.302020 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:11.303357.303357 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:11.303226.303226 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:11.303304.303304 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:11.308991.308991 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:11.308054.308054 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:11.309493.309493 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 10:01:11.309545.309545 helper.py:80] WARNING: Logits have extreme values: min=-764.00, max=1008.00
WARNING 05-06 10:01:11.309404.309404 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 10:01:11.310684.310684 cuda_h.py:27] end init_inputs_tokens cost 8.046 ms
DEBUG 05-06 10:01:11.310527.310527 lmp.py:1504] decode step 16 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:11.310959.310959 lmp.py:1510] ---- decode step 16 layer 0 ----
DEBUG 05-06 10:01:11.315765.315765 cuda_h.py:27] end decode_layer cost 4.871 ms
DEBUG 05-06 10:01:11.315515.315515 lmp.py:1510] ---- decode step 16 layer 1 ----
DEBUG 05-06 10:01:11.320089.320089 cuda_h.py:27] end decode_layer cost 5.088 ms
DEBUG 05-06 10:01:11.320562.320562 lmp.py:1510] ---- decode step 16 layer 2 ----
DEBUG 05-06 10:01:11.325771.325771 cuda_h.py:27] end decode_layer cost 4.854 ms
DEBUG 05-06 10:01:11.325237.325237 lmp.py:1510] ---- decode step 16 layer 3 ----
DEBUG 05-06 10:01:11.330871.330871 cuda_h.py:27] end decode_layer cost 4.886 ms
DEBUG 05-06 10:01:11.330098.330098 lmp.py:1510] ---- decode step 16 layer 4 ----
DEBUG 05-06 10:01:11.335704.335704 cuda_h.py:27] end decode_layer cost 4.829 ms
DEBUG 05-06 10:01:11.335884.335884 lmp.py:1510] ---- decode step 16 layer 5 ----
DEBUG 05-06 10:01:11.340734.340734 cuda_h.py:27] end decode_layer cost 5.184 ms
DEBUG 05-06 10:01:11.341961.341961 lmp.py:1510] ---- decode step 16 layer 6 ----
DEBUG 05-06 10:01:11.345362.345362 cuda_h.py:27] end decode_layer cost 4.854 ms
DEBUG 05-06 10:01:11.345636.345636 lmp.py:1510] ---- decode step 16 layer 7 ----
DEBUG 05-06 10:01:11.350112.350112 cuda_h.py:27] end decode_layer cost 4.910 ms
DEBUG 05-06 10:01:11.350577.350577 lmp.py:1510] ---- decode step 16 layer 8 ----
DEBUG 05-06 10:01:11.355537.355537 cuda_h.py:27] end decode_layer cost 4.914 ms
DEBUG 05-06 10:01:11.355479.355479 lmp.py:1510] ---- decode step 16 layer 9 ----
DEBUG 05-06 10:01:11.360534.360534 cuda_h.py:27] end decode_layer cost 4.985 ms
DEBUG 05-06 10:01:11.360238.360238 lmp.py:1510] ---- decode step 16 layer 10 ----
DEBUG 05-06 10:01:11.365961.365961 cuda_h.py:27] end decode_layer cost 4.776 ms
DEBUG 05-06 10:01:11.365711.365711 lmp.py:1510] ---- decode step 16 layer 11 ----
DEBUG 05-06 10:01:11.370538.370538 cuda_h.py:27] end decode_layer cost 5.098 ms
DEBUG 05-06 10:01:11.370096.370096 lmp.py:1510] ---- decode step 16 layer 12 ----
DEBUG 05-06 10:01:11.375688.375688 cuda_h.py:27] end decode_layer cost 4.819 ms
DEBUG 05-06 10:01:11.375200.375200 lmp.py:1510] ---- decode step 16 layer 13 ----
DEBUG 05-06 10:01:11.380847.380847 cuda_h.py:27] end decode_layer cost 4.860 ms
DEBUG 05-06 10:01:11.380359.380359 lmp.py:1510] ---- decode step 16 layer 14 ----
DEBUG 05-06 10:01:11.385308.385308 cuda_h.py:27] end decode_layer cost 4.802 ms
DEBUG 05-06 10:01:11.385535.385535 lmp.py:1510] ---- decode step 16 layer 15 ----
DEBUG 05-06 10:01:11.390195.390195 cuda_h.py:27] end decode_layer cost 4.870 ms
DEBUG 05-06 10:01:11.390422.390422 lmp.py:1510] ---- decode step 16 layer 16 ----
DEBUG 05-06 10:01:11.395732.395732 cuda_h.py:27] end decode_layer cost 4.718 ms
DEBUG 05-06 10:01:11.395768.395768 lmp.py:1510] ---- decode step 16 layer 17 ----
DEBUG 05-06 10:01:11.400462.400462 cuda_h.py:27] end decode_layer cost 5.106 ms
DEBUG 05-06 10:01:11.400166.400166 lmp.py:1510] ---- decode step 16 layer 18 ----
DEBUG 05-06 10:01:11.405152.405152 cuda_h.py:27] end decode_layer cost 4.724 ms
DEBUG 05-06 10:01:11.405280.405280 lmp.py:1510] ---- decode step 16 layer 19 ----
DEBUG 05-06 10:01:11.410468.410468 cuda_h.py:27] end decode_layer cost 4.802 ms
DEBUG 05-06 10:01:11.410311.410311 lmp.py:1510] ---- decode step 16 layer 20 ----
DEBUG 05-06 10:01:11.414342.414342 cuda_h.py:27] end decode_layer cost 4.687 ms
DEBUG 05-06 10:01:11.414662.414662 lmp.py:1510] ---- decode step 16 layer 21 ----
DEBUG 05-06 10:01:11.419140.419140 cuda_h.py:27] end decode_layer cost 4.981 ms
DEBUG 05-06 10:01:11.420606.420606 lmp.py:1510] ---- decode step 16 layer 22 ----
DEBUG 05-06 10:01:11.424076.424076 cuda_h.py:27] end decode_layer cost 4.729 ms
DEBUG 05-06 10:01:11.424157.424157 lmp.py:1510] ---- decode step 16 layer 23 ----
DEBUG 05-06 10:01:11.429535.429535 cuda_h.py:27] end decode_layer cost 4.942 ms
DEBUG 05-06 10:01:11.429093.429093 lmp.py:1510] ---- decode step 16 layer 24 ----
DEBUG 05-06 10:01:11.434078.434078 cuda_h.py:27] end decode_layer cost 4.688 ms
DEBUG 05-06 10:01:11.434398.434398 lmp.py:1510] ---- decode step 16 layer 25 ----
DEBUG 05-06 10:01:11.439968.439968 cuda_h.py:27] end decode_layer cost 4.768 ms
DEBUG 05-06 10:01:11.439811.439811 lmp.py:1510] ---- decode step 16 layer 26 ----
DEBUG 05-06 10:01:11.444419.444419 cuda_h.py:27] end decode_layer cost 4.690 ms
DEBUG 05-06 10:01:11.444169.444169 lmp.py:1510] ---- decode step 16 layer 27 ----
DEBUG 05-06 10:01:11.448056.448056 cuda_h.py:27] end decode_layer cost 4.721 ms
DEBUG 05-06 10:01:11.448376.448376 lmp.py:1510] ---- decode step 16 layer 28 ----
DEBUG 05-06 10:01:11.453916.453916 cuda_h.py:27] end decode_layer cost 4.641 ms
DEBUG 05-06 10:01:11.453236.453236 lmp.py:1510] ---- decode step 16 layer 29 ----
DEBUG 05-06 10:01:11.458234.458234 cuda_h.py:27] end decode_layer cost 5.084 ms
DEBUG 05-06 10:01:11.458071.458071 cuda_h.py:27] end decode_step cost 156.089 ms
INFO 05-06 10:01:11.458741.458741 lmp.py:1558] decode step 16 time: 0.1561276912689209 seconds
WARNING 05-06 10:01:11.459568.459568 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:11.459204.459204 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:11.459324.459324 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:11.459274.459274 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:11.464206.464206 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:11.464568.464568 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:11.464291.464291 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:11.466848.466848 cuda_h.py:27] end init_inputs_tokens cost 7.451 ms
DEBUG 05-06 10:01:11.466737.466737 lmp.py:1504] decode step 17 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:11.466500.466500 lmp.py:1510] ---- decode step 17 layer 0 ----
DEBUG 05-06 10:01:11.471393.471393 cuda_h.py:27] end decode_layer cost 4.725 ms
DEBUG 05-06 10:01:11.471475.471475 lmp.py:1510] ---- decode step 17 layer 1 ----
DEBUG 05-06 10:01:11.475712.475712 cuda_h.py:27] end decode_layer cost 4.698 ms
DEBUG 05-06 10:01:11.476270.476270 lmp.py:1510] ---- decode step 17 layer 2 ----
DEBUG 05-06 10:01:11.480491.480491 cuda_h.py:27] end decode_layer cost 4.617 ms
DEBUG 05-06 10:01:11.480573.480573 lmp.py:1510] ---- decode step 17 layer 3 ----
DEBUG 05-06 10:01:11.485236.485236 cuda_h.py:27] end decode_layer cost 4.767 ms
DEBUG 05-06 10:01:11.485986.485986 lmp.py:1510] ---- decode step 17 layer 4 ----
DEBUG 05-06 10:01:11.490760.490760 cuda_h.py:27] end decode_layer cost 4.708 ms
DEBUG 05-06 10:01:11.490841.490841 lmp.py:1510] ---- decode step 17 layer 5 ----
DEBUG 05-06 10:01:11.495484.495484 cuda_h.py:27] end decode_layer cost 5.137 ms
DEBUG 05-06 10:01:11.495042.495042 lmp.py:1510] ---- decode step 17 layer 6 ----
DEBUG 05-06 10:01:11.500645.500645 cuda_h.py:27] end decode_layer cost 4.757 ms
DEBUG 05-06 10:01:11.500349.500349 lmp.py:1510] ---- decode step 17 layer 7 ----
DEBUG 05-06 10:01:11.505436.505436 cuda_h.py:27] end decode_layer cost 4.764 ms
DEBUG 05-06 10:01:11.505995.505995 lmp.py:1510] ---- decode step 17 layer 8 ----
DEBUG 05-06 10:01:11.509794.509794 cuda_h.py:27] end decode_layer cost 4.692 ms
DEBUG 05-06 10:01:11.509829.509829 lmp.py:1510] ---- decode step 17 layer 9 ----
DEBUG 05-06 10:01:11.514294.514294 cuda_h.py:27] end decode_layer cost 4.761 ms
DEBUG 05-06 10:01:11.514945.514945 lmp.py:1510] ---- decode step 17 layer 10 ----
DEBUG 05-06 10:01:11.519158.519158 cuda_h.py:27] end decode_layer cost 4.786 ms
DEBUG 05-06 10:01:11.519823.519823 lmp.py:1510] ---- decode step 17 layer 11 ----
DEBUG 05-06 10:01:11.524828.524828 cuda_h.py:27] end decode_layer cost 4.915 ms
DEBUG 05-06 10:01:11.524864.524864 lmp.py:1510] ---- decode step 17 layer 12 ----
DEBUG 05-06 10:01:11.529522.529522 cuda_h.py:27] end decode_layer cost 4.623 ms
DEBUG 05-06 10:01:11.529603.529603 lmp.py:1510] ---- decode step 17 layer 13 ----
DEBUG 05-06 10:01:11.534542.534542 cuda_h.py:27] end decode_layer cost 4.689 ms
DEBUG 05-06 10:01:11.534147.534147 lmp.py:1510] ---- decode step 17 layer 14 ----
DEBUG 05-06 10:01:11.538659.538659 cuda_h.py:27] end decode_layer cost 4.621 ms
DEBUG 05-06 10:01:11.538979.538979 lmp.py:1510] ---- decode step 17 layer 15 ----
DEBUG 05-06 10:01:11.543853.543853 cuda_h.py:27] end decode_layer cost 4.747 ms
DEBUG 05-06 10:01:11.543657.543657 lmp.py:1510] ---- decode step 17 layer 16 ----
DEBUG 05-06 10:01:11.548450.548450 cuda_h.py:27] end decode_layer cost 4.687 ms
DEBUG 05-06 10:01:11.548770.548770 lmp.py:1510] ---- decode step 17 layer 17 ----
DEBUG 05-06 10:01:11.553940.553940 cuda_h.py:27] end decode_layer cost 4.895 ms
DEBUG 05-06 10:01:11.553784.553784 lmp.py:1510] ---- decode step 17 layer 18 ----
DEBUG 05-06 10:01:11.558287.558287 cuda_h.py:27] end decode_layer cost 4.755 ms
DEBUG 05-06 10:01:11.558753.558753 lmp.py:1510] ---- decode step 17 layer 19 ----
DEBUG 05-06 10:01:11.562176.562176 cuda_h.py:27] end decode_layer cost 4.731 ms
DEBUG 05-06 10:01:11.562973.562973 lmp.py:1510] ---- decode step 17 layer 20 ----
DEBUG 05-06 10:01:11.567413.567413 cuda_h.py:27] end decode_layer cost 4.637 ms
DEBUG 05-06 10:01:11.567985.567985 lmp.py:1510] ---- decode step 17 layer 21 ----
DEBUG 05-06 10:01:11.572700.572700 cuda_h.py:27] end decode_layer cost 4.735 ms
DEBUG 05-06 10:01:11.572736.572736 lmp.py:1510] ---- decode step 17 layer 22 ----
DEBUG 05-06 10:01:11.577188.577188 cuda_h.py:27] end decode_layer cost 4.612 ms
DEBUG 05-06 10:01:11.577078.577078 lmp.py:1510] ---- decode step 17 layer 23 ----
DEBUG 05-06 10:01:11.582666.582666 cuda_h.py:27] end decode_layer cost 4.923 ms
DEBUG 05-06 10:01:11.582986.582986 lmp.py:1510] ---- decode step 17 layer 24 ----
DEBUG 05-06 10:01:11.586639.586639 cuda_h.py:27] end decode_layer cost 4.653 ms
DEBUG 05-06 10:01:11.586628.586628 lmp.py:1510] ---- decode step 17 layer 25 ----
DEBUG 05-06 10:01:11.591231.591231 cuda_h.py:27] end decode_layer cost 4.757 ms
DEBUG 05-06 10:01:11.591870.591870 lmp.py:1510] ---- decode step 17 layer 26 ----
DEBUG 05-06 10:01:11.596141.596141 cuda_h.py:27] end decode_layer cost 4.724 ms
DEBUG 05-06 10:01:11.596129.596129 lmp.py:1510] ---- decode step 17 layer 27 ----
DEBUG 05-06 10:01:11.601959.601959 cuda_h.py:27] end decode_layer cost 4.784 ms
DEBUG 05-06 10:01:11.601802.601802 lmp.py:1510] ---- decode step 17 layer 28 ----
DEBUG 05-06 10:01:11.605627.605627 cuda_h.py:27] end decode_layer cost 4.675 ms
DEBUG 05-06 10:01:11.606755.606755 lmp.py:1510] ---- decode step 17 layer 29 ----
DEBUG 05-06 10:01:11.610727.610727 cuda_h.py:27] end decode_layer cost 4.889 ms
DEBUG 05-06 10:01:11.611412.611412 cuda_h.py:27] end decode_step cost 152.078 ms
INFO 05-06 10:01:11.611029.611029 lmp.py:1558] decode step 17 time: 0.1521143913269043 seconds
WARNING 05-06 10:01:11.611305.611305 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:11.611065.611065 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:11.611371.611371 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:11.612635.612635 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:11.617945.617945 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:11.617816.617816 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:11.617493.617493 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:11.618811.618811 cuda_h.py:27] end init_inputs_tokens cost 7.681 ms
DEBUG 05-06 10:01:11.618131.618131 lmp.py:1504] decode step 18 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:11.618563.618563 lmp.py:1510] ---- decode step 18 layer 0 ----
DEBUG 05-06 10:01:11.623044.623044 cuda_h.py:27] end decode_layer cost 4.667 ms
DEBUG 05-06 10:01:11.623516.623516 lmp.py:1510] ---- decode step 18 layer 1 ----
DEBUG 05-06 10:01:11.628031.628031 cuda_h.py:27] end decode_layer cost 4.693 ms
DEBUG 05-06 10:01:11.628590.628590 lmp.py:1510] ---- decode step 18 layer 2 ----
DEBUG 05-06 10:01:11.633572.633572 cuda_h.py:27] end decode_layer cost 4.617 ms
DEBUG 05-06 10:01:11.633177.633177 lmp.py:1510] ---- decode step 18 layer 3 ----
DEBUG 05-06 10:01:11.637380.637380 cuda_h.py:27] end decode_layer cost 4.673 ms
DEBUG 05-06 10:01:11.637746.637746 lmp.py:1510] ---- decode step 18 layer 4 ----
DEBUG 05-06 10:01:11.642303.642303 cuda_h.py:27] end decode_layer cost 4.758 ms
DEBUG 05-06 10:01:11.642444.642444 lmp.py:1510] ---- decode step 18 layer 5 ----
DEBUG 05-06 10:01:11.647465.647465 cuda_h.py:27] end decode_layer cost 5.171 ms
DEBUG 05-06 10:01:11.647885.647885 lmp.py:1510] ---- decode step 18 layer 6 ----
DEBUG 05-06 10:01:11.652922.652922 cuda_h.py:27] end decode_layer cost 4.867 ms
DEBUG 05-06 10:01:11.652832.652832 lmp.py:1510] ---- decode step 18 layer 7 ----
DEBUG 05-06 10:01:11.657055.657055 cuda_h.py:27] end decode_layer cost 4.863 ms
DEBUG 05-06 10:01:11.657043.657043 lmp.py:1510] ---- decode step 18 layer 8 ----
DEBUG 05-06 10:01:11.662126.662126 cuda_h.py:27] end decode_layer cost 4.830 ms
DEBUG 05-06 10:01:11.662591.662591 lmp.py:1510] ---- decode step 18 layer 9 ----
DEBUG 05-06 10:01:11.667942.667942 cuda_h.py:27] end decode_layer cost 4.923 ms
DEBUG 05-06 10:01:11.667908.667908 lmp.py:1510] ---- decode step 18 layer 10 ----
DEBUG 05-06 10:01:11.672930.672930 cuda_h.py:27] end decode_layer cost 4.999 ms
DEBUG 05-06 10:01:11.672157.672157 lmp.py:1510] ---- decode step 18 layer 11 ----
DEBUG 05-06 10:01:11.677705.677705 cuda_h.py:27] end decode_layer cost 5.068 ms
DEBUG 05-06 10:01:11.677740.677740 lmp.py:1510] ---- decode step 18 layer 12 ----
DEBUG 05-06 10:01:11.682230.682230 cuda_h.py:27] end decode_layer cost 4.745 ms
DEBUG 05-06 10:01:11.682504.682504 lmp.py:1510] ---- decode step 18 layer 13 ----
DEBUG 05-06 10:01:11.687431.687431 cuda_h.py:27] end decode_layer cost 4.751 ms
DEBUG 05-06 10:01:11.687573.687573 lmp.py:1510] ---- decode step 18 layer 14 ----
DEBUG 05-06 10:01:11.692554.692554 cuda_h.py:27] end decode_layer cost 4.798 ms
DEBUG 05-06 10:01:11.692020.692020 lmp.py:1510] ---- decode step 18 layer 15 ----
DEBUG 05-06 10:01:11.697876.697876 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 10:01:11.697150.697150 lmp.py:1510] ---- decode step 18 layer 16 ----
DEBUG 05-06 10:01:11.702837.702837 cuda_h.py:27] end decode_layer cost 4.679 ms
DEBUG 05-06 10:01:11.702872.702872 lmp.py:1510] ---- decode step 18 layer 17 ----
DEBUG 05-06 10:01:11.707747.707747 cuda_h.py:27] end decode_layer cost 4.958 ms
DEBUG 05-06 10:01:11.707305.707305 lmp.py:1510] ---- decode step 18 layer 18 ----
DEBUG 05-06 10:01:11.711113.711113 cuda_h.py:27] end decode_layer cost 4.733 ms
DEBUG 05-06 10:01:11.711578.711578 lmp.py:1510] ---- decode step 18 layer 19 ----
DEBUG 05-06 10:01:11.716051.716051 cuda_h.py:27] end decode_layer cost 4.801 ms
DEBUG 05-06 10:01:11.716278.716278 lmp.py:1510] ---- decode step 18 layer 20 ----
DEBUG 05-06 10:01:11.721370.721370 cuda_h.py:27] end decode_layer cost 4.732 ms
DEBUG 05-06 10:01:11.721213.721213 lmp.py:1510] ---- decode step 18 layer 21 ----
DEBUG 05-06 10:01:11.726287.726287 cuda_h.py:27] end decode_layer cost 4.753 ms
DEBUG 05-06 10:01:11.726229.726229 lmp.py:1510] ---- decode step 18 layer 22 ----
DEBUG 05-06 10:01:11.731296.731296 cuda_h.py:27] end decode_layer cost 4.749 ms
DEBUG 05-06 10:01:11.731854.731854 lmp.py:1510] ---- decode step 18 layer 23 ----
DEBUG 05-06 10:01:11.736934.736934 cuda_h.py:27] end decode_layer cost 4.969 ms
DEBUG 05-06 10:01:11.736638.736638 lmp.py:1510] ---- decode step 18 layer 24 ----
DEBUG 05-06 10:01:11.741521.741521 cuda_h.py:27] end decode_layer cost 4.788 ms
DEBUG 05-06 10:01:11.741125.741125 lmp.py:1510] ---- decode step 18 layer 25 ----
DEBUG 05-06 10:01:11.745000.745000 cuda_h.py:27] end decode_layer cost 4.747 ms
DEBUG 05-06 10:01:11.745750.745750 lmp.py:1510] ---- decode step 18 layer 26 ----
DEBUG 05-06 10:01:11.750791.750791 cuda_h.py:27] end decode_layer cost 4.764 ms
DEBUG 05-06 10:01:11.750872.750872 lmp.py:1510] ---- decode step 18 layer 27 ----
DEBUG 05-06 10:01:11.755335.755335 cuda_h.py:27] end decode_layer cost 4.724 ms
DEBUG 05-06 10:01:11.755086.755086 lmp.py:1510] ---- decode step 18 layer 28 ----
DEBUG 05-06 10:01:11.760690.760690 cuda_h.py:27] end decode_layer cost 4.794 ms
DEBUG 05-06 10:01:11.760394.760394 lmp.py:1510] ---- decode step 18 layer 29 ----
DEBUG 05-06 10:01:11.765252.765252 cuda_h.py:27] end decode_layer cost 5.051 ms
DEBUG 05-06 10:01:11.765374.765374 cuda_h.py:27] end decode_step cost 154.449 ms
INFO 05-06 10:01:11.765137.765137 lmp.py:1558] decode step 18 time: 0.15448737144470215 seconds
WARNING 05-06 10:01:11.765705.765705 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:11.766195.766195 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:11.766444.766444 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:11.766302.766302 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:11.771232.771232 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:11.771164.771164 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:11.771556.771556 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 10:01:11.771436.771436 helper.py:80] WARNING: Logits have extreme values: min=-824.00, max=1024.00
WARNING 05-06 10:01:11.772986.772986 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 10:01:11.773484.773484 cuda_h.py:27] end init_inputs_tokens cost 7.920 ms
DEBUG 05-06 10:01:11.773281.773281 lmp.py:1504] decode step 19 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:11.773236.773236 lmp.py:1510] ---- decode step 19 layer 0 ----
DEBUG 05-06 10:01:11.778337.778337 cuda_h.py:27] end decode_layer cost 4.808 ms
DEBUG 05-06 10:01:11.778088.778088 lmp.py:1510] ---- decode step 19 layer 1 ----
DEBUG 05-06 10:01:11.783268.783268 cuda_h.py:27] end decode_layer cost 4.798 ms
DEBUG 05-06 10:01:11.783496.783496 lmp.py:1510] ---- decode step 19 layer 2 ----
DEBUG 05-06 10:01:11.788603.788603 cuda_h.py:27] end decode_layer cost 4.778 ms
DEBUG 05-06 10:01:11.788068.788068 lmp.py:1510] ---- decode step 19 layer 3 ----
DEBUG 05-06 10:01:11.793215.793215 cuda_h.py:27] end decode_layer cost 4.773 ms
DEBUG 05-06 10:01:11.793773.793773 lmp.py:1510] ---- decode step 19 layer 4 ----
DEBUG 05-06 10:01:11.797277.797277 cuda_h.py:27] end decode_layer cost 4.754 ms
DEBUG 05-06 10:01:11.797074.797074 lmp.py:1510] ---- decode step 19 layer 5 ----
DEBUG 05-06 10:01:11.802465.802465 cuda_h.py:27] end decode_layer cost 4.952 ms
DEBUG 05-06 10:01:11.802931.802931 lmp.py:1510] ---- decode step 19 layer 6 ----
DEBUG 05-06 10:01:11.807135.807135 cuda_h.py:27] end decode_layer cost 4.693 ms
DEBUG 05-06 10:01:11.807647.807647 lmp.py:1510] ---- decode step 19 layer 7 ----
DEBUG 05-06 10:01:11.812515.812515 cuda_h.py:27] end decode_layer cost 4.742 ms
DEBUG 05-06 10:01:11.812265.812265 lmp.py:1510] ---- decode step 19 layer 8 ----
DEBUG 05-06 10:01:11.817404.817404 cuda_h.py:27] end decode_layer cost 4.732 ms
DEBUG 05-06 10:01:11.817962.817962 lmp.py:1510] ---- decode step 19 layer 9 ----
DEBUG 05-06 10:01:11.822162.822162 cuda_h.py:27] end decode_layer cost 4.777 ms
DEBUG 05-06 10:01:11.822436.822436 lmp.py:1510] ---- decode step 19 layer 10 ----
DEBUG 05-06 10:01:11.826094.826094 cuda_h.py:27] end decode_layer cost 4.623 ms
DEBUG 05-06 10:01:11.826414.826414 lmp.py:1510] ---- decode step 19 layer 11 ----
DEBUG 05-06 10:01:11.831723.831723 cuda_h.py:27] end decode_layer cost 4.892 ms
DEBUG 05-06 10:01:11.831567.831567 lmp.py:1510] ---- decode step 19 layer 12 ----
DEBUG 05-06 10:01:11.836642.836642 cuda_h.py:27] end decode_layer cost 4.614 ms
DEBUG 05-06 10:01:11.836200.836200 lmp.py:1510] ---- decode step 19 layer 13 ----
DEBUG 05-06 10:01:11.841423.841423 cuda_h.py:27] end decode_layer cost 4.689 ms
DEBUG 05-06 10:01:11.841790.841790 lmp.py:1510] ---- decode step 19 layer 14 ----
DEBUG 05-06 10:01:11.846663.846663 cuda_h.py:27] end decode_layer cost 5.692 ms
DEBUG 05-06 10:01:11.847109.847109 lmp.py:1510] ---- decode step 19 layer 15 ----
DEBUG 05-06 10:01:11.851350.851350 cuda_h.py:27] end decode_layer cost 4.807 ms
DEBUG 05-06 10:01:11.851292.851292 lmp.py:1510] ---- decode step 19 layer 16 ----
DEBUG 05-06 10:01:11.856810.856810 cuda_h.py:27] end decode_layer cost 4.765 ms
DEBUG 05-06 10:01:11.856467.856467 lmp.py:1510] ---- decode step 19 layer 17 ----
DEBUG 05-06 10:01:11.861080.861080 cuda_h.py:27] end decode_layer cost 5.046 ms
DEBUG 05-06 10:01:11.861877.861877 lmp.py:1510] ---- decode step 19 layer 18 ----
DEBUG 05-06 10:01:11.866334.866334 cuda_h.py:27] end decode_layer cost 4.755 ms
DEBUG 05-06 10:01:11.866039.866039 lmp.py:1510] ---- decode step 19 layer 19 ----
DEBUG 05-06 10:01:11.871373.871373 cuda_h.py:27] end decode_layer cost 4.840 ms
DEBUG 05-06 10:01:11.871170.871170 lmp.py:1510] ---- decode step 19 layer 20 ----
DEBUG 05-06 10:01:11.876308.876308 cuda_h.py:27] end decode_layer cost 4.731 ms
DEBUG 05-06 10:01:11.876251.876251 lmp.py:1510] ---- decode step 19 layer 21 ----
DEBUG 05-06 10:01:11.881991.881991 cuda_h.py:27] end decode_layer cost 4.894 ms
DEBUG 05-06 10:01:11.881934.881934 lmp.py:1510] ---- decode step 19 layer 22 ----
DEBUG 05-06 10:01:11.885964.885964 cuda_h.py:27] end decode_layer cost 4.651 ms
DEBUG 05-06 10:01:11.886522.886522 lmp.py:1510] ---- decode step 19 layer 23 ----
DEBUG 05-06 10:01:11.891997.891997 cuda_h.py:27] end decode_layer cost 5.085 ms
DEBUG 05-06 10:01:11.891794.891794 lmp.py:1510] ---- decode step 19 layer 24 ----
DEBUG 05-06 10:01:11.895211.895211 cuda_h.py:27] end decode_layer cost 4.725 ms
DEBUG 05-06 10:01:11.896199.896199 lmp.py:1510] ---- decode step 19 layer 25 ----
DEBUG 05-06 10:01:11.900956.900956 cuda_h.py:27] end decode_layer cost 4.801 ms
DEBUG 05-06 10:01:11.900959.900959 lmp.py:1510] ---- decode step 19 layer 26 ----
DEBUG 05-06 10:01:11.905235.905235 cuda_h.py:27] end decode_layer cost 4.692 ms
DEBUG 05-06 10:01:11.905747.905747 lmp.py:1510] ---- decode step 19 layer 27 ----
DEBUG 05-06 10:01:11.910393.910393 cuda_h.py:27] end decode_layer cost 4.824 ms
DEBUG 05-06 10:01:11.910236.910236 lmp.py:1510] ---- decode step 19 layer 28 ----
DEBUG 05-06 10:01:11.915148.915148 cuda_h.py:27] end decode_layer cost 4.705 ms
DEBUG 05-06 10:01:11.915707.915707 lmp.py:1510] ---- decode step 19 layer 29 ----
DEBUG 05-06 10:01:11.920488.920488 cuda_h.py:27] end decode_layer cost 5.135 ms
DEBUG 05-06 10:01:11.920279.920279 cuda_h.py:27] end decode_step cost 154.895 ms
INFO 05-06 10:01:11.920757.920757 lmp.py:1558] decode step 19 time: 0.15493345260620117 seconds
WARNING 05-06 10:01:11.920239.920239 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:11.920591.920591 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:11.921073.921073 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:11.921070.921070 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:11.926478.926478 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:11.926396.926396 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:11.926927.926927 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:11.928677.928677 cuda_h.py:27] end init_inputs_tokens cost 7.427 ms
DEBUG 05-06 10:01:11.928235.928235 lmp.py:1504] decode step 20 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:11.928528.928528 lmp.py:1510] ---- decode step 20 layer 0 ----
DEBUG 05-06 10:01:11.933348.933348 cuda_h.py:27] end decode_layer cost 4.917 ms
DEBUG 05-06 10:01:11.933622.933622 lmp.py:1510] ---- decode step 20 layer 1 ----
DEBUG 05-06 10:01:11.937086.937086 cuda_h.py:27] end decode_layer cost 4.761 ms
DEBUG 05-06 10:01:11.937645.937645 lmp.py:1510] ---- decode step 20 layer 2 ----
DEBUG 05-06 10:01:11.942809.942809 cuda_h.py:27] end decode_layer cost 4.715 ms
DEBUG 05-06 10:01:11.942891.942891 lmp.py:1510] ---- decode step 20 layer 3 ----
DEBUG 05-06 10:01:11.947706.947706 cuda_h.py:27] end decode_layer cost 4.774 ms
DEBUG 05-06 10:01:11.947788.947788 lmp.py:1510] ---- decode step 20 layer 4 ----
DEBUG 05-06 10:01:11.952890.952890 cuda_h.py:27] end decode_layer cost 4.809 ms
DEBUG 05-06 10:01:11.952594.952594 lmp.py:1510] ---- decode step 20 layer 5 ----
DEBUG 05-06 10:01:11.957919.957919 cuda_h.py:27] end decode_layer cost 5.150 ms
DEBUG 05-06 10:01:11.957960.957960 lmp.py:1510] ---- decode step 20 layer 6 ----
DEBUG 05-06 10:01:11.962577.962577 cuda_h.py:27] end decode_layer cost 4.768 ms
DEBUG 05-06 10:01:11.962089.962089 lmp.py:1510] ---- decode step 20 layer 7 ----
DEBUG 05-06 10:01:11.967636.967636 cuda_h.py:27] end decode_layer cost 4.646 ms
DEBUG 05-06 10:01:11.967717.967717 lmp.py:1510] ---- decode step 20 layer 8 ----
DEBUG 05-06 10:01:11.971458.971458 cuda_h.py:27] end decode_layer cost 4.719 ms
DEBUG 05-06 10:01:11.972063.972063 lmp.py:1510] ---- decode step 20 layer 9 ----
DEBUG 05-06 10:01:11.976172.976172 cuda_h.py:27] end decode_layer cost 4.850 ms
DEBUG 05-06 10:01:11.976446.976446 lmp.py:1510] ---- decode step 20 layer 10 ----
DEBUG 05-06 10:01:11.981263.981263 cuda_h.py:27] end decode_layer cost 4.810 ms
DEBUG 05-06 10:01:11.981537.981537 lmp.py:1510] ---- decode step 20 layer 11 ----
DEBUG 05-06 10:01:11.986752.986752 cuda_h.py:27] end decode_layer cost 5.034 ms
DEBUG 05-06 10:01:11.986740.986740 lmp.py:1510] ---- decode step 20 layer 12 ----
DEBUG 05-06 10:01:11.991747.991747 cuda_h.py:27] end decode_layer cost 4.739 ms
DEBUG 05-06 10:01:11.991451.991451 lmp.py:1510] ---- decode step 20 layer 13 ----
DEBUG 05-06 10:01:11.996022.996022 cuda_h.py:27] end decode_layer cost 4.768 ms
DEBUG 05-06 10:01:11.996726.996726 lmp.py:1510] ---- decode step 20 layer 14 ----
DEBUG 05-06 10:01:12.001807.001807 cuda_h.py:27] end decode_layer cost 4.795 ms
DEBUG 05-06 10:01:12.001557.001557 lmp.py:1510] ---- decode step 20 layer 15 ----
DEBUG 05-06 10:01:12.006234.006234 cuda_h.py:27] end decode_layer cost 4.777 ms
DEBUG 05-06 10:01:12.006031.006031 lmp.py:1510] ---- decode step 20 layer 16 ----
DEBUG 05-06 10:01:12.010334.010334 cuda_h.py:27] end decode_layer cost 4.713 ms
DEBUG 05-06 10:01:12.011131.011131 lmp.py:1510] ---- decode step 20 layer 17 ----
DEBUG 05-06 10:01:12.016311.016311 cuda_h.py:27] end decode_layer cost 4.972 ms
DEBUG 05-06 10:01:12.016969.016969 lmp.py:1510] ---- decode step 20 layer 18 ----
DEBUG 05-06 10:01:12.021127.021127 cuda_h.py:27] end decode_layer cost 5.666 ms
DEBUG 05-06 10:01:12.021852.021852 lmp.py:1510] ---- decode step 20 layer 19 ----
DEBUG 05-06 10:01:12.026440.026440 cuda_h.py:27] end decode_layer cost 4.888 ms
DEBUG 05-06 10:01:12.026998.026998 lmp.py:1510] ---- decode step 20 layer 20 ----
DEBUG 05-06 10:01:12.031531.031531 cuda_h.py:27] end decode_layer cost 4.636 ms
DEBUG 05-06 10:01:12.031566.031566 lmp.py:1510] ---- decode step 20 layer 21 ----
DEBUG 05-06 10:01:12.036412.036412 cuda_h.py:27] end decode_layer cost 4.691 ms
DEBUG 05-06 10:01:12.036732.036732 lmp.py:1510] ---- decode step 20 layer 22 ----
DEBUG 05-06 10:01:12.041558.041558 cuda_h.py:27] end decode_layer cost 4.675 ms
DEBUG 05-06 10:01:12.041739.041739 lmp.py:1510] ---- decode step 20 layer 23 ----
DEBUG 05-06 10:01:12.046939.046939 cuda_h.py:27] end decode_layer cost 4.987 ms
DEBUG 05-06 10:01:12.046451.046451 lmp.py:1510] ---- decode step 20 layer 24 ----
DEBUG 05-06 10:01:12.050561.050561 cuda_h.py:27] end decode_layer cost 4.675 ms
DEBUG 05-06 10:01:12.050835.050835 lmp.py:1510] ---- decode step 20 layer 25 ----
DEBUG 05-06 10:01:12.055244.055244 cuda_h.py:27] end decode_layer cost 4.684 ms
DEBUG 05-06 10:01:12.055577.055577 lmp.py:1510] ---- decode step 20 layer 26 ----
DEBUG 05-06 10:01:12.060936.060936 cuda_h.py:27] end decode_layer cost 4.789 ms
DEBUG 05-06 10:01:12.060972.060972 lmp.py:1510] ---- decode step 20 layer 27 ----
DEBUG 05-06 10:01:12.065651.065651 cuda_h.py:27] end decode_layer cost 4.674 ms
DEBUG 05-06 10:01:12.065733.065733 lmp.py:1510] ---- decode step 20 layer 28 ----
DEBUG 05-06 10:01:12.070006.070006 cuda_h.py:27] end decode_layer cost 4.795 ms
DEBUG 05-06 10:01:12.070233.070233 lmp.py:1510] ---- decode step 20 layer 29 ----
DEBUG 05-06 10:01:12.075921.075921 cuda_h.py:27] end decode_layer cost 5.101 ms
DEBUG 05-06 10:01:12.075182.075182 cuda_h.py:27] end decode_step cost 154.644 ms
INFO 05-06 10:01:12.075091.075091 lmp.py:1558] decode step 20 time: 0.15468120574951172 seconds
WARNING 05-06 10:01:12.075341.075341 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:12.075121.075121 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:12.076339.076339 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:12.076581.076581 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:12.081943.081943 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:12.081199.081199 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:12.081829.081829 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:12.082109.082109 cuda_h.py:27] end init_inputs_tokens cost 7.607 ms
DEBUG 05-06 10:01:12.083760.083760 lmp.py:1504] decode step 21 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:12.083284.083284 lmp.py:1510] ---- decode step 21 layer 0 ----
DEBUG 05-06 10:01:12.087351.087351 cuda_h.py:27] end decode_layer cost 4.746 ms
DEBUG 05-06 10:01:12.087817.087817 lmp.py:1510] ---- decode step 21 layer 1 ----
DEBUG 05-06 10:01:12.092449.092449 cuda_h.py:27] end decode_layer cost 4.850 ms
DEBUG 05-06 10:01:12.092438.092438 lmp.py:1510] ---- decode step 21 layer 2 ----
DEBUG 05-06 10:01:12.097112.097112 cuda_h.py:27] end decode_layer cost 4.705 ms
DEBUG 05-06 10:01:12.097909.097909 lmp.py:1510] ---- decode step 21 layer 3 ----
DEBUG 05-06 10:01:12.102078.102078 cuda_h.py:27] end decode_layer cost 4.823 ms
DEBUG 05-06 10:01:12.102159.102159 lmp.py:1510] ---- decode step 21 layer 4 ----
DEBUG 05-06 10:01:12.107784.107784 cuda_h.py:27] end decode_layer cost 4.809 ms
DEBUG 05-06 10:01:12.107342.107342 lmp.py:1510] ---- decode step 21 layer 5 ----
DEBUG 05-06 10:01:12.112830.112830 cuda_h.py:27] end decode_layer cost 5.058 ms
DEBUG 05-06 10:01:12.112818.112818 lmp.py:1510] ---- decode step 21 layer 6 ----
DEBUG 05-06 10:01:12.117567.117567 cuda_h.py:27] end decode_layer cost 4.760 ms
DEBUG 05-06 10:01:12.117126.117126 lmp.py:1510] ---- decode step 21 layer 7 ----
DEBUG 05-06 10:01:12.122752.122752 cuda_h.py:27] end decode_layer cost 4.844 ms
DEBUG 05-06 10:01:12.122264.122264 lmp.py:1510] ---- decode step 21 layer 8 ----
DEBUG 05-06 10:01:12.126514.126514 cuda_h.py:27] end decode_layer cost 4.708 ms
DEBUG 05-06 10:01:12.126119.126119 lmp.py:1510] ---- decode step 21 layer 9 ----
DEBUG 05-06 10:01:12.131308.131308 cuda_h.py:27] end decode_layer cost 4.838 ms
DEBUG 05-06 10:01:12.131965.131965 lmp.py:1510] ---- decode step 21 layer 10 ----
DEBUG 05-06 10:01:12.136336.136336 cuda_h.py:27] end decode_layer cost 4.727 ms
DEBUG 05-06 10:01:12.136801.136801 lmp.py:1510] ---- decode step 21 layer 11 ----
DEBUG 05-06 10:01:12.141574.141574 cuda_h.py:27] end decode_layer cost 5.058 ms
DEBUG 05-06 10:01:12.141324.141324 lmp.py:1510] ---- decode step 21 layer 12 ----
DEBUG 05-06 10:01:12.146028.146028 cuda_h.py:27] end decode_layer cost 4.797 ms
DEBUG 05-06 10:01:12.146063.146063 lmp.py:1510] ---- decode step 21 layer 13 ----
DEBUG 05-06 10:01:12.151587.151587 cuda_h.py:27] end decode_layer cost 4.770 ms
DEBUG 05-06 10:01:12.151291.151291 lmp.py:1510] ---- decode step 21 layer 14 ----
DEBUG 05-06 10:01:12.156983.156983 cuda_h.py:27] end decode_layer cost 4.822 ms
DEBUG 05-06 10:01:12.156117.156117 lmp.py:1510] ---- decode step 21 layer 15 ----
DEBUG 05-06 10:01:12.161340.161340 cuda_h.py:27] end decode_layer cost 4.864 ms
DEBUG 05-06 10:01:12.161375.161375 lmp.py:1510] ---- decode step 21 layer 16 ----
DEBUG 05-06 10:01:12.166575.166575 cuda_h.py:27] end decode_layer cost 4.776 ms
DEBUG 05-06 10:01:12.166041.166041 lmp.py:1510] ---- decode step 21 layer 17 ----
DEBUG 05-06 10:01:12.171277.171277 cuda_h.py:27] end decode_layer cost 5.084 ms
DEBUG 05-06 10:01:12.171981.171981 lmp.py:1510] ---- decode step 21 layer 18 ----
DEBUG 05-06 10:01:12.176341.176341 cuda_h.py:27] end decode_layer cost 4.788 ms
DEBUG 05-06 10:01:12.176806.176806 lmp.py:1510] ---- decode step 21 layer 19 ----
DEBUG 05-06 10:01:12.180874.180874 cuda_h.py:27] end decode_layer cost 4.785 ms
DEBUG 05-06 10:01:12.181770.181770 lmp.py:1510] ---- decode step 21 layer 20 ----
DEBUG 05-06 10:01:12.185758.185758 cuda_h.py:27] end decode_layer cost 4.761 ms
DEBUG 05-06 10:01:12.185077.185077 lmp.py:1510] ---- decode step 21 layer 21 ----
DEBUG 05-06 10:01:12.190206.190206 cuda_h.py:27] end decode_layer cost 4.829 ms
DEBUG 05-06 10:01:12.190288.190288 lmp.py:1510] ---- decode step 21 layer 22 ----
DEBUG 05-06 10:01:12.196945.196945 cuda_h.py:27] end decode_layer cost 5.378 ms
DEBUG 05-06 10:01:12.196121.196121 lmp.py:1510] ---- decode step 21 layer 23 ----
DEBUG 05-06 10:01:12.201953.201953 cuda_h.py:27] end decode_layer cost 5.666 ms
DEBUG 05-06 10:01:12.201180.201180 lmp.py:1510] ---- decode step 21 layer 24 ----
DEBUG 05-06 10:01:12.206856.206856 cuda_h.py:27] end decode_layer cost 4.952 ms
DEBUG 05-06 10:01:12.207799.207799 lmp.py:1510] ---- decode step 21 layer 25 ----
DEBUG 05-06 10:01:12.212184.212184 cuda_h.py:27] end decode_layer cost 4.983 ms
DEBUG 05-06 10:01:12.212127.212127 lmp.py:1510] ---- decode step 21 layer 26 ----
DEBUG 05-06 10:01:12.216709.216709 cuda_h.py:27] end decode_layer cost 4.918 ms
DEBUG 05-06 10:01:12.217605.217605 lmp.py:1510] ---- decode step 21 layer 27 ----
DEBUG 05-06 10:01:12.221784.221784 cuda_h.py:27] end decode_layer cost 4.936 ms
DEBUG 05-06 10:01:12.222249.222249 lmp.py:1510] ---- decode step 21 layer 28 ----
DEBUG 05-06 10:01:12.226625.226625 cuda_h.py:27] end decode_layer cost 4.870 ms
DEBUG 05-06 10:01:12.226329.226329 lmp.py:1510] ---- decode step 21 layer 29 ----
DEBUG 05-06 10:01:12.232076.232076 cuda_h.py:27] end decode_layer cost 5.110 ms
DEBUG 05-06 10:01:12.232636.232636 cuda_h.py:27] end decode_step cost 156.840 ms
INFO 05-06 10:01:12.232591.232591 lmp.py:1558] decode step 21 time: 0.15687990188598633 seconds
WARNING 05-06 10:01:12.232706.232706 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:12.232104.232104 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:12.233150.233150 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:12.233014.233014 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:12.238007.238007 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:12.238647.238647 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:12.238562.238562 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:12.240000.240000 cuda_h.py:27] end init_inputs_tokens cost 7.697 ms
DEBUG 05-06 10:01:12.240988.240988 lmp.py:1504] decode step 22 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:12.240612.240612 lmp.py:1510] ---- decode step 22 layer 0 ----
DEBUG 05-06 10:01:12.245183.245183 cuda_h.py:27] end decode_layer cost 4.978 ms
DEBUG 05-06 10:01:12.245602.245602 lmp.py:1510] ---- decode step 22 layer 1 ----
DEBUG 05-06 10:01:12.250107.250107 cuda_h.py:27] end decode_layer cost 4.966 ms
DEBUG 05-06 10:01:12.250526.250526 lmp.py:1510] ---- decode step 22 layer 2 ----
DEBUG 05-06 10:01:12.255438.255438 cuda_h.py:27] end decode_layer cost 4.880 ms
DEBUG 05-06 10:01:12.255857.255857 lmp.py:1510] ---- decode step 22 layer 3 ----
DEBUG 05-06 10:01:12.260387.260387 cuda_h.py:27] end decode_layer cost 4.949 ms
DEBUG 05-06 10:01:12.260615.260615 lmp.py:1510] ---- decode step 22 layer 4 ----
DEBUG 05-06 10:01:12.265452.265452 cuda_h.py:27] end decode_layer cost 4.825 ms
DEBUG 05-06 10:01:12.265441.265441 lmp.py:1510] ---- decode step 22 layer 5 ----
DEBUG 05-06 10:01:12.270707.270707 cuda_h.py:27] end decode_layer cost 5.176 ms
DEBUG 05-06 10:01:12.270842.270842 lmp.py:1510] ---- decode step 22 layer 6 ----
DEBUG 05-06 10:01:12.275849.275849 cuda_h.py:27] end decode_layer cost 4.950 ms
DEBUG 05-06 10:01:12.275420.275420 lmp.py:1510] ---- decode step 22 layer 7 ----
DEBUG 05-06 10:01:12.280763.280763 cuda_h.py:27] end decode_layer cost 4.881 ms
DEBUG 05-06 10:01:12.280229.280229 lmp.py:1510] ---- decode step 22 layer 8 ----
DEBUG 05-06 10:01:12.285155.285155 cuda_h.py:27] end decode_layer cost 4.926 ms
DEBUG 05-06 10:01:12.285105.285105 lmp.py:1510] ---- decode step 22 layer 9 ----
DEBUG 05-06 10:01:12.290787.290787 cuda_h.py:27] end decode_layer cost 4.956 ms
DEBUG 05-06 10:01:12.290253.290253 lmp.py:1510] ---- decode step 22 layer 10 ----
DEBUG 05-06 10:01:12.295820.295820 cuda_h.py:27] end decode_layer cost 4.871 ms
DEBUG 05-06 10:01:12.295763.295763 lmp.py:1510] ---- decode step 22 layer 11 ----
DEBUG 05-06 10:01:12.300752.300752 cuda_h.py:27] end decode_layer cost 5.218 ms
DEBUG 05-06 10:01:12.300648.300648 lmp.py:1510] ---- decode step 22 layer 12 ----
DEBUG 05-06 10:01:12.305800.305800 cuda_h.py:27] end decode_layer cost 4.916 ms
DEBUG 05-06 10:01:12.305981.305981 lmp.py:1510] ---- decode step 22 layer 13 ----
DEBUG 05-06 10:01:12.310902.310902 cuda_h.py:27] end decode_layer cost 4.957 ms
DEBUG 05-06 10:01:12.310129.310129 lmp.py:1510] ---- decode step 22 layer 14 ----
DEBUG 05-06 10:01:12.315009.315009 cuda_h.py:27] end decode_layer cost 4.891 ms
DEBUG 05-06 10:01:12.315474.315474 lmp.py:1510] ---- decode step 22 layer 15 ----
DEBUG 05-06 10:01:12.320319.320319 cuda_h.py:27] end decode_layer cost 4.865 ms
DEBUG 05-06 10:01:12.320831.320831 lmp.py:1510] ---- decode step 22 layer 16 ----
DEBUG 05-06 10:01:12.325537.325537 cuda_h.py:27] end decode_layer cost 4.833 ms
DEBUG 05-06 10:01:12.325049.325049 lmp.py:1510] ---- decode step 22 layer 17 ----
DEBUG 05-06 10:01:12.330428.330428 cuda_h.py:27] end decode_layer cost 5.187 ms
DEBUG 05-06 10:01:12.330530.330530 lmp.py:1510] ---- decode step 22 layer 18 ----
DEBUG 05-06 10:01:12.335291.335291 cuda_h.py:27] end decode_layer cost 4.946 ms
DEBUG 05-06 10:01:12.335141.335141 lmp.py:1510] ---- decode step 22 layer 19 ----
DEBUG 05-06 10:01:12.340684.340684 cuda_h.py:27] end decode_layer cost 4.924 ms
DEBUG 05-06 10:01:12.340626.340626 lmp.py:1510] ---- decode step 22 layer 20 ----
DEBUG 05-06 10:01:12.345338.345338 cuda_h.py:27] end decode_layer cost 4.838 ms
DEBUG 05-06 10:01:12.345804.345804 lmp.py:1510] ---- decode step 22 layer 21 ----
DEBUG 05-06 10:01:12.350984.350984 cuda_h.py:27] end decode_layer cost 4.972 ms
DEBUG 05-06 10:01:12.350165.350165 lmp.py:1510] ---- decode step 22 layer 22 ----
DEBUG 05-06 10:01:12.355661.355661 cuda_h.py:27] end decode_layer cost 4.924 ms
DEBUG 05-06 10:01:12.355603.355603 lmp.py:1510] ---- decode step 22 layer 23 ----
DEBUG 05-06 10:01:12.361592.361592 cuda_h.py:27] end decode_layer cost 5.182 ms
DEBUG 05-06 10:01:12.361581.361581 lmp.py:1510] ---- decode step 22 layer 24 ----
DEBUG 05-06 10:01:12.365246.365246 cuda_h.py:27] end decode_layer cost 4.839 ms
DEBUG 05-06 10:01:12.366758.366758 lmp.py:1510] ---- decode step 22 layer 25 ----
DEBUG 05-06 10:01:12.371296.371296 cuda_h.py:27] end decode_layer cost 4.990 ms
DEBUG 05-06 10:01:12.371431.371431 lmp.py:1510] ---- decode step 22 layer 26 ----
DEBUG 05-06 10:01:12.376354.376354 cuda_h.py:27] end decode_layer cost 5.626 ms
DEBUG 05-06 10:01:12.376681.376681 lmp.py:1510] ---- decode step 22 layer 27 ----
DEBUG 05-06 10:01:12.381413.381413 cuda_h.py:27] end decode_layer cost 5.028 ms
DEBUG 05-06 10:01:12.381355.381355 lmp.py:1510] ---- decode step 22 layer 28 ----
DEBUG 05-06 10:01:12.386488.386488 cuda_h.py:27] end decode_layer cost 4.937 ms
DEBUG 05-06 10:01:12.386953.386953 lmp.py:1510] ---- decode step 22 layer 29 ----
DEBUG 05-06 10:01:12.392859.392859 cuda_h.py:27] end decode_layer cost 5.296 ms
DEBUG 05-06 10:01:12.392558.392558 cuda_h.py:27] end decode_step cost 159.658 ms
INFO 05-06 10:01:12.392036.392036 lmp.py:1558] decode step 22 time: 0.1596965789794922 seconds
WARNING 05-06 10:01:12.392922.392922 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:12.392362.392362 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:12.393308.393308 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:12.393788.393788 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:12.398582.398582 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:12.398083.398083 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:12.398190.398190 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:12.400276.400276 cuda_h.py:27] end init_inputs_tokens cost 7.699 ms
DEBUG 05-06 10:01:12.400364.400364 lmp.py:1504] decode step 23 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:12.400134.400134 lmp.py:1510] ---- decode step 23 layer 0 ----
DEBUG 05-06 10:01:12.405737.405737 cuda_h.py:27] end decode_layer cost 4.968 ms
DEBUG 05-06 10:01:12.405203.405203 lmp.py:1510] ---- decode step 23 layer 1 ----
DEBUG 05-06 10:01:12.410396.410396 cuda_h.py:27] end decode_layer cost 4.982 ms
DEBUG 05-06 10:01:12.410577.410577 lmp.py:1510] ---- decode step 23 layer 2 ----
DEBUG 05-06 10:01:12.415058.415058 cuda_h.py:27] end decode_layer cost 4.843 ms
DEBUG 05-06 10:01:12.415000.415000 lmp.py:1510] ---- decode step 23 layer 3 ----
DEBUG 05-06 10:01:12.420440.420440 cuda_h.py:27] end decode_layer cost 5.023 ms
DEBUG 05-06 10:01:12.420383.420383 lmp.py:1510] ---- decode step 23 layer 4 ----
DEBUG 05-06 10:01:12.425461.425461 cuda_h.py:27] end decode_layer cost 4.897 ms
DEBUG 05-06 10:01:12.425688.425688 lmp.py:1510] ---- decode step 23 layer 5 ----
DEBUG 05-06 10:01:12.430841.430841 cuda_h.py:27] end decode_layer cost 5.163 ms
DEBUG 05-06 10:01:12.430022.430022 lmp.py:1510] ---- decode step 23 layer 6 ----
DEBUG 05-06 10:01:12.435530.435530 cuda_h.py:27] end decode_layer cost 5.072 ms
DEBUG 05-06 10:01:12.435599.435599 lmp.py:1510] ---- decode step 23 layer 7 ----
DEBUG 05-06 10:01:12.440149.440149 cuda_h.py:27] end decode_layer cost 5.140 ms
DEBUG 05-06 10:01:12.440568.440568 lmp.py:1510] ---- decode step 23 layer 8 ----
DEBUG 05-06 10:01:12.445399.445399 cuda_h.py:27] end decode_layer cost 4.820 ms
DEBUG 05-06 10:01:12.445248.445248 lmp.py:1510] ---- decode step 23 layer 9 ----
DEBUG 05-06 10:01:12.450354.450354 cuda_h.py:27] end decode_layer cost 4.917 ms
DEBUG 05-06 10:01:12.450342.450342 lmp.py:1510] ---- decode step 23 layer 10 ----
DEBUG 05-06 10:01:12.455602.455602 cuda_h.py:27] end decode_layer cost 4.785 ms
DEBUG 05-06 10:01:12.455876.455876 lmp.py:1510] ---- decode step 23 layer 11 ----
DEBUG 05-06 10:01:12.460382.460382 cuda_h.py:27] end decode_layer cost 5.212 ms
DEBUG 05-06 10:01:12.460815.460815 lmp.py:1510] ---- decode step 23 layer 12 ----
DEBUG 05-06 10:01:12.465185.465185 cuda_h.py:27] end decode_layer cost 4.938 ms
DEBUG 05-06 10:01:12.465843.465843 lmp.py:1510] ---- decode step 23 layer 13 ----
DEBUG 05-06 10:01:12.470934.470934 cuda_h.py:27] end decode_layer cost 4.871 ms
DEBUG 05-06 10:01:12.470161.470161 lmp.py:1510] ---- decode step 23 layer 14 ----
DEBUG 05-06 10:01:12.475578.475578 cuda_h.py:27] end decode_layer cost 4.936 ms
DEBUG 05-06 10:01:12.475759.475759 lmp.py:1510] ---- decode step 23 layer 15 ----
DEBUG 05-06 10:01:12.480351.480351 cuda_h.py:27] end decode_layer cost 4.820 ms
DEBUG 05-06 10:01:12.480340.480340 lmp.py:1510] ---- decode step 23 layer 16 ----
DEBUG 05-06 10:01:12.485653.485653 cuda_h.py:27] end decode_layer cost 4.789 ms
DEBUG 05-06 10:01:12.485449.485449 lmp.py:1510] ---- decode step 23 layer 17 ----
DEBUG 05-06 10:01:12.490937.490937 cuda_h.py:27] end decode_layer cost 5.058 ms
DEBUG 05-06 10:01:12.490641.490641 lmp.py:1510] ---- decode step 23 layer 18 ----
DEBUG 05-06 10:01:12.495841.495841 cuda_h.py:27] end decode_layer cost 4.776 ms
DEBUG 05-06 10:01:12.495638.495638 lmp.py:1510] ---- decode step 23 layer 19 ----
DEBUG 05-06 10:01:12.500126.500126 cuda_h.py:27] end decode_layer cost 4.883 ms
DEBUG 05-06 10:01:12.500353.500353 lmp.py:1510] ---- decode step 23 layer 20 ----
DEBUG 05-06 10:01:12.505880.505880 cuda_h.py:27] end decode_layer cost 4.842 ms
DEBUG 05-06 10:01:12.505153.505153 lmp.py:1510] ---- decode step 23 layer 21 ----
DEBUG 05-06 10:01:12.510824.510824 cuda_h.py:27] end decode_layer cost 4.808 ms
DEBUG 05-06 10:01:12.510383.510383 lmp.py:1510] ---- decode step 23 layer 22 ----
DEBUG 05-06 10:01:12.515066.515066 cuda_h.py:27] end decode_layer cost 4.782 ms
DEBUG 05-06 10:01:12.515585.515585 lmp.py:1510] ---- decode step 23 layer 23 ----
DEBUG 05-06 10:01:12.520206.520206 cuda_h.py:27] end decode_layer cost 5.087 ms
DEBUG 05-06 10:01:12.520148.520148 lmp.py:1510] ---- decode step 23 layer 24 ----
DEBUG 05-06 10:01:12.525846.525846 cuda_h.py:27] end decode_layer cost 4.828 ms
DEBUG 05-06 10:01:12.525690.525690 lmp.py:1510] ---- decode step 23 layer 25 ----
DEBUG 05-06 10:01:12.529189.529189 cuda_h.py:27] end decode_layer cost 4.822 ms
DEBUG 05-06 10:01:12.530986.530986 lmp.py:1510] ---- decode step 23 layer 26 ----
DEBUG 05-06 10:01:12.534635.534635 cuda_h.py:27] end decode_layer cost 4.757 ms
DEBUG 05-06 10:01:12.534670.534670 lmp.py:1510] ---- decode step 23 layer 27 ----
DEBUG 05-06 10:01:12.539137.539137 cuda_h.py:27] end decode_layer cost 4.832 ms
DEBUG 05-06 10:01:12.539364.539364 lmp.py:1510] ---- decode step 23 layer 28 ----
DEBUG 05-06 10:01:12.544757.544757 cuda_h.py:27] end decode_layer cost 4.813 ms
DEBUG 05-06 10:01:12.544746.544746 lmp.py:1510] ---- decode step 23 layer 29 ----
DEBUG 05-06 10:01:12.549229.549229 cuda_h.py:27] end decode_layer cost 5.301 ms
DEBUG 05-06 10:01:12.550589.550589 cuda_h.py:27] end decode_step cost 157.689 ms
INFO 05-06 10:01:12.550160.550160 lmp.py:1558] decode step 23 time: 0.15772557258605957 seconds
WARNING 05-06 10:01:12.550840.550840 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:12.550250.550250 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:12.550688.550688 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:12.550984.550984 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:12.556684.556684 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:12.556847.556847 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:12.556001.556001 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:12.557821.557821 cuda_h.py:27] end init_inputs_tokens cost 7.615 ms
DEBUG 05-06 10:01:12.557472.557472 lmp.py:1504] decode step 24 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:12.557712.557712 lmp.py:1510] ---- decode step 24 layer 0 ----
DEBUG 05-06 10:01:12.562312.562312 cuda_h.py:27] end decode_layer cost 4.859 ms
DEBUG 05-06 10:01:12.562870.562870 lmp.py:1510] ---- decode step 24 layer 1 ----
DEBUG 05-06 10:01:12.567038.567038 cuda_h.py:27] end decode_layer cost 4.823 ms
DEBUG 05-06 10:01:12.567882.567882 lmp.py:1510] ---- decode step 24 layer 2 ----
DEBUG 05-06 10:01:12.572252.572252 cuda_h.py:27] end decode_layer cost 4.726 ms
DEBUG 05-06 10:01:12.572340.572340 lmp.py:1510] ---- decode step 24 layer 3 ----
DEBUG 05-06 10:01:12.577376.577376 cuda_h.py:27] end decode_layer cost 4.832 ms
DEBUG 05-06 10:01:12.577034.577034 lmp.py:1510] ---- decode step 24 layer 4 ----
DEBUG 05-06 10:01:12.582012.582012 cuda_h.py:27] end decode_layer cost 4.683 ms
DEBUG 05-06 10:01:12.582379.582379 lmp.py:1510] ---- decode step 24 layer 5 ----
DEBUG 05-06 10:01:12.587419.587419 cuda_h.py:27] end decode_layer cost 4.975 ms
DEBUG 05-06 10:01:12.587786.587786 lmp.py:1510] ---- decode step 24 layer 6 ----
DEBUG 05-06 10:01:12.591656.591656 cuda_h.py:27] end decode_layer cost 4.814 ms
DEBUG 05-06 10:01:12.592075.592075 lmp.py:1510] ---- decode step 24 layer 7 ----
DEBUG 05-06 10:01:12.596602.596602 cuda_h.py:27] end decode_layer cost 4.842 ms
DEBUG 05-06 10:01:12.596398.596398 lmp.py:1510] ---- decode step 24 layer 8 ----
DEBUG 05-06 10:01:12.601895.601895 cuda_h.py:27] end decode_layer cost 4.749 ms
DEBUG 05-06 10:01:12.601977.601977 lmp.py:1510] ---- decode step 24 layer 9 ----
DEBUG 05-06 10:01:12.606901.606901 cuda_h.py:27] end decode_layer cost 4.854 ms
DEBUG 05-06 10:01:12.606559.606559 lmp.py:1510] ---- decode step 24 layer 10 ----
DEBUG 05-06 10:01:12.611869.611869 cuda_h.py:27] end decode_layer cost 4.718 ms
DEBUG 05-06 10:01:12.611904.611904 lmp.py:1510] ---- decode step 24 layer 11 ----
DEBUG 05-06 10:01:12.616570.616570 cuda_h.py:27] end decode_layer cost 5.049 ms
DEBUG 05-06 10:01:12.616559.616559 lmp.py:1510] ---- decode step 24 layer 12 ----
DEBUG 05-06 10:01:12.621315.621315 cuda_h.py:27] end decode_layer cost 4.765 ms
DEBUG 05-06 10:01:12.621827.621827 lmp.py:1510] ---- decode step 24 layer 13 ----
DEBUG 05-06 10:01:12.626677.626677 cuda_h.py:27] end decode_layer cost 4.799 ms
DEBUG 05-06 10:01:12.626758.626758 lmp.py:1510] ---- decode step 24 layer 14 ----
DEBUG 05-06 10:01:12.630186.630186 cuda_h.py:27] end decode_layer cost 4.664 ms
DEBUG 05-06 10:01:12.631744.631744 lmp.py:1510] ---- decode step 24 layer 15 ----
DEBUG 05-06 10:01:12.635713.635713 cuda_h.py:27] end decode_layer cost 4.781 ms
DEBUG 05-06 10:01:12.635893.635893 lmp.py:1510] ---- decode step 24 layer 16 ----
DEBUG 05-06 10:01:12.640856.640856 cuda_h.py:27] end decode_layer cost 4.812 ms
DEBUG 05-06 10:01:12.640322.640322 lmp.py:1510] ---- decode step 24 layer 17 ----
DEBUG 05-06 10:01:12.645463.645463 cuda_h.py:27] end decode_layer cost 5.015 ms
DEBUG 05-06 10:01:12.645929.645929 lmp.py:1510] ---- decode step 24 layer 18 ----
DEBUG 05-06 10:01:12.650023.650023 cuda_h.py:27] end decode_layer cost 4.769 ms
DEBUG 05-06 10:01:12.650535.650535 lmp.py:1510] ---- decode step 24 layer 19 ----
DEBUG 05-06 10:01:12.655874.655874 cuda_h.py:27] end decode_layer cost 4.773 ms
DEBUG 05-06 10:01:12.655670.655670 lmp.py:1510] ---- decode step 24 layer 20 ----
DEBUG 05-06 10:01:12.660816.660816 cuda_h.py:27] end decode_layer cost 4.736 ms
DEBUG 05-06 10:01:12.660805.660805 lmp.py:1510] ---- decode step 24 layer 21 ----
DEBUG 05-06 10:01:12.665913.665913 cuda_h.py:27] end decode_layer cost 4.814 ms
DEBUG 05-06 10:01:12.665471.665471 lmp.py:1510] ---- decode step 24 layer 22 ----
DEBUG 05-06 10:01:12.669515.669515 cuda_h.py:27] end decode_layer cost 4.661 ms
DEBUG 05-06 10:01:12.669120.669120 lmp.py:1510] ---- decode step 24 layer 23 ----
DEBUG 05-06 10:01:12.674822.674822 cuda_h.py:27] end decode_layer cost 4.936 ms
DEBUG 05-06 10:01:12.674903.674903 lmp.py:1510] ---- decode step 24 layer 24 ----
DEBUG 05-06 10:01:12.679695.679695 cuda_h.py:27] end decode_layer cost 4.651 ms
DEBUG 05-06 10:01:12.679313.679313 lmp.py:1510] ---- decode step 24 layer 25 ----
DEBUG 05-06 10:01:12.684505.684505 cuda_h.py:27] end decode_layer cost 4.735 ms
DEBUG 05-06 10:01:12.684686.684686 lmp.py:1510] ---- decode step 24 layer 26 ----
DEBUG 05-06 10:01:12.689035.689035 cuda_h.py:27] end decode_layer cost 4.675 ms
DEBUG 05-06 10:01:12.689070.689070 lmp.py:1510] ---- decode step 24 layer 27 ----
DEBUG 05-06 10:01:12.693865.693865 cuda_h.py:27] end decode_layer cost 4.759 ms
DEBUG 05-06 10:01:12.694947.694947 lmp.py:1510] ---- decode step 24 layer 28 ----
DEBUG 05-06 10:01:12.698256.698256 cuda_h.py:27] end decode_layer cost 4.681 ms
DEBUG 05-06 10:01:12.698814.698814 lmp.py:1510] ---- decode step 24 layer 29 ----
DEBUG 05-06 10:01:12.703020.703020 cuda_h.py:27] end decode_layer cost 4.956 ms
DEBUG 05-06 10:01:12.703712.703712 cuda_h.py:27] end decode_step cost 153.687 ms
INFO 05-06 10:01:12.703998.703998 lmp.py:1558] decode step 24 time: 0.15372467041015625 seconds
WARNING 05-06 10:01:12.703565.703565 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:12.704475.704475 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:12.704335.704335 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:12.704809.704809 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:12.709919.709919 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:12.709982.709982 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:12.709705.709705 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:12.711746.711746 cuda_h.py:27] end init_inputs_tokens cost 7.444 ms
DEBUG 05-06 10:01:12.711589.711589 lmp.py:1504] decode step 25 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:12.711643.711643 lmp.py:1510] ---- decode step 25 layer 0 ----
DEBUG 05-06 10:01:12.716451.716451 cuda_h.py:27] end decode_layer cost 4.732 ms
DEBUG 05-06 10:01:12.716771.716771 lmp.py:1510] ---- decode step 25 layer 1 ----
DEBUG 05-06 10:01:12.721654.721654 cuda_h.py:27] end decode_layer cost 4.824 ms
DEBUG 05-06 10:01:12.721166.721166 lmp.py:1510] ---- decode step 25 layer 2 ----
DEBUG 05-06 10:01:12.726113.726113 cuda_h.py:27] end decode_layer cost 4.941 ms
DEBUG 05-06 10:01:12.726340.726340 lmp.py:1510] ---- decode step 25 layer 3 ----
DEBUG 05-06 10:01:12.730806.730806 cuda_h.py:27] end decode_layer cost 4.797 ms
DEBUG 05-06 10:01:12.731603.731603 lmp.py:1510] ---- decode step 25 layer 4 ----
DEBUG 05-06 10:01:12.735304.735304 cuda_h.py:27] end decode_layer cost 4.725 ms
DEBUG 05-06 10:01:12.735101.735101 lmp.py:1510] ---- decode step 25 layer 5 ----
DEBUG 05-06 10:01:12.740387.740387 cuda_h.py:27] end decode_layer cost 4.980 ms
DEBUG 05-06 10:01:12.740992.740992 lmp.py:1510] ---- decode step 25 layer 6 ----
DEBUG 05-06 10:01:12.745912.745912 cuda_h.py:27] end decode_layer cost 4.746 ms
DEBUG 05-06 10:01:12.745232.745232 lmp.py:1510] ---- decode step 25 layer 7 ----
DEBUG 05-06 10:01:12.750298.750298 cuda_h.py:27] end decode_layer cost 4.712 ms
DEBUG 05-06 10:01:12.750617.750617 lmp.py:1510] ---- decode step 25 layer 8 ----
DEBUG 05-06 10:01:12.755174.755174 cuda_h.py:27] end decode_layer cost 4.759 ms
DEBUG 05-06 10:01:12.755733.755733 lmp.py:1510] ---- decode step 25 layer 9 ----
DEBUG 05-06 10:01:12.760322.760322 cuda_h.py:27] end decode_layer cost 4.748 ms
DEBUG 05-06 10:01:12.760165.760165 lmp.py:1510] ---- decode step 25 layer 10 ----
DEBUG 05-06 10:01:12.764721.764721 cuda_h.py:27] end decode_layer cost 4.723 ms
DEBUG 05-06 10:01:12.764802.764802 lmp.py:1510] ---- decode step 25 layer 11 ----
DEBUG 05-06 10:01:12.769935.769935 cuda_h.py:27] end decode_layer cost 4.937 ms
DEBUG 05-06 10:01:12.769970.769970 lmp.py:1510] ---- decode step 25 layer 12 ----
DEBUG 05-06 10:01:12.774413.774413 cuda_h.py:27] end decode_layer cost 4.710 ms
DEBUG 05-06 10:01:12.774640.774640 lmp.py:1510] ---- decode step 25 layer 13 ----
DEBUG 05-06 10:01:12.779204.779204 cuda_h.py:27] end decode_layer cost 4.764 ms
DEBUG 05-06 10:01:12.779047.779047 lmp.py:1510] ---- decode step 25 layer 14 ----
DEBUG 05-06 10:01:12.784894.784894 cuda_h.py:27] end decode_layer cost 4.726 ms
DEBUG 05-06 10:01:12.784459.784459 lmp.py:1510] ---- decode step 25 layer 15 ----
DEBUG 05-06 10:01:12.789138.789138 cuda_h.py:27] end decode_layer cost 4.849 ms
DEBUG 05-06 10:01:12.789412.789412 lmp.py:1510] ---- decode step 25 layer 16 ----
DEBUG 05-06 10:01:12.793835.793835 cuda_h.py:27] end decode_layer cost 4.731 ms
DEBUG 05-06 10:01:12.794201.794201 lmp.py:1510] ---- decode step 25 layer 17 ----
DEBUG 05-06 10:01:12.799445.799445 cuda_h.py:27] end decode_layer cost 5.089 ms
DEBUG 05-06 10:01:12.799003.799003 lmp.py:1510] ---- decode step 25 layer 18 ----
DEBUG 05-06 10:01:12.803533.803533 cuda_h.py:27] end decode_layer cost 4.739 ms
DEBUG 05-06 10:01:12.803899.803899 lmp.py:1510] ---- decode step 25 layer 19 ----
DEBUG 05-06 10:01:12.808767.808767 cuda_h.py:27] end decode_layer cost 4.742 ms
DEBUG 05-06 10:01:12.808563.808563 lmp.py:1510] ---- decode step 25 layer 20 ----
DEBUG 05-06 10:01:12.813528.813528 cuda_h.py:27] end decode_layer cost 4.673 ms
DEBUG 05-06 10:01:12.813371.813371 lmp.py:1510] ---- decode step 25 layer 21 ----
DEBUG 05-06 10:01:12.818551.818551 cuda_h.py:27] end decode_layer cost 4.761 ms
DEBUG 05-06 10:01:12.818447.818447 lmp.py:1510] ---- decode step 25 layer 22 ----
DEBUG 05-06 10:01:12.823306.823306 cuda_h.py:27] end decode_layer cost 5.087 ms
DEBUG 05-06 10:01:12.823825.823825 lmp.py:1510] ---- decode step 25 layer 23 ----
DEBUG 05-06 10:01:12.828073.828073 cuda_h.py:27] end decode_layer cost 5.233 ms
DEBUG 05-06 10:01:12.828685.828685 lmp.py:1510] ---- decode step 25 layer 24 ----
DEBUG 05-06 10:01:12.833650.833650 cuda_h.py:27] end decode_layer cost 4.884 ms
DEBUG 05-06 10:01:12.833023.833023 lmp.py:1510] ---- decode step 25 layer 25 ----
DEBUG 05-06 10:01:12.838465.838465 cuda_h.py:27] end decode_layer cost 4.885 ms
DEBUG 05-06 10:01:12.838977.838977 lmp.py:1510] ---- decode step 25 layer 26 ----
DEBUG 05-06 10:01:12.843043.843043 cuda_h.py:27] end decode_layer cost 4.748 ms
DEBUG 05-06 10:01:12.843793.843793 lmp.py:1510] ---- decode step 25 layer 27 ----
DEBUG 05-06 10:01:12.848033.848033 cuda_h.py:27] end decode_layer cost 4.771 ms
DEBUG 05-06 10:01:12.848591.848591 lmp.py:1510] ---- decode step 25 layer 28 ----
DEBUG 05-06 10:01:12.853642.853642 cuda_h.py:27] end decode_layer cost 4.666 ms
DEBUG 05-06 10:01:12.853154.853154 lmp.py:1510] ---- decode step 25 layer 29 ----
DEBUG 05-06 10:01:12.858557.858557 cuda_h.py:27] end decode_layer cost 4.927 ms
DEBUG 05-06 10:01:12.858719.858719 cuda_h.py:27] end decode_step cost 154.198 ms
INFO 05-06 10:01:12.858097.858097 lmp.py:1558] decode step 25 time: 0.1542341709136963 seconds
WARNING 05-06 10:01:12.858665.858665 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:12.858330.858330 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:12.858971.858971 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:12.859399.859399 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:12.864225.864225 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:12.864905.864905 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:12.864674.864674 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:12.865165.865165 cuda_h.py:27] end init_inputs_tokens cost 7.504 ms
DEBUG 05-06 10:01:12.865055.865055 lmp.py:1504] decode step 26 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:12.865533.865533 lmp.py:1510] ---- decode step 26 layer 0 ----
DEBUG 05-06 10:01:12.870789.870789 cuda_h.py:27] end decode_layer cost 4.676 ms
DEBUG 05-06 10:01:12.870917.870917 lmp.py:1510] ---- decode step 26 layer 1 ----
DEBUG 05-06 10:01:12.875068.875068 cuda_h.py:27] end decode_layer cost 4.705 ms
DEBUG 05-06 10:01:12.875865.875865 lmp.py:1510] ---- decode step 26 layer 2 ----
DEBUG 05-06 10:01:12.879279.879279 cuda_h.py:27] end decode_layer cost 4.654 ms
DEBUG 05-06 10:01:12.880076.880076 lmp.py:1510] ---- decode step 26 layer 3 ----
DEBUG 05-06 10:01:12.884134.884134 cuda_h.py:27] end decode_layer cost 4.708 ms
DEBUG 05-06 10:01:12.884646.884646 lmp.py:1510] ---- decode step 26 layer 4 ----
DEBUG 05-06 10:01:12.889784.889784 cuda_h.py:27] end decode_layer cost 4.695 ms
DEBUG 05-06 10:01:12.889011.889011 lmp.py:1510] ---- decode step 26 layer 5 ----
DEBUG 05-06 10:01:12.894635.894635 cuda_h.py:27] end decode_layer cost 4.984 ms
DEBUG 05-06 10:01:12.894955.894955 lmp.py:1510] ---- decode step 26 layer 6 ----
DEBUG 05-06 10:01:12.899848.899848 cuda_h.py:27] end decode_layer cost 4.726 ms
DEBUG 05-06 10:01:12.899691.899691 lmp.py:1510] ---- decode step 26 layer 7 ----
DEBUG 05-06 10:01:12.904205.904205 cuda_h.py:27] end decode_layer cost 5.042 ms
DEBUG 05-06 10:01:12.904531.904531 lmp.py:1510] ---- decode step 26 layer 8 ----
DEBUG 05-06 10:01:12.909413.909413 cuda_h.py:27] end decode_layer cost 4.752 ms
DEBUG 05-06 10:01:12.909209.909209 lmp.py:1510] ---- decode step 26 layer 9 ----
DEBUG 05-06 10:01:12.914914.914914 cuda_h.py:27] end decode_layer cost 4.833 ms
DEBUG 05-06 10:01:12.914996.914996 lmp.py:1510] ---- decode step 26 layer 10 ----
DEBUG 05-06 10:01:12.919155.919155 cuda_h.py:27] end decode_layer cost 4.746 ms
DEBUG 05-06 10:01:12.919998.919998 lmp.py:1510] ---- decode step 26 layer 11 ----
DEBUG 05-06 10:01:12.924486.924486 cuda_h.py:27] end decode_layer cost 5.058 ms
DEBUG 05-06 10:01:12.924951.924951 lmp.py:1510] ---- decode step 26 layer 12 ----
DEBUG 05-06 10:01:12.929338.929338 cuda_h.py:27] end decode_layer cost 4.809 ms
DEBUG 05-06 10:01:12.929704.929704 lmp.py:1510] ---- decode step 26 layer 13 ----
DEBUG 05-06 10:01:12.933153.933153 cuda_h.py:27] end decode_layer cost 4.714 ms
DEBUG 05-06 10:01:12.933718.933718 lmp.py:1510] ---- decode step 26 layer 14 ----
DEBUG 05-06 10:01:12.938962.938962 cuda_h.py:27] end decode_layer cost 4.704 ms
DEBUG 05-06 10:01:12.938997.938997 lmp.py:1510] ---- decode step 26 layer 15 ----
DEBUG 05-06 10:01:12.943015.943015 cuda_h.py:27] end decode_layer cost 4.677 ms
DEBUG 05-06 10:01:12.943812.943812 lmp.py:1510] ---- decode step 26 layer 16 ----
DEBUG 05-06 10:01:12.948897.948897 cuda_h.py:27] end decode_layer cost 4.727 ms
DEBUG 05-06 10:01:12.948740.948740 lmp.py:1510] ---- decode step 26 layer 17 ----
DEBUG 05-06 10:01:12.953309.953309 cuda_h.py:27] end decode_layer cost 4.908 ms
DEBUG 05-06 10:01:12.953914.953914 lmp.py:1510] ---- decode step 26 layer 18 ----
DEBUG 05-06 10:01:12.957582.957582 cuda_h.py:27] end decode_layer cost 4.735 ms
DEBUG 05-06 10:01:12.957333.957333 lmp.py:1510] ---- decode step 26 layer 19 ----
DEBUG 05-06 10:01:12.962074.962074 cuda_h.py:27] end decode_layer cost 4.719 ms
DEBUG 05-06 10:01:12.962778.962778 lmp.py:1510] ---- decode step 26 layer 20 ----
DEBUG 05-06 10:01:12.967854.967854 cuda_h.py:27] end decode_layer cost 4.650 ms
DEBUG 05-06 10:01:12.967651.967651 lmp.py:1510] ---- decode step 26 layer 21 ----
DEBUG 05-06 10:01:12.972855.972855 cuda_h.py:27] end decode_layer cost 4.709 ms
DEBUG 05-06 10:01:12.972460.972460 lmp.py:1510] ---- decode step 26 layer 22 ----
DEBUG 05-06 10:01:12.976092.976092 cuda_h.py:27] end decode_layer cost 4.639 ms
DEBUG 05-06 10:01:12.976935.976935 lmp.py:1510] ---- decode step 26 layer 23 ----
DEBUG 05-06 10:01:12.981709.981709 cuda_h.py:27] end decode_layer cost 4.919 ms
DEBUG 05-06 10:01:12.981791.981791 lmp.py:1510] ---- decode step 26 layer 24 ----
DEBUG 05-06 10:01:12.986231.986231 cuda_h.py:27] end decode_layer cost 4.637 ms
DEBUG 05-06 10:01:12.986551.986551 lmp.py:1510] ---- decode step 26 layer 25 ----
DEBUG 05-06 10:01:12.991811.991811 cuda_h.py:27] end decode_layer cost 4.785 ms
DEBUG 05-06 10:01:12.991131.991131 lmp.py:1510] ---- decode step 26 layer 26 ----
DEBUG 05-06 10:01:12.996250.996250 cuda_h.py:27] end decode_layer cost 4.752 ms
DEBUG 05-06 10:01:12.996716.996716 lmp.py:1510] ---- decode step 26 layer 27 ----
DEBUG 05-06 10:01:13.001195.001195 cuda_h.py:27] end decode_layer cost 4.806 ms
DEBUG 05-06 10:01:13.001992.001992 lmp.py:1510] ---- decode step 26 layer 28 ----
DEBUG 05-06 10:01:13.005825.005825 cuda_h.py:27] end decode_layer cost 4.717 ms
DEBUG 05-06 10:01:13.005430.005430 lmp.py:1510] ---- decode step 26 layer 29 ----
DEBUG 05-06 10:01:13.010451.010451 cuda_h.py:27] end decode_layer cost 4.960 ms
DEBUG 05-06 10:01:13.010142.010142 cuda_h.py:27] end decode_step cost 152.787 ms
INFO 05-06 10:01:13.011190.011190 lmp.py:1558] decode step 26 time: 0.15282440185546875 seconds
WARNING 05-06 10:01:13.011718.011718 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:13.011519.011519 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:13.011843.011843 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:13.012846.012846 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:13.017030.017030 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:13.017709.017709 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:13.017432.017432 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:13.019660.019660 cuda_h.py:27] end init_inputs_tokens cost 8.259 ms
DEBUG 05-06 10:01:13.019550.019550 lmp.py:1504] decode step 27 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:13.019028.019028 lmp.py:1510] ---- decode step 27 layer 0 ----
DEBUG 05-06 10:01:13.024645.024645 cuda_h.py:27] end decode_layer cost 4.766 ms
DEBUG 05-06 10:01:13.024110.024110 lmp.py:1510] ---- decode step 27 layer 1 ----
DEBUG 05-06 10:01:13.029562.029562 cuda_h.py:27] end decode_layer cost 4.787 ms
DEBUG 05-06 10:01:13.029359.029359 lmp.py:1510] ---- decode step 27 layer 2 ----
DEBUG 05-06 10:01:13.033464.033464 cuda_h.py:27] end decode_layer cost 4.706 ms
DEBUG 05-06 10:01:13.033545.033545 lmp.py:1510] ---- decode step 27 layer 3 ----
DEBUG 05-06 10:01:13.038321.038321 cuda_h.py:27] end decode_layer cost 4.780 ms
DEBUG 05-06 10:01:13.038072.038072 lmp.py:1510] ---- decode step 27 layer 4 ----
DEBUG 05-06 10:01:13.043907.043907 cuda_h.py:27] end decode_layer cost 4.753 ms
DEBUG 05-06 10:01:13.043180.043180 lmp.py:1510] ---- decode step 27 layer 5 ----
DEBUG 05-06 10:01:13.048905.048905 cuda_h.py:27] end decode_layer cost 5.023 ms
DEBUG 05-06 10:01:13.048609.048609 lmp.py:1510] ---- decode step 27 layer 6 ----
DEBUG 05-06 10:01:13.053003.053003 cuda_h.py:27] end decode_layer cost 4.639 ms
DEBUG 05-06 10:01:13.053276.053276 lmp.py:1510] ---- decode step 27 layer 7 ----
DEBUG 05-06 10:01:13.058874.058874 cuda_h.py:27] end decode_layer cost 4.789 ms
DEBUG 05-06 10:01:13.058293.058293 lmp.py:1510] ---- decode step 27 layer 8 ----
DEBUG 05-06 10:01:13.062604.062604 cuda_h.py:27] end decode_layer cost 4.718 ms
DEBUG 05-06 10:01:13.063162.063162 lmp.py:1510] ---- decode step 27 layer 9 ----
DEBUG 05-06 10:01:13.067949.067949 cuda_h.py:27] end decode_layer cost 4.718 ms
DEBUG 05-06 10:01:13.067508.067508 lmp.py:1510] ---- decode step 27 layer 10 ----
DEBUG 05-06 10:01:13.072684.072684 cuda_h.py:27] end decode_layer cost 4.653 ms
DEBUG 05-06 10:01:13.072480.072480 lmp.py:1510] ---- decode step 27 layer 11 ----
DEBUG 05-06 10:01:13.077589.077589 cuda_h.py:27] end decode_layer cost 5.025 ms
DEBUG 05-06 10:01:13.077717.077717 lmp.py:1510] ---- decode step 27 layer 12 ----
DEBUG 05-06 10:01:13.082932.082932 cuda_h.py:27] end decode_layer cost 4.647 ms
DEBUG 05-06 10:01:13.082252.082252 lmp.py:1510] ---- decode step 27 layer 13 ----
DEBUG 05-06 10:01:13.087661.087661 cuda_h.py:27] end decode_layer cost 4.685 ms
DEBUG 05-06 10:01:13.087266.087266 lmp.py:1510] ---- decode step 27 layer 14 ----
DEBUG 05-06 10:01:13.091556.091556 cuda_h.py:27] end decode_layer cost 4.703 ms
DEBUG 05-06 10:01:13.091114.091114 lmp.py:1510] ---- decode step 27 layer 15 ----
DEBUG 05-06 10:01:13.096675.096675 cuda_h.py:27] end decode_layer cost 4.691 ms
DEBUG 05-06 10:01:13.096757.096757 lmp.py:1510] ---- decode step 27 layer 16 ----
DEBUG 05-06 10:01:13.101515.101515 cuda_h.py:27] end decode_layer cost 4.626 ms
DEBUG 05-06 10:01:13.101073.101073 lmp.py:1510] ---- decode step 27 layer 17 ----
DEBUG 05-06 10:01:13.106793.106793 cuda_h.py:27] end decode_layer cost 4.879 ms
DEBUG 05-06 10:01:13.106113.106113 lmp.py:1510] ---- decode step 27 layer 18 ----
DEBUG 05-06 10:01:13.110494.110494 cuda_h.py:27] end decode_layer cost 4.664 ms
DEBUG 05-06 10:01:13.110860.110860 lmp.py:1510] ---- decode step 27 layer 19 ----
DEBUG 05-06 10:01:13.115423.115423 cuda_h.py:27] end decode_layer cost 4.728 ms
DEBUG 05-06 10:01:13.115266.115266 lmp.py:1510] ---- decode step 27 layer 20 ----
DEBUG 05-06 10:01:13.120774.120774 cuda_h.py:27] end decode_layer cost 4.688 ms
DEBUG 05-06 10:01:13.120617.120617 lmp.py:1510] ---- decode step 27 layer 21 ----
DEBUG 05-06 10:01:13.125794.125794 cuda_h.py:27] end decode_layer cost 4.689 ms
DEBUG 05-06 10:01:13.125637.125637 lmp.py:1510] ---- decode step 27 layer 22 ----
DEBUG 05-06 10:01:13.130174.130174 cuda_h.py:27] end decode_layer cost 4.744 ms
DEBUG 05-06 10:01:13.130494.130494 lmp.py:1510] ---- decode step 27 layer 23 ----
DEBUG 05-06 10:01:13.135520.135520 cuda_h.py:27] end decode_layer cost 4.929 ms
DEBUG 05-06 10:01:13.135793.135793 lmp.py:1510] ---- decode step 27 layer 24 ----
DEBUG 05-06 10:01:13.139793.139793 cuda_h.py:27] end decode_layer cost 4.734 ms
DEBUG 05-06 10:01:13.139067.139067 lmp.py:1510] ---- decode step 27 layer 25 ----
DEBUG 05-06 10:01:13.144704.144704 cuda_h.py:27] end decode_layer cost 4.783 ms
DEBUG 05-06 10:01:13.144408.144408 lmp.py:1510] ---- decode step 27 layer 26 ----
DEBUG 05-06 10:01:13.149195.149195 cuda_h.py:27] end decode_layer cost 4.718 ms
DEBUG 05-06 10:01:13.149038.149038 lmp.py:1510] ---- decode step 27 layer 27 ----
DEBUG 05-06 10:01:13.154311.154311 cuda_h.py:27] end decode_layer cost 4.759 ms
DEBUG 05-06 10:01:13.154637.154637 lmp.py:1510] ---- decode step 27 layer 28 ----
DEBUG 05-06 10:01:13.158786.158786 cuda_h.py:27] end decode_layer cost 4.634 ms
DEBUG 05-06 10:01:13.159060.159060 lmp.py:1510] ---- decode step 27 layer 29 ----
DEBUG 05-06 10:01:13.164533.164533 cuda_h.py:27] end decode_layer cost 5.048 ms
DEBUG 05-06 10:01:13.164695.164695 cuda_h.py:27] end decode_step cost 153.093 ms
INFO 05-06 10:01:13.164074.164074 lmp.py:1558] decode step 27 time: 0.15312886238098145 seconds
WARNING 05-06 10:01:13.164635.164635 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:13.164825.164825 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:13.165174.165174 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:13.165409.165409 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:13.170506.170506 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:13.170854.170854 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:13.170100.170100 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:13.171604.171604 cuda_h.py:27] end init_inputs_tokens cost 7.460 ms
DEBUG 05-06 10:01:13.171493.171493 lmp.py:1504] decode step 28 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:13.171256.171256 lmp.py:1510] ---- decode step 28 layer 0 ----
DEBUG 05-06 10:01:13.176636.176636 cuda_h.py:27] end decode_layer cost 4.627 ms
DEBUG 05-06 10:01:13.176479.176479 lmp.py:1510] ---- decode step 28 layer 1 ----
DEBUG 05-06 10:01:13.181451.181451 cuda_h.py:27] end decode_layer cost 4.679 ms
DEBUG 05-06 10:01:13.181724.181724 lmp.py:1510] ---- decode step 28 layer 2 ----
DEBUG 05-06 10:01:13.185813.185813 cuda_h.py:27] end decode_layer cost 4.624 ms
DEBUG 05-06 10:01:13.185087.185087 lmp.py:1510] ---- decode step 28 layer 3 ----
DEBUG 05-06 10:01:13.190695.190695 cuda_h.py:27] end decode_layer cost 4.727 ms
DEBUG 05-06 10:01:13.190254.190254 lmp.py:1510] ---- decode step 28 layer 4 ----
DEBUG 05-06 10:01:13.195588.195588 cuda_h.py:27] end decode_layer cost 4.630 ms
DEBUG 05-06 10:01:13.195669.195669 lmp.py:1510] ---- decode step 28 layer 5 ----
DEBUG 05-06 10:01:13.200212.200212 cuda_h.py:27] end decode_layer cost 4.923 ms
DEBUG 05-06 10:01:13.200916.200916 lmp.py:1510] ---- decode step 28 layer 6 ----
DEBUG 05-06 10:01:13.205279.205279 cuda_h.py:27] end decode_layer cost 4.722 ms
DEBUG 05-06 10:01:13.205076.205076 lmp.py:1510] ---- decode step 28 layer 7 ----
DEBUG 05-06 10:01:13.209440.209440 cuda_h.py:27] end decode_layer cost 4.721 ms
DEBUG 05-06 10:01:13.210475.210475 lmp.py:1510] ---- decode step 28 layer 8 ----
DEBUG 05-06 10:01:13.214438.214438 cuda_h.py:27] end decode_layer cost 4.637 ms
DEBUG 05-06 10:01:13.214997.214997 lmp.py:1510] ---- decode step 28 layer 9 ----
DEBUG 05-06 10:01:13.219726.219726 cuda_h.py:27] end decode_layer cost 4.781 ms
DEBUG 05-06 10:01:13.219431.219431 lmp.py:1510] ---- decode step 28 layer 10 ----
DEBUG 05-06 10:01:13.224818.224818 cuda_h.py:27] end decode_layer cost 4.634 ms
DEBUG 05-06 10:01:13.224899.224899 lmp.py:1510] ---- decode step 28 layer 11 ----
DEBUG 05-06 10:01:13.229024.229024 cuda_h.py:27] end decode_layer cost 4.896 ms
DEBUG 05-06 10:01:13.229390.229390 lmp.py:1510] ---- decode step 28 layer 12 ----
DEBUG 05-06 10:01:13.233148.233148 cuda_h.py:27] end decode_layer cost 4.626 ms
DEBUG 05-06 10:01:13.233991.233991 lmp.py:1510] ---- decode step 28 layer 13 ----
DEBUG 05-06 10:01:13.238082.238082 cuda_h.py:27] end decode_layer cost 4.696 ms
DEBUG 05-06 10:01:13.238117.238117 lmp.py:1510] ---- decode step 28 layer 14 ----
DEBUG 05-06 10:01:13.243550.243550 cuda_h.py:27] end decode_layer cost 4.633 ms
DEBUG 05-06 10:01:13.243870.243870 lmp.py:1510] ---- decode step 28 layer 15 ----
DEBUG 05-06 10:01:13.248224.248224 cuda_h.py:27] end decode_layer cost 4.819 ms
DEBUG 05-06 10:01:13.248451.248451 lmp.py:1510] ---- decode step 28 layer 16 ----
DEBUG 05-06 10:01:13.253140.253140 cuda_h.py:27] end decode_layer cost 4.926 ms
DEBUG 05-06 10:01:13.253844.253844 lmp.py:1510] ---- decode step 28 layer 17 ----
DEBUG 05-06 10:01:13.258215.258215 cuda_h.py:27] end decode_layer cost 5.148 ms
DEBUG 05-06 10:01:13.258396.258396 lmp.py:1510] ---- decode step 28 layer 18 ----
DEBUG 05-06 10:01:13.263614.263614 cuda_h.py:27] end decode_layer cost 4.930 ms
DEBUG 05-06 10:01:13.263841.263841 lmp.py:1510] ---- decode step 28 layer 19 ----
DEBUG 05-06 10:01:13.268431.268431 cuda_h.py:27] end decode_layer cost 4.958 ms
DEBUG 05-06 10:01:13.268612.268612 lmp.py:1510] ---- decode step 28 layer 20 ----
DEBUG 05-06 10:01:13.273113.273113 cuda_h.py:27] end decode_layer cost 4.858 ms
DEBUG 05-06 10:01:13.273910.273910 lmp.py:1510] ---- decode step 28 layer 21 ----
DEBUG 05-06 10:01:13.278256.278256 cuda_h.py:27] end decode_layer cost 4.815 ms
DEBUG 05-06 10:01:13.278576.278576 lmp.py:1510] ---- decode step 28 layer 22 ----
DEBUG 05-06 10:01:13.283571.283571 cuda_h.py:27] end decode_layer cost 4.765 ms
DEBUG 05-06 10:01:13.283367.283367 lmp.py:1510] ---- decode step 28 layer 23 ----
DEBUG 05-06 10:01:13.288968.288968 cuda_h.py:27] end decode_layer cost 5.072 ms
DEBUG 05-06 10:01:13.288672.288672 lmp.py:1510] ---- decode step 28 layer 24 ----
DEBUG 05-06 10:01:13.293562.293562 cuda_h.py:27] end decode_layer cost 4.829 ms
DEBUG 05-06 10:01:13.293882.293882 lmp.py:1510] ---- decode step 28 layer 25 ----
DEBUG 05-06 10:01:13.298509.298509 cuda_h.py:27] end decode_layer cost 4.880 ms
DEBUG 05-06 10:01:13.298306.298306 lmp.py:1510] ---- decode step 28 layer 26 ----
DEBUG 05-06 10:01:13.302901.302901 cuda_h.py:27] end decode_layer cost 4.716 ms
DEBUG 05-06 10:01:13.302744.302744 lmp.py:1510] ---- decode step 28 layer 27 ----
DEBUG 05-06 10:01:13.307297.307297 cuda_h.py:27] end decode_layer cost 4.825 ms
DEBUG 05-06 10:01:13.307570.307570 lmp.py:1510] ---- decode step 28 layer 28 ----
DEBUG 05-06 10:01:13.312126.312126 cuda_h.py:27] end decode_layer cost 4.722 ms
DEBUG 05-06 10:01:13.312353.312353 lmp.py:1510] ---- decode step 28 layer 29 ----
DEBUG 05-06 10:01:13.317040.317040 cuda_h.py:27] end decode_layer cost 5.064 ms
DEBUG 05-06 10:01:13.317639.317639 cuda_h.py:27] end decode_step cost 153.487 ms
INFO 05-06 10:01:13.317686.317686 lmp.py:1558] decode step 28 time: 0.15352463722229004 seconds
WARNING 05-06 10:01:13.317976.317976 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:13.318543.318543 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:13.318594.318594 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:13.318260.318260 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:13.323384.323384 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:13.323308.323308 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:13.323462.323462 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:13.325870.325870 cuda_h.py:27] end init_inputs_tokens cost 7.588 ms
DEBUG 05-06 10:01:13.325997.325997 lmp.py:1504] decode step 29 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:13.325999.325999 lmp.py:1510] ---- decode step 29 layer 0 ----
DEBUG 05-06 10:01:13.330636.330636 cuda_h.py:27] end decode_layer cost 4.782 ms
DEBUG 05-06 10:01:13.330386.330386 lmp.py:1510] ---- decode step 29 layer 1 ----
DEBUG 05-06 10:01:13.335891.335891 cuda_h.py:27] end decode_layer cost 4.790 ms
DEBUG 05-06 10:01:13.335165.335165 lmp.py:1510] ---- decode step 29 layer 2 ----
DEBUG 05-06 10:01:13.340386.340386 cuda_h.py:27] end decode_layer cost 4.827 ms
DEBUG 05-06 10:01:13.340329.340329 lmp.py:1510] ---- decode step 29 layer 3 ----
DEBUG 05-06 10:01:13.344550.344550 cuda_h.py:27] end decode_layer cost 4.828 ms
DEBUG 05-06 10:01:13.345108.345108 lmp.py:1510] ---- decode step 29 layer 4 ----
DEBUG 05-06 10:01:13.349767.349767 cuda_h.py:27] end decode_layer cost 4.834 ms
DEBUG 05-06 10:01:13.349379.349379 lmp.py:1510] ---- decode step 29 layer 5 ----
DEBUG 05-06 10:01:13.354680.354680 cuda_h.py:27] end decode_layer cost 5.027 ms
DEBUG 05-06 10:01:13.355907.355907 lmp.py:1510] ---- decode step 29 layer 6 ----
DEBUG 05-06 10:01:13.359424.359424 cuda_h.py:27] end decode_layer cost 4.764 ms
DEBUG 05-06 10:01:13.359036.359036 lmp.py:1510] ---- decode step 29 layer 7 ----
DEBUG 05-06 10:01:13.364595.364595 cuda_h.py:27] end decode_layer cost 4.831 ms
DEBUG 05-06 10:01:13.364206.364206 lmp.py:1510] ---- decode step 29 layer 8 ----
DEBUG 05-06 10:01:13.369412.369412 cuda_h.py:27] end decode_layer cost 4.746 ms
DEBUG 05-06 10:01:13.369208.369208 lmp.py:1510] ---- decode step 29 layer 9 ----
DEBUG 05-06 10:01:13.374025.374025 cuda_h.py:27] end decode_layer cost 4.809 ms
DEBUG 05-06 10:01:13.374537.374537 lmp.py:1510] ---- decode step 29 layer 10 ----
DEBUG 05-06 10:01:13.379908.379908 cuda_h.py:27] end decode_layer cost 4.726 ms
DEBUG 05-06 10:01:13.379420.379420 lmp.py:1510] ---- decode step 29 layer 11 ----
DEBUG 05-06 10:01:13.384801.384801 cuda_h.py:27] end decode_layer cost 5.050 ms
DEBUG 05-06 10:01:13.384982.384982 lmp.py:1510] ---- decode step 29 layer 12 ----
DEBUG 05-06 10:01:13.389024.389024 cuda_h.py:27] end decode_layer cost 4.800 ms
DEBUG 05-06 10:01:13.389966.389966 lmp.py:1510] ---- decode step 29 layer 13 ----
DEBUG 05-06 10:01:13.394201.394201 cuda_h.py:27] end decode_layer cost 4.838 ms
DEBUG 05-06 10:01:13.394951.394951 lmp.py:1510] ---- decode step 29 layer 14 ----
DEBUG 05-06 10:01:13.398695.398695 cuda_h.py:27] end decode_layer cost 4.790 ms
DEBUG 05-06 10:01:13.398253.398253 lmp.py:1510] ---- decode step 29 layer 15 ----
DEBUG 05-06 10:01:13.403672.403672 cuda_h.py:27] end decode_layer cost 4.972 ms
DEBUG 05-06 10:01:13.404283.404283 lmp.py:1510] ---- decode step 29 layer 16 ----
DEBUG 05-06 10:01:13.408633.408633 cuda_h.py:27] end decode_layer cost 4.922 ms
DEBUG 05-06 10:01:13.409576.409576 lmp.py:1510] ---- decode step 29 layer 17 ----
DEBUG 05-06 10:01:13.414929.414929 cuda_h.py:27] end decode_layer cost 5.205 ms
DEBUG 05-06 10:01:13.414872.414872 lmp.py:1510] ---- decode step 29 layer 18 ----
DEBUG 05-06 10:01:13.419973.419973 cuda_h.py:27] end decode_layer cost 4.984 ms
DEBUG 05-06 10:01:13.419346.419346 lmp.py:1510] ---- decode step 29 layer 19 ----
DEBUG 05-06 10:01:13.425408.425408 cuda_h.py:27] end decode_layer cost 6.008 ms
DEBUG 05-06 10:01:13.425808.425808 lmp.py:1510] ---- decode step 29 layer 20 ----
DEBUG 05-06 10:01:13.430100.430100 cuda_h.py:27] end decode_layer cost 4.950 ms
DEBUG 05-06 10:01:13.430327.430327 lmp.py:1510] ---- decode step 29 layer 21 ----
DEBUG 05-06 10:01:13.435997.435997 cuda_h.py:27] end decode_layer cost 4.982 ms
DEBUG 05-06 10:01:13.435463.435463 lmp.py:1510] ---- decode step 29 layer 22 ----
DEBUG 05-06 10:01:13.440270.440270 cuda_h.py:27] end decode_layer cost 4.908 ms
DEBUG 05-06 10:01:13.440166.440166 lmp.py:1510] ---- decode step 29 layer 23 ----
DEBUG 05-06 10:01:13.445780.445780 cuda_h.py:27] end decode_layer cost 5.082 ms
DEBUG 05-06 10:01:13.445769.445769 lmp.py:1510] ---- decode step 29 layer 24 ----
DEBUG 05-06 10:01:13.450752.450752 cuda_h.py:27] end decode_layer cost 4.827 ms
DEBUG 05-06 10:01:13.450979.450979 lmp.py:1510] ---- decode step 29 layer 25 ----
DEBUG 05-06 10:01:13.455049.455049 cuda_h.py:27] end decode_layer cost 4.856 ms
DEBUG 05-06 10:01:13.455323.455323 lmp.py:1510] ---- decode step 29 layer 26 ----
DEBUG 05-06 10:01:13.460808.460808 cuda_h.py:27] end decode_layer cost 4.986 ms
DEBUG 05-06 10:01:13.460943.460943 lmp.py:1510] ---- decode step 29 layer 27 ----
DEBUG 05-06 10:01:13.465559.465559 cuda_h.py:27] end decode_layer cost 4.943 ms
DEBUG 05-06 10:01:13.465117.465117 lmp.py:1510] ---- decode step 29 layer 28 ----
DEBUG 05-06 10:01:13.470171.470171 cuda_h.py:27] end decode_layer cost 4.774 ms
DEBUG 05-06 10:01:13.470398.470398 lmp.py:1510] ---- decode step 29 layer 29 ----
DEBUG 05-06 10:01:13.475852.475852 cuda_h.py:27] end decode_layer cost 5.030 ms
DEBUG 05-06 10:01:13.475689.475689 cuda_h.py:27] end decode_step cost 157.671 ms
INFO 05-06 10:01:13.475313.475313 lmp.py:1558] decode step 29 time: 0.15771245956420898 seconds
WARNING 05-06 10:01:13.475180.475180 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:01:13.476554.476554 helper.py:35]   NaN count (hidden): 8448
WARNING 05-06 10:01:13.476685.476685 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:01:13.476881.476881 helper.py:39]   NaN count (normed): 8448
WARNING 05-06 10:01:13.481941.481941 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:01:13.481336.481336 helper.py:50]   NaN count: 786432
WARNING 05-06 10:01:13.481297.481297 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:01:13.483691.483691 cuda_h.py:27] end init_inputs_tokens cost 7.873 ms
DEBUG 05-06 10:01:13.483773.483773 lmp.py:1504] decode step 30 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:01:13.483920.483920 lmp.py:1510] ---- decode step 30 layer 0 ----
DEBUG 05-06 10:01:13.488118.488118 cuda_h.py:27] end decode_layer cost 4.914 ms
DEBUG 05-06 10:01:13.488014.488014 lmp.py:1510] ---- decode step 30 layer 1 ----
DEBUG 05-06 10:01:13.493542.493542 cuda_h.py:27] end decode_layer cost 4.878 ms
DEBUG 05-06 10:01:13.493338.493338 lmp.py:1510] ---- decode step 30 layer 2 ----
DEBUG 05-06 10:01:13.498325.498325 cuda_h.py:27] end decode_layer cost 4.724 ms
DEBUG 05-06 10:01:13.498121.498121 lmp.py:1510] ---- decode step 30 layer 3 ----
DEBUG 05-06 10:01:13.503805.503805 cuda_h.py:27] end decode_layer cost 4.782 ms
DEBUG 05-06 10:01:13.503078.503078 lmp.py:1510] ---- decode step 30 layer 4 ----
DEBUG 05-06 10:01:13.507066.507066 cuda_h.py:27] end decode_layer cost 4.760 ms
DEBUG 05-06 10:01:13.507247.507247 lmp.py:1510] ---- decode step 30 layer 5 ----
DEBUG 05-06 10:01:13.513369.513369 cuda_h.py:27] end decode_layer cost 5.035 ms
DEBUG 05-06 10:01:13.513881.513881 lmp.py:1510] ---- decode step 30 layer 6 ----
DEBUG 05-06 10:01:13.517379.517379 cuda_h.py:27] end decode_layer cost 4.786 ms
DEBUG 05-06 10:01:13.517938.517938 lmp.py:1510] ---- decode step 30 layer 7 ----
DEBUG 05-06 10:01:13.522898.522898 cuda_h.py:27] end decode_layer cost 4.740 ms
DEBUG 05-06 10:01:13.522171.522171 lmp.py:1510] ---- decode step 30 layer 8 ----
DEBUG 05-06 10:01:13.527072.527072 cuda_h.py:27] end decode_layer cost 4.731 ms
DEBUG 05-06 10:01:13.527630.527630 lmp.py:1510] ---- decode step 30 layer 9 ----
DEBUG 05-06 10:01:13.532328.532328 cuda_h.py:27] end decode_layer cost 4.828 ms
DEBUG 05-06 10:01:13.532079.532079 lmp.py:1510] ---- decode step 30 layer 10 ----
DEBUG 05-06 10:01:13.537904.537904 cuda_h.py:27] end decode_layer cost 4.676 ms
DEBUG 05-06 10:01:13.537986.537986 lmp.py:1510] ---- decode step 30 layer 11 ----
DEBUG 05-06 10:01:13.542191.542191 cuda_h.py:27] end decode_layer cost 4.920 ms
DEBUG 05-06 10:01:13.542987.542987 lmp.py:1510] ---- decode step 30 layer 12 ----
DEBUG 05-06 10:01:13.546150.546150 cuda_h.py:27] end decode_layer cost 4.643 ms
DEBUG 05-06 10:01:13.546470.546470 lmp.py:1510] ---- decode step 30 layer 13 ----
DEBUG 05-06 10:01:13.551558.551558 cuda_h.py:27] end decode_layer cost 4.798 ms
DEBUG 05-06 10:01:13.551739.551739 lmp.py:1510] ---- decode step 30 layer 14 ----
DEBUG 05-06 10:01:13.556357.556357 cuda_h.py:27] end decode_layer cost 4.804 ms
DEBUG 05-06 10:01:13.556491.556491 lmp.py:1510] ---- decode step 30 layer 15 ----
DEBUG 05-06 10:01:13.561397.561397 cuda_h.py:27] end decode_layer cost 4.700 ms
DEBUG 05-06 10:01:13.561002.561002 lmp.py:1510] ---- decode step 30 layer 16 ----
DEBUG 05-06 10:01:13.566225.566225 cuda_h.py:27] end decode_layer cost 4.689 ms
DEBUG 05-06 10:01:13.566783.566783 lmp.py:1510] ---- decode step 30 layer 17 ----
DEBUG 05-06 10:01:13.571451.571451 cuda_h.py:27] end decode_layer cost 4.910 ms
DEBUG 05-06 10:01:13.571248.571248 lmp.py:1510] ---- decode step 30 layer 18 ----
DEBUG 05-06 10:01:13.575809.575809 cuda_h.py:27] end decode_layer cost 4.692 ms
DEBUG 05-06 10:01:13.575851.575851 lmp.py:1510] ---- decode step 30 layer 19 ----
DEBUG 05-06 10:01:13.580333.580333 cuda_h.py:27] end decode_layer cost 4.704 ms
DEBUG 05-06 10:01:13.580177.580177 lmp.py:1510] ---- decode step 30 layer 20 ----
DEBUG 05-06 10:01:13.585577.585577 cuda_h.py:27] end decode_layer cost 4.644 ms
DEBUG 05-06 10:01:13.585136.585136 lmp.py:1510] ---- decode step 30 layer 21 ----
DEBUG 05-06 10:01:13.590708.590708 cuda_h.py:27] end decode_layer cost 4.840 ms
DEBUG 05-06 10:01:13.590028.590028 lmp.py:1510] ---- decode step 30 layer 22 ----
DEBUG 05-06 10:01:13.594749.594749 cuda_h.py:27] end decode_layer cost 4.704 ms
DEBUG 05-06 10:01:13.595023.595023 lmp.py:1510] ---- decode step 30 layer 23 ----
DEBUG 05-06 10:01:13.600648.600648 cuda_h.py:27] end decode_layer cost 5.583 ms
DEBUG 05-06 10:01:13.600202.600202 lmp.py:1510] ---- decode step 30 layer 24 ----
DEBUG 05-06 10:01:13.605324.605324 cuda_h.py:27] end decode_layer cost 5.037 ms
DEBUG 05-06 10:01:13.605883.605883 lmp.py:1510] ---- decode step 30 layer 25 ----
DEBUG 05-06 10:01:13.610179.610179 cuda_h.py:27] end decode_layer cost 4.707 ms
DEBUG 05-06 10:01:13.610261.610261 lmp.py:1510] ---- decode step 30 layer 26 ----
DEBUG 05-06 10:01:13.615470.615470 cuda_h.py:27] end decode_layer cost 4.642 ms
DEBUG 05-06 10:01:13.615313.615313 lmp.py:1510] ---- decode step 30 layer 27 ----
DEBUG 05-06 10:01:13.620451.620451 cuda_h.py:27] end decode_layer cost 4.731 ms
DEBUG 05-06 10:01:13.620461.620461 lmp.py:1510] ---- decode step 30 layer 28 ----
DEBUG 05-06 10:01:13.624868.624868 cuda_h.py:27] end decode_layer cost 4.649 ms
DEBUG 05-06 10:01:13.624188.624188 lmp.py:1510] ---- decode step 30 layer 29 ----
DEBUG 05-06 10:01:13.629869.629869 cuda_h.py:27] end decode_layer cost 4.921 ms
DEBUG 05-06 10:01:13.629793.629793 cuda_h.py:27] end decode_step cost 154.300 ms
INFO 05-06 10:01:13.629032.629032 lmp.py:1558] decode step 30 time: 0.15433645248413086 seconds
INFO 05-06 10:01:13.629538.629538 lmp.py:1564] average decode time from step 5: 0.1558490257996779 seconds
Time taken: 9.651281975209713 seconds
X512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x5cb0d39ff940, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
CPUInfer[0x5cb0d4536a20]: Goodbye
CPUInfer[0x5cb0b98f1c80]: Goodbye
