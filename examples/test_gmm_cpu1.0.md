here pin
INFO 05-06 10:38:10.540174.540174 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-06 10:38:11.082842.082842 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-06 10:38:11.510156.510156 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-06 10:38:11.510123.510123 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 0.970s
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
INFO 05-06 10:38:19.526276.526276 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-06 10:38:19.953873.953873 cuda_h.py:27] end init_cmv_hmv cost 429.758 ms
DEBUG 05-06 10:38:19.962194.962194 cuda_memory_view.py:1366] 
DEBUG 05-06 10:38:19.962194.962194 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.002489805221557617
DEBUG 05-06 10:38:19.978991.978991 mlpmodule.py:993] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-06 10:38:19.979517.979517 cuda_memory_view.py:1370] 
DEBUG 05-06 10:38:19.979517.979517 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.016451597213745117
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-06 10:38:21.843396.843396 lmp.py:368] init kt-kernel layer 0 ok
INFO 05-06 10:38:22.622484.622484 lmp.py:368] init kt-kernel layer 1 ok
INFO 05-06 10:38:23.425834.425834 lmp.py:368] init kt-kernel layer 2 ok
INFO 05-06 10:38:24.233106.233106 lmp.py:368] init kt-kernel layer 3 ok
INFO 05-06 10:38:25.068298.068298 lmp.py:368] init kt-kernel layer 4 ok
INFO 05-06 10:38:25.879331.879331 lmp.py:368] init kt-kernel layer 5 ok
INFO 05-06 10:38:26.693505.693505 lmp.py:368] init kt-kernel layer 6 ok
INFO 05-06 10:38:27.542067.542067 lmp.py:368] init kt-kernel layer 7 ok
INFO 05-06 10:38:28.376881.376881 lmp.py:368] init kt-kernel layer 8 ok
INFO 05-06 10:38:29.201625.201625 lmp.py:368] init kt-kernel layer 9 ok
INFO 05-06 10:38:30.019241.019241 lmp.py:368] init kt-kernel layer 10 ok
INFO 05-06 10:38:30.833504.833504 lmp.py:368] init kt-kernel layer 11 ok
INFO 05-06 10:38:31.633471.633471 lmp.py:368] init kt-kernel layer 12 ok
INFO 05-06 10:38:32.479157.479157 lmp.py:368] init kt-kernel layer 13 ok
INFO 05-06 10:38:33.320197.320197 lmp.py:368] init kt-kernel layer 14 ok
INFO 05-06 10:38:34.147786.147786 lmp.py:368] init kt-kernel layer 15 ok
INFO 05-06 10:38:34.988383.988383 lmp.py:368] init kt-kernel layer 16 ok
INFO 05-06 10:38:35.816587.816587 lmp.py:368] init kt-kernel layer 17 ok
INFO 05-06 10:38:36.653613.653613 lmp.py:368] init kt-kernel layer 18 ok
INFO 05-06 10:38:37.485339.485339 lmp.py:368] init kt-kernel layer 19 ok
INFO 05-06 10:38:38.315757.315757 lmp.py:368] init kt-kernel layer 20 ok
INFO 05-06 10:38:39.155550.155550 lmp.py:368] init kt-kernel layer 21 ok
INFO 05-06 10:38:39.959085.959085 lmp.py:368] init kt-kernel layer 22 ok
CPUInfer[0x59f7376711e0]: Hello
WorkerPool[0x59f7376b1880] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x59f746752970]: Hello
WorkerPool[0x59f747127b60] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVINFO 05-06 10:38:40.768613.768613 lmp.py:368] init kt-kernel layer 23 ok
INFO 05-06 10:38:41.597784.597784 lmp.py:368] init kt-kernel layer 24 ok
INFO 05-06 10:38:42.420091.420091 lmp.py:368] init kt-kernel layer 25 ok
INFO 05-06 10:38:43.241024.241024 lmp.py:368] init kt-kernel layer 26 ok
INFO 05-06 10:38:44.043922.043922 lmp.py:368] init kt-kernel layer 27 ok
INFO 05-06 10:38:44.843102.843102 lmp.py:368] init kt-kernel layer 28 ok
INFO 05-06 10:38:45.665266.665266 lmp.py:368] init kt-kernel layer 29 ok
generate input ids cost 0.05260825157165527 s
DEBUG 05-06 10:38:49.015840.015840 cuda_h.py:27] end generate_input_ids cost 3294.391 ms
DEBUG 05-06 10:38:49.015908.015908 cuda_h.py:27] end init_cache cost 0.050 ms
INFO 05-06 10:38:49.029643.029643 lmp.py:2341] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6629859268, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7273408761529578, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-06 10:38:49.029077.029077 lmp.py:2359] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.029429.029429 lmp.py:2359] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.029960.029960 lmp.py:2359] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.029154.029154 lmp.py:2359] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.029062.029062 lmp.py:2359] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.030394.030394 lmp.py:2359] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.030349.030349 lmp.py:2359] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.030085.030085 lmp.py:2359] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.030471.030471 lmp.py:2359] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.030618.030618 lmp.py:2359] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.030795.030795 lmp.py:2359] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.030273.030273 lmp.py:2359] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.031730.031730 lmp.py:2359] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.031208.031208 lmp.py:2359] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.031122.031122 lmp.py:2359] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.031030.031030 lmp.py:2359] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.031559.031559 lmp.py:2359] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.031468.031468 lmp.py:2359] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.031026.031026 lmp.py:2359] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.031497.031497 lmp.py:2359] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.031220.031220 lmp.py:2359] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.032399.032399 lmp.py:2359] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.032453.032453 lmp.py:2359] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.032560.032560 lmp.py:2359] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.032614.032614 lmp.py:2359] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.032216.032216 lmp.py:2359] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.032985.032985 lmp.py:2359] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.032547.032547 lmp.py:2359] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.032046.032046 lmp.py:2359] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:49.032031.032031 lmp.py:2359] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 10:38:49.314919.314919 cuda_h.py:27] end init_loading_placement cost 298.118 ms
DEBUG 05-06 10:38:49.314036.314036 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:38:49.314555.314555 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:38:49 client.py:72] load_into_gpu: gemma4-26B-A4B, 021f41ce-aa12-45e9-b584-f130f49a3efc
INFO 05-06 10:38:49 client.py:135] Model loaded: gemma4-26B-A4B, 021f41ce-aa12-45e9-b584-f130f49a3efc
INFO 05-06 10:38:49 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 021f41ce-aa12-45e9-b584-f130f49a3efc
INFO 05-06 10:38:49 client.py:212] Model loaded
DEBUG 05-06 10:38:49.841078.841078 cuda_h.py:27] end init_general_sagl_loading_async cost 527.760 ms
INFO 05-06 10:38:49.888664.888664 lmp.py:2862] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 10:38:49.988693.988693 cuda_h.py:27] end restore_state_dict cost 99.216 ms
DEBUG 05-06 10:38:49.988479.988479 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:38:49.988262.988262 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:38:49 client.py:72] load_into_gpu: gemma4-26B-A4B, 57140fe4-b0d5-43ab-bda5-f0750bee423e
INFO 05-06 10:38:50 client.py:135] Model loaded: gemma4-26B-A4B, 57140fe4-b0d5-43ab-bda5-f0750bee423e
DEBUG 05-06 10:38:50.059803.059803 cuda_h.py:27] end init_experts_loading_async cost 70.974 ms
DEBUG 05-06 10:38:50.088830.088830 cuda_h.py:27] end init_inputs_tokens cost 29.351 ms
DEBUG 05-06 10:38:50.088653.088653 lmp.py:824] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 10:38:50.226442.226442 cuda_h.py:27] end *sagl cost 137.193 ms
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 75, 115, 83, 127, 31, 3, 67, 51, 55, 7, 91, 39, 103, 47, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 92, 116, 112, 64, 72, 48, 52, 32, 104, 16, 0, 68, 124, 101, 109, 85, 49, 45, 65, 93, 69, 5, 9, 13, 73, 77, 37, 89, 25, 105, 125, 117, 41, 113, 21, 121, 1, 53, 33, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18, 70, 110, 118, 122, 78, 26, 54, 74, 22, 50, 126, 46, 90], 'token_total': 4096, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 75: 29, 115: 29, 83: 33, 127: 33, 31: 34, 3: 46, 67: 47, 51: 48, 55: 51, 7: 94, 91: 99, 39: 176, 103: 178, 47: 318, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 92: 16, 116: 18, 112: 23, 64: 27, 72: 35, 48: 41, 52: 42, 32: 43, 104: 43, 16: 48, 0: 73, 68: 170, 124: 178, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 9: 17, 13: 17, 73: 19, 77: 19, 37: 20, 89: 20, 25: 24, 105: 24, 125: 25, 117: 26, 41: 27, 113: 40, 21: 48, 121: 65, 1: 75, 53: 205, 33: 210, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18, 70: 26, 110: 27, 118: 29, 122: 35, 78: 36, 26: 59, 54: 59, 74: 61, 22: 64, 50: 110, 126: 115, 46: 119, 90: 154}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:50.362646.362646 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 48.390ms | allocate_experts_across_cpu_gpu: 0.277ms
INFO 05-06 10:38:50.362836.362836 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.0531158447265625e-06 seconds
INFO 05-06 10:38:50.363877.363877 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007932186126708984 seconds
INFO 05-06 10:38:50.379379.379379 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.003832101821899414 seconds
INFO 05-06 10:38:50.380604.380604 lmp.py:1484] [layer_moe_fused] experts compute time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:50.947525.947525 lmp.py:1496] [layer_moe_fused] to time: 0.00017833709716796875 seconds
INFO 05-06 10:38:50.948207.948207 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.5676612854003906 seconds
DEBUG 05-06 10:38:50.948809.948809 cuda_h.py:27] end *layer_moe_fused cost 634.662 ms
DEBUG 05-06 10:38:50.949241.949241 cuda_h.py:27] end prefill_layer cost 860.506 ms
DEBUG 05-06 10:38:50.949211.949211 lmp.py:841] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 10:38:50.949603.949603 lmp.py:824] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 10:38:50.953723.953723 cuda_h.py:27] end *sagl cost 4.138 ms
experts_cpu_alloc {'expert_ids': [39, 115, 23, 43, 55, 31, 103, 83, 123, 87, 91, 11, 27, 95, 35, 119, 3, 79, 7, 59, 127, 47, 51, 99, 67, 24, 44, 88, 32, 40, 72, 112, 84, 108, 0, 60, 116, 76, 48, 56, 4, 104, 92, 64, 120, 124, 96, 80, 100, 12, 20, 28, 8, 68, 52, 61, 77, 117, 33, 45, 81, 57, 125, 89, 29, 121, 93, 37, 69, 85, 101, 9, 21, 105, 49, 1, 73, 65, 53, 25, 5, 109, 97, 13, 86, 6, 62, 2, 26, 18, 38, 110, 14, 66, 78, 98, 50, 74, 90, 34, 94, 42, 46, 106, 54, 118, 122, 22, 10, 82, 30], 'token_total': 4096, 'token_per_expert': {39: 1, 115: 1, 23: 2, 43: 3, 55: 3, 31: 4, 103: 4, 83: 6, 123: 6, 87: 7, 91: 8, 11: 10, 27: 15, 95: 22, 35: 23, 119: 23, 3: 25, 79: 27, 7: 37, 59: 38, 127: 46, 47: 47, 51: 58, 99: 125, 67: 140, 24: 1, 44: 2, 88: 3, 32: 4, 40: 4, 72: 5, 112: 5, 84: 6, 108: 6, 0: 7, 60: 7, 116: 7, 76: 9, 48: 10, 56: 11, 4: 15, 104: 21, 92: 23, 64: 26, 120: 28, 124: 31, 96: 36, 80: 51, 100: 53, 12: 55, 20: 59, 28: 69, 8: 108, 68: 180, 52: 277, 61: 1, 77: 1, 117: 1, 33: 3, 45: 4, 81: 4, 57: 5, 125: 5, 89: 6, 29: 7, 121: 7, 93: 9, 37: 11, 69: 11, 85: 11, 101: 13, 9: 16, 21: 16, 105: 28, 49: 30, 1: 31, 73: 35, 65: 36, 53: 38, 25: 48, 5: 108, 109: 110, 97: 123, 13: 278, 86: 1, 6: 2, 62: 3, 2: 4, 26: 4, 18: 5, 38: 8, 110: 8, 14: 9, 66: 10, 78: 12, 98: 15, 50: 16, 74: 18, 90: 19, 34: 33, 94: 33, 42: 39, 46: 40, 106: 42, 54: 58, 118: 63, 122: 123, 22: 125, 10: 171, 82: 189, 30: 250}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:50.956181.956181 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.656ms | allocate_experts_across_cpu_gpu: 0.415ms
INFO 05-06 10:38:50.956403.956403 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:50.957375.957375 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006227493286132812 seconds
INFO 05-06 10:38:50.958230.958230 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008587837219238281 seconds
INFO 05-06 10:38:50.958035.958035 lmp.py:1484] [layer_moe_fused] experts compute time: 1.6689300537109375e-06 seconds
INFO 05-06 10:38:50.999137.999137 lmp.py:1496] [layer_moe_fused] to time: 0.00018739700317382812 seconds
INFO 05-06 10:38:51.000319.000319 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04126095771789551 seconds
DEBUG 05-06 10:38:51.000038.000038 cuda_h.py:27] end *layer_moe_fused cost 45.243 ms
DEBUG 05-06 10:38:51.001264.001264 cuda_h.py:27] end prefill_layer cost 51.776 ms
DEBUG 05-06 10:38:51.001426.001426 lmp.py:841] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 10:38:51.001195.001195 lmp.py:824] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 10:38:51.003386.003386 cuda_h.py:27] end *sagl cost 2.263 ms
experts_cpu_alloc {'expert_ids': [75, 67, 99, 111, 115, 27, 95, 63, 43, 71, 23, 35, 3, 83, 119, 31, 123, 103, 107, 51, 91, 7, 55, 15, 59, 19, 127, 11, 12, 120, 40, 0, 116, 64, 88, 96, 24, 72, 44, 100, 8, 52, 56, 28, 36, 4, 124, 60, 104, 20, 84, 80, 76, 48, 108, 45, 25, 21, 5, 113, 121, 61, 57, 85, 105, 17, 77, 69, 33, 49, 53, 97, 65, 109, 81, 37, 29, 125, 1, 13, 9, 41, 66, 86, 6, 26, 50, 42, 114, 82, 58, 70, 126, 46, 98, 122, 110, 78, 34, 14, 106, 118, 18, 90, 102, 54, 62], 'token_total': 4096, 'token_per_expert': {75: 1, 67: 2, 99: 2, 111: 4, 115: 4, 27: 6, 95: 11, 63: 17, 43: 18, 71: 18, 23: 19, 35: 20, 3: 23, 83: 23, 119: 24, 31: 25, 123: 25, 103: 26, 107: 27, 51: 30, 91: 36, 7: 55, 55: 69, 15: 94, 59: 112, 19: 119, 127: 134, 11: 192, 12: 2, 120: 5, 40: 7, 0: 8, 116: 10, 64: 12, 88: 12, 96: 14, 24: 15, 72: 16, 44: 17, 100: 17, 8: 18, 52: 18, 56: 18, 28: 19, 36: 19, 4: 26, 124: 29, 60: 46, 104: 51, 20: 54, 84: 64, 80: 67, 76: 69, 48: 70, 108: 212, 45: 2, 25: 3, 21: 5, 5: 8, 113: 8, 121: 8, 61: 12, 57: 16, 85: 16, 105: 17, 17: 18, 77: 20, 69: 23, 33: 24, 49: 26, 53: 38, 97: 38, 65: 40, 109: 43, 81: 73, 37: 80, 29: 89, 125: 90, 1: 95, 13: 105, 9: 115, 41: 140, 66: 1, 86: 1, 6: 4, 26: 4, 50: 5, 42: 6, 114: 6, 82: 12, 58: 15, 70: 16, 126: 17, 46: 19, 98: 23, 122: 26, 110: 30, 78: 31, 34: 33, 14: 36, 106: 40, 118: 64, 18: 67, 90: 78, 102: 92, 54: 124, 62: 143}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.006221.006221 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.593ms | allocate_experts_across_cpu_gpu: 0.388ms
INFO 05-06 10:38:51.006218.006218 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.0994415283203125e-06 seconds
INFO 05-06 10:38:51.006300.006300 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005674362182617188 seconds
INFO 05-06 10:38:51.008303.008303 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009133815765380859 seconds
INFO 05-06 10:38:51.008598.008598 lmp.py:1484] [layer_moe_fused] experts compute time: 1.1920928955078125e-06 seconds
INFO 05-06 10:38:51.044892.044892 lmp.py:1496] [layer_moe_fused] to time: 0.0001800060272216797 seconds
INFO 05-06 10:38:51.044975.044975 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03632998466491699 seconds
DEBUG 05-06 10:38:51.123426.123426 cuda_h.py:27] end *layer_moe_fused cost 118.635 ms
DEBUG 05-06 10:38:51.124892.124892 cuda_h.py:27] end prefill_layer cost 123.631 ms
DEBUG 05-06 10:38:51.125765.125765 lmp.py:841] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 10:38:51.125795.125795 lmp.py:824] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 10:38:51.130914.130914 cuda_h.py:27] end *sagl cost 4.700 ms
experts_cpu_alloc {'expert_ids': [23, 35, 91, 127, 63, 27, 31, 67, 43, 123, 111, 19, 39, 15, 59, 3, 119, 51, 11, 107, 83, 95, 71, 75, 72, 20, 80, 36, 60, 16, 32, 48, 116, 100, 24, 56, 40, 8, 44, 64, 0, 104, 108, 120, 68, 52, 76, 84, 96, 88, 92, 4, 28, 21, 29, 41, 89, 117, 65, 57, 1, 33, 13, 101, 61, 77, 109, 17, 73, 69, 25, 53, 5, 97, 9, 93, 121, 85, 46, 126, 18, 94, 82, 98, 42, 30, 86, 6, 26, 2, 110, 114, 54, 58, 70, 74, 10, 118, 122, 34, 14, 66, 62, 102, 22, 78, 50], 'token_total': 4096, 'token_per_expert': {23: 3, 35: 3, 91: 4, 127: 6, 63: 8, 27: 13, 31: 14, 67: 14, 43: 15, 123: 22, 111: 25, 19: 28, 39: 28, 15: 30, 59: 31, 3: 33, 119: 33, 51: 39, 11: 44, 107: 48, 83: 53, 95: 69, 71: 90, 75: 93, 72: 1, 20: 2, 80: 2, 36: 3, 60: 6, 16: 7, 32: 8, 48: 10, 116: 10, 100: 11, 24: 12, 56: 12, 40: 15, 8: 16, 44: 25, 64: 25, 0: 40, 104: 44, 108: 47, 120: 53, 68: 55, 52: 64, 76: 67, 84: 67, 96: 68, 88: 69, 92: 73, 4: 74, 28: 82, 21: 2, 29: 4, 41: 4, 89: 5, 117: 5, 65: 7, 57: 8, 1: 9, 33: 13, 13: 16, 101: 18, 61: 19, 77: 19, 109: 31, 17: 44, 73: 44, 69: 46, 25: 47, 53: 59, 5: 62, 97: 65, 9: 85, 93: 87, 121: 101, 85: 135, 46: 1, 126: 1, 18: 3, 94: 3, 82: 4, 98: 5, 42: 8, 30: 9, 86: 12, 6: 21, 26: 22, 2: 27, 110: 27, 114: 30, 54: 38, 58: 38, 70: 43, 74: 49, 10: 55, 118: 56, 122: 56, 34: 84, 14: 88, 66: 102, 62: 113, 102: 115, 22: 117, 78: 146, 50: 174}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.134602.134602 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 1.000ms | allocate_experts_across_cpu_gpu: 0.652ms
INFO 05-06 10:38:51.134886.134886 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.76837158203125e-06 seconds
INFO 05-06 10:38:51.135322.135322 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008637905120849609 seconds
INFO 05-06 10:38:51.136812.136812 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008594989776611328 seconds
INFO 05-06 10:38:51.137574.137574 lmp.py:1484] [layer_moe_fused] experts compute time: 1.9073486328125e-06 seconds
INFO 05-06 10:38:51.176786.176786 lmp.py:1496] [layer_moe_fused] to time: 0.00017142295837402344 seconds
INFO 05-06 10:38:51.176848.176848 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03978991508483887 seconds
DEBUG 05-06 10:38:51.177467.177467 cuda_h.py:27] end *layer_moe_fused cost 44.333 ms
DEBUG 05-06 10:38:51.177541.177541 cuda_h.py:27] end prefill_layer cost 52.551 ms
DEBUG 05-06 10:38:51.178358.178358 lmp.py:841] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 10:38:51.178413.178413 lmp.py:824] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 10:38:51.180383.180383 cuda_h.py:27] end *sagl cost 2.260 ms
experts_cpu_alloc {'expert_ids': [35, 95, 31, 79, 7, 103, 107, 91, 75, 15, 71, 123, 87, 47, 3, 39, 19, 67, 55, 27, 83, 51, 111, 115, 23, 119, 43, 59, 63, 12, 44, 56, 120, 80, 40, 84, 108, 88, 36, 64, 52, 116, 92, 96, 4, 60, 104, 20, 28, 124, 76, 32, 24, 8, 69, 121, 21, 37, 45, 101, 109, 25, 77, 57, 81, 73, 117, 17, 97, 105, 61, 49, 5, 85, 125, 93, 29, 53, 1, 113, 89, 70, 110, 18, 46, 58, 66, 114, 122, 6, 50, 126, 34, 38, 90, 78, 30, 98, 118, 62, 94, 86, 82, 54, 26, 22, 74, 106], 'token_total': 4096, 'token_per_expert': {35: 1, 95: 1, 31: 3, 79: 3, 7: 5, 103: 8, 107: 15, 91: 16, 75: 20, 15: 21, 71: 25, 123: 32, 87: 33, 47: 34, 3: 37, 39: 50, 19: 51, 67: 51, 55: 58, 27: 65, 83: 83, 51: 87, 111: 87, 115: 108, 23: 109, 119: 137, 43: 146, 59: 177, 63: 269, 12: 2, 44: 4, 56: 5, 120: 7, 80: 8, 40: 12, 84: 12, 108: 14, 88: 15, 36: 17, 64: 19, 52: 20, 116: 25, 92: 30, 96: 32, 4: 33, 60: 34, 104: 34, 20: 35, 28: 38, 124: 46, 76: 55, 32: 56, 24: 97, 8: 172, 69: 2, 121: 3, 21: 6, 37: 6, 45: 6, 101: 7, 109: 8, 25: 9, 77: 10, 57: 11, 81: 12, 73: 15, 117: 18, 17: 20, 97: 20, 105: 21, 61: 24, 49: 34, 5: 38, 85: 43, 125: 43, 93: 44, 29: 46, 53: 52, 1: 75, 113: 80, 89: 113, 70: 1, 110: 1, 18: 2, 46: 2, 58: 2, 66: 3, 114: 3, 122: 4, 6: 5, 50: 6, 126: 6, 34: 10, 38: 10, 90: 10, 78: 17, 30: 27, 98: 27, 118: 28, 62: 29, 94: 30, 86: 38, 82: 59, 54: 74, 26: 83, 22: 87, 74: 98, 106: 114}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.182409.182409 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.572ms | allocate_experts_across_cpu_gpu: 0.394ms
INFO 05-06 10:38:51.182822.182822 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:51.183303.183303 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.000606536865234375 seconds
INFO 05-06 10:38:51.184377.184377 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009205341339111328 seconds
INFO 05-06 10:38:51.185777.185777 lmp.py:1484] [layer_moe_fused] experts compute time: 3.0994415283203125e-06 seconds
INFO 05-06 10:38:51.219524.219524 lmp.py:1496] [layer_moe_fused] to time: 0.00017261505126953125 seconds
INFO 05-06 10:38:51.220402.220402 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0350337028503418 seconds
DEBUG 05-06 10:38:51.220073.220073 cuda_h.py:27] end *layer_moe_fused cost 38.993 ms
DEBUG 05-06 10:38:51.221292.221292 cuda_h.py:27] end prefill_layer cost 43.267 ms
DEBUG 05-06 10:38:51.221109.221109 lmp.py:841] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 10:38:51.221184.221184 lmp.py:824] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 10:38:51.267753.267753 cuda_h.py:27] end *sagl cost 45.829 ms
experts_cpu_alloc {'expert_ids': [3, 15, 51, 19, 27, 115, 7, 83, 119, 79, 107, 31, 55, 75, 67, 63, 23, 43, 87, 99, 123, 111, 127, 39, 71, 124, 8, 48, 92, 32, 56, 68, 52, 84, 44, 0, 100, 96, 80, 76, 120, 116, 60, 104, 88, 28, 4, 24, 72, 36, 64, 16, 20, 112, 1, 21, 45, 69, 121, 17, 81, 57, 105, 77, 53, 37, 113, 125, 5, 29, 93, 73, 117, 9, 13, 61, 33, 49, 101, 78, 110, 30, 50, 82, 122, 26, 58, 34, 38, 54, 86, 98, 114, 10, 62, 6, 102, 106, 18, 14, 118, 46, 126, 74, 70, 94, 42, 2, 22], 'token_total': 4096, 'token_per_expert': {3: 1, 15: 1, 51: 3, 19: 5, 27: 6, 115: 7, 7: 8, 83: 10, 119: 10, 79: 17, 107: 17, 31: 23, 55: 24, 75: 24, 67: 26, 63: 30, 23: 35, 43: 35, 87: 58, 99: 58, 123: 63, 111: 83, 127: 112, 39: 157, 71: 272, 124: 1, 8: 2, 48: 3, 92: 3, 32: 4, 56: 4, 68: 9, 52: 12, 84: 12, 44: 14, 0: 21, 100: 23, 96: 25, 80: 30, 76: 35, 120: 38, 116: 40, 60: 42, 104: 47, 88: 52, 28: 64, 4: 66, 24: 67, 72: 74, 36: 87, 64: 104, 16: 118, 20: 142, 112: 142, 1: 1, 21: 1, 45: 1, 69: 1, 121: 1, 17: 2, 81: 2, 57: 3, 105: 4, 77: 5, 53: 6, 37: 11, 113: 11, 125: 26, 5: 29, 29: 38, 93: 46, 73: 55, 117: 55, 9: 59, 13: 59, 61: 77, 33: 104, 49: 173, 101: 305, 78: 1, 110: 2, 30: 3, 50: 3, 82: 3, 122: 3, 26: 4, 58: 5, 34: 7, 38: 7, 54: 7, 86: 7, 98: 10, 114: 10, 10: 11, 62: 11, 6: 12, 102: 12, 106: 13, 18: 16, 14: 17, 118: 24, 46: 31, 126: 34, 74: 41, 70: 43, 94: 44, 42: 68, 2: 99, 22: 107}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.269964.269964 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.607ms | allocate_experts_across_cpu_gpu: 0.414ms
INFO 05-06 10:38:51.269762.269762 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:51.270047.270047 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005106925964355469 seconds
INFO 05-06 10:38:51.271870.271870 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0010943412780761719 seconds
INFO 05-06 10:38:51.272869.272869 lmp.py:1484] [layer_moe_fused] experts compute time: 2.1457672119140625e-06 seconds
INFO 05-06 10:38:51.308627.308627 lmp.py:1496] [layer_moe_fused] to time: 0.00017380714416503906 seconds
INFO 05-06 10:38:51.308657.308657 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.036348581314086914 seconds
DEBUG 05-06 10:38:51.309322.309322 cuda_h.py:27] end *layer_moe_fused cost 40.407 ms
DEBUG 05-06 10:38:51.309449.309449 cuda_h.py:27] end prefill_layer cost 88.214 ms
DEBUG 05-06 10:38:51.309127.309127 lmp.py:841] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 10:38:51.309374.309374 lmp.py:824] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 10:38:51.312502.312502 cuda_h.py:27] end *sagl cost 2.254 ms
experts_cpu_alloc {'expert_ids': [31, 47, 59, 67, 111, 83, 3, 11, 15, 19, 91, 43, 127, 103, 123, 27, 51, 95, 107, 71, 119, 79, 75, 23, 115, 87, 35, 99, 84, 88, 100, 112, 52, 92, 16, 40, 72, 124, 4, 60, 120, 20, 0, 80, 76, 116, 28, 56, 44, 36, 24, 32, 96, 104, 108, 64, 68, 17, 21, 101, 109, 81, 33, 49, 97, 37, 41, 89, 105, 73, 57, 85, 125, 77, 5, 69, 113, 117, 1, 9, 121, 13, 53, 65, 93, 25, 118, 114, 38, 22, 82, 30, 18, 74, 110, 14, 42, 126, 6, 70, 2, 10, 58, 50, 122, 46, 62, 26, 78, 98, 94, 34, 90, 86, 106, 102], 'token_total': 4096, 'token_per_expert': {31: 2, 47: 2, 59: 3, 67: 3, 111: 4, 83: 5, 3: 8, 11: 8, 15: 8, 19: 8, 91: 11, 43: 12, 127: 15, 103: 16, 123: 25, 27: 26, 51: 26, 95: 26, 107: 34, 71: 37, 119: 42, 79: 44, 75: 49, 23: 76, 115: 89, 87: 92, 35: 109, 99: 206, 84: 1, 88: 1, 100: 1, 112: 1, 52: 2, 92: 2, 16: 3, 40: 3, 72: 3, 124: 3, 4: 5, 60: 6, 120: 7, 20: 10, 0: 11, 80: 11, 76: 13, 116: 18, 28: 21, 56: 25, 44: 30, 36: 34, 24: 37, 32: 40, 96: 40, 104: 52, 108: 114, 64: 127, 68: 287, 17: 1, 21: 1, 101: 2, 109: 2, 81: 3, 33: 4, 49: 4, 97: 8, 37: 13, 41: 13, 89: 14, 105: 14, 73: 15, 57: 16, 85: 16, 125: 16, 77: 18, 5: 19, 69: 27, 113: 28, 117: 31, 1: 33, 9: 40, 121: 50, 13: 65, 53: 86, 65: 103, 93: 150, 25: 225, 118: 1, 114: 2, 38: 3, 22: 4, 82: 8, 30: 9, 18: 12, 74: 12, 110: 12, 14: 14, 42: 15, 126: 17, 6: 18, 70: 21, 2: 22, 10: 23, 58: 26, 50: 35, 122: 39, 46: 45, 62: 47, 26: 51, 78: 51, 98: 74, 94: 80, 34: 83, 90: 100, 86: 110, 106: 115, 102: 136}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.314469.314469 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.576ms | allocate_experts_across_cpu_gpu: 0.418ms
INFO 05-06 10:38:51.314413.314413 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:51.315696.315696 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006258487701416016 seconds
INFO 05-06 10:38:51.316341.316341 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008821487426757812 seconds
INFO 05-06 10:38:51.316250.316250 lmp.py:1484] [layer_moe_fused] experts compute time: 1.430511474609375e-06 seconds
INFO 05-06 10:38:51.352643.352643 lmp.py:1496] [layer_moe_fused] to time: 0.00017309188842773438 seconds
INFO 05-06 10:38:51.352051.352051 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.035727500915527344 seconds
DEBUG 05-06 10:38:51.353211.353211 cuda_h.py:27] end *layer_moe_fused cost 39.569 ms
DEBUG 05-06 10:38:51.353861.353861 cuda_h.py:27] end prefill_layer cost 43.818 ms
DEBUG 05-06 10:38:51.353307.353307 lmp.py:841] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 10:38:51.353554.353554 lmp.py:824] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 10:38:51.357691.357691 cuda_h.py:27] end *sagl cost 3.057 ms
experts_cpu_alloc {'expert_ids': [3, 11, 27, 35, 75, 119, 55, 67, 15, 107, 127, 63, 23, 31, 111, 95, 99, 19, 123, 87, 47, 59, 83, 43, 51, 103, 115, 71, 7, 79, 91, 124, 36, 92, 0, 32, 16, 116, 80, 64, 88, 68, 72, 8, 112, 48, 96, 56, 104, 28, 20, 44, 60, 120, 52, 84, 4, 12, 108, 49, 37, 73, 77, 101, 17, 41, 45, 109, 9, 117, 25, 21, 125, 61, 13, 33, 105, 5, 57, 53, 113, 65, 85, 69, 29, 121, 97, 46, 30, 62, 94, 102, 38, 50, 26, 66, 78, 126, 82, 122, 98, 118, 54, 6, 18, 22, 106, 14, 86, 110, 42, 114, 34, 10, 90, 70], 'token_total': 4096, 'token_per_expert': {3: 1, 11: 1, 27: 1, 35: 1, 75: 1, 119: 1, 55: 4, 67: 4, 15: 5, 107: 5, 127: 7, 63: 9, 23: 10, 31: 11, 111: 12, 95: 15, 99: 16, 19: 21, 123: 24, 87: 25, 47: 28, 59: 28, 83: 29, 43: 33, 51: 34, 103: 34, 115: 35, 71: 40, 7: 41, 79: 60, 91: 212, 124: 2, 36: 3, 92: 3, 0: 11, 32: 11, 16: 12, 116: 15, 80: 17, 64: 18, 88: 19, 68: 22, 72: 25, 8: 27, 112: 35, 48: 43, 96: 43, 56: 46, 104: 48, 28: 59, 20: 63, 44: 64, 60: 64, 120: 66, 52: 75, 84: 83, 4: 89, 12: 122, 108: 124, 49: 2, 37: 3, 73: 4, 77: 9, 101: 10, 17: 11, 41: 12, 45: 12, 109: 12, 9: 14, 117: 15, 25: 16, 21: 17, 125: 25, 61: 26, 13: 29, 33: 35, 105: 37, 5: 44, 57: 47, 53: 51, 113: 51, 65: 78, 85: 85, 69: 93, 29: 100, 121: 122, 97: 255, 46: 1, 30: 3, 62: 3, 94: 3, 102: 3, 38: 4, 50: 4, 26: 6, 66: 7, 78: 9, 126: 11, 82: 12, 122: 13, 98: 17, 118: 17, 54: 18, 6: 19, 18: 27, 22: 32, 106: 58, 14: 59, 86: 60, 110: 63, 42: 65, 114: 65, 34: 67, 10: 80, 90: 86, 70: 112}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.360943.360943 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.608ms | allocate_experts_across_cpu_gpu: 0.420ms
INFO 05-06 10:38:51.360211.360211 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.0994415283203125e-06 seconds
INFO 05-06 10:38:51.361876.361876 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006163120269775391 seconds
INFO 05-06 10:38:51.362745.362745 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008981227874755859 seconds
INFO 05-06 10:38:51.362709.362709 lmp.py:1484] [layer_moe_fused] experts compute time: 2.1457672119140625e-06 seconds
INFO 05-06 10:38:51.397684.397684 lmp.py:1496] [layer_moe_fused] to time: 0.00017595291137695312 seconds
INFO 05-06 10:38:51.397323.397323 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.034921884536743164 seconds
DEBUG 05-06 10:38:51.398162.398162 cuda_h.py:27] end *layer_moe_fused cost 38.795 ms
DEBUG 05-06 10:38:51.399765.399765 cuda_h.py:27] end prefill_layer cost 45.265 ms
DEBUG 05-06 10:38:51.399159.399159 lmp.py:841] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 10:38:51.399312.399312 lmp.py:824] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 10:38:51.401198.401198 cuda_h.py:27] end *sagl cost 2.239 ms
experts_cpu_alloc {'expert_ids': [23, 35, 39, 91, 43, 119, 47, 99, 11, 127, 31, 111, 27, 63, 55, 123, 15, 71, 19, 75, 87, 103, 51, 7, 3, 48, 100, 24, 72, 104, 8, 68, 96, 64, 84, 92, 124, 116, 20, 44, 108, 52, 16, 76, 36, 12, 80, 32, 56, 28, 120, 0, 4, 25, 101, 117, 9, 13, 33, 37, 21, 49, 89, 17, 57, 113, 93, 61, 85, 45, 77, 53, 41, 29, 69, 81, 65, 125, 105, 121, 73, 1, 5, 18, 26, 90, 118, 82, 106, 74, 34, 86, 62, 42, 66, 22, 10, 126, 98, 14, 102, 38, 122, 46, 70, 114, 50, 110, 6, 54, 58, 2], 'token_total': 4096, 'token_per_expert': {23: 1, 35: 2, 39: 2, 91: 4, 43: 5, 119: 6, 47: 9, 99: 9, 11: 16, 127: 17, 31: 21, 111: 22, 27: 24, 63: 26, 55: 34, 123: 34, 15: 43, 71: 55, 19: 61, 75: 65, 87: 77, 103: 125, 51: 128, 7: 129, 3: 161, 48: 1, 100: 1, 24: 2, 72: 2, 104: 3, 8: 5, 68: 5, 96: 5, 64: 7, 84: 7, 92: 7, 124: 8, 116: 10, 20: 11, 44: 14, 108: 14, 52: 18, 16: 22, 76: 22, 36: 30, 12: 32, 80: 48, 32: 49, 56: 50, 28: 79, 120: 84, 0: 129, 4: 141, 25: 2, 101: 2, 117: 4, 9: 5, 13: 5, 33: 5, 37: 8, 21: 9, 49: 10, 89: 10, 17: 11, 57: 12, 113: 12, 93: 16, 61: 18, 85: 19, 45: 21, 77: 21, 53: 25, 41: 27, 29: 32, 69: 33, 81: 34, 65: 51, 125: 58, 105: 59, 121: 68, 73: 92, 1: 136, 5: 154, 18: 2, 26: 3, 90: 3, 118: 3, 82: 4, 106: 4, 74: 5, 34: 6, 86: 7, 62: 8, 42: 15, 66: 15, 22: 17, 10: 19, 126: 19, 98: 21, 14: 22, 102: 39, 38: 45, 122: 46, 46: 47, 70: 61, 114: 62, 50: 64, 110: 88, 6: 145, 54: 148, 58: 157, 2: 180}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.403743.403743 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.573ms | allocate_experts_across_cpu_gpu: 0.465ms
INFO 05-06 10:38:51.404554.404554 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-06 seconds
INFO 05-06 10:38:51.404985.404985 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005469322204589844 seconds
INFO 05-06 10:38:51.406643.406643 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.000980377197265625 seconds
INFO 05-06 10:38:51.406300.406300 lmp.py:1484] [layer_moe_fused] experts compute time: 6.9141387939453125e-06 seconds
INFO 05-06 10:38:51.441740.441740 lmp.py:1496] [layer_moe_fused] to time: 0.00017452239990234375 seconds
INFO 05-06 10:38:51.441187.441187 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03512120246887207 seconds
DEBUG 05-06 10:38:51.442636.442636 cuda_h.py:27] end *layer_moe_fused cost 39.584 ms
DEBUG 05-06 10:38:51.443624.443624 cuda_h.py:27] end prefill_layer cost 43.899 ms
DEBUG 05-06 10:38:51.443408.443408 lmp.py:841] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 10:38:51.443847.443847 lmp.py:824] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 10:38:51.445675.445675 cuda_h.py:27] end *sagl cost 2.232 ms
experts_cpu_alloc {'expert_ids': [31, 63, 91, 11, 119, 55, 79, 67, 115, 27, 39, 15, 51, 83, 111, 19, 99, 71, 127, 23, 75, 43, 103, 7, 3, 95, 84, 28, 64, 100, 104, 112, 116, 44, 96, 20, 52, 120, 8, 124, 68, 80, 40, 88, 76, 36, 24, 32, 72, 92, 48, 56, 16, 12, 0, 4, 29, 33, 41, 105, 77, 97, 73, 117, 113, 17, 37, 9, 45, 61, 21, 89, 125, 13, 57, 69, 81, 93, 101, 5, 1, 50, 14, 10, 34, 114, 26, 66, 90, 58, 82, 18, 98, 42, 86, 122, 38, 62, 22, 54, 30, 102, 74, 2, 46, 6, 70, 106], 'token_total': 4096, 'token_per_expert': {31: 1, 63: 1, 91: 1, 11: 3, 119: 4, 55: 5, 79: 5, 67: 6, 115: 12, 27: 19, 39: 23, 15: 24, 51: 25, 83: 25, 111: 26, 19: 27, 99: 27, 71: 29, 127: 33, 23: 39, 75: 78, 43: 94, 103: 101, 7: 139, 3: 140, 95: 206, 84: 1, 28: 3, 64: 3, 100: 3, 104: 3, 112: 4, 116: 4, 44: 5, 96: 5, 20: 7, 52: 7, 120: 8, 8: 10, 124: 11, 68: 13, 80: 15, 40: 16, 88: 18, 76: 30, 36: 32, 24: 34, 32: 34, 72: 34, 92: 42, 48: 55, 56: 67, 16: 95, 12: 115, 0: 136, 4: 168, 29: 2, 33: 4, 41: 4, 105: 7, 77: 10, 97: 10, 73: 12, 117: 12, 113: 13, 17: 15, 37: 15, 9: 18, 45: 18, 61: 28, 21: 32, 89: 33, 125: 34, 13: 37, 57: 38, 69: 54, 81: 58, 93: 88, 101: 95, 5: 141, 1: 160, 50: 1, 14: 2, 10: 4, 34: 4, 114: 4, 26: 5, 66: 5, 90: 5, 58: 6, 82: 7, 18: 9, 98: 10, 42: 16, 86: 17, 122: 18, 38: 21, 62: 22, 22: 26, 54: 28, 30: 29, 102: 50, 74: 89, 2: 129, 46: 131, 6: 133, 70: 142, 106: 174}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.447178.447178 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.566ms | allocate_experts_across_cpu_gpu: 0.400ms
INFO 05-06 10:38:51.447122.447122 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:51.448329.448329 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005495548248291016 seconds
INFO 05-06 10:38:51.449737.449737 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008683204650878906 seconds
INFO 05-06 10:38:51.450209.450209 lmp.py:1484] [layer_moe_fused] experts compute time: 3.337860107421875e-06 seconds
INFO 05-06 10:38:51.484321.484321 lmp.py:1496] [layer_moe_fused] to time: 0.00017571449279785156 seconds
INFO 05-06 10:38:51.485921.485921 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03505420684814453 seconds
DEBUG 05-06 10:38:51.485867.485867 cuda_h.py:27] end *layer_moe_fused cost 38.940 ms
DEBUG 05-06 10:38:51.486000.486000 cuda_h.py:27] end prefill_layer cost 43.151 ms
DEBUG 05-06 10:38:51.486248.486248 lmp.py:841] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 10:38:51.486448.486448 lmp.py:824] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 10:38:51.488815.488815 cuda_h.py:27] end *sagl cost 2.237 ms
experts_cpu_alloc {'expert_ids': [27, 35, 87, 123, 51, 91, 107, 15, 103, 111, 59, 119, 67, 11, 19, 99, 79, 83, 63, 31, 43, 47, 75, 71, 127, 39, 115, 3, 7, 12, 48, 40, 104, 124, 52, 64, 120, 28, 56, 44, 112, 84, 100, 20, 68, 92, 72, 16, 108, 88, 8, 60, 80, 4, 76, 0, 17, 77, 53, 61, 9, 109, 29, 97, 25, 33, 89, 73, 37, 93, 117, 69, 121, 13, 49, 105, 113, 85, 21, 57, 125, 41, 81, 5, 1, 38, 66, 102, 70, 110, 26, 98, 78, 50, 34, 90, 94, 54, 46, 18, 10, 58, 82, 106, 126, 42, 62, 14, 74, 86, 2, 6], 'token_total': 4096, 'token_per_expert': {27: 1, 35: 1, 87: 1, 123: 1, 51: 2, 91: 2, 107: 3, 15: 4, 103: 4, 111: 4, 59: 5, 119: 10, 67: 11, 11: 14, 19: 15, 99: 17, 79: 18, 83: 18, 63: 20, 31: 25, 43: 34, 47: 34, 75: 35, 71: 39, 127: 40, 39: 52, 115: 74, 3: 131, 7: 149, 12: 1, 48: 1, 40: 2, 104: 2, 124: 2, 52: 3, 64: 6, 120: 6, 28: 7, 56: 8, 44: 10, 112: 15, 84: 19, 100: 24, 20: 26, 68: 33, 92: 33, 72: 34, 16: 47, 108: 56, 88: 59, 8: 110, 60: 115, 80: 129, 4: 135, 76: 151, 0: 195, 17: 1, 77: 1, 53: 2, 61: 3, 9: 4, 109: 4, 29: 5, 97: 5, 25: 6, 33: 6, 89: 9, 73: 10, 37: 11, 93: 14, 117: 14, 69: 18, 121: 21, 13: 23, 49: 23, 105: 24, 113: 31, 85: 39, 21: 41, 57: 51, 125: 53, 41: 54, 81: 109, 5: 143, 1: 269, 38: 1, 66: 1, 102: 1, 70: 2, 110: 2, 26: 5, 98: 7, 78: 9, 50: 15, 34: 16, 90: 18, 94: 26, 54: 31, 46: 35, 18: 40, 10: 42, 58: 42, 82: 45, 106: 50, 126: 51, 42: 59, 62: 66, 14: 92, 74: 93, 86: 98, 2: 128, 6: 134}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.491033.491033 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.567ms | allocate_experts_across_cpu_gpu: 0.396ms
INFO 05-06 10:38:51.491347.491347 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:51.492952.492952 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005636215209960938 seconds
INFO 05-06 10:38:51.493089.493089 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008339881896972656 seconds
INFO 05-06 10:38:51.493006.493006 lmp.py:1484] [layer_moe_fused] experts compute time: 1.9073486328125e-06 seconds
INFO 05-06 10:38:51.528843.528843 lmp.py:1496] [layer_moe_fused] to time: 0.00017571449279785156 seconds
INFO 05-06 10:38:51.529237.529237 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03551745414733887 seconds
DEBUG 05-06 10:38:51.529229.529229 cuda_h.py:27] end *layer_moe_fused cost 39.450 ms
DEBUG 05-06 10:38:51.530409.530409 cuda_h.py:27] end prefill_layer cost 43.711 ms
DEBUG 05-06 10:38:51.530040.530040 lmp.py:841] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 10:38:51.530956.530956 lmp.py:824] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 10:38:51.534072.534072 cuda_h.py:27] end *sagl cost 3.365 ms
experts_cpu_alloc {'expert_ids': [15, 115, 35, 47, 63, 127, 11, 51, 39, 123, 27, 59, 43, 71, 91, 99, 119, 31, 19, 23, 67, 111, 83, 79, 87, 3, 7, 12, 84, 72, 64, 96, 88, 48, 44, 8, 40, 80, 52, 120, 36, 116, 124, 28, 112, 20, 76, 100, 24, 108, 32, 68, 92, 0, 4, 56, 16, 105, 85, 9, 21, 33, 53, 13, 97, 125, 121, 117, 25, 61, 29, 89, 37, 69, 57, 77, 49, 17, 81, 93, 113, 1, 5, 94, 34, 106, 110, 114, 118, 74, 122, 58, 62, 22, 42, 50, 46, 54, 18, 126, 82, 98, 66, 70, 38, 30, 10, 102, 2, 6], 'token_total': 4096, 'token_per_expert': {15: 1, 115: 1, 35: 2, 47: 2, 63: 2, 127: 2, 11: 7, 51: 7, 39: 8, 123: 11, 27: 16, 59: 16, 43: 17, 71: 17, 91: 23, 99: 32, 119: 33, 31: 40, 19: 46, 23: 56, 67: 57, 111: 74, 83: 96, 79: 112, 87: 126, 3: 133, 7: 191, 12: 1, 84: 1, 72: 2, 64: 3, 96: 3, 88: 4, 48: 5, 44: 7, 8: 9, 40: 9, 80: 9, 52: 12, 120: 13, 36: 16, 116: 19, 124: 19, 28: 20, 112: 37, 20: 39, 76: 41, 100: 45, 24: 46, 108: 47, 32: 54, 68: 71, 92: 93, 0: 131, 4: 135, 56: 137, 16: 142, 105: 1, 85: 2, 9: 3, 21: 3, 33: 3, 53: 3, 13: 4, 97: 4, 125: 4, 121: 12, 117: 19, 25: 22, 61: 24, 29: 25, 89: 26, 37: 28, 69: 28, 57: 33, 77: 38, 49: 75, 17: 90, 81: 97, 93: 102, 113: 112, 1: 139, 5: 139, 94: 1, 34: 2, 106: 2, 110: 2, 114: 2, 118: 2, 74: 3, 122: 3, 58: 4, 62: 4, 22: 5, 42: 6, 50: 7, 46: 9, 54: 9, 18: 12, 126: 13, 82: 14, 98: 16, 66: 21, 70: 27, 38: 28, 30: 39, 10: 42, 102: 123, 2: 172, 6: 194}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.536801.536801 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.574ms | allocate_experts_across_cpu_gpu: 0.396ms
INFO 05-06 10:38:51.536691.536691 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-06 seconds
INFO 05-06 10:38:51.537327.537327 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005404949188232422 seconds
INFO 05-06 10:38:51.538100.538100 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008754730224609375 seconds
INFO 05-06 10:38:51.538236.538236 lmp.py:1484] [layer_moe_fused] experts compute time: 3.0994415283203125e-06 seconds
INFO 05-06 10:38:51.573134.573134 lmp.py:1496] [layer_moe_fused] to time: 0.0001735687255859375 seconds
INFO 05-06 10:38:51.573257.573257 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03494000434875488 seconds
DEBUG 05-06 10:38:51.574541.574541 cuda_h.py:27] end *layer_moe_fused cost 39.088 ms
DEBUG 05-06 10:38:51.574866.574866 cuda_h.py:27] end prefill_layer cost 44.587 ms
DEBUG 05-06 10:38:51.575266.575266 lmp.py:841] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 10:38:51.575957.575957 lmp.py:824] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 10:38:51.577743.577743 cuda_h.py:27] end *sagl cost 2.248 ms
experts_cpu_alloc {'expert_ids': [59, 67, 47, 127, 31, 119, 63, 79, 107, 111, 123, 103, 95, 35, 91, 23, 115, 19, 15, 7, 39, 3, 71, 8, 48, 96, 32, 20, 24, 112, 120, 12, 88, 68, 100, 40, 80, 104, 76, 124, 36, 92, 84, 116, 108, 0, 4, 37, 81, 41, 113, 13, 17, 33, 105, 77, 65, 89, 125, 85, 101, 73, 117, 25, 97, 49, 45, 1, 53, 21, 5, 54, 102, 18, 38, 70, 94, 90, 22, 98, 58, 34, 106, 46, 118, 82, 114, 86, 74, 110, 50, 2, 6, 78], 'token_total': 4096, 'token_per_expert': {59: 1, 67: 1, 47: 2, 127: 2, 31: 3, 119: 3, 63: 4, 79: 7, 107: 7, 111: 7, 123: 7, 103: 11, 95: 24, 35: 31, 91: 33, 23: 35, 115: 35, 19: 49, 15: 83, 7: 128, 39: 141, 3: 151, 71: 155, 8: 1, 48: 1, 96: 1, 32: 2, 20: 5, 24: 5, 112: 6, 120: 6, 12: 9, 88: 11, 68: 17, 100: 18, 40: 19, 80: 19, 104: 20, 76: 22, 124: 22, 36: 26, 92: 28, 84: 36, 116: 93, 108: 116, 0: 128, 4: 129, 37: 1, 81: 1, 41: 2, 113: 2, 13: 3, 17: 3, 33: 3, 105: 4, 77: 7, 65: 10, 89: 10, 125: 10, 85: 22, 101: 31, 73: 41, 117: 42, 25: 43, 97: 50, 49: 54, 45: 97, 1: 137, 53: 148, 21: 153, 5: 183, 54: 1, 102: 4, 18: 5, 38: 8, 70: 8, 94: 9, 90: 10, 22: 12, 98: 13, 58: 22, 34: 30, 106: 57, 46: 61, 118: 65, 82: 85, 114: 89, 86: 93, 74: 98, 110: 98, 50: 112, 2: 128, 6: 172, 78: 199}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 23, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 23, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.579112.579112 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.569ms | allocate_experts_across_cpu_gpu: 0.361ms
INFO 05-06 10:38:51.579116.579116 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:51.580672.580672 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005435943603515625 seconds
INFO 05-06 10:38:51.581709.581709 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0010840892791748047 seconds
INFO 05-06 10:38:51.581230.581230 lmp.py:1484] [layer_moe_fused] experts compute time: 5.4836273193359375e-06 seconds
INFO 05-06 10:38:51.616979.616979 lmp.py:1496] [layer_moe_fused] to time: 0.00017690658569335938 seconds
INFO 05-06 10:38:51.616109.616109 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03448057174682617 seconds
DEBUG 05-06 10:38:51.617504.617504 cuda_h.py:27] end *layer_moe_fused cost 38.757 ms
DEBUG 05-06 10:38:51.617432.617432 cuda_h.py:27] end prefill_layer cost 42.863 ms
DEBUG 05-06 10:38:51.618580.618580 lmp.py:841] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 10:38:51.618111.618111 lmp.py:824] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 10:38:51.620130.620130 cuda_h.py:27] end *sagl cost 2.263 ms
experts_cpu_alloc {'expert_ids': [83, 111, 123, 47, 27, 95, 107, 11, 43, 87, 67, 55, 119, 115, 15, 99, 75, 51, 39, 103, 63, 59, 71, 79, 7, 91, 3, 31, 12, 48, 56, 72, 112, 8, 16, 96, 64, 104, 28, 40, 92, 52, 80, 68, 108, 60, 84, 124, 116, 20, 32, 120, 4, 100, 0, 45, 97, 105, 109, 61, 65, 57, 9, 73, 101, 93, 41, 117, 13, 113, 69, 21, 125, 33, 25, 81, 17, 121, 37, 5, 1, 74, 10, 66, 94, 106, 62, 90, 70, 26, 46, 122, 86, 82, 42, 34, 22, 38, 98, 118, 126, 102, 14, 78, 110, 2, 114, 6], 'token_total': 4096, 'token_per_expert': {83: 2, 111: 2, 123: 3, 47: 5, 27: 6, 95: 7, 107: 7, 11: 8, 43: 8, 87: 9, 67: 12, 55: 17, 119: 18, 115: 19, 15: 20, 99: 20, 75: 23, 51: 27, 39: 30, 103: 33, 63: 48, 59: 65, 71: 73, 79: 100, 7: 128, 91: 158, 3: 161, 31: 192, 12: 1, 48: 1, 56: 1, 72: 1, 112: 2, 8: 4, 16: 5, 96: 5, 64: 7, 104: 7, 28: 8, 40: 8, 92: 8, 52: 9, 80: 9, 68: 10, 108: 14, 60: 17, 84: 28, 124: 28, 116: 31, 20: 43, 32: 68, 120: 96, 4: 128, 100: 129, 0: 130, 45: 1, 97: 1, 105: 1, 109: 2, 61: 3, 65: 3, 57: 5, 9: 6, 73: 7, 101: 8, 93: 10, 41: 15, 117: 16, 13: 17, 113: 29, 69: 32, 21: 36, 125: 38, 33: 43, 25: 54, 81: 65, 17: 80, 121: 102, 37: 107, 5: 128, 1: 161, 74: 1, 10: 2, 66: 2, 94: 2, 106: 2, 62: 3, 90: 4, 70: 5, 26: 6, 46: 10, 122: 14, 86: 16, 82: 17, 42: 19, 34: 28, 22: 29, 38: 36, 98: 40, 118: 40, 126: 50, 102: 56, 14: 69, 78: 74, 110: 114, 2: 135, 114: 159, 6: 194}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.622064.622064 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.565ms | allocate_experts_across_cpu_gpu: 0.394ms
INFO 05-06 10:38:51.622676.622676 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:51.623865.623865 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.000545501708984375 seconds
INFO 05-06 10:38:51.625665.625665 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001062154769897461 seconds
INFO 05-06 10:38:51.625635.625635 lmp.py:1484] [layer_moe_fused] experts compute time: 6.4373016357421875e-06 seconds
INFO 05-06 10:38:51.659251.659251 lmp.py:1496] [layer_moe_fused] to time: 0.00018262863159179688 seconds
INFO 05-06 10:38:51.660506.660506 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03433871269226074 seconds
DEBUG 05-06 10:38:51.660994.660994 cuda_h.py:27] end *layer_moe_fused cost 38.830 ms
DEBUG 05-06 10:38:51.661174.661174 cuda_h.py:27] end prefill_layer cost 43.188 ms
DEBUG 05-06 10:38:51.661282.661282 lmp.py:841] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 10:38:51.661529.661529 lmp.py:824] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 10:38:51.663925.663925 cuda_h.py:27] end *sagl cost 2.237 ms
experts_cpu_alloc {'expert_ids': [27, 55, 87, 91, 111, 63, 19, 51, 67, 43, 15, 23, 35, 71, 83, 107, 11, 127, 59, 103, 99, 31, 123, 95, 47, 75, 39, 119, 3, 7, 115, 88, 36, 96, 40, 48, 64, 68, 44, 116, 28, 92, 60, 52, 108, 120, 112, 32, 16, 8, 72, 12, 76, 104, 80, 100, 24, 124, 4, 0, 29, 33, 49, 37, 85, 41, 77, 101, 73, 9, 21, 81, 109, 93, 45, 125, 57, 105, 25, 13, 53, 89, 113, 117, 65, 97, 1, 5, 121, 22, 46, 14, 18, 54, 106, 98, 70, 58, 78, 118, 126, 102, 110, 34, 90, 10, 74, 114, 122, 38, 30, 62, 42, 66, 50, 86, 26, 6, 2], 'token_total': 4096, 'token_per_expert': {27: 1, 55: 1, 87: 1, 91: 2, 111: 2, 63: 3, 19: 4, 51: 4, 67: 4, 43: 5, 15: 7, 23: 7, 35: 9, 71: 12, 83: 16, 107: 20, 11: 21, 127: 38, 59: 40, 103: 48, 99: 49, 31: 50, 123: 56, 95: 61, 47: 63, 75: 72, 39: 84, 119: 102, 3: 134, 7: 137, 115: 195, 88: 2, 36: 5, 96: 5, 40: 6, 48: 6, 64: 7, 68: 7, 44: 8, 116: 9, 28: 11, 92: 12, 60: 13, 52: 15, 108: 15, 120: 16, 112: 18, 32: 19, 16: 20, 8: 21, 72: 23, 12: 25, 76: 27, 104: 30, 80: 43, 100: 47, 24: 51, 124: 109, 4: 129, 0: 135, 29: 1, 33: 1, 49: 1, 37: 2, 85: 2, 41: 3, 77: 4, 101: 4, 73: 5, 9: 6, 21: 8, 81: 9, 109: 9, 93: 12, 45: 15, 125: 18, 57: 20, 105: 20, 25: 21, 13: 26, 53: 31, 89: 40, 113: 51, 117: 52, 65: 98, 97: 108, 1: 129, 5: 136, 121: 137, 22: 1, 46: 1, 14: 2, 18: 3, 54: 3, 106: 4, 98: 5, 70: 6, 58: 7, 78: 8, 118: 8, 126: 8, 102: 10, 110: 10, 34: 14, 90: 17, 10: 18, 74: 21, 114: 23, 122: 30, 38: 32, 30: 34, 62: 37, 42: 38, 66: 76, 50: 98, 86: 104, 26: 124, 6: 129, 2: 174}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.666754.666754 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.572ms | allocate_experts_across_cpu_gpu: 0.420ms
INFO 05-06 10:38:51.666803.666803 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:51.667265.667265 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005869865417480469 seconds
INFO 05-06 10:38:51.668577.668577 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.000965118408203125 seconds
INFO 05-06 10:38:51.668304.668304 lmp.py:1484] [layer_moe_fused] experts compute time: 1.6689300537109375e-06 seconds
INFO 05-06 10:38:51.702541.702541 lmp.py:1496] [layer_moe_fused] to time: 0.00017261505126953125 seconds
INFO 05-06 10:38:51.703657.703657 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.034624576568603516 seconds
DEBUG 05-06 10:38:51.703768.703768 cuda_h.py:27] end *layer_moe_fused cost 38.610 ms
DEBUG 05-06 10:38:51.704094.704094 cuda_h.py:27] end prefill_layer cost 42.963 ms
DEBUG 05-06 10:38:51.704825.704825 lmp.py:841] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 10:38:51.704879.704879 lmp.py:824] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 10:38:51.706996.706996 cuda_h.py:27] end *sagl cost 2.199 ms
experts_cpu_alloc {'expert_ids': [79, 35, 67, 11, 87, 111, 19, 43, 127, 107, 115, 119, 31, 55, 59, 47, 63, 95, 103, 99, 51, 23, 39, 71, 75, 83, 91, 3, 7, 12, 32, 56, 100, 8, 40, 96, 80, 48, 28, 116, 36, 16, 120, 124, 24, 64, 84, 72, 104, 88, 52, 108, 4, 0, 112, 68, 76, 45, 57, 105, 25, 117, 29, 13, 77, 17, 41, 69, 121, 33, 113, 73, 93, 21, 97, 85, 37, 81, 125, 101, 9, 109, 65, 1, 5, 62, 110, 94, 22, 54, 126, 118, 34, 82, 18, 38, 58, 86, 14, 78, 46, 114, 102, 42, 98, 70, 30, 10, 66, 90, 6, 2], 'token_total': 4096, 'token_per_expert': {79: 1, 35: 2, 67: 2, 11: 4, 87: 4, 111: 4, 19: 9, 43: 12, 127: 12, 107: 13, 115: 13, 119: 14, 31: 15, 55: 15, 59: 17, 47: 22, 63: 25, 95: 25, 103: 32, 99: 36, 51: 38, 23: 49, 39: 51, 71: 53, 75: 62, 83: 86, 91: 123, 3: 129, 7: 156, 12: 1, 32: 1, 56: 1, 100: 5, 8: 8, 40: 8, 96: 8, 80: 9, 48: 11, 28: 13, 116: 16, 36: 25, 16: 27, 120: 29, 124: 32, 24: 33, 64: 36, 84: 36, 72: 38, 104: 39, 88: 42, 52: 63, 108: 79, 4: 133, 0: 148, 112: 150, 68: 153, 76: 154, 45: 2, 57: 2, 105: 3, 25: 4, 117: 5, 29: 8, 13: 9, 77: 9, 17: 11, 41: 11, 69: 12, 121: 15, 33: 17, 113: 22, 73: 24, 93: 24, 21: 25, 97: 26, 85: 28, 37: 31, 81: 34, 125: 43, 101: 50, 9: 60, 109: 103, 65: 111, 1: 144, 5: 153, 62: 1, 110: 1, 94: 2, 22: 3, 54: 3, 126: 5, 118: 6, 34: 7, 82: 7, 18: 9, 38: 9, 58: 9, 86: 9, 14: 15, 78: 16, 46: 17, 114: 18, 102: 20, 42: 23, 98: 33, 70: 36, 30: 55, 10: 56, 66: 57, 90: 75, 6: 128, 2: 168}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.709071.709071 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.578ms | allocate_experts_across_cpu_gpu: 0.403ms
INFO 05-06 10:38:51.709545.709545 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:51.710700.710700 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005571842193603516 seconds
INFO 05-06 10:38:51.711108.711108 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008070468902587891 seconds
INFO 05-06 10:38:51.711182.711182 lmp.py:1484] [layer_moe_fused] experts compute time: 1.1920928955078125e-06 seconds
INFO 05-06 10:38:51.745255.745255 lmp.py:1496] [layer_moe_fused] to time: 0.00017118453979492188 seconds
INFO 05-06 10:38:51.745039.745039 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03425955772399902 seconds
DEBUG 05-06 10:38:51.746634.746634 cuda_h.py:27] end *layer_moe_fused cost 38.161 ms
DEBUG 05-06 10:38:51.747330.747330 cuda_h.py:27] end prefill_layer cost 42.528 ms
DEBUG 05-06 10:38:51.747339.747339 lmp.py:841] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 10:38:51.747870.747870 lmp.py:824] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 10:38:51.749357.749357 cuda_h.py:27] end *sagl cost 2.250 ms
experts_cpu_alloc {'expert_ids': [115, 11, 59, 71, 111, 43, 91, 99, 123, 51, 103, 79, 15, 119, 127, 19, 75, 63, 55, 31, 107, 83, 23, 67, 7, 87, 3, 28, 36, 64, 120, 88, 84, 112, 56, 60, 104, 24, 40, 92, 20, 116, 80, 72, 108, 76, 68, 48, 100, 96, 8, 44, 124, 12, 52, 16, 4, 0, 32, 101, 29, 49, 73, 53, 89, 13, 9, 37, 109, 45, 33, 93, 113, 21, 61, 81, 17, 57, 97, 77, 117, 65, 121, 85, 125, 105, 5, 1, 74, 94, 106, 50, 98, 122, 34, 38, 10, 118, 18, 62, 82, 22, 30, 78, 90, 70, 102, 42, 110, 14, 26, 114, 58, 54, 66, 86, 6, 2, 126], 'token_total': 4096, 'token_per_expert': {115: 1, 11: 2, 59: 3, 71: 4, 111: 4, 43: 6, 91: 7, 99: 7, 123: 9, 51: 11, 103: 11, 79: 14, 15: 15, 119: 19, 127: 23, 19: 26, 75: 30, 63: 33, 55: 36, 31: 44, 107: 44, 83: 51, 23: 65, 67: 96, 7: 131, 87: 132, 3: 147, 28: 1, 36: 2, 64: 2, 120: 2, 88: 4, 84: 5, 112: 5, 56: 7, 60: 8, 104: 9, 24: 10, 40: 11, 92: 15, 20: 19, 116: 20, 80: 24, 72: 25, 108: 27, 76: 30, 68: 36, 48: 38, 100: 39, 96: 40, 8: 42, 44: 42, 124: 43, 12: 47, 52: 110, 16: 129, 4: 162, 0: 165, 32: 166, 101: 1, 29: 2, 49: 2, 73: 2, 53: 5, 89: 5, 13: 6, 9: 7, 37: 7, 109: 7, 45: 8, 33: 11, 93: 12, 113: 12, 21: 15, 61: 15, 81: 15, 17: 16, 57: 18, 97: 18, 77: 19, 117: 22, 65: 25, 121: 25, 85: 30, 125: 33, 105: 87, 5: 157, 1: 201, 74: 1, 94: 1, 106: 1, 50: 3, 98: 3, 122: 3, 34: 4, 38: 5, 10: 6, 118: 6, 18: 10, 62: 11, 82: 11, 22: 16, 30: 18, 78: 18, 90: 19, 70: 21, 102: 21, 42: 28, 110: 28, 14: 32, 26: 33, 114: 33, 58: 37, 54: 43, 66: 54, 86: 95, 6: 134, 2: 155, 126: 207}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 32, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.751722.751722 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.573ms | allocate_experts_across_cpu_gpu: 0.429ms
INFO 05-06 10:38:51.751288.751288 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:51.752775.752775 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005488395690917969 seconds
INFO 05-06 10:38:51.753918.753918 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008273124694824219 seconds
INFO 05-06 10:38:51.754390.754390 lmp.py:1484] [layer_moe_fused] experts compute time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:51.789739.789739 lmp.py:1496] [layer_moe_fused] to time: 0.00029778480529785156 seconds
INFO 05-06 10:38:51.789056.789056 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03527402877807617 seconds
DEBUG 05-06 10:38:51.789881.789881 cuda_h.py:27] end *layer_moe_fused cost 39.181 ms
DEBUG 05-06 10:38:51.790345.790345 cuda_h.py:27] end prefill_layer cost 43.631 ms
DEBUG 05-06 10:38:51.790421.790421 lmp.py:841] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 10:38:51.790098.790098 lmp.py:824] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 10:38:51.794515.794515 cuda_h.py:27] end *sagl cost 3.507 ms
experts_cpu_alloc {'expert_ids': [51, 79, 15, 83, 91, 123, 11, 111, 87, 119, 19, 31, 99, 59, 67, 55, 103, 35, 63, 47, 71, 43, 107, 27, 39, 95, 75, 23, 7, 3, 88, 112, 8, 16, 92, 36, 96, 32, 44, 60, 104, 116, 80, 100, 108, 124, 84, 48, 120, 20, 56, 68, 12, 64, 28, 40, 52, 72, 24, 76, 0, 4, 77, 65, 29, 81, 93, 85, 97, 9, 117, 109, 33, 113, 73, 125, 13, 45, 57, 101, 49, 17, 53, 89, 61, 21, 69, 1, 37, 5, 30, 110, 62, 82, 42, 66, 102, 118, 14, 38, 90, 114, 34, 126, 98, 70, 122, 78, 94, 54, 22, 106, 10, 18, 58, 86, 74, 2, 6], 'token_total': 4096, 'token_per_expert': {51: 1, 79: 1, 15: 3, 83: 4, 91: 6, 123: 6, 11: 9, 111: 9, 87: 10, 119: 11, 19: 12, 31: 13, 99: 13, 59: 14, 67: 20, 55: 21, 103: 23, 35: 31, 63: 31, 47: 45, 71: 46, 43: 56, 107: 61, 27: 63, 39: 63, 95: 67, 75: 82, 23: 111, 7: 129, 3: 143, 88: 3, 112: 3, 8: 5, 16: 5, 92: 5, 36: 6, 96: 6, 32: 9, 44: 12, 60: 12, 104: 12, 116: 12, 80: 13, 100: 14, 108: 17, 124: 17, 84: 19, 48: 23, 120: 23, 20: 28, 56: 30, 68: 30, 12: 31, 64: 34, 28: 39, 40: 51, 52: 51, 72: 64, 24: 105, 76: 106, 0: 147, 4: 154, 77: 1, 65: 2, 29: 3, 81: 3, 93: 3, 85: 4, 97: 4, 9: 5, 117: 5, 109: 8, 33: 12, 113: 15, 73: 20, 125: 21, 13: 23, 45: 23, 57: 28, 101: 37, 49: 38, 17: 45, 53: 49, 89: 55, 61: 63, 21: 71, 69: 109, 1: 132, 37: 135, 5: 147, 30: 1, 110: 1, 62: 3, 82: 4, 42: 5, 66: 5, 102: 5, 118: 5, 14: 6, 38: 6, 90: 7, 114: 8, 34: 9, 126: 9, 98: 18, 70: 19, 122: 21, 78: 22, 94: 23, 54: 26, 22: 27, 106: 32, 10: 36, 18: 37, 58: 58, 86: 79, 74: 88, 2: 134, 6: 151}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 32, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.797948.797948 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.712ms | allocate_experts_across_cpu_gpu: 0.423ms
INFO 05-06 10:38:51.797885.797885 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:51.798463.798463 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006048679351806641 seconds
INFO 05-06 10:38:51.799562.799562 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0010786056518554688 seconds
INFO 05-06 10:38:51.799975.799975 lmp.py:1484] [layer_moe_fused] experts compute time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:51.833999.833999 lmp.py:1496] [layer_moe_fused] to time: 0.0001761913299560547 seconds
INFO 05-06 10:38:51.833897.833897 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03365802764892578 seconds
DEBUG 05-06 10:38:51.834597.834597 cuda_h.py:27] end *layer_moe_fused cost 38.306 ms
DEBUG 05-06 10:38:51.834353.834353 cuda_h.py:27] end prefill_layer cost 44.000 ms
DEBUG 05-06 10:38:51.834362.834362 lmp.py:841] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 10:38:51.835370.835370 lmp.py:824] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 10:38:51.837550.837550 cuda_h.py:27] end *sagl cost 2.280 ms
experts_cpu_alloc {'expert_ids': [27, 55, 115, 59, 63, 19, 23, 11, 39, 51, 67, 107, 35, 103, 95, 47, 91, 15, 71, 75, 123, 127, 31, 119, 87, 83, 43, 111, 99, 7, 3, 20, 96, 24, 44, 112, 52, 16, 124, 48, 56, 108, 80, 68, 116, 12, 92, 120, 104, 72, 100, 60, 84, 88, 76, 40, 8, 64, 32, 36, 0, 4, 25, 113, 117, 41, 109, 45, 89, 21, 37, 9, 97, 125, 73, 13, 29, 57, 93, 81, 17, 69, 49, 65, 53, 61, 33, 101, 77, 85, 121, 5, 1, 22, 18, 86, 26, 62, 74, 98, 42, 94, 102, 82, 66, 70, 114, 122, 90, 46, 10, 34, 30, 38, 78, 118, 110, 58, 14, 50, 54, 6, 2], 'token_total': 4096, 'token_per_expert': {27: 4, 55: 4, 115: 4, 59: 5, 63: 5, 19: 6, 23: 6, 11: 7, 39: 7, 51: 7, 67: 10, 107: 10, 35: 13, 103: 13, 95: 16, 47: 19, 91: 20, 15: 22, 71: 25, 75: 27, 123: 27, 127: 30, 31: 47, 119: 52, 87: 54, 83: 55, 43: 66, 111: 71, 99: 85, 7: 141, 3: 191, 20: 1, 96: 2, 24: 4, 44: 4, 112: 4, 52: 5, 16: 8, 124: 9, 48: 14, 56: 14, 108: 14, 80: 15, 68: 17, 116: 17, 12: 19, 92: 24, 120: 27, 104: 32, 72: 33, 100: 35, 60: 38, 84: 38, 88: 38, 76: 44, 40: 52, 8: 54, 64: 55, 32: 65, 36: 69, 0: 133, 4: 179, 25: 2, 113: 3, 117: 3, 41: 5, 109: 6, 45: 8, 89: 9, 21: 10, 37: 10, 9: 11, 97: 11, 125: 12, 73: 14, 13: 17, 29: 17, 57: 21, 93: 28, 81: 30, 17: 32, 69: 33, 49: 37, 65: 37, 53: 40, 61: 44, 33: 53, 101: 57, 77: 63, 85: 80, 121: 101, 5: 139, 1: 166, 22: 2, 18: 3, 86: 3, 26: 5, 62: 5, 74: 5, 98: 5, 42: 6, 94: 6, 102: 6, 82: 7, 66: 8, 70: 8, 114: 8, 122: 11, 90: 12, 46: 15, 10: 20, 34: 20, 30: 21, 38: 27, 78: 34, 118: 44, 110: 45, 58: 47, 14: 50, 50: 56, 54: 74, 6: 129, 2: 203}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.839995.839995 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.572ms | allocate_experts_across_cpu_gpu: 0.447ms
INFO 05-06 10:38:51.839694.839694 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:51.840979.840979 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005555152893066406 seconds
INFO 05-06 10:38:51.841674.841674 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008902549743652344 seconds
INFO 05-06 10:38:51.842241.842241 lmp.py:1484] [layer_moe_fused] experts compute time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:51.876720.876720 lmp.py:1496] [layer_moe_fused] to time: 0.00017571449279785156 seconds
INFO 05-06 10:38:51.876405.876405 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03452014923095703 seconds
DEBUG 05-06 10:38:51.877663.877663 cuda_h.py:27] end *layer_moe_fused cost 38.556 ms
DEBUG 05-06 10:38:51.877743.877743 cuda_h.py:27] end prefill_layer cost 42.787 ms
DEBUG 05-06 10:38:51.877329.877329 lmp.py:841] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 10:38:51.877827.877827 lmp.py:824] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 10:38:51.880752.880752 cuda_h.py:27] end *sagl cost 2.245 ms
experts_cpu_alloc {'expert_ids': [87, 107, 115, 67, 95, 43, 103, 127, 59, 55, 111, 83, 19, 11, 47, 99, 15, 31, 27, 119, 35, 39, 75, 63, 79, 23, 123, 51, 7, 3, 8, 124, 116, 100, 120, 112, 20, 68, 84, 108, 56, 104, 60, 36, 72, 12, 48, 16, 88, 96, 80, 76, 40, 24, 44, 92, 64, 4, 0, 52, 81, 101, 105, 113, 57, 29, 49, 65, 77, 121, 45, 53, 97, 17, 25, 73, 13, 33, 125, 21, 41, 9, 69, 109, 61, 117, 37, 89, 5, 1, 54, 62, 74, 82, 30, 34, 42, 46, 66, 70, 94, 18, 114, 110, 90, 86, 58, 98, 118, 126, 22, 26, 106, 10, 102, 50, 38, 6, 122, 2], 'token_total': 4096, 'token_per_expert': {87: 2, 107: 2, 115: 2, 67: 3, 95: 3, 43: 5, 103: 7, 127: 8, 59: 9, 55: 13, 111: 13, 83: 15, 19: 17, 11: 19, 47: 20, 99: 20, 15: 22, 31: 23, 27: 24, 119: 24, 35: 29, 39: 29, 75: 34, 63: 38, 79: 44, 23: 48, 123: 51, 51: 87, 7: 203, 3: 204, 8: 1, 124: 1, 116: 2, 100: 3, 120: 3, 112: 8, 20: 9, 68: 11, 84: 12, 108: 13, 56: 15, 104: 16, 60: 17, 36: 19, 72: 19, 12: 20, 48: 20, 16: 34, 88: 38, 96: 38, 80: 39, 76: 42, 40: 45, 24: 53, 44: 89, 92: 93, 64: 98, 4: 130, 0: 143, 52: 172, 81: 1, 101: 2, 105: 2, 113: 2, 57: 3, 29: 5, 49: 5, 65: 6, 77: 6, 121: 6, 45: 9, 53: 9, 97: 12, 17: 14, 25: 14, 73: 14, 13: 16, 33: 19, 125: 21, 21: 24, 41: 25, 9: 29, 69: 37, 109: 38, 61: 55, 117: 66, 37: 104, 89: 125, 5: 156, 1: 162, 54: 1, 62: 1, 74: 2, 82: 2, 30: 3, 34: 3, 42: 5, 46: 5, 66: 5, 70: 5, 94: 5, 18: 6, 114: 6, 110: 9, 90: 11, 86: 13, 58: 17, 98: 17, 118: 17, 126: 17, 22: 18, 26: 30, 106: 32, 10: 38, 102: 41, 50: 51, 38: 117, 6: 133, 122: 137, 2: 141}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.882654.882654 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.574ms | allocate_experts_across_cpu_gpu: 0.416ms
INFO 05-06 10:38:51.882975.882975 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:51.883313.883313 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005545616149902344 seconds
INFO 05-06 10:38:51.884286.884286 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0006413459777832031 seconds
INFO 05-06 10:38:51.884464.884464 lmp.py:1484] [layer_moe_fused] experts compute time: 1.430511474609375e-06 seconds
INFO 05-06 10:38:51.921285.921285 lmp.py:1496] [layer_moe_fused] to time: 0.00016927719116210938 seconds
INFO 05-06 10:38:51.921547.921547 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03685140609741211 seconds
DEBUG 05-06 10:38:51.921547.921547 cuda_h.py:27] end *layer_moe_fused cost 40.520 ms
DEBUG 05-06 10:38:51.922150.922150 cuda_h.py:27] end prefill_layer cost 44.647 ms
DEBUG 05-06 10:38:51.922643.922643 lmp.py:841] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 10:38:51.922843.922843 lmp.py:824] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 10:38:51.925110.925110 cuda_h.py:27] end *sagl cost 2.221 ms
experts_cpu_alloc {'expert_ids': [23, 51, 87, 67, 75, 91, 99, 119, 31, 11, 35, 111, 19, 47, 95, 83, 103, 55, 71, 79, 15, 43, 27, 123, 59, 63, 107, 7, 3, 16, 124, 80, 24, 104, 36, 60, 84, 120, 20, 12, 76, 100, 72, 88, 52, 64, 112, 116, 32, 92, 108, 28, 44, 8, 40, 56, 0, 4, 68, 17, 25, 69, 97, 117, 89, 61, 93, 105, 101, 121, 113, 41, 9, 85, 125, 81, 33, 109, 53, 57, 77, 65, 21, 13, 37, 73, 45, 49, 1, 5, 22, 34, 70, 126, 62, 74, 90, 110, 18, 86, 106, 98, 118, 38, 10, 58, 122, 46, 26, 114, 50, 54, 82, 42, 66, 102, 30, 94, 6, 2], 'token_total': 4096, 'token_per_expert': {23: 1, 51: 1, 87: 1, 67: 2, 75: 2, 91: 2, 99: 2, 119: 2, 31: 4, 11: 6, 35: 6, 111: 7, 19: 9, 47: 10, 95: 11, 83: 12, 103: 13, 55: 15, 71: 16, 79: 24, 15: 26, 43: 36, 27: 37, 123: 42, 59: 58, 63: 82, 107: 124, 7: 130, 3: 181, 16: 1, 124: 2, 80: 4, 24: 6, 104: 6, 36: 8, 60: 10, 84: 10, 120: 11, 20: 14, 12: 15, 76: 15, 100: 17, 72: 20, 88: 27, 52: 28, 64: 30, 112: 30, 116: 33, 32: 36, 92: 40, 108: 41, 28: 42, 44: 47, 8: 51, 40: 55, 56: 55, 0: 139, 4: 174, 68: 192, 17: 3, 25: 3, 69: 3, 97: 3, 117: 3, 89: 4, 61: 6, 93: 7, 105: 8, 101: 11, 121: 12, 113: 16, 41: 18, 9: 22, 85: 25, 125: 28, 81: 29, 33: 31, 109: 31, 53: 32, 57: 46, 77: 46, 65: 52, 21: 57, 13: 59, 37: 61, 73: 70, 45: 106, 49: 113, 1: 140, 5: 182, 22: 1, 34: 4, 70: 4, 126: 4, 62: 7, 74: 7, 90: 7, 110: 7, 18: 8, 86: 8, 106: 8, 98: 9, 118: 10, 38: 11, 10: 12, 58: 12, 122: 14, 46: 17, 26: 18, 114: 19, 50: 21, 54: 21, 82: 25, 42: 35, 66: 36, 102: 67, 30: 72, 94: 106, 6: 130, 2: 148}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.927191.927191 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.582ms | allocate_experts_across_cpu_gpu: 0.422ms
INFO 05-06 10:38:51.927472.927472 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:51.928525.928525 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005424022674560547 seconds
INFO 05-06 10:38:51.929768.929768 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008902549743652344 seconds
INFO 05-06 10:38:51.929414.929414 lmp.py:1484] [layer_moe_fused] experts compute time: 3.337860107421875e-06 seconds
INFO 05-06 10:38:51.964242.964242 lmp.py:1496] [layer_moe_fused] to time: 0.00017380714416503906 seconds
INFO 05-06 10:38:51.964265.964265 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.034674882888793945 seconds
DEBUG 05-06 10:38:51.965795.965795 cuda_h.py:27] end *layer_moe_fused cost 38.634 ms
DEBUG 05-06 10:38:51.965683.965683 cuda_h.py:27] end prefill_layer cost 42.885 ms
DEBUG 05-06 10:38:51.965560.965560 lmp.py:841] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 10:38:51.965714.965714 lmp.py:824] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 10:38:51.968755.968755 cuda_h.py:27] end *sagl cost 2.247 ms
experts_cpu_alloc {'expert_ids': [19, 99, 39, 107, 23, 27, 43, 59, 55, 87, 115, 119, 71, 127, 123, 79, 75, 31, 95, 67, 111, 35, 83, 103, 51, 11, 3, 7, 56, 60, 52, 104, 16, 96, 32, 88, 40, 64, 20, 44, 24, 80, 12, 36, 72, 84, 68, 76, 8, 124, 112, 92, 120, 48, 100, 0, 4, 25, 17, 89, 113, 77, 9, 101, 69, 21, 93, 121, 45, 33, 125, 81, 57, 13, 97, 61, 29, 37, 109, 53, 73, 41, 105, 65, 1, 5, 66, 98, 54, 106, 126, 94, 38, 50, 114, 74, 58, 82, 86, 42, 10, 118, 70, 102, 34, 30, 90, 62, 110, 122, 46, 18, 26, 78, 2, 6], 'token_total': 4096, 'token_per_expert': {19: 1, 99: 1, 39: 2, 107: 2, 23: 3, 27: 3, 43: 6, 59: 7, 55: 8, 87: 11, 115: 11, 119: 11, 71: 13, 127: 14, 123: 16, 79: 17, 75: 19, 31: 20, 95: 21, 67: 23, 111: 32, 35: 36, 83: 39, 103: 40, 51: 61, 11: 62, 3: 132, 7: 167, 56: 2, 60: 2, 52: 3, 104: 4, 16: 6, 96: 6, 32: 7, 88: 8, 40: 9, 64: 9, 20: 10, 44: 11, 24: 12, 80: 12, 12: 23, 36: 27, 72: 27, 84: 28, 68: 30, 76: 47, 8: 48, 124: 49, 112: 57, 92: 59, 120: 66, 48: 71, 100: 96, 0: 136, 4: 159, 25: 1, 17: 2, 89: 2, 113: 2, 77: 3, 9: 5, 101: 5, 69: 6, 21: 8, 93: 8, 121: 9, 45: 10, 33: 14, 125: 16, 81: 20, 57: 21, 13: 30, 97: 31, 61: 35, 29: 41, 37: 43, 109: 44, 53: 47, 73: 47, 41: 58, 105: 61, 65: 96, 1: 220, 5: 256, 66: 1, 98: 2, 54: 3, 106: 3, 126: 3, 94: 5, 38: 7, 50: 9, 114: 10, 74: 11, 58: 15, 82: 16, 86: 17, 42: 18, 10: 21, 118: 21, 70: 29, 102: 31, 34: 34, 30: 35, 90: 36, 62: 37, 110: 40, 122: 45, 46: 60, 18: 62, 26: 73, 78: 115, 2: 151, 6: 243}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:51.970186.970186 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.573ms | allocate_experts_across_cpu_gpu: 0.409ms
INFO 05-06 10:38:51.970865.970865 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:51.971934.971934 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005471706390380859 seconds
INFO 05-06 10:38:51.972660.972660 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008347034454345703 seconds
INFO 05-06 10:38:51.972531.972531 lmp.py:1484] [layer_moe_fused] experts compute time: 3.337860107421875e-06 seconds
INFO 05-06 10:38:52.007227.007227 lmp.py:1496] [layer_moe_fused] to time: 0.0001881122589111328 seconds
INFO 05-06 10:38:52.008688.008688 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.034902334213256836 seconds
DEBUG 05-06 10:38:52.008265.008265 cuda_h.py:27] end *layer_moe_fused cost 39.350 ms
DEBUG 05-06 10:38:52.009359.009359 cuda_h.py:27] end prefill_layer cost 43.583 ms
DEBUG 05-06 10:38:52.009852.009852 lmp.py:841] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 10:38:52.009814.009814 lmp.py:824] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 10:38:52.011746.011746 cuda_h.py:27] end *sagl cost 2.247 ms
experts_cpu_alloc {'expert_ids': [23, 71, 91, 67, 95, 39, 27, 87, 51, 63, 47, 79, 83, 99, 31, 15, 107, 11, 19, 111, 115, 75, 43, 123, 127, 119, 55, 59, 103, 35, 3, 7, 36, 56, 104, 60, 96, 84, 20, 32, 48, 124, 16, 112, 44, 40, 28, 88, 108, 76, 120, 8, 116, 92, 68, 24, 4, 72, 0, 64, 100, 13, 29, 37, 97, 105, 61, 65, 17, 81, 9, 57, 41, 25, 45, 109, 85, 113, 33, 125, 69, 101, 89, 53, 93, 117, 73, 5, 1, 14, 114, 54, 62, 106, 122, 10, 98, 110, 34, 102, 42, 30, 58, 26, 82, 94, 66, 118, 46, 70, 38, 90, 86, 74, 126, 6, 2], 'token_total': 4096, 'token_per_expert': {23: 1, 71: 1, 91: 1, 67: 2, 95: 2, 39: 3, 27: 5, 87: 5, 51: 8, 63: 9, 47: 10, 79: 10, 83: 12, 99: 14, 31: 16, 15: 18, 107: 19, 11: 22, 19: 22, 111: 30, 115: 31, 75: 34, 43: 37, 123: 48, 127: 49, 119: 53, 55: 55, 59: 57, 103: 66, 35: 113, 3: 129, 7: 157, 36: 1, 56: 1, 104: 1, 60: 2, 96: 2, 84: 5, 20: 6, 32: 9, 48: 11, 124: 16, 16: 18, 112: 19, 44: 20, 40: 22, 28: 24, 88: 26, 108: 34, 76: 39, 120: 43, 8: 46, 116: 46, 92: 48, 68: 57, 24: 84, 4: 128, 72: 128, 0: 140, 64: 142, 100: 222, 13: 1, 29: 1, 37: 1, 97: 2, 105: 2, 61: 3, 65: 3, 17: 4, 81: 4, 9: 5, 57: 6, 41: 10, 25: 12, 45: 12, 109: 13, 85: 14, 113: 14, 33: 18, 125: 18, 69: 21, 101: 22, 89: 33, 53: 55, 93: 59, 117: 62, 73: 79, 5: 130, 1: 149, 14: 2, 114: 2, 54: 3, 62: 5, 106: 5, 122: 6, 10: 8, 98: 9, 110: 9, 34: 13, 102: 19, 42: 20, 30: 22, 58: 23, 26: 24, 82: 28, 94: 32, 66: 33, 118: 33, 46: 37, 70: 40, 38: 44, 90: 49, 86: 51, 74: 80, 126: 106, 6: 130, 2: 131}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 32, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:52.014912.014912 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.571ms | allocate_experts_across_cpu_gpu: 0.414ms
INFO 05-06 10:38:52.014101.014101 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:52.015515.015515 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005576610565185547 seconds
INFO 05-06 10:38:52.016663.016663 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001032114028930664 seconds
INFO 05-06 10:38:52.016654.016654 lmp.py:1484] [layer_moe_fused] experts compute time: 9.059906005859375e-06 seconds
INFO 05-06 10:38:52.052015.052015 lmp.py:1496] [layer_moe_fused] to time: 0.00017261505126953125 seconds
INFO 05-06 10:38:52.052006.052006 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03578448295593262 seconds
DEBUG 05-06 10:38:52.053203.053203 cuda_h.py:27] end *layer_moe_fused cost 40.353 ms
DEBUG 05-06 10:38:52.054430.054430 cuda_h.py:27] end prefill_layer cost 44.687 ms
DEBUG 05-06 10:38:52.054538.054538 lmp.py:841] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 10:38:52.054069.054069 lmp.py:824] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 10:38:52.058763.058763 cuda_h.py:27] end *sagl cost 3.671 ms
experts_cpu_alloc {'expert_ids': [119, 15, 27, 111, 55, 95, 127, 11, 99, 107, 51, 23, 75, 103, 19, 71, 91, 115, 59, 87, 31, 83, 35, 79, 47, 123, 43, 67, 39, 7, 3, 64, 88, 28, 48, 32, 36, 92, 52, 40, 60, 120, 12, 76, 112, 68, 124, 80, 116, 24, 100, 72, 104, 16, 8, 108, 84, 44, 56, 0, 4, 13, 69, 77, 93, 41, 49, 101, 81, 57, 53, 89, 9, 73, 33, 105, 17, 117, 109, 85, 37, 97, 125, 25, 29, 61, 65, 21, 1, 5, 50, 114, 102, 62, 74, 54, 58, 82, 14, 110, 66, 10, 38, 42, 34, 122, 106, 22, 30, 26, 90, 78, 118, 18, 46, 98, 86, 6, 2], 'token_total': 4096, 'token_per_expert': {119: 1, 15: 2, 27: 2, 111: 2, 55: 3, 95: 3, 127: 3, 11: 5, 99: 5, 107: 6, 51: 7, 23: 10, 75: 11, 103: 12, 19: 13, 71: 14, 91: 14, 115: 17, 59: 21, 87: 23, 31: 26, 83: 38, 35: 46, 79: 46, 47: 50, 123: 53, 43: 82, 67: 91, 39: 104, 7: 135, 3: 183, 64: 2, 88: 2, 28: 4, 48: 6, 32: 7, 36: 7, 92: 7, 52: 8, 40: 9, 60: 9, 120: 9, 12: 14, 76: 14, 112: 16, 68: 17, 124: 17, 80: 20, 116: 21, 24: 22, 100: 25, 72: 30, 104: 32, 16: 34, 8: 35, 108: 39, 84: 40, 44: 52, 56: 113, 0: 130, 4: 131, 13: 1, 69: 1, 77: 1, 93: 1, 41: 3, 49: 3, 101: 3, 81: 4, 57: 9, 53: 10, 89: 11, 9: 12, 73: 15, 33: 17, 105: 22, 17: 25, 117: 26, 109: 37, 85: 47, 37: 60, 97: 61, 125: 66, 25: 69, 29: 70, 61: 73, 65: 92, 21: 152, 1: 186, 5: 190, 50: 1, 114: 1, 102: 2, 62: 3, 74: 3, 54: 4, 58: 4, 82: 4, 14: 7, 110: 7, 66: 8, 10: 10, 38: 13, 42: 14, 34: 15, 122: 15, 106: 26, 22: 27, 30: 27, 26: 28, 90: 33, 78: 38, 118: 47, 18: 51, 46: 63, 98: 65, 86: 105, 6: 147, 2: 161}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:52.060097.060097 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.585ms | allocate_experts_across_cpu_gpu: 0.417ms
INFO 05-06 10:38:52.060716.060716 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:52.061386.061386 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005590915679931641 seconds
INFO 05-06 10:38:52.062662.062662 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009455680847167969 seconds
INFO 05-06 10:38:52.062718.062718 lmp.py:1484] [layer_moe_fused] experts compute time: 1.6689300537109375e-06 seconds
INFO 05-06 10:38:52.097177.097177 lmp.py:1496] [layer_moe_fused] to time: 0.00017642974853515625 seconds
INFO 05-06 10:38:52.097107.097107 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03472161293029785 seconds
DEBUG 05-06 10:38:52.098346.098346 cuda_h.py:27] end *layer_moe_fused cost 38.898 ms
DEBUG 05-06 10:38:52.098327.098327 cuda_h.py:27] end prefill_layer cost 44.669 ms
DEBUG 05-06 10:38:52.099959.099959 lmp.py:841] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 10:38:52.099967.099967 lmp.py:824] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 10:38:52.101286.101286 cuda_h.py:27] end *sagl cost 2.244 ms
experts_cpu_alloc {'expert_ids': [51, 95, 15, 55, 31, 47, 87, 107, 79, 115, 99, 111, 119, 75, 127, 83, 43, 35, 23, 91, 67, 19, 71, 11, 63, 27, 3, 7, 112, 24, 76, 80, 84, 116, 68, 104, 124, 32, 92, 120, 28, 20, 96, 40, 100, 8, 60, 36, 48, 108, 16, 56, 12, 44, 52, 64, 0, 4, 21, 93, 25, 89, 65, 53, 61, 57, 81, 9, 105, 49, 13, 17, 109, 37, 29, 73, 77, 45, 33, 97, 5, 1, 121, 126, 22, 26, 78, 102, 18, 66, 46, 10, 38, 42, 106, 62, 110, 82, 118, 30, 50, 122, 74, 86, 94, 98, 114, 34, 70, 2, 90, 6], 'token_total': 4096, 'token_per_expert': {51: 1, 95: 1, 15: 2, 55: 2, 31: 4, 47: 4, 87: 4, 107: 6, 79: 7, 115: 8, 99: 10, 111: 12, 119: 12, 75: 14, 127: 21, 83: 27, 43: 31, 35: 36, 23: 41, 91: 47, 67: 52, 19: 58, 71: 60, 11: 79, 63: 87, 27: 89, 3: 128, 7: 148, 112: 1, 24: 2, 76: 4, 80: 4, 84: 4, 116: 4, 68: 5, 104: 6, 124: 8, 32: 9, 92: 9, 120: 9, 28: 11, 20: 12, 96: 12, 40: 14, 100: 14, 8: 17, 60: 21, 36: 24, 48: 26, 108: 28, 16: 38, 56: 65, 12: 66, 44: 74, 52: 85, 64: 124, 0: 128, 4: 173, 21: 1, 93: 1, 25: 2, 89: 2, 65: 3, 53: 6, 61: 8, 57: 9, 81: 10, 9: 12, 105: 19, 49: 20, 13: 21, 17: 23, 109: 26, 37: 29, 29: 33, 73: 43, 77: 48, 45: 57, 33: 80, 97: 98, 5: 149, 1: 167, 121: 192, 126: 2, 22: 3, 26: 3, 78: 3, 102: 4, 18: 5, 66: 5, 46: 6, 10: 7, 38: 8, 42: 9, 106: 11, 62: 13, 110: 18, 82: 22, 118: 23, 30: 24, 50: 24, 122: 24, 74: 26, 86: 30, 94: 37, 98: 46, 114: 48, 34: 71, 70: 95, 2: 128, 90: 152, 6: 202}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:52.103042.103042 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.563ms | allocate_experts_across_cpu_gpu: 0.419ms
INFO 05-06 10:38:52.103648.103648 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:52.104921.104921 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005671977996826172 seconds
INFO 05-06 10:38:52.105988.105988 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008761882781982422 seconds
INFO 05-06 10:38:52.106931.106931 lmp.py:1484] [layer_moe_fused] experts compute time: 1.9073486328125e-06 seconds
INFO 05-06 10:38:52.140687.140687 lmp.py:1496] [layer_moe_fused] to time: 0.00017499923706054688 seconds
INFO 05-06 10:38:52.140287.140287 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03452110290527344 seconds
DEBUG 05-06 10:38:52.141599.141599 cuda_h.py:27] end *layer_moe_fused cost 38.579 ms
DEBUG 05-06 10:38:52.141586.141586 cuda_h.py:27] end prefill_layer cost 42.827 ms
DEBUG 05-06 10:38:52.142179.142179 lmp.py:841] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 10:38:52.142471.142471 lmp.py:824] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 10:38:52.144263.144263 cuda_h.py:27] end *sagl cost 2.241 ms
experts_cpu_alloc {'expert_ids': [27, 55, 103, 127, 47, 119, 75, 31, 51, 87, 23, 43, 79, 99, 63, 91, 19, 71, 39, 67, 11, 111, 83, 35, 123, 107, 7, 3, 96, 108, 12, 24, 32, 76, 92, 112, 72, 124, 48, 8, 88, 56, 120, 116, 36, 44, 100, 104, 60, 80, 64, 52, 68, 4, 0, 16, 37, 65, 57, 33, 121, 29, 61, 53, 81, 13, 17, 9, 73, 77, 97, 21, 25, 89, 109, 125, 41, 49, 93, 117, 69, 45, 85, 1, 5, 62, 98, 102, 30, 94, 22, 38, 42, 46, 78, 122, 66, 126, 74, 26, 118, 50, 14, 106, 10, 82, 34, 90, 114, 70, 18, 110, 6, 58, 2], 'token_total': 4096, 'token_per_expert': {27: 1, 55: 2, 103: 2, 127: 3, 47: 6, 119: 6, 75: 7, 31: 9, 51: 11, 87: 11, 23: 13, 43: 13, 79: 13, 99: 16, 63: 18, 91: 20, 19: 22, 71: 22, 39: 23, 67: 26, 11: 27, 111: 33, 83: 47, 35: 89, 123: 100, 107: 121, 7: 153, 3: 172, 96: 3, 108: 3, 12: 4, 24: 4, 32: 4, 76: 5, 92: 6, 112: 6, 72: 8, 124: 9, 48: 12, 8: 14, 88: 17, 56: 20, 120: 21, 116: 23, 36: 26, 44: 31, 100: 32, 104: 45, 60: 59, 80: 59, 64: 64, 52: 68, 68: 127, 4: 130, 0: 144, 16: 220, 37: 1, 65: 1, 57: 2, 33: 3, 121: 3, 29: 4, 61: 4, 53: 5, 81: 6, 13: 7, 17: 7, 9: 9, 73: 9, 77: 10, 97: 14, 21: 15, 25: 16, 89: 18, 109: 18, 125: 19, 41: 20, 49: 20, 93: 22, 117: 46, 69: 70, 45: 75, 85: 82, 1: 132, 5: 144, 62: 1, 98: 1, 102: 1, 30: 2, 94: 2, 22: 3, 38: 4, 42: 6, 46: 7, 78: 8, 122: 8, 66: 11, 126: 11, 74: 12, 26: 13, 118: 13, 50: 19, 14: 22, 106: 29, 10: 31, 82: 32, 34: 37, 90: 39, 114: 39, 70: 46, 18: 90, 110: 126, 6: 134, 58: 186, 2: 231}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:52.146541.146541 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.567ms | allocate_experts_across_cpu_gpu: 0.407ms
INFO 05-06 10:38:52.146955.146955 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:52.147302.147302 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006046295166015625 seconds
INFO 05-06 10:38:52.149377.149377 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0010447502136230469 seconds
INFO 05-06 10:38:52.149194.149194 lmp.py:1484] [layer_moe_fused] experts compute time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:52.184002.184002 lmp.py:1496] [layer_moe_fused] to time: 0.0001728534698486328 seconds
INFO 05-06 10:38:52.184072.184072 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03521323204040527 seconds
DEBUG 05-06 10:38:52.185053.185053 cuda_h.py:27] end *layer_moe_fused cost 39.590 ms
DEBUG 05-06 10:38:52.185093.185093 cuda_h.py:27] end prefill_layer cost 43.883 ms
DEBUG 05-06 10:38:52.186917.186917 lmp.py:841] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 10:38:52.186972.186972 lmp.py:824] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 10:38:52.188736.188736 cuda_h.py:27] end *sagl cost 2.244 ms
experts_cpu_alloc {'expert_ids': [83, 119, 55, 107, 127, 31, 91, 63, 23, 47, 71, 35, 115, 67, 75, 103, 99, 19, 51, 79, 59, 15, 123, 27, 43, 95, 111, 7, 87, 3, 48, 100, 32, 40, 44, 92, 28, 80, 96, 72, 116, 8, 108, 112, 36, 88, 68, 56, 124, 76, 60, 52, 104, 84, 24, 4, 0, 20, 53, 93, 101, 9, 117, 33, 109, 121, 13, 41, 81, 25, 45, 57, 125, 61, 105, 29, 77, 97, 37, 49, 73, 65, 17, 5, 1, 89, 113, 85, 54, 46, 122, 74, 42, 110, 18, 30, 38, 14, 50, 118, 10, 26, 90, 66, 126, 78, 86, 102, 70, 114, 6, 2], 'token_total': 4096, 'token_per_expert': {83: 1, 119: 1, 55: 3, 107: 3, 127: 3, 31: 4, 91: 4, 63: 5, 23: 6, 47: 6, 71: 6, 35: 8, 115: 10, 67: 12, 75: 23, 103: 23, 99: 24, 19: 26, 51: 26, 79: 27, 59: 28, 15: 32, 123: 34, 27: 72, 43: 73, 95: 91, 111: 107, 7: 128, 87: 134, 3: 154, 48: 2, 100: 2, 32: 3, 40: 3, 44: 4, 92: 5, 28: 6, 80: 6, 96: 6, 72: 7, 116: 10, 8: 12, 108: 13, 112: 14, 36: 16, 88: 23, 68: 25, 56: 28, 124: 31, 76: 34, 60: 39, 52: 43, 104: 63, 84: 75, 24: 98, 4: 134, 0: 136, 20: 142, 53: 1, 93: 1, 101: 1, 9: 2, 117: 2, 33: 3, 109: 3, 121: 3, 13: 8, 41: 8, 81: 8, 25: 11, 45: 12, 57: 12, 125: 12, 61: 15, 105: 15, 29: 16, 77: 17, 97: 19, 37: 22, 49: 31, 73: 42, 65: 81, 17: 108, 5: 133, 1: 137, 89: 139, 113: 145, 85: 222, 54: 3, 46: 4, 122: 4, 74: 6, 42: 7, 110: 12, 18: 13, 30: 14, 38: 14, 14: 18, 50: 21, 118: 22, 10: 25, 26: 25, 90: 25, 66: 30, 126: 30, 78: 35, 86: 38, 102: 41, 70: 48, 114: 117, 6: 128, 2: 133}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:52.190994.190994 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.566ms | allocate_experts_across_cpu_gpu: 0.396ms
INFO 05-06 10:38:52.190931.190931 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:52.191031.191031 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005393028259277344 seconds
INFO 05-06 10:38:52.193001.193001 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012030601501464844 seconds
INFO 05-06 10:38:52.193032.193032 lmp.py:1484] [layer_moe_fused] experts compute time: 5.245208740234375e-06 seconds
INFO 05-06 10:38:52.228304.228304 lmp.py:1496] [layer_moe_fused] to time: 0.00017642974853515625 seconds
INFO 05-06 10:38:52.228380.228380 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03453230857849121 seconds
DEBUG 05-06 10:38:52.228117.228117 cuda_h.py:27] end *layer_moe_fused cost 39.257 ms
DEBUG 05-06 10:38:52.229866.229866 cuda_h.py:27] end prefill_layer cost 43.523 ms
DEBUG 05-06 10:38:52.229359.229359 lmp.py:841] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 10:38:52.229844.229844 lmp.py:824] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 10:38:52.232316.232316 cuda_h.py:27] end *sagl cost 2.232 ms
experts_cpu_alloc {'expert_ids': [99, 63, 67, 71, 39, 47, 11, 59, 15, 19, 27, 91, 55, 23, 83, 75, 35, 127, 119, 31, 111, 79, 51, 123, 103, 115, 95, 43, 87, 7, 3, 44, 72, 16, 84, 104, 60, 116, 32, 68, 80, 124, 96, 112, 40, 20, 56, 28, 8, 12, 64, 108, 36, 88, 76, 48, 100, 24, 120, 0, 4, 101, 69, 81, 89, 93, 97, 29, 113, 57, 125, 49, 21, 85, 105, 121, 41, 53, 109, 61, 13, 37, 25, 65, 33, 45, 5, 1, 22, 30, 34, 126, 86, 110, 10, 26, 74, 90, 106, 58, 122, 114, 94, 66, 18, 98, 118, 42, 54, 14, 70, 62, 78, 46, 82, 50, 2, 6], 'token_total': 4096, 'token_per_expert': {99: 1, 63: 3, 67: 3, 71: 3, 39: 4, 47: 5, 11: 6, 59: 6, 15: 7, 19: 9, 27: 9, 91: 10, 55: 13, 23: 15, 83: 19, 75: 20, 35: 22, 127: 24, 119: 32, 31: 34, 111: 38, 79: 42, 51: 51, 123: 52, 103: 69, 115: 70, 95: 75, 43: 104, 87: 108, 7: 157, 3: 171, 44: 1, 72: 1, 16: 2, 84: 2, 104: 2, 60: 5, 116: 5, 32: 6, 68: 6, 80: 6, 124: 7, 96: 8, 112: 15, 40: 16, 20: 17, 56: 21, 28: 23, 8: 32, 12: 33, 64: 33, 108: 34, 36: 39, 88: 54, 76: 59, 48: 65, 100: 76, 24: 83, 120: 89, 0: 133, 4: 144, 101: 1, 69: 2, 81: 2, 89: 3, 93: 3, 97: 3, 29: 4, 113: 5, 57: 6, 125: 7, 49: 10, 21: 11, 85: 14, 105: 24, 121: 28, 41: 29, 53: 29, 109: 34, 61: 40, 13: 48, 37: 49, 25: 51, 65: 65, 33: 66, 45: 87, 5: 129, 1: 172, 22: 1, 30: 1, 34: 1, 126: 1, 86: 4, 110: 4, 10: 5, 26: 11, 74: 12, 90: 14, 106: 15, 58: 19, 122: 22, 114: 23, 94: 24, 66: 25, 18: 26, 98: 26, 118: 26, 42: 28, 54: 31, 14: 39, 70: 39, 62: 45, 78: 45, 46: 51, 82: 68, 50: 108, 2: 128, 6: 133}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:52.234408.234408 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.573ms | allocate_experts_across_cpu_gpu: 0.410ms
INFO 05-06 10:38:52.234544.234544 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:52.235080.235080 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005512237548828125 seconds
INFO 05-06 10:38:52.236261.236261 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0007843971252441406 seconds
INFO 05-06 10:38:52.236045.236045 lmp.py:1484] [layer_moe_fused] experts compute time: 1.9073486328125e-06 seconds
INFO 05-06 10:38:52.270393.270393 lmp.py:1496] [layer_moe_fused] to time: 0.00017309188842773438 seconds
INFO 05-06 10:38:52.270755.270755 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.034003496170043945 seconds
DEBUG 05-06 10:38:52.271126.271126 cuda_h.py:27] end *layer_moe_fused cost 37.854 ms
DEBUG 05-06 10:38:52.271923.271923 cuda_h.py:27] end prefill_layer cost 42.111 ms
DEBUG 05-06 10:38:52.271046.271046 lmp.py:841] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 10:38:52.271292.271292 lmp.py:824] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 10:38:52.274167.274167 cuda_h.py:27] end *sagl cost 2.221 ms
experts_cpu_alloc {'expert_ids': [27, 59, 83, 99, 103, 31, 67, 87, 15, 19, 79, 127, 39, 23, 43, 95, 55, 123, 119, 11, 71, 75, 47, 91, 115, 7, 3, 111, 8, 28, 80, 72, 120, 92, 36, 44, 108, 48, 24, 100, 88, 84, 60, 104, 32, 68, 52, 40, 112, 76, 0, 4, 20, 12, 17, 93, 21, 29, 81, 109, 73, 33, 9, 65, 105, 69, 117, 97, 121, 37, 89, 101, 85, 77, 13, 113, 53, 57, 5, 1, 49, 34, 38, 82, 114, 118, 66, 54, 50, 58, 122, 18, 126, 62, 94, 98, 74, 106, 30, 22, 78, 70, 46, 110, 90, 2, 6], 'token_total': 4096, 'token_per_expert': {27: 1, 59: 1, 83: 1, 99: 1, 103: 1, 31: 2, 67: 2, 87: 2, 15: 3, 19: 3, 79: 6, 127: 7, 39: 8, 23: 13, 43: 16, 95: 17, 55: 20, 123: 23, 119: 31, 11: 40, 71: 47, 75: 52, 47: 53, 91: 93, 115: 124, 7: 130, 3: 131, 111: 293, 8: 1, 28: 1, 80: 1, 72: 2, 120: 2, 92: 3, 36: 4, 44: 5, 108: 5, 48: 6, 24: 8, 100: 10, 88: 15, 84: 17, 60: 19, 104: 24, 32: 35, 68: 44, 52: 58, 40: 70, 112: 90, 76: 107, 0: 128, 4: 128, 20: 221, 12: 279, 17: 1, 93: 1, 21: 2, 29: 2, 81: 4, 109: 4, 73: 6, 33: 8, 9: 9, 65: 11, 105: 11, 69: 12, 117: 14, 97: 15, 121: 17, 37: 20, 89: 23, 101: 24, 85: 25, 77: 30, 13: 33, 113: 38, 53: 52, 57: 127, 5: 144, 1: 146, 49: 211, 34: 1, 38: 1, 82: 1, 114: 1, 118: 1, 66: 2, 54: 3, 50: 5, 58: 5, 122: 9, 18: 10, 126: 11, 62: 12, 94: 12, 98: 13, 74: 16, 106: 19, 30: 21, 22: 36, 78: 43, 70: 44, 46: 45, 110: 52, 90: 81, 2: 128, 6: 130}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:52.276877.276877 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.627ms | allocate_experts_across_cpu_gpu: 0.387ms
INFO 05-06 10:38:52.276635.276635 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:52.277350.277350 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005421638488769531 seconds
INFO 05-06 10:38:52.278896.278896 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0010633468627929688 seconds
INFO 05-06 10:38:52.279130.279130 lmp.py:1484] [layer_moe_fused] experts compute time: 4.0531158447265625e-06 seconds
INFO 05-06 10:38:52.315438.315438 lmp.py:1496] [layer_moe_fused] to time: 0.0001766681671142578 seconds
INFO 05-06 10:38:52.315945.315945 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03597664833068848 seconds
DEBUG 05-06 10:38:52.316037.316037 cuda_h.py:27] end *layer_moe_fused cost 40.598 ms
DEBUG 05-06 10:38:52.316362.316362 cuda_h.py:27] end prefill_layer cost 44.810 ms
DEBUG 05-06 10:38:52.316425.316425 lmp.py:841] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 10:38:52.316101.316101 lmp.py:824] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 10:38:52.320374.320374 cuda_h.py:27] end *sagl cost 3.225 ms
experts_cpu_alloc {'expert_ids': [55, 127, 87, 111, 11, 31, 63, 83, 115, 75, 123, 15, 119, 35, 67, 95, 107, 27, 23, 71, 19, 43, 99, 91, 3, 7, 108, 40, 76, 88, 96, 32, 80, 84, 92, 24, 8, 116, 120, 44, 124, 48, 60, 16, 56, 28, 20, 64, 52, 0, 4, 17, 25, 125, 33, 105, 37, 13, 65, 21, 61, 9, 69, 85, 109, 81, 89, 77, 101, 73, 113, 49, 93, 29, 53, 97, 117, 121, 57, 5, 1, 38, 70, 74, 98, 102, 34, 118, 122, 126, 58, 94, 10, 46, 50, 114, 66, 78, 30, 62, 14, 26, 18, 54, 82, 22, 90, 42, 106, 86, 6, 2], 'token_total': 4096, 'token_per_expert': {55: 2, 127: 2, 87: 4, 111: 4, 11: 6, 31: 6, 63: 6, 83: 6, 115: 6, 75: 7, 123: 7, 15: 9, 119: 9, 35: 11, 67: 11, 95: 11, 107: 13, 27: 26, 23: 32, 71: 36, 19: 40, 43: 47, 99: 82, 91: 87, 3: 270, 7: 354, 108: 1, 40: 3, 76: 3, 88: 3, 96: 4, 32: 5, 80: 5, 84: 5, 92: 5, 24: 7, 8: 8, 116: 9, 120: 13, 44: 15, 124: 15, 48: 27, 60: 31, 16: 39, 56: 39, 28: 43, 20: 50, 64: 65, 52: 71, 0: 259, 4: 305, 17: 2, 25: 3, 125: 3, 33: 4, 105: 4, 37: 5, 13: 6, 65: 7, 21: 9, 61: 13, 9: 14, 69: 15, 85: 15, 109: 15, 81: 16, 89: 16, 77: 17, 101: 17, 73: 18, 113: 21, 49: 22, 93: 24, 29: 25, 53: 25, 97: 33, 117: 40, 121: 42, 57: 44, 5: 257, 1: 268, 38: 1, 70: 1, 74: 1, 98: 1, 102: 1, 34: 2, 118: 2, 122: 2, 126: 2, 58: 4, 94: 5, 10: 6, 46: 8, 50: 8, 114: 9, 66: 13, 78: 15, 30: 16, 62: 18, 14: 20, 26: 22, 18: 24, 54: 27, 82: 27, 22: 30, 90: 33, 42: 44, 106: 45, 86: 48, 6: 263, 2: 274}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:52.322062.322062 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.566ms | allocate_experts_across_cpu_gpu: 0.397ms
INFO 05-06 10:38:52.322059.322059 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:52.323851.323851 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005838871002197266 seconds
INFO 05-06 10:38:52.324063.324063 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008208751678466797 seconds
INFO 05-06 10:38:52.324107.324107 lmp.py:1484] [layer_moe_fused] experts compute time: 1.1920928955078125e-06 seconds
INFO 05-06 10:38:52.363963.363963 lmp.py:1496] [layer_moe_fused] to time: 0.0001735687255859375 seconds
INFO 05-06 10:38:52.363139.363139 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03926801681518555 seconds
DEBUG 05-06 10:38:52.364578.364578 cuda_h.py:27] end *layer_moe_fused cost 43.058 ms
DEBUG 05-06 10:38:52.365480.365480 cuda_h.py:27] end prefill_layer cost 48.268 ms
DEBUG 05-06 10:38:52.365919.365919 lmp.py:841] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 10:38:52.365166.365166 cuda_h.py:27] end prefill_step cost 2276.583 ms
INFO 05-06 10:38:52.365459.365459 lmp.py:843] prefill time: 2.377372980117798 seconds
WARNING 05-06 10:38:52.387358.387358 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:38:52.387788.387788 helper.py:35]   NaN count (hidden): 720896
WARNING 05-06 10:38:52.387277.387277 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:38:52.387944.387944 helper.py:39]   NaN count (normed): 720896
WARNING 05-06 10:38:52.394533.394533 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:38:52.394359.394359 helper.py:50]   NaN count: 524288
WARNING 05-06 10:38:52.394510.394510 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:38:52.498069.498069 cuda_h.py:27] end init_inputs_tokens cost 129.378 ms
DEBUG 05-06 10:38:52.498822.498822 lmp.py:899] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:38:52.498413.498413 lmp.py:904] ---- decode step 0 layer 0 ----
DEBUG 05-06 10:38:52.505183.505183 cuda_h.py:27] end *sagl cost 6.524 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 39, 47, 79, 87, 91, 103, 127], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 13, 'token_per_expert': {7: 1, 39: 1, 47: 2, 79: 2, 87: 1, 91: 2, 103: 3, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 52, 108, 116, 124], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {8: 1, 52: 1, 108: 1, 116: 2, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 53, 121], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {13: 1, 53: 2, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 22, 26, 78, 90, 106, 114], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {18: 1, 22: 2, 26: 2, 78: 1, 90: 1, 106: 1, 114: 1}}
INFO 05-06 10:38:52.507651.507651 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.563ms | allocate_experts_across_cpu_gpu: 0.124ms
INFO 05-06 10:38:52.507879.507879 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3365020751953125e-05 seconds
INFO 05-06 10:38:52.509684.509684 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0022399425506591797 seconds
INFO 05-06 10:38:52.557618.557618 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.047809600830078125 seconds
INFO 05-06 10:38:52.559927.559927 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018227100372314453 seconds
INFO 05-06 10:38:52.631548.631548 mlpmodule.py:2799] [fused_experts] gmm total=71.477ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.637665.637665 mlpmodule.py:2799] [fused_experts] gmm total=76.730ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.638393.638393 mlpmodule.py:2799] [fused_experts] gmm total=76.646ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.642484.642484 mlpmodule.py:2799] [fused_experts] gmm total=81.679ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.642713.642713 lmp.py:1484] [layer_moe_fused] experts compute time: 0.08341121673583984 seconds
INFO 05-06 10:38:52.643875.643875 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0001220703125 seconds
DEBUG 05-06 10:38:52.643518.643518 cuda_h.py:27] end *layer_moe_fused cost 137.341 ms
DEBUG 05-06 10:38:52.644821.644821 cuda_h.py:27] end decode_layer cost 146.450 ms
DEBUG 05-06 10:38:52.645164.645164 lmp.py:904] ---- decode step 0 layer 1 ----
DEBUG 05-06 10:38:52.651362.651362 cuda_h.py:27] end *sagl cost 6.332 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [67, 107], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 2, 'token_per_expert': {67: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 48, 56, 84, 96, 124], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {0: 3, 8: 1, 48: 1, 56: 2, 84: 1, 96: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 65, 97, 117, 121], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {5: 1, 9: 1, 13: 1, 65: 1, 97: 3, 117: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 22, 30, 46, 54, 106, 110], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {6: 1, 22: 2, 30: 4, 46: 1, 54: 1, 106: 1, 110: 1}}
INFO 05-06 10:38:52.655046.655046 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.707ms | allocate_experts_across_cpu_gpu: 0.155ms
INFO 05-06 10:38:52.655619.655619 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.765655517578125e-05 seconds
INFO 05-06 10:38:52.657737.657737 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016896724700927734 seconds
INFO 05-06 10:38:52.659698.659698 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0019290447235107422 seconds
INFO 05-06 10:38:52.661237.661237 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015840530395507812 seconds
INFO 05-06 10:38:52.663138.663138 mlpmodule.py:2799] [fused_experts] gmm total=2.112ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.664918.664918 mlpmodule.py:2799] [fused_experts] gmm total=2.545ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.665829.665829 mlpmodule.py:2799] [fused_experts] gmm total=3.365ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.665794.665794 mlpmodule.py:2799] [fused_experts] gmm total=3.592ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.666231.666231 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005079507827758789 seconds
INFO 05-06 10:38:52.666155.666155 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:38:52.666070.666070 cuda_h.py:27] end *layer_moe_fused cost 12.095 ms
DEBUG 05-06 10:38:52.667097.667097 cuda_h.py:27] end decode_layer cost 22.146 ms
DEBUG 05-06 10:38:52.667656.667656 lmp.py:904] ---- decode step 0 layer 2 ----
DEBUG 05-06 10:38:52.669447.669447 cuda_h.py:27] end *sagl cost 1.631 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 59, 119], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {11: 4, 59: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [52, 60, 76, 108, 120], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {52: 2, 60: 1, 76: 3, 108: 1, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 29, 41, 49, 65, 77, 81], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 1, 29: 1, 41: 1, 49: 2, 65: 1, 77: 1, 81: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [62, 78, 102, 106, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {62: 2, 78: 1, 102: 1, 106: 1, 126: 3}}
INFO 05-06 10:38:52.670260.670260 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.333ms | allocate_experts_across_cpu_gpu: 0.101ms
INFO 05-06 10:38:52.670898.670898 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.002716064453125e-05 seconds
INFO 05-06 10:38:52.671255.671255 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001523733139038086 seconds
INFO 05-06 10:38:52.673447.673447 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013034343719482422 seconds
INFO 05-06 10:38:52.674338.674338 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014336109161376953 seconds
INFO 05-06 10:38:52.677933.677933 mlpmodule.py:2799] [fused_experts] gmm total=1.899ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.677762.677762 mlpmodule.py:2799] [fused_experts] gmm total=2.197ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.677437.677437 mlpmodule.py:2799] [fused_experts] gmm total=2.216ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.677209.677209 mlpmodule.py:2799] [fused_experts] gmm total=2.230ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.678038.678038 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004030704498291016 seconds
INFO 05-06 10:38:52.679055.679055 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:38:52.679613.679613 cuda_h.py:27] end *layer_moe_fused cost 9.616 ms
DEBUG 05-06 10:38:52.680505.680505 cuda_h.py:27] end decode_layer cost 12.700 ms
DEBUG 05-06 10:38:52.680156.680156 lmp.py:904] ---- decode step 0 layer 3 ----
DEBUG 05-06 10:38:52.681765.681765 cuda_h.py:27] end *sagl cost 1.743 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 27, 39, 67, 107, 127], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {19: 1, 27: 1, 39: 1, 67: 1, 107: 2, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [48, 84, 96, 100, 104], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 10, 'token_per_expert': {48: 1, 84: 1, 96: 3, 100: 1, 104: 4}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 109, 117], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {5: 1, 109: 1, 117: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 22, 26, 34, 54, 102, 110, 118], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 22: 1, 26: 1, 34: 1, 54: 1, 102: 1, 110: 1, 118: 4}}
INFO 05-06 10:38:52.683450.683450 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.326ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 10:38:52.683519.683519 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 10:38:52.684225.684225 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014667510986328125 seconds
INFO 05-06 10:38:52.686110.686110 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012574195861816406 seconds
INFO 05-06 10:38:52.687723.687723 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014367103576660156 seconds
INFO 05-06 10:38:52.690953.690953 mlpmodule.py:2799] [fused_experts] gmm total=1.946ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.690340.690340 mlpmodule.py:2799] [fused_experts] gmm total=2.135ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.690342.690342 mlpmodule.py:2799] [fused_experts] gmm total=2.640ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.691616.691616 mlpmodule.py:2799] [fused_experts] gmm total=3.435ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.692590.692590 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004469633102416992 seconds
INFO 05-06 10:38:52.692608.692608 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:38:52.692389.692389 cuda_h.py:27] end *layer_moe_fused cost 9.897 ms
DEBUG 05-06 10:38:52.693631.693631 cuda_h.py:27] end decode_layer cost 13.196 ms
DEBUG 05-06 10:38:52.693805.693805 lmp.py:904] ---- decode step 0 layer 4 ----
DEBUG 05-06 10:38:52.694910.694910 cuda_h.py:27] end *sagl cost 1.512 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 43, 47, 51, 83, 115], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {3: 1, 43: 1, 47: 1, 51: 1, 83: 3, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 20, 24, 60, 76, 84], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {8: 1, 20: 1, 24: 1, 60: 4, 76: 1, 84: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 17, 21, 25, 45, 81, 93, 105], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 8, 'token_per_expert': {5: 1, 17: 1, 21: 1, 25: 1, 45: 1, 81: 1, 93: 1, 105: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [82, 114, 126], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {82: 2, 114: 1, 126: 3}}
INFO 05-06 10:38:52.696178.696178 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.314ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:38:52.696532.696532 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:38:52.697132.697132 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014586448669433594 seconds
INFO 05-06 10:38:52.699304.699304 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013260841369628906 seconds
INFO 05-06 10:38:52.700323.700323 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012848377227783203 seconds
INFO 05-06 10:38:52.702104.702104 mlpmodule.py:2799] [fused_experts] gmm total=1.910ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.703900.703900 mlpmodule.py:2799] [fused_experts] gmm total=2.428ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.704709.704709 mlpmodule.py:2799] [fused_experts] gmm total=3.195ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.704394.704394 mlpmodule.py:2799] [fused_experts] gmm total=3.769ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.705397.705397 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004735231399536133 seconds
INFO 05-06 10:38:52.705991.705991 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:38:52.705312.705312 cuda_h.py:27] end *layer_moe_fused cost 9.958 ms
DEBUG 05-06 10:38:52.706579.706579 cuda_h.py:27] end decode_layer cost 12.819 ms
DEBUG 05-06 10:38:52.706468.706468 lmp.py:904] ---- decode step 0 layer 5 ----
DEBUG 05-06 10:38:52.708396.708396 cuda_h.py:27] end *sagl cost 1.766 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [31, 39, 47, 63, 71, 79, 99, 119, 123, 127], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 13, 'token_per_expert': {31: 1, 39: 3, 47: 1, 63: 1, 71: 1, 79: 1, 99: 2, 119: 1, 123: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [36, 52, 60, 64], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {36: 3, 52: 2, 60: 1, 64: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 17, 29, 61, 101], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {5: 1, 17: 1, 29: 1, 61: 1, 101: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 46, 70, 74, 106], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {22: 1, 46: 1, 70: 2, 74: 1, 106: 1}}
INFO 05-06 10:38:52.709760.709760 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.316ms | allocate_experts_across_cpu_gpu: 0.102ms
INFO 05-06 10:38:52.709782.709782 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2172927856445312e-05 seconds
INFO 05-06 10:38:52.710784.710784 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014050006866455078 seconds
INFO 05-06 10:38:52.712540.712540 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001341104507446289 seconds
INFO 05-06 10:38:52.713348.713348 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015118122100830078 seconds
INFO 05-06 10:38:52.716349.716349 mlpmodule.py:2799] [fused_experts] gmm total=2.201ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.716952.716952 mlpmodule.py:2799] [fused_experts] gmm total=2.137ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.716194.716194 mlpmodule.py:2799] [fused_experts] gmm total=2.425ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.717089.717089 mlpmodule.py:2799] [fused_experts] gmm total=3.173ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.718993.718993 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004376649856567383 seconds
INFO 05-06 10:38:52.718341.718341 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:38:52.718657.718657 cuda_h.py:27] end *layer_moe_fused cost 9.825 ms
DEBUG 05-06 10:38:52.719963.719963 cuda_h.py:27] end decode_layer cost 13.000 ms
DEBUG 05-06 10:38:52.719376.719376 lmp.py:904] ---- decode step 0 layer 6 ----
DEBUG 05-06 10:38:52.720568.720568 cuda_h.py:27] end *sagl cost 1.542 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [35, 67, 71, 87, 91, 107, 111, 115], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {35: 2, 67: 1, 71: 1, 87: 2, 91: 1, 107: 1, 111: 1, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 64, 100, 104, 124], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {20: 1, 64: 1, 100: 1, 104: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 25, 53], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {1: 1, 9: 1, 13: 1, 25: 1, 53: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 42, 46, 58, 90, 106, 110], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {10: 1, 42: 1, 46: 1, 58: 2, 90: 1, 106: 4, 110: 2}}
INFO 05-06 10:38:52.722866.722866 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.293ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:38:52.722651.722651 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 10:38:52.723045.723045 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016596317291259766 seconds
INFO 05-06 10:38:52.725847.725847 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013332366943359375 seconds
INFO 05-06 10:38:52.726535.726535 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014965534210205078 seconds
INFO 05-06 10:38:52.729425.729425 mlpmodule.py:2799] [fused_experts] gmm total=2.112ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.729386.729386 mlpmodule.py:2799] [fused_experts] gmm total=2.357ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.730926.730926 mlpmodule.py:2799] [fused_experts] gmm total=2.728ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.730094.730094 mlpmodule.py:2799] [fused_experts] gmm total=3.191ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.731179.731179 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004341840744018555 seconds
INFO 05-06 10:38:52.731912.731912 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:38:52.731427.731427 cuda_h.py:27] end *layer_moe_fused cost 9.828 ms
DEBUG 05-06 10:38:52.732736.732736 cuda_h.py:27] end decode_layer cost 12.725 ms
DEBUG 05-06 10:38:52.732625.732625 lmp.py:904] ---- decode step 0 layer 7 ----
DEBUG 05-06 10:38:52.733493.733493 cuda_h.py:27] end *sagl cost 1.758 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 83], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {7: 1, 83: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 60, 72, 80, 84, 96, 108, 120], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {20: 1, 60: 1, 72: 1, 80: 3, 84: 2, 96: 1, 108: 1, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [21, 57, 65, 69, 97, 113, 121], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {21: 1, 57: 1, 65: 1, 69: 2, 97: 1, 113: 1, 121: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 34, 50, 90, 106, 126], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {18: 1, 34: 1, 50: 1, 90: 2, 106: 1, 126: 1}}
INFO 05-06 10:38:52.735873.735873 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.319ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 10:38:52.735604.735604 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:38:52.736781.736781 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012884140014648438 seconds
INFO 05-06 10:38:52.738600.738600 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012841224670410156 seconds
INFO 05-06 10:38:52.739596.739596 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011970996856689453 seconds
INFO 05-06 10:38:52.741620.741620 mlpmodule.py:2799] [fused_experts] gmm total=2.225ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.742628.742628 mlpmodule.py:2799] [fused_experts] gmm total=2.298ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.742104.742104 mlpmodule.py:2799] [fused_experts] gmm total=2.567ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.742512.742512 mlpmodule.py:2799] [fused_experts] gmm total=2.847ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.743126.743126 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004470109939575195 seconds
INFO 05-06 10:38:52.743289.743289 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:38:52.744619.744619 cuda_h.py:27] end *layer_moe_fused cost 9.463 ms
DEBUG 05-06 10:38:52.744767.744767 cuda_h.py:27] end decode_layer cost 12.740 ms
DEBUG 05-06 10:38:52.744895.744895 lmp.py:904] ---- decode step 0 layer 8 ----
DEBUG 05-06 10:38:52.746374.746374 cuda_h.py:27] end *sagl cost 1.612 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 51, 63, 71, 87], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 9, 'token_per_expert': {3: 1, 7: 1, 15: 1, 27: 1, 51: 1, 63: 1, 71: 2, 87: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 32, 64, 76], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 1, 4: 1, 28: 2, 32: 1, 64: 2, 76: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 65, 73, 125], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {1: 1, 5: 1, 33: 1, 65: 1, 73: 1, 125: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 50, 54, 98, 110, 118], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {2: 1, 6: 2, 50: 1, 54: 1, 98: 1, 110: 1, 118: 1}}
INFO 05-06 10:38:52.747063.747063 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.348ms | allocate_experts_across_cpu_gpu: 0.111ms
INFO 05-06 10:38:52.748185.748185 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2411346435546875e-05 seconds
INFO 05-06 10:38:52.749611.749611 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013980865478515625 seconds
INFO 05-06 10:38:52.750754.750754 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014481544494628906 seconds
INFO 05-06 10:38:52.752587.752587 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012831687927246094 seconds
INFO 05-06 10:38:52.754695.754695 mlpmodule.py:2799] [fused_experts] gmm total=2.038ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.754774.754774 mlpmodule.py:2799] [fused_experts] gmm total=2.161ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.754636.754636 mlpmodule.py:2799] [fused_experts] gmm total=2.297ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.755255.755255 mlpmodule.py:2799] [fused_experts] gmm total=2.330ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.756260.756260 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004014015197753906 seconds
INFO 05-06 10:38:52.756893.756893 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.792213439941406e-05 seconds
DEBUG 05-06 10:38:52.756250.756250 cuda_h.py:27] end *layer_moe_fused cost 9.413 ms
DEBUG 05-06 10:38:52.757479.757479 cuda_h.py:27] end decode_layer cost 12.504 ms
DEBUG 05-06 10:38:52.757654.757654 lmp.py:904] ---- decode step 0 layer 9 ----
DEBUG 05-06 10:38:52.759024.759024 cuda_h.py:27] end *sagl cost 1.533 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 95], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {3: 1, 7: 2, 19: 1, 95: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 48, 68, 76], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 48: 1, 68: 1, 76: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 37, 57, 69, 81, 89, 93, 101], 'expert_count': 10, 'ideal_gpu_count': 7, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 11, 'token_per_expert': {1: 1, 5: 1, 13: 1, 37: 1, 57: 1, 69: 1, 81: 1, 89: 1, 93: 1, 101: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 70, 74, 102, 106], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 9, 'token_per_expert': {2: 1, 6: 1, 38: 1, 46: 1, 70: 1, 74: 1, 102: 2, 106: 1}}
INFO 05-06 10:38:52.760578.760578 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.306ms | allocate_experts_across_cpu_gpu: 0.101ms
INFO 05-06 10:38:52.760793.760793 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2411346435546875e-05 seconds
INFO 05-06 10:38:52.761518.761518 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014467239379882812 seconds
INFO 05-06 10:38:52.763723.763723 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013232231140136719 seconds
INFO 05-06 10:38:52.764195.764195 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013592243194580078 seconds
INFO 05-06 10:38:52.767942.767942 mlpmodule.py:2799] [fused_experts] gmm total=2.061ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.767511.767511 mlpmodule.py:2799] [fused_experts] gmm total=2.335ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.767989.767989 mlpmodule.py:2799] [fused_experts] gmm total=2.376ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.767218.767218 mlpmodule.py:2799] [fused_experts] gmm total=2.378ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.769122.769122 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004411458969116211 seconds
INFO 05-06 10:38:52.769855.769855 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:38:52.769722.769722 cuda_h.py:27] end *layer_moe_fused cost 9.644 ms
DEBUG 05-06 10:38:52.770276.770276 cuda_h.py:27] end decode_layer cost 12.637 ms
DEBUG 05-06 10:38:52.770927.770927 lmp.py:904] ---- decode step 0 layer 10 ----
DEBUG 05-06 10:38:52.771052.771052 cuda_h.py:27] end *sagl cost 1.528 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 75, 115], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {3: 1, 7: 1, 75: 1, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 28, 44, 60, 76, 88], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {0: 1, 4: 1, 8: 1, 28: 2, 44: 1, 60: 3, 76: 2, 88: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 81, 121], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {1: 1, 5: 1, 37: 2, 81: 2, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 42, 46, 62], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {2: 1, 6: 1, 14: 1, 18: 1, 42: 1, 46: 1, 62: 2}}
INFO 05-06 10:38:52.773022.773022 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.322ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 10:38:52.773276.773276 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:38:52.774298.774298 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015947818756103516 seconds
INFO 05-06 10:38:52.776775.776775 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013387203216552734 seconds
INFO 05-06 10:38:52.777323.777323 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014581680297851562 seconds
INFO 05-06 10:38:52.779263.779263 mlpmodule.py:2799] [fused_experts] gmm total=1.944ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.780036.780036 mlpmodule.py:2799] [fused_experts] gmm total=2.110ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.780671.780671 mlpmodule.py:2799] [fused_experts] gmm total=2.375ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.780163.780163 mlpmodule.py:2799] [fused_experts] gmm total=2.478ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.781425.781425 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003977537155151367 seconds
INFO 05-06 10:38:52.781581.781581 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.8160552978515625e-05 seconds
DEBUG 05-06 10:38:52.781872.781872 cuda_h.py:27] end *layer_moe_fused cost 9.402 ms
DEBUG 05-06 10:38:52.782114.782114 cuda_h.py:27] end decode_layer cost 12.317 ms
DEBUG 05-06 10:38:52.782619.782619 lmp.py:904] ---- decode step 0 layer 11 ----
DEBUG 05-06 10:38:52.784353.784353 cuda_h.py:27] end *sagl cost 1.695 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 83], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 1, 7: 2, 23: 1, 83: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 32, 56, 92], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 1, 4: 1, 16: 1, 32: 1, 56: 2, 92: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 49, 57, 81, 93, 117], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 9, 'token_per_expert': {1: 1, 5: 1, 17: 1, 49: 1, 57: 1, 81: 1, 93: 1, 117: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 38, 46, 66, 102], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 1, 6: 1, 18: 1, 38: 1, 46: 2, 66: 2, 102: 1}}
INFO 05-06 10:38:52.785502.785502 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.294ms | allocate_experts_across_cpu_gpu: 0.092ms
INFO 05-06 10:38:52.785664.785664 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3603439331054688e-05 seconds
INFO 05-06 10:38:52.787650.787650 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013256072998046875 seconds
INFO 05-06 10:38:52.788542.788542 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012710094451904297 seconds
INFO 05-06 10:38:52.790207.790207 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015704631805419922 seconds
INFO 05-06 10:38:52.792287.792287 mlpmodule.py:2799] [fused_experts] gmm total=2.036ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.792666.792666 mlpmodule.py:2799] [fused_experts] gmm total=2.180ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.792720.792720 mlpmodule.py:2799] [fused_experts] gmm total=2.216ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.792783.792783 mlpmodule.py:2799] [fused_experts] gmm total=2.414ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.794301.794301 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004064321517944336 seconds
INFO 05-06 10:38:52.794034.794034 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 10:38:52.794645.794645 cuda_h.py:27] end *layer_moe_fused cost 9.586 ms
DEBUG 05-06 10:38:52.795666.795666 cuda_h.py:27] end decode_layer cost 12.669 ms
DEBUG 05-06 10:38:52.795317.795317 lmp.py:904] ---- decode step 0 layer 12 ----
DEBUG 05-06 10:38:52.796602.796602 cuda_h.py:27] end *sagl cost 1.540 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 39, 71, 95, 115], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {3: 1, 7: 1, 15: 1, 19: 1, 39: 2, 71: 1, 95: 2, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 40, 80, 84, 116], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 7, 'token_per_expert': {0: 1, 4: 1, 36: 1, 40: 1, 80: 1, 84: 1, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 45, 49, 101, 117], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 1, 5: 1, 45: 2, 49: 1, 101: 1, 117: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 46, 78, 114], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 1, 6: 1, 46: 1, 78: 3, 114: 1}}
INFO 05-06 10:38:52.798738.798738 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.315ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:38:52.798529.798529 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.4080276489257812e-05 seconds
INFO 05-06 10:38:52.799314.799314 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014531612396240234 seconds
INFO 05-06 10:38:52.801376.801376 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001382589340209961 seconds
INFO 05-06 10:38:52.802095.802095 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014469623565673828 seconds
INFO 05-06 10:38:52.804541.804541 mlpmodule.py:2799] [fused_experts] gmm total=2.003ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.805423.805423 mlpmodule.py:2799] [fused_experts] gmm total=2.151ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.805139.805139 mlpmodule.py:2799] [fused_experts] gmm total=2.315ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.805578.805578 mlpmodule.py:2799] [fused_experts] gmm total=2.668ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.806310.806310 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0041637420654296875 seconds
INFO 05-06 10:38:52.807181.807181 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.8160552978515625e-05 seconds
DEBUG 05-06 10:38:52.807942.807942 cuda_h.py:27] end *layer_moe_fused cost 9.487 ms
DEBUG 05-06 10:38:52.807979.807979 cuda_h.py:27] end decode_layer cost 12.456 ms
DEBUG 05-06 10:38:52.807915.807915 lmp.py:904] ---- decode step 0 layer 13 ----
DEBUG 05-06 10:38:52.809258.809258 cuda_h.py:27] end *sagl cost 1.514 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 31, 55, 79, 83, 99, 107], 'expert_count': 9, 'ideal_gpu_count': 7, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 10, 'token_per_expert': {3: 1, 7: 1, 15: 1, 31: 1, 55: 1, 79: 2, 83: 1, 99: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 68, 100, 104], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 1, 4: 1, 68: 1, 100: 3, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 41, 73, 113, 121], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 3, 5: 1, 41: 1, 73: 1, 113: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 78, 110], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 1, 6: 1, 14: 1, 78: 3, 110: 1}}
INFO 05-06 10:38:52.810214.810214 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.294ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 10:38:52.810706.810706 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-05 seconds
INFO 05-06 10:38:52.812918.812918 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001346588134765625 seconds
INFO 05-06 10:38:52.813763.813763 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012328624725341797 seconds
INFO 05-06 10:38:52.814105.814105 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001453399658203125 seconds
INFO 05-06 10:38:52.817357.817357 mlpmodule.py:2799] [fused_experts] gmm total=2.244ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.817396.817396 mlpmodule.py:2799] [fused_experts] gmm total=2.240ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.817862.817862 mlpmodule.py:2799] [fused_experts] gmm total=2.770ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.818668.818668 mlpmodule.py:2799] [fused_experts] gmm total=2.734ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.819553.819553 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0044078826904296875 seconds
INFO 05-06 10:38:52.819855.819855 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:38:52.819940.819940 cuda_h.py:27] end *layer_moe_fused cost 9.634 ms
DEBUG 05-06 10:38:52.820928.820928 cuda_h.py:27] end decode_layer cost 12.434 ms
DEBUG 05-06 10:38:52.820148.820148 lmp.py:904] ---- decode step 0 layer 14 ----
DEBUG 05-06 10:38:52.821132.821132 cuda_h.py:27] end *sagl cost 1.459 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 75, 115], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {3: 1, 7: 1, 31: 1, 39: 1, 47: 1, 75: 1, 115: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 44, 56, 112, 124], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 7, 'token_per_expert': {0: 1, 4: 1, 8: 1, 44: 1, 56: 1, 112: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 65, 97, 117, 121], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {1: 1, 5: 1, 65: 1, 97: 1, 117: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 26, 34, 38, 86], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 2, 6: 1, 10: 1, 26: 2, 34: 1, 38: 2, 86: 1}}
INFO 05-06 10:38:52.823484.823484 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.295ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:38:52.823500.823500 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2411346435546875e-05 seconds
INFO 05-06 10:38:52.824006.824006 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014126300811767578 seconds
INFO 05-06 10:38:52.826430.826430 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0018680095672607422 seconds
INFO 05-06 10:38:52.828083.828083 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014340877532958984 seconds
INFO 05-06 10:38:52.830525.830525 mlpmodule.py:2799] [fused_experts] gmm total=2.102ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.830095.830095 mlpmodule.py:2799] [fused_experts] gmm total=2.216ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.833180.833180 mlpmodule.py:2799] [fused_experts] gmm total=5.045ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.834470.834470 mlpmodule.py:2799] [fused_experts] gmm total=6.095ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.835505.835505 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007195949554443359 seconds
INFO 05-06 10:38:52.835622.835622 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 10:38:52.835547.835547 cuda_h.py:27] end *layer_moe_fused cost 13.149 ms
DEBUG 05-06 10:38:52.836801.836801 cuda_h.py:27] end decode_layer cost 15.925 ms
DEBUG 05-06 10:38:52.836690.836690 lmp.py:904] ---- decode step 0 layer 15 ----
DEBUG 05-06 10:38:52.838294.838294 cuda_h.py:27] end *sagl cost 1.773 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 75, 83, 91, 119, 127], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {3: 1, 7: 1, 23: 1, 75: 1, 83: 3, 91: 2, 119: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 72, 76, 104, 108, 112], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {0: 1, 4: 1, 72: 1, 76: 1, 104: 1, 108: 1, 112: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 33, 65, 81, 121], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {1: 1, 5: 1, 9: 2, 33: 1, 65: 1, 81: 2, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {2: 1, 6: 1, 38: 2}}
INFO 05-06 10:38:52.839116.839116 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.392ms | allocate_experts_across_cpu_gpu: 0.147ms
INFO 05-06 10:38:52.839120.839120 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.8371810913085938e-05 seconds
INFO 05-06 10:38:52.841014.841014 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015234947204589844 seconds
INFO 05-06 10:38:52.843284.843284 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016634464263916016 seconds
INFO 05-06 10:38:52.844903.844903 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001592874526977539 seconds
INFO 05-06 10:38:52.847687.847687 mlpmodule.py:2799] [fused_experts] gmm total=1.795ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.847378.847378 mlpmodule.py:2799] [fused_experts] gmm total=2.232ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.847974.847974 mlpmodule.py:2799] [fused_experts] gmm total=2.328ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.847818.847818 mlpmodule.py:2799] [fused_experts] gmm total=2.617ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.849412.849412 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00415802001953125 seconds
INFO 05-06 10:38:52.849251.849251 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.744529724121094e-05 seconds
DEBUG 05-06 10:38:52.849686.849686 cuda_h.py:27] end *layer_moe_fused cost 10.444 ms
DEBUG 05-06 10:38:52.850456.850456 cuda_h.py:27] end decode_layer cost 13.761 ms
DEBUG 05-06 10:38:52.850391.850391 lmp.py:904] ---- decode step 0 layer 16 ----
DEBUG 05-06 10:38:52.851523.851523 cuda_h.py:27] end *sagl cost 1.533 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 47, 55, 63, 107, 119], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {3: 1, 7: 1, 15: 1, 47: 1, 55: 1, 63: 2, 107: 2, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 44, 56], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {0: 1, 4: 3, 8: 1, 20: 2, 44: 2, 56: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 65, 77, 81], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {1: 2, 5: 1, 65: 1, 77: 1, 81: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 38, 54, 70], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 6, 'token_per_expert': {2: 1, 6: 1, 18: 1, 38: 1, 54: 1, 70: 1}}
INFO 05-06 10:38:52.852308.852308 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.321ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:38:52.853139.853139 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:38:52.854500.854500 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014579296112060547 seconds
INFO 05-06 10:38:52.855137.855137 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013527870178222656 seconds
INFO 05-06 10:38:52.857021.857021 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001428365707397461 seconds
INFO 05-06 10:38:52.859687.859687 mlpmodule.py:2799] [fused_experts] gmm total=2.157ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.860017.860017 mlpmodule.py:2799] [fused_experts] gmm total=2.162ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.860116.860116 mlpmodule.py:2799] [fused_experts] gmm total=2.808ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.860617.860617 mlpmodule.py:2799] [fused_experts] gmm total=2.759ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.861527.861527 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004344940185546875 seconds
INFO 05-06 10:38:52.861445.861445 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.76837158203125e-05 seconds
DEBUG 05-06 10:38:52.862451.862451 cuda_h.py:27] end *layer_moe_fused cost 9.618 ms
DEBUG 05-06 10:38:52.862996.862996 cuda_h.py:27] end decode_layer cost 12.519 ms
DEBUG 05-06 10:38:52.862455.862455 lmp.py:904] ---- decode step 0 layer 17 ----
DEBUG 05-06 10:38:52.864924.864924 cuda_h.py:27] end *sagl cost 1.500 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 75], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {3: 1, 7: 1, 23: 1, 27: 1, 75: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 68, 96, 100, 108, 124], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 68: 1, 96: 1, 100: 1, 108: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 37, 53, 69, 73, 113], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {1: 1, 5: 3, 9: 1, 21: 1, 37: 1, 53: 1, 69: 2, 73: 1, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 38, 78], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {2: 1, 6: 1, 18: 1, 38: 1, 78: 2}}
INFO 05-06 10:38:52.865847.865847 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.305ms | allocate_experts_across_cpu_gpu: 0.101ms
INFO 05-06 10:38:52.865771.865771 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.09808349609375e-05 seconds
INFO 05-06 10:38:52.867196.867196 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014028549194335938 seconds
INFO 05-06 10:38:52.868057.868057 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013163089752197266 seconds
INFO 05-06 10:38:52.869132.869132 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013966560363769531 seconds
INFO 05-06 10:38:52.871675.871675 mlpmodule.py:2799] [fused_experts] gmm total=1.763ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.872983.872983 mlpmodule.py:2799] [fused_experts] gmm total=2.138ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.873785.873785 mlpmodule.py:2799] [fused_experts] gmm total=2.984ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.873799.873799 mlpmodule.py:2799] [fused_experts] gmm total=3.288ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.874081.874081 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004538774490356445 seconds
INFO 05-06 10:38:52.874674.874674 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.8160552978515625e-05 seconds
DEBUG 05-06 10:38:52.874302.874302 cuda_h.py:27] end *layer_moe_fused cost 9.837 ms
DEBUG 05-06 10:38:52.875309.875309 cuda_h.py:27] end decode_layer cost 12.632 ms
DEBUG 05-06 10:38:52.875768.875768 lmp.py:904] ---- decode step 0 layer 18 ----
DEBUG 05-06 10:38:52.876682.876682 cuda_h.py:27] end *sagl cost 1.548 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 55, 75, 83, 95, 111, 115, 127], 'expert_count': 9, 'ideal_gpu_count': 7, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 9, 'token_per_expert': {3: 1, 7: 1, 55: 1, 75: 1, 83: 1, 95: 1, 111: 1, 115: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 120], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 1, 4: 1, 12: 1, 120: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 37, 77, 85, 93], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 1, 5: 1, 17: 1, 37: 2, 77: 2, 85: 2, 93: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 34, 46, 50, 54], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 7, 'token_per_expert': {2: 1, 6: 1, 30: 1, 34: 1, 46: 1, 50: 1, 54: 1}}
INFO 05-06 10:38:52.878135.878135 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.314ms | allocate_experts_across_cpu_gpu: 0.101ms
INFO 05-06 10:38:52.878489.878489 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:38:52.879413.879413 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001451730728149414 seconds
INFO 05-06 10:38:52.881228.881228 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013425350189208984 seconds
INFO 05-06 10:38:52.882577.882577 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001455545425415039 seconds
INFO 05-06 10:38:52.884608.884608 mlpmodule.py:2799] [fused_experts] gmm total=1.823ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.885001.885001 mlpmodule.py:2799] [fused_experts] gmm total=2.377ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.885895.885895 mlpmodule.py:2799] [fused_experts] gmm total=2.673ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.886461.886461 mlpmodule.py:2799] [fused_experts] gmm total=3.082ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.887564.887564 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00427699089050293 seconds
INFO 05-06 10:38:52.887104.887104 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:38:52.887620.887620 cuda_h.py:27] end *layer_moe_fused cost 9.939 ms
DEBUG 05-06 10:38:52.888709.888709 cuda_h.py:27] end decode_layer cost 12.872 ms
DEBUG 05-06 10:38:52.888360.888360 lmp.py:904] ---- decode step 0 layer 19 ----
DEBUG 05-06 10:38:52.889003.889003 cuda_h.py:27] end *sagl cost 1.558 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 75], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {3: 1, 7: 2, 35: 1, 75: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 40, 44, 52, 64, 96, 104], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 14, 'token_per_expert': {0: 1, 4: 1, 24: 2, 40: 1, 44: 3, 52: 1, 64: 1, 96: 3, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 73, 89], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {1: 1, 5: 1, 17: 1, 73: 1, 89: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 38, 106], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 1, 6: 1, 10: 2, 38: 3, 106: 1}}
INFO 05-06 10:38:52.891278.891278 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.308ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:38:52.891016.891016 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 10:38:52.892603.892603 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014705657958984375 seconds
INFO 05-06 10:38:52.894471.894471 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001669168472290039 seconds
INFO 05-06 10:38:52.896911.896911 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001577615737915039 seconds
INFO 05-06 10:38:52.898920.898920 mlpmodule.py:2799] [fused_experts] gmm total=1.886ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.899778.899778 mlpmodule.py:2799] [fused_experts] gmm total=2.267ms E=32 S=14 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.899604.899604 mlpmodule.py:2799] [fused_experts] gmm total=2.687ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.899770.899770 mlpmodule.py:2799] [fused_experts] gmm total=2.846ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.900436.900436 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004276275634765625 seconds
INFO 05-06 10:38:52.900831.900831 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.792213439941406e-05 seconds
DEBUG 05-06 10:38:52.901764.901764 cuda_h.py:27] end *layer_moe_fused cost 10.387 ms
DEBUG 05-06 10:38:52.901641.901641 cuda_h.py:27] end decode_layer cost 13.333 ms
DEBUG 05-06 10:38:52.901861.901861 lmp.py:904] ---- decode step 0 layer 20 ----
DEBUG 05-06 10:38:52.903522.903522 cuda_h.py:27] end *sagl cost 1.501 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 63, 79, 107], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {3: 2, 7: 1, 63: 2, 79: 1, 107: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 28, 68], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {0: 1, 4: 2, 20: 2, 28: 2, 68: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 41, 45, 73, 81], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 1, 5: 1, 13: 1, 41: 1, 45: 2, 73: 2, 81: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 74, 94], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {2: 2, 6: 1, 74: 1, 94: 1}}
INFO 05-06 10:38:52.904989.904989 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.304ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 10:38:52.904098.904098 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7404556274414062e-05 seconds
INFO 05-06 10:38:52.906273.906273 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014259815216064453 seconds
INFO 05-06 10:38:52.907809.907809 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001312255859375 seconds
INFO 05-06 10:38:52.908787.908787 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012531280517578125 seconds
INFO 05-06 10:38:52.911325.911325 mlpmodule.py:2799] [fused_experts] gmm total=2.049ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.911864.911864 mlpmodule.py:2799] [fused_experts] gmm total=2.249ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.912386.912386 mlpmodule.py:2799] [fused_experts] gmm total=2.995ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.912757.912757 mlpmodule.py:2799] [fused_experts] gmm total=3.297ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.913794.913794 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004510641098022461 seconds
INFO 05-06 10:38:52.913904.913904 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 10:38:52.913307.913307 cuda_h.py:27] end *layer_moe_fused cost 9.692 ms
DEBUG 05-06 10:38:52.914506.914506 cuda_h.py:27] end decode_layer cost 12.513 ms
DEBUG 05-06 10:38:52.914441.914441 lmp.py:904] ---- decode step 0 layer 21 ----
DEBUG 05-06 10:38:52.915436.915436 cuda_h.py:27] end *sagl cost 1.572 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 51, 83], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {3: 1, 7: 2, 11: 3, 51: 2, 83: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 112, 120], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {0: 1, 4: 1, 112: 1, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 57, 125], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {1: 3, 5: 1, 57: 2, 125: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 26, 34, 46, 78, 102, 126], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 11, 'token_per_expert': {2: 1, 6: 1, 18: 1, 26: 2, 34: 1, 46: 2, 78: 1, 102: 1, 126: 1}}
INFO 05-06 10:38:52.917764.917764 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.318ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 10:38:52.917018.917018 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 10:38:52.918254.918254 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014717578887939453 seconds
INFO 05-06 10:38:52.920121.920121 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013120174407958984 seconds
INFO 05-06 10:38:52.921609.921609 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012433528900146484 seconds
INFO 05-06 10:38:52.923959.923959 mlpmodule.py:2799] [fused_experts] gmm total=1.831ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.923083.923083 mlpmodule.py:2799] [fused_experts] gmm total=2.262ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.924765.924765 mlpmodule.py:2799] [fused_experts] gmm total=3.072ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.925448.925448 mlpmodule.py:2799] [fused_experts] gmm total=3.213ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.926337.926337 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004464626312255859 seconds
INFO 05-06 10:38:52.926639.926639 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:38:52.926954.926954 cuda_h.py:27] end *layer_moe_fused cost 9.857 ms
DEBUG 05-06 10:38:52.927247.927247 cuda_h.py:27] end decode_layer cost 12.772 ms
DEBUG 05-06 10:38:52.927613.927613 lmp.py:904] ---- decode step 0 layer 22 ----
DEBUG 05-06 10:38:52.928413.928413 cuda_h.py:27] end *sagl cost 1.707 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 43, 59, 75, 119, 127], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {3: 1, 7: 1, 35: 1, 43: 1, 59: 1, 75: 2, 119: 2, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 40, 72, 100, 108, 120], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 9, 'token_per_expert': {0: 1, 4: 1, 8: 1, 40: 1, 72: 1, 100: 1, 108: 1, 120: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 25, 89, 93, 109, 121], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {1: 1, 5: 1, 25: 1, 89: 2, 93: 2, 109: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 70, 126], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {2: 1, 6: 1, 70: 1, 126: 1}}
INFO 05-06 10:38:52.930938.930938 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.349ms | allocate_experts_across_cpu_gpu: 0.117ms
INFO 05-06 10:38:52.930882.930882 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3365020751953125e-05 seconds
INFO 05-06 10:38:52.931204.931204 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012826919555664062 seconds
INFO 05-06 10:38:52.933952.933952 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0015044212341308594 seconds
INFO 05-06 10:38:52.934157.934157 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014824867248535156 seconds
INFO 05-06 10:38:52.937572.937572 mlpmodule.py:2799] [fused_experts] gmm total=2.113ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.937994.937994 mlpmodule.py:2799] [fused_experts] gmm total=2.261ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.938647.938647 mlpmodule.py:2799] [fused_experts] gmm total=2.559ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.938664.938664 mlpmodule.py:2799] [fused_experts] gmm total=3.179ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.939832.939832 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0044634342193603516 seconds
INFO 05-06 10:38:52.939418.939418 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.8160552978515625e-05 seconds
DEBUG 05-06 10:38:52.939440.939440 cuda_h.py:27] end *layer_moe_fused cost 10.108 ms
DEBUG 05-06 10:38:52.940468.940468 cuda_h.py:27] end decode_layer cost 13.265 ms
DEBUG 05-06 10:38:52.940642.940642 lmp.py:904] ---- decode step 0 layer 23 ----
DEBUG 05-06 10:38:52.942833.942833 cuda_h.py:27] end *sagl cost 1.716 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 67, 99, 115], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 7, 'token_per_expert': {3: 1, 7: 1, 31: 1, 39: 1, 67: 1, 99: 1, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 24, 72], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 12: 1, 24: 1, 72: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 29, 97, 109], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {1: 2, 5: 2, 21: 1, 29: 1, 97: 2, 109: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 74, 86, 106, 118], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 2, 6: 1, 26: 1, 74: 1, 86: 1, 106: 1, 118: 3}}
INFO 05-06 10:38:52.943890.943890 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.303ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:38:52.943443.943443 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.9087066650390625e-05 seconds
INFO 05-06 10:38:52.945638.945638 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001440286636352539 seconds
INFO 05-06 10:38:52.946473.946473 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013229846954345703 seconds
INFO 05-06 10:38:52.947227.947227 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013000965118408203 seconds
INFO 05-06 10:38:52.950352.950352 mlpmodule.py:2799] [fused_experts] gmm total=1.936ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.950697.950697 mlpmodule.py:2799] [fused_experts] gmm total=2.056ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.951014.951014 mlpmodule.py:2799] [fused_experts] gmm total=3.042ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.951814.951814 mlpmodule.py:2799] [fused_experts] gmm total=3.286ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.952547.952547 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0045664310455322266 seconds
INFO 05-06 10:38:52.952081.952081 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.7206878662109375e-05 seconds
DEBUG 05-06 10:38:52.952823.952823 cuda_h.py:27] end *layer_moe_fused cost 9.867 ms
DEBUG 05-06 10:38:52.953737.953737 cuda_h.py:27] end decode_layer cost 12.916 ms
DEBUG 05-06 10:38:52.953765.953765 lmp.py:904] ---- decode step 0 layer 24 ----
DEBUG 05-06 10:38:52.954954.954954 cuda_h.py:27] end *sagl cost 1.436 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 91, 119, 123], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {3: 1, 7: 1, 19: 1, 91: 1, 119: 1, 123: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 48, 56, 60, 124], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 1, 4: 1, 12: 3, 48: 1, 56: 1, 60: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 57, 73, 97], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {1: 1, 5: 1, 57: 1, 73: 2, 97: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 26, 30, 70, 94, 110], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 10, 'token_per_expert': {2: 1, 6: 2, 10: 1, 18: 1, 26: 1, 30: 1, 70: 1, 94: 1, 110: 1}}
INFO 05-06 10:38:52.956016.956016 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.306ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:38:52.956747.956747 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0265579223632812e-05 seconds
INFO 05-06 10:38:52.957469.957469 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015342235565185547 seconds
INFO 05-06 10:38:52.961716.961716 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0034694671630859375 seconds
INFO 05-06 10:38:52.963726.963726 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017123222351074219 seconds
INFO 05-06 10:38:52.965491.965491 mlpmodule.py:2799] [fused_experts] gmm total=2.068ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.966249.966249 mlpmodule.py:2799] [fused_experts] gmm total=2.280ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.967284.967284 mlpmodule.py:2799] [fused_experts] gmm total=3.212ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.967853.967853 mlpmodule.py:2799] [fused_experts] gmm total=3.305ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.968556.968556 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004840850830078125 seconds
INFO 05-06 10:38:52.968424.968424 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 8.7738037109375e-05 seconds
DEBUG 05-06 10:38:52.969372.969372 cuda_h.py:27] end *layer_moe_fused cost 13.409 ms
DEBUG 05-06 10:38:52.970698.970698 cuda_h.py:27] end decode_layer cost 16.578 ms
DEBUG 05-06 10:38:52.970662.970662 lmp.py:904] ---- decode step 0 layer 25 ----
DEBUG 05-06 10:38:52.973273.973273 cuda_h.py:27] end *sagl cost 3.313 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 35, 59, 107], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {3: 1, 7: 1, 15: 1, 19: 2, 27: 1, 35: 1, 59: 1, 107: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 68, 92, 104], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 6, 'token_per_expert': {0: 1, 4: 1, 16: 1, 68: 1, 92: 1, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 41, 49, 89, 117], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {1: 1, 5: 1, 41: 2, 49: 1, 89: 1, 117: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 58, 70, 86, 114], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 2, 6: 1, 58: 2, 70: 1, 86: 1, 114: 1}}
INFO 05-06 10:38:52.976987.976987 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.668ms | allocate_experts_across_cpu_gpu: 0.225ms
INFO 05-06 10:38:52.976417.976417 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.220008850097656e-05 seconds
INFO 05-06 10:38:52.978356.978356 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014603137969970703 seconds
INFO 05-06 10:38:52.979821.979821 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0015666484832763672 seconds
INFO 05-06 10:38:52.981337.981337 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012903213500976562 seconds
INFO 05-06 10:38:52.983659.983659 mlpmodule.py:2799] [fused_experts] gmm total=2.293ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.983632.983632 mlpmodule.py:2799] [fused_experts] gmm total=2.394ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.984178.984178 mlpmodule.py:2799] [fused_experts] gmm total=2.674ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.984559.984559 mlpmodule.py:2799] [fused_experts] gmm total=2.877ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.985743.985743 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004549980163574219 seconds
INFO 05-06 10:38:52.985019.985019 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.507469177246094e-05 seconds
DEBUG 05-06 10:38:52.985484.985484 cuda_h.py:27] end *layer_moe_fused cost 10.610 ms
DEBUG 05-06 10:38:52.986647.986647 cuda_h.py:27] end decode_layer cost 16.349 ms
DEBUG 05-06 10:38:52.986404.986404 lmp.py:904] ---- decode step 0 layer 26 ----
DEBUG 05-06 10:38:52.988803.988803 cuda_h.py:27] end *sagl cost 1.792 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 27, 43, 103, 119, 123], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 8, 'token_per_expert': {3: 1, 7: 1, 19: 1, 27: 1, 43: 1, 103: 1, 119: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 72, 80, 104], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 72: 1, 80: 1, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 65, 85, 97], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 1, 5: 2, 17: 1, 65: 2, 85: 1, 97: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 34, 66, 70, 78, 98, 110], 'expert_count': 9, 'ideal_gpu_count': 7, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 11, 'token_per_expert': {2: 1, 6: 1, 26: 1, 34: 1, 66: 1, 70: 2, 78: 1, 98: 1, 110: 2}}
INFO 05-06 10:38:52.989386.989386 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.337ms | allocate_experts_across_cpu_gpu: 0.114ms
INFO 05-06 10:38:52.989362.989362 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.288818359375e-05 seconds
INFO 05-06 10:38:52.991578.991578 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014531612396240234 seconds
INFO 05-06 10:38:52.992616.992616 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014677047729492188 seconds
INFO 05-06 10:38:52.994615.994615 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014736652374267578 seconds
INFO 05-06 10:38:52.996748.996748 mlpmodule.py:2799] [fused_experts] gmm total=1.972ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.996310.996310 mlpmodule.py:2799] [fused_experts] gmm total=2.218ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.998875.998875 mlpmodule.py:2799] [fused_experts] gmm total=3.173ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.998643.998643 mlpmodule.py:2799] [fused_experts] gmm total=3.328ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:52.999798.999798 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004589080810546875 seconds
INFO 05-06 10:38:52.999199.999199 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:38:53.000314.000314 cuda_h.py:27] end *layer_moe_fused cost 11.352 ms
DEBUG 05-06 10:38:53.001568.001568 cuda_h.py:27] end decode_layer cost 14.582 ms
DEBUG 05-06 10:38:53.001796.001796 lmp.py:904] ---- decode step 0 layer 27 ----
DEBUG 05-06 10:38:53.002593.002593 cuda_h.py:27] end *sagl cost 1.599 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 95, 115, 119], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 1, 23: 1, 27: 2, 95: 1, 115: 1, 119: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 88], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {0: 1, 4: 1, 8: 1, 88: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 85, 105, 121], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 6, 'token_per_expert': {1: 1, 5: 1, 37: 1, 85: 1, 105: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 50, 62, 78, 82, 94, 122], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {2: 1, 6: 1, 26: 2, 50: 1, 62: 1, 78: 2, 82: 1, 94: 1, 122: 2}}
INFO 05-06 10:38:53.004008.004008 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:38:53.004216.004216 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0265579223632812e-05 seconds
INFO 05-06 10:38:53.006109.006109 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016939640045166016 seconds
INFO 05-06 10:38:53.008924.008924 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.002032756805419922 seconds
INFO 05-06 10:38:53.009412.009412 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001453399658203125 seconds
INFO 05-06 10:38:53.011269.011269 mlpmodule.py:2799] [fused_experts] gmm total=2.023ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.012275.012275 mlpmodule.py:2799] [fused_experts] gmm total=2.101ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.013565.013565 mlpmodule.py:2799] [fused_experts] gmm total=3.326ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.013109.013109 mlpmodule.py:2799] [fused_experts] gmm total=3.479ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.014366.014366 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004887104034423828 seconds
INFO 05-06 10:38:53.014953.014953 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:38:53.015509.015509 cuda_h.py:27] end *layer_moe_fused cost 11.212 ms
DEBUG 05-06 10:38:53.015478.015478 cuda_h.py:27] end decode_layer cost 14.402 ms
DEBUG 05-06 10:38:53.015605.015605 lmp.py:904] ---- decode step 0 layer 28 ----
DEBUG 05-06 10:38:53.017466.017466 cuda_h.py:27] end *sagl cost 1.543 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 91, 95, 111, 119], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {3: 1, 7: 1, 39: 1, 91: 1, 95: 1, 111: 1, 119: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 48, 76, 100, 108], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {0: 1, 4: 1, 20: 2, 48: 1, 76: 1, 100: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 49, 57, 77, 97], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {1: 1, 5: 1, 37: 1, 49: 2, 57: 2, 77: 1, 97: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 30, 98, 106, 126], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 7, 'token_per_expert': {2: 1, 6: 1, 26: 1, 30: 1, 98: 1, 106: 1, 126: 1}}
INFO 05-06 10:38:53.018118.018118 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.297ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:38:53.018187.018187 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.002716064453125e-05 seconds
INFO 05-06 10:38:53.020551.020551 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013580322265625 seconds
INFO 05-06 10:38:53.021725.021725 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013692378997802734 seconds
INFO 05-06 10:38:53.022225.022225 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014309883117675781 seconds
INFO 05-06 10:38:53.025397.025397 mlpmodule.py:2799] [fused_experts] gmm total=2.149ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.025782.025782 mlpmodule.py:2799] [fused_experts] gmm total=2.263ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.026844.026844 mlpmodule.py:2799] [fused_experts] gmm total=3.167ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.026421.026421 mlpmodule.py:2799] [fused_experts] gmm total=3.463ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.027494.027494 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004585742950439453 seconds
INFO 05-06 10:38:53.027557.027557 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:38:53.028524.028524 cuda_h.py:27] end *layer_moe_fused cost 9.959 ms
DEBUG 05-06 10:38:53.028782.028782 cuda_h.py:27] end decode_layer cost 12.800 ms
DEBUG 05-06 10:38:53.028480.028480 lmp.py:904] ---- decode step 0 layer 29 ----
DEBUG 05-06 10:38:53.030639.030639 cuda_h.py:27] end *sagl cost 1.762 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {3: 2, 7: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 56, 64, 108], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 20: 1, 56: 1, 64: 2, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 93, 101], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 3, 5: 2, 93: 1, 101: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 42, 54, 62, 74, 82, 106], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {2: 2, 6: 2, 26: 1, 42: 1, 54: 2, 62: 1, 74: 1, 82: 1, 106: 1}}
INFO 05-06 10:38:53.031750.031750 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.369ms | allocate_experts_across_cpu_gpu: 0.115ms
INFO 05-06 10:38:53.032402.032402 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:38:53.033958.033958 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013146400451660156 seconds
INFO 05-06 10:38:53.034831.034831 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014922618865966797 seconds
INFO 05-06 10:38:53.036437.036437 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012128353118896484 seconds
INFO 05-06 10:38:53.038612.038612 mlpmodule.py:2799] [fused_experts] gmm total=2.415ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.039527.039527 mlpmodule.py:2799] [fused_experts] gmm total=2.307ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.039174.039174 mlpmodule.py:2799] [fused_experts] gmm total=2.567ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.040196.040196 mlpmodule.py:2799] [fused_experts] gmm total=3.936ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:53.041193.041193 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004914283752441406 seconds
INFO 05-06 10:38:53.041257.041257 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.792213439941406e-05 seconds
DEBUG 05-06 10:38:53.041793.041793 cuda_h.py:27] end *layer_moe_fused cost 10.282 ms
DEBUG 05-06 10:38:53.042774.042774 cuda_h.py:27] end decode_layer cost 13.518 ms
DEBUG 05-06 10:38:53.042901.042901 cuda_h.py:27] end decode_step cost 673.242 ms
INFO 05-06 10:38:53.042512.042512 lmp.py:931] decode step 0 time: 0.6732735633850098 seconds
WARNING 05-06 10:38:53.042188.042188 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:38:53.042159.042159 helper.py:35]   NaN count (hidden): 5632
WARNING 05-06 10:38:53.042806.042806 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:38:53.042425.042425 helper.py:39]   NaN count (normed): 5632
WARNING 05-06 10:38:53.048709.048709 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:38:53.048363.048363 helper.py:50]   NaN count: 524288
WARNING 05-06 10:38:53.048808.048808 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:38:53.049468.049468 cuda_h.py:27] end init_inputs_tokens cost 7.708 ms
DEBUG 05-06 10:38:53.050550.050550 lmp.py:899] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:38:53.050366.050366 lmp.py:904] ---- decode step 1 layer 0 ----
DEBUG 05-06 10:38:53.051379.051379 cuda_h.py:27] end *sagl cost 1.550 ms
DEBUG 05-06 10:38:53.055858.055858 cuda_h.py:27] end *layer_moe_fused cost 2.953 ms
DEBUG 05-06 10:38:53.056094.056094 cuda_h.py:27] end decode_layer cost 5.937 ms
DEBUG 05-06 10:38:53.056243.056243 lmp.py:904] ---- decode step 1 layer 1 ----
DEBUG 05-06 10:38:53.058561.058561 cuda_h.py:27] end *sagl cost 1.932 ms
DEBUG 05-06 10:38:53.061861.061861 cuda_h.py:27] end *layer_moe_fused cost 2.523 ms
DEBUG 05-06 10:38:53.062904.062904 cuda_h.py:27] end decode_layer cost 6.101 ms
DEBUG 05-06 10:38:53.062476.062476 lmp.py:904] ---- decode step 1 layer 2 ----
DEBUG 05-06 10:38:53.064813.064813 cuda_h.py:27] end *sagl cost 1.912 ms
DEBUG 05-06 10:38:53.067796.067796 cuda_h.py:27] end *layer_moe_fused cost 2.370 ms
DEBUG 05-06 10:38:53.068044.068044 cuda_h.py:27] end decode_layer cost 5.891 ms
DEBUG 05-06 10:38:53.068616.068616 lmp.py:904] ---- decode step 1 layer 3 ----
DEBUG 05-06 10:38:53.070206.070206 cuda_h.py:27] end *sagl cost 1.922 ms
DEBUG 05-06 10:38:53.073268.073268 cuda_h.py:27] end *layer_moe_fused cost 2.144 ms
DEBUG 05-06 10:38:53.073019.073019 cuda_h.py:27] end decode_layer cost 5.699 ms
DEBUG 05-06 10:38:53.074352.074352 lmp.py:904] ---- decode step 1 layer 4 ----
DEBUG 05-06 10:38:53.075941.075941 cuda_h.py:27] end *sagl cost 1.886 ms
DEBUG 05-06 10:38:53.079739.079739 cuda_h.py:27] end *layer_moe_fused cost 2.026 ms
DEBUG 05-06 10:38:53.079405.079405 cuda_h.py:27] end decode_layer cost 5.511 ms
DEBUG 05-06 10:38:53.079773.079773 lmp.py:904] ---- decode step 1 layer 5 ----
DEBUG 05-06 10:38:53.081626.081626 cuda_h.py:27] end *sagl cost 1.867 ms
DEBUG 05-06 10:38:53.084552.084552 cuda_h.py:27] end *layer_moe_fused cost 2.056 ms
DEBUG 05-06 10:38:53.085403.085403 cuda_h.py:27] end decode_layer cost 5.551 ms
DEBUG 05-06 10:38:53.085697.085697 lmp.py:904] ---- decode step 1 layer 6 ----
DEBUG 05-06 10:38:53.087311.087311 cuda_h.py:27] end *sagl cost 1.868 ms
DEBUG 05-06 10:38:53.090102.090102 cuda_h.py:27] end *layer_moe_fused cost 2.116 ms
DEBUG 05-06 10:38:53.090999.090999 cuda_h.py:27] end decode_layer cost 5.660 ms
DEBUG 05-06 10:38:53.091618.091618 lmp.py:904] ---- decode step 1 layer 7 ----
DEBUG 05-06 10:38:53.093353.093353 cuda_h.py:27] end *sagl cost 1.924 ms
DEBUG 05-06 10:38:53.096928.096928 cuda_h.py:27] end *layer_moe_fused cost 2.263 ms
DEBUG 05-06 10:38:53.096309.096309 cuda_h.py:27] end decode_layer cost 5.807 ms
DEBUG 05-06 10:38:53.096927.096927 lmp.py:904] ---- decode step 1 layer 8 ----
DEBUG 05-06 10:38:53.098833.098833 cuda_h.py:27] end *sagl cost 1.874 ms
DEBUG 05-06 10:38:53.102598.102598 cuda_h.py:27] end *layer_moe_fused cost 2.176 ms
DEBUG 05-06 10:38:53.102071.102071 cuda_h.py:27] end decode_layer cost 5.684 ms
DEBUG 05-06 10:38:53.102928.102928 lmp.py:904] ---- decode step 1 layer 9 ----
DEBUG 05-06 10:38:53.104956.104956 cuda_h.py:27] end *sagl cost 1.965 ms
DEBUG 05-06 10:38:53.107482.107482 cuda_h.py:27] end *layer_moe_fused cost 1.958 ms
DEBUG 05-06 10:38:53.108187.108187 cuda_h.py:27] end decode_layer cost 5.558 ms
DEBUG 05-06 10:38:53.108852.108852 lmp.py:904] ---- decode step 1 layer 10 ----
DEBUG 05-06 10:38:53.110750.110750 cuda_h.py:27] end *sagl cost 1.870 ms
DEBUG 05-06 10:38:53.113277.113277 cuda_h.py:27] end *layer_moe_fused cost 1.996 ms
DEBUG 05-06 10:38:53.113359.113359 cuda_h.py:27] end decode_layer cost 5.460 ms
DEBUG 05-06 10:38:53.113931.113931 lmp.py:904] ---- decode step 1 layer 11 ----
DEBUG 05-06 10:38:53.115704.115704 cuda_h.py:27] end *sagl cost 1.882 ms
DEBUG 05-06 10:38:53.118077.118077 cuda_h.py:27] end *layer_moe_fused cost 1.539 ms
DEBUG 05-06 10:38:53.118729.118729 cuda_h.py:27] end decode_layer cost 5.043 ms
DEBUG 05-06 10:38:53.118394.118394 lmp.py:904] ---- decode step 1 layer 12 ----
DEBUG 05-06 10:38:53.120042.120042 cuda_h.py:27] end *sagl cost 1.896 ms
DEBUG 05-06 10:38:53.123142.123142 cuda_h.py:27] end *layer_moe_fused cost 1.544 ms
DEBUG 05-06 10:38:53.124868.124868 cuda_h.py:27] end decode_layer cost 5.049 ms
DEBUG 05-06 10:38:53.124770.124770 lmp.py:904] ---- decode step 1 layer 13 ----
DEBUG 05-06 10:38:53.126736.126736 cuda_h.py:27] end *sagl cost 1.884 ms
DEBUG 05-06 10:38:53.128315.128315 cuda_h.py:27] end *layer_moe_fused cost 1.560 ms
DEBUG 05-06 10:38:53.129536.129536 cuda_h.py:27] end decode_layer cost 5.053 ms
DEBUG 05-06 10:38:53.129724.129724 lmp.py:904] ---- decode step 1 layer 14 ----
DEBUG 05-06 10:38:53.131165.131165 cuda_h.py:27] end *sagl cost 1.849 ms
DEBUG 05-06 10:38:53.133265.133265 cuda_h.py:27] end *layer_moe_fused cost 1.535 ms
DEBUG 05-06 10:38:53.134579.134579 cuda_h.py:27] end decode_layer cost 4.980 ms
DEBUG 05-06 10:38:53.134005.134005 lmp.py:904] ---- decode step 1 layer 15 ----
DEBUG 05-06 10:38:53.136799.136799 cuda_h.py:27] end *sagl cost 1.898 ms
DEBUG 05-06 10:38:53.138349.138349 cuda_h.py:27] end *layer_moe_fused cost 1.519 ms
DEBUG 05-06 10:38:53.139855.139855 cuda_h.py:27] end decode_layer cost 5.011 ms
DEBUG 05-06 10:38:53.139520.139520 lmp.py:904] ---- decode step 1 layer 16 ----
DEBUG 05-06 10:38:53.141174.141174 cuda_h.py:27] end *sagl cost 1.865 ms
DEBUG 05-06 10:38:53.143353.143353 cuda_h.py:27] end *layer_moe_fused cost 1.543 ms
DEBUG 05-06 10:38:53.144581.144581 cuda_h.py:27] end decode_layer cost 4.991 ms
DEBUG 05-06 10:38:53.144769.144769 lmp.py:904] ---- decode step 1 layer 17 ----
DEBUG 05-06 10:38:53.146336.146336 cuda_h.py:27] end *sagl cost 1.836 ms
DEBUG 05-06 10:38:53.148582.148582 cuda_h.py:27] end *layer_moe_fused cost 1.535 ms
DEBUG 05-06 10:38:53.149949.149949 cuda_h.py:27] end decode_layer cost 4.973 ms
DEBUG 05-06 10:38:53.149898.149898 lmp.py:904] ---- decode step 1 layer 18 ----
DEBUG 05-06 10:38:53.151558.151558 cuda_h.py:27] end *sagl cost 1.835 ms
DEBUG 05-06 10:38:53.154071.154071 cuda_h.py:27] end *layer_moe_fused cost 1.558 ms
DEBUG 05-06 10:38:53.154914.154914 cuda_h.py:27] end decode_layer cost 5.028 ms
DEBUG 05-06 10:38:53.154685.154685 lmp.py:904] ---- decode step 1 layer 19 ----
DEBUG 05-06 10:38:53.156254.156254 cuda_h.py:27] end *sagl cost 1.907 ms
DEBUG 05-06 10:38:53.159195.159195 cuda_h.py:27] end *layer_moe_fused cost 1.506 ms
DEBUG 05-06 10:38:53.159656.159656 cuda_h.py:27] end decode_layer cost 5.064 ms
DEBUG 05-06 10:38:53.159467.159467 lmp.py:904] ---- decode step 1 layer 20 ----
DEBUG 05-06 10:38:53.161994.161994 cuda_h.py:27] end *sagl cost 1.843 ms
DEBUG 05-06 10:38:53.164120.164120 cuda_h.py:27] end *layer_moe_fused cost 1.544 ms
DEBUG 05-06 10:38:53.164249.164249 cuda_h.py:27] end decode_layer cost 4.961 ms
DEBUG 05-06 10:38:53.164390.164390 lmp.py:904] ---- decode step 1 layer 21 ----
DEBUG 05-06 10:38:53.166256.166256 cuda_h.py:27] end *sagl cost 1.881 ms
DEBUG 05-06 10:38:53.169173.169173 cuda_h.py:27] end *layer_moe_fused cost 1.718 ms
DEBUG 05-06 10:38:53.170871.170871 cuda_h.py:27] end decode_layer cost 5.266 ms
DEBUG 05-06 10:38:53.170821.170821 lmp.py:904] ---- decode step 1 layer 22 ----
DEBUG 05-06 10:38:53.172944.172944 cuda_h.py:27] end *sagl cost 1.860 ms
DEBUG 05-06 10:38:53.174388.174388 cuda_h.py:27] end *layer_moe_fused cost 1.533 ms
DEBUG 05-06 10:38:53.175848.175848 cuda_h.py:27] end decode_layer cost 4.966 ms
DEBUG 05-06 10:38:53.175559.175559 lmp.py:904] ---- decode step 1 layer 23 ----
DEBUG 05-06 10:38:53.177578.177578 cuda_h.py:27] end *sagl cost 1.889 ms
DEBUG 05-06 10:38:53.179533.179533 cuda_h.py:27] end *layer_moe_fused cost 1.568 ms
DEBUG 05-06 10:38:53.180947.180947 cuda_h.py:27] end decode_layer cost 5.057 ms
DEBUG 05-06 10:38:53.180373.180373 lmp.py:904] ---- decode step 1 layer 24 ----
DEBUG 05-06 10:38:53.182284.182284 cuda_h.py:27] end *sagl cost 1.843 ms
DEBUG 05-06 10:38:53.184960.184960 cuda_h.py:27] end *layer_moe_fused cost 1.528 ms
DEBUG 05-06 10:38:53.185473.185473 cuda_h.py:27] end decode_layer cost 4.949 ms
DEBUG 05-06 10:38:53.185376.185376 lmp.py:904] ---- decode step 1 layer 25 ----
DEBUG 05-06 10:38:53.187480.187480 cuda_h.py:27] end *sagl cost 1.882 ms
DEBUG 05-06 10:38:53.189072.189072 cuda_h.py:27] end *layer_moe_fused cost 1.547 ms
DEBUG 05-06 10:38:53.190777.190777 cuda_h.py:27] end decode_layer cost 5.066 ms
DEBUG 05-06 10:38:53.190203.190203 lmp.py:904] ---- decode step 1 layer 26 ----
DEBUG 05-06 10:38:53.192203.192203 cuda_h.py:27] end *sagl cost 1.910 ms
DEBUG 05-06 10:38:53.194468.194468 cuda_h.py:27] end *layer_moe_fused cost 1.540 ms
DEBUG 05-06 10:38:53.195444.195444 cuda_h.py:27] end decode_layer cost 5.018 ms
DEBUG 05-06 10:38:53.195202.195202 lmp.py:904] ---- decode step 1 layer 27 ----
DEBUG 05-06 10:38:53.197135.197135 cuda_h.py:27] end *sagl cost 1.896 ms
DEBUG 05-06 10:38:53.200917.200917 cuda_h.py:27] end *layer_moe_fused cost 1.548 ms
DEBUG 05-06 10:38:53.200609.200609 cuda_h.py:27] end decode_layer cost 5.035 ms
DEBUG 05-06 10:38:53.200128.200128 lmp.py:904] ---- decode step 1 layer 28 ----
DEBUG 05-06 10:38:53.202595.202595 cuda_h.py:27] end *sagl cost 1.835 ms
DEBUG 05-06 10:38:53.205920.205920 cuda_h.py:27] end *layer_moe_fused cost 1.557 ms
DEBUG 05-06 10:38:53.205135.205135 cuda_h.py:27] end decode_layer cost 4.951 ms
DEBUG 05-06 10:38:53.205700.205700 lmp.py:904] ---- decode step 1 layer 29 ----
DEBUG 05-06 10:38:53.207875.207875 cuda_h.py:27] end *sagl cost 1.830 ms
DEBUG 05-06 10:38:53.210163.210163 cuda_h.py:27] end *layer_moe_fused cost 1.537 ms
DEBUG 05-06 10:38:53.210570.210570 cuda_h.py:27] end decode_layer cost 5.026 ms
DEBUG 05-06 10:38:53.210566.210566 cuda_h.py:27] end decode_step cost 168.457 ms
INFO 05-06 10:38:53.210282.210282 lmp.py:931] decode step 1 time: 0.16849613189697266 seconds
Time taken: 7.618557788431644 seconds
generate input ids cost 0.04189014434814453 s
DEBUG 05-06 10:38:55.989110.989110 cuda_h.py:27] end generate_input_ids cost 2627.459 ms
DEBUG 05-06 10:38:55.989454.989454 cuda_h.py:27] end init_cache cost 0.034 ms
INFO 05-06 10:38:56.002048.002048 lmp.py:2341] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 5837135812, 'cuda:1': 12823166976, 'cuda:2': 12823166976, 'cuda:3': 12823166976} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7518523832377548, 'cuda:1': 0.4710385881818624, 'cuda:2': 0.4710385881818624, 'cuda:3': 0.4710385881818624}
INFO 05-06 10:38:56.002238.002238 lmp.py:2359] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.002021.002021 lmp.py:2359] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.002267.002267 lmp.py:2359] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.002129.002129 lmp.py:2359] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.003245.003245 lmp.py:2359] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.003060.003060 lmp.py:2359] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.003318.003318 lmp.py:2359] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.003658.003658 lmp.py:2359] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.003507.003507 lmp.py:2359] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.003561.003561 lmp.py:2359] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.003296.003296 lmp.py:2359] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.003920.003920 lmp.py:2359] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.004702.004702 lmp.py:2359] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.004803.004803 lmp.py:2359] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.004287.004287 lmp.py:2359] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.004149.004149 lmp.py:2359] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.004738.004738 lmp.py:2359] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.004600.004600 lmp.py:2359] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.004018.004018 lmp.py:2359] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.004973.004973 lmp.py:2359] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005224.005224 lmp.py:2359] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005755.005755 lmp.py:2359] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005768.005768 lmp.py:2359] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005246.005246 lmp.py:2359] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005019.005019 lmp.py:2359] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005928.005928 lmp.py:2359] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005694.005694 lmp.py:2359] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005603.005603 lmp.py:2359] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005615.005615 lmp.py:2359] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:38:56.005239.005239 lmp.py:2359] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 10:38:56.306723.306723 cuda_h.py:27] end init_loading_placement cost 317.113 ms
DEBUG 05-06 10:38:56.306418.306418 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:38:56.306619.306619 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:38:56 client.py:72] load_into_gpu: gemma4-26B-A4B, 15c0f03d-5765-4ed3-b4e6-99fc1e97146c
INFO 05-06 10:38:56 client.py:135] Model loaded: gemma4-26B-A4B, 15c0f03d-5765-4ed3-b4e6-99fc1e97146c
INFO 05-06 10:38:56 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 15c0f03d-5765-4ed3-b4e6-99fc1e97146c
INFO 05-06 10:38:56 client.py:212] Model loaded
DEBUG 05-06 10:38:56.834544.834544 cuda_h.py:27] end init_general_sagl_loading_async cost 527.886 ms
INFO 05-06 10:38:56.882590.882590 lmp.py:2862] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 10:38:56.985395.985395 cuda_h.py:27] end restore_state_dict cost 102.417 ms
DEBUG 05-06 10:38:56.985122.985122 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:38:56.985911.985911 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:38:56 client.py:72] load_into_gpu: gemma4-26B-A4B, c4fd3acc-6845-4c54-8b2e-ae29e0e2ea6a
INFO 05-06 10:38:57 client.py:135] Model loaded: gemma4-26B-A4B, c4fd3acc-6845-4c54-8b2e-ae29e0e2ea6a
DEBUG 05-06 10:38:57.109658.109658 cuda_h.py:27] end init_experts_loading_async cost 124.019 ms
DEBUG 05-06 10:38:57.110743.110743 cuda_h.py:27] end init_inputs_tokens cost 0.709 ms
DEBUG 05-06 10:38:57.110328.110328 lmp.py:824] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 10:38:57.113102.113102 cuda_h.py:27] end *sagl cost 3.077 ms
experts_cpu_alloc {'expert_ids': [11, 19, 27, 87, 63, 111, 119, 79, 23, 59, 107, 71, 123, 99, 75, 115, 83, 127, 31, 3, 67, 51, 55, 7, 91, 39, 103, 47, 100, 4, 36, 84, 8, 20, 44, 80, 108, 60, 24, 28, 76, 92, 116, 112, 64, 72, 48, 52, 32, 104, 16, 0, 68, 124, 101, 109, 85, 49, 45, 65, 93, 69, 5, 9, 13, 73, 77, 37, 89, 25, 105, 125, 117, 41, 113, 21, 121, 1, 53, 33, 86, 94, 66, 14, 106, 2, 10, 34, 114, 38, 102, 18, 70, 110, 118, 122, 78, 26, 54, 74, 22, 50, 126, 46, 90], 'token_total': 4096, 'token_per_expert': {11: 1, 19: 1, 27: 1, 87: 1, 63: 3, 111: 3, 119: 5, 79: 8, 23: 9, 59: 9, 107: 9, 71: 15, 123: 18, 99: 26, 75: 29, 115: 29, 83: 33, 127: 33, 31: 34, 3: 46, 67: 47, 51: 48, 55: 51, 7: 94, 91: 99, 39: 176, 103: 178, 47: 318, 100: 1, 4: 2, 36: 2, 84: 2, 8: 4, 20: 4, 44: 7, 80: 9, 108: 10, 60: 12, 24: 16, 28: 16, 76: 16, 92: 16, 116: 18, 112: 23, 64: 27, 72: 35, 48: 41, 52: 42, 32: 43, 104: 43, 16: 48, 0: 73, 68: 170, 124: 178, 101: 1, 109: 1, 85: 2, 49: 3, 45: 4, 65: 5, 93: 5, 69: 9, 5: 16, 9: 17, 13: 17, 73: 19, 77: 19, 37: 20, 89: 20, 25: 24, 105: 24, 125: 25, 117: 26, 41: 27, 113: 40, 21: 48, 121: 65, 1: 75, 53: 205, 33: 210, 86: 1, 94: 1, 66: 2, 14: 4, 106: 6, 2: 8, 10: 8, 34: 9, 114: 9, 38: 13, 102: 14, 18: 18, 70: 26, 110: 27, 118: 29, 122: 35, 78: 36, 26: 59, 54: 59, 74: 61, 22: 64, 50: 110, 126: 115, 46: 119, 90: 154}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.116114.116114 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.768ms | allocate_experts_across_cpu_gpu: 0.297ms
INFO 05-06 10:38:57.116813.116813 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.0994415283203125e-06 seconds
INFO 05-06 10:38:57.116059.116059 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005939006805419922 seconds
INFO 05-06 10:38:57.117583.117583 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.000316619873046875 seconds
INFO 05-06 10:38:57.117654.117654 lmp.py:1484] [layer_moe_fused] experts compute time: 2.002716064453125e-05 seconds
INFO 05-06 10:38:57.160938.160938 lmp.py:1496] [layer_moe_fused] to time: 0.00017452239990234375 seconds
INFO 05-06 10:38:57.160604.160604 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.042160749435424805 seconds
DEBUG 05-06 10:38:57.161869.161869 cuda_h.py:27] end *layer_moe_fused cost 46.148 ms
DEBUG 05-06 10:38:57.161381.161381 cuda_h.py:27] end prefill_layer cost 51.397 ms
DEBUG 05-06 10:38:57.161205.161205 lmp.py:841] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 10:38:57.162067.162067 lmp.py:824] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 10:38:57.164033.164033 cuda_h.py:27] end *sagl cost 2.322 ms
experts_cpu_alloc {'expert_ids': [15, 39, 103, 31, 123, 55, 83, 43, 91, 87, 115, 11, 35, 59, 95, 119, 27, 79, 47, 51, 127, 99, 67, 3, 7, 16, 40, 60, 72, 88, 84, 44, 108, 32, 48, 116, 112, 56, 64, 76, 92, 104, 124, 80, 96, 120, 100, 12, 20, 28, 8, 4, 0, 68, 52, 41, 61, 117, 45, 77, 33, 57, 89, 101, 125, 37, 93, 29, 121, 81, 69, 85, 21, 105, 9, 49, 65, 53, 73, 25, 109, 97, 1, 5, 13, 18, 38, 110, 62, 26, 90, 50, 66, 78, 34, 74, 14, 98, 94, 106, 42, 54, 46, 118, 122, 22, 10, 82, 6, 2, 30], 'token_total': 4096, 'token_per_expert': {15: 1, 39: 1, 103: 1, 31: 2, 123: 2, 55: 4, 83: 5, 43: 8, 91: 8, 87: 10, 115: 12, 11: 15, 35: 17, 59: 18, 95: 18, 119: 19, 27: 23, 79: 30, 47: 40, 51: 45, 127: 48, 99: 72, 67: 108, 3: 150, 7: 182, 16: 2, 40: 2, 60: 2, 72: 2, 88: 2, 84: 4, 44: 6, 108: 7, 32: 8, 48: 8, 116: 9, 112: 11, 56: 12, 64: 15, 76: 15, 92: 18, 104: 19, 124: 27, 80: 30, 96: 33, 120: 40, 100: 41, 12: 43, 20: 61, 28: 75, 8: 87, 4: 132, 0: 136, 68: 141, 52: 184, 41: 1, 61: 1, 117: 1, 45: 2, 77: 2, 33: 3, 57: 3, 89: 5, 101: 5, 125: 5, 37: 6, 93: 6, 29: 7, 121: 7, 81: 8, 69: 9, 85: 12, 21: 13, 105: 13, 9: 14, 49: 24, 65: 25, 53: 28, 73: 28, 25: 43, 109: 56, 97: 92, 1: 141, 5: 209, 13: 219, 18: 3, 38: 3, 110: 3, 62: 4, 26: 7, 90: 8, 50: 10, 66: 12, 78: 12, 34: 14, 74: 14, 14: 18, 98: 20, 94: 21, 106: 29, 42: 36, 54: 42, 46: 43, 118: 51, 122: 55, 22: 69, 10: 101, 82: 111, 6: 131, 2: 133, 30: 147}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.166598.166598 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.592ms | allocate_experts_across_cpu_gpu: 0.405ms
INFO 05-06 10:38:57.166588.166588 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:57.167803.167803 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005824565887451172 seconds
INFO 05-06 10:38:57.169923.169923 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001077413558959961 seconds
INFO 05-06 10:38:57.169026.169026 lmp.py:1484] [layer_moe_fused] experts compute time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:57.204551.204551 lmp.py:1496] [layer_moe_fused] to time: 0.00017523765563964844 seconds
INFO 05-06 10:38:57.204574.204574 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.034906864166259766 seconds
DEBUG 05-06 10:38:57.205349.205349 cuda_h.py:27] end *layer_moe_fused cost 39.436 ms
DEBUG 05-06 10:38:57.205482.205482 cuda_h.py:27] end prefill_layer cost 43.835 ms
DEBUG 05-06 10:38:57.205544.205544 lmp.py:841] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 10:38:57.205883.205883 lmp.py:824] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 10:38:57.208479.208479 cuda_h.py:27] end *sagl cost 2.348 ms
experts_cpu_alloc {'expert_ids': [67, 27, 99, 111, 23, 123, 35, 95, 115, 31, 43, 51, 63, 71, 107, 83, 119, 91, 103, 55, 19, 15, 127, 11, 59, 3, 7, 40, 120, 116, 72, 44, 56, 64, 96, 36, 52, 88, 100, 80, 124, 8, 28, 60, 24, 48, 20, 104, 84, 76, 108, 4, 0, 45, 69, 21, 61, 25, 121, 105, 113, 17, 85, 77, 33, 49, 57, 53, 65, 97, 109, 81, 125, 37, 9, 29, 13, 41, 5, 1, 26, 38, 66, 42, 82, 50, 114, 70, 126, 98, 46, 58, 14, 78, 110, 122, 34, 106, 118, 18, 90, 102, 54, 62, 2, 6], 'token_total': 4096, 'token_per_expert': {67: 1, 27: 2, 99: 3, 111: 3, 23: 8, 123: 9, 35: 12, 95: 13, 115: 13, 31: 15, 43: 15, 51: 15, 63: 15, 71: 15, 107: 19, 83: 21, 119: 22, 91: 31, 103: 32, 55: 59, 19: 83, 15: 85, 127: 89, 11: 114, 59: 114, 3: 151, 7: 205, 40: 3, 120: 3, 116: 6, 72: 8, 44: 11, 56: 11, 64: 11, 96: 14, 36: 15, 52: 15, 88: 15, 100: 16, 80: 19, 124: 19, 8: 22, 28: 27, 60: 33, 24: 34, 48: 35, 20: 37, 104: 38, 84: 52, 76: 63, 108: 125, 4: 150, 0: 151, 45: 1, 69: 4, 21: 5, 61: 5, 25: 6, 121: 7, 105: 8, 113: 13, 17: 15, 85: 15, 77: 17, 33: 18, 49: 19, 57: 20, 53: 21, 65: 25, 97: 25, 109: 31, 81: 48, 125: 57, 37: 58, 9: 68, 29: 69, 13: 71, 41: 99, 5: 137, 1: 209, 26: 1, 38: 1, 66: 2, 42: 4, 82: 6, 50: 7, 114: 11, 70: 12, 126: 12, 98: 15, 46: 17, 58: 19, 14: 23, 78: 23, 110: 23, 122: 25, 34: 28, 106: 28, 118: 47, 18: 48, 90: 53, 102: 67, 54: 87, 62: 111, 2: 128, 6: 130}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.210044.210044 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.592ms | allocate_experts_across_cpu_gpu: 0.397ms
INFO 05-06 10:38:57.210948.210948 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.0994415283203125e-06 seconds
INFO 05-06 10:38:57.211195.211195 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005888938903808594 seconds
INFO 05-06 10:38:57.212760.212760 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009605884552001953 seconds
INFO 05-06 10:38:57.213334.213334 lmp.py:1484] [layer_moe_fused] experts compute time: 4.0531158447265625e-06 seconds
INFO 05-06 10:38:57.248532.248532 lmp.py:1496] [layer_moe_fused] to time: 0.00017976760864257812 seconds
INFO 05-06 10:38:57.249489.249489 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.035829782485961914 seconds
DEBUG 05-06 10:38:57.250696.250696 cuda_h.py:27] end *layer_moe_fused cost 40.234 ms
DEBUG 05-06 10:38:57.250718.250718 cuda_h.py:27] end prefill_layer cost 44.704 ms
DEBUG 05-06 10:38:57.250695.250695 lmp.py:841] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 10:38:57.250034.250034 lmp.py:824] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 10:38:57.253920.253920 cuda_h.py:27] end *sagl cost 2.336 ms
experts_cpu_alloc {'expert_ids': [23, 87, 35, 55, 63, 43, 91, 127, 27, 67, 31, 123, 19, 39, 59, 15, 111, 119, 107, 51, 83, 11, 95, 71, 75, 7, 3, 32, 72, 20, 36, 80, 100, 60, 40, 48, 8, 16, 24, 116, 56, 64, 44, 76, 96, 108, 88, 120, 68, 104, 84, 52, 92, 28, 0, 4, 21, 29, 89, 41, 65, 117, 57, 33, 13, 101, 61, 77, 73, 109, 17, 69, 25, 93, 53, 121, 9, 97, 85, 1, 5, 38, 46, 94, 82, 106, 98, 30, 42, 18, 86, 74, 110, 54, 26, 114, 70, 118, 122, 34, 58, 10, 14, 78, 62, 22, 102, 66, 2, 50, 6], 'token_total': 4096, 'token_per_expert': {23: 1, 87: 1, 35: 3, 55: 4, 63: 4, 43: 6, 91: 7, 127: 7, 27: 9, 67: 10, 31: 12, 123: 18, 19: 20, 39: 23, 59: 24, 15: 31, 111: 31, 119: 31, 107: 33, 51: 36, 83: 39, 11: 40, 95: 41, 71: 59, 75: 83, 7: 128, 3: 158, 32: 2, 72: 2, 20: 3, 36: 3, 80: 5, 100: 5, 60: 6, 40: 8, 48: 8, 8: 9, 16: 9, 24: 10, 116: 10, 56: 12, 64: 12, 44: 24, 76: 26, 96: 33, 108: 36, 88: 37, 120: 37, 68: 38, 104: 40, 84: 56, 52: 59, 92: 60, 28: 113, 0: 157, 4: 176, 21: 1, 29: 2, 89: 3, 41: 4, 65: 4, 117: 4, 57: 6, 33: 7, 13: 8, 101: 8, 61: 13, 77: 18, 73: 26, 109: 31, 17: 38, 69: 38, 25: 43, 93: 47, 53: 49, 121: 49, 9: 57, 97: 60, 85: 106, 1: 138, 5: 165, 38: 1, 46: 1, 94: 1, 82: 2, 106: 2, 98: 3, 30: 4, 42: 5, 18: 8, 86: 11, 74: 12, 110: 13, 54: 15, 26: 16, 114: 31, 70: 34, 118: 38, 122: 43, 34: 44, 58: 44, 10: 50, 14: 65, 78: 72, 62: 84, 22: 85, 102: 91, 66: 110, 2: 141, 50: 142, 6: 148}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.255536.255536 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.638ms | allocate_experts_across_cpu_gpu: 0.474ms
INFO 05-06 10:38:57.255757.255757 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:57.256638.256638 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005939006805419922 seconds
INFO 05-06 10:38:57.258297.258297 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0015625953674316406 seconds
INFO 05-06 10:38:57.259126.259126 lmp.py:1484] [layer_moe_fused] experts compute time: 1.430511474609375e-06 seconds
INFO 05-06 10:38:57.296469.296469 lmp.py:1496] [layer_moe_fused] to time: 0.00017213821411132812 seconds
INFO 05-06 10:38:57.296797.296797 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03716564178466797 seconds
DEBUG 05-06 10:38:57.297413.297413 cuda_h.py:27] end *layer_moe_fused cost 42.947 ms
DEBUG 05-06 10:38:57.298092.298092 cuda_h.py:27] end prefill_layer cost 47.395 ms
DEBUG 05-06 10:38:57.298121.298121 lmp.py:841] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 10:38:57.298606.298606 lmp.py:824] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 10:38:57.300621.300621 cuda_h.py:27] end *sagl cost 2.353 ms
experts_cpu_alloc {'expert_ids': [31, 79, 103, 91, 107, 75, 19, 123, 15, 39, 71, 47, 87, 51, 67, 55, 83, 27, 115, 111, 23, 43, 59, 119, 63, 7, 3, 12, 120, 56, 44, 80, 108, 40, 64, 84, 36, 88, 52, 96, 116, 28, 104, 92, 20, 124, 32, 60, 76, 24, 8, 0, 4, 69, 121, 21, 37, 45, 73, 109, 81, 97, 101, 77, 117, 17, 25, 57, 61, 105, 53, 49, 93, 113, 85, 125, 29, 89, 5, 1, 58, 66, 70, 114, 18, 46, 122, 30, 94, 126, 34, 38, 90, 118, 86, 78, 62, 98, 82, 54, 26, 22, 106, 74, 2, 6], 'token_total': 4096, 'token_per_expert': {31: 1, 79: 1, 103: 2, 91: 9, 107: 9, 75: 11, 19: 13, 123: 13, 15: 14, 39: 16, 71: 16, 47: 18, 87: 18, 51: 30, 67: 34, 55: 37, 83: 37, 27: 41, 115: 41, 111: 53, 23: 67, 43: 70, 59: 75, 119: 89, 63: 139, 7: 257, 3: 273, 12: 1, 120: 2, 56: 3, 44: 4, 80: 6, 108: 6, 40: 7, 64: 7, 84: 8, 36: 10, 88: 11, 52: 13, 96: 15, 116: 15, 28: 17, 104: 17, 92: 18, 20: 20, 124: 20, 32: 21, 60: 22, 76: 22, 24: 67, 8: 91, 0: 256, 4: 270, 69: 2, 121: 2, 21: 3, 37: 4, 45: 4, 73: 4, 109: 4, 81: 5, 97: 5, 101: 5, 77: 7, 117: 8, 17: 9, 25: 9, 57: 9, 61: 9, 105: 15, 53: 21, 49: 23, 93: 26, 113: 27, 85: 28, 125: 30, 29: 33, 89: 49, 5: 269, 1: 303, 58: 1, 66: 1, 70: 1, 114: 1, 18: 2, 46: 2, 122: 3, 30: 4, 94: 5, 126: 5, 34: 6, 38: 6, 90: 7, 118: 9, 86: 11, 78: 12, 62: 13, 98: 14, 82: 21, 54: 30, 26: 32, 22: 33, 106: 49, 74: 66, 2: 256, 6: 260}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.303142.303142 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.642ms | allocate_experts_across_cpu_gpu: 0.406ms
INFO 05-06 10:38:57.303787.303787 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.337860107421875e-06 seconds
INFO 05-06 10:38:57.304696.304696 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005588531494140625 seconds
INFO 05-06 10:38:57.306392.306392 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017442703247070312 seconds
INFO 05-06 10:38:57.306285.306285 lmp.py:1484] [layer_moe_fused] experts compute time: 4.0531158447265625e-06 seconds
INFO 05-06 10:38:57.346337.346337 lmp.py:1496] [layer_moe_fused] to time: 0.00017142295837402344 seconds
INFO 05-06 10:38:57.347851.347851 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04057765007019043 seconds
DEBUG 05-06 10:38:57.348760.348760 cuda_h.py:27] end *layer_moe_fused cost 45.815 ms
DEBUG 05-06 10:38:57.348729.348729 cuda_h.py:27] end prefill_layer cost 50.326 ms
DEBUG 05-06 10:38:57.348076.348076 lmp.py:841] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 10:38:57.348097.348097 lmp.py:824] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 10:38:57.352944.352944 cuda_h.py:27] end *sagl cost 4.086 ms
experts_cpu_alloc {'expert_ids': [15, 19, 27, 51, 115, 119, 31, 75, 107, 67, 83, 79, 55, 63, 43, 23, 99, 123, 111, 87, 127, 39, 71, 3, 7, 32, 92, 124, 8, 56, 52, 68, 84, 44, 96, 100, 120, 60, 80, 104, 88, 116, 76, 72, 36, 24, 64, 28, 16, 20, 112, 0, 4, 17, 21, 77, 121, 57, 105, 37, 113, 53, 125, 73, 29, 61, 93, 117, 9, 13, 33, 49, 101, 1, 5, 30, 38, 78, 82, 50, 58, 54, 102, 26, 86, 114, 34, 98, 106, 62, 14, 18, 118, 74, 46, 70, 126, 94, 22, 42, 6, 2], 'token_total': 4096, 'token_per_expert': {15: 1, 19: 2, 27: 2, 51: 3, 115: 5, 119: 6, 31: 7, 75: 7, 107: 7, 67: 8, 83: 8, 79: 11, 55: 14, 63: 16, 43: 21, 23: 22, 99: 23, 123: 27, 111: 28, 87: 32, 127: 49, 39: 73, 71: 130, 3: 256, 7: 265, 32: 1, 92: 1, 124: 1, 8: 2, 56: 2, 52: 6, 68: 6, 84: 6, 44: 9, 96: 14, 100: 17, 120: 17, 60: 20, 80: 22, 104: 23, 88: 24, 116: 24, 76: 26, 72: 29, 36: 39, 24: 42, 64: 43, 28: 46, 16: 65, 20: 73, 112: 84, 0: 266, 4: 298, 17: 1, 21: 1, 77: 1, 121: 1, 57: 2, 105: 2, 37: 4, 113: 4, 53: 5, 125: 13, 73: 17, 29: 18, 61: 22, 93: 28, 117: 34, 9: 36, 13: 46, 33: 56, 49: 98, 101: 128, 1: 257, 5: 282, 30: 1, 38: 1, 78: 1, 82: 1, 50: 2, 58: 2, 54: 3, 102: 3, 26: 4, 86: 4, 114: 4, 34: 5, 98: 5, 106: 5, 62: 8, 14: 10, 18: 10, 118: 10, 74: 15, 46: 16, 70: 18, 126: 25, 94: 28, 22: 29, 42: 30, 6: 261, 2: 310}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 22, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.355898.355898 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.591ms | allocate_experts_across_cpu_gpu: 0.382ms
INFO 05-06 10:38:57.355404.355404 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:57.355303.355303 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.000545501708984375 seconds
INFO 05-06 10:38:57.356733.356733 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0007302761077880859 seconds
INFO 05-06 10:38:57.357154.357154 lmp.py:1484] [layer_moe_fused] experts compute time: 1.1920928955078125e-06 seconds
INFO 05-06 10:38:57.399167.399167 lmp.py:1496] [layer_moe_fused] to time: 0.0001804828643798828 seconds
INFO 05-06 10:38:57.400481.400481 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.042966604232788086 seconds
DEBUG 05-06 10:38:57.400530.400530 cuda_h.py:27] end *layer_moe_fused cost 46.734 ms
DEBUG 05-06 10:38:57.401591.401591 cuda_h.py:27] end prefill_layer cost 52.708 ms
DEBUG 05-06 10:38:57.401415.401415 lmp.py:841] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 10:38:57.401515.401515 lmp.py:824] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 10:38:57.404535.404535 cuda_h.py:27] end *sagl cost 2.326 ms
experts_cpu_alloc {'expert_ids': [31, 47, 83, 59, 111, 67, 15, 127, 11, 91, 43, 51, 103, 71, 75, 27, 95, 123, 107, 79, 119, 35, 87, 23, 115, 99, 3, 7, 52, 92, 124, 16, 20, 40, 72, 80, 120, 60, 76, 56, 116, 32, 104, 24, 28, 44, 96, 36, 64, 108, 68, 4, 0, 21, 33, 97, 101, 109, 81, 49, 89, 73, 125, 37, 85, 105, 41, 57, 77, 113, 117, 69, 121, 9, 13, 53, 65, 93, 25, 5, 1, 22, 82, 114, 38, 126, 110, 30, 18, 42, 74, 14, 70, 10, 58, 50, 62, 94, 78, 46, 122, 98, 26, 34, 90, 102, 86, 106, 6, 2], 'token_total': 4096, 'token_per_expert': {31: 1, 47: 1, 83: 1, 59: 2, 111: 2, 67: 3, 15: 4, 127: 6, 11: 7, 91: 7, 43: 8, 51: 11, 103: 11, 71: 14, 75: 14, 27: 15, 95: 15, 123: 15, 107: 18, 79: 19, 119: 35, 35: 41, 87: 43, 23: 47, 115: 52, 99: 124, 3: 257, 7: 257, 52: 1, 92: 1, 124: 1, 16: 2, 20: 2, 40: 2, 72: 2, 80: 4, 120: 4, 60: 5, 76: 5, 56: 10, 116: 10, 32: 15, 104: 15, 24: 16, 28: 16, 44: 20, 96: 21, 36: 23, 64: 45, 108: 66, 68: 142, 4: 259, 0: 264, 21: 1, 33: 1, 97: 1, 101: 1, 109: 1, 81: 2, 49: 4, 89: 4, 73: 5, 125: 8, 37: 9, 85: 9, 105: 9, 41: 11, 57: 11, 77: 12, 113: 12, 117: 17, 69: 19, 121: 26, 9: 27, 13: 35, 53: 41, 65: 52, 93: 77, 25: 110, 5: 267, 1: 272, 22: 1, 82: 1, 114: 1, 38: 3, 126: 4, 110: 5, 30: 6, 18: 7, 42: 7, 74: 7, 14: 9, 70: 11, 10: 12, 58: 14, 50: 22, 62: 22, 94: 22, 78: 24, 46: 25, 122: 25, 98: 34, 26: 35, 34: 36, 90: 39, 102: 41, 86: 54, 106: 67, 6: 266, 2: 271}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.406122.406122 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.621ms | allocate_experts_across_cpu_gpu: 0.399ms
INFO 05-06 10:38:57.406311.406311 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:57.407267.407267 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005400180816650391 seconds
INFO 05-06 10:38:57.408114.408114 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008227825164794922 seconds
INFO 05-06 10:38:57.408528.408528 lmp.py:1484] [layer_moe_fused] experts compute time: 4.76837158203125e-06 seconds
INFO 05-06 10:38:57.451880.451880 lmp.py:1496] [layer_moe_fused] to time: 0.0002129077911376953 seconds
INFO 05-06 10:38:57.452507.452507 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04271697998046875 seconds
DEBUG 05-06 10:38:57.452276.452276 cuda_h.py:27] end *layer_moe_fused cost 47.341 ms
DEBUG 05-06 10:38:57.453942.453942 cuda_h.py:27] end prefill_layer cost 51.877 ms
DEBUG 05-06 10:38:57.453045.453045 lmp.py:841] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 10:38:57.453483.453483 lmp.py:824] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 10:38:57.456753.456753 cuda_h.py:27] end *sagl cost 2.818 ms
experts_cpu_alloc {'expert_ids': [11, 27, 35, 67, 119, 15, 55, 107, 127, 63, 51, 95, 23, 111, 115, 87, 99, 83, 123, 19, 43, 47, 59, 103, 71, 79, 91, 3, 7, 36, 32, 88, 92, 8, 116, 112, 16, 64, 68, 72, 80, 20, 48, 104, 60, 96, 52, 56, 44, 120, 28, 108, 84, 12, 0, 4, 73, 77, 109, 37, 41, 101, 125, 17, 21, 117, 9, 45, 25, 13, 61, 105, 57, 113, 53, 33, 85, 69, 29, 65, 121, 97, 1, 5, 26, 30, 62, 94, 102, 38, 50, 66, 82, 54, 126, 78, 98, 122, 118, 106, 18, 22, 86, 114, 14, 42, 34, 110, 90, 10, 70, 2, 6], 'token_total': 4096, 'token_per_expert': {11: 1, 27: 1, 35: 1, 67: 1, 119: 1, 15: 2, 55: 4, 107: 4, 127: 4, 63: 6, 51: 7, 95: 7, 23: 8, 111: 9, 115: 9, 87: 11, 99: 12, 83: 13, 123: 13, 19: 14, 43: 14, 47: 14, 59: 16, 103: 21, 71: 26, 79: 31, 91: 91, 3: 256, 7: 270, 36: 1, 32: 3, 88: 4, 92: 4, 8: 6, 116: 6, 112: 8, 16: 9, 64: 10, 68: 12, 72: 16, 80: 16, 20: 22, 48: 22, 104: 22, 60: 26, 96: 27, 52: 31, 56: 31, 44: 34, 120: 38, 28: 39, 108: 45, 84: 49, 12: 76, 0: 258, 4: 290, 73: 1, 77: 2, 109: 2, 37: 3, 41: 4, 101: 4, 125: 6, 17: 7, 21: 7, 117: 7, 9: 9, 45: 10, 25: 12, 13: 15, 61: 17, 105: 21, 57: 25, 113: 26, 53: 32, 33: 33, 85: 40, 69: 41, 29: 43, 65: 53, 121: 64, 97: 123, 1: 256, 5: 271, 26: 2, 30: 2, 62: 2, 94: 2, 102: 2, 38: 3, 50: 3, 66: 3, 82: 4, 54: 5, 126: 5, 78: 7, 98: 7, 122: 7, 118: 8, 106: 14, 18: 15, 22: 20, 86: 28, 114: 32, 14: 34, 42: 37, 34: 39, 110: 40, 90: 41, 10: 48, 70: 54, 2: 256, 6: 270}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.459374.459374 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.619ms | allocate_experts_across_cpu_gpu: 0.408ms
INFO 05-06 10:38:57.459079.459079 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:57.460643.460643 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.000576019287109375 seconds
INFO 05-06 10:38:57.461575.461575 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008785724639892578 seconds
INFO 05-06 10:38:57.461585.461585 lmp.py:1484] [layer_moe_fused] experts compute time: 1.1920928955078125e-06 seconds
INFO 05-06 10:38:57.504051.504051 lmp.py:1496] [layer_moe_fused] to time: 0.00017452239990234375 seconds
INFO 05-06 10:38:57.504810.504810 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04272198677062988 seconds
DEBUG 05-06 10:38:57.505699.505699 cuda_h.py:27] end *layer_moe_fused cost 47.124 ms
DEBUG 05-06 10:38:57.505216.505216 cuda_h.py:27] end prefill_layer cost 52.147 ms
DEBUG 05-06 10:38:57.505510.505510 lmp.py:841] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 10:38:57.505611.505611 lmp.py:824] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 10:38:57.508926.508926 cuda_h.py:27] end *sagl cost 2.305 ms
experts_cpu_alloc {'expert_ids': [23, 35, 91, 43, 119, 127, 31, 63, 99, 47, 55, 11, 27, 111, 123, 15, 71, 75, 19, 87, 51, 103, 7, 3, 8, 48, 96, 104, 72, 84, 68, 92, 64, 116, 124, 20, 44, 108, 16, 36, 76, 52, 12, 80, 56, 32, 120, 28, 0, 4, 25, 101, 13, 117, 33, 89, 37, 85, 17, 29, 49, 57, 21, 41, 113, 45, 93, 61, 69, 77, 81, 53, 65, 121, 105, 125, 73, 1, 5, 26, 18, 74, 90, 106, 118, 34, 82, 86, 22, 62, 66, 42, 10, 14, 98, 126, 122, 102, 46, 38, 70, 114, 50, 110, 54, 58, 6, 2], 'token_total': 4096, 'token_per_expert': {23: 1, 35: 2, 91: 2, 43: 4, 119: 5, 127: 7, 31: 9, 63: 9, 99: 10, 47: 11, 55: 11, 11: 14, 27: 14, 111: 19, 123: 25, 15: 29, 71: 44, 75: 44, 19: 45, 87: 59, 51: 88, 103: 96, 7: 256, 3: 281, 8: 1, 48: 1, 96: 2, 104: 2, 72: 3, 84: 3, 68: 5, 92: 5, 64: 6, 116: 6, 124: 6, 20: 8, 44: 8, 108: 8, 16: 11, 36: 16, 76: 16, 52: 17, 12: 23, 80: 23, 56: 34, 32: 40, 120: 61, 28: 67, 0: 257, 4: 267, 25: 2, 101: 2, 13: 3, 117: 3, 33: 4, 89: 4, 37: 5, 85: 7, 17: 8, 29: 8, 49: 8, 57: 8, 21: 9, 41: 9, 113: 11, 45: 14, 93: 14, 61: 15, 69: 15, 77: 17, 81: 19, 53: 21, 65: 23, 121: 30, 105: 37, 125: 39, 73: 56, 1: 258, 5: 272, 26: 1, 18: 2, 74: 2, 90: 2, 106: 2, 118: 2, 34: 3, 82: 4, 86: 5, 22: 6, 62: 6, 66: 7, 42: 8, 10: 14, 14: 14, 98: 14, 126: 14, 122: 26, 102: 29, 46: 30, 38: 33, 70: 36, 114: 45, 50: 49, 110: 64, 54: 103, 58: 113, 6: 270, 2: 290}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.510566.510566 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.633ms | allocate_experts_across_cpu_gpu: 0.397ms
INFO 05-06 10:38:57.510324.510324 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.0994415283203125e-06 seconds
INFO 05-06 10:38:57.511688.511688 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005617141723632812 seconds
INFO 05-06 10:38:57.513722.513722 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009684562683105469 seconds
INFO 05-06 10:38:57.513341.513341 lmp.py:1484] [layer_moe_fused] experts compute time: 1.9073486328125e-06 seconds
INFO 05-06 10:38:57.555516.555516 lmp.py:1496] [layer_moe_fused] to time: 0.00018787384033203125 seconds
INFO 05-06 10:38:57.556315.556315 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04256391525268555 seconds
DEBUG 05-06 10:38:57.556490.556490 cuda_h.py:27] end *layer_moe_fused cost 47.218 ms
DEBUG 05-06 10:38:57.557127.557127 cuda_h.py:27] end prefill_layer cost 51.601 ms
DEBUG 05-06 10:38:57.557328.557328 lmp.py:841] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 10:38:57.557429.557429 lmp.py:824] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 10:38:57.560267.560267 cuda_h.py:27] end *sagl cost 2.300 ms
experts_cpu_alloc {'expert_ids': [11, 31, 63, 79, 115, 119, 67, 51, 19, 27, 39, 83, 71, 15, 111, 99, 127, 23, 43, 75, 103, 95, 7, 3, 84, 112, 64, 44, 52, 96, 116, 20, 8, 24, 68, 120, 124, 40, 88, 76, 80, 72, 32, 36, 92, 48, 56, 16, 12, 0, 4, 29, 33, 41, 105, 117, 77, 97, 113, 73, 9, 37, 17, 45, 61, 13, 21, 89, 57, 125, 69, 81, 101, 93, 5, 1, 14, 26, 122, 10, 34, 90, 58, 114, 82, 98, 62, 42, 86, 22, 30, 38, 54, 102, 74, 46, 70, 106, 2, 6], 'token_total': 4096, 'token_per_expert': {11: 1, 31: 1, 63: 1, 79: 1, 115: 2, 119: 2, 67: 5, 51: 7, 19: 8, 27: 13, 39: 13, 83: 17, 71: 18, 15: 19, 111: 19, 99: 23, 127: 24, 23: 32, 43: 47, 75: 57, 103: 68, 95: 147, 7: 263, 3: 266, 84: 1, 112: 1, 64: 2, 44: 3, 52: 3, 96: 3, 116: 3, 20: 6, 8: 7, 24: 9, 68: 9, 120: 9, 124: 9, 40: 12, 88: 12, 76: 13, 80: 14, 72: 20, 32: 23, 36: 26, 92: 29, 48: 40, 56: 50, 16: 69, 12: 77, 0: 262, 4: 284, 29: 1, 33: 1, 41: 2, 105: 3, 117: 6, 77: 8, 97: 8, 113: 10, 73: 11, 9: 12, 37: 12, 17: 14, 45: 15, 61: 16, 13: 19, 21: 19, 89: 23, 57: 27, 125: 27, 69: 32, 81: 46, 101: 76, 93: 77, 5: 261, 1: 269, 14: 1, 26: 1, 122: 2, 10: 3, 34: 3, 90: 3, 58: 4, 114: 4, 82: 5, 98: 5, 62: 10, 42: 11, 86: 12, 22: 16, 30: 18, 38: 19, 54: 20, 102: 39, 74: 63, 46: 89, 70: 103, 106: 105, 2: 257, 6: 258}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.562381.562381 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.600ms | allocate_experts_across_cpu_gpu: 0.375ms
INFO 05-06 10:38:57.562934.562934 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:57.563311.563311 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005431175231933594 seconds
INFO 05-06 10:38:57.564185.564185 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009775161743164062 seconds
INFO 05-06 10:38:57.564440.564440 lmp.py:1484] [layer_moe_fused] experts compute time: 1.6689300537109375e-06 seconds
INFO 05-06 10:38:57.606787.606787 lmp.py:1496] [layer_moe_fused] to time: 0.0001742839813232422 seconds
INFO 05-06 10:38:57.607631.607631 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0419771671295166 seconds
DEBUG 05-06 10:38:57.608403.608403 cuda_h.py:27] end *layer_moe_fused cost 46.707 ms
DEBUG 05-06 10:38:57.608510.608510 cuda_h.py:27] end prefill_layer cost 51.070 ms
DEBUG 05-06 10:38:57.608566.608566 lmp.py:841] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 10:38:57.608190.608190 lmp.py:824] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 10:38:57.611808.611808 cuda_h.py:27] end *sagl cost 2.266 ms
experts_cpu_alloc {'expert_ids': [27, 51, 107, 123, 35, 15, 67, 103, 111, 11, 59, 83, 79, 19, 99, 39, 31, 63, 43, 71, 47, 75, 127, 115, 3, 7, 52, 12, 40, 56, 124, 64, 120, 28, 44, 112, 100, 20, 68, 84, 92, 72, 108, 16, 88, 80, 60, 8, 76, 4, 0, 77, 37, 109, 53, 61, 9, 25, 29, 97, 89, 69, 73, 93, 117, 105, 121, 13, 49, 113, 41, 85, 21, 125, 57, 81, 5, 1, 38, 66, 70, 26, 98, 34, 50, 78, 94, 90, 10, 46, 82, 54, 126, 58, 18, 106, 14, 74, 42, 62, 86, 2, 6], 'token_total': 4096, 'token_per_expert': {27: 1, 51: 1, 107: 1, 123: 1, 35: 2, 15: 3, 67: 3, 103: 3, 111: 3, 11: 4, 59: 4, 83: 6, 79: 9, 19: 13, 99: 13, 39: 15, 31: 17, 63: 18, 43: 20, 71: 24, 47: 25, 75: 26, 127: 26, 115: 52, 3: 258, 7: 273, 52: 1, 12: 2, 40: 2, 56: 2, 124: 2, 64: 3, 120: 4, 28: 8, 44: 10, 112: 10, 100: 11, 20: 16, 68: 17, 84: 18, 92: 23, 72: 29, 108: 29, 16: 31, 88: 50, 80: 74, 60: 76, 8: 78, 76: 115, 4: 261, 0: 318, 77: 1, 37: 2, 109: 2, 53: 3, 61: 3, 9: 4, 25: 4, 29: 4, 97: 5, 89: 7, 69: 8, 73: 9, 93: 10, 117: 10, 105: 12, 121: 12, 13: 15, 49: 20, 113: 24, 41: 29, 85: 32, 21: 34, 125: 35, 57: 38, 81: 67, 5: 271, 1: 342, 38: 1, 66: 1, 70: 1, 26: 4, 98: 6, 34: 7, 50: 8, 78: 9, 94: 13, 90: 16, 10: 17, 46: 18, 82: 21, 54: 22, 126: 27, 58: 35, 18: 40, 106: 42, 14: 49, 74: 54, 42: 56, 62: 60, 86: 61, 2: 256, 6: 258}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.613433.613433 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.601ms | allocate_experts_across_cpu_gpu: 0.381ms
INFO 05-06 10:38:57.613648.613648 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:57.614549.614549 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005474090576171875 seconds
INFO 05-06 10:38:57.615403.615403 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00084686279296875 seconds
INFO 05-06 10:38:57.615923.615923 lmp.py:1484] [layer_moe_fused] experts compute time: 1.1920928955078125e-06 seconds
INFO 05-06 10:38:57.657190.657190 lmp.py:1496] [layer_moe_fused] to time: 0.0001773834228515625 seconds
INFO 05-06 10:38:57.658220.658220 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04227733612060547 seconds
DEBUG 05-06 10:38:57.658071.658071 cuda_h.py:27] end *layer_moe_fused cost 46.325 ms
DEBUG 05-06 10:38:57.659126.659126 cuda_h.py:27] end prefill_layer cost 50.713 ms
DEBUG 05-06 10:38:57.659388.659388 lmp.py:841] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 10:38:57.659840.659840 lmp.py:824] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 10:38:57.662469.662469 cuda_h.py:27] end *sagl cost 3.117 ms
experts_cpu_alloc {'expert_ids': [15, 35, 47, 115, 63, 127, 59, 71, 39, 51, 11, 91, 43, 27, 123, 119, 19, 31, 99, 23, 67, 111, 87, 79, 83, 3, 7, 12, 64, 84, 40, 72, 80, 48, 52, 28, 44, 120, 8, 36, 124, 116, 100, 20, 112, 76, 24, 108, 32, 68, 16, 92, 56, 4, 0, 21, 53, 9, 33, 13, 125, 117, 121, 61, 29, 89, 25, 37, 57, 69, 77, 17, 49, 93, 81, 113, 1, 5, 94, 106, 114, 22, 34, 58, 74, 126, 98, 110, 118, 122, 42, 50, 62, 70, 54, 18, 82, 46, 66, 38, 30, 10, 102, 2, 6], 'token_total': 4096, 'token_per_expert': {15: 1, 35: 1, 47: 1, 115: 1, 63: 2, 127: 2, 59: 3, 71: 6, 39: 7, 51: 7, 11: 8, 91: 8, 43: 10, 27: 11, 123: 12, 119: 21, 19: 23, 31: 27, 99: 28, 23: 43, 67: 47, 111: 56, 87: 74, 79: 75, 83: 76, 3: 260, 7: 301, 12: 1, 64: 1, 84: 1, 40: 2, 72: 3, 80: 3, 48: 4, 52: 4, 28: 5, 44: 5, 120: 5, 8: 6, 36: 8, 124: 10, 116: 19, 100: 27, 20: 28, 112: 31, 76: 32, 24: 33, 108: 35, 32: 38, 68: 50, 16: 84, 92: 87, 56: 95, 4: 258, 0: 259, 21: 1, 53: 1, 9: 2, 33: 2, 13: 3, 125: 3, 117: 8, 121: 12, 61: 15, 29: 16, 89: 16, 25: 17, 37: 21, 57: 21, 69: 21, 77: 22, 17: 47, 49: 52, 93: 57, 81: 72, 113: 73, 1: 264, 5: 264, 94: 1, 106: 1, 114: 1, 22: 2, 34: 2, 58: 2, 74: 2, 126: 2, 98: 3, 110: 3, 118: 3, 122: 4, 42: 5, 50: 5, 62: 5, 70: 5, 54: 6, 18: 7, 82: 7, 46: 9, 66: 15, 38: 16, 30: 24, 10: 25, 102: 99, 2: 286, 6: 301}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 23, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.665763.665763 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.609ms | allocate_experts_across_cpu_gpu: 0.396ms
INFO 05-06 10:38:57.665746.665746 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:57.666721.666721 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005557537078857422 seconds
INFO 05-06 10:38:57.667999.667999 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0011057853698730469 seconds
INFO 05-06 10:38:57.667236.667236 lmp.py:1484] [layer_moe_fused] experts compute time: 1.430511474609375e-06 seconds
INFO 05-06 10:38:57.712962.712962 lmp.py:1496] [layer_moe_fused] to time: 0.0001842975616455078 seconds
INFO 05-06 10:38:57.713853.713853 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.045470476150512695 seconds
DEBUG 05-06 10:38:57.714438.714438 cuda_h.py:27] end *layer_moe_fused cost 49.852 ms
DEBUG 05-06 10:38:57.714963.714963 cuda_h.py:27] end prefill_layer cost 54.891 ms
DEBUG 05-06 10:38:57.714409.714409 lmp.py:841] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 10:38:57.714510.714510 lmp.py:824] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 10:38:57.717628.717628 cuda_h.py:27] end *sagl cost 2.293 ms
experts_cpu_alloc {'expert_ids': [47, 59, 31, 63, 111, 127, 119, 107, 123, 79, 103, 35, 95, 23, 115, 91, 19, 15, 71, 39, 7, 3, 8, 32, 48, 96, 20, 24, 112, 88, 12, 104, 36, 68, 76, 80, 100, 40, 84, 124, 92, 116, 108, 0, 4, 17, 37, 81, 113, 13, 33, 125, 105, 77, 89, 101, 85, 117, 25, 49, 73, 97, 45, 21, 53, 1, 5, 18, 94, 102, 90, 58, 70, 22, 38, 98, 34, 106, 46, 82, 118, 110, 114, 86, 50, 74, 78, 2, 6], 'token_total': 4096, 'token_per_expert': {47: 1, 59: 1, 31: 2, 63: 2, 111: 2, 127: 2, 119: 3, 107: 4, 123: 5, 79: 7, 103: 7, 35: 19, 95: 19, 23: 20, 115: 25, 91: 28, 19: 40, 15: 68, 71: 92, 39: 99, 7: 256, 3: 271, 8: 1, 32: 1, 48: 1, 96: 1, 20: 3, 24: 5, 112: 5, 88: 7, 12: 8, 104: 8, 36: 10, 68: 11, 76: 11, 80: 11, 100: 11, 40: 12, 84: 12, 124: 14, 92: 19, 116: 59, 108: 67, 0: 256, 4: 257, 17: 1, 37: 1, 81: 1, 113: 1, 13: 2, 33: 3, 125: 3, 105: 4, 77: 6, 89: 7, 101: 12, 85: 16, 117: 21, 25: 29, 49: 29, 73: 38, 97: 38, 45: 60, 21: 101, 53: 104, 1: 264, 5: 290, 18: 2, 94: 2, 102: 2, 90: 3, 58: 5, 70: 5, 22: 7, 38: 7, 98: 7, 34: 14, 106: 34, 46: 43, 82: 50, 118: 52, 110: 55, 114: 65, 86: 67, 50: 76, 74: 94, 78: 159, 2: 256, 6: 297}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 22, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 23, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 22, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 22, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.719451.719451 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.616ms | allocate_experts_across_cpu_gpu: 0.349ms
INFO 05-06 10:38:57.719310.719310 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.0994415283203125e-06 seconds
INFO 05-06 10:38:57.720508.720508 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005395412445068359 seconds
INFO 05-06 10:38:57.721379.721379 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001028299331665039 seconds
INFO 05-06 10:38:57.721528.721528 lmp.py:1484] [layer_moe_fused] experts compute time: 1.6689300537109375e-06 seconds
INFO 05-06 10:38:57.761372.761372 lmp.py:1496] [layer_moe_fused] to time: 0.00017213821411132812 seconds
INFO 05-06 10:38:57.761926.761926 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03940725326538086 seconds
DEBUG 05-06 10:38:57.762742.762742 cuda_h.py:27] end *layer_moe_fused cost 43.879 ms
DEBUG 05-06 10:38:57.763399.763399 cuda_h.py:27] end prefill_layer cost 48.294 ms
DEBUG 05-06 10:38:57.763646.763646 lmp.py:841] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 10:38:57.763509.763509 lmp.py:824] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 10:38:57.765969.765969 cuda_h.py:27] end *sagl cost 2.236 ms
experts_cpu_alloc {'expert_ids': [95, 47, 83, 43, 87, 123, 107, 27, 11, 67, 75, 115, 99, 55, 119, 15, 39, 51, 103, 63, 59, 71, 79, 91, 31, 7, 3, 48, 104, 8, 52, 16, 40, 64, 96, 68, 92, 28, 60, 108, 124, 84, 116, 20, 32, 120, 100, 4, 0, 45, 53, 65, 97, 61, 9, 57, 73, 13, 93, 101, 117, 41, 125, 113, 69, 33, 21, 25, 17, 37, 81, 121, 5, 1, 10, 66, 74, 70, 106, 90, 82, 26, 122, 38, 46, 42, 86, 34, 118, 22, 126, 98, 102, 78, 14, 110, 114, 2, 6], 'token_total': 4096, 'token_per_expert': {95: 1, 47: 2, 83: 2, 43: 3, 87: 3, 123: 3, 107: 4, 27: 6, 11: 7, 67: 8, 75: 8, 115: 8, 99: 10, 55: 13, 119: 14, 15: 16, 39: 21, 51: 25, 103: 25, 63: 37, 59: 38, 71: 46, 79: 75, 91: 100, 31: 119, 7: 256, 3: 274, 48: 1, 104: 3, 8: 5, 52: 5, 16: 6, 40: 6, 64: 6, 96: 6, 68: 7, 92: 7, 28: 8, 60: 10, 108: 12, 124: 17, 84: 21, 116: 21, 20: 25, 32: 49, 120: 68, 100: 104, 4: 256, 0: 257, 45: 1, 53: 1, 65: 1, 97: 1, 61: 2, 9: 3, 57: 3, 73: 5, 13: 6, 93: 7, 101: 8, 117: 9, 41: 12, 125: 21, 113: 22, 69: 23, 33: 27, 21: 31, 25: 42, 17: 48, 37: 51, 81: 51, 121: 65, 5: 256, 1: 277, 10: 1, 66: 1, 74: 1, 70: 2, 106: 2, 90: 3, 82: 4, 26: 5, 122: 5, 38: 8, 46: 8, 42: 11, 86: 13, 34: 24, 118: 25, 22: 26, 126: 33, 98: 34, 102: 36, 78: 46, 14: 49, 110: 92, 114: 106, 2: 263, 6: 301}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 22, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.767489.767489 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.613ms | allocate_experts_across_cpu_gpu: 0.435ms
INFO 05-06 10:38:57.768387.768387 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:57.768658.768658 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005462169647216797 seconds
INFO 05-06 10:38:57.769255.769255 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0007388591766357422 seconds
INFO 05-06 10:38:57.769821.769821 lmp.py:1484] [layer_moe_fused] experts compute time: 1.6689300537109375e-06 seconds
INFO 05-06 10:38:57.812548.812548 lmp.py:1496] [layer_moe_fused] to time: 0.00017714500427246094 seconds
INFO 05-06 10:38:57.812717.812717 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.042324066162109375 seconds
DEBUG 05-06 10:38:57.813244.813244 cuda_h.py:27] end *layer_moe_fused cost 46.339 ms
DEBUG 05-06 10:38:57.813239.813239 cuda_h.py:27] end prefill_layer cost 50.668 ms
DEBUG 05-06 10:38:57.813870.813870 lmp.py:841] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 10:38:57.814494.814494 lmp.py:824] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 10:38:57.816664.816664 cuda_h.py:27] end *sagl cost 2.294 ms
experts_cpu_alloc {'expert_ids': [91, 111, 19, 51, 63, 71, 15, 35, 43, 67, 23, 107, 83, 11, 127, 31, 59, 123, 95, 99, 103, 47, 75, 39, 119, 115, 3, 7, 56, 68, 96, 48, 64, 36, 40, 44, 108, 116, 60, 120, 28, 16, 92, 112, 52, 72, 8, 32, 24, 12, 104, 76, 80, 100, 124, 4, 0, 29, 33, 37, 41, 73, 77, 85, 101, 9, 109, 81, 125, 21, 93, 25, 45, 105, 57, 89, 13, 53, 113, 117, 97, 65, 121, 1, 5, 18, 46, 54, 14, 70, 98, 10, 106, 118, 78, 102, 110, 58, 126, 90, 38, 114, 34, 74, 122, 30, 42, 62, 66, 50, 86, 26, 6, 2], 'token_total': 4096, 'token_per_expert': {91: 1, 111: 1, 19: 2, 51: 2, 63: 2, 71: 2, 15: 3, 35: 4, 43: 4, 67: 4, 23: 6, 107: 7, 83: 9, 11: 17, 127: 20, 31: 25, 59: 28, 123: 30, 95: 31, 99: 35, 103: 42, 47: 50, 75: 50, 39: 61, 119: 68, 115: 144, 3: 260, 7: 266, 56: 2, 68: 3, 96: 3, 48: 4, 64: 5, 36: 6, 40: 6, 44: 6, 108: 6, 116: 6, 60: 8, 120: 9, 28: 10, 16: 11, 92: 11, 112: 12, 52: 14, 72: 16, 8: 17, 32: 17, 24: 18, 12: 19, 104: 19, 76: 20, 80: 28, 100: 38, 124: 79, 4: 256, 0: 261, 29: 1, 33: 1, 37: 1, 41: 2, 73: 2, 77: 2, 85: 2, 101: 3, 9: 4, 109: 4, 81: 5, 125: 9, 21: 10, 93: 10, 25: 11, 45: 11, 105: 14, 57: 16, 89: 17, 13: 23, 53: 24, 113: 39, 117: 40, 97: 54, 65: 59, 121: 98, 1: 256, 5: 262, 18: 1, 46: 1, 54: 1, 14: 2, 70: 3, 98: 3, 10: 4, 106: 4, 118: 5, 78: 7, 102: 7, 110: 7, 58: 8, 126: 8, 90: 9, 38: 11, 114: 11, 34: 13, 74: 16, 122: 24, 30: 25, 42: 28, 62: 28, 66: 47, 50: 54, 86: 77, 26: 79, 6: 258, 2: 291}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.818852.818852 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.604ms | allocate_experts_across_cpu_gpu: 0.409ms
INFO 05-06 10:38:57.818518.818518 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:57.819541.819541 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005540847778320312 seconds
INFO 05-06 10:38:57.820511.820511 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0007917881011962891 seconds
INFO 05-06 10:38:57.820806.820806 lmp.py:1484] [layer_moe_fused] experts compute time: 1.9073486328125e-06 seconds
INFO 05-06 10:38:57.863211.863211 lmp.py:1496] [layer_moe_fused] to time: 0.00018477439880371094 seconds
INFO 05-06 10:38:57.864095.864095 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04298973083496094 seconds
DEBUG 05-06 10:38:57.864125.864125 cuda_h.py:27] end *layer_moe_fused cost 47.143 ms
DEBUG 05-06 10:38:57.865915.865915 cuda_h.py:27] end prefill_layer cost 51.527 ms
DEBUG 05-06 10:38:57.865700.865700 lmp.py:841] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 10:38:57.865516.865516 lmp.py:824] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 10:38:57.868969.868969 cuda_h.py:27] end *sagl cost 2.265 ms
experts_cpu_alloc {'expert_ids': [79, 35, 67, 111, 11, 19, 127, 55, 119, 43, 107, 115, 31, 59, 95, 103, 47, 63, 99, 39, 51, 23, 71, 75, 83, 91, 3, 7, 32, 56, 100, 80, 8, 28, 96, 36, 48, 116, 16, 64, 120, 24, 84, 72, 88, 104, 124, 52, 108, 68, 112, 76, 4, 0, 45, 57, 25, 105, 29, 117, 33, 17, 77, 121, 13, 41, 69, 97, 113, 73, 125, 93, 85, 21, 37, 81, 101, 9, 65, 109, 1, 5, 22, 54, 94, 126, 82, 118, 18, 58, 34, 38, 78, 46, 86, 114, 102, 14, 42, 70, 98, 66, 10, 30, 90, 6, 2], 'token_total': 4096, 'token_per_expert': {79: 1, 35: 2, 67: 2, 111: 2, 11: 3, 19: 3, 127: 3, 55: 5, 119: 9, 43: 10, 107: 10, 115: 11, 31: 14, 59: 15, 95: 15, 103: 17, 47: 18, 63: 21, 99: 26, 39: 31, 51: 31, 23: 34, 71: 39, 75: 44, 83: 73, 91: 86, 3: 256, 7: 279, 32: 1, 56: 1, 100: 1, 80: 5, 8: 6, 28: 6, 96: 7, 36: 8, 48: 8, 116: 13, 16: 15, 64: 18, 120: 18, 24: 22, 84: 22, 72: 23, 88: 23, 104: 23, 124: 26, 52: 32, 108: 49, 68: 85, 112: 90, 76: 101, 4: 260, 0: 265, 45: 1, 57: 1, 25: 3, 105: 3, 29: 5, 117: 5, 33: 6, 17: 7, 77: 7, 121: 7, 13: 8, 41: 10, 69: 11, 97: 16, 113: 16, 73: 18, 125: 21, 93: 23, 85: 24, 21: 25, 37: 25, 81: 26, 101: 31, 9: 40, 65: 68, 109: 72, 1: 270, 5: 279, 22: 1, 54: 1, 94: 1, 126: 1, 82: 3, 118: 4, 18: 5, 58: 5, 34: 6, 38: 6, 78: 7, 46: 8, 86: 8, 114: 10, 102: 11, 14: 13, 42: 15, 70: 24, 98: 27, 66: 33, 10: 42, 30: 44, 90: 67, 6: 256, 2: 282}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.870429.870429 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.615ms | allocate_experts_across_cpu_gpu: 0.393ms
INFO 05-06 10:38:57.870697.870697 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:57.871095.871095 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.00054931640625 seconds
INFO 05-06 10:38:57.872537.872537 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008792877197265625 seconds
INFO 05-06 10:38:57.872402.872402 lmp.py:1484] [layer_moe_fused] experts compute time: 1.6689300537109375e-06 seconds
INFO 05-06 10:38:57.920858.920858 lmp.py:1496] [layer_moe_fused] to time: 0.00017523765563964844 seconds
INFO 05-06 10:38:57.921756.921756 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04847860336303711 seconds
DEBUG 05-06 10:38:57.922250.922250 cuda_h.py:27] end *layer_moe_fused cost 52.597 ms
DEBUG 05-06 10:38:57.922397.922397 cuda_h.py:27] end prefill_layer cost 56.959 ms
DEBUG 05-06 10:38:57.922605.922605 lmp.py:841] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 10:38:57.922182.922182 lmp.py:824] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 10:38:57.925916.925916 cuda_h.py:27] end *sagl cost 2.304 ms
experts_cpu_alloc {'expert_ids': [11, 103, 111, 59, 99, 43, 71, 51, 123, 91, 15, 119, 79, 127, 75, 19, 23, 55, 63, 31, 83, 107, 67, 87, 7, 3, 28, 36, 60, 84, 64, 88, 120, 56, 104, 24, 92, 40, 80, 116, 20, 72, 108, 48, 76, 96, 100, 68, 124, 12, 8, 44, 52, 16, 32, 0, 4, 73, 101, 49, 33, 53, 81, 89, 13, 37, 109, 9, 61, 45, 57, 21, 93, 113, 17, 65, 121, 77, 97, 117, 85, 125, 105, 5, 1, 74, 94, 106, 50, 122, 34, 98, 10, 38, 118, 18, 62, 90, 82, 102, 30, 22, 78, 58, 110, 70, 42, 54, 114, 26, 14, 66, 86, 126, 6, 2], 'token_total': 4096, 'token_per_expert': {11: 1, 103: 1, 111: 1, 59: 2, 99: 2, 43: 3, 71: 3, 51: 5, 123: 5, 91: 6, 15: 7, 119: 7, 79: 13, 127: 13, 75: 17, 19: 20, 23: 20, 55: 23, 63: 23, 31: 34, 83: 37, 107: 41, 67: 68, 87: 95, 7: 258, 3: 268, 28: 1, 36: 1, 60: 1, 84: 1, 64: 2, 88: 2, 120: 2, 56: 3, 104: 4, 24: 6, 92: 8, 40: 11, 80: 12, 116: 14, 20: 16, 72: 16, 108: 18, 48: 21, 76: 23, 96: 24, 100: 27, 68: 28, 124: 30, 12: 31, 8: 35, 44: 39, 52: 87, 16: 94, 32: 102, 0: 283, 4: 283, 73: 1, 101: 1, 49: 2, 33: 4, 53: 4, 81: 4, 89: 4, 13: 5, 37: 5, 109: 5, 9: 7, 61: 7, 45: 8, 57: 8, 21: 10, 93: 10, 113: 10, 17: 11, 65: 12, 121: 15, 77: 16, 97: 16, 117: 16, 85: 17, 125: 23, 105: 54, 5: 276, 1: 311, 74: 1, 94: 1, 106: 1, 50: 2, 122: 2, 34: 3, 98: 3, 10: 4, 38: 4, 118: 4, 18: 6, 62: 6, 90: 7, 82: 8, 102: 8, 30: 11, 22: 13, 78: 14, 58: 18, 110: 18, 70: 19, 42: 22, 54: 23, 114: 23, 26: 24, 14: 31, 66: 44, 86: 66, 126: 124, 6: 257, 2: 269}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.927962.927962 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.618ms | allocate_experts_across_cpu_gpu: 0.482ms
INFO 05-06 10:38:57.927945.927945 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:57.928078.928078 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.000545501708984375 seconds
INFO 05-06 10:38:57.929892.929892 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0007376670837402344 seconds
INFO 05-06 10:38:57.929227.929227 lmp.py:1484] [layer_moe_fused] experts compute time: 1.6689300537109375e-06 seconds
INFO 05-06 10:38:57.973191.973191 lmp.py:1496] [layer_moe_fused] to time: 0.00017023086547851562 seconds
INFO 05-06 10:38:57.973512.973512 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.043882131576538086 seconds
DEBUG 05-06 10:38:57.974239.974239 cuda_h.py:27] end *layer_moe_fused cost 47.936 ms
DEBUG 05-06 10:38:57.975452.975452 cuda_h.py:27] end prefill_layer cost 52.365 ms
DEBUG 05-06 10:38:57.975575.975575 lmp.py:841] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 10:38:57.975867.975867 lmp.py:824] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 10:38:57.978520.978520 cuda_h.py:27] end *sagl cost 3.361 ms
experts_cpu_alloc {'expert_ids': [79, 83, 111, 127, 51, 91, 15, 87, 123, 11, 59, 19, 31, 99, 119, 55, 67, 103, 35, 63, 71, 43, 47, 107, 39, 75, 27, 95, 23, 7, 3, 88, 112, 92, 96, 8, 16, 36, 44, 80, 104, 60, 124, 116, 108, 100, 84, 120, 12, 48, 68, 56, 20, 28, 52, 64, 40, 72, 76, 24, 0, 4, 77, 81, 117, 93, 9, 65, 85, 13, 97, 29, 33, 109, 113, 125, 45, 73, 57, 53, 49, 101, 17, 89, 61, 21, 37, 69, 1, 5, 66, 102, 38, 62, 34, 42, 118, 114, 122, 14, 90, 126, 22, 78, 98, 94, 70, 54, 106, 10, 58, 18, 74, 86, 2, 6], 'token_total': 4096, 'token_per_expert': {79: 1, 83: 1, 111: 1, 127: 1, 51: 2, 91: 2, 15: 3, 87: 3, 123: 4, 11: 5, 59: 5, 19: 7, 31: 7, 99: 7, 119: 7, 55: 11, 67: 13, 103: 18, 35: 20, 63: 22, 71: 22, 43: 25, 47: 25, 107: 36, 39: 51, 75: 51, 27: 53, 95: 66, 23: 78, 7: 256, 3: 268, 88: 1, 112: 2, 92: 3, 96: 4, 8: 6, 16: 6, 36: 6, 44: 7, 80: 7, 104: 7, 60: 8, 124: 8, 116: 9, 108: 12, 100: 13, 84: 14, 120: 15, 12: 20, 48: 22, 68: 22, 56: 25, 20: 26, 28: 28, 52: 30, 64: 31, 40: 33, 72: 36, 76: 57, 24: 77, 0: 264, 4: 273, 77: 1, 81: 1, 117: 1, 93: 2, 9: 3, 65: 3, 85: 3, 13: 4, 97: 4, 29: 5, 33: 6, 109: 6, 113: 6, 125: 9, 45: 14, 73: 16, 57: 21, 53: 26, 49: 28, 101: 30, 17: 34, 89: 36, 61: 39, 21: 51, 37: 73, 69: 86, 1: 259, 5: 271, 66: 1, 102: 2, 38: 3, 62: 3, 34: 4, 42: 4, 118: 4, 114: 5, 122: 5, 14: 6, 90: 6, 126: 7, 22: 12, 78: 13, 98: 13, 94: 14, 70: 18, 54: 22, 106: 22, 10: 23, 58: 30, 18: 35, 74: 62, 86: 67, 2: 260, 6: 274}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:57.981915.981915 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.606ms | allocate_experts_across_cpu_gpu: 0.421ms
INFO 05-06 10:38:57.981044.981044 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:57.982747.982747 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005393028259277344 seconds
INFO 05-06 10:38:57.983113.983113 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0008158683776855469 seconds
INFO 05-06 10:38:57.983998.983998 lmp.py:1484] [layer_moe_fused] experts compute time: 1.1920928955078125e-06 seconds
INFO 05-06 10:38:58.026243.026243 lmp.py:1496] [layer_moe_fused] to time: 0.0001773834228515625 seconds
INFO 05-06 10:38:58.026465.026465 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.042893171310424805 seconds
DEBUG 05-06 10:38:58.027145.027145 cuda_h.py:27] end *layer_moe_fused cost 47.266 ms
DEBUG 05-06 10:38:58.028412.028412 cuda_h.py:27] end prefill_layer cost 52.850 ms
DEBUG 05-06 10:38:58.028236.028236 lmp.py:841] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 10:38:58.028290.028290 lmp.py:824] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 10:38:58.030002.030002 cuda_h.py:27] end *sagl cost 2.234 ms
experts_cpu_alloc {'expert_ids': [11, 19, 23, 27, 55, 63, 115, 51, 59, 67, 39, 103, 107, 35, 95, 15, 47, 75, 91, 123, 71, 87, 127, 119, 31, 83, 43, 111, 99, 7, 3, 44, 24, 112, 96, 16, 52, 124, 56, 80, 12, 116, 48, 68, 108, 92, 60, 88, 120, 72, 84, 104, 40, 100, 76, 64, 8, 32, 36, 0, 4, 25, 41, 113, 21, 109, 13, 45, 89, 97, 9, 29, 37, 125, 73, 57, 69, 81, 93, 65, 17, 53, 33, 61, 49, 101, 85, 77, 121, 5, 1, 18, 86, 74, 82, 94, 98, 102, 66, 42, 30, 62, 26, 70, 114, 122, 46, 90, 34, 10, 38, 78, 110, 118, 58, 14, 50, 54, 6, 2], 'token_total': 4096, 'token_per_expert': {11: 2, 19: 2, 23: 2, 27: 2, 55: 3, 63: 3, 115: 3, 51: 4, 59: 4, 67: 4, 39: 5, 103: 8, 107: 8, 35: 10, 95: 12, 15: 14, 47: 14, 75: 14, 91: 15, 123: 16, 71: 20, 87: 21, 127: 22, 119: 25, 31: 34, 83: 39, 43: 49, 111: 53, 99: 60, 7: 264, 3: 297, 44: 1, 24: 2, 112: 2, 96: 3, 16: 5, 52: 5, 124: 6, 56: 8, 80: 10, 12: 11, 116: 11, 48: 12, 68: 12, 108: 12, 92: 16, 60: 17, 88: 19, 120: 19, 72: 22, 84: 23, 104: 23, 40: 24, 100: 24, 76: 27, 64: 28, 8: 31, 32: 44, 36: 48, 0: 262, 4: 286, 25: 1, 41: 1, 113: 1, 21: 5, 109: 6, 13: 7, 45: 7, 89: 7, 97: 7, 9: 8, 29: 8, 37: 10, 125: 11, 73: 12, 57: 14, 69: 15, 81: 17, 93: 22, 65: 24, 17: 28, 53: 32, 33: 33, 61: 35, 49: 37, 101: 40, 85: 44, 77: 57, 121: 75, 5: 264, 1: 283, 18: 1, 86: 1, 74: 3, 82: 3, 94: 3, 98: 3, 102: 3, 66: 4, 42: 5, 30: 6, 62: 6, 26: 7, 70: 7, 114: 7, 122: 9, 46: 10, 90: 10, 34: 12, 10: 13, 38: 20, 78: 25, 110: 29, 118: 30, 58: 34, 14: 36, 50: 42, 54: 56, 6: 256, 2: 302}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.033319.033319 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.618ms | allocate_experts_across_cpu_gpu: 0.431ms
INFO 05-06 10:38:58.033785.033785 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:58.033601.033601 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005483627319335938 seconds
INFO 05-06 10:38:58.035956.035956 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0010538101196289062 seconds
INFO 05-06 10:38:58.035052.035052 lmp.py:1484] [layer_moe_fused] experts compute time: 1.1920928955078125e-06 seconds
INFO 05-06 10:38:58.079671.079671 lmp.py:1496] [layer_moe_fused] to time: 0.00017786026000976562 seconds
INFO 05-06 10:38:58.079146.079146 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0434112548828125 seconds
DEBUG 05-06 10:38:58.080879.080879 cuda_h.py:27] end *layer_moe_fused cost 48.231 ms
DEBUG 05-06 10:38:58.080953.080953 cuda_h.py:27] end prefill_layer cost 52.618 ms
DEBUG 05-06 10:38:58.080307.080307 lmp.py:841] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 10:38:58.081838.081838 lmp.py:824] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 10:38:58.083390.083390 cuda_h.py:27] end *sagl cost 2.502 ms
experts_cpu_alloc {'expert_ids': [43, 107, 115, 67, 127, 59, 103, 15, 83, 111, 27, 99, 55, 47, 11, 19, 35, 75, 63, 119, 39, 23, 31, 79, 123, 51, 7, 3, 8, 100, 116, 124, 68, 20, 56, 112, 96, 108, 84, 60, 104, 12, 36, 48, 72, 80, 16, 76, 40, 24, 88, 92, 44, 64, 52, 4, 0, 49, 113, 45, 81, 101, 29, 53, 57, 77, 121, 25, 65, 97, 13, 17, 33, 73, 41, 69, 125, 21, 109, 9, 61, 117, 37, 89, 5, 1, 74, 18, 46, 66, 70, 82, 94, 114, 42, 118, 90, 106, 86, 58, 22, 98, 126, 26, 50, 102, 10, 38, 122, 6, 2], 'token_total': 4096, 'token_per_expert': {43: 2, 107: 2, 115: 2, 67: 3, 127: 5, 59: 7, 103: 7, 15: 10, 83: 10, 111: 10, 27: 11, 99: 13, 55: 14, 47: 15, 11: 17, 19: 17, 35: 17, 75: 18, 63: 20, 119: 20, 39: 21, 23: 24, 31: 24, 79: 27, 123: 33, 51: 56, 7: 302, 3: 305, 8: 1, 100: 1, 116: 1, 124: 1, 68: 3, 20: 5, 56: 6, 112: 6, 96: 10, 108: 10, 84: 11, 60: 12, 104: 12, 12: 16, 36: 17, 48: 17, 72: 17, 80: 22, 16: 24, 76: 30, 40: 34, 24: 35, 88: 35, 92: 58, 44: 65, 64: 67, 52: 109, 4: 258, 0: 263, 49: 1, 113: 1, 45: 2, 81: 2, 101: 2, 29: 3, 53: 3, 57: 3, 77: 4, 121: 4, 25: 5, 65: 5, 97: 9, 13: 10, 17: 10, 33: 10, 73: 11, 41: 16, 69: 16, 125: 16, 21: 22, 109: 26, 9: 28, 61: 31, 117: 52, 37: 79, 89: 80, 5: 272, 1: 281, 74: 1, 18: 2, 46: 2, 66: 2, 70: 2, 82: 2, 94: 2, 114: 2, 42: 5, 118: 5, 90: 9, 106: 9, 86: 10, 58: 12, 22: 14, 98: 15, 126: 16, 26: 21, 50: 31, 102: 32, 10: 37, 38: 86, 122: 88, 6: 261, 2: 268}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.086925.086925 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.627ms | allocate_experts_across_cpu_gpu: 0.405ms
INFO 05-06 10:38:58.086067.086067 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.337860107421875e-06 seconds
INFO 05-06 10:38:58.087736.087736 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005462169647216797 seconds
INFO 05-06 10:38:58.088934.088934 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009243488311767578 seconds
INFO 05-06 10:38:58.088569.088569 lmp.py:1484] [layer_moe_fused] experts compute time: 5.0067901611328125e-06 seconds
INFO 05-06 10:38:58.130374.130374 lmp.py:1496] [layer_moe_fused] to time: 0.00018024444580078125 seconds
INFO 05-06 10:38:58.130079.130079 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04196524620056152 seconds
DEBUG 05-06 10:38:58.131032.131032 cuda_h.py:27] end *layer_moe_fused cost 46.547 ms
DEBUG 05-06 10:38:58.132153.132153 cuda_h.py:27] end prefill_layer cost 51.310 ms
DEBUG 05-06 10:38:58.132361.132361 lmp.py:841] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 10:38:58.132415.132415 lmp.py:824] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 10:38:58.134585.134585 cuda_h.py:27] end *sagl cost 2.293 ms
experts_cpu_alloc {'expert_ids': [31, 87, 119, 75, 91, 99, 35, 11, 83, 19, 79, 111, 103, 47, 95, 55, 15, 71, 123, 27, 43, 59, 63, 107, 7, 3, 16, 80, 104, 124, 20, 24, 60, 76, 36, 12, 84, 120, 100, 72, 64, 112, 88, 32, 52, 108, 116, 44, 8, 92, 28, 40, 56, 68, 0, 4, 17, 89, 61, 25, 97, 105, 117, 69, 93, 101, 9, 121, 41, 85, 81, 113, 53, 33, 109, 125, 57, 65, 13, 77, 37, 21, 73, 49, 45, 1, 5, 106, 110, 126, 34, 86, 90, 70, 62, 74, 26, 98, 18, 10, 118, 38, 50, 58, 114, 122, 46, 54, 82, 42, 66, 30, 102, 94, 6, 2], 'token_total': 4096, 'token_per_expert': {31: 1, 87: 1, 119: 1, 75: 2, 91: 2, 99: 2, 35: 3, 11: 4, 83: 6, 19: 7, 79: 7, 111: 7, 103: 8, 47: 9, 95: 9, 55: 10, 15: 11, 71: 12, 123: 26, 27: 29, 43: 29, 59: 31, 63: 70, 107: 85, 7: 257, 3: 296, 16: 1, 80: 1, 104: 1, 124: 1, 20: 6, 24: 6, 60: 6, 76: 6, 36: 7, 12: 8, 84: 8, 120: 8, 100: 9, 72: 10, 64: 13, 112: 14, 88: 20, 32: 23, 52: 24, 108: 25, 116: 27, 44: 28, 8: 31, 92: 34, 28: 35, 40: 38, 56: 49, 68: 144, 0: 265, 4: 295, 17: 1, 89: 1, 61: 2, 25: 3, 97: 3, 105: 3, 117: 4, 69: 5, 93: 5, 101: 5, 9: 9, 121: 9, 41: 12, 85: 13, 81: 14, 113: 14, 53: 19, 33: 20, 109: 20, 125: 21, 57: 28, 65: 28, 13: 31, 77: 35, 37: 37, 21: 38, 73: 42, 49: 79, 45: 83, 1: 266, 5: 287, 106: 2, 110: 2, 126: 2, 34: 3, 86: 3, 90: 3, 70: 4, 62: 5, 74: 6, 26: 7, 98: 7, 18: 8, 10: 9, 118: 9, 38: 10, 50: 10, 58: 10, 114: 11, 122: 11, 46: 12, 54: 17, 82: 21, 42: 24, 66: 26, 30: 30, 102: 47, 94: 71, 6: 257, 2: 264}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.137010.137010 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.641ms | allocate_experts_across_cpu_gpu: 0.428ms
INFO 05-06 10:38:58.137622.137622 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:58.138651.138651 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005512237548828125 seconds
INFO 05-06 10:38:58.140896.140896 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001443624496459961 seconds
INFO 05-06 10:38:58.140630.140630 lmp.py:1484] [layer_moe_fused] experts compute time: 3.5762786865234375e-06 seconds
INFO 05-06 10:38:58.184188.184188 lmp.py:1496] [layer_moe_fused] to time: 0.00018668174743652344 seconds
INFO 05-06 10:38:58.184562.184562 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.04419970512390137 seconds
DEBUG 05-06 10:38:58.185746.185746 cuda_h.py:27] end *layer_moe_fused cost 49.577 ms
DEBUG 05-06 10:38:58.186120.186120 cuda_h.py:27] end prefill_layer cost 54.053 ms
DEBUG 05-06 10:38:58.186858.186858 lmp.py:841] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 10:38:58.186150.186150 lmp.py:824] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 10:38:58.189712.189712 cuda_h.py:27] end *sagl cost 2.287 ms
experts_cpu_alloc {'expert_ids': [19, 27, 99, 107, 39, 59, 23, 43, 55, 75, 87, 115, 119, 95, 31, 71, 79, 123, 127, 67, 83, 103, 111, 11, 35, 51, 3, 7, 56, 16, 60, 52, 104, 32, 64, 96, 88, 20, 80, 40, 44, 24, 12, 36, 68, 72, 84, 8, 124, 76, 92, 120, 112, 48, 100, 0, 4, 9, 17, 77, 89, 25, 113, 101, 69, 121, 125, 93, 33, 21, 81, 45, 57, 41, 13, 29, 61, 97, 37, 109, 73, 53, 105, 65, 1, 5, 54, 98, 106, 126, 38, 50, 94, 114, 74, 10, 34, 42, 58, 118, 82, 86, 30, 70, 62, 102, 90, 46, 110, 122, 18, 26, 78, 2, 6], 'token_total': 4096, 'token_per_expert': {19: 1, 27: 1, 99: 1, 107: 1, 39: 2, 59: 2, 23: 3, 43: 6, 55: 7, 75: 7, 87: 7, 115: 7, 119: 7, 95: 10, 31: 11, 71: 11, 79: 11, 123: 12, 127: 13, 67: 14, 83: 24, 103: 26, 111: 27, 11: 31, 35: 32, 51: 34, 3: 257, 7: 284, 56: 1, 16: 2, 60: 2, 52: 3, 104: 3, 32: 4, 64: 4, 96: 4, 88: 5, 20: 6, 80: 8, 40: 9, 44: 9, 24: 10, 12: 16, 36: 18, 68: 21, 72: 21, 84: 26, 8: 27, 124: 34, 76: 35, 92: 35, 120: 36, 112: 39, 48: 54, 100: 63, 0: 257, 4: 280, 9: 1, 17: 1, 77: 1, 89: 1, 25: 2, 113: 2, 101: 4, 69: 5, 121: 5, 125: 5, 93: 6, 33: 8, 21: 9, 81: 9, 45: 12, 57: 14, 41: 21, 13: 24, 29: 24, 61: 24, 97: 24, 37: 26, 109: 28, 73: 35, 53: 36, 105: 44, 65: 57, 1: 322, 5: 352, 54: 1, 98: 1, 106: 2, 126: 2, 38: 6, 50: 6, 94: 6, 114: 6, 74: 7, 10: 10, 34: 11, 42: 11, 58: 11, 118: 12, 82: 13, 86: 14, 30: 20, 70: 20, 62: 22, 102: 24, 90: 27, 46: 32, 110: 32, 122: 33, 18: 40, 26: 55, 78: 86, 2: 274, 6: 329}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.191385.191385 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.637ms | allocate_experts_across_cpu_gpu: 0.432ms
INFO 05-06 10:38:58.191066.191066 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:58.192682.192682 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005486011505126953 seconds
INFO 05-06 10:38:58.193743.193743 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013513565063476562 seconds
INFO 05-06 10:38:58.194936.194936 lmp.py:1484] [layer_moe_fused] experts compute time: 8.821487426757812e-06 seconds
INFO 05-06 10:38:58.236618.236618 lmp.py:1496] [layer_moe_fused] to time: 0.00017380714416503906 seconds
INFO 05-06 10:38:58.236919.236919 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.041555166244506836 seconds
DEBUG 05-06 10:38:58.237670.237670 cuda_h.py:27] end *layer_moe_fused cost 46.750 ms
DEBUG 05-06 10:38:58.237512.237512 cuda_h.py:27] end prefill_layer cost 51.136 ms
DEBUG 05-06 10:38:58.237898.237898 lmp.py:841] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 10:38:58.237999.237999 lmp.py:824] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 10:38:58.240200.240200 cuda_h.py:27] end *sagl cost 2.257 ms
experts_cpu_alloc {'expert_ids': [23, 71, 27, 63, 95, 39, 67, 87, 47, 51, 79, 15, 83, 107, 115, 11, 31, 99, 19, 111, 43, 75, 127, 55, 119, 59, 123, 103, 35, 3, 7, 36, 60, 96, 104, 20, 84, 112, 32, 124, 16, 48, 44, 88, 40, 8, 28, 76, 108, 120, 92, 116, 68, 24, 72, 64, 100, 4, 0, 13, 17, 61, 105, 9, 29, 65, 81, 109, 125, 57, 45, 85, 113, 41, 25, 33, 89, 101, 69, 53, 117, 73, 93, 5, 1, 14, 110, 114, 54, 62, 122, 106, 98, 10, 42, 34, 26, 102, 58, 30, 66, 118, 46, 94, 82, 90, 70, 38, 86, 74, 126, 2, 6], 'token_total': 4096, 'token_per_expert': {23: 1, 71: 1, 27: 2, 63: 2, 95: 2, 39: 3, 67: 3, 87: 4, 47: 6, 51: 6, 79: 7, 15: 9, 83: 9, 107: 11, 115: 12, 11: 14, 31: 14, 99: 15, 19: 19, 111: 23, 43: 24, 75: 32, 127: 35, 55: 36, 119: 36, 59: 37, 123: 38, 103: 47, 35: 77, 3: 257, 7: 283, 36: 1, 60: 1, 96: 1, 104: 1, 20: 2, 84: 3, 112: 6, 32: 7, 124: 9, 16: 10, 48: 10, 44: 11, 88: 14, 40: 18, 8: 21, 28: 21, 76: 26, 108: 27, 120: 29, 92: 30, 116: 32, 68: 42, 24: 58, 72: 95, 64: 105, 100: 152, 4: 256, 0: 260, 13: 1, 17: 1, 61: 1, 105: 1, 9: 2, 29: 2, 65: 3, 81: 4, 109: 5, 125: 5, 57: 6, 45: 7, 85: 7, 113: 7, 41: 9, 25: 11, 33: 12, 89: 14, 101: 14, 69: 19, 53: 31, 117: 33, 73: 47, 93: 49, 5: 257, 1: 270, 14: 1, 110: 1, 114: 1, 54: 2, 62: 3, 122: 3, 106: 4, 98: 5, 10: 6, 42: 9, 34: 10, 26: 12, 102: 12, 58: 14, 30: 15, 66: 19, 118: 19, 46: 20, 94: 21, 82: 22, 90: 28, 70: 30, 38: 31, 86: 32, 74: 64, 126: 67, 2: 257, 6: 257}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 31, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.242843.242843 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.577ms | allocate_experts_across_cpu_gpu: 0.412ms
INFO 05-06 10:38:58.242726.242726 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:58.243135.243135 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005359649658203125 seconds
INFO 05-06 10:38:58.244108.244108 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009260177612304688 seconds
INFO 05-06 10:38:58.244418.244418 lmp.py:1484] [layer_moe_fused] experts compute time: 1.430511474609375e-06 seconds
INFO 05-06 10:38:58.283928.283928 lmp.py:1496] [layer_moe_fused] to time: 0.00017404556274414062 seconds
INFO 05-06 10:38:58.283659.283659 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03898358345031738 seconds
DEBUG 05-06 10:38:58.284342.284342 cuda_h.py:27] end *layer_moe_fused cost 43.387 ms
DEBUG 05-06 10:38:58.285125.285125 cuda_h.py:27] end prefill_layer cost 47.659 ms
DEBUG 05-06 10:38:58.285995.285995 lmp.py:841] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 10:38:58.285334.285334 lmp.py:824] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 10:38:58.288015.288015 cuda_h.py:27] end *sagl cost 2.956 ms
experts_cpu_alloc {'expert_ids': [11, 23, 27, 55, 95, 99, 127, 107, 51, 75, 103, 115, 19, 71, 91, 59, 87, 31, 83, 35, 47, 123, 79, 43, 39, 67, 7, 3, 28, 88, 64, 92, 52, 12, 32, 36, 48, 68, 120, 40, 76, 60, 112, 80, 116, 24, 124, 8, 72, 16, 104, 100, 108, 84, 44, 56, 0, 4, 13, 53, 77, 101, 41, 49, 81, 57, 89, 9, 33, 73, 105, 17, 117, 109, 85, 29, 65, 37, 125, 25, 97, 61, 21, 1, 5, 50, 74, 62, 58, 82, 54, 14, 66, 110, 10, 38, 122, 34, 30, 42, 106, 26, 22, 78, 18, 90, 118, 98, 46, 86, 6, 2], 'token_total': 4096, 'token_per_expert': {11: 2, 23: 2, 27: 2, 55: 2, 95: 2, 99: 2, 127: 2, 107: 3, 51: 5, 75: 6, 103: 6, 115: 7, 19: 8, 71: 9, 91: 10, 59: 16, 87: 17, 31: 18, 83: 27, 35: 28, 47: 29, 123: 43, 79: 46, 43: 55, 39: 58, 67: 75, 7: 261, 3: 285, 28: 1, 88: 1, 64: 2, 92: 2, 52: 3, 12: 4, 32: 5, 36: 5, 48: 5, 68: 5, 120: 5, 40: 6, 76: 6, 60: 8, 112: 9, 80: 13, 116: 13, 24: 14, 124: 14, 8: 16, 72: 17, 16: 20, 104: 21, 100: 22, 108: 24, 84: 25, 44: 40, 56: 54, 0: 257, 4: 257, 13: 1, 53: 1, 77: 1, 101: 1, 41: 3, 49: 3, 81: 4, 57: 6, 89: 9, 9: 10, 33: 10, 73: 13, 105: 18, 17: 23, 117: 23, 109: 28, 85: 37, 29: 38, 65: 43, 37: 45, 125: 47, 25: 49, 97: 52, 61: 60, 21: 105, 1: 293, 5: 302, 50: 1, 74: 1, 62: 2, 58: 3, 82: 3, 54: 4, 14: 5, 66: 5, 110: 5, 10: 6, 38: 6, 122: 8, 34: 13, 30: 14, 42: 14, 106: 14, 26: 17, 22: 19, 78: 19, 18: 30, 90: 33, 118: 38, 98: 42, 46: 45, 86: 81, 6: 271, 2: 272}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.291347.291347 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.567ms | allocate_experts_across_cpu_gpu: 0.406ms
INFO 05-06 10:38:58.291846.291846 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:58.291584.291584 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005445480346679688 seconds
INFO 05-06 10:38:58.293454.293454 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001062631607055664 seconds
INFO 05-06 10:38:58.293981.293981 lmp.py:1484] [layer_moe_fused] experts compute time: 1.1920928955078125e-06 seconds
INFO 05-06 10:38:58.333534.333534 lmp.py:1496] [layer_moe_fused] to time: 0.00017142295837402344 seconds
INFO 05-06 10:38:58.333835.333835 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0400846004486084 seconds
DEBUG 05-06 10:38:58.334237.334237 cuda_h.py:27] end *layer_moe_fused cost 44.573 ms
DEBUG 05-06 10:38:58.335953.335953 cuda_h.py:27] end prefill_layer cost 49.532 ms
DEBUG 05-06 10:38:58.335347.335347 lmp.py:841] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 10:38:58.335223.335223 lmp.py:824] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 10:38:58.337022.337022 cuda_h.py:27] end *sagl cost 2.258 ms
experts_cpu_alloc {'expert_ids': [15, 51, 31, 55, 115, 47, 87, 107, 99, 127, 79, 119, 75, 111, 43, 83, 23, 35, 91, 71, 19, 67, 63, 11, 27, 3, 7, 24, 28, 80, 120, 76, 104, 112, 32, 40, 68, 84, 92, 124, 20, 96, 60, 100, 108, 8, 36, 48, 16, 12, 56, 44, 52, 64, 0, 4, 25, 65, 93, 53, 61, 57, 105, 9, 81, 13, 49, 109, 29, 37, 17, 73, 77, 45, 33, 97, 121, 5, 1, 10, 26, 78, 106, 22, 66, 126, 42, 46, 102, 38, 62, 82, 110, 86, 118, 74, 50, 122, 30, 94, 98, 114, 34, 70, 90, 2, 6], 'token_total': 4096, 'token_per_expert': {15: 1, 51: 1, 31: 2, 55: 2, 115: 2, 47: 3, 87: 4, 107: 4, 99: 5, 127: 6, 79: 7, 119: 7, 75: 8, 111: 11, 43: 16, 83: 18, 23: 23, 35: 26, 91: 28, 71: 33, 19: 37, 67: 45, 63: 54, 11: 57, 27: 71, 3: 256, 7: 270, 24: 1, 28: 2, 80: 2, 120: 2, 76: 3, 104: 3, 112: 3, 32: 4, 40: 4, 68: 4, 84: 4, 92: 5, 124: 7, 20: 8, 96: 8, 60: 10, 100: 11, 108: 11, 8: 14, 36: 18, 48: 23, 16: 27, 12: 42, 56: 42, 44: 53, 52: 66, 64: 78, 0: 256, 4: 291, 25: 2, 65: 2, 93: 2, 53: 3, 61: 4, 57: 5, 105: 5, 9: 7, 81: 8, 13: 13, 49: 14, 109: 18, 29: 19, 37: 21, 17: 23, 73: 28, 77: 28, 45: 37, 33: 60, 97: 71, 121: 120, 5: 273, 1: 282, 10: 1, 26: 2, 78: 2, 106: 2, 22: 3, 66: 3, 126: 3, 42: 4, 46: 4, 102: 5, 38: 7, 62: 12, 82: 13, 110: 13, 86: 15, 118: 16, 74: 17, 50: 18, 122: 18, 30: 21, 94: 26, 98: 31, 114: 37, 34: 45, 70: 70, 90: 105, 2: 256, 6: 303}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 23, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.339379.339379 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.569ms | allocate_experts_across_cpu_gpu: 0.401ms
INFO 05-06 10:38:58.339686.339686 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:58.340104.340104 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005261898040771484 seconds
INFO 05-06 10:38:58.342956.342956 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0010495185852050781 seconds
INFO 05-06 10:38:58.342149.342149 lmp.py:1484] [layer_moe_fused] experts compute time: 1.9073486328125e-06 seconds
INFO 05-06 10:38:58.381441.381441 lmp.py:1496] [layer_moe_fused] to time: 0.00017762184143066406 seconds
INFO 05-06 10:38:58.381994.381994 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03944826126098633 seconds
DEBUG 05-06 10:38:58.382181.382181 cuda_h.py:27] end *layer_moe_fused cost 43.907 ms
DEBUG 05-06 10:38:58.383196.383196 cuda_h.py:27] end prefill_layer cost 48.040 ms
DEBUG 05-06 10:38:58.383874.383874 lmp.py:841] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 10:38:58.383213.383213 lmp.py:824] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 10:38:58.386143.386143 cuda_h.py:27] end *sagl cost 2.230 ms
experts_cpu_alloc {'expert_ids': [27, 55, 23, 103, 47, 75, 31, 87, 119, 79, 99, 51, 43, 91, 19, 39, 63, 71, 11, 67, 83, 111, 35, 123, 107, 7, 3, 84, 76, 108, 12, 24, 92, 112, 8, 72, 88, 124, 48, 56, 120, 116, 36, 44, 100, 80, 104, 60, 64, 52, 68, 16, 4, 0, 37, 57, 65, 61, 73, 13, 33, 53, 29, 121, 9, 81, 125, 17, 49, 25, 109, 41, 77, 89, 97, 21, 93, 117, 85, 69, 45, 1, 5, 62, 94, 22, 74, 42, 122, 26, 46, 66, 126, 118, 78, 14, 50, 10, 82, 114, 34, 90, 106, 70, 18, 110, 58, 6, 2], 'token_total': 4096, 'token_per_expert': {27: 1, 55: 2, 23: 3, 103: 3, 47: 4, 75: 4, 31: 6, 87: 6, 119: 6, 79: 7, 99: 8, 51: 9, 43: 10, 91: 12, 19: 13, 39: 13, 63: 13, 71: 17, 11: 19, 67: 22, 83: 25, 111: 26, 35: 60, 123: 65, 107: 76, 7: 274, 3: 287, 84: 1, 76: 2, 108: 2, 12: 3, 24: 3, 92: 4, 112: 4, 8: 7, 72: 7, 88: 9, 124: 9, 48: 11, 56: 12, 120: 13, 116: 14, 36: 20, 44: 20, 100: 23, 80: 27, 104: 34, 60: 44, 64: 45, 52: 60, 68: 84, 16: 155, 4: 258, 0: 271, 37: 1, 57: 1, 65: 1, 61: 2, 73: 2, 13: 3, 33: 3, 53: 4, 29: 5, 121: 5, 9: 6, 81: 6, 125: 6, 17: 7, 49: 8, 25: 9, 109: 9, 41: 10, 77: 10, 89: 10, 97: 12, 21: 13, 93: 16, 117: 35, 85: 45, 69: 48, 45: 66, 1: 257, 5: 268, 62: 1, 94: 1, 22: 3, 74: 3, 42: 4, 122: 5, 26: 6, 46: 6, 66: 6, 126: 7, 118: 8, 78: 9, 14: 10, 50: 14, 10: 17, 82: 22, 114: 24, 34: 25, 90: 25, 106: 25, 70: 31, 18: 63, 110: 90, 58: 125, 6: 260, 2: 305}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.388568.388568 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.590ms | allocate_experts_across_cpu_gpu: 0.397ms
INFO 05-06 10:38:58.388511.388511 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:58.389080.389080 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005278587341308594 seconds
INFO 05-06 10:38:58.390652.390652 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009617805480957031 seconds
INFO 05-06 10:38:58.390395.390395 lmp.py:1484] [layer_moe_fused] experts compute time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:58.430962.430962 lmp.py:1496] [layer_moe_fused] to time: 0.00017452239990234375 seconds
INFO 05-06 10:38:58.430933.430933 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.039536237716674805 seconds
DEBUG 05-06 10:38:58.431225.431225 cuda_h.py:27] end *layer_moe_fused cost 43.958 ms
DEBUG 05-06 10:38:58.432420.432420 cuda_h.py:27] end prefill_layer cost 48.489 ms
DEBUG 05-06 10:38:58.432635.432635 lmp.py:841] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 10:38:58.432212.432212 lmp.py:824] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 10:38:58.434573.434573 cuda_h.py:27] end *sagl cost 2.219 ms
experts_cpu_alloc {'expert_ids': [127, 55, 31, 91, 107, 71, 47, 63, 23, 67, 35, 115, 99, 79, 19, 59, 75, 103, 51, 15, 123, 43, 27, 95, 111, 87, 7, 3, 48, 32, 40, 44, 80, 92, 96, 116, 28, 72, 108, 88, 112, 8, 68, 36, 76, 56, 124, 60, 52, 104, 24, 84, 20, 0, 4, 9, 53, 117, 13, 121, 33, 81, 109, 41, 125, 29, 105, 57, 61, 97, 45, 37, 77, 25, 49, 73, 65, 17, 89, 113, 85, 5, 1, 58, 82, 54, 122, 42, 18, 74, 50, 26, 14, 30, 38, 118, 90, 10, 66, 78, 102, 126, 86, 70, 114, 6, 2], 'token_total': 4096, 'token_per_expert': {127: 1, 55: 2, 31: 3, 91: 3, 107: 3, 71: 4, 47: 5, 63: 5, 23: 6, 67: 7, 35: 8, 115: 8, 99: 12, 79: 13, 19: 15, 59: 16, 75: 16, 103: 17, 51: 23, 15: 24, 123: 24, 43: 48, 27: 62, 95: 63, 111: 73, 87: 92, 7: 256, 3: 275, 48: 1, 32: 2, 40: 4, 44: 4, 80: 4, 92: 4, 96: 4, 116: 4, 28: 5, 72: 5, 108: 7, 88: 10, 112: 10, 8: 12, 68: 14, 36: 15, 76: 15, 56: 16, 124: 27, 60: 28, 52: 32, 104: 40, 24: 47, 84: 56, 20: 99, 0: 259, 4: 260, 9: 1, 53: 1, 117: 1, 13: 2, 121: 2, 33: 3, 81: 3, 109: 3, 41: 7, 125: 7, 29: 9, 105: 9, 57: 10, 61: 10, 97: 10, 45: 11, 37: 12, 77: 12, 25: 13, 49: 18, 73: 28, 65: 60, 17: 74, 89: 77, 113: 106, 85: 142, 5: 259, 1: 262, 58: 1, 82: 1, 54: 3, 122: 3, 42: 4, 18: 7, 74: 8, 50: 9, 26: 11, 14: 12, 30: 12, 38: 12, 118: 15, 90: 16, 10: 18, 66: 19, 78: 22, 102: 25, 126: 25, 86: 27, 70: 28, 114: 81, 6: 256, 2: 261}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.436058.436058 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.583ms | allocate_experts_across_cpu_gpu: 0.391ms
INFO 05-06 10:38:58.436286.436286 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.86102294921875e-06 seconds
INFO 05-06 10:38:58.437722.437722 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005245208740234375 seconds
INFO 05-06 10:38:58.438284.438284 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009157657623291016 seconds
INFO 05-06 10:38:58.438185.438185 lmp.py:1484] [layer_moe_fused] experts compute time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:58.477604.477604 lmp.py:1496] [layer_moe_fused] to time: 0.00017881393432617188 seconds
INFO 05-06 10:38:58.477972.477972 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03908872604370117 seconds
DEBUG 05-06 10:38:58.478606.478606 cuda_h.py:27] end *layer_moe_fused cost 43.057 ms
DEBUG 05-06 10:38:58.479846.479846 cuda_h.py:27] end prefill_layer cost 47.192 ms
DEBUG 05-06 10:38:58.479769.479769 lmp.py:841] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 10:38:58.479870.479870 lmp.py:824] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 10:38:58.482079.482079 cuda_h.py:27] end *sagl cost 2.255 ms
experts_cpu_alloc {'expert_ids': [11, 63, 71, 19, 59, 67, 15, 39, 47, 91, 27, 23, 127, 83, 75, 35, 31, 119, 79, 111, 51, 123, 103, 115, 95, 87, 43, 7, 3, 44, 60, 72, 84, 124, 80, 32, 116, 68, 40, 20, 96, 112, 28, 56, 8, 64, 108, 12, 36, 48, 76, 24, 88, 120, 100, 0, 4, 57, 69, 93, 101, 81, 97, 89, 113, 125, 29, 21, 49, 105, 85, 41, 121, 53, 61, 109, 13, 37, 25, 33, 45, 65, 5, 1, 10, 22, 30, 126, 86, 26, 110, 74, 58, 90, 54, 106, 122, 66, 94, 114, 42, 118, 18, 78, 98, 62, 70, 46, 14, 82, 50, 2, 6], 'token_total': 4096, 'token_per_expert': {11: 1, 63: 1, 71: 1, 19: 3, 59: 3, 67: 3, 15: 4, 39: 4, 47: 4, 91: 6, 27: 7, 23: 10, 127: 11, 83: 12, 75: 13, 35: 18, 31: 21, 119: 22, 79: 29, 111: 30, 51: 32, 123: 39, 103: 46, 115: 50, 95: 57, 87: 68, 43: 69, 7: 277, 3: 291, 44: 1, 60: 1, 72: 1, 84: 1, 124: 3, 80: 4, 32: 5, 116: 5, 68: 6, 40: 7, 20: 8, 96: 8, 112: 12, 28: 16, 56: 17, 8: 19, 64: 19, 108: 20, 12: 21, 36: 33, 48: 36, 76: 44, 24: 47, 88: 47, 120: 60, 100: 61, 0: 261, 4: 270, 57: 1, 69: 1, 93: 1, 101: 1, 81: 2, 97: 2, 89: 3, 113: 3, 125: 4, 29: 5, 21: 7, 49: 7, 105: 11, 85: 12, 41: 18, 121: 19, 53: 21, 61: 23, 109: 26, 13: 28, 37: 28, 25: 30, 33: 38, 45: 46, 65: 50, 5: 257, 1: 296, 10: 1, 22: 1, 30: 1, 126: 1, 86: 2, 26: 3, 110: 3, 74: 6, 58: 10, 90: 10, 54: 11, 106: 12, 122: 12, 66: 16, 94: 17, 114: 19, 42: 21, 118: 21, 18: 22, 78: 22, 98: 23, 62: 27, 70: 28, 46: 30, 14: 32, 82: 54, 50: 70, 2: 256, 6: 260}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 28, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 29, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.484413.484413 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.608ms | allocate_experts_across_cpu_gpu: 0.403ms
INFO 05-06 10:38:58.484780.484780 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:58.485877.485877 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005414485931396484 seconds
INFO 05-06 10:38:58.486643.486643 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0010273456573486328 seconds
INFO 05-06 10:38:58.486057.486057 lmp.py:1484] [layer_moe_fused] experts compute time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:58.525671.525671 lmp.py:1496] [layer_moe_fused] to time: 0.00017380714416503906 seconds
INFO 05-06 10:38:58.526925.526925 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0387270450592041 seconds
DEBUG 05-06 10:38:58.527760.527760 cuda_h.py:27] end *layer_moe_fused cost 43.824 ms
DEBUG 05-06 10:38:58.527670.527670 cuda_h.py:27] end prefill_layer cost 48.229 ms
DEBUG 05-06 10:38:58.527202.527202 lmp.py:841] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 10:38:58.527111.527111 lmp.py:824] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 10:38:58.530407.530407 cuda_h.py:27] end *sagl cost 2.251 ms
experts_cpu_alloc {'expert_ids': [19, 59, 83, 127, 15, 67, 87, 79, 23, 39, 43, 95, 123, 55, 119, 47, 71, 11, 75, 91, 115, 111, 7, 3, 8, 28, 80, 120, 44, 108, 36, 92, 48, 24, 88, 84, 100, 60, 104, 68, 32, 52, 40, 76, 112, 20, 12, 0, 4, 17, 21, 29, 93, 109, 81, 9, 33, 73, 117, 65, 69, 105, 37, 121, 97, 89, 101, 85, 77, 13, 53, 113, 57, 49, 1, 5, 34, 58, 118, 38, 66, 50, 54, 94, 18, 62, 122, 98, 126, 106, 74, 30, 78, 46, 70, 22, 110, 90, 2, 6], 'token_total': 4096, 'token_per_expert': {19: 1, 59: 1, 83: 1, 127: 1, 15: 2, 67: 2, 87: 2, 79: 5, 23: 6, 39: 6, 43: 8, 95: 13, 123: 13, 55: 18, 119: 24, 47: 31, 71: 33, 11: 39, 75: 43, 91: 45, 115: 82, 111: 196, 7: 256, 3: 258, 8: 1, 28: 1, 80: 1, 120: 2, 44: 3, 108: 3, 36: 4, 92: 4, 48: 5, 24: 6, 88: 6, 84: 9, 100: 9, 60: 12, 104: 15, 68: 25, 32: 33, 52: 38, 40: 44, 76: 62, 112: 66, 20: 145, 12: 190, 0: 256, 4: 256, 17: 1, 21: 1, 29: 1, 93: 1, 109: 2, 81: 4, 9: 5, 33: 6, 73: 6, 117: 7, 65: 9, 69: 9, 105: 9, 37: 10, 121: 10, 97: 11, 89: 16, 101: 17, 85: 18, 77: 22, 13: 24, 53: 31, 113: 31, 57: 90, 49: 130, 1: 268, 5: 271, 34: 1, 58: 1, 118: 1, 38: 2, 66: 2, 50: 3, 54: 3, 94: 4, 18: 7, 62: 7, 122: 7, 98: 9, 126: 9, 106: 11, 74: 14, 30: 18, 78: 20, 46: 24, 70: 30, 22: 32, 110: 38, 90: 48, 2: 256, 6: 257}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 27, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 24, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.532056.532056 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.605ms | allocate_experts_across_cpu_gpu: 0.371ms
INFO 05-06 10:38:58.532702.532702 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:58.533219.533219 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005297660827636719 seconds
INFO 05-06 10:38:58.534878.534878 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009889602661132812 seconds
INFO 05-06 10:38:58.535822.535822 lmp.py:1484] [layer_moe_fused] experts compute time: 4.5299530029296875e-06 seconds
INFO 05-06 10:38:58.573994.573994 lmp.py:1496] [layer_moe_fused] to time: 0.000171661376953125 seconds
INFO 05-06 10:38:58.574441.574441 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.03859448432922363 seconds
DEBUG 05-06 10:38:58.575502.575502 cuda_h.py:27] end *layer_moe_fused cost 43.545 ms
DEBUG 05-06 10:38:58.575492.575492 cuda_h.py:27] end prefill_layer cost 47.951 ms
DEBUG 05-06 10:38:58.575468.575468 lmp.py:841] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 10:38:58.575761.575761 lmp.py:824] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 10:38:58.579442.579442 cuda_h.py:27] end *sagl cost 2.915 ms
experts_cpu_alloc {'expert_ids': [55, 127, 87, 111, 83, 11, 31, 75, 115, 123, 63, 35, 119, 95, 15, 107, 67, 27, 23, 71, 19, 43, 99, 91, 3, 7, 108, 40, 76, 88, 96, 92, 80, 84, 32, 8, 24, 116, 120, 44, 124, 48, 60, 16, 56, 28, 20, 64, 52, 0, 4, 17, 125, 25, 33, 37, 105, 13, 65, 21, 77, 9, 81, 85, 89, 61, 69, 109, 101, 73, 113, 93, 29, 49, 53, 97, 121, 117, 57, 5, 1, 38, 74, 98, 34, 70, 118, 122, 126, 58, 94, 10, 50, 46, 114, 66, 78, 30, 62, 14, 26, 18, 54, 82, 22, 90, 86, 42, 106, 6, 2], 'token_total': 4096, 'token_per_expert': {55: 2, 127: 2, 87: 3, 111: 4, 83: 5, 11: 6, 31: 6, 75: 6, 115: 6, 123: 6, 63: 7, 35: 8, 119: 9, 95: 10, 15: 11, 107: 12, 67: 14, 27: 26, 23: 32, 71: 35, 19: 39, 43: 48, 99: 84, 91: 87, 3: 269, 7: 354, 108: 1, 40: 3, 76: 3, 88: 3, 96: 3, 92: 4, 80: 5, 84: 5, 32: 6, 8: 8, 24: 9, 116: 10, 120: 14, 44: 16, 124: 18, 48: 24, 60: 30, 16: 36, 56: 42, 28: 44, 20: 47, 64: 69, 52: 71, 0: 259, 4: 305, 17: 2, 125: 3, 25: 4, 33: 4, 37: 4, 105: 4, 13: 5, 65: 5, 21: 9, 77: 13, 9: 14, 81: 14, 85: 14, 89: 15, 61: 16, 69: 16, 109: 16, 101: 17, 73: 18, 113: 19, 93: 23, 29: 24, 49: 24, 53: 25, 97: 31, 121: 41, 117: 42, 57: 44, 5: 256, 1: 268, 38: 1, 74: 1, 98: 1, 34: 2, 70: 2, 118: 2, 122: 2, 126: 2, 58: 4, 94: 5, 10: 6, 50: 7, 46: 9, 114: 9, 66: 13, 78: 17, 30: 18, 62: 19, 14: 21, 26: 22, 18: 25, 54: 25, 82: 26, 22: 33, 90: 34, 86: 42, 42: 47, 106: 49, 6: 264, 2: 272}}
experts_gpu_alloc_device_0 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 26, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_1 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 25, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_2 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_3 {'expert_ids': [], 'expert_count': 0, 'ideal_gpu_count': 0, 'keep_on_gpu': 0, 'hit_count_on_device': 30, 'token_total': 0, 'token_per_expert': {}}
INFO 05-06 10:38:58.581409.581409 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.567ms | allocate_experts_across_cpu_gpu: 0.398ms
INFO 05-06 10:38:58.581101.581101 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-06 seconds
INFO 05-06 10:38:58.582367.582367 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005288124084472656 seconds
INFO 05-06 10:38:58.583376.583376 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0009174346923828125 seconds
INFO 05-06 10:38:58.583245.583245 lmp.py:1484] [layer_moe_fused] experts compute time: 2.6226043701171875e-06 seconds
INFO 05-06 10:38:58.622507.622507 lmp.py:1496] [layer_moe_fused] to time: 0.00017714500427246094 seconds
INFO 05-06 10:38:58.623569.623569 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.038953304290771484 seconds
DEBUG 05-06 10:38:58.624128.624128 cuda_h.py:27] end *layer_moe_fused cost 43.769 ms
DEBUG 05-06 10:38:58.624029.624029 cuda_h.py:27] end prefill_layer cost 48.749 ms
DEBUG 05-06 10:38:58.624423.624423 lmp.py:841] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 10:38:58.624683.624683 cuda_h.py:27] end prefill_step cost 1514.352 ms
INFO 05-06 10:38:58.624253.624253 lmp.py:843] prefill time: 1.6394257545471191 seconds
WARNING 05-06 10:38:58.625911.625911 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:38:58.625831.625831 helper.py:35]   NaN count (hidden): 720896
WARNING 05-06 10:38:58.626976.626976 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:38:58.626921.626921 helper.py:39]   NaN count (normed): 720896
WARNING 05-06 10:38:58.631172.631172 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:38:58.631283.631283 helper.py:50]   NaN count: 524288
WARNING 05-06 10:38:58.632892.632892 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:38:58.634217.634217 cuda_h.py:27] end init_inputs_tokens cost 8.765 ms
DEBUG 05-06 10:38:58.634457.634457 lmp.py:899] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:38:58.634853.634853 lmp.py:904] ---- decode step 0 layer 0 ----
DEBUG 05-06 10:38:58.636596.636596 cuda_h.py:27] end *sagl cost 2.178 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 47, 79, 83, 87, 91, 103, 127], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 14, 'token_per_expert': {15: 1, 47: 2, 79: 3, 83: 1, 87: 1, 91: 1, 103: 4, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 52, 108, 116], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {8: 1, 52: 1, 108: 1, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [53, 81, 121], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {53: 2, 81: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 22, 26, 78, 90, 106, 114], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {18: 1, 22: 1, 26: 2, 78: 1, 90: 2, 106: 1, 114: 2}}
INFO 05-06 10:38:58.638121.638121 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.497ms | allocate_experts_across_cpu_gpu: 0.175ms
INFO 05-06 10:38:58.638099.638099 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.100799560546875e-05 seconds
INFO 05-06 10:38:58.640732.640732 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014812946319580078 seconds
INFO 05-06 10:38:58.643886.643886 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.002956390380859375 seconds
INFO 05-06 10:38:58.644441.644441 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000667572021484375 seconds
INFO 05-06 10:38:58.646635.646635 mlpmodule.py:2799] [fused_experts] gmm total=1.418ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.646588.646588 mlpmodule.py:2799] [fused_experts] gmm total=1.747ms E=32 S=14 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.646171.646171 mlpmodule.py:2799] [fused_experts] gmm total=1.720ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.646049.646049 mlpmodule.py:2799] [fused_experts] gmm total=1.734ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.647735.647735 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003369569778442383 seconds
INFO 05-06 10:38:58.647436.647436 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.888938903808594e-05 seconds
DEBUG 05-06 10:38:58.648182.648182 cuda_h.py:27] end *layer_moe_fused cost 10.412 ms
DEBUG 05-06 10:38:58.648954.648954 cuda_h.py:27] end decode_layer cost 14.289 ms
DEBUG 05-06 10:38:58.649665.649665 lmp.py:904] ---- decode step 0 layer 1 ----
DEBUG 05-06 10:38:58.651149.651149 cuda_h.py:27] end *sagl cost 1.957 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 67, 107], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {3: 1, 7: 1, 67: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 56, 84, 92, 96, 124], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {0: 4, 4: 1, 56: 2, 84: 1, 92: 1, 96: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 97, 117, 121], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {1: 1, 5: 1, 97: 2, 117: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 30, 46, 54, 106, 110], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 6: 2, 22: 1, 30: 3, 46: 1, 54: 1, 106: 1, 110: 1}}
INFO 05-06 10:38:58.652504.652504 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.410ms | allocate_experts_across_cpu_gpu: 0.160ms
INFO 05-06 10:38:58.652700.652700 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.266334533691406e-05 seconds
INFO 05-06 10:38:58.653223.653223 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006053447723388672 seconds
INFO 05-06 10:38:58.655954.655954 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001768350601196289 seconds
INFO 05-06 10:38:58.656943.656943 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005831718444824219 seconds
INFO 05-06 10:38:58.657780.657780 mlpmodule.py:2799] [fused_experts] gmm total=1.099ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.657791.657791 mlpmodule.py:2799] [fused_experts] gmm total=1.174ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.657003.657003 mlpmodule.py:2799] [fused_experts] gmm total=1.295ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.657342.657342 mlpmodule.py:2799] [fused_experts] gmm total=1.493ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.658228.658228 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0025200843811035156 seconds
INFO 05-06 10:38:58.658399.658399 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.698204040527344e-05 seconds
DEBUG 05-06 10:38:58.659503.659503 cuda_h.py:27] end *layer_moe_fused cost 7.279 ms
DEBUG 05-06 10:38:58.659904.659904 cuda_h.py:27] end decode_layer cost 10.916 ms
DEBUG 05-06 10:38:58.660854.660854 lmp.py:904] ---- decode step 0 layer 2 ----
DEBUG 05-06 10:38:58.661984.661984 cuda_h.py:27] end *sagl cost 1.873 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 119, 123], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 1, 7: 1, 11: 3, 119: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 52, 60, 76, 120], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {0: 1, 4: 2, 24: 1, 52: 1, 60: 1, 76: 3, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 41, 49, 81], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {1: 1, 5: 1, 41: 2, 49: 3, 81: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 62, 78, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {2: 1, 6: 1, 62: 1, 78: 1, 126: 2}}
INFO 05-06 10:38:58.663090.663090 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.393ms | allocate_experts_across_cpu_gpu: 0.144ms
INFO 05-06 10:38:58.663823.663823 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.3392181396484375e-05 seconds
INFO 05-06 10:38:58.664101.664101 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005471706390380859 seconds
INFO 05-06 10:38:58.665828.665828 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001657247543334961 seconds
INFO 05-06 10:38:58.666510.666510 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005326271057128906 seconds
INFO 05-06 10:38:58.667815.667815 mlpmodule.py:2799] [fused_experts] gmm total=1.147ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.668866.668866 mlpmodule.py:2799] [fused_experts] gmm total=1.339ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.668898.668898 mlpmodule.py:2799] [fused_experts] gmm total=1.330ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.668199.668199 mlpmodule.py:2799] [fused_experts] gmm total=1.461ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.669991.669991 lmp.py:1484] [layer_moe_fused] experts compute time: 0.002604246139526367 seconds
INFO 05-06 10:38:58.669531.669531 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.14984130859375e-05 seconds
DEBUG 05-06 10:38:58.669112.669112 cuda_h.py:27] end *layer_moe_fused cost 6.911 ms
DEBUG 05-06 10:38:58.670916.670916 cuda_h.py:27] end decode_layer cost 10.364 ms
DEBUG 05-06 10:38:58.670435.670435 lmp.py:904] ---- decode step 0 layer 3 ----
DEBUG 05-06 10:38:58.672250.672250 cuda_h.py:27] end *sagl cost 1.957 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 63, 67, 107], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {3: 1, 7: 1, 63: 1, 67: 2, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 48, 84, 96, 104, 116], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {0: 1, 4: 1, 48: 1, 84: 1, 96: 3, 104: 3, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 109, 117], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {1: 1, 5: 1, 33: 1, 109: 1, 117: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 34, 118], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {2: 3, 6: 1, 22: 1, 26: 1, 34: 1, 118: 3}}
INFO 05-06 10:38:58.673748.673748 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.314ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 10:38:58.673241.673241 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:38:58.674502.674502 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006012916564941406 seconds
INFO 05-06 10:38:58.676940.676940 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001356363296508789 seconds
INFO 05-06 10:38:58.677098.677098 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008955001831054688 seconds
INFO 05-06 10:38:58.678698.678698 mlpmodule.py:2799] [fused_experts] gmm total=0.933ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.678099.678099 mlpmodule.py:2799] [fused_experts] gmm total=0.951ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.678341.678341 mlpmodule.py:2799] [fused_experts] gmm total=1.103ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.678125.678125 mlpmodule.py:2799] [fused_experts] gmm total=1.274ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.679295.679295 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0020906925201416016 seconds
INFO 05-06 10:38:58.679544.679544 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.649162292480469e-05 seconds
DEBUG 05-06 10:38:58.679318.679318 cuda_h.py:27] end *layer_moe_fused cost 6.347 ms
DEBUG 05-06 10:38:58.680569.680569 cuda_h.py:27] end decode_layer cost 9.751 ms
DEBUG 05-06 10:38:58.680459.680459 lmp.py:904] ---- decode step 0 layer 4 ----
DEBUG 05-06 10:38:58.681688.681688 cuda_h.py:27] end *sagl cost 1.545 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 43, 51, 83], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 43: 1, 51: 1, 83: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 60, 84], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 20: 1, 24: 1, 60: 2, 84: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 25, 45], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 3, 17: 1, 21: 1, 25: 1, 45: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 82, 126], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 82: 1, 126: 2}}
INFO 05-06 10:38:58.683400.683400 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.307ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:38:58.683839.683839 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 10:38:58.684628.684628 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008175373077392578 seconds
INFO 05-06 10:38:58.685348.685348 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001283407211303711 seconds
INFO 05-06 10:38:58.686479.686479 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008771419525146484 seconds
INFO 05-06 10:38:58.688448.688448 mlpmodule.py:2799] [fused_experts] gmm total=1.063ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.688261.688261 mlpmodule.py:2799] [fused_experts] gmm total=1.255ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.688407.688407 mlpmodule.py:2799] [fused_experts] gmm total=2.209ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.689275.689275 mlpmodule.py:2799] [fused_experts] gmm total=2.339ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.690185.690185 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003409862518310547 seconds
INFO 05-06 10:38:58.690507.690507 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 10:38:58.690644.690644 cuda_h.py:27] end *layer_moe_fused cost 7.882 ms
DEBUG 05-06 10:38:58.691266.691266 cuda_h.py:27] end decode_layer cost 10.842 ms
DEBUG 05-06 10:38:58.691387.691387 lmp.py:904] ---- decode step 0 layer 5 ----
DEBUG 05-06 10:38:58.692917.692917 cuda_h.py:27] end *sagl cost 1.546 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 79, 99, 119], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 2, 31: 1, 39: 2, 79: 1, 99: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 52], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 36: 2, 52: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 61, 101], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {1: 2, 5: 3, 29: 1, 61: 1, 101: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 70, 74], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 22: 1, 70: 1, 74: 1}}
INFO 05-06 10:38:58.694787.694787 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.306ms | allocate_experts_across_cpu_gpu: 0.089ms
INFO 05-06 10:38:58.694896.694896 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:38:58.695569.695569 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012655258178710938 seconds
INFO 05-06 10:38:58.696734.696734 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013306140899658203 seconds
INFO 05-06 10:38:58.698287.698287 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012123584747314453 seconds
INFO 05-06 10:38:58.699383.699383 mlpmodule.py:2799] [fused_experts] gmm total=1.203ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.699903.699903 mlpmodule.py:2799] [fused_experts] gmm total=1.425ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.699296.699296 mlpmodule.py:2799] [fused_experts] gmm total=1.615ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.701532.701532 mlpmodule.py:2799] [fused_experts] gmm total=2.549ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.701951.701951 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0035486221313476562 seconds
INFO 05-06 10:38:58.701532.701532 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.2928924560546875e-05 seconds
DEBUG 05-06 10:38:58.702597.702597 cuda_h.py:27] end *layer_moe_fused cost 8.753 ms
DEBUG 05-06 10:38:58.702975.702975 cuda_h.py:27] end decode_layer cost 11.629 ms
DEBUG 05-06 10:38:58.702480.702480 lmp.py:904] ---- decode step 0 layer 6 ----
DEBUG 05-06 10:38:58.704897.704897 cuda_h.py:27] end *sagl cost 1.532 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 67, 87, 91], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 35: 1, 67: 1, 87: 2, 91: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 64, 100], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 20: 1, 64: 1, 100: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 53], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 3, 5: 2, 13: 1, 53: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 42, 58, 106, 110], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {2: 2, 6: 2, 42: 1, 58: 1, 106: 2, 110: 1}}
INFO 05-06 10:38:58.705111.705111 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.304ms | allocate_experts_across_cpu_gpu: 0.092ms
INFO 05-06 10:38:58.705081.705081 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2411346435546875e-05 seconds
INFO 05-06 10:38:58.707316.707316 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012617111206054688 seconds
INFO 05-06 10:38:58.708382.708382 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013267993927001953 seconds
INFO 05-06 10:38:58.709591.709591 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014233589172363281 seconds
INFO 05-06 10:38:58.711061.711061 mlpmodule.py:2799] [fused_experts] gmm total=1.413ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.711005.711005 mlpmodule.py:2799] [fused_experts] gmm total=1.664ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.711620.711620 mlpmodule.py:2799] [fused_experts] gmm total=1.771ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.712074.712074 mlpmodule.py:2799] [fused_experts] gmm total=2.562ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.713970.713970 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0035429000854492188 seconds
INFO 05-06 10:38:58.713174.713174 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.3882598876953125e-05 seconds
DEBUG 05-06 10:38:58.714350.714350 cuda_h.py:27] end *layer_moe_fused cost 8.931 ms
DEBUG 05-06 10:38:58.714118.714118 cuda_h.py:27] end decode_layer cost 11.736 ms
DEBUG 05-06 10:38:58.714677.714677 lmp.py:904] ---- decode step 0 layer 7 ----
DEBUG 05-06 10:38:58.716524.716524 cuda_h.py:27] end *sagl cost 1.534 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 83], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 83: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 72, 80, 84], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 20: 1, 72: 1, 80: 2, 84: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 69, 97, 113, 121], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {1: 2, 5: 2, 69: 2, 97: 1, 113: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 90, 106, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 90: 1, 106: 1, 126: 1}}
INFO 05-06 10:38:58.717910.717910 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.293ms | allocate_experts_across_cpu_gpu: 0.085ms
INFO 05-06 10:38:58.717535.717535 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6927719116210938e-05 seconds
INFO 05-06 10:38:58.718458.718458 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014193058013916016 seconds
INFO 05-06 10:38:58.720462.720462 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014519691467285156 seconds
INFO 05-06 10:38:58.721406.721406 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014405250549316406 seconds
INFO 05-06 10:38:58.724396.724396 mlpmodule.py:2799] [fused_experts] gmm total=2.061ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.724389.724389 mlpmodule.py:2799] [fused_experts] gmm total=2.156ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.724105.724105 mlpmodule.py:2799] [fused_experts] gmm total=2.277ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.724156.724156 mlpmodule.py:2799] [fused_experts] gmm total=2.308ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.726312.726312 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0044231414794921875 seconds
INFO 05-06 10:38:58.726291.726291 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.318092346191406e-05 seconds
DEBUG 05-06 10:38:58.726240.726240 cuda_h.py:27] end *layer_moe_fused cost 9.857 ms
DEBUG 05-06 10:38:58.727810.727810 cuda_h.py:27] end decode_layer cost 12.662 ms
DEBUG 05-06 10:38:58.727646.727646 lmp.py:904] ---- decode step 0 layer 8 ----
DEBUG 05-06 10:38:58.728485.728485 cuda_h.py:27] end *sagl cost 1.493 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 51, 71, 87], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 15: 1, 27: 1, 51: 1, 71: 1, 87: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 32, 64], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {0: 2, 4: 2, 28: 1, 32: 1, 64: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 65, 73, 125], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 33: 1, 65: 1, 73: 1, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 54, 110], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {2: 2, 6: 3, 54: 1, 110: 1}}
INFO 05-06 10:38:58.730533.730533 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.302ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:38:58.730125.730125 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-05 seconds
INFO 05-06 10:38:58.731023.731023 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014522075653076172 seconds
INFO 05-06 10:38:58.733284.733284 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001539468765258789 seconds
INFO 05-06 10:38:58.735968.735968 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015475749969482422 seconds
INFO 05-06 10:38:58.737224.737224 mlpmodule.py:2799] [fused_experts] gmm total=2.136ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.737747.737747 mlpmodule.py:2799] [fused_experts] gmm total=2.210ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.737562.737562 mlpmodule.py:2799] [fused_experts] gmm total=2.325ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.738043.738043 mlpmodule.py:2799] [fused_experts] gmm total=2.349ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.739606.739606 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004462242126464844 seconds
INFO 05-06 10:38:58.739518.739518 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.125999450683594e-05 seconds
DEBUG 05-06 10:38:58.740258.740258 cuda_h.py:27] end *layer_moe_fused cost 10.584 ms
DEBUG 05-06 10:38:58.740404.740404 cuda_h.py:27] end decode_layer cost 13.335 ms
DEBUG 05-06 10:38:58.740717.740717 lmp.py:904] ---- decode step 0 layer 9 ----
DEBUG 05-06 10:38:58.742252.742252 cuda_h.py:27] end *sagl cost 1.516 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 95], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {3: 2, 7: 3, 19: 1, 95: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 48, 76], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 48: 1, 76: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 69, 89, 101], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 37: 1, 69: 1, 89: 1, 101: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 70, 74, 102, 106], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 2, 38: 1, 70: 1, 74: 1, 102: 1, 106: 1}}
INFO 05-06 10:38:58.743481.743481 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.304ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 10:38:58.743497.743497 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-05 seconds
INFO 05-06 10:38:58.745626.745626 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014281272888183594 seconds
INFO 05-06 10:38:58.746340.746340 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013077259063720703 seconds
INFO 05-06 10:38:58.748655.748655 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014331340789794922 seconds
INFO 05-06 10:38:58.750217.750217 mlpmodule.py:2799] [fused_experts] gmm total=2.118ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.750932.750932 mlpmodule.py:2799] [fused_experts] gmm total=2.197ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.750808.750808 mlpmodule.py:2799] [fused_experts] gmm total=2.377ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.750010.750010 mlpmodule.py:2799] [fused_experts] gmm total=2.388ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.752842.752842 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00431513786315918 seconds
INFO 05-06 10:38:58.752377.752377 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.1975250244140625e-05 seconds
DEBUG 05-06 10:38:58.752612.752612 cuda_h.py:27] end *layer_moe_fused cost 9.671 ms
DEBUG 05-06 10:38:58.753287.753287 cuda_h.py:27] end decode_layer cost 12.669 ms
DEBUG 05-06 10:38:58.753977.753977 lmp.py:904] ---- decode step 0 layer 10 ----
DEBUG 05-06 10:38:58.755209.755209 cuda_h.py:27] end *sagl cost 1.972 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {3: 2, 7: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 28, 44, 60, 76, 88], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {0: 2, 4: 2, 8: 1, 28: 2, 44: 1, 60: 2, 76: 1, 88: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 81, 121], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 37: 2, 81: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 46, 62], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 2, 6: 2, 14: 1, 18: 1, 46: 1, 62: 1}}
INFO 05-06 10:38:58.756677.756677 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.326ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:38:58.757747.757747 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-05 seconds
INFO 05-06 10:38:58.758604.758604 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014350414276123047 seconds
INFO 05-06 10:38:58.760395.760395 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013942718505859375 seconds
INFO 05-06 10:38:58.761973.761973 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012018680572509766 seconds
INFO 05-06 10:38:58.763407.763407 mlpmodule.py:2799] [fused_experts] gmm total=1.868ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.763190.763190 mlpmodule.py:2799] [fused_experts] gmm total=2.193ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.764753.764753 mlpmodule.py:2799] [fused_experts] gmm total=2.297ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.764376.764376 mlpmodule.py:2799] [fused_experts] gmm total=2.408ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.765721.765721 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003862619400024414 seconds
INFO 05-06 10:38:58.765969.765969 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.7206878662109375e-05 seconds
DEBUG 05-06 10:38:58.765557.765557 cuda_h.py:27] end *layer_moe_fused cost 9.294 ms
DEBUG 05-06 10:38:58.766742.766742 cuda_h.py:27] end decode_layer cost 12.742 ms
DEBUG 05-06 10:38:58.766532.766532 lmp.py:904] ---- decode step 0 layer 11 ----
DEBUG 05-06 10:38:58.767321.767321 cuda_h.py:27] end *sagl cost 1.561 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 83], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 23: 1, 83: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 32, 56, 92], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 2, 4: 2, 16: 1, 32: 1, 56: 1, 92: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 49, 81, 117], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 17: 1, 49: 1, 81: 1, 117: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 66, 102], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 2, 6: 2, 38: 1, 46: 1, 66: 1, 102: 1}}
INFO 05-06 10:38:58.769436.769436 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.315ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:38:58.769644.769644 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:38:58.770806.770806 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001417398452758789 seconds
INFO 05-06 10:38:58.772699.772699 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012984275817871094 seconds
INFO 05-06 10:38:58.773032.773032 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001361846923828125 seconds
INFO 05-06 10:38:58.775986.775986 mlpmodule.py:2799] [fused_experts] gmm total=1.697ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.776698.776698 mlpmodule.py:2799] [fused_experts] gmm total=2.087ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.776161.776161 mlpmodule.py:2799] [fused_experts] gmm total=2.275ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.776239.776239 mlpmodule.py:2799] [fused_experts] gmm total=2.526ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.777531.777531 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003871440887451172 seconds
INFO 05-06 10:38:58.777796.777796 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 7.939338684082031e-05 seconds
DEBUG 05-06 10:38:58.778132.778132 cuda_h.py:27] end *layer_moe_fused cost 9.516 ms
DEBUG 05-06 10:38:58.779184.779184 cuda_h.py:27] end decode_layer cost 12.721 ms
DEBUG 05-06 10:38:58.779558.779558 lmp.py:904] ---- decode step 0 layer 12 ----
DEBUG 05-06 10:38:58.782392.782392 cuda_h.py:27] end *sagl cost 2.892 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 39, 71, 95], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 15: 1, 19: 1, 39: 1, 71: 1, 95: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 40, 80, 84, 116], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 36: 1, 40: 1, 80: 1, 84: 1, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 45, 49, 101, 117], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 45: 1, 49: 1, 101: 1, 117: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 78], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {2: 2, 6: 2, 78: 2}}
INFO 05-06 10:38:58.784030.784030 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.503ms | allocate_experts_across_cpu_gpu: 0.122ms
INFO 05-06 10:38:58.784695.784695 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.956390380859375e-05 seconds
INFO 05-06 10:38:58.785040.785040 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013315677642822266 seconds
INFO 05-06 10:38:58.787004.787004 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00162506103515625 seconds
INFO 05-06 10:38:58.788132.788132 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013751983642578125 seconds
INFO 05-06 10:38:58.791685.791685 mlpmodule.py:2799] [fused_experts] gmm total=2.123ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.791523.791523 mlpmodule.py:2799] [fused_experts] gmm total=2.226ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.791874.791874 mlpmodule.py:2799] [fused_experts] gmm total=2.339ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.792301.792301 mlpmodule.py:2799] [fused_experts] gmm total=2.610ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.793569.793569 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004403114318847656 seconds
INFO 05-06 10:38:58.793706.793706 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.626678466796875e-05 seconds
DEBUG 05-06 10:38:58.794824.794824 cuda_h.py:27] end *layer_moe_fused cost 10.308 ms
DEBUG 05-06 10:38:58.794059.794059 cuda_h.py:27] end decode_layer cost 15.361 ms
DEBUG 05-06 10:38:58.794816.794816 lmp.py:904] ---- decode step 0 layer 13 ----
DEBUG 05-06 10:38:58.796981.796981 cuda_h.py:27] end *sagl cost 1.900 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 55, 79, 99, 107], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 2, 31: 1, 55: 1, 79: 2, 99: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 100, 104], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 100: 2, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 73, 113, 121], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {1: 3, 5: 2, 73: 1, 113: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 78, 110], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 78: 2, 110: 1}}
INFO 05-06 10:38:58.798602.798602 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.372ms | allocate_experts_across_cpu_gpu: 0.114ms
INFO 05-06 10:38:58.798877.798877 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.5272369384765625e-05 seconds
INFO 05-06 10:38:58.799391.799391 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00145721435546875 seconds
INFO 05-06 10:38:58.801360.801360 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0015876293182373047 seconds
INFO 05-06 10:38:58.802436.802436 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015802383422851562 seconds
INFO 05-06 10:38:58.805370.805370 mlpmodule.py:2799] [fused_experts] gmm total=2.124ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.805349.805349 mlpmodule.py:2799] [fused_experts] gmm total=2.378ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.805052.805052 mlpmodule.py:2799] [fused_experts] gmm total=2.416ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.806957.806957 mlpmodule.py:2799] [fused_experts] gmm total=2.419ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.807713.807713 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004379749298095703 seconds
INFO 05-06 10:38:58.807737.807737 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.5789947509765625e-05 seconds
DEBUG 05-06 10:38:58.807455.807455 cuda_h.py:27] end *layer_moe_fused cost 10.353 ms
DEBUG 05-06 10:38:58.808424.808424 cuda_h.py:27] end decode_layer cost 13.786 ms
DEBUG 05-06 10:38:58.808452.808452 lmp.py:904] ---- decode step 0 layer 14 ----
DEBUG 05-06 10:38:58.809682.809682 cuda_h.py:27] end *sagl cost 1.501 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 75, 115], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 39: 1, 75: 1, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 44, 56, 112], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 2, 4: 2, 8: 1, 44: 1, 56: 1, 112: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 65, 121], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 2, 5: 2, 65: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 26, 34, 38, 86], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 2, 6: 2, 10: 1, 26: 1, 34: 1, 38: 2, 86: 1}}
INFO 05-06 10:38:58.811665.811665 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.306ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:38:58.811403.811403 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 10:38:58.812518.812518 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014190673828125 seconds
INFO 05-06 10:38:58.814028.814028 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013384819030761719 seconds
INFO 05-06 10:38:58.815477.815477 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012860298156738281 seconds
INFO 05-06 10:38:58.817816.817816 mlpmodule.py:2799] [fused_experts] gmm total=2.029ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.818794.818794 mlpmodule.py:2799] [fused_experts] gmm total=2.141ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.818171.818171 mlpmodule.py:2799] [fused_experts] gmm total=2.460ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.818732.818732 mlpmodule.py:2799] [fused_experts] gmm total=2.498ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.819244.819244 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004049062728881836 seconds
INFO 05-06 10:38:58.819593.819593 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.245208740234375e-05 seconds
DEBUG 05-06 10:38:58.820563.820563 cuda_h.py:27] end *layer_moe_fused cost 9.421 ms
DEBUG 05-06 10:38:58.820145.820145 cuda_h.py:27] end decode_layer cost 12.199 ms
DEBUG 05-06 10:38:58.820651.820651 lmp.py:904] ---- decode step 0 layer 15 ----
DEBUG 05-06 10:38:58.822385.822385 cuda_h.py:27] end *sagl cost 1.522 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 75, 83, 91, 119, 127], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 2, 75: 1, 83: 2, 91: 1, 119: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 76, 112], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 76: 1, 112: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 33, 65, 81, 121], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 2, 5: 2, 9: 1, 33: 1, 65: 1, 81: 2, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {2: 2, 6: 2, 38: 2}}
INFO 05-06 10:38:58.823633.823633 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.306ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 10:38:58.823357.823357 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.71661376953125e-05 seconds
INFO 05-06 10:38:58.825307.825307 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014262199401855469 seconds
INFO 05-06 10:38:58.826670.826670 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001293182373046875 seconds
INFO 05-06 10:38:58.827886.827886 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001428842544555664 seconds
INFO 05-06 10:38:58.830708.830708 mlpmodule.py:2799] [fused_experts] gmm total=2.168ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.830985.830985 mlpmodule.py:2799] [fused_experts] gmm total=2.238ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.830883.830883 mlpmodule.py:2799] [fused_experts] gmm total=2.129ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.830428.830428 mlpmodule.py:2799] [fused_experts] gmm total=2.494ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.832961.832961 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0045549869537353516 seconds
INFO 05-06 10:38:58.832416.832416 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.76837158203125e-05 seconds
DEBUG 05-06 10:38:58.832008.832008 cuda_h.py:27] end *layer_moe_fused cost 9.842 ms
DEBUG 05-06 10:38:58.833015.833015 cuda_h.py:27] end decode_layer cost 12.657 ms
DEBUG 05-06 10:38:58.833566.833566 lmp.py:904] ---- decode step 0 layer 16 ----
DEBUG 05-06 10:38:58.834765.834765 cuda_h.py:27] end *sagl cost 1.547 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 47, 55, 63, 107, 119], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 2, 47: 1, 55: 1, 63: 1, 107: 2, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 44, 56], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {0: 2, 4: 3, 20: 1, 44: 2, 56: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 77], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {1: 3, 5: 2, 77: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 38, 70], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 18: 1, 38: 1, 70: 1}}
INFO 05-06 10:38:58.836095.836095 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.323ms | allocate_experts_across_cpu_gpu: 0.099ms
INFO 05-06 10:38:58.836164.836164 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:38:58.837542.837542 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013647079467773438 seconds
INFO 05-06 10:38:58.839688.839688 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013484954833984375 seconds
INFO 05-06 10:38:58.840264.840264 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013039112091064453 seconds
INFO 05-06 10:38:58.843489.843489 mlpmodule.py:2799] [fused_experts] gmm total=2.282ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.843580.843580 mlpmodule.py:2799] [fused_experts] gmm total=2.542ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.843068.843068 mlpmodule.py:2799] [fused_experts] gmm total=2.439ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.843699.843699 mlpmodule.py:2799] [fused_experts] gmm total=2.580ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.845403.845403 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004678249359130859 seconds
INFO 05-06 10:38:58.845605.845605 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.76837158203125e-05 seconds
DEBUG 05-06 10:38:58.845960.845960 cuda_h.py:27] end *layer_moe_fused cost 9.840 ms
DEBUG 05-06 10:38:58.846867.846867 cuda_h.py:27] end decode_layer cost 12.716 ms
DEBUG 05-06 10:38:58.846372.846372 lmp.py:904] ---- decode step 0 layer 17 ----
DEBUG 05-06 10:38:58.847291.847291 cuda_h.py:27] end *sagl cost 1.517 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 75], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 23: 1, 27: 1, 75: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 68, 100], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 68: 1, 100: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 53, 69, 73, 113], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {1: 2, 5: 4, 9: 1, 21: 1, 53: 1, 69: 1, 73: 1, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 38, 78], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 18: 1, 38: 1, 78: 1}}
INFO 05-06 10:38:58.848116.848116 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.306ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:38:58.849469.849469 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1219253540039062e-05 seconds
INFO 05-06 10:38:58.850390.850390 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013456344604492188 seconds
INFO 05-06 10:38:58.851900.851900 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013382434844970703 seconds
INFO 05-06 10:38:58.853831.853831 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014276504516601562 seconds
INFO 05-06 10:38:58.855457.855457 mlpmodule.py:2799] [fused_experts] gmm total=2.084ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.855880.855880 mlpmodule.py:2799] [fused_experts] gmm total=2.202ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.856126.856126 mlpmodule.py:2799] [fused_experts] gmm total=2.186ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.856184.856184 mlpmodule.py:2799] [fused_experts] gmm total=2.446ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.857535.857535 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004345417022705078 seconds
INFO 05-06 10:38:58.857970.857970 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.245208740234375e-05 seconds
DEBUG 05-06 10:38:58.858252.858252 cuda_h.py:27] end *layer_moe_fused cost 9.633 ms
DEBUG 05-06 10:38:58.858296.858296 cuda_h.py:27] end decode_layer cost 12.580 ms
DEBUG 05-06 10:38:58.858086.858086 lmp.py:904] ---- decode step 0 layer 18 ----
DEBUG 05-06 10:38:58.860502.860502 cuda_h.py:27] end *sagl cost 1.497 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 55, 75, 95, 111, 115], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 55: 1, 75: 1, 95: 1, 111: 1, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 120], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 120: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 37, 77, 85, 93], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 2, 5: 2, 17: 1, 37: 1, 77: 2, 85: 1, 93: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 46, 50], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 30: 1, 46: 1, 50: 1}}
INFO 05-06 10:38:58.861596.861596 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.302ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:38:58.861182.861182 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1219253540039062e-05 seconds
INFO 05-06 10:38:58.863727.863727 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014195442199707031 seconds
INFO 05-06 10:38:58.864786.864786 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013153553009033203 seconds
INFO 05-06 10:38:58.866616.866616 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015485286712646484 seconds
INFO 05-06 10:38:58.868860.868860 mlpmodule.py:2799] [fused_experts] gmm total=2.059ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.868051.868051 mlpmodule.py:2799] [fused_experts] gmm total=2.335ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.869769.869769 mlpmodule.py:2799] [fused_experts] gmm total=2.295ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.869668.869668 mlpmodule.py:2799] [fused_experts] gmm total=2.760ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.870407.870407 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004689693450927734 seconds
INFO 05-06 10:38:58.871424.871424 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.7206878662109375e-05 seconds
DEBUG 05-06 10:38:58.871063.871063 cuda_h.py:27] end *layer_moe_fused cost 10.185 ms
DEBUG 05-06 10:38:58.871393.871393 cuda_h.py:27] end decode_layer cost 13.106 ms
DEBUG 05-06 10:38:58.871137.871137 lmp.py:904] ---- decode step 0 layer 19 ----
DEBUG 05-06 10:38:58.873229.873229 cuda_h.py:27] end *sagl cost 1.539 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 75], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {3: 2, 7: 3, 75: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 40, 44, 52, 64, 96], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {0: 2, 4: 2, 24: 2, 40: 1, 44: 2, 52: 1, 64: 1, 96: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 73, 89], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {1: 2, 5: 2, 73: 1, 89: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 38, 106], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 2, 6: 2, 10: 1, 38: 2, 106: 1}}
INFO 05-06 10:38:58.874815.874815 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.319ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 10:38:58.874347.874347 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.621246337890625e-05 seconds
INFO 05-06 10:38:58.876031.876031 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013687610626220703 seconds
INFO 05-06 10:38:58.877777.877777 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014083385467529297 seconds
INFO 05-06 10:38:58.879125.879125 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014204978942871094 seconds
INFO 05-06 10:38:58.881263.881263 mlpmodule.py:2799] [fused_experts] gmm total=2.115ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.881201.881201 mlpmodule.py:2799] [fused_experts] gmm total=2.191ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.882340.882340 mlpmodule.py:2799] [fused_experts] gmm total=2.505ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.882364.882364 mlpmodule.py:2799] [fused_experts] gmm total=2.431ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.884249.884249 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005003929138183594 seconds
INFO 05-06 10:38:58.884700.884700 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00012421607971191406 seconds
DEBUG 05-06 10:38:58.884102.884102 cuda_h.py:27] end *layer_moe_fused cost 10.560 ms
DEBUG 05-06 10:38:58.885298.885298 cuda_h.py:27] end decode_layer cost 13.527 ms
DEBUG 05-06 10:38:58.885426.885426 lmp.py:904] ---- decode step 0 layer 20 ----
DEBUG 05-06 10:38:58.887934.887934 cuda_h.py:27] end *sagl cost 1.879 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 63, 107], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {3: 3, 7: 2, 63: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 28, 68], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {0: 2, 4: 3, 20: 1, 28: 1, 68: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 41, 45, 73, 81], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 2, 5: 2, 13: 1, 41: 1, 45: 1, 73: 1, 81: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 74], 'expert_count': 3, 'ideal_gpu_count': 4, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {2: 3, 6: 2, 74: 1}}
INFO 05-06 10:38:58.889438.889438 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.421ms | allocate_experts_across_cpu_gpu: 0.110ms
INFO 05-06 10:38:58.889090.889090 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9550323486328125e-05 seconds
INFO 05-06 10:38:58.890520.890520 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013284683227539062 seconds
INFO 05-06 10:38:58.892629.892629 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001596212387084961 seconds
INFO 05-06 10:38:58.893296.893296 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00147247314453125 seconds
INFO 05-06 10:38:58.896562.896562 mlpmodule.py:2799] [fused_experts] gmm total=2.331ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.896621.896621 mlpmodule.py:2799] [fused_experts] gmm total=2.199ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.896219.896219 mlpmodule.py:2799] [fused_experts] gmm total=2.558ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.896243.896243 mlpmodule.py:2799] [fused_experts] gmm total=2.802ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.898961.898961 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004728794097900391 seconds
INFO 05-06 10:38:58.898840.898840 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.507469177246094e-05 seconds
DEBUG 05-06 10:38:58.899553.899553 cuda_h.py:27] end *layer_moe_fused cost 10.486 ms
DEBUG 05-06 10:38:58.899150.899150 cuda_h.py:27] end decode_layer cost 13.907 ms
DEBUG 05-06 10:38:58.899894.899894 lmp.py:904] ---- decode step 0 layer 21 ----
DEBUG 05-06 10:38:58.901332.901332 cuda_h.py:27] end *sagl cost 1.583 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 51], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {3: 2, 7: 3, 11: 2, 51: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 120], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {0: 2, 4: 2, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 57, 125], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 3, 5: 2, 57: 1, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 26, 34, 46, 102, 126], 'expert_count': 8, 'ideal_gpu_count': 4, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {2: 2, 6: 2, 18: 1, 26: 2, 34: 1, 46: 2, 102: 1, 126: 1}}
INFO 05-06 10:38:58.902712.902712 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.085ms
INFO 05-06 10:38:58.902006.902006 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8596649169921875e-05 seconds
INFO 05-06 10:38:58.903654.903654 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001321554183959961 seconds
INFO 05-06 10:38:58.905150.905150 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012941360473632812 seconds
INFO 05-06 10:38:58.906215.906215 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012843608856201172 seconds
INFO 05-06 10:38:58.908526.908526 mlpmodule.py:2799] [fused_experts] gmm total=2.074ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.909949.909949 mlpmodule.py:2799] [fused_experts] gmm total=2.325ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.909251.909251 mlpmodule.py:2799] [fused_experts] gmm total=2.275ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.909545.909545 mlpmodule.py:2799] [fused_experts] gmm total=2.468ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.911852.911852 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0045130252838134766 seconds
INFO 05-06 10:38:58.911486.911486 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.125999450683594e-05 seconds
DEBUG 05-06 10:38:58.911080.911080 cuda_h.py:27] end *layer_moe_fused cost 9.579 ms
DEBUG 05-06 10:38:58.912709.912709 cuda_h.py:27] end decode_layer cost 12.464 ms
DEBUG 05-06 10:38:58.912260.912260 lmp.py:904] ---- decode step 0 layer 22 ----
DEBUG 05-06 10:38:58.913874.913874 cuda_h.py:27] end *sagl cost 1.468 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 43, 75, 119, 127], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 43: 1, 75: 1, 119: 2, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 40, 108, 120], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 2, 4: 2, 8: 1, 40: 1, 108: 1, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 25, 89, 93, 109], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 25: 1, 89: 2, 93: 1, 109: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 70, 126], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {2: 2, 6: 2, 70: 1, 126: 1}}
INFO 05-06 10:38:58.914081.914081 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.306ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:38:58.914143.914143 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.09808349609375e-05 seconds
INFO 05-06 10:38:58.916466.916466 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014884471893310547 seconds
INFO 05-06 10:38:58.918989.918989 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014560222625732422 seconds
INFO 05-06 10:38:58.919190.919190 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015795230865478516 seconds
INFO 05-06 10:38:58.922028.922028 mlpmodule.py:2799] [fused_experts] gmm total=2.258ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.922550.922550 mlpmodule.py:2799] [fused_experts] gmm total=2.335ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.922471.922471 mlpmodule.py:2799] [fused_experts] gmm total=2.441ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.922944.922944 mlpmodule.py:2799] [fused_experts] gmm total=2.617ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.924076.924076 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004920482635498047 seconds
INFO 05-06 10:38:58.924963.924963 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.723403930664062e-05 seconds
DEBUG 05-06 10:38:58.925152.925152 cuda_h.py:27] end *layer_moe_fused cost 11.016 ms
DEBUG 05-06 10:38:58.925151.925151 cuda_h.py:27] end decode_layer cost 13.919 ms
DEBUG 05-06 10:38:58.926829.926829 lmp.py:904] ---- decode step 0 layer 23 ----
DEBUG 05-06 10:38:58.928099.928099 cuda_h.py:27] end *sagl cost 2.072 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 99, 115], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 99: 1, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 72], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 12: 1, 72: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 97, 109], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {1: 3, 5: 2, 29: 1, 97: 1, 109: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 74, 86, 106, 118], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {2: 3, 6: 2, 26: 1, 74: 1, 86: 1, 106: 1, 118: 2}}
INFO 05-06 10:38:58.929003.929003 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.317ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:38:58.929211.929211 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6927719116210938e-05 seconds
INFO 05-06 10:38:58.931785.931785 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001474618911743164 seconds
INFO 05-06 10:38:58.932389.932389 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013642311096191406 seconds
INFO 05-06 10:38:58.933931.933931 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00128173828125 seconds
INFO 05-06 10:38:58.936196.936196 mlpmodule.py:2799] [fused_experts] gmm total=1.891ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.936811.936811 mlpmodule.py:2799] [fused_experts] gmm total=2.323ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.936078.936078 mlpmodule.py:2799] [fused_experts] gmm total=2.499ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.936075.936075 mlpmodule.py:2799] [fused_experts] gmm total=2.508ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.938773.938773 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004313230514526367 seconds
INFO 05-06 10:38:58.938917.938917 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.221366882324219e-05 seconds
DEBUG 05-06 10:38:58.938914.938914 cuda_h.py:27] end *layer_moe_fused cost 9.618 ms
DEBUG 05-06 10:38:58.939868.939868 cuda_h.py:27] end decode_layer cost 13.052 ms
DEBUG 05-06 10:38:58.939374.939374 lmp.py:904] ---- decode step 0 layer 24 ----
DEBUG 05-06 10:38:58.940274.940274 cuda_h.py:27] end *sagl cost 1.537 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 123], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 19: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 48, 60, 124], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 12: 2, 48: 1, 60: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 73, 97], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 2, 5: 2, 73: 2, 97: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 26, 30, 94, 110], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {2: 2, 6: 2, 10: 1, 18: 1, 26: 1, 30: 1, 94: 1, 110: 1}}
INFO 05-06 10:38:58.942330.942330 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.323ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:38:58.942253.942253 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 10:38:58.943341.943341 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013985633850097656 seconds
INFO 05-06 10:38:58.944771.944771 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013146400451660156 seconds
INFO 05-06 10:38:58.946253.946253 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00127410888671875 seconds
INFO 05-06 10:38:58.948820.948820 mlpmodule.py:2799] [fused_experts] gmm total=2.263ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.948819.948819 mlpmodule.py:2799] [fused_experts] gmm total=2.327ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.949668.949668 mlpmodule.py:2799] [fused_experts] gmm total=2.367ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.949685.949685 mlpmodule.py:2799] [fused_experts] gmm total=2.583ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.950617.950617 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0044939517974853516 seconds
INFO 05-06 10:38:58.950542.950542 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 10:38:58.951884.951884 cuda_h.py:27] end *layer_moe_fused cost 9.642 ms
DEBUG 05-06 10:38:58.951938.951938 cuda_h.py:27] end decode_layer cost 12.488 ms
DEBUG 05-06 10:38:58.951873.951873 lmp.py:904] ---- decode step 0 layer 25 ----
DEBUG 05-06 10:38:58.953662.953662 cuda_h.py:27] end *sagl cost 1.559 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 35, 107], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 15: 1, 19: 1, 27: 1, 35: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 64, 68, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 2, 4: 2, 16: 1, 64: 1, 68: 1, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 41, 49, 117], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 41: 2, 49: 1, 117: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 58, 114], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 58: 2, 114: 1}}
INFO 05-06 10:38:58.954632.954632 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.310ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 10:38:58.954078.954078 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.288818359375e-05 seconds
INFO 05-06 10:38:58.956623.956623 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001383066177368164 seconds
INFO 05-06 10:38:58.957669.957669 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013108253479003906 seconds
INFO 05-06 10:38:58.958696.958696 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013606548309326172 seconds
INFO 05-06 10:38:58.961209.961209 mlpmodule.py:2799] [fused_experts] gmm total=2.476ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.961873.961873 mlpmodule.py:2799] [fused_experts] gmm total=2.408ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.961250.961250 mlpmodule.py:2799] [fused_experts] gmm total=2.775ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.962182.962182 mlpmodule.py:2799] [fused_experts] gmm total=2.827ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.963175.963175 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004746675491333008 seconds
INFO 05-06 10:38:58.963763.963763 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.316734313964844e-05 seconds
DEBUG 05-06 10:38:58.964582.964582 cuda_h.py:27] end *layer_moe_fused cost 9.988 ms
DEBUG 05-06 10:38:58.964233.964233 cuda_h.py:27] end decode_layer cost 12.900 ms
DEBUG 05-06 10:38:58.964453.964453 lmp.py:904] ---- decode step 0 layer 26 ----
DEBUG 05-06 10:38:58.966076.966076 cuda_h.py:27] end *sagl cost 1.543 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 27, 103, 119], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {3: 2, 7: 2, 19: 1, 27: 1, 103: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 52, 72, 104], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 52: 1, 72: 1, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 65, 97], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 17: 1, 65: 2, 97: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 34, 70, 78, 110], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 2, 26: 1, 34: 1, 70: 1, 78: 1, 110: 1}}
INFO 05-06 10:38:58.967601.967601 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.310ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:38:58.967559.967559 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.719329833984375e-05 seconds
INFO 05-06 10:38:58.968948.968948 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013041496276855469 seconds
INFO 05-06 10:38:58.970059.970059 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001291036605834961 seconds
INFO 05-06 10:38:58.971161.971161 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013806819915771484 seconds
INFO 05-06 10:38:58.974707.974707 mlpmodule.py:2799] [fused_experts] gmm total=2.251ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.974229.974229 mlpmodule.py:2799] [fused_experts] gmm total=2.327ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.974588.974588 mlpmodule.py:2799] [fused_experts] gmm total=2.449ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.974374.974374 mlpmodule.py:2799] [fused_experts] gmm total=2.470ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.976099.976099 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004628658294677734 seconds
INFO 05-06 10:38:58.976025.976025 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.53131103515625e-05 seconds
DEBUG 05-06 10:38:58.976672.976672 cuda_h.py:27] end *layer_moe_fused cost 9.833 ms
DEBUG 05-06 10:38:58.977812.977812 cuda_h.py:27] end decode_layer cost 12.688 ms
DEBUG 05-06 10:38:58.977039.977039 lmp.py:904] ---- decode step 0 layer 27 ----
DEBUG 05-06 10:38:58.979476.979476 cuda_h.py:27] end *sagl cost 1.717 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 95, 115, 119], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 11, 'token_per_expert': {3: 3, 7: 2, 27: 2, 95: 1, 115: 1, 119: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 88], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {0: 2, 4: 2, 88: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 85, 121], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 2, 5: 2, 37: 1, 85: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 62, 78, 94, 122], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 2, 26: 1, 62: 1, 78: 1, 94: 1, 122: 1}}
INFO 05-06 10:38:58.980159.980159 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.336ms | allocate_experts_across_cpu_gpu: 0.102ms
INFO 05-06 10:38:58.980844.980844 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.811981201171875e-05 seconds
INFO 05-06 10:38:58.982471.982471 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014770030975341797 seconds
INFO 05-06 10:38:58.983110.983110 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014221668243408203 seconds
INFO 05-06 10:38:58.985859.985859 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013244152069091797 seconds
INFO 05-06 10:38:58.987490.987490 mlpmodule.py:2799] [fused_experts] gmm total=2.203ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.987562.987562 mlpmodule.py:2799] [fused_experts] gmm total=2.318ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.987451.987451 mlpmodule.py:2799] [fused_experts] gmm total=2.334ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.988662.988662 mlpmodule.py:2799] [fused_experts] gmm total=2.582ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:58.989648.989648 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004513263702392578 seconds
INFO 05-06 10:38:58.989474.989474 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.245208740234375e-05 seconds
DEBUG 05-06 10:38:58.990929.990929 cuda_h.py:27] end *layer_moe_fused cost 9.988 ms
DEBUG 05-06 10:38:58.990200.990200 cuda_h.py:27] end decode_layer cost 13.105 ms
DEBUG 05-06 10:38:58.990659.990659 lmp.py:904] ---- decode step 0 layer 28 ----
DEBUG 05-06 10:38:58.992075.992075 cuda_h.py:27] end *sagl cost 1.498 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 95, 119], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {3: 2, 7: 2, 39: 1, 95: 1, 119: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 48, 76, 100, 108], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 20: 1, 48: 1, 76: 1, 100: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 49, 57, 77], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 49: 1, 57: 2, 77: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 106, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 26: 1, 106: 1, 126: 1}}
INFO 05-06 10:38:58.993745.993745 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.298ms | allocate_experts_across_cpu_gpu: 0.087ms
INFO 05-06 10:38:58.993092.993092 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2411346435546875e-05 seconds
INFO 05-06 10:38:58.994671.994671 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001445770263671875 seconds
INFO 05-06 10:38:58.996174.996174 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012927055358886719 seconds
INFO 05-06 10:38:58.997033.997033 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001483917236328125 seconds
INFO 05-06 10:38:59.000556.000556 mlpmodule.py:2799] [fused_experts] gmm total=2.222ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:59.000510.000510 mlpmodule.py:2799] [fused_experts] gmm total=2.537ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:59.000188.000188 mlpmodule.py:2799] [fused_experts] gmm total=2.447ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:59.001411.001411 mlpmodule.py:2799] [fused_experts] gmm total=2.705ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:59.002383.002383 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004645109176635742 seconds
INFO 05-06 10:38:59.002548.002548 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.340576171875e-05 seconds
DEBUG 05-06 10:38:59.003255.003255 cuda_h.py:27] end *layer_moe_fused cost 10.067 ms
DEBUG 05-06 10:38:59.003645.003645 cuda_h.py:27] end decode_layer cost 12.835 ms
DEBUG 05-06 10:38:59.003627.003627 lmp.py:904] ---- decode step 0 layer 29 ----
DEBUG 05-06 10:38:59.005893.005893 cuda_h.py:27] end *sagl cost 1.561 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {3: 2, 7: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 56, 64, 108], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 20: 1, 56: 1, 64: 2, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 93, 101], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 3, 5: 2, 93: 1, 101: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 42, 54, 62, 74, 82, 106], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {2: 2, 6: 2, 26: 1, 42: 1, 54: 2, 62: 1, 74: 1, 82: 1, 106: 1}}
INFO 05-06 10:38:59.006711.006711 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.315ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:38:59.006165.006165 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-05 seconds
INFO 05-06 10:38:59.007127.007127 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014102458953857422 seconds
INFO 05-06 10:38:59.009226.009226 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013158321380615234 seconds
INFO 05-06 10:38:59.010589.010589 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001293182373046875 seconds
INFO 05-06 10:38:59.012307.012307 mlpmodule.py:2799] [fused_experts] gmm total=2.018ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:59.013767.013767 mlpmodule.py:2799] [fused_experts] gmm total=2.246ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:59.013682.013682 mlpmodule.py:2799] [fused_experts] gmm total=2.357ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:59.013133.013133 mlpmodule.py:2799] [fused_experts] gmm total=2.493ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:38:59.015472.015472 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0043408870697021484 seconds
INFO 05-06 10:38:59.015344.015344 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.269050598144531e-05 seconds
DEBUG 05-06 10:38:59.015462.015462 cuda_h.py:27] end *layer_moe_fused cost 9.573 ms
DEBUG 05-06 10:38:59.016921.016921 cuda_h.py:27] end decode_layer cost 12.536 ms
DEBUG 05-06 10:38:59.016380.016380 cuda_h.py:27] end decode_step cost 390.636 ms
INFO 05-06 10:38:59.016275.016275 lmp.py:931] decode step 0 time: 0.3906667232513428 seconds
WARNING 05-06 10:38:59.016528.016528 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:38:59.016783.016783 helper.py:35]   NaN count (hidden): 5632
WARNING 05-06 10:38:59.016007.016007 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:38:59.016917.016917 helper.py:39]   NaN count (normed): 5632
WARNING 05-06 10:38:59.022823.022823 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:38:59.022032.022032 helper.py:50]   NaN count: 524288
WARNING 05-06 10:38:59.022617.022617 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:38:59.023453.023453 cuda_h.py:27] end init_inputs_tokens cost 7.591 ms
DEBUG 05-06 10:38:59.023150.023150 lmp.py:899] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:38:59.023105.023105 lmp.py:904] ---- decode step 1 layer 0 ----
DEBUG 05-06 10:38:59.025422.025422 cuda_h.py:27] end *sagl cost 1.529 ms
DEBUG 05-06 10:38:59.030535.030535 cuda_h.py:27] end *layer_moe_fused cost 3.684 ms
DEBUG 05-06 10:38:59.030387.030387 cuda_h.py:27] end decode_layer cost 6.625 ms
DEBUG 05-06 10:38:59.030435.030435 lmp.py:904] ---- decode step 1 layer 1 ----
DEBUG 05-06 10:38:59.032778.032778 cuda_h.py:27] end *sagl cost 2.055 ms
DEBUG 05-06 10:38:59.037327.037327 cuda_h.py:27] end *layer_moe_fused cost 3.211 ms
DEBUG 05-06 10:38:59.037423.037423 cuda_h.py:27] end decode_layer cost 6.938 ms
DEBUG 05-06 10:38:59.037756.037756 lmp.py:904] ---- decode step 1 layer 2 ----
DEBUG 05-06 10:38:59.039458.039458 cuda_h.py:27] end *sagl cost 1.899 ms
DEBUG 05-06 10:38:59.043571.043571 cuda_h.py:27] end *layer_moe_fused cost 2.435 ms
DEBUG 05-06 10:38:59.043734.043734 cuda_h.py:27] end decode_layer cost 6.053 ms
DEBUG 05-06 10:38:59.043399.043399 lmp.py:904] ---- decode step 1 layer 3 ----
DEBUG 05-06 10:38:59.045286.045286 cuda_h.py:27] end *sagl cost 1.926 ms
DEBUG 05-06 10:38:59.048335.048335 cuda_h.py:27] end *layer_moe_fused cost 2.138 ms
DEBUG 05-06 10:38:59.049656.049656 cuda_h.py:27] end decode_layer cost 5.697 ms
DEBUG 05-06 10:38:59.049367.049367 lmp.py:904] ---- decode step 1 layer 4 ----
DEBUG 05-06 10:38:59.051431.051431 cuda_h.py:27] end *sagl cost 1.852 ms
DEBUG 05-06 10:38:59.054688.054688 cuda_h.py:27] end *layer_moe_fused cost 2.062 ms
DEBUG 05-06 10:38:59.055717.055717 cuda_h.py:27] end decode_layer cost 5.520 ms
DEBUG 05-06 10:38:59.055474.055474 lmp.py:904] ---- decode step 1 layer 5 ----
DEBUG 05-06 10:38:59.057057.057057 cuda_h.py:27] end *sagl cost 1.919 ms
DEBUG 05-06 10:38:59.060096.060096 cuda_h.py:27] end *layer_moe_fused cost 2.081 ms
DEBUG 05-06 10:38:59.060510.060510 cuda_h.py:27] end decode_layer cost 5.604 ms
DEBUG 05-06 10:38:59.060267.060267 lmp.py:904] ---- decode step 1 layer 6 ----
DEBUG 05-06 10:38:59.062496.062496 cuda_h.py:27] end *sagl cost 1.834 ms
DEBUG 05-06 10:38:59.065039.065039 cuda_h.py:27] end *layer_moe_fused cost 2.060 ms
DEBUG 05-06 10:38:59.066128.066128 cuda_h.py:27] end decode_layer cost 5.545 ms
DEBUG 05-06 10:38:59.066124.066124 lmp.py:904] ---- decode step 1 layer 7 ----
DEBUG 05-06 10:38:59.068249.068249 cuda_h.py:27] end *sagl cost 1.897 ms
DEBUG 05-06 10:38:59.071751.071751 cuda_h.py:27] end *layer_moe_fused cost 2.058 ms
DEBUG 05-06 10:38:59.072517.072517 cuda_h.py:27] end decode_layer cost 5.586 ms
DEBUG 05-06 10:38:59.072227.072227 lmp.py:904] ---- decode step 1 layer 8 ----
DEBUG 05-06 10:38:59.073046.073046 cuda_h.py:27] end *sagl cost 1.847 ms
DEBUG 05-06 10:38:59.077038.077038 cuda_h.py:27] end *layer_moe_fused cost 2.072 ms
DEBUG 05-06 10:38:59.077498.077498 cuda_h.py:27] end decode_layer cost 5.496 ms
DEBUG 05-06 10:38:59.077924.077924 lmp.py:904] ---- decode step 1 layer 9 ----
DEBUG 05-06 10:38:59.079943.079943 cuda_h.py:27] end *sagl cost 1.890 ms
DEBUG 05-06 10:38:59.082381.082381 cuda_h.py:27] end *layer_moe_fused cost 2.070 ms
DEBUG 05-06 10:38:59.083179.083179 cuda_h.py:27] end decode_layer cost 5.625 ms
DEBUG 05-06 10:38:59.083651.083651 lmp.py:904] ---- decode step 1 layer 10 ----
DEBUG 05-06 10:38:59.085855.085855 cuda_h.py:27] end *sagl cost 1.886 ms
DEBUG 05-06 10:38:59.088627.088627 cuda_h.py:27] end *layer_moe_fused cost 2.210 ms
DEBUG 05-06 10:38:59.089194.089194 cuda_h.py:27] end decode_layer cost 5.732 ms
DEBUG 05-06 10:38:59.089157.089157 lmp.py:904] ---- decode step 1 layer 11 ----
DEBUG 05-06 10:38:59.091213.091213 cuda_h.py:27] end *sagl cost 2.017 ms
DEBUG 05-06 10:38:59.094359.094359 cuda_h.py:27] end *layer_moe_fused cost 2.064 ms
DEBUG 05-06 10:38:59.094580.094580 cuda_h.py:27] end decode_layer cost 5.714 ms
DEBUG 05-06 10:38:59.094099.094099 lmp.py:904] ---- decode step 1 layer 12 ----
DEBUG 05-06 10:38:59.096634.096634 cuda_h.py:27] end *sagl cost 1.884 ms
DEBUG 05-06 10:38:59.100633.100633 cuda_h.py:27] end *layer_moe_fused cost 2.077 ms
DEBUG 05-06 10:38:59.100907.100907 cuda_h.py:27] end decode_layer cost 5.542 ms
DEBUG 05-06 10:38:59.100949.100949 lmp.py:904] ---- decode step 1 layer 13 ----
DEBUG 05-06 10:38:59.102126.102126 cuda_h.py:27] end *sagl cost 1.867 ms
DEBUG 05-06 10:38:59.105313.105313 cuda_h.py:27] end *layer_moe_fused cost 2.132 ms
DEBUG 05-06 10:38:59.106488.106488 cuda_h.py:27] end decode_layer cost 5.588 ms
DEBUG 05-06 10:38:59.106768.106768 lmp.py:904] ---- decode step 1 layer 14 ----
DEBUG 05-06 10:38:59.108878.108878 cuda_h.py:27] end *sagl cost 1.852 ms
DEBUG 05-06 10:38:59.111539.111539 cuda_h.py:27] end *layer_moe_fused cost 2.077 ms
DEBUG 05-06 10:38:59.111504.111504 cuda_h.py:27] end decode_layer cost 5.558 ms
DEBUG 05-06 10:38:59.111738.111738 lmp.py:904] ---- decode step 1 layer 15 ----
DEBUG 05-06 10:38:59.113643.113643 cuda_h.py:27] end *sagl cost 1.842 ms
DEBUG 05-06 10:38:59.116170.116170 cuda_h.py:27] end *layer_moe_fused cost 2.057 ms
DEBUG 05-06 10:38:59.117808.117808 cuda_h.py:27] end decode_layer cost 5.454 ms
DEBUG 05-06 10:38:59.117373.117373 lmp.py:904] ---- decode step 1 layer 16 ----
DEBUG 05-06 10:38:59.119589.119589 cuda_h.py:27] end *sagl cost 1.814 ms
DEBUG 05-06 10:38:59.122218.122218 cuda_h.py:27] end *layer_moe_fused cost 2.073 ms
DEBUG 05-06 10:38:59.122678.122678 cuda_h.py:27] end decode_layer cost 5.522 ms
DEBUG 05-06 10:38:59.122435.122435 lmp.py:904] ---- decode step 1 layer 17 ----
DEBUG 05-06 10:38:59.124531.124531 cuda_h.py:27] end *sagl cost 1.843 ms
DEBUG 05-06 10:38:59.127113.127113 cuda_h.py:27] end *layer_moe_fused cost 2.071 ms
DEBUG 05-06 10:38:59.128666.128666 cuda_h.py:27] end decode_layer cost 5.502 ms
DEBUG 05-06 10:38:59.128707.128707 lmp.py:904] ---- decode step 1 layer 18 ----
DEBUG 05-06 10:38:59.130307.130307 cuda_h.py:27] end *sagl cost 1.827 ms
DEBUG 05-06 10:38:59.133000.133000 cuda_h.py:27] end *layer_moe_fused cost 2.072 ms
DEBUG 05-06 10:38:59.134692.134692 cuda_h.py:27] end decode_layer cost 5.461 ms
DEBUG 05-06 10:38:59.134019.134019 lmp.py:904] ---- decode step 1 layer 19 ----
DEBUG 05-06 10:38:59.136224.136224 cuda_h.py:27] end *sagl cost 1.922 ms
DEBUG 05-06 10:38:59.139639.139639 cuda_h.py:27] end *layer_moe_fused cost 2.063 ms
DEBUG 05-06 10:38:59.139258.139258 cuda_h.py:27] end decode_layer cost 5.579 ms
DEBUG 05-06 10:38:59.139731.139731 lmp.py:904] ---- decode step 1 layer 20 ----
DEBUG 05-06 10:38:59.141800.141800 cuda_h.py:27] end *sagl cost 1.823 ms
DEBUG 05-06 10:38:59.144208.144208 cuda_h.py:27] end *layer_moe_fused cost 2.042 ms
DEBUG 05-06 10:38:59.145661.145661 cuda_h.py:27] end decode_layer cost 5.421 ms
DEBUG 05-06 10:38:59.145749.145749 lmp.py:904] ---- decode step 1 layer 21 ----
DEBUG 05-06 10:38:59.147959.147959 cuda_h.py:27] end *sagl cost 1.856 ms
DEBUG 05-06 10:38:59.150706.150706 cuda_h.py:27] end *layer_moe_fused cost 2.040 ms
DEBUG 05-06 10:38:59.150874.150874 cuda_h.py:27] end decode_layer cost 5.494 ms
DEBUG 05-06 10:38:59.150916.150916 lmp.py:904] ---- decode step 1 layer 22 ----
DEBUG 05-06 10:38:59.152861.152861 cuda_h.py:27] end *sagl cost 1.872 ms
DEBUG 05-06 10:38:59.155733.155733 cuda_h.py:27] end *layer_moe_fused cost 2.057 ms
DEBUG 05-06 10:38:59.156187.156187 cuda_h.py:27] end decode_layer cost 5.532 ms
DEBUG 05-06 10:38:59.156183.156183 lmp.py:904] ---- decode step 1 layer 23 ----
DEBUG 05-06 10:38:59.158299.158299 cuda_h.py:27] end *sagl cost 1.821 ms
DEBUG 05-06 10:38:59.161616.161616 cuda_h.py:27] end *layer_moe_fused cost 2.100 ms
DEBUG 05-06 10:38:59.161599.161599 cuda_h.py:27] end decode_layer cost 5.498 ms
DEBUG 05-06 10:38:59.161402.161402 lmp.py:904] ---- decode step 1 layer 24 ----
DEBUG 05-06 10:38:59.163976.163976 cuda_h.py:27] end *sagl cost 1.843 ms
DEBUG 05-06 10:38:59.166656.166656 cuda_h.py:27] end *layer_moe_fused cost 2.061 ms
DEBUG 05-06 10:38:59.167685.167685 cuda_h.py:27] end decode_layer cost 5.470 ms
DEBUG 05-06 10:38:59.167965.167965 lmp.py:904] ---- decode step 1 layer 25 ----
DEBUG 05-06 10:38:59.169217.169217 cuda_h.py:27] end *sagl cost 1.922 ms
DEBUG 05-06 10:38:59.172613.172613 cuda_h.py:27] end *layer_moe_fused cost 2.077 ms
DEBUG 05-06 10:38:59.173020.173020 cuda_h.py:27] end decode_layer cost 5.583 ms
DEBUG 05-06 10:38:59.173585.173585 lmp.py:904] ---- decode step 1 layer 26 ----
DEBUG 05-06 10:38:59.175284.175284 cuda_h.py:27] end *sagl cost 1.830 ms
DEBUG 05-06 10:38:59.178039.178039 cuda_h.py:27] end *layer_moe_fused cost 2.066 ms
DEBUG 05-06 10:38:59.178505.178505 cuda_h.py:27] end decode_layer cost 5.518 ms
DEBUG 05-06 10:38:59.178262.178262 lmp.py:904] ---- decode step 1 layer 27 ----
DEBUG 05-06 10:38:59.180003.180003 cuda_h.py:27] end *sagl cost 1.895 ms
DEBUG 05-06 10:38:59.183725.183725 cuda_h.py:27] end *layer_moe_fused cost 2.101 ms
DEBUG 05-06 10:38:59.184893.184893 cuda_h.py:27] end decode_layer cost 5.586 ms
DEBUG 05-06 10:38:59.184266.184266 lmp.py:904] ---- decode step 1 layer 28 ----
DEBUG 05-06 10:38:59.186442.186442 cuda_h.py:27] end *sagl cost 1.831 ms
DEBUG 05-06 10:38:59.189385.189385 cuda_h.py:27] end *layer_moe_fused cost 2.014 ms
DEBUG 05-06 10:38:59.189653.189653 cuda_h.py:27] end decode_layer cost 5.407 ms
DEBUG 05-06 10:38:59.189695.189695 lmp.py:904] ---- decode step 1 layer 29 ----
DEBUG 05-06 10:38:59.191727.191727 cuda_h.py:27] end *sagl cost 1.890 ms
DEBUG 05-06 10:38:59.194771.194771 cuda_h.py:27] end *layer_moe_fused cost 2.026 ms
DEBUG 05-06 10:38:59.195078.195078 cuda_h.py:27] end decode_layer cost 5.509 ms
DEBUG 05-06 10:38:59.195405.195405 cuda_h.py:27] end decode_step cost 179.225 ms
INFO 05-06 10:38:59.195790.195790 lmp.py:931] decode step 1 time: 0.17926383018493652 seconds
Time taken: 5.963669620454311 seconds
X512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x59f747127b60, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
CPUInfer[0x59f746752970]: Goodbye
