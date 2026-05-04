here pin
INFO 05-03 22:31:01.347261.347261 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-03 22:31:01.912691.912691 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-03 22:31:02.369133.369133 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-03 22:31:02.369492.369492 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 1.022s
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
INFO 05-03 22:31:09.682262.682262 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-03 22:31:10.120838.120838 cuda_h.py:27] end init_cmv_hmv cost 438.614 ms
DEBUG 05-03 22:31:10.130206.130206 cuda_memory_view.py:1366] 
DEBUG 05-03 22:31:10.130206.130206 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.002696514129638672
DEBUG 05-03 22:31:10.148268.148268 mlpmodule.py:993] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-03 22:31:10.148316.148316 cuda_memory_view.py:1370] 
DEBUG 05-03 22:31:10.148316.148316 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.0177154541015625
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-03 22:31:12.096946.096946 lmp.py:368] init kt-kernel layer 0 ok
INFO 05-03 22:31:13.077254.077254 lmp.py:368] init kt-kernel layer 1 ok
INFO 05-03 22:31:13.960776.960776 lmp.py:368] init kt-kernel layer 2 ok
INFO 05-03 22:31:14.803923.803923 lmp.py:368] init kt-kernel layer 3 ok
INFO 05-03 22:31:15.644577.644577 lmp.py:368] init kt-kernel layer 4 ok
INFO 05-03 22:31:16.488894.488894 lmp.py:368] init kt-kernel layer 5 ok
INFO 05-03 22:31:17.321773.321773 lmp.py:368] init kt-kernel layer 6 ok
INFO 05-03 22:31:18.171736.171736 lmp.py:368] init kt-kernel layer 7 ok
INFO 05-03 22:31:19.023761.023761 lmp.py:368] init kt-kernel layer 8 ok
INFO 05-03 22:31:19.847379.847379 lmp.py:368] init kt-kernel layer 9 ok
INFO 05-03 22:31:20.699013.699013 lmp.py:368] init kt-kernel layer 10 ok
INFO 05-03 22:31:21.530956.530956 lmp.py:368] init kt-kernel layer 11 ok
INFO 05-03 22:31:22.362875.362875 lmp.py:368] init kt-kernel layer 12 ok
INFO 05-03 22:31:23.163919.163919 lmp.py:368] init kt-kernel layer 13 ok
INFO 05-03 22:31:23.988798.988798 lmp.py:368] init kt-kernel layer 14 ok
INFO 05-03 22:31:24.818970.818970 lmp.py:368] init kt-kernel layer 15 ok
INFO 05-03 22:31:25.667764.667764 lmp.py:368] init kt-kernel layer 16 ok
INFO 05-03 22:31:26.515021.515021 lmp.py:368] init kt-kernel layer 17 ok
INFO 05-03 22:31:27.363565.363565 lmp.py:368] init kt-kernel layer 18 ok
INFO 05-03 22:31:28.202052.202052 lmp.py:368] init kt-kernel layer 19 ok
INFO 05-03 22:31:29.024441.024441 lmp.py:368] init kt-kernel layer 20 ok
INFO 05-03 22:31:29.850474.850474 lmp.py:368] init kt-kernel layer 21 ok
INFO 05-03 22:31:30.683210.683210 lmp.py:368] init kt-kernel layer 22 ok
CPUInfer[0x55c4ddcffd30]: Hello
WorkerPool[0x55c4ddd376b0] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x55c4f46ca900]: Hello
WorkerPool[0x55c536ff5930] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVINFO 05-03 22:31:31.520784.520784 lmp.py:368] init kt-kernel layer 23 ok
INFO 05-03 22:31:32.361422.361422 lmp.py:368] init kt-kernel layer 24 ok
INFO 05-03 22:31:33.182138.182138 lmp.py:368] init kt-kernel layer 25 ok
INFO 05-03 22:31:34.002437.002437 lmp.py:368] init kt-kernel layer 26 ok
INFO 05-03 22:31:34.867899.867899 lmp.py:368] init kt-kernel layer 27 ok
INFO 05-03 22:31:35.706688.706688 lmp.py:368] init kt-kernel layer 28 ok
INFO 05-03 22:31:36.534689.534689 lmp.py:368] init kt-kernel layer 29 ok
generate input ids cost 0.0476839542388916 s
DEBUG 05-03 22:31:39.663917.663917 cuda_h.py:27] end generate_input_ids cost 3071.820 ms
DEBUG 05-03 22:31:39.663871.663871 cuda_h.py:27] end init_cache cost 0.048 ms
INFO 05-03 22:31:39.676381.676381 lmp.py:2040] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6629859268, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7273408761529578, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-03 22:31:39.738326.738326 lmp.py:2058] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.738626.738626 lmp.py:2058] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.739588.739588 lmp.py:2058] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.739909.739909 lmp.py:2058] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.739110.739110 lmp.py:2058] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.739690.739690 lmp.py:2058] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.740494.740494 lmp.py:2058] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.740244.740244 lmp.py:2058] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.740093.740093 lmp.py:2058] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.740572.740572 lmp.py:2058] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.741447.741447 lmp.py:2058] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.741610.741610 lmp.py:2058] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.741281.741281 lmp.py:2058] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.741910.741910 lmp.py:2058] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.742309.742309 lmp.py:2058] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.742614.742614 lmp.py:2058] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.742167.742167 lmp.py:2058] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.742169.742169 lmp.py:2058] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.743370.743370 lmp.py:2058] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.743821.743821 lmp.py:2058] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.743716.743716 lmp.py:2058] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.743704.743704 lmp.py:2058] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.744784.744784 lmp.py:2058] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.744252.744252 lmp.py:2058] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.744351.744351 lmp.py:2058] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.745483.745483 lmp.py:2058] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.745396.745396 lmp.py:2058] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.745860.745860 lmp.py:2058] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.745174.745174 lmp.py:2058] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:39.745420.745420 lmp.py:2058] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-03 22:31:40.038128.038128 cuda_h.py:27] end init_loading_placement cost 374.654 ms
DEBUG 05-03 22:31:40.038181.038181 sllm_store_c.py:27] get device uuid map
DEBUG 05-03 22:31:40.038806.038806 sllm_store_c.py:29] call client load into gpu
DEBUG 05-03 22:31:40 client.py:72] load_into_gpu: gemma4-26B-A4B, 2cfbd149-c823-4bdd-8b5c-2d49f9123be8
INFO 05-03 22:31:40 client.py:135] Model loaded: gemma4-26B-A4B, 2cfbd149-c823-4bdd-8b5c-2d49f9123be8
INFO 05-03 22:31:40 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 2cfbd149-c823-4bdd-8b5c-2d49f9123be8
INFO 05-03 22:31:40 client.py:212] Model loaded
DEBUG 05-03 22:31:40.576337.576337 cuda_h.py:27] end init_general_sagl_loading_async cost 538.075 ms
DEBUG 05-03 22:31:40.592535.592535 sllm_store_c.py:27] get device uuid map
DEBUG 05-03 22:31:40.592300.592300 sllm_store_c.py:29] call client load into gpu
DEBUG 05-03 22:31:40 client.py:72] load_into_gpu: gemma4-26B-A4B, e5f9bf63-4f36-489c-a3de-3cfa25bb7461
INFO 05-03 22:31:40 client.py:135] Model loaded: gemma4-26B-A4B, e5f9bf63-4f36-489c-a3de-3cfa25bb7461
DEBUG 05-03 22:31:40.674849.674849 cuda_h.py:27] end init_experts_loading_async cost 97.862 ms
INFO 05-03 22:31:40.714684.714684 lmp.py:2561] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-03 22:31:40.820359.820359 cuda_h.py:27] end restore_state_dict cost 105.419 ms
DEBUG 05-03 22:31:40.841342.841342 cuda_h.py:27] end init_inputs_tokens cost 20.715 ms
DEBUG 05-03 22:31:40.841347.841347 lmp.py:729] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-03 22:31:41.075061.075061 cuda_h.py:27] end *sagl cost 234.273 ms
experts_cpu_alloc {'expert_ids': [11, 27, 59, 83, 99, 31, 67, 19, 43, 71, 127, 79, 87, 23, 4, 100, 72, 84, 8, 92, 120, 20, 108, 28, 44, 80, 45, 61, 85, 101, 109, 49, 65, 93, 17, 29, 77, 5, 69, 37, 9, 86, 14, 6, 94, 102, 30, 106, 114, 10, 2, 38, 118, 70], 'token_total': 499, 'token_per_expert': {11: 2, 27: 2, 59: 9, 83: 9, 99: 10, 31: 11, 67: 12, 19: 14, 43: 15, 71: 15, 127: 21, 79: 22, 87: 22, 23: 25, 4: 2, 100: 2, 72: 3, 84: 3, 8: 4, 92: 8, 120: 9, 20: 11, 108: 11, 28: 15, 44: 25, 80: 29, 45: 1, 61: 1, 85: 1, 101: 1, 109: 2, 49: 3, 65: 4, 93: 5, 17: 6, 29: 6, 77: 7, 5: 10, 69: 15, 37: 16, 9: 17, 86: 1, 14: 2, 6: 3, 94: 3, 102: 5, 30: 6, 106: 7, 114: 7, 10: 8, 2: 9, 38: 11, 118: 13, 70: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 47, 51, 55, 63, 75, 91, 103, 107, 111, 115, 123], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 987, 'token_per_expert': {3: 64, 7: 64, 39: 137, 47: 209, 51: 32, 55: 105, 63: 45, 75: 29, 91: 66, 103: 88, 107: 37, 111: 28, 115: 29, 123: 54}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 24, 32, 48, 52, 60, 64, 68, 76, 104, 112, 116, 124], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 972, 'token_per_expert': {0: 79, 16: 47, 24: 30, 32: 53, 48: 48, 52: 69, 60: 39, 64: 45, 68: 157, 76: 53, 104: 41, 112: 45, 116: 88, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 21, 25, 33, 41, 53, 73, 89, 105, 113, 117, 121, 125], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 774, 'token_per_expert': {1: 89, 13: 18, 21: 22, 25: 18, 33: 155, 41: 22, 53: 172, 73: 29, 89: 17, 105: 56, 113: 37, 117: 23, 121: 96, 125: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 22, 26, 46, 50, 54, 58, 74, 78, 90, 110, 122, 126], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 864, 'token_per_expert': {18: 43, 22: 98, 26: 52, 46: 84, 50: 71, 54: 52, 58: 24, 74: 71, 78: 37, 90: 148, 110: 38, 122: 71, 126: 75}}
INFO 05-03 22:31:41.123621.123621 lmp.py:1059] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 47.716ms | allocate_experts_across_cpu_gpu: 0.290ms
INFO 05-03 22:31:41.124468.124468 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.5299530029296875e-05 seconds
INFO 05-03 22:31:41.126548.126548 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001939535140991211 seconds
INFO 05-03 22:31:41.127743.127743 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008032321929931641 seconds
INFO 05-03 22:31:41.210126.210126 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.07747411727905273 seconds
INFO 05-03 22:31:41.211343.211343 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011925697326660156 seconds
INFO 05-03 22:31:41.232382.232382 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=3.311ms act=11.520ms bmm2=4.980ms unpad=1.103ms total=20.914ms E=32 maxT=209 S=1176 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.234802.234802 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=4.690ms act=14.846ms bmm2=0.277ms unpad=1.867ms total=21.681ms E=32 maxT=178 S=1094 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.234813.234813 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=5.243ms act=13.828ms bmm2=0.562ms unpad=2.065ms total=21.699ms E=32 maxT=148 S=957 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.234615.234615 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=5.420ms act=13.940ms bmm2=0.797ms unpad=1.987ms total=22.144ms E=32 maxT=172 S=869 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.235650.235650 lmp.py:1204] [layer_moe_fused] experts compute time: 0.023676633834838867 seconds
INFO 05-03 22:31:41.235634.235634 lmp.py:1215] [layer_moe_fused] to time: 4.696846008300781e-05 seconds
INFO 05-03 22:31:41.235934.235934 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00031828880310058594 seconds
DEBUG 05-03 22:31:41.235981.235981 cuda_h.py:27] end *layer_moe_fused cost 159.846 ms
DEBUG 05-03 22:31:41.235633.235633 cuda_h.py:27] end prefill_layer cost 394.424 ms
DEBUG 05-03 22:31:41.236561.236561 lmp.py:765] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-03 22:31:41.236077.236077 lmp.py:729] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-03 22:31:41.243419.243419 cuda_h.py:27] end *sagl cost 6.913 ms
experts_cpu_alloc {'expert_ids': [43, 95, 107, 55, 63, 11, 27, 83, 39, 51, 79, 115, 111, 48, 100, 24, 0, 124, 40, 56, 36, 84, 116, 112, 108, 69, 25, 65, 17, 29, 33, 117, 61, 41, 73, 81, 53, 77, 121, 93, 89, 105, 58, 102, 126, 54, 62, 74, 6, 86, 66, 78, 26, 110, 118, 14, 50, 70, 34], 'token_total': 549, 'token_per_expert': {43: 1, 95: 1, 107: 1, 55: 2, 63: 3, 11: 4, 27: 4, 83: 4, 39: 5, 51: 5, 79: 7, 115: 11, 111: 12, 48: 1, 100: 1, 24: 3, 0: 5, 124: 6, 40: 7, 56: 8, 36: 9, 84: 10, 116: 10, 112: 15, 108: 16, 69: 2, 25: 3, 65: 3, 17: 4, 29: 4, 33: 5, 117: 5, 61: 6, 41: 7, 73: 8, 81: 9, 53: 13, 77: 14, 121: 18, 93: 21, 89: 22, 105: 22, 58: 1, 102: 1, 126: 1, 54: 2, 62: 9, 74: 11, 6: 12, 86: 12, 66: 13, 78: 16, 26: 17, 110: 18, 118: 18, 14: 21, 50: 24, 70: 25, 34: 31}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 23, 31, 35, 47, 59, 67, 71, 87, 99, 103, 119, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 695, 'token_per_expert': {7: 36, 15: 18, 23: 35, 31: 18, 35: 26, 47: 140, 59: 28, 67: 101, 71: 13, 87: 35, 99: 75, 103: 15, 119: 31, 123: 19, 127: 105}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 16, 20, 28, 52, 60, 64, 68, 72, 76, 80, 96, 104, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 802, 'token_per_expert': {4: 19, 8: 164, 16: 18, 20: 22, 28: 112, 52: 57, 60: 24, 64: 39, 68: 56, 72: 36, 76: 21, 80: 135, 96: 40, 104: 37, 120: 22}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 37, 45, 49, 57, 85, 97, 101, 109, 113, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 863, 'token_per_expert': {1: 84, 5: 210, 9: 47, 13: 55, 21: 23, 37: 28, 45: 50, 49: 28, 57: 41, 85: 52, 97: 72, 101: 34, 109: 59, 113: 31, 125: 49}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 18, 22, 30, 38, 42, 46, 82, 90, 94, 98, 106, 114, 122], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1187, 'token_per_expert': {10: 183, 18: 48, 22: 102, 30: 88, 38: 59, 42: 69, 46: 91, 82: 133, 90: 115, 94: 51, 98: 43, 106: 46, 114: 32, 122: 127}}
INFO 05-03 22:31:41.244599.244599 lmp.py:1059] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.579ms | allocate_experts_across_cpu_gpu: 0.345ms
INFO 05-03 22:31:41.244683.244683 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.078315734863281e-05 seconds
INFO 05-03 22:31:41.245654.245654 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011131763458251953 seconds
INFO 05-03 22:31:41.246715.246715 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005753040313720703 seconds
INFO 05-03 22:31:41.264856.264856 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.018157243728637695 seconds
INFO 05-03 22:31:41.265881.265881 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010063648223876953 seconds
INFO 05-03 22:31:41.268056.268056 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.162ms act=0.171ms bmm2=0.047ms unpad=2.036ms total=2.415ms E=32 maxT=140 S=755 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.268941.268941 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.209ms act=0.115ms bmm2=0.045ms unpad=2.095ms total=2.464ms E=32 maxT=164 S=893 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.269668.269668 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.533ms act=0.126ms bmm2=0.631ms unpad=2.248ms total=3.539ms E=32 maxT=210 S=1029 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.270300.270300 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=1.022ms act=0.223ms bmm2=1.334ms unpad=1.551ms total=4.130ms E=32 maxT=183 S=1419 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.270903.270903 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004912853240966797 seconds
INFO 05-03 22:31:41.271961.271961 lmp.py:1215] [layer_moe_fused] to time: 5.245208740234375e-05 seconds
INFO 05-03 22:31:41.271099.271099 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00023698806762695312 seconds
DEBUG 05-03 22:31:41.271181.271181 cuda_h.py:27] end *layer_moe_fused cost 28.006 ms
DEBUG 05-03 22:31:41.271978.271978 cuda_h.py:27] end prefill_layer cost 35.212 ms
DEBUG 05-03 22:31:41.271574.271574 lmp.py:765] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-03 22:31:41.271217.271217 lmp.py:729] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-03 22:31:41.278776.278776 cuda_h.py:27] end *sagl cost 6.024 ms
experts_cpu_alloc {'expert_ids': [47, 59, 95, 27, 15, 75, 39, 63, 67, 79, 83, 111, 115, 55, 103, 43, 4, 40, 112, 88, 32, 76, 0, 36, 100, 12, 72, 64, 116, 56, 60, 29, 117, 101, 49, 37, 85, 93, 89, 121, 45, 21, 33, 77, 73, 22, 50, 86, 98, 6, 58, 90, 46, 10, 94, 14, 26, 66, 106], 'token_total': 444, 'token_per_expert': {47: 1, 59: 2, 95: 2, 27: 3, 15: 4, 75: 4, 39: 7, 63: 15, 67: 17, 79: 17, 83: 18, 111: 19, 115: 23, 55: 24, 103: 25, 43: 26, 4: 1, 40: 1, 112: 1, 88: 2, 32: 4, 76: 4, 0: 5, 36: 5, 100: 6, 12: 8, 72: 9, 64: 10, 116: 11, 56: 16, 60: 16, 29: 1, 117: 1, 101: 2, 49: 3, 37: 4, 85: 4, 93: 5, 89: 6, 121: 7, 45: 8, 21: 10, 33: 10, 77: 13, 73: 15, 22: 1, 50: 1, 86: 1, 98: 1, 6: 2, 58: 2, 90: 2, 46: 3, 10: 4, 94: 4, 14: 6, 26: 7, 66: 7, 106: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 51, 71, 91, 99, 107, 119, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1047, 'token_per_expert': {3: 30, 7: 60, 11: 121, 19: 114, 23: 45, 31: 41, 35: 39, 51: 119, 71: 35, 91: 91, 99: 56, 107: 77, 119: 29, 123: 105, 127: 85}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 16, 20, 24, 28, 44, 48, 52, 68, 80, 84, 96, 104, 108, 124], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1006, 'token_per_expert': {8: 18, 16: 31, 20: 17, 24: 51, 28: 20, 44: 26, 48: 153, 52: 24, 68: 42, 80: 61, 84: 68, 96: 33, 104: 107, 108: 304, 124: 51}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 25, 41, 53, 57, 61, 65, 69, 81, 97, 105, 109, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 830, 'token_per_expert': {1: 85, 9: 115, 13: 49, 25: 34, 41: 155, 53: 31, 57: 35, 61: 27, 65: 91, 69: 21, 81: 23, 97: 16, 105: 63, 109: 40, 125: 45}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 30, 42, 54, 62, 70, 74, 78, 82, 102, 110, 118, 122, 126], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 769, 'token_per_expert': {18: 68, 30: 24, 42: 26, 54: 185, 62: 61, 70: 74, 74: 31, 78: 25, 82: 34, 102: 10, 110: 42, 118: 39, 122: 87, 126: 63}}
INFO 05-03 22:31:41.279983.279983 lmp.py:1059] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.522ms | allocate_experts_across_cpu_gpu: 0.423ms
INFO 05-03 22:31:41.279325.279325 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.628036499023438e-05 seconds
INFO 05-03 22:31:41.280065.280065 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007462501525878906 seconds
INFO 05-03 22:31:41.280178.280178 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005419254302978516 seconds
INFO 05-03 22:31:41.298515.298515 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01664566993713379 seconds
INFO 05-03 22:31:41.299713.299713 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010228157043457031 seconds
INFO 05-03 22:31:41.302186.302186 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.166ms act=0.297ms bmm2=0.065ms unpad=2.501ms total=3.028ms E=32 maxT=121 S=1254 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.303732.303732 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.185ms act=0.182ms bmm2=0.050ms unpad=3.162ms total=3.578ms E=32 maxT=155 S=919 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.303671.303671 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.146ms act=0.135ms bmm2=0.064ms unpad=3.686ms total=4.032ms E=32 maxT=185 S=818 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.304286.304286 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.199ms act=0.215ms bmm2=1.335ms unpad=2.797ms total=4.546ms E=32 maxT=304 S=1105 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.304902.304902 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005184173583984375 seconds
INFO 05-03 22:31:41.304079.304079 lmp.py:1215] [layer_moe_fused] to time: 4.982948303222656e-05 seconds
INFO 05-03 22:31:41.304423.304423 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002448558807373047 seconds
DEBUG 05-03 22:31:41.304391.304391 cuda_h.py:27] end *layer_moe_fused cost 26.773 ms
DEBUG 05-03 22:31:41.305326.305326 cuda_h.py:27] end prefill_layer cost 33.121 ms
DEBUG 05-03 22:31:41.305873.305873 lmp.py:765] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-03 22:31:41.305899.305899 lmp.py:729] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-03 22:31:41.311032.311032 cuda_h.py:27] end *sagl cost 6.075 ms
experts_cpu_alloc {'expert_ids': [23, 99, 63, 103, 7, 87, 19, 55, 59, 15, 79, 67, 115, 12, 16, 112, 92, 80, 8, 24, 28, 36, 44, 48, 84, 124, 52, 60, 116, 29, 89, 105, 125, 49, 17, 53, 21, 65, 41, 81, 90, 18, 30, 38, 82, 98, 106, 58, 62, 118, 6, 46, 94, 2, 102, 114, 126, 42], 'token_total': 364, 'token_per_expert': {23: 1, 99: 1, 63: 2, 103: 2, 7: 3, 87: 4, 19: 6, 55: 6, 59: 7, 15: 9, 79: 9, 67: 10, 115: 13, 12: 1, 16: 1, 112: 1, 92: 2, 80: 4, 8: 5, 24: 5, 28: 5, 36: 6, 44: 6, 48: 6, 84: 6, 124: 7, 52: 12, 60: 12, 116: 13, 29: 1, 89: 1, 105: 1, 125: 1, 49: 2, 17: 5, 53: 5, 21: 6, 65: 8, 41: 10, 81: 11, 90: 1, 18: 2, 30: 2, 38: 2, 82: 3, 98: 3, 106: 5, 58: 6, 62: 7, 118: 7, 6: 9, 46: 10, 94: 10, 2: 14, 102: 15, 114: 15, 126: 18, 42: 19}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 27, 31, 39, 43, 51, 71, 75, 83, 95, 107, 111, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 862, 'token_per_expert': {3: 32, 11: 109, 27: 77, 31: 43, 39: 74, 43: 18, 51: 17, 71: 114, 75: 42, 83: 42, 95: 83, 107: 92, 111: 60, 123: 20, 127: 39}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 32, 40, 56, 64, 68, 72, 76, 88, 96, 100, 104, 108, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 654, 'token_per_expert': {0: 35, 4: 30, 32: 46, 40: 94, 56: 36, 64: 68, 68: 52, 72: 34, 76: 48, 88: 56, 96: 40, 100: 20, 104: 59, 108: 14, 120: 22}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 37, 61, 69, 73, 77, 85, 93, 97, 101, 113, 117, 121], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 921, 'token_per_expert': {5: 41, 9: 125, 13: 17, 37: 27, 61: 16, 69: 85, 73: 54, 77: 36, 85: 49, 93: 107, 97: 89, 101: 126, 113: 30, 117: 80, 121: 39}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 14, 22, 26, 34, 50, 54, 66, 70, 74, 78, 86, 110, 122], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1295, 'token_per_expert': {10: 128, 14: 20, 22: 114, 26: 76, 34: 58, 50: 155, 54: 104, 66: 50, 70: 55, 74: 72, 78: 227, 86: 99, 110: 19, 122: 118}}
INFO 05-03 22:31:41.312665.312665 lmp.py:1059] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.580ms | allocate_experts_across_cpu_gpu: 0.422ms
INFO 05-03 22:31:41.312571.312571 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.437301635742188e-05 seconds
INFO 05-03 22:31:41.313309.313309 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000705718994140625 seconds
INFO 05-03 22:31:41.314700.314700 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005397796630859375 seconds
INFO 05-03 22:31:41.331465.331465 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.017146825790405273 seconds
INFO 05-03 22:31:41.332919.332919 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009229183197021484 seconds
INFO 05-03 22:31:41.335307.335307 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.121ms act=0.124ms bmm2=0.074ms unpad=2.013ms total=2.331ms E=32 maxT=126 S=972 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.335214.335214 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.221ms act=0.147ms bmm2=0.056ms unpad=2.441ms total=2.866ms E=32 maxT=114 S=935 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.336589.336589 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.573ms act=0.121ms bmm2=0.276ms unpad=2.526ms total=3.495ms E=32 maxT=94 S=746 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.337534.337534 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.787ms act=0.340ms bmm2=1.374ms unpad=1.600ms total=4.101ms E=32 maxT=227 S=1443 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.337520.337520 lmp.py:1204] [layer_moe_fused] experts compute time: 0.00482177734375 seconds
INFO 05-03 22:31:41.337729.337729 lmp.py:1215] [layer_moe_fused] to time: 4.863739013671875e-05 seconds
INFO 05-03 22:31:41.337615.337615 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00022149085998535156 seconds
DEBUG 05-03 22:31:41.338174.338174 cuda_h.py:27] end *layer_moe_fused cost 26.484 ms
DEBUG 05-03 22:31:41.338633.338633 cuda_h.py:27] end prefill_layer cost 32.809 ms
DEBUG 05-03 22:31:41.338142.338142 lmp.py:765] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-03 22:31:41.338728.338728 lmp.py:729] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-03 22:31:41.343078.343078 cuda_h.py:27] end *sagl cost 5.324 ms
experts_cpu_alloc {'expert_ids': [71, 127, 119, 35, 95, 107, 11, 75, 103, 123, 87, 31, 111, 15, 68, 100, 72, 12, 28, 116, 120, 124, 48, 76, 24, 16, 40, 56, 33, 69, 113, 45, 61, 77, 125, 37, 117, 13, 121, 25, 101, 89, 10, 42, 74, 102, 122, 126, 30, 62, 90, 38, 110, 58, 94, 114, 2], 'token_total': 334, 'token_per_expert': {71: 1, 127: 1, 119: 2, 35: 3, 95: 5, 107: 5, 11: 6, 75: 6, 103: 7, 123: 9, 87: 12, 31: 15, 111: 16, 15: 17, 68: 1, 100: 1, 72: 2, 12: 3, 28: 3, 116: 4, 120: 5, 124: 5, 48: 6, 76: 6, 24: 9, 16: 11, 40: 11, 56: 19, 33: 1, 69: 1, 113: 1, 45: 2, 61: 4, 77: 5, 125: 5, 37: 6, 117: 6, 13: 8, 121: 8, 25: 11, 101: 13, 89: 15, 10: 1, 42: 1, 74: 1, 102: 1, 122: 1, 126: 1, 30: 2, 62: 2, 90: 2, 38: 5, 110: 6, 58: 7, 94: 8, 114: 9, 2: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 27, 39, 43, 47, 51, 59, 63, 67, 79, 83, 91, 115], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1197, 'token_per_expert': {3: 31, 7: 29, 19: 24, 27: 29, 39: 61, 43: 137, 47: 66, 51: 259, 59: 35, 63: 35, 67: 99, 79: 20, 83: 95, 91: 86, 115: 191}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 32, 36, 44, 52, 60, 64, 84, 92, 96, 104, 108], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 990, 'token_per_expert': {0: 32, 4: 89, 8: 28, 20: 157, 32: 66, 36: 77, 44: 67, 52: 42, 60: 48, 64: 58, 84: 22, 92: 52, 96: 57, 104: 168, 108: 27}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 29, 49, 53, 57, 65, 73, 81, 85, 93, 97, 105], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 618, 'token_per_expert': {1: 29, 5: 118, 21: 42, 29: 42, 49: 63, 53: 47, 57: 35, 65: 20, 73: 36, 81: 35, 85: 69, 93: 23, 97: 18, 105: 41}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 22, 26, 34, 46, 50, 54, 66, 78, 82, 86, 98, 106, 118], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 957, 'token_per_expert': {6: 51, 22: 84, 26: 99, 34: 31, 46: 74, 50: 72, 54: 36, 66: 61, 78: 89, 82: 36, 86: 30, 98: 57, 106: 84, 118: 153}}
INFO 05-03 22:31:41.344827.344827 lmp.py:1059] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.509ms | allocate_experts_across_cpu_gpu: 0.416ms
INFO 05-03 22:31:41.345779.345779 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.29425048828125e-05 seconds
INFO 05-03 22:31:41.345967.345967 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006814002990722656 seconds
INFO 05-03 22:31:41.346411.346411 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005452632904052734 seconds
INFO 05-03 22:31:41.363492.363492 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0165102481842041 seconds
INFO 05-03 22:31:41.364451.364451 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000988006591796875 seconds
INFO 05-03 22:31:41.366536.366536 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.099ms act=0.151ms bmm2=0.060ms unpad=1.652ms total=1.962ms E=32 maxT=118 S=704 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.367340.367340 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.106ms act=0.101ms bmm2=0.047ms unpad=2.134ms total=2.388ms E=32 maxT=153 S=1014 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.367606.367606 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=1.333ms act=0.181ms bmm2=0.066ms unpad=1.565ms total=3.144ms E=32 maxT=168 S=1076 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.369598.369598 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.153ms act=0.247ms bmm2=2.016ms unpad=2.550ms total=4.966ms E=32 maxT=259 S=1302 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.369680.369680 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005385875701904297 seconds
INFO 05-03 22:31:41.369911.369911 lmp.py:1215] [layer_moe_fused] to time: 4.935264587402344e-05 seconds
INFO 05-03 22:31:41.370819.370819 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002739429473876953 seconds
DEBUG 05-03 22:31:41.370034.370034 cuda_h.py:27] end *layer_moe_fused cost 26.502 ms
DEBUG 05-03 22:31:41.370831.370831 cuda_h.py:27] end prefill_layer cost 32.132 ms
DEBUG 05-03 22:31:41.370616.370616 lmp.py:765] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-03 22:31:41.370141.370141 lmp.py:729] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-03 22:31:41.416034.416034 cuda_h.py:27] end *sagl cost 45.321 ms
experts_cpu_alloc {'expert_ids': [15, 83, 11, 43, 91, 103, 79, 67, 95, 27, 31, 19, 55, 119, 4, 60, 92, 8, 100, 40, 48, 80, 12, 76, 16, 36, 116, 112, 41, 69, 121, 5, 17, 65, 109, 29, 37, 125, 85, 77, 14, 34, 38, 66, 70, 10, 46, 18, 86, 98, 102, 62, 118, 126], 'token_total': 324, 'token_per_expert': {15: 1, 83: 1, 11: 3, 43: 3, 91: 3, 103: 7, 79: 9, 67: 10, 95: 13, 27: 17, 31: 18, 19: 20, 55: 21, 119: 22, 4: 1, 60: 1, 92: 1, 8: 2, 100: 2, 40: 3, 48: 3, 80: 5, 12: 6, 76: 8, 16: 10, 36: 10, 116: 15, 112: 19, 41: 1, 69: 1, 121: 1, 5: 2, 17: 2, 65: 2, 109: 2, 29: 3, 37: 5, 125: 5, 85: 6, 77: 8, 14: 1, 34: 1, 38: 2, 66: 2, 70: 2, 10: 3, 46: 3, 18: 4, 86: 5, 98: 5, 102: 5, 62: 6, 118: 6, 126: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 23, 39, 51, 63, 71, 75, 87, 99, 107, 111, 115, 123, 127], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1109, 'token_per_expert': {3: 38, 23: 37, 39: 208, 51: 23, 63: 34, 71: 326, 75: 43, 87: 59, 99: 114, 107: 35, 111: 76, 115: 30, 123: 47, 127: 39}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 24, 28, 32, 44, 52, 56, 64, 68, 72, 88, 96, 104, 120], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 812, 'token_per_expert': {20: 185, 24: 89, 28: 26, 32: 31, 44: 42, 52: 30, 56: 20, 64: 64, 68: 22, 72: 34, 88: 32, 96: 68, 104: 93, 120: 76}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 21, 33, 49, 57, 61, 73, 81, 93, 97, 101, 113, 117], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1131, 'token_per_expert': {1: 63, 13: 94, 21: 25, 33: 84, 49: 133, 57: 29, 61: 181, 73: 63, 81: 8, 93: 10, 97: 54, 101: 193, 113: 66, 117: 128}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 30, 50, 58, 74, 82, 90, 94, 110, 114, 122], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 720, 'token_per_expert': {2: 134, 6: 72, 22: 166, 30: 14, 50: 59, 58: 28, 74: 107, 82: 21, 90: 9, 94: 63, 110: 16, 114: 9, 122: 22}}
INFO 05-03 22:31:41.417315.417315 lmp.py:1059] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.489ms | allocate_experts_across_cpu_gpu: 0.271ms
INFO 05-03 22:31:41.417802.417802 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.172325134277344e-05 seconds
INFO 05-03 22:31:41.418698.418698 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010647773742675781 seconds
INFO 05-03 22:31:41.418300.418300 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004904270172119141 seconds
INFO 05-03 22:31:41.434780.434780 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.015637636184692383 seconds
INFO 05-03 22:31:41.435481.435481 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010077953338623047 seconds
INFO 05-03 22:31:41.438401.438401 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.149ms act=0.127ms bmm2=0.029ms unpad=2.126ms total=2.431ms E=32 maxT=185 S=898 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.439750.439750 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.158ms act=0.104ms bmm2=0.064ms unpad=2.795ms total=3.121ms E=32 maxT=193 S=1169 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.440417.440417 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.845ms act=0.218ms bmm2=0.079ms unpad=2.834ms total=3.975ms E=32 maxT=166 S=772 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.440164.440164 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.667ms act=0.093ms bmm2=0.892ms unpad=2.892ms total=4.544ms E=32 maxT=326 S=1257 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.440036.440036 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005030632019042969 seconds
INFO 05-03 22:31:41.441537.441537 lmp.py:1215] [layer_moe_fused] to time: 5.0067901611328125e-05 seconds
INFO 05-03 22:31:41.441900.441900 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00022411346435546875 seconds
DEBUG 05-03 22:31:41.441016.441016 cuda_h.py:27] end *layer_moe_fused cost 25.262 ms
DEBUG 05-03 22:31:41.441044.441044 cuda_h.py:27] end prefill_layer cost 70.890 ms
DEBUG 05-03 22:31:41.441766.441766 lmp.py:765] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-03 22:31:41.441029.441029 lmp.py:729] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-03 22:31:41.447861.447861 cuda_h.py:27] end *sagl cost 5.065 ms
experts_cpu_alloc {'expert_ids': [43, 59, 127, 15, 27, 39, 47, 107, 7, 11, 83, 71, 67, 95, 111, 23, 60, 72, 92, 100, 8, 28, 48, 96, 4, 12, 36, 52, 56, 9, 61, 81, 1, 45, 5, 21, 29, 65, 37, 49, 69, 121, 25, 2, 30, 38, 58, 114, 14, 70, 106, 6, 18, 86, 54, 78, 98, 66], 'token_total': 202, 'token_per_expert': {43: 1, 59: 1, 127: 1, 15: 2, 27: 2, 39: 2, 47: 2, 107: 2, 7: 3, 11: 3, 83: 4, 71: 5, 67: 6, 95: 7, 111: 7, 23: 9, 60: 1, 72: 1, 92: 1, 100: 1, 8: 3, 28: 3, 48: 3, 96: 3, 4: 4, 12: 4, 36: 4, 52: 5, 56: 6, 9: 1, 61: 1, 81: 1, 1: 2, 45: 2, 5: 3, 21: 3, 29: 3, 65: 4, 37: 5, 49: 8, 69: 8, 121: 9, 25: 10, 2: 1, 30: 1, 38: 1, 58: 1, 114: 1, 14: 2, 70: 2, 106: 2, 6: 3, 18: 4, 86: 4, 54: 5, 78: 6, 98: 6, 66: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 19, 31, 35, 51, 55, 63, 75, 79, 87, 91, 99, 115, 119, 123], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 551, 'token_per_expert': {3: 37, 19: 23, 31: 12, 35: 56, 51: 21, 55: 18, 63: 14, 75: 136, 79: 87, 87: 38, 91: 16, 99: 53, 115: 11, 119: 9, 123: 20}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 20, 32, 40, 64, 68, 80, 84, 104, 108, 112, 116, 120, 124], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 723, 'token_per_expert': {0: 43, 16: 29, 20: 42, 32: 22, 40: 88, 64: 119, 68: 209, 80: 9, 84: 14, 104: 23, 108: 47, 112: 23, 116: 6, 120: 10, 124: 39}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 17, 33, 41, 53, 57, 73, 77, 89, 93, 97, 109, 113, 117, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1420, 'token_per_expert': {13: 42, 17: 12, 33: 18, 41: 81, 53: 102, 57: 28, 73: 33, 77: 217, 89: 53, 93: 269, 97: 208, 109: 20, 113: 230, 117: 10, 125: 97}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 34, 42, 46, 62, 74, 82, 90, 94, 102, 110, 118, 122, 126], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1200, 'token_per_expert': {10: 43, 34: 299, 42: 207, 46: 248, 62: 16, 74: 12, 82: 49, 90: 60, 94: 28, 102: 112, 110: 42, 118: 26, 122: 29, 126: 29}}
INFO 05-03 22:31:41.448564.448564 lmp.py:1059] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.531ms | allocate_experts_across_cpu_gpu: 0.432ms
INFO 05-03 22:31:41.448099.448099 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.508827209472656e-05 seconds
INFO 05-03 22:31:41.449147.449147 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007278919219970703 seconds
INFO 05-03 22:31:41.450234.450234 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005617141723632812 seconds
INFO 05-03 22:31:41.469932.469932 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01914381980895996 seconds
INFO 05-03 22:31:41.470898.470898 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010249614715576172 seconds
INFO 05-03 22:31:41.473726.473726 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.139ms act=0.124ms bmm2=0.033ms unpad=2.533ms total=2.830ms E=32 maxT=209 S=762 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.474210.474210 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.125ms act=0.096ms bmm2=0.374ms unpad=3.323ms total=3.918ms E=32 maxT=299 S=1246 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.475509.475509 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.596ms act=0.093ms bmm2=0.044ms unpad=3.605ms total=4.337ms E=32 maxT=136 S=608 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.475844.475844 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.873ms act=0.108ms bmm2=0.478ms unpad=3.049ms total=4.508ms E=32 maxT=269 S=1480 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.475285.475285 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005315065383911133 seconds
INFO 05-03 22:31:41.476739.476739 lmp.py:1215] [layer_moe_fused] to time: 5.173683166503906e-05 seconds
INFO 05-03 22:31:41.476009.476009 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00040221214294433594 seconds
DEBUG 05-03 22:31:41.476987.476987 cuda_h.py:27] end *layer_moe_fused cost 29.401 ms
DEBUG 05-03 22:31:41.476492.476492 cuda_h.py:27] end prefill_layer cost 34.786 ms
DEBUG 05-03 22:31:41.476709.476709 lmp.py:765] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-03 22:31:41.477365.477365 lmp.py:729] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-03 22:31:41.482554.482554 cuda_h.py:27] end *sagl cost 5.657 ms
experts_cpu_alloc {'expert_ids': [75, 79, 99, 119, 127, 15, 19, 7, 27, 71, 55, 111, 31, 35, 47, 16, 40, 64, 120, 32, 12, 36, 104, 72, 124, 28, 100, 108, 0, 21, 37, 57, 61, 73, 97, 101, 1, 5, 29, 26, 74, 110, 50, 54, 18, 2, 14, 22, 118], 'token_total': 168, 'token_per_expert': {75: 1, 79: 1, 99: 1, 119: 1, 127: 1, 15: 2, 19: 2, 7: 3, 27: 3, 71: 3, 55: 5, 111: 5, 31: 9, 35: 9, 47: 9, 16: 1, 40: 1, 64: 1, 120: 1, 32: 2, 12: 3, 36: 3, 104: 4, 72: 5, 124: 8, 28: 9, 100: 10, 108: 10, 0: 11, 21: 1, 37: 1, 57: 1, 61: 1, 73: 1, 97: 1, 101: 2, 1: 3, 5: 3, 29: 4, 26: 1, 74: 1, 110: 1, 50: 2, 54: 2, 18: 3, 2: 4, 14: 4, 22: 4, 118: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 39, 43, 59, 63, 83, 87, 91, 95, 103, 115, 123], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 657, 'token_per_expert': {3: 25, 11: 10, 39: 21, 43: 13, 59: 113, 63: 11, 83: 35, 87: 11, 91: 17, 95: 209, 103: 39, 115: 23, 123: 130}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 20, 24, 44, 48, 52, 56, 60, 84, 88, 96, 116], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1808, 'token_per_expert': {4: 185, 8: 14, 20: 338, 24: 148, 44: 27, 48: 164, 52: 316, 56: 147, 60: 102, 84: 29, 88: 13, 96: 257, 116: 68}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 45, 65, 69, 77, 85, 105, 109, 113, 117, 121, 125], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 675, 'token_per_expert': {9: 8, 45: 15, 65: 13, 69: 186, 77: 10, 85: 6, 105: 60, 109: 207, 113: 13, 117: 13, 121: 128, 125: 16}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 30, 34, 62, 70, 90, 98, 106, 114, 122, 126], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 788, 'token_per_expert': {6: 154, 10: 141, 30: 50, 34: 11, 62: 7, 70: 19, 90: 12, 98: 17, 106: 323, 114: 10, 122: 38, 126: 6}}
INFO 05-03 22:31:41.483583.483583 lmp.py:1059] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.578ms | allocate_experts_across_cpu_gpu: 0.398ms
INFO 05-03 22:31:41.484481.484481 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.507469177246094e-05 seconds
INFO 05-03 22:31:41.485604.485604 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007252693176269531 seconds
INFO 05-03 22:31:41.485267.485267 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005292892456054688 seconds
INFO 05-03 22:31:41.503809.503809 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01765298843383789 seconds
INFO 05-03 22:31:41.504773.504773 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009558200836181641 seconds
INFO 05-03 22:31:41.506285.506285 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.116ms act=0.134ms bmm2=0.058ms unpad=1.181ms total=1.490ms E=32 maxT=207 S=693 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.508478.508478 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.131ms act=0.095ms bmm2=1.351ms unpad=1.859ms total=3.436ms E=32 maxT=323 S=814 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.508119.508119 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.710ms act=0.084ms bmm2=0.031ms unpad=2.955ms total=3.779ms E=32 maxT=209 S=712 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.508740.508740 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.141ms act=0.175ms bmm2=1.118ms unpad=2.417ms total=3.851ms E=32 maxT=338 S=1877 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.509068.509068 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004769086837768555 seconds
INFO 05-03 22:31:41.509515.509515 lmp.py:1215] [layer_moe_fused] to time: 4.9114227294921875e-05 seconds
INFO 05-03 22:31:41.509186.509186 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0003085136413574219 seconds
DEBUG 05-03 22:31:41.510580.510580 cuda_h.py:27] end *layer_moe_fused cost 26.996 ms
DEBUG 05-03 22:31:41.510900.510900 cuda_h.py:27] end prefill_layer cost 32.978 ms
DEBUG 05-03 22:31:41.510494.510494 lmp.py:765] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-03 22:31:41.510705.510705 lmp.py:729] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-03 22:31:41.516309.516309 cuda_h.py:27] end *sagl cost 5.620 ms
experts_cpu_alloc {'expert_ids': [35, 127, 43, 47, 111, 39, 19, 3, 7, 107, 8, 36, 48, 28, 32, 12, 56, 60, 76, 104, 24, 0, 100, 61, 17, 9, 57, 101, 117, 65, 69, 105, 1, 5, 41, 33, 49, 70, 74, 94, 102, 2, 38, 14, 110, 126], 'token_total': 263, 'token_per_expert': {35: 1, 127: 1, 43: 2, 47: 2, 111: 2, 39: 4, 19: 5, 3: 6, 7: 6, 107: 7, 8: 1, 36: 1, 48: 1, 28: 2, 32: 2, 12: 3, 56: 3, 60: 4, 76: 4, 104: 4, 24: 5, 0: 6, 100: 7, 61: 1, 17: 2, 9: 3, 57: 5, 101: 7, 117: 7, 65: 8, 69: 10, 105: 10, 1: 12, 5: 12, 41: 16, 33: 22, 49: 23, 70: 1, 74: 1, 94: 1, 102: 2, 2: 5, 38: 7, 14: 9, 110: 10, 126: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 27, 31, 51, 55, 63, 71, 79, 87, 91, 103, 123], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 389, 'token_per_expert': {15: 12, 27: 12, 31: 30, 51: 20, 55: 25, 63: 14, 71: 19, 79: 34, 87: 11, 91: 18, 103: 56, 123: 138}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 16, 40, 44, 64, 68, 80, 84, 92, 96, 120, 124], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 563, 'token_per_expert': {4: 8, 16: 15, 40: 10, 44: 15, 64: 10, 68: 27, 80: 9, 84: 135, 92: 174, 96: 8, 120: 11, 124: 141}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 21, 29, 73, 77, 81, 85, 89, 93, 121, 125], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1671, 'token_per_expert': {13: 184, 21: 34, 29: 37, 73: 222, 77: 171, 81: 198, 85: 33, 89: 189, 93: 29, 121: 290, 125: 284}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 18, 22, 30, 46, 50, 54, 58, 86, 98, 118], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1210, 'token_per_expert': {6: 408, 18: 80, 22: 114, 30: 82, 46: 81, 50: 17, 54: 32, 58: 228, 86: 15, 98: 139, 118: 14}}
INFO 05-03 22:31:41.517798.517798 lmp.py:1059] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.507ms | allocate_experts_across_cpu_gpu: 0.367ms
INFO 05-03 22:31:41.517935.517935 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.435943603515625e-05 seconds
INFO 05-03 22:31:41.518763.518763 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006983280181884766 seconds
INFO 05-03 22:31:41.518586.518586 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005769729614257812 seconds
INFO 05-03 22:31:41.536676.536676 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.017238855361938477 seconds
INFO 05-03 22:31:41.537738.537738 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009250640869140625 seconds
INFO 05-03 22:31:41.540505.540505 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.211ms act=0.178ms bmm2=0.044ms unpad=1.661ms total=2.093ms E=32 maxT=138 S=425 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.541719.541719 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.132ms act=0.136ms bmm2=0.782ms unpad=2.319ms total=3.368ms E=32 maxT=408 S=1256 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.541419.541419 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.117ms act=0.156ms bmm2=0.378ms unpad=2.903ms total=3.553ms E=32 maxT=290 S=1809 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.541093.541093 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.467ms act=0.087ms bmm2=0.040ms unpad=3.166ms total=3.760ms E=32 maxT=174 S=606 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.542040.542040 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004778146743774414 seconds
INFO 05-03 22:31:41.542395.542395 lmp.py:1215] [layer_moe_fused] to time: 5.0067901611328125e-05 seconds
INFO 05-03 22:31:41.542482.542482 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002655982971191406 seconds
DEBUG 05-03 22:31:41.543405.543405 cuda_h.py:27] end *layer_moe_fused cost 26.920 ms
DEBUG 05-03 22:31:41.543964.543964 cuda_h.py:27] end prefill_layer cost 32.844 ms
DEBUG 05-03 22:31:41.543625.543625 lmp.py:765] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-03 22:31:41.543772.543772 lmp.py:729] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-03 22:31:41.551861.551861 cuda_h.py:27] end *sagl cost 8.052 ms
experts_cpu_alloc {'expert_ids': [3, 23, 111, 75, 7, 103, 20, 104, 84, 120, 60, 92, 4, 8, 36, 0, 112, 64, 33, 45, 109, 117, 121, 69, 89, 61, 21, 5, 1, 34, 66, 14, 30, 102, 26, 74, 18, 82], 'token_total': 190, 'token_per_expert': {3: 2, 23: 2, 111: 3, 75: 5, 7: 7, 103: 12, 20: 1, 104: 1, 84: 2, 120: 2, 60: 3, 92: 4, 4: 6, 8: 6, 36: 6, 0: 8, 112: 8, 64: 12, 33: 1, 45: 1, 109: 1, 117: 1, 121: 2, 69: 3, 89: 3, 61: 5, 21: 6, 5: 9, 1: 14, 34: 1, 66: 1, 14: 4, 30: 5, 102: 5, 26: 8, 74: 9, 18: 10, 82: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 19, 27, 39, 43, 51, 79, 83, 91, 95], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1041, 'token_per_expert': {15: 343, 19: 83, 27: 15, 39: 41, 43: 37, 51: 118, 79: 43, 83: 15, 91: 104, 95: 242}}
experts_gpu_alloc_device_1 {'expert_ids': [12, 16, 24, 28, 32, 48, 68, 88, 108, 124], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1016, 'token_per_expert': {12: 212, 16: 143, 24: 37, 28: 141, 32: 15, 48: 136, 68: 23, 88: 226, 108: 16, 124: 67}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 13, 25, 29, 37, 41, 57, 65, 97], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 850, 'token_per_expert': {9: 338, 13: 30, 25: 40, 29: 65, 37: 30, 41: 16, 57: 40, 65: 17, 97: 274}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 22, 46, 70, 86, 98, 106, 114, 122], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 999, 'token_per_expert': {2: 54, 22: 323, 46: 22, 70: 13, 86: 12, 98: 228, 106: 156, 114: 179, 122: 12}}
INFO 05-03 22:31:41.552587.552587 lmp.py:1059] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.498ms | allocate_experts_across_cpu_gpu: 0.306ms
INFO 05-03 22:31:41.552094.552094 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.8160552978515625e-05 seconds
INFO 05-03 22:31:41.553374.553374 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006563663482666016 seconds
INFO 05-03 22:31:41.554486.554486 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005121231079101562 seconds
INFO 05-03 22:31:41.570363.570363 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.016103267669677734 seconds
INFO 05-03 22:31:41.571886.571886 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008606910705566406 seconds
INFO 05-03 22:31:41.573614.573614 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.182ms act=0.186ms bmm2=0.056ms unpad=1.636ms total=2.060ms E=32 maxT=226 S=1075 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.574214.574214 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.075ms act=0.171ms bmm2=0.031ms unpad=2.558ms total=2.835ms E=32 maxT=323 S=1053 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.574509.574509 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.220ms act=0.190ms bmm2=0.442ms unpad=2.288ms total=3.140ms E=32 maxT=343 S=1072 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.575006.575006 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.185ms act=0.134ms bmm2=0.746ms unpad=2.346ms total=3.411ms E=32 maxT=338 S=896 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.575680.575680 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004212379455566406 seconds
INFO 05-03 22:31:41.576823.576823 lmp.py:1215] [layer_moe_fused] to time: 5.078315734863281e-05 seconds
INFO 05-03 22:31:41.576764.576764 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002639293670654297 seconds
DEBUG 05-03 22:31:41.576979.576979 cuda_h.py:27] end *layer_moe_fused cost 24.797 ms
DEBUG 05-03 22:31:41.576299.576299 cuda_h.py:27] end prefill_layer cost 33.148 ms
DEBUG 05-03 22:31:41.576074.576074 lmp.py:765] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-03 22:31:41.576363.576363 lmp.py:729] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-03 22:31:41.582247.582247 cuda_h.py:27] end *sagl cost 5.467 ms
experts_cpu_alloc {'expert_ids': [15, 47, 83, 119, 75, 51, 127, 19, 107, 99, 87, 63, 91, 71, 103, 124, 60, 72, 84, 104, 88, 28, 52, 92, 1, 117, 25, 69, 57, 61, 9, 22, 34, 70, 114, 58, 74, 62, 106], 'token_total': 159, 'token_per_expert': {15: 1, 47: 1, 83: 1, 119: 1, 75: 2, 51: 3, 127: 3, 19: 4, 107: 4, 99: 5, 87: 9, 63: 14, 91: 14, 71: 17, 103: 17, 124: 1, 60: 2, 72: 2, 84: 2, 104: 2, 88: 4, 28: 5, 52: 5, 92: 7, 1: 1, 117: 1, 25: 2, 69: 2, 57: 3, 61: 3, 9: 7, 22: 1, 34: 1, 70: 1, 114: 1, 58: 2, 74: 2, 62: 3, 106: 3}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 31, 39, 43, 67, 79, 111, 115], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1055, 'token_per_expert': {3: 59, 7: 69, 11: 145, 31: 22, 39: 22, 43: 108, 67: 32, 79: 52, 111: 158, 115: 388}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 20, 44, 48, 56, 68, 80, 100, 108, 120], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 876, 'token_per_expert': {8: 20, 20: 30, 44: 30, 48: 28, 56: 17, 68: 215, 80: 56, 100: 133, 108: 337, 120: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 29, 33, 37, 41, 53, 81, 113, 121, 125], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1399, 'token_per_expert': {13: 120, 29: 274, 33: 369, 37: 65, 41: 11, 53: 210, 81: 174, 113: 41, 121: 25, 125: 110}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 14, 18, 46, 66, 82, 94, 98], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 607, 'token_per_expert': {6: 58, 10: 26, 14: 17, 18: 56, 46: 24, 66: 10, 82: 282, 94: 123, 98: 11}}
INFO 05-03 22:31:41.583569.583569 lmp.py:1059] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.506ms | allocate_experts_across_cpu_gpu: 0.315ms
INFO 05-03 22:31:41.583653.583653 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.935264587402344e-05 seconds
INFO 05-03 22:31:41.584883.584883 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006701946258544922 seconds
INFO 05-03 22:31:41.585942.585942 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005052089691162109 seconds
INFO 05-03 22:31:41.600920.600920 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0154571533203125 seconds
INFO 05-03 22:31:41.601019.601019 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008635520935058594 seconds
INFO 05-03 22:31:41.604375.604375 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.166ms act=0.105ms bmm2=0.044ms unpad=1.993ms total=2.308ms E=32 maxT=282 S=621 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.604794.604794 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.202ms act=0.201ms bmm2=0.064ms unpad=2.420ms total=2.886ms E=32 maxT=337 S=906 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.605446.605446 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.216ms act=0.205ms bmm2=0.314ms unpad=2.694ms total=3.429ms E=32 maxT=388 S=1151 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.605057.605057 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.114ms act=0.188ms bmm2=1.355ms unpad=2.097ms total=3.754ms E=32 maxT=369 S=1418 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.606489.606489 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004447460174560547 seconds
INFO 05-03 22:31:41.606606.606606 lmp.py:1215] [layer_moe_fused] to time: 4.982948303222656e-05 seconds
INFO 05-03 22:31:41.606114.606114 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00022339820861816406 seconds
DEBUG 05-03 22:31:41.606138.606138 cuda_h.py:27] end *layer_moe_fused cost 24.135 ms
DEBUG 05-03 22:31:41.606835.606835 cuda_h.py:27] end prefill_layer cost 29.912 ms
DEBUG 05-03 22:31:41.607450.607450 lmp.py:765] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-03 22:31:41.607926.607926 lmp.py:729] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-03 22:31:41.614527.614527 cuda_h.py:27] end *sagl cost 6.899 ms
experts_cpu_alloc {'expert_ids': [19, 59, 67, 103, 111, 71, 115, 27, 99, 11, 7, 0, 8, 56, 80, 84, 108, 16, 20, 92, 12, 68, 24, 124, 100, 104, 5, 13, 37, 73, 97, 105, 121, 125, 21, 26, 30, 34, 46, 78, 42, 70, 114, 126], 'token_total': 140, 'token_per_expert': {19: 1, 59: 1, 67: 1, 103: 1, 111: 1, 71: 2, 115: 2, 27: 4, 99: 4, 11: 7, 7: 11, 0: 1, 8: 1, 56: 1, 80: 1, 84: 1, 108: 1, 16: 3, 20: 3, 92: 3, 12: 4, 68: 5, 24: 7, 124: 11, 100: 14, 104: 24, 5: 1, 13: 1, 37: 1, 73: 1, 97: 1, 105: 1, 121: 1, 125: 2, 21: 3, 26: 1, 30: 1, 34: 1, 46: 1, 78: 1, 42: 2, 70: 2, 114: 2, 126: 2}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 23, 31, 43, 63, 79, 83, 87, 91, 119, 127], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1517, 'token_per_expert': {3: 355, 23: 380, 31: 13, 43: 183, 63: 23, 79: 19, 83: 16, 87: 13, 91: 204, 119: 278, 127: 33}}
experts_gpu_alloc_device_1 {'expert_ids': [28, 32, 36, 40, 44, 48, 64, 76, 88, 112, 120], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1648, 'token_per_expert': {28: 62, 32: 72, 36: 213, 40: 46, 44: 317, 48: 154, 64: 91, 76: 309, 88: 35, 112: 68, 120: 281}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 17, 25, 33, 49, 61, 65, 77, 81, 93, 117], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 487, 'token_per_expert': {1: 9, 17: 220, 25: 3, 33: 8, 49: 51, 61: 8, 65: 12, 77: 20, 81: 56, 93: 34, 117: 66}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 18, 22, 38, 50, 54, 62, 74, 98, 102, 118], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 304, 'token_per_expert': {6: 15, 18: 10, 22: 22, 38: 9, 50: 4, 54: 78, 62: 96, 74: 49, 98: 6, 102: 7, 118: 8}}
INFO 05-03 22:31:41.615082.615082 lmp.py:1059] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.504ms | allocate_experts_across_cpu_gpu: 0.346ms
INFO 05-03 22:31:41.615243.615243 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.628036499023438e-05 seconds
INFO 05-03 22:31:41.616619.616619 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006823539733886719 seconds
INFO 05-03 22:31:41.616698.616698 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005211830139160156 seconds
INFO 05-03 22:31:41.631501.631501 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014284849166870117 seconds
INFO 05-03 22:31:41.632377.632377 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009038448333740234 seconds
INFO 05-03 22:31:41.635391.635391 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.106ms act=0.114ms bmm2=0.069ms unpad=1.865ms total=2.154ms E=32 maxT=96 S=317 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.636946.636946 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.179ms act=0.162ms bmm2=0.044ms unpad=2.870ms total=3.255ms E=32 maxT=317 S=1728 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.636797.636797 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.651ms act=0.088ms bmm2=0.045ms unpad=2.719ms total=3.502ms E=32 maxT=220 S=499 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.636641.636641 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.153ms act=0.272ms bmm2=0.056ms unpad=3.422ms total=3.903ms E=32 maxT=380 S=1552 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.637887.637887 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004476308822631836 seconds
INFO 05-03 22:31:41.637480.637480 lmp.py:1215] [layer_moe_fused] to time: 5.0067901611328125e-05 seconds
INFO 05-03 22:31:41.637512.637512 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002243518829345703 seconds
DEBUG 05-03 22:31:41.637563.637563 cuda_h.py:27] end *layer_moe_fused cost 23.329 ms
DEBUG 05-03 22:31:41.637836.637836 cuda_h.py:27] end prefill_layer cost 30.551 ms
DEBUG 05-03 22:31:41.637088.637088 lmp.py:765] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-03 22:31:41.638921.638921 lmp.py:729] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-03 22:31:41.643653.643653 cuda_h.py:27] end *sagl cost 4.905 ms
experts_cpu_alloc {'expert_ids': [43, 47, 71, 127, 11, 91, 103, 99, 15, 95, 39, 59, 24, 60, 108, 4, 8, 44, 76, 97, 53, 61, 5, 33, 93, 37, 49, 26, 70, 118, 46, 2, 10, 82, 18, 58, 6, 94, 98, 106, 22, 50, 54], 'token_total': 156, 'token_per_expert': {43: 1, 47: 1, 71: 1, 127: 1, 11: 2, 91: 2, 103: 2, 99: 3, 15: 4, 95: 4, 39: 13, 59: 13, 24: 1, 60: 1, 108: 1, 4: 2, 8: 2, 44: 2, 76: 2, 97: 1, 53: 2, 61: 2, 5: 3, 33: 3, 93: 3, 37: 4, 49: 4, 26: 1, 70: 1, 118: 1, 46: 3, 2: 4, 10: 4, 82: 4, 18: 5, 58: 5, 6: 6, 94: 6, 98: 6, 106: 6, 22: 7, 50: 7, 54: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 19, 23, 27, 31, 35, 67, 107, 111, 119, 123], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1522, 'token_per_expert': {3: 129, 19: 54, 23: 84, 27: 57, 31: 160, 35: 98, 67: 24, 107: 353, 111: 28, 119: 341, 123: 194}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 36, 40, 72, 84, 88, 92, 104, 116, 120, 124], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 509, 'token_per_expert': {20: 4, 36: 8, 40: 176, 72: 3, 84: 130, 88: 96, 92: 5, 104: 5, 116: 3, 120: 5, 124: 74}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 17, 21, 41, 45, 85, 101, 105, 109, 117], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1068, 'token_per_expert': {1: 26, 13: 196, 17: 29, 21: 20, 41: 147, 45: 372, 85: 5, 101: 173, 105: 57, 109: 4, 117: 39}}
experts_gpu_alloc_device_3 {'expert_ids': [14, 34, 38, 74, 78, 86, 90, 102, 110, 122], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 841, 'token_per_expert': {14: 27, 34: 210, 38: 14, 74: 283, 78: 11, 86: 20, 90: 61, 102: 23, 110: 178, 122: 14}}
INFO 05-03 22:31:41.644453.644453 lmp.py:1059] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.518ms | allocate_experts_across_cpu_gpu: 0.339ms
INFO 05-03 22:31:41.644213.644213 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.435943603515625e-05 seconds
INFO 05-03 22:31:41.645792.645792 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006499290466308594 seconds
INFO 05-03 22:31:41.645791.645791 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004985332489013672 seconds
INFO 05-03 22:31:41.661955.661955 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.015451192855834961 seconds
INFO 05-03 22:31:41.662200.662200 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008633136749267578 seconds
INFO 05-03 22:31:41.664051.664051 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.181ms act=0.185ms bmm2=0.054ms unpad=1.576ms total=1.996ms E=32 maxT=176 S=520 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.665255.665255 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.152ms act=0.267ms bmm2=0.057ms unpad=2.632ms total=3.108ms E=32 maxT=372 S=1090 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.666273.666273 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.155ms act=0.104ms bmm2=0.046ms unpad=3.068ms total=3.372ms E=32 maxT=283 S=917 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.666721.666721 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.212ms act=0.198ms bmm2=0.057ms unpad=3.209ms total=3.677ms E=32 maxT=353 S=1569 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.666710.666710 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004302024841308594 seconds
INFO 05-03 22:31:41.666258.666258 lmp.py:1215] [layer_moe_fused] to time: 5.0067901611328125e-05 seconds
INFO 05-03 22:31:41.666310.666310 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00023984909057617188 seconds
DEBUG 05-03 22:31:41.667095.667095 cuda_h.py:27] end *layer_moe_fused cost 23.980 ms
DEBUG 05-03 22:31:41.667845.667845 cuda_h.py:27] end prefill_layer cost 29.192 ms
DEBUG 05-03 22:31:41.667917.667917 lmp.py:765] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-03 22:31:41.667174.667174 lmp.py:729] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-03 22:31:41.672043.672043 cuda_h.py:27] end *sagl cost 5.004 ms
experts_cpu_alloc {'expert_ids': [3, 43, 107, 15, 87, 91, 31, 4, 48, 68, 108, 112, 88, 24, 76, 0, 8, 5, 9, 37, 65, 97, 113, 17, 25, 85, 93, 109, 13, 14, 18, 94, 126, 90, 106, 46, 54, 82, 50, 22, 58, 74, 10, 118, 114, 2], 'token_total': 249, 'token_per_expert': {3: 1, 43: 1, 107: 5, 15: 8, 87: 10, 91: 13, 31: 15, 4: 1, 48: 1, 68: 1, 108: 2, 112: 2, 88: 6, 24: 10, 76: 10, 0: 11, 8: 14, 5: 1, 9: 1, 37: 1, 65: 1, 97: 1, 113: 1, 17: 2, 25: 2, 85: 3, 93: 3, 109: 3, 13: 9, 14: 1, 18: 1, 94: 1, 126: 1, 90: 2, 106: 3, 46: 4, 54: 4, 82: 5, 50: 6, 22: 7, 58: 9, 74: 11, 10: 13, 118: 13, 114: 14, 2: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 19, 39, 47, 55, 59, 67, 75, 79, 99, 115, 119], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 811, 'token_per_expert': {7: 69, 19: 17, 39: 75, 47: 102, 55: 17, 59: 17, 67: 125, 75: 42, 79: 20, 99: 101, 115: 128, 119: 98}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 32, 40, 56, 60, 64, 80, 84, 100, 104, 116, 120], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1097, 'token_per_expert': {20: 113, 32: 14, 40: 295, 56: 46, 60: 297, 64: 178, 80: 19, 84: 24, 100: 16, 104: 19, 116: 24, 120: 52}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 33, 45, 61, 73, 81, 89, 105, 117, 121], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1223, 'token_per_expert': {1: 11, 21: 12, 33: 157, 45: 24, 61: 10, 73: 306, 81: 75, 89: 189, 105: 138, 117: 260, 121: 41}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 34, 38, 42, 70, 78, 86, 98, 102, 110, 122], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 716, 'token_per_expert': {6: 34, 34: 134, 38: 15, 42: 30, 70: 77, 78: 144, 86: 16, 98: 105, 102: 81, 110: 21, 122: 59}}
INFO 05-03 22:31:41.673505.673505 lmp.py:1059] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.498ms | allocate_experts_across_cpu_gpu: 0.356ms
INFO 05-03 22:31:41.673788.673788 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.6743621826171875e-05 seconds
INFO 05-03 22:31:41.674068.674068 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006587505340576172 seconds
INFO 05-03 22:31:41.675802.675802 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005121231079101562 seconds
INFO 05-03 22:31:41.690693.690693 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014608621597290039 seconds
INFO 05-03 22:31:41.691814.691814 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009059906005859375 seconds
INFO 05-03 22:31:41.693220.693220 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.212ms act=0.224ms bmm2=0.058ms unpad=1.724ms total=2.218ms E=32 maxT=128 S=864 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.694058.694058 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.144ms act=0.230ms bmm2=0.070ms unpad=2.759ms total=3.202ms E=32 maxT=306 S=1251 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.694027.694027 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.180ms act=0.200ms bmm2=0.075ms unpad=3.070ms total=3.524ms E=32 maxT=297 S=1155 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.695616.695616 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.153ms act=0.135ms bmm2=0.122ms unpad=3.182ms total=3.591ms E=32 maxT=144 S=826 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.695222.695222 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0044023990631103516 seconds
INFO 05-03 22:31:41.695875.695875 lmp.py:1215] [layer_moe_fused] to time: 5.8650970458984375e-05 seconds
INFO 05-03 22:31:41.695474.695474 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00033473968505859375 seconds
DEBUG 05-03 22:31:41.696769.696769 cuda_h.py:27] end *layer_moe_fused cost 23.383 ms
DEBUG 05-03 22:31:41.696950.696950 cuda_h.py:27] end prefill_layer cost 28.732 ms
DEBUG 05-03 22:31:41.696658.696658 lmp.py:765] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-03 22:31:41.696613.696613 lmp.py:729] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-03 22:31:41.703217.703217 cuda_h.py:27] end *sagl cost 6.630 ms
experts_cpu_alloc {'expert_ids': [11, 51, 67, 71, 79, 111, 3, 23, 107, 43, 99, 115, 55, 7, 119, 4, 20, 36, 60, 64, 12, 32, 40, 68, 44, 76, 21, 45, 69, 105, 81, 85, 25, 109, 73, 82, 106, 102, 110, 2, 6, 114], 'token_total': 152, 'token_per_expert': {11: 1, 51: 1, 67: 1, 71: 1, 79: 1, 111: 1, 3: 2, 23: 2, 107: 2, 43: 3, 99: 4, 115: 4, 55: 5, 7: 16, 119: 17, 4: 1, 20: 1, 36: 1, 60: 1, 64: 1, 12: 2, 32: 2, 40: 4, 68: 5, 44: 11, 76: 14, 21: 1, 45: 1, 69: 1, 105: 1, 81: 2, 85: 2, 25: 3, 109: 3, 73: 13, 82: 1, 106: 1, 102: 3, 110: 3, 2: 4, 6: 4, 114: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 19, 39, 59, 63, 75, 83, 91, 95, 103, 123], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 925, 'token_per_expert': {15: 27, 19: 47, 39: 103, 59: 26, 63: 21, 75: 84, 83: 418, 91: 57, 95: 32, 103: 57, 123: 53}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 24, 48, 52, 80, 96, 100, 108, 112, 120], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1216, 'token_per_expert': {0: 15, 8: 112, 24: 267, 48: 191, 52: 42, 80: 24, 96: 160, 100: 40, 108: 15, 112: 233, 120: 117}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 29, 33, 49, 57, 65, 77, 89, 97, 125], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1378, 'token_per_expert': {9: 216, 29: 14, 33: 288, 49: 32, 57: 124, 65: 125, 77: 22, 89: 48, 97: 251, 125: 258}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 18, 26, 38, 42, 46, 50, 70, 98, 126], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 425, 'token_per_expert': {10: 15, 18: 7, 26: 207, 38: 15, 42: 6, 46: 9, 50: 16, 70: 28, 98: 58, 126: 64}}
INFO 05-03 22:31:41.704003.704003 lmp.py:1059] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.516ms | allocate_experts_across_cpu_gpu: 0.331ms
INFO 05-03 22:31:41.704994.704994 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.173683166503906e-05 seconds
INFO 05-03 22:31:41.705850.705850 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006668567657470703 seconds
INFO 05-03 22:31:41.705300.705300 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005135536193847656 seconds
INFO 05-03 22:31:41.721982.721982 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01531076431274414 seconds
INFO 05-03 22:31:41.722148.722148 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008802413940429688 seconds
INFO 05-03 22:31:41.725029.725029 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.129ms act=0.146ms bmm2=0.032ms unpad=2.256ms total=2.563ms E=32 maxT=207 S=446 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.725605.725605 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.176ms act=0.188ms bmm2=0.046ms unpad=2.488ms total=2.899ms E=32 maxT=267 S=1259 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.725778.725778 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.143ms act=0.180ms bmm2=0.098ms unpad=2.615ms total=3.036ms E=32 maxT=288 S=1405 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.726648.726648 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.210ms act=0.198ms bmm2=0.633ms unpad=3.008ms total=4.048ms E=32 maxT=418 S=986 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.726591.726591 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004464626312255859 seconds
INFO 05-03 22:31:41.726497.726497 lmp.py:1215] [layer_moe_fused] to time: 5.030632019042969e-05 seconds
INFO 05-03 22:31:41.727364.727364 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.000244140625 seconds
DEBUG 05-03 22:31:41.727474.727474 cuda_h.py:27] end *layer_moe_fused cost 24.021 ms
DEBUG 05-03 22:31:41.727794.727794 cuda_h.py:27] end prefill_layer cost 30.926 ms
DEBUG 05-03 22:31:41.727205.727205 lmp.py:765] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-03 22:31:41.727852.727852 lmp.py:729] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-03 22:31:41.733523.733523 cuda_h.py:27] end *sagl cost 5.447 ms
experts_cpu_alloc {'expert_ids': [23, 35, 63, 103, 27, 87, 123, 32, 64, 104, 28, 72, 4, 92, 112, 120, 1, 21, 25, 49, 61, 69, 97, 105, 33, 45, 117, 109, 29, 65, 10, 102, 6, 26, 34, 38, 46, 70, 106, 126, 30, 14, 50, 118], 'token_total': 121, 'token_per_expert': {23: 1, 35: 1, 63: 1, 103: 1, 27: 2, 87: 2, 123: 2, 32: 1, 64: 1, 104: 1, 28: 2, 72: 2, 4: 4, 92: 4, 112: 6, 120: 6, 1: 1, 21: 1, 25: 1, 49: 1, 61: 1, 69: 1, 97: 2, 105: 2, 33: 3, 45: 3, 117: 4, 109: 5, 29: 6, 65: 7, 10: 1, 102: 1, 6: 2, 26: 2, 34: 2, 38: 2, 46: 2, 70: 2, 106: 3, 126: 3, 30: 5, 14: 6, 50: 7, 118: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 51, 55, 59, 67, 75, 107, 115, 127], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 934, 'token_per_expert': {3: 5, 7: 401, 19: 165, 51: 71, 55: 8, 59: 159, 67: 24, 75: 3, 107: 30, 115: 13, 127: 55}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 20, 36, 52, 76, 84, 88, 96, 100, 116], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1014, 'token_per_expert': {8: 103, 12: 22, 20: 9, 36: 48, 52: 8, 76: 432, 84: 16, 88: 117, 96: 129, 100: 51, 116: 79}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 13, 37, 53, 57, 73, 81, 89, 101, 113, 125], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 842, 'token_per_expert': {9: 33, 13: 48, 37: 35, 53: 24, 57: 379, 73: 211, 81: 29, 89: 28, 101: 7, 113: 7, 125: 41}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 18, 22, 54, 66, 78, 82, 86, 90, 94, 114], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1185, 'token_per_expert': {2: 15, 18: 25, 22: 15, 54: 352, 66: 18, 78: 57, 82: 292, 86: 295, 90: 66, 94: 31, 114: 19}}
INFO 05-03 22:31:41.734587.734587 lmp.py:1059] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.496ms | allocate_experts_across_cpu_gpu: 0.343ms
INFO 05-03 22:31:41.734863.734863 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.14984130859375e-05 seconds
INFO 05-03 22:31:41.735560.735560 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006642341613769531 seconds
INFO 05-03 22:31:41.735215.735215 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005245208740234375 seconds
INFO 05-03 22:31:41.748221.748221 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012522220611572266 seconds
INFO 05-03 22:31:41.749434.749434 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008790493011474609 seconds
INFO 05-03 22:31:41.753716.753716 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.178ms act=0.155ms bmm2=0.321ms unpad=2.747ms total=3.401ms E=32 maxT=432 S=1041 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.753118.753118 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.207ms act=0.246ms bmm2=0.045ms unpad=3.297ms total=3.795ms E=32 maxT=401 S=944 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.753485.753485 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.155ms act=0.150ms bmm2=0.059ms unpad=3.350ms total=3.714ms E=32 maxT=352 S=1231 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.753590.753590 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.146ms act=0.205ms bmm2=0.347ms unpad=3.432ms total=4.130ms E=32 maxT=379 S=880 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.754198.754198 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004947185516357422 seconds
INFO 05-03 22:31:41.754407.754407 lmp.py:1215] [layer_moe_fused] to time: 4.9114227294921875e-05 seconds
INFO 05-03 22:31:41.754427.754427 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002498626708984375 seconds
DEBUG 05-03 22:31:41.755497.755497 cuda_h.py:27] end *layer_moe_fused cost 21.703 ms
DEBUG 05-03 22:31:41.755771.755771 cuda_h.py:27] end prefill_layer cost 27.432 ms
DEBUG 05-03 22:31:41.755081.755081 lmp.py:765] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-03 22:31:41.755530.755530 lmp.py:729] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-03 22:31:41.761995.761995 cuda_h.py:27] end *sagl cost 5.916 ms
experts_cpu_alloc {'expert_ids': [3, 7, 11, 15, 23, 47, 63, 71, 115, 35, 20, 32, 36, 80, 100, 108, 116, 48, 120, 84, 9, 13, 33, 109, 1, 5, 113, 49, 57, 53, 105, 61, 22, 34, 90, 98, 102, 18, 50, 58, 2, 126, 38, 82], 'token_total': 114, 'token_per_expert': {3: 1, 7: 1, 11: 1, 15: 1, 23: 1, 47: 1, 63: 1, 71: 1, 115: 1, 35: 2, 20: 1, 32: 1, 36: 1, 80: 1, 100: 1, 108: 1, 116: 1, 48: 2, 120: 2, 84: 4, 9: 1, 13: 1, 33: 1, 109: 1, 1: 2, 5: 2, 113: 2, 49: 4, 57: 4, 53: 6, 105: 6, 61: 9, 22: 1, 34: 1, 90: 1, 98: 1, 102: 1, 18: 4, 50: 4, 58: 4, 2: 5, 126: 6, 38: 10, 82: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 27, 31, 39, 43, 55, 83, 99, 103, 111, 127], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 2018, 'token_per_expert': {19: 370, 27: 9, 31: 490, 39: 5, 43: 211, 55: 27, 83: 62, 99: 2, 103: 308, 111: 225, 127: 309}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 40, 56, 60, 64, 68, 92, 96, 112], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 380, 'token_per_expert': {4: 24, 8: 8, 12: 9, 40: 5, 56: 110, 60: 5, 64: 12, 68: 29, 92: 4, 96: 124, 112: 50}}
experts_gpu_alloc_device_2 {'expert_ids': [17, 25, 29, 65, 69, 77, 81, 85, 101, 117, 121], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1057, 'token_per_expert': {17: 107, 25: 17, 29: 103, 65: 129, 69: 179, 77: 19, 81: 9, 85: 236, 101: 165, 117: 63, 121: 30}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 30, 46, 62, 66, 70, 74, 86, 114, 118], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 527, 'token_per_expert': {6: 46, 10: 109, 30: 22, 46: 31, 62: 43, 66: 34, 70: 162, 74: 28, 86: 26, 114: 12, 118: 14}}
INFO 05-03 22:31:41.762270.762270 lmp.py:1059] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.481ms | allocate_experts_across_cpu_gpu: 0.343ms
INFO 05-03 22:31:41.762831.762831 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.173683166503906e-05 seconds
INFO 05-03 22:31:41.777655.777655 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.014833927154541016 seconds
INFO 05-03 22:31:41.778846.778846 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004966259002685547 seconds
INFO 05-03 22:31:41.786724.786724 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0081024169921875 seconds
INFO 05-03 22:31:41.787402.787402 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009400844573974609 seconds
INFO 05-03 22:31:41.790866.790866 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.178ms act=0.226ms bmm2=0.059ms unpad=2.225ms total=2.688ms E=32 maxT=124 S=395 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.791426.791426 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.114ms act=0.218ms bmm2=0.077ms unpad=2.679ms total=3.088ms E=32 maxT=162 S=576 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.791411.791411 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.145ms act=0.186ms bmm2=0.100ms unpad=2.846ms total=3.276ms E=32 maxT=236 S=1096 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.791313.791313 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.212ms act=0.265ms bmm2=0.298ms unpad=3.082ms total=3.857ms E=32 maxT=490 S=2029 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.792711.792711 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004399299621582031 seconds
INFO 05-03 22:31:41.792350.792350 lmp.py:1215] [layer_moe_fused] to time: 4.887580871582031e-05 seconds
INFO 05-03 22:31:41.792390.792390 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002651214599609375 seconds
DEBUG 05-03 22:31:41.792283.792283 cuda_h.py:27] end *layer_moe_fused cost 31.068 ms
DEBUG 05-03 22:31:41.792510.792510 cuda_h.py:27] end prefill_layer cost 37.379 ms
DEBUG 05-03 22:31:41.793153.793153 lmp.py:765] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-03 22:31:41.793881.793881 lmp.py:729] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-03 22:31:41.796349.796349 cuda_h.py:27] end *sagl cost 2.920 ms
experts_cpu_alloc {'expert_ids': [63, 107, 111, 119, 23, 47, 95, 27, 32, 40, 60, 68, 76, 84, 116, 16, 52, 112, 44, 80, 65, 89, 93, 97, 113, 17, 21, 53, 117, 9, 33, 77, 85, 57, 73, 37, 10, 22, 38, 50, 98, 102, 106, 58, 78, 126, 54, 118], 'token_total': 160, 'token_per_expert': {63: 1, 107: 1, 111: 1, 119: 1, 23: 2, 47: 2, 95: 2, 27: 5, 32: 1, 40: 1, 60: 1, 68: 1, 76: 1, 84: 1, 116: 1, 16: 2, 52: 2, 112: 3, 44: 4, 80: 6, 65: 1, 89: 1, 93: 1, 97: 1, 113: 1, 17: 2, 21: 2, 53: 4, 117: 5, 9: 6, 33: 7, 77: 9, 85: 10, 57: 11, 73: 18, 37: 19, 10: 1, 22: 1, 38: 1, 50: 1, 98: 1, 102: 1, 106: 1, 58: 2, 78: 3, 126: 3, 54: 4, 118: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 51, 59, 67, 75, 79, 83, 91, 99, 103], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 526, 'token_per_expert': {3: 17, 7: 36, 31: 26, 35: 22, 51: 188, 59: 93, 67: 9, 75: 38, 79: 24, 83: 44, 91: 14, 99: 10, 103: 5}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 24, 36, 48, 72, 92, 96, 100, 104], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1391, 'token_per_expert': {0: 52, 4: 303, 8: 14, 12: 103, 24: 10, 36: 183, 48: 34, 72: 388, 92: 153, 96: 30, 100: 10, 104: 111}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 45, 49, 61, 69, 81, 105, 121, 125], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1319, 'token_per_expert': {1: 282, 5: 404, 13: 67, 25: 76, 45: 39, 49: 54, 61: 53, 69: 117, 81: 23, 105: 31, 121: 30, 125: 143}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 34, 62, 66, 70, 74, 82, 90, 94, 114], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 700, 'token_per_expert': {2: 178, 6: 5, 26: 8, 34: 19, 62: 31, 66: 18, 70: 7, 74: 144, 82: 137, 90: 91, 94: 19, 114: 43}}
INFO 05-03 22:31:41.797380.797380 lmp.py:1059] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.482ms | allocate_experts_across_cpu_gpu: 0.370ms
INFO 05-03 22:31:41.797663.797663 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.745887756347656e-05 seconds
INFO 05-03 22:31:41.798808.798808 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005788803100585938 seconds
INFO 05-03 22:31:41.798045.798045 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004966259002685547 seconds
INFO 05-03 22:31:41.808319.808319 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009253978729248047 seconds
INFO 05-03 22:31:41.809639.809639 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009162425994873047 seconds
INFO 05-03 22:31:41.811009.811009 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.204ms act=0.224ms bmm2=0.046ms unpad=1.642ms total=2.116ms E=32 maxT=188 S=541 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.812562.812562 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.075ms act=0.186ms bmm2=0.029ms unpad=2.340ms total=2.630ms E=32 maxT=178 S=723 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.813071.813071 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.234ms act=0.152ms bmm2=0.059ms unpad=3.061ms total=3.506ms E=32 maxT=388 S=1415 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.813873.813873 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.104ms act=0.173ms bmm2=0.262ms unpad=3.144ms total=3.683ms E=32 maxT=404 S=1417 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.813134.813134 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004427194595336914 seconds
INFO 05-03 22:31:41.813774.813774 lmp.py:1215] [layer_moe_fused] to time: 5.0067901611328125e-05 seconds
INFO 05-03 22:31:41.814888.814888 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00028443336486816406 seconds
DEBUG 05-03 22:31:41.814370.814370 cuda_h.py:27] end *layer_moe_fused cost 17.947 ms
DEBUG 05-03 22:31:41.814835.814835 cuda_h.py:27] end prefill_layer cost 21.173 ms
DEBUG 05-03 22:31:41.814358.814358 lmp.py:765] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-03 22:31:41.814342.814342 lmp.py:729] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-03 22:31:41.817493.817493 cuda_h.py:27] end *sagl cost 2.374 ms
experts_cpu_alloc {'expert_ids': [15, 39, 51, 91, 103, 119, 35, 71, 107, 55, 127, 11, 27, 16, 60, 64, 120, 124, 56, 48, 108, 40, 52, 72, 80, 68, 8, 104, 84, 13, 85, 25, 105, 1, 9, 38, 62, 102, 54, 46, 86, 30, 78, 10, 66, 118, 74, 126], 'token_total': 226, 'token_per_expert': {15: 1, 39: 1, 51: 1, 91: 1, 103: 1, 119: 1, 35: 2, 71: 2, 107: 2, 55: 6, 127: 8, 11: 15, 27: 17, 16: 1, 60: 1, 64: 1, 120: 1, 124: 1, 56: 2, 48: 3, 108: 3, 40: 5, 52: 5, 72: 6, 80: 6, 68: 9, 8: 12, 104: 12, 84: 13, 13: 1, 85: 2, 25: 3, 105: 5, 1: 6, 9: 6, 38: 1, 62: 1, 102: 1, 54: 2, 46: 3, 86: 4, 30: 5, 78: 5, 10: 6, 66: 7, 118: 8, 74: 9, 126: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 23, 43, 47, 59, 67, 75, 79, 87, 95, 99, 115, 123], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1338, 'token_per_expert': {7: 149, 23: 57, 43: 90, 47: 34, 59: 23, 67: 29, 75: 46, 79: 24, 87: 17, 95: 288, 99: 426, 115: 52, 123: 103}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 28, 32, 44, 76, 92, 96, 100, 112], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 684, 'token_per_expert': {0: 25, 4: 19, 20: 34, 24: 85, 28: 67, 32: 63, 44: 41, 76: 185, 92: 67, 96: 29, 100: 54, 112: 15}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 17, 29, 37, 41, 57, 61, 65, 73, 77, 89, 125], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 661, 'token_per_expert': {5: 12, 17: 49, 29: 7, 37: 8, 41: 154, 57: 308, 61: 14, 65: 9, 73: 18, 77: 22, 89: 47, 125: 13}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 18, 22, 34, 42, 50, 82, 94, 98, 106, 110, 114], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1187, 'token_per_expert': {2: 14, 18: 148, 22: 78, 34: 73, 42: 47, 50: 78, 82: 376, 94: 36, 98: 77, 106: 41, 110: 50, 114: 169}}
INFO 05-03 22:31:41.818578.818578 lmp.py:1059] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.489ms | allocate_experts_across_cpu_gpu: 0.368ms
INFO 05-03 22:31:41.818245.818245 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.7697296142578125e-05 seconds
INFO 05-03 22:31:41.819101.819101 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005629062652587891 seconds
INFO 05-03 22:31:41.819843.819843 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005536079406738281 seconds
INFO 05-03 22:31:41.828126.828126 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008507966995239258 seconds
INFO 05-03 22:31:41.829843.829843 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00092315673828125 seconds
INFO 05-03 22:31:41.831531.831531 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.170ms act=0.186ms bmm2=0.030ms unpad=1.653ms total=2.039ms E=32 maxT=185 S=765 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.832089.832089 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.135ms act=0.211ms bmm2=0.064ms unpad=2.610ms total=3.021ms E=32 maxT=308 S=684 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.833568.833568 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.204ms act=0.236ms bmm2=0.062ms unpad=3.106ms total=3.608ms E=32 maxT=426 S=1396 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.833313.833313 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.107ms act=0.227ms bmm2=0.062ms unpad=3.125ms total=3.521ms E=32 maxT=376 S=1251 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.834789.834789 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004443645477294922 seconds
INFO 05-03 22:31:41.834621.834621 lmp.py:1215] [layer_moe_fused] to time: 5.054473876953125e-05 seconds
INFO 05-03 22:31:41.834912.834912 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002410411834716797 seconds
DEBUG 05-03 22:31:41.834917.834917 cuda_h.py:27] end *layer_moe_fused cost 17.332 ms
DEBUG 05-03 22:31:41.834429.834429 cuda_h.py:27] end prefill_layer cost 20.007 ms
DEBUG 05-03 22:31:41.835860.835860 lmp.py:765] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-03 22:31:41.835939.835939 lmp.py:729] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-03 22:31:41.837539.837539 cuda_h.py:27] end *sagl cost 2.349 ms
experts_cpu_alloc {'expert_ids': [27, 47, 51, 55, 115, 15, 19, 7, 111, 103, 28, 80, 120, 20, 44, 36, 100, 8, 48, 68, 72, 104, 16, 5, 41, 9, 69, 37, 45, 77, 125, 89, 17, 30, 62, 86, 90, 38, 74, 110, 114, 126, 22, 46, 54, 70], 'token_total': 270, 'token_per_expert': {27: 1, 47: 1, 51: 1, 55: 1, 115: 1, 15: 2, 19: 3, 7: 8, 111: 8, 103: 12, 28: 1, 80: 1, 120: 1, 20: 2, 44: 2, 36: 3, 100: 3, 8: 6, 48: 6, 68: 8, 72: 9, 104: 9, 16: 10, 5: 2, 41: 3, 9: 7, 69: 7, 37: 10, 45: 11, 77: 11, 125: 11, 89: 23, 17: 30, 30: 1, 62: 1, 86: 1, 90: 1, 38: 2, 74: 2, 110: 2, 114: 2, 126: 6, 22: 7, 46: 9, 54: 10, 70: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 23, 43, 59, 63, 71, 75, 83, 91, 119, 123, 127], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1035, 'token_per_expert': {11: 96, 23: 27, 43: 48, 59: 90, 63: 51, 71: 262, 75: 40, 83: 297, 91: 30, 119: 18, 123: 39, 127: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 32, 40, 52, 56, 64, 76, 92, 112, 124], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 526, 'token_per_expert': {0: 103, 4: 20, 12: 13, 32: 43, 40: 15, 52: 20, 56: 155, 64: 14, 76: 69, 92: 17, 112: 18, 124: 39}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 21, 33, 49, 57, 61, 73, 81, 85, 93, 97, 121], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1356, 'token_per_expert': {13: 57, 21: 188, 33: 105, 49: 93, 57: 151, 61: 37, 73: 150, 81: 30, 85: 30, 93: 248, 97: 74, 121: 193}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 26, 58, 82, 98, 102, 106, 118, 122], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 909, 'token_per_expert': {2: 52, 6: 98, 18: 45, 26: 32, 58: 67, 82: 22, 98: 105, 102: 61, 106: 116, 118: 291, 122: 20}}
INFO 05-03 22:31:41.838054.838054 lmp.py:1059] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.498ms | allocate_experts_across_cpu_gpu: 0.360ms
INFO 05-03 22:31:41.838569.838569 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.1975250244140625e-05 seconds
INFO 05-03 22:31:41.839208.839208 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005841255187988281 seconds
INFO 05-03 22:31:41.840354.840354 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005333423614501953 seconds
INFO 05-03 22:31:41.848947.848947 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008484125137329102 seconds
INFO 05-03 22:31:41.849876.849876 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009107589721679688 seconds
INFO 05-03 22:31:41.852157.852157 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.191ms act=0.246ms bmm2=0.052ms unpad=2.277ms total=2.765ms E=32 maxT=155 S=587 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.853537.853537 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.200ms act=0.204ms bmm2=0.045ms unpad=2.869ms total=3.318ms E=32 maxT=297 S=1073 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.853358.853358 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.212ms act=0.146ms bmm2=0.044ms unpad=2.928ms total=3.330ms E=32 maxT=291 S=965 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.853459.853459 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.110ms act=0.145ms bmm2=0.133ms unpad=3.248ms total=3.636ms E=32 maxT=248 S=1471 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.854191.854191 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004403829574584961 seconds
INFO 05-03 22:31:41.854977.854977 lmp.py:1215] [layer_moe_fused] to time: 5.030632019042969e-05 seconds
INFO 05-03 22:31:41.854045.854045 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0003228187561035156 seconds
DEBUG 05-03 22:31:41.855774.855774 cuda_h.py:27] end *layer_moe_fused cost 17.402 ms
DEBUG 05-03 22:31:41.855763.855763 cuda_h.py:27] end prefill_layer cost 20.059 ms
DEBUG 05-03 22:31:41.855292.855292 lmp.py:765] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-03 22:31:41.855881.855881 lmp.py:729] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-03 22:31:41.858423.858423 cuda_h.py:27] end *sagl cost 2.349 ms
experts_cpu_alloc {'expert_ids': [27, 91, 95, 107, 75, 119, 67, 3, 15, 31, 35, 36, 80, 124, 72, 104, 100, 116, 48, 52, 84, 88, 32, 40, 60, 4, 41, 53, 69, 81, 109, 13, 61, 85, 65, 73, 5, 26, 38, 2, 22, 30, 42, 50, 78, 110, 70, 94, 122], 'token_total': 198, 'token_per_expert': {27: 1, 91: 1, 95: 1, 107: 1, 75: 2, 119: 4, 67: 6, 3: 7, 15: 8, 31: 8, 35: 8, 36: 1, 80: 1, 124: 1, 72: 2, 104: 2, 100: 3, 116: 3, 48: 7, 52: 7, 84: 9, 88: 9, 32: 10, 40: 11, 60: 15, 4: 17, 41: 1, 53: 1, 69: 1, 81: 1, 109: 1, 13: 2, 61: 2, 85: 2, 65: 3, 73: 3, 5: 4, 26: 1, 38: 1, 2: 2, 22: 2, 30: 2, 42: 3, 50: 3, 78: 3, 110: 3, 70: 4, 94: 4, 122: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 19, 23, 55, 63, 71, 79, 87, 99, 111, 123, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 622, 'token_per_expert': {7: 11, 11: 124, 19: 27, 23: 21, 55: 10, 63: 57, 71: 126, 79: 9, 87: 97, 99: 30, 111: 14, 123: 43, 127: 53}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 16, 20, 24, 28, 44, 56, 68, 96, 108, 112, 120], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1254, 'token_per_expert': {8: 25, 12: 79, 16: 26, 20: 19, 24: 77, 28: 46, 44: 127, 56: 446, 68: 76, 96: 30, 108: 193, 112: 17, 120: 93}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 33, 37, 45, 57, 89, 93, 101, 105, 113, 117, 125], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 994, 'token_per_expert': {9: 176, 33: 6, 37: 17, 45: 67, 57: 9, 89: 131, 93: 6, 101: 34, 105: 12, 113: 143, 117: 101, 125: 292}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 18, 46, 54, 58, 62, 66, 90, 98, 102, 106], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1028, 'token_per_expert': {6: 8, 10: 53, 18: 407, 46: 87, 54: 137, 58: 49, 62: 91, 66: 39, 90: 8, 98: 46, 102: 97, 106: 6}}
INFO 05-03 22:31:41.859614.859614 lmp.py:1059] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.489ms | allocate_experts_across_cpu_gpu: 0.375ms
INFO 05-03 22:31:41.859089.859089 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-03 22:31:41.860916.860916 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006191730499267578 seconds
INFO 05-03 22:31:41.860141.860141 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.000522613525390625 seconds
INFO 05-03 22:31:41.870826.870826 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00943756103515625 seconds
INFO 05-03 22:31:41.871576.871576 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009124279022216797 seconds
INFO 05-03 22:31:41.872437.872437 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.209ms act=0.175ms bmm2=0.044ms unpad=0.922ms total=1.351ms E=32 maxT=126 S=669 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.874890.874890 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.142ms act=0.214ms bmm2=0.049ms unpad=2.497ms total=2.903ms E=32 maxT=292 S=1015 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.875238.875238 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.147ms act=0.119ms bmm2=0.043ms unpad=3.033ms total=3.341ms E=32 maxT=407 S=1060 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.875972.875972 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.172ms act=0.160ms bmm2=0.273ms unpad=3.222ms total=3.827ms E=32 maxT=446 S=1352 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.875485.875485 lmp.py:1204] [layer_moe_fused] experts compute time: 0.00452113151550293 seconds
INFO 05-03 22:31:41.875463.875463 lmp.py:1215] [layer_moe_fused] to time: 5.14984130859375e-05 seconds
INFO 05-03 22:31:41.876098.876098 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00025010108947753906 seconds
DEBUG 05-03 22:31:41.876959.876959 cuda_h.py:27] end *layer_moe_fused cost 18.290 ms
DEBUG 05-03 22:31:41.876994.876994 cuda_h.py:27] end prefill_layer cost 20.974 ms
DEBUG 05-03 22:31:41.876317.876317 lmp.py:765] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-03 22:31:41.876825.876825 lmp.py:729] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-03 22:31:41.879234.879234 cuda_h.py:27] end *sagl cost 2.346 ms
experts_cpu_alloc {'expert_ids': [43, 67, 71, 107, 11, 31, 39, 95, 23, 87, 91, 111, 0, 76, 80, 84, 116, 4, 112, 40, 88, 92, 100, 108, 28, 9, 49, 93, 101, 37, 77, 85, 89, 109, 45, 25, 6, 38, 58, 94, 102, 10, 62, 78, 98, 18, 66, 114, 122], 'token_total': 155, 'token_per_expert': {43: 1, 67: 1, 71: 1, 107: 1, 11: 3, 31: 3, 39: 4, 95: 5, 23: 6, 87: 7, 91: 7, 111: 10, 0: 1, 76: 1, 80: 1, 84: 1, 116: 1, 4: 2, 112: 2, 40: 3, 88: 3, 92: 3, 100: 3, 108: 4, 28: 5, 9: 1, 49: 1, 93: 1, 101: 1, 37: 2, 77: 2, 85: 2, 89: 3, 109: 3, 45: 4, 25: 5, 6: 1, 38: 1, 58: 1, 94: 1, 102: 1, 10: 2, 62: 2, 78: 2, 98: 3, 18: 4, 66: 6, 114: 12, 122: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 19, 27, 35, 47, 51, 59, 63, 83, 99, 115, 119, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1533, 'token_per_expert': {3: 281, 19: 265, 27: 64, 35: 101, 47: 186, 51: 14, 59: 26, 63: 11, 83: 29, 99: 12, 115: 18, 119: 303, 127: 223}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 16, 24, 32, 36, 44, 48, 60, 68, 72, 104], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 465, 'token_per_expert': {8: 10, 12: 15, 16: 18, 24: 32, 32: 46, 36: 65, 44: 12, 48: 16, 60: 118, 68: 13, 72: 50, 104: 70}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 13, 17, 21, 33, 41, 53, 61, 69, 97, 113, 117], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 966, 'token_per_expert': {5: 89, 13: 229, 17: 48, 21: 468, 33: 7, 41: 16, 53: 10, 61: 24, 69: 42, 97: 20, 113: 7, 117: 6}}
experts_gpu_alloc_device_3 {'expert_ids': [14, 22, 26, 30, 34, 46, 54, 70, 74, 82, 86, 106], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 977, 'token_per_expert': {14: 43, 22: 43, 26: 38, 30: 16, 34: 32, 46: 49, 54: 27, 70: 262, 74: 22, 82: 36, 86: 256, 106: 153}}
INFO 05-03 22:31:41.880412.880412 lmp.py:1059] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.518ms | allocate_experts_across_cpu_gpu: 0.372ms
INFO 05-03 22:31:41.880734.880734 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.4836273193359375e-05 seconds
INFO 05-03 22:31:41.881312.881312 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006449222564697266 seconds
INFO 05-03 22:31:41.881987.881987 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005035400390625 seconds
INFO 05-03 22:31:41.891980.891980 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008835315704345703 seconds
INFO 05-03 22:31:41.892876.892876 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009160041809082031 seconds
INFO 05-03 22:31:41.894581.894581 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.178ms act=0.137ms bmm2=0.104ms unpad=1.565ms total=1.984ms E=32 maxT=118 S=495 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.895776.895776 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.209ms act=0.252ms bmm2=0.046ms unpad=2.590ms total=3.097ms E=32 maxT=303 S=1582 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.895861.895861 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.131ms act=0.153ms bmm2=0.076ms unpad=2.941ms total=3.302ms E=32 maxT=262 S=1028 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.896905.896905 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.272ms act=0.124ms bmm2=0.267ms unpad=3.004ms total=3.666ms E=32 maxT=468 S=991 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.896740.896740 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004348039627075195 seconds
INFO 05-03 22:31:41.896426.896426 lmp.py:1215] [layer_moe_fused] to time: 5.078315734863281e-05 seconds
INFO 05-03 22:31:41.896831.896831 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00028896331787109375 seconds
DEBUG 05-03 22:31:41.897612.897612 cuda_h.py:27] end *layer_moe_fused cost 17.754 ms
DEBUG 05-03 22:31:41.897455.897455 cuda_h.py:27] end prefill_layer cost 20.408 ms
DEBUG 05-03 22:31:41.897865.897865 lmp.py:765] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-03 22:31:41.897286.897286 lmp.py:729] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-03 22:31:41.900497.900497 cuda_h.py:27] end *sagl cost 2.379 ms
experts_cpu_alloc {'expert_ids': [51, 107, 123, 3, 87, 111, 11, 67, 23, 83, 127, 99, 24, 60, 64, 84, 100, 104, 124, 112, 76, 48, 88, 68, 17, 49, 101, 125, 5, 53, 21, 41, 29, 89, 61, 65, 33, 77, 10, 30, 54, 86, 118, 126, 102, 14, 90, 2, 62, 46], 'token_total': 211, 'token_per_expert': {51: 1, 107: 1, 123: 1, 3: 2, 87: 2, 111: 2, 11: 3, 67: 4, 23: 6, 83: 6, 127: 7, 99: 10, 24: 1, 60: 1, 64: 1, 84: 1, 100: 1, 104: 1, 124: 1, 112: 2, 76: 3, 48: 4, 88: 7, 68: 8, 17: 1, 49: 1, 101: 2, 125: 2, 5: 3, 53: 3, 21: 4, 41: 6, 29: 7, 89: 8, 61: 11, 65: 12, 33: 16, 77: 25, 10: 1, 30: 1, 54: 1, 86: 1, 118: 1, 126: 1, 102: 2, 14: 3, 90: 3, 2: 5, 62: 7, 46: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 27, 31, 35, 47, 55, 63, 71, 79, 95, 103, 115, 119], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1360, 'token_per_expert': {7: 203, 27: 35, 31: 85, 35: 97, 47: 81, 55: 131, 63: 16, 71: 379, 79: 62, 95: 26, 103: 122, 115: 30, 119: 93}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 28, 36, 44, 52, 56, 108, 116, 120], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 745, 'token_per_expert': {0: 17, 4: 14, 8: 147, 12: 152, 16: 84, 28: 22, 36: 157, 44: 20, 52: 15, 56: 13, 108: 25, 116: 27, 120: 52}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 25, 37, 57, 69, 81, 85, 97, 105, 113], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 977, 'token_per_expert': {1: 30, 9: 26, 13: 100, 25: 224, 37: 102, 57: 30, 69: 75, 81: 27, 85: 72, 97: 124, 105: 142, 113: 25}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 18, 22, 38, 42, 50, 66, 70, 78, 98, 106, 114], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 803, 'token_per_expert': {6: 16, 18: 88, 22: 254, 38: 28, 42: 113, 50: 25, 66: 8, 70: 62, 78: 85, 98: 38, 106: 55, 114: 31}}
INFO 05-03 22:31:41.901695.901695 lmp.py:1059] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.491ms | allocate_experts_across_cpu_gpu: 0.379ms
INFO 05-03 22:31:41.901448.901448 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.8650970458984375e-05 seconds
INFO 05-03 22:31:41.902734.902734 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006670951843261719 seconds
INFO 05-03 22:31:41.902601.902601 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005056858062744141 seconds
INFO 05-03 22:31:41.911728.911728 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008605718612670898 seconds
INFO 05-03 22:31:41.912637.912637 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009267330169677734 seconds
INFO 05-03 22:31:41.915294.915294 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.191ms act=0.237ms bmm2=0.051ms unpad=2.421ms total=2.900ms E=32 maxT=157 S=776 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.916033.916033 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.206ms act=0.173ms bmm2=0.044ms unpad=3.266ms total=3.689ms E=32 maxT=379 S=1405 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.916714.916714 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.108ms act=0.238ms bmm2=0.044ms unpad=3.327ms total=3.717ms E=32 maxT=224 S=1078 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.916442.916442 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.180ms act=0.115ms bmm2=0.057ms unpad=3.492ms total=3.843ms E=32 maxT=254 S=837 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.917096.917096 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004720926284790039 seconds
INFO 05-03 22:31:41.917451.917451 lmp.py:1215] [layer_moe_fused] to time: 5.054473876953125e-05 seconds
INFO 05-03 22:31:41.917241.917241 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0003275871276855469 seconds
DEBUG 05-03 22:31:41.918400.918400 cuda_h.py:27] end *layer_moe_fused cost 17.791 ms
DEBUG 05-03 22:31:41.918866.918866 cuda_h.py:27] end prefill_layer cost 20.476 ms
DEBUG 05-03 22:31:41.918552.918552 lmp.py:765] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-03 22:31:41.918045.918045 lmp.py:729] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-03 22:31:41.921350.921350 cuda_h.py:27] end *sagl cost 3.443 ms
experts_cpu_alloc {'expert_ids': [11, 19, 59, 71, 75, 79, 95, 127, 63, 103, 111, 67, 7, 23, 36, 120, 104, 108, 4, 16, 112, 28, 24, 44, 100, 116, 5, 17, 45, 49, 61, 93, 125, 73, 57, 113, 69, 9, 86, 98, 118, 26, 54, 106, 122, 90, 14, 30, 82, 110, 6, 46, 114, 18, 74], 'token_total': 206, 'token_per_expert': {11: 1, 19: 1, 59: 1, 71: 1, 75: 1, 79: 1, 95: 1, 127: 1, 63: 2, 103: 2, 111: 2, 67: 3, 7: 6, 23: 6, 36: 1, 120: 1, 104: 2, 108: 2, 4: 3, 16: 3, 112: 3, 28: 5, 24: 6, 44: 7, 100: 7, 116: 7, 5: 1, 17: 1, 45: 1, 49: 1, 61: 1, 93: 1, 125: 1, 73: 2, 57: 3, 113: 3, 69: 4, 9: 7, 86: 1, 98: 1, 118: 1, 26: 2, 54: 2, 106: 2, 122: 2, 90: 3, 14: 6, 30: 6, 82: 6, 110: 6, 6: 8, 46: 9, 114: 9, 18: 11, 74: 29}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 27, 31, 35, 39, 47, 51, 55, 83, 87, 91, 99, 107, 115], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 968, 'token_per_expert': {15: 260, 27: 40, 31: 6, 35: 28, 39: 11, 47: 15, 51: 130, 55: 213, 83: 6, 87: 10, 91: 95, 99: 134, 107: 8, 115: 12}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 12, 20, 32, 40, 48, 52, 56, 60, 68, 72, 76, 88, 124], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1161, 'token_per_expert': {0: 50, 12: 36, 20: 235, 32: 14, 40: 131, 48: 14, 52: 163, 56: 34, 60: 17, 68: 111, 72: 10, 76: 275, 88: 54, 124: 17}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 21, 33, 37, 41, 53, 65, 77, 81, 85, 89, 101, 105, 121], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 477, 'token_per_expert': {13: 18, 21: 15, 33: 41, 37: 26, 41: 160, 53: 40, 65: 22, 77: 30, 81: 24, 85: 16, 89: 29, 101: 36, 105: 9, 121: 11}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 22, 34, 38, 50, 58, 62, 66, 70, 78, 94, 102, 126], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1284, 'token_per_expert': {2: 50, 10: 153, 22: 32, 34: 35, 38: 74, 50: 30, 58: 73, 62: 76, 66: 135, 70: 46, 78: 78, 94: 70, 102: 96, 126: 336}}
INFO 05-03 22:31:41.922542.922542 lmp.py:1059] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.490ms | allocate_experts_across_cpu_gpu: 0.409ms
INFO 05-03 22:31:41.922771.922771 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.793571472167969e-05 seconds
INFO 05-03 22:31:41.923477.923477 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006844997406005859 seconds
INFO 05-03 22:31:41.924755.924755 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005257129669189453 seconds
INFO 05-03 22:31:41.934718.934718 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009353876113891602 seconds
INFO 05-03 22:31:41.935821.935821 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00095367431640625 seconds
INFO 05-03 22:31:41.938219.938219 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.136ms act=0.245ms bmm2=0.051ms unpad=2.804ms total=3.236ms E=32 maxT=160 S=503 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.939652.939652 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.210ms act=0.177ms bmm2=0.044ms unpad=3.394ms total=3.825ms E=32 maxT=260 S=997 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.939822.939822 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.168ms act=0.180ms bmm2=0.105ms unpad=3.427ms total=3.881ms E=32 maxT=275 S=1208 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.940980.940980 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.151ms act=0.170ms bmm2=0.062ms unpad=3.957ms total=4.341ms E=32 maxT=336 S=1388 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.940586.940586 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005153656005859375 seconds
INFO 05-03 22:31:41.940371.940371 lmp.py:1215] [layer_moe_fused] to time: 5.078315734863281e-05 seconds
INFO 05-03 22:31:41.940558.940558 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00030541419982910156 seconds
DEBUG 05-03 22:31:41.941081.941081 cuda_h.py:27] end *layer_moe_fused cost 19.382 ms
DEBUG 05-03 22:31:41.941216.941216 cuda_h.py:27] end prefill_layer cost 23.088 ms
DEBUG 05-03 22:31:41.941257.941257 lmp.py:765] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-03 22:31:41.941553.941553 lmp.py:729] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-03 22:31:41.944373.944373 cuda_h.py:27] end *sagl cost 2.412 ms
experts_cpu_alloc {'expert_ids': [55, 75, 79, 91, 19, 71, 83, 3, 107, 39, 63, 15, 119, 67, 35, 31, 0, 20, 24, 44, 56, 68, 108, 4, 60, 124, 76, 48, 92, 104, 88, 33, 65, 97, 5, 81, 105, 109, 117, 29, 73, 113, 45, 14, 50, 102, 70, 74, 110, 114, 58, 78], 'token_total': 175, 'token_per_expert': {55: 1, 75: 1, 79: 1, 91: 1, 19: 2, 71: 2, 83: 2, 3: 3, 107: 3, 39: 4, 63: 5, 15: 6, 119: 6, 67: 7, 35: 8, 31: 14, 0: 1, 20: 1, 24: 1, 44: 1, 56: 1, 68: 1, 108: 1, 4: 2, 60: 2, 124: 3, 76: 5, 48: 6, 92: 6, 104: 7, 88: 9, 33: 1, 65: 1, 97: 1, 5: 2, 81: 2, 105: 2, 109: 2, 117: 2, 29: 4, 73: 4, 113: 8, 45: 14, 14: 1, 50: 1, 102: 1, 70: 2, 74: 2, 110: 2, 114: 2, 58: 3, 78: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 27, 43, 47, 51, 87, 95, 99, 103, 111, 115, 123, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1153, 'token_per_expert': {11: 36, 27: 50, 43: 18, 47: 17, 51: 64, 87: 41, 95: 155, 99: 352, 103: 146, 111: 89, 115: 30, 123: 118, 127: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 28, 36, 40, 64, 72, 80, 84, 96, 100, 112, 116, 120], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 869, 'token_per_expert': {8: 14, 28: 98, 36: 80, 40: 117, 64: 61, 72: 23, 80: 46, 84: 26, 96: 30, 100: 15, 112: 26, 116: 53, 120: 280}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 13, 17, 21, 37, 41, 49, 57, 69, 77, 93, 101, 125], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1222, 'token_per_expert': {9: 121, 13: 34, 17: 129, 21: 274, 37: 28, 41: 218, 49: 47, 57: 43, 69: 139, 77: 49, 93: 21, 101: 92, 125: 27}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 18, 22, 26, 30, 34, 54, 62, 82, 90, 98, 126], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 677, 'token_per_expert': {2: 26, 10: 74, 18: 99, 22: 133, 26: 28, 30: 26, 34: 7, 54: 13, 62: 22, 82: 166, 90: 5, 98: 21, 126: 57}}
INFO 05-03 22:31:41.945015.945015 lmp.py:1059] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.493ms | allocate_experts_across_cpu_gpu: 0.386ms
INFO 05-03 22:31:41.945960.945960 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.698204040527344e-05 seconds
INFO 05-03 22:31:41.946333.946333 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006666183471679688 seconds
INFO 05-03 22:31:41.946215.946215 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005512237548828125 seconds
INFO 05-03 22:31:41.955411.955411 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00889730453491211 seconds
INFO 05-03 22:31:41.956823.956823 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009083747863769531 seconds
INFO 05-03 22:31:41.959368.959368 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.144ms act=0.128ms bmm2=0.037ms unpad=2.109ms total=2.418ms E=32 maxT=166 S=696 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.960275.960275 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.192ms act=0.255ms bmm2=0.042ms unpad=3.207ms total=3.697ms E=32 maxT=280 S=916 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.961694.961694 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.110ms act=0.190ms bmm2=0.042ms unpad=3.572ms total=3.914ms E=32 maxT=274 S=1265 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.961367.961367 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.205ms act=0.188ms bmm2=0.034ms unpad=3.717ms total=4.145ms E=32 maxT=352 S=1219 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.961005.961005 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0047605037689208984 seconds
INFO 05-03 22:31:41.961791.961791 lmp.py:1215] [layer_moe_fused] to time: 4.935264587402344e-05 seconds
INFO 05-03 22:31:41.962361.962361 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002696514129638672 seconds
DEBUG 05-03 22:31:41.962591.962591 cuda_h.py:27] end *layer_moe_fused cost 18.091 ms
DEBUG 05-03 22:31:41.962719.962719 cuda_h.py:27] end prefill_layer cost 20.767 ms
DEBUG 05-03 22:31:41.962797.962797 lmp.py:765] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-03 22:31:41.962517.962517 lmp.py:729] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-03 22:31:41.965849.965849 cuda_h.py:27] end *sagl cost 2.344 ms
experts_cpu_alloc {'expert_ids': [119, 127, 3, 31, 91, 103, 83, 115, 47, 51, 4, 36, 64, 76, 96, 68, 52, 92, 108, 124, 104, 120, 16, 8, 40, 32, 37, 97, 21, 117, 121, 13, 89, 57, 77, 9, 101, 81, 33, 125, 14, 26, 42, 90, 122, 118, 18, 34, 70, 98, 10, 126, 22, 30], 'token_total': 261, 'token_per_expert': {119: 1, 127: 1, 3: 2, 31: 2, 91: 2, 103: 2, 83: 3, 115: 5, 47: 6, 51: 6, 4: 1, 36: 1, 64: 1, 76: 1, 96: 1, 68: 2, 52: 3, 92: 3, 108: 3, 124: 3, 104: 4, 120: 5, 16: 6, 8: 7, 40: 11, 32: 22, 37: 1, 97: 1, 21: 2, 117: 2, 121: 2, 13: 4, 89: 4, 57: 7, 77: 10, 9: 11, 101: 13, 81: 16, 33: 17, 125: 23, 14: 1, 26: 1, 42: 1, 90: 1, 122: 1, 118: 2, 18: 3, 34: 3, 70: 3, 98: 3, 10: 4, 126: 4, 22: 8, 30: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 19, 23, 27, 35, 39, 59, 63, 67, 75, 95, 99, 123], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 616, 'token_per_expert': {7: 7, 11: 8, 19: 123, 23: 105, 27: 24, 35: 10, 39: 18, 59: 21, 63: 22, 67: 129, 75: 30, 95: 7, 99: 8, 123: 104}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 12, 20, 24, 28, 48, 56, 60, 72, 84, 88, 100, 112, 116], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1055, 'token_per_expert': {0: 81, 12: 29, 20: 47, 24: 30, 28: 98, 48: 81, 56: 26, 60: 65, 72: 192, 84: 107, 88: 30, 100: 65, 112: 77, 116: 127}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 17, 25, 29, 41, 49, 53, 65, 69, 73, 105, 109, 113], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1530, 'token_per_expert': {1: 197, 17: 105, 25: 101, 29: 32, 41: 58, 49: 218, 53: 156, 65: 141, 69: 32, 73: 172, 105: 63, 109: 162, 113: 93}}
experts_gpu_alloc_device_3 {'expert_ids': [38, 46, 50, 54, 62, 66, 74, 82, 86, 94, 102, 106, 110], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 634, 'token_per_expert': {38: 28, 46: 45, 50: 77, 54: 13, 62: 37, 66: 31, 74: 37, 82: 258, 86: 16, 94: 9, 102: 14, 106: 11, 110: 58}}
INFO 05-03 22:31:41.966881.966881 lmp.py:1059] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.490ms | allocate_experts_across_cpu_gpu: 0.399ms
INFO 05-03 22:31:41.966356.966356 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.29425048828125e-05 seconds
INFO 05-03 22:31:41.967697.967697 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006833076477050781 seconds
INFO 05-03 22:31:41.967412.967412 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005335807800292969 seconds
INFO 05-03 22:31:41.977387.977387 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009393692016601562 seconds
INFO 05-03 22:31:41.978404.978404 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000965118408203125 seconds
INFO 05-03 22:31:41.982473.982473 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.214ms act=0.211ms bmm2=0.063ms unpad=2.914ms total=3.402ms E=32 maxT=129 S=646 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:41.982791.982791 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.183ms act=0.204ms bmm2=0.104ms unpad=3.262ms total=3.752ms E=32 maxT=192 S=1129 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:41.982986.982986 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.149ms act=0.191ms bmm2=0.076ms unpad=3.558ms total=3.973ms E=32 maxT=218 S=1643 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:41.983598.983598 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.159ms act=0.112ms bmm2=0.043ms unpad=3.876ms total=4.190ms E=32 maxT=258 S=678 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:41.983229.983229 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004987239837646484 seconds
INFO 05-03 22:31:41.983651.983651 lmp.py:1215] [layer_moe_fused] to time: 6.0558319091796875e-05 seconds
INFO 05-03 22:31:41.983854.983854 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0003986358642578125 seconds
DEBUG 05-03 22:31:41.984531.984531 cuda_h.py:27] end *layer_moe_fused cost 19.100 ms
DEBUG 05-03 22:31:41.984612.984612 cuda_h.py:27] end prefill_layer cost 21.629 ms
DEBUG 05-03 22:31:41.984877.984877 lmp.py:765] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-03 22:31:41.984070.984070 lmp.py:729] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-03 22:31:41.987599.987599 cuda_h.py:27] end *sagl cost 2.378 ms
experts_cpu_alloc {'expert_ids': [19, 127, 3, 27, 103, 15, 31, 111, 39, 67, 71, 51, 115, 79, 12, 84, 104, 72, 16, 48, 0, 112, 56, 124, 76, 120, 96, 68, 97, 5, 105, 113, 21, 89, 117, 121, 109, 61, 29, 69, 57, 85, 125, 49, 81, 6, 46, 126, 26, 102, 86, 94, 30, 54, 58, 82, 98, 18, 90, 118, 78], 'token_total': 456, 'token_per_expert': {19: 1, 127: 1, 3: 2, 27: 3, 103: 3, 15: 4, 31: 4, 111: 4, 39: 5, 67: 5, 71: 6, 51: 10, 115: 10, 79: 11, 12: 1, 84: 1, 104: 1, 72: 2, 16: 3, 48: 4, 0: 7, 112: 9, 56: 10, 124: 10, 76: 11, 120: 11, 96: 12, 68: 13, 97: 1, 5: 2, 105: 2, 113: 2, 21: 3, 89: 3, 117: 3, 121: 3, 109: 6, 61: 8, 29: 13, 69: 14, 57: 18, 85: 18, 125: 18, 49: 20, 81: 20, 6: 2, 46: 3, 126: 3, 26: 4, 102: 4, 86: 5, 94: 7, 30: 8, 54: 8, 58: 9, 82: 13, 98: 13, 18: 14, 90: 14, 118: 14, 78: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 23, 35, 43, 47, 55, 59, 63, 75, 83, 87, 99, 107, 119, 123], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 598, 'token_per_expert': {7: 42, 11: 49, 23: 14, 35: 44, 43: 30, 47: 26, 55: 46, 59: 37, 63: 41, 75: 29, 83: 39, 87: 13, 99: 25, 107: 35, 119: 53, 123: 75}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 20, 24, 28, 32, 36, 40, 44, 52, 60, 64, 88, 92, 100, 108, 116], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 671, 'token_per_expert': {8: 32, 20: 37, 24: 51, 28: 39, 32: 18, 36: 15, 40: 17, 44: 39, 52: 16, 60: 72, 64: 31, 88: 95, 92: 14, 100: 27, 108: 67, 116: 101}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 17, 25, 33, 37, 41, 45, 53, 65, 73, 77, 93, 101], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1553, 'token_per_expert': {1: 29, 9: 48, 13: 194, 17: 70, 25: 341, 33: 109, 37: 141, 41: 91, 45: 34, 53: 82, 65: 38, 73: 124, 77: 98, 93: 125, 101: 29}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 22, 34, 38, 42, 50, 66, 70, 74, 106, 110, 114, 122], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 818, 'token_per_expert': {2: 29, 10: 94, 14: 70, 22: 96, 34: 42, 38: 77, 42: 140, 50: 56, 66: 18, 70: 28, 74: 29, 106: 29, 110: 19, 114: 24, 122: 67}}
INFO 05-03 22:31:41.988203.988203 lmp.py:1059] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.519ms | allocate_experts_across_cpu_gpu: 0.440ms
INFO 05-03 22:31:41.988777.988777 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.532669067382812e-05 seconds
INFO 05-03 22:31:41.989019.989019 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007059574127197266 seconds
INFO 05-03 22:31:41.989304.989304 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005278587341308594 seconds
INFO 05-03 22:31:42.000456.000456 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010209083557128906 seconds
INFO 05-03 22:31:42.001593.001593 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000978708267211914 seconds
INFO 05-03 22:31:42.005451.005451 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.215ms act=0.244ms bmm2=0.087ms unpad=2.907ms total=3.452ms E=32 maxT=75 S=667 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:42.005127.005127 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.182ms act=0.182ms bmm2=0.058ms unpad=3.359ms total=3.781ms E=32 maxT=101 S=766 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:42.006234.006234 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.139ms act=0.150ms bmm2=0.027ms unpad=3.777ms total=4.094ms E=32 maxT=140 S=956 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:42.006305.006305 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.144ms act=0.190ms bmm2=0.044ms unpad=4.103ms total=4.481ms E=32 maxT=341 S=1707 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:42.007673.007673 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005284547805786133 seconds
INFO 05-03 22:31:42.007412.007412 lmp.py:1215] [layer_moe_fused] to time: 5.0067901611328125e-05 seconds
INFO 05-03 22:31:42.007875.007875 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00022792816162109375 seconds
DEBUG 05-03 22:31:42.007214.007214 cuda_h.py:27] end *layer_moe_fused cost 20.456 ms
DEBUG 05-03 22:31:42.007441.007441 cuda_h.py:27] end prefill_layer cost 23.130 ms
DEBUG 05-03 22:31:42.008967.008967 lmp.py:765] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-03 22:31:42.008853.008853 lmp.py:729] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-03 22:31:42.010805.010805 cuda_h.py:27] end *sagl cost 2.368 ms
experts_cpu_alloc {'expert_ids': [115, 119, 99, 51, 83, 35, 43, 87, 103, 39, 3, 31, 111, 95, 116, 12, 48, 88, 0, 20, 124, 4, 52, 112, 28, 44, 76, 84, 56, 80, 37, 49, 105, 109, 17, 85, 121, 1, 53, 33, 69, 57, 61, 89, 5, 6, 42, 98, 118, 10, 122, 30, 58, 94, 114, 2, 106, 54, 46, 90, 38], 'token_total': 457, 'token_per_expert': {115: 1, 119: 1, 99: 2, 51: 3, 83: 3, 35: 4, 43: 4, 87: 6, 103: 6, 39: 8, 3: 9, 31: 9, 111: 12, 95: 13, 116: 2, 12: 3, 48: 3, 88: 4, 0: 6, 20: 6, 124: 7, 4: 8, 52: 8, 112: 8, 28: 13, 44: 13, 76: 14, 84: 18, 56: 21, 80: 27, 37: 1, 49: 1, 105: 1, 109: 1, 17: 2, 85: 2, 121: 2, 1: 3, 53: 3, 33: 4, 69: 5, 57: 7, 61: 7, 89: 14, 5: 20, 6: 1, 42: 1, 98: 1, 118: 1, 10: 2, 122: 4, 30: 5, 58: 5, 94: 8, 114: 9, 2: 10, 106: 10, 54: 18, 46: 19, 90: 23, 38: 25}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 15, 19, 23, 47, 55, 59, 63, 67, 71, 75, 79, 107, 123, 127], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 884, 'token_per_expert': {7: 68, 11: 25, 15: 89, 19: 40, 23: 54, 47: 56, 55: 72, 59: 19, 63: 48, 67: 39, 71: 32, 75: 34, 79: 50, 107: 93, 123: 40, 127: 125}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 16, 24, 32, 36, 40, 60, 64, 68, 72, 92, 96, 100, 104, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1027, 'token_per_expert': {8: 40, 16: 69, 24: 39, 32: 131, 36: 35, 40: 48, 60: 33, 64: 31, 68: 115, 72: 59, 92: 63, 96: 210, 100: 34, 104: 87, 120: 33}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 13, 25, 29, 41, 45, 65, 73, 77, 81, 93, 97, 101, 113, 117], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 635, 'token_per_expert': {9: 42, 13: 56, 25: 38, 29: 41, 41: 26, 45: 20, 65: 25, 73: 31, 77: 101, 81: 50, 93: 24, 97: 37, 101: 25, 113: 51, 117: 68}}
experts_gpu_alloc_device_3 {'expert_ids': [14, 18, 22, 26, 34, 50, 62, 66, 70, 74, 78, 86, 102, 110, 126], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1093, 'token_per_expert': {14: 57, 18: 46, 22: 97, 26: 41, 34: 78, 50: 34, 62: 32, 66: 65, 70: 32, 74: 154, 78: 34, 86: 274, 102: 39, 110: 56, 126: 54}}
INFO 05-03 22:31:42.011741.011741 lmp.py:1059] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.501ms | allocate_experts_across_cpu_gpu: 0.456ms
INFO 05-03 22:31:42.011699.011699 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.198883056640625e-05 seconds
INFO 05-03 22:31:42.012231.012231 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006806850433349609 seconds
INFO 05-03 22:31:42.013543.013543 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005524158477783203 seconds
INFO 05-03 22:31:42.023294.023294 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009954214096069336 seconds
INFO 05-03 22:31:42.024450.024450 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009508132934570312 seconds
INFO 05-03 22:31:42.028947.028947 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.196ms act=0.166ms bmm2=0.044ms unpad=3.171ms total=3.576ms E=32 maxT=101 S=708 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:42.028729.028729 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.169ms act=0.165ms bmm2=0.033ms unpad=3.502ms total=3.870ms E=32 maxT=210 S=1188 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:42.029521.029521 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.158ms act=0.269ms bmm2=0.056ms unpad=3.584ms total=4.067ms E=32 maxT=125 S=965 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:42.029181.029181 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.074ms act=0.178ms bmm2=0.033ms unpad=4.091ms total=4.376ms E=32 maxT=274 S=1235 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:42.030821.030821 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005247354507446289 seconds
INFO 05-03 22:31:42.030845.030845 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:42.030461.030461 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002715587615966797 seconds
DEBUG 05-03 22:31:42.030568.030568 cuda_h.py:27] end *layer_moe_fused cost 19.932 ms
DEBUG 05-03 22:31:42.030226.030226 cuda_h.py:27] end prefill_layer cost 22.606 ms
DEBUG 05-03 22:31:42.031517.031517 lmp.py:765] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-03 22:31:42.031170.031170 lmp.py:729] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-03 22:31:42.033210.033210 cuda_h.py:27] end *sagl cost 2.403 ms
experts_cpu_alloc {'expert_ids': [67, 95, 23, 71, 19, 51, 11, 87, 103, 115, 83, 111, 55, 107, 56, 108, 4, 72, 84, 8, 100, 40, 44, 64, 68, 32, 20, 60, 104, 28, 41, 69, 73, 13, 89, 109, 97, 37, 57, 65, 121, 49, 113, 61, 125, 5, 26, 126, 42, 34, 110, 2, 6, 30, 38, 90, 78, 114, 122, 74, 14], 'token_total': 484, 'token_per_expert': {67: 1, 95: 1, 23: 3, 71: 3, 19: 4, 51: 5, 11: 7, 87: 8, 103: 9, 115: 10, 83: 11, 111: 12, 55: 13, 107: 18, 56: 1, 108: 1, 4: 2, 72: 2, 84: 3, 8: 5, 100: 6, 40: 8, 44: 8, 64: 10, 68: 10, 32: 13, 20: 16, 60: 17, 104: 17, 28: 20, 41: 2, 69: 2, 73: 3, 13: 4, 89: 4, 109: 5, 97: 8, 37: 9, 57: 9, 65: 9, 121: 10, 49: 11, 113: 14, 61: 18, 125: 19, 5: 22, 26: 1, 126: 1, 42: 2, 34: 4, 110: 4, 2: 6, 6: 6, 30: 6, 38: 6, 90: 7, 78: 8, 114: 8, 122: 8, 74: 9, 14: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 31, 35, 39, 43, 47, 59, 63, 79, 91, 99, 119, 123], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 999, 'token_per_expert': {3: 19, 7: 67, 15: 48, 27: 29, 31: 31, 35: 175, 39: 19, 43: 46, 47: 19, 59: 30, 63: 128, 79: 57, 91: 128, 99: 99, 119: 34, 123: 70}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 12, 16, 24, 36, 48, 52, 76, 80, 88, 92, 96, 112, 116, 120, 124], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 760, 'token_per_expert': {0: 29, 12: 53, 16: 44, 24: 42, 36: 37, 48: 39, 52: 27, 76: 42, 80: 44, 88: 59, 92: 90, 96: 75, 112: 28, 116: 60, 120: 58, 124: 33}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 17, 21, 25, 29, 33, 45, 53, 77, 81, 85, 93, 101, 117], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 788, 'token_per_expert': {1: 31, 9: 31, 17: 39, 21: 80, 25: 101, 29: 27, 33: 67, 45: 35, 53: 58, 77: 29, 81: 63, 85: 90, 93: 86, 101: 26, 117: 25}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 18, 46, 50, 54, 58, 62, 66, 70, 82, 86, 94, 102, 106, 118], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1065, 'token_per_expert': {10: 137, 18: 54, 46: 22, 50: 59, 54: 106, 58: 93, 62: 53, 66: 20, 70: 105, 82: 84, 86: 38, 94: 25, 102: 96, 106: 77, 118: 96}}
INFO 05-03 22:31:42.034748.034748 lmp.py:1059] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.501ms | allocate_experts_across_cpu_gpu: 0.443ms
INFO 05-03 22:31:42.034767.034767 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.651878356933594e-05 seconds
INFO 05-03 22:31:42.035931.035931 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007765293121337891 seconds
INFO 05-03 22:31:42.036674.036674 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005519390106201172 seconds
INFO 05-03 22:31:42.046210.046210 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009855985641479492 seconds
INFO 05-03 22:31:42.047785.047785 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010128021240234375 seconds
INFO 05-03 22:31:42.051960.051960 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.167ms act=0.213ms bmm2=0.031ms unpad=2.849ms total=3.259ms E=32 maxT=101 S=937 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:42.052950.052950 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.134ms act=0.177ms bmm2=0.062ms unpad=3.516ms total=3.889ms E=32 maxT=137 S=1156 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:42.052573.052573 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.203ms act=0.220ms bmm2=0.071ms unpad=3.757ms total=4.250ms E=32 maxT=90 S=899 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:42.052184.052184 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.161ms act=0.231ms bmm2=0.042ms unpad=4.337ms total=4.772ms E=32 maxT=175 S=1104 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:42.053516.053516 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0053408145904541016 seconds
INFO 05-03 22:31:42.053017.053017 lmp.py:1215] [layer_moe_fused] to time: 5.030632019042969e-05 seconds
INFO 05-03 22:31:42.053947.053947 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00036215782165527344 seconds
DEBUG 05-03 22:31:42.054360.054360 cuda_h.py:27] end *layer_moe_fused cost 20.393 ms
DEBUG 05-03 22:31:42.054209.054209 cuda_h.py:27] end prefill_layer cost 23.133 ms
DEBUG 05-03 22:31:42.054988.054988 lmp.py:765] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-03 22:31:42.054833.054833 lmp.py:729] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-03 22:31:42.057688.057688 cuda_h.py:27] end *sagl cost 3.033 ms
experts_cpu_alloc {'expert_ids': [123, 87, 127, 39, 67, 51, 7, 63, 119, 15, 91, 19, 47, 75, 11, 103, 52, 16, 36, 64, 28, 108, 96, 56, 20, 4, 12, 84, 8, 60, 40, 0, 73, 117, 1, 21, 89, 13, 57, 49, 93, 53, 125, 101, 109, 77, 14, 82, 2, 62, 38, 66, 46, 126, 22, 50, 26, 106, 114, 86, 54, 30], 'token_total': 713, 'token_per_expert': {123: 4, 87: 8, 127: 9, 39: 11, 67: 11, 51: 12, 7: 13, 63: 13, 119: 13, 15: 14, 91: 18, 19: 22, 47: 22, 75: 22, 11: 24, 103: 26, 52: 1, 16: 2, 36: 3, 64: 3, 28: 8, 108: 9, 96: 10, 56: 11, 20: 12, 4: 14, 12: 14, 84: 14, 8: 18, 60: 18, 40: 20, 0: 21, 73: 2, 117: 4, 1: 7, 21: 8, 89: 8, 13: 9, 57: 9, 49: 10, 93: 11, 53: 12, 125: 13, 101: 14, 109: 14, 77: 18, 14: 1, 82: 3, 2: 5, 62: 5, 38: 6, 66: 6, 46: 7, 126: 7, 22: 9, 50: 10, 26: 14, 106: 14, 114: 14, 86: 15, 54: 17, 30: 21}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 23, 27, 31, 35, 43, 55, 59, 71, 79, 83, 95, 99, 107, 111, 115], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 924, 'token_per_expert': {3: 72, 23: 34, 27: 44, 31: 59, 35: 88, 43: 114, 55: 33, 59: 57, 71: 75, 79: 54, 83: 35, 95: 48, 99: 30, 107: 53, 111: 48, 115: 80}}
experts_gpu_alloc_device_1 {'expert_ids': [24, 32, 44, 48, 68, 72, 76, 80, 88, 92, 100, 104, 112, 116, 120, 124], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 708, 'token_per_expert': {24: 103, 32: 24, 44: 35, 48: 36, 68: 50, 72: 21, 76: 42, 80: 71, 88: 32, 92: 32, 100: 21, 104: 79, 112: 56, 116: 23, 120: 38, 124: 45}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 17, 25, 29, 33, 41, 45, 61, 65, 69, 81, 85, 97, 105, 113, 121], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 1020, 'token_per_expert': {5: 138, 17: 67, 25: 103, 29: 31, 33: 24, 41: 160, 45: 27, 61: 40, 65: 47, 69: 24, 81: 50, 85: 59, 97: 28, 105: 42, 113: 96, 121: 84}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 18, 34, 42, 58, 70, 78, 90, 94, 98, 102, 110, 118, 122], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 731, 'token_per_expert': {6: 51, 10: 30, 18: 50, 34: 28, 42: 36, 58: 70, 70: 50, 78: 25, 90: 43, 94: 30, 98: 92, 102: 69, 110: 109, 118: 27, 122: 21}}
INFO 05-03 22:31:42.058770.058770 lmp.py:1059] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.503ms | allocate_experts_across_cpu_gpu: 0.454ms
INFO 05-03 22:31:42.059119.059119 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.961822509765625e-05 seconds
INFO 05-03 22:31:42.060449.060449 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007212162017822266 seconds
INFO 05-03 22:31:42.060112.060112 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005633831024169922 seconds
INFO 05-03 22:31:42.071324.071324 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010376691818237305 seconds
INFO 05-03 22:31:42.072098.072098 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010204315185546875 seconds
INFO 05-03 22:31:42.076373.076373 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.219ms act=0.288ms bmm2=0.034ms unpad=3.457ms total=3.998ms E=32 maxT=114 S=1166 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:42.077605.077605 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.180ms act=0.168ms bmm2=0.098ms unpad=3.902ms total=4.349ms E=32 maxT=103 S=886 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:42.077061.077061 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.147ms act=0.206ms bmm2=0.043ms unpad=4.256ms total=4.653ms E=32 maxT=160 S=1159 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:42.077866.077866 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.163ms act=0.142ms bmm2=0.049ms unpad=4.553ms total=4.908ms E=32 maxT=109 S=885 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:42.078170.078170 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0058290958404541016 seconds
INFO 05-03 22:31:42.085702.085702 lmp.py:1215] [layer_moe_fused] to time: 5.555152893066406e-05 seconds
INFO 05-03 22:31:42.085045.085045 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.007245302200317383 seconds
DEBUG 05-03 22:31:42.086813.086813 cuda_h.py:27] end *layer_moe_fused cost 28.170 ms
DEBUG 05-03 22:31:42.086471.086471 cuda_h.py:27] end prefill_layer cost 31.509 ms
DEBUG 05-03 22:31:42.086731.086731 lmp.py:765] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-03 22:31:42.086419.086419 cuda_h.py:27] end prefill cost 1245.254 ms
INFO 05-03 22:31:42.086656.086656 lmp.py:767] prefill time: 1.2454278469085693 seconds
Time taken: 5.664849761873484 seconds
generate input ids cost 0.042078256607055664 s
DEBUG 05-03 22:31:44.926153.926153 cuda_h.py:27] end generate_input_ids cost 2655.506 ms
DEBUG 05-03 22:31:44.927774.927774 cuda_h.py:27] end init_cache cost 0.028 ms
INFO 05-03 22:31:44.939743.939743 lmp.py:2040] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6495641540, 'cuda:1': 12789612544, 'cuda:2': 12791709696, 'cuda:3': 12816875520} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7313779572644639, 'cuda:1': 0.47169147306128417, 'cuda:2': 0.47165061473745756, 'cuda:3': 0.47116086639084936}
INFO 05-03 22:31:44.939482.939482 lmp.py:2058] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.939312.939312 lmp.py:2058] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.939989.939989 lmp.py:2058] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.939566.939566 lmp.py:2058] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.939358.939358 lmp.py:2058] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.939319.939319 lmp.py:2058] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.939577.939577 lmp.py:2058] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.939917.939917 lmp.py:2058] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.940831.940831 lmp.py:2058] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.940170.940170 lmp.py:2058] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.940037.940037 lmp.py:2058] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.940615.940615 lmp.py:2058] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.940640.940640 lmp.py:2058] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.940456.940456 lmp.py:2058] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.940819.940819 lmp.py:2058] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.940920.940920 lmp.py:2058] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941044.941044 lmp.py:2058] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941622.941622 lmp.py:2058] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941415.941415 lmp.py:2058] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941754.941754 lmp.py:2058] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941779.941779 lmp.py:2058] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941218.941218 lmp.py:2058] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941581.941581 lmp.py:2058] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941966.941966 lmp.py:2058] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941448.941448 lmp.py:2058] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.941310.941310 lmp.py:2058] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.942157.942157 lmp.py:2058] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.942973.942973 lmp.py:2058] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.942276.942276 lmp.py:2058] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 22:31:44.942900.942900 lmp.py:2058] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-03 22:31:45.231094.231094 cuda_h.py:27] end init_loading_placement cost 304.497 ms
DEBUG 05-03 22:31:45.231102.231102 sllm_store_c.py:27] get device uuid map
DEBUG 05-03 22:31:45.231965.231965 sllm_store_c.py:29] call client load into gpu
DEBUG 05-03 22:31:45 client.py:72] load_into_gpu: gemma4-26B-A4B, 9d549756-235e-4ce4-9af3-945c68d11eb5
INFO 05-03 22:31:45 client.py:135] Model loaded: gemma4-26B-A4B, 9d549756-235e-4ce4-9af3-945c68d11eb5
INFO 05-03 22:31:45 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 9d549756-235e-4ce4-9af3-945c68d11eb5
INFO 05-03 22:31:45 client.py:212] Model loaded
DEBUG 05-03 22:31:45.765525.765525 cuda_h.py:27] end init_general_sagl_loading_async cost 533.199 ms
DEBUG 05-03 22:31:45.784144.784144 sllm_store_c.py:27] get device uuid map
DEBUG 05-03 22:31:45.785751.785751 sllm_store_c.py:29] call client load into gpu
DEBUG 05-03 22:31:45 client.py:72] load_into_gpu: gemma4-26B-A4B, f95816bf-0b5f-445a-b45c-1d89d9637124
INFO 05-03 22:31:45 client.py:135] Model loaded: gemma4-26B-A4B, f95816bf-0b5f-445a-b45c-1d89d9637124
DEBUG 05-03 22:31:45.916979.916979 cuda_h.py:27] end init_experts_loading_async cost 151.462 ms
INFO 05-03 22:31:45.951687.951687 lmp.py:2561] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-03 22:31:46.056961.056961 cuda_h.py:27] end restore_state_dict cost 105.143 ms
DEBUG 05-03 22:31:46.057572.057572 cuda_h.py:27] end init_inputs_tokens cost 0.671 ms
DEBUG 05-03 22:31:46.057297.057297 lmp.py:729] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-03 22:31:46.060949.060949 cuda_h.py:27] end *sagl cost 2.777 ms
experts_cpu_alloc {'expert_ids': [11, 27, 59, 83, 99, 31, 67, 19, 43, 71, 127, 79, 87, 23, 86, 14, 6, 94, 102, 30, 106, 114, 10, 2, 38, 118, 45, 61, 85, 101, 109, 49, 65, 93, 17, 29, 77, 5, 69, 37, 9, 4, 100, 72, 84, 8, 92, 120, 20, 108, 28, 44, 80, 24], 'token_total': 511, 'token_per_expert': {11: 2, 27: 2, 59: 9, 83: 9, 99: 10, 31: 11, 67: 12, 19: 14, 43: 15, 71: 15, 127: 21, 79: 22, 87: 22, 23: 25, 86: 1, 14: 2, 6: 3, 94: 3, 102: 5, 30: 6, 106: 7, 114: 7, 10: 8, 2: 9, 38: 11, 118: 13, 45: 1, 61: 1, 85: 1, 101: 1, 109: 2, 49: 3, 65: 4, 93: 5, 17: 6, 29: 6, 77: 7, 5: 10, 69: 15, 37: 16, 9: 17, 4: 2, 100: 2, 72: 3, 84: 3, 8: 4, 92: 8, 120: 9, 20: 11, 108: 11, 28: 15, 44: 25, 80: 29, 24: 30}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 47, 51, 55, 63, 75, 91, 103, 107, 111, 115, 123], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 987, 'token_per_expert': {3: 64, 7: 64, 39: 137, 47: 209, 51: 32, 55: 105, 63: 45, 75: 29, 91: 66, 103: 88, 107: 37, 111: 28, 115: 29, 123: 54}}
experts_gpu_alloc_device_1 {'expert_ids': [18, 22, 26, 46, 50, 54, 58, 70, 74, 78, 90, 110, 122, 126], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 882, 'token_per_expert': {18: 43, 22: 98, 26: 52, 46: 84, 50: 71, 54: 52, 58: 24, 70: 18, 74: 71, 78: 37, 90: 148, 110: 38, 122: 71, 126: 75}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 21, 25, 33, 41, 53, 73, 89, 105, 113, 117, 121, 125], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 774, 'token_per_expert': {1: 89, 13: 18, 21: 22, 25: 18, 33: 155, 41: 22, 53: 172, 73: 29, 89: 17, 105: 56, 113: 37, 117: 23, 121: 96, 125: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 16, 32, 48, 52, 60, 64, 68, 76, 104, 112, 116, 124], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 942, 'token_per_expert': {0: 79, 16: 47, 32: 53, 48: 48, 52: 69, 60: 39, 64: 45, 68: 157, 76: 53, 104: 41, 112: 45, 116: 88, 124: 178}}
INFO 05-03 22:31:46.061518.061518 lmp.py:1059] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.655ms | allocate_experts_across_cpu_gpu: 0.246ms
INFO 05-03 22:31:46.062290.062290 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 3.743171691894531e-05 seconds
INFO 05-03 22:31:46.064077.064077 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018355846405029297 seconds
INFO 05-03 22:31:46.064724.064724 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004887580871582031 seconds
INFO 05-03 22:31:46.084572.084572 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.019482851028442383 seconds
INFO 05-03 22:31:46.085647.085647 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011053085327148438 seconds
INFO 05-03 22:31:46.091896.091896 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.293ms act=1.118ms bmm2=2.657ms unpad=1.228ms total=5.297ms E=32 maxT=209 S=1176 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.091363.091363 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.254ms act=2.044ms bmm2=1.975ms unpad=1.404ms total=5.678ms E=32 maxT=148 S=957 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.093021.093021 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.503ms act=2.964ms bmm2=2.034ms unpad=1.496ms total=6.996ms E=32 maxT=178 S=1094 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.093191.093191 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.201ms act=2.300ms bmm2=2.735ms unpad=1.950ms total=7.186ms E=32 maxT=172 S=869 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.093090.093090 lmp.py:1204] [layer_moe_fused] experts compute time: 0.008208036422729492 seconds
INFO 05-03 22:31:46.094267.094267 lmp.py:1215] [layer_moe_fused] to time: 5.435943603515625e-05 seconds
INFO 05-03 22:31:46.094154.094154 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00026702880859375 seconds
DEBUG 05-03 22:31:46.094851.094851 cuda_h.py:27] end *layer_moe_fused cost 33.596 ms
DEBUG 05-03 22:31:46.094939.094939 cuda_h.py:27] end prefill_layer cost 36.779 ms
DEBUG 05-03 22:31:46.094952.094952 lmp.py:765] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-03 22:31:46.095977.095977 lmp.py:729] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-03 22:31:46.100561.100561 cuda_h.py:27] end *sagl cost 4.759 ms
experts_cpu_alloc {'expert_ids': [19, 43, 91, 95, 107, 55, 27, 63, 11, 15, 39, 83, 51, 115, 79, 102, 54, 58, 74, 110, 62, 66, 86, 118, 26, 50, 70, 78, 14, 114, 34, 25, 29, 69, 41, 65, 17, 33, 117, 61, 73, 81, 21, 121, 77, 89, 53, 93, 48, 100, 124, 24, 56, 36, 40, 108, 112, 116, 84, 76], 'token_total': 431, 'token_per_expert': {19: 1, 43: 1, 91: 1, 95: 1, 107: 1, 55: 2, 27: 3, 63: 3, 11: 4, 15: 5, 39: 5, 83: 5, 51: 6, 115: 6, 79: 7, 102: 1, 54: 2, 58: 2, 74: 6, 110: 6, 62: 8, 66: 11, 86: 11, 118: 12, 26: 13, 50: 16, 70: 17, 78: 17, 14: 21, 114: 23, 34: 26, 25: 2, 29: 2, 69: 2, 41: 3, 65: 3, 17: 4, 33: 4, 117: 4, 61: 5, 73: 5, 81: 6, 21: 10, 121: 10, 77: 11, 89: 11, 53: 12, 93: 16, 48: 1, 100: 1, 124: 2, 24: 4, 56: 4, 36: 6, 40: 6, 108: 10, 112: 10, 116: 10, 84: 11, 76: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 31, 35, 47, 59, 67, 71, 87, 99, 103, 111, 119, 123, 127], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 778, 'token_per_expert': {3: 128, 7: 152, 23: 26, 31: 12, 35: 16, 47: 100, 59: 20, 67: 74, 71: 13, 87: 25, 99: 63, 103: 14, 111: 8, 119: 28, 123: 16, 127: 83}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 18, 22, 30, 38, 42, 46, 82, 90, 94, 98, 106, 122], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1146, 'token_per_expert': {2: 128, 6: 140, 10: 154, 18: 39, 22: 78, 30: 72, 38: 44, 42: 41, 46: 72, 82: 108, 90: 75, 94: 40, 98: 29, 106: 36, 122: 90}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 37, 45, 49, 57, 85, 97, 101, 105, 109, 113, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 898, 'token_per_expert': {1: 190, 5: 276, 9: 37, 13: 43, 37: 21, 45: 39, 49: 21, 57: 29, 85: 39, 97: 57, 101: 26, 105: 17, 109: 41, 113: 22, 125: 40}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 16, 20, 28, 52, 60, 64, 68, 72, 80, 96, 104, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 843, 'token_per_expert': {0: 132, 4: 144, 8: 116, 16: 14, 20: 16, 28: 85, 52: 46, 60: 18, 64: 29, 68: 38, 72: 25, 80: 102, 96: 31, 104: 30, 120: 17}}
INFO 05-03 22:31:46.101397.101397 lmp.py:1059] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.526ms | allocate_experts_across_cpu_gpu: 0.429ms
INFO 05-03 22:31:46.101832.101832 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.318092346191406e-05 seconds
INFO 05-03 22:31:46.102212.102212 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007283687591552734 seconds
INFO 05-03 22:31:46.102451.102451 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005321502685546875 seconds
INFO 05-03 22:31:46.121565.121565 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.019047975540161133 seconds
INFO 05-03 22:31:46.123774.123774 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000957489013671875 seconds
INFO 05-03 22:31:46.126809.126809 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.228ms act=0.262ms bmm2=0.048ms unpad=2.829ms total=3.367ms E=32 maxT=152 S=829 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.126018.126018 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.141ms act=0.184ms bmm2=0.044ms unpad=2.924ms total=3.294ms E=32 maxT=144 S=921 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.127115.127115 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.157ms act=0.193ms bmm2=0.402ms unpad=3.349ms total=4.101ms E=32 maxT=276 S=1008 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.128182.128182 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.190ms act=0.245ms bmm2=1.095ms unpad=3.243ms total=4.773ms E=32 maxT=154 S=1338 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.128142.128142 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0053827762603759766 seconds
INFO 05-03 22:31:46.128213.128213 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:46.128675.128675 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00022602081298828125 seconds
DEBUG 05-03 22:31:46.129617.129617 cuda_h.py:27] end *layer_moe_fused cost 29.103 ms
DEBUG 05-03 22:31:46.129706.129706 cuda_h.py:27] end prefill_layer cost 34.209 ms
DEBUG 05-03 22:31:46.129706.129706 lmp.py:765] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-03 22:31:46.129374.129374 lmp.py:729] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-03 22:31:46.136945.136945 cuda_h.py:27] end *sagl cost 6.442 ms
experts_cpu_alloc {'expert_ids': [47, 95, 59, 15, 27, 75, 39, 79, 63, 111, 119, 67, 83, 55, 43, 103, 26, 46, 50, 86, 98, 58, 90, 10, 66, 94, 14, 102, 106, 29, 49, 101, 117, 37, 45, 121, 85, 33, 89, 93, 21, 77, 97, 61, 112, 40, 12, 36, 32, 72, 76, 100, 116, 64, 60, 20, 56, 28, 8], 'token_total': 317, 'token_per_expert': {47: 1, 95: 1, 59: 2, 15: 3, 27: 3, 75: 3, 39: 5, 79: 7, 63: 8, 111: 8, 119: 9, 67: 11, 83: 11, 55: 12, 43: 18, 103: 18, 26: 1, 46: 1, 50: 1, 86: 1, 98: 1, 58: 2, 90: 2, 10: 3, 66: 3, 94: 3, 14: 4, 102: 4, 106: 6, 29: 1, 49: 1, 101: 1, 117: 1, 37: 3, 45: 3, 121: 3, 85: 4, 33: 5, 89: 5, 93: 5, 21: 7, 77: 8, 97: 9, 61: 12, 112: 1, 40: 2, 12: 3, 36: 3, 32: 4, 72: 4, 76: 4, 100: 5, 116: 6, 64: 8, 60: 10, 20: 11, 56: 11, 28: 12, 8: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 51, 71, 91, 99, 107, 115, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1048, 'token_per_expert': {3: 212, 7: 232, 11: 88, 19: 76, 23: 20, 31: 21, 35: 31, 51: 64, 71: 27, 91: 58, 99: 37, 107: 42, 115: 18, 123: 70, 127: 52}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 30, 42, 54, 62, 70, 74, 78, 82, 110, 118, 122, 126], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 868, 'token_per_expert': {2: 195, 6: 195, 18: 51, 30: 13, 42: 19, 54: 128, 62: 41, 70: 38, 74: 11, 78: 18, 82: 19, 110: 24, 118: 23, 122: 59, 126: 34}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 41, 53, 57, 65, 69, 73, 81, 105, 109, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 899, 'token_per_expert': {1: 248, 5: 195, 9: 63, 13: 31, 25: 23, 41: 109, 53: 13, 57: 25, 65: 57, 69: 14, 73: 13, 81: 20, 105: 33, 109: 25, 125: 30}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 16, 24, 44, 48, 52, 68, 80, 84, 96, 104, 108, 124], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 964, 'token_per_expert': {0: 199, 4: 195, 16: 22, 24: 28, 44: 18, 48: 86, 52: 13, 68: 19, 80: 27, 84: 46, 96: 14, 104: 57, 108: 215, 124: 25}}
INFO 05-03 22:31:46.137105.137105 lmp.py:1059] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.525ms | allocate_experts_across_cpu_gpu: 0.420ms
INFO 05-03 22:31:46.137103.137103 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.222724914550781e-05 seconds
INFO 05-03 22:31:46.138677.138677 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007240772247314453 seconds
INFO 05-03 22:31:46.138002.138002 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005252361297607422 seconds
INFO 05-03 22:31:46.156254.156254 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.017653942108154297 seconds
INFO 05-03 22:31:46.157416.157416 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009624958038330078 seconds
INFO 05-03 22:31:46.160142.160142 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.083ms act=0.208ms bmm2=0.051ms unpad=2.175ms total=2.517ms E=32 maxT=248 S=967 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.161157.161157 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.178ms act=0.219ms bmm2=0.596ms unpad=2.622ms total=3.616ms E=32 maxT=195 S=900 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.162933.162933 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.146ms act=0.159ms bmm2=0.771ms unpad=2.843ms total=3.919ms E=32 maxT=215 S=1061 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.162995.162995 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.274ms act=0.246ms bmm2=1.706ms unpad=2.568ms total=4.795ms E=32 maxT=232 S=1168 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.163031.163031 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005218029022216797 seconds
INFO 05-03 22:31:46.163300.163300 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:46.163167.163167 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00024437904357910156 seconds
DEBUG 05-03 22:31:46.163585.163585 cuda_h.py:27] end *layer_moe_fused cost 27.531 ms
DEBUG 05-03 22:31:46.163958.163958 cuda_h.py:27] end prefill_layer cost 34.243 ms
DEBUG 05-03 22:31:46.164644.164644 lmp.py:765] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-03 22:31:46.164814.164814 lmp.py:729] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-03 22:31:46.169397.169397 cuda_h.py:27] end *sagl cost 5.417 ms
experts_cpu_alloc {'expert_ids': [19, 23, 99, 103, 115, 59, 55, 15, 67, 79, 43, 51, 18, 90, 106, 82, 62, 118, 58, 102, 46, 126, 94, 110, 14, 42, 114, 29, 49, 17, 53, 65, 81, 89, 41, 61, 21, 13, 16, 24, 92, 112, 8, 12, 28, 36, 124, 48, 80, 84, 116, 52, 44, 108, 60], 'token_total': 253, 'token_per_expert': {19: 1, 23: 1, 99: 1, 103: 1, 115: 3, 59: 4, 55: 5, 15: 6, 67: 8, 79: 8, 43: 10, 51: 13, 18: 1, 90: 1, 106: 1, 82: 2, 62: 3, 118: 3, 58: 5, 102: 7, 46: 8, 126: 8, 94: 9, 110: 10, 14: 11, 42: 12, 114: 12, 29: 1, 49: 1, 17: 2, 53: 3, 65: 3, 81: 3, 89: 3, 41: 4, 61: 4, 21: 5, 13: 8, 16: 1, 24: 1, 92: 1, 112: 1, 8: 2, 12: 2, 28: 3, 36: 3, 124: 3, 48: 4, 80: 4, 84: 4, 116: 4, 52: 5, 44: 6, 108: 8, 60: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 31, 39, 71, 75, 83, 95, 107, 111, 123, 127], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 882, 'token_per_expert': {3: 265, 7: 256, 11: 63, 27: 33, 31: 19, 39: 33, 71: 45, 75: 15, 83: 30, 95: 35, 107: 29, 111: 25, 123: 16, 127: 18}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 22, 26, 34, 50, 54, 66, 70, 74, 78, 86, 122], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1134, 'token_per_expert': {2: 263, 6: 263, 10: 73, 22: 67, 26: 22, 34: 27, 50: 76, 54: 48, 66: 16, 70: 28, 74: 30, 78: 104, 86: 54, 122: 63}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 37, 69, 73, 77, 85, 93, 97, 101, 113, 117, 121], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 992, 'token_per_expert': {1: 256, 5: 275, 9: 70, 37: 12, 69: 60, 73: 26, 77: 21, 85: 28, 93: 51, 97: 48, 101: 63, 113: 10, 117: 50, 121: 22}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 32, 40, 56, 64, 68, 72, 76, 88, 96, 100, 104, 120], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 835, 'token_per_expert': {0: 283, 4: 270, 32: 22, 40: 45, 56: 15, 64: 23, 68: 31, 72: 23, 76: 24, 88: 24, 96: 22, 100: 16, 104: 23, 120: 14}}
INFO 05-03 22:31:46.170954.170954 lmp.py:1059] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.519ms | allocate_experts_across_cpu_gpu: 0.405ms
INFO 05-03 22:31:46.170137.170137 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.936622619628906e-05 seconds
INFO 05-03 22:31:46.171226.171226 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007135868072509766 seconds
INFO 05-03 22:31:46.172000.172000 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005052089691162109 seconds
INFO 05-03 22:31:46.190130.190130 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01795482635498047 seconds
INFO 05-03 22:31:46.191988.191988 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009520053863525391 seconds
INFO 05-03 22:31:46.194521.194521 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.158ms act=0.195ms bmm2=0.030ms unpad=2.721ms total=3.105ms E=32 maxT=275 S=1029 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.195406.195406 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.125ms act=0.121ms bmm2=0.413ms unpad=2.666ms total=3.326ms E=32 maxT=283 S=897 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.195187.195187 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.196ms act=0.212ms bmm2=0.620ms unpad=2.844ms total=3.872ms E=32 maxT=263 S=1227 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.196675.196675 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.152ms act=0.338ms bmm2=1.475ms unpad=2.686ms total=4.651ms E=32 maxT=265 S=943 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.196842.196842 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005078792572021484 seconds
INFO 05-03 22:31:46.196582.196582 lmp.py:1215] [layer_moe_fused] to time: 5.030632019042969e-05 seconds
INFO 05-03 22:31:46.196162.196162 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0001742839813232422 seconds
DEBUG 05-03 22:31:46.197308.197308 cuda_h.py:27] end *layer_moe_fused cost 27.521 ms
DEBUG 05-03 22:31:46.197012.197012 cuda_h.py:27] end prefill_layer cost 33.222 ms
DEBUG 05-03 22:31:46.197081.197081 lmp.py:765] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-03 22:31:46.197367.197367 lmp.py:729] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-03 22:31:46.204929.204929 cuda_h.py:27] end *sagl cost 6.610 ms
experts_cpu_alloc {'expert_ids': [103, 35, 95, 107, 119, 11, 123, 75, 31, 87, 15, 19, 27, 30, 42, 58, 74, 90, 62, 38, 94, 110, 114, 54, 34, 9, 45, 69, 37, 77, 117, 125, 13, 121, 89, 25, 97, 101, 48, 80, 24, 28, 12, 76, 116, 120, 124, 16, 40, 108, 56, 84], 'token_total': 280, 'token_per_expert': {103: 1, 35: 2, 95: 2, 107: 2, 119: 2, 11: 4, 123: 4, 75: 5, 31: 7, 87: 10, 15: 12, 19: 12, 27: 13, 30: 1, 42: 2, 58: 2, 74: 2, 90: 2, 62: 3, 38: 4, 94: 5, 110: 6, 114: 10, 54: 17, 34: 21, 9: 1, 45: 1, 69: 1, 37: 2, 77: 2, 117: 2, 125: 3, 13: 4, 121: 4, 89: 6, 25: 7, 97: 9, 101: 9, 48: 1, 80: 1, 24: 2, 28: 2, 12: 3, 76: 3, 116: 3, 120: 3, 124: 5, 16: 7, 40: 7, 108: 10, 56: 14, 84: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 43, 47, 51, 59, 63, 67, 79, 83, 91, 111, 115], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1094, 'token_per_expert': {3: 274, 7: 274, 39: 32, 43: 53, 47: 35, 51: 108, 59: 13, 63: 21, 67: 53, 79: 15, 83: 60, 91: 44, 111: 18, 115: 94}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 22, 26, 46, 50, 66, 78, 82, 86, 98, 106, 118], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 917, 'token_per_expert': {2: 257, 6: 272, 22: 44, 26: 35, 46: 37, 50: 29, 66: 21, 78: 38, 82: 29, 86: 26, 98: 27, 106: 61, 118: 41}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 29, 49, 53, 57, 65, 73, 81, 85, 93, 105], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 819, 'token_per_expert': {1: 266, 5: 303, 21: 17, 29: 41, 49: 26, 53: 19, 57: 13, 65: 14, 73: 20, 81: 22, 85: 34, 93: 13, 105: 31}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 20, 32, 36, 44, 52, 60, 64, 92, 96, 104], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 986, 'token_per_expert': {0: 265, 4: 306, 8: 23, 20: 62, 32: 28, 36: 29, 44: 19, 52: 27, 60: 21, 64: 48, 92: 47, 96: 41, 104: 70}}
INFO 05-03 22:31:46.205313.205313 lmp.py:1059] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.516ms | allocate_experts_across_cpu_gpu: 0.387ms
INFO 05-03 22:31:46.205781.205781 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.7220458984375e-05 seconds
INFO 05-03 22:31:46.206986.206986 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007176399230957031 seconds
INFO 05-03 22:31:46.207860.207860 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005095005035400391 seconds
INFO 05-03 22:31:46.225362.225362 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0184173583984375 seconds
INFO 05-03 22:31:46.226180.226180 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009584426879882812 seconds
INFO 05-03 22:31:46.228478.228478 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.084ms act=0.203ms bmm2=0.675ms unpad=0.857ms total=1.819ms E=32 maxT=303 S=870 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.230009.230009 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.174ms act=0.193ms bmm2=1.706ms unpad=1.008ms total=3.081ms E=32 maxT=272 S=992 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.230333.230333 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.074ms act=0.182ms bmm2=2.191ms unpad=1.104ms total=3.552ms E=32 maxT=306 S=1064 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.234687.234687 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.311ms act=0.148ms bmm2=3.177ms unpad=4.336ms total=7.972ms E=32 maxT=274 S=1170 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.235424.235424 lmp.py:1204] [layer_moe_fused] experts compute time: 0.008382797241210938 seconds
INFO 05-03 22:31:46.235117.235117 lmp.py:1215] [layer_moe_fused] to time: 5.173683166503906e-05 seconds
INFO 05-03 22:31:46.235412.235412 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00017571449279785156 seconds
DEBUG 05-03 22:31:46.235896.235896 cuda_h.py:27] end *layer_moe_fused cost 31.391 ms
DEBUG 05-03 22:31:46.235316.235316 cuda_h.py:27] end prefill_layer cost 38.234 ms
DEBUG 05-03 22:31:46.236535.236535 lmp.py:765] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-03 22:31:46.236363.236363 lmp.py:729] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-03 22:31:46.240905.240905 cuda_h.py:27] end *sagl cost 4.289 ms
experts_cpu_alloc {'expert_ids': [43, 11, 51, 79, 95, 103, 67, 19, 55, 23, 115, 119, 10, 14, 34, 38, 46, 66, 18, 102, 86, 98, 126, 30, 114, 17, 37, 41, 85, 89, 125, 65, 77, 29, 81, 48, 92, 8, 32, 100, 12, 16, 68, 76, 36, 56, 80, 116, 40, 112, 72], 'token_total': 223, 'token_per_expert': {43: 2, 11: 4, 51: 5, 79: 5, 95: 6, 103: 6, 67: 7, 19: 9, 55: 12, 23: 13, 115: 13, 119: 13, 10: 1, 14: 1, 34: 1, 38: 1, 46: 1, 66: 1, 18: 2, 102: 2, 86: 3, 98: 3, 126: 4, 30: 5, 114: 5, 17: 1, 37: 1, 41: 1, 85: 1, 89: 1, 125: 2, 65: 3, 77: 4, 29: 5, 81: 6, 48: 1, 92: 1, 8: 2, 32: 2, 100: 2, 12: 4, 16: 4, 68: 4, 76: 4, 36: 5, 56: 5, 80: 5, 116: 6, 40: 9, 112: 9, 72: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 39, 63, 71, 75, 87, 99, 107, 111, 123, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1029, 'token_per_expert': {3: 267, 7: 256, 27: 21, 39: 91, 63: 15, 71: 165, 75: 25, 87: 23, 99: 44, 107: 17, 111: 37, 123: 50, 127: 18}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 22, 50, 58, 62, 74, 82, 90, 94, 110, 118, 122], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 874, 'token_per_expert': {2: 307, 6: 294, 22: 61, 50: 54, 58: 11, 62: 7, 74: 31, 82: 6, 90: 9, 94: 66, 110: 9, 118: 9, 122: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 33, 49, 57, 61, 73, 97, 101, 113, 117], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1119, 'token_per_expert': {1: 275, 5: 257, 13: 63, 21: 22, 33: 46, 49: 62, 57: 36, 61: 90, 73: 25, 97: 18, 101: 104, 113: 34, 117: 87}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 20, 24, 28, 44, 52, 64, 88, 96, 104, 120], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 851, 'token_per_expert': {0: 256, 4: 256, 20: 133, 24: 28, 28: 17, 44: 24, 52: 12, 64: 43, 88: 17, 96: 19, 104: 15, 120: 31}}
INFO 05-03 22:31:46.241461.241461 lmp.py:1059] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.507ms | allocate_experts_across_cpu_gpu: 0.378ms
INFO 05-03 22:31:46.241644.241644 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.7697296142578125e-05 seconds
INFO 05-03 22:31:46.242450.242450 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007112026214599609 seconds
INFO 05-03 22:31:46.243316.243316 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005042552947998047 seconds
INFO 05-03 22:31:46.259484.259484 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.015749216079711914 seconds
INFO 05-03 22:31:46.260023.260023 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009329319000244141 seconds
INFO 05-03 22:31:46.263079.263079 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.215ms act=0.267ms bmm2=0.030ms unpad=2.901ms total=3.412ms E=32 maxT=267 S=1124 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.264743.264743 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.138ms act=0.198ms bmm2=0.030ms unpad=3.158ms total=3.524ms E=32 maxT=275 S=1144 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.264416.264416 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.106ms act=0.168ms bmm2=0.060ms unpad=3.457ms total=3.791ms E=32 maxT=256 S=924 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.264642.264642 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.174ms act=0.198ms bmm2=0.432ms unpad=3.517ms total=4.322ms E=32 maxT=307 S=904 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.265610.265610 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004976511001586914 seconds
INFO 05-03 22:31:46.265541.265541 lmp.py:1215] [layer_moe_fused] to time: 4.982948303222656e-05 seconds
INFO 05-03 22:31:46.265422.265422 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00025343894958496094 seconds
DEBUG 05-03 22:31:46.266225.266225 cuda_h.py:27] end *layer_moe_fused cost 25.292 ms
DEBUG 05-03 22:31:46.266168.266168 cuda_h.py:27] end prefill_layer cost 29.771 ms
DEBUG 05-03 22:31:46.266558.266558 lmp.py:765] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-03 22:31:46.266842.266842 lmp.py:729] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-03 22:31:46.273947.273947 cuda_h.py:27] end *sagl cost 6.718 ms
experts_cpu_alloc {'expert_ids': [27, 47, 71, 83, 95, 107, 15, 23, 51, 115, 11, 14, 18, 54, 66, 70, 74, 86, 118, 106, 62, 98, 126, 21, 81, 29, 33, 65, 9, 57, 25, 37, 117, 17, 69, 49, 109, 28, 36, 60, 76, 84, 80, 96, 104, 116, 120, 112, 8, 32], 'token_total': 149, 'token_per_expert': {27: 1, 47: 1, 71: 1, 83: 1, 95: 1, 107: 1, 15: 2, 23: 5, 51: 5, 115: 5, 11: 6, 14: 1, 18: 1, 54: 1, 66: 1, 70: 1, 74: 1, 86: 2, 118: 2, 106: 3, 62: 4, 98: 4, 126: 4, 21: 1, 81: 1, 29: 2, 33: 2, 65: 3, 9: 4, 57: 4, 25: 5, 37: 5, 117: 6, 17: 8, 69: 8, 49: 9, 109: 11, 28: 1, 36: 1, 60: 1, 76: 1, 84: 1, 80: 2, 96: 2, 104: 2, 116: 2, 120: 2, 112: 3, 8: 4, 32: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 35, 55, 75, 79, 87, 91, 99, 119, 123], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 734, 'token_per_expert': {3: 273, 7: 257, 19: 10, 31: 6, 35: 13, 55: 6, 75: 32, 79: 74, 87: 10, 91: 7, 99: 29, 119: 7, 123: 10}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 34, 42, 46, 78, 82, 90, 94, 102, 110, 122], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1098, 'token_per_expert': {2: 256, 6: 256, 10: 22, 34: 135, 42: 99, 46: 154, 78: 6, 82: 30, 90: 19, 94: 11, 102: 86, 110: 13, 122: 11}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 41, 53, 73, 77, 89, 93, 97, 113, 125], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1216, 'token_per_expert': {1: 257, 5: 256, 13: 16, 41: 58, 53: 56, 73: 27, 77: 159, 89: 18, 93: 137, 97: 81, 113: 105, 125: 46}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 16, 20, 40, 52, 56, 64, 68, 108, 124], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 899, 'token_per_expert': {0: 272, 4: 259, 12: 5, 16: 19, 20: 12, 40: 92, 52: 5, 56: 9, 64: 40, 68: 128, 108: 30, 124: 28}}
INFO 05-03 22:31:46.274012.274012 lmp.py:1059] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.507ms | allocate_experts_across_cpu_gpu: 0.370ms
INFO 05-03 22:31:46.274858.274858 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.340576171875e-05 seconds
INFO 05-03 22:31:46.275031.275031 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006747245788574219 seconds
INFO 05-03 22:31:46.275997.275997 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005066394805908203 seconds
INFO 05-03 22:31:46.292279.292279 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01699519157409668 seconds
INFO 05-03 22:31:46.294990.294990 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009102821350097656 seconds
INFO 05-03 22:31:46.297684.297684 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.173ms act=0.179ms bmm2=0.036ms unpad=2.641ms total=3.028ms E=32 maxT=256 S=1123 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.297763.297763 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.211ms act=0.226ms bmm2=0.048ms unpad=3.195ms total=3.680ms E=32 maxT=273 S=763 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.298938.298938 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.131ms act=0.159ms bmm2=0.053ms unpad=3.428ms total=3.771ms E=32 maxT=272 S=925 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.298266.298266 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.140ms act=0.184ms bmm2=0.058ms unpad=3.546ms total=3.928ms E=32 maxT=257 S=1285 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.298824.298824 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004854917526245117 seconds
INFO 05-03 22:31:46.299941.299941 lmp.py:1215] [layer_moe_fused] to time: 4.982948303222656e-05 seconds
INFO 05-03 22:31:46.299478.299478 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00028133392333984375 seconds
DEBUG 05-03 22:31:46.299912.299912 cuda_h.py:27] end *layer_moe_fused cost 26.368 ms
DEBUG 05-03 22:31:46.299384.299384 cuda_h.py:27] end prefill_layer cost 33.404 ms
DEBUG 05-03 22:31:46.300469.300469 lmp.py:765] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-03 22:31:46.300230.300230 lmp.py:729] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-03 22:31:46.307792.307792 cuda_h.py:27] end *sagl cost 6.742 ms
experts_cpu_alloc {'expert_ids': [11, 27, 71, 19, 31, 47, 111, 55, 35, 63, 22, 34, 74, 94, 118, 18, 50, 70, 90, 98, 21, 29, 57, 101, 45, 65, 113, 77, 64, 124, 8, 12, 108, 28, 84, 44, 116], 'token_total': 188, 'token_per_expert': {11: 1, 27: 1, 71: 1, 19: 3, 31: 3, 47: 3, 111: 3, 55: 4, 35: 5, 63: 5, 22: 1, 34: 1, 74: 1, 94: 1, 118: 1, 18: 2, 50: 2, 70: 3, 90: 4, 98: 4, 21: 1, 29: 1, 57: 1, 101: 1, 45: 3, 65: 5, 113: 5, 77: 7, 64: 1, 124: 1, 8: 2, 12: 2, 108: 3, 28: 4, 84: 22, 44: 24, 116: 56}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 59, 83, 87, 95, 103, 115, 123], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 770, 'token_per_expert': {3: 257, 7: 256, 39: 11, 59: 47, 83: 7, 87: 10, 95: 99, 103: 24, 115: 9, 123: 50}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 30, 62, 106, 114, 122, 126], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 863, 'token_per_expert': {2: 256, 6: 298, 10: 36, 30: 44, 62: 19, 106: 191, 114: 4, 122: 8, 126: 7}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 69, 85, 105, 109, 117, 121, 125], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 791, 'token_per_expert': {1: 256, 5: 256, 69: 102, 85: 11, 105: 27, 109: 74, 117: 7, 121: 42, 125: 16}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 20, 24, 48, 52, 56, 60, 96], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1484, 'token_per_expert': {0: 256, 4: 374, 20: 181, 24: 74, 48: 108, 52: 152, 56: 83, 60: 57, 96: 199}}
INFO 05-03 22:31:46.307026.307026 lmp.py:1059] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.498ms | allocate_experts_across_cpu_gpu: 0.293ms
INFO 05-03 22:31:46.308050.308050 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.363059997558594e-05 seconds
INFO 05-03 22:31:46.308713.308713 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006794929504394531 seconds
INFO 05-03 22:31:46.309407.309407 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004837512969970703 seconds
INFO 05-03 22:31:46.325399.325399 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0157473087310791 seconds
INFO 05-03 22:31:46.326510.326510 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008363723754882812 seconds
INFO 05-03 22:31:46.329284.329284 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.203ms act=0.234ms bmm2=0.028ms unpad=2.061ms total=2.526ms E=32 maxT=257 S=799 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.329517.329517 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.134ms act=0.200ms bmm2=0.034ms unpad=2.456ms total=2.824ms E=32 maxT=256 S=815 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.330861.330861 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.102ms act=0.167ms bmm2=0.561ms unpad=2.330ms total=3.160ms E=32 maxT=374 S=1599 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.330681.330681 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.168ms act=0.194ms bmm2=0.047ms unpad=3.016ms total=3.425ms E=32 maxT=298 S=883 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.330965.330965 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004189014434814453 seconds
INFO 05-03 22:31:46.330273.330273 lmp.py:1215] [layer_moe_fused] to time: 5.054473876953125e-05 seconds
INFO 05-03 22:31:46.331470.331470 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00020503997802734375 seconds
DEBUG 05-03 22:31:46.331922.331922 cuda_h.py:27] end *layer_moe_fused cost 24.383 ms
DEBUG 05-03 22:31:46.331131.331131 cuda_h.py:27] end prefill_layer cost 31.458 ms
DEBUG 05-03 22:31:46.331535.331535 lmp.py:765] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-03 22:31:46.332018.332018 lmp.py:729] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-03 22:31:46.338611.338611 cuda_h.py:27] end *sagl cost 5.888 ms
experts_cpu_alloc {'expert_ids': [43, 91, 15, 39, 87, 27, 31, 106, 126, 42, 50, 94, 98, 14, 86, 41, 57, 29, 65, 49, 33, 93, 69, 21, 85, 12, 24, 32, 40, 48, 100, 76, 96, 44], 'token_total': 195, 'token_per_expert': {43: 1, 91: 1, 15: 2, 39: 4, 87: 4, 27: 5, 31: 6, 106: 1, 126: 1, 42: 2, 50: 3, 94: 3, 98: 5, 14: 6, 86: 8, 41: 1, 57: 1, 29: 2, 65: 3, 49: 7, 33: 9, 93: 16, 69: 21, 21: 31, 85: 36, 12: 1, 24: 1, 32: 1, 40: 1, 48: 1, 100: 1, 76: 3, 96: 3, 44: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 51, 55, 63, 71, 79, 103, 123], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 604, 'token_per_expert': {3: 257, 7: 256, 51: 10, 55: 6, 63: 7, 71: 7, 79: 10, 103: 30, 123: 21}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 22, 30, 46, 54, 58, 118], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1155, 'token_per_expert': {2: 256, 6: 482, 18: 66, 22: 66, 30: 129, 46: 15, 54: 13, 58: 118, 118: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 73, 77, 81, 89, 121, 125], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1217, 'token_per_expert': {1: 256, 5: 258, 13: 104, 73: 56, 77: 141, 81: 41, 89: 147, 121: 81, 125: 133}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 64, 68, 84, 92, 120, 124], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 925, 'token_per_expert': {0: 256, 4: 257, 64: 5, 68: 39, 84: 118, 92: 140, 120: 13, 124: 97}}
INFO 05-03 22:31:46.338779.338779 lmp.py:1059] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.498ms | allocate_experts_across_cpu_gpu: 0.279ms
INFO 05-03 22:31:46.339796.339796 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.0531158447265625e-05 seconds
INFO 05-03 22:31:46.339064.339064 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007076263427734375 seconds
INFO 05-03 22:31:46.340758.340758 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.00048160552978515625 seconds
INFO 05-03 22:31:46.356308.356308 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.015520334243774414 seconds
INFO 05-03 22:31:46.357173.357173 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008053779602050781 seconds
INFO 05-03 22:31:46.359653.359653 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.145ms act=0.172ms bmm2=0.029ms unpad=1.504ms total=1.851ms E=32 maxT=257 S=627 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.360661.360661 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.093ms act=0.178ms bmm2=0.028ms unpad=2.376ms total=2.675ms E=32 maxT=257 S=941 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.360094.360094 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.265ms act=0.117ms bmm2=0.515ms unpad=2.199ms total=3.096ms E=32 maxT=482 S=1184 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.360462.360462 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.127ms act=0.177ms bmm2=0.029ms unpad=2.802ms total=3.136ms E=32 maxT=258 S=1344 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.361973.361973 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004018306732177734 seconds
INFO 05-03 22:31:46.361944.361944 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:46.361701.361701 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0003371238708496094 seconds
DEBUG 05-03 22:31:46.361034.361034 cuda_h.py:27] end *layer_moe_fused cost 23.827 ms
DEBUG 05-03 22:31:46.362454.362454 cuda_h.py:27] end prefill_layer cost 30.026 ms
DEBUG 05-03 22:31:46.362963.362963 lmp.py:765] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-03 22:31:46.362702.362702 lmp.py:729] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-03 22:31:46.369661.369661 cuda_h.py:27] end *sagl cost 7.326 ms
experts_cpu_alloc {'expert_ids': [23, 27, 75, 115, 83, 103, 39, 43, 70, 82, 14, 66, 30, 102, 74, 41, 81, 89, 21, 69, 25, 60, 92, 32, 36, 72, 24, 108, 68, 16], 'token_total': 163, 'token_per_expert': {23: 1, 27: 2, 75: 2, 115: 2, 83: 4, 103: 4, 39: 6, 43: 6, 70: 1, 82: 1, 14: 2, 66: 4, 30: 5, 102: 5, 74: 6, 41: 1, 81: 2, 89: 3, 21: 4, 69: 6, 25: 7, 60: 1, 92: 1, 32: 2, 36: 5, 72: 6, 24: 7, 108: 16, 68: 19, 16: 32}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 51, 79, 91, 95], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 928, 'token_per_expert': {3: 256, 7: 257, 15: 187, 19: 19, 51: 43, 79: 34, 91: 72, 95: 60}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 22, 46, 86, 98, 106, 114], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1065, 'token_per_expert': {2: 273, 6: 256, 22: 174, 46: 8, 86: 13, 98: 211, 106: 68, 114: 62}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 29, 57, 65, 97], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 974, 'token_per_expert': {1: 264, 5: 256, 9: 208, 29: 11, 57: 38, 65: 23, 97: 174}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 28, 48, 88, 124], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 966, 'token_per_expert': {0: 256, 4: 256, 12: 151, 28: 47, 48: 95, 88: 129, 124: 32}}
INFO 05-03 22:31:46.370013.370013 lmp.py:1059] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.490ms | allocate_experts_across_cpu_gpu: 0.249ms
INFO 05-03 22:31:46.370692.370692 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 3.7670135498046875e-05 seconds
INFO 05-03 22:31:46.371017.371017 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006601810455322266 seconds
INFO 05-03 22:31:46.372542.372542 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003840923309326172 seconds
INFO 05-03 22:31:46.385042.385042 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013209819793701172 seconds
INFO 05-03 22:31:46.386369.386369 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007662773132324219 seconds
INFO 05-03 22:31:46.388117.388117 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.218ms act=0.159ms bmm2=0.029ms unpad=1.936ms total=2.342ms E=32 maxT=257 S=955 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.389363.389363 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.081ms act=0.227ms bmm2=0.054ms unpad=2.174ms total=2.536ms E=32 maxT=273 S=1089 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.389313.389313 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.148ms act=0.225ms bmm2=0.059ms unpad=2.326ms total=2.757ms E=32 maxT=264 S=997 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.389803.389803 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.114ms act=0.152ms bmm2=0.040ms unpad=2.563ms total=2.868ms E=32 maxT=256 S=1055 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.390145.390145 lmp.py:1204] [layer_moe_fused] experts compute time: 0.00373077392578125 seconds
INFO 05-03 22:31:46.390739.390739 lmp.py:1215] [layer_moe_fused] to time: 4.982948303222656e-05 seconds
INFO 05-03 22:31:46.390208.390208 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00022983551025390625 seconds
DEBUG 05-03 22:31:46.390176.390176 cuda_h.py:27] end *layer_moe_fused cost 20.841 ms
DEBUG 05-03 22:31:46.390072.390072 cuda_h.py:27] end prefill_layer cost 28.468 ms
DEBUG 05-03 22:31:46.391985.391985 lmp.py:765] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-03 22:31:46.391976.391976 lmp.py:729] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-03 22:31:46.397450.397450 cuda_h.py:27] end *sagl cost 6.093 ms
experts_cpu_alloc {'expert_ids': [31, 51, 63, 59, 91, 67, 71, 79, 39, 26, 110, 10, 66, 118, 9, 61, 85, 25, 57, 125, 121, 113, 37, 52, 84, 92, 124, 20, 120, 44], 'token_total': 212, 'token_per_expert': {31: 1, 51: 1, 63: 1, 59: 3, 91: 7, 67: 8, 71: 12, 79: 14, 39: 16, 26: 1, 110: 1, 10: 2, 66: 2, 118: 2, 9: 1, 61: 1, 85: 1, 25: 2, 57: 2, 125: 14, 121: 21, 113: 35, 37: 49, 52: 1, 84: 1, 92: 1, 124: 1, 20: 2, 120: 3, 44: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 43, 87, 111, 115], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1096, 'token_per_expert': {3: 264, 7: 266, 11: 77, 19: 25, 43: 104, 87: 20, 111: 162, 115: 178}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 18, 46, 82, 94, 98], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 772, 'token_per_expert': {2: 256, 6: 281, 14: 13, 18: 38, 46: 7, 82: 89, 94: 84, 98: 4}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 33, 53, 81], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1180, 'token_per_expert': {1: 256, 5: 256, 13: 51, 29: 188, 33: 224, 53: 146, 81: 59}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 48, 68, 80, 100, 108], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 836, 'token_per_expert': {0: 256, 4: 256, 48: 14, 68: 87, 80: 15, 100: 26, 108: 182}}
INFO 05-03 22:31:46.398517.398517 lmp.py:1059] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.487ms | allocate_experts_across_cpu_gpu: 0.251ms
INFO 05-03 22:31:46.398766.398766 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 3.647804260253906e-05 seconds
INFO 05-03 22:31:46.399636.399636 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006322860717773438 seconds
INFO 05-03 22:31:46.399333.399333 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003814697265625 seconds
INFO 05-03 22:31:46.412746.412746 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01283574104309082 seconds
INFO 05-03 22:31:46.413782.413782 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007588863372802734 seconds
INFO 05-03 22:31:46.416398.416398 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.072ms act=0.190ms bmm2=0.030ms unpad=1.886ms total=2.178ms E=32 maxT=256 S=851 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.416734.416734 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.164ms act=0.205ms bmm2=0.063ms unpad=2.002ms total=2.434ms E=32 maxT=281 S=780 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.416780.416780 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.289ms act=0.165ms bmm2=0.059ms unpad=2.366ms total=2.879ms E=32 maxT=266 S=1159 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.416903.416903 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.080ms act=0.174ms bmm2=0.034ms unpad=2.598ms total=2.886ms E=32 maxT=256 S=1306 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.417919.417919 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0038204193115234375 seconds
INFO 05-03 22:31:46.417605.417605 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:46.417103.417103 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00028586387634277344 seconds
DEBUG 05-03 22:31:46.418872.418872 cuda_h.py:27] end *layer_moe_fused cost 20.630 ms
DEBUG 05-03 22:31:46.418623.418623 cuda_h.py:27] end prefill_layer cost 27.021 ms
DEBUG 05-03 22:31:46.418074.418074 lmp.py:765] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-03 22:31:46.418137.418137 lmp.py:729] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-03 22:31:46.426946.426946 cuda_h.py:27] end *sagl cost 7.397 ms
experts_cpu_alloc {'expert_ids': [79, 83, 87, 99, 111, 11, 31, 115, 27, 42, 102, 118, 25, 97, 33, 53, 105, 61, 65, 24, 56, 52, 68, 124, 12, 16, 104, 32, 40, 112, 88, 64], 'token_total': 261, 'token_per_expert': {79: 1, 83: 1, 87: 1, 99: 1, 111: 2, 11: 3, 31: 3, 115: 9, 27: 10, 42: 3, 102: 3, 118: 6, 25: 1, 97: 1, 33: 4, 53: 4, 105: 5, 61: 6, 65: 6, 24: 1, 56: 1, 52: 2, 68: 2, 124: 4, 12: 7, 16: 8, 104: 8, 32: 15, 40: 21, 112: 28, 88: 30, 64: 64}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 43, 63, 91, 119, 127], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1229, 'token_per_expert': {3: 480, 7: 257, 23: 115, 43: 115, 63: 11, 91: 88, 119: 146, 127: 17}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 22, 54, 62, 74, 98], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 661, 'token_per_expert': {2: 256, 6: 256, 18: 10, 22: 18, 54: 24, 62: 48, 74: 32, 98: 17}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 49, 77, 81, 93, 117], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 642, 'token_per_expert': {1: 261, 5: 256, 17: 57, 49: 19, 77: 10, 81: 9, 93: 9, 117: 21}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 28, 36, 44, 48, 76, 120], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1303, 'token_per_expert': {0: 261, 4: 256, 28: 76, 36: 84, 44: 159, 48: 98, 76: 197, 120: 172}}
INFO 05-03 22:31:46.427644.427644 lmp.py:1059] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.501ms | allocate_experts_across_cpu_gpu: 0.272ms
INFO 05-03 22:31:46.427046.427046 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.267692565917969e-05 seconds
INFO 05-03 22:31:46.428690.428690 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006470680236816406 seconds
INFO 05-03 22:31:46.428719.428719 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.00037860870361328125 seconds
INFO 05-03 22:31:46.442047.442047 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01410055160522461 seconds
INFO 05-03 22:31:46.443821.443821 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008323192596435547 seconds
INFO 05-03 22:31:46.445444.445444 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.189ms act=0.211ms bmm2=0.039ms unpad=0.889ms total=1.328ms E=32 maxT=256 S=673 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.446384.446384 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.107ms act=0.194ms bmm2=0.046ms unpad=2.285ms total=2.632ms E=32 maxT=261 S=669 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.447221.447221 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.203ms act=0.224ms bmm2=0.634ms unpad=1.955ms total=3.017ms E=32 maxT=480 S=1260 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.447430.447430 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.120ms act=0.139ms bmm2=0.074ms unpad=2.626ms total=2.959ms E=32 maxT=261 S=1494 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.447867.447867 lmp.py:1204] [layer_moe_fused] experts compute time: 0.003889799118041992 seconds
INFO 05-03 22:31:46.447415.447415 lmp.py:1215] [layer_moe_fused] to time: 5.125999450683594e-05 seconds
INFO 05-03 22:31:46.448036.448036 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00023937225341796875 seconds
DEBUG 05-03 22:31:46.448266.448266 cuda_h.py:27] end *layer_moe_fused cost 22.302 ms
DEBUG 05-03 22:31:46.448354.448354 cuda_h.py:27] end prefill_layer cost 30.005 ms
DEBUG 05-03 22:31:46.448537.448537 lmp.py:765] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-03 22:31:46.448745.448745 lmp.py:729] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-03 22:31:46.453813.453813 cuda_h.py:27] end *sagl cost 4.675 ms
experts_cpu_alloc {'expert_ids': [39, 99, 11, 95, 71, 67, 19, 111, 27, 103, 122, 94, 70, 78, 106, 14, 102, 22, 33, 93, 109, 37, 17, 117, 85, 28, 36, 92, 108, 120, 20], 'token_total': 127, 'token_per_expert': {39: 1, 99: 1, 11: 2, 95: 2, 71: 4, 67: 6, 19: 8, 111: 10, 27: 12, 103: 12, 122: 1, 94: 2, 70: 3, 78: 3, 106: 3, 14: 4, 102: 4, 22: 12, 33: 1, 93: 1, 109: 1, 37: 2, 17: 3, 117: 3, 85: 19, 28: 1, 36: 1, 92: 1, 108: 1, 120: 1, 20: 2}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 31, 35, 107, 119, 123], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1346, 'token_per_expert': {3: 389, 7: 256, 23: 82, 31: 62, 35: 90, 107: 174, 119: 201, 123: 92}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 34, 74, 86, 90, 98, 110], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 911, 'token_per_expert': {2: 256, 6: 258, 34: 116, 74: 132, 86: 27, 90: 40, 98: 29, 110: 53}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 41, 45, 101, 105], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 923, 'token_per_expert': {1: 263, 5: 257, 13: 26, 21: 33, 41: 29, 45: 220, 101: 58, 105: 37}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 40, 76, 84, 88, 116, 124], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 789, 'token_per_expert': {0: 256, 4: 256, 40: 140, 76: 10, 84: 41, 88: 77, 116: 3, 124: 6}}
INFO 05-03 22:31:46.454822.454822 lmp.py:1059] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.496ms | allocate_experts_across_cpu_gpu: 0.268ms
INFO 05-03 22:31:46.454077.454077 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.029273986816406e-05 seconds
INFO 05-03 22:31:46.455174.455174 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006608963012695312 seconds
INFO 05-03 22:31:46.456832.456832 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.00037741661071777344 seconds
INFO 05-03 22:31:46.469594.469594 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013660430908203125 seconds
INFO 05-03 22:31:46.470206.470206 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007660388946533203 seconds
INFO 05-03 22:31:46.473796.473796 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.083ms act=0.173ms bmm2=0.072ms unpad=1.803ms total=2.131ms E=32 maxT=258 S=943 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.473072.473072 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.124ms act=0.089ms bmm2=0.067ms unpad=1.997ms total=2.277ms E=32 maxT=256 S=796 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.473038.473038 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.148ms act=0.179ms bmm2=0.032ms unpad=2.207ms total=2.566ms E=32 maxT=263 S=953 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.474122.474122 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.167ms act=0.249ms bmm2=0.044ms unpad=2.744ms total=3.205ms E=32 maxT=389 S=1404 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.474120.474120 lmp.py:1204] [layer_moe_fused] experts compute time: 0.003689289093017578 seconds
INFO 05-03 22:31:46.474475.474475 lmp.py:1215] [layer_moe_fused] to time: 5.030632019042969e-05 seconds
INFO 05-03 22:31:46.474240.474240 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00016760826110839844 seconds
DEBUG 05-03 22:31:46.475501.475501 cuda_h.py:27] end *layer_moe_fused cost 21.459 ms
DEBUG 05-03 22:31:46.475298.475298 cuda_h.py:27] end prefill_layer cost 26.391 ms
DEBUG 05-03 22:31:46.475625.475625 lmp.py:765] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-03 22:31:46.475689.475689 lmp.py:729] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-03 22:31:46.481147.481147 cuda_h.py:27] end *sagl cost 5.430 ms
experts_cpu_alloc {'expert_ids': [11, 15, 43, 87, 19, 55, 107, 75, 39, 50, 54, 38, 42, 58, 14, 118, 102, 10, 9, 13, 37, 53, 109, 113, 25, 21, 17, 93, 8, 108, 112, 116, 12, 32, 80, 76, 56, 100], 'token_total': 136, 'token_per_expert': {11: 1, 15: 3, 43: 4, 87: 4, 19: 5, 55: 5, 107: 5, 75: 8, 39: 16, 50: 1, 54: 1, 38: 2, 42: 2, 58: 2, 14: 3, 118: 5, 102: 7, 10: 9, 9: 1, 13: 1, 37: 1, 53: 1, 109: 1, 113: 1, 25: 2, 21: 3, 17: 8, 93: 9, 8: 1, 108: 1, 112: 1, 116: 1, 12: 2, 32: 2, 80: 3, 76: 4, 56: 5, 100: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 47, 67, 79, 91, 99, 115, 119], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 976, 'token_per_expert': {3: 256, 7: 265, 31: 48, 47: 53, 67: 74, 79: 38, 91: 81, 99: 37, 115: 102, 119: 22}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 22, 34, 70, 78, 86, 98, 110, 122], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1014, 'token_per_expert': {2: 297, 6: 272, 22: 49, 34: 89, 70: 30, 78: 145, 86: 9, 98: 26, 110: 82, 122: 15}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 45, 73, 81, 89, 105, 117, 121], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1038, 'token_per_expert': {1: 270, 5: 256, 33: 119, 45: 26, 73: 131, 81: 19, 89: 89, 105: 39, 117: 47, 121: 42}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 20, 40, 60, 64, 84, 104, 120], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 932, 'token_per_expert': {0: 261, 4: 256, 20: 13, 40: 165, 60: 92, 64: 108, 84: 6, 104: 10, 120: 21}}
INFO 05-03 22:31:46.482527.482527 lmp.py:1059] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.491ms | allocate_experts_across_cpu_gpu: 0.302ms
INFO 05-03 22:31:46.482935.482935 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.792213439941406e-05 seconds
INFO 05-03 22:31:46.483385.483385 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006220340728759766 seconds
INFO 05-03 22:31:46.483398.483398 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004963874816894531 seconds
INFO 05-03 22:31:46.497851.497851 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014217138290405273 seconds
INFO 05-03 22:31:46.498086.498086 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007894039154052734 seconds
INFO 05-03 22:31:46.501249.501249 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.144ms act=0.270ms bmm2=0.037ms unpad=1.808ms total=2.259ms E=32 maxT=265 S=1027 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.501944.501944 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.167ms act=0.170ms bmm2=0.037ms unpad=2.410ms total=2.784ms E=32 maxT=297 S=1046 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.502929.502929 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.153ms act=0.204ms bmm2=0.076ms unpad=2.840ms total=3.274ms E=32 maxT=270 S=1066 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.502265.502265 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.116ms act=0.191ms bmm2=0.028ms unpad=3.003ms total=3.339ms E=32 maxT=261 S=957 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.503271.503271 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004238128662109375 seconds
INFO 05-03 22:31:46.503156.503156 lmp.py:1215] [layer_moe_fused] to time: 5.030632019042969e-05 seconds
INFO 05-03 22:31:46.503507.503507 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002841949462890625 seconds
DEBUG 05-03 22:31:46.503132.503132 cuda_h.py:27] end *layer_moe_fused cost 22.625 ms
DEBUG 05-03 22:31:46.504644.504644 cuda_h.py:27] end prefill_layer cost 28.337 ms
DEBUG 05-03 22:31:46.504364.504364 lmp.py:765] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-03 22:31:46.504557.504557 lmp.py:729] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-03 22:31:46.510195.510195 cuda_h.py:27] end *sagl cost 6.284 ms
experts_cpu_alloc {'expert_ids': [55, 119, 59, 107, 115, 123, 39, 91, 86, 110, 70, 18, 126, 42, 10, 53, 121, 57, 49, 65, 92, 108, 60, 120, 96, 8], 'token_total': 262, 'token_per_expert': {55: 1, 119: 1, 59: 2, 107: 2, 115: 2, 123: 11, 39: 13, 91: 17, 86: 1, 110: 2, 70: 3, 18: 5, 126: 5, 42: 8, 10: 10, 53: 1, 121: 1, 57: 7, 49: 15, 65: 88, 92: 2, 108: 3, 60: 7, 120: 16, 96: 17, 8: 22}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 75, 83, 95, 99, 103], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 921, 'token_per_expert': {3: 256, 7: 261, 75: 41, 83: 238, 95: 50, 99: 21, 103: 54}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 26, 38, 46, 50, 98], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 715, 'token_per_expert': {2: 257, 6: 257, 26: 116, 38: 11, 46: 13, 50: 38, 98: 23}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 33, 89, 97, 125], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1275, 'token_per_expert': {1: 258, 5: 256, 9: 161, 33: 231, 89: 90, 97: 106, 125: 173}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 24, 48, 52, 112], 'expert_count': 6, 'target_expert_count': 6, 'token_total': 923, 'token_per_expert': {0: 285, 4: 256, 24: 129, 48: 68, 52: 51, 112: 134}}
INFO 05-03 22:31:46.511996.511996 lmp.py:1059] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.488ms | allocate_experts_across_cpu_gpu: 0.230ms
INFO 05-03 22:31:46.511146.511146 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 3.361701965332031e-05 seconds
INFO 05-03 22:31:46.512759.512759 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006372928619384766 seconds
INFO 05-03 22:31:46.512554.512554 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.00034809112548828125 seconds
INFO 05-03 22:31:46.526687.526687 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013530254364013672 seconds
INFO 05-03 22:31:46.527323.527323 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007157325744628906 seconds
INFO 05-03 22:31:46.529325.529325 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.157ms act=0.152ms bmm2=0.039ms unpad=1.759ms total=2.107ms E=32 maxT=258 S=1387 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.529373.529373 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.140ms act=0.197ms bmm2=0.046ms unpad=2.086ms total=2.470ms E=32 maxT=261 S=970 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.530180.530180 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.172ms act=0.222ms bmm2=0.032ms unpad=2.159ms total=2.585ms E=32 maxT=257 S=749 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.530462.530462 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.110ms act=0.125ms bmm2=0.070ms unpad=2.235ms total=2.540ms E=32 maxT=285 S=990 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.530462.530462 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0034720897674560547 seconds
INFO 05-03 22:31:46.530194.530194 lmp.py:1215] [layer_moe_fused] to time: 4.887580871582031e-05 seconds
INFO 05-03 22:31:46.531652.531652 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002899169921875 seconds
DEBUG 05-03 22:31:46.531883.531883 cuda_h.py:27] end *layer_moe_fused cost 21.072 ms
DEBUG 05-03 22:31:46.531018.531018 cuda_h.py:27] end prefill_layer cost 27.585 ms
DEBUG 05-03 22:31:46.531671.531671 lmp.py:765] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-03 22:31:46.532206.532206 lmp.py:729] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-03 22:31:46.538571.538571 cuda_h.py:27] end *sagl cost 6.653 ms
experts_cpu_alloc {'expert_ids': [103, 115, 127, 67, 14, 34, 118, 18, 114, 126, 94, 78, 66, 69, 97, 101, 117, 113, 37, 89, 109, 9, 84, 92, 64, 8, 36, 100], 'token_total': 204, 'token_per_expert': {103: 1, 115: 4, 127: 4, 67: 6, 14: 1, 34: 1, 118: 1, 18: 2, 114: 7, 126: 7, 94: 11, 78: 15, 66: 33, 69: 1, 97: 1, 101: 1, 117: 2, 113: 6, 37: 9, 89: 9, 109: 10, 9: 18, 84: 1, 92: 3, 64: 7, 8: 11, 36: 14, 100: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 51, 59, 75, 107], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1084, 'token_per_expert': {3: 256, 7: 406, 19: 62, 51: 151, 59: 139, 75: 17, 107: 53}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 22, 54, 82, 86, 90], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1045, 'token_per_expert': {2: 257, 6: 256, 22: 52, 54: 52, 82: 182, 86: 178, 90: 68}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 57, 73, 81, 125], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 879, 'token_per_expert': {1: 256, 5: 256, 13: 79, 57: 134, 73: 51, 81: 21, 125: 82}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 76, 88, 96, 116], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 884, 'token_per_expert': {0: 256, 4: 264, 12: 22, 76: 218, 88: 40, 96: 32, 116: 52}}
INFO 05-03 22:31:46.539910.539910 lmp.py:1059] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.484ms | allocate_experts_across_cpu_gpu: 0.244ms
INFO 05-03 22:31:46.539874.539874 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 3.528594970703125e-05 seconds
INFO 05-03 22:31:46.540651.540651 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006146430969238281 seconds
INFO 05-03 22:31:46.541075.541075 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003559589385986328 seconds
INFO 05-03 22:31:46.554130.554130 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013143539428710938 seconds
INFO 05-03 22:31:46.555681.555681 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007236003875732422 seconds
INFO 05-03 22:31:46.557437.557437 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.179ms act=0.203ms bmm2=0.053ms unpad=1.721ms total=2.155ms E=32 maxT=257 S=1123 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.557845.557845 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.127ms act=0.190ms bmm2=0.033ms unpad=1.857ms total=2.207ms E=32 maxT=256 S=936 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.558771.558771 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.115ms act=0.160ms bmm2=0.068ms unpad=2.179ms total=2.521ms E=32 maxT=264 S=938 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.558900.558900 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.212ms act=0.205ms bmm2=0.047ms unpad=2.338ms total=2.803ms E=32 maxT=406 S=1099 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.558285.558285 lmp.py:1204] [layer_moe_fused] experts compute time: 0.003533601760864258 seconds
INFO 05-03 22:31:46.559640.559640 lmp.py:1215] [layer_moe_fused] to time: 4.935264587402344e-05 seconds
INFO 05-03 22:31:46.559117.559117 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00026988983154296875 seconds
DEBUG 05-03 22:31:46.559022.559022 cuda_h.py:27] end *layer_moe_fused cost 20.897 ms
DEBUG 05-03 22:31:46.559918.559918 cuda_h.py:27] end prefill_layer cost 27.807 ms
DEBUG 05-03 22:31:46.560063.560063 lmp.py:765] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-03 22:31:46.560048.560048 lmp.py:729] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-03 22:31:46.566556.566556 cuda_h.py:27] end *sagl cost 6.307 ms
experts_cpu_alloc {'expert_ids': [83, 115, 15, 99, 87, 55, 27, 127, 78, 34, 38, 50, 74, 46, 82, 30, 114, 25, 77, 113, 29, 81, 17, 64, 20, 8, 112, 32], 'token_total': 214, 'token_per_expert': {83: 2, 115: 2, 15: 3, 99: 3, 87: 5, 55: 10, 27: 41, 127: 64, 78: 1, 34: 2, 38: 2, 50: 2, 74: 2, 46: 4, 82: 5, 30: 6, 114: 9, 25: 2, 77: 3, 113: 3, 29: 5, 81: 6, 17: 16, 64: 1, 20: 2, 8: 3, 112: 3, 32: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 43, 103, 111], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1295, 'token_per_expert': {3: 265, 7: 256, 19: 143, 31: 256, 43: 96, 103: 152, 111: 127}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 62, 66, 70, 86], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1007, 'token_per_expert': {2: 256, 6: 269, 10: 110, 62: 68, 66: 116, 70: 72, 86: 116}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 65, 69, 85, 101, 117], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 799, 'token_per_expert': {1: 256, 5: 257, 65: 62, 69: 29, 85: 82, 101: 24, 117: 89}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 40, 52, 56, 68, 96], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 781, 'token_per_expert': {0: 256, 4: 256, 40: 9, 52: 7, 56: 47, 68: 92, 96: 114}}
INFO 05-03 22:31:46.567934.567934 lmp.py:1059] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.482ms | allocate_experts_across_cpu_gpu: 0.240ms
INFO 05-03 22:31:46.567468.567468 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 3.552436828613281e-05 seconds
INFO 05-03 22:31:46.568558.568558 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006160736083984375 seconds
INFO 05-03 22:31:46.568128.568128 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003578662872314453 seconds
INFO 05-03 22:31:46.580415.580415 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01163792610168457 seconds
INFO 05-03 22:31:46.581119.581119 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007281303405761719 seconds
INFO 05-03 22:31:46.584670.584670 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.142ms act=0.254ms bmm2=0.035ms unpad=1.769ms total=2.200ms E=32 maxT=265 S=1425 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.584619.584619 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.161ms act=0.183ms bmm2=0.033ms unpad=2.218ms total=2.595ms E=32 maxT=269 S=1040 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.584837.584837 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.147ms act=0.152ms bmm2=0.037ms unpad=2.399ms total=2.735ms E=32 maxT=257 S=834 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.584219.584219 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.143ms act=0.132ms bmm2=0.033ms unpad=2.489ms total=2.797ms E=32 maxT=256 S=797 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.585473.585473 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0037691593170166016 seconds
INFO 05-03 22:31:46.585589.585589 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:46.585852.585852 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0003936290740966797 seconds
DEBUG 05-03 22:31:46.586737.586737 cuda_h.py:27] end *layer_moe_fused cost 19.750 ms
DEBUG 05-03 22:31:46.586633.586633 cuda_h.py:27] end prefill_layer cost 26.379 ms
DEBUG 05-03 22:31:46.586371.586371 lmp.py:765] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-03 22:31:46.586879.586879 lmp.py:729] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-03 22:31:46.593691.593691 cuda_h.py:27] end *sagl cost 6.482 ms
experts_cpu_alloc {'expert_ids': [27, 35, 43, 91, 95, 10, 58, 90, 94, 70, 33, 53, 101, 93, 77, 73, 37, 57, 69, 105, 121, 81, 49, 8, 44, 52, 76, 96, 24, 48], 'token_total': 120, 'token_per_expert': {27: 1, 35: 1, 43: 1, 91: 1, 95: 1, 10: 1, 58: 1, 90: 4, 94: 4, 70: 6, 33: 1, 53: 1, 101: 1, 93: 2, 77: 3, 73: 4, 37: 5, 57: 5, 69: 5, 105: 7, 121: 10, 81: 11, 49: 17, 8: 1, 44: 1, 52: 1, 76: 1, 96: 3, 24: 5, 48: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 51, 59, 75, 79, 83], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 788, 'token_per_expert': {3: 272, 7: 310, 31: 6, 51: 40, 59: 8, 75: 34, 79: 111, 83: 7}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 26, 54, 62, 74, 82, 114], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 921, 'token_per_expert': {2: 308, 6: 273, 26: 19, 54: 13, 62: 20, 74: 61, 82: 219, 114: 8}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 45, 61, 117, 125], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1143, 'token_per_expert': {1: 291, 5: 438, 13: 116, 25: 46, 45: 24, 61: 87, 117: 61, 125: 80}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 36, 72, 92, 104], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1124, 'token_per_expert': {0: 263, 4: 397, 12: 65, 36: 110, 72: 192, 92: 34, 104: 63}}
INFO 05-03 22:31:46.594526.594526 lmp.py:1059] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.483ms | allocate_experts_across_cpu_gpu: 0.259ms
INFO 05-03 22:31:46.594166.594166 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 3.814697265625e-05 seconds
INFO 05-03 22:31:46.595263.595263 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006310939788818359 seconds
INFO 05-03 22:31:46.595529.595529 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003790855407714844 seconds
INFO 05-03 22:31:46.612100.612100 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01605367660522461 seconds
INFO 05-03 22:31:46.613162.613162 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007474422454833984 seconds
INFO 05-03 22:31:46.614993.614993 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.169ms act=0.150ms bmm2=0.039ms unpad=1.251ms total=1.609ms E=32 maxT=308 S=937 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.615420.615420 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.200ms act=0.185ms bmm2=0.045ms unpad=1.344ms total=1.774ms E=32 maxT=310 S=793 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.615424.615424 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.144ms act=0.134ms bmm2=0.762ms unpad=1.264ms total=2.305ms E=32 maxT=397 S=1151 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.616076.616076 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.132ms act=0.252ms bmm2=1.415ms unpad=1.252ms total=3.052ms E=32 maxT=438 S=1215 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.616426.616426 lmp.py:1204] [layer_moe_fused] experts compute time: 0.003668069839477539 seconds
INFO 05-03 22:31:46.616490.616490 lmp.py:1215] [layer_moe_fused] to time: 4.887580871582031e-05 seconds
INFO 05-03 22:31:46.617894.617894 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002503395080566406 seconds
DEBUG 05-03 22:31:46.617721.617721 cuda_h.py:27] end *layer_moe_fused cost 23.957 ms
DEBUG 05-03 22:31:46.617187.617187 cuda_h.py:27] end prefill_layer cost 30.739 ms
DEBUG 05-03 22:31:46.618561.618561 lmp.py:765] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-03 22:31:46.618295.618295 lmp.py:729] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-03 22:31:46.623080.623080 cuda_h.py:27] end *sagl cost 5.084 ms
experts_cpu_alloc {'expert_ids': [59, 27, 31, 91, 43, 15, 87, 119, 23, 127, 79, 54, 58, 122, 26, 74, 94, 106, 110, 30, 126, 10, 42, 22, 9, 21, 33, 81, 17, 20, 40, 96, 8, 64, 84, 32, 112], 'token_total': 186, 'token_per_expert': {59: 1, 27: 3, 31: 3, 91: 3, 43: 5, 15: 8, 87: 11, 119: 11, 23: 13, 127: 13, 79: 25, 54: 1, 58: 1, 122: 1, 26: 2, 74: 2, 94: 2, 106: 2, 110: 2, 30: 6, 126: 6, 10: 7, 42: 7, 22: 9, 9: 1, 21: 1, 33: 1, 81: 1, 17: 2, 20: 2, 40: 2, 96: 2, 8: 3, 64: 5, 84: 6, 32: 8, 112: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 47, 67, 75, 95, 99, 115, 123], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1198, 'token_per_expert': {3: 257, 7: 351, 11: 31, 47: 29, 67: 26, 75: 36, 95: 93, 99: 235, 115: 27, 123: 113}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 34, 50, 66, 82, 98, 114, 118], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 973, 'token_per_expert': {2: 270, 6: 256, 18: 71, 34: 55, 50: 29, 66: 11, 82: 130, 98: 25, 114: 117, 118: 9}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 41, 57, 61, 93, 105, 125], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 864, 'token_per_expert': {1: 258, 5: 260, 29: 20, 41: 4, 57: 172, 61: 121, 93: 4, 105: 2, 125: 23}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 24, 28, 48, 52, 76, 92, 100], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 875, 'token_per_expert': {0: 267, 4: 259, 24: 14, 28: 93, 48: 9, 52: 44, 76: 112, 92: 22, 100: 55}}
INFO 05-03 22:31:46.624136.624136 lmp.py:1059] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.504ms | allocate_experts_across_cpu_gpu: 0.296ms
INFO 05-03 22:31:46.624451.624451 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.5299530029296875e-05 seconds
INFO 05-03 22:31:46.625054.625054 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006387233734130859 seconds
INFO 05-03 22:31:46.625901.625901 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004897117614746094 seconds
INFO 05-03 22:31:46.640816.640816 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014286518096923828 seconds
INFO 05-03 22:31:46.641954.641954 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008149147033691406 seconds
INFO 05-03 22:31:46.644846.644846 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.124ms act=0.158ms bmm2=0.056ms unpad=2.161ms total=2.499ms E=32 maxT=267 S=911 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.644700.644700 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.134ms act=0.184ms bmm2=0.032ms unpad=2.415ms total=2.765ms E=32 maxT=260 S=870 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.644026.644026 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.175ms act=0.180ms bmm2=0.035ms unpad=2.680ms total=3.070ms E=32 maxT=270 S=1021 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.645766.645766 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.211ms act=0.219ms bmm2=0.067ms unpad=3.358ms total=3.855ms E=32 maxT=351 S=1294 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.645517.645517 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004270315170288086 seconds
INFO 05-03 22:31:46.645541.645541 lmp.py:1215] [layer_moe_fused] to time: 5.078315734863281e-05 seconds
INFO 05-03 22:31:46.646327.646327 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002193450927734375 seconds
DEBUG 05-03 22:31:46.646577.646577 cuda_h.py:27] end *layer_moe_fused cost 23.073 ms
DEBUG 05-03 22:31:46.646420.646420 cuda_h.py:27] end prefill_layer cost 28.462 ms
DEBUG 05-03 22:31:46.646581.646581 lmp.py:765] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-03 22:31:46.647129.647129 lmp.py:729] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-03 22:31:46.652513.652513 cuda_h.py:27] end *sagl cost 5.555 ms
experts_cpu_alloc {'expert_ids': [39, 99, 103, 119, 123, 63, 127, 74, 94, 26, 102, 18, 62, 98, 126, 66, 54, 13, 53, 117, 37, 109, 89, 41, 97, 33, 12, 104, 112, 108, 32, 92, 52, 64], 'token_total': 221, 'token_per_expert': {39: 1, 99: 1, 103: 1, 119: 3, 123: 4, 63: 38, 127: 42, 74: 1, 94: 1, 26: 2, 102: 2, 18: 3, 62: 3, 98: 3, 126: 3, 66: 5, 54: 9, 13: 1, 53: 1, 117: 2, 37: 3, 109: 3, 89: 5, 41: 8, 97: 20, 33: 33, 12: 1, 104: 1, 112: 1, 108: 2, 32: 3, 92: 4, 52: 5, 64: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 43, 59, 71, 75, 83, 91], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1142, 'token_per_expert': {3: 256, 7: 256, 11: 57, 43: 54, 59: 160, 71: 137, 75: 43, 83: 54, 91: 125}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 22, 58, 90, 106, 114, 118, 122], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 797, 'token_per_expert': {2: 277, 6: 314, 22: 13, 58: 28, 90: 60, 106: 30, 114: 11, 118: 54, 122: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 49, 57, 73, 93, 121], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1075, 'token_per_expert': {1: 256, 5: 256, 17: 47, 21: 129, 49: 69, 57: 99, 73: 49, 93: 124, 121: 46}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 36, 40, 56, 76, 100, 124], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 861, 'token_per_expert': {0: 256, 4: 256, 36: 29, 40: 10, 56: 119, 76: 104, 100: 12, 124: 75}}
INFO 05-03 22:31:46.653343.653343 lmp.py:1059] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.494ms | allocate_experts_across_cpu_gpu: 0.279ms
INFO 05-03 22:31:46.653460.653460 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 3.9577484130859375e-05 seconds
INFO 05-03 22:31:46.654269.654269 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006425380706787109 seconds
INFO 05-03 22:31:46.655062.655062 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004868507385253906 seconds
INFO 05-03 22:31:46.668540.668540 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013139724731445312 seconds
INFO 05-03 22:31:46.669868.669868 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007994174957275391 seconds
INFO 05-03 22:31:46.671931.671931 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.141ms act=0.257ms bmm2=0.033ms unpad=1.615ms total=2.047ms E=32 maxT=256 S=1232 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.672082.672082 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.158ms act=0.136ms bmm2=0.041ms unpad=2.553ms total=2.888ms E=32 maxT=256 S=1151 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.673844.673844 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.169ms act=0.199ms bmm2=0.043ms unpad=2.742ms total=3.152ms E=32 maxT=314 S=829 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.673127.673127 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.146ms act=0.146ms bmm2=0.032ms unpad=2.825ms total=3.149ms E=32 maxT=256 S=884 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.673386.673386 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0041010379791259766 seconds
INFO 05-03 22:31:46.674880.674880 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:46.674199.674199 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002949237823486328 seconds
DEBUG 05-03 22:31:46.674449.674449 cuda_h.py:27] end *layer_moe_fused cost 21.921 ms
DEBUG 05-03 22:31:46.674868.674868 cuda_h.py:27] end prefill_layer cost 27.791 ms
DEBUG 05-03 22:31:46.675101.675101 lmp.py:765] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-03 22:31:46.675387.675387 lmp.py:729] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-03 22:31:46.681432.681432 cuda_h.py:27] end *sagl cost 5.925 ms
experts_cpu_alloc {'expert_ids': [19, 127, 115, 123, 23, 99, 55, 58, 126, 122, 102, 10, 25, 49, 61, 97, 117, 101, 44, 96, 116, 12, 52, 112, 60, 68, 72, 84, 88, 40, 92, 20], 'token_total': 146, 'token_per_expert': {19: 1, 127: 1, 115: 2, 123: 4, 23: 5, 99: 6, 55: 8, 58: 3, 126: 3, 122: 5, 102: 7, 10: 8, 25: 2, 49: 2, 61: 2, 97: 4, 117: 4, 101: 7, 44: 1, 96: 1, 116: 1, 12: 2, 52: 2, 112: 2, 60: 5, 68: 5, 72: 7, 84: 7, 88: 7, 40: 9, 92: 11, 20: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 63, 67, 71, 87, 95, 119], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 824, 'token_per_expert': {3: 256, 7: 271, 11: 9, 63: 22, 67: 15, 71: 51, 87: 123, 95: 69, 119: 8}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 46, 54, 62, 66, 98], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1002, 'token_per_expert': {2: 257, 6: 259, 18: 241, 46: 42, 54: 125, 62: 57, 66: 9, 98: 12}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 37, 45, 89, 113, 125], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1068, 'token_per_expert': {1: 256, 5: 256, 9: 27, 37: 57, 45: 90, 89: 28, 113: 148, 125: 206}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 16, 24, 32, 56, 108, 120], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1056, 'token_per_expert': {0: 256, 4: 271, 16: 91, 24: 32, 32: 21, 56: 209, 108: 118, 120: 58}}
INFO 05-03 22:31:46.682036.682036 lmp.py:1059] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.485ms | allocate_experts_across_cpu_gpu: 0.265ms
INFO 05-03 22:31:46.682483.682483 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.00543212890625e-05 seconds
INFO 05-03 22:31:46.683069.683069 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006260871887207031 seconds
INFO 05-03 22:31:46.683991.683991 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003733634948730469 seconds
INFO 05-03 22:31:46.697112.697112 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013769865036010742 seconds
INFO 05-03 22:31:46.703475.703475 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0062029361724853516 seconds
INFO 05-03 22:31:46.706694.706694 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.164ms act=0.191ms bmm2=0.036ms unpad=1.707ms total=2.099ms E=32 maxT=259 S=1028 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.706648.706648 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.293ms act=0.150ms bmm2=0.047ms unpad=2.094ms total=2.585ms E=32 maxT=271 S=851 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.706072.706072 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.080ms act=0.187ms bmm2=0.034ms unpad=2.546ms total=2.846ms E=32 maxT=256 S=1089 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.707330.707330 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.072ms act=0.162ms bmm2=0.065ms unpad=2.893ms total=3.192ms E=32 maxT=271 S=1128 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.707886.707886 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004093170166015625 seconds
INFO 05-03 22:31:46.707142.707142 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:46.708784.708784 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002522468566894531 seconds
DEBUG 05-03 22:31:46.708617.708617 cuda_h.py:27] end *layer_moe_fused cost 27.410 ms
DEBUG 05-03 22:31:46.708275.708275 cuda_h.py:27] end prefill_layer cost 33.624 ms
DEBUG 05-03 22:31:46.709562.709562 lmp.py:765] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-03 22:31:46.709894.709894 lmp.py:729] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-03 22:31:46.715250.715250 cuda_h.py:27] end *sagl cost 5.925 ms
experts_cpu_alloc {'expert_ids': [67, 47, 59, 83, 91, 123, 99, 39, 115, 35, 127, 14, 18, 66, 54, 74, 102, 82, 25, 85, 101, 97, 17, 113, 20, 68, 100, 44, 12, 72, 8], 'token_total': 149, 'token_per_expert': {67: 1, 47: 2, 59: 2, 83: 3, 91: 3, 123: 3, 99: 8, 39: 10, 115: 13, 35: 14, 127: 36, 14: 1, 18: 1, 66: 1, 54: 2, 74: 2, 102: 2, 82: 3, 25: 1, 85: 1, 101: 1, 97: 3, 17: 4, 113: 4, 20: 1, 68: 2, 100: 2, 44: 4, 12: 5, 72: 6, 8: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 27, 51, 75, 111, 119], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1202, 'token_per_expert': {3: 427, 7: 264, 19: 104, 27: 94, 51: 46, 75: 47, 111: 98, 119: 122}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 26, 34, 46, 70, 86, 106], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 909, 'token_per_expert': {2: 256, 6: 256, 26: 39, 34: 11, 46: 39, 70: 49, 86: 205, 106: 54}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 33, 45, 61, 69], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1100, 'token_per_expert': {1: 256, 5: 268, 13: 97, 21: 199, 33: 122, 45: 9, 61: 25, 69: 124}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 24, 32, 36, 60, 104, 112], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 736, 'token_per_expert': {0: 256, 4: 256, 24: 37, 32: 12, 36: 55, 60: 101, 104: 10, 112: 9}}
INFO 05-03 22:31:46.715682.715682 lmp.py:1059] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.507ms | allocate_experts_across_cpu_gpu: 0.259ms
INFO 05-03 22:31:46.716322.716322 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 3.910064697265625e-05 seconds
INFO 05-03 22:31:46.717055.717055 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006306171417236328 seconds
INFO 05-03 22:31:46.717930.717930 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0003752708435058594 seconds
INFO 05-03 22:31:46.729291.729291 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01205301284790039 seconds
INFO 05-03 22:31:46.737973.737973 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.007416725158691406 seconds
INFO 05-03 22:31:46.739316.739316 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.088ms act=0.268ms bmm2=0.052ms unpad=1.250ms total=1.657ms E=32 maxT=256 S=921 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.740557.740557 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.176ms act=0.149ms bmm2=0.046ms unpad=2.104ms total=2.474ms E=32 maxT=268 S=1114 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.740455.740455 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.114ms act=0.171ms bmm2=0.033ms unpad=2.148ms total=2.466ms E=32 maxT=256 S=764 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.740748.740748 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.176ms act=0.214ms bmm2=0.070ms unpad=2.703ms total=3.163ms E=32 maxT=427 S=1297 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.741649.741649 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0036919116973876953 seconds
INFO 05-03 22:31:46.741189.741189 lmp.py:1215] [layer_moe_fused] to time: 4.9114227294921875e-05 seconds
INFO 05-03 22:31:46.741739.741739 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002532005310058594 seconds
DEBUG 05-03 22:31:46.741115.741115 cuda_h.py:27] end *layer_moe_fused cost 26.753 ms
DEBUG 05-03 22:31:46.742819.742819 cuda_h.py:27] end prefill_layer cost 32.949 ms
DEBUG 05-03 22:31:46.742971.742971 lmp.py:765] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-03 22:31:46.742930.742930 lmp.py:729] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-03 22:31:46.747075.747075 cuda_h.py:27] end *sagl cost 4.741 ms
experts_cpu_alloc {'expert_ids': [83, 111, 63, 119, 127, 59, 23, 58, 126, 46, 82, 106, 114, 14, 38, 94, 34, 26, 45, 73, 97, 125, 33, 57, 29, 37, 49, 89, 113, 65, 9, 32, 80, 88, 104, 108, 116, 28, 120, 72], 'token_total': 176, 'token_per_expert': {83: 1, 111: 1, 63: 3, 119: 3, 127: 3, 59: 4, 23: 7, 58: 1, 126: 1, 46: 2, 82: 2, 106: 2, 114: 3, 14: 4, 38: 4, 94: 4, 34: 5, 26: 9, 45: 1, 73: 1, 97: 1, 125: 1, 33: 2, 57: 3, 29: 5, 37: 8, 49: 10, 89: 12, 113: 12, 65: 13, 9: 16, 32: 1, 80: 1, 88: 1, 104: 1, 108: 1, 116: 3, 28: 5, 120: 8, 72: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 35, 47, 71, 79, 95, 103], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1048, 'token_per_expert': {3: 256, 7: 368, 11: 8, 27: 8, 35: 33, 47: 56, 71: 197, 79: 21, 95: 35, 103: 66}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 22, 42, 62, 70, 78, 98, 102], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1085, 'token_per_expert': {2: 278, 6: 256, 18: 13, 22: 128, 42: 28, 62: 43, 70: 126, 78: 65, 98: 135, 102: 13}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 41, 61, 69, 77, 85, 105], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 851, 'token_per_expert': {1: 256, 5: 256, 13: 42, 25: 85, 41: 23, 61: 21, 69: 20, 77: 57, 85: 60, 105: 31}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 16, 20, 44, 52, 68, 124], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 936, 'token_per_expert': {0: 258, 4: 256, 8: 84, 12: 90, 16: 40, 20: 44, 44: 38, 52: 15, 68: 24, 124: 87}}
INFO 05-03 22:31:46.748152.748152 lmp.py:1059] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.506ms | allocate_experts_across_cpu_gpu: 0.311ms
INFO 05-03 22:31:46.748282.748282 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.673004150390625e-05 seconds
INFO 05-03 22:31:46.749626.749626 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006246566772460938 seconds
INFO 05-03 22:31:46.749049.749049 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004949569702148438 seconds
INFO 05-03 22:31:46.764655.764655 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014306783676147461 seconds
INFO 05-03 22:31:46.766254.766254 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0025370121002197266 seconds
INFO 05-03 22:31:46.770244.770244 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.206ms act=0.207ms bmm2=0.083ms unpad=2.548ms total=3.043ms E=32 maxT=368 S=1070 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.770944.770944 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.078ms act=0.175ms bmm2=0.085ms unpad=2.656ms total=2.994ms E=32 maxT=258 S=968 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.770131.770131 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.172ms act=0.169ms bmm2=0.049ms unpad=2.791ms total=3.180ms E=32 maxT=278 S=1122 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.770388.770388 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.181ms act=0.146ms bmm2=0.082ms unpad=2.870ms total=3.278ms E=32 maxT=256 S=936 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.771131.771131 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004365682601928711 seconds
INFO 05-03 22:31:46.771817.771817 lmp.py:1215] [layer_moe_fused] to time: 5.030632019042969e-05 seconds
INFO 05-03 22:31:46.771363.771363 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0003223419189453125 seconds
DEBUG 05-03 22:31:46.772428.772428 cuda_h.py:27] end *layer_moe_fused cost 24.899 ms
DEBUG 05-03 22:31:46.772132.772132 cuda_h.py:27] end prefill_layer cost 29.959 ms
DEBUG 05-03 22:31:46.772099.772099 lmp.py:765] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-03 22:31:46.772978.772978 lmp.py:729] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-03 22:31:46.780842.780842 cuda_h.py:27] end *sagl cost 7.845 ms
experts_cpu_alloc {'expert_ids': [119, 31, 103, 111, 115, 87, 91, 30, 106, 118, 98, 114, 22, 94, 14, 50, 58, 38, 62, 29, 45, 105, 33, 37, 49, 113, 68, 104, 112, 120, 12, 96, 56, 60, 40], 'token_total': 251, 'token_per_expert': {119: 1, 31: 5, 103: 7, 111: 11, 115: 12, 87: 17, 91: 33, 30: 1, 106: 1, 118: 1, 98: 2, 114: 2, 22: 6, 94: 7, 14: 10, 50: 11, 58: 20, 38: 23, 62: 25, 29: 1, 45: 1, 105: 1, 33: 3, 37: 3, 49: 3, 113: 3, 68: 1, 104: 2, 112: 2, 120: 2, 12: 6, 96: 6, 56: 7, 60: 7, 40: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 47, 51, 55, 67, 83, 99], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1126, 'token_per_expert': {3: 256, 7: 347, 15: 66, 47: 125, 51: 41, 55: 61, 67: 98, 83: 78, 99: 54}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 34, 66, 70, 78, 102, 126], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1157, 'token_per_expert': {2: 258, 6: 256, 10: 89, 34: 38, 66: 126, 70: 68, 78: 53, 102: 70, 126: 199}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 41, 77, 81, 85, 89, 101], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 775, 'token_per_expert': {1: 256, 5: 256, 9: 20, 41: 11, 77: 11, 81: 27, 85: 37, 89: 126, 101: 31}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 16, 20, 32, 52, 76, 88], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 787, 'token_per_expert': {0: 257, 4: 256, 16: 9, 20: 64, 32: 17, 52: 34, 76: 77, 88: 73}}
INFO 05-03 22:31:46.781527.781527 lmp.py:1059] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.493ms | allocate_experts_across_cpu_gpu: 0.279ms
INFO 05-03 22:31:46.781266.781266 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.1961669921875e-05 seconds
INFO 05-03 22:31:46.782632.782632 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006356239318847656 seconds
INFO 05-03 22:31:46.783855.783855 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004887580871582031 seconds
INFO 05-03 22:31:46.796307.796307 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0129241943359375 seconds
INFO 05-03 22:31:46.799605.799605 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0028443336486816406 seconds
INFO 05-03 22:31:46.802902.802902 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.172ms act=0.202ms bmm2=0.067ms unpad=1.745ms total=2.187ms E=32 maxT=256 S=790 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.802499.802499 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.074ms act=0.189ms bmm2=0.078ms unpad=2.411ms total=2.753ms E=32 maxT=257 S=828 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.803458.803458 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.169ms act=0.303ms bmm2=0.042ms unpad=2.708ms total=3.222ms E=32 maxT=258 S=1266 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.803093.803093 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.204ms act=0.191ms bmm2=0.059ms unpad=3.019ms total=3.473ms E=32 maxT=347 S=1212 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.803181.803181 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0040738582611083984 seconds
INFO 05-03 22:31:46.803013.803013 lmp.py:1215] [layer_moe_fused] to time: 5.1021575927734375e-05 seconds
INFO 05-03 22:31:46.804193.804193 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00029921531677246094 seconds
DEBUG 05-03 22:31:46.804404.804404 cuda_h.py:27] end *layer_moe_fused cost 23.823 ms
DEBUG 05-03 22:31:46.804963.804963 cuda_h.py:27] end prefill_layer cost 31.975 ms
DEBUG 05-03 22:31:46.804668.804668 lmp.py:765] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-03 22:31:46.805805.805805 lmp.py:729] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-03 22:31:46.810726.810726 cuda_h.py:27] end *sagl cost 4.947 ms
experts_cpu_alloc {'expert_ids': [15, 39, 43, 51, 127, 83, 119, 59, 87, 123, 31, 67, 111, 47, 55, 58, 70, 74, 10, 126, 13, 85, 97, 113, 29, 89, 93, 17, 57, 49, 92, 120, 124, 16, 28, 48, 104, 88, 8, 60], 'token_total': 152, 'token_per_expert': {15: 1, 39: 1, 43: 1, 51: 1, 127: 2, 83: 3, 119: 3, 59: 5, 87: 8, 123: 8, 31: 9, 67: 10, 111: 10, 47: 13, 55: 19, 58: 1, 70: 1, 74: 1, 10: 2, 126: 2, 13: 1, 85: 1, 97: 1, 113: 1, 29: 2, 89: 2, 93: 2, 17: 4, 57: 5, 49: 6, 92: 1, 120: 1, 124: 1, 16: 2, 28: 2, 48: 2, 104: 2, 88: 4, 8: 5, 60: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 35, 79, 99, 103, 107, 115], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1128, 'token_per_expert': {3: 256, 7: 258, 11: 28, 27: 128, 35: 101, 79: 28, 99: 95, 103: 103, 107: 36, 115: 95}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 22, 46, 54, 82, 94, 118, 122], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 967, 'token_per_expert': {2: 259, 6: 256, 18: 10, 22: 150, 46: 3, 54: 4, 82: 210, 94: 7, 118: 30, 122: 38}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 37, 41, 69, 77, 101, 125], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 911, 'token_per_expert': {1: 256, 5: 256, 9: 90, 21: 84, 37: 39, 41: 32, 69: 111, 77: 23, 101: 11, 125: 9}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 36, 40, 44, 64, 72, 80, 112, 116], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 938, 'token_per_expert': {0: 256, 4: 256, 36: 123, 40: 121, 44: 12, 64: 54, 72: 7, 80: 6, 112: 77, 116: 26}}
INFO 05-03 22:31:46.811180.811180 lmp.py:1059] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.502ms | allocate_experts_across_cpu_gpu: 0.314ms
INFO 05-03 22:31:46.811356.811356 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 4.839897155761719e-05 seconds
INFO 05-03 22:31:46.812366.812366 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006365776062011719 seconds
INFO 05-03 22:31:46.812676.812676 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.00048160552978515625 seconds
INFO 05-03 22:31:46.827634.827634 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014680624008178711 seconds
INFO 05-03 22:31:46.837172.837172 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.009230852127075195 seconds
INFO 05-03 22:31:46.840652.840652 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.163ms act=0.202ms bmm2=0.053ms unpad=2.559ms total=2.976ms E=32 maxT=259 S=974 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.840869.840869 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.118ms act=0.251ms bmm2=0.030ms unpad=2.625ms total=3.024ms E=32 maxT=256 S=964 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.840693.840693 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.077ms act=0.204ms bmm2=0.041ms unpad=2.837ms total=3.159ms E=32 maxT=256 S=936 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.840338.840338 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.253ms act=0.163ms bmm2=0.040ms unpad=3.093ms total=3.548ms E=32 maxT=258 S=1222 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.841147.841147 lmp.py:1204] [layer_moe_fused] experts compute time: 0.0043277740478515625 seconds
INFO 05-03 22:31:46.841264.841264 lmp.py:1215] [layer_moe_fused] to time: 5.054473876953125e-05 seconds
INFO 05-03 22:31:46.841548.841548 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002357959747314453 seconds
DEBUG 05-03 22:31:46.842037.842037 cuda_h.py:27] end *layer_moe_fused cost 31.961 ms
DEBUG 05-03 22:31:46.842503.842503 cuda_h.py:27] end prefill_layer cost 37.241 ms
DEBUG 05-03 22:31:46.842588.842588 lmp.py:765] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-03 22:31:46.842758.842758 lmp.py:729] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-03 22:31:46.849193.849193 cuda_h.py:27] end *sagl cost 6.518 ms
experts_cpu_alloc {'expert_ids': [39, 43, 83, 59, 87, 91, 10, 30, 22, 66, 118, 126, 62, 82, 106, 18, 74, 13, 17, 33, 45, 85, 93, 9, 61, 69, 25, 29, 77, 49, 65, 109, 36, 56, 76, 32, 40, 92, 104, 52, 44, 84, 100, 60, 8], 'token_total': 185, 'token_per_expert': {39: 1, 43: 1, 83: 1, 59: 2, 87: 2, 91: 2, 10: 1, 30: 1, 22: 2, 66: 2, 118: 2, 126: 2, 62: 3, 82: 3, 106: 3, 18: 6, 74: 6, 13: 1, 17: 1, 33: 1, 45: 1, 85: 1, 93: 1, 9: 2, 61: 2, 69: 2, 25: 4, 29: 4, 77: 4, 49: 5, 65: 7, 109: 14, 36: 1, 56: 1, 76: 1, 32: 2, 40: 2, 92: 2, 104: 2, 52: 9, 44: 10, 84: 11, 100: 15, 60: 19, 8: 20}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 27, 35, 47, 67, 75, 99, 111, 115, 123], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1081, 'token_per_expert': {3: 257, 7: 262, 19: 143, 27: 8, 35: 18, 47: 10, 67: 125, 75: 88, 99: 35, 111: 5, 115: 9, 123: 121}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 38, 46, 54, 58, 70, 78, 86, 102, 110, 114], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 716, 'token_per_expert': {2: 256, 6: 281, 38: 26, 46: 12, 54: 17, 58: 7, 70: 40, 78: 8, 86: 10, 102: 10, 110: 38, 114: 11}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 41, 53, 73, 101, 105, 113, 121, 125], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1082, 'token_per_expert': {1: 319, 5: 256, 37: 31, 41: 128, 53: 26, 73: 38, 101: 37, 105: 39, 113: 101, 121: 74, 125: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 20, 24, 28, 48, 72, 88, 108, 116, 120], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1032, 'token_per_expert': {0: 278, 4: 260, 20: 125, 24: 36, 28: 34, 48: 23, 72: 25, 88: 32, 108: 88, 116: 72, 120: 59}}
INFO 05-03 22:31:46.850993.850993 lmp.py:1059] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.505ms | allocate_experts_across_cpu_gpu: 0.353ms
INFO 05-03 22:31:46.850945.850945 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.626678466796875e-05 seconds
INFO 05-03 22:31:46.851928.851928 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006806850433349609 seconds
INFO 05-03 22:31:46.851841.851841 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005049705505371094 seconds
INFO 05-03 22:31:46.867115.867115 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.015039920806884766 seconds
INFO 05-03 22:31:46.872923.872923 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.005208015441894531 seconds
INFO 05-03 22:31:46.875028.875028 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.205ms act=0.194ms bmm2=0.036ms unpad=2.082ms total=2.517ms E=32 maxT=262 S=1090 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.876835.876835 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.171ms act=0.248ms bmm2=0.057ms unpad=2.922ms total=3.399ms E=32 maxT=281 S=747 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.876384.876384 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.178ms act=0.148ms bmm2=0.047ms unpad=3.338ms total=3.710ms E=32 maxT=319 S=1132 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.876210.876210 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.079ms act=0.215ms bmm2=0.044ms unpad=3.468ms total=3.806ms E=32 maxT=278 S=1127 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.877254.877254 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004658699035644531 seconds
INFO 05-03 22:31:46.877993.877993 lmp.py:1215] [layer_moe_fused] to time: 5.078315734863281e-05 seconds
INFO 05-03 22:31:46.877810.877810 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00031280517578125 seconds
DEBUG 05-03 22:31:46.878545.878545 cuda_h.py:27] end *layer_moe_fused cost 28.830 ms
DEBUG 05-03 22:31:46.878203.878203 cuda_h.py:27] end prefill_layer cost 35.650 ms
DEBUG 05-03 22:31:46.878642.878642 lmp.py:765] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-03 22:31:46.878568.878568 lmp.py:729] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-03 22:31:46.884242.884242 cuda_h.py:27] end *sagl cost 5.744 ms
experts_cpu_alloc {'expert_ids': [15, 71, 127, 35, 47, 27, 87, 26, 50, 94, 126, 102, 42, 58, 62, 14, 18, 78, 82, 98, 66, 122, 113, 9, 109, 125, 17, 89, 41, 101, 29, 21, 13, 25, 20, 48, 64, 32, 36, 40, 68, 100, 16, 44, 52, 120, 84, 56], 'token_total': 304, 'token_per_expert': {15: 3, 71: 3, 127: 4, 35: 5, 47: 5, 27: 6, 87: 6, 26: 1, 50: 2, 94: 3, 126: 5, 102: 6, 42: 7, 58: 7, 62: 8, 14: 9, 18: 11, 78: 13, 82: 13, 98: 13, 66: 18, 122: 21, 113: 1, 9: 3, 109: 3, 125: 3, 17: 4, 89: 4, 41: 5, 101: 6, 29: 12, 21: 14, 13: 17, 25: 19, 20: 1, 48: 1, 64: 1, 32: 2, 36: 3, 40: 3, 68: 3, 100: 3, 16: 4, 44: 4, 52: 4, 120: 4, 84: 5, 56: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 43, 51, 55, 59, 83, 99, 107, 111], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 895, 'token_per_expert': {3: 256, 7: 285, 11: 57, 23: 86, 43: 14, 51: 9, 55: 24, 59: 8, 83: 83, 99: 18, 107: 40, 111: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 22, 30, 38, 70, 74, 90, 106, 110, 114], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1166, 'token_per_expert': {2: 264, 6: 258, 10: 44, 22: 133, 30: 68, 38: 30, 70: 96, 74: 68, 90: 61, 106: 91, 110: 25, 114: 28}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 37, 49, 53, 61, 73, 77, 81, 93, 121], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 939, 'token_per_expert': {1: 258, 5: 256, 33: 99, 37: 49, 49: 36, 53: 29, 61: 46, 73: 26, 77: 49, 81: 34, 93: 24, 121: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 24, 28, 60, 88, 92, 96, 112, 116, 124], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 792, 'token_per_expert': {0: 257, 4: 256, 8: 36, 24: 12, 28: 77, 60: 12, 88: 6, 92: 6, 96: 36, 112: 9, 116: 66, 124: 19}}
INFO 05-03 22:31:46.885586.885586 lmp.py:1059] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.509ms | allocate_experts_across_cpu_gpu: 0.362ms
INFO 05-03 22:31:46.885577.885577 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 5.316734313964844e-05 seconds
INFO 05-03 22:31:46.886297.886297 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000675201416015625 seconds
INFO 05-03 22:31:46.887396.887396 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005013942718505859 seconds
INFO 05-03 22:31:46.902287.902287 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.015107393264770508 seconds
INFO 05-03 22:31:46.903058.903058 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009279251098632812 seconds
INFO 05-03 22:31:46.906425.906425 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.209ms act=0.170ms bmm2=0.040ms unpad=2.395ms total=2.813ms E=32 maxT=285 S=927 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.907230.907230 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.133ms act=0.189ms bmm2=0.030ms unpad=2.862ms total=3.213ms E=32 maxT=258 S=1030 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.908005.908005 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.169ms act=0.226ms bmm2=0.038ms unpad=3.448ms total=3.881ms E=32 maxT=264 S=1303 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.908105.908105 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.121ms act=0.169ms bmm2=0.043ms unpad=3.645ms total=3.978ms E=32 maxT=257 S=836 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.908209.908209 lmp.py:1204] [layer_moe_fused] experts compute time: 0.004843950271606445 seconds
INFO 05-03 22:31:46.908703.908703 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:46.909507.909507 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00033736228942871094 seconds
DEBUG 05-03 22:31:46.909642.909642 cuda_h.py:27] end *layer_moe_fused cost 25.107 ms
DEBUG 05-03 22:31:46.909777.909777 cuda_h.py:27] end prefill_layer cost 31.158 ms
DEBUG 05-03 22:31:46.910769.910769 lmp.py:765] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-03 22:31:46.910770.910770 lmp.py:729] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-03 22:31:46.916993.916993 cuda_h.py:27] end *sagl cost 6.311 ms
experts_cpu_alloc {'expert_ids': [15, 99, 11, 23, 27, 63, 87, 31, 35, 115, 83, 103, 71, 123, 111, 30, 50, 94, 98, 118, 10, 42, 106, 110, 78, 114, 62, 18, 90, 54, 17, 109, 57, 25, 9, 89, 101, 33, 37, 69, 45, 53, 97, 8, 28, 108, 120, 12, 48, 124, 88, 76, 112, 80, 40, 24, 20, 36], 'token_total': 230, 'token_per_expert': {15: 1, 99: 1, 11: 2, 23: 2, 27: 2, 63: 2, 87: 2, 31: 3, 35: 3, 115: 3, 83: 5, 103: 5, 71: 7, 123: 7, 111: 8, 30: 1, 50: 1, 94: 1, 98: 1, 118: 1, 10: 2, 42: 3, 106: 4, 110: 4, 78: 5, 114: 5, 62: 6, 18: 7, 90: 7, 54: 9, 17: 1, 109: 1, 57: 2, 25: 3, 9: 4, 89: 4, 101: 4, 33: 5, 37: 5, 69: 5, 45: 6, 53: 6, 97: 6, 8: 1, 28: 1, 108: 2, 120: 2, 12: 3, 48: 3, 124: 3, 88: 4, 76: 5, 112: 5, 80: 6, 40: 7, 24: 8, 20: 9, 36: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39, 47, 55, 59, 67, 75, 79, 91, 95, 107, 119, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 858, 'token_per_expert': {3: 258, 7: 262, 19: 66, 39: 9, 47: 12, 55: 15, 59: 10, 67: 10, 75: 16, 79: 17, 91: 9, 95: 16, 107: 40, 119: 81, 127: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 22, 26, 34, 38, 46, 58, 66, 70, 74, 86, 102, 126], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1042, 'token_per_expert': {2: 258, 6: 256, 14: 14, 22: 12, 26: 15, 34: 79, 38: 26, 46: 73, 58: 26, 66: 47, 70: 12, 74: 49, 86: 132, 102: 20, 126: 23}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 29, 41, 65, 73, 77, 81, 85, 93, 105, 113, 117], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 935, 'token_per_expert': {1: 261, 5: 266, 13: 20, 21: 7, 29: 84, 41: 79, 65: 16, 73: 33, 77: 13, 81: 7, 85: 12, 93: 25, 105: 13, 113: 12, 117: 87}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 16, 44, 52, 56, 60, 68, 72, 84, 92, 96, 100, 104], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1031, 'token_per_expert': {0: 259, 4: 267, 16: 40, 44: 21, 52: 9, 56: 23, 60: 11, 68: 156, 72: 87, 84: 57, 92: 15, 96: 43, 100: 27, 104: 16}}
INFO 05-03 22:31:46.917947.917947 lmp.py:1059] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.518ms | allocate_experts_across_cpu_gpu: 0.418ms
INFO 05-03 22:31:46.917429.917429 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.4849853515625e-05 seconds
INFO 05-03 22:31:46.918445.918445 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006983280181884766 seconds
INFO 05-03 22:31:46.919020.919020 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005033016204833984 seconds
INFO 05-03 22:31:46.935059.935059 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.016479015350341797 seconds
INFO 05-03 22:31:46.940920.940920 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.004782199859619141 seconds
INFO 05-03 22:31:46.945179.945179 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.258ms act=0.177ms bmm2=0.034ms unpad=3.490ms total=3.959ms E=32 maxT=262 S=911 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.945652.945652 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.122ms act=0.203ms bmm2=0.030ms unpad=3.701ms total=4.056ms E=32 maxT=267 S=1099 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.945796.945796 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.081ms act=0.211ms bmm2=0.033ms unpad=3.928ms total=4.253ms E=32 maxT=266 S=987 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.945760.945760 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.167ms act=0.175ms bmm2=0.030ms unpad=4.055ms total=4.427ms E=32 maxT=258 S=1099 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.946986.946986 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005267620086669922 seconds
INFO 05-03 22:31:46.946911.946911 lmp.py:1215] [layer_moe_fused] to time: 4.982948303222656e-05 seconds
INFO 05-03 22:31:46.946904.946904 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.00026679039001464844 seconds
DEBUG 05-03 22:31:46.947071.947071 cuda_h.py:27] end *layer_moe_fused cost 30.523 ms
DEBUG 05-03 22:31:46.947920.947920 cuda_h.py:27] end prefill_layer cost 37.118 ms
DEBUG 05-03 22:31:46.947762.947762 lmp.py:765] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-03 22:31:46.947217.947217 lmp.py:729] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-03 22:31:46.954451.954451 cuda_h.py:27] end *sagl cost 6.728 ms
experts_cpu_alloc {'expert_ids': [23, 59, 111, 75, 11, 51, 103, 31, 43, 19, 47, 55, 83, 79, 26, 50, 114, 46, 98, 30, 78, 22, 42, 14, 18, 66, 110, 34, 58, 102, 53, 105, 121, 41, 13, 61, 69, 97, 113, 29, 57, 89, 109, 81, 45, 49, 56, 108, 40, 76, 32, 12, 20, 24, 68, 72, 112, 124, 64, 60], 'token_total': 302, 'token_per_expert': {23: 1, 59: 1, 111: 1, 75: 2, 11: 3, 51: 3, 103: 3, 31: 4, 43: 6, 19: 7, 47: 7, 55: 7, 83: 7, 79: 10, 26: 1, 50: 1, 114: 1, 46: 2, 98: 2, 30: 3, 78: 3, 22: 4, 42: 4, 14: 6, 18: 6, 66: 6, 110: 6, 34: 8, 58: 11, 102: 11, 53: 1, 105: 1, 121: 1, 41: 2, 13: 4, 61: 4, 69: 4, 97: 4, 113: 5, 29: 6, 57: 6, 89: 6, 109: 6, 81: 8, 45: 10, 49: 12, 56: 1, 108: 1, 40: 2, 76: 3, 32: 6, 12: 7, 20: 7, 24: 7, 68: 7, 72: 7, 112: 8, 124: 8, 64: 9, 60: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 35, 39, 63, 71, 91, 99, 107, 115, 119, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1015, 'token_per_expert': {3: 273, 7: 309, 15: 17, 27: 34, 35: 70, 39: 16, 63: 22, 71: 44, 91: 10, 99: 25, 107: 13, 115: 11, 119: 56, 123: 93, 127: 22}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 38, 54, 62, 70, 74, 82, 86, 90, 94, 106, 118, 122], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 947, 'token_per_expert': {2: 269, 6: 256, 10: 88, 38: 12, 54: 47, 62: 27, 70: 15, 74: 71, 82: 44, 86: 13, 90: 21, 94: 28, 106: 15, 118: 30, 122: 11}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 25, 33, 65, 73, 77, 85, 93, 101, 117, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 995, 'token_per_expert': {1: 310, 5: 268, 9: 13, 17: 20, 21: 49, 25: 59, 33: 34, 65: 38, 73: 13, 77: 24, 85: 54, 93: 19, 101: 18, 117: 55, 125: 21}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 16, 28, 36, 44, 48, 80, 88, 92, 96, 100, 104, 116, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 837, 'token_per_expert': {0: 275, 4: 256, 16: 26, 28: 15, 36: 17, 44: 17, 48: 27, 80: 14, 88: 37, 92: 14, 96: 24, 100: 21, 104: 27, 116: 46, 120: 21}}
INFO 05-03 22:31:46.955273.955273 lmp.py:1059] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.519ms | allocate_experts_across_cpu_gpu: 0.424ms
INFO 05-03 22:31:46.955040.955040 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.222724914550781e-05 seconds
INFO 05-03 22:31:46.956235.956235 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006818771362304688 seconds
INFO 05-03 22:31:46.956201.956201 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005080699920654297 seconds
INFO 05-03 22:31:46.972750.972750 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.015267372131347656 seconds
INFO 05-03 22:31:46.974831.974831 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018908977508544922 seconds
INFO 05-03 22:31:46.978097.978097 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.177ms act=0.179ms bmm2=0.031ms unpad=3.478ms total=3.864ms E=32 maxT=269 S=1022 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:46.979111.979111 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.162ms act=0.193ms bmm2=0.031ms unpad=3.848ms total=4.234ms E=32 maxT=310 S=1075 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:46.979395.979395 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.116ms act=0.145ms bmm2=0.046ms unpad=4.014ms total=4.322ms E=32 maxT=275 S=922 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:46.979614.979614 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.163ms act=0.280ms bmm2=0.067ms unpad=4.227ms total=4.737ms E=32 maxT=309 S=1077 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:46.980182.980182 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005443572998046875 seconds
INFO 05-03 22:31:46.980444.980444 lmp.py:1215] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-03 22:31:46.980314.980314 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0003151893615722656 seconds
DEBUG 05-03 22:31:46.981699.981699 cuda_h.py:27] end *layer_moe_fused cost 26.927 ms
DEBUG 05-03 22:31:46.981072.981072 cuda_h.py:27] end prefill_layer cost 33.846 ms
DEBUG 05-03 22:31:46.981713.981713 lmp.py:765] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-03 22:31:46.981188.981188 lmp.py:729] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-03 22:31:46.988890.988890 cuda_h.py:27] end *sagl cost 6.753 ms
experts_cpu_alloc {'expert_ids': [39, 59, 123, 127, 19, 119, 63, 111, 11, 87, 51, 31, 71, 103, 23, 75, 91, 18, 26, 50, 106, 22, 118, 62, 114, 54, 66, 78, 90, 10, 30, 89, 81, 57, 117, 101, 77, 93, 61, 13, 97, 49, 109, 21, 16, 64, 124, 116, 36, 52, 72, 92, 100, 28, 8, 20, 56, 96, 44, 108], 'token_total': 296, 'token_per_expert': {39: 3, 59: 3, 123: 3, 127: 3, 19: 4, 119: 4, 63: 5, 111: 5, 11: 6, 87: 6, 51: 7, 31: 8, 71: 9, 103: 9, 23: 10, 75: 10, 91: 10, 18: 1, 26: 1, 50: 1, 106: 1, 22: 2, 118: 2, 62: 3, 114: 3, 54: 4, 66: 5, 78: 5, 90: 6, 10: 8, 30: 9, 89: 1, 81: 2, 57: 3, 117: 4, 101: 5, 77: 6, 93: 6, 61: 7, 13: 8, 97: 9, 49: 10, 109: 11, 21: 12, 16: 1, 64: 1, 124: 1, 116: 2, 36: 3, 52: 3, 72: 3, 92: 3, 100: 3, 28: 4, 8: 5, 20: 5, 56: 5, 96: 5, 44: 6, 108: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 35, 43, 47, 55, 67, 79, 83, 95, 99, 107, 115], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1061, 'token_per_expert': {3: 302, 7: 261, 15: 28, 27: 16, 35: 17, 43: 104, 47: 12, 55: 100, 67: 20, 79: 43, 83: 38, 95: 65, 99: 14, 107: 29, 115: 12}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 34, 38, 42, 46, 58, 70, 82, 86, 94, 98, 102, 110, 126], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1040, 'token_per_expert': {2: 261, 6: 266, 34: 18, 38: 31, 42: 24, 46: 36, 58: 9, 70: 39, 82: 9, 86: 17, 94: 33, 98: 49, 102: 76, 110: 119, 126: 53}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 25, 29, 33, 41, 45, 65, 69, 85, 105, 113, 121, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 945, 'token_per_expert': {1: 262, 5: 285, 17: 95, 25: 25, 29: 24, 33: 17, 41: 28, 45: 18, 65: 37, 69: 16, 85: 28, 105: 17, 113: 16, 121: 42, 125: 35}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 24, 32, 40, 48, 60, 68, 76, 80, 84, 104, 112, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 754, 'token_per_expert': {0: 282, 4: 262, 12: 7, 24: 62, 32: 7, 40: 23, 48: 7, 60: 14, 68: 7, 76: 12, 80: 24, 84: 7, 104: 20, 112: 8, 120: 12}}
INFO 05-03 22:31:46.989626.989626 lmp.py:1059] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.523ms | allocate_experts_across_cpu_gpu: 0.428ms
INFO 05-03 22:31:46.989962.989962 lmp.py:1073] [layer_moe_fused] get_experts_task_ids time: 6.556510925292969e-05 seconds
INFO 05-03 22:31:46.990947.990947 lmp.py:1081] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006797313690185547 seconds
INFO 05-03 22:31:46.991106.991106 lmp.py:1107] [layer_moe_fused] kt_kernel_prep_submit time: 0.00051116943359375 seconds
INFO 05-03 22:31:47.001849.001849 lmp.py:1129] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009958028793334961 seconds
INFO 05-03 22:31:47.005715.005715 lmp.py:1139] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.003767728805541992 seconds
INFO 05-03 22:31:47.009609.009609 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.172ms act=0.177ms bmm2=0.031ms unpad=3.661ms total=4.042ms E=32 maxT=266 S=1091 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-03 22:31:47.010322.010322 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.199ms act=0.159ms bmm2=0.035ms unpad=3.726ms total=4.119ms E=32 maxT=285 S=1029 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-03 22:31:47.010284.010284 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.075ms act=0.220ms bmm2=0.041ms unpad=3.955ms total=4.292ms E=32 maxT=282 S=810 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-03 22:31:47.010773.010773 mlpmodule.py:2811] [fused_experts] bmm_from_padded bmm1=0.157ms act=0.250ms bmm2=0.094ms unpad=4.155ms total=4.655ms E=32 maxT=302 S=1166 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-03 22:31:47.011449.011449 lmp.py:1204] [layer_moe_fused] experts compute time: 0.005410194396972656 seconds
INFO 05-03 22:31:47.011625.011625 lmp.py:1215] [layer_moe_fused] to time: 5.030632019042969e-05 seconds
INFO 05-03 22:31:47.011406.011406 lmp.py:1221] [layer_moe_fused] scatter_reduce_ time: 0.0002524852752685547 seconds
DEBUG 05-03 22:31:47.011526.011526 cuda_h.py:27] end *layer_moe_fused cost 23.280 ms
DEBUG 05-03 22:31:47.012284.012284 cuda_h.py:27] end prefill_layer cost 30.336 ms
DEBUG 05-03 22:31:47.012871.012871 lmp.py:765] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-03 22:31:47.012689.012689 cuda_h.py:27] end prefill cost 954.367 ms
INFO 05-03 22:31:47.012924.012924 lmp.py:767] prefill time: 0.9544680118560791 seconds
Time taken: 4.908282354474068 seconds
X512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x55c536ff5930, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
CPUInfer[0x55c4f46ca900]: Goodbye
