here pin
INFO 05-03 18:01:29.834292.834292 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-03 18:01:30.375129.375129 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-03 18:01:30.816327.816327 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-03 18:01:30.816189.816189 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 0.982s
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
INFO 05-03 18:01:38.107827.107827 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-03 18:01:38.519736.519736 cuda_h.py:27] end init_cmv_hmv cost 412.684 ms
DEBUG 05-03 18:01:38.527868.527868 cuda_memory_view.py:1366] 
DEBUG 05-03 18:01:38.527868.527868 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.002547025680541992
DEBUG 05-03 18:01:38.543912.543912 mlpmodule.py:993] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-03 18:01:38.543681.543681 cuda_memory_view.py:1370] 
DEBUG 05-03 18:01:38.543681.543681 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.016166210174560547
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-03 18:01:40.394686.394686 lmp.py:322] init kt-kernel layer 0 ok
INFO 05-03 18:01:41.183916.183916 lmp.py:322] init kt-kernel layer 1 ok
INFO 05-03 18:01:41.993963.993963 lmp.py:322] init kt-kernel layer 2 ok
INFO 05-03 18:01:42.817036.817036 lmp.py:322] init kt-kernel layer 3 ok
INFO 05-03 18:01:43.668625.668625 lmp.py:322] init kt-kernel layer 4 ok
INFO 05-03 18:01:44.581755.581755 lmp.py:322] init kt-kernel layer 5 ok
INFO 05-03 18:01:45.580116.580116 lmp.py:322] init kt-kernel layer 6 ok
INFO 05-03 18:01:46.739024.739024 lmp.py:322] init kt-kernel layer 7 ok
INFO 05-03 18:01:47.890914.890914 lmp.py:322] init kt-kernel layer 8 ok
INFO 05-03 18:01:49.081300.081300 lmp.py:322] init kt-kernel layer 9 ok
INFO 05-03 18:01:50.268679.268679 lmp.py:322] init kt-kernel layer 10 ok
INFO 05-03 18:01:51.434439.434439 lmp.py:322] init kt-kernel layer 11 ok
INFO 05-03 18:01:52.500237.500237 lmp.py:322] init kt-kernel layer 12 ok
INFO 05-03 18:01:53.530658.530658 lmp.py:322] init kt-kernel layer 13 ok
INFO 05-03 18:01:54.520855.520855 lmp.py:322] init kt-kernel layer 14 ok
INFO 05-03 18:01:55.512572.512572 lmp.py:322] init kt-kernel layer 15 ok
INFO 05-03 18:01:56.540593.540593 lmp.py:322] init kt-kernel layer 16 ok
INFO 05-03 18:01:57.565997.565997 lmp.py:322] init kt-kernel layer 17 ok
INFO 05-03 18:01:58.578836.578836 lmp.py:322] init kt-kernel layer 18 ok
INFO 05-03 18:01:59.661935.661935 lmp.py:322] init kt-kernel layer 19 ok
INFO 05-03 18:02:00.737225.737225 lmp.py:322] init kt-kernel layer 20 ok
INFO 05-03 18:02:01.814582.814582 lmp.py:322] init kt-kernel layer 21 ok
INFO 05-03 18:02:02.880995.880995 lmp.py:322] init kt-kernel layer 22 ok
CPUInfer[0x605de3359800]: Hello
WorkerPool[0x605de3366490] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x605df8757520]: Hello
WorkerPool[0x605df6609030] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVINFO 05-03 18:02:03.961254.961254 lmp.py:322] init kt-kernel layer 23 ok
INFO 05-03 18:02:04.908815.908815 lmp.py:322] init kt-kernel layer 24 ok
INFO 05-03 18:02:05.708967.708967 lmp.py:322] init kt-kernel layer 25 ok
INFO 05-03 18:02:06.530274.530274 lmp.py:322] init kt-kernel layer 26 ok
INFO 05-03 18:02:07.353547.353547 lmp.py:322] init kt-kernel layer 27 ok
INFO 05-03 18:02:08.158404.158404 lmp.py:322] init kt-kernel layer 28 ok
INFO 05-03 18:02:08.953605.953605 lmp.py:322] init kt-kernel layer 29 ok
INFO 05-03 18:02:09.059218.059218 lmp.py:264] LMP_MOE_CUDA_GRAPH=1: MoE BMM CUDAGraph streams ready; capture on first forward per cache key (multi-bucket dict); fused experts run sequentially (no thread pool).
generate input ids cost 0.05202627182006836 s
DEBUG 05-03 18:02:12.144470.144470 cuda_h.py:27] end generate_input_ids cost 3076.674 ms
DEBUG 05-03 18:02:12.144132.144132 cuda_h.py:27] end init_cache cost 0.037 ms
INFO 05-03 18:02:12.157954.157954 lmp.py:1985] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6608887748, 'cuda:1': 12877692928, 'cuda:2': 12877692928, 'cuda:3': 12877692928} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7279687297090615, 'cuda:1': 0.4699814963667064, 'cuda:2': 0.4699814963667064, 'cuda:3': 0.4699814963667064}
INFO 05-03 18:02:12.157766.157766 lmp.py:2003] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.157165.157165 lmp.py:2003] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.157981.157981 lmp.py:2003] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.157697.157697 lmp.py:2003] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.157559.157559 lmp.py:2003] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.157706.157706 lmp.py:2003] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.158486.158486 lmp.py:2003] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.158156.158156 lmp.py:2003] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.158984.158984 lmp.py:2003] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.158654.158654 lmp.py:2003] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.158177.158177 lmp.py:2003] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.158847.158847 lmp.py:2003] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.159111.159111 lmp.py:2003] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.159590.159590 lmp.py:2003] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.159616.159616 lmp.py:2003] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.159571.159571 lmp.py:2003] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.159127.159127 lmp.py:2003] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.159367.159367 lmp.py:2003] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.159386.159386 lmp.py:2003] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.159003.159003 lmp.py:2003] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.159249.159249 lmp.py:2003] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.160999.160999 lmp.py:2003] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.160291.160291 lmp.py:2003] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.160581.160581 lmp.py:2003] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.160351.160351 lmp.py:2003] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.160879.160879 lmp.py:2003] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.160171.160171 lmp.py:2003] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.160130.160130 lmp.py:2003] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.160184.160184 lmp.py:2003] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:12.160429.160429 lmp.py:2003] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-03 18:02:12.438258.438258 cuda_h.py:27] end init_loading_placement cost 293.632 ms
DEBUG 05-03 18:02:12.438530.438530 sllm_store_c.py:27] get device uuid map
DEBUG 05-03 18:02:12.438208.438208 sllm_store_c.py:29] call client load into gpu
DEBUG 05-03 18:02:12 client.py:72] load_into_gpu: gemma4-26B-A4B, 8def34fe-e631-4007-9423-71b96d61a2fe
INFO 05-03 18:02:12 client.py:135] Model loaded: gemma4-26B-A4B, 8def34fe-e631-4007-9423-71b96d61a2fe
INFO 05-03 18:02:12 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 8def34fe-e631-4007-9423-71b96d61a2fe
INFO 05-03 18:02:12 client.py:212] Model loaded
DEBUG 05-03 18:02:12.965848.965848 cuda_h.py:27] end init_general_sagl_loading_async cost 527.478 ms
DEBUG 05-03 18:02:12.984253.984253 sllm_store_c.py:27] get device uuid map
DEBUG 05-03 18:02:12.984734.984734 sllm_store_c.py:29] call client load into gpu
DEBUG 05-03 18:02:12 client.py:72] load_into_gpu: gemma4-26B-A4B, aadfc1ee-5d64-48f6-b40c-b9143b0067d3
INFO 05-03 18:02:13 client.py:135] Model loaded: gemma4-26B-A4B, aadfc1ee-5d64-48f6-b40c-b9143b0067d3
DEBUG 05-03 18:02:13.060032.060032 cuda_h.py:27] end init_experts_loading_async cost 94.990 ms
INFO 05-03 18:02:13.098126.098126 lmp.py:2506] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-03 18:02:13.196657.196657 cuda_h.py:27] end restore_state_dict cost 97.048 ms
DEBUG 05-03 18:02:13.216112.216112 cuda_h.py:27] end init_inputs_tokens cost 20.781 ms
DEBUG 05-03 18:02:13.217958.217958 lmp.py:675] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-03 18:02:13.447363.447363 cuda_h.py:27] end *sagl cost 230.772 ms
experts_cpu_alloc {'expert_ids': [11, 27, 59, 83, 99, 31, 67, 19, 43, 71, 127, 79, 87, 23, 4, 100, 72, 84, 8, 92, 120, 20, 108, 28, 44, 80, 45, 61, 85, 101, 109, 49, 65, 93, 17, 29, 77, 5, 69, 37, 9, 86, 14, 6, 94, 102, 30, 106, 114, 10, 2, 38, 118, 70], 'token_total': 499, 'token_per_expert': {11: 2, 27: 2, 59: 9, 83: 9, 99: 10, 31: 11, 67: 12, 19: 14, 43: 15, 71: 15, 127: 21, 79: 22, 87: 22, 23: 25, 4: 2, 100: 2, 72: 3, 84: 3, 8: 4, 92: 8, 120: 9, 20: 11, 108: 11, 28: 15, 44: 25, 80: 29, 45: 1, 61: 1, 85: 1, 101: 1, 109: 2, 49: 3, 65: 4, 93: 5, 17: 6, 29: 6, 77: 7, 5: 10, 69: 15, 37: 16, 9: 17, 86: 1, 14: 2, 6: 3, 94: 3, 102: 5, 30: 6, 106: 7, 114: 7, 10: 8, 2: 9, 38: 11, 118: 13, 70: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 47, 51, 55, 63, 75, 91, 103, 107, 111, 115, 123], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 987, 'token_per_expert': {3: 64, 7: 64, 39: 137, 47: 209, 51: 32, 55: 105, 63: 45, 75: 29, 91: 66, 103: 88, 107: 37, 111: 28, 115: 29, 123: 54}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 24, 32, 48, 52, 60, 64, 68, 76, 104, 112, 116, 124], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 972, 'token_per_expert': {0: 79, 16: 47, 24: 30, 32: 53, 48: 48, 52: 69, 60: 39, 64: 45, 68: 157, 76: 53, 104: 41, 112: 45, 116: 88, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 21, 25, 33, 41, 53, 73, 89, 105, 113, 117, 121, 125], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 774, 'token_per_expert': {1: 89, 13: 18, 21: 22, 25: 18, 33: 155, 41: 22, 53: 172, 73: 29, 89: 17, 105: 56, 113: 37, 117: 23, 121: 96, 125: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 22, 26, 46, 50, 54, 58, 74, 78, 90, 110, 122, 126], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 864, 'token_per_expert': {18: 43, 22: 98, 26: 52, 46: 84, 50: 71, 54: 52, 58: 24, 74: 71, 78: 37, 90: 148, 110: 38, 122: 71, 126: 75}}
INFO 05-03 18:02:13.497383.497383 lmp.py:1005] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 49.206ms | allocate_experts_across_cpu_gpu: 0.273ms
INFO 05-03 18:02:13.497382.497382 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.363059997558594e-05 seconds
INFO 05-03 18:02:13.500002.500002 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0022194385528564453 seconds
INFO 05-03 18:02:13.559361.559361 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.05811619758605957 seconds
INFO 05-03 18:02:13.560197.560197 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012586116790771484 seconds
INFO 05-03 18:02:13.577041.577041 mlpmodule.py:2707] [fused_experts] gmm total=16.765ms E=32 S=1176 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.691797.691797 mlpmodule.py:2707] [fused_experts] gmm total=112.740ms E=32 S=1094 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.715984.715984 mlpmodule.py:2707] [fused_experts] gmm total=23.890ms E=32 S=869 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.730177.730177 mlpmodule.py:2707] [fused_experts] gmm total=14.688ms E=32 S=957 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.731023.731023 lmp.py:1149] [layer_moe_fused] experts compute time: 0.17035341262817383 seconds
INFO 05-03 18:02:13.731869.731869 lmp.py:1160] [layer_moe_fused] to time: 4.124641418457031e-05 seconds
INFO 05-03 18:02:13.731774.731774 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0003781318664550781 seconds
DEBUG 05-03 18:02:13.732995.732995 cuda_h.py:27] end *layer_moe_fused cost 283.944 ms
DEBUG 05-03 18:02:13.732958.732958 cuda_h.py:27] end prefill_layer cost 515.035 ms
DEBUG 05-03 18:02:13.732376.732376 lmp.py:711] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-03 18:02:13.732741.732741 lmp.py:675] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-03 18:02:13.736372.736372 cuda_h.py:27] end *sagl cost 3.797 ms
experts_cpu_alloc {'expert_ids': [43, 95, 107, 55, 63, 11, 27, 83, 39, 51, 3, 79, 111, 115, 48, 100, 24, 124, 40, 56, 36, 84, 116, 0, 112, 108, 25, 69, 17, 65, 29, 33, 117, 41, 61, 73, 81, 53, 77, 121, 93, 21, 89, 58, 102, 54, 2, 62, 74, 86, 66, 78, 26, 110, 118, 6, 14, 50, 70], 'token_total': 528, 'token_per_expert': {43: 1, 95: 1, 107: 1, 55: 2, 63: 3, 11: 4, 27: 4, 83: 4, 39: 5, 51: 5, 3: 7, 79: 7, 111: 10, 115: 11, 48: 1, 100: 1, 24: 4, 124: 5, 40: 7, 56: 7, 36: 9, 84: 10, 116: 10, 0: 12, 112: 14, 108: 16, 25: 2, 69: 2, 17: 3, 65: 3, 29: 4, 33: 5, 117: 5, 41: 6, 61: 6, 73: 8, 81: 9, 53: 13, 77: 14, 121: 17, 93: 20, 21: 21, 89: 21, 58: 1, 102: 1, 54: 2, 2: 7, 62: 9, 74: 11, 86: 12, 66: 13, 78: 16, 26: 17, 110: 17, 118: 18, 6: 19, 14: 21, 50: 22, 70: 22}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 23, 31, 35, 47, 59, 67, 71, 87, 99, 103, 119, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 694, 'token_per_expert': {7: 42, 15: 15, 23: 35, 31: 16, 35: 25, 47: 140, 59: 28, 67: 101, 71: 14, 87: 35, 99: 74, 103: 15, 119: 31, 123: 19, 127: 104}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 16, 20, 28, 52, 60, 64, 68, 72, 76, 80, 96, 104, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 803, 'token_per_expert': {4: 25, 8: 162, 16: 18, 20: 23, 28: 112, 52: 57, 60: 23, 64: 39, 68: 55, 72: 38, 76: 19, 80: 133, 96: 41, 104: 37, 120: 21}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 37, 45, 49, 57, 85, 97, 101, 105, 109, 113, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 863, 'token_per_expert': {1: 88, 5: 214, 9: 48, 13: 54, 37: 28, 45: 48, 49: 29, 57: 40, 85: 50, 97: 72, 101: 34, 105: 22, 109: 57, 113: 31, 125: 48}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 18, 22, 30, 34, 38, 42, 46, 82, 90, 94, 98, 106, 114, 122], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1208, 'token_per_expert': {10: 183, 18: 48, 22: 100, 30: 87, 34: 31, 38: 59, 42: 63, 46: 92, 82: 133, 90: 113, 94: 53, 98: 41, 106: 46, 114: 32, 122: 127}}
INFO 05-03 18:02:13.737559.737559 lmp.py:1005] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.495ms | allocate_experts_across_cpu_gpu: 0.265ms
INFO 05-03 18:02:13.737754.737754 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.100799560546875e-05 seconds
INFO 05-03 18:02:13.739891.739891 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018341541290283203 seconds
INFO 05-03 18:02:13.753506.753506 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014067649841308594 seconds
INFO 05-03 18:02:13.755171.755171 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013031959533691406 seconds
INFO 05-03 18:02:13.759029.759029 mlpmodule.py:2707] [fused_experts] gmm total=3.734ms E=32 S=759 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.762613.762613 mlpmodule.py:2707] [fused_experts] gmm total=3.446ms E=32 S=899 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.765514.765514 mlpmodule.py:2707] [fused_experts] gmm total=1.979ms E=32 S=1022 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.767732.767732 mlpmodule.py:2707] [fused_experts] gmm total=2.147ms E=32 S=1416 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.769704.769704 lmp.py:1149] [layer_moe_fused] experts compute time: 0.014029264450073242 seconds
INFO 05-03 18:02:13.769595.769595 lmp.py:1160] [layer_moe_fused] to time: 3.361701965332031e-05 seconds
INFO 05-03 18:02:13.769690.769690 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002930164337158203 seconds
DEBUG 05-03 18:02:13.770871.770871 cuda_h.py:27] end *layer_moe_fused cost 33.821 ms
DEBUG 05-03 18:02:13.770337.770337 cuda_h.py:27] end prefill_layer cost 37.972 ms
DEBUG 05-03 18:02:13.770437.770437 lmp.py:711] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-03 18:02:13.770727.770727 lmp.py:675] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-03 18:02:13.774090.774090 cuda_h.py:27] end *sagl cost 3.737 ms
experts_cpu_alloc {'expert_ids': [47, 87, 95, 59, 15, 27, 75, 39, 63, 83, 111, 67, 79, 55, 115, 103, 3, 40, 88, 4, 112, 32, 76, 100, 0, 12, 36, 72, 116, 64, 56, 60, 5, 117, 101, 49, 85, 37, 93, 45, 89, 121, 21, 33, 77, 73, 2, 38, 50, 86, 98, 6, 46, 58, 90, 94, 10, 14, 26, 66], 'token_total': 427, 'token_per_expert': {47: 1, 87: 1, 95: 1, 59: 2, 15: 4, 27: 4, 75: 5, 39: 7, 63: 14, 83: 16, 111: 16, 67: 17, 79: 18, 55: 23, 115: 23, 103: 25, 3: 26, 40: 1, 88: 1, 4: 2, 112: 2, 32: 4, 76: 5, 100: 5, 0: 6, 12: 6, 36: 6, 72: 7, 116: 10, 64: 11, 56: 15, 60: 15, 5: 1, 117: 1, 101: 2, 49: 3, 85: 3, 37: 4, 93: 5, 45: 6, 89: 6, 121: 7, 21: 9, 33: 11, 77: 13, 73: 15, 2: 1, 38: 1, 50: 1, 86: 1, 98: 1, 6: 3, 46: 3, 58: 3, 90: 3, 94: 3, 10: 4, 14: 5, 26: 6, 66: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 19, 23, 31, 35, 43, 51, 71, 91, 99, 107, 119, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1077, 'token_per_expert': {7: 63, 11: 123, 19: 120, 23: 53, 31: 43, 35: 50, 43: 27, 51: 125, 71: 36, 91: 87, 99: 57, 107: 76, 119: 28, 123: 103, 127: 86}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 16, 20, 24, 28, 44, 48, 52, 68, 80, 84, 96, 104, 108, 124], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1003, 'token_per_expert': {8: 16, 16: 31, 20: 16, 24: 55, 28: 17, 44: 25, 48: 153, 52: 25, 68: 36, 80: 65, 84: 69, 96: 33, 104: 106, 108: 307, 124: 49}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 25, 41, 53, 57, 61, 65, 69, 81, 97, 105, 109, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 821, 'token_per_expert': {1: 85, 9: 120, 13: 47, 25: 33, 41: 156, 53: 31, 57: 33, 61: 25, 65: 88, 69: 19, 81: 24, 97: 16, 105: 63, 109: 37, 125: 44}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 30, 42, 54, 62, 70, 74, 78, 82, 102, 106, 110, 118, 122, 126], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 768, 'token_per_expert': {18: 65, 30: 23, 42: 26, 54: 184, 62: 60, 70: 76, 74: 29, 78: 23, 82: 34, 102: 10, 106: 7, 110: 42, 118: 37, 122: 85, 126: 67}}
INFO 05-03 18:02:13.775226.775226 lmp.py:1005] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.552ms | allocate_experts_across_cpu_gpu: 0.442ms
INFO 05-03 18:02:13.775006.775006 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.437301635742188e-05 seconds
INFO 05-03 18:02:13.776892.776892 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007064342498779297 seconds
INFO 05-03 18:02:13.790783.790783 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012868642807006836 seconds
INFO 05-03 18:02:13.791731.791731 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010783672332763672 seconds
INFO 05-03 18:02:13.793772.793772 mlpmodule.py:2707] [fused_experts] gmm total=2.296ms E=32 S=1280 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.795817.795817 mlpmodule.py:2707] [fused_experts] gmm total=1.971ms E=32 S=1099 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.798203.798203 mlpmodule.py:2707] [fused_experts] gmm total=1.846ms E=32 S=907 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.800086.800086 mlpmodule.py:2707] [fused_experts] gmm total=1.839ms E=32 S=810 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.800867.800867 lmp.py:1149] [layer_moe_fused] experts compute time: 0.009490728378295898 seconds
INFO 05-03 18:02:13.800679.800679 lmp.py:1160] [layer_moe_fused] to time: 3.719329833984375e-05 seconds
INFO 05-03 18:02:13.801971.801971 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002696514129638672 seconds
DEBUG 05-03 18:02:13.801292.801292 cuda_h.py:27] end *layer_moe_fused cost 27.098 ms
DEBUG 05-03 18:02:13.801519.801519 cuda_h.py:27] end prefill_layer cost 31.166 ms
DEBUG 05-03 18:02:13.801335.801335 lmp.py:711] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-03 18:02:13.801432.801432 lmp.py:675] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-03 18:02:13.805730.805730 cuda_h.py:27] end *sagl cost 3.357 ms
experts_cpu_alloc {'expert_ids': [63, 23, 99, 103, 19, 55, 59, 79, 87, 15, 67, 115, 43, 16, 92, 12, 28, 24, 80, 48, 44, 84, 124, 36, 8, 108, 52, 116, 29, 33, 109, 17, 45, 49, 89, 105, 125, 53, 21, 41, 13, 65, 61, 18, 90, 30, 38, 98, 106, 62, 58, 82, 46, 94, 118, 114, 102, 126, 110, 14], 'token_total': 372, 'token_per_expert': {63: 1, 23: 2, 99: 2, 103: 2, 19: 5, 55: 6, 59: 7, 79: 7, 87: 7, 15: 9, 67: 10, 115: 12, 43: 15, 16: 1, 92: 1, 12: 2, 28: 3, 24: 4, 80: 4, 48: 5, 44: 6, 84: 6, 124: 6, 36: 8, 8: 9, 108: 9, 52: 10, 116: 11, 29: 1, 33: 1, 109: 1, 17: 2, 45: 2, 49: 2, 89: 2, 105: 2, 125: 2, 53: 3, 21: 6, 41: 9, 13: 13, 65: 13, 61: 14, 18: 1, 90: 1, 30: 2, 38: 2, 98: 2, 106: 4, 62: 5, 58: 6, 82: 6, 46: 9, 94: 9, 118: 9, 114: 12, 102: 14, 126: 14, 110: 15, 14: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 31, 39, 51, 71, 75, 83, 95, 107, 111, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 892, 'token_per_expert': {3: 117, 7: 89, 11: 87, 27: 62, 31: 30, 39: 65, 51: 18, 71: 95, 75: 31, 83: 38, 95: 77, 107: 75, 111: 55, 123: 22, 127: 31}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 32, 40, 56, 60, 64, 68, 72, 76, 88, 96, 100, 104, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 689, 'token_per_expert': {0: 117, 4: 108, 32: 33, 40: 69, 56: 28, 60: 14, 64: 43, 68: 41, 72: 30, 76: 41, 88: 44, 96: 31, 100: 17, 104: 50, 120: 23}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 37, 69, 73, 77, 81, 85, 93, 97, 101, 113, 117, 121], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 924, 'token_per_expert': {1: 88, 5: 124, 9: 98, 37: 25, 69: 80, 73: 42, 77: 34, 81: 14, 85: 46, 93: 83, 97: 73, 101: 99, 113: 25, 117: 66, 121: 27}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 26, 34, 42, 50, 54, 66, 70, 74, 78, 86, 122], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1219, 'token_per_expert': {2: 97, 6: 94, 10: 105, 22: 87, 26: 55, 34: 48, 42: 23, 50: 125, 54: 88, 66: 39, 70: 49, 74: 60, 78: 172, 86: 93, 122: 84}}
INFO 05-03 18:02:13.806582.806582 lmp.py:1005] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.585ms | allocate_experts_across_cpu_gpu: 0.432ms
INFO 05-03 18:02:13.806316.806316 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.628036499023438e-05 seconds
INFO 05-03 18:02:13.807054.807054 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006427764892578125 seconds
INFO 05-03 18:02:13.821120.821120 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01372671127319336 seconds
INFO 05-03 18:02:13.823636.823636 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001043558120727539 seconds
INFO 05-03 18:02:13.825352.825352 mlpmodule.py:2707] [fused_experts] gmm total=2.060ms E=32 S=977 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.827472.827472 mlpmodule.py:2707] [fused_experts] gmm total=1.688ms E=32 S=774 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.829062.829062 mlpmodule.py:2707] [fused_experts] gmm total=1.844ms E=32 S=997 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.831313.831313 mlpmodule.py:2707] [fused_experts] gmm total=1.969ms E=32 S=1348 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.832273.832273 lmp.py:1149] [layer_moe_fused] experts compute time: 0.009055852890014648 seconds
INFO 05-03 18:02:13.832795.832795 lmp.py:1160] [layer_moe_fused] to time: 3.0517578125e-05 seconds
INFO 05-03 18:02:13.832166.832166 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00024628639221191406 seconds
DEBUG 05-03 18:02:13.832619.832619 cuda_h.py:27] end *layer_moe_fused cost 27.453 ms
DEBUG 05-03 18:02:13.832085.832085 cuda_h.py:27] end prefill_layer cost 31.169 ms
DEBUG 05-03 18:02:13.832947.832947 lmp.py:711] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-03 18:02:13.833906.833906 lmp.py:675] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-03 18:02:13.837696.837696 cuda_h.py:27] end *sagl cost 3.844 ms
experts_cpu_alloc {'expert_ids': [127, 99, 119, 95, 75, 35, 123, 11, 103, 71, 87, 107, 15, 19, 111, 68, 100, 72, 116, 80, 12, 48, 76, 124, 28, 120, 24, 16, 56, 40, 33, 9, 45, 109, 117, 69, 77, 17, 37, 61, 121, 125, 13, 97, 93, 30, 70, 74, 10, 14, 38, 42, 62, 122, 90, 94, 58, 110, 114], 'token_total': 318, 'token_per_expert': {127: 1, 99: 2, 119: 3, 95: 4, 75: 5, 35: 6, 123: 6, 11: 7, 103: 7, 71: 8, 87: 8, 107: 9, 15: 15, 19: 17, 111: 18, 68: 1, 100: 1, 72: 2, 116: 3, 80: 4, 12: 5, 48: 5, 76: 5, 124: 5, 28: 6, 120: 6, 24: 7, 16: 8, 56: 13, 40: 16, 33: 1, 9: 2, 45: 2, 109: 2, 117: 2, 69: 3, 77: 3, 17: 4, 37: 4, 61: 4, 121: 6, 125: 6, 13: 8, 97: 10, 93: 12, 30: 1, 70: 1, 74: 1, 10: 2, 14: 2, 38: 2, 42: 2, 62: 2, 122: 2, 90: 3, 94: 5, 58: 6, 110: 7, 114: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 31, 39, 43, 47, 51, 59, 63, 67, 79, 83, 91, 115], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1125, 'token_per_expert': {3: 111, 7: 104, 27: 20, 31: 21, 39: 51, 43: 99, 47: 61, 51: 195, 59: 22, 63: 25, 67: 72, 79: 26, 83: 85, 91: 71, 115: 162}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 32, 36, 44, 52, 60, 64, 84, 92, 96, 104, 108], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 992, 'token_per_expert': {0: 116, 4: 160, 8: 21, 20: 149, 32: 49, 36: 56, 44: 56, 52: 46, 60: 37, 64: 48, 84: 17, 92: 50, 96: 49, 104: 115, 108: 23}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25, 29, 49, 53, 57, 65, 73, 81, 85, 89, 101, 105], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 660, 'token_per_expert': {1: 104, 5: 180, 21: 34, 25: 13, 29: 41, 49: 52, 53: 33, 57: 23, 65: 17, 73: 19, 81: 31, 85: 45, 89: 15, 101: 15, 105: 38}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 34, 46, 50, 54, 66, 78, 82, 86, 98, 106, 118], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1001, 'token_per_expert': {2: 92, 6: 132, 22: 65, 26: 61, 34: 27, 46: 53, 50: 87, 54: 35, 66: 70, 78: 65, 82: 34, 86: 24, 98: 43, 106: 78, 118: 135}}
INFO 05-03 18:02:13.838414.838414 lmp.py:1005] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.537ms | allocate_experts_across_cpu_gpu: 0.430ms
INFO 05-03 18:02:13.838485.838485 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.651878356933594e-05 seconds
INFO 05-03 18:02:13.839766.839766 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006225109100341797 seconds
INFO 05-03 18:02:13.852416.852416 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012456893920898438 seconds
INFO 05-03 18:02:13.853565.853565 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009560585021972656 seconds
INFO 05-03 18:02:13.855546.855546 mlpmodule.py:2707] [fused_experts] gmm total=2.083ms E=32 S=1241 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.857980.857980 mlpmodule.py:2707] [fused_experts] gmm total=1.911ms E=32 S=1079 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.859572.859572 mlpmodule.py:2707] [fused_experts] gmm total=1.655ms E=32 S=729 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.861312.861312 mlpmodule.py:2707] [fused_experts] gmm total=1.794ms E=32 S=1047 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.862298.862298 lmp.py:1149] [layer_moe_fused] experts compute time: 0.008952140808105469 seconds
INFO 05-03 18:02:13.862535.862535 lmp.py:1160] [layer_moe_fused] to time: 3.695487976074219e-05 seconds
INFO 05-03 18:02:13.862449.862449 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00026917457580566406 seconds
DEBUG 05-03 18:02:13.863075.863075 cuda_h.py:27] end *layer_moe_fused cost 25.859 ms
DEBUG 05-03 18:02:13.863494.863494 cuda_h.py:27] end prefill_layer cost 30.033 ms
DEBUG 05-03 18:02:13.863595.863595 lmp.py:711] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-03 18:02:13.863308.863308 lmp.py:675] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-03 18:02:13.908178.908178 cuda_h.py:27] end *sagl cost 45.343 ms
experts_cpu_alloc {'expert_ids': [35, 11, 15, 83, 95, 91, 51, 67, 103, 47, 43, 79, 27, 119, 31, 19, 60, 84, 92, 100, 124, 116, 40, 8, 108, 36, 48, 12, 52, 16, 80, 76, 28, 53, 89, 93, 121, 45, 109, 17, 29, 125, 81, 85, 65, 38, 42, 66, 70, 106, 14, 26, 86, 46, 62, 98, 126, 54, 18, 118, 90], 'token_total': 288, 'token_per_expert': {35: 1, 11: 2, 15: 2, 83: 2, 95: 2, 91: 4, 51: 5, 67: 5, 103: 5, 47: 6, 43: 7, 79: 7, 27: 9, 119: 9, 31: 10, 19: 13, 60: 1, 84: 1, 92: 1, 100: 1, 124: 1, 116: 3, 40: 4, 8: 5, 108: 5, 36: 6, 48: 8, 12: 9, 52: 12, 16: 13, 80: 13, 76: 14, 28: 15, 53: 1, 89: 1, 93: 1, 121: 1, 45: 2, 109: 2, 17: 3, 29: 4, 125: 5, 81: 7, 85: 7, 65: 8, 38: 1, 42: 1, 66: 1, 70: 1, 106: 1, 14: 2, 26: 2, 86: 2, 46: 3, 62: 3, 98: 3, 126: 3, 54: 4, 18: 5, 118: 6, 90: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 55, 59, 63, 71, 75, 87, 99, 107, 111, 115, 123, 127], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 1063, 'token_per_expert': {3: 143, 7: 133, 23: 32, 39: 136, 55: 14, 59: 17, 63: 21, 71: 243, 75: 23, 87: 31, 99: 103, 107: 31, 111: 41, 115: 16, 123: 49, 127: 30}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 32, 44, 56, 64, 68, 72, 88, 96, 104, 112, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 760, 'token_per_expert': {0: 132, 4: 140, 20: 132, 24: 36, 32: 16, 44: 26, 56: 26, 64: 57, 68: 32, 72: 29, 88: 20, 96: 31, 104: 15, 112: 16, 120: 52}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 33, 37, 49, 57, 61, 73, 77, 97, 101, 113, 117], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1152, 'token_per_expert': {1: 240, 5: 134, 13: 95, 21: 19, 33: 49, 37: 9, 49: 105, 57: 28, 61: 114, 73: 33, 77: 16, 97: 39, 101: 140, 113: 24, 117: 107}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 30, 34, 50, 58, 74, 82, 94, 102, 110, 114, 122], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 833, 'token_per_expert': {2: 193, 6: 180, 10: 21, 22: 85, 30: 10, 34: 11, 50: 61, 58: 31, 74: 98, 82: 12, 94: 57, 102: 28, 110: 13, 114: 11, 122: 22}}
INFO 05-03 18:02:13.909498.909498 lmp.py:1005] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.481ms | allocate_experts_across_cpu_gpu: 0.270ms
INFO 05-03 18:02:13.909469.909469 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.1484832763671875e-05 seconds
INFO 05-03 18:02:13.911183.911183 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012650489807128906 seconds
INFO 05-03 18:02:13.923007.923007 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011844158172607422 seconds
INFO 05-03 18:02:13.925663.925663 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010402202606201172 seconds
INFO 05-03 18:02:13.927634.927634 mlpmodule.py:2707] [fused_experts] gmm total=2.172ms E=32 S=1152 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.929042.929042 mlpmodule.py:2707] [fused_experts] gmm total=1.546ms E=32 S=872 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.931590.931590 mlpmodule.py:2707] [fused_experts] gmm total=1.796ms E=32 S=1194 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.933932.933932 mlpmodule.py:2707] [fused_experts] gmm total=1.698ms E=32 S=878 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.933904.933904 lmp.py:1149] [layer_moe_fused] experts compute time: 0.008694887161254883 seconds
INFO 05-03 18:02:13.933544.933544 lmp.py:1160] [layer_moe_fused] to time: 3.1948089599609375e-05 seconds
INFO 05-03 18:02:13.934729.934729 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002532005310058594 seconds
DEBUG 05-03 18:02:13.934237.934237 cuda_h.py:27] end *layer_moe_fused cost 25.596 ms
DEBUG 05-03 18:02:13.934173.934173 cuda_h.py:27] end prefill_layer cost 71.318 ms
DEBUG 05-03 18:02:13.934088.934088 lmp.py:711] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-03 18:02:13.934266.934266 lmp.py:675] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-03 18:02:13.938191.938191 cuda_h.py:27] end *sagl cost 3.078 ms
experts_cpu_alloc {'expert_ids': [95, 27, 43, 47, 63, 107, 71, 23, 51, 67, 111, 115, 96, 116, 120, 92, 76, 80, 12, 36, 52, 28, 8, 100, 112, 9, 45, 85, 101, 105, 29, 81, 21, 121, 25, 37, 49, 17, 69, 109, 57, 117, 26, 50, 58, 114, 22, 70, 78, 18, 86, 66, 74, 98, 106, 62, 118], 'token_total': 328, 'token_per_expert': {95: 1, 27: 2, 43: 3, 47: 3, 63: 3, 107: 3, 71: 4, 23: 6, 51: 9, 67: 9, 111: 9, 115: 10, 96: 1, 116: 1, 120: 1, 92: 2, 76: 3, 80: 3, 12: 4, 36: 4, 52: 4, 28: 7, 8: 10, 100: 10, 112: 13, 9: 1, 45: 1, 85: 1, 101: 1, 105: 1, 29: 2, 81: 2, 21: 3, 121: 3, 25: 5, 37: 7, 49: 8, 17: 10, 69: 10, 109: 11, 57: 16, 117: 17, 26: 1, 50: 1, 58: 1, 114: 1, 22: 2, 70: 2, 78: 4, 18: 5, 86: 8, 66: 10, 74: 13, 98: 13, 106: 13, 62: 15, 118: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 35, 55, 75, 79, 83, 87, 91, 99, 119, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 672, 'token_per_expert': {3: 159, 7: 134, 19: 38, 31: 21, 35: 50, 55: 27, 75: 29, 79: 61, 83: 14, 87: 14, 91: 11, 99: 55, 119: 19, 123: 16, 127: 24}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 32, 40, 48, 56, 64, 68, 84, 104, 108, 124], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 870, 'token_per_expert': {0: 174, 4: 134, 16: 17, 20: 29, 32: 16, 40: 93, 48: 17, 56: 27, 64: 72, 68: 158, 84: 16, 104: 38, 108: 42, 124: 37}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 33, 41, 53, 65, 73, 77, 89, 93, 97, 113, 125], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1206, 'token_per_expert': {1: 132, 5: 133, 13: 25, 33: 35, 41: 52, 53: 104, 65: 30, 73: 47, 77: 159, 89: 34, 93: 188, 97: 115, 113: 111, 125: 41}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 34, 42, 46, 54, 82, 90, 94, 102, 110, 122, 126], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1020, 'token_per_expert': {2: 131, 6: 132, 10: 43, 34: 147, 42: 115, 46: 156, 54: 18, 82: 31, 90: 38, 94: 50, 102: 70, 110: 21, 122: 20, 126: 48}}
INFO 05-03 18:02:13.939126.939126 lmp.py:1005] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.530ms | allocate_experts_across_cpu_gpu: 0.421ms
INFO 05-03 18:02:13.939568.939568 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.365776062011719e-05 seconds
INFO 05-03 18:02:13.940931.940931 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006692409515380859 seconds
INFO 05-03 18:02:13.954625.954625 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014162063598632812 seconds
INFO 05-03 18:02:13.956493.956493 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010678768157958984 seconds
INFO 05-03 18:02:13.958582.958582 mlpmodule.py:2707] [fused_experts] gmm total=1.738ms E=32 S=734 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.960782.960782 mlpmodule.py:2707] [fused_experts] gmm total=1.738ms E=32 S=933 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.962983.962983 mlpmodule.py:2707] [fused_experts] gmm total=1.647ms E=32 S=1305 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.964650.964650 mlpmodule.py:2707] [fused_experts] gmm total=1.807ms E=32 S=1124 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.965927.965927 lmp.py:1149] [layer_moe_fused] experts compute time: 0.008730173110961914 seconds
INFO 05-03 18:02:13.965468.965468 lmp.py:1160] [layer_moe_fused] to time: 3.0517578125e-05 seconds
INFO 05-03 18:02:13.965099.965099 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002987384796142578 seconds
DEBUG 05-03 18:02:13.965058.965058 cuda_h.py:27] end *layer_moe_fused cost 27.590 ms
DEBUG 05-03 18:02:13.965477.965477 cuda_h.py:27] end prefill_layer cost 31.019 ms
DEBUG 05-03 18:02:13.965247.965247 lmp.py:711] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-03 18:02:13.966405.966405 lmp.py:675] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-03 18:02:13.970253.970253 cuda_h.py:27] end *sagl cost 4.138 ms
experts_cpu_alloc {'expert_ids': [23, 27, 99, 119, 15, 19, 71, 127, 111, 51, 75, 11, 55, 35, 47, 68, 76, 120, 12, 72, 64, 32, 36, 28, 108, 8, 100, 92, 41, 49, 53, 61, 81, 21, 93, 57, 73, 101, 9, 89, 22, 66, 78, 102, 58, 94, 118, 18, 42, 90, 14, 26, 38, 62, 34, 114], 'token_total': 201, 'token_per_expert': {23: 1, 27: 1, 99: 1, 119: 1, 15: 2, 19: 2, 71: 2, 127: 2, 111: 3, 51: 5, 75: 5, 11: 6, 55: 6, 35: 8, 47: 11, 68: 1, 76: 1, 120: 1, 12: 2, 72: 2, 64: 3, 32: 6, 36: 6, 28: 8, 108: 8, 8: 10, 100: 11, 92: 13, 41: 1, 49: 1, 53: 1, 61: 1, 81: 1, 21: 2, 93: 2, 57: 3, 73: 3, 101: 3, 9: 4, 89: 4, 22: 1, 66: 1, 78: 1, 102: 1, 58: 2, 94: 2, 118: 2, 18: 3, 42: 3, 90: 3, 14: 4, 26: 4, 38: 4, 62: 4, 34: 5, 114: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 43, 59, 63, 83, 87, 91, 95, 103, 107, 115, 123], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 795, 'token_per_expert': {3: 197, 7: 135, 31: 13, 39: 14, 43: 14, 59: 54, 63: 11, 83: 17, 87: 24, 91: 27, 95: 144, 103: 61, 107: 25, 115: 13, 123: 46}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 44, 48, 52, 56, 60, 84, 88, 96, 116, 124], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1531, 'token_per_expert': {0: 137, 4: 251, 20: 240, 24: 56, 44: 23, 48: 106, 52: 144, 56: 126, 60: 106, 84: 23, 88: 38, 96: 197, 116: 64, 124: 20}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 45, 65, 69, 77, 85, 105, 109, 113, 117, 121, 125], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 733, 'token_per_expert': {1: 137, 5: 139, 29: 7, 45: 7, 65: 20, 69: 128, 77: 7, 85: 23, 105: 31, 109: 83, 113: 7, 117: 11, 121: 99, 125: 34}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 30, 46, 50, 54, 70, 74, 98, 106, 110, 122, 126], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 836, 'token_per_expert': {2: 137, 6: 204, 10: 88, 30: 42, 46: 19, 50: 9, 54: 17, 70: 18, 74: 11, 98: 13, 106: 213, 110: 10, 122: 35, 126: 20}}
INFO 05-03 18:02:13.971223.971223 lmp.py:1005] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.581ms | allocate_experts_across_cpu_gpu: 0.417ms
INFO 05-03 18:02:13.971639.971639 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 7.390975952148438e-05 seconds
INFO 05-03 18:02:13.972542.972542 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006563663482666016 seconds
INFO 05-03 18:02:13.985900.985900 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0128021240234375 seconds
INFO 05-03 18:02:13.987851.987851 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009844303131103516 seconds
INFO 05-03 18:02:13.989637.989637 mlpmodule.py:2707] [fused_experts] gmm total=1.970ms E=32 S=851 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.991634.991634 mlpmodule.py:2707] [fused_experts] gmm total=1.776ms E=32 S=1603 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.994098.994098 mlpmodule.py:2707] [fused_experts] gmm total=1.646ms E=32 S=759 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.996714.996714 mlpmodule.py:2707] [fused_experts] gmm total=1.781ms E=32 S=883 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:13.997799.997799 lmp.py:1149] [layer_moe_fused] experts compute time: 0.010020017623901367 seconds
INFO 05-03 18:02:13.997897.997897 lmp.py:1160] [layer_moe_fused] to time: 3.790855407714844e-05 seconds
INFO 05-03 18:02:13.997221.997221 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00025844573974609375 seconds
DEBUG 05-03 18:02:13.997192.997192 cuda_h.py:27] end *layer_moe_fused cost 27.348 ms
DEBUG 05-03 18:02:13.997419.997419 cuda_h.py:27] end prefill_layer cost 31.857 ms
DEBUG 05-03 18:02:13.997281.997281 lmp.py:711] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-03 18:02:13.998809.998809 lmp.py:675] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-03 18:02:14.000568.000568 cuda_h.py:27] end *sagl cost 2.636 ms
experts_cpu_alloc {'expert_ids': [31, 35, 43, 67, 127, 83, 99, 111, 11, 107, 15, 59, 39, 87, 32, 36, 60, 88, 104, 112, 8, 20, 48, 76, 56, 40, 97, 105, 109, 113, 17, 101, 25, 65, 57, 9, 33, 41, 117, 69, 34, 62, 90, 102, 26, 70, 94, 106, 126, 10, 50, 74, 110, 66], 'token_total': 199, 'token_per_expert': {31: 1, 35: 1, 43: 1, 67: 1, 127: 1, 83: 2, 99: 2, 111: 2, 11: 3, 107: 3, 15: 4, 59: 4, 39: 6, 87: 7, 32: 1, 36: 1, 60: 1, 88: 1, 104: 1, 112: 1, 8: 2, 20: 3, 48: 3, 76: 3, 56: 4, 40: 5, 97: 1, 105: 1, 109: 1, 113: 1, 17: 4, 101: 4, 25: 5, 65: 7, 57: 8, 9: 10, 33: 11, 41: 11, 117: 11, 69: 12, 34: 1, 62: 1, 90: 1, 102: 1, 26: 2, 70: 2, 94: 3, 106: 3, 126: 3, 10: 4, 50: 5, 74: 5, 110: 7, 66: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 27, 51, 55, 63, 71, 75, 79, 91, 103, 119, 123], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 567, 'token_per_expert': {3: 134, 7: 131, 19: 26, 27: 16, 51: 30, 55: 12, 63: 8, 71: 28, 75: 11, 79: 22, 91: 19, 103: 58, 119: 7, 123: 65}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 24, 44, 64, 68, 80, 84, 92, 96, 120, 124], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 846, 'token_per_expert': {0: 158, 4: 133, 12: 8, 16: 45, 24: 6, 44: 39, 64: 13, 68: 47, 80: 8, 84: 119, 92: 129, 96: 8, 120: 10, 124: 123}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 29, 49, 73, 77, 81, 85, 89, 93, 121, 125], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1313, 'token_per_expert': {1: 142, 5: 131, 13: 105, 21: 41, 29: 38, 49: 27, 73: 154, 77: 154, 81: 78, 85: 39, 89: 106, 93: 22, 121: 145, 125: 131}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 30, 38, 46, 54, 58, 86, 98, 118], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1171, 'token_per_expert': {2: 131, 6: 427, 14: 12, 18: 51, 22: 73, 30: 109, 38: 40, 46: 84, 54: 20, 58: 166, 86: 16, 98: 21, 118: 21}}
INFO 05-03 18:02:14.001814.001814 lmp.py:1005] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.530ms | allocate_experts_across_cpu_gpu: 0.407ms
INFO 05-03 18:02:14.002309.002309 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.532669067382812e-05 seconds
INFO 05-03 18:02:14.003471.003471 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006201267242431641 seconds
INFO 05-03 18:02:14.016602.016602 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01260685920715332 seconds
INFO 05-03 18:02:14.017864.017864 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009770393371582031 seconds
INFO 05-03 18:02:14.019475.019475 mlpmodule.py:2707] [fused_experts] gmm total=1.706ms E=32 S=605 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.021985.021985 mlpmodule.py:2707] [fused_experts] gmm total=1.694ms E=32 S=872 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.023972.023972 mlpmodule.py:2707] [fused_experts] gmm total=1.783ms E=32 S=1400 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.025211.025211 mlpmodule.py:2707] [fused_experts] gmm total=1.793ms E=32 S=1219 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.025046.025046 lmp.py:1149] [layer_moe_fused] experts compute time: 0.00850224494934082 seconds
INFO 05-03 18:02:14.026871.026871 lmp.py:1160] [layer_moe_fused] to time: 3.0517578125e-05 seconds
INFO 05-03 18:02:14.026920.026920 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0003247261047363281 seconds
DEBUG 05-03 18:02:14.026712.026712 cuda_h.py:27] end *layer_moe_fused cost 25.583 ms
DEBUG 05-03 18:02:14.026701.026701 cuda_h.py:27] end prefill_layer cost 28.606 ms
DEBUG 05-03 18:02:14.026328.026328 lmp.py:711] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-03 18:02:14.027686.027686 lmp.py:675] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-03 18:02:14.030545.030545 cuda_h.py:27] end *sagl cost 3.674 ms
experts_cpu_alloc {'expert_ids': [119, 11, 35, 23, 27, 71, 75, 63, 83, 115, 96, 112, 44, 52, 104, 120, 40, 56, 92, 64, 72, 68, 84, 36, 73, 81, 113, 117, 41, 13, 49, 61, 33, 89, 21, 14, 82, 126, 38, 54, 110, 74, 18, 122, 66], 'token_total': 181, 'token_per_expert': {119: 1, 11: 2, 35: 2, 23: 3, 27: 3, 71: 3, 75: 5, 63: 8, 83: 9, 115: 12, 96: 1, 112: 1, 44: 2, 52: 2, 104: 2, 120: 2, 40: 4, 56: 5, 92: 5, 64: 6, 72: 7, 68: 9, 84: 9, 36: 10, 73: 1, 81: 1, 113: 1, 117: 1, 41: 2, 13: 3, 49: 3, 61: 5, 33: 6, 89: 6, 21: 7, 14: 1, 82: 1, 126: 1, 38: 2, 54: 2, 110: 2, 74: 4, 18: 6, 122: 6, 66: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 39, 43, 51, 79, 91, 95, 103, 111], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 951, 'token_per_expert': {3: 132, 7: 151, 15: 229, 19: 72, 39: 34, 43: 30, 51: 45, 79: 31, 91: 68, 95: 128, 103: 16, 111: 15}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 24, 28, 32, 48, 76, 88, 108, 124], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1100, 'token_per_expert': {0: 140, 4: 138, 12: 170, 16: 97, 24: 37, 28: 79, 32: 20, 48: 145, 76: 42, 88: 151, 108: 24, 124: 57}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 25, 29, 37, 53, 57, 65, 69, 97], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 944, 'token_per_expert': {1: 154, 5: 130, 9: 285, 25: 14, 29: 36, 37: 22, 53: 8, 57: 68, 65: 19, 69: 8, 97: 200}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 46, 70, 86, 98, 102, 106, 114], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 920, 'token_per_expert': {2: 174, 6: 150, 22: 187, 26: 20, 46: 14, 70: 8, 86: 14, 98: 157, 102: 13, 106: 118, 114: 65}}
INFO 05-03 18:02:14.031029.031029 lmp.py:1005] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.522ms | allocate_experts_across_cpu_gpu: 0.376ms
INFO 05-03 18:02:14.031457.031457 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.507469177246094e-05 seconds
INFO 05-03 18:02:14.032623.032623 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006198883056640625 seconds
INFO 05-03 18:02:14.045178.045178 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012647867202758789 seconds
INFO 05-03 18:02:14.046704.046704 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009393692016601562 seconds
INFO 05-03 18:02:14.048608.048608 mlpmodule.py:2707] [fused_experts] gmm total=1.787ms E=32 S=999 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.050983.050983 mlpmodule.py:2707] [fused_experts] gmm total=1.793ms E=32 S=1165 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.053688.053688 mlpmodule.py:2707] [fused_experts] gmm total=1.683ms E=32 S=980 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.055431.055431 mlpmodule.py:2707] [fused_experts] gmm total=1.616ms E=32 S=952 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.055477.055477 lmp.py:1149] [layer_moe_fused] experts compute time: 0.008387088775634766 seconds
INFO 05-03 18:02:14.055714.055714 lmp.py:1160] [layer_moe_fused] to time: 3.7670135498046875e-05 seconds
INFO 05-03 18:02:14.055066.055066 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00048828125 seconds
DEBUG 05-03 18:02:14.056733.056733 cuda_h.py:27] end *layer_moe_fused cost 25.508 ms
DEBUG 05-03 18:02:14.056676.056676 cuda_h.py:27] end prefill_layer cost 29.382 ms
DEBUG 05-03 18:02:14.056830.056830 lmp.py:711] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-03 18:02:14.056213.056213 lmp.py:675] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-03 18:02:14.059151.059151 cuda_h.py:27] end *sagl cost 2.893 ms
experts_cpu_alloc {'expert_ids': [123, 127, 47, 71, 95, 99, 107, 119, 19, 23, 55, 83, 31, 51, 16, 28, 40, 72, 96, 24, 52, 60, 84, 112, 116, 56, 92, 124, 49, 77, 97, 101, 65, 9, 41, 57, 93, 25, 85, 38, 42, 62, 78, 102, 114, 74, 58, 22, 86, 90, 98], 'token_total': 139, 'token_per_expert': {123: 1, 127: 1, 47: 2, 71: 3, 95: 3, 99: 3, 107: 3, 119: 3, 19: 4, 23: 4, 55: 4, 83: 4, 31: 6, 51: 7, 16: 1, 28: 1, 40: 1, 72: 1, 96: 1, 24: 2, 52: 2, 60: 2, 84: 2, 112: 2, 116: 3, 56: 4, 92: 5, 124: 8, 49: 1, 77: 1, 97: 1, 101: 1, 65: 2, 9: 3, 41: 3, 57: 3, 93: 3, 25: 4, 85: 4, 38: 1, 42: 1, 62: 1, 78: 1, 102: 1, 114: 1, 74: 2, 58: 3, 22: 4, 86: 4, 90: 5, 98: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 39, 43, 63, 67, 79, 87, 91, 103, 111, 115], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1091, 'token_per_expert': {3: 147, 7: 170, 11: 140, 39: 50, 43: 111, 63: 33, 67: 17, 79: 37, 87: 9, 91: 33, 103: 18, 111: 81, 115: 245}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 44, 48, 68, 80, 88, 100, 104, 108, 120], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 850, 'token_per_expert': {0: 129, 4: 130, 8: 11, 20: 49, 44: 33, 48: 23, 68: 144, 80: 48, 88: 11, 100: 44, 104: 21, 108: 188, 120: 19}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 33, 37, 53, 69, 81, 109, 113, 121, 125], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1259, 'token_per_expert': {1: 129, 5: 129, 13: 119, 29: 206, 33: 302, 37: 22, 53: 147, 69: 5, 81: 49, 109: 15, 113: 62, 121: 12, 125: 62}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 46, 66, 82, 94, 106, 110, 118], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 757, 'token_per_expert': {2: 129, 6: 188, 10: 7, 14: 15, 18: 67, 46: 19, 66: 26, 82: 99, 94: 170, 106: 20, 110: 10, 118: 7}}
INFO 05-03 18:02:14.060131.060131 lmp.py:1005] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.528ms | allocate_experts_across_cpu_gpu: 0.387ms
INFO 05-03 18:02:14.060097.060097 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.175041198730469e-05 seconds
INFO 05-03 18:02:14.061364.061364 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006039142608642578 seconds
INFO 05-03 18:02:14.075605.075605 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012909412384033203 seconds
INFO 05-03 18:02:14.076982.076982 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010616779327392578 seconds
INFO 05-03 18:02:14.078332.078332 mlpmodule.py:2707] [fused_experts] gmm total=1.828ms E=32 S=1139 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.080360.080360 mlpmodule.py:2707] [fused_experts] gmm total=1.492ms E=32 S=885 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.082889.082889 mlpmodule.py:2707] [fused_experts] gmm total=1.647ms E=32 S=1285 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.083484.083484 mlpmodule.py:2707] [fused_experts] gmm total=1.342ms E=32 S=787 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.084958.084958 lmp.py:1149] [layer_moe_fused] experts compute time: 0.0076906681060791016 seconds
INFO 05-03 18:02:14.084545.084545 lmp.py:1160] [layer_moe_fused] to time: 2.9802322387695312e-05 seconds
INFO 05-03 18:02:14.084591.084591 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002532005310058594 seconds
DEBUG 05-03 18:02:14.084490.084490 cuda_h.py:27] end *layer_moe_fused cost 25.095 ms
DEBUG 05-03 18:02:14.084910.084910 cuda_h.py:27] end prefill_layer cost 28.323 ms
DEBUG 05-03 18:02:14.084772.084772 lmp.py:711] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-03 18:02:14.085566.085566 lmp.py:675] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-03 18:02:14.090583.090583 cuda_h.py:27] end *sagl cost 5.231 ms
experts_cpu_alloc {'expert_ids': [11, 19, 35, 39, 51, 71, 83, 99, 123, 75, 111, 103, 115, 59, 72, 60, 68, 100, 108, 116, 84, 12, 56, 92, 24, 16, 124, 104, 20, 112, 21, 37, 53, 73, 97, 109, 33, 45, 25, 14, 30, 66, 70, 106, 122, 38, 10, 110, 126], 'token_total': 193, 'token_per_expert': {11: 1, 19: 1, 35: 1, 39: 1, 51: 1, 71: 1, 83: 1, 99: 1, 123: 1, 75: 2, 111: 2, 103: 3, 115: 3, 59: 4, 72: 1, 60: 2, 68: 2, 100: 2, 108: 2, 116: 2, 84: 3, 12: 4, 56: 4, 92: 4, 24: 7, 16: 12, 124: 14, 104: 22, 20: 25, 112: 33, 21: 1, 37: 1, 53: 1, 73: 1, 97: 1, 109: 1, 33: 2, 45: 2, 25: 4, 14: 1, 30: 1, 66: 1, 70: 1, 106: 1, 122: 1, 38: 2, 10: 3, 110: 3, 126: 3}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 31, 43, 47, 63, 79, 87, 91, 119, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1184, 'token_per_expert': {3: 364, 7: 138, 23: 154, 27: 12, 31: 10, 43: 115, 47: 6, 63: 45, 79: 6, 87: 5, 91: 128, 119: 182, 127: 19}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 32, 36, 40, 44, 48, 64, 76, 88, 120], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1407, 'token_per_expert': {0: 150, 4: 132, 28: 114, 32: 71, 36: 75, 40: 43, 44: 200, 48: 137, 64: 57, 76: 209, 88: 47, 120: 172}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 49, 61, 65, 69, 77, 81, 93, 105, 117], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 587, 'token_per_expert': {1: 158, 5: 130, 17: 122, 49: 25, 61: 4, 65: 8, 69: 5, 77: 9, 81: 23, 93: 47, 105: 9, 117: 47}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 42, 46, 54, 62, 74, 98, 102, 118], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 725, 'token_per_expert': {2: 130, 6: 130, 18: 25, 22: 36, 42: 10, 46: 7, 54: 83, 62: 198, 74: 57, 98: 11, 102: 20, 118: 18}}
INFO 05-03 18:02:14.091836.091836 lmp.py:1005] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.545ms | allocate_experts_across_cpu_gpu: 0.391ms
INFO 05-03 18:02:14.091854.091854 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.604194641113281e-05 seconds
INFO 05-03 18:02:14.092915.092915 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000736236572265625 seconds
INFO 05-03 18:02:14.106224.106224 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013196945190429688 seconds
INFO 05-03 18:02:14.107594.107594 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010199546813964844 seconds
INFO 05-03 18:02:14.109653.109653 mlpmodule.py:2707] [fused_experts] gmm total=1.855ms E=32 S=1207 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.111033.111033 mlpmodule.py:2707] [fused_experts] gmm total=1.535ms E=32 S=1546 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.113926.113926 mlpmodule.py:2707] [fused_experts] gmm total=1.377ms E=32 S=601 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.115478.115478 mlpmodule.py:2707] [fused_experts] gmm total=1.666ms E=32 S=742 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.115138.115138 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007861614227294922 seconds
INFO 05-03 18:02:14.115023.115023 lmp.py:1160] [layer_moe_fused] to time: 3.409385681152344e-05 seconds
INFO 05-03 18:02:14.116814.116814 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00031304359436035156 seconds
DEBUG 05-03 18:02:14.116369.116369 cuda_h.py:27] end *layer_moe_fused cost 25.770 ms
DEBUG 05-03 18:02:14.116120.116120 cuda_h.py:27] end prefill_layer cost 31.377 ms
DEBUG 05-03 18:02:14.116074.116074 lmp.py:711] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-03 18:02:14.116233.116233 lmp.py:675] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-03 18:02:14.119207.119207 cuda_h.py:27] end *sagl cost 2.945 ms
experts_cpu_alloc {'expert_ids': [55, 63, 75, 83, 91, 11, 43, 47, 59, 39, 95, 103, 127, 99, 71, 8, 28, 32, 52, 60, 64, 76, 100, 104, 120, 20, 36, 25, 73, 89, 97, 49, 81, 33, 37, 93, 109, 30, 54, 66, 70, 46, 50, 118, 122, 38, 58, 78, 94, 82, 106], 'token_total': 129, 'token_per_expert': {55: 1, 63: 1, 75: 1, 83: 1, 91: 1, 11: 2, 43: 2, 47: 2, 59: 2, 39: 3, 95: 3, 103: 3, 127: 3, 99: 4, 71: 8, 8: 1, 28: 1, 32: 1, 52: 1, 60: 1, 64: 1, 76: 1, 100: 1, 104: 1, 120: 1, 20: 2, 36: 2, 25: 1, 73: 1, 89: 1, 97: 1, 49: 2, 81: 2, 33: 3, 37: 3, 93: 3, 109: 3, 30: 1, 54: 1, 66: 1, 70: 1, 46: 2, 50: 2, 118: 2, 122: 2, 38: 3, 58: 5, 78: 8, 94: 8, 82: 11, 106: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 67, 107, 111, 119, 123], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1476, 'token_per_expert': {3: 268, 7: 129, 15: 18, 19: 48, 23: 97, 27: 10, 31: 209, 35: 51, 67: 14, 107: 239, 111: 17, 119: 253, 123: 123}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 40, 44, 80, 84, 88, 92, 108, 112, 116, 124], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 639, 'token_per_expert': {0: 129, 4: 129, 12: 4, 40: 142, 44: 14, 80: 5, 84: 35, 88: 92, 92: 19, 108: 21, 112: 2, 116: 10, 124: 37}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 21, 41, 45, 53, 85, 101, 105, 117, 125], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 906, 'token_per_expert': {1: 136, 5: 142, 13: 64, 17: 7, 21: 5, 41: 24, 45: 286, 53: 11, 85: 22, 101: 78, 105: 102, 117: 17, 125: 12}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 34, 74, 86, 90, 98, 102, 110], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 946, 'token_per_expert': {2: 129, 6: 130, 14: 28, 18: 16, 22: 28, 34: 146, 74: 223, 86: 37, 90: 52, 98: 17, 102: 29, 110: 111}}
INFO 05-03 18:02:14.120922.120922 lmp.py:1005] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.514ms | allocate_experts_across_cpu_gpu: 0.384ms
INFO 05-03 18:02:14.121172.121172 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.198883056640625e-05 seconds
INFO 05-03 18:02:14.121272.121272 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006079673767089844 seconds
INFO 05-03 18:02:14.128386.128386 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0061528682708740234 seconds
INFO 05-03 18:02:14.129568.129568 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009596347808837891 seconds
INFO 05-03 18:02:14.131465.131465 mlpmodule.py:2707] [fused_experts] gmm total=1.954ms E=32 S=1513 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.133381.133381 mlpmodule.py:2707] [fused_experts] gmm total=1.344ms E=32 S=653 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.135772.135772 mlpmodule.py:2707] [fused_experts] gmm total=1.486ms E=32 S=926 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.137391.137391 mlpmodule.py:2707] [fused_experts] gmm total=1.686ms E=32 S=1004 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.137285.137285 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007888555526733398 seconds
INFO 05-03 18:02:14.137269.137269 lmp.py:1160] [layer_moe_fused] to time: 3.0279159545898438e-05 seconds
INFO 05-03 18:02:14.138872.138872 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.000240325927734375 seconds
DEBUG 05-03 18:02:14.138777.138777 cuda_h.py:27] end *layer_moe_fused cost 18.314 ms
DEBUG 05-03 18:02:14.138760.138760 cuda_h.py:27] end prefill_layer cost 21.666 ms
DEBUG 05-03 18:02:14.138006.138006 lmp.py:711] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-03 18:02:14.138841.138841 lmp.py:675] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-03 18:02:14.143591.143591 cuda_h.py:27] end *sagl cost 4.619 ms
experts_cpu_alloc {'expert_ids': [11, 27, 51, 91, 95, 19, 43, 107, 111, 15, 35, 87, 16, 36, 48, 52, 96, 108, 116, 8, 32, 80, 76, 120, 9, 65, 125, 41, 53, 101, 13, 29, 61, 37, 18, 26, 62, 90, 106, 114, 74, 10, 50, 66, 126, 14, 82, 118, 22, 54, 58, 46], 'token_total': 197, 'token_per_expert': {11: 1, 27: 1, 51: 1, 91: 1, 95: 1, 19: 2, 43: 2, 107: 2, 111: 2, 15: 7, 35: 8, 87: 10, 16: 1, 36: 1, 48: 1, 52: 1, 96: 1, 108: 1, 116: 1, 8: 5, 32: 7, 80: 8, 76: 9, 120: 9, 9: 1, 65: 1, 125: 1, 41: 2, 53: 2, 101: 2, 13: 4, 29: 7, 61: 7, 37: 8, 18: 1, 26: 1, 62: 2, 90: 2, 106: 2, 114: 2, 74: 3, 10: 4, 50: 4, 66: 4, 126: 4, 14: 6, 82: 6, 118: 6, 22: 7, 54: 7, 58: 8, 46: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 55, 59, 67, 75, 79, 99, 115, 119], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 974, 'token_per_expert': {3: 130, 7: 148, 31: 39, 39: 128, 47: 47, 55: 29, 59: 18, 67: 115, 75: 12, 79: 25, 99: 48, 115: 193, 119: 42}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 28, 40, 56, 60, 64, 84, 88, 100, 104], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1015, 'token_per_expert': {0: 135, 4: 129, 20: 75, 24: 12, 28: 12, 40: 207, 56: 23, 60: 253, 64: 113, 84: 17, 88: 12, 100: 11, 104: 16}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25, 33, 45, 73, 81, 89, 93, 105, 109, 117], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 922, 'token_per_expert': {1: 137, 5: 129, 21: 11, 25: 11, 33: 46, 45: 18, 73: 208, 81: 69, 89: 146, 93: 18, 105: 41, 109: 8, 117: 80}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34, 38, 42, 70, 78, 86, 94, 98, 102, 110, 122], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 988, 'token_per_expert': {2: 184, 6: 138, 34: 170, 38: 35, 42: 25, 70: 89, 78: 92, 86: 21, 94: 10, 98: 35, 102: 13, 110: 56, 122: 120}}
INFO 05-03 18:02:14.144299.144299 lmp.py:1005] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.506ms | allocate_experts_across_cpu_gpu: 0.383ms
INFO 05-03 18:02:14.144019.144019 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.7220458984375e-05 seconds
INFO 05-03 18:02:14.156187.156187 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.011345624923706055 seconds
INFO 05-03 18:02:14.162758.162758 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.005816221237182617 seconds
INFO 05-03 18:02:14.163465.163465 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000982522964477539 seconds
INFO 05-03 18:02:14.165518.165518 mlpmodule.py:2707] [fused_experts] gmm total=1.477ms E=32 S=1012 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.166847.166847 mlpmodule.py:2707] [fused_experts] gmm total=1.449ms E=32 S=1060 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.168197.168197 mlpmodule.py:2707] [fused_experts] gmm total=1.415ms E=32 S=957 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.170602.170602 mlpmodule.py:2707] [fused_experts] gmm total=1.456ms E=32 S=1067 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.170162.170162 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007104396820068359 seconds
INFO 05-03 18:02:14.170272.170272 lmp.py:1160] [layer_moe_fused] to time: 2.9802322387695312e-05 seconds
INFO 05-03 18:02:14.171590.171590 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00024175643920898438 seconds
DEBUG 05-03 18:02:14.171839.171839 cuda_h.py:27] end *layer_moe_fused cost 27.904 ms
DEBUG 05-03 18:02:14.171398.171398 cuda_h.py:27] end prefill_layer cost 32.854 ms
DEBUG 05-03 18:02:14.171545.171545 lmp.py:711] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-03 18:02:14.171019.171019 lmp.py:675] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-03 18:02:14.174917.174917 cuda_h.py:27] end *sagl cost 2.279 ms
experts_cpu_alloc {'expert_ids': [23, 27, 43, 55, 51, 107, 31, 119, 115, 59, 15, 12, 20, 36, 40, 60, 84, 88, 124, 16, 28, 104, 108, 92, 76, 80, 13, 17, 37, 45, 69, 121, 93, 53, 73, 58, 62, 94, 110, 122, 22, 82, 86, 102, 106, 50, 54, 46, 74], 'token_total': 114, 'token_per_expert': {23: 1, 27: 1, 43: 1, 55: 1, 51: 2, 107: 2, 31: 3, 119: 3, 115: 4, 59: 11, 15: 12, 12: 1, 20: 1, 36: 1, 40: 1, 60: 1, 84: 1, 88: 1, 124: 1, 16: 2, 28: 2, 104: 2, 108: 2, 92: 3, 76: 5, 80: 5, 13: 1, 17: 1, 37: 1, 45: 1, 69: 1, 121: 1, 93: 2, 53: 3, 73: 4, 58: 1, 62: 1, 94: 1, 110: 1, 122: 1, 22: 2, 82: 2, 86: 2, 102: 2, 106: 2, 50: 3, 54: 3, 46: 4, 74: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39, 63, 75, 83, 91, 95, 99, 103, 111, 123], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1082, 'token_per_expert': {3: 129, 7: 142, 19: 32, 39: 90, 63: 22, 75: 102, 83: 262, 91: 25, 95: 35, 99: 53, 103: 65, 111: 40, 123: 85}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 24, 44, 48, 52, 56, 68, 96, 100, 112, 120], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1156, 'token_per_expert': {0: 137, 4: 129, 8: 49, 24: 114, 44: 8, 48: 124, 52: 91, 56: 18, 68: 18, 96: 54, 100: 63, 112: 194, 120: 157}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 29, 33, 49, 57, 65, 89, 97, 109, 125], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1068, 'token_per_expert': {1: 135, 5: 131, 9: 88, 29: 19, 33: 253, 49: 18, 57: 34, 65: 62, 89: 24, 97: 190, 109: 8, 125: 106}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 26, 30, 38, 42, 70, 98, 114, 126], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 676, 'token_per_expert': {2: 135, 6: 137, 10: 19, 18: 17, 26: 161, 30: 8, 38: 16, 42: 13, 70: 39, 98: 58, 114: 24, 126: 49}}
INFO 05-03 18:02:14.175114.175114 lmp.py:1005] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.495ms | allocate_experts_across_cpu_gpu: 0.372ms
INFO 05-03 18:02:14.175549.175549 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.793571472167969e-05 seconds
INFO 05-03 18:02:14.176867.176867 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006289482116699219 seconds
INFO 05-03 18:02:14.182959.182959 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.005696535110473633 seconds
INFO 05-03 18:02:14.183165.183165 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009136199951171875 seconds
INFO 05-03 18:02:14.185446.185446 mlpmodule.py:2707] [fused_experts] gmm total=1.732ms E=32 S=1123 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.187910.187910 mlpmodule.py:2707] [fused_experts] gmm total=1.676ms E=32 S=1185 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.189296.189296 mlpmodule.py:2707] [fused_experts] gmm total=1.364ms E=32 S=1083 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.191948.191948 mlpmodule.py:2707] [fused_experts] gmm total=1.501ms E=32 S=705 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.191721.191721 lmp.py:1149] [layer_moe_fused] experts compute time: 0.00779271125793457 seconds
INFO 05-03 18:02:14.191307.191307 lmp.py:1160] [layer_moe_fused] to time: 2.9802322387695312e-05 seconds
INFO 05-03 18:02:14.191677.191677 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0003838539123535156 seconds
DEBUG 05-03 18:02:14.192264.192264 cuda_h.py:27] end *layer_moe_fused cost 17.798 ms
DEBUG 05-03 18:02:14.192252.192252 cuda_h.py:27] end prefill_layer cost 20.432 ms
DEBUG 05-03 18:02:14.192161.192161 lmp.py:711] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-03 18:02:14.192073.192073 lmp.py:675] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-03 18:02:14.194056.194056 cuda_h.py:27] end *sagl cost 2.212 ms
experts_cpu_alloc {'expert_ids': [11, 91, 103, 87, 111, 55, 28, 48, 68, 92, 104, 20, 52, 24, 56, 64, 120, 60, 36, 17, 33, 61, 77, 85, 105, 121, 45, 69, 97, 49, 101, 9, 65, 113, 26, 62, 10, 38, 106, 118, 34, 66, 94, 42, 22, 30], 'token_total': 164, 'token_per_expert': {11: 1, 91: 1, 103: 1, 87: 2, 111: 2, 55: 3, 28: 1, 48: 1, 68: 1, 92: 1, 104: 1, 20: 2, 52: 2, 24: 3, 56: 3, 64: 3, 120: 3, 60: 4, 36: 9, 17: 1, 33: 1, 61: 1, 77: 1, 85: 1, 105: 1, 121: 1, 45: 2, 69: 2, 97: 2, 49: 3, 101: 4, 9: 9, 65: 12, 113: 12, 26: 1, 62: 1, 10: 2, 38: 2, 106: 3, 118: 5, 34: 6, 66: 7, 94: 8, 42: 9, 22: 11, 30: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39, 51, 59, 67, 71, 75, 107, 115, 127], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 769, 'token_per_expert': {3: 129, 7: 355, 19: 100, 39: 4, 51: 25, 59: 42, 67: 7, 71: 4, 75: 4, 107: 36, 115: 27, 127: 36}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 32, 76, 84, 88, 96, 100, 112, 116], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1019, 'token_per_expert': {0: 132, 4: 129, 8: 81, 12: 19, 32: 14, 76: 277, 84: 24, 88: 112, 96: 121, 100: 23, 112: 33, 116: 54}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 37, 53, 57, 73, 81, 89, 109, 117, 125], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 857, 'token_per_expert': {1: 131, 5: 130, 13: 72, 37: 28, 53: 54, 57: 197, 73: 140, 81: 28, 89: 27, 109: 16, 117: 20, 125: 14}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 54, 78, 82, 86, 90, 114, 126], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1287, 'token_per_expert': {2: 130, 6: 129, 14: 23, 18: 79, 54: 175, 78: 44, 82: 136, 86: 273, 90: 168, 114: 84, 126: 46}}
INFO 05-03 18:02:14.195718.195718 lmp.py:1005] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.486ms | allocate_experts_across_cpu_gpu: 0.406ms
INFO 05-03 18:02:14.196722.196722 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.650520324707031e-05 seconds
INFO 05-03 18:02:14.196531.196531 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006117820739746094 seconds
INFO 05-03 18:02:14.203334.203334 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.005606174468994141 seconds
INFO 05-03 18:02:14.204085.204085 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009665489196777344 seconds
INFO 05-03 18:02:14.205566.205566 mlpmodule.py:2707] [fused_experts] gmm total=1.408ms E=32 S=779 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.207576.207576 mlpmodule.py:2707] [fused_experts] gmm total=1.591ms E=32 S=1053 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.209741.209741 mlpmodule.py:2707] [fused_experts] gmm total=1.366ms E=32 S=910 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.211560.211560 mlpmodule.py:2707] [fused_experts] gmm total=1.586ms E=32 S=1354 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.211089.211089 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007365226745605469 seconds
INFO 05-03 18:02:14.211691.211691 lmp.py:1160] [layer_moe_fused] to time: 4.5299530029296875e-05 seconds
INFO 05-03 18:02:14.211567.211567 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0003170967102050781 seconds
DEBUG 05-03 18:02:14.212300.212300 cuda_h.py:27] end *layer_moe_fused cost 17.300 ms
DEBUG 05-03 18:02:14.212858.212858 cuda_h.py:27] end prefill_layer cost 19.891 ms
DEBUG 05-03 18:02:14.212052.212052 lmp.py:711] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-03 18:02:14.212898.212898 lmp.py:675] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-03 18:02:14.215941.215941 cuda_h.py:27] end *sagl cost 2.256 ms
experts_cpu_alloc {'expert_ids': [23, 35, 51, 87, 123, 67, 91, 107, 115, 39, 71, 79, 75, 16, 28, 104, 116, 48, 108, 124, 40, 60, 20, 24, 52, 84, 72, 21, 41, 61, 97, 109, 125, 13, 77, 89, 33, 73, 81, 105, 113, 18, 54, 98, 122, 126, 22, 34, 50, 26, 58], 'token_total': 142, 'token_per_expert': {23: 1, 35: 1, 51: 1, 87: 1, 123: 1, 67: 2, 91: 2, 107: 2, 115: 4, 39: 7, 71: 7, 79: 7, 75: 9, 16: 1, 28: 1, 104: 1, 116: 1, 48: 2, 108: 2, 124: 2, 40: 3, 60: 3, 20: 4, 24: 4, 52: 4, 84: 4, 72: 6, 21: 1, 41: 1, 61: 1, 97: 1, 109: 1, 125: 1, 13: 2, 77: 2, 89: 2, 33: 4, 73: 5, 81: 7, 105: 7, 113: 7, 18: 1, 54: 1, 98: 1, 122: 1, 126: 1, 22: 2, 34: 2, 50: 2, 26: 3, 58: 3}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 31, 43, 55, 83, 99, 103, 111, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1739, 'token_per_expert': {3: 135, 7: 131, 15: 11, 19: 156, 27: 54, 31: 363, 43: 125, 55: 74, 83: 14, 99: 14, 103: 188, 111: 276, 127: 198}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 36, 56, 64, 68, 80, 92, 96, 112, 120], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 596, 'token_per_expert': {0: 129, 4: 157, 8: 10, 12: 9, 36: 41, 56: 95, 64: 31, 68: 30, 80: 14, 92: 7, 96: 13, 112: 22, 120: 38}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 25, 29, 49, 53, 65, 69, 85, 101, 117, 121], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 975, 'token_per_expert': {1: 129, 5: 130, 17: 58, 25: 28, 29: 22, 49: 8, 53: 9, 65: 28, 69: 147, 85: 72, 101: 183, 117: 152, 121: 9}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 30, 46, 62, 66, 70, 74, 78, 86, 90, 114], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 644, 'token_per_expert': {2: 129, 6: 137, 10: 50, 30: 88, 46: 40, 62: 50, 66: 14, 70: 83, 74: 11, 78: 3, 86: 24, 90: 4, 114: 11}}
INFO 05-03 18:02:14.215781.215781 lmp.py:1005] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.490ms | allocate_experts_across_cpu_gpu: 0.393ms
INFO 05-03 18:02:14.216971.216971 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.031990051269531e-05 seconds
INFO 05-03 18:02:14.216694.216694 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000606536865234375 seconds
INFO 05-03 18:02:14.223540.223540 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0054776668548583984 seconds
INFO 05-03 18:02:14.224086.224086 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000980377197265625 seconds
INFO 05-03 18:02:14.226070.226070 mlpmodule.py:2707] [fused_experts] gmm total=1.769ms E=32 S=1784 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.229903.229903 mlpmodule.py:2707] [fused_experts] gmm total=2.679ms E=32 S=634 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.231414.231414 mlpmodule.py:2707] [fused_experts] gmm total=1.478ms E=32 S=1017 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.232552.232552 mlpmodule.py:2707] [fused_experts] gmm total=1.363ms E=32 S=661 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.233430.233430 lmp.py:1149] [layer_moe_fused] experts compute time: 0.008983373641967773 seconds
INFO 05-03 18:02:14.233383.233383 lmp.py:1160] [layer_moe_fused] to time: 3.24249267578125e-05 seconds
INFO 05-03 18:02:14.233555.233555 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00024175643920898438 seconds
DEBUG 05-03 18:02:14.233533.233533 cuda_h.py:27] end *layer_moe_fused cost 18.786 ms
DEBUG 05-03 18:02:14.233860.233860 cuda_h.py:27] end prefill_layer cost 21.389 ms
DEBUG 05-03 18:02:14.233768.233768 lmp.py:711] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-03 18:02:14.234720.234720 lmp.py:675] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-03 18:02:14.237812.237812 cuda_h.py:27] end *sagl cost 2.963 ms
experts_cpu_alloc {'expert_ids': [11, 15, 63, 67, 71, 119, 23, 91, 19, 43, 47, 111, 28, 44, 60, 76, 88, 120, 116, 20, 68, 100, 124, 84, 29, 53, 89, 101, 37, 105, 21, 77, 81, 17, 57, 121, 25, 10, 50, 86, 102, 18, 30, 46, 70, 98, 114, 122, 22, 58, 94], 'token_total': 152, 'token_per_expert': {11: 1, 15: 1, 63: 1, 67: 1, 71: 1, 119: 1, 23: 2, 91: 2, 19: 3, 43: 3, 47: 3, 111: 4, 28: 1, 44: 1, 60: 1, 76: 1, 88: 1, 120: 1, 116: 2, 20: 3, 68: 3, 100: 4, 124: 4, 84: 8, 29: 1, 53: 1, 89: 1, 101: 1, 37: 2, 105: 2, 21: 3, 77: 3, 81: 7, 17: 10, 57: 12, 121: 13, 25: 15, 10: 1, 50: 1, 86: 1, 102: 1, 18: 2, 30: 2, 46: 2, 70: 2, 98: 2, 114: 2, 122: 2, 22: 3, 58: 3, 94: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 31, 35, 51, 59, 75, 79, 83, 99, 107, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 882, 'token_per_expert': {3: 188, 7: 179, 27: 33, 31: 19, 35: 48, 51: 130, 59: 95, 75: 47, 79: 112, 83: 12, 99: 8, 107: 6, 127: 5}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 36, 48, 52, 72, 80, 92, 96, 104], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1014, 'token_per_expert': {0: 178, 4: 174, 8: 12, 12: 21, 16: 9, 36: 109, 48: 47, 52: 13, 72: 248, 80: 9, 92: 87, 96: 21, 104: 86}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 33, 45, 49, 61, 69, 73, 93, 117, 125], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1156, 'token_per_expert': {1: 307, 5: 393, 9: 18, 13: 116, 33: 15, 45: 58, 49: 21, 61: 19, 69: 103, 73: 17, 93: 22, 117: 31, 125: 36}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34, 42, 54, 62, 66, 74, 82, 90, 110, 118, 126], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 892, 'token_per_expert': {2: 330, 6: 136, 34: 5, 42: 9, 54: 5, 62: 29, 66: 9, 74: 76, 82: 261, 90: 14, 110: 5, 118: 6, 126: 7}}
INFO 05-03 18:02:14.238918.238918 lmp.py:1005] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.521ms | allocate_experts_across_cpu_gpu: 0.381ms
INFO 05-03 18:02:14.238009.238009 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.626678466796875e-05 seconds
INFO 05-03 18:02:14.239937.239937 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006215572357177734 seconds
INFO 05-03 18:02:14.245219.245219 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00583648681640625 seconds
INFO 05-03 18:02:14.246380.246380 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009322166442871094 seconds
INFO 05-03 18:02:14.248148.248148 mlpmodule.py:2707] [fused_experts] gmm total=1.648ms E=32 S=905 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.250829.250829 mlpmodule.py:2707] [fused_experts] gmm total=1.470ms E=32 S=1044 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.252636.252636 mlpmodule.py:2707] [fused_experts] gmm total=1.538ms E=32 S=1227 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.254848.254848 mlpmodule.py:2707] [fused_experts] gmm total=1.617ms E=32 S=920 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.254105.254105 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007715463638305664 seconds
INFO 05-03 18:02:14.254342.254342 lmp.py:1160] [layer_moe_fused] to time: 4.029273986816406e-05 seconds
INFO 05-03 18:02:14.254679.254679 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002300739288330078 seconds
DEBUG 05-03 18:02:14.255610.255610 cuda_h.py:27] end *layer_moe_fused cost 17.813 ms
DEBUG 05-03 18:02:14.255930.255930 cuda_h.py:27] end prefill_layer cost 21.117 ms
DEBUG 05-03 18:02:14.255885.255885 lmp.py:711] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-03 18:02:14.255545.255545 lmp.py:675] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-03 18:02:14.257748.257748 cuda_h.py:27] end *sagl cost 2.269 ms
experts_cpu_alloc {'expert_ids': [51, 91, 119, 71, 111, 11, 55, 107, 15, 103, 79, 87, 59, 39, 108, 120, 124, 36, 64, 72, 112, 16, 68, 116, 8, 56, 40, 52, 104, 21, 49, 65, 85, 113, 25, 69, 81, 33, 105, 38, 46, 54, 58, 78, 86, 102, 106, 14, 70, 126, 22, 30, 66, 110], 'token_total': 195, 'token_per_expert': {51: 1, 91: 1, 119: 1, 71: 2, 111: 2, 11: 3, 55: 3, 107: 5, 15: 8, 103: 8, 79: 11, 87: 11, 59: 13, 39: 18, 108: 1, 120: 1, 124: 1, 36: 2, 64: 2, 72: 2, 112: 2, 16: 3, 68: 3, 116: 3, 8: 4, 56: 5, 40: 6, 52: 6, 104: 7, 21: 1, 49: 1, 65: 1, 85: 1, 113: 1, 25: 2, 69: 2, 81: 2, 33: 3, 105: 3, 38: 1, 46: 1, 54: 1, 58: 1, 78: 1, 86: 1, 102: 1, 106: 1, 14: 2, 70: 2, 126: 4, 22: 5, 30: 5, 66: 5, 110: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 43, 47, 67, 75, 95, 99, 115, 123, 127], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1345, 'token_per_expert': {3: 132, 7: 199, 19: 35, 23: 43, 27: 40, 43: 19, 47: 22, 67: 60, 75: 53, 95: 271, 99: 332, 115: 32, 123: 84, 127: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 28, 32, 44, 48, 76, 80, 84, 92, 96, 100], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 921, 'token_per_expert': {0: 155, 4: 131, 20: 13, 24: 8, 28: 84, 32: 35, 44: 10, 48: 37, 76: 149, 80: 38, 84: 13, 92: 201, 96: 11, 100: 36}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 29, 41, 57, 61, 73, 77, 89, 97, 125], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 684, 'token_per_expert': {1: 131, 5: 132, 9: 9, 17: 9, 29: 9, 41: 95, 57: 214, 61: 22, 73: 5, 77: 11, 89: 34, 97: 6, 125: 7}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 34, 42, 50, 74, 82, 94, 98, 114, 118, 122], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 951, 'token_per_expert': {2: 131, 6: 130, 18: 93, 34: 109, 42: 18, 50: 50, 74: 15, 82: 150, 94: 34, 98: 28, 114: 135, 118: 36, 122: 22}}
INFO 05-03 18:02:14.258066.258066 lmp.py:1005] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.489ms | allocate_experts_across_cpu_gpu: 0.396ms
INFO 05-03 18:02:14.259402.259402 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.175041198730469e-05 seconds
INFO 05-03 18:02:14.259181.259181 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000637054443359375 seconds
INFO 05-03 18:02:14.266763.266763 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.005907773971557617 seconds
INFO 05-03 18:02:14.267396.267396 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009572505950927734 seconds
INFO 05-03 18:02:14.269729.269729 mlpmodule.py:2707] [fused_experts] gmm total=1.913ms E=32 S=1432 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.271263.271263 mlpmodule.py:2707] [fused_experts] gmm total=1.584ms E=32 S=969 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.273880.273880 mlpmodule.py:2707] [fused_experts] gmm total=1.255ms E=32 S=701 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.274062.274062 mlpmodule.py:2707] [fused_experts] gmm total=1.372ms E=32 S=994 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.275615.275615 lmp.py:1149] [layer_moe_fused] experts compute time: 0.0074427127838134766 seconds
INFO 05-03 18:02:14.275613.275613 lmp.py:1160] [layer_moe_fused] to time: 3.528594970703125e-05 seconds
INFO 05-03 18:02:14.275725.275725 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002346038818359375 seconds
DEBUG 05-03 18:02:14.275591.275591 cuda_h.py:27] end *layer_moe_fused cost 17.777 ms
DEBUG 05-03 18:02:14.275441.275441 cuda_h.py:27] end prefill_layer cost 20.429 ms
DEBUG 05-03 18:02:14.275257.275257 lmp.py:711] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-03 18:02:14.276636.276636 lmp.py:675] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-03 18:02:14.278302.278302 cuda_h.py:27] end *sagl cost 2.265 ms
experts_cpu_alloc {'expert_ids': [15, 47, 51, 67, 95, 23, 39, 79, 87, 99, 111, 27, 31, 75, 119, 103, 20, 60, 72, 80, 84, 16, 68, 116, 32, 96, 104, 9, 25, 45, 53, 105, 29, 81, 77, 69, 61, 85, 37, 46, 78, 50, 62, 90, 94, 110, 126, 86, 66, 102, 114, 106], 'token_total': 199, 'token_per_expert': {15: 1, 47: 1, 51: 1, 67: 1, 95: 1, 23: 2, 39: 2, 79: 2, 87: 2, 99: 2, 111: 2, 27: 3, 31: 4, 75: 6, 119: 6, 103: 10, 20: 1, 60: 1, 72: 1, 80: 2, 84: 2, 16: 3, 68: 3, 116: 3, 32: 5, 96: 6, 104: 6, 9: 1, 25: 1, 45: 1, 53: 1, 105: 1, 29: 2, 81: 4, 77: 6, 69: 8, 61: 14, 85: 16, 37: 19, 46: 1, 78: 1, 50: 2, 62: 2, 90: 2, 94: 2, 110: 2, 126: 2, 86: 3, 66: 6, 102: 7, 114: 7, 106: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 43, 55, 59, 63, 71, 83, 91, 123, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1406, 'token_per_expert': {3: 129, 7: 143, 11: 169, 19: 59, 43: 120, 55: 101, 59: 250, 63: 66, 71: 93, 83: 46, 91: 131, 123: 38, 127: 61}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 28, 36, 40, 52, 56, 64, 76, 100, 112, 124], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 593, 'token_per_expert': {0: 153, 4: 133, 8: 39, 28: 8, 36: 11, 40: 15, 52: 9, 56: 14, 64: 58, 76: 113, 100: 18, 112: 6, 124: 16}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 21, 33, 49, 57, 73, 93, 97, 121, 125], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1142, 'token_per_expert': {1: 130, 5: 157, 13: 118, 17: 23, 21: 127, 33: 123, 49: 39, 57: 123, 73: 27, 93: 184, 97: 40, 121: 21, 125: 30}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 26, 54, 58, 70, 74, 82, 98, 118, 122], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 756, 'token_per_expert': {2: 176, 6: 142, 18: 18, 22: 11, 26: 41, 54: 31, 58: 43, 70: 42, 74: 26, 82: 32, 98: 117, 118: 62, 122: 15}}
INFO 05-03 18:02:14.279745.279745 lmp.py:1005] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.488ms | allocate_experts_across_cpu_gpu: 0.384ms
INFO 05-03 18:02:14.279280.279280 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.793571472167969e-05 seconds
INFO 05-03 18:02:14.280347.280347 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000598907470703125 seconds
INFO 05-03 18:02:14.286223.286223 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00559234619140625 seconds
INFO 05-03 18:02:14.287557.287557 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009317398071289062 seconds
INFO 05-03 18:02:14.289690.289690 mlpmodule.py:2707] [fused_experts] gmm total=1.884ms E=32 S=1452 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.291175.291175 mlpmodule.py:2707] [fused_experts] gmm total=1.302ms E=32 S=626 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.293376.293376 mlpmodule.py:2707] [fused_experts] gmm total=1.304ms E=32 S=1216 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.294178.294178 mlpmodule.py:2707] [fused_experts] gmm total=1.490ms E=32 S=802 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.295858.295858 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007339000701904297 seconds
INFO 05-03 18:02:14.295352.295352 lmp.py:1160] [layer_moe_fused] to time: 3.0994415283203125e-05 seconds
INFO 05-03 18:02:14.295000.295000 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002040863037109375 seconds
DEBUG 05-03 18:02:14.295196.295196 cuda_h.py:27] end *layer_moe_fused cost 17.112 ms
DEBUG 05-03 18:02:14.295900.295900 cuda_h.py:27] end prefill_layer cost 19.752 ms
DEBUG 05-03 18:02:14.295570.295570 lmp.py:711] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-03 18:02:14.296416.296416 lmp.py:675] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-03 18:02:14.298749.298749 cuda_h.py:27] end *sagl cost 2.270 ms
experts_cpu_alloc {'expert_ids': [27, 31, 59, 83, 43, 127, 35, 75, 95, 79, 107, 111, 68, 100, 116, 36, 88, 120, 8, 72, 92, 64, 84, 48, 40, 52, 32, 13, 29, 65, 69, 17, 41, 57, 81, 93, 109, 26, 118, 38, 50, 114, 126, 42, 66, 90, 58, 82, 94, 78], 'token_total': 184, 'token_per_expert': {27: 1, 31: 1, 59: 2, 83: 2, 43: 3, 127: 3, 35: 4, 75: 4, 95: 5, 79: 6, 107: 6, 111: 7, 68: 1, 100: 1, 116: 1, 36: 2, 88: 2, 120: 2, 8: 3, 72: 3, 92: 3, 64: 4, 84: 4, 48: 5, 40: 6, 52: 9, 32: 12, 13: 1, 29: 1, 65: 2, 69: 2, 17: 3, 41: 4, 57: 5, 81: 5, 93: 5, 109: 6, 26: 1, 118: 1, 38: 2, 50: 2, 114: 2, 126: 2, 42: 3, 66: 3, 90: 3, 58: 6, 82: 6, 94: 7, 78: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 55, 63, 67, 71, 87, 99, 115, 123], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 695, 'token_per_expert': {3: 137, 7: 142, 11: 84, 15: 8, 19: 10, 55: 9, 63: 17, 67: 12, 71: 57, 87: 55, 99: 82, 115: 29, 123: 53}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 20, 24, 28, 44, 56, 60, 96, 108, 112], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1216, 'token_per_expert': {0: 129, 4: 145, 12: 14, 16: 24, 20: 25, 24: 62, 28: 106, 44: 159, 56: 188, 60: 18, 96: 41, 108: 207, 112: 98}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 33, 37, 45, 73, 89, 101, 113, 117, 121, 125], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 993, 'token_per_expert': {1: 129, 5: 144, 9: 45, 33: 15, 37: 111, 45: 13, 73: 7, 89: 59, 101: 53, 113: 77, 117: 48, 121: 8, 125: 284}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 30, 46, 54, 62, 98, 102, 106], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1008, 'token_per_expert': {2: 135, 6: 150, 10: 114, 18: 267, 22: 15, 30: 12, 46: 47, 54: 129, 62: 50, 98: 52, 102: 21, 106: 16}}
INFO 05-03 18:02:14.299416.299416 lmp.py:1005] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.486ms | allocate_experts_across_cpu_gpu: 0.375ms
INFO 05-03 18:02:14.299699.299699 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.8650970458984375e-05 seconds
INFO 05-03 18:02:14.300792.300792 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005917549133300781 seconds
INFO 05-03 18:02:14.306332.306332 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.005870342254638672 seconds
INFO 05-03 18:02:14.307161.307161 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009131431579589844 seconds
INFO 05-03 18:02:14.309357.309357 mlpmodule.py:2707] [fused_experts] gmm total=1.583ms E=32 S=739 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.311666.311666 mlpmodule.py:2707] [fused_experts] gmm total=1.424ms E=32 S=1274 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.312147.312147 mlpmodule.py:2707] [fused_experts] gmm total=1.386ms E=32 S=1027 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.314480.314480 mlpmodule.py:2707] [fused_experts] gmm total=1.478ms E=32 S=1056 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.315220.315220 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007197856903076172 seconds
INFO 05-03 18:02:14.315137.315137 lmp.py:1160] [layer_moe_fused] to time: 3.0279159545898438e-05 seconds
INFO 05-03 18:02:14.315699.315699 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00021004676818847656 seconds
DEBUG 05-03 18:02:14.315902.315902 cuda_h.py:27] end *layer_moe_fused cost 17.162 ms
DEBUG 05-03 18:02:14.315130.315130 cuda_h.py:27] end prefill_layer cost 19.733 ms
DEBUG 05-03 18:02:14.315277.315277 lmp.py:711] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-03 18:02:14.315527.315527 lmp.py:675] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-03 18:02:14.318024.318024 cuda_h.py:27] end *sagl cost 2.244 ms
experts_cpu_alloc {'expert_ids': [23, 43, 55, 123, 11, 79, 103, 71, 111, 63, 67, 15, 87, 91, 107, 40, 84, 88, 96, 108, 24, 116, 56, 80, 28, 64, 12, 44, 53, 73, 85, 125, 9, 25, 29, 105, 77, 101, 10, 78, 90, 110, 122, 30, 66, 98, 50, 118, 82, 102, 34, 18], 'token_total': 208, 'token_per_expert': {23: 1, 43: 1, 55: 1, 123: 1, 11: 2, 79: 2, 103: 2, 71: 3, 111: 3, 63: 4, 67: 9, 15: 13, 87: 14, 91: 16, 107: 19, 40: 1, 84: 1, 88: 1, 96: 1, 108: 1, 24: 2, 116: 2, 56: 3, 80: 4, 28: 5, 64: 5, 12: 6, 44: 7, 53: 1, 73: 1, 85: 1, 125: 1, 9: 2, 25: 2, 29: 2, 105: 2, 77: 4, 101: 7, 10: 1, 78: 1, 90: 1, 110: 1, 122: 1, 30: 3, 66: 3, 98: 3, 50: 4, 118: 4, 82: 5, 102: 5, 34: 11, 18: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 27, 35, 39, 47, 51, 83, 99, 115, 119, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1348, 'token_per_expert': {3: 261, 7: 129, 19: 73, 27: 195, 35: 40, 39: 22, 47: 29, 51: 103, 83: 20, 99: 81, 115: 28, 119: 112, 127: 255}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 32, 36, 48, 60, 68, 72, 92, 104, 112], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 550, 'token_per_expert': {0: 130, 4: 130, 8: 9, 16: 16, 32: 18, 36: 19, 48: 88, 60: 50, 68: 26, 72: 9, 92: 25, 104: 13, 112: 17}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 21, 33, 41, 45, 61, 69, 97, 109, 113], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1099, 'token_per_expert': {1: 130, 5: 145, 13: 243, 17: 37, 21: 208, 33: 80, 41: 24, 45: 10, 61: 31, 69: 105, 97: 15, 109: 15, 113: 56}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 26, 46, 54, 70, 74, 86, 94, 106, 114], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 891, 'token_per_expert': {2: 129, 6: 129, 14: 27, 22: 119, 26: 59, 46: 13, 54: 47, 70: 70, 74: 39, 86: 136, 94: 32, 106: 35, 114: 56}}
INFO 05-03 18:02:14.319083.319083 lmp.py:1005] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.488ms | allocate_experts_across_cpu_gpu: 0.381ms
INFO 05-03 18:02:14.319220.319220 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.626678466796875e-05 seconds
INFO 05-03 18:02:14.320596.320596 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005929470062255859 seconds
INFO 05-03 18:02:14.326605.326605 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.005627632141113281 seconds
INFO 05-03 18:02:14.327435.327435 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009124279022216797 seconds
INFO 05-03 18:02:14.329402.329402 mlpmodule.py:2707] [fused_experts] gmm total=1.692ms E=32 S=1439 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.331245.331245 mlpmodule.py:2707] [fused_experts] gmm total=1.361ms E=32 S=589 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.332154.332154 mlpmodule.py:2707] [fused_experts] gmm total=1.307ms E=32 S=1122 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.334270.334270 mlpmodule.py:2707] [fused_experts] gmm total=1.359ms E=32 S=946 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.334784.334784 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006987571716308594 seconds
INFO 05-03 18:02:14.334086.334086 lmp.py:1160] [layer_moe_fused] to time: 3.0279159545898438e-05 seconds
INFO 05-03 18:02:14.335198.335198 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00022912025451660156 seconds
DEBUG 05-03 18:02:14.335812.335812 cuda_h.py:27] end *layer_moe_fused cost 16.930 ms
DEBUG 05-03 18:02:14.335609.335609 cuda_h.py:27] end prefill_layer cost 19.478 ms
DEBUG 05-03 18:02:14.335803.335803 lmp.py:711] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-03 18:02:14.335536.335536 lmp.py:675] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-03 18:02:14.338054.338054 cuda_h.py:27] end *sagl cost 2.260 ms
experts_cpu_alloc {'expert_ids': [75, 87, 127, 11, 39, 123, 19, 51, 63, 83, 107, 23, 95, 111, 20, 40, 60, 88, 92, 124, 48, 76, 112, 56, 44, 28, 96, 21, 61, 93, 121, 41, 49, 117, 125, 73, 37, 109, 77, 81, 9, 50, 86, 94, 30, 34, 38, 66, 118, 82, 102, 110, 10, 126, 98, 122], 'token_total': 237, 'token_per_expert': {75: 1, 87: 1, 127: 1, 11: 2, 39: 2, 123: 2, 19: 3, 51: 3, 63: 3, 83: 3, 107: 3, 23: 4, 95: 5, 111: 5, 20: 1, 40: 1, 60: 1, 88: 3, 92: 3, 124: 3, 48: 4, 76: 6, 112: 6, 56: 7, 44: 8, 28: 9, 96: 9, 21: 1, 61: 1, 93: 1, 121: 1, 41: 2, 49: 2, 117: 2, 125: 2, 73: 4, 37: 7, 109: 9, 77: 10, 81: 14, 9: 20, 50: 1, 86: 1, 94: 1, 30: 2, 34: 2, 38: 3, 66: 3, 118: 4, 82: 5, 102: 5, 110: 5, 10: 6, 126: 6, 98: 9, 122: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 31, 35, 47, 55, 59, 71, 79, 99, 103, 119], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1094, 'token_per_expert': {3: 131, 7: 262, 15: 7, 27: 102, 31: 68, 35: 47, 47: 8, 55: 145, 59: 9, 71: 151, 79: 17, 99: 11, 103: 64, 119: 72}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 32, 36, 52, 68, 72, 108, 116, 120], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 939, 'token_per_expert': {0: 189, 4: 129, 8: 153, 12: 93, 16: 11, 24: 11, 32: 47, 36: 84, 52: 66, 68: 60, 72: 11, 108: 19, 116: 34, 120: 32}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 29, 33, 53, 65, 69, 85, 89, 97, 105, 113], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1024, 'token_per_expert': {1: 135, 5: 129, 13: 173, 25: 189, 29: 21, 33: 26, 53: 24, 65: 25, 69: 35, 85: 69, 89: 31, 97: 41, 105: 59, 113: 67}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 42, 46, 58, 62, 70, 78, 106, 114], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 802, 'token_per_expert': {2: 138, 6: 143, 14: 12, 18: 75, 22: 91, 26: 26, 42: 103, 46: 28, 58: 21, 62: 13, 70: 40, 78: 43, 106: 39, 114: 30}}
INFO 05-03 18:02:14.339955.339955 lmp.py:1005] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.489ms | allocate_experts_across_cpu_gpu: 0.403ms
INFO 05-03 18:02:14.339145.339145 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.151199340820312e-05 seconds
INFO 05-03 18:02:14.340247.340247 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006697177886962891 seconds
INFO 05-03 18:02:14.346669.346669 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.006075382232666016 seconds
INFO 05-03 18:02:14.347832.347832 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009696483612060547 seconds
INFO 05-03 18:02:14.349503.349503 mlpmodule.py:2707] [fused_experts] gmm total=1.541ms E=32 S=1132 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.351505.351505 mlpmodule.py:2707] [fused_experts] gmm total=1.552ms E=32 S=1000 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.353675.353675 mlpmodule.py:2707] [fused_experts] gmm total=1.553ms E=32 S=1100 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.354382.354382 mlpmodule.py:2707] [fused_experts] gmm total=1.402ms E=32 S=864 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.355942.355942 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007370710372924805 seconds
INFO 05-03 18:02:14.355897.355897 lmp.py:1160] [layer_moe_fused] to time: 0.00016880035400390625 seconds
INFO 05-03 18:02:14.355178.355178 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0004904270172119141 seconds
DEBUG 05-03 18:02:14.356306.356306 cuda_h.py:27] end *layer_moe_fused cost 18.321 ms
DEBUG 05-03 18:02:14.356640.356640 cuda_h.py:27] end prefill_layer cost 20.897 ms
DEBUG 05-03 18:02:14.356933.356933 lmp.py:711] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-03 18:02:14.356782.356782 lmp.py:675] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-03 18:02:14.361936.361936 cuda_h.py:27] end *sagl cost 4.082 ms
experts_cpu_alloc {'expert_ids': [11, 23, 71, 95, 127, 111, 63, 79, 83, 103, 119, 27, 96, 44, 112, 124, 92, 104, 28, 88, 16, 68, 36, 29, 49, 61, 73, 77, 81, 17, 57, 9, 97, 93, 117, 45, 13, 18, 42, 54, 106, 118, 82, 90, 102, 122, 74, 98, 46, 38, 114, 110, 78], 'token_total': 282, 'token_per_expert': {11: 1, 23: 1, 71: 1, 95: 1, 127: 1, 111: 2, 63: 5, 79: 5, 83: 5, 103: 5, 119: 5, 27: 7, 96: 1, 44: 2, 112: 3, 124: 3, 92: 4, 104: 4, 28: 7, 88: 10, 16: 12, 68: 12, 36: 14, 29: 1, 49: 1, 61: 1, 73: 1, 77: 1, 81: 1, 17: 2, 57: 2, 9: 3, 97: 3, 93: 4, 117: 4, 45: 5, 13: 9, 18: 1, 42: 1, 54: 1, 106: 1, 118: 1, 82: 3, 90: 4, 102: 8, 122: 8, 74: 9, 98: 10, 46: 11, 38: 12, 114: 15, 110: 19, 78: 29}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 31, 35, 47, 51, 55, 67, 87, 91, 99, 107, 115], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 925, 'token_per_expert': {3: 130, 7: 136, 15: 193, 31: 11, 35: 17, 47: 79, 51: 75, 55: 14, 67: 24, 87: 61, 91: 49, 99: 59, 107: 40, 115: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 40, 48, 52, 56, 60, 76, 84, 100], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 937, 'token_per_expert': {0: 219, 4: 131, 12: 59, 20: 126, 24: 23, 40: 23, 48: 27, 52: 182, 56: 14, 60: 26, 76: 78, 84: 14, 100: 15}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 37, 41, 53, 65, 69, 85, 89, 101, 105, 113], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 687, 'token_per_expert': {1: 129, 5: 129, 21: 11, 37: 15, 41: 86, 53: 40, 65: 12, 69: 14, 85: 35, 89: 145, 101: 41, 105: 14, 113: 16}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 26, 30, 34, 58, 62, 66, 70, 94, 126], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1265, 'token_per_expert': {2: 163, 6: 139, 10: 123, 22: 51, 26: 30, 30: 51, 34: 83, 58: 59, 62: 78, 66: 85, 70: 150, 94: 108, 126: 145}}
INFO 05-03 18:02:14.362208.362208 lmp.py:1005] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.631ms | allocate_experts_across_cpu_gpu: 0.442ms
INFO 05-03 18:02:14.362161.362161 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 8.249282836914062e-05 seconds
INFO 05-03 18:02:14.363230.363230 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007672309875488281 seconds
INFO 05-03 18:02:14.370523.370523 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0066339969635009766 seconds
INFO 05-03 18:02:14.371170.371170 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000982522964477539 seconds
INFO 05-03 18:02:14.373093.373093 mlpmodule.py:2707] [fused_experts] gmm total=1.542ms E=32 S=964 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.375410.375410 mlpmodule.py:2707] [fused_experts] gmm total=1.459ms E=32 S=1009 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.376407.376407 mlpmodule.py:2707] [fused_experts] gmm total=1.329ms E=32 S=725 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.378722.378722 mlpmodule.py:2707] [fused_experts] gmm total=1.565ms E=32 S=1398 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.379859.379859 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007231950759887695 seconds
INFO 05-03 18:02:14.379653.379653 lmp.py:1160] [layer_moe_fused] to time: 3.600120544433594e-05 seconds
INFO 05-03 18:02:14.379262.379262 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00025463104248046875 seconds
DEBUG 05-03 18:02:14.379334.379334 cuda_h.py:27] end *layer_moe_fused cost 18.693 ms
DEBUG 05-03 18:02:14.379429.379429 cuda_h.py:27] end prefill_layer cost 23.090 ms
DEBUG 05-03 18:02:14.379861.379861 lmp.py:711] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-03 18:02:14.380885.380885 lmp.py:675] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-03 18:02:14.382964.382964 cuda_h.py:27] end *sagl cost 2.424 ms
experts_cpu_alloc {'expert_ids': [31, 39, 47, 71, 107, 75, 123, 59, 63, 127, 43, 115, 51, 12, 64, 124, 8, 100, 24, 20, 60, 44, 116, 96, 48, 104, 93, 105, 121, 25, 33, 97, 113, 61, 37, 73, 13, 125, 101, 30, 50, 58, 74, 78, 90, 122, 42, 102, 14], 'token_total': 183, 'token_per_expert': {31: 1, 39: 1, 47: 1, 71: 1, 107: 1, 75: 2, 123: 2, 59: 3, 63: 4, 127: 6, 43: 7, 115: 8, 51: 10, 12: 1, 64: 1, 124: 1, 8: 2, 100: 2, 24: 3, 20: 4, 60: 4, 44: 5, 116: 5, 96: 7, 48: 9, 104: 10, 93: 1, 105: 1, 121: 1, 25: 2, 33: 2, 97: 2, 113: 2, 61: 5, 37: 6, 73: 8, 13: 10, 125: 10, 101: 11, 30: 1, 50: 1, 58: 1, 74: 1, 78: 2, 90: 2, 122: 2, 42: 3, 102: 3, 14: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 27, 35, 67, 83, 87, 95, 99, 103, 111], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 987, 'token_per_expert': {3: 133, 7: 133, 11: 21, 15: 19, 27: 65, 35: 120, 67: 11, 83: 13, 87: 56, 95: 52, 99: 120, 103: 137, 111: 107}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 36, 40, 68, 72, 76, 80, 84, 88, 92, 120], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1224, 'token_per_expert': {0: 130, 4: 140, 28: 33, 36: 195, 40: 219, 68: 15, 72: 21, 76: 14, 80: 73, 84: 128, 88: 64, 92: 55, 120: 137}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 29, 41, 45, 49, 69, 77, 89], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 828, 'token_per_expert': {1: 129, 5: 139, 9: 72, 17: 21, 21: 96, 29: 23, 41: 47, 45: 49, 49: 13, 69: 56, 77: 114, 89: 69}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 26, 54, 62, 82, 94, 98, 118, 126], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 874, 'token_per_expert': {2: 143, 6: 130, 18: 40, 22: 227, 26: 13, 54: 122, 62: 5, 82: 88, 94: 18, 98: 66, 118: 5, 126: 17}}
INFO 05-03 18:02:14.383899.383899 lmp.py:1005] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.543ms | allocate_experts_across_cpu_gpu: 0.406ms
INFO 05-03 18:02:14.384851.384851 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.0558319091796875e-05 seconds
INFO 05-03 18:02:14.384816.384816 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006043910980224609 seconds
INFO 05-03 18:02:14.391339.391339 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.005727529525756836 seconds
INFO 05-03 18:02:14.392263.392263 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009837150573730469 seconds
INFO 05-03 18:02:14.393865.393865 mlpmodule.py:2707] [fused_experts] gmm total=1.463ms E=32 S=1034 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.395098.395098 mlpmodule.py:2707] [fused_experts] gmm total=1.351ms E=32 S=1278 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.397161.397161 mlpmodule.py:2707] [fused_experts] gmm total=1.306ms E=32 S=889 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.398702.398702 mlpmodule.py:2707] [fused_experts] gmm total=1.386ms E=32 S=895 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.399428.399428 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006819248199462891 seconds
INFO 05-03 18:02:14.399387.399387 lmp.py:1160] [layer_moe_fused] to time: 3.266334533691406e-05 seconds
INFO 05-03 18:02:14.399607.399607 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00027561187744140625 seconds
DEBUG 05-03 18:02:14.400743.400743 cuda_h.py:27] end *layer_moe_fused cost 17.235 ms
DEBUG 05-03 18:02:14.400685.400685 cuda_h.py:27] end prefill_layer cost 19.937 ms
DEBUG 05-03 18:02:14.400071.400071 lmp.py:711] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-03 18:02:14.400069.400069 lmp.py:675] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-03 18:02:14.402692.402692 cuda_h.py:27] end *sagl cost 2.246 ms
experts_cpu_alloc {'expert_ids': [31, 59, 71, 87, 91, 103, 127, 35, 47, 111, 15, 43, 55, 95, 36, 44, 80, 124, 52, 92, 96, 16, 104, 56, 120, 76, 32, 40, 108, 33, 121, 21, 97, 61, 85, 37, 45, 113, 9, 53, 117, 125, 105, 89, 29, 98, 118, 58, 10, 34, 46, 114, 14, 62, 78, 122, 126, 22, 86, 90], 'token_total': 236, 'token_per_expert': {31: 1, 59: 1, 71: 1, 87: 1, 91: 1, 103: 1, 127: 1, 35: 2, 47: 2, 111: 2, 15: 3, 43: 3, 55: 4, 95: 5, 36: 1, 44: 1, 80: 1, 124: 1, 52: 2, 92: 2, 96: 2, 16: 3, 104: 4, 56: 5, 120: 5, 76: 6, 32: 7, 40: 14, 108: 14, 33: 1, 121: 1, 21: 2, 97: 2, 61: 3, 85: 4, 37: 5, 45: 5, 113: 5, 9: 6, 53: 6, 117: 6, 125: 6, 105: 7, 89: 14, 29: 16, 98: 1, 118: 1, 58: 2, 10: 3, 34: 3, 46: 3, 114: 3, 14: 4, 62: 4, 78: 4, 122: 4, 126: 4, 22: 5, 86: 5, 90: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 39, 51, 63, 67, 75, 99, 107, 115, 119, 123], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 825, 'token_per_expert': {3: 130, 7: 132, 11: 11, 19: 116, 23: 52, 27: 6, 39: 23, 51: 7, 63: 8, 67: 129, 75: 84, 99: 7, 107: 9, 115: 8, 119: 6, 123: 97}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 24, 28, 48, 60, 68, 72, 84, 88, 100, 112, 116], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1182, 'token_per_expert': {0: 255, 4: 131, 8: 41, 20: 77, 24: 144, 28: 52, 48: 15, 60: 28, 68: 22, 72: 166, 84: 54, 88: 82, 100: 17, 112: 38, 116: 60}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 25, 41, 49, 57, 65, 69, 73, 77, 81, 101, 109], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1138, 'token_per_expert': {1: 203, 5: 129, 13: 26, 17: 23, 25: 74, 41: 165, 49: 47, 57: 29, 65: 63, 69: 32, 73: 114, 77: 60, 81: 20, 101: 80, 109: 73}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 30, 38, 50, 54, 66, 70, 74, 82, 94, 102, 106, 110], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 715, 'token_per_expert': {2: 130, 6: 131, 18: 9, 30: 16, 38: 20, 50: 70, 54: 47, 66: 31, 70: 26, 74: 41, 82: 29, 94: 6, 102: 44, 106: 27, 110: 88}}
INFO 05-03 18:02:14.403668.403668 lmp.py:1005] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.500ms | allocate_experts_across_cpu_gpu: 0.485ms
INFO 05-03 18:02:14.404819.404819 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.628036499023438e-05 seconds
INFO 05-03 18:02:14.404869.404869 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005950927734375 seconds
INFO 05-03 18:02:14.412563.412563 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.006496429443359375 seconds
INFO 05-03 18:02:14.413257.413257 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001005411148071289 seconds
INFO 05-03 18:02:14.414432.414432 mlpmodule.py:2707] [fused_experts] gmm total=1.568ms E=32 S=853 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.416279.416279 mlpmodule.py:2707] [fused_experts] gmm total=1.480ms E=32 S=1250 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.418065.418065 mlpmodule.py:2707] [fused_experts] gmm total=1.788ms E=32 S=1227 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.420523.420523 mlpmodule.py:2707] [fused_experts] gmm total=1.426ms E=32 S=766 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.420931.420931 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007582187652587891 seconds
INFO 05-03 18:02:14.420279.420279 lmp.py:1160] [layer_moe_fused] to time: 2.956390380859375e-05 seconds
INFO 05-03 18:02:14.421319.421319 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002474784851074219 seconds
DEBUG 05-03 18:02:14.421115.421115 cuda_h.py:27] end *layer_moe_fused cost 18.698 ms
DEBUG 05-03 18:02:14.421819.421819 cuda_h.py:27] end prefill_layer cost 21.234 ms
DEBUG 05-03 18:02:14.421966.421966 lmp.py:711] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-03 18:02:14.421366.421366 lmp.py:675] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-03 18:02:14.424885.424885 cuda_h.py:27] end *sagl cost 2.289 ms
experts_cpu_alloc {'expert_ids': [39, 91, 43, 71, 15, 27, 79, 95, 31, 83, 19, 59, 67, 11, 87, 16, 48, 100, 104, 72, 12, 80, 84, 52, 92, 64, 68, 124, 56, 36, 24, 45, 113, 121, 69, 29, 65, 89, 109, 81, 105, 57, 49, 21, 61, 101, 62, 10, 46, 50, 86, 82, 30, 98, 34, 70, 94, 110, 118, 26, 54, 106, 66], 'token_total': 480, 'token_per_expert': {39: 1, 91: 1, 43: 2, 71: 2, 15: 3, 27: 3, 79: 3, 95: 3, 31: 4, 83: 6, 19: 7, 59: 7, 67: 7, 11: 9, 87: 9, 16: 1, 48: 1, 100: 1, 104: 1, 72: 3, 12: 4, 80: 4, 84: 4, 52: 5, 92: 6, 64: 7, 68: 8, 124: 8, 56: 10, 36: 13, 24: 15, 45: 1, 113: 1, 121: 2, 69: 4, 29: 5, 65: 5, 89: 6, 109: 6, 81: 10, 105: 12, 57: 13, 49: 14, 21: 23, 61: 23, 101: 23, 62: 1, 10: 4, 46: 5, 50: 5, 86: 5, 82: 7, 30: 9, 98: 10, 34: 12, 70: 13, 94: 13, 110: 13, 118: 14, 26: 15, 54: 15, 106: 16, 66: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 35, 47, 51, 55, 63, 75, 99, 103, 107, 111, 115, 119, 123], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 639, 'token_per_expert': {3: 129, 7: 146, 23: 75, 35: 38, 47: 41, 51: 19, 55: 21, 63: 13, 75: 16, 99: 14, 103: 23, 107: 23, 111: 31, 115: 17, 119: 23, 123: 10}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 28, 32, 40, 44, 60, 76, 88, 96, 108, 112, 116, 120], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 990, 'token_per_expert': {0: 181, 4: 130, 8: 44, 20: 36, 28: 18, 32: 40, 40: 33, 44: 62, 60: 61, 76: 67, 88: 108, 96: 16, 108: 60, 112: 27, 116: 90, 120: 17}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 25, 33, 37, 41, 53, 73, 77, 85, 93, 97, 125], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 1162, 'token_per_expert': {1: 165, 5: 131, 9: 37, 13: 102, 17: 27, 25: 133, 33: 52, 37: 36, 41: 35, 53: 77, 73: 91, 77: 144, 85: 40, 93: 25, 97: 40, 125: 27}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 38, 42, 58, 74, 78, 90, 102, 114, 122, 126], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 825, 'token_per_expert': {2: 140, 6: 133, 14: 46, 18: 39, 22: 38, 38: 98, 42: 62, 58: 26, 74: 22, 78: 20, 90: 58, 102: 36, 114: 20, 122: 68, 126: 19}}
INFO 05-03 18:02:14.425562.425562 lmp.py:1005] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.498ms | allocate_experts_across_cpu_gpu: 0.440ms
INFO 05-03 18:02:14.425043.425043 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.580352783203125e-05 seconds
INFO 05-03 18:02:14.426203.426203 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005846023559570312 seconds
INFO 05-03 18:02:14.434797.434797 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.007016181945800781 seconds
INFO 05-03 18:02:14.435624.435624 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010287761688232422 seconds
INFO 05-03 18:02:14.436808.436808 mlpmodule.py:2707] [fused_experts] gmm total=1.644ms E=32 S=706 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.438517.438517 mlpmodule.py:2707] [fused_experts] gmm total=1.691ms E=32 S=1081 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.440002.440002 mlpmodule.py:2707] [fused_experts] gmm total=1.491ms E=32 S=1310 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.442156.442156 mlpmodule.py:2707] [fused_experts] gmm total=1.496ms E=32 S=999 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.442994.442994 lmp.py:1149] [layer_moe_fused] experts compute time: 0.0076029300689697266 seconds
INFO 05-03 18:02:14.443608.443608 lmp.py:1160] [layer_moe_fused] to time: 3.170967102050781e-05 seconds
INFO 05-03 18:02:14.443641.443641 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002455711364746094 seconds
DEBUG 05-03 18:02:14.443417.443417 cuda_h.py:27] end *layer_moe_fused cost 19.162 ms
DEBUG 05-03 18:02:14.443644.443644 cuda_h.py:27] end prefill_layer cost 21.764 ms
DEBUG 05-03 18:02:14.443553.443553 lmp.py:711] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-03 18:02:14.443323.443323 lmp.py:675] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-03 18:02:14.446736.446736 cuda_h.py:27] end *sagl cost 2.255 ms
experts_cpu_alloc {'expert_ids': [35, 63, 99, 39, 43, 27, 31, 59, 103, 11, 51, 47, 87, 119, 115, 88, 120, 48, 76, 12, 124, 28, 40, 36, 116, 20, 84, 112, 44, 24, 9, 121, 109, 125, 33, 21, 73, 105, 93, 69, 25, 113, 45, 41, 101, 117, 98, 122, 30, 54, 82, 94, 50, 114, 106, 118, 90, 62, 102, 58, 18, 46], 'token_total': 454, 'token_per_expert': {35: 2, 63: 2, 99: 2, 39: 3, 43: 3, 27: 4, 31: 4, 59: 4, 103: 4, 11: 6, 51: 7, 47: 10, 87: 13, 119: 16, 115: 20, 88: 1, 120: 1, 48: 3, 76: 3, 12: 4, 124: 4, 28: 6, 40: 7, 36: 9, 116: 10, 20: 13, 84: 13, 112: 13, 44: 20, 24: 28, 9: 2, 121: 2, 109: 3, 125: 3, 33: 4, 21: 5, 73: 5, 105: 6, 93: 9, 69: 10, 25: 12, 113: 14, 45: 15, 41: 16, 101: 16, 117: 16, 98: 1, 122: 1, 30: 2, 54: 2, 82: 2, 94: 2, 50: 3, 114: 3, 106: 6, 118: 6, 90: 7, 62: 8, 102: 8, 58: 9, 18: 10, 46: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 55, 67, 71, 75, 79, 83, 91, 107, 111, 123, 127], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 990, 'token_per_expert': {3: 131, 7: 197, 15: 46, 19: 50, 23: 21, 55: 30, 67: 70, 71: 32, 75: 30, 79: 48, 83: 30, 91: 34, 107: 83, 111: 30, 123: 28, 127: 130}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 32, 52, 56, 60, 64, 68, 72, 80, 92, 96, 100, 104], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 1086, 'token_per_expert': {0: 133, 4: 139, 8: 44, 16: 72, 32: 93, 52: 32, 56: 33, 60: 67, 64: 56, 68: 77, 72: 42, 80: 30, 92: 41, 96: 109, 100: 70, 104: 48}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 29, 49, 53, 57, 61, 65, 77, 81, 85, 89, 97], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 776, 'token_per_expert': {1: 145, 5: 138, 13: 62, 17: 25, 29: 37, 49: 18, 53: 30, 57: 17, 61: 19, 65: 40, 77: 127, 81: 41, 85: 22, 89: 25, 97: 30}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 26, 34, 38, 66, 70, 74, 78, 86, 110, 126], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 790, 'token_per_expert': {2: 161, 6: 142, 10: 40, 14: 28, 22: 26, 26: 38, 34: 19, 38: 16, 66: 34, 70: 11, 74: 40, 78: 16, 86: 153, 110: 47, 126: 19}}
INFO 05-03 18:02:14.447366.447366 lmp.py:1005] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.497ms | allocate_experts_across_cpu_gpu: 0.442ms
INFO 05-03 18:02:14.447246.447246 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 7.653236389160156e-05 seconds
INFO 05-03 18:02:14.448141.448141 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006108283996582031 seconds
INFO 05-03 18:02:14.455507.455507 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.006726980209350586 seconds
INFO 05-03 18:02:14.456890.456890 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010199546813964844 seconds
INFO 05-03 18:02:14.458836.458836 mlpmodule.py:2707] [fused_experts] gmm total=1.643ms E=32 S=1090 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.460745.460745 mlpmodule.py:2707] [fused_experts] gmm total=1.558ms E=32 S=1221 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.462526.462526 mlpmodule.py:2707] [fused_experts] gmm total=1.438ms E=32 S=914 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.464976.464976 mlpmodule.py:2707] [fused_experts] gmm total=1.463ms E=32 S=871 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.464092.464092 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007338047027587891 seconds
INFO 05-03 18:02:14.464308.464308 lmp.py:1160] [layer_moe_fused] to time: 3.075599670410156e-05 seconds
INFO 05-03 18:02:14.464084.464084 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002644062042236328 seconds
DEBUG 05-03 18:02:14.465109.465109 cuda_h.py:27] end *layer_moe_fused cost 18.590 ms
DEBUG 05-03 18:02:14.465290.465290 cuda_h.py:27] end prefill_layer cost 21.183 ms
DEBUG 05-03 18:02:14.465345.465345 lmp.py:711] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-03 18:02:14.465387.465387 lmp.py:675] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-03 18:02:14.467060.467060 cuda_h.py:27] end *sagl cost 2.317 ms
experts_cpu_alloc {'expert_ids': [51, 67, 95, 19, 23, 47, 27, 71, 127, 39, 31, 103, 119, 115, 87, 56, 8, 72, 108, 28, 32, 84, 104, 48, 24, 44, 76, 112, 40, 68, 52, 97, 125, 65, 73, 37, 81, 89, 113, 61, 101, 69, 109, 41, 49, 13, 26, 42, 34, 110, 122, 90, 22, 14, 30, 74, 66, 114, 18, 86, 98, 126, 38], 'token_total': 557, 'token_per_expert': {51: 2, 67: 2, 95: 2, 19: 3, 23: 3, 47: 7, 27: 8, 71: 9, 127: 10, 39: 11, 31: 12, 103: 12, 119: 13, 115: 17, 87: 18, 56: 1, 8: 4, 72: 4, 108: 4, 28: 5, 32: 5, 84: 9, 104: 9, 48: 12, 24: 13, 44: 13, 76: 14, 112: 14, 40: 15, 68: 15, 52: 18, 97: 3, 125: 5, 65: 8, 73: 8, 37: 9, 81: 9, 89: 10, 113: 10, 61: 11, 101: 12, 69: 14, 109: 15, 41: 16, 49: 18, 13: 22, 26: 1, 42: 1, 34: 2, 110: 2, 122: 2, 90: 3, 22: 4, 14: 5, 30: 6, 74: 7, 66: 8, 114: 8, 18: 9, 86: 9, 98: 10, 126: 10, 38: 16}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 35, 43, 55, 59, 63, 79, 83, 91, 99, 107, 111, 123], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 1043, 'token_per_expert': {3: 144, 7: 250, 11: 32, 15: 22, 35: 138, 43: 40, 55: 94, 59: 29, 63: 45, 79: 37, 83: 55, 91: 52, 99: 36, 107: 20, 111: 30, 123: 19}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 20, 36, 60, 64, 80, 88, 92, 96, 100, 116, 120, 124], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 816, 'token_per_expert': {0: 135, 4: 134, 12: 35, 16: 20, 20: 24, 36: 55, 60: 27, 64: 18, 80: 26, 88: 50, 92: 41, 96: 92, 100: 35, 116: 57, 120: 42, 124: 25}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 25, 29, 33, 45, 53, 57, 77, 85, 93, 117, 121], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 804, 'token_per_expert': {1: 146, 5: 133, 9: 31, 17: 37, 21: 38, 25: 34, 29: 26, 33: 95, 45: 27, 53: 23, 57: 24, 77: 29, 85: 35, 93: 58, 117: 45, 121: 23}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 46, 50, 54, 58, 62, 70, 78, 82, 94, 102, 106, 118], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 876, 'token_per_expert': {2: 140, 6: 132, 10: 45, 46: 26, 50: 53, 54: 39, 58: 23, 62: 31, 70: 121, 78: 42, 82: 24, 94: 28, 102: 75, 106: 21, 118: 76}}
INFO 05-03 18:02:14.469545.469545 lmp.py:1005] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.495ms | allocate_experts_across_cpu_gpu: 0.443ms
INFO 05-03 18:02:14.469411.469411 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.747245788574219e-05 seconds
INFO 05-03 18:02:14.469557.469557 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005872249603271484 seconds
INFO 05-03 18:02:14.477714.477714 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0066852569580078125 seconds
INFO 05-03 18:02:14.478835.478835 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010609626770019531 seconds
INFO 05-03 18:02:14.480491.480491 mlpmodule.py:2707] [fused_experts] gmm total=1.881ms E=32 S=1172 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.482888.482888 mlpmodule.py:2707] [fused_experts] gmm total=1.499ms E=32 S=971 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.483722.483722 mlpmodule.py:2707] [fused_experts] gmm total=1.453ms E=32 S=974 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.485405.485405 mlpmodule.py:2707] [fused_experts] gmm total=1.483ms E=32 S=979 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.486395.486395 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007563591003417969 seconds
INFO 05-03 18:02:14.486412.486412 lmp.py:1160] [layer_moe_fused] to time: 3.075599670410156e-05 seconds
INFO 05-03 18:02:14.486438.486438 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00020194053649902344 seconds
DEBUG 05-03 18:02:14.486278.486278 cuda_h.py:27] end *layer_moe_fused cost 18.686 ms
DEBUG 05-03 18:02:14.486936.486936 cuda_h.py:27] end prefill_layer cost 21.331 ms
DEBUG 05-03 18:02:14.486844.486844 lmp.py:711] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-03 18:02:14.486026.486026 lmp.py:675] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-03 18:02:14.490150.490150 cuda_h.py:27] end *sagl cost 3.139 ms
experts_cpu_alloc {'expert_ids': [39, 127, 75, 91, 123, 87, 99, 23, 47, 19, 63, 111, 51, 103, 67, 55, 16, 56, 88, 20, 40, 8, 36, 52, 100, 108, 96, 32, 28, 72, 76, 112, 37, 57, 73, 81, 9, 89, 53, 45, 117, 93, 125, 101, 65, 69, 77, 21, 18, 14, 114, 94, 66, 86, 78, 126, 54, 62, 122, 26, 50, 58, 118], 'token_total': 524, 'token_per_expert': {39: 1, 127: 2, 75: 3, 91: 3, 123: 4, 87: 8, 99: 9, 23: 10, 47: 12, 19: 15, 63: 17, 111: 17, 51: 19, 103: 19, 67: 21, 55: 26, 16: 1, 56: 1, 88: 1, 20: 2, 40: 3, 8: 4, 36: 5, 52: 5, 100: 5, 108: 7, 96: 8, 32: 9, 28: 11, 72: 11, 76: 12, 112: 12, 37: 1, 57: 1, 73: 2, 81: 2, 9: 4, 89: 4, 53: 5, 45: 8, 117: 8, 93: 9, 125: 11, 101: 12, 65: 15, 69: 15, 77: 16, 21: 17, 18: 1, 14: 2, 114: 2, 94: 3, 66: 4, 86: 5, 78: 8, 126: 8, 54: 9, 62: 10, 122: 10, 26: 12, 50: 12, 58: 12, 118: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 27, 31, 35, 43, 59, 71, 79, 83, 95, 107, 115, 119], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 1026, 'token_per_expert': {3: 177, 7: 136, 11: 37, 15: 28, 27: 50, 31: 54, 35: 32, 43: 37, 59: 56, 71: 69, 79: 31, 83: 53, 95: 86, 107: 45, 115: 79, 119: 56}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 24, 44, 48, 60, 64, 68, 80, 84, 92, 104, 116, 120, 124], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 666, 'token_per_expert': {0: 169, 4: 146, 12: 52, 24: 18, 44: 30, 48: 16, 60: 38, 64: 24, 68: 16, 80: 19, 84: 16, 92: 20, 104: 60, 116: 13, 120: 13, 124: 16}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 25, 29, 33, 41, 49, 61, 85, 97, 105, 109, 113, 121], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 1041, 'token_per_expert': {1: 142, 5: 275, 13: 28, 17: 31, 25: 109, 29: 17, 33: 18, 41: 137, 49: 27, 61: 38, 85: 54, 97: 21, 105: 42, 109: 18, 113: 47, 121: 37}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 30, 34, 38, 42, 46, 70, 74, 90, 98, 102, 106, 110], 'expert_count': 16, 'target_expert_count': 16, 'token_total': 839, 'token_per_expert': {2: 162, 6: 148, 10: 36, 22: 36, 30: 20, 34: 35, 38: 18, 42: 28, 46: 30, 70: 80, 74: 14, 90: 27, 98: 68, 102: 31, 106: 25, 110: 81}}
INFO 05-03 18:02:14.491562.491562 lmp.py:1005] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.503ms | allocate_experts_across_cpu_gpu: 0.453ms
INFO 05-03 18:02:14.491666.491666 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.723403930664062e-05 seconds
INFO 05-03 18:02:14.492382.492382 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006773471832275391 seconds
INFO 05-03 18:02:14.499194.499194 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.006838560104370117 seconds
INFO 05-03 18:02:14.500288.500288 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010688304901123047 seconds
INFO 05-03 18:02:14.502879.502879 mlpmodule.py:2707] [fused_experts] gmm total=1.903ms E=32 S=1212 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.504542.504542 mlpmodule.py:2707] [fused_experts] gmm total=1.498ms E=32 S=763 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.506342.506342 mlpmodule.py:2707] [fused_experts] gmm total=1.637ms E=32 S=1171 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.508622.508622 mlpmodule.py:2707] [fused_experts] gmm total=1.467ms E=32 S=950 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:14.508798.508798 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007809638977050781 seconds
INFO 05-03 18:02:14.508338.508338 lmp.py:1160] [layer_moe_fused] to time: 3.0994415283203125e-05 seconds
INFO 05-03 18:02:14.509363.509363 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00020265579223632812 seconds
DEBUG 05-03 18:02:14.509648.509648 cuda_h.py:27] end *layer_moe_fused cost 19.138 ms
DEBUG 05-03 18:02:14.509113.509113 cuda_h.py:27] end prefill_layer cost 22.570 ms
DEBUG 05-03 18:02:14.509260.509260 lmp.py:711] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-03 18:02:14.509067.509067 cuda_h.py:27] end prefill cost 1292.677 ms
INFO 05-03 18:02:14.509410.509410 lmp.py:713] prefill time: 1.2928223609924316 seconds
Time taken: 5.57016247138381 seconds
generate input ids cost 0.04287362098693848 s
DEBUG 05-03 18:02:17.210588.210588 cuda_h.py:27] end generate_input_ids cost 2556.646 ms
DEBUG 05-03 18:02:17.210560.210560 cuda_h.py:27] end init_cache cost 0.035 ms
INFO 05-03 18:02:17.222171.222171 lmp.py:1985] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6604693444, 'cuda:1': 12877692928, 'cuda:2': 12877692928, 'cuda:3': 12877692928} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.728094430516583, 'cuda:1': 0.4699814963667064, 'cuda:2': 0.4699814963667064, 'cuda:3': 0.4699814963667064}
INFO 05-03 18:02:17.222638.222638 lmp.py:2003] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.222607.222607 lmp.py:2003] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.222184.222184 lmp.py:2003] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.222900.222900 lmp.py:2003] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.223750.223750 lmp.py:2003] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.223239.223239 lmp.py:2003] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.223082.223082 lmp.py:2003] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224129.224129 lmp.py:2003] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224945.224945 lmp.py:2003] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224500.224500 lmp.py:2003] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224362.224362 lmp.py:2003] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224945.224945 lmp.py:2003] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224999.224999 lmp.py:2003] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224204.224204 lmp.py:2003] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224589.224589 lmp.py:2003] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224768.224768 lmp.py:2003] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.224200.224200 lmp.py:2003] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.225020.225020 lmp.py:2003] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.225167.225167 lmp.py:2003] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.225701.225701 lmp.py:2003] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.225905.225905 lmp.py:2003] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.225860.225860 lmp.py:2003] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.225020.225020 lmp.py:2003] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.225167.225167 lmp.py:2003] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.225126.225126 lmp.py:2003] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.226709.226709 lmp.py:2003] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.226379.226379 lmp.py:2003] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.226054.226054 lmp.py:2003] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.226055.226055 lmp.py:2003] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-03 18:02:17.226054.226054 lmp.py:2003] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-03 18:02:17.515462.515462 cuda_h.py:27] end init_loading_placement cost 304.952 ms
DEBUG 05-03 18:02:17.515043.515043 sllm_store_c.py:27] get device uuid map
DEBUG 05-03 18:02:17.515309.515309 sllm_store_c.py:29] call client load into gpu
DEBUG 05-03 18:02:17 client.py:72] load_into_gpu: gemma4-26B-A4B, 0ec12a55-d378-42da-84f5-27546cc73009
INFO 05-03 18:02:17 client.py:135] Model loaded: gemma4-26B-A4B, 0ec12a55-d378-42da-84f5-27546cc73009
INFO 05-03 18:02:17 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 0ec12a55-d378-42da-84f5-27546cc73009
INFO 05-03 18:02:18 client.py:212] Model loaded
DEBUG 05-03 18:02:18.044505.044505 cuda_h.py:27] end init_general_sagl_loading_async cost 528.832 ms
DEBUG 05-03 18:02:18.064136.064136 sllm_store_c.py:27] get device uuid map
DEBUG 05-03 18:02:18.064855.064855 sllm_store_c.py:29] call client load into gpu
DEBUG 05-03 18:02:18 client.py:72] load_into_gpu: gemma4-26B-A4B, 7ccaec92-dc78-4e36-a05e-d4dee4f1e8f1
INFO 05-03 18:02:18 client.py:135] Model loaded: gemma4-26B-A4B, 7ccaec92-dc78-4e36-a05e-d4dee4f1e8f1
DEBUG 05-03 18:02:18.191365.191365 cuda_h.py:27] end init_experts_loading_async cost 146.775 ms
INFO 05-03 18:02:18.234506.234506 lmp.py:2506] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-03 18:02:18.333223.333223 cuda_h.py:27] end restore_state_dict cost 98.737 ms
DEBUG 05-03 18:02:18.334107.334107 cuda_h.py:27] end init_inputs_tokens cost 0.705 ms
DEBUG 05-03 18:02:18.334136.334136 lmp.py:675] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-03 18:02:18.339597.339597 cuda_h.py:27] end *sagl cost 4.828 ms
experts_cpu_alloc {'expert_ids': [11, 27, 59, 83, 99, 31, 67, 19, 43, 71, 127, 79, 87, 23, 4, 100, 72, 84, 8, 92, 120, 20, 108, 28, 44, 80, 45, 61, 85, 101, 109, 49, 65, 93, 17, 29, 77, 5, 69, 37, 9, 86, 14, 6, 94, 102, 30, 106, 114, 10, 2, 38, 118, 70], 'token_total': 499, 'token_per_expert': {11: 2, 27: 2, 59: 9, 83: 9, 99: 10, 31: 11, 67: 12, 19: 14, 43: 15, 71: 15, 127: 21, 79: 22, 87: 22, 23: 25, 4: 2, 100: 2, 72: 3, 84: 3, 8: 4, 92: 8, 120: 9, 20: 11, 108: 11, 28: 15, 44: 25, 80: 29, 45: 1, 61: 1, 85: 1, 101: 1, 109: 2, 49: 3, 65: 4, 93: 5, 17: 6, 29: 6, 77: 7, 5: 10, 69: 15, 37: 16, 9: 17, 86: 1, 14: 2, 6: 3, 94: 3, 102: 5, 30: 6, 106: 7, 114: 7, 10: 8, 2: 9, 38: 11, 118: 13, 70: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 47, 51, 55, 63, 75, 91, 103, 107, 111, 115, 123], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 987, 'token_per_expert': {3: 64, 7: 64, 39: 137, 47: 209, 51: 32, 55: 105, 63: 45, 75: 29, 91: 66, 103: 88, 107: 37, 111: 28, 115: 29, 123: 54}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 24, 32, 48, 52, 60, 64, 68, 76, 104, 112, 116, 124], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 972, 'token_per_expert': {0: 79, 16: 47, 24: 30, 32: 53, 48: 48, 52: 69, 60: 39, 64: 45, 68: 157, 76: 53, 104: 41, 112: 45, 116: 88, 124: 178}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 21, 25, 33, 41, 53, 73, 89, 105, 113, 117, 121, 125], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 774, 'token_per_expert': {1: 89, 13: 18, 21: 22, 25: 18, 33: 155, 41: 22, 53: 172, 73: 29, 89: 17, 105: 56, 113: 37, 117: 23, 121: 96, 125: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 22, 26, 46, 50, 54, 58, 74, 78, 90, 110, 122, 126], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 864, 'token_per_expert': {18: 43, 22: 98, 26: 52, 46: 84, 50: 71, 54: 52, 58: 24, 74: 71, 78: 37, 90: 148, 110: 38, 122: 71, 126: 75}}
INFO 05-03 18:02:18.340113.340113 lmp.py:1005] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.663ms | allocate_experts_across_cpu_gpu: 0.233ms
INFO 05-03 18:02:18.340548.340548 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 3.838539123535156e-05 seconds
INFO 05-03 18:02:18.342163.342163 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018754005432128906 seconds
INFO 05-03 18:02:18.356059.356059 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012967109680175781 seconds
INFO 05-03 18:02:18.357345.357345 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00127410888671875 seconds
INFO 05-03 18:02:18.359788.359788 mlpmodule.py:2707] [fused_experts] gmm total=2.022ms E=32 S=1176 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.363914.363914 mlpmodule.py:2707] [fused_experts] gmm total=2.215ms E=32 S=1094 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.365433.365433 mlpmodule.py:2707] [fused_experts] gmm total=2.311ms E=32 S=869 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.367725.367725 mlpmodule.py:2707] [fused_experts] gmm total=1.845ms E=32 S=957 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.368643.368643 lmp.py:1149] [layer_moe_fused] experts compute time: 0.010385990142822266 seconds
INFO 05-03 18:02:18.368972.368972 lmp.py:1160] [layer_moe_fused] to time: 3.719329833984375e-05 seconds
INFO 05-03 18:02:18.368211.368211 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00026726722717285156 seconds
DEBUG 05-03 18:02:18.369125.369125 cuda_h.py:27] end *layer_moe_fused cost 29.527 ms
DEBUG 05-03 18:02:18.369690.369690 cuda_h.py:27] end prefill_layer cost 34.538 ms
DEBUG 05-03 18:02:18.369837.369837 lmp.py:711] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-03 18:02:18.369417.369417 lmp.py:675] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-03 18:02:18.372406.372406 cuda_h.py:27] end *sagl cost 3.296 ms
experts_cpu_alloc {'expert_ids': [43, 95, 107, 55, 63, 11, 27, 83, 51, 39, 79, 115, 111, 15, 48, 100, 24, 124, 56, 36, 40, 84, 116, 112, 108, 16, 25, 69, 65, 17, 29, 33, 117, 61, 73, 41, 81, 53, 121, 77, 89, 93, 21, 102, 126, 54, 58, 62, 86, 74, 66, 118, 78, 110, 26, 70, 14, 50, 2, 34], 'token_total': 552, 'token_per_expert': {43: 1, 95: 1, 107: 1, 55: 2, 63: 3, 11: 4, 27: 4, 83: 4, 51: 5, 39: 6, 79: 7, 115: 8, 111: 9, 15: 11, 48: 1, 100: 1, 24: 4, 124: 4, 56: 5, 36: 8, 40: 8, 84: 11, 116: 11, 112: 13, 108: 17, 16: 18, 25: 2, 69: 2, 65: 3, 17: 4, 29: 4, 33: 5, 117: 5, 61: 6, 73: 6, 41: 7, 81: 10, 53: 12, 121: 14, 77: 15, 89: 17, 93: 20, 21: 21, 102: 1, 126: 1, 54: 2, 58: 2, 62: 7, 86: 10, 74: 12, 66: 13, 118: 14, 78: 16, 110: 16, 26: 17, 70: 21, 14: 22, 50: 23, 2: 26, 34: 29}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 31, 35, 47, 59, 67, 71, 87, 99, 103, 119, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 703, 'token_per_expert': {3: 26, 7: 60, 23: 31, 31: 17, 35: 23, 47: 133, 59: 26, 67: 99, 71: 13, 87: 35, 99: 72, 103: 15, 119: 32, 123: 20, 127: 101}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 28, 52, 60, 64, 68, 72, 76, 80, 96, 104, 120], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 798, 'token_per_expert': {0: 31, 4: 44, 8: 152, 20: 21, 28: 107, 52: 55, 60: 23, 64: 37, 68: 49, 72: 37, 76: 21, 80: 128, 96: 37, 104: 36, 120: 20}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 37, 45, 49, 57, 85, 97, 101, 105, 109, 113, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 877, 'token_per_expert': {1: 105, 5: 223, 9: 44, 13: 56, 37: 28, 45: 49, 49: 27, 57: 39, 85: 47, 97: 68, 101: 33, 105: 22, 109: 57, 113: 32, 125: 47}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 18, 22, 30, 38, 42, 46, 82, 90, 94, 98, 106, 114, 122], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1166, 'token_per_expert': {6: 38, 10: 181, 18: 45, 22: 93, 30: 80, 38: 57, 42: 63, 46: 89, 82: 126, 90: 106, 94: 51, 98: 42, 106: 45, 114: 32, 122: 118}}
INFO 05-03 18:02:18.373329.373329 lmp.py:1005] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.546ms | allocate_experts_across_cpu_gpu: 0.434ms
INFO 05-03 18:02:18.373811.373811 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.389617919921875e-05 seconds
INFO 05-03 18:02:18.374006.374006 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006911754608154297 seconds
INFO 05-03 18:02:18.390019.390019 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014576911926269531 seconds
INFO 05-03 18:02:18.391087.391087 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001081228256225586 seconds
INFO 05-03 18:02:18.393183.393183 mlpmodule.py:2707] [fused_experts] gmm total=1.776ms E=32 S=769 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.394601.394601 mlpmodule.py:2707] [fused_experts] gmm total=1.490ms E=32 S=899 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.396408.396408 mlpmodule.py:2707] [fused_experts] gmm total=1.638ms E=32 S=1030 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.398499.398499 mlpmodule.py:2707] [fused_experts] gmm total=1.602ms E=32 S=1398 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.399087.399087 lmp.py:1149] [layer_moe_fused] experts compute time: 0.008619308471679688 seconds
INFO 05-03 18:02:18.400178.400178 lmp.py:1160] [layer_moe_fused] to time: 3.361701965332031e-05 seconds
INFO 05-03 18:02:18.400251.400251 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002789497375488281 seconds
DEBUG 05-03 18:02:18.400556.400556 cuda_h.py:27] end *layer_moe_fused cost 27.939 ms
DEBUG 05-03 18:02:18.400214.400214 cuda_h.py:27] end prefill_layer cost 31.521 ms
DEBUG 05-03 18:02:18.400215.400215 lmp.py:711] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-03 18:02:18.400796.400796 lmp.py:675] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-03 18:02:18.404852.404852 cuda_h.py:27] end *sagl cost 3.684 ms
experts_cpu_alloc {'expert_ids': [27, 59, 95, 15, 75, 39, 111, 63, 79, 119, 43, 55, 67, 83, 103, 112, 40, 72, 12, 20, 36, 76, 116, 32, 100, 60, 64, 96, 8, 29, 49, 117, 85, 101, 37, 45, 89, 121, 33, 93, 21, 77, 97, 53, 61, 26, 46, 50, 86, 98, 10, 58, 94, 14, 66, 106, 102, 30], 'token_total': 268, 'token_per_expert': {27: 1, 59: 1, 95: 1, 15: 3, 75: 3, 39: 5, 111: 5, 63: 6, 79: 7, 119: 7, 43: 10, 55: 10, 67: 11, 83: 12, 103: 15, 112: 1, 40: 2, 72: 2, 12: 3, 20: 3, 36: 3, 76: 4, 116: 4, 32: 5, 100: 5, 60: 8, 64: 8, 96: 9, 8: 10, 29: 1, 49: 1, 117: 1, 85: 2, 101: 2, 37: 3, 45: 3, 89: 3, 121: 3, 33: 5, 93: 5, 21: 7, 77: 7, 97: 8, 53: 9, 61: 9, 26: 1, 46: 1, 50: 1, 86: 1, 98: 1, 10: 2, 58: 2, 94: 2, 14: 3, 66: 3, 106: 4, 102: 5, 30: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 51, 71, 91, 99, 107, 115, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1046, 'token_per_expert': {3: 270, 7: 286, 11: 72, 19: 58, 23: 16, 31: 17, 35: 26, 51: 47, 71: 26, 91: 44, 99: 34, 107: 34, 115: 15, 123: 53, 127: 48}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 24, 28, 44, 48, 52, 56, 68, 80, 84, 104, 108, 124], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 972, 'token_per_expert': {0: 258, 4: 256, 16: 20, 24: 24, 28: 10, 44: 17, 48: 67, 52: 11, 56: 10, 68: 15, 80: 19, 84: 43, 104: 42, 108: 168, 124: 12}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 41, 57, 65, 69, 73, 81, 105, 109, 125], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 916, 'token_per_expert': {1: 305, 5: 256, 9: 44, 13: 24, 25: 20, 41: 90, 57: 24, 65: 44, 69: 9, 73: 13, 81: 18, 105: 20, 109: 20, 125: 29}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 42, 54, 62, 70, 74, 78, 82, 110, 118, 122, 126], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 894, 'token_per_expert': {2: 256, 6: 256, 18: 43, 42: 9, 54: 109, 62: 34, 70: 29, 74: 10, 78: 16, 82: 18, 110: 20, 118: 22, 122: 42, 126: 30}}
INFO 05-03 18:02:18.405205.405205 lmp.py:1005] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.549ms | allocate_experts_across_cpu_gpu: 0.432ms
INFO 05-03 18:02:18.406396.406396 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.984306335449219e-05 seconds
INFO 05-03 18:02:18.406344.406344 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006458759307861328 seconds
INFO 05-03 18:02:18.420464.420464 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013079643249511719 seconds
INFO 05-03 18:02:18.421082.421082 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010759830474853516 seconds
INFO 05-03 18:02:18.423991.423991 mlpmodule.py:2707] [fused_experts] gmm total=1.925ms E=32 S=1143 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.425129.425129 mlpmodule.py:2707] [fused_experts] gmm total=1.837ms E=32 S=1039 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.427950.427950 mlpmodule.py:2707] [fused_experts] gmm total=1.621ms E=32 S=985 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.429002.429002 mlpmodule.py:2707] [fused_experts] gmm total=1.414ms E=32 S=929 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.429675.429675 lmp.py:1149] [layer_moe_fused] experts compute time: 0.008150100708007812 seconds
INFO 05-03 18:02:18.430561.430561 lmp.py:1160] [layer_moe_fused] to time: 3.4332275390625e-05 seconds
INFO 05-03 18:02:18.430800.430800 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002608299255371094 seconds
DEBUG 05-03 18:02:18.430959.430959 cuda_h.py:27] end *layer_moe_fused cost 25.871 ms
DEBUG 05-03 18:02:18.430948.430948 cuda_h.py:27] end prefill_layer cost 29.852 ms
DEBUG 05-03 18:02:18.430234.430234 lmp.py:711] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-03 18:02:18.431844.431844 lmp.py:675] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-03 18:02:18.434783.434783 cuda_h.py:27] end *sagl cost 3.380 ms
experts_cpu_alloc {'expert_ids': [23, 63, 99, 103, 19, 115, 59, 55, 79, 15, 67, 43, 12, 92, 112, 16, 36, 24, 28, 80, 124, 8, 48, 44, 52, 84, 116, 108, 60, 17, 29, 33, 49, 41, 53, 65, 89, 81, 21, 61, 13, 18, 90, 98, 118, 62, 82, 58, 46, 102, 126, 94, 110, 114, 14, 42], 'token_total': 265, 'token_per_expert': {23: 1, 63: 1, 99: 1, 103: 1, 19: 3, 115: 4, 59: 5, 55: 6, 79: 6, 15: 7, 67: 8, 43: 9, 12: 1, 92: 1, 112: 1, 16: 2, 36: 2, 24: 3, 28: 3, 80: 3, 124: 3, 8: 4, 48: 4, 44: 5, 52: 5, 84: 5, 116: 5, 108: 9, 60: 12, 17: 1, 29: 1, 33: 1, 49: 1, 41: 3, 53: 3, 65: 3, 89: 3, 81: 4, 21: 5, 61: 5, 13: 9, 18: 1, 90: 1, 98: 1, 118: 1, 62: 2, 82: 4, 58: 5, 46: 8, 102: 8, 126: 8, 94: 11, 110: 12, 114: 13, 14: 14, 42: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 31, 39, 51, 71, 75, 83, 95, 107, 111, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 877, 'token_per_expert': {3: 265, 7: 257, 11: 63, 27: 28, 31: 16, 39: 30, 51: 12, 71: 46, 75: 14, 83: 32, 95: 35, 107: 25, 111: 23, 123: 14, 127: 17}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 32, 40, 56, 64, 68, 72, 76, 88, 96, 100, 104, 120], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 829, 'token_per_expert': {0: 281, 4: 272, 32: 17, 40: 49, 56: 15, 64: 23, 68: 31, 72: 23, 76: 23, 88: 25, 96: 21, 100: 14, 104: 22, 120: 13}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 37, 69, 73, 77, 85, 93, 97, 101, 113, 117, 121], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1000, 'token_per_expert': {1: 258, 5: 275, 9: 61, 37: 15, 69: 62, 73: 26, 77: 22, 85: 30, 93: 59, 97: 48, 101: 65, 113: 9, 117: 48, 121: 22}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 26, 34, 50, 54, 66, 70, 74, 78, 86, 122], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1125, 'token_per_expert': {2: 263, 6: 262, 10: 73, 22: 68, 26: 24, 34: 26, 50: 79, 54: 47, 66: 17, 70: 28, 74: 26, 78: 102, 86: 49, 122: 61}}
INFO 05-03 18:02:18.435679.435679 lmp.py:1005] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.533ms | allocate_experts_across_cpu_gpu: 0.428ms
INFO 05-03 18:02:18.435015.435015 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.198883056640625e-05 seconds
INFO 05-03 18:02:18.436288.436288 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006375312805175781 seconds
INFO 05-03 18:02:18.450883.450883 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013220548629760742 seconds
INFO 05-03 18:02:18.451427.451427 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010576248168945312 seconds
INFO 05-03 18:02:18.453337.453337 mlpmodule.py:2707] [fused_experts] gmm total=1.958ms E=32 S=929 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.455801.455801 mlpmodule.py:2707] [fused_experts] gmm total=1.840ms E=32 S=897 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.457408.457408 mlpmodule.py:2707] [fused_experts] gmm total=1.370ms E=32 S=1039 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.459221.459221 mlpmodule.py:2707] [fused_experts] gmm total=1.607ms E=32 S=1231 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.459432.459432 lmp.py:1149] [layer_moe_fused] experts compute time: 0.008172273635864258 seconds
INFO 05-03 18:02:18.460310.460310 lmp.py:1160] [layer_moe_fused] to time: 3.24249267578125e-05 seconds
INFO 05-03 18:02:18.460118.460118 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00025582313537597656 seconds
DEBUG 05-03 18:02:18.460770.460770 cuda_h.py:27] end *layer_moe_fused cost 26.029 ms
DEBUG 05-03 18:02:18.460236.460236 cuda_h.py:27] end prefill_layer cost 29.712 ms
DEBUG 05-03 18:02:18.460522.460522 lmp.py:711] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-03 18:02:18.460863.460863 lmp.py:675] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-03 18:02:18.464270.464270 cuda_h.py:27] end *sagl cost 3.494 ms
experts_cpu_alloc {'expert_ids': [35, 95, 103, 71, 119, 11, 107, 123, 75, 31, 27, 87, 19, 15, 124, 12, 76, 80, 24, 116, 28, 48, 16, 120, 40, 56, 108, 84, 9, 45, 61, 69, 37, 77, 117, 125, 121, 97, 13, 89, 25, 101, 90, 38, 42, 58, 62, 94, 114, 110, 66, 34], 'token_total': 269, 'token_per_expert': {35: 1, 95: 1, 103: 1, 71: 2, 119: 2, 11: 3, 107: 3, 123: 4, 75: 5, 31: 9, 27: 10, 87: 10, 19: 11, 15: 14, 124: 1, 12: 2, 76: 2, 80: 2, 24: 3, 116: 3, 28: 4, 48: 4, 16: 5, 120: 5, 40: 7, 56: 8, 108: 9, 84: 14, 9: 1, 45: 1, 61: 1, 69: 1, 37: 2, 77: 2, 117: 2, 125: 3, 121: 4, 97: 6, 13: 7, 89: 7, 25: 8, 101: 8, 90: 1, 38: 2, 42: 2, 58: 2, 62: 2, 94: 6, 114: 7, 110: 9, 66: 19, 34: 21}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 43, 47, 51, 59, 63, 67, 79, 83, 91, 111, 115], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1121, 'token_per_expert': {3: 275, 7: 269, 39: 31, 43: 55, 47: 37, 51: 126, 59: 17, 63: 21, 67: 59, 79: 15, 83: 58, 91: 44, 111: 14, 115: 100}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 32, 36, 44, 52, 60, 64, 92, 96, 104], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 973, 'token_per_expert': {0: 267, 4: 299, 8: 24, 20: 53, 32: 28, 36: 30, 44: 16, 52: 30, 60: 25, 64: 41, 92: 43, 96: 41, 104: 76}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 29, 49, 53, 57, 65, 73, 81, 85, 93, 105], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 820, 'token_per_expert': {1: 267, 5: 306, 21: 17, 29: 41, 49: 34, 53: 21, 57: 11, 65: 14, 73: 22, 81: 19, 85: 30, 93: 11, 105: 27}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 46, 50, 54, 78, 82, 86, 98, 106, 118], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 913, 'token_per_expert': {2: 257, 6: 277, 22: 39, 26: 36, 46: 34, 50: 27, 54: 21, 78: 33, 82: 29, 86: 24, 98: 32, 106: 61, 118: 43}}
INFO 05-03 18:02:18.465470.465470 lmp.py:1005] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.540ms | allocate_experts_across_cpu_gpu: 0.397ms
INFO 05-03 18:02:18.465845.465845 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.6743621826171875e-05 seconds
INFO 05-03 18:02:18.466798.466798 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006318092346191406 seconds
INFO 05-03 18:02:18.480469.480469 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013322114944458008 seconds
INFO 05-03 18:02:18.481933.481933 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010480880737304688 seconds
INFO 05-03 18:02:18.483499.483499 mlpmodule.py:2707] [fused_experts] gmm total=1.781ms E=32 S=1197 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.485762.485762 mlpmodule.py:2707] [fused_experts] gmm total=1.554ms E=32 S=1042 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.487348.487348 mlpmodule.py:2707] [fused_experts] gmm total=1.516ms E=32 S=873 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.488272.488272 mlpmodule.py:2707] [fused_experts] gmm total=1.351ms E=32 S=984 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.489614.489614 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007621049880981445 seconds
INFO 05-03 18:02:18.489777.489777 lmp.py:1160] [layer_moe_fused] to time: 3.147125244140625e-05 seconds
INFO 05-03 18:02:18.489513.489513 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002720355987548828 seconds
DEBUG 05-03 18:02:18.490066.490066 cuda_h.py:27] end *layer_moe_fused cost 25.466 ms
DEBUG 05-03 18:02:18.490346.490346 cuda_h.py:27] end prefill_layer cost 29.253 ms
DEBUG 05-03 18:02:18.490347.490347 lmp.py:711] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-03 18:02:18.490305.490305 lmp.py:675] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-03 18:02:18.496025.496025 cuda_h.py:27] end *sagl cost 5.936 ms
experts_cpu_alloc {'expert_ids': [31, 47, 43, 51, 95, 11, 67, 103, 79, 19, 119, 55, 75, 23, 48, 92, 68, 116, 12, 56, 76, 8, 36, 16, 40, 72, 80, 112, 65, 85, 29, 125, 17, 81, 37, 77, 97, 14, 34, 38, 54, 66, 86, 98, 126, 62, 102, 110, 82, 90], 'token_total': 256, 'token_per_expert': {31: 1, 47: 1, 43: 4, 51: 4, 95: 4, 11: 5, 67: 6, 103: 6, 79: 7, 19: 10, 119: 10, 55: 12, 75: 14, 23: 15, 48: 1, 92: 1, 68: 2, 116: 3, 12: 4, 56: 4, 76: 4, 8: 5, 36: 6, 16: 7, 40: 8, 72: 8, 80: 8, 112: 8, 65: 1, 85: 1, 29: 2, 125: 2, 17: 3, 81: 7, 37: 8, 77: 8, 97: 20, 14: 1, 34: 1, 38: 1, 54: 1, 66: 1, 86: 2, 98: 2, 126: 2, 62: 3, 102: 4, 110: 5, 82: 6, 90: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 39, 63, 71, 87, 99, 107, 111, 115, 123, 127], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1025, 'token_per_expert': {3: 277, 7: 256, 27: 18, 39: 84, 63: 17, 71: 179, 87: 21, 99: 45, 107: 17, 111: 26, 115: 17, 123: 42, 127: 26}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 28, 32, 44, 52, 64, 88, 96, 104, 120], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 857, 'token_per_expert': {0: 257, 4: 257, 20: 119, 24: 39, 28: 11, 32: 12, 44: 18, 52: 12, 64: 45, 88: 21, 96: 14, 104: 14, 120: 38}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 33, 49, 57, 61, 73, 101, 113, 117], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1078, 'token_per_expert': {1: 277, 5: 256, 13: 74, 21: 29, 33: 42, 49: 52, 57: 27, 61: 79, 73: 25, 101: 112, 113: 28, 117: 77}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 30, 50, 58, 74, 94, 114, 118, 122], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 880, 'token_per_expert': {2: 314, 6: 292, 18: 9, 22: 56, 30: 9, 50: 54, 58: 13, 74: 39, 94: 57, 114: 7, 118: 14, 122: 16}}
INFO 05-03 18:02:18.497906.497906 lmp.py:1005] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.529ms | allocate_experts_across_cpu_gpu: 0.380ms
INFO 05-03 18:02:18.497520.497520 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.3882598876953125e-05 seconds
INFO 05-03 18:02:18.498684.498684 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007998943328857422 seconds
INFO 05-03 18:02:18.511360.511360 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012269258499145508 seconds
INFO 05-03 18:02:18.512678.512678 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001035451889038086 seconds
INFO 05-03 18:02:18.514862.514862 mlpmodule.py:2707] [fused_experts] gmm total=1.634ms E=32 S=1124 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.516682.516682 mlpmodule.py:2707] [fused_experts] gmm total=1.444ms E=32 S=926 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.517795.517795 mlpmodule.py:2707] [fused_experts] gmm total=1.269ms E=32 S=1130 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.519492.519492 mlpmodule.py:2707] [fused_experts] gmm total=1.689ms E=32 S=916 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.519643.519643 lmp.py:1149] [layer_moe_fused] experts compute time: 0.0073659420013427734 seconds
INFO 05-03 18:02:18.520468.520468 lmp.py:1160] [layer_moe_fused] to time: 3.0994415283203125e-05 seconds
INFO 05-03 18:02:18.520554.520554 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00024819374084472656 seconds
DEBUG 05-03 18:02:18.520126.520126 cuda_h.py:27] end *layer_moe_fused cost 24.282 ms
DEBUG 05-03 18:02:18.520737.520737 cuda_h.py:27] end prefill_layer cost 30.515 ms
DEBUG 05-03 18:02:18.520169.520169 lmp.py:711] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-03 18:02:18.521887.521887 lmp.py:675] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-03 18:02:18.524222.524222 cuda_h.py:27] end *sagl cost 3.641 ms
experts_cpu_alloc {'expert_ids': [47, 63, 83, 111, 23, 55, 31, 67, 19, 52, 96, 12, 28, 32, 36, 48, 60, 104, 120, 8, 84, 9, 21, 29, 25, 69, 121, 37, 65, 49, 33, 17, 57, 109, 117, 14, 26, 70, 106, 66, 78, 98, 74, 86, 118, 18, 54, 126, 62], 'token_total': 182, 'token_per_expert': {47: 1, 63: 1, 83: 1, 111: 1, 23: 3, 55: 3, 31: 4, 67: 4, 19: 5, 52: 1, 96: 1, 12: 2, 28: 2, 32: 2, 36: 2, 48: 2, 60: 2, 104: 2, 120: 2, 8: 3, 84: 3, 9: 1, 21: 2, 29: 2, 25: 3, 69: 3, 121: 3, 37: 4, 65: 5, 49: 7, 33: 8, 17: 9, 57: 10, 109: 15, 117: 17, 14: 1, 26: 1, 70: 1, 106: 1, 66: 2, 78: 2, 98: 2, 74: 3, 86: 3, 118: 3, 18: 6, 54: 6, 126: 6, 62: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 35, 51, 75, 79, 87, 91, 99, 115, 119, 123], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 745, 'token_per_expert': {3: 276, 7: 258, 11: 6, 35: 12, 51: 5, 75: 18, 79: 93, 87: 11, 91: 13, 99: 27, 115: 6, 119: 8, 123: 12}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 40, 56, 64, 68, 80, 108, 112, 116, 124], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 914, 'token_per_expert': {0: 268, 4: 259, 16: 10, 20: 12, 40: 105, 56: 21, 64: 35, 68: 140, 80: 5, 108: 27, 112: 6, 116: 3, 124: 23}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 41, 53, 73, 77, 89, 93, 97, 113, 125], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1247, 'token_per_expert': {1: 256, 5: 256, 13: 18, 41: 50, 53: 71, 73: 30, 77: 147, 89: 19, 93: 135, 97: 101, 113: 108, 125: 56}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 34, 42, 46, 82, 90, 94, 102, 110, 122], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1008, 'token_per_expert': {2: 256, 6: 256, 10: 23, 34: 120, 42: 83, 46: 135, 82: 24, 90: 13, 94: 11, 102: 64, 110: 13, 122: 10}}
INFO 05-03 18:02:18.525851.525851 lmp.py:1005] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.521ms | allocate_experts_across_cpu_gpu: 0.380ms
INFO 05-03 18:02:18.525227.525227 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.4836273193359375e-05 seconds
INFO 05-03 18:02:18.526605.526605 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000659942626953125 seconds
INFO 05-03 18:02:18.540015.540015 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013253211975097656 seconds
INFO 05-03 18:02:18.541620.541620 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009448528289794922 seconds
INFO 05-03 18:02:18.543022.543022 mlpmodule.py:2707] [fused_experts] gmm total=1.417ms E=32 S=768 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.545527.545527 mlpmodule.py:2707] [fused_experts] gmm total=1.538ms E=32 S=938 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.546114.546114 mlpmodule.py:2707] [fused_experts] gmm total=1.383ms E=32 S=1336 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.549534.549534 mlpmodule.py:2707] [fused_experts] gmm total=1.361ms E=32 S=1054 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.549174.549174 lmp.py:1149] [layer_moe_fused] experts compute time: 0.00794672966003418 seconds
INFO 05-03 18:02:18.549337.549337 lmp.py:1160] [layer_moe_fused] to time: 3.075599670410156e-05 seconds
INFO 05-03 18:02:18.550345.550345 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002961158752441406 seconds
DEBUG 05-03 18:02:18.550898.550898 cuda_h.py:27] end *layer_moe_fused cost 25.622 ms
DEBUG 05-03 18:02:18.550032.550032 cuda_h.py:27] end prefill_layer cost 29.461 ms
DEBUG 05-03 18:02:18.550610.550610 lmp.py:711] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-03 18:02:18.550245.550245 lmp.py:675] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-03 18:02:18.554249.554249 cuda_h.py:27] end *sagl cost 3.896 ms
experts_cpu_alloc {'expert_ids': [19, 127, 71, 47, 55, 87, 91, 111, 31, 43, 35, 11, 72, 36, 108, 12, 8, 28, 84, 44, 29, 85, 9, 57, 101, 65, 45, 113, 14, 70, 18, 26, 34, 50, 74, 114, 98], 'token_total': 121, 'token_per_expert': {19: 1, 127: 1, 71: 2, 47: 3, 55: 3, 87: 3, 91: 3, 111: 3, 31: 4, 43: 4, 35: 5, 11: 6, 72: 1, 36: 2, 108: 2, 12: 3, 8: 5, 28: 5, 84: 9, 44: 20, 29: 1, 85: 1, 9: 2, 57: 2, 101: 2, 65: 3, 45: 4, 113: 4, 14: 1, 70: 1, 18: 2, 26: 2, 34: 2, 50: 2, 74: 2, 114: 2, 98: 3}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 59, 63, 83, 95, 103, 115, 123], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 791, 'token_per_expert': {3: 257, 7: 256, 39: 12, 59: 51, 63: 12, 83: 7, 95: 100, 103: 53, 115: 9, 123: 34}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 48, 52, 56, 60, 96, 116], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1506, 'token_per_expert': {0: 256, 4: 369, 20: 181, 24: 81, 48: 88, 52: 149, 56: 67, 60: 78, 96: 196, 116: 41}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 69, 77, 105, 109, 117, 121, 125], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 847, 'token_per_expert': {1: 256, 5: 256, 69: 119, 77: 8, 105: 36, 109: 78, 117: 5, 121: 63, 125: 26}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 30, 62, 90, 106, 122, 126], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 831, 'token_per_expert': {2: 256, 6: 282, 10: 53, 30: 39, 62: 6, 90: 4, 106: 172, 122: 13, 126: 6}}
INFO 05-03 18:02:18.555439.555439 lmp.py:1005] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.516ms | allocate_experts_across_cpu_gpu: 0.299ms
INFO 05-03 18:02:18.555847.555847 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.482269287109375e-05 seconds
INFO 05-03 18:02:18.556625.556625 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006144046783447266 seconds
INFO 05-03 18:02:18.568182.568182 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011121034622192383 seconds
INFO 05-03 18:02:18.569227.569227 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008389949798583984 seconds
INFO 05-03 18:02:18.570408.570408 mlpmodule.py:2707] [fused_experts] gmm total=1.368ms E=32 S=829 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.572966.572966 mlpmodule.py:2707] [fused_experts] gmm total=1.562ms E=32 S=1553 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.576432.576432 mlpmodule.py:2707] [fused_experts] gmm total=1.170ms E=32 S=866 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.577528.577528 mlpmodule.py:2707] [fused_experts] gmm total=1.358ms E=32 S=848 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.578209.578209 lmp.py:1149] [layer_moe_fused] experts compute time: 0.008688211441040039 seconds
INFO 05-03 18:02:18.578207.578207 lmp.py:1160] [layer_moe_fused] to time: 3.170967102050781e-05 seconds
INFO 05-03 18:02:18.578368.578368 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0003039836883544922 seconds
DEBUG 05-03 18:02:18.579615.579615 cuda_h.py:27] end *layer_moe_fused cost 24.087 ms
DEBUG 05-03 18:02:18.579273.579273 cuda_h.py:27] end prefill_layer cost 28.321 ms
DEBUG 05-03 18:02:18.579181.579181 lmp.py:711] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-03 18:02:18.579809.579809 lmp.py:675] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-03 18:02:18.582362.582362 cuda_h.py:27] end *sagl cost 2.831 ms
experts_cpu_alloc {'expert_ids': [15, 87, 63, 91, 27, 11, 39, 32, 48, 24, 40, 76, 96, 104, 16, 60, 56, 80, 44, 117, 57, 65, 41, 69, 49, 33, 29, 93, 85, 21, 122, 26, 50, 14, 86, 98], 'token_total': 234, 'token_per_expert': {15: 1, 87: 2, 63: 3, 91: 3, 27: 5, 11: 6, 39: 6, 32: 1, 48: 1, 24: 2, 40: 2, 76: 2, 96: 2, 104: 2, 16: 3, 60: 3, 56: 4, 80: 4, 44: 5, 117: 1, 57: 2, 65: 3, 41: 4, 69: 9, 49: 10, 33: 11, 29: 13, 93: 22, 85: 36, 21: 37, 122: 1, 26: 3, 50: 3, 14: 6, 86: 8, 98: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 51, 55, 71, 79, 103, 107, 123], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 644, 'token_per_expert': {3: 258, 7: 256, 31: 9, 51: 10, 55: 11, 71: 7, 79: 15, 103: 34, 107: 7, 123: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 64, 68, 84, 92, 100, 120, 124], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 934, 'token_per_expert': {0: 258, 4: 257, 64: 8, 68: 34, 84: 132, 92: 125, 100: 9, 120: 7, 124: 104}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 73, 77, 81, 89, 121, 125], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1126, 'token_per_expert': {1: 256, 5: 257, 13: 93, 73: 69, 77: 99, 81: 65, 89: 96, 121: 91, 125: 100}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 30, 46, 54, 58, 118], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1158, 'token_per_expert': {2: 256, 6: 479, 18: 80, 22: 51, 30: 112, 46: 26, 54: 18, 58: 128, 118: 8}}
INFO 05-03 18:02:18.583365.583365 lmp.py:1005] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.497ms | allocate_experts_across_cpu_gpu: 0.295ms
INFO 05-03 18:02:18.583687.583687 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.458427429199219e-05 seconds
INFO 05-03 18:02:18.584078.584078 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006051063537597656 seconds
INFO 05-03 18:02:18.598208.598208 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013193130493164062 seconds
INFO 05-03 18:02:18.599510.599510 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008165836334228516 seconds
INFO 05-03 18:02:18.600649.600649 mlpmodule.py:2707] [fused_experts] gmm total=1.267ms E=32 S=670 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.601539.601539 mlpmodule.py:2707] [fused_experts] gmm total=1.214ms E=32 S=965 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.603452.603452 mlpmodule.py:2707] [fused_experts] gmm total=1.231ms E=32 S=1274 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.605005.605005 mlpmodule.py:2707] [fused_experts] gmm total=1.316ms E=32 S=1187 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.605773.605773 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006403207778930664 seconds
INFO 05-03 18:02:18.605744.605744 lmp.py:1160] [layer_moe_fused] to time: 3.0279159545898438e-05 seconds
INFO 05-03 18:02:18.605667.605667 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00030040740966796875 seconds
DEBUG 05-03 18:02:18.606901.606901 cuda_h.py:27] end *layer_moe_fused cost 23.908 ms
DEBUG 05-03 18:02:18.606936.606936 cuda_h.py:27] end prefill_layer cost 27.130 ms
DEBUG 05-03 18:02:18.606129.606129 lmp.py:711] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-03 18:02:18.606932.606932 lmp.py:675] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-03 18:02:18.610126.610126 cuda_h.py:27] end *sagl cost 3.823 ms
experts_cpu_alloc {'expert_ids': [11, 71, 27, 111, 83, 75, 103, 43, 39, 32, 112, 64, 24, 108, 36, 124, 25, 81, 89, 45, 69, 37, 14, 18, 46, 70, 82, 102, 126, 74, 122], 'token_total': 130, 'token_per_expert': {11: 1, 71: 1, 27: 2, 111: 2, 83: 3, 75: 6, 103: 6, 43: 12, 39: 13, 32: 1, 112: 1, 64: 3, 24: 11, 108: 11, 36: 13, 124: 21, 25: 1, 81: 1, 89: 1, 45: 2, 69: 3, 37: 4, 14: 1, 18: 1, 46: 1, 70: 1, 82: 1, 102: 1, 126: 1, 74: 2, 122: 2}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 51, 79, 91, 95], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 969, 'token_per_expert': {3: 256, 7: 257, 15: 203, 19: 34, 51: 51, 79: 22, 91: 36, 95: 110}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 28, 48, 68, 88], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1006, 'token_per_expert': {0: 260, 4: 256, 12: 139, 16: 38, 28: 60, 48: 84, 68: 26, 88: 143}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 29, 57, 65, 97], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 947, 'token_per_expert': {1: 263, 5: 256, 9: 212, 21: 9, 29: 18, 57: 11, 65: 9, 97: 169}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 86, 98, 106, 114], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1044, 'token_per_expert': {2: 290, 6: 256, 22: 172, 26: 3, 86: 9, 98: 158, 106: 82, 114: 74}}
INFO 05-03 18:02:18.611340.611340 lmp.py:1005] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.496ms | allocate_experts_across_cpu_gpu: 0.274ms
INFO 05-03 18:02:18.611033.611033 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 3.814697265625e-05 seconds
INFO 05-03 18:02:18.612229.612229 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005781650543212891 seconds
INFO 05-03 18:02:18.624895.624895 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011716842651367188 seconds
INFO 05-03 18:02:18.625507.625507 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007731914520263672 seconds
INFO 05-03 18:02:18.627815.627815 mlpmodule.py:2707] [fused_experts] gmm total=1.391ms E=32 S=1015 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.628713.628713 mlpmodule.py:2707] [fused_experts] gmm total=1.411ms E=32 S=1067 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.630501.630501 mlpmodule.py:2707] [fused_experts] gmm total=1.241ms E=32 S=959 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.632028.632028 mlpmodule.py:2707] [fused_experts] gmm total=1.296ms E=32 S=1055 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.632371.632371 lmp.py:1149] [layer_moe_fused] experts compute time: 0.0067408084869384766 seconds
INFO 05-03 18:02:18.632574.632574 lmp.py:1160] [layer_moe_fused] to time: 3.0040740966796875e-05 seconds
INFO 05-03 18:02:18.632118.632118 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002665519714355469 seconds
DEBUG 05-03 18:02:18.633212.633212 cuda_h.py:27] end *layer_moe_fused cost 22.484 ms
DEBUG 05-03 18:02:18.633678.633678 cuda_h.py:27] end prefill_layer cost 26.638 ms
DEBUG 05-03 18:02:18.633110.633110 lmp.py:711] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-03 18:02:18.633084.633084 lmp.py:675] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-03 18:02:18.637447.637447 cuda_h.py:27] end *sagl cost 3.735 ms
experts_cpu_alloc {'expert_ids': [103, 31, 51, 63, 39, 71, 87, 91, 67, 20, 84, 44, 120, 85, 9, 57, 25, 125, 113, 121, 37, 10, 26, 78, 98, 118, 62, 66], 'token_total': 160, 'token_per_expert': {103: 1, 31: 2, 51: 2, 63: 6, 39: 8, 71: 8, 87: 9, 91: 10, 67: 17, 20: 1, 84: 1, 44: 3, 120: 5, 85: 1, 9: 2, 57: 4, 25: 6, 125: 12, 113: 15, 121: 15, 37: 20, 10: 1, 26: 1, 78: 1, 98: 1, 118: 1, 62: 2, 66: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 43, 79, 111, 115], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1049, 'token_per_expert': {3: 269, 7: 260, 11: 39, 19: 18, 43: 145, 79: 20, 111: 131, 115: 167}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 48, 68, 80, 100, 108], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 938, 'token_per_expert': {0: 257, 4: 256, 48: 19, 68: 176, 80: 28, 100: 10, 108: 192}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 33, 53, 81], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1121, 'token_per_expert': {1: 256, 5: 256, 13: 62, 29: 186, 33: 227, 53: 97, 81: 37}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 46, 82, 94], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 828, 'token_per_expert': {2: 256, 6: 275, 14: 18, 18: 22, 46: 11, 82: 124, 94: 122}}
INFO 05-03 18:02:18.638468.638468 lmp.py:1005] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.488ms | allocate_experts_across_cpu_gpu: 0.248ms
INFO 05-03 18:02:18.638578.638578 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 3.600120544433594e-05 seconds
INFO 05-03 18:02:18.639519.639519 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005950927734375 seconds
INFO 05-03 18:02:18.650988.650988 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010981082916259766 seconds
INFO 05-03 18:02:18.651653.651653 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007746219635009766 seconds
INFO 05-03 18:02:18.653299.653299 mlpmodule.py:2707] [fused_experts] gmm total=1.391ms E=32 S=1112 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.654210.654210 mlpmodule.py:2707] [fused_experts] gmm total=1.174ms E=32 S=948 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.656591.656591 mlpmodule.py:2707] [fused_experts] gmm total=1.291ms E=32 S=1196 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.657809.657809 mlpmodule.py:2707] [fused_experts] gmm total=1.060ms E=32 S=840 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.658714.658714 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006288766860961914 seconds
INFO 05-03 18:02:18.658343.658343 lmp.py:1160] [layer_moe_fused] to time: 3.1948089599609375e-05 seconds
INFO 05-03 18:02:18.658237.658237 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002465248107910156 seconds
DEBUG 05-03 18:02:18.658853.658853 cuda_h.py:27] end *layer_moe_fused cost 21.143 ms
DEBUG 05-03 18:02:18.658273.658273 cuda_h.py:27] end prefill_layer cost 25.209 ms
DEBUG 05-03 18:02:18.658658.658658 lmp.py:711] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-03 18:02:18.659323.659323 lmp.py:675] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-03 18:02:18.664010.664010 cuda_h.py:27] end *sagl cost 5.464 ms
experts_cpu_alloc {'expert_ids': [27, 83, 87, 31, 127, 20, 56, 100, 24, 124, 12, 104, 88, 32, 40, 112, 28, 21, 25, 33, 73, 89, 81, 34, 42, 110, 50, 102, 18, 98], 'token_total': 162, 'token_per_expert': {27: 1, 83: 1, 87: 1, 31: 2, 127: 3, 20: 1, 56: 1, 100: 1, 24: 2, 124: 2, 12: 3, 104: 7, 88: 9, 32: 14, 40: 14, 112: 16, 28: 43, 21: 1, 25: 1, 33: 1, 73: 1, 89: 1, 81: 11, 34: 1, 42: 1, 110: 1, 50: 2, 102: 2, 18: 7, 98: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 43, 63, 79, 91, 119], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1224, 'token_per_expert': {3: 446, 7: 260, 23: 115, 43: 113, 63: 15, 79: 5, 91: 107, 119: 163}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 44, 48, 64, 76, 120], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1327, 'token_per_expert': {0: 257, 4: 256, 36: 71, 44: 211, 48: 96, 64: 104, 76: 208, 120: 124}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 49, 65, 77, 93, 117], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 728, 'token_per_expert': {1: 265, 5: 256, 17: 93, 49: 28, 65: 24, 77: 15, 93: 16, 117: 31}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 54, 62, 74, 118], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 655, 'token_per_expert': {2: 256, 6: 259, 22: 24, 54: 19, 62: 43, 74: 31, 118: 23}}
INFO 05-03 18:02:18.665051.665051 lmp.py:1005] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.489ms | allocate_experts_across_cpu_gpu: 0.258ms
INFO 05-03 18:02:18.665260.665260 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 3.8623809814453125e-05 seconds
INFO 05-03 18:02:18.666191.666191 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006012916564941406 seconds
INFO 05-03 18:02:18.678464.678464 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010894775390625 seconds
INFO 05-03 18:02:18.678883.678883 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007338523864746094 seconds
INFO 05-03 18:02:18.680852.680852 mlpmodule.py:2707] [fused_experts] gmm total=1.348ms E=32 S=1232 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.682081.682081 mlpmodule.py:2707] [fused_experts] gmm total=1.374ms E=32 S=1440 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.684767.684767 mlpmodule.py:2707] [fused_experts] gmm total=1.932ms E=32 S=744 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.685065.685065 mlpmodule.py:2707] [fused_experts] gmm total=1.060ms E=32 S=680 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.685532.685532 lmp.py:1149] [layer_moe_fused] experts compute time: 0.0070378780364990234 seconds
INFO 05-03 18:02:18.686688.686688 lmp.py:1160] [layer_moe_fused] to time: 3.0279159545898438e-05 seconds
INFO 05-03 18:02:18.686535.686535 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002086162567138672 seconds
DEBUG 05-03 18:02:18.686762.686762 cuda_h.py:27] end *layer_moe_fused cost 21.779 ms
DEBUG 05-03 18:02:18.686227.686227 cuda_h.py:27] end prefill_layer cost 27.597 ms
DEBUG 05-03 18:02:18.686421.686421 lmp.py:711] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-03 18:02:18.687315.687315 lmp.py:675] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-03 18:02:18.689048.689048 cuda_h.py:27] end *sagl cost 2.749 ms
experts_cpu_alloc {'expert_ids': [11, 63, 67, 103, 19, 111, 36, 92, 104, 64, 72, 17, 49, 57, 93, 109, 53, 33, 21, 37, 85, 46, 50, 58, 106, 122, 10, 78, 94, 102, 14, 22], 'token_total': 94, 'token_per_expert': {11: 1, 63: 1, 67: 1, 103: 1, 19: 4, 111: 15, 36: 1, 92: 1, 104: 1, 64: 3, 72: 3, 17: 1, 49: 1, 57: 1, 93: 1, 109: 1, 53: 2, 33: 3, 21: 4, 37: 4, 85: 5, 46: 1, 50: 1, 58: 1, 106: 1, 122: 1, 10: 2, 78: 2, 94: 3, 102: 5, 14: 10, 22: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 31, 35, 107, 119, 123], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1266, 'token_per_expert': {3: 315, 7: 256, 23: 44, 27: 65, 31: 57, 35: 94, 107: 130, 119: 222, 123: 83}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 40, 44, 84, 88, 116, 124], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 886, 'token_per_expert': {0: 256, 4: 256, 40: 170, 44: 7, 84: 23, 88: 123, 116: 4, 124: 47}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 41, 45, 101, 105, 117], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 896, 'token_per_expert': {1: 262, 5: 257, 13: 82, 41: 35, 45: 178, 101: 45, 105: 31, 117: 6}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34, 74, 86, 90, 98, 110], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 954, 'token_per_expert': {2: 256, 6: 257, 34: 167, 74: 139, 86: 14, 90: 25, 98: 15, 110: 81}}
INFO 05-03 18:02:18.690711.690711 lmp.py:1005] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.485ms | allocate_experts_across_cpu_gpu: 0.268ms
INFO 05-03 18:02:18.690828.690828 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 3.9577484130859375e-05 seconds
INFO 05-03 18:02:18.691171.691171 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005893707275390625 seconds
INFO 05-03 18:02:18.703725.703725 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011534452438354492 seconds
INFO 05-03 18:02:18.712920.712920 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.008507728576660156 seconds
INFO 05-03 18:02:18.713106.713106 mlpmodule.py:2707] [fused_experts] gmm total=1.466ms E=32 S=1289 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.715073.715073 mlpmodule.py:2707] [fused_experts] gmm total=1.115ms E=32 S=895 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.716839.716839 mlpmodule.py:2707] [fused_experts] gmm total=1.197ms E=32 S=919 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.718233.718233 mlpmodule.py:2707] [fused_experts] gmm total=1.168ms E=32 S=993 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.718900.718900 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006222248077392578 seconds
INFO 05-03 18:02:18.718056.718056 lmp.py:1160] [layer_moe_fused] to time: 2.9802322387695312e-05 seconds
INFO 05-03 18:02:18.718130.718130 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00027108192443847656 seconds
DEBUG 05-03 18:02:18.719080.719080 cuda_h.py:27] end *layer_moe_fused cost 29.363 ms
DEBUG 05-03 18:02:18.719069.719069 cuda_h.py:27] end prefill_layer cost 32.402 ms
DEBUG 05-03 18:02:18.719322.719322 lmp.py:711] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-03 18:02:18.719131.719131 lmp.py:675] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-03 18:02:18.723532.723532 cuda_h.py:27] end *sagl cost 3.837 ms
experts_cpu_alloc {'expert_ids': [31, 15, 87, 79, 55, 39, 75, 32, 48, 68, 76, 80, 84, 96, 112, 116, 108, 100, 61, 81, 93, 113, 13, 37, 45, 21, 30, 42, 50, 102, 118, 38, 10, 86], 'token_total': 198, 'token_per_expert': {31: 2, 15: 5, 87: 7, 79: 10, 55: 12, 39: 15, 75: 15, 32: 1, 48: 1, 68: 1, 76: 1, 80: 2, 84: 2, 96: 2, 112: 2, 116: 2, 108: 4, 100: 8, 61: 1, 81: 3, 93: 3, 113: 3, 13: 4, 37: 6, 45: 15, 21: 16, 30: 1, 42: 1, 50: 3, 102: 3, 118: 6, 38: 9, 10: 16, 86: 16}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 43, 47, 67, 91, 99, 115, 119], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 911, 'token_per_expert': {3: 256, 7: 270, 43: 43, 47: 46, 67: 39, 91: 61, 99: 52, 115: 126, 119: 18}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 40, 56, 60, 64, 120], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 939, 'token_per_expert': {0: 259, 4: 257, 20: 52, 24: 25, 40: 196, 56: 10, 60: 55, 64: 72, 120: 13}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 53, 73, 89, 105, 117, 121], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1140, 'token_per_expert': {1: 268, 5: 256, 33: 135, 53: 19, 73: 193, 89: 90, 105: 82, 117: 79, 121: 18}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 34, 70, 78, 98, 110, 122], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 908, 'token_per_expert': {2: 264, 6: 274, 34: 72, 70: 44, 78: 136, 98: 79, 110: 18, 122: 21}}
INFO 05-03 18:02:18.724689.724689 lmp.py:1005] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.549ms | allocate_experts_across_cpu_gpu: 0.285ms
INFO 05-03 18:02:18.724998.724998 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.1961669921875e-05 seconds
INFO 05-03 18:02:18.725129.725129 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005915164947509766 seconds
INFO 05-03 18:02:18.737874.737874 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01179814338684082 seconds
INFO 05-03 18:02:18.744466.744466 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.007283449172973633 seconds
INFO 05-03 18:02:18.746869.746869 mlpmodule.py:2707] [fused_experts] gmm total=1.251ms E=32 S=977 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.747714.747714 mlpmodule.py:2707] [fused_experts] gmm total=1.250ms E=32 S=965 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.749970.749970 mlpmodule.py:2707] [fused_experts] gmm total=1.203ms E=32 S=1191 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.751453.751453 mlpmodule.py:2707] [fused_experts] gmm total=1.429ms E=32 S=963 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.751883.751883 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006445646286010742 seconds
INFO 05-03 18:02:18.751993.751993 lmp.py:1160] [layer_moe_fused] to time: 3.0279159545898438e-05 seconds
INFO 05-03 18:02:18.752573.752573 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0005748271942138672 seconds
DEBUG 05-03 18:02:18.752286.752286 cuda_h.py:27] end *layer_moe_fused cost 29.158 ms
DEBUG 05-03 18:02:18.752752.752752 cuda_h.py:27] end prefill_layer cost 33.182 ms
DEBUG 05-03 18:02:18.752661.752661 lmp.py:711] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-03 18:02:18.752521.752521 lmp.py:675] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-03 18:02:18.756787.756787 cuda_h.py:27] end *sagl cost 3.210 ms
experts_cpu_alloc {'expert_ids': [43, 15, 59, 99, 123, 63, 103, 95, 100, 44, 108, 96, 76, 8, 53, 77, 29, 109, 49, 89, 57, 97, 14, 126, 38, 10], 'token_total': 303, 'token_per_expert': {43: 2, 15: 4, 59: 4, 99: 5, 123: 7, 63: 15, 103: 17, 95: 19, 100: 1, 44: 2, 108: 5, 96: 8, 76: 22, 8: 42, 53: 1, 77: 1, 29: 3, 109: 3, 49: 10, 89: 10, 57: 12, 97: 78, 14: 2, 126: 7, 38: 11, 10: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39, 75, 83, 91], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 915, 'token_per_expert': {3: 256, 7: 256, 19: 45, 39: 97, 75: 30, 83: 202, 91: 29}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 48, 52, 112, 120], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1010, 'token_per_expert': {0: 264, 4: 256, 24: 203, 48: 45, 52: 57, 112: 135, 120: 50}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 33, 65, 125], 'expert_count': 6, 'target_expert_count': 6, 'token_total': 1067, 'token_per_expert': {1: 256, 5: 256, 9: 93, 33: 197, 65: 94, 125: 171}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 50, 70, 98], 'expert_count': 6, 'target_expert_count': 6, 'token_total': 801, 'token_per_expert': {2: 256, 6: 256, 26: 102, 50: 111, 70: 17, 98: 59}}
INFO 05-03 18:02:18.757058.757058 lmp.py:1005] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.480ms | allocate_experts_across_cpu_gpu: 0.232ms
INFO 05-03 18:02:18.757745.757745 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 3.2901763916015625e-05 seconds
INFO 05-03 18:02:18.757459.757459 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005748271942138672 seconds
INFO 05-03 18:02:18.769752.769752 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010888814926147461 seconds
INFO 05-03 18:02:18.771075.771075 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0024247169494628906 seconds
INFO 05-03 18:02:18.773183.773183 mlpmodule.py:2707] [fused_experts] gmm total=1.346ms E=32 S=988 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.774906.774906 mlpmodule.py:2707] [fused_experts] gmm total=1.118ms E=32 S=1090 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.776248.776248 mlpmodule.py:2707] [fused_experts] gmm total=1.190ms E=32 S=1185 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.777660.777660 mlpmodule.py:2707] [fused_experts] gmm total=1.035ms E=32 S=833 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.777511.777511 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006038665771484375 seconds
INFO 05-03 18:02:18.778144.778144 lmp.py:1160] [layer_moe_fused] to time: 3.0040740966796875e-05 seconds
INFO 05-03 18:02:18.778369.778369 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00024127960205078125 seconds
DEBUG 05-03 18:02:18.778591.778591 cuda_h.py:27] end *layer_moe_fused cost 22.349 ms
DEBUG 05-03 18:02:18.778010.778010 cuda_h.py:27] end prefill_layer cost 25.887 ms
DEBUG 05-03 18:02:18.778634.778634 lmp.py:711] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-03 18:02:18.779523.779523 lmp.py:675] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-03 18:02:18.782467.782467 cuda_h.py:27] end *sagl cost 3.435 ms
experts_cpu_alloc {'expert_ids': [31, 55, 127, 75, 115, 28, 84, 36, 12, 8, 92, 109, 9, 85, 89, 117, 53, 113, 29, 37, 18, 106, 14, 50, 114, 66, 78, 94, 118], 'token_total': 115, 'token_per_expert': {31: 1, 55: 1, 127: 1, 75: 2, 115: 2, 28: 1, 84: 1, 36: 2, 12: 3, 8: 5, 92: 5, 109: 1, 9: 2, 85: 2, 89: 2, 117: 5, 53: 8, 113: 9, 29: 12, 37: 13, 18: 1, 106: 1, 14: 2, 50: 2, 114: 2, 66: 5, 78: 5, 94: 9, 118: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 51, 59, 67, 103, 107], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1073, 'token_per_expert': {3: 256, 7: 468, 19: 94, 51: 113, 59: 129, 67: 3, 103: 6, 107: 4}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 76, 88, 96, 100, 116], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 949, 'token_per_expert': {0: 256, 4: 256, 76: 234, 88: 73, 96: 13, 100: 20, 116: 97}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 57, 73, 81, 125], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 852, 'token_per_expert': {1: 256, 5: 256, 13: 34, 57: 143, 73: 117, 81: 17, 125: 29}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 54, 82, 86, 90], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1107, 'token_per_expert': {2: 260, 6: 256, 22: 27, 54: 174, 82: 181, 86: 178, 90: 31}}
INFO 05-03 18:02:18.783673.783673 lmp.py:1005] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.482ms | allocate_experts_across_cpu_gpu: 0.250ms
INFO 05-03 18:02:18.783479.783479 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 3.695487976074219e-05 seconds
INFO 05-03 18:02:18.784793.784793 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005636215209960938 seconds
INFO 05-03 18:02:18.796656.796656 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011502504348754883 seconds
INFO 05-03 18:02:18.809659.809659 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.01265406608581543 seconds
INFO 05-03 18:02:18.810561.810561 mlpmodule.py:2707] [fused_experts] gmm total=1.301ms E=32 S=1080 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.812294.812294 mlpmodule.py:2707] [fused_experts] gmm total=1.229ms E=32 S=966 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.813756.813756 mlpmodule.py:2707] [fused_experts] gmm total=1.150ms E=32 S=906 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.815367.815367 mlpmodule.py:2707] [fused_experts] gmm total=1.120ms E=32 S=1144 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.815768.815768 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006148338317871094 seconds
INFO 05-03 18:02:18.815985.815985 lmp.py:1160] [layer_moe_fused] to time: 2.9325485229492188e-05 seconds
INFO 05-03 18:02:18.815595.815595 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002903938293457031 seconds
DEBUG 05-03 18:02:18.816631.816631 cuda_h.py:27] end *layer_moe_fused cost 33.498 ms
DEBUG 05-03 18:02:18.816143.816143 cuda_h.py:27] end prefill_layer cost 37.325 ms
DEBUG 05-03 18:02:18.816098.816098 lmp.py:711] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-03 18:02:18.816994.816994 lmp.py:675] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-03 18:02:18.820124.820124 cuda_h.py:27] end *sagl cost 3.640 ms
experts_cpu_alloc {'expert_ids': [27, 99, 87, 43, 12, 36, 116, 81, 105, 53, 25, 33, 121, 77, 29, 65, 117, 34, 126, 50, 114, 46, 74, 30, 118, 66], 'token_total': 226, 'token_per_expert': {27: 1, 99: 1, 87: 2, 43: 53, 12: 1, 36: 1, 116: 4, 81: 1, 105: 1, 53: 3, 25: 4, 33: 7, 121: 10, 77: 11, 29: 12, 65: 25, 117: 33, 34: 1, 126: 1, 50: 2, 114: 4, 46: 6, 74: 6, 30: 7, 118: 10, 66: 19}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 103, 111, 127], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1357, 'token_per_expert': {3: 262, 7: 257, 19: 183, 31: 249, 103: 223, 111: 75, 127: 108}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 56, 64, 68, 96], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 772, 'token_per_expert': {0: 256, 4: 256, 20: 7, 56: 71, 64: 19, 68: 30, 96: 133}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 69, 85, 101], 'expert_count': 6, 'target_expert_count': 6, 'token_total': 891, 'token_per_expert': {1: 256, 5: 256, 17: 70, 69: 149, 85: 71, 101: 89}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 62, 70, 86], 'expert_count': 6, 'target_expert_count': 6, 'token_total': 850, 'token_per_expert': {2: 256, 6: 262, 10: 98, 62: 38, 70: 135, 86: 61}}
INFO 05-03 18:02:18.821495.821495 lmp.py:1005] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.478ms | allocate_experts_across_cpu_gpu: 0.235ms
INFO 05-03 18:02:18.821267.821267 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 3.218650817871094e-05 seconds
INFO 05-03 18:02:18.822435.822435 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005695819854736328 seconds
INFO 05-03 18:02:18.833344.833344 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010925531387329102 seconds
INFO 05-03 18:02:18.847138.847138 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.014109373092651367 seconds
INFO 05-03 18:02:18.849190.849190 mlpmodule.py:2707] [fused_experts] gmm total=1.234ms E=32 S=1414 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.850888.850888 mlpmodule.py:2707] [fused_experts] gmm total=0.995ms E=32 S=778 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.852627.852627 mlpmodule.py:2707] [fused_experts] gmm total=1.176ms E=32 S=998 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.853072.853072 mlpmodule.py:2707] [fused_experts] gmm total=1.098ms E=32 S=906 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.853916.853916 lmp.py:1149] [layer_moe_fused] experts compute time: 0.005762815475463867 seconds
INFO 05-03 18:02:18.853973.853973 lmp.py:1160] [layer_moe_fused] to time: 2.8133392333984375e-05 seconds
INFO 05-03 18:02:18.854621.854621 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00019979476928710938 seconds
DEBUG 05-03 18:02:18.854656.854656 cuda_h.py:27] end *layer_moe_fused cost 33.805 ms
DEBUG 05-03 18:02:18.854168.854168 cuda_h.py:27] end prefill_layer cost 37.839 ms
DEBUG 05-03 18:02:18.854361.854361 lmp.py:711] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-03 18:02:18.854810.854810 lmp.py:675] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-03 18:02:18.859148.859148 cuda_h.py:27] end *sagl cost 4.771 ms
experts_cpu_alloc {'expert_ids': [67, 91, 23, 83, 63, 119, 27, 44, 52, 96, 112, 80, 8, 100, 104, 33, 37, 77, 121, 125, 117, 73, 105, 61, 26, 54, 90, 62], 'token_total': 90, 'token_per_expert': {67: 1, 91: 1, 23: 2, 83: 3, 63: 4, 119: 4, 27: 5, 44: 1, 52: 1, 96: 1, 112: 1, 80: 3, 8: 4, 100: 4, 104: 7, 33: 1, 37: 1, 77: 1, 121: 1, 125: 4, 117: 6, 73: 8, 105: 8, 61: 9, 26: 2, 54: 2, 90: 2, 62: 3}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 51, 59, 75, 79], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 764, 'token_per_expert': {3: 258, 7: 283, 31: 8, 35: 17, 51: 130, 59: 11, 75: 23, 79: 34}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 36, 48, 72, 92], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1237, 'token_per_expert': {0: 261, 4: 442, 12: 77, 36: 163, 48: 38, 72: 122, 92: 134}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 45, 49, 81], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 1104, 'token_per_expert': {1: 393, 5: 431, 13: 73, 25: 128, 45: 12, 49: 50, 81: 17}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 58, 74, 82, 94, 114], 'expert_count': 7, 'target_expert_count': 7, 'token_total': 901, 'token_per_expert': {2: 299, 6: 269, 58: 14, 74: 69, 82: 137, 94: 7, 114: 106}}
INFO 05-03 18:02:18.860102.860102 lmp.py:1005] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.475ms | allocate_experts_across_cpu_gpu: 0.245ms
INFO 05-03 18:02:18.860590.860590 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 3.337860107421875e-05 seconds
INFO 05-03 18:02:18.861283.861283 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005550384521484375 seconds
INFO 05-03 18:02:18.873802.873802 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012099504470825195 seconds
INFO 05-03 18:02:18.874055.874055 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007569789886474609 seconds
INFO 05-03 18:02:18.876913.876913 mlpmodule.py:2707] [fused_experts] gmm total=1.374ms E=32 S=784 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.877410.877410 mlpmodule.py:2707] [fused_experts] gmm total=1.297ms E=32 S=1259 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.879969.879969 mlpmodule.py:2707] [fused_experts] gmm total=1.286ms E=32 S=1143 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.880493.880493 mlpmodule.py:2707] [fused_experts] gmm total=0.997ms E=32 S=910 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.881120.881120 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006395101547241211 seconds
INFO 05-03 18:02:18.881649.881649 lmp.py:1160] [layer_moe_fused] to time: 3.24249267578125e-05 seconds
INFO 05-03 18:02:18.881921.881921 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00024318695068359375 seconds
DEBUG 05-03 18:02:18.881300.881300 cuda_h.py:27] end *layer_moe_fused cost 22.253 ms
DEBUG 05-03 18:02:18.882766.882766 cuda_h.py:27] end prefill_layer cost 27.354 ms
DEBUG 05-03 18:02:18.882436.882436 lmp.py:711] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-03 18:02:18.882991.882991 lmp.py:675] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-03 18:02:18.886260.886260 cuda_h.py:27] end *sagl cost 4.127 ms
experts_cpu_alloc {'expert_ids': [127, 55, 59, 63, 75, 31, 87, 115, 79, 20, 40, 8, 104, 112, 16, 44, 96, 68, 52, 24, 29, 17, 125, 118, 122, 30, 86, 10, 62, 66, 42, 98, 22], 'token_total': 195, 'token_per_expert': {127: 1, 55: 2, 59: 2, 63: 2, 75: 4, 31: 5, 87: 7, 115: 7, 79: 8, 20: 1, 40: 1, 8: 2, 104: 2, 112: 2, 16: 3, 44: 5, 96: 7, 68: 8, 52: 9, 24: 14, 29: 1, 17: 2, 125: 2, 118: 1, 122: 1, 30: 2, 86: 2, 10: 3, 62: 4, 66: 16, 42: 19, 98: 23, 22: 27}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 47, 67, 95, 99, 123], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1002, 'token_per_expert': {3: 256, 7: 277, 11: 85, 23: 21, 47: 66, 67: 11, 95: 92, 99: 175, 123: 19}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 32, 48, 76, 92, 100], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 959, 'token_per_expert': {0: 363, 4: 259, 28: 81, 32: 67, 48: 25, 76: 78, 92: 28, 100: 58}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 41, 57, 61, 73, 77, 105], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 837, 'token_per_expert': {1: 281, 5: 260, 41: 16, 57: 213, 61: 12, 73: 18, 77: 4, 105: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 34, 50, 82, 110, 114], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 1103, 'token_per_expert': {2: 256, 6: 256, 18: 217, 34: 37, 50: 102, 82: 121, 110: 31, 114: 83}}
INFO 05-03 18:02:18.887732.887732 lmp.py:1005] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.477ms | allocate_experts_across_cpu_gpu: 0.273ms
INFO 05-03 18:02:18.887564.887564 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.100799560546875e-05 seconds
INFO 05-03 18:02:18.888819.888819 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005545616149902344 seconds
INFO 05-03 18:02:18.899874.899874 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010973691940307617 seconds
INFO 05-03 18:02:18.914789.914789 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.014372587203979492 seconds
INFO 05-03 18:02:18.915663.915663 mlpmodule.py:2707] [fused_experts] gmm total=1.279ms E=32 S=1040 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.917124.917124 mlpmodule.py:2707] [fused_experts] gmm total=1.207ms E=32 S=1013 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.918889.918889 mlpmodule.py:2707] [fused_experts] gmm total=0.984ms E=32 S=842 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.920674.920674 mlpmodule.py:2707] [fused_experts] gmm total=1.349ms E=32 S=1201 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.920269.920269 lmp.py:1149] [layer_moe_fused] experts compute time: 0.0061283111572265625 seconds
INFO 05-03 18:02:18.920902.920902 lmp.py:1160] [layer_moe_fused] to time: 2.956390380859375e-05 seconds
INFO 05-03 18:02:18.920274.920274 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002791881561279297 seconds
DEBUG 05-03 18:02:18.921052.921052 cuda_h.py:27] end *layer_moe_fused cost 34.663 ms
DEBUG 05-03 18:02:18.921803.921803 cuda_h.py:27] end prefill_layer cost 39.081 ms
DEBUG 05-03 18:02:18.921235.921235 lmp.py:711] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-03 18:02:18.921679.921679 lmp.py:675] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-03 18:02:18.925734.925734 cuda_h.py:27] end *sagl cost 3.373 ms
experts_cpu_alloc {'expert_ids': [23, 31, 111, 103, 119, 123, 99, 127, 43, 28, 72, 96, 104, 12, 77, 125, 113, 97, 17, 61, 73, 45, 57, 85, 33, 78, 74, 122, 62, 18, 66, 114, 54, 106], 'token_total': 265, 'token_per_expert': {23: 1, 31: 2, 111: 5, 103: 8, 119: 10, 123: 10, 99: 13, 127: 13, 43: 21, 28: 1, 72: 1, 96: 1, 104: 1, 12: 3, 77: 1, 125: 1, 113: 2, 97: 3, 17: 4, 61: 8, 73: 8, 45: 9, 57: 11, 85: 12, 33: 13, 78: 1, 74: 3, 122: 4, 62: 6, 18: 11, 66: 11, 114: 18, 54: 20, 106: 29}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 59, 63, 71, 75, 83, 91], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1064, 'token_per_expert': {3: 256, 7: 258, 11: 34, 59: 119, 63: 23, 71: 124, 75: 33, 83: 173, 91: 44}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 32, 36, 40, 56, 76, 92], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 681, 'token_per_expert': {0: 258, 4: 256, 16: 18, 32: 14, 36: 20, 40: 25, 56: 84, 76: 3, 92: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 41, 49, 89, 93, 121], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1229, 'token_per_expert': {1: 256, 5: 256, 13: 60, 21: 89, 41: 30, 49: 161, 89: 102, 93: 114, 121: 161}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 26, 58, 90, 98, 118], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 857, 'token_per_expert': {2: 257, 6: 273, 22: 71, 26: 49, 58: 42, 90: 38, 98: 74, 118: 53}}
INFO 05-03 18:02:18.926551.926551 lmp.py:1005] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.481ms | allocate_experts_across_cpu_gpu: 0.280ms
INFO 05-03 18:02:18.926190.926190 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.00543212890625e-05 seconds
INFO 05-03 18:02:18.926931.926931 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005462169647216797 seconds
INFO 05-03 18:02:18.939928.939928 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011800050735473633 seconds
INFO 05-03 18:02:18.940812.940812 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007860660552978516 seconds
INFO 05-03 18:02:18.941495.941495 mlpmodule.py:2707] [fused_experts] gmm total=1.314ms E=32 S=1147 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.942228.942228 mlpmodule.py:2707] [fused_experts] gmm total=1.057ms E=32 S=688 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.944731.944731 mlpmodule.py:2707] [fused_experts] gmm total=1.257ms E=32 S=1301 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.946099.946099 mlpmodule.py:2707] [fused_experts] gmm total=1.171ms E=32 S=960 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.946010.946010 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006035566329956055 seconds
INFO 05-03 18:02:18.946166.946166 lmp.py:1160] [layer_moe_fused] to time: 3.0040740966796875e-05 seconds
INFO 05-03 18:02:18.946293.946293 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002753734588623047 seconds
DEBUG 05-03 18:02:18.947370.947370 cuda_h.py:27] end *layer_moe_fused cost 21.874 ms
DEBUG 05-03 18:02:18.947597.947597 cuda_h.py:27] end prefill_layer cost 25.642 ms
DEBUG 05-03 18:02:18.947029.947029 lmp.py:711] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-03 18:02:18.947988.947988 lmp.py:675] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-03 18:02:18.951389.951389 cuda_h.py:27] end *sagl cost 4.275 ms
experts_cpu_alloc {'expert_ids': [19, 43, 63, 87, 119, 91, 35, 115, 20, 24, 40, 96, 8, 60, 32, 112, 48, 52, 12, 88, 28, 13, 81, 33, 41, 101, 37, 57, 93, 122, 98, 42, 66, 74, 102], 'token_total': 177, 'token_per_expert': {19: 1, 43: 1, 63: 1, 87: 2, 119: 2, 91: 4, 35: 5, 115: 5, 20: 1, 24: 1, 40: 1, 96: 1, 8: 2, 60: 2, 32: 3, 112: 4, 48: 5, 52: 14, 12: 19, 88: 22, 28: 31, 13: 1, 81: 2, 33: 3, 41: 3, 101: 3, 37: 4, 57: 7, 93: 10, 122: 1, 98: 2, 42: 3, 66: 3, 74: 3, 102: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 55, 71, 95, 99, 127], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 681, 'token_per_expert': {3: 256, 7: 260, 11: 23, 23: 6, 55: 6, 71: 88, 95: 17, 99: 16, 127: 9}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 36, 44, 56, 68, 108, 120], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1194, 'token_per_expert': {0: 256, 4: 276, 16: 108, 36: 41, 44: 120, 56: 215, 68: 88, 108: 34, 120: 56}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 45, 89, 105, 113, 117, 125], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1086, 'token_per_expert': {1: 256, 5: 256, 9: 104, 45: 77, 89: 11, 105: 27, 113: 106, 117: 128, 125: 121}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 46, 54, 58, 62], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 958, 'token_per_expert': {2: 256, 6: 257, 10: 22, 18: 241, 46: 13, 54: 77, 58: 49, 62: 43}}
INFO 05-03 18:02:18.952398.952398 lmp.py:1005] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.480ms | allocate_experts_across_cpu_gpu: 0.281ms
INFO 05-03 18:02:18.952190.952190 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.100799560546875e-05 seconds
INFO 05-03 18:02:18.953978.953978 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000568389892578125 seconds
INFO 05-03 18:02:18.966988.966988 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011825323104858398 seconds
INFO 05-03 18:02:18.982296.982296 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.016412973403930664 seconds
INFO 05-03 18:02:18.983388.983388 mlpmodule.py:2707] [fused_experts] gmm total=1.265ms E=32 S=702 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.985286.985286 mlpmodule.py:2707] [fused_experts] gmm total=1.427ms E=32 S=1300 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.987422.987422 mlpmodule.py:2707] [fused_experts] gmm total=1.151ms E=32 S=1119 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.988084.988084 mlpmodule.py:2707] [fused_experts] gmm total=1.243ms E=32 S=975 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:18.989574.989574 lmp.py:1149] [layer_moe_fused] experts compute time: 0.00644683837890625 seconds
INFO 05-03 18:02:18.989823.989823 lmp.py:1160] [layer_moe_fused] to time: 2.86102294921875e-05 seconds
INFO 05-03 18:02:18.989525.989525 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002410411834716797 seconds
DEBUG 05-03 18:02:18.989914.989914 cuda_h.py:27] end *layer_moe_fused cost 37.943 ms
DEBUG 05-03 18:02:18.989141.989141 cuda_h.py:27] end prefill_layer cost 42.564 ms
DEBUG 05-03 18:02:18.990335.990335 lmp.py:711] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-03 18:02:18.990178.990178 lmp.py:675] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-03 18:02:18.994993.994993 cuda_h.py:27] end *sagl cost 4.140 ms
experts_cpu_alloc {'expert_ids': [67, 91, 103, 27, 63, 83, 107, 59, 87, 51, 24, 68, 104, 72, 37, 9, 33, 53, 61, 101, 81, 46, 94, 98, 102, 114, 42, 58, 66, 30, 38, 74, 34], 'token_total': 136, 'token_per_expert': {67: 1, 91: 1, 103: 4, 27: 6, 63: 6, 83: 6, 107: 7, 59: 12, 87: 18, 51: 22, 24: 1, 68: 1, 104: 1, 72: 4, 37: 1, 9: 2, 33: 2, 53: 2, 61: 2, 101: 2, 81: 4, 46: 1, 94: 1, 98: 1, 102: 1, 114: 1, 42: 2, 58: 2, 66: 2, 30: 3, 38: 3, 74: 6, 34: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 35, 47, 99, 111, 119, 127], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1476, 'token_per_expert': {3: 454, 7: 257, 19: 127, 35: 144, 47: 132, 99: 80, 111: 73, 119: 161, 127: 48}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 32, 36, 44, 48, 60], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 630, 'token_per_expert': {0: 256, 4: 256, 16: 21, 32: 26, 36: 49, 44: 9, 48: 5, 60: 8}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 21, 69, 97, 113], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 924, 'token_per_expert': {1: 256, 5: 268, 13: 58, 17: 7, 21: 223, 69: 79, 97: 15, 113: 18}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 54, 70, 82, 86, 106], 'expert_count': 8, 'target_expert_count': 8, 'token_total': 930, 'token_per_expert': {2: 256, 6: 256, 26: 21, 54: 91, 70: 102, 82: 23, 86: 124, 106: 57}}
INFO 05-03 18:02:18.995094.995094 lmp.py:1005] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.487ms | allocate_experts_across_cpu_gpu: 0.272ms
INFO 05-03 18:02:18.995277.995277 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.982948303222656e-05 seconds
INFO 05-03 18:02:18.996695.996695 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005626678466796875 seconds
INFO 05-03 18:02:19.007735.007735 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010550737380981445 seconds
INFO 05-03 18:02:19.016035.016035 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.008637428283691406 seconds
INFO 05-03 18:02:19.017127.017127 mlpmodule.py:2707] [fused_experts] gmm total=1.474ms E=32 S=1559 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.019985.019985 mlpmodule.py:2707] [fused_experts] gmm total=1.028ms E=32 S=637 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.020173.020173 mlpmodule.py:2707] [fused_experts] gmm total=1.130ms E=32 S=939 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.022204.022204 mlpmodule.py:2707] [fused_experts] gmm total=1.215ms E=32 S=961 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.022248.022248 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006081819534301758 seconds
INFO 05-03 18:02:19.022258.022258 lmp.py:1160] [layer_moe_fused] to time: 2.9325485229492188e-05 seconds
INFO 05-03 18:02:19.022245.022245 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00024080276489257812 seconds
DEBUG 05-03 18:02:19.023514.023514 cuda_h.py:27] end *layer_moe_fused cost 28.528 ms
DEBUG 05-03 18:02:19.023794.023794 cuda_h.py:27] end prefill_layer cost 33.034 ms
DEBUG 05-03 18:02:19.023800.023800 lmp.py:711] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-03 18:02:19.023093.023093 lmp.py:675] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-03 18:02:19.027156.027156 cuda_h.py:27] end *sagl cost 3.263 ms
experts_cpu_alloc {'expert_ids': [95, 23, 99, 35, 63, 83, 103, 40, 56, 76, 104, 112, 16, 52, 64, 124, 53, 81, 93, 41, 73, 89, 113, 9, 13, 37, 125, 69, 50, 58, 46, 102, 18, 70, 38, 90], 'token_total': 104, 'token_per_expert': {95: 1, 23: 2, 99: 4, 35: 5, 63: 5, 83: 5, 103: 19, 40: 1, 56: 1, 76: 1, 104: 1, 112: 1, 16: 2, 52: 2, 64: 2, 124: 2, 53: 1, 81: 1, 93: 1, 41: 2, 73: 2, 89: 2, 113: 2, 9: 3, 13: 3, 37: 3, 125: 3, 69: 4, 50: 1, 58: 1, 46: 2, 102: 2, 18: 3, 70: 4, 38: 5, 90: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 31, 47, 55, 71, 111, 119], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1183, 'token_per_expert': {3: 256, 7: 388, 27: 20, 31: 91, 47: 28, 55: 112, 71: 123, 111: 65, 119: 100}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 28, 36, 44, 48, 68], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 968, 'token_per_expert': {0: 310, 4: 259, 8: 167, 12: 69, 28: 3, 36: 22, 44: 8, 48: 3, 68: 127}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 25, 29, 57, 61, 65, 77, 105], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 752, 'token_per_expert': {1: 256, 5: 256, 25: 118, 29: 15, 57: 11, 61: 38, 65: 30, 77: 8, 105: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 42, 54, 78, 98, 114], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1089, 'token_per_expert': {2: 259, 6: 345, 10: 12, 22: 172, 42: 72, 54: 34, 78: 95, 98: 69, 114: 31}}
INFO 05-03 18:02:19.027107.027107 lmp.py:1005] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.481ms | allocate_experts_across_cpu_gpu: 0.346ms
INFO 05-03 18:02:19.028939.028939 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.124641418457031e-05 seconds
INFO 05-03 18:02:19.028934.028934 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005462169647216797 seconds
INFO 05-03 18:02:19.040267.040267 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011011838912963867 seconds
INFO 05-03 18:02:19.047180.047180 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.007172822952270508 seconds
INFO 05-03 18:02:19.049078.049078 mlpmodule.py:2707] [fused_experts] gmm total=1.399ms E=32 S=1224 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.050072.050072 mlpmodule.py:2707] [fused_experts] gmm total=1.341ms E=32 S=981 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.052299.052299 mlpmodule.py:2707] [fused_experts] gmm total=1.272ms E=32 S=779 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.054141.054141 mlpmodule.py:2707] [fused_experts] gmm total=1.287ms E=32 S=1112 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.054021.054021 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006649494171142578 seconds
INFO 05-03 18:02:19.054085.054085 lmp.py:1160] [layer_moe_fused] to time: 2.9802322387695312e-05 seconds
INFO 05-03 18:02:19.054541.054541 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00023674964904785156 seconds
DEBUG 05-03 18:02:19.055381.055381 cuda_h.py:27] end *layer_moe_fused cost 28.209 ms
DEBUG 05-03 18:02:19.055131.055131 cuda_h.py:27] end prefill_layer cost 31.804 ms
DEBUG 05-03 18:02:19.055371.055371 lmp.py:711] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-03 18:02:19.055297.055297 lmp.py:675] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-03 18:02:19.060402.060402 cuda_h.py:27] end *sagl cost 4.100 ms
experts_cpu_alloc {'expert_ids': [19, 27, 75, 103, 115, 87, 111, 67, 123, 31, 16, 100, 36, 124, 96, 112, 56, 88, 17, 33, 93, 121, 13, 53, 113, 18, 26, 46, 110, 34, 30, 50, 66, 82, 78, 38, 14], 'token_total': 216, 'token_per_expert': {19: 1, 27: 1, 75: 1, 103: 1, 115: 1, 87: 2, 111: 2, 67: 3, 123: 3, 31: 5, 16: 1, 100: 1, 36: 4, 124: 6, 96: 8, 112: 9, 56: 10, 88: 11, 17: 1, 33: 1, 93: 1, 121: 1, 13: 2, 53: 3, 113: 7, 18: 2, 26: 2, 46: 2, 110: 2, 34: 3, 30: 6, 50: 7, 66: 7, 82: 9, 78: 24, 38: 29, 14: 37}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 23, 47, 51, 55, 63, 91, 99], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 990, 'token_per_expert': {3: 266, 7: 263, 15: 30, 23: 11, 47: 62, 51: 90, 55: 185, 63: 6, 91: 48, 99: 29}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 40, 52, 60, 68, 76, 84], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1031, 'token_per_expert': {0: 277, 4: 256, 12: 24, 20: 32, 40: 12, 52: 38, 60: 40, 68: 84, 76: 186, 84: 82}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 41, 65, 81, 89, 105], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 832, 'token_per_expert': {1: 256, 5: 256, 9: 38, 21: 79, 41: 15, 65: 31, 81: 12, 89: 120, 105: 25}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 62, 70, 94, 102, 126], 'expert_count': 9, 'target_expert_count': 9, 'token_total': 1027, 'token_per_expert': {2: 271, 6: 256, 10: 44, 22: 43, 62: 92, 70: 47, 94: 43, 102: 68, 126: 163}}
INFO 05-03 18:02:19.060392.060392 lmp.py:1005] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.488ms | allocate_experts_across_cpu_gpu: 0.295ms
INFO 05-03 18:02:19.061508.061508 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.410743713378906e-05 seconds
INFO 05-03 18:02:19.061916.061916 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005862712860107422 seconds
INFO 05-03 18:02:19.073547.073547 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011370658874511719 seconds
INFO 05-03 18:02:19.076141.076141 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.002398252487182617 seconds
INFO 05-03 18:02:19.078835.078835 mlpmodule.py:2707] [fused_experts] gmm total=1.627ms E=32 S=1010 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.079588.079588 mlpmodule.py:2707] [fused_experts] gmm total=1.236ms E=32 S=1081 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.081702.081702 mlpmodule.py:2707] [fused_experts] gmm total=1.101ms E=32 S=848 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.082258.082258 mlpmodule.py:2707] [fused_experts] gmm total=1.251ms E=32 S=1157 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.083580.083580 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006524562835693359 seconds
INFO 05-03 18:02:19.083306.083306 lmp.py:1160] [layer_moe_fused] to time: 2.9802322387695312e-05 seconds
INFO 05-03 18:02:19.083531.083531 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00024247169494628906 seconds
DEBUG 05-03 18:02:19.083231.083231 cuda_h.py:27] end *layer_moe_fused cost 23.664 ms
DEBUG 05-03 18:02:19.083127.083127 cuda_h.py:27] end prefill_layer cost 28.139 ms
DEBUG 05-03 18:02:19.083413.083413 lmp.py:711] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-03 18:02:19.084935.084935 lmp.py:675] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-03 18:02:19.087999.087999 cuda_h.py:27] end *sagl cost 3.116 ms
experts_cpu_alloc {'expert_ids': [19, 67, 107, 47, 63, 87, 35, 95, 88, 96, 16, 24, 48, 112, 68, 28, 33, 73, 89, 125, 57, 49, 61, 53, 37, 9, 45, 85, 14, 62, 114, 94, 98, 90, 66, 18, 30, 118, 10, 70], 'token_total': 148, 'token_per_expert': {19: 1, 67: 1, 107: 1, 47: 2, 63: 2, 87: 3, 35: 5, 95: 8, 88: 1, 96: 1, 16: 2, 24: 2, 48: 3, 112: 3, 68: 6, 28: 10, 33: 1, 73: 1, 89: 1, 125: 1, 57: 2, 49: 3, 61: 3, 53: 4, 37: 6, 9: 7, 45: 7, 85: 7, 14: 1, 62: 1, 114: 1, 94: 2, 98: 2, 90: 3, 66: 5, 18: 6, 30: 7, 118: 7, 10: 9, 70: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 31, 51, 99, 103, 111, 115, 127], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 1059, 'token_per_expert': {3: 256, 7: 256, 11: 18, 27: 110, 31: 18, 51: 28, 99: 152, 103: 84, 111: 21, 115: 68, 127: 48}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 36, 40, 44, 64, 76, 80, 120], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 681, 'token_per_expert': {0: 256, 4: 256, 8: 31, 36: 19, 40: 46, 44: 17, 64: 22, 76: 11, 80: 10, 120: 13}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 21, 41, 69, 77, 101, 117], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 999, 'token_per_expert': {1: 256, 5: 264, 13: 23, 17: 63, 21: 67, 41: 80, 69: 60, 77: 76, 101: 48, 117: 62}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 38, 46, 54, 78, 82, 122, 126], 'expert_count': 10, 'target_expert_count': 10, 'token_total': 1209, 'token_per_expert': {2: 340, 6: 256, 22: 121, 38: 72, 46: 15, 54: 13, 78: 93, 82: 150, 122: 43, 126: 106}}
INFO 05-03 18:02:19.088651.088651 lmp.py:1005] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.500ms | allocate_experts_across_cpu_gpu: 0.318ms
INFO 05-03 18:02:19.088536.088536 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 4.7206878662109375e-05 seconds
INFO 05-03 18:02:19.089927.089927 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005648136138916016 seconds
INFO 05-03 18:02:19.101917.101917 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012229204177856445 seconds
INFO 05-03 18:02:19.117077.117077 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.014928817749023438 seconds
INFO 05-03 18:02:19.118277.118277 mlpmodule.py:2707] [fused_experts] gmm total=1.340ms E=32 S=1082 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.119994.119994 mlpmodule.py:2707] [fused_experts] gmm total=1.189ms E=32 S=709 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.121623.121623 mlpmodule.py:2707] [fused_experts] gmm total=1.249ms E=32 S=1042 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.123363.123363 mlpmodule.py:2707] [fused_experts] gmm total=1.423ms E=32 S=1263 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.123977.123977 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006447792053222656 seconds
INFO 05-03 18:02:19.123246.123246 lmp.py:1160] [layer_moe_fused] to time: 3.0994415283203125e-05 seconds
INFO 05-03 18:02:19.123750.123750 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002391338348388672 seconds
DEBUG 05-03 18:02:19.124893.124893 cuda_h.py:27] end *layer_moe_fused cost 36.917 ms
DEBUG 05-03 18:02:19.124736.124736 cuda_h.py:27] end prefill_layer cost 40.337 ms
DEBUG 05-03 18:02:19.124307.124307 lmp.py:711] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-03 18:02:19.124422.124422 lmp.py:675] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-03 18:02:19.128427.128427 cuda_h.py:27] end *sagl cost 3.288 ms
experts_cpu_alloc {'expert_ids': [11, 47, 91, 103, 111, 35, 87, 23, 92, 100, 24, 40, 68, 104, 124, 52, 112, 44, 120, 8, 12, 17, 69, 81, 33, 77, 93, 101, 25, 9, 21, 14, 18, 30, 34, 46, 122, 106, 114, 26, 118, 126, 22, 58, 66, 70], 'token_total': 113, 'token_per_expert': {11: 1, 47: 1, 91: 1, 103: 1, 111: 1, 35: 2, 87: 2, 23: 3, 92: 1, 100: 1, 24: 2, 40: 2, 68: 2, 104: 2, 124: 2, 52: 3, 112: 3, 44: 4, 120: 4, 8: 5, 12: 8, 17: 1, 69: 1, 81: 1, 33: 2, 77: 2, 93: 3, 101: 3, 25: 4, 9: 5, 21: 5, 14: 1, 18: 1, 30: 1, 34: 1, 46: 1, 122: 1, 106: 2, 114: 2, 26: 3, 118: 3, 126: 3, 22: 4, 58: 4, 66: 4, 70: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39, 51, 55, 63, 67, 75, 99, 115, 123], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1017, 'token_per_expert': {3: 256, 7: 260, 19: 169, 39: 3, 51: 3, 55: 13, 63: 4, 67: 26, 75: 122, 99: 27, 115: 3, 123: 131}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 28, 32, 60, 72, 84, 88, 108, 116], 'expert_count': 12, 'target_expert_count': 12, 'token_total': 1224, 'token_per_expert': {0: 337, 4: 257, 16: 10, 20: 12, 28: 15, 32: 18, 60: 202, 72: 48, 84: 105, 88: 26, 108: 158, 116: 36}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 41, 49, 53, 65, 73, 105, 109, 113, 121], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 909, 'token_per_expert': {1: 372, 5: 256, 41: 20, 49: 23, 53: 18, 65: 5, 73: 49, 105: 60, 109: 75, 113: 9, 121: 22}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 38, 50, 54, 62, 74, 82, 86, 110], 'expert_count': 11, 'target_expert_count': 11, 'token_total': 833, 'token_per_expert': {2: 256, 6: 322, 10: 9, 38: 13, 50: 12, 54: 9, 62: 91, 74: 54, 82: 48, 86: 11, 110: 8}}
INFO 05-03 18:02:19.129359.129359 lmp.py:1005] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.498ms | allocate_experts_across_cpu_gpu: 0.352ms
INFO 05-03 18:02:19.129774.129774 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.316734313964844e-05 seconds
INFO 05-03 18:02:19.130084.130084 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005793571472167969 seconds
INFO 05-03 18:02:19.143750.143750 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012443065643310547 seconds
INFO 05-03 18:02:19.149004.149004 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0064585208892822266 seconds
INFO 05-03 18:02:19.151410.151410 mlpmodule.py:2707] [fused_experts] gmm total=1.355ms E=32 S=1029 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.152774.152774 mlpmodule.py:2707] [fused_experts] gmm total=1.517ms E=32 S=1263 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.154349.154349 mlpmodule.py:2707] [fused_experts] gmm total=1.368ms E=32 S=936 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.156542.156542 mlpmodule.py:2707] [fused_experts] gmm total=1.475ms E=32 S=868 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.156620.156620 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007047414779663086 seconds
INFO 05-03 18:02:19.156776.156776 lmp.py:1160] [layer_moe_fused] to time: 3.0994415283203125e-05 seconds
INFO 05-03 18:02:19.157101.157101 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002472400665283203 seconds
DEBUG 05-03 18:02:19.157523.157523 cuda_h.py:27] end *layer_moe_fused cost 29.309 ms
DEBUG 05-03 18:02:19.157512.157512 cuda_h.py:27] end prefill_layer cost 32.966 ms
DEBUG 05-03 18:02:19.157752.157752 lmp.py:711] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-03 18:02:19.157256.157256 lmp.py:675] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-03 18:02:19.161299.161299 cuda_h.py:27] end *sagl cost 3.490 ms
experts_cpu_alloc {'expert_ids': [123, 87, 111, 39, 99, 115, 47, 103, 55, 16, 48, 56, 96, 12, 20, 72, 40, 84, 28, 76, 36, 57, 85, 97, 17, 113, 9, 29, 105, 89, 109, 125, 93, 21, 101, 45, 69, 94, 98, 90, 102, 82, 106, 114, 126, 74, 26, 14, 66, 110, 58, 42], 'token_total': 212, 'token_per_expert': {123: 1, 87: 2, 111: 2, 39: 4, 99: 4, 115: 4, 47: 5, 103: 5, 55: 6, 16: 1, 48: 1, 56: 1, 96: 1, 12: 2, 20: 2, 72: 2, 40: 3, 84: 3, 28: 4, 76: 4, 36: 5, 57: 1, 85: 1, 97: 1, 17: 2, 113: 3, 9: 4, 29: 4, 105: 4, 89: 5, 109: 5, 125: 6, 93: 7, 21: 8, 101: 8, 45: 9, 69: 10, 94: 1, 98: 1, 90: 2, 102: 2, 82: 3, 106: 3, 114: 3, 126: 3, 74: 4, 26: 6, 14: 8, 66: 8, 110: 8, 58: 9, 42: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 35, 43, 51, 59, 63, 71, 79, 83, 107], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 968, 'token_per_expert': {3: 256, 7: 260, 11: 204, 23: 7, 35: 11, 43: 8, 51: 11, 59: 24, 63: 27, 71: 25, 79: 12, 83: 38, 107: 85}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 24, 44, 52, 60, 88, 92, 112, 116, 120, 124], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 887, 'token_per_expert': {0: 256, 4: 256, 8: 107, 24: 50, 44: 7, 52: 18, 60: 26, 88: 34, 92: 39, 112: 53, 116: 24, 120: 7, 124: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 33, 37, 41, 49, 53, 61, 65, 73, 77], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 1093, 'token_per_expert': {1: 259, 5: 258, 13: 76, 25: 47, 33: 103, 37: 87, 41: 11, 49: 50, 53: 25, 61: 12, 65: 18, 73: 41, 77: 106}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 30, 38, 46, 50, 70, 78, 118, 122], 'expert_count': 13, 'target_expert_count': 13, 'token_total': 936, 'token_per_expert': {2: 265, 6: 258, 10: 73, 18: 51, 22: 29, 30: 93, 38: 34, 46: 14, 50: 23, 70: 14, 78: 14, 118: 33, 122: 35}}
INFO 05-03 18:02:19.162498.162498 lmp.py:1005] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.509ms | allocate_experts_across_cpu_gpu: 0.394ms
INFO 05-03 18:02:19.162827.162827 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 5.936622619628906e-05 seconds
INFO 05-03 18:02:19.163454.163454 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006015300750732422 seconds
INFO 05-03 18:02:19.177162.177162 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013642072677612305 seconds
INFO 05-03 18:02:19.179324.179324 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013575553894042969 seconds
INFO 05-03 18:02:19.180854.180854 mlpmodule.py:2707] [fused_experts] gmm total=1.477ms E=32 S=1001 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.182955.182955 mlpmodule.py:2707] [fused_experts] gmm total=1.350ms E=32 S=916 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.184040.184040 mlpmodule.py:2707] [fused_experts] gmm total=1.437ms E=32 S=1171 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.185118.185118 mlpmodule.py:2707] [fused_experts] gmm total=1.412ms E=32 S=1008 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.186500.186500 lmp.py:1149] [layer_moe_fused] experts compute time: 0.006933450698852539 seconds
INFO 05-03 18:02:19.186941.186941 lmp.py:1160] [layer_moe_fused] to time: 3.0279159545898438e-05 seconds
INFO 05-03 18:02:19.186066.186066 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002410411834716797 seconds
DEBUG 05-03 18:02:19.187556.187556 cuda_h.py:27] end *layer_moe_fused cost 25.385 ms
DEBUG 05-03 18:02:19.187412.187412 cuda_h.py:27] end prefill_layer cost 29.187 ms
DEBUG 05-03 18:02:19.187129.187129 lmp.py:711] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-03 18:02:19.187505.187505 lmp.py:675] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-03 18:02:19.191951.191951 cuda_h.py:27] end *sagl cost 3.662 ms
experts_cpu_alloc {'expert_ids': [23, 27, 31, 39, 87, 35, 63, 83, 15, 59, 79, 95, 8, 40, 108, 116, 28, 48, 64, 120, 76, 92, 112, 104, 52, 20, 124, 24, 17, 37, 93, 97, 49, 101, 61, 69, 109, 89, 25, 9, 45, 113, 53, 62, 82, 94, 30, 50, 66, 70, 90, 98, 106, 18, 78, 10, 74], 'token_total': 180, 'token_per_expert': {23: 1, 27: 1, 31: 1, 39: 1, 87: 1, 35: 2, 63: 2, 83: 2, 15: 4, 59: 4, 79: 5, 95: 5, 8: 1, 40: 1, 108: 1, 116: 1, 28: 3, 48: 3, 64: 3, 120: 3, 76: 4, 92: 4, 112: 4, 104: 5, 52: 6, 20: 7, 124: 7, 24: 8, 17: 1, 37: 1, 93: 1, 97: 1, 49: 2, 101: 2, 61: 3, 69: 3, 109: 3, 89: 4, 25: 5, 9: 6, 45: 8, 113: 8, 53: 11, 62: 1, 82: 1, 94: 1, 30: 2, 50: 2, 66: 2, 70: 2, 90: 2, 98: 2, 106: 2, 18: 3, 78: 3, 10: 4, 74: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 47, 55, 67, 71, 75, 103, 107, 111, 119, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1023, 'token_per_expert': {3: 257, 7: 274, 11: 19, 19: 20, 47: 11, 55: 19, 67: 38, 71: 106, 75: 7, 103: 7, 107: 35, 111: 59, 119: 50, 123: 8, 127: 113}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 32, 36, 44, 56, 60, 68, 72, 80, 84, 96, 100], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 737, 'token_per_expert': {0: 258, 4: 256, 16: 12, 32: 10, 36: 10, 44: 55, 56: 9, 60: 18, 68: 24, 72: 28, 80: 8, 84: 8, 96: 9, 100: 32}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 33, 41, 57, 65, 73, 77, 81, 85, 105, 117], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 1305, 'token_per_expert': {1: 257, 5: 273, 13: 84, 29: 106, 33: 68, 41: 173, 57: 61, 65: 58, 73: 11, 77: 135, 81: 11, 85: 31, 105: 18, 117: 19}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 26, 34, 38, 46, 54, 58, 86, 102, 110, 126], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 851, 'token_per_expert': {2: 259, 6: 256, 14: 6, 22: 25, 26: 7, 34: 6, 38: 9, 46: 83, 54: 57, 58: 6, 86: 69, 102: 35, 110: 16, 126: 17}}
INFO 05-03 18:02:19.192469.192469 lmp.py:1005] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.534ms | allocate_experts_across_cpu_gpu: 0.429ms
INFO 05-03 18:02:19.192666.192666 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.628036499023438e-05 seconds
INFO 05-03 18:02:19.193262.193262 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006649494171142578 seconds
INFO 05-03 18:02:19.207564.207564 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013976812362670898 seconds
INFO 05-03 18:02:19.217292.217292 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.01012277603149414 seconds
INFO 05-03 18:02:19.219007.219007 mlpmodule.py:2707] [fused_experts] gmm total=1.673ms E=32 S=1052 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.221410.221410 mlpmodule.py:2707] [fused_experts] gmm total=1.457ms E=32 S=798 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.223131.223131 mlpmodule.py:2707] [fused_experts] gmm total=1.442ms E=32 S=1364 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.225746.225746 mlpmodule.py:2707] [fused_experts] gmm total=1.415ms E=32 S=882 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.225459.225459 lmp.py:1149] [layer_moe_fused] experts compute time: 0.00727081298828125 seconds
INFO 05-03 18:02:19.225853.225853 lmp.py:1160] [layer_moe_fused] to time: 3.0994415283203125e-05 seconds
INFO 05-03 18:02:19.225761.225761 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00025725364685058594 seconds
DEBUG 05-03 18:02:19.226835.226835 cuda_h.py:27] end *layer_moe_fused cost 35.012 ms
DEBUG 05-03 18:02:19.226208.226208 cuda_h.py:27] end prefill_layer cost 38.965 ms
DEBUG 05-03 18:02:19.226984.226984 lmp.py:711] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-03 18:02:19.226793.226793 lmp.py:675] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-03 18:02:19.229733.229733 cuda_h.py:27] end *sagl cost 3.498 ms
experts_cpu_alloc {'expert_ids': [67, 75, 99, 111, 47, 95, 23, 55, 51, 27, 71, 103, 107, 59, 8, 56, 40, 64, 84, 112, 20, 60, 68, 72, 80, 108, 28, 44, 120, 32, 13, 45, 89, 73, 37, 41, 57, 69, 77, 53, 101, 117, 17, 81, 97, 66, 78, 90, 22, 122, 30, 110, 86, 74, 126, 34, 94, 102, 54], 'token_total': 256, 'token_per_expert': {67: 1, 75: 1, 99: 1, 111: 1, 47: 2, 95: 2, 23: 3, 55: 3, 51: 5, 27: 8, 71: 8, 103: 9, 107: 9, 59: 11, 8: 1, 56: 1, 40: 2, 64: 2, 84: 3, 112: 3, 20: 4, 60: 4, 68: 4, 72: 4, 80: 5, 108: 5, 28: 6, 44: 8, 120: 8, 32: 11, 13: 1, 45: 1, 89: 1, 73: 2, 37: 3, 41: 3, 57: 3, 69: 3, 77: 3, 53: 4, 101: 4, 117: 4, 17: 6, 81: 6, 97: 6, 66: 1, 78: 1, 90: 1, 22: 2, 122: 2, 30: 3, 110: 3, 86: 5, 74: 6, 126: 7, 34: 9, 94: 10, 102: 10, 54: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 31, 35, 39, 43, 63, 79, 83, 87, 91, 123, 127], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 1045, 'token_per_expert': {3: 274, 7: 307, 11: 26, 15: 34, 31: 32, 35: 17, 39: 34, 43: 67, 63: 71, 79: 59, 83: 27, 87: 18, 91: 11, 123: 25, 127: 43}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 24, 36, 48, 52, 76, 88, 92, 96, 100, 104, 116, 124], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 930, 'token_per_expert': {0: 257, 4: 256, 16: 16, 24: 72, 36: 14, 48: 18, 52: 38, 76: 41, 88: 30, 92: 26, 96: 31, 100: 55, 104: 14, 116: 20, 124: 42}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 21, 25, 29, 33, 61, 65, 85, 93, 109, 113, 121, 125], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 961, 'token_per_expert': {1: 280, 5: 258, 9: 20, 21: 29, 25: 100, 29: 9, 33: 77, 61: 28, 65: 47, 85: 17, 93: 23, 109: 24, 113: 35, 121: 8, 125: 6}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 26, 46, 50, 62, 70, 82, 106, 114, 118], 'expert_count': 14, 'target_expert_count': 14, 'token_total': 904, 'token_per_expert': {2: 260, 6: 261, 10: 15, 14: 25, 18: 20, 26: 13, 46: 22, 50: 30, 62: 46, 70: 99, 82: 46, 106: 11, 114: 30, 118: 26}}
INFO 05-03 18:02:19.231258.231258 lmp.py:1005] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.533ms | allocate_experts_across_cpu_gpu: 0.436ms
INFO 05-03 18:02:19.231879.231879 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.413459777832031e-05 seconds
INFO 05-03 18:02:19.231462.231462 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00063323974609375 seconds
INFO 05-03 18:02:19.247396.247396 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014642953872680664 seconds
INFO 05-03 18:02:19.251547.251547 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.004571676254272461 seconds
INFO 05-03 18:02:19.253083.253083 mlpmodule.py:2707] [fused_experts] gmm total=1.653ms E=32 S=1109 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.255856.255856 mlpmodule.py:2707] [fused_experts] gmm total=1.456ms E=32 S=1001 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.257688.257688 mlpmodule.py:2707] [fused_experts] gmm total=1.575ms E=32 S=1011 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.259738.259738 mlpmodule.py:2707] [fused_experts] gmm total=1.344ms E=32 S=975 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.259120.259120 lmp.py:1149] [layer_moe_fused] experts compute time: 0.007328987121582031 seconds
INFO 05-03 18:02:19.259514.259514 lmp.py:1160] [layer_moe_fused] to time: 3.24249267578125e-05 seconds
INFO 05-03 18:02:19.259329.259329 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.0002589225769042969 seconds
DEBUG 05-03 18:02:19.260919.260919 cuda_h.py:27] end *layer_moe_fused cost 30.215 ms
DEBUG 05-03 18:02:19.260908.260908 cuda_h.py:27] end prefill_layer cost 33.897 ms
DEBUG 05-03 18:02:19.260208.260208 lmp.py:711] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-03 18:02:19.260063.260063 lmp.py:675] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-03 18:02:19.266709.266709 cuda_h.py:27] end *sagl cost 5.598 ms
experts_cpu_alloc {'expert_ids': [123, 99, 19, 39, 87, 127, 35, 71, 75, 115, 15, 103, 55, 11, 63, 43, 28, 40, 68, 8, 52, 84, 88, 96, 108, 72, 60, 16, 32, 117, 9, 69, 97, 33, 65, 93, 109, 13, 45, 89, 125, 81, 113, 21, 10, 90, 94, 114, 46, 58, 106, 18, 22, 38, 78, 14, 62, 86, 54], 'token_total': 309, 'token_per_expert': {123: 2, 99: 3, 19: 4, 39: 4, 87: 4, 127: 4, 35: 5, 71: 6, 75: 6, 115: 6, 15: 9, 103: 9, 55: 10, 11: 11, 63: 12, 43: 15, 28: 1, 40: 1, 68: 1, 8: 2, 52: 4, 84: 4, 88: 4, 96: 4, 108: 4, 72: 5, 60: 6, 16: 9, 32: 9, 117: 1, 9: 2, 69: 2, 97: 2, 33: 3, 65: 3, 93: 3, 109: 4, 13: 6, 45: 6, 89: 7, 125: 7, 81: 8, 113: 8, 21: 9, 10: 1, 90: 1, 94: 1, 114: 1, 46: 2, 58: 2, 106: 2, 18: 5, 22: 5, 38: 5, 78: 8, 14: 9, 62: 9, 86: 11, 54: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 31, 51, 59, 67, 79, 83, 91, 95, 107, 111, 119], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 960, 'token_per_expert': {3: 275, 7: 265, 23: 27, 27: 31, 31: 29, 51: 16, 59: 53, 67: 40, 79: 35, 83: 30, 91: 17, 95: 71, 107: 26, 111: 17, 119: 28}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 24, 36, 44, 76, 80, 92, 100, 104, 112, 116, 120, 124], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 933, 'token_per_expert': {0: 280, 4: 285, 12: 55, 24: 43, 36: 40, 44: 31, 76: 13, 80: 26, 92: 11, 100: 35, 104: 69, 112: 15, 116: 11, 120: 9, 124: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 25, 29, 37, 41, 49, 61, 73, 77, 85, 101, 105, 121], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 989, 'token_per_expert': {1: 256, 5: 306, 17: 27, 25: 25, 29: 26, 37: 18, 41: 68, 49: 9, 61: 66, 73: 20, 77: 15, 85: 41, 101: 11, 105: 48, 121: 53}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 30, 34, 42, 50, 66, 70, 98, 102, 110, 118, 122, 126], 'expert_count': 15, 'target_expert_count': 15, 'token_total': 905, 'token_per_expert': {2: 261, 6: 281, 26: 13, 30: 63, 34: 53, 42: 15, 50: 21, 66: 13, 70: 71, 98: 14, 102: 33, 110: 27, 118: 12, 122: 15, 126: 13}}
INFO 05-03 18:02:19.267883.267883 lmp.py:1005] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.517ms | allocate_experts_across_cpu_gpu: 0.436ms
INFO 05-03 18:02:19.267550.267550 lmp.py:1019] [layer_moe_fused] get_experts_task_ids time: 6.341934204101562e-05 seconds
INFO 05-03 18:02:19.267999.267999 lmp.py:1027] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006239414215087891 seconds
INFO 05-03 18:02:19.275342.275342 lmp.py:1074] [layer_moe_fused] prepare_fused_expert_work_items time: 0.006906032562255859 seconds
INFO 05-03 18:02:19.284450.284450 lmp.py:1084] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.008821725845336914 seconds
INFO 05-03 18:02:19.286099.286099 mlpmodule.py:2707] [fused_experts] gmm total=1.668ms E=32 S=1070 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.287326.287326 mlpmodule.py:2707] [fused_experts] gmm total=1.369ms E=32 S=987 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.289655.289655 mlpmodule.py:2707] [fused_experts] gmm total=1.397ms E=32 S=1060 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.291944.291944 mlpmodule.py:2707] [fused_experts] gmm total=1.368ms E=32 S=979 H=2816 dtype=torch.bfloat16
INFO 05-03 18:02:19.291318.291318 lmp.py:1149] [layer_moe_fused] experts compute time: 0.0070536136627197266 seconds
INFO 05-03 18:02:19.291164.291164 lmp.py:1160] [layer_moe_fused] to time: 3.147125244140625e-05 seconds
INFO 05-03 18:02:19.291773.291773 lmp.py:1166] [layer_moe_fused] scatter_reduce_ time: 0.00024771690368652344 seconds
DEBUG 05-03 18:02:19.292701.292701 cuda_h.py:27] end *layer_moe_fused cost 26.382 ms
DEBUG 05-03 18:02:19.292975.292975 cuda_h.py:27] end prefill_layer cost 32.167 ms
DEBUG 05-03 18:02:19.292274.292274 lmp.py:711] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-03 18:02:19.292414.292414 cuda_h.py:27] end prefill cost 958.103 ms
INFO 05-03 18:02:19.292601.292601 lmp.py:713] prefill time: 0.9581327438354492 seconds
Time taken: 4.75229187309742 seconds
X512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x605df6609030, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
CPUInfer[0x605df8757520]: Goodbye
