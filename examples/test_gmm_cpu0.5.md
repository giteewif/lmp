here pin
INFO 05-06 11:01:33.273981.273981 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-06 11:01:33.814198.814198 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-06 11:01:34.246024.246024 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-06 11:01:34.246977.246977 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 0.974s
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
INFO 05-06 11:01:41.880045.880045 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-06 11:01:42.287042.287042 cuda_h.py:27] end init_cmv_hmv cost 407.736 ms
DEBUG 05-06 11:01:42.296536.296536 cuda_memory_view.py:1366] 
DEBUG 05-06 11:01:42.296536.296536 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.0027854442596435547
DEBUG 05-06 11:01:42.312343.312343 mlpmodule.py:993] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-06 11:01:42.312315.312315 cuda_memory_view.py:1370] 
DEBUG 05-06 11:01:42.312315.312315 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.015635251998901367
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-06 11:01:44.200379.200379 lmp.py:368] init kt-kernel layer 0 ok
INFO 05-06 11:01:44.987328.987328 lmp.py:368] init kt-kernel layer 1 ok
INFO 05-06 11:01:45.785436.785436 lmp.py:368] init kt-kernel layer 2 ok
INFO 05-06 11:01:46.617229.617229 lmp.py:368] init kt-kernel layer 3 ok
INFO 05-06 11:01:47.434235.434235 lmp.py:368] init kt-kernel layer 4 ok
INFO 05-06 11:01:48.261649.261649 lmp.py:368] init kt-kernel layer 5 ok
INFO 05-06 11:01:49.089636.089636 lmp.py:368] init kt-kernel layer 6 ok
INFO 05-06 11:01:49.924307.924307 lmp.py:368] init kt-kernel layer 7 ok
INFO 05-06 11:01:50.752651.752651 lmp.py:368] init kt-kernel layer 8 ok
INFO 05-06 11:01:51.561932.561932 lmp.py:368] init kt-kernel layer 9 ok
INFO 05-06 11:01:52.386279.386279 lmp.py:368] init kt-kernel layer 10 ok
INFO 05-06 11:01:53.205732.205732 lmp.py:368] init kt-kernel layer 11 ok
INFO 05-06 11:01:54.036465.036465 lmp.py:368] init kt-kernel layer 12 ok
INFO 05-06 11:01:54.889034.889034 lmp.py:368] init kt-kernel layer 13 ok
INFO 05-06 11:01:55.730562.730562 lmp.py:368] init kt-kernel layer 14 ok
INFO 05-06 11:01:56.594881.594881 lmp.py:368] init kt-kernel layer 15 ok
INFO 05-06 11:01:57.462593.462593 lmp.py:368] init kt-kernel layer 16 ok
INFO 05-06 11:01:58.286977.286977 lmp.py:368] init kt-kernel layer 17 ok
INFO 05-06 11:01:59.118027.118027 lmp.py:368] init kt-kernel layer 18 ok
INFO 05-06 11:01:59.929236.929236 lmp.py:368] init kt-kernel layer 19 ok
INFO 05-06 11:02:00.735120.735120 lmp.py:368] init kt-kernel layer 20 ok
INFO 05-06 11:02:01.545667.545667 lmp.py:368] init kt-kernel layer 21 ok
INFO 05-06 11:02:02.366428.366428 lmp.py:368] init kt-kernel layer 22 ok
CPUInfer[0x5866ec59ac10]: Hello
WorkerPool[0x5866ec5cddc0] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x5866ff84f310]: Hello
WorkerPool[0x586709301d10] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVINFO 05-06 11:02:03.172167.172167 lmp.py:368] init kt-kernel layer 23 ok
INFO 05-06 11:02:04.001011.001011 lmp.py:368] init kt-kernel layer 24 ok
INFO 05-06 11:02:04.823341.823341 lmp.py:368] init kt-kernel layer 25 ok
INFO 05-06 11:02:05.641127.641127 lmp.py:368] init kt-kernel layer 26 ok
INFO 05-06 11:02:06.453205.453205 lmp.py:368] init kt-kernel layer 27 ok
INFO 05-06 11:02:07.259148.259148 lmp.py:368] init kt-kernel layer 28 ok
INFO 05-06 11:02:08.060676.060676 lmp.py:368] init kt-kernel layer 29 ok
generate input ids cost 0.08986973762512207 s
DEBUG 05-06 11:02:11.150294.150294 cuda_h.py:27] end generate_input_ids cost 3035.532 ms
DEBUG 05-06 11:02:11.151109.151109 cuda_h.py:27] end init_cache cost 0.051 ms
INFO 05-06 11:02:11.164797.164797 lmp.py:2350] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6617276356, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7277174582576769, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-06 11:02:11.164059.164059 lmp.py:2368] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.164035.164035 lmp.py:2368] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.164473.164473 lmp.py:2368] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.164574.164574 lmp.py:2368] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.164619.164619 lmp.py:2368] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.164242.164242 lmp.py:2368] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.165924.165924 lmp.py:2368] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.165310.165310 lmp.py:2368] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.165747.165747 lmp.py:2368] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.165371.165371 lmp.py:2368] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.165868.165868 lmp.py:2368] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.165730.165730 lmp.py:2368] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.165557.165557 lmp.py:2368] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.165466.165466 lmp.py:2368] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166976.166976 lmp.py:2368] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166692.166692 lmp.py:2368] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166738.166738 lmp.py:2368] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166739.166739 lmp.py:2368] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166793.166793 lmp.py:2368] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166210.166210 lmp.py:2368] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166972.166972 lmp.py:2368] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166649.166649 lmp.py:2368] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166889.166889 lmp.py:2368] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166744.166744 lmp.py:2368] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166600.166600 lmp.py:2368] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166694.166694 lmp.py:2368] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166933.166933 lmp.py:2368] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166550.166550 lmp.py:2368] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166264.166264 lmp.py:2368] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:11.166265.166265 lmp.py:2368] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 11:02:11.441284.441284 cuda_h.py:27] end init_loading_placement cost 289.956 ms
DEBUG 05-06 11:02:11.441282.441282 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:02:11.441907.441907 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:02:11 client.py:72] load_into_gpu: gemma4-26B-A4B, 109b45c8-ecb8-4582-86f2-3834186e8a5a
INFO 05-06 11:02:11 client.py:135] Model loaded: gemma4-26B-A4B, 109b45c8-ecb8-4582-86f2-3834186e8a5a
INFO 05-06 11:02:11 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 109b45c8-ecb8-4582-86f2-3834186e8a5a
INFO 05-06 11:02:11 client.py:212] Model loaded
DEBUG 05-06 11:02:11.968177.968177 cuda_h.py:27] end init_general_sagl_loading_async cost 527.508 ms
INFO 05-06 11:02:12.018801.018801 lmp.py:2871] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 11:02:12.118863.118863 cuda_h.py:27] end restore_state_dict cost 99.633 ms
DEBUG 05-06 11:02:12.118113.118113 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:02:12.118565.118565 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:02:12 client.py:72] load_into_gpu: gemma4-26B-A4B, 16111b63-52ff-49cb-bce6-dd7db164d180
INFO 05-06 11:02:12 client.py:135] Model loaded: gemma4-26B-A4B, 16111b63-52ff-49cb-bce6-dd7db164d180
DEBUG 05-06 11:02:12.193998.193998 cuda_h.py:27] end init_experts_loading_async cost 74.474 ms
DEBUG 05-06 11:02:12.224955.224955 cuda_h.py:27] end init_inputs_tokens cost 30.792 ms
DEBUG 05-06 11:02:12.224223.224223 lmp.py:824] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 11:02:12.370141.370141 cuda_h.py:27] end *sagl cost 145.838 ms
experts_cpu_alloc {'expert_ids': [19, 87, 27, 15, 119, 63, 23, 111, 107, 11, 123, 59, 71, 79, 88, 120, 12, 36, 100, 96, 8, 4, 20, 44, 84, 80, 60, 112, 76, 17, 85, 97, 109, 101, 29, 81, 93, 45, 49, 65, 73, 13, 5, 9, 69, 30, 86, 6, 66, 106, 94, 14, 2, 10, 34, 114, 38, 18], 'token_total': 1512, 'token_per_expert': {19: 2, 87: 2, 27: 10, 15: 13, 119: 13, 63: 15, 23: 23, 111: 23, 107: 25, 11: 31, 123: 39, 59: 43, 71: 65, 79: 76, 88: 1, 120: 1, 12: 2, 36: 4, 100: 5, 96: 8, 8: 9, 4: 11, 20: 15, 44: 17, 84: 21, 80: 28, 60: 55, 112: 68, 76: 74, 17: 3, 85: 3, 97: 3, 109: 3, 101: 6, 29: 7, 81: 11, 93: 12, 45: 14, 49: 17, 65: 39, 73: 60, 13: 61, 5: 66, 9: 68, 69: 78, 30: 1, 86: 2, 6: 4, 66: 6, 106: 14, 94: 24, 14: 29, 2: 30, 10: 36, 34: 38, 114: 48, 38: 59, 18: 71}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 51, 55, 67, 75, 83, 91, 99, 103, 115, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4710, 'token_per_expert': {3: 160, 7: 374, 31: 134, 39: 718, 47: 1304, 51: 186, 55: 208, 67: 183, 75: 89, 83: 105, 91: 458, 99: 161, 103: 432, 115: 89, 127: 109}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 24, 28, 32, 48, 52, 64, 68, 72, 92, 104, 108, 116, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3138, 'token_per_expert': {0: 249, 16: 201, 24: 81, 28: 123, 32: 183, 48: 146, 52: 150, 64: 106, 68: 694, 72: 100, 92: 87, 104: 134, 108: 78, 116: 82, 124: 724}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 21, 25, 33, 37, 41, 53, 77, 89, 105, 113, 117, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 3326, 'token_per_expert': {1: 273, 21: 171, 25: 110, 33: 828, 37: 81, 41: 142, 53: 819, 77: 99, 89: 133, 105: 89, 113: 157, 117: 97, 121: 226, 125: 101}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 46, 50, 54, 70, 74, 78, 90, 102, 110, 118, 122, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 3698, 'token_per_expert': {22: 255, 26: 304, 46: 450, 50: 520, 54: 275, 70: 140, 74: 224, 78: 109, 90: 546, 102: 74, 110: 83, 118: 89, 122: 114, 126: 515}}
INFO 05-06 11:02:12.516691.516691 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 61.956ms | allocate_experts_across_cpu_gpu: 0.292ms
INFO 05-06 11:02:12.516445.516445 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.8623809814453125e-05 seconds
INFO 05-06 11:02:12.518240.518240 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0013477802276611328 seconds
INFO 05-06 11:02:12.725293.725293 lmp.py:1387] [layer_moe_fused] to time: 0.0001571178436279297 seconds
INFO 05-06 11:02:12.726150.726150 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:12.727683.727683 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015652179718017578 seconds
INFO 05-06 11:02:12.728482.728482 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006570816040039062 seconds
INFO 05-06 11:02:12.728471.728471 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0023627281188964844 seconds
INFO 05-06 11:02:12.790182.790182 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.06132364273071289 seconds
INFO 05-06 11:02:12.840237.840237 mlpmodule.py:2799] [fused_experts] gmm total=50.230ms E=32 S=5090 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:12.870573.870573 mlpmodule.py:2799] [fused_experts] gmm total=79.435ms E=32 S=3777 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:12.876587.876587 mlpmodule.py:2799] [fused_experts] gmm total=85.708ms E=32 S=3457 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:12.876464.876464 mlpmodule.py:2799] [fused_experts] gmm total=85.610ms E=32 S=4060 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:12.877780.877780 lmp.py:1500] [layer_moe_fused] experts compute time: 0.08759069442749023 seconds
INFO 05-06 11:02:12.878366.878366 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 0.0001232624053955078 seconds
DEBUG 05-06 11:02:12.878938.878938 cuda_h.py:27] end *layer_moe_fused cost 424.272 ms
DEBUG 05-06 11:02:12.892698.892698 cuda_h.py:27] end prefill_layer cost 668.065 ms
DEBUG 05-06 11:02:12.892849.892849 lmp.py:841] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 11:02:12.892388.892388 lmp.py:824] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 11:02:12.895354.895354 cuda_h.py:27] end *sagl cost 2.814 ms
experts_cpu_alloc {'expert_ids': [63, 43, 39, 107, 23, 75, 115, 91, 31, 55, 87, 15, 103, 27, 24, 44, 16, 40, 112, 0, 116, 32, 60, 84, 88, 72, 76, 48, 56, 108, 61, 117, 33, 125, 41, 81, 89, 77, 93, 29, 57, 121, 37, 69, 45, 70, 114, 2, 18, 14, 26, 62, 110, 66, 78, 6, 38, 50, 90], 'token_total': 1235, 'token_per_expert': {63: 3, 43: 4, 39: 5, 107: 5, 23: 6, 75: 7, 115: 8, 91: 13, 31: 14, 55: 15, 87: 20, 15: 21, 103: 22, 27: 33, 24: 3, 44: 6, 16: 7, 40: 20, 112: 21, 0: 23, 116: 24, 32: 26, 60: 26, 84: 26, 88: 26, 72: 29, 76: 29, 48: 39, 56: 50, 108: 50, 61: 2, 117: 2, 33: 5, 125: 8, 41: 10, 81: 10, 89: 14, 77: 16, 93: 25, 29: 27, 57: 28, 121: 33, 37: 36, 69: 40, 45: 47, 70: 2, 114: 5, 2: 9, 18: 12, 14: 15, 26: 17, 62: 24, 110: 24, 66: 29, 78: 29, 6: 35, 38: 40, 50: 49, 90: 61}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 35, 47, 51, 59, 67, 79, 83, 95, 99, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 2511, 'token_per_expert': {3: 61, 7: 157, 11: 48, 35: 63, 47: 157, 51: 236, 59: 150, 67: 446, 79: 74, 83: 35, 95: 62, 99: 547, 119: 149, 123: 36, 127: 290}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 20, 28, 52, 64, 68, 80, 92, 96, 100, 104, 120, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4054, 'token_per_expert': {4: 83, 8: 472, 12: 219, 20: 207, 28: 190, 52: 1229, 64: 89, 68: 692, 80: 150, 92: 67, 96: 204, 100: 205, 104: 65, 120: 100, 124: 82}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 49, 53, 65, 73, 85, 97, 101, 105, 109], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3770, 'token_per_expert': {1: 142, 5: 409, 9: 102, 13: 1123, 21: 71, 25: 164, 49: 104, 53: 159, 65: 154, 73: 96, 85: 102, 97: 491, 101: 52, 105: 71, 109: 530}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 22, 30, 34, 42, 46, 54, 74, 82, 94, 98, 106, 118, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 4814, 'token_per_expert': {10: 623, 22: 467, 30: 967, 34: 64, 42: 161, 46: 199, 54: 229, 74: 63, 82: 791, 94: 118, 98: 62, 106: 213, 118: 291, 122: 566}}
INFO 05-06 11:02:12.897650.897650 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.552ms | allocate_experts_across_cpu_gpu: 0.318ms
INFO 05-06 11:02:12.898767.898767 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.76837158203125e-05 seconds
INFO 05-06 11:02:12.899209.899209 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0013031959533691406 seconds
INFO 05-06 11:02:13.741349.741349 lmp.py:1387] [layer_moe_fused] to time: 0.00022530555725097656 seconds
INFO 05-06 11:02:13.741134.741134 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:13.743722.743722 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001336812973022461 seconds
INFO 05-06 11:02:13.744475.744475 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006380081176757812 seconds
INFO 05-06 11:02:13.744278.744278 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002145051956176758 seconds
INFO 05-06 11:02:13.755377.755377 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011011600494384766 seconds
INFO 05-06 11:02:13.758934.758934 mlpmodule.py:2799] [fused_experts] gmm total=2.898ms E=32 S=2687 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.759077.759077 mlpmodule.py:2799] [fused_experts] gmm total=3.022ms E=32 S=5165 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.759902.759902 mlpmodule.py:2799] [fused_experts] gmm total=3.192ms E=32 S=4073 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.759104.759104 mlpmodule.py:2799] [fused_experts] gmm total=3.333ms E=32 S=4459 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.760512.760512 lmp.py:1500] [layer_moe_fused] experts compute time: 0.005300998687744141 seconds
INFO 05-06 11:02:13.761755.761755 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 6.008148193359375e-05 seconds
DEBUG 05-06 11:02:13.761472.761472 cuda_h.py:27] end *layer_moe_fused cost 864.430 ms
DEBUG 05-06 11:02:13.767422.767422 cuda_h.py:27] end prefill_layer cost 875.021 ms
DEBUG 05-06 11:02:13.767670.767670 lmp.py:841] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 11:02:13.767009.767009 lmp.py:824] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 11:02:13.770809.770809 cuda_h.py:27] end *sagl cost 2.069 ms
experts_cpu_alloc {'expert_ids': [47, 87, 79, 67, 75, 99, 23, 35, 95, 115, 27, 111, 71, 103, 43, 107, 16, 32, 68, 12, 64, 40, 96, 120, 88, 72, 0, 52, 28, 56, 24, 93, 117, 101, 25, 21, 45, 5, 61, 121, 105, 113, 85, 17, 33, 77, 22, 74, 86, 66, 26, 50, 6, 82, 114, 42, 58, 46, 70], 'token_total': 1869, 'token_per_expert': {47: 1, 87: 1, 79: 4, 67: 6, 75: 6, 99: 6, 23: 25, 35: 27, 95: 28, 115: 40, 27: 45, 111: 45, 71: 67, 103: 69, 43: 75, 107: 79, 16: 4, 32: 4, 68: 6, 12: 18, 64: 18, 40: 21, 96: 34, 120: 34, 88: 46, 72: 48, 0: 50, 52: 52, 28: 54, 56: 60, 24: 71, 93: 1, 117: 1, 101: 2, 25: 8, 21: 10, 45: 19, 5: 21, 61: 29, 121: 29, 105: 33, 113: 44, 85: 66, 17: 73, 33: 80, 77: 100, 22: 1, 74: 1, 86: 5, 66: 9, 26: 12, 50: 12, 6: 25, 82: 25, 114: 27, 42: 37, 58: 42, 46: 45, 70: 68}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 31, 51, 55, 59, 63, 83, 91, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4257, 'token_per_expert': {3: 140, 7: 257, 11: 1054, 15: 371, 19: 579, 31: 98, 51: 104, 55: 220, 59: 444, 63: 92, 83: 96, 91: 148, 119: 79, 123: 104, 127: 471}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 20, 36, 44, 48, 60, 76, 80, 84, 100, 104, 108, 116, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3150, 'token_per_expert': {4: 81, 8: 117, 20: 217, 36: 92, 44: 101, 48: 233, 60: 210, 76: 270, 80: 234, 84: 227, 100: 71, 104: 149, 108: 989, 116: 73, 124: 86}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 29, 37, 41, 49, 53, 57, 65, 69, 81, 97, 109, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4118, 'token_per_expert': {1: 416, 9: 434, 13: 405, 29: 312, 37: 271, 41: 541, 49: 115, 53: 185, 57: 125, 65: 180, 69: 109, 81: 391, 97: 140, 109: 143, 125: 351}}
experts_gpu_alloc_device_3 {'expert_ids': [14, 18, 34, 54, 62, 78, 90, 98, 102, 106, 110, 118, 122, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 2990, 'token_per_expert': {14: 124, 18: 170, 34: 151, 54: 389, 62: 567, 78: 205, 90: 249, 98: 88, 102: 330, 106: 183, 110: 101, 118: 218, 122: 109, 126: 106}}
INFO 05-06 11:02:13.772647.772647 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.887ms | allocate_experts_across_cpu_gpu: 0.441ms
INFO 05-06 11:02:13.772552.772552 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.127357482910156e-05 seconds
INFO 05-06 11:02:13.774039.774039 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0012116432189941406 seconds
INFO 05-06 11:02:13.795389.795389 lmp.py:1387] [layer_moe_fused] to time: 0.00014519691467285156 seconds
INFO 05-06 11:02:13.795555.795555 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:13.796930.796930 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012359619140625 seconds
INFO 05-06 11:02:13.797761.797761 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005998611450195312 seconds
INFO 05-06 11:02:13.797657.797657 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019922256469726562 seconds
INFO 05-06 11:02:13.807094.807094 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009650707244873047 seconds
INFO 05-06 11:02:13.810676.810676 mlpmodule.py:2799] [fused_experts] gmm total=2.687ms E=32 S=3299 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.811399.811399 mlpmodule.py:2799] [fused_experts] gmm total=4.053ms E=32 S=3670 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.813049.813049 mlpmodule.py:2799] [fused_experts] gmm total=6.193ms E=32 S=4634 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.819510.819510 mlpmodule.py:2799] [fused_experts] gmm total=12.557ms E=32 S=4781 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.820004.820004 lmp.py:1500] [layer_moe_fused] experts compute time: 0.013041257858276367 seconds
INFO 05-06 11:02:13.820174.820174 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.340576171875e-05 seconds
DEBUG 05-06 11:02:13.820030.820030 cuda_h.py:27] end *layer_moe_fused cost 49.155 ms
DEBUG 05-06 11:02:13.821002.821002 cuda_h.py:27] end prefill_layer cost 53.183 ms
DEBUG 05-06 11:02:13.821574.821574 lmp.py:841] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 11:02:13.821105.821105 lmp.py:824] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 11:02:13.823813.823813 cuda_h.py:27] end *sagl cost 2.359 ms
experts_cpu_alloc {'expert_ids': [99, 103, 35, 87, 115, 23, 55, 91, 27, 127, 67, 111, 31, 12, 36, 80, 20, 32, 72, 16, 60, 56, 24, 48, 8, 116, 40, 100, 81, 113, 45, 105, 21, 29, 125, 37, 65, 89, 57, 41, 1, 117, 33, 13, 126, 38, 46, 106, 18, 94, 82, 98, 42, 86, 30, 6, 26, 58, 110, 2], 'token_total': 1818, 'token_per_expert': {99: 1, 103: 2, 35: 7, 87: 8, 115: 10, 23: 17, 55: 17, 91: 18, 27: 24, 127: 47, 67: 52, 111: 62, 31: 64, 12: 4, 36: 6, 80: 7, 20: 8, 32: 15, 72: 25, 16: 33, 60: 38, 56: 39, 24: 42, 48: 44, 8: 51, 116: 51, 40: 57, 100: 73, 81: 1, 113: 1, 45: 2, 105: 2, 21: 4, 29: 6, 125: 6, 37: 8, 65: 15, 89: 21, 57: 25, 41: 33, 1: 37, 117: 43, 33: 52, 13: 70, 126: 1, 38: 2, 46: 2, 106: 4, 18: 9, 94: 11, 82: 19, 98: 19, 42: 29, 86: 33, 30: 41, 6: 81, 26: 92, 58: 106, 110: 108, 2: 113}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 39, 43, 51, 59, 63, 71, 75, 83, 95, 107, 119, 123], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 29, 'token_total': 2452, 'token_per_expert': {3: 109, 11: 131, 15: 187, 19: 109, 39: 88, 43: 84, 51: 162, 59: 98, 63: 84, 71: 331, 75: 392, 83: 180, 95: 201, 107: 109, 119: 103, 123: 84}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 28, 44, 52, 64, 68, 76, 84, 88, 92, 96, 104, 108, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3436, 'token_per_expert': {0: 159, 4: 292, 28: 248, 44: 97, 52: 282, 64: 101, 68: 209, 76: 243, 84: 242, 88: 299, 92: 282, 96: 317, 104: 240, 108: 160, 120: 265}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 17, 25, 53, 61, 69, 73, 77, 85, 93, 97, 101, 109, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3545, 'token_per_expert': {5: 289, 9: 316, 17: 179, 25: 179, 53: 253, 61: 72, 69: 156, 73: 252, 77: 83, 85: 543, 93: 346, 97: 269, 101: 87, 109: 77, 121: 444}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 14, 22, 34, 50, 54, 62, 66, 70, 74, 78, 102, 114, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 5133, 'token_per_expert': {10: 166, 14: 311, 22: 541, 34: 265, 50: 684, 54: 196, 62: 431, 66: 343, 70: 147, 74: 163, 78: 692, 102: 425, 114: 126, 118: 251, 122: 392}}
INFO 05-06 11:02:13.826949.826949 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.611ms | allocate_experts_across_cpu_gpu: 0.450ms
INFO 05-06 11:02:13.826000.826000 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.413459777832031e-05 seconds
INFO 05-06 11:02:13.827972.827972 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0012526512145996094 seconds
INFO 05-06 11:02:13.848868.848868 lmp.py:1387] [layer_moe_fused] to time: 0.00014972686767578125 seconds
INFO 05-06 11:02:13.849637.849637 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:13.850502.850502 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012385845184326172 seconds
INFO 05-06 11:02:13.851180.851180 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005953311920166016 seconds
INFO 05-06 11:02:13.851838.851838 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019876956939697266 seconds
INFO 05-06 11:02:13.861167.861167 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010764360427856445 seconds
INFO 05-06 11:02:13.864751.864751 mlpmodule.py:2799] [fused_experts] gmm total=2.507ms E=32 S=3929 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.864186.864186 mlpmodule.py:2799] [fused_experts] gmm total=2.601ms E=32 S=3871 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.865732.865732 mlpmodule.py:2799] [fused_experts] gmm total=2.805ms E=32 S=2781 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.865253.865253 mlpmodule.py:2799] [fused_experts] gmm total=3.164ms E=32 S=5803 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.866438.866438 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004855632781982422 seconds
INFO 05-06 11:02:13.867409.867409 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.245208740234375e-05 seconds
DEBUG 05-06 11:02:13.867079.867079 cuda_h.py:27] end *layer_moe_fused cost 42.309 ms
DEBUG 05-06 11:02:13.873314.873314 cuda_h.py:27] end prefill_layer cost 52.323 ms
DEBUG 05-06 11:02:13.873562.873562 lmp.py:841] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 11:02:13.873139.873139 lmp.py:824] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 11:02:13.875920.875920 cuda_h.py:27] end *sagl cost 1.997 ms
experts_cpu_alloc {'expert_ids': [35, 95, 127, 79, 31, 7, 103, 15, 107, 75, 91, 123, 87, 71, 100, 48, 16, 12, 72, 120, 80, 44, 56, 88, 84, 40, 36, 64, 13, 9, 65, 121, 41, 69, 21, 37, 73, 109, 77, 57, 25, 101, 45, 81, 42, 70, 102, 14, 2, 110, 50, 58, 126, 66, 114, 46, 122, 38, 6, 18], 'token_total': 1828, 'token_per_expert': {35: 1, 95: 4, 127: 8, 79: 9, 31: 12, 7: 30, 103: 34, 15: 55, 107: 64, 75: 65, 91: 72, 123: 109, 87: 110, 71: 121, 100: 1, 48: 2, 16: 3, 12: 11, 72: 12, 120: 14, 80: 20, 44: 28, 56: 32, 88: 40, 84: 49, 40: 52, 36: 56, 64: 61, 13: 1, 9: 2, 65: 4, 121: 5, 41: 14, 69: 26, 21: 28, 37: 32, 73: 37, 109: 46, 77: 50, 57: 51, 25: 52, 101: 53, 45: 57, 81: 65, 42: 1, 70: 1, 102: 1, 14: 2, 2: 4, 110: 7, 50: 11, 58: 13, 126: 14, 66: 16, 114: 16, 46: 24, 122: 24, 38: 31, 6: 32, 18: 33}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 19, 23, 27, 39, 43, 47, 51, 55, 59, 63, 67, 83, 111, 115, 119], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 6294, 'token_per_expert': {3: 156, 19: 165, 23: 467, 27: 271, 39: 208, 43: 570, 47: 183, 51: 295, 55: 240, 59: 642, 63: 1031, 67: 274, 83: 432, 111: 455, 115: 402, 119: 503}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 20, 24, 28, 32, 52, 60, 76, 92, 96, 104, 108, 116, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 2673, 'token_per_expert': {4: 139, 8: 706, 20: 137, 24: 314, 28: 124, 32: 151, 52: 82, 60: 127, 76: 208, 92: 141, 96: 104, 104: 122, 108: 86, 116: 101, 124: 131}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 29, 49, 53, 61, 85, 89, 93, 97, 105, 113, 117, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 2641, 'token_per_expert': {1: 295, 5: 185, 17: 103, 29: 208, 49: 79, 53: 246, 61: 105, 85: 123, 89: 447, 93: 195, 97: 114, 105: 111, 113: 250, 117: 74, 125: 106}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 30, 34, 54, 62, 74, 78, 82, 86, 90, 94, 98, 106, 118], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 2948, 'token_per_expert': {22: 411, 26: 351, 30: 82, 34: 36, 54: 339, 62: 94, 74: 409, 78: 95, 82: 252, 86: 111, 90: 43, 94: 84, 98: 80, 106: 507, 118: 54}}
INFO 05-06 11:02:13.878836.878836 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.875ms | allocate_experts_across_cpu_gpu: 0.448ms
INFO 05-06 11:02:13.878324.878324 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.413459777832031e-05 seconds
INFO 05-06 11:02:13.879897.879897 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0012927055358886719 seconds
INFO 05-06 11:02:13.901259.901259 lmp.py:1387] [layer_moe_fused] to time: 0.00014495849609375 seconds
INFO 05-06 11:02:13.901882.901882 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:13.902083.902083 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012178421020507812 seconds
INFO 05-06 11:02:13.903026.903026 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005795955657958984 seconds
INFO 05-06 11:02:13.903591.903591 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019521713256835938 seconds
INFO 05-06 11:02:13.911533.911533 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008301496505737305 seconds
INFO 05-06 11:02:13.914064.914064 mlpmodule.py:2799] [fused_experts] gmm total=2.501ms E=32 S=3054 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.914976.914976 mlpmodule.py:2799] [fused_experts] gmm total=2.592ms E=32 S=3164 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.914979.914979 mlpmodule.py:2799] [fused_experts] gmm total=2.613ms E=32 S=3178 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.914194.914194 mlpmodule.py:2799] [fused_experts] gmm total=2.875ms E=32 S=6988 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:13.916171.916171 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004525661468505859 seconds
INFO 05-06 11:02:13.916288.916288 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.221366882324219e-05 seconds
DEBUG 05-06 11:02:13.916683.916683 cuda_h.py:27] end *layer_moe_fused cost 39.869 ms
DEBUG 05-06 11:02:13.921048.921048 cuda_h.py:27] end prefill_layer cost 47.712 ms
DEBUG 05-06 11:02:13.921580.921580 lmp.py:841] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 11:02:13.921158.921158 lmp.py:824] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 11:02:13.961974.961974 cuda_h.py:27] end *sagl cost 40.166 ms
experts_cpu_alloc {'expert_ids': [103, 95, 3, 11, 15, 51, 115, 27, 19, 7, 107, 31, 83, 40, 124, 8, 48, 56, 92, 32, 68, 84, 44, 0, 52, 100, 80, 76, 69, 109, 121, 1, 21, 45, 41, 53, 97, 17, 81, 77, 37, 66, 90, 82, 78, 110, 38, 122, 30, 62, 50, 58, 34, 26, 54, 102, 86, 10, 98], 'token_total': 1378, 'token_per_expert': {103: 1, 95: 2, 3: 3, 11: 3, 15: 10, 51: 12, 115: 17, 27: 20, 19: 25, 7: 26, 107: 33, 31: 54, 83: 56, 40: 1, 124: 2, 8: 3, 48: 7, 56: 10, 92: 10, 32: 17, 68: 47, 84: 48, 44: 69, 0: 75, 52: 80, 100: 84, 80: 112, 76: 115, 69: 1, 109: 1, 121: 1, 1: 3, 21: 4, 45: 4, 41: 8, 53: 10, 97: 11, 17: 12, 81: 16, 77: 26, 37: 34, 66: 1, 90: 1, 82: 3, 78: 5, 110: 6, 38: 10, 122: 10, 30: 13, 62: 13, 50: 18, 58: 19, 34: 25, 26: 26, 54: 26, 102: 26, 86: 32, 10: 33, 98: 38}}
experts_gpu_alloc_device_0 {'expert_ids': [23, 39, 43, 55, 63, 67, 71, 75, 79, 87, 99, 111, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 3891, 'token_per_expert': {23: 159, 39: 576, 43: 112, 55: 80, 63: 120, 67: 98, 71: 1127, 75: 113, 79: 63, 87: 191, 99: 310, 111: 271, 119: 79, 123: 198, 127: 394}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 16, 20, 24, 28, 36, 60, 64, 72, 88, 96, 104, 112, 116, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4333, 'token_per_expert': {4: 198, 16: 452, 20: 615, 24: 205, 28: 227, 36: 357, 60: 186, 64: 443, 72: 288, 88: 244, 96: 123, 104: 188, 112: 508, 116: 163, 120: 136}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 29, 33, 49, 57, 61, 73, 93, 101, 105, 113, 117, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 4399, 'token_per_expert': {5: 153, 9: 213, 13: 244, 29: 174, 33: 489, 49: 658, 57: 36, 61: 324, 73: 207, 93: 149, 101: 1267, 105: 34, 113: 73, 117: 302, 125: 76}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 42, 46, 70, 74, 94, 106, 114, 118, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 32, 'token_total': 2383, 'token_per_expert': {2: 417, 6: 45, 14: 53, 18: 83, 22: 391, 42: 254, 46: 156, 70: 197, 74: 160, 94: 232, 106: 64, 114: 52, 118: 129, 126: 150}}
INFO 05-06 11:02:13.964405.964405 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 1.124ms | allocate_experts_across_cpu_gpu: 0.274ms
INFO 05-06 11:02:13.964700.964700 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.0531158447265625e-05 seconds
INFO 05-06 11:02:13.965135.965135 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010919570922851562 seconds
INFO 05-06 11:02:14.025785.025785 lmp.py:1387] [layer_moe_fused] to time: 0.0001628398895263672 seconds
INFO 05-06 11:02:14.026025.026025 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.027414.027414 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012819766998291016 seconds
INFO 05-06 11:02:14.028623.028623 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006308555603027344 seconds
INFO 05-06 11:02:14.028334.028334 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002076864242553711 seconds
INFO 05-06 11:02:14.038765.038765 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010029792785644531 seconds
INFO 05-06 11:02:14.041515.041515 mlpmodule.py:2799] [fused_experts] gmm total=2.422ms E=32 S=4530 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.041407.041407 mlpmodule.py:2799] [fused_experts] gmm total=2.502ms E=32 S=2688 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.041055.041055 mlpmodule.py:2799] [fused_experts] gmm total=2.869ms E=32 S=4153 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.041758.041758 mlpmodule.py:2799] [fused_experts] gmm total=3.224ms E=32 S=5013 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.042479.042479 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004434108734130859 seconds
INFO 05-06 11:02:14.042040.042040 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.650520324707031e-05 seconds
DEBUG 05-06 11:02:14.043746.043746 cuda_h.py:27] end *layer_moe_fused cost 80.408 ms
DEBUG 05-06 11:02:14.049772.049772 cuda_h.py:27] end prefill_layer cost 127.691 ms
DEBUG 05-06 11:02:14.049734.049734 lmp.py:841] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 11:02:14.049073.049073 lmp.py:824] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 11:02:14.051822.051822 cuda_h.py:27] end *sagl cost 1.981 ms
experts_cpu_alloc {'expert_ids': [55, 7, 31, 83, 59, 67, 15, 47, 111, 11, 91, 19, 3, 43, 84, 88, 8, 100, 92, 112, 4, 52, 40, 72, 16, 124, 120, 60, 20, 49, 33, 97, 21, 81, 17, 29, 101, 109, 37, 41, 57, 89, 125, 85, 54, 66, 38, 114, 22, 18, 74, 82, 30, 14, 42, 110, 126, 70, 58, 2], 'token_total': 1553, 'token_per_expert': {55: 2, 7: 7, 31: 8, 83: 10, 59: 11, 67: 13, 15: 14, 47: 14, 111: 24, 11: 30, 91: 30, 19: 34, 3: 37, 43: 42, 84: 1, 88: 1, 8: 3, 100: 4, 92: 6, 112: 6, 4: 7, 52: 10, 40: 12, 72: 13, 16: 14, 124: 14, 120: 15, 60: 28, 20: 32, 49: 4, 33: 5, 97: 5, 21: 6, 81: 9, 17: 11, 29: 13, 101: 13, 109: 17, 37: 42, 41: 49, 57: 57, 89: 57, 125: 62, 85: 66, 54: 1, 66: 4, 38: 10, 114: 13, 22: 16, 18: 19, 74: 32, 82: 40, 30: 51, 14: 53, 42: 57, 110: 62, 126: 62, 70: 83, 58: 93, 2: 99}}
experts_gpu_alloc_device_0 {'expert_ids': [23, 27, 35, 51, 71, 75, 79, 87, 95, 99, 103, 107, 115, 119, 123, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3744, 'token_per_expert': {23: 355, 27: 93, 35: 539, 51: 159, 71: 120, 75: 186, 79: 196, 87: 360, 95: 92, 99: 824, 103: 62, 107: 136, 115: 323, 119: 152, 123: 98, 127: 49}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 24, 28, 32, 36, 44, 56, 64, 68, 76, 80, 96, 104, 108, 116], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3580, 'token_per_expert': {0: 60, 24: 175, 28: 85, 32: 148, 36: 159, 44: 96, 56: 90, 64: 503, 68: 1238, 76: 40, 80: 44, 96: 164, 104: 210, 108: 481, 116: 87}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 53, 65, 69, 73, 77, 93, 105, 113, 117, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3647, 'token_per_expert': {1: 94, 5: 72, 9: 162, 13: 281, 25: 820, 53: 420, 65: 362, 69: 101, 73: 75, 77: 81, 93: 644, 105: 69, 113: 122, 117: 191, 121: 153}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 26, 34, 46, 50, 62, 78, 86, 90, 94, 98, 102, 106, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3860, 'token_per_expert': {6: 111, 10: 106, 26: 140, 34: 341, 46: 165, 50: 138, 62: 138, 78: 211, 86: 512, 90: 433, 94: 273, 98: 264, 102: 527, 106: 394, 122: 107}}
INFO 05-06 11:02:14.053196.053196 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.897ms | allocate_experts_across_cpu_gpu: 0.456ms
INFO 05-06 11:02:14.054877.054877 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.508827209472656e-05 seconds
INFO 05-06 11:02:14.055969.055969 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011725425720214844 seconds
INFO 05-06 11:02:14.073752.073752 lmp.py:1387] [layer_moe_fused] to time: 0.00014519691467285156 seconds
INFO 05-06 11:02:14.073898.073898 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.075358.075358 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012269020080566406 seconds
INFO 05-06 11:02:14.075668.075668 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006725788116455078 seconds
INFO 05-06 11:02:14.076710.076710 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002057313919067383 seconds
INFO 05-06 11:02:14.085508.085508 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009740591049194336 seconds
INFO 05-06 11:02:14.088544.088544 mlpmodule.py:2799] [fused_experts] gmm total=2.349ms E=32 S=3746 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.088529.088529 mlpmodule.py:2799] [fused_experts] gmm total=2.547ms E=32 S=4020 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.088580.088580 mlpmodule.py:2799] [fused_experts] gmm total=2.519ms E=32 S=4555 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.088387.088387 mlpmodule.py:2799] [fused_experts] gmm total=2.744ms E=32 S=4063 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.089555.089555 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004093647003173828 seconds
INFO 05-06 11:02:14.090049.090049 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.1975250244140625e-05 seconds
DEBUG 05-06 11:02:14.090838.090838 cuda_h.py:27] end *layer_moe_fused cost 37.924 ms
DEBUG 05-06 11:02:14.096095.096095 cuda_h.py:27] end prefill_layer cost 47.514 ms
DEBUG 05-06 11:02:14.096012.096012 lmp.py:841] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 11:02:14.096066.096066 lmp.py:824] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 11:02:14.099898.099898 cuda_h.py:27] end *sagl cost 2.307 ms
experts_cpu_alloc {'expert_ids': [27, 75, 3, 119, 39, 11, 31, 107, 67, 63, 55, 127, 35, 15, 111, 23, 76, 124, 24, 92, 100, 36, 32, 88, 0, 16, 116, 80, 64, 8, 68, 112, 1, 37, 73, 109, 49, 17, 77, 21, 41, 25, 45, 101, 117, 9, 46, 58, 102, 38, 50, 94, 30, 62, 78, 26, 126, 82, 66, 54, 98], 'token_total': 1998, 'token_per_expert': {27: 2, 75: 3, 3: 4, 119: 5, 39: 7, 11: 8, 31: 14, 107: 15, 67: 21, 63: 26, 55: 28, 127: 31, 35: 33, 15: 50, 111: 53, 23: 54, 76: 1, 124: 3, 24: 5, 92: 7, 100: 13, 36: 16, 32: 29, 88: 32, 0: 33, 16: 40, 116: 46, 80: 51, 64: 72, 8: 87, 68: 88, 112: 122, 1: 5, 37: 16, 73: 20, 109: 21, 49: 23, 17: 33, 77: 34, 21: 47, 41: 48, 25: 54, 45: 58, 101: 69, 117: 84, 9: 94, 46: 1, 58: 3, 102: 7, 38: 12, 50: 12, 94: 13, 30: 16, 62: 16, 78: 24, 26: 35, 126: 43, 82: 47, 66: 51, 54: 52, 98: 61}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 19, 43, 47, 51, 59, 71, 79, 83, 87, 91, 95, 99, 103, 115, 123], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 2650, 'token_per_expert': {7: 156, 19: 85, 43: 137, 47: 113, 51: 97, 59: 139, 71: 112, 79: 214, 83: 113, 87: 93, 91: 859, 95: 78, 99: 97, 103: 168, 115: 115, 123: 74}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 12, 20, 28, 44, 48, 52, 56, 60, 72, 84, 96, 104, 108, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4068, 'token_per_expert': {4: 372, 12: 434, 20: 290, 28: 223, 44: 252, 48: 193, 52: 353, 56: 183, 60: 167, 72: 127, 84: 320, 96: 169, 104: 152, 108: 543, 120: 290}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 13, 29, 33, 53, 57, 61, 65, 69, 85, 97, 105, 113, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4484, 'token_per_expert': {5: 169, 13: 127, 29: 491, 33: 104, 53: 233, 57: 185, 61: 138, 65: 260, 69: 350, 85: 297, 97: 983, 105: 125, 113: 230, 121: 610, 125: 182}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 14, 18, 22, 34, 42, 70, 86, 90, 106, 110, 114, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3184, 'token_per_expert': {6: 75, 10: 267, 14: 251, 18: 141, 22: 80, 34: 395, 42: 226, 70: 353, 86: 231, 90: 328, 106: 185, 110: 263, 114: 252, 118: 69, 122: 68}}
INFO 05-06 11:02:14.101009.101009 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.620ms | allocate_experts_across_cpu_gpu: 0.453ms
INFO 05-06 11:02:14.102405.102405 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.437301635742188e-05 seconds
INFO 05-06 11:02:14.103996.103996 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0012650489807128906 seconds
INFO 05-06 11:02:14.126429.126429 lmp.py:1387] [layer_moe_fused] to time: 0.0001537799835205078 seconds
INFO 05-06 11:02:14.126874.126874 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.127586.127586 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012412071228027344 seconds
INFO 05-06 11:02:14.128828.128828 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006210803985595703 seconds
INFO 05-06 11:02:14.128532.128532 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0020194053649902344 seconds
INFO 05-06 11:02:14.139836.139836 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010606050491333008 seconds
INFO 05-06 11:02:14.141918.141918 mlpmodule.py:2799] [fused_experts] gmm total=2.404ms E=32 S=3004 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.141599.141599 mlpmodule.py:2799] [fused_experts] gmm total=2.381ms E=32 S=3577 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.142763.142763 mlpmodule.py:2799] [fused_experts] gmm total=2.646ms E=32 S=4713 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.142452.142452 mlpmodule.py:2799] [fused_experts] gmm total=2.778ms E=32 S=5090 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.143357.143357 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0041921138763427734 seconds
INFO 05-06 11:02:14.143725.143725 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.459785461425781e-05 seconds
DEBUG 05-06 11:02:14.143732.143732 cuda_h.py:27] end *layer_moe_fused cost 43.041 ms
DEBUG 05-06 11:02:14.150787.150787 cuda_h.py:27] end prefill_layer cost 53.578 ms
DEBUG 05-06 11:02:14.150519.150519 lmp.py:841] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 11:02:14.150335.150335 lmp.py:824] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 11:02:14.152257.152257 cuda_h.py:27] end *sagl cost 2.026 ms
experts_cpu_alloc {'expert_ids': [83, 59, 67, 107, 23, 35, 79, 7, 39, 119, 91, 43, 99, 11, 112, 60, 40, 0, 100, 24, 72, 104, 92, 48, 96, 8, 64, 68, 116, 84, 109, 101, 13, 25, 37, 33, 9, 117, 49, 1, 57, 17, 89, 113, 53, 77, 30, 18, 90, 118, 34, 26, 62, 82, 74, 86, 106, 10, 14, 66, 42], 'token_total': 1853, 'token_per_expert': {83: 2, 59: 3, 67: 5, 107: 6, 23: 9, 35: 9, 79: 10, 7: 11, 39: 14, 119: 28, 91: 32, 43: 35, 99: 39, 11: 65, 112: 1, 60: 2, 40: 6, 0: 7, 100: 7, 24: 11, 72: 12, 104: 18, 92: 22, 48: 23, 96: 23, 8: 29, 64: 33, 68: 33, 116: 39, 84: 41, 109: 1, 101: 21, 13: 24, 25: 26, 37: 27, 33: 28, 9: 29, 117: 32, 49: 46, 1: 55, 57: 66, 17: 72, 89: 73, 113: 73, 53: 77, 77: 81, 30: 1, 18: 8, 90: 10, 118: 21, 34: 24, 26: 25, 62: 28, 82: 30, 74: 36, 86: 43, 106: 45, 10: 65, 14: 66, 66: 67, 42: 78}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 19, 27, 31, 47, 51, 55, 63, 71, 75, 87, 103, 111, 123, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3964, 'token_per_expert': {3: 125, 15: 228, 19: 290, 27: 175, 31: 118, 47: 66, 51: 644, 55: 169, 63: 156, 71: 211, 75: 284, 87: 416, 103: 777, 111: 77, 123: 155, 127: 73}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 12, 16, 20, 28, 32, 36, 44, 52, 56, 76, 80, 108, 120, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 2631, 'token_per_expert': {4: 49, 12: 160, 16: 103, 20: 85, 28: 413, 32: 271, 36: 178, 44: 111, 52: 117, 56: 242, 76: 138, 80: 247, 108: 65, 120: 360, 124: 92}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 21, 29, 41, 45, 61, 65, 69, 73, 81, 85, 93, 105, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3173, 'token_per_expert': {5: 165, 21: 137, 29: 114, 41: 98, 45: 87, 61: 122, 65: 276, 69: 199, 73: 534, 81: 170, 85: 124, 93: 136, 105: 368, 121: 348, 125: 295}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 38, 46, 50, 54, 58, 70, 98, 102, 110, 114, 122, 126], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4763, 'token_per_expert': {2: 214, 6: 148, 22: 88, 38: 241, 46: 255, 50: 338, 54: 853, 58: 977, 70: 404, 98: 108, 102: 226, 110: 342, 114: 293, 122: 172, 126: 104}}
INFO 05-06 11:02:14.155280.155280 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.891ms | allocate_experts_across_cpu_gpu: 0.454ms
INFO 05-06 11:02:14.155331.155331 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.222724914550781e-05 seconds
INFO 05-06 11:02:14.156002.156002 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011699199676513672 seconds
INFO 05-06 11:02:14.177078.177078 lmp.py:1387] [layer_moe_fused] to time: 0.00014638900756835938 seconds
INFO 05-06 11:02:14.178377.178377 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.179055.179055 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012178421020507812 seconds
INFO 05-06 11:02:14.180464.180464 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006785392761230469 seconds
INFO 05-06 11:02:14.180699.180699 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0020525455474853516 seconds
INFO 05-06 11:02:14.190925.190925 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00967097282409668 seconds
INFO 05-06 11:02:14.192098.192098 mlpmodule.py:2799] [fused_experts] gmm total=2.372ms E=32 S=4232 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.192606.192606 mlpmodule.py:2799] [fused_experts] gmm total=2.387ms E=32 S=3904 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.192719.192719 mlpmodule.py:2799] [fused_experts] gmm total=2.666ms E=32 S=2938 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.193437.193437 mlpmodule.py:2799] [fused_experts] gmm total=2.634ms E=32 S=5310 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.194588.194588 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004086017608642578 seconds
INFO 05-06 11:02:14.194287.194287 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.555152893066406e-05 seconds
DEBUG 05-06 11:02:14.194358.194358 cuda_h.py:27] end *layer_moe_fused cost 40.661 ms
DEBUG 05-06 11:02:14.200746.200746 cuda_h.py:27] end prefill_layer cost 50.048 ms
DEBUG 05-06 11:02:14.200801.200801 lmp.py:841] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 11:02:14.200425.200425 lmp.py:824] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 11:02:14.203260.203260 cuda_h.py:27] end *sagl cost 1.995 ms
experts_cpu_alloc {'expert_ids': [91, 63, 31, 47, 11, 59, 119, 55, 123, 79, 67, 115, 15, 3, 108, 84, 96, 0, 112, 64, 100, 28, 120, 104, 8, 20, 52, 44, 116, 80, 49, 25, 53, 33, 29, 121, 41, 77, 105, 113, 117, 73, 97, 5, 2, 118, 14, 94, 126, 66, 110, 50, 6, 26, 90, 34, 58, 18, 114, 10], 'token_total': 1406, 'token_per_expert': {91: 1, 63: 3, 31: 5, 47: 5, 11: 6, 59: 6, 119: 11, 55: 14, 123: 17, 79: 28, 67: 33, 115: 37, 15: 91, 3: 92, 108: 2, 84: 4, 96: 12, 0: 13, 112: 13, 64: 14, 100: 14, 28: 19, 120: 24, 104: 27, 8: 28, 20: 29, 52: 33, 44: 37, 116: 42, 80: 59, 49: 1, 25: 2, 53: 3, 33: 14, 29: 15, 121: 15, 41: 21, 77: 27, 105: 30, 113: 49, 117: 51, 73: 54, 97: 59, 5: 67, 2: 3, 118: 3, 14: 4, 94: 6, 126: 6, 66: 13, 110: 13, 50: 14, 6: 19, 26: 20, 90: 21, 34: 23, 58: 26, 18: 30, 114: 37, 10: 41}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 19, 23, 27, 39, 43, 51, 71, 75, 83, 95, 99, 103, 111, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4027, 'token_per_expert': {7: 109, 19: 126, 23: 212, 27: 137, 39: 158, 43: 437, 51: 134, 71: 175, 75: 411, 83: 142, 95: 1098, 99: 115, 103: 473, 111: 130, 127: 170}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 12, 16, 24, 32, 36, 40, 48, 56, 68, 72, 76, 88, 92, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3549, 'token_per_expert': {4: 187, 12: 716, 16: 428, 24: 140, 32: 175, 36: 199, 40: 241, 48: 257, 56: 315, 68: 110, 72: 152, 76: 203, 88: 110, 92: 224, 124: 92}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 17, 21, 37, 45, 57, 61, 69, 81, 89, 93, 101, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3337, 'token_per_expert': {1: 205, 9: 110, 13: 162, 17: 90, 21: 168, 37: 111, 45: 109, 57: 187, 61: 169, 69: 313, 81: 324, 89: 176, 93: 461, 101: 596, 125: 156}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 30, 38, 42, 46, 54, 62, 70, 74, 82, 86, 98, 102, 106, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4065, 'token_per_expert': {22: 182, 30: 163, 38: 131, 42: 68, 46: 843, 54: 143, 62: 89, 70: 785, 74: 362, 82: 45, 86: 107, 98: 56, 102: 184, 106: 823, 122: 84}}
INFO 05-06 11:02:14.205557.205557 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.569ms | allocate_experts_across_cpu_gpu: 0.445ms
INFO 05-06 11:02:14.205801.205801 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.318092346191406e-05 seconds
INFO 05-06 11:02:14.206878.206878 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0012919902801513672 seconds
INFO 05-06 11:02:14.223550.223550 lmp.py:1387] [layer_moe_fused] to time: 0.00014257431030273438 seconds
INFO 05-06 11:02:14.224405.224405 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.225679.225679 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012009143829345703 seconds
INFO 05-06 11:02:14.226239.226239 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006072521209716797 seconds
INFO 05-06 11:02:14.226565.226565 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019681453704833984 seconds
INFO 05-06 11:02:14.235404.235404 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009559154510498047 seconds
INFO 05-06 11:02:14.238297.238297 mlpmodule.py:2799] [fused_experts] gmm total=2.190ms E=32 S=3745 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.238885.238885 mlpmodule.py:2799] [fused_experts] gmm total=2.293ms E=32 S=4344 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.238934.238934 mlpmodule.py:2799] [fused_experts] gmm total=2.535ms E=32 S=4376 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.238380.238380 mlpmodule.py:2799] [fused_experts] gmm total=2.556ms E=32 S=3919 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.239188.239188 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003965616226196289 seconds
INFO 05-06 11:02:14.239219.239219 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.435943603515625e-05 seconds
DEBUG 05-06 11:02:14.240841.240841 cuda_h.py:27] end *layer_moe_fused cost 36.197 ms
DEBUG 05-06 11:02:14.246129.246129 cuda_h.py:27] end prefill_layer cost 45.524 ms
DEBUG 05-06 11:02:14.246323.246323 lmp.py:841] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 11:02:14.246185.246185 lmp.py:824] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 11:02:14.248763.248763 cuda_h.py:27] end *sagl cost 1.995 ms
experts_cpu_alloc {'expert_ids': [23, 95, 123, 35, 55, 51, 59, 87, 107, 91, 15, 3, 103, 27, 119, 11, 24, 36, 52, 48, 40, 32, 104, 12, 120, 124, 64, 112, 28, 56, 45, 17, 65, 77, 101, 53, 109, 25, 33, 73, 61, 29, 93, 97, 117, 89, 110, 118, 122, 22, 114, 30, 38, 66, 102, 70, 26, 2, 50, 6, 34, 78, 98], 'token_total': 1338, 'token_per_expert': {23: 1, 95: 1, 123: 1, 35: 2, 55: 3, 51: 8, 59: 10, 87: 13, 107: 15, 91: 16, 15: 20, 3: 22, 103: 23, 27: 30, 119: 36, 11: 50, 24: 1, 36: 2, 52: 4, 48: 5, 40: 6, 32: 10, 104: 10, 12: 18, 120: 27, 124: 29, 64: 32, 112: 43, 28: 47, 56: 50, 45: 1, 17: 4, 65: 4, 77: 8, 101: 12, 53: 14, 109: 15, 25: 17, 33: 27, 73: 29, 61: 35, 29: 42, 93: 46, 97: 54, 117: 60, 89: 64, 110: 1, 118: 1, 122: 1, 22: 3, 114: 5, 30: 6, 38: 6, 66: 10, 102: 17, 70: 20, 26: 24, 2: 26, 50: 31, 6: 44, 34: 50, 78: 56, 98: 70}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 19, 31, 39, 43, 47, 63, 67, 71, 75, 79, 83, 99, 111, 115, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 2350, 'token_per_expert': {7: 162, 19: 64, 31: 139, 39: 200, 43: 114, 47: 192, 63: 119, 67: 56, 71: 175, 75: 196, 79: 76, 83: 106, 99: 137, 111: 64, 115: 337, 127: 213}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 44, 60, 68, 72, 76, 80, 84, 88, 92, 100, 108], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 4778, 'token_per_expert': {0: 376, 4: 81, 8: 573, 16: 227, 20: 143, 44: 88, 60: 520, 68: 166, 72: 149, 76: 784, 80: 661, 84: 107, 88: 305, 92: 189, 100: 148, 108: 261}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 37, 41, 49, 57, 69, 81, 85, 105, 113, 121, 125], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 3905, 'token_per_expert': {1: 817, 5: 72, 9: 71, 13: 176, 21: 283, 37: 148, 41: 224, 49: 254, 57: 240, 69: 77, 81: 661, 85: 151, 105: 90, 113: 202, 121: 78, 125: 361}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 14, 18, 42, 46, 54, 58, 62, 74, 82, 86, 90, 94, 106, 126], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 4013, 'token_per_expert': {10: 217, 14: 419, 18: 175, 42: 259, 46: 211, 54: 127, 58: 158, 62: 307, 74: 512, 82: 237, 86: 553, 90: 93, 94: 127, 106: 370, 126: 248}}
INFO 05-06 11:02:14.251659.251659 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.881ms | allocate_experts_across_cpu_gpu: 0.460ms
INFO 05-06 11:02:14.251486.251486 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.842613220214844e-05 seconds
INFO 05-06 11:02:14.252982.252982 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.001180410385131836 seconds
INFO 05-06 11:02:14.269266.269266 lmp.py:1387] [layer_moe_fused] to time: 0.00013518333435058594 seconds
INFO 05-06 11:02:14.269067.269067 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.270084.270084 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012218952178955078 seconds
INFO 05-06 11:02:14.271225.271225 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005855560302734375 seconds
INFO 05-06 11:02:14.271598.271598 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001960277557373047 seconds
INFO 05-06 11:02:14.282493.282493 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011040449142456055 seconds
INFO 05-06 11:02:14.285143.285143 mlpmodule.py:2799] [fused_experts] gmm total=2.381ms E=32 S=2601 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.285764.285764 mlpmodule.py:2799] [fused_experts] gmm total=2.408ms E=32 S=4337 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.285574.285574 mlpmodule.py:2799] [fused_experts] gmm total=2.534ms E=32 S=5062 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.285906.285906 mlpmodule.py:2799] [fused_experts] gmm total=2.639ms E=32 S=4384 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.286869.286869 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004226207733154297 seconds
INFO 05-06 11:02:14.286045.286045 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.53131103515625e-05 seconds
DEBUG 05-06 11:02:14.287639.287639 cuda_h.py:27] end *layer_moe_fused cost 37.530 ms
DEBUG 05-06 11:02:14.294290.294290 cuda_h.py:27] end prefill_layer cost 47.610 ms
DEBUG 05-06 11:02:14.294299.294299 lmp.py:841] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 11:02:14.294830.294830 lmp.py:824] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 11:02:14.297047.297047 cuda_h.py:27] end *sagl cost 3.149 ms
experts_cpu_alloc {'expert_ids': [15, 127, 35, 63, 115, 47, 3, 39, 11, 123, 59, 51, 12, 104, 84, 96, 72, 88, 4, 0, 52, 80, 64, 8, 44, 40, 120, 48, 109, 45, 41, 65, 105, 85, 9, 21, 97, 13, 53, 125, 33, 1, 5, 78, 90, 26, 114, 106, 94, 110, 58, 34, 122, 74, 118, 22, 50, 62, 42], 'token_total': 1373, 'token_per_expert': {15: 1, 127: 6, 35: 11, 63: 14, 115: 17, 47: 21, 3: 23, 39: 29, 11: 42, 123: 51, 59: 68, 51: 69, 12: 1, 104: 1, 84: 6, 96: 7, 72: 8, 88: 12, 4: 17, 0: 19, 52: 23, 80: 35, 64: 39, 8: 40, 44: 41, 40: 51, 120: 69, 48: 80, 109: 1, 45: 2, 41: 5, 65: 5, 105: 5, 85: 11, 9: 12, 21: 13, 97: 16, 13: 18, 53: 19, 125: 24, 33: 27, 1: 46, 5: 50, 78: 1, 90: 1, 26: 3, 114: 4, 106: 10, 94: 12, 110: 12, 58: 15, 34: 19, 122: 21, 74: 23, 118: 27, 22: 32, 50: 33, 62: 51, 42: 54}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 19, 23, 27, 31, 43, 67, 71, 79, 83, 87, 91, 99, 111, 119], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 27, 'token_total': 4546, 'token_per_expert': {7: 357, 19: 260, 23: 344, 27: 77, 31: 218, 43: 100, 67: 354, 71: 72, 79: 601, 83: 524, 87: 809, 91: 99, 99: 137, 111: 306, 119: 288}}
experts_gpu_alloc_device_1 {'expert_ids': [16, 20, 24, 28, 32, 36, 56, 68, 76, 92, 100, 108, 112, 116, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4174, 'token_per_expert': {16: 730, 20: 147, 24: 223, 28: 96, 32: 352, 36: 130, 56: 621, 68: 293, 76: 247, 92: 507, 100: 291, 108: 182, 112: 139, 116: 123, 124: 93}}
experts_gpu_alloc_device_2 {'expert_ids': [17, 25, 29, 37, 49, 57, 61, 69, 77, 81, 89, 93, 113, 117, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3773, 'token_per_expert': {17: 435, 25: 133, 29: 112, 37: 189, 49: 383, 57: 192, 61: 127, 69: 140, 77: 140, 81: 560, 89: 106, 93: 486, 113: 585, 117: 92, 121: 93}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 30, 38, 46, 54, 66, 70, 82, 98, 102, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 2518, 'token_per_expert': {2: 141, 6: 431, 10: 135, 18: 69, 30: 191, 38: 148, 46: 76, 54: 70, 66: 139, 70: 103, 82: 100, 98: 90, 102: 736, 126: 89}}
INFO 05-06 11:02:14.301056.301056 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 2.417ms | allocate_experts_across_cpu_gpu: 0.438ms
INFO 05-06 11:02:14.301836.301836 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.604194641113281e-05 seconds
INFO 05-06 11:02:14.303993.303993 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011725425720214844 seconds
INFO 05-06 11:02:14.319495.319495 lmp.py:1387] [layer_moe_fused] to time: 0.00015115737915039062 seconds
INFO 05-06 11:02:14.319211.319211 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.321074.321074 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012140274047851562 seconds
INFO 05-06 11:02:14.321619.321619 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005671977996826172 seconds
INFO 05-06 11:02:14.321846.321846 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019314289093017578 seconds
INFO 05-06 11:02:14.331621.331621 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009232521057128906 seconds
INFO 05-06 11:02:14.333987.333987 mlpmodule.py:2799] [fused_experts] gmm total=2.207ms E=32 S=2836 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.334184.334184 mlpmodule.py:2799] [fused_experts] gmm total=2.527ms E=32 S=4898 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.334563.334563 mlpmodule.py:2799] [fused_experts] gmm total=2.482ms E=32 S=4027 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.334851.334851 mlpmodule.py:2799] [fused_experts] gmm total=2.609ms E=32 S=4623 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.335559.335559 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004008769989013672 seconds
INFO 05-06 11:02:14.335590.335590 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.3882598876953125e-05 seconds
DEBUG 05-06 11:02:14.335012.335012 cuda_h.py:27] end *layer_moe_fused cost 37.164 ms
DEBUG 05-06 11:02:14.341468.341468 cuda_h.py:27] end prefill_layer cost 47.513 ms
DEBUG 05-06 11:02:14.341185.341185 lmp.py:841] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 11:02:14.341286.341286 lmp.py:824] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 11:02:14.344224.344224 cuda_h.py:27] end *sagl cost 2.031 ms
experts_cpu_alloc {'expert_ids': [7, 27, 51, 87, 59, 83, 127, 67, 31, 111, 47, 63, 119, 0, 52, 96, 48, 4, 56, 8, 20, 32, 64, 120, 24, 112, 29, 109, 121, 69, 93, 81, 37, 33, 41, 105, 13, 113, 17, 65, 89, 30, 126, 10, 62, 2, 54, 18, 94, 102, 38, 70, 98, 90, 22], 'token_total': 1063, 'token_per_expert': {7: 1, 27: 2, 51: 2, 87: 3, 59: 4, 83: 4, 127: 12, 67: 18, 31: 20, 111: 28, 47: 32, 63: 35, 119: 40, 0: 1, 52: 1, 96: 1, 48: 3, 4: 6, 56: 7, 8: 11, 20: 17, 32: 17, 64: 17, 120: 20, 24: 33, 112: 37, 29: 1, 109: 1, 121: 1, 69: 2, 93: 7, 81: 10, 37: 13, 33: 14, 41: 14, 105: 17, 13: 20, 113: 27, 17: 31, 65: 36, 89: 37, 30: 1, 126: 1, 10: 3, 62: 6, 2: 7, 54: 7, 18: 9, 94: 33, 102: 33, 38: 49, 70: 51, 98: 65, 90: 92, 22: 103}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 19, 23, 35, 39, 71, 79, 91, 95, 103, 107, 115, 123], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 3331, 'token_per_expert': {3: 172, 15: 480, 19: 226, 23: 237, 35: 122, 39: 753, 71: 592, 79: 51, 91: 168, 95: 195, 103: 62, 107: 57, 115: 174, 123: 42}}
experts_gpu_alloc_device_1 {'expert_ids': [12, 36, 40, 68, 76, 80, 84, 88, 92, 100, 104, 108, 116, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 2572, 'token_per_expert': {12: 71, 36: 98, 40: 99, 68: 70, 76: 154, 80: 100, 84: 167, 88: 105, 92: 200, 100: 92, 104: 95, 108: 668, 116: 511, 124: 142}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25, 45, 49, 53, 73, 77, 85, 97, 101, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 3904, 'token_per_expert': {1: 44, 5: 212, 21: 721, 25: 192, 45: 590, 49: 376, 53: 784, 73: 194, 77: 105, 85: 100, 97: 287, 101: 99, 117: 161, 125: 39}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 34, 46, 50, 58, 74, 78, 82, 86, 106, 110, 114, 118], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 27, 'token_total': 5514, 'token_per_expert': {6: 204, 34: 112, 46: 362, 50: 635, 58: 113, 74: 457, 78: 1134, 82: 432, 86: 527, 106: 302, 110: 408, 114: 531, 118: 297}}
INFO 05-06 11:02:14.346880.346880 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.851ms | allocate_experts_across_cpu_gpu: 0.418ms
INFO 05-06 11:02:14.346176.346176 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.127357482910156e-05 seconds
INFO 05-06 11:02:14.348482.348482 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011637210845947266 seconds
INFO 05-06 11:02:14.362262.362262 lmp.py:1387] [layer_moe_fused] to time: 0.00012373924255371094 seconds
INFO 05-06 11:02:14.362911.362911 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.363257.363257 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011806488037109375 seconds
INFO 05-06 11:02:14.364869.364869 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006124973297119141 seconds
INFO 05-06 11:02:14.364243.364243 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019505023956298828 seconds
INFO 05-06 11:02:14.374968.374968 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010144233703613281 seconds
INFO 05-06 11:02:14.376185.376185 mlpmodule.py:2799] [fused_experts] gmm total=2.037ms E=32 S=2743 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.376893.376893 mlpmodule.py:2799] [fused_experts] gmm total=2.271ms E=32 S=3532 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.377289.377289 mlpmodule.py:2799] [fused_experts] gmm total=2.306ms E=32 S=4135 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.377416.377416 mlpmodule.py:2799] [fused_experts] gmm total=2.486ms E=32 S=5974 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.378122.378122 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003925800323486328 seconds
INFO 05-06 11:02:14.378755.378755 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.030632019042969e-05 seconds
DEBUG 05-06 11:02:14.378643.378643 cuda_h.py:27] end *layer_moe_fused cost 33.752 ms
DEBUG 05-06 11:02:14.385944.385944 cuda_h.py:27] end prefill_layer cost 43.172 ms
DEBUG 05-06 11:02:14.385768.385768 lmp.py:841] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 11:02:14.385153.385153 lmp.py:824] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 11:02:14.387520.387520 cuda_h.py:27] end *sagl cost 1.986 ms
experts_cpu_alloc {'expert_ids': [19, 7, 111, 23, 123, 11, 83, 95, 87, 47, 107, 67, 43, 27, 55, 76, 12, 72, 112, 36, 56, 96, 48, 8, 52, 80, 16, 0, 77, 97, 53, 89, 45, 61, 105, 57, 65, 109, 9, 93, 101, 73, 58, 18, 50, 66, 10, 74, 62, 90, 106, 94, 26, 46, 70, 2, 42, 122], 'token_total': 1453, 'token_per_expert': {19: 1, 7: 4, 111: 4, 23: 10, 123: 27, 11: 28, 83: 31, 95: 33, 87: 34, 47: 37, 107: 48, 67: 53, 43: 56, 27: 58, 55: 82, 76: 3, 12: 4, 72: 7, 112: 9, 36: 11, 56: 12, 96: 12, 48: 17, 8: 24, 52: 32, 80: 34, 16: 35, 0: 36, 77: 1, 97: 3, 53: 4, 89: 4, 45: 10, 61: 13, 105: 14, 57: 19, 65: 21, 109: 27, 9: 37, 93: 40, 101: 45, 73: 63, 58: 1, 18: 2, 50: 2, 66: 7, 10: 9, 74: 9, 62: 19, 90: 21, 106: 22, 94: 29, 26: 31, 46: 36, 70: 39, 2: 52, 42: 58, 122: 73}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 31, 39, 51, 59, 63, 71, 75, 79, 91, 99, 103, 115, 119], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4443, 'token_per_expert': {3: 144, 15: 164, 31: 868, 39: 282, 51: 156, 59: 333, 63: 221, 71: 387, 75: 83, 79: 519, 91: 809, 99: 87, 103: 126, 115: 148, 119: 116}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 28, 32, 40, 60, 64, 68, 84, 92, 100, 104, 108, 116, 120, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 2559, 'token_per_expert': {20: 257, 28: 39, 32: 435, 40: 63, 60: 148, 64: 40, 68: 52, 84: 134, 92: 40, 100: 566, 104: 42, 108: 50, 116: 112, 120: 472, 124: 109}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 17, 21, 25, 33, 37, 41, 69, 81, 113, 117, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 3759, 'token_per_expert': {1: 283, 13: 93, 17: 530, 21: 203, 25: 232, 33: 232, 37: 536, 41: 107, 69: 132, 81: 466, 113: 128, 117: 64, 121: 551, 125: 202}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 14, 22, 34, 38, 78, 82, 86, 98, 102, 110, 114, 118, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 4170, 'token_per_expert': {6: 284, 14: 398, 22: 156, 34: 124, 38: 183, 78: 512, 82: 82, 86: 141, 98: 203, 102: 242, 110: 629, 114: 876, 118: 158, 126: 182}}
INFO 05-06 11:02:14.389267.389267 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.554ms | allocate_experts_across_cpu_gpu: 0.435ms
INFO 05-06 11:02:14.389901.389901 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.461143493652344e-05 seconds
INFO 05-06 11:02:14.391761.391761 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011522769927978516 seconds
INFO 05-06 11:02:14.405267.405267 lmp.py:1387] [layer_moe_fused] to time: 0.00013637542724609375 seconds
INFO 05-06 11:02:14.405261.405261 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.407587.407587 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011696815490722656 seconds
INFO 05-06 11:02:14.407770.407770 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006527900695800781 seconds
INFO 05-06 11:02:14.407720.407720 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001978158950805664 seconds
INFO 05-06 11:02:14.417903.417903 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009181737899780273 seconds
INFO 05-06 11:02:14.419031.419031 mlpmodule.py:2799] [fused_experts] gmm total=1.925ms E=32 S=2795 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.419095.419095 mlpmodule.py:2799] [fused_experts] gmm total=2.154ms E=32 S=4580 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.419980.419980 mlpmodule.py:2799] [fused_experts] gmm total=2.328ms E=32 S=4060 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.420066.420066 mlpmodule.py:2799] [fused_experts] gmm total=2.650ms E=32 S=4949 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.420723.420723 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0036420822143554688 seconds
INFO 05-06 11:02:14.421270.421270 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.245208740234375e-05 seconds
DEBUG 05-06 11:02:14.421150.421150 cuda_h.py:27] end *layer_moe_fused cost 32.890 ms
DEBUG 05-06 11:02:14.426667.426667 cuda_h.py:27] end prefill_layer cost 41.837 ms
DEBUG 05-06 11:02:14.427437.427437 lmp.py:841] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 11:02:14.427061.427061 lmp.py:824] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 11:02:14.429481.429481 cuda_h.py:27] end *sagl cost 1.999 ms
experts_cpu_alloc {'expert_ids': [79, 87, 27, 111, 91, 51, 67, 35, 15, 3, 63, 43, 23, 19, 107, 4, 84, 56, 88, 96, 116, 68, 40, 48, 44, 64, 36, 0, 16, 28, 49, 1, 17, 29, 33, 41, 37, 85, 21, 77, 93, 9, 73, 101, 109, 82, 94, 46, 6, 14, 106, 18, 54, 22, 78, 58, 118, 126, 70, 98, 90, 102], 'token_total': 1480, 'token_per_expert': {79: 2, 87: 7, 27: 10, 111: 10, 91: 12, 51: 14, 67: 17, 35: 20, 15: 21, 3: 23, 63: 31, 43: 34, 23: 56, 19: 65, 107: 80, 4: 1, 84: 4, 56: 5, 88: 7, 96: 11, 116: 18, 68: 21, 40: 28, 48: 35, 44: 40, 64: 45, 36: 46, 0: 54, 16: 58, 28: 67, 49: 1, 1: 8, 17: 8, 29: 9, 33: 9, 41: 10, 37: 14, 85: 14, 21: 16, 77: 16, 93: 17, 9: 19, 73: 22, 101: 25, 109: 34, 82: 1, 94: 1, 46: 5, 6: 8, 14: 8, 106: 9, 18: 13, 54: 13, 22: 15, 78: 20, 58: 22, 118: 30, 126: 37, 70: 50, 98: 51, 90: 60, 102: 73}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 31, 39, 47, 59, 71, 75, 83, 95, 99, 103, 115, 119, 123, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 5347, 'token_per_expert': {7: 104, 11: 184, 31: 281, 39: 499, 47: 365, 59: 305, 71: 89, 75: 371, 83: 106, 95: 438, 99: 217, 103: 289, 115: 1068, 119: 519, 123: 342, 127: 170}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 24, 32, 52, 60, 72, 76, 80, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 2827, 'token_per_expert': {8: 202, 12: 116, 24: 154, 32: 114, 52: 133, 60: 85, 72: 97, 76: 159, 80: 201, 92: 77, 100: 376, 104: 187, 108: 105, 112: 137, 120: 94, 124: 590}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 13, 25, 45, 53, 57, 65, 81, 89, 97, 105, 113, 117, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3209, 'token_per_expert': {5: 42, 13: 130, 25: 58, 45: 69, 53: 134, 57: 115, 65: 449, 81: 78, 89: 184, 97: 506, 105: 96, 113: 225, 117: 262, 121: 736, 125: 125}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 26, 30, 34, 38, 42, 50, 62, 66, 74, 86, 110, 114, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 3521, 'token_per_expert': {2: 193, 10: 128, 26: 511, 30: 103, 34: 113, 38: 181, 42: 191, 50: 391, 62: 137, 66: 461, 74: 121, 86: 539, 110: 74, 114: 174, 122: 204}}
INFO 05-06 11:02:14.431539.431539 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.912ms | allocate_experts_across_cpu_gpu: 0.510ms
INFO 05-06 11:02:14.432173.432173 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.580352783203125e-05 seconds
INFO 05-06 11:02:14.433235.433235 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.001199483871459961 seconds
INFO 05-06 11:02:14.451464.451464 lmp.py:1387] [layer_moe_fused] to time: 0.000141143798828125 seconds
INFO 05-06 11:02:14.451941.451941 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.452102.452102 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011608600616455078 seconds
INFO 05-06 11:02:14.453428.453428 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005781650543212891 seconds
INFO 05-06 11:02:14.453325.453325 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0018961429595947266 seconds
INFO 05-06 11:02:14.462751.462751 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008939743041992188 seconds
INFO 05-06 11:02:14.464323.464323 mlpmodule.py:2799] [fused_experts] gmm total=2.120ms E=32 S=3267 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.465083.465083 mlpmodule.py:2799] [fused_experts] gmm total=2.191ms E=32 S=3937 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.465112.465112 mlpmodule.py:2799] [fused_experts] gmm total=2.616ms E=32 S=5749 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.467057.467057 mlpmodule.py:2799] [fused_experts] gmm total=4.817ms E=32 S=3431 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.468878.468878 lmp.py:1500] [layer_moe_fused] experts compute time: 0.005658149719238281 seconds
INFO 05-06 11:02:14.468975.468975 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.626678466796875e-05 seconds
DEBUG 05-06 11:02:14.468299.468299 cuda_h.py:27] end *layer_moe_fused cost 38.360 ms
DEBUG 05-06 11:02:14.472144.472144 cuda_h.py:27] end prefill_layer cost 44.997 ms
DEBUG 05-06 11:02:14.472769.472769 lmp.py:841] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 11:02:14.472870.472870 lmp.py:824] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 11:02:14.474915.474915 cuda_h.py:27] end *sagl cost 1.986 ms
experts_cpu_alloc {'expert_ids': [87, 3, 123, 35, 11, 67, 79, 111, 19, 115, 107, 43, 119, 127, 20, 60, 92, 56, 12, 32, 100, 44, 80, 96, 4, 40, 8, 48, 28, 36, 61, 89, 57, 45, 25, 117, 105, 77, 121, 13, 17, 33, 41, 1, 113, 62, 26, 106, 22, 74, 50, 6, 110, 126, 94, 54, 38, 58, 82, 34, 118], 'token_total': 1915, 'token_per_expert': {87: 10, 3: 12, 123: 12, 35: 13, 11: 16, 67: 16, 79: 18, 111: 31, 19: 33, 115: 52, 107: 54, 43: 64, 119: 83, 127: 86, 20: 1, 60: 1, 92: 1, 56: 2, 12: 5, 32: 12, 100: 12, 44: 18, 80: 19, 96: 35, 4: 42, 40: 50, 8: 53, 48: 58, 28: 91, 36: 108, 61: 5, 89: 6, 57: 16, 45: 19, 25: 23, 117: 32, 105: 35, 77: 38, 121: 50, 13: 55, 17: 64, 33: 65, 41: 68, 1: 73, 113: 79, 62: 1, 26: 3, 106: 3, 22: 4, 74: 4, 50: 5, 6: 9, 110: 13, 126: 13, 94: 17, 54: 20, 38: 28, 58: 36, 82: 40, 34: 41, 118: 42}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 23, 31, 39, 47, 51, 55, 59, 63, 71, 75, 83, 91, 95, 99, 103], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3549, 'token_per_expert': {7: 228, 23: 238, 31: 94, 39: 251, 47: 116, 51: 233, 55: 114, 59: 93, 63: 99, 71: 236, 75: 330, 83: 504, 91: 623, 95: 130, 99: 135, 103: 125}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 24, 52, 64, 68, 72, 76, 84, 88, 104, 108, 112, 116, 120, 124], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 4975, 'token_per_expert': {0: 111, 16: 198, 24: 119, 52: 260, 64: 278, 68: 623, 72: 198, 76: 943, 84: 191, 88: 243, 104: 159, 108: 411, 112: 864, 116: 154, 120: 112, 124: 111}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 21, 29, 37, 65, 69, 73, 81, 85, 93, 97, 101, 109, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3296, 'token_per_expert': {5: 100, 9: 290, 21: 138, 29: 86, 37: 233, 65: 522, 69: 132, 73: 100, 81: 196, 85: 137, 93: 135, 97: 170, 101: 255, 109: 594, 125: 208}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 18, 30, 42, 46, 66, 70, 78, 86, 90, 98, 102, 114], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 2649, 'token_per_expert': {2: 204, 10: 410, 14: 93, 18: 46, 30: 337, 42: 129, 46: 92, 66: 247, 70: 208, 78: 82, 86: 82, 90: 366, 98: 199, 102: 78, 114: 76}}
INFO 05-06 11:02:14.476404.476404 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.577ms | allocate_experts_across_cpu_gpu: 0.457ms
INFO 05-06 11:02:14.476178.476178 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.723403930664062e-05 seconds
INFO 05-06 11:02:14.477579.477579 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011801719665527344 seconds
INFO 05-06 11:02:14.499398.499398 lmp.py:1387] [layer_moe_fused] to time: 0.00014853477478027344 seconds
INFO 05-06 11:02:14.500180.500180 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.501037.501037 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012080669403076172 seconds
INFO 05-06 11:02:14.501788.501788 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005779266357421875 seconds
INFO 05-06 11:02:14.502115.502115 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019390583038330078 seconds
INFO 05-06 11:02:14.512416.512416 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010146856307983398 seconds
INFO 05-06 11:02:14.514307.514307 mlpmodule.py:2799] [fused_experts] gmm total=2.226ms E=32 S=4049 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.514842.514842 mlpmodule.py:2799] [fused_experts] gmm total=2.205ms E=32 S=2928 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.515125.515125 mlpmodule.py:2799] [fused_experts] gmm total=2.399ms E=32 S=3924 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.515996.515996 mlpmodule.py:2799] [fused_experts] gmm total=2.569ms E=32 S=5483 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.516230.516230 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0040509700775146484 seconds
INFO 05-06 11:02:14.516453.516453 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.412101745605469e-05 seconds
DEBUG 05-06 11:02:14.516325.516325 cuda_h.py:27] end *layer_moe_fused cost 41.393 ms
DEBUG 05-06 11:02:14.522906.522906 cuda_h.py:27] end prefill_layer cost 50.720 ms
DEBUG 05-06 11:02:14.523386.523386 lmp.py:841] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 11:02:14.523248.523248 lmp.py:824] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 11:02:14.525899.525899 cuda_h.py:27] end *sagl cost 2.001 ms
experts_cpu_alloc {'expert_ids': [35, 95, 115, 27, 7, 71, 59, 103, 11, 99, 43, 111, 91, 123, 28, 120, 36, 88, 112, 64, 104, 60, 56, 84, 92, 24, 40, 80, 116, 72, 25, 101, 73, 29, 49, 41, 53, 89, 9, 33, 13, 37, 45, 109, 69, 93, 46, 94, 106, 74, 50, 122, 98, 6, 34, 38, 62, 30, 10, 18, 118, 102, 22], 'token_total': 2194, 'token_per_expert': {35: 2, 95: 4, 115: 4, 27: 5, 7: 19, 71: 20, 59: 22, 103: 24, 11: 27, 99: 41, 43: 46, 111: 47, 91: 49, 123: 52, 28: 6, 120: 8, 36: 15, 88: 19, 112: 21, 64: 26, 104: 29, 60: 35, 56: 43, 84: 54, 92: 58, 24: 60, 40: 68, 80: 69, 116: 98, 72: 114, 25: 1, 101: 3, 73: 4, 29: 9, 49: 10, 41: 13, 53: 18, 89: 23, 9: 24, 33: 32, 13: 43, 37: 43, 45: 44, 109: 47, 69: 49, 93: 62, 46: 1, 94: 4, 106: 4, 74: 11, 50: 16, 122: 19, 98: 21, 6: 32, 34: 33, 38: 36, 62: 42, 30: 48, 10: 59, 18: 69, 118: 85, 102: 97, 22: 107}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 19, 23, 31, 51, 55, 63, 67, 75, 79, 83, 87, 107, 119, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3567, 'token_per_expert': {3: 160, 15: 54, 19: 148, 23: 291, 31: 218, 51: 59, 55: 140, 63: 200, 67: 643, 75: 213, 79: 53, 83: 288, 87: 736, 107: 168, 119: 64, 127: 132}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 32, 44, 48, 52, 68, 76, 96, 100, 108, 124], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 4758, 'token_per_expert': {0: 201, 4: 240, 8: 240, 12: 259, 16: 636, 20: 188, 32: 898, 44: 132, 48: 145, 52: 718, 68: 199, 76: 159, 96: 164, 100: 234, 108: 200, 124: 145}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 57, 61, 65, 77, 81, 85, 97, 105, 113, 117, 121, 125], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 2261, 'token_per_expert': {1: 353, 5: 225, 17: 69, 21: 63, 57: 70, 61: 103, 65: 113, 77: 80, 81: 76, 85: 118, 97: 72, 105: 508, 113: 72, 117: 130, 121: 72, 125: 137}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 14, 26, 42, 54, 58, 66, 70, 78, 82, 86, 90, 110, 114, 126], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 3604, 'token_per_expert': {2: 137, 14: 116, 26: 146, 42: 185, 54: 201, 58: 128, 66: 275, 70: 117, 78: 178, 82: 122, 86: 476, 90: 112, 110: 161, 114: 185, 126: 1065}}
INFO 05-06 11:02:14.527859.527859 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.931ms | allocate_experts_across_cpu_gpu: 0.521ms
INFO 05-06 11:02:14.527679.527679 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.771087646484375e-05 seconds
INFO 05-06 11:02:14.529149.529149 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.001224517822265625 seconds
INFO 05-06 11:02:14.553556.553556 lmp.py:1387] [layer_moe_fused] to time: 0.000152587890625 seconds
INFO 05-06 11:02:14.553762.553762 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.555015.555015 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012087821960449219 seconds
INFO 05-06 11:02:14.556244.556244 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006113052368164062 seconds
INFO 05-06 11:02:14.556286.556286 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019800662994384766 seconds
INFO 05-06 11:02:14.566406.566406 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0098724365234375 seconds
INFO 05-06 11:02:14.568829.568829 mlpmodule.py:2799] [fused_experts] gmm total=1.988ms E=32 S=2686 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.568455.568455 mlpmodule.py:2799] [fused_experts] gmm total=2.216ms E=32 S=3929 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.568873.568873 mlpmodule.py:2799] [fused_experts] gmm total=2.355ms E=32 S=5481 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.568199.568199 mlpmodule.py:2799] [fused_experts] gmm total=2.315ms E=32 S=4288 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.569633.569633 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003730297088623047 seconds
INFO 05-06 11:02:14.570095.570095 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.626678466796875e-05 seconds
DEBUG 05-06 11:02:14.570436.570436 cuda_h.py:27] end *layer_moe_fused cost 44.189 ms
DEBUG 05-06 11:02:14.576401.576401 cuda_h.py:27] end prefill_layer cost 53.404 ms
DEBUG 05-06 11:02:14.576794.576794 lmp.py:841] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 11:02:14.576371.576371 lmp.py:824] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 11:02:14.579848.579848 cuda_h.py:27] end *sagl cost 2.911 ms
experts_cpu_alloc {'expert_ids': [79, 7, 127, 51, 15, 11, 19, 83, 87, 123, 67, 91, 59, 119, 99, 112, 8, 88, 44, 92, 32, 96, 16, 36, 60, 116, 100, 104, 48, 0, 120, 25, 41, 77, 81, 93, 29, 85, 1, 65, 117, 97, 113, 33, 109, 9, 46, 110, 42, 102, 82, 30, 34, 2, 62, 90, 66, 14, 38, 118, 126], 'token_total': 2126, 'token_per_expert': {79: 7, 7: 10, 127: 11, 51: 12, 15: 13, 11: 22, 19: 36, 83: 39, 87: 42, 123: 45, 67: 46, 91: 47, 59: 51, 119: 59, 99: 61, 112: 10, 8: 13, 88: 15, 44: 30, 92: 31, 32: 37, 96: 43, 16: 44, 36: 45, 60: 50, 116: 64, 100: 66, 104: 79, 48: 82, 0: 84, 120: 84, 25: 1, 41: 1, 77: 14, 81: 15, 93: 19, 29: 20, 85: 20, 1: 23, 65: 27, 117: 35, 97: 36, 113: 50, 33: 52, 109: 52, 9: 55, 46: 1, 110: 4, 42: 15, 102: 16, 82: 17, 30: 20, 34: 24, 2: 27, 62: 28, 90: 29, 66: 38, 14: 44, 38: 50, 118: 54, 126: 61}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 23, 27, 31, 35, 39, 43, 47, 55, 63, 71, 75, 95, 103, 107, 111], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 4092, 'token_per_expert': {3: 86, 23: 633, 27: 321, 31: 150, 35: 111, 39: 318, 43: 305, 47: 172, 55: 100, 63: 183, 71: 215, 75: 555, 95: 507, 103: 133, 107: 229, 111: 74}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 12, 20, 24, 28, 40, 52, 56, 64, 68, 72, 76, 80, 84, 108, 124], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 3764, 'token_per_expert': {4: 153, 12: 151, 20: 214, 24: 603, 28: 185, 40: 261, 52: 303, 56: 201, 64: 182, 68: 174, 72: 325, 76: 578, 80: 114, 84: 96, 108: 125, 124: 99}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 13, 17, 21, 37, 45, 49, 53, 57, 61, 69, 73, 89, 101, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3649, 'token_per_expert': {5: 115, 13: 68, 17: 162, 21: 357, 37: 763, 45: 111, 49: 201, 53: 289, 57: 124, 61: 268, 69: 455, 73: 107, 89: 319, 101: 225, 125: 85}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 18, 22, 54, 58, 70, 74, 78, 86, 94, 98, 106, 114, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 2753, 'token_per_expert': {6: 113, 10: 215, 18: 208, 22: 155, 54: 141, 58: 334, 70: 110, 74: 513, 78: 119, 86: 350, 94: 73, 98: 104, 106: 152, 114: 79, 122: 87}}
INFO 05-06 11:02:14.584410.584410 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 2.731ms | allocate_experts_across_cpu_gpu: 0.458ms
INFO 05-06 11:02:14.584567.584567 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.318092346191406e-05 seconds
INFO 05-06 11:02:14.585659.585659 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011758804321289062 seconds
INFO 05-06 11:02:14.609516.609516 lmp.py:1387] [layer_moe_fused] to time: 0.0001556873321533203 seconds
INFO 05-06 11:02:14.609384.609384 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.610162.610162 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001220703125 seconds
INFO 05-06 11:02:14.611621.611621 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000576019287109375 seconds
INFO 05-06 11:02:14.611087.611087 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019450187683105469 seconds
INFO 05-06 11:02:14.621174.621174 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009482383728027344 seconds
INFO 05-06 11:02:14.623409.623409 mlpmodule.py:2799] [fused_experts] gmm total=2.290ms E=32 S=4069 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.623905.623905 mlpmodule.py:2799] [fused_experts] gmm total=2.399ms E=32 S=3181 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.624715.624715 mlpmodule.py:2799] [fused_experts] gmm total=2.574ms E=32 S=4541 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.624717.624717 mlpmodule.py:2799] [fused_experts] gmm total=2.710ms E=32 S=4593 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.625652.625652 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004086732864379883 seconds
INFO 05-06 11:02:14.625352.625352 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.3882598876953125e-05 seconds
DEBUG 05-06 11:02:14.625922.625922 cuda_h.py:27] end *layer_moe_fused cost 45.003 ms
DEBUG 05-06 11:02:14.631711.631711 cuda_h.py:27] end prefill_layer cost 55.227 ms
DEBUG 05-06 11:02:14.631051.631051 lmp.py:841] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 11:02:14.631582.631582 lmp.py:824] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 11:02:14.634639.634639 cuda_h.py:27] end *sagl cost 1.985 ms
experts_cpu_alloc {'expert_ids': [115, 55, 63, 11, 59, 19, 27, 39, 67, 23, 51, 7, 15, 35, 107, 28, 20, 16, 52, 44, 0, 24, 96, 112, 124, 48, 116, 56, 108, 68, 12, 105, 117, 113, 41, 21, 25, 89, 9, 45, 109, 73, 97, 13, 29, 125, 57, 106, 6, 86, 22, 126, 102, 18, 42, 62, 94, 114, 82, 66, 70, 90, 122], 'token_total': 2345, 'token_per_expert': {115: 3, 55: 15, 63: 16, 11: 23, 59: 26, 19: 30, 27: 32, 39: 42, 67: 46, 23: 53, 51: 54, 7: 55, 15: 56, 35: 56, 107: 62, 28: 1, 20: 2, 16: 11, 52: 12, 44: 14, 0: 17, 24: 23, 96: 28, 112: 33, 124: 44, 48: 55, 116: 72, 56: 73, 108: 78, 68: 83, 12: 96, 105: 1, 117: 5, 113: 8, 41: 16, 21: 22, 25: 27, 89: 32, 9: 34, 45: 41, 109: 51, 73: 67, 97: 72, 13: 75, 29: 77, 125: 108, 57: 110, 106: 2, 6: 5, 86: 5, 22: 7, 126: 7, 102: 14, 18: 23, 42: 25, 62: 33, 94: 33, 114: 33, 82: 34, 66: 39, 70: 40, 90: 44, 122: 44}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 31, 43, 47, 71, 75, 83, 87, 91, 95, 99, 103, 111, 119, 123, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 3537, 'token_per_expert': {3: 278, 31: 232, 43: 238, 47: 104, 71: 101, 75: 138, 83: 378, 87: 200, 91: 75, 95: 142, 99: 540, 103: 80, 111: 371, 119: 347, 123: 159, 127: 154}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 32, 36, 40, 60, 64, 72, 76, 80, 84, 88, 92, 100, 104, 120], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 3773, 'token_per_expert': {4: 231, 8: 282, 32: 366, 36: 338, 40: 269, 60: 223, 64: 300, 72: 216, 76: 352, 80: 131, 84: 214, 88: 146, 92: 126, 100: 155, 104: 251, 120: 173}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 33, 37, 49, 53, 61, 65, 69, 77, 81, 85, 93, 101, 121], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 3724, 'token_per_expert': {1: 143, 5: 116, 17: 176, 33: 276, 37: 120, 49: 145, 53: 307, 61: 218, 65: 233, 69: 142, 77: 283, 81: 202, 85: 404, 93: 177, 101: 278, 121: 504}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 26, 30, 34, 38, 46, 50, 54, 58, 74, 78, 98, 110, 118], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 3005, 'token_per_expert': {2: 386, 10: 115, 14: 237, 26: 64, 30: 84, 34: 141, 38: 173, 46: 66, 50: 375, 54: 358, 58: 258, 74: 59, 78: 172, 98: 52, 110: 218, 118: 247}}
INFO 05-06 11:02:14.636811.636811 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.979ms | allocate_experts_across_cpu_gpu: 0.458ms
INFO 05-06 11:02:14.636875.636875 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.723403930664062e-05 seconds
INFO 05-06 11:02:14.638680.638680 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011937618255615234 seconds
INFO 05-06 11:02:14.663920.663920 lmp.py:1387] [layer_moe_fused] to time: 0.00015544891357421875 seconds
INFO 05-06 11:02:14.664464.664464 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.665970.665970 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00122833251953125 seconds
INFO 05-06 11:02:14.666038.666038 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005688667297363281 seconds
INFO 05-06 11:02:14.666219.666219 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019469261169433594 seconds
INFO 05-06 11:02:14.676935.676935 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009855270385742188 seconds
INFO 05-06 11:02:14.678969.678969 mlpmodule.py:2799] [fused_experts] gmm total=2.045ms E=32 S=4470 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.678235.678235 mlpmodule.py:2799] [fused_experts] gmm total=2.396ms E=32 S=4106 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.678007.678007 mlpmodule.py:2799] [fused_experts] gmm total=2.443ms E=32 S=4415 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.678963.678963 mlpmodule.py:2799] [fused_experts] gmm total=2.412ms E=32 S=3393 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.679798.679798 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003744363784790039 seconds
INFO 05-06 11:02:14.680875.680875 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.412101745605469e-05 seconds
DEBUG 05-06 11:02:14.680949.680949 cuda_h.py:27] end *layer_moe_fused cost 45.115 ms
DEBUG 05-06 11:02:14.686265.686265 cuda_h.py:27] end prefill_layer cost 54.393 ms
DEBUG 05-06 11:02:14.686605.686605 lmp.py:841] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 11:02:14.686183.686183 lmp.py:824] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 11:02:14.688622.688622 cuda_h.py:27] end *sagl cost 2.021 ms
experts_cpu_alloc {'expert_ids': [91, 107, 71, 115, 95, 87, 67, 127, 43, 55, 103, 59, 111, 99, 11, 83, 116, 124, 8, 100, 120, 4, 56, 68, 20, 112, 108, 12, 84, 72, 81, 85, 49, 113, 93, 105, 77, 57, 101, 65, 45, 29, 121, 73, 97, 25, 17, 82, 34, 74, 46, 62, 6, 66, 70, 30, 54, 94, 114, 126, 18, 42], 'token_total': 1977, 'token_per_expert': {91: 2, 107: 3, 71: 5, 115: 11, 95: 13, 87: 19, 67: 21, 127: 25, 43: 26, 55: 36, 103: 49, 59: 56, 111: 66, 99: 81, 11: 88, 83: 88, 116: 4, 124: 4, 8: 12, 100: 13, 120: 26, 4: 40, 56: 44, 68: 46, 20: 50, 112: 51, 108: 52, 12: 66, 84: 79, 72: 82, 81: 1, 85: 3, 49: 8, 113: 8, 93: 10, 105: 12, 77: 15, 57: 17, 101: 18, 65: 19, 45: 30, 29: 41, 121: 42, 73: 65, 97: 68, 25: 73, 17: 80, 82: 4, 34: 7, 74: 12, 46: 14, 62: 14, 6: 16, 66: 16, 70: 17, 30: 24, 54: 24, 94: 24, 114: 28, 126: 34, 18: 37, 42: 38}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 39, 47, 51, 63, 75, 79, 119, 123], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 3356, 'token_per_expert': {3: 347, 7: 327, 15: 124, 19: 111, 23: 233, 27: 186, 31: 114, 35: 140, 39: 149, 47: 100, 51: 492, 63: 195, 75: 194, 79: 216, 119: 141, 123: 287}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 24, 36, 40, 44, 48, 52, 60, 64, 76, 80, 88, 92, 96, 104], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 4602, 'token_per_expert': {0: 103, 16: 173, 24: 393, 36: 85, 40: 243, 44: 580, 48: 130, 52: 930, 60: 87, 64: 492, 76: 189, 80: 230, 88: 147, 92: 551, 96: 166, 104: 103}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 33, 37, 41, 53, 61, 69, 89, 109, 117, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 3672, 'token_per_expert': {1: 149, 5: 89, 9: 232, 13: 86, 21: 241, 33: 93, 37: 492, 41: 142, 53: 121, 61: 264, 69: 177, 89: 821, 109: 207, 117: 386, 125: 172}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 22, 26, 38, 50, 58, 86, 90, 98, 102, 106, 110, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 2777, 'token_per_expert': {2: 115, 10: 115, 22: 68, 26: 168, 38: 678, 50: 218, 58: 58, 86: 43, 90: 107, 98: 76, 102: 197, 106: 142, 110: 38, 118: 72, 122: 682}}
INFO 05-06 11:02:14.690422.690422 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.562ms | allocate_experts_across_cpu_gpu: 0.456ms
INFO 05-06 11:02:14.690057.690057 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.318092346191406e-05 seconds
INFO 05-06 11:02:14.692884.692884 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0012791156768798828 seconds
INFO 05-06 11:02:14.715705.715705 lmp.py:1387] [layer_moe_fused] to time: 0.0001552104949951172 seconds
INFO 05-06 11:02:14.715063.715063 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.716386.716386 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012660026550292969 seconds
INFO 05-06 11:02:14.717323.717323 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005872249603271484 seconds
INFO 05-06 11:02:14.717187.717187 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002044677734375 seconds
INFO 05-06 11:02:14.727207.727207 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009865045547485352 seconds
INFO 05-06 11:02:14.729621.729621 mlpmodule.py:2799] [fused_experts] gmm total=2.407ms E=32 S=3945 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.730262.730262 mlpmodule.py:2799] [fused_experts] gmm total=2.435ms E=32 S=4182 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.730403.730403 mlpmodule.py:2799] [fused_experts] gmm total=2.454ms E=32 S=3086 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.730507.730507 mlpmodule.py:2799] [fused_experts] gmm total=2.708ms E=32 S=5171 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.731313.731313 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004057168960571289 seconds
INFO 05-06 11:02:14.731411.731411 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 6.413459777832031e-05 seconds
DEBUG 05-06 11:02:14.732287.732287 cuda_h.py:27] end *layer_moe_fused cost 42.283 ms
DEBUG 05-06 11:02:14.738605.738605 cuda_h.py:27] end prefill_layer cost 51.779 ms
DEBUG 05-06 11:02:14.738422.738422 lmp.py:841] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 11:02:14.738238.738238 lmp.py:824] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 11:02:14.740682.740682 cuda_h.py:27] end *sagl cost 1.968 ms
experts_cpu_alloc {'expert_ids': [87, 99, 23, 75, 51, 119, 67, 7, 127, 91, 11, 31, 35, 47, 124, 16, 24, 104, 36, 80, 120, 0, 84, 60, 12, 76, 20, 52, 29, 97, 17, 25, 89, 69, 61, 101, 117, 105, 1, 93, 121, 113, 41, 33, 85, 14, 22, 70, 6, 106, 126, 34, 10, 26, 98, 86, 38, 58, 62, 90, 110], 'token_total': 2415, 'token_per_expert': {87: 2, 99: 3, 23: 4, 75: 6, 51: 7, 119: 7, 67: 8, 7: 9, 127: 9, 91: 10, 11: 13, 31: 20, 35: 26, 47: 38, 124: 8, 16: 11, 24: 27, 104: 28, 36: 37, 80: 49, 120: 49, 0: 52, 84: 66, 60: 75, 12: 76, 76: 83, 20: 86, 52: 93, 29: 7, 97: 7, 17: 11, 25: 14, 89: 16, 69: 19, 61: 25, 101: 36, 117: 37, 105: 63, 1: 65, 93: 75, 121: 83, 113: 123, 41: 127, 33: 158, 85: 158, 14: 2, 22: 7, 70: 13, 6: 16, 106: 17, 126: 18, 34: 29, 10: 35, 26: 35, 98: 36, 86: 42, 38: 43, 58: 44, 62: 48, 90: 51, 110: 53}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 19, 27, 43, 55, 59, 63, 71, 79, 83, 95, 103, 107, 111, 123], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 2864, 'token_per_expert': {3: 246, 15: 94, 19: 42, 27: 158, 43: 148, 55: 69, 59: 289, 63: 494, 71: 115, 79: 116, 83: 77, 95: 45, 103: 52, 107: 632, 111: 51, 123: 236}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 28, 32, 40, 44, 56, 64, 68, 72, 88, 92, 100, 108, 112, 116], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3769, 'token_per_expert': {4: 214, 8: 239, 28: 217, 32: 172, 40: 277, 44: 182, 56: 256, 64: 102, 68: 940, 72: 106, 88: 216, 92: 210, 100: 142, 108: 182, 112: 156, 116: 158}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 21, 37, 45, 49, 53, 57, 65, 73, 77, 81, 109, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 4515, 'token_per_expert': {5: 210, 9: 174, 13: 209, 21: 314, 37: 314, 45: 579, 49: 671, 53: 171, 57: 327, 65: 369, 73: 338, 77: 283, 81: 211, 109: 172, 125: 173}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 18, 30, 42, 46, 50, 54, 66, 74, 82, 94, 102, 114, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 2821, 'token_per_expert': {2: 85, 18: 82, 30: 406, 42: 204, 46: 164, 50: 103, 54: 65, 66: 167, 74: 61, 82: 110, 94: 737, 102: 335, 114: 59, 118: 81, 122: 162}}
INFO 05-06 11:02:14.743098.743098 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.954ms | allocate_experts_across_cpu_gpu: 0.455ms
INFO 05-06 11:02:14.743242.743242 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.389617919921875e-05 seconds
INFO 05-06 11:02:14.744708.744708 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0012116432189941406 seconds
INFO 05-06 11:02:14.771879.771879 lmp.py:1387] [layer_moe_fused] to time: 0.0001652240753173828 seconds
INFO 05-06 11:02:14.771516.771516 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.773909.773909 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012187957763671875 seconds
INFO 05-06 11:02:14.773104.773104 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005929470062255859 seconds
INFO 05-06 11:02:14.773523.773523 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019600391387939453 seconds
INFO 05-06 11:02:14.784453.784453 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01071619987487793 seconds
INFO 05-06 11:02:14.787896.787896 mlpmodule.py:2799] [fused_experts] gmm total=2.282ms E=32 S=3026 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.787895.787895 mlpmodule.py:2799] [fused_experts] gmm total=2.293ms E=32 S=3310 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.787675.787675 mlpmodule.py:2799] [fused_experts] gmm total=2.552ms E=32 S=4509 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.787175.787175 mlpmodule.py:2799] [fused_experts] gmm total=2.584ms E=32 S=5539 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.788227.788227 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004027843475341797 seconds
INFO 05-06 11:02:14.788589.788589 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.340576171875e-05 seconds
DEBUG 05-06 11:02:14.789896.789896 cuda_h.py:27] end *layer_moe_fused cost 47.784 ms
DEBUG 05-06 11:02:14.795233.795233 cuda_h.py:27] end prefill_layer cost 57.572 ms
DEBUG 05-06 11:02:14.795142.795142 lmp.py:841] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 11:02:14.796720.796720 lmp.py:824] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 11:02:14.798205.798205 cuda_h.py:27] end *sagl cost 1.984 ms
experts_cpu_alloc {'expert_ids': [27, 19, 39, 15, 99, 107, 23, 47, 115, 71, 43, 59, 119, 3, 28, 116, 60, 104, 96, 64, 52, 88, 16, 56, 40, 20, 44, 0, 32, 24, 117, 89, 77, 25, 101, 9, 17, 69, 93, 113, 121, 45, 21, 81, 125, 66, 98, 22, 94, 54, 114, 74, 126, 50, 106, 42, 118, 38, 82, 70, 58], 'token_total': 2161, 'token_per_expert': {27: 2, 19: 3, 39: 3, 15: 4, 99: 4, 107: 8, 23: 12, 47: 13, 115: 24, 71: 45, 43: 46, 59: 50, 119: 51, 3: 56, 28: 3, 116: 3, 60: 7, 104: 9, 96: 14, 64: 21, 52: 22, 88: 29, 16: 31, 56: 35, 40: 43, 20: 49, 44: 57, 0: 61, 32: 64, 24: 68, 117: 1, 89: 3, 77: 7, 25: 14, 101: 15, 9: 21, 17: 23, 69: 23, 93: 26, 113: 38, 121: 41, 45: 49, 21: 50, 81: 74, 125: 90, 66: 3, 98: 8, 22: 12, 94: 17, 54: 27, 114: 32, 74: 38, 126: 47, 50: 57, 106: 60, 42: 73, 118: 82, 38: 93, 82: 93, 70: 102, 58: 105}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 31, 35, 51, 55, 67, 75, 79, 83, 87, 95, 103, 111, 123, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 2439, 'token_per_expert': {7: 194, 11: 465, 31: 144, 35: 124, 51: 180, 55: 59, 67: 178, 75: 129, 79: 101, 83: 228, 87: 88, 95: 108, 103: 183, 111: 93, 123: 72, 127: 93}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 36, 48, 68, 72, 76, 80, 84, 92, 100, 112, 120, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3335, 'token_per_expert': {4: 178, 8: 182, 12: 134, 36: 160, 48: 363, 68: 139, 72: 174, 76: 240, 80: 74, 84: 173, 92: 311, 100: 464, 112: 262, 120: 256, 124: 225}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 29, 33, 37, 41, 53, 57, 61, 65, 73, 97, 105, 109], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4235, 'token_per_expert': {1: 433, 5: 623, 13: 147, 29: 286, 33: 145, 37: 219, 41: 249, 53: 269, 57: 196, 61: 197, 65: 482, 73: 296, 97: 167, 105: 336, 109: 190}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 26, 30, 34, 46, 62, 78, 86, 90, 102, 110, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4214, 'token_per_expert': {2: 134, 6: 630, 10: 152, 18: 331, 26: 358, 30: 167, 34: 111, 46: 333, 62: 225, 78: 763, 86: 153, 90: 192, 102: 140, 110: 189, 122: 336}}
INFO 05-06 11:02:14.800635.800635 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.629ms | allocate_experts_across_cpu_gpu: 0.449ms
INFO 05-06 11:02:14.800017.800017 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.29425048828125e-05 seconds
INFO 05-06 11:02:14.801673.801673 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.001199960708618164 seconds
INFO 05-06 11:02:14.826836.826836 lmp.py:1387] [layer_moe_fused] to time: 0.00015401840209960938 seconds
INFO 05-06 11:02:14.826704.826704 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.827482.827482 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012211799621582031 seconds
INFO 05-06 11:02:14.828154.828154 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005919933319091797 seconds
INFO 05-06 11:02:14.828619.828619 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001963376998901367 seconds
INFO 05-06 11:02:14.838029.838029 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010390758514404297 seconds
INFO 05-06 11:02:14.841176.841176 mlpmodule.py:2799] [fused_experts] gmm total=2.182ms E=32 S=5063 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.841259.841259 mlpmodule.py:2799] [fused_experts] gmm total=2.485ms E=32 S=2760 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.841984.841984 mlpmodule.py:2799] [fused_experts] gmm total=2.443ms E=32 S=4710 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.841020.841020 mlpmodule.py:2799] [fused_experts] gmm total=2.596ms E=32 S=3851 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.842853.842853 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003962993621826172 seconds
INFO 05-06 11:02:14.843215.843215 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.340576171875e-05 seconds
DEBUG 05-06 11:02:14.843979.843979 cuda_h.py:27] end *layer_moe_fused cost 44.171 ms
DEBUG 05-06 11:02:14.850011.850011 cuda_h.py:27] end prefill_layer cost 54.174 ms
DEBUG 05-06 11:02:14.850589.850589 lmp.py:841] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 11:02:14.850405.850405 lmp.py:824] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 11:02:14.852769.852769 cuda_h.py:27] end *sagl cost 1.966 ms
experts_cpu_alloc {'expert_ids': [91, 23, 71, 95, 67, 87, 39, 27, 79, 51, 63, 83, 3, 47, 31, 107, 80, 12, 56, 36, 104, 96, 60, 20, 84, 112, 32, 4, 48, 44, 40, 21, 29, 17, 49, 13, 77, 105, 37, 97, 61, 81, 65, 9, 25, 57, 18, 78, 114, 22, 106, 122, 54, 10, 14, 98, 34, 110, 62, 6, 2, 42], 'token_total': 2011, 'token_per_expert': {91: 7, 23: 8, 71: 8, 95: 10, 67: 16, 87: 16, 39: 23, 27: 26, 79: 35, 51: 37, 63: 37, 83: 42, 3: 68, 47: 79, 31: 90, 107: 93, 80: 1, 12: 3, 56: 4, 36: 5, 104: 5, 96: 19, 60: 31, 20: 34, 84: 36, 112: 60, 32: 61, 4: 65, 48: 82, 44: 88, 40: 90, 21: 3, 29: 4, 17: 6, 49: 6, 13: 8, 77: 9, 105: 10, 37: 11, 97: 12, 61: 13, 81: 23, 65: 29, 9: 38, 25: 39, 57: 40, 18: 3, 78: 6, 114: 6, 22: 7, 106: 15, 122: 25, 54: 30, 10: 32, 14: 32, 98: 41, 34: 42, 110: 42, 62: 55, 6: 75, 2: 79, 42: 91}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 15, 19, 35, 43, 55, 59, 75, 99, 103, 111, 115, 119, 123, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 3814, 'token_per_expert': {7: 176, 11: 103, 15: 137, 19: 97, 35: 557, 43: 162, 55: 276, 59: 458, 75: 209, 99: 109, 103: 421, 111: 145, 115: 150, 119: 348, 123: 201, 127: 265}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 16, 24, 28, 64, 68, 72, 76, 88, 92, 100, 108, 116, 120, 124], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 5154, 'token_per_expert': {0: 103, 8: 253, 16: 102, 24: 500, 28: 163, 64: 622, 68: 301, 72: 566, 76: 164, 88: 133, 92: 271, 100: 1156, 108: 223, 116: 247, 120: 245, 124: 105}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 41, 45, 53, 69, 73, 85, 89, 93, 101, 109, 113, 117, 125], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 2208, 'token_per_expert': {1: 168, 5: 69, 33: 95, 41: 84, 45: 87, 53: 240, 69: 107, 73: 299, 85: 76, 89: 141, 93: 359, 101: 65, 109: 58, 113: 61, 117: 243, 125: 56}}
experts_gpu_alloc_device_3 {'expert_ids': [26, 30, 38, 46, 58, 66, 70, 74, 82, 86, 90, 94, 102, 118, 126], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3197, 'token_per_expert': {26: 96, 30: 166, 38: 219, 46: 176, 58: 99, 66: 151, 70: 169, 74: 517, 82: 201, 86: 220, 90: 169, 94: 203, 102: 110, 118: 167, 126: 534}}
INFO 05-06 11:02:14.855198.855198 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.974ms | allocate_experts_across_cpu_gpu: 0.459ms
INFO 05-06 11:02:14.855488.855488 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.604194641113281e-05 seconds
INFO 05-06 11:02:14.856941.856941 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011782646179199219 seconds
INFO 05-06 11:02:14.879083.879083 lmp.py:1387] [layer_moe_fused] to time: 0.0001506805419921875 seconds
INFO 05-06 11:02:14.879812.879812 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.880418.880418 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012345314025878906 seconds
INFO 05-06 11:02:14.881387.881387 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005650520324707031 seconds
INFO 05-06 11:02:14.881091.881091 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019507408142089844 seconds
INFO 05-06 11:02:14.891682.891682 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010071277618408203 seconds
INFO 05-06 11:02:14.894047.894047 mlpmodule.py:2799] [fused_experts] gmm total=1.998ms E=32 S=3778 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.894171.894171 mlpmodule.py:2799] [fused_experts] gmm total=2.177ms E=32 S=2459 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.894176.894176 mlpmodule.py:2799] [fused_experts] gmm total=2.437ms E=32 S=4409 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.894440.894440 mlpmodule.py:2799] [fused_experts] gmm total=2.568ms E=32 S=5738 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.895945.895945 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0038318634033203125 seconds
INFO 05-06 11:02:14.895943.895943 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 6.723403930664062e-05 seconds
DEBUG 05-06 11:02:14.896970.896970 cuda_h.py:27] end *layer_moe_fused cost 42.488 ms
DEBUG 05-06 11:02:14.901163.901163 cuda_h.py:27] end prefill_layer cost 51.499 ms
DEBUG 05-06 11:02:14.901166.901166 lmp.py:841] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 11:02:14.901266.901266 lmp.py:824] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 11:02:14.905365.905365 cuda_h.py:27] end *sagl cost 2.882 ms
experts_cpu_alloc {'expert_ids': [119, 111, 15, 55, 127, 27, 11, 99, 107, 23, 95, 51, 75, 103, 19, 91, 20, 88, 28, 64, 92, 60, 68, 40, 48, 36, 52, 12, 120, 124, 32, 112, 13, 69, 93, 77, 49, 41, 81, 101, 53, 57, 89, 9, 33, 73, 70, 50, 82, 74, 54, 110, 102, 58, 14, 62, 66, 10, 122, 38], 'token_total': 1069, 'token_per_expert': {119: 1, 111: 3, 15: 4, 55: 5, 127: 8, 27: 9, 11: 11, 99: 12, 107: 18, 23: 22, 95: 23, 51: 24, 75: 24, 103: 26, 19: 34, 91: 41, 20: 7, 88: 7, 28: 9, 64: 11, 92: 11, 60: 16, 68: 18, 40: 22, 48: 22, 36: 23, 52: 24, 12: 26, 120: 29, 124: 33, 32: 35, 112: 42, 13: 3, 69: 3, 93: 4, 77: 5, 49: 8, 41: 9, 81: 11, 101: 12, 53: 20, 57: 28, 89: 29, 9: 37, 33: 46, 73: 50, 70: 1, 50: 3, 82: 5, 74: 7, 54: 8, 110: 8, 102: 9, 58: 12, 14: 15, 62: 19, 66: 20, 10: 28, 122: 32, 38: 37}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 39, 43, 47, 59, 67, 71, 79, 83, 87, 115, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3792, 'token_per_expert': {3: 1186, 7: 1031, 31: 74, 35: 117, 39: 258, 43: 189, 47: 152, 59: 52, 67: 231, 71: 61, 79: 145, 83: 71, 87: 54, 115: 71, 123: 100}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 24, 44, 56, 72, 76, 80, 84, 100, 104, 108, 116], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3408, 'token_per_expert': {0: 1030, 4: 1027, 8: 88, 16: 95, 24: 61, 44: 126, 56: 346, 72: 94, 76: 54, 80: 65, 84: 90, 100: 66, 104: 87, 108: 119, 116: 60}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 21, 25, 29, 37, 61, 65, 85, 97, 105, 109, 117, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4484, 'token_per_expert': {1: 1191, 5: 1185, 17: 80, 21: 409, 25: 189, 29: 166, 37: 193, 61: 195, 65: 245, 85: 120, 97: 140, 105: 63, 109: 102, 117: 63, 125: 143}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 26, 30, 34, 42, 46, 78, 86, 90, 98, 106, 118], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3631, 'token_per_expert': {2: 1108, 6: 1065, 18: 118, 22: 43, 26: 83, 30: 69, 34: 42, 42: 39, 46: 183, 78: 103, 86: 291, 90: 89, 98: 182, 106: 71, 118: 145}}
INFO 05-06 11:02:14.909200.909200 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 2.592ms | allocate_experts_across_cpu_gpu: 0.445ms
INFO 05-06 11:02:14.909066.909066 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.318092346191406e-05 seconds
INFO 05-06 11:02:14.910412.910412 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010833740234375 seconds
INFO 05-06 11:02:14.924536.924536 lmp.py:1387] [layer_moe_fused] to time: 0.0001289844512939453 seconds
INFO 05-06 11:02:14.924231.924231 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.925802.925802 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011746883392333984 seconds
INFO 05-06 11:02:14.926671.926671 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005652904510498047 seconds
INFO 05-06 11:02:14.926137.926137 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0018856525421142578 seconds
INFO 05-06 11:02:14.936403.936403 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00970149040222168 seconds
INFO 05-06 11:02:14.938497.938497 mlpmodule.py:2799] [fused_experts] gmm total=2.144ms E=32 S=4057 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.938951.938951 mlpmodule.py:2799] [fused_experts] gmm total=2.106ms E=32 S=3835 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.938386.938386 mlpmodule.py:2799] [fused_experts] gmm total=2.355ms E=32 S=3743 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.938229.938229 mlpmodule.py:2799] [fused_experts] gmm total=2.362ms E=32 S=4749 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.940834.940834 lmp.py:1500] [layer_moe_fused] experts compute time: 0.00388336181640625 seconds
INFO 05-06 11:02:14.940388.940388 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.3882598876953125e-05 seconds
DEBUG 05-06 11:02:14.940332.940332 cuda_h.py:27] end *layer_moe_fused cost 34.456 ms
DEBUG 05-06 11:02:14.946408.946408 cuda_h.py:27] end prefill_layer cost 44.759 ms
DEBUG 05-06 11:02:14.946317.946317 lmp.py:841] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 11:02:14.946180.946180 lmp.py:824] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 11:02:14.949313.949313 cuda_h.py:27] end *sagl cost 1.993 ms
experts_cpu_alloc {'expert_ids': [51, 123, 59, 95, 55, 99, 15, 87, 115, 47, 107, 31, 119, 79, 88, 24, 112, 116, 80, 84, 68, 76, 124, 104, 96, 28, 92, 120, 8, 69, 85, 101, 113, 117, 125, 93, 21, 25, 65, 89, 61, 57, 53, 105, 9, 54, 22, 58, 78, 126, 10, 38, 46, 102, 26, 66, 18, 62, 42, 106, 118], 'token_total': 864, 'token_per_expert': {51: 2, 123: 4, 59: 5, 95: 6, 55: 7, 99: 9, 15: 10, 87: 12, 115: 14, 47: 16, 107: 18, 31: 19, 119: 23, 79: 24, 88: 1, 24: 2, 112: 5, 116: 7, 80: 8, 84: 8, 68: 11, 76: 11, 124: 19, 104: 21, 96: 23, 28: 25, 92: 27, 120: 31, 8: 32, 69: 1, 85: 1, 101: 1, 113: 1, 117: 1, 125: 1, 93: 2, 21: 4, 25: 6, 65: 11, 89: 11, 61: 13, 57: 24, 53: 26, 105: 30, 9: 31, 54: 2, 22: 3, 58: 5, 78: 8, 126: 8, 10: 9, 38: 16, 46: 19, 102: 19, 26: 20, 66: 21, 18: 22, 62: 22, 42: 28, 106: 44, 118: 54}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 35, 43, 63, 67, 71, 75, 83, 91, 111, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3775, 'token_per_expert': {3: 1028, 7: 1084, 11: 214, 19: 148, 23: 101, 27: 226, 35: 75, 43: 53, 63: 210, 67: 114, 71: 165, 75: 32, 83: 82, 91: 152, 111: 29, 127: 62}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 20, 32, 36, 40, 44, 48, 52, 56, 60, 64, 100, 108], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 3708, 'token_per_expert': {0: 1024, 4: 1127, 12: 172, 16: 111, 20: 38, 32: 38, 36: 52, 40: 34, 44: 167, 48: 85, 52: 191, 56: 148, 60: 77, 64: 291, 100: 46, 108: 107}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 29, 33, 37, 45, 49, 73, 77, 81, 97, 109, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4031, 'token_per_expert': {1: 1104, 5: 1081, 13: 54, 17: 68, 29: 86, 33: 199, 37: 65, 45: 130, 49: 31, 73: 132, 77: 147, 81: 45, 97: 301, 109: 68, 121: 520}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 34, 50, 70, 74, 82, 86, 90, 94, 98, 110, 114, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4006, 'token_per_expert': {2: 1027, 6: 1229, 30: 71, 34: 157, 50: 64, 70: 271, 74: 59, 82: 61, 86: 79, 90: 461, 94: 131, 98: 111, 110: 83, 114: 136, 122: 66}}
INFO 05-06 11:02:14.951462.951462 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.908ms | allocate_experts_across_cpu_gpu: 0.452ms
INFO 05-06 11:02:14.951130.951130 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.179115295410156e-05 seconds
INFO 05-06 11:02:14.952378.952378 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010824203491210938 seconds
INFO 05-06 11:02:14.965040.965040 lmp.py:1387] [layer_moe_fused] to time: 0.00012159347534179688 seconds
INFO 05-06 11:02:14.965582.965582 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:14.966080.966080 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001157522201538086 seconds
INFO 05-06 11:02:14.967379.967379 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005664825439453125 seconds
INFO 05-06 11:02:14.967514.967514 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0018706321716308594 seconds
INFO 05-06 11:02:14.977374.977374 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009821653366088867 seconds
INFO 05-06 11:02:14.979450.979450 mlpmodule.py:2799] [fused_experts] gmm total=1.874ms E=32 S=4306 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.979302.979302 mlpmodule.py:2799] [fused_experts] gmm total=2.112ms E=32 S=3939 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.979189.979189 mlpmodule.py:2799] [fused_experts] gmm total=2.220ms E=32 S=4195 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.979562.979562 mlpmodule.py:2799] [fused_experts] gmm total=2.392ms E=32 S=3944 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:14.980192.980192 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003570556640625 seconds
INFO 05-06 11:02:14.980269.980269 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.221366882324219e-05 seconds
DEBUG 05-06 11:02:14.981530.981530 cuda_h.py:27] end *layer_moe_fused cost 31.293 ms
DEBUG 05-06 11:02:14.987081.987081 cuda_h.py:27] end prefill_layer cost 40.755 ms
DEBUG 05-06 11:02:14.987256.987256 lmp.py:841] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 11:02:14.987834.987834 lmp.py:824] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 11:02:14.989946.989946 cuda_h.py:27] end *sagl cost 1.954 ms
experts_cpu_alloc {'expert_ids': [75, 15, 115, 55, 27, 103, 127, 23, 119, 31, 47, 99, 43, 51, 40, 84, 96, 76, 24, 32, 108, 12, 92, 124, 112, 8, 48, 72, 88, 37, 61, 57, 65, 81, 101, 17, 33, 53, 121, 21, 77, 73, 13, 29, 54, 62, 94, 98, 30, 86, 102, 22, 122, 46, 42, 66, 74, 38, 78, 126, 26], 'token_total': 809, 'token_per_expert': {75: 3, 15: 4, 115: 4, 55: 7, 27: 12, 103: 12, 127: 12, 23: 13, 119: 16, 31: 22, 47: 27, 99: 27, 43: 30, 51: 31, 40: 1, 84: 1, 96: 2, 76: 10, 24: 14, 32: 14, 108: 14, 12: 16, 92: 19, 124: 20, 112: 22, 8: 23, 48: 30, 72: 32, 88: 43, 37: 1, 61: 3, 57: 4, 65: 4, 81: 5, 101: 6, 17: 7, 33: 7, 53: 8, 121: 8, 21: 11, 77: 15, 73: 16, 13: 18, 29: 22, 54: 1, 62: 1, 94: 1, 98: 1, 30: 2, 86: 3, 102: 3, 22: 7, 122: 10, 46: 12, 42: 13, 66: 15, 74: 19, 38: 23, 78: 23, 126: 28, 26: 31}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 35, 39, 63, 67, 71, 79, 83, 87, 91, 107, 111, 123], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3779, 'token_per_expert': {3: 1160, 7: 1085, 11: 72, 19: 50, 35: 224, 39: 53, 63: 68, 67: 79, 71: 66, 79: 44, 83: 115, 87: 44, 91: 79, 107: 335, 111: 74, 123: 231}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 36, 44, 52, 56, 60, 64, 68, 80, 100, 104, 116, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4232, 'token_per_expert': {0: 1060, 4: 1031, 16: 611, 36: 67, 44: 75, 52: 189, 56: 52, 60: 169, 64: 111, 68: 364, 80: 158, 100: 71, 104: 158, 116: 55, 120: 61}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 25, 41, 45, 49, 69, 85, 89, 93, 97, 109, 117, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3331, 'token_per_expert': {1: 1032, 5: 1055, 9: 36, 25: 38, 41: 58, 45: 216, 49: 60, 69: 172, 85: 208, 89: 80, 93: 59, 97: 58, 109: 38, 117: 194, 125: 27}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 34, 50, 58, 70, 82, 90, 106, 110, 114, 118], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 4233, 'token_per_expert': {2: 1296, 6: 1045, 10: 93, 14: 60, 18: 216, 34: 63, 50: 55, 58: 533, 70: 160, 82: 71, 90: 113, 106: 48, 110: 347, 114: 101, 118: 32}}
INFO 05-06 11:02:14.992309.992309 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.570ms | allocate_experts_across_cpu_gpu: 0.496ms
INFO 05-06 11:02:14.992136.992136 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.794929504394531e-05 seconds
INFO 05-06 11:02:14.993864.993864 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010802745819091797 seconds
INFO 05-06 11:02:15.005754.005754 lmp.py:1387] [layer_moe_fused] to time: 0.00012040138244628906 seconds
INFO 05-06 11:02:15.005251.005251 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.006219.006219 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011870861053466797 seconds
INFO 05-06 11:02:15.007440.007440 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005743503570556641 seconds
INFO 05-06 11:02:15.007859.007859 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001912832260131836 seconds
INFO 05-06 11:02:15.017479.017479 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009749650955200195 seconds
INFO 05-06 11:02:15.019788.019788 mlpmodule.py:2799] [fused_experts] gmm total=2.276ms E=32 S=3999 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.019918.019918 mlpmodule.py:2799] [fused_experts] gmm total=2.287ms E=32 S=3466 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.020618.020618 mlpmodule.py:2799] [fused_experts] gmm total=2.376ms E=32 S=4426 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.020998.020998 mlpmodule.py:2799] [fused_experts] gmm total=2.547ms E=32 S=4493 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.021219.021219 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0038292407989501953 seconds
INFO 05-06 11:02:15.021912.021912 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.1975250244140625e-05 seconds
DEBUG 05-06 11:02:15.021712.021712 cuda_h.py:27] end *layer_moe_fused cost 30.894 ms
DEBUG 05-06 11:02:15.027450.027450 cuda_h.py:27] end prefill_layer cost 40.159 ms
DEBUG 05-06 11:02:15.027453.027453 lmp.py:841] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 11:02:15.028268.028268 lmp.py:824] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 11:02:15.030223.030223 cuda_h.py:27] end *sagl cost 1.934 ms
experts_cpu_alloc {'expert_ids': [119, 11, 127, 55, 47, 107, 23, 83, 31, 63, 91, 71, 35, 115, 67, 75, 12, 16, 32, 48, 28, 100, 92, 80, 40, 116, 44, 72, 8, 68, 36, 93, 53, 33, 117, 121, 9, 109, 125, 41, 13, 81, 77, 61, 25, 22, 82, 98, 58, 34, 46, 54, 18, 74, 122, 110, 42, 118, 38], 'token_total': 912, 'token_per_expert': {119: 1, 11: 2, 127: 2, 55: 4, 47: 7, 107: 8, 23: 10, 83: 10, 31: 13, 63: 14, 91: 16, 71: 17, 35: 20, 115: 22, 67: 31, 75: 41, 12: 1, 16: 1, 32: 3, 48: 5, 28: 8, 100: 8, 92: 10, 80: 12, 40: 14, 116: 17, 44: 21, 72: 28, 8: 29, 68: 30, 36: 33, 93: 1, 53: 3, 33: 4, 117: 4, 121: 4, 9: 7, 109: 19, 125: 23, 41: 28, 13: 29, 81: 30, 77: 31, 61: 42, 25: 43, 22: 1, 82: 2, 98: 3, 58: 4, 34: 5, 46: 9, 54: 9, 18: 17, 74: 18, 122: 19, 110: 22, 42: 27, 118: 28, 38: 42}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 43, 51, 59, 79, 87, 95, 99, 103, 111, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3927, 'token_per_expert': {3: 1072, 7: 1024, 15: 100, 19: 92, 27: 202, 43: 183, 51: 75, 59: 67, 79: 50, 87: 302, 95: 239, 99: 56, 103: 58, 111: 308, 123: 99}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 24, 52, 56, 60, 76, 84, 88, 96, 104, 108, 112, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3593, 'token_per_expert': {0: 1043, 4: 1033, 20: 354, 24: 245, 52: 87, 56: 81, 60: 88, 76: 86, 84: 220, 88: 50, 96: 34, 104: 121, 108: 40, 112: 39, 124: 72}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 29, 37, 45, 49, 57, 65, 73, 85, 89, 97, 105, 113], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4630, 'token_per_expert': {1: 1046, 5: 1042, 17: 289, 29: 43, 37: 44, 45: 51, 49: 101, 57: 50, 65: 239, 73: 156, 85: 605, 89: 404, 97: 66, 105: 86, 113: 408}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 26, 30, 50, 66, 70, 78, 86, 90, 102, 114, 126], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3322, 'token_per_expert': {2: 1039, 6: 1024, 10: 51, 14: 67, 26: 50, 30: 49, 50: 82, 66: 70, 70: 122, 78: 110, 86: 85, 90: 77, 102: 85, 114: 338, 126: 73}}
INFO 05-06 11:02:15.032717.032717 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.933ms | allocate_experts_across_cpu_gpu: 0.438ms
INFO 05-06 11:02:15.032728.032728 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.29425048828125e-05 seconds
INFO 05-06 11:02:15.034081.034081 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010519027709960938 seconds
INFO 05-06 11:02:15.046085.046085 lmp.py:1387] [layer_moe_fused] to time: 0.0001239776611328125 seconds
INFO 05-06 11:02:15.046297.046297 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.048373.048373 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012328624725341797 seconds
INFO 05-06 11:02:15.048409.048409 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006158351898193359 seconds
INFO 05-06 11:02:15.048875.048875 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019969940185546875 seconds
INFO 05-06 11:02:15.058912.058912 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009566783905029297 seconds
INFO 05-06 11:02:15.061129.061129 mlpmodule.py:2799] [fused_experts] gmm total=2.117ms E=32 S=3528 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.061794.061794 mlpmodule.py:2799] [fused_experts] gmm total=2.384ms E=32 S=4145 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.061419.061419 mlpmodule.py:2799] [fused_experts] gmm total=2.346ms E=32 S=4898 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.061521.061521 mlpmodule.py:2799] [fused_experts] gmm total=2.476ms E=32 S=3813 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.062793.062793 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0038640499114990234 seconds
INFO 05-06 11:02:15.062916.062916 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.269050598144531e-05 seconds
DEBUG 05-06 11:02:15.063044.063044 cuda_h.py:27] end *layer_moe_fused cost 31.705 ms
DEBUG 05-06 11:02:15.069261.069261 cuda_h.py:27] end prefill_layer cost 41.103 ms
DEBUG 05-06 11:02:15.069647.069647 lmp.py:841] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 11:02:15.069225.069225 lmp.py:824] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 11:02:15.071883.071883 cuda_h.py:27] end *sagl cost 1.951 ms
experts_cpu_alloc {'expert_ids': [99, 67, 39, 63, 47, 59, 11, 15, 27, 19, 71, 91, 55, 23, 75, 127, 72, 16, 44, 84, 96, 116, 124, 32, 68, 60, 80, 56, 40, 20, 81, 9, 17, 77, 89, 93, 101, 57, 97, 29, 69, 113, 125, 21, 49, 22, 34, 30, 86, 110, 126, 58, 74, 10, 90, 26, 114, 94, 54, 106], 'token_total': 1135, 'token_per_expert': {99: 1, 67: 4, 39: 11, 63: 11, 47: 13, 59: 14, 11: 16, 15: 16, 27: 18, 19: 21, 71: 25, 91: 25, 55: 31, 23: 45, 75: 58, 127: 62, 72: 3, 16: 5, 44: 6, 84: 10, 96: 10, 116: 11, 124: 12, 32: 14, 68: 15, 60: 18, 80: 18, 56: 41, 40: 42, 20: 43, 81: 1, 9: 2, 17: 2, 77: 2, 89: 2, 93: 2, 101: 3, 57: 4, 97: 4, 29: 6, 69: 7, 113: 14, 125: 16, 21: 30, 49: 32, 22: 3, 34: 3, 30: 6, 86: 6, 110: 9, 126: 12, 58: 25, 74: 25, 10: 27, 90: 41, 26: 43, 114: 44, 94: 45, 54: 50, 106: 50}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 43, 51, 79, 83, 87, 95, 103, 111, 115, 119, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4299, 'token_per_expert': {3: 1123, 7: 1089, 31: 131, 35: 90, 43: 272, 51: 131, 79: 129, 83: 68, 87: 350, 95: 172, 103: 209, 111: 120, 115: 188, 119: 89, 123: 138}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 24, 28, 36, 48, 64, 76, 88, 100, 108, 112, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3751, 'token_per_expert': {0: 1038, 4: 1065, 8: 88, 12: 84, 24: 201, 28: 72, 36: 107, 48: 151, 64: 100, 76: 140, 88: 184, 100: 184, 108: 71, 112: 49, 120: 217}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 33, 37, 41, 45, 53, 61, 65, 85, 105, 109, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3702, 'token_per_expert': {1: 1118, 5: 1025, 13: 123, 25: 132, 33: 202, 37: 155, 41: 90, 45: 257, 53: 68, 61: 90, 65: 187, 85: 44, 105: 48, 109: 85, 121: 78}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 42, 46, 50, 62, 66, 70, 78, 82, 98, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3497, 'token_per_expert': {2: 1027, 6: 1031, 14: 80, 18: 76, 42: 76, 46: 118, 50: 262, 62: 131, 66: 69, 70: 116, 78: 157, 82: 165, 98: 70, 118: 64, 122: 55}}
INFO 05-06 11:02:15.073226.073226 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.545ms | allocate_experts_across_cpu_gpu: 0.447ms
INFO 05-06 11:02:15.073039.073039 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.508827209472656e-05 seconds
INFO 05-06 11:02:15.075194.075194 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010619163513183594 seconds
INFO 05-06 11:02:15.089948.089948 lmp.py:1387] [layer_moe_fused] to time: 0.0001285076141357422 seconds
INFO 05-06 11:02:15.089882.089882 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.090457.090457 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011069774627685547 seconds
INFO 05-06 11:02:15.091876.091876 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005810260772705078 seconds
INFO 05-06 11:02:15.091726.091726 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001840829849243164 seconds
INFO 05-06 11:02:15.101479.101479 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009391307830810547 seconds
INFO 05-06 11:02:15.103182.103182 mlpmodule.py:2799] [fused_experts] gmm total=1.806ms E=32 S=3886 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.103426.103426 mlpmodule.py:2799] [fused_experts] gmm total=2.052ms E=32 S=3999 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.103711.103711 mlpmodule.py:2799] [fused_experts] gmm total=2.208ms E=32 S=3829 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.103750.103750 mlpmodule.py:2799] [fused_experts] gmm total=2.488ms E=32 S=4670 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.104325.104325 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003414154052734375 seconds
INFO 05-06 11:02:15.104925.104925 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.14984130859375e-05 seconds
DEBUG 05-06 11:02:15.105674.105674 cuda_h.py:27] end *layer_moe_fused cost 32.572 ms
DEBUG 05-06 11:02:15.111902.111902 cuda_h.py:27] end prefill_layer cost 41.783 ms
DEBUG 05-06 11:02:15.111620.111620 lmp.py:841] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 11:02:15.111482.111482 lmp.py:824] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 11:02:15.113203.113203 cuda_h.py:27] end *sagl cost 1.978 ms
experts_cpu_alloc {'expert_ids': [63, 83, 103, 19, 31, 35, 99, 15, 67, 87, 51, 59, 39, 127, 43, 28, 80, 8, 56, 124, 120, 72, 108, 92, 36, 48, 44, 88, 100, 45, 93, 61, 17, 21, 29, 73, 81, 109, 65, 105, 117, 33, 9, 69, 26, 38, 102, 114, 82, 118, 66, 54, 58, 50, 122, 98, 74], 'token_total': 612, 'token_per_expert': {63: 1, 83: 1, 103: 1, 19: 2, 31: 2, 35: 2, 99: 2, 15: 5, 67: 5, 87: 5, 51: 6, 59: 6, 39: 8, 127: 23, 43: 25, 28: 1, 80: 2, 8: 3, 56: 3, 124: 3, 120: 4, 72: 5, 108: 8, 92: 10, 36: 11, 48: 20, 44: 28, 88: 35, 100: 38, 45: 1, 93: 1, 61: 2, 17: 3, 21: 4, 29: 4, 73: 10, 81: 12, 109: 13, 65: 28, 105: 28, 117: 29, 33: 32, 9: 33, 69: 37, 26: 1, 38: 1, 102: 1, 114: 1, 82: 3, 118: 3, 66: 4, 54: 8, 58: 9, 50: 12, 122: 20, 98: 22, 74: 25}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 47, 55, 71, 75, 79, 91, 95, 111, 115, 119, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4246, 'token_per_expert': {3: 1032, 7: 1027, 11: 115, 23: 37, 47: 147, 55: 63, 71: 140, 75: 139, 79: 25, 91: 222, 95: 39, 111: 785, 115: 292, 119: 114, 123: 69}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 32, 40, 52, 60, 68, 76, 84, 104, 112], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 4673, 'token_per_expert': {0: 1024, 4: 1024, 12: 722, 20: 611, 24: 46, 32: 84, 40: 201, 52: 144, 60: 42, 68: 107, 76: 315, 84: 53, 104: 65, 112: 235}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 37, 49, 53, 57, 77, 85, 89, 97, 101, 113, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 3737, 'token_per_expert': {1: 1065, 5: 1079, 13: 80, 37: 59, 49: 577, 53: 144, 57: 275, 77: 104, 85: 50, 89: 64, 97: 45, 101: 76, 113: 77, 121: 42}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 30, 46, 62, 70, 78, 90, 94, 106, 110, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 3116, 'token_per_expert': {2: 1024, 6: 1026, 18: 51, 22: 91, 30: 56, 46: 112, 62: 48, 70: 113, 78: 137, 90: 204, 94: 36, 106: 63, 110: 118, 126: 37}}
INFO 05-06 11:02:15.115517.115517 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.905ms | allocate_experts_across_cpu_gpu: 0.435ms
INFO 05-06 11:02:15.115753.115753 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.079673767089844e-05 seconds
INFO 05-06 11:02:15.117982.117982 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010437965393066406 seconds
INFO 05-06 11:02:15.127412.127412 lmp.py:1387] [layer_moe_fused] to time: 0.0001232624053955078 seconds
INFO 05-06 11:02:15.127385.127385 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.128763.128763 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011365413665771484 seconds
INFO 05-06 11:02:15.129188.129188 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005517005920410156 seconds
INFO 05-06 11:02:15.129614.129614 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0018429756164550781 seconds
INFO 05-06 11:02:15.138267.138267 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00938868522644043 seconds
INFO 05-06 11:02:15.140583.140583 mlpmodule.py:2799] [fused_experts] gmm total=1.910ms E=32 S=3226 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.141762.141762 mlpmodule.py:2799] [fused_experts] gmm total=2.216ms E=32 S=4844 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.141011.141011 mlpmodule.py:2799] [fused_experts] gmm total=2.276ms E=32 S=3974 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.141914.141914 mlpmodule.py:2799] [fused_experts] gmm total=2.457ms E=32 S=4340 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.142783.142783 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0036301612854003906 seconds
INFO 05-06 11:02:15.142906.142906 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.245208740234375e-05 seconds
DEBUG 05-06 11:02:15.142018.142018 cuda_h.py:27] end *layer_moe_fused cost 28.574 ms
DEBUG 05-06 11:02:15.148439.148439 cuda_h.py:27] end prefill_layer cost 37.671 ms
DEBUG 05-06 11:02:15.148395.148395 lmp.py:841] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 11:02:15.148496.148496 lmp.py:824] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 11:02:15.151813.151813 cuda_h.py:27] end *sagl cost 2.629 ms
experts_cpu_alloc {'expert_ids': [79, 39, 111, 127, 55, 87, 75, 63, 119, 31, 35, 83, 11, 72, 12, 112, 68, 100, 40, 88, 8, 76, 24, 84, 96, 32, 108, 120, 17, 37, 65, 125, 33, 13, 25, 105, 21, 89, 9, 73, 69, 101, 77, 118, 38, 122, 126, 70, 74, 34, 98, 102, 46, 94, 50, 58, 114, 10, 66], 'token_total': 1084, 'token_per_expert': {79: 2, 39: 5, 111: 6, 127: 6, 55: 10, 87: 14, 75: 17, 63: 20, 119: 21, 31: 25, 35: 25, 83: 25, 11: 34, 72: 1, 12: 2, 112: 2, 68: 3, 100: 3, 40: 11, 88: 12, 8: 13, 76: 19, 24: 21, 84: 24, 96: 25, 32: 29, 108: 34, 120: 34, 17: 2, 37: 8, 65: 10, 125: 10, 33: 12, 13: 23, 25: 26, 105: 29, 21: 33, 89: 40, 9: 51, 73: 51, 69: 55, 101: 55, 77: 57, 118: 1, 38: 3, 122: 3, 126: 5, 70: 6, 74: 6, 34: 7, 98: 7, 102: 7, 46: 14, 94: 14, 50: 16, 58: 17, 114: 21, 10: 24, 66: 28}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 43, 67, 71, 91, 95, 99, 107, 115, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 4220, 'token_per_expert': {3: 1054, 7: 1409, 15: 53, 19: 216, 23: 128, 27: 142, 43: 208, 67: 63, 71: 118, 91: 347, 95: 64, 99: 253, 107: 57, 115: 45, 123: 63}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 28, 44, 48, 52, 56, 60, 64, 80, 92, 116, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3900, 'token_per_expert': {0: 1028, 4: 1244, 16: 102, 20: 215, 28: 144, 44: 45, 48: 76, 52: 270, 56: 125, 60: 117, 64: 307, 80: 37, 92: 48, 116: 44, 124: 98}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 49, 53, 57, 61, 81, 85, 93, 97, 109, 113, 117, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3391, 'token_per_expert': {1: 1057, 5: 1031, 29: 93, 49: 101, 53: 66, 57: 162, 61: 108, 81: 90, 85: 69, 93: 76, 97: 125, 109: 57, 113: 84, 117: 134, 121: 138}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 30, 42, 54, 62, 78, 82, 86, 90, 106], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3789, 'token_per_expert': {2: 1070, 6: 1045, 14: 79, 18: 142, 22: 147, 26: 149, 30: 80, 42: 185, 54: 88, 62: 113, 78: 41, 82: 73, 86: 216, 90: 108, 106: 253}}
INFO 05-06 11:02:15.156141.156141 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 2.928ms | allocate_experts_across_cpu_gpu: 0.436ms
INFO 05-06 11:02:15.156523.156523 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.29425048828125e-05 seconds
INFO 05-06 11:02:15.157176.157176 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010831356048583984 seconds
INFO 05-06 11:02:15.171697.171697 lmp.py:1387] [layer_moe_fused] to time: 0.0001277923583984375 seconds
INFO 05-06 11:02:15.171545.171545 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.172884.172884 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011816024780273438 seconds
INFO 05-06 11:02:15.173660.173660 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005686283111572266 seconds
INFO 05-06 11:02:15.173172.173172 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001894235610961914 seconds
INFO 05-06 11:02:15.182842.182842 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009471893310546875 seconds
INFO 05-06 11:02:15.185704.185704 mlpmodule.py:2799] [fused_experts] gmm total=2.094ms E=32 S=4133 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.185285.185285 mlpmodule.py:2799] [fused_experts] gmm total=2.145ms E=32 S=3968 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.185440.185440 mlpmodule.py:2799] [fused_experts] gmm total=2.264ms E=32 S=3853 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.185965.185965 mlpmodule.py:2799] [fused_experts] gmm total=2.457ms E=32 S=4430 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.186243.186243 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0037920475006103516 seconds
INFO 05-06 11:02:15.186453.186453 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.030632019042969e-05 seconds
DEBUG 05-06 11:02:15.187031.187031 cuda_h.py:27] end *layer_moe_fused cost 34.500 ms
DEBUG 05-06 11:02:15.193158.193158 cuda_h.py:27] end prefill_layer cost 44.475 ms
DEBUG 05-06 11:02:15.193736.193736 lmp.py:841] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 11:02:15.193075.193075 cuda_h.py:27] end prefill_step cost 2969.112 ms
INFO 05-06 11:02:15.193471.193471 lmp.py:843] prefill time: 3.074775457382202 seconds
WARNING 05-06 11:02:15.239624.239624 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:02:15.239602.239602 helper.py:35]   NaN count (hidden): 2883584
WARNING 05-06 11:02:15.240823.240823 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:02:15.241893.241893 helper.py:39]   NaN count (normed): 2883584
WARNING 05-06 11:02:15.246170.246170 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:02:15.246718.246718 helper.py:50]   NaN count: 524288
WARNING 05-06 11:02:15.246230.246230 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:02:15.330831.330831 cuda_h.py:27] end init_inputs_tokens cost 104.755 ms
DEBUG 05-06 11:02:15.330861.330861 lmp.py:899] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:02:15.330837.330837 lmp.py:904] ---- decode step 0 layer 0 ----
DEBUG 05-06 11:02:15.336864.336864 cuda_h.py:27] end *sagl cost 6.503 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 23, 47, 55, 63, 79, 83, 87, 91, 103, 123, 127], 'expert_count': 12, 'ideal_gpu_count': 6, 'keep_on_gpu': 12, 'hit_count_on_device': 12, 'token_total': 17, 'token_per_expert': {15: 2, 23: 1, 47: 1, 55: 1, 63: 1, 79: 2, 83: 2, 87: 2, 91: 1, 103: 1, 123: 1, 127: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 48, 60, 64, 116], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {8: 2, 48: 1, 60: 2, 64: 1, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 33, 45], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {13: 1, 33: 1, 45: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 50, 90], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {22: 2, 26: 1, 50: 1, 90: 1}}
INFO 05-06 11:02:15.338740.338740 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.523ms | allocate_experts_across_cpu_gpu: 0.128ms
INFO 05-06 11:02:15.338445.338445 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 11:02:15.338056.338056 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.340212.340212 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016651153564453125 seconds
INFO 05-06 11:02:15.341496.341496 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012996196746826172 seconds
INFO 05-06 11:02:15.341948.341948 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.003092527389526367 seconds
INFO 05-06 11:02:15.343367.343367 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0019958019256591797 seconds
INFO 05-06 11:02:15.345248.345248 mlpmodule.py:2799] [fused_experts] gmm total=1.275ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.346425.346425 mlpmodule.py:2799] [fused_experts] gmm total=2.088ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.346809.346809 mlpmodule.py:2799] [fused_experts] gmm total=2.163ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.346231.346231 mlpmodule.py:2799] [fused_experts] gmm total=2.795ms E=32 S=17 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.347970.347970 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003887176513671875 seconds
INFO 05-06 11:02:15.347364.347364 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.76837158203125e-05 seconds
DEBUG 05-06 11:02:15.348323.348323 cuda_h.py:27] end *layer_moe_fused cost 10.356 ms
DEBUG 05-06 11:02:15.348636.348636 cuda_h.py:27] end decode_layer cost 18.526 ms
DEBUG 05-06 11:02:15.348810.348810 lmp.py:904] ---- decode step 0 layer 1 ----
DEBUG 05-06 11:02:15.350591.350591 cuda_h.py:27] end *sagl cost 1.729 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [83, 107, 119, 123], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {83: 1, 107: 1, 119: 2, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 56, 92, 96, 116, 124], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 13, 'token_per_expert': {0: 3, 8: 1, 56: 2, 92: 2, 96: 2, 116: 1, 124: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 73, 121], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {9: 2, 73: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [26, 30, 34, 46, 54, 110], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {26: 1, 30: 2, 34: 1, 46: 1, 54: 2, 110: 2}}
INFO 05-06 11:02:15.352755.352755 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.328ms | allocate_experts_across_cpu_gpu: 0.102ms
INFO 05-06 11:02:15.352870.352870 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 11:02:15.352911.352911 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.353483.353483 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012362003326416016 seconds
INFO 05-06 11:02:15.354043.354043 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014486312866210938 seconds
INFO 05-06 11:02:15.355217.355217 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002799510955810547 seconds
INFO 05-06 11:02:15.356260.356260 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012235641479492188 seconds
INFO 05-06 11:02:15.358694.358694 mlpmodule.py:2799] [fused_experts] gmm total=1.693ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.358839.358839 mlpmodule.py:2799] [fused_experts] gmm total=2.185ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.359600.359600 mlpmodule.py:2799] [fused_experts] gmm total=2.435ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.359892.359892 mlpmodule.py:2799] [fused_experts] gmm total=2.749ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.360925.360925 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003968477249145508 seconds
INFO 05-06 11:02:15.360789.360789 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.601478576660156e-05 seconds
DEBUG 05-06 11:02:15.360291.360291 cuda_h.py:27] end *layer_moe_fused cost 8.940 ms
DEBUG 05-06 11:02:15.361218.361218 cuda_h.py:27] end decode_layer cost 12.375 ms
DEBUG 05-06 11:02:15.361723.361723 lmp.py:904] ---- decode step 0 layer 2 ----
DEBUG 05-06 11:02:15.362283.362283 cuda_h.py:27] end *sagl cost 1.463 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 19, 91, 127], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {11: 2, 19: 1, 91: 2, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 24, 52, 76, 100, 108], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {8: 1, 12: 1, 24: 1, 52: 1, 76: 3, 100: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 41, 45, 49, 81], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {1: 1, 41: 1, 45: 1, 49: 2, 81: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [34, 62, 70, 86, 90, 102, 106, 126], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 9, 'token_per_expert': {34: 1, 62: 1, 70: 1, 86: 1, 90: 1, 102: 1, 106: 1, 126: 2}}
INFO 05-06 11:02:15.364542.364542 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.852ms | allocate_experts_across_cpu_gpu: 0.106ms
INFO 05-06 11:02:15.364990.364990 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.09808349609375e-05 seconds
INFO 05-06 11:02:15.364508.364508 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.366758.366758 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013148784637451172 seconds
INFO 05-06 11:02:15.367399.367399 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010862350463867188 seconds
INFO 05-06 11:02:15.367036.367036 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002506732940673828 seconds
INFO 05-06 11:02:15.368356.368356 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013959407806396484 seconds
INFO 05-06 11:02:15.371774.371774 mlpmodule.py:2799] [fused_experts] gmm total=2.162ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.371836.371836 mlpmodule.py:2799] [fused_experts] gmm total=2.211ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.371989.371989 mlpmodule.py:2799] [fused_experts] gmm total=2.524ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.371941.371941 mlpmodule.py:2799] [fused_experts] gmm total=2.487ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.372044.372044 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004269599914550781 seconds
INFO 05-06 11:02:15.373869.373869 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.054473876953125e-05 seconds
DEBUG 05-06 11:02:15.373994.373994 cuda_h.py:27] end *layer_moe_fused cost 9.904 ms
DEBUG 05-06 11:02:15.373692.373692 cuda_h.py:27] end decode_layer cost 12.718 ms
DEBUG 05-06 11:02:15.373482.373482 lmp.py:904] ---- decode step 0 layer 3 ----
DEBUG 05-06 11:02:15.375935.375935 cuda_h.py:27] end *sagl cost 1.804 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 39, 107], 'expert_count': 3, 'ideal_gpu_count': 7, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {3: 1, 39: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [24, 32, 40, 96, 104], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {24: 1, 32: 1, 40: 1, 96: 3, 104: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 33, 41, 73, 85, 101, 117, 125], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {5: 1, 33: 1, 41: 1, 73: 2, 85: 1, 101: 2, 117: 2, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 42, 50, 54, 62, 110, 118, 126], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 10, 'token_per_expert': {22: 1, 26: 1, 42: 1, 50: 2, 54: 1, 62: 1, 110: 1, 118: 1, 126: 1}}
INFO 05-06 11:02:15.377044.377044 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.336ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 11:02:15.377537.377537 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 11:02:15.377147.377147 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.378210.378210 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014553070068359375 seconds
INFO 05-06 11:02:15.380145.380145 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011553764343261719 seconds
INFO 05-06 11:02:15.380689.380689 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0027272701263427734 seconds
INFO 05-06 11:02:15.381853.381853 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001279592514038086 seconds
INFO 05-06 11:02:15.383976.383976 mlpmodule.py:2799] [fused_experts] gmm total=1.869ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.383384.383384 mlpmodule.py:2799] [fused_experts] gmm total=2.109ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.384656.384656 mlpmodule.py:2799] [fused_experts] gmm total=2.441ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.384283.384283 mlpmodule.py:2799] [fused_experts] gmm total=2.684ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.385207.385207 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003988504409790039 seconds
INFO 05-06 11:02:15.385787.385787 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.601478576660156e-05 seconds
DEBUG 05-06 11:02:15.385480.385480 cuda_h.py:27] end *layer_moe_fused cost 9.119 ms
DEBUG 05-06 11:02:15.386065.386065 cuda_h.py:27] end decode_layer cost 12.468 ms
DEBUG 05-06 11:02:15.386286.386286 lmp.py:904] ---- decode step 0 layer 4 ----
DEBUG 05-06 11:02:15.388688.388688 cuda_h.py:27] end *sagl cost 1.486 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 51, 67, 83, 87], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {3: 2, 51: 1, 67: 2, 83: 2, 87: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 60], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {20: 2, 60: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [17, 25, 45, 49, 85, 93, 101, 121], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 9, 'token_per_expert': {17: 1, 25: 1, 45: 2, 49: 1, 85: 1, 93: 1, 101: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 50, 78, 90, 106, 114, 122, 126], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 50: 2, 78: 1, 90: 1, 106: 1, 114: 1, 122: 2, 126: 2}}
INFO 05-06 11:02:15.389603.389603 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.303ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 11:02:15.389434.389434 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0742416381835938e-05 seconds
INFO 05-06 11:02:15.389713.389713 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.390803.390803 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012631416320800781 seconds
INFO 05-06 11:02:15.391464.391464 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008904933929443359 seconds
INFO 05-06 11:02:15.391432.391432 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0022573471069335938 seconds
INFO 05-06 11:02:15.392726.392726 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012378692626953125 seconds
INFO 05-06 11:02:15.395511.395511 mlpmodule.py:2799] [fused_experts] gmm total=1.955ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.395397.395397 mlpmodule.py:2799] [fused_experts] gmm total=2.198ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.395110.395110 mlpmodule.py:2799] [fused_experts] gmm total=2.159ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.395346.395346 mlpmodule.py:2799] [fused_experts] gmm total=2.468ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.397558.397558 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003989458084106445 seconds
INFO 05-06 11:02:15.397422.397422 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.601478576660156e-05 seconds
DEBUG 05-06 11:02:15.397648.397648 cuda_h.py:27] end *layer_moe_fused cost 8.651 ms
DEBUG 05-06 11:02:15.397617.397617 cuda_h.py:27] end decode_layer cost 11.475 ms
DEBUG 05-06 11:02:15.398930.398930 lmp.py:904] ---- decode step 0 layer 5 ----
DEBUG 05-06 11:02:15.399425.399425 cuda_h.py:27] end *sagl cost 1.695 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [39, 71, 95, 99, 123], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 10, 'token_per_expert': {39: 2, 71: 2, 95: 2, 99: 2, 123: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [36, 52, 60, 72, 116], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {36: 1, 52: 2, 60: 1, 72: 2, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 29, 33, 57, 61, 65], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {5: 2, 29: 1, 33: 1, 57: 1, 61: 1, 65: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 46, 70, 74, 94, 118], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 1, 46: 1, 70: 2, 74: 1, 94: 2, 118: 1}}
INFO 05-06 11:02:15.401846.401846 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.308ms | allocate_experts_across_cpu_gpu: 0.092ms
INFO 05-06 11:02:15.401240.401240 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 11:02:15.401042.401042 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.402891.402891 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014042854309082031 seconds
INFO 05-06 11:02:15.403488.403488 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009407997131347656 seconds
INFO 05-06 11:02:15.403271.403271 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0024607181549072266 seconds
INFO 05-06 11:02:15.404843.404843 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012309551239013672 seconds
INFO 05-06 11:02:15.407259.407259 mlpmodule.py:2799] [fused_experts] gmm total=2.109ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.407551.407551 mlpmodule.py:2799] [fused_experts] gmm total=2.225ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.407744.407744 mlpmodule.py:2799] [fused_experts] gmm total=2.359ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.407979.407979 mlpmodule.py:2799] [fused_experts] gmm total=2.392ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.409897.409897 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0041196346282958984 seconds
INFO 05-06 11:02:15.409669.409669 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.6253204345703125e-05 seconds
DEBUG 05-06 11:02:15.409557.409557 cuda_h.py:27] end *layer_moe_fused cost 8.968 ms
DEBUG 05-06 11:02:15.410697.410697 cuda_h.py:27] end decode_layer cost 12.006 ms
DEBUG 05-06 11:02:15.410248.410248 lmp.py:904] ---- decode step 0 layer 6 ----
DEBUG 05-06 11:02:15.411364.411364 cuda_h.py:27] end *sagl cost 1.444 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [35, 43, 87, 99, 103, 115], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {35: 1, 43: 1, 87: 3, 99: 1, 103: 1, 115: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 24, 32, 36, 68, 96, 108], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 7, 'token_per_expert': {4: 1, 24: 1, 32: 1, 36: 1, 68: 1, 96: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 25], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {1: 1, 9: 1, 13: 2, 25: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 70, 78, 90, 106, 118], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 1, 70: 1, 78: 1, 90: 1, 106: 2, 118: 1}}
INFO 05-06 11:02:15.412491.412491 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.303ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 11:02:15.412123.412123 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 11:02:15.412687.412687 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.414806.414806 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013580322265625 seconds
INFO 05-06 11:02:15.415372.415372 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014183521270751953 seconds
INFO 05-06 11:02:15.415354.415354 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002889394760131836 seconds
INFO 05-06 11:02:15.417185.417185 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012438297271728516 seconds
INFO 05-06 11:02:15.419434.419434 mlpmodule.py:2799] [fused_experts] gmm total=2.064ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.419652.419652 mlpmodule.py:2799] [fused_experts] gmm total=2.152ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.419788.419788 mlpmodule.py:2799] [fused_experts] gmm total=2.161ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.419226.419226 mlpmodule.py:2799] [fused_experts] gmm total=2.305ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.421647.421647 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004032135009765625 seconds
INFO 05-06 11:02:15.421703.421703 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.6253204345703125e-05 seconds
DEBUG 05-06 11:02:15.421861.421861 cuda_h.py:27] end *layer_moe_fused cost 9.275 ms
DEBUG 05-06 11:02:15.422934.422934 cuda_h.py:27] end decode_layer cost 11.983 ms
DEBUG 05-06 11:02:15.422201.422201 lmp.py:904] ---- decode step 0 layer 7 ----
DEBUG 05-06 11:02:15.423702.423702 cuda_h.py:27] end *sagl cost 1.664 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 19, 43], 'expert_count': 3, 'ideal_gpu_count': 7, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {11: 1, 19: 1, 43: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 64, 68, 80, 96, 104, 112], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {20: 2, 64: 1, 68: 1, 80: 1, 96: 1, 104: 2, 112: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 29, 65, 69, 97, 105, 113, 121, 125], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {9: 2, 29: 1, 65: 1, 69: 1, 97: 2, 105: 1, 113: 1, 121: 2, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 14, 18, 34, 90, 114], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {10: 2, 14: 1, 18: 1, 34: 1, 90: 1, 114: 2}}
INFO 05-06 11:02:15.425696.425696 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.310ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 11:02:15.425142.425142 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 11:02:15.425706.425706 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.426744.426744 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001299142837524414 seconds
INFO 05-06 11:02:15.427714.427714 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010149478912353516 seconds
INFO 05-06 11:02:15.427397.427397 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0024166107177734375 seconds
INFO 05-06 11:02:15.429950.429950 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001253366470336914 seconds
INFO 05-06 11:02:15.431076.431076 mlpmodule.py:2799] [fused_experts] gmm total=1.941ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.431889.431889 mlpmodule.py:2799] [fused_experts] gmm total=2.164ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.431908.431908 mlpmodule.py:2799] [fused_experts] gmm total=2.195ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.432447.432447 mlpmodule.py:2799] [fused_experts] gmm total=2.532ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.433468.433468 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003935575485229492 seconds
INFO 05-06 11:02:15.433287.433287 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.6253204345703125e-05 seconds
DEBUG 05-06 11:02:15.433710.433710 cuda_h.py:27] end *layer_moe_fused cost 8.532 ms
DEBUG 05-06 11:02:15.433504.433504 cuda_h.py:27] end decode_layer cost 11.805 ms
DEBUG 05-06 11:02:15.434532.434532 lmp.py:904] ---- decode step 0 layer 8 ----
DEBUG 05-06 11:02:15.435099.435099 cuda_h.py:27] end *sagl cost 1.468 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 51, 55, 63, 75, 103, 127], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {11: 1, 51: 3, 55: 1, 63: 2, 75: 1, 103: 2, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [12, 24, 32, 64, 88], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {12: 1, 24: 1, 32: 1, 64: 2, 88: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 61, 69, 93, 105], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {5: 1, 61: 1, 69: 1, 93: 1, 105: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 42, 46, 50, 54, 110, 114], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {6: 1, 42: 1, 46: 2, 50: 2, 54: 2, 110: 1, 114: 1}}
INFO 05-06 11:02:15.436253.436253 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.303ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 11:02:15.436038.436038 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 11:02:15.436078.436078 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.437796.437796 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010271072387695312 seconds
INFO 05-06 11:02:15.438979.438979 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008547306060791016 seconds
INFO 05-06 11:02:15.438424.438424 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019855499267578125 seconds
INFO 05-06 11:02:15.440169.440169 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012538433074951172 seconds
INFO 05-06 11:02:15.442423.442423 mlpmodule.py:2799] [fused_experts] gmm total=2.196ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.442906.442906 mlpmodule.py:2799] [fused_experts] gmm total=2.279ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.442308.442308 mlpmodule.py:2799] [fused_experts] gmm total=2.354ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.443505.443505 mlpmodule.py:2799] [fused_experts] gmm total=2.401ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.444127.444127 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004226207733154297 seconds
INFO 05-06 11:02:15.444395.444395 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.00543212890625e-05 seconds
DEBUG 05-06 11:02:15.445285.445285 cuda_h.py:27] end *layer_moe_fused cost 8.678 ms
DEBUG 05-06 11:02:15.445751.445751 cuda_h.py:27] end decode_layer cost 11.463 ms
DEBUG 05-06 11:02:15.445110.445110 lmp.py:904] ---- decode step 0 layer 9 ----
DEBUG 05-06 11:02:15.447783.447783 cuda_h.py:27] end *sagl cost 1.477 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 51, 83, 95], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 1, 7: 1, 15: 1, 19: 1, 51: 1, 83: 1, 95: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [36, 48, 68, 76, 92], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {36: 2, 48: 1, 68: 1, 76: 2, 92: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 21, 37, 69, 81, 89, 101], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {13: 1, 21: 1, 37: 1, 69: 2, 81: 2, 89: 2, 101: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [30, 54, 74], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {30: 2, 54: 1, 74: 1}}
INFO 05-06 11:02:15.448580.448580 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.300ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 11:02:15.448643.448643 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 11:02:15.448160.448160 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.449095.449095 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008015632629394531 seconds
INFO 05-06 11:02:15.449502.449502 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006351470947265625 seconds
INFO 05-06 11:02:15.449425.449425 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001539468765258789 seconds
INFO 05-06 11:02:15.451016.451016 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001210927963256836 seconds
INFO 05-06 11:02:15.453258.453258 mlpmodule.py:2799] [fused_experts] gmm total=1.712ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.453909.453909 mlpmodule.py:2799] [fused_experts] gmm total=1.869ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.453472.453472 mlpmodule.py:2799] [fused_experts] gmm total=2.323ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.453793.453793 mlpmodule.py:2799] [fused_experts] gmm total=2.339ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.454817.454817 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003686189651489258 seconds
INFO 05-06 11:02:15.455920.455920 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.649162292480469e-05 seconds
DEBUG 05-06 11:02:15.455521.455521 cuda_h.py:27] end *layer_moe_fused cost 7.515 ms
DEBUG 05-06 11:02:15.455316.455316 cuda_h.py:27] end decode_layer cost 10.294 ms
DEBUG 05-06 11:02:15.455629.455629 lmp.py:904] ---- decode step 0 layer 10 ----
DEBUG 05-06 11:02:15.457916.457916 cuda_h.py:27] end *sagl cost 1.403 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [71, 75, 79], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {71: 1, 75: 1, 79: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 16, 28, 44, 60, 92], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {8: 3, 16: 1, 28: 1, 44: 1, 60: 1, 92: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [21, 37, 57, 81, 97, 105, 113], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {21: 1, 37: 1, 57: 1, 81: 2, 97: 3, 105: 3, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 46, 54, 58, 62, 126], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {18: 3, 46: 1, 54: 2, 58: 1, 62: 1, 126: 1}}
INFO 05-06 11:02:15.458292.458292 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.286ms | allocate_experts_across_cpu_gpu: 0.088ms
INFO 05-06 11:02:15.458540.458540 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 11:02:15.458673.458673 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.459912.459912 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011529922485351562 seconds
INFO 05-06 11:02:15.460692.460692 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008413791656494141 seconds
INFO 05-06 11:02:15.460191.460191 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0021839141845703125 seconds
INFO 05-06 11:02:15.462796.462796 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012204647064208984 seconds
INFO 05-06 11:02:15.464160.464160 mlpmodule.py:2799] [fused_experts] gmm total=1.782ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.464423.464423 mlpmodule.py:2799] [fused_experts] gmm total=2.013ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.464572.464572 mlpmodule.py:2799] [fused_experts] gmm total=2.413ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.465893.465893 mlpmodule.py:2799] [fused_experts] gmm total=2.594ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.466663.466663 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003875732421875 seconds
INFO 05-06 11:02:15.466912.466912 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.673004150390625e-05 seconds
DEBUG 05-06 11:02:15.466368.466368 cuda_h.py:27] end *layer_moe_fused cost 8.367 ms
DEBUG 05-06 11:02:15.466898.466898 cuda_h.py:27] end decode_layer cost 11.008 ms
DEBUG 05-06 11:02:15.466927.466927 lmp.py:904] ---- decode step 0 layer 11 ----
DEBUG 05-06 11:02:15.468891.468891 cuda_h.py:27] end *sagl cost 1.654 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [23, 43, 67, 79, 83, 87, 99, 119], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 13, 'token_per_expert': {23: 1, 43: 1, 67: 1, 79: 2, 83: 4, 87: 1, 99: 2, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [24, 116, 124], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {24: 1, 116: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 49, 81, 93, 113], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {9: 1, 49: 1, 81: 2, 93: 1, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 38, 46, 50, 102, 114], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {2: 2, 38: 2, 46: 2, 50: 1, 102: 2, 114: 1}}
INFO 05-06 11:02:15.469477.469477 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.296ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 11:02:15.470446.470446 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 11:02:15.470725.470725 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.471732.471732 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011563301086425781 seconds
INFO 05-06 11:02:15.473288.473288 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0024826526641845703 seconds
INFO 05-06 11:02:15.473922.473922 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0038514137268066406 seconds
INFO 05-06 11:02:15.477187.477187 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0034618377685546875 seconds
INFO 05-06 11:02:15.479452.479452 mlpmodule.py:2799] [fused_experts] gmm total=1.903ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.480323.480323 mlpmodule.py:2799] [fused_experts] gmm total=2.290ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.480521.480521 mlpmodule.py:2799] [fused_experts] gmm total=2.274ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.480333.480333 mlpmodule.py:2799] [fused_experts] gmm total=2.287ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.481517.481517 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004151344299316406 seconds
INFO 05-06 11:02:15.482507.482507 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 0.00011229515075683594 seconds
DEBUG 05-06 11:02:15.482048.482048 cuda_h.py:27] end *layer_moe_fused cost 13.089 ms
DEBUG 05-06 11:02:15.483878.483878 cuda_h.py:27] end decode_layer cost 16.727 ms
DEBUG 05-06 11:02:15.483598.483598 lmp.py:904] ---- decode step 0 layer 12 ----
DEBUG 05-06 11:02:15.488778.488778 cuda_h.py:27] end *sagl cost 4.025 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 19, 39], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {15: 1, 19: 1, 39: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [36, 76], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 2, 'token_per_expert': {36: 1, 76: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [21, 45, 49, 73, 97, 117], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {21: 1, 45: 1, 49: 1, 73: 1, 97: 2, 117: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [46, 50, 74, 78, 86, 98, 106, 114, 118], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 18, 'token_per_expert': {46: 2, 50: 1, 74: 2, 78: 4, 86: 3, 98: 1, 106: 2, 114: 2, 118: 1}}
INFO 05-06 11:02:15.490996.490996 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.573ms | allocate_experts_across_cpu_gpu: 0.192ms
INFO 05-06 11:02:15.490285.490285 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.0040740966796875e-05 seconds
INFO 05-06 11:02:15.490751.490751 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.492652.492652 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017170906066894531 seconds
INFO 05-06 11:02:15.494583.494583 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015900135040283203 seconds
INFO 05-06 11:02:15.494487.494487 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.003535032272338867 seconds
INFO 05-06 11:02:15.496317.496317 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0023658275604248047 seconds
INFO 05-06 11:02:15.499043.499043 mlpmodule.py:2799] [fused_experts] gmm total=1.893ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.499234.499234 mlpmodule.py:2799] [fused_experts] gmm total=2.167ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.499579.499579 mlpmodule.py:2799] [fused_experts] gmm total=2.189ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.499630.499630 mlpmodule.py:2799] [fused_experts] gmm total=2.234ms E=32 S=18 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.500566.500566 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003968715667724609 seconds
INFO 05-06 11:02:15.501511.501511 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.555152893066406e-05 seconds
DEBUG 05-06 11:02:15.501134.501134 cuda_h.py:27] end *layer_moe_fused cost 11.446 ms
DEBUG 05-06 11:02:15.501430.501430 cuda_h.py:27] end decode_layer cost 17.926 ms
DEBUG 05-06 11:02:15.501472.501472 lmp.py:904] ---- decode step 0 layer 13 ----
DEBUG 05-06 11:02:15.503077.503077 cuda_h.py:27] end *sagl cost 1.803 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 47, 59, 71, 79, 107], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {3: 1, 47: 1, 59: 1, 71: 2, 79: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [32, 80, 100, 104], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {32: 1, 80: 1, 100: 3, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 17, 33, 45, 125], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {1: 1, 17: 1, 33: 1, 45: 1, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 14, 22, 26, 38, 78, 110, 114, 126], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 13, 'token_per_expert': {2: 1, 14: 1, 22: 1, 26: 1, 38: 1, 78: 2, 110: 2, 114: 3, 126: 1}}
INFO 05-06 11:02:15.505087.505087 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.360ms | allocate_experts_across_cpu_gpu: 0.116ms
INFO 05-06 11:02:15.505408.505408 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.288818359375e-05 seconds
INFO 05-06 11:02:15.505171.505171 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.506355.506355 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015058517456054688 seconds
INFO 05-06 11:02:15.508775.508775 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001405954360961914 seconds
INFO 05-06 11:02:15.508095.508095 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0030395984649658203 seconds
INFO 05-06 11:02:15.509142.509142 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00136566162109375 seconds
INFO 05-06 11:02:15.511730.511730 mlpmodule.py:2799] [fused_experts] gmm total=1.859ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.512426.512426 mlpmodule.py:2799] [fused_experts] gmm total=2.161ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.512750.512750 mlpmodule.py:2799] [fused_experts] gmm total=2.169ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.512549.512549 mlpmodule.py:2799] [fused_experts] gmm total=2.221ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.513340.513340 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003869295120239258 seconds
INFO 05-06 11:02:15.513834.513834 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 11:02:15.514703.514703 cuda_h.py:27] end *layer_moe_fused cost 9.573 ms
DEBUG 05-06 11:02:15.514418.514418 cuda_h.py:27] end decode_layer cost 12.806 ms
DEBUG 05-06 11:02:15.514115.514115 lmp.py:904] ---- decode step 0 layer 14 ----
DEBUG 05-06 11:02:15.516458.516458 cuda_h.py:27] end *sagl cost 1.513 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 31, 39, 47, 75, 99, 115], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {11: 1, 31: 1, 39: 1, 47: 2, 75: 1, 99: 1, 115: 4}}
experts_gpu_alloc_device_1 {'expert_ids': [56, 100, 108, 112], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {56: 1, 100: 1, 108: 1, 112: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 25, 57, 81, 85, 89, 97, 109, 113, 121], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 13, 'token_per_expert': {9: 1, 25: 2, 57: 1, 81: 1, 85: 1, 89: 1, 97: 1, 109: 1, 113: 1, 121: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 26], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {2: 3, 26: 1}}
INFO 05-06 11:02:15.517202.517202 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.310ms | allocate_experts_across_cpu_gpu: 0.099ms
INFO 05-06 11:02:15.517457.517457 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.002716064453125e-05 seconds
INFO 05-06 11:02:15.517974.517974 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.519122.519122 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014126300811767578 seconds
INFO 05-06 11:02:15.520669.520669 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012385845184326172 seconds
INFO 05-06 11:02:15.520376.520376 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002858877182006836 seconds
INFO 05-06 11:02:15.522566.522566 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014269351959228516 seconds
INFO 05-06 11:02:15.524191.524191 mlpmodule.py:2799] [fused_experts] gmm total=1.918ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.524549.524549 mlpmodule.py:2799] [fused_experts] gmm total=1.952ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.524020.524020 mlpmodule.py:2799] [fused_experts] gmm total=2.457ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.524646.524646 mlpmodule.py:2799] [fused_experts] gmm total=2.363ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.526203.526203 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003950595855712891 seconds
INFO 05-06 11:02:15.526459.526459 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9591064453125e-05 seconds
DEBUG 05-06 11:02:15.526469.526469 cuda_h.py:27] end *layer_moe_fused cost 9.563 ms
DEBUG 05-06 11:02:15.527955.527955 cuda_h.py:27] end decode_layer cost 12.411 ms
DEBUG 05-06 11:02:15.527858.527858 lmp.py:904] ---- decode step 0 layer 15 ----
DEBUG 05-06 11:02:15.528705.528705 cuda_h.py:27] end *sagl cost 1.524 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 75, 83, 91, 99, 119], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {7: 1, 75: 1, 83: 2, 91: 1, 99: 1, 119: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [24, 68, 72, 108, 112], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {24: 1, 68: 1, 72: 1, 108: 2, 112: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [33, 65, 69, 77, 81, 93, 101, 105], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {33: 1, 65: 1, 69: 2, 77: 1, 81: 2, 93: 1, 101: 1, 105: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 30, 34, 54, 110, 118], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {10: 1, 30: 2, 34: 1, 54: 1, 110: 1, 118: 1}}
INFO 05-06 11:02:15.530852.530852 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.303ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 11:02:15.530968.530968 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 11:02:15.530008.530008 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.531010.531010 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014100074768066406 seconds
INFO 05-06 11:02:15.532949.532949 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012514591217041016 seconds
INFO 05-06 11:02:15.532878.532878 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0027925968170166016 seconds
INFO 05-06 11:02:15.534478.534478 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012865066528320312 seconds
INFO 05-06 11:02:15.536592.536592 mlpmodule.py:2799] [fused_experts] gmm total=2.168ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.536430.536430 mlpmodule.py:2799] [fused_experts] gmm total=2.212ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.537151.537151 mlpmodule.py:2799] [fused_experts] gmm total=2.499ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.538854.538854 mlpmodule.py:2799] [fused_experts] gmm total=3.798ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.539732.539732 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004754781723022461 seconds
INFO 05-06 11:02:15.539948.539948 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.555152893066406e-05 seconds
DEBUG 05-06 11:02:15.539658.539658 cuda_h.py:27] end *layer_moe_fused cost 10.020 ms
DEBUG 05-06 11:02:15.540203.540203 cuda_h.py:27] end decode_layer cost 12.900 ms
DEBUG 05-06 11:02:15.540570.540570 lmp.py:904] ---- decode step 0 layer 16 ----
DEBUG 05-06 11:02:15.541139.541139 cuda_h.py:27] end *sagl cost 1.539 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 19, 63, 71, 87, 107], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {7: 1, 15: 1, 19: 1, 63: 1, 71: 1, 87: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 20, 32, 44, 52], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {4: 1, 20: 1, 32: 2, 44: 2, 52: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 61, 85, 105, 113], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 1, 5: 2, 29: 1, 61: 1, 85: 2, 105: 2, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [54, 62, 66, 78, 102], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {54: 2, 62: 1, 66: 1, 78: 2, 102: 1}}
INFO 05-06 11:02:15.543414.543414 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.319ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 11:02:15.543198.543198 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3603439331054688e-05 seconds
INFO 05-06 11:02:15.543954.543954 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.544030.544030 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001428365707397461 seconds
INFO 05-06 11:02:15.546841.546841 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014073848724365234 seconds
INFO 05-06 11:02:15.546445.546445 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002969503402709961 seconds
INFO 05-06 11:02:15.547014.547014 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013301372528076172 seconds
INFO 05-06 11:02:15.549393.549393 mlpmodule.py:2799] [fused_experts] gmm total=2.094ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.550677.550677 mlpmodule.py:2799] [fused_experts] gmm total=2.364ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.550414.550414 mlpmodule.py:2799] [fused_experts] gmm total=2.316ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.550934.550934 mlpmodule.py:2799] [fused_experts] gmm total=2.498ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.551048.551048 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004312276840209961 seconds
INFO 05-06 11:02:15.552112.552112 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 11:02:15.552527.552527 cuda_h.py:27] end *layer_moe_fused cost 9.911 ms
DEBUG 05-06 11:02:15.552679.552679 cuda_h.py:27] end decode_layer cost 12.751 ms
DEBUG 05-06 11:02:15.552615.552615 lmp.py:904] ---- decode step 0 layer 17 ----
DEBUG 05-06 11:02:15.554177.554177 cuda_h.py:27] end *sagl cost 1.533 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 43, 47, 63, 83], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 1, 7: 2, 23: 2, 43: 1, 47: 1, 63: 1, 83: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [16, 28, 68, 96, 108], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {16: 1, 28: 1, 68: 1, 96: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 33, 53, 73, 81, 89, 93, 113], 'expert_count': 10, 'ideal_gpu_count': 7, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 11, 'token_per_expert': {5: 1, 9: 1, 13: 1, 33: 1, 53: 1, 73: 2, 81: 1, 89: 1, 93: 1, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 22, 34, 70, 102, 106], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {18: 1, 22: 2, 34: 1, 70: 1, 102: 1, 106: 1}}
INFO 05-06 11:02:15.555526.555526 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.321ms | allocate_experts_across_cpu_gpu: 0.105ms
INFO 05-06 11:02:15.555694.555694 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.574920654296875e-05 seconds
INFO 05-06 11:02:15.555689.555689 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.557179.557179 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013508796691894531 seconds
INFO 05-06 11:02:15.558678.558678 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014014244079589844 seconds
INFO 05-06 11:02:15.558282.558282 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002870321273803711 seconds
INFO 05-06 11:02:15.560620.560620 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013341903686523438 seconds
INFO 05-06 11:02:15.562254.562254 mlpmodule.py:2799] [fused_experts] gmm total=2.121ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.562584.562584 mlpmodule.py:2799] [fused_experts] gmm total=2.215ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.562596.562596 mlpmodule.py:2799] [fused_experts] gmm total=2.188ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.563265.563265 mlpmodule.py:2799] [fused_experts] gmm total=2.494ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.564159.564159 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003981351852416992 seconds
INFO 05-06 11:02:15.564230.564230 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9591064453125e-05 seconds
DEBUG 05-06 11:02:15.564871.564871 cuda_h.py:27] end *layer_moe_fused cost 9.356 ms
DEBUG 05-06 11:02:15.565071.565071 cuda_h.py:27] end decode_layer cost 12.234 ms
DEBUG 05-06 11:02:15.565768.565768 lmp.py:904] ---- decode step 0 layer 18 ----
DEBUG 05-06 11:02:15.566437.566437 cuda_h.py:27] end *sagl cost 1.533 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 23, 55, 59, 75, 83, 111, 127], 'expert_count': 8, 'ideal_gpu_count': 8, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 8, 'token_per_expert': {11: 1, 23: 1, 55: 1, 59: 1, 75: 1, 83: 1, 111: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 36, 40, 44, 72, 80, 84, 104], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 9, 'token_per_expert': {4: 1, 36: 1, 40: 1, 44: 1, 72: 1, 80: 1, 84: 1, 104: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [25, 37, 73, 77, 97, 105, 121], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {25: 1, 37: 2, 73: 1, 77: 1, 97: 1, 105: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 26, 30, 42, 50, 58], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {10: 1, 26: 1, 30: 1, 42: 1, 50: 1, 58: 2}}
INFO 05-06 11:02:15.568196.568196 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.311ms | allocate_experts_across_cpu_gpu: 0.104ms
INFO 05-06 11:02:15.568364.568364 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6464462280273438e-05 seconds
INFO 05-06 11:02:15.568597.568597 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.569352.569352 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001371145248413086 seconds
INFO 05-06 11:02:15.571604.571604 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013265609741210938 seconds
INFO 05-06 11:02:15.571764.571764 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002802610397338867 seconds
INFO 05-06 11:02:15.572321.572321 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013599395751953125 seconds
INFO 05-06 11:02:15.574209.574209 mlpmodule.py:2799] [fused_experts] gmm total=2.014ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.574733.574733 mlpmodule.py:2799] [fused_experts] gmm total=1.987ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.575939.575939 mlpmodule.py:2799] [fused_experts] gmm total=2.360ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.575751.575751 mlpmodule.py:2799] [fused_experts] gmm total=2.387ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.576150.576150 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003917694091796875 seconds
INFO 05-06 11:02:15.576756.576756 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.267692565917969e-05 seconds
DEBUG 05-06 11:02:15.576516.576516 cuda_h.py:27] end *layer_moe_fused cost 9.186 ms
DEBUG 05-06 11:02:15.577820.577820 cuda_h.py:27] end decode_layer cost 12.041 ms
DEBUG 05-06 11:02:15.577471.577471 lmp.py:904] ---- decode step 0 layer 19 ----
DEBUG 05-06 11:02:15.578689.578689 cuda_h.py:27] end *sagl cost 1.522 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 31, 47, 111], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {19: 2, 31: 3, 47: 1, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 40, 44, 76, 84], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {8: 1, 40: 2, 44: 2, 76: 1, 84: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 25, 29, 37, 61, 125], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {5: 1, 25: 2, 29: 1, 37: 1, 61: 1, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 38, 78, 82, 86, 106, 122], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {10: 2, 38: 1, 78: 1, 82: 1, 86: 1, 106: 2, 122: 2}}
INFO 05-06 11:02:15.580938.580938 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.336ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 11:02:15.580676.580676 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2172927856445312e-05 seconds
INFO 05-06 11:02:15.580194.580194 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.581289.581289 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014095306396484375 seconds
INFO 05-06 11:02:15.582386.582386 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010628700256347656 seconds
INFO 05-06 11:02:15.582222.582222 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002594470977783203 seconds
INFO 05-06 11:02:15.584994.584994 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012722015380859375 seconds
INFO 05-06 11:02:15.586266.586266 mlpmodule.py:2799] [fused_experts] gmm total=1.943ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.586781.586781 mlpmodule.py:2799] [fused_experts] gmm total=2.032ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.586119.586119 mlpmodule.py:2799] [fused_experts] gmm total=2.174ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.586342.586342 mlpmodule.py:2799] [fused_experts] gmm total=2.194ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.588911.588911 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003980875015258789 seconds
INFO 05-06 11:02:15.588352.588352 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.8160552978515625e-05 seconds
DEBUG 05-06 11:02:15.588600.588600 cuda_h.py:27] end *layer_moe_fused cost 9.103 ms
DEBUG 05-06 11:02:15.589662.589662 cuda_h.py:27] end decode_layer cost 12.005 ms
DEBUG 05-06 11:02:15.589167.589167 lmp.py:904] ---- decode step 0 layer 20 ----
DEBUG 05-06 11:02:15.590640.590640 cuda_h.py:27] end *sagl cost 1.435 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 19, 27, 55, 95, 107], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {3: 2, 19: 1, 27: 1, 55: 1, 95: 3, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [28, 36, 40, 52], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {28: 1, 36: 1, 40: 1, 52: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 21, 45, 73, 85, 93, 117], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {13: 2, 21: 3, 45: 1, 73: 1, 85: 1, 93: 1, 117: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 62, 90, 94, 102], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 1, 62: 1, 90: 1, 94: 2, 102: 2}}
INFO 05-06 11:02:15.592734.592734 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.305ms | allocate_experts_across_cpu_gpu: 0.092ms
INFO 05-06 11:02:15.592227.592227 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1219253540039062e-05 seconds
INFO 05-06 11:02:15.592314.592314 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.593585.593585 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009441375732421875 seconds
INFO 05-06 11:02:15.594577.594577 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007145404815673828 seconds
INFO 05-06 11:02:15.594553.594553 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0017666816711425781 seconds
INFO 05-06 11:02:15.595085.595085 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001237630844116211 seconds
INFO 05-06 11:02:15.597270.597270 mlpmodule.py:2799] [fused_experts] gmm total=2.006ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.597547.597547 mlpmodule.py:2799] [fused_experts] gmm total=2.284ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.598011.598011 mlpmodule.py:2799] [fused_experts] gmm total=2.263ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.598544.598544 mlpmodule.py:2799] [fused_experts] gmm total=2.279ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.599656.599656 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004113674163818359 seconds
INFO 05-06 11:02:15.599806.599806 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 7.891654968261719e-05 seconds
DEBUG 05-06 11:02:15.600309.600309 cuda_h.py:27] end *layer_moe_fused cost 8.266 ms
DEBUG 05-06 11:02:15.600951.600951 cuda_h.py:27] end decode_layer cost 10.966 ms
DEBUG 05-06 11:02:15.600172.600172 lmp.py:904] ---- decode step 0 layer 21 ----
DEBUG 05-06 11:02:15.602967.602967 cuda_h.py:27] end *sagl cost 1.566 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 51, 83, 87, 103], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {11: 1, 51: 1, 83: 1, 87: 1, 103: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [80, 120, 124], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {80: 1, 120: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 13, 21, 25, 29, 45, 57], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {5: 3, 13: 1, 21: 2, 25: 1, 29: 1, 45: 1, 57: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 14, 26, 34, 58, 82, 86, 94, 110], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 13, 'token_per_expert': {2: 1, 14: 1, 26: 3, 34: 1, 58: 1, 82: 1, 86: 2, 94: 1, 110: 2}}
INFO 05-06 11:02:15.603083.603083 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.309ms | allocate_experts_across_cpu_gpu: 0.099ms
INFO 05-06 11:02:15.603814.603814 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1219253540039062e-05 seconds
INFO 05-06 11:02:15.603855.603855 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.604910.604910 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010306835174560547 seconds
INFO 05-06 11:02:15.605154.605154 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006885528564453125 seconds
INFO 05-06 11:02:15.605599.605599 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0018224716186523438 seconds
INFO 05-06 11:02:15.606828.606828 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012600421905517578 seconds
INFO 05-06 11:02:15.608844.608844 mlpmodule.py:2799] [fused_experts] gmm total=2.069ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.609491.609491 mlpmodule.py:2799] [fused_experts] gmm total=2.134ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.609737.609737 mlpmodule.py:2799] [fused_experts] gmm total=2.240ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.609106.609106 mlpmodule.py:2799] [fused_experts] gmm total=2.258ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.610494.610494 lmp.py:1500] [layer_moe_fused] experts compute time: 0.00400853157043457 seconds
INFO 05-06 11:02:15.610935.610935 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 11:02:15.611459.611459 cuda_h.py:27] end *layer_moe_fused cost 8.272 ms
DEBUG 05-06 11:02:15.611307.611307 cuda_h.py:27] end decode_layer cost 11.155 ms
DEBUG 05-06 11:02:15.611051.611051 lmp.py:904] ---- decode step 0 layer 22 ----
DEBUG 05-06 11:02:15.613424.613424 cuda_h.py:27] end *sagl cost 1.432 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 43, 99, 119, 123, 127], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {3: 1, 7: 1, 43: 1, 99: 1, 119: 3, 123: 2, 127: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 24, 76, 100, 108, 120], 'expert_count': 9, 'ideal_gpu_count': 7, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 9, 'token_per_expert': {0: 1, 4: 1, 8: 1, 16: 1, 24: 1, 76: 1, 100: 1, 108: 1, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 61, 101], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {1: 1, 5: 1, 61: 1, 101: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 38, 58, 90, 94], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {2: 1, 6: 1, 26: 1, 38: 1, 58: 1, 90: 1, 94: 2}}
INFO 05-06 11:02:15.614312.614312 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.294ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 11:02:15.614612.614612 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3126602172851562e-05 seconds
INFO 05-06 11:02:15.614984.614984 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.615601.615601 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010025501251220703 seconds
INFO 05-06 11:02:15.616789.616789 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009984970092773438 seconds
INFO 05-06 11:02:15.616903.616903 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002106904983520508 seconds
INFO 05-06 11:02:15.618795.618795 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012924671173095703 seconds
INFO 05-06 11:02:15.620491.620491 mlpmodule.py:2799] [fused_experts] gmm total=2.149ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.620384.620384 mlpmodule.py:2799] [fused_experts] gmm total=2.139ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.620252.620252 mlpmodule.py:2799] [fused_experts] gmm total=2.457ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.620713.620713 mlpmodule.py:2799] [fused_experts] gmm total=2.389ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.622789.622789 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004108428955078125 seconds
INFO 05-06 11:02:15.622707.622707 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 11:02:15.623777.623777 cuda_h.py:27] end *layer_moe_fused cost 9.149 ms
DEBUG 05-06 11:02:15.623573.623573 cuda_h.py:27] end decode_layer cost 11.826 ms
DEBUG 05-06 11:02:15.623802.623802 lmp.py:904] ---- decode step 0 layer 23 ----
DEBUG 05-06 11:02:15.625600.625600 cuda_h.py:27] end *sagl cost 1.635 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 47, 67], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 47: 1, 67: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 32, 84, 108], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 8: 1, 12: 1, 32: 1, 84: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 81, 97, 109], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 17: 1, 81: 1, 97: 2, 109: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 86, 118], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 2, 6: 2, 22: 1, 86: 2, 118: 1}}
INFO 05-06 11:02:15.626873.626873 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.292ms | allocate_experts_across_cpu_gpu: 0.089ms
INFO 05-06 11:02:15.626558.626558 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0742416381835938e-05 seconds
INFO 05-06 11:02:15.626406.626406 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.627434.627434 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011737346649169922 seconds
INFO 05-06 11:02:15.629792.629792 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010967254638671875 seconds
INFO 05-06 11:02:15.629999.629999 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002451181411743164 seconds
INFO 05-06 11:02:15.630770.630770 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012383460998535156 seconds
INFO 05-06 11:02:15.632842.632842 mlpmodule.py:2799] [fused_experts] gmm total=2.109ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.632398.632398 mlpmodule.py:2799] [fused_experts] gmm total=2.208ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.633318.633318 mlpmodule.py:2799] [fused_experts] gmm total=2.296ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.633269.633269 mlpmodule.py:2799] [fused_experts] gmm total=2.303ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.634038.634038 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0041692256927490234 seconds
INFO 05-06 11:02:15.634625.634625 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 11:02:15.634181.634181 cuda_h.py:27] end *layer_moe_fused cost 8.791 ms
DEBUG 05-06 11:02:15.635351.635351 cuda_h.py:27] end decode_layer cost 11.846 ms
DEBUG 05-06 11:02:15.635002.635002 lmp.py:904] ---- decode step 0 layer 24 ----
DEBUG 05-06 11:02:15.637538.637538 cuda_h.py:27] end *sagl cost 1.505 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 63, 79, 123], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 63: 1, 79: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 40, 44], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 12: 1, 40: 1, 44: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 65, 109, 113], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 33: 2, 65: 1, 109: 1, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 66, 90, 110, 118], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 2, 30: 1, 66: 1, 90: 1, 110: 1, 118: 1}}
INFO 05-06 11:02:15.638719.638719 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.308ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 11:02:15.638589.638589 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8358230590820312e-05 seconds
INFO 05-06 11:02:15.638392.638392 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.639968.639968 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001169443130493164 seconds
INFO 05-06 11:02:15.640480.640480 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007810592651367188 seconds
INFO 05-06 11:02:15.640117.640117 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0020554065704345703 seconds
INFO 05-06 11:02:15.641015.641015 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001256704330444336 seconds
INFO 05-06 11:02:15.644165.644165 mlpmodule.py:2799] [fused_experts] gmm total=2.095ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.644188.644188 mlpmodule.py:2799] [fused_experts] gmm total=2.216ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.644373.644373 mlpmodule.py:2799] [fused_experts] gmm total=2.315ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.644277.644277 mlpmodule.py:2799] [fused_experts] gmm total=2.569ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.646169.646169 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004155635833740234 seconds
INFO 05-06 11:02:15.646848.646848 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 11:02:15.646597.646597 cuda_h.py:27] end *layer_moe_fused cost 8.428 ms
DEBUG 05-06 11:02:15.646986.646986 cuda_h.py:27] end decode_layer cost 11.349 ms
DEBUG 05-06 11:02:15.646968.646968 lmp.py:904] ---- decode step 0 layer 25 ----
DEBUG 05-06 11:02:15.648087.648087 cuda_h.py:27] end *sagl cost 1.561 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 47, 67, 95], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 3, 7: 2, 19: 1, 47: 1, 67: 1, 95: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 44, 68, 104], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 3, 4: 2, 16: 1, 44: 1, 68: 1, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 45, 93, 117, 121], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 45: 1, 93: 1, 117: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 58], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {2: 2, 6: 2, 58: 2}}
INFO 05-06 11:02:15.649289.649289 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.319ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 11:02:15.649596.649596 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.4557113647460938e-05 seconds
INFO 05-06 11:02:15.649876.649876 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.651799.651799 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001249551773071289 seconds
INFO 05-06 11:02:15.652900.652900 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011813640594482422 seconds
INFO 05-06 11:02:15.652537.652537 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0025353431701660156 seconds
INFO 05-06 11:02:15.653672.653672 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001226663589477539 seconds
INFO 05-06 11:02:15.656622.656622 mlpmodule.py:2799] [fused_experts] gmm total=2.120ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.656952.656952 mlpmodule.py:2799] [fused_experts] gmm total=2.117ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.656490.656490 mlpmodule.py:2799] [fused_experts] gmm total=2.609ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.656440.656440 mlpmodule.py:2799] [fused_experts] gmm total=2.517ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.658538.658538 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004435300827026367 seconds
INFO 05-06 11:02:15.658079.658079 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 11:02:15.658179.658179 cuda_h.py:27] end *layer_moe_fused cost 9.179 ms
DEBUG 05-06 11:02:15.659727.659727 cuda_h.py:27] end decode_layer cost 12.177 ms
DEBUG 05-06 11:02:15.659755.659755 lmp.py:904] ---- decode step 0 layer 26 ----
DEBUG 05-06 11:02:15.660623.660623 cuda_h.py:27] end *sagl cost 1.545 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 43, 79, 119], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 19: 1, 23: 1, 43: 1, 79: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 52, 84], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 3, 8: 1, 20: 1, 52: 1, 84: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 49, 65, 85], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 2, 5: 2, 49: 1, 65: 1, 85: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 38, 90], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 10: 1, 38: 1, 90: 1}}
INFO 05-06 11:02:15.662526.662526 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 11:02:15.662449.662449 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 11:02:15.662444.662444 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.663066.663066 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013422966003417969 seconds
INFO 05-06 11:02:15.664546.664546 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010311603546142578 seconds
INFO 05-06 11:02:15.664183.664183 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0024862289428710938 seconds
INFO 05-06 11:02:15.666525.666525 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012722015380859375 seconds
INFO 05-06 11:02:15.668802.668802 mlpmodule.py:2799] [fused_experts] gmm total=2.295ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.668701.668701 mlpmodule.py:2799] [fused_experts] gmm total=2.177ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.668844.668844 mlpmodule.py:2799] [fused_experts] gmm total=2.356ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.668987.668987 mlpmodule.py:2799] [fused_experts] gmm total=2.575ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.670125.670125 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004202365875244141 seconds
INFO 05-06 11:02:15.670487.670487 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 11:02:15.670434.670434 cuda_h.py:27] end *layer_moe_fused cost 8.945 ms
DEBUG 05-06 11:02:15.671139.671139 cuda_h.py:27] end decode_layer cost 11.865 ms
DEBUG 05-06 11:02:15.671598.671598 lmp.py:904] ---- decode step 0 layer 27 ----
DEBUG 05-06 11:02:15.672280.672280 cuda_h.py:27] end *sagl cost 1.553 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 87, 103, 115], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 87: 1, 103: 1, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 32, 48, 108], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 32: 1, 48: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 41, 61, 85, 97, 121], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {1: 3, 5: 2, 29: 1, 41: 1, 61: 1, 85: 1, 97: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 58, 62, 114], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 58: 1, 62: 1, 114: 1}}
INFO 05-06 11:02:15.674892.674892 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.308ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 11:02:15.674915.674915 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1219253540039062e-05 seconds
INFO 05-06 11:02:15.674194.674194 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.675872.675872 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012447834014892578 seconds
INFO 05-06 11:02:15.676768.676768 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009965896606445312 seconds
INFO 05-06 11:02:15.676260.676260 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0023419857025146484 seconds
INFO 05-06 11:02:15.677912.677912 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001257181167602539 seconds
INFO 05-06 11:02:15.679444.679444 mlpmodule.py:2799] [fused_experts] gmm total=1.991ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.680873.680873 mlpmodule.py:2799] [fused_experts] gmm total=2.085ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.680112.680112 mlpmodule.py:2799] [fused_experts] gmm total=2.128ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.680189.680189 mlpmodule.py:2799] [fused_experts] gmm total=2.339ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.681327.681327 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003996610641479492 seconds
INFO 05-06 11:02:15.681675.681675 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 11:02:15.682217.682217 cuda_h.py:27] end *layer_moe_fused cost 8.720 ms
DEBUG 05-06 11:02:15.682587.682587 cuda_h.py:27] end decode_layer cost 11.539 ms
DEBUG 05-06 11:02:15.682331.682331 lmp.py:904] ---- decode step 0 layer 28 ----
DEBUG 05-06 11:02:15.684155.684155 cuda_h.py:27] end *sagl cost 1.448 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39, 67, 115, 119], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 19: 1, 39: 1, 67: 1, 115: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 32, 104, 108], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 32: 1, 104: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 49, 53, 57, 65, 89, 105, 113], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 12, 'token_per_expert': {1: 2, 5: 2, 13: 1, 49: 1, 53: 1, 57: 1, 65: 1, 89: 1, 105: 1, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {2: 2, 6: 2}}
INFO 05-06 11:02:15.685897.685897 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.291ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 11:02:15.685429.685429 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 11:02:15.685086.685086 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.687342.687342 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001477956771850586 seconds
INFO 05-06 11:02:15.688945.688945 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016255378723144531 seconds
INFO 05-06 11:02:15.688168.688168 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0033965110778808594 seconds
INFO 05-06 11:02:15.692882.692882 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.003625631332397461 seconds
INFO 05-06 11:02:15.695489.695489 mlpmodule.py:2799] [fused_experts] gmm total=2.148ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.695866.695866 mlpmodule.py:2799] [fused_experts] gmm total=2.221ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.695432.695432 mlpmodule.py:2799] [fused_experts] gmm total=2.102ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.695022.695022 mlpmodule.py:2799] [fused_experts] gmm total=2.444ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.697583.697583 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0043566226959228516 seconds
INFO 05-06 11:02:15.697989.697989 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 0.00010895729064941406 seconds
DEBUG 05-06 11:02:15.698250.698250 cuda_h.py:27] end *layer_moe_fused cost 13.099 ms
DEBUG 05-06 11:02:15.699934.699934 cuda_h.py:27] end decode_layer cost 16.401 ms
DEBUG 05-06 11:02:15.699263.699263 lmp.py:904] ---- decode step 0 layer 29 ----
DEBUG 05-06 11:02:15.702953.702953 cuda_h.py:27] end *sagl cost 3.322 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {3: 2, 7: 2, 23: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 52, 56, 60, 64], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 3, 52: 1, 56: 1, 60: 1, 64: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 49, 73, 81, 97], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 49: 1, 73: 1, 81: 1, 97: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 26, 30, 66, 78], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 3, 6: 2, 18: 1, 26: 1, 30: 1, 66: 1, 78: 1}}
INFO 05-06 11:02:15.705728.705728 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.562ms | allocate_experts_across_cpu_gpu: 0.191ms
INFO 05-06 11:02:15.705832.705832 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.7670135498046875e-05 seconds
INFO 05-06 11:02:15.705821.705821 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:15.706072.706072 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014986991882324219 seconds
INFO 05-06 11:02:15.708952.708952 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014805793762207031 seconds
INFO 05-06 11:02:15.708431.708431 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0031392574310302734 seconds
INFO 05-06 11:02:15.710909.710909 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001569986343383789 seconds
INFO 05-06 11:02:15.712619.712619 mlpmodule.py:2799] [fused_experts] gmm total=1.895ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.712850.712850 mlpmodule.py:2799] [fused_experts] gmm total=2.137ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.712188.712188 mlpmodule.py:2799] [fused_experts] gmm total=2.183ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.712384.712384 mlpmodule.py:2799] [fused_experts] gmm total=2.195ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:15.714683.714683 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004032135009765625 seconds
INFO 05-06 11:02:15.714444.714444 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 7.200241088867188e-05 seconds
DEBUG 05-06 11:02:15.714804.714804 cuda_h.py:27] end *layer_moe_fused cost 10.539 ms
DEBUG 05-06 11:02:15.715335.715335 cuda_h.py:27] end decode_layer cost 16.203 ms
DEBUG 05-06 11:02:15.715948.715948 cuda_h.py:27] end decode_step cost 490.395 ms
INFO 05-06 11:02:15.715400.715400 lmp.py:931] decode step 0 time: 0.4904518127441406 seconds
WARNING 05-06 11:02:15.716863.716863 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:02:15.716174.716174 helper.py:35]   NaN count (hidden): 5632
WARNING 05-06 11:02:15.716435.716435 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:02:15.717189.717189 helper.py:39]   NaN count (normed): 5632
WARNING 05-06 11:02:15.722113.722113 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:02:15.722509.722509 helper.py:50]   NaN count: 524288
WARNING 05-06 11:02:15.722206.722206 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 11:02:15.723113.723113 helper.py:80] WARNING: Logits have extreme values: min=-896.00, max=1032.00
WARNING 05-06 11:02:15.723564.723564 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 11:02:15.725569.725569 cuda_h.py:27] end init_inputs_tokens cost 9.207 ms
DEBUG 05-06 11:02:15.725578.725578 lmp.py:899] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:02:15.725176.725176 lmp.py:904] ---- decode step 1 layer 0 ----
DEBUG 05-06 11:02:15.727926.727926 cuda_h.py:27] end *sagl cost 1.975 ms
DEBUG 05-06 11:02:15.731401.731401 cuda_h.py:27] end *layer_moe_fused cost 3.176 ms
DEBUG 05-06 11:02:15.732073.732073 cuda_h.py:27] end decode_layer cost 6.800 ms
DEBUG 05-06 11:02:15.732691.732691 lmp.py:904] ---- decode step 1 layer 1 ----
DEBUG 05-06 11:02:15.734598.734598 cuda_h.py:27] end *sagl cost 1.911 ms
DEBUG 05-06 11:02:15.737781.737781 cuda_h.py:27] end *layer_moe_fused cost 2.597 ms
DEBUG 05-06 11:02:15.738340.738340 cuda_h.py:27] end decode_layer cost 6.126 ms
DEBUG 05-06 11:02:15.738622.738622 lmp.py:904] ---- decode step 1 layer 2 ----
DEBUG 05-06 11:02:15.740702.740702 cuda_h.py:27] end *sagl cost 1.930 ms
DEBUG 05-06 11:02:15.743374.743374 cuda_h.py:27] end *layer_moe_fused cost 2.406 ms
DEBUG 05-06 11:02:15.744357.744357 cuda_h.py:27] end decode_layer cost 5.945 ms
DEBUG 05-06 11:02:15.744784.744784 lmp.py:904] ---- decode step 1 layer 3 ----
DEBUG 05-06 11:02:15.746603.746603 cuda_h.py:27] end *sagl cost 1.882 ms
DEBUG 05-06 11:02:15.749798.749798 cuda_h.py:27] end *layer_moe_fused cost 2.163 ms
DEBUG 05-06 11:02:15.750065.750065 cuda_h.py:27] end decode_layer cost 5.645 ms
DEBUG 05-06 11:02:15.750684.750684 lmp.py:904] ---- decode step 1 layer 4 ----
DEBUG 05-06 11:02:15.752464.752464 cuda_h.py:27] end *sagl cost 1.889 ms
DEBUG 05-06 11:02:15.755751.755751 cuda_h.py:27] end *layer_moe_fused cost 2.202 ms
DEBUG 05-06 11:02:15.755331.755331 cuda_h.py:27] end decode_layer cost 5.668 ms
DEBUG 05-06 11:02:15.756187.756187 lmp.py:904] ---- decode step 1 layer 5 ----
DEBUG 05-06 11:02:15.757244.757244 cuda_h.py:27] end *sagl cost 1.846 ms
DEBUG 05-06 11:02:15.761116.761116 cuda_h.py:27] end *layer_moe_fused cost 2.224 ms
DEBUG 05-06 11:02:15.761045.761045 cuda_h.py:27] end decode_layer cost 5.649 ms
DEBUG 05-06 11:02:15.761472.761472 lmp.py:904] ---- decode step 1 layer 6 ----
DEBUG 05-06 11:02:15.763835.763835 cuda_h.py:27] end *sagl cost 1.898 ms
DEBUG 05-06 11:02:15.766959.766959 cuda_h.py:27] end *layer_moe_fused cost 2.073 ms
DEBUG 05-06 11:02:15.767465.767465 cuda_h.py:27] end decode_layer cost 5.539 ms
DEBUG 05-06 11:02:15.767414.767414 lmp.py:904] ---- decode step 1 layer 7 ----
DEBUG 05-06 11:02:15.769665.769665 cuda_h.py:27] end *sagl cost 1.884 ms
DEBUG 05-06 11:02:15.772585.772585 cuda_h.py:27] end *layer_moe_fused cost 2.313 ms
DEBUG 05-06 11:02:15.773144.773144 cuda_h.py:27] end decode_layer cost 5.765 ms
DEBUG 05-06 11:02:15.773570.773570 lmp.py:904] ---- decode step 1 layer 8 ----
DEBUG 05-06 11:02:15.775104.775104 cuda_h.py:27] end *sagl cost 1.848 ms
DEBUG 05-06 11:02:15.778261.778261 cuda_h.py:27] end *layer_moe_fused cost 2.247 ms
DEBUG 05-06 11:02:15.778496.778496 cuda_h.py:27] end decode_layer cost 5.699 ms
DEBUG 05-06 11:02:15.778161.778161 lmp.py:904] ---- decode step 1 layer 9 ----
DEBUG 05-06 11:02:15.780606.780606 cuda_h.py:27] end *sagl cost 1.957 ms
DEBUG 05-06 11:02:15.784857.784857 cuda_h.py:27] end *layer_moe_fused cost 2.277 ms
DEBUG 05-06 11:02:15.784668.784668 cuda_h.py:27] end decode_layer cost 5.846 ms
DEBUG 05-06 11:02:15.784809.784809 lmp.py:904] ---- decode step 1 layer 10 ----
DEBUG 05-06 11:02:15.786913.786913 cuda_h.py:27] end *sagl cost 1.845 ms
DEBUG 05-06 11:02:15.790811.790811 cuda_h.py:27] end *layer_moe_fused cost 2.201 ms
DEBUG 05-06 11:02:15.790317.790317 cuda_h.py:27] end decode_layer cost 5.672 ms
DEBUG 05-06 11:02:15.790505.790505 lmp.py:904] ---- decode step 1 layer 11 ----
DEBUG 05-06 11:02:15.792908.792908 cuda_h.py:27] end *sagl cost 1.890 ms
DEBUG 05-06 11:02:15.795799.795799 cuda_h.py:27] end *layer_moe_fused cost 2.008 ms
DEBUG 05-06 11:02:15.796828.796828 cuda_h.py:27] end decode_layer cost 5.500 ms
DEBUG 05-06 11:02:15.796586.796586 lmp.py:904] ---- decode step 1 layer 12 ----
DEBUG 05-06 11:02:15.798199.798199 cuda_h.py:27] end *sagl cost 1.835 ms
DEBUG 05-06 11:02:15.801775.801775 cuda_h.py:27] end *layer_moe_fused cost 2.133 ms
DEBUG 05-06 11:02:15.801758.801758 cuda_h.py:27] end decode_layer cost 5.529 ms
DEBUG 05-06 11:02:15.801753.801753 lmp.py:904] ---- decode step 1 layer 13 ----
DEBUG 05-06 11:02:15.803687.803687 cuda_h.py:27] end *sagl cost 1.931 ms
DEBUG 05-06 11:02:15.806276.806276 cuda_h.py:27] end *layer_moe_fused cost 2.078 ms
DEBUG 05-06 11:02:15.807550.807550 cuda_h.py:27] end decode_layer cost 5.603 ms
DEBUG 05-06 11:02:15.807076.807076 lmp.py:904] ---- decode step 1 layer 14 ----
DEBUG 05-06 11:02:15.809326.809326 cuda_h.py:27] end *sagl cost 1.884 ms
DEBUG 05-06 11:02:15.812343.812343 cuda_h.py:27] end *layer_moe_fused cost 2.238 ms
DEBUG 05-06 11:02:15.813578.813578 cuda_h.py:27] end decode_layer cost 5.702 ms
DEBUG 05-06 11:02:15.813958.813958 lmp.py:904] ---- decode step 1 layer 15 ----
DEBUG 05-06 11:02:15.815540.815540 cuda_h.py:27] end *sagl cost 1.882 ms
DEBUG 05-06 11:02:15.818193.818193 cuda_h.py:27] end *layer_moe_fused cost 2.226 ms
DEBUG 05-06 11:02:15.818944.818944 cuda_h.py:27] end decode_layer cost 5.707 ms
DEBUG 05-06 11:02:15.819085.819085 lmp.py:904] ---- decode step 1 layer 16 ----
DEBUG 05-06 11:02:15.820574.820574 cuda_h.py:27] end *sagl cost 1.883 ms
DEBUG 05-06 11:02:15.824631.824631 cuda_h.py:27] end *layer_moe_fused cost 2.227 ms
DEBUG 05-06 11:02:15.824197.824197 cuda_h.py:27] end decode_layer cost 5.695 ms
DEBUG 05-06 11:02:15.824669.824669 lmp.py:904] ---- decode step 1 layer 17 ----
DEBUG 05-06 11:02:15.826111.826111 cuda_h.py:27] end *sagl cost 1.849 ms
DEBUG 05-06 11:02:15.830911.830911 cuda_h.py:27] end *layer_moe_fused cost 2.215 ms
DEBUG 05-06 11:02:15.830424.830424 cuda_h.py:27] end decode_layer cost 5.713 ms
DEBUG 05-06 11:02:15.830327.830327 lmp.py:904] ---- decode step 1 layer 18 ----
DEBUG 05-06 11:02:15.832371.832371 cuda_h.py:27] end *sagl cost 1.873 ms
DEBUG 05-06 11:02:15.835687.835687 cuda_h.py:27] end *layer_moe_fused cost 2.248 ms
DEBUG 05-06 11:02:15.836292.836292 cuda_h.py:27] end decode_layer cost 5.693 ms
DEBUG 05-06 11:02:15.836149.836149 lmp.py:904] ---- decode step 1 layer 19 ----
DEBUG 05-06 11:02:15.838168.838168 cuda_h.py:27] end *sagl cost 1.889 ms
DEBUG 05-06 11:02:15.841914.841914 cuda_h.py:27] end *layer_moe_fused cost 2.225 ms
DEBUG 05-06 11:02:15.842758.842758 cuda_h.py:27] end decode_layer cost 5.711 ms
DEBUG 05-06 11:02:15.842707.842707 lmp.py:904] ---- decode step 1 layer 20 ----
DEBUG 05-06 11:02:15.844374.844374 cuda_h.py:27] end *sagl cost 1.875 ms
DEBUG 05-06 11:02:15.847878.847878 cuda_h.py:27] end *layer_moe_fused cost 2.355 ms
DEBUG 05-06 11:02:15.847141.847141 cuda_h.py:27] end decode_layer cost 5.827 ms
DEBUG 05-06 11:02:15.848520.848520 lmp.py:904] ---- decode step 1 layer 21 ----
DEBUG 05-06 11:02:15.849328.849328 cuda_h.py:27] end *sagl cost 1.908 ms
DEBUG 05-06 11:02:15.853834.853834 cuda_h.py:27] end *layer_moe_fused cost 2.184 ms
DEBUG 05-06 11:02:15.853241.853241 cuda_h.py:27] end decode_layer cost 5.688 ms
DEBUG 05-06 11:02:15.853428.853428 lmp.py:904] ---- decode step 1 layer 22 ----
DEBUG 05-06 11:02:15.855751.855751 cuda_h.py:27] end *sagl cost 1.865 ms
DEBUG 05-06 11:02:15.859449.859449 cuda_h.py:27] end *layer_moe_fused cost 2.383 ms
DEBUG 05-06 11:02:15.859369.859369 cuda_h.py:27] end decode_layer cost 5.919 ms
DEBUG 05-06 11:02:15.859133.859133 lmp.py:904] ---- decode step 1 layer 23 ----
DEBUG 05-06 11:02:15.861488.861488 cuda_h.py:27] end *sagl cost 1.854 ms
DEBUG 05-06 11:02:15.864555.864555 cuda_h.py:27] end *layer_moe_fused cost 1.955 ms
DEBUG 05-06 11:02:15.865485.865485 cuda_h.py:27] end decode_layer cost 5.381 ms
DEBUG 05-06 11:02:15.865673.865673 lmp.py:904] ---- decode step 1 layer 24 ----
DEBUG 05-06 11:02:15.867478.867478 cuda_h.py:27] end *sagl cost 1.836 ms
DEBUG 05-06 11:02:15.869680.869680 cuda_h.py:27] end *layer_moe_fused cost 1.569 ms
DEBUG 05-06 11:02:15.870418.870418 cuda_h.py:27] end decode_layer cost 5.039 ms
DEBUG 05-06 11:02:15.870367.870367 lmp.py:904] ---- decode step 1 layer 25 ----
DEBUG 05-06 11:02:15.872439.872439 cuda_h.py:27] end *sagl cost 1.893 ms
DEBUG 05-06 11:02:15.874646.874646 cuda_h.py:27] end *layer_moe_fused cost 1.558 ms
DEBUG 05-06 11:02:15.875138.875138 cuda_h.py:27] end decode_layer cost 5.022 ms
DEBUG 05-06 11:02:15.875895.875895 lmp.py:904] ---- decode step 1 layer 26 ----
DEBUG 05-06 11:02:15.877371.877371 cuda_h.py:27] end *sagl cost 1.875 ms
DEBUG 05-06 11:02:15.879239.879239 cuda_h.py:27] end *layer_moe_fused cost 1.582 ms
DEBUG 05-06 11:02:15.880878.880878 cuda_h.py:27] end decode_layer cost 5.003 ms
DEBUG 05-06 11:02:15.880396.880396 lmp.py:904] ---- decode step 1 layer 27 ----
DEBUG 05-06 11:02:15.882560.882560 cuda_h.py:27] end *sagl cost 1.857 ms
DEBUG 05-06 11:02:15.885059.885059 cuda_h.py:27] end *layer_moe_fused cost 1.599 ms
DEBUG 05-06 11:02:15.885274.885274 cuda_h.py:27] end decode_layer cost 5.031 ms
DEBUG 05-06 11:02:15.885792.885792 lmp.py:904] ---- decode step 1 layer 28 ----
DEBUG 05-06 11:02:15.887948.887948 cuda_h.py:27] end *sagl cost 1.815 ms
DEBUG 05-06 11:02:15.890438.890438 cuda_h.py:27] end *layer_moe_fused cost 1.530 ms
DEBUG 05-06 11:02:15.890838.890838 cuda_h.py:27] end decode_layer cost 4.909 ms
DEBUG 05-06 11:02:15.890072.890072 lmp.py:904] ---- decode step 1 layer 29 ----
DEBUG 05-06 11:02:15.892779.892779 cuda_h.py:27] end *sagl cost 1.870 ms
DEBUG 05-06 11:02:15.895594.895594 cuda_h.py:27] end *layer_moe_fused cost 1.557 ms
DEBUG 05-06 11:02:15.895769.895769 cuda_h.py:27] end decode_layer cost 4.992 ms
DEBUG 05-06 11:02:15.895195.895195 cuda_h.py:27] end decode_step cost 179.620 ms
INFO 05-06 11:02:15.895389.895389 lmp.py:931] decode step 1 time: 0.1796586513519287 seconds
Time taken: 7.910366505384445 seconds
generate input ids cost 0.0767674446105957 s
DEBUG 05-06 11:02:18.476310.476310 cuda_h.py:27] end generate_input_ids cost 2430.968 ms
DEBUG 05-06 11:02:18.476429.476429 cuda_h.py:27] end init_cache cost 0.047 ms
INFO 05-06 11:02:18.489002.489002 lmp.py:2350] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6319480772, 'cuda:1': 12802195456, 'cuda:2': 12808486912, 'cuda:3': 12808486912} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7367451361367423, 'cuda:1': 0.4714464292478611, 'cuda:2': 0.4713240027915884, 'cuda:3': 0.4713240027915884}
INFO 05-06 11:02:18.489006.489006 lmp.py:2368] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489306.489306 lmp.py:2368] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489929.489929 lmp.py:2368] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489215.489215 lmp.py:2368] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489323.489323 lmp.py:2368] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489324.489324 lmp.py:2368] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489656.489656 lmp.py:2368] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489750.489750 lmp.py:2368] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489367.489367 lmp.py:2368] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489030.489030 lmp.py:2368] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489648.489648 lmp.py:2368] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489788.489788 lmp.py:2368] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489166.489166 lmp.py:2368] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489783.489783 lmp.py:2368] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489923.489923 lmp.py:2368] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489017.489017 lmp.py:2368] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489303.489303 lmp.py:2368] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489920.489920 lmp.py:2368] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.489200.489200 lmp.py:2368] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.490393.490393 lmp.py:2368] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.490174.490174 lmp.py:2368] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.490891.490891 lmp.py:2368] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.490526.490526 lmp.py:2368] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.490720.490720 lmp.py:2368] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.490740.490740 lmp.py:2368] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.490218.490218 lmp.py:2368] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.490178.490178 lmp.py:2368] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.490610.490610 lmp.py:2368] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.491975.491975 lmp.py:2368] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 11:02:18.491261.491261 lmp.py:2368] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 11:02:18.771966.771966 cuda_h.py:27] end init_loading_placement cost 294.645 ms
DEBUG 05-06 11:02:18.771831.771831 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:02:18.771481.771481 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:02:18 client.py:72] load_into_gpu: gemma4-26B-A4B, 06ca9b98-7627-4245-91d1-9edc282b1b46
INFO 05-06 11:02:18 client.py:135] Model loaded: gemma4-26B-A4B, 06ca9b98-7627-4245-91d1-9edc282b1b46
INFO 05-06 11:02:18 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 06ca9b98-7627-4245-91d1-9edc282b1b46
INFO 05-06 11:02:19 client.py:212] Model loaded
DEBUG 05-06 11:02:19.298778.298778 cuda_h.py:27] end init_general_sagl_loading_async cost 527.087 ms
INFO 05-06 11:02:19.345806.345806 lmp.py:2871] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 11:02:19.444356.444356 cuda_h.py:27] end restore_state_dict cost 98.601 ms
DEBUG 05-06 11:02:19.444850.444850 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 11:02:19.444918.444918 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 11:02:19 client.py:72] load_into_gpu: gemma4-26B-A4B, 2cbe07d6-d192-4431-af7e-f430a33211b9
INFO 05-06 11:02:19 client.py:135] Model loaded: gemma4-26B-A4B, 2cbe07d6-d192-4431-af7e-f430a33211b9
DEBUG 05-06 11:02:19.577962.577962 cuda_h.py:27] end init_experts_loading_async cost 132.858 ms
DEBUG 05-06 11:02:19.578680.578680 cuda_h.py:27] end init_inputs_tokens cost 1.018 ms
DEBUG 05-06 11:02:19.578744.578744 lmp.py:824] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 11:02:19.584843.584843 cuda_h.py:27] end *sagl cost 5.582 ms
experts_cpu_alloc {'expert_ids': [19, 87, 27, 15, 119, 63, 23, 111, 107, 11, 123, 59, 71, 79, 30, 86, 6, 66, 106, 94, 14, 2, 10, 34, 114, 38, 88, 120, 12, 36, 100, 96, 8, 4, 20, 44, 84, 80, 60, 112, 76, 108, 17, 85, 97, 109, 101, 29, 81, 93, 45, 49, 65, 73, 13, 5, 9, 69], 'token_total': 1519, 'token_per_expert': {19: 2, 87: 2, 27: 10, 15: 13, 119: 13, 63: 15, 23: 23, 111: 23, 107: 25, 11: 31, 123: 39, 59: 43, 71: 65, 79: 76, 30: 1, 86: 2, 6: 4, 66: 6, 106: 14, 94: 24, 14: 29, 2: 30, 10: 36, 34: 38, 114: 48, 38: 59, 88: 1, 120: 1, 12: 2, 36: 4, 100: 5, 96: 8, 8: 9, 4: 11, 20: 15, 44: 17, 84: 21, 80: 28, 60: 55, 112: 68, 76: 74, 108: 78, 17: 3, 85: 3, 97: 3, 109: 3, 101: 6, 29: 7, 81: 11, 93: 12, 45: 14, 49: 17, 65: 39, 73: 60, 13: 61, 5: 66, 9: 68, 69: 78}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 39, 47, 51, 55, 67, 75, 83, 91, 99, 103, 115, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4710, 'token_per_expert': {3: 160, 7: 374, 31: 134, 39: 718, 47: 1304, 51: 186, 55: 208, 67: 183, 75: 89, 83: 105, 91: 458, 99: 161, 103: 432, 115: 89, 127: 109}}
experts_gpu_alloc_device_1 {'expert_ids': [18, 22, 26, 46, 50, 54, 70, 74, 78, 90, 102, 110, 118, 122, 126], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 27, 'token_total': 3769, 'token_per_expert': {18: 71, 22: 255, 26: 304, 46: 450, 50: 520, 54: 275, 70: 140, 74: 224, 78: 109, 90: 546, 102: 74, 110: 83, 118: 89, 122: 114, 126: 515}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 16, 24, 28, 32, 48, 52, 64, 68, 72, 92, 104, 116, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 3060, 'token_per_expert': {0: 249, 16: 201, 24: 81, 28: 123, 32: 183, 48: 146, 52: 150, 64: 106, 68: 694, 72: 100, 92: 87, 104: 134, 116: 82, 124: 724}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 21, 25, 33, 37, 41, 53, 77, 89, 105, 113, 117, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 3326, 'token_per_expert': {1: 273, 21: 171, 25: 110, 33: 828, 37: 81, 41: 142, 53: 819, 77: 99, 89: 133, 105: 89, 113: 157, 117: 97, 121: 226, 125: 101}}
INFO 05-06 11:02:19.588722.588722 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 1.111ms | allocate_experts_across_cpu_gpu: 0.537ms
INFO 05-06 11:02:19.588444.588444 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.225440979003906e-05 seconds
INFO 05-06 11:02:19.590115.590115 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.001961946487426758 seconds
INFO 05-06 11:02:19.614226.614226 lmp.py:1387] [layer_moe_fused] to time: 0.00015211105346679688 seconds
INFO 05-06 11:02:19.615571.615571 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:19.617204.617204 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=116 time: 0.001495361328125 seconds
INFO 05-06 11:02:19.717091.717091 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.09995746612548828 seconds
INFO 05-06 11:02:19.717348.717348 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.10170650482177734 seconds
INFO 05-06 11:02:19.748889.748889 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.03131699562072754 seconds
INFO 05-06 11:02:19.753530.753530 mlpmodule.py:2799] [fused_experts] gmm total=4.350ms E=32 S=5090 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.754607.754607 mlpmodule.py:2799] [fused_experts] gmm total=4.933ms E=32 S=4060 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.754633.754633 mlpmodule.py:2799] [fused_experts] gmm total=5.598ms E=32 S=3457 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.755341.755341 mlpmodule.py:2799] [fused_experts] gmm total=6.264ms E=32 S=3777 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.757738.757738 lmp.py:1500] [layer_moe_fused] experts compute time: 0.008666515350341797 seconds
INFO 05-06 11:02:19.757146.757146 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.650520324707031e-05 seconds
DEBUG 05-06 11:02:19.757027.757027 cuda_h.py:27] end *layer_moe_fused cost 171.200 ms
DEBUG 05-06 11:02:19.776188.776188 cuda_h.py:27] end prefill_layer cost 197.978 ms
DEBUG 05-06 11:02:19.776322.776322 lmp.py:841] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 11:02:19.776741.776741 lmp.py:824] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 11:02:19.778227.778227 cuda_h.py:27] end *sagl cost 1.626 ms
experts_cpu_alloc {'expert_ids': [63, 75, 39, 43, 23, 115, 15, 31, 55, 91, 103, 87, 83, 114, 26, 18, 62, 66, 14, 110, 78, 38, 50, 74, 98, 90, 44, 16, 72, 88, 32, 84, 60, 116, 40, 112, 48, 76, 108, 56, 104, 92, 61, 33, 125, 77, 89, 41, 81, 45, 93, 57, 37, 69, 29, 121, 101], 'token_total': 807, 'token_per_expert': {63: 1, 75: 1, 39: 3, 43: 3, 23: 4, 115: 6, 15: 10, 31: 10, 55: 10, 91: 12, 103: 12, 87: 13, 83: 16, 114: 3, 26: 8, 18: 9, 62: 10, 66: 12, 14: 13, 110: 17, 78: 19, 38: 25, 50: 31, 74: 39, 98: 39, 90: 40, 44: 3, 16: 4, 72: 10, 88: 10, 32: 11, 84: 12, 60: 14, 116: 14, 40: 15, 112: 17, 48: 18, 76: 19, 108: 21, 56: 24, 104: 30, 92: 40, 61: 1, 33: 3, 125: 4, 77: 5, 89: 7, 41: 8, 81: 8, 45: 12, 93: 16, 57: 17, 37: 18, 69: 18, 29: 19, 121: 19, 101: 24}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 27, 35, 47, 51, 59, 67, 79, 95, 99, 119, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 3231, 'token_per_expert': {3: 998, 7: 1052, 11: 24, 27: 25, 35: 32, 47: 84, 51: 122, 59: 74, 67: 230, 79: 44, 95: 37, 99: 295, 119: 78, 123: 16, 127: 120}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 22, 30, 34, 42, 46, 54, 82, 94, 106, 118, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 4490, 'token_per_expert': {2: 965, 6: 970, 10: 317, 22: 284, 30: 543, 34: 49, 42: 83, 46: 122, 54: 118, 82: 421, 94: 63, 106: 104, 118: 156, 122: 295}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 12, 20, 28, 52, 64, 68, 80, 96, 100, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 3910, 'token_per_expert': {0: 974, 4: 995, 8: 246, 12: 116, 20: 111, 28: 112, 52: 561, 64: 53, 68: 329, 80: 90, 96: 105, 100: 112, 120: 57, 124: 49}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 21, 25, 49, 53, 65, 73, 85, 97, 105, 109], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 3946, 'token_per_expert': {1: 1030, 5: 1203, 9: 41, 13: 608, 21: 39, 25: 80, 49: 63, 53: 89, 65: 78, 73: 55, 85: 54, 97: 268, 105: 37, 109: 301}}
INFO 05-06 11:02:19.780285.780285 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.995ms | allocate_experts_across_cpu_gpu: 0.247ms
INFO 05-06 11:02:19.780435.780435 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.1484832763671875e-05 seconds
INFO 05-06 11:02:19.781460.781460 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0009331703186035156 seconds
INFO 05-06 11:02:19.861278.861278 lmp.py:1387] [layer_moe_fused] to time: 0.00016021728515625 seconds
INFO 05-06 11:02:19.861604.861604 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:19.863537.863537 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001302957534790039 seconds
INFO 05-06 11:02:19.864487.864487 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006177425384521484 seconds
INFO 05-06 11:02:19.864437.864437 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002080678939819336 seconds
INFO 05-06 11:02:19.895921.895921 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0316309928894043 seconds
INFO 05-06 11:02:19.898780.898780 mlpmodule.py:2799] [fused_experts] gmm total=2.437ms E=32 S=3332 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.899197.899197 mlpmodule.py:2799] [fused_experts] gmm total=2.958ms E=32 S=4755 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.899190.899190 mlpmodule.py:2799] [fused_experts] gmm total=3.277ms E=32 S=4172 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.899411.899411 mlpmodule.py:2799] [fused_experts] gmm total=3.490ms E=32 S=4125 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.902779.902779 lmp.py:1500] [layer_moe_fused] experts compute time: 0.006159782409667969 seconds
INFO 05-06 11:02:19.902697.902697 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.078315734863281e-05 seconds
DEBUG 05-06 11:02:19.902737.902737 cuda_h.py:27] end *layer_moe_fused cost 123.518 ms
DEBUG 05-06 11:02:19.924552.924552 cuda_h.py:27] end prefill_layer cost 147.427 ms
DEBUG 05-06 11:02:19.924693.924693 lmp.py:841] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 11:02:19.924065.924065 lmp.py:824] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 11:02:19.926532.926532 cuda_h.py:27] end *sagl cost 1.643 ms
experts_cpu_alloc {'expert_ids': [75, 79, 67, 99, 27, 115, 35, 95, 111, 23, 103, 107, 83, 119, 63, 26, 66, 86, 50, 42, 114, 82, 58, 46, 70, 126, 98, 16, 32, 68, 12, 120, 64, 40, 96, 52, 88, 28, 72, 116, 24, 44, 56, 21, 25, 45, 121, 61, 105, 113, 17, 85, 57, 33, 77, 49], 'token_total': 1261, 'token_per_expert': {75: 1, 79: 1, 67: 4, 99: 5, 27: 16, 115: 16, 35: 19, 95: 20, 111: 20, 23: 25, 103: 33, 107: 35, 83: 36, 119: 41, 63: 43, 26: 3, 66: 3, 86: 3, 50: 6, 42: 14, 114: 17, 82: 18, 58: 23, 46: 32, 70: 35, 126: 42, 98: 47, 16: 2, 32: 2, 68: 4, 12: 5, 120: 12, 64: 13, 40: 14, 96: 15, 52: 21, 88: 27, 28: 29, 72: 33, 116: 34, 24: 35, 44: 37, 56: 39, 21: 3, 25: 4, 45: 6, 121: 10, 61: 15, 105: 19, 113: 27, 17: 40, 85: 41, 57: 48, 33: 50, 77: 57, 49: 61}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 31, 43, 51, 55, 59, 71, 91, 123, 127], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 4096, 'token_per_expert': {3: 1092, 7: 1151, 11: 455, 15: 179, 19: 315, 31: 48, 43: 51, 51: 54, 55: 126, 59: 227, 71: 44, 91: 56, 123: 57, 127: 241}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 18, 34, 54, 62, 78, 90, 102, 106, 110, 118, 122], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 3430, 'token_per_expert': {2: 1024, 6: 1032, 14: 64, 18: 80, 34: 59, 54: 199, 62: 309, 78: 67, 90: 124, 102: 168, 106: 74, 110: 56, 118: 117, 122: 57}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 20, 36, 48, 60, 76, 80, 84, 100, 104, 108, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 3575, 'token_per_expert': {0: 1043, 4: 1066, 8: 42, 20: 109, 36: 43, 48: 123, 60: 99, 76: 131, 80: 136, 84: 129, 100: 41, 104: 82, 108: 491, 124: 40}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 29, 37, 41, 53, 65, 69, 81, 97, 109, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 4022, 'token_per_expert': {1: 1256, 5: 1037, 9: 243, 13: 205, 29: 164, 37: 140, 41: 268, 53: 89, 65: 88, 69: 67, 81: 140, 97: 66, 109: 82, 125: 177}}
INFO 05-06 11:02:19.928246.928246 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 1.419ms | allocate_experts_across_cpu_gpu: 0.256ms
INFO 05-06 11:02:19.928826.928826 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.076957702636719e-05 seconds
INFO 05-06 11:02:19.929790.929790 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.000850677490234375 seconds
INFO 05-06 11:02:19.966862.966862 lmp.py:1387] [layer_moe_fused] to time: 0.00014543533325195312 seconds
INFO 05-06 11:02:19.966525.966525 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:19.968315.968315 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013709068298339844 seconds
INFO 05-06 11:02:19.969685.969685 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006809234619140625 seconds
INFO 05-06 11:02:19.969343.969343 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0022077560424804688 seconds
INFO 05-06 11:02:19.996303.996303 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02745366096496582 seconds
INFO 05-06 11:02:19.998501.998501 mlpmodule.py:2799] [fused_experts] gmm total=2.036ms E=32 S=3673 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.999618.999618 mlpmodule.py:2799] [fused_experts] gmm total=2.100ms E=32 S=3897 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.999987.999987 mlpmodule.py:2799] [fused_experts] gmm total=2.358ms E=32 S=4411 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:19.999824.999824 mlpmodule.py:2799] [fused_experts] gmm total=2.260ms E=32 S=4403 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.002756.002756 lmp.py:1500] [layer_moe_fused] experts compute time: 0.005513191223144531 seconds
INFO 05-06 11:02:20.002296.002296 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.221366882324219e-05 seconds
DEBUG 05-06 11:02:20.003434.003434 cuda_h.py:27] end *layer_moe_fused cost 76.110 ms
DEBUG 05-06 11:02:20.022827.022827 cuda_h.py:27] end prefill_layer cost 98.305 ms
DEBUG 05-06 11:02:20.022160.022160 lmp.py:841] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 11:02:20.022102.022102 lmp.py:824] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 11:02:20.024775.024775 cuda_h.py:27] end *sagl cost 1.693 ms
experts_cpu_alloc {'expert_ids': [115, 23, 87, 91, 35, 55, 27, 127, 63, 31, 111, 67, 43, 46, 106, 126, 38, 18, 94, 82, 42, 98, 30, 86, 26, 58, 110, 114, 70, 20, 36, 80, 32, 72, 16, 56, 24, 60, 48, 116, 8, 40, 100, 64, 45, 37, 29, 125, 21, 65, 41, 57, 89, 117, 33, 101, 13, 61], 'token_total': 1120, 'token_per_expert': {115: 1, 23: 5, 87: 5, 91: 6, 35: 7, 55: 9, 27: 14, 127: 19, 63: 29, 31: 31, 111: 33, 67: 36, 43: 38, 46: 1, 106: 1, 126: 1, 38: 2, 18: 4, 94: 6, 82: 7, 42: 8, 98: 14, 30: 18, 86: 22, 26: 43, 58: 58, 110: 59, 114: 65, 70: 80, 20: 2, 36: 4, 80: 4, 32: 9, 72: 11, 16: 14, 56: 14, 24: 19, 60: 20, 48: 21, 116: 21, 8: 27, 40: 31, 100: 39, 64: 50, 45: 1, 37: 2, 29: 3, 125: 3, 21: 4, 65: 7, 41: 11, 57: 11, 89: 11, 117: 19, 33: 24, 101: 38, 13: 39, 61: 39}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 39, 51, 59, 71, 75, 83, 95, 107, 119, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 3205, 'token_per_expert': {3: 1082, 7: 1024, 11: 74, 15: 74, 19: 55, 39: 48, 51: 81, 59: 51, 71: 157, 75: 200, 83: 85, 95: 100, 107: 74, 119: 58, 123: 42}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 14, 22, 34, 50, 54, 62, 66, 74, 78, 102, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4584, 'token_per_expert': {2: 1082, 6: 1076, 10: 95, 14: 186, 22: 233, 34: 121, 50: 329, 54: 107, 62: 229, 66: 180, 74: 94, 78: 328, 102: 216, 118: 113, 122: 195}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 28, 44, 52, 68, 76, 84, 88, 92, 96, 104, 108, 120], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 3687, 'token_per_expert': {0: 1103, 4: 1202, 28: 156, 44: 58, 52: 116, 68: 94, 76: 95, 84: 139, 88: 146, 92: 134, 96: 147, 104: 109, 108: 83, 120: 105}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 17, 25, 53, 69, 73, 77, 85, 93, 97, 109, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 3788, 'token_per_expert': {1: 1038, 5: 1148, 9: 200, 17: 91, 25: 93, 53: 126, 69: 75, 73: 116, 77: 45, 85: 298, 93: 169, 97: 125, 109: 43, 121: 221}}
INFO 05-06 11:02:20.026203.026203 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.916ms | allocate_experts_across_cpu_gpu: 0.263ms
INFO 05-06 11:02:20.026783.026783 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.220008850097656e-05 seconds
INFO 05-06 11:02:20.027105.027105 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008683204650878906 seconds
INFO 05-06 11:02:20.061483.061483 lmp.py:1387] [layer_moe_fused] to time: 0.00014710426330566406 seconds
INFO 05-06 11:02:20.061193.061193 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.063152.063152 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012848377227783203 seconds
INFO 05-06 11:02:20.063917.063917 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007951259613037109 seconds
INFO 05-06 11:02:20.064389.064389 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002241373062133789 seconds
INFO 05-06 11:02:20.094126.094126 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.030869483947753906 seconds
INFO 05-06 11:02:20.097867.097867 mlpmodule.py:2799] [fused_experts] gmm total=2.291ms E=32 S=3438 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.097621.097621 mlpmodule.py:2799] [fused_experts] gmm total=2.274ms E=32 S=4000 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.097494.097494 mlpmodule.py:2799] [fused_experts] gmm total=2.470ms E=32 S=3973 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.098482.098482 mlpmodule.py:2799] [fused_experts] gmm total=2.996ms E=32 S=4973 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.099393.099393 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004704713821411133 seconds
INFO 05-06 11:02:20.099901.099901 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.507469177246094e-05 seconds
DEBUG 05-06 11:02:20.100889.100889 cuda_h.py:27] end *layer_moe_fused cost 75.068 ms
DEBUG 05-06 11:02:20.122023.122023 cuda_h.py:27] end prefill_layer cost 99.684 ms
DEBUG 05-06 11:02:20.122449.122449 lmp.py:841] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 11:02:20.122013.122013 lmp.py:824] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 11:02:20.124461.124461 cuda_h.py:27] end *sagl cost 1.701 ms
experts_cpu_alloc {'expert_ids': [35, 127, 95, 31, 79, 103, 107, 15, 75, 71, 91, 87, 123, 67, 47, 14, 70, 110, 50, 58, 66, 122, 46, 114, 126, 18, 34, 38, 90, 100, 16, 48, 12, 72, 56, 44, 120, 80, 88, 84, 40, 36, 52, 64, 13, 121, 41, 69, 21, 37, 77, 45, 25, 101, 81, 57, 73, 109, 117], 'token_total': 1150, 'token_per_expert': {35: 1, 127: 1, 95: 2, 31: 4, 79: 5, 103: 18, 107: 30, 15: 31, 75: 41, 71: 48, 91: 49, 87: 50, 123: 58, 67: 92, 47: 101, 14: 1, 70: 1, 110: 3, 50: 5, 58: 5, 66: 7, 122: 7, 46: 8, 114: 8, 126: 10, 18: 11, 34: 13, 38: 17, 90: 19, 100: 1, 16: 2, 48: 2, 12: 4, 72: 6, 56: 9, 44: 12, 120: 12, 80: 18, 88: 23, 84: 24, 40: 25, 36: 27, 52: 30, 64: 33, 13: 1, 121: 3, 41: 7, 69: 8, 21: 12, 37: 14, 77: 22, 45: 23, 25: 24, 101: 24, 81: 26, 57: 27, 73: 27, 109: 27, 117: 31}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 39, 43, 51, 55, 59, 63, 83, 111, 115, 119], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 5002, 'token_per_expert': {3: 1119, 7: 1041, 19: 106, 23: 237, 27: 136, 39: 124, 43: 282, 51: 162, 55: 121, 59: 336, 63: 523, 83: 189, 111: 187, 115: 196, 119: 243}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 22, 26, 30, 54, 62, 74, 78, 82, 86, 94, 98, 106, 118], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3525, 'token_per_expert': {2: 1028, 6: 1036, 22: 196, 26: 166, 30: 47, 54: 170, 62: 43, 74: 207, 78: 41, 82: 140, 86: 60, 94: 54, 98: 34, 106: 266, 118: 37}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 20, 24, 28, 32, 60, 76, 92, 96, 104, 108, 116, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3383, 'token_per_expert': {0: 1024, 4: 1093, 8: 360, 20: 73, 24: 158, 28: 61, 32: 96, 60: 56, 76: 99, 92: 65, 96: 65, 104: 67, 108: 44, 116: 45, 124: 77}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 17, 29, 49, 53, 61, 85, 89, 93, 97, 105, 113, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 3324, 'token_per_expert': {1: 1170, 5: 1101, 17: 34, 29: 88, 49: 43, 53: 127, 61: 47, 85: 72, 89: 211, 93: 96, 97: 65, 105: 52, 113: 153, 125: 65}}
INFO 05-06 11:02:20.126579.126579 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 1.314ms | allocate_experts_across_cpu_gpu: 0.266ms
INFO 05-06 11:02:20.126774.126774 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.291534423828125e-05 seconds
INFO 05-06 11:02:20.127182.127182 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008625984191894531 seconds
INFO 05-06 11:02:20.163759.163759 lmp.py:1387] [layer_moe_fused] to time: 0.0001678466796875 seconds
INFO 05-06 11:02:20.163450.163450 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.165199.165199 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013358592987060547 seconds
INFO 05-06 11:02:20.165733.165733 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006284713745117188 seconds
INFO 05-06 11:02:20.165529.165529 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0021152496337890625 seconds
INFO 05-06 11:02:20.191421.191421 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0257871150970459 seconds
INFO 05-06 11:02:20.194659.194659 mlpmodule.py:2799] [fused_experts] gmm total=2.179ms E=32 S=3611 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.194928.194928 mlpmodule.py:2799] [fused_experts] gmm total=2.256ms E=32 S=3600 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.194984.194984 mlpmodule.py:2799] [fused_experts] gmm total=2.441ms E=32 S=3640 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.194314.194314 mlpmodule.py:2799] [fused_experts] gmm total=2.682ms E=32 S=5533 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.198726.198726 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0062389373779296875 seconds
INFO 05-06 11:02:20.198479.198479 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.7220458984375e-05 seconds
DEBUG 05-06 11:02:20.198106.198106 cuda_h.py:27] end *layer_moe_fused cost 73.702 ms
DEBUG 05-06 11:02:20.214618.214618 cuda_h.py:27] end prefill_layer cost 92.044 ms
DEBUG 05-06 11:02:20.214806.214806 lmp.py:841] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 11:02:20.214416.214416 lmp.py:824] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 11:02:20.218380.218380 cuda_h.py:27] end *sagl cost 3.733 ms
experts_cpu_alloc {'expert_ids': [11, 95, 103, 15, 51, 27, 115, 19, 107, 79, 31, 119, 83, 66, 82, 110, 38, 78, 50, 122, 30, 34, 62, 26, 58, 86, 102, 54, 98, 10, 8, 124, 48, 32, 56, 92, 68, 84, 52, 100, 44, 96, 76, 80, 120, 45, 69, 21, 97, 17, 41, 53, 81, 105, 57, 37, 77], 'token_total': 842, 'token_per_expert': {11: 1, 95: 1, 103: 1, 15: 3, 51: 5, 27: 8, 115: 8, 19: 14, 107: 24, 79: 26, 31: 28, 119: 28, 83: 29, 66: 1, 82: 2, 110: 4, 38: 5, 78: 5, 50: 7, 122: 7, 30: 8, 34: 9, 62: 9, 26: 11, 58: 12, 86: 12, 102: 14, 54: 17, 98: 20, 10: 22, 8: 2, 124: 2, 48: 3, 32: 4, 56: 6, 92: 8, 68: 21, 84: 22, 52: 28, 100: 41, 44: 43, 96: 49, 76: 60, 80: 66, 120: 70, 45: 1, 69: 1, 21: 2, 97: 3, 17: 4, 41: 5, 53: 6, 81: 7, 105: 8, 57: 9, 37: 13, 77: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 43, 55, 63, 67, 71, 75, 87, 99, 111, 123, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 3968, 'token_per_expert': {3: 1025, 7: 1037, 23: 85, 39: 277, 43: 62, 55: 34, 63: 57, 67: 53, 71: 556, 75: 63, 87: 99, 99: 129, 111: 162, 123: 120, 127: 209}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 18, 22, 42, 46, 70, 74, 94, 106, 114, 118, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 31, 'token_total': 3197, 'token_per_expert': {2: 1252, 6: 1043, 14: 28, 18: 35, 22: 203, 42: 142, 46: 73, 70: 89, 74: 68, 94: 88, 106: 33, 114: 24, 118: 51, 126: 68}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 16, 20, 24, 28, 36, 60, 64, 72, 88, 104, 112, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 4123, 'token_per_expert': {0: 1067, 4: 1130, 16: 241, 20: 321, 24: 99, 28: 119, 36: 175, 60: 73, 64: 213, 72: 127, 88: 128, 104: 98, 112: 253, 116: 79}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 29, 33, 49, 61, 73, 93, 101, 113, 117, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 26, 'token_total': 4254, 'token_per_expert': {1: 1026, 5: 1082, 9: 109, 13: 119, 29: 84, 33: 245, 49: 330, 61: 180, 73: 111, 93: 89, 101: 676, 113: 32, 117: 127, 125: 44}}
INFO 05-06 11:02:20.222141.222141 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 2.726ms | allocate_experts_across_cpu_gpu: 0.261ms
INFO 05-06 11:02:20.222283.222283 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.933906555175781e-05 seconds
INFO 05-06 11:02:20.223268.223268 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0009014606475830078 seconds
INFO 05-06 11:02:20.259693.259693 lmp.py:1387] [layer_moe_fused] to time: 0.00015211105346679688 seconds
INFO 05-06 11:02:20.259582.259582 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.261727.261727 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017042160034179688 seconds
INFO 05-06 11:02:20.262969.262969 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006260871887207031 seconds
INFO 05-06 11:02:20.262826.262826 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002487659454345703 seconds
INFO 05-06 11:02:20.291357.291357 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.029066801071166992 seconds
INFO 05-06 11:02:20.293522.293522 mlpmodule.py:2799] [fused_experts] gmm total=1.915ms E=32 S=4330 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.294937.294937 mlpmodule.py:2799] [fused_experts] gmm total=2.230ms E=32 S=4144 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.294173.294173 mlpmodule.py:2799] [fused_experts] gmm total=2.289ms E=32 S=3362 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.294428.294428 mlpmodule.py:2799] [fused_experts] gmm total=2.706ms E=32 S=4548 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.296610.296610 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004678249359130859 seconds
INFO 05-06 11:02:20.296402.296402 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.4836273193359375e-05 seconds
DEBUG 05-06 11:02:20.297361.297361 cuda_h.py:27] end *layer_moe_fused cost 77.587 ms
DEBUG 05-06 11:02:20.317822.317822 cuda_h.py:27] end prefill_layer cost 102.289 ms
DEBUG 05-06 11:02:20.317586.317586 lmp.py:841] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 11:02:20.317719.317719 lmp.py:824] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 11:02:20.318378.318378 cuda_h.py:27] end *sagl cost 1.647 ms
experts_cpu_alloc {'expert_ids': [55, 59, 31, 15, 83, 47, 67, 111, 91, 11, 19, 43, 103, 127, 95, 66, 38, 22, 114, 18, 74, 82, 30, 110, 126, 58, 42, 14, 70, 122, 88, 8, 92, 52, 112, 124, 40, 72, 16, 20, 120, 60, 76, 33, 49, 17, 81, 97, 21, 29, 109, 101, 37, 57, 41, 77, 89, 85, 105], 'token_total': 852, 'token_per_expert': {55: 1, 59: 3, 31: 4, 15: 5, 83: 5, 47: 6, 67: 6, 111: 9, 91: 15, 11: 16, 19: 17, 43: 20, 103: 24, 127: 26, 95: 40, 66: 1, 38: 5, 22: 6, 114: 8, 18: 14, 74: 15, 82: 17, 30: 23, 110: 27, 126: 29, 58: 34, 42: 35, 14: 36, 70: 37, 122: 42, 88: 1, 8: 2, 92: 3, 52: 4, 112: 4, 124: 4, 40: 5, 72: 6, 16: 8, 20: 8, 120: 11, 60: 17, 76: 22, 33: 2, 49: 2, 17: 3, 81: 4, 97: 4, 21: 5, 29: 5, 109: 6, 101: 8, 37: 17, 57: 19, 41: 24, 77: 30, 89: 31, 85: 34, 105: 37}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 35, 51, 71, 75, 79, 87, 99, 107, 115, 119, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3846, 'token_per_expert': {3: 1039, 7: 1025, 23: 155, 27: 52, 35: 270, 51: 67, 71: 70, 75: 97, 79: 86, 87: 195, 99: 421, 107: 70, 115: 164, 119: 71, 123: 64}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 26, 34, 46, 50, 62, 78, 86, 90, 94, 98, 102, 106], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4052, 'token_per_expert': {2: 1071, 6: 1082, 10: 52, 26: 68, 34: 192, 46: 96, 50: 76, 62: 79, 78: 102, 86: 258, 90: 205, 94: 145, 98: 146, 102: 278, 106: 202}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 24, 28, 32, 36, 44, 56, 64, 68, 80, 96, 104, 108, 116], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 3805, 'token_per_expert': {0: 1056, 4: 1030, 24: 73, 28: 34, 32: 74, 36: 70, 44: 46, 56: 43, 64: 260, 68: 648, 80: 22, 96: 85, 104: 116, 108: 212, 116: 36}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 25, 53, 65, 69, 73, 93, 113, 117, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 3829, 'token_per_expert': {1: 1079, 5: 1063, 9: 85, 13: 146, 25: 401, 53: 202, 65: 163, 69: 50, 73: 44, 93: 338, 113: 66, 117: 61, 121: 93, 125: 38}}
INFO 05-06 11:02:20.321768.321768 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 1.331ms | allocate_experts_across_cpu_gpu: 0.269ms
INFO 05-06 11:02:20.321448.321448 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.649162292480469e-05 seconds
INFO 05-06 11:02:20.322286.322286 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008640289306640625 seconds
INFO 05-06 11:02:20.364267.364267 lmp.py:1387] [layer_moe_fused] to time: 0.0001461505889892578 seconds
INFO 05-06 11:02:20.365593.365593 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.366699.366699 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013556480407714844 seconds
INFO 05-06 11:02:20.367010.367010 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006475448608398438 seconds
INFO 05-06 11:02:20.367774.367774 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0021953582763671875 seconds
INFO 05-06 11:02:20.395207.395207 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.028326749801635742 seconds
INFO 05-06 11:02:20.398676.398676 mlpmodule.py:2799] [fused_experts] gmm total=2.096ms E=32 S=4043 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.398173.398173 mlpmodule.py:2799] [fused_experts] gmm total=2.223ms E=32 S=4381 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.398792.398792 mlpmodule.py:2799] [fused_experts] gmm total=2.262ms E=32 S=3900 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.398474.398474 mlpmodule.py:2799] [fused_experts] gmm total=2.393ms E=32 S=4060 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.399769.399769 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003833293914794922 seconds
INFO 05-06 11:02:20.399839.399839 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.2928924560546875e-05 seconds
DEBUG 05-06 11:02:20.400319.400319 cuda_h.py:27] end *layer_moe_fused cost 80.726 ms
DEBUG 05-06 11:02:20.422011.422011 cuda_h.py:27] end prefill_layer cost 105.436 ms
DEBUG 05-06 11:02:20.422967.422967 lmp.py:841] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 11:02:20.422054.422054 lmp.py:824] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 11:02:20.424165.424165 cuda_h.py:27] end *sagl cost 1.698 ms
experts_cpu_alloc {'expert_ids': [11, 39, 75, 119, 35, 31, 67, 107, 127, 55, 63, 15, 111, 23, 95, 123, 58, 50, 30, 38, 62, 102, 78, 94, 26, 82, 126, 54, 66, 118, 122, 124, 24, 100, 92, 36, 32, 16, 88, 116, 80, 64, 68, 8, 112, 72, 37, 49, 73, 109, 77, 41, 17, 21, 101, 25, 45, 9, 117, 33], 'token_total': 1183, 'token_per_expert': {11: 1, 39: 2, 75: 2, 119: 2, 35: 6, 31: 8, 67: 9, 107: 9, 127: 12, 55: 13, 63: 14, 15: 18, 111: 19, 23: 22, 95: 41, 123: 41, 58: 3, 50: 4, 30: 5, 38: 6, 62: 6, 102: 6, 78: 9, 94: 10, 26: 11, 82: 21, 126: 22, 54: 31, 66: 34, 118: 34, 122: 38, 124: 2, 24: 3, 100: 3, 92: 4, 36: 5, 32: 19, 16: 20, 88: 24, 116: 25, 80: 28, 64: 39, 68: 44, 8: 49, 112: 55, 72: 66, 37: 7, 49: 9, 73: 12, 109: 14, 77: 15, 41: 22, 17: 23, 21: 24, 101: 25, 25: 27, 45: 31, 9: 41, 117: 43, 33: 45}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 43, 47, 51, 59, 71, 79, 83, 87, 91, 99, 103, 115], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3306, 'token_per_expert': {3: 1026, 7: 1117, 19: 43, 43: 68, 47: 59, 51: 53, 59: 67, 71: 65, 79: 117, 83: 55, 87: 45, 91: 436, 99: 43, 103: 53, 115: 59}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 14, 18, 22, 34, 42, 70, 86, 90, 98, 106, 110, 114], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3536, 'token_per_expert': {2: 1024, 6: 1062, 10: 138, 14: 119, 18: 61, 22: 42, 34: 158, 42: 106, 70: 186, 86: 111, 90: 142, 98: 39, 106: 104, 110: 117, 114: 127}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 12, 20, 28, 44, 48, 52, 56, 60, 84, 96, 104, 108, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4120, 'token_per_expert': {0: 1048, 4: 1224, 12: 222, 20: 161, 28: 122, 44: 137, 48: 102, 52: 187, 56: 86, 60: 100, 84: 137, 96: 78, 104: 91, 108: 278, 120: 147}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 29, 53, 57, 61, 65, 69, 85, 97, 105, 113, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4239, 'token_per_expert': {1: 1025, 5: 1116, 13: 68, 29: 235, 53: 119, 57: 94, 61: 65, 65: 132, 69: 176, 85: 162, 97: 522, 105: 63, 113: 116, 121: 284, 125: 62}}
INFO 05-06 11:02:20.426878.426878 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.890ms | allocate_experts_across_cpu_gpu: 0.276ms
INFO 05-06 11:02:20.426697.426697 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.410743713378906e-05 seconds
INFO 05-06 11:02:20.427130.427130 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008363723754882812 seconds
INFO 05-06 11:02:20.466432.466432 lmp.py:1387] [layer_moe_fused] to time: 0.00014543533325195312 seconds
INFO 05-06 11:02:20.466049.466049 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.468910.468910 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013158321380615234 seconds
INFO 05-06 11:02:20.468410.468410 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006012916564941406 seconds
INFO 05-06 11:02:20.468922.468922 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0020711421966552734 seconds
INFO 05-06 11:02:20.498404.498404 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.029980897903442383 seconds
INFO 05-06 11:02:20.501975.501975 mlpmodule.py:2799] [fused_experts] gmm total=2.312ms E=32 S=3776 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.501860.501860 mlpmodule.py:2799] [fused_experts] gmm total=2.507ms E=32 S=3525 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.501777.501777 mlpmodule.py:2799] [fused_experts] gmm total=2.454ms E=32 S=4506 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.501051.501051 mlpmodule.py:2799] [fused_experts] gmm total=2.684ms E=32 S=4577 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.503499.503499 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004127979278564453 seconds
INFO 05-06 11:02:20.503893.503893 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.054473876953125e-05 seconds
DEBUG 05-06 11:02:20.503660.503660 cuda_h.py:27] end *layer_moe_fused cost 78.495 ms
DEBUG 05-06 11:02:20.525429.525429 cuda_h.py:27] end prefill_layer cost 102.548 ms
DEBUG 05-06 11:02:20.525524.525524 lmp.py:841] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 11:02:20.525373.525373 lmp.py:824] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 11:02:20.527033.527033 cuda_h.py:27] end *sagl cost 1.682 ms
experts_cpu_alloc {'expert_ids': [59, 83, 23, 107, 35, 39, 79, 99, 119, 91, 43, 11, 47, 127, 90, 18, 26, 118, 74, 106, 82, 34, 62, 86, 66, 14, 10, 22, 60, 112, 40, 100, 24, 72, 104, 96, 48, 92, 8, 64, 68, 84, 116, 124, 101, 117, 9, 25, 33, 13, 37, 49, 57, 17, 113, 89, 45, 77, 21], 'token_total': 1022, 'token_per_expert': {59: 1, 83: 1, 23: 2, 107: 3, 35: 7, 39: 7, 79: 7, 99: 13, 119: 13, 91: 16, 43: 17, 11: 30, 47: 37, 127: 38, 90: 4, 18: 5, 26: 7, 118: 9, 74: 14, 106: 14, 82: 15, 34: 16, 62: 18, 86: 22, 66: 35, 14: 36, 10: 38, 22: 39, 60: 1, 112: 1, 40: 2, 100: 3, 24: 6, 72: 6, 104: 8, 96: 9, 48: 12, 92: 13, 8: 14, 64: 15, 68: 16, 84: 21, 116: 26, 124: 32, 101: 8, 117: 10, 9: 13, 25: 14, 33: 15, 13: 16, 37: 17, 49: 17, 57: 32, 17: 34, 113: 34, 89: 37, 45: 41, 77: 42, 21: 43}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 31, 51, 55, 63, 71, 75, 87, 103, 111, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3949, 'token_per_expert': {3: 1099, 7: 1029, 15: 112, 19: 141, 27: 100, 31: 55, 51: 319, 55: 85, 63: 72, 71: 128, 75: 143, 87: 195, 103: 338, 111: 40, 123: 93}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 38, 42, 46, 50, 54, 58, 70, 98, 102, 110, 114, 122, 126], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4456, 'token_per_expert': {2: 1137, 6: 1096, 38: 103, 42: 46, 46: 131, 50: 190, 54: 439, 58: 455, 70: 204, 98: 56, 102: 121, 110: 184, 114: 164, 122: 77, 126: 53}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 12, 16, 20, 28, 32, 36, 44, 52, 56, 76, 80, 108, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3334, 'token_per_expert': {0: 1027, 4: 1051, 12: 94, 16: 47, 20: 39, 28: 218, 32: 102, 36: 99, 44: 58, 52: 49, 56: 121, 76: 70, 80: 126, 108: 33, 120: 200}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 29, 41, 53, 61, 65, 69, 73, 81, 85, 93, 105, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3623, 'token_per_expert': {1: 1050, 5: 1112, 29: 59, 41: 55, 53: 56, 61: 57, 65: 133, 69: 78, 73: 268, 81: 89, 85: 57, 93: 64, 105: 168, 121: 204, 125: 173}}
INFO 05-06 11:02:20.529877.529877 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 1.259ms | allocate_experts_across_cpu_gpu: 0.262ms
INFO 05-06 11:02:20.529934.529934 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.3392181396484375e-05 seconds
INFO 05-06 11:02:20.530507.530507 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008440017700195312 seconds
INFO 05-06 11:02:20.566923.566923 lmp.py:1387] [layer_moe_fused] to time: 0.00014781951904296875 seconds
INFO 05-06 11:02:20.566527.566527 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.568129.568129 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001302480697631836 seconds
INFO 05-06 11:02:20.568992.568992 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005869865417480469 seconds
INFO 05-06 11:02:20.568173.568173 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0020465850830078125 seconds
INFO 05-06 11:02:20.596431.596431 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02781224250793457 seconds
INFO 05-06 11:02:20.599170.599170 mlpmodule.py:2799] [fused_experts] gmm total=2.127ms E=32 S=3519 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.599430.599430 mlpmodule.py:2799] [fused_experts] gmm total=2.294ms E=32 S=3996 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.599903.599903 mlpmodule.py:2799] [fused_experts] gmm total=2.586ms E=32 S=4141 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.599767.599767 mlpmodule.py:2799] [fused_experts] gmm total=2.591ms E=32 S=4728 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.600506.600506 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003932476043701172 seconds
INFO 05-06 11:02:20.600291.600291 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.269050598144531e-05 seconds
DEBUG 05-06 11:02:20.601809.601809 cuda_h.py:27] end *layer_moe_fused cost 73.662 ms
DEBUG 05-06 11:02:20.623990.623990 cuda_h.py:27] end prefill_layer cost 97.742 ms
DEBUG 05-06 11:02:20.623747.623747 lmp.py:841] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 11:02:20.623119.623119 lmp.py:824] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 11:02:20.624388.624388 cuda_h.py:27] end *sagl cost 1.676 ms
experts_cpu_alloc {'expert_ids': [47, 63, 11, 31, 123, 119, 55, 79, 67, 115, 15, 99, 14, 126, 118, 94, 50, 110, 66, 90, 26, 34, 114, 58, 18, 82, 10, 98, 108, 84, 100, 64, 112, 120, 28, 96, 52, 104, 20, 44, 116, 8, 80, 124, 25, 53, 121, 29, 41, 33, 77, 105, 73, 113, 117, 97, 17, 45], 'token_total': 887, 'token_per_expert': {47: 1, 63: 1, 11: 3, 31: 4, 123: 4, 119: 8, 55: 12, 79: 12, 67: 18, 115: 26, 15: 52, 99: 57, 14: 1, 126: 1, 118: 2, 94: 4, 50: 6, 110: 7, 66: 8, 90: 9, 26: 11, 34: 14, 114: 17, 58: 18, 18: 22, 82: 26, 10: 27, 98: 30, 108: 1, 84: 2, 100: 5, 64: 8, 112: 9, 120: 10, 28: 11, 96: 11, 52: 16, 104: 17, 20: 20, 44: 20, 116: 21, 8: 24, 80: 30, 124: 37, 25: 1, 53: 1, 121: 5, 29: 7, 41: 10, 33: 12, 77: 12, 105: 20, 73: 22, 113: 22, 117: 25, 97: 31, 17: 37, 45: 39}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 39, 43, 51, 71, 75, 83, 95, 103, 111, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 27, 'token_total': 4110, 'token_per_expert': {3: 1054, 7: 1058, 19: 62, 23: 105, 27: 65, 39: 73, 43: 275, 51: 80, 71: 82, 75: 214, 83: 62, 95: 566, 103: 261, 111: 63, 127: 90}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 22, 30, 38, 42, 46, 54, 62, 70, 74, 86, 102, 106, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4054, 'token_per_expert': {2: 1025, 6: 1033, 22: 95, 30: 76, 38: 70, 42: 36, 46: 401, 54: 54, 62: 43, 70: 369, 74: 210, 86: 48, 102: 105, 106: 434, 122: 55}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 12, 16, 24, 32, 36, 40, 48, 56, 68, 72, 76, 88, 92], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3699, 'token_per_expert': {0: 1030, 4: 1118, 12: 348, 16: 216, 24: 74, 32: 88, 36: 92, 40: 101, 48: 129, 56: 141, 68: 57, 72: 71, 76: 83, 88: 48, 92: 103}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 21, 37, 57, 61, 69, 81, 89, 93, 101, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 3634, 'token_per_expert': {1: 1122, 5: 1052, 9: 50, 13: 100, 21: 79, 37: 40, 57: 121, 61: 90, 69: 157, 81: 162, 89: 86, 93: 226, 101: 277, 125: 72}}
INFO 05-06 11:02:20.627323.627323 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.971ms | allocate_experts_across_cpu_gpu: 0.282ms
INFO 05-06 11:02:20.627095.627095 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.291534423828125e-05 seconds
INFO 05-06 11:02:20.628457.628457 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008997917175292969 seconds
INFO 05-06 11:02:20.668904.668904 lmp.py:1387] [layer_moe_fused] to time: 0.0001575946807861328 seconds
INFO 05-06 11:02:20.668853.668853 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.670183.670183 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001313924789428711 seconds
INFO 05-06 11:02:20.670162.670162 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006659030914306641 seconds
INFO 05-06 11:02:20.670808.670808 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002168416976928711 seconds
INFO 05-06 11:02:20.691574.691574 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.020075082778930664 seconds
INFO 05-06 11:02:20.693415.693415 mlpmodule.py:2799] [fused_experts] gmm total=2.123ms E=32 S=4308 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.693830.693830 mlpmodule.py:2799] [fused_experts] gmm total=2.188ms E=32 S=4257 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.693250.693250 mlpmodule.py:2799] [fused_experts] gmm total=2.197ms E=32 S=3941 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.693000.693000 mlpmodule.py:2799] [fused_experts] gmm total=2.203ms E=32 S=3878 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.694468.694468 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0036618709564208984 seconds
INFO 05-06 11:02:20.694584.694584 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.3882598876953125e-05 seconds
DEBUG 05-06 11:02:20.695118.695118 cuda_h.py:27] end *layer_moe_fused cost 69.711 ms
DEBUG 05-06 11:02:20.716991.716991 cuda_h.py:27] end prefill_layer cost 92.870 ms
DEBUG 05-06 11:02:20.716398.716398 lmp.py:841] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 11:02:20.716737.716737 lmp.py:824] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 11:02:20.718317.718317 cuda_h.py:27] end *sagl cost 1.925 ms
experts_cpu_alloc {'expert_ids': [23, 123, 35, 55, 51, 87, 91, 107, 59, 15, 103, 27, 67, 111, 119, 11, 30, 114, 122, 38, 66, 102, 26, 70, 78, 50, 34, 98, 90, 94, 24, 36, 32, 40, 48, 104, 12, 124, 120, 64, 28, 56, 112, 44, 17, 65, 77, 101, 53, 25, 109, 33, 61, 9, 29, 73, 97, 93, 89, 117], 'token_total': 822, 'token_per_expert': {23: 1, 123: 1, 35: 2, 55: 2, 51: 3, 87: 5, 91: 6, 107: 6, 59: 7, 15: 9, 103: 9, 27: 16, 67: 23, 111: 24, 119: 27, 11: 38, 30: 1, 114: 1, 122: 1, 38: 3, 66: 6, 102: 9, 26: 11, 70: 14, 78: 15, 50: 25, 34: 27, 98: 28, 90: 36, 94: 67, 24: 1, 36: 1, 32: 2, 40: 3, 48: 4, 104: 4, 12: 8, 124: 8, 120: 11, 64: 13, 28: 20, 56: 23, 112: 28, 44: 30, 17: 1, 65: 2, 77: 4, 101: 4, 53: 5, 25: 7, 109: 11, 33: 14, 61: 16, 9: 20, 29: 20, 73: 20, 97: 23, 93: 28, 89: 32, 117: 36}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 39, 43, 47, 63, 71, 75, 79, 83, 99, 115, 127], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3211, 'token_per_expert': {3: 1031, 7: 1097, 19: 39, 31: 79, 39: 116, 43: 67, 47: 83, 63: 61, 71: 96, 75: 99, 79: 39, 83: 51, 99: 64, 115: 182, 127: 107}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 14, 18, 42, 46, 54, 58, 62, 74, 82, 86, 106, 126], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4007, 'token_per_expert': {2: 1029, 6: 1052, 10: 97, 14: 217, 18: 93, 42: 121, 46: 101, 54: 77, 58: 70, 62: 160, 74: 262, 82: 121, 86: 290, 106: 187, 126: 130}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 16, 20, 60, 68, 72, 76, 80, 84, 88, 92, 100, 108], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4429, 'token_per_expert': {0: 1207, 4: 1058, 8: 275, 16: 107, 20: 72, 60: 275, 68: 89, 72: 60, 76: 399, 80: 383, 84: 52, 88: 131, 92: 103, 100: 64, 108: 154}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 21, 37, 41, 49, 57, 69, 81, 85, 105, 113, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3915, 'token_per_expert': {1: 1444, 5: 1068, 13: 83, 21: 131, 37: 50, 41: 136, 49: 115, 57: 104, 69: 47, 81: 316, 85: 78, 105: 50, 113: 84, 121: 47, 125: 162}}
INFO 05-06 11:02:20.720776.720776 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 1.180ms | allocate_experts_across_cpu_gpu: 0.286ms
INFO 05-06 11:02:20.721555.721555 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.57763671875e-05 seconds
INFO 05-06 11:02:20.722552.722552 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008766651153564453 seconds
INFO 05-06 11:02:20.755943.755943 lmp.py:1387] [layer_moe_fused] to time: 0.00015735626220703125 seconds
INFO 05-06 11:02:20.755700.755700 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.759767.759767 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0033295154571533203 seconds
INFO 05-06 11:02:20.759484.759484 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005853176116943359 seconds
INFO 05-06 11:02:20.759142.759142 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.004072904586791992 seconds
INFO 05-06 11:02:20.769170.769170 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010087251663208008 seconds
INFO 05-06 11:02:20.772122.772122 mlpmodule.py:2799] [fused_experts] gmm total=1.992ms E=32 S=4251 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.772524.772524 mlpmodule.py:2799] [fused_experts] gmm total=2.217ms E=32 S=3390 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.772634.772634 mlpmodule.py:2799] [fused_experts] gmm total=2.154ms E=32 S=4158 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.772697.772697 mlpmodule.py:2799] [fused_experts] gmm total=2.287ms E=32 S=4585 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.773910.773910 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003615856170654297 seconds
INFO 05-06 11:02:20.773265.773265 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.507469177246094e-05 seconds
DEBUG 05-06 11:02:20.774298.774298 cuda_h.py:27] end *layer_moe_fused cost 55.045 ms
DEBUG 05-06 11:02:20.780717.780717 cuda_h.py:27] end prefill_layer cost 64.266 ms
DEBUG 05-06 11:02:20.780302.780302 lmp.py:841] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 11:02:20.780972.780972 lmp.py:824] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 11:02:20.783859.783859 cuda_h.py:27] end *sagl cost 2.786 ms
experts_cpu_alloc {'expert_ids': [15, 47, 35, 115, 127, 63, 39, 11, 51, 59, 123, 71, 90, 26, 114, 106, 110, 94, 118, 58, 22, 122, 74, 34, 42, 50, 62, 12, 96, 84, 88, 72, 52, 64, 44, 80, 8, 40, 120, 48, 124, 36, 28, 45, 105, 21, 85, 9, 13, 53, 125, 97, 33, 121, 29, 117], 'token_total': 974, 'token_per_expert': {15: 1, 47: 3, 35: 4, 115: 5, 127: 5, 63: 6, 39: 17, 11: 22, 51: 30, 59: 31, 123: 33, 71: 36, 90: 1, 26: 3, 114: 3, 106: 6, 110: 6, 94: 7, 118: 9, 58: 11, 22: 12, 122: 13, 74: 14, 34: 15, 42: 19, 50: 20, 62: 25, 12: 2, 96: 4, 84: 5, 88: 5, 72: 7, 52: 17, 64: 19, 44: 24, 80: 26, 8: 27, 40: 27, 120: 32, 48: 35, 124: 51, 36: 60, 28: 62, 45: 1, 105: 1, 21: 6, 85: 7, 9: 8, 13: 8, 53: 8, 125: 8, 97: 9, 33: 12, 121: 44, 29: 50, 117: 52}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 31, 43, 67, 79, 83, 87, 91, 99, 111, 119], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 27, 'token_total': 4203, 'token_per_expert': {3: 1040, 7: 1206, 19: 131, 23: 152, 27: 44, 31: 105, 43: 50, 67: 173, 79: 275, 83: 264, 87: 338, 91: 63, 99: 60, 111: 182, 119: 120}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 18, 30, 38, 46, 54, 66, 70, 82, 98, 102, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 3315, 'token_per_expert': {2: 1126, 6: 1202, 10: 89, 18: 33, 30: 106, 38: 64, 46: 27, 54: 39, 66: 77, 70: 63, 82: 45, 98: 49, 102: 352, 126: 43}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 16, 20, 24, 32, 56, 68, 76, 92, 100, 108, 112, 116], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 30, 'token_total': 3980, 'token_per_expert': {0: 1031, 4: 1036, 16: 367, 20: 73, 24: 99, 32: 149, 56: 362, 68: 146, 76: 115, 92: 236, 100: 139, 108: 94, 112: 70, 116: 63}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 17, 25, 37, 49, 57, 61, 69, 77, 81, 89, 93, 113], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 3912, 'token_per_expert': {1: 1051, 5: 1057, 17: 236, 25: 55, 37: 80, 49: 200, 57: 98, 61: 76, 69: 63, 77: 79, 81: 266, 89: 60, 93: 291, 113: 300}}
INFO 05-06 11:02:20.787142.787142 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 2.720ms | allocate_experts_across_cpu_gpu: 0.439ms
INFO 05-06 11:02:20.788723.788723 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.818771362304688e-05 seconds
INFO 05-06 11:02:20.789174.789174 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.001094818115234375 seconds
INFO 05-06 11:02:20.802868.802868 lmp.py:1387] [layer_moe_fused] to time: 0.00013017654418945312 seconds
INFO 05-06 11:02:20.802180.802180 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.803586.803586 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011949539184570312 seconds
INFO 05-06 11:02:20.804269.804269 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005650520324707031 seconds
INFO 05-06 11:02:20.804020.804020 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019087791442871094 seconds
INFO 05-06 11:02:20.813920.813920 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009431600570678711 seconds
INFO 05-06 11:02:20.816874.816874 mlpmodule.py:2799] [fused_experts] gmm total=1.773ms E=32 S=4126 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.816991.816991 mlpmodule.py:2799] [fused_experts] gmm total=1.981ms E=32 S=4383 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.816062.816062 mlpmodule.py:2799] [fused_experts] gmm total=2.157ms E=32 S=3479 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.816561.816561 mlpmodule.py:2799] [fused_experts] gmm total=2.305ms E=32 S=4396 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.817364.817364 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003470182418823242 seconds
INFO 05-06 11:02:20.817017.817017 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.4836273193359375e-05 seconds
DEBUG 05-06 11:02:20.818743.818743 cuda_h.py:27] end *layer_moe_fused cost 33.554 ms
DEBUG 05-06 11:02:20.824081.824081 cuda_h.py:27] end prefill_layer cost 43.399 ms
DEBUG 05-06 11:02:20.824806.824806 lmp.py:841] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 11:02:20.824714.824714 lmp.py:824] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 11:02:20.826133.826133 cuda_h.py:27] end *sagl cost 2.002 ms
experts_cpu_alloc {'expert_ids': [27, 51, 83, 59, 31, 127, 67, 47, 111, 119, 63, 123, 79, 10, 122, 54, 62, 18, 102, 70, 38, 94, 98, 90, 22, 58, 48, 56, 64, 20, 8, 32, 120, 24, 112, 12, 68, 40, 69, 81, 93, 41, 37, 105, 113, 13, 33, 17, 89, 125, 65], 'token_total': 734, 'token_per_expert': {27: 1, 51: 1, 83: 1, 59: 2, 31: 6, 127: 6, 67: 7, 47: 11, 111: 12, 119: 17, 63: 18, 123: 19, 79: 23, 10: 1, 122: 1, 54: 2, 62: 2, 18: 4, 102: 14, 70: 19, 38: 22, 94: 25, 98: 29, 90: 36, 22: 52, 58: 55, 48: 2, 56: 5, 64: 6, 20: 7, 8: 8, 32: 10, 120: 11, 24: 13, 112: 24, 12: 37, 68: 38, 40: 47, 69: 2, 81: 4, 93: 4, 41: 5, 37: 6, 105: 6, 113: 8, 13: 10, 33: 11, 17: 13, 89: 21, 125: 21, 65: 29}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 35, 39, 71, 91, 95, 103, 107, 115], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 3644, 'token_per_expert': {3: 1092, 7: 1024, 15: 233, 19: 95, 23: 108, 35: 64, 39: 400, 71: 300, 91: 92, 95: 85, 103: 33, 107: 23, 115: 95}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 34, 46, 50, 74, 78, 82, 86, 106, 110, 114, 118], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 4782, 'token_per_expert': {2: 1027, 6: 1117, 34: 62, 46: 175, 50: 312, 74: 204, 78: 555, 82: 237, 86: 259, 106: 152, 110: 235, 114: 271, 118: 176}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 36, 76, 80, 84, 88, 92, 100, 104, 108, 116, 124], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 25, 'token_total': 3209, 'token_per_expert': {0: 1024, 4: 1027, 36: 57, 76: 58, 80: 58, 84: 95, 88: 51, 92: 102, 100: 48, 104: 54, 108: 338, 116: 240, 124: 57}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 21, 25, 45, 49, 53, 73, 77, 85, 97, 101, 117], 'expert_count': 13, 'ideal_gpu_count': 13, 'keep_on_gpu': 13, 'hit_count_on_device': 26, 'token_total': 4015, 'token_per_expert': {1: 1044, 5: 1149, 21: 368, 25: 110, 45: 299, 49: 174, 53: 407, 73: 82, 77: 47, 85: 38, 97: 154, 101: 59, 117: 84}}
INFO 05-06 11:02:20.828513.828513 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.942ms | allocate_experts_across_cpu_gpu: 0.396ms
INFO 05-06 11:02:20.828703.828703 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.0558319091796875e-05 seconds
INFO 05-06 11:02:20.830277.830277 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011010169982910156 seconds
INFO 05-06 11:02:20.840719.840719 lmp.py:1387] [layer_moe_fused] to time: 0.0001380443572998047 seconds
INFO 05-06 11:02:20.841322.841322 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.842111.842111 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011615753173828125 seconds
INFO 05-06 11:02:20.842292.842292 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005815029144287109 seconds
INFO 05-06 11:02:20.843042.843042 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0018923282623291016 seconds
INFO 05-06 11:02:20.852473.852473 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009637594223022461 seconds
INFO 05-06 11:02:20.854400.854400 mlpmodule.py:2799] [fused_experts] gmm total=1.871ms E=32 S=5044 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.855503.855503 mlpmodule.py:2799] [fused_experts] gmm total=2.058ms E=32 S=3768 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.855154.855154 mlpmodule.py:2799] [fused_experts] gmm total=1.991ms E=32 S=3417 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.855702.855702 mlpmodule.py:2799] [fused_experts] gmm total=2.061ms E=32 S=4155 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.856454.856454 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0035185813903808594 seconds
INFO 05-06 11:02:20.856048.856048 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.316734313964844e-05 seconds
DEBUG 05-06 11:02:20.857620.857620 cuda_h.py:27] end *layer_moe_fused cost 29.648 ms
DEBUG 05-06 11:02:20.863966.863966 cuda_h.py:27] end prefill_layer cost 38.940 ms
DEBUG 05-06 11:02:20.863836.863836 lmp.py:841] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 11:02:20.863936.863936 lmp.py:824] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 11:02:20.865604.865604 cuda_h.py:27] end *sagl cost 2.015 ms
experts_cpu_alloc {'expert_ids': [111, 23, 83, 123, 11, 107, 47, 87, 43, 95, 27, 67, 55, 99, 10, 74, 66, 62, 90, 106, 94, 70, 46, 26, 82, 42, 122, 12, 72, 76, 36, 48, 96, 112, 56, 8, 104, 52, 16, 40, 64, 28, 77, 45, 61, 53, 97, 105, 57, 109, 65, 93, 9, 101, 73, 117], 'token_total': 790, 'token_per_expert': {111: 3, 23: 8, 83: 13, 123: 13, 11: 14, 107: 14, 47: 15, 87: 20, 43: 22, 95: 24, 27: 25, 67: 29, 55: 38, 99: 42, 10: 2, 74: 2, 66: 5, 62: 9, 90: 9, 106: 12, 94: 15, 70: 16, 46: 20, 26: 23, 82: 30, 42: 32, 122: 36, 12: 1, 72: 2, 76: 2, 36: 3, 48: 3, 96: 3, 112: 4, 56: 5, 8: 10, 104: 12, 52: 15, 16: 17, 40: 17, 64: 17, 28: 19, 77: 1, 45: 2, 61: 2, 53: 3, 97: 3, 105: 7, 57: 9, 109: 12, 65: 13, 93: 17, 9: 18, 101: 19, 73: 25, 117: 38}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 31, 39, 51, 59, 63, 71, 75, 79, 91, 103, 115, 119], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4341, 'token_per_expert': {3: 1113, 7: 1025, 15: 75, 31: 479, 39: 101, 51: 92, 59: 157, 63: 117, 71: 213, 75: 45, 79: 275, 91: 452, 103: 77, 115: 65, 119: 55}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 22, 34, 38, 78, 86, 98, 102, 110, 114, 118, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 4108, 'token_per_expert': {2: 1053, 6: 1171, 14: 200, 22: 67, 34: 71, 38: 98, 78: 259, 86: 52, 98: 80, 102: 120, 110: 311, 114: 431, 118: 85, 126: 110}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 20, 32, 60, 68, 80, 84, 92, 100, 108, 116, 120, 124], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 3252, 'token_per_expert': {0: 1034, 4: 1024, 20: 130, 32: 207, 60: 64, 68: 22, 80: 22, 84: 68, 92: 28, 100: 308, 108: 21, 116: 62, 120: 212, 124: 50}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 17, 21, 25, 33, 37, 41, 69, 81, 113, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 3893, 'token_per_expert': {1: 1165, 5: 1024, 13: 51, 17: 254, 21: 98, 25: 119, 33: 104, 37: 276, 41: 50, 69: 65, 81: 214, 113: 67, 121: 273, 125: 133}}
INFO 05-06 11:02:20.867668.867668 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.567ms | allocate_experts_across_cpu_gpu: 0.427ms
INFO 05-06 11:02:20.867324.867324 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 0.00012302398681640625 seconds
INFO 05-06 11:02:20.869916.869916 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010952949523925781 seconds
INFO 05-06 11:02:20.879492.879492 lmp.py:1387] [layer_moe_fused] to time: 0.00012636184692382812 seconds
INFO 05-06 11:02:20.879557.879557 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.880568.880568 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001247406005859375 seconds
INFO 05-06 11:02:20.881518.881518 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005862712860107422 seconds
INFO 05-06 11:02:20.881791.881791 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019822120666503906 seconds
INFO 05-06 11:02:20.890098.890098 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009310007095336914 seconds
INFO 05-06 11:02:20.892973.892973 mlpmodule.py:2799] [fused_experts] gmm total=2.049ms E=32 S=3382 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.893230.893230 mlpmodule.py:2799] [fused_experts] gmm total=2.147ms E=32 S=4062 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.893016.893016 mlpmodule.py:2799] [fused_experts] gmm total=2.418ms E=32 S=4319 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.893489.893489 mlpmodule.py:2799] [fused_experts] gmm total=2.547ms E=32 S=4621 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.894230.894230 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0036611557006835938 seconds
INFO 05-06 11:02:20.894261.894261 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.3882598876953125e-05 seconds
DEBUG 05-06 11:02:20.895996.895996 cuda_h.py:27] end *layer_moe_fused cost 28.766 ms
DEBUG 05-06 11:02:20.900375.900375 cuda_h.py:27] end prefill_layer cost 37.605 ms
DEBUG 05-06 11:02:20.900808.900808 lmp.py:841] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 11:02:20.900001.900001 lmp.py:824] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 11:02:20.903061.903061 cuda_h.py:27] end *sagl cost 1.947 ms
experts_cpu_alloc {'expert_ids': [79, 87, 111, 27, 91, 51, 67, 15, 63, 35, 19, 23, 43, 107, 71, 94, 14, 46, 54, 106, 18, 22, 78, 58, 118, 70, 98, 126, 102, 90, 56, 84, 88, 96, 68, 116, 48, 40, 64, 36, 44, 28, 120, 16, 60, 72, 17, 49, 29, 85, 33, 77, 41, 37, 21, 93, 101, 9, 73, 109, 25], 'token_total': 862, 'token_per_expert': {79: 2, 87: 4, 111: 4, 27: 5, 91: 5, 51: 8, 67: 9, 15: 14, 63: 14, 35: 16, 19: 17, 23: 18, 43: 18, 107: 39, 71: 40, 94: 1, 14: 4, 46: 4, 54: 5, 106: 7, 18: 8, 22: 8, 78: 9, 58: 11, 118: 15, 70: 20, 98: 20, 126: 21, 102: 24, 90: 28, 56: 2, 84: 2, 88: 4, 96: 5, 68: 13, 116: 14, 48: 16, 40: 17, 64: 22, 36: 24, 44: 24, 28: 31, 120: 33, 16: 39, 60: 39, 72: 40, 17: 1, 49: 1, 29: 2, 85: 3, 33: 4, 77: 4, 41: 5, 37: 7, 21: 8, 93: 8, 101: 11, 9: 13, 73: 13, 109: 26, 25: 33}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 31, 39, 47, 59, 75, 83, 95, 99, 103, 115, 119, 123, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 4623, 'token_per_expert': {3: 1035, 7: 1070, 11: 61, 31: 161, 39: 266, 47: 183, 59: 153, 75: 205, 83: 44, 95: 196, 99: 124, 103: 130, 115: 505, 119: 249, 123: 139, 127: 102}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 26, 30, 34, 38, 42, 50, 62, 66, 74, 86, 110, 114, 122], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 3957, 'token_per_expert': {2: 1129, 6: 1027, 10: 67, 26: 295, 30: 63, 34: 61, 38: 97, 42: 83, 50: 231, 62: 83, 66: 261, 74: 53, 86: 281, 110: 35, 114: 90, 122: 101}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 12, 24, 32, 52, 76, 80, 92, 100, 104, 108, 112, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3306, 'token_per_expert': {0: 1047, 4: 1025, 8: 85, 12: 66, 24: 100, 32: 53, 52: 53, 76: 61, 80: 107, 92: 40, 100: 166, 104: 95, 108: 52, 112: 58, 124: 298}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 45, 53, 57, 65, 81, 89, 97, 105, 113, 117, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3636, 'token_per_expert': {1: 1028, 5: 1040, 13: 67, 45: 39, 53: 64, 57: 49, 65: 230, 81: 38, 89: 89, 97: 254, 105: 53, 113: 123, 117: 130, 121: 375, 125: 57}}
INFO 05-06 11:02:20.905197.905197 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.894ms | allocate_experts_across_cpu_gpu: 0.452ms
INFO 05-06 11:02:20.905586.905586 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.413459777832031e-05 seconds
INFO 05-06 11:02:20.907242.907242 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010874271392822266 seconds
INFO 05-06 11:02:20.918668.918668 lmp.py:1387] [layer_moe_fused] to time: 0.00012636184692382812 seconds
INFO 05-06 11:02:20.918495.918495 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.919536.919536 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001171112060546875 seconds
INFO 05-06 11:02:20.920108.920108 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005843639373779297 seconds
INFO 05-06 11:02:20.920574.920574 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019109249114990234 seconds
INFO 05-06 11:02:20.929485.929485 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009371042251586914 seconds
INFO 05-06 11:02:20.932545.932545 mlpmodule.py:2799] [fused_experts] gmm total=1.793ms E=32 S=3775 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.932930.932930 mlpmodule.py:2799] [fused_experts] gmm total=2.054ms E=32 S=3631 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.932132.932132 mlpmodule.py:2799] [fused_experts] gmm total=2.220ms E=32 S=4142 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.932992.932992 mlpmodule.py:2799] [fused_experts] gmm total=2.459ms E=32 S=4836 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.933819.933819 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0033566951751708984 seconds
INFO 05-06 11:02:20.933088.933088 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.3882598876953125e-05 seconds
DEBUG 05-06 11:02:20.934724.934724 cuda_h.py:27] end *layer_moe_fused cost 29.931 ms
DEBUG 05-06 11:02:20.939850.939850 cuda_h.py:27] end prefill_layer cost 38.609 ms
DEBUG 05-06 11:02:20.939250.939250 lmp.py:841] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 11:02:20.939397.939397 lmp.py:824] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 11:02:20.941377.941377 cuda_h.py:27] end *sagl cost 1.963 ms
experts_cpu_alloc {'expert_ids': [35, 67, 11, 87, 123, 79, 111, 19, 107, 115, 43, 119, 31, 127, 26, 50, 62, 74, 106, 22, 110, 54, 94, 126, 38, 118, 34, 82, 58, 18, 20, 56, 12, 100, 44, 32, 80, 96, 8, 48, 40, 120, 28, 36, 124, 49, 61, 89, 57, 45, 105, 25, 117, 77, 13, 17, 121, 41, 29, 33, 69], 'token_total': 1086, 'token_per_expert': {35: 4, 67: 4, 11: 6, 87: 6, 123: 6, 79: 10, 111: 17, 19: 21, 107: 26, 115: 28, 43: 29, 119: 29, 31: 33, 127: 43, 26: 1, 50: 1, 62: 1, 74: 2, 106: 2, 22: 4, 110: 5, 54: 6, 94: 10, 126: 11, 38: 17, 118: 18, 34: 19, 82: 22, 58: 23, 18: 25, 20: 1, 56: 1, 12: 2, 100: 4, 44: 6, 32: 11, 80: 12, 96: 16, 8: 20, 48: 22, 40: 37, 120: 54, 28: 56, 36: 56, 124: 57, 49: 1, 61: 2, 89: 2, 57: 6, 45: 9, 105: 10, 25: 13, 117: 17, 77: 21, 13: 26, 17: 27, 121: 29, 41: 31, 29: 33, 33: 35, 69: 40}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 47, 51, 55, 59, 63, 71, 75, 83, 91, 95, 99, 103], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3771, 'token_per_expert': {3: 1028, 7: 1119, 23: 128, 39: 119, 47: 59, 51: 97, 55: 49, 59: 51, 63: 52, 71: 126, 75: 152, 83: 225, 91: 359, 95: 67, 99: 74, 103: 66}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 14, 30, 42, 46, 66, 70, 78, 86, 90, 98, 102, 114], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3338, 'token_per_expert': {2: 1115, 6: 1026, 10: 232, 14: 41, 30: 154, 42: 62, 46: 50, 66: 110, 70: 102, 78: 42, 86: 27, 90: 182, 98: 111, 102: 42, 114: 42}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 16, 24, 52, 64, 68, 72, 76, 84, 88, 104, 108, 112, 116], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4504, 'token_per_expert': {0: 1078, 4: 1041, 16: 96, 24: 61, 52: 150, 64: 146, 68: 341, 72: 105, 76: 476, 84: 93, 88: 108, 104: 96, 108: 210, 112: 435, 116: 68}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 21, 37, 65, 73, 81, 85, 93, 97, 101, 109, 113, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3685, 'token_per_expert': {1: 1066, 5: 1074, 9: 155, 21: 64, 37: 101, 65: 296, 73: 49, 81: 88, 85: 71, 93: 69, 97: 83, 101: 127, 109: 294, 113: 46, 125: 102}}
INFO 05-06 11:02:20.944389.944389 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.565ms | allocate_experts_across_cpu_gpu: 0.470ms
INFO 05-06 11:02:20.944169.944169 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.794929504394531e-05 seconds
INFO 05-06 11:02:20.945873.945873 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010924339294433594 seconds
INFO 05-06 11:02:20.959327.959327 lmp.py:1387] [layer_moe_fused] to time: 0.0001385211944580078 seconds
INFO 05-06 11:02:20.959348.959348 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:20.960389.960389 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011699199676513672 seconds
INFO 05-06 11:02:20.961133.961133 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005757808685302734 seconds
INFO 05-06 11:02:20.961360.961360 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0018939971923828125 seconds
INFO 05-06 11:02:20.971657.971657 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010216712951660156 seconds
INFO 05-06 11:02:20.974366.974366 mlpmodule.py:2799] [fused_experts] gmm total=1.929ms E=32 S=4033 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.974785.974785 mlpmodule.py:2799] [fused_experts] gmm total=2.145ms E=32 S=3505 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.974545.974545 mlpmodule.py:2799] [fused_experts] gmm total=2.179ms E=32 S=3987 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.974663.974663 mlpmodule.py:2799] [fused_experts] gmm total=2.385ms E=32 S=4859 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:20.975530.975530 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0038208961486816406 seconds
INFO 05-06 11:02:20.975832.975832 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.173683166503906e-05 seconds
DEBUG 05-06 11:02:20.976709.976709 cuda_h.py:27] end *layer_moe_fused cost 33.926 ms
DEBUG 05-06 11:02:20.982726.982726 cuda_h.py:27] end prefill_layer cost 42.649 ms
DEBUG 05-06 11:02:20.982696.982696 lmp.py:841] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 11:02:20.982320.982320 lmp.py:824] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 11:02:20.984923.984923 cuda_h.py:27] end *sagl cost 2.027 ms
experts_cpu_alloc {'expert_ids': [95, 27, 35, 115, 71, 59, 11, 103, 99, 91, 43, 111, 123, 79, 94, 106, 74, 50, 98, 122, 34, 38, 10, 62, 118, 18, 30, 82, 22, 120, 28, 88, 36, 104, 112, 60, 64, 84, 24, 40, 56, 92, 80, 116, 72, 25, 73, 101, 89, 29, 41, 49, 69, 53, 9, 13, 37, 45, 109, 33, 21, 113], 'token_total': 1017, 'token_per_expert': {95: 1, 27: 2, 35: 2, 115: 2, 71: 4, 59: 9, 11: 12, 103: 14, 99: 20, 91: 24, 43: 25, 111: 25, 123: 27, 79: 28, 94: 2, 106: 2, 74: 4, 50: 7, 98: 7, 122: 7, 34: 11, 38: 17, 10: 19, 62: 21, 118: 27, 18: 32, 30: 32, 82: 50, 22: 52, 120: 3, 28: 4, 88: 5, 36: 7, 104: 10, 112: 13, 60: 14, 64: 14, 84: 25, 24: 27, 40: 27, 56: 28, 92: 30, 80: 32, 116: 47, 72: 57, 25: 1, 73: 2, 101: 2, 89: 3, 29: 4, 41: 5, 49: 6, 69: 6, 53: 7, 9: 8, 13: 18, 37: 18, 45: 18, 109: 18, 33: 19, 21: 26, 113: 28}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 31, 51, 55, 63, 67, 75, 83, 87, 107, 119, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3858, 'token_per_expert': {3: 1098, 7: 1028, 15: 29, 19: 65, 23: 173, 31: 96, 51: 35, 55: 82, 63: 100, 67: 322, 75: 102, 83: 145, 87: 391, 107: 93, 119: 36, 127: 63}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 26, 42, 54, 58, 66, 70, 78, 86, 90, 102, 110, 114, 126], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 3904, 'token_per_expert': {2: 1099, 6: 1049, 14: 72, 26: 75, 42: 86, 54: 91, 58: 70, 66: 141, 70: 54, 78: 80, 86: 260, 90: 54, 102: 54, 110: 86, 114: 86, 126: 547}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 12, 16, 20, 32, 44, 48, 52, 68, 76, 96, 100, 108, 124], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 4447, 'token_per_expert': {0: 1137, 4: 1129, 8: 129, 12: 134, 16: 332, 20: 77, 32: 475, 44: 66, 48: 75, 52: 363, 68: 84, 76: 81, 96: 82, 100: 107, 108: 81, 124: 95}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 17, 57, 61, 65, 77, 81, 85, 93, 97, 105, 117, 121, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 3158, 'token_per_expert': {1: 1215, 5: 1141, 17: 32, 57: 41, 61: 54, 65: 54, 77: 39, 81: 38, 85: 57, 93: 35, 97: 36, 105: 232, 117: 68, 121: 48, 125: 68}}
INFO 05-06 11:02:20.987629.987629 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.880ms | allocate_experts_across_cpu_gpu: 0.462ms
INFO 05-06 11:02:20.987886.987886 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.842613220214844e-05 seconds
INFO 05-06 11:02:20.988369.988369 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011179447174072266 seconds
INFO 05-06 11:02:21.002636.002636 lmp.py:1387] [layer_moe_fused] to time: 0.0001347064971923828 seconds
INFO 05-06 11:02:21.002960.002960 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.003206.003206 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011487007141113281 seconds
INFO 05-06 11:02:21.004662.004662 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006628036499023438 seconds
INFO 05-06 11:02:21.004843.004843 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.00197601318359375 seconds
INFO 05-06 11:02:21.014624.014624 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00983572006225586 seconds
INFO 05-06 11:02:21.016445.016445 mlpmodule.py:2799] [fused_experts] gmm total=2.113ms E=32 S=4053 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.016978.016978 mlpmodule.py:2799] [fused_experts] gmm total=2.166ms E=32 S=4194 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.016680.016680 mlpmodule.py:2799] [fused_experts] gmm total=2.254ms E=32 S=3347 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.017861.017861 mlpmodule.py:2799] [fused_experts] gmm total=2.375ms E=32 S=4790 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.018353.018353 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003679037094116211 seconds
INFO 05-06 11:02:21.018708.018708 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.2928924560546875e-05 seconds
DEBUG 05-06 11:02:21.019445.019445 cuda_h.py:27] end *layer_moe_fused cost 33.230 ms
DEBUG 05-06 11:02:21.024115.024115 cuda_h.py:27] end prefill_layer cost 42.326 ms
DEBUG 05-06 11:02:21.024693.024693 lmp.py:841] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 11:02:21.024840.024840 lmp.py:824] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 11:02:21.027822.027822 cuda_h.py:27] end *sagl cost 2.778 ms
experts_cpu_alloc {'expert_ids': [79, 51, 127, 15, 11, 83, 91, 19, 119, 123, 99, 59, 67, 87, 111, 110, 30, 102, 82, 42, 34, 90, 14, 62, 66, 118, 38, 114, 126, 8, 88, 112, 96, 92, 36, 16, 44, 60, 100, 32, 116, 104, 48, 80, 120, 84, 25, 77, 29, 93, 85, 81, 65, 117, 97, 109, 9, 113, 33, 125], 'token_total': 1203, 'token_per_expert': {79: 4, 51: 5, 127: 5, 15: 6, 11: 10, 83: 14, 91: 18, 19: 19, 119: 28, 123: 28, 99: 29, 59: 31, 67: 31, 87: 34, 111: 36, 110: 1, 30: 7, 102: 8, 82: 10, 42: 11, 34: 16, 90: 19, 14: 20, 62: 21, 66: 21, 118: 22, 38: 25, 114: 28, 126: 34, 8: 6, 88: 6, 112: 6, 96: 14, 92: 15, 36: 17, 16: 21, 44: 22, 60: 23, 100: 27, 32: 28, 116: 34, 104: 35, 48: 41, 80: 50, 120: 50, 84: 54, 25: 1, 77: 3, 29: 4, 93: 6, 85: 7, 81: 9, 65: 10, 117: 10, 97: 16, 109: 19, 9: 24, 113: 28, 33: 30, 125: 46}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 31, 35, 39, 43, 47, 55, 63, 71, 75, 95, 103, 107], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 3991, 'token_per_expert': {3: 1064, 7: 1025, 23: 285, 27: 147, 31: 58, 35: 60, 39: 154, 43: 154, 47: 92, 55: 58, 63: 94, 71: 103, 75: 277, 95: 228, 103: 70, 107: 122}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 18, 22, 54, 58, 70, 74, 78, 86, 94, 98, 106, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3398, 'token_per_expert': {2: 1038, 6: 1081, 10: 106, 18: 95, 22: 83, 54: 64, 58: 164, 70: 61, 74: 230, 78: 51, 86: 194, 94: 45, 98: 55, 106: 77, 122: 54}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 12, 20, 24, 28, 40, 52, 56, 64, 68, 72, 76, 108, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 3893, 'token_per_expert': {0: 1071, 4: 1095, 12: 67, 20: 83, 24: 287, 28: 98, 40: 148, 52: 162, 56: 85, 64: 82, 68: 82, 72: 170, 76: 341, 108: 62, 124: 60}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 17, 21, 37, 45, 49, 53, 57, 61, 69, 73, 89, 101], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3899, 'token_per_expert': {1: 1035, 5: 1079, 13: 47, 17: 77, 21: 176, 37: 390, 45: 62, 49: 96, 53: 150, 57: 65, 61: 146, 69: 250, 73: 53, 89: 173, 101: 100}}
INFO 05-06 11:02:21.032425.032425 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 2.779ms | allocate_experts_across_cpu_gpu: 0.447ms
INFO 05-06 11:02:21.032536.032536 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.651878356933594e-05 seconds
INFO 05-06 11:02:21.033479.033479 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011034011840820312 seconds
INFO 05-06 11:02:21.048785.048785 lmp.py:1387] [layer_moe_fused] to time: 0.000141143798828125 seconds
INFO 05-06 11:02:21.048718.048718 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.049566.049566 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00113677978515625 seconds
INFO 05-06 11:02:21.050859.050859 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005638599395751953 seconds
INFO 05-06 11:02:21.050325.050325 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0018427371978759766 seconds
INFO 05-06 11:02:21.060816.060816 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009691476821899414 seconds
INFO 05-06 11:02:21.062924.062924 mlpmodule.py:2799] [fused_experts] gmm total=1.915ms E=32 S=3641 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.062836.062836 mlpmodule.py:2799] [fused_experts] gmm total=1.960ms E=32 S=4112 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.062304.062304 mlpmodule.py:2799] [fused_experts] gmm total=2.143ms E=32 S=4342 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.062369.062369 mlpmodule.py:2799] [fused_experts] gmm total=2.409ms E=32 S=4289 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.063321.063321 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003511667251586914 seconds
INFO 05-06 11:02:21.063437.063437 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.2928924560546875e-05 seconds
DEBUG 05-06 11:02:21.064678.064678 cuda_h.py:27] end *layer_moe_fused cost 35.679 ms
DEBUG 05-06 11:02:21.070543.070543 cuda_h.py:27] end prefill_layer cost 45.314 ms
DEBUG 05-06 11:02:21.070076.070076 lmp.py:841] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 11:02:21.070223.070223 lmp.py:824] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 11:02:21.072008.072008 cuda_h.py:27] end *sagl cost 2.025 ms
experts_cpu_alloc {'expert_ids': [115, 55, 63, 11, 59, 27, 39, 23, 19, 67, 107, 15, 35, 51, 91, 22, 106, 86, 126, 102, 18, 42, 62, 82, 94, 114, 66, 74, 98, 70, 122, 20, 16, 52, 24, 44, 96, 124, 112, 48, 108, 116, 56, 68, 12, 92, 113, 117, 25, 41, 21, 45, 89, 9, 109, 73, 97, 13, 29, 37, 125, 57], 'token_total': 1296, 'token_per_expert': {115: 2, 55: 5, 63: 10, 11: 11, 59: 12, 27: 13, 39: 16, 23: 20, 19: 21, 67: 25, 107: 29, 15: 30, 35: 31, 51: 32, 91: 39, 22: 1, 106: 2, 86: 3, 126: 4, 102: 6, 18: 7, 42: 8, 62: 13, 82: 14, 94: 16, 114: 18, 66: 20, 74: 20, 98: 20, 70: 23, 122: 23, 20: 2, 16: 4, 52: 5, 24: 8, 44: 12, 96: 14, 124: 23, 112: 25, 48: 28, 108: 33, 116: 35, 56: 40, 68: 45, 12: 54, 92: 59, 113: 3, 117: 3, 25: 7, 41: 11, 21: 12, 45: 13, 89: 17, 9: 21, 109: 30, 73: 35, 97: 35, 13: 41, 29: 44, 37: 46, 125: 48, 57: 49}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 43, 47, 71, 75, 83, 87, 95, 99, 103, 111, 119, 123, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 3812, 'token_per_expert': {3: 1183, 7: 1054, 31: 122, 43: 128, 47: 56, 71: 67, 75: 72, 83: 167, 87: 96, 95: 68, 99: 263, 103: 44, 111: 184, 119: 159, 123: 76, 127: 73}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 14, 26, 30, 34, 38, 46, 50, 54, 58, 78, 90, 110, 118], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 3545, 'token_per_expert': {2: 1238, 6: 1026, 10: 55, 14: 116, 26: 28, 30: 52, 34: 68, 38: 83, 46: 39, 50: 183, 54: 192, 58: 134, 78: 87, 90: 25, 110: 106, 118: 113}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 32, 36, 40, 60, 64, 72, 76, 80, 84, 88, 100, 104, 120], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 3905, 'token_per_expert': {0: 1028, 4: 1135, 8: 141, 32: 183, 36: 159, 40: 151, 60: 119, 64: 168, 72: 94, 76: 172, 80: 63, 84: 117, 88: 83, 100: 80, 104: 115, 120: 97}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 17, 33, 49, 53, 61, 65, 69, 77, 81, 85, 93, 101, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3826, 'token_per_expert': {1: 1097, 5: 1070, 17: 72, 33: 134, 49: 83, 53: 144, 61: 110, 65: 102, 69: 74, 77: 126, 81: 104, 85: 212, 93: 98, 101: 144, 121: 256}}
INFO 05-06 11:02:21.075834.075834 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.898ms | allocate_experts_across_cpu_gpu: 0.455ms
INFO 05-06 11:02:21.075693.075693 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.461143493652344e-05 seconds
INFO 05-06 11:02:21.076017.076017 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011301040649414062 seconds
INFO 05-06 11:02:21.092700.092700 lmp.py:1387] [layer_moe_fused] to time: 0.00014281272888183594 seconds
INFO 05-06 11:02:21.092601.092601 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.093052.093052 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011587142944335938 seconds
INFO 05-06 11:02:21.094895.094895 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005793571472167969 seconds
INFO 05-06 11:02:21.094838.094838 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001886129379272461 seconds
INFO 05-06 11:02:21.104650.104650 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009788036346435547 seconds
INFO 05-06 11:02:21.106406.106406 mlpmodule.py:2799] [fused_experts] gmm total=2.075ms E=32 S=3743 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.106740.106740 mlpmodule.py:2799] [fused_experts] gmm total=2.082ms E=32 S=4241 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.107583.107583 mlpmodule.py:2799] [fused_experts] gmm total=2.305ms E=32 S=4108 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.107508.107508 mlpmodule.py:2799] [fused_experts] gmm total=2.309ms E=32 S=4292 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.108351.108351 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0036232471466064453 seconds
INFO 05-06 11:02:21.108037.108037 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.2928924560546875e-05 seconds
DEBUG 05-06 11:02:21.109327.109327 cuda_h.py:27] end *layer_moe_fused cost 35.250 ms
DEBUG 05-06 11:02:21.114384.114384 cuda_h.py:27] end prefill_layer cost 44.561 ms
DEBUG 05-06 11:02:21.115029.115029 lmp.py:841] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 11:02:21.115700.115700 lmp.py:824] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 11:02:21.117716.117716 cuda_h.py:27] end *sagl cost 2.038 ms
experts_cpu_alloc {'expert_ids': [71, 107, 115, 67, 43, 87, 95, 55, 127, 103, 111, 59, 99, 83, 11, 62, 82, 34, 46, 74, 66, 54, 30, 70, 94, 114, 42, 126, 18, 110, 124, 100, 8, 120, 108, 20, 68, 112, 56, 12, 84, 72, 36, 60, 85, 93, 49, 105, 113, 57, 101, 65, 77, 45, 121, 29, 73, 25, 17, 97], 'token_total': 1040, 'token_per_expert': {71: 2, 107: 3, 115: 3, 67: 5, 43: 8, 87: 9, 95: 9, 55: 11, 127: 13, 103: 25, 111: 28, 59: 29, 99: 30, 83: 38, 11: 42, 62: 4, 82: 4, 34: 6, 46: 6, 74: 6, 66: 8, 54: 9, 30: 10, 70: 10, 94: 11, 114: 15, 42: 16, 126: 18, 18: 20, 110: 24, 124: 3, 100: 6, 8: 9, 120: 14, 108: 21, 20: 26, 68: 28, 112: 28, 56: 29, 12: 36, 84: 36, 72: 37, 36: 39, 60: 44, 85: 2, 93: 2, 49: 4, 105: 4, 113: 5, 57: 7, 101: 7, 65: 8, 77: 10, 45: 18, 121: 18, 29: 22, 73: 31, 25: 40, 17: 42, 97: 42}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 39, 47, 51, 63, 75, 79, 119, 123], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 3691, 'token_per_expert': {3: 1194, 7: 1192, 15: 67, 19: 44, 23: 112, 27: 106, 31: 47, 35: 79, 39: 68, 47: 51, 51: 225, 63: 99, 75: 112, 79: 113, 119: 58, 123: 124}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 22, 26, 38, 50, 58, 86, 90, 98, 102, 106, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3443, 'token_per_expert': {2: 1082, 6: 1030, 10: 65, 22: 34, 26: 76, 38: 346, 50: 107, 58: 25, 86: 25, 90: 41, 98: 43, 102: 99, 106: 68, 118: 42, 122: 360}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 16, 24, 40, 44, 48, 52, 64, 76, 80, 88, 92, 96, 104], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4310, 'token_per_expert': {0: 1076, 4: 1040, 16: 90, 24: 198, 40: 131, 44: 265, 48: 77, 52: 472, 64: 255, 76: 85, 80: 118, 88: 84, 92: 293, 96: 73, 104: 53}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 21, 33, 37, 41, 53, 61, 69, 89, 109, 117, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3900, 'token_per_expert': {1: 1091, 5: 1081, 9: 97, 13: 46, 21: 121, 33: 65, 37: 243, 41: 73, 53: 50, 61: 153, 69: 104, 89: 396, 109: 106, 117: 190, 125: 84}}
INFO 05-06 11:02:21.119426.119426 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.593ms | allocate_experts_across_cpu_gpu: 0.453ms
INFO 05-06 11:02:21.119769.119769 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.532669067382812e-05 seconds
INFO 05-06 11:02:21.120924.120924 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011105537414550781 seconds
INFO 05-06 11:02:21.134575.134575 lmp.py:1387] [layer_moe_fused] to time: 0.0001342296600341797 seconds
INFO 05-06 11:02:21.134462.134462 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.136638.136638 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012378692626953125 seconds
INFO 05-06 11:02:21.136282.136282 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005650520324707031 seconds
INFO 05-06 11:02:21.136463.136463 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019588470458984375 seconds
INFO 05-06 11:02:21.146369.146369 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009823799133300781 seconds
INFO 05-06 11:02:21.148689.148689 mlpmodule.py:2799] [fused_experts] gmm total=1.929ms E=32 S=3610 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.149593.149593 mlpmodule.py:2799] [fused_experts] gmm total=1.978ms E=32 S=4666 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.149044.149044 mlpmodule.py:2799] [fused_experts] gmm total=2.118ms E=32 S=4162 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.149169.149169 mlpmodule.py:2799] [fused_experts] gmm total=2.489ms E=32 S=3946 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.150101.150101 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003365755081176758 seconds
INFO 05-06 11:02:21.150741.150741 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.340576171875e-05 seconds
DEBUG 05-06 11:02:21.151950.151950 cuda_h.py:27] end *layer_moe_fused cost 32.720 ms
DEBUG 05-06 11:02:21.156659.156659 cuda_h.py:27] end prefill_layer cost 41.885 ms
DEBUG 05-06 11:02:21.157237.157237 lmp.py:841] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 11:02:21.157623.157623 lmp.py:824] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 11:02:21.159157.159157 cuda_h.py:27] end *sagl cost 1.991 ms
experts_cpu_alloc {'expert_ids': [23, 67, 87, 91, 99, 75, 51, 119, 127, 11, 31, 35, 47, 19, 14, 22, 70, 106, 126, 34, 86, 10, 26, 90, 62, 98, 38, 58, 110, 16, 124, 24, 36, 104, 120, 80, 84, 52, 76, 12, 60, 20, 100, 72, 29, 97, 69, 17, 25, 89, 61, 117, 101, 93, 105, 121, 113, 41, 33, 53, 9], 'token_total': 1319, 'token_per_expert': {23: 2, 67: 2, 87: 2, 91: 2, 99: 2, 75: 3, 51: 4, 119: 5, 127: 5, 11: 11, 31: 11, 35: 13, 47: 15, 19: 22, 14: 1, 22: 2, 70: 4, 106: 8, 126: 8, 34: 10, 86: 16, 10: 20, 26: 20, 90: 20, 62: 21, 98: 23, 38: 24, 58: 25, 110: 27, 16: 3, 124: 5, 24: 11, 36: 17, 104: 20, 120: 23, 80: 25, 84: 27, 52: 41, 76: 42, 12: 43, 60: 46, 20: 50, 100: 59, 72: 63, 29: 2, 97: 4, 69: 6, 17: 8, 25: 9, 89: 10, 61: 12, 117: 12, 101: 18, 93: 20, 105: 28, 121: 37, 113: 57, 41: 69, 33: 73, 53: 74, 9: 77}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 43, 55, 59, 63, 71, 79, 83, 95, 103, 107, 111, 123], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3440, 'token_per_expert': {3: 1154, 7: 1026, 15: 58, 27: 66, 43: 91, 55: 37, 59: 134, 63: 246, 71: 50, 79: 59, 83: 45, 95: 22, 103: 27, 107: 290, 111: 27, 123: 108}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 30, 42, 46, 50, 54, 66, 74, 82, 94, 102, 114, 118, 122], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 3457, 'token_per_expert': {2: 1075, 6: 1031, 18: 46, 30: 193, 42: 88, 46: 79, 50: 53, 54: 40, 66: 96, 74: 37, 82: 49, 94: 346, 102: 176, 114: 34, 118: 42, 122: 72}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 28, 32, 40, 44, 56, 64, 68, 88, 92, 108, 112, 116], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3849, 'token_per_expert': {0: 1050, 4: 1135, 8: 125, 28: 112, 32: 86, 40: 125, 44: 102, 56: 129, 64: 64, 68: 471, 88: 85, 92: 93, 108: 118, 112: 82, 116: 72}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 21, 37, 45, 49, 57, 65, 73, 77, 81, 85, 109, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 4319, 'token_per_expert': {1: 1048, 5: 1159, 13: 113, 21: 174, 37: 179, 45: 271, 49: 317, 57: 182, 65: 211, 73: 170, 77: 138, 81: 84, 85: 90, 109: 89, 125: 94}}
INFO 05-06 11:02:21.161506.161506 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.918ms | allocate_experts_across_cpu_gpu: 0.453ms
INFO 05-06 11:02:21.162379.162379 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.890296936035156e-05 seconds
INFO 05-06 11:02:21.163307.163307 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011069774627685547 seconds
INFO 05-06 11:02:21.178844.178844 lmp.py:1387] [layer_moe_fused] to time: 0.00014281272888183594 seconds
INFO 05-06 11:02:21.179314.179314 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.180707.180707 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011870861053466797 seconds
INFO 05-06 11:02:21.180927.180927 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005772113800048828 seconds
INFO 05-06 11:02:21.181870.181870 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019121170043945312 seconds
INFO 05-06 11:02:21.191783.191783 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010039329528808594 seconds
INFO 05-06 11:02:21.193529.193529 mlpmodule.py:2799] [fused_experts] gmm total=1.948ms E=32 S=4324 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.193010.193010 mlpmodule.py:2799] [fused_experts] gmm total=2.173ms E=32 S=3539 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.193279.193279 mlpmodule.py:2799] [fused_experts] gmm total=2.137ms E=32 S=4835 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.193136.193136 mlpmodule.py:2799] [fused_experts] gmm total=2.310ms E=32 S=3686 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.194011.194011 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003552675247192383 seconds
INFO 05-06 11:02:21.194612.194612 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.412101745605469e-05 seconds
DEBUG 05-06 11:02:21.195796.195796 cuda_h.py:27] end *layer_moe_fused cost 35.328 ms
DEBUG 05-06 11:02:21.201876.201876 cuda_h.py:27] end prefill_layer cost 44.737 ms
DEBUG 05-06 11:02:21.201832.201832 lmp.py:841] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 11:02:21.201502.201502 lmp.py:824] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 11:02:21.204185.204185 cuda_h.py:27] end *sagl cost 2.066 ms
experts_cpu_alloc {'expert_ids': [19, 15, 39, 99, 107, 47, 23, 115, 71, 55, 43, 59, 119, 127, 66, 98, 22, 94, 54, 74, 126, 114, 106, 50, 42, 38, 82, 118, 58, 70, 116, 28, 60, 104, 52, 88, 96, 16, 64, 56, 40, 20, 32, 44, 24, 80, 89, 77, 9, 17, 101, 25, 69, 93, 113, 121, 45, 21, 81, 125], 'token_total': 1072, 'token_per_expert': {19: 1, 15: 2, 39: 2, 99: 3, 107: 3, 47: 6, 23: 9, 115: 17, 71: 20, 55: 23, 43: 25, 59: 26, 119: 31, 127: 31, 66: 1, 98: 5, 22: 8, 94: 8, 54: 11, 74: 17, 126: 19, 114: 23, 106: 24, 50: 27, 42: 35, 38: 39, 82: 40, 118: 41, 58: 42, 70: 44, 116: 1, 28: 2, 60: 2, 104: 4, 52: 8, 88: 9, 96: 9, 16: 12, 64: 12, 56: 17, 40: 20, 20: 27, 32: 30, 44: 31, 24: 36, 80: 39, 89: 1, 77: 4, 9: 6, 17: 6, 101: 9, 25: 10, 69: 11, 93: 11, 113: 12, 121: 21, 45: 25, 21: 26, 81: 42, 125: 46}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 31, 35, 51, 67, 75, 79, 83, 87, 95, 103, 111, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3219, 'token_per_expert': {3: 1039, 7: 1120, 11: 239, 31: 71, 35: 62, 51: 104, 67: 70, 75: 56, 79: 49, 83: 123, 87: 44, 95: 60, 103: 93, 111: 54, 123: 35}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 18, 26, 30, 34, 46, 62, 78, 86, 90, 102, 110, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4101, 'token_per_expert': {2: 1099, 6: 1334, 10: 67, 18: 179, 26: 165, 30: 75, 34: 61, 46: 165, 62: 107, 78: 344, 86: 61, 90: 103, 102: 80, 110: 92, 122: 169}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 12, 36, 48, 68, 72, 76, 84, 92, 100, 112, 120, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3772, 'token_per_expert': {0: 1056, 4: 1103, 8: 102, 12: 63, 36: 72, 48: 212, 68: 75, 72: 82, 76: 128, 84: 89, 92: 143, 100: 248, 112: 133, 120: 152, 124: 114}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 29, 33, 37, 41, 53, 57, 61, 65, 73, 97, 105, 109], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4220, 'token_per_expert': {1: 1265, 5: 1346, 13: 69, 29: 142, 33: 58, 37: 113, 41: 150, 53: 132, 57: 77, 61: 104, 65: 278, 73: 139, 97: 79, 105: 171, 109: 97}}
INFO 05-06 11:02:21.206854.206854 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.579ms | allocate_experts_across_cpu_gpu: 0.444ms
INFO 05-06 11:02:21.206727.206727 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.508827209472656e-05 seconds
INFO 05-06 11:02:21.207439.207439 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011091232299804688 seconds
INFO 05-06 11:02:21.221710.221710 lmp.py:1387] [layer_moe_fused] to time: 0.000133514404296875 seconds
INFO 05-06 11:02:21.222790.222790 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.223001.223001 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011227130889892578 seconds
INFO 05-06 11:02:21.224878.224878 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006024837493896484 seconds
INFO 05-06 11:02:21.224205.224205 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001878976821899414 seconds
INFO 05-06 11:02:21.234880.234880 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010038614273071289 seconds
INFO 05-06 11:02:21.236569.236569 mlpmodule.py:2799] [fused_experts] gmm total=2.132ms E=32 S=3418 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.236718.236718 mlpmodule.py:2799] [fused_experts] gmm total=2.130ms E=32 S=4031 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.236820.236820 mlpmodule.py:2799] [fused_experts] gmm total=2.252ms E=32 S=4485 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.236566.236566 mlpmodule.py:2799] [fused_experts] gmm total=2.274ms E=32 S=4450 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.237410.237410 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0036728382110595703 seconds
INFO 05-06 11:02:21.238395.238395 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.602836608886719e-05 seconds
DEBUG 05-06 11:02:21.238389.238389 cuda_h.py:27] end *layer_moe_fused cost 33.405 ms
DEBUG 05-06 11:02:21.244148.244148 cuda_h.py:27] end prefill_layer cost 43.024 ms
DEBUG 05-06 11:02:21.245396.245396 lmp.py:841] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 11:02:21.245781.245781 lmp.py:824] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 11:02:21.247378.247378 cuda_h.py:27] end *sagl cost 2.016 ms
experts_cpu_alloc {'expert_ids': [91, 23, 71, 67, 95, 87, 39, 83, 27, 51, 63, 79, 47, 31, 107, 99, 22, 18, 78, 106, 54, 14, 10, 122, 98, 110, 62, 34, 58, 42, 56, 80, 12, 36, 104, 96, 60, 84, 20, 32, 48, 112, 16, 44, 40, 124, 13, 29, 21, 77, 49, 105, 61, 17, 97, 81, 37, 65, 25, 9, 57, 109], 'token_total': 1117, 'token_per_expert': {91: 2, 23: 3, 71: 4, 67: 7, 95: 8, 87: 9, 39: 11, 83: 19, 27: 20, 51: 21, 63: 23, 79: 26, 47: 37, 31: 39, 107: 41, 99: 55, 22: 1, 18: 2, 78: 4, 106: 8, 54: 10, 14: 11, 10: 14, 122: 17, 98: 18, 110: 18, 62: 25, 34: 26, 58: 50, 42: 54, 56: 1, 80: 1, 12: 2, 36: 3, 104: 3, 96: 8, 60: 11, 84: 18, 20: 26, 32: 39, 48: 40, 112: 43, 16: 44, 44: 44, 40: 53, 124: 58, 13: 1, 29: 1, 21: 2, 77: 3, 49: 4, 105: 4, 61: 5, 17: 6, 97: 7, 81: 8, 37: 9, 65: 10, 25: 16, 9: 17, 57: 21, 109: 26}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 35, 43, 55, 59, 75, 103, 111, 115, 119, 123, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 32, 'token_total': 3901, 'token_per_expert': {3: 1026, 7: 1083, 11: 62, 15: 88, 19: 56, 35: 292, 43: 85, 55: 131, 59: 219, 75: 107, 103: 216, 111: 72, 115: 74, 119: 166, 123: 103, 127: 121}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 26, 30, 38, 46, 66, 70, 74, 82, 86, 90, 94, 102, 118, 126], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3688, 'token_per_expert': {2: 1029, 6: 1027, 26: 57, 30: 80, 38: 118, 46: 91, 66: 90, 70: 91, 74: 258, 82: 97, 86: 129, 90: 112, 94: 105, 102: 55, 118: 78, 126: 271}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 24, 28, 64, 68, 72, 76, 88, 92, 100, 108, 116, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4565, 'token_per_expert': {0: 1049, 4: 1024, 8: 128, 24: 253, 28: 70, 64: 346, 68: 149, 72: 290, 76: 73, 88: 69, 92: 153, 100: 603, 108: 116, 116: 121, 120: 121}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 33, 41, 45, 53, 69, 73, 85, 89, 93, 101, 113, 117, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3113, 'token_per_expert': {1: 1081, 5: 1028, 33: 58, 41: 31, 45: 50, 53: 132, 69: 47, 73: 160, 85: 33, 89: 86, 93: 163, 101: 45, 113: 27, 117: 144, 125: 28}}
INFO 05-06 11:02:21.249289.249289 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.870ms | allocate_experts_across_cpu_gpu: 0.453ms
INFO 05-06 11:02:21.249685.249685 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.4849853515625e-05 seconds
INFO 05-06 11:02:21.251903.251903 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010945796966552734 seconds
INFO 05-06 11:02:21.264463.264463 lmp.py:1387] [layer_moe_fused] to time: 0.0001354217529296875 seconds
INFO 05-06 11:02:21.264258.264258 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.266603.266603 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011518001556396484 seconds
INFO 05-06 11:02:21.266286.266286 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005655288696289062 seconds
INFO 05-06 11:02:21.266991.266991 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001867532730102539 seconds
INFO 05-06 11:02:21.276609.276609 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009714126586914062 seconds
INFO 05-06 11:02:21.278498.278498 mlpmodule.py:2799] [fused_experts] gmm total=1.829ms E=32 S=3253 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.278927.278927 mlpmodule.py:2799] [fused_experts] gmm total=2.075ms E=32 S=3946 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.278909.278909 mlpmodule.py:2799] [fused_experts] gmm total=2.191ms E=32 S=4226 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.278321.278321 mlpmodule.py:2799] [fused_experts] gmm total=2.140ms E=32 S=4959 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.280559.280559 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0035588741302490234 seconds
INFO 05-06 11:02:21.280921.280921 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.6743621826171875e-05 seconds
DEBUG 05-06 11:02:21.281557.281557 cuda_h.py:27] end *layer_moe_fused cost 32.667 ms
DEBUG 05-06 11:02:21.286522.286522 cuda_h.py:27] end prefill_layer cost 41.690 ms
DEBUG 05-06 11:02:21.286577.286577 lmp.py:841] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 11:02:21.286486.286486 lmp.py:824] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 11:02:21.289436.289436 cuda_h.py:27] end *sagl cost 2.781 ms
experts_cpu_alloc {'expert_ids': [119, 111, 15, 55, 127, 27, 11, 99, 107, 23, 95, 51, 75, 103, 19, 91, 70, 50, 82, 74, 110, 54, 102, 58, 14, 62, 66, 10, 122, 38, 88, 20, 28, 64, 92, 60, 68, 48, 36, 40, 52, 12, 120, 124, 32, 112, 13, 69, 93, 77, 49, 41, 81, 101, 53, 89, 57, 9, 33, 73], 'token_total': 1070, 'token_per_expert': {119: 1, 111: 3, 15: 4, 55: 5, 127: 8, 27: 9, 11: 11, 99: 12, 107: 18, 23: 22, 95: 22, 51: 24, 75: 24, 103: 25, 19: 35, 91: 41, 70: 1, 50: 4, 82: 5, 74: 7, 110: 8, 54: 9, 102: 10, 58: 11, 14: 15, 62: 19, 66: 21, 10: 28, 122: 32, 38: 35, 88: 6, 20: 8, 28: 9, 64: 11, 92: 11, 60: 16, 68: 18, 48: 21, 36: 23, 40: 23, 52: 25, 12: 26, 120: 31, 124: 33, 32: 35, 112: 40, 13: 3, 69: 3, 93: 4, 77: 5, 49: 8, 41: 9, 81: 11, 101: 12, 53: 20, 89: 26, 57: 28, 9: 39, 33: 47, 73: 50}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 39, 43, 47, 59, 67, 71, 79, 83, 87, 115, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3801, 'token_per_expert': {3: 1185, 7: 1030, 31: 74, 35: 117, 39: 262, 43: 190, 47: 155, 59: 52, 67: 233, 71: 63, 79: 147, 83: 73, 87: 54, 115: 70, 123: 96}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 22, 26, 30, 34, 42, 46, 78, 86, 90, 98, 106, 118], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3629, 'token_per_expert': {2: 1108, 6: 1066, 18: 117, 22: 44, 26: 84, 30: 66, 34: 41, 42: 38, 46: 180, 78: 102, 86: 290, 90: 94, 98: 183, 106: 73, 118: 143}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 16, 24, 44, 56, 72, 76, 80, 84, 100, 104, 108, 116], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3408, 'token_per_expert': {0: 1031, 4: 1028, 8: 87, 16: 96, 24: 61, 44: 128, 56: 346, 72: 94, 76: 52, 80: 64, 84: 92, 100: 65, 104: 86, 108: 119, 116: 59}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 17, 21, 25, 29, 37, 61, 65, 85, 97, 105, 109, 117, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 4476, 'token_per_expert': {1: 1191, 5: 1187, 17: 79, 21: 408, 25: 185, 29: 166, 37: 187, 61: 195, 65: 250, 85: 122, 97: 141, 105: 64, 109: 100, 117: 60, 125: 141}}
INFO 05-06 11:02:21.294083.294083 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 2.711ms | allocate_experts_across_cpu_gpu: 0.446ms
INFO 05-06 11:02:21.294433.294433 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.532669067382812e-05 seconds
INFO 05-06 11:02:21.295387.295387 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011363029479980469 seconds
INFO 05-06 11:02:21.310541.310541 lmp.py:1387] [layer_moe_fused] to time: 0.0001404285430908203 seconds
INFO 05-06 11:02:21.310051.310051 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.311768.311768 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011773109436035156 seconds
INFO 05-06 11:02:21.312604.312604 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005769729614257812 seconds
INFO 05-06 11:02:21.312931.312931 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019023418426513672 seconds
INFO 05-06 11:02:21.321251.321251 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009706497192382812 seconds
INFO 05-06 11:02:21.324483.324483 mlpmodule.py:2799] [fused_experts] gmm total=1.864ms E=32 S=3834 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.324486.324486 mlpmodule.py:2799] [fused_experts] gmm total=1.918ms E=32 S=3744 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.324660.324660 mlpmodule.py:2799] [fused_experts] gmm total=2.095ms E=32 S=4065 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.324009.324009 mlpmodule.py:2799] [fused_experts] gmm total=2.040ms E=32 S=4741 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.325873.325873 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003461122512817383 seconds
INFO 05-06 11:02:21.325798.325798 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.173683166503906e-05 seconds
DEBUG 05-06 11:02:21.326223.326223 cuda_h.py:27] end *layer_moe_fused cost 35.424 ms
DEBUG 05-06 11:02:21.332668.332668 cuda_h.py:27] end prefill_layer cost 45.384 ms
DEBUG 05-06 11:02:21.332684.332684 lmp.py:841] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 11:02:21.332414.332414 lmp.py:824] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 11:02:21.334870.334870 cuda_h.py:27] end *sagl cost 2.045 ms
experts_cpu_alloc {'expert_ids': [51, 123, 59, 95, 99, 15, 55, 87, 115, 47, 107, 31, 119, 79, 22, 54, 58, 78, 126, 10, 46, 38, 102, 18, 66, 26, 62, 42, 106, 88, 24, 112, 84, 116, 80, 68, 76, 124, 96, 104, 28, 92, 120, 8, 40, 69, 85, 101, 113, 117, 125, 93, 21, 25, 89, 65, 61, 53, 57, 105, 9], 'token_total': 842, 'token_per_expert': {51: 2, 123: 4, 59: 6, 95: 6, 99: 9, 15: 10, 55: 10, 87: 12, 115: 14, 47: 16, 107: 17, 31: 19, 119: 24, 79: 25, 22: 2, 54: 2, 58: 6, 78: 8, 126: 8, 10: 9, 46: 17, 38: 18, 102: 18, 18: 20, 66: 20, 26: 21, 62: 21, 42: 30, 106: 43, 88: 1, 24: 2, 112: 6, 84: 7, 116: 7, 80: 8, 68: 11, 76: 11, 124: 17, 96: 20, 104: 21, 28: 25, 92: 25, 120: 32, 8: 33, 40: 36, 69: 1, 85: 1, 101: 1, 113: 1, 117: 1, 125: 1, 93: 3, 21: 4, 25: 7, 89: 10, 65: 11, 61: 12, 53: 24, 57: 25, 105: 30, 9: 31}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 35, 43, 63, 67, 71, 75, 83, 91, 111, 127], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3777, 'token_per_expert': {3: 1027, 7: 1081, 11: 214, 19: 151, 23: 102, 27: 227, 35: 75, 43: 51, 63: 210, 67: 114, 71: 164, 75: 32, 83: 84, 91: 153, 111: 31, 127: 61}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 30, 34, 50, 70, 74, 82, 86, 90, 94, 98, 110, 114, 118, 122], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 31, 'token_total': 4059, 'token_per_expert': {2: 1027, 6: 1229, 30: 74, 34: 154, 50: 64, 70: 271, 74: 60, 82: 63, 86: 79, 90: 463, 94: 131, 98: 107, 110: 85, 114: 134, 118: 55, 122: 63}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 12, 16, 20, 32, 36, 44, 48, 52, 56, 60, 64, 100, 108], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3682, 'token_per_expert': {0: 1024, 4: 1125, 12: 170, 16: 115, 20: 39, 32: 41, 36: 53, 44: 170, 48: 84, 52: 187, 56: 150, 60: 78, 64: 293, 100: 47, 108: 106}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 17, 29, 33, 37, 45, 49, 73, 77, 81, 97, 109, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 4024, 'token_per_expert': {1: 1100, 5: 1081, 13: 54, 17: 68, 29: 86, 33: 198, 37: 65, 45: 130, 49: 31, 73: 131, 77: 145, 81: 43, 97: 298, 109: 71, 121: 523}}
INFO 05-06 11:02:21.337806.337806 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.817ms | allocate_experts_across_cpu_gpu: 0.453ms
INFO 05-06 11:02:21.337672.337672 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.771087646484375e-05 seconds
INFO 05-06 11:02:21.338544.338544 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010981559753417969 seconds
INFO 05-06 11:02:21.350258.350258 lmp.py:1387] [layer_moe_fused] to time: 0.0001316070556640625 seconds
INFO 05-06 11:02:21.350576.350576 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.351876.351876 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011875629425048828 seconds
INFO 05-06 11:02:21.352322.352322 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005652904510498047 seconds
INFO 05-06 11:02:21.352880.352880 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019009113311767578 seconds
INFO 05-06 11:02:21.362344.362344 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009848833084106445 seconds
INFO 05-06 11:02:21.364856.364856 mlpmodule.py:2799] [fused_experts] gmm total=1.997ms E=32 S=3951 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.364059.364059 mlpmodule.py:2799] [fused_experts] gmm total=2.038ms E=32 S=3944 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.364157.364157 mlpmodule.py:2799] [fused_experts] gmm total=2.227ms E=32 S=4302 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.364206.364206 mlpmodule.py:2799] [fused_experts] gmm total=2.193ms E=32 S=4187 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.366833.366833 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0036439895629882812 seconds
INFO 05-06 11:02:21.366625.366625 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.4836273193359375e-05 seconds
DEBUG 05-06 11:02:21.367487.367487 cuda_h.py:27] end *layer_moe_fused cost 31.091 ms
DEBUG 05-06 11:02:21.372014.372014 cuda_h.py:27] end prefill_layer cost 40.461 ms
DEBUG 05-06 11:02:21.372315.372315 lmp.py:841] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 11:02:21.372416.372416 lmp.py:824] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 11:02:21.375249.375249 cuda_h.py:27] end *sagl cost 2.053 ms
experts_cpu_alloc {'expert_ids': [75, 115, 15, 55, 27, 103, 23, 127, 119, 31, 99, 47, 51, 43, 54, 62, 98, 30, 94, 86, 102, 22, 122, 46, 42, 66, 74, 78, 38, 118, 126, 40, 84, 96, 76, 108, 32, 12, 24, 92, 124, 8, 112, 48, 72, 88, 37, 57, 61, 65, 101, 81, 17, 33, 53, 121, 21, 73, 77, 13, 29], 'token_total': 816, 'token_per_expert': {75: 3, 115: 4, 15: 5, 55: 6, 27: 10, 103: 12, 23: 13, 127: 15, 119: 17, 31: 22, 99: 26, 47: 29, 51: 32, 43: 33, 54: 1, 62: 1, 98: 1, 30: 2, 94: 2, 86: 3, 102: 3, 22: 7, 122: 10, 46: 12, 42: 13, 66: 14, 74: 19, 78: 21, 38: 22, 118: 30, 126: 30, 40: 1, 84: 2, 96: 2, 76: 9, 108: 13, 32: 14, 12: 15, 24: 15, 92: 19, 124: 21, 8: 23, 112: 24, 48: 28, 72: 32, 88: 43, 37: 1, 57: 4, 61: 4, 65: 4, 101: 5, 81: 6, 17: 7, 33: 8, 53: 8, 121: 8, 21: 12, 73: 14, 77: 16, 13: 18, 29: 22}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 35, 39, 63, 67, 71, 79, 83, 87, 91, 107, 111, 123], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 30, 'token_total': 3787, 'token_per_expert': {3: 1160, 7: 1085, 11: 74, 19: 51, 35: 222, 39: 53, 63: 68, 67: 79, 71: 65, 79: 44, 83: 117, 87: 43, 91: 81, 107: 339, 111: 76, 123: 230}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 14, 18, 26, 34, 50, 58, 70, 82, 90, 106, 110, 114], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 32, 'token_total': 4232, 'token_per_expert': {2: 1298, 6: 1044, 10: 92, 14: 58, 18: 217, 26: 32, 34: 62, 50: 57, 58: 532, 70: 161, 82: 70, 90: 112, 106: 49, 110: 348, 114: 100}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 16, 36, 44, 52, 56, 60, 64, 68, 80, 100, 104, 116, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4232, 'token_per_expert': {0: 1061, 4: 1031, 16: 610, 36: 65, 44: 74, 52: 188, 56: 51, 60: 169, 64: 111, 68: 363, 80: 159, 100: 72, 104: 162, 116: 54, 120: 62}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 25, 41, 45, 49, 69, 85, 89, 93, 97, 109, 117, 125], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3317, 'token_per_expert': {1: 1031, 5: 1057, 9: 34, 25: 37, 41: 61, 45: 215, 49: 58, 69: 170, 85: 206, 89: 78, 93: 58, 97: 54, 109: 39, 117: 192, 125: 27}}
INFO 05-06 11:02:21.377435.377435 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.584ms | allocate_experts_across_cpu_gpu: 0.453ms
INFO 05-06 11:02:21.377162.377162 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.937980651855469e-05 seconds
INFO 05-06 11:02:21.378470.378470 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010950565338134766 seconds
INFO 05-06 11:02:21.390247.390247 lmp.py:1387] [layer_moe_fused] to time: 0.0001327991485595703 seconds
INFO 05-06 11:02:21.390082.390082 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.391832.391832 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012035369873046875 seconds
INFO 05-06 11:02:21.392511.392511 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006008148193359375 seconds
INFO 05-06 11:02:21.392844.392844 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.001957416534423828 seconds
INFO 05-06 11:02:21.402384.402384 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00976419448852539 seconds
INFO 05-06 11:02:21.404996.404996 mlpmodule.py:2799] [fused_experts] gmm total=2.003ms E=32 S=4014 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.404347.404347 mlpmodule.py:2799] [fused_experts] gmm total=2.074ms E=32 S=4493 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.404609.404609 mlpmodule.py:2799] [fused_experts] gmm total=2.139ms E=32 S=3454 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.405422.405422 mlpmodule.py:2799] [fused_experts] gmm total=2.391ms E=32 S=4423 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.405096.405096 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0035619735717773438 seconds
INFO 05-06 11:02:21.406273.406273 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.555152893066406e-05 seconds
DEBUG 05-06 11:02:21.406104.406104 cuda_h.py:27] end *layer_moe_fused cost 30.499 ms
DEBUG 05-06 11:02:21.412322.412322 cuda_h.py:27] end prefill_layer cost 39.703 ms
DEBUG 05-06 11:02:21.412808.412808 lmp.py:841] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 11:02:21.412001.412001 lmp.py:824] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 11:02:21.415840.415840 cuda_h.py:27] end *sagl cost 2.020 ms
experts_cpu_alloc {'expert_ids': [119, 11, 127, 55, 107, 47, 83, 23, 63, 31, 91, 71, 35, 115, 67, 75, 22, 82, 98, 34, 58, 54, 46, 18, 74, 122, 110, 42, 118, 38, 16, 32, 48, 100, 28, 92, 40, 80, 116, 44, 8, 72, 68, 36, 93, 53, 121, 33, 117, 9, 109, 125, 41, 81, 13, 77, 61, 29, 25], 'token_total': 937, 'token_per_expert': {119: 1, 11: 2, 127: 2, 55: 4, 107: 6, 47: 8, 83: 8, 23: 10, 63: 10, 31: 13, 91: 14, 71: 18, 35: 21, 115: 22, 67: 29, 75: 41, 22: 1, 82: 2, 98: 2, 34: 5, 58: 5, 54: 8, 46: 9, 18: 16, 74: 19, 122: 19, 110: 20, 42: 26, 118: 27, 38: 45, 16: 3, 32: 3, 48: 5, 100: 7, 28: 8, 92: 10, 40: 13, 80: 13, 116: 17, 44: 19, 8: 29, 72: 30, 68: 31, 36: 33, 93: 1, 53: 3, 121: 3, 33: 4, 117: 4, 9: 7, 109: 19, 125: 22, 41: 26, 81: 28, 13: 29, 77: 30, 61: 40, 29: 43, 25: 44}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 27, 43, 51, 59, 79, 87, 95, 99, 103, 111, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3950, 'token_per_expert': {3: 1073, 7: 1024, 15: 101, 19: 91, 27: 208, 43: 186, 51: 76, 59: 69, 79: 49, 87: 303, 95: 246, 99: 54, 103: 57, 111: 313, 123: 100}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 14, 26, 30, 50, 66, 70, 78, 86, 90, 102, 114, 126], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3318, 'token_per_expert': {2: 1039, 6: 1024, 10: 52, 14: 66, 26: 50, 30: 48, 50: 82, 66: 67, 70: 122, 78: 110, 86: 85, 90: 74, 102: 88, 114: 338, 126: 73}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 20, 24, 52, 56, 60, 76, 84, 88, 96, 104, 108, 112, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3598, 'token_per_expert': {0: 1043, 4: 1033, 20: 352, 24: 245, 52: 87, 56: 84, 60: 91, 76: 85, 84: 223, 88: 54, 96: 34, 104: 121, 108: 39, 112: 39, 124: 68}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 17, 37, 45, 49, 57, 65, 73, 85, 89, 97, 105, 113], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 4581, 'token_per_expert': {1: 1047, 5: 1043, 17: 283, 37: 45, 45: 52, 49: 102, 57: 48, 65: 240, 73: 155, 85: 602, 89: 406, 97: 68, 105: 85, 113: 405}}
INFO 05-06 11:02:21.417140.417140 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.833ms | allocate_experts_across_cpu_gpu: 0.439ms
INFO 05-06 11:02:21.417760.417760 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.270408630371094e-05 seconds
INFO 05-06 11:02:21.418145.418145 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010755062103271484 seconds
INFO 05-06 11:02:21.431614.431614 lmp.py:1387] [layer_moe_fused] to time: 0.00013899803161621094 seconds
INFO 05-06 11:02:21.431959.431959 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.432021.432021 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011851787567138672 seconds
INFO 05-06 11:02:21.433142.433142 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005738735198974609 seconds
INFO 05-06 11:02:21.433938.433938 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019047260284423828 seconds
INFO 05-06 11:02:21.443742.443742 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010097980499267578 seconds
INFO 05-06 11:02:21.446756.446756 mlpmodule.py:2799] [fused_experts] gmm total=1.857ms E=32 S=3819 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.446648.446648 mlpmodule.py:2799] [fused_experts] gmm total=2.044ms E=32 S=3522 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.446103.446103 mlpmodule.py:2799] [fused_experts] gmm total=2.229ms E=32 S=4159 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.446929.446929 mlpmodule.py:2799] [fused_experts] gmm total=2.216ms E=32 S=4884 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.447293.447293 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003721475601196289 seconds
INFO 05-06 11:02:21.447224.447224 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.435943603515625e-05 seconds
DEBUG 05-06 11:02:21.448247.448247 cuda_h.py:27] end *layer_moe_fused cost 32.357 ms
DEBUG 05-06 11:02:21.454115.454115 cuda_h.py:27] end prefill_layer cost 41.191 ms
DEBUG 05-06 11:02:21.454840.454840 lmp.py:841] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 11:02:21.454464.454464 lmp.py:824] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 11:02:21.456945.456945 cuda_h.py:27] end *sagl cost 2.038 ms
experts_cpu_alloc {'expert_ids': [67, 63, 39, 47, 59, 11, 15, 27, 19, 71, 91, 55, 23, 75, 127, 22, 34, 30, 86, 110, 126, 10, 58, 74, 90, 94, 26, 114, 106, 54, 72, 16, 44, 84, 96, 116, 124, 32, 68, 60, 80, 40, 56, 20, 9, 17, 77, 81, 89, 93, 101, 57, 97, 29, 69, 113, 125, 21, 49], 'token_total': 1133, 'token_per_expert': {67: 4, 63: 11, 39: 12, 47: 14, 59: 14, 11: 16, 15: 17, 27: 18, 19: 21, 71: 23, 91: 27, 55: 32, 23: 45, 75: 61, 127: 62, 22: 3, 34: 3, 30: 6, 86: 7, 110: 8, 126: 12, 10: 23, 58: 24, 74: 24, 90: 40, 94: 44, 26: 45, 114: 45, 106: 49, 54: 50, 72: 3, 16: 5, 44: 5, 84: 8, 96: 10, 116: 11, 124: 12, 32: 16, 68: 16, 60: 17, 80: 19, 40: 40, 56: 42, 20: 43, 9: 2, 17: 2, 77: 2, 81: 2, 89: 2, 93: 3, 101: 3, 57: 4, 97: 4, 29: 6, 69: 6, 113: 13, 125: 15, 21: 28, 49: 34}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 43, 51, 79, 83, 87, 95, 103, 111, 115, 119, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4308, 'token_per_expert': {3: 1126, 7: 1088, 31: 128, 35: 91, 43: 268, 51: 128, 79: 128, 83: 70, 87: 356, 95: 172, 103: 214, 111: 122, 115: 186, 119: 87, 123: 144}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 18, 42, 46, 50, 62, 66, 70, 78, 82, 98, 118, 122], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3497, 'token_per_expert': {2: 1027, 6: 1031, 14: 79, 18: 76, 42: 74, 46: 120, 50: 261, 62: 133, 66: 72, 70: 117, 78: 154, 82: 165, 98: 70, 118: 63, 122: 55}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 12, 24, 28, 36, 48, 64, 76, 88, 100, 108, 112, 120], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 29, 'token_total': 3747, 'token_per_expert': {0: 1037, 4: 1066, 8: 85, 12: 83, 24: 199, 28: 70, 36: 106, 48: 152, 64: 100, 76: 142, 88: 184, 100: 187, 108: 72, 112: 46, 120: 218}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 25, 33, 37, 41, 45, 53, 61, 65, 85, 105, 109, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3699, 'token_per_expert': {1: 1114, 5: 1025, 13: 124, 25: 128, 33: 200, 37: 156, 41: 90, 45: 254, 53: 70, 61: 93, 65: 189, 85: 44, 105: 49, 109: 85, 121: 78}}
INFO 05-06 11:02:21.458912.458912 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.581ms | allocate_experts_across_cpu_gpu: 0.440ms
INFO 05-06 11:02:21.458215.458215 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.628036499023438e-05 seconds
INFO 05-06 11:02:21.460762.460762 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011286735534667969 seconds
INFO 05-06 11:02:21.474784.474784 lmp.py:1387] [layer_moe_fused] to time: 0.0001392364501953125 seconds
INFO 05-06 11:02:21.474294.474294 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.476637.476637 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018863677978515625 seconds
INFO 05-06 11:02:21.477322.477322 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00060272216796875 seconds
INFO 05-06 11:02:21.477072.477072 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002636432647705078 seconds
INFO 05-06 11:02:21.487065.487065 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009429454803466797 seconds
INFO 05-06 11:02:21.489764.489764 mlpmodule.py:2799] [fused_experts] gmm total=1.912ms E=32 S=3994 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.489801.489801 mlpmodule.py:2799] [fused_experts] gmm total=2.100ms E=32 S=3880 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.489064.489064 mlpmodule.py:2799] [fused_experts] gmm total=2.110ms E=32 S=3825 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.489450.489450 mlpmodule.py:2799] [fused_experts] gmm total=2.356ms E=32 S=4685 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.490787.490787 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0034618377685546875 seconds
INFO 05-06 11:02:21.490632.490632 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.6743621826171875e-05 seconds
DEBUG 05-06 11:02:21.491123.491123 cuda_h.py:27] end *layer_moe_fused cost 34.043 ms
DEBUG 05-06 11:02:21.497240.497240 cuda_h.py:27] end prefill_layer cost 42.943 ms
DEBUG 05-06 11:02:21.497971.497971 lmp.py:841] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 11:02:21.497356.497356 lmp.py:824] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 11:02:21.499544.499544 cuda_h.py:27] end *sagl cost 2.015 ms
experts_cpu_alloc {'expert_ids': [63, 83, 103, 19, 31, 35, 99, 15, 67, 87, 59, 51, 39, 43, 127, 26, 38, 102, 82, 114, 118, 66, 54, 58, 50, 122, 98, 74, 28, 80, 8, 56, 124, 120, 72, 108, 36, 92, 48, 44, 100, 88, 17, 29, 45, 61, 93, 21, 73, 81, 109, 105, 65, 117, 33, 9, 69], 'token_total': 604, 'token_per_expert': {63: 1, 83: 1, 103: 1, 19: 2, 31: 2, 35: 2, 99: 3, 15: 5, 67: 5, 87: 5, 59: 6, 51: 7, 39: 9, 43: 24, 127: 24, 26: 1, 38: 1, 102: 1, 82: 2, 114: 2, 118: 3, 66: 4, 54: 7, 58: 9, 50: 12, 122: 20, 98: 23, 74: 26, 28: 1, 80: 2, 8: 3, 56: 3, 124: 3, 120: 4, 72: 5, 108: 7, 36: 10, 92: 10, 48: 19, 44: 30, 100: 35, 88: 36, 17: 2, 29: 2, 45: 2, 61: 2, 93: 2, 21: 4, 73: 10, 81: 11, 109: 13, 105: 26, 65: 27, 117: 28, 33: 30, 9: 32, 69: 37}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 47, 55, 71, 75, 79, 91, 95, 111, 115, 119, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 4239, 'token_per_expert': {3: 1031, 7: 1027, 11: 115, 23: 35, 47: 146, 55: 65, 71: 141, 75: 139, 79: 25, 91: 217, 95: 39, 111: 782, 115: 293, 119: 116, 123: 68}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 22, 30, 46, 62, 70, 78, 90, 94, 106, 110, 126], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 27, 'token_total': 3128, 'token_per_expert': {2: 1024, 6: 1026, 18: 55, 22: 93, 30: 57, 46: 112, 62: 50, 70: 113, 78: 138, 90: 207, 94: 35, 106: 62, 110: 118, 126: 38}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 12, 20, 24, 32, 40, 52, 60, 68, 76, 84, 104, 112], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 28, 'token_total': 4670, 'token_per_expert': {0: 1024, 4: 1024, 12: 728, 20: 608, 24: 46, 32: 86, 40: 195, 52: 148, 60: 41, 68: 108, 76: 310, 84: 51, 104: 66, 112: 235}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 37, 49, 53, 57, 77, 85, 89, 97, 101, 113, 121], 'expert_count': 14, 'ideal_gpu_count': 14, 'keep_on_gpu': 14, 'hit_count_on_device': 29, 'token_total': 3743, 'token_per_expert': {1: 1064, 5: 1079, 13: 78, 37: 61, 49: 581, 53: 146, 57: 276, 77: 100, 85: 51, 89: 67, 97: 45, 101: 75, 113: 79, 121: 41}}
INFO 05-06 11:02:21.501388.501388 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.819ms | allocate_experts_across_cpu_gpu: 0.429ms
INFO 05-06 11:02:21.502068.502068 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.461143493652344e-05 seconds
INFO 05-06 11:02:21.503886.503886 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010716915130615234 seconds
INFO 05-06 11:02:21.513481.513481 lmp.py:1387] [layer_moe_fused] to time: 0.0001266002655029297 seconds
INFO 05-06 11:02:21.513786.513786 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.514410.514410 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001180410385131836 seconds
INFO 05-06 11:02:21.515485.515485 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005767345428466797 seconds
INFO 05-06 11:02:21.515281.515281 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0019021034240722656 seconds
INFO 05-06 11:02:21.524759.524759 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009472370147705078 seconds
INFO 05-06 11:02:21.526605.526605 mlpmodule.py:2799] [fused_experts] gmm total=1.840ms E=32 S=3239 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.527075.527075 mlpmodule.py:2799] [fused_experts] gmm total=2.068ms E=32 S=4336 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.527979.527979 mlpmodule.py:2799] [fused_experts] gmm total=2.014ms E=32 S=3971 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.527206.527206 mlpmodule.py:2799] [fused_experts] gmm total=2.124ms E=32 S=4838 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.528744.528744 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0034132003784179688 seconds
INFO 05-06 11:02:21.528907.528907 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.1975250244140625e-05 seconds
DEBUG 05-06 11:02:21.529923.529923 cuda_h.py:27] end *layer_moe_fused cost 28.717 ms
DEBUG 05-06 11:02:21.534215.534215 cuda_h.py:27] end prefill_layer cost 37.492 ms
DEBUG 05-06 11:02:21.534555.534555 lmp.py:841] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 11:02:21.534940.534940 lmp.py:824] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 11:02:21.537030.537030 cuda_h.py:27] end *sagl cost 2.757 ms
experts_cpu_alloc {'expert_ids': [79, 39, 111, 127, 55, 87, 75, 63, 119, 83, 31, 35, 11, 118, 38, 122, 74, 126, 70, 102, 34, 98, 94, 46, 50, 58, 114, 10, 66, 72, 12, 112, 68, 100, 40, 88, 8, 76, 24, 84, 96, 32, 108, 80, 17, 37, 125, 65, 33, 13, 25, 105, 21, 89, 9, 73, 69, 101, 77], 'token_total': 1081, 'token_per_expert': {79: 2, 39: 4, 111: 5, 127: 7, 55: 9, 87: 13, 75: 18, 63: 19, 119: 19, 83: 25, 31: 26, 35: 26, 11: 34, 118: 1, 38: 2, 122: 2, 74: 5, 126: 5, 70: 6, 102: 6, 34: 7, 98: 7, 94: 14, 46: 15, 50: 16, 58: 16, 114: 21, 10: 25, 66: 28, 72: 1, 12: 2, 112: 2, 68: 3, 100: 3, 40: 8, 88: 12, 8: 13, 76: 17, 24: 20, 84: 23, 96: 28, 32: 31, 108: 35, 80: 36, 17: 3, 37: 8, 125: 10, 65: 11, 33: 12, 13: 24, 25: 25, 105: 28, 21: 32, 89: 40, 9: 50, 73: 53, 69: 55, 101: 56, 77: 57}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 43, 67, 71, 91, 95, 99, 107, 115, 123], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 28, 'token_total': 4222, 'token_per_expert': {3: 1052, 7: 1410, 15: 55, 19: 216, 23: 131, 27: 144, 43: 208, 67: 63, 71: 113, 91: 348, 95: 62, 99: 254, 107: 59, 115: 45, 123: 62}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 18, 22, 26, 30, 42, 54, 62, 78, 82, 86, 90, 106], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 31, 'token_total': 3791, 'token_per_expert': {2: 1070, 6: 1044, 14: 78, 18: 139, 22: 147, 26: 149, 30: 79, 42: 190, 54: 86, 62: 114, 78: 47, 82: 74, 86: 219, 90: 103, 106: 252}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 16, 20, 28, 44, 48, 52, 56, 60, 64, 92, 116, 120, 124], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3904, 'token_per_expert': {0: 1028, 4: 1245, 16: 102, 20: 216, 28: 145, 44: 45, 48: 74, 52: 273, 56: 125, 60: 115, 64: 307, 92: 48, 116: 44, 120: 38, 124: 99}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 29, 49, 53, 57, 61, 81, 85, 93, 97, 109, 113, 117, 121], 'expert_count': 15, 'ideal_gpu_count': 15, 'keep_on_gpu': 15, 'hit_count_on_device': 30, 'token_total': 3386, 'token_per_expert': {1: 1058, 5: 1032, 29: 93, 49: 101, 53: 67, 57: 161, 61: 106, 81: 88, 85: 69, 93: 74, 97: 125, 109: 59, 113: 84, 117: 131, 121: 138}}
INFO 05-06 11:02:21.542792.542792 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 2.768ms | allocate_experts_across_cpu_gpu: 0.440ms
INFO 05-06 11:02:21.542572.542572 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.747245788574219e-05 seconds
INFO 05-06 11:02:21.543062.543062 lmp.py:1377] [layer_moe_fused] kt_kernel_prep_submit time: 0.0011019706726074219 seconds
INFO 05-06 11:02:21.557423.557423 lmp.py:1387] [layer_moe_fused] to time: 0.00014019012451171875 seconds
INFO 05-06 11:02:21.557179.557179 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.559638.559638 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011992454528808594 seconds
INFO 05-06 11:02:21.559027.559027 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00066375732421875 seconds
INFO 05-06 11:02:21.560499.560499 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0020177364349365234 seconds
INFO 05-06 11:02:21.569959.569959 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009528398513793945 seconds
INFO 05-06 11:02:21.571972.571972 mlpmodule.py:2799] [fused_experts] gmm total=2.119ms E=32 S=4429 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.572936.572936 mlpmodule.py:2799] [fused_experts] gmm total=2.069ms E=32 S=3850 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.572746.572746 mlpmodule.py:2799] [fused_experts] gmm total=2.240ms E=32 S=3967 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.572710.572710 mlpmodule.py:2799] [fused_experts] gmm total=2.305ms E=32 S=4138 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.573374.573374 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003586292266845703 seconds
INFO 05-06 11:02:21.573457.573457 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.555152893066406e-05 seconds
DEBUG 05-06 11:02:21.574883.574883 cuda_h.py:27] end *layer_moe_fused cost 35.193 ms
DEBUG 05-06 11:02:21.579694.579694 cuda_h.py:27] end prefill_layer cost 45.035 ms
DEBUG 05-06 11:02:21.579319.579319 lmp.py:841] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 11:02:21.579750.579750 cuda_h.py:27] end prefill_step cost 2001.186 ms
INFO 05-06 11:02:21.580188.580188 lmp.py:843] prefill time: 2.1356775760650635 seconds
WARNING 05-06 11:02:21.605448.605448 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:02:21.605933.605933 helper.py:35]   NaN count (hidden): 2883584
WARNING 05-06 11:02:21.606628.606628 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:02:21.606799.606799 helper.py:39]   NaN count (normed): 2883584
WARNING 05-06 11:02:21.611825.611825 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:02:21.611080.611080 helper.py:50]   NaN count: 524288
WARNING 05-06 11:02:21.611141.611141 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 11:02:21.613845.613845 cuda_h.py:27] end init_inputs_tokens cost 7.902 ms
DEBUG 05-06 11:02:21.613258.613258 lmp.py:899] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:02:21.613498.613498 lmp.py:904] ---- decode step 0 layer 0 ----
DEBUG 05-06 11:02:21.615779.615779 cuda_h.py:27] end *sagl cost 1.649 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 39, 47, 55, 63, 79, 83, 87, 91, 127], 'expert_count': 11, 'ideal_gpu_count': 7, 'keep_on_gpu': 11, 'hit_count_on_device': 11, 'token_total': 13, 'token_per_expert': {7: 1, 15: 1, 39: 1, 47: 2, 55: 1, 63: 1, 79: 1, 83: 1, 87: 1, 91: 1, 127: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [22, 74, 90, 126], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {22: 2, 74: 1, 90: 2, 126: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [8, 32, 48, 60, 68, 116, 124], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 7, 'token_per_expert': {8: 1, 32: 1, 48: 1, 60: 1, 68: 1, 116: 1, 124: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [25, 33, 45, 53], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {25: 1, 33: 2, 45: 1, 53: 2}}
INFO 05-06 11:02:21.616310.616310 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.370ms | allocate_experts_across_cpu_gpu: 0.119ms
INFO 05-06 11:02:21.616332.616332 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3126602172851562e-05 seconds
INFO 05-06 11:02:21.616333.616333 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.617101.617101 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009577274322509766 seconds
INFO 05-06 11:02:21.618172.618172 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0004925727844238281 seconds
INFO 05-06 11:02:21.618094.618094 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0015530586242675781 seconds
INFO 05-06 11:02:21.620064.620064 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0018420219421386719 seconds
INFO 05-06 11:02:21.621066.621066 mlpmodule.py:2799] [fused_experts] gmm total=0.982ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.621435.621435 mlpmodule.py:2799] [fused_experts] gmm total=1.001ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.621524.621524 mlpmodule.py:2799] [fused_experts] gmm total=1.156ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.621367.621367 mlpmodule.py:2799] [fused_experts] gmm total=1.330ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.622111.622111 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0023145675659179688 seconds
INFO 05-06 11:02:21.622890.622890 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9591064453125e-05 seconds
DEBUG 05-06 11:02:21.623746.623746 cuda_h.py:27] end *layer_moe_fused cost 7.205 ms
DEBUG 05-06 11:02:21.623912.623912 cuda_h.py:27] end decode_layer cost 10.201 ms
DEBUG 05-06 11:02:21.623947.623947 lmp.py:904] ---- decode step 0 layer 1 ----
DEBUG 05-06 11:02:21.625365.625365 cuda_h.py:27] end *sagl cost 1.569 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 107, 119, 123], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 107: 1, 119: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 30, 54, 110], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 30: 1, 54: 1, 110: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 56, 92, 96, 124], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {0: 4, 4: 2, 8: 1, 56: 1, 92: 1, 96: 1, 124: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 73, 121], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 2, 5: 2, 9: 1, 73: 1, 121: 1}}
INFO 05-06 11:02:21.626654.626654 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.318ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 11:02:21.626505.626505 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 11:02:21.626546.626546 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.628697.628697 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013120174407958984 seconds
INFO 05-06 11:02:21.629370.629370 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008642673492431641 seconds
INFO 05-06 11:02:21.629530.629530 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002281665802001953 seconds
INFO 05-06 11:02:21.630587.630587 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012743473052978516 seconds
INFO 05-06 11:02:21.631284.631284 mlpmodule.py:2799] [fused_experts] gmm total=1.248ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.632473.632473 mlpmodule.py:2799] [fused_experts] gmm total=1.289ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.632610.632610 mlpmodule.py:2799] [fused_experts] gmm total=1.318ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.632911.632911 mlpmodule.py:2799] [fused_experts] gmm total=1.459ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.633256.633256 lmp.py:1500] [layer_moe_fused] experts compute time: 0.002498149871826172 seconds
INFO 05-06 11:02:21.633716.633716 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.1484832763671875e-05 seconds
DEBUG 05-06 11:02:21.633714.633714 cuda_h.py:27] end *layer_moe_fused cost 7.343 ms
DEBUG 05-06 11:02:21.634892.634892 cuda_h.py:27] end decode_layer cost 10.207 ms
DEBUG 05-06 11:02:21.634682.634682 lmp.py:904] ---- decode step 0 layer 2 ----
DEBUG 05-06 11:02:21.635701.635701 cuda_h.py:27] end *sagl cost 1.521 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 91], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 11: 1, 91: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 62, 70, 90, 102, 106, 126], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {2: 2, 6: 2, 62: 1, 70: 1, 90: 1, 102: 1, 106: 1, 126: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 12, 76], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {0: 2, 4: 2, 8: 1, 12: 1, 76: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 41, 45, 49, 81], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 41: 1, 45: 1, 49: 1, 81: 1}}
INFO 05-06 11:02:21.636174.636174 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.318ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 11:02:21.637951.637951 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 11:02:21.637515.637515 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.638870.638870 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012826919555664062 seconds
INFO 05-06 11:02:21.639658.639658 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009145736694335938 seconds
INFO 05-06 11:02:21.639341.639341 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0023009777069091797 seconds
INFO 05-06 11:02:21.640432.640432 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001299142837524414 seconds
INFO 05-06 11:02:21.642010.642010 mlpmodule.py:2799] [fused_experts] gmm total=1.645ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.642797.642797 mlpmodule.py:2799] [fused_experts] gmm total=1.736ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.642613.642613 mlpmodule.py:2799] [fused_experts] gmm total=1.915ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.643073.643073 mlpmodule.py:2799] [fused_experts] gmm total=1.953ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.643148.643148 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0030896663665771484 seconds
INFO 05-06 11:02:21.643449.643449 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 11:02:21.644976.644976 cuda_h.py:27] end *layer_moe_fused cost 7.961 ms
DEBUG 05-06 11:02:21.644591.644591 cuda_h.py:27] end decode_layer cost 10.742 ms
DEBUG 05-06 11:02:21.644242.644242 lmp.py:904] ---- decode step 0 layer 3 ----
DEBUG 05-06 11:02:21.646786.646786 cuda_h.py:27] end *sagl cost 1.556 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {3: 2, 7: 2, 39: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 26, 50, 54, 110, 126], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 2, 26: 1, 50: 1, 54: 1, 110: 1, 126: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 24, 40, 96, 104], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 24: 1, 40: 1, 96: 2, 104: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 73, 101, 117, 125], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 3, 73: 1, 101: 1, 117: 1, 125: 1}}
INFO 05-06 11:02:21.647134.647134 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.334ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 11:02:21.647435.647435 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 11:02:21.647522.647522 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.649902.649902 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012354850769042969 seconds
INFO 05-06 11:02:21.650876.650876 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009458065032958984 seconds
INFO 05-06 11:02:21.650037.650037 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0022864341735839844 seconds
INFO 05-06 11:02:21.651312.651312 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012586116790771484 seconds
INFO 05-06 11:02:21.653222.653222 mlpmodule.py:2799] [fused_experts] gmm total=1.224ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.653977.653977 mlpmodule.py:2799] [fused_experts] gmm total=1.860ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.654084.654084 mlpmodule.py:2799] [fused_experts] gmm total=2.062ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.654495.654495 mlpmodule.py:2799] [fused_experts] gmm total=2.547ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.655176.655176 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0035860538482666016 seconds
INFO 05-06 11:02:21.655478.655478 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 11:02:21.655846.655846 cuda_h.py:27] end *layer_moe_fused cost 8.433 ms
DEBUG 05-06 11:02:21.656753.656753 cuda_h.py:27] end decode_layer cost 11.291 ms
DEBUG 05-06 11:02:21.656497.656497 lmp.py:904] ---- decode step 0 layer 4 ----
DEBUG 05-06 11:02:21.657197.657197 cuda_h.py:27] end *sagl cost 1.496 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 51, 87], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {3: 3, 7: 2, 51: 1, 87: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 50, 106, 114, 122, 126], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 3, 6: 2, 50: 1, 106: 1, 114: 1, 122: 1, 126: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 20, 60], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 20: 1, 60: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 17, 25, 45, 93, 121], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 17: 1, 25: 1, 45: 1, 93: 1, 121: 1}}
INFO 05-06 11:02:21.659080.659080 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 11:02:21.659858.659858 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8596649169921875e-05 seconds
INFO 05-06 11:02:21.659183.659183 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.660448.660448 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013611316680908203 seconds
INFO 05-06 11:02:21.662207.662207 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014531612396240234 seconds
INFO 05-06 11:02:21.662858.662858 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0029311180114746094 seconds
INFO 05-06 11:02:21.663279.663279 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001257181167602539 seconds
INFO 05-06 11:02:21.665904.665904 mlpmodule.py:2799] [fused_experts] gmm total=1.669ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.665968.665968 mlpmodule.py:2799] [fused_experts] gmm total=1.907ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.665720.665720 mlpmodule.py:2799] [fused_experts] gmm total=2.122ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.666611.666611 mlpmodule.py:2799] [fused_experts] gmm total=2.342ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.667028.667028 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0037102699279785156 seconds
INFO 05-06 11:02:21.667568.667568 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 11:02:21.667239.667239 cuda_h.py:27] end *layer_moe_fused cost 9.129 ms
DEBUG 05-06 11:02:21.668716.668716 cuda_h.py:27] end decode_layer cost 11.884 ms
DEBUG 05-06 11:02:21.668652.668652 lmp.py:904] ---- decode step 0 layer 5 ----
DEBUG 05-06 11:02:21.669973.669973 cuda_h.py:27] end *sagl cost 1.462 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 71, 95, 99, 123], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 39: 1, 71: 1, 95: 1, 99: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 46, 70, 94], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 3, 6: 2, 46: 1, 70: 1, 94: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 36, 52, 72, 116], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 2, 4: 2, 36: 1, 52: 1, 72: 1, 116: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 61, 65], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 2, 5: 3, 61: 1, 65: 1}}
INFO 05-06 11:02:21.670890.670890 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.314ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 11:02:21.671767.671767 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9550323486328125e-05 seconds
INFO 05-06 11:02:21.671106.671106 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.672746.672746 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014486312866210938 seconds
INFO 05-06 11:02:21.674805.674805 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014677047729492188 seconds
INFO 05-06 11:02:21.674933.674933 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0031147003173828125 seconds
INFO 05-06 11:02:21.675493.675493 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001255035400390625 seconds
INFO 05-06 11:02:21.677182.677182 mlpmodule.py:2799] [fused_experts] gmm total=2.001ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.678971.678971 mlpmodule.py:2799] [fused_experts] gmm total=2.329ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.678110.678110 mlpmodule.py:2799] [fused_experts] gmm total=2.338ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.678478.678478 mlpmodule.py:2799] [fused_experts] gmm total=2.346ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.679063.679063 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004258394241333008 seconds
INFO 05-06 11:02:21.680841.680841 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 11:02:21.680394.680394 cuda_h.py:27] end *layer_moe_fused cost 9.909 ms
DEBUG 05-06 11:02:21.680302.680302 cuda_h.py:27] end decode_layer cost 12.693 ms
DEBUG 05-06 11:02:21.680860.680860 lmp.py:904] ---- decode step 0 layer 6 ----
DEBUG 05-06 11:02:21.682695.682695 cuda_h.py:27] end *sagl cost 1.558 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 43, 87, 115], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 35: 1, 43: 1, 87: 2, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 78, 106, 118], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 3, 6: 2, 78: 1, 106: 1, 118: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 24, 32, 68, 96], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 2, 4: 2, 24: 1, 32: 1, 68: 1, 96: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 25], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 3, 5: 2, 13: 1, 25: 1}}
INFO 05-06 11:02:21.683143.683143 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.337ms | allocate_experts_across_cpu_gpu: 0.101ms
INFO 05-06 11:02:21.683067.683067 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 11:02:21.684015.684015 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.685035.685035 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011789798736572266 seconds
INFO 05-06 11:02:21.686631.686631 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011205673217773438 seconds
INFO 05-06 11:02:21.686898.686898 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0024156570434570312 seconds
INFO 05-06 11:02:21.687539.687539 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001316070556640625 seconds
INFO 05-06 11:02:21.690013.690013 mlpmodule.py:2799] [fused_experts] gmm total=2.063ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.690525.690525 mlpmodule.py:2799] [fused_experts] gmm total=2.243ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.690796.690796 mlpmodule.py:2799] [fused_experts] gmm total=2.309ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.690396.690396 mlpmodule.py:2799] [fused_experts] gmm total=2.349ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.691485.691485 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0038526058197021484 seconds
INFO 05-06 11:02:21.691595.691595 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.792213439941406e-05 seconds
DEBUG 05-06 11:02:21.692774.692774 cuda_h.py:27] end *layer_moe_fused cost 8.981 ms
DEBUG 05-06 11:02:21.692484.692484 cuda_h.py:27] end decode_layer cost 11.902 ms
DEBUG 05-06 11:02:21.692612.692612 lmp.py:904] ---- decode step 0 layer 7 ----
DEBUG 05-06 11:02:21.694790.694790 cuda_h.py:27] end *sagl cost 1.529 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 43], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 19: 1, 43: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 18, 34, 90, 106, 114], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {2: 2, 6: 2, 10: 1, 18: 1, 34: 1, 90: 1, 106: 1, 114: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 20, 64, 96, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 2, 4: 2, 20: 1, 64: 1, 96: 1, 104: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 69, 97, 121], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 9: 1, 69: 1, 97: 1, 121: 1}}
INFO 05-06 11:02:21.696273.696273 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.789ms | allocate_experts_across_cpu_gpu: 0.110ms
INFO 05-06 11:02:21.696893.696893 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0742416381835938e-05 seconds
INFO 05-06 11:02:21.696172.696172 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.697882.697882 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014081001281738281 seconds
INFO 05-06 11:02:21.699879.699879 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014157295227050781 seconds
INFO 05-06 11:02:21.699245.699245 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002943277359008789 seconds
INFO 05-06 11:02:21.700130.700130 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014944076538085938 seconds
INFO 05-06 11:02:21.703936.703936 mlpmodule.py:2799] [fused_experts] gmm total=1.948ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.703078.703078 mlpmodule.py:2799] [fused_experts] gmm total=2.313ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.703192.703192 mlpmodule.py:2799] [fused_experts] gmm total=2.251ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.703074.703074 mlpmodule.py:2799] [fused_experts] gmm total=2.587ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.705370.705370 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004106044769287109 seconds
INFO 05-06 11:02:21.705963.705963 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.340576171875e-05 seconds
DEBUG 05-06 11:02:21.705251.705251 cuda_h.py:27] end *layer_moe_fused cost 10.335 ms
DEBUG 05-06 11:02:21.706782.706782 cuda_h.py:27] end decode_layer cost 13.181 ms
DEBUG 05-06 11:02:21.706718.706718 lmp.py:904] ---- decode step 0 layer 8 ----
DEBUG 05-06 11:02:21.707681.707681 cuda_h.py:27] end *sagl cost 1.619 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 51, 55, 75, 103], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 51: 2, 55: 1, 75: 1, 103: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 42, 46, 50, 54, 110], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 2, 6: 2, 42: 1, 46: 1, 50: 1, 54: 2, 110: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 12, 24, 64], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 12: 1, 24: 1, 64: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 69, 93], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {1: 2, 5: 2, 69: 1, 93: 1}}
INFO 05-06 11:02:21.709652.709652 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.331ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 11:02:21.709952.709952 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 11:02:21.709801.709801 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.710360.710360 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001435995101928711 seconds
INFO 05-06 11:02:21.712943.712943 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013158321380615234 seconds
INFO 05-06 11:02:21.712011.712011 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0028684139251708984 seconds
INFO 05-06 11:02:21.713929.713929 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012755393981933594 seconds
INFO 05-06 11:02:21.715504.715504 mlpmodule.py:2799] [fused_experts] gmm total=2.125ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.716697.716697 mlpmodule.py:2799] [fused_experts] gmm total=2.233ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.716226.716226 mlpmodule.py:2799] [fused_experts] gmm total=2.335ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.716607.716607 mlpmodule.py:2799] [fused_experts] gmm total=2.368ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.717777.717777 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004099369049072266 seconds
INFO 05-06 11:02:21.717840.717840 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 11:02:21.718274.718274 cuda_h.py:27] end *layer_moe_fused cost 9.317 ms
DEBUG 05-06 11:02:21.718599.718599 cuda_h.py:27] end decode_layer cost 12.266 ms
DEBUG 05-06 11:02:21.718204.718204 lmp.py:904] ---- decode step 0 layer 9 ----
DEBUG 05-06 11:02:21.720723.720723 cuda_h.py:27] end *sagl cost 1.643 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 95], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {3: 2, 7: 3, 15: 1, 19: 1, 95: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 30, 74], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {2: 2, 6: 2, 30: 1, 74: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 36, 48, 76], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 36: 1, 48: 1, 76: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 37, 69, 81, 89, 101], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 2, 5: 2, 37: 1, 69: 1, 81: 1, 89: 1, 101: 2}}
INFO 05-06 11:02:21.721455.721455 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.308ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 11:02:21.721570.721570 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.811981201171875e-05 seconds
INFO 05-06 11:02:21.721134.721134 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.722325.722325 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013051033020019531 seconds
INFO 05-06 11:02:21.724065.724065 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013020038604736328 seconds
INFO 05-06 11:02:21.724987.724987 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002711057662963867 seconds
INFO 05-06 11:02:21.725103.725103 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012464523315429688 seconds
INFO 05-06 11:02:21.727069.727069 mlpmodule.py:2799] [fused_experts] gmm total=1.960ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.727485.727485 mlpmodule.py:2799] [fused_experts] gmm total=2.036ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.728336.728336 mlpmodule.py:2799] [fused_experts] gmm total=2.060ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.728278.728278 mlpmodule.py:2799] [fused_experts] gmm total=2.227ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.729733.729733 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0039033889770507812 seconds
INFO 05-06 11:02:21.729558.729558 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 11:02:21.730359.730359 cuda_h.py:27] end *layer_moe_fused cost 9.218 ms
DEBUG 05-06 11:02:21.730928.730928 cuda_h.py:27] end decode_layer cost 12.151 ms
DEBUG 05-06 11:02:21.730626.730626 lmp.py:904] ---- decode step 0 layer 10 ----
DEBUG 05-06 11:02:21.732329.732329 cuda_h.py:27] end *sagl cost 1.568 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 79], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {3: 2, 7: 2, 79: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 46, 54, 126], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 2, 6: 2, 18: 1, 46: 1, 54: 1, 126: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 44, 60], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 8: 1, 44: 1, 60: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 21, 57, 81, 97, 105], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {1: 2, 5: 2, 21: 1, 57: 1, 81: 2, 97: 2, 105: 2}}
INFO 05-06 11:02:21.733060.733060 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.319ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 11:02:21.733838.733838 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 11:02:21.733117.733117 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.735875.735875 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012679100036621094 seconds
INFO 05-06 11:02:21.736428.736428 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001232147216796875 seconds
INFO 05-06 11:02:21.736781.736781 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0026068687438964844 seconds
INFO 05-06 11:02:21.737739.737739 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012693405151367188 seconds
INFO 05-06 11:02:21.740144.740144 mlpmodule.py:2799] [fused_experts] gmm total=2.091ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.740613.740613 mlpmodule.py:2799] [fused_experts] gmm total=2.334ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.740604.740604 mlpmodule.py:2799] [fused_experts] gmm total=2.214ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.740972.740972 mlpmodule.py:2799] [fused_experts] gmm total=2.394ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.741584.741584 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004198551177978516 seconds
INFO 05-06 11:02:21.742740.742740 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 11:02:21.742087.742087 cuda_h.py:27] end *layer_moe_fused cost 9.144 ms
DEBUG 05-06 11:02:21.742564.742564 cuda_h.py:27] end decode_layer cost 11.992 ms
DEBUG 05-06 11:02:21.742069.742069 lmp.py:904] ---- decode step 0 layer 11 ----
DEBUG 05-06 11:02:21.744599.744599 cuda_h.py:27] end *sagl cost 1.546 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 67, 79, 83, 99], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 2, 23: 1, 67: 1, 79: 1, 83: 2, 99: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 38, 46, 50, 102, 114], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 3, 6: 2, 38: 1, 46: 1, 50: 1, 102: 1, 114: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 124], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {0: 2, 4: 2, 124: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 49, 81], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 2, 5: 2, 9: 1, 49: 1, 81: 1}}
INFO 05-06 11:02:21.745444.745444 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.311ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 11:02:21.745036.745036 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 11:02:21.745553.745553 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.747796.747796 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014829635620117188 seconds
INFO 05-06 11:02:21.748019.748019 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010495185852050781 seconds
INFO 05-06 11:02:21.748994.748994 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0026526451110839844 seconds
INFO 05-06 11:02:21.749283.749283 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012679100036621094 seconds
INFO 05-06 11:02:21.752738.752738 mlpmodule.py:2799] [fused_experts] gmm total=2.129ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.752730.752730 mlpmodule.py:2799] [fused_experts] gmm total=2.369ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.752296.752296 mlpmodule.py:2799] [fused_experts] gmm total=2.303ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.752966.752966 mlpmodule.py:2799] [fused_experts] gmm total=2.759ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.754984.754984 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004273414611816406 seconds
INFO 05-06 11:02:21.754240.754240 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 11:02:21.754732.754732 cuda_h.py:27] end *layer_moe_fused cost 9.611 ms
DEBUG 05-06 11:02:21.755539.755539 cuda_h.py:27] end decode_layer cost 12.455 ms
DEBUG 05-06 11:02:21.755190.755190 lmp.py:904] ---- decode step 0 layer 12 ----
DEBUG 05-06 11:02:21.756103.756103 cuda_h.py:27] end *sagl cost 1.512 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 19: 1, 39: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 46, 50, 74, 78, 86, 106, 114], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {2: 2, 6: 2, 46: 1, 50: 1, 74: 1, 78: 2, 86: 1, 106: 1, 114: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 36], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {0: 2, 4: 2, 36: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 21, 45, 73, 97, 117], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 21: 1, 45: 1, 73: 1, 97: 1, 117: 1}}
INFO 05-06 11:02:21.758807.758807 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.317ms | allocate_experts_across_cpu_gpu: 0.099ms
INFO 05-06 11:02:21.758446.758446 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 11:02:21.758010.758010 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.759641.759641 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014181137084960938 seconds
INFO 05-06 11:02:21.760811.760811 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001257181167602539 seconds
INFO 05-06 11:02:21.761310.761310 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0027937889099121094 seconds
INFO 05-06 11:02:21.762030.762030 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013055801391601562 seconds
INFO 05-06 11:02:21.764987.764987 mlpmodule.py:2799] [fused_experts] gmm total=1.839ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.764521.764521 mlpmodule.py:2799] [fused_experts] gmm total=2.084ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.764328.764328 mlpmodule.py:2799] [fused_experts] gmm total=2.169ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.765971.765971 mlpmodule.py:2799] [fused_experts] gmm total=2.496ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.766505.766505 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0038237571716308594 seconds
INFO 05-06 11:02:21.766092.766092 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 11:02:21.766445.766445 cuda_h.py:27] end *layer_moe_fused cost 9.175 ms
DEBUG 05-06 11:02:21.767783.767783 cuda_h.py:27] end decode_layer cost 11.948 ms
DEBUG 05-06 11:02:21.767718.767718 lmp.py:904] ---- decode step 0 layer 13 ----
DEBUG 05-06 11:02:21.768209.768209 cuda_h.py:27] end *sagl cost 1.551 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 47, 107], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 47: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 22, 78, 110, 114], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {2: 3, 6: 2, 14: 1, 22: 1, 78: 1, 110: 1, 114: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 32, 80, 100, 104], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 32: 1, 80: 1, 100: 2, 104: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 125], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {1: 3, 5: 2, 125: 1}}
INFO 05-06 11:02:21.770868.770868 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.328ms | allocate_experts_across_cpu_gpu: 0.092ms
INFO 05-06 11:02:21.770308.770308 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.71661376953125e-05 seconds
INFO 05-06 11:02:21.770686.770686 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.771154.771154 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014731884002685547 seconds
INFO 05-06 11:02:21.773482.773482 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014429092407226562 seconds
INFO 05-06 11:02:21.773087.773087 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0030443668365478516 seconds
INFO 05-06 11:02:21.774475.774475 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001268148422241211 seconds
INFO 05-06 11:02:21.777880.777880 mlpmodule.py:2799] [fused_experts] gmm total=2.171ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.777396.777396 mlpmodule.py:2799] [fused_experts] gmm total=2.065ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.777108.777108 mlpmodule.py:2799] [fused_experts] gmm total=2.342ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.777108.777108 mlpmodule.py:2799] [fused_experts] gmm total=2.454ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.778214.778214 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0041964054107666016 seconds
INFO 05-06 11:02:21.779277.779277 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 11:02:21.779123.779123 cuda_h.py:27] end *layer_moe_fused cost 9.808 ms
DEBUG 05-06 11:02:21.779182.779182 cuda_h.py:27] end decode_layer cost 12.661 ms
DEBUG 05-06 11:02:21.780403.780403 lmp.py:904] ---- decode step 0 layer 14 ----
DEBUG 05-06 11:02:21.781700.781700 cuda_h.py:27] end *sagl cost 1.514 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 39, 47, 99, 115], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 2, 11: 1, 39: 1, 47: 1, 99: 1, 115: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 26], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 7, 'token_per_expert': {2: 4, 6: 2, 26: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 56], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {0: 2, 4: 2, 56: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 25, 81, 97, 109, 121], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {1: 2, 5: 2, 13: 1, 25: 1, 81: 1, 97: 1, 109: 1, 121: 1}}
INFO 05-06 11:02:21.782272.782272 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.325ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 11:02:21.782665.782665 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7642974853515625e-05 seconds
INFO 05-06 11:02:21.782752.782752 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.784916.784916 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014955997467041016 seconds
INFO 05-06 11:02:21.785021.785021 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011034011840820312 seconds
INFO 05-06 11:02:21.785281.785281 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0027184486389160156 seconds
INFO 05-06 11:02:21.787570.787570 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012676715850830078 seconds
INFO 05-06 11:02:21.789786.789786 mlpmodule.py:2799] [fused_experts] gmm total=2.066ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.789308.789308 mlpmodule.py:2799] [fused_experts] gmm total=2.138ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.789351.789351 mlpmodule.py:2799] [fused_experts] gmm total=2.143ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.789760.789760 mlpmodule.py:2799] [fused_experts] gmm total=2.193ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.791393.791393 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004327297210693359 seconds
INFO 05-06 11:02:21.791171.791171 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 11:02:21.791703.791703 cuda_h.py:27] end *layer_moe_fused cost 9.552 ms
DEBUG 05-06 11:02:21.792418.792418 cuda_h.py:27] end decode_layer cost 12.331 ms
DEBUG 05-06 11:02:21.792354.792354 lmp.py:904] ---- decode step 0 layer 15 ----
DEBUG 05-06 11:02:21.794340.794340 cuda_h.py:27] end *sagl cost 1.531 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 75, 83, 119], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 75: 1, 83: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 30, 34], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {2: 2, 6: 2, 30: 1, 34: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 24, 68, 72, 108, 112], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 24: 1, 68: 1, 72: 1, 108: 1, 112: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 29, 33, 69, 81, 93, 101], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {1: 2, 5: 2, 29: 1, 33: 1, 69: 1, 81: 1, 93: 1, 101: 1}}
INFO 05-06 11:02:21.795866.795866 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.314ms | allocate_experts_across_cpu_gpu: 0.099ms
INFO 05-06 11:02:21.795597.795597 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.09808349609375e-05 seconds
INFO 05-06 11:02:21.795684.795684 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.796602.796602 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012807846069335938 seconds
INFO 05-06 11:02:21.798882.798882 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014145374298095703 seconds
INFO 05-06 11:02:21.798487.798487 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.003165006637573242 seconds
INFO 05-06 11:02:21.799751.799751 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013170242309570312 seconds
INFO 05-06 11:02:21.802019.802019 mlpmodule.py:2799] [fused_experts] gmm total=2.032ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.802715.802715 mlpmodule.py:2799] [fused_experts] gmm total=2.163ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.802696.802696 mlpmodule.py:2799] [fused_experts] gmm total=2.296ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.803166.803166 mlpmodule.py:2799] [fused_experts] gmm total=2.589ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.804297.804297 lmp.py:1500] [layer_moe_fused] experts compute time: 0.00419306755065918 seconds
INFO 05-06 11:02:21.804030.804030 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.0067901611328125e-05 seconds
DEBUG 05-06 11:02:21.804259.804259 cuda_h.py:27] end *layer_moe_fused cost 9.764 ms
DEBUG 05-06 11:02:21.805731.805731 cuda_h.py:27] end decode_layer cost 12.632 ms
DEBUG 05-06 11:02:21.805143.805143 lmp.py:904] ---- decode step 0 layer 16 ----
DEBUG 05-06 11:02:21.806560.806560 cuda_h.py:27] end *sagl cost 1.532 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 87, 107], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 19: 1, 87: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 54, 62, 66, 78, 102], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 2, 6: 2, 54: 2, 62: 1, 66: 1, 78: 1, 102: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 20, 32, 44], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {0: 2, 4: 3, 20: 1, 32: 1, 44: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 85, 105], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 2, 5: 3, 85: 1, 105: 1}}
INFO 05-06 11:02:21.807245.807245 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.323ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 11:02:21.808361.808361 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8358230590820312e-05 seconds
INFO 05-06 11:02:21.808402.808402 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.809505.809505 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012767314910888672 seconds
INFO 05-06 11:02:21.810442.810442 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014452934265136719 seconds
INFO 05-06 11:02:21.810047.810047 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0028390884399414062 seconds
INFO 05-06 11:02:21.812250.812250 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012710094451904297 seconds
INFO 05-06 11:02:21.814871.814871 mlpmodule.py:2799] [fused_experts] gmm total=1.948ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.814712.814712 mlpmodule.py:2799] [fused_experts] gmm total=2.237ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.814903.814903 mlpmodule.py:2799] [fused_experts] gmm total=2.332ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.815126.815126 mlpmodule.py:2799] [fused_experts] gmm total=2.388ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.816927.816927 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0039408206939697266 seconds
INFO 05-06 11:02:21.816229.816229 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 11:02:21.816499.816499 cuda_h.py:27] end *layer_moe_fused cost 9.389 ms
DEBUG 05-06 11:02:21.817307.817307 cuda_h.py:27] end decode_layer cost 12.194 ms
DEBUG 05-06 11:02:21.817719.817719 lmp.py:904] ---- decode step 0 layer 17 ----
DEBUG 05-06 11:02:21.818500.818500 cuda_h.py:27] end *sagl cost 1.520 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 47], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {3: 2, 7: 3, 23: 2, 47: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 22, 34, 70], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 2, 6: 2, 18: 1, 22: 1, 34: 1, 70: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 16, 68], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 16: 1, 68: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 33, 53, 73, 113], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 2, 5: 3, 13: 1, 33: 1, 53: 1, 73: 1, 113: 1}}
INFO 05-06 11:02:21.820750.820750 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.318ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 11:02:21.820905.820905 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7642974853515625e-05 seconds
INFO 05-06 11:02:21.820992.820992 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.821485.821485 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014216899871826172 seconds
INFO 05-06 11:02:21.823302.823302 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001413106918334961 seconds
INFO 05-06 11:02:21.823669.823669 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0029668807983398438 seconds
INFO 05-06 11:02:21.824971.824971 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012748241424560547 seconds
INFO 05-06 11:02:21.826735.826735 mlpmodule.py:2799] [fused_experts] gmm total=2.013ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.827225.827225 mlpmodule.py:2799] [fused_experts] gmm total=2.098ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.827132.827132 mlpmodule.py:2799] [fused_experts] gmm total=2.212ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.827229.827229 mlpmodule.py:2799] [fused_experts] gmm total=2.242ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.829211.829211 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0043179988861083984 seconds
INFO 05-06 11:02:21.829036.829036 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 11:02:21.829927.829927 cuda_h.py:27] end *layer_moe_fused cost 9.639 ms
DEBUG 05-06 11:02:21.829027.829027 cuda_h.py:27] end decode_layer cost 12.546 ms
DEBUG 05-06 11:02:21.829201.829201 lmp.py:904] ---- decode step 0 layer 18 ----
DEBUG 05-06 11:02:21.831658.831658 cuda_h.py:27] end *sagl cost 1.526 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 75, 83, 111], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {3: 2, 7: 2, 23: 1, 75: 1, 83: 1, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 26, 30, 42, 54, 58], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 2, 26: 1, 30: 1, 42: 1, 54: 1, 58: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 40, 104], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 40: 1, 104: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 37, 73, 77, 97, 105], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 37: 1, 73: 1, 77: 1, 97: 1, 105: 1}}
INFO 05-06 11:02:21.832708.832708 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.324ms | allocate_experts_across_cpu_gpu: 0.102ms
INFO 05-06 11:02:21.832677.832677 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-05 seconds
INFO 05-06 11:02:21.832241.832241 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.834250.834250 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014181137084960938 seconds
INFO 05-06 11:02:21.835654.835654 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013341903686523438 seconds
INFO 05-06 11:02:21.835768.835768 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0028569698333740234 seconds
INFO 05-06 11:02:21.837309.837309 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012793540954589844 seconds
INFO 05-06 11:02:21.839540.839540 mlpmodule.py:2799] [fused_experts] gmm total=1.934ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.839029.839029 mlpmodule.py:2799] [fused_experts] gmm total=2.291ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.839898.839898 mlpmodule.py:2799] [fused_experts] gmm total=2.265ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.840478.840478 mlpmodule.py:2799] [fused_experts] gmm total=2.523ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.841876.841876 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004204511642456055 seconds
INFO 05-06 11:02:21.841470.841470 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 11:02:21.841606.841606 cuda_h.py:27] end *layer_moe_fused cost 9.447 ms
DEBUG 05-06 11:02:21.842460.842460 cuda_h.py:27] end decode_layer cost 12.264 ms
DEBUG 05-06 11:02:21.842395.842395 lmp.py:904] ---- decode step 0 layer 19 ----
DEBUG 05-06 11:02:21.843451.843451 cuda_h.py:27] end *sagl cost 1.617 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 111], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 19: 1, 31: 1, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 78, 82, 86, 106, 122], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 2, 6: 2, 10: 1, 78: 1, 82: 1, 86: 1, 106: 2, 122: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 40, 44, 84], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 40: 1, 44: 1, 84: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 25, 61], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 2, 5: 3, 25: 1, 61: 1}}
INFO 05-06 11:02:21.845846.845846 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.322ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 11:02:21.845623.845623 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 11:02:21.845902.845902 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.846791.846791 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013992786407470703 seconds
INFO 05-06 11:02:21.848758.848758 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013279914855957031 seconds
INFO 05-06 11:02:21.848919.848919 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002831697463989258 seconds
INFO 05-06 11:02:21.849075.849075 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012767314910888672 seconds
INFO 05-06 11:02:21.851556.851556 mlpmodule.py:2799] [fused_experts] gmm total=1.915ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.851709.851709 mlpmodule.py:2799] [fused_experts] gmm total=1.972ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.852695.852695 mlpmodule.py:2799] [fused_experts] gmm total=2.059ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.852057.852057 mlpmodule.py:2799] [fused_experts] gmm total=2.349ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.853511.853511 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0038657188415527344 seconds
INFO 05-06 11:02:21.853337.853337 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 11:02:21.854690.854690 cuda_h.py:27] end *layer_moe_fused cost 9.225 ms
DEBUG 05-06 11:02:21.854114.854114 cuda_h.py:27] end decode_layer cost 12.146 ms
DEBUG 05-06 11:02:21.854857.854857 lmp.py:904] ---- decode step 0 layer 20 ----
DEBUG 05-06 11:02:21.856458.856458 cuda_h.py:27] end *sagl cost 1.493 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 95, 107], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {3: 3, 7: 2, 95: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 62, 94, 102], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 3, 6: 2, 62: 1, 94: 1, 102: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 36, 40, 52], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 36: 1, 40: 1, 52: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 21, 73, 85, 117], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {1: 2, 5: 2, 13: 1, 21: 2, 73: 1, 85: 1, 117: 1}}
INFO 05-06 11:02:21.857713.857713 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.315ms | allocate_experts_across_cpu_gpu: 0.089ms
INFO 05-06 11:02:21.857895.857895 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 11:02:21.857220.857220 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.858184.858184 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014541149139404297 seconds
INFO 05-06 11:02:21.860924.860924 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010666847229003906 seconds
INFO 05-06 11:02:21.860293.860293 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0027229785919189453 seconds
INFO 05-06 11:02:21.862964.862964 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016472339630126953 seconds
INFO 05-06 11:02:21.864677.864677 mlpmodule.py:2799] [fused_experts] gmm total=2.066ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.864940.864940 mlpmodule.py:2799] [fused_experts] gmm total=2.307ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.865571.865571 mlpmodule.py:2799] [fused_experts] gmm total=2.460ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.866673.866673 mlpmodule.py:2799] [fused_experts] gmm total=3.426ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.866378.866378 lmp.py:1500] [layer_moe_fused] experts compute time: 0.00458836555480957 seconds
INFO 05-06 11:02:21.867779.867779 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.2928924560546875e-05 seconds
DEBUG 05-06 11:02:21.867111.867111 cuda_h.py:27] end *layer_moe_fused cost 10.706 ms
DEBUG 05-06 11:02:21.868343.868343 cuda_h.py:27] end decode_layer cost 13.519 ms
DEBUG 05-06 11:02:21.868233.868233 lmp.py:904] ---- decode step 0 layer 21 ----
DEBUG 05-06 11:02:21.869554.869554 cuda_h.py:27] end *sagl cost 1.637 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 83, 87, 103], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {3: 2, 7: 2, 11: 1, 83: 1, 87: 1, 103: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 14, 26, 34, 86, 94, 110], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 3, 6: 2, 14: 1, 26: 1, 34: 1, 86: 1, 94: 1, 110: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 80, 124], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 80: 1, 124: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 25], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 7, 'token_per_expert': {1: 2, 5: 4, 25: 1}}
INFO 05-06 11:02:21.871519.871519 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.332ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 11:02:21.871912.871912 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.811981201171875e-05 seconds
INFO 05-06 11:02:21.871284.871284 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.872353.872353 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014262199401855469 seconds
INFO 05-06 11:02:21.873981.873981 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011034011840820312 seconds
INFO 05-06 11:02:21.873526.873526 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0026476383209228516 seconds
INFO 05-06 11:02:21.875883.875883 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013134479522705078 seconds
INFO 05-06 11:02:21.877588.877588 mlpmodule.py:2799] [fused_experts] gmm total=2.257ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.877104.877104 mlpmodule.py:2799] [fused_experts] gmm total=2.240ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.878054.878054 mlpmodule.py:2799] [fused_experts] gmm total=2.247ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.878724.878724 mlpmodule.py:2799] [fused_experts] gmm total=2.654ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.879631.879631 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004267454147338867 seconds
INFO 05-06 11:02:21.879171.879171 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 11:02:21.880482.880482 cuda_h.py:27] end *layer_moe_fused cost 9.597 ms
DEBUG 05-06 11:02:21.880634.880634 cuda_h.py:27] end decode_layer cost 12.581 ms
DEBUG 05-06 11:02:21.880901.880901 lmp.py:904] ---- decode step 0 layer 22 ----
DEBUG 05-06 11:02:21.882596.882596 cuda_h.py:27] end *sagl cost 1.527 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 43, 119, 123, 127], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 43: 1, 119: 2, 123: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 26, 38, 94], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 26: 1, 38: 1, 94: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 16, 24, 76, 108, 120], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {0: 2, 4: 2, 8: 1, 16: 1, 24: 1, 76: 1, 108: 1, 120: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 61, 101], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {1: 2, 5: 2, 61: 1, 101: 1}}
INFO 05-06 11:02:21.883356.883356 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.311ms | allocate_experts_across_cpu_gpu: 0.133ms
INFO 05-06 11:02:21.883133.883133 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.002716064453125e-05 seconds
INFO 05-06 11:02:21.883982.883982 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.885048.885048 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013546943664550781 seconds
INFO 05-06 11:02:21.886926.886926 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001050710678100586 seconds
INFO 05-06 11:02:21.886563.886563 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0025110244750976562 seconds
INFO 05-06 11:02:21.887150.887150 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012695789337158203 seconds
INFO 05-06 11:02:21.889192.889192 mlpmodule.py:2799] [fused_experts] gmm total=2.033ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.890472.890472 mlpmodule.py:2799] [fused_experts] gmm total=2.173ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.890427.890427 mlpmodule.py:2799] [fused_experts] gmm total=2.322ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.890470.890470 mlpmodule.py:2799] [fused_experts] gmm total=2.348ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.891630.891630 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0039691925048828125 seconds
INFO 05-06 11:02:21.891024.891024 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 11:02:21.892316.892316 cuda_h.py:27] end *layer_moe_fused cost 9.160 ms
DEBUG 05-06 11:02:21.892462.892462 cuda_h.py:27] end decode_layer cost 11.995 ms
DEBUG 05-06 11:02:21.892623.892623 lmp.py:904] ---- decode step 0 layer 23 ----
DEBUG 05-06 11:02:21.894946.894946 cuda_h.py:27] end *sagl cost 1.499 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 47, 67], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 47: 1, 67: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 22, 86, 118], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 2, 6: 2, 22: 1, 86: 2, 118: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 12, 32, 84, 108], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 8: 1, 12: 1, 32: 1, 84: 1, 108: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 17, 81, 97, 109], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 17: 1, 81: 1, 97: 2, 109: 1}}
INFO 05-06 11:02:21.895187.895187 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.306ms | allocate_experts_across_cpu_gpu: 0.114ms
INFO 05-06 11:02:21.895389.895389 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8358230590820312e-05 seconds
INFO 05-06 11:02:21.895761.895761 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.897605.897605 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012624263763427734 seconds
INFO 05-06 11:02:21.898389.898389 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010173320770263672 seconds
INFO 05-06 11:02:21.898802.898802 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0023946762084960938 seconds
INFO 05-06 11:02:21.899872.899872 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012848377227783203 seconds
INFO 05-06 11:02:21.901880.901880 mlpmodule.py:2799] [fused_experts] gmm total=1.977ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.901239.901239 mlpmodule.py:2799] [fused_experts] gmm total=2.114ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.902128.902128 mlpmodule.py:2799] [fused_experts] gmm total=2.248ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.902668.902668 mlpmodule.py:2799] [fused_experts] gmm total=2.271ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.903855.903855 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003954172134399414 seconds
INFO 05-06 11:02:21.903184.903184 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 5.626678466796875e-05 seconds
DEBUG 05-06 11:02:21.904670.904670 cuda_h.py:27] end *layer_moe_fused cost 8.888 ms
DEBUG 05-06 11:02:21.904053.904053 cuda_h.py:27] end decode_layer cost 11.652 ms
DEBUG 05-06 11:02:21.904320.904320 lmp.py:904] ---- decode step 0 layer 24 ----
DEBUG 05-06 11:02:21.906087.906087 cuda_h.py:27] end *sagl cost 1.511 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 63, 79, 123], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 63: 1, 79: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 30, 66, 90, 110, 118], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 2, 30: 1, 66: 1, 90: 1, 110: 1, 118: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 12, 40, 44], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 12: 1, 40: 1, 44: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 33, 65, 109, 113], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 33: 2, 65: 1, 109: 1, 113: 1}}
INFO 05-06 11:02:21.907865.907865 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.317ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 11:02:21.907934.907934 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.002716064453125e-05 seconds
INFO 05-06 11:02:21.907498.907498 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.908780.908780 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001268625259399414 seconds
INFO 05-06 11:02:21.910769.910769 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014116764068603516 seconds
INFO 05-06 11:02:21.910659.910659 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002797842025756836 seconds
INFO 05-06 11:02:21.911326.911326 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001298666000366211 seconds
INFO 05-06 11:02:21.913172.913172 mlpmodule.py:2799] [fused_experts] gmm total=1.983ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.914687.914687 mlpmodule.py:2799] [fused_experts] gmm total=2.061ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.914741.914741 mlpmodule.py:2799] [fused_experts] gmm total=2.442ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.914122.914122 mlpmodule.py:2799] [fused_experts] gmm total=2.288ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.915431.915431 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004239559173583984 seconds
INFO 05-06 11:02:21.916256.916256 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 11:02:21.916446.916446 cuda_h.py:27] end *layer_moe_fused cost 9.446 ms
DEBUG 05-06 11:02:21.916209.916209 cuda_h.py:27] end decode_layer cost 12.307 ms
DEBUG 05-06 11:02:21.916383.916383 lmp.py:904] ---- decode step 0 layer 25 ----
DEBUG 05-06 11:02:21.918802.918802 cuda_h.py:27] end *sagl cost 1.568 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 47, 67, 95], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 3, 7: 2, 19: 1, 47: 1, 67: 1, 95: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 58], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {2: 2, 6: 2, 58: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 16, 44, 68, 104], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 3, 4: 2, 16: 1, 44: 1, 68: 1, 104: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 45, 93, 117, 121], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 45: 1, 93: 1, 117: 1, 121: 1}}
INFO 05-06 11:02:21.919860.919860 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.360ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 11:02:21.919829.919829 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 11:02:21.919916.919916 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.921279.921279 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012927055358886719 seconds
INFO 05-06 11:02:21.922371.922371 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013165473937988281 seconds
INFO 05-06 11:02:21.922578.922578 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002711057662963867 seconds
INFO 05-06 11:02:21.924873.924873 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012743473052978516 seconds
INFO 05-06 11:02:21.926014.926014 mlpmodule.py:2799] [fused_experts] gmm total=1.772ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.926512.926512 mlpmodule.py:2799] [fused_experts] gmm total=2.068ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.926392.926392 mlpmodule.py:2799] [fused_experts] gmm total=2.030ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.926614.926614 mlpmodule.py:2799] [fused_experts] gmm total=2.206ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.927873.927873 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0038938522338867188 seconds
INFO 05-06 11:02:21.928414.928414 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 11:02:21.928068.928068 cuda_h.py:27] end *layer_moe_fused cost 9.246 ms
DEBUG 05-06 11:02:21.929743.929743 cuda_h.py:27] end decode_layer cost 12.128 ms
DEBUG 05-06 11:02:21.929156.929156 lmp.py:904] ---- decode step 0 layer 26 ----
DEBUG 05-06 11:02:21.930453.930453 cuda_h.py:27] end *sagl cost 1.515 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 43, 79, 119], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 19: 1, 23: 1, 43: 1, 79: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 10, 38, 90], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 10: 1, 38: 1, 90: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 8, 20, 52, 84], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 3, 8: 1, 20: 1, 52: 1, 84: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 49, 65, 85], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 2, 5: 2, 49: 1, 65: 1, 85: 1}}
INFO 05-06 11:02:21.931569.931569 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.329ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 11:02:21.932346.932346 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8596649169921875e-05 seconds
INFO 05-06 11:02:21.932195.932195 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.933286.933286 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001302480697631836 seconds
INFO 05-06 11:02:21.934050.934050 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014214515686035156 seconds
INFO 05-06 11:02:21.934417.934417 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.002841949462890625 seconds
INFO 05-06 11:02:21.936170.936170 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001291513442993164 seconds
INFO 05-06 11:02:21.938557.938557 mlpmodule.py:2799] [fused_experts] gmm total=2.226ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.938642.938642 mlpmodule.py:2799] [fused_experts] gmm total=2.315ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.938731.938731 mlpmodule.py:2799] [fused_experts] gmm total=2.344ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.939106.939106 mlpmodule.py:2799] [fused_experts] gmm total=2.374ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.940025.940025 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004315376281738281 seconds
INFO 05-06 11:02:21.940089.940089 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.9591064453125e-05 seconds
DEBUG 05-06 11:02:21.941550.941550 cuda_h.py:27] end *layer_moe_fused cost 9.962 ms
DEBUG 05-06 11:02:21.941557.941557 cuda_h.py:27] end decode_layer cost 12.755 ms
DEBUG 05-06 11:02:21.941823.941823 lmp.py:904] ---- decode step 0 layer 27 ----
DEBUG 05-06 11:02:21.943221.943221 cuda_h.py:27] end *sagl cost 1.554 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 87, 103, 115], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 87: 1, 103: 1, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 58, 62, 114], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 58: 1, 62: 1, 114: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 32, 48, 108], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 32: 1, 48: 1, 108: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 29, 41, 61, 85, 97, 121], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {1: 3, 5: 2, 29: 1, 41: 1, 61: 1, 85: 1, 97: 1, 121: 1}}
INFO 05-06 11:02:21.944112.944112 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.311ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 11:02:21.944943.944943 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0265579223632812e-05 seconds
INFO 05-06 11:02:21.944745.944745 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.946006.946006 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00142669677734375 seconds
INFO 05-06 11:02:21.947470.947470 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001331329345703125 seconds
INFO 05-06 11:02:21.947359.947359 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0028853416442871094 seconds
INFO 05-06 11:02:21.949416.949416 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001270294189453125 seconds
INFO 05-06 11:02:21.951748.951748 mlpmodule.py:2799] [fused_experts] gmm total=2.056ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.951901.951901 mlpmodule.py:2799] [fused_experts] gmm total=2.171ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.951092.951092 mlpmodule.py:2799] [fused_experts] gmm total=2.271ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.952454.952454 mlpmodule.py:2799] [fused_experts] gmm total=2.671ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.953304.953304 lmp.py:1500] [layer_moe_fused] experts compute time: 0.004096031188964844 seconds
INFO 05-06 11:02:21.953606.953606 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 11:02:21.953715.953715 cuda_h.py:27] end *layer_moe_fused cost 9.528 ms
DEBUG 05-06 11:02:21.954199.954199 cuda_h.py:27] end decode_layer cost 12.385 ms
DEBUG 05-06 11:02:21.954134.954134 lmp.py:904] ---- decode step 0 layer 28 ----
DEBUG 05-06 11:02:21.955736.955736 cuda_h.py:27] end *sagl cost 1.529 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39, 67, 115, 119], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 2, 19: 1, 39: 1, 67: 1, 115: 2, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {2: 2, 6: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 104, 108], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 104: 1, 108: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 49, 53, 57, 65, 89, 105, 113], 'expert_count': 10, 'ideal_gpu_count': 5, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 12, 'token_per_expert': {1: 2, 5: 2, 13: 1, 49: 1, 53: 1, 57: 1, 65: 1, 89: 1, 105: 1, 113: 1}}
INFO 05-06 11:02:21.957468.957468 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.316ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 11:02:21.957769.957769 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 11:02:21.957624.957624 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.958718.958718 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013740062713623047 seconds
INFO 05-06 11:02:21.959605.959605 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009541511535644531 seconds
INFO 05-06 11:02:21.959004.959004 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0024323463439941406 seconds
INFO 05-06 11:02:21.961691.961691 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012803077697753906 seconds
INFO 05-06 11:02:21.963597.963597 mlpmodule.py:2799] [fused_experts] gmm total=1.952ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.963378.963378 mlpmodule.py:2799] [fused_experts] gmm total=2.048ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.963511.963511 mlpmodule.py:2799] [fused_experts] gmm total=2.174ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.963257.963257 mlpmodule.py:2799] [fused_experts] gmm total=2.232ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.965311.965311 lmp.py:1500] [layer_moe_fused] experts compute time: 0.0038254261016845703 seconds
INFO 05-06 11:02:21.965613.965613 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 11:02:21.965258.965258 cuda_h.py:27] end *layer_moe_fused cost 8.796 ms
DEBUG 05-06 11:02:21.965258.965258 cuda_h.py:27] end decode_layer cost 11.603 ms
DEBUG 05-06 11:02:21.966955.966955 lmp.py:904] ---- decode step 0 layer 29 ----
DEBUG 05-06 11:02:21.967597.967597 cuda_h.py:27] end *sagl cost 1.512 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {3: 2, 7: 2, 23: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [2, 6, 18, 26, 30, 66, 78], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 3, 6: 2, 18: 1, 26: 1, 30: 1, 66: 1, 78: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [0, 4, 52, 56, 60, 64], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 3, 52: 1, 56: 1, 60: 1, 64: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 49, 73, 81, 97], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 49: 1, 73: 1, 81: 1, 97: 1}}
INFO 05-06 11:02:21.968581.968581 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.092ms
INFO 05-06 11:02:21.968881.968881 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9550323486328125e-05 seconds
INFO 05-06 11:02:21.969445.969445 lmp.py:1390] [layer_moe_fused] cpu_experts time: {time.time() - time_start} seconds
INFO 05-06 11:02:21.970116.970116 lmp.py:1401] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014109611511230469 seconds
INFO 05-06 11:02:21.971227.971227 lmp.py:1413] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010724067687988281 seconds
INFO 05-06 11:02:21.971262.971262 lmp.py:1415] [layer_moe_fused] load_gpu_experts time: 0.0026116371154785156 seconds
INFO 05-06 11:02:21.972849.972849 lmp.py:1434] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012764930725097656 seconds
INFO 05-06 11:02:21.975953.975953 mlpmodule.py:2799] [fused_experts] gmm total=1.920ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.975576.975576 mlpmodule.py:2799] [fused_experts] gmm total=2.041ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.975179.975179 mlpmodule.py:2799] [fused_experts] gmm total=2.164ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.975951.975951 mlpmodule.py:2799] [fused_experts] gmm total=2.203ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 11:02:21.976527.976527 lmp.py:1500] [layer_moe_fused] experts compute time: 0.003762960433959961 seconds
INFO 05-06 11:02:21.976730.976730 lmp.py:1510] [layer_moe_fused] scatter_reduce_ time: 4.792213439941406e-05 seconds
DEBUG 05-06 11:02:21.977656.977656 cuda_h.py:27] end *layer_moe_fused cost 8.994 ms
DEBUG 05-06 11:02:21.977980.977980 cuda_h.py:27] end decode_layer cost 11.811 ms
DEBUG 05-06 11:02:21.977485.977485 cuda_h.py:27] end decode_step cost 372.355 ms
INFO 05-06 11:02:21.977679.977679 lmp.py:931] decode step 0 time: 0.37239527702331543 seconds
WARNING 05-06 11:02:21.978057.978057 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 11:02:21.978055.978055 helper.py:35]   NaN count (hidden): 5632
WARNING 05-06 11:02:21.978277.978277 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 11:02:21.978227.978227 helper.py:39]   NaN count (normed): 5632
WARNING 05-06 11:02:21.983734.983734 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 11:02:21.983751.983751 helper.py:50]   NaN count: 524288
WARNING 05-06 11:02:21.984659.984659 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 11:02:21.984825.984825 helper.py:80] WARNING: Logits have extreme values: min=-896.00, max=1032.00
WARNING 05-06 11:02:21.984478.984478 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 11:02:21.986061.986061 cuda_h.py:27] end init_inputs_tokens cost 7.956 ms
DEBUG 05-06 11:02:21.986188.986188 lmp.py:899] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 11:02:21.986766.986766 lmp.py:904] ---- decode step 1 layer 0 ----
DEBUG 05-06 11:02:21.987845.987845 cuda_h.py:27] end *sagl cost 1.528 ms
DEBUG 05-06 11:02:21.991002.991002 cuda_h.py:27] end *layer_moe_fused cost 3.439 ms
DEBUG 05-06 11:02:21.992720.992720 cuda_h.py:27] end decode_layer cost 6.349 ms
DEBUG 05-06 11:02:21.992292.992292 lmp.py:904] ---- decode step 1 layer 1 ----
DEBUG 05-06 11:02:21.994524.994524 cuda_h.py:27] end *sagl cost 1.905 ms
DEBUG 05-06 11:02:21.998408.998408 cuda_h.py:27] end *layer_moe_fused cost 3.183 ms
DEBUG 05-06 11:02:21.999252.999252 cuda_h.py:27] end decode_layer cost 6.706 ms
DEBUG 05-06 11:02:21.999301.999301 lmp.py:904] ---- decode step 1 layer 2 ----
DEBUG 05-06 11:02:22.001943.001943 cuda_h.py:27] end *sagl cost 1.928 ms
DEBUG 05-06 11:02:22.005969.005969 cuda_h.py:27] end *layer_moe_fused cost 2.868 ms
DEBUG 05-06 11:02:22.005051.005051 cuda_h.py:27] end decode_layer cost 6.376 ms
DEBUG 05-06 11:02:22.005192.005192 lmp.py:904] ---- decode step 1 layer 3 ----
DEBUG 05-06 11:02:22.007021.007021 cuda_h.py:27] end *sagl cost 1.959 ms
DEBUG 05-06 11:02:22.010023.010023 cuda_h.py:27] end *layer_moe_fused cost 2.117 ms
DEBUG 05-06 11:02:22.011820.011820 cuda_h.py:27] end decode_layer cost 5.690 ms
DEBUG 05-06 11:02:22.011723.011723 lmp.py:904] ---- decode step 1 layer 4 ----
DEBUG 05-06 11:02:22.013595.013595 cuda_h.py:27] end *sagl cost 1.851 ms
DEBUG 05-06 11:02:22.016336.016336 cuda_h.py:27] end *layer_moe_fused cost 2.107 ms
DEBUG 05-06 11:02:22.017749.017749 cuda_h.py:27] end decode_layer cost 5.526 ms
DEBUG 05-06 11:02:22.017368.017368 lmp.py:904] ---- decode step 1 layer 5 ----
DEBUG 05-06 11:02:22.018338.018338 cuda_h.py:27] end *sagl cost 1.818 ms
DEBUG 05-06 11:02:22.022636.022636 cuda_h.py:27] end *layer_moe_fused cost 2.085 ms
DEBUG 05-06 11:02:22.022579.022579 cuda_h.py:27] end decode_layer cost 5.523 ms
DEBUG 05-06 11:02:22.022436.022436 lmp.py:904] ---- decode step 1 layer 6 ----
DEBUG 05-06 11:02:22.024376.024376 cuda_h.py:27] end *sagl cost 1.901 ms
DEBUG 05-06 11:02:22.027899.027899 cuda_h.py:27] end *layer_moe_fused cost 2.106 ms
DEBUG 05-06 11:02:22.028319.028319 cuda_h.py:27] end decode_layer cost 5.597 ms
DEBUG 05-06 11:02:22.028368.028368 lmp.py:904] ---- decode step 1 layer 7 ----
DEBUG 05-06 11:02:22.030989.030989 cuda_h.py:27] end *sagl cost 1.868 ms
DEBUG 05-06 11:02:22.033520.033520 cuda_h.py:27] end *layer_moe_fused cost 2.146 ms
DEBUG 05-06 11:02:22.034363.034363 cuda_h.py:27] end decode_layer cost 5.610 ms
DEBUG 05-06 11:02:22.034320.034320 lmp.py:904] ---- decode step 1 layer 8 ----
DEBUG 05-06 11:02:22.036371.036371 cuda_h.py:27] end *sagl cost 1.879 ms
DEBUG 05-06 11:02:22.039608.039608 cuda_h.py:27] end *layer_moe_fused cost 2.067 ms
DEBUG 05-06 11:02:22.039242.039242 cuda_h.py:27] end decode_layer cost 5.590 ms
DEBUG 05-06 11:02:22.039768.039768 lmp.py:904] ---- decode step 1 layer 9 ----
DEBUG 05-06 11:02:22.041296.041296 cuda_h.py:27] end *sagl cost 1.880 ms
DEBUG 05-06 11:02:22.044501.044501 cuda_h.py:27] end *layer_moe_fused cost 2.104 ms
DEBUG 05-06 11:02:22.045438.045438 cuda_h.py:27] end decode_layer cost 5.580 ms
DEBUG 05-06 11:02:22.045579.045579 lmp.py:904] ---- decode step 1 layer 10 ----
DEBUG 05-06 11:02:22.047968.047968 cuda_h.py:27] end *sagl cost 1.845 ms
DEBUG 05-06 11:02:22.050541.050541 cuda_h.py:27] end *layer_moe_fused cost 2.153 ms
DEBUG 05-06 11:02:22.051815.051815 cuda_h.py:27] end decode_layer cost 5.646 ms
DEBUG 05-06 11:02:22.051910.051910 lmp.py:904] ---- decode step 1 layer 11 ----
DEBUG 05-06 11:02:22.053934.053934 cuda_h.py:27] end *sagl cost 1.859 ms
DEBUG 05-06 11:02:22.056504.056504 cuda_h.py:27] end *layer_moe_fused cost 2.091 ms
DEBUG 05-06 11:02:22.056871.056871 cuda_h.py:27] end decode_layer cost 5.549 ms
DEBUG 05-06 11:02:22.056728.056728 lmp.py:904] ---- decode step 1 layer 12 ----
DEBUG 05-06 11:02:22.058533.058533 cuda_h.py:27] end *sagl cost 1.838 ms
DEBUG 05-06 11:02:22.061155.061155 cuda_h.py:27] end *layer_moe_fused cost 2.094 ms
DEBUG 05-06 11:02:22.062853.062853 cuda_h.py:27] end decode_layer cost 5.530 ms
DEBUG 05-06 11:02:22.062233.062233 lmp.py:904] ---- decode step 1 layer 13 ----
DEBUG 05-06 11:02:22.064882.064882 cuda_h.py:27] end *sagl cost 1.934 ms
DEBUG 05-06 11:02:22.067352.067352 cuda_h.py:27] end *layer_moe_fused cost 2.091 ms
DEBUG 05-06 11:02:22.067694.067694 cuda_h.py:27] end decode_layer cost 5.636 ms
DEBUG 05-06 11:02:22.068650.068650 lmp.py:904] ---- decode step 1 layer 14 ----
DEBUG 05-06 11:02:22.069885.069885 cuda_h.py:27] end *sagl cost 1.839 ms
DEBUG 05-06 11:02:22.073065.073065 cuda_h.py:27] end *layer_moe_fused cost 2.136 ms
DEBUG 05-06 11:02:22.073856.073856 cuda_h.py:27] end decode_layer cost 5.554 ms
DEBUG 05-06 11:02:22.073189.073189 lmp.py:904] ---- decode step 1 layer 15 ----
DEBUG 05-06 11:02:22.075095.075095 cuda_h.py:27] end *sagl cost 1.877 ms
DEBUG 05-06 11:02:22.078392.078392 cuda_h.py:27] end *layer_moe_fused cost 2.074 ms
DEBUG 05-06 11:02:22.079852.079852 cuda_h.py:27] end decode_layer cost 5.539 ms
DEBUG 05-06 11:02:22.079470.079470 lmp.py:904] ---- decode step 1 layer 16 ----
DEBUG 05-06 11:02:22.081130.081130 cuda_h.py:27] end *sagl cost 1.871 ms
DEBUG 05-06 11:02:22.084274.084274 cuda_h.py:27] end *layer_moe_fused cost 2.045 ms
DEBUG 05-06 11:02:22.084548.084548 cuda_h.py:27] end decode_layer cost 5.495 ms
DEBUG 05-06 11:02:22.084928.084928 lmp.py:904] ---- decode step 1 layer 17 ----
DEBUG 05-06 11:02:22.086441.086441 cuda_h.py:27] end *sagl cost 1.834 ms
DEBUG 05-06 11:02:22.089521.089521 cuda_h.py:27] end *layer_moe_fused cost 2.068 ms
DEBUG 05-06 11:02:22.090034.090034 cuda_h.py:27] end decode_layer cost 5.551 ms
DEBUG 05-06 11:02:22.090653.090653 lmp.py:904] ---- decode step 1 layer 18 ----
DEBUG 05-06 11:02:22.092538.092538 cuda_h.py:27] end *sagl cost 1.862 ms
DEBUG 05-06 11:02:22.095437.095437 cuda_h.py:27] end *layer_moe_fused cost 2.070 ms
DEBUG 05-06 11:02:22.096085.096085 cuda_h.py:27] end decode_layer cost 5.580 ms
DEBUG 05-06 11:02:22.096087.096087 lmp.py:904] ---- decode step 1 layer 19 ----
DEBUG 05-06 11:02:22.098006.098006 cuda_h.py:27] end *sagl cost 1.886 ms
DEBUG 05-06 11:02:22.101358.101358 cuda_h.py:27] end *layer_moe_fused cost 2.099 ms
DEBUG 05-06 11:02:22.101248.101248 cuda_h.py:27] end decode_layer cost 5.591 ms
DEBUG 05-06 11:02:22.101688.101688 lmp.py:904] ---- decode step 1 layer 20 ----
DEBUG 05-06 11:02:22.103144.103144 cuda_h.py:27] end *sagl cost 1.895 ms
DEBUG 05-06 11:02:22.106003.106003 cuda_h.py:27] end *layer_moe_fused cost 2.074 ms
DEBUG 05-06 11:02:22.107993.107993 cuda_h.py:27] end decode_layer cost 5.557 ms
DEBUG 05-06 11:02:22.107850.107850 lmp.py:904] ---- decode step 1 layer 21 ----
DEBUG 05-06 11:02:22.109055.109055 cuda_h.py:27] end *sagl cost 1.922 ms
DEBUG 05-06 11:02:22.112341.112341 cuda_h.py:27] end *layer_moe_fused cost 2.118 ms
DEBUG 05-06 11:02:22.113562.113562 cuda_h.py:27] end decode_layer cost 5.647 ms
DEBUG 05-06 11:02:22.113134.113134 lmp.py:904] ---- decode step 1 layer 22 ----
DEBUG 05-06 11:02:22.115432.115432 cuda_h.py:27] end *sagl cost 1.920 ms
DEBUG 05-06 11:02:22.118935.118935 cuda_h.py:27] end *layer_moe_fused cost 2.068 ms
DEBUG 05-06 11:02:22.118732.118732 cuda_h.py:27] end decode_layer cost 5.598 ms
DEBUG 05-06 11:02:22.118589.118589 lmp.py:904] ---- decode step 1 layer 23 ----
DEBUG 05-06 11:02:22.120668.120668 cuda_h.py:27] end *sagl cost 1.898 ms
DEBUG 05-06 11:02:22.123071.123071 cuda_h.py:27] end *layer_moe_fused cost 2.090 ms
DEBUG 05-06 11:02:22.124670.124670 cuda_h.py:27] end decode_layer cost 5.567 ms
DEBUG 05-06 11:02:22.124857.124857 lmp.py:904] ---- decode step 1 layer 24 ----
DEBUG 05-06 11:02:22.126570.126570 cuda_h.py:27] end *sagl cost 1.840 ms
DEBUG 05-06 11:02:22.129537.129537 cuda_h.py:27] end *layer_moe_fused cost 2.118 ms
DEBUG 05-06 11:02:22.130765.130765 cuda_h.py:27] end decode_layer cost 5.544 ms
DEBUG 05-06 11:02:22.130906.130906 lmp.py:904] ---- decode step 1 layer 25 ----
DEBUG 05-06 11:02:22.132999.132999 cuda_h.py:27] end *sagl cost 1.908 ms
DEBUG 05-06 11:02:22.135899.135899 cuda_h.py:27] end *layer_moe_fused cost 2.078 ms
DEBUG 05-06 11:02:22.135579.135579 cuda_h.py:27] end decode_layer cost 5.652 ms
DEBUG 05-06 11:02:22.135866.135866 lmp.py:904] ---- decode step 1 layer 26 ----
DEBUG 05-06 11:02:22.137400.137400 cuda_h.py:27] end *sagl cost 1.848 ms
DEBUG 05-06 11:02:22.140135.140135 cuda_h.py:27] end *layer_moe_fused cost 2.091 ms
DEBUG 05-06 11:02:22.141263.141263 cuda_h.py:27] end decode_layer cost 5.519 ms
DEBUG 05-06 11:02:22.141604.141604 lmp.py:904] ---- decode step 1 layer 27 ----
DEBUG 05-06 11:02:22.143338.143338 cuda_h.py:27] end *sagl cost 1.890 ms
DEBUG 05-06 11:02:22.146041.146041 cuda_h.py:27] end *layer_moe_fused cost 2.086 ms
DEBUG 05-06 11:02:22.147263.147263 cuda_h.py:27] end decode_layer cost 5.643 ms
DEBUG 05-06 11:02:22.147927.147927 lmp.py:904] ---- decode step 1 layer 28 ----
DEBUG 05-06 11:02:22.149568.149568 cuda_h.py:27] end *sagl cost 1.857 ms
DEBUG 05-06 11:02:22.152493.152493 cuda_h.py:27] end *layer_moe_fused cost 2.065 ms
DEBUG 05-06 11:02:22.152906.152906 cuda_h.py:27] end decode_layer cost 5.491 ms
DEBUG 05-06 11:02:22.152048.152048 lmp.py:904] ---- decode step 1 layer 29 ----
DEBUG 05-06 11:02:22.154230.154230 cuda_h.py:27] end *sagl cost 1.835 ms
DEBUG 05-06 11:02:22.157355.157355 cuda_h.py:27] end *layer_moe_fused cost 2.069 ms
DEBUG 05-06 11:02:22.158007.158007 cuda_h.py:27] end decode_layer cost 5.512 ms
DEBUG 05-06 11:02:22.158579.158579 cuda_h.py:27] end decode_step cost 180.288 ms
INFO 05-06 11:02:22.158488.158488 lmp.py:931] decode step 1 time: 0.1803271770477295 seconds
Time taken: 6.2465232983231544 seconds
X512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x586709301d10, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
CPUInfer[0x5866ff84f310]: Goodbye
