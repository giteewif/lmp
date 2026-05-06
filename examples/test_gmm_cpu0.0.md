here pin
INFO 05-06 10:41:54.824679.824679 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-06 10:41:55.367056.367056 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-06 10:41:55.797113.797113 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-06 10:41:55.797060.797060 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 0.973s
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
INFO 05-06 10:42:03.351482.351482 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-06 10:42:03.790482.790482 cuda_h.py:27] end init_cmv_hmv cost 440.025 ms
DEBUG 05-06 10:42:03.798836.798836 cuda_memory_view.py:1366] 
DEBUG 05-06 10:42:03.798836.798836 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.002657651901245117
DEBUG 05-06 10:42:03.815065.815065 mlpmodule.py:993] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-06 10:42:03.815870.815870 cuda_memory_view.py:1370] 
DEBUG 05-06 10:42:03.815870.815870 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.017276763916015625
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-06 10:42:05.704181.704181 lmp.py:368] init kt-kernel layer 0 ok
INFO 05-06 10:42:06.509749.509749 lmp.py:368] init kt-kernel layer 1 ok
INFO 05-06 10:42:07.331526.331526 lmp.py:368] init kt-kernel layer 2 ok
INFO 05-06 10:42:08.169141.169141 lmp.py:368] init kt-kernel layer 3 ok
INFO 05-06 10:42:08.977960.977960 lmp.py:368] init kt-kernel layer 4 ok
INFO 05-06 10:42:09.776046.776046 lmp.py:368] init kt-kernel layer 5 ok
INFO 05-06 10:42:10.598989.598989 lmp.py:368] init kt-kernel layer 6 ok
INFO 05-06 10:42:11.439599.439599 lmp.py:368] init kt-kernel layer 7 ok
INFO 05-06 10:42:12.277247.277247 lmp.py:368] init kt-kernel layer 8 ok
INFO 05-06 10:42:13.111622.111622 lmp.py:368] init kt-kernel layer 9 ok
INFO 05-06 10:42:13.946304.946304 lmp.py:368] init kt-kernel layer 10 ok
INFO 05-06 10:42:14.780901.780901 lmp.py:368] init kt-kernel layer 11 ok
INFO 05-06 10:42:15.583471.583471 lmp.py:368] init kt-kernel layer 12 ok
INFO 05-06 10:42:16.440501.440501 lmp.py:368] init kt-kernel layer 13 ok
INFO 05-06 10:42:17.279379.279379 lmp.py:368] init kt-kernel layer 14 ok
INFO 05-06 10:42:18.117013.117013 lmp.py:368] init kt-kernel layer 15 ok
INFO 05-06 10:42:18.957758.957758 lmp.py:368] init kt-kernel layer 16 ok
INFO 05-06 10:42:19.778523.778523 lmp.py:368] init kt-kernel layer 17 ok
INFO 05-06 10:42:20.596169.596169 lmp.py:368] init kt-kernel layer 18 ok
INFO 05-06 10:42:21.423515.423515 lmp.py:368] init kt-kernel layer 19 ok
INFO 05-06 10:42:22.240405.240405 lmp.py:368] init kt-kernel layer 20 ok
INFO 05-06 10:42:23.076923.076923 lmp.py:368] init kt-kernel layer 21 ok
INFO 05-06 10:42:23.902474.902474 lmp.py:368] init kt-kernel layer 22 ok
CPUInfer[0x6290f7a3aea0]: Hello
WorkerPool[0x6290f7a4f520] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x62910ef33910]: Hello
WorkerPool[0x6291152952c0] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVINFO 05-06 10:42:24.724852.724852 lmp.py:368] init kt-kernel layer 23 ok
INFO 05-06 10:42:25.572807.572807 lmp.py:368] init kt-kernel layer 24 ok
INFO 05-06 10:42:26.402771.402771 lmp.py:368] init kt-kernel layer 25 ok
INFO 05-06 10:42:27.226626.226626 lmp.py:368] init kt-kernel layer 26 ok
INFO 05-06 10:42:28.040162.040162 lmp.py:368] init kt-kernel layer 27 ok
INFO 05-06 10:42:28.839464.839464 lmp.py:368] init kt-kernel layer 28 ok
INFO 05-06 10:42:29.664786.664786 lmp.py:368] init kt-kernel layer 29 ok
generate input ids cost 0.08766818046569824 s
DEBUG 05-06 10:42:32.887168.887168 cuda_h.py:27] end generate_input_ids cost 3167.839 ms
DEBUG 05-06 10:42:32.887008.887008 cuda_h.py:27] end init_cache cost 0.042 ms
INFO 05-06 10:42:32.900961.900961 lmp.py:2341] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6617276356, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7277174582576769, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-06 10:42:32.901488.901488 lmp.py:2359] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901510.901510 lmp.py:2359] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901233.901233 lmp.py:2359] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901049.901049 lmp.py:2359] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901149.901149 lmp.py:2359] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901174.901174 lmp.py:2359] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901275.901275 lmp.py:2359] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901852.901852 lmp.py:2359] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901522.901522 lmp.py:2359] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901715.901715 lmp.py:2359] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.901605.901605 lmp.py:2359] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.902660.902660 lmp.py:2359] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.902045.902045 lmp.py:2359] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.902052.902052 lmp.py:2359] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.902583.902583 lmp.py:2359] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.902934.902934 lmp.py:2359] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.902988.902988 lmp.py:2359] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.902140.902140 lmp.py:2359] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.902718.902718 lmp.py:2359] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.902399.902399 lmp.py:2359] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903785.903785 lmp.py:2359] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903654.903654 lmp.py:2359] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903086.903086 lmp.py:2359] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903279.903279 lmp.py:2359] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903763.903763 lmp.py:2359] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903864.903864 lmp.py:2359] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903772.903772 lmp.py:2359] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903167.903167 lmp.py:2359] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903267.903267 lmp.py:2359] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:32.903447.903447 lmp.py:2359] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 10:42:33.182071.182071 cuda_h.py:27] end init_loading_placement cost 294.761 ms
DEBUG 05-06 10:42:33.182936.182936 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:42:33.182580.182580 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:42:33 client.py:72] load_into_gpu: gemma4-26B-A4B, 8e0330f9-67fd-4f74-8e99-e2be7d4befc4
INFO 05-06 10:42:33 client.py:135] Model loaded: gemma4-26B-A4B, 8e0330f9-67fd-4f74-8e99-e2be7d4befc4
INFO 05-06 10:42:33 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 8e0330f9-67fd-4f74-8e99-e2be7d4befc4
INFO 05-06 10:42:33 client.py:212] Model loaded
DEBUG 05-06 10:42:33.715067.715067 cuda_h.py:27] end init_general_sagl_loading_async cost 532.774 ms
INFO 05-06 10:42:33.763084.763084 lmp.py:2862] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 10:42:33.861006.861006 cuda_h.py:27] end restore_state_dict cost 98.028 ms
DEBUG 05-06 10:42:33.861838.861838 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:42:33.861112.861112 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:42:33 client.py:72] load_into_gpu: gemma4-26B-A4B, a907e65c-43f2-4dda-a4bd-68a27dadb99d
INFO 05-06 10:42:33 client.py:135] Model loaded: gemma4-26B-A4B, a907e65c-43f2-4dda-a4bd-68a27dadb99d
DEBUG 05-06 10:42:33.936627.936627 cuda_h.py:27] end init_experts_loading_async cost 74.755 ms
DEBUG 05-06 10:42:33.968913.968913 cuda_h.py:27] end init_inputs_tokens cost 31.751 ms
DEBUG 05-06 10:42:33.968412.968412 lmp.py:824] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 10:42:34.116591.116591 cuda_h.py:27] end *sagl cost 148.272 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 29, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 5090, 'token_per_expert': {3: 160, 7: 374, 11: 31, 15: 13, 19: 2, 23: 23, 27: 10, 31: 134, 39: 718, 47: 1304, 51: 186, 55: 208, 59: 43, 63: 15, 67: 183, 71: 65, 75: 89, 79: 76, 83: 105, 87: 2, 91: 458, 99: 161, 103: 432, 107: 25, 111: 23, 115: 89, 119: 13, 123: 39, 127: 109}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 48, 52, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3457, 'token_per_expert': {0: 249, 4: 11, 8: 9, 12: 2, 16: 201, 20: 15, 24: 81, 28: 123, 32: 183, 36: 4, 44: 17, 48: 146, 52: 150, 60: 55, 64: 106, 68: 694, 72: 100, 76: 74, 80: 28, 84: 21, 88: 1, 92: 87, 96: 8, 100: 5, 104: 134, 108: 78, 112: 68, 116: 82, 120: 1, 124: 724}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3777, 'token_per_expert': {1: 273, 5: 66, 9: 68, 13: 61, 17: 3, 21: 171, 25: 110, 29: 7, 33: 828, 37: 81, 41: 142, 45: 14, 49: 17, 53: 819, 65: 39, 69: 78, 73: 60, 77: 99, 81: 11, 85: 3, 89: 133, 93: 12, 97: 3, 101: 6, 105: 89, 109: 3, 113: 157, 117: 97, 121: 226, 125: 101}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 46, 50, 54, 66, 70, 74, 78, 86, 90, 94, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 27, 'ideal_gpu_count': 29, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 4060, 'token_per_expert': {2: 30, 6: 4, 10: 36, 14: 29, 18: 71, 22: 255, 26: 304, 30: 1, 34: 38, 38: 59, 46: 450, 50: 520, 54: 275, 66: 6, 70: 140, 74: 224, 78: 109, 86: 2, 90: 546, 94: 24, 102: 74, 106: 14, 110: 83, 114: 48, 118: 89, 122: 114, 126: 515}}
INFO 05-06 10:42:34.257189.257189 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 59.920ms | allocate_experts_across_cpu_gpu: 0.316ms
INFO 05-06 10:42:34.258090.258090 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.843971252441406e-05 seconds
INFO 05-06 10:42:34.260173.260173 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0022139549255371094 seconds
INFO 05-06 10:42:34.317671.317671 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.056673526763916016 seconds
INFO 05-06 10:42:34.318897.318897 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016961097717285156 seconds
INFO 05-06 10:42:34.367576.367576 mlpmodule.py:2799] [fused_experts] gmm total=47.982ms E=32 S=5090 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.411881.411881 mlpmodule.py:2799] [fused_experts] gmm total=90.732ms E=32 S=3777 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.425916.425916 mlpmodule.py:2799] [fused_experts] gmm total=105.743ms E=32 S=3457 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.425707.425707 mlpmodule.py:2799] [fused_experts] gmm total=104.931ms E=32 S=4060 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.427520.427520 lmp.py:1484] [layer_moe_fused] experts compute time: 0.10811519622802734 seconds
INFO 05-06 10:42:34.427868.427868 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00012421607971191406 seconds
DEBUG 05-06 10:42:34.427041.427041 cuda_h.py:27] end *layer_moe_fused cost 230.291 ms
DEBUG 05-06 10:42:34.440047.440047 cuda_h.py:27] end prefill_layer cost 471.660 ms
DEBUG 05-06 10:42:34.440635.440635 lmp.py:841] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 10:42:34.440949.440949 lmp.py:824] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 10:42:34.444347.444347 cuda_h.py:27] end *sagl cost 4.190 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 75, 79, 83, 87, 91, 95, 99, 103, 107, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 2687, 'token_per_expert': {3: 61, 7: 155, 11: 48, 15: 21, 23: 6, 27: 33, 31: 14, 35: 63, 39: 5, 43: 4, 47: 158, 51: 234, 55: 15, 59: 152, 63: 3, 67: 449, 75: 7, 79: 73, 83: 35, 87: 20, 91: 13, 95: 62, 99: 546, 103: 22, 107: 5, 115: 8, 119: 150, 123: 37, 127: 288}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4457, 'token_per_expert': {0: 23, 4: 81, 8: 472, 12: 219, 16: 7, 20: 207, 24: 3, 28: 190, 32: 26, 40: 20, 44: 6, 48: 39, 52: 1229, 56: 49, 60: 26, 64: 90, 68: 693, 72: 28, 76: 29, 80: 151, 84: 26, 88: 26, 92: 67, 96: 204, 100: 203, 104: 64, 108: 50, 112: 21, 116: 24, 120: 102, 124: 82}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4068, 'token_per_expert': {1: 142, 5: 410, 9: 102, 13: 1121, 21: 71, 25: 164, 29: 27, 33: 5, 37: 36, 41: 10, 45: 47, 49: 104, 53: 158, 57: 28, 61: 2, 65: 154, 69: 40, 73: 96, 77: 16, 81: 10, 85: 101, 89: 14, 93: 25, 97: 487, 101: 52, 105: 71, 109: 531, 117: 3, 121: 33, 125: 8}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 62, 66, 70, 74, 78, 82, 90, 94, 98, 106, 110, 114, 118, 122], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 5172, 'token_per_expert': {2: 9, 6: 35, 10: 623, 14: 15, 18: 12, 22: 466, 26: 17, 30: 968, 34: 65, 38: 40, 42: 161, 46: 198, 50: 49, 54: 230, 62: 24, 66: 29, 70: 2, 74: 63, 78: 29, 82: 794, 90: 61, 94: 120, 98: 62, 106: 213, 110: 24, 114: 5, 118: 291, 122: 567}}
INFO 05-06 10:42:34.447443.447443 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.594ms | allocate_experts_across_cpu_gpu: 0.383ms
INFO 05-06 10:42:34.447230.447230 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.62939453125e-05 seconds
INFO 05-06 10:42:34.449896.449896 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018336772918701172 seconds
INFO 05-06 10:42:34.480358.480358 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.031055927276611328 seconds
INFO 05-06 10:42:34.482820.482820 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016779899597167969 seconds
INFO 05-06 10:42:34.485001.485001 mlpmodule.py:2799] [fused_experts] gmm total=3.361ms E=32 S=2687 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.486845.486845 mlpmodule.py:2799] [fused_experts] gmm total=3.566ms E=32 S=4457 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.487459.487459 mlpmodule.py:2799] [fused_experts] gmm total=4.669ms E=32 S=5172 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.487023.487023 mlpmodule.py:2799] [fused_experts] gmm total=4.979ms E=32 S=4068 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.490565.490565 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007734537124633789 seconds
INFO 05-06 10:42:34.490570.490570 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.103515625e-05 seconds
DEBUG 05-06 10:42:34.490990.490990 cuda_h.py:27] end *layer_moe_fused cost 44.431 ms
DEBUG 05-06 10:42:34.513243.513243 cuda_h.py:27] end prefill_layer cost 72.863 ms
DEBUG 05-06 10:42:34.513662.513662 lmp.py:841] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 10:42:34.513080.513080 lmp.py:824] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 10:42:34.514003.514003 cuda_h.py:27] end *sagl cost 1.594 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4791, 'token_per_expert': {3: 140, 7: 257, 11: 1057, 15: 370, 19: 583, 23: 27, 27: 46, 31: 99, 35: 28, 43: 76, 47: 1, 51: 104, 55: 221, 59: 444, 63: 91, 67: 6, 71: 68, 75: 6, 79: 4, 83: 95, 87: 1, 91: 147, 95: 28, 99: 6, 103: 69, 107: 78, 111: 46, 115: 39, 119: 80, 123: 103, 127: 471}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 96, 100, 104, 108, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3670, 'token_per_expert': {0: 50, 4: 81, 8: 117, 12: 18, 16: 4, 20: 216, 24: 72, 28: 54, 32: 4, 36: 92, 40: 21, 44: 100, 48: 234, 52: 52, 56: 60, 60: 215, 64: 18, 68: 6, 72: 51, 76: 270, 80: 234, 84: 228, 88: 45, 96: 33, 100: 73, 104: 149, 108: 980, 116: 72, 120: 33, 124: 88}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 77, 81, 85, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4620, 'token_per_expert': {1: 412, 5: 21, 9: 434, 13: 398, 17: 74, 21: 10, 25: 8, 29: 310, 33: 78, 37: 273, 41: 544, 45: 18, 49: 115, 53: 184, 57: 125, 61: 31, 65: 179, 69: 108, 77: 100, 81: 391, 85: 65, 93: 1, 97: 138, 101: 2, 105: 35, 109: 142, 113: 42, 117: 1, 121: 30, 125: 351}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 14, 18, 22, 26, 34, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 27, 'ideal_gpu_count': 29, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 3303, 'token_per_expert': {6: 25, 14: 123, 18: 169, 22: 1, 26: 12, 34: 151, 42: 38, 46: 47, 50: 13, 54: 391, 58: 42, 62: 568, 66: 9, 70: 68, 74: 1, 78: 207, 82: 24, 86: 5, 90: 248, 98: 89, 102: 329, 106: 182, 110: 100, 114: 28, 118: 219, 122: 109, 126: 105}}
INFO 05-06 10:42:34.517061.517061 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 1.354ms | allocate_experts_across_cpu_gpu: 0.265ms
INFO 05-06 10:42:34.517383.517383 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.413459777832031e-05 seconds
INFO 05-06 10:42:34.519909.519909 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015976428985595703 seconds
INFO 05-06 10:42:34.546477.546477 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.027310609817504883 seconds
INFO 05-06 10:42:34.548432.548432 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015535354614257812 seconds
INFO 05-06 10:42:34.551960.551960 mlpmodule.py:2799] [fused_experts] gmm total=2.999ms E=32 S=3670 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.551619.551619 mlpmodule.py:2799] [fused_experts] gmm total=3.138ms E=32 S=3303 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.559735.559735 mlpmodule.py:2799] [fused_experts] gmm total=11.241ms E=32 S=4620 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.576510.576510 mlpmodule.py:2799] [fused_experts] gmm total=28.065ms E=32 S=4791 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.576792.576792 lmp.py:1484] [layer_moe_fused] experts compute time: 0.028498411178588867 seconds
INFO 05-06 10:42:34.577292.577292 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00012159347534179688 seconds
DEBUG 05-06 10:42:34.577865.577865 cuda_h.py:27] end *layer_moe_fused cost 61.822 ms
DEBUG 05-06 10:42:34.578777.578777 cuda_h.py:27] end prefill_layer cost 65.283 ms
DEBUG 05-06 10:42:34.578323.578323 lmp.py:841] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 10:42:34.578391.578391 lmp.py:824] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 10:42:34.581964.581964 cuda_h.py:27] end *sagl cost 2.646 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 23, 27, 31, 35, 39, 43, 51, 55, 59, 63, 67, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 31, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 2785, 'token_per_expert': {3: 109, 11: 130, 15: 187, 19: 108, 23: 17, 27: 26, 31: 64, 35: 7, 39: 89, 43: 82, 51: 162, 55: 18, 59: 97, 63: 85, 67: 51, 71: 329, 75: 394, 83: 182, 87: 8, 91: 18, 95: 202, 99: 1, 103: 2, 107: 109, 111: 64, 115: 9, 119: 103, 123: 85, 127: 47}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 116, 120], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3935, 'token_per_expert': {0: 160, 4: 292, 8: 51, 12: 4, 16: 35, 20: 7, 24: 42, 28: 253, 32: 15, 36: 6, 40: 54, 44: 99, 48: 45, 52: 281, 56: 39, 60: 37, 64: 103, 68: 207, 72: 25, 76: 249, 80: 7, 84: 244, 88: 299, 92: 282, 96: 314, 100: 75, 104: 236, 108: 159, 116: 50, 120: 265}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3866, 'token_per_expert': {1: 34, 5: 291, 9: 318, 13: 71, 17: 179, 21: 5, 25: 178, 29: 6, 33: 53, 37: 7, 41: 33, 45: 1, 53: 252, 57: 24, 61: 71, 65: 15, 69: 155, 73: 251, 77: 86, 81: 1, 85: 545, 89: 22, 93: 346, 97: 263, 101: 87, 105: 2, 109: 78, 113: 1, 117: 44, 121: 441, 125: 6}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5798, 'token_per_expert': {2: 114, 6: 81, 10: 168, 14: 312, 18: 9, 22: 540, 26: 90, 30: 41, 34: 263, 38: 2, 42: 29, 46: 2, 50: 687, 54: 196, 58: 105, 62: 430, 66: 341, 70: 148, 74: 161, 78: 693, 82: 20, 86: 34, 94: 11, 98: 19, 102: 423, 106: 4, 110: 108, 114: 125, 118: 248, 122: 393, 126: 1}}
INFO 05-06 10:42:34.584674.584674 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.647ms | allocate_experts_across_cpu_gpu: 0.436ms
INFO 05-06 10:42:34.584236.584236 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.082389831542969e-05 seconds
INFO 05-06 10:42:34.586268.586268 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016469955444335938 seconds
INFO 05-06 10:42:34.618221.618221 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.032087087631225586 seconds
INFO 05-06 10:42:34.620125.620125 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016064643859863281 seconds
INFO 05-06 10:42:34.624598.624598 mlpmodule.py:2799] [fused_experts] gmm total=4.068ms E=32 S=3866 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.625936.625936 mlpmodule.py:2799] [fused_experts] gmm total=4.479ms E=32 S=2785 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.626852.626852 mlpmodule.py:2799] [fused_experts] gmm total=6.219ms E=32 S=3935 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.628141.628141 mlpmodule.py:2799] [fused_experts] gmm total=7.919ms E=32 S=5798 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.631709.631709 lmp.py:1484] [layer_moe_fused] experts compute time: 0.01135396957397461 seconds
INFO 05-06 10:42:34.631156.631156 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.14984130859375e-05 seconds
DEBUG 05-06 10:42:34.631326.631326 cuda_h.py:27] end *layer_moe_fused cost 48.830 ms
DEBUG 05-06 10:42:34.651723.651723 cuda_h.py:27] end prefill_layer cost 73.202 ms
DEBUG 05-06 10:42:34.651519.651519 lmp.py:841] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 10:42:34.652130.652130 lmp.py:824] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 10:42:34.653748.653748 cuda_h.py:27] end *sagl cost 1.613 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 6969, 'token_per_expert': {3: 156, 7: 30, 15: 54, 19: 169, 23: 467, 27: 272, 31: 12, 35: 1, 39: 205, 43: 567, 47: 184, 51: 298, 55: 239, 59: 643, 63: 1029, 67: 272, 71: 116, 75: 64, 79: 9, 83: 428, 87: 107, 91: 73, 95: 4, 103: 34, 107: 64, 111: 453, 115: 400, 119: 502, 123: 109, 127: 8}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 116, 120, 124], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 3056, 'token_per_expert': {4: 139, 8: 708, 12: 11, 16: 3, 20: 137, 24: 310, 28: 124, 32: 151, 36: 58, 40: 52, 44: 29, 48: 2, 52: 82, 56: 32, 60: 128, 64: 61, 72: 13, 76: 210, 80: 19, 84: 49, 88: 42, 92: 141, 96: 106, 100: 1, 104: 123, 108: 84, 116: 97, 120: 14, 124: 130}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3180, 'token_per_expert': {1: 296, 5: 182, 9: 2, 13: 1, 17: 102, 21: 29, 25: 51, 29: 207, 37: 33, 41: 15, 45: 58, 49: 79, 53: 249, 57: 53, 61: 106, 65: 4, 69: 28, 73: 36, 77: 51, 81: 63, 85: 124, 89: 455, 93: 197, 97: 114, 101: 54, 105: 109, 109: 46, 113: 250, 117: 75, 121: 5, 125: 106}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3179, 'token_per_expert': {2: 5, 6: 31, 14: 2, 18: 34, 22: 411, 26: 349, 30: 82, 34: 35, 38: 31, 42: 1, 46: 24, 50: 11, 54: 345, 58: 12, 62: 93, 66: 15, 70: 1, 74: 410, 78: 97, 82: 256, 86: 108, 90: 43, 94: 84, 98: 78, 102: 1, 106: 505, 110: 7, 114: 16, 118: 54, 122: 24, 126: 14}}
INFO 05-06 10:42:34.656248.656248 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 1.290ms | allocate_experts_across_cpu_gpu: 0.272ms
INFO 05-06 10:42:34.656464.656464 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.817413330078125e-05 seconds
INFO 05-06 10:42:34.657327.657327 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016002655029296875 seconds
INFO 05-06 10:42:34.680345.680345 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.021953344345092773 seconds
INFO 05-06 10:42:34.681776.681776 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015168190002441406 seconds
INFO 05-06 10:42:34.686911.686911 mlpmodule.py:2799] [fused_experts] gmm total=4.082ms E=32 S=3056 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.686290.686290 mlpmodule.py:2799] [fused_experts] gmm total=4.421ms E=32 S=6969 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.686322.686322 mlpmodule.py:2799] [fused_experts] gmm total=4.802ms E=32 S=3180 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.687132.687132 mlpmodule.py:2799] [fused_experts] gmm total=5.176ms E=32 S=3179 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.688983.688983 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006899833679199219 seconds
INFO 05-06 10:42:34.688331.688331 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.030632019042969e-05 seconds
DEBUG 05-06 10:42:34.688387.688387 cuda_h.py:27] end *layer_moe_fused cost 34.392 ms
DEBUG 05-06 10:42:34.704716.704716 cuda_h.py:27] end prefill_layer cost 52.916 ms
DEBUG 05-06 10:42:34.705089.705089 lmp.py:841] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 10:42:34.705183.705183 lmp.py:824] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 10:42:34.741561.741561 cuda_h.py:27] end *sagl cost 36.756 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 51, 55, 63, 67, 71, 75, 79, 83, 87, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 28, 'ideal_gpu_count': 30, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 4158, 'token_per_expert': {3: 4, 7: 28, 11: 4, 15: 10, 19: 24, 23: 160, 27: 20, 31: 53, 39: 581, 43: 113, 51: 12, 55: 79, 63: 120, 67: 101, 71: 1125, 75: 112, 79: 62, 83: 53, 87: 191, 95: 2, 99: 309, 103: 1, 107: 33, 111: 269, 115: 18, 119: 81, 123: 199, 127: 394}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 5014, 'token_per_expert': {0: 75, 4: 196, 8: 3, 16: 453, 20: 616, 24: 200, 28: 226, 32: 17, 36: 357, 40: 1, 44: 68, 48: 7, 52: 78, 56: 9, 60: 188, 64: 448, 68: 45, 72: 283, 76: 120, 80: 112, 84: 48, 88: 244, 92: 10, 96: 124, 100: 85, 104: 188, 112: 510, 116: 164, 120: 138, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 37, 41, 45, 49, 53, 57, 61, 69, 73, 77, 81, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 4528, 'token_per_expert': {1: 3, 5: 149, 9: 214, 13: 240, 17: 12, 21: 5, 29: 173, 33: 488, 37: 34, 41: 9, 45: 4, 49: 659, 53: 9, 57: 36, 61: 324, 69: 1, 73: 209, 77: 27, 81: 17, 93: 148, 97: 11, 101: 1267, 105: 33, 109: 1, 113: 73, 117: 305, 121: 1, 125: 76}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 29, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 2684, 'token_per_expert': {2: 417, 6: 45, 10: 32, 14: 53, 18: 83, 22: 391, 26: 26, 30: 12, 34: 26, 38: 10, 42: 256, 46: 155, 50: 18, 54: 24, 58: 20, 62: 14, 66: 1, 70: 197, 74: 161, 78: 5, 82: 3, 86: 32, 90: 1, 94: 229, 98: 36, 102: 27, 106: 64, 110: 6, 114: 52, 118: 128, 122: 10, 126: 150}}
INFO 05-06 10:42:34.746639.746639 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 3.052ms | allocate_experts_across_cpu_gpu: 0.286ms
INFO 05-06 10:42:34.746246.746246 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.747245788574219e-05 seconds
INFO 05-06 10:42:34.747624.747624 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015616416931152344 seconds
INFO 05-06 10:42:34.775417.775417 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.027875661849975586 seconds
INFO 05-06 10:42:34.777852.777852 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016446113586425781 seconds
INFO 05-06 10:42:34.782675.782675 mlpmodule.py:2799] [fused_experts] gmm total=4.379ms E=32 S=4528 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.782089.782089 mlpmodule.py:2799] [fused_experts] gmm total=4.904ms E=32 S=4158 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.783739.783739 mlpmodule.py:2799] [fused_experts] gmm total=5.027ms E=32 S=5014 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.783519.783519 mlpmodule.py:2799] [fused_experts] gmm total=5.353ms E=32 S=2684 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.785469.785469 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007611751556396484 seconds
INFO 05-06 10:42:34.785347.785347 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.1021575927734375e-05 seconds
DEBUG 05-06 10:42:34.785379.785379 cuda_h.py:27] end *layer_moe_fused cost 42.863 ms
DEBUG 05-06 10:42:34.806148.806148 cuda_h.py:27] end prefill_layer cost 101.196 ms
DEBUG 05-06 10:42:34.806660.806660 lmp.py:841] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 10:42:34.806032.806032 lmp.py:824] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 10:42:34.807324.807324 cuda_h.py:27] end *sagl cost 1.589 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4034, 'token_per_expert': {3: 33, 7: 7, 11: 29, 15: 15, 19: 35, 23: 360, 27: 94, 31: 8, 35: 543, 43: 43, 47: 14, 51: 161, 55: 2, 59: 10, 63: 1, 67: 13, 71: 125, 75: 188, 79: 195, 83: 10, 87: 359, 91: 30, 95: 90, 99: 828, 103: 60, 107: 135, 111: 25, 115: 321, 119: 150, 123: 100, 127: 50}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3740, 'token_per_expert': {0: 60, 4: 7, 8: 4, 16: 14, 20: 31, 24: 172, 28: 85, 32: 148, 36: 154, 40: 13, 44: 95, 52: 10, 56: 91, 60: 28, 64: 505, 68: 1239, 72: 13, 76: 37, 80: 44, 84: 2, 88: 1, 92: 6, 96: 163, 100: 3, 104: 208, 108: 482, 112: 7, 116: 86, 120: 17, 124: 15}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 49, 53, 57, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4068, 'token_per_expert': {1: 92, 5: 72, 9: 161, 13: 282, 17: 11, 21: 6, 25: 814, 29: 15, 33: 4, 37: 43, 41: 50, 49: 4, 53: 425, 57: 58, 65: 366, 69: 101, 73: 78, 77: 79, 81: 9, 85: 66, 89: 57, 93: 644, 97: 6, 101: 14, 105: 69, 109: 17, 113: 119, 117: 191, 121: 153, 125: 62}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4542, 'token_per_expert': {2: 97, 6: 108, 10: 105, 14: 53, 18: 20, 22: 16, 26: 139, 30: 50, 34: 345, 38: 11, 42: 55, 46: 159, 50: 138, 54: 1, 58: 93, 62: 138, 66: 4, 70: 83, 74: 32, 78: 211, 82: 40, 86: 512, 90: 427, 94: 270, 98: 258, 102: 536, 106: 395, 110: 62, 114: 13, 122: 109, 126: 62}}
INFO 05-06 10:42:34.810031.810031 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 1.313ms | allocate_experts_across_cpu_gpu: 0.275ms
INFO 05-06 10:42:34.810109.810109 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.341934204101562e-05 seconds
INFO 05-06 10:42:34.812987.812987 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016467571258544922 seconds
INFO 05-06 10:42:34.839965.839965 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02708601951599121 seconds
INFO 05-06 10:42:34.841139.841139 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015680789947509766 seconds
INFO 05-06 10:42:34.845102.845102 mlpmodule.py:2799] [fused_experts] gmm total=4.020ms E=32 S=4034 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.845519.845519 mlpmodule.py:2799] [fused_experts] gmm total=4.095ms E=32 S=3740 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.847287.847287 mlpmodule.py:2799] [fused_experts] gmm total=6.073ms E=32 S=4542 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.848500.848500 mlpmodule.py:2799] [fused_experts] gmm total=6.396ms E=32 S=4068 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.849089.849089 lmp.py:1484] [layer_moe_fused] experts compute time: 0.008151054382324219 seconds
INFO 05-06 10:42:34.849775.849775 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.0067901611328125e-05 seconds
DEBUG 05-06 10:42:34.849315.849315 cuda_h.py:27] end *layer_moe_fused cost 40.914 ms
DEBUG 05-06 10:42:34.869246.869246 cuda_h.py:27] end prefill_layer cost 62.754 ms
DEBUG 05-06 10:42:34.869851.869851 lmp.py:841] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 10:42:34.869746.869746 lmp.py:824] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 10:42:34.870975.870975 cuda_h.py:27] end *sagl cost 1.682 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 2985, 'token_per_expert': {3: 4, 7: 151, 11: 8, 15: 51, 19: 85, 23: 54, 27: 2, 31: 14, 35: 32, 39: 7, 43: 137, 47: 114, 51: 97, 55: 28, 59: 138, 63: 28, 67: 21, 71: 110, 75: 3, 79: 212, 83: 111, 87: 96, 91: 856, 95: 75, 99: 98, 103: 164, 107: 15, 111: 52, 115: 115, 119: 5, 123: 71, 127: 31}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 48, 52, 56, 60, 64, 68, 72, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4724, 'token_per_expert': {0: 34, 4: 375, 8: 87, 12: 434, 16: 40, 20: 289, 24: 5, 28: 220, 32: 27, 36: 17, 44: 251, 48: 193, 52: 353, 56: 184, 60: 173, 64: 72, 68: 87, 72: 129, 80: 52, 84: 325, 88: 32, 92: 7, 96: 170, 100: 14, 104: 151, 108: 539, 112: 124, 116: 46, 120: 291, 124: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 85, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 5099, 'token_per_expert': {1: 4, 5: 170, 9: 93, 13: 127, 17: 34, 21: 46, 25: 56, 29: 490, 33: 103, 37: 15, 41: 50, 45: 59, 49: 22, 53: 234, 57: 183, 61: 138, 65: 261, 69: 355, 73: 20, 77: 33, 85: 297, 97: 990, 101: 69, 105: 125, 109: 20, 113: 229, 117: 84, 121: 610, 125: 182}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3576, 'token_per_expert': {6: 74, 10: 271, 14: 253, 18: 141, 22: 78, 26: 35, 30: 16, 34: 400, 38: 12, 42: 225, 46: 1, 50: 12, 54: 51, 58: 3, 62: 15, 66: 49, 70: 349, 78: 24, 82: 48, 86: 233, 90: 331, 94: 14, 98: 62, 102: 7, 106: 184, 110: 261, 114: 251, 118: 68, 122: 66, 126: 42}}
INFO 05-06 10:42:34.872680.872680 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.774ms | allocate_experts_across_cpu_gpu: 0.260ms
INFO 05-06 10:42:34.873081.873081 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.8650970458984375e-05 seconds
INFO 05-06 10:42:34.874809.874809 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017120838165283203 seconds
INFO 05-06 10:42:34.904486.904486 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0299530029296875 seconds
INFO 05-06 10:42:34.906329.906329 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015614032745361328 seconds
INFO 05-06 10:42:34.911949.911949 mlpmodule.py:2799] [fused_experts] gmm total=4.098ms E=32 S=2985 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.911610.911610 mlpmodule.py:2799] [fused_experts] gmm total=4.370ms E=32 S=4724 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.912001.912001 mlpmodule.py:2799] [fused_experts] gmm total=4.946ms E=32 S=3576 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.912599.912599 mlpmodule.py:2799] [fused_experts] gmm total=5.314ms E=32 S=5099 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.914296.914296 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007777214050292969 seconds
INFO 05-06 10:42:34.914598.914598 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.0067901611328125e-05 seconds
DEBUG 05-06 10:42:34.914198.914198 cuda_h.py:27] end *layer_moe_fused cost 42.817 ms
DEBUG 05-06 10:42:34.937752.937752 cuda_h.py:27] end prefill_layer cost 68.218 ms
DEBUG 05-06 10:42:34.937549.937549 lmp.py:841] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 10:42:34.937205.937205 lmp.py:824] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 10:42:34.939516.939516 cuda_h.py:27] end *sagl cost 1.532 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4240, 'token_per_expert': {3: 126, 7: 11, 11: 65, 15: 228, 19: 292, 23: 9, 27: 175, 31: 118, 35: 9, 39: 15, 43: 35, 47: 68, 51: 643, 55: 170, 59: 3, 63: 158, 67: 5, 71: 210, 75: 281, 79: 10, 83: 2, 87: 417, 91: 31, 99: 40, 103: 778, 107: 7, 111: 77, 119: 28, 123: 157, 127: 72}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2937, 'token_per_expert': {0: 7, 4: 49, 8: 28, 12: 155, 16: 100, 20: 88, 24: 11, 28: 403, 32: 274, 36: 174, 40: 6, 44: 110, 48: 24, 52: 116, 56: 240, 60: 2, 64: 34, 68: 36, 72: 13, 76: 138, 80: 252, 84: 41, 92: 23, 96: 25, 100: 8, 104: 18, 108: 65, 112: 1, 116: 40, 120: 361, 124: 95}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 101, 105, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3897, 'token_per_expert': {1: 56, 5: 162, 9: 28, 13: 25, 17: 72, 21: 137, 25: 29, 29: 114, 33: 28, 37: 25, 41: 99, 45: 86, 49: 46, 53: 77, 57: 67, 61: 124, 65: 277, 69: 198, 73: 530, 77: 80, 81: 167, 85: 123, 89: 72, 93: 135, 101: 21, 105: 371, 113: 71, 117: 33, 121: 351, 125: 293}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 5310, 'token_per_expert': {2: 212, 6: 150, 10: 67, 14: 66, 18: 8, 22: 89, 26: 24, 34: 24, 38: 243, 42: 77, 46: 257, 50: 336, 54: 856, 58: 976, 62: 29, 66: 67, 70: 404, 74: 37, 82: 30, 86: 42, 90: 10, 98: 108, 102: 224, 106: 45, 110: 341, 114: 291, 118: 21, 122: 172, 126: 104}}
INFO 05-06 10:42:34.941084.941084 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 1.324ms | allocate_experts_across_cpu_gpu: 0.263ms
INFO 05-06 10:42:34.941055.941055 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.555152893066406e-05 seconds
INFO 05-06 10:42:34.943166.943166 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015995502471923828 seconds
INFO 05-06 10:42:34.970286.970286 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02676248550415039 seconds
INFO 05-06 10:42:34.972116.972116 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015735626220703125 seconds
INFO 05-06 10:42:34.976861.976861 mlpmodule.py:2799] [fused_experts] gmm total=4.099ms E=32 S=4240 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.976539.976539 mlpmodule.py:2799] [fused_experts] gmm total=4.239ms E=32 S=2937 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.977726.977726 mlpmodule.py:2799] [fused_experts] gmm total=4.287ms E=32 S=5310 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.977909.977909 mlpmodule.py:2799] [fused_experts] gmm total=5.109ms E=32 S=3897 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:34.979198.979198 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0067348480224609375 seconds
INFO 05-06 10:42:34.979792.979792 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.340576171875e-05 seconds
DEBUG 05-06 10:42:34.979401.979401 cuda_h.py:27] end *layer_moe_fused cost 39.534 ms
DEBUG 05-06 10:42:35.000616.000616 cuda_h.py:27] end prefill_layer cost 62.898 ms
DEBUG 05-06 10:42:35.000036.000036 lmp.py:841] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 10:42:35.000931.000931 lmp.py:824] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 10:42:35.002297.002297 cuda_h.py:27] end *sagl cost 1.606 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 4381, 'token_per_expert': {3: 94, 7: 113, 11: 6, 15: 90, 19: 126, 23: 210, 27: 136, 31: 4, 39: 159, 43: 437, 47: 5, 51: 133, 55: 15, 59: 6, 63: 3, 67: 32, 71: 176, 75: 410, 79: 26, 83: 144, 95: 1097, 99: 115, 103: 475, 107: 1, 111: 133, 115: 36, 119: 11, 123: 16, 127: 172}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3910, 'token_per_expert': {0: 14, 4: 187, 8: 29, 12: 714, 16: 427, 20: 29, 24: 140, 28: 18, 32: 175, 36: 200, 40: 241, 44: 36, 48: 258, 52: 32, 56: 315, 64: 15, 68: 110, 72: 151, 76: 203, 80: 57, 84: 4, 88: 111, 92: 221, 96: 11, 100: 14, 104: 26, 108: 2, 112: 14, 116: 42, 120: 22, 124: 92}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 69, 73, 77, 81, 89, 93, 97, 101, 105, 113, 117, 121, 125], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 3757, 'token_per_expert': {1: 205, 5: 67, 9: 106, 13: 163, 17: 93, 21: 171, 25: 2, 29: 16, 33: 14, 37: 112, 41: 21, 45: 111, 49: 1, 53: 2, 57: 186, 61: 168, 69: 315, 73: 52, 77: 28, 81: 326, 89: 179, 93: 461, 97: 60, 101: 593, 105: 31, 113: 50, 117: 51, 121: 15, 125: 158}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4336, 'token_per_expert': {2: 3, 6: 20, 10: 41, 14: 4, 18: 31, 22: 183, 26: 20, 30: 163, 34: 22, 38: 131, 42: 66, 46: 841, 50: 15, 54: 142, 58: 26, 62: 89, 66: 13, 70: 782, 74: 364, 82: 47, 86: 103, 90: 23, 94: 6, 98: 55, 102: 184, 106: 820, 110: 12, 114: 37, 118: 3, 122: 85, 126: 5}}
INFO 05-06 10:42:35.004906.004906 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.999ms | allocate_experts_across_cpu_gpu: 0.265ms
INFO 05-06 10:42:35.004360.004360 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.103515625e-05 seconds
INFO 05-06 10:42:35.006877.006877 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001688241958618164 seconds
INFO 05-06 10:42:35.021854.021854 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.014968156814575195 seconds
INFO 05-06 10:42:35.023416.023416 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016677379608154297 seconds
INFO 05-06 10:42:35.027823.027823 mlpmodule.py:2799] [fused_experts] gmm total=3.872ms E=32 S=4381 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.027202.027202 mlpmodule.py:2799] [fused_experts] gmm total=3.993ms E=32 S=3910 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.028411.028411 mlpmodule.py:2799] [fused_experts] gmm total=4.205ms E=32 S=3757 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.028982.028982 mlpmodule.py:2799] [fused_experts] gmm total=4.949ms E=32 S=4336 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.030839.030839 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006764411926269531 seconds
INFO 05-06 10:42:35.030995.030995 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 10:42:35.030047.030047 cuda_h.py:27] end *layer_moe_fused cost 27.635 ms
DEBUG 05-06 10:42:35.040798.040798 cuda_h.py:27] end prefill_layer cost 40.344 ms
DEBUG 05-06 10:42:35.040688.040688 lmp.py:841] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 10:42:35.040583.040583 lmp.py:824] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 10:42:35.042595.042595 cuda_h.py:27] end *sagl cost 1.523 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2597, 'token_per_expert': {3: 21, 7: 161, 11: 51, 15: 21, 19: 65, 23: 1, 27: 30, 31: 137, 35: 2, 39: 198, 43: 114, 47: 191, 51: 8, 55: 3, 59: 9, 63: 119, 67: 54, 71: 172, 75: 197, 79: 73, 83: 102, 87: 16, 91: 15, 99: 138, 103: 20, 107: 18, 111: 68, 115: 342, 119: 37, 123: 1, 127: 213}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 5060, 'token_per_expert': {0: 376, 4: 81, 8: 575, 12: 19, 16: 228, 20: 149, 24: 1, 28: 45, 32: 10, 36: 2, 40: 6, 44: 85, 48: 5, 52: 3, 56: 50, 60: 523, 64: 32, 68: 168, 72: 147, 76: 781, 80: 657, 84: 105, 88: 306, 92: 189, 100: 150, 104: 9, 108: 261, 112: 44, 120: 26, 124: 27}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4339, 'token_per_expert': {1: 817, 5: 70, 9: 71, 13: 176, 17: 3, 21: 282, 25: 20, 29: 41, 33: 27, 37: 147, 41: 225, 49: 256, 53: 14, 57: 237, 61: 36, 65: 4, 69: 74, 73: 28, 77: 8, 81: 658, 85: 152, 89: 64, 93: 47, 97: 54, 101: 11, 105: 89, 109: 18, 113: 204, 117: 64, 121: 78, 125: 364}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4388, 'token_per_expert': {2: 26, 6: 44, 10: 215, 14: 423, 18: 169, 22: 3, 26: 25, 30: 7, 34: 50, 38: 6, 42: 260, 46: 211, 50: 30, 54: 127, 58: 157, 62: 313, 66: 10, 70: 20, 74: 508, 78: 56, 82: 240, 86: 558, 90: 92, 94: 131, 98: 71, 102: 17, 106: 365, 110: 1, 114: 5, 118: 1, 122: 1, 126: 246}}
INFO 05-06 10:42:35.044771.044771 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 1.324ms | allocate_experts_across_cpu_gpu: 0.273ms
INFO 05-06 10:42:35.045338.045338 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.510185241699219e-05 seconds
INFO 05-06 10:42:35.059751.059751 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.014271259307861328 seconds
INFO 05-06 10:42:35.072289.072289 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013005733489990234 seconds
INFO 05-06 10:42:35.074903.074903 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016226768493652344 seconds
INFO 05-06 10:42:35.078976.078976 mlpmodule.py:2799] [fused_experts] gmm total=3.780ms E=32 S=2597 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.078731.078731 mlpmodule.py:2799] [fused_experts] gmm total=4.060ms E=32 S=5060 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.079268.079268 mlpmodule.py:2799] [fused_experts] gmm total=4.183ms E=32 S=4339 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.080698.080698 mlpmodule.py:2799] [fused_experts] gmm total=5.063ms E=32 S=4388 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.081156.081156 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0068051815032958984 seconds
INFO 05-06 10:42:35.081088.081088 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.269050598144531e-05 seconds
DEBUG 05-06 10:42:35.081720.081720 cuda_h.py:27] end *layer_moe_fused cost 38.254 ms
DEBUG 05-06 10:42:35.087457.087457 cuda_h.py:27] end prefill_layer cost 46.668 ms
DEBUG 05-06 10:42:35.087783.087783 lmp.py:841] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 10:42:35.087016.087016 lmp.py:824] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 10:42:35.090409.090409 cuda_h.py:27] end *sagl cost 2.397 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 59, 63, 67, 71, 79, 83, 87, 91, 99, 111, 115, 119, 123, 127], 'expert_count': 27, 'ideal_gpu_count': 30, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 4900, 'token_per_expert': {3: 23, 7: 356, 11: 43, 15: 1, 19: 258, 23: 338, 27: 78, 31: 219, 35: 10, 39: 32, 43: 102, 47: 21, 51: 71, 59: 68, 63: 13, 67: 354, 71: 70, 79: 599, 83: 527, 87: 807, 91: 97, 99: 138, 111: 310, 115: 17, 119: 289, 123: 53, 127: 6}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 29, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4617, 'token_per_expert': {0: 19, 4: 18, 8: 39, 12: 1, 16: 733, 20: 146, 24: 222, 28: 96, 32: 358, 36: 129, 40: 51, 44: 40, 48: 81, 52: 22, 56: 615, 64: 39, 68: 294, 72: 8, 76: 246, 80: 36, 84: 7, 88: 10, 92: 507, 96: 7, 100: 291, 104: 1, 108: 179, 112: 138, 116: 121, 120: 70, 124: 93}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 77, 81, 85, 89, 93, 97, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4026, 'token_per_expert': {1: 46, 5: 48, 9: 12, 13: 18, 17: 434, 21: 13, 25: 133, 29: 110, 33: 26, 37: 184, 41: 5, 45: 2, 49: 382, 53: 19, 57: 193, 61: 128, 65: 4, 69: 140, 77: 140, 81: 563, 85: 12, 89: 106, 93: 487, 97: 17, 105: 3, 109: 1, 113: 588, 117: 93, 121: 95, 125: 24}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 29, 'ideal_gpu_count': 29, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 2841, 'token_per_expert': {2: 138, 6: 434, 10: 144, 18: 69, 22: 31, 26: 3, 30: 191, 34: 20, 38: 147, 42: 55, 46: 76, 50: 33, 54: 72, 58: 16, 62: 51, 66: 137, 70: 103, 74: 23, 82: 99, 90: 1, 94: 12, 98: 90, 102: 733, 106: 10, 110: 13, 114: 5, 118: 27, 122: 21, 126: 87}}
INFO 05-06 10:42:35.094434.094434 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 3.153ms | allocate_experts_across_cpu_gpu: 0.252ms
INFO 05-06 10:42:35.094272.094272 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.437301635742188e-05 seconds
INFO 05-06 10:42:35.096661.096661 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016760826110839844 seconds
INFO 05-06 10:42:35.104171.104171 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008180856704711914 seconds
INFO 05-06 10:42:35.106994.106994 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015435218811035156 seconds
INFO 05-06 10:42:35.110519.110519 mlpmodule.py:2799] [fused_experts] gmm total=3.643ms E=32 S=4900 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.110834.110834 mlpmodule.py:2799] [fused_experts] gmm total=3.933ms E=32 S=4026 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.110291.110291 mlpmodule.py:2799] [fused_experts] gmm total=4.254ms E=32 S=4617 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.111111.111111 mlpmodule.py:2799] [fused_experts] gmm total=4.601ms E=32 S=2841 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.112318.112318 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00656437873840332 seconds
INFO 05-06 10:42:35.113573.113573 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.078315734863281e-05 seconds
DEBUG 05-06 10:42:35.113941.113941 cuda_h.py:27] end *layer_moe_fused cost 22.321 ms
DEBUG 05-06 10:42:35.118459.118459 cuda_h.py:27] end prefill_layer cost 30.621 ms
DEBUG 05-06 10:42:35.118441.118441 lmp.py:841] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 10:42:35.118382.118382 lmp.py:824] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 10:42:35.119076.119076 cuda_h.py:27] end *sagl cost 1.499 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 39, 47, 51, 59, 63, 67, 71, 79, 83, 87, 91, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 27, 'ideal_gpu_count': 28, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 3526, 'token_per_expert': {3: 169, 7: 2, 15: 483, 19: 225, 23: 235, 27: 1, 31: 20, 35: 117, 39: 753, 47: 33, 51: 2, 59: 4, 63: 34, 67: 16, 71: 595, 79: 52, 83: 4, 87: 3, 91: 168, 95: 197, 103: 62, 107: 56, 111: 28, 115: 175, 119: 40, 123: 41, 127: 11}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 24, 32, 36, 40, 48, 52, 56, 64, 68, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 27, 'ideal_gpu_count': 28, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 2745, 'token_per_expert': {0: 1, 4: 6, 8: 11, 12: 70, 20: 18, 24: 34, 32: 17, 36: 97, 40: 97, 48: 3, 52: 1, 56: 7, 64: 17, 68: 69, 76: 158, 80: 98, 84: 162, 88: 109, 92: 201, 96: 1, 100: 89, 104: 97, 108: 667, 112: 39, 116: 512, 120: 20, 124: 144}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 29, 'ideal_gpu_count': 27, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 4132, 'token_per_expert': {1: 44, 5: 214, 13: 18, 17: 32, 21: 712, 25: 199, 29: 2, 33: 14, 37: 13, 41: 14, 45: 590, 49: 377, 53: 781, 65: 36, 69: 1, 73: 191, 77: 105, 81: 10, 85: 101, 89: 35, 93: 5, 97: 289, 101: 103, 105: 18, 109: 1, 113: 25, 117: 163, 121: 1, 125: 38}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 30, 34, 38, 46, 50, 54, 58, 62, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 126], 'expert_count': 27, 'ideal_gpu_count': 27, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 5981, 'token_per_expert': {2: 7, 6: 210, 10: 3, 18: 9, 22: 101, 30: 1, 34: 113, 38: 47, 46: 359, 50: 636, 54: 7, 58: 113, 62: 6, 70: 52, 74: 455, 78: 1142, 82: 437, 86: 526, 90: 93, 94: 33, 98: 63, 102: 30, 106: 301, 110: 410, 114: 525, 118: 301, 126: 1}}
INFO 05-06 10:42:35.122084.122084 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 1.318ms | allocate_experts_across_cpu_gpu: 0.244ms
INFO 05-06 10:42:35.122539.122539 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.341934204101562e-05 seconds
INFO 05-06 10:42:35.124997.124997 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015521049499511719 seconds
INFO 05-06 10:42:35.133747.133747 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009027957916259766 seconds
INFO 05-06 10:42:35.134813.134813 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015273094177246094 seconds
INFO 05-06 10:42:35.138571.138571 mlpmodule.py:2799] [fused_experts] gmm total=3.593ms E=32 S=3526 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.138346.138346 mlpmodule.py:2799] [fused_experts] gmm total=3.711ms E=32 S=2745 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.139050.139050 mlpmodule.py:2799] [fused_experts] gmm total=3.845ms E=32 S=4132 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.140926.140926 mlpmodule.py:2799] [fused_experts] gmm total=4.997ms E=32 S=5981 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.141120.141120 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006906986236572266 seconds
INFO 05-06 10:42:35.141951.141951 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.078315734863281e-05 seconds
DEBUG 05-06 10:42:35.142568.142568 cuda_h.py:27] end *layer_moe_fused cost 21.549 ms
DEBUG 05-06 10:42:35.147825.147825 cuda_h.py:27] end prefill_layer cost 29.041 ms
DEBUG 05-06 10:42:35.147761.147761 lmp.py:841] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 10:42:35.147133.147133 lmp.py:824] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 10:42:35.149671.149671 cuda_h.py:27] end *sagl cost 1.593 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4942, 'token_per_expert': {3: 144, 7: 4, 11: 28, 15: 162, 19: 1, 23: 10, 27: 54, 31: 866, 39: 282, 43: 56, 47: 33, 51: 158, 55: 81, 59: 333, 63: 221, 67: 51, 71: 390, 75: 83, 79: 526, 83: 32, 87: 34, 91: 804, 95: 34, 99: 88, 103: 126, 107: 46, 111: 4, 115: 148, 119: 116, 123: 27}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 12, 16, 20, 28, 32, 36, 40, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 2806, 'token_per_expert': {0: 39, 8: 24, 12: 3, 16: 35, 20: 257, 28: 40, 32: 436, 36: 10, 40: 64, 48: 17, 52: 31, 56: 13, 60: 146, 64: 38, 68: 52, 72: 8, 76: 3, 80: 35, 84: 135, 92: 40, 96: 12, 100: 572, 104: 44, 108: 51, 112: 9, 116: 114, 120: 471, 124: 107}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 17, 21, 25, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 77, 81, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 4052, 'token_per_expert': {1: 285, 9: 37, 13: 94, 17: 529, 21: 203, 25: 231, 33: 230, 37: 534, 41: 107, 45: 10, 53: 4, 57: 18, 61: 13, 65: 18, 69: 132, 73: 63, 77: 1, 81: 469, 89: 5, 93: 41, 97: 3, 101: 46, 105: 15, 109: 24, 113: 128, 117: 65, 121: 544, 125: 203}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 38, 42, 46, 50, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 29, 'ideal_gpu_count': 28, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 4584, 'token_per_expert': {2: 53, 6: 287, 10: 9, 14: 400, 18: 2, 22: 152, 26: 32, 34: 125, 38: 180, 42: 58, 46: 36, 50: 2, 62: 20, 66: 7, 70: 37, 74: 10, 78: 518, 82: 83, 86: 142, 90: 22, 94: 28, 98: 202, 102: 240, 106: 24, 110: 627, 114: 881, 118: 157, 122: 74, 126: 176}}
INFO 05-06 10:42:35.151031.151031 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.926ms | allocate_experts_across_cpu_gpu: 0.248ms
INFO 05-06 10:42:35.151221.151221 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.843971252441406e-05 seconds
INFO 05-06 10:42:35.153714.153714 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016257762908935547 seconds
INFO 05-06 10:42:35.161721.161721 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008049726486206055 seconds
INFO 05-06 10:42:35.162828.162828 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015592575073242188 seconds
INFO 05-06 10:42:35.166286.166286 mlpmodule.py:2799] [fused_experts] gmm total=3.970ms E=32 S=4942 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.167446.167446 mlpmodule.py:2799] [fused_experts] gmm total=4.096ms E=32 S=2806 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.167546.167546 mlpmodule.py:2799] [fused_experts] gmm total=4.226ms E=32 S=4052 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.168778.168778 mlpmodule.py:2799] [fused_experts] gmm total=4.943ms E=32 S=4584 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.169912.169912 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006598711013793945 seconds
INFO 05-06 10:42:35.169406.169406 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.0067901611328125e-05 seconds
DEBUG 05-06 10:42:35.169888.169888 cuda_h.py:27] end *layer_moe_fused cost 19.646 ms
DEBUG 05-06 10:42:35.174844.174844 cuda_h.py:27] end prefill_layer cost 27.376 ms
DEBUG 05-06 10:42:35.175893.175893 lmp.py:841] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 10:42:35.175802.175802 lmp.py:824] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 10:42:35.177392.177392 cuda_h.py:27] end *sagl cost 1.973 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5745, 'token_per_expert': {3: 22, 7: 102, 11: 185, 15: 22, 19: 68, 23: 55, 27: 10, 31: 286, 35: 20, 39: 497, 43: 34, 47: 364, 51: 12, 59: 306, 63: 32, 67: 16, 71: 87, 75: 373, 79: 2, 83: 105, 87: 7, 91: 12, 95: 436, 99: 219, 103: 291, 107: 75, 111: 10, 115: 1069, 119: 514, 123: 340, 127: 174}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3255, 'token_per_expert': {0: 54, 4: 1, 8: 198, 12: 117, 16: 58, 24: 160, 28: 69, 32: 116, 36: 46, 40: 29, 44: 41, 48: 35, 52: 132, 56: 4, 60: 86, 64: 44, 68: 24, 72: 96, 76: 159, 80: 198, 84: 4, 88: 7, 92: 71, 96: 11, 100: 374, 104: 184, 108: 104, 112: 136, 116: 17, 120: 94, 124: 586}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 65, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3441, 'token_per_expert': {1: 8, 5: 41, 9: 19, 13: 126, 17: 11, 21: 16, 25: 58, 29: 9, 33: 11, 37: 14, 41: 9, 45: 68, 49: 1, 53: 131, 57: 115, 65: 448, 73: 21, 77: 16, 81: 80, 85: 13, 89: 187, 93: 17, 97: 509, 101: 26, 105: 95, 109: 34, 113: 227, 117: 264, 121: 740, 125: 127}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 3943, 'token_per_expert': {2: 196, 6: 9, 10: 127, 14: 8, 18: 13, 22: 16, 26: 512, 30: 107, 34: 116, 38: 180, 42: 188, 46: 5, 50: 390, 54: 13, 58: 23, 62: 136, 66: 465, 70: 50, 74: 122, 78: 20, 82: 1, 86: 537, 90: 58, 94: 1, 98: 52, 102: 72, 106: 10, 110: 75, 114: 171, 118: 30, 122: 203, 126: 37}}
INFO 05-06 10:42:35.179416.179416 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.785ms | allocate_experts_across_cpu_gpu: 0.358ms
INFO 05-06 10:42:35.179170.179170 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.511543273925781e-05 seconds
INFO 05-06 10:42:35.181460.181460 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016584396362304688 seconds
INFO 05-06 10:42:35.190078.190078 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008803606033325195 seconds
INFO 05-06 10:42:35.191197.191197 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001672983169555664 seconds
INFO 05-06 10:42:35.196568.196568 mlpmodule.py:2799] [fused_experts] gmm total=4.087ms E=32 S=3255 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.196264.196264 mlpmodule.py:2799] [fused_experts] gmm total=4.219ms E=32 S=3441 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.196045.196045 mlpmodule.py:2799] [fused_experts] gmm total=4.604ms E=32 S=5745 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.197723.197723 mlpmodule.py:2799] [fused_experts] gmm total=5.170ms E=32 S=3943 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.199863.199863 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00708317756652832 seconds
INFO 05-06 10:42:35.199171.199171 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.173683166503906e-05 seconds
DEBUG 05-06 10:42:35.199137.199137 cuda_h.py:27] end *layer_moe_fused cost 21.272 ms
DEBUG 05-06 10:42:35.204960.204960 cuda_h.py:27] end prefill_layer cost 29.041 ms
DEBUG 05-06 10:42:35.204816.204816 lmp.py:841] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 10:42:35.204010.204010 lmp.py:824] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 10:42:35.206388.206388 cuda_h.py:27] end *sagl cost 1.956 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4036, 'token_per_expert': {3: 12, 7: 228, 11: 14, 19: 35, 23: 239, 31: 94, 35: 13, 39: 251, 43: 64, 47: 113, 51: 234, 55: 113, 59: 92, 63: 97, 67: 16, 71: 238, 75: 326, 79: 18, 83: 497, 87: 10, 91: 626, 95: 129, 99: 135, 103: 125, 107: 54, 111: 31, 115: 52, 119: 84, 123: 11, 127: 85}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5482, 'token_per_expert': {0: 112, 4: 40, 8: 53, 12: 5, 16: 199, 20: 1, 24: 117, 28: 90, 32: 13, 36: 107, 40: 50, 44: 17, 48: 57, 52: 260, 56: 1, 60: 1, 64: 279, 68: 625, 72: 196, 76: 946, 80: 19, 84: 192, 88: 244, 96: 36, 100: 12, 104: 161, 108: 409, 112: 860, 116: 156, 120: 113, 124: 111}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3932, 'token_per_expert': {1: 74, 5: 102, 9: 287, 13: 57, 17: 65, 21: 139, 25: 22, 29: 88, 33: 65, 37: 233, 41: 68, 45: 18, 57: 16, 61: 5, 65: 519, 69: 132, 73: 101, 77: 40, 81: 197, 85: 136, 89: 5, 93: 134, 97: 171, 101: 255, 105: 34, 109: 596, 113: 80, 117: 34, 121: 49, 125: 210}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2934, 'token_per_expert': {2: 203, 6: 8, 10: 408, 14: 91, 18: 47, 22: 5, 26: 3, 30: 338, 34: 40, 38: 26, 42: 131, 46: 93, 50: 5, 54: 20, 58: 36, 62: 1, 66: 252, 70: 207, 74: 4, 78: 79, 82: 41, 86: 83, 90: 370, 94: 17, 98: 201, 102: 78, 106: 3, 110: 14, 114: 74, 118: 43, 126: 13}}
INFO 05-06 10:42:35.208920.208920 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.871ms | allocate_experts_across_cpu_gpu: 0.373ms
INFO 05-06 10:42:35.208120.208120 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.369850158691406e-05 seconds
INFO 05-06 10:42:35.210976.210976 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001760244369506836 seconds
INFO 05-06 10:42:35.219436.219436 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009040594100952148 seconds
INFO 05-06 10:42:35.221566.221566 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016274452209472656 seconds
INFO 05-06 10:42:35.225535.225535 mlpmodule.py:2799] [fused_experts] gmm total=3.781ms E=32 S=4036 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.226103.226103 mlpmodule.py:2799] [fused_experts] gmm total=3.932ms E=32 S=3932 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.226778.226778 mlpmodule.py:2799] [fused_experts] gmm total=4.599ms E=32 S=5482 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.227128.227128 mlpmodule.py:2799] [fused_experts] gmm total=4.970ms E=32 S=2934 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.228228.228228 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006966829299926758 seconds
INFO 05-06 10:42:35.228525.228525 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.222724914550781e-05 seconds
DEBUG 05-06 10:42:35.229472.229472 cuda_h.py:27] end *layer_moe_fused cost 21.735 ms
DEBUG 05-06 10:42:35.234118.234118 cuda_h.py:27] end prefill_layer cost 29.914 ms
DEBUG 05-06 10:42:35.234418.234418 lmp.py:841] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 10:42:35.234049.234049 lmp.py:824] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 10:42:35.236029.236029 cuda_h.py:27] end *sagl cost 2.323 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 32, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3941, 'token_per_expert': {3: 166, 7: 17, 11: 27, 15: 55, 19: 150, 23: 293, 27: 5, 31: 216, 35: 3, 43: 48, 51: 59, 55: 145, 59: 21, 63: 199, 67: 645, 71: 19, 75: 210, 79: 56, 83: 289, 87: 739, 91: 48, 95: 3, 99: 43, 103: 23, 107: 165, 111: 45, 115: 4, 119: 64, 123: 51, 127: 133}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 32, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 5473, 'token_per_expert': {0: 199, 4: 238, 8: 241, 12: 260, 16: 637, 20: 190, 24: 58, 28: 5, 32: 902, 36: 15, 40: 68, 44: 129, 48: 143, 52: 715, 56: 44, 60: 36, 64: 25, 68: 200, 72: 114, 76: 158, 80: 67, 84: 55, 88: 19, 92: 58, 96: 162, 100: 235, 104: 28, 108: 202, 112: 21, 116: 100, 120: 7, 124: 142}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 2684, 'token_per_expert': {1: 351, 5: 224, 9: 24, 13: 42, 17: 70, 21: 63, 25: 1, 29: 9, 33: 33, 37: 44, 41: 14, 45: 44, 49: 11, 53: 17, 57: 70, 61: 99, 65: 112, 69: 48, 73: 4, 77: 80, 81: 77, 85: 114, 89: 23, 93: 64, 97: 75, 101: 3, 105: 507, 109: 46, 113: 74, 117: 132, 121: 73, 125: 136}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4286, 'token_per_expert': {2: 138, 6: 34, 10: 60, 14: 116, 18: 72, 22: 108, 26: 145, 30: 48, 34: 33, 38: 35, 42: 184, 46: 1, 50: 16, 54: 197, 58: 129, 62: 42, 66: 270, 70: 117, 74: 12, 78: 176, 82: 121, 86: 476, 90: 114, 94: 4, 98: 22, 102: 98, 106: 4, 110: 161, 114: 182, 118: 84, 122: 18, 126: 1069}}
INFO 05-06 10:42:35.238794.238794 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.550ms | allocate_experts_across_cpu_gpu: 0.421ms
INFO 05-06 10:42:35.239324.239324 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.416175842285156e-05 seconds
INFO 05-06 10:42:35.240815.240815 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001569986343383789 seconds
INFO 05-06 10:42:35.249775.249775 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009014129638671875 seconds
INFO 05-06 10:42:35.251466.251466 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015666484832763672 seconds
INFO 05-06 10:42:35.255084.255084 mlpmodule.py:2799] [fused_experts] gmm total=3.596ms E=32 S=2684 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.255981.255981 mlpmodule.py:2799] [fused_experts] gmm total=4.058ms E=32 S=3941 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.256936.256936 mlpmodule.py:2799] [fused_experts] gmm total=4.208ms E=32 S=5473 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.256378.256378 mlpmodule.py:2799] [fused_experts] gmm total=4.938ms E=32 S=4286 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.258232.258232 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0066072940826416016 seconds
INFO 05-06 10:42:35.258203.258203 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.125999450683594e-05 seconds
DEBUG 05-06 10:42:35.258837.258837 cuda_h.py:27] end *layer_moe_fused cost 20.462 ms
DEBUG 05-06 10:42:35.263968.263968 cuda_h.py:27] end prefill_layer cost 29.428 ms
DEBUG 05-06 10:42:35.263096.263096 lmp.py:841] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 10:42:35.263230.263230 lmp.py:824] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 10:42:35.265055.265055 cuda_h.py:27] end *sagl cost 2.088 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4598, 'token_per_expert': {3: 84, 7: 10, 11: 22, 15: 13, 19: 37, 23: 631, 27: 325, 31: 148, 35: 110, 39: 321, 43: 306, 47: 170, 51: 13, 55: 101, 59: 52, 63: 188, 67: 48, 71: 214, 75: 556, 79: 7, 83: 40, 87: 43, 91: 47, 95: 505, 99: 61, 103: 131, 107: 224, 111: 74, 119: 61, 123: 45, 127: 11}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4539, 'token_per_expert': {0: 83, 4: 150, 8: 14, 12: 151, 16: 45, 20: 216, 24: 606, 28: 183, 32: 38, 36: 44, 40: 262, 44: 30, 48: 81, 52: 303, 56: 201, 60: 50, 64: 185, 68: 175, 72: 326, 76: 575, 80: 115, 84: 95, 88: 15, 92: 31, 96: 42, 100: 64, 104: 78, 108: 126, 112: 10, 116: 64, 120: 82, 124: 99}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 109, 113, 117, 125], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4057, 'token_per_expert': {1: 24, 5: 115, 9: 54, 13: 68, 17: 158, 21: 350, 25: 2, 29: 19, 33: 51, 37: 759, 41: 1, 45: 111, 49: 202, 53: 291, 57: 125, 61: 265, 65: 28, 69: 451, 73: 109, 77: 14, 81: 16, 85: 21, 89: 319, 93: 19, 97: 35, 101: 225, 109: 54, 113: 50, 117: 35, 125: 86}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 30, 34, 38, 42, 46, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3190, 'token_per_expert': {2: 27, 6: 117, 10: 215, 14: 45, 18: 207, 22: 155, 30: 20, 34: 24, 38: 52, 42: 15, 46: 1, 54: 140, 58: 335, 62: 31, 66: 37, 70: 112, 74: 514, 78: 116, 82: 18, 86: 353, 90: 28, 94: 72, 98: 100, 102: 19, 106: 153, 110: 4, 114: 77, 118: 54, 122: 88, 126: 61}}
INFO 05-06 10:42:35.270976.270976 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 3.399ms | allocate_experts_across_cpu_gpu: 0.256ms
INFO 05-06 10:42:35.270477.270477 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.5789947509765625e-05 seconds
INFO 05-06 10:42:35.272339.272339 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015673637390136719 seconds
INFO 05-06 10:42:35.281879.281879 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009464740753173828 seconds
INFO 05-06 10:42:35.283367.283367 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016334056854248047 seconds
INFO 05-06 10:42:35.287191.287191 mlpmodule.py:2799] [fused_experts] gmm total=3.843ms E=32 S=4598 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.288243.288243 mlpmodule.py:2799] [fused_experts] gmm total=4.116ms E=32 S=4539 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.288297.288297 mlpmodule.py:2799] [fused_experts] gmm total=4.237ms E=32 S=4057 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.288888.288888 mlpmodule.py:2799] [fused_experts] gmm total=4.782ms E=32 S=3190 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.290260.290260 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006546974182128906 seconds
INFO 05-06 10:42:35.290608.290608 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:42:35.290335.290335 cuda_h.py:27] end *layer_moe_fused cost 23.691 ms
DEBUG 05-06 10:42:35.295366.295366 cuda_h.py:27] end prefill_layer cost 32.147 ms
DEBUG 05-06 10:42:35.296415.296415 lmp.py:841] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 10:42:35.296701.296701 lmp.py:824] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 10:42:35.297824.297824 cuda_h.py:27] end *sagl cost 1.877 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 32, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4092, 'token_per_expert': {3: 271, 7: 56, 11: 24, 15: 56, 19: 30, 23: 53, 27: 34, 31: 233, 35: 55, 39: 41, 43: 236, 47: 102, 51: 54, 55: 15, 59: 26, 63: 15, 67: 46, 71: 99, 75: 137, 83: 378, 87: 199, 91: 76, 95: 140, 99: 541, 103: 80, 107: 61, 111: 371, 115: 3, 119: 345, 123: 162, 127: 153}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 32, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4410, 'token_per_expert': {0: 16, 4: 233, 8: 281, 12: 95, 16: 11, 20: 2, 24: 23, 28: 1, 32: 365, 36: 338, 40: 268, 44: 14, 48: 54, 52: 12, 56: 72, 60: 223, 64: 300, 68: 81, 72: 214, 76: 356, 80: 130, 84: 214, 88: 145, 92: 128, 96: 28, 100: 155, 104: 249, 108: 79, 112: 32, 116: 72, 120: 176, 124: 43}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 32, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4485, 'token_per_expert': {1: 145, 5: 112, 9: 35, 13: 71, 17: 176, 21: 20, 25: 27, 29: 74, 33: 277, 37: 119, 41: 17, 45: 43, 49: 149, 53: 308, 57: 111, 61: 219, 65: 228, 69: 140, 73: 68, 77: 286, 81: 203, 85: 414, 89: 33, 93: 176, 97: 72, 101: 280, 105: 1, 109: 53, 113: 8, 117: 5, 121: 506, 125: 109}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 3397, 'token_per_expert': {2: 390, 6: 5, 10: 113, 14: 239, 18: 23, 22: 6, 26: 64, 30: 84, 34: 145, 38: 174, 42: 25, 46: 69, 50: 376, 54: 360, 58: 256, 62: 33, 66: 36, 70: 41, 74: 60, 78: 171, 82: 34, 86: 4, 90: 46, 94: 32, 98: 54, 102: 13, 106: 3, 110: 218, 114: 34, 118: 238, 122: 44, 126: 7}}
INFO 05-06 10:42:35.300521.300521 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.973ms | allocate_experts_across_cpu_gpu: 0.344ms
INFO 05-06 10:42:35.300255.300255 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.43865966796875e-05 seconds
INFO 05-06 10:42:35.302749.302749 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016362667083740234 seconds
INFO 05-06 10:42:35.312912.312912 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009871244430541992 seconds
INFO 05-06 10:42:35.314814.314814 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017197132110595703 seconds
INFO 05-06 10:42:35.318360.318360 mlpmodule.py:2799] [fused_experts] gmm total=3.854ms E=32 S=4092 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.318036.318036 mlpmodule.py:2799] [fused_experts] gmm total=3.960ms E=32 S=4410 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.318070.318070 mlpmodule.py:2799] [fused_experts] gmm total=4.109ms E=32 S=4485 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.319910.319910 mlpmodule.py:2799] [fused_experts] gmm total=4.952ms E=32 S=3397 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.320324.320324 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006583213806152344 seconds
INFO 05-06 10:42:35.320103.320103 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.0067901611328125e-05 seconds
DEBUG 05-06 10:42:35.321399.321399 cuda_h.py:27] end *layer_moe_fused cost 22.018 ms
DEBUG 05-06 10:42:35.326058.326058 cuda_h.py:27] end prefill_layer cost 30.627 ms
DEBUG 05-06 10:42:35.326007.326007 lmp.py:841] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 10:42:35.326293.326293 lmp.py:824] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 10:42:35.328925.328925 cuda_h.py:27] end *sagl cost 1.832 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 3935, 'token_per_expert': {3: 350, 7: 327, 11: 88, 15: 124, 19: 110, 23: 235, 27: 187, 31: 115, 35: 142, 39: 148, 43: 26, 47: 97, 51: 493, 55: 35, 59: 56, 63: 189, 67: 21, 71: 5, 75: 194, 79: 213, 83: 88, 87: 18, 91: 2, 95: 13, 99: 81, 103: 49, 107: 3, 111: 66, 115: 11, 119: 141, 123: 283, 127: 25}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 5177, 'token_per_expert': {0: 100, 4: 39, 8: 13, 12: 68, 16: 172, 20: 50, 24: 393, 36: 86, 40: 239, 44: 580, 48: 132, 52: 929, 56: 47, 60: 89, 64: 496, 68: 46, 72: 83, 76: 185, 80: 230, 84: 78, 88: 145, 92: 552, 96: 169, 100: 15, 104: 104, 108: 52, 112: 51, 116: 5, 120: 24, 124: 5}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4178, 'token_per_expert': {1: 148, 5: 85, 9: 233, 13: 87, 17: 81, 21: 244, 25: 73, 29: 41, 33: 94, 37: 486, 41: 139, 45: 32, 49: 8, 53: 121, 57: 17, 61: 266, 65: 19, 69: 179, 73: 64, 77: 15, 81: 1, 85: 3, 89: 818, 93: 9, 97: 66, 101: 17, 105: 12, 109: 209, 113: 7, 117: 387, 121: 47, 125: 170}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3094, 'token_per_expert': {2: 119, 6: 17, 10: 117, 18: 34, 22: 66, 26: 170, 30: 22, 34: 7, 38: 679, 42: 38, 46: 14, 50: 220, 54: 27, 58: 57, 62: 14, 66: 16, 70: 17, 74: 12, 82: 4, 86: 43, 90: 107, 94: 25, 98: 75, 102: 197, 106: 140, 110: 38, 114: 27, 118: 74, 122: 685, 126: 33}}
INFO 05-06 10:42:35.330845.330845 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.583ms | allocate_experts_across_cpu_gpu: 0.330ms
INFO 05-06 10:42:35.330665.330665 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.937980651855469e-05 seconds
INFO 05-06 10:42:35.332513.332513 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016345977783203125 seconds
INFO 05-06 10:42:35.342764.342764 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009917736053466797 seconds
INFO 05-06 10:42:35.344797.344797 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016789436340332031 seconds
INFO 05-06 10:42:35.349576.349576 mlpmodule.py:2799] [fused_experts] gmm total=4.111ms E=32 S=3935 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.349080.349080 mlpmodule.py:2799] [fused_experts] gmm total=4.242ms E=32 S=5177 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.349636.349636 mlpmodule.py:2799] [fused_experts] gmm total=4.347ms E=32 S=4178 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.350117.350117 mlpmodule.py:2799] [fused_experts] gmm total=4.954ms E=32 S=3094 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.351851.351851 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006821632385253906 seconds
INFO 05-06 10:42:35.351067.351067 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.1021575927734375e-05 seconds
DEBUG 05-06 10:42:35.351754.351754 cuda_h.py:27] end *layer_moe_fused cost 22.128 ms
DEBUG 05-06 10:42:35.357788.357788 cuda_h.py:27] end prefill_layer cost 30.797 ms
DEBUG 05-06 10:42:35.357552.357552 lmp.py:841] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 10:42:35.357984.357984 lmp.py:824] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 10:42:35.359168.359168 cuda_h.py:27] end *sagl cost 1.921 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3031, 'token_per_expert': {3: 247, 7: 9, 11: 13, 15: 95, 19: 44, 23: 4, 27: 160, 31: 20, 35: 25, 43: 149, 47: 37, 51: 5, 55: 67, 59: 288, 63: 500, 67: 7, 71: 112, 75: 6, 79: 115, 83: 76, 87: 2, 91: 9, 95: 46, 99: 3, 103: 52, 107: 629, 111: 53, 119: 7, 123: 242, 127: 9}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4493, 'token_per_expert': {0: 49, 4: 211, 8: 239, 12: 75, 16: 12, 20: 86, 24: 28, 28: 221, 32: 172, 36: 34, 40: 278, 44: 182, 52: 92, 56: 254, 60: 75, 64: 99, 68: 937, 72: 107, 76: 81, 80: 49, 84: 68, 88: 213, 92: 210, 100: 139, 104: 29, 108: 182, 112: 159, 116: 156, 120: 49, 124: 7}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 5548, 'token_per_expert': {1: 67, 5: 206, 9: 174, 13: 214, 17: 11, 21: 314, 25: 14, 29: 9, 33: 160, 37: 312, 41: 132, 45: 580, 49: 666, 53: 171, 57: 330, 61: 24, 65: 368, 69: 19, 73: 338, 77: 282, 81: 212, 85: 162, 89: 17, 93: 76, 97: 7, 101: 37, 105: 62, 109: 174, 113: 120, 117: 38, 121: 83, 125: 169}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3312, 'token_per_expert': {2: 85, 6: 15, 10: 34, 14: 2, 18: 83, 22: 6, 26: 37, 30: 408, 34: 28, 38: 43, 42: 205, 46: 161, 50: 103, 54: 65, 58: 44, 62: 49, 66: 166, 70: 12, 74: 62, 82: 110, 86: 41, 90: 52, 94: 733, 98: 36, 102: 337, 106: 20, 110: 53, 114: 58, 118: 80, 122: 166, 126: 18}}
INFO 05-06 10:42:35.362617.362617 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.880ms | allocate_experts_across_cpu_gpu: 0.347ms
INFO 05-06 10:42:35.362729.362729 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.05718994140625e-05 seconds
INFO 05-06 10:42:35.363207.363207 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017654895782470703 seconds
INFO 05-06 10:42:35.374300.374300 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010140180587768555 seconds
INFO 05-06 10:42:35.376318.376318 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016548633575439453 seconds
INFO 05-06 10:42:35.380589.380589 mlpmodule.py:2799] [fused_experts] gmm total=3.772ms E=32 S=3031 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.380810.380810 mlpmodule.py:2799] [fused_experts] gmm total=3.939ms E=32 S=4493 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.381924.381924 mlpmodule.py:2799] [fused_experts] gmm total=4.488ms E=32 S=5548 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.381060.381060 mlpmodule.py:2799] [fused_experts] gmm total=5.089ms E=32 S=3312 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.383779.383779 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006910800933837891 seconds
INFO 05-06 10:42:35.383796.383796 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.0067901611328125e-05 seconds
DEBUG 05-06 10:42:35.383060.383060 cuda_h.py:27] end *layer_moe_fused cost 22.555 ms
DEBUG 05-06 10:42:35.388097.388097 cuda_h.py:27] end prefill_layer cost 31.253 ms
DEBUG 05-06 10:42:35.389814.389814 lmp.py:841] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 10:42:35.389723.389723 lmp.py:824] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 10:42:35.391168.391168 cuda_h.py:27] end *sagl cost 1.971 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 67, 71, 75, 79, 83, 87, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 2759, 'token_per_expert': {3: 55, 7: 192, 11: 468, 15: 5, 19: 3, 23: 14, 27: 2, 31: 144, 35: 124, 39: 3, 43: 45, 47: 11, 51: 185, 55: 60, 59: 51, 67: 174, 71: 45, 75: 130, 79: 101, 83: 230, 87: 85, 95: 104, 99: 3, 103: 181, 107: 7, 111: 95, 115: 24, 119: 53, 123: 72, 127: 93}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3852, 'token_per_expert': {0: 60, 4: 175, 8: 179, 12: 136, 16: 30, 20: 51, 24: 68, 28: 3, 32: 63, 36: 156, 40: 44, 44: 60, 48: 361, 52: 22, 56: 36, 60: 6, 64: 22, 68: 139, 72: 175, 76: 241, 80: 78, 84: 174, 88: 27, 92: 311, 96: 14, 100: 459, 104: 9, 112: 265, 116: 3, 120: 259, 124: 226}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 77, 81, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4718, 'token_per_expert': {1: 437, 5: 625, 9: 22, 13: 150, 17: 24, 21: 51, 25: 13, 29: 286, 33: 145, 37: 217, 41: 245, 45: 49, 53: 271, 57: 195, 61: 199, 65: 480, 69: 21, 73: 296, 77: 7, 81: 73, 89: 3, 93: 26, 97: 169, 101: 13, 105: 338, 109: 191, 113: 38, 117: 1, 121: 41, 125: 92}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5055, 'token_per_expert': {2: 134, 6: 628, 10: 151, 18: 333, 22: 12, 26: 357, 30: 170, 34: 112, 38: 94, 42: 72, 46: 328, 50: 57, 54: 25, 58: 106, 62: 224, 66: 2, 70: 103, 74: 36, 78: 765, 82: 95, 86: 154, 90: 189, 94: 17, 98: 7, 102: 141, 106: 58, 110: 191, 114: 32, 118: 82, 122: 334, 126: 46}}
INFO 05-06 10:42:35.393248.393248 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.476ms | allocate_experts_across_cpu_gpu: 0.355ms
INFO 05-06 10:42:35.393935.393935 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.2479248046875e-05 seconds
INFO 05-06 10:42:35.395016.395016 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001750946044921875 seconds
INFO 05-06 10:42:35.404611.404611 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00931549072265625 seconds
INFO 05-06 10:42:35.406809.406809 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016794204711914062 seconds
INFO 05-06 10:42:35.410533.410533 mlpmodule.py:2799] [fused_experts] gmm total=3.850ms E=32 S=2759 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.410625.410625 mlpmodule.py:2799] [fused_experts] gmm total=3.926ms E=32 S=3852 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.410031.410031 mlpmodule.py:2799] [fused_experts] gmm total=4.092ms E=32 S=4718 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.411872.411872 mlpmodule.py:2799] [fused_experts] gmm total=5.002ms E=32 S=5055 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.413855.413855 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0067691802978515625 seconds
INFO 05-06 10:42:35.413157.413157 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:42:35.413937.413937 cuda_h.py:27] end *layer_moe_fused cost 21.217 ms
DEBUG 05-06 10:42:35.419210.419210 cuda_h.py:27] end prefill_layer cost 30.547 ms
DEBUG 05-06 10:42:35.419967.419967 lmp.py:841] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 10:42:35.419776.419776 lmp.py:824] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 10:42:35.421919.421919 cuda_h.py:27] end *sagl cost 1.857 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 32, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4424, 'token_per_expert': {3: 4, 7: 112, 11: 109, 15: 142, 19: 101, 23: 10, 27: 27, 31: 97, 35: 571, 39: 23, 43: 172, 47: 83, 51: 40, 55: 283, 59: 464, 63: 37, 67: 16, 71: 8, 75: 214, 79: 36, 83: 43, 87: 16, 91: 7, 95: 9, 99: 117, 103: 433, 107: 99, 111: 153, 115: 157, 119: 358, 123: 208, 127: 275}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5811, 'token_per_expert': {0: 43, 4: 1, 8: 254, 12: 3, 16: 107, 20: 36, 24: 521, 28: 169, 32: 62, 36: 5, 40: 92, 44: 93, 48: 84, 56: 4, 60: 32, 64: 644, 68: 317, 72: 584, 76: 169, 80: 1, 84: 37, 88: 137, 92: 277, 96: 19, 100: 1204, 104: 4, 108: 227, 112: 64, 116: 260, 120: 251, 124: 110}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2394, 'token_per_expert': {1: 105, 5: 5, 9: 39, 13: 8, 17: 4, 21: 3, 25: 41, 29: 4, 33: 99, 37: 14, 41: 88, 45: 89, 49: 6, 53: 245, 57: 41, 61: 12, 65: 31, 69: 109, 73: 310, 77: 10, 81: 21, 85: 76, 89: 151, 93: 370, 97: 11, 101: 66, 105: 9, 109: 58, 113: 63, 117: 249, 125: 57}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3755, 'token_per_expert': {2: 16, 6: 12, 10: 30, 14: 31, 18: 3, 22: 6, 26: 101, 30: 170, 34: 43, 38: 230, 42: 92, 46: 187, 54: 29, 58: 100, 62: 59, 66: 153, 70: 174, 74: 532, 78: 7, 82: 205, 86: 230, 90: 174, 94: 207, 98: 40, 102: 117, 106: 15, 110: 41, 114: 6, 118: 175, 122: 25, 126: 545}}
INFO 05-06 10:42:35.423010.423010 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.939ms | allocate_experts_across_cpu_gpu: 0.295ms
INFO 05-06 10:42:35.424087.424087 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.8650970458984375e-05 seconds
INFO 05-06 10:42:35.425958.425958 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014460086822509766 seconds
INFO 05-06 10:42:35.434536.434536 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008510112762451172 seconds
INFO 05-06 10:42:35.435377.435377 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015044212341308594 seconds
INFO 05-06 10:42:35.439161.439161 mlpmodule.py:2799] [fused_experts] gmm total=4.007ms E=32 S=4424 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.440041.440041 mlpmodule.py:2799] [fused_experts] gmm total=4.005ms E=32 S=2394 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.440869.440869 mlpmodule.py:2799] [fused_experts] gmm total=4.499ms E=32 S=5811 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.441364.441364 mlpmodule.py:2799] [fused_experts] gmm total=4.860ms E=32 S=3755 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.442722.442722 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006902217864990234 seconds
INFO 05-06 10:42:35.442309.442309 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.078315734863281e-05 seconds
DEBUG 05-06 10:42:35.442572.442572 cuda_h.py:27] end *layer_moe_fused cost 20.226 ms
DEBUG 05-06 10:42:35.448191.448191 cuda_h.py:27] end prefill_layer cost 28.421 ms
DEBUG 05-06 10:42:35.448763.448763 lmp.py:841] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 10:42:35.448957.448957 lmp.py:824] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 10:42:35.450182.450182 cuda_h.py:27] end *sagl cost 2.337 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4014, 'token_per_expert': {3: 248, 7: 11, 11: 24, 15: 16, 19: 66, 23: 40, 27: 21, 31: 134, 35: 219, 39: 527, 43: 359, 47: 341, 51: 52, 55: 11, 59: 102, 67: 589, 71: 118, 75: 63, 79: 305, 83: 121, 87: 106, 91: 80, 95: 32, 99: 38, 103: 44, 107: 27, 111: 7, 115: 98, 119: 5, 123: 192, 127: 18}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3379, 'token_per_expert': {0: 15, 4: 5, 8: 132, 12: 58, 16: 190, 20: 18, 24: 117, 28: 30, 32: 59, 36: 40, 40: 73, 44: 278, 48: 44, 52: 48, 56: 669, 60: 30, 64: 20, 68: 29, 72: 146, 76: 122, 80: 141, 84: 181, 88: 10, 92: 10, 100: 128, 104: 146, 108: 283, 112: 86, 116: 136, 120: 73, 124: 62}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 117, 125], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 5446, 'token_per_expert': {1: 287, 5: 311, 9: 67, 13: 6, 17: 169, 21: 899, 25: 355, 29: 404, 33: 109, 37: 419, 41: 22, 49: 15, 53: 32, 57: 49, 61: 394, 65: 415, 69: 3, 73: 124, 77: 7, 81: 36, 85: 216, 89: 54, 93: 11, 97: 299, 101: 18, 105: 131, 109: 205, 117: 134, 125: 255}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3545, 'token_per_expert': {2: 158, 6: 84, 10: 60, 14: 18, 18: 205, 22: 92, 26: 174, 30: 133, 34: 108, 38: 80, 42: 67, 46: 368, 50: 8, 54: 22, 58: 32, 62: 38, 66: 32, 70: 2, 74: 19, 78: 196, 82: 14, 86: 607, 90: 175, 98: 316, 102: 30, 106: 143, 110: 21, 114: 2, 118: 281, 122: 56, 126: 4}}
INFO 05-06 10:42:35.455123.455123 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 3.155ms | allocate_experts_across_cpu_gpu: 0.255ms
INFO 05-06 10:42:35.455385.455385 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.650520324707031e-05 seconds
INFO 05-06 10:42:35.457271.457271 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016887187957763672 seconds
INFO 05-06 10:42:35.466673.466673 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008879661560058594 seconds
INFO 05-06 10:42:35.467685.467685 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016396045684814453 seconds
INFO 05-06 10:42:35.472182.472182 mlpmodule.py:2799] [fused_experts] gmm total=4.182ms E=32 S=3379 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.472290.472290 mlpmodule.py:2799] [fused_experts] gmm total=4.312ms E=32 S=5446 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.472021.472021 mlpmodule.py:2799] [fused_experts] gmm total=4.822ms E=32 S=4014 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.473625.473625 mlpmodule.py:2799] [fused_experts] gmm total=5.137ms E=32 S=3545 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.474910.474910 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006876230239868164 seconds
INFO 05-06 10:42:35.474241.474241 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 7.224082946777344e-05 seconds
DEBUG 05-06 10:42:35.475627.475627 cuda_h.py:27] end *layer_moe_fused cost 23.407 ms
DEBUG 05-06 10:42:35.480151.480151 cuda_h.py:27] end prefill_layer cost 32.507 ms
DEBUG 05-06 10:42:35.480359.480359 lmp.py:841] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 10:42:35.480566.480566 lmp.py:824] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 10:42:35.483246.483246 cuda_h.py:27] end *sagl cost 2.450 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3791, 'token_per_expert': {3: 6, 7: 114, 11: 400, 15: 14, 19: 244, 23: 229, 27: 443, 31: 56, 35: 172, 39: 1, 43: 84, 47: 40, 51: 4, 55: 17, 59: 14, 63: 435, 67: 211, 71: 357, 75: 68, 79: 69, 83: 162, 87: 31, 91: 323, 95: 8, 99: 17, 107: 44, 111: 46, 115: 27, 119: 37, 123: 6, 127: 112}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3742, 'token_per_expert': {4: 188, 8: 58, 12: 304, 16: 213, 20: 91, 24: 10, 28: 47, 32: 69, 36: 133, 40: 70, 44: 309, 48: 175, 52: 389, 56: 290, 60: 151, 64: 626, 68: 39, 76: 25, 80: 12, 84: 8, 88: 1, 92: 52, 96: 42, 100: 100, 104: 34, 108: 190, 112: 22, 116: 14, 120: 49, 124: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4229, 'token_per_expert': {1: 137, 5: 135, 9: 64, 13: 102, 17: 122, 21: 6, 25: 7, 29: 161, 33: 376, 37: 147, 45: 206, 49: 74, 53: 51, 57: 63, 61: 16, 65: 30, 69: 2, 73: 292, 77: 261, 81: 74, 85: 1, 89: 12, 93: 8, 97: 616, 101: 5, 105: 53, 109: 156, 113: 4, 117: 2, 121: 1045, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4622, 'token_per_expert': {2: 17, 6: 386, 10: 15, 18: 30, 22: 5, 26: 49, 30: 143, 34: 374, 38: 51, 42: 66, 46: 36, 50: 160, 54: 3, 58: 14, 62: 49, 66: 52, 70: 510, 74: 127, 78: 10, 82: 120, 86: 135, 90: 928, 94: 263, 98: 205, 102: 28, 106: 70, 110: 213, 114: 267, 118: 129, 122: 152, 126: 15}}
INFO 05-06 10:42:35.485764.485764 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.539ms | allocate_experts_across_cpu_gpu: 0.255ms
INFO 05-06 10:42:35.485880.485880 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.459785461425781e-05 seconds
INFO 05-06 10:42:35.487807.487807 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015256404876708984 seconds
INFO 05-06 10:42:35.496742.496742 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008672714233398438 seconds
INFO 05-06 10:42:35.497920.497920 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001508474349975586 seconds
INFO 05-06 10:42:35.501951.501951 mlpmodule.py:2799] [fused_experts] gmm total=3.750ms E=32 S=3742 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.502680.502680 mlpmodule.py:2799] [fused_experts] gmm total=4.061ms E=32 S=4229 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.503570.503570 mlpmodule.py:2799] [fused_experts] gmm total=5.043ms E=32 S=4622 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.503003.503003 mlpmodule.py:2799] [fused_experts] gmm total=5.631ms E=32 S=3791 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.504629.504629 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0070648193359375 seconds
INFO 05-06 10:42:35.504891.504891 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.1975250244140625e-05 seconds
DEBUG 05-06 10:42:35.505016.505016 cuda_h.py:27] end *layer_moe_fused cost 20.192 ms
DEBUG 05-06 10:42:35.510468.510468 cuda_h.py:27] end prefill_layer cost 29.099 ms
DEBUG 05-06 10:42:35.510086.510086 lmp.py:841] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 10:42:35.510803.510803 lmp.py:824] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 10:42:35.512445.512445 cuda_h.py:27] end *sagl cost 1.943 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3874, 'token_per_expert': {3: 255, 7: 136, 11: 117, 15: 8, 19: 87, 23: 20, 27: 25, 31: 31, 35: 433, 39: 99, 43: 53, 47: 58, 51: 52, 55: 21, 63: 143, 67: 175, 71: 152, 75: 6, 79: 108, 83: 245, 87: 92, 91: 168, 95: 4, 99: 44, 103: 34, 107: 648, 111: 143, 115: 8, 119: 28, 123: 456, 127: 25}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4867, 'token_per_expert': {0: 89, 4: 14, 8: 57, 12: 32, 16: 1233, 24: 23, 32: 25, 36: 153, 40: 1, 44: 155, 48: 66, 52: 360, 56: 108, 60: 346, 64: 242, 68: 728, 72: 60, 76: 28, 80: 280, 84: 9, 88: 93, 92: 27, 96: 2, 100: 142, 104: 253, 108: 30, 112: 43, 116: 114, 120: 115, 124: 39}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2871, 'token_per_expert': {1: 14, 5: 62, 9: 69, 13: 40, 17: 18, 21: 29, 25: 78, 29: 69, 33: 31, 37: 1, 41: 127, 45: 424, 49: 125, 53: 15, 57: 6, 61: 13, 65: 8, 69: 290, 73: 38, 77: 36, 81: 8, 85: 444, 89: 162, 93: 125, 97: 101, 101: 13, 109: 80, 113: 3, 117: 393, 121: 12, 125: 37}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4772, 'token_per_expert': {2: 451, 6: 49, 10: 158, 14: 145, 18: 431, 22: 22, 26: 71, 30: 3, 34: 152, 38: 51, 42: 24, 46: 28, 50: 127, 54: 1, 58: 1034, 62: 2, 66: 26, 70: 338, 74: 36, 78: 65, 82: 153, 86: 3, 90: 230, 94: 4, 98: 1, 102: 6, 106: 108, 110: 689, 114: 215, 118: 61, 122: 24, 126: 64}}
INFO 05-06 10:42:35.514543.514543 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.461ms | allocate_experts_across_cpu_gpu: 0.347ms
INFO 05-06 10:42:35.514992.514992 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.891654968261719e-05 seconds
INFO 05-06 10:42:35.516240.516240 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015764236450195312 seconds
INFO 05-06 10:42:35.524862.524862 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008611679077148438 seconds
INFO 05-06 10:42:35.526391.526391 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00151824951171875 seconds
INFO 05-06 10:42:35.530918.530918 mlpmodule.py:2799] [fused_experts] gmm total=3.926ms E=32 S=4867 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.530474.530474 mlpmodule.py:2799] [fused_experts] gmm total=4.218ms E=32 S=3874 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.531071.531071 mlpmodule.py:2799] [fused_experts] gmm total=4.256ms E=32 S=2871 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.531887.531887 mlpmodule.py:2799] [fused_experts] gmm total=4.983ms E=32 S=4772 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.532951.532951 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006563663482666016 seconds
INFO 05-06 10:42:35.533684.533684 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.054473876953125e-05 seconds
DEBUG 05-06 10:42:35.533954.533954 cuda_h.py:27] end *layer_moe_fused cost 20.006 ms
DEBUG 05-06 10:42:35.539756.539756 cuda_h.py:27] end prefill_layer cost 28.854 ms
DEBUG 05-06 10:42:35.539043.539043 lmp.py:841] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 10:42:35.539528.539528 lmp.py:824] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 10:42:35.541270.541270 cuda_h.py:27] end *sagl cost 1.944 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4317, 'token_per_expert': {3: 138, 7: 3, 11: 10, 15: 210, 19: 183, 23: 24, 27: 342, 31: 50, 35: 50, 43: 352, 47: 21, 51: 139, 55: 6, 59: 150, 63: 36, 67: 57, 71: 30, 75: 80, 79: 115, 83: 40, 87: 640, 91: 44, 95: 497, 99: 102, 103: 123, 107: 12, 111: 631, 115: 47, 119: 2, 123: 176, 127: 7}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3580, 'token_per_expert': {0: 32, 4: 20, 8: 36, 12: 1, 16: 5, 20: 723, 24: 462, 28: 16, 32: 9, 36: 83, 40: 27, 44: 30, 48: 9, 52: 189, 56: 194, 60: 177, 64: 1, 68: 75, 72: 61, 76: 169, 80: 29, 84: 498, 88: 81, 92: 21, 96: 66, 100: 8, 104: 267, 108: 68, 112: 81, 116: 34, 124: 108}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5645, 'token_per_expert': {1: 48, 5: 34, 9: 10, 13: 43, 17: 593, 21: 2, 25: 81, 29: 91, 33: 6, 37: 69, 41: 72, 45: 99, 49: 214, 53: 8, 57: 96, 61: 84, 65: 494, 73: 276, 77: 71, 81: 79, 85: 1198, 89: 772, 93: 5, 97: 127, 101: 2, 105: 169, 109: 60, 113: 797, 117: 12, 121: 5, 125: 28}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 2842, 'token_per_expert': {2: 33, 6: 3, 10: 101, 14: 157, 18: 37, 22: 6, 26: 105, 30: 108, 34: 16, 38: 79, 42: 54, 46: 12, 50: 158, 54: 13, 58: 8, 62: 1, 66: 109, 70: 225, 74: 24, 78: 190, 82: 6, 86: 180, 90: 146, 98: 2, 102: 134, 110: 32, 114: 653, 118: 73, 122: 38, 126: 139}}
INFO 05-06 10:42:35.543836.543836 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.819ms | allocate_experts_across_cpu_gpu: 0.340ms
INFO 05-06 10:42:35.543225.543225 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.985664367675781e-05 seconds
INFO 05-06 10:42:35.545871.545871 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001621246337890625 seconds
INFO 05-06 10:42:35.554972.554972 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008883476257324219 seconds
INFO 05-06 10:42:35.555025.555025 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015196800231933594 seconds
INFO 05-06 10:42:35.559206.559206 mlpmodule.py:2799] [fused_experts] gmm total=3.809ms E=32 S=4317 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.560834.560834 mlpmodule.py:2799] [fused_experts] gmm total=4.069ms E=32 S=3580 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.560241.560241 mlpmodule.py:2799] [fused_experts] gmm total=4.456ms E=32 S=5645 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.561536.561536 mlpmodule.py:2799] [fused_experts] gmm total=4.856ms E=32 S=2842 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.562006.562006 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0067005157470703125 seconds
INFO 05-06 10:42:35.562111.562111 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.127357482910156e-05 seconds
DEBUG 05-06 10:42:35.562217.562217 cuda_h.py:27] end *layer_moe_fused cost 20.648 ms
DEBUG 05-06 10:42:35.568881.568881 cuda_h.py:27] end prefill_layer cost 28.938 ms
DEBUG 05-06 10:42:35.568659.568659 lmp.py:841] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 10:42:35.568620.568620 lmp.py:824] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 10:42:35.570807.570807 cuda_h.py:27] end *sagl cost 2.163 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 5295, 'token_per_expert': {3: 187, 7: 172, 11: 27, 15: 26, 19: 43, 23: 86, 27: 33, 31: 243, 35: 194, 39: 42, 43: 547, 47: 18, 51: 261, 55: 56, 59: 23, 63: 26, 67: 6, 71: 51, 75: 96, 79: 281, 83: 177, 87: 682, 91: 30, 95: 380, 99: 2, 103: 403, 107: 3, 111: 249, 115: 377, 119: 199, 123: 273, 127: 102}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3934, 'token_per_expert': {0: 38, 4: 80, 8: 162, 12: 183, 16: 22, 20: 85, 24: 398, 28: 116, 32: 30, 36: 250, 40: 58, 44: 9, 48: 310, 56: 101, 60: 36, 64: 197, 68: 25, 72: 3, 76: 279, 80: 35, 84: 16, 88: 398, 92: 2, 96: 21, 100: 397, 104: 4, 108: 125, 112: 73, 116: 27, 120: 433, 124: 21}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3560, 'token_per_expert': {1: 202, 5: 1, 9: 4, 13: 251, 17: 3, 21: 46, 25: 257, 29: 23, 33: 372, 37: 276, 41: 189, 45: 511, 49: 79, 53: 144, 57: 9, 61: 172, 65: 395, 69: 11, 77: 6, 81: 6, 85: 88, 89: 2, 93: 12, 97: 9, 101: 3, 105: 96, 109: 161, 113: 41, 121: 127, 125: 64}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3595, 'token_per_expert': {2: 15, 6: 16, 10: 38, 14: 141, 18: 173, 22: 8, 26: 61, 30: 9, 34: 8, 42: 166, 46: 254, 50: 563, 54: 86, 58: 42, 62: 288, 66: 119, 70: 204, 74: 43, 78: 275, 82: 294, 86: 17, 90: 119, 94: 78, 98: 140, 106: 101, 110: 17, 114: 105, 118: 105, 122: 93, 126: 17}}
INFO 05-06 10:42:35.572860.572860 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.515ms | allocate_experts_across_cpu_gpu: 0.390ms
INFO 05-06 10:42:35.572323.572323 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.987022399902344e-05 seconds
INFO 05-06 10:42:35.574903.574903 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018482208251953125 seconds
INFO 05-06 10:42:35.583202.583202 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008460521697998047 seconds
INFO 05-06 10:42:35.584190.584190 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015423297882080078 seconds
INFO 05-06 10:42:35.588076.588076 mlpmodule.py:2799] [fused_experts] gmm total=3.317ms E=32 S=5295 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.589675.589675 mlpmodule.py:2799] [fused_experts] gmm total=3.943ms E=32 S=3560 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.589411.589411 mlpmodule.py:2799] [fused_experts] gmm total=4.292ms E=32 S=3934 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.590007.590007 mlpmodule.py:2799] [fused_experts] gmm total=4.867ms E=32 S=3595 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.591378.591378 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0065233707427978516 seconds
INFO 05-06 10:42:35.591639.591639 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00012254714965820312 seconds
DEBUG 05-06 10:42:35.592552.592552 cuda_h.py:27] end *layer_moe_fused cost 20.753 ms
DEBUG 05-06 10:42:35.596853.596853 cuda_h.py:27] end prefill_layer cost 28.274 ms
DEBUG 05-06 10:42:35.596096.596096 lmp.py:841] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 10:42:35.596649.596649 lmp.py:824] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 10:42:35.601249.601249 cuda_h.py:27] end *sagl cost 3.990 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 29, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4693, 'token_per_expert': {3: 15, 7: 5, 11: 249, 15: 13, 19: 8, 23: 59, 27: 3, 31: 2, 35: 7, 39: 35, 43: 34, 47: 297, 51: 15, 55: 135, 59: 8, 63: 1, 67: 13, 71: 278, 75: 318, 79: 49, 83: 5, 87: 12, 91: 509, 95: 73, 99: 5, 103: 4, 111: 1540, 115: 554, 119: 254, 123: 138, 127: 55}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 26, 'ideal_gpu_count': 29, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 5565, 'token_per_expert': {8: 21, 12: 1392, 20: 1176, 24: 92, 28: 2, 32: 196, 36: 24, 40: 363, 44: 48, 48: 38, 52: 285, 56: 9, 60: 55, 68: 236, 72: 18, 76: 667, 80: 2, 84: 122, 88: 79, 92: 28, 100: 83, 104: 142, 108: 9, 112: 465, 120: 9, 124: 4}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 37, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121], 'expert_count': 29, 'ideal_gpu_count': 28, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 3826, 'token_per_expert': {1: 88, 5: 105, 9: 66, 13: 148, 17: 8, 21: 11, 29: 4, 33: 52, 37: 112, 45: 9, 49: 1132, 53: 289, 57: 545, 61: 4, 65: 62, 69: 81, 73: 21, 77: 186, 81: 20, 85: 93, 89: 144, 93: 6, 97: 80, 101: 152, 105: 75, 109: 24, 113: 156, 117: 66, 121: 87}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 28, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 2300, 'token_per_expert': {6: 12, 18: 113, 22: 171, 26: 4, 30: 108, 34: 4, 38: 3, 42: 1, 46: 250, 50: 16, 54: 10, 58: 14, 62: 78, 66: 5, 70: 202, 74: 55, 78: 230, 82: 14, 90: 388, 94: 50, 98: 53, 102: 2, 106: 133, 110: 254, 114: 7, 118: 5, 122: 43, 126: 75}}
INFO 05-06 10:42:35.605282.605282 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.937ms | allocate_experts_across_cpu_gpu: 0.725ms
INFO 05-06 10:42:35.605338.605338 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 0.00013685226440429688 seconds
INFO 05-06 10:42:35.607936.607936 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001798868179321289 seconds
INFO 05-06 10:42:35.616965.616965 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008827686309814453 seconds
INFO 05-06 10:42:35.618623.618623 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017611980438232422 seconds
INFO 05-06 10:42:35.620846.620846 mlpmodule.py:2799] [fused_experts] gmm total=1.945ms E=32 S=4693 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.622219.622219 mlpmodule.py:2799] [fused_experts] gmm total=3.675ms E=32 S=5565 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.622578.622578 mlpmodule.py:2799] [fused_experts] gmm total=3.820ms E=32 S=3826 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.624576.624576 mlpmodule.py:2799] [fused_experts] gmm total=5.037ms E=32 S=2300 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.625827.625827 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006532192230224609 seconds
INFO 05-06 10:42:35.625182.625182 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:42:35.625321.625321 cuda_h.py:27] end *layer_moe_fused cost 22.212 ms
DEBUG 05-06 10:42:35.630326.630326 cuda_h.py:27] end prefill_layer cost 33.468 ms
DEBUG 05-06 10:42:35.630328.630328 lmp.py:841] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 10:42:35.630521.630521 lmp.py:824] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 10:42:35.633706.633706 cuda_h.py:27] end *sagl cost 2.691 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 51, 55, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 31, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 4597, 'token_per_expert': {3: 68, 7: 696, 11: 71, 15: 95, 19: 437, 23: 246, 27: 254, 31: 51, 35: 48, 39: 11, 43: 440, 51: 4, 55: 13, 63: 42, 67: 137, 71: 229, 75: 54, 79: 3, 83: 43, 87: 22, 91: 681, 95: 102, 99: 462, 107: 112, 111: 15, 115: 88, 119: 31, 123: 134, 127: 8}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4260, 'token_per_expert': {0: 6, 4: 449, 8: 30, 12: 4, 16: 204, 20: 407, 24: 59, 28: 347, 32: 66, 36: 2, 40: 17, 44: 72, 48: 162, 52: 554, 56: 282, 60: 211, 64: 584, 68: 4, 72: 9, 76: 29, 80: 77, 84: 49, 88: 42, 92: 73, 96: 46, 100: 5, 104: 1, 108: 71, 112: 7, 116: 88, 120: 101, 124: 202}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 3623, 'token_per_expert': {1: 75, 5: 10, 9: 101, 13: 52, 17: 7, 21: 63, 25: 66, 29: 196, 33: 23, 37: 17, 41: 1, 45: 2, 49: 194, 53: 139, 57: 310, 61: 222, 65: 17, 69: 115, 73: 112, 77: 135, 81: 177, 85: 110, 89: 104, 93: 153, 97: 236, 101: 95, 105: 50, 109: 100, 113: 147, 117: 281, 121: 281, 125: 32}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3904, 'token_per_expert': {2: 96, 6: 32, 10: 63, 14: 189, 18: 255, 22: 317, 26: 255, 30: 172, 34: 12, 38: 10, 42: 359, 46: 23, 50: 28, 54: 146, 58: 33, 62: 218, 66: 44, 70: 8, 74: 8, 78: 91, 82: 182, 86: 471, 90: 220, 94: 23, 98: 20, 102: 15, 106: 543, 114: 51, 118: 7, 122: 5, 126: 8}}
INFO 05-06 10:42:35.637622.637622 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 2.731ms | allocate_experts_across_cpu_gpu: 0.358ms
INFO 05-06 10:42:35.637462.637462 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.702278137207031e-05 seconds
INFO 05-06 10:42:35.638210.638210 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015342235565185547 seconds
INFO 05-06 10:42:35.647544.647544 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008336067199707031 seconds
INFO 05-06 10:42:35.649453.649453 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015878677368164062 seconds
INFO 05-06 10:42:35.652054.652054 mlpmodule.py:2799] [fused_experts] gmm total=3.693ms E=32 S=4597 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.653749.653749 mlpmodule.py:2799] [fused_experts] gmm total=4.004ms E=32 S=4260 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.653173.653173 mlpmodule.py:2799] [fused_experts] gmm total=4.114ms E=32 S=3623 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.654733.654733 mlpmodule.py:2799] [fused_experts] gmm total=4.693ms E=32 S=3904 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.655903.655903 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006235599517822266 seconds
INFO 05-06 10:42:35.655443.655443 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:42:35.655456.655456 cuda_h.py:27] end *layer_moe_fused cost 21.475 ms
DEBUG 05-06 10:42:35.661233.661233 cuda_h.py:27] end prefill_layer cost 30.748 ms
DEBUG 05-06 10:42:35.661275.661275 lmp.py:841] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 10:42:35.661323.661323 cuda_h.py:27] end prefill_step cost 1692.751 ms
INFO 05-06 10:42:35.661887.661887 lmp.py:843] prefill time: 1.799511432647705 seconds
DEBUG 05-06 10:42:35.798645.798645 cuda_h.py:27] end init_inputs_tokens cost 106.175 ms
DEBUG 05-06 10:42:35.798822.798822 lmp.py:899] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:42:35.798843.798843 lmp.py:904] ---- decode step 0 layer 0 ----
DEBUG 05-06 10:42:35.805631.805631 cuda_h.py:27] end *sagl cost 6.467 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 47, 55, 63, 79, 83, 87, 103, 123, 127], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 16, 'token_per_expert': {15: 2, 47: 1, 55: 1, 63: 2, 79: 2, 83: 2, 87: 2, 103: 1, 123: 1, 127: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 32, 48, 60, 116], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {8: 2, 32: 1, 48: 1, 60: 2, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [33, 45, 53], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {33: 1, 45: 2, 53: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 90, 114], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {22: 2, 26: 1, 90: 1, 114: 1}}
INFO 05-06 10:42:35.807712.807712 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.581ms | allocate_experts_across_cpu_gpu: 0.126ms
INFO 05-06 10:42:35.807656.807656 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3365020751953125e-05 seconds
INFO 05-06 10:42:35.809451.809451 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0019278526306152344 seconds
INFO 05-06 10:42:35.812951.812951 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0027015209197998047 seconds
INFO 05-06 10:42:35.814684.814684 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016360282897949219 seconds
INFO 05-06 10:42:35.817148.817148 mlpmodule.py:2799] [fused_experts] gmm total=2.442ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.817493.817493 mlpmodule.py:2799] [fused_experts] gmm total=2.772ms E=32 S=16 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.817607.817607 mlpmodule.py:2799] [fused_experts] gmm total=2.782ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.817301.817301 mlpmodule.py:2799] [fused_experts] gmm total=2.833ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.819321.819321 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0048100948333740234 seconds
INFO 05-06 10:42:35.819663.819663 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.745887756347656e-05 seconds
DEBUG 05-06 10:42:35.819736.819736 cuda_h.py:27] end *layer_moe_fused cost 12.703 ms
DEBUG 05-06 10:42:35.820498.820498 cuda_h.py:27] end decode_layer cost 21.122 ms
DEBUG 05-06 10:42:35.820348.820348 lmp.py:904] ---- decode step 0 layer 1 ----
DEBUG 05-06 10:42:35.822050.822050 cuda_h.py:27] end *sagl cost 1.948 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [83, 107, 119, 123], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {83: 1, 107: 2, 119: 2, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 56, 92, 96, 124], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 13, 'token_per_expert': {0: 3, 8: 2, 56: 3, 92: 2, 96: 1, 124: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 13, 65, 73, 121], 'expert_count': 5, 'ideal_gpu_count': 4, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {9: 2, 13: 1, 65: 1, 73: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [30, 54, 110], 'expert_count': 3, 'ideal_gpu_count': 4, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {30: 2, 54: 2, 110: 2}}
INFO 05-06 10:42:35.823292.823292 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.382ms | allocate_experts_across_cpu_gpu: 0.114ms
INFO 05-06 10:42:35.823567.823567 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0742416381835938e-05 seconds
INFO 05-06 10:42:35.825554.825554 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013539791107177734 seconds
INFO 05-06 10:42:35.826734.826734 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013699531555175781 seconds
INFO 05-06 10:42:35.829821.829821 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.003119945526123047 seconds
INFO 05-06 10:42:35.832629.832629 mlpmodule.py:2799] [fused_experts] gmm total=2.169ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.832720.832720 mlpmodule.py:2799] [fused_experts] gmm total=2.253ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.832304.832304 mlpmodule.py:2799] [fused_experts] gmm total=2.394ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.832594.832594 mlpmodule.py:2799] [fused_experts] gmm total=2.420ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.834309.834309 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004227638244628906 seconds
INFO 05-06 10:42:35.834419.834419 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:42:35.834498.834498 cuda_h.py:27] end *layer_moe_fused cost 11.186 ms
DEBUG 05-06 10:42:35.834792.834792 cuda_h.py:27] end decode_layer cost 14.860 ms
DEBUG 05-06 10:42:35.835490.835490 lmp.py:904] ---- decode step 0 layer 2 ----
DEBUG 05-06 10:42:35.836028.836028 cuda_h.py:27] end *sagl cost 1.585 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 91], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 5, 'token_per_expert': {11: 3, 91: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 76, 120], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {8: 1, 12: 2, 76: 4, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [41, 45, 49, 61, 81, 97], 'expert_count': 6, 'ideal_gpu_count': 4, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {41: 2, 45: 1, 49: 1, 61: 1, 81: 3, 97: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [62, 70, 90, 102, 106, 126], 'expert_count': 6, 'ideal_gpu_count': 4, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {62: 2, 70: 1, 90: 1, 102: 1, 106: 2, 126: 3}}
INFO 05-06 10:42:35.837999.837999 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.325ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 10:42:35.838107.838107 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7642974853515625e-05 seconds
INFO 05-06 10:42:35.839724.839724 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015769004821777344 seconds
INFO 05-06 10:42:35.841194.841194 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013005733489990234 seconds
INFO 05-06 10:42:35.842651.842651 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001529693603515625 seconds
INFO 05-06 10:42:35.844963.844963 mlpmodule.py:2799] [fused_experts] gmm total=1.762ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.844038.844038 mlpmodule.py:2799] [fused_experts] gmm total=1.937ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.845330.845330 mlpmodule.py:2799] [fused_experts] gmm total=2.054ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.845709.845709 mlpmodule.py:2799] [fused_experts] gmm total=2.812ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.846773.846773 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0041675567626953125 seconds
INFO 05-06 10:42:35.846929.846929 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.792213439941406e-05 seconds
DEBUG 05-06 10:42:35.847996.847996 cuda_h.py:27] end *layer_moe_fused cost 9.828 ms
DEBUG 05-06 10:42:35.847700.847700 cuda_h.py:27] end decode_layer cost 12.789 ms
DEBUG 05-06 10:42:35.847443.847443 lmp.py:904] ---- decode step 0 layer 3 ----
DEBUG 05-06 10:42:35.849562.849562 cuda_h.py:27] end *sagl cost 1.733 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [39, 67], 'expert_count': 2, 'ideal_gpu_count': 7, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 2, 'token_per_expert': {39: 1, 67: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [16, 24, 40, 44, 96, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {16: 1, 24: 1, 40: 1, 44: 1, 96: 3, 104: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 73, 85, 101, 117, 125], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {5: 1, 9: 1, 73: 1, 85: 1, 101: 1, 117: 2, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [26, 30, 34, 42, 50, 54, 70, 110, 118, 126], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 12, 'token_per_expert': {26: 2, 30: 1, 34: 1, 42: 1, 50: 2, 54: 1, 70: 1, 110: 1, 118: 1, 126: 1}}
INFO 05-06 10:42:35.851231.851231 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.372ms | allocate_experts_across_cpu_gpu: 0.124ms
INFO 05-06 10:42:35.851691.851691 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3603439331054688e-05 seconds
INFO 05-06 10:42:35.852673.852673 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015778541564941406 seconds
INFO 05-06 10:42:35.854124.854124 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016589164733886719 seconds
INFO 05-06 10:42:35.856896.856896 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001600503921508789 seconds
INFO 05-06 10:42:35.858273.858273 mlpmodule.py:2799] [fused_experts] gmm total=1.848ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.858673.858673 mlpmodule.py:2799] [fused_experts] gmm total=2.043ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.859475.859475 mlpmodule.py:2799] [fused_experts] gmm total=2.150ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.860777.860777 mlpmodule.py:2799] [fused_experts] gmm total=3.365ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.861636.861636 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004556179046630859 seconds
INFO 05-06 10:42:35.861130.861130 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:42:35.861056.861056 cuda_h.py:27] end *layer_moe_fused cost 10.904 ms
DEBUG 05-06 10:42:35.862780.862780 cuda_h.py:27] end decode_layer cost 14.218 ms
DEBUG 05-06 10:42:35.862955.862955 lmp.py:904] ---- decode step 0 layer 4 ----
DEBUG 05-06 10:42:35.863566.863566 cuda_h.py:27] end *sagl cost 1.605 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 51, 67, 83, 87], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {3: 3, 51: 2, 67: 1, 83: 1, 87: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 48, 60, 84], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {20: 2, 48: 1, 60: 2, 84: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [17, 25, 45, 93, 113, 121], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {17: 1, 25: 2, 45: 2, 93: 1, 113: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 50, 82, 106, 114, 122, 126], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 1, 50: 2, 82: 1, 106: 2, 114: 1, 122: 1, 126: 2}}
INFO 05-06 10:42:35.865285.865285 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.318ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:42:35.865347.865347 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:42:35.866563.866563 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014560222625732422 seconds
INFO 05-06 10:42:35.868317.868317 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012989044189453125 seconds
INFO 05-06 10:42:35.869331.869331 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015256404876708984 seconds
INFO 05-06 10:42:35.871879.871879 mlpmodule.py:2799] [fused_experts] gmm total=1.938ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.871740.871740 mlpmodule.py:2799] [fused_experts] gmm total=2.075ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.872901.872901 mlpmodule.py:2799] [fused_experts] gmm total=2.258ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.873149.873149 mlpmodule.py:2799] [fused_experts] gmm total=3.068ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.874776.874776 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0043354034423828125 seconds
INFO 05-06 10:42:35.874886.874886 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:42:35.874865.874865 cuda_h.py:27] end *layer_moe_fused cost 9.838 ms
DEBUG 05-06 10:42:35.875192.875192 cuda_h.py:27] end decode_layer cost 12.810 ms
DEBUG 05-06 10:42:35.875412.875412 lmp.py:904] ---- decode step 0 layer 5 ----
DEBUG 05-06 10:42:35.876453.876453 cuda_h.py:27] end *sagl cost 1.781 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [39, 71, 95, 99, 123], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 11, 'token_per_expert': {39: 2, 71: 2, 95: 2, 99: 3, 123: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 36, 52, 72, 116], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {4: 1, 36: 1, 52: 2, 72: 1, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 29, 61, 65], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {5: 2, 29: 1, 61: 3, 65: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 34, 46, 70, 74, 94], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 1, 34: 1, 46: 2, 70: 2, 74: 1, 94: 1}}
INFO 05-06 10:42:35.878921.878921 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.309ms | allocate_experts_across_cpu_gpu: 0.089ms
INFO 05-06 10:42:35.878984.878984 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 10:42:35.879044.879044 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001378774642944336 seconds
INFO 05-06 10:42:35.880114.880114 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012209415435791016 seconds
INFO 05-06 10:42:35.882562.882562 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014617443084716797 seconds
INFO 05-06 10:42:35.884575.884575 mlpmodule.py:2799] [fused_experts] gmm total=2.131ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.885007.885007 mlpmodule.py:2799] [fused_experts] gmm total=2.281ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.885206.885206 mlpmodule.py:2799] [fused_experts] gmm total=2.394ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.886425.886425 mlpmodule.py:2799] [fused_experts] gmm total=3.087ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.887860.887860 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004475593566894531 seconds
INFO 05-06 10:42:35.887970.887970 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:42:35.887346.887346 cuda_h.py:27] end *layer_moe_fused cost 9.526 ms
DEBUG 05-06 10:42:35.887161.887161 cuda_h.py:27] end decode_layer cost 12.872 ms
DEBUG 05-06 10:42:35.887666.887666 lmp.py:904] ---- decode step 0 layer 6 ----
DEBUG 05-06 10:42:35.889845.889845 cuda_h.py:27] end *sagl cost 1.532 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [35, 43, 87, 115], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {35: 2, 43: 1, 87: 3, 115: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [24, 32, 68, 96, 100, 104, 124], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {24: 2, 32: 1, 68: 1, 96: 2, 100: 1, 104: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 25, 101], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {1: 2, 13: 2, 25: 1, 101: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 70, 78, 86, 90, 106, 118], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 70: 1, 78: 2, 86: 1, 90: 1, 106: 1, 118: 1}}
INFO 05-06 10:42:35.890577.890577 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.321ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:42:35.890070.890070 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 10:42:35.892131.892131 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013780593872070312 seconds
INFO 05-06 10:42:35.893017.893017 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012974739074707031 seconds
INFO 05-06 10:42:35.895044.895044 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013103485107421875 seconds
INFO 05-06 10:42:35.897605.897605 mlpmodule.py:2799] [fused_experts] gmm total=2.168ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.897090.897090 mlpmodule.py:2799] [fused_experts] gmm total=2.210ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.898342.898342 mlpmodule.py:2799] [fused_experts] gmm total=2.505ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.898860.898860 mlpmodule.py:2799] [fused_experts] gmm total=2.749ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.899150.899150 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004404306411743164 seconds
INFO 05-06 10:42:35.899002.899002 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.817413330078125e-05 seconds
DEBUG 05-06 10:42:35.900553.900553 cuda_h.py:27] end *layer_moe_fused cost 9.778 ms
DEBUG 05-06 10:42:35.900131.900131 cuda_h.py:27] end decode_layer cost 12.697 ms
DEBUG 05-06 10:42:35.900782.900782 lmp.py:904] ---- decode step 0 layer 7 ----
DEBUG 05-06 10:42:35.902147.902147 cuda_h.py:27] end *sagl cost 1.739 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 43], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 3, 'token_per_expert': {19: 1, 43: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 64, 68, 80, 96, 100, 104], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {20: 2, 64: 1, 68: 1, 80: 1, 96: 2, 100: 1, 104: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 65, 69, 97, 101, 121], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {9: 1, 65: 1, 69: 1, 97: 3, 101: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 14, 18, 34, 82, 90, 106, 114], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {10: 1, 14: 1, 18: 1, 34: 1, 82: 1, 90: 2, 106: 1, 114: 2}}
INFO 05-06 10:42:35.903803.903803 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.337ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:42:35.904004.904004 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8358230590820312e-05 seconds
INFO 05-06 10:42:35.905647.905647 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015461444854736328 seconds
INFO 05-06 10:42:35.907940.907940 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001878976821899414 seconds
INFO 05-06 10:42:35.909578.909578 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015482902526855469 seconds
INFO 05-06 10:42:35.911572.911572 mlpmodule.py:2799] [fused_experts] gmm total=2.047ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.912491.912491 mlpmodule.py:2799] [fused_experts] gmm total=2.169ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.912670.912670 mlpmodule.py:2799] [fused_experts] gmm total=2.468ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.913204.913204 mlpmodule.py:2799] [fused_experts] gmm total=3.195ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.913290.913290 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004513263702392578 seconds
INFO 05-06 10:42:35.914488.914488 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 7.43865966796875e-05 seconds
DEBUG 05-06 10:42:35.914748.914748 cuda_h.py:27] end *layer_moe_fused cost 11.063 ms
DEBUG 05-06 10:42:35.915586.915586 cuda_h.py:27] end decode_layer cost 14.485 ms
DEBUG 05-06 10:42:35.915854.915854 lmp.py:904] ---- decode step 0 layer 8 ----
DEBUG 05-06 10:42:35.918854.918854 cuda_h.py:27] end *sagl cost 2.704 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 19, 27, 51, 55, 75, 103], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {15: 1, 19: 1, 27: 1, 51: 3, 55: 2, 75: 1, 103: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [12, 24, 32, 64], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {12: 2, 24: 1, 32: 1, 64: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [33, 69, 93, 105], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {33: 1, 69: 2, 93: 1, 105: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 38, 42, 46, 50, 54, 110, 114], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {6: 1, 38: 1, 42: 2, 46: 1, 50: 2, 54: 2, 110: 1, 114: 1}}
INFO 05-06 10:42:35.920163.920163 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.511ms | allocate_experts_across_cpu_gpu: 0.125ms
INFO 05-06 10:42:35.920922.920922 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3365020751953125e-05 seconds
INFO 05-06 10:42:35.921588.921588 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012531280517578125 seconds
INFO 05-06 10:42:35.923894.923894 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001562356948852539 seconds
INFO 05-06 10:42:35.924964.924964 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014150142669677734 seconds
INFO 05-06 10:42:35.927602.927602 mlpmodule.py:2799] [fused_experts] gmm total=2.114ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.927039.927039 mlpmodule.py:2799] [fused_experts] gmm total=2.232ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.927313.927313 mlpmodule.py:2799] [fused_experts] gmm total=2.697ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.928471.928471 mlpmodule.py:2799] [fused_experts] gmm total=3.363ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.929915.929915 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004598379135131836 seconds
INFO 05-06 10:42:35.929290.929290 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.6743621826171875e-05 seconds
DEBUG 05-06 10:42:35.930325.930325 cuda_h.py:27] end *layer_moe_fused cost 10.415 ms
DEBUG 05-06 10:42:35.930666.930666 cuda_h.py:27] end decode_layer cost 15.167 ms
DEBUG 05-06 10:42:35.930000.930000 lmp.py:904] ---- decode step 0 layer 9 ----
DEBUG 05-06 10:42:35.932387.932387 cuda_h.py:27] end *sagl cost 2.025 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 19, 83, 95, 111], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {7: 2, 15: 1, 19: 2, 83: 1, 95: 3, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 36, 48, 76], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {4: 1, 36: 1, 48: 1, 76: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [37, 57, 69, 81, 89, 101, 125], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {37: 1, 57: 1, 69: 2, 81: 1, 89: 2, 101: 3, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [30, 54, 70, 74], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {30: 2, 54: 1, 70: 2, 74: 2}}
INFO 05-06 10:42:35.934184.934184 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.395ms | allocate_experts_across_cpu_gpu: 0.123ms
INFO 05-06 10:42:35.934320.934320 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.288818359375e-05 seconds
INFO 05-06 10:42:35.935522.935522 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014238357543945312 seconds
INFO 05-06 10:42:35.937294.937294 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017714500427246094 seconds
INFO 05-06 10:42:35.939071.939071 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015685558319091797 seconds
INFO 05-06 10:42:35.943818.943818 mlpmodule.py:2799] [fused_experts] gmm total=2.083ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.943745.943745 mlpmodule.py:2799] [fused_experts] gmm total=2.170ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.943359.943359 mlpmodule.py:2799] [fused_experts] gmm total=2.716ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.944250.944250 mlpmodule.py:2799] [fused_experts] gmm total=3.109ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.945950.945950 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005731344223022461 seconds
INFO 05-06 10:42:35.945391.945391 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.316734313964844e-05 seconds
DEBUG 05-06 10:42:35.945636.945636 cuda_h.py:27] end *layer_moe_fused cost 12.057 ms
DEBUG 05-06 10:42:35.946160.946160 cuda_h.py:27] end decode_layer cost 15.669 ms
DEBUG 05-06 10:42:35.946381.946381 lmp.py:904] ---- decode step 0 layer 10 ----
DEBUG 05-06 10:42:35.947733.947733 cuda_h.py:27] end *sagl cost 1.589 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 47, 67, 79, 87, 127], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 6, 'token_per_expert': {19: 1, 47: 1, 67: 1, 79: 1, 87: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 28, 44, 60], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {8: 3, 28: 1, 44: 2, 60: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 21, 37, 57, 81, 97, 105], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {5: 1, 21: 1, 37: 1, 57: 1, 81: 3, 97: 3, 105: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 46, 54, 62, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {18: 1, 46: 2, 54: 1, 62: 1, 126: 1}}
INFO 05-06 10:42:35.949559.949559 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.324ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:42:35.949767.949767 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 10:42:35.950872.950872 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001340627670288086 seconds
INFO 05-06 10:42:35.952951.952951 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013000965118408203 seconds
INFO 05-06 10:42:35.953928.953928 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014295578002929688 seconds
INFO 05-06 10:42:35.955748.955748 mlpmodule.py:2799] [fused_experts] gmm total=2.091ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.956749.956749 mlpmodule.py:2799] [fused_experts] gmm total=2.239ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.956614.956614 mlpmodule.py:2799] [fused_experts] gmm total=2.271ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.956394.956394 mlpmodule.py:2799] [fused_experts] gmm total=2.732ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.957201.957201 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0043141841888427734 seconds
INFO 05-06 10:42:35.958834.958834 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:42:35.958199.958199 cuda_h.py:27] end *layer_moe_fused cost 9.672 ms
DEBUG 05-06 10:42:35.958984.958984 cuda_h.py:27] end decode_layer cost 12.659 ms
DEBUG 05-06 10:42:35.959250.959250 lmp.py:904] ---- decode step 0 layer 11 ----
DEBUG 05-06 10:42:35.960458.960458 cuda_h.py:27] end *sagl cost 1.798 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 23, 63, 67, 79, 83, 99], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 13, 'token_per_expert': {7: 1, 23: 1, 63: 1, 67: 1, 79: 3, 83: 4, 99: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [124], 'expert_count': 1, 'ideal_gpu_count': 5, 'keep_on_gpu': 1, 'hit_count_on_device': 1, 'token_total': 2, 'token_per_expert': {124: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 49, 81], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {9: 1, 49: 2, 81: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 50, 70, 102, 114], 'expert_count': 8, 'ideal_gpu_count': 4, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 6: 1, 38: 3, 46: 2, 50: 1, 70: 1, 102: 1, 114: 1}}
INFO 05-06 10:42:35.962547.962547 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.348ms | allocate_experts_across_cpu_gpu: 0.105ms
INFO 05-06 10:42:35.962676.962676 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2172927856445312e-05 seconds
INFO 05-06 10:42:35.963407.963407 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014324188232421875 seconds
INFO 05-06 10:42:35.965198.965198 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017154216766357422 seconds
INFO 05-06 10:42:35.967048.967048 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015599727630615234 seconds
INFO 05-06 10:42:35.969310.969310 mlpmodule.py:2799] [fused_experts] gmm total=1.802ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.969047.969047 mlpmodule.py:2799] [fused_experts] gmm total=1.964ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.970743.970743 mlpmodule.py:2799] [fused_experts] gmm total=2.355ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.971804.971804 mlpmodule.py:2799] [fused_experts] gmm total=3.091ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.971710.971710 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004413127899169922 seconds
INFO 05-06 10:42:35.972105.972105 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:42:35.972133.972133 cuda_h.py:27] end *layer_moe_fused cost 10.650 ms
DEBUG 05-06 10:42:35.973880.973880 cuda_h.py:27] end decode_layer cost 14.047 ms
DEBUG 05-06 10:42:35.973100.973100 lmp.py:904] ---- decode step 0 layer 12 ----
DEBUG 05-06 10:42:35.974133.974133 cuda_h.py:27] end *sagl cost 1.530 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 19, 39], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {3: 1, 15: 1, 19: 2, 39: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [36, 80, 92], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {36: 2, 80: 2, 92: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [21, 45, 73, 97, 117], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {21: 1, 45: 1, 73: 1, 97: 1, 117: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [46, 50, 74, 78, 86, 106, 114], 'expert_count': 7, 'ideal_gpu_count': 4, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 13, 'token_per_expert': {46: 2, 50: 1, 74: 2, 78: 4, 86: 1, 106: 2, 114: 1}}
INFO 05-06 10:42:35.975612.975612 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.323ms | allocate_experts_across_cpu_gpu: 0.089ms
INFO 05-06 10:42:35.976721.976721 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 10:42:35.977552.977552 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014538764953613281 seconds
INFO 05-06 10:42:35.978137.978137 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0011775493621826172 seconds
INFO 05-06 10:42:35.980219.980219 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001371622085571289 seconds
INFO 05-06 10:42:35.982885.982885 mlpmodule.py:2799] [fused_experts] gmm total=2.137ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.982608.982608 mlpmodule.py:2799] [fused_experts] gmm total=2.249ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.983908.983908 mlpmodule.py:2799] [fused_experts] gmm total=2.397ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.984240.984240 mlpmodule.py:2799] [fused_experts] gmm total=3.317ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:35.985630.985630 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004618644714355469 seconds
INFO 05-06 10:42:35.985816.985816 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 8.487701416015625e-05 seconds
DEBUG 05-06 10:42:35.985116.985116 cuda_h.py:27] end *layer_moe_fused cost 10.292 ms
DEBUG 05-06 10:42:35.986142.986142 cuda_h.py:27] end decode_layer cost 13.499 ms
DEBUG 05-06 10:42:35.986523.986523 lmp.py:904] ---- decode step 0 layer 13 ----
DEBUG 05-06 10:42:35.990945.990945 cuda_h.py:27] end *sagl cost 3.212 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [47, 55, 71, 107], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {47: 1, 55: 1, 71: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [32, 76, 80, 100, 104], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {32: 1, 76: 1, 80: 2, 100: 3, 104: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 41, 109, 125], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 2, 13: 1, 41: 1, 109: 1, 125: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 26, 78, 110, 114], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 6: 1, 14: 2, 22: 1, 26: 1, 78: 2, 110: 1, 114: 2}}
INFO 05-06 10:42:35.992848.992848 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.460ms | allocate_experts_across_cpu_gpu: 0.148ms
INFO 05-06 10:42:35.992150.992150 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.956390380859375e-05 seconds
INFO 05-06 10:42:35.994487.994487 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015265941619873047 seconds
INFO 05-06 10:42:35.996919.996919 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0019130706787109375 seconds
INFO 05-06 10:42:35.997705.997705 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001434326171875 seconds
INFO 05-06 10:42:35.999531.999531 mlpmodule.py:2799] [fused_experts] gmm total=2.078ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.000592.000592 mlpmodule.py:2799] [fused_experts] gmm total=2.216ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.000521.000521 mlpmodule.py:2799] [fused_experts] gmm total=2.389ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.001527.001527 mlpmodule.py:2799] [fused_experts] gmm total=3.230ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.002145.002145 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004553079605102539 seconds
INFO 05-06 10:42:36.002805.002805 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.602836608886719e-05 seconds
DEBUG 05-06 10:42:36.002297.002297 cuda_h.py:27] end *layer_moe_fused cost 11.056 ms
DEBUG 05-06 10:42:36.003366.003366 cuda_h.py:27] end decode_layer cost 16.455 ms
DEBUG 05-06 10:42:36.003838.003838 lmp.py:904] ---- decode step 0 layer 14 ----
DEBUG 05-06 10:42:36.005154.005154 cuda_h.py:27] end *sagl cost 1.871 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 31, 39, 47, 99, 115], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 11, 'token_per_expert': {11: 2, 31: 1, 39: 2, 47: 1, 99: 1, 115: 4}}
experts_gpu_alloc_device_1 {'expert_ids': [56, 112], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 2, 'token_per_expert': {56: 1, 112: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 25, 81, 97, 109, 121], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {9: 1, 25: 2, 81: 2, 97: 1, 109: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 26, 38, 46, 62], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {2: 3, 10: 2, 26: 2, 38: 1, 46: 1, 62: 1}}
INFO 05-06 10:42:36.006033.006033 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.373ms | allocate_experts_across_cpu_gpu: 0.112ms
INFO 05-06 10:42:36.006599.006599 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2649765014648438e-05 seconds
INFO 05-06 10:42:36.008272.008272 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014688968658447266 seconds
INFO 05-06 10:42:36.009797.009797 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001538991928100586 seconds
INFO 05-06 10:42:36.011779.011779 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013728141784667969 seconds
INFO 05-06 10:42:36.013670.013670 mlpmodule.py:2799] [fused_experts] gmm total=2.012ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.014393.014393 mlpmodule.py:2799] [fused_experts] gmm total=2.322ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.014369.014369 mlpmodule.py:2799] [fused_experts] gmm total=2.385ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.015522.015522 mlpmodule.py:2799] [fused_experts] gmm total=3.101ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.016884.016884 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004567861557006836 seconds
INFO 05-06 10:42:36.016897.016897 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 7.534027099609375e-05 seconds
DEBUG 05-06 10:42:36.016862.016862 cuda_h.py:27] end *layer_moe_fused cost 10.399 ms
DEBUG 05-06 10:42:36.017681.017681 cuda_h.py:27] end decode_layer cost 14.094 ms
DEBUG 05-06 10:42:36.017188.017188 lmp.py:904] ---- decode step 0 layer 15 ----
DEBUG 05-06 10:42:36.020527.020527 cuda_h.py:27] end *sagl cost 2.744 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [47, 75, 83, 119], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {47: 1, 75: 2, 83: 2, 119: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [24, 68, 72, 84, 100, 108, 112], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {24: 1, 68: 2, 72: 2, 84: 1, 100: 1, 108: 2, 112: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [29, 33, 69, 81, 93, 97, 101], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {29: 1, 33: 2, 69: 1, 81: 3, 93: 1, 97: 1, 101: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [30, 34], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 2, 'token_per_expert': {30: 1, 34: 1}}
INFO 05-06 10:42:36.022870.022870 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.422ms | allocate_experts_across_cpu_gpu: 0.130ms
INFO 05-06 10:42:36.022489.022489 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3126602172851562e-05 seconds
INFO 05-06 10:42:36.023348.023348 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001459360122680664 seconds
INFO 05-06 10:42:36.025388.025388 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016903877258300781 seconds
INFO 05-06 10:42:36.027570.027570 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015797615051269531 seconds
INFO 05-06 10:42:36.029196.029196 mlpmodule.py:2799] [fused_experts] gmm total=2.111ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.030337.030337 mlpmodule.py:2799] [fused_experts] gmm total=2.226ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.030663.030663 mlpmodule.py:2799] [fused_experts] gmm total=2.366ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.030129.030129 mlpmodule.py:2799] [fused_experts] gmm total=2.928ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.031182.031182 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004477024078369141 seconds
INFO 05-06 10:42:36.032710.032710 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.698204040527344e-05 seconds
DEBUG 05-06 10:42:36.032897.032897 cuda_h.py:27] end *layer_moe_fused cost 10.863 ms
DEBUG 05-06 10:42:36.033987.033987 cuda_h.py:27] end decode_layer cost 15.579 ms
DEBUG 05-06 10:42:36.033606.033606 lmp.py:904] ---- decode step 0 layer 16 ----
DEBUG 05-06 10:42:36.035935.035935 cuda_h.py:27] end *sagl cost 1.880 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 19, 63, 87, 107], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {15: 1, 19: 2, 63: 1, 87: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 32, 44, 52], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 7, 'token_per_expert': {0: 1, 4: 1, 16: 1, 20: 1, 32: 1, 44: 1, 52: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 85, 105], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 9, 'token_per_expert': {1: 3, 5: 2, 85: 2, 105: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [54, 62, 66, 78, 82, 102, 122], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {54: 1, 62: 2, 66: 1, 78: 1, 82: 1, 102: 2, 122: 1}}
INFO 05-06 10:42:36.036604.036604 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.381ms | allocate_experts_across_cpu_gpu: 0.120ms
INFO 05-06 10:42:36.036356.036356 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.2411346435546875e-05 seconds
INFO 05-06 10:42:36.038576.038576 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015766620635986328 seconds
INFO 05-06 10:42:36.040197.040197 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017819404602050781 seconds
INFO 05-06 10:42:36.041170.041170 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001477956771850586 seconds
INFO 05-06 10:42:36.044242.044242 mlpmodule.py:2799] [fused_experts] gmm total=2.278ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.044248.044248 mlpmodule.py:2799] [fused_experts] gmm total=2.262ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.044805.044805 mlpmodule.py:2799] [fused_experts] gmm total=2.596ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.045906.045906 mlpmodule.py:2799] [fused_experts] gmm total=2.659ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.046912.046912 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004445314407348633 seconds
INFO 05-06 10:42:36.046725.046725 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 7.295608520507812e-05 seconds
DEBUG 05-06 10:42:36.047002.047002 cuda_h.py:27] end *layer_moe_fused cost 11.012 ms
DEBUG 05-06 10:42:36.047972.047972 cuda_h.py:27] end decode_layer cost 14.690 ms
DEBUG 05-06 10:42:36.047571.047571 lmp.py:904] ---- decode step 0 layer 17 ----
DEBUG 05-06 10:42:36.050600.050600 cuda_h.py:27] end *sagl cost 2.588 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 23, 27, 39, 47, 55, 95], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {7: 1, 15: 1, 23: 3, 27: 1, 39: 1, 47: 1, 55: 1, 95: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [16, 120], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {16: 3, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 13, 33, 37, 53, 65, 73, 113], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {5: 1, 13: 2, 33: 1, 37: 1, 53: 1, 65: 1, 73: 2, 113: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [18, 22, 34, 70], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {18: 3, 22: 1, 34: 2, 70: 1}}
INFO 05-06 10:42:36.052408.052408 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.425ms | allocate_experts_across_cpu_gpu: 0.134ms
INFO 05-06 10:42:36.052942.052942 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.5272369384765625e-05 seconds
INFO 05-06 10:42:36.054919.054919 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014400482177734375 seconds
INFO 05-06 10:42:36.055464.055464 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017535686492919922 seconds
INFO 05-06 10:42:36.057816.057816 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015413761138916016 seconds
INFO 05-06 10:42:36.059339.059339 mlpmodule.py:2799] [fused_experts] gmm total=2.025ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.060376.060376 mlpmodule.py:2799] [fused_experts] gmm total=2.412ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.060948.060948 mlpmodule.py:2799] [fused_experts] gmm total=2.520ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.061880.061880 mlpmodule.py:2799] [fused_experts] gmm total=2.963ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.062231.062231 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004397392272949219 seconds
INFO 05-06 10:42:36.062283.062283 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.008148193359375e-05 seconds
DEBUG 05-06 10:42:36.062459.062459 cuda_h.py:27] end *layer_moe_fused cost 10.782 ms
DEBUG 05-06 10:42:36.063416.063416 cuda_h.py:27] end decode_layer cost 15.330 ms
DEBUG 05-06 10:42:36.063703.063703 lmp.py:904] ---- decode step 0 layer 18 ----
DEBUG 05-06 10:42:36.065657.065657 cuda_h.py:27] end *sagl cost 1.954 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [23, 59, 75, 83, 111], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {23: 2, 59: 1, 75: 2, 83: 2, 111: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [32, 36, 40, 80, 104], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {32: 1, 36: 1, 40: 2, 80: 1, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [37, 69, 73, 77, 97, 101, 105], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {37: 2, 69: 1, 73: 1, 77: 1, 97: 1, 101: 1, 105: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [26, 30, 42, 54, 58], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {26: 1, 30: 2, 42: 1, 54: 3, 58: 2}}
INFO 05-06 10:42:36.066413.066413 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.396ms | allocate_experts_across_cpu_gpu: 0.121ms
INFO 05-06 10:42:36.067741.067741 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3365020751953125e-05 seconds
INFO 05-06 10:42:36.068599.068599 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016243457794189453 seconds
INFO 05-06 10:42:36.070528.070528 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016906261444091797 seconds
INFO 05-06 10:42:36.072866.072866 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015044212341308594 seconds
INFO 05-06 10:42:36.074817.074817 mlpmodule.py:2799] [fused_experts] gmm total=2.049ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.074585.074585 mlpmodule.py:2799] [fused_experts] gmm total=2.150ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.074429.074429 mlpmodule.py:2799] [fused_experts] gmm total=2.310ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.075882.075882 mlpmodule.py:2799] [fused_experts] gmm total=3.068ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.076310.076310 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0044744014739990234 seconds
INFO 05-06 10:42:36.076757.076757 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.0067901611328125e-05 seconds
DEBUG 05-06 10:42:36.077932.077932 cuda_h.py:27] end *layer_moe_fused cost 10.876 ms
DEBUG 05-06 10:42:36.077884.077884 cuda_h.py:27] end decode_layer cost 14.490 ms
DEBUG 05-06 10:42:36.077058.077058 lmp.py:904] ---- decode step 0 layer 19 ----
DEBUG 05-06 10:42:36.079894.079894 cuda_h.py:27] end *sagl cost 1.595 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 31, 111], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {19: 1, 31: 1, 111: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [40, 44, 84, 96], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 9, 'token_per_expert': {40: 3, 44: 3, 84: 2, 96: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 25, 61, 125], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {5: 1, 25: 2, 61: 1, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 22, 38, 62, 78, 82, 86, 106, 122], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 14, 'token_per_expert': {10: 3, 22: 1, 38: 1, 62: 1, 78: 2, 82: 1, 86: 1, 106: 3, 122: 1}}
INFO 05-06 10:42:36.080410.080410 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.314ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 10:42:36.080565.080565 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7881393432617188e-05 seconds
INFO 05-06 10:42:36.082260.082260 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013413429260253906 seconds
INFO 05-06 10:42:36.083654.083654 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013554096221923828 seconds
INFO 05-06 10:42:36.085678.085678 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014276504516601562 seconds
INFO 05-06 10:42:36.087817.087817 mlpmodule.py:2799] [fused_experts] gmm total=1.944ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.087350.087350 mlpmodule.py:2799] [fused_experts] gmm total=2.132ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.088133.088133 mlpmodule.py:2799] [fused_experts] gmm total=2.472ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.089848.089848 mlpmodule.py:2799] [fused_experts] gmm total=3.183ms E=32 S=14 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.089537.089537 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0044307708740234375 seconds
INFO 05-06 10:42:36.089269.089269 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:42:36.090983.090983 cuda_h.py:27] end *layer_moe_fused cost 9.827 ms
DEBUG 05-06 10:42:36.090521.090521 cuda_h.py:27] end decode_layer cost 12.836 ms
DEBUG 05-06 10:42:36.090933.090933 lmp.py:904] ---- decode step 0 layer 20 ----
DEBUG 05-06 10:42:36.092895.092895 cuda_h.py:27] end *sagl cost 1.582 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 95, 107], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {3: 1, 95: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [36, 40, 52, 60, 76, 92], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {36: 1, 40: 2, 52: 1, 60: 1, 76: 1, 92: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 21, 45, 65, 73, 85, 117], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 14, 'token_per_expert': {13: 2, 21: 4, 45: 1, 65: 1, 73: 2, 85: 2, 117: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 58, 62, 94, 102], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 58: 1, 62: 1, 94: 2, 102: 1}}
INFO 05-06 10:42:36.093919.093919 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.323ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:42:36.093704.093704 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:42:36.095999.095999 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012693405151367188 seconds
INFO 05-06 10:42:36.096887.096887 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013332366943359375 seconds
INFO 05-06 10:42:36.098149.098149 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001424551010131836 seconds
INFO 05-06 10:42:36.100514.100514 mlpmodule.py:2799] [fused_experts] gmm total=2.085ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.100944.100944 mlpmodule.py:2799] [fused_experts] gmm total=2.357ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.100601.100601 mlpmodule.py:2799] [fused_experts] gmm total=2.402ms E=32 S=14 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.100592.100592 mlpmodule.py:2799] [fused_experts] gmm total=2.447ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.102713.102713 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004526853561401367 seconds
INFO 05-06 10:42:36.102823.102823 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:42:36.103239.103239 cuda_h.py:27] end *layer_moe_fused cost 9.963 ms
DEBUG 05-06 10:42:36.103513.103513 cuda_h.py:27] end decode_layer cost 12.957 ms
DEBUG 05-06 10:42:36.103826.103826 lmp.py:904] ---- decode step 0 layer 21 ----
DEBUG 05-06 10:42:36.105574.105574 cuda_h.py:27] end *sagl cost 1.532 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 59, 71, 83, 87, 103], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {11: 2, 59: 1, 71: 1, 83: 1, 87: 2, 103: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [68, 80, 124], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {68: 1, 80: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 21, 25, 57], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {5: 4, 21: 1, 25: 1, 57: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 14, 26, 34, 58, 78, 86, 94, 110], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 13, 'token_per_expert': {2: 1, 14: 1, 26: 2, 34: 2, 58: 1, 78: 1, 86: 2, 94: 2, 110: 1}}
INFO 05-06 10:42:36.106563.106563 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.303ms | allocate_experts_across_cpu_gpu: 0.092ms
INFO 05-06 10:42:36.106142.106142 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6689300537109375e-05 seconds
INFO 05-06 10:42:36.108734.108734 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001405477523803711 seconds
INFO 05-06 10:42:36.110308.110308 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0018045902252197266 seconds
INFO 05-06 10:42:36.111165.111165 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001569509506225586 seconds
INFO 05-06 10:42:36.114120.114120 mlpmodule.py:2799] [fused_experts] gmm total=2.048ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.114645.114645 mlpmodule.py:2799] [fused_experts] gmm total=2.085ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.114554.114554 mlpmodule.py:2799] [fused_experts] gmm total=2.441ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.115651.115651 mlpmodule.py:2799] [fused_experts] gmm total=3.147ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.116531.116531 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004643917083740234 seconds
INFO 05-06 10:42:36.116597.116597 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 7.700920104980469e-05 seconds
DEBUG 05-06 10:42:36.117331.117331 cuda_h.py:27] end *layer_moe_fused cost 10.860 ms
DEBUG 05-06 10:42:36.117964.117964 cuda_h.py:27] end decode_layer cost 13.993 ms
DEBUG 05-06 10:42:36.117477.117477 lmp.py:904] ---- decode step 0 layer 22 ----
DEBUG 05-06 10:42:36.120038.120038 cuda_h.py:27] end *sagl cost 2.833 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [43, 67, 107, 119, 123, 127], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {43: 2, 67: 1, 107: 1, 119: 3, 123: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 16, 24, 32, 60, 76, 108, 120], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {8: 1, 16: 1, 24: 2, 32: 1, 60: 1, 76: 1, 108: 2, 120: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [61, 101, 109], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {61: 1, 101: 1, 109: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [26, 38, 74, 94], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {26: 1, 38: 2, 74: 1, 94: 3}}
INFO 05-06 10:42:36.123403.123403 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.519ms | allocate_experts_across_cpu_gpu: 0.127ms
INFO 05-06 10:42:36.123162.123162 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.4557113647460938e-05 seconds
INFO 05-06 10:42:36.124028.124028 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015039443969726562 seconds
INFO 05-06 10:42:36.126681.126681 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016388893127441406 seconds
INFO 05-06 10:42:36.127389.127389 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014464855194091797 seconds
INFO 05-06 10:42:36.130372.130372 mlpmodule.py:2799] [fused_experts] gmm total=2.125ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.130175.130175 mlpmodule.py:2799] [fused_experts] gmm total=2.268ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.130940.130940 mlpmodule.py:2799] [fused_experts] gmm total=2.276ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.131521.131521 mlpmodule.py:2799] [fused_experts] gmm total=2.710ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.132316.132316 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004408597946166992 seconds
INFO 05-06 10:42:36.132844.132844 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.984306335449219e-05 seconds
DEBUG 05-06 10:42:36.132150.132150 cuda_h.py:27] end *layer_moe_fused cost 10.506 ms
DEBUG 05-06 10:42:36.133089.133089 cuda_h.py:27] end decode_layer cost 15.686 ms
DEBUG 05-06 10:42:36.133760.133760 lmp.py:904] ---- decode step 0 layer 23 ----
DEBUG 05-06 10:42:36.135073.135073 cuda_h.py:27] end *sagl cost 2.179 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 47, 67], 'expert_count': 3, 'ideal_gpu_count': 4, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {19: 1, 47: 1, 67: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 32, 84, 108], 'expert_count': 5, 'ideal_gpu_count': 4, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {8: 1, 12: 2, 32: 2, 84: 3, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [17, 61, 81, 97, 109], 'expert_count': 5, 'ideal_gpu_count': 4, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 10, 'token_per_expert': {17: 3, 61: 1, 81: 1, 97: 4, 109: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 86, 118], 'expert_count': 3, 'ideal_gpu_count': 4, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 8, 'token_per_expert': {22: 2, 86: 4, 118: 2}}
INFO 05-06 10:42:36.137341.137341 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.407ms | allocate_experts_across_cpu_gpu: 0.114ms
INFO 05-06 10:42:36.137245.137245 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:42:36.139330.139330 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014858245849609375 seconds
INFO 05-06 10:42:36.140459.140459 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0015926361083984375 seconds
INFO 05-06 10:42:36.142485.142485 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001481771469116211 seconds
INFO 05-06 10:42:36.144037.144037 mlpmodule.py:2799] [fused_experts] gmm total=2.026ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.144967.144967 mlpmodule.py:2799] [fused_experts] gmm total=2.203ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.145960.145960 mlpmodule.py:2799] [fused_experts] gmm total=2.311ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.145076.145076 mlpmodule.py:2799] [fused_experts] gmm total=2.504ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.146207.146207 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004279375076293945 seconds
INFO 05-06 10:42:36.146794.146794 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 10:42:36.147535.147535 cuda_h.py:27] end *layer_moe_fused cost 10.079 ms
DEBUG 05-06 10:42:36.147724.147724 cuda_h.py:27] end decode_layer cost 13.944 ms
DEBUG 05-06 10:42:36.147467.147467 lmp.py:904] ---- decode step 0 layer 24 ----
DEBUG 05-06 10:42:36.149891.149891 cuda_h.py:27] end *sagl cost 1.538 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [27, 63, 79, 123], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {27: 1, 63: 1, 79: 2, 123: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [12, 40, 44], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 7, 'token_per_expert': {12: 3, 40: 2, 44: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 33, 65, 109, 113, 121], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {5: 1, 33: 3, 65: 1, 109: 2, 113: 2, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 30, 66, 90, 110, 118], 'expert_count': 6, 'ideal_gpu_count': 4, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {10: 1, 30: 2, 66: 2, 90: 1, 110: 2, 118: 1}}
INFO 05-06 10:42:36.150000.150000 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.313ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:42:36.150155.150155 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7404556274414062e-05 seconds
INFO 05-06 10:42:36.152853.152853 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014257431030273438 seconds
INFO 05-06 10:42:36.153261.153261 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012536048889160156 seconds
INFO 05-06 10:42:36.154636.154636 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014407634735107422 seconds
INFO 05-06 10:42:36.157823.157823 mlpmodule.py:2799] [fused_experts] gmm total=1.983ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.157657.157657 mlpmodule.py:2799] [fused_experts] gmm total=2.069ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.157297.157297 mlpmodule.py:2799] [fused_experts] gmm total=2.302ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.158084.158084 mlpmodule.py:2799] [fused_experts] gmm total=2.975ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.159709.159709 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004273414611816406 seconds
INFO 05-06 10:42:36.159441.159441 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 10:42:36.159323.159323 cuda_h.py:27] end *layer_moe_fused cost 9.667 ms
DEBUG 05-06 10:42:36.160443.160443 cuda_h.py:27] end decode_layer cost 12.511 ms
DEBUG 05-06 10:42:36.160140.160140 lmp.py:904] ---- decode step 0 layer 25 ----
DEBUG 05-06 10:42:36.161944.161944 cuda_h.py:27] end *sagl cost 1.607 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 19, 27, 47, 59, 67, 95], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {3: 1, 19: 1, 27: 1, 47: 1, 59: 1, 67: 2, 95: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 44, 52, 68, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 1, 16: 1, 44: 2, 52: 1, 68: 2, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [29, 41, 45, 85, 93, 97, 117, 121], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {29: 1, 41: 1, 45: 1, 85: 1, 93: 3, 97: 1, 117: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 58], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {2: 1, 6: 1, 58: 4}}
INFO 05-06 10:42:36.163026.163026 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.308ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:42:36.163281.163281 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:42:36.164487.164487 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013799667358398438 seconds
INFO 05-06 10:42:36.166823.166823 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012791156768798828 seconds
INFO 05-06 10:42:36.167019.167019 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001415252685546875 seconds
INFO 05-06 10:42:36.169128.169128 mlpmodule.py:2799] [fused_experts] gmm total=2.242ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.170842.170842 mlpmodule.py:2799] [fused_experts] gmm total=2.308ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.170171.170171 mlpmodule.py:2799] [fused_experts] gmm total=2.347ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.170744.170744 mlpmodule.py:2799] [fused_experts] gmm total=2.371ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.174770.174770 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007332324981689453 seconds
INFO 05-06 10:42:36.175006.175006 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.745887756347656e-05 seconds
DEBUG 05-06 10:42:36.175167.175167 cuda_h.py:27] end *layer_moe_fused cost 12.518 ms
DEBUG 05-06 10:42:36.175202.175202 cuda_h.py:27] end decode_layer cost 15.698 ms
DEBUG 05-06 10:42:36.176629.176629 lmp.py:904] ---- decode step 0 layer 26 ----
DEBUG 05-06 10:42:36.177873.177873 cuda_h.py:27] end *sagl cost 1.921 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 23, 27, 43, 79, 119], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {19: 2, 23: 1, 27: 1, 43: 3, 79: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 20, 52, 84], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {4: 3, 8: 1, 20: 2, 52: 1, 84: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [49, 65, 85], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 7, 'token_per_expert': {49: 2, 65: 3, 85: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [10, 38, 62, 70, 90], 'expert_count': 5, 'ideal_gpu_count': 4, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {10: 1, 38: 2, 62: 1, 70: 2, 90: 2}}
INFO 05-06 10:42:36.179847.179847 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.408ms | allocate_experts_across_cpu_gpu: 0.114ms
INFO 05-06 10:42:36.179281.179281 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0265579223632812e-05 seconds
INFO 05-06 10:42:36.181979.181979 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014040470123291016 seconds
INFO 05-06 10:42:36.183844.183844 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0019483566284179688 seconds
INFO 05-06 10:42:36.184937.184937 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013275146484375 seconds
INFO 05-06 10:42:36.187512.187512 mlpmodule.py:2799] [fused_experts] gmm total=2.076ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.187234.187234 mlpmodule.py:2799] [fused_experts] gmm total=2.190ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.187063.187063 mlpmodule.py:2799] [fused_experts] gmm total=2.602ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.188324.188324 mlpmodule.py:2799] [fused_experts] gmm total=2.788ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.189009.189009 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004590034484863281 seconds
INFO 05-06 10:42:36.189688.189688 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:42:36.189224.189224 cuda_h.py:27] end *layer_moe_fused cost 10.775 ms
DEBUG 05-06 10:42:36.190794.190794 cuda_h.py:27] end decode_layer cost 14.233 ms
DEBUG 05-06 10:42:36.190299.190299 lmp.py:904] ---- decode step 0 layer 27 ----
DEBUG 05-06 10:42:36.191686.191686 cuda_h.py:27] end *sagl cost 1.615 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [87, 103, 115], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {87: 1, 103: 3, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [32, 48, 100, 108], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {32: 1, 48: 1, 100: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 25, 29, 41, 45, 53, 61, 85, 97, 121], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 14, 'token_per_expert': {1: 1, 25: 1, 29: 2, 41: 2, 45: 1, 53: 1, 61: 1, 85: 2, 97: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [14, 50, 58, 62, 82, 114], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {14: 1, 50: 1, 58: 2, 62: 1, 82: 2, 114: 2}}
INFO 05-06 10:42:36.193384.193384 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.308ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 10:42:36.193255.193255 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 10:42:36.194509.194509 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014145374298095703 seconds
INFO 05-06 10:42:36.196845.196845 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012731552124023438 seconds
INFO 05-06 10:42:36.197531.197531 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012505054473876953 seconds
INFO 05-06 10:42:36.199827.199827 mlpmodule.py:2799] [fused_experts] gmm total=1.885ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.199089.199089 mlpmodule.py:2799] [fused_experts] gmm total=2.084ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.200481.200481 mlpmodule.py:2799] [fused_experts] gmm total=2.261ms E=32 S=14 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.200868.200868 mlpmodule.py:2799] [fused_experts] gmm total=2.852ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.201705.201705 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004141807556152344 seconds
INFO 05-06 10:42:36.201815.201815 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:42:36.202757.202757 cuda_h.py:27] end *layer_moe_fused cost 9.355 ms
DEBUG 05-06 10:42:36.202446.202446 cuda_h.py:27] end decode_layer cost 12.287 ms
DEBUG 05-06 10:42:36.202190.202190 lmp.py:904] ---- decode step 0 layer 28 ----
DEBUG 05-06 10:42:36.204826.204826 cuda_h.py:27] end *sagl cost 1.553 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 39, 67, 111, 115, 119], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 11, 'token_per_expert': {19: 2, 39: 2, 67: 2, 111: 1, 115: 2, 119: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 32, 104, 108, 112], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {20: 1, 32: 2, 104: 2, 108: 1, 112: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 33, 49, 53, 57, 65, 89, 105, 113], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {13: 1, 33: 1, 49: 3, 53: 1, 57: 2, 65: 1, 89: 1, 105: 1, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [22], 'expert_count': 1, 'ideal_gpu_count': 5, 'keep_on_gpu': 1, 'hit_count_on_device': 1, 'token_total': 2, 'token_per_expert': {22: 2}}
INFO 05-06 10:42:36.205948.205948 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.313ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:42:36.205580.205580 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.71661376953125e-05 seconds
INFO 05-06 10:42:36.206828.206828 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012357234954833984 seconds
INFO 05-06 10:42:36.208943.208943 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012226104736328125 seconds
INFO 05-06 10:42:36.209803.209803 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001272439956665039 seconds
INFO 05-06 10:42:36.211824.211824 mlpmodule.py:2799] [fused_experts] gmm total=2.187ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.212911.212911 mlpmodule.py:2799] [fused_experts] gmm total=2.327ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.212527.212527 mlpmodule.py:2799] [fused_experts] gmm total=2.343ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.212657.212657 mlpmodule.py:2799] [fused_experts] gmm total=2.740ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.213679.213679 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0041692256927490234 seconds
INFO 05-06 10:42:36.213372.213372 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.7697296142578125e-05 seconds
DEBUG 05-06 10:42:36.214932.214932 cuda_h.py:27] end *layer_moe_fused cost 9.214 ms
DEBUG 05-06 10:42:36.214490.214490 cuda_h.py:27] end decode_layer cost 12.119 ms
DEBUG 05-06 10:42:36.214233.214233 lmp.py:904] ---- decode step 0 layer 29 ----
DEBUG 05-06 10:42:36.216173.216173 cuda_h.py:27] end *sagl cost 1.531 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [23, 71], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {23: 3, 71: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 52, 56, 60, 64], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 10, 'token_per_expert': {4: 2, 52: 2, 56: 2, 60: 2, 64: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 49, 73, 81, 97], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 1, 49: 1, 73: 2, 81: 2, 97: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 18, 26, 30, 66, 78, 82, 106], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 18: 2, 26: 2, 30: 1, 66: 2, 78: 1, 82: 1, 106: 1}}
INFO 05-06 10:42:36.217375.217375 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.304ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:42:36.217152.217152 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.5735626220703125e-05 seconds
INFO 05-06 10:42:36.219501.219501 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012755393981933594 seconds
INFO 05-06 10:42:36.220928.220928 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012404918670654297 seconds
INFO 05-06 10:42:36.221645.221645 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011680126190185547 seconds
INFO 05-06 10:42:36.223466.223466 mlpmodule.py:2799] [fused_experts] gmm total=1.745ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.223651.223651 mlpmodule.py:2799] [fused_experts] gmm total=2.029ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.224910.224910 mlpmodule.py:2799] [fused_experts] gmm total=2.463ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.224418.224418 mlpmodule.py:2799] [fused_experts] gmm total=2.877ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:36.225639.225639 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004114389419555664 seconds
INFO 05-06 10:42:36.225987.225987 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:42:36.226358.226358 cuda_h.py:27] end *layer_moe_fused cost 8.987 ms
DEBUG 05-06 10:42:36.226775.226775 cuda_h.py:27] end decode_layer cost 11.841 ms
DEBUG 05-06 10:42:36.226380.226380 cuda_h.py:27] end decode_step cost 534.195 ms
INFO 05-06 10:42:36.226043.226043 lmp.py:931] decode step 0 time: 0.5342323780059814 seconds
WARNING 05-06 10:42:36.232848.232848 helper.py:80] WARNING: Logits have extreme values: min=-896.00, max=1032.00
WARNING 05-06 10:42:36.232257.232257 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 10:42:36.233357.233357 cuda_h.py:27] end init_inputs_tokens cost 7.108 ms
DEBUG 05-06 10:42:36.234624.234624 lmp.py:899] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:42:36.234148.234148 lmp.py:904] ---- decode step 1 layer 0 ----
DEBUG 05-06 10:42:36.235278.235278 cuda_h.py:27] end *sagl cost 1.461 ms
DEBUG 05-06 10:42:36.243241.243241 cuda_h.py:27] end *layer_moe_fused cost 6.946 ms
DEBUG 05-06 10:42:36.244476.244476 cuda_h.py:27] end decode_layer cost 10.583 ms
DEBUG 05-06 10:42:36.244024.244024 lmp.py:904] ---- decode step 1 layer 1 ----
DEBUG 05-06 10:42:36.248730.248730 cuda_h.py:27] end *sagl cost 3.783 ms
DEBUG 05-06 10:42:36.254135.254135 cuda_h.py:27] end *layer_moe_fused cost 3.862 ms
DEBUG 05-06 10:42:36.255270.255270 cuda_h.py:27] end decode_layer cost 10.330 ms
DEBUG 05-06 10:42:36.255550.255550 lmp.py:904] ---- decode step 1 layer 2 ----
DEBUG 05-06 10:42:36.257068.257068 cuda_h.py:27] end *sagl cost 1.940 ms
DEBUG 05-06 10:42:36.260168.260168 cuda_h.py:27] end *layer_moe_fused cost 2.504 ms
DEBUG 05-06 10:42:36.261801.261801 cuda_h.py:27] end decode_layer cost 6.078 ms
DEBUG 05-06 10:42:36.261703.261703 lmp.py:904] ---- decode step 1 layer 3 ----
DEBUG 05-06 10:42:36.263791.263791 cuda_h.py:27] end *sagl cost 1.975 ms
DEBUG 05-06 10:42:36.267253.267253 cuda_h.py:27] end *layer_moe_fused cost 2.916 ms
DEBUG 05-06 10:42:36.267913.267913 cuda_h.py:27] end decode_layer cost 6.644 ms
DEBUG 05-06 10:42:36.268385.268385 lmp.py:904] ---- decode step 1 layer 4 ----
DEBUG 05-06 10:42:36.270862.270862 cuda_h.py:27] end *sagl cost 1.904 ms
DEBUG 05-06 10:42:36.273738.273738 cuda_h.py:27] end *layer_moe_fused cost 2.552 ms
DEBUG 05-06 10:42:36.274033.274033 cuda_h.py:27] end decode_layer cost 6.091 ms
DEBUG 05-06 10:42:36.274221.274221 lmp.py:904] ---- decode step 1 layer 5 ----
DEBUG 05-06 10:42:36.276606.276606 cuda_h.py:27] end *sagl cost 1.948 ms
DEBUG 05-06 10:42:36.279743.279743 cuda_h.py:27] end *layer_moe_fused cost 2.189 ms
DEBUG 05-06 10:42:36.280773.280773 cuda_h.py:27] end decode_layer cost 5.808 ms
DEBUG 05-06 10:42:36.280484.280484 lmp.py:904] ---- decode step 1 layer 6 ----
DEBUG 05-06 10:42:36.282550.282550 cuda_h.py:27] end *sagl cost 1.922 ms
DEBUG 05-06 10:42:36.285823.285823 cuda_h.py:27] end *layer_moe_fused cost 2.316 ms
DEBUG 05-06 10:42:36.285250.285250 cuda_h.py:27] end decode_layer cost 5.862 ms
DEBUG 05-06 10:42:36.286530.286530 lmp.py:904] ---- decode step 1 layer 7 ----
DEBUG 05-06 10:42:36.288705.288705 cuda_h.py:27] end *sagl cost 2.003 ms
DEBUG 05-06 10:42:36.291726.291726 cuda_h.py:27] end *layer_moe_fused cost 2.304 ms
DEBUG 05-06 10:42:36.291240.291240 cuda_h.py:27] end decode_layer cost 5.961 ms
DEBUG 05-06 10:42:36.292428.292428 lmp.py:904] ---- decode step 1 layer 8 ----
DEBUG 05-06 10:42:36.294242.294242 cuda_h.py:27] end *sagl cost 1.913 ms
DEBUG 05-06 10:42:36.297226.297226 cuda_h.py:27] end *layer_moe_fused cost 2.199 ms
DEBUG 05-06 10:42:36.297336.297336 cuda_h.py:27] end decode_layer cost 5.756 ms
DEBUG 05-06 10:42:36.297239.297239 lmp.py:904] ---- decode step 1 layer 9 ----
DEBUG 05-06 10:42:36.299043.299043 cuda_h.py:27] end *sagl cost 2.010 ms
DEBUG 05-06 10:42:36.303575.303575 cuda_h.py:27] end *layer_moe_fused cost 2.124 ms
DEBUG 05-06 10:42:36.303839.303839 cuda_h.py:27] end decode_layer cost 5.845 ms
DEBUG 05-06 10:42:36.303311.303311 lmp.py:904] ---- decode step 1 layer 10 ----
DEBUG 05-06 10:42:36.305947.305947 cuda_h.py:27] end *sagl cost 1.923 ms
DEBUG 05-06 10:42:36.309252.309252 cuda_h.py:27] end *layer_moe_fused cost 2.298 ms
DEBUG 05-06 10:42:36.309355.309355 cuda_h.py:27] end decode_layer cost 5.856 ms
DEBUG 05-06 10:42:36.309828.309828 lmp.py:904] ---- decode step 1 layer 11 ----
DEBUG 05-06 10:42:36.311247.311247 cuda_h.py:27] end *sagl cost 1.973 ms
DEBUG 05-06 10:42:36.315538.315538 cuda_h.py:27] end *layer_moe_fused cost 2.268 ms
DEBUG 05-06 10:42:36.315523.315523 cuda_h.py:27] end decode_layer cost 5.914 ms
DEBUG 05-06 10:42:36.315518.315518 lmp.py:904] ---- decode step 1 layer 12 ----
DEBUG 05-06 10:42:36.317777.317777 cuda_h.py:27] end *sagl cost 1.924 ms
DEBUG 05-06 10:42:36.320056.320056 cuda_h.py:27] end *layer_moe_fused cost 2.103 ms
DEBUG 05-06 10:42:36.321536.321536 cuda_h.py:27] end decode_layer cost 5.661 ms
DEBUG 05-06 10:42:36.321531.321531 lmp.py:904] ---- decode step 1 layer 13 ----
DEBUG 05-06 10:42:36.323612.323612 cuda_h.py:27] end *sagl cost 1.970 ms
DEBUG 05-06 10:42:36.326510.326510 cuda_h.py:27] end *layer_moe_fused cost 2.338 ms
DEBUG 05-06 10:42:36.327374.327374 cuda_h.py:27] end decode_layer cost 5.989 ms
DEBUG 05-06 10:42:36.327608.327608 lmp.py:904] ---- decode step 1 layer 14 ----
DEBUG 05-06 10:42:36.329312.329312 cuda_h.py:27] end *sagl cost 1.972 ms
DEBUG 05-06 10:42:36.332636.332636 cuda_h.py:27] end *layer_moe_fused cost 2.072 ms
DEBUG 05-06 10:42:36.333640.333640 cuda_h.py:27] end decode_layer cost 5.671 ms
DEBUG 05-06 10:42:36.333635.333635 lmp.py:904] ---- decode step 1 layer 15 ----
DEBUG 05-06 10:42:36.335816.335816 cuda_h.py:27] end *sagl cost 1.972 ms
DEBUG 05-06 10:42:36.338905.338905 cuda_h.py:27] end *layer_moe_fused cost 2.111 ms
DEBUG 05-06 10:42:36.339478.339478 cuda_h.py:27] end decode_layer cost 5.778 ms
DEBUG 05-06 10:42:36.339758.339758 lmp.py:904] ---- decode step 1 layer 16 ----
DEBUG 05-06 10:42:36.341567.341567 cuda_h.py:27] end *sagl cost 1.945 ms
DEBUG 05-06 10:42:36.344005.344005 cuda_h.py:27] end *layer_moe_fused cost 2.120 ms
DEBUG 05-06 10:42:36.344631.344631 cuda_h.py:27] end decode_layer cost 5.695 ms
DEBUG 05-06 10:42:36.344680.344680 lmp.py:904] ---- decode step 1 layer 17 ----
DEBUG 05-06 10:42:36.346938.346938 cuda_h.py:27] end *sagl cost 1.925 ms
DEBUG 05-06 10:42:36.350742.350742 cuda_h.py:27] end *layer_moe_fused cost 2.519 ms
DEBUG 05-06 10:42:36.351753.351753 cuda_h.py:27] end decode_layer cost 6.123 ms
DEBUG 05-06 10:42:36.351510.351510 lmp.py:904] ---- decode step 1 layer 18 ----
DEBUG 05-06 10:42:36.353896.353896 cuda_h.py:27] end *sagl cost 1.985 ms
DEBUG 05-06 10:42:36.356861.356861 cuda_h.py:27] end *layer_moe_fused cost 2.613 ms
DEBUG 05-06 10:42:36.357540.357540 cuda_h.py:27] end decode_layer cost 6.232 ms
DEBUG 05-06 10:42:36.357820.357820 lmp.py:904] ---- decode step 1 layer 19 ----
DEBUG 05-06 10:42:36.359054.359054 cuda_h.py:27] end *sagl cost 1.976 ms
DEBUG 05-06 10:42:36.362505.362505 cuda_h.py:27] end *layer_moe_fused cost 2.220 ms
DEBUG 05-06 10:42:36.363131.363131 cuda_h.py:27] end decode_layer cost 5.912 ms
DEBUG 05-06 10:42:36.363411.363411 lmp.py:904] ---- decode step 1 layer 20 ----
DEBUG 05-06 10:42:36.365644.365644 cuda_h.py:27] end *sagl cost 1.941 ms
DEBUG 05-06 10:42:36.368471.368471 cuda_h.py:27] end *layer_moe_fused cost 2.261 ms
DEBUG 05-06 10:42:36.369097.369097 cuda_h.py:27] end decode_layer cost 5.837 ms
DEBUG 05-06 10:42:36.369954.369954 lmp.py:904] ---- decode step 1 layer 21 ----
DEBUG 05-06 10:42:36.371187.371187 cuda_h.py:27] end *sagl cost 1.976 ms
DEBUG 05-06 10:42:36.374848.374848 cuda_h.py:27] end *layer_moe_fused cost 2.196 ms
DEBUG 05-06 10:42:36.375329.375329 cuda_h.py:27] end decode_layer cost 5.854 ms
DEBUG 05-06 10:42:36.375371.375371 lmp.py:904] ---- decode step 1 layer 22 ----
DEBUG 05-06 10:42:36.377611.377611 cuda_h.py:27] end *sagl cost 1.982 ms
DEBUG 05-06 10:42:36.380024.380024 cuda_h.py:27] end *layer_moe_fused cost 2.338 ms
DEBUG 05-06 10:42:36.381511.381511 cuda_h.py:27] end decode_layer cost 5.962 ms
DEBUG 05-06 10:42:36.381268.381268 lmp.py:904] ---- decode step 1 layer 23 ----
DEBUG 05-06 10:42:36.383665.383665 cuda_h.py:27] end *sagl cost 1.922 ms
DEBUG 05-06 10:42:36.386193.386193 cuda_h.py:27] end *layer_moe_fused cost 2.145 ms
DEBUG 05-06 10:42:36.387905.387905 cuda_h.py:27] end decode_layer cost 5.767 ms
DEBUG 05-06 10:42:36.387424.387424 lmp.py:904] ---- decode step 1 layer 24 ----
DEBUG 05-06 10:42:36.389517.389517 cuda_h.py:27] end *sagl cost 1.944 ms
DEBUG 05-06 10:42:36.392288.392288 cuda_h.py:27] end *layer_moe_fused cost 2.149 ms
DEBUG 05-06 10:42:36.392914.392914 cuda_h.py:27] end decode_layer cost 5.728 ms
DEBUG 05-06 10:42:36.392148.392148 lmp.py:904] ---- decode step 1 layer 25 ----
DEBUG 05-06 10:42:36.394560.394560 cuda_h.py:27] end *sagl cost 1.968 ms
DEBUG 05-06 10:42:36.398969.398969 cuda_h.py:27] end *layer_moe_fused cost 2.194 ms
DEBUG 05-06 10:42:36.398880.398880 cuda_h.py:27] end decode_layer cost 5.837 ms
DEBUG 05-06 10:42:36.398445.398445 lmp.py:904] ---- decode step 1 layer 26 ----
DEBUG 05-06 10:42:36.400930.400930 cuda_h.py:27] end *sagl cost 1.987 ms
DEBUG 05-06 10:42:36.404114.404114 cuda_h.py:27] end *layer_moe_fused cost 2.209 ms
DEBUG 05-06 10:42:36.404170.404170 cuda_h.py:27] end decode_layer cost 5.831 ms
DEBUG 05-06 10:42:36.404689.404689 lmp.py:904] ---- decode step 1 layer 27 ----
DEBUG 05-06 10:42:36.406154.406154 cuda_h.py:27] end *sagl cost 1.972 ms
DEBUG 05-06 10:42:36.410522.410522 cuda_h.py:27] end *layer_moe_fused cost 2.889 ms
DEBUG 05-06 10:42:36.411261.411261 cuda_h.py:27] end decode_layer cost 6.596 ms
DEBUG 05-06 10:42:36.411733.411733 lmp.py:904] ---- decode step 1 layer 28 ----
DEBUG 05-06 10:42:36.413622.413622 cuda_h.py:27] end *sagl cost 1.968 ms
DEBUG 05-06 10:42:36.416116.416116 cuda_h.py:27] end *layer_moe_fused cost 2.184 ms
DEBUG 05-06 10:42:36.417504.417504 cuda_h.py:27] end decode_layer cost 5.795 ms
DEBUG 05-06 10:42:36.417738.417738 lmp.py:904] ---- decode step 1 layer 29 ----
DEBUG 05-06 10:42:36.419202.419202 cuda_h.py:27] end *sagl cost 1.935 ms
DEBUG 05-06 10:42:36.422623.422623 cuda_h.py:27] end *layer_moe_fused cost 2.327 ms
DEBUG 05-06 10:42:36.423593.423593 cuda_h.py:27] end decode_layer cost 5.962 ms
DEBUG 05-06 10:42:36.423410.423410 cuda_h.py:27] end decode_step cost 196.406 ms
INFO 05-06 10:42:36.423604.423604 lmp.py:931] decode step 1 time: 0.19644546508789062 seconds
Time taken: 6.833728291094303 seconds
generate input ids cost 0.0794825553894043 s
DEBUG 05-06 10:42:39.168904.168904 cuda_h.py:27] end generate_input_ids cost 2593.044 ms
DEBUG 05-06 10:42:39.168022.168022 cuda_h.py:27] end init_cache cost 0.038 ms
INFO 05-06 10:42:39.180538.180538 lmp.py:2341] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6386589636, 'cuda:1': 12831555584, 'cuda:2': 12808486912, 'cuda:3': 12829458432} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7346912339729151, 'cuda:1': 0.4708756492646622, 'cuda:2': 0.4713240027915884, 'cuda:3': 0.4709163734250085}
INFO 05-06 10:42:39.180860.180860 lmp.py:2359] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.180444.180444 lmp.py:2359] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.180114.180114 lmp.py:2359] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.180638.180638 lmp.py:2359] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.180447.180447 lmp.py:2359] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.181854.181854 lmp.py:2359] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.181174.181174 lmp.py:2359] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.181699.181699 lmp.py:2359] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.181952.181952 lmp.py:2359] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182964.182964 lmp.py:2359] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182250.182250 lmp.py:2359] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182037.182037 lmp.py:2359] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182515.182515 lmp.py:2359] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182662.182662 lmp.py:2359] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182257.182257 lmp.py:2359] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182020.182020 lmp.py:2359] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182352.182352 lmp.py:2359] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182185.182185 lmp.py:2359] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182756.182756 lmp.py:2359] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.182067.182067 lmp.py:2359] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183161.183161 lmp.py:2359] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183220.183220 lmp.py:2359] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183029.183029 lmp.py:2359] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183315.183315 lmp.py:2359] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183663.183663 lmp.py:2359] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183188.183188 lmp.py:2359] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183997.183997 lmp.py:2359] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183228.183228 lmp.py:2359] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183706.183706 lmp.py:2359] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:42:39.183515.183515 lmp.py:2359] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 10:42:39.466524.466524 cuda_h.py:27] end init_loading_placement cost 298.347 ms
DEBUG 05-06 10:42:39.466296.466296 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:42:39.466894.466894 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:42:39 client.py:72] load_into_gpu: gemma4-26B-A4B, 7f6e75fc-22c6-4e34-a7fa-51419960e984
INFO 05-06 10:42:39 client.py:135] Model loaded: gemma4-26B-A4B, 7f6e75fc-22c6-4e34-a7fa-51419960e984
INFO 05-06 10:42:39 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 7f6e75fc-22c6-4e34-a7fa-51419960e984
INFO 05-06 10:42:39 client.py:212] Model loaded
DEBUG 05-06 10:42:39.994790.994790 cuda_h.py:27] end init_general_sagl_loading_async cost 527.590 ms
INFO 05-06 10:42:40.043237.043237 lmp.py:2862] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 10:42:40.142020.142020 cuda_h.py:27] end restore_state_dict cost 99.001 ms
DEBUG 05-06 10:42:40.142428.142428 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:42:40.142596.142596 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:42:40 client.py:72] load_into_gpu: gemma4-26B-A4B, aa21ffd4-4718-424d-a81a-c23144d37003
INFO 05-06 10:42:40 client.py:135] Model loaded: gemma4-26B-A4B, aa21ffd4-4718-424d-a81a-c23144d37003
DEBUG 05-06 10:42:40.273128.273128 cuda_h.py:27] end init_experts_loading_async cost 130.446 ms
DEBUG 05-06 10:42:40.274842.274842 cuda_h.py:27] end init_inputs_tokens cost 0.967 ms
DEBUG 05-06 10:42:40.274768.274768 lmp.py:824] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 10:42:40.280945.280945 cuda_h.py:27] end *sagl cost 5.718 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 29, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 5090, 'token_per_expert': {3: 160, 7: 374, 11: 31, 15: 13, 19: 2, 23: 23, 27: 10, 31: 134, 39: 718, 47: 1304, 51: 186, 55: 208, 59: 43, 63: 15, 67: 183, 71: 65, 75: 89, 79: 76, 83: 105, 87: 2, 91: 458, 99: 161, 103: 432, 107: 25, 111: 23, 115: 89, 119: 13, 123: 39, 127: 109}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 48, 52, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3457, 'token_per_expert': {0: 249, 4: 11, 8: 9, 12: 2, 16: 201, 20: 15, 24: 81, 28: 123, 32: 183, 36: 4, 44: 17, 48: 146, 52: 150, 60: 55, 64: 106, 68: 694, 72: 100, 76: 74, 80: 28, 84: 21, 88: 1, 92: 87, 96: 8, 100: 5, 104: 134, 108: 78, 112: 68, 116: 82, 120: 1, 124: 724}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 46, 50, 54, 66, 70, 74, 78, 86, 90, 94, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 27, 'ideal_gpu_count': 29, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 4060, 'token_per_expert': {2: 30, 6: 4, 10: 36, 14: 29, 18: 71, 22: 255, 26: 304, 30: 1, 34: 38, 38: 59, 46: 450, 50: 520, 54: 275, 66: 6, 70: 140, 74: 224, 78: 109, 86: 2, 90: 546, 94: 24, 102: 74, 106: 14, 110: 83, 114: 48, 118: 89, 122: 114, 126: 515}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3777, 'token_per_expert': {1: 273, 5: 66, 9: 68, 13: 61, 17: 3, 21: 171, 25: 110, 29: 7, 33: 828, 37: 81, 41: 142, 45: 14, 49: 17, 53: 819, 65: 39, 69: 78, 73: 60, 77: 99, 81: 11, 85: 3, 89: 133, 93: 12, 97: 3, 101: 6, 105: 89, 109: 3, 113: 157, 117: 97, 121: 226, 125: 101}}
INFO 05-06 10:42:40.284371.284371 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 1.054ms | allocate_experts_across_cpu_gpu: 0.564ms
INFO 05-06 10:42:40.284459.284459 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 0.0001266002655029297 seconds
INFO 05-06 10:42:40.286976.286976 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=232 time: 0.0015838146209716797 seconds
INFO 05-06 10:42:40.299907.299907 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.013259410858154297 seconds
INFO 05-06 10:42:40.391583.391583 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.09149622917175293 seconds
INFO 05-06 10:42:40.395823.395823 mlpmodule.py:2799] [fused_experts] gmm total=4.147ms E=32 S=5090 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.397201.397201 mlpmodule.py:2799] [fused_experts] gmm total=5.308ms E=32 S=4060 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.397658.397658 mlpmodule.py:2799] [fused_experts] gmm total=5.988ms E=32 S=3457 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.398387.398387 mlpmodule.py:2799] [fused_experts] gmm total=6.807ms E=32 S=3777 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.400412.400412 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00935983657836914 seconds
INFO 05-06 10:42:40.401938.401938 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00011730194091796875 seconds
DEBUG 05-06 10:42:40.402202.402202 cuda_h.py:27] end *layer_moe_fused cost 119.576 ms
DEBUG 05-06 10:42:40.419525.419525 cuda_h.py:27] end prefill_layer cost 145.646 ms
DEBUG 05-06 10:42:40.420396.420396 lmp.py:841] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 10:42:40.420596.420596 lmp.py:824] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 10:42:40.421141.421141 cuda_h.py:27] end *sagl cost 1.794 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 75, 79, 83, 87, 91, 95, 99, 103, 107, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 2697, 'token_per_expert': {3: 62, 7: 155, 11: 49, 15: 21, 23: 6, 27: 33, 31: 15, 35: 64, 39: 5, 43: 5, 47: 157, 51: 234, 55: 15, 59: 152, 63: 3, 67: 446, 75: 7, 79: 75, 83: 35, 87: 20, 91: 14, 95: 62, 99: 549, 103: 22, 107: 5, 115: 8, 119: 151, 123: 37, 127: 290}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4457, 'token_per_expert': {0: 23, 4: 81, 8: 472, 12: 219, 16: 6, 20: 207, 24: 3, 28: 192, 32: 26, 40: 20, 44: 6, 48: 39, 52: 1230, 56: 50, 60: 25, 64: 91, 68: 693, 72: 30, 76: 29, 80: 148, 84: 26, 88: 26, 92: 67, 96: 203, 100: 203, 104: 63, 108: 50, 112: 20, 116: 24, 120: 102, 124: 83}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 62, 66, 70, 74, 78, 82, 90, 94, 98, 106, 110, 114, 118, 122], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 5166, 'token_per_expert': {2: 8, 6: 35, 10: 623, 14: 15, 18: 12, 22: 467, 26: 17, 30: 970, 34: 65, 38: 40, 42: 160, 46: 196, 50: 49, 54: 231, 62: 24, 66: 29, 70: 2, 74: 63, 78: 29, 82: 791, 90: 61, 94: 118, 98: 62, 106: 213, 110: 24, 114: 5, 118: 290, 122: 567}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4064, 'token_per_expert': {1: 143, 5: 406, 9: 102, 13: 1122, 21: 71, 25: 164, 29: 27, 33: 5, 37: 36, 41: 10, 45: 46, 49: 104, 53: 158, 57: 28, 61: 2, 65: 154, 69: 40, 73: 96, 77: 16, 81: 10, 85: 103, 89: 14, 93: 25, 97: 487, 101: 52, 105: 70, 109: 531, 117: 2, 121: 32, 125: 8}}
INFO 05-06 10:42:40.424385.424385 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.979ms | allocate_experts_across_cpu_gpu: 0.256ms
INFO 05-06 10:42:40.424409.424409 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.198883056640625e-05 seconds
INFO 05-06 10:42:40.426217.426217 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0020394325256347656 seconds
INFO 05-06 10:42:40.456979.456979 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.029912471771240234 seconds
INFO 05-06 10:42:40.458754.458754 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017232894897460938 seconds
INFO 05-06 10:42:40.462911.462911 mlpmodule.py:2799] [fused_experts] gmm total=3.922ms E=32 S=2697 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.463087.463087 mlpmodule.py:2799] [fused_experts] gmm total=4.337ms E=32 S=4457 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.463504.463504 mlpmodule.py:2799] [fused_experts] gmm total=4.598ms E=32 S=5166 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.464280.464280 mlpmodule.py:2799] [fused_experts] gmm total=4.930ms E=32 S=4064 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.466757.466757 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007653713226318359 seconds
INFO 05-06 10:42:40.466966.466966 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.340576171875e-05 seconds
DEBUG 05-06 10:42:40.466021.466021 cuda_h.py:27] end *layer_moe_fused cost 43.771 ms
DEBUG 05-06 10:42:40.489022.489022 cuda_h.py:27] end prefill_layer cost 69.478 ms
DEBUG 05-06 10:42:40.489156.489156 lmp.py:841] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 10:42:40.489581.489581 lmp.py:824] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 10:42:40.491397.491397 cuda_h.py:27] end *sagl cost 1.589 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4795, 'token_per_expert': {3: 140, 7: 258, 11: 1055, 15: 370, 19: 585, 23: 27, 27: 47, 31: 99, 35: 28, 43: 75, 47: 1, 51: 103, 55: 219, 59: 445, 63: 91, 67: 6, 71: 68, 75: 6, 79: 4, 83: 96, 87: 1, 91: 149, 95: 28, 99: 6, 103: 69, 107: 79, 111: 46, 115: 39, 119: 81, 123: 104, 127: 470}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 96, 100, 104, 108, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3673, 'token_per_expert': {0: 50, 4: 81, 8: 117, 12: 18, 16: 4, 20: 216, 24: 70, 28: 55, 32: 3, 36: 92, 40: 22, 44: 100, 48: 234, 52: 52, 56: 60, 60: 215, 64: 17, 68: 6, 72: 51, 76: 269, 80: 234, 84: 228, 88: 46, 96: 34, 100: 72, 104: 151, 108: 983, 116: 72, 120: 34, 124: 87}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 14, 18, 22, 26, 34, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 27, 'ideal_gpu_count': 29, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 3299, 'token_per_expert': {6: 25, 14: 124, 18: 169, 22: 1, 26: 12, 34: 150, 42: 37, 46: 46, 50: 12, 54: 392, 58: 42, 62: 566, 66: 9, 70: 68, 74: 1, 78: 206, 82: 25, 86: 5, 90: 252, 98: 88, 102: 328, 106: 181, 110: 100, 114: 28, 118: 219, 122: 108, 126: 105}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 77, 81, 85, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4617, 'token_per_expert': {1: 412, 5: 21, 9: 430, 13: 402, 17: 74, 21: 10, 25: 8, 29: 311, 33: 78, 37: 273, 41: 542, 45: 17, 49: 116, 53: 186, 57: 123, 61: 30, 65: 180, 69: 109, 77: 100, 81: 389, 85: 65, 93: 1, 97: 137, 101: 2, 105: 34, 109: 145, 113: 42, 117: 1, 121: 29, 125: 350}}
INFO 05-06 10:42:40.493586.493586 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 1.276ms | allocate_experts_across_cpu_gpu: 0.254ms
INFO 05-06 10:42:40.493604.493604 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.6743621826171875e-05 seconds
INFO 05-06 10:42:40.495899.495899 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001852273941040039 seconds
INFO 05-06 10:42:40.522712.522712 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02630019187927246 seconds
INFO 05-06 10:42:40.524190.524190 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017480850219726562 seconds
INFO 05-06 10:42:40.528654.528654 mlpmodule.py:2799] [fused_experts] gmm total=3.969ms E=32 S=3299 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.528883.528883 mlpmodule.py:2799] [fused_experts] gmm total=4.379ms E=32 S=3673 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.529233.529233 mlpmodule.py:2799] [fused_experts] gmm total=4.666ms E=32 S=4795 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.530816.530816 mlpmodule.py:2799] [fused_experts] gmm total=5.354ms E=32 S=4617 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.532920.532920 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00860595703125 seconds
INFO 05-06 10:42:40.532687.532687 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.91278076171875e-05 seconds
DEBUG 05-06 10:42:40.533386.533386 cuda_h.py:27] end *layer_moe_fused cost 41.025 ms
DEBUG 05-06 10:42:40.551467.551467 cuda_h.py:27] end prefill_layer cost 62.057 ms
DEBUG 05-06 10:42:40.551549.551549 lmp.py:841] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 10:42:40.551444.551444 lmp.py:824] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 10:42:40.553425.553425 cuda_h.py:27] end *sagl cost 1.569 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 23, 27, 31, 35, 39, 43, 51, 55, 59, 63, 67, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 31, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 2776, 'token_per_expert': {3: 108, 11: 130, 15: 188, 19: 108, 23: 17, 27: 26, 31: 64, 35: 7, 39: 86, 43: 82, 51: 163, 55: 16, 59: 98, 63: 86, 67: 51, 71: 327, 75: 394, 83: 182, 87: 7, 91: 19, 95: 198, 99: 1, 103: 1, 107: 110, 111: 65, 115: 9, 119: 103, 123: 82, 127: 48}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 116, 120], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3932, 'token_per_expert': {0: 158, 4: 291, 8: 51, 12: 3, 16: 32, 20: 6, 24: 42, 28: 251, 32: 16, 36: 6, 40: 55, 44: 99, 48: 43, 52: 282, 56: 39, 60: 37, 64: 103, 68: 210, 72: 25, 76: 249, 80: 6, 84: 244, 88: 301, 92: 282, 96: 319, 100: 72, 104: 236, 108: 160, 116: 51, 120: 263}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5806, 'token_per_expert': {2: 114, 6: 80, 10: 167, 14: 312, 18: 9, 22: 540, 26: 89, 30: 41, 34: 265, 38: 2, 42: 30, 46: 2, 50: 687, 54: 196, 58: 106, 62: 430, 66: 342, 70: 147, 74: 161, 78: 699, 82: 19, 86: 34, 94: 11, 98: 19, 102: 425, 106: 4, 110: 109, 114: 125, 118: 249, 122: 391, 126: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3870, 'token_per_expert': {1: 35, 5: 290, 9: 320, 13: 71, 17: 179, 21: 4, 25: 179, 29: 6, 33: 53, 37: 7, 41: 32, 45: 2, 53: 251, 57: 25, 61: 72, 65: 16, 69: 155, 73: 251, 77: 83, 81: 1, 85: 545, 89: 22, 93: 347, 97: 266, 101: 85, 105: 2, 109: 78, 113: 1, 117: 44, 121: 442, 125: 6}}
INFO 05-06 10:42:40.555688.555688 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.940ms | allocate_experts_across_cpu_gpu: 0.268ms
INFO 05-06 10:42:40.555487.555487 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.270408630371094e-05 seconds
INFO 05-06 10:42:40.557230.557230 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015988349914550781 seconds
INFO 05-06 10:42:40.588217.588217 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.031177282333374023 seconds
INFO 05-06 10:42:40.590493.590493 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018634796142578125 seconds
INFO 05-06 10:42:40.594264.594264 mlpmodule.py:2799] [fused_experts] gmm total=3.300ms E=32 S=3932 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.594512.594512 mlpmodule.py:2799] [fused_experts] gmm total=3.677ms E=32 S=2776 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.595978.595978 mlpmodule.py:2799] [fused_experts] gmm total=4.683ms E=32 S=5806 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.596456.596456 mlpmodule.py:2799] [fused_experts] gmm total=5.103ms E=32 S=3870 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.598963.598963 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007452964782714844 seconds
INFO 05-06 10:42:40.598450.598450 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:42:40.598869.598869 cuda_h.py:27] end *layer_moe_fused cost 44.005 ms
DEBUG 05-06 10:42:40.619953.619953 cuda_h.py:27] end prefill_layer cost 67.607 ms
DEBUG 05-06 10:42:40.619796.619796 lmp.py:841] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 10:42:40.619022.619022 lmp.py:824] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 10:42:40.621003.621003 cuda_h.py:27] end *sagl cost 1.570 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 6964, 'token_per_expert': {3: 153, 7: 30, 15: 55, 19: 168, 23: 466, 27: 274, 31: 12, 35: 1, 39: 207, 43: 564, 47: 183, 51: 295, 55: 241, 59: 639, 63: 1030, 67: 275, 71: 116, 75: 64, 79: 9, 83: 429, 87: 109, 91: 71, 95: 4, 103: 34, 107: 64, 111: 451, 115: 398, 119: 505, 123: 109, 127: 8}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 116, 120, 124], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 3064, 'token_per_expert': {4: 140, 8: 711, 12: 11, 16: 3, 20: 137, 24: 310, 28: 124, 32: 151, 36: 60, 40: 52, 44: 29, 48: 2, 52: 82, 56: 34, 60: 128, 64: 61, 72: 13, 76: 209, 80: 19, 84: 51, 88: 41, 92: 141, 96: 106, 100: 1, 104: 123, 108: 84, 116: 98, 120: 14, 124: 129}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3181, 'token_per_expert': {2: 5, 6: 32, 14: 2, 18: 34, 22: 412, 26: 351, 30: 81, 34: 35, 38: 31, 42: 1, 46: 24, 50: 10, 54: 346, 58: 12, 62: 93, 66: 16, 70: 1, 74: 409, 78: 96, 82: 259, 86: 110, 90: 43, 94: 83, 98: 77, 102: 1, 106: 505, 110: 7, 114: 15, 118: 52, 122: 24, 126: 14}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3175, 'token_per_expert': {1: 296, 5: 184, 9: 2, 13: 1, 17: 103, 21: 26, 25: 52, 29: 205, 37: 33, 41: 15, 45: 58, 49: 78, 53: 253, 57: 53, 61: 106, 65: 4, 69: 27, 73: 37, 77: 52, 81: 65, 85: 123, 89: 451, 93: 196, 97: 117, 101: 52, 105: 108, 109: 46, 113: 247, 117: 73, 121: 5, 125: 107}}
INFO 05-06 10:42:40.623046.623046 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 1.287ms | allocate_experts_across_cpu_gpu: 0.272ms
INFO 05-06 10:42:40.623905.623905 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.079673767089844e-05 seconds
INFO 05-06 10:42:40.625531.625531 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014736652374267578 seconds
INFO 05-06 10:42:40.647732.647732 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.021960973739624023 seconds
INFO 05-06 10:42:40.649232.649232 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014281272888183594 seconds
INFO 05-06 10:42:40.653941.653941 mlpmodule.py:2799] [fused_experts] gmm total=3.628ms E=32 S=3064 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.653602.653602 mlpmodule.py:2799] [fused_experts] gmm total=4.070ms E=32 S=6964 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.653753.653753 mlpmodule.py:2799] [fused_experts] gmm total=3.944ms E=32 S=3175 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.654558.654558 mlpmodule.py:2799] [fused_experts] gmm total=4.750ms E=32 S=3181 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.656472.656472 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0068891048431396484 seconds
INFO 05-06 10:42:40.656866.656866 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9591064453125e-05 seconds
DEBUG 05-06 10:42:40.656739.656739 cuda_h.py:27] end *layer_moe_fused cost 34.516 ms
DEBUG 05-06 10:42:40.672146.672146 cuda_h.py:27] end prefill_layer cost 52.746 ms
DEBUG 05-06 10:42:40.672745.672745 lmp.py:841] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 10:42:40.672117.672117 lmp.py:824] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 10:42:40.676899.676899 cuda_h.py:27] end *sagl cost 3.733 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 51, 55, 63, 67, 71, 75, 79, 83, 87, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 28, 'ideal_gpu_count': 30, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 4146, 'token_per_expert': {3: 3, 7: 27, 11: 3, 15: 10, 19: 24, 23: 161, 27: 20, 31: 54, 39: 576, 43: 112, 51: 12, 55: 80, 63: 119, 67: 101, 71: 1122, 75: 113, 79: 63, 83: 54, 87: 188, 95: 2, 99: 305, 103: 1, 107: 34, 111: 271, 115: 18, 119: 81, 123: 197, 127: 395}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 5017, 'token_per_expert': {0: 75, 4: 197, 8: 3, 16: 450, 20: 614, 24: 201, 28: 225, 32: 16, 36: 357, 40: 1, 44: 69, 48: 7, 52: 77, 56: 9, 60: 183, 64: 450, 68: 48, 72: 285, 76: 120, 80: 113, 84: 47, 88: 245, 92: 10, 96: 126, 100: 85, 104: 189, 112: 515, 116: 161, 120: 138, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 29, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 2694, 'token_per_expert': {2: 422, 6: 44, 10: 33, 14: 53, 18: 81, 22: 393, 26: 27, 30: 14, 34: 28, 38: 10, 42: 255, 46: 156, 50: 18, 54: 24, 58: 20, 62: 12, 66: 1, 70: 198, 74: 160, 78: 5, 82: 3, 86: 32, 90: 1, 94: 229, 98: 36, 102: 27, 106: 64, 110: 6, 114: 53, 118: 128, 122: 10, 126: 151}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 37, 41, 45, 49, 53, 57, 61, 69, 73, 77, 81, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 4527, 'token_per_expert': {1: 3, 5: 150, 9: 214, 13: 243, 17: 12, 21: 5, 29: 172, 33: 489, 37: 33, 41: 9, 45: 4, 49: 657, 53: 9, 57: 35, 61: 324, 69: 1, 73: 208, 77: 25, 81: 17, 93: 148, 97: 11, 101: 1265, 105: 36, 109: 1, 113: 73, 117: 306, 121: 1, 125: 76}}
INFO 05-06 10:42:40.680552.680552 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 2.638ms | allocate_experts_across_cpu_gpu: 0.278ms
INFO 05-06 10:42:40.680583.680583 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.198883056640625e-05 seconds
INFO 05-06 10:42:40.681991.681991 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014564990997314453 seconds
INFO 05-06 10:42:40.709995.709995 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0277557373046875 seconds
INFO 05-06 10:42:40.711827.711827 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016262531280517578 seconds
INFO 05-06 10:42:40.715571.715571 mlpmodule.py:2799] [fused_experts] gmm total=3.662ms E=32 S=4146 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.715916.715916 mlpmodule.py:2799] [fused_experts] gmm total=3.880ms E=32 S=2694 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.716820.716820 mlpmodule.py:2799] [fused_experts] gmm total=4.852ms E=32 S=4527 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.716927.716927 mlpmodule.py:2799] [fused_experts] gmm total=5.281ms E=32 S=5017 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.719192.719192 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007830381393432617 seconds
INFO 05-06 10:42:40.719978.719978 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:42:40.719304.719304 cuda_h.py:27] end *layer_moe_fused cost 42.588 ms
DEBUG 05-06 10:42:40.739604.739604 cuda_h.py:27] end prefill_layer cost 66.749 ms
DEBUG 05-06 10:42:40.739679.739679 lmp.py:841] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 10:42:40.739905.739905 lmp.py:824] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 10:42:40.740322.740322 cuda_h.py:27] end *sagl cost 1.538 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 47, 51, 55, 59, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4034, 'token_per_expert': {3: 33, 7: 7, 11: 30, 15: 15, 19: 33, 23: 358, 27: 94, 31: 8, 35: 541, 43: 42, 47: 15, 51: 162, 55: 2, 59: 11, 67: 13, 71: 125, 75: 192, 79: 200, 83: 10, 87: 359, 91: 30, 95: 90, 99: 825, 103: 61, 107: 136, 111: 25, 115: 318, 119: 150, 123: 100, 127: 49}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3739, 'token_per_expert': {0: 60, 4: 7, 8: 4, 16: 14, 20: 34, 24: 174, 28: 85, 32: 148, 36: 152, 40: 13, 44: 95, 52: 9, 56: 91, 60: 28, 64: 503, 68: 1238, 72: 13, 76: 40, 80: 44, 84: 2, 88: 1, 92: 6, 96: 163, 100: 3, 104: 207, 108: 483, 112: 6, 116: 84, 120: 17, 124: 15}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4542, 'token_per_expert': {2: 98, 6: 111, 10: 107, 14: 54, 18: 18, 22: 16, 26: 141, 30: 50, 34: 342, 38: 11, 42: 57, 46: 159, 50: 138, 54: 1, 58: 93, 62: 138, 66: 4, 70: 83, 74: 31, 78: 208, 82: 39, 86: 516, 90: 426, 94: 271, 98: 258, 102: 527, 106: 396, 110: 63, 114: 13, 122: 111, 126: 62}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 49, 53, 57, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4069, 'token_per_expert': {1: 92, 5: 72, 9: 162, 13: 284, 17: 11, 21: 6, 25: 813, 29: 15, 33: 4, 37: 43, 41: 49, 49: 2, 53: 423, 57: 58, 65: 363, 69: 101, 73: 81, 77: 79, 81: 9, 85: 66, 89: 56, 93: 641, 97: 6, 101: 13, 105: 69, 109: 17, 113: 120, 117: 198, 121: 155, 125: 61}}
INFO 05-06 10:42:40.743684.743684 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 1.366ms | allocate_experts_across_cpu_gpu: 0.247ms
INFO 05-06 10:42:40.743833.743833 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.1975250244140625e-05 seconds
INFO 05-06 10:42:40.745882.745882 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001600503921508789 seconds
INFO 05-06 10:42:40.772852.772852 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.027434587478637695 seconds
INFO 05-06 10:42:40.774910.774910 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001859426498413086 seconds
INFO 05-06 10:42:40.778497.778497 mlpmodule.py:2799] [fused_experts] gmm total=3.715ms E=32 S=4034 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.779880.779880 mlpmodule.py:2799] [fused_experts] gmm total=3.771ms E=32 S=4069 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.779688.779688 mlpmodule.py:2799] [fused_experts] gmm total=4.180ms E=32 S=3739 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.780854.780854 mlpmodule.py:2799] [fused_experts] gmm total=5.077ms E=32 S=4542 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.781603.781603 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0067217350006103516 seconds
INFO 05-06 10:42:40.781912.781912 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.1975250244140625e-05 seconds
DEBUG 05-06 10:42:40.782849.782849 cuda_h.py:27] end *layer_moe_fused cost 40.360 ms
DEBUG 05-06 10:42:40.802831.802831 cuda_h.py:27] end prefill_layer cost 63.667 ms
DEBUG 05-06 10:42:40.802198.802198 lmp.py:841] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 10:42:40.803377.803377 lmp.py:824] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 10:42:40.804313.804313 cuda_h.py:27] end *sagl cost 1.607 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 2989, 'token_per_expert': {3: 4, 7: 154, 11: 8, 15: 52, 19: 85, 23: 55, 27: 2, 31: 14, 35: 31, 39: 7, 43: 136, 47: 113, 51: 95, 55: 28, 59: 137, 63: 28, 67: 20, 71: 111, 75: 3, 79: 211, 83: 113, 87: 93, 91: 857, 95: 76, 99: 98, 103: 167, 107: 15, 111: 53, 115: 115, 119: 5, 123: 72, 127: 31}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 48, 52, 56, 60, 64, 68, 72, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4724, 'token_per_expert': {0: 34, 4: 378, 8: 87, 12: 434, 16: 39, 20: 289, 24: 5, 28: 222, 32: 28, 36: 17, 44: 247, 48: 193, 52: 350, 56: 186, 60: 172, 64: 72, 68: 87, 72: 127, 80: 51, 84: 325, 88: 34, 92: 7, 96: 170, 100: 14, 104: 151, 108: 543, 112: 122, 116: 45, 120: 292, 124: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3578, 'token_per_expert': {6: 76, 10: 272, 14: 253, 18: 140, 22: 79, 26: 35, 30: 16, 34: 400, 38: 12, 42: 224, 46: 1, 50: 13, 54: 51, 58: 3, 62: 16, 66: 49, 70: 347, 78: 24, 82: 47, 86: 232, 90: 329, 94: 14, 98: 64, 102: 7, 106: 184, 110: 261, 114: 251, 118: 68, 122: 66, 126: 44}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 85, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 5093, 'token_per_expert': {1: 4, 5: 169, 9: 92, 13: 127, 17: 34, 21: 45, 25: 56, 29: 488, 33: 106, 37: 15, 41: 50, 45: 58, 49: 22, 53: 234, 57: 185, 61: 138, 65: 259, 69: 350, 73: 20, 77: 33, 85: 292, 97: 992, 101: 70, 105: 127, 109: 21, 113: 230, 117: 83, 121: 611, 125: 182}}
INFO 05-06 10:42:40.806601.806601 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.856ms | allocate_experts_across_cpu_gpu: 0.257ms
INFO 05-06 10:42:40.806618.806618 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-06 10:42:40.808321.808321 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0019404888153076172 seconds
INFO 05-06 10:42:40.839961.839961 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.030069589614868164 seconds
INFO 05-06 10:42:40.840666.840666 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001405477523803711 seconds
INFO 05-06 10:42:40.844047.844047 mlpmodule.py:2799] [fused_experts] gmm total=3.924ms E=32 S=4724 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.845412.845412 mlpmodule.py:2799] [fused_experts] gmm total=4.255ms E=32 S=2989 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.845148.845148 mlpmodule.py:2799] [fused_experts] gmm total=4.271ms E=32 S=3578 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.847445.847445 mlpmodule.py:2799] [fused_experts] gmm total=6.111ms E=32 S=5093 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.848413.848413 lmp.py:1484] [layer_moe_fused] experts compute time: 0.008389949798583984 seconds
INFO 05-06 10:42:40.849900.849900 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.839897155761719e-05 seconds
DEBUG 05-06 10:42:40.849027.849027 cuda_h.py:27] end *layer_moe_fused cost 43.705 ms
DEBUG 05-06 10:42:40.871356.871356 cuda_h.py:27] end prefill_layer cost 67.989 ms
DEBUG 05-06 10:42:40.871292.871292 lmp.py:841] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 10:42:40.871995.871995 lmp.py:824] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 10:42:40.872997.872997 cuda_h.py:27] end *sagl cost 1.622 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4226, 'token_per_expert': {3: 125, 7: 11, 11: 65, 15: 223, 19: 293, 23: 9, 27: 174, 31: 116, 35: 9, 39: 13, 43: 35, 47: 67, 51: 642, 55: 169, 59: 3, 63: 156, 67: 5, 71: 212, 75: 281, 79: 11, 83: 2, 87: 419, 91: 31, 99: 39, 103: 775, 107: 8, 111: 76, 119: 27, 123: 158, 127: 72}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2944, 'token_per_expert': {0: 7, 4: 49, 8: 29, 12: 156, 16: 103, 20: 89, 24: 11, 28: 405, 32: 273, 36: 175, 40: 5, 44: 113, 48: 23, 52: 115, 56: 239, 60: 3, 64: 33, 68: 34, 72: 13, 76: 138, 80: 253, 84: 41, 92: 23, 96: 25, 100: 9, 104: 18, 108: 66, 112: 1, 116: 39, 120: 361, 124: 95}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 5318, 'token_per_expert': {2: 214, 6: 149, 10: 66, 14: 66, 18: 8, 22: 88, 26: 25, 34: 24, 38: 244, 42: 77, 46: 256, 50: 337, 54: 860, 58: 979, 62: 28, 66: 67, 70: 401, 74: 36, 82: 31, 86: 42, 90: 12, 98: 106, 102: 225, 106: 45, 110: 341, 114: 292, 118: 21, 122: 173, 126: 105}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3896, 'token_per_expert': {1: 53, 5: 163, 9: 28, 13: 24, 17: 71, 21: 139, 25: 27, 29: 114, 33: 26, 37: 26, 41: 99, 45: 87, 49: 45, 53: 77, 57: 66, 61: 122, 65: 277, 69: 201, 73: 528, 77: 80, 81: 167, 85: 125, 89: 74, 93: 135, 101: 21, 105: 372, 109: 1, 113: 72, 117: 33, 121: 351, 125: 292}}
INFO 05-06 10:42:40.875716.875716 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 1.310ms | allocate_experts_across_cpu_gpu: 0.252ms
INFO 05-06 10:42:40.875687.875687 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.793571472167969e-05 seconds
INFO 05-06 10:42:40.877118.877118 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017404556274414062 seconds
INFO 05-06 10:42:40.903122.903122 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.026686906814575195 seconds
INFO 05-06 10:42:40.905866.905866 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015518665313720703 seconds
INFO 05-06 10:42:40.909214.909214 mlpmodule.py:2799] [fused_experts] gmm total=3.887ms E=32 S=4226 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.910383.910383 mlpmodule.py:2799] [fused_experts] gmm total=4.088ms E=32 S=2944 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.910066.910066 mlpmodule.py:2799] [fused_experts] gmm total=4.201ms E=32 S=5318 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.911389.911389 mlpmodule.py:2799] [fused_experts] gmm total=4.877ms E=32 S=3896 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.912503.912503 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0066068172454833984 seconds
INFO 05-06 10:42:40.912799.912799 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.318092346191406e-05 seconds
DEBUG 05-06 10:42:40.913366.913366 cuda_h.py:27] end *layer_moe_fused cost 39.365 ms
DEBUG 05-06 10:42:40.933756.933756 cuda_h.py:27] end prefill_layer cost 62.667 ms
DEBUG 05-06 10:42:40.933214.933214 lmp.py:841] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 10:42:40.933202.933202 lmp.py:824] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 10:42:40.935096.935096 cuda_h.py:27] end *sagl cost 1.542 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 4383, 'token_per_expert': {3: 91, 7: 112, 11: 6, 15: 90, 19: 127, 23: 212, 27: 136, 31: 4, 39: 157, 43: 437, 47: 5, 51: 133, 55: 14, 59: 6, 63: 4, 67: 34, 71: 176, 75: 408, 79: 26, 83: 143, 95: 1097, 99: 115, 103: 479, 107: 1, 111: 133, 115: 36, 119: 12, 123: 16, 127: 173}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3925, 'token_per_expert': {0: 14, 4: 190, 8: 27, 12: 718, 16: 427, 20: 29, 24: 140, 28: 18, 32: 175, 36: 202, 40: 243, 44: 37, 48: 257, 52: 33, 56: 315, 64: 15, 68: 109, 72: 151, 76: 201, 80: 59, 84: 4, 88: 115, 92: 220, 96: 11, 100: 14, 104: 27, 108: 2, 112: 13, 116: 43, 120: 23, 124: 93}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4329, 'token_per_expert': {2: 3, 6: 20, 10: 41, 14: 3, 18: 30, 22: 184, 26: 20, 30: 161, 34: 23, 38: 132, 42: 66, 46: 842, 50: 15, 54: 141, 58: 24, 62: 88, 66: 12, 70: 784, 74: 362, 82: 46, 86: 105, 90: 23, 94: 6, 98: 54, 102: 183, 106: 820, 110: 13, 114: 35, 118: 3, 122: 85, 126: 5}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 69, 73, 77, 81, 89, 93, 97, 101, 105, 113, 117, 121, 125], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 3747, 'token_per_expert': {1: 205, 5: 69, 9: 110, 13: 158, 17: 89, 21: 172, 25: 2, 29: 15, 33: 14, 37: 113, 41: 21, 45: 110, 49: 1, 53: 2, 57: 184, 61: 169, 69: 316, 73: 53, 77: 28, 81: 320, 89: 182, 93: 460, 97: 60, 101: 593, 105: 30, 113: 48, 117: 51, 121: 15, 125: 157}}
INFO 05-06 10:42:40.937043.937043 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.926ms | allocate_experts_across_cpu_gpu: 0.252ms
INFO 05-06 10:42:40.937437.937437 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.459785461425781e-05 seconds
INFO 05-06 10:42:40.939246.939246 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0017368793487548828 seconds
INFO 05-06 10:42:40.966661.966661 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.027070999145507812 seconds
INFO 05-06 10:42:40.968261.968261 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016262531280517578 seconds
INFO 05-06 10:42:40.972046.972046 mlpmodule.py:2799] [fused_experts] gmm total=3.682ms E=32 S=4383 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.972575.972575 mlpmodule.py:2799] [fused_experts] gmm total=3.756ms E=32 S=3925 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.973316.973316 mlpmodule.py:2799] [fused_experts] gmm total=4.411ms E=32 S=4329 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.973592.973592 mlpmodule.py:2799] [fused_experts] gmm total=4.866ms E=32 S=3747 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:40.975850.975850 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006575107574462891 seconds
INFO 05-06 10:42:40.975199.975199 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.0067901611328125e-05 seconds
DEBUG 05-06 10:42:40.975354.975354 cuda_h.py:27] end *layer_moe_fused cost 39.366 ms
DEBUG 05-06 10:42:40.996246.996246 cuda_h.py:27] end prefill_layer cost 62.417 ms
DEBUG 05-06 10:42:40.996188.996188 lmp.py:841] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 10:42:40.996607.996607 lmp.py:824] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 10:42:40.998294.998294 cuda_h.py:27] end *sagl cost 1.704 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2606, 'token_per_expert': {3: 20, 7: 164, 11: 52, 15: 22, 19: 64, 23: 1, 27: 30, 31: 137, 35: 2, 39: 198, 43: 114, 47: 193, 51: 8, 55: 3, 59: 9, 63: 121, 67: 55, 71: 174, 75: 196, 79: 74, 83: 105, 87: 15, 91: 14, 99: 139, 103: 20, 107: 18, 111: 67, 115: 342, 119: 36, 123: 1, 127: 212}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 5067, 'token_per_expert': {0: 371, 4: 81, 8: 574, 12: 18, 16: 229, 20: 149, 24: 1, 28: 45, 32: 10, 36: 2, 40: 6, 44: 85, 48: 5, 52: 3, 56: 50, 60: 522, 64: 33, 68: 169, 72: 149, 76: 779, 80: 665, 84: 110, 88: 305, 92: 187, 100: 150, 104: 9, 108: 262, 112: 45, 120: 25, 124: 28}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4374, 'token_per_expert': {2: 25, 6: 44, 10: 217, 14: 417, 18: 171, 22: 3, 26: 25, 30: 6, 34: 50, 38: 6, 42: 257, 46: 210, 50: 29, 54: 126, 58: 156, 62: 312, 66: 11, 70: 20, 74: 506, 78: 57, 82: 237, 86: 557, 90: 93, 94: 131, 98: 70, 102: 18, 106: 364, 110: 1, 114: 5, 118: 1, 122: 1, 126: 248}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4337, 'token_per_expert': {1: 815, 5: 72, 9: 73, 13: 175, 17: 3, 21: 280, 25: 20, 29: 40, 33: 28, 37: 147, 41: 226, 49: 256, 53: 14, 57: 237, 61: 34, 65: 4, 69: 77, 73: 28, 77: 8, 81: 658, 85: 151, 89: 63, 93: 48, 97: 55, 101: 11, 105: 89, 109: 17, 113: 210, 117: 64, 121: 78, 125: 356}}
INFO 05-06 10:42:41.000187.000187 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 1.127ms | allocate_experts_across_cpu_gpu: 0.254ms
INFO 05-06 10:42:41.000641.000641 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-06 10:42:41.002787.002787 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016620159149169922 seconds
INFO 05-06 10:42:41.032417.032417 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02965235710144043 seconds
INFO 05-06 10:42:41.033242.033242 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014219284057617188 seconds
INFO 05-06 10:42:41.037869.037869 mlpmodule.py:2799] [fused_experts] gmm total=3.675ms E=32 S=2606 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.037496.037496 mlpmodule.py:2799] [fused_experts] gmm total=3.885ms E=32 S=5067 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.038749.038749 mlpmodule.py:2799] [fused_experts] gmm total=4.010ms E=32 S=4374 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.039521.039521 mlpmodule.py:2799] [fused_experts] gmm total=4.836ms E=32 S=4337 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.040407.040407 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00644993782043457 seconds
INFO 05-06 10:42:41.040471.040471 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9591064453125e-05 seconds
DEBUG 05-06 10:42:41.040849.040849 cuda_h.py:27] end *layer_moe_fused cost 41.723 ms
DEBUG 05-06 10:42:41.064023.064023 cuda_h.py:27] end prefill_layer cost 68.593 ms
DEBUG 05-06 10:42:41.065435.065435 lmp.py:841] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 10:42:41.065138.065138 lmp.py:824] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 10:42:41.068692.068692 cuda_h.py:27] end *sagl cost 3.241 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 59, 63, 67, 71, 79, 83, 87, 91, 99, 111, 115, 119, 123, 127], 'expert_count': 27, 'ideal_gpu_count': 30, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 4899, 'token_per_expert': {3: 23, 7: 360, 11: 42, 15: 1, 19: 264, 23: 341, 27: 77, 31: 216, 35: 10, 39: 30, 43: 100, 47: 21, 51: 71, 59: 68, 63: 14, 67: 351, 71: 69, 79: 600, 83: 529, 87: 802, 91: 99, 99: 136, 111: 305, 115: 19, 119: 290, 123: 54, 127: 7}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 29, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4612, 'token_per_expert': {0: 19, 4: 18, 8: 39, 12: 1, 16: 732, 20: 147, 24: 222, 28: 93, 32: 351, 36: 130, 40: 51, 44: 39, 48: 79, 52: 22, 56: 614, 64: 39, 68: 294, 72: 8, 76: 246, 80: 37, 84: 7, 88: 11, 92: 508, 96: 7, 100: 292, 104: 1, 108: 177, 112: 140, 116: 122, 120: 70, 124: 96}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 29, 'ideal_gpu_count': 29, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 2837, 'token_per_expert': {2: 139, 6: 431, 10: 140, 18: 68, 22: 30, 26: 3, 30: 191, 34: 20, 38: 147, 42: 54, 46: 78, 50: 32, 54: 75, 58: 16, 62: 52, 66: 138, 70: 102, 74: 23, 82: 100, 90: 1, 94: 12, 98: 90, 102: 734, 106: 10, 110: 12, 114: 5, 118: 26, 122: 21, 126: 87}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 77, 81, 85, 89, 93, 97, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4036, 'token_per_expert': {1: 43, 5: 48, 9: 15, 13: 18, 17: 439, 21: 13, 25: 132, 29: 113, 33: 26, 37: 185, 41: 5, 45: 2, 49: 383, 53: 19, 57: 193, 61: 128, 65: 5, 69: 138, 77: 139, 81: 563, 85: 12, 89: 106, 93: 489, 97: 17, 105: 3, 109: 1, 113: 592, 117: 93, 121: 93, 125: 23}}
INFO 05-06 10:42:41.071919.071919 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 2.252ms | allocate_experts_across_cpu_gpu: 0.273ms
INFO 05-06 10:42:41.072877.072877 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.295608520507812e-05 seconds
INFO 05-06 10:42:41.073493.073493 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015251636505126953 seconds
INFO 05-06 10:42:41.099159.099159 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.025890111923217773 seconds
INFO 05-06 10:42:41.101153.101153 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001546621322631836 seconds
INFO 05-06 10:42:41.105378.105378 mlpmodule.py:2799] [fused_experts] gmm total=4.092ms E=32 S=4899 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.105783.105783 mlpmodule.py:2799] [fused_experts] gmm total=4.188ms E=32 S=4612 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.106063.106063 mlpmodule.py:2799] [fused_experts] gmm total=4.323ms E=32 S=2837 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.106507.106507 mlpmodule.py:2799] [fused_experts] gmm total=5.055ms E=32 S=4036 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.108380.108380 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007089376449584961 seconds
INFO 05-06 10:42:41.108443.108443 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.030632019042969e-05 seconds
DEBUG 05-06 10:42:41.108597.108597 cuda_h.py:27] end *layer_moe_fused cost 39.406 ms
DEBUG 05-06 10:42:41.129661.129661 cuda_h.py:27] end prefill_layer cost 64.243 ms
DEBUG 05-06 10:42:41.129994.129994 lmp.py:841] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 10:42:41.129174.129174 lmp.py:824] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 10:42:41.131549.131549 cuda_h.py:27] end *sagl cost 1.683 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 39, 47, 51, 59, 63, 67, 71, 79, 83, 87, 91, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 27, 'ideal_gpu_count': 28, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 3535, 'token_per_expert': {3: 170, 7: 2, 15: 481, 19: 227, 23: 234, 27: 1, 31: 20, 35: 120, 39: 754, 47: 32, 51: 2, 59: 4, 63: 34, 67: 16, 71: 595, 79: 53, 83: 5, 87: 3, 91: 168, 95: 198, 103: 64, 107: 55, 111: 27, 115: 176, 119: 41, 123: 42, 127: 11}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 24, 32, 36, 40, 48, 52, 56, 64, 68, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 27, 'ideal_gpu_count': 28, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 2744, 'token_per_expert': {0: 1, 4: 6, 8: 10, 12: 70, 20: 18, 24: 33, 32: 17, 36: 98, 40: 98, 48: 3, 52: 1, 56: 7, 64: 17, 68: 70, 76: 158, 80: 99, 84: 164, 88: 109, 92: 202, 96: 1, 100: 90, 104: 95, 108: 666, 112: 37, 116: 510, 120: 20, 124: 144}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 18, 22, 30, 34, 38, 46, 50, 54, 58, 62, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 28, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 5978, 'token_per_expert': {2: 7, 6: 211, 10: 3, 18: 9, 22: 101, 30: 1, 34: 112, 38: 49, 46: 360, 50: 641, 54: 6, 58: 112, 62: 6, 70: 51, 74: 452, 78: 1143, 82: 436, 86: 525, 90: 92, 94: 32, 98: 63, 102: 31, 106: 301, 110: 405, 114: 526, 118: 301, 122: 1, 126: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 29, 'ideal_gpu_count': 27, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 4127, 'token_per_expert': {1: 43, 5: 213, 13: 18, 17: 31, 21: 710, 25: 197, 29: 2, 33: 14, 37: 13, 41: 14, 45: 598, 49: 378, 53: 780, 65: 36, 69: 1, 73: 191, 77: 104, 81: 10, 85: 100, 89: 37, 93: 5, 97: 287, 101: 102, 105: 17, 109: 1, 113: 26, 117: 160, 121: 1, 125: 38}}
INFO 05-06 10:42:41.133168.133168 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 1.212ms | allocate_experts_across_cpu_gpu: 0.258ms
INFO 05-06 10:42:41.133867.133867 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.222724914550781e-05 seconds
INFO 05-06 10:42:41.135814.135814 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015385150909423828 seconds
INFO 05-06 10:42:41.163572.163572 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.027917861938476562 seconds
INFO 05-06 10:42:41.164758.164758 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015120506286621094 seconds
INFO 05-06 10:42:41.168255.168255 mlpmodule.py:2799] [fused_experts] gmm total=3.742ms E=32 S=3535 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.169031.169031 mlpmodule.py:2799] [fused_experts] gmm total=3.862ms E=32 S=2744 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.169894.169894 mlpmodule.py:2799] [fused_experts] gmm total=4.223ms E=32 S=5978 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.170302.170302 mlpmodule.py:2799] [fused_experts] gmm total=4.851ms E=32 S=4127 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.171380.171380 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006860494613647461 seconds
INFO 05-06 10:42:41.171059.171059 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9591064453125e-05 seconds
DEBUG 05-06 10:42:41.172725.172725 cuda_h.py:27] end *layer_moe_fused cost 40.262 ms
DEBUG 05-06 10:42:41.193905.193905 cuda_h.py:27] end prefill_layer cost 64.154 ms
DEBUG 05-06 10:42:41.193364.193364 lmp.py:841] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 10:42:41.193067.193067 lmp.py:824] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 10:42:41.195696.195696 cuda_h.py:27] end *sagl cost 1.557 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4946, 'token_per_expert': {3: 141, 7: 4, 11: 29, 15: 162, 19: 1, 23: 10, 27: 57, 31: 865, 39: 281, 43: 59, 47: 34, 51: 155, 55: 80, 59: 334, 63: 217, 67: 54, 71: 390, 75: 82, 79: 526, 83: 32, 87: 34, 91: 807, 95: 34, 99: 85, 103: 126, 107: 48, 111: 4, 115: 150, 119: 117, 123: 28}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 12, 16, 20, 28, 32, 36, 40, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 2784, 'token_per_expert': {0: 37, 8: 22, 12: 3, 16: 35, 20: 256, 28: 40, 32: 433, 36: 11, 40: 61, 48: 17, 52: 31, 56: 13, 60: 144, 64: 38, 68: 53, 72: 7, 76: 3, 80: 33, 84: 133, 92: 40, 96: 13, 100: 565, 104: 45, 108: 50, 112: 9, 116: 112, 120: 471, 124: 109}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 38, 42, 46, 50, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4590, 'token_per_expert': {2: 52, 6: 283, 10: 9, 14: 397, 18: 2, 22: 153, 26: 32, 34: 126, 38: 180, 42: 60, 46: 37, 50: 2, 58: 1, 62: 19, 66: 7, 70: 38, 74: 10, 78: 515, 82: 82, 86: 142, 90: 21, 94: 31, 98: 205, 102: 242, 106: 23, 110: 631, 114: 880, 118: 158, 122: 74, 126: 178}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 9, 13, 17, 21, 25, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 77, 81, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 4064, 'token_per_expert': {1: 281, 9: 37, 13: 93, 17: 530, 21: 202, 25: 232, 33: 229, 37: 537, 41: 107, 45: 10, 53: 4, 57: 18, 61: 13, 65: 19, 69: 132, 73: 64, 77: 1, 81: 471, 89: 6, 93: 41, 97: 3, 101: 45, 105: 16, 109: 25, 113: 131, 117: 65, 121: 546, 125: 206}}
INFO 05-06 10:42:41.197372.197372 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.951ms | allocate_experts_across_cpu_gpu: 0.252ms
INFO 05-06 10:42:41.197727.197727 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.7220458984375e-05 seconds
INFO 05-06 10:42:41.199124.199124 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015566349029541016 seconds
INFO 05-06 10:42:41.224884.224884 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.025393009185791016 seconds
INFO 05-06 10:42:41.226173.226173 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014491081237792969 seconds
INFO 05-06 10:42:41.230694.230694 mlpmodule.py:2799] [fused_experts] gmm total=3.737ms E=32 S=2784 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.230402.230402 mlpmodule.py:2799] [fused_experts] gmm total=4.030ms E=32 S=4946 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.230992.230992 mlpmodule.py:2799] [fused_experts] gmm total=4.021ms E=32 S=4590 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.231920.231920 mlpmodule.py:2799] [fused_experts] gmm total=4.740ms E=32 S=4064 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.233921.233921 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006844043731689453 seconds
INFO 05-06 10:42:41.233123.233123 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:42:41.233442.233442 cuda_h.py:27] end *layer_moe_fused cost 37.317 ms
DEBUG 05-06 10:42:41.252006.252006 cuda_h.py:27] end prefill_layer cost 58.900 ms
DEBUG 05-06 10:42:41.252657.252657 lmp.py:841] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 10:42:41.252121.252121 lmp.py:824] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 10:42:41.254902.254902 cuda_h.py:27] end *sagl cost 1.528 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5742, 'token_per_expert': {3: 22, 7: 102, 11: 185, 15: 23, 19: 66, 23: 54, 27: 9, 31: 286, 35: 20, 39: 497, 43: 34, 47: 364, 51: 12, 59: 308, 63: 30, 67: 16, 71: 85, 75: 378, 79: 2, 83: 103, 87: 7, 91: 12, 95: 438, 99: 219, 103: 285, 107: 77, 111: 11, 115: 1069, 119: 516, 123: 340, 127: 172}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3273, 'token_per_expert': {0: 52, 4: 1, 8: 201, 12: 118, 16: 56, 24: 161, 28: 69, 32: 115, 36: 46, 40: 28, 44: 40, 48: 36, 52: 132, 56: 5, 60: 85, 64: 44, 68: 24, 72: 97, 76: 160, 80: 199, 84: 4, 88: 8, 92: 72, 96: 11, 100: 377, 104: 184, 108: 104, 112: 139, 116: 18, 120: 94, 124: 593}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 3945, 'token_per_expert': {2: 194, 6: 9, 10: 128, 14: 8, 18: 13, 22: 15, 26: 514, 30: 106, 34: 115, 38: 183, 42: 191, 46: 5, 50: 393, 54: 13, 58: 22, 62: 135, 66: 462, 70: 50, 74: 120, 78: 20, 82: 1, 86: 540, 90: 58, 94: 1, 98: 52, 102: 72, 106: 10, 110: 76, 114: 170, 118: 30, 122: 202, 126: 37}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 65, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3424, 'token_per_expert': {1: 9, 5: 41, 9: 17, 13: 124, 17: 8, 21: 16, 25: 57, 29: 9, 33: 11, 37: 14, 41: 10, 45: 69, 49: 1, 53: 133, 57: 112, 65: 450, 73: 21, 77: 16, 81: 80, 85: 13, 89: 186, 93: 18, 97: 507, 101: 27, 105: 95, 109: 33, 113: 227, 117: 264, 121: 731, 125: 125}}
INFO 05-06 10:42:41.256835.256835 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 1.397ms | allocate_experts_across_cpu_gpu: 0.262ms
INFO 05-06 10:42:41.256495.256495 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.224082946777344e-05 seconds
INFO 05-06 10:42:41.258393.258393 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001470327377319336 seconds
INFO 05-06 10:42:41.282797.282797 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02425980567932129 seconds
INFO 05-06 10:42:41.284178.284178 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0018138885498046875 seconds
INFO 05-06 10:42:41.288994.288994 mlpmodule.py:2799] [fused_experts] gmm total=3.797ms E=32 S=5742 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.289957.289957 mlpmodule.py:2799] [fused_experts] gmm total=3.889ms E=32 S=3945 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.289263.289263 mlpmodule.py:2799] [fused_experts] gmm total=4.212ms E=32 S=3273 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.290224.290224 mlpmodule.py:2799] [fused_experts] gmm total=5.041ms E=32 S=3424 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.291271.291271 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0067713260650634766 seconds
INFO 05-06 10:42:41.291832.291832 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.413459777832031e-05 seconds
DEBUG 05-06 10:42:41.292362.292362 cuda_h.py:27] end *layer_moe_fused cost 37.159 ms
DEBUG 05-06 10:42:41.310012.310012 cuda_h.py:27] end prefill_layer cost 57.636 ms
DEBUG 05-06 10:42:41.310994.310994 lmp.py:841] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 10:42:41.310604.310604 lmp.py:824] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 10:42:41.312871.312871 cuda_h.py:27] end *sagl cost 1.605 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4035, 'token_per_expert': {3: 12, 7: 228, 11: 14, 19: 34, 23: 237, 31: 95, 35: 13, 39: 251, 43: 65, 47: 114, 51: 236, 55: 113, 59: 92, 63: 97, 67: 19, 71: 237, 75: 328, 79: 18, 83: 495, 87: 10, 91: 625, 95: 129, 99: 134, 103: 125, 107: 54, 111: 31, 115: 51, 119: 84, 123: 11, 127: 83}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 60, 64, 68, 72, 76, 80, 84, 88, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 5472, 'token_per_expert': {0: 111, 4: 37, 8: 52, 12: 5, 16: 198, 20: 1, 24: 117, 28: 92, 32: 12, 36: 107, 40: 50, 44: 16, 48: 57, 52: 261, 60: 1, 64: 277, 68: 621, 72: 196, 76: 949, 80: 20, 84: 191, 88: 244, 96: 35, 100: 12, 104: 160, 108: 409, 112: 862, 116: 155, 120: 113, 124: 111}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2939, 'token_per_expert': {2: 204, 6: 8, 10: 410, 14: 92, 18: 46, 22: 5, 26: 3, 30: 337, 34: 42, 38: 26, 42: 130, 46: 93, 50: 5, 54: 20, 58: 35, 62: 1, 66: 251, 70: 206, 74: 4, 78: 79, 82: 42, 86: 83, 90: 370, 94: 17, 98: 200, 102: 78, 106: 3, 110: 14, 114: 76, 118: 45, 126: 14}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3938, 'token_per_expert': {1: 73, 5: 100, 9: 287, 13: 57, 17: 65, 21: 141, 25: 22, 29: 88, 33: 66, 37: 233, 41: 68, 45: 18, 49: 1, 57: 16, 61: 5, 65: 520, 69: 133, 73: 103, 77: 38, 81: 197, 85: 136, 89: 4, 93: 135, 97: 173, 101: 256, 105: 35, 109: 595, 113: 81, 117: 33, 121: 49, 125: 210}}
INFO 05-06 10:42:41.314686.314686 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.794ms | allocate_experts_across_cpu_gpu: 0.250ms
INFO 05-06 10:42:41.314300.314300 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.367134094238281e-05 seconds
INFO 05-06 10:42:41.315188.315188 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015521049499511719 seconds
INFO 05-06 10:42:41.340572.340572 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02438187599182129 seconds
INFO 05-06 10:42:41.341642.341642 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016183853149414062 seconds
INFO 05-06 10:42:41.346356.346356 mlpmodule.py:2799] [fused_experts] gmm total=3.737ms E=32 S=4035 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.346501.346501 mlpmodule.py:2799] [fused_experts] gmm total=3.712ms E=32 S=2939 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.346769.346769 mlpmodule.py:2799] [fused_experts] gmm total=4.498ms E=32 S=5472 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.347944.347944 mlpmodule.py:2799] [fused_experts] gmm total=4.778ms E=32 S=3938 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.348153.348153 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006796121597290039 seconds
INFO 05-06 10:42:41.349111.349111 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.175041198730469e-05 seconds
DEBUG 05-06 10:42:41.349456.349456 cuda_h.py:27] end *layer_moe_fused cost 36.688 ms
DEBUG 05-06 10:42:41.370713.370713 cuda_h.py:27] end prefill_layer cost 60.220 ms
DEBUG 05-06 10:42:41.370649.370649 lmp.py:841] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 10:42:41.370352.370352 lmp.py:824] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 10:42:41.372671.372671 cuda_h.py:27] end *sagl cost 1.574 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 32, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3938, 'token_per_expert': {3: 166, 7: 19, 11: 28, 15: 57, 19: 146, 23: 290, 27: 5, 31: 217, 35: 3, 43: 49, 51: 61, 55: 142, 59: 20, 63: 199, 67: 644, 71: 19, 75: 211, 79: 55, 83: 288, 87: 738, 91: 48, 95: 4, 99: 43, 103: 23, 107: 164, 111: 46, 115: 4, 119: 64, 123: 51, 127: 134}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 32, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 5467, 'token_per_expert': {0: 199, 4: 238, 8: 238, 12: 259, 16: 635, 20: 190, 24: 59, 28: 5, 32: 903, 36: 14, 40: 67, 44: 131, 48: 143, 52: 716, 56: 46, 60: 35, 64: 26, 68: 198, 72: 114, 76: 158, 80: 65, 84: 55, 88: 19, 92: 57, 96: 163, 100: 234, 104: 28, 108: 202, 112: 21, 116: 98, 120: 7, 124: 144}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4290, 'token_per_expert': {2: 138, 6: 33, 10: 58, 14: 115, 18: 73, 22: 105, 26: 145, 30: 48, 34: 33, 38: 38, 42: 184, 46: 1, 50: 16, 54: 196, 58: 130, 62: 42, 66: 271, 70: 119, 74: 12, 78: 178, 82: 127, 86: 477, 90: 112, 94: 4, 98: 21, 102: 98, 106: 4, 110: 161, 114: 185, 118: 83, 122: 19, 126: 1064}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 2689, 'token_per_expert': {1: 354, 5: 225, 9: 24, 13: 40, 17: 70, 21: 63, 25: 1, 29: 9, 33: 33, 37: 43, 41: 14, 45: 43, 49: 11, 53: 17, 57: 70, 61: 97, 65: 112, 69: 49, 73: 5, 77: 80, 81: 76, 85: 117, 89: 23, 93: 61, 97: 72, 101: 4, 105: 514, 109: 46, 113: 73, 117: 133, 121: 73, 125: 137}}
INFO 05-06 10:42:41.374013.374013 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 1.310ms | allocate_experts_across_cpu_gpu: 0.256ms
INFO 05-06 10:42:41.374222.374222 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.555152893066406e-05 seconds
INFO 05-06 10:42:41.376153.376153 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014586448669433594 seconds
INFO 05-06 10:42:41.385055.385055 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009283065795898438 seconds
INFO 05-06 10:42:41.387134.387134 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016999244689941406 seconds
INFO 05-06 10:42:41.391918.391918 mlpmodule.py:2799] [fused_experts] gmm total=3.820ms E=32 S=3938 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.392873.392873 mlpmodule.py:2799] [fused_experts] gmm total=4.145ms E=32 S=5467 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.392231.392231 mlpmodule.py:2799] [fused_experts] gmm total=4.266ms E=32 S=4290 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.392741.392741 mlpmodule.py:2799] [fused_experts] gmm total=4.749ms E=32 S=2689 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.394199.394199 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006520986557006836 seconds
INFO 05-06 10:42:41.394686.394686 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 10:42:41.394046.394046 cuda_h.py:27] end *layer_moe_fused cost 21.362 ms
DEBUG 05-06 10:42:41.415533.415533 cuda_h.py:27] end prefill_layer cost 44.963 ms
DEBUG 05-06 10:42:41.415423.415423 lmp.py:841] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 10:42:41.415126.415126 lmp.py:824] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 10:42:41.418798.418798 cuda_h.py:27] end *sagl cost 2.853 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4591, 'token_per_expert': {3: 84, 7: 9, 11: 21, 15: 14, 19: 37, 23: 633, 27: 322, 31: 147, 35: 111, 39: 321, 43: 308, 47: 170, 51: 13, 55: 100, 59: 49, 63: 186, 67: 48, 71: 213, 75: 557, 79: 7, 83: 39, 87: 42, 91: 47, 95: 504, 99: 61, 103: 132, 107: 226, 111: 74, 119: 60, 123: 45, 127: 11}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4546, 'token_per_expert': {0: 83, 4: 154, 8: 14, 12: 151, 16: 44, 20: 214, 24: 610, 28: 188, 32: 38, 36: 44, 40: 261, 44: 30, 48: 82, 52: 304, 56: 201, 60: 50, 64: 185, 68: 176, 72: 321, 76: 576, 80: 113, 84: 94, 88: 15, 92: 31, 96: 43, 100: 64, 104: 78, 108: 126, 112: 10, 116: 64, 120: 83, 124: 99}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 30, 34, 38, 42, 46, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3184, 'token_per_expert': {2: 26, 6: 117, 10: 215, 14: 44, 18: 208, 22: 153, 30: 19, 34: 24, 38: 51, 42: 15, 46: 1, 54: 139, 58: 336, 62: 30, 66: 35, 70: 112, 74: 514, 78: 116, 82: 19, 86: 355, 90: 28, 94: 72, 98: 101, 102: 18, 106: 151, 110: 4, 114: 78, 118: 54, 122: 88, 126: 61}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 109, 113, 117, 125], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4063, 'token_per_expert': {1: 26, 5: 114, 9: 53, 13: 67, 17: 159, 21: 355, 25: 2, 29: 19, 33: 51, 37: 759, 41: 1, 45: 111, 49: 200, 53: 292, 57: 125, 61: 267, 65: 27, 69: 452, 73: 106, 77: 14, 81: 16, 85: 20, 89: 322, 93: 19, 97: 36, 101: 224, 109: 54, 113: 49, 117: 35, 125: 88}}
INFO 05-06 10:42:41.422472.422472 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 2.743ms | allocate_experts_across_cpu_gpu: 0.252ms
INFO 05-06 10:42:41.422635.422635 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.602836608886719e-05 seconds
INFO 05-06 10:42:41.440611.440611 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.018224477767944336 seconds
INFO 05-06 10:42:41.453701.453701 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012629270553588867 seconds
INFO 05-06 10:42:41.455747.455747 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001661062240600586 seconds
INFO 05-06 10:42:41.459138.459138 mlpmodule.py:2799] [fused_experts] gmm total=3.956ms E=32 S=4591 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.460648.460648 mlpmodule.py:2799] [fused_experts] gmm total=4.072ms E=32 S=4546 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.460638.460638 mlpmodule.py:2799] [fused_experts] gmm total=4.075ms E=32 S=3184 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.460839.460839 mlpmodule.py:2799] [fused_experts] gmm total=4.829ms E=32 S=4063 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.462971.462971 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006742715835571289 seconds
INFO 05-06 10:42:41.462069.462069 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.0558319091796875e-05 seconds
DEBUG 05-06 10:42:41.462345.462345 cuda_h.py:27] end *layer_moe_fused cost 43.315 ms
DEBUG 05-06 10:42:41.468260.468260 cuda_h.py:27] end prefill_layer cost 52.263 ms
DEBUG 05-06 10:42:41.468137.468137 lmp.py:841] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 10:42:41.468152.468152 lmp.py:824] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 10:42:41.470291.470291 cuda_h.py:27] end *sagl cost 2.508 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 32, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4085, 'token_per_expert': {3: 274, 7: 57, 11: 24, 15: 55, 19: 30, 23: 53, 27: 34, 31: 231, 35: 55, 39: 40, 43: 235, 47: 101, 51: 53, 55: 14, 59: 26, 63: 17, 67: 46, 71: 100, 75: 137, 83: 377, 87: 199, 91: 75, 95: 141, 99: 539, 103: 80, 107: 61, 111: 374, 115: 3, 119: 341, 123: 162, 127: 151}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 32, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4418, 'token_per_expert': {0: 17, 4: 232, 8: 282, 12: 95, 16: 11, 20: 2, 24: 22, 28: 1, 32: 366, 36: 340, 40: 267, 44: 15, 48: 56, 52: 12, 56: 73, 60: 223, 64: 304, 68: 83, 72: 212, 76: 353, 80: 128, 84: 213, 88: 146, 92: 126, 96: 28, 100: 155, 104: 250, 108: 79, 112: 33, 116: 73, 120: 177, 124: 44}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 32, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 3395, 'token_per_expert': {2: 386, 6: 5, 10: 112, 14: 239, 18: 24, 22: 7, 26: 63, 30: 84, 34: 145, 38: 175, 42: 25, 46: 69, 50: 376, 54: 362, 58: 257, 62: 33, 66: 37, 70: 40, 74: 60, 78: 171, 82: 34, 86: 4, 90: 46, 94: 31, 98: 54, 102: 14, 106: 3, 110: 214, 114: 34, 118: 240, 122: 44, 126: 7}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4486, 'token_per_expert': {1: 147, 5: 112, 9: 36, 13: 69, 17: 176, 21: 20, 25: 27, 29: 75, 33: 277, 37: 119, 41: 18, 45: 41, 49: 147, 53: 311, 57: 112, 61: 222, 65: 228, 69: 140, 73: 68, 77: 287, 81: 203, 85: 410, 89: 32, 93: 177, 97: 73, 101: 279, 105: 1, 109: 52, 113: 8, 117: 5, 121: 506, 125: 108}}
INFO 05-06 10:42:41.473274.473274 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.626ms | allocate_experts_across_cpu_gpu: 0.466ms
INFO 05-06 10:42:41.473115.473115 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.632110595703125e-05 seconds
INFO 05-06 10:42:41.475372.475372 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001505136489868164 seconds
INFO 05-06 10:42:41.483847.483847 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008582353591918945 seconds
INFO 05-06 10:42:41.485049.485049 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014214515686035156 seconds
INFO 05-06 10:42:41.489806.489806 mlpmodule.py:2799] [fused_experts] gmm total=3.806ms E=32 S=4085 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.489601.489601 mlpmodule.py:2799] [fused_experts] gmm total=3.903ms E=32 S=4418 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.489487.489487 mlpmodule.py:2799] [fused_experts] gmm total=4.133ms E=32 S=3395 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.490910.490910 mlpmodule.py:2799] [fused_experts] gmm total=4.797ms E=32 S=4486 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.491760.491760 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006542205810546875 seconds
INFO 05-06 10:42:41.491254.491254 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.412101745605469e-05 seconds
DEBUG 05-06 10:42:41.492499.492499 cuda_h.py:27] end *layer_moe_fused cost 20.682 ms
DEBUG 05-06 10:42:41.497630.497630 cuda_h.py:27] end prefill_layer cost 29.488 ms
DEBUG 05-06 10:42:41.497764.497764 lmp.py:841] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 10:42:41.497381.497381 lmp.py:824] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 10:42:41.499002.499002 cuda_h.py:27] end *sagl cost 1.894 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 3936, 'token_per_expert': {3: 346, 7: 330, 11: 89, 15: 123, 19: 111, 23: 234, 27: 187, 31: 113, 35: 140, 39: 151, 43: 27, 47: 97, 51: 494, 55: 35, 59: 56, 63: 188, 67: 20, 71: 5, 75: 194, 79: 215, 83: 88, 87: 17, 91: 2, 95: 13, 99: 81, 103: 48, 107: 3, 111: 66, 115: 11, 119: 141, 123: 285, 127: 26}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 5182, 'token_per_expert': {0: 101, 4: 40, 8: 13, 12: 69, 16: 172, 20: 49, 24: 394, 36: 86, 40: 240, 44: 583, 48: 130, 52: 935, 56: 47, 60: 87, 64: 489, 68: 45, 72: 84, 76: 187, 80: 230, 84: 79, 88: 147, 92: 553, 96: 168, 100: 13, 104: 103, 108: 53, 112: 51, 116: 5, 120: 24, 124: 5}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3083, 'token_per_expert': {2: 116, 6: 16, 10: 119, 18: 35, 22: 66, 26: 169, 30: 23, 34: 7, 38: 677, 42: 39, 46: 14, 50: 220, 54: 26, 58: 57, 62: 11, 66: 15, 70: 16, 74: 11, 82: 4, 86: 42, 90: 107, 94: 22, 98: 75, 102: 197, 106: 141, 110: 38, 114: 27, 118: 74, 122: 686, 126: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4183, 'token_per_expert': {1: 151, 5: 89, 9: 235, 13: 88, 17: 82, 21: 244, 25: 73, 29: 40, 33: 93, 37: 486, 41: 140, 45: 30, 49: 7, 53: 123, 57: 18, 61: 264, 65: 20, 69: 180, 73: 62, 77: 15, 81: 1, 85: 3, 89: 816, 93: 9, 97: 67, 101: 17, 105: 13, 109: 208, 113: 7, 117: 386, 121: 45, 125: 171}}
INFO 05-06 10:42:41.501908.501908 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.480ms | allocate_experts_across_cpu_gpu: 0.319ms
INFO 05-06 10:42:41.501973.501973 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.866455078125e-05 seconds
INFO 05-06 10:42:41.503441.503441 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016546249389648438 seconds
INFO 05-06 10:42:41.513705.513705 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009898662567138672 seconds
INFO 05-06 10:42:41.515976.515976 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001485586166381836 seconds
INFO 05-06 10:42:41.519580.519580 mlpmodule.py:2799] [fused_experts] gmm total=3.869ms E=32 S=5182 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.519632.519632 mlpmodule.py:2799] [fused_experts] gmm total=4.145ms E=32 S=3936 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.519555.519555 mlpmodule.py:2799] [fused_experts] gmm total=4.209ms E=32 S=3083 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.520524.520524 mlpmodule.py:2799] [fused_experts] gmm total=4.943ms E=32 S=4183 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.521875.521875 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0068149566650390625 seconds
INFO 05-06 10:42:41.522177.522177 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:42:41.522435.522435 cuda_h.py:27] end *layer_moe_fused cost 21.871 ms
DEBUG 05-06 10:42:41.527061.527061 cuda_h.py:27] end prefill_layer cost 29.987 ms
DEBUG 05-06 10:42:41.527964.527964 lmp.py:841] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 10:42:41.527681.527681 lmp.py:824] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 10:42:41.529252.529252 cuda_h.py:27] end *sagl cost 1.995 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3025, 'token_per_expert': {3: 247, 7: 9, 11: 13, 15: 95, 19: 41, 23: 5, 27: 161, 31: 21, 35: 26, 43: 147, 47: 38, 51: 6, 55: 67, 59: 285, 63: 495, 67: 7, 71: 112, 75: 6, 79: 117, 83: 75, 87: 2, 91: 9, 95: 46, 99: 3, 103: 52, 107: 631, 111: 53, 119: 7, 123: 240, 127: 9}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4487, 'token_per_expert': {0: 48, 4: 212, 8: 239, 12: 74, 16: 11, 20: 84, 24: 27, 28: 219, 32: 172, 36: 34, 40: 281, 44: 182, 52: 92, 56: 254, 60: 75, 64: 99, 68: 936, 72: 107, 76: 81, 80: 48, 84: 67, 88: 212, 92: 210, 100: 140, 104: 29, 108: 183, 112: 156, 116: 157, 120: 49, 124: 9}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3314, 'token_per_expert': {2: 85, 6: 15, 10: 34, 14: 2, 18: 82, 22: 6, 26: 36, 30: 408, 34: 28, 38: 43, 42: 205, 46: 164, 50: 104, 54: 65, 58: 45, 62: 50, 66: 167, 70: 13, 74: 63, 82: 109, 86: 40, 90: 51, 94: 735, 98: 36, 102: 338, 106: 18, 110: 54, 114: 60, 118: 80, 122: 160, 126: 18}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 30, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 5558, 'token_per_expert': {1: 66, 5: 208, 9: 172, 13: 213, 17: 11, 21: 315, 25: 14, 29: 9, 33: 161, 37: 313, 41: 132, 45: 581, 49: 670, 53: 172, 57: 330, 61: 24, 65: 367, 69: 20, 73: 339, 77: 284, 81: 213, 85: 162, 89: 17, 93: 76, 97: 7, 101: 37, 105: 61, 109: 173, 113: 120, 117: 38, 121: 83, 125: 170}}
INFO 05-06 10:42:41.532820.532820 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.792ms | allocate_experts_across_cpu_gpu: 0.351ms
INFO 05-06 10:42:41.532838.532838 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.319450378417969e-05 seconds
INFO 05-06 10:42:41.533098.533098 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015826225280761719 seconds
INFO 05-06 10:42:41.543319.543319 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.00949859619140625 seconds
INFO 05-06 10:42:41.545359.545359 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015025138854980469 seconds
INFO 05-06 10:42:41.548730.548730 mlpmodule.py:2799] [fused_experts] gmm total=3.561ms E=32 S=3025 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.549182.549182 mlpmodule.py:2799] [fused_experts] gmm total=3.707ms E=32 S=4487 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.549607.549607 mlpmodule.py:2799] [fused_experts] gmm total=3.858ms E=32 S=3314 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.551476.551476 mlpmodule.py:2799] [fused_experts] gmm total=5.390ms E=32 S=5558 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.552802.552802 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00725102424621582 seconds
INFO 05-06 10:42:41.552150.552150 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:42:41.552311.552311 cuda_h.py:27] end *layer_moe_fused cost 21.769 ms
DEBUG 05-06 10:42:41.557283.557283 cuda_h.py:27] end prefill_layer cost 30.035 ms
DEBUG 05-06 10:42:41.557901.557901 lmp.py:841] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 10:42:41.558187.558187 lmp.py:824] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 10:42:41.559260.559260 cuda_h.py:27] end *sagl cost 1.945 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 67, 71, 75, 79, 83, 87, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 2760, 'token_per_expert': {3: 54, 7: 194, 11: 465, 15: 4, 19: 3, 23: 15, 27: 2, 31: 144, 35: 125, 39: 3, 43: 47, 47: 11, 51: 181, 55: 60, 59: 51, 67: 174, 71: 45, 75: 131, 79: 101, 83: 227, 87: 88, 95: 105, 99: 3, 103: 180, 107: 8, 111: 94, 115: 24, 119: 54, 123: 73, 127: 94}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3856, 'token_per_expert': {0: 61, 4: 179, 8: 178, 12: 134, 16: 30, 20: 48, 24: 68, 28: 3, 32: 64, 36: 159, 40: 44, 44: 60, 48: 364, 52: 22, 56: 35, 60: 6, 64: 22, 68: 138, 72: 174, 76: 239, 80: 77, 84: 176, 88: 29, 92: 309, 96: 14, 100: 462, 104: 8, 112: 267, 116: 3, 120: 259, 124: 224}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5056, 'token_per_expert': {2: 137, 6: 628, 10: 152, 18: 335, 22: 12, 26: 357, 30: 169, 34: 112, 38: 92, 42: 72, 46: 328, 50: 58, 54: 25, 58: 104, 62: 222, 66: 2, 70: 103, 74: 37, 78: 764, 82: 94, 86: 153, 90: 189, 94: 17, 98: 7, 102: 140, 106: 61, 110: 189, 114: 32, 118: 80, 122: 336, 126: 49}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 77, 81, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4712, 'token_per_expert': {1: 438, 5: 625, 9: 21, 13: 149, 17: 23, 21: 50, 25: 14, 29: 285, 33: 143, 37: 216, 41: 241, 45: 49, 53: 271, 57: 194, 61: 196, 65: 480, 69: 24, 73: 296, 77: 7, 81: 77, 89: 3, 93: 26, 97: 168, 101: 13, 105: 339, 109: 192, 113: 39, 117: 1, 121: 41, 125: 91}}
INFO 05-06 10:42:41.561259.561259 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.471ms | allocate_experts_across_cpu_gpu: 0.344ms
INFO 05-06 10:42:41.562801.562801 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.343292236328125e-05 seconds
INFO 05-06 10:42:41.563666.563666 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016314983367919922 seconds
INFO 05-06 10:42:41.573732.573732 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009578704833984375 seconds
INFO 05-06 10:42:41.575224.575224 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015604496002197266 seconds
INFO 05-06 10:42:41.579136.579136 mlpmodule.py:2799] [fused_experts] gmm total=3.663ms E=32 S=2760 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.579036.579036 mlpmodule.py:2799] [fused_experts] gmm total=3.735ms E=32 S=3856 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.579886.579886 mlpmodule.py:2799] [fused_experts] gmm total=3.911ms E=32 S=5056 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.580995.580995 mlpmodule.py:2799] [fused_experts] gmm total=4.682ms E=32 S=4712 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.581784.581784 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006374359130859375 seconds
INFO 05-06 10:42:41.581444.581444 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.8650970458984375e-05 seconds
DEBUG 05-06 10:42:41.582038.582038 cuda_h.py:27] end *layer_moe_fused cost 20.914 ms
DEBUG 05-06 10:42:41.588497.588497 cuda_h.py:27] end prefill_layer cost 30.031 ms
DEBUG 05-06 10:42:41.588254.588254 lmp.py:841] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 10:42:41.588587.588587 lmp.py:824] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 10:42:41.590512.590512 cuda_h.py:27] end *sagl cost 1.871 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 32, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4425, 'token_per_expert': {3: 4, 7: 111, 11: 110, 15: 141, 19: 100, 23: 9, 27: 28, 31: 99, 35: 572, 39: 23, 43: 174, 47: 80, 51: 37, 55: 282, 59: 463, 63: 37, 67: 16, 71: 7, 75: 214, 79: 36, 83: 43, 87: 16, 91: 7, 95: 10, 99: 117, 103: 432, 107: 97, 111: 155, 115: 156, 119: 360, 123: 210, 127: 279}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5812, 'token_per_expert': {0: 41, 4: 1, 8: 251, 12: 3, 16: 107, 20: 36, 24: 516, 28: 168, 32: 62, 36: 5, 40: 92, 44: 93, 48: 82, 56: 4, 60: 32, 64: 655, 68: 318, 72: 585, 76: 170, 80: 1, 84: 36, 88: 138, 92: 279, 96: 19, 100: 1199, 104: 6, 108: 227, 112: 65, 116: 259, 120: 252, 124: 110}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3746, 'token_per_expert': {2: 16, 6: 12, 10: 30, 14: 32, 18: 3, 22: 6, 26: 96, 30: 171, 34: 44, 38: 234, 42: 95, 46: 186, 54: 29, 58: 100, 62: 58, 66: 152, 70: 172, 74: 530, 78: 7, 82: 204, 86: 228, 90: 173, 94: 204, 98: 42, 102: 116, 106: 16, 110: 41, 114: 5, 118: 176, 122: 25, 126: 543}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2401, 'token_per_expert': {1: 104, 5: 4, 9: 39, 13: 8, 17: 5, 21: 3, 25: 41, 29: 4, 33: 99, 37: 12, 41: 91, 45: 92, 49: 6, 53: 248, 57: 41, 61: 12, 65: 31, 69: 110, 73: 307, 77: 9, 81: 21, 85: 81, 89: 149, 93: 371, 97: 12, 101: 65, 105: 9, 109: 57, 113: 63, 117: 250, 125: 57}}
INFO 05-06 10:42:41.592477.592477 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.907ms | allocate_experts_across_cpu_gpu: 0.334ms
INFO 05-06 10:42:41.592866.592866 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.009506225585938e-05 seconds
INFO 05-06 10:42:41.594917.594917 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016639232635498047 seconds
INFO 05-06 10:42:41.604874.604874 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010250329971313477 seconds
INFO 05-06 10:42:41.606106.606106 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016849040985107422 seconds
INFO 05-06 10:42:41.610751.610751 mlpmodule.py:2799] [fused_experts] gmm total=3.375ms E=32 S=2401 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.610813.610813 mlpmodule.py:2799] [fused_experts] gmm total=3.924ms E=32 S=3746 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.611882.611882 mlpmodule.py:2799] [fused_experts] gmm total=4.414ms E=32 S=4425 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.611620.611620 mlpmodule.py:2799] [fused_experts] gmm total=4.575ms E=32 S=5812 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.613308.613308 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006464242935180664 seconds
INFO 05-06 10:42:41.613305.613305 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.38690185546875e-05 seconds
DEBUG 05-06 10:42:41.613055.613055 cuda_h.py:27] end *layer_moe_fused cost 22.282 ms
DEBUG 05-06 10:42:41.618680.618680 cuda_h.py:27] end prefill_layer cost 30.082 ms
DEBUG 05-06 10:42:41.618100.618100 lmp.py:841] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 10:42:41.618524.618524 lmp.py:824] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 10:42:41.620491.620491 cuda_h.py:27] end *sagl cost 2.322 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4011, 'token_per_expert': {3: 245, 7: 11, 11: 25, 15: 16, 19: 66, 23: 39, 27: 21, 31: 133, 35: 218, 39: 528, 43: 357, 47: 340, 51: 50, 55: 13, 59: 100, 67: 590, 71: 118, 75: 65, 79: 306, 83: 120, 87: 104, 91: 80, 95: 32, 99: 38, 103: 45, 107: 27, 111: 7, 115: 101, 119: 4, 123: 194, 127: 18}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3378, 'token_per_expert': {0: 16, 4: 6, 8: 133, 12: 58, 16: 188, 20: 17, 24: 117, 28: 31, 32: 60, 36: 39, 40: 72, 44: 281, 48: 43, 52: 48, 56: 665, 60: 30, 64: 19, 68: 30, 72: 147, 76: 123, 80: 139, 84: 181, 88: 10, 92: 11, 100: 129, 104: 145, 108: 280, 112: 85, 116: 138, 120: 75, 124: 62}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3545, 'token_per_expert': {2: 155, 6: 84, 10: 57, 14: 18, 18: 208, 22: 92, 26: 176, 30: 128, 34: 105, 38: 83, 42: 66, 46: 366, 50: 8, 54: 22, 58: 33, 62: 40, 66: 30, 70: 3, 74: 20, 78: 196, 82: 14, 86: 615, 90: 175, 98: 315, 102: 30, 106: 141, 110: 20, 114: 1, 118: 284, 122: 56, 126: 4}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 117, 125], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 5450, 'token_per_expert': {1: 286, 5: 308, 9: 68, 13: 6, 17: 168, 21: 897, 25: 355, 29: 410, 33: 111, 37: 422, 41: 22, 49: 15, 53: 31, 57: 49, 61: 396, 65: 416, 69: 4, 73: 123, 77: 7, 81: 36, 85: 219, 89: 53, 93: 10, 97: 300, 101: 17, 105: 129, 109: 202, 117: 136, 125: 254}}
INFO 05-06 10:42:41.625680.625680 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 3.098ms | allocate_experts_across_cpu_gpu: 0.308ms
INFO 05-06 10:42:41.625055.625055 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.866455078125e-05 seconds
INFO 05-06 10:42:41.626550.626550 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014824867248535156 seconds
INFO 05-06 10:42:41.635296.635296 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008766889572143555 seconds
INFO 05-06 10:42:41.637721.637721 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015478134155273438 seconds
INFO 05-06 10:42:41.639430.639430 mlpmodule.py:2799] [fused_experts] gmm total=1.636ms E=32 S=4011 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.641335.641335 mlpmodule.py:2799] [fused_experts] gmm total=3.612ms E=32 S=3378 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.641315.641315 mlpmodule.py:2799] [fused_experts] gmm total=3.700ms E=32 S=3545 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.641126.641126 mlpmodule.py:2799] [fused_experts] gmm total=4.081ms E=32 S=5450 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.643560.643560 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005997657775878906 seconds
INFO 05-06 10:42:41.643524.643524 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9591064453125e-05 seconds
DEBUG 05-06 10:42:41.643885.643885 cuda_h.py:27] end *layer_moe_fused cost 21.994 ms
DEBUG 05-06 10:42:41.649651.649651 cuda_h.py:27] end prefill_layer cost 30.668 ms
DEBUG 05-06 10:42:41.649739.649739 lmp.py:841] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 10:42:41.649734.649734 lmp.py:824] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 10:42:41.650727.650727 cuda_h.py:27] end *sagl cost 1.750 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3791, 'token_per_expert': {3: 6, 7: 115, 11: 400, 15: 14, 19: 243, 23: 229, 27: 450, 31: 56, 35: 170, 39: 1, 43: 85, 47: 42, 51: 4, 55: 16, 59: 13, 63: 437, 67: 208, 71: 359, 75: 68, 79: 69, 83: 161, 87: 32, 91: 319, 95: 8, 99: 15, 107: 42, 111: 45, 115: 28, 119: 37, 123: 7, 127: 112}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3733, 'token_per_expert': {4: 187, 8: 57, 12: 304, 16: 214, 20: 91, 24: 10, 28: 47, 32: 68, 36: 132, 40: 70, 44: 304, 48: 171, 52: 389, 56: 289, 60: 154, 64: 628, 68: 39, 76: 24, 80: 10, 84: 8, 88: 1, 92: 55, 96: 42, 100: 102, 104: 35, 108: 190, 112: 22, 116: 14, 120: 47, 124: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4640, 'token_per_expert': {2: 18, 6: 392, 10: 15, 18: 31, 22: 5, 26: 50, 30: 141, 34: 374, 38: 50, 42: 67, 46: 35, 50: 160, 54: 3, 58: 12, 62: 48, 66: 52, 70: 511, 74: 126, 78: 10, 82: 123, 86: 135, 90: 933, 94: 265, 98: 207, 102: 31, 106: 68, 110: 210, 114: 268, 118: 133, 122: 151, 126: 16}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4220, 'token_per_expert': {1: 137, 5: 134, 9: 63, 13: 101, 17: 122, 21: 8, 25: 7, 29: 159, 33: 377, 37: 145, 45: 207, 49: 74, 53: 51, 57: 63, 61: 16, 65: 30, 69: 2, 73: 290, 77: 256, 81: 73, 85: 1, 89: 12, 93: 7, 97: 618, 101: 5, 105: 53, 109: 159, 113: 4, 117: 2, 121: 1043, 125: 1}}
INFO 05-06 10:42:41.653344.653344 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 1.025ms | allocate_experts_across_cpu_gpu: 0.299ms
INFO 05-06 10:42:41.653951.653951 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.437301635742188e-05 seconds
INFO 05-06 10:42:41.654353.654353 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014863014221191406 seconds
INFO 05-06 10:42:41.663446.663446 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008858203887939453 seconds
INFO 05-06 10:42:41.665636.665636 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014452934265136719 seconds
INFO 05-06 10:42:41.667688.667688 mlpmodule.py:2799] [fused_experts] gmm total=1.628ms E=32 S=3791 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.669507.669507 mlpmodule.py:2799] [fused_experts] gmm total=3.610ms E=32 S=3733 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.669740.669740 mlpmodule.py:2799] [fused_experts] gmm total=3.731ms E=32 S=4640 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.669799.669799 mlpmodule.py:2799] [fused_experts] gmm total=3.815ms E=32 S=4220 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.671550.671550 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005885124206542969 seconds
INFO 05-06 10:42:41.671229.671229 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:42:41.671491.671491 cuda_h.py:27] end *layer_moe_fused cost 19.764 ms
DEBUG 05-06 10:42:41.677551.677551 cuda_h.py:27] end prefill_layer cost 28.159 ms
DEBUG 05-06 10:42:41.677672.677672 lmp.py:841] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 10:42:41.677183.677183 lmp.py:824] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 10:42:41.678214.678214 cuda_h.py:27] end *sagl cost 1.504 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3869, 'token_per_expert': {3: 253, 7: 140, 11: 119, 15: 10, 19: 88, 23: 20, 27: 24, 31: 30, 35: 430, 39: 100, 43: 53, 47: 54, 51: 53, 55: 21, 63: 143, 67: 172, 71: 155, 75: 6, 79: 105, 83: 245, 87: 91, 91: 167, 95: 4, 99: 45, 103: 32, 107: 647, 111: 142, 115: 8, 119: 29, 123: 458, 127: 25}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 4862, 'token_per_expert': {0: 87, 4: 14, 8: 56, 12: 31, 16: 1231, 24: 24, 32: 25, 36: 150, 40: 1, 44: 153, 48: 67, 52: 360, 56: 109, 60: 343, 64: 239, 68: 727, 72: 60, 76: 28, 80: 280, 84: 9, 88: 93, 92: 26, 96: 2, 100: 142, 104: 255, 108: 31, 112: 46, 116: 120, 120: 114, 124: 39}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4762, 'token_per_expert': {2: 450, 6: 49, 10: 158, 14: 146, 18: 432, 22: 22, 26: 70, 30: 3, 34: 153, 38: 50, 42: 23, 46: 29, 50: 127, 54: 1, 58: 1031, 62: 3, 66: 25, 70: 335, 74: 36, 78: 66, 82: 152, 86: 3, 90: 228, 94: 4, 98: 1, 102: 6, 106: 108, 110: 685, 114: 216, 118: 63, 122: 24, 126: 63}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 2891, 'token_per_expert': {1: 14, 5: 63, 9: 72, 13: 38, 17: 17, 21: 29, 25: 73, 29: 69, 33: 31, 37: 1, 41: 126, 45: 428, 49: 127, 53: 16, 57: 6, 61: 15, 65: 8, 69: 287, 73: 40, 77: 37, 81: 8, 85: 446, 89: 168, 93: 126, 97: 101, 101: 12, 109: 82, 113: 3, 117: 399, 121: 13, 125: 36}}
INFO 05-06 10:42:41.680217.680217 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.901ms | allocate_experts_across_cpu_gpu: 0.253ms
INFO 05-06 10:42:41.681619.681619 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.175041198730469e-05 seconds
INFO 05-06 10:42:41.682336.682336 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016012191772460938 seconds
INFO 05-06 10:42:41.691378.691378 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008808612823486328 seconds
INFO 05-06 10:42:41.693916.693916 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015625953674316406 seconds
INFO 05-06 10:42:41.695770.695770 mlpmodule.py:2799] [fused_experts] gmm total=1.650ms E=32 S=3869 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.697879.697879 mlpmodule.py:2799] [fused_experts] gmm total=3.777ms E=32 S=4862 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.697125.697125 mlpmodule.py:2799] [fused_experts] gmm total=3.891ms E=32 S=4762 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.697576.697576 mlpmodule.py:2799] [fused_experts] gmm total=4.012ms E=32 S=2891 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.699789.699789 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0060596466064453125 seconds
INFO 05-06 10:42:41.699184.699184 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.030632019042969e-05 seconds
DEBUG 05-06 10:42:41.699593.699593 cuda_h.py:27] end *layer_moe_fused cost 20.077 ms
DEBUG 05-06 10:42:41.705878.705878 cuda_h.py:27] end prefill_layer cost 28.068 ms
DEBUG 05-06 10:42:41.705621.705621 lmp.py:841] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 10:42:41.705847.705847 lmp.py:824] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 10:42:41.707044.707044 cuda_h.py:27] end *sagl cost 1.484 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4323, 'token_per_expert': {3: 141, 7: 3, 11: 10, 15: 209, 19: 183, 23: 26, 27: 344, 31: 51, 35: 50, 43: 347, 47: 20, 51: 139, 55: 6, 59: 151, 63: 37, 67: 56, 71: 31, 75: 77, 79: 114, 83: 40, 87: 639, 91: 46, 95: 497, 99: 105, 103: 125, 107: 12, 111: 629, 115: 47, 119: 2, 123: 179, 127: 7}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3586, 'token_per_expert': {0: 33, 4: 20, 8: 37, 12: 1, 16: 6, 20: 728, 24: 461, 28: 17, 32: 9, 36: 85, 40: 26, 44: 30, 48: 9, 52: 191, 56: 190, 60: 176, 64: 1, 68: 74, 72: 61, 76: 171, 80: 27, 84: 499, 88: 83, 92: 20, 96: 65, 100: 8, 104: 267, 108: 69, 112: 81, 116: 34, 124: 107}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 2850, 'token_per_expert': {2: 34, 6: 3, 10: 103, 14: 161, 18: 36, 22: 6, 26: 105, 30: 109, 34: 16, 38: 77, 42: 54, 46: 12, 50: 157, 54: 12, 58: 8, 62: 1, 66: 108, 70: 223, 74: 24, 78: 191, 82: 6, 86: 178, 90: 149, 98: 3, 102: 136, 110: 31, 114: 661, 118: 72, 122: 37, 126: 137}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5625, 'token_per_expert': {1: 49, 5: 33, 9: 10, 13: 44, 17: 585, 21: 3, 25: 80, 29: 90, 33: 6, 37: 69, 41: 73, 45: 102, 49: 213, 53: 8, 57: 97, 61: 83, 65: 491, 73: 275, 77: 72, 81: 79, 85: 1197, 89: 767, 93: 7, 97: 125, 101: 2, 105: 169, 109: 60, 113: 791, 117: 12, 121: 5, 125: 28}}
INFO 05-06 10:42:41.709829.709829 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 1.341ms | allocate_experts_across_cpu_gpu: 0.250ms
INFO 05-06 10:42:41.709647.709647 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.340576171875e-05 seconds
INFO 05-06 10:42:41.711663.711663 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016109943389892578 seconds
INFO 05-06 10:42:41.720582.720582 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008715629577636719 seconds
INFO 05-06 10:42:41.721188.721188 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016105175018310547 seconds
INFO 05-06 10:42:41.723985.723985 mlpmodule.py:2799] [fused_experts] gmm total=1.700ms E=32 S=4323 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.725087.725087 mlpmodule.py:2799] [fused_experts] gmm total=3.812ms E=32 S=3586 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.726935.726935 mlpmodule.py:2799] [fused_experts] gmm total=3.906ms E=32 S=2850 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.726722.726722 mlpmodule.py:2799] [fused_experts] gmm total=4.157ms E=32 S=5625 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.728134.728134 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00642704963684082 seconds
INFO 05-06 10:42:41.728714.728714 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.76837158203125e-05 seconds
DEBUG 05-06 10:42:41.728895.728895 cuda_h.py:27] end *layer_moe_fused cost 20.778 ms
DEBUG 05-06 10:42:41.733749.733749 cuda_h.py:27] end prefill_layer cost 28.115 ms
DEBUG 05-06 10:42:41.733347.733347 lmp.py:841] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 10:42:41.733812.733812 lmp.py:824] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 10:42:41.735492.735492 cuda_h.py:27] end *sagl cost 1.490 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 5301, 'token_per_expert': {3: 191, 7: 173, 11: 28, 15: 26, 19: 44, 23: 86, 27: 35, 31: 242, 35: 193, 39: 41, 43: 550, 47: 20, 51: 260, 55: 57, 59: 25, 63: 26, 67: 6, 71: 48, 75: 97, 79: 281, 83: 177, 87: 679, 91: 30, 95: 379, 99: 1, 103: 407, 107: 3, 111: 245, 115: 378, 119: 197, 123: 273, 127: 103}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3931, 'token_per_expert': {0: 35, 4: 79, 8: 161, 12: 183, 16: 22, 20: 85, 24: 395, 28: 115, 32: 27, 36: 248, 40: 56, 44: 10, 48: 312, 56: 102, 60: 38, 64: 198, 68: 26, 72: 4, 76: 281, 80: 33, 84: 17, 88: 399, 92: 2, 96: 22, 100: 399, 104: 3, 108: 129, 112: 73, 116: 27, 120: 429, 124: 21}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3596, 'token_per_expert': {2: 15, 6: 15, 10: 39, 14: 142, 18: 173, 22: 8, 26: 61, 30: 8, 34: 8, 42: 164, 46: 253, 50: 566, 54: 87, 58: 43, 62: 290, 66: 119, 70: 204, 74: 43, 78: 273, 82: 294, 86: 16, 90: 118, 94: 78, 98: 141, 106: 100, 110: 18, 114: 106, 118: 105, 122: 93, 126: 16}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3556, 'token_per_expert': {1: 199, 5: 1, 9: 4, 13: 252, 17: 3, 21: 46, 25: 256, 29: 24, 33: 373, 37: 280, 41: 186, 45: 510, 49: 79, 53: 142, 57: 9, 61: 172, 65: 395, 69: 11, 77: 5, 81: 6, 85: 87, 89: 2, 93: 11, 97: 10, 101: 3, 105: 95, 109: 161, 113: 42, 121: 128, 125: 64}}
INFO 05-06 10:42:41.737296.737296 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.891ms | allocate_experts_across_cpu_gpu: 0.246ms
INFO 05-06 10:42:41.737260.737260 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.626678466796875e-05 seconds
INFO 05-06 10:42:41.739526.739526 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015485286712646484 seconds
INFO 05-06 10:42:41.747755.747755 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008102893829345703 seconds
INFO 05-06 10:42:41.748094.748094 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015416145324707031 seconds
INFO 05-06 10:42:41.750615.750615 mlpmodule.py:2799] [fused_experts] gmm total=1.762ms E=32 S=5301 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.753015.753015 mlpmodule.py:2799] [fused_experts] gmm total=3.895ms E=32 S=3931 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.753856.753856 mlpmodule.py:2799] [fused_experts] gmm total=3.993ms E=32 S=3596 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.753075.753075 mlpmodule.py:2799] [fused_experts] gmm total=4.083ms E=32 S=3556 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.755973.755973 lmp.py:1484] [layer_moe_fused] experts compute time: 0.006134510040283203 seconds
INFO 05-06 10:42:41.755414.755414 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:42:41.755088.755088 cuda_h.py:27] end *layer_moe_fused cost 19.837 ms
DEBUG 05-06 10:42:41.760465.760465 cuda_h.py:27] end prefill_layer cost 26.806 ms
DEBUG 05-06 10:42:41.760453.760453 lmp.py:841] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 10:42:41.760209.760209 lmp.py:824] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 10:42:41.762928.762928 cuda_h.py:27] end *sagl cost 1.654 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 29, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 4689, 'token_per_expert': {3: 15, 7: 5, 11: 248, 15: 13, 19: 8, 23: 62, 27: 3, 31: 2, 35: 7, 39: 35, 43: 33, 47: 299, 51: 15, 55: 132, 59: 8, 63: 1, 67: 14, 71: 277, 75: 315, 79: 49, 83: 4, 87: 12, 91: 509, 95: 75, 99: 4, 103: 4, 111: 1542, 115: 550, 119: 258, 123: 138, 127: 52}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 26, 'ideal_gpu_count': 29, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 5556, 'token_per_expert': {8: 21, 12: 1392, 20: 1176, 24: 90, 28: 3, 32: 195, 36: 24, 40: 364, 44: 44, 48: 38, 52: 284, 56: 8, 60: 59, 68: 236, 72: 18, 76: 658, 80: 2, 84: 123, 88: 80, 92: 28, 100: 84, 104: 142, 108: 9, 112: 465, 120: 9, 124: 4}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 2294, 'token_per_expert': {6: 12, 18: 109, 22: 171, 26: 4, 30: 109, 34: 4, 38: 3, 42: 1, 46: 250, 50: 16, 54: 9, 58: 14, 62: 80, 66: 5, 70: 202, 74: 55, 78: 231, 82: 14, 90: 388, 94: 49, 98: 53, 102: 2, 106: 134, 110: 250, 114: 7, 118: 4, 122: 44, 126: 74}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121], 'expert_count': 30, 'ideal_gpu_count': 28, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3845, 'token_per_expert': {1: 89, 5: 105, 9: 65, 13: 149, 17: 9, 21: 11, 29: 3, 33: 53, 37: 114, 41: 1, 45: 9, 49: 1136, 53: 293, 57: 549, 61: 4, 65: 62, 69: 81, 73: 20, 77: 187, 81: 20, 85: 93, 89: 143, 93: 7, 97: 79, 101: 150, 105: 75, 109: 24, 113: 160, 117: 68, 121: 86}}
INFO 05-06 10:42:41.764126.764126 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 1.116ms | allocate_experts_across_cpu_gpu: 0.273ms
INFO 05-06 10:42:41.764250.764250 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.0558319091796875e-05 seconds
INFO 05-06 10:42:41.766318.766318 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016791820526123047 seconds
INFO 05-06 10:42:41.777963.777963 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010170698165893555 seconds
INFO 05-06 10:42:41.778227.778227 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016298294067382812 seconds
INFO 05-06 10:42:41.780529.780529 mlpmodule.py:2799] [fused_experts] gmm total=1.723ms E=32 S=4689 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.782156.782156 mlpmodule.py:2799] [fused_experts] gmm total=3.432ms E=32 S=2294 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.782540.782540 mlpmodule.py:2799] [fused_experts] gmm total=3.711ms E=32 S=5556 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.783886.783886 mlpmodule.py:2799] [fused_experts] gmm total=3.765ms E=32 S=3845 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.784614.784614 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0057332515716552734 seconds
INFO 05-06 10:42:41.784379.784379 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.553794860839844e-05 seconds
DEBUG 05-06 10:42:41.785773.785773 cuda_h.py:27] end *layer_moe_fused cost 21.882 ms
DEBUG 05-06 10:42:41.790662.790662 cuda_h.py:27] end prefill_layer cost 29.715 ms
DEBUG 05-06 10:42:41.790359.790359 lmp.py:841] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 10:42:41.790969.790969 lmp.py:824] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 10:42:41.792120.792120 cuda_h.py:27] end *sagl cost 2.115 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 51, 55, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 31, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 4589, 'token_per_expert': {3: 69, 7: 694, 11: 70, 15: 94, 19: 434, 23: 241, 27: 258, 31: 49, 35: 47, 39: 10, 43: 443, 51: 4, 55: 11, 63: 40, 67: 139, 71: 232, 75: 54, 79: 3, 83: 43, 87: 21, 91: 682, 95: 100, 99: 463, 107: 112, 111: 13, 115: 89, 119: 31, 123: 135, 127: 8}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 4269, 'token_per_expert': {0: 6, 4: 449, 8: 30, 12: 4, 16: 205, 20: 407, 24: 60, 28: 346, 32: 66, 36: 2, 40: 16, 44: 74, 48: 157, 52: 554, 56: 282, 60: 212, 64: 587, 68: 3, 72: 9, 76: 30, 80: 75, 84: 49, 88: 43, 92: 76, 96: 47, 100: 5, 104: 1, 108: 69, 112: 7, 116: 87, 120: 105, 124: 206}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 3901, 'token_per_expert': {2: 97, 6: 34, 10: 63, 14: 188, 18: 252, 22: 320, 26: 258, 30: 171, 34: 13, 38: 10, 42: 353, 46: 23, 50: 27, 54: 145, 58: 32, 62: 218, 66: 43, 70: 9, 74: 8, 78: 91, 82: 186, 86: 470, 90: 218, 94: 24, 98: 19, 102: 13, 106: 545, 114: 51, 118: 7, 122: 5, 126: 8}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 3625, 'token_per_expert': {1: 73, 5: 9, 9: 100, 13: 52, 17: 7, 21: 63, 25: 69, 29: 196, 33: 24, 37: 19, 41: 1, 45: 2, 49: 190, 53: 138, 57: 307, 61: 225, 65: 17, 69: 114, 73: 114, 77: 136, 81: 177, 85: 110, 89: 101, 93: 153, 97: 236, 101: 95, 105: 51, 109: 100, 113: 151, 117: 281, 121: 281, 125: 33}}
INFO 05-06 10:42:41.797472.797472 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 3.356ms | allocate_experts_across_cpu_gpu: 0.301ms
INFO 05-06 10:42:41.797881.797881 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.222724914550781e-05 seconds
INFO 05-06 10:42:41.798867.798867 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00153350830078125 seconds
INFO 05-06 10:42:41.807506.807506 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.008349180221557617 seconds
INFO 05-06 10:42:41.808774.808774 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015900135040283203 seconds
INFO 05-06 10:42:41.811056.811056 mlpmodule.py:2799] [fused_experts] gmm total=1.764ms E=32 S=4589 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.813721.813721 mlpmodule.py:2799] [fused_experts] gmm total=3.804ms E=32 S=3901 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.813039.813039 mlpmodule.py:2799] [fused_experts] gmm total=3.906ms E=32 S=3625 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.813933.813933 mlpmodule.py:2799] [fused_experts] gmm total=4.299ms E=32 S=4269 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.815654.815654 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0061702728271484375 seconds
INFO 05-06 10:42:41.815717.815717 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.078315734863281e-05 seconds
DEBUG 05-06 10:42:41.815019.815019 cuda_h.py:27] end *layer_moe_fused cost 22.144 ms
DEBUG 05-06 10:42:41.820520.820520 cuda_h.py:27] end prefill_layer cost 30.490 ms
DEBUG 05-06 10:42:41.820933.820933 lmp.py:841] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 10:42:41.821496.821496 cuda_h.py:27] end prefill_step cost 1546.702 ms
INFO 05-06 10:42:41.821815.821815 lmp.py:843] prefill time: 1.6784839630126953 seconds
DEBUG 05-06 10:42:41.858752.858752 cuda_h.py:27] end init_inputs_tokens cost 6.860 ms
DEBUG 05-06 10:42:41.858449.858449 lmp.py:899] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:42:41.859835.859835 lmp.py:904] ---- decode step 0 layer 0 ----
DEBUG 05-06 10:42:41.860551.860551 cuda_h.py:27] end *sagl cost 1.584 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 47, 55, 63, 79, 83, 87, 103, 123, 127], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 16, 'token_per_expert': {15: 2, 47: 1, 55: 1, 63: 2, 79: 2, 83: 2, 87: 2, 103: 1, 123: 1, 127: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 32, 48, 60, 116], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {8: 2, 32: 1, 48: 1, 60: 2, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [22, 26, 90, 114], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {22: 2, 26: 1, 90: 1, 114: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [33, 45, 53], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {33: 1, 45: 2, 53: 1}}
INFO 05-06 10:42:41.861768.861768 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.340ms | allocate_experts_across_cpu_gpu: 0.116ms
INFO 05-06 10:42:41.862830.862830 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0265579223632812e-05 seconds
INFO 05-06 10:42:41.863752.863752 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015935897827148438 seconds
INFO 05-06 10:42:41.865194.865194 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016727447509765625 seconds
INFO 05-06 10:42:41.866224.866224 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014278888702392578 seconds
INFO 05-06 10:42:41.868207.868207 mlpmodule.py:2799] [fused_experts] gmm total=1.099ms E=32 S=16 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.869752.869752 mlpmodule.py:2799] [fused_experts] gmm total=2.074ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.869281.869281 mlpmodule.py:2799] [fused_experts] gmm total=2.041ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.869256.869256 mlpmodule.py:2799] [fused_experts] gmm total=2.371ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.870464.870464 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003931283950805664 seconds
INFO 05-06 10:42:41.870130.870130 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.0531158447265625e-05 seconds
DEBUG 05-06 10:42:41.871546.871546 cuda_h.py:27] end *layer_moe_fused cost 10.056 ms
DEBUG 05-06 10:42:41.871103.871103 cuda_h.py:27] end decode_layer cost 12.937 ms
DEBUG 05-06 10:42:41.872416.872416 lmp.py:904] ---- decode step 0 layer 1 ----
DEBUG 05-06 10:42:41.873091.873091 cuda_h.py:27] end *sagl cost 1.548 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [83, 107, 119, 123], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {83: 1, 107: 2, 119: 2, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 56, 92, 96, 124], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 13, 'token_per_expert': {0: 3, 8: 2, 56: 3, 92: 2, 96: 1, 124: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [30, 54, 110], 'expert_count': 3, 'ideal_gpu_count': 4, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {30: 2, 54: 2, 110: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [9, 13, 65, 73, 121], 'expert_count': 5, 'ideal_gpu_count': 4, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {9: 2, 13: 1, 65: 1, 73: 1, 121: 2}}
INFO 05-06 10:42:41.874240.874240 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.310ms | allocate_experts_across_cpu_gpu: 0.082ms
INFO 05-06 10:42:41.874196.874196 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.52587890625e-05 seconds
INFO 05-06 10:42:41.876405.876405 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014405250549316406 seconds
INFO 05-06 10:42:41.877555.877555 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013897418975830078 seconds
INFO 05-06 10:42:41.879862.879862 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013744831085205078 seconds
INFO 05-06 10:42:41.880069.880069 mlpmodule.py:2799] [fused_experts] gmm total=1.093ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.882457.882457 mlpmodule.py:2799] [fused_experts] gmm total=2.262ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.882788.882788 mlpmodule.py:2799] [fused_experts] gmm total=2.318ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.882441.882441 mlpmodule.py:2799] [fused_experts] gmm total=2.359ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.883676.883676 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004018306732177734 seconds
INFO 05-06 10:42:41.883441.883441 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.744529724121094e-05 seconds
DEBUG 05-06 10:42:41.884526.884526 cuda_h.py:27] end *layer_moe_fused cost 9.734 ms
DEBUG 05-06 10:42:41.884798.884798 cuda_h.py:27] end decode_layer cost 12.601 ms
DEBUG 05-06 10:42:41.884065.884065 lmp.py:904] ---- decode step 0 layer 2 ----
DEBUG 05-06 10:42:41.886891.886891 cuda_h.py:27] end *sagl cost 1.484 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 91], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 5, 'token_per_expert': {11: 3, 91: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 76, 120], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {8: 1, 12: 2, 76: 4, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [62, 70, 90, 102, 106, 126], 'expert_count': 6, 'ideal_gpu_count': 4, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {62: 2, 70: 1, 90: 1, 102: 1, 106: 2, 126: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [41, 45, 49, 61, 81, 97], 'expert_count': 6, 'ideal_gpu_count': 4, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {41: 2, 45: 1, 49: 1, 61: 1, 81: 3, 97: 1}}
INFO 05-06 10:42:41.887448.887448 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.299ms | allocate_experts_across_cpu_gpu: 0.082ms
INFO 05-06 10:42:41.887842.887842 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.621246337890625e-05 seconds
INFO 05-06 10:42:41.888952.888952 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012769699096679688 seconds
INFO 05-06 10:42:41.890074.890074 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012311935424804688 seconds
INFO 05-06 10:42:41.891843.891843 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013415813446044922 seconds
INFO 05-06 10:42:41.892142.892142 mlpmodule.py:2799] [fused_experts] gmm total=1.071ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.893987.893987 mlpmodule.py:2799] [fused_experts] gmm total=2.042ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.894863.894863 mlpmodule.py:2799] [fused_experts] gmm total=2.191ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.894042.894042 mlpmodule.py:2799] [fused_experts] gmm total=2.307ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.895556.895556 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0036308765411376953 seconds
INFO 05-06 10:42:41.895169.895169 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.100799560546875e-05 seconds
DEBUG 05-06 10:42:41.895951.895951 cuda_h.py:27] end *layer_moe_fused cost 8.689 ms
DEBUG 05-06 10:42:41.896388.896388 cuda_h.py:27] end decode_layer cost 11.434 ms
DEBUG 05-06 10:42:41.896225.896225 lmp.py:904] ---- decode step 0 layer 3 ----
DEBUG 05-06 10:42:41.897291.897291 cuda_h.py:27] end *sagl cost 1.546 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [39, 67], 'expert_count': 2, 'ideal_gpu_count': 7, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 2, 'token_per_expert': {39: 1, 67: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [16, 24, 40, 44, 96, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {16: 1, 24: 1, 40: 1, 44: 1, 96: 3, 104: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [26, 30, 34, 42, 50, 54, 70, 110, 118, 126], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 12, 'token_per_expert': {26: 2, 30: 1, 34: 1, 42: 1, 50: 2, 54: 1, 70: 1, 110: 1, 118: 1, 126: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [5, 9, 73, 85, 101, 117, 125], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {5: 1, 9: 1, 73: 1, 85: 1, 101: 1, 117: 2, 125: 1}}
INFO 05-06 10:42:41.899917.899917 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.301ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:42:41.899641.899641 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9550323486328125e-05 seconds
INFO 05-06 10:42:41.900925.900925 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013353824615478516 seconds
INFO 05-06 10:42:41.901327.901327 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001260995864868164 seconds
INFO 05-06 10:42:41.903903.903903 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001306772232055664 seconds
INFO 05-06 10:42:41.904904.904904 mlpmodule.py:2799] [fused_experts] gmm total=1.102ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.905407.905407 mlpmodule.py:2799] [fused_experts] gmm total=2.138ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.905282.905282 mlpmodule.py:2799] [fused_experts] gmm total=2.168ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.905520.905520 mlpmodule.py:2799] [fused_experts] gmm total=2.429ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.907887.907887 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003831148147583008 seconds
INFO 05-06 10:42:41.907308.907308 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.00543212890625e-05 seconds
DEBUG 05-06 10:42:41.907937.907937 cuda_h.py:27] end *layer_moe_fused cost 8.912 ms
DEBUG 05-06 10:42:41.907408.907408 cuda_h.py:27] end decode_layer cost 11.787 ms
DEBUG 05-06 10:42:41.908866.908866 lmp.py:904] ---- decode step 0 layer 4 ----
DEBUG 05-06 10:42:41.909422.909422 cuda_h.py:27] end *sagl cost 1.530 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 51, 67, 83, 87], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {3: 3, 51: 2, 67: 1, 83: 1, 87: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 48, 60, 84], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {20: 2, 48: 1, 60: 1, 84: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 50, 82, 106, 114, 122, 126], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 1, 50: 2, 82: 1, 106: 2, 114: 1, 122: 1, 126: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [17, 25, 45, 93, 101, 113, 121], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {17: 1, 25: 2, 45: 2, 93: 1, 101: 1, 113: 1, 121: 1}}
INFO 05-06 10:42:41.910351.910351 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.303ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 10:42:41.910884.910884 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 10:42:41.912424.912424 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012788772583007812 seconds
INFO 05-06 10:42:41.913402.913402 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012638568878173828 seconds
INFO 05-06 10:42:41.914334.914334 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012524127960205078 seconds
INFO 05-06 10:42:41.916533.916533 mlpmodule.py:2799] [fused_experts] gmm total=1.072ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.917099.917099 mlpmodule.py:2799] [fused_experts] gmm total=2.017ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.917782.917782 mlpmodule.py:2799] [fused_experts] gmm total=2.133ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.917795.917795 mlpmodule.py:2799] [fused_experts] gmm total=2.218ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.918533.918533 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003707408905029297 seconds
INFO 05-06 10:42:41.918861.918861 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.100799560546875e-05 seconds
DEBUG 05-06 10:42:41.918195.918195 cuda_h.py:27] end *layer_moe_fused cost 8.570 ms
DEBUG 05-06 10:42:41.919315.919315 cuda_h.py:27] end decode_layer cost 11.421 ms
DEBUG 05-06 10:42:41.919344.919344 lmp.py:904] ---- decode step 0 layer 5 ----
DEBUG 05-06 10:42:41.921536.921536 cuda_h.py:27] end *sagl cost 1.543 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [39, 71, 95, 99, 123], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 11, 'token_per_expert': {39: 2, 71: 2, 95: 2, 99: 3, 123: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 36, 52, 72, 116], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {4: 1, 36: 1, 52: 2, 72: 1, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 34, 46, 70, 74, 94], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 1, 34: 1, 46: 2, 70: 2, 74: 1, 94: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [5, 29, 61, 65], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {5: 2, 29: 1, 61: 3, 65: 1}}
INFO 05-06 10:42:41.922109.922109 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.330ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:42:41.922317.922317 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.811981201171875e-05 seconds
INFO 05-06 10:42:41.923428.923428 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013108253479003906 seconds
INFO 05-06 10:42:41.925605.925605 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012679100036621094 seconds
INFO 05-06 10:42:41.926823.926823 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012862682342529297 seconds
INFO 05-06 10:42:41.927102.927102 mlpmodule.py:2799] [fused_experts] gmm total=1.095ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.928723.928723 mlpmodule.py:2799] [fused_experts] gmm total=2.084ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.929008.929008 mlpmodule.py:2799] [fused_experts] gmm total=2.105ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.929902.929902 mlpmodule.py:2799] [fused_experts] gmm total=2.389ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.930014.930014 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003878355026245117 seconds
INFO 05-06 10:42:41.930627.930627 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.0531158447265625e-05 seconds
DEBUG 05-06 10:42:41.930563.930563 cuda_h.py:27] end *layer_moe_fused cost 8.833 ms
DEBUG 05-06 10:42:41.931933.931933 cuda_h.py:27] end decode_layer cost 11.665 ms
DEBUG 05-06 10:42:41.931008.931008 lmp.py:904] ---- decode step 0 layer 6 ----
DEBUG 05-06 10:42:41.932386.932386 cuda_h.py:27] end *sagl cost 1.575 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [35, 43, 87, 115], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {35: 2, 43: 1, 87: 3, 115: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [24, 32, 68, 96, 100, 104, 124], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {24: 2, 32: 1, 68: 1, 96: 2, 100: 1, 104: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 70, 78, 86, 90, 106, 118], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 70: 1, 78: 2, 86: 1, 90: 1, 106: 1, 118: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 13, 25, 101], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {1: 2, 13: 2, 25: 1, 101: 1}}
INFO 05-06 10:42:41.934905.934905 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.301ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:42:41.934292.934292 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6927719116210938e-05 seconds
INFO 05-06 10:42:41.935085.935085 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012898445129394531 seconds
INFO 05-06 10:42:41.936659.936659 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012807846069335938 seconds
INFO 05-06 10:42:41.938591.938591 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012545585632324219 seconds
INFO 05-06 10:42:41.939231.939231 mlpmodule.py:2799] [fused_experts] gmm total=1.013ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.940006.940006 mlpmodule.py:2799] [fused_experts] gmm total=2.142ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.940813.940813 mlpmodule.py:2799] [fused_experts] gmm total=2.215ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.940990.940990 mlpmodule.py:2799] [fused_experts] gmm total=2.550ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.942214.942214 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004126548767089844 seconds
INFO 05-06 10:42:41.942065.942065 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.1484832763671875e-05 seconds
DEBUG 05-06 10:42:41.942506.942506 cuda_h.py:27] end *layer_moe_fused cost 9.227 ms
DEBUG 05-06 10:42:41.943228.943228 cuda_h.py:27] end decode_layer cost 12.095 ms
DEBUG 05-06 10:42:41.943587.943587 lmp.py:904] ---- decode step 0 layer 7 ----
DEBUG 05-06 10:42:41.944526.944526 cuda_h.py:27] end *sagl cost 1.498 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 43], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 3, 'token_per_expert': {19: 1, 43: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 64, 68, 80, 96, 100, 104, 112], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {20: 2, 64: 1, 68: 1, 80: 1, 96: 2, 100: 1, 104: 2, 112: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [10, 14, 18, 34, 82, 90, 114], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {10: 1, 14: 1, 18: 1, 34: 1, 82: 1, 90: 2, 114: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [9, 65, 69, 97, 101, 121], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {9: 1, 65: 1, 69: 1, 97: 3, 101: 1, 121: 2}}
INFO 05-06 10:42:41.946527.946527 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.300ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:42:41.946060.946060 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7404556274414062e-05 seconds
INFO 05-06 10:42:41.947110.947110 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001268148422241211 seconds
INFO 05-06 10:42:41.948293.948293 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012755393981933594 seconds
INFO 05-06 10:42:41.950674.950674 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014064311981201172 seconds
INFO 05-06 10:42:41.951486.951486 mlpmodule.py:2799] [fused_experts] gmm total=0.991ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.952547.952547 mlpmodule.py:2799] [fused_experts] gmm total=2.274ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.953162.953162 mlpmodule.py:2799] [fused_experts] gmm total=2.364ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.953023.953023 mlpmodule.py:2799] [fused_experts] gmm total=2.494ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.954076.954076 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0038521289825439453 seconds
INFO 05-06 10:42:41.954259.954259 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.0531158447265625e-05 seconds
DEBUG 05-06 10:42:41.954640.954640 cuda_h.py:27] end *layer_moe_fused cost 9.113 ms
DEBUG 05-06 10:42:41.955462.955462 cuda_h.py:27] end decode_layer cost 11.876 ms
DEBUG 05-06 10:42:41.955629.955629 lmp.py:904] ---- decode step 0 layer 8 ----
DEBUG 05-06 10:42:41.956773.956773 cuda_h.py:27] end *sagl cost 1.509 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 19, 27, 51, 55, 63, 75, 103], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {15: 1, 19: 1, 27: 1, 51: 3, 55: 2, 63: 1, 75: 1, 103: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [12, 24, 32, 64], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {12: 2, 24: 1, 32: 1, 64: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 38, 42, 46, 50, 54, 110], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {6: 1, 38: 1, 42: 2, 46: 1, 50: 3, 54: 1, 110: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [33, 69, 93, 105], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {33: 1, 69: 2, 93: 1, 105: 1}}
INFO 05-06 10:42:41.958147.958147 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.300ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:42:41.958586.958586 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6927719116210938e-05 seconds
INFO 05-06 10:42:41.959194.959194 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012929439544677734 seconds
INFO 05-06 10:42:41.960416.960416 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012345314025878906 seconds
INFO 05-06 10:42:41.962944.962944 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012729167938232422 seconds
INFO 05-06 10:42:41.963892.963892 mlpmodule.py:2799] [fused_experts] gmm total=1.097ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.964894.964894 mlpmodule.py:2799] [fused_experts] gmm total=2.019ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.964110.964110 mlpmodule.py:2799] [fused_experts] gmm total=2.117ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.964546.964546 mlpmodule.py:2799] [fused_experts] gmm total=2.384ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.965520.965520 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037055015563964844 seconds
INFO 05-06 10:42:41.966941.966941 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.076957702636719e-05 seconds
DEBUG 05-06 10:42:41.966827.966827 cuda_h.py:27] end *layer_moe_fused cost 8.828 ms
DEBUG 05-06 10:42:41.966250.966250 cuda_h.py:27] end decode_layer cost 11.618 ms
DEBUG 05-06 10:42:41.966609.966609 lmp.py:904] ---- decode step 0 layer 9 ----
DEBUG 05-06 10:42:41.968650.968650 cuda_h.py:27] end *sagl cost 1.573 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 19, 83, 95, 111], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {7: 2, 15: 1, 19: 2, 83: 1, 95: 3, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 36, 48, 76], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {4: 1, 36: 1, 48: 1, 76: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [30, 54, 70, 74], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {30: 2, 54: 1, 70: 2, 74: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [37, 57, 69, 81, 89, 101, 125], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {37: 1, 57: 1, 69: 2, 81: 1, 89: 2, 101: 3, 125: 1}}
INFO 05-06 10:42:41.969546.969546 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.303ms | allocate_experts_across_cpu_gpu: 0.088ms
INFO 05-06 10:42:41.969171.969171 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6689300537109375e-05 seconds
INFO 05-06 10:42:41.971355.971355 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012969970703125 seconds
INFO 05-06 10:42:41.972472.972472 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012621879577636719 seconds
INFO 05-06 10:42:41.973034.973034 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012979507446289062 seconds
INFO 05-06 10:42:41.975818.975818 mlpmodule.py:2799] [fused_experts] gmm total=1.116ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.976612.976612 mlpmodule.py:2799] [fused_experts] gmm total=2.047ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.976660.976660 mlpmodule.py:2799] [fused_experts] gmm total=2.197ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.976150.976150 mlpmodule.py:2799] [fused_experts] gmm total=2.585ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.977214.977214 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003941059112548828 seconds
INFO 05-06 10:42:41.978642.978642 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.38690185546875e-05 seconds
DEBUG 05-06 10:42:41.978896.978896 cuda_h.py:27] end *layer_moe_fused cost 8.867 ms
DEBUG 05-06 10:42:41.978870.978870 cuda_h.py:27] end decode_layer cost 11.739 ms
DEBUG 05-06 10:42:41.978706.978706 lmp.py:904] ---- decode step 0 layer 10 ----
DEBUG 05-06 10:42:41.980032.980032 cuda_h.py:27] end *sagl cost 1.571 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 47, 67, 79, 87, 127], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 6, 'token_per_expert': {19: 1, 47: 1, 67: 1, 79: 1, 87: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 28, 44, 60], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {8: 3, 28: 1, 44: 2, 60: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [18, 46, 54, 62, 126], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {18: 1, 46: 2, 54: 1, 62: 1, 126: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [5, 21, 37, 57, 81, 97, 105], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {5: 1, 21: 1, 37: 1, 57: 1, 81: 3, 97: 3, 105: 2}}
INFO 05-06 10:42:41.981604.981604 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.308ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:42:41.981944.981944 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.71661376953125e-05 seconds
INFO 05-06 10:42:41.983616.983616 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012357234954833984 seconds
INFO 05-06 10:42:41.984648.984648 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013055801391601562 seconds
INFO 05-06 10:42:41.985145.985145 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013213157653808594 seconds
INFO 05-06 10:42:41.987816.987816 mlpmodule.py:2799] [fused_experts] gmm total=1.141ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.988959.988959 mlpmodule.py:2799] [fused_experts] gmm total=2.086ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.988033.988033 mlpmodule.py:2799] [fused_experts] gmm total=2.208ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.988999.988999 mlpmodule.py:2799] [fused_experts] gmm total=2.309ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.989868.989868 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003954887390136719 seconds
INFO 05-06 10:42:41.989481.989481 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.0531158447265625e-05 seconds
DEBUG 05-06 10:42:41.990889.990889 cuda_h.py:27] end *layer_moe_fused cost 9.108 ms
DEBUG 05-06 10:42:41.990882.990882 cuda_h.py:27] end decode_layer cost 11.957 ms
DEBUG 05-06 10:42:41.990480.990480 lmp.py:904] ---- decode step 0 layer 11 ----
DEBUG 05-06 10:42:41.992056.992056 cuda_h.py:27] end *sagl cost 1.546 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 23, 63, 67, 79, 83, 99], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 13, 'token_per_expert': {7: 1, 23: 1, 63: 1, 67: 1, 79: 3, 83: 4, 99: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [124], 'expert_count': 1, 'ideal_gpu_count': 5, 'keep_on_gpu': 1, 'hit_count_on_device': 1, 'token_total': 2, 'token_per_expert': {124: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 38, 46, 50, 70, 102, 114], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 6: 1, 38: 3, 46: 2, 50: 1, 70: 1, 102: 1, 114: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [9, 49, 81], 'expert_count': 3, 'ideal_gpu_count': 4, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {9: 1, 49: 2, 81: 3}}
INFO 05-06 10:42:41.993396.993396 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.297ms | allocate_experts_across_cpu_gpu: 0.088ms
INFO 05-06 10:42:41.993267.993267 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:42:41.995077.995077 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001232147216796875 seconds
INFO 05-06 10:42:41.996945.996945 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0011487007141113281 seconds
INFO 05-06 10:42:41.997593.997593 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012917518615722656 seconds
INFO 05-06 10:42:41.998475.998475 mlpmodule.py:2799] [fused_experts] gmm total=1.086ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:41.999971.999971 mlpmodule.py:2799] [fused_experts] gmm total=1.918ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.000706.000706 mlpmodule.py:2799] [fused_experts] gmm total=2.315ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.000058.000058 mlpmodule.py:2799] [fused_experts] gmm total=2.646ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.001924.001924 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0038754940032958984 seconds
INFO 05-06 10:42:42.001536.001536 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.100799560546875e-05 seconds
DEBUG 05-06 10:42:42.002781.002781 cuda_h.py:27] end *layer_moe_fused cost 8.922 ms
DEBUG 05-06 10:42:42.002631.002631 cuda_h.py:27] end decode_layer cost 11.834 ms
DEBUG 05-06 10:42:42.002183.002183 lmp.py:904] ---- decode step 0 layer 12 ----
DEBUG 05-06 10:42:42.004470.004470 cuda_h.py:27] end *sagl cost 1.613 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 19, 39], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {3: 1, 15: 1, 19: 2, 39: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [36, 80, 92], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {36: 2, 80: 2, 92: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [46, 50, 74, 78, 86, 106, 114], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 13, 'token_per_expert': {46: 2, 50: 1, 74: 2, 78: 4, 86: 1, 106: 2, 114: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [21, 45, 73, 97, 117], 'expert_count': 5, 'ideal_gpu_count': 4, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {21: 1, 45: 1, 73: 1, 97: 1, 117: 3}}
INFO 05-06 10:42:42.005685.005685 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.321ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:42:42.005025.005025 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6450881958007812e-05 seconds
INFO 05-06 10:42:42.006114.006114 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012259483337402344 seconds
INFO 05-06 10:42:42.008182.008182 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0011909008026123047 seconds
INFO 05-06 10:42:42.009479.009479 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013148784637451172 seconds
INFO 05-06 10:42:42.010183.010183 mlpmodule.py:2799] [fused_experts] gmm total=1.092ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.011920.011920 mlpmodule.py:2799] [fused_experts] gmm total=2.022ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.012435.012435 mlpmodule.py:2799] [fused_experts] gmm total=2.159ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.012342.012342 mlpmodule.py:2799] [fused_experts] gmm total=2.434ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.013616.013616 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004014015197753906 seconds
INFO 05-06 10:42:42.013467.013467 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.1961669921875e-05 seconds
DEBUG 05-06 10:42:42.014345.014345 cuda_h.py:27] end *layer_moe_fused cost 9.047 ms
DEBUG 05-06 10:42:42.014530.014530 cuda_h.py:27] end decode_layer cost 11.943 ms
DEBUG 05-06 10:42:42.014512.014512 lmp.py:904] ---- decode step 0 layer 13 ----
DEBUG 05-06 10:42:42.016558.016558 cuda_h.py:27] end *sagl cost 1.541 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [47, 55, 71, 107], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {47: 1, 55: 1, 71: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [32, 76, 80, 100, 104], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {32: 1, 76: 1, 80: 2, 100: 3, 104: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 14, 22, 26, 78, 110, 114], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 6: 1, 14: 2, 22: 2, 26: 1, 78: 2, 110: 1, 114: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 13, 41, 109, 125], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 2, 13: 1, 41: 1, 109: 1, 125: 2}}
INFO 05-06 10:42:42.017931.017931 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.304ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:42:42.017079.017079 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7881393432617188e-05 seconds
INFO 05-06 10:42:42.018241.018241 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012459754943847656 seconds
INFO 05-06 10:42:42.020272.020272 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012671947479248047 seconds
INFO 05-06 10:42:42.021445.021445 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001522064208984375 seconds
INFO 05-06 10:42:42.023255.023255 mlpmodule.py:2799] [fused_experts] gmm total=1.073ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.024555.024555 mlpmodule.py:2799] [fused_experts] gmm total=2.220ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.024408.024408 mlpmodule.py:2799] [fused_experts] gmm total=2.194ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.024740.024740 mlpmodule.py:2799] [fused_experts] gmm total=2.488ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.026756.026756 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004194498062133789 seconds
INFO 05-06 10:42:42.026892.026892 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.1484832763671875e-05 seconds
DEBUG 05-06 10:42:42.026109.026109 cuda_h.py:27] end *layer_moe_fused cost 9.657 ms
DEBUG 05-06 10:42:42.027546.027546 cuda_h.py:27] end decode_layer cost 12.484 ms
DEBUG 05-06 10:42:42.027813.027813 lmp.py:904] ---- decode step 0 layer 14 ----
DEBUG 05-06 10:42:42.028727.028727 cuda_h.py:27] end *sagl cost 1.548 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 31, 39, 47, 99, 115], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {11: 1, 31: 1, 39: 2, 47: 1, 99: 1, 115: 4}}
experts_gpu_alloc_device_1 {'expert_ids': [56, 100, 112], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {56: 1, 100: 1, 112: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 10, 26, 38, 46, 62], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {2: 3, 10: 2, 26: 2, 38: 1, 46: 1, 62: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [9, 25, 81, 97, 109, 121], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {9: 1, 25: 2, 81: 2, 97: 1, 109: 1, 121: 2}}
INFO 05-06 10:42:42.030934.030934 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.295ms | allocate_experts_across_cpu_gpu: 0.089ms
INFO 05-06 10:42:42.030042.030042 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:42:42.031626.031626 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00138092041015625 seconds
INFO 05-06 10:42:42.032907.032907 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0011992454528808594 seconds
INFO 05-06 10:42:42.034446.034446 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012123584747314453 seconds
INFO 05-06 10:42:42.035592.035592 mlpmodule.py:2799] [fused_experts] gmm total=1.070ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.036974.036974 mlpmodule.py:2799] [fused_experts] gmm total=2.065ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.036905.036905 mlpmodule.py:2799] [fused_experts] gmm total=2.246ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.036719.036719 mlpmodule.py:2799] [fused_experts] gmm total=2.323ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.037695.037695 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003815174102783203 seconds
INFO 05-06 10:42:42.038453.038453 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.220008850097656e-05 seconds
DEBUG 05-06 10:42:42.038947.038947 cuda_h.py:27] end *layer_moe_fused cost 8.893 ms
DEBUG 05-06 10:42:42.038907.038907 cuda_h.py:27] end decode_layer cost 11.722 ms
DEBUG 05-06 10:42:42.039313.039313 lmp.py:904] ---- decode step 0 layer 15 ----
DEBUG 05-06 10:42:42.040319.040319 cuda_h.py:27] end *sagl cost 1.513 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [47, 75, 83, 119], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {47: 1, 75: 2, 83: 2, 119: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [24, 68, 72, 84, 100, 108, 112], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {24: 1, 68: 2, 72: 2, 84: 1, 100: 1, 108: 2, 112: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [30, 34], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 2, 'token_per_expert': {30: 1, 34: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [33, 65, 69, 81, 93, 97, 101], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {33: 2, 65: 1, 69: 1, 81: 3, 93: 1, 97: 1, 101: 2}}
INFO 05-06 10:42:42.041851.041851 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.331ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 10:42:42.041430.041430 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.5974044799804688e-05 seconds
INFO 05-06 10:42:42.043466.043466 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001257181167602539 seconds
INFO 05-06 10:42:42.044576.044576 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012562274932861328 seconds
INFO 05-06 10:42:42.045532.045532 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013620853424072266 seconds
INFO 05-06 10:42:42.047768.047768 mlpmodule.py:2799] [fused_experts] gmm total=1.024ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.048777.048777 mlpmodule.py:2799] [fused_experts] gmm total=2.160ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.048908.048908 mlpmodule.py:2799] [fused_experts] gmm total=2.207ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.048615.048615 mlpmodule.py:2799] [fused_experts] gmm total=2.241ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.050074.050074 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003986358642578125 seconds
INFO 05-06 10:42:42.050018.050018 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.029273986816406e-05 seconds
DEBUG 05-06 10:42:42.050498.050498 cuda_h.py:27] end *layer_moe_fused cost 9.259 ms
DEBUG 05-06 10:42:42.051359.051359 cuda_h.py:27] end decode_layer cost 12.037 ms
DEBUG 05-06 10:42:42.051526.051526 lmp.py:904] ---- decode step 0 layer 16 ----
DEBUG 05-06 10:42:42.052646.052646 cuda_h.py:27] end *sagl cost 1.561 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 19, 63, 87, 107], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {15: 1, 19: 2, 63: 1, 87: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 32, 44, 52], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 7, 'token_per_expert': {0: 1, 4: 1, 16: 1, 20: 1, 32: 1, 44: 1, 52: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [54, 62, 66, 78, 82, 102, 122], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {54: 2, 62: 2, 66: 1, 78: 1, 82: 1, 102: 2, 122: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 5, 85, 105], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 85: 2, 105: 2}}
INFO 05-06 10:42:42.054921.054921 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.333ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:42:42.054268.054268 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 10:42:42.055325.055325 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012717247009277344 seconds
INFO 05-06 10:42:42.056450.056450 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013017654418945312 seconds
INFO 05-06 10:42:42.058691.058691 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001409292221069336 seconds
INFO 05-06 10:42:42.059540.059540 mlpmodule.py:2799] [fused_experts] gmm total=1.083ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.060383.060383 mlpmodule.py:2799] [fused_experts] gmm total=2.198ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.060318.060318 mlpmodule.py:2799] [fused_experts] gmm total=2.226ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.061225.061225 mlpmodule.py:2799] [fused_experts] gmm total=2.509ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.062868.062868 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0038242340087890625 seconds
INFO 05-06 10:42:42.062103.062103 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.220008850097656e-05 seconds
DEBUG 05-06 10:42:42.062022.062022 cuda_h.py:27] end *layer_moe_fused cost 9.178 ms
DEBUG 05-06 10:42:42.063326.063326 cuda_h.py:27] end decode_layer cost 12.031 ms
DEBUG 05-06 10:42:42.063375.063375 lmp.py:904] ---- decode step 0 layer 17 ----
DEBUG 05-06 10:42:42.064788.064788 cuda_h.py:27] end *sagl cost 1.589 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 23, 27, 39, 47, 55, 95], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {7: 1, 23: 3, 27: 1, 39: 1, 47: 1, 55: 1, 95: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [16, 68, 120], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {16: 3, 68: 1, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [18, 22, 34, 70], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {18: 3, 22: 1, 34: 2, 70: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [5, 13, 33, 37, 53, 65, 73, 113], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {5: 1, 13: 2, 33: 1, 37: 1, 53: 1, 65: 1, 73: 2, 113: 2}}
INFO 05-06 10:42:42.066307.066307 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.296ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:42:42.066369.066369 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 10:42:42.067578.067578 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012805461883544922 seconds
INFO 05-06 10:42:42.068252.068252 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001285552978515625 seconds
INFO 05-06 10:42:42.070964.070964 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014042854309082031 seconds
INFO 05-06 10:42:42.071170.071170 mlpmodule.py:2799] [fused_experts] gmm total=1.069ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.072940.072940 mlpmodule.py:2799] [fused_experts] gmm total=1.999ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.072734.072734 mlpmodule.py:2799] [fused_experts] gmm total=2.066ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.073398.073398 mlpmodule.py:2799] [fused_experts] gmm total=2.221ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.074238.074238 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037567615509033203 seconds
INFO 05-06 10:42:42.074757.074757 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.172325134277344e-05 seconds
DEBUG 05-06 10:42:42.074544.074544 cuda_h.py:27] end *layer_moe_fused cost 9.061 ms
DEBUG 05-06 10:42:42.075842.075842 cuda_h.py:27] end decode_layer cost 11.961 ms
DEBUG 05-06 10:42:42.075725.075725 lmp.py:904] ---- decode step 0 layer 18 ----
DEBUG 05-06 10:42:42.076699.076699 cuda_h.py:27] end *sagl cost 1.559 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [23, 59, 75, 83, 111], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {23: 2, 59: 1, 75: 2, 83: 2, 111: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [32, 36, 40, 80, 104], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {32: 1, 36: 1, 40: 2, 80: 2, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [26, 30, 42, 54, 58], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {26: 1, 30: 2, 42: 1, 54: 2, 58: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [37, 69, 73, 77, 97, 101, 105], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {37: 2, 69: 1, 73: 1, 77: 1, 97: 1, 101: 1, 105: 1}}
INFO 05-06 10:42:42.078249.078249 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.297ms | allocate_experts_across_cpu_gpu: 0.088ms
INFO 05-06 10:42:42.078113.078113 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7642974853515625e-05 seconds
INFO 05-06 10:42:42.079520.079520 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012505054473876953 seconds
INFO 05-06 10:42:42.080895.080895 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012776851654052734 seconds
INFO 05-06 10:42:42.082485.082485 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013141632080078125 seconds
INFO 05-06 10:42:42.083677.083677 mlpmodule.py:2799] [fused_experts] gmm total=1.068ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.084815.084815 mlpmodule.py:2799] [fused_experts] gmm total=2.085ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.084814.084814 mlpmodule.py:2799] [fused_experts] gmm total=2.152ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.084683.084683 mlpmodule.py:2799] [fused_experts] gmm total=2.285ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.085890.085890 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037555694580078125 seconds
INFO 05-06 10:42:42.086265.086265 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.076957702636719e-05 seconds
DEBUG 05-06 10:42:42.086203.086203 cuda_h.py:27] end *layer_moe_fused cost 8.884 ms
DEBUG 05-06 10:42:42.086341.086341 cuda_h.py:27] end decode_layer cost 11.685 ms
DEBUG 05-06 10:42:42.086701.086701 lmp.py:904] ---- decode step 0 layer 19 ----
DEBUG 05-06 10:42:42.088556.088556 cuda_h.py:27] end *sagl cost 1.576 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 31, 111], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {19: 1, 31: 1, 111: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [40, 44, 84, 96], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 9, 'token_per_expert': {40: 3, 44: 3, 84: 2, 96: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [10, 22, 38, 62, 78, 82, 86, 106, 122], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 14, 'token_per_expert': {10: 3, 22: 1, 38: 1, 62: 1, 78: 2, 82: 1, 86: 1, 106: 3, 122: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [5, 25, 61, 125], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {5: 1, 25: 2, 61: 1, 125: 1}}
INFO 05-06 10:42:42.089388.089388 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.309ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:42:42.090113.090113 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.5974044799804688e-05 seconds
INFO 05-06 10:42:42.091428.091428 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012881755828857422 seconds
INFO 05-06 10:42:42.092876.092876 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012600421905517578 seconds
INFO 05-06 10:42:42.094942.094942 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001317739486694336 seconds
INFO 05-06 10:42:42.095018.095018 mlpmodule.py:2799] [fused_experts] gmm total=0.948ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.096296.096296 mlpmodule.py:2799] [fused_experts] gmm total=1.834ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.096026.096026 mlpmodule.py:2799] [fused_experts] gmm total=1.908ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.096274.096274 mlpmodule.py:2799] [fused_experts] gmm total=2.457ms E=32 S=14 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.097887.097887 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0036258697509765625 seconds
INFO 05-06 10:42:42.097784.097784 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.100799560546875e-05 seconds
DEBUG 05-06 10:42:42.098405.098405 cuda_h.py:27] end *layer_moe_fused cost 8.814 ms
DEBUG 05-06 10:42:42.098351.098351 cuda_h.py:27] end decode_layer cost 11.711 ms
DEBUG 05-06 10:42:42.098426.098426 lmp.py:904] ---- decode step 0 layer 20 ----
DEBUG 05-06 10:42:42.100129.100129 cuda_h.py:27] end *sagl cost 1.569 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 95, 107], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {3: 1, 95: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [36, 40, 52, 60, 76, 92], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {36: 1, 40: 2, 52: 1, 60: 1, 76: 1, 92: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 58, 62, 94, 102], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 58: 1, 62: 1, 94: 2, 102: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [13, 21, 45, 65, 73, 85, 117], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 14, 'token_per_expert': {13: 2, 21: 4, 45: 1, 65: 1, 73: 2, 85: 2, 117: 2}}
INFO 05-06 10:42:42.101349.101349 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.298ms | allocate_experts_across_cpu_gpu: 0.088ms
INFO 05-06 10:42:42.101074.101074 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.71661376953125e-05 seconds
INFO 05-06 10:42:42.103727.103727 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012922286987304688 seconds
INFO 05-06 10:42:42.104369.104369 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012793540954589844 seconds
INFO 05-06 10:42:42.105957.105957 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012841224670410156 seconds
INFO 05-06 10:42:42.106441.106441 mlpmodule.py:2799] [fused_experts] gmm total=1.071ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.108960.108960 mlpmodule.py:2799] [fused_experts] gmm total=2.189ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.108052.108052 mlpmodule.py:2799] [fused_experts] gmm total=2.254ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.108076.108076 mlpmodule.py:2799] [fused_experts] gmm total=2.278ms E=32 S=14 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.109739.109739 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003927707672119141 seconds
INFO 05-06 10:42:42.109305.109305 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.1484832763671875e-05 seconds
DEBUG 05-06 10:42:42.110865.110865 cuda_h.py:27] end *layer_moe_fused cost 8.880 ms
DEBUG 05-06 10:42:42.110395.110395 cuda_h.py:27] end decode_layer cost 11.735 ms
DEBUG 05-06 10:42:42.110053.110053 lmp.py:904] ---- decode step 0 layer 21 ----
DEBUG 05-06 10:42:42.112304.112304 cuda_h.py:27] end *sagl cost 1.518 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 59, 71, 83, 87, 103], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {11: 2, 59: 1, 71: 1, 83: 1, 87: 2, 103: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [68, 80, 124], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {68: 1, 80: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 14, 26, 34, 58, 78, 86, 94, 110], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 13, 'token_per_expert': {2: 1, 14: 1, 26: 2, 34: 2, 58: 1, 78: 1, 86: 2, 94: 2, 110: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [5, 21, 25, 57], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {5: 4, 21: 1, 25: 1, 57: 1}}
INFO 05-06 10:42:42.113219.113219 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.297ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:42:42.113321.113321 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6927719116210938e-05 seconds
INFO 05-06 10:42:42.114029.114029 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013320446014404297 seconds
INFO 05-06 10:42:42.116330.116330 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001218557357788086 seconds
INFO 05-06 10:42:42.117580.117580 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012805461883544922 seconds
INFO 05-06 10:42:42.118429.118429 mlpmodule.py:2799] [fused_experts] gmm total=1.093ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.119966.119966 mlpmodule.py:2799] [fused_experts] gmm total=1.935ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.119188.119188 mlpmodule.py:2799] [fused_experts] gmm total=2.055ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.120432.120432 mlpmodule.py:2799] [fused_experts] gmm total=2.329ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.121059.121059 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003564119338989258 seconds
INFO 05-06 10:42:42.121864.121864 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.172325134277344e-05 seconds
DEBUG 05-06 10:42:42.121588.121588 cuda_h.py:27] end *layer_moe_fused cost 8.632 ms
DEBUG 05-06 10:42:42.121919.121919 cuda_h.py:27] end decode_layer cost 11.413 ms
DEBUG 05-06 10:42:42.122755.122755 lmp.py:904] ---- decode step 0 layer 22 ----
DEBUG 05-06 10:42:42.123568.123568 cuda_h.py:27] end *sagl cost 1.510 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [43, 67, 107, 119, 123, 127], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {43: 2, 67: 1, 107: 1, 119: 3, 123: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 16, 24, 32, 60, 76, 108, 120], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {8: 1, 16: 1, 24: 2, 32: 1, 60: 1, 76: 1, 108: 2, 120: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [26, 38, 74, 94], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {26: 1, 38: 2, 74: 1, 94: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [61, 101, 109], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {61: 1, 101: 1, 109: 2}}
INFO 05-06 10:42:42.124353.124353 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.305ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:42:42.124740.124740 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.71661376953125e-05 seconds
INFO 05-06 10:42:42.126558.126558 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012738704681396484 seconds
INFO 05-06 10:42:42.127475.127475 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001220703125 seconds
INFO 05-06 10:42:42.128393.128393 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012297630310058594 seconds
INFO 05-06 10:42:42.130717.130717 mlpmodule.py:2799] [fused_experts] gmm total=1.031ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.131407.131407 mlpmodule.py:2799] [fused_experts] gmm total=2.070ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.131818.131818 mlpmodule.py:2799] [fused_experts] gmm total=2.346ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.131302.131302 mlpmodule.py:2799] [fused_experts] gmm total=2.395ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.132313.132313 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0040285587310791016 seconds
INFO 05-06 10:42:42.133688.133688 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.1484832763671875e-05 seconds
DEBUG 05-06 10:42:42.133004.133004 cuda_h.py:27] end *layer_moe_fused cost 9.174 ms
DEBUG 05-06 10:42:42.134196.134196 cuda_h.py:27] end decode_layer cost 11.978 ms
DEBUG 05-06 10:42:42.134363.134363 lmp.py:904] ---- decode step 0 layer 23 ----
DEBUG 05-06 10:42:42.135943.135943 cuda_h.py:27] end *sagl cost 1.444 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 47, 67], 'expert_count': 3, 'ideal_gpu_count': 4, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {19: 1, 47: 1, 67: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 32, 84, 108], 'expert_count': 5, 'ideal_gpu_count': 4, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {8: 1, 12: 2, 32: 2, 84: 3, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [22, 86, 118], 'expert_count': 3, 'ideal_gpu_count': 4, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 8, 'token_per_expert': {22: 2, 86: 4, 118: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [17, 61, 81, 97, 109], 'expert_count': 5, 'ideal_gpu_count': 4, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 10, 'token_per_expert': {17: 3, 61: 1, 81: 1, 97: 4, 109: 1}}
INFO 05-06 10:42:42.136076.136076 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.293ms | allocate_experts_across_cpu_gpu: 0.077ms
INFO 05-06 10:42:42.136794.136794 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.4066696166992188e-05 seconds
INFO 05-06 10:42:42.138439.138439 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012507438659667969 seconds
INFO 05-06 10:42:42.139897.139897 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0011620521545410156 seconds
INFO 05-06 10:42:42.141333.141333 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001641988754272461 seconds
INFO 05-06 10:42:42.142989.142989 mlpmodule.py:2799] [fused_experts] gmm total=1.057ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.143582.143582 mlpmodule.py:2799] [fused_experts] gmm total=2.031ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.143668.143668 mlpmodule.py:2799] [fused_experts] gmm total=2.136ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.144152.144152 mlpmodule.py:2799] [fused_experts] gmm total=2.267ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.145119.145119 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003948688507080078 seconds
INFO 05-06 10:42:42.145752.145752 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.650520324707031e-05 seconds
DEBUG 05-06 10:42:42.145543.145543 cuda_h.py:27] end *layer_moe_fused cost 9.179 ms
DEBUG 05-06 10:42:42.145318.145318 cuda_h.py:27] end decode_layer cost 11.900 ms
DEBUG 05-06 10:42:42.146154.146154 lmp.py:904] ---- decode step 0 layer 24 ----
DEBUG 05-06 10:42:42.147219.147219 cuda_h.py:27] end *sagl cost 1.484 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [27, 63, 79, 123], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {27: 1, 63: 1, 79: 2, 123: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [12, 40, 44], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 7, 'token_per_expert': {12: 3, 40: 2, 44: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [10, 30, 66, 90, 110, 118], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {10: 1, 30: 2, 66: 2, 90: 1, 110: 2, 118: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [5, 33, 65, 109, 113, 121], 'expert_count': 6, 'ideal_gpu_count': 4, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {5: 1, 33: 3, 65: 1, 109: 2, 113: 2, 121: 1}}
INFO 05-06 10:42:42.148030.148030 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.310ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 10:42:42.148655.148655 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6927719116210938e-05 seconds
INFO 05-06 10:42:42.150757.150757 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012359619140625 seconds
INFO 05-06 10:42:42.151721.151721 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012135505676269531 seconds
INFO 05-06 10:42:42.152020.152020 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011758804321289062 seconds
INFO 05-06 10:42:42.153493.153493 mlpmodule.py:2799] [fused_experts] gmm total=1.125ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.155298.155298 mlpmodule.py:2799] [fused_experts] gmm total=2.070ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.155244.155244 mlpmodule.py:2799] [fused_experts] gmm total=2.131ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.155384.155384 mlpmodule.py:2799] [fused_experts] gmm total=2.279ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.156238.156238 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0038604736328125 seconds
INFO 05-06 10:42:42.156565.156565 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.172325134277344e-05 seconds
DEBUG 05-06 10:42:42.157467.157467 cuda_h.py:27] end *layer_moe_fused cost 8.913 ms
DEBUG 05-06 10:42:42.157342.157342 cuda_h.py:27] end decode_layer cost 11.720 ms
DEBUG 05-06 10:42:42.157416.157416 lmp.py:904] ---- decode step 0 layer 25 ----
DEBUG 05-06 10:42:42.159953.159953 cuda_h.py:27] end *sagl cost 1.552 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 19, 27, 47, 59, 67, 95], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {3: 1, 19: 1, 27: 1, 47: 1, 59: 1, 67: 2, 95: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 44, 52, 68, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 1, 16: 1, 44: 2, 52: 1, 68: 2, 104: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 58], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 5, 'token_per_expert': {6: 1, 58: 4}}
experts_gpu_alloc_device_3 {'expert_ids': [29, 41, 45, 85, 93, 97, 117, 121], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {29: 1, 41: 1, 45: 1, 85: 1, 93: 3, 97: 1, 117: 1, 121: 1}}
INFO 05-06 10:42:42.160925.160925 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.327ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 10:42:42.160749.160749 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:42:42.162526.162526 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012054443359375 seconds
INFO 05-06 10:42:42.163591.163591 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012927055358886719 seconds
INFO 05-06 10:42:42.164955.164955 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013170242309570312 seconds
INFO 05-06 10:42:42.166030.166030 mlpmodule.py:2799] [fused_experts] gmm total=1.117ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.167460.167460 mlpmodule.py:2799] [fused_experts] gmm total=2.142ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.167029.167029 mlpmodule.py:2799] [fused_experts] gmm total=2.207ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.167351.167351 mlpmodule.py:2799] [fused_experts] gmm total=2.242ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.168039.168039 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0038330554962158203 seconds
INFO 05-06 10:42:42.168844.168844 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.076957702636719e-05 seconds
DEBUG 05-06 10:42:42.169175.169175 cuda_h.py:27] end *layer_moe_fused cost 9.073 ms
DEBUG 05-06 10:42:42.169844.169844 cuda_h.py:27] end decode_layer cost 11.967 ms
DEBUG 05-06 10:42:42.169349.169349 lmp.py:904] ---- decode step 0 layer 26 ----
DEBUG 05-06 10:42:42.171834.171834 cuda_h.py:27] end *sagl cost 1.579 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 23, 27, 43, 79, 119], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {19: 2, 23: 1, 27: 1, 43: 3, 79: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 20, 52, 84], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {4: 3, 8: 1, 20: 2, 52: 1, 84: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [10, 38, 62, 70, 90], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {10: 1, 38: 2, 62: 1, 70: 2, 90: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [49, 65, 85], 'expert_count': 3, 'ideal_gpu_count': 4, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 7, 'token_per_expert': {49: 2, 65: 3, 85: 2}}
INFO 05-06 10:42:42.172870.172870 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.088ms
INFO 05-06 10:42:42.172164.172164 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.5974044799804688e-05 seconds
INFO 05-06 10:42:42.174397.174397 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011913776397705078 seconds
INFO 05-06 10:42:42.175963.175963 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012054443359375 seconds
INFO 05-06 10:42:42.176411.176411 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012483596801757812 seconds
INFO 05-06 10:42:42.177950.177950 mlpmodule.py:2799] [fused_experts] gmm total=0.929ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.178300.178300 mlpmodule.py:2799] [fused_experts] gmm total=2.022ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.179100.179100 mlpmodule.py:2799] [fused_experts] gmm total=2.114ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.179975.179975 mlpmodule.py:2799] [fused_experts] gmm total=2.237ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.180506.180506 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00359344482421875 seconds
INFO 05-06 10:42:42.180596.180596 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.124641418457031e-05 seconds
DEBUG 05-06 10:42:42.181385.181385 cuda_h.py:27] end *layer_moe_fused cost 8.636 ms
DEBUG 05-06 10:42:42.181484.181484 cuda_h.py:27] end decode_layer cost 11.541 ms
DEBUG 05-06 10:42:42.181651.181651 lmp.py:904] ---- decode step 0 layer 27 ----
DEBUG 05-06 10:42:42.182967.182967 cuda_h.py:27] end *sagl cost 1.496 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [27, 87, 103, 115], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {27: 1, 87: 1, 103: 3, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [32, 48, 100, 108], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {32: 1, 48: 1, 100: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [14, 50, 58, 62, 82, 114], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {14: 1, 50: 1, 58: 2, 62: 1, 82: 2, 114: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 25, 29, 41, 45, 53, 61, 85, 97, 121], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 14, 'token_per_expert': {1: 1, 25: 1, 29: 2, 41: 2, 45: 1, 53: 1, 61: 1, 85: 2, 97: 1, 121: 2}}
INFO 05-06 10:42:42.184771.184771 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.301ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 10:42:42.184111.184111 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7881393432617188e-05 seconds
INFO 05-06 10:42:42.185494.185494 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012943744659423828 seconds
INFO 05-06 10:42:42.186318.186318 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012221336364746094 seconds
INFO 05-06 10:42:42.190490.190490 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0030829906463623047 seconds
INFO 05-06 10:42:42.191588.191588 mlpmodule.py:2799] [fused_experts] gmm total=1.033ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.192076.192076 mlpmodule.py:2799] [fused_experts] gmm total=2.034ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.192553.192553 mlpmodule.py:2799] [fused_experts] gmm total=2.301ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.192149.192149 mlpmodule.py:2799] [fused_experts] gmm total=2.396ms E=32 S=14 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.193605.193605 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037877559661865234 seconds
INFO 05-06 10:42:42.194602.194602 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.100799560546875e-05 seconds
DEBUG 05-06 10:42:42.194665.194665 cuda_h.py:27] end *layer_moe_fused cost 10.676 ms
DEBUG 05-06 10:42:42.194757.194757 cuda_h.py:27] end decode_layer cost 13.451 ms
DEBUG 05-06 10:42:42.194355.194355 lmp.py:904] ---- decode step 0 layer 28 ----
DEBUG 05-06 10:42:42.196050.196050 cuda_h.py:27] end *sagl cost 1.529 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 39, 67, 111, 115, 119], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 11, 'token_per_expert': {19: 2, 39: 2, 67: 2, 111: 1, 115: 2, 119: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 32, 104, 108, 112], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {20: 1, 32: 2, 104: 2, 108: 1, 112: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [22], 'expert_count': 1, 'ideal_gpu_count': 5, 'keep_on_gpu': 1, 'hit_count_on_device': 1, 'token_total': 2, 'token_per_expert': {22: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [13, 33, 49, 53, 57, 65, 89, 105, 113], 'expert_count': 9, 'ideal_gpu_count': 5, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {13: 1, 33: 1, 49: 3, 53: 1, 57: 2, 65: 1, 89: 1, 105: 1, 113: 1}}
INFO 05-06 10:42:42.197827.197827 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.297ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:42:42.197167.197167 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.6450881958007812e-05 seconds
INFO 05-06 10:42:42.199549.199549 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013027191162109375 seconds
INFO 05-06 10:42:42.200671.200671 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0011949539184570312 seconds
INFO 05-06 10:42:42.202679.202679 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015361309051513672 seconds
INFO 05-06 10:42:42.203196.203196 mlpmodule.py:2799] [fused_experts] gmm total=1.045ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.204275.204275 mlpmodule.py:2799] [fused_experts] gmm total=1.968ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.204652.204652 mlpmodule.py:2799] [fused_experts] gmm total=2.258ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.205484.205484 mlpmodule.py:2799] [fused_experts] gmm total=2.369ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.206697.206697 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003979921340942383 seconds
INFO 05-06 10:42:42.206933.206933 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.172325134277344e-05 seconds
DEBUG 05-06 10:42:42.206240.206240 cuda_h.py:27] end *layer_moe_fused cost 9.366 ms
DEBUG 05-06 10:42:42.207584.207584 cuda_h.py:27] end decode_layer cost 12.172 ms
DEBUG 05-06 10:42:42.207202.207202 lmp.py:904] ---- decode step 0 layer 29 ----
DEBUG 05-06 10:42:42.208289.208289 cuda_h.py:27] end *sagl cost 1.561 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [23, 71], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {23: 3, 71: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 52, 56, 60, 64], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {4: 2, 52: 2, 56: 1, 60: 2, 64: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 18, 26, 30, 66, 78, 82, 106], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 18: 2, 26: 2, 30: 1, 66: 2, 78: 1, 82: 1, 106: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [1, 49, 73, 81, 97], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {1: 1, 49: 1, 73: 2, 81: 3, 97: 1}}
INFO 05-06 10:42:42.210013.210013 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.293ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:42:42.210207.210207 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.5735626220703125e-05 seconds
INFO 05-06 10:42:42.211458.211458 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001310586929321289 seconds
INFO 05-06 10:42:42.212555.212555 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012814998626708984 seconds
INFO 05-06 10:42:42.214673.214673 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012753009796142578 seconds
INFO 05-06 10:42:42.215314.215314 mlpmodule.py:2799] [fused_experts] gmm total=1.012ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.217386.217386 mlpmodule.py:2799] [fused_experts] gmm total=2.600ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.217392.217392 mlpmodule.py:2799] [fused_experts] gmm total=2.675ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.217284.217284 mlpmodule.py:2799] [fused_experts] gmm total=2.698ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:42:42.218772.218772 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004477739334106445 seconds
INFO 05-06 10:42:42.218392.218392 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.291534423828125e-05 seconds
DEBUG 05-06 10:42:42.219199.219199 cuda_h.py:27] end *layer_moe_fused cost 9.686 ms
DEBUG 05-06 10:42:42.219127.219127 cuda_h.py:27] end decode_layer cost 12.579 ms
DEBUG 05-06 10:42:42.219155.219155 cuda_h.py:27] end decode_step cost 367.802 ms
INFO 05-06 10:42:42.219335.219335 lmp.py:931] decode step 0 time: 0.3678312301635742 seconds
WARNING 05-06 10:42:42.225316.225316 helper.py:80] WARNING: Logits have extreme values: min=-896.00, max=1032.00
WARNING 05-06 10:42:42.225061.225061 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 10:42:42.226683.226683 cuda_h.py:27] end init_inputs_tokens cost 6.757 ms
DEBUG 05-06 10:42:42.226619.226619 lmp.py:899] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:42:42.226050.226050 lmp.py:904] ---- decode step 1 layer 0 ----
DEBUG 05-06 10:42:42.228866.228866 cuda_h.py:27] end *sagl cost 1.579 ms
DEBUG 05-06 10:42:42.235518.235518 cuda_h.py:27] end *layer_moe_fused cost 6.114 ms
DEBUG 05-06 10:42:42.236765.236765 cuda_h.py:27] end decode_layer cost 10.051 ms
DEBUG 05-06 10:42:42.237632.237632 lmp.py:904] ---- decode step 1 layer 1 ----
DEBUG 05-06 10:42:42.242036.242036 cuda_h.py:27] end *sagl cost 5.581 ms
DEBUG 05-06 10:42:42.248689.248689 cuda_h.py:27] end *layer_moe_fused cost 3.309 ms
DEBUG 05-06 10:42:42.248027.248027 cuda_h.py:27] end decode_layer cost 11.602 ms
DEBUG 05-06 10:42:42.248851.248851 lmp.py:904] ---- decode step 1 layer 2 ----
DEBUG 05-06 10:42:42.251296.251296 cuda_h.py:27] end *sagl cost 2.331 ms
DEBUG 05-06 10:42:42.255460.255460 cuda_h.py:27] end *layer_moe_fused cost 2.771 ms
DEBUG 05-06 10:42:42.255698.255698 cuda_h.py:27] end decode_layer cost 7.081 ms
DEBUG 05-06 10:42:42.255885.255885 lmp.py:904] ---- decode step 1 layer 3 ----
DEBUG 05-06 10:42:42.257311.257311 cuda_h.py:27] end *sagl cost 1.979 ms
DEBUG 05-06 10:42:42.261239.261239 cuda_h.py:27] end *layer_moe_fused cost 2.866 ms
DEBUG 05-06 10:42:42.262011.262011 cuda_h.py:27] end decode_layer cost 6.514 ms
DEBUG 05-06 10:42:42.262152.262152 lmp.py:904] ---- decode step 1 layer 4 ----
DEBUG 05-06 10:42:42.264698.264698 cuda_h.py:27] end *sagl cost 1.996 ms
DEBUG 05-06 10:42:42.268065.268065 cuda_h.py:27] end *layer_moe_fused cost 2.754 ms
DEBUG 05-06 10:42:42.268950.268950 cuda_h.py:27] end decode_layer cost 6.411 ms
DEBUG 05-06 10:42:42.268138.268138 lmp.py:904] ---- decode step 1 layer 5 ----
DEBUG 05-06 10:42:42.270404.270404 cuda_h.py:27] end *sagl cost 1.964 ms
DEBUG 05-06 10:42:42.274283.274283 cuda_h.py:27] end *layer_moe_fused cost 2.731 ms
DEBUG 05-06 10:42:42.275154.275154 cuda_h.py:27] end decode_layer cost 6.432 ms
DEBUG 05-06 10:42:42.275203.275203 lmp.py:904] ---- decode step 1 layer 6 ----
DEBUG 05-06 10:42:42.277258.277258 cuda_h.py:27] end *sagl cost 1.984 ms
DEBUG 05-06 10:42:42.281526.281526 cuda_h.py:27] end *layer_moe_fused cost 2.754 ms
DEBUG 05-06 10:42:42.281444.281444 cuda_h.py:27] end decode_layer cost 6.387 ms
DEBUG 05-06 10:42:42.281155.281155 lmp.py:904] ---- decode step 1 layer 7 ----
DEBUG 05-06 10:42:42.283726.283726 cuda_h.py:27] end *sagl cost 1.977 ms
DEBUG 05-06 10:42:42.287966.287966 cuda_h.py:27] end *layer_moe_fused cost 2.699 ms
DEBUG 05-06 10:42:42.288790.288790 cuda_h.py:27] end decode_layer cost 6.327 ms
DEBUG 05-06 10:42:42.288455.288455 lmp.py:904] ---- decode step 1 layer 8 ----
DEBUG 05-06 10:42:42.290826.290826 cuda_h.py:27] end *sagl cost 1.936 ms
DEBUG 05-06 10:42:42.294843.294843 cuda_h.py:27] end *layer_moe_fused cost 2.953 ms
DEBUG 05-06 10:42:42.294628.294628 cuda_h.py:27] end decode_layer cost 6.548 ms
DEBUG 05-06 10:42:42.294531.294531 lmp.py:904] ---- decode step 1 layer 9 ----
DEBUG 05-06 10:42:42.297264.297264 cuda_h.py:27] end *sagl cost 2.062 ms
DEBUG 05-06 10:42:42.301052.301052 cuda_h.py:27] end *layer_moe_fused cost 2.826 ms
DEBUG 05-06 10:42:42.301169.301169 cuda_h.py:27] end decode_layer cost 6.574 ms
DEBUG 05-06 10:42:42.301502.301502 lmp.py:904] ---- decode step 1 layer 10 ----
DEBUG 05-06 10:42:42.303783.303783 cuda_h.py:27] end *sagl cost 1.951 ms
DEBUG 05-06 10:42:42.307813.307813 cuda_h.py:27] end *layer_moe_fused cost 2.953 ms
DEBUG 05-06 10:42:42.308360.308360 cuda_h.py:27] end decode_layer cost 6.632 ms
DEBUG 05-06 10:42:42.308117.308117 lmp.py:904] ---- decode step 1 layer 11 ----
DEBUG 05-06 10:42:42.310787.310787 cuda_h.py:27] end *sagl cost 1.945 ms
DEBUG 05-06 10:42:42.314630.314630 cuda_h.py:27] end *layer_moe_fused cost 2.688 ms
DEBUG 05-06 10:42:42.314177.314177 cuda_h.py:27] end decode_layer cost 6.324 ms
DEBUG 05-06 10:42:42.314080.314080 lmp.py:904] ---- decode step 1 layer 12 ----
DEBUG 05-06 10:42:42.316492.316492 cuda_h.py:27] end *sagl cost 1.966 ms
DEBUG 05-06 10:42:42.320200.320200 cuda_h.py:27] end *layer_moe_fused cost 2.822 ms
DEBUG 05-06 10:42:42.321117.321117 cuda_h.py:27] end decode_layer cost 6.448 ms
DEBUG 05-06 10:42:42.321113.321113 lmp.py:904] ---- decode step 1 layer 13 ----
DEBUG 05-06 10:42:42.323015.323015 cuda_h.py:27] end *sagl cost 1.978 ms
DEBUG 05-06 10:42:42.326842.326842 cuda_h.py:27] end *layer_moe_fused cost 2.599 ms
DEBUG 05-06 10:42:42.327038.327038 cuda_h.py:27] end decode_layer cost 6.260 ms
DEBUG 05-06 10:42:42.327510.327510 lmp.py:904] ---- decode step 1 layer 14 ----
DEBUG 05-06 10:42:42.329274.329274 cuda_h.py:27] end *sagl cost 1.981 ms
DEBUG 05-06 10:42:42.333196.333196 cuda_h.py:27] end *layer_moe_fused cost 2.705 ms
DEBUG 05-06 10:42:42.333021.333021 cuda_h.py:27] end decode_layer cost 6.340 ms
DEBUG 05-06 10:42:42.333208.333208 lmp.py:904] ---- decode step 1 layer 15 ----
DEBUG 05-06 10:42:42.336007.336007 cuda_h.py:27] end *sagl cost 2.032 ms
DEBUG 05-06 10:42:42.339775.339775 cuda_h.py:27] end *layer_moe_fused cost 2.661 ms
DEBUG 05-06 10:42:42.340501.340501 cuda_h.py:27] end decode_layer cost 6.356 ms
DEBUG 05-06 10:42:42.340973.340973 lmp.py:904] ---- decode step 1 layer 16 ----
DEBUG 05-06 10:42:42.342775.342775 cuda_h.py:27] end *sagl cost 1.939 ms
DEBUG 05-06 10:42:42.346365.346365 cuda_h.py:27] end *layer_moe_fused cost 2.664 ms
DEBUG 05-06 10:42:42.346958.346958 cuda_h.py:27] end decode_layer cost 6.268 ms
DEBUG 05-06 10:42:42.346384.346384 lmp.py:904] ---- decode step 1 layer 17 ----
DEBUG 05-06 10:42:42.348266.348266 cuda_h.py:27] end *sagl cost 1.964 ms
DEBUG 05-06 10:42:42.352428.352428 cuda_h.py:27] end *layer_moe_fused cost 2.722 ms
DEBUG 05-06 10:42:42.353253.353253 cuda_h.py:27] end decode_layer cost 6.359 ms
DEBUG 05-06 10:42:42.353633.353633 lmp.py:904] ---- decode step 1 layer 18 ----
DEBUG 05-06 10:42:42.355196.355196 cuda_h.py:27] end *sagl cost 1.939 ms
DEBUG 05-06 10:42:42.359261.359261 cuda_h.py:27] end *layer_moe_fused cost 3.005 ms
DEBUG 05-06 10:42:42.359061.359061 cuda_h.py:27] end decode_layer cost 6.666 ms
DEBUG 05-06 10:42:42.359964.359964 lmp.py:904] ---- decode step 1 layer 19 ----
DEBUG 05-06 10:42:42.361596.361596 cuda_h.py:27] end *sagl cost 1.989 ms
DEBUG 05-06 10:42:42.365159.365159 cuda_h.py:27] end *layer_moe_fused cost 2.633 ms
DEBUG 05-06 10:42:42.366222.366222 cuda_h.py:27] end decode_layer cost 6.294 ms
DEBUG 05-06 10:42:42.366933.366933 lmp.py:904] ---- decode step 1 layer 20 ----
DEBUG 05-06 10:42:42.368266.368266 cuda_h.py:27] end *sagl cost 1.980 ms
DEBUG 05-06 10:42:42.372702.372702 cuda_h.py:27] end *layer_moe_fused cost 2.632 ms
DEBUG 05-06 10:42:42.372003.372003 cuda_h.py:27] end decode_layer cost 6.262 ms
DEBUG 05-06 10:42:42.372714.372714 lmp.py:904] ---- decode step 1 layer 21 ----
DEBUG 05-06 10:42:42.374100.374100 cuda_h.py:27] end *sagl cost 1.984 ms
DEBUG 05-06 10:42:42.378973.378973 cuda_h.py:27] end *layer_moe_fused cost 2.742 ms
DEBUG 05-06 10:42:42.379182.379182 cuda_h.py:27] end decode_layer cost 6.449 ms
DEBUG 05-06 10:42:42.379708.379708 lmp.py:904] ---- decode step 1 layer 22 ----
DEBUG 05-06 10:42:42.381345.381345 cuda_h.py:27] end *sagl cost 1.958 ms
DEBUG 05-06 10:42:42.385065.385065 cuda_h.py:27] end *layer_moe_fused cost 2.797 ms
DEBUG 05-06 10:42:42.385750.385750 cuda_h.py:27] end decode_layer cost 6.417 ms
DEBUG 05-06 10:42:42.385461.385461 lmp.py:904] ---- decode step 1 layer 23 ----
DEBUG 05-06 10:42:42.387184.387184 cuda_h.py:27] end *sagl cost 1.952 ms
DEBUG 05-06 10:42:42.391512.391512 cuda_h.py:27] end *layer_moe_fused cost 2.756 ms
DEBUG 05-06 10:42:42.392346.392346 cuda_h.py:27] end decode_layer cost 6.440 ms
DEBUG 05-06 10:42:42.392156.392156 lmp.py:904] ---- decode step 1 layer 24 ----
DEBUG 05-06 10:42:42.394217.394217 cuda_h.py:27] end *sagl cost 1.954 ms
DEBUG 05-06 10:42:42.397798.397798 cuda_h.py:27] end *layer_moe_fused cost 2.584 ms
DEBUG 05-06 10:42:42.398722.398722 cuda_h.py:27] end decode_layer cost 6.205 ms
DEBUG 05-06 10:42:42.398625.398625 lmp.py:904] ---- decode step 1 layer 25 ----
DEBUG 05-06 10:42:42.400458.400458 cuda_h.py:27] end *sagl cost 2.067 ms
DEBUG 05-06 10:42:42.404578.404578 cuda_h.py:27] end *layer_moe_fused cost 2.845 ms
DEBUG 05-06 10:42:42.405138.405138 cuda_h.py:27] end decode_layer cost 6.624 ms
DEBUG 05-06 10:42:42.405472.405472 lmp.py:904] ---- decode step 1 layer 26 ----
DEBUG 05-06 10:42:42.407083.407083 cuda_h.py:27] end *sagl cost 1.973 ms
DEBUG 05-06 10:42:42.411147.411147 cuda_h.py:27] end *layer_moe_fused cost 2.728 ms
DEBUG 05-06 10:42:42.411495.411495 cuda_h.py:27] end decode_layer cost 6.438 ms
DEBUG 05-06 10:42:42.411657.411657 lmp.py:904] ---- decode step 1 layer 27 ----
DEBUG 05-06 10:42:42.413594.413594 cuda_h.py:27] end *sagl cost 2.002 ms
DEBUG 05-06 10:42:42.417315.417315 cuda_h.py:27] end *layer_moe_fused cost 2.812 ms
DEBUG 05-06 10:42:42.418378.418378 cuda_h.py:27] end decode_layer cost 6.496 ms
DEBUG 05-06 10:42:42.418851.418851 lmp.py:904] ---- decode step 1 layer 28 ----
DEBUG 05-06 10:42:42.420839.420839 cuda_h.py:27] end *sagl cost 1.959 ms
DEBUG 05-06 10:42:42.424396.424396 cuda_h.py:27] end *layer_moe_fused cost 2.676 ms
DEBUG 05-06 10:42:42.424744.424744 cuda_h.py:27] end decode_layer cost 6.307 ms
DEBUG 05-06 10:42:42.424693.424693 lmp.py:904] ---- decode step 1 layer 29 ----
DEBUG 05-06 10:42:42.426734.426734 cuda_h.py:27] end *sagl cost 1.939 ms
DEBUG 05-06 10:42:42.430397.430397 cuda_h.py:27] end *layer_moe_fused cost 2.649 ms
DEBUG 05-06 10:42:42.430791.430791 cuda_h.py:27] end decode_layer cost 6.282 ms
DEBUG 05-06 10:42:42.430456.430456 cuda_h.py:27] end decode_step cost 211.018 ms
INFO 05-06 10:42:42.430126.430126 lmp.py:931] decode step 1 time: 0.21105694770812988 seconds
Time taken: 5.988800052553415 seconds
X512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x6291152952c0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
CPUInfer[0x62910ef33910]: Goodbye
