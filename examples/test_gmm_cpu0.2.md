here pin
INFO 05-06 10:49:35.459451.459451 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-06 10:49:35.999143.999143 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-06 10:49:36.425136.425136 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-06 10:49:36.425029.425029 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 0.966s
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
INFO 05-06 10:49:44.031817.031817 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-06 10:49:44.442188.442188 cuda_h.py:27] end init_cmv_hmv cost 411.671 ms
DEBUG 05-06 10:49:44.450147.450147 cuda_memory_view.py:1366] 
DEBUG 05-06 10:49:44.450147.450147 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.0024111270904541016
DEBUG 05-06 10:49:44.465228.465228 mlpmodule.py:993] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-06 10:49:44.466519.466519 cuda_memory_view.py:1370] 
DEBUG 05-06 10:49:44.466519.466519 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.015683412551879883
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-06 10:49:46.368624.368624 lmp.py:368] init kt-kernel layer 0 ok
INFO 05-06 10:49:47.175880.175880 lmp.py:368] init kt-kernel layer 1 ok
INFO 05-06 10:49:47.988222.988222 lmp.py:368] init kt-kernel layer 2 ok
INFO 05-06 10:49:48.822081.822081 lmp.py:368] init kt-kernel layer 3 ok
INFO 05-06 10:49:49.648463.648463 lmp.py:368] init kt-kernel layer 4 ok
INFO 05-06 10:49:50.463358.463358 lmp.py:368] init kt-kernel layer 5 ok
INFO 05-06 10:49:51.296111.296111 lmp.py:368] init kt-kernel layer 6 ok
INFO 05-06 10:49:52.110536.110536 lmp.py:368] init kt-kernel layer 7 ok
INFO 05-06 10:49:52.943143.943143 lmp.py:368] init kt-kernel layer 8 ok
INFO 05-06 10:49:53.769645.769645 lmp.py:368] init kt-kernel layer 9 ok
INFO 05-06 10:49:54.607102.607102 lmp.py:368] init kt-kernel layer 10 ok
INFO 05-06 10:49:55.438880.438880 lmp.py:368] init kt-kernel layer 11 ok
INFO 05-06 10:49:56.288633.288633 lmp.py:368] init kt-kernel layer 12 ok
INFO 05-06 10:49:57.140319.140319 lmp.py:368] init kt-kernel layer 13 ok
INFO 05-06 10:49:57.980537.980537 lmp.py:368] init kt-kernel layer 14 ok
INFO 05-06 10:49:58.828016.828016 lmp.py:368] init kt-kernel layer 15 ok
INFO 05-06 10:49:59.671239.671239 lmp.py:368] init kt-kernel layer 16 ok
INFO 05-06 10:50:00.490003.490003 lmp.py:368] init kt-kernel layer 17 ok
INFO 05-06 10:50:01.326762.326762 lmp.py:368] init kt-kernel layer 18 ok
INFO 05-06 10:50:02.159430.159430 lmp.py:368] init kt-kernel layer 19 ok
INFO 05-06 10:50:02.972285.972285 lmp.py:368] init kt-kernel layer 20 ok
INFO 05-06 10:50:03.790746.790746 lmp.py:368] init kt-kernel layer 21 ok
INFO 05-06 10:50:04.618084.618084 lmp.py:368] init kt-kernel layer 22 ok
CPUInfer[0x63aceca677f0]: Hello
WorkerPool[0x63aceca78d20] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x63ad055e43e0]: Hello
WorkerPool[0x63ad57e2e490] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVINFO 05-06 10:50:05.448434.448434 lmp.py:368] init kt-kernel layer 23 ok
INFO 05-06 10:50:06.299790.299790 lmp.py:368] init kt-kernel layer 24 ok
INFO 05-06 10:50:07.121402.121402 lmp.py:368] init kt-kernel layer 25 ok
INFO 05-06 10:50:07.951117.951117 lmp.py:368] init kt-kernel layer 26 ok
INFO 05-06 10:50:08.790448.790448 lmp.py:368] init kt-kernel layer 27 ok
INFO 05-06 10:50:09.564226.564226 lmp.py:368] init kt-kernel layer 28 ok
INFO 05-06 10:50:10.362102.362102 lmp.py:368] init kt-kernel layer 29 ok
generate input ids cost 0.08877372741699219 s
DEBUG 05-06 10:50:13.596815.596815 cuda_h.py:27] end generate_input_ids cost 3181.561 ms
DEBUG 05-06 10:50:13.597272.597272 cuda_h.py:27] end init_cache cost 0.044 ms
INFO 05-06 10:50:13.610709.610709 lmp.py:2341] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6617276356, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7277174582576769, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-06 10:50:13.610329.610329 lmp.py:2359] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.610728.610728 lmp.py:2359] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.610828.610828 lmp.py:2359] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.610591.610591 lmp.py:2359] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.610354.610354 lmp.py:2359] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.610588.610588 lmp.py:2359] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.610589.610589 lmp.py:2359] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.610160.610160 lmp.py:2359] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.611261.611261 lmp.py:2359] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.611785.611785 lmp.py:2359] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.611521.611521 lmp.py:2359] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.611045.611045 lmp.py:2359] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.611696.611696 lmp.py:2359] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.612703.612703 lmp.py:2359] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.612659.612659 lmp.py:2359] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.612237.612237 lmp.py:2359] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.612384.612384 lmp.py:2359] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.612899.612899 lmp.py:2359] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.612191.612191 lmp.py:2359] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.612920.612920 lmp.py:2359] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613133.613133 lmp.py:2359] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613054.613054 lmp.py:2359] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613340.613340 lmp.py:2359] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613425.613425 lmp.py:2359] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613234.613234 lmp.py:2359] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613179.613179 lmp.py:2359] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613419.613419 lmp.py:2359] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613189.613189 lmp.py:2359] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613789.613789 lmp.py:2359] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:13.613744.613744 lmp.py:2359] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 10:50:13.891264.891264 cuda_h.py:27] end init_loading_placement cost 294.384 ms
DEBUG 05-06 10:50:13.891699.891699 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:50:13.891470.891470 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:50:13 client.py:72] load_into_gpu: gemma4-26B-A4B, 28112fe6-0b59-4400-87d2-edd54a11512f
INFO 05-06 10:50:13 client.py:135] Model loaded: gemma4-26B-A4B, 28112fe6-0b59-4400-87d2-edd54a11512f
INFO 05-06 10:50:13 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 28112fe6-0b59-4400-87d2-edd54a11512f
INFO 05-06 10:50:14 client.py:212] Model loaded
DEBUG 05-06 10:50:14.419591.419591 cuda_h.py:27] end init_general_sagl_loading_async cost 527.826 ms
INFO 05-06 10:50:14.468662.468662 lmp.py:2862] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 10:50:14.571695.571695 cuda_h.py:27] end restore_state_dict cost 102.911 ms
DEBUG 05-06 10:50:14.571647.571647 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:50:14.571099.571099 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:50:14 client.py:72] load_into_gpu: gemma4-26B-A4B, 8338b8cf-ee25-4cc6-b4c9-b9b04efda045
INFO 05-06 10:50:14 client.py:135] Model loaded: gemma4-26B-A4B, 8338b8cf-ee25-4cc6-b4c9-b9b04efda045
DEBUG 05-06 10:50:14.646255.646255 cuda_h.py:27] end init_experts_loading_async cost 74.729 ms
DEBUG 05-06 10:50:14.677262.677262 cuda_h.py:27] end init_inputs_tokens cost 30.849 ms
DEBUG 05-06 10:50:14.677616.677616 lmp.py:824] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 10:50:14.817445.817445 cuda_h.py:27] end *sagl cost 139.800 ms
experts_cpu_alloc {'expert_ids': [19, 87, 27, 15, 119, 88, 120, 12, 36, 100, 96, 8, 17, 85, 97, 109, 101, 29, 81, 30, 86, 6, 66], 'token_total': 119, 'token_per_expert': {19: 2, 87: 2, 27: 10, 15: 13, 119: 13, 88: 1, 120: 1, 12: 2, 36: 4, 100: 5, 96: 8, 8: 9, 17: 3, 85: 3, 97: 3, 109: 3, 101: 6, 29: 7, 81: 11, 30: 1, 86: 2, 6: 4, 66: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 31, 39, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 91, 99, 103, 107, 111, 115, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 5050, 'token_per_expert': {3: 160, 7: 374, 11: 31, 23: 23, 31: 134, 39: 718, 47: 1304, 51: 186, 55: 208, 59: 43, 63: 15, 67: 183, 71: 65, 75: 89, 79: 76, 83: 105, 91: 458, 99: 161, 103: 432, 107: 25, 111: 23, 115: 89, 123: 39, 127: 109}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 28, 32, 44, 48, 52, 60, 64, 68, 72, 76, 80, 84, 92, 104, 108, 112, 116, 124], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 3427, 'token_per_expert': {0: 249, 4: 11, 16: 201, 20: 15, 24: 81, 28: 123, 32: 183, 44: 17, 48: 146, 52: 150, 60: 55, 64: 106, 68: 694, 72: 100, 76: 74, 80: 28, 84: 21, 92: 87, 104: 134, 108: 78, 112: 68, 116: 82, 124: 724}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 33, 37, 41, 45, 49, 53, 65, 69, 73, 77, 89, 93, 105, 113, 117, 121, 125], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 3741, 'token_per_expert': {1: 273, 5: 66, 9: 68, 13: 61, 21: 171, 25: 110, 33: 828, 37: 81, 41: 142, 45: 14, 49: 17, 53: 819, 65: 39, 69: 78, 73: 60, 77: 99, 89: 133, 93: 12, 105: 89, 113: 157, 117: 97, 121: 226, 125: 101}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 18, 22, 26, 34, 38, 46, 50, 54, 70, 74, 78, 90, 94, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 27, 'token_total': 4047, 'token_per_expert': {2: 30, 10: 36, 14: 29, 18: 71, 22: 255, 26: 304, 34: 38, 38: 59, 46: 450, 50: 520, 54: 275, 70: 140, 74: 224, 78: 109, 90: 546, 94: 24, 102: 74, 106: 14, 110: 83, 114: 48, 118: 89, 122: 114, 126: 515}}
INFO 05-06 10:50:14.977658.977658 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 61.889ms | allocate_experts_across_cpu_gpu: 0.313ms
INFO 05-06 10:50:14.977578.977578 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.054473876953125e-05 seconds
INFO 05-06 10:50:14.979425.979425 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0019097328186035156 seconds
INFO 05-06 10:50:14.980137.980137 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007977485656738281 seconds
INFO 05-06 10:50:15.044900.044900 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.06379532814025879 seconds
INFO 05-06 10:50:15.046285.046285 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012578964233398438 seconds
INFO 05-06 10:50:15.084292.084292 mlpmodule.py:2799] [fused_experts] gmm total=37.460ms E=32 S=5090 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.112774.112774 mlpmodule.py:2799] [fused_experts] gmm total=65.128ms E=32 S=3777 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.113893.113893 mlpmodule.py:2799] [fused_experts] gmm total=66.091ms E=32 S=4060 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.114131.114131 mlpmodule.py:2799] [fused_experts] gmm total=67.345ms E=32 S=3457 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.115734.115734 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0692894458770752 seconds
INFO 05-06 10:50:15.115473.115473 lmp.py:1496] [layer_moe_fused] to time: 9.1552734375e-05 seconds
INFO 05-06 10:50:15.116943.116943 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0005061626434326172 seconds
DEBUG 05-06 10:50:15.116756.116756 cuda_h.py:27] end *layer_moe_fused cost 201.402 ms
DEBUG 05-06 10:50:15.135175.135175 cuda_h.py:27] end prefill_layer cost 458.344 ms
DEBUG 05-06 10:50:15.135939.135939 lmp.py:841] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 10:50:15.135132.135132 lmp.py:824] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 10:50:15.138275.138275 cuda_h.py:27] end *sagl cost 2.839 ms
experts_cpu_alloc {'expert_ids': [63, 39, 43, 107, 23, 24, 44, 16, 40, 112, 0, 116, 61, 117, 33, 125, 41, 81, 70, 114, 2, 18, 14], 'token_total': 206, 'token_per_expert': {63: 3, 39: 5, 43: 5, 107: 5, 23: 6, 24: 3, 44: 6, 16: 7, 40: 20, 112: 20, 0: 23, 116: 24, 61: 2, 117: 2, 33: 5, 125: 8, 41: 10, 81: 10, 70: 2, 114: 5, 2: 8, 18: 12, 14: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 27, 31, 35, 47, 51, 55, 59, 67, 75, 79, 83, 87, 91, 95, 99, 103, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 2667, 'token_per_expert': {3: 61, 7: 155, 11: 48, 15: 21, 27: 33, 31: 14, 35: 64, 47: 157, 51: 234, 55: 15, 59: 153, 67: 450, 75: 7, 79: 74, 83: 35, 87: 20, 91: 14, 95: 62, 99: 549, 103: 22, 115: 8, 119: 148, 123: 36, 127: 287}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 20, 28, 32, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4353, 'token_per_expert': {4: 84, 8: 473, 12: 218, 20: 207, 28: 190, 32: 26, 48: 39, 52: 1229, 56: 49, 60: 26, 64: 91, 68: 692, 72: 28, 76: 29, 80: 150, 84: 26, 88: 26, 92: 67, 96: 203, 100: 203, 104: 64, 108: 50, 120: 101, 124: 82}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 29, 37, 45, 49, 53, 57, 65, 69, 73, 77, 85, 89, 93, 97, 101, 105, 109, 121], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4033, 'token_per_expert': {1: 142, 5: 409, 9: 102, 13: 1122, 21: 71, 25: 164, 29: 27, 37: 36, 45: 47, 49: 105, 53: 157, 57: 28, 65: 154, 69: 40, 73: 96, 77: 15, 85: 105, 89: 14, 93: 26, 97: 486, 101: 52, 105: 71, 109: 531, 121: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 22, 26, 30, 34, 38, 42, 46, 50, 54, 62, 66, 74, 78, 82, 90, 94, 98, 106, 110, 118, 122], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 28, 'token_total': 5125, 'token_per_expert': {6: 35, 10: 620, 22: 468, 26: 17, 30: 971, 34: 65, 38: 40, 42: 161, 46: 197, 50: 49, 54: 231, 62: 24, 66: 29, 74: 63, 78: 29, 82: 789, 90: 61, 94: 119, 98: 62, 106: 215, 110: 24, 118: 290, 122: 566}}
INFO 05-06 10:50:15.140417.140417 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.531ms | allocate_experts_across_cpu_gpu: 0.309ms
INFO 05-06 10:50:15.140732.140732 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.982948303222656e-05 seconds
INFO 05-06 10:50:15.142321.142321 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015065670013427734 seconds
INFO 05-06 10:50:15.143263.143263 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010962486267089844 seconds
INFO 05-06 10:50:15.175659.175659 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.031538963317871094 seconds
INFO 05-06 10:50:15.176251.176251 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011839866638183594 seconds
INFO 05-06 10:50:15.179738.179738 mlpmodule.py:2799] [fused_experts] gmm total=3.051ms E=32 S=2691 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.180523.180523 mlpmodule.py:2799] [fused_experts] gmm total=3.292ms E=32 S=4456 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.180890.180890 mlpmodule.py:2799] [fused_experts] gmm total=3.503ms E=32 S=4070 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.181987.181987 mlpmodule.py:2799] [fused_experts] gmm total=4.584ms E=32 S=5167 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.184419.184419 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007917165756225586 seconds
INFO 05-06 10:50:15.184768.184768 lmp.py:1496] [layer_moe_fused] to time: 4.744529724121094e-05 seconds
INFO 05-06 10:50:15.185877.185877 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00034928321838378906 seconds
DEBUG 05-06 10:50:15.185144.185144 cuda_h.py:27] end *layer_moe_fused cost 45.302 ms
DEBUG 05-06 10:50:15.208528.208528 cuda_h.py:27] end prefill_layer cost 72.158 ms
DEBUG 05-06 10:50:15.208253.208253 lmp.py:841] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 10:50:15.208069.208069 lmp.py:824] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 10:50:15.210076.210076 cuda_h.py:27] end *sagl cost 1.768 ms
experts_cpu_alloc {'expert_ids': [47, 87, 79, 67, 75, 99, 23, 32, 16, 68, 12, 64, 40, 93, 117, 101, 25, 21, 45, 22, 74, 86, 66], 'token_total': 178, 'token_per_expert': {47: 1, 87: 1, 79: 4, 67: 6, 75: 6, 99: 6, 23: 28, 32: 3, 16: 4, 68: 6, 12: 18, 64: 18, 40: 21, 93: 1, 117: 1, 101: 2, 25: 8, 21: 10, 45: 18, 22: 1, 74: 1, 86: 5, 66: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 35, 43, 51, 55, 59, 63, 71, 83, 91, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4741, 'token_per_expert': {3: 138, 7: 257, 11: 1057, 15: 370, 19: 582, 27: 47, 31: 100, 35: 28, 43: 75, 51: 103, 55: 220, 59: 446, 63: 92, 71: 67, 83: 96, 91: 149, 95: 29, 103: 66, 107: 79, 111: 46, 115: 39, 119: 81, 123: 104, 127: 470}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 24, 28, 36, 44, 48, 52, 56, 60, 72, 76, 80, 84, 88, 96, 100, 104, 108, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3606, 'token_per_expert': {0: 50, 4: 81, 8: 117, 20: 216, 24: 69, 28: 55, 36: 92, 44: 100, 48: 233, 52: 52, 56: 61, 60: 214, 72: 51, 76: 269, 80: 234, 84: 230, 88: 46, 96: 34, 100: 72, 104: 152, 108: 984, 116: 73, 120: 34, 124: 87}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 29, 33, 37, 41, 49, 53, 57, 61, 65, 69, 77, 81, 85, 97, 105, 109, 113, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4579, 'token_per_expert': {1: 412, 5: 20, 9: 431, 13: 398, 17: 74, 29: 311, 33: 78, 37: 273, 41: 543, 49: 117, 53: 181, 57: 124, 61: 29, 65: 180, 69: 109, 77: 99, 81: 391, 85: 66, 97: 137, 105: 36, 109: 144, 113: 43, 121: 29, 125: 354}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 14, 18, 26, 34, 42, 46, 50, 54, 58, 62, 70, 78, 82, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 27, 'token_total': 3280, 'token_per_expert': {6: 25, 14: 124, 18: 169, 26: 12, 34: 151, 42: 37, 46: 47, 50: 12, 54: 392, 58: 41, 62: 566, 70: 68, 78: 205, 82: 25, 90: 249, 98: 87, 102: 329, 106: 181, 110: 100, 114: 28, 118: 218, 122: 109, 126: 105}}
INFO 05-06 10:50:15.212730.212730 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 1.277ms | allocate_experts_across_cpu_gpu: 0.279ms
INFO 05-06 10:50:15.212761.212761 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.413459777832031e-05 seconds
INFO 05-06 10:50:15.214960.214960 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011398792266845703 seconds
INFO 05-06 10:50:15.214008.214008 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007183551788330078 seconds
INFO 05-06 10:50:15.243112.243112 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.028708934783935547 seconds
INFO 05-06 10:50:15.244925.244925 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010688304901123047 seconds
INFO 05-06 10:50:15.247616.247616 mlpmodule.py:2799] [fused_experts] gmm total=2.068ms E=32 S=3296 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.251438.251438 mlpmodule.py:2799] [fused_experts] gmm total=5.939ms E=32 S=3676 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.251171.251171 mlpmodule.py:2799] [fused_experts] gmm total=6.809ms E=32 S=4619 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.273610.273610 mlpmodule.py:2799] [fused_experts] gmm total=28.140ms E=32 S=4793 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.273045.273045 lmp.py:1484] [layer_moe_fused] experts compute time: 0.028629541397094727 seconds
INFO 05-06 10:50:15.273640.273640 lmp.py:1496] [layer_moe_fused] to time: 4.6253204345703125e-05 seconds
INFO 05-06 10:50:15.273276.273276 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002455711364746094 seconds
DEBUG 05-06 10:50:15.274468.274468 cuda_h.py:27] end *layer_moe_fused cost 62.952 ms
DEBUG 05-06 10:50:15.274881.274881 cuda_h.py:27] end prefill_layer cost 66.780 ms
DEBUG 05-06 10:50:15.275386.275386 lmp.py:841] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 10:50:15.275438.275438 lmp.py:824] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 10:50:15.277938.277938 cuda_h.py:27] end *sagl cost 1.841 ms
experts_cpu_alloc {'expert_ids': [99, 103, 35, 87, 12, 36, 20, 80, 32, 72, 45, 81, 113, 105, 21, 29, 125, 126, 38, 46, 106, 18, 94, 98], 'token_total': 149, 'token_per_expert': {99: 1, 103: 2, 35: 7, 87: 7, 12: 3, 36: 6, 20: 7, 80: 8, 32: 16, 72: 25, 45: 1, 81: 1, 113: 1, 105: 2, 21: 4, 29: 5, 125: 6, 126: 1, 38: 2, 46: 2, 106: 4, 18: 9, 94: 10, 98: 19}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 23, 27, 31, 39, 43, 51, 55, 59, 63, 67, 71, 75, 83, 91, 95, 107, 111, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 29, 'token_total': 2770, 'token_per_expert': {3: 108, 11: 130, 15: 188, 19: 108, 23: 17, 27: 25, 31: 64, 39: 89, 43: 84, 51: 161, 55: 17, 59: 97, 63: 86, 67: 54, 71: 329, 75: 395, 83: 183, 91: 18, 95: 201, 107: 110, 111: 63, 115: 9, 119: 103, 123: 83, 127: 48}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 24, 28, 40, 44, 48, 52, 56, 60, 64, 68, 76, 84, 88, 92, 96, 100, 104, 108, 116, 120], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3877, 'token_per_expert': {0: 160, 4: 290, 8: 51, 16: 35, 24: 42, 28: 251, 40: 57, 44: 99, 48: 44, 52: 284, 56: 39, 60: 36, 64: 103, 68: 208, 76: 247, 84: 245, 88: 301, 92: 283, 96: 317, 100: 74, 104: 235, 108: 158, 116: 51, 120: 267}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 25, 33, 37, 41, 53, 57, 61, 65, 69, 73, 77, 85, 89, 93, 97, 101, 109, 117, 121], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3839, 'token_per_expert': {1: 35, 5: 286, 9: 318, 13: 68, 17: 179, 25: 179, 33: 52, 37: 8, 41: 32, 53: 253, 57: 24, 61: 71, 65: 16, 69: 154, 73: 248, 77: 85, 85: 546, 89: 22, 93: 344, 97: 268, 101: 87, 109: 77, 117: 44, 121: 443}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 26, 30, 34, 42, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 102, 110, 114, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 5749, 'token_per_expert': {2: 113, 6: 80, 10: 164, 14: 310, 22: 544, 26: 90, 30: 41, 34: 265, 42: 29, 50: 685, 54: 197, 58: 104, 62: 430, 66: 341, 70: 149, 74: 162, 78: 695, 82: 21, 86: 34, 102: 426, 110: 107, 114: 126, 118: 247, 122: 389}}
INFO 05-06 10:50:15.279422.279422 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.681ms | allocate_experts_across_cpu_gpu: 0.498ms
INFO 05-06 10:50:15.279442.279442 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 0.00012445449829101562 seconds
INFO 05-06 10:50:15.281709.281709 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013148784637451172 seconds
INFO 05-06 10:50:15.283062.283062 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0015480518341064453 seconds
INFO 05-06 10:50:15.315687.315687 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.03258776664733887 seconds
INFO 05-06 10:50:15.316051.316051 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010852813720703125 seconds
INFO 05-06 10:50:15.319236.319236 mlpmodule.py:2799] [fused_experts] gmm total=2.443ms E=32 S=3859 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.319029.319029 mlpmodule.py:2799] [fused_experts] gmm total=2.630ms E=32 S=3942 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.320643.320643 mlpmodule.py:2799] [fused_experts] gmm total=3.636ms E=32 S=2787 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.322866.322866 mlpmodule.py:2799] [fused_experts] gmm total=5.386ms E=32 S=5796 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.324272.324272 lmp.py:1484] [layer_moe_fused] experts compute time: 0.007536411285400391 seconds
INFO 05-06 10:50:15.324058.324058 lmp.py:1496] [layer_moe_fused] to time: 4.982948303222656e-05 seconds
INFO 05-06 10:50:15.324606.324606 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002193450927734375 seconds
DEBUG 05-06 10:50:15.325006.325006 cuda_h.py:27] end *layer_moe_fused cost 46.907 ms
DEBUG 05-06 10:50:15.346967.346967 cuda_h.py:27] end prefill_layer cost 71.763 ms
DEBUG 05-06 10:50:15.346671.346671 lmp.py:841] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 10:50:15.347566.347566 lmp.py:824] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 10:50:15.348887.348887 cuda_h.py:27] end *sagl cost 1.644 ms
experts_cpu_alloc {'expert_ids': [35, 95, 127, 79, 31, 100, 48, 16, 12, 72, 13, 9, 65, 121, 41, 21, 69, 42, 70, 102, 14, 2, 110, 50], 'token_total': 171, 'token_per_expert': {35: 1, 95: 4, 127: 8, 79: 9, 31: 12, 100: 1, 48: 2, 16: 3, 12: 11, 72: 12, 13: 1, 9: 2, 65: 4, 121: 5, 41: 15, 21: 27, 69: 27, 42: 1, 70: 1, 102: 1, 14: 2, 2: 6, 110: 7, 50: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 83, 87, 91, 103, 107, 111, 115, 119, 123], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 6932, 'token_per_expert': {3: 154, 7: 30, 15: 55, 19: 168, 23: 465, 27: 271, 39: 208, 43: 566, 47: 184, 51: 295, 55: 241, 59: 643, 63: 1029, 67: 273, 71: 116, 75: 66, 83: 427, 87: 109, 91: 71, 103: 35, 107: 63, 111: 454, 115: 396, 119: 504, 123: 109}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 20, 24, 28, 32, 36, 40, 44, 52, 56, 60, 64, 76, 80, 84, 88, 92, 96, 104, 108, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 3037, 'token_per_expert': {4: 139, 8: 712, 20: 137, 24: 313, 28: 124, 32: 151, 36: 60, 40: 52, 44: 29, 52: 83, 56: 33, 60: 129, 64: 61, 76: 208, 80: 20, 84: 48, 88: 42, 92: 141, 96: 106, 104: 123, 108: 83, 116: 97, 120: 15, 124: 131}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 25, 29, 37, 45, 49, 53, 57, 61, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3092, 'token_per_expert': {1: 297, 5: 182, 17: 105, 25: 52, 29: 203, 37: 33, 45: 57, 49: 78, 53: 250, 57: 52, 61: 106, 73: 36, 77: 51, 81: 66, 85: 124, 89: 451, 93: 197, 97: 115, 101: 52, 105: 112, 109: 46, 113: 247, 117: 74, 125: 106}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 18, 22, 26, 30, 34, 38, 46, 54, 58, 62, 66, 74, 78, 82, 86, 90, 94, 98, 106, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3152, 'token_per_expert': {6: 32, 18: 34, 22: 412, 26: 350, 30: 82, 34: 33, 38: 31, 46: 25, 54: 346, 58: 12, 62: 94, 66: 15, 74: 411, 78: 98, 82: 256, 86: 109, 90: 43, 94: 84, 98: 77, 106: 502, 114: 16, 118: 51, 122: 25, 126: 14}}
INFO 05-06 10:50:15.351850.351850 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 1.312ms | allocate_experts_across_cpu_gpu: 0.266ms
INFO 05-06 10:50:15.351198.351198 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.2928924560546875e-05 seconds
INFO 05-06 10:50:15.352744.352744 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010347366333007812 seconds
INFO 05-06 10:50:15.352995.352995 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005185604095458984 seconds
INFO 05-06 10:50:15.376487.376487 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.023518085479736328 seconds
INFO 05-06 10:50:15.377198.377198 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009627342224121094 seconds
INFO 05-06 10:50:15.380898.380898 mlpmodule.py:2799] [fused_experts] gmm total=2.238ms E=32 S=3066 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.380457.380457 mlpmodule.py:2799] [fused_experts] gmm total=2.784ms E=32 S=3179 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.380057.380057 mlpmodule.py:2799] [fused_experts] gmm total=3.090ms E=32 S=6966 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.381811.381811 mlpmodule.py:2799] [fused_experts] gmm total=3.140ms E=32 S=3173 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.383397.383397 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005479335784912109 seconds
INFO 05-06 10:50:15.383183.383183 lmp.py:1496] [layer_moe_fused] to time: 4.839897155761719e-05 seconds
INFO 05-06 10:50:15.383149.383149 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002448558807373047 seconds
DEBUG 05-06 10:50:15.383628.383628 cuda_h.py:27] end *layer_moe_fused cost 34.105 ms
DEBUG 05-06 10:50:15.400561.400561 cuda_h.py:27] end prefill_layer cost 53.637 ms
DEBUG 05-06 10:50:15.400470.400470 lmp.py:841] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 10:50:15.400333.400333 lmp.py:824] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 10:50:15.442764.442764 cuda_h.py:27] end *sagl cost 41.498 ms
experts_cpu_alloc {'expert_ids': [103, 11, 95, 15, 27, 124, 8, 48, 56, 32, 92, 69, 45, 21, 66, 82, 110, 78, 30, 38, 62, 122], 'token_total': 111, 'token_per_expert': {103: 1, 11: 2, 95: 2, 15: 6, 27: 9, 124: 2, 8: 3, 48: 6, 56: 7, 32: 9, 92: 10, 69: 1, 45: 2, 21: 3, 66: 1, 82: 3, 110: 5, 78: 6, 30: 7, 38: 8, 62: 9, 122: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 31, 39, 43, 51, 55, 63, 67, 71, 75, 79, 83, 87, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 28, 'token_total': 4153, 'token_per_expert': {3: 515, 7: 530, 19: 21, 23: 129, 31: 44, 39: 410, 43: 87, 51: 9, 55: 51, 63: 83, 67: 79, 71: 867, 75: 90, 79: 42, 83: 42, 87: 156, 99: 216, 107: 28, 111: 219, 115: 11, 119: 58, 123: 149, 127: 317}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 28, 36, 44, 52, 60, 64, 68, 72, 76, 80, 84, 88, 96, 100, 104, 112, 116, 120], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 4709, 'token_per_expert': {0: 571, 4: 652, 16: 339, 20: 467, 24: 147, 28: 163, 36: 266, 44: 52, 52: 47, 60: 117, 64: 343, 68: 33, 72: 202, 76: 85, 80: 90, 84: 36, 88: 201, 96: 86, 100: 60, 104: 142, 112: 385, 116: 123, 120: 102}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 29, 33, 37, 41, 49, 53, 57, 61, 73, 77, 81, 93, 97, 101, 105, 113, 117, 125], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 26, 'token_total': 4438, 'token_per_expert': {1: 515, 5: 608, 9: 152, 13: 197, 17: 7, 29: 117, 33: 355, 37: 27, 41: 6, 49: 499, 53: 6, 57: 21, 61: 249, 73: 162, 77: 22, 81: 12, 93: 115, 97: 9, 101: 987, 105: 29, 113: 55, 117: 231, 125: 57}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 42, 46, 50, 54, 58, 70, 74, 86, 94, 98, 102, 106, 114, 118, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 31, 'token_total': 2973, 'token_per_expert': {2: 843, 6: 548, 10: 27, 14: 42, 18: 67, 22: 311, 26: 18, 34: 20, 42: 196, 46: 117, 50: 12, 54: 23, 58: 17, 70: 138, 74: 104, 86: 22, 94: 139, 98: 28, 102: 24, 106: 51, 114: 43, 118: 84, 126: 99}}
INFO 05-06 10:50:15.444257.444257 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 1.168ms | allocate_experts_across_cpu_gpu: 0.286ms
INFO 05-06 10:50:15.445420.445420 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 4.982948303222656e-05 seconds
INFO 05-06 10:50:15.446017.446017 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001373291015625 seconds
INFO 05-06 10:50:15.447602.447602 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007224082946777344 seconds
INFO 05-06 10:50:15.477916.477916 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02982807159423828 seconds
INFO 05-06 10:50:15.478135.478135 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011219978332519531 seconds
INFO 05-06 10:50:15.481730.481730 mlpmodule.py:2799] [fused_experts] gmm total=2.321ms E=32 S=3021 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.481441.481441 mlpmodule.py:2799] [fused_experts] gmm total=2.990ms E=32 S=4173 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.483663.483663 mlpmodule.py:2799] [fused_experts] gmm total=4.985ms E=32 S=4746 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.484215.484215 mlpmodule.py:2799] [fused_experts] gmm total=5.193ms E=32 S=4444 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.484934.484934 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00638127326965332 seconds
INFO 05-06 10:50:15.485819.485819 lmp.py:1496] [layer_moe_fused] to time: 4.863739013671875e-05 seconds
INFO 05-06 10:50:15.485871.485871 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00024008750915527344 seconds
DEBUG 05-06 10:50:15.485297.485297 cuda_h.py:27] end *layer_moe_fused cost 42.007 ms
DEBUG 05-06 10:50:15.506580.506580 cuda_h.py:27] end prefill_layer cost 105.477 ms
DEBUG 05-06 10:50:15.506138.506138 lmp.py:841] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 10:50:15.506748.506748 lmp.py:824] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 10:50:15.507929.507929 cuda_h.py:27] end *sagl cost 1.611 ms
experts_cpu_alloc {'expert_ids': [63, 55, 31, 59, 83, 15, 84, 88, 100, 8, 92, 72, 49, 33, 21, 97, 81, 109, 66, 38, 114, 22, 18, 74], 'token_total': 160, 'token_per_expert': {63: 1, 55: 2, 31: 7, 59: 7, 83: 9, 15: 11, 84: 1, 88: 1, 100: 2, 8: 3, 92: 4, 72: 6, 49: 2, 33: 3, 21: 6, 97: 7, 81: 8, 109: 9, 66: 2, 38: 7, 114: 10, 22: 11, 18: 15, 74: 26}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 35, 43, 47, 51, 67, 71, 75, 79, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3963, 'token_per_expert': {3: 532, 7: 514, 11: 25, 19: 24, 23: 248, 27: 73, 35: 426, 43: 30, 47: 12, 51: 109, 67: 11, 71: 97, 75: 153, 79: 151, 87: 279, 91: 22, 95: 59, 99: 577, 103: 37, 107: 108, 111: 15, 115: 239, 119: 101, 123: 84, 127: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 20, 24, 28, 32, 36, 40, 44, 52, 56, 60, 64, 68, 76, 80, 96, 104, 108, 112, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3842, 'token_per_expert': {0: 556, 4: 518, 16: 10, 20: 26, 24: 132, 28: 53, 32: 110, 36: 102, 40: 10, 44: 72, 52: 7, 56: 72, 60: 23, 64: 403, 68: 950, 76: 31, 80: 39, 96: 130, 104: 182, 108: 332, 112: 8, 116: 54, 120: 12, 124: 10}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 25, 29, 37, 41, 53, 57, 65, 69, 73, 77, 85, 89, 93, 101, 105, 113, 117, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4033, 'token_per_expert': {1: 581, 5: 570, 9: 133, 13: 199, 17: 10, 25: 611, 29: 12, 37: 28, 41: 32, 53: 341, 57: 39, 65: 260, 69: 71, 73: 66, 77: 47, 85: 52, 89: 45, 93: 470, 101: 12, 105: 53, 113: 99, 117: 130, 121: 124, 125: 48}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 26, 30, 34, 42, 46, 50, 58, 62, 70, 78, 82, 86, 90, 94, 98, 102, 106, 110, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4386, 'token_per_expert': {2: 580, 6: 596, 10: 87, 14: 44, 26: 106, 30: 39, 34: 259, 42: 47, 46: 120, 50: 103, 58: 57, 62: 118, 70: 54, 78: 147, 82: 27, 86: 397, 90: 309, 94: 207, 98: 208, 102: 428, 106: 297, 110: 40, 122: 74, 126: 42}}
INFO 05-06 10:50:15.510503.510503 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 1.303ms | allocate_experts_across_cpu_gpu: 0.267ms
INFO 05-06 10:50:15.510666.510666 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.269050598144531e-05 seconds
INFO 05-06 10:50:15.511685.511685 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009267330169677734 seconds
INFO 05-06 10:50:15.512647.512647 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005784034729003906 seconds
INFO 05-06 10:50:15.540769.540769 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02849411964416504 seconds
INFO 05-06 10:50:15.541855.541855 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009241104125976562 seconds
INFO 05-06 10:50:15.544203.544203 mlpmodule.py:2799] [fused_experts] gmm total=2.264ms E=32 S=4000 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.544340.544340 mlpmodule.py:2799] [fused_experts] gmm total=2.347ms E=32 S=3859 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.544319.544319 mlpmodule.py:2799] [fused_experts] gmm total=2.398ms E=32 S=4457 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.544162.544162 mlpmodule.py:2799] [fused_experts] gmm total=2.898ms E=32 S=4068 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.546097.546097 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004177093505859375 seconds
INFO 05-06 10:50:15.546155.546155 lmp.py:1496] [layer_moe_fused] to time: 5.0067901611328125e-05 seconds
INFO 05-06 10:50:15.546300.546300 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00023984909057617188 seconds
DEBUG 05-06 10:50:15.546157.546157 cuda_h.py:27] end *layer_moe_fused cost 37.727 ms
DEBUG 05-06 10:50:15.569117.569117 cuda_h.py:27] end prefill_layer cost 63.009 ms
DEBUG 05-06 10:50:15.569006.569006 lmp.py:841] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 10:50:15.569808.569808 lmp.py:824] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 10:50:15.571962.571962 cuda_h.py:27] end *sagl cost 1.794 ms
experts_cpu_alloc {'expert_ids': [27, 75, 11, 119, 39, 107, 31, 92, 124, 24, 100, 36, 37, 73, 49, 109, 77, 46, 58, 30, 102, 50, 62, 94], 'token_total': 200, 'token_per_expert': {27: 2, 75: 2, 11: 4, 119: 4, 39: 5, 107: 11, 31: 12, 92: 3, 124: 3, 24: 5, 100: 6, 36: 10, 37: 10, 73: 14, 49: 19, 109: 20, 77: 24, 46: 1, 58: 3, 30: 7, 102: 7, 50: 9, 62: 9, 94: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 35, 43, 47, 51, 55, 59, 63, 67, 71, 79, 83, 87, 91, 95, 99, 103, 111, 115, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 3206, 'token_per_expert': {3: 515, 7: 640, 15: 36, 19: 63, 23: 41, 35: 14, 43: 97, 47: 87, 51: 81, 55: 18, 59: 109, 63: 22, 67: 16, 71: 87, 79: 165, 83: 77, 87: 74, 91: 621, 95: 63, 99: 73, 103: 101, 111: 28, 115: 97, 123: 60, 127: 21}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 28, 32, 44, 48, 52, 56, 60, 64, 68, 72, 80, 84, 88, 96, 104, 108, 112, 116, 120], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4604, 'token_per_expert': {0: 541, 4: 815, 8: 78, 12: 311, 16: 34, 20: 228, 28: 162, 32: 25, 44: 202, 48: 149, 52: 277, 56: 140, 60: 139, 64: 56, 68: 64, 72: 94, 80: 39, 84: 209, 88: 23, 96: 112, 104: 125, 108: 437, 112: 94, 116: 36, 120: 214}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 41, 45, 53, 57, 61, 65, 69, 85, 97, 101, 105, 113, 117, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 4738, 'token_per_expert': {1: 516, 5: 642, 9: 61, 13: 100, 17: 30, 21: 34, 25: 44, 29: 365, 33: 60, 41: 34, 45: 52, 53: 182, 57: 142, 61: 95, 65: 184, 69: 278, 85: 232, 97: 744, 101: 42, 105: 92, 113: 168, 117: 67, 121: 446, 125: 128}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 38, 42, 54, 66, 70, 78, 82, 86, 90, 98, 106, 110, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3636, 'token_per_expert': {2: 512, 6: 568, 10: 205, 14: 178, 18: 95, 22: 63, 26: 31, 34: 285, 38: 11, 42: 156, 54: 42, 66: 43, 70: 259, 78: 17, 82: 31, 86: 176, 90: 244, 98: 50, 106: 155, 110: 184, 114: 201, 118: 49, 122: 52, 126: 29}}
INFO 05-06 10:50:15.573777.573777 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.689ms | allocate_experts_across_cpu_gpu: 0.276ms
INFO 05-06 10:50:15.573894.573894 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.173683166503906e-05 seconds
INFO 05-06 10:50:15.574889.574889 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001013040542602539 seconds
INFO 05-06 10:50:15.575768.575768 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004909038543701172 seconds
INFO 05-06 10:50:15.605752.605752 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.030501365661621094 seconds
INFO 05-06 10:50:15.606212.606212 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009903907775878906 seconds
INFO 05-06 10:50:15.609947.609947 mlpmodule.py:2799] [fused_experts] gmm total=2.161ms E=32 S=3246 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.609702.609702 mlpmodule.py:2799] [fused_experts] gmm total=2.317ms E=32 S=4631 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.609450.609450 mlpmodule.py:2799] [fused_experts] gmm total=2.429ms E=32 S=4825 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.610442.610442 mlpmodule.py:2799] [fused_experts] gmm total=2.931ms E=32 S=3682 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.610153.610153 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0038895606994628906 seconds
INFO 05-06 10:50:15.610839.610839 lmp.py:1496] [layer_moe_fused] to time: 4.744529724121094e-05 seconds
INFO 05-06 10:50:15.611933.611933 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003020763397216797 seconds
DEBUG 05-06 10:50:15.611367.611367 cuda_h.py:27] end *layer_moe_fused cost 38.935 ms
DEBUG 05-06 10:50:15.635049.635049 cuda_h.py:27] end prefill_layer cost 66.321 ms
DEBUG 05-06 10:50:15.635177.635177 lmp.py:841] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 10:50:15.635264.635264 lmp.py:824] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 10:50:15.637265.637265 cuda_h.py:27] end *sagl cost 1.582 ms
experts_cpu_alloc {'expert_ids': [59, 83, 67, 107, 23, 60, 112, 40, 100, 24, 72, 109, 101, 117, 25, 13, 33, 37, 30, 90, 18, 118, 26, 34], 'token_total': 217, 'token_per_expert': {59: 1, 83: 1, 67: 4, 107: 4, 23: 5, 60: 1, 112: 1, 40: 4, 100: 7, 24: 9, 72: 9, 109: 1, 101: 13, 117: 14, 25: 17, 13: 19, 33: 19, 37: 19, 30: 1, 90: 6, 18: 7, 118: 15, 26: 19, 34: 21}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 35, 39, 43, 47, 51, 55, 63, 71, 75, 79, 87, 91, 99, 103, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4158, 'token_per_expert': {3: 616, 7: 520, 11: 48, 15: 152, 19: 209, 27: 141, 31: 94, 35: 8, 39: 9, 43: 24, 47: 53, 51: 462, 55: 139, 63: 114, 71: 169, 75: 201, 79: 9, 87: 287, 91: 22, 99: 21, 103: 581, 111: 60, 119: 22, 123: 129, 127: 68}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 28, 32, 36, 44, 48, 52, 56, 64, 68, 76, 80, 84, 92, 96, 104, 108, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3196, 'token_per_expert': {0: 519, 4: 555, 8: 23, 12: 120, 16: 77, 20: 67, 28: 308, 32: 182, 36: 133, 44: 84, 48: 21, 52: 73, 56: 186, 64: 24, 68: 19, 76: 109, 80: 185, 84: 30, 92: 16, 96: 12, 104: 13, 108: 45, 116: 33, 120: 301, 124: 61}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 29, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 105, 113, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3907, 'token_per_expert': {1: 553, 5: 646, 9: 21, 17: 63, 21: 102, 29: 96, 41: 85, 45: 62, 49: 29, 53: 67, 57: 43, 61: 93, 65: 210, 69: 152, 73: 423, 77: 61, 81: 131, 85: 94, 89: 58, 93: 99, 105: 256, 113: 50, 121: 269, 125: 244}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 98, 102, 106, 110, 114, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4906, 'token_per_expert': {2: 672, 6: 622, 10: 52, 14: 45, 22: 71, 38: 152, 42: 63, 46: 190, 50: 255, 54: 649, 58: 723, 62: 22, 66: 50, 70: 290, 74: 21, 82: 23, 86: 34, 98: 77, 102: 177, 106: 35, 110: 270, 114: 222, 122: 115, 126: 76}}
INFO 05-06 10:50:15.640431.640431 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 1.438ms | allocate_experts_across_cpu_gpu: 0.282ms
INFO 05-06 10:50:15.640647.640647 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.555152893066406e-05 seconds
INFO 05-06 10:50:15.641883.641883 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009813308715820312 seconds
INFO 05-06 10:50:15.642670.642670 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005178451538085938 seconds
INFO 05-06 10:50:15.670477.670477 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.028748750686645508 seconds
INFO 05-06 10:50:15.671739.671739 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010144710540771484 seconds
INFO 05-06 10:50:15.674188.674188 mlpmodule.py:2799] [fused_experts] gmm total=2.127ms E=32 S=4173 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.674816.674816 mlpmodule.py:2799] [fused_experts] gmm total=2.256ms E=32 S=3227 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.674729.674729 mlpmodule.py:2799] [fused_experts] gmm total=2.345ms E=32 S=4009 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.675767.675767 mlpmodule.py:2799] [fused_experts] gmm total=2.847ms E=32 S=4975 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.675567.675567 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00391077995300293 seconds
INFO 05-06 10:50:15.676922.676922 lmp.py:1496] [layer_moe_fused] to time: 4.9114227294921875e-05 seconds
INFO 05-06 10:50:15.676625.676625 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002605915069580078 seconds
DEBUG 05-06 10:50:15.676576.676576 cuda_h.py:27] end *layer_moe_fused cost 38.274 ms
DEBUG 05-06 10:50:15.699717.699717 cuda_h.py:27] end prefill_layer cost 63.563 ms
DEBUG 05-06 10:50:15.699037.699037 lmp.py:841] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 10:50:15.699124.699124 lmp.py:824] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 10:50:15.701556.701556 cuda_h.py:27] end *sagl cost 1.629 ms
experts_cpu_alloc {'expert_ids': [107, 47, 59, 63, 31, 108, 84, 100, 96, 112, 64, 120, 49, 53, 25, 121, 29, 14, 118, 94, 126, 66, 110, 50], 'token_total': 145, 'token_per_expert': {107: 1, 47: 2, 59: 2, 63: 2, 31: 4, 108: 2, 84: 3, 100: 7, 96: 11, 112: 11, 64: 12, 120: 14, 49: 1, 53: 1, 25: 2, 121: 8, 29: 12, 14: 2, 118: 3, 94: 4, 126: 4, 66: 12, 110: 12, 50: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 39, 43, 51, 55, 67, 71, 75, 79, 83, 95, 99, 103, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 4343, 'token_per_expert': {3: 573, 7: 570, 11: 5, 15: 70, 19: 101, 23: 174, 27: 116, 39: 119, 43: 368, 51: 111, 55: 13, 67: 27, 71: 124, 75: 305, 79: 25, 83: 97, 95: 819, 99: 82, 103: 367, 111: 87, 115: 32, 119: 12, 123: 10, 127: 136}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 68, 72, 76, 80, 88, 92, 104, 116, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3885, 'token_per_expert': {0: 522, 4: 654, 8: 26, 12: 562, 16: 310, 20: 23, 24: 121, 28: 15, 32: 128, 36: 137, 40: 180, 44: 30, 48: 199, 52: 22, 56: 210, 68: 83, 72: 111, 76: 152, 80: 42, 88: 83, 92: 158, 104: 25, 116: 36, 124: 56}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 33, 37, 41, 45, 57, 61, 69, 73, 77, 81, 89, 93, 97, 101, 105, 113, 117, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 3776, 'token_per_expert': {1: 665, 5: 563, 9: 84, 13: 133, 17: 57, 21: 119, 33: 14, 37: 87, 41: 18, 45: 81, 57: 155, 61: 139, 69: 231, 73: 32, 77: 17, 81: 235, 89: 131, 93: 330, 97: 47, 101: 416, 105: 30, 113: 36, 117: 42, 125: 114}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 54, 58, 62, 70, 74, 82, 86, 90, 98, 102, 106, 114, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4235, 'token_per_expert': {2: 514, 6: 527, 10: 35, 18: 28, 22: 145, 26: 16, 30: 120, 34: 17, 38: 98, 42: 55, 46: 634, 54: 85, 58: 18, 62: 69, 70: 560, 74: 273, 82: 31, 86: 67, 90: 21, 98: 40, 102: 148, 106: 623, 114: 35, 122: 76}}
INFO 05-06 10:50:15.703186.703186 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.970ms | allocate_experts_across_cpu_gpu: 0.276ms
INFO 05-06 10:50:15.703304.703304 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.724761962890625e-05 seconds
INFO 05-06 10:50:15.704198.704198 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009377002716064453 seconds
INFO 05-06 10:50:15.705416.705416 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005283355712890625 seconds
INFO 05-06 10:50:15.734608.734608 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.029212236404418945 seconds
INFO 05-06 10:50:15.735499.735499 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000985860824584961 seconds
INFO 05-06 10:50:15.738918.738918 mlpmodule.py:2799] [fused_experts] gmm total=2.136ms E=32 S=3800 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.738109.738109 mlpmodule.py:2799] [fused_experts] gmm total=2.331ms E=32 S=3945 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.738131.738131 mlpmodule.py:2799] [fused_experts] gmm total=2.472ms E=32 S=4354 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.738876.738876 mlpmodule.py:2799] [fused_experts] gmm total=2.789ms E=32 S=4285 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.739811.739811 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003914594650268555 seconds
INFO 05-06 10:50:15.739404.739404 lmp.py:1496] [layer_moe_fused] to time: 5.1021575927734375e-05 seconds
INFO 05-06 10:50:15.740872.740872 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003695487976074219 seconds
DEBUG 05-06 10:50:15.740883.740883 cuda_h.py:27] end *layer_moe_fused cost 38.277 ms
DEBUG 05-06 10:50:15.758255.758255 cuda_h.py:27] end prefill_layer cost 59.075 ms
DEBUG 05-06 10:50:15.758926.758926 lmp.py:841] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 10:50:15.758265.758265 lmp.py:824] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 10:50:15.760330.760330 cuda_h.py:27] end *sagl cost 1.944 ms
experts_cpu_alloc {'expert_ids': [23, 123, 35, 55, 51, 59, 24, 36, 52, 40, 48, 65, 17, 77, 53, 101, 109, 25, 22, 122, 38, 114, 30, 66], 'token_total': 108, 'token_per_expert': {23: 1, 123: 1, 35: 2, 55: 3, 51: 7, 59: 7, 24: 1, 36: 1, 52: 1, 40: 4, 48: 5, 65: 2, 17: 3, 77: 5, 53: 7, 101: 8, 109: 13, 25: 15, 22: 1, 122: 1, 38: 4, 114: 4, 30: 5, 66: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 39, 43, 47, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 2946, 'token_per_expert': {3: 531, 7: 623, 11: 41, 15: 16, 19: 53, 27: 24, 31: 101, 39: 167, 43: 85, 47: 129, 63: 80, 67: 44, 71: 123, 75: 143, 79: 50, 83: 89, 87: 11, 91: 11, 99: 101, 103: 17, 107: 11, 111: 48, 115: 261, 119: 32, 127: 155}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 28, 32, 44, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4832, 'token_per_expert': {0: 776, 4: 559, 8: 401, 12: 14, 16: 186, 20: 116, 28: 26, 32: 7, 44: 56, 56: 44, 60: 410, 64: 21, 68: 130, 72: 110, 76: 580, 80: 535, 84: 83, 88: 227, 92: 158, 100: 113, 104: 6, 108: 205, 112: 35, 120: 18, 124: 16}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 29, 33, 37, 41, 49, 57, 61, 69, 73, 81, 85, 89, 93, 97, 105, 113, 117, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4198, 'token_per_expert': {1: 1118, 5: 567, 9: 52, 13: 133, 21: 210, 29: 30, 33: 22, 37: 88, 41: 177, 49: 185, 57: 184, 61: 28, 69: 64, 73: 23, 81: 495, 85: 105, 89: 48, 93: 38, 97: 40, 105: 68, 113: 132, 117: 49, 121: 65, 125: 277}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 26, 34, 42, 46, 50, 54, 58, 62, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4300, 'token_per_expert': {2: 533, 6: 543, 10: 177, 14: 336, 18: 124, 26: 17, 34: 40, 42: 176, 46: 162, 50: 29, 54: 107, 58: 102, 62: 240, 70: 16, 74: 383, 78: 33, 82: 184, 86: 416, 90: 66, 94: 99, 98: 48, 102: 17, 106: 255, 126: 197}}
INFO 05-06 10:50:15.763588.763588 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 1.001ms | allocate_experts_across_cpu_gpu: 0.261ms
INFO 05-06 10:50:15.763997.763997 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.7697296142578125e-05 seconds
INFO 05-06 10:50:15.781500.781500 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.018238306045532227 seconds
INFO 05-06 10:50:15.782625.782625 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006871223449707031 seconds
INFO 05-06 10:50:15.794805.794805 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011639833450317383 seconds
INFO 05-06 10:50:15.795100.795100 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010044574737548828 seconds
INFO 05-06 10:50:15.798318.798318 mlpmodule.py:2799] [fused_experts] gmm total=2.540ms E=32 S=2967 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.798120.798120 mlpmodule.py:2799] [fused_experts] gmm total=2.587ms E=32 S=4251 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.798178.798178 mlpmodule.py:2799] [fused_experts] gmm total=2.802ms E=32 S=4844 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.798261.798261 mlpmodule.py:2799] [fused_experts] gmm total=2.973ms E=32 S=4322 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.799058.799058 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004388570785522461 seconds
INFO 05-06 10:50:15.799651.799651 lmp.py:1496] [layer_moe_fused] to time: 4.9114227294921875e-05 seconds
INFO 05-06 10:50:15.800448.800448 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003306865692138672 seconds
DEBUG 05-06 10:50:15.800294.800294 cuda_h.py:27] end *layer_moe_fused cost 38.756 ms
DEBUG 05-06 10:50:15.807528.807528 cuda_h.py:27] end prefill_layer cost 48.993 ms
DEBUG 05-06 10:50:15.807577.807577 lmp.py:841] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 10:50:15.807155.807155 lmp.py:824] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 10:50:15.811199.811199 cuda_h.py:27] end *sagl cost 2.983 ms
experts_cpu_alloc {'expert_ids': [15, 127, 35, 104, 84, 72, 96, 88, 52, 44, 45, 109, 65, 41, 105, 85, 9, 90, 26, 114, 94, 106, 110], 'token_total': 152, 'token_per_expert': {15: 1, 127: 5, 35: 6, 104: 1, 84: 3, 72: 6, 96: 6, 88: 8, 52: 19, 44: 29, 45: 1, 109: 1, 65: 2, 41: 3, 105: 4, 85: 10, 9: 11, 90: 1, 26: 3, 114: 5, 94: 8, 106: 9, 110: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 31, 39, 43, 47, 51, 59, 63, 67, 71, 79, 83, 87, 91, 99, 111, 115, 119, 123], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 27, 'token_total': 4591, 'token_per_expert': {3: 533, 7: 782, 11: 29, 19: 188, 23: 241, 27: 63, 31: 166, 39: 20, 43: 73, 47: 17, 51: 43, 59: 58, 63: 9, 67: 254, 71: 56, 79: 410, 83: 389, 87: 596, 91: 80, 99: 86, 111: 239, 115: 13, 119: 204, 123: 42}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 68, 76, 80, 92, 100, 108, 112, 116, 120, 124], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 4421, 'token_per_expert': {0: 526, 4: 527, 8: 34, 16: 580, 20: 103, 24: 157, 28: 76, 32: 230, 36: 101, 40: 35, 48: 65, 56: 485, 64: 29, 68: 230, 76: 170, 80: 31, 92: 380, 100: 219, 108: 133, 112: 100, 116: 89, 120: 48, 124: 73}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 21, 25, 29, 33, 37, 49, 53, 57, 61, 69, 77, 81, 89, 93, 97, 113, 117, 121, 125], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 4058, 'token_per_expert': {1: 542, 5: 552, 13: 14, 17: 347, 21: 13, 25: 95, 29: 86, 33: 19, 37: 125, 49: 290, 53: 18, 57: 145, 61: 107, 69: 103, 77: 105, 81: 384, 89: 88, 93: 405, 97: 12, 113: 445, 117: 75, 121: 68, 125: 20}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 98, 102, 118, 122, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 3162, 'token_per_expert': {2: 630, 6: 830, 10: 113, 18: 60, 22: 20, 30: 157, 34: 17, 38: 103, 42: 32, 46: 45, 50: 27, 54: 52, 58: 16, 62: 42, 66: 115, 70: 90, 74: 20, 82: 73, 98: 76, 102: 537, 118: 13, 122: 18, 126: 76}}
INFO 05-06 10:50:15.815580.815580 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 2.460ms | allocate_experts_across_cpu_gpu: 0.448ms
INFO 05-06 10:50:15.815103.815103 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.96453857421875e-05 seconds
INFO 05-06 10:50:15.816757.816757 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006501674652099609 seconds
INFO 05-06 10:50:15.817705.817705 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.000774383544921875 seconds
INFO 05-06 10:50:15.827091.827091 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009856939315795898 seconds
INFO 05-06 10:50:15.828909.828909 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010063648223876953 seconds
INFO 05-06 10:50:15.830463.830463 mlpmodule.py:2799] [fused_experts] gmm total=2.109ms E=32 S=4603 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.830839.830839 mlpmodule.py:2799] [fused_experts] gmm total=2.196ms E=32 S=4493 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.830267.830267 mlpmodule.py:2799] [fused_experts] gmm total=2.287ms E=32 S=4090 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.831500.831500 mlpmodule.py:2799] [fused_experts] gmm total=2.650ms E=32 S=3198 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.832959.832959 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003793478012084961 seconds
INFO 05-06 10:50:15.832738.832738 lmp.py:1496] [layer_moe_fused] to time: 4.76837158203125e-05 seconds
INFO 05-06 10:50:15.832404.832404 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003731250762939453 seconds
DEBUG 05-06 10:50:15.832998.832998 cuda_h.py:27] end *layer_moe_fused cost 20.530 ms
DEBUG 05-06 10:50:15.839739.839739 cuda_h.py:27] end prefill_layer cost 31.497 ms
DEBUG 05-06 10:50:15.839311.839311 lmp.py:841] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 10:50:15.839935.839935 lmp.py:824] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 10:50:15.841427.841427 cuda_h.py:27] end *sagl cost 1.905 ms
experts_cpu_alloc {'expert_ids': [27, 51, 87, 59, 83, 96, 48, 56, 8, 69, 121, 29, 93, 81, 33, 30, 122, 126, 10, 62, 54], 'token_total': 71, 'token_per_expert': {27: 1, 51: 2, 87: 2, 59: 3, 83: 3, 96: 1, 48: 3, 56: 6, 8: 9, 69: 1, 121: 1, 29: 2, 93: 5, 81: 6, 33: 12, 30: 1, 122: 1, 126: 1, 10: 2, 62: 4, 54: 5}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 31, 35, 39, 47, 63, 67, 71, 79, 91, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 27, 'token_total': 3579, 'token_per_expert': {3: 614, 7: 514, 15: 326, 19: 129, 23: 177, 31: 15, 35: 90, 39: 590, 47: 25, 63: 26, 67: 14, 71: 455, 79: 37, 91: 127, 95: 133, 103: 56, 107: 35, 111: 17, 115: 134, 119: 27, 123: 32, 127: 6}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 32, 36, 40, 64, 68, 76, 80, 84, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 26, 'token_total': 3081, 'token_per_expert': {0: 513, 4: 517, 12: 57, 20: 13, 24: 21, 32: 12, 36: 83, 40: 70, 64: 12, 68: 54, 76: 113, 80: 76, 84: 137, 88: 81, 92: 142, 100: 64, 104: 84, 108: 518, 112: 30, 116: 367, 120: 17, 124: 100}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 21, 25, 37, 41, 45, 49, 53, 65, 73, 77, 85, 89, 97, 101, 105, 113, 117, 125], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 28, 'token_total': 4123, 'token_per_expert': {1: 543, 5: 682, 13: 17, 17: 29, 21: 548, 25: 148, 37: 12, 41: 12, 45: 429, 49: 285, 53: 593, 65: 33, 73: 134, 77: 76, 85: 58, 89: 24, 97: 231, 101: 88, 105: 12, 113: 15, 117: 125, 125: 29}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 34, 38, 46, 50, 58, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 28, 'token_total': 5530, 'token_per_expert': {2: 521, 6: 663, 18: 6, 22: 80, 34: 87, 38: 43, 46: 258, 50: 483, 58: 96, 70: 41, 74: 311, 78: 854, 82: 336, 86: 385, 90: 66, 94: 29, 98: 46, 102: 26, 106: 232, 110: 326, 114: 406, 118: 235}}
INFO 05-06 10:50:15.844225.844225 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.962ms | allocate_experts_across_cpu_gpu: 0.419ms
INFO 05-06 10:50:15.844211.844211 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.891654968261719e-05 seconds
INFO 05-06 10:50:15.845998.845998 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006315708160400391 seconds
INFO 05-06 10:50:15.846383.846383 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007452964782714844 seconds
INFO 05-06 10:50:15.857088.857088 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011075496673583984 seconds
INFO 05-06 10:50:15.858976.858976 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009188652038574219 seconds
INFO 05-06 10:50:15.860456.860456 mlpmodule.py:2799] [fused_experts] gmm total=2.095ms E=32 S=3590 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.860804.860804 mlpmodule.py:2799] [fused_experts] gmm total=2.115ms E=32 S=4150 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.860728.860728 mlpmodule.py:2799] [fused_experts] gmm total=2.247ms E=32 S=3100 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.861745.861745 mlpmodule.py:2799] [fused_experts] gmm total=2.652ms E=32 S=5544 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.862891.862891 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037305355072021484 seconds
INFO 05-06 10:50:15.862961.862961 lmp.py:1496] [layer_moe_fused] to time: 4.887580871582031e-05 seconds
INFO 05-06 10:50:15.862083.862083 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00032329559326171875 seconds
DEBUG 05-06 10:50:15.862173.862173 cuda_h.py:27] end *layer_moe_fused cost 19.990 ms
DEBUG 05-06 10:50:15.869271.869271 cuda_h.py:27] end prefill_layer cost 30.402 ms
DEBUG 05-06 10:50:15.869366.869366 lmp.py:841] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 10:50:15.869951.869951 lmp.py:824] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 10:50:15.872395.872395 cuda_h.py:27] end *sagl cost 1.922 ms
experts_cpu_alloc {'expert_ids': [111, 23, 83, 11, 47, 12, 76, 72, 112, 56, 48, 77, 53, 97, 89, 45, 61, 50, 18, 74, 66, 10, 90], 'token_total': 162, 'token_per_expert': {111: 3, 23: 10, 83: 19, 11: 20, 47: 24, 12: 2, 76: 2, 72: 4, 112: 5, 56: 6, 48: 8, 77: 1, 53: 3, 97: 3, 89: 4, 45: 6, 61: 10, 50: 1, 18: 2, 74: 3, 66: 6, 10: 9, 90: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 31, 39, 43, 51, 55, 59, 63, 67, 71, 75, 79, 87, 91, 95, 99, 103, 107, 115, 119, 123], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 4708, 'token_per_expert': {3: 618, 7: 514, 15: 119, 27: 45, 31: 698, 39: 197, 43: 45, 51: 116, 55: 51, 59: 252, 63: 166, 67: 38, 71: 298, 75: 69, 79: 383, 87: 28, 91: 631, 95: 31, 99: 63, 103: 104, 107: 25, 115: 108, 119: 84, 123: 25}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 28, 32, 36, 40, 52, 60, 64, 68, 80, 84, 92, 96, 100, 104, 108, 116, 120, 124], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 3022, 'token_per_expert': {0: 531, 4: 512, 8: 18, 16: 26, 20: 181, 28: 29, 32: 301, 36: 9, 40: 39, 52: 19, 60: 105, 64: 31, 68: 40, 80: 26, 84: 102, 92: 34, 96: 9, 100: 450, 104: 23, 108: 40, 116: 81, 120: 342, 124: 74}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 33, 37, 41, 57, 65, 69, 73, 81, 93, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 4039, 'token_per_expert': {1: 725, 5: 512, 9: 32, 13: 76, 17: 380, 21: 143, 25: 166, 33: 179, 37: 403, 41: 71, 57: 13, 65: 18, 69: 106, 73: 45, 81: 331, 93: 22, 101: 29, 105: 11, 109: 22, 113: 96, 117: 54, 121: 437, 125: 168}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 26, 34, 38, 42, 46, 62, 70, 78, 82, 86, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 4453, 'token_per_expert': {2: 550, 6: 716, 14: 305, 22: 98, 26: 26, 34: 91, 38: 149, 42: 43, 46: 26, 62: 15, 70: 30, 78: 403, 82: 66, 86: 107, 94: 26, 98: 143, 102: 175, 106: 16, 110: 446, 114: 682, 118: 125, 122: 66, 126: 149}}
INFO 05-06 10:50:15.874179.874179 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.523ms | allocate_experts_across_cpu_gpu: 0.435ms
INFO 05-06 10:50:15.874641.874641 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.630752563476562e-05 seconds
INFO 05-06 10:50:15.875287.875287 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006172657012939453 seconds
INFO 05-06 10:50:15.876382.876382 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007777214050292969 seconds
INFO 05-06 10:50:15.885531.885531 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009717941284179688 seconds
INFO 05-06 10:50:15.886406.886406 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009443759918212891 seconds
INFO 05-06 10:50:15.889034.889034 mlpmodule.py:2799] [fused_experts] gmm total=1.869ms E=32 S=3049 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.889692.889692 mlpmodule.py:2799] [fused_experts] gmm total=1.925ms E=32 S=4066 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.889104.889104 mlpmodule.py:2799] [fused_experts] gmm total=2.640ms E=32 S=4784 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.890656.890656 mlpmodule.py:2799] [fused_experts] gmm total=2.758ms E=32 S=4485 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.890905.890905 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037398338317871094 seconds
INFO 05-06 10:50:15.890353.890353 lmp.py:1496] [layer_moe_fused] to time: 4.792213439941406e-05 seconds
INFO 05-06 10:50:15.891485.891485 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00025916099548339844 seconds
DEBUG 05-06 10:50:15.891304.891304 cuda_h.py:27] end *layer_moe_fused cost 18.154 ms
DEBUG 05-06 10:50:15.897435.897435 cuda_h.py:27] end prefill_layer cost 27.702 ms
DEBUG 05-06 10:50:15.897245.897245 lmp.py:841] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 10:50:15.897584.897584 lmp.py:824] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 10:50:15.899767.899767 cuda_h.py:27] end *sagl cost 1.931 ms
experts_cpu_alloc {'expert_ids': [79, 87, 111, 91, 27, 51, 56, 84, 88, 96, 116, 68, 49, 17, 41, 29, 33, 82, 94, 46, 106, 14, 54, 18], 'token_total': 156, 'token_per_expert': {79: 2, 87: 6, 111: 6, 91: 7, 27: 8, 51: 12, 56: 3, 84: 3, 88: 5, 96: 7, 116: 16, 68: 18, 49: 1, 17: 4, 41: 5, 29: 7, 33: 7, 82: 1, 94: 1, 46: 5, 106: 7, 14: 8, 54: 8, 18: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 31, 35, 39, 43, 47, 59, 63, 67, 71, 75, 83, 95, 99, 103, 107, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 5305, 'token_per_expert': {3: 530, 7: 577, 11: 115, 15: 18, 19: 40, 23: 38, 31: 235, 35: 21, 39: 366, 43: 26, 47: 278, 59: 233, 63: 23, 67: 12, 71: 73, 75: 303, 83: 78, 95: 321, 99: 165, 103: 210, 107: 52, 115: 802, 119: 396, 123: 245, 127: 148}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 48, 52, 60, 64, 72, 76, 80, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3409, 'token_per_expert': {0: 554, 4: 513, 8: 152, 12: 97, 16: 50, 24: 142, 28: 42, 32: 74, 36: 36, 40: 23, 44: 35, 48: 24, 52: 81, 60: 64, 64: 37, 72: 73, 76: 115, 80: 157, 92: 51, 100: 252, 104: 129, 108: 84, 112: 84, 120: 75, 124: 465}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 37, 45, 53, 57, 65, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3572, 'token_per_expert': {1: 519, 5: 536, 9: 16, 13: 82, 21: 11, 25: 46, 37: 11, 45: 54, 53: 99, 57: 80, 65: 344, 73: 15, 77: 11, 81: 66, 85: 10, 89: 130, 93: 15, 97: 392, 101: 18, 105: 69, 109: 27, 113: 161, 117: 197, 121: 570, 125: 93}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 26, 30, 34, 38, 42, 50, 58, 62, 66, 70, 74, 78, 86, 90, 98, 102, 110, 114, 118, 122, 126], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 3942, 'token_per_expert': {2: 654, 6: 519, 10: 109, 22: 12, 26: 387, 30: 73, 34: 82, 38: 150, 42: 127, 50: 295, 58: 17, 62: 93, 66: 366, 70: 32, 74: 89, 78: 14, 86: 428, 90: 38, 98: 32, 102: 43, 110: 58, 114: 125, 118: 25, 122: 146, 126: 28}}
INFO 05-06 10:50:15.902299.902299 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.878ms | allocate_experts_across_cpu_gpu: 0.460ms
INFO 05-06 10:50:15.902053.902053 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.940696716308594e-05 seconds
INFO 05-06 10:50:15.903406.903406 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006475448608398438 seconds
INFO 05-06 10:50:15.904018.904018 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008080005645751953 seconds
INFO 05-06 10:50:15.913650.913650 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009685516357421875 seconds
INFO 05-06 10:50:15.915486.915486 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009489059448242188 seconds
INFO 05-06 10:50:15.917990.917990 mlpmodule.py:2799] [fused_experts] gmm total=1.957ms E=32 S=3461 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.917683.917683 mlpmodule.py:2799] [fused_experts] gmm total=2.034ms E=32 S=3596 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.917820.917820 mlpmodule.py:2799] [fused_experts] gmm total=2.264ms E=32 S=5346 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.917976.917976 mlpmodule.py:2799] [fused_experts] gmm total=2.444ms E=32 S=3981 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.918504.918504 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0036308765411376953 seconds
INFO 05-06 10:50:15.918475.918475 lmp.py:1496] [layer_moe_fused] to time: 4.76837158203125e-05 seconds
INFO 05-06 10:50:15.919742.919742 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00032401084899902344 seconds
DEBUG 05-06 10:50:15.919122.919122 cuda_h.py:27] end *layer_moe_fused cost 18.416 ms
DEBUG 05-06 10:50:15.925258.925258 cuda_h.py:27] end prefill_layer cost 27.842 ms
DEBUG 05-06 10:50:15.925069.925069 lmp.py:841] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 10:50:15.925646.925646 lmp.py:824] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 10:50:15.927706.927706 cuda_h.py:27] end *sagl cost 1.896 ms
experts_cpu_alloc {'expert_ids': [35, 67, 87, 123, 11, 20, 56, 60, 12, 100, 32, 49, 61, 89, 57, 45, 25, 62, 26, 106, 22, 74, 50, 110], 'token_total': 146, 'token_per_expert': {35: 6, 67: 9, 87: 9, 123: 9, 11: 10, 20: 1, 56: 1, 60: 1, 12: 5, 100: 8, 32: 12, 49: 1, 61: 3, 89: 3, 57: 10, 45: 12, 25: 19, 62: 1, 26: 2, 106: 3, 22: 4, 74: 4, 50: 5, 110: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 31, 39, 43, 47, 51, 55, 59, 63, 71, 75, 79, 83, 91, 95, 99, 103, 107, 111, 115, 119, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3991, 'token_per_expert': {3: 518, 7: 670, 19: 30, 23: 183, 31: 58, 39: 181, 43: 43, 47: 81, 51: 165, 55: 84, 59: 63, 63: 79, 71: 187, 75: 238, 79: 17, 83: 380, 91: 502, 95: 95, 99: 102, 103: 101, 107: 32, 111: 23, 115: 38, 119: 56, 127: 65}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 24, 28, 36, 40, 44, 48, 52, 64, 68, 72, 76, 80, 84, 88, 96, 104, 108, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 5178, 'token_per_expert': {0: 595, 4: 535, 8: 31, 16: 158, 24: 79, 28: 75, 36: 91, 40: 45, 44: 13, 48: 37, 52: 206, 64: 220, 68: 477, 72: 155, 76: 740, 80: 19, 84: 129, 88: 174, 96: 22, 104: 122, 108: 329, 112: 647, 116: 115, 120: 84, 124: 80}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 37, 41, 65, 69, 73, 77, 81, 85, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3885, 'token_per_expert': {1: 563, 5: 583, 9: 215, 13: 42, 17: 45, 21: 96, 29: 62, 33: 54, 37: 172, 41: 48, 65: 407, 69: 76, 73: 79, 77: 28, 81: 115, 85: 113, 93: 101, 97: 119, 101: 198, 105: 27, 109: 457, 113: 64, 117: 27, 121: 40, 125: 154}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 30, 34, 38, 42, 46, 54, 58, 66, 70, 78, 82, 86, 90, 94, 98, 102, 114, 118, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3184, 'token_per_expert': {2: 660, 6: 519, 10: 325, 14: 62, 18: 38, 30: 241, 34: 29, 38: 22, 42: 92, 46: 73, 54: 12, 58: 27, 66: 190, 70: 148, 78: 59, 82: 30, 86: 57, 90: 274, 94: 12, 98: 151, 102: 65, 114: 55, 118: 29, 126: 14}}
INFO 05-06 10:50:15.930644.930644 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.551ms | allocate_experts_across_cpu_gpu: 0.454ms
INFO 05-06 10:50:15.930298.930298 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.654594421386719e-05 seconds
INFO 05-06 10:50:15.931122.931122 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006749629974365234 seconds
INFO 05-06 10:50:15.931389.931389 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007994174957275391 seconds
INFO 05-06 10:50:15.942822.942822 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010231256484985352 seconds
INFO 05-06 10:50:15.943109.943109 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009667873382568359 seconds
INFO 05-06 10:50:15.945609.945609 mlpmodule.py:2799] [fused_experts] gmm total=2.065ms E=32 S=4034 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.945330.945330 mlpmodule.py:2799] [fused_experts] gmm total=2.204ms E=32 S=5206 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.945089.945089 mlpmodule.py:2799] [fused_experts] gmm total=2.277ms E=32 S=3933 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.946026.946026 mlpmodule.py:2799] [fused_experts] gmm total=2.705ms E=32 S=3211 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.947162.947162 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0036869049072265625 seconds
INFO 05-06 10:50:15.947894.947894 lmp.py:1496] [layer_moe_fused] to time: 4.744529724121094e-05 seconds
INFO 05-06 10:50:15.947780.947780 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003936290740966797 seconds
DEBUG 05-06 10:50:15.947903.947903 cuda_h.py:27] end *layer_moe_fused cost 18.825 ms
DEBUG 05-06 10:50:15.954317.954317 cuda_h.py:27] end prefill_layer cost 28.927 ms
DEBUG 05-06 10:50:15.954842.954842 lmp.py:841] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 10:50:15.954751.954751 lmp.py:824] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 10:50:15.956789.956789 cuda_h.py:27] end *sagl cost 1.911 ms
experts_cpu_alloc {'expert_ids': [35, 27, 95, 115, 71, 28, 120, 88, 36, 64, 104, 112, 25, 73, 101, 49, 29, 41, 89, 106, 94, 74, 50, 122, 98], 'token_total': 202, 'token_per_expert': {35: 1, 27: 2, 95: 3, 115: 4, 71: 12, 28: 4, 120: 4, 88: 9, 36: 12, 64: 15, 104: 19, 112: 20, 25: 1, 73: 3, 101: 3, 49: 7, 29: 9, 41: 9, 89: 10, 106: 2, 94: 4, 74: 8, 50: 11, 122: 13, 98: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 31, 43, 51, 55, 59, 63, 67, 75, 79, 83, 87, 91, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4011, 'token_per_expert': {3: 627, 7: 521, 11: 18, 15: 35, 19: 101, 23: 237, 31: 154, 43: 36, 51: 53, 55: 109, 59: 17, 63: 152, 67: 514, 75: 182, 79: 38, 83: 228, 87: 549, 91: 34, 99: 32, 103: 21, 107: 120, 111: 35, 119: 55, 123: 44, 127: 99}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 32, 40, 44, 48, 52, 56, 60, 68, 72, 76, 80, 84, 92, 96, 100, 108, 116, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 5024, 'token_per_expert': {0: 671, 4: 702, 8: 174, 12: 212, 16: 478, 20: 144, 24: 46, 32: 658, 40: 41, 44: 106, 48: 108, 52: 500, 56: 37, 60: 23, 68: 144, 72: 89, 76: 119, 80: 44, 84: 47, 92: 50, 96: 117, 100: 172, 108: 142, 116: 85, 124: 115}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 33, 37, 45, 53, 57, 61, 65, 69, 77, 81, 85, 93, 97, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 2936, 'token_per_expert': {1: 762, 5: 676, 9: 16, 13: 28, 17: 49, 21: 37, 33: 31, 37: 32, 45: 29, 53: 11, 57: 58, 61: 72, 65: 86, 69: 25, 77: 58, 81: 56, 85: 80, 93: 48, 97: 53, 105: 400, 109: 27, 113: 46, 117: 95, 121: 58, 125: 103}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 54, 58, 62, 66, 70, 78, 82, 86, 90, 102, 110, 114, 118, 126], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4211, 'token_per_expert': {2: 618, 6: 542, 10: 41, 14: 92, 18: 62, 22: 75, 26: 104, 30: 42, 34: 23, 38: 29, 42: 145, 54: 145, 58: 104, 62: 31, 66: 200, 70: 100, 78: 110, 82: 83, 86: 350, 90: 83, 102: 71, 110: 122, 114: 138, 118: 58, 126: 843}}
INFO 05-06 10:50:15.959873.959873 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.926ms | allocate_experts_across_cpu_gpu: 0.458ms
INFO 05-06 10:50:15.959050.959050 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.654594421386719e-05 seconds
INFO 05-06 10:50:15.960223.960223 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0005965232849121094 seconds
INFO 05-06 10:50:15.961457.961457 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008065700531005859 seconds
INFO 05-06 10:50:15.971141.971141 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010461807250976562 seconds
INFO 05-06 10:50:15.972277.972277 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009944438934326172 seconds
INFO 05-06 10:50:15.975821.975821 mlpmodule.py:2799] [fused_experts] gmm total=2.208ms E=32 S=4033 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.975680.975680 mlpmodule.py:2799] [fused_experts] gmm total=2.246ms E=32 S=2978 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.975407.975407 mlpmodule.py:2799] [fused_experts] gmm total=2.452ms E=32 S=5107 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.975900.975900 mlpmodule.py:2799] [fused_experts] gmm total=2.434ms E=32 S=4266 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:15.976287.976287 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037984848022460938 seconds
INFO 05-06 10:50:15.976643.976643 lmp.py:1496] [layer_moe_fused] to time: 4.76837158203125e-05 seconds
INFO 05-06 10:50:15.977377.977377 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.000247955322265625 seconds
DEBUG 05-06 10:50:15.977488.977488 cuda_h.py:27] end *layer_moe_fused cost 19.362 ms
DEBUG 05-06 10:50:15.984228.984228 cuda_h.py:27] end prefill_layer cost 29.663 ms
DEBUG 05-06 10:50:15.984561.984561 lmp.py:841] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 10:50:15.984424.984424 lmp.py:824] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 10:50:15.987451.987451 cuda_h.py:27] end *sagl cost 2.605 ms
experts_cpu_alloc {'expert_ids': [127, 79, 15, 51, 11, 19, 112, 88, 8, 92, 96, 44, 16, 25, 77, 29, 93, 85, 46, 110, 30, 42, 82, 102], 'token_total': 315, 'token_per_expert': {127: 6, 79: 7, 15: 8, 51: 9, 11: 15, 19: 29, 112: 8, 88: 10, 8: 12, 92: 24, 96: 26, 44: 28, 16: 30, 25: 1, 77: 9, 29: 12, 93: 13, 85: 14, 46: 1, 110: 2, 30: 12, 42: 13, 82: 13, 102: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 27, 31, 35, 39, 43, 47, 55, 59, 63, 67, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 119, 123], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4392, 'token_per_expert': {3: 568, 7: 518, 23: 470, 27: 227, 31: 93, 35: 88, 39: 230, 43: 254, 47: 144, 55: 80, 59: 45, 63: 135, 67: 37, 71: 170, 75: 408, 83: 29, 87: 40, 91: 41, 95: 341, 99: 44, 103: 102, 107: 174, 111: 64, 119: 52, 123: 38}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 28, 32, 36, 40, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 100, 104, 108, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4312, 'token_per_expert': {0: 574, 4: 632, 12: 108, 20: 138, 24: 440, 28: 143, 32: 33, 36: 31, 40: 209, 48: 66, 52: 237, 56: 170, 60: 33, 64: 137, 68: 132, 72: 236, 76: 458, 80: 82, 84: 70, 100: 41, 104: 53, 108: 92, 116: 51, 120: 64, 124: 82}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 33, 37, 45, 49, 53, 57, 61, 65, 69, 73, 81, 89, 97, 101, 109, 113, 117, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 4014, 'token_per_expert': {1: 533, 5: 599, 9: 28, 13: 55, 17: 118, 21: 254, 33: 44, 37: 587, 45: 82, 49: 144, 53: 232, 57: 91, 61: 191, 65: 21, 69: 334, 73: 85, 81: 16, 89: 245, 97: 23, 101: 150, 109: 42, 113: 44, 117: 21, 125: 75}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 34, 38, 54, 58, 62, 66, 70, 74, 78, 86, 90, 94, 98, 106, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3351, 'token_per_expert': {2: 527, 6: 596, 10: 167, 14: 28, 18: 153, 22: 122, 34: 17, 38: 40, 54: 104, 58: 248, 62: 27, 66: 27, 70: 80, 74: 374, 78: 72, 86: 260, 90: 25, 94: 62, 98: 82, 106: 130, 114: 59, 118: 35, 122: 70, 126: 46}}
INFO 05-06 10:50:15.991964.991964 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 2.846ms | allocate_experts_across_cpu_gpu: 0.461ms
INFO 05-06 10:50:15.992818.992818 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.797645568847656e-05 seconds
INFO 05-06 10:50:15.992655.992655 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006122589111328125 seconds
INFO 05-06 10:50:15.993577.993577 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007915496826171875 seconds
INFO 05-06 10:50:16.003904.003904 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009882926940917969 seconds
INFO 05-06 10:50:16.004962.004962 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008318424224853516 seconds
INFO 05-06 10:50:16.007007.007007 mlpmodule.py:2799] [fused_experts] gmm total=2.752ms E=32 S=4466 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.007436.007436 mlpmodule.py:2799] [fused_experts] gmm total=2.790ms E=32 S=4063 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.007693.007693 mlpmodule.py:2799] [fused_experts] gmm total=2.894ms E=32 S=3405 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.007218.007218 mlpmodule.py:2799] [fused_experts] gmm total=3.065ms E=32 S=4450 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.009914.009914 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00441741943359375 seconds
INFO 05-06 10:50:16.009428.009428 lmp.py:1496] [layer_moe_fused] to time: 4.8160552978515625e-05 seconds
INFO 05-06 10:50:16.009458.009458 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003612041473388672 seconds
DEBUG 05-06 10:50:16.009907.009907 cuda_h.py:27] end *layer_moe_fused cost 21.385 ms
DEBUG 05-06 10:50:16.016414.016414 cuda_h.py:27] end prefill_layer cost 31.937 ms
DEBUG 05-06 10:50:16.016747.016747 lmp.py:841] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 10:50:16.016610.016610 lmp.py:824] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 10:50:16.018440.018440 cuda_h.py:27] end *sagl cost 1.883 ms
experts_cpu_alloc {'expert_ids': [115, 55, 63, 11, 27, 28, 20, 16, 52, 44, 24, 96, 113, 117, 41, 21, 25, 9, 106, 22, 86, 126, 102, 42, 18], 'token_total': 275, 'token_per_expert': {115: 2, 55: 9, 63: 12, 11: 16, 27: 19, 28: 1, 20: 2, 16: 9, 52: 11, 44: 14, 24: 17, 96: 20, 113: 5, 117: 5, 41: 12, 21: 13, 25: 22, 9: 25, 106: 3, 22: 4, 86: 4, 126: 5, 102: 9, 42: 16, 18: 20}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 31, 35, 39, 43, 47, 51, 59, 67, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 26, 'ideal_gpu_count': 26, 'keep_on_gpu': 26, 'hit_count_on_device': 31, 'token_total': 4021, 'token_per_expert': {3: 735, 7: 558, 15: 41, 19: 27, 23: 37, 31: 170, 35: 35, 39: 22, 43: 167, 47: 80, 51: 46, 59: 20, 67: 38, 71: 82, 75: 112, 83: 282, 87: 148, 91: 55, 95: 103, 99: 424, 103: 64, 107: 40, 111: 259, 119: 251, 123: 120, 127: 105}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 32, 36, 40, 48, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4313, 'token_per_expert': {0: 520, 4: 689, 8: 208, 12: 82, 32: 256, 36: 245, 40: 219, 48: 47, 56: 64, 60: 159, 64: 248, 68: 63, 72: 154, 76: 263, 80: 94, 84: 156, 88: 115, 92: 105, 100: 115, 104: 181, 108: 51, 112: 30, 116: 58, 120: 156, 124: 35}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 17, 29, 33, 37, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 109, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4248, 'token_per_expert': {1: 625, 5: 597, 13: 59, 17: 116, 29: 59, 33: 198, 37: 81, 45: 33, 49: 115, 53: 222, 57: 79, 61: 171, 65: 158, 69: 111, 73: 53, 77: 211, 81: 152, 85: 287, 89: 25, 93: 138, 97: 55, 101: 208, 109: 39, 121: 379, 125: 77}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 26, 30, 34, 38, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 90, 94, 98, 110, 114, 118, 122], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 3527, 'token_per_expert': {2: 822, 6: 517, 10: 89, 14: 191, 26: 45, 30: 74, 34: 110, 38: 130, 46: 50, 50: 269, 54: 260, 58: 211, 62: 22, 66: 30, 70: 32, 74: 39, 78: 126, 82: 29, 90: 36, 94: 21, 98: 34, 110: 156, 114: 25, 118: 174, 122: 35}}
INFO 05-06 10:50:16.021115.021115 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.928ms | allocate_experts_across_cpu_gpu: 0.461ms
INFO 05-06 10:50:16.021875.021875 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.96453857421875e-05 seconds
INFO 05-06 10:50:16.022525.022525 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008721351623535156 seconds
INFO 05-06 10:50:16.023091.023091 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008087158203125 seconds
INFO 05-06 10:50:16.033381.033381 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010382652282714844 seconds
INFO 05-06 10:50:16.035305.035305 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009748935699462891 seconds
INFO 05-06 10:50:16.037999.037999 mlpmodule.py:2799] [fused_experts] gmm total=2.302ms E=32 S=4387 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.037824.037824 mlpmodule.py:2799] [fused_experts] gmm total=2.467ms E=32 S=4079 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.037032.037032 mlpmodule.py:2799] [fused_experts] gmm total=2.399ms E=32 S=3588 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.038038.038038 mlpmodule.py:2799] [fused_experts] gmm total=2.805ms E=32 S=4330 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.038842.038842 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003906726837158203 seconds
INFO 05-06 10:50:16.039574.039574 lmp.py:1496] [layer_moe_fused] to time: 4.744529724121094e-05 seconds
INFO 05-06 10:50:16.039766.039766 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00023245811462402344 seconds
DEBUG 05-06 10:50:16.039917.039917 cuda_h.py:27] end *layer_moe_fused cost 19.609 ms
DEBUG 05-06 10:50:16.046216.046216 cuda_h.py:27] end prefill_layer cost 29.968 ms
DEBUG 05-06 10:50:16.046741.046741 lmp.py:841] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 10:50:16.046365.046365 lmp.py:824] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 10:50:16.048894.048894 cuda_h.py:27] end *sagl cost 1.936 ms
experts_cpu_alloc {'expert_ids': [107, 71, 115, 67, 95, 87, 116, 124, 100, 8, 120, 81, 85, 49, 113, 93, 105, 101, 82, 34, 62, 46, 66, 74], 'token_total': 181, 'token_per_expert': {107: 3, 71: 4, 115: 8, 67: 10, 95: 10, 87: 13, 116: 3, 124: 4, 100: 10, 8: 11, 120: 18, 81: 1, 85: 2, 49: 6, 113: 6, 93: 8, 105: 9, 101: 12, 82: 4, 34: 5, 62: 7, 46: 9, 66: 9, 74: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 75, 79, 83, 99, 103, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3871, 'token_per_expert': {3: 752, 7: 752, 11: 60, 15: 94, 19: 85, 23: 180, 27: 144, 31: 86, 35: 117, 39: 119, 43: 16, 47: 60, 51: 358, 55: 27, 59: 37, 63: 142, 75: 163, 79: 169, 83: 61, 99: 57, 103: 36, 111: 41, 119: 93, 123: 206, 127: 16}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 20, 24, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 104, 108, 112], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4913, 'token_per_expert': {0: 591, 4: 539, 12: 50, 16: 143, 20: 41, 24: 295, 36: 58, 40: 168, 44: 429, 48: 102, 52: 726, 56: 39, 60: 64, 64: 374, 68: 41, 72: 53, 76: 124, 80: 184, 84: 57, 88: 114, 92: 436, 96: 120, 104: 86, 108: 40, 112: 39}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 77, 89, 97, 109, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4157, 'token_per_expert': {1: 625, 5: 580, 9: 153, 13: 69, 17: 66, 21: 175, 25: 62, 29: 33, 33: 83, 37: 369, 41: 102, 45: 21, 53: 98, 57: 13, 61: 217, 65: 15, 69: 140, 73: 46, 77: 13, 89: 621, 97: 50, 109: 156, 117: 300, 121: 26, 125: 124}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 38, 42, 50, 54, 58, 70, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3262, 'token_per_expert': {2: 605, 6: 522, 10: 89, 18: 28, 22: 53, 26: 127, 30: 18, 38: 489, 42: 28, 50: 166, 54: 16, 58: 39, 70: 15, 86: 34, 90: 71, 94: 19, 98: 53, 102: 138, 106: 117, 110: 33, 114: 23, 118: 60, 122: 500, 126: 19}}
INFO 05-06 10:50:16.050514.050514 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.545ms | allocate_experts_across_cpu_gpu: 0.462ms
INFO 05-06 10:50:16.051222.051222 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.392333984375e-05 seconds
INFO 05-06 10:50:16.052974.052974 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007138252258300781 seconds
INFO 05-06 10:50:16.052301.052301 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008072853088378906 seconds
INFO 05-06 10:50:16.063494.063494 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010415077209472656 seconds
INFO 05-06 10:50:16.064259.064259 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009996891021728516 seconds
INFO 05-06 10:50:16.066985.066985 mlpmodule.py:2799] [fused_experts] gmm total=2.095ms E=32 S=3919 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.067899.067899 mlpmodule.py:2799] [fused_experts] gmm total=2.262ms E=32 S=4959 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.067109.067109 mlpmodule.py:2799] [fused_experts] gmm total=2.331ms E=32 S=4201 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.067516.067516 mlpmodule.py:2799] [fused_experts] gmm total=2.580ms E=32 S=3305 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.068484.068484 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0036764144897460938 seconds
INFO 05-06 10:50:16.068978.068978 lmp.py:1496] [layer_moe_fused] to time: 4.744529724121094e-05 seconds
INFO 05-06 10:50:16.068816.068816 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003590583801269531 seconds
DEBUG 05-06 10:50:16.069286.069286 cuda_h.py:27] end *layer_moe_fused cost 19.099 ms
DEBUG 05-06 10:50:16.075231.075231 cuda_h.py:27] end prefill_layer cost 29.212 ms
DEBUG 05-06 10:50:16.076280.076280 lmp.py:841] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 10:50:16.076857.076857 lmp.py:824] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 10:50:16.078087.078087 cuda_h.py:27] end *sagl cost 1.922 ms
experts_cpu_alloc {'expert_ids': [87, 99, 67, 23, 51, 124, 16, 24, 104, 36, 29, 97, 17, 25, 69, 89, 61, 14, 22, 70, 106, 126, 34, 10], 'token_total': 265, 'token_per_expert': {87: 2, 99: 3, 67: 4, 23: 5, 51: 5, 124: 6, 16: 7, 24: 21, 104: 24, 36: 30, 29: 5, 97: 5, 17: 10, 25: 13, 69: 13, 89: 14, 61: 15, 14: 1, 22: 3, 70: 6, 106: 11, 126: 16, 34: 20, 10: 26}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 35, 43, 47, 55, 59, 63, 71, 75, 79, 83, 91, 95, 103, 107, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3287, 'token_per_expert': {3: 707, 7: 518, 11: 13, 15: 83, 19: 32, 27: 122, 31: 15, 35: 18, 43: 121, 47: 20, 55: 48, 59: 230, 63: 361, 71: 85, 75: 5, 79: 90, 83: 61, 91: 9, 95: 31, 103: 36, 107: 458, 111: 35, 119: 6, 123: 176, 127: 7}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 28, 32, 40, 44, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 108, 112, 116, 120], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4293, 'token_per_expert': {0: 543, 4: 673, 8: 196, 12: 58, 20: 77, 28: 157, 32: 134, 40: 196, 44: 133, 52: 67, 56: 178, 60: 63, 64: 82, 68: 701, 72: 86, 76: 65, 80: 43, 84: 46, 88: 155, 92: 124, 100: 93, 108: 158, 112: 124, 116: 110, 120: 31}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 33, 37, 41, 45, 49, 53, 57, 65, 73, 77, 81, 85, 93, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 5118, 'token_per_expert': {1: 556, 5: 674, 9: 129, 13: 162, 21: 243, 33: 131, 37: 242, 41: 92, 45: 418, 49: 492, 53: 115, 57: 267, 65: 291, 73: 252, 77: 229, 81: 145, 85: 118, 93: 42, 101: 29, 105: 40, 109: 129, 113: 97, 117: 21, 121: 70, 125: 134}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 26, 30, 38, 42, 46, 50, 54, 58, 62, 66, 74, 82, 86, 90, 94, 98, 102, 110, 114, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3421, 'token_per_expert': {2: 584, 6: 520, 18: 68, 26: 26, 30: 326, 38: 36, 42: 161, 46: 123, 50: 83, 54: 51, 58: 35, 62: 35, 66: 124, 74: 47, 82: 75, 86: 34, 90: 31, 94: 533, 98: 31, 102: 247, 110: 42, 114: 40, 118: 59, 122: 110}}
INFO 05-06 10:50:16.080867.080867 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.981ms | allocate_experts_across_cpu_gpu: 0.455ms
INFO 05-06 10:50:16.080151.080151 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.916854858398438e-05 seconds
INFO 05-06 10:50:16.081441.081441 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006041526794433594 seconds
INFO 05-06 10:50:16.082555.082555 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007920265197753906 seconds
INFO 05-06 10:50:16.093551.093551 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011250495910644531 seconds
INFO 05-06 10:50:16.094395.094395 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009853839874267578 seconds
INFO 05-06 10:50:16.097987.097987 mlpmodule.py:2799] [fused_experts] gmm total=2.070ms E=32 S=3306 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.097610.097610 mlpmodule.py:2799] [fused_experts] gmm total=2.235ms E=32 S=4381 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.097529.097529 mlpmodule.py:2799] [fused_experts] gmm total=2.511ms E=32 S=5193 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.097501.097501 mlpmodule.py:2799] [fused_experts] gmm total=2.617ms E=32 S=3504 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.098641.098641 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003728151321411133 seconds
INFO 05-06 10:50:16.098374.098374 lmp.py:1496] [layer_moe_fused] to time: 4.76837158203125e-05 seconds
INFO 05-06 10:50:16.099378.099378 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00037598609924316406 seconds
DEBUG 05-06 10:50:16.099158.099158 cuda_h.py:27] end *layer_moe_fused cost 20.229 ms
DEBUG 05-06 10:50:16.106969.106969 cuda_h.py:27] end prefill_layer cost 30.411 ms
DEBUG 05-06 10:50:16.106779.106779 lmp.py:841] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 10:50:16.106880.106880 lmp.py:824] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 10:50:16.108859.108859 cuda_h.py:27] end *sagl cost 1.914 ms
experts_cpu_alloc {'expert_ids': [27, 99, 107, 15, 19, 39, 28, 60, 104, 96, 52, 64, 89, 77, 101, 25, 17, 66, 98, 94, 22, 54, 74, 114], 'token_total': 204, 'token_per_expert': {27: 2, 99: 2, 107: 2, 15: 3, 19: 3, 39: 3, 28: 3, 60: 3, 104: 8, 96: 11, 52: 15, 64: 15, 89: 1, 77: 5, 101: 8, 25: 11, 17: 14, 66: 1, 98: 7, 94: 10, 22: 11, 54: 15, 74: 24, 114: 27}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 31, 35, 43, 47, 51, 55, 59, 67, 71, 75, 79, 83, 87, 95, 103, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3090, 'token_per_expert': {3: 549, 7: 670, 11: 347, 23: 10, 31: 104, 35: 93, 43: 42, 47: 11, 51: 139, 55: 44, 59: 41, 67: 125, 71: 31, 75: 90, 79: 84, 83: 182, 87: 55, 95: 74, 103: 142, 111: 75, 115: 20, 119: 41, 123: 55, 127: 66}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 32, 36, 40, 44, 48, 56, 68, 72, 76, 80, 84, 88, 92, 100, 112, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3841, 'token_per_expert': {0: 565, 4: 626, 8: 126, 12: 100, 16: 22, 20: 35, 24: 50, 32: 54, 36: 110, 40: 33, 44: 48, 48: 300, 56: 25, 68: 104, 72: 121, 76: 187, 80: 58, 84: 117, 88: 16, 92: 224, 100: 352, 112: 191, 120: 215, 124: 162}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 81, 93, 97, 105, 109, 113, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 4560, 'token_per_expert': {1: 843, 5: 980, 9: 15, 13: 101, 21: 38, 29: 229, 33: 98, 37: 164, 41: 202, 45: 40, 53: 205, 57: 132, 61: 148, 65: 392, 69: 18, 73: 225, 81: 60, 93: 19, 97: 121, 105: 258, 109: 143, 113: 23, 121: 30, 125: 76}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 26, 30, 34, 38, 42, 46, 50, 58, 62, 70, 78, 82, 86, 90, 102, 106, 110, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4689, 'token_per_expert': {2: 623, 6: 969, 10: 114, 18: 241, 26: 255, 30: 117, 34: 95, 38: 66, 42: 56, 46: 249, 50: 40, 58: 74, 62: 178, 70: 78, 78: 581, 82: 70, 86: 126, 90: 134, 102: 103, 106: 40, 110: 134, 118: 55, 122: 257, 126: 34}}
INFO 05-06 10:50:16.110783.110783 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.542ms | allocate_experts_across_cpu_gpu: 0.449ms
INFO 05-06 10:50:16.111968.111968 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.559226989746094e-05 seconds
INFO 05-06 10:50:16.111879.111879 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006554126739501953 seconds
INFO 05-06 10:50:16.112994.112994 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007922649383544922 seconds
INFO 05-06 10:50:16.123994.123994 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010799169540405273 seconds
INFO 05-06 10:50:16.124678.124678 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009398460388183594 seconds
INFO 05-06 10:50:16.126986.126986 mlpmodule.py:2799] [fused_experts] gmm total=1.689ms E=32 S=3105 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.127491.127491 mlpmodule.py:2799] [fused_experts] gmm total=1.981ms E=32 S=4784 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.127707.127707 mlpmodule.py:2799] [fused_experts] gmm total=2.197ms E=32 S=3896 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.127955.127955 mlpmodule.py:2799] [fused_experts] gmm total=2.622ms E=32 S=4599 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.128514.128514 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0035305023193359375 seconds
INFO 05-06 10:50:16.128246.128246 lmp.py:1496] [layer_moe_fused] to time: 4.744529724121094e-05 seconds
INFO 05-06 10:50:16.128722.128722 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0004067420959472656 seconds
DEBUG 05-06 10:50:16.129412.129412 cuda_h.py:27] end *layer_moe_fused cost 19.364 ms
DEBUG 05-06 10:50:16.136639.136639 cuda_h.py:27] end prefill_layer cost 29.590 ms
DEBUG 05-06 10:50:16.136403.136403 lmp.py:841] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 10:50:16.136742.136742 lmp.py:824] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 10:50:16.138104.138104 cuda_h.py:27] end *sagl cost 1.902 ms
experts_cpu_alloc {'expert_ids': [91, 23, 71, 87, 67, 95, 39, 80, 12, 56, 36, 104, 96, 29, 21, 49, 13, 17, 61, 18, 22, 114, 78, 106, 122], 'token_total': 165, 'token_per_expert': {91: 6, 23: 7, 71: 7, 87: 9, 67: 10, 95: 10, 39: 20, 80: 1, 12: 2, 56: 2, 36: 4, 104: 4, 96: 18, 29: 1, 21: 3, 49: 4, 13: 5, 17: 5, 61: 8, 18: 3, 22: 3, 114: 4, 78: 5, 106: 7, 122: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 35, 43, 47, 51, 55, 59, 63, 75, 79, 83, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4291, 'token_per_expert': {3: 515, 7: 594, 11: 89, 15: 116, 19: 84, 27: 23, 31: 72, 35: 426, 43: 126, 47: 56, 51: 25, 55: 202, 59: 342, 63: 29, 75: 171, 79: 28, 83: 34, 99: 89, 103: 332, 107: 66, 111: 113, 115: 134, 119: 280, 123: 149, 127: 196}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 40, 44, 48, 60, 64, 68, 72, 76, 84, 88, 92, 100, 108, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 5295, 'token_per_expert': {0: 545, 4: 512, 8: 187, 16: 76, 20: 31, 24: 390, 28: 112, 32: 51, 40: 79, 44: 71, 48: 65, 60: 22, 64: 492, 68: 250, 72: 400, 76: 117, 84: 27, 88: 105, 92: 218, 100: 885, 108: 153, 112: 53, 116: 190, 120: 177, 124: 87}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 25, 33, 37, 41, 45, 53, 57, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 2814, 'token_per_expert': {1: 590, 5: 516, 9: 24, 25: 26, 33: 76, 37: 11, 41: 55, 45: 68, 53: 201, 57: 35, 65: 21, 69: 69, 73: 227, 77: 8, 81: 11, 85: 60, 89: 131, 93: 276, 97: 10, 101: 57, 105: 9, 109: 48, 113: 48, 117: 195, 125: 42}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 26, 30, 34, 38, 42, 46, 54, 58, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 110, 118, 126], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3819, 'token_per_expert': {2: 521, 6: 523, 10: 24, 14: 24, 26: 75, 30: 114, 34: 32, 38: 174, 42: 86, 46: 131, 54: 19, 58: 85, 62: 40, 66: 117, 70: 129, 74: 407, 82: 137, 86: 170, 90: 140, 94: 162, 98: 34, 102: 90, 110: 36, 118: 130, 126: 419}}
INFO 05-06 10:50:16.140474.140474 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.955ms | allocate_experts_across_cpu_gpu: 0.465ms
INFO 05-06 10:50:16.141373.141373 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.678436279296875e-05 seconds
INFO 05-06 10:50:16.141541.141541 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00064849853515625 seconds
INFO 05-06 10:50:16.142153.142153 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008087158203125 seconds
INFO 05-06 10:50:16.153766.153766 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010691404342651367 seconds
INFO 05-06 10:50:16.154614.154614 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009250640869140625 seconds
INFO 05-06 10:50:16.156190.156190 mlpmodule.py:2799] [fused_experts] gmm total=2.068ms E=32 S=2840 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.157452.157452 mlpmodule.py:2799] [fused_experts] gmm total=2.288ms E=32 S=4360 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.157948.157948 mlpmodule.py:2799] [fused_experts] gmm total=2.604ms E=32 S=5326 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.157748.157748 mlpmodule.py:2799] [fused_experts] gmm total=2.675ms E=32 S=3858 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.158718.158718 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003835916519165039 seconds
INFO 05-06 10:50:16.158073.158073 lmp.py:1496] [layer_moe_fused] to time: 4.935264587402344e-05 seconds
INFO 05-06 10:50:16.158989.158989 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003120899200439453 seconds
DEBUG 05-06 10:50:16.159956.159956 cuda_h.py:27] end *layer_moe_fused cost 19.792 ms
DEBUG 05-06 10:50:16.165618.165618 cuda_h.py:27] end prefill_layer cost 29.536 ms
DEBUG 05-06 10:50:16.165197.165197 lmp.py:841] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 10:50:16.165251.165251 lmp.py:824] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 10:50:16.169513.169513 cuda_h.py:27] end *sagl cost 2.932 ms
experts_cpu_alloc {'expert_ids': [119, 111, 15, 55, 127, 27, 88, 92, 20, 28, 64, 68, 69, 13, 77, 93, 49, 114, 70, 126, 50, 74, 82, 110], 'token_total': 218, 'token_per_expert': {119: 3, 111: 5, 15: 10, 55: 10, 127: 11, 27: 14, 88: 9, 92: 10, 20: 14, 28: 14, 64: 15, 68: 23, 69: 3, 13: 5, 77: 7, 93: 7, 49: 13, 114: 1, 70: 2, 126: 2, 50: 6, 74: 10, 82: 10, 110: 14}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 39, 43, 47, 51, 59, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 115, 123], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3976, 'token_per_expert': {3: 720, 7: 521, 11: 20, 19: 49, 23: 31, 31: 113, 35: 165, 39: 407, 43: 267, 47: 248, 51: 38, 59: 80, 67: 417, 71: 85, 75: 35, 79: 222, 83: 92, 87: 73, 91: 61, 95: 29, 99: 27, 103: 34, 107: 22, 115: 83, 123: 137}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 32, 36, 40, 44, 48, 52, 56, 60, 72, 76, 80, 84, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3511, 'token_per_expert': {0: 526, 4: 516, 8: 114, 12: 48, 16: 144, 24: 89, 32: 47, 36: 33, 40: 47, 44: 216, 48: 28, 52: 39, 56: 513, 60: 25, 72: 118, 76: 86, 80: 113, 84: 139, 100: 100, 104: 116, 108: 185, 112: 66, 116: 98, 120: 52, 124: 53}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 25, 29, 33, 37, 41, 53, 57, 61, 65, 73, 81, 85, 89, 97, 101, 105, 109, 117, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 5042, 'token_per_expert': {1: 743, 5: 744, 9: 54, 17: 108, 21: 667, 25: 261, 29: 281, 33: 83, 37: 295, 41: 16, 53: 26, 57: 38, 61: 290, 65: 339, 73: 92, 81: 19, 85: 171, 89: 42, 97: 218, 101: 16, 105: 99, 109: 153, 117: 94, 125: 193}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 54, 58, 62, 66, 78, 86, 90, 98, 102, 106, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3637, 'token_per_expert': {2: 645, 6: 571, 10: 48, 14: 15, 18: 164, 22: 62, 26: 125, 30: 100, 34: 76, 38: 53, 42: 53, 46: 277, 54: 16, 58: 20, 62: 28, 66: 28, 78: 157, 86: 450, 90: 133, 98: 240, 102: 22, 106: 96, 118: 218, 122: 40}}
INFO 05-06 10:50:16.173407.173407 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 2.590ms | allocate_experts_across_cpu_gpu: 0.452ms
INFO 05-06 10:50:16.173208.173208 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.869171142578125e-05 seconds
INFO 05-06 10:50:16.174043.174043 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006170272827148438 seconds
INFO 05-06 10:50:16.175244.175244 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007855892181396484 seconds
INFO 05-06 10:50:16.185710.185710 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010232686996459961 seconds
INFO 05-06 10:50:16.186241.186241 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009362697601318359 seconds
INFO 05-06 10:50:16.188354.188354 mlpmodule.py:2799] [fused_experts] gmm total=1.778ms E=32 S=3596 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.188879.188879 mlpmodule.py:2799] [fused_experts] gmm total=1.922ms E=32 S=3682 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.189775.189775 mlpmodule.py:2799] [fused_experts] gmm total=2.310ms E=32 S=4029 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.189604.189604 mlpmodule.py:2799] [fused_experts] gmm total=2.645ms E=32 S=5077 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.190448.190448 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0035600662231445312 seconds
INFO 05-06 10:50:16.190426.190426 lmp.py:1496] [layer_moe_fused] to time: 4.8160552978515625e-05 seconds
INFO 05-06 10:50:16.190442.190442 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00035071372985839844 seconds
DEBUG 05-06 10:50:16.190268.190268 cuda_h.py:27] end *layer_moe_fused cost 20.639 ms
DEBUG 05-06 10:50:16.197844.197844 cuda_h.py:27] end prefill_layer cost 31.539 ms
DEBUG 05-06 10:50:16.197416.197416 lmp.py:841] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 10:50:16.197994.197994 lmp.py:824] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 10:50:16.199382.199382 cuda_h.py:27] end *sagl cost 1.947 ms
experts_cpu_alloc {'expert_ids': [51, 123, 95, 59, 15, 88, 24, 84, 80, 116, 112, 85, 125, 69, 117, 101, 113, 54, 22, 58, 78, 126, 10, 18], 'token_total': 177, 'token_per_expert': {51: 4, 123: 5, 95: 8, 59: 9, 15: 12, 88: 1, 24: 8, 84: 8, 80: 10, 116: 10, 112: 14, 85: 1, 125: 1, 69: 2, 117: 2, 101: 3, 113: 3, 54: 2, 22: 4, 58: 10, 78: 10, 126: 11, 10: 12, 18: 27}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 31, 35, 43, 47, 55, 63, 67, 71, 75, 79, 83, 87, 91, 99, 107, 111, 115, 119, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3862, 'token_per_expert': {3: 517, 7: 604, 11: 312, 19: 192, 23: 166, 27: 315, 31: 42, 35: 120, 43: 68, 47: 29, 55: 13, 63: 347, 67: 160, 71: 254, 75: 56, 79: 41, 83: 130, 87: 23, 91: 242, 99: 15, 107: 32, 111: 35, 115: 22, 119: 35, 127: 92}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 76, 92, 96, 100, 104, 108, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3786, 'token_per_expert': {0: 512, 4: 665, 8: 42, 12: 251, 16: 176, 20: 55, 28: 43, 32: 58, 36: 87, 40: 61, 44: 230, 48: 126, 52: 278, 56: 215, 60: 116, 64: 446, 68: 34, 76: 15, 92: 36, 96: 28, 100: 76, 104: 31, 108: 144, 120: 41, 124: 20}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 45, 49, 53, 57, 61, 65, 73, 77, 81, 89, 93, 97, 105, 109, 121], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4163, 'token_per_expert': {1: 624, 5: 599, 9: 42, 13: 74, 17: 95, 21: 4, 25: 5, 29: 130, 33: 287, 37: 97, 45: 167, 49: 48, 53: 39, 57: 35, 61: 13, 65: 24, 73: 209, 77: 210, 81: 54, 89: 11, 93: 5, 97: 454, 105: 41, 109: 119, 121: 777}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 30, 34, 38, 42, 46, 50, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4396, 'token_per_expert': {2: 521, 6: 823, 26: 35, 30: 109, 34: 234, 38: 36, 42: 49, 46: 28, 50: 113, 62: 37, 66: 31, 70: 398, 74: 98, 82: 94, 86: 126, 90: 707, 94: 191, 98: 145, 102: 27, 106: 52, 110: 158, 114: 191, 118: 86, 122: 107}}
INFO 05-06 10:50:16.202782.202782 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.877ms | allocate_experts_across_cpu_gpu: 0.463ms
INFO 05-06 10:50:16.202483.202483 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.535385131835938e-05 seconds
INFO 05-06 10:50:16.203609.203609 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006189346313476562 seconds
INFO 05-06 10:50:16.204473.204473 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0010292530059814453 seconds
INFO 05-06 10:50:16.214112.214112 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010498046875 seconds
INFO 05-06 10:50:16.215602.215602 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001079559326171875 seconds
INFO 05-06 10:50:16.217021.217021 mlpmodule.py:2799] [fused_experts] gmm total=1.791ms E=32 S=3837 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.218177.218177 mlpmodule.py:2799] [fused_experts] gmm total=2.067ms E=32 S=4175 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.218323.218323 mlpmodule.py:2799] [fused_experts] gmm total=2.368ms E=32 S=3900 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.218738.218738 mlpmodule.py:2799] [fused_experts] gmm total=2.361ms E=32 S=4472 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.219804.219804 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0034148693084716797 seconds
INFO 05-06 10:50:16.219583.219583 lmp.py:1496] [layer_moe_fused] to time: 4.839897155761719e-05 seconds
INFO 05-06 10:50:16.219219.219219 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00024390220642089844 seconds
DEBUG 05-06 10:50:16.220966.220966 cuda_h.py:27] end *layer_moe_fused cost 19.317 ms
DEBUG 05-06 10:50:16.226494.226494 cuda_h.py:27] end prefill_layer cost 29.364 ms
DEBUG 05-06 10:50:16.227734.227734 lmp.py:841] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 10:50:16.227550.227550 lmp.py:824] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 10:50:16.229776.229776 cuda_h.py:27] end *sagl cost 1.926 ms
experts_cpu_alloc {'expert_ids': [75, 15, 115, 55, 23, 40, 96, 84, 24, 32, 37, 113, 57, 65, 81, 61, 54, 94, 98, 62, 30, 86, 102, 22], 'token_total': 149, 'token_per_expert': {75: 5, 15: 6, 115: 8, 55: 11, 23: 15, 40: 1, 96: 3, 84: 5, 24: 18, 32: 20, 37: 1, 113: 1, 57: 5, 65: 6, 81: 6, 61: 10, 54: 1, 94: 1, 98: 1, 62: 2, 30: 3, 86: 3, 102: 6, 22: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 27, 31, 35, 39, 43, 47, 51, 63, 67, 71, 79, 83, 87, 91, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3923, 'token_per_expert': {3: 702, 7: 616, 11: 91, 19: 74, 27: 18, 31: 26, 35: 347, 39: 84, 43: 41, 47: 40, 51: 40, 63: 107, 67: 124, 71: 120, 79: 79, 83: 171, 87: 71, 91: 122, 99: 35, 103: 20, 107: 497, 111: 114, 119: 22, 123: 340, 127: 22}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 36, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4579, 'token_per_expert': {0: 571, 4: 524, 8: 38, 12: 22, 16: 922, 36: 101, 44: 112, 48: 37, 52: 250, 56: 84, 60: 257, 64: 168, 68: 538, 72: 49, 76: 21, 80: 223, 88: 63, 92: 22, 100: 101, 104: 200, 108: 22, 112: 38, 116: 101, 120: 87, 124: 28}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 41, 45, 49, 53, 69, 73, 77, 85, 89, 93, 97, 101, 109, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3113, 'token_per_expert': {1: 525, 5: 563, 9: 51, 13: 25, 17: 12, 21: 25, 25: 61, 29: 45, 33: 12, 41: 104, 45: 313, 49: 90, 53: 12, 69: 230, 73: 34, 77: 22, 85: 307, 89: 130, 93: 95, 97: 85, 101: 10, 109: 58, 117: 265, 121: 11, 125: 28}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 26, 34, 38, 42, 46, 50, 58, 66, 70, 74, 78, 82, 90, 106, 110, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 32, 'token_total': 4620, 'token_per_expert': {2: 898, 6: 550, 10: 133, 14: 112, 18: 329, 26: 53, 34: 103, 38: 40, 42: 20, 46: 17, 50: 100, 58: 807, 66: 21, 70: 244, 74: 30, 78: 36, 82: 104, 90: 163, 106: 75, 110: 524, 114: 158, 118: 47, 122: 17, 126: 39}}
INFO 05-06 10:50:16.231854.231854 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.571ms | allocate_experts_across_cpu_gpu: 0.459ms
INFO 05-06 10:50:16.231661.231661 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.298324584960938e-05 seconds
INFO 05-06 10:50:16.232290.232290 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006091594696044922 seconds
INFO 05-06 10:50:16.233429.233429 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0009145736694335938 seconds
INFO 05-06 10:50:16.243110.243110 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010350704193115234 seconds
INFO 05-06 10:50:16.244728.244728 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009639263153076172 seconds
INFO 05-06 10:50:16.247650.247650 mlpmodule.py:2799] [fused_experts] gmm total=2.032ms E=32 S=3968 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.247806.247806 mlpmodule.py:2799] [fused_experts] gmm total=2.052ms E=32 S=3142 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.247950.247950 mlpmodule.py:2799] [fused_experts] gmm total=2.241ms E=32 S=4626 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.247859.247859 mlpmodule.py:2799] [fused_experts] gmm total=2.602ms E=32 S=4648 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.248659.248659 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0036008358001708984 seconds
INFO 05-06 10:50:16.248391.248391 lmp.py:1496] [layer_moe_fused] to time: 4.792213439941406e-05 seconds
INFO 05-06 10:50:16.248167.248167 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002772808074951172 seconds
DEBUG 05-06 10:50:16.249358.249358 cuda_h.py:27] end *layer_moe_fused cost 18.856 ms
DEBUG 05-06 10:50:16.256320.256320 cuda_h.py:27] end prefill_layer cost 29.106 ms
DEBUG 05-06 10:50:16.256892.256892 lmp.py:841] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 10:50:16.256469.256469 lmp.py:824] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 10:50:16.258061.258061 cuda_h.py:27] end *sagl cost 1.898 ms
experts_cpu_alloc {'expert_ids': [119, 55, 127, 11, 107, 47, 12, 16, 48, 32, 100, 28, 21, 101, 93, 121, 33, 53, 117, 98, 82, 22, 58, 46], 'token_total': 136, 'token_per_expert': {119: 1, 55: 4, 127: 5, 11: 9, 107: 10, 47: 14, 12: 2, 16: 4, 48: 6, 32: 7, 100: 7, 28: 13, 21: 1, 101: 1, 93: 4, 121: 4, 33: 5, 53: 6, 117: 7, 98: 2, 82: 4, 22: 5, 58: 5, 46: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 43, 51, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 111, 115, 123], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4138, 'token_per_expert': {3: 592, 7: 512, 15: 150, 19: 127, 23: 16, 27: 265, 31: 28, 35: 40, 43: 266, 51: 116, 59: 100, 63: 19, 67: 40, 71: 27, 75: 57, 79: 83, 83: 30, 87: 467, 91: 28, 95: 384, 99: 84, 103: 86, 111: 463, 115: 31, 123: 127}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 24, 36, 40, 44, 52, 56, 60, 68, 72, 76, 80, 84, 88, 92, 96, 104, 108, 112, 116, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3655, 'token_per_expert': {0: 536, 4: 527, 8: 32, 20: 544, 24: 359, 36: 64, 40: 17, 44: 22, 52: 139, 56: 129, 60: 126, 68: 56, 72: 47, 76: 126, 80: 22, 84: 349, 88: 72, 92: 18, 96: 50, 104: 180, 108: 59, 112: 59, 116: 28, 124: 94}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 25, 29, 37, 41, 45, 49, 57, 61, 65, 73, 77, 81, 85, 89, 97, 105, 109, 113, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 5262, 'token_per_expert': {1: 549, 5: 538, 9: 9, 13: 40, 17: 439, 25: 61, 29: 77, 37: 57, 41: 51, 45: 72, 49: 162, 57: 75, 61: 63, 65: 348, 73: 208, 77: 53, 81: 67, 85: 918, 89: 596, 97: 92, 105: 124, 109: 36, 113: 605, 125: 22}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 26, 30, 34, 38, 42, 50, 54, 66, 70, 74, 78, 86, 90, 102, 110, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 3193, 'token_per_expert': {2: 537, 6: 513, 10: 68, 14: 118, 18: 25, 26: 82, 30: 79, 34: 12, 38: 62, 42: 39, 50: 118, 54: 10, 66: 93, 70: 197, 74: 22, 78: 147, 86: 147, 90: 102, 102: 118, 110: 28, 114: 497, 118: 42, 122: 28, 126: 109}}
INFO 05-06 10:50:16.260774.260774 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.942ms | allocate_experts_across_cpu_gpu: 0.453ms
INFO 05-06 10:50:16.261812.261812 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.344650268554688e-05 seconds
INFO 05-06 10:50:16.261544.261544 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006287097930908203 seconds
INFO 05-06 10:50:16.262229.262229 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007920265197753906 seconds
INFO 05-06 10:50:16.273448.273448 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010434865951538086 seconds
INFO 05-06 10:50:16.274822.274822 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009915828704833984 seconds
INFO 05-06 10:50:16.276710.276710 mlpmodule.py:2799] [fused_experts] gmm total=2.130ms E=32 S=3694 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.276303.276303 mlpmodule.py:2799] [fused_experts] gmm total=2.300ms E=32 S=4181 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.276273.276273 mlpmodule.py:2799] [fused_experts] gmm total=2.238ms E=32 S=3219 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.277947.277947 mlpmodule.py:2799] [fused_experts] gmm total=2.432ms E=32 S=5290 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.278231.278231 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0036377906799316406 seconds
INFO 05-06 10:50:16.278580.278580 lmp.py:1496] [layer_moe_fused] to time: 4.863739013671875e-05 seconds
INFO 05-06 10:50:16.278720.278720 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003020763397216797 seconds
DEBUG 05-06 10:50:16.278118.278118 cuda_h.py:27] end *layer_moe_fused cost 19.384 ms
DEBUG 05-06 10:50:16.285933.285933 cuda_h.py:27] end prefill_layer cost 29.478 ms
DEBUG 05-06 10:50:16.285174.285174 lmp.py:841] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 10:50:16.285798.285798 lmp.py:824] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 10:50:16.288717.288717 cuda_h.py:27] end *sagl cost 1.927 ms
experts_cpu_alloc {'expert_ids': [99, 107, 67, 47, 59, 63, 15, 104, 72, 44, 96, 16, 9, 17, 89, 81, 101, 77, 34, 22, 30, 110, 126, 86], 'token_total': 192, 'token_per_expert': {99: 1, 107: 1, 67: 5, 47: 16, 59: 17, 63: 19, 15: 21, 104: 1, 72: 3, 44: 8, 96: 13, 16: 14, 9: 2, 17: 2, 89: 2, 81: 3, 101: 3, 77: 4, 34: 3, 22: 6, 30: 6, 110: 12, 126: 14, 86: 16}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 31, 35, 39, 43, 51, 55, 71, 75, 79, 83, 87, 91, 95, 103, 111, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4895, 'token_per_expert': {3: 644, 7: 630, 11: 24, 19: 34, 23: 67, 27: 27, 31: 200, 35: 142, 39: 24, 43: 416, 51: 199, 55: 50, 71: 37, 75: 79, 79: 220, 83: 118, 87: 497, 91: 27, 95: 257, 103: 305, 111: 182, 115: 269, 119: 154, 123: 203, 127: 90}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 24, 28, 32, 36, 40, 48, 56, 60, 64, 68, 76, 80, 84, 88, 100, 108, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3922, 'token_per_expert': {0: 534, 4: 569, 8: 131, 12: 147, 20: 63, 24: 307, 28: 99, 32: 24, 36: 165, 40: 52, 48: 250, 56: 75, 60: 29, 64: 152, 68: 25, 76: 196, 80: 23, 84: 15, 88: 268, 100: 292, 108: 97, 112: 57, 116: 16, 120: 319, 124: 17}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 85, 93, 97, 105, 109, 113, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3721, 'token_per_expert': {1: 636, 5: 513, 13: 182, 21: 43, 25: 181, 29: 14, 33: 311, 37: 215, 41: 144, 45: 404, 49: 63, 53: 104, 57: 7, 61: 144, 65: 318, 69: 8, 85: 70, 93: 7, 97: 7, 105: 81, 109: 119, 113: 23, 121: 94, 125: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 26, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 90, 94, 98, 106, 114, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3654, 'token_per_expert': {2: 521, 6: 522, 10: 27, 14: 102, 18: 127, 26: 51, 42: 99, 46: 198, 50: 399, 54: 74, 58: 34, 62: 217, 66: 91, 70: 154, 74: 36, 78: 210, 82: 226, 90: 85, 94: 61, 98: 107, 106: 70, 114: 76, 118: 88, 122: 79}}
INFO 05-06 10:50:16.290914.290914 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.600ms | allocate_experts_across_cpu_gpu: 0.452ms
INFO 05-06 10:50:16.290099.290099 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.58306884765625e-05 seconds
INFO 05-06 10:50:16.291610.291610 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006084442138671875 seconds
INFO 05-06 10:50:16.292692.292692 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008034706115722656 seconds
INFO 05-06 10:50:16.302358.302358 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.009922027587890625 seconds
INFO 05-06 10:50:16.303500.303500 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009617805480957031 seconds
INFO 05-06 10:50:16.305054.305054 mlpmodule.py:2799] [fused_experts] gmm total=1.939ms E=32 S=4975 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.305349.305349 mlpmodule.py:2799] [fused_experts] gmm total=1.904ms E=32 S=3711 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.305945.305945 mlpmodule.py:2799] [fused_experts] gmm total=2.194ms E=32 S=3961 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.305859.305859 mlpmodule.py:2799] [fused_experts] gmm total=2.534ms E=32 S=3737 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.306022.306022 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003475666046142578 seconds
INFO 05-06 10:50:16.306331.306331 lmp.py:1496] [layer_moe_fused] to time: 4.839897155761719e-05 seconds
INFO 05-06 10:50:16.307942.307942 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002968311309814453 seconds
DEBUG 05-06 10:50:16.307253.307253 cuda_h.py:27] end *layer_moe_fused cost 18.298 ms
DEBUG 05-06 10:50:16.313689.313689 cuda_h.py:27] end prefill_layer cost 27.791 ms
DEBUG 05-06 10:50:16.313261.313261 lmp.py:841] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 10:50:16.313362.313362 lmp.py:824] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 10:50:16.316951.316951 cuda_h.py:27] end *sagl cost 1.935 ms
experts_cpu_alloc {'expert_ids': [27, 63, 83, 31, 103, 99, 19, 28, 80, 124, 120, 29, 61, 93, 17, 45, 21, 38, 42, 102, 34, 26, 118], 'token_total': 70, 'token_per_expert': {27: 1, 63: 1, 83: 1, 31: 2, 103: 2, 99: 3, 19: 5, 28: 1, 80: 2, 124: 4, 120: 6, 29: 3, 61: 3, 93: 4, 17: 6, 45: 6, 21: 8, 38: 1, 42: 1, 102: 1, 34: 2, 26: 3, 118: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 23, 35, 39, 43, 47, 51, 55, 59, 67, 71, 75, 79, 87, 91, 95, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4491, 'token_per_expert': {3: 522, 7: 515, 11: 181, 15: 8, 23: 49, 35: 6, 39: 17, 43: 28, 47: 209, 51: 13, 55: 104, 59: 7, 67: 6, 71: 211, 75: 230, 79: 36, 87: 7, 91: 367, 95: 55, 111: 1167, 115: 402, 119: 194, 123: 110, 127: 47}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 24, 32, 36, 40, 44, 48, 52, 56, 60, 68, 72, 76, 84, 88, 92, 100, 104, 108, 112], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 28, 'token_total': 5207, 'token_per_expert': {0: 512, 4: 512, 8: 11, 12: 1076, 20: 884, 24: 71, 32: 136, 36: 18, 40: 294, 44: 34, 48: 27, 52: 221, 56: 8, 60: 51, 68: 180, 72: 13, 76: 480, 84: 90, 88: 60, 92: 18, 100: 65, 104: 111, 108: 7, 112: 328}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 33, 37, 49, 53, 57, 65, 69, 73, 77, 81, 85, 89, 97, 101, 105, 109, 113, 117, 121], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 3896, 'token_per_expert': {1: 578, 5: 587, 9: 48, 13: 115, 33: 44, 37: 78, 49: 890, 53: 213, 57: 406, 65: 46, 69: 59, 73: 14, 77: 142, 81: 15, 85: 72, 89: 103, 97: 67, 101: 106, 105: 53, 109: 15, 113: 122, 117: 53, 121: 70}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 30, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 90, 94, 98, 106, 110, 114, 122, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 2720, 'token_per_expert': {2: 512, 6: 518, 18: 71, 22: 127, 30: 84, 46: 181, 50: 12, 54: 8, 58: 12, 62: 64, 66: 5, 70: 154, 74: 39, 78: 188, 82: 11, 90: 293, 94: 45, 98: 41, 106: 90, 110: 178, 114: 6, 122: 26, 126: 55}}
INFO 05-06 10:50:16.318132.318132 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.878ms | allocate_experts_across_cpu_gpu: 0.440ms
INFO 05-06 10:50:16.318548.318548 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.20159912109375e-05 seconds
INFO 05-06 10:50:16.319411.319411 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000606536865234375 seconds
INFO 05-06 10:50:16.320672.320672 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.001004934310913086 seconds
INFO 05-06 10:50:16.330336.330336 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010273456573486328 seconds
INFO 05-06 10:50:16.331208.331208 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001012563705444336 seconds
INFO 05-06 10:50:16.334619.334619 mlpmodule.py:2799] [fused_experts] gmm total=2.009ms E=32 S=4506 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.334282.334282 mlpmodule.py:2799] [fused_experts] gmm total=2.169ms E=32 S=5220 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.334379.334379 mlpmodule.py:2799] [fused_experts] gmm total=2.264ms E=32 S=3926 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.334428.334428 mlpmodule.py:2799] [fused_experts] gmm total=2.284ms E=32 S=2732 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.335142.335142 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037088394165039062 seconds
INFO 05-06 10:50:16.335636.335636 lmp.py:1496] [layer_moe_fused] to time: 4.76837158203125e-05 seconds
INFO 05-06 10:50:16.336656.336656 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002467632293701172 seconds
DEBUG 05-06 10:50:16.336571.336571 cuda_h.py:27] end *layer_moe_fused cost 19.377 ms
DEBUG 05-06 10:50:16.343495.343495 cuda_h.py:27] end prefill_layer cost 29.304 ms
DEBUG 05-06 10:50:16.343498.343498 lmp.py:841] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 10:50:16.343360.343360 lmp.py:824] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 10:50:16.346505.346505 cuda_h.py:27] end *sagl cost 2.615 ms
experts_cpu_alloc {'expert_ids': [79, 51, 111, 127, 104, 68, 100, 12, 112, 72, 41, 45, 17, 65, 37, 125, 33, 38, 118, 122, 74, 70, 126, 102], 'token_total': 159, 'token_per_expert': {79: 2, 51: 4, 111: 8, 127: 8, 104: 1, 68: 3, 100: 3, 12: 4, 112: 5, 72: 7, 41: 1, 45: 2, 17: 5, 65: 13, 37: 14, 125: 15, 33: 17, 38: 5, 118: 5, 122: 5, 74: 6, 70: 8, 126: 8, 102: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 55, 63, 67, 71, 75, 83, 87, 91, 95, 99, 107, 115, 119, 123], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 29, 'token_total': 4473, 'token_per_expert': {3: 558, 7: 1021, 11: 56, 15: 79, 19: 343, 23: 173, 27: 222, 31: 33, 35: 32, 39: 9, 43: 337, 55: 12, 63: 30, 67: 101, 71: 175, 75: 33, 83: 30, 87: 16, 91: 504, 95: 87, 99: 349, 107: 81, 115: 68, 119: 26, 123: 98}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 40, 44, 48, 52, 56, 60, 64, 76, 80, 84, 88, 92, 96, 108, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4191, 'token_per_expert': {0: 518, 4: 886, 8: 21, 16: 139, 20: 303, 24: 42, 28: 253, 32: 48, 40: 11, 44: 56, 48: 112, 52: 411, 56: 212, 60: 165, 64: 453, 76: 24, 80: 53, 84: 32, 88: 25, 92: 60, 96: 37, 108: 59, 116: 65, 120: 68, 124: 138}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 29, 49, 53, 57, 61, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 3660, 'token_per_expert': {1: 553, 5: 522, 9: 78, 13: 37, 21: 43, 25: 48, 29: 133, 49: 140, 53: 98, 57: 243, 61: 198, 69: 79, 73: 81, 77: 100, 81: 132, 85: 87, 89: 74, 93: 117, 97: 181, 101: 71, 105: 39, 109: 78, 113: 120, 117: 200, 121: 208}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 42, 46, 50, 54, 58, 62, 66, 78, 82, 86, 90, 94, 98, 106, 114], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3901, 'token_per_expert': {2: 579, 6: 540, 10: 40, 14: 145, 18: 204, 22: 244, 26: 208, 30: 116, 34: 11, 42: 253, 46: 20, 50: 21, 54: 111, 58: 26, 62: 153, 66: 35, 78: 65, 82: 123, 86: 356, 90: 167, 94: 19, 98: 13, 106: 417, 114: 35}}
INFO 05-06 10:50:16.350787.350787 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 2.911ms | allocate_experts_across_cpu_gpu: 0.455ms
INFO 05-06 10:50:16.350117.350117 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.7738037109375e-05 seconds
INFO 05-06 10:50:16.351832.351832 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00061798095703125 seconds
INFO 05-06 10:50:16.352555.352555 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.000949859619140625 seconds
INFO 05-06 10:50:16.362172.362172 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010026216506958008 seconds
INFO 05-06 10:50:16.363009.363009 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000982522964477539 seconds
INFO 05-06 10:50:16.365316.365316 mlpmodule.py:2799] [fused_experts] gmm total=1.756ms E=32 S=3727 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.366564.366564 mlpmodule.py:2799] [fused_experts] gmm total=2.096ms E=32 S=4214 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.366482.366482 mlpmodule.py:2799] [fused_experts] gmm total=2.258ms E=32 S=4495 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.366400.366400 mlpmodule.py:2799] [fused_experts] gmm total=2.645ms E=32 S=3948 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.367370.367370 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003599882125854492 seconds
INFO 05-06 10:50:16.367149.367149 lmp.py:1496] [layer_moe_fused] to time: 4.8160552978515625e-05 seconds
INFO 05-06 10:50:16.367222.367222 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002522468566894531 seconds
DEBUG 05-06 10:50:16.368514.368514 cuda_h.py:27] end *layer_moe_fused cost 20.935 ms
DEBUG 05-06 10:50:16.374150.374150 cuda_h.py:27] end prefill_layer cost 31.298 ms
DEBUG 05-06 10:50:16.374245.374245 lmp.py:841] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 10:50:16.374346.374346 cuda_h.py:27] end prefill_step cost 1697.250 ms
INFO 05-06 10:50:16.374023.374023 lmp.py:843] prefill time: 1.8033685684204102 seconds
WARNING 05-06 10:50:16.417804.417804 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:50:16.417284.417284 helper.py:35]   NaN count (hidden): 1441792
WARNING 05-06 10:50:16.417796.417796 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:50:16.418900.418900 helper.py:39]   NaN count (normed): 1441792
WARNING 05-06 10:50:16.423738.423738 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:50:16.423146.423146 helper.py:50]   NaN count: 262144
WARNING 05-06 10:50:16.423115.423115 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:50:16.507836.507836 cuda_h.py:27] end init_inputs_tokens cost 106.081 ms
DEBUG 05-06 10:50:16.507304.507304 lmp.py:899] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:50:16.507233.507233 lmp.py:904] ---- decode step 0 layer 0 ----
DEBUG 05-06 10:50:16.514560.514560 cuda_h.py:27] end *sagl cost 7.133 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 47, 55, 63, 79, 83, 87, 91, 103, 127], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 17, 'token_per_expert': {15: 2, 47: 2, 55: 1, 63: 2, 79: 2, 83: 2, 87: 2, 91: 1, 103: 1, 127: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 32, 48, 60, 116, 124], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {0: 1, 8: 2, 32: 1, 48: 1, 60: 1, 116: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [33, 45, 53], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {33: 1, 45: 2, 53: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [22, 26, 90], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {22: 1, 26: 1, 90: 1}}
INFO 05-06 10:50:16.516283.516283 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.611ms | allocate_experts_across_cpu_gpu: 0.128ms
INFO 05-06 10:50:16.517597.517597 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-05 seconds
INFO 05-06 10:50:16.519973.519973 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0020868778228759766 seconds
INFO 05-06 10:50:16.521606.521606 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0021424293518066406 seconds
INFO 05-06 10:50:16.523007.523007 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001617431640625 seconds
INFO 05-06 10:50:16.524568.524568 mlpmodule.py:2799] [fused_experts] gmm total=1.188ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.525354.525354 mlpmodule.py:2799] [fused_experts] gmm total=1.237ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.525710.525710 mlpmodule.py:2799] [fused_experts] gmm total=2.151ms E=32 S=17 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.526706.526706 mlpmodule.py:2799] [fused_experts] gmm total=2.183ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.526168.526168 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0034456253051757812 seconds
INFO 05-06 10:50:16.526138.526138 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:50:16.527491.527491 cuda_h.py:27] end *layer_moe_fused cost 11.119 ms
DEBUG 05-06 10:50:16.527818.527818 cuda_h.py:27] end decode_layer cost 20.052 ms
DEBUG 05-06 10:50:16.527708.527708 lmp.py:904] ---- decode step 0 layer 1 ----
DEBUG 05-06 10:50:16.529232.529232 cuda_h.py:27] end *sagl cost 1.785 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [107, 119, 123], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {107: 2, 119: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 20, 28, 56, 92, 96, 124], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 15, 'token_per_expert': {0: 3, 8: 2, 20: 1, 28: 1, 56: 2, 92: 2, 96: 1, 124: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 73, 97, 121], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {9: 2, 73: 1, 97: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [30, 46, 54, 110], 'expert_count': 4, 'ideal_gpu_count': 4, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {30: 2, 46: 1, 54: 2, 110: 2}}
INFO 05-06 10:50:16.531202.531202 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.328ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 10:50:16.531357.531357 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8596649169921875e-05 seconds
INFO 05-06 10:50:16.532387.532387 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001458883285522461 seconds
INFO 05-06 10:50:16.534324.534324 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012314319610595703 seconds
INFO 05-06 10:50:16.535580.535580 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014600753784179688 seconds
INFO 05-06 10:50:16.537290.537290 mlpmodule.py:2799] [fused_experts] gmm total=1.428ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.537216.537216 mlpmodule.py:2799] [fused_experts] gmm total=1.714ms E=32 S=15 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.537661.537661 mlpmodule.py:2799] [fused_experts] gmm total=2.044ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.538251.538251 mlpmodule.py:2799] [fused_experts] gmm total=2.560ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.539786.539786 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037009716033935547 seconds
INFO 05-06 10:50:16.539042.539042 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 10:50:16.539470.539470 cuda_h.py:27] end *layer_moe_fused cost 9.224 ms
DEBUG 05-06 10:50:16.540902.540902 cuda_h.py:27] end decode_layer cost 12.533 ms
DEBUG 05-06 10:50:16.540408.540408 lmp.py:904] ---- decode step 0 layer 2 ----
DEBUG 05-06 10:50:16.541101.541101 cuda_h.py:27] end *sagl cost 1.490 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 91], 'expert_count': 2, 'ideal_gpu_count': 5, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 5, 'token_per_expert': {11: 3, 91: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [8, 12, 76, 120], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 8, 'token_per_expert': {8: 1, 12: 2, 76: 4, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [13, 41, 45, 49, 61, 81, 97, 109], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {13: 1, 41: 2, 45: 1, 49: 1, 61: 1, 81: 2, 97: 1, 109: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [62, 70, 90, 102, 106, 126], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {62: 2, 70: 1, 90: 1, 102: 1, 106: 2, 126: 2}}
INFO 05-06 10:50:16.543639.543639 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.313ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:50:16.543986.543986 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8358230590820312e-05 seconds
INFO 05-06 10:50:16.544053.544053 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013477802276611328 seconds
INFO 05-06 10:50:16.546600.546600 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012590885162353516 seconds
INFO 05-06 10:50:16.547429.547429 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015516281127929688 seconds
INFO 05-06 10:50:16.549431.549431 mlpmodule.py:2799] [fused_experts] gmm total=1.846ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.550189.550189 mlpmodule.py:2799] [fused_experts] gmm total=2.032ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.550242.550242 mlpmodule.py:2799] [fused_experts] gmm total=2.109ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.551105.551105 mlpmodule.py:2799] [fused_experts] gmm total=2.858ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.552437.552437 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004236936569213867 seconds
INFO 05-06 10:50:16.552047.552047 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 8.320808410644531e-05 seconds
DEBUG 05-06 10:50:16.553787.553787 cuda_h.py:27] end *layer_moe_fused cost 10.157 ms
DEBUG 05-06 10:50:16.553847.553847 cuda_h.py:27] end decode_layer cost 13.512 ms
DEBUG 05-06 10:50:16.554420.554420 lmp.py:904] ---- decode step 0 layer 3 ----
DEBUG 05-06 10:50:16.557930.557930 cuda_h.py:27] end *sagl cost 3.279 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [39, 107], 'expert_count': 2, 'ideal_gpu_count': 7, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 2, 'token_per_expert': {39: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 24, 40, 44, 56, 96, 104, 116], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {4: 1, 24: 1, 40: 1, 44: 1, 56: 1, 96: 3, 104: 3, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 73, 85, 101, 117, 125], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {5: 1, 73: 1, 85: 1, 101: 1, 117: 2, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [26, 30, 34, 50, 54, 70, 110, 118, 126], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 11, 'token_per_expert': {26: 2, 30: 1, 34: 1, 50: 2, 54: 1, 70: 1, 110: 1, 118: 1, 126: 1}}
INFO 05-06 10:50:16.559290.559290 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.440ms | allocate_experts_across_cpu_gpu: 0.152ms
INFO 05-06 10:50:16.559446.559446 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.7894973754882812e-05 seconds
INFO 05-06 10:50:16.561735.561735 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014472007751464844 seconds
INFO 05-06 10:50:16.563904.563904 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0019381046295166016 seconds
INFO 05-06 10:50:16.564626.564626 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014834403991699219 seconds
INFO 05-06 10:50:16.567574.567574 mlpmodule.py:2799] [fused_experts] gmm total=1.839ms E=32 S=2 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.567563.567563 mlpmodule.py:2799] [fused_experts] gmm total=2.199ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.567604.567604 mlpmodule.py:2799] [fused_experts] gmm total=2.340ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.568383.568383 mlpmodule.py:2799] [fused_experts] gmm total=3.342ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.569366.569366 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004537820816040039 seconds
INFO 05-06 10:50:16.569318.569318 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.745887756347656e-05 seconds
DEBUG 05-06 10:50:16.570489.570489 cuda_h.py:27] end *layer_moe_fused cost 11.196 ms
DEBUG 05-06 10:50:16.570546.570546 cuda_h.py:27] end decode_layer cost 16.592 ms
DEBUG 05-06 10:50:16.570641.570641 lmp.py:904] ---- decode step 0 layer 4 ----
DEBUG 05-06 10:50:16.572248.572248 cuda_h.py:27] end *sagl cost 1.872 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 31, 51, 59, 67, 71, 87, 111], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {3: 2, 31: 1, 51: 2, 59: 1, 67: 1, 71: 1, 87: 1, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [20, 60, 104], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {20: 3, 60: 1, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [17, 25, 29, 45, 93, 101, 121], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {17: 1, 25: 1, 29: 1, 45: 2, 93: 1, 101: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 50, 106, 114, 118, 122, 126], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 1, 50: 2, 106: 2, 114: 1, 118: 1, 122: 1, 126: 1}}
INFO 05-06 10:50:16.574300.574300 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.385ms | allocate_experts_across_cpu_gpu: 0.128ms
INFO 05-06 10:50:16.574635.574635 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.4557113647460938e-05 seconds
INFO 05-06 10:50:16.576167.576167 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015950202941894531 seconds
INFO 05-06 10:50:16.578994.578994 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.002004384994506836 seconds
INFO 05-06 10:50:16.579364.579364 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014874935150146484 seconds
INFO 05-06 10:50:16.582567.582567 mlpmodule.py:2799] [fused_experts] gmm total=1.974ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.582897.582897 mlpmodule.py:2799] [fused_experts] gmm total=2.243ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.582462.582462 mlpmodule.py:2799] [fused_experts] gmm total=2.307ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.583208.583208 mlpmodule.py:2799] [fused_experts] gmm total=3.147ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.584130.584130 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00439906120300293 seconds
INFO 05-06 10:50:16.584478.584478 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:50:16.584486.584486 cuda_h.py:27] end *layer_moe_fused cost 11.012 ms
DEBUG 05-06 10:50:16.585940.585940 cuda_h.py:27] end decode_layer cost 14.472 ms
DEBUG 05-06 10:50:16.585876.585876 lmp.py:904] ---- decode step 0 layer 5 ----
DEBUG 05-06 10:50:16.586084.586084 cuda_h.py:27] end *sagl cost 1.614 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 39, 71, 95, 99, 123], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {3: 1, 7: 1, 39: 2, 71: 1, 95: 2, 99: 2, 123: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 52, 72, 116], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {0: 1, 4: 1, 36: 1, 52: 2, 72: 1, 116: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 61, 65], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {1: 1, 5: 2, 61: 2, 65: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 46, 70, 74, 94], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 2, 6: 1, 46: 1, 70: 2, 74: 1, 94: 1}}
INFO 05-06 10:50:16.588000.588000 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.298ms | allocate_experts_across_cpu_gpu: 0.090ms
INFO 05-06 10:50:16.588871.588871 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9550323486328125e-05 seconds
INFO 05-06 10:50:16.589887.589887 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014510154724121094 seconds
INFO 05-06 10:50:16.591674.591674 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012831687927246094 seconds
INFO 05-06 10:50:16.592533.592533 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014452934265136719 seconds
INFO 05-06 10:50:16.595903.595903 mlpmodule.py:2799] [fused_experts] gmm total=2.029ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.595136.595136 mlpmodule.py:2799] [fused_experts] gmm total=2.150ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.595001.595001 mlpmodule.py:2799] [fused_experts] gmm total=2.474ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.595946.595946 mlpmodule.py:2799] [fused_experts] gmm total=2.721ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.597275.597275 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004307985305786133 seconds
INFO 05-06 10:50:16.597869.597869 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9591064453125e-05 seconds
DEBUG 05-06 10:50:16.597245.597245 cuda_h.py:27] end *layer_moe_fused cost 9.490 ms
DEBUG 05-06 10:50:16.597604.597604 cuda_h.py:27] end decode_layer cost 12.645 ms
DEBUG 05-06 10:50:16.598493.598493 lmp.py:904] ---- decode step 0 layer 6 ----
DEBUG 05-06 10:50:16.599141.599141 cuda_h.py:27] end *sagl cost 1.701 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 35, 43, 87, 115], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {3: 1, 7: 1, 35: 2, 43: 1, 87: 3, 115: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 32, 68, 96], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 1, 4: 1, 24: 2, 32: 1, 68: 1, 96: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {1: 3, 5: 1, 13: 1, 25: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 78, 106, 118], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 3, 6: 1, 78: 2, 106: 1, 118: 1}}
INFO 05-06 10:50:16.601016.601016 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.356ms | allocate_experts_across_cpu_gpu: 0.107ms
INFO 05-06 10:50:16.601185.601185 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.002716064453125e-05 seconds
INFO 05-06 10:50:16.602759.602759 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001468658447265625 seconds
INFO 05-06 10:50:16.604312.604312 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001422882080078125 seconds
INFO 05-06 10:50:16.605458.605458 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001508951187133789 seconds
INFO 05-06 10:50:16.608655.608655 mlpmodule.py:2799] [fused_experts] gmm total=1.992ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.608873.608873 mlpmodule.py:2799] [fused_experts] gmm total=2.264ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.608025.608025 mlpmodule.py:2799] [fused_experts] gmm total=2.281ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.608370.608370 mlpmodule.py:2799] [fused_experts] gmm total=2.405ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.610121.610121 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00431513786315918 seconds
INFO 05-06 10:50:16.610377.610377 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:50:16.610622.610622 cuda_h.py:27] end *layer_moe_fused cost 9.867 ms
DEBUG 05-06 10:50:16.611892.611892 cuda_h.py:27] end decode_layer cost 13.105 ms
DEBUG 05-06 10:50:16.611636.611636 lmp.py:904] ---- decode step 0 layer 7 ----
DEBUG 05-06 10:50:16.612066.612066 cuda_h.py:27] end *sagl cost 1.716 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 43], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {3: 1, 7: 1, 19: 1, 43: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 64, 96, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 1, 4: 1, 20: 2, 64: 1, 96: 2, 104: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 69, 97, 121], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 1, 5: 1, 9: 1, 69: 1, 97: 2, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 18, 34, 82, 90, 106, 114], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 10, 'token_per_expert': {2: 1, 6: 1, 10: 1, 18: 1, 34: 1, 82: 1, 90: 2, 106: 1, 114: 1}}
INFO 05-06 10:50:16.614907.614907 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.325ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 10:50:16.614976.614976 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:50:16.615267.615267 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013375282287597656 seconds
INFO 05-06 10:50:16.617054.617054 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001295328140258789 seconds
INFO 05-06 10:50:16.619591.619591 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001911163330078125 seconds
INFO 05-06 10:50:16.621497.621497 mlpmodule.py:2799] [fused_experts] gmm total=1.918ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.621320.621320 mlpmodule.py:2799] [fused_experts] gmm total=2.077ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.621670.621670 mlpmodule.py:2799] [fused_experts] gmm total=2.343ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.622065.622065 mlpmodule.py:2799] [fused_experts] gmm total=2.969ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.623691.623691 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004327058792114258 seconds
INFO 05-06 10:50:16.623053.623053 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.650520324707031e-05 seconds
DEBUG 05-06 10:50:16.623117.623117 cuda_h.py:27] end *layer_moe_fused cost 10.061 ms
DEBUG 05-06 10:50:16.624185.624185 cuda_h.py:27] end decode_layer cost 13.214 ms
DEBUG 05-06 10:50:16.624167.624167 lmp.py:904] ---- decode step 0 layer 8 ----
DEBUG 05-06 10:50:16.626199.626199 cuda_h.py:27] end *sagl cost 1.530 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 51, 55, 63, 75, 103], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {3: 1, 7: 1, 27: 1, 51: 3, 55: 2, 63: 1, 75: 1, 103: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 24, 64], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {0: 1, 4: 1, 12: 2, 24: 1, 64: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 69, 93], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {1: 1, 5: 1, 69: 2, 93: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 42, 46, 50, 54, 110], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 1, 6: 2, 42: 1, 46: 1, 50: 2, 54: 1, 110: 1}}
INFO 05-06 10:50:16.627480.627480 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.313ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:50:16.627496.627496 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:50:16.628468.628468 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014748573303222656 seconds
INFO 05-06 10:50:16.630993.630993 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001310110092163086 seconds
INFO 05-06 10:50:16.631054.631054 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001373291015625 seconds
INFO 05-06 10:50:16.634465.634465 mlpmodule.py:2799] [fused_experts] gmm total=2.210ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.634481.634481 mlpmodule.py:2799] [fused_experts] gmm total=2.367ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.634035.634035 mlpmodule.py:2799] [fused_experts] gmm total=2.399ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.635032.635032 mlpmodule.py:2799] [fused_experts] gmm total=2.834ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.636569.636569 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0043523311614990234 seconds
INFO 05-06 10:50:16.636109.636109 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 10:50:16.636499.636499 cuda_h.py:27] end *layer_moe_fused cost 9.696 ms
DEBUG 05-06 10:50:16.637710.637710 cuda_h.py:27] end decode_layer cost 12.648 ms
DEBUG 05-06 10:50:16.637645.637645 lmp.py:904] ---- decode step 0 layer 9 ----
DEBUG 05-06 10:50:16.638851.638851 cuda_h.py:27] end *sagl cost 1.552 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 95], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 9, 'token_per_expert': {3: 1, 7: 2, 15: 1, 19: 2, 95: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 48, 76], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 36: 1, 48: 1, 76: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 69, 81, 89, 101], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 11, 'token_per_expert': {1: 1, 5: 1, 37: 1, 69: 2, 81: 1, 89: 2, 101: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 70, 74], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 1, 6: 1, 30: 2, 70: 1, 74: 2}}
INFO 05-06 10:50:16.640682.640682 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.093ms
INFO 05-06 10:50:16.640698.640698 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8596649169921875e-05 seconds
INFO 05-06 10:50:16.641285.641285 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012755393981933594 seconds
INFO 05-06 10:50:16.642655.642655 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013031959533691406 seconds
INFO 05-06 10:50:16.644070.644070 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001260519027709961 seconds
INFO 05-06 10:50:16.646131.646131 mlpmodule.py:2799] [fused_experts] gmm total=1.831ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.646436.646436 mlpmodule.py:2799] [fused_experts] gmm total=1.969ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.646841.646841 mlpmodule.py:2799] [fused_experts] gmm total=2.116ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.647993.647993 mlpmodule.py:2799] [fused_experts] gmm total=2.814ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.648499.648499 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0040471553802490234 seconds
INFO 05-06 10:50:16.648324.648324 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:50:16.648474.648474 cuda_h.py:27] end *layer_moe_fused cost 9.243 ms
DEBUG 05-06 10:50:16.649814.649814 cuda_h.py:27] end decode_layer cost 12.162 ms
DEBUG 05-06 10:50:16.649227.649227 lmp.py:904] ---- decode step 0 layer 10 ----
DEBUG 05-06 10:50:16.650431.650431 cuda_h.py:27] end *sagl cost 1.516 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 79], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {3: 1, 7: 1, 79: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 28, 44, 60], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 1, 4: 1, 8: 2, 28: 1, 44: 2, 60: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 37, 57, 81, 97, 105], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 13, 'token_per_expert': {1: 1, 5: 1, 21: 1, 37: 1, 57: 1, 81: 3, 97: 3, 105: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 46, 54, 126], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {2: 1, 6: 1, 18: 1, 46: 2, 54: 1, 126: 1}}
INFO 05-06 10:50:16.652288.652288 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.311ms | allocate_experts_across_cpu_gpu: 0.094ms
INFO 05-06 10:50:16.652880.652880 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 10:50:16.653033.653033 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015347003936767578 seconds
INFO 05-06 10:50:16.655070.655070 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013537406921386719 seconds
INFO 05-06 10:50:16.656587.656587 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013213157653808594 seconds
INFO 05-06 10:50:16.659125.659125 mlpmodule.py:2799] [fused_experts] gmm total=1.865ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.659919.659919 mlpmodule.py:2799] [fused_experts] gmm total=2.133ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.659223.659223 mlpmodule.py:2799] [fused_experts] gmm total=2.227ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.660646.660646 mlpmodule.py:2799] [fused_experts] gmm total=2.685ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.661855.661855 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0040988922119140625 seconds
INFO 05-06 10:50:16.661965.661965 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:50:16.661156.661156 cuda_h.py:27] end *layer_moe_fused cost 9.510 ms
DEBUG 05-06 10:50:16.661968.661968 cuda_h.py:27] end decode_layer cost 12.444 ms
DEBUG 05-06 10:50:16.661758.661758 lmp.py:904] ---- decode step 0 layer 11 ----
DEBUG 05-06 10:50:16.663962.663962 cuda_h.py:27] end *sagl cost 1.689 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 67, 79, 83, 99], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 1, 7: 1, 23: 1, 67: 1, 79: 2, 83: 3, 99: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 124], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {0: 1, 4: 1, 124: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 49, 81], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 1, 5: 1, 9: 1, 49: 2, 81: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 38, 46, 50, 70, 102, 114], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 2, 6: 1, 38: 2, 46: 2, 50: 1, 70: 1, 102: 1, 114: 1}}
INFO 05-06 10:50:16.664569.664569 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.322ms | allocate_experts_across_cpu_gpu: 0.092ms
INFO 05-06 10:50:16.665439.665439 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 10:50:16.666587.666587 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014073848724365234 seconds
INFO 05-06 10:50:16.667295.667295 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012984275817871094 seconds
INFO 05-06 10:50:16.669550.669550 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001420736312866211 seconds
INFO 05-06 10:50:16.671422.671422 mlpmodule.py:2799] [fused_experts] gmm total=1.799ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.671184.671184 mlpmodule.py:2799] [fused_experts] gmm total=2.278ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.672224.672224 mlpmodule.py:2799] [fused_experts] gmm total=2.291ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.672706.672706 mlpmodule.py:2799] [fused_experts] gmm total=2.774ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.673555.673555 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004046440124511719 seconds
INFO 05-06 10:50:16.673857.673857 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 10:50:16.673268.673268 cuda_h.py:27] end *layer_moe_fused cost 9.428 ms
DEBUG 05-06 10:50:16.674508.674508 cuda_h.py:27] end decode_layer cost 12.469 ms
DEBUG 05-06 10:50:16.674443.674443 lmp.py:904] ---- decode step 0 layer 12 ----
DEBUG 05-06 10:50:16.676509.676509 cuda_h.py:27] end *sagl cost 1.517 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 1, 7: 1, 19: 2, 39: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 80], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 36: 2, 80: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 45, 73, 97, 117], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {1: 1, 5: 1, 21: 1, 45: 1, 73: 1, 97: 1, 117: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 46, 50, 74, 78, 86, 106, 114], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 13, 'token_per_expert': {2: 1, 6: 1, 46: 1, 50: 1, 74: 2, 78: 3, 86: 1, 106: 2, 114: 1}}
INFO 05-06 10:50:16.677206.677206 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.305ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 10:50:16.677481.677481 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:50:16.678344.678344 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014071464538574219 seconds
INFO 05-06 10:50:16.680661.680661 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001291036605834961 seconds
INFO 05-06 10:50:16.681757.681757 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014100074768066406 seconds
INFO 05-06 10:50:16.683654.683654 mlpmodule.py:2799] [fused_experts] gmm total=2.049ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.684131.684131 mlpmodule.py:2799] [fused_experts] gmm total=2.130ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.684504.684504 mlpmodule.py:2799] [fused_experts] gmm total=2.307ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.686047.686047 mlpmodule.py:2799] [fused_experts] gmm total=4.418ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.687582.687582 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005597114562988281 seconds
INFO 05-06 10:50:16.687580.687580 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.151199340820312e-05 seconds
DEBUG 05-06 10:50:16.687131.687131 cuda_h.py:27] end *layer_moe_fused cost 11.021 ms
DEBUG 05-06 10:50:16.688505.688505 cuda_h.py:27] end decode_layer cost 13.909 ms
DEBUG 05-06 10:50:16.688679.688679 lmp.py:904] ---- decode step 0 layer 13 ----
DEBUG 05-06 10:50:16.690501.690501 cuda_h.py:27] end *sagl cost 1.758 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 47, 107], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {3: 1, 7: 1, 47: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 32, 80, 100, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {0: 1, 4: 1, 32: 1, 80: 2, 100: 3, 104: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 41, 125], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 3, 5: 1, 41: 1, 125: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 26, 78, 110, 114], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {2: 2, 6: 1, 14: 1, 22: 1, 26: 1, 78: 2, 110: 1, 114: 1}}
INFO 05-06 10:50:16.691540.691540 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.380ms | allocate_experts_across_cpu_gpu: 0.117ms
INFO 05-06 10:50:16.691145.691145 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0503997802734375e-05 seconds
INFO 05-06 10:50:16.693474.693474 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014624595642089844 seconds
INFO 05-06 10:50:16.694859.694859 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0015430450439453125 seconds
INFO 05-06 10:50:16.696733.696733 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001310110092163086 seconds
INFO 05-06 10:50:16.698833.698833 mlpmodule.py:2799] [fused_experts] gmm total=1.970ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.698993.698993 mlpmodule.py:2799] [fused_experts] gmm total=2.041ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.698621.698621 mlpmodule.py:2799] [fused_experts] gmm total=2.296ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.699698.699698 mlpmodule.py:2799] [fused_experts] gmm total=2.273ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.700603.700603 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003944873809814453 seconds
INFO 05-06 10:50:16.700143.700143 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.935264587402344e-05 seconds
DEBUG 05-06 10:50:16.700620.700620 cuda_h.py:27] end *layer_moe_fused cost 9.619 ms
DEBUG 05-06 10:50:16.701902.701902 cuda_h.py:27] end decode_layer cost 12.974 ms
DEBUG 05-06 10:50:16.701123.701123 lmp.py:904] ---- decode step 0 layer 14 ----
DEBUG 05-06 10:50:16.703910.703910 cuda_h.py:27] end *sagl cost 1.524 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 39, 47, 99, 115], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 1, 7: 1, 11: 1, 39: 2, 47: 1, 99: 1, 115: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 56], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {0: 1, 4: 1, 56: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 81, 97, 109, 121], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {1: 1, 5: 1, 13: 1, 25: 2, 81: 2, 97: 1, 109: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 26, 46], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 3, 6: 1, 10: 1, 26: 2, 46: 1}}
INFO 05-06 10:50:16.704523.704523 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.099ms
INFO 05-06 10:50:16.704632.704632 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7404556274414062e-05 seconds
INFO 05-06 10:50:16.705657.705657 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001318216323852539 seconds
INFO 05-06 10:50:16.707391.707391 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012905597686767578 seconds
INFO 05-06 10:50:16.708256.708256 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012402534484863281 seconds
INFO 05-06 10:50:16.710195.710195 mlpmodule.py:2799] [fused_experts] gmm total=1.841ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.710466.710466 mlpmodule.py:2799] [fused_experts] gmm total=2.114ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.710564.710564 mlpmodule.py:2799] [fused_experts] gmm total=2.111ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.711474.711474 mlpmodule.py:2799] [fused_experts] gmm total=2.910ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.712267.712267 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004231929779052734 seconds
INFO 05-06 10:50:16.712138.712138 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.744529724121094e-05 seconds
DEBUG 05-06 10:50:16.713095.713095 cuda_h.py:27] end *layer_moe_fused cost 9.403 ms
DEBUG 05-06 10:50:16.713345.713345 cuda_h.py:27] end decode_layer cost 12.377 ms
DEBUG 05-06 10:50:16.713804.713804 lmp.py:904] ---- decode step 0 layer 15 ----
DEBUG 05-06 10:50:16.715293.715293 cuda_h.py:27] end *sagl cost 1.515 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 75, 83, 119], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 1, 7: 1, 75: 2, 83: 1, 119: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 24, 68, 72, 108, 112], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 1, 4: 1, 24: 1, 68: 1, 72: 2, 108: 1, 112: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 65, 69, 81, 93, 97, 101], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {1: 1, 5: 1, 33: 2, 65: 1, 69: 1, 81: 2, 93: 1, 97: 1, 101: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 30, 34], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {2: 1, 6: 1, 30: 1, 34: 1}}
INFO 05-06 10:50:16.716575.716575 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.305ms | allocate_experts_across_cpu_gpu: 0.103ms
INFO 05-06 10:50:16.716445.716445 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:50:16.718991.718991 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014193058013916016 seconds
INFO 05-06 10:50:16.719645.719645 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012793540954589844 seconds
INFO 05-06 10:50:16.720375.720375 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011410713195800781 seconds
INFO 05-06 10:50:16.723129.723129 mlpmodule.py:2799] [fused_experts] gmm total=2.131ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.723359.723359 mlpmodule.py:2799] [fused_experts] gmm total=2.200ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.723741.723741 mlpmodule.py:2799] [fused_experts] gmm total=2.340ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.724673.724673 mlpmodule.py:2799] [fused_experts] gmm total=2.757ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.725822.725822 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004202365875244141 seconds
INFO 05-06 10:50:16.725647.725647 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 10:50:16.725236.725236 cuda_h.py:27] end *layer_moe_fused cost 9.080 ms
DEBUG 05-06 10:50:16.726622.726622 cuda_h.py:27] end decode_layer cost 12.149 ms
DEBUG 05-06 10:50:16.726080.726080 lmp.py:904] ---- decode step 0 layer 16 ----
DEBUG 05-06 10:50:16.727630.727630 cuda_h.py:27] end *sagl cost 1.524 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 87, 107], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {3: 1, 7: 1, 19: 2, 87: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 20, 32, 44], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {0: 1, 4: 2, 20: 1, 32: 1, 44: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 85, 105], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 9, 'token_per_expert': {1: 2, 5: 3, 85: 2, 105: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 54, 62, 66, 78, 82, 102], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 6: 1, 54: 2, 62: 2, 66: 1, 78: 1, 82: 1, 102: 2}}
INFO 05-06 10:50:16.728560.728560 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.316ms | allocate_experts_across_cpu_gpu: 0.092ms
INFO 05-06 10:50:16.729337.729337 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.7642974853515625e-05 seconds
INFO 05-06 10:50:16.730673.730673 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014758110046386719 seconds
INFO 05-06 10:50:16.731021.731021 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012450218200683594 seconds
INFO 05-06 10:50:16.733139.733139 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014607906341552734 seconds
INFO 05-06 10:50:16.735334.735334 mlpmodule.py:2799] [fused_experts] gmm total=1.857ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.735584.735584 mlpmodule.py:2799] [fused_experts] gmm total=2.119ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.736902.736902 mlpmodule.py:2799] [fused_experts] gmm total=2.317ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.736142.736142 mlpmodule.py:2799] [fused_experts] gmm total=2.852ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.737051.737051 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004046440124511719 seconds
INFO 05-06 10:50:16.737784.737784 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:50:16.738771.738771 cuda_h.py:27] end *layer_moe_fused cost 9.484 ms
DEBUG 05-06 10:50:16.738010.738010 cuda_h.py:27] end decode_layer cost 12.337 ms
DEBUG 05-06 10:50:16.738423.738423 lmp.py:904] ---- decode step 0 layer 17 ----
DEBUG 05-06 10:50:16.740669.740669 cuda_h.py:27] end *sagl cost 1.581 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 47], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {3: 1, 7: 2, 23: 3, 47: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 68], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 16: 2, 68: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 33, 53, 65, 73, 113], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {1: 1, 5: 2, 13: 2, 33: 1, 53: 1, 65: 1, 73: 2, 113: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 34, 70], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 1, 6: 1, 18: 2, 22: 1, 34: 2, 70: 1}}
INFO 05-06 10:50:16.741357.741357 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.323ms | allocate_experts_across_cpu_gpu: 0.103ms
INFO 05-06 10:50:16.741757.741757 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 10:50:16.742251.742251 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012755393981933594 seconds
INFO 05-06 10:50:16.744549.744549 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001318216323852539 seconds
INFO 05-06 10:50:16.745876.745876 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013892650604248047 seconds
INFO 05-06 10:50:16.747521.747521 mlpmodule.py:2799] [fused_experts] gmm total=1.882ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.748391.748391 mlpmodule.py:2799] [fused_experts] gmm total=2.064ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.748987.748987 mlpmodule.py:2799] [fused_experts] gmm total=2.162ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.749517.749517 mlpmodule.py:2799] [fused_experts] gmm total=2.844ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.750417.750417 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0043375492095947266 seconds
INFO 05-06 10:50:16.750957.750957 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:50:16.751302.751302 cuda_h.py:27] end *layer_moe_fused cost 10.066 ms
DEBUG 05-06 10:50:16.751324.751324 cuda_h.py:27] end decode_layer cost 13.053 ms
DEBUG 05-06 10:50:16.751506.751506 lmp.py:904] ---- decode step 0 layer 18 ----
DEBUG 05-06 10:50:16.753108.753108 cuda_h.py:27] end *sagl cost 1.528 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 59, 75, 83, 111], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 1, 7: 1, 23: 2, 59: 1, 75: 2, 83: 2, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 40, 84, 104], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {0: 1, 4: 1, 40: 2, 84: 1, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 37, 73, 77, 97, 105], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {1: 1, 5: 1, 37: 2, 73: 1, 77: 1, 97: 1, 105: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 30, 42, 54, 58], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {2: 1, 6: 1, 26: 1, 30: 2, 42: 1, 54: 1, 58: 1}}
INFO 05-06 10:50:16.754859.754859 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.307ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:50:16.754537.754537 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9788742065429688e-05 seconds
INFO 05-06 10:50:16.755942.755942 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013570785522460938 seconds
INFO 05-06 10:50:16.757959.757959 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013577938079833984 seconds
INFO 05-06 10:50:16.759503.759503 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015518665313720703 seconds
INFO 05-06 10:50:16.761101.761101 mlpmodule.py:2799] [fused_experts] gmm total=2.253ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.761247.761247 mlpmodule.py:2799] [fused_experts] gmm total=2.257ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.762867.762867 mlpmodule.py:2799] [fused_experts] gmm total=2.472ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.762781.762781 mlpmodule.py:2799] [fused_experts] gmm total=3.082ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.763964.763964 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0045092105865478516 seconds
INFO 05-06 10:50:16.763882.763882 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.8160552978515625e-05 seconds
DEBUG 05-06 10:50:16.764149.764149 cuda_h.py:27] end *layer_moe_fused cost 10.276 ms
DEBUG 05-06 10:50:16.764345.764345 cuda_h.py:27] end decode_layer cost 13.183 ms
DEBUG 05-06 10:50:16.764658.764658 lmp.py:904] ---- decode step 0 layer 19 ----
DEBUG 05-06 10:50:16.766109.766109 cuda_h.py:27] end *sagl cost 1.558 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 111], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {3: 1, 7: 1, 19: 1, 31: 1, 111: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 40, 44, 84], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 1, 4: 1, 40: 2, 44: 2, 84: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 25, 61, 125], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 1, 5: 2, 25: 2, 61: 1, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 78, 82, 86, 106, 122], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {2: 1, 6: 1, 10: 2, 78: 2, 82: 1, 86: 1, 106: 3, 122: 1}}
INFO 05-06 10:50:16.767099.767099 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.325ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:50:16.767261.767261 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.811981201171875e-05 seconds
INFO 05-06 10:50:16.769323.769323 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014126300811767578 seconds
INFO 05-06 10:50:16.770659.770659 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012707710266113281 seconds
INFO 05-06 10:50:16.772941.772941 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014433860778808594 seconds
INFO 05-06 10:50:16.774014.774014 mlpmodule.py:2799] [fused_experts] gmm total=1.880ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.774267.774267 mlpmodule.py:2799] [fused_experts] gmm total=2.193ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.774183.774183 mlpmodule.py:2799] [fused_experts] gmm total=2.255ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.775055.775055 mlpmodule.py:2799] [fused_experts] gmm total=3.084ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.776153.776153 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004302263259887695 seconds
INFO 05-06 10:50:16.776363.776363 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.982948303222656e-05 seconds
DEBUG 05-06 10:50:16.777913.777913 cuda_h.py:27] end *layer_moe_fused cost 9.697 ms
DEBUG 05-06 10:50:16.777591.777591 cuda_h.py:27] end decode_layer cost 12.610 ms
DEBUG 05-06 10:50:16.777526.777526 lmp.py:904] ---- decode step 0 layer 20 ----
DEBUG 05-06 10:50:16.779296.779296 cuda_h.py:27] end *sagl cost 1.565 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 95, 107], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 1, 95: 1, 107: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 36, 40, 52, 60], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 6, 'token_per_expert': {0: 1, 4: 1, 36: 1, 40: 1, 52: 1, 60: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 73, 85, 117], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 13, 'token_per_expert': {1: 1, 5: 1, 13: 2, 21: 3, 73: 2, 85: 2, 117: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 62, 94, 102], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 3, 6: 1, 62: 1, 94: 1, 102: 1}}
INFO 05-06 10:50:16.780187.780187 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.311ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 10:50:16.780534.780534 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8358230590820312e-05 seconds
INFO 05-06 10:50:16.782305.782305 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014448165893554688 seconds
INFO 05-06 10:50:16.783581.783581 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001260995864868164 seconds
INFO 05-06 10:50:16.784600.784600 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001279592514038086 seconds
INFO 05-06 10:50:16.786658.786658 mlpmodule.py:2799] [fused_experts] gmm total=1.820ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.787202.787202 mlpmodule.py:2799] [fused_experts] gmm total=2.118ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.787486.787486 mlpmodule.py:2799] [fused_experts] gmm total=2.127ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.788702.788702 mlpmodule.py:2799] [fused_experts] gmm total=2.960ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.789240.789240 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004200458526611328 seconds
INFO 05-06 10:50:16.789734.789734 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.863739013671875e-05 seconds
DEBUG 05-06 10:50:16.789988.789988 cuda_h.py:27] end *layer_moe_fused cost 9.500 ms
DEBUG 05-06 10:50:16.790452.790452 cuda_h.py:27] end decode_layer cost 12.404 ms
DEBUG 05-06 10:50:16.790387.790387 lmp.py:904] ---- decode step 0 layer 21 ----
DEBUG 05-06 10:50:16.791107.791107 cuda_h.py:27] end *sagl cost 1.684 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 59, 83, 87, 103], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 1, 7: 1, 11: 2, 59: 1, 83: 1, 87: 2, 103: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 80, 124], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {0: 1, 4: 1, 80: 1, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 21, 25], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 1, 5: 4, 21: 1, 25: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 26, 34, 86, 94, 110], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 12, 'token_per_expert': {2: 2, 6: 1, 14: 1, 26: 2, 34: 2, 86: 2, 94: 1, 110: 1}}
INFO 05-06 10:50:16.793987.793987 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.331ms | allocate_experts_across_cpu_gpu: 0.103ms
INFO 05-06 10:50:16.793103.793103 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 10:50:16.794020.794020 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014445781707763672 seconds
INFO 05-06 10:50:16.796193.796193 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013568401336669922 seconds
INFO 05-06 10:50:16.797209.797209 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015778541564941406 seconds
INFO 05-06 10:50:16.800941.800941 mlpmodule.py:2799] [fused_experts] gmm total=1.784ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.800467.800467 mlpmodule.py:2799] [fused_experts] gmm total=1.989ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.800321.800321 mlpmodule.py:2799] [fused_experts] gmm total=2.379ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.801896.801896 mlpmodule.py:2799] [fused_experts] gmm total=3.055ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.802126.802126 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004283428192138672 seconds
INFO 05-06 10:50:16.802382.802382 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:50:16.802136.802136 cuda_h.py:27] end *layer_moe_fused cost 10.027 ms
DEBUG 05-06 10:50:16.803971.803971 cuda_h.py:27] end decode_layer cost 13.065 ms
DEBUG 05-06 10:50:16.803099.803099 lmp.py:904] ---- decode step 0 layer 22 ----
DEBUG 05-06 10:50:16.804747.804747 cuda_h.py:27] end *sagl cost 1.526 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 43, 119, 123, 127], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 1, 7: 1, 43: 2, 119: 3, 123: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 24, 76, 108, 120], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {0: 1, 4: 1, 8: 1, 16: 1, 24: 2, 76: 1, 108: 2, 120: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 61, 101, 109], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {1: 1, 5: 1, 61: 1, 101: 1, 109: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 26, 38, 94], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 1, 6: 1, 26: 1, 38: 2, 94: 2}}
INFO 05-06 10:50:16.805701.805701 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.296ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:50:16.806101.806101 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9550323486328125e-05 seconds
INFO 05-06 10:50:16.807340.807340 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001371145248413086 seconds
INFO 05-06 10:50:16.808101.808101 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001310586929321289 seconds
INFO 05-06 10:50:16.810977.810977 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013546943664550781 seconds
INFO 05-06 10:50:16.812343.812343 mlpmodule.py:2799] [fused_experts] gmm total=2.022ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.812088.812088 mlpmodule.py:2799] [fused_experts] gmm total=2.216ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.812710.812710 mlpmodule.py:2799] [fused_experts] gmm total=2.267ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.813590.813590 mlpmodule.py:2799] [fused_experts] gmm total=2.908ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.814726.814726 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0043108463287353516 seconds
INFO 05-06 10:50:16.814196.814196 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 8.273124694824219e-05 seconds
DEBUG 05-06 10:50:16.815925.815925 cuda_h.py:27] end *layer_moe_fused cost 9.851 ms
DEBUG 05-06 10:50:16.816945.816945 cuda_h.py:27] end decode_layer cost 13.004 ms
DEBUG 05-06 10:50:16.816326.816326 lmp.py:904] ---- decode step 0 layer 23 ----
DEBUG 05-06 10:50:16.819255.819255 cuda_h.py:27] end *sagl cost 3.133 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 47, 67], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {3: 1, 7: 1, 47: 1, 67: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 32, 84, 108], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {0: 1, 4: 1, 8: 1, 12: 2, 32: 2, 84: 2, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 17, 81, 97, 109], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 1, 5: 1, 17: 2, 81: 1, 97: 3, 109: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22, 86, 118], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 1, 6: 1, 22: 1, 86: 3, 118: 2}}
INFO 05-06 10:50:16.821944.821944 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.451ms | allocate_experts_across_cpu_gpu: 0.146ms
INFO 05-06 10:50:16.821147.821147 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.5272369384765625e-05 seconds
INFO 05-06 10:50:16.823404.823404 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015032291412353516 seconds
INFO 05-06 10:50:16.825629.825629 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001857757568359375 seconds
INFO 05-06 10:50:16.827222.827222 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015974044799804688 seconds
INFO 05-06 10:50:16.829397.829397 mlpmodule.py:2799] [fused_experts] gmm total=2.077ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.829272.829272 mlpmodule.py:2799] [fused_experts] gmm total=2.109ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.830129.830129 mlpmodule.py:2799] [fused_experts] gmm total=2.485ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.830852.830852 mlpmodule.py:2799] [fused_experts] gmm total=2.914ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.831690.831690 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004336357116699219 seconds
INFO 05-06 10:50:16.831741.831741 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.507469177246094e-05 seconds
DEBUG 05-06 10:50:16.832093.832093 cuda_h.py:27] end *layer_moe_fused cost 10.963 ms
DEBUG 05-06 10:50:16.832870.832870 cuda_h.py:27] end decode_layer cost 16.233 ms
DEBUG 05-06 10:50:16.832581.832581 lmp.py:904] ---- decode step 0 layer 24 ----
DEBUG 05-06 10:50:16.834570.834570 cuda_h.py:27] end *sagl cost 1.805 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 63, 79, 123], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {3: 1, 7: 1, 63: 1, 79: 1, 123: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 40, 44], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {0: 1, 4: 1, 12: 2, 40: 1, 44: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 33, 65, 109, 113], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 1, 5: 1, 33: 2, 65: 1, 109: 2, 113: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 30, 66, 90, 110, 118], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 6: 1, 10: 1, 30: 2, 66: 2, 90: 1, 110: 2, 118: 1}}
INFO 05-06 10:50:16.836210.836210 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.388ms | allocate_experts_across_cpu_gpu: 0.122ms
INFO 05-06 10:50:16.836724.836724 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1696090698242188e-05 seconds
INFO 05-06 10:50:16.837787.837787 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001440286636352539 seconds
INFO 05-06 10:50:16.839205.839205 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001529693603515625 seconds
INFO 05-06 10:50:16.840623.840623 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013575553894042969 seconds
INFO 05-06 10:50:16.842108.842108 mlpmodule.py:2799] [fused_experts] gmm total=1.802ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.842206.842206 mlpmodule.py:2799] [fused_experts] gmm total=2.064ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.843094.843094 mlpmodule.py:2799] [fused_experts] gmm total=2.220ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.843052.843052 mlpmodule.py:2799] [fused_experts] gmm total=2.848ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.844488.844488 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004183530807495117 seconds
INFO 05-06 10:50:16.845002.845002 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 10:50:16.845565.845565 cuda_h.py:27] end *layer_moe_fused cost 9.730 ms
DEBUG 05-06 10:50:16.845431.845431 cuda_h.py:27] end decode_layer cost 13.122 ms
DEBUG 05-06 10:50:16.845459.845459 lmp.py:904] ---- decode step 0 layer 25 ----
DEBUG 05-06 10:50:16.847817.847817 cuda_h.py:27] end *sagl cost 1.560 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 47, 59, 67, 95], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 1, 19: 1, 47: 1, 59: 1, 67: 2, 95: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 16, 44, 68, 104], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 2, 4: 1, 16: 1, 44: 2, 68: 1, 104: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 41, 45, 93, 97, 117, 121], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 10, 'token_per_expert': {1: 1, 5: 1, 29: 1, 41: 1, 45: 1, 93: 2, 97: 1, 117: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 58], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {2: 1, 6: 1, 58: 3}}
INFO 05-06 10:50:16.848471.848471 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.101ms
INFO 05-06 10:50:16.848030.848030 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 10:50:16.850318.850318 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001439809799194336 seconds
INFO 05-06 10:50:16.851789.851789 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0013344287872314453 seconds
INFO 05-06 10:50:16.853212.853212 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012857913970947266 seconds
INFO 05-06 10:50:16.855289.855289 mlpmodule.py:2799] [fused_experts] gmm total=2.129ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.857008.857008 mlpmodule.py:2799] [fused_experts] gmm total=3.918ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.857807.857807 mlpmodule.py:2799] [fused_experts] gmm total=3.879ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.857586.857586 mlpmodule.py:2799] [fused_experts] gmm total=4.083ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.859251.859251 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005968332290649414 seconds
INFO 05-06 10:50:16.859706.859706 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.125999450683594e-05 seconds
DEBUG 05-06 10:50:16.859097.859097 cuda_h.py:27] end *layer_moe_fused cost 11.200 ms
DEBUG 05-06 10:50:16.860320.860320 cuda_h.py:27] end decode_layer cost 14.216 ms
DEBUG 05-06 10:50:16.860778.860778 lmp.py:904] ---- decode step 0 layer 26 ----
DEBUG 05-06 10:50:16.861847.861847 cuda_h.py:27] end *sagl cost 1.625 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 43, 79, 119], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 1, 7: 1, 19: 2, 23: 1, 43: 2, 79: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 20, 52, 84], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {0: 1, 4: 3, 8: 1, 20: 1, 52: 1, 84: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 49, 65, 85], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 1, 5: 1, 49: 2, 65: 2, 85: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 38, 70, 90], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 1, 6: 1, 10: 1, 38: 2, 70: 1, 90: 2}}
INFO 05-06 10:50:16.863114.863114 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.358ms | allocate_experts_across_cpu_gpu: 0.110ms
INFO 05-06 10:50:16.863613.863613 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.09808349609375e-05 seconds
INFO 05-06 10:50:16.865930.865930 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0020847320556640625 seconds
INFO 05-06 10:50:16.866953.866953 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0014209747314453125 seconds
INFO 05-06 10:50:16.868005.868005 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014772415161132812 seconds
INFO 05-06 10:50:16.870610.870610 mlpmodule.py:2799] [fused_experts] gmm total=1.848ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.870330.870330 mlpmodule.py:2799] [fused_experts] gmm total=2.090ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.871995.871995 mlpmodule.py:2799] [fused_experts] gmm total=2.191ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.871643.871643 mlpmodule.py:2799] [fused_experts] gmm total=2.857ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.872920.872920 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00411534309387207 seconds
INFO 05-06 10:50:16.872315.872315 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 10:50:16.872732.872732 cuda_h.py:27] end *layer_moe_fused cost 10.232 ms
DEBUG 05-06 10:50:16.873563.873563 cuda_h.py:27] end decode_layer cost 13.353 ms
DEBUG 05-06 10:50:16.873069.873069 lmp.py:904] ---- decode step 0 layer 27 ----
DEBUG 05-06 10:50:16.875235.875235 cuda_h.py:27] end *sagl cost 1.558 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 87, 103, 115], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {3: 1, 7: 1, 27: 1, 87: 1, 103: 2, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 32, 48, 108], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 32: 1, 48: 1, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 29, 41, 53, 61, 85, 97, 121], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 14, 'token_per_expert': {1: 2, 5: 1, 29: 2, 41: 2, 53: 1, 61: 1, 85: 2, 97: 1, 121: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 58, 62, 82], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {2: 1, 6: 1, 58: 2, 62: 1, 82: 1}}
INFO 05-06 10:50:16.876431.876431 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.314ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:50:16.876731.876731 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 10:50:16.877601.877601 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001413583755493164 seconds
INFO 05-06 10:50:16.879534.879534 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012896060943603516 seconds
INFO 05-06 10:50:16.880319.880319 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014271736145019531 seconds
INFO 05-06 10:50:16.882254.882254 mlpmodule.py:2799] [fused_experts] gmm total=1.974ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.883539.883539 mlpmodule.py:2799] [fused_experts] gmm total=2.076ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.883547.883547 mlpmodule.py:2799] [fused_experts] gmm total=2.251ms E=32 S=14 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.884979.884979 mlpmodule.py:2799] [fused_experts] gmm total=3.016ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.885011.885011 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004372835159301758 seconds
INFO 05-06 10:50:16.885810.885810 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:50:16.885820.885820 cuda_h.py:27] end *layer_moe_fused cost 9.867 ms
DEBUG 05-06 10:50:16.886178.886178 cuda_h.py:27] end decode_layer cost 12.750 ms
DEBUG 05-06 10:50:16.886690.886690 lmp.py:904] ---- decode step 0 layer 28 ----
DEBUG 05-06 10:50:16.887397.887397 cuda_h.py:27] end *sagl cost 1.500 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39, 67, 115, 119], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 13, 'token_per_expert': {3: 1, 7: 1, 19: 2, 39: 2, 67: 2, 115: 3, 119: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 104, 108], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 104: 2, 108: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 49, 53, 57, 65, 89, 105, 113], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 11, 'token_per_expert': {1: 1, 5: 1, 13: 1, 49: 2, 53: 1, 57: 1, 65: 1, 89: 1, 105: 1, 113: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 22], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {2: 1, 6: 1, 22: 1}}
INFO 05-06 10:50:16.889273.889273 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.309ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:50:16.889620.889620 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 10:50:16.890636.890636 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014164447784423828 seconds
INFO 05-06 10:50:16.892675.892675 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001299142837524414 seconds
INFO 05-06 10:50:16.893539.893539 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001416921615600586 seconds
INFO 05-06 10:50:16.895167.895167 mlpmodule.py:2799] [fused_experts] gmm total=2.038ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.896313.896313 mlpmodule.py:2799] [fused_experts] gmm total=2.306ms E=32 S=13 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.896936.896936 mlpmodule.py:2799] [fused_experts] gmm total=2.330ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.896450.896450 mlpmodule.py:2799] [fused_experts] gmm total=2.366ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.897033.897033 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004415035247802734 seconds
INFO 05-06 10:50:16.898335.898335 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.887580871582031e-05 seconds
DEBUG 05-06 10:50:16.898581.898581 cuda_h.py:27] end *layer_moe_fused cost 9.591 ms
DEBUG 05-06 10:50:16.898877.898877 cuda_h.py:27] end decode_layer cost 12.491 ms
DEBUG 05-06 10:50:16.898667.898667 lmp.py:904] ---- decode step 0 layer 29 ----
DEBUG 05-06 10:50:16.900567.900567 cuda_h.py:27] end *sagl cost 1.538 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {3: 1, 7: 1, 23: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 52, 56, 60, 64], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {0: 1, 4: 2, 52: 2, 56: 1, 60: 2, 64: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 49, 73, 81, 97], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 1, 49: 1, 73: 2, 81: 2, 97: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 26, 30, 66, 78], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 1, 18: 1, 26: 2, 30: 1, 66: 1, 78: 1}}
INFO 05-06 10:50:16.901345.901345 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.312ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 10:50:16.901268.901268 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.811981201171875e-05 seconds
INFO 05-06 10:50:16.903131.903131 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014071464538574219 seconds
INFO 05-06 10:50:16.904388.904388 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0012826919555664062 seconds
INFO 05-06 10:50:16.906509.906509 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015537738800048828 seconds
INFO 05-06 10:50:16.908964.908964 mlpmodule.py:2799] [fused_experts] gmm total=1.932ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.908049.908049 mlpmodule.py:2799] [fused_experts] gmm total=2.093ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.909393.909393 mlpmodule.py:2799] [fused_experts] gmm total=2.387ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.909795.909795 mlpmodule.py:2799] [fused_experts] gmm total=2.720ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:16.910257.910257 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004199504852294922 seconds
INFO 05-06 10:50:16.910321.910321 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.9114227294921875e-05 seconds
DEBUG 05-06 10:50:16.910858.910858 cuda_h.py:27] end *layer_moe_fused cost 9.602 ms
DEBUG 05-06 10:50:16.911336.911336 cuda_h.py:27] end decode_layer cost 12.460 ms
DEBUG 05-06 10:50:16.911794.911794 cuda_h.py:27] end decode_step cost 509.936 ms
INFO 05-06 10:50:16.911213.911213 lmp.py:931] decode step 0 time: 0.5099658966064453 seconds
WARNING 05-06 10:50:16.911664.911664 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:50:16.911019.911019 helper.py:35]   NaN count (hidden): 2816
WARNING 05-06 10:50:16.912228.912228 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:50:16.912708.912708 helper.py:39]   NaN count (normed): 2816
WARNING 05-06 10:50:16.917951.917951 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:50:16.917737.917737 helper.py:50]   NaN count: 262144
WARNING 05-06 10:50:16.917513.917513 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 10:50:16.917819.917819 helper.py:80] WARNING: Logits have extreme values: min=-896.00, max=1032.00
WARNING 05-06 10:50:16.917842.917842 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 10:50:16.919518.919518 cuda_h.py:27] end init_inputs_tokens cost 8.034 ms
DEBUG 05-06 10:50:16.919520.919520 lmp.py:899] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:50:16.919243.919243 lmp.py:904] ---- decode step 1 layer 0 ----
DEBUG 05-06 10:50:16.921628.921628 cuda_h.py:27] end *sagl cost 1.541 ms
DEBUG 05-06 10:50:16.925138.925138 cuda_h.py:27] end *layer_moe_fused cost 3.083 ms
DEBUG 05-06 10:50:16.925789.925789 cuda_h.py:27] end decode_layer cost 6.013 ms
DEBUG 05-06 10:50:16.925680.925680 lmp.py:904] ---- decode step 1 layer 1 ----
DEBUG 05-06 10:50:16.927700.927700 cuda_h.py:27] end *sagl cost 1.922 ms
DEBUG 05-06 10:50:16.931899.931899 cuda_h.py:27] end *layer_moe_fused cost 2.695 ms
DEBUG 05-06 10:50:16.932525.932525 cuda_h.py:27] end decode_layer cost 6.235 ms
DEBUG 05-06 10:50:16.932447.932447 lmp.py:904] ---- decode step 1 layer 2 ----
DEBUG 05-06 10:50:16.934034.934034 cuda_h.py:27] end *sagl cost 1.849 ms
DEBUG 05-06 10:50:16.937364.937364 cuda_h.py:27] end *layer_moe_fused cost 2.470 ms
DEBUG 05-06 10:50:16.938532.938532 cuda_h.py:27] end decode_layer cost 5.886 ms
DEBUG 05-06 10:50:16.938766.938766 lmp.py:904] ---- decode step 1 layer 3 ----
DEBUG 05-06 10:50:16.940302.940302 cuda_h.py:27] end *sagl cost 1.920 ms
DEBUG 05-06 10:50:16.943084.943084 cuda_h.py:27] end *layer_moe_fused cost 2.113 ms
DEBUG 05-06 10:50:16.943992.943992 cuda_h.py:27] end decode_layer cost 5.743 ms
DEBUG 05-06 10:50:16.944179.944179 lmp.py:904] ---- decode step 1 layer 4 ----
DEBUG 05-06 10:50:16.945310.945310 cuda_h.py:27] end *sagl cost 1.856 ms
DEBUG 05-06 10:50:16.949368.949368 cuda_h.py:27] end *layer_moe_fused cost 2.265 ms
DEBUG 05-06 10:50:16.949550.949550 cuda_h.py:27] end decode_layer cost 5.710 ms
DEBUG 05-06 10:50:16.949406.949406 lmp.py:904] ---- decode step 1 layer 5 ----
DEBUG 05-06 10:50:16.951459.951459 cuda_h.py:27] end *sagl cost 1.914 ms
DEBUG 05-06 10:50:16.955210.955210 cuda_h.py:27] end *layer_moe_fused cost 2.191 ms
DEBUG 05-06 10:50:16.955909.955909 cuda_h.py:27] end decode_layer cost 5.702 ms
DEBUG 05-06 10:50:16.955077.955077 lmp.py:904] ---- decode step 1 layer 6 ----
DEBUG 05-06 10:50:16.957261.957261 cuda_h.py:27] end *sagl cost 1.868 ms
DEBUG 05-06 10:50:16.960664.960664 cuda_h.py:27] end *layer_moe_fused cost 2.098 ms
DEBUG 05-06 10:50:16.961693.961693 cuda_h.py:27] end decode_layer cost 5.540 ms
DEBUG 05-06 10:50:16.961166.961166 lmp.py:904] ---- decode step 1 layer 7 ----
DEBUG 05-06 10:50:16.963071.963071 cuda_h.py:27] end *sagl cost 1.876 ms
DEBUG 05-06 10:50:16.966143.966143 cuda_h.py:27] end *layer_moe_fused cost 1.845 ms
DEBUG 05-06 10:50:16.966166.966166 cuda_h.py:27] end decode_layer cost 5.332 ms
DEBUG 05-06 10:50:16.966684.966684 lmp.py:904] ---- decode step 1 layer 8 ----
DEBUG 05-06 10:50:16.968902.968902 cuda_h.py:27] end *sagl cost 1.896 ms
DEBUG 05-06 10:50:16.971201.971201 cuda_h.py:27] end *layer_moe_fused cost 1.562 ms
DEBUG 05-06 10:50:16.971748.971748 cuda_h.py:27] end decode_layer cost 5.062 ms
DEBUG 05-06 10:50:16.971028.971028 lmp.py:904] ---- decode step 1 layer 9 ----
DEBUG 05-06 10:50:16.973511.973511 cuda_h.py:27] end *sagl cost 1.880 ms
DEBUG 05-06 10:50:16.976988.976988 cuda_h.py:27] end *layer_moe_fused cost 1.715 ms
DEBUG 05-06 10:50:16.976056.976056 cuda_h.py:27] end decode_layer cost 5.176 ms
DEBUG 05-06 10:50:16.977337.977337 lmp.py:904] ---- decode step 1 layer 10 ----
DEBUG 05-06 10:50:16.978036.978036 cuda_h.py:27] end *sagl cost 1.829 ms
DEBUG 05-06 10:50:16.981376.981376 cuda_h.py:27] end *layer_moe_fused cost 1.574 ms
DEBUG 05-06 10:50:16.982921.982921 cuda_h.py:27] end decode_layer cost 4.990 ms
DEBUG 05-06 10:50:16.982208.982208 lmp.py:904] ---- decode step 1 layer 11 ----
DEBUG 05-06 10:50:16.984618.984618 cuda_h.py:27] end *sagl cost 1.897 ms
DEBUG 05-06 10:50:16.986137.986137 cuda_h.py:27] end *layer_moe_fused cost 1.567 ms
DEBUG 05-06 10:50:16.987921.987921 cuda_h.py:27] end decode_layer cost 5.049 ms
DEBUG 05-06 10:50:16.987784.987784 lmp.py:904] ---- decode step 1 layer 12 ----
DEBUG 05-06 10:50:16.989822.989822 cuda_h.py:27] end *sagl cost 1.868 ms
DEBUG 05-06 10:50:16.991696.991696 cuda_h.py:27] end *layer_moe_fused cost 1.544 ms
DEBUG 05-06 10:50:16.992149.992149 cuda_h.py:27] end decode_layer cost 4.969 ms
DEBUG 05-06 10:50:16.992761.992761 lmp.py:904] ---- decode step 1 layer 13 ----
DEBUG 05-06 10:50:16.994527.994527 cuda_h.py:27] end *sagl cost 1.880 ms
DEBUG 05-06 10:50:16.996947.996947 cuda_h.py:27] end *layer_moe_fused cost 1.571 ms
DEBUG 05-06 10:50:16.997446.997446 cuda_h.py:27] end decode_layer cost 5.029 ms
DEBUG 05-06 10:50:16.997488.997488 lmp.py:904] ---- decode step 1 layer 14 ----
DEBUG 05-06 10:50:16.999147.999147 cuda_h.py:27] end *sagl cost 1.835 ms
DEBUG 05-06 10:50:17.001037.001037 cuda_h.py:27] end *layer_moe_fused cost 1.557 ms
DEBUG 05-06 10:50:17.002742.002742 cuda_h.py:27] end decode_layer cost 5.027 ms
DEBUG 05-06 10:50:17.002784.002784 lmp.py:904] ---- decode step 1 layer 15 ----
DEBUG 05-06 10:50:17.004419.004419 cuda_h.py:27] end *sagl cost 1.887 ms
DEBUG 05-06 10:50:17.007195.007195 cuda_h.py:27] end *layer_moe_fused cost 1.569 ms
DEBUG 05-06 10:50:17.007079.007079 cuda_h.py:27] end decode_layer cost 5.022 ms
DEBUG 05-06 10:50:17.007882.007882 lmp.py:904] ---- decode step 1 layer 16 ----
DEBUG 05-06 10:50:17.009186.009186 cuda_h.py:27] end *sagl cost 1.842 ms
DEBUG 05-06 10:50:17.012591.012591 cuda_h.py:27] end *layer_moe_fused cost 1.583 ms
DEBUG 05-06 10:50:17.012428.012428 cuda_h.py:27] end decode_layer cost 5.032 ms
DEBUG 05-06 10:50:17.012709.012709 lmp.py:904] ---- decode step 1 layer 17 ----
DEBUG 05-06 10:50:17.014759.014759 cuda_h.py:27] end *sagl cost 1.834 ms
DEBUG 05-06 10:50:17.017483.017483 cuda_h.py:27] end *layer_moe_fused cost 1.542 ms
DEBUG 05-06 10:50:17.017837.017837 cuda_h.py:27] end decode_layer cost 5.005 ms
DEBUG 05-06 10:50:17.017163.017163 lmp.py:904] ---- decode step 1 layer 18 ----
DEBUG 05-06 10:50:17.019466.019466 cuda_h.py:27] end *sagl cost 1.854 ms
DEBUG 05-06 10:50:17.022897.022897 cuda_h.py:27] end *layer_moe_fused cost 1.571 ms
DEBUG 05-06 10:50:17.022058.022058 cuda_h.py:27] end decode_layer cost 4.973 ms
DEBUG 05-06 10:50:17.022100.022100 lmp.py:904] ---- decode step 1 layer 19 ----
DEBUG 05-06 10:50:17.024597.024597 cuda_h.py:27] end *sagl cost 1.927 ms
DEBUG 05-06 10:50:17.027048.027048 cuda_h.py:27] end *layer_moe_fused cost 1.524 ms
DEBUG 05-06 10:50:17.027197.027197 cuda_h.py:27] end decode_layer cost 5.051 ms
DEBUG 05-06 10:50:17.027100.027100 lmp.py:904] ---- decode step 1 layer 20 ----
DEBUG 05-06 10:50:17.029315.029315 cuda_h.py:27] end *sagl cost 1.823 ms
DEBUG 05-06 10:50:17.032268.032268 cuda_h.py:27] end *layer_moe_fused cost 1.536 ms
DEBUG 05-06 10:50:17.032337.032337 cuda_h.py:27] end decode_layer cost 4.910 ms
DEBUG 05-06 10:50:17.032141.032141 lmp.py:904] ---- decode step 1 layer 21 ----
DEBUG 05-06 10:50:17.034079.034079 cuda_h.py:27] end *sagl cost 1.866 ms
DEBUG 05-06 10:50:17.037498.037498 cuda_h.py:27] end *layer_moe_fused cost 1.551 ms
DEBUG 05-06 10:50:17.037421.037421 cuda_h.py:27] end decode_layer cost 5.012 ms
DEBUG 05-06 10:50:17.037179.037179 lmp.py:904] ---- decode step 1 layer 22 ----
DEBUG 05-06 10:50:17.039177.039177 cuda_h.py:27] end *sagl cost 1.874 ms
DEBUG 05-06 10:50:17.042084.042084 cuda_h.py:27] end *layer_moe_fused cost 1.535 ms
DEBUG 05-06 10:50:17.042630.042630 cuda_h.py:27] end decode_layer cost 4.962 ms
DEBUG 05-06 10:50:17.043910.043910 lmp.py:904] ---- decode step 1 layer 23 ----
DEBUG 05-06 10:50:17.044238.044238 cuda_h.py:27] end *sagl cost 1.837 ms
DEBUG 05-06 10:50:17.047994.047994 cuda_h.py:27] end *layer_moe_fused cost 1.562 ms
DEBUG 05-06 10:50:17.047891.047891 cuda_h.py:27] end decode_layer cost 4.967 ms
DEBUG 05-06 10:50:17.048410.048410 lmp.py:904] ---- decode step 1 layer 24 ----
DEBUG 05-06 10:50:17.049327.049327 cuda_h.py:27] end *sagl cost 1.815 ms
DEBUG 05-06 10:50:17.052937.052937 cuda_h.py:27] end *layer_moe_fused cost 1.565 ms
DEBUG 05-06 10:50:17.052767.052767 cuda_h.py:27] end decode_layer cost 4.928 ms
DEBUG 05-06 10:50:17.053286.053286 lmp.py:904] ---- decode step 1 layer 25 ----
DEBUG 05-06 10:50:17.054562.054562 cuda_h.py:27] end *sagl cost 1.869 ms
DEBUG 05-06 10:50:17.057698.057698 cuda_h.py:27] end *layer_moe_fused cost 1.549 ms
DEBUG 05-06 10:50:17.058051.058051 cuda_h.py:27] end decode_layer cost 5.053 ms
DEBUG 05-06 10:50:17.058093.058093 lmp.py:904] ---- decode step 1 layer 26 ----
DEBUG 05-06 10:50:17.060396.060396 cuda_h.py:27] end *sagl cost 1.854 ms
DEBUG 05-06 10:50:17.062826.062826 cuda_h.py:27] end *layer_moe_fused cost 1.539 ms
DEBUG 05-06 10:50:17.063564.063564 cuda_h.py:27] end decode_layer cost 4.941 ms
DEBUG 05-06 10:50:17.063129.063129 lmp.py:904] ---- decode step 1 layer 27 ----
DEBUG 05-06 10:50:17.065552.065552 cuda_h.py:27] end *sagl cost 1.908 ms
DEBUG 05-06 10:50:17.067309.067309 cuda_h.py:27] end *layer_moe_fused cost 1.573 ms
DEBUG 05-06 10:50:17.068808.068808 cuda_h.py:27] end decode_layer cost 5.060 ms
DEBUG 05-06 10:50:17.068850.068850 lmp.py:904] ---- decode step 1 layer 28 ----
DEBUG 05-06 10:50:17.070542.070542 cuda_h.py:27] end *sagl cost 1.825 ms
DEBUG 05-06 10:50:17.072664.072664 cuda_h.py:27] end *layer_moe_fused cost 1.621 ms
DEBUG 05-06 10:50:17.073971.073971 cuda_h.py:27] end decode_layer cost 4.999 ms
DEBUG 05-06 10:50:17.073490.073490 lmp.py:904] ---- decode step 1 layer 29 ----
DEBUG 05-06 10:50:17.075944.075944 cuda_h.py:27] end *sagl cost 1.824 ms
DEBUG 05-06 10:50:17.077212.077212 cuda_h.py:27] end *layer_moe_fused cost 1.567 ms
DEBUG 05-06 10:50:17.078803.078803 cuda_h.py:27] end decode_layer cost 5.001 ms
DEBUG 05-06 10:50:17.078283.078283 cuda_h.py:27] end decode_step cost 166.960 ms
INFO 05-06 10:50:17.078953.078953 lmp.py:931] decode step 1 time: 0.16699862480163574 seconds
Time taken: 6.79152250289917 seconds
generate input ids cost 0.08188295364379883 s
DEBUG 05-06 10:50:19.842681.842681 cuda_h.py:27] end generate_input_ids cost 2611.281 ms
DEBUG 05-06 10:50:19.842428.842428 cuda_h.py:27] end init_cache cost 0.046 ms
INFO 05-06 10:50:19.854639.854639 lmp.py:2341] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6384492484, 'cuda:1': 12831555584, 'cuda:2': 12808486912, 'cuda:3': 12839944192} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7347552450583675, 'cuda:1': 0.4708756492646622, 'cuda:2': 0.4713240027915884, 'cuda:3': 0.47071282303423906}
INFO 05-06 10:50:19.854728.854728 lmp.py:2359] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.854266.854266 lmp.py:2359] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.854413.854413 lmp.py:2359] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.854226.854226 lmp.py:2359] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.855069.855069 lmp.py:2359] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.855963.855963 lmp.py:2359] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.855128.855128 lmp.py:2359] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.855672.855672 lmp.py:2359] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.855705.855705 lmp.py:2359] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.856804.856804 lmp.py:2359] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.856916.856916 lmp.py:2359] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.856348.856348 lmp.py:2359] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.856560.856560 lmp.py:2359] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.856514.856514 lmp.py:2359] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.856778.856778 lmp.py:2359] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.856448.856448 lmp.py:2359] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.856705.856705 lmp.py:2359] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.856945.856945 lmp.py:2359] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857493.857493 lmp.py:2359] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857971.857971 lmp.py:2359] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857235.857235 lmp.py:2359] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857713.857713 lmp.py:2359] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857160.857160 lmp.py:2359] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857877.857877 lmp.py:2359] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857748.857748 lmp.py:2359] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857525.857525 lmp.py:2359] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857053.857053 lmp.py:2359] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.857100.857100 lmp.py:2359] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.858742.858742 lmp.py:2359] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 10:50:19.858075.858075 lmp.py:2359] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 10:50:20.140239.140239 cuda_h.py:27] end init_loading_placement cost 297.901 ms
DEBUG 05-06 10:50:20.140601.140601 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:50:20.140166.140166 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:50:20 client.py:72] load_into_gpu: gemma4-26B-A4B, a09b60d1-e259-4aaa-9f85-e6979a549556
INFO 05-06 10:50:20 client.py:135] Model loaded: gemma4-26B-A4B, a09b60d1-e259-4aaa-9f85-e6979a549556
INFO 05-06 10:50:20 client.py:204] confirm_model_loaded: gemma4-26B-A4B, a09b60d1-e259-4aaa-9f85-e6979a549556
INFO 05-06 10:50:20 client.py:212] Model loaded
DEBUG 05-06 10:50:20.666163.666163 cuda_h.py:27] end init_general_sagl_loading_async cost 526.065 ms
INFO 05-06 10:50:20.714452.714452 lmp.py:2862] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 10:50:20.814354.814354 cuda_h.py:27] end restore_state_dict cost 99.559 ms
DEBUG 05-06 10:50:20.814772.814772 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 10:50:20.814343.814343 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 10:50:20 client.py:72] load_into_gpu: gemma4-26B-A4B, 7717975f-4f86-46d5-8061-0c0b419d76be
INFO 05-06 10:50:20 client.py:135] Model loaded: gemma4-26B-A4B, 7717975f-4f86-46d5-8061-0c0b419d76be
DEBUG 05-06 10:50:20.941716.941716 cuda_h.py:27] end init_experts_loading_async cost 126.698 ms
DEBUG 05-06 10:50:20.943373.943373 cuda_h.py:27] end init_inputs_tokens cost 1.378 ms
DEBUG 05-06 10:50:20.943822.943822 lmp.py:824] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 10:50:20.949381.949381 cuda_h.py:27] end *sagl cost 6.293 ms
experts_cpu_alloc {'expert_ids': [19, 87, 27, 15, 119, 17, 85, 97, 109, 101, 29, 81, 30, 86, 6, 66, 88, 120, 12, 36, 100, 96, 8], 'token_total': 119, 'token_per_expert': {19: 2, 87: 2, 27: 10, 15: 13, 119: 13, 17: 3, 85: 3, 97: 3, 109: 3, 101: 6, 29: 7, 81: 11, 30: 1, 86: 2, 6: 4, 66: 6, 88: 1, 120: 1, 12: 2, 36: 4, 100: 5, 96: 8, 8: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 31, 39, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 91, 99, 103, 107, 111, 115, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 5050, 'token_per_expert': {3: 160, 7: 374, 11: 31, 23: 23, 31: 134, 39: 718, 47: 1304, 51: 186, 55: 208, 59: 43, 63: 15, 67: 183, 71: 65, 75: 89, 79: 76, 83: 105, 91: 458, 99: 161, 103: 432, 107: 25, 111: 23, 115: 89, 123: 39, 127: 109}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 21, 25, 33, 37, 41, 45, 49, 53, 65, 69, 73, 77, 89, 93, 105, 113, 117, 121, 125], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 3741, 'token_per_expert': {1: 273, 5: 66, 9: 68, 13: 61, 21: 171, 25: 110, 33: 828, 37: 81, 41: 142, 45: 14, 49: 17, 53: 819, 65: 39, 69: 78, 73: 60, 77: 99, 89: 133, 93: 12, 105: 89, 113: 157, 117: 97, 121: 226, 125: 101}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 10, 14, 18, 22, 26, 34, 38, 46, 50, 54, 70, 74, 78, 90, 94, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 27, 'token_total': 4047, 'token_per_expert': {2: 30, 10: 36, 14: 29, 18: 71, 22: 255, 26: 304, 34: 38, 38: 59, 46: 450, 50: 520, 54: 275, 70: 140, 74: 224, 78: 109, 90: 546, 94: 24, 102: 74, 106: 14, 110: 83, 114: 48, 118: 89, 122: 114, 126: 515}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 16, 20, 24, 28, 32, 44, 48, 52, 60, 64, 68, 72, 76, 80, 84, 92, 104, 108, 112, 116, 124], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 3427, 'token_per_expert': {0: 249, 4: 11, 16: 201, 20: 15, 24: 81, 28: 123, 32: 183, 44: 17, 48: 146, 52: 150, 60: 55, 64: 106, 68: 694, 72: 100, 76: 74, 80: 28, 84: 21, 92: 87, 104: 134, 108: 78, 112: 68, 116: 82, 124: 724}}
INFO 05-06 10:50:20.955475.955475 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 1.636ms | allocate_experts_across_cpu_gpu: 0.850ms
INFO 05-06 10:50:20.955180.955180 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 0.00014853477478027344 seconds
INFO 05-06 10:50:20.957982.957982 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=186 time: 0.0015606880187988281 seconds
INFO 05-06 10:50:20.958776.958776 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006973743438720703 seconds
INFO 05-06 10:50:20.972991.972991 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0140380859375 seconds
INFO 05-06 10:50:21.072366.072366 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.09972453117370605 seconds
INFO 05-06 10:50:21.074283.074283 mlpmodule.py:2799] [fused_experts] gmm total=2.402ms E=32 S=5090 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.076614.076614 mlpmodule.py:2799] [fused_experts] gmm total=4.138ms E=32 S=4060 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.078460.078460 mlpmodule.py:2799] [fused_experts] gmm total=5.775ms E=32 S=3777 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.079532.079532 mlpmodule.py:2799] [fused_experts] gmm total=6.329ms E=32 S=3457 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.080582.080582 lmp.py:1484] [layer_moe_fused] experts compute time: 0.008267402648925781 seconds
INFO 05-06 10:50:21.080883.080883 lmp.py:1496] [layer_moe_fused] to time: 0.00010848045349121094 seconds
INFO 05-06 10:50:21.081282.081282 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00039196014404296875 seconds
DEBUG 05-06 10:50:21.082391.082391 cuda_h.py:27] end *layer_moe_fused cost 129.545 ms
DEBUG 05-06 10:50:21.100454.100454 cuda_h.py:27] end prefill_layer cost 157.143 ms
DEBUG 05-06 10:50:21.100695.100695 lmp.py:841] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 10:50:21.100497.100497 lmp.py:824] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 10:50:21.102464.102464 cuda_h.py:27] end *sagl cost 1.733 ms
experts_cpu_alloc {'expert_ids': [63, 43, 39, 107, 23, 61, 117, 33, 125, 41, 81, 70, 114, 2, 18, 24, 16, 44, 40, 112, 0, 116, 32], 'token_total': 219, 'token_per_expert': {63: 3, 43: 4, 39: 5, 107: 5, 23: 6, 61: 2, 117: 2, 33: 5, 125: 8, 41: 10, 81: 10, 70: 2, 114: 5, 2: 9, 18: 12, 24: 3, 16: 7, 44: 7, 40: 20, 112: 21, 0: 23, 116: 24, 32: 26}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 27, 31, 35, 47, 51, 55, 59, 67, 75, 79, 83, 87, 91, 95, 99, 103, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 2673, 'token_per_expert': {3: 62, 7: 155, 11: 49, 15: 21, 27: 33, 31: 14, 35: 64, 47: 156, 51: 235, 55: 15, 59: 152, 67: 449, 75: 7, 79: 74, 83: 35, 87: 20, 91: 14, 95: 62, 99: 552, 103: 22, 115: 8, 119: 149, 123: 37, 127: 288}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 21, 25, 29, 37, 45, 49, 53, 57, 65, 69, 73, 77, 85, 89, 93, 97, 101, 105, 109, 121], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4031, 'token_per_expert': {1: 144, 5: 409, 9: 102, 13: 1122, 21: 71, 25: 164, 29: 27, 37: 36, 45: 46, 49: 104, 53: 158, 57: 28, 65: 155, 69: 40, 73: 95, 77: 15, 85: 103, 89: 14, 93: 25, 97: 485, 101: 52, 105: 70, 109: 533, 121: 33}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 10, 14, 22, 26, 30, 34, 38, 42, 46, 50, 54, 62, 66, 74, 78, 82, 90, 94, 98, 106, 110, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 28, 'token_total': 5137, 'token_per_expert': {6: 35, 10: 623, 14: 15, 22: 467, 26: 17, 30: 967, 34: 64, 38: 40, 42: 161, 46: 197, 50: 49, 54: 230, 62: 23, 66: 29, 74: 63, 78: 29, 82: 791, 90: 61, 94: 119, 98: 62, 106: 213, 110: 24, 118: 291, 122: 567}}
experts_gpu_alloc_device_3 {'expert_ids': [4, 8, 12, 20, 28, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 120, 124], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 31, 'token_total': 4324, 'token_per_expert': {4: 82, 8: 471, 12: 218, 20: 207, 28: 189, 48: 39, 52: 1230, 56: 50, 60: 26, 64: 92, 68: 692, 72: 29, 76: 28, 80: 148, 84: 26, 88: 26, 92: 67, 96: 203, 100: 203, 104: 65, 108: 50, 120: 101, 124: 82}}
INFO 05-06 10:50:21.104317.104317 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.911ms | allocate_experts_across_cpu_gpu: 0.278ms
INFO 05-06 10:50:21.104341.104341 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.841255187988281e-05 seconds
INFO 05-06 10:50:21.106185.106185 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0016238689422607422 seconds
INFO 05-06 10:50:21.106317.106317 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006253719329833984 seconds
INFO 05-06 10:50:21.139080.139080 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.032021522521972656 seconds
INFO 05-06 10:50:21.140721.140721 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010495185852050781 seconds
INFO 05-06 10:50:21.142759.142759 mlpmodule.py:2799] [fused_experts] gmm total=1.882ms E=32 S=4068 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.142816.142816 mlpmodule.py:2799] [fused_experts] gmm total=2.045ms E=32 S=2696 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.142251.142251 mlpmodule.py:2799] [fused_experts] gmm total=2.472ms E=32 S=5165 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.143042.143042 mlpmodule.py:2799] [fused_experts] gmm total=3.316ms E=32 S=4455 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.145137.145137 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0050470829010009766 seconds
INFO 05-06 10:50:21.145267.145267 lmp.py:1496] [layer_moe_fused] to time: 4.935264587402344e-05 seconds
INFO 05-06 10:50:21.145612.145612 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002815723419189453 seconds
DEBUG 05-06 10:50:21.146294.146294 cuda_h.py:27] end *layer_moe_fused cost 43.027 ms
DEBUG 05-06 10:50:21.169493.169493 cuda_h.py:27] end prefill_layer cost 69.300 ms
DEBUG 05-06 10:50:21.169721.169721 lmp.py:841] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 10:50:21.169662.169662 lmp.py:824] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 10:50:21.171211.171211 cuda_h.py:27] end *sagl cost 1.534 ms
experts_cpu_alloc {'expert_ids': [47, 87, 79, 67, 99, 75, 23, 93, 117, 101, 25, 21, 45, 22, 74, 86, 32, 16, 68, 12, 64, 40, 120], 'token_total': 204, 'token_per_expert': {47: 1, 87: 1, 79: 4, 67: 6, 99: 6, 75: 7, 23: 28, 93: 1, 117: 1, 101: 2, 25: 8, 21: 10, 45: 19, 22: 1, 74: 1, 86: 5, 32: 3, 16: 4, 68: 6, 12: 18, 64: 18, 40: 21, 120: 33}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 35, 43, 51, 55, 59, 63, 71, 83, 91, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4750, 'token_per_expert': {3: 138, 7: 259, 11: 1058, 15: 372, 19: 583, 27: 46, 31: 99, 35: 28, 43: 75, 51: 104, 55: 221, 59: 445, 63: 92, 71: 68, 83: 96, 91: 148, 95: 30, 103: 67, 107: 79, 111: 46, 115: 39, 119: 81, 123: 104, 127: 472}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 29, 33, 37, 41, 49, 53, 57, 61, 65, 69, 77, 81, 85, 97, 105, 109, 113, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4580, 'token_per_expert': {1: 416, 5: 20, 9: 431, 13: 398, 17: 74, 29: 311, 33: 77, 37: 273, 41: 547, 49: 117, 53: 183, 57: 125, 61: 29, 65: 178, 69: 109, 77: 100, 81: 389, 85: 64, 97: 139, 105: 34, 109: 145, 113: 43, 121: 30, 125: 348}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 14, 18, 26, 34, 42, 46, 50, 54, 58, 62, 66, 70, 78, 82, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 27, 'token_total': 3285, 'token_per_expert': {6: 25, 14: 124, 18: 170, 26: 12, 34: 151, 42: 36, 46: 45, 50: 13, 54: 391, 58: 42, 62: 567, 66: 9, 70: 68, 78: 207, 82: 24, 90: 247, 98: 86, 102: 330, 106: 181, 110: 99, 114: 29, 118: 218, 122: 106, 126: 105}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 20, 24, 28, 36, 44, 48, 52, 56, 60, 72, 76, 80, 84, 88, 96, 100, 104, 108, 116, 124], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 3565, 'token_per_expert': {0: 52, 4: 81, 8: 117, 20: 216, 24: 68, 28: 55, 36: 92, 44: 99, 48: 232, 52: 52, 56: 62, 60: 215, 72: 50, 76: 269, 80: 233, 84: 228, 88: 46, 96: 34, 100: 71, 104: 151, 108: 983, 116: 72, 124: 87}}
INFO 05-06 10:50:21.174482.174482 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 1.372ms | allocate_experts_across_cpu_gpu: 0.260ms
INFO 05-06 10:50:21.174062.174062 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.173683166503906e-05 seconds
INFO 05-06 10:50:21.175446.175446 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001230478286743164 seconds
INFO 05-06 10:50:21.176289.176289 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005340576171875 seconds
INFO 05-06 10:50:21.205869.205869 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.028902292251586914 seconds
INFO 05-06 10:50:21.206805.206805 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009884834289550781 seconds
INFO 05-06 10:50:21.208350.208350 mlpmodule.py:2799] [fused_experts] gmm total=1.715ms E=32 S=3292 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.209959.209959 mlpmodule.py:2799] [fused_experts] gmm total=2.409ms E=32 S=4803 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.209912.209912 mlpmodule.py:2799] [fused_experts] gmm total=2.424ms E=32 S=3668 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.209716.209716 mlpmodule.py:2799] [fused_experts] gmm total=2.809ms E=32 S=4621 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.210712.210712 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00407719612121582 seconds
INFO 05-06 10:50:21.210968.210968 lmp.py:1496] [layer_moe_fused] to time: 4.792213439941406e-05 seconds
INFO 05-06 10:50:21.210087.210087 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002510547637939453 seconds
DEBUG 05-06 10:50:21.211206.211206 cuda_h.py:27] end *layer_moe_fused cost 39.069 ms
DEBUG 05-06 10:50:21.231136.231136 cuda_h.py:27] end prefill_layer cost 61.612 ms
DEBUG 05-06 10:50:21.231118.231118 lmp.py:841] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 10:50:21.231311.231311 lmp.py:824] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 10:50:21.233693.233693 cuda_h.py:27] end *sagl cost 1.572 ms
experts_cpu_alloc {'expert_ids': [99, 103, 35, 87, 45, 81, 113, 105, 21, 29, 125, 126, 38, 46, 106, 18, 94, 82, 12, 36, 20, 80, 32, 72], 'token_total': 154, 'token_per_expert': {99: 1, 103: 1, 35: 7, 87: 8, 45: 1, 81: 1, 113: 1, 105: 2, 21: 4, 29: 6, 125: 6, 126: 1, 38: 2, 46: 2, 106: 4, 18: 9, 94: 11, 82: 19, 12: 4, 36: 6, 20: 8, 80: 9, 32: 16, 72: 25}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 23, 27, 31, 39, 43, 51, 55, 59, 63, 67, 71, 75, 83, 91, 95, 107, 111, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 29, 'token_total': 2754, 'token_per_expert': {3: 109, 11: 130, 15: 184, 19: 108, 23: 17, 27: 24, 31: 64, 39: 87, 43: 84, 51: 162, 55: 18, 59: 97, 63: 87, 67: 50, 71: 328, 75: 395, 83: 180, 91: 20, 95: 195, 107: 110, 111: 62, 115: 8, 119: 104, 123: 83, 127: 48}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 25, 33, 37, 41, 53, 57, 61, 65, 69, 73, 77, 85, 89, 93, 97, 101, 109, 117, 121], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3846, 'token_per_expert': {1: 34, 5: 291, 9: 318, 13: 70, 17: 179, 25: 179, 33: 51, 37: 8, 41: 32, 53: 253, 57: 25, 61: 72, 65: 15, 69: 156, 73: 250, 77: 85, 85: 546, 89: 22, 93: 343, 97: 265, 101: 87, 109: 78, 117: 43, 121: 444}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 22, 26, 30, 34, 42, 50, 54, 58, 62, 66, 70, 74, 78, 86, 98, 102, 110, 114, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 5756, 'token_per_expert': {2: 114, 6: 81, 10: 167, 14: 314, 22: 541, 26: 89, 30: 41, 34: 265, 42: 29, 50: 687, 54: 197, 58: 105, 62: 430, 66: 341, 70: 146, 74: 160, 78: 697, 86: 34, 98: 19, 102: 424, 110: 109, 114: 127, 118: 250, 122: 389}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 16, 24, 28, 40, 44, 48, 52, 56, 60, 64, 68, 76, 84, 88, 92, 96, 100, 104, 108, 116, 120], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3874, 'token_per_expert': {0: 158, 4: 291, 8: 51, 16: 32, 24: 42, 28: 251, 40: 60, 44: 99, 48: 45, 52: 280, 56: 39, 60: 38, 64: 103, 68: 208, 76: 248, 84: 244, 88: 301, 92: 283, 96: 317, 100: 74, 104: 237, 108: 157, 116: 52, 120: 264}}
INFO 05-06 10:50:21.235019.235019 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 1.088ms | allocate_experts_across_cpu_gpu: 0.257ms
INFO 05-06 10:50:21.235785.235785 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.271766662597656e-05 seconds
INFO 05-06 10:50:21.236712.236712 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009741783142089844 seconds
INFO 05-06 10:50:21.237791.237791 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.000518798828125 seconds
INFO 05-06 10:50:21.270867.270867 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.03263688087463379 seconds
INFO 05-06 10:50:21.271263.271263 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010747909545898438 seconds
INFO 05-06 10:50:21.273561.273561 mlpmodule.py:2799] [fused_experts] gmm total=1.779ms E=32 S=2771 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.273740.273740 mlpmodule.py:2799] [fused_experts] gmm total=1.921ms E=32 S=3867 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.273929.273929 mlpmodule.py:2799] [fused_experts] gmm total=1.907ms E=32 S=3942 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.275900.275900 mlpmodule.py:2799] [fused_experts] gmm total=3.656ms E=32 S=5804 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.277924.277924 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005614280700683594 seconds
INFO 05-06 10:50:21.277511.277511 lmp.py:1496] [layer_moe_fused] to time: 4.8160552978515625e-05 seconds
INFO 05-06 10:50:21.277750.277750 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002694129943847656 seconds
DEBUG 05-06 10:50:21.277525.277525 cuda_h.py:27] end *layer_moe_fused cost 43.748 ms
DEBUG 05-06 10:50:21.300737.300737 cuda_h.py:27] end prefill_layer cost 68.665 ms
DEBUG 05-06 10:50:21.300302.300302 lmp.py:841] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 10:50:21.300912.300912 lmp.py:824] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 10:50:21.302053.302053 cuda_h.py:27] end *sagl cost 1.616 ms
experts_cpu_alloc {'expert_ids': [35, 95, 127, 79, 31, 13, 9, 65, 121, 41, 21, 69, 42, 70, 102, 14, 2, 110, 50, 100, 48, 16, 12, 72], 'token_total': 169, 'token_per_expert': {35: 1, 95: 4, 127: 8, 79: 9, 31: 12, 13: 1, 9: 2, 65: 4, 121: 5, 41: 15, 21: 25, 69: 27, 42: 1, 70: 1, 102: 1, 14: 2, 2: 5, 110: 7, 50: 10, 100: 1, 48: 2, 16: 3, 12: 11, 72: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 83, 87, 91, 103, 107, 111, 115, 119, 123], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 6948, 'token_per_expert': {3: 154, 7: 31, 15: 55, 19: 168, 23: 466, 27: 272, 39: 210, 43: 571, 47: 185, 51: 294, 55: 241, 59: 640, 63: 1031, 67: 275, 71: 118, 75: 65, 83: 427, 87: 108, 91: 73, 103: 34, 107: 64, 111: 453, 115: 401, 119: 503, 123: 109}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 17, 25, 29, 37, 45, 49, 53, 57, 61, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3089, 'token_per_expert': {1: 295, 5: 184, 17: 102, 25: 53, 29: 206, 37: 33, 45: 58, 49: 78, 53: 249, 57: 52, 61: 106, 73: 34, 77: 51, 81: 64, 85: 127, 89: 453, 93: 195, 97: 115, 101: 52, 105: 111, 109: 46, 113: 248, 117: 72, 125: 105}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 18, 22, 26, 30, 34, 38, 46, 54, 58, 62, 66, 74, 78, 82, 86, 90, 94, 98, 106, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3152, 'token_per_expert': {6: 31, 18: 33, 22: 411, 26: 352, 30: 83, 34: 34, 38: 31, 46: 24, 54: 345, 58: 11, 62: 93, 66: 16, 74: 408, 78: 97, 82: 257, 86: 108, 90: 43, 94: 84, 98: 76, 106: 505, 114: 16, 118: 56, 122: 24, 126: 14}}
experts_gpu_alloc_device_3 {'expert_ids': [4, 8, 20, 24, 28, 32, 36, 40, 44, 52, 56, 60, 64, 76, 80, 84, 88, 92, 96, 104, 108, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 3026, 'token_per_expert': {4: 140, 8: 711, 20: 136, 24: 311, 28: 124, 32: 149, 36: 59, 40: 52, 44: 29, 52: 82, 56: 33, 60: 125, 64: 61, 76: 209, 80: 19, 84: 49, 88: 42, 92: 141, 96: 106, 104: 123, 108: 85, 116: 97, 120: 14, 124: 129}}
INFO 05-06 10:50:21.304903.304903 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 1.474ms | allocate_experts_across_cpu_gpu: 0.280ms
INFO 05-06 10:50:21.304887.304887 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.340576171875e-05 seconds
INFO 05-06 10:50:21.306521.306521 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012359619140625 seconds
INFO 05-06 10:50:21.307517.307517 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007462501525878906 seconds
INFO 05-06 10:50:21.330761.330761 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.023398160934448242 seconds
INFO 05-06 10:50:21.331779.331779 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010523796081542969 seconds
INFO 05-06 10:50:21.333466.333466 mlpmodule.py:2799] [fused_experts] gmm total=1.867ms E=32 S=3168 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.334534.334534 mlpmodule.py:2799] [fused_experts] gmm total=2.075ms E=32 S=3179 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.334149.334149 mlpmodule.py:2799] [fused_experts] gmm total=2.370ms E=32 S=3055 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.334848.334848 mlpmodule.py:2799] [fused_experts] gmm total=2.657ms E=32 S=6982 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.336134.336134 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004195213317871094 seconds
INFO 05-06 10:50:21.336767.336767 lmp.py:1496] [layer_moe_fused] to time: 4.744529724121094e-05 seconds
INFO 05-06 10:50:21.336992.336992 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002586841583251953 seconds
DEBUG 05-06 10:50:21.336601.336601 cuda_h.py:27] end *layer_moe_fused cost 34.081 ms
DEBUG 05-06 10:50:21.354241.354241 cuda_h.py:27] end prefill_layer cost 53.920 ms
DEBUG 05-06 10:50:21.354051.354051 lmp.py:841] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 10:50:21.354198.354198 lmp.py:824] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 10:50:21.358092.358092 cuda_h.py:27] end *sagl cost 4.169 ms
experts_cpu_alloc {'expert_ids': [103, 95, 3, 11, 69, 109, 121, 1, 66, 90, 82, 78, 110, 38, 122, 30, 40, 124, 8, 48, 56, 92, 32], 'token_total': 114, 'token_per_expert': {103: 1, 95: 2, 3: 4, 11: 4, 69: 1, 109: 1, 121: 1, 1: 3, 66: 1, 90: 1, 82: 3, 78: 5, 110: 6, 38: 10, 122: 10, 30: 12, 40: 1, 124: 1, 8: 3, 48: 7, 56: 9, 92: 11, 32: 17}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 19, 23, 27, 31, 39, 43, 51, 55, 63, 67, 71, 75, 79, 83, 87, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 28, 'token_total': 4147, 'token_per_expert': {7: 27, 15: 10, 19: 24, 23: 160, 27: 20, 31: 53, 39: 581, 43: 113, 51: 12, 55: 79, 63: 117, 67: 102, 71: 1119, 75: 116, 79: 63, 83: 55, 87: 193, 99: 305, 107: 34, 111: 273, 115: 18, 119: 79, 123: 197, 127: 397}}
experts_gpu_alloc_device_1 {'expert_ids': [5, 9, 13, 17, 21, 29, 33, 37, 41, 45, 49, 53, 57, 61, 73, 77, 81, 93, 97, 101, 105, 113, 117, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 28, 'token_total': 4503, 'token_per_expert': {5: 145, 9: 213, 13: 245, 17: 12, 21: 5, 29: 172, 33: 483, 37: 32, 41: 9, 45: 4, 49: 655, 53: 9, 57: 35, 61: 322, 73: 208, 77: 27, 81: 17, 93: 149, 97: 11, 101: 1268, 105: 34, 113: 71, 117: 301, 125: 76}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 42, 46, 50, 54, 58, 62, 70, 74, 86, 94, 98, 102, 106, 114, 118, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 32, 'token_total': 2646, 'token_per_expert': {2: 424, 6: 43, 10: 32, 14: 53, 18: 82, 22: 391, 26: 27, 34: 27, 42: 256, 46: 156, 50: 18, 54: 25, 58: 20, 62: 14, 70: 197, 74: 159, 86: 33, 94: 230, 98: 36, 102: 27, 106: 64, 114: 52, 118: 129, 126: 151}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 16, 20, 24, 28, 36, 44, 52, 60, 64, 68, 72, 76, 80, 84, 88, 96, 100, 104, 112, 116, 120], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 4974, 'token_per_expert': {0: 75, 4: 198, 16: 450, 20: 611, 24: 201, 28: 226, 36: 356, 44: 69, 52: 79, 60: 186, 64: 449, 68: 46, 72: 285, 76: 121, 80: 115, 84: 48, 88: 246, 96: 122, 100: 86, 104: 190, 112: 513, 116: 166, 120: 136}}
INFO 05-06 10:50:21.362908.362908 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 2.005ms | allocate_experts_across_cpu_gpu: 0.276ms
INFO 05-06 10:50:21.362455.362455 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.221366882324219e-05 seconds
INFO 05-06 10:50:21.363682.363682 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006835460662841797 seconds
INFO 05-06 10:50:21.363158.363158 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005083084106445312 seconds
INFO 05-06 10:50:21.392729.392729 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02886056900024414 seconds
INFO 05-06 10:50:21.393473.393473 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009531974792480469 seconds
INFO 05-06 10:50:21.395822.395822 mlpmodule.py:2799] [fused_experts] gmm total=1.920ms E=32 S=4158 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.395342.395342 mlpmodule.py:2799] [fused_experts] gmm total=1.922ms E=32 S=2694 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.396540.396540 mlpmodule.py:2799] [fused_experts] gmm total=2.153ms E=32 S=4509 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.396462.396462 mlpmodule.py:2799] [fused_experts] gmm total=2.904ms E=32 S=5023 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.398231.398231 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004773855209350586 seconds
INFO 05-06 10:50:21.398056.398056 lmp.py:1496] [layer_moe_fused] to time: 4.887580871582031e-05 seconds
INFO 05-06 10:50:21.398439.398439 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002353191375732422 seconds
DEBUG 05-06 10:50:21.399924.399924 cuda_h.py:27] end *layer_moe_fused cost 39.548 ms
DEBUG 05-06 10:50:21.419215.419215 cuda_h.py:27] end prefill_layer cost 65.560 ms
DEBUG 05-06 10:50:21.420866.420866 lmp.py:841] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 10:50:21.420045.420045 lmp.py:824] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 10:50:21.421390.421390 cuda_h.py:27] end *sagl cost 1.586 ms
experts_cpu_alloc {'expert_ids': [55, 63, 7, 31, 83, 59, 49, 33, 21, 97, 81, 54, 66, 38, 114, 22, 18, 74, 84, 88, 100, 8, 92, 4], 'token_total': 184, 'token_per_expert': {55: 2, 63: 2, 7: 7, 31: 8, 83: 10, 59: 11, 49: 3, 33: 4, 21: 6, 97: 6, 81: 9, 54: 1, 66: 4, 38: 11, 114: 13, 22: 16, 18: 18, 74: 31, 84: 1, 88: 1, 100: 3, 8: 5, 92: 5, 4: 7}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 23, 27, 35, 43, 47, 51, 67, 71, 75, 79, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3988, 'token_per_expert': {3: 31, 11: 28, 15: 14, 19: 36, 23: 359, 27: 93, 35: 539, 43: 42, 47: 14, 51: 160, 67: 12, 71: 126, 75: 191, 79: 194, 87: 357, 91: 30, 95: 93, 99: 830, 103: 60, 107: 136, 111: 24, 115: 319, 119: 152, 123: 99, 127: 49}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 25, 29, 37, 41, 53, 57, 65, 69, 73, 77, 85, 89, 93, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4042, 'token_per_expert': {1: 93, 5: 72, 9: 161, 13: 282, 17: 11, 25: 817, 29: 15, 37: 44, 41: 48, 53: 419, 57: 61, 65: 364, 69: 101, 73: 77, 77: 80, 85: 66, 89: 57, 93: 643, 101: 13, 105: 69, 109: 17, 113: 122, 117: 196, 121: 154, 125: 60}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 26, 30, 34, 42, 46, 50, 58, 62, 70, 78, 82, 86, 90, 94, 98, 102, 106, 110, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4454, 'token_per_expert': {2: 97, 6: 110, 10: 106, 14: 53, 26: 140, 30: 50, 34: 345, 42: 56, 46: 161, 50: 138, 58: 93, 62: 139, 70: 85, 78: 210, 82: 39, 86: 513, 90: 431, 94: 271, 98: 260, 102: 529, 106: 394, 110: 62, 122: 110, 126: 62}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 16, 20, 24, 28, 32, 36, 40, 44, 52, 56, 60, 64, 68, 72, 76, 80, 96, 104, 108, 112, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3716, 'token_per_expert': {0: 60, 16: 14, 20: 33, 24: 172, 28: 84, 32: 148, 36: 157, 40: 12, 44: 95, 52: 9, 56: 90, 60: 28, 64: 504, 68: 1234, 72: 13, 76: 40, 80: 44, 96: 163, 104: 207, 108: 483, 112: 8, 116: 85, 120: 17, 124: 16}}
INFO 05-06 10:50:21.424987.424987 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 1.415ms | allocate_experts_across_cpu_gpu: 0.283ms
INFO 05-06 10:50:21.424633.424633 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.7220458984375e-05 seconds
INFO 05-06 10:50:21.425896.425896 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011317729949951172 seconds
INFO 05-06 10:50:21.426141.426141 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006570816040039062 seconds
INFO 05-06 10:50:21.455995.455995 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.029000520706176758 seconds
INFO 05-06 10:50:21.456806.456806 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010018348693847656 seconds
INFO 05-06 10:50:21.459966.459966 mlpmodule.py:2799] [fused_experts] gmm total=1.992ms E=32 S=4028 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.459165.459165 mlpmodule.py:2799] [fused_experts] gmm total=2.107ms E=32 S=4548 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.459825.459825 mlpmodule.py:2799] [fused_experts] gmm total=2.292ms E=32 S=4070 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.459228.459228 mlpmodule.py:2799] [fused_experts] gmm total=2.346ms E=32 S=3738 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.460965.460965 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003700733184814453 seconds
INFO 05-06 10:50:21.460460.460460 lmp.py:1496] [layer_moe_fused] to time: 4.744529724121094e-05 seconds
INFO 05-06 10:50:21.460837.460837 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002644062042236328 seconds
DEBUG 05-06 10:50:21.461521.461521 cuda_h.py:27] end *layer_moe_fused cost 38.820 ms
DEBUG 05-06 10:50:21.484895.484895 cuda_h.py:27] end prefill_layer cost 63.925 ms
DEBUG 05-06 10:50:21.484261.484261 lmp.py:841] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 10:50:21.484871.484871 lmp.py:824] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 10:50:21.485647.485647 cuda_h.py:27] end *sagl cost 1.587 ms
experts_cpu_alloc {'expert_ids': [27, 75, 3, 119, 11, 39, 31, 1, 37, 73, 109, 49, 46, 58, 102, 38, 50, 94, 124, 24, 92, 100, 36, 32], 'token_total': 244, 'token_per_expert': {27: 2, 75: 3, 3: 4, 119: 5, 11: 7, 39: 7, 31: 14, 1: 4, 37: 15, 73: 19, 109: 20, 49: 22, 46: 1, 58: 3, 102: 7, 38: 12, 50: 12, 94: 13, 124: 3, 24: 5, 92: 7, 100: 13, 36: 18, 32: 28}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 19, 23, 35, 43, 47, 51, 55, 59, 63, 67, 71, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 2949, 'token_per_expert': {7: 158, 15: 52, 19: 85, 23: 54, 35: 32, 43: 136, 47: 115, 51: 94, 55: 29, 59: 137, 63: 27, 67: 21, 71: 112, 79: 214, 83: 114, 87: 94, 91: 856, 95: 75, 99: 99, 103: 163, 107: 15, 111: 52, 115: 115, 123: 70, 127: 30}}
experts_gpu_alloc_device_1 {'expert_ids': [5, 9, 13, 17, 21, 25, 29, 33, 41, 45, 53, 57, 61, 65, 69, 77, 85, 97, 101, 105, 113, 117, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 5019, 'token_per_expert': {5: 171, 9: 91, 13: 128, 17: 35, 21: 45, 25: 56, 29: 488, 33: 108, 41: 49, 45: 58, 53: 232, 57: 190, 61: 137, 65: 257, 69: 355, 77: 33, 85: 292, 97: 990, 101: 69, 105: 130, 113: 232, 117: 82, 121: 610, 125: 181}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 10, 14, 18, 22, 26, 30, 34, 42, 54, 62, 66, 70, 78, 82, 86, 90, 98, 106, 110, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3532, 'token_per_expert': {6: 74, 10: 273, 14: 254, 18: 137, 22: 81, 26: 34, 30: 15, 34: 405, 42: 221, 54: 51, 62: 16, 66: 49, 70: 349, 78: 23, 82: 48, 86: 232, 90: 333, 98: 64, 106: 184, 110: 260, 114: 249, 118: 68, 122: 68, 126: 44}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 16, 20, 28, 44, 48, 52, 56, 60, 64, 68, 72, 80, 84, 88, 96, 104, 108, 112, 116, 120], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4640, 'token_per_expert': {0: 34, 4: 375, 8: 87, 12: 432, 16: 39, 20: 291, 28: 222, 44: 246, 48: 192, 52: 350, 56: 186, 60: 172, 64: 72, 68: 86, 72: 127, 80: 51, 84: 321, 88: 33, 96: 170, 104: 152, 108: 541, 112: 124, 116: 44, 120: 293}}
INFO 05-06 10:50:21.488961.488961 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 1.072ms | allocate_experts_across_cpu_gpu: 0.285ms
INFO 05-06 10:50:21.488184.488184 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.9604644775390625e-05 seconds
INFO 05-06 10:50:21.489119.489119 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011591911315917969 seconds
INFO 05-06 10:50:21.490108.490108 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.000736236572265625 seconds
INFO 05-06 10:50:21.524346.524346 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.03429985046386719 seconds
INFO 05-06 10:50:21.526590.526590 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001177072525024414 seconds
INFO 05-06 10:50:21.528039.528039 mlpmodule.py:2799] [fused_experts] gmm total=1.912ms E=32 S=3580 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.528395.528395 mlpmodule.py:2799] [fused_experts] gmm total=2.203ms E=32 S=2991 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.529108.529108 mlpmodule.py:2799] [fused_experts] gmm total=2.549ms E=32 S=4714 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.529755.529755 mlpmodule.py:2799] [fused_experts] gmm total=2.786ms E=32 S=5099 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.530253.530253 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004680156707763672 seconds
INFO 05-06 10:50:21.531517.531517 lmp.py:1496] [layer_moe_fused] to time: 5.7220458984375e-05 seconds
INFO 05-06 10:50:21.531732.531732 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003731250762939453 seconds
DEBUG 05-06 10:50:21.532977.532977 cuda_h.py:27] end *layer_moe_fused cost 45.324 ms
DEBUG 05-06 10:50:21.555146.555146 cuda_h.py:27] end prefill_layer cost 71.292 ms
DEBUG 05-06 10:50:21.555996.555996 lmp.py:841] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 10:50:21.555845.555845 lmp.py:824] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 10:50:21.557710.557710 cuda_h.py:27] end *sagl cost 1.867 ms
experts_cpu_alloc {'expert_ids': [83, 59, 67, 107, 23, 101, 13, 37, 33, 25, 9, 30, 18, 90, 118, 26, 34, 112, 60, 40, 0, 100, 24, 72], 'token_total': 313, 'token_per_expert': {83: 2, 59: 3, 67: 5, 107: 7, 23: 9, 101: 21, 13: 23, 37: 25, 33: 26, 25: 27, 9: 29, 30: 1, 18: 8, 90: 11, 118: 22, 26: 24, 34: 24, 112: 1, 60: 2, 40: 5, 0: 7, 100: 7, 24: 11, 72: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 35, 39, 43, 47, 51, 55, 63, 71, 75, 79, 87, 91, 99, 103, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4201, 'token_per_expert': {3: 126, 7: 10, 11: 65, 15: 222, 19: 294, 27: 173, 31: 118, 35: 10, 39: 15, 43: 35, 47: 68, 51: 643, 55: 168, 63: 154, 71: 213, 75: 284, 79: 11, 87: 416, 91: 32, 99: 38, 103: 777, 111: 76, 119: 27, 123: 154, 127: 72}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 17, 21, 29, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 105, 113, 117, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3745, 'token_per_expert': {1: 52, 5: 163, 17: 70, 21: 138, 29: 114, 41: 99, 45: 88, 49: 45, 53: 78, 57: 66, 61: 125, 65: 278, 69: 201, 73: 531, 77: 80, 81: 168, 85: 122, 89: 75, 93: 133, 105: 371, 113: 73, 117: 33, 121: 347, 125: 295}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 22, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 98, 102, 106, 110, 114, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 5228, 'token_per_expert': {2: 213, 6: 152, 10: 66, 14: 65, 22: 87, 38: 242, 42: 77, 46: 258, 50: 334, 54: 859, 58: 980, 62: 29, 66: 67, 70: 404, 74: 36, 82: 29, 86: 41, 98: 107, 102: 227, 106: 45, 110: 341, 114: 292, 122: 173, 126: 104}}
experts_gpu_alloc_device_3 {'expert_ids': [4, 8, 12, 16, 20, 28, 32, 36, 44, 48, 52, 56, 64, 68, 76, 80, 84, 92, 96, 104, 108, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 2897, 'token_per_expert': {4: 50, 8: 29, 12: 158, 16: 103, 20: 87, 28: 405, 32: 274, 36: 176, 44: 112, 48: 24, 52: 115, 56: 241, 64: 33, 68: 35, 76: 137, 80: 249, 84: 42, 92: 23, 96: 25, 104: 18, 108: 65, 116: 40, 120: 361, 124: 95}}
INFO 05-06 10:50:21.560532.560532 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 1.252ms | allocate_experts_across_cpu_gpu: 0.938ms
INFO 05-06 10:50:21.560922.560922 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 0.00014519691467285156 seconds
INFO 05-06 10:50:21.562846.562846 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001397848129272461 seconds
INFO 05-06 10:50:21.564397.564397 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0016536712646484375 seconds
INFO 05-06 10:50:21.597468.597468 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.03300142288208008 seconds
INFO 05-06 10:50:21.598003.598003 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010728836059570312 seconds
INFO 05-06 10:50:21.600330.600330 mlpmodule.py:2799] [fused_experts] gmm total=1.807ms E=32 S=4227 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.601501.601501 mlpmodule.py:2799] [fused_experts] gmm total=1.912ms E=32 S=3896 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.601207.601207 mlpmodule.py:2799] [fused_experts] gmm total=2.385ms E=32 S=5318 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.602796.602796 mlpmodule.py:2799] [fused_experts] gmm total=3.124ms E=32 S=2943 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.602726.602726 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003894805908203125 seconds
INFO 05-06 10:50:21.602062.602062 lmp.py:1496] [layer_moe_fused] to time: 5.221366882324219e-05 seconds
INFO 05-06 10:50:21.603079.603079 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0001838207244873047 seconds
DEBUG 05-06 10:50:21.603432.603432 cuda_h.py:27] end *layer_moe_fused cost 45.292 ms
DEBUG 05-06 10:50:21.624711.624711 cuda_h.py:27] end prefill_layer cost 68.992 ms
DEBUG 05-06 10:50:21.624290.624290 lmp.py:841] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 10:50:21.624914.624914 lmp.py:824] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 10:50:21.626083.626083 cuda_h.py:27] end *sagl cost 2.067 ms
experts_cpu_alloc {'expert_ids': [107, 63, 31, 47, 11, 49, 53, 25, 33, 29, 2, 14, 118, 126, 94, 50, 66, 108, 84, 96, 0, 100, 64, 112], 'token_total': 174, 'token_per_expert': {107: 1, 63: 3, 31: 4, 47: 5, 11: 6, 49: 1, 53: 1, 25: 2, 33: 14, 29: 15, 2: 3, 14: 3, 118: 3, 126: 5, 94: 7, 50: 13, 66: 13, 108: 2, 84: 3, 96: 12, 0: 14, 100: 14, 64: 15, 112: 15}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 39, 43, 51, 55, 59, 67, 71, 75, 79, 83, 95, 99, 103, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 4361, 'token_per_expert': {3: 94, 7: 112, 15: 91, 19: 126, 23: 212, 27: 136, 39: 158, 43: 437, 51: 133, 55: 15, 59: 6, 67: 32, 71: 174, 75: 411, 79: 26, 83: 144, 95: 1099, 99: 112, 103: 473, 111: 131, 115: 37, 119: 14, 123: 16, 127: 172}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 21, 37, 41, 45, 57, 61, 69, 73, 77, 81, 89, 93, 97, 101, 105, 113, 117, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 3725, 'token_per_expert': {1: 207, 5: 66, 9: 109, 13: 162, 17: 91, 21: 172, 37: 112, 41: 21, 45: 111, 57: 185, 61: 168, 69: 315, 73: 54, 77: 28, 81: 322, 89: 179, 93: 462, 97: 60, 101: 599, 105: 31, 113: 50, 117: 51, 121: 15, 125: 155}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 10, 18, 22, 26, 30, 34, 38, 42, 46, 54, 58, 62, 70, 74, 82, 86, 90, 98, 102, 106, 110, 114, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4281, 'token_per_expert': {6: 20, 10: 41, 18: 31, 22: 185, 26: 21, 30: 159, 34: 22, 38: 131, 42: 64, 46: 845, 54: 141, 58: 24, 62: 88, 70: 783, 74: 361, 82: 47, 86: 105, 90: 24, 98: 56, 102: 183, 106: 819, 110: 13, 114: 34, 122: 84}}
experts_gpu_alloc_device_3 {'expert_ids': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 68, 72, 76, 80, 88, 92, 104, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3843, 'token_per_expert': {4: 191, 8: 28, 12: 712, 16: 431, 20: 30, 24: 141, 28: 18, 32: 174, 36: 201, 40: 242, 44: 38, 48: 257, 52: 31, 56: 311, 68: 109, 72: 153, 76: 201, 80: 58, 88: 111, 92: 222, 104: 27, 116: 42, 120: 23, 124: 92}}
INFO 05-06 10:50:21.629126.629126 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.625ms | allocate_experts_across_cpu_gpu: 0.471ms
INFO 05-06 10:50:21.629979.629979 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.606910705566406e-05 seconds
INFO 05-06 10:50:21.630264.630264 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009372234344482422 seconds
INFO 05-06 10:50:21.631850.631850 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008227825164794922 seconds
INFO 05-06 10:50:21.660153.660153 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.029280662536621094 seconds
INFO 05-06 10:50:21.661541.661541 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010223388671875 seconds
INFO 05-06 10:50:21.664588.664588 mlpmodule.py:2799] [fused_experts] gmm total=1.913ms E=32 S=3758 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.664777.664777 mlpmodule.py:2799] [fused_experts] gmm total=2.072ms E=32 S=4380 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.664573.664573 mlpmodule.py:2799] [fused_experts] gmm total=1.974ms E=32 S=3918 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.664612.664612 mlpmodule.py:2799] [fused_experts] gmm total=2.575ms E=32 S=4328 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.665339.665339 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003573179244995117 seconds
INFO 05-06 10:50:21.665409.665409 lmp.py:1496] [layer_moe_fused] to time: 4.8160552978515625e-05 seconds
INFO 05-06 10:50:21.665137.665137 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00024509429931640625 seconds
DEBUG 05-06 10:50:21.666512.666512 cuda_h.py:27] end *layer_moe_fused cost 38.532 ms
DEBUG 05-06 10:50:21.688797.688797 cuda_h.py:27] end prefill_layer cost 63.952 ms
DEBUG 05-06 10:50:21.688647.688647 lmp.py:841] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 10:50:21.688542.688542 lmp.py:824] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 10:50:21.690975.690975 cuda_h.py:27] end *sagl cost 1.620 ms
experts_cpu_alloc {'expert_ids': [23, 123, 35, 55, 51, 59, 17, 65, 77, 101, 53, 109, 110, 118, 122, 22, 114, 30, 38, 24, 36, 52, 48, 40], 'token_total': 123, 'token_per_expert': {23: 1, 123: 1, 35: 2, 55: 3, 51: 9, 59: 9, 17: 3, 65: 4, 77: 8, 101: 12, 53: 14, 109: 17, 110: 1, 118: 1, 122: 1, 22: 3, 114: 5, 30: 6, 38: 6, 24: 1, 36: 2, 52: 3, 48: 5, 40: 6}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 39, 43, 47, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 2572, 'token_per_expert': {3: 20, 7: 164, 11: 52, 15: 19, 19: 66, 27: 30, 31: 138, 39: 198, 43: 115, 47: 190, 63: 121, 67: 53, 71: 173, 75: 195, 79: 73, 83: 105, 87: 15, 91: 14, 99: 139, 103: 20, 107: 18, 111: 67, 115: 339, 119: 37, 127: 211}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 21, 25, 29, 33, 37, 41, 49, 57, 61, 69, 73, 81, 85, 89, 93, 97, 105, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4276, 'token_per_expert': {1: 816, 5: 71, 9: 70, 13: 176, 21: 283, 25: 18, 29: 40, 33: 28, 37: 147, 41: 227, 49: 250, 57: 237, 61: 34, 69: 78, 73: 28, 81: 659, 85: 149, 89: 64, 93: 47, 97: 54, 105: 89, 113: 209, 117: 60, 121: 79, 125: 363}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 26, 34, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 126], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4351, 'token_per_expert': {2: 24, 6: 42, 10: 218, 14: 418, 18: 171, 26: 26, 34: 50, 42: 258, 46: 210, 50: 29, 54: 126, 58: 159, 62: 312, 66: 10, 70: 19, 74: 506, 78: 57, 82: 237, 86: 556, 90: 93, 94: 128, 98: 73, 102: 18, 106: 363, 126: 248}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 16, 20, 28, 32, 44, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 5062, 'token_per_expert': {0: 379, 4: 83, 8: 577, 12: 18, 16: 228, 20: 147, 28: 45, 32: 10, 44: 84, 56: 52, 60: 524, 64: 32, 68: 172, 72: 147, 76: 783, 80: 659, 84: 110, 88: 305, 92: 188, 100: 150, 104: 9, 108: 263, 112: 45, 120: 25, 124: 27}}
INFO 05-06 10:50:21.692094.692094 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 1.393ms | allocate_experts_across_cpu_gpu: 0.289ms
INFO 05-06 10:50:21.693000.693000 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.009506225585938e-05 seconds
INFO 05-06 10:50:21.694174.694174 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009622573852539062 seconds
INFO 05-06 10:50:21.694180.694180 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005137920379638672 seconds
INFO 05-06 10:50:21.726753.726753 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.031881093978881836 seconds
INFO 05-06 10:50:21.727002.727002 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001004934310913086 seconds
INFO 05-06 10:50:21.729005.729005 mlpmodule.py:2799] [fused_experts] gmm total=1.866ms E=32 S=2597 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.730111.730111 mlpmodule.py:2799] [fused_experts] gmm total=2.206ms E=32 S=4334 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.730499.730499 mlpmodule.py:2799] [fused_experts] gmm total=2.264ms E=32 S=4374 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.730230.730230 mlpmodule.py:2799] [fused_experts] gmm total=2.292ms E=32 S=5079 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.731660.731660 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037508010864257812 seconds
INFO 05-06 10:50:21.731015.731015 lmp.py:1496] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-06 10:50:21.732405.732405 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00024247169494628906 seconds
DEBUG 05-06 10:50:21.732301.732301 cuda_h.py:27] end *layer_moe_fused cost 41.476 ms
DEBUG 05-06 10:50:21.757441.757441 cuda_h.py:27] end prefill_layer cost 69.158 ms
DEBUG 05-06 10:50:21.757330.757330 lmp.py:841] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 10:50:21.757186.757186 lmp.py:824] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 10:50:21.760433.760433 cuda_h.py:27] end *sagl cost 2.818 ms
experts_cpu_alloc {'expert_ids': [15, 35, 127, 45, 109, 65, 105, 41, 9, 85, 90, 26, 114, 94, 106, 110, 104, 84, 72, 96, 88, 52, 64], 'token_total': 154, 'token_per_expert': {15: 1, 35: 5, 127: 5, 45: 1, 109: 1, 65: 2, 105: 2, 41: 4, 9: 11, 85: 11, 90: 1, 26: 3, 114: 5, 94: 8, 106: 10, 110: 10, 104: 1, 84: 4, 72: 6, 96: 6, 88: 9, 52: 20, 64: 28}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 27, 31, 39, 43, 47, 51, 59, 63, 67, 71, 79, 83, 87, 91, 99, 111, 115, 119, 123], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 27, 'token_total': 4594, 'token_per_expert': {3: 533, 7: 782, 11: 30, 19: 189, 23: 240, 27: 63, 31: 165, 39: 21, 43: 75, 47: 17, 51: 43, 59: 58, 63: 9, 67: 251, 71: 57, 79: 409, 83: 389, 87: 596, 91: 82, 99: 84, 111: 243, 115: 13, 119: 203, 123: 42}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 13, 17, 21, 25, 29, 33, 37, 49, 53, 57, 61, 69, 77, 81, 89, 93, 97, 113, 117, 121, 125], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 4066, 'token_per_expert': {1: 542, 5: 552, 13: 14, 17: 350, 21: 13, 25: 97, 29: 85, 33: 19, 37: 128, 49: 290, 53: 17, 57: 146, 61: 109, 69: 102, 77: 102, 81: 383, 89: 88, 93: 407, 97: 11, 113: 444, 117: 78, 121: 68, 125: 21}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 18, 22, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 98, 102, 118, 122, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 3157, 'token_per_expert': {2: 630, 6: 830, 10: 112, 18: 61, 22: 21, 30: 154, 34: 17, 38: 103, 42: 32, 46: 44, 50: 27, 54: 54, 58: 15, 62: 41, 66: 114, 70: 90, 74: 20, 82: 74, 98: 75, 102: 537, 118: 13, 122: 18, 126: 75}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 68, 76, 80, 92, 100, 108, 112, 116, 120, 124], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 4413, 'token_per_expert': {0: 526, 4: 528, 8: 32, 16: 579, 20: 103, 24: 155, 28: 76, 32: 228, 36: 101, 40: 35, 44: 30, 48: 64, 56: 486, 68: 229, 76: 168, 80: 30, 92: 376, 100: 219, 108: 136, 112: 99, 116: 91, 120: 48, 124: 74}}
INFO 05-06 10:50:21.764469.764469 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 2.838ms | allocate_experts_across_cpu_gpu: 0.271ms
INFO 05-06 10:50:21.764222.764222 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.009506225585938e-05 seconds
INFO 05-06 10:50:21.766583.766583 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010454654693603516 seconds
INFO 05-06 10:50:21.766031.766031 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004897117614746094 seconds
INFO 05-06 10:50:21.794835.794835 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.02747201919555664 seconds
INFO 05-06 10:50:21.795055.795055 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00095367431640625 seconds
INFO 05-06 10:50:21.797638.797638 mlpmodule.py:2799] [fused_experts] gmm total=1.778ms E=32 S=4605 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.797835.797835 mlpmodule.py:2799] [fused_experts] gmm total=1.800ms E=32 S=3194 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.797394.797394 mlpmodule.py:2799] [fused_experts] gmm total=1.949ms E=32 S=4098 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.798419.798419 mlpmodule.py:2799] [fused_experts] gmm total=2.771ms E=32 S=4487 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.799347.799347 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003692150115966797 seconds
INFO 05-06 10:50:21.799225.799225 lmp.py:1496] [layer_moe_fused] to time: 4.863739013671875e-05 seconds
INFO 05-06 10:50:21.799039.799039 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00023651123046875 seconds
DEBUG 05-06 10:50:21.800307.800307 cuda_h.py:27] end *layer_moe_fused cost 38.418 ms
DEBUG 05-06 10:50:21.821687.821687 cuda_h.py:27] end prefill_layer cost 63.424 ms
DEBUG 05-06 10:50:21.821431.821431 lmp.py:841] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 10:50:21.821564.821564 lmp.py:824] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 10:50:21.823100.823100 cuda_h.py:27] end *sagl cost 1.571 ms
experts_cpu_alloc {'expert_ids': [27, 51, 87, 59, 83, 69, 121, 29, 93, 81, 33, 30, 126, 10, 54, 62, 96, 48, 56, 8, 32], 'token_total': 81, 'token_per_expert': {27: 1, 51: 1, 87: 2, 59: 3, 83: 4, 69: 1, 121: 1, 29: 2, 93: 5, 81: 6, 33: 12, 30: 1, 126: 1, 10: 2, 54: 4, 62: 4, 96: 1, 48: 3, 56: 6, 8: 9, 32: 12}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 31, 35, 39, 47, 63, 67, 71, 79, 91, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 27, 'token_total': 3584, 'token_per_expert': {3: 614, 7: 514, 15: 327, 19: 133, 23: 171, 31: 14, 35: 89, 39: 595, 47: 26, 63: 25, 67: 14, 71: 460, 79: 38, 91: 127, 95: 132, 103: 55, 107: 35, 111: 17, 115: 134, 119: 26, 123: 32, 127: 6}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 13, 17, 21, 25, 37, 41, 45, 49, 53, 65, 73, 77, 85, 89, 97, 101, 105, 113, 117, 125], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 28, 'token_total': 4108, 'token_per_expert': {1: 542, 5: 681, 13: 14, 17: 28, 21: 543, 25: 148, 37: 12, 41: 12, 45: 432, 49: 285, 53: 593, 65: 33, 73: 132, 77: 74, 85: 58, 89: 24, 97: 230, 101: 87, 105: 12, 113: 14, 117: 125, 125: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 18, 22, 34, 38, 46, 50, 58, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 27, 'token_total': 5537, 'token_per_expert': {2: 519, 6: 663, 18: 6, 22: 80, 34: 88, 38: 43, 46: 265, 50: 487, 58: 98, 70: 41, 74: 308, 78: 854, 82: 338, 86: 385, 90: 68, 94: 31, 98: 44, 102: 26, 106: 230, 110: 323, 114: 405, 118: 235}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 20, 24, 36, 40, 64, 68, 76, 80, 84, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 21, 'ideal_gpu_count': 21, 'keep_on_gpu': 21, 'hit_count_on_device': 26, 'token_total': 3074, 'token_per_expert': {0: 513, 4: 516, 12: 55, 20: 13, 24: 23, 36: 83, 40: 70, 64: 12, 68: 54, 76: 113, 80: 77, 84: 136, 88: 82, 92: 141, 100: 64, 104: 83, 108: 518, 112: 31, 116: 372, 120: 17, 124: 101}}
INFO 05-06 10:50:21.825834.825834 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 1.358ms | allocate_experts_across_cpu_gpu: 0.258ms
INFO 05-06 10:50:21.825513.825513 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.1975250244140625e-05 seconds
INFO 05-06 10:50:21.826061.826061 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008947849273681641 seconds
INFO 05-06 10:50:21.827542.827542 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0004780292510986328 seconds
INFO 05-06 10:50:21.857235.857235 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.030106782913208008 seconds
INFO 05-06 10:50:21.858342.858342 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009424686431884766 seconds
INFO 05-06 10:50:21.860079.860079 mlpmodule.py:2799] [fused_experts] gmm total=2.032ms E=32 S=3595 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.860554.860554 mlpmodule.py:2799] [fused_experts] gmm total=2.120ms E=32 S=4135 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.861705.861705 mlpmodule.py:2799] [fused_experts] gmm total=2.215ms E=32 S=5549 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.861462.861462 mlpmodule.py:2799] [fused_experts] gmm total=2.229ms E=32 S=3105 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.862161.862161 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0036258697509765625 seconds
INFO 05-06 10:50:21.862000.862000 lmp.py:1496] [layer_moe_fused] to time: 4.9591064453125e-05 seconds
INFO 05-06 10:50:21.862705.862705 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003330707550048828 seconds
DEBUG 05-06 10:50:21.863977.863977 cuda_h.py:27] end *layer_moe_fused cost 39.116 ms
DEBUG 05-06 10:50:21.886499.886499 cuda_h.py:27] end prefill_layer cost 65.288 ms
DEBUG 05-06 10:50:21.886196.886196 lmp.py:841] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 10:50:21.886568.886568 lmp.py:824] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 10:50:21.888927.888927 cuda_h.py:27] end *sagl cost 1.603 ms
experts_cpu_alloc {'expert_ids': [111, 23, 83, 11, 107, 47, 53, 97, 89, 45, 61, 50, 18, 74, 66, 10, 90, 12, 76, 72, 112, 56, 48], 'token_total': 182, 'token_per_expert': {111: 3, 23: 10, 83: 19, 11: 20, 107: 23, 47: 24, 53: 2, 97: 3, 89: 4, 45: 6, 61: 9, 50: 1, 18: 2, 74: 3, 66: 6, 10: 9, 90: 11, 12: 2, 76: 2, 72: 4, 112: 5, 56: 6, 48: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 27, 31, 39, 43, 51, 55, 59, 63, 67, 71, 75, 79, 87, 91, 95, 99, 103, 115, 119, 123], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 4685, 'token_per_expert': {3: 620, 7: 514, 15: 120, 27: 43, 31: 698, 39: 200, 43: 46, 51: 113, 55: 52, 59: 256, 63: 165, 67: 38, 71: 298, 75: 69, 79: 380, 87: 28, 91: 632, 95: 32, 99: 64, 103: 100, 115: 108, 119: 84, 123: 25}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 33, 37, 41, 57, 65, 69, 73, 81, 93, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 28, 'token_total': 4045, 'token_per_expert': {1: 725, 5: 512, 9: 32, 13: 76, 17: 375, 21: 143, 25: 165, 33: 178, 37: 402, 41: 71, 57: 13, 65: 19, 69: 106, 73: 46, 81: 336, 93: 22, 101: 29, 105: 12, 109: 22, 113: 97, 117: 55, 121: 440, 125: 169}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 14, 22, 26, 34, 38, 42, 46, 62, 70, 78, 82, 86, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 4443, 'token_per_expert': {2: 550, 6: 716, 14: 306, 22: 99, 26: 26, 34: 96, 38: 150, 42: 45, 46: 27, 62: 15, 70: 30, 78: 396, 82: 63, 86: 108, 94: 25, 98: 138, 102: 174, 106: 16, 110: 449, 114: 682, 118: 124, 122: 66, 126: 142}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 16, 20, 28, 32, 36, 40, 52, 60, 64, 68, 80, 84, 92, 96, 100, 104, 108, 116, 120, 124], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 3029, 'token_per_expert': {0: 531, 4: 512, 8: 20, 16: 25, 20: 179, 28: 30, 32: 303, 36: 9, 40: 37, 52: 20, 60: 107, 64: 32, 68: 40, 80: 30, 84: 101, 92: 35, 96: 9, 100: 447, 104: 24, 108: 41, 116: 81, 120: 341, 124: 75}}
INFO 05-06 10:50:21.890073.890073 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 1.004ms | allocate_experts_across_cpu_gpu: 0.272ms
INFO 05-06 10:50:21.890660.890660 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 5.245208740234375e-05 seconds
INFO 05-06 10:50:21.891315.891315 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011360645294189453 seconds
INFO 05-06 10:50:21.892751.892751 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0006213188171386719 seconds
INFO 05-06 10:50:21.919366.919366 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.026819944381713867 seconds
INFO 05-06 10:50:21.920844.920844 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010380744934082031 seconds
INFO 05-06 10:50:21.922883.922883 mlpmodule.py:2799] [fused_experts] gmm total=1.759ms E=32 S=4784 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.922020.922020 mlpmodule.py:2799] [fused_experts] gmm total=1.841ms E=32 S=4069 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.922282.922282 mlpmodule.py:2799] [fused_experts] gmm total=1.908ms E=32 S=4475 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.923297.923297 mlpmodule.py:2799] [fused_experts] gmm total=2.288ms E=32 S=3056 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.924643.924643 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0032215118408203125 seconds
INFO 05-06 10:50:21.924282.924282 lmp.py:1496] [layer_moe_fused] to time: 4.887580871582031e-05 seconds
INFO 05-06 10:50:21.924587.924587 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002484321594238281 seconds
DEBUG 05-06 10:50:21.925727.925727 cuda_h.py:27] end *layer_moe_fused cost 35.642 ms
DEBUG 05-06 10:50:21.946096.946096 cuda_h.py:27] end prefill_layer cost 59.685 ms
DEBUG 05-06 10:50:21.946622.946622 lmp.py:841] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 10:50:21.946054.946054 lmp.py:824] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 10:50:21.948905.948905 cuda_h.py:27] end *sagl cost 1.730 ms
experts_cpu_alloc {'expert_ids': [79, 87, 91, 111, 27, 51, 49, 17, 33, 41, 29, 82, 94, 46, 14, 54, 106, 18, 56, 84, 88, 96, 116, 68], 'token_total': 153, 'token_per_expert': {79: 2, 87: 6, 91: 6, 111: 6, 27: 8, 51: 12, 49: 1, 17: 3, 33: 6, 41: 6, 29: 7, 82: 1, 94: 1, 46: 5, 14: 8, 54: 8, 106: 8, 18: 9, 56: 2, 84: 3, 88: 5, 96: 7, 116: 15, 68: 18}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 31, 35, 39, 43, 47, 59, 63, 67, 71, 75, 83, 95, 99, 103, 107, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 5306, 'token_per_expert': {3: 530, 7: 575, 11: 118, 15: 18, 19: 39, 23: 38, 31: 236, 35: 21, 39: 367, 43: 26, 47: 275, 59: 236, 63: 23, 67: 12, 71: 75, 75: 305, 83: 80, 95: 320, 99: 165, 103: 207, 107: 53, 115: 799, 119: 392, 123: 246, 127: 150}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 21, 25, 37, 45, 53, 57, 65, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3568, 'token_per_expert': {1: 519, 5: 536, 9: 15, 13: 81, 21: 11, 25: 45, 37: 11, 45: 56, 53: 97, 57: 79, 65: 340, 73: 15, 77: 11, 81: 66, 85: 10, 89: 129, 93: 15, 97: 392, 101: 18, 105: 70, 109: 26, 113: 162, 117: 197, 121: 573, 125: 94}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 22, 26, 30, 34, 38, 42, 50, 58, 62, 66, 70, 74, 78, 86, 90, 98, 102, 110, 114, 118, 122, 126], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 3950, 'token_per_expert': {2: 654, 6: 519, 10: 110, 22: 12, 26: 383, 30: 73, 34: 85, 38: 149, 42: 127, 50: 291, 58: 18, 62: 96, 66: 369, 70: 32, 74: 88, 78: 14, 86: 425, 90: 38, 98: 32, 102: 43, 110: 59, 114: 129, 118: 25, 122: 150, 126: 29}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 48, 52, 60, 64, 72, 76, 80, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3407, 'token_per_expert': {0: 556, 4: 513, 8: 149, 12: 97, 16: 50, 24: 142, 28: 42, 32: 73, 36: 35, 40: 24, 44: 34, 48: 22, 52: 82, 60: 64, 64: 36, 72: 72, 76: 115, 80: 157, 92: 50, 100: 254, 104: 129, 108: 85, 112: 83, 120: 75, 124: 468}}
INFO 05-06 10:50:21.951660.951660 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 1.358ms | allocate_experts_across_cpu_gpu: 0.305ms
INFO 05-06 10:50:21.951942.951942 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 7.2479248046875e-05 seconds
INFO 05-06 10:50:21.952581.952581 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008356571197509766 seconds
INFO 05-06 10:50:21.953772.953772 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005080699920654297 seconds
INFO 05-06 10:50:21.980618.980618 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.027550697326660156 seconds
INFO 05-06 10:50:21.981070.981070 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009355545043945312 seconds
INFO 05-06 10:50:21.983219.983219 mlpmodule.py:2799] [fused_experts] gmm total=1.818ms E=32 S=3591 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.984994.984994 mlpmodule.py:2799] [fused_experts] gmm total=1.985ms E=32 S=3990 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.984470.984470 mlpmodule.py:2799] [fused_experts] gmm total=2.285ms E=32 S=3457 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.984191.984191 mlpmodule.py:2799] [fused_experts] gmm total=2.622ms E=32 S=5346 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:21.985373.985373 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003243684768676758 seconds
INFO 05-06 10:50:21.985112.985112 lmp.py:1496] [layer_moe_fused] to time: 4.935264587402344e-05 seconds
INFO 05-06 10:50:21.985513.985513 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003552436828613281 seconds
DEBUG 05-06 10:50:21.986475.986475 cuda_h.py:27] end *layer_moe_fused cost 36.711 ms
DEBUG 05-06 10:50:22.005828.005828 cuda_h.py:27] end prefill_layer cost 59.306 ms
DEBUG 05-06 10:50:22.006214.006214 lmp.py:841] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 10:50:22.006838.006838 lmp.py:824] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 10:50:22.008587.008587 cuda_h.py:27] end *sagl cost 1.913 ms
experts_cpu_alloc {'expert_ids': [35, 67, 87, 123, 11, 49, 61, 89, 57, 45, 25, 62, 26, 22, 106, 74, 50, 110, 60, 20, 12, 100, 44, 32], 'token_total': 158, 'token_per_expert': {35: 6, 67: 9, 87: 9, 123: 9, 11: 11, 49: 1, 61: 3, 89: 3, 57: 9, 45: 12, 25: 19, 62: 1, 26: 2, 22: 3, 106: 3, 74: 4, 50: 5, 110: 8, 60: 1, 20: 2, 12: 5, 100: 8, 44: 12, 32: 13}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 31, 39, 43, 47, 51, 55, 59, 63, 71, 75, 79, 83, 91, 95, 99, 103, 107, 111, 115, 119, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3993, 'token_per_expert': {3: 518, 7: 666, 19: 31, 23: 184, 31: 57, 39: 181, 43: 43, 47: 80, 51: 169, 55: 84, 59: 63, 63: 81, 71: 188, 75: 237, 79: 17, 83: 376, 91: 504, 95: 94, 99: 103, 103: 101, 107: 32, 111: 23, 115: 40, 119: 56, 127: 65}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 37, 41, 65, 69, 73, 77, 81, 85, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3887, 'token_per_expert': {1: 564, 5: 584, 9: 213, 13: 41, 17: 45, 21: 94, 29: 62, 33: 54, 37: 170, 41: 48, 65: 405, 69: 78, 73: 78, 77: 27, 81: 118, 85: 113, 93: 103, 97: 121, 101: 198, 105: 26, 109: 458, 113: 64, 117: 28, 121: 40, 125: 155}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 30, 34, 38, 42, 46, 54, 58, 66, 70, 78, 82, 86, 90, 94, 98, 102, 114, 118, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3183, 'token_per_expert': {2: 658, 6: 519, 10: 322, 14: 64, 18: 37, 30: 243, 34: 29, 38: 21, 42: 93, 46: 73, 54: 12, 58: 27, 66: 187, 70: 148, 78: 60, 82: 31, 86: 57, 90: 275, 94: 12, 98: 150, 102: 65, 114: 56, 118: 30, 126: 14}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 16, 24, 28, 36, 40, 48, 52, 64, 68, 72, 76, 80, 84, 88, 96, 104, 108, 112, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 5163, 'token_per_expert': {0: 595, 4: 534, 8: 32, 16: 157, 24: 79, 28: 76, 36: 92, 40: 45, 48: 37, 52: 207, 64: 217, 68: 482, 72: 155, 76: 735, 80: 18, 84: 129, 88: 176, 96: 21, 104: 121, 108: 328, 112: 647, 116: 114, 120: 84, 124: 82}}
INFO 05-06 10:50:22.010216.010216 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.725ms | allocate_experts_across_cpu_gpu: 0.292ms
INFO 05-06 10:50:22.010148.010148 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 6.0558319091796875e-05 seconds
INFO 05-06 10:50:22.011506.011506 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008356571197509766 seconds
INFO 05-06 10:50:22.011631.011631 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0005295276641845703 seconds
INFO 05-06 10:50:22.034929.034929 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.022478342056274414 seconds
INFO 05-06 10:50:22.035938.035938 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009710788726806641 seconds
INFO 05-06 10:50:22.037457.037457 mlpmodule.py:2799] [fused_experts] gmm total=2.023ms E=32 S=4037 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.037217.037217 mlpmodule.py:2799] [fused_experts] gmm total=1.992ms E=32 S=5204 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.038789.038789 mlpmodule.py:2799] [fused_experts] gmm total=2.168ms E=32 S=3934 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.038495.038495 mlpmodule.py:2799] [fused_experts] gmm total=2.255ms E=32 S=3209 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.039319.039319 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003560304641723633 seconds
INFO 05-06 10:50:22.039151.039151 lmp.py:1496] [layer_moe_fused] to time: 4.792213439941406e-05 seconds
INFO 05-06 10:50:22.039677.039677 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003414154052734375 seconds
DEBUG 05-06 10:50:22.040655.040655 cuda_h.py:27] end *layer_moe_fused cost 31.117 ms
DEBUG 05-06 10:50:22.060388.060388 cuda_h.py:27] end prefill_layer cost 54.188 ms
DEBUG 05-06 10:50:22.060344.060344 lmp.py:841] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 10:50:22.060729.060729 lmp.py:824] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 10:50:22.062033.062033 cuda_h.py:27] end *sagl cost 1.978 ms
experts_cpu_alloc {'expert_ids': [27, 35, 95, 115, 71, 25, 73, 101, 49, 29, 41, 89, 106, 94, 74, 50, 122, 98, 28, 120, 88, 36, 64, 104, 112], 'token_total': 209, 'token_per_expert': {27: 2, 35: 2, 95: 3, 115: 4, 71: 12, 25: 1, 73: 3, 101: 3, 49: 8, 29: 9, 41: 9, 89: 9, 106: 3, 94: 4, 74: 8, 50: 11, 122: 13, 98: 19, 28: 4, 120: 5, 88: 9, 36: 14, 64: 14, 104: 20, 112: 20}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 31, 43, 51, 55, 59, 63, 67, 75, 79, 83, 87, 91, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 4026, 'token_per_expert': {3: 628, 7: 521, 11: 18, 15: 35, 19: 102, 23: 238, 31: 155, 43: 38, 51: 53, 55: 112, 59: 18, 63: 153, 67: 515, 75: 180, 79: 38, 83: 226, 87: 549, 91: 34, 99: 34, 103: 21, 107: 122, 111: 38, 119: 55, 123: 45, 127: 98}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 21, 33, 37, 45, 53, 57, 61, 65, 69, 77, 81, 85, 93, 97, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 2926, 'token_per_expert': {1: 762, 5: 677, 9: 14, 13: 28, 17: 48, 21: 37, 33: 30, 37: 31, 45: 29, 53: 10, 57: 59, 61: 71, 65: 87, 69: 25, 77: 57, 81: 57, 85: 80, 93: 48, 97: 53, 105: 401, 109: 25, 113: 42, 117: 96, 121: 57, 125: 102}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 54, 58, 62, 66, 70, 78, 82, 86, 90, 102, 110, 114, 118, 126], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4210, 'token_per_expert': {2: 618, 6: 542, 10: 40, 14: 90, 18: 62, 22: 76, 26: 103, 30: 42, 34: 23, 38: 29, 42: 145, 54: 148, 58: 102, 62: 31, 66: 199, 70: 102, 78: 113, 82: 85, 86: 351, 90: 83, 102: 71, 110: 122, 114: 136, 118: 58, 126: 839}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 32, 40, 44, 48, 52, 56, 60, 68, 72, 76, 80, 84, 92, 96, 100, 108, 116, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 5013, 'token_per_expert': {0: 673, 4: 702, 8: 173, 12: 210, 16: 478, 20: 144, 24: 47, 32: 661, 40: 41, 44: 106, 48: 109, 52: 500, 56: 37, 60: 22, 68: 143, 72: 91, 76: 118, 80: 42, 84: 46, 92: 47, 96: 116, 100: 172, 108: 141, 116: 80, 124: 114}}
INFO 05-06 10:50:22.065428.065428 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.807ms | allocate_experts_across_cpu_gpu: 0.530ms
INFO 05-06 10:50:22.065911.065911 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.107589721679688e-05 seconds
INFO 05-06 10:50:22.066622.066622 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001039743423461914 seconds
INFO 05-06 10:50:22.067788.067788 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007603168487548828 seconds
INFO 05-06 10:50:22.078171.078171 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011164188385009766 seconds
INFO 05-06 10:50:22.079368.079368 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010325908660888672 seconds
INFO 05-06 10:50:22.082831.082831 mlpmodule.py:2799] [fused_experts] gmm total=2.139ms E=32 S=4049 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.082464.082464 mlpmodule.py:2799] [fused_experts] gmm total=2.202ms E=32 S=2968 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.082373.082373 mlpmodule.py:2799] [fused_experts] gmm total=2.223ms E=32 S=4268 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.082318.082318 mlpmodule.py:2799] [fused_experts] gmm total=2.481ms E=32 S=5099 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.083351.083351 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003952503204345703 seconds
INFO 05-06 10:50:22.084779.084779 lmp.py:1496] [layer_moe_fused] to time: 5.4836273193359375e-05 seconds
INFO 05-06 10:50:22.084192.084192 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00033473968505859375 seconds
DEBUG 05-06 10:50:22.084964.084964 cuda_h.py:27] end *layer_moe_fused cost 21.119 ms
DEBUG 05-06 10:50:22.105078.105078 cuda_h.py:27] end prefill_layer cost 44.805 ms
DEBUG 05-06 10:50:22.105180.105180 lmp.py:841] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 10:50:22.105088.105088 lmp.py:824] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 10:50:22.108093.108093 cuda_h.py:27] end *sagl cost 3.125 ms
experts_cpu_alloc {'expert_ids': [127, 79, 15, 51, 11, 83, 25, 77, 29, 93, 46, 110, 30, 42, 82, 102, 112, 88, 8, 92, 96, 44, 16, 36], 'token_total': 332, 'token_per_expert': {127: 6, 79: 7, 15: 8, 51: 9, 11: 14, 83: 28, 25: 2, 77: 9, 29: 12, 93: 13, 46: 1, 110: 2, 30: 12, 42: 13, 82: 13, 102: 13, 112: 8, 88: 10, 8: 13, 92: 24, 96: 26, 44: 28, 16: 30, 36: 31}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 27, 31, 35, 39, 43, 47, 55, 59, 63, 67, 71, 75, 87, 91, 95, 99, 103, 107, 111, 119, 123], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4391, 'token_per_expert': {3: 567, 7: 518, 19: 29, 23: 465, 27: 228, 31: 94, 35: 90, 39: 230, 43: 253, 47: 143, 55: 80, 59: 43, 63: 135, 67: 38, 71: 171, 75: 406, 87: 40, 91: 41, 95: 342, 99: 43, 103: 102, 107: 176, 111: 65, 119: 53, 123: 39}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 21, 33, 37, 45, 49, 53, 57, 61, 65, 69, 73, 81, 85, 89, 97, 101, 109, 113, 117, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 29, 'token_total': 4025, 'token_per_expert': {1: 534, 5: 600, 9: 27, 13: 54, 17: 119, 21: 253, 33: 45, 37: 587, 45: 81, 49: 144, 53: 232, 57: 90, 61: 194, 65: 21, 69: 330, 73: 83, 81: 15, 85: 14, 89: 247, 97: 23, 101: 151, 109: 43, 113: 44, 117: 21, 125: 73}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 34, 38, 54, 58, 62, 66, 70, 74, 78, 86, 90, 94, 98, 106, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3354, 'token_per_expert': {2: 526, 6: 597, 10: 167, 14: 29, 18: 150, 22: 122, 34: 17, 38: 37, 54: 106, 58: 250, 62: 27, 66: 28, 70: 82, 74: 375, 78: 75, 86: 259, 90: 25, 94: 62, 98: 83, 106: 127, 114: 59, 118: 35, 122: 70, 126: 46}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 20, 24, 28, 32, 40, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 100, 104, 108, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 32, 'token_total': 4282, 'token_per_expert': {0: 574, 4: 634, 12: 109, 20: 138, 24: 439, 28: 143, 32: 33, 40: 209, 48: 65, 52: 234, 56: 170, 60: 33, 64: 136, 68: 134, 72: 238, 76: 456, 80: 84, 84: 70, 100: 41, 104: 53, 108: 93, 116: 51, 120: 63, 124: 82}}
INFO 05-06 10:50:22.112475.112475 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 2.524ms | allocate_experts_across_cpu_gpu: 0.462ms
INFO 05-06 10:50:22.112283.112283 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.845329284667969e-05 seconds
INFO 05-06 10:50:22.113127.113127 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006945133209228516 seconds
INFO 05-06 10:50:22.114494.114494 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008039474487304688 seconds
INFO 05-06 10:50:22.126023.126023 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011540651321411133 seconds
INFO 05-06 10:50:22.145301.145301 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.01909613609313965 seconds
INFO 05-06 10:50:22.147813.147813 mlpmodule.py:2799] [fused_experts] gmm total=1.924ms E=32 S=3408 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.147287.147287 mlpmodule.py:2799] [fused_experts] gmm total=2.134ms E=32 S=4463 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.147084.147084 mlpmodule.py:2799] [fused_experts] gmm total=2.143ms E=32 S=4061 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.148778.148778 mlpmodule.py:2799] [fused_experts] gmm total=2.353ms E=32 S=4452 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.149073.149073 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037899017333984375 seconds
INFO 05-06 10:50:22.149582.149582 lmp.py:1496] [layer_moe_fused] to time: 5.5789947509765625e-05 seconds
INFO 05-06 10:50:22.149199.149199 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003123283386230469 seconds
DEBUG 05-06 10:50:22.150091.150091 cuda_h.py:27] end *layer_moe_fused cost 40.767 ms
DEBUG 05-06 10:50:22.156964.156964 cuda_h.py:27] end prefill_layer cost 51.269 ms
DEBUG 05-06 10:50:22.156159.156159 lmp.py:841] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 10:50:22.156067.156067 lmp.py:824] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 10:50:22.158522.158522 cuda_h.py:27] end *sagl cost 1.941 ms
experts_cpu_alloc {'expert_ids': [115, 55, 63, 11, 27, 113, 117, 41, 21, 25, 9, 106, 22, 86, 126, 102, 42, 18, 28, 20, 16, 52, 44, 24, 96], 'token_total': 274, 'token_per_expert': {115: 2, 55: 9, 63: 12, 11: 17, 27: 18, 113: 5, 117: 5, 41: 12, 21: 13, 25: 21, 9: 25, 106: 3, 22: 4, 86: 4, 126: 6, 102: 9, 42: 16, 18: 20, 28: 1, 20: 2, 16: 9, 52: 11, 44: 13, 24: 17, 96: 20}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 31, 35, 39, 43, 47, 51, 59, 67, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 26, 'ideal_gpu_count': 26, 'keep_on_gpu': 26, 'hit_count_on_device': 31, 'token_total': 4010, 'token_per_expert': {3: 734, 7: 557, 15: 40, 19: 27, 23: 37, 31: 169, 35: 35, 39: 23, 43: 167, 47: 78, 51: 47, 59: 19, 67: 39, 71: 82, 75: 112, 83: 282, 87: 147, 91: 55, 95: 103, 99: 421, 103: 66, 107: 39, 111: 259, 119: 251, 123: 117, 127: 104}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 13, 17, 29, 33, 37, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 109, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4253, 'token_per_expert': {1: 626, 5: 596, 13: 58, 17: 122, 29: 58, 33: 198, 37: 82, 45: 31, 49: 116, 53: 221, 57: 78, 61: 171, 65: 159, 69: 110, 73: 53, 77: 210, 81: 150, 85: 290, 89: 25, 93: 137, 97: 56, 101: 211, 109: 38, 121: 379, 125: 78}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 26, 30, 34, 38, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 90, 94, 98, 110, 114, 118, 122], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 3540, 'token_per_expert': {2: 820, 6: 517, 10: 90, 14: 192, 26: 46, 30: 75, 34: 113, 38: 131, 46: 49, 50: 268, 54: 261, 58: 212, 62: 22, 66: 32, 70: 32, 74: 39, 78: 128, 82: 30, 90: 37, 94: 23, 98: 34, 110: 158, 114: 24, 118: 173, 122: 34}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 32, 36, 40, 48, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4307, 'token_per_expert': {0: 520, 4: 688, 8: 204, 12: 82, 32: 256, 36: 246, 40: 219, 48: 47, 56: 63, 60: 161, 64: 246, 68: 63, 72: 156, 76: 268, 80: 92, 84: 154, 88: 114, 92: 106, 100: 115, 104: 180, 108: 51, 112: 30, 116: 57, 120: 154, 124: 35}}
INFO 05-06 10:50:22.161013.161013 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.920ms | allocate_experts_across_cpu_gpu: 0.481ms
INFO 05-06 10:50:22.161734.161734 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.489059448242188e-05 seconds
INFO 05-06 10:50:22.162836.162836 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006659030914306641 seconds
INFO 05-06 10:50:22.163727.163727 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008389949798583984 seconds
INFO 05-06 10:50:22.173029.173029 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01049351692199707 seconds
INFO 05-06 10:50:22.175762.175762 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010437965393066406 seconds
INFO 05-06 10:50:22.177325.177325 mlpmodule.py:2799] [fused_experts] gmm total=1.911ms E=32 S=4334 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.177555.177555 mlpmodule.py:2799] [fused_experts] gmm total=2.005ms E=32 S=3602 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.177979.177979 mlpmodule.py:2799] [fused_experts] gmm total=2.522ms E=32 S=4068 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.177621.177621 mlpmodule.py:2799] [fused_experts] gmm total=2.512ms E=32 S=4380 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.179775.179775 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0038650035858154297 seconds
INFO 05-06 10:50:22.179224.179224 lmp.py:1496] [layer_moe_fused] to time: 5.4836273193359375e-05 seconds
INFO 05-06 10:50:22.179483.179483 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002911090850830078 seconds
DEBUG 05-06 10:50:22.180383.180383 cuda_h.py:27] end *layer_moe_fused cost 20.180 ms
DEBUG 05-06 10:50:22.186752.186752 cuda_h.py:27] end prefill_layer cost 29.534 ms
DEBUG 05-06 10:50:22.186377.186377 lmp.py:841] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 10:50:22.186047.186047 lmp.py:824] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 10:50:22.188925.188925 cuda_h.py:27] end *sagl cost 1.955 ms
experts_cpu_alloc {'expert_ids': [107, 71, 67, 115, 95, 87, 81, 85, 49, 113, 93, 105, 101, 82, 34, 62, 46, 66, 116, 124, 8, 100, 120, 56], 'token_total': 216, 'token_per_expert': {107: 3, 71: 4, 67: 9, 115: 9, 95: 10, 87: 13, 81: 1, 85: 2, 49: 5, 113: 6, 93: 8, 105: 11, 101: 12, 82: 4, 34: 5, 62: 8, 46: 9, 66: 9, 116: 3, 124: 3, 8: 12, 100: 13, 120: 18, 56: 39}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 75, 79, 83, 99, 103, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3864, 'token_per_expert': {3: 751, 7: 756, 11: 61, 15: 96, 19: 83, 23: 181, 27: 143, 31: 86, 35: 117, 39: 118, 43: 15, 47: 58, 51: 359, 55: 26, 59: 37, 63: 142, 75: 161, 79: 167, 83: 61, 99: 55, 103: 35, 111: 41, 119: 91, 123: 208, 127: 16}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 77, 89, 97, 109, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4158, 'token_per_expert': {1: 624, 5: 581, 9: 154, 13: 66, 17: 64, 21: 175, 25: 62, 29: 31, 33: 83, 37: 367, 41: 102, 45: 21, 53: 97, 57: 13, 61: 219, 65: 17, 69: 142, 73: 46, 77: 13, 89: 626, 97: 49, 109: 156, 117: 298, 121: 27, 125: 125}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 18, 22, 26, 30, 38, 42, 50, 54, 58, 70, 74, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3269, 'token_per_expert': {2: 606, 6: 522, 10: 87, 18: 28, 22: 51, 26: 125, 30: 18, 38: 489, 42: 29, 50: 165, 54: 16, 58: 38, 70: 16, 74: 9, 86: 33, 90: 71, 94: 19, 98: 55, 102: 139, 106: 116, 110: 33, 114: 24, 118: 61, 122: 500, 126: 19}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 16, 20, 24, 36, 40, 44, 48, 52, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 104, 108, 112], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4877, 'token_per_expert': {0: 593, 4: 539, 12: 50, 16: 142, 20: 42, 24: 299, 36: 58, 40: 171, 44: 431, 48: 100, 52: 726, 60: 65, 64: 375, 68: 42, 72: 52, 76: 119, 80: 181, 84: 57, 88: 114, 92: 433, 96: 124, 104: 85, 108: 40, 112: 39}}
INFO 05-06 10:50:22.190255.190255 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.558ms | allocate_experts_across_cpu_gpu: 0.471ms
INFO 05-06 10:50:22.190115.190115 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.131431579589844e-05 seconds
INFO 05-06 10:50:22.191688.191688 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006651878356933594 seconds
INFO 05-06 10:50:22.192955.192955 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007984638214111328 seconds
INFO 05-06 10:50:22.203925.203925 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010286808013916016 seconds
INFO 05-06 10:50:22.204115.204115 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010287761688232422 seconds
INFO 05-06 10:50:22.206206.206206 mlpmodule.py:2799] [fused_experts] gmm total=1.880ms E=32 S=4203 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.206542.206542 mlpmodule.py:2799] [fused_experts] gmm total=1.974ms E=32 S=3304 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.206136.206136 mlpmodule.py:2799] [fused_experts] gmm total=2.197ms E=32 S=3912 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.207030.207030 mlpmodule.py:2799] [fused_experts] gmm total=2.404ms E=32 S=4965 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.208350.208350 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003782987594604492 seconds
INFO 05-06 10:50:22.208236.208236 lmp.py:1496] [layer_moe_fused] to time: 5.4836273193359375e-05 seconds
INFO 05-06 10:50:22.208134.208134 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003781318664550781 seconds
DEBUG 05-06 10:50:22.209748.209748 cuda_h.py:27] end *layer_moe_fused cost 19.514 ms
DEBUG 05-06 10:50:22.215627.215627 cuda_h.py:27] end prefill_layer cost 29.001 ms
DEBUG 05-06 10:50:22.215776.215776 lmp.py:841] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 10:50:22.215446.215446 lmp.py:824] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 10:50:22.217370.217370 cuda_h.py:27] end *sagl cost 1.934 ms
experts_cpu_alloc {'expert_ids': [87, 99, 23, 67, 75, 29, 97, 17, 69, 25, 89, 61, 14, 22, 70, 106, 126, 34, 124, 16, 24, 104, 36, 120], 'token_total': 270, 'token_per_expert': {87: 2, 99: 3, 23: 4, 67: 4, 75: 5, 29: 5, 97: 5, 17: 10, 69: 12, 25: 13, 89: 14, 61: 16, 14: 1, 22: 3, 70: 6, 106: 10, 126: 16, 34: 20, 124: 6, 16: 7, 24: 21, 104: 26, 36: 30, 120: 31}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 35, 43, 47, 51, 55, 59, 63, 71, 79, 83, 91, 95, 103, 107, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3291, 'token_per_expert': {3: 707, 7: 518, 11: 12, 15: 82, 19: 32, 27: 123, 31: 15, 35: 18, 43: 120, 47: 21, 51: 6, 55: 49, 59: 229, 63: 360, 71: 84, 79: 90, 83: 62, 91: 9, 95: 30, 103: 37, 107: 460, 111: 36, 119: 6, 123: 178, 127: 7}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 21, 33, 37, 41, 45, 49, 53, 57, 65, 73, 77, 81, 85, 93, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 5115, 'token_per_expert': {1: 554, 5: 676, 9: 127, 13: 162, 21: 242, 33: 132, 37: 237, 41: 94, 45: 413, 49: 494, 53: 118, 57: 267, 65: 293, 73: 251, 77: 232, 81: 146, 85: 117, 93: 42, 101: 28, 105: 40, 109: 128, 113: 96, 117: 20, 121: 69, 125: 137}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 18, 26, 30, 38, 42, 46, 50, 54, 58, 62, 66, 74, 82, 86, 90, 94, 98, 102, 110, 114, 118, 122], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3450, 'token_per_expert': {2: 586, 6: 520, 10: 25, 18: 68, 26: 28, 30: 328, 38: 35, 42: 160, 46: 120, 50: 83, 54: 51, 58: 35, 62: 32, 66: 124, 74: 48, 82: 74, 86: 33, 90: 33, 94: 534, 98: 31, 102: 248, 110: 42, 114: 42, 118: 59, 122: 111}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 20, 28, 32, 40, 44, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 108, 112, 116], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4258, 'token_per_expert': {0: 546, 4: 672, 8: 199, 12: 59, 20: 76, 28: 155, 32: 134, 40: 194, 44: 134, 52: 67, 56: 179, 60: 62, 64: 82, 68: 698, 72: 84, 76: 66, 80: 42, 84: 46, 88: 153, 92: 125, 100: 93, 108: 158, 112: 125, 116: 109}}
INFO 05-06 10:50:22.220303.220303 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.969ms | allocate_experts_across_cpu_gpu: 0.469ms
INFO 05-06 10:50:22.220402.220402 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.059906005859375e-05 seconds
INFO 05-06 10:50:22.221538.221538 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006284713745117188 seconds
INFO 05-06 10:50:22.222951.222951 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008022785186767578 seconds
INFO 05-06 10:50:22.233151.233151 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.01122736930847168 seconds
INFO 05-06 10:50:22.234991.234991 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010504722595214844 seconds
INFO 05-06 10:50:22.236468.236468 mlpmodule.py:2799] [fused_experts] gmm total=1.975ms E=32 S=3309 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.237927.237927 mlpmodule.py:2799] [fused_experts] gmm total=2.145ms E=32 S=3506 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.237660.237660 mlpmodule.py:2799] [fused_experts] gmm total=2.312ms E=32 S=5190 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.237503.237503 mlpmodule.py:2799] [fused_experts] gmm total=2.432ms E=32 S=4379 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.238237.238237 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0038938522338867188 seconds
INFO 05-06 10:50:22.238885.238885 lmp.py:1496] [layer_moe_fused] to time: 5.435943603515625e-05 seconds
INFO 05-06 10:50:22.239629.239629 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00033354759216308594 seconds
DEBUG 05-06 10:50:22.239185.239185 cuda_h.py:27] end *layer_moe_fused cost 21.052 ms
DEBUG 05-06 10:50:22.246365.246365 cuda_h.py:27] end prefill_layer cost 30.795 ms
DEBUG 05-06 10:50:22.246420.246420 lmp.py:841] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 10:50:22.246852.246852 lmp.py:824] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 10:50:22.248215.248215 cuda_h.py:27] end *sagl cost 1.990 ms
experts_cpu_alloc {'expert_ids': [19, 107, 39, 99, 15, 89, 77, 9, 17, 25, 66, 98, 94, 22, 54, 74, 126, 28, 60, 104, 96, 52, 88], 'token_total': 150, 'token_per_expert': {19: 1, 107: 1, 39: 2, 99: 2, 15: 3, 89: 1, 77: 4, 9: 7, 17: 7, 25: 9, 66: 1, 98: 6, 94: 7, 22: 9, 54: 11, 74: 16, 126: 24, 28: 2, 60: 2, 104: 5, 96: 9, 52: 10, 88: 11}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 31, 35, 43, 47, 51, 55, 59, 67, 71, 75, 79, 83, 87, 95, 103, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 3378, 'token_per_expert': {3: 977, 7: 1066, 11: 250, 23: 8, 31: 76, 35: 68, 43: 26, 47: 8, 51: 108, 55: 28, 59: 26, 67: 79, 71: 22, 75: 67, 79: 54, 83: 129, 87: 41, 95: 62, 103: 105, 111: 56, 115: 18, 119: 28, 123: 39, 127: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 13, 21, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 81, 93, 97, 101, 105, 109, 113, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 4454, 'token_per_expert': {1: 1215, 5: 1305, 13: 71, 21: 25, 29: 149, 33: 65, 37: 118, 41: 156, 45: 29, 53: 146, 57: 91, 61: 108, 65: 293, 69: 13, 73: 150, 81: 46, 93: 10, 97: 84, 101: 9, 105: 183, 109: 102, 113: 11, 121: 23, 125: 52}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 18, 26, 30, 34, 38, 42, 46, 50, 58, 62, 70, 78, 82, 86, 90, 102, 106, 110, 114, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4433, 'token_per_expert': {2: 1036, 6: 1286, 10: 75, 18: 185, 26: 173, 30: 80, 34: 63, 38: 40, 42: 36, 46: 179, 50: 27, 58: 46, 62: 114, 70: 50, 78: 375, 82: 39, 86: 74, 90: 101, 102: 82, 106: 27, 110: 97, 114: 25, 118: 44, 122: 179}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 32, 36, 40, 44, 48, 56, 64, 68, 72, 76, 80, 84, 92, 100, 112, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3969, 'token_per_expert': {0: 994, 4: 1045, 8: 110, 12: 70, 16: 13, 20: 28, 24: 38, 32: 30, 36: 77, 40: 19, 44: 33, 48: 221, 56: 18, 64: 12, 68: 77, 72: 83, 76: 129, 80: 41, 84: 91, 92: 153, 100: 256, 112: 142, 120: 164, 124: 125}}
INFO 05-06 10:50:22.250274.250274 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.560ms | allocate_experts_across_cpu_gpu: 0.467ms
INFO 05-06 10:50:22.251359.251359 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.7738037109375e-05 seconds
INFO 05-06 10:50:22.251086.251086 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007581710815429688 seconds
INFO 05-06 10:50:22.252062.252062 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007932186126708984 seconds
INFO 05-06 10:50:22.263904.263904 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010793924331665039 seconds
INFO 05-06 10:50:22.264920.264920 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009770393371582031 seconds
INFO 05-06 10:50:22.267076.267076 mlpmodule.py:2799] [fused_experts] gmm total=2.086ms E=32 S=3387 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.267571.267571 mlpmodule.py:2799] [fused_experts] gmm total=2.144ms E=32 S=4507 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.267527.267527 mlpmodule.py:2799] [fused_experts] gmm total=2.267ms E=32 S=4482 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.267814.267814 mlpmodule.py:2799] [fused_experts] gmm total=2.570ms E=32 S=4008 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.269604.269604 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004536151885986328 seconds
INFO 05-06 10:50:22.269821.269821 lmp.py:1496] [layer_moe_fused] to time: 5.3882598876953125e-05 seconds
INFO 05-06 10:50:22.269357.269357 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00025010108947753906 seconds
DEBUG 05-06 10:50:22.270230.270230 cuda_h.py:27] end *layer_moe_fused cost 20.696 ms
DEBUG 05-06 10:50:22.276557.276557 cuda_h.py:27] end prefill_layer cost 30.147 ms
DEBUG 05-06 10:50:22.276321.276321 lmp.py:841] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 10:50:22.276753.276753 lmp.py:824] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 10:50:22.278073.278073 cuda_h.py:27] end *sagl cost 1.921 ms
experts_cpu_alloc {'expert_ids': [91, 23, 71, 67, 87, 95, 39, 29, 13, 21, 49, 77, 61, 22, 114, 18, 78, 106, 54, 56, 80, 12, 36, 104, 96], 'token_total': 115, 'token_per_expert': {91: 3, 23: 4, 71: 5, 67: 8, 87: 9, 95: 9, 39: 11, 29: 1, 13: 2, 21: 3, 49: 4, 77: 4, 61: 5, 22: 1, 114: 1, 18: 3, 78: 4, 106: 7, 54: 11, 56: 1, 80: 1, 12: 2, 36: 3, 104: 3, 96: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 27, 31, 35, 43, 47, 51, 55, 59, 63, 75, 79, 83, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4193, 'token_per_expert': {3: 962, 7: 1022, 11: 67, 15: 91, 19: 62, 27: 20, 31: 40, 35: 311, 43: 90, 47: 38, 51: 22, 55: 140, 59: 231, 63: 23, 75: 115, 79: 25, 83: 25, 99: 58, 103: 224, 107: 44, 111: 74, 115: 86, 119: 188, 123: 104, 127: 131}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 17, 25, 33, 37, 41, 45, 53, 57, 65, 69, 73, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3195, 'token_per_expert': {1: 1021, 5: 964, 9: 18, 17: 6, 25: 20, 33: 56, 37: 12, 41: 30, 45: 52, 53: 142, 57: 22, 65: 13, 69: 49, 73: 166, 81: 7, 85: 40, 89: 97, 93: 175, 97: 10, 101: 46, 105: 6, 109: 33, 113: 34, 117: 147, 125: 29}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 26, 30, 34, 38, 42, 46, 58, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 110, 118, 122, 126], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3910, 'token_per_expert': {2: 965, 6: 963, 10: 17, 14: 13, 26: 60, 30: 86, 34: 28, 38: 121, 42: 57, 46: 93, 58: 58, 62: 28, 66: 95, 70: 95, 74: 275, 82: 99, 86: 135, 90: 113, 94: 113, 98: 21, 102: 59, 110: 23, 118: 90, 122: 17, 126: 286}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 40, 44, 48, 60, 64, 68, 72, 76, 84, 88, 92, 100, 108, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4971, 'token_per_expert': {0: 985, 4: 960, 8: 139, 16: 46, 20: 25, 24: 272, 28: 73, 32: 40, 40: 53, 44: 48, 48: 51, 60: 13, 64: 354, 68: 161, 72: 299, 76: 78, 84: 19, 88: 79, 92: 160, 100: 638, 108: 116, 112: 47, 116: 127, 120: 127, 124: 61}}
INFO 05-06 10:50:22.281773.281773 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.855ms | allocate_experts_across_cpu_gpu: 0.474ms
INFO 05-06 10:50:22.281004.281004 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.036064147949219e-05 seconds
INFO 05-06 10:50:22.282802.282802 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.000644683837890625 seconds
INFO 05-06 10:50:22.283089.283089 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008158683776855469 seconds
INFO 05-06 10:50:22.293413.293413 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010761022567749023 seconds
INFO 05-06 10:50:22.295693.295693 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009632110595703125 seconds
INFO 05-06 10:50:22.297369.297369 mlpmodule.py:2799] [fused_experts] gmm total=2.147ms E=32 S=3214 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.297962.297962 mlpmodule.py:2799] [fused_experts] gmm total=2.223ms E=32 S=3937 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.297848.297848 mlpmodule.py:2799] [fused_experts] gmm total=2.446ms E=32 S=4242 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.298064.298064 mlpmodule.py:2799] [fused_experts] gmm total=2.579ms E=32 S=4991 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.299367.299367 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004037380218505859 seconds
INFO 05-06 10:50:22.299399.299399 lmp.py:1496] [layer_moe_fused] to time: 5.602836608886719e-05 seconds
INFO 05-06 10:50:22.299330.299330 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00035643577575683594 seconds
DEBUG 05-06 10:50:22.300890.300890 cuda_h.py:27] end *layer_moe_fused cost 20.448 ms
DEBUG 05-06 10:50:22.306014.306014 cuda_h.py:27] end prefill_layer cost 29.749 ms
DEBUG 05-06 10:50:22.306539.306539 lmp.py:841] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 10:50:22.306448.306448 lmp.py:824] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 10:50:22.309606.309606 cuda_h.py:27] end *sagl cost 2.852 ms
experts_cpu_alloc {'expert_ids': [119, 111, 15, 55, 127, 27, 13, 69, 93, 77, 49, 70, 114, 50, 82, 74, 54, 88, 20, 92, 28, 64, 60, 68], 'token_total': 153, 'token_per_expert': {119: 1, 111: 3, 15: 4, 55: 5, 127: 8, 27: 9, 13: 3, 69: 3, 93: 4, 77: 5, 49: 8, 70: 1, 114: 1, 50: 3, 82: 5, 74: 7, 54: 8, 88: 6, 20: 8, 92: 8, 28: 9, 64: 10, 60: 15, 68: 19}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 39, 43, 47, 51, 59, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 115, 123], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4040, 'token_per_expert': {3: 1186, 7: 1030, 11: 13, 19: 35, 23: 22, 31: 74, 35: 117, 39: 260, 43: 192, 47: 154, 51: 24, 59: 53, 67: 231, 71: 61, 75: 24, 79: 148, 83: 74, 87: 54, 91: 42, 95: 23, 99: 14, 103: 26, 107: 18, 115: 69, 123: 96}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 17, 21, 25, 29, 33, 37, 41, 53, 57, 61, 65, 73, 81, 85, 89, 97, 101, 105, 109, 117, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 4725, 'token_per_expert': {1: 1192, 5: 1187, 9: 38, 17: 79, 21: 406, 25: 188, 29: 168, 33: 46, 37: 194, 41: 9, 53: 20, 57: 29, 61: 195, 65: 247, 73: 50, 81: 9, 85: 120, 89: 27, 97: 141, 101: 13, 105: 65, 109: 100, 117: 61, 125: 141}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 58, 62, 66, 78, 86, 90, 98, 102, 106, 110, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3807, 'token_per_expert': {2: 1108, 6: 1064, 10: 28, 14: 15, 18: 120, 22: 44, 26: 84, 30: 68, 34: 43, 38: 36, 42: 40, 46: 182, 58: 12, 62: 19, 66: 21, 78: 101, 86: 291, 90: 90, 98: 181, 102: 10, 106: 69, 110: 8, 118: 141, 122: 32}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 16, 24, 32, 36, 40, 44, 48, 52, 56, 72, 76, 80, 84, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3659, 'token_per_expert': {0: 1030, 4: 1028, 8: 87, 12: 26, 16: 95, 24: 60, 32: 35, 36: 22, 40: 23, 44: 128, 48: 22, 52: 24, 56: 345, 72: 92, 76: 54, 80: 65, 84: 92, 100: 64, 104: 87, 108: 118, 112: 41, 116: 58, 120: 30, 124: 33}}
INFO 05-06 10:50:22.313539.313539 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 2.560ms | allocate_experts_across_cpu_gpu: 0.456ms
INFO 05-06 10:50:22.313816.313816 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.012222290039062e-05 seconds
INFO 05-06 10:50:22.314552.314552 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006020069122314453 seconds
INFO 05-06 10:50:22.315502.315502 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008115768432617188 seconds
INFO 05-06 10:50:22.326280.326280 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010286092758178711 seconds
INFO 05-06 10:50:22.327593.327593 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009546279907226562 seconds
INFO 05-06 10:50:22.329600.329600 mlpmodule.py:2799] [fused_experts] gmm total=2.142ms E=32 S=4748 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.329069.329069 mlpmodule.py:2799] [fused_experts] gmm total=2.367ms E=32 S=4070 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.329352.329352 mlpmodule.py:2799] [fused_experts] gmm total=2.410ms E=32 S=3832 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.329312.329312 mlpmodule.py:2799] [fused_experts] gmm total=2.478ms E=32 S=3734 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.331261.331261 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004070758819580078 seconds
INFO 05-06 10:50:22.331472.331472 lmp.py:1496] [layer_moe_fused] to time: 5.245208740234375e-05 seconds
INFO 05-06 10:50:22.331616.331616 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0004134178161621094 seconds
DEBUG 05-06 10:50:22.332853.332853 cuda_h.py:27] end *layer_moe_fused cost 21.684 ms
DEBUG 05-06 10:50:22.338352.338352 cuda_h.py:27] end prefill_layer cost 31.944 ms
DEBUG 05-06 10:50:22.338732.338732 lmp.py:841] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 10:50:22.338641.338641 lmp.py:824] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 10:50:22.340228.340228 cuda_h.py:27] end *sagl cost 1.922 ms
experts_cpu_alloc {'expert_ids': [51, 123, 59, 95, 55, 69, 85, 101, 113, 117, 125, 22, 54, 58, 78, 126, 10, 88, 24, 112, 84, 116, 80, 76], 'token_total': 106, 'token_per_expert': {51: 2, 123: 4, 59: 5, 95: 6, 55: 8, 69: 1, 85: 1, 101: 1, 113: 1, 117: 1, 125: 1, 22: 2, 54: 2, 58: 5, 78: 8, 126: 8, 10: 9, 88: 1, 24: 2, 112: 6, 84: 7, 116: 7, 80: 8, 76: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 47, 63, 67, 71, 75, 79, 83, 87, 91, 99, 107, 111, 115, 119, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3918, 'token_per_expert': {3: 1028, 7: 1082, 11: 215, 15: 10, 19: 147, 23: 99, 27: 223, 31: 18, 35: 77, 43: 50, 47: 18, 63: 212, 67: 114, 71: 161, 75: 33, 79: 25, 83: 83, 87: 13, 91: 154, 99: 10, 107: 18, 111: 29, 115: 13, 119: 25, 127: 61}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 45, 49, 53, 57, 61, 65, 73, 77, 81, 89, 93, 97, 105, 109, 121], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4185, 'token_per_expert': {1: 1102, 5: 1079, 9: 30, 13: 55, 17: 69, 21: 5, 25: 5, 29: 86, 33: 199, 37: 67, 45: 131, 49: 31, 53: 26, 57: 24, 61: 12, 65: 11, 73: 132, 77: 147, 81: 42, 89: 11, 93: 2, 97: 297, 105: 30, 109: 71, 121: 521}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 18, 26, 30, 34, 38, 42, 46, 50, 62, 66, 70, 74, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 4264, 'token_per_expert': {2: 1027, 6: 1227, 18: 21, 26: 21, 30: 72, 34: 156, 38: 15, 42: 28, 46: 16, 50: 65, 62: 23, 66: 20, 70: 274, 74: 58, 82: 62, 86: 78, 90: 469, 94: 130, 98: 107, 102: 19, 106: 41, 110: 85, 114: 134, 118: 52, 122: 64}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 16, 20, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 92, 96, 100, 104, 108, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3911, 'token_per_expert': {0: 1024, 4: 1127, 8: 32, 12: 169, 16: 115, 20: 39, 28: 25, 32: 40, 36: 56, 40: 37, 44: 166, 48: 85, 52: 190, 56: 145, 60: 78, 64: 296, 68: 13, 92: 28, 96: 21, 100: 47, 104: 21, 108: 107, 120: 32, 124: 18}}
INFO 05-06 10:50:22.343291.343291 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.902ms | allocate_experts_across_cpu_gpu: 0.467ms
INFO 05-06 10:50:22.343045.343045 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.083747863769531e-05 seconds
INFO 05-06 10:50:22.344351.344351 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006704330444335938 seconds
INFO 05-06 10:50:22.345565.345565 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007963180541992188 seconds
INFO 05-06 10:50:22.355494.355494 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010847091674804688 seconds
INFO 05-06 10:50:22.356013.356013 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009634494781494141 seconds
INFO 05-06 10:50:22.359968.359968 mlpmodule.py:2799] [fused_experts] gmm total=2.011ms E=32 S=3943 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.359577.359577 mlpmodule.py:2799] [fused_experts] gmm total=2.167ms E=32 S=4191 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.359733.359733 mlpmodule.py:2799] [fused_experts] gmm total=2.235ms E=32 S=4298 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.359451.359451 mlpmodule.py:2799] [fused_experts] gmm total=2.591ms E=32 S=3952 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.360770.360770 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003744840621948242 seconds
INFO 05-06 10:50:22.360272.360272 lmp.py:1496] [layer_moe_fused] to time: 5.4836273193359375e-05 seconds
INFO 05-06 10:50:22.361047.361047 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002856254577636719 seconds
DEBUG 05-06 10:50:22.361257.361257 cuda_h.py:27] end *layer_moe_fused cost 20.131 ms
DEBUG 05-06 10:50:22.368167.368167 cuda_h.py:27] end prefill_layer cost 29.769 ms
DEBUG 05-06 10:50:22.368024.368024 lmp.py:841] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 10:50:22.368217.368217 lmp.py:824] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 10:50:22.370392.370392 cuda_h.py:27] end *sagl cost 1.908 ms
experts_cpu_alloc {'expert_ids': [95, 75, 115, 15, 55, 27, 37, 57, 61, 65, 101, 54, 62, 94, 98, 30, 86, 102, 40, 84, 96, 76, 108, 24], 'token_total': 102, 'token_per_expert': {95: 1, 75: 3, 115: 4, 15: 6, 55: 6, 27: 11, 37: 1, 57: 4, 61: 4, 65: 4, 101: 5, 54: 1, 62: 1, 94: 1, 98: 1, 30: 2, 86: 3, 102: 3, 40: 1, 84: 1, 96: 2, 76: 10, 108: 13, 24: 14}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 31, 35, 39, 43, 47, 51, 63, 67, 71, 79, 83, 87, 91, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 31, 'token_total': 3963, 'token_per_expert': {3: 1160, 7: 1084, 11: 71, 19: 49, 23: 13, 31: 22, 35: 223, 39: 52, 43: 28, 47: 26, 51: 33, 63: 66, 67: 78, 71: 65, 79: 45, 83: 117, 87: 42, 91: 82, 99: 26, 103: 14, 107: 335, 111: 73, 119: 15, 123: 231, 127: 13}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 41, 45, 49, 53, 69, 73, 77, 81, 85, 89, 93, 97, 109, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 30, 'token_total': 3449, 'token_per_expert': {1: 1032, 5: 1056, 9: 35, 13: 18, 17: 7, 21: 12, 25: 36, 29: 22, 33: 9, 41: 63, 45: 213, 49: 57, 53: 8, 69: 171, 73: 16, 77: 14, 81: 6, 85: 209, 89: 80, 93: 60, 97: 61, 109: 39, 117: 193, 121: 6, 125: 26}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 38, 42, 46, 50, 58, 66, 70, 74, 78, 82, 90, 106, 110, 114, 118, 122, 126], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 32, 'token_total': 4420, 'token_per_expert': {2: 1296, 6: 1044, 10: 92, 14: 58, 18: 219, 22: 7, 26: 31, 34: 62, 38: 23, 42: 14, 46: 11, 50: 57, 58: 531, 66: 15, 70: 163, 74: 19, 78: 23, 82: 71, 90: 113, 106: 49, 110: 351, 114: 102, 118: 30, 122: 10, 126: 29}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 16, 32, 36, 44, 48, 52, 56, 60, 64, 68, 72, 80, 88, 92, 100, 104, 112, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4450, 'token_per_expert': {0: 1061, 4: 1030, 8: 25, 12: 15, 16: 611, 32: 14, 36: 64, 44: 75, 48: 30, 52: 187, 56: 53, 60: 171, 64: 108, 68: 366, 72: 33, 80: 157, 88: 42, 92: 19, 100: 72, 104: 157, 112: 21, 116: 55, 120: 62, 124: 22}}
INFO 05-06 10:50:22.372947.372947 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.556ms | allocate_experts_across_cpu_gpu: 0.460ms
INFO 05-06 10:50:22.372900.372900 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.1552734375e-05 seconds
INFO 05-06 10:50:22.373066.373066 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006532669067382812 seconds
INFO 05-06 10:50:22.374849.374849 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007941722869873047 seconds
INFO 05-06 10:50:22.385021.385021 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010788202285766602 seconds
INFO 05-06 10:50:22.386771.386771 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009579658508300781 seconds
INFO 05-06 10:50:22.388234.388234 mlpmodule.py:2799] [fused_experts] gmm total=2.253ms E=32 S=4432 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.389212.389212 mlpmodule.py:2799] [fused_experts] gmm total=2.421ms E=32 S=3467 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.389033.389033 mlpmodule.py:2799] [fused_experts] gmm total=2.662ms E=32 S=3994 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.389195.389195 mlpmodule.py:2799] [fused_experts] gmm total=2.548ms E=32 S=4491 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.390954.390954 lmp.py:1484] [layer_moe_fused] experts compute time: 0.003990650177001953 seconds
INFO 05-06 10:50:22.390211.390211 lmp.py:1496] [layer_moe_fused] to time: 5.4836273193359375e-05 seconds
INFO 05-06 10:50:22.390641.390641 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.000240325927734375 seconds
DEBUG 05-06 10:50:22.391182.391182 cuda_h.py:27] end *layer_moe_fused cost 19.908 ms
DEBUG 05-06 10:50:22.398032.398032 cuda_h.py:27] end prefill_layer cost 29.666 ms
DEBUG 05-06 10:50:22.398412.398412 lmp.py:841] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 10:50:22.398559.398559 lmp.py:824] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 10:50:22.400702.400702 cuda_h.py:27] end *sagl cost 1.952 ms
experts_cpu_alloc {'expert_ids': [119, 11, 127, 55, 107, 47, 83, 93, 53, 33, 117, 121, 22, 82, 98, 58, 34, 12, 16, 32, 48, 100, 28], 'token_total': 86, 'token_per_expert': {119: 1, 11: 2, 127: 2, 55: 4, 107: 6, 47: 7, 83: 9, 93: 1, 53: 3, 33: 4, 117: 4, 121: 4, 22: 1, 82: 2, 98: 2, 58: 3, 34: 5, 12: 1, 16: 2, 32: 3, 48: 5, 100: 7, 28: 8}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 43, 51, 59, 63, 67, 71, 75, 79, 87, 91, 95, 99, 103, 111, 115, 123], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4119, 'token_per_expert': {3: 1072, 7: 1024, 15: 102, 19: 93, 23: 10, 27: 203, 31: 13, 35: 20, 43: 182, 51: 76, 59: 67, 63: 11, 67: 30, 71: 17, 75: 40, 79: 49, 87: 303, 91: 14, 95: 245, 99: 54, 103: 57, 111: 315, 115: 22, 123: 100}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 17, 25, 29, 37, 41, 45, 49, 57, 61, 65, 73, 77, 81, 85, 89, 97, 105, 109, 113, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 4876, 'token_per_expert': {1: 1045, 5: 1043, 9: 7, 13: 29, 17: 290, 25: 44, 29: 44, 37: 44, 41: 27, 45: 53, 49: 98, 57: 48, 61: 42, 65: 237, 73: 151, 77: 31, 81: 29, 85: 605, 89: 407, 97: 68, 105: 85, 109: 19, 113: 409, 125: 21}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 26, 30, 38, 42, 46, 50, 54, 66, 70, 74, 78, 86, 90, 102, 110, 114, 118, 122, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 3506, 'token_per_expert': {2: 1039, 6: 1024, 10: 50, 14: 68, 18: 17, 26: 52, 30: 46, 38: 45, 42: 26, 46: 9, 50: 82, 54: 8, 66: 71, 70: 122, 74: 20, 78: 108, 86: 83, 90: 75, 102: 86, 110: 20, 114: 336, 118: 28, 122: 20, 126: 71}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 20, 24, 36, 40, 44, 52, 56, 60, 68, 72, 76, 80, 84, 88, 92, 96, 104, 108, 112, 116, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3797, 'token_per_expert': {0: 1045, 4: 1034, 8: 28, 20: 352, 24: 246, 36: 32, 40: 13, 44: 20, 52: 88, 56: 84, 60: 89, 68: 31, 72: 31, 76: 87, 80: 13, 84: 222, 88: 51, 92: 11, 96: 34, 104: 120, 108: 41, 112: 38, 116: 17, 124: 70}}
INFO 05-06 10:50:22.402451.402451 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.861ms | allocate_experts_across_cpu_gpu: 0.451ms
INFO 05-06 10:50:22.403444.403444 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.654594421386719e-05 seconds
INFO 05-06 10:50:22.403155.403155 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006535053253173828 seconds
INFO 05-06 10:50:22.404250.404250 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0007770061492919922 seconds
INFO 05-06 10:50:22.415405.415405 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.010496854782104492 seconds
INFO 05-06 10:50:22.416237.416237 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001020193099975586 seconds
INFO 05-06 10:50:22.418383.418383 mlpmodule.py:2799] [fused_experts] gmm total=1.933ms E=32 S=4892 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.418671.418671 mlpmodule.py:2799] [fused_experts] gmm total=1.996ms E=32 S=3519 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.418548.418548 mlpmodule.py:2799] [fused_experts] gmm total=2.178ms E=32 S=4150 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.419332.419332 mlpmodule.py:2799] [fused_experts] gmm total=2.240ms E=32 S=3823 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.420649.420649 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037381649017333984 seconds
INFO 05-06 10:50:22.420191.420191 lmp.py:1496] [layer_moe_fused] to time: 5.412101745605469e-05 seconds
INFO 05-06 10:50:22.420348.420348 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00038909912109375 seconds
DEBUG 05-06 10:50:22.421745.421745 cuda_h.py:27] end *layer_moe_fused cost 19.965 ms
DEBUG 05-06 10:50:22.427381.427381 cuda_h.py:27] end prefill_layer cost 29.496 ms
DEBUG 05-06 10:50:22.427145.427145 lmp.py:841] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 10:50:22.427054.427054 lmp.py:824] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 10:50:22.429437.429437 cuda_h.py:27] end *sagl cost 1.903 ms
experts_cpu_alloc {'expert_ids': [99, 67, 39, 63, 47, 59, 11, 81, 9, 17, 77, 89, 93, 22, 34, 30, 86, 110, 126, 72, 16, 44, 84, 96], 'token_total': 153, 'token_per_expert': {99: 1, 67: 4, 39: 11, 63: 11, 47: 13, 59: 14, 11: 16, 81: 1, 9: 2, 17: 2, 77: 2, 89: 2, 93: 2, 22: 3, 34: 3, 30: 6, 86: 6, 110: 9, 126: 11, 72: 3, 16: 5, 44: 7, 84: 9, 96: 10}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 43, 51, 55, 71, 75, 79, 83, 87, 91, 95, 103, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 4604, 'token_per_expert': {3: 1122, 7: 1090, 15: 17, 19: 21, 23: 45, 27: 19, 31: 127, 35: 89, 43: 271, 51: 130, 55: 31, 71: 24, 75: 61, 79: 129, 83: 69, 87: 353, 91: 25, 95: 170, 103: 213, 111: 120, 115: 188, 119: 90, 123: 138, 127: 62}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 13, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 85, 97, 101, 105, 109, 113, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3819, 'token_per_expert': {1: 1117, 5: 1025, 13: 125, 21: 29, 25: 129, 29: 6, 33: 202, 37: 158, 41: 90, 45: 259, 49: 32, 53: 71, 57: 4, 61: 91, 65: 185, 69: 7, 85: 43, 97: 3, 101: 3, 105: 48, 109: 84, 113: 13, 121: 79, 125: 16}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 26, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 90, 94, 98, 106, 114, 118, 122], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3832, 'token_per_expert': {2: 1027, 6: 1033, 10: 24, 14: 81, 18: 74, 26: 41, 42: 76, 46: 122, 50: 263, 54: 50, 58: 25, 62: 130, 66: 69, 70: 114, 74: 25, 78: 155, 82: 161, 90: 40, 94: 44, 98: 70, 106: 48, 114: 44, 118: 63, 122: 53}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 20, 24, 28, 32, 36, 40, 48, 56, 60, 64, 68, 76, 80, 88, 100, 108, 112, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 29, 'token_total': 3976, 'token_per_expert': {0: 1038, 4: 1065, 8: 90, 12: 86, 20: 43, 24: 202, 28: 70, 32: 15, 36: 107, 40: 41, 48: 152, 56: 41, 60: 17, 64: 99, 68: 17, 76: 141, 80: 19, 88: 188, 100: 187, 108: 71, 112: 46, 116: 11, 120: 217, 124: 13}}
INFO 05-06 10:50:22.432450.432450 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.632ms | allocate_experts_across_cpu_gpu: 0.453ms
INFO 05-06 10:50:22.432649.432649 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 9.417533874511719e-05 seconds
INFO 05-06 10:50:22.433045.433045 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008709430694580078 seconds
INFO 05-06 10:50:22.434417.434417 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0009133815765380859 seconds
INFO 05-06 10:50:22.445315.445315 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011073112487792969 seconds
INFO 05-06 10:50:22.446244.446244 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001081705093383789 seconds
INFO 05-06 10:50:22.449528.449528 mlpmodule.py:2799] [fused_experts] gmm total=1.955ms E=32 S=3870 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.449172.449172 mlpmodule.py:2799] [fused_experts] gmm total=2.251ms E=32 S=3830 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.449971.449971 mlpmodule.py:2799] [fused_experts] gmm total=2.445ms E=32 S=4674 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.449070.449070 mlpmodule.py:2799] [fused_experts] gmm total=2.421ms E=32 S=4010 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.450909.450909 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037381649017333984 seconds
INFO 05-06 10:50:22.450053.450053 lmp.py:1496] [layer_moe_fused] to time: 5.221366882324219e-05 seconds
INFO 05-06 10:50:22.451970.451970 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0003581047058105469 seconds
DEBUG 05-06 10:50:22.452451.452451 cuda_h.py:27] end *layer_moe_fused cost 21.062 ms
DEBUG 05-06 10:50:22.458485.458485 cuda_h.py:27] end prefill_layer cost 30.258 ms
DEBUG 05-06 10:50:22.458640.458640 lmp.py:841] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 10:50:22.458217.458217 lmp.py:824] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 10:50:22.460614.460614 cuda_h.py:27] end *sagl cost 2.095 ms
experts_cpu_alloc {'expert_ids': [63, 83, 103, 19, 31, 35, 99, 45, 93, 61, 17, 21, 29, 26, 38, 102, 114, 28, 80, 8, 124, 56], 'token_total': 44, 'token_per_expert': {63: 1, 83: 1, 103: 1, 19: 2, 31: 2, 35: 2, 99: 3, 45: 1, 93: 1, 61: 2, 17: 3, 21: 4, 29: 4, 26: 1, 38: 1, 102: 1, 114: 1, 28: 1, 80: 2, 8: 3, 124: 3, 56: 4}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 23, 39, 43, 47, 51, 55, 59, 67, 71, 75, 79, 87, 91, 95, 111, 115, 119, 123, 127], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 30, 'token_total': 4325, 'token_per_expert': {3: 1031, 7: 1027, 11: 113, 15: 5, 23: 38, 39: 9, 43: 25, 47: 147, 51: 7, 55: 64, 59: 6, 67: 5, 71: 143, 75: 138, 79: 25, 87: 5, 91: 221, 95: 40, 111: 784, 115: 288, 119: 114, 123: 68, 127: 22}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 13, 33, 37, 49, 53, 57, 65, 69, 73, 77, 81, 85, 89, 97, 101, 105, 109, 113, 117, 121], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 29, 'token_total': 3952, 'token_per_expert': {1: 1062, 5: 1077, 9: 32, 13: 76, 33: 30, 37: 61, 49: 576, 53: 143, 57: 277, 65: 28, 69: 38, 73: 12, 77: 100, 81: 11, 85: 51, 89: 67, 97: 45, 101: 75, 105: 28, 109: 13, 113: 79, 117: 29, 121: 42}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 18, 22, 30, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 90, 94, 98, 106, 110, 118, 122, 126], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 27, 'token_total': 3227, 'token_per_expert': {2: 1024, 6: 1026, 18: 55, 22: 94, 30: 57, 46: 111, 50: 12, 54: 7, 58: 9, 62: 48, 66: 4, 70: 113, 74: 26, 78: 137, 82: 3, 90: 207, 94: 34, 98: 23, 106: 59, 110: 117, 118: 4, 122: 20, 126: 37}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 20, 24, 32, 36, 40, 44, 48, 52, 60, 68, 72, 76, 84, 88, 92, 100, 104, 108, 112, 120], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 28, 'token_total': 4836, 'token_per_expert': {0: 1024, 4: 1024, 12: 727, 20: 614, 24: 45, 32: 87, 36: 10, 40: 198, 44: 31, 48: 20, 52: 145, 60: 40, 68: 112, 72: 5, 76: 311, 84: 50, 88: 35, 92: 10, 100: 36, 104: 65, 108: 8, 112: 235, 120: 4}}
INFO 05-06 10:50:22.463655.463655 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.794ms | allocate_experts_across_cpu_gpu: 0.616ms
INFO 05-06 10:50:22.463867.463867 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.749961853027344e-05 seconds
INFO 05-06 10:50:22.464213.464213 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0007412433624267578 seconds
INFO 05-06 10:50:22.465125.465125 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008440017700195312 seconds
INFO 05-06 10:50:22.477395.477395 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.012144088745117188 seconds
INFO 05-06 10:50:22.478592.478592 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010309219360351562 seconds
INFO 05-06 10:50:22.480475.480475 mlpmodule.py:2799] [fused_experts] gmm total=2.014ms E=32 S=4337 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.481286.481286 mlpmodule.py:2799] [fused_experts] gmm total=2.025ms E=32 S=3231 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.481689.481689 mlpmodule.py:2799] [fused_experts] gmm total=2.231ms E=32 S=3967 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.481074.481074 mlpmodule.py:2799] [fused_experts] gmm total=2.293ms E=32 S=4849 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.482164.482164 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037436485290527344 seconds
INFO 05-06 10:50:22.482402.482402 lmp.py:1496] [layer_moe_fused] to time: 6.604194641113281e-05 seconds
INFO 05-06 10:50:22.482163.482163 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.0002541542053222656 seconds
DEBUG 05-06 10:50:22.484063.484063 cuda_h.py:27] end *layer_moe_fused cost 22.335 ms
DEBUG 05-06 10:50:22.489114.489114 cuda_h.py:27] end prefill_layer cost 31.727 ms
DEBUG 05-06 10:50:22.490501.490501 lmp.py:841] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 10:50:22.490509.490509 lmp.py:824] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 10:50:22.493186.493186 cuda_h.py:27] end *sagl cost 3.431 ms
experts_cpu_alloc {'expert_ids': [79, 39, 111, 127, 17, 37, 65, 125, 33, 13, 118, 38, 122, 126, 34, 70, 74, 12, 72, 112, 68, 100, 40], 'token_total': 132, 'token_per_expert': {79: 2, 39: 4, 111: 5, 127: 7, 17: 3, 37: 8, 65: 9, 125: 10, 33: 12, 13: 23, 118: 1, 38: 3, 122: 3, 126: 5, 34: 6, 70: 6, 74: 6, 12: 1, 72: 1, 112: 2, 68: 3, 100: 3, 40: 9}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 43, 55, 63, 67, 71, 75, 83, 87, 91, 95, 99, 107, 115, 119, 123], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 28, 'token_total': 4408, 'token_per_expert': {3: 1054, 7: 1409, 11: 34, 15: 55, 19: 216, 23: 131, 27: 143, 31: 26, 35: 25, 43: 209, 55: 10, 63: 18, 67: 65, 71: 117, 75: 17, 83: 25, 87: 12, 91: 348, 95: 63, 99: 246, 107: 58, 115: 43, 119: 21, 123: 63}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 21, 25, 29, 49, 53, 57, 61, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 3802, 'token_per_expert': {1: 1056, 5: 1033, 9: 49, 21: 33, 25: 25, 29: 96, 49: 103, 53: 67, 57: 165, 61: 108, 69: 57, 73: 52, 77: 55, 81: 90, 85: 70, 89: 39, 93: 74, 97: 125, 101: 57, 105: 31, 109: 59, 113: 86, 117: 133, 121: 139}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 42, 46, 50, 54, 58, 62, 66, 78, 82, 86, 90, 94, 98, 102, 106, 114], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 31, 'token_total': 3940, 'token_per_expert': {2: 1069, 6: 1045, 10: 23, 14: 79, 18: 137, 22: 149, 26: 152, 30: 77, 42: 188, 46: 16, 50: 16, 54: 89, 58: 16, 62: 114, 66: 29, 78: 42, 82: 72, 86: 218, 90: 105, 94: 15, 98: 7, 102: 6, 106: 255, 114: 21}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 44, 48, 52, 56, 60, 64, 76, 80, 84, 88, 92, 96, 108, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 30, 'token_total': 4102, 'token_per_expert': {0: 1029, 4: 1244, 8: 13, 16: 102, 20: 212, 24: 21, 28: 141, 32: 30, 44: 45, 48: 73, 52: 274, 56: 126, 60: 114, 64: 306, 76: 18, 80: 38, 84: 22, 88: 12, 92: 45, 96: 28, 108: 32, 116: 45, 120: 37, 124: 95}}
INFO 05-06 10:50:22.497627.497627 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 1.998ms | allocate_experts_across_cpu_gpu: 0.459ms
INFO 05-06 10:50:22.497203.497203 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 8.96453857421875e-05 seconds
INFO 05-06 10:50:22.498871.498871 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0006415843963623047 seconds
INFO 05-06 10:50:22.499682.499682 lmp.py:1387] [layer_moe_fused] kt_kernel_prep_submit time: 0.0008120536804199219 seconds
INFO 05-06 10:50:22.510268.510268 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.011301040649414062 seconds
INFO 05-06 10:50:22.511950.511950 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010764598846435547 seconds
INFO 05-06 10:50:22.514617.514617 mlpmodule.py:2799] [fused_experts] gmm total=1.849ms E=32 S=3867 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.514217.514217 mlpmodule.py:2799] [fused_experts] gmm total=1.929ms E=32 S=3970 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.514102.514102 mlpmodule.py:2799] [fused_experts] gmm total=2.154ms E=32 S=4426 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.514721.514721 mlpmodule.py:2799] [fused_experts] gmm total=2.233ms E=32 S=4121 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.515536.515536 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0035827159881591797 seconds
INFO 05-06 10:50:22.515244.515244 lmp.py:1496] [layer_moe_fused] to time: 6.389617919921875e-05 seconds
INFO 05-06 10:50:22.516145.516145 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00028395652770996094 seconds
DEBUG 05-06 10:50:22.517753.517753 cuda_h.py:27] end *layer_moe_fused cost 22.039 ms
DEBUG 05-06 10:50:22.522198.522198 cuda_h.py:27] end prefill_layer cost 32.838 ms
DEBUG 05-06 10:50:22.523631.523631 lmp.py:841] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 10:50:22.523354.523354 cuda_h.py:27] end prefill_step cost 1579.807 ms
INFO 05-06 10:50:22.523259.523259 lmp.py:843] prefill time: 1.708404302597046 seconds
WARNING 05-06 10:50:22.545242.545242 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:50:22.545832.545832 helper.py:35]   NaN count (hidden): 2883584
WARNING 05-06 10:50:22.546607.546607 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:50:22.546592.546592 helper.py:39]   NaN count (normed): 2883584
WARNING 05-06 10:50:22.551217.551217 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:50:22.551501.551501 helper.py:50]   NaN count: 524288
WARNING 05-06 10:50:22.551291.551291 helper.py:51]   Logits shape: (4, 262144)
DEBUG 05-06 10:50:22.554158.554158 cuda_h.py:27] end init_inputs_tokens cost 9.173 ms
DEBUG 05-06 10:50:22.554597.554597 lmp.py:899] decode step 0 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:50:22.554898.554898 lmp.py:904] ---- decode step 0 layer 0 ----
DEBUG 05-06 10:50:22.556905.556905 cuda_h.py:27] end *sagl cost 2.138 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 47, 55, 63, 79, 83, 87, 103, 123, 127], 'expert_count': 10, 'ideal_gpu_count': 6, 'keep_on_gpu': 10, 'hit_count_on_device': 10, 'token_total': 12, 'token_per_expert': {15: 1, 47: 1, 55: 1, 63: 1, 79: 1, 83: 2, 87: 1, 103: 1, 123: 1, 127: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [33, 45, 53], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 3, 'token_per_expert': {33: 1, 45: 1, 53: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [22, 26, 50, 90, 114], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {22: 2, 26: 2, 50: 1, 90: 2, 114: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [8, 48, 60, 64, 116, 124], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {8: 2, 48: 1, 60: 3, 64: 1, 116: 1, 124: 1}}
INFO 05-06 10:50:22.558233.558233 lmp.py:1338] [layer_moe_fused] layer=0 moe_prefix_before_alloc: 0.442ms | allocate_experts_across_cpu_gpu: 0.114ms
INFO 05-06 10:50:22.558110.558110 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.002716064453125e-05 seconds
INFO 05-06 10:50:22.559858.559858 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001132965087890625 seconds
INFO 05-06 10:50:22.562435.562435 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0022602081298828125 seconds
INFO 05-06 10:50:22.563607.563607 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0009000301361083984 seconds
INFO 05-06 10:50:22.564566.564566 mlpmodule.py:2799] [fused_experts] gmm total=0.913ms E=32 S=3 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.564744.564744 mlpmodule.py:2799] [fused_experts] gmm total=0.992ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.564038.564038 mlpmodule.py:2799] [fused_experts] gmm total=1.235ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.564239.564239 mlpmodule.py:2799] [fused_experts] gmm total=1.061ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.565747.565747 lmp.py:1484] [layer_moe_fused] experts compute time: 0.002142667770385742 seconds
INFO 05-06 10:50:22.565342.565342 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.270408630371094e-05 seconds
DEBUG 05-06 10:50:22.566318.566318 cuda_h.py:27] end *layer_moe_fused cost 8.029 ms
DEBUG 05-06 10:50:22.566613.566613 cuda_h.py:27] end decode_layer cost 11.826 ms
DEBUG 05-06 10:50:22.566502.566502 lmp.py:904] ---- decode step 0 layer 1 ----
DEBUG 05-06 10:50:22.568764.568764 cuda_h.py:27] end *sagl cost 2.046 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [79, 83, 107, 119, 123], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {79: 1, 83: 1, 107: 1, 119: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [5, 9, 13, 21, 29, 73, 121], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {5: 1, 9: 2, 13: 1, 21: 1, 29: 1, 73: 1, 121: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [30, 34, 46, 54, 110], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {30: 2, 34: 1, 46: 1, 54: 1, 110: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 8, 56, 92, 96, 116, 124], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {0: 2, 8: 2, 56: 2, 92: 2, 96: 1, 116: 1, 124: 2}}
INFO 05-06 10:50:22.570242.570242 lmp.py:1338] [layer_moe_fused] layer=1 moe_prefix_before_alloc: 0.384ms | allocate_experts_across_cpu_gpu: 0.105ms
INFO 05-06 10:50:22.570795.570795 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0742416381835938e-05 seconds
INFO 05-06 10:50:22.571200.571200 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0008187294006347656 seconds
INFO 05-06 10:50:22.573511.573511 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017070770263671875 seconds
INFO 05-06 10:50:22.574464.574464 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001077890396118164 seconds
INFO 05-06 10:50:22.575475.575475 mlpmodule.py:2799] [fused_experts] gmm total=1.171ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.576301.576301 mlpmodule.py:2799] [fused_experts] gmm total=1.133ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.576139.576139 mlpmodule.py:2799] [fused_experts] gmm total=1.316ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.576355.576355 mlpmodule.py:2799] [fused_experts] gmm total=1.274ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.577010.577010 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0025103092193603516 seconds
INFO 05-06 10:50:22.577107.577107 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.0558319091796875e-05 seconds
DEBUG 05-06 10:50:22.577242.577242 cuda_h.py:27] end *layer_moe_fused cost 8.005 ms
DEBUG 05-06 10:50:22.578430.578430 cuda_h.py:27] end decode_layer cost 11.703 ms
DEBUG 05-06 10:50:22.578227.578227 lmp.py:904] ---- decode step 0 layer 2 ----
DEBUG 05-06 10:50:22.580713.580713 cuda_h.py:27] end *sagl cost 2.002 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 59, 71, 91], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {11: 2, 59: 1, 71: 1, 91: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 41, 45, 49, 65, 77, 81], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {1: 1, 41: 1, 45: 1, 49: 1, 65: 1, 77: 1, 81: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [14, 62, 70, 90, 102, 106, 126], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {14: 1, 62: 1, 70: 1, 90: 1, 102: 2, 106: 1, 126: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [8, 12, 24, 52, 76, 80, 108], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {8: 1, 12: 1, 24: 1, 52: 1, 76: 4, 80: 1, 108: 1}}
INFO 05-06 10:50:22.582375.582375 lmp.py:1338] [layer_moe_fused] layer=2 moe_prefix_before_alloc: 0.384ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:50:22.582252.582252 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0265579223632812e-05 seconds
INFO 05-06 10:50:22.583840.583840 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010995864868164062 seconds
INFO 05-06 10:50:22.585250.585250 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001718759536743164 seconds
INFO 05-06 10:50:22.586876.586876 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0010256767272949219 seconds
INFO 05-06 10:50:22.587630.587630 mlpmodule.py:2799] [fused_experts] gmm total=1.345ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.587302.587302 mlpmodule.py:2799] [fused_experts] gmm total=1.405ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.587898.587898 mlpmodule.py:2799] [fused_experts] gmm total=1.544ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.588490.588490 mlpmodule.py:2799] [fused_experts] gmm total=1.560ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.589121.589121 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0030994415283203125 seconds
INFO 05-06 10:50:22.589372.589372 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 7.367134094238281e-05 seconds
DEBUG 05-06 10:50:22.589912.589912 cuda_h.py:27] end *layer_moe_fused cost 8.431 ms
DEBUG 05-06 10:50:22.590717.590717 cuda_h.py:27] end decode_layer cost 12.047 ms
DEBUG 05-06 10:50:22.590699.590699 lmp.py:904] ---- decode step 0 layer 3 ----
DEBUG 05-06 10:50:22.592104.592104 cuda_h.py:27] end *sagl cost 1.978 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [39, 67, 107, 123], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {39: 1, 67: 1, 107: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [5, 73, 85, 93, 101, 117, 125], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {5: 1, 73: 1, 85: 1, 93: 1, 101: 1, 117: 2, 125: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [26, 50, 54, 62, 70, 110, 114, 118, 126], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 10, 'token_per_expert': {26: 1, 50: 2, 54: 1, 62: 1, 70: 1, 110: 1, 114: 1, 118: 1, 126: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [24, 40, 52, 96, 104, 108], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 10, 'token_per_expert': {24: 1, 40: 1, 52: 1, 96: 3, 104: 3, 108: 1}}
INFO 05-06 10:50:22.594514.594514 lmp.py:1338] [layer_moe_fused] layer=3 moe_prefix_before_alloc: 0.373ms | allocate_experts_across_cpu_gpu: 0.102ms
INFO 05-06 10:50:22.594961.594961 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9550323486328125e-05 seconds
INFO 05-06 10:50:22.595406.595406 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012054443359375 seconds
INFO 05-06 10:50:22.597606.597606 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017385482788085938 seconds
INFO 05-06 10:50:22.598090.598090 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001332998275756836 seconds
INFO 05-06 10:50:22.600530.600530 mlpmodule.py:2799] [fused_experts] gmm total=1.024ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.601149.601149 mlpmodule.py:2799] [fused_experts] gmm total=2.215ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.601926.601926 mlpmodule.py:2799] [fused_experts] gmm total=2.670ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.602645.602645 mlpmodule.py:2799] [fused_experts] gmm total=3.100ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.603990.603990 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0050144195556640625 seconds
INFO 05-06 10:50:22.603018.603018 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00011682510375976562 seconds
DEBUG 05-06 10:50:22.604604.604604 cuda_h.py:27] end *layer_moe_fused cost 11.070 ms
DEBUG 05-06 10:50:22.605960.605960 cuda_h.py:27] end decode_layer cost 14.890 ms
DEBUG 05-06 10:50:22.605281.605281 lmp.py:904] ---- decode step 0 layer 4 ----
DEBUG 05-06 10:50:22.608401.608401 cuda_h.py:27] end *sagl cost 2.724 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 31, 51, 83, 87], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {3: 2, 31: 1, 51: 1, 83: 1, 87: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [17, 25, 45, 57, 93, 101, 105, 113, 117], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 11, 'token_per_expert': {17: 1, 25: 1, 45: 2, 57: 1, 93: 2, 101: 1, 105: 1, 113: 1, 117: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 50, 82, 106, 114, 118, 122, 126], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 1, 50: 1, 82: 1, 106: 1, 114: 1, 118: 1, 122: 2, 126: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [20, 24, 56], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {20: 2, 24: 1, 56: 1}}
INFO 05-06 10:50:22.610297.610297 lmp.py:1338] [layer_moe_fused] layer=4 moe_prefix_before_alloc: 0.582ms | allocate_experts_across_cpu_gpu: 0.205ms
INFO 05-06 10:50:22.610110.610110 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.743171691894531e-05 seconds
INFO 05-06 10:50:22.612468.612468 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015413761138916016 seconds
INFO 05-06 10:50:22.613205.613205 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0015654563903808594 seconds
INFO 05-06 10:50:22.615718.615718 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001222372055053711 seconds
INFO 05-06 10:50:22.616300.616300 mlpmodule.py:2799] [fused_experts] gmm total=1.352ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.617849.617849 mlpmodule.py:2799] [fused_experts] gmm total=2.028ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.618576.618576 mlpmodule.py:2799] [fused_experts] gmm total=2.483ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.619201.619201 mlpmodule.py:2799] [fused_experts] gmm total=4.426ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.620655.620655 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005572319030761719 seconds
INFO 05-06 10:50:22.621740.621740 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.888938903808594e-05 seconds
DEBUG 05-06 10:50:22.621971.621971 cuda_h.py:27] end *layer_moe_fused cost 11.825 ms
DEBUG 05-06 10:50:22.622923.622923 cuda_h.py:27] end decode_layer cost 16.621 ms
DEBUG 05-06 10:50:22.622211.622211 lmp.py:904] ---- decode step 0 layer 5 ----
DEBUG 05-06 10:50:22.624939.624939 cuda_h.py:27] end *sagl cost 2.134 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 39, 47, 71, 95, 99, 119, 123, 127], 'expert_count': 9, 'ideal_gpu_count': 7, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 12, 'token_per_expert': {11: 1, 39: 1, 47: 1, 71: 3, 95: 1, 99: 2, 119: 1, 123: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [5, 29, 61, 65], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {5: 1, 29: 1, 61: 1, 65: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 18, 34, 46, 70, 94, 118], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {2: 1, 18: 1, 34: 1, 46: 1, 70: 1, 94: 1, 118: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [28, 32, 36, 52, 60, 72, 116], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {28: 1, 32: 1, 36: 1, 52: 2, 60: 1, 72: 1, 116: 1}}
INFO 05-06 10:50:22.626437.626437 lmp.py:1338] [layer_moe_fused] layer=5 moe_prefix_before_alloc: 0.441ms | allocate_experts_across_cpu_gpu: 0.161ms
INFO 05-06 10:50:22.626732.626732 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.409385681152344e-05 seconds
INFO 05-06 10:50:22.627070.627070 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015332698822021484 seconds
INFO 05-06 10:50:22.629685.629685 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0018820762634277344 seconds
INFO 05-06 10:50:22.631230.631230 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015425682067871094 seconds
INFO 05-06 10:50:22.634435.634435 mlpmodule.py:2799] [fused_experts] gmm total=2.305ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.634939.634939 mlpmodule.py:2799] [fused_experts] gmm total=2.401ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.634558.634558 mlpmodule.py:2799] [fused_experts] gmm total=2.426ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.634143.634143 mlpmodule.py:2799] [fused_experts] gmm total=2.581ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.636526.636526 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004587650299072266 seconds
INFO 05-06 10:50:22.636266.636266 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.626678466796875e-05 seconds
DEBUG 05-06 10:50:22.636274.636274 cuda_h.py:27] end *layer_moe_fused cost 11.372 ms
DEBUG 05-06 10:50:22.637824.637824 cuda_h.py:27] end decode_layer cost 15.162 ms
DEBUG 05-06 10:50:22.637945.637945 lmp.py:904] ---- decode step 0 layer 6 ----
DEBUG 05-06 10:50:22.639111.639111 cuda_h.py:27] end *sagl cost 1.558 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [35, 43, 67, 71, 87, 103, 115], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {35: 2, 43: 1, 67: 1, 71: 1, 87: 2, 103: 1, 115: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 13, 25], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {1: 1, 13: 2, 25: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 70, 78, 90, 106, 118], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 1, 70: 1, 78: 3, 90: 1, 106: 1, 118: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [24, 32, 36, 68, 96, 104, 108], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {24: 1, 32: 1, 36: 1, 68: 2, 96: 1, 104: 1, 108: 1}}
INFO 05-06 10:50:22.640911.640911 lmp.py:1338] [layer_moe_fused] layer=6 moe_prefix_before_alloc: 0.324ms | allocate_experts_across_cpu_gpu: 0.105ms
INFO 05-06 10:50:22.640980.640980 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.47955322265625e-05 seconds
INFO 05-06 10:50:22.641856.641856 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014183521270751953 seconds
INFO 05-06 10:50:22.643499.643499 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001325368881225586 seconds
INFO 05-06 10:50:22.644616.644616 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012500286102294922 seconds
INFO 05-06 10:50:22.647297.647297 mlpmodule.py:2799] [fused_experts] gmm total=2.320ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.647807.647807 mlpmodule.py:2799] [fused_experts] gmm total=2.420ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.647980.647980 mlpmodule.py:2799] [fused_experts] gmm total=2.573ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.647498.647498 mlpmodule.py:2799] [fused_experts] gmm total=2.709ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.649090.649090 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004603862762451172 seconds
INFO 05-06 10:50:22.649546.649546 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.555152893066406e-05 seconds
DEBUG 05-06 10:50:22.649411.649411 cuda_h.py:27] end *layer_moe_fused cost 9.824 ms
DEBUG 05-06 10:50:22.650173.650173 cuda_h.py:27] end decode_layer cost 12.732 ms
DEBUG 05-06 10:50:22.650009.650009 lmp.py:904] ---- decode step 0 layer 7 ----
DEBUG 05-06 10:50:22.651932.651932 cuda_h.py:27] end *sagl cost 1.589 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 43, 63, 103], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {19: 1, 43: 1, 63: 1, 103: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [9, 29, 53, 69, 73, 97, 121], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {9: 1, 29: 2, 53: 1, 69: 1, 73: 1, 97: 3, 121: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [10, 18, 34, 54, 90, 106, 114], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {10: 1, 18: 1, 34: 2, 54: 1, 90: 1, 106: 1, 114: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [12, 20, 36, 64, 96, 104, 108, 116], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {12: 1, 20: 2, 36: 1, 64: 1, 96: 2, 104: 1, 108: 1, 116: 1}}
INFO 05-06 10:50:22.653856.653856 lmp.py:1338] [layer_moe_fused] layer=7 moe_prefix_before_alloc: 0.325ms | allocate_experts_across_cpu_gpu: 0.107ms
INFO 05-06 10:50:22.653740.653740 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.6464462280273438e-05 seconds
INFO 05-06 10:50:22.654545.654545 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012612342834472656 seconds
INFO 05-06 10:50:22.657222.657222 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.002747774124145508 seconds
INFO 05-06 10:50:22.659874.659874 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00153350830078125 seconds
INFO 05-06 10:50:22.661796.661796 mlpmodule.py:2799] [fused_experts] gmm total=1.892ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.661510.661510 mlpmodule.py:2799] [fused_experts] gmm total=2.142ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.661458.661458 mlpmodule.py:2799] [fused_experts] gmm total=2.268ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.663896.663896 mlpmodule.py:2799] [fused_experts] gmm total=3.397ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.663247.663247 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0046617984771728516 seconds
INFO 05-06 10:50:22.664627.664627 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 0.00011777877807617188 seconds
DEBUG 05-06 10:50:22.665422.665422 cuda_h.py:27] end *layer_moe_fused cost 12.564 ms
DEBUG 05-06 10:50:22.666189.666189 cuda_h.py:27] end decode_layer cost 16.230 ms
DEBUG 05-06 10:50:22.666306.666306 lmp.py:904] ---- decode step 0 layer 8 ----
DEBUG 05-06 10:50:22.670307.670307 cuda_h.py:27] end *sagl cost 4.295 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [19, 51, 55, 63, 75, 87, 103], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {19: 1, 51: 4, 55: 2, 63: 1, 75: 1, 87: 1, 103: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [65, 69, 93], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {65: 1, 69: 2, 93: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [6, 42, 46, 50, 54, 110], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {6: 1, 42: 1, 46: 1, 50: 1, 54: 3, 110: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [12, 24, 36, 44, 64, 88], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {12: 1, 24: 1, 36: 1, 44: 1, 64: 2, 88: 1}}
INFO 05-06 10:50:22.674345.674345 lmp.py:1338] [layer_moe_fused] layer=8 moe_prefix_before_alloc: 0.782ms | allocate_experts_across_cpu_gpu: 0.229ms
INFO 05-06 10:50:22.674370.674370 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.528594970703125e-05 seconds
INFO 05-06 10:50:22.675321.675321 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014338493347167969 seconds
INFO 05-06 10:50:22.678538.678538 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.002828359603881836 seconds
INFO 05-06 10:50:22.680341.680341 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015287399291992188 seconds
INFO 05-06 10:50:22.683406.683406 mlpmodule.py:2799] [fused_experts] gmm total=2.308ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.683789.683789 mlpmodule.py:2799] [fused_experts] gmm total=2.387ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.683059.683059 mlpmodule.py:2799] [fused_experts] gmm total=2.417ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.683192.683192 mlpmodule.py:2799] [fused_experts] gmm total=2.556ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.685249.685249 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004697561264038086 seconds
INFO 05-06 10:50:22.685619.685619 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.651878356933594e-05 seconds
DEBUG 05-06 10:50:22.685853.685853 cuda_h.py:27] end *layer_moe_fused cost 12.632 ms
DEBUG 05-06 10:50:22.686936.686936 cuda_h.py:27] end decode_layer cost 19.812 ms
DEBUG 05-06 10:50:22.686640.686640 lmp.py:904] ---- decode step 0 layer 9 ----
DEBUG 05-06 10:50:22.688793.688793 cuda_h.py:27] end *sagl cost 2.141 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 95, 111], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 1, 7: 1, 15: 1, 19: 1, 95: 4, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [37, 57, 69, 81, 89, 101, 105, 117], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {37: 1, 57: 1, 69: 1, 81: 2, 89: 2, 101: 2, 105: 1, 117: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [30, 54, 70, 74], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {30: 1, 54: 1, 70: 1, 74: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [4, 36, 48, 52, 76, 92], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {4: 2, 36: 1, 48: 1, 52: 1, 76: 1, 92: 2}}
INFO 05-06 10:50:22.690245.690245 lmp.py:1338] [layer_moe_fused] layer=9 moe_prefix_before_alloc: 0.392ms | allocate_experts_across_cpu_gpu: 0.106ms
INFO 05-06 10:50:22.690698.690698 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.193450927734375e-05 seconds
INFO 05-06 10:50:22.691223.691223 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014050006866455078 seconds
INFO 05-06 10:50:22.693641.693641 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017199516296386719 seconds
INFO 05-06 10:50:22.695592.695592 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001438140869140625 seconds
INFO 05-06 10:50:22.697007.697007 mlpmodule.py:2799] [fused_experts] gmm total=2.303ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.697562.697562 mlpmodule.py:2799] [fused_experts] gmm total=2.291ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.698196.698196 mlpmodule.py:2799] [fused_experts] gmm total=2.695ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.698336.698336 mlpmodule.py:2799] [fused_experts] gmm total=3.141ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.699782.699782 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00462651252746582 seconds
INFO 05-06 10:50:22.699245.699245 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.508827209472656e-05 seconds
DEBUG 05-06 10:50:22.700352.700352 cuda_h.py:27] end *layer_moe_fused cost 10.621 ms
DEBUG 05-06 10:50:22.700991.700991 cuda_h.py:27] end decode_layer cost 14.437 ms
DEBUG 05-06 10:50:22.700980.700980 lmp.py:904] ---- decode step 0 layer 10 ----
DEBUG 05-06 10:50:22.702106.702106 cuda_h.py:27] end *sagl cost 1.948 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [47, 67, 79], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 4, 'token_per_expert': {47: 1, 67: 2, 79: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [21, 37, 57, 81, 97, 105, 113], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {21: 1, 37: 1, 57: 1, 81: 3, 97: 3, 105: 2, 113: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [10, 18, 46, 54, 74, 126], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {10: 1, 18: 2, 46: 1, 54: 1, 74: 1, 126: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [8, 12, 44, 60, 72, 88, 92], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {8: 2, 12: 1, 44: 1, 60: 2, 72: 1, 88: 1, 92: 1}}
INFO 05-06 10:50:22.704193.704193 lmp.py:1338] [layer_moe_fused] layer=10 moe_prefix_before_alloc: 0.392ms | allocate_experts_across_cpu_gpu: 0.102ms
INFO 05-06 10:50:22.704647.704647 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:50:22.706736.706736 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014324188232421875 seconds
INFO 05-06 10:50:22.707987.707987 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016963481903076172 seconds
INFO 05-06 10:50:22.709270.709270 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014696121215820312 seconds
INFO 05-06 10:50:22.711069.711069 mlpmodule.py:2799] [fused_experts] gmm total=1.865ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.712184.712184 mlpmodule.py:2799] [fused_experts] gmm total=2.235ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.712001.712001 mlpmodule.py:2799] [fused_experts] gmm total=2.388ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.712757.712757 mlpmodule.py:2799] [fused_experts] gmm total=2.741ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.713586.713586 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004192829132080078 seconds
INFO 05-06 10:50:22.713101.713101 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.67572021484375e-05 seconds
DEBUG 05-06 10:50:22.714447.714447 cuda_h.py:27] end *layer_moe_fused cost 10.233 ms
DEBUG 05-06 10:50:22.714695.714695 cuda_h.py:27] end decode_layer cost 13.828 ms
DEBUG 05-06 10:50:22.714015.714015 lmp.py:904] ---- decode step 0 layer 11 ----
DEBUG 05-06 10:50:22.716415.716415 cuda_h.py:27] end *sagl cost 2.009 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 67, 79, 83, 99], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 1, 7: 2, 23: 1, 67: 1, 79: 2, 83: 2, 99: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 9, 25, 49, 81], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 7, 'token_per_expert': {1: 1, 5: 1, 9: 1, 25: 1, 49: 1, 81: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 38, 46, 50, 66, 102, 114], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {2: 2, 6: 2, 38: 1, 46: 1, 50: 1, 66: 1, 102: 1, 114: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 108, 124], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 108: 1, 124: 2}}
INFO 05-06 10:50:22.718827.718827 lmp.py:1338] [layer_moe_fused] layer=11 moe_prefix_before_alloc: 0.402ms | allocate_experts_across_cpu_gpu: 0.103ms
INFO 05-06 10:50:22.718565.718565 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.4318695068359375e-05 seconds
INFO 05-06 10:50:22.720946.720946 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014390945434570312 seconds
INFO 05-06 10:50:22.721224.721224 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017216205596923828 seconds
INFO 05-06 10:50:22.723129.723129 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012290477752685547 seconds
INFO 05-06 10:50:22.725348.725348 mlpmodule.py:2799] [fused_experts] gmm total=2.203ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.725628.725628 mlpmodule.py:2799] [fused_experts] gmm total=2.341ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.726373.726373 mlpmodule.py:2799] [fused_experts] gmm total=2.548ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.726376.726376 mlpmodule.py:2799] [fused_experts] gmm total=3.110ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.727979.727979 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004560708999633789 seconds
INFO 05-06 10:50:22.727733.727733 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.461143493652344e-05 seconds
DEBUG 05-06 10:50:22.728324.728324 cuda_h.py:27] end *layer_moe_fused cost 10.599 ms
DEBUG 05-06 10:50:22.729619.729619 cuda_h.py:27] end decode_layer cost 14.272 ms
DEBUG 05-06 10:50:22.729654.729654 lmp.py:904] ---- decode step 0 layer 12 ----
DEBUG 05-06 10:50:22.732425.732425 cuda_h.py:27] end *sagl cost 3.191 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 39], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {3: 1, 7: 1, 15: 1, 19: 1, 39: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 21, 45, 73, 89, 97, 117], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 8, 'token_per_expert': {1: 1, 5: 1, 21: 1, 45: 1, 73: 1, 89: 1, 97: 1, 117: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 46, 50, 74, 78, 86, 106, 114], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 15, 'token_per_expert': {2: 1, 6: 2, 46: 2, 50: 2, 74: 1, 78: 3, 86: 1, 106: 1, 114: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 36, 76], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 4, 'token_per_expert': {0: 1, 4: 1, 36: 1, 76: 1}}
INFO 05-06 10:50:22.736204.736204 lmp.py:1338] [layer_moe_fused] layer=12 moe_prefix_before_alloc: 0.732ms | allocate_experts_across_cpu_gpu: 0.190ms
INFO 05-06 10:50:22.736361.736361 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 3.266334533691406e-05 seconds
INFO 05-06 10:50:22.738193.738193 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014379024505615234 seconds
INFO 05-06 10:50:22.740206.740206 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0018982887268066406 seconds
INFO 05-06 10:50:22.741071.741071 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014429092407226562 seconds
INFO 05-06 10:50:22.743179.743179 mlpmodule.py:2799] [fused_experts] gmm total=1.423ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.743397.743397 mlpmodule.py:2799] [fused_experts] gmm total=1.732ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.744445.744445 mlpmodule.py:2799] [fused_experts] gmm total=2.356ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.744911.744911 mlpmodule.py:2799] [fused_experts] gmm total=2.713ms E=32 S=15 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.745887.745887 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0037267208099365234 seconds
INFO 05-06 10:50:22.745263.745263 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.413459777832031e-05 seconds
DEBUG 05-06 10:50:22.746575.746575 cuda_h.py:27] end *layer_moe_fused cost 10.612 ms
DEBUG 05-06 10:50:22.746863.746863 cuda_h.py:27] end decode_layer cost 17.493 ms
DEBUG 05-06 10:50:22.746514.746514 lmp.py:904] ---- decode step 0 layer 13 ----
DEBUG 05-06 10:50:22.748206.748206 cuda_h.py:27] end *sagl cost 2.044 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 47, 71, 103, 107], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 6, 'token_per_expert': {3: 1, 7: 1, 47: 1, 71: 1, 103: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 21, 125], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {1: 2, 5: 1, 21: 1, 125: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 14, 26, 78, 110, 114], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 12, 'token_per_expert': {2: 2, 6: 2, 14: 2, 26: 1, 78: 1, 110: 1, 114: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 20, 32, 80, 100, 104], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 1, 4: 1, 20: 1, 32: 2, 80: 1, 100: 2, 104: 1}}
INFO 05-06 10:50:22.750718.750718 lmp.py:1338] [layer_moe_fused] layer=13 moe_prefix_before_alloc: 0.391ms | allocate_experts_across_cpu_gpu: 0.119ms
INFO 05-06 10:50:22.750072.750072 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.0742416381835938e-05 seconds
INFO 05-06 10:50:22.751780.751780 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013289451599121094 seconds
INFO 05-06 10:50:22.753918.753918 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016932487487792969 seconds
INFO 05-06 10:50:22.755505.755505 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014526844024658203 seconds
INFO 05-06 10:50:22.757211.757211 mlpmodule.py:2799] [fused_experts] gmm total=2.053ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.757620.757620 mlpmodule.py:2799] [fused_experts] gmm total=2.128ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.757150.757150 mlpmodule.py:2799] [fused_experts] gmm total=2.122ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.758394.758394 mlpmodule.py:2799] [fused_experts] gmm total=2.385ms E=32 S=12 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.759196.759196 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004126071929931641 seconds
INFO 05-06 10:50:22.759657.759657 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.5789947509765625e-05 seconds
DEBUG 05-06 10:50:22.761610.761610 cuda_h.py:27] end *layer_moe_fused cost 11.119 ms
DEBUG 05-06 10:50:22.761977.761977 cuda_h.py:27] end decode_layer cost 15.012 ms
DEBUG 05-06 10:50:22.761773.761773 lmp.py:904] ---- decode step 0 layer 14 ----
DEBUG 05-06 10:50:22.764986.764986 cuda_h.py:27] end *sagl cost 2.150 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 39, 47, 99, 115], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 1, 7: 1, 11: 1, 39: 1, 47: 1, 99: 1, 115: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 25, 81, 97, 105, 109, 113, 121], 'expert_count': 9, 'ideal_gpu_count': 7, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 10, 'token_per_expert': {1: 1, 5: 1, 25: 1, 81: 1, 97: 1, 105: 2, 109: 1, 113: 1, 121: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 26, 42, 66], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 4, 6: 1, 26: 1, 42: 1, 66: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 56, 72, 100], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {0: 1, 4: 1, 56: 1, 72: 1, 100: 1}}
INFO 05-06 10:50:22.765553.765553 lmp.py:1338] [layer_moe_fused] layer=14 moe_prefix_before_alloc: 0.457ms | allocate_experts_across_cpu_gpu: 0.111ms
INFO 05-06 10:50:22.765113.765113 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.3603439331054688e-05 seconds
INFO 05-06 10:50:22.767666.767666 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014581680297851562 seconds
INFO 05-06 10:50:22.769765.769765 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0019061565399169922 seconds
INFO 05-06 10:50:22.770668.770668 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001356363296508789 seconds
INFO 05-06 10:50:22.773474.773474 mlpmodule.py:2799] [fused_experts] gmm total=2.120ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.773072.773072 mlpmodule.py:2799] [fused_experts] gmm total=2.080ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.773239.773239 mlpmodule.py:2799] [fused_experts] gmm total=2.402ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.774430.774430 mlpmodule.py:2799] [fused_experts] gmm total=3.265ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.775742.775742 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004514455795288086 seconds
INFO 05-06 10:50:22.775151.775151 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.437301635742188e-05 seconds
DEBUG 05-06 10:50:22.776788.776788 cuda_h.py:27] end *layer_moe_fused cost 10.925 ms
DEBUG 05-06 10:50:22.776758.776758 cuda_h.py:27] end decode_layer cost 14.765 ms
DEBUG 05-06 10:50:22.776363.776363 lmp.py:904] ---- decode step 0 layer 15 ----
DEBUG 05-06 10:50:22.778682.778682 cuda_h.py:27] end *sagl cost 1.985 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 47, 63, 75, 83, 119], 'expert_count': 8, 'ideal_gpu_count': 8, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 8, 'token_per_expert': {3: 1, 7: 1, 23: 1, 47: 1, 63: 1, 75: 1, 83: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 29, 33, 69, 81, 93, 101], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 8, 'token_per_expert': {1: 1, 5: 1, 29: 1, 33: 1, 69: 1, 81: 1, 93: 1, 101: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 30, 34, 70], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {2: 1, 6: 1, 30: 1, 34: 1, 70: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 24, 36, 68, 72, 108, 112], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {0: 1, 4: 1, 24: 1, 36: 1, 68: 1, 72: 2, 108: 2, 112: 2}}
INFO 05-06 10:50:22.780538.780538 lmp.py:1338] [layer_moe_fused] layer=15 moe_prefix_before_alloc: 0.387ms | allocate_experts_across_cpu_gpu: 0.107ms
INFO 05-06 10:50:22.780706.780706 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.4080276489257812e-05 seconds
INFO 05-06 10:50:22.781083.781083 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013301372528076172 seconds
INFO 05-06 10:50:22.783075.783075 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001863241195678711 seconds
INFO 05-06 10:50:22.785839.785839 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013914108276367188 seconds
INFO 05-06 10:50:22.787972.787972 mlpmodule.py:2799] [fused_experts] gmm total=2.402ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.788640.788640 mlpmodule.py:2799] [fused_experts] gmm total=2.372ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.788451.788451 mlpmodule.py:2799] [fused_experts] gmm total=2.577ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.789493.789493 mlpmodule.py:2799] [fused_experts] gmm total=3.407ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.790327.790327 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00492095947265625 seconds
INFO 05-06 10:50:22.790889.790889 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.794929504394531e-05 seconds
DEBUG 05-06 10:50:22.790315.790315 cuda_h.py:27] end *layer_moe_fused cost 11.040 ms
DEBUG 05-06 10:50:22.791272.791272 cuda_h.py:27] end decode_layer cost 14.674 ms
DEBUG 05-06 10:50:22.791161.791161 lmp.py:904] ---- decode step 0 layer 16 ----
DEBUG 05-06 10:50:22.793043.793043 cuda_h.py:27] end *sagl cost 1.978 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 87, 107], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 6, 'token_per_expert': {3: 1, 7: 1, 15: 1, 87: 2, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 85, 105], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 7, 'token_per_expert': {1: 2, 5: 3, 85: 1, 105: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 22, 54, 62, 66, 78, 102], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {2: 1, 6: 1, 22: 1, 54: 2, 62: 1, 66: 1, 78: 1, 102: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 20, 32, 44, 108, 116], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 1, 4: 2, 20: 1, 32: 2, 44: 1, 108: 1, 116: 1}}
INFO 05-06 10:50:22.795328.795328 lmp.py:1338] [layer_moe_fused] layer=16 moe_prefix_before_alloc: 0.380ms | allocate_experts_across_cpu_gpu: 0.100ms
INFO 05-06 10:50:22.795635.795635 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1457672119140625e-05 seconds
INFO 05-06 10:50:22.796591.796591 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0014061927795410156 seconds
INFO 05-06 10:50:22.798835.798835 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017011165618896484 seconds
INFO 05-06 10:50:22.799161.799161 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00152587890625 seconds
INFO 05-06 10:50:22.802710.802710 mlpmodule.py:2799] [fused_experts] gmm total=1.994ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.802047.802047 mlpmodule.py:2799] [fused_experts] gmm total=2.072ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.802237.802237 mlpmodule.py:2799] [fused_experts] gmm total=2.284ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.803822.803822 mlpmodule.py:2799] [fused_experts] gmm total=3.250ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.804511.804511 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004555702209472656 seconds
INFO 05-06 10:50:22.804823.804823 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 9.012222290039062e-05 seconds
DEBUG 05-06 10:50:22.805780.805780 cuda_h.py:27] end *layer_moe_fused cost 11.203 ms
DEBUG 05-06 10:50:22.806953.806953 cuda_h.py:27] end decode_layer cost 15.144 ms
DEBUG 05-06 10:50:22.806837.806837 lmp.py:904] ---- decode step 0 layer 17 ----
DEBUG 05-06 10:50:22.810163.810163 cuda_h.py:27] end *sagl cost 3.331 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 39, 47, 63], 'expert_count': 6, 'ideal_gpu_count': 7, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {3: 1, 7: 2, 23: 2, 39: 1, 47: 1, 63: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 13, 33, 53, 73, 113], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {1: 1, 5: 2, 13: 1, 33: 1, 53: 1, 73: 1, 113: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 18, 22, 34, 70, 106], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {2: 1, 6: 1, 18: 2, 22: 1, 34: 1, 70: 1, 106: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 16, 28, 68, 100, 120], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {0: 1, 4: 1, 16: 2, 28: 1, 68: 1, 100: 1, 120: 1}}
INFO 05-06 10:50:22.812174.812174 lmp.py:1338] [layer_moe_fused] layer=17 moe_prefix_before_alloc: 0.494ms | allocate_experts_across_cpu_gpu: 0.141ms
INFO 05-06 10:50:22.812046.812046 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.8133392333984375e-05 seconds
INFO 05-06 10:50:22.813147.813147 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013933181762695312 seconds
INFO 05-06 10:50:22.816771.816771 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0023469924926757812 seconds
INFO 05-06 10:50:22.818324.818324 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0015702247619628906 seconds
INFO 05-06 10:50:22.820083.820083 mlpmodule.py:2799] [fused_experts] gmm total=2.285ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.821545.821545 mlpmodule.py:2799] [fused_experts] gmm total=2.654ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.821310.821310 mlpmodule.py:2799] [fused_experts] gmm total=2.656ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.821439.821439 mlpmodule.py:2799] [fused_experts] gmm total=2.921ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.822760.822760 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0047550201416015625 seconds
INFO 05-06 10:50:22.823927.823927 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 8.916854858398438e-05 seconds
DEBUG 05-06 10:50:22.824312.824312 cuda_h.py:27] end *layer_moe_fused cost 12.250 ms
DEBUG 05-06 10:50:22.824962.824962 cuda_h.py:27] end decode_layer cost 18.166 ms
DEBUG 05-06 10:50:22.824661.824661 lmp.py:904] ---- decode step 0 layer 18 ----
DEBUG 05-06 10:50:22.828867.828867 cuda_h.py:27] end *sagl cost 3.312 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23, 75, 83, 111], 'expert_count': 6, 'ideal_gpu_count': 8, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 6, 'token_per_expert': {3: 1, 7: 1, 23: 1, 75: 1, 83: 1, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 37, 73, 77, 97, 101, 105], 'expert_count': 8, 'ideal_gpu_count': 8, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 9, 'token_per_expert': {1: 1, 5: 1, 37: 1, 73: 1, 77: 2, 97: 1, 101: 1, 105: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 26, 30, 42, 50, 58], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {2: 2, 6: 1, 26: 1, 30: 1, 42: 1, 50: 1, 58: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 32, 36, 40, 80, 84, 92, 104], 'expert_count': 9, 'ideal_gpu_count': 7, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 9, 'token_per_expert': {0: 1, 4: 1, 32: 1, 36: 1, 40: 1, 80: 1, 84: 1, 92: 1, 104: 1}}
INFO 05-06 10:50:22.830538.830538 lmp.py:1338] [layer_moe_fused] layer=18 moe_prefix_before_alloc: 0.621ms | allocate_experts_across_cpu_gpu: 0.189ms
INFO 05-06 10:50:22.831614.831614 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.384185791015625e-05 seconds
INFO 05-06 10:50:22.832621.832621 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013730525970458984 seconds
INFO 05-06 10:50:22.834666.834666 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0018661022186279297 seconds
INFO 05-06 10:50:22.835833.835833 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013499259948730469 seconds
INFO 05-06 10:50:22.838844.838844 mlpmodule.py:2799] [fused_experts] gmm total=2.116ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.838552.838552 mlpmodule.py:2799] [fused_experts] gmm total=2.398ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.838466.838466 mlpmodule.py:2799] [fused_experts] gmm total=2.501ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.839981.839981 mlpmodule.py:2799] [fused_experts] gmm total=2.958ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.840121.840121 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004385948181152344 seconds
INFO 05-06 10:50:22.840167.840167 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.628036499023438e-05 seconds
DEBUG 05-06 10:50:22.840004.840004 cuda_h.py:27] end *layer_moe_fused cost 10.778 ms
DEBUG 05-06 10:50:22.841120.841120 cuda_h.py:27] end decode_layer cost 16.488 ms
DEBUG 05-06 10:50:22.841533.841533 lmp.py:904] ---- decode step 0 layer 19 ----
DEBUG 05-06 10:50:22.843827.843827 cuda_h.py:27] end *sagl cost 2.036 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 31, 111], 'expert_count': 5, 'ideal_gpu_count': 7, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 5, 'token_per_expert': {3: 1, 7: 1, 19: 1, 31: 1, 111: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 25, 29, 61, 89, 109], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {1: 1, 5: 2, 25: 1, 29: 1, 61: 1, 89: 1, 109: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 62, 78, 82, 86, 106, 122], 'expert_count': 9, 'ideal_gpu_count': 7, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 11, 'token_per_expert': {2: 1, 6: 1, 10: 1, 62: 1, 78: 1, 82: 1, 86: 1, 106: 2, 122: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 40, 44, 84, 92], 'expert_count': 7, 'ideal_gpu_count': 7, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 8, 'token_per_expert': {0: 1, 4: 1, 8: 1, 40: 2, 44: 1, 84: 1, 92: 1}}
INFO 05-06 10:50:22.845418.845418 lmp.py:1338] [layer_moe_fused] layer=19 moe_prefix_before_alloc: 0.386ms | allocate_experts_across_cpu_gpu: 0.107ms
INFO 05-06 10:50:22.845203.845203 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.09808349609375e-05 seconds
INFO 05-06 10:50:22.846759.846759 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013566017150878906 seconds
INFO 05-06 10:50:22.848266.848266 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0018208026885986328 seconds
INFO 05-06 10:50:22.849016.849016 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013573169708251953 seconds
INFO 05-06 10:50:22.851371.851371 mlpmodule.py:2799] [fused_experts] gmm total=1.613ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.852425.852425 mlpmodule.py:2799] [fused_experts] gmm total=1.912ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.852257.852257 mlpmodule.py:2799] [fused_experts] gmm total=2.241ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.854785.854785 mlpmodule.py:2799] [fused_experts] gmm total=4.080ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.855930.855930 lmp.py:1484] [layer_moe_fused] experts compute time: 0.005271434783935547 seconds
INFO 05-06 10:50:22.855207.855207 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.580352783203125e-05 seconds
DEBUG 05-06 10:50:22.856846.856846 cuda_h.py:27] end *layer_moe_fused cost 11.484 ms
DEBUG 05-06 10:50:22.856763.856763 cuda_h.py:27] end decode_layer cost 15.192 ms
DEBUG 05-06 10:50:22.856368.856368 lmp.py:904] ---- decode step 0 layer 20 ----
DEBUG 05-06 10:50:22.858302.858302 cuda_h.py:27] end *sagl cost 1.947 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 95, 107], 'expert_count': 4, 'ideal_gpu_count': 7, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 5, 'token_per_expert': {3: 2, 7: 1, 95: 1, 107: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 13, 21, 65, 73, 85, 117], 'expert_count': 8, 'ideal_gpu_count': 7, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 10, 'token_per_expert': {1: 1, 5: 1, 13: 1, 21: 2, 65: 1, 73: 1, 85: 2, 117: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 34, 62, 94, 102], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {2: 2, 6: 1, 34: 1, 62: 1, 94: 2, 102: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 32, 36, 40, 52, 108, 112], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 9, 'token_per_expert': {0: 1, 4: 1, 32: 1, 36: 1, 40: 2, 52: 1, 108: 1, 112: 1}}
INFO 05-06 10:50:22.860475.860475 lmp.py:1338] [layer_moe_fused] layer=20 moe_prefix_before_alloc: 0.387ms | allocate_experts_across_cpu_gpu: 0.104ms
INFO 05-06 10:50:22.860551.860551 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1696090698242188e-05 seconds
INFO 05-06 10:50:22.861153.861153 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013201236724853516 seconds
INFO 05-06 10:50:22.863908.863908 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017242431640625 seconds
INFO 05-06 10:50:22.864388.864388 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012249946594238281 seconds
INFO 05-06 10:50:22.867346.867346 mlpmodule.py:2799] [fused_experts] gmm total=2.061ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.867178.867178 mlpmodule.py:2799] [fused_experts] gmm total=2.211ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.867629.867629 mlpmodule.py:2799] [fused_experts] gmm total=2.523ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.868301.868301 mlpmodule.py:2799] [fused_experts] gmm total=3.224ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.869766.869766 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00449061393737793 seconds
INFO 05-06 10:50:22.869428.869428 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.818771362304688e-05 seconds
DEBUG 05-06 10:50:22.870230.870230 cuda_h.py:27] end *layer_moe_fused cost 10.389 ms
DEBUG 05-06 10:50:22.870842.870842 cuda_h.py:27] end decode_layer cost 13.967 ms
DEBUG 05-06 10:50:22.870732.870732 lmp.py:904] ---- decode step 0 layer 21 ----
DEBUG 05-06 10:50:22.872140.872140 cuda_h.py:27] end *sagl cost 2.050 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 83, 87, 103], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {3: 2, 7: 2, 11: 1, 83: 1, 87: 1, 103: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 25], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 7, 'token_per_expert': {1: 2, 5: 4, 25: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 14, 26, 34, 86, 94, 110], 'expert_count': 8, 'ideal_gpu_count': 5, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {2: 3, 6: 2, 14: 1, 26: 1, 34: 1, 86: 1, 94: 1, 110: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 80, 124], 'expert_count': 4, 'ideal_gpu_count': 5, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {0: 2, 4: 2, 80: 1, 124: 1}}
INFO 05-06 10:50:22.874908.874908 lmp.py:1338] [layer_moe_fused] layer=21 moe_prefix_before_alloc: 0.380ms | allocate_experts_across_cpu_gpu: 0.096ms
INFO 05-06 10:50:22.874639.874639 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.811981201171875e-05 seconds
INFO 05-06 10:50:22.875011.875011 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0011854171752929688 seconds
INFO 05-06 10:50:22.877433.877433 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.001651763916015625 seconds
INFO 05-06 10:50:22.878234.878234 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012903213500976562 seconds
INFO 05-06 10:50:22.881266.881266 mlpmodule.py:2799] [fused_experts] gmm total=2.021ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.881661.881661 mlpmodule.py:2799] [fused_experts] gmm total=2.443ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.881038.881038 mlpmodule.py:2799] [fused_experts] gmm total=2.464ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.882437.882437 mlpmodule.py:2799] [fused_experts] gmm total=2.778ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.883633.883633 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004543781280517578 seconds
INFO 05-06 10:50:22.883733.883733 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.67572021484375e-05 seconds
DEBUG 05-06 10:50:22.884783.884783 cuda_h.py:27] end *layer_moe_fused cost 10.181 ms
DEBUG 05-06 10:50:22.884336.884336 cuda_h.py:27] end decode_layer cost 13.887 ms
DEBUG 05-06 10:50:22.884325.884325 lmp.py:904] ---- decode step 0 layer 22 ----
DEBUG 05-06 10:50:22.886693.886693 cuda_h.py:27] end *sagl cost 2.054 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 43, 75, 119, 123, 127], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 2, 43: 1, 75: 1, 119: 2, 123: 1, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 61, 101], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {1: 2, 5: 2, 61: 1, 101: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 26, 38, 94], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 26: 1, 38: 1, 94: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 16, 24, 108, 120], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 8: 1, 16: 1, 24: 1, 108: 1, 120: 1}}
INFO 05-06 10:50:22.888179.888179 lmp.py:1338] [layer_moe_fused] layer=22 moe_prefix_before_alloc: 0.399ms | allocate_experts_across_cpu_gpu: 0.103ms
INFO 05-06 10:50:22.888063.888063 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1219253540039062e-05 seconds
INFO 05-06 10:50:22.889076.889076 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013408660888671875 seconds
INFO 05-06 10:50:22.891721.891721 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0017824172973632812 seconds
INFO 05-06 10:50:22.893270.893270 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013155937194824219 seconds
INFO 05-06 10:50:22.895232.895232 mlpmodule.py:2799] [fused_experts] gmm total=2.383ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.895807.895807 mlpmodule.py:2799] [fused_experts] gmm total=2.364ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.896792.896792 mlpmodule.py:2799] [fused_experts] gmm total=2.593ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.896080.896080 mlpmodule.py:2799] [fused_experts] gmm total=2.872ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.897911.897911 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004615068435668945 seconds
INFO 05-06 10:50:22.898804.898804 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.413459777832031e-05 seconds
DEBUG 05-06 10:50:22.898435.898435 cuda_h.py:27] end *layer_moe_fused cost 10.525 ms
DEBUG 05-06 10:50:22.898200.898200 cuda_h.py:27] end decode_layer cost 14.261 ms
DEBUG 05-06 10:50:22.899997.899997 lmp.py:904] ---- decode step 0 layer 23 ----
DEBUG 05-06 10:50:22.901117.901117 cuda_h.py:27] end *sagl cost 1.978 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 47, 67], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {3: 2, 7: 2, 47: 1, 67: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 17, 81, 97, 109], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 17: 1, 81: 1, 97: 2, 109: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 22, 86, 118], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 8, 'token_per_expert': {2: 2, 6: 2, 22: 1, 86: 2, 118: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 12, 32, 84, 108], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {0: 2, 4: 2, 8: 1, 12: 1, 32: 1, 84: 1, 108: 1}}
INFO 05-06 10:50:22.902078.902078 lmp.py:1338] [layer_moe_fused] layer=23 moe_prefix_before_alloc: 0.381ms | allocate_experts_across_cpu_gpu: 0.099ms
INFO 05-06 10:50:22.902214.902214 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 10:50:22.904827.904827 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.00125885009765625 seconds
INFO 05-06 10:50:22.905017.905017 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016586780548095703 seconds
INFO 05-06 10:50:22.907308.907308 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013034343719482422 seconds
INFO 05-06 10:50:22.909830.909830 mlpmodule.py:2799] [fused_experts] gmm total=2.131ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.909697.909697 mlpmodule.py:2799] [fused_experts] gmm total=2.214ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.909613.909613 mlpmodule.py:2799] [fused_experts] gmm total=2.388ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.910462.910462 mlpmodule.py:2799] [fused_experts] gmm total=3.132ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.911959.911959 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004553556442260742 seconds
INFO 05-06 10:50:22.911229.911229 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.389617919921875e-05 seconds
DEBUG 05-06 10:50:22.912722.912722 cuda_h.py:27] end *layer_moe_fused cost 10.438 ms
DEBUG 05-06 10:50:22.913733.913733 cuda_h.py:27] end decode_layer cost 14.054 ms
DEBUG 05-06 10:50:22.913529.913529 lmp.py:904] ---- decode step 0 layer 24 ----
DEBUG 05-06 10:50:22.915715.915715 cuda_h.py:27] end *sagl cost 1.958 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 63, 79, 123], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {3: 2, 7: 2, 63: 1, 79: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 33, 65, 109, 113], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {1: 2, 5: 2, 33: 2, 65: 1, 109: 1, 113: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 30, 66, 90, 110, 118], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {2: 2, 6: 2, 30: 1, 66: 1, 90: 1, 110: 1, 118: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 12, 40, 44], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 12: 1, 40: 1, 44: 1}}
INFO 05-06 10:50:22.916523.916523 lmp.py:1338] [layer_moe_fused] layer=24 moe_prefix_before_alloc: 0.382ms | allocate_experts_across_cpu_gpu: 0.098ms
INFO 05-06 10:50:22.916592.916592 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.002716064453125e-05 seconds
INFO 05-06 10:50:22.918895.918895 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013306140899658203 seconds
INFO 05-06 10:50:22.920020.920020 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016825199127197266 seconds
INFO 05-06 10:50:22.921424.921424 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013189315795898438 seconds
INFO 05-06 10:50:22.924728.924728 mlpmodule.py:2799] [fused_experts] gmm total=2.317ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.924204.924204 mlpmodule.py:2799] [fused_experts] gmm total=2.394ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.924179.924179 mlpmodule.py:2799] [fused_experts] gmm total=2.556ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.925837.925837 mlpmodule.py:2799] [fused_experts] gmm total=3.279ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.926075.926075 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004726886749267578 seconds
INFO 05-06 10:50:22.926683.926683 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.437301635742188e-05 seconds
DEBUG 05-06 10:50:22.927335.927335 cuda_h.py:27] end *layer_moe_fused cost 10.875 ms
DEBUG 05-06 10:50:22.927782.927782 cuda_h.py:27] end decode_layer cost 14.470 ms
DEBUG 05-06 10:50:22.927533.927533 lmp.py:904] ---- decode step 0 layer 25 ----
DEBUG 05-06 10:50:22.929589.929589 cuda_h.py:27] end *sagl cost 2.037 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 47, 67, 95], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {3: 3, 7: 2, 19: 1, 47: 1, 67: 1, 95: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 45, 93, 117, 121], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 45: 1, 93: 1, 117: 1, 121: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 58], 'expert_count': 3, 'ideal_gpu_count': 5, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 6, 'token_per_expert': {2: 2, 6: 2, 58: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 16, 44, 68, 104], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 3, 4: 2, 16: 1, 44: 1, 68: 1, 104: 1}}
INFO 05-06 10:50:22.931569.931569 lmp.py:1338] [layer_moe_fused] layer=25 moe_prefix_before_alloc: 0.373ms | allocate_experts_across_cpu_gpu: 0.091ms
INFO 05-06 10:50:22.931492.931492 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.002716064453125e-05 seconds
INFO 05-06 10:50:22.932903.932903 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.001355886459350586 seconds
INFO 05-06 10:50:22.934861.934861 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016660690307617188 seconds
INFO 05-06 10:50:22.935861.935861 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012984275817871094 seconds
INFO 05-06 10:50:22.938917.938917 mlpmodule.py:2799] [fused_experts] gmm total=2.235ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.938631.938631 mlpmodule.py:2799] [fused_experts] gmm total=2.239ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.938764.938764 mlpmodule.py:2799] [fused_experts] gmm total=2.531ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.939863.939863 mlpmodule.py:2799] [fused_experts] gmm total=2.732ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.940070.940070 lmp.py:1484] [layer_moe_fused] experts compute time: 0.004401683807373047 seconds
INFO 05-06 10:50:22.940771.940771 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 6.389617919921875e-05 seconds
DEBUG 05-06 10:50:22.941191.941191 cuda_h.py:27] end *layer_moe_fused cost 10.160 ms
DEBUG 05-06 10:50:22.941002.941002 cuda_h.py:27] end decode_layer cost 13.823 ms
DEBUG 05-06 10:50:22.941084.941084 lmp.py:904] ---- decode step 0 layer 26 ----
DEBUG 05-06 10:50:22.943628.943628 cuda_h.py:27] end *sagl cost 1.964 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 23, 43, 79, 119], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 9, 'token_per_expert': {3: 2, 7: 2, 19: 1, 23: 1, 43: 1, 79: 1, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 49, 65, 85], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {1: 2, 5: 2, 49: 1, 65: 1, 85: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 10, 38, 90], 'expert_count': 5, 'ideal_gpu_count': 6, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {2: 2, 6: 2, 10: 1, 38: 1, 90: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 8, 20, 52, 84], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 3, 8: 1, 20: 1, 52: 1, 84: 1}}
INFO 05-06 10:50:22.945416.945416 lmp.py:1338] [layer_moe_fused] layer=26 moe_prefix_before_alloc: 0.381ms | allocate_experts_across_cpu_gpu: 0.095ms
INFO 05-06 10:50:22.945538.945538 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.8835067749023438e-05 seconds
INFO 05-06 10:50:22.946856.946856 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013570785522460938 seconds
INFO 05-06 10:50:22.948703.948703 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016875267028808594 seconds
INFO 05-06 10:50:22.949504.949504 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013265609741210938 seconds
INFO 05-06 10:50:22.952740.952740 mlpmodule.py:2799] [fused_experts] gmm total=2.261ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.952441.952441 mlpmodule.py:2799] [fused_experts] gmm total=2.273ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.952445.952445 mlpmodule.py:2799] [fused_experts] gmm total=2.456ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.952831.952831 mlpmodule.py:2799] [fused_experts] gmm total=2.539ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.954004.954004 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0043659210205078125 seconds
INFO 05-06 10:50:22.954824.954824 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.553794860839844e-05 seconds
DEBUG 05-06 10:50:22.954714.954714 cuda_h.py:27] end *layer_moe_fused cost 10.177 ms
DEBUG 05-06 10:50:22.955333.955333 cuda_h.py:27] end decode_layer cost 13.766 ms
DEBUG 05-06 10:50:22.955083.955083 lmp.py:904] ---- decode step 0 layer 27 ----
DEBUG 05-06 10:50:22.957490.957490 cuda_h.py:27] end *sagl cost 2.013 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 27, 87, 103, 115], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {3: 2, 7: 2, 27: 1, 87: 1, 103: 1, 115: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 29, 41, 61, 85, 97, 121], 'expert_count': 8, 'ideal_gpu_count': 6, 'keep_on_gpu': 8, 'hit_count_on_device': 8, 'token_total': 11, 'token_per_expert': {1: 3, 5: 2, 29: 1, 41: 1, 61: 1, 85: 1, 97: 1, 121: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 58, 62], 'expert_count': 4, 'ideal_gpu_count': 6, 'keep_on_gpu': 4, 'hit_count_on_device': 4, 'token_total': 6, 'token_per_expert': {2: 2, 6: 2, 58: 1, 62: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 32, 48, 108], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 32: 1, 48: 1, 108: 1}}
INFO 05-06 10:50:22.959072.959072 lmp.py:1338] [layer_moe_fused] layer=27 moe_prefix_before_alloc: 0.371ms | allocate_experts_across_cpu_gpu: 0.097ms
INFO 05-06 10:50:22.959518.959518 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9073486328125e-05 seconds
INFO 05-06 10:50:22.960007.960007 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013074874877929688 seconds
INFO 05-06 10:50:22.962926.962926 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0016689300537109375 seconds
INFO 05-06 10:50:22.963150.963150 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012862682342529297 seconds
INFO 05-06 10:50:22.965427.965427 mlpmodule.py:2799] [fused_experts] gmm total=1.066ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.965017.965017 mlpmodule.py:2799] [fused_experts] gmm total=2.123ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.966632.966632 mlpmodule.py:2799] [fused_experts] gmm total=2.098ms E=32 S=6 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.966116.966116 mlpmodule.py:2799] [fused_experts] gmm total=2.411ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.967058.967058 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00368499755859375 seconds
INFO 05-06 10:50:22.967022.967022 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 4.76837158203125e-05 seconds
DEBUG 05-06 10:50:22.968807.968807 cuda_h.py:27] end *layer_moe_fused cost 9.579 ms
DEBUG 05-06 10:50:22.968228.968228 cuda_h.py:27] end decode_layer cost 13.197 ms
DEBUG 05-06 10:50:22.968594.968594 lmp.py:904] ---- decode step 0 layer 28 ----
DEBUG 05-06 10:50:22.970887.970887 cuda_h.py:27] end *sagl cost 2.001 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 19, 39, 67, 115, 119], 'expert_count': 7, 'ideal_gpu_count': 6, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {3: 2, 7: 2, 19: 1, 39: 1, 67: 1, 115: 2, 119: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 13, 49, 53, 57, 65, 89, 113], 'expert_count': 9, 'ideal_gpu_count': 6, 'keep_on_gpu': 9, 'hit_count_on_device': 9, 'token_total': 11, 'token_per_expert': {1: 2, 5: 2, 13: 1, 49: 1, 53: 1, 57: 1, 65: 1, 89: 1, 113: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6], 'expert_count': 2, 'ideal_gpu_count': 6, 'keep_on_gpu': 2, 'hit_count_on_device': 2, 'token_total': 4, 'token_per_expert': {2: 2, 6: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 32, 104, 108], 'expert_count': 5, 'ideal_gpu_count': 5, 'keep_on_gpu': 5, 'hit_count_on_device': 5, 'token_total': 7, 'token_per_expert': {0: 2, 4: 2, 32: 1, 104: 1, 108: 1}}
INFO 05-06 10:50:22.972656.972656 lmp.py:1338] [layer_moe_fused] layer=28 moe_prefix_before_alloc: 0.376ms | allocate_experts_across_cpu_gpu: 0.104ms
INFO 05-06 10:50:22.972056.972056 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 1.9311904907226562e-05 seconds
INFO 05-06 10:50:22.973717.973717 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013294219970703125 seconds
INFO 05-06 10:50:22.975968.975968 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.0018801689147949219 seconds
INFO 05-06 10:50:22.977065.977065 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0012633800506591797 seconds
INFO 05-06 10:50:22.978880.978880 mlpmodule.py:2799] [fused_experts] gmm total=1.231ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.979621.979621 mlpmodule.py:2799] [fused_experts] gmm total=1.977ms E=32 S=4 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.980917.980917 mlpmodule.py:2799] [fused_experts] gmm total=2.095ms E=32 S=7 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.980084.980084 mlpmodule.py:2799] [fused_experts] gmm total=3.634ms E=32 S=11 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.982075.982075 lmp.py:1484] [layer_moe_fused] experts compute time: 0.00517725944519043 seconds
INFO 05-06 10:50:22.982377.982377 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.173683166503906e-05 seconds
DEBUG 05-06 10:50:22.982893.982893 cuda_h.py:27] end *layer_moe_fused cost 11.118 ms
DEBUG 05-06 10:50:22.983105.983105 cuda_h.py:27] end decode_layer cost 14.870 ms
DEBUG 05-06 10:50:22.983507.983507 lmp.py:904] ---- decode step 0 layer 29 ----
DEBUG 05-06 10:50:22.986456.986456 cuda_h.py:27] end *sagl cost 2.405 ms
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 23], 'expert_count': 3, 'ideal_gpu_count': 6, 'keep_on_gpu': 3, 'hit_count_on_device': 3, 'token_total': 5, 'token_per_expert': {3: 2, 7: 2, 23: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [1, 5, 49, 73, 81, 97], 'expert_count': 6, 'ideal_gpu_count': 6, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 8, 'token_per_expert': {1: 2, 5: 2, 49: 1, 73: 1, 81: 1, 97: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [2, 6, 18, 26, 30, 66, 78], 'expert_count': 7, 'ideal_gpu_count': 5, 'keep_on_gpu': 7, 'hit_count_on_device': 7, 'token_total': 10, 'token_per_expert': {2: 3, 6: 2, 18: 1, 26: 1, 30: 1, 66: 1, 78: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [0, 4, 52, 56, 60, 64], 'expert_count': 6, 'ideal_gpu_count': 5, 'keep_on_gpu': 6, 'hit_count_on_device': 6, 'token_total': 9, 'token_per_expert': {0: 2, 4: 3, 52: 1, 56: 1, 60: 1, 64: 1}}
INFO 05-06 10:50:22.988535.988535 lmp.py:1338] [layer_moe_fused] layer=29 moe_prefix_before_alloc: 0.476ms | allocate_experts_across_cpu_gpu: 0.120ms
INFO 05-06 10:50:22.988578.988578 lmp.py:1352] [layer_moe_fused] get_experts_task_ids time: 2.1696090698242188e-05 seconds
INFO 05-06 10:50:22.989763.989763 lmp.py:1360] [layer_moe_fused] submit_high_priority_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013203620910644531 seconds
INFO 05-06 10:50:22.991305.991305 lmp.py:1409] [layer_moe_fused] prepare_fused_expert_work_items time: 0.002084016799926758 seconds
INFO 05-06 10:50:22.993862.993862 lmp.py:1419] [layer_moe_fused] wait_copy_tasks(gpu experts) ok=True pending_count=0 time: 0.0013136863708496094 seconds
INFO 05-06 10:50:22.994190.994190 mlpmodule.py:2799] [fused_experts] gmm total=1.154ms E=32 S=5 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.995364.995364 mlpmodule.py:2799] [fused_experts] gmm total=2.271ms E=32 S=8 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.995894.995894 mlpmodule.py:2799] [fused_experts] gmm total=2.350ms E=32 S=10 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.996630.996630 mlpmodule.py:2799] [fused_experts] gmm total=2.472ms E=32 S=9 H=2816 dtype=torch.bfloat16
INFO 05-06 10:50:22.997883.997883 lmp.py:1484] [layer_moe_fused] experts compute time: 0.0039212703704833984 seconds
INFO 05-06 10:50:22.997445.997445 lmp.py:1501] [layer_moe_fused] scatter_reduce_ time: 5.364418029785156e-05 seconds
DEBUG 05-06 10:50:22.997237.997237 cuda_h.py:27] end *layer_moe_fused cost 10.282 ms
DEBUG 05-06 10:50:22.998049.998049 cuda_h.py:27] end decode_layer cost 14.616 ms
DEBUG 05-06 10:50:22.998369.998369 cuda_h.py:27] end decode_step cost 453.194 ms
INFO 05-06 10:50:22.998648.998648 lmp.py:931] decode step 0 time: 0.45322680473327637 seconds
WARNING 05-06 10:50:22.998988.998988 helper.py:31] NaN in hidden_states before final norm (prefill output or decode stack); check MoE/CPU expert ratio LMP_MOE_CPU_EXPERT_RATIO and layer_moe_fused.
WARNING 05-06 10:50:22.998456.998456 helper.py:35]   NaN count (hidden): 5632
WARNING 05-06 10:50:22.999150.999150 helper.py:38] NaN after final norm (norm weights or input scale)
WARNING 05-06 10:50:22.999320.999320 helper.py:39]   NaN count (normed): 5632
WARNING 05-06 10:50:23.004748.004748 helper.py:49] next_token_logits contains NaN; replacing with 0.0 for sampling.
WARNING 05-06 10:50:23.004090.004090 helper.py:50]   NaN count: 524288
WARNING 05-06 10:50:23.004120.004120 helper.py:51]   Logits shape: (4, 262144)
WARNING 05-06 10:50:23.005090.005090 helper.py:80] WARNING: Logits have extreme values: min=-896.00, max=1032.00
WARNING 05-06 10:50:23.005729.005729 helper.py:83]   Clamped logits to [-100, 100]
DEBUG 05-06 10:50:23.006455.006455 cuda_h.py:27] end init_inputs_tokens cost 8.361 ms
DEBUG 05-06 10:50:23.006774.006774 lmp.py:899] decode step 1 next_inputs_tokens shape=(4, 1, 2816)
DEBUG 05-06 10:50:23.006206.006206 lmp.py:904] ---- decode step 1 layer 0 ----
DEBUG 05-06 10:50:23.008618.008618 cuda_h.py:27] end *sagl cost 1.983 ms
DEBUG 05-06 10:50:23.014010.014010 cuda_h.py:27] end *layer_moe_fused cost 4.446 ms
DEBUG 05-06 10:50:23.015891.015891 cuda_h.py:27] end decode_layer cost 8.186 ms
DEBUG 05-06 10:50:23.015178.015178 lmp.py:904] ---- decode step 1 layer 1 ----
DEBUG 05-06 10:50:23.017126.017126 cuda_h.py:27] end *sagl cost 2.535 ms
DEBUG 05-06 10:50:23.023462.023462 cuda_h.py:27] end *layer_moe_fused cost 4.084 ms
DEBUG 05-06 10:50:23.023749.023749 cuda_h.py:27] end decode_layer cost 8.736 ms
DEBUG 05-06 10:50:23.024648.024648 lmp.py:904] ---- decode step 1 layer 2 ----
DEBUG 05-06 10:50:23.026029.026029 cuda_h.py:27] end *sagl cost 2.431 ms
DEBUG 05-06 10:50:23.031580.031580 cuda_h.py:27] end *layer_moe_fused cost 3.585 ms
DEBUG 05-06 10:50:23.032072.032072 cuda_h.py:27] end decode_layer cost 8.101 ms
DEBUG 05-06 10:50:23.032789.032789 lmp.py:904] ---- decode step 1 layer 3 ----
DEBUG 05-06 10:50:23.034132.034132 cuda_h.py:27] end *sagl cost 2.475 ms
DEBUG 05-06 10:50:23.039204.039204 cuda_h.py:27] end *layer_moe_fused cost 3.508 ms
DEBUG 05-06 10:50:23.040277.040277 cuda_h.py:27] end decode_layer cost 8.044 ms
DEBUG 05-06 10:50:23.040326.040326 lmp.py:904] ---- decode step 1 layer 4 ----
DEBUG 05-06 10:50:23.042926.042926 cuda_h.py:27] end *sagl cost 2.418 ms
DEBUG 05-06 10:50:23.047081.047081 cuda_h.py:27] end *layer_moe_fused cost 3.659 ms
DEBUG 05-06 10:50:23.048976.048976 cuda_h.py:27] end decode_layer cost 8.127 ms
DEBUG 05-06 10:50:23.048786.048786 lmp.py:904] ---- decode step 1 layer 5 ----
DEBUG 05-06 10:50:23.051678.051678 cuda_h.py:27] end *sagl cost 2.458 ms
DEBUG 05-06 10:50:23.056886.056886 cuda_h.py:27] end *layer_moe_fused cost 3.587 ms
DEBUG 05-06 10:50:23.056873.056873 cuda_h.py:27] end decode_layer cost 8.134 ms
DEBUG 05-06 10:50:23.056114.056114 lmp.py:904] ---- decode step 1 layer 6 ----
DEBUG 05-06 10:50:23.059516.059516 cuda_h.py:27] end *sagl cost 2.447 ms
DEBUG 05-06 10:50:23.064180.064180 cuda_h.py:27] end *layer_moe_fused cost 3.440 ms
DEBUG 05-06 10:50:23.064353.064353 cuda_h.py:27] end decode_layer cost 7.930 ms
DEBUG 05-06 10:50:23.064117.064117 lmp.py:904] ---- decode step 1 layer 7 ----
DEBUG 05-06 10:50:23.067963.067963 cuda_h.py:27] end *sagl cost 2.460 ms
DEBUG 05-06 10:50:23.072324.072324 cuda_h.py:27] end *layer_moe_fused cost 3.667 ms
DEBUG 05-06 10:50:23.073544.073544 cuda_h.py:27] end decode_layer cost 8.173 ms
DEBUG 05-06 10:50:23.073354.073354 lmp.py:904] ---- decode step 1 layer 8 ----
DEBUG 05-06 10:50:23.075344.075344 cuda_h.py:27] end *sagl cost 2.425 ms
DEBUG 05-06 10:50:23.080275.080275 cuda_h.py:27] end *layer_moe_fused cost 3.660 ms
DEBUG 05-06 10:50:23.081017.081017 cuda_h.py:27] end decode_layer cost 8.137 ms
DEBUG 05-06 10:50:23.081828.081828 lmp.py:904] ---- decode step 1 layer 9 ----
DEBUG 05-06 10:50:23.083304.083304 cuda_h.py:27] end *sagl cost 2.502 ms
DEBUG 05-06 10:50:23.088977.088977 cuda_h.py:27] end *layer_moe_fused cost 3.680 ms
DEBUG 05-06 10:50:23.089912.089912 cuda_h.py:27] end decode_layer cost 8.237 ms
DEBUG 05-06 10:50:23.089636.089636 lmp.py:904] ---- decode step 1 layer 10 ----
DEBUG 05-06 10:50:23.092111.092111 cuda_h.py:27] end *sagl cost 2.465 ms
DEBUG 05-06 10:50:23.097681.097681 cuda_h.py:27] end *layer_moe_fused cost 3.592 ms
DEBUG 05-06 10:50:23.097351.097351 cuda_h.py:27] end decode_layer cost 8.105 ms
DEBUG 05-06 10:50:23.097115.097115 lmp.py:904] ---- decode step 1 layer 11 ----
DEBUG 05-06 10:50:23.100373.100373 cuda_h.py:27] end *sagl cost 2.481 ms
DEBUG 05-06 10:50:23.105949.105949 cuda_h.py:27] end *layer_moe_fused cost 3.578 ms
DEBUG 05-06 10:50:23.105752.105752 cuda_h.py:27] end decode_layer cost 8.118 ms
DEBUG 05-06 10:50:23.105323.105323 lmp.py:904] ---- decode step 1 layer 12 ----
DEBUG 05-06 10:50:23.108329.108329 cuda_h.py:27] end *sagl cost 2.471 ms
DEBUG 05-06 10:50:23.113036.113036 cuda_h.py:27] end *layer_moe_fused cost 3.535 ms
DEBUG 05-06 10:50:23.114123.114123 cuda_h.py:27] end decode_layer cost 8.063 ms
DEBUG 05-06 10:50:23.114172.114172 lmp.py:904] ---- decode step 1 layer 13 ----
DEBUG 05-06 10:50:23.116954.116954 cuda_h.py:27] end *sagl cost 2.516 ms
DEBUG 05-06 10:50:23.121588.121588 cuda_h.py:27] end *layer_moe_fused cost 3.123 ms
DEBUG 05-06 10:50:23.121224.121224 cuda_h.py:27] end decode_layer cost 7.687 ms
DEBUG 05-06 10:50:23.121796.121796 lmp.py:904] ---- decode step 1 layer 14 ----
DEBUG 05-06 10:50:23.124152.124152 cuda_h.py:27] end *sagl cost 2.439 ms
DEBUG 05-06 10:50:23.128917.128917 cuda_h.py:27] end *layer_moe_fused cost 2.900 ms
DEBUG 05-06 10:50:23.129553.129553 cuda_h.py:27] end decode_layer cost 7.364 ms
DEBUG 05-06 10:50:23.129887.129887 lmp.py:904] ---- decode step 1 layer 15 ----
DEBUG 05-06 10:50:23.131203.131203 cuda_h.py:27] end *sagl cost 2.456 ms
DEBUG 05-06 10:50:23.136255.136255 cuda_h.py:27] end *layer_moe_fused cost 2.961 ms
DEBUG 05-06 10:50:23.136222.136222 cuda_h.py:27] end decode_layer cost 7.439 ms
DEBUG 05-06 10:50:23.136032.136032 lmp.py:904] ---- decode step 1 layer 16 ----
DEBUG 05-06 10:50:23.139387.139387 cuda_h.py:27] end *sagl cost 2.413 ms
DEBUG 05-06 10:50:23.143350.143350 cuda_h.py:27] end *layer_moe_fused cost 2.895 ms
DEBUG 05-06 10:50:23.144690.144690 cuda_h.py:27] end decode_layer cost 7.361 ms
DEBUG 05-06 10:50:23.144738.144738 lmp.py:904] ---- decode step 1 layer 17 ----
DEBUG 05-06 10:50:23.146026.146026 cuda_h.py:27] end *sagl cost 2.399 ms
DEBUG 05-06 10:50:23.151468.151468 cuda_h.py:27] end *layer_moe_fused cost 2.896 ms
DEBUG 05-06 10:50:23.151145.151145 cuda_h.py:27] end decode_layer cost 7.386 ms
DEBUG 05-06 10:50:23.151624.151624 lmp.py:904] ---- decode step 1 layer 18 ----
DEBUG 05-06 10:50:23.154455.154455 cuda_h.py:27] end *sagl cost 2.414 ms
DEBUG 05-06 10:50:23.158253.158253 cuda_h.py:27] end *layer_moe_fused cost 2.896 ms
DEBUG 05-06 10:50:23.159657.159657 cuda_h.py:27] end decode_layer cost 7.322 ms
DEBUG 05-06 10:50:23.159945.159945 lmp.py:904] ---- decode step 1 layer 19 ----
DEBUG 05-06 10:50:23.161944.161944 cuda_h.py:27] end *sagl cost 2.503 ms
DEBUG 05-06 10:50:23.165160.165160 cuda_h.py:27] end *layer_moe_fused cost 2.889 ms
DEBUG 05-06 10:50:23.166320.166320 cuda_h.py:27] end decode_layer cost 7.434 ms
DEBUG 05-06 10:50:23.166282.166282 lmp.py:904] ---- decode step 1 layer 20 ----
DEBUG 05-06 10:50:23.169691.169691 cuda_h.py:27] end *sagl cost 2.453 ms
DEBUG 05-06 10:50:23.173628.173628 cuda_h.py:27] end *layer_moe_fused cost 2.889 ms
DEBUG 05-06 10:50:23.174171.174171 cuda_h.py:27] end decode_layer cost 7.356 ms
DEBUG 05-06 10:50:23.174220.174220 lmp.py:904] ---- decode step 1 layer 21 ----
DEBUG 05-06 10:50:23.176683.176683 cuda_h.py:27] end *sagl cost 2.493 ms
DEBUG 05-06 10:50:23.180329.180329 cuda_h.py:27] end *layer_moe_fused cost 2.890 ms
DEBUG 05-06 10:50:23.181158.181158 cuda_h.py:27] end decode_layer cost 7.427 ms
DEBUG 05-06 10:50:23.181683.181683 lmp.py:904] ---- decode step 1 layer 22 ----
DEBUG 05-06 10:50:23.184780.184780 cuda_h.py:27] end *sagl cost 2.435 ms
DEBUG 05-06 10:50:23.188817.188817 cuda_h.py:27] end *layer_moe_fused cost 2.942 ms
DEBUG 05-06 10:50:23.188884.188884 cuda_h.py:27] end decode_layer cost 7.375 ms
DEBUG 05-06 10:50:23.189933.189933 lmp.py:904] ---- decode step 1 layer 23 ----
DEBUG 05-06 10:50:23.191584.191584 cuda_h.py:27] end *sagl cost 2.386 ms
DEBUG 05-06 10:50:23.195733.195733 cuda_h.py:27] end *layer_moe_fused cost 2.894 ms
DEBUG 05-06 10:50:23.196747.196747 cuda_h.py:27] end decode_layer cost 7.300 ms
DEBUG 05-06 10:50:23.196365.196365 lmp.py:904] ---- decode step 1 layer 24 ----
DEBUG 05-06 10:50:23.198123.198123 cuda_h.py:27] end *sagl cost 2.394 ms
DEBUG 05-06 10:50:23.203555.203555 cuda_h.py:27] end *layer_moe_fused cost 2.872 ms
DEBUG 05-06 10:50:23.203696.203696 cuda_h.py:27] end decode_layer cost 7.295 ms
DEBUG 05-06 10:50:23.203698.203698 lmp.py:904] ---- decode step 1 layer 25 ----
DEBUG 05-06 10:50:23.206862.206862 cuda_h.py:27] end *sagl cost 2.449 ms
DEBUG 05-06 10:50:23.210997.210997 cuda_h.py:27] end *layer_moe_fused cost 2.875 ms
DEBUG 05-06 10:50:23.211580.211580 cuda_h.py:27] end decode_layer cost 7.351 ms
DEBUG 05-06 10:50:23.211914.211914 lmp.py:904] ---- decode step 1 layer 26 ----
DEBUG 05-06 10:50:23.213374.213374 cuda_h.py:27] end *sagl cost 2.421 ms
DEBUG 05-06 10:50:23.217655.217655 cuda_h.py:27] end *layer_moe_fused cost 2.907 ms
DEBUG 05-06 10:50:23.218338.218338 cuda_h.py:27] end decode_layer cost 7.330 ms
DEBUG 05-06 10:50:23.218387.218387 lmp.py:904] ---- decode step 1 layer 27 ----
DEBUG 05-06 10:50:23.221665.221665 cuda_h.py:27] end *sagl cost 2.488 ms
DEBUG 05-06 10:50:23.225986.225986 cuda_h.py:27] end *layer_moe_fused cost 2.914 ms
DEBUG 05-06 10:50:23.226762.226762 cuda_h.py:27] end decode_layer cost 7.434 ms
DEBUG 05-06 10:50:23.226618.226618 lmp.py:904] ---- decode step 1 layer 28 ----
DEBUG 05-06 10:50:23.228662.228662 cuda_h.py:27] end *sagl cost 2.431 ms
DEBUG 05-06 10:50:23.232227.232227 cuda_h.py:27] end *layer_moe_fused cost 2.870 ms
DEBUG 05-06 10:50:23.233770.233770 cuda_h.py:27] end decode_layer cost 7.304 ms
DEBUG 05-06 10:50:23.233296.233296 lmp.py:904] ---- decode step 1 layer 29 ----
DEBUG 05-06 10:50:23.236604.236604 cuda_h.py:27] end *sagl cost 2.414 ms
DEBUG 05-06 10:50:23.240496.240496 cuda_h.py:27] end *layer_moe_fused cost 2.897 ms
DEBUG 05-06 10:50:23.240039.240039 cuda_h.py:27] end decode_layer cost 7.354 ms
DEBUG 05-06 10:50:23.240995.240995 cuda_h.py:27] end decode_step cost 242.473 ms
INFO 05-06 10:50:23.241288.241288 lmp.py:931] decode step 1 time: 0.24251484870910645 seconds
Time taken: 6.141082189977169 seconds
X512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x63ad57e2e490, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
CPUInfer[0x63ad055e43e0]: Goodbye
