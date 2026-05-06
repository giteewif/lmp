here pin
INFO 05-06 15:59:11.679877.679877 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-06 15:59:12.232503.232503 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-06 15:59:12.680076.680076 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-06 15:59:12.680996.680996 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 1.002s
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
INFO 05-06 15:59:20.167379.167379 mlpmodule.py:110] max attention head_dim=512 > 256 (e.g. Gemma4 global_head_dim): FlashAttention-2 unsupported, using sdpa
DEBUG 05-06 15:59:20.610409.610409 cuda_h.py:27] end init_cmv_hmv cost 443.655 ms
DEBUG 05-06 15:59:20.620555.620555 cuda_memory_view.py:1366] 
DEBUG 05-06 15:59:20.620555.620555 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.002530813217163086
DEBUG 05-06 15:59:20.637306.637306 mlpmodule.py:993] restore_hm_state_dict2model loaded 657 language_model tensors for Gemma4 model
DEBUG 05-06 15:59:20.637096.637096 cuda_memory_view.py:1370] 
DEBUG 05-06 15:59:20.637096.637096 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.0171358585357666
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-06 15:59:22.609799.609799 lmp.py:255] init kt-kernel layer 0 ok
INFO 05-06 15:59:23.555300.555300 lmp.py:255] init kt-kernel layer 1 ok
INFO 05-06 15:59:24.535951.535951 lmp.py:255] init kt-kernel layer 2 ok
INFO 05-06 15:59:25.484944.484944 lmp.py:255] init kt-kernel layer 3 ok
INFO 05-06 15:59:26.445187.445187 lmp.py:255] init kt-kernel layer 4 ok
INFO 05-06 15:59:27.423613.423613 lmp.py:255] init kt-kernel layer 5 ok
INFO 05-06 15:59:28.394656.394656 lmp.py:255] init kt-kernel layer 6 ok
INFO 05-06 15:59:29.391463.391463 lmp.py:255] init kt-kernel layer 7 ok
INFO 05-06 15:59:30.420944.420944 lmp.py:255] init kt-kernel layer 8 ok
INFO 05-06 15:59:31.524306.524306 lmp.py:255] init kt-kernel layer 9 ok
INFO 05-06 15:59:32.618052.618052 lmp.py:255] init kt-kernel layer 10 ok
INFO 05-06 15:59:33.726225.726225 lmp.py:255] init kt-kernel layer 11 ok
INFO 05-06 15:59:34.822740.822740 lmp.py:255] init kt-kernel layer 12 ok
INFO 05-06 15:59:35.935472.935472 lmp.py:255] init kt-kernel layer 13 ok
INFO 05-06 15:59:37.091031.091031 lmp.py:255] init kt-kernel layer 14 ok
INFO 05-06 15:59:38.082731.082731 lmp.py:255] init kt-kernel layer 15 ok
INFO 05-06 15:59:38.907094.907094 lmp.py:255] init kt-kernel layer 16 ok
INFO 05-06 15:59:39.722687.722687 lmp.py:255] init kt-kernel layer 17 ok
INFO 05-06 15:59:40.537194.537194 lmp.py:255] init kt-kernel layer 18 ok
INFO 05-06 15:59:41.350059.350059 lmp.py:255] init kt-kernel layer 19 ok
INFO 05-06 15:59:42.164086.164086 lmp.py:255] init kt-kernel layer 20 ok
INFO 05-06 15:59:42.984183.984183 lmp.py:255] init kt-kernel layer 21 ok
INFO 05-06 15:59:43.827160.827160 lmp.py:255] init kt-kernel layer 22 ok
CPUInfer[0x59320c224070]: Hello
WorkerPool[0x59320c1f6980] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x59321b9a56e0]: Hello
WorkerPool[0x59321bc09f00] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 6, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 7, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 8, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 9, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 10, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 11, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 12, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 13, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 14, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 15, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 16, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 17, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 18, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 19, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 20, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 21, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 22, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 23, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVINFO 05-06 15:59:44.642582.642582 lmp.py:255] init kt-kernel layer 23 ok
INFO 05-06 15:59:45.464591.464591 lmp.py:255] init kt-kernel layer 24 ok
INFO 05-06 15:59:46.296333.296333 lmp.py:255] init kt-kernel layer 25 ok
INFO 05-06 15:59:47.133988.133988 lmp.py:255] init kt-kernel layer 26 ok
INFO 05-06 15:59:47.999837.999837 lmp.py:255] init kt-kernel layer 27 ok
INFO 05-06 15:59:48.885184.885184 lmp.py:255] init kt-kernel layer 28 ok
INFO 05-06 15:59:49.763724.763724 lmp.py:255] init kt-kernel layer 29 ok
INFO 05-06 15:59:50.657875.657875 lmp.py:186] vLLM Triton fused-MoE enabled (CUDAGraph=False).
generate input ids cost 0.12020659446716309 s
DEBUG 05-06 15:59:53.752022.752022 cuda_h.py:27] end generate_input_ids cost 3075.759 ms
DEBUG 05-06 15:59:53.752755.752755 cuda_h.py:27] end init_cache cost 0.053 ms
INFO 05-06 15:59:53.763033.763033 lmp.py:367] _ensure_static_kv_cache (Gemma4 list): 30 layers, 3520.0 MiB on cuda:0
INFO 05-06 15:59:53.763064.763064 lmp.py:1162] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 15:59:53.777314.777314 lmp.py:2808] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 2915803076, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.8584664929204888, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-06 15:59:53.777603.777603 lmp.py:2826] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.777478.777478 lmp.py:2826] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.777785.777785 lmp.py:2826] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.777932.777932 lmp.py:2826] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.777588.777588 lmp.py:2826] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.777450.777450 lmp.py:2826] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.777690.777690 lmp.py:2826] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778842.778842 lmp.py:2826] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778750.778750 lmp.py:2826] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778990.778990 lmp.py:2826] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778774.778774 lmp.py:2826] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778225.778225 lmp.py:2826] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778610.778610 lmp.py:2826] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778565.778565 lmp.py:2826] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778418.778418 lmp.py:2826] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778565.778565 lmp.py:2826] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.778805.778805 lmp.py:2826] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779274.779274 lmp.py:2826] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779183.779183 lmp.py:2826] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779237.779237 lmp.py:2826] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779475.779475 lmp.py:2826] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779906.779906 lmp.py:2826] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779385.779385 lmp.py:2826] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779648.779648 lmp.py:2826] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779941.779941 lmp.py:2826] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779657.779657 lmp.py:2826] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779166.779166 lmp.py:2826] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779505.779505 lmp.py:2826] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.779937.779937 lmp.py:2826] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 15:59:53.780632.780632 lmp.py:2826] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 15:59:54.079031.079031 cuda_h.py:27] end init_loading_placement cost 315.981 ms
DEBUG 05-06 15:59:54.079515.079515 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 15:59:54.079306.079306 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 15:59:54 client.py:72] load_into_gpu: gemma4-26B-A4B, caf614e7-6fb8-411c-ba09-24dedab7a4bd
INFO 05-06 15:59:54 client.py:135] Model loaded: gemma4-26B-A4B, caf614e7-6fb8-411c-ba09-24dedab7a4bd
INFO 05-06 15:59:54 client.py:204] confirm_model_loaded: gemma4-26B-A4B, caf614e7-6fb8-411c-ba09-24dedab7a4bd
INFO 05-06 15:59:54 client.py:212] Model loaded
DEBUG 05-06 15:59:54.610878.610878 cuda_h.py:27] end init_general_sagl_loading_async cost 531.092 ms
INFO 05-06 15:59:54.659901.659901 lmp.py:3329] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 15:59:54.763167.763167 cuda_h.py:27] end restore_state_dict cost 103.851 ms
WARNING 05-06 15:59:54 [fused_moe.py:1090] Using default MoE config. Performance might be sub-optimal! Config file not found at /mnt/zhengcf3/lmp_env/fslmp/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=32,N=704,device_name=NVIDIA_GeForce_RTX_4090.json
INFO 05-06 15:59:55.826316.826316 lmp.py:1291] vLLM Triton pre-warmup done in 1063.2 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 15:59:55.826820.826820 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 15:59:55.826246.826246 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 15:59:55 client.py:72] load_into_gpu: gemma4-26B-A4B, c567d8a5-77a8-40a8-b4ec-2429a37316a3
INFO 05-06 15:59:55 client.py:135] Model loaded: gemma4-26B-A4B, c567d8a5-77a8-40a8-b4ec-2429a37316a3
DEBUG 05-06 15:59:55.900988.900988 cuda_h.py:27] end init_experts_loading_async cost 73.850 ms
DEBUG 05-06 15:59:55.925339.925339 cuda_h.py:27] end init_inputs_tokens cost 24.249 ms
DEBUG 05-06 15:59:55.925138.925138 lmp.py:1350] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 15:59:55.995255.995255 cuda_h.py:27] end prefill_ln cost 69.678 ms
DEBUG 05-06 15:59:56.075263.075263 cuda_h.py:27] end prefill_attn cost 79.804 ms
DEBUG 05-06 15:59:56.075745.075745 cuda_h.py:27] end prefill_ffn_prep cost 0.375 ms
DEBUG 05-06 15:59:56.157851.157851 cuda_h.py:27] end prefill_gate cost 74.843 ms
INFO 05-06 15:59:56.188391.188391 lmp.py:1823] [layer_moe_fused] layer=0 active_experts=118 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 10162, 'token_per_expert': {3: 376, 7: 688, 11: 47, 15: 30, 19: 11, 23: 104, 27: 25, 31: 316, 39: 1391, 43: 2, 47: 2548, 51: 345, 55: 445, 59: 76, 63: 39, 67: 340, 71: 98, 75: 171, 79: 164, 83: 147, 87: 15, 91: 946, 99: 318, 103: 877, 107: 43, 111: 84, 115: 171, 119: 17, 123: 132, 127: 196}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 48, 52, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 6978, 'token_per_expert': {0: 446, 4: 26, 8: 26, 12: 4, 16: 364, 20: 85, 24: 135, 28: 228, 32: 390, 36: 8, 44: 62, 48: 300, 52: 353, 60: 182, 64: 202, 68: 1322, 72: 136, 76: 147, 80: 29, 84: 60, 88: 1, 92: 135, 96: 14, 100: 11, 104: 288, 108: 206, 112: 142, 116: 189, 120: 21, 124: 1466}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 7647, 'token_per_expert': {1: 579, 5: 121, 9: 128, 13: 115, 17: 8, 21: 317, 25: 253, 29: 13, 33: 1632, 37: 154, 41: 262, 45: 59, 49: 63, 53: 1710, 65: 70, 69: 153, 73: 129, 77: 224, 81: 24, 85: 6, 89: 279, 93: 37, 97: 6, 101: 14, 105: 154, 109: 4, 113: 331, 117: 137, 121: 478, 125: 187}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 46, 50, 54, 58, 66, 70, 74, 78, 86, 90, 94, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 7981, 'token_per_expert': {2: 59, 6: 10, 10: 80, 14: 76, 18: 109, 22: 483, 26: 628, 30: 2, 34: 118, 38: 185, 46: 816, 50: 944, 54: 541, 58: 7, 66: 9, 70: 306, 74: 462, 78: 218, 86: 9, 90: 1092, 94: 41, 102: 106, 106: 27, 110: 114, 114: 152, 118: 116, 122: 264, 126: 1007}}
INFO 05-06 15:59:56.188134.188134 lmp.py:1845] [layer_moe_fused] layer=0 prefix: 30.654ms alloc: 0.331ms
INFO 05-06 15:59:56.188554.188554 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 7.43865966796875e-05 seconds
INFO 05-06 15:59:56.191955.191955 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.002063751220703125s
INFO 05-06 15:59:56.192219.192219 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010609626770019531s
DEBUG 05-06 15:59:56.192071.192071 cuda_h.py:27] end moe_wait_copy_tasks cost 1.201 ms
DEBUG 05-06 15:59:56.247258.247258 cuda_h.py:27] end moe_vllm_forward cost 55.001 ms
DEBUG 05-06 15:59:56.247791.247791 cuda_h.py:27] end moe_shared_experts cost 0.009 ms
INFO 05-06 15:59:56.247747.247747 lmp.py:1964] [layer_moe_fused] vllm triton time: 55.419ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.248327.248327 cuda_h.py:27] end *layer_moe_fused cost 90.551 ms
DEBUG 05-06 15:59:56.250198.250198 cuda_h.py:27] end prefill_merge_scale cost 1.809 ms
DEBUG 05-06 15:59:56.250440.250440 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.047 ms
DEBUG 05-06 15:59:56.250991.250991 cuda_h.py:27] end prefill_layer cost 324.955 ms
DEBUG 05-06 15:59:56.250847.250847 lmp.py:1394] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 15:59:56.250404.250404 lmp.py:1350] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 15:59:56.250370.250370 cuda_h.py:27] end prefill_ln cost 0.178 ms
DEBUG 05-06 15:59:56.258657.258657 cuda_h.py:27] end prefill_attn cost 7.904 ms
DEBUG 05-06 15:59:56.259735.259735 cuda_h.py:27] end prefill_ffn_prep cost 0.308 ms
DEBUG 05-06 15:59:56.260885.260885 cuda_h.py:27] end prefill_gate cost 0.402 ms
INFO 05-06 15:59:56.264310.264310 lmp.py:1823] [layer_moe_fused] layer=1 active_experts=122 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 23, 27, 31, 35, 39, 47, 51, 55, 59, 63, 67, 71, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 31, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 6041, 'token_per_expert': {3: 47, 7: 120, 11: 203, 15: 111, 23: 35, 27: 321, 31: 116, 35: 78, 39: 43, 47: 664, 51: 586, 55: 111, 59: 422, 63: 2, 67: 144, 71: 79, 79: 422, 83: 208, 87: 49, 91: 16, 95: 69, 99: 772, 103: 221, 107: 30, 111: 15, 115: 217, 119: 530, 123: 249, 127: 161}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 9602, 'token_per_expert': {0: 578, 4: 340, 8: 1552, 12: 320, 16: 60, 20: 1109, 24: 30, 28: 10, 32: 39, 36: 23, 40: 79, 44: 9, 48: 33, 52: 689, 56: 80, 60: 203, 64: 523, 68: 162, 72: 475, 76: 21, 80: 1027, 84: 224, 88: 48, 92: 116, 96: 424, 100: 328, 104: 520, 108: 135, 112: 29, 116: 138, 120: 131, 124: 147}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 8041, 'token_per_expert': {1: 89, 5: 133, 9: 624, 13: 782, 21: 148, 25: 537, 29: 151, 33: 29, 37: 296, 41: 107, 45: 134, 49: 248, 53: 258, 57: 141, 61: 17, 65: 421, 69: 102, 73: 262, 77: 126, 81: 131, 85: 373, 89: 542, 93: 459, 97: 782, 101: 185, 105: 90, 109: 477, 113: 34, 117: 227, 121: 83, 125: 53}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 9084, 'token_per_expert': {2: 17, 6: 888, 10: 302, 14: 9, 18: 82, 22: 695, 26: 228, 30: 642, 34: 249, 38: 173, 42: 422, 46: 400, 50: 224, 54: 36, 62: 69, 66: 109, 70: 7, 74: 278, 78: 117, 82: 812, 86: 35, 90: 231, 94: 414, 98: 415, 102: 196, 106: 426, 110: 150, 114: 85, 118: 815, 122: 558}}
INFO 05-06 15:59:56.264938.264938 lmp.py:1845] [layer_moe_fused] layer=1 prefix: 3.442ms alloc: 0.298ms
INFO 05-06 15:59:56.264508.264508 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 6.67572021484375e-05 seconds
INFO 05-06 15:59:56.266530.266530 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0015931129455566406s
INFO 05-06 15:59:56.267565.267565 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011937618255615234s
DEBUG 05-06 15:59:56.267320.267320 cuda_h.py:27] end moe_wait_copy_tasks cost 1.402 ms
DEBUG 05-06 15:59:56.273717.273717 cuda_h.py:27] end moe_vllm_forward cost 5.550 ms
DEBUG 05-06 15:59:56.273182.273182 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:59:56.273363.273363 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.858ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.273486.273486 cuda_h.py:27] end *layer_moe_fused cost 13.169 ms
DEBUG 05-06 15:59:56.274228.274228 cuda_h.py:27] end prefill_merge_scale cost 0.302 ms
DEBUG 05-06 15:59:56.274019.274019 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.035 ms
DEBUG 05-06 15:59:56.274657.274657 cuda_h.py:27] end prefill_layer cost 24.167 ms
DEBUG 05-06 15:59:56.274094.274094 lmp.py:1394] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 15:59:56.274367.274367 lmp.py:1350] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 15:59:56.275525.275525 cuda_h.py:27] end prefill_ln cost 0.158 ms
DEBUG 05-06 15:59:56.320590.320590 cuda_h.py:27] end prefill_attn cost 45.397 ms
DEBUG 05-06 15:59:56.320576.320576 cuda_h.py:27] end prefill_ffn_prep cost 0.309 ms
DEBUG 05-06 15:59:56.321377.321377 cuda_h.py:27] end prefill_gate cost 0.330 ms
INFO 05-06 15:59:56.326444.326444 lmp.py:1823] [layer_moe_fused] layer=2 active_experts=123 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 9300, 'token_per_expert': {3: 327, 7: 199, 11: 1029, 15: 420, 19: 1121, 23: 28, 27: 19, 31: 78, 35: 779, 39: 2, 43: 1015, 47: 19, 51: 56, 55: 90, 59: 658, 63: 31, 67: 54, 71: 288, 75: 16, 79: 1, 83: 1323, 87: 133, 91: 70, 95: 219, 99: 116, 103: 92, 107: 163, 111: 10, 115: 360, 119: 76, 123: 213, 127: 295}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 8349, 'token_per_expert': {0: 1120, 4: 106, 8: 76, 12: 32, 16: 99, 20: 169, 24: 41, 28: 203, 32: 2, 36: 96, 40: 58, 44: 94, 48: 174, 52: 155, 56: 182, 60: 348, 64: 159, 68: 85, 72: 707, 76: 400, 80: 913, 84: 130, 88: 185, 92: 7, 96: 120, 100: 164, 104: 122, 108: 1909, 116: 36, 120: 113, 124: 344}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 93, 97, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 11254, 'token_per_expert': {1: 1938, 5: 52, 9: 1506, 13: 537, 17: 213, 21: 10, 25: 45, 29: 567, 33: 99, 37: 331, 41: 254, 45: 55, 49: 292, 53: 40, 57: 79, 61: 38, 65: 313, 69: 438, 73: 252, 77: 248, 81: 1485, 85: 327, 93: 24, 97: 170, 105: 12, 109: 1736, 113: 27, 117: 10, 121: 5, 125: 151}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3865, 'token_per_expert': {2: 6, 6: 1, 10: 13, 14: 47, 18: 73, 22: 13, 26: 41, 34: 441, 38: 6, 42: 276, 46: 147, 50: 222, 54: 88, 58: 92, 62: 290, 66: 90, 70: 62, 74: 47, 78: 13, 82: 96, 86: 34, 90: 162, 98: 17, 102: 601, 106: 280, 110: 224, 114: 65, 118: 71, 122: 177, 126: 170}}
INFO 05-06 15:59:56.327945.327945 lmp.py:1845] [layer_moe_fused] layer=2 prefix: 4.811ms alloc: 0.277ms
INFO 05-06 15:59:56.327767.327767 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.8650970458984375e-05 seconds
INFO 05-06 15:59:56.328356.328356 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0015189647674560547s
INFO 05-06 15:59:56.330402.330402 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001130819320678711s
DEBUG 05-06 15:59:56.330002.330002 cuda_h.py:27] end moe_wait_copy_tasks cost 1.224 ms
DEBUG 05-06 15:59:56.335053.335053 cuda_h.py:27] end moe_vllm_forward cost 5.485 ms
DEBUG 05-06 15:59:56.335711.335711 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:59:56.335369.335369 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.785ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.336305.336305 cuda_h.py:27] end *layer_moe_fused cost 14.064 ms
DEBUG 05-06 15:59:56.336343.336343 cuda_h.py:27] end prefill_merge_scale cost 0.308 ms
DEBUG 05-06 15:59:56.336465.336465 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.037 ms
DEBUG 05-06 15:59:56.336625.336625 cuda_h.py:27] end prefill_layer cost 61.917 ms
DEBUG 05-06 15:59:56.336252.336252 lmp.py:1394] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 15:59:56.336570.336570 lmp.py:1350] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 15:59:56.337497.337497 cuda_h.py:27] end prefill_ln cost 0.164 ms
DEBUG 05-06 15:59:56.381356.381356 cuda_h.py:27] end prefill_attn cost 44.577 ms
DEBUG 05-06 15:59:56.382530.382530 cuda_h.py:27] end prefill_ffn_prep cost 0.368 ms
DEBUG 05-06 15:59:56.383352.383352 cuda_h.py:27] end prefill_gate cost 0.396 ms
INFO 05-06 15:59:56.387973.387973 lmp.py:1823] [layer_moe_fused] layer=3 active_experts=122 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 23, 27, 31, 35, 39, 43, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 31, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 5825, 'token_per_expert': {3: 125, 11: 245, 15: 382, 19: 269, 23: 260, 27: 24, 31: 30, 35: 117, 39: 263, 43: 287, 51: 409, 55: 25, 59: 176, 63: 141, 67: 619, 71: 358, 75: 312, 79: 110, 83: 108, 87: 114, 91: 547, 99: 2, 103: 79, 107: 359, 111: 7, 115: 3, 119: 177, 123: 82, 127: 195}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 10561, 'token_per_expert': {0: 121, 4: 849, 8: 161, 12: 67, 16: 862, 20: 550, 24: 95, 28: 1575, 32: 6, 36: 4, 40: 76, 44: 1042, 48: 4, 52: 1003, 56: 156, 60: 419, 64: 46, 68: 79, 72: 23, 76: 8, 80: 67, 84: 999, 88: 503, 92: 812, 96: 273, 100: 244, 104: 118, 108: 209, 112: 2, 116: 52, 120: 136}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 7530, 'token_per_expert': {1: 421, 5: 127, 9: 483, 17: 384, 21: 124, 25: 328, 29: 214, 33: 212, 37: 52, 41: 126, 45: 2, 49: 100, 53: 390, 57: 53, 61: 172, 65: 15, 69: 296, 73: 101, 77: 896, 81: 236, 85: 1762, 89: 8, 93: 132, 97: 16, 101: 334, 105: 6, 109: 177, 113: 13, 117: 142, 121: 74, 125: 134}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 8852, 'token_per_expert': {2: 172, 6: 401, 10: 259, 14: 196, 18: 127, 22: 221, 26: 356, 30: 346, 34: 87, 38: 45, 42: 56, 46: 28, 50: 437, 54: 12, 58: 123, 62: 1567, 66: 523, 70: 163, 74: 52, 78: 52, 82: 28, 86: 102, 90: 9, 94: 1443, 98: 38, 102: 1486, 106: 19, 110: 156, 114: 259, 118: 73, 122: 16}}
INFO 05-06 15:59:56.387891.387891 lmp.py:1845] [layer_moe_fused] layer=3 prefix: 3.715ms alloc: 0.267ms
INFO 05-06 15:59:56.387403.387403 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 6.699562072753906e-05 seconds
INFO 05-06 15:59:56.389969.389969 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014309883117675781s
INFO 05-06 15:59:56.390202.390202 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009510517120361328s
DEBUG 05-06 15:59:56.390701.390701 cuda_h.py:27] end moe_wait_copy_tasks cost 1.041 ms
DEBUG 05-06 15:59:56.395572.395572 cuda_h.py:27] end moe_vllm_forward cost 5.415 ms
DEBUG 05-06 15:59:56.395329.395329 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:56.395464.395464 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.725ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.396102.396102 cuda_h.py:27] end *layer_moe_fused cost 12.674 ms
DEBUG 05-06 15:59:56.396446.396446 cuda_h.py:27] end prefill_merge_scale cost 0.301 ms
DEBUG 05-06 15:59:56.396283.396283 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.037 ms
DEBUG 05-06 15:59:56.396967.396967 cuda_h.py:27] end prefill_layer cost 59.912 ms
DEBUG 05-06 15:59:56.397294.397294 lmp.py:1394] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 15:59:56.397375.397375 lmp.py:1350] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 15:59:56.397586.397586 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 15:59:56.442663.442663 cuda_h.py:27] end prefill_attn cost 45.513 ms
DEBUG 05-06 15:59:56.443158.443158 cuda_h.py:27] end prefill_ffn_prep cost 0.301 ms
DEBUG 05-06 15:59:56.444331.444331 cuda_h.py:27] end prefill_gate cost 0.310 ms
INFO 05-06 15:59:56.449789.449789 lmp.py:1823] [layer_moe_fused] layer=4 active_experts=124 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 13486, 'token_per_expert': {3: 204, 7: 60, 15: 149, 19: 216, 23: 2017, 27: 955, 31: 370, 35: 284, 39: 30, 43: 177, 47: 458, 51: 389, 55: 1884, 59: 354, 63: 667, 67: 904, 71: 335, 75: 382, 79: 65, 83: 268, 87: 345, 91: 24, 95: 13, 99: 1, 103: 479, 107: 220, 111: 186, 115: 70, 119: 1861, 123: 38, 127: 81}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 6951, 'token_per_expert': {0: 16, 4: 12, 8: 1548, 12: 39, 16: 52, 20: 85, 24: 137, 28: 270, 32: 747, 36: 191, 40: 157, 44: 40, 52: 7, 56: 471, 60: 27, 64: 106, 68: 119, 72: 393, 76: 67, 80: 222, 84: 263, 88: 217, 92: 292, 96: 25, 100: 356, 108: 36, 112: 49, 116: 369, 120: 227, 124: 411}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 6320, 'token_per_expert': {1: 1241, 5: 472, 9: 16, 13: 20, 17: 249, 21: 70, 25: 2, 29: 197, 33: 15, 37: 338, 41: 191, 45: 124, 49: 201, 53: 103, 57: 22, 61: 359, 65: 44, 69: 97, 73: 17, 77: 343, 81: 124, 85: 151, 89: 313, 93: 193, 97: 135, 101: 112, 105: 553, 109: 92, 113: 294, 117: 57, 121: 29, 125: 146}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 6011, 'token_per_expert': {2: 620, 6: 4, 14: 6, 18: 76, 22: 253, 26: 555, 30: 246, 34: 74, 38: 94, 42: 7, 46: 39, 50: 35, 54: 151, 58: 7, 62: 92, 66: 79, 70: 300, 74: 1866, 78: 109, 82: 115, 86: 180, 90: 299, 94: 300, 98: 53, 102: 3, 106: 136, 110: 21, 114: 1, 118: 6, 122: 69, 126: 215}}
INFO 05-06 15:59:56.449661.449661 lmp.py:1845] [layer_moe_fused] layer=4 prefix: 4.816ms alloc: 0.270ms
INFO 05-06 15:59:56.449021.449021 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.841255187988281e-05 seconds
INFO 05-06 15:59:56.451932.451932 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012664794921875s
INFO 05-06 15:59:56.452858.452858 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010838508605957031s
DEBUG 05-06 15:59:56.452265.452265 cuda_h.py:27] end moe_wait_copy_tasks cost 1.177 ms
DEBUG 05-06 15:59:56.457255.457255 cuda_h.py:27] end moe_vllm_forward cost 5.447 ms
DEBUG 05-06 15:59:56.458813.458813 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:59:56.458325.458325 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.735ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.458586.458586 cuda_h.py:27] end *layer_moe_fused cost 13.740 ms
DEBUG 05-06 15:59:56.458402.458402 cuda_h.py:27] end prefill_merge_scale cost 0.304 ms
DEBUG 05-06 15:59:56.458425.458425 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.033 ms
DEBUG 05-06 15:59:56.458678.458678 cuda_h.py:27] end prefill_layer cost 61.880 ms
DEBUG 05-06 15:59:56.459689.459689 lmp.py:1394] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 15:59:56.459008.459008 lmp.py:1350] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 15:59:56.459358.459358 cuda_h.py:27] end prefill_ln cost 0.160 ms
DEBUG 05-06 15:59:56.507368.507368 cuda_h.py:27] end prefill_attn cost 47.708 ms
DEBUG 05-06 15:59:56.507369.507369 cuda_h.py:27] end prefill_ffn_prep cost 0.337 ms
DEBUG 05-06 15:59:56.508888.508888 cuda_h.py:27] end prefill_gate cost 0.289 ms
INFO 05-06 15:59:56.513621.513621 lmp.py:1823] [layer_moe_fused] layer=5 active_experts=117 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 15, 19, 23, 31, 39, 43, 47, 51, 55, 63, 67, 71, 75, 83, 87, 91, 95, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 26, 'ideal_gpu_count': 30, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 6473, 'token_per_expert': {7: 497, 11: 3, 15: 111, 19: 63, 23: 49, 31: 24, 39: 444, 43: 80, 47: 117, 51: 226, 55: 142, 63: 78, 67: 265, 71: 1205, 75: 301, 83: 136, 87: 123, 91: 4, 95: 10, 99: 914, 107: 25, 111: 515, 115: 5, 119: 277, 123: 688, 127: 171}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 112, 116, 120, 124], 'expert_count': 29, 'ideal_gpu_count': 29, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 9768, 'token_per_expert': {0: 69, 4: 147, 8: 64, 16: 696, 20: 97, 24: 163, 28: 533, 32: 46, 36: 78, 44: 386, 48: 281, 52: 57, 56: 2749, 60: 43, 64: 5, 68: 117, 72: 1455, 76: 6, 80: 15, 84: 194, 88: 32, 92: 311, 96: 83, 100: 464, 104: 672, 112: 190, 116: 308, 120: 448, 124: 59}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 29, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 9377, 'token_per_expert': {1: 7, 5: 121, 9: 541, 13: 637, 17: 1264, 21: 14, 29: 57, 33: 21, 37: 50, 41: 231, 45: 825, 49: 52, 53: 9, 57: 574, 61: 276, 65: 5, 69: 3, 73: 2303, 77: 81, 81: 63, 85: 2, 89: 408, 93: 88, 97: 45, 101: 651, 105: 8, 109: 1, 113: 1014, 117: 23, 121: 1, 125: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 29, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 7150, 'token_per_expert': {2: 205, 6: 163, 10: 125, 14: 518, 18: 18, 22: 117, 26: 1, 30: 124, 34: 44, 38: 86, 42: 309, 46: 68, 50: 1599, 54: 15, 58: 27, 66: 11, 70: 193, 74: 565, 78: 21, 82: 113, 86: 194, 90: 22, 94: 1087, 98: 83, 102: 47, 106: 180, 110: 3, 114: 32, 118: 260, 122: 9, 126: 911}}
INFO 05-06 15:59:56.513778.513778 lmp.py:1845] [layer_moe_fused] layer=5 prefix: 5.132ms alloc: 0.269ms
INFO 05-06 15:59:56.514138.514138 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.793571472167969e-05 seconds
INFO 05-06 15:59:56.515858.515858 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0013015270233154297s
INFO 05-06 15:59:56.516214.516214 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010602474212646484s
DEBUG 05-06 15:59:56.516945.516945 cuda_h.py:27] end moe_wait_copy_tasks cost 1.146 ms
DEBUG 05-06 15:59:56.522695.522695 cuda_h.py:27] end moe_vllm_forward cost 5.377 ms
DEBUG 05-06 15:59:56.522969.522969 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:56.522719.522719 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.665ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.522649.522649 cuda_h.py:27] end *layer_moe_fused cost 14.015 ms
DEBUG 05-06 15:59:56.523063.523063 cuda_h.py:27] end prefill_merge_scale cost 0.302 ms
DEBUG 05-06 15:59:56.523324.523324 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.033 ms
DEBUG 05-06 15:59:56.523054.523054 cuda_h.py:27] end prefill_layer cost 64.288 ms
DEBUG 05-06 15:59:56.523496.523496 lmp.py:1394] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 15:59:56.523530.523530 lmp.py:1350] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 15:59:56.523603.523603 cuda_h.py:27] end prefill_ln cost 0.166 ms
DEBUG 05-06 15:59:56.569029.569029 cuda_h.py:27] end prefill_attn cost 45.907 ms
DEBUG 05-06 15:59:56.570671.570671 cuda_h.py:27] end prefill_ffn_prep cost 0.303 ms
DEBUG 05-06 15:59:56.571913.571913 cuda_h.py:27] end prefill_gate cost 0.303 ms
INFO 05-06 15:59:56.575739.575739 lmp.py:1823] [layer_moe_fused] layer=6 active_experts=120 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 11216, 'token_per_expert': {3: 41, 7: 4, 11: 85, 15: 27, 19: 304, 23: 353, 27: 170, 31: 23, 35: 1001, 39: 28, 47: 37, 51: 99, 55: 16, 59: 1, 63: 4, 67: 181, 71: 13, 75: 703, 79: 19, 83: 13, 87: 1229, 91: 93, 95: 482, 99: 3623, 103: 365, 107: 206, 111: 217, 115: 187, 119: 1097, 123: 325, 127: 270}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 7649, 'token_per_expert': {0: 1768, 4: 34, 8: 29, 12: 6, 16: 359, 20: 11, 24: 73, 28: 61, 32: 182, 36: 476, 44: 146, 48: 6, 52: 4, 56: 163, 60: 128, 64: 10, 68: 149, 72: 27, 76: 2, 80: 116, 84: 6, 88: 53, 92: 2, 96: 340, 100: 509, 104: 197, 108: 2394, 112: 216, 116: 51, 120: 87, 124: 44}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 45, 53, 57, 61, 65, 69, 77, 81, 85, 89, 93, 101, 105, 109, 113, 117, 121], 'expert_count': 27, 'ideal_gpu_count': 30, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 5549, 'token_per_expert': {1: 70, 5: 406, 9: 85, 13: 283, 17: 41, 21: 12, 25: 997, 29: 19, 33: 6, 37: 619, 45: 64, 53: 49, 57: 44, 61: 206, 65: 253, 69: 417, 77: 230, 81: 37, 85: 122, 89: 143, 93: 244, 101: 637, 105: 132, 109: 34, 113: 27, 117: 145, 121: 227}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 8354, 'token_per_expert': {2: 429, 6: 295, 10: 50, 14: 29, 18: 202, 22: 427, 26: 302, 30: 194, 34: 823, 38: 38, 42: 493, 46: 80, 50: 216, 58: 460, 62: 45, 66: 1, 70: 129, 74: 47, 78: 46, 82: 115, 86: 1516, 90: 243, 94: 536, 98: 25, 102: 43, 106: 290, 110: 237, 114: 107, 118: 2, 122: 16, 126: 918}}
INFO 05-06 15:59:56.575128.575128 lmp.py:1845] [layer_moe_fused] layer=6 prefix: 4.146ms alloc: 0.262ms
INFO 05-06 15:59:56.575567.575567 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.745887756347656e-05 seconds
INFO 05-06 15:59:56.577907.577907 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014383792877197266s
INFO 05-06 15:59:56.578957.578957 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0008177757263183594s
DEBUG 05-06 15:59:56.578595.578595 cuda_h.py:27] end moe_wait_copy_tasks cost 0.905 ms
DEBUG 05-06 15:59:56.584726.584726 cuda_h.py:27] end moe_vllm_forward cost 5.688 ms
DEBUG 05-06 15:59:56.584192.584192 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:56.584373.584373 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.983ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.584926.584926 cuda_h.py:27] end *layer_moe_fused cost 13.215 ms
DEBUG 05-06 15:59:56.584084.584084 cuda_h.py:27] end prefill_merge_scale cost 0.303 ms
DEBUG 05-06 15:59:56.585014.585014 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:56.585744.585744 cuda_h.py:27] end prefill_layer cost 61.568 ms
DEBUG 05-06 15:59:56.585130.585130 lmp.py:1394] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 15:59:56.585926.585926 lmp.py:1350] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 15:59:56.585329.585329 cuda_h.py:27] end prefill_ln cost 0.164 ms
DEBUG 05-06 15:59:56.630330.630330 cuda_h.py:27] end prefill_attn cost 45.046 ms
DEBUG 05-06 15:59:56.631839.631839 cuda_h.py:27] end prefill_ffn_prep cost 0.302 ms
DEBUG 05-06 15:59:56.632568.632568 cuda_h.py:27] end prefill_gate cost 0.394 ms
INFO 05-06 15:59:56.636722.636722 lmp.py:1823] [layer_moe_fused] layer=7 active_experts=120 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 30, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 10623, 'token_per_expert': {3: 11, 7: 273, 11: 8, 15: 151, 19: 128, 23: 257, 27: 168, 31: 265, 35: 146, 39: 6, 43: 819, 47: 342, 51: 265, 55: 206, 59: 12, 63: 33, 67: 158, 71: 345, 75: 12, 79: 330, 83: 1074, 87: 28, 91: 1386, 95: 1, 99: 137, 103: 1573, 107: 769, 111: 285, 115: 175, 119: 1147, 123: 36, 127: 77}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5340, 'token_per_expert': {0: 86, 4: 1, 8: 136, 12: 244, 16: 37, 20: 180, 24: 55, 28: 315, 32: 213, 36: 6, 40: 7, 44: 189, 48: 21, 52: 9, 56: 246, 60: 3, 64: 230, 68: 450, 72: 579, 76: 19, 80: 310, 84: 72, 88: 40, 92: 669, 96: 93, 104: 267, 108: 64, 112: 251, 116: 89, 120: 417, 124: 42}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 89, 97, 101, 105, 113, 117, 121, 125], 'expert_count': 28, 'ideal_gpu_count': 30, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 9479, 'token_per_expert': {1: 1, 5: 123, 9: 34, 13: 249, 17: 201, 21: 158, 25: 97, 29: 597, 33: 91, 41: 229, 45: 125, 49: 19, 53: 794, 57: 294, 61: 503, 65: 489, 69: 2933, 73: 13, 77: 215, 81: 2, 89: 257, 97: 117, 101: 175, 105: 200, 113: 416, 117: 12, 121: 301, 125: 834}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 26, 30, 34, 38, 42, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 7326, 'token_per_expert': {2: 11, 6: 18, 10: 417, 14: 465, 18: 254, 26: 117, 30: 20, 34: 12, 38: 16, 42: 178, 50: 2, 54: 291, 58: 107, 62: 3, 66: 81, 70: 489, 74: 367, 78: 8, 82: 659, 86: 635, 90: 369, 98: 12, 102: 1364, 106: 239, 110: 65, 114: 495, 118: 320, 122: 199, 126: 113}}
INFO 05-06 15:59:56.636753.636753 lmp.py:1845] [layer_moe_fused] layer=7 prefix: 3.621ms alloc: 0.279ms
INFO 05-06 15:59:56.636583.636583 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 6.127357482910156e-05 seconds
INFO 05-06 15:59:56.638833.638833 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014786720275878906s
INFO 05-06 15:59:56.639980.639980 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011680126190185547s
DEBUG 05-06 15:59:56.639063.639063 cuda_h.py:27] end moe_wait_copy_tasks cost 1.268 ms
DEBUG 05-06 15:59:56.646234.646234 cuda_h.py:27] end moe_vllm_forward cost 6.858 ms
DEBUG 05-06 15:59:56.646534.646534 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:59:56.646299.646299 lmp.py:1964] [layer_moe_fused] vllm triton time: 7.229ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.647532.647532 cuda_h.py:27] end *layer_moe_fused cost 14.686 ms
DEBUG 05-06 15:59:56.647626.647626 cuda_h.py:27] end prefill_merge_scale cost 0.498 ms
DEBUG 05-06 15:59:56.647425.647425 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.055 ms
DEBUG 05-06 15:59:56.648424.648424 cuda_h.py:27] end prefill_layer cost 62.755 ms
DEBUG 05-06 15:59:56.648737.648737 lmp.py:1394] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 15:59:56.648391.648391 lmp.py:1350] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 15:59:56.648872.648872 cuda_h.py:27] end prefill_ln cost 0.259 ms
DEBUG 05-06 15:59:56.691634.691634 cuda_h.py:27] end prefill_attn cost 42.656 ms
DEBUG 05-06 15:59:56.691375.691375 cuda_h.py:27] end prefill_ffn_prep cost 0.306 ms
DEBUG 05-06 15:59:56.692413.692413 cuda_h.py:27] end prefill_gate cost 0.302 ms
INFO 05-06 15:59:56.697941.697941 lmp.py:1823] [layer_moe_fused] layer=8 active_experts=115 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 29, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 7797, 'token_per_expert': {3: 268, 7: 39, 11: 24, 15: 1015, 19: 362, 23: 31, 27: 488, 31: 42, 35: 41, 39: 3, 43: 73, 47: 509, 51: 1152, 55: 499, 59: 12, 63: 59, 71: 233, 75: 1430, 83: 3, 87: 184, 91: 152, 95: 50, 99: 194, 103: 181, 107: 6, 111: 417, 119: 66, 123: 227, 127: 37}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 29, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 8981, 'token_per_expert': {0: 143, 4: 140, 8: 18, 12: 900, 16: 1, 20: 114, 24: 370, 28: 28, 32: 1137, 36: 373, 40: 217, 44: 1042, 48: 90, 52: 292, 56: 349, 60: 7, 64: 368, 68: 127, 72: 21, 76: 292, 80: 269, 84: 3, 88: 68, 96: 7, 100: 25, 104: 73, 108: 150, 112: 8, 116: 214, 120: 680, 124: 1455}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 25, 29, 33, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 93, 101, 105, 109, 113, 121, 125], 'expert_count': 27, 'ideal_gpu_count': 29, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 8136, 'token_per_expert': {1: 60, 5: 553, 9: 303, 17: 376, 21: 40, 25: 32, 29: 440, 33: 36, 41: 167, 45: 198, 49: 116, 53: 392, 57: 252, 61: 923, 65: 51, 69: 211, 73: 2458, 77: 148, 81: 156, 85: 29, 93: 91, 101: 294, 105: 553, 109: 3, 113: 191, 121: 40, 125: 23}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 26, 30, 34, 38, 42, 46, 50, 58, 62, 66, 70, 74, 78, 82, 86, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 28, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 7854, 'token_per_expert': {2: 270, 6: 122, 10: 27, 14: 155, 22: 5, 26: 6, 30: 4, 34: 73, 38: 169, 42: 112, 46: 440, 50: 659, 58: 1223, 62: 104, 66: 388, 70: 416, 74: 72, 78: 3, 82: 79, 86: 253, 98: 216, 102: 376, 106: 439, 110: 418, 114: 377, 118: 354, 122: 967, 126: 127}}
INFO 05-06 15:59:56.698223.698223 lmp.py:1845] [layer_moe_fused] layer=8 prefix: 4.943ms alloc: 0.255ms
INFO 05-06 15:59:56.698523.698523 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.269050598144531e-05 seconds
INFO 05-06 15:59:56.699301.699301 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014483928680419922s
INFO 05-06 15:59:56.700279.700279 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0008704662322998047s
DEBUG 05-06 15:59:56.700872.700872 cuda_h.py:27] end moe_wait_copy_tasks cost 0.959 ms
DEBUG 05-06 15:59:56.706509.706509 cuda_h.py:27] end moe_vllm_forward cost 5.399 ms
DEBUG 05-06 15:59:56.706021.706021 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:56.706487.706487 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.688ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.706378.706378 cuda_h.py:27] end *layer_moe_fused cost 13.777 ms
DEBUG 05-06 15:59:56.707397.707397 cuda_h.py:27] end prefill_merge_scale cost 0.306 ms
DEBUG 05-06 15:59:56.707850.707850 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:56.707627.707627 cuda_h.py:27] end prefill_layer cost 59.061 ms
DEBUG 05-06 15:59:56.707149.707149 lmp.py:1394] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 15:59:56.707667.707667 lmp.py:1350] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 15:59:56.707547.707547 cuda_h.py:27] end prefill_ln cost 0.164 ms
DEBUG 05-06 15:59:56.754927.754927 cuda_h.py:27] end prefill_attn cost 46.491 ms
DEBUG 05-06 15:59:56.754939.754939 cuda_h.py:27] end prefill_ffn_prep cost 0.296 ms
DEBUG 05-06 15:59:56.755145.755145 cuda_h.py:27] end prefill_gate cost 0.319 ms
INFO 05-06 15:59:56.759353.759353 lmp.py:1823] [layer_moe_fused] layer=9 active_experts=112 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 63, 67, 71, 75, 79, 83, 87, 95, 103, 111, 115, 119, 123, 127], 'expert_count': 28, 'ideal_gpu_count': 28, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 6592, 'token_per_expert': {3: 1812, 7: 18, 11: 82, 15: 55, 19: 774, 23: 376, 27: 143, 31: 5, 35: 1, 39: 103, 43: 52, 47: 1, 51: 6, 55: 19, 63: 1, 67: 97, 71: 714, 75: 485, 79: 5, 83: 328, 87: 2, 95: 662, 103: 194, 111: 188, 115: 108, 119: 2, 123: 26, 127: 333}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 68, 72, 76, 80, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 28, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 6846, 'token_per_expert': {0: 140, 4: 820, 8: 11, 12: 236, 16: 621, 20: 13, 24: 273, 28: 3, 32: 605, 36: 91, 40: 12, 44: 37, 48: 567, 52: 330, 56: 77, 60: 3, 68: 65, 72: 515, 76: 397, 80: 12, 88: 52, 92: 639, 96: 3, 100: 239, 104: 34, 108: 91, 112: 102, 116: 15, 120: 25, 124: 818}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 57, 61, 69, 73, 77, 81, 85, 89, 93, 101, 105, 109, 113, 117, 125], 'expert_count': 27, 'ideal_gpu_count': 28, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 12064, 'token_per_expert': {1: 319, 5: 226, 9: 165, 13: 679, 17: 9, 21: 141, 25: 78, 29: 1049, 33: 57, 37: 2239, 41: 317, 45: 573, 57: 784, 61: 913, 69: 309, 73: 302, 77: 5, 81: 364, 85: 2, 89: 18, 93: 50, 101: 2410, 105: 24, 109: 7, 113: 42, 117: 261, 125: 721}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 54, 58, 62, 66, 70, 74, 82, 86, 90, 102, 106, 110, 114, 122, 126], 'expert_count': 27, 'ideal_gpu_count': 28, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 7266, 'token_per_expert': {2: 6, 6: 2, 10: 135, 14: 95, 18: 217, 22: 2, 26: 585, 30: 365, 34: 50, 38: 1320, 42: 69, 46: 760, 54: 347, 58: 2, 62: 373, 66: 979, 70: 233, 74: 207, 82: 658, 86: 29, 90: 37, 102: 15, 106: 588, 110: 4, 114: 3, 122: 180, 126: 5}}
INFO 05-06 15:59:56.760760.760760 lmp.py:1845] [layer_moe_fused] layer=9 prefix: 4.074ms alloc: 0.244ms
INFO 05-06 15:59:56.760277.760277 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.817413330078125e-05 seconds
INFO 05-06 15:59:56.761305.761305 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012125968933105469s
INFO 05-06 15:59:56.762175.762175 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010082721710205078s
DEBUG 05-06 15:59:56.762999.762999 cuda_h.py:27] end moe_wait_copy_tasks cost 1.092 ms
DEBUG 05-06 15:59:56.768633.768633 cuda_h.py:27] end moe_vllm_forward cost 5.467 ms
DEBUG 05-06 15:59:56.768714.768714 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:56.768703.768703 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.753ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.768832.768832 cuda_h.py:27] end *layer_moe_fused cost 12.800 ms
DEBUG 05-06 15:59:56.769313.769313 cuda_h.py:27] end prefill_merge_scale cost 0.298 ms
DEBUG 05-06 15:59:56.769098.769098 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:56.769874.769874 cuda_h.py:27] end prefill_layer cost 61.774 ms
DEBUG 05-06 15:59:56.769583.769583 lmp.py:1394] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 15:59:56.769617.769617 lmp.py:1350] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 15:59:56.769014.769014 cuda_h.py:27] end prefill_ln cost 0.159 ms
DEBUG 05-06 15:59:56.814366.814366 cuda_h.py:27] end prefill_attn cost 44.861 ms
DEBUG 05-06 15:59:56.815179.815179 cuda_h.py:27] end prefill_ffn_prep cost 0.290 ms
DEBUG 05-06 15:59:56.816463.816463 cuda_h.py:27] end prefill_gate cost 0.301 ms
INFO 05-06 15:59:56.821242.821242 lmp.py:1823] [layer_moe_fused] layer=10 active_experts=105 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [15, 19, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 115, 119, 127], 'expert_count': 25, 'ideal_gpu_count': 27, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 4303, 'token_per_expert': {15: 37, 19: 39, 27: 7, 31: 1228, 39: 85, 43: 8, 47: 1, 51: 281, 55: 1, 59: 3, 63: 2, 67: 134, 71: 20, 75: 429, 79: 480, 83: 120, 87: 2, 91: 10, 95: 1, 99: 633, 103: 203, 107: 4, 115: 4, 119: 132, 127: 439}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 28, 'ideal_gpu_count': 26, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 7618, 'token_per_expert': {0: 352, 4: 48, 8: 70, 12: 1, 16: 48, 20: 4, 24: 3, 28: 1, 32: 40, 40: 15, 44: 7, 52: 41, 56: 1979, 60: 1061, 64: 11, 68: 1, 72: 348, 76: 1, 80: 149, 84: 175, 88: 741, 92: 797, 100: 1113, 104: 5, 108: 134, 112: 44, 120: 5, 124: 424}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 37, 41, 45, 49, 57, 61, 65, 69, 77, 81, 85, 89, 93, 105, 109, 113, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 26, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 12089, 'token_per_expert': {1: 292, 5: 4, 9: 66, 13: 6, 17: 5, 21: 488, 29: 2, 37: 3507, 41: 1761, 45: 422, 49: 3, 57: 31, 61: 68, 65: 35, 69: 4, 77: 3, 81: 2478, 85: 138, 89: 53, 93: 151, 105: 16, 109: 5, 113: 2364, 121: 132, 125: 55}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 18, 22, 30, 34, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 122, 126], 'expert_count': 27, 'ideal_gpu_count': 26, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 8758, 'token_per_expert': {2: 4, 10: 2165, 14: 944, 18: 1, 22: 876, 30: 10, 34: 10, 42: 189, 46: 491, 50: 8, 54: 29, 58: 386, 62: 5, 66: 1744, 70: 232, 74: 206, 78: 174, 82: 23, 86: 55, 90: 167, 98: 76, 102: 2, 106: 168, 110: 130, 114: 3, 122: 1, 126: 659}}
INFO 05-06 15:59:56.821444.821444 lmp.py:1845] [layer_moe_fused] layer=10 prefix: 4.917ms alloc: 0.232ms
INFO 05-06 15:59:56.821183.821183 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.626678466796875e-05 seconds
INFO 05-06 15:59:56.823973.823973 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014221668243408203s
INFO 05-06 15:59:56.824863.824863 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009796619415283203s
DEBUG 05-06 15:59:56.824594.824594 cuda_h.py:27] end moe_wait_copy_tasks cost 1.065 ms
DEBUG 05-06 15:59:56.829218.829218 cuda_h.py:27] end moe_vllm_forward cost 5.391 ms
DEBUG 05-06 15:59:56.829253.829253 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:56.830765.830765 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.676ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.830040.830040 cuda_h.py:27] end *layer_moe_fused cost 13.841 ms
DEBUG 05-06 15:59:56.830165.830165 cuda_h.py:27] end prefill_merge_scale cost 0.300 ms
DEBUG 05-06 15:59:56.830427.830427 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:56.830203.830203 cuda_h.py:27] end prefill_layer cost 61.211 ms
DEBUG 05-06 15:59:56.830405.830405 lmp.py:1394] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 15:59:56.830201.830201 lmp.py:1350] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 15:59:56.831022.831022 cuda_h.py:27] end prefill_ln cost 0.157 ms
DEBUG 05-06 15:59:56.877864.877864 cuda_h.py:27] end prefill_attn cost 46.065 ms
DEBUG 05-06 15:59:56.877677.877677 cuda_h.py:27] end prefill_ffn_prep cost 0.289 ms
DEBUG 05-06 15:59:56.878058.878058 cuda_h.py:27] end prefill_gate cost 0.287 ms
INFO 05-06 15:59:56.887603.887603 lmp.py:1823] [layer_moe_fused] layer=11 active_experts=93 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 59, 67, 71, 79, 83, 87, 91, 95, 99, 103, 111, 115, 119, 123], 'expert_count': 26, 'ideal_gpu_count': 24, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 6491, 'token_per_expert': {7: 156, 11: 93, 15: 2, 19: 127, 23: 1211, 27: 1093, 31: 314, 35: 1, 39: 2, 43: 249, 47: 4, 51: 22, 59: 372, 67: 54, 71: 161, 79: 634, 83: 827, 87: 493, 91: 4, 95: 1, 99: 36, 103: 4, 111: 264, 115: 364, 119: 2, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 52, 56, 68, 80, 84, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 22, 'ideal_gpu_count': 23, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 7748, 'token_per_expert': {0: 1763, 4: 72, 8: 64, 16: 755, 20: 577, 24: 279, 28: 3, 32: 12, 36: 1767, 52: 3, 56: 94, 68: 484, 80: 2, 84: 184, 92: 746, 100: 262, 104: 54, 108: 373, 112: 18, 116: 15, 120: 11, 124: 210}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 25, 29, 33, 45, 49, 53, 57, 61, 69, 73, 77, 81, 85, 89, 93, 105, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 23, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 9357, 'token_per_expert': {1: 2, 5: 8, 9: 2, 17: 2930, 21: 18, 25: 40, 29: 39, 33: 1, 45: 1, 49: 284, 53: 3, 57: 45, 61: 339, 69: 227, 73: 21, 77: 146, 81: 884, 85: 5, 89: 217, 93: 345, 105: 9, 113: 87, 117: 3638, 121: 54, 125: 12}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 58, 66, 70, 82, 98, 102, 110, 126], 'expert_count': 20, 'ideal_gpu_count': 23, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 9172, 'token_per_expert': {2: 138, 6: 637, 14: 2, 18: 397, 22: 7, 26: 3, 30: 258, 34: 247, 38: 267, 42: 1886, 46: 651, 50: 388, 58: 8, 66: 573, 70: 293, 82: 40, 98: 997, 102: 1914, 110: 395, 126: 71}}
INFO 05-06 15:59:56.888217.888217 lmp.py:1845] [layer_moe_fused] layer=11 prefix: 8.575ms alloc: 0.691ms
INFO 05-06 15:59:56.888561.888561 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 6.341934204101562e-05 seconds
INFO 05-06 15:59:56.889001.889001 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014290809631347656s
INFO 05-06 15:59:56.891597.891597 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011129379272460938s
DEBUG 05-06 15:59:56.891858.891858 cuda_h.py:27] end moe_wait_copy_tasks cost 1.203 ms
DEBUG 05-06 15:59:56.896748.896748 cuda_h.py:27] end moe_vllm_forward cost 5.393 ms
DEBUG 05-06 15:59:56.896214.896214 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:56.896441.896441 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.698ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.897472.897472 cuda_h.py:27] end *layer_moe_fused cost 18.286 ms
DEBUG 05-06 15:59:56.897136.897136 cuda_h.py:27] end prefill_merge_scale cost 0.300 ms
DEBUG 05-06 15:59:56.897066.897066 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:56.897273.897273 cuda_h.py:27] end prefill_layer cost 66.836 ms
DEBUG 05-06 15:59:56.897330.897330 lmp.py:1394] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 15:59:56.897841.897841 lmp.py:1350] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 15:59:56.898761.898761 cuda_h.py:27] end prefill_ln cost 0.159 ms
DEBUG 05-06 15:59:56.943561.943561 cuda_h.py:27] end prefill_attn cost 44.975 ms
DEBUG 05-06 15:59:56.943327.943327 cuda_h.py:27] end prefill_ffn_prep cost 0.290 ms
DEBUG 05-06 15:59:56.944874.944874 cuda_h.py:27] end prefill_gate cost 0.288 ms
INFO 05-06 15:59:56.949994.949994 lmp.py:1823] [layer_moe_fused] layer=12 active_experts=73 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 35, 39, 47, 51, 59, 63, 71, 79, 83, 91, 95], 'expert_count': 13, 'ideal_gpu_count': 19, 'keep_on_gpu': 13, 'hit_count_on_device': 13, 'token_total': 5326, 'token_per_expert': {3: 2449, 15: 454, 35: 6, 39: 60, 47: 219, 51: 4, 59: 70, 63: 4, 71: 1320, 79: 101, 83: 153, 91: 184, 95: 302}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 12, 20, 36, 40, 48, 64, 68, 76, 80, 84, 92, 100, 104, 108, 116, 120, 124], 'expert_count': 19, 'ideal_gpu_count': 18, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 7339, 'token_per_expert': {0: 1, 8: 14, 12: 54, 20: 42, 36: 491, 40: 144, 48: 454, 64: 41, 68: 4, 76: 366, 80: 438, 84: 609, 92: 2222, 100: 406, 104: 8, 108: 321, 116: 185, 120: 352, 124: 1187}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 29, 33, 37, 41, 45, 49, 53, 65, 73, 77, 85, 93, 97, 101, 117, 125], 'expert_count': 20, 'ideal_gpu_count': 18, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 15437, 'token_per_expert': {1: 53, 5: 1121, 13: 25, 25: 91, 29: 21, 33: 24, 37: 2, 41: 37, 45: 3006, 49: 3194, 53: 3028, 65: 54, 73: 99, 77: 1319, 85: 1524, 93: 6, 97: 1, 101: 1, 117: 1822, 125: 9}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 14, 18, 22, 34, 38, 46, 50, 58, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 122, 126], 'expert_count': 21, 'ideal_gpu_count': 18, 'keep_on_gpu': 21, 'hit_count_on_device': 21, 'token_total': 4666, 'token_per_expert': {6: 27, 14: 16, 18: 18, 22: 159, 34: 3, 38: 10, 46: 43, 50: 579, 58: 150, 78: 5, 82: 2371, 86: 4, 90: 22, 94: 433, 98: 143, 102: 311, 106: 262, 110: 44, 114: 58, 122: 7, 126: 1}}
INFO 05-06 15:59:56.949612.949612 lmp.py:1845] [layer_moe_fused] layer=12 prefix: 5.030ms alloc: 0.187ms
INFO 05-06 15:59:56.950702.950702 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.076957702636719e-05 seconds
INFO 05-06 15:59:56.952946.952946 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.002109050750732422s
INFO 05-06 15:59:56.953480.953480 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0008580684661865234s
DEBUG 05-06 15:59:56.953450.953450 cuda_h.py:27] end moe_wait_copy_tasks cost 0.944 ms
DEBUG 05-06 15:59:56.958001.958001 cuda_h.py:27] end moe_vllm_forward cost 5.391 ms
DEBUG 05-06 15:59:56.958513.958513 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:56.959502.959502 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.690ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:56.959877.959877 cuda_h.py:27] end *layer_moe_fused cost 14.508 ms
DEBUG 05-06 15:59:56.959237.959237 cuda_h.py:27] end prefill_merge_scale cost 0.312 ms
DEBUG 05-06 15:59:56.959764.959764 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.038 ms
DEBUG 05-06 15:59:56.959401.959401 cuda_h.py:27] end prefill_layer cost 62.035 ms
DEBUG 05-06 15:59:56.960309.960309 lmp.py:1394] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 15:59:56.960628.960628 lmp.py:1350] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 15:59:56.960217.960217 cuda_h.py:27] end prefill_ln cost 0.161 ms
DEBUG 05-06 15:59:56.994149.994149 cuda_h.py:27] end prefill_attn cost 34.044 ms
DEBUG 05-06 15:59:56.995667.995667 cuda_h.py:27] end prefill_ffn_prep cost 0.345 ms
DEBUG 05-06 15:59:56.995507.995507 cuda_h.py:27] end prefill_gate cost 0.332 ms
INFO 05-06 15:59:57.000865.000865 lmp.py:1823] [layer_moe_fused] layer=13 active_experts=79 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 23, 27, 31, 35, 43, 47, 59, 63, 75, 87, 91, 95, 99, 107, 119, 123], 'expert_count': 20, 'ideal_gpu_count': 20, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 5734, 'token_per_expert': {3: 310, 11: 10, 15: 62, 19: 117, 23: 1, 27: 12, 31: 1881, 35: 2, 43: 288, 47: 37, 59: 317, 63: 7, 75: 972, 87: 1, 91: 1416, 95: 190, 99: 7, 107: 13, 119: 88, 123: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 32, 40, 52, 56, 64, 68, 80, 84, 96, 104, 108, 112, 116, 120, 124], 'expert_count': 17, 'ideal_gpu_count': 20, 'keep_on_gpu': 17, 'hit_count_on_device': 17, 'token_total': 4300, 'token_per_expert': {0: 1, 16: 8, 32: 402, 40: 22, 52: 74, 56: 1, 64: 299, 68: 2, 80: 333, 84: 503, 96: 210, 104: 1521, 108: 5, 112: 62, 116: 843, 120: 11, 124: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 17, 21, 25, 33, 37, 41, 45, 53, 61, 65, 69, 85, 93, 101, 113, 121, 125], 'expert_count': 20, 'ideal_gpu_count': 20, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 10281, 'token_per_expert': {1: 786, 9: 258, 13: 3847, 17: 19, 21: 44, 25: 97, 33: 1, 37: 110, 41: 41, 45: 263, 53: 484, 61: 154, 65: 853, 69: 43, 85: 228, 93: 1629, 101: 18, 113: 235, 121: 471, 125: 700}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 22, 38, 42, 46, 54, 58, 62, 66, 70, 78, 82, 94, 98, 102, 110, 114, 118, 122, 126], 'expert_count': 22, 'ideal_gpu_count': 19, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 12453, 'token_per_expert': {2: 1058, 6: 6, 14: 102, 22: 1, 38: 3369, 42: 427, 46: 81, 54: 78, 58: 971, 62: 72, 66: 58, 70: 1, 78: 3844, 82: 520, 94: 30, 98: 1, 102: 1488, 110: 80, 114: 242, 118: 1, 122: 22, 126: 1}}
INFO 05-06 15:59:57.000179.000179 lmp.py:1845] [layer_moe_fused] layer=13 prefix: 4.005ms alloc: 0.209ms
INFO 05-06 15:59:57.000087.000087 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.649162292480469e-05 seconds
INFO 05-06 15:59:57.002652.002652 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001432657241821289s
INFO 05-06 15:59:57.003274.003274 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010972023010253906s
DEBUG 05-06 15:59:57.003861.003861 cuda_h.py:27] end moe_wait_copy_tasks cost 1.217 ms
DEBUG 05-06 15:59:57.008586.008586 cuda_h.py:27] end moe_vllm_forward cost 4.859 ms
DEBUG 05-06 15:59:57.008052.008052 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.008233.008233 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.175ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.008806.008806 cuda_h.py:27] end *layer_moe_fused cost 12.653 ms
DEBUG 05-06 15:59:57.009817.009817 cuda_h.py:27] end prefill_merge_scale cost 0.300 ms
DEBUG 05-06 15:59:57.009675.009675 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.037 ms
DEBUG 05-06 15:59:57.009696.009696 cuda_h.py:27] end prefill_layer cost 49.312 ms
DEBUG 05-06 15:59:57.009224.009224 lmp.py:1394] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 15:59:57.009020.009020 lmp.py:1350] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 15:59:57.009039.009039 cuda_h.py:27] end prefill_ln cost 0.160 ms
DEBUG 05-06 15:59:57.035073.035073 cuda_h.py:27] end prefill_attn cost 25.675 ms
DEBUG 05-06 15:59:57.035298.035298 cuda_h.py:27] end prefill_ffn_prep cost 0.297 ms
DEBUG 05-06 15:59:57.036041.036041 cuda_h.py:27] end prefill_gate cost 0.295 ms
INFO 05-06 15:59:57.041430.041430 lmp.py:1823] [layer_moe_fused] layer=14 active_experts=64 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 31, 35, 39, 47, 51, 59, 71, 83, 91, 95, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 18, 'ideal_gpu_count': 16, 'keep_on_gpu': 18, 'hit_count_on_device': 18, 'token_total': 6367, 'token_per_expert': {7: 115, 31: 233, 35: 4, 39: 22, 47: 61, 51: 4, 59: 2223, 71: 207, 83: 220, 91: 193, 95: 1059, 99: 32, 107: 245, 111: 1645, 115: 34, 119: 53, 123: 8, 127: 9}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 44, 48, 52, 60, 68, 76, 80, 108, 112], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 16, 'token_total': 12664, 'token_per_expert': {0: 356, 4: 1, 8: 4049, 12: 48, 16: 573, 20: 4, 24: 59, 44: 16, 48: 406, 52: 3089, 60: 958, 68: 332, 76: 2, 80: 90, 108: 18, 112: 2663}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 29, 33, 45, 65, 73, 81, 89, 97, 105, 117, 121, 125], 'expert_count': 13, 'ideal_gpu_count': 16, 'keep_on_gpu': 13, 'hit_count_on_device': 13, 'token_total': 5889, 'token_per_expert': {9: 1, 29: 6, 33: 27, 45: 28, 65: 251, 73: 11, 81: 3, 89: 282, 97: 3321, 105: 11, 117: 255, 121: 1395, 125: 298}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 18, 22, 26, 30, 38, 42, 50, 54, 74, 78, 86, 98, 102, 110, 122], 'expert_count': 17, 'ideal_gpu_count': 16, 'keep_on_gpu': 17, 'hit_count_on_device': 17, 'token_total': 7848, 'token_per_expert': {6: 15, 10: 3145, 18: 852, 22: 2, 26: 63, 30: 2, 38: 1464, 42: 11, 50: 420, 54: 69, 74: 50, 78: 1, 86: 302, 98: 35, 102: 5, 110: 1400, 122: 12}}
INFO 05-06 15:59:57.042113.042113 lmp.py:1845] [layer_moe_fused] layer=14 prefix: 4.943ms alloc: 0.164ms
INFO 05-06 15:59:57.042911.042911 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.1484832763671875e-05 seconds
INFO 05-06 15:59:57.055768.055768 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.01330423355102539s
INFO 05-06 15:59:57.056172.056172 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011019706726074219s
DEBUG 05-06 15:59:57.057639.057639 cuda_h.py:27] end moe_wait_copy_tasks cost 1.203 ms
DEBUG 05-06 15:59:57.063376.063376 cuda_h.py:27] end moe_vllm_forward cost 5.746 ms
DEBUG 05-06 15:59:57.063147.063147 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 15:59:57.063540.063540 lmp.py:1964] [layer_moe_fused] vllm triton time: 6.101ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.063054.063054 cuda_h.py:27] end *layer_moe_fused cost 26.307 ms
DEBUG 05-06 15:59:57.063625.063625 cuda_h.py:27] end prefill_merge_scale cost 0.395 ms
DEBUG 05-06 15:59:57.064821.064821 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.042 ms
DEBUG 05-06 15:59:57.064729.064729 cuda_h.py:27] end prefill_layer cost 54.592 ms
DEBUG 05-06 15:59:57.064637.064637 lmp.py:1394] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 15:59:57.064347.064347 lmp.py:1350] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 15:59:57.064180.064180 cuda_h.py:27] end prefill_ln cost 0.204 ms
DEBUG 05-06 15:59:57.075328.075328 cuda_h.py:27] end prefill_attn cost 10.475 ms
DEBUG 05-06 15:59:57.075155.075155 cuda_h.py:27] end prefill_ffn_prep cost 0.298 ms
DEBUG 05-06 15:59:57.076278.076278 cuda_h.py:27] end prefill_gate cost 0.288 ms
INFO 05-06 15:59:57.080716.080716 lmp.py:1823] [layer_moe_fused] layer=15 active_experts=76 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 19, 23, 35, 39, 47, 55, 59, 67, 71, 75, 83, 91, 95, 99, 103, 107, 127], 'expert_count': 18, 'ideal_gpu_count': 19, 'keep_on_gpu': 18, 'hit_count_on_device': 18, 'token_total': 3669, 'token_per_expert': {3: 5, 19: 1, 23: 18, 35: 2, 39: 86, 47: 44, 55: 4, 59: 1, 67: 870, 71: 1, 75: 44, 83: 1, 91: 535, 95: 96, 99: 125, 103: 247, 107: 1555, 127: 34}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 20, 28, 32, 36, 40, 48, 52, 56, 60, 68, 72, 76, 80, 84, 88, 100, 112, 116, 120], 'expert_count': 23, 'ideal_gpu_count': 19, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 19319, 'token_per_expert': {0: 4, 4: 217, 12: 2403, 16: 213, 20: 14, 28: 638, 32: 2, 36: 3851, 40: 153, 48: 98, 52: 3, 56: 8, 60: 3, 68: 707, 72: 4, 76: 3785, 80: 1, 84: 1294, 88: 5, 100: 2185, 112: 2161, 116: 1569, 120: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 33, 37, 41, 45, 65, 69, 85, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 19, 'ideal_gpu_count': 19, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 7999, 'token_per_expert': {5: 5, 9: 1900, 13: 2, 33: 420, 37: 3, 41: 85, 45: 1, 65: 895, 69: 51, 85: 154, 93: 3, 97: 574, 101: 101, 105: 401, 109: 876, 113: 3, 117: 488, 121: 109, 125: 1928}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 30, 34, 38, 42, 46, 70, 78, 82, 98, 110, 114, 118], 'expert_count': 16, 'ideal_gpu_count': 19, 'keep_on_gpu': 16, 'hit_count_on_device': 16, 'token_total': 1781, 'token_per_expert': {2: 75, 6: 159, 18: 26, 22: 12, 30: 18, 34: 690, 38: 109, 42: 31, 46: 145, 70: 239, 78: 131, 82: 22, 98: 6, 110: 3, 114: 4, 118: 111}}
INFO 05-06 15:59:57.080148.080148 lmp.py:1845] [layer_moe_fused] layer=15 prefix: 4.035ms alloc: 0.190ms
INFO 05-06 15:59:57.081596.081596 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.649162292480469e-05 seconds
INFO 05-06 15:59:57.082103.082103 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014591217041015625s
INFO 05-06 15:59:57.083352.083352 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010333061218261719s
DEBUG 05-06 15:59:57.083851.083851 cuda_h.py:27] end moe_wait_copy_tasks cost 1.124 ms
DEBUG 05-06 15:59:57.088556.088556 cuda_h.py:27] end moe_vllm_forward cost 4.639 ms
DEBUG 05-06 15:59:57.088175.088175 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.088945.088945 lmp.py:1964] [layer_moe_fused] vllm triton time: 4.952ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.089134.089134 cuda_h.py:27] end *layer_moe_fused cost 12.328 ms
DEBUG 05-06 15:59:57.089809.089809 cuda_h.py:27] end prefill_merge_scale cost 0.302 ms
DEBUG 05-06 15:59:57.089262.089262 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.089469.089469 cuda_h.py:27] end prefill_layer cost 25.406 ms
DEBUG 05-06 15:59:57.089673.089673 lmp.py:1394] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 15:59:57.089945.089945 lmp.py:1350] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 15:59:57.090395.090395 cuda_h.py:27] end prefill_ln cost 0.162 ms
DEBUG 05-06 15:59:57.101793.101793 cuda_h.py:27] end prefill_attn cost 11.408 ms
DEBUG 05-06 15:59:57.102695.102695 cuda_h.py:27] end prefill_ffn_prep cost 0.305 ms
DEBUG 05-06 15:59:57.102480.102480 cuda_h.py:27] end prefill_gate cost 0.304 ms
INFO 05-06 15:59:57.107998.107998 lmp.py:1823] [layer_moe_fused] layer=16 active_experts=84 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 23, 31, 35, 39, 47, 51, 55, 63, 67, 71, 79, 87, 99, 103, 115, 119, 123], 'expert_count': 19, 'ideal_gpu_count': 21, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 7569, 'token_per_expert': {3: 1, 15: 35, 23: 56, 31: 1153, 35: 785, 39: 611, 47: 93, 51: 34, 55: 743, 63: 257, 67: 35, 71: 134, 79: 96, 87: 137, 99: 328, 103: 2600, 115: 127, 119: 227, 123: 117}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 28, 32, 36, 44, 48, 52, 56, 60, 68, 72, 80, 84, 92, 96, 100, 104, 108, 112, 120], 'expert_count': 24, 'ideal_gpu_count': 21, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 9642, 'token_per_expert': {0: 3, 4: 625, 12: 420, 20: 1476, 24: 400, 28: 1, 32: 94, 36: 68, 44: 92, 48: 18, 52: 1, 56: 3622, 60: 1, 68: 397, 72: 175, 80: 13, 84: 328, 92: 115, 96: 3, 100: 409, 104: 49, 108: 126, 112: 1, 120: 1205}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 45, 49, 61, 73, 81, 105, 109, 117, 121], 'expert_count': 17, 'ideal_gpu_count': 21, 'keep_on_gpu': 17, 'hit_count_on_device': 17, 'token_total': 12418, 'token_per_expert': {1: 8, 5: 54, 9: 1660, 13: 107, 17: 2330, 21: 462, 25: 753, 29: 2217, 45: 3, 49: 4, 61: 1, 73: 1, 81: 3824, 105: 5, 109: 906, 117: 80, 121: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 18, 26, 30, 34, 38, 42, 46, 50, 54, 58, 66, 74, 78, 82, 86, 90, 94, 102, 106, 110, 114, 126], 'expert_count': 24, 'ideal_gpu_count': 21, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 3139, 'token_per_expert': {6: 24, 10: 25, 18: 999, 26: 178, 30: 3, 34: 103, 38: 242, 42: 76, 46: 288, 50: 51, 54: 3, 58: 214, 66: 8, 74: 52, 78: 15, 82: 60, 86: 207, 90: 94, 94: 3, 102: 187, 106: 16, 110: 148, 114: 111, 126: 32}}
INFO 05-06 15:59:57.108887.108887 lmp.py:1845] [layer_moe_fused] layer=16 prefix: 4.834ms alloc: 0.207ms
INFO 05-06 15:59:57.108229.108229 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.458427429199219e-05 seconds
INFO 05-06 15:59:57.110956.110956 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014815330505371094s
INFO 05-06 15:59:57.111514.111514 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011930465698242188s
DEBUG 05-06 15:59:57.111293.111293 cuda_h.py:27] end moe_wait_copy_tasks cost 1.313 ms
DEBUG 05-06 15:59:57.116401.116401 cuda_h.py:27] end moe_vllm_forward cost 4.732 ms
DEBUG 05-06 15:59:57.116628.116628 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.116571.116571 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.031ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.116912.116912 cuda_h.py:27] end *layer_moe_fused cost 13.703 ms
DEBUG 05-06 15:59:57.117287.117287 cuda_h.py:27] end prefill_merge_scale cost 0.301 ms
DEBUG 05-06 15:59:57.117787.117787 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.033 ms
DEBUG 05-06 15:59:57.117617.117617 cuda_h.py:27] end prefill_layer cost 27.565 ms
DEBUG 05-06 15:59:57.117611.117611 lmp.py:1394] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 15:59:57.117407.117407 lmp.py:1350] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 15:59:57.117711.117711 cuda_h.py:27] end prefill_ln cost 0.161 ms
DEBUG 05-06 15:59:57.129962.129962 cuda_h.py:27] end prefill_attn cost 11.776 ms
DEBUG 05-06 15:59:57.130557.130557 cuda_h.py:27] end prefill_ffn_prep cost 0.289 ms
DEBUG 05-06 15:59:57.131708.131708 cuda_h.py:27] end prefill_gate cost 0.288 ms
INFO 05-06 15:59:57.138196.138196 lmp.py:1823] [layer_moe_fused] layer=17 active_experts=96 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 35, 39, 43, 47, 51, 55, 67, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 4720, 'token_per_expert': {3: 705, 7: 8, 11: 2, 19: 56, 23: 154, 35: 17, 39: 345, 43: 385, 47: 150, 51: 705, 55: 455, 67: 1, 71: 342, 75: 4, 83: 332, 87: 11, 91: 81, 95: 21, 99: 68, 103: 484, 107: 6, 111: 356, 115: 8, 119: 1, 123: 23}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 56, 60, 64, 68, 76, 80, 84, 88, 100, 104, 108, 112, 116, 124], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 6390, 'token_per_expert': {0: 232, 4: 932, 8: 6, 12: 222, 16: 9, 20: 1007, 24: 29, 28: 77, 32: 231, 36: 1, 40: 375, 56: 8, 60: 228, 64: 2187, 68: 131, 76: 3, 80: 2, 84: 4, 88: 50, 100: 7, 104: 56, 108: 123, 112: 269, 116: 155, 124: 46}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 21, 25, 29, 33, 37, 41, 45, 53, 57, 65, 73, 77, 81, 85, 97, 109, 113, 125], 'expert_count': 21, 'ideal_gpu_count': 24, 'keep_on_gpu': 21, 'hit_count_on_device': 21, 'token_total': 15928, 'token_per_expert': {5: 15, 9: 2685, 13: 1, 21: 56, 25: 19, 29: 842, 33: 3, 37: 30, 41: 10, 45: 983, 53: 3350, 57: 213, 65: 33, 73: 12, 77: 688, 81: 2274, 85: 27, 97: 268, 109: 73, 113: 335, 125: 4011}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 22, 26, 30, 34, 38, 42, 46, 54, 58, 62, 66, 70, 74, 78, 86, 90, 98, 106, 110, 114, 118, 122], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 5730, 'token_per_expert': {2: 2, 10: 456, 14: 8, 22: 21, 26: 1181, 30: 4, 34: 1, 38: 222, 42: 71, 46: 1, 54: 600, 58: 17, 62: 2, 66: 11, 70: 1, 74: 4, 78: 90, 86: 135, 90: 59, 98: 250, 106: 17, 110: 2490, 114: 4, 118: 38, 122: 45}}
INFO 05-06 15:59:57.139013.139013 lmp.py:1845] [layer_moe_fused] layer=17 prefix: 7.475ms alloc: 0.231ms
INFO 05-06 15:59:57.139447.139447 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.935264587402344e-05 seconds
INFO 05-06 15:59:57.140307.140307 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0015444755554199219s
INFO 05-06 15:59:57.142673.142673 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011556148529052734s
DEBUG 05-06 15:59:57.142311.142311 cuda_h.py:27] end moe_wait_copy_tasks cost 1.243 ms
DEBUG 05-06 15:59:57.147324.147324 cuda_h.py:27] end moe_vllm_forward cost 4.726 ms
DEBUG 05-06 15:59:57.147505.147505 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.147209.147209 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.018ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.147027.147027 cuda_h.py:27] end *layer_moe_fused cost 16.040 ms
DEBUG 05-06 15:59:57.148936.148936 cuda_h.py:27] end prefill_merge_scale cost 0.302 ms
DEBUG 05-06 15:59:57.148866.148866 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.148119.148119 cuda_h.py:27] end prefill_layer cost 30.370 ms
DEBUG 05-06 15:59:57.148450.148450 lmp.py:1394] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 15:59:57.148768.148768 lmp.py:1350] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 15:59:57.148688.148688 cuda_h.py:27] end prefill_ln cost 0.158 ms
DEBUG 05-06 15:59:57.159202.159202 cuda_h.py:27] end prefill_attn cost 11.143 ms
DEBUG 05-06 15:59:57.160208.160208 cuda_h.py:27] end prefill_ffn_prep cost 0.291 ms
DEBUG 05-06 15:59:57.160990.160990 cuda_h.py:27] end prefill_gate cost 0.288 ms
INFO 05-06 15:59:57.166801.166801 lmp.py:1823] [layer_moe_fused] layer=18 active_experts=103 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 27, 31, 35, 39, 47, 51, 55, 59, 63, 67, 75, 79, 83, 87, 91, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 26, 'ideal_gpu_count': 26, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 9869, 'token_per_expert': {3: 1117, 11: 24, 15: 20, 19: 763, 27: 319, 31: 424, 35: 143, 39: 332, 47: 103, 51: 6, 55: 7, 59: 1190, 63: 452, 67: 67, 75: 721, 79: 33, 83: 138, 87: 136, 91: 217, 99: 2, 103: 196, 107: 1, 111: 87, 119: 871, 123: 2371, 127: 129}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 52, 56, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 112, 116, 120, 124], 'expert_count': 27, 'ideal_gpu_count': 26, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 13337, 'token_per_expert': {0: 2, 4: 463, 8: 366, 12: 428, 16: 1390, 20: 32, 24: 134, 28: 1, 32: 343, 40: 1030, 44: 1, 52: 1130, 56: 154, 64: 6, 68: 1035, 72: 508, 76: 9, 80: 229, 84: 2792, 88: 1189, 92: 1, 96: 70, 100: 1325, 112: 29, 116: 47, 120: 41, 124: 582}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 21, 25, 29, 33, 37, 41, 49, 53, 57, 61, 69, 73, 77, 81, 85, 93, 97, 105, 109, 113, 125], 'expert_count': 24, 'ideal_gpu_count': 26, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 6039, 'token_per_expert': {1: 75, 9: 1, 13: 1454, 21: 102, 25: 31, 29: 1, 33: 705, 37: 9, 41: 2, 49: 12, 53: 85, 57: 8, 61: 2, 69: 260, 73: 128, 77: 26, 81: 197, 85: 1568, 93: 290, 97: 283, 105: 425, 109: 367, 113: 2, 125: 6}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 78, 86, 90, 94, 98, 102, 106, 110, 114, 126], 'expert_count': 26, 'ideal_gpu_count': 25, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 3523, 'token_per_expert': {2: 218, 6: 1, 14: 103, 18: 37, 26: 105, 30: 287, 34: 3, 38: 335, 42: 459, 46: 3, 50: 317, 54: 1, 58: 1, 62: 478, 66: 47, 70: 18, 78: 11, 86: 2, 90: 94, 94: 242, 98: 19, 102: 1, 106: 46, 110: 402, 114: 242, 126: 51}}
INFO 05-06 15:59:57.166653.166653 lmp.py:1845] [layer_moe_fused] layer=18 prefix: 5.081ms alloc: 0.256ms
INFO 05-06 15:59:57.166849.166849 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.1021575927734375e-05 seconds
INFO 05-06 15:59:57.168681.168681 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014886856079101562s
INFO 05-06 15:59:57.169652.169652 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010395050048828125s
DEBUG 05-06 15:59:57.169860.169860 cuda_h.py:27] end moe_wait_copy_tasks cost 1.126 ms
DEBUG 05-06 15:59:57.174262.174262 cuda_h.py:27] end moe_vllm_forward cost 4.699 ms
DEBUG 05-06 15:59:57.174821.174821 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.174379.174379 lmp.py:1964] [layer_moe_fused] vllm triton time: 4.985ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.174197.174197 cuda_h.py:27] end *layer_moe_fused cost 13.501 ms
DEBUG 05-06 15:59:57.175614.175614 cuda_h.py:27] end prefill_merge_scale cost 0.304 ms
DEBUG 05-06 15:59:57.175783.175783 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.175321.175321 cuda_h.py:27] end prefill_layer cost 26.986 ms
DEBUG 05-06 15:59:57.175333.175333 lmp.py:1394] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 15:59:57.175366.175366 lmp.py:1350] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 15:59:57.175015.175015 cuda_h.py:27] end prefill_ln cost 0.169 ms
DEBUG 05-06 15:59:57.187157.187157 cuda_h.py:27] end prefill_attn cost 11.287 ms
DEBUG 05-06 15:59:57.187924.187924 cuda_h.py:27] end prefill_ffn_prep cost 0.291 ms
DEBUG 05-06 15:59:57.188775.188775 cuda_h.py:27] end prefill_gate cost 0.292 ms
INFO 05-06 15:59:57.192077.192077 lmp.py:1823] [layer_moe_fused] layer=19 active_experts=84 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 19, 23, 27, 35, 39, 43, 47, 51, 63, 67, 71, 75, 83, 87, 103, 111, 115, 119], 'expert_count': 20, 'ideal_gpu_count': 21, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 8564, 'token_per_expert': {7: 10, 15: 1, 19: 148, 23: 72, 27: 1712, 35: 118, 39: 112, 43: 2, 47: 22, 51: 287, 63: 411, 67: 1, 71: 20, 75: 2096, 83: 6, 87: 55, 103: 1, 111: 52, 115: 4, 119: 3434}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 16, 20, 24, 28, 32, 44, 48, 52, 56, 60, 64, 72, 80, 84, 88, 92, 100, 104, 112, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 21, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 11346, 'token_per_expert': {4: 14, 8: 20, 12: 805, 16: 1, 20: 156, 24: 1173, 28: 1239, 32: 210, 44: 3, 48: 3351, 52: 1736, 56: 150, 60: 3, 64: 109, 72: 1, 80: 1044, 84: 176, 88: 121, 92: 183, 100: 315, 104: 32, 112: 6, 120: 400, 124: 98}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 37, 41, 45, 53, 65, 69, 73, 81, 89, 97, 101, 109, 117, 121, 125], 'expert_count': 20, 'ideal_gpu_count': 21, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 6782, 'token_per_expert': {1: 361, 5: 91, 9: 162, 13: 20, 25: 111, 37: 814, 41: 416, 45: 12, 53: 7, 65: 1, 69: 2076, 73: 283, 81: 1617, 89: 235, 97: 1, 101: 10, 109: 37, 117: 1, 121: 39, 125: 488}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 14, 18, 30, 38, 46, 50, 58, 70, 78, 82, 90, 94, 98, 102, 106, 110, 118, 122, 126], 'expert_count': 20, 'ideal_gpu_count': 21, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 6076, 'token_per_expert': {6: 28, 14: 5, 18: 14, 30: 1, 38: 3118, 46: 178, 50: 167, 58: 56, 70: 4, 78: 3, 82: 2, 90: 374, 94: 1294, 98: 1, 102: 203, 106: 170, 110: 64, 118: 74, 122: 310, 126: 10}}
INFO 05-06 15:59:57.192344.192344 lmp.py:1845] [layer_moe_fused] layer=19 prefix: 4.111ms alloc: 0.200ms
INFO 05-06 15:59:57.193998.193998 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.38690185546875e-05 seconds
INFO 05-06 15:59:57.194411.194411 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014252662658691406s
INFO 05-06 15:59:57.195012.195012 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010833740234375s
DEBUG 05-06 15:59:57.195459.195459 cuda_h.py:27] end moe_wait_copy_tasks cost 1.170 ms
DEBUG 05-06 15:59:57.200257.200257 cuda_h.py:27] end moe_vllm_forward cost 4.639 ms
DEBUG 05-06 15:59:57.200007.200007 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.200426.200426 lmp.py:1964] [layer_moe_fused] vllm triton time: 4.928ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.200583.200583 cuda_h.py:27] end *layer_moe_fused cost 12.433 ms
DEBUG 05-06 15:59:57.201079.201079 cuda_h.py:27] end prefill_merge_scale cost 0.303 ms
DEBUG 05-06 15:59:57.201009.201009 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.201262.201262 cuda_h.py:27] end prefill_layer cost 26.151 ms
DEBUG 05-06 15:59:57.201891.201891 lmp.py:1394] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 15:59:57.201732.201732 lmp.py:1350] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 15:59:57.202606.202606 cuda_h.py:27] end prefill_ln cost 0.159 ms
DEBUG 05-06 15:59:57.213013.213013 cuda_h.py:27] end prefill_attn cost 11.309 ms
DEBUG 05-06 15:59:57.213303.213303 cuda_h.py:27] end prefill_ffn_prep cost 0.291 ms
DEBUG 05-06 15:59:57.214720.214720 cuda_h.py:27] end prefill_gate cost 0.286 ms
INFO 05-06 15:59:57.219864.219864 lmp.py:1823] [layer_moe_fused] layer=20 active_experts=89 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 79, 83, 91, 103, 111, 115, 123, 127], 'expert_count': 22, 'ideal_gpu_count': 23, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 11730, 'token_per_expert': {3: 243, 7: 20, 11: 203, 19: 180, 27: 168, 31: 4, 35: 11, 39: 48, 43: 2, 47: 1, 51: 273, 55: 61, 59: 162, 63: 1, 79: 162, 83: 1594, 91: 304, 103: 9, 111: 129, 115: 3372, 123: 2657, 127: 2126}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 16, 20, 28, 32, 36, 44, 52, 56, 60, 72, 80, 84, 88, 92, 96, 100, 104, 116, 120], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 3056, 'token_per_expert': {4: 5, 8: 402, 12: 28, 16: 16, 20: 173, 28: 47, 32: 97, 36: 1, 44: 2, 52: 186, 56: 34, 60: 1, 72: 1, 80: 86, 84: 364, 88: 4, 92: 346, 96: 184, 100: 407, 104: 20, 116: 31, 120: 621}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 97, 101, 105, 117, 121, 125], 'expert_count': 26, 'ideal_gpu_count': 22, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 15297, 'token_per_expert': {1: 6, 5: 3186, 9: 106, 13: 3, 17: 46, 21: 26, 29: 11, 33: 624, 45: 658, 49: 3294, 53: 14, 57: 906, 61: 53, 65: 762, 69: 1, 73: 1851, 77: 1, 81: 86, 85: 686, 89: 51, 97: 2283, 101: 180, 105: 73, 117: 1, 121: 35, 125: 354}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 30, 38, 50, 62, 66, 70, 78, 82, 90, 102, 106, 110, 114, 122], 'expert_count': 19, 'ideal_gpu_count': 22, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 2685, 'token_per_expert': {2: 401, 6: 21, 14: 61, 18: 116, 22: 24, 30: 4, 38: 113, 50: 157, 62: 89, 66: 419, 70: 244, 78: 9, 82: 37, 90: 31, 102: 187, 106: 475, 110: 241, 114: 20, 122: 36}}
INFO 05-06 15:59:57.220985.220985 lmp.py:1845] [layer_moe_fused] layer=20 prefix: 4.941ms alloc: 0.207ms
INFO 05-06 15:59:57.220557.220557 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.482269287109375e-05 seconds
INFO 05-06 15:59:57.221532.221532 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012085437774658203s
INFO 05-06 15:59:57.222635.222635 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010373592376708984s
DEBUG 05-06 15:59:57.222889.222889 cuda_h.py:27] end moe_wait_copy_tasks cost 1.123 ms
DEBUG 05-06 15:59:57.227973.227973 cuda_h.py:27] end moe_vllm_forward cost 4.659 ms
DEBUG 05-06 15:59:57.227723.227723 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.227143.227143 lmp.py:1964] [layer_moe_fused] vllm triton time: 4.965ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.227921.227921 cuda_h.py:27] end *layer_moe_fused cost 12.933 ms
DEBUG 05-06 15:59:57.228700.228700 cuda_h.py:27] end prefill_merge_scale cost 0.302 ms
DEBUG 05-06 15:59:57.228630.228630 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.228168.228168 cuda_h.py:27] end prefill_layer cost 26.655 ms
DEBUG 05-06 15:59:57.228617.228617 lmp.py:1394] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 15:59:57.228174.228174 lmp.py:1350] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 15:59:57.228048.228048 cuda_h.py:27] end prefill_ln cost 0.160 ms
DEBUG 05-06 15:59:57.241130.241130 cuda_h.py:27] end prefill_attn cost 12.076 ms
DEBUG 05-06 15:59:57.241866.241866 cuda_h.py:27] end prefill_ffn_prep cost 0.344 ms
DEBUG 05-06 15:59:57.242693.242693 cuda_h.py:27] end prefill_gate cost 0.333 ms
INFO 05-06 15:59:57.245144.245144 lmp.py:1823] [layer_moe_fused] layer=21 active_experts=102 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 15, 27, 31, 35, 39, 43, 47, 55, 59, 67, 71, 75, 79, 83, 91, 95, 99, 103, 111, 115, 119, 123], 'expert_count': 23, 'ideal_gpu_count': 26, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 7558, 'token_per_expert': {11: 1407, 15: 2541, 27: 16, 31: 15, 35: 207, 39: 172, 43: 11, 47: 9, 55: 114, 59: 92, 67: 499, 71: 91, 75: 456, 79: 37, 83: 331, 91: 11, 95: 749, 99: 191, 103: 179, 111: 1, 115: 367, 119: 55, 123: 7}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 52, 56, 68, 72, 76, 80, 88, 92, 96, 100, 104, 108, 112, 120, 124], 'expert_count': 26, 'ideal_gpu_count': 26, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 12443, 'token_per_expert': {0: 855, 4: 142, 8: 3542, 12: 921, 16: 3, 24: 215, 28: 89, 32: 283, 36: 144, 40: 27, 44: 9, 52: 6, 56: 72, 68: 229, 72: 125, 76: 3044, 80: 887, 88: 422, 92: 190, 96: 8, 100: 132, 104: 908, 108: 15, 112: 6, 120: 168, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 37, 41, 45, 49, 53, 57, 61, 65, 73, 77, 81, 85, 89, 93, 97, 105, 109, 117, 121], 'expert_count': 27, 'ideal_gpu_count': 25, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 6667, 'token_per_expert': {1: 215, 5: 479, 9: 160, 13: 303, 17: 163, 21: 16, 25: 20, 29: 791, 37: 8, 41: 191, 45: 14, 49: 358, 53: 297, 57: 1, 61: 104, 65: 1531, 73: 363, 77: 1, 81: 2, 85: 151, 89: 4, 93: 4, 97: 142, 105: 793, 109: 361, 117: 2, 121: 193}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 58, 62, 66, 78, 82, 90, 94, 98, 102, 106, 110, 118, 122, 126], 'expert_count': 26, 'ideal_gpu_count': 25, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 6100, 'token_per_expert': {2: 49, 6: 5, 10: 214, 14: 49, 18: 2259, 22: 125, 26: 43, 30: 223, 34: 211, 38: 42, 42: 2, 46: 1753, 58: 117, 62: 205, 66: 83, 78: 7, 82: 146, 90: 22, 94: 117, 98: 12, 102: 1, 106: 12, 110: 32, 118: 2, 122: 360, 126: 9}}
INFO 05-06 15:59:57.246876.246876 lmp.py:1845] [layer_moe_fused] layer=21 prefix: 3.229ms alloc: 0.239ms
INFO 05-06 15:59:57.246243.246243 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.982948303222656e-05 seconds
INFO 05-06 15:59:57.247333.247333 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014677047729492188s
INFO 05-06 15:59:57.249235.249235 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011641979217529297s
DEBUG 05-06 15:59:57.249755.249755 cuda_h.py:27] end moe_wait_copy_tasks cost 1.270 ms
DEBUG 05-06 15:59:57.254653.254653 cuda_h.py:27] end moe_vllm_forward cost 4.829 ms
DEBUG 05-06 15:59:57.254833.254833 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 15:59:57.254014.254014 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.139ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.254131.254131 cuda_h.py:27] end *layer_moe_fused cost 11.878 ms
DEBUG 05-06 15:59:57.255242.255242 cuda_h.py:27] end prefill_merge_scale cost 0.301 ms
DEBUG 05-06 15:59:57.255218.255218 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.255286.255286 cuda_h.py:27] end prefill_layer cost 26.486 ms
DEBUG 05-06 15:59:57.255914.255914 lmp.py:1394] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 15:59:57.255947.255947 lmp.py:1350] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 15:59:57.255490.255490 cuda_h.py:27] end prefill_ln cost 0.162 ms
DEBUG 05-06 15:59:57.266892.266892 cuda_h.py:27] end prefill_attn cost 11.341 ms
DEBUG 05-06 15:59:57.267612.267612 cuda_h.py:27] end prefill_ffn_prep cost 0.293 ms
DEBUG 05-06 15:59:57.268186.268186 cuda_h.py:27] end prefill_gate cost 0.297 ms
INFO 05-06 15:59:57.273262.273262 lmp.py:1823] [layer_moe_fused] layer=22 active_experts=96 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 15, 19, 23, 35, 39, 47, 51, 59, 63, 67, 71, 75, 83, 87, 95, 99, 103, 107, 119, 123, 127], 'expert_count': 23, 'ideal_gpu_count': 24, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 3761, 'token_per_expert': {7: 27, 11: 3, 15: 28, 19: 191, 23: 337, 35: 135, 39: 10, 47: 2, 51: 4, 59: 4, 63: 187, 67: 84, 71: 44, 75: 162, 83: 70, 87: 63, 95: 2, 99: 394, 103: 1138, 107: 541, 119: 26, 123: 306, 127: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 16, 20, 28, 32, 40, 44, 48, 52, 60, 64, 72, 76, 80, 84, 88, 92, 96, 100, 108, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 14062, 'token_per_expert': {0: 263, 8: 71, 16: 105, 20: 25, 28: 127, 32: 1903, 40: 442, 44: 62, 48: 158, 52: 3, 60: 43, 64: 1, 72: 627, 76: 169, 80: 3, 84: 684, 88: 3410, 92: 45, 96: 4, 100: 666, 108: 1964, 116: 232, 120: 314, 124: 2741}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 81, 85, 93, 97, 101, 105, 113, 117, 121, 125], 'expert_count': 26, 'ideal_gpu_count': 24, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 10932, 'token_per_expert': {1: 7, 5: 69, 9: 309, 13: 22, 17: 171, 29: 19, 33: 33, 37: 294, 41: 3868, 45: 81, 53: 204, 57: 1, 61: 58, 65: 1, 69: 291, 73: 1354, 81: 16, 85: 18, 93: 51, 97: 1, 101: 1478, 105: 1, 113: 12, 117: 2452, 121: 1, 125: 120}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 30, 34, 38, 42, 50, 54, 58, 66, 70, 74, 78, 82, 86, 90, 94, 110, 114, 118, 126], 'expert_count': 23, 'ideal_gpu_count': 24, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 4013, 'token_per_expert': {2: 166, 6: 11, 10: 77, 22: 2, 30: 4, 34: 600, 38: 29, 42: 201, 50: 785, 54: 193, 58: 2, 66: 494, 70: 23, 74: 6, 78: 41, 82: 131, 86: 5, 90: 104, 94: 86, 110: 55, 114: 1, 118: 203, 126: 794}}
INFO 05-06 15:59:57.273417.273417 lmp.py:1845] [layer_moe_fused] layer=22 prefix: 4.891ms alloc: 0.234ms
INFO 05-06 15:59:57.273157.273157 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.673004150390625e-05 seconds
INFO 05-06 15:59:57.275193.275193 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014276504516601562s
INFO 05-06 15:59:57.276619.276619 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009899139404296875s
DEBUG 05-06 15:59:57.276257.276257 cuda_h.py:27] end moe_wait_copy_tasks cost 1.077 ms
DEBUG 05-06 15:59:57.281345.281345 cuda_h.py:27] end moe_vllm_forward cost 4.607 ms
DEBUG 05-06 15:59:57.281573.281573 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.281277.281277 lmp.py:1964] [layer_moe_fused] vllm triton time: 4.897ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.281109.281109 cuda_h.py:27] end *layer_moe_fused cost 13.115 ms
DEBUG 05-06 15:59:57.282492.282492 cuda_h.py:27] end prefill_merge_scale cost 0.301 ms
DEBUG 05-06 15:59:57.282945.282945 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.282960.282960 cuda_h.py:27] end prefill_layer cost 26.869 ms
DEBUG 05-06 15:59:57.282582.282582 lmp.py:1394] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 15:59:57.282377.282377 lmp.py:1350] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 15:59:57.282820.282820 cuda_h.py:27] end prefill_ln cost 0.159 ms
DEBUG 05-06 15:59:57.294208.294208 cuda_h.py:27] end prefill_attn cost 11.716 ms
DEBUG 05-06 15:59:57.294167.294167 cuda_h.py:27] end prefill_ffn_prep cost 0.291 ms
DEBUG 05-06 15:59:57.295595.295595 cuda_h.py:27] end prefill_gate cost 0.303 ms
INFO 05-06 15:59:57.302003.302003 lmp.py:1823] [layer_moe_fused] layer=23 active_experts=105 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123], 'expert_count': 29, 'ideal_gpu_count': 27, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 7930, 'token_per_expert': {3: 9, 7: 5, 11: 116, 15: 11, 19: 17, 23: 409, 27: 62, 31: 538, 39: 251, 43: 7, 47: 3, 51: 3375, 55: 2, 59: 5, 63: 9, 67: 11, 71: 50, 75: 2, 79: 308, 83: 108, 87: 5, 91: 305, 99: 12, 103: 51, 107: 7, 111: 123, 115: 1, 119: 2114, 123: 14}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 36, 40, 44, 56, 60, 64, 72, 76, 80, 84, 88, 92, 96, 100, 104, 112, 116, 120, 124], 'expert_count': 27, 'ideal_gpu_count': 26, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 9004, 'token_per_expert': {0: 2, 4: 12, 8: 19, 12: 214, 16: 1, 20: 181, 24: 30, 28: 613, 36: 52, 40: 68, 44: 191, 56: 851, 60: 1, 64: 100, 72: 25, 76: 421, 80: 54, 84: 5, 88: 1716, 92: 879, 96: 33, 100: 12, 104: 1952, 112: 19, 116: 56, 120: 974, 124: 523}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 17, 21, 25, 29, 33, 37, 53, 57, 61, 65, 69, 73, 77, 81, 93, 97, 101, 105, 109, 117, 121], 'expert_count': 24, 'ideal_gpu_count': 26, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 9009, 'token_per_expert': {1: 472, 9: 350, 13: 263, 17: 257, 21: 106, 25: 17, 29: 916, 33: 998, 37: 135, 53: 2, 57: 214, 61: 210, 65: 14, 69: 1, 73: 3770, 77: 13, 81: 16, 93: 154, 97: 11, 101: 74, 105: 996, 109: 17, 117: 2, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 38, 42, 54, 62, 66, 70, 74, 78, 82, 90, 98, 106, 110, 114, 118, 122, 126], 'expert_count': 25, 'ideal_gpu_count': 26, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 6825, 'token_per_expert': {2: 57, 6: 1452, 10: 2, 14: 1557, 18: 214, 22: 54, 26: 461, 30: 15, 38: 410, 42: 398, 54: 2, 62: 482, 66: 25, 70: 20, 74: 1, 78: 230, 82: 1, 90: 26, 98: 17, 106: 230, 110: 55, 114: 1001, 118: 68, 122: 5, 126: 42}}
INFO 05-06 15:59:57.303278.303278 lmp.py:1845] [layer_moe_fused] layer=23 prefix: 7.105ms alloc: 0.249ms
INFO 05-06 15:59:57.303951.303951 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.173683166503906e-05 seconds
INFO 05-06 15:59:57.304678.304678 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0013060569763183594s
INFO 05-06 15:59:57.306093.306093 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001058340072631836s
DEBUG 05-06 15:59:57.306063.306063 cuda_h.py:27] end moe_wait_copy_tasks cost 1.145 ms
DEBUG 05-06 15:59:57.310610.310610 cuda_h.py:27] end moe_vllm_forward cost 4.661 ms
DEBUG 05-06 15:59:57.311360.311360 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.311111.311111 lmp.py:1964] [layer_moe_fused] vllm triton time: 4.953ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.311280.311280 cuda_h.py:27] end *layer_moe_fused cost 15.328 ms
DEBUG 05-06 15:59:57.357322.357322 cuda_h.py:27] end prefill_merge_scale cost 0.749 ms
DEBUG 05-06 15:59:57.358461.358461 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.073 ms
DEBUG 05-06 15:59:57.358169.358169 cuda_h.py:27] end prefill_layer cost 76.065 ms
DEBUG 05-06 15:59:57.358920.358920 lmp.py:1394] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 15:59:57.358724.358724 lmp.py:1350] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 15:59:57.359788.359788 cuda_h.py:27] end prefill_ln cost 0.357 ms
DEBUG 05-06 15:59:57.362278.362278 cuda_h.py:27] end prefill_attn cost 2.845 ms
DEBUG 05-06 15:59:57.363033.363033 cuda_h.py:27] end prefill_ffn_prep cost 0.669 ms
DEBUG 05-06 15:59:57.364993.364993 cuda_h.py:27] end prefill_gate cost 0.661 ms
INFO 05-06 15:59:57.367415.367415 lmp.py:1823] [layer_moe_fused] layer=24 active_experts=97 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 31, 43, 47, 51, 55, 59, 63, 67, 75, 79, 83, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 25, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 6493, 'token_per_expert': {3: 65, 7: 3, 11: 4, 15: 73, 19: 3, 23: 316, 31: 162, 43: 2663, 47: 424, 51: 163, 55: 22, 59: 269, 63: 77, 67: 77, 75: 304, 79: 12, 83: 611, 99: 8, 103: 3, 107: 4, 111: 184, 119: 3, 123: 3, 127: 1040}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 28, 32, 44, 60, 68, 72, 76, 80, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 23, 'ideal_gpu_count': 24, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 6834, 'token_per_expert': {0: 256, 4: 409, 8: 204, 12: 163, 16: 30, 24: 243, 28: 634, 32: 4, 44: 94, 60: 20, 68: 5, 72: 117, 76: 1386, 80: 304, 88: 6, 92: 59, 100: 2475, 104: 23, 108: 8, 112: 307, 116: 56, 120: 1, 124: 30}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 17, 21, 29, 37, 41, 45, 49, 53, 57, 73, 77, 85, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 8863, 'token_per_expert': {5: 1, 9: 32, 13: 55, 17: 114, 21: 1, 29: 184, 37: 117, 41: 58, 45: 205, 49: 1247, 53: 4, 57: 2, 73: 66, 77: 17, 85: 2, 93: 42, 97: 3541, 101: 105, 105: 131, 109: 2146, 113: 3, 117: 133, 121: 427, 125: 230}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 18, 30, 34, 38, 42, 46, 50, 58, 62, 66, 70, 74, 78, 82, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 26, 'ideal_gpu_count': 24, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 10578, 'token_per_expert': {2: 5, 10: 305, 14: 249, 18: 1978, 30: 1229, 34: 210, 38: 2, 42: 56, 46: 417, 50: 544, 58: 1, 62: 223, 66: 330, 70: 307, 74: 590, 78: 182, 82: 9, 90: 24, 98: 199, 102: 63, 106: 5, 110: 18, 114: 2169, 118: 470, 122: 4, 126: 989}}
INFO 05-06 15:59:57.368800.368800 lmp.py:1845] [layer_moe_fused] layer=24 prefix: 2.329ms alloc: 0.343ms
INFO 05-06 15:59:57.368804.368804 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 6.937980651855469e-05 seconds
INFO 05-06 15:59:57.370180.370180 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0016698837280273438s
INFO 05-06 15:59:57.371598.371598 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0015194416046142578s
DEBUG 05-06 15:59:57.371814.371814 cuda_h.py:27] end moe_wait_copy_tasks cost 1.649 ms
DEBUG 05-06 15:59:57.377089.377089 cuda_h.py:27] end moe_vllm_forward cost 5.741 ms
DEBUG 05-06 15:59:57.377647.377647 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.378180.378180 lmp.py:1964] [layer_moe_fused] vllm triton time: 6.120ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.378449.378449 cuda_h.py:27] end *layer_moe_fused cost 12.778 ms
DEBUG 05-06 15:59:57.378402.378402 cuda_h.py:27] end prefill_merge_scale cost 0.303 ms
DEBUG 05-06 15:59:57.378047.378047 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.378777.378777 cuda_h.py:27] end prefill_layer cost 19.988 ms
DEBUG 05-06 15:59:57.378404.378404 lmp.py:1394] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 15:59:57.379200.379200 lmp.py:1350] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 15:59:57.379895.379895 cuda_h.py:27] end prefill_ln cost 0.168 ms
DEBUG 05-06 15:59:57.390540.390540 cuda_h.py:27] end prefill_attn cost 11.100 ms
DEBUG 05-06 15:59:57.390135.390135 cuda_h.py:27] end prefill_ffn_prep cost 0.306 ms
DEBUG 05-06 15:59:57.391409.391409 cuda_h.py:27] end prefill_gate cost 0.297 ms
INFO 05-06 15:59:57.395440.395440 lmp.py:1823] [layer_moe_fused] layer=25 active_experts=107 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 27, 31, 35, 39, 47, 55, 59, 63, 67, 75, 79, 87, 95, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 27, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 7967, 'token_per_expert': {3: 12, 7: 824, 11: 1042, 19: 44, 27: 44, 31: 106, 35: 22, 39: 235, 47: 413, 55: 27, 59: 151, 63: 1092, 67: 16, 75: 56, 79: 55, 87: 15, 95: 484, 99: 1674, 107: 48, 111: 720, 115: 746, 119: 134, 123: 5, 127: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 92, 96, 100, 104, 120, 124], 'expert_count': 27, 'ideal_gpu_count': 27, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 5844, 'token_per_expert': {0: 1, 4: 1, 8: 9, 12: 30, 16: 60, 20: 12, 24: 5, 32: 202, 36: 493, 40: 2206, 44: 100, 48: 14, 52: 212, 56: 254, 60: 61, 64: 1, 68: 24, 72: 220, 76: 2, 80: 377, 84: 167, 92: 490, 96: 2, 100: 772, 104: 7, 120: 45, 124: 77}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 17, 21, 25, 29, 33, 37, 45, 49, 53, 57, 65, 69, 73, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 27, 'ideal_gpu_count': 27, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 9609, 'token_per_expert': {1: 7, 9: 35, 13: 284, 17: 24, 21: 766, 25: 368, 29: 4, 33: 2, 37: 897, 45: 457, 49: 28, 53: 983, 57: 20, 65: 2, 69: 801, 73: 21, 85: 10, 89: 656, 93: 126, 97: 12, 101: 10, 105: 73, 109: 114, 113: 219, 117: 1843, 121: 1844, 125: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 42, 46, 50, 54, 58, 62, 66, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 118, 122, 126], 'expert_count': 29, 'ideal_gpu_count': 26, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 9348, 'token_per_expert': {2: 176, 6: 80, 10: 614, 14: 96, 18: 2, 22: 145, 26: 83, 30: 19, 34: 1, 42: 3597, 46: 1, 50: 29, 54: 2, 58: 46, 62: 314, 66: 137, 74: 1392, 78: 377, 82: 1337, 86: 63, 90: 1, 94: 342, 98: 4, 102: 54, 106: 6, 110: 1, 118: 400, 122: 3, 126: 26}}
INFO 05-06 15:59:57.396602.396602 lmp.py:1845] [layer_moe_fused] layer=25 prefix: 4.095ms alloc: 0.237ms
INFO 05-06 15:59:57.396397.396397 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.817413330078125e-05 seconds
INFO 05-06 15:59:57.397109.397109 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012600421905517578s
INFO 05-06 15:59:57.398091.398091 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009829998016357422s
DEBUG 05-06 15:59:57.398299.398299 cuda_h.py:27] end moe_wait_copy_tasks cost 1.069 ms
DEBUG 05-06 15:59:57.403063.403063 cuda_h.py:27] end moe_vllm_forward cost 4.617 ms
DEBUG 05-06 15:59:57.403655.403655 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.403836.403836 lmp.py:1964] [layer_moe_fused] vllm triton time: 4.929ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.403728.403728 cuda_h.py:27] end *layer_moe_fused cost 12.046 ms
DEBUG 05-06 15:59:57.404724.404724 cuda_h.py:27] end prefill_merge_scale cost 0.300 ms
DEBUG 05-06 15:59:57.404939.404939 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.404716.404716 cuda_h.py:27] end prefill_layer cost 25.484 ms
DEBUG 05-06 15:59:57.404218.404218 lmp.py:1394] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 15:59:57.404013.404013 lmp.py:1350] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 15:59:57.404980.404980 cuda_h.py:27] end prefill_ln cost 0.157 ms
DEBUG 05-06 15:59:57.416416.416416 cuda_h.py:27] end prefill_attn cost 11.386 ms
DEBUG 05-06 15:59:57.416819.416819 cuda_h.py:27] end prefill_ffn_prep cost 0.305 ms
DEBUG 05-06 15:59:57.417881.417881 cuda_h.py:27] end prefill_gate cost 0.289 ms
INFO 05-06 15:59:57.422957.422957 lmp.py:1823] [layer_moe_fused] layer=26 active_experts=96 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 27, 31, 39, 43, 47, 51, 67, 79, 83, 87, 95, 99, 103, 107, 111, 115, 123, 127], 'expert_count': 22, 'ideal_gpu_count': 24, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 5875, 'token_per_expert': {3: 2, 7: 515, 11: 7, 23: 14, 27: 23, 31: 2, 39: 177, 43: 24, 47: 2, 51: 564, 67: 336, 79: 2, 83: 4, 87: 893, 95: 1, 99: 35, 103: 26, 107: 175, 111: 527, 115: 214, 123: 1171, 127: 1161}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 32, 36, 44, 60, 64, 68, 76, 80, 84, 88, 92, 96, 112, 120], 'expert_count': 21, 'ideal_gpu_count': 24, 'keep_on_gpu': 21, 'hit_count_on_device': 21, 'token_total': 4590, 'token_per_expert': {0: 4, 4: 2, 8: 4, 12: 44, 16: 114, 20: 191, 24: 143, 32: 2, 36: 175, 44: 2, 60: 328, 64: 27, 68: 122, 76: 40, 80: 10, 84: 10, 88: 18, 92: 38, 96: 311, 112: 136, 120: 2869}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 25, 29, 33, 37, 45, 49, 57, 61, 65, 69, 81, 85, 89, 93, 97, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 11248, 'token_per_expert': {1: 243, 5: 950, 13: 300, 21: 3619, 25: 14, 29: 2701, 33: 71, 37: 281, 45: 161, 49: 4, 57: 1, 61: 383, 65: 88, 69: 10, 81: 11, 85: 1841, 89: 114, 93: 111, 97: 2, 105: 23, 109: 26, 113: 262, 117: 9, 121: 2, 125: 21}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 14, 18, 22, 26, 30, 38, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 24, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 11055, 'token_per_expert': {6: 930, 10: 303, 14: 3762, 18: 218, 22: 335, 26: 164, 30: 27, 38: 1, 46: 575, 50: 1, 54: 1, 58: 1, 62: 196, 66: 144, 70: 23, 74: 44, 78: 907, 82: 100, 86: 1615, 90: 731, 94: 34, 98: 190, 102: 57, 106: 2, 110: 158, 114: 2, 122: 385, 126: 149}}
INFO 05-06 15:59:57.422828.422828 lmp.py:1845] [layer_moe_fused] layer=26 prefix: 4.891ms alloc: 0.234ms
INFO 05-06 15:59:57.423333.423333 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.601478576660156e-05 seconds
INFO 05-06 15:59:57.424186.424186 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012941360473632812s
INFO 05-06 15:59:57.425553.425553 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001024007797241211s
DEBUG 05-06 15:59:57.425377.425377 cuda_h.py:27] end moe_wait_copy_tasks cost 1.107 ms
DEBUG 05-06 15:59:57.430142.430142 cuda_h.py:27] end moe_vllm_forward cost 4.651 ms
DEBUG 05-06 15:59:57.430654.430654 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.430404.430404 lmp.py:1964] [layer_moe_fused] vllm triton time: 4.938ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.430058.430058 cuda_h.py:27] end *layer_moe_fused cost 12.986 ms
DEBUG 05-06 15:59:57.431747.431747 cuda_h.py:27] end prefill_merge_scale cost 0.299 ms
DEBUG 05-06 15:59:57.431677.431677 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 15:59:57.431599.431599 cuda_h.py:27] end prefill_layer cost 26.801 ms
DEBUG 05-06 15:59:57.431342.431342 lmp.py:1394] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 15:59:57.431105.431105 lmp.py:1350] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 15:59:57.431946.431946 cuda_h.py:27] end prefill_ln cost 0.169 ms
DEBUG 05-06 15:59:57.443627.443627 cuda_h.py:27] end prefill_attn cost 11.194 ms
DEBUG 05-06 15:59:57.443011.443011 cuda_h.py:27] end prefill_ffn_prep cost 0.327 ms
DEBUG 05-06 15:59:57.444074.444074 cuda_h.py:27] end prefill_gate cost 0.303 ms
INFO 05-06 15:59:57.448151.448151 lmp.py:1823] [layer_moe_fused] layer=27 active_experts=96 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 24, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 8954, 'token_per_expert': {3: 4, 7: 2205, 11: 17, 15: 36, 19: 89, 23: 508, 27: 2010, 39: 6, 43: 4, 47: 430, 51: 232, 55: 2, 59: 1489, 63: 44, 67: 46, 71: 155, 75: 59, 79: 16, 83: 26, 87: 259, 91: 42, 99: 6, 103: 186, 107: 85, 111: 620, 115: 12, 119: 37, 123: 232, 127: 97}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 12, 16, 20, 28, 36, 40, 44, 48, 56, 64, 68, 72, 76, 80, 84, 88, 92, 96, 104, 112, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 12616, 'token_per_expert': {0: 1, 8: 1099, 12: 59, 16: 822, 20: 81, 28: 24, 36: 806, 40: 63, 44: 53, 48: 463, 56: 61, 64: 2326, 68: 73, 72: 3, 76: 101, 80: 2315, 84: 20, 88: 1, 92: 11, 96: 571, 104: 491, 112: 2, 120: 8, 124: 3162}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 29, 37, 49, 53, 57, 61, 69, 77, 81, 85, 89, 93, 97, 101, 109, 113, 117, 125], 'expert_count': 23, 'ideal_gpu_count': 24, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 5320, 'token_per_expert': {1: 101, 5: 1, 9: 135, 13: 400, 25: 1183, 29: 7, 37: 34, 49: 3, 53: 3, 57: 497, 61: 655, 69: 40, 77: 96, 81: 212, 85: 733, 89: 385, 93: 700, 97: 15, 101: 28, 109: 5, 113: 46, 117: 8, 125: 33}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 18, 22, 34, 50, 54, 58, 62, 66, 74, 82, 86, 90, 94, 110, 114, 118, 126], 'expert_count': 20, 'ideal_gpu_count': 24, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 5878, 'token_per_expert': {2: 1721, 10: 11, 14: 207, 18: 537, 22: 291, 34: 27, 50: 104, 54: 332, 58: 464, 62: 268, 66: 341, 74: 90, 82: 42, 86: 2, 90: 1208, 94: 1, 110: 71, 114: 153, 118: 2, 126: 6}}
INFO 05-06 15:59:57.449591.449591 lmp.py:1845] [layer_moe_fused] layer=27 prefix: 4.115ms alloc: 0.232ms
INFO 05-06 15:59:57.449373.449373 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-06 15:59:57.450575.450575 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014448165893554688s
INFO 05-06 15:59:57.452263.452263 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011134147644042969s
DEBUG 05-06 15:59:57.452310.452310 cuda_h.py:27] end moe_wait_copy_tasks cost 1.326 ms
DEBUG 05-06 15:59:57.457003.457003 cuda_h.py:27] end moe_vllm_forward cost 5.033 ms
DEBUG 05-06 15:59:57.457230.457230 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.457173.457173 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.331ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.457257.457257 cuda_h.py:27] end *layer_moe_fused cost 13.161 ms
DEBUG 05-06 15:59:57.458263.458263 cuda_h.py:27] end prefill_merge_scale cost 0.309 ms
DEBUG 05-06 15:59:57.458199.458199 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.038 ms
DEBUG 05-06 15:59:57.458506.458506 cuda_h.py:27] end prefill_layer cost 26.806 ms
DEBUG 05-06 15:59:57.458803.458803 lmp.py:1394] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 15:59:57.458122.458122 lmp.py:1350] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 15:59:57.458334.458334 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 15:59:57.470933.470933 cuda_h.py:27] end prefill_attn cost 11.135 ms
DEBUG 05-06 15:59:57.470999.470999 cuda_h.py:27] end prefill_ffn_prep cost 0.301 ms
DEBUG 05-06 15:59:57.471101.471101 cuda_h.py:27] end prefill_gate cost 0.302 ms
INFO 05-06 15:59:57.476718.476718 lmp.py:1823] [layer_moe_fused] layer=28 active_experts=100 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 23, 27, 31, 35, 39, 43, 55, 59, 63, 79, 87, 91, 99, 111, 115, 119, 123], 'expert_count': 21, 'ideal_gpu_count': 25, 'keep_on_gpu': 21, 'hit_count_on_device': 21, 'token_total': 8665, 'token_per_expert': {3: 940, 7: 36, 11: 39, 15: 1075, 23: 111, 27: 58, 31: 31, 35: 2291, 39: 218, 43: 362, 55: 631, 59: 15, 63: 40, 79: 1244, 87: 136, 91: 286, 99: 45, 111: 498, 115: 342, 119: 6, 123: 261}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 12, 16, 20, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 84, 88, 92, 96, 104, 112, 116, 124], 'expert_count': 26, 'ideal_gpu_count': 25, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 7092, 'token_per_expert': {0: 256, 8: 35, 12: 59, 16: 1, 20: 3, 24: 845, 32: 1036, 36: 1231, 40: 1214, 44: 4, 48: 363, 52: 980, 56: 197, 60: 85, 64: 2, 68: 7, 72: 8, 76: 269, 84: 121, 88: 70, 92: 7, 96: 37, 104: 250, 112: 2, 116: 3, 124: 7}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 45, 49, 53, 57, 65, 73, 77, 81, 85, 89, 93, 97, 105, 109, 113, 117, 121], 'expert_count': 27, 'ideal_gpu_count': 25, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 13091, 'token_per_expert': {1: 103, 5: 244, 9: 71, 13: 14, 17: 40, 21: 12, 25: 3, 29: 11, 33: 299, 37: 1, 45: 3639, 49: 458, 53: 99, 57: 499, 65: 2242, 73: 4, 77: 186, 81: 135, 85: 85, 89: 667, 93: 7, 97: 53, 105: 24, 109: 1622, 113: 2278, 117: 204, 121: 91}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 58, 62, 70, 74, 78, 82, 90, 98, 102, 106, 110, 114, 118, 122], 'expert_count': 26, 'ideal_gpu_count': 25, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 3920, 'token_per_expert': {2: 77, 6: 28, 14: 9, 18: 329, 22: 1, 26: 17, 30: 190, 34: 56, 38: 68, 42: 1, 46: 1, 50: 12, 58: 303, 62: 1, 70: 1511, 74: 8, 78: 101, 82: 67, 90: 202, 98: 6, 102: 108, 106: 207, 110: 8, 114: 2, 118: 321, 122: 286}}
INFO 05-06 15:59:57.476490.476490 lmp.py:1845] [layer_moe_fused] layer=28 prefix: 4.834ms alloc: 0.229ms
INFO 05-06 15:59:57.476195.476195 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.76837158203125e-05 seconds
INFO 05-06 15:59:57.478452.478452 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014848709106445312s
INFO 05-06 15:59:57.479442.479442 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0012297630310058594s
DEBUG 05-06 15:59:57.479650.479650 cuda_h.py:27] end moe_wait_copy_tasks cost 1.316 ms
DEBUG 05-06 15:59:57.484475.484475 cuda_h.py:27] end moe_vllm_forward cost 4.659 ms
DEBUG 05-06 15:59:57.484464.484464 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.484168.484168 lmp.py:1964] [layer_moe_fused] vllm triton time: 4.949ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.484444.484444 cuda_h.py:27] end *layer_moe_fused cost 13.388 ms
DEBUG 05-06 15:59:57.485353.485353 cuda_h.py:27] end prefill_merge_scale cost 0.306 ms
DEBUG 05-06 15:59:57.485144.485144 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.037 ms
DEBUG 05-06 15:59:57.485020.485020 cuda_h.py:27] end prefill_layer cost 26.820 ms
DEBUG 05-06 15:59:57.485364.485364 lmp.py:1394] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 15:59:57.485160.485160 lmp.py:1350] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 15:59:57.485471.485471 cuda_h.py:27] end prefill_ln cost 0.166 ms
DEBUG 05-06 15:59:57.497703.497703 cuda_h.py:27] end prefill_attn cost 11.425 ms
DEBUG 05-06 15:59:57.497636.497636 cuda_h.py:27] end prefill_ffn_prep cost 0.309 ms
DEBUG 05-06 15:59:57.498681.498681 cuda_h.py:27] end prefill_gate cost 0.297 ms
INFO 05-06 15:59:57.506411.506411 lmp.py:1823] [layer_moe_fused] layer=29 active_experts=96 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 31, 35, 43, 47, 51, 55, 59, 63, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 4601, 'token_per_expert': {3: 27, 7: 360, 11: 101, 15: 39, 31: 1, 35: 35, 43: 27, 47: 5, 51: 1041, 55: 40, 59: 51, 63: 720, 71: 2, 75: 1, 79: 462, 83: 227, 87: 12, 91: 4, 95: 14, 99: 30, 103: 162, 107: 264, 111: 547, 115: 47, 119: 382}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 24, 28, 32, 36, 40, 44, 48, 56, 60, 64, 72, 76, 80, 84, 92, 96, 100, 104, 108, 112, 124], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 8820, 'token_per_expert': {0: 612, 4: 45, 12: 19, 24: 1, 28: 23, 32: 1, 36: 428, 40: 1002, 44: 11, 48: 691, 56: 412, 60: 620, 64: 1, 72: 262, 76: 71, 80: 11, 84: 3, 92: 63, 96: 76, 100: 3, 104: 41, 108: 29, 112: 3747, 124: 648}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 33, 41, 45, 49, 53, 57, 65, 69, 77, 85, 89, 93, 97, 101, 105, 109, 113, 121], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 10712, 'token_per_expert': {1: 66, 5: 84, 9: 72, 13: 183, 17: 3556, 21: 17, 25: 4, 33: 170, 41: 30, 45: 24, 49: 1, 53: 5, 57: 321, 65: 2, 69: 55, 77: 1236, 85: 1091, 89: 141, 93: 574, 97: 2, 101: 36, 105: 2299, 109: 93, 113: 14, 121: 636}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 30, 38, 42, 46, 54, 58, 62, 70, 78, 82, 90, 94, 98, 102, 106, 110, 114, 118, 126], 'expert_count': 22, 'ideal_gpu_count': 24, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 8635, 'token_per_expert': {2: 996, 6: 334, 10: 215, 30: 2634, 38: 58, 42: 6, 46: 6, 54: 84, 58: 25, 62: 15, 70: 2, 78: 619, 82: 19, 90: 45, 94: 65, 98: 3, 102: 40, 106: 6, 110: 2716, 114: 687, 118: 3, 126: 57}}
INFO 05-06 15:59:57.506261.506261 lmp.py:1845] [layer_moe_fused] layer=29 prefix: 7.584ms alloc: 0.221ms
INFO 05-06 15:59:57.506789.506789 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.982948303222656e-05 seconds
INFO 05-06 15:59:57.508688.508688 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012936592102050781s
INFO 05-06 15:59:57.509305.509305 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009624958038330078s
DEBUG 05-06 15:59:57.509844.509844 cuda_h.py:27] end moe_wait_copy_tasks cost 1.047 ms
DEBUG 05-06 15:59:57.514155.514155 cuda_h.py:27] end moe_vllm_forward cost 4.738 ms
DEBUG 05-06 15:59:57.514905.514905 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 15:59:57.514894.514894 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.026ms (seq_len=512 cg=False)
DEBUG 05-06 15:59:57.514031.514031 cuda_h.py:27] end *layer_moe_fused cost 15.784 ms
DEBUG 05-06 15:59:57.515251.515251 cuda_h.py:27] end prefill_merge_scale cost 0.777 ms
DEBUG 05-06 15:59:57.515990.515990 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.033 ms
DEBUG 05-06 15:59:57.515958.515958 cuda_h.py:27] end prefill_layer cost 29.976 ms
DEBUG 05-06 15:59:57.515148.515148 lmp.py:1394] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 15:59:57.515944.515944 cuda_h.py:27] end prefill_step cost 1590.564 ms
INFO 05-06 15:59:57.515593.515593 lmp.py:1397] prefill time: 1.689072847366333 seconds
INFO 05-06 15:59:57.552417.552417 lmp.py:1409] Static-KV prefill complete; seqlens set to 512.
DEBUG 05-06 15:59:57.699054.699054 cuda_h.py:27] end init_inputs_tokens cost 146.856 ms
DEBUG 05-06 15:59:57.699986.699986 lmp.py:1510] decode step 0 next_inputs_tokens shape=(8, 1, 2816)
DEBUG 05-06 15:59:57.699491.699491 lmp.py:1516] ---- decode step 0 layer 0 ----
DEBUG 05-06 15:59:57.728168.728168 cuda_h.py:27] end decode_layer cost 29.690 ms
DEBUG 05-06 15:59:57.729813.729813 lmp.py:1516] ---- decode step 0 layer 1 ----
DEBUG 05-06 15:59:57.734451.734451 cuda_h.py:27] end decode_layer cost 5.590 ms
DEBUG 05-06 15:59:57.734115.734115 lmp.py:1516] ---- decode step 0 layer 2 ----
DEBUG 05-06 15:59:57.739469.739469 cuda_h.py:27] end decode_layer cost 5.029 ms
DEBUG 05-06 15:59:57.739511.739511 lmp.py:1516] ---- decode step 0 layer 3 ----
DEBUG 05-06 15:59:57.745499.745499 cuda_h.py:27] end decode_layer cost 5.358 ms
DEBUG 05-06 15:59:57.745965.745965 lmp.py:1516] ---- decode step 0 layer 4 ----
DEBUG 05-06 15:59:57.750061.750061 cuda_h.py:27] end decode_layer cost 5.051 ms
DEBUG 05-06 15:59:57.750811.750811 lmp.py:1516] ---- decode step 0 layer 5 ----
DEBUG 05-06 15:59:57.772493.772493 cuda_h.py:27] end decode_layer cost 22.336 ms
DEBUG 05-06 15:59:57.772535.772535 lmp.py:1516] ---- decode step 0 layer 6 ----
DEBUG 05-06 15:59:57.777942.777942 cuda_h.py:27] end decode_layer cost 5.034 ms
DEBUG 05-06 15:59:57.777786.777786 lmp.py:1516] ---- decode step 0 layer 7 ----
DEBUG 05-06 15:59:57.783362.783362 cuda_h.py:27] end decode_layer cost 5.546 ms
DEBUG 05-06 15:59:57.783682.783682 lmp.py:1516] ---- decode step 0 layer 8 ----
DEBUG 05-06 15:59:57.788877.788877 cuda_h.py:27] end decode_layer cost 5.018 ms
DEBUG 05-06 15:59:57.788389.788389 lmp.py:1516] ---- decode step 0 layer 9 ----
DEBUG 05-06 15:59:57.793446.793446 cuda_h.py:27] end decode_layer cost 5.057 ms
DEBUG 05-06 15:59:57.793289.793289 lmp.py:1516] ---- decode step 0 layer 10 ----
DEBUG 05-06 15:59:57.798979.798979 cuda_h.py:27] end decode_layer cost 4.998 ms
DEBUG 05-06 15:59:57.798346.798346 lmp.py:1516] ---- decode step 0 layer 11 ----
DEBUG 05-06 15:59:57.804053.804053 cuda_h.py:27] end decode_layer cost 5.501 ms
DEBUG 05-06 15:59:57.804764.804764 lmp.py:1516] ---- decode step 0 layer 12 ----
DEBUG 05-06 15:59:57.809361.809361 cuda_h.py:27] end decode_layer cost 4.964 ms
DEBUG 05-06 15:59:57.809489.809489 lmp.py:1516] ---- decode step 0 layer 13 ----
DEBUG 05-06 15:59:57.814956.814956 cuda_h.py:27] end decode_layer cost 5.043 ms
DEBUG 05-06 15:59:57.814091.814091 lmp.py:1516] ---- decode step 0 layer 14 ----
DEBUG 05-06 15:59:57.819608.819608 cuda_h.py:27] end decode_layer cost 4.975 ms
DEBUG 05-06 15:59:57.819120.819120 lmp.py:1516] ---- decode step 0 layer 15 ----
DEBUG 05-06 15:59:57.825084.825084 cuda_h.py:27] end decode_layer cost 6.042 ms
DEBUG 05-06 15:59:57.825543.825543 lmp.py:1516] ---- decode step 0 layer 16 ----
DEBUG 05-06 15:59:57.830524.830524 cuda_h.py:27] end decode_layer cost 4.966 ms
DEBUG 05-06 15:59:57.830367.830367 lmp.py:1516] ---- decode step 0 layer 17 ----
DEBUG 05-06 15:59:57.835807.835807 cuda_h.py:27] end decode_layer cost 5.198 ms
DEBUG 05-06 15:59:57.835411.835411 lmp.py:1516] ---- decode step 0 layer 18 ----
DEBUG 05-06 15:59:57.840555.840555 cuda_h.py:27] end decode_layer cost 4.875 ms
DEBUG 05-06 15:59:57.840828.840828 lmp.py:1516] ---- decode step 0 layer 19 ----
DEBUG 05-06 15:59:57.845853.845853 cuda_h.py:27] end decode_layer cost 5.068 ms
DEBUG 05-06 15:59:57.845788.845788 lmp.py:1516] ---- decode step 0 layer 20 ----
DEBUG 05-06 15:59:57.850533.850533 cuda_h.py:27] end decode_layer cost 5.038 ms
DEBUG 05-06 15:59:57.851152.851152 lmp.py:1516] ---- decode step 0 layer 21 ----
DEBUG 05-06 15:59:57.856238.856238 cuda_h.py:27] end decode_layer cost 4.938 ms
DEBUG 05-06 15:59:57.856796.856796 lmp.py:1516] ---- decode step 0 layer 22 ----
DEBUG 05-06 15:59:57.861071.861071 cuda_h.py:27] end decode_layer cost 5.033 ms
DEBUG 05-06 15:59:57.861437.861437 lmp.py:1516] ---- decode step 0 layer 23 ----
DEBUG 05-06 15:59:57.866459.866459 cuda_h.py:27] end decode_layer cost 5.382 ms
DEBUG 05-06 15:59:57.866110.866110 lmp.py:1516] ---- decode step 0 layer 24 ----
DEBUG 05-06 15:59:57.871599.871599 cuda_h.py:27] end decode_layer cost 4.919 ms
DEBUG 05-06 15:59:57.871363.871363 lmp.py:1516] ---- decode step 0 layer 25 ----
DEBUG 05-06 15:59:57.876698.876698 cuda_h.py:27] end decode_layer cost 5.052 ms
DEBUG 05-06 15:59:57.876541.876541 lmp.py:1516] ---- decode step 0 layer 26 ----
DEBUG 05-06 15:59:57.881398.881398 cuda_h.py:27] end decode_layer cost 5.015 ms
DEBUG 05-06 15:59:57.881810.881810 lmp.py:1516] ---- decode step 0 layer 27 ----
DEBUG 05-06 15:59:57.886022.886022 cuda_h.py:27] end decode_layer cost 4.926 ms
DEBUG 05-06 15:59:57.886216.886216 lmp.py:1516] ---- decode step 0 layer 28 ----
DEBUG 05-06 15:59:57.891344.891344 cuda_h.py:27] end decode_layer cost 5.004 ms
DEBUG 05-06 15:59:57.891426.891426 lmp.py:1516] ---- decode step 0 layer 29 ----
DEBUG 05-06 15:59:57.897018.897018 cuda_h.py:27] end decode_layer cost 5.206 ms
DEBUG 05-06 15:59:57.897268.897268 cuda_h.py:27] end decode_step cost 345.107 ms
INFO 05-06 15:59:57.897706.897706 lmp.py:1564] decode step 0 time: 0.3451540470123291 seconds
DEBUG 05-06 15:59:57.904977.904977 cuda_h.py:27] end init_inputs_tokens cost 6.827 ms
DEBUG 05-06 15:59:57.904966.904966 lmp.py:1510] decode step 1 next_inputs_tokens shape=(8, 1, 2816)
DEBUG 05-06 15:59:57.904736.904736 lmp.py:1516] ---- decode step 1 layer 0 ----
DEBUG 05-06 15:59:57.909616.909616 cuda_h.py:27] end decode_layer cost 4.926 ms
DEBUG 05-06 15:59:57.909221.909221 lmp.py:1516] ---- decode step 1 layer 1 ----
DEBUG 05-06 15:59:57.914937.914937 cuda_h.py:27] end decode_layer cost 4.982 ms
DEBUG 05-06 15:59:57.914588.914588 lmp.py:1516] ---- decode step 1 layer 2 ----
DEBUG 05-06 15:59:57.919100.919100 cuda_h.py:27] end decode_layer cost 5.181 ms
DEBUG 05-06 15:59:57.919327.919327 lmp.py:1516] ---- decode step 1 layer 3 ----
DEBUG 05-06 15:59:57.924276.924276 cuda_h.py:27] end decode_layer cost 5.013 ms
DEBUG 05-06 15:59:57.924027.924027 lmp.py:1516] ---- decode step 1 layer 4 ----
DEBUG 05-06 15:59:57.929140.929140 cuda_h.py:27] end decode_layer cost 4.958 ms
DEBUG 05-06 15:59:57.929937.929937 lmp.py:1516] ---- decode step 1 layer 5 ----
DEBUG 05-06 15:59:57.935535.935535 cuda_h.py:27] end decode_layer cost 5.386 ms
DEBUG 05-06 15:59:57.935662.935662 lmp.py:1516] ---- decode step 1 layer 6 ----
DEBUG 05-06 15:59:57.940909.940909 cuda_h.py:27] end decode_layer cost 4.986 ms
DEBUG 05-06 15:59:57.940659.940659 lmp.py:1516] ---- decode step 1 layer 7 ----
DEBUG 05-06 15:59:57.945578.945578 cuda_h.py:27] end decode_layer cost 4.886 ms
DEBUG 05-06 15:59:57.945229.945229 lmp.py:1516] ---- decode step 1 layer 8 ----
DEBUG 05-06 15:59:57.950837.950837 cuda_h.py:27] end decode_layer cost 4.902 ms
DEBUG 05-06 15:59:57.950588.950588 lmp.py:1516] ---- decode step 1 layer 9 ----
DEBUG 05-06 15:59:57.957372.957372 cuda_h.py:27] end decode_layer cost 7.204 ms
DEBUG 05-06 15:59:57.957064.957064 lmp.py:1516] ---- decode step 1 layer 10 ----
DEBUG 05-06 15:59:57.962523.962523 cuda_h.py:27] end decode_layer cost 5.003 ms
DEBUG 05-06 15:59:57.962412.962412 lmp.py:1516] ---- decode step 1 layer 11 ----
DEBUG 05-06 15:59:57.967502.967502 cuda_h.py:27] end decode_layer cost 5.256 ms
DEBUG 05-06 15:59:57.967061.967061 lmp.py:1516] ---- decode step 1 layer 12 ----
DEBUG 05-06 15:59:57.972127.972127 cuda_h.py:27] end decode_layer cost 4.923 ms
DEBUG 05-06 15:59:57.972162.972162 lmp.py:1516] ---- decode step 1 layer 13 ----
DEBUG 05-06 15:59:57.977650.977650 cuda_h.py:27] end decode_layer cost 5.095 ms
DEBUG 05-06 15:59:57.977209.977209 lmp.py:1516] ---- decode step 1 layer 14 ----
DEBUG 05-06 15:59:57.982864.982864 cuda_h.py:27] end decode_layer cost 4.937 ms
DEBUG 05-06 15:59:57.982277.982277 lmp.py:1516] ---- decode step 1 layer 15 ----
DEBUG 05-06 15:59:57.987600.987600 cuda_h.py:27] end decode_layer cost 4.902 ms
DEBUG 05-06 15:59:57.987397.987397 lmp.py:1516] ---- decode step 1 layer 16 ----
DEBUG 05-06 15:59:57.992465.992465 cuda_h.py:27] end decode_layer cost 4.995 ms
DEBUG 05-06 15:59:57.992738.992738 lmp.py:1516] ---- decode step 1 layer 17 ----
DEBUG 05-06 15:59:57.998250.998250 cuda_h.py:27] end decode_layer cost 5.181 ms
DEBUG 05-06 15:59:57.998808.998808 lmp.py:1516] ---- decode step 1 layer 18 ----
DEBUG 05-06 15:59:58.003815.003815 cuda_h.py:27] end decode_layer cost 4.950 ms
DEBUG 05-06 15:59:58.003327.003327 lmp.py:1516] ---- decode step 1 layer 19 ----
DEBUG 05-06 15:59:58.008631.008631 cuda_h.py:27] end decode_layer cost 4.923 ms
DEBUG 05-06 15:59:58.008382.008382 lmp.py:1516] ---- decode step 1 layer 20 ----
DEBUG 05-06 15:59:58.013575.013575 cuda_h.py:27] end decode_layer cost 4.983 ms
DEBUG 05-06 15:59:58.013465.013465 lmp.py:1516] ---- decode step 1 layer 21 ----
DEBUG 05-06 15:59:58.018947.018947 cuda_h.py:27] end decode_layer cost 4.915 ms
DEBUG 05-06 15:59:58.018837.018837 lmp.py:1516] ---- decode step 1 layer 22 ----
DEBUG 05-06 15:59:58.023295.023295 cuda_h.py:27] end decode_layer cost 4.966 ms
DEBUG 05-06 15:59:58.023065.023065 lmp.py:1516] ---- decode step 1 layer 23 ----
DEBUG 05-06 15:59:58.028814.028814 cuda_h.py:27] end decode_layer cost 5.146 ms
DEBUG 05-06 15:59:58.028611.028611 lmp.py:1516] ---- decode step 1 layer 24 ----
DEBUG 05-06 15:59:58.033520.033520 cuda_h.py:27] end decode_layer cost 4.983 ms
DEBUG 05-06 15:59:58.033747.033747 lmp.py:1516] ---- decode step 1 layer 25 ----
DEBUG 05-06 15:59:58.038840.038840 cuda_h.py:27] end decode_layer cost 4.943 ms
DEBUG 05-06 15:59:58.038491.038491 lmp.py:1516] ---- decode step 1 layer 26 ----
DEBUG 05-06 15:59:58.043675.043675 cuda_h.py:27] end decode_layer cost 4.905 ms
DEBUG 05-06 15:59:58.043326.043326 lmp.py:1516] ---- decode step 1 layer 27 ----
DEBUG 05-06 15:59:58.048892.048892 cuda_h.py:27] end decode_layer cost 5.046 ms
DEBUG 05-06 15:59:58.048120.048120 lmp.py:1516] ---- decode step 1 layer 28 ----
DEBUG 05-06 15:59:58.053524.053524 cuda_h.py:27] end decode_layer cost 4.963 ms
DEBUG 05-06 15:59:58.053083.053083 lmp.py:1516] ---- decode step 1 layer 29 ----
DEBUG 05-06 15:59:58.058190.058190 cuda_h.py:27] end decode_layer cost 5.165 ms
DEBUG 05-06 15:59:58.058809.058809 cuda_h.py:27] end decode_step cost 161.624 ms
INFO 05-06 15:59:58.059148.059148 lmp.py:1564] decode step 1 time: 0.1616654396057129 seconds
DEBUG 05-06 15:59:58.065140.065140 cuda_h.py:27] end init_inputs_tokens cost 6.799 ms
DEBUG 05-06 15:59:58.065559.065559 lmp.py:1510] decode step 2 next_inputs_tokens shape=(8, 1, 2816)
DEBUG 05-06 15:59:58.066190.066190 lmp.py:1516] ---- decode step 2 layer 0 ----
DEBUG 05-06 15:59:58.070615.070615 cuda_h.py:27] end decode_layer cost 4.976 ms
DEBUG 05-06 15:59:58.071650.071650 lmp.py:1516] ---- decode step 2 layer 1 ----
DEBUG 05-06 15:59:58.076639.076639 cuda_h.py:27] end decode_layer cost 5.007 ms
DEBUG 05-06 15:59:58.076105.076105 lmp.py:1516] ---- decode step 2 layer 2 ----
DEBUG 05-06 15:59:58.081441.081441 cuda_h.py:27] end decode_layer cost 5.442 ms
DEBUG 05-06 15:59:58.081829.081829 lmp.py:1516] ---- decode step 2 layer 3 ----
DEBUG 05-06 15:59:58.086953.086953 cuda_h.py:27] end decode_layer cost 5.109 ms
DEBUG 05-06 15:59:58.086558.086558 lmp.py:1516] ---- decode step 2 layer 4 ----
DEBUG 05-06 15:59:58.091458.091458 cuda_h.py:27] end decode_layer cost 4.905 ms
DEBUG 05-06 15:59:58.091208.091208 lmp.py:1516] ---- decode step 2 layer 5 ----
DEBUG 05-06 15:59:58.097931.097931 cuda_h.py:27] end decode_layer cost 5.163 ms
DEBUG 05-06 15:59:58.097582.097582 lmp.py:1516] ---- decode step 2 layer 6 ----
DEBUG 05-06 15:59:58.102410.102410 cuda_h.py:27] end decode_layer cost 4.959 ms
DEBUG 05-06 15:59:58.102684.102684 lmp.py:1516] ---- decode step 2 layer 7 ----
DEBUG 05-06 15:59:58.107736.107736 cuda_h.py:27] end decode_layer cost 4.914 ms
DEBUG 05-06 15:59:58.107864.107864 lmp.py:1516] ---- decode step 2 layer 8 ----
DEBUG 05-06 15:59:58.112971.112971 cuda_h.py:27] end decode_layer cost 4.989 ms
DEBUG 05-06 15:59:58.112921.112921 lmp.py:1516] ---- decode step 2 layer 9 ----
DEBUG 05-06 15:59:58.117146.117146 cuda_h.py:27] end decode_layer cost 4.935 ms
DEBUG 05-06 15:59:58.117942.117942 lmp.py:1516] ---- decode step 2 layer 10 ----
DEBUG 05-06 15:59:58.122955.122955 cuda_h.py:27] end decode_layer cost 4.919 ms
DEBUG 05-06 15:59:58.122606.122606 lmp.py:1516] ---- decode step 2 layer 11 ----
DEBUG 05-06 15:59:58.127659.127659 cuda_h.py:27] end decode_layer cost 5.124 ms
DEBUG 05-06 15:59:58.127071.127071 lmp.py:1516] ---- decode step 2 layer 12 ----
DEBUG 05-06 15:59:58.132224.132224 cuda_h.py:27] end decode_layer cost 4.951 ms
DEBUG 05-06 15:59:58.132736.132736 lmp.py:1516] ---- decode step 2 layer 13 ----
DEBUG 05-06 15:59:58.137776.137776 cuda_h.py:27] end decode_layer cost 4.939 ms
DEBUG 05-06 15:59:58.137903.137903 lmp.py:1516] ---- decode step 2 layer 14 ----
DEBUG 05-06 15:59:58.142989.142989 cuda_h.py:27] end decode_layer cost 4.939 ms
DEBUG 05-06 15:59:58.142117.142117 lmp.py:1516] ---- decode step 2 layer 15 ----
DEBUG 05-06 15:59:58.147764.147764 cuda_h.py:27] end decode_layer cost 4.860 ms
DEBUG 05-06 15:59:58.147176.147176 lmp.py:1516] ---- decode step 2 layer 16 ----
DEBUG 05-06 15:59:58.152006.152006 cuda_h.py:27] end decode_layer cost 4.995 ms
DEBUG 05-06 15:59:58.152948.152948 lmp.py:1516] ---- decode step 2 layer 17 ----
DEBUG 05-06 15:59:58.157964.157964 cuda_h.py:27] end decode_layer cost 5.202 ms
DEBUG 05-06 15:59:58.157522.157522 lmp.py:1516] ---- decode step 2 layer 18 ----
DEBUG 05-06 15:59:58.162239.162239 cuda_h.py:27] end decode_layer cost 4.982 ms
DEBUG 05-06 15:59:58.162274.162274 lmp.py:1516] ---- decode step 2 layer 19 ----
DEBUG 05-06 15:59:58.167072.167072 cuda_h.py:27] end decode_layer cost 5.041 ms
DEBUG 05-06 15:59:58.167253.167253 lmp.py:1516] ---- decode step 2 layer 20 ----
DEBUG 05-06 15:59:58.172605.172605 cuda_h.py:27] end decode_layer cost 4.958 ms
DEBUG 05-06 15:59:58.172309.172309 lmp.py:1516] ---- decode step 2 layer 21 ----
DEBUG 05-06 15:59:58.177785.177785 cuda_h.py:27] end decode_layer cost 5.120 ms
DEBUG 05-06 15:59:58.177251.177251 lmp.py:1516] ---- decode step 2 layer 22 ----
DEBUG 05-06 15:59:58.182457.182457 cuda_h.py:27] end decode_layer cost 4.956 ms
DEBUG 05-06 15:59:58.183114.183114 lmp.py:1516] ---- decode step 2 layer 23 ----
DEBUG 05-06 15:59:58.188240.188240 cuda_h.py:27] end decode_layer cost 5.143 ms
DEBUG 05-06 15:59:58.188468.188468 lmp.py:1516] ---- decode step 2 layer 24 ----
DEBUG 05-06 15:59:58.193467.193467 cuda_h.py:27] end decode_layer cost 4.909 ms
DEBUG 05-06 15:59:58.193071.193071 lmp.py:1516] ---- decode step 2 layer 25 ----
DEBUG 05-06 15:59:58.198013.198013 cuda_h.py:27] end decode_layer cost 4.972 ms
DEBUG 05-06 15:59:58.198194.198194 lmp.py:1516] ---- decode step 2 layer 26 ----
DEBUG 05-06 15:59:58.203836.203836 cuda_h.py:27] end decode_layer cost 4.926 ms
DEBUG 05-06 15:59:58.203109.203109 lmp.py:1516] ---- decode step 2 layer 27 ----
DEBUG 05-06 15:59:58.208649.208649 cuda_h.py:27] end decode_layer cost 5.026 ms
DEBUG 05-06 15:59:58.208922.208922 lmp.py:1516] ---- decode step 2 layer 28 ----
DEBUG 05-06 15:59:58.213042.213042 cuda_h.py:27] end decode_layer cost 4.964 ms
DEBUG 05-06 15:59:58.213409.213409 lmp.py:1516] ---- decode step 2 layer 29 ----
DEBUG 05-06 15:59:58.218055.218055 cuda_h.py:27] end decode_layer cost 5.245 ms
DEBUG 05-06 15:59:58.218415.218415 cuda_h.py:27] end decode_step cost 159.606 ms
INFO 05-06 15:59:58.218728.218728 lmp.py:1564] decode step 2 time: 0.15966176986694336 seconds
Time taken: 7.661822412163019 seconds
generate input ids cost 0.10551595687866211 s
DEBUG 05-06 16:00:01.055337.055337 cuda_h.py:27] end generate_input_ids cost 2707.155 ms
DEBUG 05-06 16:00:01.055006.055006 cuda_h.py:27] end init_cache cost 0.051 ms
INFO 05-06 16:00:01.055820.055820 lmp.py:1162] Static KV buffers pre-allocated before prefill (30 layers, max_seq=2048).
INFO 05-06 16:00:01.068899.068899 lmp.py:2808] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 2823528388, 'cuda:1': 12875595776, 'cuda:2': 12875595776, 'cuda:3': 12875595776} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.8623288871419585, 'cuda:1': 0.4700220660037874, 'cuda:2': 0.4700220660037874, 'cuda:3': 0.4700220660037874}
INFO 05-06 16:00:01.068948.068948 lmp.py:2826] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.068917.068917 lmp.py:2826] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.068018.068018 lmp.py:2826] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.069926.069926 lmp.py:2826] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.069792.069792 lmp.py:2826] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.069608.069608 lmp.py:2826] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.069517.069517 lmp.py:2826] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.069724.069724 lmp.py:2826] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.069302.069302 lmp.py:2826] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.069449.069449 lmp.py:2826] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070950.070950 lmp.py:2826] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070766.070766 lmp.py:2826] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070151.070151 lmp.py:2826] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070476.070476 lmp.py:2826] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070007.070007 lmp.py:2826] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070916.070916 lmp.py:2826] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070104.070104 lmp.py:2826] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070158.070158 lmp.py:2826] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070067.070067 lmp.py:2826] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.070377.070377 lmp.py:2826] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071954.071954 lmp.py:2826] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071340.071340 lmp.py:2826] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071087.071087 lmp.py:2826] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071234.071234 lmp.py:2826] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071189.071189 lmp.py:2826] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071267.071267 lmp.py:2826] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071652.071652 lmp.py:2826] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071084.071084 lmp.py:2826] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071189.071189 lmp.py:2826] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-06 16:00:01.071343.071343 lmp.py:2826] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-06 16:00:01.359803.359803 cuda_h.py:27] end init_loading_placement cost 303.047 ms
DEBUG 05-06 16:00:01.359799.359799 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 16:00:01.359014.359014 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 16:00:01 client.py:72] load_into_gpu: gemma4-26B-A4B, 6ed51670-7d61-4743-86c4-cdfbf2a273ca
INFO 05-06 16:00:01 client.py:135] Model loaded: gemma4-26B-A4B, 6ed51670-7d61-4743-86c4-cdfbf2a273ca
INFO 05-06 16:00:01 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 6ed51670-7d61-4743-86c4-cdfbf2a273ca
INFO 05-06 16:00:01 client.py:212] Model loaded
DEBUG 05-06 16:00:01.886761.886761 cuda_h.py:27] end init_general_sagl_loading_async cost 527.516 ms
INFO 05-06 16:00:01.937823.937823 lmp.py:3329] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-06 16:00:02.042107.042107 cuda_h.py:27] end restore_state_dict cost 105.241 ms
INFO 05-06 16:00:02.045484.045484 lmp.py:1291] vLLM Triton pre-warmup done in 2.1 ms (layer=0, devs=[1, 2, 3, 0])
DEBUG 05-06 16:00:02.045215.045215 sllm_store_c.py:27] get device uuid map
DEBUG 05-06 16:00:02.045303.045303 sllm_store_c.py:29] call client load into gpu
DEBUG 05-06 16:00:02 client.py:72] load_into_gpu: gemma4-26B-A4B, f4057292-3c31-4cb2-93d6-c2da5f5cbfda
INFO 05-06 16:00:02 client.py:135] Model loaded: gemma4-26B-A4B, f4057292-3c31-4cb2-93d6-c2da5f5cbfda
DEBUG 05-06 16:00:02.172327.172327 cuda_h.py:27] end init_experts_loading_async cost 126.926 ms
DEBUG 05-06 16:00:02.173709.173709 cuda_h.py:27] end init_inputs_tokens cost 1.191 ms
DEBUG 05-06 16:00:02.173004.173004 lmp.py:1350] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-06 16:00:02.174783.174783 cuda_h.py:27] end prefill_ln cost 1.229 ms
DEBUG 05-06 16:00:02.179210.179210 cuda_h.py:27] end prefill_attn cost 4.711 ms
DEBUG 05-06 16:00:02.180470.180470 cuda_h.py:27] end prefill_ffn_prep cost 0.685 ms
DEBUG 05-06 16:00:02.182492.182492 cuda_h.py:27] end prefill_gate cost 0.901 ms
INFO 05-06 16:00:02.184123.184123 lmp.py:1823] [layer_moe_fused] layer=0 active_experts=118 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 10162, 'token_per_expert': {3: 376, 7: 688, 11: 47, 15: 30, 19: 11, 23: 104, 27: 25, 31: 316, 39: 1391, 43: 2, 47: 2548, 51: 345, 55: 445, 59: 76, 63: 39, 67: 340, 71: 98, 75: 171, 79: 164, 83: 147, 87: 15, 91: 946, 99: 318, 103: 877, 107: 43, 111: 84, 115: 171, 119: 17, 123: 132, 127: 196}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 48, 52, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 6978, 'token_per_expert': {0: 446, 4: 26, 8: 26, 12: 4, 16: 364, 20: 85, 24: 135, 28: 228, 32: 390, 36: 8, 44: 62, 48: 300, 52: 353, 60: 182, 64: 202, 68: 1322, 72: 136, 76: 147, 80: 29, 84: 60, 88: 1, 92: 135, 96: 14, 100: 11, 104: 288, 108: 206, 112: 142, 116: 189, 120: 21, 124: 1466}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 7647, 'token_per_expert': {1: 579, 5: 121, 9: 128, 13: 115, 17: 8, 21: 317, 25: 253, 29: 13, 33: 1632, 37: 154, 41: 262, 45: 59, 49: 63, 53: 1710, 65: 70, 69: 153, 73: 129, 77: 224, 81: 24, 85: 6, 89: 279, 93: 37, 97: 6, 101: 14, 105: 154, 109: 4, 113: 331, 117: 137, 121: 478, 125: 187}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 46, 50, 54, 58, 66, 70, 74, 78, 86, 90, 94, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 7981, 'token_per_expert': {2: 59, 6: 10, 10: 80, 14: 76, 18: 109, 22: 483, 26: 628, 30: 2, 34: 118, 38: 185, 46: 816, 50: 944, 54: 541, 58: 7, 66: 9, 70: 306, 74: 462, 78: 218, 86: 9, 90: 1092, 94: 41, 102: 106, 106: 27, 110: 114, 114: 152, 118: 116, 122: 264, 126: 1007}}
INFO 05-06 16:00:02.184573.184573 lmp.py:1845] [layer_moe_fused] layer=0 prefix: 1.042ms alloc: 0.304ms
INFO 05-06 16:00:02.184120.184120 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 6.556510925292969e-05 seconds
INFO 05-06 16:00:02.186278.186278 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=236 time: 0.001589059829711914s
INFO 05-06 16:00:02.288380.288380 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.10245084762573242s
DEBUG 05-06 16:00:02.289256.289256 cuda_h.py:27] end moe_wait_copy_tasks cost 103.104 ms
DEBUG 05-06 16:00:02.310404.310404 cuda_h.py:27] end moe_vllm_forward cost 19.950 ms
DEBUG 05-06 16:00:02.310399.310399 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 16:00:02.310772.310772 lmp.py:1964] [layer_moe_fused] vllm triton time: 20.645ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.310010.310010 cuda_h.py:27] end *layer_moe_fused cost 127.352 ms
DEBUG 05-06 16:00:02.311500.311500 cuda_h.py:27] end prefill_merge_scale cost 1.122 ms
DEBUG 05-06 16:00:02.311390.311390 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.035 ms
DEBUG 05-06 16:00:02.311458.311458 cuda_h.py:27] end prefill_layer cost 138.366 ms
DEBUG 05-06 16:00:02.312816.312816 lmp.py:1394] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-06 16:00:02.312088.312088 lmp.py:1350] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-06 16:00:02.312538.312538 cuda_h.py:27] end prefill_ln cost 0.162 ms
DEBUG 05-06 16:00:02.345511.345511 cuda_h.py:27] end prefill_attn cost 33.144 ms
DEBUG 05-06 16:00:02.345953.345953 cuda_h.py:27] end prefill_ffn_prep cost 0.294 ms
DEBUG 05-06 16:00:02.389660.389660 cuda_h.py:27] end prefill_gate cost 0.696 ms
INFO 05-06 16:00:02.390983.390983 lmp.py:1823] [layer_moe_fused] layer=1 active_experts=122 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 23, 27, 31, 35, 39, 47, 51, 55, 59, 63, 67, 71, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 31, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 6079, 'token_per_expert': {3: 38, 7: 124, 11: 195, 15: 105, 23: 37, 27: 331, 31: 111, 35: 66, 39: 47, 47: 673, 51: 622, 55: 120, 59: 429, 63: 2, 67: 145, 71: 90, 79: 403, 83: 203, 87: 57, 91: 17, 95: 69, 99: 786, 103: 226, 107: 27, 111: 17, 115: 207, 119: 513, 123: 249, 127: 170}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 9429, 'token_per_expert': {0: 505, 4: 332, 8: 1545, 12: 313, 16: 60, 20: 1106, 24: 33, 28: 8, 32: 40, 36: 25, 40: 94, 44: 9, 48: 32, 52: 732, 56: 79, 60: 203, 64: 518, 68: 123, 72: 501, 76: 20, 80: 1010, 84: 223, 88: 48, 92: 114, 96: 421, 100: 315, 104: 454, 108: 143, 112: 30, 116: 131, 120: 123, 124: 139}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 7990, 'token_per_expert': {1: 93, 5: 130, 9: 637, 13: 779, 21: 156, 25: 484, 29: 147, 33: 29, 37: 274, 41: 89, 45: 137, 49: 259, 53: 262, 57: 178, 61: 14, 65: 415, 69: 107, 73: 247, 77: 124, 81: 126, 85: 367, 89: 551, 93: 485, 97: 797, 101: 185, 105: 91, 109: 448, 113: 35, 117: 220, 121: 77, 125: 47}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 9270, 'token_per_expert': {2: 17, 6: 907, 10: 333, 14: 9, 18: 92, 22: 699, 26: 229, 30: 632, 34: 252, 38: 178, 42: 423, 46: 391, 50: 232, 54: 31, 62: 61, 66: 116, 70: 6, 74: 283, 78: 126, 82: 810, 86: 38, 90: 233, 94: 440, 98: 447, 102: 212, 106: 432, 110: 150, 114: 86, 118: 811, 122: 594}}
INFO 05-06 16:00:02.391611.391611 lmp.py:1845] [layer_moe_fused] layer=1 prefix: 0.989ms alloc: 0.656ms
INFO 05-06 16:00:02.391230.391230 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 0.00017070770263671875 seconds
INFO 05-06 16:00:02.393532.393532 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0017979145050048828s
INFO 05-06 16:00:02.394953.394953 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011532306671142578s
DEBUG 05-06 16:00:02.395390.395390 cuda_h.py:27] end moe_wait_copy_tasks cost 1.340 ms
DEBUG 05-06 16:00:02.400995.400995 cuda_h.py:27] end moe_vllm_forward cost 5.043 ms
DEBUG 05-06 16:00:02.400409.400409 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 16:00:02.400068.400068 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.644ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.401639.401639 cuda_h.py:27] end *layer_moe_fused cost 11.363 ms
DEBUG 05-06 16:00:02.401850.401850 cuda_h.py:27] end prefill_merge_scale cost 0.489 ms
DEBUG 05-06 16:00:02.401080.401080 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.052 ms
DEBUG 05-06 16:00:02.402068.402068 cuda_h.py:27] end prefill_layer cost 89.973 ms
DEBUG 05-06 16:00:02.402944.402944 lmp.py:1394] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-06 16:00:02.402352.402352 lmp.py:1350] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-06 16:00:02.402759.402759 cuda_h.py:27] end prefill_ln cost 0.250 ms
DEBUG 05-06 16:00:02.447611.447611 cuda_h.py:27] end prefill_attn cost 44.757 ms
DEBUG 05-06 16:00:02.447434.447434 cuda_h.py:27] end prefill_ffn_prep cost 0.332 ms
DEBUG 05-06 16:00:02.448133.448133 cuda_h.py:27] end prefill_gate cost 0.345 ms
INFO 05-06 16:00:02.453042.453042 lmp.py:1823] [layer_moe_fused] layer=2 active_experts=123 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 9387, 'token_per_expert': {3: 340, 7: 221, 11: 1085, 15: 404, 19: 1155, 23: 36, 27: 19, 31: 76, 35: 682, 39: 3, 43: 1058, 47: 18, 51: 65, 55: 76, 59: 643, 63: 40, 67: 65, 71: 290, 75: 16, 83: 1467, 87: 159, 91: 55, 95: 216, 99: 69, 103: 92, 107: 151, 111: 11, 115: 328, 119: 77, 123: 183, 127: 287}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 8314, 'token_per_expert': {0: 1122, 4: 105, 8: 86, 12: 36, 16: 92, 20: 167, 24: 40, 28: 204, 32: 4, 36: 108, 40: 73, 44: 76, 48: 171, 52: 86, 56: 185, 60: 333, 64: 165, 68: 81, 72: 679, 76: 385, 80: 804, 84: 118, 88: 192, 92: 5, 96: 121, 100: 164, 104: 113, 108: 2071, 116: 28, 120: 112, 124: 388}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 11158, 'token_per_expert': {1: 1897, 5: 51, 9: 1460, 13: 523, 17: 214, 21: 16, 25: 54, 29: 556, 33: 89, 37: 312, 41: 239, 45: 48, 49: 291, 53: 39, 57: 71, 61: 39, 65: 325, 69: 415, 73: 264, 77: 231, 81: 1425, 85: 307, 93: 30, 97: 175, 101: 1, 105: 12, 109: 1855, 113: 29, 117: 13, 121: 4, 125: 173}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 3909, 'token_per_expert': {2: 7, 6: 2, 10: 17, 14: 49, 18: 69, 22: 10, 26: 43, 34: 470, 38: 13, 42: 330, 46: 139, 50: 252, 54: 67, 58: 97, 62: 293, 66: 91, 70: 70, 74: 50, 78: 8, 82: 82, 86: 39, 90: 141, 98: 24, 102: 593, 106: 289, 110: 222, 114: 68, 118: 62, 122: 177, 126: 135}}
INFO 05-06 16:00:02.454589.454589 lmp.py:1845] [layer_moe_fused] layer=2 prefix: 4.832ms alloc: 0.274ms
INFO 05-06 16:00:02.454842.454842 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 6.341934204101562e-05 seconds
INFO 05-06 16:00:02.455792.455792 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014340877532958984s
INFO 05-06 16:00:02.456147.456147 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0008652210235595703s
DEBUG 05-06 16:00:02.456164.456164 cuda_h.py:27] end moe_wait_copy_tasks cost 0.986 ms
DEBUG 05-06 16:00:02.460382.460382 cuda_h.py:27] end moe_vllm_forward cost 3.189 ms
DEBUG 05-06 16:00:02.460232.460232 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:02.460697.460697 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.503ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.460093.460093 cuda_h.py:27] end *layer_moe_fused cost 11.665 ms
DEBUG 05-06 16:00:02.461752.461752 cuda_h.py:27] end prefill_merge_scale cost 0.303 ms
DEBUG 05-06 16:00:02.461682.461682 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:02.461035.461035 cuda_h.py:27] end prefill_layer cost 58.967 ms
DEBUG 05-06 16:00:02.461503.461503 lmp.py:1394] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-06 16:00:02.461298.461298 lmp.py:1350] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-06 16:00:02.461556.461556 cuda_h.py:27] end prefill_ln cost 0.161 ms
DEBUG 05-06 16:00:02.506842.506842 cuda_h.py:27] end prefill_attn cost 44.472 ms
DEBUG 05-06 16:00:02.506662.506662 cuda_h.py:27] end prefill_ffn_prep cost 0.294 ms
DEBUG 05-06 16:00:02.507693.507693 cuda_h.py:27] end prefill_gate cost 0.311 ms
INFO 05-06 16:00:02.511004.511004 lmp.py:1823] [layer_moe_fused] layer=3 active_experts=123 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 31, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 5865, 'token_per_expert': {3: 116, 11: 247, 15: 421, 19: 256, 23: 252, 27: 23, 31: 31, 35: 124, 39: 301, 43: 285, 47: 1, 51: 425, 55: 32, 59: 177, 63: 128, 67: 637, 71: 337, 75: 301, 79: 79, 83: 157, 87: 107, 91: 510, 99: 4, 103: 77, 107: 366, 111: 6, 115: 2, 119: 180, 123: 82, 127: 201}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 10419, 'token_per_expert': {0: 121, 4: 797, 8: 182, 12: 65, 16: 855, 20: 488, 24: 93, 28: 1611, 32: 6, 36: 6, 40: 78, 44: 1010, 48: 5, 52: 958, 56: 150, 60: 423, 64: 44, 68: 80, 72: 28, 76: 12, 80: 64, 84: 1015, 88: 479, 92: 797, 96: 259, 100: 244, 104: 128, 108: 213, 112: 3, 116: 54, 120: 151}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 31, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 7437, 'token_per_expert': {1: 411, 5: 182, 9: 442, 17: 385, 21: 106, 25: 327, 29: 259, 33: 213, 37: 54, 41: 115, 45: 2, 49: 93, 53: 377, 57: 56, 61: 167, 65: 12, 69: 305, 73: 120, 77: 875, 81: 192, 85: 1619, 89: 10, 93: 134, 97: 20, 101: 381, 105: 9, 109: 175, 113: 17, 117: 184, 121: 65, 125: 130}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 9047, 'token_per_expert': {2: 169, 6: 417, 10: 261, 14: 216, 18: 147, 22: 204, 26: 317, 30: 338, 34: 96, 38: 62, 42: 47, 46: 31, 50: 444, 54: 9, 58: 142, 62: 1558, 66: 527, 70: 154, 74: 54, 78: 56, 82: 22, 86: 106, 90: 9, 94: 1599, 98: 42, 102: 1485, 106: 30, 110: 156, 114: 253, 118: 76, 122: 20}}
INFO 05-06 16:00:02.512001.512001 lmp.py:1845] [layer_moe_fused] layer=3 prefix: 4.176ms alloc: 0.256ms
INFO 05-06 16:00:02.512617.512617 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.3882598876953125e-05 seconds
INFO 05-06 16:00:02.513190.513190 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014367103576660156s
INFO 05-06 16:00:02.514237.514237 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011303424835205078s
DEBUG 05-06 16:00:02.515067.515067 cuda_h.py:27] end moe_wait_copy_tasks cost 1.219 ms
DEBUG 05-06 16:00:02.518324.518324 cuda_h.py:27] end moe_vllm_forward cost 3.114 ms
DEBUG 05-06 16:00:02.518790.518790 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:02.518732.518732 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.406ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.518982.518982 cuda_h.py:27] end *layer_moe_fused cost 10.986 ms
DEBUG 05-06 16:00:02.519815.519815 cuda_h.py:27] end prefill_merge_scale cost 0.304 ms
DEBUG 05-06 16:00:02.519143.519143 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:02.519873.519873 cuda_h.py:27] end prefill_layer cost 57.902 ms
DEBUG 05-06 16:00:02.519071.519071 lmp.py:1394] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-06 16:00:02.519628.519628 lmp.py:1350] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-06 16:00:02.519323.519323 cuda_h.py:27] end prefill_ln cost 0.169 ms
DEBUG 05-06 16:00:02.564888.564888 cuda_h.py:27] end prefill_attn cost 45.079 ms
DEBUG 05-06 16:00:02.565139.565139 cuda_h.py:27] end prefill_ffn_prep cost 0.293 ms
DEBUG 05-06 16:00:02.566982.566982 cuda_h.py:27] end prefill_gate cost 0.299 ms
INFO 05-06 16:00:02.571413.571413 lmp.py:1823] [layer_moe_fused] layer=4 active_experts=126 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 32, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 13592, 'token_per_expert': {3: 194, 7: 60, 15: 141, 19: 221, 23: 2032, 27: 1013, 31: 395, 35: 284, 39: 29, 43: 213, 47: 474, 51: 431, 55: 1876, 59: 333, 63: 640, 67: 831, 71: 353, 75: 385, 79: 68, 83: 316, 87: 346, 91: 24, 95: 16, 103: 462, 107: 210, 111: 177, 115: 88, 119: 1871, 123: 31, 127: 78}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 32, 'ideal_gpu_count': 32, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 6910, 'token_per_expert': {0: 15, 4: 15, 8: 1549, 12: 38, 16: 51, 20: 105, 24: 141, 28: 271, 32: 722, 36: 193, 40: 156, 44: 42, 48: 1, 52: 7, 56: 466, 60: 32, 64: 106, 68: 129, 72: 391, 76: 58, 80: 217, 84: 275, 88: 214, 92: 289, 96: 23, 100: 344, 104: 1, 108: 38, 112: 44, 116: 354, 120: 217, 124: 406}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 6328, 'token_per_expert': {1: 1277, 5: 494, 9: 15, 13: 22, 17: 234, 21: 71, 25: 4, 29: 188, 33: 21, 37: 342, 41: 185, 45: 128, 49: 197, 53: 143, 57: 25, 61: 303, 65: 48, 69: 96, 73: 18, 77: 350, 81: 98, 85: 144, 89: 328, 93: 179, 97: 105, 101: 107, 105: 595, 109: 92, 113: 272, 117: 60, 121: 30, 125: 157}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 5938, 'token_per_expert': {2: 609, 6: 3, 10: 1, 14: 7, 18: 80, 22: 221, 26: 577, 30: 249, 34: 80, 38: 91, 42: 5, 46: 42, 50: 35, 54: 159, 58: 7, 62: 90, 66: 81, 70: 290, 74: 1865, 78: 110, 82: 110, 86: 150, 90: 299, 94: 288, 98: 56, 102: 2, 106: 135, 110: 21, 114: 1, 118: 7, 122: 81, 126: 186}}
INFO 05-06 16:00:02.571378.571378 lmp.py:1845] [layer_moe_fused] layer=4 prefix: 5.013ms alloc: 0.267ms
INFO 05-06 16:00:02.571810.571810 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.602836608886719e-05 seconds
INFO 05-06 16:00:02.573554.573554 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014233589172363281s
INFO 05-06 16:00:02.574870.574870 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010478496551513672s
DEBUG 05-06 16:00:02.574032.574032 cuda_h.py:27] end moe_wait_copy_tasks cost 1.135 ms
DEBUG 05-06 16:00:02.577089.577089 cuda_h.py:27] end moe_vllm_forward cost 3.110 ms
DEBUG 05-06 16:00:02.577317.577317 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:02.577259.577259 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.400ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.578032.578032 cuda_h.py:27] end *layer_moe_fused cost 11.768 ms
DEBUG 05-06 16:00:02.578228.578228 cuda_h.py:27] end prefill_merge_scale cost 0.303 ms
DEBUG 05-06 16:00:02.578443.578443 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:02.578173.578173 cuda_h.py:27] end prefill_layer cost 59.178 ms
DEBUG 05-06 16:00:02.578139.578139 lmp.py:1394] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-06 16:00:02.578696.578696 lmp.py:1350] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-06 16:00:02.579001.579001 cuda_h.py:27] end prefill_ln cost 0.161 ms
DEBUG 05-06 16:00:02.625396.625396 cuda_h.py:27] end prefill_attn cost 46.340 ms
DEBUG 05-06 16:00:02.625304.625304 cuda_h.py:27] end prefill_ffn_prep cost 0.301 ms
DEBUG 05-06 16:00:02.626772.626772 cuda_h.py:27] end prefill_gate cost 0.298 ms
INFO 05-06 16:00:02.633941.633941 lmp.py:1823] [layer_moe_fused] layer=5 active_experts=116 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 15, 19, 23, 31, 39, 43, 47, 51, 55, 63, 67, 71, 75, 83, 87, 91, 95, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 26, 'ideal_gpu_count': 29, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 6616, 'token_per_expert': {7: 509, 11: 3, 15: 113, 19: 56, 23: 63, 31: 21, 39: 464, 43: 102, 47: 112, 51: 235, 55: 147, 63: 72, 67: 176, 71: 1323, 75: 319, 83: 130, 87: 129, 91: 9, 95: 12, 99: 1003, 107: 19, 111: 472, 115: 7, 119: 264, 123: 658, 127: 198}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 112, 116, 120, 124], 'expert_count': 29, 'ideal_gpu_count': 29, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 9959, 'token_per_expert': {0: 81, 4: 148, 8: 62, 16: 783, 20: 71, 24: 156, 28: 553, 32: 50, 36: 76, 44: 368, 48: 299, 52: 58, 56: 2873, 60: 33, 64: 3, 68: 144, 72: 1392, 76: 9, 80: 14, 84: 189, 88: 45, 92: 297, 96: 73, 100: 494, 104: 662, 112: 189, 116: 306, 120: 484, 124: 47}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 31, 'ideal_gpu_count': 29, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 9567, 'token_per_expert': {1: 7, 5: 112, 9: 530, 13: 688, 17: 1265, 21: 15, 29: 68, 33: 18, 37: 63, 41: 225, 45: 958, 49: 59, 53: 9, 57: 512, 61: 299, 65: 2, 69: 2, 73: 2479, 77: 115, 81: 83, 85: 2, 89: 345, 93: 88, 97: 71, 101: 725, 105: 8, 109: 1, 113: 790, 117: 23, 121: 3, 125: 2}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 30, 34, 38, 42, 46, 50, 54, 58, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 6626, 'token_per_expert': {2: 251, 6: 170, 10: 132, 14: 540, 18: 18, 22: 117, 30: 117, 34: 57, 38: 83, 42: 303, 46: 69, 50: 1306, 54: 9, 58: 30, 66: 12, 70: 195, 74: 355, 78: 17, 82: 110, 86: 197, 90: 23, 94: 908, 98: 82, 102: 48, 106: 179, 110: 2, 114: 44, 118: 289, 122: 13, 126: 950}}
INFO 05-06 16:00:02.633938.633938 lmp.py:1845] [layer_moe_fused] layer=5 prefix: 6.080ms alloc: 0.256ms
INFO 05-06 16:00:02.633377.633377 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.364418029785156e-05 seconds
INFO 05-06 16:00:02.635308.635308 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014545917510986328s
INFO 05-06 16:00:02.636878.636878 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001130819320678711s
DEBUG 05-06 16:00:02.636557.636557 cuda_h.py:27] end moe_wait_copy_tasks cost 1.248 ms
DEBUG 05-06 16:00:02.639636.639636 cuda_h.py:27] end moe_vllm_forward cost 3.150 ms
DEBUG 05-06 16:00:02.639864.639864 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:02.639091.639091 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.449ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.640109.640109 cuda_h.py:27] end *layer_moe_fused cost 13.033 ms
DEBUG 05-06 16:00:02.640298.640298 cuda_h.py:27] end prefill_merge_scale cost 0.302 ms
DEBUG 05-06 16:00:02.640182.640182 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:02.640442.640442 cuda_h.py:27] end prefill_layer cost 61.819 ms
DEBUG 05-06 16:00:02.640652.640652 lmp.py:1394] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-06 16:00:02.640647.640647 lmp.py:1350] -------------------------------- start prefill layer 6 --------------------------------
DEBUG 05-06 16:00:02.641812.641812 cuda_h.py:27] end prefill_ln cost 0.162 ms
DEBUG 05-06 16:00:02.685386.685386 cuda_h.py:27] end prefill_attn cost 43.806 ms
DEBUG 05-06 16:00:02.685160.685160 cuda_h.py:27] end prefill_ffn_prep cost 0.295 ms
DEBUG 05-06 16:00:02.686689.686689 cuda_h.py:27] end prefill_gate cost 0.296 ms
INFO 05-06 16:00:02.691648.691648 lmp.py:1823] [layer_moe_fused] layer=6 active_experts=120 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 11303, 'token_per_expert': {3: 30, 7: 3, 11: 74, 15: 33, 19: 278, 23: 349, 27: 168, 31: 22, 35: 948, 39: 30, 47: 35, 51: 99, 55: 17, 59: 1, 63: 6, 67: 283, 71: 16, 75: 814, 79: 17, 83: 19, 87: 1218, 91: 101, 95: 490, 99: 3638, 103: 392, 107: 205, 111: 232, 115: 174, 119: 1010, 123: 315, 127: 286}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 7476, 'token_per_expert': {0: 1789, 4: 32, 8: 24, 12: 3, 16: 365, 20: 6, 24: 74, 28: 61, 32: 177, 36: 533, 44: 152, 48: 2, 52: 4, 56: 161, 60: 126, 64: 10, 68: 164, 72: 28, 76: 1, 80: 121, 84: 8, 88: 48, 92: 4, 96: 342, 100: 458, 104: 203, 108: 2186, 112: 206, 116: 47, 120: 93, 124: 48}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 45, 53, 57, 61, 65, 69, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121], 'expert_count': 28, 'ideal_gpu_count': 30, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 5490, 'token_per_expert': {1: 75, 5: 408, 9: 77, 13: 278, 17: 41, 21: 13, 25: 1039, 29: 18, 33: 4, 37: 673, 45: 51, 53: 60, 57: 44, 61: 209, 65: 281, 69: 417, 77: 144, 81: 37, 85: 127, 89: 134, 93: 234, 97: 1, 101: 614, 105: 128, 109: 33, 113: 13, 117: 119, 121: 218}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 58, 62, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 8499, 'token_per_expert': {2: 430, 6: 291, 10: 41, 14: 29, 18: 191, 22: 444, 26: 292, 30: 187, 34: 988, 38: 36, 42: 455, 46: 64, 50: 215, 58: 489, 62: 53, 70: 128, 74: 47, 78: 45, 82: 78, 86: 1600, 90: 258, 94: 547, 98: 35, 102: 53, 106: 266, 110: 245, 114: 104, 118: 3, 122: 14, 126: 871}}
INFO 05-06 16:00:02.691488.691488 lmp.py:1845] [layer_moe_fused] layer=6 prefix: 4.978ms alloc: 0.281ms
INFO 05-06 16:00:02.691852.691852 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.698204040527344e-05 seconds
INFO 05-06 16:00:02.693055.693055 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001481771469116211s
INFO 05-06 16:00:02.694181.694181 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011179447174072266s
DEBUG 05-06 16:00:02.694972.694972 cuda_h.py:27] end moe_wait_copy_tasks cost 1.212 ms
DEBUG 05-06 16:00:02.698780.698780 cuda_h.py:27] end moe_vllm_forward cost 3.170 ms
DEBUG 05-06 16:00:02.698007.698007 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:02.698950.698950 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.462ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.698299.698299 cuda_h.py:27] end *layer_moe_fused cost 11.898 ms
DEBUG 05-06 16:00:02.698482.698482 cuda_h.py:27] end prefill_merge_scale cost 0.304 ms
DEBUG 05-06 16:00:02.699697.699697 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:02.699235.699235 cuda_h.py:27] end prefill_layer cost 58.123 ms
DEBUG 05-06 16:00:02.699572.699572 lmp.py:1394] -------------------------------- end prefill layer 6 --------------------------------
DEBUG 05-06 16:00:02.699652.699652 lmp.py:1350] -------------------------------- start prefill layer 7 --------------------------------
DEBUG 05-06 16:00:02.699857.699857 cuda_h.py:27] end prefill_ln cost 0.159 ms
DEBUG 05-06 16:00:02.743796.743796 cuda_h.py:27] end prefill_attn cost 44.392 ms
DEBUG 05-06 16:00:02.744185.744185 cuda_h.py:27] end prefill_ffn_prep cost 0.292 ms
DEBUG 05-06 16:00:02.745519.745519 cuda_h.py:27] end prefill_gate cost 0.297 ms
INFO 05-06 16:00:02.749898.749898 lmp.py:1823] [layer_moe_fused] layer=7 active_experts=121 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 32, 'ideal_gpu_count': 31, 'keep_on_gpu': 32, 'hit_count_on_device': 32, 'token_total': 10484, 'token_per_expert': {3: 9, 7: 285, 11: 6, 15: 125, 19: 125, 23: 258, 27: 185, 31: 252, 35: 112, 39: 5, 43: 836, 47: 357, 51: 269, 55: 200, 59: 9, 63: 31, 67: 179, 71: 350, 75: 12, 79: 335, 83: 1122, 87: 30, 91: 1359, 95: 1, 99: 137, 103: 1327, 107: 801, 111: 314, 115: 179, 119: 1153, 123: 34, 127: 87}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 30, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 5407, 'token_per_expert': {0: 89, 4: 2, 8: 137, 12: 236, 16: 50, 20: 179, 24: 67, 28: 300, 32: 219, 36: 5, 40: 9, 44: 192, 48: 18, 52: 6, 56: 290, 60: 2, 64: 223, 68: 471, 72: 588, 76: 20, 80: 304, 84: 73, 88: 47, 92: 684, 96: 93, 104: 272, 108: 61, 112: 244, 116: 87, 120: 407, 124: 32}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 89, 93, 97, 101, 105, 113, 117, 121, 125], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 9514, 'token_per_expert': {1: 1, 5: 124, 9: 36, 13: 249, 17: 204, 21: 158, 25: 102, 29: 618, 33: 75, 41: 232, 45: 124, 49: 17, 53: 777, 57: 293, 61: 512, 65: 519, 69: 3029, 73: 13, 77: 223, 81: 2, 89: 251, 93: 2, 97: 122, 101: 172, 105: 210, 113: 444, 117: 12, 121: 307, 125: 686}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 26, 30, 34, 38, 42, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 7363, 'token_per_expert': {2: 8, 6: 19, 10: 407, 14: 492, 18: 277, 26: 127, 30: 21, 34: 14, 38: 13, 42: 177, 50: 3, 54: 306, 58: 90, 62: 4, 66: 90, 70: 499, 74: 357, 78: 4, 82: 660, 86: 643, 90: 347, 98: 12, 102: 1319, 106: 243, 110: 63, 114: 527, 118: 306, 122: 212, 126: 123}}
INFO 05-06 16:00:02.749027.749027 lmp.py:1845] [layer_moe_fused] layer=7 prefix: 4.235ms alloc: 0.250ms
INFO 05-06 16:00:02.750776.750776 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.3882598876953125e-05 seconds
INFO 05-06 16:00:02.751998.751998 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014615058898925781s
INFO 05-06 16:00:02.752107.752107 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0008275508880615234s
DEBUG 05-06 16:00:02.752316.752316 cuda_h.py:27] end moe_wait_copy_tasks cost 0.913 ms
DEBUG 05-06 16:00:02.755617.755617 cuda_h.py:27] end moe_vllm_forward cost 3.081 ms
DEBUG 05-06 16:00:02.755652.755652 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:02.755926.755926 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.367ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.756083.756083 cuda_h.py:27] end *layer_moe_fused cost 10.708 ms
DEBUG 05-06 16:00:02.756317.756317 cuda_h.py:27] end prefill_merge_scale cost 0.302 ms
DEBUG 05-06 16:00:02.756009.756009 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:02.756785.756785 cuda_h.py:27] end prefill_layer cost 57.365 ms
DEBUG 05-06 16:00:02.756982.756982 lmp.py:1394] -------------------------------- end prefill layer 7 --------------------------------
DEBUG 05-06 16:00:02.756539.756539 lmp.py:1350] -------------------------------- start prefill layer 8 --------------------------------
DEBUG 05-06 16:00:02.757983.757983 cuda_h.py:27] end prefill_ln cost 0.158 ms
DEBUG 05-06 16:00:02.803725.803725 cuda_h.py:27] end prefill_attn cost 46.229 ms
DEBUG 05-06 16:00:02.803991.803991 cuda_h.py:27] end prefill_ffn_prep cost 0.330 ms
DEBUG 05-06 16:00:02.804577.804577 cuda_h.py:27] end prefill_gate cost 0.366 ms
INFO 05-06 16:00:02.809279.809279 lmp.py:1823] [layer_moe_fused] layer=8 active_experts=117 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 30, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 7500, 'token_per_expert': {3: 280, 7: 45, 11: 25, 15: 788, 19: 384, 23: 34, 27: 508, 31: 33, 35: 42, 39: 3, 43: 56, 47: 498, 51: 1166, 55: 491, 59: 13, 63: 55, 71: 214, 75: 1367, 79: 2, 83: 2, 87: 184, 91: 143, 95: 51, 99: 181, 103: 166, 107: 8, 111: 434, 119: 71, 123: 218, 127: 38}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 31, 'ideal_gpu_count': 29, 'keep_on_gpu': 31, 'hit_count_on_device': 31, 'token_total': 8696, 'token_per_expert': {0: 171, 4: 138, 8: 24, 12: 901, 16: 1, 20: 125, 24: 375, 28: 24, 32: 1106, 36: 385, 40: 174, 44: 974, 48: 97, 52: 305, 56: 355, 60: 1, 64: 451, 68: 77, 72: 23, 76: 288, 80: 332, 84: 1, 88: 78, 96: 8, 100: 25, 104: 72, 108: 143, 112: 9, 116: 192, 120: 680, 124: 1161}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 25, 29, 33, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 93, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 8323, 'token_per_expert': {1: 63, 5: 571, 9: 286, 17: 354, 21: 37, 25: 32, 29: 470, 33: 45, 41: 167, 45: 223, 49: 110, 53: 446, 57: 257, 61: 799, 65: 47, 69: 217, 73: 2571, 77: 171, 81: 139, 85: 36, 93: 77, 101: 359, 105: 576, 109: 3, 113: 186, 117: 1, 121: 42, 125: 38}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 22, 26, 30, 34, 38, 42, 46, 50, 58, 62, 66, 70, 74, 78, 82, 86, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 29, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 8249, 'token_per_expert': {2: 266, 6: 125, 10: 28, 14: 161, 22: 7, 26: 17, 30: 1, 34: 76, 38: 177, 42: 107, 46: 445, 50: 648, 58: 1378, 62: 105, 66: 402, 70: 457, 74: 78, 78: 2, 82: 76, 86: 260, 98: 230, 102: 433, 106: 488, 110: 429, 114: 384, 118: 321, 122: 1025, 126: 123}}
INFO 05-06 16:00:02.809900.809900 lmp.py:1845] [layer_moe_fused] layer=8 prefix: 4.609ms alloc: 0.293ms
INFO 05-06 16:00:02.810408.810408 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.7697296142578125e-05 seconds
INFO 05-06 16:00:02.811681.811681 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0017788410186767578s
INFO 05-06 16:00:02.812955.812955 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010161399841308594s
DEBUG 05-06 16:00:02.813556.813556 cuda_h.py:27] end moe_wait_copy_tasks cost 1.146 ms
DEBUG 05-06 16:00:02.816532.816532 cuda_h.py:27] end moe_vllm_forward cost 3.488 ms
DEBUG 05-06 16:00:02.816667.816667 cuda_h.py:27] end moe_shared_experts cost 0.006 ms
INFO 05-06 16:00:02.817371.817371 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.819ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.817701.817701 cuda_h.py:27] end *layer_moe_fused cost 12.206 ms
DEBUG 05-06 16:00:02.817511.817511 cuda_h.py:27] end prefill_merge_scale cost 0.307 ms
DEBUG 05-06 16:00:02.817203.817203 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.035 ms
DEBUG 05-06 16:00:02.817608.817608 cuda_h.py:27] end prefill_layer cost 61.005 ms
DEBUG 05-06 16:00:02.817123.817123 lmp.py:1394] -------------------------------- end prefill layer 8 --------------------------------
DEBUG 05-06 16:00:02.817680.817680 lmp.py:1350] -------------------------------- start prefill layer 9 --------------------------------
DEBUG 05-06 16:00:02.818322.818322 cuda_h.py:27] end prefill_ln cost 0.165 ms
DEBUG 05-06 16:00:02.863685.863685 cuda_h.py:27] end prefill_attn cost 45.404 ms
DEBUG 05-06 16:00:02.864750.864750 cuda_h.py:27] end prefill_ffn_prep cost 0.298 ms
DEBUG 05-06 16:00:02.864619.864619 cuda_h.py:27] end prefill_gate cost 0.312 ms
INFO 05-06 16:00:02.869773.869773 lmp.py:1823] [layer_moe_fused] layer=9 active_experts=117 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 63, 67, 71, 75, 79, 83, 87, 95, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 30, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 6555, 'token_per_expert': {3: 1623, 7: 21, 11: 110, 15: 63, 19: 800, 23: 405, 27: 139, 31: 4, 35: 2, 39: 87, 43: 46, 47: 3, 51: 8, 55: 21, 63: 1, 67: 110, 71: 753, 75: 477, 79: 2, 83: 367, 87: 2, 95: 648, 103: 189, 107: 1, 111: 182, 115: 120, 119: 3, 123: 18, 127: 350}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 68, 72, 76, 80, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 30, 'ideal_gpu_count': 29, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 6282, 'token_per_expert': {0: 149, 4: 826, 8: 10, 12: 246, 16: 649, 20: 22, 24: 272, 28: 3, 32: 419, 36: 120, 40: 9, 44: 37, 48: 445, 52: 334, 56: 78, 60: 1, 68: 84, 72: 504, 76: 382, 80: 13, 88: 40, 92: 531, 96: 1, 100: 246, 104: 34, 108: 82, 112: 114, 116: 13, 120: 16, 124: 602}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 53, 57, 61, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 125], 'expert_count': 29, 'ideal_gpu_count': 29, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 12509, 'token_per_expert': {1: 332, 5: 228, 9: 202, 13: 744, 17: 10, 21: 154, 25: 60, 29: 1058, 33: 53, 37: 2422, 41: 342, 45: 601, 53: 3, 57: 878, 61: 973, 69: 335, 73: 334, 77: 6, 81: 311, 85: 4, 89: 21, 93: 41, 97: 1, 101: 2358, 105: 29, 109: 5, 113: 39, 117: 237, 125: 728}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 82, 86, 90, 94, 102, 106, 110, 114, 122, 126], 'expert_count': 29, 'ideal_gpu_count': 29, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 7422, 'token_per_expert': {2: 4, 6: 1, 10: 138, 14: 91, 18: 216, 22: 3, 26: 617, 30: 367, 34: 54, 38: 1382, 42: 72, 46: 785, 50: 1, 54: 373, 58: 5, 62: 368, 66: 997, 70: 236, 74: 230, 82: 648, 86: 20, 90: 42, 94: 1, 102: 10, 106: 569, 110: 4, 114: 1, 122: 180, 126: 7}}
INFO 05-06 16:00:02.869108.869108 lmp.py:1845] [layer_moe_fused] layer=9 prefix: 4.242ms alloc: 0.261ms
INFO 05-06 16:00:02.869617.869617 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.650520324707031e-05 seconds
INFO 05-06 16:00:02.870611.870611 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0011887550354003906s
INFO 05-06 16:00:02.871412.871412 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009202957153320312s
DEBUG 05-06 16:00:02.871150.871150 cuda_h.py:27] end moe_wait_copy_tasks cost 1.011 ms
DEBUG 05-06 16:00:02.875414.875414 cuda_h.py:27] end moe_vllm_forward cost 3.118 ms
DEBUG 05-06 16:00:02.875926.875926 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 16:00:02.875199.875199 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.410ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.875701.875701 cuda_h.py:27] end *layer_moe_fused cost 10.559 ms
DEBUG 05-06 16:00:02.876309.876309 cuda_h.py:27] end prefill_merge_scale cost 0.311 ms
DEBUG 05-06 16:00:02.876239.876239 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:02.876969.876969 cuda_h.py:27] end prefill_layer cost 58.288 ms
DEBUG 05-06 16:00:02.876511.876511 lmp.py:1394] -------------------------------- end prefill layer 9 --------------------------------
DEBUG 05-06 16:00:02.876114.876114 lmp.py:1350] -------------------------------- start prefill layer 10 --------------------------------
DEBUG 05-06 16:00:02.876604.876604 cuda_h.py:27] end prefill_ln cost 0.158 ms
DEBUG 05-06 16:00:02.921052.921052 cuda_h.py:27] end prefill_attn cost 44.767 ms
DEBUG 05-06 16:00:02.921156.921156 cuda_h.py:27] end prefill_ffn_prep cost 0.293 ms
DEBUG 05-06 16:00:02.922550.922550 cuda_h.py:27] end prefill_gate cost 0.303 ms
INFO 05-06 16:00:02.927849.927849 lmp.py:1823] [layer_moe_fused] layer=10 active_experts=102 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 19, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 119, 127], 'expert_count': 25, 'ideal_gpu_count': 26, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 4498, 'token_per_expert': {3: 2, 15: 43, 19: 46, 27: 7, 31: 1253, 39: 81, 43: 7, 47: 2, 51: 270, 55: 1, 59: 2, 63: 1, 67: 160, 71: 17, 75: 500, 79: 555, 83: 124, 87: 2, 91: 12, 95: 1, 99: 664, 103: 185, 107: 2, 119: 130, 127: 431}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 32, 40, 44, 52, 56, 60, 64, 68, 72, 80, 84, 88, 92, 100, 104, 108, 112, 120, 124], 'expert_count': 26, 'ideal_gpu_count': 26, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 7522, 'token_per_expert': {0: 338, 4: 32, 8: 62, 12: 1, 16: 47, 20: 1, 24: 2, 32: 41, 40: 16, 44: 4, 52: 40, 56: 2101, 60: 1070, 64: 9, 68: 1, 72: 391, 80: 137, 84: 84, 88: 701, 92: 740, 100: 932, 104: 4, 108: 100, 112: 65, 120: 4, 124: 599}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 37, 41, 45, 49, 57, 61, 65, 69, 77, 81, 85, 89, 93, 105, 109, 113, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 11870, 'token_per_expert': {1: 323, 5: 2, 9: 69, 13: 6, 17: 5, 21: 514, 29: 2, 37: 3341, 41: 1832, 45: 442, 49: 2, 57: 52, 61: 55, 65: 33, 69: 5, 77: 2, 81: 2580, 85: 129, 89: 40, 93: 156, 105: 17, 109: 1, 113: 2075, 121: 124, 125: 63}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 18, 22, 30, 34, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 98, 102, 106, 110, 114, 126], 'expert_count': 26, 'ideal_gpu_count': 25, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 8878, 'token_per_expert': {2: 6, 10: 2134, 14: 925, 18: 2, 22: 876, 30: 8, 34: 14, 42: 158, 46: 560, 50: 7, 54: 23, 58: 422, 62: 4, 66: 1727, 70: 234, 74: 204, 78: 164, 82: 15, 86: 47, 90: 165, 98: 94, 102: 2, 106: 166, 110: 155, 114: 4, 126: 762}}
INFO 05-06 16:00:02.928813.928813 lmp.py:1845] [layer_moe_fused] layer=10 prefix: 5.016ms alloc: 0.233ms
INFO 05-06 16:00:02.928628.928628 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.364418029785156e-05 seconds
INFO 05-06 16:00:02.929751.929751 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012843608856201172s
INFO 05-06 16:00:02.930151.930151 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001012563705444336s
DEBUG 05-06 16:00:02.930234.930234 cuda_h.py:27] end moe_wait_copy_tasks cost 1.112 ms
DEBUG 05-06 16:00:02.934273.934273 cuda_h.py:27] end moe_vllm_forward cost 3.128 ms
DEBUG 05-06 16:00:02.934308.934308 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:02.934105.934105 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.419ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.934653.934653 cuda_h.py:27] end *layer_moe_fused cost 11.558 ms
DEBUG 05-06 16:00:02.934517.934517 cuda_h.py:27] end prefill_merge_scale cost 0.300 ms
DEBUG 05-06 16:00:02.935540.935540 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:02.935555.935555 cuda_h.py:27] end prefill_layer cost 58.645 ms
DEBUG 05-06 16:00:02.935177.935177 lmp.py:1394] -------------------------------- end prefill layer 10 --------------------------------
DEBUG 05-06 16:00:02.935542.935542 lmp.py:1350] -------------------------------- start prefill layer 11 --------------------------------
DEBUG 05-06 16:00:02.935707.935707 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 16:00:02.980724.980724 cuda_h.py:27] end prefill_attn cost 44.939 ms
DEBUG 05-06 16:00:02.980670.980670 cuda_h.py:27] end prefill_ffn_prep cost 0.306 ms
DEBUG 05-06 16:00:02.981554.981554 cuda_h.py:27] end prefill_gate cost 0.327 ms
INFO 05-06 16:00:02.989950.989950 lmp.py:1823] [layer_moe_fused] layer=11 active_experts=92 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 59, 67, 71, 79, 83, 87, 91, 95, 99, 103, 111, 115, 119, 123], 'expert_count': 26, 'ideal_gpu_count': 23, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 6664, 'token_per_expert': {7: 150, 11: 93, 15: 2, 19: 124, 23: 1358, 27: 1138, 31: 297, 35: 1, 39: 1, 43: 234, 47: 8, 51: 20, 59: 377, 67: 39, 71: 164, 79: 642, 83: 888, 87: 501, 91: 1, 95: 1, 99: 30, 103: 5, 111: 280, 115: 308, 119: 1, 123: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 16, 20, 24, 28, 32, 36, 52, 56, 68, 80, 84, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 22, 'ideal_gpu_count': 23, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 7860, 'token_per_expert': {0: 1580, 4: 74, 8: 70, 16: 713, 20: 568, 24: 278, 28: 2, 32: 13, 36: 1993, 52: 3, 56: 57, 68: 502, 80: 1, 84: 230, 92: 766, 100: 263, 104: 67, 108: 421, 112: 9, 116: 13, 120: 19, 124: 218}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 17, 21, 25, 29, 33, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 105, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 23, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 9336, 'token_per_expert': {1: 1, 5: 6, 9: 2, 17: 2907, 21: 20, 25: 59, 29: 40, 33: 2, 49: 248, 53: 7, 57: 44, 61: 353, 65: 2, 69: 263, 73: 18, 77: 156, 81: 900, 85: 8, 89: 245, 93: 263, 105: 7, 113: 81, 117: 3654, 121: 39, 125: 11}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 26, 30, 34, 38, 42, 46, 50, 58, 66, 70, 82, 98, 102, 110, 126], 'expert_count': 19, 'ideal_gpu_count': 23, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 8908, 'token_per_expert': {2: 143, 6: 656, 18: 437, 22: 4, 26: 2, 30: 266, 34: 277, 38: 265, 42: 1508, 46: 632, 50: 450, 58: 9, 66: 513, 70: 299, 82: 35, 98: 945, 102: 1971, 110: 430, 126: 66}}
INFO 05-06 16:00:02.989437.989437 lmp.py:1845] [layer_moe_fused] layer=11 prefix: 7.511ms alloc: 0.231ms
INFO 05-06 16:00:02.990678.990678 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.5789947509765625e-05 seconds
INFO 05-06 16:00:02.991465.991465 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0015263557434082031s
INFO 05-06 16:00:02.992721.992721 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010068416595458984s
DEBUG 05-06 16:00:02.992555.992555 cuda_h.py:27] end moe_wait_copy_tasks cost 1.204 ms
DEBUG 05-06 16:00:02.996855.996855 cuda_h.py:27] end moe_vllm_forward cost 3.517 ms
DEBUG 05-06 16:00:02.996082.996082 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 16:00:02.996786.996786 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.823ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:02.996864.996864 cuda_h.py:27] end *layer_moe_fused cost 14.975 ms
DEBUG 05-06 16:00:02.997908.997908 cuda_h.py:27] end prefill_merge_scale cost 0.309 ms
DEBUG 05-06 16:00:02.997693.997693 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:02.997330.997330 cuda_h.py:27] end prefill_layer cost 62.307 ms
DEBUG 05-06 16:00:02.997911.997911 lmp.py:1394] -------------------------------- end prefill layer 11 --------------------------------
DEBUG 05-06 16:00:02.997753.997753 lmp.py:1350] -------------------------------- start prefill layer 12 --------------------------------
DEBUG 05-06 16:00:02.997203.997203 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 16:00:03.043376.043376 cuda_h.py:27] end prefill_attn cost 45.052 ms
DEBUG 05-06 16:00:03.043461.043461 cuda_h.py:27] end prefill_ffn_prep cost 0.302 ms
DEBUG 05-06 16:00:03.044698.044698 cuda_h.py:27] end prefill_gate cost 0.329 ms
INFO 05-06 16:00:03.049694.049694 lmp.py:1823] [layer_moe_fused] layer=12 active_experts=75 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 35, 39, 47, 51, 59, 63, 71, 79, 83, 91, 95], 'expert_count': 13, 'ideal_gpu_count': 19, 'keep_on_gpu': 13, 'hit_count_on_device': 13, 'token_total': 5016, 'token_per_expert': {3: 2075, 15: 489, 35: 4, 39: 59, 47: 259, 51: 7, 59: 46, 63: 4, 71: 1449, 79: 105, 83: 180, 91: 90, 95: 249}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 12, 20, 36, 40, 48, 64, 68, 76, 80, 84, 92, 100, 104, 108, 116, 120, 124], 'expert_count': 19, 'ideal_gpu_count': 19, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 7321, 'token_per_expert': {0: 1, 8: 13, 12: 59, 20: 38, 36: 499, 40: 157, 48: 458, 64: 48, 68: 2, 76: 310, 80: 476, 84: 596, 92: 1832, 100: 466, 104: 13, 108: 368, 116: 194, 120: 353, 124: 1438}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 25, 29, 33, 37, 41, 45, 49, 53, 65, 73, 77, 81, 85, 89, 93, 97, 101, 117, 125], 'expert_count': 22, 'ideal_gpu_count': 19, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 15475, 'token_per_expert': {1: 54, 5: 1145, 13: 20, 25: 114, 29: 19, 33: 24, 37: 2, 41: 26, 45: 3215, 49: 3111, 53: 3144, 65: 53, 73: 95, 77: 1384, 81: 1, 85: 1153, 89: 1, 93: 5, 97: 1, 101: 3, 117: 1896, 125: 9}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 14, 18, 22, 34, 38, 46, 50, 58, 70, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 122], 'expert_count': 21, 'ideal_gpu_count': 18, 'keep_on_gpu': 21, 'hit_count_on_device': 21, 'token_total': 4956, 'token_per_expert': {6: 28, 14: 19, 18: 35, 22: 148, 34: 6, 38: 10, 46: 31, 50: 575, 58: 142, 70: 1, 78: 7, 82: 2632, 86: 6, 90: 29, 94: 477, 98: 77, 102: 354, 106: 276, 110: 38, 114: 61, 122: 4}}
INFO 05-06 16:00:03.049855.049855 lmp.py:1845] [layer_moe_fused] layer=12 prefix: 4.863ms alloc: 0.203ms
INFO 05-06 16:00:03.049568.049568 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.76837158203125e-05 seconds
INFO 05-06 16:00:03.051257.051257 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0015594959259033203s
INFO 05-06 16:00:03.052683.052683 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009799003601074219s
DEBUG 05-06 16:00:03.052580.052580 cuda_h.py:27] end moe_wait_copy_tasks cost 1.083 ms
DEBUG 05-06 16:00:03.056748.056748 cuda_h.py:27] end moe_vllm_forward cost 3.420 ms
DEBUG 05-06 16:00:03.056452.056452 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 16:00:03.056872.056872 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.728ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.056864.056864 cuda_h.py:27] end *layer_moe_fused cost 12.061 ms
DEBUG 05-06 16:00:03.057370.057370 cuda_h.py:27] end prefill_merge_scale cost 0.320 ms
DEBUG 05-06 16:00:03.057400.057400 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.037 ms
DEBUG 05-06 16:00:03.057991.057991 cuda_h.py:27] end prefill_layer cost 59.514 ms
DEBUG 05-06 16:00:03.057016.057016 lmp.py:1394] -------------------------------- end prefill layer 12 --------------------------------
DEBUG 05-06 16:00:03.057812.057812 lmp.py:1350] -------------------------------- start prefill layer 13 --------------------------------
DEBUG 05-06 16:00:03.057169.057169 cuda_h.py:27] end prefill_ln cost 0.165 ms
DEBUG 05-06 16:00:03.103291.103291 cuda_h.py:27] end prefill_attn cost 45.333 ms
DEBUG 05-06 16:00:03.103363.103363 cuda_h.py:27] end prefill_ffn_prep cost 0.303 ms
DEBUG 05-06 16:00:03.104181.104181 cuda_h.py:27] end prefill_gate cost 0.327 ms
INFO 05-06 16:00:03.108347.108347 lmp.py:1823] [layer_moe_fused] layer=13 active_experts=82 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 23, 27, 31, 35, 43, 47, 59, 63, 71, 75, 87, 91, 95, 99, 107, 111, 115, 119, 123], 'expert_count': 23, 'ideal_gpu_count': 21, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 5736, 'token_per_expert': {3: 298, 11: 12, 15: 53, 19: 116, 23: 2, 27: 10, 31: 1827, 35: 1, 43: 279, 47: 32, 59: 300, 63: 5, 71: 2, 75: 991, 87: 2, 91: 1514, 95: 185, 99: 3, 107: 9, 111: 2, 115: 1, 119: 88, 123: 4}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 16, 32, 40, 52, 56, 64, 68, 80, 84, 96, 104, 108, 112, 116, 120, 124], 'expert_count': 17, 'ideal_gpu_count': 21, 'keep_on_gpu': 17, 'hit_count_on_device': 17, 'token_total': 4436, 'token_per_expert': {0: 1, 16: 7, 32: 406, 40: 16, 52: 76, 56: 1, 64: 294, 68: 6, 80: 347, 84: 512, 96: 189, 104: 1641, 108: 5, 112: 42, 116: 887, 120: 3, 124: 3}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 17, 21, 25, 33, 37, 41, 45, 53, 61, 65, 69, 85, 93, 101, 113, 121, 125], 'expert_count': 20, 'ideal_gpu_count': 20, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 9789, 'token_per_expert': {1: 726, 9: 252, 13: 3830, 17: 33, 21: 36, 25: 82, 33: 2, 37: 73, 41: 35, 45: 261, 53: 516, 61: 162, 65: 1013, 69: 49, 85: 205, 93: 1164, 101: 14, 113: 260, 121: 444, 125: 632}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 38, 42, 46, 54, 58, 62, 66, 70, 78, 82, 86, 94, 98, 102, 110, 114, 118, 122, 126], 'expert_count': 22, 'ideal_gpu_count': 20, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 12807, 'token_per_expert': {2: 1242, 6: 8, 14: 115, 38: 3459, 42: 472, 46: 79, 54: 82, 58: 939, 62: 70, 66: 57, 70: 1, 78: 3877, 82: 529, 86: 2, 94: 46, 98: 6, 102: 1531, 110: 60, 114: 195, 118: 1, 122: 33, 126: 3}}
INFO 05-06 16:00:03.108330.108330 lmp.py:1845] [layer_moe_fused] layer=13 prefix: 4.214ms alloc: 0.211ms
INFO 05-06 16:00:03.109472.109472 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.9591064453125e-05 seconds
INFO 05-06 16:00:03.110039.110039 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014696121215820312s
INFO 05-06 16:00:03.111807.111807 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009257793426513672s
DEBUG 05-06 16:00:03.111359.111359 cuda_h.py:27] end moe_wait_copy_tasks cost 1.021 ms
DEBUG 05-06 16:00:03.115964.115964 cuda_h.py:27] end moe_vllm_forward cost 3.216 ms
DEBUG 05-06 16:00:03.115668.115668 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 16:00:03.115657.115657 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.523ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.115981.115981 cuda_h.py:27] end *layer_moe_fused cost 11.016 ms
DEBUG 05-06 16:00:03.116605.116605 cuda_h.py:27] end prefill_merge_scale cost 0.322 ms
DEBUG 05-06 16:00:03.116350.116350 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.038 ms
DEBUG 05-06 16:00:03.116372.116372 cuda_h.py:27] end prefill_layer cost 58.664 ms
DEBUG 05-06 16:00:03.116960.116960 lmp.py:1394] -------------------------------- end prefill layer 13 --------------------------------
DEBUG 05-06 16:00:03.116563.116563 lmp.py:1350] -------------------------------- start prefill layer 14 --------------------------------
DEBUG 05-06 16:00:03.116205.116205 cuda_h.py:27] end prefill_ln cost 0.164 ms
DEBUG 05-06 16:00:03.161864.161864 cuda_h.py:27] end prefill_attn cost 45.129 ms
DEBUG 05-06 16:00:03.162533.162533 cuda_h.py:27] end prefill_ffn_prep cost 0.304 ms
DEBUG 05-06 16:00:03.163977.163977 cuda_h.py:27] end prefill_gate cost 0.349 ms
INFO 05-06 16:00:03.168757.168757 lmp.py:1823] [layer_moe_fused] layer=14 active_experts=65 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 31, 35, 39, 47, 51, 59, 71, 83, 91, 95, 99, 107, 111, 115, 119, 123, 127], 'expert_count': 19, 'ideal_gpu_count': 17, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 6574, 'token_per_expert': {3: 1, 7: 118, 31: 259, 35: 7, 39: 21, 47: 59, 51: 2, 59: 2257, 71: 190, 83: 290, 91: 80, 95: 1337, 99: 22, 107: 182, 111: 1632, 115: 45, 119: 43, 123: 15, 127: 14}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 12, 16, 20, 24, 44, 48, 52, 60, 68, 76, 80, 108, 112, 124], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 16, 'token_total': 11897, 'token_per_expert': {0: 300, 8: 4068, 12: 44, 16: 479, 20: 5, 24: 49, 44: 20, 48: 436, 52: 2874, 60: 947, 68: 418, 76: 1, 80: 91, 108: 39, 112: 2125, 124: 1}}
experts_gpu_alloc_device_2 {'expert_ids': [9, 29, 33, 45, 65, 73, 77, 81, 89, 97, 105, 117, 121, 125], 'expert_count': 14, 'ideal_gpu_count': 16, 'keep_on_gpu': 14, 'hit_count_on_device': 14, 'token_total': 6189, 'token_per_expert': {9: 1, 29: 5, 33: 59, 45: 36, 65: 265, 73: 13, 77: 2, 81: 2, 89: 380, 97: 3446, 105: 12, 117: 251, 121: 1474, 125: 243}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 18, 22, 26, 38, 42, 50, 54, 74, 78, 86, 98, 102, 110, 122], 'expert_count': 16, 'ideal_gpu_count': 16, 'keep_on_gpu': 16, 'hit_count_on_device': 16, 'token_total': 8108, 'token_per_expert': {6: 13, 10: 3130, 18: 900, 22: 1, 26: 40, 38: 1622, 42: 14, 50: 380, 54: 68, 74: 68, 78: 1, 86: 286, 98: 51, 102: 2, 110: 1527, 122: 5}}
INFO 05-06 16:00:03.168229.168229 lmp.py:1845] [layer_moe_fused] layer=14 prefix: 4.772ms alloc: 0.184ms
INFO 05-06 16:00:03.168543.168543 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.078315734863281e-05 seconds
INFO 05-06 16:00:03.170179.170179 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0015554428100585938s
INFO 05-06 16:00:03.171211.171211 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009105205535888672s
DEBUG 05-06 16:00:03.171824.171824 cuda_h.py:27] end moe_wait_copy_tasks cost 1.014 ms
DEBUG 05-06 16:00:03.174451.174451 cuda_h.py:27] end moe_vllm_forward cost 3.479 ms
DEBUG 05-06 16:00:03.174208.174208 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 16:00:03.175581.175581 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.791ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.175196.175196 cuda_h.py:27] end *layer_moe_fused cost 11.887 ms
DEBUG 05-06 16:00:03.175672.175672 cuda_h.py:27] end prefill_merge_scale cost 0.335 ms
DEBUG 05-06 16:00:03.175271.175271 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.036 ms
DEBUG 05-06 16:00:03.175670.175670 cuda_h.py:27] end prefill_layer cost 59.558 ms
DEBUG 05-06 16:00:03.176490.176490 lmp.py:1394] -------------------------------- end prefill layer 14 --------------------------------
DEBUG 05-06 16:00:03.176047.176047 lmp.py:1350] -------------------------------- start prefill layer 15 --------------------------------
DEBUG 05-06 16:00:03.176974.176974 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 16:00:03.220583.220583 cuda_h.py:27] end prefill_attn cost 44.253 ms
DEBUG 05-06 16:00:03.220648.220648 cuda_h.py:27] end prefill_ffn_prep cost 0.299 ms
DEBUG 05-06 16:00:03.221447.221447 cuda_h.py:27] end prefill_gate cost 0.321 ms
INFO 05-06 16:00:03.226998.226998 lmp.py:1823] [layer_moe_fused] layer=15 active_experts=74 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 23, 35, 39, 47, 55, 59, 67, 71, 75, 91, 95, 99, 103, 107, 127], 'expert_count': 16, 'ideal_gpu_count': 19, 'keep_on_gpu': 16, 'hit_count_on_device': 16, 'token_total': 3220, 'token_per_expert': {3: 4, 23: 17, 35: 1, 39: 86, 47: 44, 55: 6, 59: 3, 67: 842, 71: 1, 75: 23, 91: 589, 95: 95, 99: 129, 103: 291, 107: 1048, 127: 41}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 16, 20, 28, 32, 36, 40, 48, 52, 56, 60, 68, 72, 76, 80, 84, 88, 100, 112, 116, 120], 'expert_count': 23, 'ideal_gpu_count': 19, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 19784, 'token_per_expert': {0: 5, 4: 220, 12: 2692, 16: 219, 20: 11, 28: 618, 32: 2, 36: 3845, 40: 149, 48: 94, 52: 4, 56: 10, 60: 6, 68: 724, 72: 3, 76: 3879, 80: 1, 84: 1423, 88: 4, 100: 2167, 112: 2354, 116: 1352, 120: 2}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 33, 37, 41, 45, 65, 69, 85, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 19, 'ideal_gpu_count': 18, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 8143, 'token_per_expert': {5: 7, 9: 1347, 13: 3, 33: 463, 37: 4, 41: 101, 45: 1, 65: 1047, 69: 54, 85: 162, 93: 1, 97: 557, 101: 99, 105: 582, 109: 974, 113: 2, 117: 602, 121: 101, 125: 2036}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 18, 22, 30, 34, 38, 42, 46, 70, 78, 82, 98, 110, 114, 118], 'expert_count': 16, 'ideal_gpu_count': 18, 'keep_on_gpu': 16, 'hit_count_on_device': 16, 'token_total': 1621, 'token_per_expert': {2: 63, 6: 140, 18: 23, 22: 13, 30: 15, 34: 666, 38: 79, 42: 44, 46: 131, 70: 233, 78: 111, 82: 16, 98: 8, 110: 3, 114: 5, 118: 71}}
INFO 05-06 16:00:03.226099.226099 lmp.py:1845] [layer_moe_fused] layer=15 prefix: 4.046ms alloc: 0.194ms
INFO 05-06 16:00:03.226414.226414 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.982948303222656e-05 seconds
INFO 05-06 16:00:03.228681.228681 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014240741729736328s
INFO 05-06 16:00:03.229403.229403 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009272098541259766s
DEBUG 05-06 16:00:03.229810.229810 cuda_h.py:27] end moe_wait_copy_tasks cost 1.020 ms
DEBUG 05-06 16:00:03.232810.232810 cuda_h.py:27] end moe_vllm_forward cost 3.163 ms
DEBUG 05-06 16:00:03.232084.232084 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.232265.232265 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.462ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.232065.232065 cuda_h.py:27] end *layer_moe_fused cost 10.748 ms
DEBUG 05-06 16:00:03.233491.233491 cuda_h.py:27] end prefill_merge_scale cost 0.313 ms
DEBUG 05-06 16:00:03.233097.233097 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.036 ms
DEBUG 05-06 16:00:03.233019.233019 cuda_h.py:27] end prefill_layer cost 57.317 ms
DEBUG 05-06 16:00:03.233522.233522 lmp.py:1394] -------------------------------- end prefill layer 15 --------------------------------
DEBUG 05-06 16:00:03.233364.233364 lmp.py:1350] -------------------------------- start prefill layer 16 --------------------------------
DEBUG 05-06 16:00:03.233052.233052 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 16:00:03.268857.268857 cuda_h.py:27] end prefill_attn cost 34.250 ms
DEBUG 05-06 16:00:03.268498.268498 cuda_h.py:27] end prefill_ffn_prep cost 0.299 ms
DEBUG 05-06 16:00:03.269185.269185 cuda_h.py:27] end prefill_gate cost 0.318 ms
INFO 05-06 16:00:03.274469.274469 lmp.py:1823] [layer_moe_fused] layer=16 active_experts=82 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 15, 23, 31, 35, 39, 47, 51, 55, 63, 67, 71, 79, 87, 99, 103, 115, 119, 123], 'expert_count': 19, 'ideal_gpu_count': 21, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 7128, 'token_per_expert': {3: 1, 15: 18, 23: 58, 31: 1091, 35: 470, 39: 603, 47: 66, 51: 33, 55: 735, 63: 246, 67: 18, 71: 117, 79: 103, 87: 123, 99: 298, 103: 2662, 115: 133, 119: 225, 123: 128}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 20, 24, 28, 32, 36, 44, 48, 56, 60, 68, 72, 80, 84, 92, 96, 100, 104, 108, 112, 120], 'expert_count': 23, 'ideal_gpu_count': 21, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 9734, 'token_per_expert': {0: 2, 4: 700, 12: 458, 20: 1587, 24: 159, 28: 1, 32: 80, 36: 80, 44: 86, 48: 16, 56: 3639, 60: 1, 68: 454, 72: 155, 80: 7, 84: 314, 92: 119, 96: 2, 100: 421, 104: 33, 108: 119, 112: 4, 120: 1297}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 49, 61, 81, 105, 109, 117, 121], 'expert_count': 15, 'ideal_gpu_count': 20, 'keep_on_gpu': 15, 'hit_count_on_device': 15, 'token_total': 12660, 'token_per_expert': {1: 4, 5: 31, 9: 1860, 13: 87, 17: 2347, 21: 482, 25: 1002, 29: 2035, 49: 1, 61: 1, 81: 3910, 105: 3, 109: 789, 117: 107, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 18, 26, 30, 34, 38, 42, 46, 50, 54, 58, 66, 70, 74, 78, 82, 86, 90, 94, 102, 106, 110, 114, 126], 'expert_count': 25, 'ideal_gpu_count': 20, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 3246, 'token_per_expert': {6: 15, 10: 24, 18: 1039, 26: 175, 30: 4, 34: 92, 38: 201, 42: 68, 46: 399, 50: 46, 54: 3, 58: 256, 66: 8, 70: 1, 74: 72, 78: 20, 82: 56, 86: 199, 90: 83, 94: 2, 102: 183, 106: 14, 110: 145, 114: 121, 126: 20}}
INFO 05-06 16:00:03.274491.274491 lmp.py:1845] [layer_moe_fused] layer=16 prefix: 4.972ms alloc: 0.206ms
INFO 05-06 16:00:03.274074.274074 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.76837158203125e-05 seconds
INFO 05-06 16:00:03.276000.276000 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001489400863647461s
INFO 05-06 16:00:03.277248.277248 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0008308887481689453s
DEBUG 05-06 16:00:03.277748.277748 cuda_h.py:27] end moe_wait_copy_tasks cost 0.921 ms
DEBUG 05-06 16:00:03.280768.280768 cuda_h.py:27] end moe_vllm_forward cost 3.183 ms
DEBUG 05-06 16:00:03.280280.280280 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.280554.280554 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.477ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.281308.281308 cuda_h.py:27] end *layer_moe_fused cost 11.576 ms
DEBUG 05-06 16:00:03.281107.281107 cuda_h.py:27] end prefill_merge_scale cost 0.307 ms
DEBUG 05-06 16:00:03.281845.281845 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:03.281098.281098 cuda_h.py:27] end prefill_layer cost 48.223 ms
DEBUG 05-06 16:00:03.281554.281554 lmp.py:1394] -------------------------------- end prefill layer 16 --------------------------------
DEBUG 05-06 16:00:03.282687.282687 lmp.py:1350] -------------------------------- start prefill layer 17 --------------------------------
DEBUG 05-06 16:00:03.282614.282614 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 16:00:03.310010.310010 cuda_h.py:27] end prefill_attn cost 28.400 ms
DEBUG 05-06 16:00:03.311837.311837 cuda_h.py:27] end prefill_ffn_prep cost 0.295 ms
DEBUG 05-06 16:00:03.311545.311545 cuda_h.py:27] end prefill_gate cost 0.311 ms
INFO 05-06 16:00:03.319286.319286 lmp.py:1823] [layer_moe_fused] layer=17 active_experts=97 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 23, 35, 39, 43, 47, 51, 55, 71, 75, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 4590, 'token_per_expert': {3: 768, 7: 12, 11: 1, 19: 69, 23: 146, 35: 19, 39: 390, 43: 354, 47: 167, 51: 320, 55: 513, 71: 321, 75: 6, 83: 306, 87: 9, 91: 89, 95: 23, 99: 80, 103: 598, 107: 7, 111: 353, 115: 11, 119: 4, 123: 22, 127: 2}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 56, 60, 64, 68, 76, 80, 84, 88, 100, 104, 108, 112, 116, 124], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 6253, 'token_per_expert': {0: 182, 4: 955, 8: 8, 12: 213, 16: 10, 20: 1189, 24: 12, 28: 75, 32: 223, 36: 1, 40: 239, 56: 10, 60: 235, 64: 2019, 68: 99, 76: 1, 80: 3, 84: 3, 88: 54, 100: 8, 104: 62, 108: 137, 112: 290, 116: 173, 124: 52}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 53, 57, 65, 73, 77, 81, 85, 97, 109, 113, 125], 'expert_count': 22, 'ideal_gpu_count': 24, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 15833, 'token_per_expert': {5: 23, 9: 2159, 13: 1, 17: 1, 21: 26, 25: 16, 29: 966, 33: 5, 37: 17, 41: 10, 45: 1094, 53: 3419, 57: 213, 65: 34, 73: 9, 77: 767, 81: 2318, 85: 14, 97: 277, 109: 78, 113: 360, 125: 4026}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 22, 26, 30, 38, 42, 46, 54, 58, 62, 66, 70, 74, 78, 86, 90, 94, 98, 106, 110, 114, 118, 122], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 6092, 'token_per_expert': {2: 2, 10: 461, 14: 14, 22: 17, 26: 1202, 30: 7, 38: 242, 42: 73, 46: 2, 54: 645, 58: 13, 62: 1, 66: 16, 70: 2, 74: 4, 78: 53, 86: 95, 90: 74, 94: 2, 98: 282, 106: 15, 110: 2773, 114: 4, 118: 39, 122: 54}}
INFO 05-06 16:00:03.319323.319323 lmp.py:1845] [layer_moe_fused] layer=17 prefix: 7.517ms alloc: 0.251ms
INFO 05-06 16:00:03.320410.320410 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.6743621826171875e-05 seconds
INFO 05-06 16:00:03.341353.341353 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.02140641212463379s
INFO 05-06 16:00:03.342695.342695 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0012221336364746094s
DEBUG 05-06 16:00:03.343626.343626 cuda_h.py:27] end moe_wait_copy_tasks cost 1.350 ms
DEBUG 05-06 16:00:03.348520.348520 cuda_h.py:27] end moe_vllm_forward cost 4.634 ms
DEBUG 05-06 16:00:03.348536.348536 cuda_h.py:27] end moe_shared_experts cost 0.005 ms
INFO 05-06 16:00:03.348234.348234 lmp.py:1964] [layer_moe_fused] vllm triton time: 5.057ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.348600.348600 cuda_h.py:27] end *layer_moe_fused cost 36.413 ms
DEBUG 05-06 16:00:03.349253.349253 cuda_h.py:27] end prefill_merge_scale cost 0.405 ms
DEBUG 05-06 16:00:03.349025.349025 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.045 ms
DEBUG 05-06 16:00:03.349979.349979 cuda_h.py:27] end prefill_layer cost 67.452 ms
DEBUG 05-06 16:00:03.349603.349603 lmp.py:1394] -------------------------------- end prefill layer 17 --------------------------------
DEBUG 05-06 16:00:03.349166.349166 lmp.py:1350] -------------------------------- start prefill layer 18 --------------------------------
DEBUG 05-06 16:00:03.349902.349902 cuda_h.py:27] end prefill_ln cost 0.203 ms
DEBUG 05-06 16:00:03.359976.359976 cuda_h.py:27] end prefill_attn cost 9.854 ms
DEBUG 05-06 16:00:03.360888.360888 cuda_h.py:27] end prefill_ffn_prep cost 0.293 ms
DEBUG 05-06 16:00:03.361143.361143 cuda_h.py:27] end prefill_gate cost 0.339 ms
INFO 05-06 16:00:03.366917.366917 lmp.py:1823] [layer_moe_fused] layer=18 active_experts=97 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 11, 15, 19, 27, 31, 35, 39, 47, 51, 55, 59, 63, 67, 75, 79, 83, 87, 91, 99, 103, 111, 119, 123, 127], 'expert_count': 25, 'ideal_gpu_count': 25, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 9868, 'token_per_expert': {3: 1231, 11: 28, 15: 18, 19: 808, 27: 135, 31: 447, 35: 53, 39: 398, 47: 109, 51: 12, 55: 15, 59: 1233, 63: 490, 67: 63, 75: 787, 79: 15, 83: 162, 87: 134, 91: 224, 99: 3, 103: 234, 111: 86, 119: 752, 123: 2294, 127: 137}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 52, 56, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 112, 116, 120, 124], 'expert_count': 26, 'ideal_gpu_count': 24, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 13207, 'token_per_expert': {0: 1, 4: 576, 8: 363, 12: 494, 16: 1581, 20: 13, 24: 111, 28: 1, 32: 341, 40: 1122, 52: 1360, 56: 134, 64: 10, 68: 1093, 72: 544, 76: 10, 80: 108, 84: 2052, 88: 1152, 92: 4, 96: 58, 100: 1392, 112: 25, 116: 49, 120: 38, 124: 575}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 13, 21, 25, 29, 33, 37, 49, 53, 57, 61, 69, 73, 77, 81, 85, 93, 97, 105, 109, 113, 125], 'expert_count': 22, 'ideal_gpu_count': 24, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 5938, 'token_per_expert': {1: 74, 13: 1573, 21: 104, 25: 40, 29: 1, 33: 678, 37: 5, 49: 12, 53: 81, 57: 6, 61: 1, 69: 270, 73: 134, 77: 26, 81: 176, 85: 1306, 93: 271, 97: 271, 105: 453, 109: 454, 113: 1, 125: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 62, 66, 70, 78, 90, 94, 98, 102, 106, 110, 114, 126], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 3755, 'token_per_expert': {2: 227, 6: 1, 14: 83, 18: 66, 22: 1, 26: 88, 30: 271, 34: 3, 38: 329, 42: 544, 46: 2, 50: 315, 62: 555, 66: 37, 70: 15, 78: 11, 90: 97, 94: 244, 98: 18, 102: 2, 106: 19, 110: 547, 114: 242, 126: 38}}
INFO 05-06 16:00:03.366688.366688 lmp.py:1845] [layer_moe_fused] layer=18 prefix: 4.948ms alloc: 0.232ms
INFO 05-06 16:00:03.366775.366775 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.841255187988281e-05 seconds
INFO 05-06 16:00:03.368904.368904 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014238357543945312s
INFO 05-06 16:00:03.369373.369373 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010917186737060547s
DEBUG 05-06 16:00:03.369541.369541 cuda_h.py:27] end moe_wait_copy_tasks cost 1.183 ms
DEBUG 05-06 16:00:03.372007.372007 cuda_h.py:27] end moe_vllm_forward cost 3.229 ms
DEBUG 05-06 16:00:03.372864.372864 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.373456.373456 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.557ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.373039.373039 cuda_h.py:27] end *layer_moe_fused cost 11.900 ms
DEBUG 05-06 16:00:03.373314.373314 cuda_h.py:27] end prefill_merge_scale cost 0.324 ms
DEBUG 05-06 16:00:03.373006.373006 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:03.373975.373975 cuda_h.py:27] end prefill_layer cost 24.332 ms
DEBUG 05-06 16:00:03.374947.374947 lmp.py:1394] -------------------------------- end prefill layer 18 --------------------------------
DEBUG 05-06 16:00:03.374550.374550 lmp.py:1350] -------------------------------- start prefill layer 19 --------------------------------
DEBUG 05-06 16:00:03.374993.374993 cuda_h.py:27] end prefill_ln cost 0.159 ms
DEBUG 05-06 16:00:03.385793.385793 cuda_h.py:27] end prefill_attn cost 11.178 ms
DEBUG 05-06 16:00:03.385805.385805 cuda_h.py:27] end prefill_ffn_prep cost 0.295 ms
DEBUG 05-06 16:00:03.386313.386313 cuda_h.py:27] end prefill_gate cost 0.310 ms
INFO 05-06 16:00:03.390760.390760 lmp.py:1823] [layer_moe_fused] layer=19 active_experts=87 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 15, 19, 23, 27, 35, 39, 43, 47, 51, 63, 67, 71, 75, 83, 87, 103, 111, 115, 119], 'expert_count': 20, 'ideal_gpu_count': 22, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 8674, 'token_per_expert': {7: 15, 15: 3, 19: 166, 23: 97, 27: 1923, 35: 137, 39: 196, 43: 3, 47: 10, 51: 280, 63: 477, 67: 1, 71: 24, 75: 2098, 83: 8, 87: 76, 103: 2, 111: 66, 115: 7, 119: 3085}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 52, 56, 60, 64, 72, 80, 84, 88, 92, 100, 104, 112, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 22, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 11122, 'token_per_expert': {4: 6, 8: 18, 12: 967, 16: 1, 20: 177, 24: 1236, 28: 979, 32: 133, 40: 1, 44: 1, 48: 2914, 52: 1695, 56: 147, 60: 2, 64: 112, 72: 1, 80: 1124, 84: 216, 88: 184, 92: 206, 100: 400, 104: 44, 112: 4, 120: 472, 124: 82}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 25, 37, 41, 45, 53, 65, 69, 73, 81, 89, 97, 101, 109, 117, 121, 125], 'expert_count': 20, 'ideal_gpu_count': 22, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 6492, 'token_per_expert': {1: 365, 5: 106, 9: 186, 13: 25, 25: 120, 37: 761, 41: 421, 45: 17, 53: 15, 65: 1, 69: 1386, 73: 302, 81: 1728, 89: 402, 97: 1, 101: 15, 109: 44, 117: 1, 121: 56, 125: 540}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 14, 18, 22, 30, 38, 46, 50, 58, 70, 74, 78, 82, 90, 94, 98, 102, 106, 110, 118, 122, 126], 'expert_count': 22, 'ideal_gpu_count': 21, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 6480, 'token_per_expert': {6: 26, 14: 14, 18: 13, 22: 1, 30: 1, 38: 3054, 46: 191, 50: 170, 58: 59, 70: 5, 74: 1, 78: 4, 82: 1, 90: 410, 94: 1687, 98: 1, 102: 207, 106: 177, 110: 66, 118: 71, 122: 316, 126: 5}}
INFO 05-06 16:00:03.391504.391504 lmp.py:1845] [layer_moe_fused] layer=19 prefix: 4.066ms alloc: 0.210ms
INFO 05-06 16:00:03.391505.391505 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.792213439941406e-05 seconds
INFO 05-06 16:00:03.392371.392371 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0013053417205810547s
INFO 05-06 16:00:03.393254.393254 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0009827613830566406s
DEBUG 05-06 16:00:03.393316.393316 cuda_h.py:27] end moe_wait_copy_tasks cost 1.067 ms
DEBUG 05-06 16:00:03.397799.397799 cuda_h.py:27] end moe_vllm_forward cost 3.141 ms
DEBUG 05-06 16:00:03.397311.397311 cuda_h.py:27] end moe_shared_experts cost 0.004 ms
INFO 05-06 16:00:03.397823.397823 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.432ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.397630.397630 cuda_h.py:27] end *layer_moe_fused cost 10.621 ms
DEBUG 05-06 16:00:03.398654.398654 cuda_h.py:27] end prefill_merge_scale cost 0.317 ms
DEBUG 05-06 16:00:03.398299.398299 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:03.398983.398983 cuda_h.py:27] end prefill_layer cost 24.194 ms
DEBUG 05-06 16:00:03.398438.398438 lmp.py:1394] -------------------------------- end prefill layer 19 --------------------------------
DEBUG 05-06 16:00:03.398042.398042 lmp.py:1350] -------------------------------- start prefill layer 20 --------------------------------
DEBUG 05-06 16:00:03.398969.398969 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 16:00:03.410291.410291 cuda_h.py:27] end prefill_attn cost 11.352 ms
DEBUG 05-06 16:00:03.410680.410680 cuda_h.py:27] end prefill_ffn_prep cost 0.293 ms
DEBUG 05-06 16:00:03.411968.411968 cuda_h.py:27] end prefill_gate cost 0.291 ms
INFO 05-06 16:00:03.416318.416318 lmp.py:1823] [layer_moe_fused] layer=20 active_experts=88 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 79, 83, 91, 103, 111, 115, 123, 127], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 11621, 'token_per_expert': {3: 251, 7: 11, 11: 228, 19: 190, 27: 179, 31: 4, 35: 12, 39: 56, 43: 3, 47: 1, 51: 299, 55: 63, 59: 163, 63: 1, 79: 155, 83: 1677, 91: 401, 103: 6, 111: 98, 115: 3506, 123: 2665, 127: 1652}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 28, 32, 36, 44, 52, 56, 60, 80, 84, 88, 92, 96, 100, 104, 116, 120], 'expert_count': 22, 'ideal_gpu_count': 22, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 3055, 'token_per_expert': {0: 1, 4: 2, 8: 405, 12: 21, 16: 18, 20: 178, 28: 48, 32: 109, 36: 2, 44: 2, 52: 199, 56: 17, 60: 4, 80: 83, 84: 398, 88: 4, 92: 330, 96: 131, 100: 419, 104: 24, 116: 24, 120: 636}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 29, 33, 45, 49, 53, 57, 61, 65, 69, 73, 81, 85, 89, 97, 101, 105, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 22, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 15301, 'token_per_expert': {1: 9, 5: 3158, 9: 109, 13: 1, 17: 41, 21: 31, 29: 9, 33: 828, 45: 677, 49: 2739, 53: 2, 57: 1104, 61: 61, 65: 723, 69: 1, 73: 2104, 81: 126, 85: 583, 89: 45, 97: 2281, 101: 165, 105: 82, 117: 1, 121: 39, 125: 382}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 30, 38, 50, 62, 66, 70, 78, 82, 90, 102, 106, 110, 114, 122], 'expert_count': 19, 'ideal_gpu_count': 22, 'keep_on_gpu': 19, 'hit_count_on_device': 19, 'token_total': 2791, 'token_per_expert': {2: 410, 6: 23, 14: 43, 18: 210, 22: 20, 30: 3, 38: 143, 50: 147, 62: 89, 66: 433, 70: 233, 78: 15, 82: 33, 90: 40, 102: 188, 106: 510, 110: 198, 114: 12, 122: 41}}
INFO 05-06 16:00:03.416943.416943 lmp.py:1845] [layer_moe_fused] layer=20 prefix: 4.973ms alloc: 0.209ms
INFO 05-06 16:00:03.416839.416839 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.7206878662109375e-05 seconds
INFO 05-06 16:00:03.418571.418571 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012776851654052734s
INFO 05-06 16:00:03.419661.419661 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001031637191772461s
DEBUG 05-06 16:00:03.419293.419293 cuda_h.py:27] end moe_wait_copy_tasks cost 1.114 ms
DEBUG 05-06 16:00:03.422602.422602 cuda_h.py:27] end moe_vllm_forward cost 3.120 ms
DEBUG 05-06 16:00:03.422161.422161 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.422242.422242 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.407ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.422619.422619 cuda_h.py:27] end *layer_moe_fused cost 11.511 ms
DEBUG 05-06 16:00:03.423316.423316 cuda_h.py:27] end prefill_merge_scale cost 0.333 ms
DEBUG 05-06 16:00:03.423961.423961 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.035 ms
DEBUG 05-06 16:00:03.423214.423214 cuda_h.py:27] end prefill_layer cost 25.126 ms
DEBUG 05-06 16:00:03.423823.423823 lmp.py:1394] -------------------------------- end prefill layer 20 --------------------------------
DEBUG 05-06 16:00:03.423095.423095 lmp.py:1350] -------------------------------- start prefill layer 21 --------------------------------
DEBUG 05-06 16:00:03.423737.423737 cuda_h.py:27] end prefill_ln cost 0.165 ms
DEBUG 05-06 16:00:03.435291.435291 cuda_h.py:27] end prefill_attn cost 11.348 ms
DEBUG 05-06 16:00:03.435695.435695 cuda_h.py:27] end prefill_ffn_prep cost 0.341 ms
DEBUG 05-06 16:00:03.436403.436403 cuda_h.py:27] end prefill_gate cost 0.293 ms
INFO 05-06 16:00:03.440283.440283 lmp.py:1823] [layer_moe_fused] layer=21 active_experts=103 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [11, 15, 19, 27, 31, 35, 39, 43, 47, 55, 59, 67, 71, 75, 79, 83, 91, 95, 99, 103, 111, 115, 119, 123], 'expert_count': 24, 'ideal_gpu_count': 26, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 7551, 'token_per_expert': {11: 1234, 15: 2603, 19: 1, 27: 36, 31: 10, 35: 210, 39: 171, 43: 13, 47: 7, 55: 123, 59: 96, 67: 533, 71: 119, 75: 475, 79: 34, 83: 342, 91: 10, 95: 672, 99: 245, 103: 187, 111: 2, 115: 355, 119: 65, 123: 8}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 52, 56, 68, 72, 76, 80, 88, 92, 96, 100, 104, 108, 112, 116, 120], 'expert_count': 26, 'ideal_gpu_count': 26, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 12713, 'token_per_expert': {0: 1120, 4: 88, 8: 3702, 12: 972, 16: 1, 24: 215, 28: 76, 32: 276, 36: 138, 40: 25, 44: 4, 52: 4, 56: 65, 68: 262, 72: 132, 76: 2861, 80: 877, 88: 362, 92: 147, 96: 7, 100: 145, 104: 1052, 108: 22, 112: 4, 116: 1, 120: 155}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 37, 41, 45, 49, 53, 57, 61, 65, 73, 77, 81, 85, 93, 97, 101, 105, 109, 117, 121], 'expert_count': 27, 'ideal_gpu_count': 26, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 6818, 'token_per_expert': {1: 213, 5: 468, 9: 188, 13: 307, 17: 153, 21: 10, 25: 15, 29: 854, 37: 7, 41: 191, 45: 14, 49: 491, 53: 310, 57: 1, 61: 89, 65: 1551, 73: 404, 77: 1, 81: 2, 85: 91, 93: 3, 97: 114, 101: 1, 105: 810, 109: 348, 117: 4, 121: 178}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 58, 62, 66, 78, 82, 86, 90, 94, 98, 106, 110, 118, 122, 126], 'expert_count': 26, 'ideal_gpu_count': 25, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 5686, 'token_per_expert': {2: 63, 6: 4, 10: 259, 14: 36, 18: 1629, 22: 133, 26: 48, 30: 233, 34: 220, 38: 40, 42: 4, 46: 1874, 58: 114, 62: 196, 66: 88, 78: 6, 82: 163, 86: 1, 90: 13, 94: 111, 98: 24, 106: 16, 110: 31, 118: 4, 122: 370, 126: 6}}
INFO 05-06 16:00:03.441585.441585 lmp.py:1845] [layer_moe_fused] layer=21 prefix: 4.186ms alloc: 0.237ms
INFO 05-06 16:00:03.441865.441865 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.125999450683594e-05 seconds
INFO 05-06 16:00:03.442578.442578 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012612342834472656s
INFO 05-06 16:00:03.443469.443469 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001058340072631836s
DEBUG 05-06 16:00:03.443008.443008 cuda_h.py:27] end moe_wait_copy_tasks cost 1.143 ms
DEBUG 05-06 16:00:03.447005.447005 cuda_h.py:27] end moe_vllm_forward cost 3.067 ms
DEBUG 05-06 16:00:03.447656.447656 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.447235.447235 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.365ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.447791.447791 cuda_h.py:27] end *layer_moe_fused cost 10.791 ms
DEBUG 05-06 16:00:03.447854.447854 cuda_h.py:27] end prefill_merge_scale cost 0.316 ms
DEBUG 05-06 16:00:03.448023.448023 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.035 ms
DEBUG 05-06 16:00:03.448753.448753 cuda_h.py:27] end prefill_layer cost 24.395 ms
DEBUG 05-06 16:00:03.448083.448083 lmp.py:1394] -------------------------------- end prefill layer 21 --------------------------------
DEBUG 05-06 16:00:03.448640.448640 lmp.py:1350] -------------------------------- start prefill layer 22 --------------------------------
DEBUG 05-06 16:00:03.448137.448137 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 16:00:03.459391.459391 cuda_h.py:27] end prefill_attn cost 11.287 ms
DEBUG 05-06 16:00:03.460827.460827 cuda_h.py:27] end prefill_ffn_prep cost 0.292 ms
DEBUG 05-06 16:00:03.461059.461059 cuda_h.py:27] end prefill_gate cost 0.291 ms
INFO 05-06 16:00:03.466219.466219 lmp.py:1823] [layer_moe_fused] layer=22 active_experts=102 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [7, 11, 15, 19, 23, 27, 35, 39, 47, 51, 59, 63, 67, 71, 75, 83, 87, 95, 99, 103, 107, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 26, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 4108, 'token_per_expert': {7: 40, 11: 3, 15: 33, 19: 228, 23: 265, 27: 2, 35: 150, 39: 11, 47: 2, 51: 6, 59: 3, 63: 195, 67: 79, 71: 53, 75: 170, 83: 144, 87: 58, 95: 1, 99: 405, 103: 1278, 107: 630, 119: 25, 123: 324, 127: 3}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 16, 20, 28, 32, 40, 44, 48, 52, 60, 64, 72, 76, 80, 84, 88, 92, 96, 100, 108, 116, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 26, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 13074, 'token_per_expert': {0: 297, 8: 87, 16: 125, 20: 13, 28: 150, 32: 1892, 40: 467, 44: 55, 48: 167, 52: 3, 60: 42, 64: 1, 72: 554, 76: 162, 80: 3, 84: 751, 88: 2948, 92: 49, 96: 2, 100: 676, 108: 1215, 116: 264, 120: 315, 124: 2836}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69, 73, 77, 81, 85, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 28, 'ideal_gpu_count': 25, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 11621, 'token_per_expert': {1: 7, 5: 48, 9: 330, 13: 46, 17: 249, 29: 25, 33: 49, 37: 285, 41: 3759, 45: 78, 53: 199, 57: 1, 61: 73, 65: 1, 69: 285, 73: 1251, 77: 1, 81: 23, 85: 19, 93: 40, 97: 1, 101: 1725, 105: 4, 109: 1, 113: 10, 117: 2923, 121: 2, 125: 186}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 22, 30, 34, 38, 42, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 106, 110, 114, 118, 122, 126], 'expert_count': 26, 'ideal_gpu_count': 25, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 3965, 'token_per_expert': {2: 161, 6: 13, 10: 85, 22: 1, 30: 2, 34: 551, 38: 35, 42: 214, 50: 796, 54: 206, 58: 3, 62: 1, 66: 240, 70: 24, 74: 7, 78: 60, 82: 179, 86: 1, 90: 126, 94: 106, 106: 1, 110: 51, 114: 6, 118: 222, 122: 1, 126: 873}}
INFO 05-06 16:00:03.466798.466798 lmp.py:1845] [layer_moe_fused] layer=22 prefix: 5.020ms alloc: 0.233ms
INFO 05-06 16:00:03.466422.466422 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.078315734863281e-05 seconds
INFO 05-06 16:00:03.468916.468916 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014522075653076172s
INFO 05-06 16:00:03.469336.469336 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010280609130859375s
DEBUG 05-06 16:00:03.469253.469253 cuda_h.py:27] end moe_wait_copy_tasks cost 1.110 ms
DEBUG 05-06 16:00:03.472450.472450 cuda_h.py:27] end moe_vllm_forward cost 3.108 ms
DEBUG 05-06 16:00:03.472485.472485 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.472249.472249 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.396ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.473155.473155 cuda_h.py:27] end *layer_moe_fused cost 11.768 ms
DEBUG 05-06 16:00:03.473138.473138 cuda_h.py:27] end prefill_merge_scale cost 0.323 ms
DEBUG 05-06 16:00:03.473592.473592 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.035 ms
DEBUG 05-06 16:00:03.473845.473845 cuda_h.py:27] end prefill_layer cost 25.290 ms
DEBUG 05-06 16:00:03.473917.473917 lmp.py:1394] -------------------------------- end prefill layer 22 --------------------------------
DEBUG 05-06 16:00:03.473236.473236 lmp.py:1350] -------------------------------- start prefill layer 23 --------------------------------
DEBUG 05-06 16:00:03.474448.474448 cuda_h.py:27] end prefill_ln cost 0.163 ms
DEBUG 05-06 16:00:03.485356.485356 cuda_h.py:27] end prefill_attn cost 11.433 ms
DEBUG 05-06 16:00:03.485441.485441 cuda_h.py:27] end prefill_ffn_prep cost 0.306 ms
DEBUG 05-06 16:00:03.486266.486266 cuda_h.py:27] end prefill_gate cost 0.284 ms
INFO 05-06 16:00:03.494338.494338 lmp.py:1823] [layer_moe_fused] layer=23 active_experts=105 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 30, 'ideal_gpu_count': 27, 'keep_on_gpu': 30, 'hit_count_on_device': 30, 'token_total': 7677, 'token_per_expert': {3: 15, 7: 8, 11: 101, 15: 5, 19: 16, 23: 467, 27: 60, 31: 621, 39: 257, 43: 4, 47: 3, 51: 3170, 55: 1, 59: 4, 63: 12, 67: 20, 71: 27, 75: 6, 79: 327, 83: 120, 87: 5, 91: 303, 99: 11, 103: 30, 107: 3, 111: 169, 115: 1, 119: 1898, 123: 12, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [4, 8, 12, 16, 20, 24, 28, 36, 40, 44, 56, 64, 72, 76, 80, 84, 88, 92, 96, 100, 104, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 26, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 9109, 'token_per_expert': {4: 6, 8: 27, 12: 213, 16: 1, 20: 165, 24: 24, 28: 734, 36: 52, 40: 64, 44: 185, 56: 927, 64: 126, 72: 27, 76: 435, 80: 45, 84: 3, 88: 1644, 92: 796, 96: 29, 100: 18, 104: 1973, 112: 31, 116: 48, 120: 1050, 124: 486}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 17, 21, 25, 29, 33, 37, 49, 53, 57, 61, 65, 69, 73, 77, 81, 93, 97, 101, 105, 109, 117, 121], 'expert_count': 25, 'ideal_gpu_count': 26, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 9183, 'token_per_expert': {1: 621, 9: 376, 13: 278, 17: 277, 21: 123, 25: 29, 29: 975, 33: 1071, 37: 47, 49: 1, 53: 4, 57: 207, 61: 207, 65: 11, 69: 1, 73: 3575, 77: 17, 81: 10, 93: 198, 97: 9, 101: 68, 105: 1058, 109: 18, 117: 1, 121: 1}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 38, 42, 50, 54, 62, 66, 70, 78, 82, 90, 98, 106, 110, 114, 118, 122, 126], 'expert_count': 25, 'ideal_gpu_count': 26, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 6799, 'token_per_expert': {2: 48, 6: 820, 10: 1, 14: 1970, 18: 222, 22: 102, 26: 435, 30: 27, 38: 374, 42: 581, 50: 1, 54: 6, 62: 484, 66: 31, 70: 30, 78: 249, 82: 1, 90: 22, 98: 22, 106: 231, 110: 59, 114: 975, 118: 44, 122: 3, 126: 61}}
INFO 05-06 16:00:03.494447.494447 lmp.py:1845] [layer_moe_fused] layer=23 prefix: 7.345ms alloc: 0.236ms
INFO 05-06 16:00:03.494589.494589 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.982948303222656e-05 seconds
INFO 05-06 16:00:03.496177.496177 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001520395278930664s
INFO 05-06 16:00:03.497338.497338 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011813640594482422s
DEBUG 05-06 16:00:03.497592.497592 cuda_h.py:27] end moe_wait_copy_tasks cost 1.266 ms
DEBUG 05-06 16:00:03.501898.501898 cuda_h.py:27] end moe_vllm_forward cost 3.188 ms
DEBUG 05-06 16:00:03.501410.501410 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.501922.501922 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.477ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.501187.501187 cuda_h.py:27] end *layer_moe_fused cost 14.449 ms
DEBUG 05-06 16:00:03.501081.501081 cuda_h.py:27] end prefill_merge_scale cost 0.319 ms
DEBUG 05-06 16:00:03.501919.501919 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.036 ms
DEBUG 05-06 16:00:03.501603.501603 cuda_h.py:27] end prefill_layer cost 28.148 ms
DEBUG 05-06 16:00:03.502151.502151 lmp.py:1394] -------------------------------- end prefill layer 23 --------------------------------
DEBUG 05-06 16:00:03.502231.502231 lmp.py:1350] -------------------------------- start prefill layer 24 --------------------------------
DEBUG 05-06 16:00:03.502628.502628 cuda_h.py:27] end prefill_ln cost 0.160 ms
DEBUG 05-06 16:00:03.513381.513381 cuda_h.py:27] end prefill_attn cost 11.355 ms
DEBUG 05-06 16:00:03.514950.514950 cuda_h.py:27] end prefill_ffn_prep cost 0.324 ms
DEBUG 05-06 16:00:03.515125.515125 cuda_h.py:27] end prefill_gate cost 0.293 ms
INFO 05-06 16:00:03.520222.520222 lmp.py:1823] [layer_moe_fused] layer=24 active_experts=96 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 31, 35, 43, 47, 51, 55, 59, 63, 67, 75, 79, 83, 99, 103, 111, 119, 123, 127], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 5917, 'token_per_expert': {3: 88, 7: 4, 11: 1, 15: 80, 19: 4, 23: 301, 31: 169, 35: 1, 43: 2161, 47: 458, 51: 184, 55: 22, 59: 166, 63: 82, 67: 76, 75: 314, 79: 8, 83: 647, 99: 18, 103: 3, 111: 188, 119: 5, 123: 2, 127: 935}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 60, 68, 72, 76, 80, 88, 92, 100, 104, 108, 112, 116, 120, 124], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 6657, 'token_per_expert': {0: 265, 4: 443, 8: 223, 12: 186, 16: 39, 20: 1, 24: 228, 28: 641, 32: 3, 40: 2, 44: 82, 60: 14, 68: 5, 72: 142, 76: 1414, 80: 279, 88: 6, 92: 38, 100: 2073, 104: 21, 108: 4, 112: 414, 116: 92, 120: 1, 124: 41}}
experts_gpu_alloc_device_2 {'expert_ids': [5, 9, 13, 17, 29, 37, 41, 45, 49, 53, 57, 73, 77, 85, 93, 97, 101, 105, 109, 117, 121, 125], 'expert_count': 22, 'ideal_gpu_count': 24, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 9044, 'token_per_expert': {5: 1, 9: 28, 13: 52, 17: 92, 29: 197, 37: 91, 41: 99, 45: 173, 49: 1421, 53: 8, 57: 1, 73: 55, 77: 20, 85: 7, 93: 36, 97: 3318, 101: 92, 105: 123, 109: 2400, 117: 145, 121: 449, 125: 236}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 18, 30, 34, 38, 42, 46, 50, 62, 66, 70, 74, 78, 82, 90, 98, 102, 106, 110, 114, 118, 122, 126], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 11150, 'token_per_expert': {2: 12, 10: 307, 14: 306, 18: 1982, 30: 1329, 34: 170, 38: 6, 42: 56, 46: 468, 50: 546, 62: 246, 66: 375, 70: 302, 74: 713, 78: 192, 82: 17, 90: 29, 98: 219, 102: 67, 106: 4, 110: 17, 114: 2164, 118: 556, 122: 2, 126: 1065}}
INFO 05-06 16:00:03.520735.520735 lmp.py:1845] [layer_moe_fused] layer=24 prefix: 4.940ms alloc: 0.216ms
INFO 05-06 16:00:03.520756.520756 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.839897155761719e-05 seconds
INFO 05-06 16:00:03.522236.522236 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001439809799194336s
INFO 05-06 16:00:03.523731.523731 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010762214660644531s
DEBUG 05-06 16:00:03.523985.523985 cuda_h.py:27] end moe_wait_copy_tasks cost 1.162 ms
DEBUG 05-06 16:00:03.526917.526917 cuda_h.py:27] end moe_vllm_forward cost 3.123 ms
DEBUG 05-06 16:00:03.526443.526443 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.526452.526452 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.437ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.526611.526611 cuda_h.py:27] end *layer_moe_fused cost 11.737 ms
DEBUG 05-06 16:00:03.527108.527108 cuda_h.py:27] end prefill_merge_scale cost 0.311 ms
DEBUG 05-06 16:00:03.527799.527799 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:03.527291.527291 cuda_h.py:27] end prefill_layer cost 25.324 ms
DEBUG 05-06 16:00:03.527022.527022 lmp.py:1394] -------------------------------- end prefill layer 24 --------------------------------
DEBUG 05-06 16:00:03.527771.527771 lmp.py:1350] -------------------------------- start prefill layer 25 --------------------------------
DEBUG 05-06 16:00:03.527651.527651 cuda_h.py:27] end prefill_ln cost 0.164 ms
DEBUG 05-06 16:00:03.539806.539806 cuda_h.py:27] end prefill_attn cost 11.273 ms
DEBUG 05-06 16:00:03.539102.539102 cuda_h.py:27] end prefill_ffn_prep cost 0.470 ms
DEBUG 05-06 16:00:03.540580.540580 cuda_h.py:27] end prefill_gate cost 0.315 ms
INFO 05-06 16:00:03.544925.544925 lmp.py:1823] [layer_moe_fused] layer=25 active_experts=104 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 19, 27, 31, 35, 39, 47, 55, 59, 63, 67, 75, 79, 87, 95, 99, 107, 111, 115, 119, 123], 'expert_count': 23, 'ideal_gpu_count': 26, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 7902, 'token_per_expert': {3: 5, 7: 874, 11: 1001, 19: 51, 27: 53, 31: 82, 35: 24, 39: 231, 47: 431, 55: 27, 59: 137, 63: 974, 67: 21, 75: 70, 79: 68, 87: 20, 95: 462, 99: 1758, 107: 45, 111: 702, 115: 710, 119: 146, 123: 10}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 92, 96, 100, 104, 120, 124], 'expert_count': 27, 'ideal_gpu_count': 26, 'keep_on_gpu': 27, 'hit_count_on_device': 27, 'token_total': 6010, 'token_per_expert': {0: 3, 4: 5, 8: 13, 12: 32, 16: 62, 20: 16, 24: 3, 32: 211, 36: 564, 40: 2014, 44: 124, 48: 12, 52: 242, 56: 302, 60: 54, 64: 4, 68: 24, 72: 246, 76: 4, 80: 399, 84: 172, 92: 544, 96: 1, 100: 783, 104: 11, 120: 74, 124: 91}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 9, 13, 17, 21, 25, 29, 37, 45, 49, 53, 57, 65, 69, 73, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'expert_count': 26, 'ideal_gpu_count': 26, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 9248, 'token_per_expert': {1: 6, 9: 47, 13: 164, 17: 26, 21: 968, 25: 418, 29: 8, 37: 847, 45: 514, 49: 15, 53: 1051, 57: 19, 65: 1, 69: 1019, 73: 37, 85: 10, 89: 341, 93: 143, 97: 7, 101: 10, 105: 70, 109: 149, 113: 206, 117: 1142, 121: 2027, 125: 3}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 18, 22, 26, 30, 38, 42, 46, 50, 54, 58, 62, 66, 74, 78, 82, 86, 94, 98, 102, 106, 110, 118, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 26, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 9608, 'token_per_expert': {2: 188, 6: 87, 10: 610, 14: 105, 18: 1, 22: 141, 26: 83, 30: 20, 38: 2, 42: 3584, 46: 3, 50: 23, 54: 1, 58: 50, 62: 411, 66: 136, 74: 1401, 78: 324, 82: 1377, 86: 54, 94: 428, 98: 4, 102: 53, 106: 3, 110: 1, 118: 480, 122: 12, 126: 26}}
INFO 05-06 16:00:03.545696.545696 lmp.py:1845] [layer_moe_fused] layer=25 prefix: 4.031ms alloc: 0.234ms
INFO 05-06 16:00:03.545528.545528 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 5.888938903808594e-05 seconds
INFO 05-06 16:00:03.547293.547293 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014171600341796875s
INFO 05-06 16:00:03.548845.548845 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010128021240234375s
DEBUG 05-06 16:00:03.548530.548530 cuda_h.py:27] end moe_wait_copy_tasks cost 1.099 ms
DEBUG 05-06 16:00:03.551128.551128 cuda_h.py:27] end moe_vllm_forward cost 3.229 ms
DEBUG 05-06 16:00:03.551879.551879 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.551060.551060 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.518ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.552741.552741 cuda_h.py:27] end *layer_moe_fused cost 11.146 ms
DEBUG 05-06 16:00:03.552047.552047 cuda_h.py:27] end prefill_merge_scale cost 0.312 ms
DEBUG 05-06 16:00:03.552170.552170 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.035 ms
DEBUG 05-06 16:00:03.552900.552900 cuda_h.py:27] end prefill_layer cost 24.813 ms
DEBUG 05-06 16:00:03.552236.552236 lmp.py:1394] -------------------------------- end prefill layer 25 --------------------------------
DEBUG 05-06 16:00:03.552316.552316 lmp.py:1350] -------------------------------- start prefill layer 26 --------------------------------
DEBUG 05-06 16:00:03.552190.552190 cuda_h.py:27] end prefill_ln cost 0.160 ms
DEBUG 05-06 16:00:03.564231.564231 cuda_h.py:27] end prefill_attn cost 11.253 ms
DEBUG 05-06 16:00:03.564098.564098 cuda_h.py:27] end prefill_ffn_prep cost 0.295 ms
DEBUG 05-06 16:00:03.565636.565636 cuda_h.py:27] end prefill_gate cost 0.294 ms
INFO 05-06 16:00:03.570573.570573 lmp.py:1823] [layer_moe_fused] layer=26 active_experts=96 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 23, 27, 31, 39, 43, 47, 51, 55, 59, 67, 79, 83, 87, 99, 103, 107, 111, 115, 123, 127], 'expert_count': 23, 'ideal_gpu_count': 24, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 6098, 'token_per_expert': {3: 1, 7: 533, 11: 9, 23: 14, 27: 21, 31: 5, 39: 125, 43: 21, 47: 2, 51: 528, 55: 3, 59: 2, 67: 358, 79: 5, 83: 3, 87: 846, 99: 30, 103: 21, 107: 125, 111: 666, 115: 209, 123: 1308, 127: 1263}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 16, 20, 24, 36, 44, 60, 64, 68, 76, 80, 84, 88, 92, 96, 112, 120], 'expert_count': 20, 'ideal_gpu_count': 24, 'keep_on_gpu': 20, 'hit_count_on_device': 20, 'token_total': 4559, 'token_per_expert': {0: 3, 4: 7, 8: 1, 12: 63, 16: 85, 20: 192, 24: 133, 36: 165, 44: 3, 60: 329, 64: 30, 68: 128, 76: 31, 80: 5, 84: 13, 88: 15, 92: 38, 96: 302, 112: 147, 120: 2869}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 13, 21, 25, 29, 33, 37, 45, 49, 57, 61, 65, 69, 81, 85, 89, 93, 97, 105, 109, 113, 117, 121, 125], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 11072, 'token_per_expert': {1: 248, 5: 1243, 13: 367, 21: 3613, 25: 10, 29: 2617, 33: 79, 37: 319, 45: 98, 49: 2, 57: 1, 61: 356, 65: 71, 69: 3, 81: 13, 85: 1608, 89: 90, 93: 113, 97: 2, 105: 12, 109: 12, 113: 172, 117: 11, 121: 1, 125: 11}}
experts_gpu_alloc_device_3 {'expert_ids': [6, 10, 14, 18, 22, 26, 30, 38, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 122, 126], 'expert_count': 28, 'ideal_gpu_count': 24, 'keep_on_gpu': 28, 'hit_count_on_device': 28, 'token_total': 11039, 'token_per_expert': {6: 887, 10: 310, 14: 3715, 18: 214, 22: 324, 26: 177, 30: 31, 38: 3, 46: 580, 50: 1, 54: 4, 58: 3, 62: 222, 66: 92, 70: 20, 74: 54, 78: 940, 82: 88, 86: 1613, 90: 798, 94: 25, 98: 187, 102: 58, 106: 1, 110: 156, 114: 3, 122: 393, 126: 140}}
INFO 05-06 16:00:03.570377.570377 lmp.py:1845] [layer_moe_fused] layer=26 prefix: 4.896ms alloc: 0.218ms
INFO 05-06 16:00:03.571427.571427 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.744529724121094e-05 seconds
INFO 05-06 16:00:03.572800.572800 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.001607656478881836s
INFO 05-06 16:00:03.573611.573611 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0010352134704589844s
DEBUG 05-06 16:00:03.573481.573481 cuda_h.py:27] end moe_wait_copy_tasks cost 1.118 ms
DEBUG 05-06 16:00:03.577314.577314 cuda_h.py:27] end moe_vllm_forward cost 3.122 ms
DEBUG 05-06 16:00:03.577303.577303 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.577100.577100 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.409ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.577040.577040 cuda_h.py:27] end *layer_moe_fused cost 11.854 ms
DEBUG 05-06 16:00:03.578315.578315 cuda_h.py:27] end prefill_merge_scale cost 0.310 ms
DEBUG 05-06 16:00:03.578053.578053 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:03.578829.578829 cuda_h.py:27] end prefill_layer cost 25.412 ms
DEBUG 05-06 16:00:03.578768.578768 lmp.py:1394] -------------------------------- end prefill layer 26 --------------------------------
DEBUG 05-06 16:00:03.578610.578610 lmp.py:1350] -------------------------------- start prefill layer 27 --------------------------------
DEBUG 05-06 16:00:03.578007.578007 cuda_h.py:27] end prefill_ln cost 0.159 ms
DEBUG 05-06 16:00:03.589134.589134 cuda_h.py:27] end prefill_attn cost 11.243 ms
DEBUG 05-06 16:00:03.590854.590854 cuda_h.py:27] end prefill_ffn_prep cost 0.293 ms
DEBUG 05-06 16:00:03.591108.591108 cuda_h.py:27] end prefill_gate cost 0.293 ms
INFO 05-06 16:00:03.595186.595186 lmp.py:1823] [layer_moe_fused] layer=27 active_experts=98 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 19, 23, 27, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 99, 103, 107, 111, 115, 119, 123, 127], 'expert_count': 29, 'ideal_gpu_count': 25, 'keep_on_gpu': 29, 'hit_count_on_device': 29, 'token_total': 8587, 'token_per_expert': {3: 5, 7: 2230, 11: 24, 15: 42, 19: 93, 23: 503, 27: 1401, 39: 7, 43: 7, 47: 454, 51: 219, 55: 1, 59: 1616, 63: 47, 67: 43, 71: 170, 75: 54, 79: 10, 83: 22, 87: 274, 91: 59, 99: 8, 103: 196, 107: 76, 111: 585, 115: 11, 119: 50, 123: 284, 127: 96}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 8, 12, 16, 20, 28, 36, 40, 44, 48, 56, 64, 68, 72, 76, 80, 84, 88, 92, 96, 104, 112, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 25, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 12568, 'token_per_expert': {0: 2, 8: 1060, 12: 76, 16: 839, 20: 62, 28: 23, 36: 970, 40: 49, 44: 51, 48: 166, 56: 69, 64: 2382, 68: 135, 72: 4, 76: 105, 80: 2329, 84: 16, 88: 4, 92: 9, 96: 646, 104: 661, 112: 1, 120: 7, 124: 2902}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 21, 25, 29, 37, 49, 53, 57, 61, 69, 77, 81, 85, 89, 93, 97, 101, 109, 113, 117, 125], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 5425, 'token_per_expert': {1: 111, 5: 2, 9: 155, 13: 461, 21: 1, 25: 1020, 29: 11, 37: 69, 49: 2, 53: 4, 57: 540, 61: 577, 69: 56, 77: 114, 81: 184, 85: 789, 89: 427, 93: 776, 97: 11, 101: 35, 109: 11, 113: 45, 117: 9, 125: 15}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 10, 14, 18, 22, 34, 42, 50, 54, 58, 62, 66, 74, 82, 86, 90, 94, 110, 114, 118, 126], 'expert_count': 21, 'ideal_gpu_count': 24, 'keep_on_gpu': 21, 'hit_count_on_device': 21, 'token_total': 6188, 'token_per_expert': {2: 1756, 10: 11, 14: 220, 18: 655, 22: 287, 34: 37, 42: 1, 50: 115, 54: 315, 58: 430, 62: 257, 66: 315, 74: 95, 82: 66, 86: 2, 90: 1374, 94: 8, 110: 74, 114: 155, 118: 3, 126: 12}}
INFO 05-06 16:00:03.595454.595454 lmp.py:1845] [layer_moe_fused] layer=27 prefix: 4.152ms alloc: 0.249ms
INFO 05-06 16:00:03.596476.596476 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 0.0001347064971923828 seconds
INFO 05-06 16:00:03.597207.597207 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014870166778564453s
INFO 05-06 16:00:03.598905.598905 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001024007797241211s
DEBUG 05-06 16:00:03.598398.598398 cuda_h.py:27] end moe_wait_copy_tasks cost 1.109 ms
DEBUG 05-06 16:00:03.602441.602441 cuda_h.py:27] end moe_vllm_forward cost 3.067 ms
DEBUG 05-06 16:00:03.602569.602569 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.602743.602743 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.348ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.602346.602346 cuda_h.py:27] end *layer_moe_fused cost 11.240 ms
DEBUG 05-06 16:00:03.603417.603417 cuda_h.py:27] end prefill_merge_scale cost 0.308 ms
DEBUG 05-06 16:00:03.603393.603393 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:03.603362.603362 cuda_h.py:27] end prefill_layer cost 24.815 ms
DEBUG 05-06 16:00:03.603264.603264 lmp.py:1394] -------------------------------- end prefill layer 27 --------------------------------
DEBUG 05-06 16:00:03.603106.603106 lmp.py:1350] -------------------------------- start prefill layer 28 --------------------------------
DEBUG 05-06 16:00:03.603661.603661 cuda_h.py:27] end prefill_ln cost 0.164 ms
DEBUG 05-06 16:00:03.614707.614707 cuda_h.py:27] end prefill_attn cost 10.947 ms
DEBUG 05-06 16:00:03.615899.615899 cuda_h.py:27] end prefill_ffn_prep cost 0.326 ms
DEBUG 05-06 16:00:03.616441.616441 cuda_h.py:27] end prefill_gate cost 0.295 ms
INFO 05-06 16:00:03.621567.621567 lmp.py:1823] [layer_moe_fused] layer=28 active_experts=99 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 23, 27, 31, 35, 39, 43, 55, 59, 63, 79, 87, 91, 99, 111, 115, 119, 123, 127], 'expert_count': 22, 'ideal_gpu_count': 25, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 8627, 'token_per_expert': {3: 958, 7: 36, 11: 38, 15: 1137, 23: 114, 27: 90, 31: 26, 35: 2374, 39: 273, 43: 398, 55: 514, 59: 23, 63: 30, 79: 1141, 87: 83, 91: 185, 99: 30, 111: 371, 115: 529, 119: 7, 123: 269, 127: 1}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 8, 12, 20, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 84, 88, 92, 96, 104, 112, 116, 124], 'expert_count': 26, 'ideal_gpu_count': 25, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 7246, 'token_per_expert': {0: 263, 4: 1, 8: 31, 12: 50, 20: 6, 24: 908, 32: 1160, 36: 1222, 40: 1228, 44: 8, 48: 336, 52: 1015, 56: 176, 60: 105, 64: 2, 68: 7, 72: 18, 76: 279, 84: 109, 88: 64, 92: 7, 96: 27, 104: 214, 112: 1, 116: 3, 124: 6}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 29, 33, 45, 49, 53, 57, 65, 73, 77, 81, 85, 89, 93, 97, 105, 109, 113, 117, 121], 'expert_count': 26, 'ideal_gpu_count': 25, 'keep_on_gpu': 26, 'hit_count_on_device': 26, 'token_total': 13131, 'token_per_expert': {1: 96, 5: 207, 9: 61, 13: 18, 17: 30, 21: 14, 25: 3, 29: 16, 33: 213, 45: 3639, 49: 782, 53: 107, 57: 394, 65: 2200, 73: 5, 77: 214, 81: 216, 85: 80, 89: 661, 93: 7, 97: 51, 105: 37, 109: 1641, 113: 2103, 117: 212, 121: 124}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 14, 18, 22, 26, 30, 34, 38, 42, 50, 58, 62, 70, 74, 78, 82, 90, 98, 102, 106, 110, 114, 118, 122], 'expert_count': 25, 'ideal_gpu_count': 24, 'keep_on_gpu': 25, 'hit_count_on_device': 25, 'token_total': 3764, 'token_per_expert': {2: 92, 6: 23, 14: 5, 18: 333, 22: 2, 26: 13, 30: 190, 34: 55, 38: 78, 42: 5, 50: 13, 58: 288, 62: 2, 70: 1486, 74: 13, 78: 56, 82: 33, 90: 266, 98: 6, 102: 90, 106: 199, 110: 7, 114: 2, 118: 234, 122: 273}}
INFO 05-06 16:00:03.621245.621245 lmp.py:1845] [layer_moe_fused] layer=28 prefix: 4.786ms alloc: 0.234ms
INFO 05-06 16:00:03.621029.621029 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.76837158203125e-05 seconds
INFO 05-06 16:00:03.623217.623217 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0014355182647705078s
INFO 05-06 16:00:03.624987.624987 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.0011734962463378906s
DEBUG 05-06 16:00:03.624241.624241 cuda_h.py:27] end moe_wait_copy_tasks cost 1.258 ms
DEBUG 05-06 16:00:03.627724.627724 cuda_h.py:27] end moe_vllm_forward cost 3.143 ms
DEBUG 05-06 16:00:03.627759.627759 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.627033.627033 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.431ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.628298.628298 cuda_h.py:27] end *layer_moe_fused cost 11.731 ms
DEBUG 05-06 16:00:03.628413.628413 cuda_h.py:27] end prefill_merge_scale cost 0.313 ms
DEBUG 05-06 16:00:03.628297.628297 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.035 ms
DEBUG 05-06 16:00:03.628027.628027 cuda_h.py:27] end prefill_layer cost 25.327 ms
DEBUG 05-06 16:00:03.629240.629240 lmp.py:1394] -------------------------------- end prefill layer 28 --------------------------------
DEBUG 05-06 16:00:03.629320.629320 lmp.py:1350] -------------------------------- start prefill layer 29 --------------------------------
DEBUG 05-06 16:00:03.629863.629863 cuda_h.py:27] end prefill_ln cost 0.162 ms
DEBUG 05-06 16:00:03.640148.640148 cuda_h.py:27] end prefill_attn cost 11.591 ms
DEBUG 05-06 16:00:03.641678.641678 cuda_h.py:27] end prefill_ffn_prep cost 0.322 ms
DEBUG 05-06 16:00:03.642166.642166 cuda_h.py:27] end prefill_gate cost 0.289 ms
INFO 05-06 16:00:03.649202.649202 lmp.py:1823] [layer_moe_fused] layer=29 active_experts=93 (nonzero tokens)
experts_cpu_alloc {'expert_ids': [], 'token_total': 0, 'token_per_expert': {}}
experts_gpu_alloc_device_0 {'expert_ids': [3, 7, 11, 15, 31, 35, 43, 47, 51, 55, 59, 63, 71, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119], 'expert_count': 24, 'ideal_gpu_count': 24, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 4606, 'token_per_expert': {3: 31, 7: 310, 11: 104, 15: 39, 31: 1, 35: 31, 43: 22, 47: 2, 51: 1150, 55: 48, 59: 55, 63: 693, 71: 2, 79: 296, 83: 257, 87: 6, 91: 3, 95: 7, 99: 25, 103: 185, 107: 270, 111: 656, 115: 46, 119: 367}}
experts_gpu_alloc_device_1 {'expert_ids': [0, 4, 12, 24, 28, 36, 40, 44, 48, 56, 60, 64, 72, 76, 80, 84, 92, 96, 100, 104, 108, 112, 120, 124], 'expert_count': 24, 'ideal_gpu_count': 23, 'keep_on_gpu': 24, 'hit_count_on_device': 24, 'token_total': 8852, 'token_per_expert': {0: 674, 4: 33, 12: 7, 24: 1, 28: 10, 36: 479, 40: 993, 44: 20, 48: 762, 56: 383, 60: 614, 64: 1, 72: 220, 76: 82, 80: 20, 84: 1, 92: 62, 96: 71, 100: 3, 104: 39, 108: 29, 112: 3811, 120: 2, 124: 535}}
experts_gpu_alloc_device_2 {'expert_ids': [1, 5, 9, 13, 17, 21, 25, 33, 41, 45, 53, 57, 69, 77, 85, 89, 93, 97, 101, 105, 109, 113, 121], 'expert_count': 23, 'ideal_gpu_count': 23, 'keep_on_gpu': 23, 'hit_count_on_device': 23, 'token_total': 10805, 'token_per_expert': {1: 67, 5: 115, 9: 85, 13: 201, 17: 3448, 21: 16, 25: 9, 33: 175, 41: 27, 45: 16, 53: 5, 57: 439, 69: 50, 77: 1152, 85: 1121, 89: 139, 93: 622, 97: 2, 101: 32, 105: 2297, 109: 47, 113: 12, 121: 728}}
experts_gpu_alloc_device_3 {'expert_ids': [2, 6, 10, 14, 30, 38, 42, 46, 54, 58, 62, 78, 82, 90, 94, 98, 102, 106, 110, 114, 118, 126], 'expert_count': 22, 'ideal_gpu_count': 23, 'keep_on_gpu': 22, 'hit_count_on_device': 22, 'token_total': 8505, 'token_per_expert': {2: 877, 6: 296, 10: 219, 14: 1, 30: 2470, 38: 55, 42: 6, 46: 7, 54: 63, 58: 30, 62: 13, 78: 809, 82: 21, 90: 46, 94: 68, 98: 4, 102: 37, 106: 6, 110: 2764, 114: 652, 118: 5, 126: 56}}
INFO 05-06 16:00:03.649059.649059 lmp.py:1845] [layer_moe_fused] layer=29 prefix: 7.070ms alloc: 0.226ms
INFO 05-06 16:00:03.650301.650301 lmp.py:1859] [layer_moe_fused] get_experts_task_ids time: 4.9114227294921875e-05 seconds
INFO 05-06 16:00:03.651828.651828 lmp.py:1867] [layer_moe_fused] submit_high_priority_copy_tasks ok=True pending=0 time: 0.0012657642364501953s
INFO 05-06 16:00:03.652414.652414 lmp.py:1910] [layer_moe_fused] wait_copy_tasks ok=True pending=0 time: 0.001047372817993164s
DEBUG 05-06 16:00:03.652523.652523 cuda_h.py:27] end moe_wait_copy_tasks cost 1.130 ms
DEBUG 05-06 16:00:03.655824.655824 cuda_h.py:27] end moe_vllm_forward cost 3.080 ms
DEBUG 05-06 16:00:03.655098.655098 cuda_h.py:27] end moe_shared_experts cost 0.003 ms
INFO 05-06 16:00:03.655279.655279 lmp.py:1964] [layer_moe_fused] vllm triton time: 3.371ms (seq_len=512 cg=False)
DEBUG 05-06 16:00:03.656305.656305 cuda_h.py:27] end *layer_moe_fused cost 13.659 ms
DEBUG 05-06 16:00:03.656257.656257 cuda_h.py:27] end prefill_merge_scale cost 0.304 ms
DEBUG 05-06 16:00:03.656041.656041 cuda_h.py:27] end prefill_merge_scale_apply_decoder_layer_scale cost 0.034 ms
DEBUG 05-06 16:00:03.656771.656771 cuda_h.py:27] end prefill_layer cost 27.719 ms
DEBUG 05-06 16:00:03.656453.656453 lmp.py:1394] -------------------------------- end prefill layer 29 --------------------------------
DEBUG 05-06 16:00:03.657536.657536 cuda_h.py:27] end prefill_step cost 1483.649 ms
INFO 05-06 16:00:03.657976.657976 lmp.py:1397] prefill time: 1.6121063232421875 seconds
INFO 05-06 16:00:03.694736.694736 lmp.py:1409] Static-KV prefill complete; seqlens set to 512.
DEBUG 05-06 16:00:03.702339.702339 cuda_h.py:27] end init_inputs_tokens cost 7.422 ms
DEBUG 05-06 16:00:03.702997.702997 lmp.py:1510] decode step 0 next_inputs_tokens shape=(8, 1, 2816)
DEBUG 05-06 16:00:03.702628.702628 lmp.py:1516] ---- decode step 0 layer 0 ----
DEBUG 05-06 16:00:03.708497.708497 cuda_h.py:27] end decode_layer cost 5.796 ms
DEBUG 05-06 16:00:03.708009.708009 lmp.py:1516] ---- decode step 0 layer 1 ----
DEBUG 05-06 16:00:03.713320.713320 cuda_h.py:27] end decode_layer cost 5.104 ms
DEBUG 05-06 16:00:03.713639.713639 lmp.py:1516] ---- decode step 0 layer 2 ----
DEBUG 05-06 16:00:03.718556.718556 cuda_h.py:27] end decode_layer cost 5.024 ms
DEBUG 05-06 16:00:03.718399.718399 lmp.py:1516] ---- decode step 0 layer 3 ----
DEBUG 05-06 16:00:03.723334.723334 cuda_h.py:27] end decode_layer cost 4.967 ms
DEBUG 05-06 16:00:03.723039.723039 lmp.py:1516] ---- decode step 0 layer 4 ----
DEBUG 05-06 16:00:03.728504.728504 cuda_h.py:27] end decode_layer cost 4.972 ms
DEBUG 05-06 16:00:03.728301.728301 lmp.py:1516] ---- decode step 0 layer 5 ----
DEBUG 05-06 16:00:03.734681.734681 cuda_h.py:27] end decode_layer cost 6.208 ms
DEBUG 05-06 16:00:03.735253.735253 lmp.py:1516] ---- decode step 0 layer 6 ----
DEBUG 05-06 16:00:03.740938.740938 cuda_h.py:27] end decode_layer cost 5.030 ms
DEBUG 05-06 16:00:03.740735.740735 lmp.py:1516] ---- decode step 0 layer 7 ----
DEBUG 05-06 16:00:03.745203.745203 cuda_h.py:27] end decode_layer cost 5.080 ms
DEBUG 05-06 16:00:03.745139.745139 lmp.py:1516] ---- decode step 0 layer 8 ----
DEBUG 05-06 16:00:03.750676.750676 cuda_h.py:27] end decode_layer cost 4.955 ms
DEBUG 05-06 16:00:03.750956.750956 lmp.py:1516] ---- decode step 0 layer 9 ----
DEBUG 05-06 16:00:03.755644.755644 cuda_h.py:27] end decode_layer cost 5.101 ms
DEBUG 05-06 16:00:03.755891.755891 lmp.py:1516] ---- decode step 0 layer 10 ----
DEBUG 05-06 16:00:03.760288.760288 cuda_h.py:27] end decode_layer cost 4.922 ms
DEBUG 05-06 16:00:03.760469.760469 lmp.py:1516] ---- decode step 0 layer 11 ----
DEBUG 05-06 16:00:03.765001.765001 cuda_h.py:27] end decode_layer cost 5.197 ms
DEBUG 05-06 16:00:03.765460.765460 lmp.py:1516] ---- decode step 0 layer 12 ----
DEBUG 05-06 16:00:03.770780.770780 cuda_h.py:27] end decode_layer cost 5.005 ms
DEBUG 05-06 16:00:03.770861.770861 lmp.py:1516] ---- decode step 0 layer 13 ----
DEBUG 05-06 16:00:03.775139.775139 cuda_h.py:27] end decode_layer cost 4.940 ms
DEBUG 05-06 16:00:03.775459.775459 lmp.py:1516] ---- decode step 0 layer 14 ----
DEBUG 05-06 16:00:03.780939.780939 cuda_h.py:27] end decode_layer cost 5.018 ms
DEBUG 05-06 16:00:03.780212.780212 lmp.py:1516] ---- decode step 0 layer 15 ----
DEBUG 05-06 16:00:03.785550.785550 cuda_h.py:27] end decode_layer cost 4.949 ms
DEBUG 05-06 16:00:03.785724.785724 lmp.py:1516] ---- decode step 0 layer 16 ----
DEBUG 05-06 16:00:03.790840.790840 cuda_h.py:27] end decode_layer cost 5.030 ms
DEBUG 05-06 16:00:03.790398.790398 lmp.py:1516] ---- decode step 0 layer 17 ----
DEBUG 05-06 16:00:03.796632.796632 cuda_h.py:27] end decode_layer cost 5.187 ms
DEBUG 05-06 16:00:03.796429.796429 lmp.py:1516] ---- decode step 0 layer 18 ----
DEBUG 05-06 16:00:03.801497.801497 cuda_h.py:27] end decode_layer cost 4.996 ms
DEBUG 05-06 16:00:03.801101.801101 lmp.py:1516] ---- decode step 0 layer 19 ----
DEBUG 05-06 16:00:03.806158.806158 cuda_h.py:27] end decode_layer cost 5.058 ms
DEBUG 05-06 16:00:03.806525.806525 lmp.py:1516] ---- decode step 0 layer 20 ----
DEBUG 05-06 16:00:03.811036.811036 cuda_h.py:27] end decode_layer cost 4.971 ms
DEBUG 05-06 16:00:03.811210.811210 lmp.py:1516] ---- decode step 0 layer 21 ----
DEBUG 05-06 16:00:03.816726.816726 cuda_h.py:27] end decode_layer cost 4.940 ms
DEBUG 05-06 16:00:03.816046.816046 lmp.py:1516] ---- decode step 0 layer 22 ----
DEBUG 05-06 16:00:03.821457.821457 cuda_h.py:27] end decode_layer cost 4.932 ms
DEBUG 05-06 16:00:03.821843.821843 lmp.py:1516] ---- decode step 0 layer 23 ----
DEBUG 05-06 16:00:03.826488.826488 cuda_h.py:27] end decode_layer cost 5.210 ms
DEBUG 05-06 16:00:03.826708.826708 lmp.py:1516] ---- decode step 0 layer 24 ----
DEBUG 05-06 16:00:03.831691.831691 cuda_h.py:27] end decode_layer cost 5.002 ms
DEBUG 05-06 16:00:03.831249.831249 lmp.py:1516] ---- decode step 0 layer 25 ----
DEBUG 05-06 16:00:03.836410.836410 cuda_h.py:27] end decode_layer cost 4.993 ms
DEBUG 05-06 16:00:03.836028.836028 lmp.py:1516] ---- decode step 0 layer 26 ----
DEBUG 05-06 16:00:03.841398.841398 cuda_h.py:27] end decode_layer cost 4.902 ms
DEBUG 05-06 16:00:03.841618.841618 lmp.py:1516] ---- decode step 0 layer 27 ----
DEBUG 05-06 16:00:03.846267.846267 cuda_h.py:27] end decode_layer cost 4.932 ms
DEBUG 05-06 16:00:03.846633.846633 lmp.py:1516] ---- decode step 0 layer 28 ----
DEBUG 05-06 16:00:03.851806.851806 cuda_h.py:27] end decode_layer cost 4.968 ms
DEBUG 05-06 16:00:03.851457.851457 lmp.py:1516] ---- decode step 0 layer 29 ----
DEBUG 05-06 16:00:03.857704.857704 cuda_h.py:27] end decode_layer cost 5.197 ms
DEBUG 05-06 16:00:03.857509.857509 cuda_h.py:27] end decode_step cost 162.221 ms
INFO 05-06 16:00:03.857179.857179 lmp.py:1564] decode step 0 time: 0.1622633934020996 seconds
DEBUG 05-06 16:00:03.863338.863338 cuda_h.py:27] end init_inputs_tokens cost 6.653 ms
DEBUG 05-06 16:00:03.864088.864088 lmp.py:1510] decode step 1 next_inputs_tokens shape=(8, 1, 2816)
DEBUG 05-06 16:00:03.864427.864427 lmp.py:1516] ---- decode step 1 layer 0 ----
DEBUG 05-06 16:00:03.868307.868307 cuda_h.py:27] end decode_layer cost 4.891 ms
DEBUG 05-06 16:00:03.868719.868719 lmp.py:1516] ---- decode step 1 layer 1 ----
DEBUG 05-06 16:00:03.874231.874231 cuda_h.py:27] end decode_layer cost 5.007 ms
DEBUG 05-06 16:00:03.874882.874882 lmp.py:1516] ---- decode step 1 layer 2 ----
DEBUG 05-06 16:00:03.879286.879286 cuda_h.py:27] end decode_layer cost 4.927 ms
DEBUG 05-06 16:00:03.879222.879222 lmp.py:1516] ---- decode step 1 layer 3 ----
DEBUG 05-06 16:00:03.884551.884551 cuda_h.py:27] end decode_layer cost 5.083 ms
DEBUG 05-06 16:00:03.884109.884109 lmp.py:1516] ---- decode step 1 layer 4 ----
DEBUG 05-06 16:00:03.889751.889751 cuda_h.py:27] end decode_layer cost 4.928 ms
DEBUG 05-06 16:00:03.889448.889448 lmp.py:1516] ---- decode step 1 layer 5 ----
DEBUG 05-06 16:00:03.894848.894848 cuda_h.py:27] end decode_layer cost 5.205 ms
DEBUG 05-06 16:00:03.894784.894784 lmp.py:1516] ---- decode step 1 layer 6 ----
DEBUG 05-06 16:00:03.899934.899934 cuda_h.py:27] end decode_layer cost 4.881 ms
DEBUG 05-06 16:00:03.899631.899631 lmp.py:1516] ---- decode step 1 layer 7 ----
DEBUG 05-06 16:00:03.905887.905887 cuda_h.py:27] end decode_layer cost 5.804 ms
DEBUG 05-06 16:00:03.905334.905334 lmp.py:1516] ---- decode step 1 layer 8 ----
DEBUG 05-06 16:00:03.910922.910922 cuda_h.py:27] end decode_layer cost 5.273 ms
DEBUG 05-06 16:00:03.910718.910718 lmp.py:1516] ---- decode step 1 layer 9 ----
DEBUG 05-06 16:00:03.915777.915777 cuda_h.py:27] end decode_layer cost 4.903 ms
DEBUG 05-06 16:00:03.915620.915620 lmp.py:1516] ---- decode step 1 layer 10 ----
DEBUG 05-06 16:00:03.920697.920697 cuda_h.py:27] end decode_layer cost 4.861 ms
DEBUG 05-06 16:00:03.920209.920209 lmp.py:1516] ---- decode step 1 layer 11 ----
DEBUG 05-06 16:00:03.925232.925232 cuda_h.py:27] end decode_layer cost 5.208 ms
DEBUG 05-06 16:00:03.925359.925359 lmp.py:1516] ---- decode step 1 layer 12 ----
DEBUG 05-06 16:00:03.930316.930316 cuda_h.py:27] end decode_layer cost 5.018 ms
DEBUG 05-06 16:00:03.930874.930874 lmp.py:1516] ---- decode step 1 layer 13 ----
DEBUG 05-06 16:00:03.936659.936659 cuda_h.py:27] end decode_layer cost 5.032 ms
DEBUG 05-06 16:00:03.936932.936932 lmp.py:1516] ---- decode step 1 layer 14 ----
DEBUG 05-06 16:00:03.940997.940997 cuda_h.py:27] end decode_layer cost 4.888 ms
DEBUG 05-06 16:00:03.941078.941078 lmp.py:1516] ---- decode step 1 layer 15 ----
DEBUG 05-06 16:00:03.946834.946834 cuda_h.py:27] end decode_layer cost 4.976 ms
DEBUG 05-06 16:00:03.946916.946916 lmp.py:1516] ---- decode step 1 layer 16 ----
DEBUG 05-06 16:00:03.951041.951041 cuda_h.py:27] end decode_layer cost 4.933 ms
DEBUG 05-06 16:00:03.951454.951454 lmp.py:1516] ---- decode step 1 layer 17 ----
DEBUG 05-06 16:00:03.956912.956912 cuda_h.py:27] end decode_layer cost 5.178 ms
DEBUG 05-06 16:00:03.956994.956994 lmp.py:1516] ---- decode step 1 layer 18 ----
DEBUG 05-06 16:00:03.961014.961014 cuda_h.py:27] end decode_layer cost 4.961 ms
DEBUG 05-06 16:00:03.961381.961381 lmp.py:1516] ---- decode step 1 layer 19 ----
DEBUG 05-06 16:00:03.966321.966321 cuda_h.py:27] end decode_layer cost 4.936 ms
DEBUG 05-06 16:00:03.966541.966541 lmp.py:1516] ---- decode step 1 layer 20 ----
DEBUG 05-06 16:00:03.971661.971661 cuda_h.py:27] end decode_layer cost 4.963 ms
DEBUG 05-06 16:00:03.971743.971743 lmp.py:1516] ---- decode step 1 layer 21 ----
DEBUG 05-06 16:00:03.976605.976605 cuda_h.py:27] end decode_layer cost 4.984 ms
DEBUG 05-06 16:00:03.976402.976402 lmp.py:1516] ---- decode step 1 layer 22 ----
DEBUG 05-06 16:00:03.981289.981289 cuda_h.py:27] end decode_layer cost 4.933 ms
DEBUG 05-06 16:00:03.981132.981132 lmp.py:1516] ---- decode step 1 layer 23 ----
DEBUG 05-06 16:00:03.986770.986770 cuda_h.py:27] end decode_layer cost 5.205 ms
DEBUG 05-06 16:00:03.986090.986090 lmp.py:1516] ---- decode step 1 layer 24 ----
DEBUG 05-06 16:00:03.991973.991973 cuda_h.py:27] end decode_layer cost 4.999 ms
DEBUG 05-06 16:00:03.991816.991816 lmp.py:1516] ---- decode step 1 layer 25 ----
DEBUG 05-06 16:00:03.996804.996804 cuda_h.py:27] end decode_layer cost 4.971 ms
DEBUG 05-06 16:00:03.996362.996362 lmp.py:1516] ---- decode step 1 layer 26 ----
DEBUG 05-06 16:00:04.001732.001732 cuda_h.py:27] end decode_layer cost 4.902 ms
DEBUG 05-06 16:00:04.001906.001906 lmp.py:1516] ---- decode step 1 layer 27 ----
DEBUG 05-06 16:00:04.006510.006510 cuda_h.py:27] end decode_layer cost 4.969 ms
DEBUG 05-06 16:00:04.006214.006214 lmp.py:1516] ---- decode step 1 layer 28 ----
DEBUG 05-06 16:00:04.011691.011691 cuda_h.py:27] end decode_layer cost 4.946 ms
DEBUG 05-06 16:00:04.011011.011011 lmp.py:1516] ---- decode step 1 layer 29 ----
DEBUG 05-06 16:00:04.017297.017297 cuda_h.py:27] end decode_layer cost 5.192 ms
DEBUG 05-06 16:00:04.017605.017605 cuda_h.py:27] end decode_step cost 159.804 ms
INFO 05-06 16:00:04.017752.017752 lmp.py:1564] decode step 1 time: 0.15984296798706055 seconds
DEBUG 05-06 16:00:04.023944.023944 cuda_h.py:27] end init_inputs_tokens cost 6.678 ms
DEBUG 05-06 16:00:04.023456.023456 lmp.py:1510] decode step 2 next_inputs_tokens shape=(8, 1, 2816)
DEBUG 05-06 16:00:04.023226.023226 lmp.py:1516] ---- decode step 2 layer 0 ----
DEBUG 05-06 16:00:04.028536.028536 cuda_h.py:27] end decode_layer cost 4.892 ms
DEBUG 05-06 16:00:04.028856.028856 lmp.py:1516] ---- decode step 2 layer 1 ----
DEBUG 05-06 16:00:04.033486.033486 cuda_h.py:27] end decode_layer cost 4.989 ms
DEBUG 05-06 16:00:04.033091.033091 lmp.py:1516] ---- decode step 2 layer 2 ----
DEBUG 05-06 16:00:04.038962.038962 cuda_h.py:27] end decode_layer cost 4.851 ms
DEBUG 05-06 16:00:04.038944.038944 lmp.py:1516] ---- decode step 2 layer 3 ----
DEBUG 05-06 16:00:04.043156.043156 cuda_h.py:27] end decode_layer cost 4.925 ms
DEBUG 05-06 16:00:04.043668.043668 lmp.py:1516] ---- decode step 2 layer 4 ----
DEBUG 05-06 16:00:04.048423.048423 cuda_h.py:27] end decode_layer cost 4.941 ms
DEBUG 05-06 16:00:04.048597.048597 lmp.py:1516] ---- decode step 2 layer 5 ----
DEBUG 05-06 16:00:04.054989.054989 cuda_h.py:27] end decode_layer cost 5.164 ms
DEBUG 05-06 16:00:04.054447.054447 lmp.py:1516] ---- decode step 2 layer 6 ----
DEBUG 05-06 16:00:04.059592.059592 cuda_h.py:27] end decode_layer cost 4.911 ms
DEBUG 05-06 16:00:04.059866.059866 lmp.py:1516] ---- decode step 2 layer 7 ----
DEBUG 05-06 16:00:04.064574.064574 cuda_h.py:27] end decode_layer cost 4.942 ms
DEBUG 05-06 16:00:04.064325.064325 lmp.py:1516] ---- decode step 2 layer 8 ----
DEBUG 05-06 16:00:04.069893.069893 cuda_h.py:27] end decode_layer cost 4.908 ms
DEBUG 05-06 16:00:04.069736.069736 lmp.py:1516] ---- decode step 2 layer 9 ----
DEBUG 05-06 16:00:04.074964.074964 cuda_h.py:27] end decode_layer cost 5.008 ms
DEBUG 05-06 16:00:04.074834.074834 lmp.py:1516] ---- decode step 2 layer 10 ----
DEBUG 05-06 16:00:04.079799.079799 cuda_h.py:27] end decode_layer cost 4.885 ms
DEBUG 05-06 16:00:04.079735.079735 lmp.py:1516] ---- decode step 2 layer 11 ----
DEBUG 05-06 16:00:04.085814.085814 cuda_h.py:27] end decode_layer cost 6.120 ms
DEBUG 05-06 16:00:04.085724.085724 lmp.py:1516] ---- decode step 2 layer 12 ----
DEBUG 05-06 16:00:04.090575.090575 cuda_h.py:27] end decode_layer cost 5.047 ms
DEBUG 05-06 16:00:04.090611.090611 lmp.py:1516] ---- decode step 2 layer 13 ----
DEBUG 05-06 16:00:04.095517.095517 cuda_h.py:27] end decode_layer cost 4.912 ms
DEBUG 05-06 16:00:04.095737.095737 lmp.py:1516] ---- decode step 2 layer 14 ----
DEBUG 05-06 16:00:04.100489.100489 cuda_h.py:27] end decode_layer cost 4.832 ms
DEBUG 05-06 16:00:04.100332.100332 lmp.py:1516] ---- decode step 2 layer 15 ----
DEBUG 05-06 16:00:04.105798.105798 cuda_h.py:27] end decode_layer cost 5.008 ms
DEBUG 05-06 16:00:04.105833.105833 lmp.py:1516] ---- decode step 2 layer 16 ----
DEBUG 05-06 16:00:04.110806.110806 cuda_h.py:27] end decode_layer cost 4.925 ms
DEBUG 05-06 16:00:04.110364.110364 lmp.py:1516] ---- decode step 2 layer 17 ----
DEBUG 05-06 16:00:04.115078.115078 cuda_h.py:27] end decode_layer cost 5.086 ms
DEBUG 05-06 16:00:04.115490.115490 lmp.py:1516] ---- decode step 2 layer 18 ----
DEBUG 05-06 16:00:04.120627.120627 cuda_h.py:27] end decode_layer cost 4.870 ms
DEBUG 05-06 16:00:04.120185.120185 lmp.py:1516] ---- decode step 2 layer 19 ----
DEBUG 05-06 16:00:04.125903.125903 cuda_h.py:27] end decode_layer cost 5.018 ms
DEBUG 05-06 16:00:04.125746.125746 lmp.py:1516] ---- decode step 2 layer 20 ----
DEBUG 05-06 16:00:04.130237.130237 cuda_h.py:27] end decode_layer cost 4.956 ms
DEBUG 05-06 16:00:04.130895.130895 lmp.py:1516] ---- decode step 2 layer 21 ----
DEBUG 05-06 16:00:04.135356.135356 cuda_h.py:27] end decode_layer cost 4.864 ms
DEBUG 05-06 16:00:04.135576.135576 lmp.py:1516] ---- decode step 2 layer 22 ----
DEBUG 05-06 16:00:04.140330.140330 cuda_h.py:27] end decode_layer cost 4.904 ms
DEBUG 05-06 16:00:04.140173.140173 lmp.py:1516] ---- decode step 2 layer 23 ----
DEBUG 05-06 16:00:04.145963.145963 cuda_h.py:27] end decode_layer cost 5.177 ms
DEBUG 05-06 16:00:04.145183.145183 lmp.py:1516] ---- decode step 2 layer 24 ----
DEBUG 05-06 16:00:04.150781.150781 cuda_h.py:27] end decode_layer cost 5.000 ms
DEBUG 05-06 16:00:04.150147.150147 lmp.py:1516] ---- decode step 2 layer 25 ----
DEBUG 05-06 16:00:04.155655.155655 cuda_h.py:27] end decode_layer cost 4.847 ms
DEBUG 05-06 16:00:04.155326.155326 lmp.py:1516] ---- decode step 2 layer 26 ----
DEBUG 05-06 16:00:04.160669.160669 cuda_h.py:27] end decode_layer cost 4.881 ms
DEBUG 05-06 16:00:04.160750.160750 lmp.py:1516] ---- decode step 2 layer 27 ----
DEBUG 05-06 16:00:04.165054.165054 cuda_h.py:27] end decode_layer cost 4.924 ms
DEBUG 05-06 16:00:04.165228.165228 lmp.py:1516] ---- decode step 2 layer 28 ----
DEBUG 05-06 16:00:04.170128.170128 cuda_h.py:27] end decode_layer cost 4.906 ms
DEBUG 05-06 16:00:04.170779.170779 lmp.py:1516] ---- decode step 2 layer 29 ----
DEBUG 05-06 16:00:04.175480.175480 cuda_h.py:27] end decode_layer cost 5.111 ms
DEBUG 05-06 16:00:04.175371.175371 cuda_h.py:27] end decode_step cost 158.647 ms
INFO 05-06 16:00:04.175564.175564 lmp.py:1564] decode step 2 time: 0.15868687629699707 seconds
Time taken: 5.947339933365583 seconds
X512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 24, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 25, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 26, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
TP MOE layer 27, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 28, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 29, pool: 0x59321bc09f00, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
CPUInfer[0x59321b9a56e0]: Goodbye
CPUInfer[0x59320c224070]: Goodbye
