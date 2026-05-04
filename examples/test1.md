here pin
INFO 05-01 16:27:16.406890.406890 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-01 16:27:16.992053.992053 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-01 16:27:17.459722.459722 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-01 16:27:17.459094.459094 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 1.053s
DEBUG 05-01 16:27:24.886019.886019 cuda_h.py:27] end init_cmv_hmv cost 465.110 ms
DEBUG 05-01 16:27:24.899209.899209 cuda_memory_view.py:1369] 
DEBUG 05-01 16:27:24.899209.899209 cuda_memory_view.py:1369] restore_tensors_from_shared_memory_names time: 0.0030410289764404297
DEBUG 05-01 16:27:24.917343.917343 mlpmodule.py:933] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-01 16:27:24.918510.918510 cuda_memory_view.py:1373] 
DEBUG 05-01 16:27:24.918510.918510 cuda_memory_view.py:1373] restore_tensors_from_shared_memory_names time: 0.018311738967895508
INFO 05-01 16:27:24.979795.979795 lmp.py:2842] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 7954624218, 'cuda:1': 13407813632, 'cuda:2': 13407813632, 'cuda:3': 6161150186} expected_used_bytes={'cuda:0': 16465793318, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 12638899990} expected_used_mib={'cuda:0': 15703.004186630249, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 12053.39430809021} usage_ratio={'cuda:0': 0.6742633820132894, 'cuda:1': 0.4599460967671259, 'cuda:2': 0.4599460967671259, 'cuda:3': 0.6722801200889731}
INFO 05-01 16:27:24.979767.979767 lmp.py:2860] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.980194.980194 lmp.py:2860] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.980656.980656 lmp.py:2860] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.980675.980675 lmp.py:2860] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.980712.980712 lmp.py:2860] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.980273.980273 lmp.py:2860] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.980264.980264 lmp.py:2860] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.980156.980156 lmp.py:2860] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.981240.981240 lmp.py:2860] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.981754.981754 lmp.py:2860] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.981123.981123 lmp.py:2860] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.981492.981492 lmp.py:2860] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.981709.981709 lmp.py:2860] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.981112.981112 lmp.py:2860] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.981157.981157 lmp.py:2860] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.982725.982725 lmp.py:2860] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.982662.982662 lmp.py:2860] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.982063.982063 lmp.py:2860] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.982132.982132 lmp.py:2860] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.982486.982486 lmp.py:2860] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.982066.982066 lmp.py:2860] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.982473.982473 lmp.py:2860] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.982311.982311 lmp.py:2860] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.982250.982250 lmp.py:2860] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.983208.983208 lmp.py:2860] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.983093.983093 lmp.py:2860] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.983316.983316 lmp.py:2860] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.983632.983632 lmp.py:2860] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.983094.983094 lmp.py:2860] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-01 16:27:24.983449.983449 lmp.py:2860] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
{'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:0': 16465793318, 'cuda:3': 12638899990}
{'cuda:0': 16465793318, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 12638899990}
extracted_tensor_device_offsets {0: {'lm_head.weight': 11418992640, 'model.language_model.embed_tokens.weight': 11418992640, 'model.language_model.norm.weight': 12895387648, 'model.language_model.layers.0.self_attn.q_proj.weight': 12895393282, 'model.language_model.layers.0.self_attn.k_proj.weight': 12918462978, 'model.language_model.layers.0.self_attn.v_proj.weight': 12929997314, 'model.language_model.layers.0.self_attn.o_proj.weight': 12941531650, 'model.language_model.layers.0.self_attn.q_norm.weight': 12918461954, 'model.language_model.layers.0.self_attn.k_norm.weight': 12918462466, 'model.language_model.layers.0.router.proj.weight': 13000313090, 'model.language_model.layers.0.router.scale': 13000307202, 'model.language_model.layers.0.router.per_expert_scale': 13000312834, 'model.language_model.layers.0.input_layernorm.weight': 13000284674, 'model.language_model.layers.0.post_attention_layernorm.weight': 13000290306, 'model.language_model.layers.0.post_feedforward_layernorm.weight': 13000301570, 'model.language_model.layers.0.post_feedforward_layernorm_1.weight': 13001033986, 'model.language_model.layers.0.post_feedforward_layernorm_2.weight': 13001039618, 'model.language_model.layers.0.pre_feedforward_layernorm.weight': 13000295938, 'model.language_model.layers.0.pre_feedforward_layernorm_2.weight': 13001045250, 'model.language_model.layers.0.layer_scalar': 12895393280, 'model.language_model.layers.0.mlp.down_proj.weight': 12988389890, 'model.language_model.layers.0.mlp.up_proj.weight': 12976495106, 'model.language_model.layers.0.mlp.gate_proj.weight': 12964600322}}
extracted_tensor_copy_chunks {0: [(0, 1476395008, 11418992640, 0, 1, 0, False), (50466278128, 5632, 12895387648, 0, 2, 0, False), (1510999048, 11534336, 12929997314, 0, 5, 0, False), (1522533384, 23068672, 12941531650, 0, 6, 0, False), (1499463688, 512, 12918461954, 0, 7, 0, False), (1499464200, 512, 12918462466, 0, 8, 0, False), (1581314824, 720896, 13000313090, 0, 9, 0, False), (1581308936, 5632, 13000307202, 0, 10, 0, False), (1581314568, 256, 13000312834, 0, 11, 0, False), (1581286408, 5632, 13000284674, 0, 12, 0, False), (1581292040, 5632, 13000290306, 0, 13, 0, False), (1581303304, 5632, 13000301570, 0, 14, 0, False), (3104568072, 5632, 13001033986, 0, 15, 0, False), (3104573704, 5632, 13001039618, 0, 16, 0, False), (1581297672, 5632, 13000295938, 0, 17, 0, False), (3104579336, 5632, 13001045250, 0, 18, 0, False), (1476395008, 2, 12895393280, 0, 19, 0, False), (1569391624, 11894784, 12988389890, 0, 20, 0, False), (1557496840, 11894784, 12976495106, 0, 21, 0, False), (1545602056, 11894784, 12964600322, 0, 22, 0, False), (1597895432, 7929856, 0, 0, 27, 0, False), (2604987144, 3964928, 253755392, 0, 28, 0, False)]}
general_cuda_memory_handles {0: b'\xe0\x9f/\xa2>X\x00\x00l\xd4\x02\x00\x00\x00\x00\x00&\x11p\xd5\x03\x00\x00\x00\x02W=\x00\x00\x00\x00\x00\x00\x84\x00\x00\x00\xff\x00\x00l\x00\x00\x00\x00\x00\x00\x00N-\xd8\xc18\x01\x00\\\x00\x00\x00\x00\x00\x00\x00\x00'}
DEBUG 05-01 16:27:25.608529.608529 sllm_store_c.py:27] get device uuid map
DEBUG 05-01 16:27:25.608025.608025 sllm_store_c.py:29] call client load into gpu
DEBUG 05-01 16:27:25 client.py:72] load_into_gpu: gemma4-26B-A4B, e7926365-0f0e-46ca-9fcf-2c91add16413
INFO 05-01 16:27:25 client.py:135] Model loaded: gemma4-26B-A4B, e7926365-0f0e-46ca-9fcf-2c91add16413
INFO 05-01 16:27:25 client.py:204] confirm_model_loaded: gemma4-26B-A4B, e7926365-0f0e-46ca-9fcf-2c91add16413
INFO 05-01 16:27:25 client.py:212] Model loaded
load_into_gpu_async time 0.2467339038848877 seconds
DEBUG 05-01 16:27:25.855141.855141 sllm_store_c.py:27] get device uuid map
DEBUG 05-01 16:27:25.855357.855357 sllm_store_c.py:29] call client load into gpu
DEBUG 05-01 16:27:25 client.py:72] load_into_gpu: gemma4-26B-A4B, b951f6e7-ec93-47dd-a574-ec32ab575fc6
INFO 05-01 16:27:25 client.py:135] Model loaded: gemma4-26B-A4B, b951f6e7-ec93-47dd-a574-ec32ab575fc6
{'layer_idx': 0, 'expert_id_to_device': {0: 'cuda:1', 1: 'cuda:2', 2: 'cuda:0', 3: 'cuda:3', 4: 'cuda:1', 5: 'cuda:2', 6: 'cuda:0', 7: 'cuda:3', 8: 'cuda:1', 9: 'cuda:2', 10: 'cuda:0', 11: 'cuda:3', 12: 'cuda:1', 13: 'cuda:2', 14: 'cuda:0', 15: 'cuda:3', 16: 'cuda:1', 17: 'cuda:2', 18: 'cuda:0', 19: 'cuda:3', 20: 'cuda:1', 21: 'cuda:2', 22: 'cuda:0', 23: 'cuda:3', 24: 'cuda:1', 25: 'cuda:2', 26: 'cuda:0', 27: 'cuda:3', 28: 'cuda:1', 29: 'cuda:2', 30: 'cuda:0', 31: 'cuda:3', 32: 'cuda:1', 33: 'cuda:2', 34: 'cuda:0', 35: 'cuda:3', 36: 'cuda:1', 37: 'cuda:2', 38: 'cuda:0', 39: 'cuda:3', 40: 'cuda:1', 41: 'cuda:2', 42: 'cuda:0', 43: 'cuda:3', 44: 'cuda:1', 45: 'cuda:2', 46: 'cuda:0', 47: 'cuda:3', 48: 'cuda:1', 49: 'cuda:2', 50: 'cuda:0', 51: 'cuda:3', 52: 'cuda:1', 53: 'cuda:2', 54: 'cuda:0', 55: 'cuda:3', 56: 'cuda:1', 57: 'cuda:2', 58: 'cuda:0', 59: 'cuda:3', 60: 'cuda:1', 61: 'cuda:2', 62: 'cuda:0', 63: 'cuda:3', 64: 'cuda:1', 65: 'cuda:2', 66: 'cuda:0', 67: 'cuda:3', 68: 'cuda:1', 69: 'cuda:2', 70: 'cuda:0', 71: 'cuda:3', 72: 'cuda:1', 73: 'cuda:2', 74: 'cuda:0', 75: 'cuda:3', 76: 'cuda:1', 77: 'cuda:2', 78: 'cuda:0', 79: 'cuda:3', 80: 'cuda:1', 81: 'cuda:2', 82: 'cuda:0', 83: 'cuda:3', 84: 'cuda:1', 85: 'cuda:2', 86: 'cuda:0', 87: 'cuda:3', 88: 'cuda:1', 89: 'cuda:2', 90: 'cuda:0', 91: 'cuda:3', 92: 'cuda:1', 93: 'cuda:2', 94: 'cuda:0', 95: 'cuda:3', 96: 'cuda:1', 97: 'cuda:2', 98: 'cuda:0', 99: 'cuda:3', 100: 'cuda:1', 101: 'cuda:2', 102: 'cuda:0', 103: 'cuda:3', 104: 'cuda:1', 105: 'cuda:2', 106: 'cuda:0', 107: 'cuda:3', 108: 'cuda:1', 109: 'cuda:2', 110: 'cuda:0', 111: 'cuda:3', 112: 'cuda:1', 113: 'cuda:2', 114: 'cuda:0', 115: 'cuda:3', 116: 'cuda:1', 117: 'cuda:2', 118: 'cuda:0', 119: 'cuda:3', 120: 'cuda:1', 121: 'cuda:2', 122: 'cuda:0', 123: 'cuda:3', 124: 'cuda:1', 125: 'cuda:2', 126: 'cuda:0', 127: 'cuda:3'}, 'device_to_expert_ids': {'cuda:1': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124], 'cuda:2': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125], 'cuda:0': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126], 'cuda:3': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]}, 'source_tensors': ['model.language_model.layers.0.experts.down_proj', 'model.language_model.layers.0.experts.gate_up_proj']}
bytes_per_device {0: 105657602}
total_bytes 105657602
layer0_task_ids [295, 279, 283, 284, 280, 281, 282, 298, 297, 296, 288, 289, 293, 290, 286, 287, 285, 291, 292, 294]
layer0_task_ids_count 20
submit_high_priority_copy_tasks ok= True pending_count= 0
submit_high_priority_copy_tasks time 0.0015223026275634766
wait_copy_tasks time 0.0008118152618408203
wait_copy_tasks(layer0 task_ids) ok= True pending_count= 0
submit_high_priority_copy_tasks ok= True pending_count= 0
submit_high_priority_copy_tasks time 0.0007040500640869141
wait_copy_tasks time 0.0007183551788330078
wait_copy_tasks(layer2 task_ids) ok= True pending_count= 0
wait_copy_tasks(layer2 task_ids) ok= True pending_count= 0
INFO 05-01 16:27:26 client.py:204] confirm_model_loaded: gemma4-26B-A4B, b951f6e7-ec93-47dd-a574-ec32ab575fc6
INFO 05-01 16:27:27 client.py:212] Model loaded
load_into_gpu_async time 1.481116771697998 seconds
======= 


CPUInfer[0x583ea6eac760]: Hello
WorkerPool[0x583ea6eac640] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x583ea6eac760]: Goodbye
