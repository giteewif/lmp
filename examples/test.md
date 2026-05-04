here pin
INFO 05-02 10:55:05.333830.333830 pinpool.py:28] Initializing PinnedMemoryPool with 2GB total, allocating in 1024MB chunks...
INFO 05-02 10:55:05.890251.890251 pinpool.py:40] Allocated chunk 1: 536870912 elements (1024.0 MB)
INFO 05-02 10:55:06.330457.330457 pinpool.py:40] Allocated chunk 2: 536870912 elements (1024.0 MB)
INFO 05-02 10:55:06.330133.330133 pinpool.py:52] Successfully allocated 2 chunks, total 1073741824 elements (2048.0 MB) in 0.997s
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
DEBUG 05-02 10:55:14.485756.485756 cuda_h.py:27] end init_cmv_hmv cost 414.901 ms
DEBUG 05-02 10:55:14.494274.494274 cuda_memory_view.py:1366] 
DEBUG 05-02 10:55:14.494274.494274 cuda_memory_view.py:1366] restore_tensors_from_shared_memory_names time: 0.003563404083251953
DEBUG 05-02 10:55:14.512923.512923 mlpmodule.py:927] restore_hm_state_dict2model loaded 627 language_model tensors for Gemma4 model
DEBUG 05-02 10:55:14.512315.512315 cuda_memory_view.py:1370] 
DEBUG 05-02 10:55:14.512315.512315 cuda_memory_view.py:1370] restore_hm_state_dict2model time: 0.017595767974853516
kt-kernel version      : 0.6.1
kt-kernel CPU variant : avx512_vnni
INFO 05-02 10:55:16.436546.436546 lmp.py:163] init kt-kernel layer 0 ok
INFO 05-02 10:55:17.301369.301369 lmp.py:163] init kt-kernel layer 1 ok
INFO 05-02 10:55:18.159666.159666 lmp.py:163] init kt-kernel layer 2 ok
INFO 05-02 10:55:18.987351.987351 lmp.py:163] init kt-kernel layer 3 ok
INFO 05-02 10:55:19.820199.820199 lmp.py:163] init kt-kernel layer 4 ok
INFO 05-02 10:55:20.645377.645377 lmp.py:163] init kt-kernel layer 5 ok
generate input ids cost 0.05198335647583008 s
DEBUG 05-02 10:55:23.756810.756810 cuda_h.py:27] end generate_input_ids cost 3059.731 ms
DEBUG 05-02 10:55:23.757526.757526 cuda_h.py:27] end init_cache cost 0.032 ms
INFO 05-02 10:55:23.769119.769119 lmp.py:1704] [predo_tensor_index_locate] tensors=658 skipped_tensors=356 sagln_layers=30 expert_groups=30 general_tensors=3 first_device=cuda:0 remaining={'cuda:0': 6629859268, 'cuda:1': 12898664448, 'cuda:2': 12898664448, 'cuda:3': 12898664448} expected_used_bytes={'cuda:0': 17685700668, 'cuda:1': 11418992640, 'cuda:2': 11418992640, 'cuda:3': 11418992640} expected_used_mib={'cuda:0': 16866.39849472046, 'cuda:1': 10890.0, 'cuda:2': 10890.0, 'cuda:3': 10890.0} usage_ratio={'cuda:0': 0.7273408761529578, 'cuda:1': 0.46957618485519786, 'cuda:2': 0.46957618485519786, 'cuda:3': 0.46957618485519786}
INFO 05-02 10:55:23.769321.769321 lmp.py:1722] [predo_tensor_index_locate] layer=0 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.769197.769197 lmp.py:1722] [predo_tensor_index_locate] layer=1 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.769682.769682 lmp.py:1722] [predo_tensor_index_locate] layer=2 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.770388.770388 lmp.py:1722] [predo_tensor_index_locate] layer=3 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.770966.770966 lmp.py:1722] [predo_tensor_index_locate] layer=4 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.770504.770504 lmp.py:1722] [predo_tensor_index_locate] layer=5 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.770366.770366 lmp.py:1722] [predo_tensor_index_locate] layer=6 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771863.771863 lmp.py:1722] [predo_tensor_index_locate] layer=7 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771924.771924 lmp.py:1722] [predo_tensor_index_locate] layer=8 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771791.771791 lmp.py:1722] [predo_tensor_index_locate] layer=9 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771938.771938 lmp.py:1722] [predo_tensor_index_locate] layer=10 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771102.771102 lmp.py:1722] [predo_tensor_index_locate] layer=11 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771772.771772 lmp.py:1722] [predo_tensor_index_locate] layer=12 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771440.771440 lmp.py:1722] [predo_tensor_index_locate] layer=13 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771495.771495 lmp.py:1722] [predo_tensor_index_locate] layer=14 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771956.771956 lmp.py:1722] [predo_tensor_index_locate] layer=15 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.771010.771010 lmp.py:1722] [predo_tensor_index_locate] layer=16 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772612.772612 lmp.py:1722] [predo_tensor_index_locate] layer=17 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772666.772666 lmp.py:1722] [predo_tensor_index_locate] layer=18 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772720.772720 lmp.py:1722] [predo_tensor_index_locate] layer=19 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772251.772251 lmp.py:1722] [predo_tensor_index_locate] layer=20 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772667.772667 lmp.py:1722] [predo_tensor_index_locate] layer=21 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772483.772483 lmp.py:1722] [predo_tensor_index_locate] layer=22 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772992.772992 lmp.py:1722] [predo_tensor_index_locate] layer=23 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772808.772808 lmp.py:1722] [predo_tensor_index_locate] layer=24 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772211.772211 lmp.py:1722] [predo_tensor_index_locate] layer=25 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.772597.772597 lmp.py:1722] [predo_tensor_index_locate] layer=26 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.773867.773867 lmp.py:1722] [predo_tensor_index_locate] layer=27 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.773776.773776 lmp.py:1722] [predo_tensor_index_locate] layer=28 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
INFO 05-02 10:55:23.773954.773954 lmp.py:1722] [predo_tensor_index_locate] layer=29 gate_up/down expert_id_device_counts={'cuda:0': 32, 'cuda:1': 32, 'cuda:2': 32, 'cuda:3': 32}
DEBUG 05-02 10:55:24.000055.000055 cuda_h.py:27] end init_loading_placement cost 242.985 ms
DEBUG 05-02 10:55:24.000271.000271 sllm_store_c.py:27] get device uuid map
DEBUG 05-02 10:55:24.000399.000399 sllm_store_c.py:29] call client load into gpu
DEBUG 05-02 10:55:24 client.py:72] load_into_gpu: gemma4-26B-A4B, 8c423a0a-1ed5-40b4-87fe-83a3a1d30f00
INFO 05-02 10:55:24 client.py:135] Model loaded: gemma4-26B-A4B, 8c423a0a-1ed5-40b4-87fe-83a3a1d30f00
INFO 05-02 10:55:24 client.py:204] confirm_model_loaded: gemma4-26B-A4B, 8c423a0a-1ed5-40b4-87fe-83a3a1d30f00
INFO 05-02 10:55:24 client.py:212] Model loaded
DEBUG 05-02 10:55:24.528602.528602 cuda_h.py:27] end init_general_sagl_loading_async cost 528.673 ms
DEBUG 05-02 10:55:24.547555.547555 sllm_store_c.py:27] get device uuid map
DEBUG 05-02 10:55:24.547618.547618 sllm_store_c.py:29] call client load into gpu
DEBUG 05-02 10:55:24 client.py:72] load_into_gpu: gemma4-26B-A4B, ca5f78b6-75ca-4ef7-908e-5198ac8a4e0a
INFO 05-02 10:55:24 client.py:135] Model loaded: gemma4-26B-A4B, ca5f78b6-75ca-4ef7-908e-5198ac8a4e0a
DEBUG 05-02 10:55:24.617569.617569 cuda_h.py:27] end init_experts_loading_async cost 88.677 ms
INFO 05-02 10:55:24.657510.657510 lmp.py:2225] [split-check] queue_tids=8277  sagl=597  experts=7680 | overlap=0  lost=0  extra=0 | missing_experts=0  nonexpert_in_experts=0 | sagl_rows={0: 597}  exp_rows={0: 1920, 1: 1920, 2: 1920, 3: 1920} | off_expert=7680  off_nonexpert=0
DEBUG 05-02 10:55:24.755227.755227 cuda_h.py:27] end restore_state_dict cost 97.255 ms
DEBUG 05-02 10:55:24.777509.777509 cuda_h.py:27] end init_inputs_tokens cost 22.121 ms
DEBUG 05-02 10:55:24.777135.777135 lmp.py:500] -------------------------------- start prefill layer 0 --------------------------------
DEBUG 05-02 10:55:25.011494.011494 cuda_h.py:27] end *sagl cost 233.554 ms
INFO 05-02 10:55:25.059041.059041 lmp.py:1145] experts_cpu_alloc
INFO 05-02 10:55:25.060740.060740 lmp.py:1155] experts_gpu_alloc_device_0
INFO 05-02 10:55:25.060311.060311 lmp.py:1155] experts_gpu_alloc_device_1
INFO 05-02 10:55:25.060106.060106 lmp.py:1155] experts_gpu_alloc_device_2
INFO 05-02 10:55:25.060233.060233 lmp.py:1155] experts_gpu_alloc_device_3
INFO 05-02 10:55:25.123207.123207 lmp.py:769] kt-kernel experts time: 0.06302690505981445 seconds
submit_high_priority_copy_tasks(gpu experts) ok= True pending_count= 0
submit_high_priority_copy_tasks(gpu experts) time 0.0013391971588134766
wait_copy_tasks(gpu experts) time 0.0006146430969238281
wait_copy_tasks(gpu experts) ok= True pending_count= 0
INFO 05-02 10:55:25.149137.149137 lmp.py:809] wait_copy_tasks(gpu experts) time: 0.0006237030029296875 seconds
prepare_fused_expert_work_items time: 0.06836533546447754 seconds
INFO 05-02 10:55:25.225953.225953 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.144ms act=5.890ms bmm2=0.747ms unpad=0.724ms total=7.504ms E=32 maxT=209 S=1176 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-02 10:55:25.231143.231143 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.704ms act=2.413ms bmm2=1.169ms unpad=0.617ms total=4.904ms E=32 maxT=178 S=1094 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-02 10:55:25.236936.236936 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.229ms act=2.199ms bmm2=1.384ms unpad=0.697ms total=4.509ms E=32 maxT=172 S=869 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-02 10:55:25.240810.240810 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.261ms act=2.097ms bmm2=1.450ms unpad=0.646ms total=4.455ms E=32 maxT=148 S=957 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-02 10:55:25.241555.241555 lmp.py:876] experts compute time: 0.022965192794799805 seconds
INFO 05-02 10:55:25.241753.241753 lmp.py:885] scatter_reduce_ time: 0.00012135505676269531 seconds
DEBUG 05-02 10:55:25.241158.241158 cuda_h.py:27] end *layer_moe_fused cost 230.719 ms
DEBUG 05-02 10:55:25.241610.241610 cuda_h.py:27] end prefill_layer cost 464.558 ms
DEBUG 05-02 10:55:25.241074.241074 lmp.py:534] -------------------------------- end prefill layer 0 --------------------------------
DEBUG 05-02 10:55:25.242622.242622 lmp.py:500] -------------------------------- start prefill layer 1 --------------------------------
DEBUG 05-02 10:55:25.265282.265282 cuda_h.py:27] end *sagl cost 23.606 ms
INFO 05-02 10:55:25.268551.268551 lmp.py:1145] experts_cpu_alloc
INFO 05-02 10:55:25.268157.268157 lmp.py:1155] experts_gpu_alloc_device_0
INFO 05-02 10:55:25.268218.268218 lmp.py:1155] experts_gpu_alloc_device_1
INFO 05-02 10:55:25.268961.268961 lmp.py:1155] experts_gpu_alloc_device_2
INFO 05-02 10:55:25.268201.268201 lmp.py:1155] experts_gpu_alloc_device_3
INFO 05-02 10:55:25.296738.296738 lmp.py:769] kt-kernel experts time: 0.027719736099243164 seconds
submit_high_priority_copy_tasks(gpu experts) ok= True pending_count= 0
submit_high_priority_copy_tasks(gpu experts) time 0.0012645721435546875
wait_copy_tasks(gpu experts) time 0.0006215572357177734
wait_copy_tasks(gpu experts) ok= True pending_count= 0
INFO 05-02 10:55:25.322652.322652 lmp.py:809] wait_copy_tasks(gpu experts) time: 0.0006299018859863281 seconds
prepare_fused_expert_work_items time: 0.01963186264038086 seconds
INFO 05-02 10:55:25.345166.345166 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.152ms act=0.122ms bmm2=1.067ms unpad=1.024ms total=2.365ms E=32 maxT=142 S=756 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-02 10:55:25.346054.346054 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.078ms act=0.079ms bmm2=0.044ms unpad=1.069ms total=1.271ms E=32 maxT=164 S=894 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-02 10:55:25.348383.348383 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.081ms act=0.063ms bmm2=0.112ms unpad=1.133ms total=1.388ms E=32 maxT=211 S=1025 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-02 10:55:25.349121.349121 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.054ms act=0.059ms bmm2=0.050ms unpad=1.011ms total=1.174ms E=32 maxT=182 S=1421 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-02 10:55:25.350820.350820 lmp.py:876] experts compute time: 0.007500171661376953 seconds
INFO 05-02 10:55:25.350864.350864 lmp.py:885] scatter_reduce_ time: 8.440017700195312e-05 seconds
DEBUG 05-02 10:55:25.350348.350348 cuda_h.py:27] end *layer_moe_fused cost 84.899 ms
DEBUG 05-02 10:55:25.350211.350211 cuda_h.py:27] end prefill_layer cost 108.755 ms
DEBUG 05-02 10:55:25.351951.351951 lmp.py:534] -------------------------------- end prefill layer 1 --------------------------------
DEBUG 05-02 10:55:25.351932.351932 lmp.py:500] -------------------------------- start prefill layer 2 --------------------------------
DEBUG 05-02 10:55:25.355954.355954 cuda_h.py:27] end *sagl cost 4.455 ms
INFO 05-02 10:55:25.357995.357995 lmp.py:1145] experts_cpu_alloc
INFO 05-02 10:55:25.357752.357752 lmp.py:1155] experts_gpu_alloc_device_0
INFO 05-02 10:55:25.358203.358203 lmp.py:1155] experts_gpu_alloc_device_1
INFO 05-02 10:55:25.358322.358322 lmp.py:1155] experts_gpu_alloc_device_2
INFO 05-02 10:55:25.358817.358817 lmp.py:1155] experts_gpu_alloc_device_3
INFO 05-02 10:55:25.371591.371591 lmp.py:769] kt-kernel experts time: 0.012595176696777344 seconds
submit_high_priority_copy_tasks(gpu experts) ok= True pending_count= 0
submit_high_priority_copy_tasks(gpu experts) time 0.0011858940124511719
wait_copy_tasks(gpu experts) time 0.0005824565887451172
wait_copy_tasks(gpu experts) ok= True pending_count= 0
INFO 05-02 10:55:25.397897.397897 lmp.py:809] wait_copy_tasks(gpu experts) time: 0.0005900859832763672 seconds
prepare_fused_expert_work_items time: 0.017357587814331055 seconds
INFO 05-02 10:55:25.416290.416290 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.135ms act=0.105ms bmm2=0.064ms unpad=1.122ms total=1.426ms E=32 maxT=121 S=1258 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-02 10:55:25.418340.418340 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.068ms act=0.064ms bmm2=0.056ms unpad=1.227ms total=1.415ms E=32 maxT=303 S=1111 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-02 10:55:25.419502.419502 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.062ms act=0.059ms bmm2=0.035ms unpad=1.029ms total=1.185ms E=32 maxT=153 S=909 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-02 10:55:25.421801.421801 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.053ms act=0.057ms bmm2=0.038ms unpad=0.983ms total=1.131ms E=32 maxT=184 S=818 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-02 10:55:25.421109.421109 lmp.py:876] experts compute time: 0.0063588619232177734 seconds
INFO 05-02 10:55:25.421946.421946 lmp.py:885] scatter_reduce_ time: 7.796287536621094e-05 seconds
DEBUG 05-02 10:55:25.421681.421681 cuda_h.py:27] end *layer_moe_fused cost 65.944 ms
DEBUG 05-02 10:55:25.421371.421371 cuda_h.py:27] end prefill_layer cost 70.689 ms
DEBUG 05-02 10:55:25.421882.421882 lmp.py:534] -------------------------------- end prefill layer 2 --------------------------------
DEBUG 05-02 10:55:25.422873.422873 lmp.py:500] -------------------------------- start prefill layer 3 --------------------------------
DEBUG 05-02 10:55:25.426967.426967 cuda_h.py:27] end *sagl cost 3.923 ms
INFO 05-02 10:55:25.428555.428555 lmp.py:1145] experts_cpu_alloc
INFO 05-02 10:55:25.428372.428372 lmp.py:1155] experts_gpu_alloc_device_0
INFO 05-02 10:55:25.428889.428889 lmp.py:1155] experts_gpu_alloc_device_1
INFO 05-02 10:55:25.428033.428033 lmp.py:1155] experts_gpu_alloc_device_2
INFO 05-02 10:55:25.428733.428733 lmp.py:1155] experts_gpu_alloc_device_3
INFO 05-02 10:55:25.437816.437816 lmp.py:769] kt-kernel experts time: 0.008238792419433594 seconds
submit_high_priority_copy_tasks(gpu experts) ok= True pending_count= 0
submit_high_priority_copy_tasks(gpu experts) time 0.0011584758758544922
wait_copy_tasks(gpu experts) time 0.0005767345428466797
wait_copy_tasks(gpu experts) ok= True pending_count= 0
INFO 05-02 10:55:25.463180.463180 lmp.py:809] wait_copy_tasks(gpu experts) time: 0.0005848407745361328 seconds
prepare_fused_expert_work_items time: 0.017287015914916992 seconds
INFO 05-02 10:55:25.481588.481588 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.132ms act=0.102ms bmm2=0.043ms unpad=1.085ms total=1.363ms E=32 maxT=108 S=936 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-02 10:55:25.483483.483483 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.071ms act=0.074ms bmm2=0.141ms unpad=1.018ms total=1.304ms E=32 maxT=101 S=765 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-02 10:55:25.485154.485154 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.073ms act=0.074ms bmm2=0.040ms unpad=1.117ms total=1.304ms E=32 maxT=125 S=973 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-02 10:55:25.486674.486674 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.079ms act=0.073ms bmm2=0.112ms unpad=1.094ms total=1.357ms E=32 maxT=226 S=1422 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-02 10:55:25.487869.487869 lmp.py:876] experts compute time: 0.006638050079345703 seconds
INFO 05-02 10:55:25.487570.487570 lmp.py:885] scatter_reduce_ time: 7.390975952148438e-05 seconds
DEBUG 05-02 10:55:25.487624.487624 cuda_h.py:27] end *layer_moe_fused cost 61.278 ms
DEBUG 05-02 10:55:25.487507.487507 cuda_h.py:27] end prefill_layer cost 65.599 ms
DEBUG 05-02 10:55:25.487786.487786 lmp.py:534] -------------------------------- end prefill layer 3 --------------------------------
DEBUG 05-02 10:55:25.487732.487732 lmp.py:500] -------------------------------- start prefill layer 4 --------------------------------
DEBUG 05-02 10:55:25.492066.492066 cuda_h.py:27] end *sagl cost 4.245 ms
INFO 05-02 10:55:25.494609.494609 lmp.py:1145] experts_cpu_alloc
INFO 05-02 10:55:25.494743.494743 lmp.py:1155] experts_gpu_alloc_device_0
INFO 05-02 10:55:25.494307.494307 lmp.py:1155] experts_gpu_alloc_device_1
INFO 05-02 10:55:25.494679.494679 lmp.py:1155] experts_gpu_alloc_device_2
INFO 05-02 10:55:25.494136.494136 lmp.py:1155] experts_gpu_alloc_device_3
INFO 05-02 10:55:25.502998.502998 lmp.py:769] kt-kernel experts time: 0.008436918258666992 seconds
submit_high_priority_copy_tasks(gpu experts) ok= True pending_count= 0
submit_high_priority_copy_tasks(gpu experts) time 0.0011756420135498047
wait_copy_tasks(gpu experts) time 0.0005915164947509766
wait_copy_tasks(gpu experts) ok= True pending_count= 0
INFO 05-02 10:55:25.528144.528144 lmp.py:809] wait_copy_tasks(gpu experts) time: 0.0006003379821777344 seconds
prepare_fused_expert_work_items time: 0.016439437866210938 seconds
INFO 05-02 10:55:25.546962.546962 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.122ms act=0.095ms bmm2=0.064ms unpad=1.340ms total=1.620ms E=32 maxT=270 S=1322 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-02 10:55:25.548414.548414 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.076ms act=0.072ms bmm2=0.052ms unpad=1.130ms total=1.331ms E=32 maxT=162 S=1021 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-02 10:55:25.550821.550821 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.073ms act=0.071ms bmm2=0.041ms unpad=1.137ms total=1.322ms E=32 maxT=134 S=727 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-02 10:55:25.551004.551004 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.073ms act=0.070ms bmm2=0.056ms unpad=0.994ms total=1.193ms E=32 maxT=132 S=1026 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-02 10:55:25.552570.552570 lmp.py:876] experts compute time: 0.006742000579833984 seconds
INFO 05-02 10:55:25.552310.552310 lmp.py:885] scatter_reduce_ time: 8.320808410644531e-05 seconds
DEBUG 05-02 10:55:25.552569.552569 cuda_h.py:27] end *layer_moe_fused cost 60.056 ms
DEBUG 05-02 10:55:25.552545.552545 cuda_h.py:27] end prefill_layer cost 64.608 ms
DEBUG 05-02 10:55:25.552771.552771 lmp.py:534] -------------------------------- end prefill layer 4 --------------------------------
DEBUG 05-02 10:55:25.552079.552079 lmp.py:500] -------------------------------- start prefill layer 5 --------------------------------
DEBUG 05-02 10:55:25.907575.907575 cuda_h.py:27] end *sagl cost 354.202 ms
INFO 05-02 10:55:25.909835.909835 lmp.py:1145] experts_cpu_alloc
INFO 05-02 10:55:25.909348.909348 lmp.py:1155] experts_gpu_alloc_device_0
INFO 05-02 10:55:25.909396.909396 lmp.py:1155] experts_gpu_alloc_device_1
INFO 05-02 10:55:25.909860.909860 lmp.py:1155] experts_gpu_alloc_device_2
INFO 05-02 10:55:25.909656.909656 lmp.py:1155] experts_gpu_alloc_device_3
INFO 05-02 10:55:25.919906.919906 lmp.py:769] kt-kernel experts time: 0.010085344314575195 seconds
submit_high_priority_copy_tasks(gpu experts) ok= True pending_count= 0
submit_high_priority_copy_tasks(gpu experts) time 0.0012857913970947266
wait_copy_tasks(gpu experts) time 0.0006012916564941406
wait_copy_tasks(gpu experts) ok= True pending_count= 0
INFO 05-02 10:55:25.952183.952183 lmp.py:809] wait_copy_tasks(gpu experts) time: 0.0006101131439208984 seconds
prepare_fused_expert_work_items time: 0.012463808059692383 seconds
INFO 05-02 10:55:25.967687.967687 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.137ms act=0.122ms bmm2=0.074ms unpad=1.591ms total=1.924ms E=32 maxT=320 S=1227 H=2816 dev=cuda:0 dtype=torch.bfloat16
INFO 05-02 10:55:25.969246.969246 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.087ms act=0.092ms bmm2=0.057ms unpad=1.465ms total=1.702ms E=32 maxT=220 S=905 H=2816 dev=cuda:1 dtype=torch.bfloat16
INFO 05-02 10:55:25.971794.971794 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.080ms act=0.078ms bmm2=0.057ms unpad=1.180ms total=1.395ms E=32 maxT=188 S=1157 H=2816 dev=cuda:2 dtype=torch.bfloat16
INFO 05-02 10:55:25.972446.972446 mlpmodule.py:2746] [fused_experts] bmm_from_padded bmm1=0.076ms act=0.074ms bmm2=0.052ms unpad=1.152ms total=1.354ms E=32 maxT=156 S=807 H=2816 dev=cuda:3 dtype=torch.bfloat16
INFO 05-02 10:55:25.973152.973152 lmp.py:876] experts compute time: 0.007761240005493164 seconds
INFO 05-02 10:55:25.973563.973563 lmp.py:885] scatter_reduce_ time: 9.036064147949219e-05 seconds
DEBUG 05-02 10:55:25.973439.973439 cuda_h.py:27] end *layer_moe_fused cost 66.383 ms
DEBUG 05-02 10:55:25.973937.973937 cuda_h.py:27] end prefill_layer cost 420.997 ms
DEBUG 05-02 10:55:25.973486.973486 lmp.py:534] -------------------------------- end prefill layer 5 --------------------------------
DEBUG 05-02 10:55:25.973921.973921 cuda_h.py:27] end prefill cost 1196.572 ms
INFO 05-02 10:55:25.974516.974516 lmp.py:536] prefill time: 1.196727991104126 seconds
Time taken: 5.403879806399345 seconds
CPUInfer[0x64b408acde10]: Hello
WorkerPool[0x64b408abf2f0] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
CPUInfer[0x64b41c611270]: Hello
WorkerPool[0x64b42295e0b0] 4 subpools, [numa:threads][0:24] [1:24] [2:24] [3:24] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 2, 24 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 3, 24 threads
TP MOE layer 0, pool: 0x64b42295e0b0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 1, pool: 0x64b42295e0b0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 2, pool: 0x64b42295e0b0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
TP MOE layer 3, pool: 0x64b42295e0b0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
TP MOE layer 4, pool: 0x64b42295e0b0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
TP MOE layer 5, pool: 0x64b42295e0b0, expert num: 128, num_experts_per_tok: 8
Created AVX512_BF16_MOE_TP 1 at numa 1 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 2 at numa 2 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 0 at numa 0 (vdpbf16ps kernel)
Created AVX512_BF16_MOE_TP 3 at numa 3 (vdpbf16ps kernel)
CPUInfer[0x64b41c611270]: Goodbye
CPUInfer[0x64b408acde10]: Goodbye
