# 多模型 MoE dispatch（prefill vs decode）

- **Generated:** 2026-04-22 14:00:02 UTC
- **models root:** `/mnt/zhengcf3/models`
- **devices measured:** cpu
- **dtype:** bfloat16
- **GPU timing:** warmup=5 iters=30
- **CPU timing:** warmup=0 iters=1（CPU 默认较少次迭代以缩短总耗时；可与 GPU 不可直接比绝对 ms）
- **batches (CPU / default):** 64 **prefill T (CPU):** 128 **alloc:** uniform
- **CUDA prefill T:** 32,48,64,80,96,112,128
- **CUDA batches (decode / shared):** (同 CPU)
- **CUDA prefill B-only:** 8,16,32
- **max_gather_gb (CPU/default):** 500.0 **CUDA only:** (同左)
- **prefill_skip_cpu:** False
- **max_S_sequential:** 0
- **rows OK:** 16 **skipped (gather cap):** 0 **parse err:** 0 **OOM:** 0

| 列 | 含义 |
|----|------|
| bat_ms | gather+bmm 整段 |
| gath_ms | 仅 gather（无 bmm） |
| bmm_ms | 仅 bmm（已预 gather） |
| gmm_ms | 仅 `_grouped_mm` |

## `DeepSeek-V2-Lite`

- `/mnt/zhengcf3/models/DeepSeek-V2-Lite/config.json`
- **H×N×E×k:** 2048×1408×64×6 — deepseek n_routed_experts

### prefill

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 128 | 49152 | 264.0 | — | 172.8458 | 175.626 | 55.6898 | 45.3167 | 118.803 | 1.0161 | 0.3814 | — | — |

### decode

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 1 | 384 | 2.0625 | — | 17.8672 | 7.9355 | 1.5377 | 5.0615 | 16.3887 | 0.4441 | 0.3088 | — | — |

## `ERNIE-4.5-21B-A3B-Thinking`

- `/mnt/zhengcf3/models/ERNIE-4.5-21B-A3B-Thinking/config.json`
- **H×N×E×k:** 2560×1536×64×6 — ernie4_5_moe

### prefill

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 128 | 49152 | 360.0 | — | 763.4379 | 227.2682 | 65.844 | 80.3086 | 653.9727 | 0.2977 | 0.1228 | — | — |

### decode

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 1 | 384 | 2.8125 | — | 20.6125 | 23.2616 | 1.5986 | 21.6889 | 19.6022 | 1.1285 | 1.1064 | — | — |

## `ERNIE-4.5-VL-28B-A3B-Thinking`

- `/mnt/zhengcf3/models/ERNIE-4.5-VL-28B-A3B-Thinking/config.json`
- **H×N×E×k:** 2560×1536×64×6 — ernie4_5_moe_vl text slice

### prefill

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 128 | 49152 | 360.0 | — | 777.4383 | 233.8023 | 72.3128 | 87.5631 | 742.5816 | 0.3007 | 0.1179 | — | — |

### decode

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 1 | 384 | 2.8125 | — | 20.2834 | 12.6497 | 1.8714 | 9.3681 | 19.612 | 0.6236 | 0.4777 | — | — |

## `Qwen1.5-MoE-A2.7B`

- `/mnt/zhengcf3/models/Qwen1.5-MoE-A2.7B/config.json`
- **H×N×E×k:** 2048×1408×60×4 — qwen2_moe

### prefill

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 128 | 32768 | 176.0 | — | 143.0871 | 116.6723 | 37.7208 | 31.7751 | 109.8848 | 0.8154 | 0.2892 | — | — |

### decode

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 1 | 256 | 1.375 | — | 15.3427 | 8.1641 | 1.1516 | 5.9464 | 14.6333 | 0.5321 | 0.4064 | — | — |

## `Qwen3-30B-A3B`

- `/mnt/zhengcf3/models/Qwen3-30B-A3B/config.json`
- **H×N×E×k:** 2048×768×128×8 — qwen3_moe N=moe_intermediate_size

### prefill

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 128 | 65536 | 192.0 | — | 286.5863 | 168.3577 | 77.0216 | 31.9779 | 228.0034 | 0.5875 | 0.1403 | — | — |

### decode

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 1 | 512 | 1.5 | — | 26.9816 | 11.6301 | 2.076 | 8.498 | 28.0293 | 0.431 | 0.3032 | — | — |

## `Qwen3.5-35B`

- `/mnt/zhengcf3/models/Qwen3.5-35B/config.json`
- **H×N×E×k:** 2048×512×256×8 — qwen3_5_moe text_config

### prefill

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 128 | 65536 | 128.0 | — | 191.8245 | 191.9662 | 78.8149 | 55.3731 | 136.4995 | 1.0007 | 0.4057 | — | — |

### decode

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 1 | 512 | 1.0 | — | 41.896 | 19.8682 | 3.253 | 15.0123 | 43.0258 | 0.4742 | 0.3489 | — | — |

## `deepseek-moe-16b-base`

- `/mnt/zhengcf3/models/deepseek-moe-16b-base/config.json`
- **H×N×E×k:** 2048×1408×64×6 — deepseek n_routed_experts

### prefill

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 128 | 49152 | 264.0 | — | 173.4458 | 170.0944 | 54.2686 | 46.093 | 118.9389 | 0.9807 | 0.3875 | — | — |

### decode

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 1 | 384 | 2.0625 | — | 16.111 | 7.4736 | 1.2678 | 5.0012 | 15.5671 | 0.4639 | 0.3213 | — | — |

## `gemma4-26B-A4B`

- `/mnt/zhengcf3/models/gemma4-26B-A4B/config.json`
- **H×N×E×k:** 2816×1408×128×8 — gemma4 text gate+up N=2×704

### prefill

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 128 | 65536 | 484.0 | — | 1276.8184 | 1481.0268 | 97.9406 | 92.5268 | 259.7925 | 1.1599 | 0.3562 | — | — |

### decode

| dev | alloc | B | T | S | GiB | seq_ms | grp_ms | bat_ms | gath_ms | bmm_ms | gmm_ms | bat/grp | bmm/gmm | grp/seq | bat/seq |
|-----|-------|--:|--:|--:|----:|-------:|-------:|-------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|
| cpu | uniform | 64 | 1 | 512 | 3.7812 | — | 53.289 | 22.8823 | 3.7593 | 17.2666 | 45.8205 | 0.4294 | 0.3768 | — | — |

## 复现

```bash
python /mnt/zhengcf3/lmp/test/benchmark_expert_mm_dispatch.py --moe-multi-model \
  --moe-models-root /mnt/zhengcf3/models \
  --moe-batches "64" \
  --moe-prefill-seq "128" \
  --moe-prefill-seq-cuda "32,48,64,80,96,112,128" \
  --moe-prefill-batches-cuda "8,16,32" \
  --moe-max-gather-gb 500.0 \
  --moe-alloc uniform \
  --moe-max-s-sequential 0 \
  --moe-cpu-warmup 0 \
  --moe-cpu-iters 1 \
  --dtype bfloat16 \
  --warmup 5 \
  --iters 30
```

