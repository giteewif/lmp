# 多模型 MoE dispatch（prefill vs decode）

- **Generated:** 2026-04-21 07:30:31 UTC
- **models root:** `/mnt/zhengcf3/models`
- **devices measured:** cpu
- **dtype:** bfloat16
- **GPU timing:** warmup=0 iters=1
- **CPU timing:** warmup=0 iters=1（CPU 默认较少次迭代以缩短总耗时；可与 GPU 不可直接比绝对 ms）
- **batches (CPU / default):** 32,64 **prefill T (CPU):** 32 **alloc:** uniform
- **CUDA prefill T:** (同 CPU)
- **CUDA batches (decode / shared):** (同 CPU)
- **CUDA prefill B-only:** 8,16,32
- **max_gather_gb (CPU/default):** 48.0 **CUDA only:** (同左)
- **prefill_skip_cpu:** False
- **max_S_sequential:** 8192
- **rows OK:** 26 **skipped (gather cap):** 6 **parse err:** 0 **OOM:** 0

| 列       | 含义               |
| ------- | ---------------- |
| bat_ms  | gather+bmm 整段    |
| gath_ms | 仅 gather（无 bmm）  |
| bmm_ms  | 仅 bmm（已预 gather） |
| gmm_ms  | 仅 `_grouped_mm`  |

## `DeepSeek-V2-Lite`

- `/mnt/zhengcf3/models/DeepSeek-V2-Lite/config.json`
- **H×N×E×k:** 2048×1408×64×6 — deepseek n_routed_experts

### prefill

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 6144 | 33.0  | 46.0409  | 34.9957  | 30.4271  | 5.9232    | 17.7995  | 31.701   | 0.8695    | 0.5615    | 0.7601    | 0.6609    |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 192 | 1.0312 | 26.219   | 15.3673  | 8.0177   | 1.2947    | 5.7002   | 14.9758  | 0.5217    | 0.3806    | 0.5861    | 0.3058    |
| cpu   | uniform | 64  | 1   | 384 | 2.0625 | 25.7731  | 15.7593  | 8.4825   | 1.511     | 5.7527   | 15.2154  | 0.5383    | 0.3781    | 0.6115    | 0.3291    |

## `ERNIE-4.5-21B-A3B-Thinking`

- `/mnt/zhengcf3/models/ERNIE-4.5-21B-A3B-Thinking/config.json`
- **H×N×E×k:** 2560×1536×64×6 — ernie4_5_moe

### prefill

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 6144 | 45.0  | 46.1687  | 34.0657  | 31.9904  | 5.9338    | 19.1508  | 30.6626  | 0.9391    | 0.6246    | 0.7379    | 0.6929    |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 192 | 1.4062 | 30.3113  | 18.9643  | 17.8702  | 1.9123    | 14.6003  | 18.7973  | 0.9423    | 0.7767    | 0.6257    | 0.5896    |
| cpu   | uniform | 64  | 1   | 384 | 2.8125 | 32.3347  | 19.8406  | 18.8006  | 1.9285    | 16.1156  | 19.4612  | 0.9476    | 0.8281    | 0.6136    | 0.5814    |

## `ERNIE-4.5-VL-28B-A3B-Thinking`

- `/mnt/zhengcf3/models/ERNIE-4.5-VL-28B-A3B-Thinking/config.json`
- **H×N×E×k:** 2560×1536×64×6 — ernie4_5_moe_vl text slice

### prefill

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 6144 | 45.0  | 51.6843  | 41.316   | 40.7195  | 7.6803    | 20.8533  | 76.8299  | 0.9856    | 0.2714    | 0.7994    | 0.7878    |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 192 | 1.4062 | 29.4569  | 19.8453  | 21.3605  | 1.3992    | 19.6905  | 19.4885  | 1.0764    | 1.0104    | 0.6737    | 0.7251    |
| cpu   | uniform | 64  | 1   | 384 | 2.8125 | 32.8735  | 22.5574  | 20.469   | 1.6099    | 19.5524  | 22.0255  | 0.9074    | 0.8877    | 0.6862    | 0.6227    |

## `Qwen1.5-MoE-A2.7B`

- `/mnt/zhengcf3/models/Qwen1.5-MoE-A2.7B/config.json`
- **H×N×E×k:** 2048×1408×60×4 — qwen2_moe

### prefill

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 4096 | 22.0  | 38.8677  | 30.0305  | 14.8774  | 5.3853    | 5.0372   | 31.8445  | 0.4954    | 0.1582    | 0.7726    | 0.3828    |
| cpu   | uniform | 64  | 32  | 8192 | 44.0  | 46.9575  | 53.6276  | 39.8055  | 6.3887    | 21.3213  | 32.5598  | 0.7423    | 0.6548    | 1.142     | 0.8477    |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 128 | 0.6875 | 24.8525  | 17.2079  | 6.5412   | 1.2545    | 4.1384   | 17.1355  | 0.3801    | 0.2415    | 0.6924    | 0.2632    |
| cpu   | uniform | 64  | 1   | 256 | 1.375  | 26.0909  | 17.3837  | 7.9335   | 1.214     | 5.5443   | 17.3651  | 0.4564    | 0.3193    | 0.6663    | 0.3041    |

## `Qwen3-30B-A3B`

- `/mnt/zhengcf3/models/Qwen3-30B-A3B/config.json`
- **H×N×E×k:** 2048×768×128×8 — qwen3_moe N=moe_intermediate_size

### prefill

| dev   | alloc   | B   | T   | S     | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:   | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 8192  | 24.0  | 60.9782  | 34.727   | 26.9016  | 11.4442   | 7.5718   | 33.005   | 0.7747    | 0.2294    | 0.5695    | 0.4412    |
| cpu   | uniform | 64  | 32  | 16384 | 48.0  | —        | 50.535   | 44.4705  | 21.9183   | 9.1962   | 37.4977  | 0.88      | 0.2452    | —         | —         |

### decode

| dev   | alloc   | B   | T   | S   | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 256 | 0.75  | 45.5282  | 30.8689  | 7.8227   | 2.4465    | 4.0922   | 29.7273  | 0.2534    | 0.1377    | 0.678     | 0.1718    |
| cpu   | uniform | 64  | 1   | 512 | 1.5   | 50.5267  | 31.0904  | 8.4091   | 2.4402    | 5.0362   | 28.2709  | 0.2705    | 0.1781    | 0.6153    | 0.1664    |

## `Qwen3.5-35B`

- `/mnt/zhengcf3/models/Qwen3.5-35B/config.json`
- **H×N×E×k:** 2048×512×256×8 — qwen3_5_moe text_config

### prefill

| dev   | alloc   | B   | T   | S     | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:   | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 8192  | 16.0  | 97.5707  | 57.8282  | 35.0941  | 17.0286   | 10.8565  | 48.189   | 0.6069    | 0.2253    | 0.5927    | 0.3597    |
| cpu   | uniform | 64  | 32  | 16384 | 32.0  | —        | 63.8717  | 53.6529  | 29.4614   | 13.2662  | 52.7644  | 0.84      | 0.2514    | —         | —         |

### decode

| dev   | alloc   | B   | T   | S   | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 256 | 0.5   | 65.0393  | 40.9991  | 14.098   | 3.0437    | 9.1049   | 42.4614  | 0.3439    | 0.2144    | 0.6304    | 0.2168    |
| cpu   | uniform | 64  | 1   | 512 | 1.0   | 73.7833  | 45.4736  | 15.0045  | 3.4744    | 9.8639   | 46.3254  | 0.33      | 0.2129    | 0.6163    | 0.2034    |

## `deepseek-moe-16b-base`

- `/mnt/zhengcf3/models/deepseek-moe-16b-base/config.json`
- **H×N×E×k:** 2048×1408×64×6 — deepseek n_routed_experts

### prefill

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 6144 | 33.0  | 45.538   | 36.137   | 28.8319  | 7.3416    | 17.2766  | 32.6585  | 0.7979    | 0.529     | 0.7936    | 0.6331    |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 192 | 1.0312 | 27.4653  | 18.0146  | 7.9106   | 1.267     | 5.4909   | 17.7323  | 0.4391    | 0.3097    | 0.6559    | 0.288     |
| cpu   | uniform | 64  | 1   | 384 | 2.0625 | 28.2652  | 18.2806  | 7.7399   | 1.524     | 5.1183   | 17.9198  | 0.4234    | 0.2856    | 0.6468    | 0.2738    |

## `gemma4-26B-A4B`

- `/mnt/zhengcf3/models/gemma4-26B-A4B/config.json`
- **H×N×E×k:** 2816×1408×128×8 — gemma4 text gate+up N=2×704

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 256 | 1.8906 | 60.211   | 45.2602  | 18.6141  | 1.9781    | 15.1011  | 43.0313  | 0.4113    | 0.3509    | 0.7517    | 0.3091    |
| cpu   | uniform | 64  | 1   | 512 | 3.7812 | 64.117   | 41.6851  | 19.3863  | 2.6842    | 15.1901  | 41.157   | 0.4651    | 0.3691    | 0.6501    | 0.3024    |

## 复现

```bash
python /mnt/zhengcf3/lmp/test/benchmark_expert_mm_dispatch.py --moe-multi-model \
  --moe-models-root /mnt/zhengcf3/models \
  --moe-batches "32,64" \
  --moe-prefill-seq "32" \
  --moe-prefill-seq-cuda "" \
  --moe-prefill-batches-cuda "8,16,32" \
  --moe-max-gather-gb 48.0 \
  --moe-alloc uniform \
  --moe-max-s-sequential 8192 \
  --moe-cpu-warmup 0 \
  --moe-cpu-iters 1 \
  --dtype bfloat16 \
  --warmup 0 \
  --iters 1
```

