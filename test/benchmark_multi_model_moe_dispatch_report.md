# 多模型 MoE dispatch（prefill vs decode）

- **Generated:** 2026-04-21 04:06:13 UTC
- **models root:** `/mnt/zhengcf3/models`
- **devices measured:** cpu, cuda
- **dtype:** bfloat16
- **GPU timing:** warmup=1 iters=2
- **CPU timing:** warmup=0 iters=1（CPU 默认较少次迭代以缩短总耗时；可与 GPU 不可直接比绝对 ms）
- **batches (CPU / default):** 32,64 **prefill T (CPU):** 32,64 **alloc:** uniform
- **CUDA prefill T:** 32,64
- **CUDA batches (decode / shared):** (同 CPU)
- **CUDA prefill B-only:** 8,16,32
- **max_gather_gb (CPU/default):** 48.0 **CUDA only:** 120.0
- **prefill_skip_cpu:** False
- **max_S_sequential:** 8192
- **rows OK:** 48 **skipped (gather cap):** 20 **parse err:** 0 **OOM:** 44

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

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms    | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------:  | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 6144 | 33.0  | 55.5576  | 73.544   | 4631.7875 | 1963.9903 | 289.8913 | 63.6149  | 62.9798   | 4.557     | 1.3237    | 83.3691   |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 192 | 1.0312 | 27.14    | 16.8294  | 169.3216 | 71.5089   | 8.1699   | 18.231   | 10.061    | 0.4481    | 0.6201    | 6.2388    |
| cpu   | uniform | 64  | 1   | 384 | 2.0625 | 27.151   | 17.0972  | 328.1345 | 138.4916  | 15.0215  | 18.5534  | 19.1923   | 0.8096    | 0.6297    | 12.0855   |
| cuda  | uniform | 32  | 1   | 192 | 1.0312 | 10.3232  | 1.5083   | 2.8586   | 1.6864    | 1.1926   | 1.106    | 1.8953    | 1.0783    | 0.1461    | 0.2769    |
| cuda  | uniform | 64  | 1   | 384 | 2.0625 | 9.499    | 1.7451   | 6.1272   | 3.5466    | 2.5888   | 1.1096   | 3.5111    | 2.3331    | 0.1837    | 0.645     |

## `ERNIE-4.5-21B-A3B-Thinking`

- `/mnt/zhengcf3/models/ERNIE-4.5-21B-A3B-Thinking/config.json`
- **H×N×E×k:** 2560×1536×64×6 — ernie4_5_moe

### prefill

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms    | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------:  | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 6144 | 45.0  | 51.3197  | 117.6063 | 6063.5421 | 2723.8638 | 484.2772 | 158.3317 | 51.558    | 3.0586    | 2.2916    | 118.1523  |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 192 | 1.4062 | 31.593   | 21.932   | 228.4125 | 93.6176   | 17.191   | 24.0023  | 10.4146   | 0.7162    | 0.6942    | 7.2298    |
| cpu   | uniform | 64  | 1   | 384 | 2.8125 | 29.9257  | 19.5891  | 450.4002 | 191.221   | 30.7265  | 20.8635  | 22.9923   | 1.4727    | 0.6546    | 15.0506   |
| cuda  | uniform | 32  | 1   | 192 | 1.4062 | 7.5089   | 1.5461   | 3.9097   | 2.2852    | 1.6307   | 1.2872   | 2.5288    | 1.2669    | 0.2059    | 0.5207    |
| cuda  | uniform | 64  | 1   | 384 | 2.8125 | 7.5255   | 1.5339   | 8.3693   | 4.8441    | 3.541    | 1.3703   | 5.4563    | 2.5841    | 0.2038    | 1.1121    |

## `ERNIE-4.5-VL-28B-A3B-Thinking`

- `/mnt/zhengcf3/models/ERNIE-4.5-VL-28B-A3B-Thinking/config.json`
- **H×N×E×k:** 2560×1536×64×6 — ernie4_5_moe_vl text slice

### prefill

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 6144 | 45.0  | 47.6673  | 98.957   | 6729.758 | 2757.8507 | 513.2167 | 133.2346 | 68.0069   | 3.852     | 2.076     | 141.1818  |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 192 | 1.4062 | 27.2354  | 17.9689  | 226.7186 | 90.5374   | 22.9091  | 19.648   | 12.6173   | 1.166     | 0.6598    | 8.3244    |
| cpu   | uniform | 64  | 1   | 384 | 2.8125 | 26.1632  | 16.6173  | 451.7556 | 182.4982  | 37.9336  | 19.8871  | 27.1858   | 1.9075    | 0.6351    | 17.2669   |
| cuda  | uniform | 32  | 1   | 192 | 1.4062 | 9.977    | 1.6091   | 3.9274   | 2.2989    | 1.6492   | 1.3758   | 2.4408    | 1.1987    | 0.1613    | 0.3936    |
| cuda  | uniform | 64  | 1   | 384 | 2.8125 | 10.2633  | 1.6727   | 8.3863   | 4.8624    | 3.5485   | 1.3514   | 5.0137    | 2.6259    | 0.163     | 0.8171    |

## `Qwen1.5-MoE-A2.7B`

- `/mnt/zhengcf3/models/Qwen1.5-MoE-A2.7B/config.json`
- **H×N×E×k:** 2048×1408×60×4 — qwen2_moe

### prefill

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms    | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------:  | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 4096 | 22.0  | 39.3299  | 27.7707  | 2904.0833 | 1293.6775 | 180.3132 | 37.3447  | 104.5735  | 4.8284    | 0.7061    | 73.839    |
| cpu   | uniform | 32  | 64  | 8192 | 44.0  | 48.8859  | 56.6796  | 5766.5475 | 2763.3127 | 444.7886 | 108.5307 | 101.7393  | 4.0983    | 1.1594    | 117.9593  |
| cpu   | uniform | 64  | 32  | 8192 | 44.0  | 49.3665  | 75.0064  | 6372.5839 | 2753.118  | 342.9204 | 67.6479  | 84.9605   | 5.0692    | 1.5194    | 129.0873  |
| cuda  | uniform | 8   | 32  | 1024 | 5.5   | 7.2128   | 1.1696   | 14.5098   | 8.3179    | 6.2009   | 0.8461   | 12.4056   | 7.3291    | 0.1622    | 2.0117    |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 128 | 0.6875 | 23.3939  | 16.0417  | 112.5994 | 48.8526   | 6.66     | 16.7778  | 7.0192    | 0.397     | 0.6857    | 4.8132    |
| cpu   | uniform | 64  | 1   | 256 | 1.375  | 25.14    | 16.4165  | 223.7956 | 96.3553   | 10.6502  | 21.0087  | 13.6323   | 0.5069    | 0.653     | 8.902     |
| cuda  | uniform | 32  | 1   | 128 | 0.6875 | 7.204    | 1.1771   | 1.9996   | 1.2246    | 0.7906   | 0.9931   | 1.6987    | 0.7961    | 0.1634    | 0.2776    |
| cuda  | uniform | 64  | 1   | 256 | 1.375  | 7.7784   | 1.234    | 3.9126   | 2.2107    | 1.7262   | 1.0333   | 3.1707    | 1.6706    | 0.1586    | 0.503     |

## `Qwen3-30B-A3B`

- `/mnt/zhengcf3/models/Qwen3-30B-A3B/config.json`
- **H×N×E×k:** 2048×768×128×8 — qwen3_moe N=moe_intermediate_size

### prefill

| dev   | alloc   | B   | T   | S     | GiB   | seq_ms   | grp_ms   | bat_ms    | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:   | ----: | -------: | -------: | -------:  | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 8192  | 24.0  | 61.7993  | 40.2943  | 3375.4102 | 1497.6887 | 229.5275 | 82.6759  | 83.7688   | 2.7762    | 0.652     | 54.6189   |
| cpu   | uniform | 32  | 64  | 16384 | 48.0  | —        | 93.6775  | 6991.8721 | 3017.7711 | 361.8003 | 40.3374  | 74.6377   | 8.9693    | —         | —         |
| cpu   | uniform | 64  | 32  | 16384 | 48.0  | —        | 84.4622  | 6257.3885 | 2722.6901 | 436.5839 | 59.1966  | 74.0851   | 7.3751    | —         | —         |
| cuda  | uniform | 8   | 32  | 2048  | 6.0   | 15.0243  | 1.9927   | 16.4047   | 9.6773    | 6.7644   | 1.764    | 8.2322    | 3.8346    | 0.1326    | 1.0919    |

### decode

| dev   | alloc   | B   | T   | S   | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 256 | 0.75  | 44.0256  | 29.0858  | 128.6362 | 53.6112   | 7.5694   | 31.12    | 4.4226    | 0.2432    | 0.6607    | 2.9218    |
| cpu   | uniform | 64  | 1   | 512 | 1.5   | 51.3798  | 29.5325  | 237.1728 | 103.8595  | 11.6082  | 35.4843  | 8.0309    | 0.3271    | 0.5748    | 4.6161    |
| cuda  | uniform | 32  | 1   | 256 | 0.75  | 14.7828  | 2.0127   | 2.2152   | 1.3623    | 0.8615   | 1.7498   | 1.1006    | 0.4923    | 0.1362    | 0.1498    |
| cuda  | uniform | 64  | 1   | 512 | 1.5   | 15.3121  | 2.0709   | 4.409    | 2.5662    | 1.883    | 1.769    | 2.129     | 1.0644    | 0.1352    | 0.2879    |

## `Qwen3.5-35B`

- `/mnt/zhengcf3/models/Qwen3.5-35B/config.json`
- **H×N×E×k:** 2048×512×256×8 — qwen3_5_moe text_config

### prefill

| dev   | alloc   | B   | T   | S     | GiB   | seq_ms   | grp_ms   | bat_ms    | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:   | ----: | -------: | -------: | -------:  | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 8192  | 16.0  | 101.2372 | 57.5981  | 2226.0095 | 948.2185  | 151.7264 | 53.1269  | 38.6473   | 2.8559    | 0.5689    | 21.9881   |
| cpu   | uniform | 32  | 64  | 16384 | 32.0  | —        | 71.7511  | 4672.5196 | 1657.1252 | 306.53   | 61.3115  | 65.1212   | 4.9996    | —         | —         |
| cpu   | uniform | 64  | 32  | 16384 | 32.0  | —        | 80.2954  | 4696.2283 | 2004.9031 | 322.0867 | 54.4484  | 58.4869   | 5.9154    | —         | —         |
| cuda  | uniform | 8   | 32  | 2048  | 4.0   | 32.3041  | 3.9509   | 10.9317   | 6.1459    | 4.55     | 3.8131   | 2.7669    | 1.1933    | 0.1223    | 0.3384    |

### decode

| dev   | alloc   | B   | T   | S   | GiB   | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----: | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 256 | 0.5   | 67.3857  | 48.6869  | 85.5643  | 38.4003   | 6.3203   | 49.0534  | 1.7574    | 0.1288    | 0.7225    | 1.2698    |
| cpu   | uniform | 64  | 1   | 512 | 1.0   | 74.5164  | 46.2478  | 164.2388 | 72.8091   | 9.6204   | 47.7225  | 3.5513    | 0.2016    | 0.6206    | 2.2041    |
| cuda  | uniform | 32  | 1   | 256 | 0.5   | 29.4053  | 2.8666   | 1.7941   | 1.1971    | 0.609    | 2.6284   | 0.6259    | 0.2317    | 0.0975    | 0.061     |
| cuda  | uniform | 64  | 1   | 512 | 1.0   | 32.1059  | 3.91     | 2.9889   | 1.8235    | 1.172    | 3.6269   | 0.7644    | 0.3231    | 0.1218    | 0.0931    |

## `deepseek-moe-16b-base`

- `/mnt/zhengcf3/models/deepseek-moe-16b-base/config.json`
- **H×N×E×k:** 2048×1408×64×6 — deepseek n_routed_experts

### prefill

| dev   | alloc   | B   | T   | S    | GiB   | seq_ms   | grp_ms   | bat_ms    | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --:  | ----: | -------: | -------: | -------:  | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 32  | 6144 | 33.0  | 47.1215  | 63.322   | 4898.0954 | 2068.1438 | 275.56   | 67.2406  | 77.3521   | 4.0981    | 1.3438    | 103.9462  |

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 192 | 1.0312 | 27.8378  | 18.248   | 162.4629 | 67.0122   | 7.6786   | 19.3035  | 8.9031    | 0.3978    | 0.6555    | 5.836     |
| cpu   | uniform | 64  | 1   | 384 | 2.0625 | 28.9598  | 18.6328  | 317.0007 | 106.6172  | 16.1016  | 20.1698  | 17.013    | 0.7983    | 0.6434    | 10.9462   |
| cuda  | uniform | 32  | 1   | 192 | 1.0312 | 7.2468   | 1.2377   | 2.847    | 1.675     | 1.1822   | 1.0601   | 2.3002    | 1.1152    | 0.1708    | 0.3929    |
| cuda  | uniform | 64  | 1   | 384 | 2.0625 | 7.1503   | 1.4125   | 6.1128   | 3.5486    | 2.5844   | 1.101    | 4.3275    | 2.3474    | 0.1976    | 0.8549    |

## `gemma4-26B-A4B`

- `/mnt/zhengcf3/models/gemma4-26B-A4B/config.json`
- **H×N×E×k:** 2816×1408×128×8 — gemma4 text gate+up N=2×704

### decode

| dev   | alloc   | B   | T   | S   | GiB    | seq_ms   | grp_ms   | bat_ms   | gath_ms   | bmm_ms   | gmm_ms   | bat/grp   | bmm/gmm   | grp/seq   | bat/seq   |
| ----- | ------- | --: | --: | --: | ----:  | -------: | -------: | -------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: |
| cpu   | uniform | 32  | 1   | 256 | 1.8906 | 47.3335  | 31.573   | 271.9271 | 110.1932  | 18.0457  | 37.0913  | 8.6127    | 0.4865    | 0.667     | 5.7449    |
| cpu   | uniform | 64  | 1   | 512 | 3.7812 | 63.4062  | 40.2074  | 548.7312 | 239.2164  | 36.2086  | 56.6083  | 13.6475   | 0.6396    | 0.6341    | 8.6542    |
| cuda  | uniform | 32  | 1   | 256 | 1.8906 | 19.1426  | 2.9652   | 5.6399   | 3.4501    | 2.2055   | 2.7276   | 1.902     | 0.8086    | 0.1549    | 0.2946    |
| cuda  | uniform | 64  | 1   | 512 | 3.7812 | 19.5269  | 3.0372   | 11.2123  | 6.8652    | 4.3313   | 2.7023   | 3.6917    | 1.6028    | 0.1555    | 0.5742    |

## OOM

- gemma4-26B-A4B cuda prefill B=8 T=32 uniform
- gemma4-26B-A4B cuda prefill B=8 T=64 uniform
- gemma4-26B-A4B cuda prefill B=16 T=32 uniform
- gemma4-26B-A4B cuda prefill B=16 T=64 uniform
- gemma4-26B-A4B cuda prefill B=32 T=32 uniform
- ERNIE-4.5-VL-28B-A3B-Thinking cuda prefill B=8 T=32 uniform
- ERNIE-4.5-VL-28B-A3B-Thinking cuda prefill B=8 T=64 uniform
- ERNIE-4.5-VL-28B-A3B-Thinking cuda prefill B=16 T=32 uniform
- ERNIE-4.5-VL-28B-A3B-Thinking cuda prefill B=16 T=64 uniform
- ERNIE-4.5-VL-28B-A3B-Thinking cuda prefill B=32 T=32 uniform
- ERNIE-4.5-VL-28B-A3B-Thinking cuda prefill B=32 T=64 uniform
- ERNIE-4.5-21B-A3B-Thinking cuda prefill B=8 T=32 uniform
- ERNIE-4.5-21B-A3B-Thinking cuda prefill B=8 T=64 uniform
- ERNIE-4.5-21B-A3B-Thinking cuda prefill B=16 T=32 uniform
- ERNIE-4.5-21B-A3B-Thinking cuda prefill B=16 T=64 uniform
- ERNIE-4.5-21B-A3B-Thinking cuda prefill B=32 T=32 uniform
- ERNIE-4.5-21B-A3B-Thinking cuda prefill B=32 T=64 uniform
- DeepSeek-V2-Lite cuda prefill B=8 T=32 uniform
- DeepSeek-V2-Lite cuda prefill B=8 T=64 uniform
- DeepSeek-V2-Lite cuda prefill B=16 T=32 uniform
- DeepSeek-V2-Lite cuda prefill B=16 T=64 uniform
- DeepSeek-V2-Lite cuda prefill B=32 T=32 uniform
- DeepSeek-V2-Lite cuda prefill B=32 T=64 uniform
- deepseek-moe-16b-base cuda prefill B=8 T=32 uniform
- deepseek-moe-16b-base cuda prefill B=8 T=64 uniform
- deepseek-moe-16b-base cuda prefill B=16 T=32 uniform
- deepseek-moe-16b-base cuda prefill B=16 T=64 uniform
- deepseek-moe-16b-base cuda prefill B=32 T=32 uniform
- deepseek-moe-16b-base cuda prefill B=32 T=64 uniform
- Qwen1.5-MoE-A2.7B cuda prefill B=8 T=64 uniform
- Qwen1.5-MoE-A2.7B cuda prefill B=16 T=32 uniform
- Qwen1.5-MoE-A2.7B cuda prefill B=16 T=64 uniform
- Qwen1.5-MoE-A2.7B cuda prefill B=32 T=32 uniform
- Qwen1.5-MoE-A2.7B cuda prefill B=32 T=64 uniform
- Qwen3-30B-A3B cuda prefill B=8 T=64 uniform
- Qwen3-30B-A3B cuda prefill B=16 T=32 uniform
- Qwen3-30B-A3B cuda prefill B=16 T=64 uniform
- Qwen3-30B-A3B cuda prefill B=32 T=32 uniform
- Qwen3-30B-A3B cuda prefill B=32 T=64 uniform
- Qwen3.5-35B cuda prefill B=8 T=64 uniform
- Qwen3.5-35B cuda prefill B=16 T=32 uniform
- Qwen3.5-35B cuda prefill B=16 T=64 uniform
- Qwen3.5-35B cuda prefill B=32 T=32 uniform
- Qwen3.5-35B cuda prefill B=32 T=64 uniform

## 复现

```bash
python /mnt/zhengcf3/lmp/test/benchmark_expert_mm_dispatch.py --moe-multi-model \
  --moe-models-root /mnt/zhengcf3/models \
  --moe-batches "32,64" \
  --moe-prefill-seq "32,64" \
  --moe-prefill-seq-cuda "32,64" \
  --moe-prefill-batches-cuda "8,16,32" \
  --moe-max-gather-gb 48.0 \
  --moe-max-gather-gb-cuda 120.0 \
  --moe-alloc uniform \
  --moe-max-s-sequential 8192 \
  --moe-cpu-warmup 0 \
  --moe-cpu-iters 1 \
  --dtype bfloat16 \
  --warmup 1 \
  --iters 2
```

