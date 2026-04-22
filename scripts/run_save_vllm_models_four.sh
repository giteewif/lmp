#!/usr/bin/env bash
# Run ServerlessLLM vLLM export for Readme.md models (lines 122–128).
# This machine: 3×A6000. TP=4 needs 4 GPUs; TP=3 breaks head divisibility (20/32 heads).
# Use TP=2 + bf16 + MoE env; save_vllm_model.py sets disable_custom_all_reduce=True.
set -euo pipefail

source /mnt/zhengcf3/lmp_env/lmp/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-1}"
export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-latency}"

TP="${TENSOR_PARALLEL_SIZE:-2}"
DTYPE="${SAVE_VLLM_DTYPE:-bfloat16}"
# Default 0.6 is too low for large MoE shards on 48GB GPUs (KV budget goes negative).
GPU_MEM="${SAVE_VLLM_GPU_MEM:-0.95}"
LOGDIR="${SAVE_VLLM_LOGDIR:-/mnt/zhengcf3/lmp/logs/save_vllm_models}"
STORAGE=/mnt/zhengcf3/models/vllm_sllm_models
SCRIPT=/mnt/zhengcf3/ServerlessLLM/sllm_store/examples/save_vllm_model.py

mkdir -p "$LOGDIR"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGDIR/master.log"; }

already_exported() {
  local model_name="$1"
  [[ -d "$STORAGE/$model_name/rank_0" ]]
}

run_one() {
  local model_name="$1"
  local local_path="$2"
  if already_exported "$model_name"; then
    log "SKIP $model_name (already under $STORAGE/$model_name/rank_0)"
    return 0
  fi
  local logfile="$LOGDIR/${model_name//\//_}.log"
  log "START $model_name (TP=$TP dtype=$DTYPE gpu_mem=$GPU_MEM)"
  if ! python3 "$SCRIPT" \
    --model-name "$model_name" \
    --local-model-path "$local_path" \
    --storage-path "$STORAGE" \
    --tensor-parallel-size "$TP" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEM" \
    >>"$logfile" 2>&1; then
    log "FAIL $model_name (see $logfile)"
    exit 1
  fi
  log "DONE $model_name"
}

run_one "Qwen3.5-35B" "/mnt/zhengcf3/models/Qwen3.5-35B"
run_one "Qwen3-30B-A3B" "/mnt/zhengcf3/models/Qwen3-30B-A3B"
run_one "ERNIE-4.5-VL-28B-A3B-Thinking" "/mnt/zhengcf3/models/ERNIE-4.5-VL-28B-A3B-Thinking"
run_one "ERNIE-4.5-21B-A3B-Thinking" "/mnt/zhengcf3/models/ERNIE-4.5-21B-A3B-Thinking"
log "ALL OK"
