#!/bin/bash
set -euo pipefail

# Continue the restored d36_4b_climbmix_w3 checkpoint on standalone Basilisk.
# The Hugging Face export currently contains model+metadata but not optimizer shards,
# so this uses --allow-missing-optimizer-state and restarts optimizer moments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCH_DISTRIBUTED_TIMEOUT_SECONDS="${TORCH_DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"

export RUN_ROOT="${RUN_ROOT:-/home/l39gu/nanochat-runs}"
export RUN_NAME="${RUN_NAME:-d36_4b_climbmix}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$RUN_ROOT/$RUN_NAME}"
export CLIMBMIX_DATA_DIR="${CLIMBMIX_DATA_DIR:-/home/l39gu/projects/climbmix-400b-shuffle}"
export NANOCHAT_ENV_DIR="${NANOCHAT_ENV_DIR:-/home/l39gu/.cache/nanochat-venv-gpu}"
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-/home/l39gu/.local/bin}"
export JOB_TMP_DIR="${JOB_TMP_DIR:-/tmp/${USER:-nanochat}/nanochat-tmp}"
mkdir -p "$NANOCHAT_BASE_DIR" "$CLIMBMIX_DATA_DIR" "$(dirname "$NANOCHAT_ENV_DIR")" "$UV_INSTALL_DIR" "$JOB_TMP_DIR"

if [ -z "${NPROC_PER_NODE:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
        IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
        export NPROC_PER_NODE="${#visible_gpus[@]}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        export NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"
    else
        export NPROC_PER_NODE=8
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
fi

export MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix_w3}"
export MODEL_DEPTH="${MODEL_DEPTH:-36}"
export TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-12}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
export EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
export SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
export DATASET_SHARDS="${DATASET_SHARDS:-1500}"
export DATASET_WORKERS="${DATASET_WORKERS:-8}"
export TOKENIZER_SHARDS="${TOKENIZER_SHARDS:-8}"
export SAVE_EVERY="${SAVE_EVERY:-50}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-8}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export REQUIRE_RESUME="${REQUIRE_RESUME:-1}"
export MIN_RESUME_STEP="${MIN_RESUME_STEP:-13769}"
export USE_FP8="${USE_FP8:-0}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"
export SKIP_TOKENIZER_TRAIN="${SKIP_TOKENIZER_TRAIN:-1}"
export SKIP_IDENTITY_DOWNLOAD="${SKIP_IDENTITY_DOWNLOAD:-1}"
export PREFETCH_TASK_DATA="${PREFETCH_TASK_DATA:-0}"
export STOP_AFTER="${STOP_AFTER:-base_train}"
export WANDB_RUN="${WANDB_RUN:-d36_4b_climbmix_w3_continue_corrected_basilisk}"

# Preserve the step-to-token accounting from the saved checkpoint:
# step 13,769 * 2,064,384 tokens/step = 28,424,503,296 tokens.
export EXTRA_BASE_TRAIN_ARGS="--total-batch-size=2064384 --window-pattern=L --allow-missing-optimizer-state ${EXTRA_BASE_TRAIN_ARGS:-}"
if [ "$ACTIVATION_CHECKPOINTING" = "1" ] && [[ " ${EXTRA_BASE_TRAIN_ARGS:-} " != *" --activation-checkpointing"* ]]; then
    export EXTRA_BASE_TRAIN_ARGS="--activation-checkpointing ${EXTRA_BASE_TRAIN_ARGS:-}"
fi
if [ "$USE_TORCH_COMPILE" = "0" ] && [[ " ${EXTRA_BASE_TRAIN_ARGS:-} " != *" --no-compile"* ]]; then
    export EXTRA_BASE_TRAIN_ARGS="--no-compile ${EXTRA_BASE_TRAIN_ARGS:-}"
fi

bash runs/train_4b.sh
