#!/bin/bash
set -euo pipefail

# Standalone Basilisk launcher for the d36 (~3.8B parameter) ClimbMix run.
# Basilisk has 8x RTX A6000 GPUs and no Slurm scheduler in PATH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCH_DISTRIBUTED_TIMEOUT_SECONDS="${TORCH_DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"

export RUN_ROOT="${RUN_ROOT:-/home/l39gu/nanochat-runs}"
export RUN_NAME="${RUN_NAME:-d36_4b_climbmix_basilisk}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$RUN_ROOT/$RUN_NAME}"
export CLIMBMIX_DATA_DIR="${CLIMBMIX_DATA_DIR:-/home/l39gu/projects/climbmix-400b-shuffle}"
export NANOCHAT_ENV_DIR="${NANOCHAT_ENV_DIR:-/home/l39gu/.cache/nanochat-venv-gpu}"
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-/home/l39gu/.local/bin}"
export JOB_TMP_DIR="${JOB_TMP_DIR:-/tmp/${USER:-nanochat}/nanochat-tmp}"
mkdir -p "$NANOCHAT_BASE_DIR" "$CLIMBMIX_DATA_DIR" "$(dirname "$NANOCHAT_ENV_DIR")" "$UV_INSTALL_DIR" "$JOB_TMP_DIR"
if [ -d "$NANOCHAT_ENV_DIR" ] && [ ! -f "$NANOCHAT_ENV_DIR/bin/activate" ]; then
    if ! rmdir "$NANOCHAT_ENV_DIR" 2>/dev/null; then
        echo "Found non-venv directory at NANOCHAT_ENV_DIR=$NANOCHAT_ENV_DIR" >&2
        echo "Set NANOCHAT_ENV_DIR to a valid venv path or remove that directory." >&2
        exit 2
    fi
fi

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

export MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix_basilisk}"
export MODEL_DEPTH="${MODEL_DEPTH:-36}"
export TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-12}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
export EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
export SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
export DATASET_SHARDS="${DATASET_SHARDS:-1500}"
export DATASET_WORKERS="${DATASET_WORKERS:-8}"
export TOKENIZER_SHARDS="${TOKENIZER_SHARDS:-8}"
export SAVE_EVERY="${SAVE_EVERY:-250}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export KEEP_LAST_SFT_CHECKPOINTS="${KEEP_LAST_SFT_CHECKPOINTS:-3}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export AUTO_RESUME_SFT="${AUTO_RESUME_SFT:-1}"
export USE_FP8="${USE_FP8:-0}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"
export WANDB_RUN="${WANDB_RUN:-dummy}"

# RTX A6000 is Ampere, so FA3 is unavailable and PyTorch SDPA is used.
# Full-context attention avoids the slow SDPA sliding-window path.
if [[ " ${EXTRA_BASE_TRAIN_ARGS:-} " != *" --window-pattern"* ]]; then
    export EXTRA_BASE_TRAIN_ARGS="${EXTRA_BASE_TRAIN_ARGS:-} --window-pattern=L"
fi
if [ "$ACTIVATION_CHECKPOINTING" = "1" ] && [[ " ${EXTRA_BASE_TRAIN_ARGS:-} " != *" --activation-checkpointing"* ]]; then
    export EXTRA_BASE_TRAIN_ARGS="${EXTRA_BASE_TRAIN_ARGS:-} --activation-checkpointing"
fi
if [ "$USE_TORCH_COMPILE" = "0" ] && [[ " ${EXTRA_BASE_TRAIN_ARGS:-} " != *" --no-compile"* ]]; then
    export EXTRA_BASE_TRAIN_ARGS="${EXTRA_BASE_TRAIN_ARGS:-} --no-compile"
fi

bash runs/train_4b.sh
