#!/bin/bash
#SBATCH --partition=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH -o nanochat4b-continue-corrected-%j.out
#SBATCH -e nanochat4b-continue-corrected-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/u201/l39gu/projects/nanochat}"

# Continue the existing d36_4b_climbmix_w3 base checkpoint from its saved
# dataloader position. This reaches the corrected ratio-12 horizon derived from
# total params (~45.6B tokens) without replaying the first 28.4B tokens.
export RUN_ROOT="${RUN_ROOT:-/u201/l39gu/nanoknow-climbmix/nanochat-runs}"
export RUN_NAME="${RUN_NAME:-d36_4b_climbmix}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$RUN_ROOT/$RUN_NAME}"
export CLIMBMIX_DATA_DIR="${CLIMBMIX_DATA_DIR:-/u201/l39gu/projects/climbmix-400b-shuffle}"
mkdir -p "$NANOCHAT_BASE_DIR"

export MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix_w3}"
export MODEL_DEPTH="${MODEL_DEPTH:-36}"
export TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-12}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-7}"
export DATASET_SHARDS="${DATASET_SHARDS:-1500}"
export DATASET_WORKERS="${DATASET_WORKERS:-8}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export SAVE_EVERY="${SAVE_EVERY:-250}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-8}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export REQUIRE_RESUME="${REQUIRE_RESUME:-1}"
export MIN_RESUME_STEP="${MIN_RESUME_STEP:-13769}"

# Preserve the old step-to-token accounting. The checkpoint's step 13769
# represents 28.4B tokens only with this total batch size.
export EXTRA_BASE_TRAIN_ARGS="--total-batch-size=2064384 --window-pattern=L ${EXTRA_BASE_TRAIN_ARGS:-}"

export SKIP_TOKENIZER_TRAIN="${SKIP_TOKENIZER_TRAIN:-1}"
export SKIP_IDENTITY_DOWNLOAD="${SKIP_IDENTITY_DOWNLOAD:-1}"
export PREFETCH_TASK_DATA="${PREFETCH_TASK_DATA:-0}"
export STOP_AFTER="${STOP_AFTER:-base_train}"
export WANDB_RUN="${WANDB_RUN:-d36_4b_climbmix_w3_continue_corrected}"

bash runs/train_4b.sh
