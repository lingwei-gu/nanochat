#!/bin/bash
set -euo pipefail

# Continue d36_4b_climbmix_w3 from the latest local base checkpoint to a
# 100B-token total pretraining horizon on Basilisk without replaying ClimbMix.

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

export MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix_w3}"
export MODEL_DEPTH="${MODEL_DEPTH:-36}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
export EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
export SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
export DATASET_SHARDS="${DATASET_SHARDS:-2501}"
export DATASET_WORKERS="${DATASET_WORKERS:-16}"
export TOKENIZER_SHARDS="${TOKENIZER_SHARDS:-8}"
export SAVE_EVERY="${SAVE_EVERY:-100}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-8}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export REQUIRE_RESUME="${REQUIRE_RESUME:-1}"
export MIN_RESUME_STEP="${MIN_RESUME_STEP:-22107}"
export USE_FP8="${USE_FP8:-0}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"
export SKIP_TOKENIZER_TRAIN="${SKIP_TOKENIZER_TRAIN:-1}"
export SKIP_IDENTITY_DOWNLOAD="${SKIP_IDENTITY_DOWNLOAD:-1}"
export PREFETCH_TASK_DATA="${PREFETCH_TASK_DATA:-0}"
export STOP_AFTER="${STOP_AFTER:-base_train}"
export WANDB_RUN="${WANDB_RUN:-d36_4b_climbmix_w3_continue_100b_basilisk}"

# 48,441 steps * 2,064,384 tokens/step = 100,000,825,344 total tokens.
# Keep optimizer state mandatory: if any rank-local shard is missing, fail
# instead of silently restarting moments.
TARGET_NUM_ITERATIONS="${TARGET_NUM_ITERATIONS:-48441}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-2064384}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-26.29393956372832}"
export TARGET_NUM_ITERATIONS TOTAL_BATCH_SIZE TARGET_PARAM_DATA_RATIO

# The ratio-12 run ended at lrm=0.05 and effectively zero weight decay. Directly
# extending the old schedule to 100B would jump lrm to ~0.8445 at step 22107.
# Use the old final effective LR as a constant continuation LR and keep WD at 0.
CRITICAL_BASE_ARGS=(
    --num-iterations="$TARGET_NUM_ITERATIONS"
    --total-batch-size="$TOTAL_BATCH_SIZE"
    --window-pattern=L
    --embedding-lr=0.015
    --unembedding-lr=0.0004
    --matrix-lr=0.001
    --scalar-lr=0.025
    --weight-decay=0
    --final-lr-frac=1.0
    --muon-momentum-override=0.90
    --eval-every=250
    --eval-tokens=41943040
    --core-metric-every=2000
    --core-metric-max-per-task=500
    --sample-every=2000
)

if [ "$ACTIVATION_CHECKPOINTING" = "1" ]; then
    CRITICAL_BASE_ARGS+=(--activation-checkpointing)
fi
if [ "$USE_TORCH_COMPILE" = "0" ]; then
    CRITICAL_BASE_ARGS+=(--no-compile)
fi

export EXTRA_BASE_TRAIN_ARGS="${EXTRA_BASE_TRAIN_ARGS:-} ${CRITICAL_BASE_ARGS[*]}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
fi

python - <<'PY'
import glob
import json
import os
import sys

base_dir = os.environ["NANOCHAT_BASE_DIR"]
model_tag = os.environ["MODEL_TAG"]
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)
steps = []
for path in glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
    try:
        steps.append(int(os.path.basename(path).split("_")[-1].split(".")[0]))
    except ValueError:
        pass
if not steps:
    raise SystemExit(f"No base checkpoints found in {checkpoint_dir}")
step = max(steps)
min_step = int(os.environ["MIN_RESUME_STEP"])
if step < min_step:
    raise SystemExit(f"Latest checkpoint step {step} is below MIN_RESUME_STEP={min_step}")

meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
if "dataloader_state_dict" not in meta:
    raise SystemExit(f"{meta_path} has no dataloader_state_dict")
if meta.get("total_batch_size") != int(os.environ["TOTAL_BATCH_SIZE"]):
    raise SystemExit(f"Checkpoint total_batch_size={meta.get('total_batch_size')} does not match TOTAL_BATCH_SIZE={os.environ['TOTAL_BATCH_SIZE']}")
if meta["dataloader_state_dict"].get("epoch") != 1:
    raise SystemExit(f"Latest dataloader state is already epoch {meta['dataloader_state_dict'].get('epoch')}; refusing possible data replay")

world = int(os.environ["NPROC_PER_NODE"])
missing_optim = [
    os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
    for rank in range(world)
    if not os.path.exists(os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt"))
]
if missing_optim:
    raise SystemExit("Missing optimizer shards:\n" + "\n".join(missing_optim))

data_dir = os.environ["CLIMBMIX_DATA_DIR"]
required = int(os.environ["DATASET_SHARDS"])
missing_shards = [
    i for i in range(required)
    if not os.path.exists(os.path.join(data_dir, f"shard_{i:05d}.parquet"))
]
if missing_shards:
    raise SystemExit(f"Missing {len(missing_shards)} train shards before launch, first={missing_shards[:5]}, last={missing_shards[-5:]}")
val_path = os.path.join(data_dir, "shard_06542.parquet")
if not os.path.exists(val_path):
    raise SystemExit(f"Missing validation shard: {val_path}")

target_steps = int(os.environ["TARGET_NUM_ITERATIONS"])
if step >= target_steps:
    raise SystemExit(f"Latest checkpoint step {step} is already at or beyond target {target_steps}")
tokens_now = step * int(os.environ["TOTAL_BATCH_SIZE"])
tokens_target = target_steps * int(os.environ["TOTAL_BATCH_SIZE"])
print(f"Preflight OK: resume step={step}, tokens_now={tokens_now:,}, target_step={target_steps}, target_tokens={tokens_target:,}, dataloader_state={meta['dataloader_state_dict']}")
PY

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "PREFLIGHT_ONLY=1, stopping before train_4b"
    exit 0
fi

bash runs/train_4b.sh
