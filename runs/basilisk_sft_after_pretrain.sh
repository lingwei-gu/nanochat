#!/bin/bash
set -euo pipefail

# Wait for the d36 ClimbMix base continuation to finish, then launch SFT.
# This is intended for standalone Basilisk, not Slurm.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCH_DISTRIBUTED_TIMEOUT_SECONDS="${TORCH_DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

export RUN_ROOT="${RUN_ROOT:-/home/l39gu/nanochat-runs}"
export RUN_NAME="${RUN_NAME:-d36_4b_climbmix}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$RUN_ROOT/$RUN_NAME}"
export NANOCHAT_ENV_DIR="${NANOCHAT_ENV_DIR:-/home/l39gu/.cache/nanochat-venv-gpu}"
export JOB_TMP_DIR="${JOB_TMP_DIR:-/tmp/${USER:-nanochat}/nanochat-tmp}"
export TMPDIR="$JOB_TMP_DIR"
export TMP="$JOB_TMP_DIR"
export TEMP="$JOB_TMP_DIR"
mkdir -p "$NANOCHAT_BASE_DIR" "$JOB_TMP_DIR"

export MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix_w3}"
export FINAL_BASE_STEP="${FINAL_BASE_STEP:-22107}"
export WAIT_SLEEP_SECONDS="${WAIT_SLEEP_SECONDS:-60}"
export SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-1}"
export SFT_SAVE_EVERY="${SFT_SAVE_EVERY:-25}"
export KEEP_LAST_SFT_CHECKPOINTS="${KEEP_LAST_SFT_CHECKPOINTS:-4}"
export SFT_RUN_NAME="${SFT_RUN_NAME:-${MODEL_TAG}_sft_basilisk}"
export SFT_LOAD_OPTIMIZER="${SFT_LOAD_OPTIMIZER:-0}"
export IDENTITY_COUNT="${IDENTITY_COUNT:-512}"
export PREFETCH_TASK_DATA="${PREFETCH_TASK_DATA:-1}"

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

if [ ! -f "$NANOCHAT_ENV_DIR/bin/activate" ]; then
    echo "Missing venv: $NANOCHAT_ENV_DIR/bin/activate" >&2
    exit 2
fi
source "$NANOCHAT_ENV_DIR/bin/activate"

base_checkpoint_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
sft_checkpoint_dir="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"

latest_base_step() {
    find "$base_checkpoint_dir" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
        | sed -E 's/model_([0-9]+)\.pt/\1/' \
        | sort -n \
        | tail -n 1
}

base_running() {
    (pgrep -af 'scripts.base_train' || true) | grep -F -- "--model-tag=$MODEL_TAG" >/dev/null
}

sft_running() {
    (pgrep -af 'scripts.chat_sft' || true) | grep -F -- "--model-tag=$MODEL_TAG" >/dev/null
}

final_base_checkpoint_ready() {
    local step
    step="$(printf '%06d' "$FINAL_BASE_STEP")"
    [ -f "$base_checkpoint_dir/model_${step}.pt" ] || return 1
    [ -f "$base_checkpoint_dir/meta_${step}.json" ] || return 1
    local rank
    for rank in $(seq 0 $((NPROC_PER_NODE - 1))); do
        [ -f "$base_checkpoint_dir/optim_${step}_rank${rank}.pt" ] || return 1
    done
}

if sft_running; then
    log "SFT already appears to be running for $MODEL_TAG; exiting watcher."
    exit 0
fi

log "Waiting for final base checkpoint step $FINAL_BASE_STEP in $base_checkpoint_dir"
while ! final_base_checkpoint_ready; do
    latest="$(latest_base_step || true)"
    if [ -z "$latest" ]; then
        latest="none"
    fi
    if base_running; then
        log "Base train still running; latest durable step: $latest"
    else
        log "Base train not detected and final checkpoint is not ready; latest durable step: $latest"
    fi
    sleep "$WAIT_SLEEP_SECONDS"
done

while base_running; do
    log "Final checkpoint exists; waiting for base_train process to release GPUs"
    sleep "$WAIT_SLEEP_SECONDS"
done

identity_path="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ -f "$identity_path" ] && ! grep -q 'd36_4b_climbmix_w3' "$identity_path"; then
    backup_path="$NANOCHAT_BASE_DIR/identity_conversations.pre_d36.$(date '+%Y%m%d_%H%M%S').jsonl"
    cp "$identity_path" "$backup_path"
    log "Backed up stale identity conversations to $backup_path"
fi
python runs/generate_d36_identity.py --output "$identity_path" --count "$IDENTITY_COUNT"

if [ "$PREFETCH_TASK_DATA" = "1" ]; then
    log "Prefetching SFT and chat-eval datasets"
    python - <<'PY'
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee

tasks_to_touch = [
    ("SmolTalk train", lambda: SmolTalk(split="train")),
    ("SmolTalk test", lambda: SmolTalk(split="test")),
    ("MMLU auxiliary_train", lambda: MMLU(subset="all", split="auxiliary_train")),
    ("MMLU test", lambda: MMLU(subset="all", split="test")),
    ("GSM8K train", lambda: GSM8K(subset="main", split="train")),
    ("GSM8K test", lambda: GSM8K(subset="main", split="test")),
    ("ARC-Easy test", lambda: ARC(subset="ARC-Easy", split="test")),
    ("ARC-Challenge test", lambda: ARC(subset="ARC-Challenge", split="test")),
    ("HumanEval test", lambda: HumanEval()),
    ("SimpleSpelling train", lambda: SimpleSpelling(size=1, split="train")),
    ("SpellingBee train", lambda: SpellingBee(size=1, split="train")),
    ("SpellingBee test", lambda: SpellingBee(size=1, split="test")),
]

for name, make_task in tasks_to_touch:
    task = make_task()
    if len(task) > 0:
        _ = task[0]
    print(f"Prefetched {name}: {len(task):,} rows")
PY
fi

SFT_ARGS=(
    --model-tag="$MODEL_TAG"
    --device-batch-size="$SFT_DEVICE_BATCH_SIZE"
    --save-every="$SFT_SAVE_EVERY"
    --keep-last-checkpoints="$KEEP_LAST_SFT_CHECKPOINTS"
    --run="$SFT_RUN_NAME"
    --activation-checkpointing
    --no-compile
)

latest_sft_step="$(
    find "$sft_checkpoint_dir" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
        | sed -E 's/model_([0-9]+)\.pt/\1/' \
        | sort -n \
        | tail -n 1 || true
)"
if [ -n "$latest_sft_step" ]; then
    latest_sft_step="$((10#$latest_sft_step))"
    log "Found existing SFT checkpoint step $latest_sft_step; resuming SFT"
    SFT_ARGS+=(--resume-from-step="$latest_sft_step")
else
    log "No existing SFT checkpoint; starting from base step $FINAL_BASE_STEP"
    SFT_ARGS+=(--model-step="$FINAL_BASE_STEP" --load-optimizer="$SFT_LOAD_OPTIMIZER")
fi

if [ -n "${EXTRA_SFT_ARGS:-}" ]; then
    read -r -a extra_sft_args <<< "$EXTRA_SFT_ARGS"
    SFT_ARGS+=("${extra_sft_args[@]}")
fi

log "Launching SFT for $MODEL_TAG on $NPROC_PER_NODE GPU(s)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
fi
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- "${SFT_ARGS[@]}"
log "SFT finished for $MODEL_TAG"
