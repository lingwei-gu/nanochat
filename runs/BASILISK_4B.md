# Basilisk handoff for d36 4B-class nanochat

Basilisk is a standalone 8x RTX A6000 host, not a Slurm node. Use:

```bash
bash runs/basilisk_4b.sh
```

Defaults:

- Run/checkpoints: `/home/l39gu/nanochat-runs/d36_4b_climbmix_basilisk`
- ClimbMix shards: `/home/l39gu/projects/climbmix-400b-shuffle`
- uv environment: `/home/l39gu/.cache/nanochat-venv-gpu`
- GPUs: all visible GPUs, normally 8
- Model: `MODEL_DEPTH=36`, `MODEL_TAG=d36_4b_climbmix_basilisk`
- Batch: `DEVICE_BATCH_SIZE=1`, auto total batch size and gradient accumulation
- Attention: `--window-pattern=L` for PyTorch SDPA on A6000
- Precision: `USE_FP8=0`
- Resume: `AUTO_RESUME=1`
- NCCL: disables P2P and IB by default (`NCCL_P2P_DISABLE=1`,
  `NCCL_IB_DISABLE=1`) because default NCCL hangs on Basilisk before the first
  barrier.
- Allocator: sets `PYTORCH_ALLOC_CONF=expandable_segments:True` and mirrors it
  to `PYTORCH_CUDA_ALLOC_CONF` for compatibility.
- Memory: enables `--activation-checkpointing` and disables `torch.compile` by
  default on Basilisk (`USE_TORCH_COMPILE=0`) to keep the d36 run inside A6000
  VRAM with optimizer state loaded.

Run a one-step 4B smoke test before a full launch:

```bash
RUN_NAME=d36_4b_climbmix_basilisk_smoke \
MODEL_TAG=d36_4b_climbmix_basilisk_smoke \
DATASET_SHARDS=1 \
TOKENIZER_SHARDS=1 \
TOKENIZER_MAX_CHARS=50000000 \
PREFETCH_TASK_DATA=0 \
STOP_AFTER=base_train \
EXTRA_BASE_TRAIN_ARGS="--num-iterations=1 --save-every=-1 --core-metric-every=-1 --sample-every=-1 --eval-every=-1" \
bash runs/basilisk_4b.sh
```

For a detached full run on Basilisk, use `tmux`:

```bash
tmux new-session -d -s nanochat4b \
  'cd /home/l39gu/projects/nanochat && bash runs/basilisk_4b.sh > /home/l39gu/nanochat-runs/d36_4b_climbmix_basilisk/basilisk4b.tmux.log 2>&1'
```

Monitor with:

```bash
tail -f /home/l39gu/nanochat-runs/d36_4b_climbmix_basilisk/basilisk4b.tmux.log
nvidia-smi
tmux attach -t nanochat4b
```

## Corrected continuation

The handoff checkpoint is not a fresh run. It resumes `d36_4b_climbmix_w3` from
step `13769`, after `28,424,503,296` tokens, toward the corrected ratio-12
horizon of `45,638,269,656` target tokens. That leaves `8,338` optimizer steps,
or `17,212,833,792` tokens, at the saved batch size of `2,064,384` tokens/step.

On Basilisk, after restoring the HF base export into
`/home/l39gu/nanochat-runs/d36_4b_climbmix`, use:

```bash
tmux new-session -d -s nanochat4b_continue \
  'cd /home/l39gu/projects/nanochat && bash runs/basilisk_4b_continue_corrected.sh > /home/l39gu/nanochat-runs/d36_4b_climbmix/basilisk4b-continue.log 2>&1'
```

The Hugging Face export contains base model metadata and weights but not the
rank-local optimizer shards. The first Basilisk bootstrap step resumes the model
and dataloader with a fresh optimizer, then writes a local checkpoint with
rank-local optimizer shards. Subsequent continuation resumes from that local
checkpoint with optimizer state.
