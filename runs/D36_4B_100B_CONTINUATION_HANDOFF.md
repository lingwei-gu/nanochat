# d36 4B ClimbMix 100B Continuation Handoff

Last updated: 2026-07-08 16:42 EDT

## Goal

Continue the `d36_4b_climbmix_w3` base model from the corrected ratio-12
checkpoint to a fixed 100B-token total pretraining horizon, without replaying
ClimbMix data.

This is base pretraining only. Do not start SFT until the base run reaches the
100B horizon.

## Current Live Run

- Host: Basilisk, standalone 8x RTX A6000
- Repo: `/home/l39gu/projects/nanochat`
- Branch: `codex/4b-climbmix-setup`
- Run root: `/home/l39gu/nanochat-runs/d36_4b_climbmix`
- Model tag: `d36_4b_climbmix_w3`
- Log: `/home/l39gu/nanochat-runs/d36_4b_climbmix/basilisk4b-continue-100b.log`
- Launcher: `runs/basilisk_4b_continue_100b.sh`
- Active wrapper PID at handoff time: `90580`
- Active torchrun PID at handoff time: `90767`

Latest live log line seen at handoff:

```text
step 23036/48441 (47.55%) | loss: 2.099327 | lrm: 1.00 | dt: 96938.04ms | tok/sec: 21,295 | bf16_mfu: 0.00 | epoch: 1 pq: 1030 rg: 40
```

Latest durable checkpoint at handoff:

- Checkpoint step: `23000`
- Model: `/home/l39gu/nanochat-runs/d36_4b_climbmix/base_checkpoints/d36_4b_climbmix_w3/model_023000.pt`
- Metadata: `/home/l39gu/nanochat-runs/d36_4b_climbmix/base_checkpoints/d36_4b_climbmix_w3/meta_023000.json`
- Optimizer shards: `optim_023000_rank0.pt` through `optim_023000_rank7.pt`
- Validation BPB at step `23000`: `0.6229287795106019`
- Saved dataloader state: `{"pq_idx": 1028, "rg_idx": 72, "epoch": 1}`

## Token Horizon

The 100B target is implemented as an explicit step count:

- Total batch size: `2,064,384` tokens/step
- Target total steps: `48,441`
- Target total tokens: `48,441 * 2,064,384 = 100,000,825,344`
- Initial 100B continuation resume point: step `22,107`
- Tokens at step `22,107`: `45,637,337,088`
- Remaining at 100B launch: `26,334` steps / `54,363,488,256` tokens

At durable checkpoint `23,000`:

- Total tokens: `47,480,832,000`
- New continuation tokens since step `22,107`: `1,843,494,912`
- Remaining to target: `52,519,993,344` tokens / `25,441` steps

## Data Inventory And No-Replay Requirement

The local ClimbMix mirror is:

```text
/home/l39gu/projects/climbmix-400b-shuffle
```

Required files before launch:

- Train shards: contiguous `shard_00000.parquet` through `shard_02500.parquet`
- Validation shard: `shard_06542.parquet`

The 100B endpoint was estimated around train shard `2169`, so the contiguous
mirror through `2500` gives about 332 shards of margin. The saved dataloader
states are still `epoch=1`; if a checkpoint ever has `epoch > 1`, stop and
audit before continuing because that means the loader has wrapped.

Do not shrink `DATASET_SHARDS` below `2501`. Do not run with a data directory
that only contains the old `0..1499` train shards; that would force a wrap and
duplicate data before the 100B horizon.

## Continuation Hyperparameters

The active run uses the guarded wrapper:

```bash
runs/basilisk_4b_continue_100b.sh
```

Important pinned settings:

- `MODEL_TAG=d36_4b_climbmix_w3`
- `MODEL_DEPTH=36`
- `NPROC_PER_NODE=8`
- `DEVICE_BATCH_SIZE=1`
- `TOTAL_BATCH_SIZE=2064384`
- `TARGET_NUM_ITERATIONS=48441`
- `DATASET_SHARDS=2501`
- `SAVE_EVERY=100`
- `KEEP_LAST_CHECKPOINTS=8`
- `STOP_AFTER=base_train`
- `USE_FP8=0`
- `ACTIVATION_CHECKPOINTING=1`
- `USE_TORCH_COMPILE=0`
- `window-pattern=L`

Learning-rate continuation parameters:

- `embedding_lr=0.015`
- `unembedding_lr=0.0004`
- `matrix_lr=0.001`
- `scalar_lr=0.025`
- `weight_decay=0`
- `final_lr_frac=1.0`
- `muon_momentum_override=0.90`

These are intentional. Directly extending the old ratio-12 schedule to 100B
would have jumped the schedule multiplier from `0.05` to about `0.8445` at
step `22107`. This run instead continues at the old final effective LR, keeps
weight decay at zero, and fixes Muon momentum at `0.90`.

## Safe Resume Procedure

First confirm that no training job is already running:

```bash
pgrep -af 'torchrun|scripts.base_train|basilisk_4b_continue_100b|train_4b' || true
nvidia-smi
```

If the current run is still active, do not start another one.

If the run stopped, run the exact wrapper preflight:

```bash
cd /home/l39gu/projects/nanochat
PREFLIGHT_ONLY=1 SKIP_SETUP=1 bash runs/basilisk_4b_continue_100b.sh
```

The preflight must report:

- A latest checkpoint step at or above `22107`
- `total_batch_size=2064384`
- A `dataloader_state_dict`
- `epoch=1`
- 8 optimizer shards for the selected checkpoint
- All train shards `0..2500`
- Validation shard `06542`

Then relaunch detached:

```bash
cd /home/l39gu/projects/nanochat
setsid bash -lc 'cd /home/l39gu/projects/nanochat && export SKIP_SETUP=1 && exec bash runs/basilisk_4b_continue_100b.sh' \
  > /home/l39gu/nanochat-runs/d36_4b_climbmix/basilisk4b-continue-100b.log 2>&1 < /dev/null &
```

Monitor:

```bash
tail -f /home/l39gu/nanochat-runs/d36_4b_climbmix/basilisk4b-continue-100b.log
nvidia-smi
```

## Guardrails

Do not use `runs/basilisk_4b_continue_corrected.sh` for this run. That wrapper
targets the old ratio-12 horizon, not 100B.

Do not change these values on resume:

- `MODEL_TAG`
- `NANOCHAT_BASE_DIR`
- `CLIMBMIX_DATA_DIR`
- `TOTAL_BATCH_SIZE`
- `TARGET_NUM_ITERATIONS`
- `DATASET_SHARDS`
- model architecture settings
- LR/WD/momentum continuation settings

Do not pass `--allow-missing-optimizer-state` for the 100B continuation. The
latest local checkpoints should have all 8 optimizer shards. If any shard is
missing, stop and repair/audit rather than restarting optimizer moments.

`scripts/base_train.py` also refuses to resume if:

- checkpoint metadata has no dataloader state
- checkpoint total batch size differs from the requested total batch size
- checkpoint step is beyond the requested horizon

## Expected Runtime And Progress

Recent speed is about `96.8` seconds/step, or roughly `21.3k` tokens/second.
From step `23036`, remaining time is about 28.5 days if speed is stable.

Checkpointing is every 100 steps. Validation is every 250 steps. The next
durable checkpoint after `23000` should be `23100`; the next validation after
`23000` should be `23250`.

## After Base Reaches 100B

When base pretraining reaches step `48441`, then run base eval, regenerate any
identity data if needed, run SFT from the final base checkpoint, and upload both
the final base and final SFT checkpoints under the agreed `castorini` naming
scheme. Do not reuse the old SFT output from the earlier 45.6B base checkpoint.
