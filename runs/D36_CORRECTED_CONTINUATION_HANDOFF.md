# d36 Corrected Continuation Handoff

## Goal

Continue the current `d36_4b_climbmix_w3` base checkpoint from the already-trained
28.4B tokens to the corrected ratio-12 horizon derived from total trainable
parameters.

The bug was that the token horizon used only `transformer_matrices + lm_head`
(about 2.37B params). It now uses total trainable params (about 3.80B params).

## Current Checkpoint

- Run root: `/u201/l39gu/nanoknow-climbmix/nanochat-runs/d36_4b_climbmix`
- Model tag: `d36_4b_climbmix_w3`
- Resume checkpoint: step `13769`
- Already-trained tokens: `28,424,503,296`
- Saved dataloader position: carried in `meta_013769.json` as `dataloader_state_dict`
- Saved batch size: `2,064,384` tokens
- Optimizer shards available for ranks `0`, `1`, and `2`

## Corrected Horizon

With `TARGET_PARAM_DATA_RATIO=12` and total params `3,803,189,138`:

- Target token budget: `45,638,269,656`
- Total steps with the saved batch size: `22,107`
- Actual total tokens at that step: `45,637,337,088`
- Remaining from step `13,769`: `8,338` optimizer steps
- Remaining tokens: `17,212,833,792`

This is not a fixed 100B-token run. It is the natural ratio-derived continuation.

## Launch

Use the guarded continuation wrapper:

```bash
sbatch runs/watgpu_4b_continue_corrected.sbatch
```

The wrapper:

- Uses `MODEL_TAG=d36_4b_climbmix_w3`
- Requires an existing checkpoint with `REQUIRE_RESUME=1`
- Refuses to resume before step `13769` with `MIN_RESUME_STEP=13769`
- Pins `--total-batch-size=2064384`
- Uses `DEVICE_BATCH_SIZE=7` and `NPROC_PER_NODE=3` to match the saved optimizer shards
- Stops after base pretraining with `STOP_AFTER=base_train`

## No Data Replay Guardrails

The continuation should not replay the first 28.4B tokens because:

- `AUTO_RESUME=1` selects the latest checkpoint for `d36_4b_climbmix_w3`
- `REQUIRE_RESUME=1` exits if no checkpoint is found, instead of starting at shard 0
- `MIN_RESUME_STEP=13769` exits if an older checkpoint is selected
- `scripts/base_train.py` refuses to resume without `dataloader_state_dict`
- `scripts/base_train.py` refuses to resume with a changed `total_batch_size`
- The train loader receives the saved dataloader state and resumes from that corpus position

Do not run this continuation with a fresh model tag unless intentionally starting a new run.

## After Base Continuation

Do not immediately reuse the old SFT output. Regenerate the identity conversations
first: the public identity file describes the old d24/FineWeb speedrun model, not
this d36 ClimbMix model. After the corrected base run finishes, run base eval,
then regenerate identity data, then rerun SFT and chat eval.

## Notes

Under the corrected 45.6B-token horizon, step `13769` is already in the LR
warmdown phase. Monitor early continuation loss after resume; if it spikes, stop
and restart from the saved checkpoint with a smoother LR recovery plan.
