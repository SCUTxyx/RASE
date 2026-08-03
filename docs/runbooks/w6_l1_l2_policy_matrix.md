# W6 L1–L2 coverage and policy-matrix pilot

This is the next experiment after the completed W5 temperature sweep. The W5
L3–L5 proposal-temperature line is closed at 0/576 one-shot candidate outcomes.

## 1. Frozen cohort decision

The 2026-07-27 collection completed 40/40 episodes and retained 762 states, but
all 40 episodes failed. The exact eligible distinct-episode counts are camera
L1=11, camera L2=9, robot L1=8, and robot L2=12. This is consistent with the
full Plus collapse map: camera L1/L2 had 0/558 successes and robot L1/L2 had
5/537. Therefore, do not brute-force a success-balanced Plus pool.

W6 uses two non-interchangeable cohorts:

1. **Cohort A, active now:** failure-conditioned L1-L2 challenge states for the
   paired Smol→Smol versus Smol→OFT recovery matrix.
2. **Cohort B, later:** separately sampled clean-success controls for clean
   regret and escalation-cost evaluation.

Never pool the two denominators or report Cohort A as an unconditional NGC rate.

The collection command below is provenance only; the current 40 episodes are
complete and should not be rerun with the same seed.

```bash
cd /root/autodl-tmp/RASE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

python scripts/collect_state_pool.py \
  --config configs/collect_w5_l1_l2_camera_robot.json
```

The collector retains all failure snapshots and a deterministic 20% sample of
success snapshots. A new append-only batch must use a new `--seed`; it is not
required for the current W6 Cohort A gate.

## 2. Hard-audit and freeze failure-challenge keys

```bash
python scripts/sample_state_keys.py \
  --config configs/ngc_w6_l1_l2_screen.yaml \
  --inventory-only \
  --require-complete \
  --output runs/ngc_w6_l1_l2_inventory.json

python scripts/sample_state_keys.py \
  --config configs/ngc_w6_l1_l2_screen.yaml \
  --output runs/ngc_w6_l1_l2_state_keys.json
```

The expected pilot is exactly 8 states: two distinct failure episodes in each
`camera/robot × L1/L2` cell. The audit must report `coverage_complete=true`,
`n_cells=4`, `max_per_cell>=2`, and zero deficit cells. The freeze command must
report `n_states=8`. The script overwrites a stale key artifact with an empty
artifact before exiting non-zero if the gate is incomplete.

## 3. Smol→Smol one-shot screen

Completed on 2026-07-28: 0/64 candidate hits and 0/8 portfolio state hits.
Candidate diversity passed and no retries occurred. Do not rerun this arm or
regenerate its candidates.

```bash
python scripts/generate_pool_candidates.py \
  --config configs/ngc_w6_l1_l2_screen.yaml \
  --state-keys-json runs/ngc_w6_l1_l2_state_keys.json \
  --output-dir runs/ngc_w6_l1_l2_candidates_t07

python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w6_l1_l2_screen.yaml \
  --mode smolvla-screen \
  --state-keys-json runs/ngc_w6_l1_l2_state_keys.json \
  --candidates-dir runs/ngc_w6_l1_l2_candidates_t07 \
  --output-dir runs/ngc_w6_l1_l2_screen_t07 \
  --fresh-run
```

## 4. Smol→OFT deterministic verification

Use exactly the same state keys and SmolVLA candidate artifacts:

First verify that no unrelated process occupies the GPU:

```bash
nvidia-smi
```

The runner requires at least 20,000 MiB free before loading OFT. The first
2026-07-28 attempt did not pass this resource condition and produced no OFT
outcomes. A later launch completed Spatial and Object, then exposed an OFT
runner lifecycle bug after the first terminal success in Goal. That bug is
fixed by restoring every candidate into a fresh environment. Do not bypass the
GPU gate with `ALLOW_BUSY_GPU=1` for a paper run.

```bash
OUTPUT_PREFIX=ngc_w6_l1_l2_oft \
STATE_KEYS_JSON=runs/ngc_w6_l1_l2_state_keys.json \
CANDIDATES_DIR=runs/ngc_w6_l1_l2_candidates_t07 \
./scripts/run_oft_verify_suites.sh \
  configs/ngc_w6_l1_l2_screen.yaml matrix
```

This produces suite-specific `runs/ngc_w6_l1_l2_oft_*_matrix/summary.json`
files. OFT remains deterministic one-shot portfolio evidence, not Wilson Set
certification.

The command is resumable. If a suite is interrupted, run the same command
without `FRESH_RUN=1`: completed scheduler records are reused, including
successful outcomes. Never delete a partial scheduler merely because a later
candidate raised an infrastructure exception.

## 5. Validate and summarize the paired matrix

Run only after all four OFT summaries exist:

```bash
python scripts/summarize_w6_policy_matrix.py \
  --state-keys runs/ngc_w6_l1_l2_state_keys.json \
  --smol-summary runs/ngc_w6_l1_l2_screen_t07/summary.json \
  --oft-summary Spatial=runs/ngc_w6_l1_l2_oft_spatial_matrix/summary.json \
  --oft-summary Object=runs/ngc_w6_l1_l2_oft_object_matrix/summary.json \
  --oft-summary Goal=runs/ngc_w6_l1_l2_oft_goal_matrix/summary.json \
  --oft-summary Long=runs/ngc_w6_l1_l2_oft_10_matrix/summary.json \
  --output-json runs/ngc_w6_l1_l2_policy_matrix.json \
  --output-md runs/ngc_w6_l1_l2_policy_matrix.md
```

The summarizer hard-validates the frozen state-key checksum, pool manifest, and
candidate-directory hash across both arms. State-level portfolio outcomes are
the inferential unit; candidate pairs are descriptive only.

## 6. Confirm only frozen Smol screen hits

The current Smol screen has zero hits, so **do not run confirm**. The commands
below are retained only for a future cohort with a non-empty frozen hit union.

```bash
python scripts/select_screen_hits.py \
  --summary runs/ngc_w6_l1_l2_screen_t07/summary.json \
  --output runs/ngc_w6_l1_l2_confirm_state_keys.json
```

Inspect `n_states`. If it is zero, do not run confirm. If it is positive:

```bash
python scripts/generate_pool_candidates.py \
  --config configs/ngc_w6_l1_l2_confirm.yaml \
  --state-keys-json runs/ngc_w6_l1_l2_confirm_state_keys.json \
  --output-dir runs/ngc_w6_l1_l2_confirm_candidates

python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w6_l1_l2_confirm.yaml \
  --mode smolvla-primary \
  --state-keys-json runs/ngc_w6_l1_l2_confirm_state_keys.json \
  --candidates-dir runs/ngc_w6_l1_l2_confirm_candidates \
  --output-dir runs/ngc_w6_l1_l2_confirm \
  --fresh-run
```

After Sections 1–5 complete, freeze the artifacts before changing sampling or
policy configuration. Cohort B must then be built as a separate clean-success
control artifact before fitting or evaluating the three-arm
`EXECUTE / ESCALATE / ABSTAIN` selector.
