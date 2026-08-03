# W5 SmolVLA recoverability frontier

> **Completed 2026-07-27:** proposal temperatures 0.3/0.7/1.0 produced
> `0/576` one-shot candidate outcomes on the frozen 24-state failure frontier.
> Candidate temperature metadata and diversity were verified, so this line is
> closed and formal confirm remains intentionally skipped. Continue with
> [`w6_l1_l2_policy_matrix.md`](w6_l1_l2_policy_matrix.md).

W5 is a two-stage diagnostic. A cheap screen locates candidate/state cells with
non-zero SmolVLA hits; only frozen screen hits enter the formal Wilson confirm.
Screen output is never reported as Set A/B/C ground truth.

## 1. Preflight and unit tests

```bash
cd /root/autodl-tmp/RASE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

python scripts/preflight_runner.py \
  --libero-plus-root /root/autodl-tmp/src/LIBERO-plus \
  --checkpoints-root ckpts
pytest -q
```

## 2. Finish W4 summaries without rollouts

```bash
SUMMARY_ONLY=1 ./scripts/run_w4_adequate_pipeline.sh
```

This writes the v2 dual-track summary. SmolVLA keeps Wilson A/B/C semantics;
OFT remains deterministic portfolio ground truth.

## 3. Inspect W5 cell inventory before sampling

`pool/ngc_step1_scale200` currently has **0 retained success snapshots**
(manifest is 3666/3666 failure). The retained-success positive-control config
is blocked on this pool. Use the OFT-recovered smoke cohort instead, then the
failure frontier sample.

```bash
python scripts/sample_state_keys.py \
  --config configs/ngc_w5_failure_frontier_screen.yaml \
  --inventory-only \
  --output runs/ngc_w5_failure_frontier_inventory.json

python scripts/sample_state_keys.py \
  --config configs/ngc_w5_failure_frontier_screen.yaml \
  --output runs/ngc_w5_failure_frontier_state_keys.json

python scripts/export_dual_oracle_split_keys.py \
  --dual-oracle runs/ngc_w4_adequate_dual_oracle_summary.json \
  --split oft_only \
  --output runs/ngc_w5_oft_recovered_state_keys.json
```

Do not proceed if requested failure cells are empty. The failure config uses
`suite×dim×level` with `max_t0=40` because the full middle/late `t0_bin` grid
has empty cells on this pool.

## 4. Proposal-temperature screen

Run the OFT-recovered smoke first (pipeline sanity only; not Set A/B evidence):

```bash
python scripts/generate_pool_candidates.py \
  --config configs/ngc_w5_oft_recovered_smoke.yaml \
  --state-keys-json runs/ngc_w5_oft_recovered_state_keys.json \
  --output-dir runs/ngc_w5_oft_recovered_candidates_t07

python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w5_oft_recovered_smoke.yaml \
  --mode smolvla-screen \
  --state-keys-json runs/ngc_w5_oft_recovered_state_keys.json \
  --candidates-dir runs/ngc_w5_oft_recovered_candidates_t07 \
  --output-dir runs/ngc_w5_oft_recovered_screen_t07 \
  --fresh-run
```

If this smoke cannot even restore/roll out, stop and diagnose
policy/environment restoration before interpreting the failure frontier.

Run each temperature in a separate candidate and rollout directory:

```bash
for tag_temp in t03:0.3 t07:0.7 t10:1.0; do
  tag=${tag_temp%%:*}
  temp=${tag_temp##*:}

  python scripts/generate_pool_candidates.py \
    --config configs/ngc_w5_failure_frontier_screen.yaml \
    --state-keys-json runs/ngc_w5_failure_frontier_state_keys.json \
    --temperature "$temp" \
    --output-dir "runs/ngc_w5_failure_frontier_candidates_${tag}"

  python -u scripts/rollout_pool_candidates.py \
    --config configs/ngc_w5_failure_frontier_screen.yaml \
    --mode smolvla-screen \
    --state-keys-json runs/ngc_w5_failure_frontier_state_keys.json \
    --candidates-dir "runs/ngc_w5_failure_frontier_candidates_${tag}" \
    --output-dir "runs/ngc_w5_failure_frontier_screen_${tag}" \
    --fresh-run
done
```

Freeze the union of hits:

```bash
python scripts/select_screen_hits.py \
  --summary runs/ngc_w5_failure_frontier_screen_t03/summary.json \
  --summary runs/ngc_w5_failure_frontier_screen_t07/summary.json \
  --summary runs/ngc_w5_failure_frontier_screen_t10/summary.json \
  --output runs/ngc_w5_frontier_confirm_state_keys.json
```

## 5. Formal confirm

Generate a fresh candidate artifact for the pre-registered confirm temperature,
then use the frozen `n1=6 -> n=20`, `tau=0.5` protocol:

```bash
python scripts/generate_pool_candidates.py \
  --config configs/ngc_w5_frontier_confirm.yaml \
  --state-keys-json runs/ngc_w5_frontier_confirm_state_keys.json

python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w5_frontier_confirm.yaml \
  --mode smolvla-primary \
  --state-keys-json runs/ngc_w5_frontier_confirm_state_keys.json \
  --fresh-run
```

If the existing L3-L5 pool has no screen hits, collect the optional L1-L2
camera/robot pool:

```bash
python scripts/collect_state_pool.py \
  --config configs/collect_w5_l1_l2_camera_robot.json
```

That command is a new GPU experiment and is intentionally not run by setup or
summary scripts.

## 6. Selected QC traces and dataset export

Trace only explicit states and use a new output directory so durable completed
records are not silently skipped:

```bash
python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w5_frontier_confirm.yaml \
  --mode smolvla-primary \
  --state-keys-json runs/ngc_w5_frontier_confirm_state_keys.json \
  --output-dir runs/ngc_w5_frontier_confirm_video_replay \
  --trace-dir runs/ngc_w5_frontier_traces \
  --trace-all --trace-outcomes all --trace-format archive \
  --fresh-run
```

Convert an archive to MP4:

```bash
pip install -e '.[video]'
python scripts/render_trace_archive.py \
  --trace runs/ngc_w5_frontier_traces/STATE_KEY/c0/r0
```

Export candidate-level labels and split files:

```bash
python scripts/export_recovery_dataset.py \
  --dual-oracle runs/ngc_w4_adequate_dual_oracle_summary.json \
  --pool pool/ngc_step1_scale200 \
  --candidates-dir runs/ngc_w4_adequate_candidates \
  --traces-dir runs/ngc_w5_frontier_traces \
  --output runs/ngc_w4_recovery_dataset.jsonl
```
