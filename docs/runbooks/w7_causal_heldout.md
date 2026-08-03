# W7 prefix attribution and held-out validation

W7 separates mechanism attribution from validation. Never tune the proposal
temperature, candidate count, or intervention definition using held-out results.

## 1. Freeze the W6 OFT-only attribution states

```bash
cd /root/autodl-tmp/RASE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

python scripts/export_policy_matrix_split_keys.py \
  --matrix runs/ngc_w6_l1_l2_policy_matrix.json \
  --label oft_only \
  --output runs/ngc_w7_oft_only_state_keys.json
```

The artifact must contain exactly two states. It is derived from W6 and is for
mechanism attribution only, not an unbiased estimate of recovery prevalence.

## 2. Run the causal action-prefix ablation

The ten deterministic arms are `direct_oft`, time-matched `zero_10`, and the
eight frozen W6 candidate prefixes. Goal and Long are the only represented
suites. The suite runner reuses the normal GPU preflight, OFT server, lock,
fresh-environment restore, and resumable scheduler.

```bash
OFT_RUNNER=prefix-ablation \
OFT_SUITE_SHORTS=goal,10 \
OUTPUT_PREFIX=ngc_w7_prefix_ablation \
STATE_KEYS_JSON=runs/ngc_w7_oft_only_state_keys.json \
CANDIDATES_DIR=runs/ngc_w6_l1_l2_candidates_t07 \
./scripts/run_oft_verify_suites.sh \
  configs/ngc_w7_prefix_ablation.yaml causal
```

Do not set `FRESH_RUN=1` when resuming. Summarize only after both suites finish:

```bash
python scripts/summarize_prefix_ablation.py \
  --state-keys runs/ngc_w7_oft_only_state_keys.json \
  --summary libero_goal=runs/ngc_w7_prefix_ablation_goal_causal/summary.json \
  --summary libero_10=runs/ngc_w7_prefix_ablation_10_causal/summary.json \
  --output-json runs/ngc_w7_prefix_ablation.json \
  --output-md runs/ngc_w7_prefix_ablation.md
```

`candidate_specific_rescue` requires both controls to fail and at least one
candidate to succeed. A direct-OFT success is continuation sufficiency, not
evidence for proposal quality. A zero-prefix success indicates that passive
delay/no-op dynamics are sufficient.

## 3. Freeze the held-out 24-state validation cohort

Every episode group represented by a W6 pilot state is excluded before
sampling, including all other snapshots from those episodes. The target is six
distinct failure episodes in every `camera/robot × L1/L2` cell.

```bash
python scripts/sample_state_keys.py \
  --config configs/ngc_w7_heldout24_screen.yaml \
  --inventory-only --require-complete \
  --output runs/ngc_w7_heldout24_inventory.json

python scripts/sample_state_keys.py \
  --config configs/ngc_w7_heldout24_screen.yaml \
  --output runs/ngc_w7_heldout24_state_keys.json

python scripts/audit_state_key_split.py \
  --reference runs/ngc_w6_l1_l2_state_keys.json \
  --heldout runs/ngc_w7_heldout24_state_keys.json \
  --expected-states 24 --expected-per-cell 6 \
  --output runs/ngc_w7_heldout24_split_audit.json
```

The gates are `coverage_complete=true`, four cells with at least six eligible
distinct episodes, `n_states=24`, 24 distinct selected episode groups, and no
episode-group overlap with the W6 artifact. Exact state-key non-overlap alone is
not sufficient.

## 4. Generate candidates and run the held-out Smol arm

```bash
python scripts/generate_pool_candidates.py \
  --config configs/ngc_w7_heldout24_screen.yaml \
  --state-keys-json runs/ngc_w7_heldout24_state_keys.json \
  --output-dir runs/ngc_w7_heldout24_candidates_t07

python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w7_heldout24_screen.yaml \
  --mode smolvla-screen \
  --state-keys-json runs/ngc_w7_heldout24_state_keys.json \
  --candidates-dir runs/ngc_w7_heldout24_candidates_t07 \
  --output-dir runs/ngc_w7_heldout24_smol_screen_t07 \
  --fresh-run
```

Stop if candidate diversity collapses. Do not select states or candidates from
these outcomes before running the matched OFT arm.

## 5. Run the held-out OFT arm and matrix

The suite runner automatically skips suites absent from the frozen key artifact.

```bash
OUTPUT_PREFIX=ngc_w7_heldout24_oft \
STATE_KEYS_JSON=runs/ngc_w7_heldout24_state_keys.json \
CANDIDATES_DIR=runs/ngc_w7_heldout24_candidates_t07 \
./scripts/run_oft_verify_suites.sh \
  configs/ngc_w7_heldout24_screen.yaml heldout
```

Pass every suite summary that exists to the matrix summarizer. If all four
suites are represented:

```bash
python scripts/summarize_w6_policy_matrix.py \
  --title "W7 held-out L1-L2 paired one-shot policy matrix" \
  --state-keys runs/ngc_w7_heldout24_state_keys.json \
  --smol-summary runs/ngc_w7_heldout24_smol_screen_t07/summary.json \
  --oft-summary Spatial=runs/ngc_w7_heldout24_oft_spatial_heldout/summary.json \
  --oft-summary Object=runs/ngc_w7_heldout24_oft_object_heldout/summary.json \
  --oft-summary Goal=runs/ngc_w7_heldout24_oft_goal_heldout/summary.json \
  --oft-summary Long=runs/ngc_w7_heldout24_oft_10_heldout/summary.json \
  --output-json runs/ngc_w7_heldout24_policy_matrix.json \
  --output-md runs/ngc_w7_heldout24_policy_matrix.md
```

The primary endpoint is the paired state-level OFT-only versus Smol-only count.
Candidate-level rates are descriptive. With 24 states this remains a validation
pilot, but it is independent of W6 and materially stronger than reusing the
eight discovery states.
