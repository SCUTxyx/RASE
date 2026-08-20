# R6-C.1B: Targeted Re-collection — Infrastructure & Screening Status

Date: 2026-08-10  
Status: infrastructure complete; source-only screening in flight; targeted
OFT-labelled collection + dataset merge + R6-C.1C OOF queued behind it.

## Scientific scope (methodology red lines)

R6-C.1B implements the revised plan's separation rules:

1. **strict parity vs reproducibility.** Only existing `(state, policy, seed)`
   triples with a frozen R6-A reference are strict-parity rechecked.  New
   states/seeds have no historical reference, so they get a repeat-run
   reproducibility protocol instead (never called parity).
2. **enrichment vs natural eval.** 48 natural-development-eval states were
   selected purely from pre-registered pool metadata (suite, robot/camera
   perturbation, level, task); screening-selected hard cases go only into
   `train_enrichment` and are over-sampled in training folds, never in the
   natural-distribution gate.
3. **fold binding.** All states, seeds and replicas of the same `task_id`
   stay in the same outer fold.
4. **early-window boundaries.** New collection records `t={0,8,16}` only;
   decisions after t16 lock the source (no emergency trigger).

## Frozen manifest

`runs/pre_c0_r6/r6c1b_initial_keys_v1.json` (created by
`scripts/freeze_r6c1b_initial_keys.py`):

| cohort | states | tasks | selection |
|---|---|---|---|
| `natural_development_eval` | 48 | 48 | pool metadata only (no outcome-based selection) |
| `train_enrichment` | 96 | 48 | screening-selected hard cases (source failure / difficult success) |

Pool: `runs/rase_ui_phase1a_replacement48_initial_pool_v2` (740 candidate
snapshots).  Every state is validated via `StatePool.read_state`.

## Reproducibility protocol (non-parity)

- Existing atlas states/seeds keep strict parity
  (`audit_r6b1_source_parity.py` + `r6b1_b12_exclusions_v1.json`).
- New triples are collected twice (`--rollout-index 0` and `1`); if success /
  terminal steps / boundary labels disagree, a third run is triggered.
- Success flips are excluded or kept as probabilistic-label groups (never hard
  labels); step-only differences keep all replicas with median/quantile cost
  supervision.
- `scripts/audit_r6c1b_repro.py` classifies triples
  (reproducible / step_diff / success_flip / pending) and writes the extended
  exclusion manifest `runs/pre_c0_r6/r6c1b_repro_exclusions_v1.json`.

## Source-only screening (in flight)

`scripts/run_r6c1b_screen.sh` runs the cheapest mode (`--no-oracle`,
`--bookkeeping-mode none`) over Pi0.5 seeds 2-3 and Pi0Fast seed 1 for all four
suites at `t={0,8,16}`.  Purpose: select hard cases for `train_enrichment`
and give the natural eval cohort its source outcomes.

Early observed failure rates on the new Pi0.5 states are well above the B1.2
seed-0/1 support (Spatial: ~8-17% of boundary rows fail), confirming the
targeted hard-state selection is needed and effective.

Partial failure support at 39% of the 432 rollouts (Spatial+Object done,
Goal+Long queued):

| policy | screened files | failure rows | failure groups |
|---|---|---|---|
| pi05_libero | 136 | 81 | 27 |
| pi0fast_libero | 36 | 63 | 21 |

These already approach the per-VLA >=30 failure-group target before the two
remaining suites; the enrichment hard-case filter (source failure OR difficult
success, i.e. still running at the final t16 boundary) keeps exactly the
states that enter `train_enrichment`.

## Targeted OFT collection (queued)

`scripts/run_r6c1b_collect.sh`:

- starts the OpenVLA-OFT oracle server per suite and probes it;
- collects `t={0,8,16}` persistent-OFT labels on the natural eval cohort and
  on the screening-kept enrichment hard cases, twice per triple (replicas);
- is idempotent (skips batches whose per-state metadata already exists);
- ends with the reproducibility audit and extended exclusion manifest.

## Merged dataset (queued)

`scripts/build_candidate_arm_dataset.py --input-root B1.2 --input-root collect`
merges the two collections.  Groups without an R6-A reference are recorded as
`no_reference` and validated by the reproducibility audit instead of strict
parity.  Duplicate groups are skipped; all same-task groups land in one fold.

## Gate targets (enrichment-internal; natural distribution not balanced)

- every VLA >= 30 source-failure groups;
- every VLA >= 20 early-rescuable groups;
- every suite has source failure and matched success.

## Queued next stage

R6-C.1C early-window selector OOF (5 seeds, R6-C.1 stage gate) — see
`scripts/run_r6c1_early_selector_oof.sh` and
`scripts/audit_r6c1_selector_stability.py`.

## Related artifacts

- `scripts/freeze_r6c1b_initial_keys.py`
- `scripts/run_r6c1b_screen.sh`
- `scripts/run_r6c1b_collect.sh`
- `scripts/audit_r6c1b_repro.py`
- `scripts/run_r6c1b_resume.sh` (post-screening orchestration)
- `configs/r6c1b_dynamic_boundary_protocol_v1.json`
