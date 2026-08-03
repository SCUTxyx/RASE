# W9 clean-control and direct-policy selector results

## Status

**Paused / Blocked at coverage gate.** Code gates from the 2026-07-30 handoff
landed in-tree. The preregistered clean-control collection from 2026-07-29
already exhausted the 140-episode hard stop with incomplete suite×`t0_bin`
coverage (`coverage_complete: false`, `n_states: 0`). Direct clean32 labeling,
selector training, and method decision were **not** re-run; re-collecting past
the frozen batch limit would violate the preregistered protocol.

See also: [W9 pipeline](2026-07-29_w9_clean_selector_pipeline.md),
[coverage gate](2026-07-29_w9_clean_control_coverage_gate.md).

## Environment

- Machine: AutoDL (RTX 5090)
- Repository path: `/root/autodl-tmp/RASE`
- Git SHA: `ea7ad403c002302234cf7aa81476bb869e86b586` (working tree has W9 handoff increments; not committed)
- Python / conda env: Python 3.10.8 / `smolvla` (`/root/autodl-tmp/envs/smolvla`); `oft` also present
- GPU: NVIDIA GeForce RTX 5090 (idle; no OFT server)
- LIBERO-plus root used by smoke: `/root/autodl-tmp/src/LIBERO-plus`
- Smol checkpoint: `ckpts/smolvla_libero`
- OFT checkpoints: `ckpts/oft_*` present
- Configs: `configs/collect_w9_clean_controls.json`,
  `configs/ngc_w9_clean_controls.yaml` (protocol unchanged)

## Code gates

Handoff increments applied without resetting prior working-tree changes:

- Focused pytest before GPU in `scripts/run_w9_clean_selector_pipeline.sh`
- Cohort-semantic readiness (`clean_control`∧success / `failure_challenge`∧failure)
- `matched_random_actions` (escalate+abstain count matched) + paired bootstrap CI
- `n_clean_regret_evaluable` denominator
- Task/episode `method_decision` in `scripts/summarize_selector_gate.py`
- Cross-platform path `.resolve()` assert in backend path test
- Fingerprint v2 already present (`rase-task-identity/v2`; no full XML)

- Focused pytest: **60 passed** (handoff list + backend path test)
- Full pytest: **170 passed, 5 skipped**
- Shell syntax / compile: **pass**
- Live fingerprint smoke: **pass**
  (`runs/ngc_w9_fingerprint_smoke_20260730.json`)
  - task A: `libero_spatial_000005` /
    `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_14.bddl`
  - fingerprint stayed
    `3481c90fe95752a6f46c5e537b5502589cdbfe80156f8f5a399a072ccb377856`
    across snapshot + 3 zero actions
  - restore repeat: `image_allclose=true`, `joint_allclose=true`
  - task B: `libero_object_000003`; restore rejected with `TaskMismatchError`
- Cross-task restore rejection: **pass**
- W9 pipeline re-launch: **not started** (coverage gate already frozen at 140)

## Collection

Frozen from 2026-07-29 (do not treat as new evidence from this session):

- Batch 1 (`20260730`): 54 new ep; 7 success / 47 failure
- Batch 2 (`20260731`): 4 success / 36 failure
- Batch 3 (`20260732`): 1 success / 39 failure
- Aggregate: 138 ep, 10/128 success, 26 success states retained
- Successful independent episode-groups by suite/t0 bin: Spatial early/mid only;
  Object/Goal/Long early+mid all empty (Long sole success outside mid bin)
- Frozen clean32 state-key SHA-256: **not created** (`n_states: 0`)

## Direct action outcomes

### Failure challenge

- n: 24 (from `runs/ngc_w9_direct_smol_failure24/summary.json`)
- Direct Smol: 0/24
- Direct OFT on clean32: **not run** (coverage blocked)
- Four-cell Smol×OFT table for this W9 clean pilot: **not available**

### Clean controls

- Direct Smol / OFT clean32: **not run**
- Four-cell table: **not available**

## Selector readiness

- Episode split ready: **not run**
- Task split ready: **not run**
- Reasons if rejected: coverage incomplete before dataset export
- Train/val/test counts: n/a
- Leakage audit: n/a
- Feature list: n/a

## Selector results（仅 ready 时）

Not applicable — readiness / labeling never reached.

## Scientific interpretation

Only prior frozen evidence applies. The 7.2% clean-control success rate versus
the 70% clean LIBERO baseline remains an unresolved task/environment mapping
audit, not a signal to lower `per_cell` or bypass the coverage gate.

## Next gate

Implementation audit complete:
[2026-07-30_w9_clean_collection_implementation_audit.md](2026-07-30_w9_clean_collection_implementation_audit.md).
Root cause: hardcoded `episode_index=0` → always `init_states[0]`. Pool marked
`diagnostic_invalid_for_control`. **Await W9B approval before any recollect.**
