# R6-B1 collector repair: two-stage "snapshot now, features after" restores exact source parity

Date: 2026-08-10

## Root cause (confirmed by bisection, not just hypothesis)

`scripts/collect_r6b1_dynamic_boundaries.py` used to read deployable boundary
features *inside* the source trajectory loop.  The feature read calls
`raw_libero_to_oracle_arrays(main.handle.control_env)` with
`force_update=True`, which invokes robosuite `_update_observables(force=True)`.
Per `robosuite/utils/observables.py:update`, every forced update re-runs the
delayer (`self._current_delay = self._delayer()`, consuming the observable RNG)
and mutates `_sampled` / `_time_since_last_sample`, i.e. it rewrites the
observation sampling schedule the source policy relies on for its next greedy
action.

A `--bookkeeping-mode` control was added to the collector and run on the Pi0Fast
Spatial success state `sp1_0660d272e7256c6b204caf666e94c875` (seed 0,
R6-A reference 116 steps / success):

| bookkeeping mode | source steps | source success | parity |
|---|---|---|---|
| `none` (source only) | 116 | true | PASS |
| `snapshot_only` (`forkable.snapshot()` only) | 116 | true | PASS |
| `obs_only` (in-loop force-updated observable read) | 149 | true | FAIL (same drift as the original collector) |
| `full` (original behavior, `--boundary 0`) | 149 | true | FAIL |
| `full` fixed (two-stage, `--boundary 0 16 32`) | 116 | true | PASS |

`forkable.snapshot()` is read-only (deep-copies sim + observable + RNG state
without calling `_update_observables`), which is why `snapshot_only` keeps
parity.

### Full 4-mode × 3-process batch (`runs/pre_c0_r6/r6b1_bisect_v1/summary.json`)

Every configuration was run in 3 independent processes; all were internally
consistent.  Pi0Fast state `sp1_0660d272e7256c6b204caf666e94c875` (ref 116):

| mode | steps × 3 reps | first action divergence vs `none` |
|---|---|---|
| `none` server=0/1 | 116 / 116 / 116 | — |
| `snapshot_only` server=1 | 116 / 116 / 116 | none |
| `obs_only` server=1 | 149 / 149 / 149 | step 10 (first re-sampled action chunk after the t=0 read) |
| `full` fixed server=0 | 116 / 116 / 116 | none |
| `full` fixed server=1 | 116 / 116 / 116 | none |

Pi0.5 state `sp1_4e5fbfad8277abf157439e0b7a22c06a` (ref 77): `none`/`snapshot_only`/
`full` fixed all reproduce 77/77/77 with no divergence; `obs_only` perturbs the
action trace at step 10 (the step count coincidentally stays 77 for this state,
so the action-level trace is the decisive invariant).  The no-OFT-server control
confirms the OFT server process is not a source of nondeterminism.

## Fix (two-stage collector, `--bookkeeping-mode full`)

- **Stage 1 — source rollout:** only `main.forkable.snapshot()` is recorded at
  each boundary (plus the source action, which is a pure array copy).  Nothing
  else touches the main environment, so the source trajectory is bit-identical
  to R6-A regardless of the boundary list.
- **Stage 2 — after the source outcome is fixed:** each boundary snapshot is
  restored into the separate `branch` environment inside `preserve_rng_state()`;
  deployable features (two RGB views, 8-D proprio, canonical source action
  summary) are read from the restored branch env, then the persistent-OFT
  counterfactual runs.  Verified that the staged features are pixel-identical to
  the live observation at the same boundary (max abs diff 0).

Legacy `obs_only` behavior is retained behind `--bookkeeping-mode obs_only`
only as the bisection control that reproduces the 149-step drift.

## Hard gate

`scripts/audit_r6b1_source_parity.py` compares every collected trajectory
against the frozen R6-A references (`policy_pair_atlas_v1/<policy>/seed_<k>/summary.json`):
rollout seed, source final success, env steps, and finiteness of every saved
feature array; it exits non-zero on any violation.  Wired into
`run_r6b1_pi0fast_pair_parity.sh` and `run_r6b1_pilot.sh`.

Validated on real data: rejects the legacy `obs_only` output
(`env_steps 149` vs expected `116`) and passes the fixed `full` output
(3 boundaries at 0/16/32, all `116`).

## Regression tests

- `tests/test_r6b1_source_parity_audit.py` — 7 unit tests of the hard gate
  (pass on exact reproduction; fail on env-steps mismatch, seed mismatch,
  success mismatch, nonfinite features, missing reference, empty output).
- `tests/test_r6b1_collector_side_effect_contract.py` — opt-in
  (`RASE_TEST_BDDL`) environment-level tests that encode the two properties the
  fix depends on: an interleaved `forkable.snapshot()` is side-effect-free,
  while an in-loop force-updated observable read is not, and two-stage restored
  features reproduce the live boundary observation exactly.

## Status

The R6-B1.0 source-parity gate is again **PASS on the fixed collector**
(single-state, all boundary lists); the full two-state pair parity with the
hard gate is the next rerun before freezing.
