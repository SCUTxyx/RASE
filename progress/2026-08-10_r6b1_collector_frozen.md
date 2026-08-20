# R6-B1 two-stage dynamic-boundary collector FROZEN after pair-parity PASS

Date: 2026-08-10

## Gate result

The two-state Pi0Fast Spatial pair parity rerun **PASSED** with the hard
programmatic gate:

Artifact: `runs/pre_c0_r6/r6b1_smoke_pi0fast_spatial_pair_parity_v2/`
Gate: `runs/pre_c0_r6/r6b1_smoke_pi0fast_spatial_pair_parity_v2/data/parity_audit.json` → `status: pass`

| State | rollout seed | source success | env steps | R6-A ref | Result |
|---|---|---|---|---|---|
| `sp1_0632d5ef6c45e2f304a01f2c133f0bfe` | 997110703 | failure | 270 | 270 | PASS |
| `sp1_0660d272e7256c6b204caf666e94c875` | 154127683 | success | 116 | 116 | PASS |

All four invariants hold for every boundary row: rollout seed identical,
source final success identical, source env steps identical, all saved feature
arrays finite.  No parity failures, no nonfinite files, no missing references.

## Freeze decision

The two-stage collector (snapshot-only during source rollout; features read
post-hoc from restored branch environments) is **frozen** as the production
dynamic-boundary collector.

- Collector: `scripts/collect_r6b1_dynamic_boundaries.py`
  sha256 `0d512d2c5e37fb0af7ba5a5f3b696c87aa1472053c12605b9f6975a6554d61a3`
  (production mode is `--bookkeeping-mode full`; `obs_only` remains only as the
  bisection control that reproduces the historical 149-step drift and must not
  be used for data collection).
- Protocol: `configs/r6b1_dynamic_boundary_protocol_v1.json`
  sha256 `2fe72433239985b2e92ceffdd201748e09900304e5a61f56d419eee97855509c`.
- Runner: `scripts/run_r6b1_pi0fast_pair_parity.sh` (writes to immutable
  attempt dir `r6b1_smoke_pi0fast_spatial_pair_parity_v2`).
- Hard gate: `scripts/audit_r6b1_source_parity.py`.

## Boundary-list regression

Boundary list `{0,16,32,64,96,128}` on the Pi0Fast success state (no oracle,
post-hoc features only) reproduces R6-A exactly: 116 steps, success true,
5 recorded boundaries (128 is beyond the trajectory and correctly not recorded),
all arrays finite
(`runs/pre_c0_r6/r6b1_bisect_v1/full_boundary_set/`).  The earlier failing
configuration with 3 boundaries produced 155 steps; the fixed collector gives
116 regardless of the boundary list.

## Status change

- R6-B1 dynamic collector: **FROZEN**.
- R6-B1.1 pilot manifest (`runs/pre_c0_r6/r6b1_pilot_manifest_v1.json`):
  **un-gated; next executable stage**.  Execute the frozen pilot and gate it
  with `audit_r6b1_pilot.py` (all parity, both label classes per VLA, later
  boundaries, four suites, no NaN/Inf/unexpected trajectories) plus
  `audit_r6b1_source_parity.py` (source parity hard gate).

## Runtime dependency re-verification (2026-08-10, same day)

The pair parity was re-run **after** the snapshot/restore runtime gained full
raw-`mjData` capture (`rase/envs/forkable_env.py`) and passed again:

Artifact: `runs/pre_c0_r6/r6b1_smoke_pi0fast_spatial_pair_parity_v2/data/parity_audit.json`
→ `status: pass`, 0 parity failures, 0 nonfinite files, 2 trajectories checked.

- `sp1_0632d5ef6c45e2f304a01f2c133f0bfe`: 270 steps / failure — matches R6-A.
- `sp1_0660d272e7256c6b204caf666e94c875`: 116 steps / success — matches R6-A.

## Legacy pool restore compatibility (2026-08-10, later same day)

While hardening the side-effect contract, `forkable_env.py` also began
capturing the controller runtime caches (`ee_pos`, `J_pos`, `mass_matrix`,
etc.) in full snapshots.  The pair parity and pilot runners failed against
historical pool bundles (`SnapshotError: controller[0] state keys differ`),
because pre-existing `StatePool` payloads were built with the original
controller field set.  Fix: full snapshots keep the complete field set, while
legacy pool payloads (no `mujoco_data` member) are restored with the legacy
controller field set — no data migration required.

The parity gate was re-run a third time with the final runtime and passed:

- Gate artifact (16:21): `runs/pre_c0_r6/r6b1_smoke_pi0fast_spatial_pair_parity_v2/data/parity_audit.json`
  → `status: pass`, 0 parity failures, 0 nonfinite files, 2 trajectories checked.
- `sp1_0632d5ef6c45e2f304a01f2c133f0bfe`: 270 steps / failure — matches R6-A.
- `sp1_0660d272e7256c6b204caf666e94c875`: 116 steps / success — matches R6-A.
- `collect_r6b1_dynamic_boundaries.py` still byte-identical to the frozen hash;
  only `forkable_env.py` moved (new hash pinned above).  The env-level side-effect
  contract tests and the pool snapshot round-trip / fork integration tests all
  pass (20 passed, 1 skipped).

`collect_r6b1_dynamic_boundaries.py` is byte-identical to the frozen hash
(`0d512d2c...`); only the runtime it depends on changed, and the env-level
side-effect contract tests (`tests/test_r6b1_collector_side_effect_contract.py`)
now assert snapshot interleaving is side-effect-free and two-stage restored
features reproduce live boundary features bit-for-bit.

Frozen runtime hashes (now pinned together with the collector):

- `rase/envs/forkable_env.py` sha256 `445b28c912a70fa56d94e15ae0da3929dc366b68c7dcaa16f28388321715c085`
  (full raw-`mjData` capture; restore swaps in a fresh raw data verbatim and
  does **not** re-`forward()`, preserving the solver-corrected post-step
  geometry the live source rollout rendered; additionally pins the controller
  runtime caches in full snapshots while restoring historical StatePool
  payloads with the legacy controller field set so pre-freeze pool bundles
  remain restorable).
- `rase/envs/snapshot.py` sha256 `6fc680606c8bd58485f442fb5432c39e72d2e54fbc2dd5acaf45007ad7dd268d`
  (snapshot version 2; `mujoco_data` payload member).
- `scripts/audit_r6b1_source_parity.py` sha256 `95299758e564adb547145f7e825e714ac09756f1baedfcc5b2019cdcf027faf8`.
