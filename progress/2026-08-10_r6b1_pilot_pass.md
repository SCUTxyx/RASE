# R6-B1.1 frozen dynamic-boundary pilot PASS

Date: 2026-08-10

## Gate result

The R6-B1.1 pilot (16 task-distinct states across four suites, boundaries
`{0,16,32}`, Pi0Fast seed 0, Pi0.5 seeds 0/1, two-stage frozen collector in
`--bookkeeping-mode full`) **PASSED** both hard gates.

Artifact: `runs/pre_c0_r6/r6b1_pilot_v4/`

| Gate | Output | Result |
|---|---|---|
| Pilot audit | `runs/pre_c0_r6/r6b1_pilot_v4/audit.json` | `status: pass` |
| Source parity hard gate | `runs/pre_c0_r6/r6b1_pilot_v4/parity_audit.json` | `status: pass` |

## Pilot audit (`audit.json`)

- 24/24 expected trajectories seen, 72 boundary rows (3 per trajectory).
- 0 parity failures, 0 nonfinite npz files, 0 missing trajectories.
- Per policy: both source label classes present, later boundaries present
  (every trajectory contributes a boundary at elapsed source step 0 plus 16/32).
- All four suites (Spatial / Object / Goal / Long) have data.

| Policy | rows | trajectory groups | source failures | source successes | later boundaries | persistent successes |
|---|---|---|---|---|---|---|
| pi0fast_libero | 24 | 8 | 12 | 12 | 16 | 16 |
| pi05_libero | 48 | 16 | 12 | 36 | 32 | 38 |

## Source parity audit (`parity_audit.json`)

- 72 rows parity-checked against the R6-A frozen atlas
  (`runs/pre_c0_r6/policy_pair_atlas_v1.json`): rollout seed identical, source
  final success identical, source total steps identical.
- 0 parity failures, 0 nonfinite files, 0 missing references, 0 mode violations.

## Status change

- R6-B1.1 frozen pilot: **PASS**.
- Next executable stage: R6-B1.2 full grouped dynamic-boundary collection
  (48 tasks, `h={0,16,32,64,96,128}`, task/trajectory-group split) via
  `scripts/run_r6b1_b1p2_collect.sh`.

## Files

- Collector: `scripts/collect_r6b1_dynamic_boundaries.py`
  sha256 `0d512d2c5e37fb0af7ba5a5f3b696c87aa1472053c12605b9f6975a6554d61a3`
- Pilot runner: `scripts/run_r6b1_pilot.sh`
- Pilot manifest: `runs/pre_c0_r6/r6b1_pilot_manifest_v1.json`
- Gates: `scripts/audit_r6b1_pilot.py`, `scripts/audit_r6b1_source_parity.py`
