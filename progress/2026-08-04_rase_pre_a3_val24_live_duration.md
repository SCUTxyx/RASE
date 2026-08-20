# PRE-A3 val24 live duration — 2026-08-04

## Status
**Step 3 complete. GO/NO-GO = NOGO.**

## Outputs
- run: `runs/rase_pre_a3_recovery_duration120_val_v1/` (192 LIVE_DURATION lines = 24×8)
- audit: `runs/rase_pre_a3_recovery_duration_audit120_val_v1/audit_val.json`
- snapshot: `runs/rase_pre_a3_val_gating_snapshot.json`
- decision file: `runs/rase_pre_a3_val_go_nogo.txt` → **NOGO**

## Val gate (frozen thresholds; no retune)
- `gate_pass`: false
- status: `duration_structure_signal_unconfirmed`
- pass_conditions:
  - oracle_gap_ge_8pp: **true** (oracle−base = 50.0 pp)
  - rescues_ge_4_task_disjoint: **true** (12 fixed-duration rescues)
  - rescues_cover_ge_2_suites: **true** (Goal/Long/Object/Spatial)
  - rescues_cover_ge_2_cells: **true** (camera/clean/robot)
  - duration_heterogeneity: **true**
  - best_fixed_harm_le_5pct: **true** (0.0)
  - adaptive_headroom_ge_5pp: **false** ← blocking

## Constraint honored
No h / cohort / gate retune after val. Hidden test **not** unblinded.
