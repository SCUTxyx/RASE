# PRE-C0 Gate A smoke (4 states)

Date: 2026-08-04

## Result

- Exit: `PRE_C0_SMOKE_EXIT:0`
- Output: `runs/rase_pre_c0_same_policy_smoke4_v1` (4 states × T1/T3 path)
- Arms present: current_suffix, strict_resample, fresh_replan, receding_horizon
- Analyzer/bootstrap path exercised; smoke decision is **not** a scientific Gate A claim

## Path checks

- Frozen manifest: `artifacts/pre_c0/pre_c0_48_state_manifest.json`
- Generator accepts `records` / `selected_states`
- Receding horizon uses `RecedingHorizonSmolVLAContinuation`

Proceed to full 48-state Natural Gate A.
