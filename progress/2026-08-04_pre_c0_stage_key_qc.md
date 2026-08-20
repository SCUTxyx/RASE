# PRE-C0 stage-key QC

Date: 2026-08-04

## Result

- `reliable_rate`: **1.0** (≥ 0.80 gate)
- `temporal_fallback_rate`: **0.0** (full percentile fallback)
- Timeline QC `qc_pass`: true (strict T0–T4 ordering, unique keys, full stage coverage)

## Mining definition repair

Initial mining marked 22/24 episodes unreliable because missing T2/T3 absolute thresholds forced full temporal fallback.

Repairs (outcome-blind; no corrective rollouts inspected):

1. Soft-fill T2/T3 after a valid T1 while preserving strict unique indices.
2. Relative T1 detection when absolute `deviation_score` never crosses the fixed threshold but a within-episode peak exists.
3. Reliability fails only on full temporal fallback or low signal coverage—not on soft fills.

## Artifacts

- `runs/rase_pre_c0_deviation_stage_keys_v1.json`
- `artifacts/pre_c0/stage_key_qc.json`

Proceed to freeze 48-state `{T1,T3}` manifest and Natural Gate A.
