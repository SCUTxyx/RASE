# PRE-C1.2 E0 + E1 interim results

Date: 2026-08-05

## E0 successor (locked 9)

- Output: `runs/rase_pre_c1_2_successor_v1.json`
- Decision: **proceed** (`block_training=false`)
- Notes: sim-floor≈0 on perfect restore; interface rule uses robust floor + abs cross threshold to avoid false positives. Env-action MAE across states ≈ 0.03–0.17 (cross-policy gap, not near-identical actions).

## E1 horizon sweep

- Output: `runs/rase_pre_c1_2_horizon_sweep_v1.json`
- All `H ∈ {1,2,4,8,10}`: base **0/9**, adapted **0/9**
- Receding invariants: passed
- Selection: **fallback H=2** (`mode=fallback`)
- Protocol frozen: `selected_horizon=2`, sha256 `87916082ceacf3415bc28c40b0d968c86f5b36924fd05e224907994504c57790`

Interpretation: shortening execution horizon alone does **not** unlock recovery on the C1.1 adapter; proceed to student-query DAgger.
