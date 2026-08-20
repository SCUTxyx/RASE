# PRE-C0 Gate A results (48-state Natural Same-Policy)

Date: 2026-08-04

## Decision (frozen)

- decision: `run_privileged_guidance_upper_bound`
- natural_same_policy_gate: `closed`
- candidate_critic_gate: `closed`
- natural_headroom_pp: `2.0833333333333335`
- bootstrap_ci95_pp: `[0.0, 6.25]`
- audit: `runs/rase_pre_c0_same_policy_audit_v1.json`
- artifact copy: `artifacts/pre_c0/gate_a_results.json`

## Nested oracle

- n_states: 48
- S0_current: 6
- S1_resample: 6
- S2_replan: 6
- S3_closed_loop: 7

## Headroom (pp)

- sampling: 0.0
- reconditioning: 0.0
- closed_loop: 2.0833333333333335
- natural_total: 2.0833333333333335

## Pass conditions

- `control_harm_le_5pct`: `True`
- `early_late_direction_nonnegative`: `True`
- `natural_headroom_ge_5pp`: `False`
- `rescues_cover_ge_2_suites`: `False`
- `rescues_cover_ge_3_tasks`: `False`

## Bootstrap / sensitivity

- episode-cluster CI95 pp: `[0.0, 6.25]`
- ci95_lower_positive: `False`
- leave-one-task all nonnegative: `True`
- H_adaptive_horizon_pp: `0.0`
- control_harm_rate: `0.0`

## Protocol locks

- Thresholds unchanged from protocol lock (5pp / K / arms / cohort).
- PRE-A3 method gate remains closed; hidden test24 sealed.
- World model gate remains closed.

