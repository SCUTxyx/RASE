# PRE-A3 live closed-loop smoke4 results — 2026-08-04

## Status

**Plumbing PASS; confirmatory gate remains closed.**

Live OFT prefix → Smol handback runner, suite orchestration, merge, and PRE-A3
analysis/method-gate path all completed on a 4-state development subset.

## Results

| State focus | h=0 | h=8 | h=16 | h=32 | Persistent OFT |
|-------------|----:|----:|-----:|-----:|---------------:|
| Spatial clean | 1 | 1 | 1 | 0 | 1 |
| Spatial camera | 0 | 0 | 0 | 0 | 1 |
| Spatial robot | 0 | 0 | 0 | 0 | 1 |
| Object clean | 1 | 1 | 1 | 1 | 1 |
| **Total /4** | **2** | **2** | **2** | **1** | **4** |

- Finite-duration rescues: 0
- Direct-only / persistent rescues: 2
- False-handback harm at h=32: 1 base success flipped to failure
- Audit status (`all`): `episode_persistent_fallback`
- Method gate: `benchmark_diagnosis_only`, `termination_model_gate=closed`

## Interpretation

Smoke confirms:

1. live closed-loop OFT prefixes execute and hand back to Smol;
2. analysis emits rescue/harm/persistent-gap fields;
3. preregistered gate correctly refuses to open on a tiny non-confirmatory set.

It does **not** replace the frozen 120-state task-disjoint confirmatory study.

## Artifacts

- `runs/rase_pre_a3_smoke4_live_duration_v1/summary.json`
- `runs/rase_pre_a3_smoke4_audit_v1/audit_*.json`
- `runs/rase_pre_a3_smoke4_audit_v1/method_gate.json`
- `runs/rase_pre_a3_smoke4_live_duration_v1.log`

## Next

Collect `runs/rase_pre_a3_recovery120_pool_v1` from
`runs/rase_pre_a3_design120_v1.json`, freeze keys, then run
`./scripts/run_pre_a3_recovery_duration.sh` for hidden confirmation.
