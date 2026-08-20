# PRE-A3 confirmatory method gate — 2026-08-04

## Status
**Step 4 closed without test unblinding (val NO-GO).**

## Decision
- `runs/rase_pre_a3_method_gate_confirmatory_v1.json`
- status: **FAIL**
- track: `benchmark_diagnosis_only`
- `pre_b_allowed`: false
- world_model / critic / termination gates: **closed**
- reason: val go/no-go failed (`adaptive_headroom_ge_5pp`); hidden test not unblinded

## What this means
Duration structure and rescue coverage appear on the task-disjoint val split, but adaptive headroom over best fixed duration does not clear the preregistered 5pp gate. Per protocol: stop method line; do not open PRE-B safe-handback; do not train world model / critic.

## Constraint
No post-hoc edits to h / cohort / gates after inspecting val outcomes.
