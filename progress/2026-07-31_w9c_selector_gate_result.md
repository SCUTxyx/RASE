# W9C selector gate result

**Date:** 2026-07-31  
**Pipeline:** `scripts/run_w9c_clean_selector_pipeline.sh` → exit 0  
**Artifacts:** `runs/ngc_w9c_selector_gate_summary.{json,md}`

## Collect / coverage

- Pool: `pool/ngc_w9c_clean_controls` (official clean-10 + correct Long language)
- Clean32 keys: `runs/ngc_w9c_clean_control_state_keys.json` — `coverage_complete=true`, n=32
- Direct Smol clean32: **16/32**
- Failure challenge reuses W7/W8 OFT summaries; Smol failure24 refreshed under `runs/ngc_w9c_direct_smol_failure24`

## Direct-action support

| Cohort | n | both | Smol-only | OFT-only | neither |
|---|---:|---:|---:|---:|---:|
| clean_control | 32 | 15 | 1 | 3 | 13 |
| failure_challenge | 24 | 0 | 0 | 9 | 15 |

Optimal labels: continue_smol 16 / escalate_oft 12 / abstain 28.

## Gates

| Gate | Result |
|---|---|
| episode readiness | **ready** |
| task readiness | **ready** |
| task-held-out vs matched-random | **kill_method_branch** (Δutility=0, CI crosses 0) |
| episode-held-out vs matched-random | **kill_method_branch** (Δutility=0) |

## Decision

Per preregistered kill: do **not** promote ridge; do **not** train MLP/RL.  
Paper main track remains **diagnosis / benchmark** (W7/W8). Selector readiness is now unblocked for future work, but method claim is killed.

## Task-held-out limitation

The task-disjoint test has **8 clean-control states, no failure-challenge states,
and zero learned escalation actions** (7 continue / 0 escalate / 1 abstain; no Goal
test state). Learned and action-matched random utility are identical (Δ=0, 95%
bootstrap CI [-0.0075, 0.0075], n=8). This split therefore supports the kill
decision only; it does not establish unseen-task failure routing or escalation.
