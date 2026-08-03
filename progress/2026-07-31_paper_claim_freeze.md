# Paper claim freeze (W7/W8 primary; selector gated)

**Date:** 2026-07-31  
**Track decision rule:** W9C probe+selector gate → Method+Benchmark; else **Benchmark / diagnosis** main track.

## Claim spine (allowed)

1. **Policy-relative recoverability:** under matched held-out failure prefixes, weak-policy continuation fails while strong-policy continuation can succeed.
2. **Direct escalation:** `CONTINUE_SMOL / ESCALATE_OFT / ABSTAIN` is the deployable action space; OFT can succeed without Smol candidate proposals.
3. **Candidate-specific rescue does not hold** in the tested regime (mechanism ablation rescue = 0).

## Main-table evidence (frozen; no new GPU required)

| Result | Evidence |
|---|---|
| Clean SmolVLA baseline ~70% | `lerobot-eval` clean LIBERO |
| Plus collapse ~0.38% | prior Plus eval |
| W4 asymmetry | existing W4 artifacts |
| W5 temperature falsification | existing W5 |
| W7 held-out | Smol direct **0/24** vs prefix+OFT **8/24** |
| W8 direct OFT | **9/24** |
| Candidate-specific rescue | **0** |

## Negative results (must report)

- Selector readiness historically `NOT_READY` without valid clean controls
- W9A: `diagnostic_invalid_for_control` (hardcoded `episode_index=0`)
- W9B: `diagnostic_wrong_task_identity` (Plus index 0–9 ≠ clean-10)

## Claim boundaries (do not claim)

- Do not claim any-of-K portfolio labels as the deploy policy
- Do not claim ridge/MLP/RL selector until W9C readiness + task-held-out ≥ matched-random
- Do not mix W9A/W9B pools into training controls
- Do not report failure-conditioned recovery alone as the headline metric

## Selector branch (conditional)

Only if W9C probe passes and `run_w9c_clean_selector_pipeline.sh` clears readiness + held-out gate:

- Report task/net success, FEB, clean regret, strong-policy usage, latency
- Otherwise: selector = appendix / future work; paper = diagnosis + benchmark

## Decision log

| Gate | Outcome | Paper posture |
|---|---|---|
| Code fix (this doc’s sibling) | landed 2026-07-31 | enables probe |
| Probe | **PASS** (mean SR 0.6125; Object/Goal restored; Long after language fix) | unlocks W9C collect+selector |
| Collect + clean32 coverage | **PASS** (`coverage_complete`, n=32) | enables readiness |
| Readiness | **PASS** (episode + task ready) | enables ridge MVP |
| Ridge vs matched-random | **KILL** (`kill_method_branch`, Δutility=0 on task-held-out) | **diagnosis/benchmark main track**; selector = future work / appendix |
| W10 Object/Spatial failure benchmark | **COMPLETED / split NOT_READY** (Smol 0/16, OFT 1/16; escalate oracle=1) | suite-coverage diagnosis: Object/Spatial L1–L2 failures are mostly both-fail; do not broaden Goal/Long recoverability claims; selector remains closed |

**Final paper posture (2026-07-31):** Benchmark + diagnosis (W7/W8 recoverability & direct escalation). Do not claim a learned selector. Do not escalate to MLP/RL.

### W10 addendum (same day)

Object/Spatial direct-escalation coverage is now measured and mostly negative:
failure-challenge paired outcomes are both 0 / Smol-only 0 / OFT-only 1 / neither 15.
Positive recoverability claims must remain centered on Goal/Long evidence unless a
new preregistered cohort shows otherwise. See
`progress/2026-07-31_w10_object_spatial_benchmark.md`.
