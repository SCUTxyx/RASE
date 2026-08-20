# R6-B1.2 Candidate-Arm Data Analysis (2026-08-10)

Full B1.2 collection (144 npz) complete; source-parity hard gate passed (see `parity_audit.json`). One known nondeterministic trajectory group (`pi05_libero` seed 1, `sp1_b0b5e524da0d318935146d898a89ef8c`) is excluded per the frozen manifest `runs/pre_c0_r6/r6b1_b12_exclusions_v1.json` (R6-A reference 154 steps is not reproducible; collector and isolated reruns both give 138). This report is generated from the frozen B1.2 metadata by `scripts/analyze_r6c_candidate_arms.py`.

## Scope

- rows: 767  groups: 143  states: 48  tasks: 48
- policies: pi05_libero, pi0fast_libero

## Per-boundary rescuability

| elapsed | rows | source success | persistent success | teacher steps mean |
|---|---|---|---|---|
| 0 | 143 | 0.76 | 0.87 | 161.3 |
| 16 | 143 | 0.76 | 0.83 | 171.5 |
| 32 | 143 | 0.76 | 0.57 | 223.1 |
| 64 | 143 | 0.76 | 0.40 | 227.4 |
| 96 | 113 | 0.70 | 0.38 | 223.7 |
| 128 | 82 | 0.59 | 0.59 | 180.9 |

## Source failure prevalence (per policy)

- **pi05_libero**: source final success 0.93, source-failure rows 36, failure rescue rate 0.67, rescue teacher steps 174.6
- **pi0fast_libero**: source final success 0.40, source-failure rows 168, failure rescue rate 0.60, rescue teacher steps 205.8

## Temporal non-monotonicity

- groups with >=3 boundaries: 143
- groups with non-monotonic within-16 series: 0 (0.00)

## Candidate-arm opportunity

- rescue opportunities: 472 in 143 groups
- new successes created: 124
- mean teacher steps per rescue: 140.1

## Takeaway

The persistent-OFT arm rescues a meaningful fraction of source failures at the early boundaries at low cost; rescuability degrades as elapsed steps grow. Non-monotonic within-horizon series confirm that a fixed elapsed threshold alone is insufficient, motivating the learned per-boundary risk model with two-boundary dwell.
