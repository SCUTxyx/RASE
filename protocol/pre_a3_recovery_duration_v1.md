# PRE-A3 Recovery Duration Protocol v1

**Status:** frozen before confirmatory outcomes  
**Date:** 2026-08-04  
**Depends on:** PRE-A0/A1/A2 mechanism audits

## Scientific question

How long must a stronger closed-loop recovery policy remain in control before a
frozen base policy can safely continue, under false-handback harm constraints?

## Cohort

- 120 outcome-independent initial snapshots
- 40 logical tasks: 10 / suite × {Spatial, Object, Goal, Long}
- Each logical task appears under clean:L0, camera:L1, robot:L1
- Official clean-10 names may be reused with new episode seeds
- Camera/robot concrete Plus L1 tasks exclude prior development cohorts
- Design artifact: `runs/rase_pre_a3_design120_v1.json`

## Splits

Per suite task assignment with seed `2026080401`:

| Split | Tasks / suite | States |
|------|---------------|--------|
| train | 6 | 72 |
| val | 2 | 24 |
| test (hidden) | 2 | 24 |

Hidden-test outcomes must not be inspected before analysis code and gates are
frozen. No post-hoc horizon, threshold, or cohort edits.

## Arms

Live closed-loop OFT for:

\[
h \in \{0,8,16,32,64,96,128\}
\]

plus persistent OFT. Deterministic replay is diagnostic only and not main
evidence. All arms share the same snapshot and Smol continuation seed.

## Primary metrics

- finite-duration oracle minus base success (pp)
- task-cluster bootstrap 95% CI on that gap
- rescue/harm paired counts
- best fixed-h harm rate on base successes
- adaptive oracle minus best fixed-h headroom
- persistent-OFT gap / direct-only rescues
- recovery action cost and wall/GPU time

## Confirmatory gate (all required on hidden test)

1. oracle gap ≥ 8pp and bootstrap lower bound > 0
2. ≥ 4 task-disjoint rescues
3. rescues cover ≥ 2 suites and ≥ 2 perturbation cells
4. duration heterogeneity (rescues not all at one minimum h)
5. best fixed-h harm ≤ 5% of base successes
6. adaptive oracle headroom over best fixed-h ≥ 5pp

## Method gates after confirmatory analysis

| Gate | Open only if |
|------|----------------|
| termination / safe-handback model | hidden + val gates pass |
| candidate critic | separate PRE-A3-S opportunity gate passes |
| generative world model | termination gate passed and residual predictive gap proven |

## Kill / stop rules

- Any confirmatory condition fails → `benchmark_diagnosis_only`
- Do not reopen ridge/MLP/RL three-arm selector
- Do not scale same-profile temperature candidates
- Do not train a generative world model to “rescue” a failed duration gate

## Entry points

```bash
python scripts/freeze_pre_a3_design.py
# collect pool from configs/collect_pre_a3_recovery120.json
./scripts/run_pre_a3_recovery_duration.sh
```
