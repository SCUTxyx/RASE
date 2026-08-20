# R4-D protocol re-audit and action-conditioned delta results

Date: 2026-08-09  
Status: **ORIGINAL R4-D CLAIM INVALIDATED; CORRECTED OFFLINE GATE FAIL**

## Research question

Does the 0.6M LightRiskStudent safely recover persistent-OFT success while saving at least
20% OFT steps, and does correctly paired V-JEPA 2-AC action-conditioned delta improve the
success-cost Pareto?

## Why this record supersedes the previous R4-D interpretation

The original engineering artifacts remain intact, but code/data audit found six formal
evaluation failures:

1. the model was initialized outside the outer-CV loop, so later validation tasks had been
   seen in earlier folds;
2. ensemble members were not independently initialized/trained, and task bootstrap kept only
   one row from each sampled task;
3. threshold calibration was infeasible and silently fell back to 0.5 in all folds;
4. 442 boundaries were evaluated as independent decisions although each of 71 states permits
   only one earliest handback;
5. M5 used `policy_success = baseline | handback`, so selecting handback could never reduce
   success, and its cost bootstrap resampled a constant aggregate;
6. the alleged M4 holdout overlaps training in 24/24 states and 144/144 boundary keys.

The original V-JEPA cache was also invalid for the intended ablation: 1407/1407 records had
identical Student/OFT deltas because the world-model window contained no Student chunk, and
the trainer read a missing `window_start` field. Therefore the old V-JEPA replacement AUC
0.573 did not test action-conditioned distillation.

## Corrected protocol

- 5-fold outer split by true task ID;
- independent model initialization per fold and per ensemble member;
- complete-task cluster bootstrap;
- train-only normalization;
- inner calibration tasks only;
- state-level earliest-handback evaluation;
- fold-specific thresholds retained for OOF decisions;
- correct handback-vs-persistent success accounting;
- task-cluster bootstrap with add-one p-value correction;
- fail-closed state/task/key-overlap and teacher-evidence audits.

Data: 442 boundary rows, 71 states, 24 tasks, four suites.  
Seed: 20260809.  
Persistent reference: 65/71 = 91.55% success, 11,049 executed OFT steps.

## Corrected results

| Model | Input dim | Row AUC | Policy success | Gap vs persistent | Conditional false handback | OFT savings | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| hard-label baseline | 128 | 0.9221 | 84.51% | -7.04pp | 7.69% | 21.09% | FAIL |
| baseline + projected V-JEPA deltas | 229 | 0.9066 | 78.87% | -12.68pp | 13.85% | 23.78% | FAIL |
| baseline + five V-JEPA delta scalars | 133 | 0.9239 | 83.10% | -8.45pp | 9.23% | 19.02% | FAIL |

The hard-label model retains real ranking signal but fails both success non-inferiority and
the 5% false-handback constraint. High row AUC is therefore insufficient evidence for a safe
stopping controller.

The rebuilt teacher cache passed integrity checks: 442/442 exact boundary keys and 442/442
distinct Student/OFT deltas. Projected delta versus baseline changed policy success by
-5.63pp (task-bootstrap 95% CI [-15.28pp, +2.86pp]) and savings by +2.70pp
(95% CI [-3.76pp, +9.95pp]). Five scalar summaries also failed to improve either outcome.

## Interpretation

The immediate bottleneck is risk calibration/sample size, not representation ranking. Each
fold has only 11-12 calibration states; one false handback already exceeds a 5% constraint.
This produces thresholds spanning roughly 0.34-0.9998 and unstable control behavior.

Correctly paired single-step V-JEPA delta is action-sensitive but does not improve the frozen
success-cost Pareto at this sample size. It is removed from the main method. A future world-
model experiment is allowed only as one preregistered next-latent prediction probe using
time-aligned real next frames and K={1,4,8}; no further post-hoc projection search is allowed.

## Go / No-Go

- Existing 0.6M checkpoint as engineering/export artifact: **retain**.
- Existing checkpoint as validated deployment model: **NO-GO**.
- Original M4/M5 and `p=0.0` claim: **invalidated; do not cite**.
- Lightweight hard-label risk monitor: **GO only for expanded-data development**.
- V-JEPA delta in the main model: **NO-GO**.
- Large closed-loop evaluation: **blocked until all corrected offline gates pass**.

## Next execution order

1. freeze split manifests before collection and require protocol audit on every run;
2. collect at least 300 development states, at least 100 persistent-rescuable calibration
   states, and at least 100 truly task-disjoint frozen-test states;
3. add a second VLA and evaluate shared risk heads with VLA-specific canonical action adapters;
4. train handback-success, persistent-success and cost-quantile heads; remove the duplicated
   `unsafe_ood = 1 - success` target;
5. calibrate a lower-confidence-bound controller with two-boundary dwell/hysteresis;
6. enter 100+ paired closed-loop episodes only after success gap >= -5pp, conditional false
   handback <=5%, and OFT savings >=20% all pass with task-cluster intervals.

## Artifacts

- `runs/pre_c0_r4/light_student_v2_protocol_baseline/`
- `runs/pre_c0_r4/teacher_evidence_boundary_v2/`
- `runs/pre_c0_r4/light_student_v2_protocol_delta/`
- `runs/pre_c0_r4/light_student_v2_protocol_delta_scalar/`
- `runs/pre_c0_r4/protocol_audit_old_v1.json`
- `runs/pre_c0_r4/protocol_audit_teacher_v2.json`
- `runs/pre_c0_r4/PROTOCOL_AUDIT_AND_R5_PLAN_2026-08-09.md`

Corrected scripts:

- `scripts/audit_r4d_protocol_v2.py`
- `scripts/cache_r4d_teacher_evidence_v2.py`
- `scripts/train_r4d_light_risk_student_v2.py`
- `scripts/eval_r4d_offline_oof_v2.py`
- `scripts/compare_r4d_oof_v2.py`

