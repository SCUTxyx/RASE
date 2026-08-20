# R6-C.1 replica-aware decisive OOF: negative result and next decision

Date: 2026-08-12

## Decision

R6-C.1C is complete and fails its predeclared gate.  The current learned
early-window selector must stop.  Do not run shared, shared-ID, shared-calibrated,
zero-shot, leave-one-VLA-out, third-VLA, world-model, independent-validation,
test, or closed-loop experiments from this branch.

This result does not show that all real-time VLA risk control is impossible.
It shows that the present observation representation, Pi0Fast/OFT pair, sample
support and LCB controller cannot jointly preserve persistent-OFT success and
save teacher steps under true task-held-out evaluation.

## Frozen data and label quality

- New reproducibility adjudication: 366 triples; 267 exact, 77 cost-variable,
  18 boundary-label-variable, 4 source-step-variable, 0 seed mismatch, 0 source
  success flip and 0 unresolved third replica.
- Boundary variability: t0 0/366 (0%), t8 9/366 (2.46%), t16 15/366 (4.10%),
  pooled 2.19%.
- Complete t0/t8/t16 sequences are non-monotone in 107/366 (29.2%).  This is a
  real property of the recovery process, not label noise to be projected away.
- Teacher-step q10/q50/q90: 72/214/434.
- Replica-aware dataset: 509 independent policy/seed/state groups, 1,384 rows,
  190 states and 48 true tasks.  Replica distribution by group is 143 single,
  344 double and 22 triple.  All 143 atlas-reference groups pass parity.
- Dataset SHA256:
  `5678e8c6cb811918cb4f1879c2c97da47535426f3dae898305ac1a163f21bd50`.

The training path now aggregates replicas into empirical success counts and
teacher-cost quantiles.  It uses a beta-binomial likelihood, ordered
non-negative q10/q50/q90 cost heads, and never counts replicas as independent
task-held-out examples.

## Policy-specific readiness

Pi0Fast passes label support:

- 190 groups, 117 source failures, 72 early rescues;
- 38 source-failure tasks and 33 early-rescue tasks;
- early rescues across Goal/Long/Object/Spatial: 12/23/16/21;
- natural opportunity: persistent t0 success 82.29%, cost-aware oracle success
  91.67%, oracle gap +9.38pp and teacher savings 55.67%.

Its weak natural-only task-held-out probe also passes readiness: advantage
AUROC 0.736, AP 0.620 at prevalence 0.396, bootstrap AUROC 95% lower bound
0.641, and all four suites above chance.  The probe's false-continue and harm
are diagnostics, not readiness gates; the strict 5% limits remain formal OOF
gates.

Pi0.5 does not pass readiness.  It has 66 source failures but only 18 early
rescues (<20), while the natural advantage probe is near random (AUROC 0.492).
It is retained only as a high-capability safety/harm cohort.  More enrichment
from Pi0.5 is not justified.

## Formal experiment

Configuration:

- policy: Pi0Fast only;
- boundaries: t={0,8,16}, no takeover after t16;
- 5 training seeds {10,11,12,13,14};
- 5 true-task-held-out folds;
- 3 task-bootstrap ensemble members;
- 60 epochs;
- train-fold-only calibration and fold-correct OOF aggregation;
- enrichment train-only, natural cohort calibration/OOF only.

Gate per seed: success gap >= -5pp, false-continue <=5%, absolute paired harm
<=5%, teacher savings >=20%, no suite-concentrated harm.  At least 4/5 seeds
must pass.

Result: **0/5 seeds pass**.

| Metric | Mean across seeds | Formal target |
|---|---:|---:|
| Success gap vs t0 persistent OFT | -11.67pp | >= -5pp |
| False-continue | 13.42% | <= 5% |
| Absolute paired harm (corrected) | 13.54% | <= 5% |
| Teacher-step savings | 21.90% | >= 20% |
| Conditional missed-rescue | 24.09% | report only |

Only mean savings clears its target.  The upper end of the empirical
success-gap interval remains below -5pp (-6.77pp), so this is not a one-seed or
rounding failure.

Post-closeout auditing found that the original 11.04% harm implementation
counted only no-entry missed rescues.  It omitted late entries at t8/t16 that
failed even though the paired t0 persistent baseline succeeded.  Recomputing
harm directly from paired baseline/controller outcomes gives 13.54% mean harm;
2.50pp is explicitly late-entry harm.  Every seed still fails, so the R6
decision is unchanged, but the corrected definition is mandatory from R7 on.

## Mechanism diagnosis

The failure is not caused by unusable labels: replica stability and support
passed.  It is also not the old late-takeover failure: the controller is
restricted to the early window.

At t0, averaged over five training seeds:

| Ranking | AUROC |
|---|---:|
| Source failure, ensemble mean | 0.691 |
| Source failure, LCB | 0.562 |
| Rescue advantage, ensemble mean | 0.607 |
| Rescue advantage, LCB | 0.591 |
| Persistent success | 0.601 |

The source-risk signal exists, but fallback recoverability is weakly
observable.  LCB also degrades source-risk ordering because between-member
dispersion is large: 36--54% of t0 source LCB values clip to zero.  Fold-fitted
risk thresholds range from approximately 0.0004 to 1.01 and some advantage
thresholds fall to the always-enter sentinel, which demonstrates calibration
instability rather than a transferable controller.

Across five seeds there are 53 missed rescues: 25 are blocked by the advantage
LCB, 20 by source-risk, and 8 by both.  Therefore removing only dwell, adding
only more source failures, or tuning only one threshold will not fix the
method.

## What is now locked

1. No shared/multi-VLA training: a universal selector cannot be claimed before
   one within-VLA selector is safe.
2. No world-model promotion: adding residual/disagreement after a failed
   no-WM baseline would be an unregistered rescue attempt and cannot establish
   a clean Pareto contribution.
3. No Pi0.5 primary model: its rescue-positive support and observability fail.
4. No independent validation, test or 100+ closed-loop episodes.
5. No repeated architecture escalation on the same split.

## Revised next plan

### R6 closeout (execute now)

1. Freeze hashes, reports, exclusions and all five seed outputs.
2. Keep the old rep0-only result as a sensitivity analysis; use the
   replica-aware v2 result as canonical.
3. Report the negative result honestly: risk ranking alone is insufficient for
   cost-aware fallback selection because recoverability and uncertainty
   calibration fail task-held-out transfer.
4. Add deterministic baselines and oracle numbers to the final R6 table, but
   do not fit any new threshold on OOF labels.

### R7-A: new opportunity/data protocol, not an R6 model tweak

If the universal-selector idea remains the main research goal, restart it as a
new preregistered data study rather than modifying R6-C.1 after seeing OOF:

- select source/fallback pairs using natural-distribution criteria: source
  success roughly 40--80%, fallback t0 success >=80%, and cost-aware oracle
  savings >=30%; avoid a nearly solved source such as Pi0.5 as the main source;
- collect at least 100 natural rescue-benefit groups per source VLA, spanning
  >=25 true tasks and all suites, with multiple independent states per task;
- include at least two fallback arms so the model learns arm-conditioned
  recoverability rather than equating risk with one OFT policy;
- freeze policy descriptors and a small few-shot calibration budget for each
  VLA; pure zero-shot remains a challenge metric;
- first gate a model-free early-window oracle and a simple task-held-out probe;
  then run exactly one prespecified within-VLA selector before shared training.

The universal architecture should factor into a shared source-risk encoder,
an arm-conditioned recoverability head, and a separately calibrated uncertainty
layer.  The current data show that these are not interchangeable targets.

### R7-B: recovery-student alternative (separate project)

As a complementary line, train a persistent recovery student from OFT
trajectories using LoRA plus DAgger-style student-state relabeling.  This is a
new recovery policy, not an online short-prefix action corrector.  It may reduce
latency/memory and later become another selector arm, but it must not be used to
retroactively rescue the failed R6 selector claim.

## Canonical artifacts

- `runs/pre_c0_r6/r6c1b_label_support.json`
- `runs/pre_c0_r6/r6c1b_pretrain_readiness.json`
- `runs/pre_c0_r6/r6c1b_replica_aggregated_v2/r6c_candidate_arm_dataset.npz.report.json`
- `runs/pre_c0_r6/r6c1_early_selector_oof_v2/per_vla_pi0fast/stability.json`
- `runs/pre_c0_r6/r6c1_early_selector_oof_v2/per_vla_pi0fast/failure_mechanism.json`
