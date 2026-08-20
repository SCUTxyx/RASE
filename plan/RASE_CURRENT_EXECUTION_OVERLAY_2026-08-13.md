# RASE current execution overlay

Date: 2026-08-13  
Purpose: map the CVPR 2027 long-form plans to the frozen evidence and live server state.

## 1. Authoritative documents

1. Idea: `outputs/RASE_IDEA_DETAILED_CVPR2027.md`.
2. Main plan: `plan/RASE_NEXT_STEPS_EXECUTION_PLAN_CVPR2027.md`.
3. RL/public data: `plan/RASE_RL_AND_PUBLIC_DATASETS_DETAILED_PLAN.md`.
4. Experiment truth: `progress/CURRENT.md` plus frozen progress/audit artifacts.

If a planning document conflicts with a frozen result, the frozen result wins.

## 2. Evidence that constrains the next stage

- PRE-C0-R3: persistent OFT already matches the success-only oracle; operator
  choice must optimize success, harm, latency and correction cost, not success alone.
- R6-C.1: learned switching fails the paired safety gate and late switching
  loses recoverability; threshold/controller tuning is not the bottleneck.
- R7-A: t0 final-failure prediction fails 0/5 seeds (mean AUROC 0.631).
- R8-B: local recoverability-hazard prediction fails 0/5 seeds (mean AUROC 0.568).
- R9-C: unconditional temporal hazard has only 12/480 positives and fails the
  information-support gate; collecting more rows under the same target is not justified.
- R10-B: deterministic case/control labels fail K=3 reproducibility; the
  probability diagnostic is weak (temporal AUROC 0.609, lower bound 0.429).
- R10-B trace diagnostic: first OFT actions are identical, but later closed-loop
  OFT action traces diverge. The current question is where the first input/output
  divergence occurs, not whether a larger classifier can fit the labels.

## 3. Current unlocked work: P0 only

Complete the frozen 18-group, three-replica per-chunk diagnostic in tmux
`r10b_chunk_full`. Audit t8 and t16 for the earliest differing query.

The instrumentation is diagnostic metadata only:

- agent-view SHA-256 and shape;
- wrist-view SHA-256 and shape;
- proprio SHA-256 and shape;
- returned OFT action-chunk SHA-256 and shape;
- query index and executed-action offset.

It is forbidden to expose these post-failure traces as model inputs or training labels.

## 4. Decision immediately after P0

### A. Initial query input differs

Decision: audit snapshot restore, observable delay/sampling state, simulator RNG,
camera rendering and proprio extraction. Do not collect a learning dataset until
the initial boundary contract is exact or the difference is explicitly modeled.

### B. Initial query input matches, later query input differs

Decision: record closed-loop amplification as the root cause. Freeze a new
decision-point data contract around causal state change, action history and
operator value. This is evidence that temporal observation may matter, but it
does not by itself unlock training.

### C. Query inputs match but returned action chunks differ

Decision: audit OFT inference determinism, batching, precision, server state and
RNG. Do not treat the resulting outcome variance as environment stochasticity.

### D. No chunk divergence is reproduced

Decision: keep R10-B failed. Estimate repeat requirements or stop this cohort;
do not select a preferred repeat or revive deterministic labels.

## 5. Gate to unlock the vNext dataset

After the root cause is classified, write and freeze a one-page protocol for
exactly three targets:

1. short-horizon source risk;
2. intervention urgency / recoverability-window loss;
3. correction-operator value under explicit cost.

Before any model training, a model-free opportunity audit must show:

- at least two operators win on non-trivial, reproducible subsets;
- the best fixed operator does not match the privileged per-state oracle after cost;
- opportunity spans enough tasks, suites and both current source VLAs;
- stochastic outcomes are retained as repeat counts/distributions;
- task-held-out folds have support for the targets and operator actions.

If these fail, narrow the claim or redesign collection; do not add model capacity.

## 6. Implementation order after the opportunity gate

1. Freeze a benchmark/policy/action/observation adapter contract; remove the
   hard-coded Hx7 assumption from the shared core while preserving LIBERO parity.
2. Implement correction operators v1: continue, shorten+requery, resample,
   persistent fallback and abort. Replan is deferred.
3. Build a natural decision-point cohort plus event-triggered points. Keep
   root/snapshot groups intact across splits.
4. Run low-capacity information gates against policy+horizon, suite+horizon and
   other outcome-independent priors.
5. Only if the information gate passes, train the small shared risk/urgency/value
   model with task-held-out OOF and abstention.
6. Only if the closed-loop LIBERO safety/success/cost gate passes, evaluate
   zero-shot, unlabeled calibration and lightweight post-training tiers.
7. Only after a stable two-VLA LIBERO result, add a third VLA and then one second benchmark.

## 7. RL and public-data policy

RL is locked until supervised advantage/contextual-bandit baselines demonstrate
sequential benefit and sufficient operator coverage. If unlocked, use RL only
for the correction decision layer while keeping source VLAs frozen. The planned
order is supervised advantage regression, contextual bandit, then shielded
SMDP-IQL; online VLA RL is not a current milestone.

Public data may pretrain temporal visual-proprio representations, action semantics
or OOD support. It cannot provide RASE counterfactual operator labels unless the
same decision state includes comparable correction branches and costs. Priority:
RoboMIND/failure-aware data for representation, RoboTwin 2.0 for controllable
branch generation, then Open X-Embodiment/DROID/BridgeData for broad pretraining.

## 8. Explicit locks

Until their preceding gates pass, do not start:

- larger risk models or threshold sweeps;
- selector or offline RL training;
- world-model rescue experiments;
- handback optimization;
- validation/test evaluation;
- a third VLA or new benchmark rollout campaign;
- public-dataset downloads or large preprocessing jobs.

## 9. Immediate deliverables

1. `r10b_chunk_input_divergence_audit_v1.json`.
2. A concise progress record with root-cause classification and artifact hashes.
3. A frozen vNext target/data protocol, only if the diagnostic supports it.
4. A model-free correction-opportunity preregistration before new collection.

