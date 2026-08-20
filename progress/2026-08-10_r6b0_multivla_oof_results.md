# R6-B0 Multi-VLA Initial-Boundary Result

Date: 2026-08-10

## Stage decision

R6-A remains **READY**: Pi0Fast and Pi0.5 both passed the two-seed model-free
policy-pair opportunity gate. R6-B0 is **NO-GO as a learned initial-state
controller**: no shared or per-VLA configuration passed the frozen safety/cost
gate in at least four of five task-held-out seeds. Dynamic closed-loop
evaluation, independent validation, test, and the world-model ablation remain
locked.

This result does not reject the real-time risk/correction hypothesis. It rejects
learning state-level failure risk from 48 tasks with only one pre-action state
per task.

## Frozen data and model

- 48 exact pre-action states, 48 true tasks, all four suites;
- three source VLAs, with Pi0Fast and Pi0.5 qualified by R6-A;
- two source-rollout labels per state-policy pair;
- deployable inputs only: two RGB views, 8-D proprioception, canonical ten-step
  action summary, and optional policy ID;
- 95,374 parameters for the shared-ID model;
- Beta-binomial source head, persistent-success head, and ordered
  q10/q50/q90 teacher-cost head;
- three task-bootstrap members, 1.64-sigma LCB, nested task-held-out
  calibration, five fixed training seeds;
- per-policy and overall gates: success gap at least -5pp, false continue at
  most 5%, and teacher-step savings at least 20%.

Dataset: `runs/pre_c0_r6/r6b0_takeover_v1.npz`

SHA-256: `7d53abf2a8bfe1a727adbad875014d64cc9efbaecbfeccacddb7da0c9463997f`

## Main five-seed results

| Configuration | Passing seeds | Mean success gap | Mean false continue | Mean savings | Mean episode AUROC |
|---|---:|---:|---:|---:|---:|
| Shared encoder, per-VLA calibrated threshold | 0/5 | -2.08pp | 7.86% | 48.52% | not used for claim |
| Shared universal, no policy ID | 0/5 | -2.29pp | 8.10% | 47.14% | 0.750 overall |
| Per-VLA Pi0Fast | 0/5 | -7.08pp | 8.10% | 8.43% | 0.478 |
| Per-VLA Pi0.5 | 0/5 | +1.25pp | 7.14% | 76.17% | 0.473 |
| Zero-shot Pi0Fast -> Pi0.5 | 2/5 | +1.04pp | 1.67% | 21.24% | 0.538 |
| Zero-shot Pi0.5 -> Pi0Fast | 0/5 | -37.50pp | 48.10% | 68.85% | 0.545 |
| Leave-Pi0Fast-out | 0/5 | -13.75pp | 19.05% | 35.76% | 0.608 |
| Leave-Pi0.5-out | 1/5 | +1.25pp | 0.95% | 15.32% | 0.558 |

The 0.750 shared-universal AUROC is misleading when pooled across VLAs. Within
each VLA, AUROC is near chance. The network distinguishes the globally strong
Pi0.5 distribution from Pi0Fast, rather than identifying failures within a VLA.

## Post-hoc deployable-language diagnostic

The original B0 feature lock omitted the task instruction even though it is
available at deployment. A clearly marked exploratory v2 adds a deterministic
256-D hashed lexical instruction vector; it changes no images, actions, states,
or outcomes. It cannot retroactively pass the preregistered B0 result.

| Language-conditioned configuration | Passing seeds | Mean success gap | Mean false continue | Mean savings | Mean AUROC |
|---|---:|---:|---:|---:|---:|
| Shared ID + per-VLA calibration | 0/5 | -1.25pp | 6.19% | 51.33% | 0.713 Pi0Fast / 0.543 Pi0.5 |
| Per-VLA Pi0Fast | 1/5 | -5.42pp | 6.19% | 20.15% | 0.702 |
| Per-VLA Pi0.5 | 0/5 | +2.92pp | 7.14% | 85.63% | 0.400 |

Language materially improves Pi0Fast discrimination but not seed stability or
rare Pi0.5 failure detection. The repair therefore remains diagnostic only.

## Opportunity ceiling

The exact gate-constrained cost oracle proves that opportunity remains:

| Policy | Both-seed-safe oracle savings | Gate-constrained oracle savings | Learned result |
|---|---:|---:|---:|
| Pi0Fast | 42.80% | 57.08% | about 8-20%, unstable |
| Pi0.5 | 92.17% | 98.54% | high savings, but 7.14% false continue |

The gap is therefore an observability/sample-support problem, not an absent
cost-aware Pareto opportunity.

## Next locked stage: R6-B1

Collect source-trajectory boundaries rather than more copies of the initial
state. For each qualified VLA and source seed, save boundaries at task progress
and event-triggered hard-negative times. At every boundary record only
deployment-time features, then use simulator forks only for labels:

1. source failure within future horizons 8/16/32 and final source success;
2. persistent-correction success and teacher cost if takeover occurs now;
3. canonical source/corrective action residual and disagreement;
4. instruction, two RGB views, proprioception, action chunk, policy identity;
5. a group ID tying every boundary and counterfactual from one trajectory.

Splits remain task-held-out. Threshold calibration must use cross-fitted
training-task predictions with binomial uncertainty, not six raw calibration
tasks. Two-boundary dwell is evaluated only on complete trajectories. A
pretrained world-model residual/disagreement is still an ablation and is kept
only if it adds a state-level Pareto gain beyond the lightweight observable
baseline.

Independent validation remains locked until at least four of five seeds pass
for both Pi0Fast and Pi0.5. Test and 100+ paired closed-loop episodes remain
sealed until that independent validation passes.
