# R8: Local recoverability-hazard restart

Date: 2026-08-13

## Why R8 replaces further R7 source-risk tuning

R7-A remains a formal negative result.  After the late-divergence exclusion and
amended 191-state exact-repeat gate, the Pi0Fast t=0 final-failure model passed
0/5 seeds (mean task-held-out AUROC 0.631).  The project must therefore not keep
increasing model complexity for the same weak episode-level target.

R8 changes the prediction target, not merely the network.  It asks an online,
local question: if the source executes the next eight steps, will the currently
available persistent fallback cease to be recoverable?  The deployable inputs
remain the current observation, proprioception, source action proposal, causal
action history, elapsed context, instruction, and a small seen-VLA identity
calibration.  Future frames, OFT actions, teacher cost, world-model features,
task ordinals, and outcome labels are forbidden inputs.

## R8-A0 model-free opportunity and label-support audit

Frozen aggregate input:

- 1,384 boundary rows;
- 509 trajectory groups;
- 190 states / 48 tasks;
- Pi0.5 and Pi0Fast;
- complete t={0,8,16} sequences for 366 groups;
- dataset SHA-256
  `5678e8c6cb811918cb4f1879c2c97da47535426f3dae898305ac1a163f21bd50`.

Formal R8-A0 decision: **PASS — unlock only the label-stability pilot.**

- hard local recoverability-loss groups: 159 / 366;
- contributing tasks: 36;
- suite support: Goal 52, Long 24, Object 34, Spatial 49;
- ambiguous transition fraction: 3.69%;
- 0→8 hard losses: 91;
- 8→16 hard losses: 68;
- gain-after-loss sequences: 70 / 366 = 19.13%.

The non-monotonicity is scientifically important: recoverability is not a
monotone countdown.  R8 must learn an action-conditioned local transition; an
irreversible hazard assumption is invalid.

Natural-cohort privileged early-window opportunity also passed for both pairs:

| Source | Groups | Oracle success | t0 fallback success | Gap | Teacher savings |
|---|---:|---:|---:|---:|---:|
| Pi0.5 | 96 | 98.96% | 77.08% | +21.88pp | 94.67% |
| Pi0Fast | 48 | 85.42% | 77.08% | +8.33pp | 60.54% |

These are privileged upper bounds, not learned-selector results.

## R8-A1 fixed third-replica stability pilot

Formal result: **PASS**.

Before observing third-replica outcomes, 32 stable K=2 groups were selected by
a frozen hash salt, balanced over:

- two source policies;
- four suites;
- hazard-positive versus control;
- two groups per stratum.

The manifest binds the canonical rep0 metadata and NPZ hashes, rollout seed,
identity, and K=2 labels.  Every group receives exactly one third rollout using
the same seed/checkpoint and boundaries t={0,8,16}.  t=0 feature parity is a hard
gate.  Full late action-trace equality is not required because R8 explicitly
models stochastic future outcomes.

Gate:

- protocol/parity errors: stop;
- third-trial boundary disagreement Wilson upper 95% >10%: stop for label
  instability;
- otherwise, if any K=3 label is mixed, collect exactly two more replicas only
  for those predeclared mixed groups and freeze K=5 probabilities;
- otherwise freeze the K=3 pilot.

No repeated collection until a preferred outcome is allowed.

Observed result:

- audited records: 32/32;
- t=0 feature/identity/seed protocol errors: 0;
- persistent boundary third-trial disagreements: 0/96;
- disagreement Wilson upper 95%: 3.85%, below the frozen 10% gate;
- source-outcome third-trial disagreements: 0/32;
- K=5 expansion groups: 0.

The K=2 labels used by R8-B are therefore reproducible in this balanced pilot.

## R8-B frozen no-world-model probe

The protocol was frozen before R8-A1 outcomes at
`configs/r8b_local_recoverability_hazard_probe_v1.json`.

Primary model: lightweight shared state/action/history core plus a 16-D seen-VLA
ID calibration embedding, with three heads:

1. current fallback recoverability;
2. recoverability after eight more source steps;
3. conditional local loss hazard.

Evaluation is five-fold task held out.  Fit tasks may use natural and enrichment
rows; calibration and validation are natural only.  Three task-bootstrap members
are used per fold.  Five fixed seeds are required and at least 4/5 must pass all
AUROC, bootstrap, AP, calibration, per-policy, per-horizon, and per-suite gates.

## R8-B formal result

Canonical five-seed OOF result: **FAIL, 0/5 seeds — stop local-hazard model
escalation.**

Across seeds:

- conditional hazard AUROC: mean 0.568, range 0.550–0.588;
- AP above prevalence: mean +0.0469, range +0.0221 to +0.0755;
- ECE: mean 0.196, range 0.178–0.226;
- current-recoverability AUROC: 0.465–0.534, approximately random;
- task-bootstrap hazard AUROC lower bounds: approximately 0.41, far below 0.60.

Breakdown shows a representation/generalization failure rather than a threshold
or calibration failure:

- t=0 hazard AUROC is only 0.407–0.490;
- t=8 hazard AUROC is 0.559–0.683;
- Pi0Fast is somewhat stronger than Pi0.5, but neither satisfies all gates;
- Goal AUROC is 0.802–0.926, while Long is 0.208–0.405 and Object is mostly
  0.371–0.596;
- fold AUROC ranges from about 0.316 to 0.976, indicating severe task-composition
  sensitivity.

Platt calibration cannot explain poor AUROC because positive-temperature Platt
scaling preserves ranking.  The frozen current observation, source action
proposal, four-step action history, elapsed context, instruction hash, and small
VLA ID do not provide a task-general local hazard signal.

Consequences of the pre-registered decision:

- do not run shared-no-ID or per-VLA comparator training as method candidates;
- do not train a selector on these predictions;
- do not add world-model residual/disagreement to rescue this model;
- independent validation, test, and closed-loop evaluation remain locked.

## Decision tree

1. R8-A1 protocol/parity or label-stability FAIL: stop R8 model training.
2. R8-A1 PASS: run canonical five-seed R8-B OOF. **Completed.**
3. R8-B <4/5 seeds: record the local-hazard negative result and stop model
   escalation; do not add a world model. **Triggered: 0/5.**
4. R8-B ≥4/5 seeds: run shared-no-ID and per-VLA comparators, then build an
   offline selector using current recoverability LCB plus local hazard.
5. Only a selector meeting paired success gap ≥−5pp, harm ≤5%, and teacher
   savings ≥20% on both policy pairs can unlock the pre-registered
   residual/disagreement world-model ablation.

## Recommended next stage: R9 observability audit, not another model

R9 must separate three hypotheses before any new learning experiment:

1. **Insufficient temporal observation:** four past actions plus one boundary
   frame may omit contact, object velocity, gripper history, or progress change.
2. **Task-held-out semantic shift:** the Goal/Long split suggests the current
   features encode task-specific correlations rather than a general failure
   mechanism.
3. **Fallback-specific target:** recoverability is a property of the
   source/fallback/task triple, not source risk alone; a universal model may need
   a fallback behavior descriptor as well as a source descriptor.

The next authorized work is read-only/frozen-data diagnosis plus a new data
contract proposal:

- measure label prevalence and mutual-information proxies by task, suite,
  policy, horizon, perturbation, contact/progress bins;
- compare outcome-independent constant, policy+horizon, suite+horizon, and
  task-prior diagnostic baselines, clearly marked non-deployable;
- inventory which causal temporal signals were actually saved and which require
  a new collector;
- design a small paired collection at t={0,4,8,12,16} with short observation
  histories, object/contact deltas, source action chunks, and source/fallback
  descriptors;
- pre-register an information-support gate before training.

No additional VLA, world model, selector, validation, or test experiment should
start until that audit shows a task-general observable signal.

## R9 frozen-data observability audit result

The read-only R9 diagnostic completed on the 192 natural, currently-recoverable
transitions used for formal hazard evaluation:

- positives: 87 / 192 = 45.31%;
- five-seed mean model score AUROC: 0.577;
- policy+horizon smoothed task-held-out prior AUROC: 0.598;
- leaky within-task prevalence AUROC: 0.852 (diagnostic only);
- normalized label information explained by task identity: 39.71%;
- explained by policy+horizon: 8.19%;
- explained by perturbation dimension: 3.17%.

Prevalence mechanisms:

- t=0: 30.84% versus t=8: 63.53%;
- robot perturbation: 62.00%;
- camera perturbation: 43.55%;
- clean: 36.25%;
- Pi0.5: 47.24% versus Pi0Fast: 41.54%.

This rules out a simple "collect more of the same rows" response.  The existing
model does not beat a policy+horizon frequency prior, while task membership is
highly informative but unavailable on held-out tasks.  The next data contract
must expose task-general causal state change and meaningful task semantics.

Recommended R9-B pilot before any model:

- 24 tasks (6 per suite) with new initial states; folds are task-held-out within
  the pilot.  These are not independent never-seen tasks because all 48 current
  tasks have already participated in development, so they cannot be called
  validation or test;
- 2 source VLAs, balanced clean/camera/robot initial states;
- boundaries t={0,4,8,12,16};
- 3 fixed replicas per source/state/boundary fallback branch;
- save the last 4 observations, proprio deltas, gripper/contact proxy history,
  raw source action chunk, short progress deltas, a frozen semantic instruction
  embedding, and deployable source/fallback descriptors;
- no world-model output or future outcome enters an input row;
- run label support and simple linear/kNN representation probes before any MLP.

Pre-training information gate:

- at least 12 held-out tasks contribute both local hazard labels;
- every suite and perturbation dimension contributes at least 20 positive and
  20 negative transitions;
- task-held-out linear probe AUROC ≥0.65 on at least four causal feature groups;
- removal of task semantics may reduce performance, but temporal-state features
  alone must remain AUROC ≥0.60;
- at least two source policies must each exceed AUROC 0.60.

If this information gate fails, the universal learned risk/selector line should
be terminated for the present simulator/task family.  If it passes, a new model
protocol may be proposed; R8-B thresholds and negative results remain unchanged.

## R9-B execution status (2026-08-13)

The temporal collector is implemented as an opt-in extension of the validated
R6-B1 collector.  Legacy invocations remain byte/protocol compatible because
`--temporal-history` defaults to zero.  With `--temporal-history 4`, each saved
boundary adds only causal data captured before the source action:

- four-frame two-camera history `(4,2,3,96,96)`;
- four proprio states `(4,8)` and their first differences;
- four source action vectors `(4,7)`;
- boundaries `t={0,4,8,12,16}`;
- no OFT action, future frame, teacher cost, or outcome is placed in inputs.

The frozen R9-B manifest contains 24 metadata-selected tasks (six per suite),
one state per task, and excludes every state key present in the prior R6/R7/R8
dynamic dataset.  The 740-state replacement pool was used only as a metadata
source.  StatePool validation passed for all 24 keys.  Manifest SHA-256:
`33ec9adca504d4e2ab109a513e4919737c9d4cb4c934861fb7ba99ae4ca4db17`.

A two-state Spatial smoke with Pi0.5 and Pi0Fast completed successfully.  All
four NPZ files had the expected temporal shapes and no leakage was detected.
The full 24-task × 2-policy × 3-replica pilot is running in server tmux
`r9b_full`; no training or selector is started automatically.
