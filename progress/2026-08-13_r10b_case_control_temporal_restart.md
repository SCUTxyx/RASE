# R10-B case-control temporal representation restart

Date: 2026-08-13

## Why R9-C stopped

R9-C correctly failed its frozen information-support gate.  The independent
R9-B temporal cohort produced only 12 one-step recoverability-loss hazards in
480 transitions (2.5%).  Object and Spatial had zero positives, camera
perturbations had zero positives, and two of five task-held-out folds were
single-class.  The all-causal probe (0.829 AUROC on three valid folds) did not
beat the cross-policy/horizon frequency signal (0.842).  R10 shared-risk
training, selector, world-model, validation and test therefore remained locked.

This does not reject the RASE core idea.  It rejects the sparse unconditional
target on that cohort.  A larger MLP or world-model feature cannot repair absent
label support.

## Frozen R10-B question

The next question is selector-aligned and conditional:

> Given that persistent fallback is certainly successful at t=8, can causal
> source history predict whether fallback recoverability will be lost by t=16?

Case/control labels are frozen from the completed K=2 replica aggregate:

- case: t8 fallback succeeds in all K=2 trials and t16 fails in all K=2 trials;
- control: fallback succeeds in all K=2 trials at both t8 and t16;
- ambiguous K=2 boundaries are ineligible.

The cohort is explicitly label-balanced development enrichment.  It is valid
for representation/information testing only, not prevalence, calibration,
validation, test, or closed-loop selector claims.  A later natural cohort is
required if the information gate passes.

## Frozen manifest

`runs/pre_c0_r10/r10b_case_control_manifest_v1.json`

- manifest SHA256:
  `9f1febbd3db17d9e6178bf13d7b4f36900c12d6117261f8923b25fef6b37eaca`;
- 66 state-policy-seed groups and 198 planned K=3 trajectories;
- 33 cases / 33 controls;
- 36 true tasks and all four suites;
- Pi0.5: 39 groups; Pi0Fast: 27 groups;
- per-class suite targets: Goal 7, Long 8, Object 12, Spatial 6;
- all five frozen outer folds contain both labels;
- selection is deterministic from the already-frozen R8 labels and a fixed
  hash salt; no model score or threshold was searched.

## Data contract

At boundaries `t={0,4,8,12,16}`, save only causal deployable inputs:

- eight-step two-camera image history;
- eight proprio states, first differences and second differences;
- eight source action vectors and action differences;
- current source action summary and instruction hash;
- source policy identity retained only for explicit ablations/calibration.

OFT actions, future frames, K=2 selection labels, task/suite identity, simulator
object state and future teacher cost are forbidden model inputs.

## Post-collection hard gate

Every selected group is recollected with three identical-seed replicas.  The
dataset may be built only if:

- all three t8 and t16 labels agree;
- the K=3 hazard label matches the frozen K=2 label;
- all t8 causal feature arrays have exact/allclose replica parity;
- all five outer folds retain both classes.

No unstable group is replaced after observing the new label.

## R10-C information gate

Before training an MLP, task-held-out low-capacity probes evaluate image,
temporal state, action history, semantics, temporal+action and all-causal
features.  Frozen requirements are:

- all five task-held-out folds valid;
- all-causal OOF AUROC >= 0.65;
- task-bootstrap all-causal AUROC lower 95% bound >= 0.58;
- temporal-state and temporal+action AUROC >= 0.60;
- each policy's all-causal AUROC >= 0.60;
- all-causal AUROC improves at least 0.05 over a cross-fitted policy-only
  prior (validation labels never estimate their own prior).

PASS unlocks a lightweight shared-risk OOF protocol only.  Selector,
world-model and natural calibration remain locked.  FAIL stops before another
learned model.

## Implementation and execution status

Implemented:

- `scripts/freeze_r10b_case_control_manifest.py`;
- `scripts/run_r10b_case_control_collect.sh`;
- `scripts/audit_r10b_case_control_repro.py`;
- `scripts/build_r10b_case_control_dataset.py`;
- `scripts/audit_r10c_case_control_information.py`;
- `tests/test_r10b_case_control_contract.py`.

Six R9/R10 contract tests pass.  A two-group, one-replica smoke completed with
all five boundaries and expected shapes: image history `(5,8,2,3,96,96)`,
proprio history `(5,8,8)`, action history `(5,8,7)`.

The full 198-trajectory collection is running in server tmux `r10b_full` and
writes to `runs/pre_c0_r10/r10b_case_control_collect_v1/`.  The runner stops
after collection; it does not automatically audit, build a dataset, train a
model, or launch a selector.

A separate gate-only watcher runs in tmux `r10b_post`.  After `COMPLETE`, it
runs the frozen reproducibility audit, conditionally builds the hash-bound
dataset, and runs the low-capacity information gate.  Any FAIL stops the chain;
even a PASS only writes `R10B_READY_FOR_R10D` and does not start model training.
