# R10-B reproducibility failure and OFT trace root-cause diagnostic

Date: 2026-08-13

## Canonical R10-B result

The frozen R10-B collection completed all 198 planned trajectories (66
state-policy-seed groups x three exact-repeat replicas).  The post-collection
reproducibility gate is formally **FAIL** and the pipeline correctly stopped
before dataset construction, R10-C information testing, or model training.

The failure is not caused by corrupt causal inputs:

- all 66 groups have exact/allclose parity for the t8 image, proprio, source
  action, action summary, and eight-step temporal histories;
- 57/66 groups have stable K=3 t8/t16 fallback outcomes;
- 9/66 groups have within-K3 outcome instability;
- only 32/66 groups match their frozen K=2 case/control label;
- the frozen 33 K=2 cases become 8 stable cases, 19 stable controls, and six
  unstable groups under K=3;
- the frozen 33 K=2 controls become 25 stable controls, five stable cases, and
  three unstable groups.

Thus the deterministic label `t8 always succeeds and t16 always fails` is not
a reproducible state property for this cohort.  Unstable groups were not
deleted or replaced, and the original R10-B status remains FAIL.

Canonical artifact:
`runs/pre_c0_r10/r10b_case_control_repro_audit_v1.json`.

## Post-failure probability diagnostic

Because repeated outcomes imply an empirical probability rather than a hard
state label, a strictly exploratory diagnostic retained all 66 groups and used
the new independent K=3 event count

`hazard = success_if_enter_t8 AND failure_if_enter_t16`.

This produced 42 hazard events in 198 trials, across 15 groups and 11 true
tasks.  The count distribution is 51 groups with 0/3, one with 1/3, one with
2/3, and 13 with 3/3.

Task-held-out low-capacity count regression reports:

- temporal-state event AUROC: 0.609, task-bootstrap 95% interval
  [0.429, 0.775];
- temporal-plus-action event AUROC: 0.591;
- all-causal event AUROC: 0.523, task-bootstrap lower bound 0.353;
- all-causal Pi0.5/Pi0Fast AUROC: 0.512/0.563;
- action-only AUROC: 0.543; image-sequence AUROC: 0.538.

Decision: `DO_NOT_ESCALATE_PROBABILISTIC_MODEL`.  The diagnostic is post-gate
and cannot unlock R10-D, a selector, world-model features, validation, or test.
It shows that replacing the hard label with a Beta-binomial head is not enough
under the currently recorded inputs.

Artifacts:

- `runs/pre_c0_r10/r10b_probabilistic_diagnostic_v1.npz`;
- `runs/pre_c0_r10/r10c_probabilistic_information_exploratory_v1.json`.

## First-action variance diagnosis

The saved OFT first action at both t8 and t16 is bitwise identical across all
three replicas for all 66 groups.  The maximum elementwise difference is 0.0,
including all nine outcome-unstable groups.  Therefore the label flips are not
explained by t8 causal feature drift or the first OFT inference call.

Artifact: `runs/pre_c0_r10/r10b_oft_first_action_variance_v1.json`.

## Running root-cause experiment

A diagnostic manifest was frozen before new collection:

- all nine outcome-unstable groups;
- nine deterministically hash-selected stable controls matched by suite and
  source policy;
- three exact-repeat replicas per group, 54 trajectories total;
- development/root-cause use only; forbidden for training or model selection.

The collector now optionally records only the SHA256 and shape of each full OFT
action trace.  This metadata is never exposed as a model feature.  Collection
is running in server tmux `r10b_trace` and will automatically run
`audit_r10b_trace_diagnostic.py` after `COMPLETE`.

Decision after completion:

- identical full OFT traces with outcome flips -> run fixed-action replay and
  audit simulator/termination nondeterminism;
- divergent later OFT traces with identical first actions -> audit chunk-query
  observations and closed-loop amplification;
- no reproduced flips -> keep R10-B failed and estimate how many replicas are
  needed for a stable probabilistic target before any new cohort.

Risk model, selector, world model, validation and test remain locked throughout
this diagnosis.
