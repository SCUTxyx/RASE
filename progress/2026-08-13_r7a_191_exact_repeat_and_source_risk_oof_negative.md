# R7-A amended reproducibility gate and source-risk verdict

Date: 2026-08-13

## Executive verdict

The one-state exact-repeat failure was resolved without weakening or rewriting
the formal protocol.  The original 192-state exact-repeat audit remains
`FAIL`.  A diagnostic third repeat confirms stable seed, t=0 features, final
failure outcome, 510-step horizon and stop reason, but late closed-loop action
divergence.  That state is frozen in a reproducibility exclusion manifest and
replaced in the 16-state audit by the next same-suite/same-outcome state under
the original hash ordering.

The exclusion-bound 191-state cohort passes label support and the amended
exact-repeat audit 16/16.  Its five-seed task-held-out source-risk OOF is now
complete and fails 0/5 seeds.  The frozen decision is
`STOP_SOURCE_RISK_ESCALATION`.  Pi0.5/SmolVLA collection, OFT labels, selector,
world-model features, validation and test therefore remain locked.

## Reproducibility adjudication

Formal mismatch:

- state: `sp1_5b2f2d114882fcce15f2a4be884ad084`;
- suite: Long;
- source outcome: failure in all three runs;
- horizon: 510 source steps in all three runs;
- rollout seed, t=0 images, proprioception, first action and causal action
  summary: identical in all three runs.

Pairwise full-trace result:

| pair | first divergence | max absolute action difference |
|---|---:|---:|
| canonical vs repeat 1 | 130 | 0.449104 |
| canonical vs repeat 2 | 120 | 0.489076 |
| repeat 1 vs repeat 2 | 120 | 0.499686 |

This is late closed-loop nondeterminism, not corruption of the deployable t=0
risk input.  It does not justify changing the full-trace rule after observing
the result.  The immutable original audit remains at
`runs/pre_c0_r7/r7a_pi0fast_source_labels_v1/exact_repeat_audit.json`.

Frozen handling:

- adjudication:
  `runs/pre_c0_r7/r7a_pi0fast_source_labels_v1/exact_repeat_adjudication_v1.json`;
- exclusion manifest:
  `runs/pre_c0_r7/r7a_pi0fast_source_labels_v1/reproducibility_exclusions_v1.json`;
- exclusion SHA256:
  `1da46e3996fa4239e12e9ba6f37fa44ab5e3840c547b25323c29430829f89e97`;
- deterministic replacement:
  `sp1_d15720f4d72bf2503482d6e75aa35781`, Long failure,
  task `libero_10_000326`.

No seed was changed, no failed audit was deleted, and no state was rerun until
an outcome matched.  The replacement rule was frozen as same suite, same
outcome class, lowest unused rank under the original selection salt.

## Amended 191-state gates

The exclusion-bound label-support audit is `PASS`:

- 191 states / 48 true tasks;
- 89 source successes / 102 source failures;
- 35 failure tasks / 20 mixed-outcome tasks;
- Spatial 25/23 success/failure;
- Object 22/26;
- Goal 25/23;
- Long 17/30;
- every fit and calibration partition contains both classes;
- every suite support gate passes.

The amended exact-repeat audit is `PASS`, 16/16 with zero errors.  The label
audit, exact-repeat audit and dataset report all bind the same exclusion hash.
The canonical dataset is
`runs/pre_c0_r7/r7a_pi0fast_source_labels_v1/r7a_source_risk_dataset_191.npz`,
SHA256
`538347f406017d68c5d3c119ae25bdc6da40944026c2ac583ddcc988a9f6bcb6`.
The excluded 192-state version is not a canonical result.

## Five-seed OOF result

Formal output:
`runs/pre_c0_r7/r7a_source_risk_oof_191_v1/stability.json`.

| seed | AUROC | AP | AP - prevalence | ECE | bootstrap AUROC lower 95% |
|---:|---:|---:|---:|---:|---:|
| 2026081207 | 0.650 | 0.694 | 0.160 | 0.152 | 0.521 |
| 2026081208 | 0.564 | 0.607 | 0.073 | 0.215 | 0.431 |
| 2026081209 | 0.675 | 0.714 | 0.180 | 0.160 | 0.571 |
| 2026081210 | 0.654 | 0.715 | 0.181 | 0.145 | 0.538 |
| 2026081211 | 0.613 | 0.707 | 0.173 | 0.173 | 0.487 |

Aggregate means are AUROC 0.631, AP 0.688, AP gain 0.153 and ECE 0.169.
No seed reaches AUROC 0.75, bootstrap lower AUROC 0.65, ECE 0.10 and all-suite
AUROC above 0.60 simultaneously.  Three seeds meet the weak near-signal row
criterion, but mean AUROC is below the preregistered 0.65 minimum; therefore
the one policy-native adapter attempt is not unlocked.

## Failure mechanism

The model has nonzero signal, but it is not universal or calibrated enough for
a safe controller:

- Object AUROC is consistently strongest: 0.628--0.757;
- Goal is 0.588--0.684;
- Spatial is 0.588--0.663;
- Long is 0.376--0.565 and fails the suite gate for every seed;
- calibration error is 0.145--0.215, above the 0.10 gate for every seed;
- task-bootstrap lower bounds remain 0.431--0.571.

A post-hoc five-seed probability average is reported only as sensitivity:
AUROC 0.660, AP 0.704, ECE 0.178, and Long AUROC 0.469.  It does not rescue the
result.  Weighted per-fold ranking is higher than globally pooled calibrated
ranking, showing that small fold-specific Platt sets create scale variation;
however every seed still fails calibration and the Long suite.  The formal gate
is not changed after seeing this diagnostic.  Reproducible diagnostic output:
`runs/pre_c0_r7/r7a_source_risk_oof_191_v1/diagnostics.json`.

The scientific conclusion is narrower than “risk prediction is impossible.”
The current claim that final episode failure is reliably inferable at t=0 from
two RGB views, proprioception, language hash and the first Pi0Fast action chunk
is rejected on task-held-out data, especially for long-horizon tasks.

## Locked next actions

Under the frozen R7 protocol, do not:

- run a policy-native adapter;
- collect Pi0.5 or SmolVLA as though Pi0Fast passed;
- collect OFT labels or train a selector;
- add world-model residual/disagreement to rescue this base result;
- open validation or test.

Any continuation must be a new preregistered hypothesis rather than another R7
model tweak.  The recommended next design decision is between:

1. close R7-A as an honest negative result and package the reproducibility and
   source-risk benchmark evidence; or
2. start a separate R8 feasibility stage for *online short-horizon hazard*, not
   t=0 final-episode failure.  Before training, R8 should measure repeated
   outcome stability and label entropy by suite, define an early intervention
   horizon compatible with the model-free rescue window, and require a
   representation-free oracle/support gate.  Multi-VLA and world-model features
   remain downstream of that new base-risk gate.

## Code and tests

Added or amended:

- `scripts/adjudicate_r7a_exact_repeat_failure.py`;
- `scripts/run_r7a_exact_repeat_adjudication.sh`;
- `scripts/run_r7a_amended_191_pipeline.sh`;
- `scripts/audit_r7a_oof_diagnostics.py`;
- exclusion-aware source audit, exact-repeat manifest/audit, dataset builder
  and trainer preflights;
- environment-selectable amended paths in the exact-repeat, dataset and OOF
  runners;
- frozen-replacement regression test.

Server regression result: 8 passed.  The amended pipeline is fail-closed: a
second unstable replacement would have stopped before dataset construction and
OOF.
