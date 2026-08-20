# R5-A16 probability-label protocol lock

Date: 2026-08-09

Final result: see
`progress/2026-08-09_r5_a16_probability_and_oof_results.md`.  The protocol gate
passed, but the model-free opportunity gate and all five OOF controller seeds
failed; downstream expansion remains closed.

## Execution correction

The first `boundary_probability_pilot16_v1` launch exited before any rollout.
With two workers, the collector validated the complete suite request against
each individual shard, so states assigned to the sibling shard were falsely
reported as missing.  No labels or model results were produced.  The fix moves
manifest coverage validation before shard slicing; the corrected, versioned
run is `boundary_probability_pilot16_v2`.  The manifest, states, boundaries,
repeat seeds, and gates are unchanged.
Status: **KICKOFF; TEST SEALED; MODEL TRAINING CONDITIONAL ON A16 QC**

## Purpose

Scale the successful four-suite K=5 smoke from four to sixteen fresh development states and
measure whether handback outcomes are deterministic, stochastic, duration-monotonic, and
consistent across suites. This stage produces labels and protocol evidence; it is not formal
validation and carries no deployment claim.

## Frozen cohort

Manifest: `runs/pre_c0_r5/probability_pilot16_manifest_v1.json`

- 16 states not used by the K=5 smoke;
- four states per suite;
- eight true tasks, two states per task;
- 12 historically finite-safe and four historically persistent-only states;
- current val cohort is reclassified as development/calibration because it has influenced
  method design;
- original test split remains inaccessible.

## Frozen collection protocol

- boundaries: h={0,16,64,128};
- five Student continuations from each exact saved boundary;
- 64 maximum boundary records and 320 maximum continuation rollouts;
- all boundaries/repeats from one state remain in one task group;
- record policy seeds, outcomes, continuation costs, stop reasons, Wilson intervals,
  checkpoint hashes, projection hash and persistent replay parity;
- two collection workers are allowed for throughput, but no cross-worker result is treated as
  an independent task replicate.

## A16 protocol gate

Advance to probability-model training only if:

1. persistent replay parity is 100%;
2. repeat-field completeness is 100%;
3. duplicate repeat-seed rows are zero;
4. all four suites and eight tasks are represented;
5. every selected state has at least one recorded boundary;
6. missing late boundaries are explained by terminal trajectories, never silently imputed;
7. same-snapshot and prefix-trajectory uncertainty remain distinguishable in the schema.

Report, but do not use as an A16 protocol failure:

- fraction of nondegenerate boundaries;
- label entropy by suite and boundary;
- handback probability curves;
- duration non-monotonicity;
- descriptive best fixed and probability oracle;
- successful/failed continuation cost distributions.

## Conditional model stage

If the protocol gate passes, train a small shared state/action encoder with:

- Beta-binomial handback head;
- persistent-success head;
- source failure-risk head;
- remaining-OFT-cost quantiles;
- task-bootstrap ensemble;
- lower-confidence-bound handback rule;
- two-consecutive-boundary dwell and switching hysteresis.

Evaluation is state-level nested task-held-out OOF over five seeds. Row AUC is diagnostic only.

## Frozen OOF gates

- success gap versus persistent OFT >= -5pp;
- conditional false handback <=5%;
- executed OFT-step savings >=20%;
- at least four of five seeds pass all three constraints;
- task-cluster intervals must be reported and may not contradict the constraints.

If the gate fails, do not add a second VLA or search new world-model features. Expand
independent development/calibration states first. If it passes, proceed to shared-vs-per-VLA,
zero-shot and leave-one-VLA-out experiments. World-model experiments are restricted to
preregistered multi-step residual/disagreement features and remain in the main model only if
they improve the state-level Pareto.
