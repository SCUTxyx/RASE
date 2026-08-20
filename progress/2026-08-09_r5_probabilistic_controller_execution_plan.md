# R5 probabilistic controller execution plan

Date: 2026-08-09

## Decision

Proceed with the proposed ordering, with two safeguards.  A16 is a development
protocol/entropy pilot, not a model result.  A head is trainable and claimable
only when its outer-training partition contains target support: both persistent
outcomes for the persistent-success head and both success/failure continuation
trials for the source-risk head.  Unsupported heads remain present in the
architecture but contribute zero loss and are reported as degenerate.

The previous validation split has already influenced method design and is now
development/calibration data.  It cannot be called independent validation in
the paper.  A new state/task-disjoint validation cohort must be collected after
the controller and thresholds are frozen; test remains sealed until that gate.

## Stage 1: A16 probability protocol

- Frozen cohort: 16 states, 8 true tasks, 4 suites, 2 states/task.
- Frozen boundaries: `h={0,16,64,128}`.
- Five Student continuations from each exact boundary snapshot.
- Maximum budget: 64 boundaries and 320 continuation rollouts; terminal OFT
  trajectories may make late boundaries unreachable.
- Required protocol checks: manifest state/task/suite coverage, K=5 field and
  seed integrity, persistent replay parity, and no unexplained missing reachable
  boundary.
- Descriptive outputs: Bernoulli label entropy by suite and boundary, fraction
  of nondegenerate boundaries, state probability curves, downward transitions,
  non-monotonic state fraction, continuation-step distributions, best fixed
  boundary, and a probability oracle.  Repeats are not independent states.

The first v1 launch produced no rollout because a multi-worker validation bug
compared a complete suite request against each individual shard.  The corrected
v2 validates coverage before shard slicing.  The manifest and experiment knobs
were not changed.

## Stage 2: probabilistic lightweight controller

Architecture:

1. shared latent/proprio state encoder;
2. canonical Student and teacher action summaries;
3. short history GRU;
4. Beta-binomial handback head;
5. persistent-success Bernoulli head;
6. source-risk binomial head using K-repeat failures at h=0;
7. ordered remaining-OFT-cost quantiles (q10/q50/q90).

Training/evaluation protocol:

- four fixed task-held-out outer folds over the eight A16 tasks;
- inner task-held-out calibration split for each outer fold;
- three task-bootstrap ensemble members;
- five training seeds with one frozen task-fold seed;
- handback LCB combines Beta predictive variance and ensemble variance;
- handback requires two consecutive recorded boundaries above the calibrated
  threshold;
- thresholds maximize calibration savings subject to success gap >= -5pp and
  conditional expected false handback <=5%; otherwise they fail closed;
- report state-level metrics and 5,000-replicate task-cluster bootstrap
  intervals. Row/repeat-level AUC is diagnostic only.

Frozen seed gate: at least four of five seeds must simultaneously have success
gap >= -5pp, false handback <=5%, and executed-OFT savings >=20%.  A16 remains a
feasibility gate because 16 states are too few for a conference claim.

## Stage 3: second VLA

This stage remains locked until the seed gate opens.  Add only a thin action
adapter and VLA identity metadata; keep the canonical risk core unchanged.
Run, on task-disjoint cohorts:

- per-VLA models;
- one shared model with balanced VLA sampling;
- train-on-VLA-A, zero-shot VLA-B;
- train-on-VLA-B, zero-shot VLA-A;
- leave-one-VLA-out once at least three source VLAs exist;
- shared encoder with per-VLA calibration as the practical intermediate
  baseline.

Report within-VLA and cross-VLA Pareto curves, calibration shift, abstention,
latency, parameter count, and adapter-only overhead.  Do not call an OFT teacher
a second Student VLA.

## Stage 4: world-model evidence

The world model is an optional feature generator, not the deployed controller.
Only preregistered multi-step latent residual and ensemble disagreement are
tested, with horizons frozen before labels are examined.  Compare the exact
same task folds and controller sweep for:

- base lightweight model;
- residual only;
- disagreement only;
- residual plus disagreement.

Retain world-model features in the main method only if they produce a
state-level Pareto improvement: no worse success/false-handback constraints and
strictly better OFT savings on outer OOF, stable in at least four seeds.  AUC-only
or row-level improvement is insufficient.  Otherwise world-model evidence is a
negative ablation and exits the main model.

## Stage 5: confirmation

After architecture, thresholds, feature set, dwell, and adapters are frozen:

1. collect a new independent validation cohort with both persistent outcomes
   and source-risk trial outcomes represented;
2. require the three controller gates without retuning;
3. only then unseal test;
4. run at least 100 paired closed-loop episodes, task-clustered, across all four
   suites and report success, rescue, harm, OFT steps, intervention duration,
   latency, and abstention;
5. confirm on a second policy pair and at least two environment/policy seeds.

## Implemented artifacts

- `scripts/collect_r4_boundary_transitions_v3.py`: K-repeat collector and fixed
  pre-shard manifest validation.
- `scripts/summarize_r5_probabilistic_boundaries.py`: manifest-aware entropy,
  coverage and non-monotonicity audit.
- `rase/risk/probabilistic_handback_student.py`: lightweight multi-head model.
- `rase/risk/probabilistic_losses.py`: Beta-binomial, binomial and quantile
  losses.
- `scripts/train_r5_probabilistic_handback_oof.py`: nested task OOF,
  task-bootstrap ensemble, LCB/dwell controller and cluster intervals.
- `scripts/run_r5_probabilistic_oof_5seed.sh`: frozen five-seed launcher.
- `scripts/analyze_r5_probabilistic_oof_seeds.py`: cross-seed gate and decision
  stability audit.

No deployment checkpoint is exported from A16 OOF, and no test data is read.
