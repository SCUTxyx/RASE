# RASE Unified Intervention Phase 0C kickoff

Date: 2026-08-01 13:45 CST  
Status: **IN PROGRESS — calibration before selector/world-model training**

## Starting evidence

Phase 0B completed a 48-state, 12-task, fully paired matrix for strict CONTINUE,
REPLAN, and SWITCH_OFT. Success was 9/48, 13/48, and 30/48 respectively. The
success patterns were `000=18`, `001=17`, `011=4`, `111=9`; therefore OFT
covered every Smol success, the same-state success oracle equalled always-OFT
at 30/48, and the episode-cluster bootstrap gap was exactly `[0, 0]`.

The success-only opportunity gate is `NOT_READY`. No selector or world model
may be trained from this screen as evidence of method readiness.

An explicitly post-hoc cost sensitivity diagnostic showed that resource-aware
utility can create three distinct regions. With success reward 1, CONTINUE cost
0, REPLAN cost 0.01, and OFT cost 0.10, the diagnostic oracle gap was 0.06375
and unique utility winners covered 11/3/9 tasks. The value 0.10 is not frozen or
physically calibrated and cannot be used as a confirmatory result.

## Phase 0C research question

Can a preregistered physical resource cost and a near-boundary state cohort
produce a stable, task-supported utility opportunity for all three intervention
families, without hiding raw-success dominance?

## Frozen execution order

1. Audit available perturbation controls and add a calibration-only protocol.
2. Instrument or summarize inference-only deployment costs separately from
   rollout duration; do not infer model cost from early termination.
3. Run a small, balanced difficulty calibration in tmux session 0. It must
   include clean and weaker-than-current failure regimes and remain separate
   from any later confirmation cohort.
4. Analyze source-policy recoverability by suite, dimension, level, and time.
5. Only if the calibration exposes non-degenerate boundary strata, freeze a
   task/episode-disjoint 96-state Phase 0C screen and run the three arms.
6. Require complete coverage, at least three task-supported unique utility
   winners, utility oracle gap >= 0.05, and episode-cluster bootstrap evidence
   before opening selector training.

## Stop rules

- If weak/clean states still yield a fixed-operator oracle, do not scale the
  matrix and do not train a selector.
- If a positive gap exists only under an arbitrary post-hoc cost, first revise
  and preregister the cost protocol; do not call the method ready.
- Keep success-only and utility-aware conclusions separate in every report.

## Planned artifacts

- Calibration config and resumable runner under `configs/` and `scripts/`.
- Calibration outputs under `runs/rase_ui_phase0c_*`.
- A new result record; this kickoff record will not be overwritten.
