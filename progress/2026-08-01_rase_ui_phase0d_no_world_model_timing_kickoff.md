# RASE-UI Phase 0D: no-world-model freeze and timing calibration kickoff

Date: 2026-08-01 16:10 CST  
Status: **IN PROGRESS — physical timing calibration; no model training**

## Decision freeze

The project will not train a generative dynamics world model at the current
stage. Phase 0C has only 64 intervention states from 16 source episodes and a
success oracle gain of 1/64 over always-OFT. This is insufficient both for
world-model learning and for demonstrating that learned intervention selection
has useful headroom.

If the opportunity gate later passes, the first learned component will be a
small supervised **intervention outcome/value model**, not a pixel-generating
world model. It will predict per-operator success, latency, progress, harm, and
uncertainty from public observations and history. A dynamics world model is
out of scope unless later experiments require multi-step imagined intervention
planning and provide a substantially larger dataset.

The selector/world-model gate remains closed until a held-out benchmark shows
stable, task-supported operator complementarity.

## Starting evidence

Phase 0C factorial16 produced 64 fully paired states:

- CONTINUE 19/64;
- REPLAN 14/64;
- SWITCH_OFT 38/64;
- same-state success oracle 39/64;
- episode-cluster oracle-gap CI `[0, 0.046875]`;
- REPLAN-only 0 and OFT-only 20.

OFT inference instrumentation is complete: 1,651 predict RPCs, 213.631 seconds,
129.395 ms/call, with 64/64 state coverage. The completed Phase 0C Smol run
predates the Smol action-selection timer, so a small dedicated timing rerun is
required rather than fabricating or inferring Smol compute from rollout time.

## Phase 0D-A experiment

Run a 16-state timing calibration using exactly one step-2 state from each
Phase 0C source episode. This gives 16 episode-independent timing trials while
preserving the four suites and clean/camera/robot factorial cells. It is a
resource-calibration cohort, not a new opportunity or confirmation cohort.

The experiment will:

1. export and audit one state per source episode at frozen step 2;
2. rerun strict CONTINUE and REPLAN with the new Smol action-selection timer;
3. join the same state keys to existing OFT RPC metrics;
4. report total and per-call/action-selection wall time, coverage, and outcome
   counts without treating the two instrumentation scopes as identical;
5. keep full rollout duration separate from policy action-acquisition time.

## Stop rules

- Do not select utility weights from this run's success outcomes.
- Do not call lower rollout duration lower inference cost.
- Do not train a selector, outcome model, or world model after this timing run.
- If instrumentation coverage is incomplete, repair and rerun the timing
  cohort before defining any physical-cost protocol.
- Do not scale the unchanged CONTINUE/REPLAN/OFT success matrix to 96 states.

## Planned artifacts

- `runs/rase_ui_phase0d_timing16_keys.json`
- `runs/rase_ui_phase0d_timing16_smol_v1/`
- `runs/rase_ui_phase0d_timing16_analysis_v1.json`
- `runs/rase_ui_phase0d_timing16_v1.log`

This kickoff is a preregistration record. Final results will be written to a new
progress file and will not overwrite this document.
