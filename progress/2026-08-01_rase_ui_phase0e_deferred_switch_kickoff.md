# RASE-UI Phase 0E deferred-switch semantics kickoff

Date: 2026-08-01 17:05 CST  
Status: **IN PROGRESS — operator-semantics calibration; no model training**

## Verified starting point

Phase 0D progress records and official artifacts were re-audited before this
stage. The recorded SHA-256 values match the live files. The official timing
analysis remains complete over 16 states/episodes/tasks, with policy time per
environment step of 29.394 ms for strict CONTINUE, 30.089 ms for REPLAN, and
16.483 ms for SWITCH_OFT. No latency evidence supports the exploratory
`switch_oft=0.10` penalty. The selector, outcome-model, and world-model gates
remain closed.

## Code audit finding

The existing `rollout_oft_prefix_ablation.py --arms full` loads prefixes from
legacy `candidates_dir/<state_key>.npz` artifacts and constructs direct, zero,
and candidate-prefix arms. It does **not** load the strict-CONTINUE active suffix
from decision-context v2. Reusing that mode as “deferred switch” would be a
semantic error.

Phase 0E will add a separate `decision-suffix` mode. The legacy full candidate
ablation and direct-only mode must remain unchanged.

## Operator contract

For every frozen state, run two same-snapshot arms:

1. `direct_oft`: discard the source action suffix and switch to OFT immediately.
2. `decision_suffix_oft`: replay exactly
   `controller_state.decision_context.active_action_suffix` in environment action
   space, then bind OFT to the resulting live public observation and continue.

The deferred arm must record:

- `prefix_source=decision_context.active_action_suffix`;
- prefix length and SHA-256 over canonical float32 bytes;
- actual candidate/prefix steps executed;
- OFT predict calls and RPC/inference time;
- success, stop reason, environment steps, and elapsed rollout time.

The runner must reject missing/invalid suffixes, non-finite or non-`[T,7]`
actions, checksum drift, suite mismatch, incomplete coverage, or prefix-step
parity failure.

## Calibration experiment

Use the already frozen Phase 0D 16-key, one-state-per-episode step-2 cohort.
Run direct and decision-suffix OFT for all 16 states, suite-serially. This is an
operator-semantics calibration and a parity check; it reuses Phase 0C snapshots
and therefore is not independent confirmation.

Primary outputs:

- direct/deferred success pairing: direct-only, deferred-only, both, neither;
- same-state oracle gain over the better fixed timing profile;
- prefix lengths/hashes and exact parity coverage;
- latency/RPC consequences of deferring the switch;
- suite and perturbation-cell diagnostics.

## Stop rules

- Do not join legacy candidate actions into this operator.
- Do not call the calibration held-out or paper-confirmatory.
- If deferred-only is zero and direct dominates, retire deferred switch from the
  next screen rather than increasing sample size.
- If both directions appear, require task/episode-disjoint 24–32-state screening
  before any 96-state confirmation.
- Do not train a selector, intervention outcome model, or world model from this
  calibration.

## Planned artifacts

- `runs/rase_ui_phase0e_deferred16_<suite>_v1/`
- `runs/rase_ui_phase0e_deferred16_analysis_v1.json`
- `runs/rase_ui_phase0e_deferred16_v1.log`

Final results will be added as a new progress record without overwriting this
kickoff.
