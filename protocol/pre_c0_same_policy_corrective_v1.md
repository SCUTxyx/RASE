# PRE-C0 same-policy corrective protocol v1

Status: frozen before PRE-C0 collection or corrective outcomes.

## Scientific boundary

- PRE-A3 finite-handback method gate remains NOGO.
- PRE-A3 hidden test24 remains sealed.
- PRE-C0 uses only PRE-A3 train tasks and is a development mechanism audit.
- Runtime actions are generated only by the frozen SmolVLA checkpoint.
- OFT is not a runtime arm. It may later provide privileged teacher labels.
- Candidate critic and world model remain closed during the pilot.

This protocol is independent from the older frozen
`runs/rase_pre_a3_s_opportunity_spec_v1.json`. That spec used five candidate
families and an 8pp gate. PRE-C0 asks the newer early-correction question and
must not rewrite or merge the PRE-A3-S gate.

## Cohort

- 24 PRE-A3 train logical tasks: six per LIBERO suite.
- One outcome-independent trajectory per task.
- Eight clean:L0, eight camera:L1, eight robot:L1 episodes.
- Deviation stages: T0 last stable, T1 first deviation, T2 sustained
  deviation, T3 failure in progress, T4 terminal.
- Main paired pilot: T1 and T3 from every episode (48 states).
- T0 is retained as a harm control; T2/T4 are diagnostic.

## Natural arms

1. Current queued SmolVLA suffix.
2. Eight strict same-profile resamples: exact generation context, only flow
   noise seed changes.
3. Four fresh replans: discard queued suffix and condition on the latest
   observation.
4. Receding-horizon SmolVLA with execution horizons 1, 2, and 4.

Every arm must record snapshot/checkpoint/context/action hashes, seed,
generation and execution horizons, terminal outcome, harm, progress, GPU time,
and wall time.

## Natural Gate A

The gate opens only if all conditions hold:

- nested natural oracle minus current is at least 5 percentage points;
- at least two suites have rescues;
- at least three task-disjoint tasks have rescues;
- clean/T0 harmful replacement rate is at most 5%;
- the gain direction is non-negative on both early T1 and late T3 states.

Passing makes a no-world-model candidate critic eligible; it does not open
PRE-B or modify the PRE-A3 method gate.

## Conditional privileged guidance

If Gate A fails, use simulator progress/grasp/contact/collision evidence to
measure a matched-compute privileged selection/guidance upper bound.

Gate B requires at least 8pp gain over the natural oracle, positive gain in at
least two suites, stable T1 gain, and harm at most 5%. If privileged gain is
below 5pp, freeze frozen-policy same-policy recovery as NOGO and move to a
recovery adapter or offline distillation.

## Stop rules

- Do not inspect or run PRE-A3 val/test for PRE-C0 model choices.
- Do not tune thresholds, K, arms, or cohort after pilot outcomes.
- If deviation-stage QC reliability is below 80%, stop before corrective
  rollout and repair only the stage definition.
