# RASE-UI intervention dataset schema v1

RASE-UI separates three objects that the legacy escalation dataset combined:

1. `rase-intervention-registry/v1`: fixed executable operator profiles;
2. `rase-intervention-snapshot/v1`: public decision identity plus a privileged
   restore-state reference that is never a model input;
3. `rase-intervention-outcome/v1`: one real continuation for one
   `(snapshot, operator, continuation_seed)` arm.

## Operator identity

An operator is the structured tuple `(family, executor, recovery_target,
parameters, requirements)`, identified by a frozen `operator_id`. The core
families are:

```text
CONTINUE / REPLAN / LOCAL_CORRECT / REWIND / SWITCH_POLICY / ABSTAIN
```

Changing executor, target, horizon, controller profile, handoff semantics, or
budget creates a new operator profile. Benchmark tables must not pool such
implementations under a family label without also reporting profile-level
results.

## Strict CONTINUE versus REPLAN

`CONTINUE` requires the remaining source-policy action suffix that was active at
the snapshot. `REPLAN` discards that suffix and calls the source policy from the
current public observation/history.

The legacy direct-Smol rollout restores a state and calls `policy.reset()`.
Therefore migration maps that arm to `replan_smol` by default. Calling it
`CONTINUE` is forbidden unless a new snapshot protocol records and replays the
active suffix. Existing W9C/W10 results remain valid for their original routing
question, but they do not by themselves provide the CONTINUE-versus-REPLAN
counterfactual required by RASE-UI.

## Snapshot contract

Required identity fields include `snapshot_id`, `state_key`, `task_id`,
`episode_id`, `step`, and `source_policy`. `restore_state_ref` is privileged and
is consumed only by the branch runner. `public_history_ref` and
`active_action_suffix_ref` identify deployable information.

The model must never consume simulator state, future outcomes, perturbation
labels, oracle progress, split membership, or other operators' results.

## Outcome contract

Each outcome stores:

- explicit feasibility and reason codes;
- whether a real rollout was observed;
- terminal success, operator completion, and stop reason;
- continuation seed and exact outcome semantics;
- measured cost vector: compute, latency, steps, energy, progress loss, human
  time, and safety penalty;
- a separately frozen `utility_cost` and `cost_source`.

Separating physical measurements from scalar utility prevents the legacy
constants (`0.02/0.10/0.0`) from being misreported as latency or compute. Proxy
and any-of-K portfolio outcomes remain diagnostic only and fail the default
opportunity gate.

## Same-state opportunity gate

Before training CIVR or any other selector, the complete-case pilot must show:

- enough snapshots with every enabled arm and required repeats;
- one enabled strict CONTINUE profile with active-suffix provenance;
- same-state oracle utility at least 0.05 above best fixed operator;
- at least three operators winning on at least two tasks each;
- non-zero intervention harm relative to strict CONTINUE;
- non-zero costly futility;
- no proxy outcomes.

Failure of this gate means the benchmark/operator design must be revised or the
claim contracted. It does not authorize a larger model, MLP, RL, or a world
model. World-model work begins only after the benchmark and history-only method
gates pass.
