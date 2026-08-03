# RASE-UI Phase 0: semantic migration and opportunity gate

## Purpose

This phase changes RASE from a three-action policy-routing dataset into a
same-state intervention benchmark without invalidating prior W7-W10 evidence.
It is CPU-only until the final collection commands are deliberately launched.

## What is already reusable

- `ForkableEnv` and state-pool checksum/restore gates;
- episode/task identity and leakage audits;
- direct OFT as `SWITCH_POLICY(openvla_oft)`;
- legacy direct Smol as `REPLAN(smolvla)`;
- abstention and preregistered scalar utility baselines;
- W7/W8/W9C/W10 negative and positive diagnostic evidence.

## What is not yet reusable as claimed

- legacy `continue_smol` is not strict CONTINUE because `policy.reset()` is
  called after restore;
- snapshots do not carry an active action suffix or public history reference;
- any-of-K candidate portfolios are not deployable operators;
- `0.02/0.10/0.0` costs are abstract utility constants, not physical latency;
- current data contain no executable LOCAL_CORRECT or physical REWIND arms.

## 0. Code-only validation

```bash
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python
$PY -m pytest -q tests/test_intervention_schema.py tests/test_intervention_dataset.py
$PY -m ruff check rase/interventions scripts/migrate_legacy_interventions.py \
  scripts/audit_intervention_opportunity.py tests/test_intervention_schema.py \
  tests/test_intervention_dataset.py
```

These commands run unit tests and lint only. They do not launch VLA rollouts.

## 1. Migrate the frozen W9C baseline conservatively

```bash
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python
$PY scripts/migrate_legacy_interventions.py \
  --input runs/ngc_w9c_selector_dataset.jsonl \
  --output-dir runs/rase_ui_legacy_w9c
```

The default maps `continue_smol -> replan_smol`. Do not pass
`--direct-smol-family continue` for W9C/W10.

## 2. Audit the legacy baseline

The legacy data have no strict CONTINUE, LOCAL_CORRECT, or REWIND. The command
below is expected to be diagnostic and may exit with code 2 (`not_ready`).

```bash
$PY scripts/audit_intervention_opportunity.py \
  --registry runs/rase_ui_legacy_w9c/operators.json \
  --snapshots runs/rase_ui_legacy_w9c/snapshots.jsonl \
  --outcomes runs/rase_ui_legacy_w9c/outcomes.jsonl \
  --output runs/rase_ui_legacy_w9c/opportunity_audit.json \
  --min-complete-snapshots 20 \
  --min-oracle-gap 0.05 \
  --min-winning-operators 3 \
  --min-tasks-per-winning-operator 2 \
  --allow-zero-harm
```

`--allow-zero-harm` lets the report compute the legacy headroom numbers, but the
gate still returns `not_ready` because strict CONTINUE is absent. Passing
`--allow-missing-continue` is permitted only for debugging the CLI; such a
report is never the new benchmark gate.

## 3. Required implementation order before new GPU experiments

1. Add decision-context v2 collection with mid-chunk snapshot time, active
   source suffix, recent public RGB/proprio/action history, and source-policy
   identity. Prove suffix replay parity.
2. Implement strict `CONTINUE` and same-policy `REPLAN` as separate arms.
3. Add one geometry-independent primitive correction profile, initially
   `retreat_realign_v1`, with explicit feasibility and termination.
4. Add physical rewind only after recording safe public milestones and a
   controller that actually traverses back; simulator teleport is restore
   machinery, not a deployable REWIND operator.
5. Run a 10-task, four-operator pilot before world-model or CIVR training.

### Implementation status (2026-08-01)

Decision-context v2, strict active-suffix `CONTINUE`, same-state `REPLAN`, and
direct `SWITCH_POLICY(OFT)` are executable. Strict suffix replay parity passed
on all eight smoke states. The three-arm smoke produced the following terminal
success matrix:

```text
pattern order: CONTINUE / REPLAN / SWITCH_OFT
111: 5 states
101: 1 state
000: 2 states
```

The success-only same-state oracle is `0.75`, equal to best fixed `CONTINUE`
and `SWITCH_OFT`; therefore the success oracle gap is `0.00`. This is a valid
pipeline smoke but fails the benchmark opportunity gate. Do not train a
selector or world model on this pool.

There is a secondary cost-routing signal. On the six oracle-supported states,
success-then-env-steps routing uses OFT on five states and CONTINUE on one,
reducing mean steps from `84.17` for best fixed CONTINUE to `74.50`. Treat this
as a hypothesis for the next opportunity screen, not as a paper result: the
sample contains only two tasks, and warm-server timings omit cold-start cost.

Canonical outputs:

```text
runs/rase_ui_phase0_switch_oft_spatial_parity8_v1
runs/rase_ui_phase0_switch_oft_object_parity8_v1
runs/rase_ui_phase0_matrix_parity8_v3
```

Recommended first executable set:

```text
CONTINUE(smol active suffix)
REPLAN(smol from current observation)
SWITCH_POLICY(OFT from public handoff)
ABSTAIN
```

LOCAL_CORRECT is the fifth arm once its feasibility contract passes. REWIND is
the sixth and should not be rushed.

## 4. New pilot launch template (do not run until profiles are enabled)

The branch runner to be added in the next implementation phase must consume a
frozen snapshot manifest and registry, never infer arms from result files:

```bash
python scripts/rollout_intervention_matrix.py \
  --config configs/interventions_phase0.json \
  --snapshot-manifest runs/rase_ui_pilot/snapshots.jsonl \
  --output-dir runs/rase_ui_pilot/outcomes \
  --continuation-seeds 3 \
  --fresh-run
```

Do not create this result by relabeling candidate-prefix matrices. Until the
runner and decision-context v2 collector land, this command is a frozen target
interface, not an executable experiment command.

## 5. Go/No-Go interpretation

- `not_ready` due to missing arms/repeats: finish data collection; do not train.
- `not_ready` due to low oracle gap or fewer than three complementary winners:
  contract the paper or revise operator profiles under a new preregistration.
- `ready_for_method`: train a history-only operator-value baseline first.
- only after history-only clears matched-random and best-fixed with positive CI
  should a short-horizon operator-conditioned world model be implemented.
