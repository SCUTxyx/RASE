# SmolVLA LIBERO-Plus collapse run

## Identity

- Date:
- Operator:
- Status: planned / running / completed / blocked
- Git SHA: (copy from `manifest.json`)
- env.lock SHA-256: (copy from `manifest.json`)
- Manifest path:

## Resolved run

- Profile: smoke / full
- Policy/checkpoint:
- Suites:
- Dimensions: camera / robot
- Levels: L1 / L2 / L3 / L4 / L5
- Episodes per task:
- Seeds:
- `num_steps` / `n_action_steps`:
- Device / GPU:
- LIBERO-Plus commit:
- LeRobot version:
- Backend hook:

## Commands

Dry-run:

```bash
python3 scripts/eval_libero_plus_collapse.py --profile smoke --dry-run
```

Execution/resume:

```bash
python3 scripts/eval_libero_plus_collapse.py --profile full
```

## Results

Record success rate and episode count for every dimension × level cell, split
by suite where relevant. Do not combine cells with missing or failed tasks.

- Camera L1-L5:
- Robot L1-L5:
- Clean baseline used for collapse delta:
- Aggregate collapse:
- Failed/retried task keys:

## Validation and caveats

- [ ] Dry-run selected the expected task count.
- [ ] Manifest provenance matches this run.
- [ ] All intended task records are `completed`.
- [ ] No result mixes `n_action_steps=1` and `n_action_steps=10`.
- [ ] Missing assets, backend changes, and excluded tasks are documented.
- Notes:
