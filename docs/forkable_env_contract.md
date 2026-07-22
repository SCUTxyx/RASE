# ForkableEnv contract

`ForkableEnv` provides deterministic, in-process replay for the inspected stack:

- LIBERO-plus commit `4976dc30028e805ff8094b55501d532c48fec182`
- robosuite `1.4.0`
- a single-arm robot using `OperationalSpaceController` (OSC)
- robosuite `LinearInterpolator`, `DeltaBuffer`, and `RingBuffer`

It deliberately fails on uninspected wrappers, robots, controllers, interpolators, buffer layouts, missing fields, and malformed snapshot keys. Supporting another upstream version requires auditing its mutable state and supplying a new explicit `CompatibilityProfile`; broad `__dict__` copying is forbidden.

## Snapshot format

`EnvSnapshot.version` is currently `1`. `EnvSnapshot.save("state")` writes:

- `state.json`: format/version, task fingerprint, typed tuple/array references, and per-array dtype, shape, and SHA-256.
- `state.npz`: compressed, non-object NumPy arrays.

Loading always uses `numpy.load(..., allow_pickle=False)`. Object-dtype arrays and unsupported Python objects are rejected. The two sibling files should be moved and retained together. Each file is atomically replaced, but the pair is not a transactional filesystem unit; an interrupted overwrite can leave a mismatched pair, which validation rejects.

## Captured state

The version-1 payload captures:

1. flattened MuJoCo state;
2. robosuite `cur_time`, `timestep`, and `done`;
3. OSC goals, impedance gains, update flag, cached output, initial/null-space references, and action scaling caches;
4. position/orientation interpolator start, goal, and step;
5. robot torque and all robosuite recent-action/proprioception `DeltaBuffer` / `RingBuffer` contents;
6. observable sample timers, delayed/current values, sampled flags, and `_obs_cache`;
7. Python `random`, NumPy's process-global legacy RNG, plus explicitly discovered `np_random` / `_np_random` Generator or RandomState instances.

LIBERO-plus's `seed()` and image corruptions use NumPy's process-global RNG. Therefore deterministic forks require one rollout-owning environment per process. Interleaving another environment or unrelated NumPy random draws in the same process changes the live global stream; restoring a snapshot intentionally rewinds that stream for the whole process.

## Task identity and restore order

The task fingerprint hashes canonical task identifiers, BDDL contents, wrapper/task classes, and the MuJoCo model XML. `restore()` recomputes and compares it before changing simulation state, so a snapshot from another task/model is rejected with `TaskMismatchError`.

Restore order is fixed:

1. MuJoCo flattened state, then `sim.forward()`;
2. episode counters;
3. OSC/controller and interpolator state;
4. robot history buffers;
5. task-derived visual state, observable scheduling, and observation cache;
6. environment, Python, and NumPy RNG state last.

RNG restoration is last so any incidental work during restore cannot advance the replay stream.

## Out of scope

- Policy-side state is not environment state. The caller must separately reset/restore VLA observation history, recurrent state, action queues, and sampling RNG.
- Multi-arm robots, IK and joint controllers, custom interpolators, simulator wrappers other than the allowlist, and vector/subprocess environments are unsupported.
- External side effects, renderer driver internals, asynchronous workers, wall-clock state, and third-party global RNGs (for example Torch) are not captured.
- Snapshots are intended for the same audited software build and process architecture, not as a stable cross-version MuJoCo checkpoint format.
- Exact image replay still depends on a deterministic renderer/driver. The acceptance test compares integer images exactly, floating observations at `1e-12`, and flattened final state at `1e-9`.

## Commands

The lightweight serialization test does not import LIBERO or initialize a renderer:

```bash
conda run -n smolvla python -m pytest -q tests/test_snapshot_serialization.py
```

Inspect one configured environment (this initializes LIBERO and offscreen rendering):

```bash
PYTHONPATH=. conda run -n smolvla python scripts/inspect_env_state.py \
  --bddl /absolute/path/to/task.bddl --render-gpu-device-id 0
```

Run the bounded smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=. \
conda run -n smolvla python scripts/smoke_test.py \
  --bddl /absolute/path/to/task.bddl --render-gpu-device-id 0 --steps 5
```

Run the opt-in integration tests:

```bash
RASE_TEST_BDDL=/absolute/path/to/task_a.bddl \
RASE_TEST_OTHER_BDDL=/absolute/path/to/task_b.bddl \
RASE_TEST_GPU_ID=0 PYTHONPATH=. \
conda run -n smolvla python -m pytest -q \
  tests/test_fork_roundtrip.py tests/test_fork_noise_rng.py tests/test_wrong_task_restore.py
```

If those environment variables or a working LIBERO/render fixture are absent, integration tests report a clear skip. Do not treat skipped integration tests as fork acceptance.
