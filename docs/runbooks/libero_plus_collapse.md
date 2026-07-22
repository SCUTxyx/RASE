# LIBERO-Plus camera/robot collapse evaluation

The entry point filters the upstream `task_classification.json` by suite,
camera/robot dimension, and L1-L5 difficulty. It creates an atomic, resumable
`manifest.json` before any policy is loaded.

## Portable path setup

Machine paths are never stored in the checked-in config. Pass them as flags or
set:

```bash
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus
export RASE_COLLAPSE_OUTPUT=/path/to/runs/collapse_smoke
export RASE_ENV_LOCK=/path/to/RASE/env.lock.md
export RASE_POLICY_PATH=/path/to/smolvla_libero
```

`LIBERO_PLUS_TASK_CATALOG=/path/to/task_classification.json` can replace
`LIBERO_PLUS_ROOT` for dry-run catalog selection, but real evaluation still
needs `LIBERO_PLUS_ROOT` (or an editable Plus install) so BDDL/init assets
resolve.

On first real evaluation the backend writes a Plus-pointing
`config.yaml` under `LIBERO_CONFIG_PATH` (default `~/.libero_plus_rase`).
This does **not** overwrite stock `~/.libero/config.yaml` used by clean LIBERO
baselines.

## Validate without loading a policy

```bash
python3 scripts/eval_libero_plus_collapse.py \
  --profile smoke --levels L1-L5 --dry-run
```

Smoke selects one deterministic task per available
`suite × dimension × level` cell and one episode per task. `--dry-run` parses
the real upstream catalog, resolves config, records provenance, and creates the
same pending manifest used by execution. It does not import LeRobot or inspect
the policy path.

For a narrower check:

```bash
python3 scripts/eval_libero_plus_collapse.py \
  --dry-run --dimensions camera --levels L3-L5 \
  --suites libero_spatial,libero_goal
```

## Execution backend

Default `--backend lerobot` loads
`rase.backends.lerobot_libero_plus:evaluate`. That adapter:

1. Points LIBERO path lookups at the Plus checkout
2. Patches LeRobot init-state loading to use Plus `Benchmark.get_task_init_states`
3. Maps catalog `task_id` (1-based) → suite index (`task_id - 1`) and asserts name match
4. Loads frozen SmolVLA once (nas10) and evaluates the requested episodes

Optional override:

```bash
--backend-hook rase.backends.lerobot_libero_plus:evaluate
```

Hook signature:

```python
def evaluate(task, task_output_dir, resolved_config) -> dict:
    ...
```

A hook exception is persisted on that task before the command exits (or continues
when `--continue-on-error` is set).

## Smoke execution (recommended first)

Use a **new** output directory (do not reuse `collapse_dry_run` if you want a
fresh provenance record after adding the policy path). Narrow filters first:

```bash
conda activate smolvla
cd /path/to/RASE

export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus
export RASE_POLICY_PATH=/path/to/ckpts/smolvla_libero
export RASE_COLLAPSE_OUTPUT=/path/to/runs/collapse_smoke_camera_l3_spatial
export RASE_ENV_LOCK=/path/to/RASE/env.lock.md
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python scripts/eval_libero_plus_collapse.py \
  --config configs/collapse_camera_robot.yaml \
  --profile smoke \
  --dimensions camera \
  --levels L3 \
  --suites libero_spatial
```

Then the full smoke grid (40 tasks × 1 episode when all cells exist):

```bash
export RASE_COLLAPSE_OUTPUT=/path/to/runs/collapse_smoke_nas10

python scripts/eval_libero_plus_collapse.py \
  --config configs/collapse_camera_robot.yaml \
  --profile smoke
```

## Full profile and resume

```bash
python3 scripts/eval_libero_plus_collapse.py --profile full
```

Full selects every matching classified task and follows the upstream
LIBERO-Plus protocol of one trial per task. A task is skipped only after its
manifest status is `completed`. Interrupted
`running` and `failed` records are retried; attempts and errors are retained.
The selected task set and order must match when resuming. Once execution has
started, changed config, Git SHA, or env-lock hash is rejected on resume; an
untouched dry-run manifest may be promoted to execution.

Each manifest contains the repository Git SHA, SHA-256 of the repository
`env.lock.md` (or the `--env-lock`/`RASE_ENV_LOCK` override), and the fully
resolved configuration. Writes use rename-based atomic replacement. Per-task
backend artifacts live below `tasks/` (`metrics.json`, optional videos).
