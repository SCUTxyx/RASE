# NGC Step 1 state-pool collection

## Smoke and dry-run

Run from the repository root:

```bash
python scripts/collect_state_pool.py --config configs/collect_smoke.yaml
```

The smoke config uses a deterministic CPU-only adapter. It does not initialize
LIBERO, a policy, or a GPU, but it does exercise quota planning, the two-action-
chunk cadence, outcome-dependent retention, atomic bundle publication,
checksums, and the manifest. Re-running the command is a resume test: existing
identical states are verified and reported as `states_idempotently_skipped`.

`--dry-run` can also override the adapter in another config:

```bash
python scripts/collect_state_pool.py --config configs/collect.yaml --dry-run
```

Be aware that this still writes synthetic bundles to the configured output
directory. It is a storage dry-run, not a no-write preview.

## Full collection adapter

Set `adapter` in a copy of `configs/collect.yaml` to `module:factory`. The
factory receives the resolved config and returns an object with:

```python
def run_episode(request, episode_id: str, cadence: int) -> EpisodeResult:
    ...
```

`request` is a deterministic `PerturbationRequest`. `EpisodeResult` must expose
`outcome` (`"success"` or `"failure"`), `task_id`, `instruction`, and
`snapshots`. Each snapshot must expose:

- `step`: action-chunk index, divisible by `cadence`;
- `sim_state`: NumPy-compatible flattened simulator state;
- `controller_state` and `rng_state`: JSON-compatible complete state;
- `observations`: mapping of lowercase camera name to PNG bytes;
- `proprio`: NumPy-compatible proprioception.

The adapter must run the complete episode before returning. This is required
because all failure snapshots are retained, while successful snapshots use a
deterministic SHA-256 20% sample. The adapter owns policy reset, action queues,
controller capture, environment reset, and task/suite integration. The
collector deliberately does not import policy or environment packages.

Before a production run, independently pass the fork round-trip check:
restoring one snapshot twice and applying the same 50-step action sequence must
produce pixel-identical observations and object-pose error below `1e-9`.

## Real camera/robot pilot

`configs/collect_pilot.json` uses the built-in
`rase.collect.lerobot_libero_plus_adapter:make_adapter`. It runs frozen
SmolVLA (`num_steps=10`, `n_action_steps=10`) in one in-process Plus
environment and snapshots at action-chunk indices 0, 2, 4, ... until success
or the suite horizon.

Set paths through environment variables:

```bash
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus
export RASE_POLICY_PATH=/path/to/ckpts/smolvla_libero
export RASE_TOKENIZER_PATH=/path/to/ckpts/SmolVLM2-500M-Instruct

python scripts/collect_state_pool.py --config configs/collect_preflight.json
# After inspecting the two real episodes:
python scripts/collect_state_pool.py --config configs/collect_pilot.json
```

The pilot intentionally samples only camera and robot perturbations. Upstream
`task_classification.json` has no camera+robot combination category; do not
silently relabel single-factor tasks as combinations. A paired-task synthesis
protocol must pass reset/fork tests before enabling the planned 20% combination
quota. Layout and light/background/noise are supported by the adapter but are
not enabled in the first pilot.

`collection.max_action_chunks` may bound a debugging run. Leave it `null` for
collection so outcome means the real task result rather than an artificial
early cutoff. `action_chunks_per_episode` remains the deterministic dry-run
length and documents the maximum LIBERO-10 chunk count.

## Protocol and quotas

- Snapshot cadence: every 2 action chunks, including chunk 0.
- Perturbations: camera 30%, robot 30%, camera+robot 20%, layout 10%, other 10%.
- Suites: Long 40%, Goal 25%, Spatial 20%, Object 15%.
- Levels: camera/robot/combination/layout L3-L5; other L4-L5.
- Retention: every failed-episode snapshot; deterministic 20% of successful
  snapshots.

For totals that are not multiples of 100, largest-remainder apportionment gives
integer counts that sum exactly to the requested episode count. Pairing and
levels are deterministic for the collection seed.

## Storage and recovery

States are addressed by a versioned key derived from canonical identity
metadata:

```text
<output>/<task_id>/<episode_id>/<step_id>/
  sim_state.npz
  obs_<camera>.png
  proprio.npy
  meta.json
  checksums.json
<output>/manifest.json
```

`meta.json` carries the versioned `sp1_<32 hex>` state key; the manifest maps
that key back to the human-auditable task/episode/step directory.
Every file is written and fsynced in a staging directory. The complete
directory is then atomically renamed into place. `checksums.json` contains
per-file SHA-256 values and a canonical bundle checksum. `manifest.json` is
updated under an advisory lock using write-fsync-replace.

On resume, a matching key is accepted only after all checksums and metadata are
verified and the newly generated bundle is byte-identical. A changed payload
for the same key raises an error instead of silently overwriting data. Delete
or quarantine a corrupted bundle only after preserving it for investigation;
the collector will not repair corruption automatically.

## Pool fork gate

After a real collection finishes, prove pool bundles restore into `ForkableEnv`
and that two restores of the same snapshot produce identical rollouts:

```bash
conda activate smolvla
cd /data/data2/yuxuan/RASE
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus

# Smoke (fast)
python -u scripts/verify_pool_fork_roundtrip.py \
  --pool pool/ngc_step1_scale200 --sample 2 --steps 5 --check-pixels

# Contract length
python -u scripts/verify_pool_fork_roundtrip.py \
  --pool pool/ngc_step1_scale200 --sample 2 --steps 50 --check-pixels
```

The gate binds the live env by `task_id` / Plus instruction, then runs the
double-restore check. Full `model.get_xml()` fingerprints are required when they
match; for some Plus `initstate≠0` robot variants the XML hash is not
bit-stable across processes, so the script relaxes only that check after the
task bind succeeds (override with `--strict-fingerprint` to fail instead).


Lightweight unit coverage (no GPU):

```bash
pytest -q tests/test_pool_snapshot_roundtrip.py
```

Opt-in integration (needs `RASE_POOL_ROOT` + EGL):

```bash
export RASE_POOL_ROOT=/data/data2/yuxuan/RASE/pool/ngc_step1_scale200
pytest -q tests/test_pool_fork_integration.py
```

After this gate passes, continue to W2 candidate generation in
[`ngc_pilot.md`](ngc_pilot.md) (`scripts/generate_pool_candidates.py`).

## Operational checks

1. Run `pytest tests/test_state_pool_schema.py tests/test_perturb_sampler.py tests/test_resume_idempotency.py tests/test_pool_snapshot_roundtrip.py`.
2. Run the smoke config twice and confirm the second run creates zero states.
3. Inspect manifest counts and quota summary before enabling the real adapter.
4. Confirm free disk capacity; PNG observations dominate state size.
5. Keep the config seed and episode count unchanged when resuming.
6. After a real pool exists, pass the pool fork gate above before NGC recovery.

The full config has no default real adapter and intentionally fails fast until
one is configured. It is sized for 4,000 episodes; actual retained-state count
depends on episode length and success rate, so monitor the manifest rather than
assuming a fixed state yield.
