# RASE

RASE is a research codebase for reliability-aware selection and fallback
execution around frozen vision-language-action policies. The repository keeps
project code and reproducibility metadata in Git while leaving checkpoints and
generated experiment artifacts local.

## Reproduced baseline

The locked W1 reference is the frozen
[`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
checkpoint evaluated with LeRobot 0.5.1:

- Python 3.12
- `policy.num_steps=10`
- `policy.n_action_steps=10`
- seed 0
- 50 episodes per task across the four clean LIBERO suites
- mean success rate: **70.0%**

The suite-level results and exact invocation are recorded in
[`progress/2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md`](progress/2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md).
This is the project baseline; the separate `n_action_steps=1` diagnostic must
not be reported as the same configuration.

Recent NGC / Plus tracks:
[`collapse full 0.38%`](progress/2026-07-18_smolvla_libero_plus_collapse_full_nas10.md),
[`scale200 camera-heavy pool`](progress/2026-07-18_ngc_step1_scale200_camera_heavy.md)
(199 ep / 3666 states),
[`W2 candidates pilot`](progress/2026-07-18_ngc_w2_candidates_pilot.md)
(2 states, K=8, mean endpoint L2≈1.69).

## Setup

Create only the environment needed for a task:

```bash
conda env create -f envs/smolvla.yaml
conda activate rase-smolvla
pip install -e /path/to/LIBERO-plus
pip install -e .
```

LIBERO-plus must be checked out at commit `4976dc3` before installation. See
[`env.lock.md`](env.lock.md) for machine observations and
[`third_party/PINS.md`](third_party/PINS.md) for upstream provenance. The
SmolVLA, OFT, and RL environments remain separate because their heavy
dependency trees can conflict.

## Repository conventions

- `configs/eval_base.yaml` is the canonical frozen-baseline evaluation config.
- `rase` must remain importable without LeRobot, LIBERO, OpenVLA-OFT, or
  PyTorch installed. Import heavy libraries inside the functions that use them.
- `ckpts/`, `runs/`, `pool/`, and `results/` are local, ignored artifacts.
- `progress/` is tracked and contains compact experiment records.
- Every run should snapshot its resolved config, Git SHA, and the SHA-256 of
  `env.lock.md` into its output directory.

Runbook links are collected in [`docs/runbooks/index.md`](docs/runbooks/index.md).

## W1–W3 implementation status

The repository now contains the code contracts for:

- `rase.envs`: versioned, pickle-free snapshots, strict `ForkableEnv`, and the
  LIBERO-Plus task catalog;
- `rase.eval`: resumable camera/robot collapse manifests with provenance;
- `rase.collect`: perturbation quotas, state-pool storage, K=8 candidates,
  Wilson triage, and a disk-backed rollout scheduler;
- `rase.oracle`: a versioned ZeroMQ client/server protocol with injectable
  model adapters.

The lightweight tests passing is not the W1 acceptance gate. Fork integration
tests are opt-in and remain skipped until real BDDL paths and EGL are supplied.
Do not start NGC collection until the 50-step integration test passes.

## Preflight commands

These commands validate code and metadata without launching a policy rollout:

```bash
conda activate smolvla
cd /data/data2/yuxuan/RASE
pip install -e '.[dev]'

export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
python -m rase.envs.task_catalog --check
pytest -q

# Expands the collapse task set and creates only a resumable manifest.
export RASE_COLLAPSE_OUTPUT=runs/collapse_dry_run
export RASE_ENV_LOCK=/data/data2/yuxuan/RASE/env.lock.md
python scripts/eval_libero_plus_collapse.py \
  --config configs/collapse_camera_robot.yaml --dry-run

# Exercises the real state-pool schema with deterministic synthetic data.
python scripts/collect_state_pool.py \
  --config configs/collect_smoke.yaml --dry-run
```

The dry-run collectors write small artifacts under ignored `runs/`; they do
not load SmolVLA, initialize CUDA, or execute a simulator episode.

## W1 ForkableEnv gate

Run this only after choosing two real BDDL tasks and configuring EGL:

```bash
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export RASE_TEST_BDDL=/absolute/path/to/task_a.bddl
export RASE_TEST_OTHER_BDDL=/absolute/path/to/task_b.bddl
export RASE_TEST_GPU_ID=1

pytest -q tests/test_fork_roundtrip.py \
  tests/test_fork_noise_rng.py tests/test_wrong_task_restore.py
```

Acceptance requires no skips: two restores of one snapshot followed by the
same 50 actions must produce identical integer images and a final flattened
state difference below `1e-9`. See
[`docs/forkable_env_contract.md`](docs/forkable_env_contract.md).

## Runbooks

- [LIBERO-Plus collapse curve](docs/runbooks/libero_plus_collapse.md)
- [NGC Step 1 state pool](docs/runbooks/collect_state_pool.md)
- [W2–W3 candidate/oracle/adaptive pilot](docs/runbooks/ngc_pilot.md)

Each runbook separates metadata checks, smoke runs, and full runs. Full
commands are documentation only at this stage; inspect their resolved config,
GPU assignment, disk destination, and adapter hook before execution.