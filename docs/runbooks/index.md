# Runbook index

## Environment bootstrap

1. Select exactly one definition from `envs/`.
2. Create and activate it from the repository root:

   ```bash
   conda env create -f envs/smolvla.yaml
   conda activate rase-smolvla
   ```

3. For SmolVLA/LIBERO-plus work, check out LIBERO-plus at `4976dc3`, install
   it editable, and verify `libero.__file__`; see `third_party/PINS.md`.
4. Compare the resulting environment with `env.lock.md`. The OFT load smoke
   is recorded but its archive-based sources still lack Git SHAs; RL remains
   a W10 bootstrap only.

## Frozen baseline evaluation

Use `configs/eval_base.yaml` as the scientific source of truth. Before a formal
run, verify that it still specifies LeRobot 0.5.1, a frozen SmolVLA checkpoint,
`num_steps=10`, `n_action_steps=10`, and seed 0. Do not mix results from the
`n_action_steps=1` diagnostic.

Run each suite serially when needed to avoid GPU memory contention. The
historical command and measured suite results are preserved in
`progress/2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md`.

## Provenance capture

Every run directory must contain:

- the fully resolved evaluation config;
- the repository Git SHA and dirty-state indicator;
- `python -V` and a resolved package inventory;
- the SHA-256 of `env.lock.md`;
- upstream source/checkpoint revisions;
- a seed and hardware summary.

Generated run directories remain ignored. Promote only compact summaries to
`progress/`.

## W1–W3 runbooks

- [`../forkable_env_contract.md`](../forkable_env_contract.md): snapshot
  boundary, deterministic restore contract, and the blocking 50-step test.
- [`libero_plus_collapse.md`](libero_plus_collapse.md): dry-run, smoke, and
  full camera/robot L1–L5 collapse evaluation.
- [`collect_state_pool.md`](collect_state_pool.md): NGC Step 1 quotas,
  snapshot retention, storage, and resume behavior.
- [`ngc_pilot.md`](ngc_pilot.md): K=8 candidates, cross-environment oracle,
  Wilson triage, scheduler resume, and the 20-rollout timing pilot.

## Safety constraints

- Keep SmolVLA, OFT, and RL in separate environments.
- Keep package-level imports lightweight; load LeRobot, LIBERO,
  OpenVLA-OFT, and PyTorch only inside the code paths that require them.
- Never edit installed upstream source to add project behavior. Use wrappers,
  and document any unavoidable patch under `third_party/`.
- Match `CUDA_VISIBLE_DEVICES` with `MUJOCO_EGL_DEVICE_ID` for EGL rendering.
