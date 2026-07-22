# Experiment progress records

This directory is tracked. It contains concise, reviewable records of completed
or blocked experiments; large logs, videos, checkpoints, and raw outputs belong
under ignored artifact directories.

## Recorded baselines

- [Frozen SmolVLA clean LIBERO, seed 0, `n_action_steps=10`](2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md):
  70.0% mean success over four suites and 2,000 episodes.
- [ForkableEnv W1 gate, 2026-07-17 11:36 CST](2026-07-17_forkable_env_w1_gate.md):
  4/4 fork integration tests passed (snapshot/restore deterministic).
- [W1 dry-run gates, 2026-07-17 11:46 CST](2026-07-17_w1_dry_run_gates.md):
  state-pool dry-run (26 states, idempotent resume) + collapse smoke dry-run (40 pending tasks).
- [SmolVLA LIBERO-Plus collapse smoke, seed 0, nas10](2026-07-17_smolvla_libero_plus_collapse_smoke_nas10.md):
  40 tasks × 1 ep → **2.5%** mean (camera 5% / robot 0%); qualitative only, not the paper curve.
- [NGC Step-1 real collect preflight + pilot](2026-07-17_ngc_step1_real_collect_preflight_pilot.md):
  SmolVLA+Plus adapter; preflight 2 ep + pilot 20 ep (1/20 success, **367** states); camera/robot only.
- [SmolVLA LIBERO-Plus collapse full, seed 0, nas10](2026-07-18_smolvla_libero_plus_collapse_full_nas10.md):
  3142 completed + 7 skipped → **0.38%** mean (camera **0%** / robot **0.78%**).
- [NGC Step-1 scale200 camera-heavy pool](2026-07-18_ngc_step1_scale200_camera_heavy.md):
  199 ep / 3666 states；pool fork gate 已过。
- [NGC W2 K=8 candidates pilot](2026-07-18_ngc_w2_candidates_pilot.md):
  2 states → `[8,10,7]` artifacts；mean endpoint L2 **1.69**（多样性通过；未测续完成功率）。
- [NGC W3 research-grade continuation pipeline](2026-07-19_ngc_w3_pipeline.md):
  旧 16-state SmolVLA **16×Set C / 0/768**；OFT spatial **8/32** / libero_10 **0/32**；`min_remaining_steps` 协议已落地。
- [NGC W3 continuation-temperature ablation](2026-07-19_ngc_w3_cont_ablation.md):
  OFT 阳性态上 SmolVLA 四温 **0/192** vs OFT **8/8** → 续完能力缺口，非温度/判据假阴性。
- [NGC W3 ADEQUATE-only pilot](2026-07-19_ngc_w3_pilot_adequate.md):
  ADEQUATE 16 态 SmolVLA 仍 **16×C / 0/768**（0 NARROW）；OFT 四 suite：spatial **7/32**，object/goal/10 **0/32**。

## Record requirements

Name new records `YYYY-MM-DD_<short-experiment-name>.md` and include:

1. status and research question;
2. checkpoint identity and frozen/trainable state;
3. resolved config path or embedded parameter summary;
4. seeds, task count, and episode count;
5. Python and dependency pins, upstream commits, Git SHA, and
   `env.lock.md` SHA-256;
6. aggregate and per-suite/task results;
7. artifact locations under `runs/`, `pool/`, or `results/`;
8. known deviations, failures, and follow-up work.

Never overwrite a historical result to reflect a new configuration. Add a new
record and link the superseded record instead.
