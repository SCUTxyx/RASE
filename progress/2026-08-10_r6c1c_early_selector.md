# R6-C.1C: Early-Window Stratified Selector — Implementation Status

Date: 2026-08-10  
Status: implementation complete and smoke-validated on B1.2; full 5-seed OOF on
the merged R6-C.1B dataset queued behind the ongoing source-only screening and
OFT-labelled collection.

## Model architecture (`rase/risk/light_risk_student.py::CandidateArmStudent`)

New policy-conditioning inputs (R6-C.1C, red lines 5):

- **VLA identity embedding** (`policy_embedding`, `nn.Embedding(n_policies,16)`)
  for seen VLAs;
- **deployable behavior descriptor** (`descriptor_mlp`, 8-dim source-observable
  rollout statistics → policy embedding): source success rate, mean/median
  source steps, action norm, proprio norm, history norm, elapsed-progress mean,
  source-within-16 rate.  Computed **only** from source-observable stats, so it
  is deployable for a new VLA;
- **per-VLA FiLM calibration adapter** (`policy_film`) applied to each member's
  fused representation (`fused * (1+scale) + bias`);
- **advantage head**: `Δ = Q_enter_OFT − Q_continue_source`, trained with MSE on
  `arm_success[:,1] − arm_success[:,0]`.

Descriptor takes precedence over identity embedding (deployable path for new
VLAs); both are optional and activated per mode.

## Two-stage controller (`controller_early_window`, no emergency trigger)

Decision points are exactly `t ∈ {0, 8, 16}` present in each trajectory group:

- `t0`: high-risk (LCB < risk_thr) **and** worth-it (advantage > adv_thr) →
  switch to persistent OFT immediately (no dwell);
- `t8`/`t16`: final judgment (risky & worth → switch; else continue);
- after `t16` (or the last decision point ≤ 16) the source is **locked** to
  termination — no emergency trigger anywhere in the controller.

Baseline for success gap / savings is `ENTER_PERSISTENT_OFT@t0`, identical to
R6-C, so the comparison is apples-to-apples.

## Training / thresholds (red lines 3-4)

- 3-member task-bootstrap ensemble, 5 outer task-held-out folds, 5 training
  seeds (10–14), 60 epochs;
- inner calibration tasks (`calibration_tasks`) select `(risk_thr, adv_thr)` on
  the outer-train partition only; the grid search keeps success gap ≥ −5pp and
  false-continue ≤ 5%, then maximizes (savings, gap, −harm);
- **fold-correct aggregation**: each fold's held-out predictions are scored by
  that fold's own train-derived controller; counts are summed across folds.
  Averaging thresholds and re-evaluating all OOF rows (avg-threshold error) is
  explicitly forbidden;
- enrichment states may be over-sampled inside training folds, never in the
  natural eval gate; all states/seeds/replicas of a task stay in one fold.

## Gate (per VLA, ≥4/5 seeds)

- fold-correct success gap ≥ −5pp;
- original-protocol false-continue ≤ 5%;
- absolute paired harm ≤ 5%;
- teacher-step savings ≥ 20%;
- no suite-concentrated harm (every suite gap ≥ −5pp and harm ≤ 5%);
- conditional missed-rescue reported as point estimate + task-cluster interval
  only (no under-powered hard gate).

## Modes (R6-C.2 ladder)

`per_vla`, `shared`, `shared_id`, `shared_desc`, `shared_calib`, `loo`
(leave-one-VLA-out, descriptor from a task-disjoint few-shot calibration split),
`zero_shot` (challenge metric only).

`shared*` modes run **once per seed** (the report carries per-policy
`metrics_by_policy` and `metrics_by_policy_suite`); `per_vla`/`loo`/`zero_shot`
run per policy/direction.  The per-VLA stage gate reads each policy's own
fold-correct metrics and suite concentration — pooled aggregates are never used
for a per-VLA verdict.

`zero_shot` is a **pure shared-core** probe: no VLA identity embedding, no
descriptor (red line 5 — identity/descriptor are seen-VLA mechanisms; the
challenge metric exercises transfer of the risk core alone).

## Smoke validation (B1.2, low epochs, single/few seeds)

- `per_vla pi05`: savings 66.6%, success gap +3.2pp, absolute paired harm 5.3%,
  false-continue 6.0% — pipeline trains and produces fold-correct metrics;
- `shared_calib` per-policy (seed 10, 6 epochs): pi05 gap +0.011 / harm 1.1% /
  savings 36.9%; pi0fast gap −0.042 / harm 0% / savings −7.4% — per-policy
  metrics are distinct and correctly routed through the audit (pooled gap was
  −0.091, which would have masked pi05's pass);
- `loo` and `zero_shot` smoke-pass end-to-end through the stability audit.

## Files

- `rase/risk/light_risk_student.py` — model (policy conditioning + FiLM + advantage head)
- `scripts/train_r6c1_early_selector.py` — trainer / controller / fold-correct gate
- `scripts/run_r6c1_early_selector_oof.sh` — 5-seed OOF orchestrator (per mode)
- `scripts/audit_r6c1_selector_stability.py` — R6-C.1 stage gate aggregation
- `scripts/calibrate_r6c1_fewshot.py` — R6-C.2 0/8/16/32-shot calibration curve
- `scripts/compare_r6c1_configs.py` — R6-C.2 cross-config comparison
- `scripts/run_r6c1b_resume.sh` — full 1B→1C autonomous pipeline driver
