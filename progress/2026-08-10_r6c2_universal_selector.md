# R6-C.2: Multi-VLA generalization comparison — interpretation template

Date: 2026-08-10  
Status: TEMPLATE — to be filled from `runs/pre_c0_r6/r6c1_config_comparison.json`
and `runs/pre_c0_r6/r6c1_fewshot_curve.json` once the auto-chain completes
(screening -> collection -> merged dataset -> 5-seed OOF -> few-shot ->
comparison).

## Claim under test (from the revised plan)

> A shared risk core generalizes to a new VLA only with a deployable behavior
> descriptor and a small per-VLA calibration adapter.  Pure zero-shot is a
> challenge metric, not a gate.

## Config ladder (read from comparison JSON)

| mode | model | gate role |
|---|---|---|
| per_vla | independent per-VLA model | upper bound for same-VLA performance |
| shared | shared core, no policy condition | pure pooling baseline |
| shared_id | shared core + VLA identity embedding | adaptation to KNOWN VLAs only |
| shared_desc | shared core + deployable behavior descriptor | generalization signal |
| shared_calib | shared core + descriptor + FiLM calibration | main gate config |
| loo | leave-one-VLA-out (descriptor from few-shot calibration split) | simulates new VLA |
| zero_shot | train source, eval target, NO identity/descriptor | challenge metric only |

## Gate (each VLA, >=4/5 seeds, from `stability.json` per mode)

- fold-correct success gap >= -5pp
- original false-continue <= 5%
- absolute paired harm <= 5%
- teacher-step savings >= 20%
- no suite-concentrated harm (per-policy-per-suite check)

## Decision tree

1. shared_calib passes both VLAs -> the generalization claim is testable:
   compare few-shot curve (0/8/16/32-shot) vs per_vla data cost.  If
   shared_calib reaches per_vla with far fewer target-VLA labels, the claim
   "shared core + small calibration" is supported -> candidate for third VLA
   expansion.
2. shared_calib passes, per_vla fails -> policy-specific risk control claim
   only (cannot claim shared core helps).
3. shared fails but shared_calib passes -> the calibration adapter carries the
   adaptation; report as the mechanism evidence.
4. Everything fails -> revisit R6-C.1B data (hard-negative balance) or
   representation; do NOT enter validation (per plan decision tree).
5. loo beats zero_shot -> evidence that a deployable descriptor matters for
   unseen VLA adaptation.
6. zero_shot must be reported as challenge only; a failing zero_shot does NOT
   falsify the main claim (red line 5).

## Few-shot calibration curve (from `calibrate_r6c1_fewshot.py`)

- x = calibration shots (0/8/16/32), y = target-VLA success gap / savings /
  paired harm (fold-correct, with bootstrap intervals).
- Expected shape: monotone improvement toward per_vla; a flat curve indicates
  the adapter cannot absorb target-VLA variation (representational limit).

## Fill-in checklist

- [ ] comparison JSON status == complete
- [ ] per-mode stage_gate_passed / policy gate
- [ ] few-shot curve monotonicity + gap-to-per_vla at 32 shots
- [ ] decision from the tree above
- [ ] record in `progress/2026-08-10_r6c2_universal_selector.md`
