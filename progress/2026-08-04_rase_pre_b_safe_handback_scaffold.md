# PRE-B safe-handback scaffold — 2026-08-04

## Status

**Scaffold only. Not a paper claim.**

PRE-B dataset builder and calibrated baselines are implemented, but the
termination-model gate is closed until PRE-A3 confirmatory hidden/val gates
pass.

## Implemented

| Component | Path |
|-----------|------|
| Dataset builder | `scripts/build_pre_b_safe_handback_dataset.py` |
| Baselines | `scripts/train_safe_handback_baselines.py` |
| Tests | `tests/test_pre_b_safe_handback.py` |
| Dev scaffold dataset | `runs/rase_pre_b_safe_handback_scaffold_from_pre_a2_v1.json` |
| Dev scaffold baselines | `runs/rase_pre_b_safe_handback_baselines_scaffold_v1.json` |

## Gate behavior

- Builder refuses to emit a claim dataset unless `audit.gate_pass` is true
- Baseline trainer refuses to run unless `method_gate.termination_model_gate==open`
- World model remains unused (`world_model_used=false`)

## After PRE-A3 pass

```bash
python scripts/build_pre_b_safe_handback_dataset.py \
  --audit runs/rase_pre_a3_recovery_duration_audit120_v1/audit_test.json \
  --output runs/rase_pre_b_safe_handback120_v1.json

python scripts/train_safe_handback_baselines.py \
  --dataset runs/rase_pre_b_safe_handback120_v1.json \
  --method-gate runs/rase_pre_a3_recovery_duration_audit120_v1/method_gate.json \
  --output runs/rase_pre_b_safe_handback_baselines120_v1.json
```

Compare against fixed durations, always-OFT, and calibrated threshold; report
success / OFT-steps / false-handback harm Pareto.
