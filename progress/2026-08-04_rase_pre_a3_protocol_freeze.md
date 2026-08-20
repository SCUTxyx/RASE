# PRE-A3 protocol freeze — 2026-08-04

## Status

**Frozen before confirmatory outcomes.** Design, splits, metrics, kill criteria,
live closed-loop runner, analysis gates, and method-decision machinery are
landed. Confirmatory 120-state collection/rollouts remain to be executed.

## Frozen question

How long must closed-loop OFT remain in control before frozen SmolVLA can safely
continue, under false-handback harm constraints?

## Frozen artifacts

| Artifact | Path |
|----------|------|
| Protocol | `protocol/pre_a3_recovery_duration_v1.md` |
| Design (120) | `runs/rase_pre_a3_design120_v1.json` |
| Collect config | `configs/collect_pre_a3_recovery120.json` |
| Eval config | `configs/pre_a3_recovery_duration120.yaml` |
| Smoke keys | `runs/rase_pre_a3_smoke12_keys_v1.json` |
| Core library | `rase/collect/pre_a3.py` |
| Live runner | `scripts/rollout_live_oft_duration_to_smol.py` |
| Pipeline | `scripts/run_pre_a3_recovery_duration.sh` |
| Analysis | `scripts/analyze_pre_a3_recovery_duration.py` |
| Method gate | `scripts/decide_pre_a3_method_gate.py` |

## Design identity

- `design_sha256`: `db5d12e7ce62d542250ee559e31f14696488324d881a4267d219c997cf91ebf8`
- 40 logical tasks × 3 conditions = 120 states
- split seed `2026080401` → train/val/test = 72/24/24
- clean-10 reused with new episode seeds; Plus camera/robot L1 excludes prior
  development concrete tasks
- durations `h={0,8,16,32,64,96,128}` + persistent OFT
- execution mode: **live closed-loop** (replay diagnostic only)

## Preregistered hidden-test gate

All required:

1. oracle gap ≥ 8pp and task-bootstrap lower bound > 0
2. ≥ 4 task-disjoint rescues
3. rescues cover ≥ 2 suites and ≥ 2 cells
4. duration heterogeneity
5. best fixed-h harm ≤ 5% of base successes
6. adaptive oracle headroom over best fixed-h ≥ 5pp

Fail → `benchmark_diagnosis_only`. Pass → open termination/safe-handback only.
World-model and candidate-critic gates remain closed.

## Stop rules preserved

- no ridge/MLP/RL three-arm selector reopen
- no same-profile temperature candidate scaling
- no generative world-model training before residual predictive gap is proven

## Immediate next commands

```bash
# 1) collect confirmatory pool from frozen design
# 2) freeze keys
python scripts/freeze_pre_a3_keys_from_pool.py \
  --design runs/rase_pre_a3_design120_v1.json \
  --pool runs/rase_pre_a3_recovery120_pool_v1 \
  --output runs/rase_pre_a3_keys120_v1.json

# 3) live duration confirmatory pipeline
KEYS=runs/rase_pre_a3_keys120_v1.json FRESH_RUN=1 \
  ./scripts/run_pre_a3_recovery_duration.sh
```

## Tests

```text
pytest tests/test_pre_a3_protocol.py tests/test_pre_b_safe_handback.py \
  tests/test_recovery_duration.py
# 9 passed
```
