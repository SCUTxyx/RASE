# RASE R4 Implementation Summary

Generated: 2026-08-08

## Completed Work (8/12 todos)

### 1. Fix World Model Dynamics [DONE]
- **Problem**: MLP dynamics achieved only 2.1% improvement (gate: >=10%)
- **Solution**: Created `train_r4_safe_handback_wm_ridge.py` using Ridge regression with latent*action interactions
- **Result**: 12.9% dynamics improvement, passing the 10% gate
- **Script**: `scripts/train_r4_safe_handback_wm_ridge.py`
- **Best run**: `runs/pre_c0_r4/world_model_ridge_v4_fixed/`

### 2. Diagnose False Handbacks [DONE]
- 4-5 false handbacks on 71-state dataset, concentrated in folds 1-3
- Problem: per-fold threshold calibration on small calibration sets (~70 rows) was too optimistic
- Fix: Switched to global threshold selection on aggregated OOF predictions

### 3. Task-History Features [DONE]
- Added: OFT progress ratio, remaining ratio, timestep progress, boundary-zero indicator, student action entropy, Student-OFT action divergence, suite one-hot (4 suites), task ordinal
- State dimension: 128 (latent) + 8 (proprio) + 9 (history) = 145

### 4. Ridge Retrain with History Features [DONE]
- **Best config**: h128, ensemble_size=3, ridge_alpha=1000, cost_credit=0.10, history features
- **Results at selected threshold (0.95)**:
  - Dynamics improvement: 12.86% PASS
  - Handback AUC: 0.94 PASS
  - Handback AP: 0.79 PASS
  - False handback: 3.08% (2 fbs) PASS
  - Success delta: -2.82pp PASS
  - OFT savings: 19.26% **FAIL** (gate: >=20%, 0.74pp short)
- **5/6 gates pass, only 0.74pp from all 6**

### 5. R4-D Design Script [DONE]
- Created `scripts/make_pre_c0_r4d_design.py`
- Generated design: 32 train tasks, 8 validation tasks, ~480 estimated states
- Output: `runs/pre_c0_r4/r4d_train_design.json`

### 6. Baselines [DONE]
- Created `scripts/evaluate_r4_handback_baselines.py`
- Results:
  - Oracle earliest-safe: 30.7% savings, 0 harm (upper bound)
  - Fixed early (h0/h32/h64): catastrophic (45-49 false handbacks)
  - Deterministic progress: <3% savings (too conservative)
  - Risk-only never-handback: 0% savings
- **Our Ridge model at 19.26% is best non-oracle result**

### 7. DynamicsBackend API [DONE]
- Created `rase/dynamics_backend.py` with ABC + 3 implementations:
  - RidgeDynamicsBackend (latent*action interactions)
  - LinearDynamicsBackend (no interactions)
  - PersistenceBackend (zero-delta)

### 8. Ablation Matrix [DONE]
- Created `scripts/run_r4_ablation_matrix.py`
- **Label misalignment: 8.60%** (38/442 rows historical vs live)
- Key findings:
  - Ridge (12.86%) > Linear (11.57%) > Persistence (0%)
  - History features: reduce false handbacks (2 vs 3)
  - h128 outperforms h96 (19.26% vs 16.65% savings)
  - Larger ensemble helps marginally (+1.1pp savings)

---

## Remaining Work (4 todos, require long-running GPU execution)

### 9. Scale Collection [PENDING - manual, 2-4 days]
```bash
# Step 1: 20-state smoke test
cd /root/autodl-tmp/RASE
conda run -n smolvla python scripts/collect_r4_boundary_transitions.py \
    --design runs/pre_c0_r4/r4d_train_design.json \
    --output-dir runs/pre_c0_r4/boundary_train_v4_smoke \
    --smoke --smoke-states 20

# Verify: persistent parity, projection hash, live label schema

# Step 2: Full >=300-state collection (requires OFT servers for all 4 suites)
conda run -n smolvla python scripts/collect_r4_boundary_transitions.py \
    --design runs/pre_c0_r4/r4d_train_design.json \
    --output-dir runs/pre_c0_r4/boundary_train_v4 \
    --target-states 300
```

### 10. R4-F Gate [PENDING - after collection]
```bash
conda run -n smolvla python scripts/train_r4_safe_handback_wm_ridge.py \
    --dataset runs/pre_c0_r4/boundary_train_v4/boundary_transitions.jsonl \
    --collection-report runs/pre_c0_r4/boundary_train_v4/report.json \
    --output-dir runs/pre_c0_r4/world_model_r4f \
    --folds 6 --ensemble-size 5 --epochs 250 --patience 40 \
    --hidden-dim 128 --ridge-alpha 1000.0 --device cuda --lr 2e-4 --cost-credit 0.10
```

Gate criteria (all must pass):
1. Dynamics improvement >= 10% over persistence
2. OFT step reduction >= 20%
3. False handback <= 5%
4. Success within 5pp of persistent
5. No catastrophic degradation on any single suite

**If all gates pass with >=300 states**: freeze code, checkpoints, thresholds, seeds, and proceed to R4-G.

**If gates still fail**: shrink paper to benchmark/negative-result or redesign Student/OFT interface.

### 11. R4-G Validation [PENDING - only if R4-F passes]
One-shot validation pilot with 24 held-out states:
- Freeze: code commit, checkpoints, projection, thresholds, stopping rule, max horizon, seeds, exclusion/QC rules, statistical analysis script
- Paired comparison: Student, persistent, best fixed finite, deterministic baseline, risk-only, full RASE
- At least 2 rollout seeds; shared initial conditions per state/seed

Validation gate:
- Paired success within 5pp of persistent
- OFT step reduction >= 20%
- False handback <= 5%
- Clean/base-success harm <= 5%
- No catastrophic degradation on any suite
- Latency within deployment budget

### 12. R4-H Final Evaluation [PENDING - only if validation passes]
- 100-200+ paired episodes
- Two policy pairs (SmolVLA/OFT + second Student/teacher)
- At least one distribution-shift setting
- Statistical analysis: paired bootstrap, cluster bootstrap, McNemar tables, calibration curves, Pareto frontier

---

## Key Files Created/Modified

| File | Purpose |
|------|---------|
| `scripts/train_r4_safe_handback_wm_ridge.py` | Ridge-based world model trainer (new) |
| `scripts/make_pre_c0_r4d_design.py` | R4-D design generator (new) |
| `scripts/evaluate_r4_handback_baselines.py` | Baseline evaluation (new) |
| `scripts/run_r4_ablation_matrix.py` | Ablation matrix runner (new) |
| `rase/dynamics_backend.py` | Unified dynamics backend API (new) |
| `runs/pre_c0_r4/world_model_ridge_v4_fixed/report.json` | Best world model results |
| `runs/pre_c0_r4/r4d_train_design.json` | 300-state train design |
| `runs/pre_c0_r4/baselines_comparison.json` | Baseline comparison results |
| `runs/pre_c0_r4/ablation_matrix.json` | Ablation matrix results |

---

## Bottleneck Analysis

The current 71-state dataset limits the model to 19.26% savings (0.74pp short of 20% gate).
With >=300 states, the model should gain enough statistical power to confidently identify
the remaining ~3% of oracle savings (30.7% oracle - 19.26% current = 11.4% remaining).
Even capturing 5pp more from scaling would push us well past the 20% gate.
