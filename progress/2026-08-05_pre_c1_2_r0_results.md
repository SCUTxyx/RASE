# PRE-C1.2 R0 Pivot — Results

**Date:** 2026-08-05  
**Status:** R0 complete → revised short-horizon train → post-train R(k) eval complete → **regression detected; C1.1 adapter remains best**

## One-line outcome

Kept DAgger R1 as the engineering carrier, inserted R0 diagnostics, **did not** run legacy E3/E4 full OFT-action BC, and trained a revised early-query / short-horizon adapter under branch `revised_short_horizon_training`.

## Hard stop

- Legacy E3/E4 blocked by default (`ALLOW_LEGACY_E3_E4` required).
- Pipeline hard-stop marker: `runs/rase_pre_c1_2_pipeline_hard_stop.json`.

## DAgger R1 global QC (frozen)

| Metric | Value |
|---|---|
| Anchors | 9/9 × 5 seeds |
| OFT queries | 1755 |
| Successful teacher queries | 484 |
| P(OFT success \| student query) | **0.276** |
| Accepted rows | 1452 (query:suffix = 484:968) |
| Failed teacher JSONs (not in BC) | 1271 |
| Median teacher recovery length | 110 |
| Success by trigger | anchor_start 45 / periodic 223 / progress_stall 216 |
| Success query-index quartiles | p25=4, p50=9, p75=18 |

Two anchors fail Round-1 per-anchor minimums (mostly `anchor_start` only): `sp1_ca72…`, `sp1_c793…` — late student states are often irrecoverable.

Artifacts:

- `artifacts/pre_c1/pre_c1_2_dagger_global_qc_r1.json`
- `runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl` (1880 rows incl. original+clean)
- `progress/2026-08-05_pre_c1_2_dagger_r1_global_qc.md`

## R0-A Teacher-forced

Adapter fits OFT **much** better than base on both distributions:

| Bucket | n | base loss | adapted loss | Δ |
|---|---:|---:|---:|---:|
| original C1.1 | 418 | 1.003 | **0.028** | −0.975 |
| R1 student_query | 484 | 1.124 | **0.060** | −1.064 |
| R1 teacher_suffix | 968 | 1.086 | 0.046 | −1.040 |
| clean | 10 | 1.230 | 0.043 | −1.188 |

⇒ Not an optimization / “never learned OFT” failure. Distribution + closed-loop recoverability remain primary.

Artifact: `runs/rase_pre_c1_2_r0_teacher_forced_v1.json`

## R0-B/C Recoverability (C1.1 adapter)

\(R(k)=P(\text{OFT succeeds after }k\text{ student steps})\) on locked 9 states:

| k | base | adapted (C1.1) |
|---:|---:|---:|
| 0 (OFT direct) | 1.000 | 1.000 |
| 1 | 0.889 | 0.889 |
| 2 | 0.778 | 0.667 |
| 4 | 0.556 | **0.667** |
| 8 | 0.444 | 0.556 |
| 16 | 0.444 | 0.444 |

- OFT replan@1: 0.778
- Adapted has small short-horizon gain at k=4/8 vs base; no one-step gain.
- Decay is gradual, not a single-step cliff.

Artifact: `runs/rase_pre_c1_2_r0_recoverability_v1/summary.json`

## R0 decision

```text
branch: revised_short_horizon_training
legacy_e3_e4_allowed: false
capacity_ladder_allowed: false
next:
  - build_early_query_dataset
  - residual_or_prefix_weighted_short_horizon_train
  - gate_on_R_k_before_terminal_8pp
```

Artifact: `runs/rase_pre_c1_2_r0_decision_v1.json`  
Progress: `progress/2026-08-05_pre_c1_2_r0_decision.md`

## Revised training (executed)

- Dataset: early / priority-trigger `student_query_state` only; suffixes dropped; original downsampled.
  - selected **372 / 1880** → `runs/rase_pre_c1_2_revised_dataset_r1_v1.jsonl`
- Residual targets annotated: `Δa = a_OFT − a_base`
  - `runs/rase_pre_c1_2_revised_dataset_r1_residual_v1.jsonl`
- Train: native flow + short-horizon weighting; student-query dominated; 5 epochs
  - adapter: `runs/rase_pre_c1_2_lora_revised_r1_v1/adapter_final`
  - final mean loss ~0.033

## Still paused

- Legacy E3/E4 full OFT-action BC
- Capacity ladder (until TF/R(k)/target gates fail in a capacity-specific way)
- Terminal +8pp as **first** acceptance criterion (kept as final gate only)

## Post-train R(k): Revised adapter vs C1.1

R(k) survival grid on locked 9 anchors for the revised adapter:

| k | base | C1.1 adapt | revised adapt | Δ(C1.1) | Δ(base) |
|---:|---:|---:|---:|---:|---:|
| 0 (OFT) | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 |
| 1 | 0.889 | 0.889 | 0.889 | +0.000 | +0.000 |
| 2 | 0.778 | 0.667 | 0.667 | +0.000 | −0.111 |
| 4 | 0.556 | **0.667** | **0.333** | **−0.333** | **−0.222** |
| 8 | 0.444 | 0.556 | **0.667** | **+0.111** | +0.222 |
| 16 | 0.444 | 0.444 | 0.444 | +0.000 | +0.000 |

### Gate checks (not terminal 8pp)

| Gate | Result |
|---|---|
| R_rev(1) ≥ R_base(1) | **PASS** (0.889 = 0.889) |
| R_rev(4) ≥ R_base(4) | **FAIL** (0.333 < 0.556) |
| R_rev(1) ≥ R_C1.1(1) | **PASS** (0.889 = 0.889) |
| R_rev(4) ≥ R_C1.1(4) | **FAIL** (0.333 < 0.667) |

### Verdict

**Revised short-horizon adapter fails the intermediate gate.** R(4) collapsed from 0.667 (C1.1) to 0.333 — a significant regression. The only gain is at k=8 (0.667 vs C1.1 0.556), but this is not reliable given the k=4 collapse.

### Root cause hypotheses

1. **Aggressive downsampling** (372/1880 rows) — lost mid-horizon coverage needed for k=4 states
2. **Overfit on k=1-like states** — early student query states are all near-distribution, so k=1 couldn't gain; mid-horizon states poorly represented
3. **Residual target noise** — OFT-base residual at k≥4 states may be large / poorly conditioned

### Next steps

- **Fall back to C1.1 adapter** as best current model
- Consider hybrid: re-add partial original recovery rows (e.g. 30-50% vs current ~20%) + retrain
- Consider further DAgger round(s) to collect more mid-horizon (k=2-8) student query states
- Terminal 8pp success remains final gate only — not to be used for intermediate yes/no

Artifacts:
- `runs/rase_pre_c1_2_r0_recoverability_revised_v1/summary.json`
- `runs/rase_pre_c1_2_r0_recoverability_revised_v1/trials/*.json`
