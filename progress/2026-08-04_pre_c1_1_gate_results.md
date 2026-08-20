# PRE-C1.1 recovery LoRA gate results

Date: 2026-08-04 / 2026-08-05

## Decision

| Field | Value |
|------|------:|
| decision | `capacity_or_data_review` |
| gate_pass | **false** |
| recovery_gain_pp | **0.0** |
| clean_retention_drop_pp | **0.0** |
| same_backbone_recovery_method | closed |
| abstention_track | not_required |

Frozen thresholds unchanged: recovery ≥8pp, retention drop ≤2pp.

## Eval (PRE-C0 current_suffix failures only)

| Arm | n | base success | adapted success |
|-----|--:|-------------:|----------------:|
| recovery (adapter on vs off) | 9 | 0 | 0 |
| retention (adapter off) | 6 | 3 | 3 |

## Vs PRE-C1

| Item | PRE-C1 | PRE-C1.1 |
|------|-------:|---------:|
| Teacher quality | 42× h=10, all fail | 35 successful OFT trajs, multi-chunk |
| Train chunks | 42 | 418 (311 train) |
| Val recovery gain | 0pp | 0pp |
| Retention drop | 0pp | 0pp |

## What was fixed in collection

1. Fixed h=128 → only 5/42 OFT successes (hard-stop)
2. Persistent episode → 9/42 (late forks starved of steps)
3. `persistent_min128_from_fork` + expand T0/T2/T4 → **35** successes / **418** chunks (hard-stop cleared)
4. Rewrote chunk obs with `robot_state` for SmolVLA BC
5. Eval restricted to PRE-C0 failures (not easy T0 teachers)

## Artifacts

- protocol: `artifacts/pre_c1/pre_c1_1_protocol_lock.yaml`
- teachers: `runs/rase_pre_c1_1_oft_success_v1/`
- chunks: `runs/rase_pre_c1_1_distill_chunks_v1.jsonl`
- dataset QC: `artifacts/pre_c1/pre_c1_1_dataset_qc.json`
- LoRA: `runs/rase_pre_c1_1_lora_train_v1/`
- eval: `runs/rase_pre_c1_1_eval_v1.json`
- decision: `runs/rase_pre_c1_1_decision_v1.json`

## Interpretation

Long successful OFT multi-chunk distillation + LoRA train (loss ↓) still yields **zero** closed-loop recovery gain on the 9 OFT-recoverable PRE-C0 failure states. Retention intact. Per protocol: further work is data/capacity only (not threshold retune; not runtime OFT; WM/handback stay closed).

Naming: recovery LoRA / OFT action distillation (offline teacher only).
