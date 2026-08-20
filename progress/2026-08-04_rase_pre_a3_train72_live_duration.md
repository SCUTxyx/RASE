# PRE-A3 train72 live duration — 2026-08-04

## Status
**Step 2 complete** (plumbing).

## Outputs
- run: `runs/rase_pre_a3_recovery_duration120_train_v1/`
- audit: `runs/rase_pre_a3_recovery_duration_audit120_train_v1/`
- LIVE_DURATION lines: 576

## Note
Train-only results are for plumbing / resume / cost checks. Not confirmatory.
Do not retune h / cohort / gates from train.

## Next
Val24 live duration go/no-go.

## Plumbing checks
- 576 LIVE_DURATION lines (= 72 states × 8 arms)
- All four suites completed: spatial → object → goal → 10
- summary: `runs/rase_pre_a3_recovery_duration120_train_v1/summary.json`
- audit_train written under `runs/rase_pre_a3_recovery_duration_audit120_train_v1/`
- Not used for confirmatory claims or gate retune
