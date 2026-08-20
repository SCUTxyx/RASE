# PRE-A3 live closed-loop smoke4 kickoff — 2026-08-04

## Status

**Running / plumbing validation only.**

## Purpose

Validate the PRE-A3 live OFT→Smol handback runner, suite orchestration, merge,
and analysis path before collecting the confirmatory 120-state cohort.

## Frozen smoke inputs

- keys: `runs/rase_pre_a3_smoke4_keys_v1.json` (4 states from PRE-A0 subset)
- config: `configs/pre_a3_smoke4.yaml`
- pool: `runs/rase_ui_phase1a_replacement48_initial_pool_v2`
- durations: `h={0,8,16,32}` + persistent OFT
- suites: spatial, object

## Explicit non-claims

- not confirmatory
- not usable for termination-model gate
- excluded from flagship hidden test

## Command

```bash
FRESH_RUN=1 ./scripts/run_pre_a3_smoke4.sh
```
