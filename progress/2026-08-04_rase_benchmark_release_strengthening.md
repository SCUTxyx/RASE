# Benchmark release strengthening — 2026-08-04

## Status

Release manifest assembled; several multi-seed / second-backbone executions remain
pending GPU time after PRE-A3 confirmatory collection.

## Ready

- Evidence index: `reports/rase_benchmark_release_manifest_v1.json`
- Seed 1/2 baseline configs: `configs/eval_base_seed1.yaml`, `configs/eval_base_seed2.yaml`
- Second policy-pair protocol: `protocol/second_policy_pair_smoke_v1.md`
- PRE-A3-S opportunity spec: `runs/rase_pre_a3_s_opportunity_spec_v1.json`
- Required metrics/statistics/figures listed in the manifest

## Pending executions

1. clean baseline seed 1 and seed 2 (2000 ep each under frozen eval_base)
2. second policy-pair smoke (16-state)
3. PRE-A3 120-state collect + live duration confirmatory
4. cost Pareto export from confirmatory arms
5. paper figures

## Claim-safe rule

Do not wait for pending extras to rewrite currently supported W7/W8/W9C/W10/PRE
negative claims. Pending items only strengthen variance, generality, and release
quality.
