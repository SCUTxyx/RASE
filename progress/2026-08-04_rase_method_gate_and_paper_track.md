# Method gate decision and paper-track freeze — 2026-08-04

## Decision status

**Current decision: `benchmark_diagnosis_only` (pending confirmatory PRE-A3).**

Reason: PRE-A3 confirmatory hidden-test outcomes are not yet available. The
protocol and gate code are frozen; until `audit_test.json` reports
`gate_pass=true` with `val` also passing, termination/safe-handback training and
world-model work remain closed.

Artifact: `runs/rase_pre_a3_method_gate_pending_v1.json`

## Allowed now

- policy-relative recoverability tables (W7/W8)
- mechanism falsification chain (candidate/temperature/short-replan/timing)
- W9C ridge selector honest negative
- W10 Object/Spatial suite boundary
- PRE-A3 collection + live closed-loop confirmatory execution
- benchmark release packaging and multi-seed baseline completion

## Forbidden now

- ridge/MLP/RL three-arm selector reopen
- termination model training for paper claims
- candidate critic training
- generative world-model training
- rewriting PRE-A2 12-state pilot as confirmatory evidence

## Paper track freeze

### Primary track (active)

**Benchmark / diagnosis:** policy-relative recoverability under visual/robot
perturbations, with strict snapshot counterfactuals and preregistered negative
results on selector/candidate/timing branches.

Working title options:

1. *Policy-Relative Recoverability of Frozen VLAs under Visual Perturbations*
2. *When Escalation Helps: A Diagnosis Benchmark for VLA Recovery Decisions*

### Conditional track (only if PRE-A3 gate passes)

**Safe handback after temporally extended recovery**, with calibrated termination
baselines beating fixed-duration and matched-cost random on hidden tasks.

### Explicitly rejected novelty frames

- learned timing selector
- same-policy candidate ranking without heterogeneous unique successes
- world-model-first recovery without residual predictive gap

## Required main figures (benchmark track)

1. Recoverability matrix (suite × perturbation × policy pair)
2. Duration response + harm curve (PRE-A2 pilot + PRE-A3 confirmatory)
3. Success–cost Pareto (base / fixed-h / persistent OFT / oracle)
4. Claim boundary diagram (what is supported vs killed)

## Evidence checklist

See `reports/rase_benchmark_release_manifest_v1.json`.
