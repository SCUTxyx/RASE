# R7 Recovery Student — Independent Fallback-Arm Project Plan

Date: 2026-08-10  
Status: **PLANNED, NOT STARTED** (must not preempt the R6-C selector mainline)

## Positioning

The R7 recovery student is a **candidate fallback arm** for the R6-C selector,
not a replacement for it.  It is a separate policy-learning paper and is kept
out of the selector paper's core claims.  Everything here is contingent on the
R6-C no-world-model baseline passing its 5-seed per-VLA stage gate first.

| Route | Evidence | Positioning |
|---|---|---|
| Short-horizon action correction + handback to source | PRE-A1/A2 reject | not a main method |
| Persistent OFT takeover | shown to rescue part of source failures | current fallback arm |
| Risk prediction + selector | R6-A/B1 opportunity + dynamic data | **mainline** |
| OFT trajectory + LoRA + DAgger | none yet | **R7 independent fallback arm** |
| "Correct OFT's own failures" | no evidence | cannot be claimed via DAgger |

The critical logic: distilling OFT recovery trajectories into a LoRA student can
at best *compress/imitate* OFT's recovery capability (lower latency/memory).
It cannot automatically solve states where OFT itself fails.  Beating OFT would
require a strictly stronger supervision source (human demos, privileged
simulator planner, extra sensors, or reward-based improvement) — that is a
different paper.

## R7 Goal

Distill OFT's persistent-recovery trajectories into a lightweight recovery
student so the selector gains a lower-cost fallback arm `ENTER_RECOVERY_STUDENT`
that approximates OFT's rescue coverage.

## Scope (candidate-arm framing)

- Source of demonstrations: OFT recovery trajectories (the `persistent_branch`
  counterfactual branches already recorded by `collect_r6b1_dynamic_boundaries.py`
  starting at each B1.2 boundary).
- Distillation: LoRA / adapter on a VLA backbone, DAgger-style relabeling to
  reduce covariate shift.
- Target metric: approximate OFT rescue coverage, lower latency + memory.

## Non-goals

- Fixing OFT's own failures (requires stronger supervision; out of scope).
- Handback-to-source after recovery entry (default: no handback).
- Claiming "universal VLA correction" from two VLAs.

## Required comparison (when started)

Direct head-to-head vs OFT on the same rescue boundaries:

1. rescue rate (persistent success at each boundary),
2. failure rate,
3. teacher/action steps (latency),
4. GPU memory footprint.

The selector then picks the lowest-cost arm among those meeting the safety
floor (`P(success) LCB` above the threshold).

## Entry conditions

- [ ] R6-C per-VLA stage gate passes (≥4/5 seeds, both Pi0Fast and Pi0.5).
- [ ] Candidate-arm schema already has a reserved `ENTER_RECOVERY_STUDENT` arm
      (already declared in `build_r6c_dynamic_dataset.py` future_arms).
- [ ] Recovery trajectories exported from B1.2 persistent branches (new
      exporter) with a parity audit.

## Placement in the R6-C architecture

```text
shared risk backbone
  -> source-risk head / arm-success heads / arm-cost heads
  -> LCB-constrained expected-utility selector
  -> decision: CONTINUE_SOURCE | ENTER_PERSISTENT_OFT | (later) ENTER_RECOVERY_STUDENT | SAFE_ABORT
```

The recovery student only enters the arm list after it independently passes the
direct OFT comparison.

## Artifacts

- This document is the plan lock.  R7 work must be scheduled as its own plan
  file (like `r6c_多臂风险-收益_selector_主线_*.plan.md`) when it starts.
