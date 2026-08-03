# W7 causal attribution and held-out validation

## Status

**Causal attribution and held-out paired evaluation complete.** The W6
OFT-only hits do not provide candidate-specific rescue evidence after explicit
controls. A fully episode-disjoint 24-state validation cohort is frozen and its
resumable end-to-end pipeline completed on 2026-07-29.

### Historical execution checkpoints

Live checkpoint (2026-07-29 12:23 CST): all 24 candidate artifacts were written
successfully and the SmolVLA one-shot screen was running. The background pipeline
process and GPU workload are healthy; candidate generation did not fail or skip
any state.

Interim checkpoint (2026-07-29 12:49 CST): **55/192** SmolVLA candidate
rollouts (7/24 states entered) are complete, with 0 successes observed so far.
This is a live engineering checkpoint, not a frozen scientific result; the
screen must finish before interpreting the held-out rate.

Verified checkpoint (2026-07-29 14:03 CST): the SmolVLA screen completed with
**0/192 candidate hits and 0/24 portfolio-state hits**. OFT Spatial completed
with 4/32 candidate hits concentrated in 1/4 states; Object completed at 0/16
and 0/2 states. Goal had reached 16/39 candidate hits, concentrated in two
8/8-hit states, when the external SSH relay stopped allocating sessions. The
remote locked pipeline and conditional attribution watcher remained independent
background processes; the later final result supersedes this checkpoint.

## Prefix-ablation result

- Frozen discovery states: 2 W6 `oft_only` states
- Arms per state: direct OFT, time-matched zero prefix, and 8 frozen SmolVLA
  candidate prefixes
- Combined summary: `runs/ngc_w7_prefix_ablation.json`
- Candidate-specific rescue states: **0/2**
- Mechanism counts:
  - `continuation_sufficient_candidate_invariant`: 1
  - `passive_prefix_sufficient`: 1

Long/camera-L1 `sp1_9224df57d14685d1a38858b3fa311f17` succeeds with direct
OFT, zero prefix, and all 8 candidates. Its W6 recovery is attributable to OFT
continuation sufficiency rather than proposal quality.

Goal/robot-L2 `sp1_f0c5d3a950eb6035ce142f78aca29ddd` fails with direct OFT,
but succeeds after the 10-step zero prefix and after candidates 4/5. Therefore
the two candidate hits are not uniquely causal: passive/no-op dynamics are a
sufficient alternative intervention.

This closes the current candidate-generation claim. The supported research
direction is state-conditioned continuation/escalation, with explicit action
cost and clean-regret controls. Candidate-level hit counts remain descriptive.

## Final held-out result

- SmolVLA: **0/192 candidates and 0/24 portfolio states**
- Prefix+OFT: **8/24 portfolio states**
- Pairs: both-hit 0, Smol-only 0, OFT-only 8, both-miss 16
- Exact state-level McNemar: **`p=0.0078125`**
- Follow-up W8 direct OFT: **9/24 states**

This establishes a policy-relative continuation gap on a frozen,
episode-disjoint failure cohort. It does not establish unconditional recovery
rate or clean-regret performance. See
`progress/2026-07-29_w8_direct_escalation_results.md`.

## Held-out cohort audit

- Artifact: `runs/ngc_w7_heldout24_state_keys.json`
- Split audit: `runs/ngc_w7_heldout24_split_audit.json`
- 24 states from 24 distinct episode groups
- Zero state-key and zero episode-group overlap with W6
- Balanced cells: camera-L1=6, camera-L2=6, robot-L1=6, robot-L2=6
- Suite counts: Spatial=4, Object=2, Goal=8, Long=10
- Snapshot timing audit: `t0` min=0, median=10, max=36. The cohort has
  early-timestep variation but does **not** cover mid/late recovery; no temporal
  generalization claim is permitted.
- Frozen key SHA-256:
  `866ae9e7c6f95088dc7d761c63937a02d017b69280e6d9cc73ac883127158829`

## Completed experiment

The resumable pipeline completed:

- Script: `scripts/run_w7_heldout_pipeline.sh`
- Log: `runs/ngc_w7_heldout24_pipeline.log`
- Stages: candidate generation → Smol one-shot screen → four OFT suites →
  paired state-level matrix
- Candidate temperature and checkpoint are frozen to W6 (`T=0.7`, `K=8`)
- No tuning is permitted after observing held-out outcomes
- Candidate generation: **24/24 complete**
- SmolVLA screen: **complete**
- OFT paired suites and final matrix: **complete**

Expected final artifacts:

- `runs/ngc_w7_heldout24_smol_screen_t07/summary.json`
- `runs/ngc_w7_heldout24_oft_{spatial,object,goal,10}_heldout/summary.json`
- `runs/ngc_w7_heldout24_policy_matrix.json`
- `runs/ngc_w7_heldout24_policy_matrix.md`

The primary endpoint is the paired state-level `OFT-only` versus `Smol-only`
count and exact McNemar test. Candidate-level rates are secondary descriptive
statistics. This failure-conditioned cohort does not estimate unconditional
task success or clean-policy regret.
