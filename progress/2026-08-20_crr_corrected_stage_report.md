# RASE CRR corrected-stage report

Date: 2026-08-20  
Server: AutoDL `bjb2:28921`  
Scope: P1/P2 continuation, code audit, corrected-label pilot, and next-stage decision

## Executive decision

The current OFT-spatial/OFT-object LIBERO benchmark should **not** proceed to RGB, π0 integration, offline-gain claims, or closed-loop evaluation.

Two separate conclusions must be kept distinct:

1. Relative CRR can predict the legacy motion-proxy ordering with strong held-out metrics.
2. After correcting the causal rollout and using task-relevant recovery labels, the benchmark has **zero within-task state-level headroom over the task-best-fixed policy**.

Therefore the current-domain P2 Eligibility Gate is **FAIL**. The next action is domain/policy-pair mining, not a larger model.

## 1. What was reproduced

P0 provenance remains PASS:

- 648 rows = 216 physical roots × 3 candidates;
- root-state proprio is identical within each root;
- offline/deployment feature builders are numerically identical;
- vocabulary, normalizer, and labels reproduce;
- the legacy B1 AUROC reproduces.

This establishes implementation consistency for the stored features. It does not establish that the stored consequence label is task-relevant.

## 2. P1 CRR on the legacy proxy label

The formal v2 run used three-seed MLP ensembles, task-prior/B1 controls, chunk-only ablations, within-root label permutation, bootstrap confidence intervals, and corrected signed selector-gain accounting.

| Split | C-MLP accuracy | C-MLP AUROC | Coverage at precision ≥0.85 | Gate |
|---|---:|---:|---:|---|
| LOVO `oft_goal` | 0.799 | 0.830 | 0.778 | Strong PASS |
| Random root 80/20 | 0.860 | 0.928 | 0.992 | Strong PASS |
| Train spatial suite → test object suite | 0.750 | 0.850 | 0.278 | Strong PASS |

Mechanism checks:

- C-MLP permutation-control accuracy falls to 0.303 / 0.457 / 0.269 across the three splits.
- The chunk-only C-MLP remains strong for LOVO and random-root splits, but degrades on suite transfer (accuracy 0.639, AUROC 0.720).
- This supports a real chunk-to-proxy relationship, not a pure label-prior artifact.
- The old gain simulator was fixed: wrong switches are now negative, abstentions contribute zero, and gain is averaged over all eligible roots.

Scientific limitation: the target is `z(EEF displacement) - 0.5*z(object displacement)`. For pick-and-place, desired object motion can be penalized as “drift”. These P1 numbers are therefore a motion-proxy diagnostic, not evidence of task-success arbitration.

## 3. Invalidated historical evidence

### Gate C circularity

The historical `oracle_future_report.json` value AUROC 0.9998 was circular:

- when `recovery_success` was absent, `analyze_oracle_future.py` inserted `consequence_label` into the oracle score as recoverability;
- evaluation then compared that score against the same `consequence_label`.

The script now requires an independent `recovery_success` or `candidate_success` target and refuses proxy displacement as a Gate-C outcome.

### Wrong recovery start state

`collect_same_root.py --label-mode reference` claimed to evaluate recovery from `s_{t+H}`, but restored the common root snapshot `s_t`. It now captures the branch endpoint snapshot and starts the evaluator from the actual `s_{t+H}`.

### Repeated frozen chunk

The legacy 64-step collection repeated one native 8-step action chunk eight times. This is not policy continuation and can drive even the correct expert into artificial failure states. The collector now defaults to `single_chunk`, rejects `horizon > native_chunk_len`, and keeps repetition only as an explicitly named legacy reproduction mode.

Consequences:

- `same_root_w1.jsonl`, `crr_pairs.jsonl`, legacy P1/P2, Gate B/C, W1/W2, and any 64-step divergence derived from repeated chunks must be labeled **legacy proxy diagnostics**;
- they must not support claims about real candidate recoverability, task progress, or selector gain.

## 4. Corrected recovery-label experiments

Corrected protocol:

```text
same physical root s_t
→ execute one native 8-step candidate chunk once
→ capture candidate-specific s_{t+8}
→ run a closed-loop reference policy from s_{t+8}
→ requery the reference every native chunk
→ record task success and steps-to-recovery
```

The task-relevant label is pre-registered multi-horizon recovery survival:

```text
q_recovery = mean over K ∈ {64, 128, 256} of I(recovered within K steps)
```

### Eight-row validation

- With the invalid repeated-64-step candidate rollout, all recovery labels were 0 even with 520 reference steps.
- With a single native chunk, all four first-root candidates eventually recovered.
- Recovery times were spatial: 70 vs 75; object: 196 vs 124.
- This confirmed that branch-end recovery semantics were working and that a 520-step binary label alone has a ceiling.

### 64-row pilot

Setup: 4 tasks (2 spatial, 2 object), 2 episodes, 4 decision roots, 2 candidates = 64 branches.

| Recovery budget | Positive rate | Informative roots |
|---:|---:|---:|
| 64 | 14.1% | 15.6% |
| 128 | 73.4% | 53.1% |
| 256 | 85.9% | 28.1% |

Multi-budget recovery-q results:

- informative-root fraction: **56.25%**;
- legacy displacement ranking agreement: **83.3% on only 18 comparable pairs**;
- strict non-suite-favorite wins: **12.5% of roots**;
- mean oracle gain over suite-favorite: **0.0521 q units**;
- strict within-task winner-flip rate: **0**;
- state oracle gain over each task's best-fixed candidate: **0**.

Per-task result: every task has one fixed candidate that is always best or tied. One object task prefers `oft_spatial`, showing that suite identity is not a perfect router, but a task lookup still explains all gains.

## 5. Revised claim table

| Claim | Revised status |
|---|---|
| Same-root feature/provenance equality | PASS |
| CRR learns legacy motion-proxy ranking | PASS, diagnostic only |
| Gate C true-future ranking 0.9998 | INVALID (circular target leakage) |
| Legacy 64-step future divergence | INVALID for policy consequence (repeated frozen chunk) |
| Corrected branch-end recovery label is usable | PASS |
| Current OFT/LIBERO runtime opportunity | FAIL (`H_within=0`, task-best-fixed headroom=0) |
| Cross-architecture transfer | Unresolved |
| Risk-driven closed-loop gain | Unresolved; previous proxy-based run remains negative |

## 6. Next execution plan

### R0 — Freeze and relabel artifacts (immediate)

Do not delete legacy artifacts; mark them diagnostic-only in progress documentation. The corrected collector, leakage guard, signed-gain evaluator, and recovery-label audit become the only allowed path for new evidence.

### R1 — Mine a new eligible regime before training (highest priority)

Run a cheap same-root screen with at least 8 tasks × 4 uniformly sampled states/task × 2 candidates. Split and report by task. Required gates:

- corrected provenance PASS;
- 10%–90% recovery prevalence for at least one pre-registered horizon;
- strict `H_within ≥ 5%`;
- state oracle gain over task-best-fixed ≥ 5% of the recovery-q range;
- nonzero lower bootstrap bound for oracle gain if sample size allows.

Recommended order:

1. Goal/Long failure-frontier timing with `continue.source` versus corrective fallback, because existing direct/deferred evidence is closest to the gate but was measured at only one state per task.
2. If that fails, a heterogeneous cross-architecture pair (suite OFT expert versus π0-fast) on intermediate-difficulty tasks, using the corrected single-chunk/recovery protocol.
3. If both fail, leave current LIBERO policy set and screen a longer, contact-rich domain; do not restart World Model work until its direct B1 baseline is low and corrected oracle headroom exceeds 10 percentage points.

### R2 — Only after R1 passes

- build CRR pairs from `q_recovery`, not displacement;
- use task-held-out splits as primary; root-random split is diagnostic only;
- compare C-MLP/C-noctx against task-router and best-fixed;
- estimate selector gain strictly out of fold;
- require precision ≥0.85 at coverage ≥0.05;
- only then add frozen visual embeddings and action-conditioned interaction.

### R3 — Cross-architecture and closed loop

After corrected-label CRR and opportunity gates pass:

- collect OFT + π0 same-root candidates with no policy ID;
- test OFT→π0 relative ranking without test calibration;
- run offline conservative arbitration with signed gain and task bootstrap;
- run a small closed-loop smoke test only if the offline gain lower bound is positive;
- formal endpoint remains ≥3 percentage-point success improvement over best-fixed with paired CI lower bound >0.

## 7. Code and artifact locations

Remote code:

- `scripts/train_crr_baselines.py`
- `scripts/measure_within_task_heterogeneity.py`
- `scripts/analyze_recovery_labels.py`
- `scripts/collect_same_root.py`
- `scripts/analyze_oracle_future.py`
- `tests/test_crr_regressions.py`

Remote results:

- `runs/oft_opportunity/crr_p1_results.json`
- `runs/oft_opportunity/crr_p2_heterogeneity.json` (legacy proxy diagnostic)
- `runs/oft_opportunity/crr_recovery_pilot64_v1.jsonl`
- `runs/oft_opportunity/crr_recovery_pilot64_v1_audit.json`

The four P1 plots are delivered alongside this report. The gain plot must be labeled proxy-label/in-sample diagnostic and must not be used as a deployment claim.
