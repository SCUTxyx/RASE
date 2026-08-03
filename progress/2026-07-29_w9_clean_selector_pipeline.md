# W9 clean controls and direct-policy selector gate

## Status

**Blocked after preregistered clean collection.** Final result:
[W9 clean-control coverage gate](2026-07-29_w9_clean_control_coverage_gate.md).

Pipeline exit 2: after 60+40+40 clean episodes, suite×`t0_bin` coverage cannot
freeze 32 success states (Object/Goal/Long early+mid all `n=0`). Direct OFT /
clean32 labeling and selector training did **not** run.

### 2026-07-29 collection incident (resolved before final batches)

The W9 pipeline reached clean-control collection and raised
`UnsupportedEnvironmentError: task/model identity changed since wrapper
construction`. The task did not change. The v1 fingerprint hashed
`sim.model.get_xml()`, but mujoco-py can serialize runtime/lazy-renderer changes
into that XML within the same episode.

The hotfix introduces fingerprint schema `rase-task-identity/v2`: it retains
wrapper/task class, task id, instruction, BDDL path/content hash, and compiled
model topology/name hashes, while excluding mutable full XML. Cross-task
restore remains fail-closed. Collection resumed and completed the remaining
preregistered episodes under this hotfix.

## Frozen evidence entering W9

- Smol prefix portfolio: 0/24 states
- Prefix+OFT portfolio: 8/24
- Direct OFT: 9/24
- Prefix/direct overlap: both 7, prefix-only 1, direct-only 2, neither 14
- Prefix versus direct exact McNemar: `p=1.0`
- Direct OFT versus Smol portfolio exact McNemar: `p=0.00390625`
- Reusable clean-success episode groups across server pools: **0**

Direct OFT becomes the deployable escalation action because it recovers 7/8
oracle-prefix hits with one branch. The evidence does not show a significant
success-rate difference between the two OFT routes.

## Implemented changes

- Explicit clean controls: `perturb_dim=clean`, `perturb_sub=none`, `level=0`.
- Clean collection uses the ten original LIBERO tasks at the front of each
  LIBERO-Plus suite.
- `scripts/rollout_direct_smol.py` evaluates true empty-prefix continuation.
- Direct OFT no longer needs fake candidate artifacts.
- Initial costs: Smol 0.02, OFT 0.10, abstain 0.0.
- Ground-truth dim/sub/level/outcome inputs trigger readiness rejection.
- Current RGB statistics, proprioception, and t0 form the deployment feature
  baseline.
- Failure and clean cohorts receive all three direct counterfactual outcomes,
  checksummed merge, and episode/task-disjoint splits.
- The W9 entry point now runs the focused fingerprint, collection, sampling,
  feature, and selector tests before any simulator/GPU rollout.
- Evaluation reports both escalation-budget-matched random triggering and a
  stricter random baseline matched to the learned counts of all three actions.
- Readiness now requires cohort-semantic clean-success and failure-challenge
  support (not merely any success/failure outcome labels).
- Selector gate summaries emit a task-heldout method decision from the paired
  utility bootstrap against action-count-matched random.

Local verification: Python compilation and shell syntax pass; an 8-episode
clean dry run wrote 20 valid L0 state bundles; eight lightweight-selector tests,
the deployable-feature check, and clean task-selection check pass. The sync
entry point runs the full focused pytest set in the server `smolvla` environment
before launching any GPU work. The fingerprint hotfix additionally passes its
three regression checks locally; server pytest and a live clean snapshot are
required before the resumed collection is treated as valid.

## Preregistered pilot (executed through step 1 only)

1. Collect 60 clean episodes, balanced across four suites. If coverage is
   incomplete, append at most two frozen 40-episode batches (maximum 140).
   **Done; coverage still incomplete.**
2. Freeze 32 successful states: four early and four mid states per suite, each
   from a distinct episode group. **Blocked.**
3. Run direct Smol on W7 failure24 and clean32 (56 rollouts).
   **failure24 only: 0/24; clean32 not run.**
4. Run direct OFT on clean32 (32 rollouts). **Not run.**
5. Export 56 rows and create episode/task-disjoint splits. **Not run.**
6. Train ridge only if readiness passes. Exit 2 is a valid gate outcome and
   must not be bypassed by lowering the 30-train-state minimum. **Exit 2 hit.**

## Decision rules (applied)

- If clean coverage is incomplete after 140 episodes, stop and audit task
  mapping; do not keep sampling adaptively. **Applied.**
- If direct Smol has no clean successes, stop: clean-regret semantics are invalid.
- If ridge does not beat matched-random escalation on task-heldout utility, do
  not add MLP/RL; improve observable features or collect more groups first.
- This is an architecture pilot, not a paper-scale selector result. Paper scale
  remains at least 100 recoverable and 100 unrecoverable episode groups.

## Expected artifacts

- `pool/ngc_w9_clean_controls/` — **written** (138 ep / 2261 states)
- `runs/ngc_w9_clean_control_state_keys.json` — **written** (`coverage_complete: false`)
- `runs/ngc_w9_direct_smol_failure24/summary.json` — **written** (0/24)
- `runs/ngc_w9_direct_smol_clean32/summary.json` — not created
- `runs/ngc_w9_direct_oft_*_clean32/summary.json` — not created
- `runs/ngc_w9_{failure24,clean32}_features.json` — not created
- `runs/ngc_w9_selector_dataset.jsonl` — not created
- `runs/ngc_w9_selector_{episode,task}/readiness_audit.json` — not created
- `runs/ngc_w9_selector_{episode,task}/metrics.json` — not created

## Entry point

```bash
bash scripts/run_w9_clean_selector_pipeline.sh \
  2>&1 | tee runs/ngc_w9_clean_selector_pipeline.log
```

Do **not** re-run this entry point unchanged until the clean-task / success-rate
audit in the coverage-gate record is resolved.
