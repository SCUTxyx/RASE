# W9C benchmark reanalysis record

**Date opened/completed:** 2026-07-31  
**Status:** **COMPLETED (CPU-only, read-only reanalysis)**  
**Decision boundary:** descriptive benchmark analysis only; W9C `kill_method_branch` remains in force.

## Executed commands

From `/root/autodl-tmp/RASE`, using the script defaults (`success_reward=1.0`,
`bootstrap_seed=20260731`, `bootstrap_samples=5000`, preregistered shortcut fallback
`continue_smol`):

```bash
python scripts/analyze_selector_benchmark.py \
  --dataset runs/ngc_w9c_selector_dataset.jsonl \
  --splits runs/ngc_w9c_selector_episode_splits.json \
  --output runs/ngc_w9c_selector_episode_benchmark_reanalysis.json

python scripts/analyze_selector_benchmark.py \
  --dataset runs/ngc_w9c_selector_dataset.jsonl \
  --splits runs/ngc_w9c_selector_task_splits.json \
  --output runs/ngc_w9c_selector_task_benchmark_reanalysis.json
```

No ridge training, simulator, rollout, GPU, or outcome mutation was used.

## Inputs, outputs, and integrity

Inputs:

- `runs/ngc_w9c_selector_dataset.jsonl` — SHA-256
  `74209425f6a7d5047897809d16adbe969bd1701175a6c8c85c557bbf9d534f6f`
- `runs/ngc_w9c_selector_episode_splits.json` — SHA-256
  `3f979680f4f15a6c290d598ea106bc982961a1e1068e51dcf1a4d914a1123573`
- `runs/ngc_w9c_selector_task_splits.json` — SHA-256
  `fc09bd82fb4d30c256b2aa6e2e10fde4ded653af6b167bab8bab028d75118c8c`

Outputs:

- `runs/ngc_w9c_selector_episode_benchmark_reanalysis.json` — SHA-256
  `629aa4b37a6846d1eef306bad3627ff4b005d718e2f2a8cf769bbe63ffabce33`
- `runs/ngc_w9c_selector_task_benchmark_reanalysis.json` — SHA-256
  `908a6d03870a175df0a3a07c9d5f98c4cee8d6cc6506b975119ac10c9adddb5b`

Both outputs use `rase-selector-benchmark-analysis/v1`, contain 56 states, and report
56 oracle-evaluable / 0 non-evaluable states. Oracle action support is
`continue_smol=16`, `escalate_oft=12`, `abstain=28`.

## Overall oracle-minus-fixed utility gaps

These full-dataset paired gaps are identical in the episode/task outputs because the
fixed-policy comparison uses the same 56 frozen states; the split files affect the
composition and train-only diagnostics, not this overall denominator.

| Fixed policy | Mean oracle − fixed gap | 95% paired bootstrap CI | n |
|---|---:|---:|---:|
| Always abstain | 0.472857 | [0.353214, 0.596786] | 56 |
| Always continue Smol | 0.207143 | [0.110714, 0.307054] | 56 |
| Always escalate OFT | 0.090714 | [0.064991, 0.133571] | 56 |

Each CI uses 5,000 paired bootstrap samples. The deterministic seeds are 20260733,
20260731, and 20260732 respectively. These gaps quantify oracle headroom under the
frozen utility, not achievable learned-selector performance.

## Composition warnings

### Episode-disjoint artifact

The composition audit correctly warns that five task IDs occur across splits:
`libero_10_000007` and `libero_10_000709` (test/train),
`libero_goal_000010` (train/val), `libero_object_000008` (test/train), and
`libero_spatial_000004` (test/val). This is allowed for an episode-disjoint split but
means it is not task-disjoint.

It also reports learned actions as **unavailable** for test, train, and val because
`ngc_w9c_selector_dataset.jsonl` has no `learned_action` annotations.

### Task-disjoint artifact

The test split is `clean_control=8, failure_challenge=0`; val is likewise
`clean_control=9, failure_challenge=0`. Therefore test/val cannot evaluate unseen-task
failure detection or failure routing. Learned actions are again **unavailable** in
this analysis JSON for all three splits.

Do not infer learned action counts from either reanalysis JSON. The formal task-test
ridge counts `7 continue / 0 escalate / 1 abstain` come from the already-existing
selector metrics (`runs/ngc_w9c_selector_task/metrics.json` and gate summary), not
from this CPU reanalysis. “No learned-action annotation” and “formal learned 7/0/1”
are different facts and must not be merged.

## Train-only suite shortcut diagnostic

This shortcut is a descriptive, oracle-label majority lookup fitted only on each
artifact's train split. It is not ridge, is not a learned model result, and does not
alter the W9C gate.

### Episode split

Train (`n=40`) selects suite actions: Goal→escalate, Long→continue,
Object→abstain, Spatial→abstain; the train-global fallback is abstain. No evaluation
suite is unseen.

- Test (`n=8`): actions A/C/E = 2/4/2; oracle-action accuracy 0.125; utility 0.0900;
  oracle utility 0.5825; gap 0.4925.
- Val (`n=8`): actions A/C/E = 3/3/2; accuracy 0.125; utility -0.0325;
  oracle utility 0.3575; gap 0.3900.
- Train: accuracy 0.500; utility 0.2395; oracle gap 0.2345.

### Task split

Train (`n=39`) selects Goal→abstain, Long→escalate, Object→continue,
Spatial→continue; the train-global fallback is abstain. No evaluation suite is
unseen.

- Test (`n=8`, all clean): actions A/C/E = 0/5/3; oracle-action accuracy 0.250;
  utility 0.3250; oracle utility 0.4800; gap 0.1550.
- Val (`n=9`, all clean): actions A/C/E = 5/3/1; accuracy 0.444; utility 0.0933;
  oracle utility 0.3178; gap 0.2244.
- Train: accuracy 0.462; utility 0.3415; oracle gap 0.1656.

The task-test shortcut's 0/5/3 counts are shortcut actions, **not** the formal ridge
7/0/1 counts. Its clean-only composition also prevents a failure-routing claim.

## Leave-one-suite-out descriptive folds

Both artifacts contain the same four LOSO descriptive folds, built directly from
the same complete 56-state dataset rather than from the supplied episode/task split.
For every held-out suite, the suite-majority shortcut has no mapping for that suite;
its train-global fallback is abstain, so it predicts abstain for every held-out state.

| Held-out suite | n test | Oracle A/C/E | Shortcut accuracy | Shortcut utility | Oracle − shortcut gap |
|---|---:|---:|---:|---:|---:|
| Goal | 16 | 8/3/5 | 0.500 | 0 | 0.4650 |
| Long | 18 | 7/4/7 | 0.389 | 0 | 0.5678 |
| Object | 10 | 6/4/0 | 0.600 | 0 | 0.3920 |
| Spatial | 12 | 7/5/0 | 0.583 | 0 | 0.4083 |

These folds are **descriptive support diagnostics only**: there are only four suites,
suite identity is perfectly unseen by construction, fallback behavior dominates the
shortcut, cohort/task composition is not balanced across held-out suites, and no
ridge was fitted or evaluated in LOSO. They show that a suite-identity lookup does
not transfer; they do not estimate learned-selector cross-suite generalization.

## Completed scope and remaining limitations

Completed here: full zero-cell composition tables, direct-action summaries, oracle
action support, overall oracle-minus-fixed paired gaps, split warnings, train-only
suite shortcut, and LOSO descriptive folds. Prefix/zero-prefix mechanism accounting
remains in the existing W7 ablation records rather than being regenerated by these
direct-action JSONs.

The analysis remains conditioned on the frozen W9C 32 clean + 24 failure-challenge
states, current policies/costs, sparse suite cells, and existing phase/backbone
coverage. It adds benchmark diagnosis but cannot supply missing task-held-out failure
support.

## Decision

The reanalysis does **not** revoke or weaken W9C `kill_method_branch`. It provides no
learned-action annotations, does not show that ridge beats matched-random, and does
not justify MLP/RL. Paper posture remains benchmark / diagnosis; any new method
experiment requires a separately dated preregistration and new evidence.
