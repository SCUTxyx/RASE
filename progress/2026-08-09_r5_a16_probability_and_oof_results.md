# R5-A16 probability labels and five-seed OOF result

Date: 2026-08-09

Status: **A16 protocol READY; safe-handback opportunity NOT READY; model gate
0/5 PASS; second VLA/world-model/test remain CLOSED.**

## Frozen identities

- Manifest: `runs/pre_c0_r5/probability_pilot16_manifest_v1.json`, SHA256
  `cdaf205aa0887b94cc73879a92e40b67667fd07ce74085f7f45cba3ed9047279`.
- Corrected collection: `runs/pre_c0_r5/boundary_probability_pilot16_v2`.
- Boundaries: `h={0,16,64,128}`; Student repeats per reachable boundary: 5.
- Cohort: 16 states, 8 true tasks, 4 suites, 2 states/task.
- OOF: 4 fixed task folds, 3 task-bootstrap members, two-boundary dwell,
  `z=1.64485`, five training seeds `20260820..20260824` and frozen split seed
  `20260820`.
- Model size: 348,423 parameters per ensemble member.
- The complete pre-label code/config hashes are recorded in
  `progress/2026-08-09_r5_a16_oof_code_lock.json`.
- Tests: 8/8 passed (probability losses, ordered nonnegative cost quantiles,
  task-fold isolation, and dwell behavior).

The first v1 launch produced no rollout.  Multi-worker validation compared the
complete suite request with each individual shard and falsely called sibling
states missing.  Validation now occurs before shard slicing; v2 kept the exact
same manifest, states, boundaries, repeats and analysis gates.

## A16 protocol audit

- 16/16 manifest states and all 8 tasks/4 suites observed.
- Persistent replay parity: **16/16**.
- 56 reachable boundaries, **280** continuation trials.
- Repeat-field completeness: **100%**; duplicate repeat-seed rows: **0**.
- Eight absent late boundaries were all beyond the exact persistent trajectory's
  terminal step; unexplained missing reachable boundaries: **0**.
- Protocol gate: **READY**.

The collector's legacy model-free opportunity process exits 2 after writing a
complete result because its separate gate is not ready.  This is expected and
must not be conflated with the A16 protocol gate.

## Probability labels

- Nondegenerate boundaries: **13/56 = 23.21%**.
- Mean handback success probability: **18.21%**.
- Mean empirical Bernoulli entropy: **0.212 bits**; maximum **0.971 bits**.
- Maximum one-sided 95% Wilson LCB: **0.649**.
- Historical single-run handback agreement: **42/56 = 75.0%**, further evidence
  that deterministic labels were too brittle.

By suite:

| suite | boundaries | trials | mean p(success) | mean entropy (bits) | nondegenerate |
|---|---:|---:|---:|---:|---:|
| Goal | 14 | 70 | 7.14% | 0.139 | 2 |
| Long | 16 | 80 | 8.75% | 0.182 | 3 |
| Object | 14 | 70 | 38.57% | 0.208 | 3 |
| Spatial | 12 | 60 | 20.00% | 0.342 | 5 |

By boundary:

| h | states | trials | mean p(success) | mean entropy (bits) | nondegenerate |
|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 80 | 21.25% | 0.151 | 3 |
| 16 | 16 | 80 | 17.50% | 0.106 | 2 |
| 64 | 16 | 80 | 12.50% | 0.243 | 4 |
| 128 | 8 | 40 | 25.00% | 0.485 | 4 |

Strict probability decreases occur in 4/16 states (25%) over five adjacent
transitions; four drops are at least 0.4.  However **0/5** drops have a left
Wilson LCB above the right Wilson UCB.  Therefore A16 shows sampled
non-monotonicity, not confidence-separated non-monotonicity.  Only the first
continuation seed is shared across boundaries; repeats 2--5 are boundary-specific,
so this curve comparison is not a paired common-random-number test and must not
be used for a causal non-monotonicity claim.

Descriptive best fixed h128 is 25.0%; the per-state probability oracle is 33.75%
(+8.75pp).  This is not a deployable oracle and repeats are not independent
states.

## Model-free opportunity ceiling

- Persistent success: **16/16**.
- Live finite-safe states: **2/16**, across two tasks.
- Persistent OFT actions: **2,406**.
- Privileged conservative minimum: **2,132**.
- Maximum conservative oracle savings: **11.39%**, below the 20% deployment
  target before any learning error.

The model gate is therefore opportunity-limited on this cohort.  A learned
controller cannot be expected to attain 20% conservative savings when its
privileged upper bound is 11.39%.

## Five-seed OOF

| seed | success gap | expected false handback | OFT savings | handback rate | diagnostic AUC | all gates |
|---:|---:|---:|---:|---:|---:|:---:|
| 20260820 | -8.75pp | 8.75% | 12.39% | 12.50% | 0.752 | FAIL |
| 20260821 | -6.25pp | 6.25% | 9.19% | 6.25% | 0.758 | FAIL |
| 20260822 | -6.25pp | 6.25% | 9.19% | 6.25% | 0.708 | FAIL |
| 20260823 | -6.25pp | 6.25% | 9.19% | 6.25% | 0.663 | FAIL |
| 20260824 | -16.25pp | 16.25% | 27.85% | 18.75% | 0.679 | FAIL |

Frozen gates require success gap >= -5pp, false handback <=5%, savings >=20%,
and at least four passing seeds.  Result: **0/5 PASS**.  The only seed exceeding
20% savings has the worst success/harm result.  Three of sixteen states (18.75%)
change handback decision across seeds; only one state is handed back by all five
seeds.

Task-cluster bootstrap intervals are very wide and include zero savings for all
seeds, as expected from only eight tasks.  Repeat-level AUC is diagnostic only;
its moderate value does not produce a valid state-level Pareto point.

The source-risk h0 target has both outcomes (17 success, 63 failure trials).
The persistent-success target is all-positive (56/56 rows), so its loss was
disabled and the head cannot be claimed.  No deployment checkpoint was written.

## Scientific conclusion

The probabilistic protocol and lightweight implementation are valid, but this
cohort does not support the safe-handback method claim.  The result does **not**
falsify the broad idea of a lightweight, policy-conditioned, multi-VLA risk and
recovery controller.  It falsifies the narrower claim that the current A16
boundary distribution contains enough conservative handback opportunity for
the proposed controller and 20% cost target.

Do not tune thresholds post hoc, add world-model features, run a second VLA,
collect independent validation, unseal test, or start 100+ closed-loop episodes
from this result.

## Next allowed stage

Return to model-free data design before any new model experiment:

1. Freeze an independent, task-balanced development cohort from the 72-state
   train split, without using A16 outcomes for state selection.
2. Include persistent failures and clean/source-success states so every outer
   training partition supports persistent-success and source-risk heads.
3. Run a probability-label opportunity screen first.  Require at least 20
   finite-safe states, at least three tasks, at least two populated stopping
   bins, and a conservative oracle savings margin above the deployment target
   (recommended >=25% for a 20% learned target).
4. Increase repeat support or preregister a fixed two-stage repeat design before
   making claims about non-monotonicity; K=5 is insufficient to separate the A16
   drops.  Reuse the same repeat-seed set at every boundary within a state so
   boundary differences are paired by common random numbers.
5. Only if this model-free gate passes, rerun the frozen multi-head five-seed OOF.
   The second-VLA/shared-vs-per-VLA/zero-shot/leave-one-out matrix remains
   conditional on >=4/5 model seeds passing.
6. World-model residual/disagreement features remain a later, preregistered
   ablation and enter the main method only with a state-level Pareto gain.

## Artifacts

- A16 summary:
  `runs/pre_c0_r5/boundary_probability_pilot16_v2/probabilistic_summary.json`
- A16 collection report:
  `runs/pre_c0_r5/boundary_probability_pilot16_v2/report.json`
- Five-seed gate:
  `runs/pre_c0_r5/probabilistic_oof_a16_v1/seed_stability.json`
- Individual reports:
  `runs/pre_c0_r5/probabilistic_oof_a16_v1/seed_*/report.json`

The key artifact SHA256 values are respectively
`6d3f5084aa54a7ce67d6d97c6a1e280c10d7c56e8eab27a96873f3456766bbdb`,
`a63e58d9c5c4bf822dd378d631dbe006f9d709cb8bdb2cbb539b59eaf9fa89fd`,
and `9ef208ea4e01a02735c5ce747c14fa8d9b496415e028c3b9594f0a91f760e97a`.
