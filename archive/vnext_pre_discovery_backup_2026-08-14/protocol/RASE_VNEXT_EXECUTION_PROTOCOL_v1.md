# RASE vNext execution protocol v1

Status: **preregistered draft; activation locked**  
Activation prerequisite: a contract-valid `r10b_chunk_input_divergence_audit_v1.json`.

This file freezes the execution order and data contracts. It does not claim that
G0, discovery, opportunity, information, or closed-loop gates have passed.

## Strict order and stop rules

1. Complete the 54/54 R10-B per-chunk diagnostic.
2. Produce the full `group_id × {t8,t16}` root-cause matrix. Mixed A/B/C causes
   remain mixed; D never overrides a reproduced A/B/C cause.
3. Freeze T1/T2/T3, cost units, metrics, and kill criteria.
4. Freeze the model-free opportunity and collection protocol.
5. Pass the minimal lossless LIBERO canonical-interface parity tests.
6. Run fixed-K discovery, then feasibility, then freeze the confirmation cohort.
7. Run the model-free opportunity gate on confirmation data.
8. Only after opportunity PASS, run target-specific low-capacity information gates.
9. Only after information PASS, run task-held-out OOF and an independent closed-loop gate.
10. Expand one axis at a time: third VLA, second benchmark, public-data ablation,
    then (only if justified) RL.

Any failed gate stops its descendants. It never authorizes a larger model, target
search on held-out outcomes, replacement of unstable roots, validation/test use,
or deletion of failed trials.

## Targets

- **T1 source risk:** probability/count of a prespecified short-horizon adverse
  source outcome under `continue.source`.
- **T2 intervention urgency:** utility loss from delaying intervention over a
  prespecified horizon.
- **T3 correction value:** per-operator utility (and advantage versus best fixed)
  with all costs explicit.

Task identity is leakage-diagnostic metadata only. Deployable inputs may include
task text, elapsed time, policy descriptor, causal observation history, and causal
source-action history. Future frames, future teacher actions/costs, simulator
object state, selection labels, and outcome-derived sampling decisions are forbidden.

## Utility and primary reporting

The composite is

`U = success - lambda_h*harm - lambda_q*query_cost - lambda_f*fallback_cost - lambda_l*latency_cost`.

Every report must also expose success, paired harm, query count/cost, fallback
steps/cost, latency, and the success-compute Pareto frontier separately. Composite
weights and the meaningful minimum effect are deliberately not guessed here: all
are required numeric fields in the machine-readable configuration before it can
be changed from `draft_locked` to `frozen`. Sensitivity specifications are frozen
with the same manifest, never selected after observing confirmation outcomes.

## Operators and candidates

The five semantic operator priors are exactly:

- `continue.source`
- `requery.source`
- `resample.source`
- `fallback.persistent`
- `abort.safe`

Multiple replan/resample candidates belong to the candidate verifier and retain
the same semantic operator ID. They cannot inflate the G-O2 operator count.
Unavailable operators are represented by an explicit policy/benchmark mask, not
by silently dropping rows.

## Collection contract

- Discovery uses fixed `K=3` for every scheduled branch.
- Confirmation uses fixed `K=5` for every scheduled branch.
- No outcome-dependent K, early stopping, root replacement, or preferred-repeat
  selection is allowed.
- Randomness is keyed separately by `init_state_id`, `environment_seed`,
  `source_sampling_seed`, `operator_seed`, and `exact_repeat_replica`.
- Discovery sampling is deterministic and outcome-independent: two tasks per
  suite, two independent roots per task, both source policies, and two
  preregistered causal decision points.
- Decision points are defined from source-time rules before branch outcomes are
  observed. Restore-state references are bookkeeping only and are never features.
- All attempts, failures, masks, raw outcomes, costs, hashes, and seed ledgers are
  retained. A missing scheduled branch is a contract error.

## Gates

### G0 root-cause diagnostic

Requires all 18 groups × 3 replicas, both t8 and t16 traces, a hash-bound manifest,
and the complete cell matrix. A/B/C/D are assigned per cell and may coexist across
pairs. G0 completion only unlocks protocol finalization; it does not unlock training.

### Discovery feasibility

Checks interface/parity, complete scheduled branches, operator-mask coverage, label
density, and non-degenerate raw outcomes. Its numeric criteria are frozen in the
machine-readable config before collection. Passing only permits freezing the
48-task confirmation cohort.

### Model-free opportunity (confirmation only)

- **G-O1:** task/root-bootstrap lower confidence bound for oracle minus best-fixed
  utility is above the preregistered meaningful effect.
- **G-O2:** at least two semantic operators are uniquely/co-best on at least 10%
  of independent roots each, cover the frozen minimum task count, and cover at
  least three suites. Decision points and replicas do not count as independent roots.
- **G-O3:** G-O1 opportunity is present separately for Pi0.5 and Pi0Fast.

The bootstrap unit is task with roots nested inside task. Confirmation outcomes
cannot change thresholds, cohort, operator definitions, weights, or K.

### Information and closed-loop gates

T1/T2 report AUROC, AP gain, Brier/log loss, ECE, and task-bootstrap lower bounds.
Probability/count variants use proper count likelihood or beta-binomial deviance.
T3 reports oracle regret, pairwise ranking, top-1 utility, and coverage. Task-held-out
folds and calibration are mandatory. The later independent closed-loop gate uses
paired-harm 95% UCB, success-gap CI, cost-saving CI, and a frozen avoidable-failure
denominator. Training seeds and rollout seeds are disjoint.

## Locked until their parent gates pass

Risk/selector/world-model training, threshold sweeps, offline RL, handback tuning,
validation/test evaluation, third-VLA or new-benchmark rollouts, public-data download,
and real CALVIN/SimplerEnv adapters remain locked.
