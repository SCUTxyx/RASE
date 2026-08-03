# RASE-UI Phase 0H suffix-prefix mechanism audit result

Date: 2026-08-01 22:17 CST  
Status: **COMPLETED — endpoint parity PASS; shared timing mechanism FAIL; close timing selector**

## Scope and frozen cohort

Phase 0H is an explicitly outcome-selected mechanism audit of the six Phase 0G
timing disagreements: four immediate-only and two deferred-only states. It is
not an independent sample, so all aggregate rates below are descriptive and
cannot be interpreted as population performance or selector accuracy.

- Six unique Phase 0G states, tasks, and episodes.
- Suites: Spatial 1, Goal 4, Long 1; no Object state.
- Cells: clean L0 1, camera L1 2, robot L1 3.
- Exact active suffix length: five actions for every state.
- Interventions: execute exact suffix prefix `k=0..5`, then switch to the same
  frozen suite-specific OFT checkpoint.
- 36/36 scheduled rollouts completed.

Frozen cohort state-key checksum:
`5b1b4508dd155b0f7aa08fb9510e2642d9bcb4cf3ec9e4a2809439d8f48f6fcb`.

## Endpoint identity gate

For every state, Phase 0H `k=0` reproduced the Phase 0G immediate success and
empty-prefix SHA-256, while `k=5` reproduced the Phase 0G deferred success and
full active-suffix SHA-256.

Endpoint success plus prefix-SHA parity: **6/6 PASS**. There were no terminal
successes during an unfinished prefix. Interior curves are therefore valid to
interpret as deterministic same-snapshot causal interventions.

## Per-state success curves

Each pattern is ordered `k=0,1,2,3,4,5`.

| suite | cell | Phase 0G class | pattern | flip steps | curve |
|---|---|---|---|---|---|
| Long | clean L0 | direct-only | `111110` | 5 | single direct harm |
| Goal | robot L1 | direct-only | `100000` | 1 | single direct harm |
| Goal | robot L1 | deferred-only | `011111` | 1 | single deferred rescue |
| Goal | robot L1 | deferred-only | `001011` | 2, 3, 4 | non-monotonic |
| Goal | camera L1 | direct-only | `111100` | 4 | single direct harm |
| Spatial | camera L1 | direct-only | `100100` | 1, 3, 4 | non-monotonic |

Classification:

- single-transition direct harm: 3/6;
- single-transition deferred rescue: 1/6;
- non-monotonic, three transitions: 2/6;
- all six patterns are different.

The descriptive success counts by prefix length are `4,3,4,4,3,2` of six for
`k=0..5`. Because the cohort was selected for opposite endpoint outcomes, these
counts must not be reported as a timing policy comparison.

## Mechanism audit

The four single-transition boundaries occur at `k={1,1,4,5}`, not at one shared
queue position. Their cumulative translation magnitudes at the transition are
approximately `{0.48, 0.73, 2.09, 2.52}`, also not a common threshold. Rotation
and gripper-magnitude summaries likewise vary. All suffixes retain a similar
closed-gripper action sign, including both rescue and harm directions, so a
gripper-phase rule does not separate them.

Two deterministic states alternate success/failure three times as adjacent
actions are added. This rules out a monotonic scalar wait-time mechanism on the
frozen disagreement cohort. The evidence is more consistent with task-specific
trajectory basin changes: small action-prefix changes can place the OFT
continuation in different downstream attractors, but the direction is not
shared across tasks.

Shared scalar-boundary audit: **FAIL**.

## Scientific decision

Apply the kickoff stop rule:

1. close the timing-selector direction;
2. do not launch a post-hoc targeted independent timing screen;
3. do not train a timing selector, outcome model, or world model;
4. use immediate OFT as the fixed deployment intervention, consistent with its
   Phase 0G 37/48 result versus deferred 35/48 and its matched inference cost;
5. preserve Phase 0H as causal mechanism evidence that intervention timing can
   matter locally without implying exploitable aggregate selector headroom.

The next project stage should leave timing selection and freeze the paper-level
benchmark protocol around fixed immediate OFT: independent evaluation across
suites and perturbation cells, uncertainty and cost reporting, failure taxonomy,
and comparison with CONTINUE. Any future learned component requires a new gate
defined from an independent benchmark question, not reuse of these six labels.

## Code and verification

- `freeze_timing_disagreement_keys.py`: source-checksummed cohort freeze with
  explicit outcome-conditioned disclosure.
- `prefix_ablation.py`: exact `k=0..T` suffix prefix construction and validation.
- `rollout_oft_prefix_ablation.py`: resumable `suffix-prefix-grid` mode and
  action-geometry descriptors.
- `analyze_suffix_prefix_mechanism.py`: exact state union, endpoint success/SHA
  parity, transition classification, shared-boundary audit, and decision record.
- `run_oft_verify_suites.sh`: supports the new grid mode.
- `run_rase_ui_phase0h_suffix_prefix6.sh`: end-to-end fresh/resume entrypoint.

Relevant tests: 25/25 passed before launch; final focused tests 18/18 and 5/5
passed after reporting fixes. Ruff, bash syntax, and `git diff --check` passed.
The only runtime deviation was a display-name/canonical-suite mismatch in the
first analysis call after all 36 rollouts completed. It was fixed and the
analysis alone was rerun; no rollout or frozen outcome was changed.

## Artifacts and identity

- Cohort: `runs/rase_ui_phase0h_disagreement6_keys_v1.json`.
- Suite summaries: `runs/rase_ui_phase0h_suffix_prefix6_{spatial,goal,10}_v1/summary.json`.
- Analysis: `runs/rase_ui_phase0h_suffix_prefix6_analysis_v1.json`.
- Human report: `runs/rase_ui_phase0h_suffix_prefix6_analysis_v1.md`.
- Log: `runs/rase_ui_phase0h_suffix_prefix6_v1.log`.
- Base commit: `454f76384e5195a750584dd9753c29b0701bb6af`.

SHA-256:

- cohort artifact: `dab650d599fc2524300556461cb6068f12801bd62373c5166824b6241dbe9e61`;
- Spatial summary: `a2d277745a7c8da253cafa17a6ea6dfb11daa18ba19ca37112d57d3c1c17955c`;
- Goal summary: `f02c568ca8a5a4a511cfcca82fe29e2b0fc80c7b8a97786d7c5498c14cfa954f`;
- Long summary: `3f654e5a393a6766c6376bf7211c763219e2f6fac4f5c7b2430030f52b54778b`;
- final analysis: `4455f6a72566e2e783440095bdda8d57a995832cd6019af76f1a0720af0487f8`;
- final Markdown analysis: `e66edffe5f2de740f461dd7851a0ad737a4536cb91fc4728b391184343b9ad65`.

Suite rollout wall time excluding model startup was 61.794 s Spatial, 183.222 s
Goal, and 48.536 s Long. tmux 0 returned to a shell prompt and the GPU was idle
after completion.
