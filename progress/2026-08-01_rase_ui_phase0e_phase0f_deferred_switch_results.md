# RASE-UI Phase 0E deferred-switch calibration and Phase 0F replay result

Date: 2026-08-01 18:20 CST  
Status: **COMPLETED — executable timing heterogeneity found; independent-opportunity gate remains closed**

## Research question and frozen cohort

Phase 0E asks whether an OFT handoff should happen immediately at the saved
decision state or after executing the exact five-action suffix already present
in `decision_context.active_action_suffix`. Both arms restore the same snapshot
and then use the same suite-specific deterministic OFT checkpoint. The cohort is
the frozen Phase 0D set: 16 states, 16 episodes, and 16 tasks, with four states
per LIBERO suite. It is operator-semantics calibration, not independent
confirmation.

Phase 0F replays only the three Phase 0E disagreement states to detect restore,
prefix, or inference-service drift. It deliberately uses the same snapshots and
deterministic policy, so it is an implementation-stability audit rather than a
new statistical sample.

## Contract and parity gate

- `direct_oft`: restore the decision state, discard the active Smol action
  suffix, and query OFT immediately.
- `decision_suffix_oft`: restore the same state, execute the exact remaining
  active suffix, and then query OFT.
- All 16 deferred arms executed exactly five frozen suffix actions.
- All 16 prefix sources equal `decision_context.active_action_suffix` and have
  recorded SHA-256 identities.
- Prefix parity status: **PASS, 16/16**.

This is not the legacy candidate-prefix ablation. Candidate rollouts remain a
separate protocol and are not used as a proxy for strict CONTINUE.

## Phase 0E result

| Result | Count | Rate |
|---|---:|---:|
| immediate OFT success | 10/16 | 62.50% |
| decision-suffix OFT success | 9/16 | 56.25% |
| same-state two-arm oracle | 11/16 | 68.75% |
| oracle minus best fixed | 1/16 | 6.25 pp |

Paired classification:

- both success: 8;
- immediate-only: 2;
- deferred-only: 1;
- neither: 5;
- exact McNemar `p=1.0` (three discordant states; no aggregate superiority
  claim is supported).

The disagreement is structured in this small calibration cohort. Both
immediate-only states are clean L0 states in LIBERO-10, while the deferred-only
state is clean L0 in LIBERO-Spatial. Clean L0 has an 8/8 two-arm oracle but only
7/8 for the best fixed timing. Camera L1 and robot L1 show no timing
complementarity here.

Measured OFT inference remains nearly identical between arms:

| Arm | Predict time/env step | Mean/predict call | Mean rollout wall time |
|---|---:|---:|---:|
| immediate OFT | 16.582 ms | 130.585 ms | 7.037 s |
| decision-suffix OFT | 16.118 ms | 129.989 ms | 6.850 s |

Therefore the outcome difference is attributed to intervention timing and
trajectory state, not to a material OFT inference-latency difference.

## Exact three-operator join

The analysis now joins Phase 0D strict CONTINUE with both Phase 0E OFT arms on
the identical 16 state keys:

| Operator | Success |
|---|---:|
| strict CONTINUE | 6/16 |
| immediate OFT | 10/16 |
| decision-suffix OFT | 9/16 |
| three-operator same-state oracle | 11/16 |

Success patterns `(C, immediate, deferred)` are:

- `(0,0,0)`: 5;
- `(0,1,0)`: 2;
- `(0,1,1)`: 3;
- `(1,0,1)`: 1;
- `(1,1,1)`: 5.

Strict CONTINUE adds no oracle success beyond the two OFT timings. The only
success unique among all three operators is immediate OFT on two LIBERO-10
tasks. The deferred-only state is also solved by strict CONTINUE, but it is
still a genuine immediate-versus-deferred timing disagreement.

## Phase 0F deterministic replay

The disagreement-key exporter froze exactly three states: two immediate-only
LIBERO-10 tasks and one deferred-only LIBERO-Spatial task. The replay reproduced
all three classifications and all three active-suffix SHA-256 values exactly:

- exact outcome and prefix match: **3/3, PASS**;
- replay immediate/deferred classifications: 2 immediate-only, 1 deferred-only;
- no partial or missing state coverage.

This rules out a transient implementation drift in the tested deterministic
configuration. It does not estimate across-seed variance and cannot replace an
episode/task-disjoint screen.

## Scientific decision

The result establishes an executable source of intervention-timing
heterogeneity, but it is not yet enough to train a selector or world model:

1. the oracle gain is only one state on a reused 16-state calibration cohort;
2. the aggregate immediate/deferred difference is not significant;
3. independent tasks have not yet reproduced both timing-specific win types;
4. strict CONTINUE and REPLAN have not shown enough unique success coverage to
   justify a broad multi-operator selector.

The current claim should be limited to: **queue-boundary timing can causally
change deterministic downstream OFT success on selected saved states**. It must
not be reported as a learned-selector improvement or general benchmark gain.

## Code completed

- `rase/collect/prefix_ablation.py`: strict decision-suffix arms, canonical
  action-prefix SHA-256, and paired state summaries.
- `scripts/rollout_oft_prefix_ablation.py`: `--arms decision-suffix`, exact
  active-suffix sourcing, manifest protocol, prefix completion and early
  terminal fields, and OFT timing counters.
- `scripts/analyze_deferred_switch.py`: exact coverage/parity enforcement,
  paired and grouped summaries, timing, McNemar test, and optional exact
  three-operator join with strict CONTINUE.
- `scripts/select_deferred_disagreement_keys.py`: deterministic disagreement
  subset export with the repository-standard compact-JSON key checksum.
- `scripts/audit_deferred_replay.py`: exact outcome and prefix-identity replay
  audit.
- `scripts/run_rase_ui_phase0e_deferred16.sh` and
  `scripts/run_rase_ui_phase0f_disagreement_replay.sh`: safe, suite-serial,
  resumable entrypoints.
- Relevant verification: **20 tests passed**, Ruff passed, Bash syntax passed,
  and `git diff --check` passed.

## Deviation record

The first Phase 0F attempt (`v1`) stopped before rollout because the newly
exported key subset used a newline-based checksum while the rollout contract
requires SHA-256 over a compact JSON array. The failed artifacts were retained
and excluded. The exporter was corrected and tested against the rollout
checksum implementation; the official replay is `v2`.

## Next stage: Phase 0G independent timing-opportunity screen

Do not train a selector or world model next. Freeze a new task- and
episode-disjoint screen before observing outcomes:

1. collect 48 states as `4 suites × 3 cells × 4 states`, using clean L0,
   camera L1, and robot L1, one state per episode and one per task;
2. exclude every task and episode used in Phases 0C–0F;
3. choose the decision step and suffix-validity rule from metadata only;
4. run strict CONTINUE, immediate OFT, and decision-suffix OFT from every state;
5. keep REPLAN out of the primary matrix because it has produced no unique
   success in the existing opportunity cohorts; retain it only as a documented
   secondary baseline if compute permits;
6. report paired classifications, task/episode-cluster bootstrap intervals,
   suite/cell strata, suffix length, terminal-during-prefix cases, and measured
   policy time without post-hoc scalar cost;
7. require both immediate-only and deferred-only wins on at least two distinct
   tasks each and a preregistered success-oracle gap of at least 0.05 before
   opening an independent 96-state confirmation;
8. only after that confirmation should a lightweight public-state timing
   selector be trained with task-grouped splits. A generative world model is
   considered only if a lightweight selector leaves a preregistered predictive
   gap and the benchmark opportunity remains non-trivial.

## Official artifacts and identity

- Phase 0E analysis: `runs/rase_ui_phase0e_deferred16_analysis_v1.json`
- Phase 0E log: `runs/rase_ui_phase0e_deferred16_v1.log`
- Phase 0F frozen keys: `runs/rase_ui_phase0f_disagreement_keys_v2.json`
- Phase 0F analysis: `runs/rase_ui_phase0f_disagreement_analysis_v2.json`
- Phase 0F replay audit: `runs/rase_ui_phase0f_disagreement_replay_audit_v2.json`
- Phase 0F log: `runs/rase_ui_phase0f_disagreement_replay_v2.log`
- Base commit: `454f76384e5195a750584dd9753c29b0701bb6af`

SHA-256:

- Phase 0E analysis: `26c499909245cd7aed42d3aa20e28fefbbb40240b4e212db90e0d26c7750cab9`
- Phase 0F frozen keys: `16133d8694075b9aa1ee136204ea50cfccc0ec1b14143937659d71e4b51b2a02`
- Phase 0F analysis: `cb626603ff26bea501b52fb90d9d4963674e70ee330ae576e7f1bc2c80d73b53`
- Phase 0F replay audit: `053c3c5fe6c3324ceaa261648972bf95d1b2c8bc12dec9de11a3372c65dc167e`

Official commands in tmux 0:

```bash
cd /root/autodl-tmp/RASE
FRESH_RUN=1 TAG=v1 ./scripts/run_rase_ui_phase0e_deferred16.sh
FRESH_RUN=1 TAG=v2 ./scripts/run_rase_ui_phase0f_disagreement_replay.sh
```
