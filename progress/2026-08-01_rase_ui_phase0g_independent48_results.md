# RASE-UI Phase 0G independent48 timing-opportunity result

Date: 2026-08-01 20:20 CST  
Status: **COMPLETED — bidirectional timing heterogeneity replicated; preregistered opportunity gate NOT READY**

## Frozen design and coverage

Phase 0G is a task- and episode-disjoint screen of immediate versus
active-suffix-preserving OFT timing. The design was frozen from catalog metadata
before collection and without intervention outcomes.

- 48 episodes, 48 selected states, 48 unique tasks.
- Four suites × three cells × four tasks.
- Cells: clean L0, camera L1, robot L1.
- Zero task or episode overlap with the Phase 0C–0F cohort.
- Source-policy episode outcomes: 48 failures; no state was removed on this
  basis.
- Pool: 192 states, four per episode at steps 0/2/4/6.
- Selected cohort: exactly one step-2 state per task and episode.
- Strict active suffix: 48/48 valid, all length five.
- Frozen design identity and state-key checksum: PASS.

## Primary two-timing result

| Result | Count | Rate |
|---|---:|---:|
| immediate OFT | 37/48 | 77.08% |
| decision-suffix OFT | 35/48 | 72.92% |
| same-state timing oracle | 39/48 | 81.25% |
| oracle minus best fixed | 2/48 | 4.17 pp |

Paired classification:

- both success: 33;
- immediate-only: 4;
- deferred-only: 2;
- neither: 9;
- exact McNemar `p=0.6875`.

The four immediate-only successes occur on four unique tasks; the two
deferred-only successes occur on two unique tasks. Thus the Phase 0E timing
heterogeneity replicates bidirectionally on disjoint tasks, but the smaller side
contributes only `2/48` oracle headroom.

The task/episode bootstrap 95% interval for the oracle gap is `[0, 0.08333]`
with 10,000 replicates and frozen seed `2026081807`.

## Preregistered gate

Requirements:

- immediate-only wins on at least two tasks: **PASS, 4**;
- deferred-only wins on at least two tasks: **PASS, 2**;
- oracle gap at least 0.05: **FAIL, 0.04167**;
- identity, coverage, and prefix parity: **PASS**.

Overall status: **NOT READY**. At `n=48`, the next attainable gap above the
threshold is `3/48 = 0.0625`; the observed result is not rounded upward and the
threshold is not changed after outcome access. The result does not authorize a
96-state confirmation or model training.

## Exact three-operator join

| Operator | Success |
|---|---:|
| strict CONTINUE | 11/48 |
| immediate OFT | 37/48 |
| decision-suffix OFT | 35/48 |
| same-state three-operator oracle | 39/48 |

Success patterns `(CONTINUE, immediate, deferred)`:

- `(0,0,0)`: 9;
- `(0,0,1)`: 2;
- `(0,1,0)`: 4;
- `(0,1,1)`: 22;
- `(1,1,1)`: 11.

Strict CONTINUE contributes no unique success and all 11 CONTINUE successes are
covered by both OFT timings. The six timing-specific OFT states are also the
only successes unique within the three-operator set.

## Stratified diagnostics

By suite:

- Goal: both 7, immediate-only 2, deferred-only 2, neither 1; oracle gap
  `2/12 = 16.67 pp`.
- Spatial: both 9, immediate-only 1, deferred-only 0, neither 2.
- Object: both 8, no timing-only wins, neither 4.
- Long: both 9, immediate-only 1, deferred-only 0, neither 2.

By perturbation cell:

- clean L0: immediate 16/16, deferred 15/16; no oracle gain over immediate.
- camera L1: immediate 11/16, deferred 9/16; no oracle gain over immediate.
- robot L1: immediate 10/16, deferred 11/16, oracle 12/16; oracle gap
  `1/16 = 6.25 pp`, with both deferred-only tasks located in this cell.

The Goal and robot-L1 concentration is a diagnostic localization, not a new
post-hoc confirmation claim.

## Timing

| Arm | Policy time/env step | Mean/predict call | Mean rollout wall time |
|---|---:|---:|---:|
| immediate OFT | 16.143 ms | 126.761 ms | 5.549 s |
| decision-suffix OFT | 15.878 ms | 128.996 ms | 5.725 s |
| strict CONTINUE | — | — | 8.267 s action-selection/trial |

The OFT inference rates remain practically matched; the outcome differences
are trajectory/timing effects rather than an inference-speed artifact.

## Code completed

- `configs/collect_rase_ui_phase0g_independent48.json`: frozen 48-state design.
- `freeze_independent_factorial_design.py`: metadata-only task resolution,
  task/episode overlap exclusion, cell counts, and design identity.
- `audit_independent_keys.py`: exact post-collection design match, step/suffix
  constraints, unique task/episode enforcement, and key checksum audit.
- `audit_timing_opportunity.py`: preregistered bidirectional task-support and
  oracle-gap gate with task/episode bootstrap.
- `rollout_smol_interventions.py`: new `--profile continue-only`, v2 summary,
  and manifest identity so primary screens need not spend compute on REPLAN.
- `analyze_deferred_switch.py`: accepts exact v1 paired or v2 CONTINUE-only
  summaries for the three-operator join.
- `run_rase_ui_phase0g_independent48.sh`: resumable end-to-end entrypoint.

## Deviation record

The official `v1` runner began the legacy paired Smol profile and completed one
CONTINUE arm before the unnecessary REPLAN workload was identified. It was
stopped precisely; the partial `v1` Smol directory is retained and excluded.
The runner was extended and tested with the manifest-bound `continue-only`
profile. Official intervention results and analysis are `v2`; the already
audited pool and frozen keys were reused without recollection.

## Scientific decision and next step

Do not lower the gate, do not start the 96-state confirmation, and do not train
a selector or world model. Phase 0G supports a limited causal claim: queue
boundary timing changes deterministic downstream OFT success on a small but
bidirectional set of independent tasks. It does not establish enough aggregate
headroom for a benchmark-level learned selector.

The next allowed stage is an exploratory Phase 0H mechanism audit on the six
frozen disagreement states:

1. execute the exact active suffix prefixes of length `k=0..5` before OFT;
2. preserve the existing `k=0` immediate and `k=5` deferred endpoints as
   identity checks;
3. report per-state success transition shape and physical trajectory changes;
4. make no population or selector-performance claim from the selected subset;
5. only design a new independent targeted screen if a reproducible mechanism,
   fixed without outcome-conditioned threshold tuning, is identified.

If the partial-prefix curves are unstable or task-specific, close the timing
selector direction and use immediate OFT as the fixed deployment baseline.

## Official artifacts and identity

- Design: `runs/rase_ui_phase0g_independent48_design.json`
- Pool: `runs/rase_ui_phase0g_independent48_pool/`
- Keys: `runs/rase_ui_phase0g_independent48_keys.json`
- Key audit: `runs/rase_ui_phase0g_independent48_keys_audit.json`
- Official CONTINUE: `runs/rase_ui_phase0g_independent48_smol_v2/`
- OFT suites: `runs/rase_ui_phase0g_independent48_oft_{spatial,object,goal,10}_v2/`
- Analysis: `runs/rase_ui_phase0g_independent48_analysis_v2.json`
- Opportunity audit: `runs/rase_ui_phase0g_independent48_opportunity_v2.json`
- Official log: `runs/rase_ui_phase0g_independent48_v2.log`
- Excluded partial: `runs/rase_ui_phase0g_independent48_smol_v1/`
- Base commit: `454f76384e5195a750584dd9753c29b0701bb6af`

SHA-256:

- design: `601deba8d572488be638e1a88128c0774ed846a2a5457c7b536b9c221ca79a24`
- keys: `1dc1b1f4500904cd3a766bef119042e488d9a12670b0906a68032e459a6ff2b2`
- key audit: `045e1c7af333292282211eaf20250449aee2b36cd2e958ab9d7358903c33ac45`
- CONTINUE summary: `d5e1ce6d271fa0e9e6abb71d8ba0a348a9cbb8f4e668a2683afa030f6df87e00`
- analysis: `91f3528c5d291d4d051569fb34ec65047718dbd10c9437bda7be2f50994dabdb`
- opportunity audit: `e7963a744bcd3b0d9bb06ee096300853e5b9d0f8e0142135e86d4d11558fb4c3`

Official resume command in tmux 0:

```bash
cd /root/autodl-tmp/RASE
FRESH_RUN=0 TAG=v2 ./scripts/run_rase_ui_phase0g_independent48.sh
```
