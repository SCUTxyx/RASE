# RASE Phase 1A replacement-audit pilot kickoff

Date: 2026-08-02 21:45 CST  
Status: **READY TO RUN — development-only 48-task replacement audit**

## Scale-up plan feasibility decision

The 2026-08-01 CVPR 2027 flagship plan is scientifically coherent and aligned
with the frozen Phase 0B–0H evidence. It correctly closes timing/operator
selector and ungated world-model training, then moves to same-state benchmarking,
replacement audit, one gated visual fixed-intervention method, cross-platform
evaluation, and real-robot validation.

It is conditionally feasible as a multi-month project, not as a single next
experiment. The 60k–80k simulation rollout, three-platform, multi-policy, hidden
test, and 240–384 real-robot targets require separate owners and throughput
measurements. The first executable gate is therefore the plan's own Phase 1A
replacement audit. No flagship-scale data generation is authorized until the
recovery framing survives this gate.

## Preregistered question

On the 48 metadata-frozen Phase 0G development tasks and identical initial
conditions, does a full-horizon source policy retain terminal-success
complementarity relative to OFT from the episode reset, and does running source
for 25 environment steps before switching improve over OFT-only?

This audit addresses the reviewer question: why not always run OFT?

## Fair episode modes

1. `SOURCE-ONLY`: SmolVLA from reset to the environment's full horizon.
2. `OFT-ONLY`: suite-specific frozen OFT from a pre-action reset snapshot.
3. `SOURCE→OFT`: the existing Phase 0G immediate OFT arm after source reaches
   the frozen env-step-25 decision state.

The previous Phase 0G source collection was capped at 80 environment steps and
is forbidden as a SOURCE-ONLY baseline. Its step-2 snapshot occurs at env step
25 and is forbidden as an OFT-only reset state. Phase 1A recollects full source
episodes with `capture_decision_context_v2=false`; chunk-0 snapshots must be at
policy step 0 before any source action. The post-run correction below freezes
the expected post-reset simulator counter. Any mismatch stops the experiment
before OFT evaluation.

## Split and claim boundary

- 48 unique tasks and episodes, four suites × three cells × four tasks.
- Cells: clean L0, camera L1, robot L1.
- Sampling seed and design identity: Phase 0G metadata-only seed `2026081807`.
- Phase 0 task reuse is explicitly allowed only for method development/prior
  evidence under the flagship plan.
- These tasks are excluded from train/validation/public-test/hidden-test claims
  in the eventual flagship benchmark.
- One deterministic episode per policy mode; the independent unit is task and
  source episode, not snapshots or arms.

## Frozen primary analysis

- Terminal success only; no short-horizon proxy.
- SOURCE/OFT quadrants: Rescue, Harm, Redundant, Unsupported.
- Paired McNemar test.
- Task/episode bootstrap 95% interval with 10,000 replicates and seed
  `2026080201`.
- Overall, suite, and perturbation-cell reporting.
- Full-episode source and OFT inference time/env-step plus wall-clock reporting.
- Historical source→OFT cost is explicitly incomplete because the stored Phase
  0G artifact contains only the post-step-25 continuation cost.

Pilot gate:

- `recovery_framing_signal` only if SOURCE-ONLY contributes at least two OFT-only
  unique wins across at least two suites;
- `replacement_risk_high_cost_audit_required` if OFT-only weakly dominates
  source overall and on clean tasks, source has zero unique wins, and
  source→OFT does not exceed OFT-only;
- otherwise the pilot is inconclusive and requires scaling or a policy-pair
  change.

The pilot cannot make the final resource-complementarity claim because the
source-prefix cost of historical source→OFT is unavailable.

## Code completed before outcome access

- `collect_rase_ui_phase1a_replacement48_source.json`: full-horizon source and
  pre-action snapshot protocol.
- `pipeline.py` and `lerobot_libero_plus_adapter.py`: per-episode source wall,
  policy-call, and env-step timing.
- `export_initial_replacement_keys.py`: exact design join, reset-timestep audit,
  and outcome-independent key freeze.
- `analyze_replacement_audit.py`: exact three-mode join, quadrants, paired tests,
  clustered bootstrap, timing, and replacement gate.
- `run_rase_ui_phase1a_replacement48.sh`: resumable suite-serial entrypoint.

Pre-run validation: 17 focused tests passed; Ruff, shell syntax, and
`git diff --check` passed.

## Identity

- Base Git commit: `454f76384e5195a750584dd9753c29b0701bb6af`.
- SmolVLA/OFT checkpoints remain frozen; no parameter training.
- Base `env.lock.md` SHA-256:
  `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2`.

## Prohibited changes

- no timing-selector reopening;
- no world-model training;
- no Phase 0 task relabeling as flagship test;
- no post-outcome task deletion;
- no shortened SOURCE-ONLY horizon;
- no using source→OFT continuation-only timing as full episode cost;
- no hidden-test access.

## Post-run reset-semantics correction (2026-08-02)

The kickoff initially assumed that a reset state must have simulator
`env_counters.timestep == 0`. LIBERO performs 10 internal robot-initialization
steps inside `reset()`, before either policy acts. The audited reset contract is
therefore policy step 0, zero source actions, and post-reset simulator timestep
10. All 48 frozen states satisfy that contract. This correction changes no
task, outcome, or selection rule; it prevents the simulator's internal reset
steps from being misreported as a source-policy prefix.
