# RASE current status

Last updated: 2026-08-13  
Current stage: **R10-B reproducibility FAIL; per-chunk input/output root-cause diagnostic running; all learned stages remain locked**

## 2026-08-13 CVPR 2027 canonical document refresh

- Canonical idea:
  `outputs/RASE_IDEA_DETAILED_CVPR2027.md`.
- Canonical execution plan:
  `plan/RASE_NEXT_STEPS_EXECUTION_PLAN_CVPR2027.md`.
- RL and public-data plan:
  `plan/RASE_RL_AND_PUBLIC_DATASETS_DETAILED_PLAN.md`.
- Server-state execution overlay:
  `plan/RASE_CURRENT_EXECUTION_OVERLAY_2026-08-13.md`.
- These documents supersede the old root idea/next-step documents and the old
  RL selector guide as current guidance. Historical progress records and
  frozen experiment decisions are not superseded.

## 2026-08-13 R10-B chunk-input diagnostic update

- The completed 54-trajectory full-action-trace audit found closed-loop trace
  divergence: `CLOSED_LOOP_TRACE_DIVERGENCE`, with the formal next decision
  `AUDIT_CHUNK_INPUT_DIVERGENCE`.
- Hash-only per-chunk instrumentation records each OFT query's two image input
  hashes, proprio hash and returned action-chunk hash/shape. Seven focused
  tests pass. A real one-group/K=3 smoke has identical first-query inputs and
  outputs at t8/t16 and complete non-empty query chains.
- The full frozen 18-group/K=3 diagnostic is running in tmux
  `r10b_chunk_full`; it writes only new diagnostic artifacts and automatically
  runs `scripts/audit_r10b_chunk_input_divergence.py` after collection.
- This diagnostic does not unlock model training. Its result selects the next
  root-cause branch described in the execution overlay.

## 2026-08-13 canonical idea rewrite

- The canonical RASE idea is now a lightweight stochastic risk-control layer
  for multiple frozen VLAs: shared causal history/action risk core,
  outcome-free behavior descriptor, small policy/fallback calibration,
  probabilistic fallback-success and recoverability-window survival heads, and
  conservative source/fallback switching with abstention.
- The rewrite preserves the original multi-VLA real-time risk + selector goal,
  while incorporating all formal negative evidence through R10-B.  World-model
  residual/disagreement remains a gated optional augmentation, never a rescue
  for a failed information gate.
- Full record:
  `progress/2026-08-13_rase_canonical_idea_rewrite.md`.

## 2026-08-13 R10-B reproducibility verdict

- Full collection completed 198/198 trajectories, but the frozen exact-repeat
  gate is formally FAIL.  All 66 t8 causal feature records have replica parity,
  while 9/66 groups have K=3 outcome instability and only 32/66 preserve the
  frozen K=2 label.  The 33 frozen cases reduce to eight stable K=3 cases, 19
  stable controls and six unstable groups.  No unstable group was removed or
  replaced; canonical dataset/model construction did not run.
- A post-failure K=3 count diagnostic retained all groups and found 42 hazard
  events / 198 trials over 15 groups and 11 tasks.  Temporal state has weak OOF
  ranking (0.609 AUROC; bootstrap lower 0.429), but all-causal is 0.523 and the
  two policy AUROCs are 0.512/0.563.  Decision:
  `DO_NOT_ESCALATE_PROBABILISTIC_MODEL`; a Beta-binomial head alone is not
  supported by the recorded inputs.
- OFT first actions at t8 and t16 are bitwise identical for all 66 groups and
  all replicas, including the nine outcome flips.  A frozen 54-trajectory
  root-cause pilot (all nine unstable groups plus nine matched stable controls,
  each K=3) is now running in tmux `r10b_trace` and records full OFT action-trace
  hashes.  It is diagnostic only and cannot train or select a model.
- Detailed record:
  `progress/2026-08-13_r10b_repro_failure_and_trace_diagnostic.md`.

## 2026-08-13 R10-B case-control temporal restart (completed; superseded by verdict above)

- R9-C remains a formal FAIL.  Its unconditional one-step hazard had only
  12/480 positives and cannot support another model.
- R10-B changes the target/data contract, not the core RASE idea: conditional
  on fallback being certainly successful at t8, predict whether recoverability
  is lost by t16 from causal source history.
- A deterministic, label-balanced development manifest is frozen from stable
  R8 K=2 labels: 66 groups, 33/33 labels, 36 tasks, four suites, Pi0.5 and
  Pi0Fast, five task-held-out folds with both classes.  Manifest SHA256:
  `9f1febbd3db17d9e6178bf13d7b4f36900c12d6117261f8923b25fef6b37eaca`.
- This is case-control representation development only; it cannot estimate
  prevalence, calibration, or natural selector performance.
- Six contract tests and a real two-group temporal smoke passed.  Full K=3
  collection subsequently completed 198/198 trajectories.
- The gate-only post-runner executed and stopped at the frozen reproducibility
  FAIL.  It did not build the canonical dataset, run the canonical information
  gate, or start R10D.

## 2026-08-13 R9-B temporal observability pilot

- R7-A remains formally closed with source-risk OOF FAIL 0/5.  This is not
  overwritten by the R9 restart; it remains the canonical negative result.
- R9-B is a development-only pilot to test whether causal short history makes
  local recoverability/hazard observable.  The frozen manifest contains 24
  metadata-selected states (six per suite), two source VLAs (Pi0.5 and
  Pi0Fast), three fixed same-seed replicas, and boundaries `t={0,4,8,12,16}`.
- Manifest SHA256 is
  `33ec9adca504d4e2ab109a513e4919737c9d4cb4c934861fb7ba99ae4ca4db17`.
  All 24 StatePool reads passed.  The pilot is not validation or test data.
- The temporal collector is an opt-in extension of the validated R6-B1
  collector.  It records only causal four-step image/proprio/action history;
  no OFT action, future outcome, teacher cost, or world-model output is an
  input.  Smoke collection and shape/leakage checks passed.
- Full collection is running in server tmux `r9b_full`.  No training,
  selector, OFT counterfactual, or world-model experiment starts automatically.
- After collection, the locked order is: temporal reproducibility audit;
  dataset freeze; R9-C low-capacity information-support gate; only if that gate
  passes, a new shared-risk OOF protocol.  If it fails, stop model escalation
  rather than using a larger model or world-model feature to rescue it.

## 2026-08-13 R9-B/R9-C completion

- Full R9-B collection completed: 144 trajectories (24 states x 2 policies x 3
  replicas).  The initial audit rejection was corrected as a contract bug:
  successful episodes may end with a valid boundary prefix, while failed or
  horizon episodes still require all planned boundaries.  Corrected audit PASS:
  48 groups, 144 replicas, zero errors.
- Frozen temporal dataset: 480 transitions / 20 tasks / 40 base groups,
  SHA256 `21e062abb5d3b877ae339bab5c3300ada16e774dc810736df439ad1a9ca59993`.
  Hazard positives are only 12/480 (2.5%).
- R9-C information-support gate is **FAIL**.  Object and Spatial have zero
  hazard positives, camera has zero positives, both policies have fewer than 20
  positives, two of five task-held-out folds are single-class, and the
  all-causal probe (0.829) does not beat the policy+horizon prior (0.842) by
  0.05.  Temporal-state probe AUROC is 0.788 on only three valid folds.
- Decision: `STOP_UNIVERSAL_RISK_FOR_CURRENT_OBSERVATIONS`.  R10 OOF, selector,
  OFT counterfactual, world-model, validation and test remain locked.  This
  does not change the core idea; it identifies the next required change as an
  outcome-independent, label-balanced data/target contract rather than a
  larger model.

## 2026-08-13 R7-A reproducibility resolution and source-risk verdict

- The original 192-state exact-repeat audit remains formally **FAIL**.  A third
  same-seed/checkpoint repeat of Long state
  `sp1_5b2f2d114882fcce15f2a4be884ad084` confirms identical t=0 features,
  outcome, 510-step horizon and stop reason, but pairwise action divergence at
  steps 120--130.  This is late closed-loop nondeterminism, not t=0 feature
  corruption.
- The state is frozen in a reproducibility exclusion manifest (SHA256
  `1da46e3996fa4239e12e9ba6f37fa44ab5e3840c547b25323c29430829f89e97`).
  The original hash ordering selects Long failure
  `sp1_d15720f4d72bf2503482d6e75aa35781` as the replacement; no seed or outcome
  was searched.
- The amended cohort passes every gate: 191 states / 48 tasks, 89 successes /
  102 failures, 35 failure tasks, 20 mixed tasks, four-suite and fold dual-class
  support, exact-repeat 16/16 with zero errors.  The frozen dataset SHA256 is
  `538347f406017d68c5d3c119ae25bdc6da40944026c2ac583ddcc988a9f6bcb6`.
- Canonical five-seed OOF is complete and **FAILS 0/5**.  Mean AUROC is 0.631
  (0.564--0.675), mean AP gain 0.153 and mean ECE 0.169.  Object carries useful
  ranking signal, while Long is 0.376--0.565 AUROC for all seeds; every seed
  also fails the calibration and bootstrap-lower-bound gates.
- Formal decision: `STOP_SOURCE_RISK_ESCALATION`.  The policy-native adapter,
  Pi0.5/SmolVLA cohorts, OFT counterfactuals, selector, world-model ablation,
  validation and test are not unlocked.  Full record:
  `progress/2026-08-13_r7a_191_exact_repeat_and_source_risk_oof_negative.md`.

## 2026-08-12 R6 closeout and R7 source-risk restart

- Replica-aware Pi0Fast R6-C.1 OOF is complete and fails 0/5 seeds.  Formal
  means are success gap -11.67pp, false-continue 13.42%, corrected absolute
  paired harm 13.54%, and savings 21.90%.  The earlier 11.04% harm value omitted
  t8/t16 late-entry failures against a successful t0 baseline; corrected harm
  now derives directly from paired outcomes.  The verdict remains FAIL.
- A frozen-prediction post-hoc audit finds no threshold/controller family that
  reaches the full gate.  The best safe cross-fit t0 families save only
  6.56--7.51%, far below 20%.  A privileged perfect source-risk t0 controller,
  however, has +5.21pp success gap, zero harm and 34.63% savings.  This isolates
  source-risk representation—not another LCB/dwell/advantage tweak—as the only
  supported learning opportunity.
- The historical 740-state pool has only one independent episode and one seed
  per task; its many rows are temporal snapshots, not independent state
  diversity.  R7 therefore freezes 192 independently reset states: 48 tasks x
  four init-state IDs, all same-task episodes bound to one outer fold.
- R7 reset collection captured only step-0 reset states and executed zero
  source actions.  The frozen design has SHA256
  `9b78cf470e1c659d26b3789c5e113a6f51470c4e6a950cb146fc87c8c207f72f`.
  CPU dry-run and real reset-only collection passed 192/192; the audited
  state-key manifest SHA256 is
  `3a083717bbcc4786264c716820e83a31c7fddc28cb9b31dfac89822f194ed81a`.
- A legacy StatePool/new-controller snapshot mismatch was fixed without
  rewriting stored states.  A real restore audit passes 12/12 suite x
  perturbation cells at the expected timestep 10.
- Pi0Fast source-only collection is now producing true final outcomes plus
  deployable t0 features without OFT.  Only its frozen label-support PASS may
  unlock a dedicated task-held-out source-risk probe.
- The source-only process stopped at 138/192 on a deterministic Pi0Fast FAST
  grammar failure.  Exact same-seed reproduction located it at source step 20,
  after the complete causal t0 action proposal.  The collector now records this
  deploy-time decoder failure as a structured source failure, while retaining a
  hard error if it occurs before ten valid actions.  The repaired record passes
  the image/proprio/action-trace contract, 12 regression tests pass, and
  `r7a_pipeline` resumed from the 138 preserved records (54 remaining).
- The first real NPZ/JSON records pass a strengthened feature/metadata/leakage
  contract.  A two-camera rank bug in the new single-target model was found and
  fixed before training; full 192-row/48-task/five-fold synthetic pipeline
  smoke passes.  Fixed five-seed OOF code is ready but remains manually locked
  behind the completed label audit.
  OFT counterfactuals, selector training, shared multi-VLA training, WM
  features, validation and test remain locked.
- Canonical plan:
  `progress/2026-08-12_r7a_source_risk_restart.md`.

## 2026-08-11 screening completion and OFT-label launch

- Source-only screening completed 432/432 trajectories and passed the frozen
  data-value gate.  It authorizes label collection only; OFT rescueability and
  selector learnability remain unknown.
- Final-outcome analysis replaces the ambiguous legacy hard-state diagnostic:
  Pi0.5 enrichment contains 52/192 failed trajectories, 33 unique failure
  states and 23 failure tasks; Pi0Fast enrichment contains 61/96 failed
  trajectories, 61 unique failure states and 35 failure tasks.  Both span all
  four suites.
- Frozen same-suite selection retains every failure plus matched controls:
  Pi0.5 selects 64 enrichment states (33 failures + 31 controls), Pi0Fast 94
  (61 + 33), in addition to the full 48-state natural cohort per policy.
- The collection-plan audit passes and freezes 732 exact-repeat groups / 2,196
  OFT counterfactual branches at t={0,8,16}.  Targeted collection launched at
  13:04 CST; Spatial OFT readiness and the first full three-boundary trajectory
  succeeded.
- Training remains blocked until exact-repeat repro is frozen and both VLAs
  have >=30 source failures, >=20 early-rescuable groups, >=12 tasks, and
  failure/matched-success support in every suite.
- Full record:
  `progress/2026-08-11_r6c1b_screening_pass_and_oft_launch.md`.
- Early paired-replica monitoring found and fixed an audit-only overcollection
  bug: 1--2-step OFT cost spread was being treated like a boundary-label
  disagreement.  Cost-only variation is now retained for median/quantile cost
  supervision without rep2; true source/label disagreement still triggers
  rep2.  The corrected first-nine-pair audit reports 7 cost-variable, 2 exact,
  0 label-variable and 0 seed-mismatch pairs; two regression tests pass.

## 2026-08-11 methodology correction (superseded operationally by the completed screening record above)

- Continue the cheap 432-trajectory source-only screening to completion; do
  not start expensive OFT collection automatically.  The old resume process
  was stopped without interrupting the screening process.
- At 10:43 CST, screening was 303/432 complete.  Pi0.5 already had 40
  enrichment failures over 18 tasks and three completed/in-progress suites;
  Long was still missing, so the formal four-suite gate was not yet evaluated.
- Added a source-only screening go/no-go audit.  A PASS authorizes only OFT
  label collection; it does not establish OFT rescueability or learnability.
- Corrected decisive 1C evaluation defects: group/prediction misalignment,
  cumulative trajectory metrics, conditional missed-rescue denominator,
  duplicate-collapsing bootstrap, averaged-threshold bootstrap replay,
  accidental policy conditioning in baselines, enrichment leakage into OOF,
  and lost per-policy bootstrap intervals.
- Exact-repeat replicas now keep the same rollout seed.  rep2 is collected only
  after rep0/rep1 disagreement; source-success flips remain excluded from hard
  labels.  Natural rep1/rep2 can augment training but never calibration/OOF.
- Execution order is now gated: screening audit -> explicit OFT approval ->
  reproducibility freeze -> per-VLA 5-seed OOF -> shared-calibrated OOF ->
  optional R6-C.2 ladder.  A clear per-VLA 1C failure is the stop point for the
  current learned-selector escalation.
- Full record:
  `progress/2026-08-11_r6c1_decisive_experiment_methodology_revision.md`.
- R6-C.0 executed-cost aggregation was regenerated as a new v2 report.  The
  old selector still fails (pooled per-VLA success gap -5.31pp,
  false-continue 6.88%, savings 51.81%); the original frozen file was not
  overwritten.
- Four server regression tests pass.  OFT resume now also has a 5 GiB free-disk
  preflight and a pre-training label-support gate (>=30 source failures and
  >=20 early-rescuable groups per VLA, with four-suite matched support).

## One-sentence status

Two source/corrective policy pairs have large model-free teacher-cost opportunity,
so the multi-VLA real-time risk/correction idea remains viable; however, five-seed
task-held-out evaluation proves that one initial state per task is insufficient to
learn safe within-VLA failure ranking.

## Established results

- R3 success-only selection remains rejected: persistent OFT already equals the
  operator success oracle.
- R5 safe handback remains rejected for the SmolVLA/OFT cohort: only one of 24
  B24 states contains genuinely recovery-created all-K-safe support.
- R6-A policy-pair atlas is READY:
  - Pi0Fast: privileged savings 42.80%, 20 source-safe tasks, both seeds identical;
  - Pi0.5: privileged savings 94.73% and 95.98%, 44/46 source-safe tasks;
  - SmolVLA: fails the opportunity gate.
- R6-B0 uses 48 exact pre-action states / 48 tasks / four suites, two source
  rollout labels per policy-state, three task-bootstrap members and five fixed
  task-held-out seeds.
- Shared, per-VLA, zero-shot and leave-one-VLA-out B0 configurations all fail the
  required 4/5-seed gate.
- Pooled shared-model AUROC is misleading: without language it is 0.750 overall,
  but per-VLA models are near chance (Pi0Fast 0.478, Pi0.5 0.473).
- A post-hoc deployable instruction feature improves Pi0Fast AUROC to about 0.70,
  but only 1/5 per-VLA seeds passes; it does not unlock the next scientific gate.
- Exact gate-constrained cost ceilings remain large: Pi0Fast 57.08% and Pi0.5
  98.54%. Opportunity exists; the learned B0 observation/sample support is weak.

## Interpretation of the idea

The main idea is now:

```text
source VLA proposal + task + recent observations/actions
                        |
                        v
lightweight policy-conditioned short-horizon risk model
                        |
       +----------------+----------------+
       |                                 |
two-boundary safe dwell                 persistent corrective takeover
       |                                 |
continue source                         episode termination
```

There is no physical rollback. An unsafe proposed chunk may be rejected before
execution; simulator restore is supervision-only. Safe handback is not a current
main-method claim.

## Current locks

- R6-B0 learned initial-state controller: **NO-GO (0/5 main seeds)**.
- R6-B1 dynamic collector: **FROZEN** as the two-stage collector
  (`collect_r6b1_dynamic_boundaries.py`
  sha256 `0d512d2c5e37fb0af7ba5a5f3b696c87aa1472053c12605b9f6975a6554d61a3`,
  production `--bookkeeping-mode full`).  The two-state Pi0Fast pair parity
  rerun **PASSED** with the hard gate
  (`runs/pre_c0_r6/r6b1_smoke_pi0fast_spatial_pair_parity_v2/`): both states
  reproduce R6-A exactly (270/failure, 116/success).  See
  `progress/2026-08-10_r6b1_collector_frozen.md`.
- R6-B1.1 pilot manifest (`runs/pre_c0_r6/r6b1_pilot_manifest_v1.json`): frozen,
  **PASSED** (16 task-distinct states, `h={0,16,32}`, Pi0Fast seed 0, Pi0.5 seeds
  0/1).  Both gates green: pilot audit and source-parity hard gate, 24/24
  trajectories, 72 rows, all four suites, both label classes per policy.  See
  `progress/2026-08-10_r6b1_pilot_pass.md`.
- World-model residual/disagreement: **LOCKED** (R6-C no-WM baseline failed its
  5-seed gate; protocol keeps the WM ablation sealed until a no-WM dynamic
  baseline passes or a pre-registered state-level Pareto comparison is possible).
- Independent validation: **LOCKED** (requires both VLAs >=4/5 OOF seeds).
- Test and 100+ paired closed-loop episodes: **SEALED**.
- Third source VLA (plan phase 6): **LOCKED** behind the R6-C dual-policy gate.

## Next executable stage: R6-C gate verdict (FAIL — locked) and rework decision

1. **B1.2 full collection COMPLETE** (`runs/pre_c0_r6/r6b1_b1p2_v1/`, 144 npz,
   48 tasks, `h={0,16,32,64,96,128}`, Pi0Fast seed 0, Pi0.5 seeds 0/1).
   Source-parity hard gate **PASSED** after excluding one known nondeterministic
   trajectory group: `pi05_libero` seed 1, Goal `libero_goal_000010`,
   `sp1_b0b5e524da0d318935146d898a89ef8c` (R6-A reference 154 steps is not
   reproducible — 4 current-code reruns of the R6-A script give
   {153, 138, 153, 138}; collector and isolated reruns both give 138).  Frozen
   manifest: `runs/pre_c0_r6/r6b1_b12_exclusions_v1.json`.  Decision record:
   `progress/2026-08-10_r6b1_b12_parity_exclusion_decision.md`.
2. Candidate-arm dataset frozen with the R6-A parity recheck
   (`runs/pre_c0_r6/r6b1_b1p2_v1/r6c_candidate_arm_dataset.npz`,
   143 groups / 767 rows / 48 states / 48 tasks, all parity-checked, 1 group
   excluded) and the analysis written
   (`runs/pre_c0_r6/r6b1_b1p2_v1/candidate_arm_analysis.json` +
   `progress/2026-08-10_r6b1_b12_candidate_arm_analysis.md`).
3. **R6-C 5-seed task-held-out OOF COMPLETE — stage gate FAILED (0/5 seeds for
   every VLA and every mode).**  Risk model learns valid failure ranking, but
   the controller fails the gate: pi05 over-conservative (false_continue 7.8% >
   5%), pi0fast late-switch rescue decay (savings negative vs t=0 baseline),
   and cross-VLA transfer fails (pi05-trained model on pi0fast false_continue
   ~48-66%).  Cross-validated two ways (official avg-threshold and fold-correct
   aggregation); both 0/5 FAIL.  Full record:
   `progress/2026-08-10_r6c_no_wm_baseline_oof_negative.md`.
4. **R6-C.0 fold-correct final report FROZEN**
   (`runs/pre_c0_r6/r6c_candidate_arm_oof_v1/r6c_fold_correct_final_report.json` +
   `progress/2026-08-10_r6c_fold_correct_final_report.md`).  A real evaluation
   bug was found and fixed (`group_boundaries` mis-indexed rows; affects only
   metric aggregation, not training) and the 5-seed OOF was rerun; FAIL 0/5 is
   robust.  New metrics reported (conditional missed-rescue, absolute paired
   harm, rescue/burden, task-cluster bootstrap, teacher-savings); data,
   protocol, collector, exclusion-manifest hashes are locked.
5. **R6-C.1A early-window model-free opportunity audit DONE**
   (`runs/pre_c0_r6/r6c1_early_window_audit.json`): cost-aware oracle success
   gap +11.9pp and savings 80.6% pooled.  Pi0Fast passes the 1A gate (26
   source-fail/early-rescuable groups >= 10, 48 decision-divergence groups, 4
   suites, 48 tasks).  Pi0.5 fails only the positive-count gate (6 rescuable
   groups < 10) — the plan's decision tree routes this to R6-C.1B targeted
   collection, not to method failure.
6. Per the protocol locks, **R6-D WM ablation, R6-E independent validation,
   the sealed test, and the 100+ paired closed-loop episodes all remain LOCKED**
   (they require both VLAs to pass >=4/5 OOF seeds).  The third source VLA
   (plan phase 6) does not start.
7. Rework decision: (a) fold-correct aggregation is now the official metric
   (done in R6-C.0); (b) false_continue denominator re-review (kept as
   original-protocol primary gate; conditional missed-rescue reported without a
   hard gate pending power analysis); (c) mechanism-driven rework R6-C.1
   (early-window t={0,8,16} layered controller, policy calibration adapters)
   now in progress.
8. **R6-C.1B infrastructure COMPLETE (in code), screening in flight.**  The
   frozen manifest `runs/pre_c0_r6/r6c1b_initial_keys_v1.json` separates 48
   natural-development-eval states (metadata-only selection) from 96
   train-enrichment candidates (96; 48 tasks, 144 records).  Source-only
   screening (`scripts/run_r6c1b_screen.sh`) is running in the background over
   432 rollouts (Pi0.5 seeds 2-3 + Pi0Fast seed 1, `t={0,8,16}`); the
   reproducibility protocol (two replicas per new triple, success-flip
   exclusion via `scripts/audit_r6c1b_repro.py`) and the targeted OFT-labelled
   collection (`scripts/run_r6c1b_collect.sh`, `t={0,8,16}`, OFT server per
   suite) are ready and idempotent.  The merged dataset builder merges B1.2
   with the new collection and treats reference-less new triples as
   reproducibility-audited (not strict-parity) per the plan.
9. **R6-C.1C early-window stratified selector implemented and smoke-validated.**
   `rase/risk/light_risk_student.py::CandidateArmStudent` now consumes VLA
   identity embeddings, deployable per-VLA behavior descriptors (source
   rollout statistics only), and optional per-VLA FiLM calibration adapters,
   plus a new advantage head.  `scripts/train_r6c1_early_selector.py` trains
   the t={0,8,16} layered controller (no emergency trigger; inner-OOF
   thresholds; fold-correct aggregation; absolute paired harm, conditional
   missed-rescue and task-cluster intervals) across the full R6-C.2 ladder
   (per_vla / shared / shared_id / shared_desc / shared_calib / loo /
   zero_shot).  `scripts/audit_r6c1_selector_stability.py` enforces the R6-C.1
   stage gate (success gap >= -5pp, false-continue <= 5%, paired harm <= 5%,
   savings >= 20%, no suite-concentrated harm; >=4/5 seeds).
10. **Per-policy gate correctness fix (2026-08-10 late session).**  Found and
    fixed a methodological bug: for `shared*` modes the OOF eval covers BOTH
    policies, so gating per-VLA on the pooled aggregate (`report["metrics"]`)
    was wrong — per-VLA results now read each policy's own fold-correct
    metrics (`metrics_by_policy`) and its own suite concentration
    (`metrics_by_policy_suite`), never pooled.  `shared*` modes now run once
    per seed (the report carries every policy); `per_vla`/`loo`/`zero_shot`
    still run per policy/direction.  Also fixed: zero-shot no longer carries a
    VLA identity embedding (pure shared-core challenge metric, red line 5),
    `calibration_tasks` guards the empty partition, the per-policy bootstrap
    interval uses per-policy OOF rows, and the config comparison tolerates
    missing modes.  Smoke-validated `per_vla`, `shared_calib`, `shared_desc`,
    `loo`, `zero_shot` modes + the stability audit on the B1.2 dataset.
11. **R6-C.1B pipeline staging fix.**  The resume driver now passes
    `DATASET_ROOT="$MERGED_ROOT"` to the R6-C.2 OOF ladder (previously it fell
    back to the B1.2 dataset root), and the R6-C.1B protocol drives the merged
    build (verified: B1.2 seeds ⊂ R6-C.1B qualified set, dry-run build passes
    with 143 groups / 767 rows / 0 parity failures).  `run_r6c1b_resume.sh`
    relaunched (detached) and is waiting for screening COMPLETE.

## Canonical records

- `progress/2026-08-10_r6b1_source_parity_gate_no_go.md`
- `progress/2026-08-10_r6b1_collector_zero_side_effect_fix.md`
- `progress/2026-08-10_r6b1_collector_frozen.md`
- `progress/2026-08-10_r6b1_pilot_pass.md`
- `progress/2026-08-10_r6b1_b12_candidate_arm_analysis.md`
- `progress/2026-08-10_r6b1_b12_parity_exclusion_decision.md`
- `progress/2026-08-10_r6c_no_wm_baseline_oof_negative.md`
- `progress/2026-08-10_r6c_fold_correct_final_report.md`
- `progress/2026-08-10_r6c1_early_window_audit.md`

- `progress/2026-08-10_r6b0_multivla_oof_results.md`
- `runs/pre_c0_r6/policy_pair_atlas_v1.json`
- `runs/pre_c0_r6/r6b0_opportunity_ceiling_v1.json`
- `runs/pre_c0_r6/r6b0_shared_id_calibrated_5seed/stability.json`
- `runs/pre_c0_r6/r6b0_comparisons_v1/`
- `runs/pre_c0_r6/r6b0_language_exploratory_v1/`
