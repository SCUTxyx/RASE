# R6-C.1 decisive experiment: methodology correction and gated execution

Date: 2026-08-11

## Decision

Continue the cheap source-only screening to completion.  Do not automatically
enter OFT-labelled collection.  The old resume process was stopped while the
screening process was left running.  Expensive collection now requires both a
completed screening audit and an explicit approval marker.

R6-C.1C remains worth executing once, as the decisive experiment for the
current learned early-selector line.  Its interpretation is narrower than
previously stated: the 99.3% success / 80.6% savings cost-aware oracle is a
privileged hindsight upper bound.  It proves value exists in the arm set, but
does not prove that deployment-time observations identify the oracle choice.

## Screening status at the methodology freeze

At 2026-08-11 10:43 CST, 303/432 source-only trajectories were complete.
Spatial and Object were complete, Goal was in progress, and Long had not
started.  Before Long, Pi0.5 already had 40 enrichment failures covering 18
tasks across three suites; Pi0Fast already had more than 20 failures.  This is
encouraging evidence that the source-failure support problem is being fixed,
but it is not evidence of OFT rescueability and is not a method gate.

### Completion update

Screening later completed 432/432 and passed the frozen source-only gate.
Final-outcome analysis found 33 unique Pi0.5 enrichment failure states over 23
tasks and 61 unique Pi0Fast enrichment failure states over 35 tasks, both over
all four suites.  The targeted, same-suite-matched OFT-label plan was frozen at
732 exact-repeat trajectory groups / 2,196 boundary branches and launched only
after explicit approval.  See
`progress/2026-08-11_r6c1b_screening_pass_and_oft_launch.md`.

## Mandatory code corrections made before 1C

1. Fixed group/prediction alignment in `controller_early_window`.  The old
   implementation restarted the prediction index at zero for every group and
   reused the first group's scores.
2. Fixed trajectory records that previously stored cumulative success and
   teacher-step totals instead of per-trajectory outcomes.
3. Corrected conditional missed-rescue denominator to the number of true
   rescue opportunities.  The frozen original false-continue denominator is
   retained separately.
4. Replaced duplicate-collapsing bootstrap replay and averaged fold thresholds
   with task-cluster resampling of already frozen fold-specific trajectory
   decisions.
5. Removed accidental policy embeddings from `per_vla` and `shared`; only
   `shared_id` receives VLA identity.
6. Made `train_enrichment` training-only.  Threshold calibration and OOF gates
   use the natural cohort; natural rep1/rep2 trajectories can augment training
   but cannot duplicate evaluation weight.
7. Fixed the stability audit to retain per-policy bootstrap intervals.
8. Renamed the actual threshold procedure honestly: it is an outer-train,
   task-held-out calibration split, not inner OOF.
9. Counted executed OFT steps on failed persistent rollouts in the R6-C.0 cost
   report instead of assigning them zero cost.
10. Fixed exact-repeat reproducibility: `rollout_index` no longer changes the
    rollout seed.  Disagreement after rep0/rep1 triggers only the affected rep2
    trajectories; source-success flips after rep2 are excluded from hard-label
    training.
11. Added a protocol hash lock to every 1C training report.
12. Replaced the automatic all-mode ladder with gated execution.

Server verification: all modified Python files compile, all modified shell
drivers pass `bash -n`, and four controller/data-split regression tests pass.

The corrected R6-C.0 executed-cost aggregation was also rerun into a new file
without overwriting the frozen report:
`runs/pre_c0_r6/r6c_candidate_arm_oof_v1/r6c_fold_correct_final_report_v2_executed_cost.json`.
The old method remains FAIL: pooled per-VLA success gap -5.31pp,
false-continue 6.88%, and savings 51.81%.  Counting failed persistent rollouts'
executed OFT steps changes cost accounting, not the scientific verdict.

Preflight storage at 11:35 CST was 15 GiB free.  Existing R6 data occupy well
under 1 GiB, but the resume driver now refuses to start OFT collection below a
5 GiB free-space floor.

## Revised execution plan

### Gate S: finish and audit screening

Run `audit_r6c1b_screening_go_no_go.py` after all 432 trajectories finish.
PASS only authorizes label collection.  It requires complete counts, at least
30 Pi0.5 enrichment failures, four-suite and at least 12-task Pi0.5 failure
coverage, and at least 20 Pi0Fast failures.  It cannot judge OFT rescueability.

If Gate S fails, stop before OFT and redesign or terminate enrichment.  If it
passes, review the report and create
`runs/pre_c0_r6/r6c1b_screen_v1/APPROVE_OFT_LABEL_COLLECTION` before resuming.

### Gate L: collect counterfactual labels and freeze reproducibility

Collect t={0,8,16} OFT labels with rep0/rep1 using the exact same rollout seed.
Collect rep2 only for triples with outcome, terminal-step, or boundary-label
disagreement.  Strict parity applies only to atlas-referenced triples.  The
hard-label dataset cannot build until the reproducibility manifest is frozen.

After collection, measure per VLA and suite:

- source-failure groups;
- early-rescuable groups;
- persistent success at t0/t8/t16;
- source/OFT label variability and excluded flips;
- actual OFT calls and cost distribution.

Require at least 30 source failures and 20 early-rescuable groups per VLA, with
failure and matched-success support in every suite.  If rescueability is below
support, stop before training; additional risk-model complexity cannot create a
recovery arm.

### Gate D: build the leakage-safe dataset

Merge B1.2 and R6-C.1B with explicit `cohort_role`, `base_group_id`, and
`replicate_index`.  Enrichment and extra natural replicas may appear only in
outer-training folds.  Natural rep0 trajectories alone define calibration and
OOF evaluation.  All rows sharing a real task remain in the same fold.

### Gate 1C-P: decisive per-VLA selector

Run `per_vla` first for five training seeds.  Each VLA must pass at least 4/5
seeds with fold-correct success gap >= -5pp, original false-continue <=5%,
absolute paired harm <=5%, savings >=20%, and no suite-concentrated harm.

Outcomes:

- Pass both VLAs: continue to shared+calibration.
- Fail clearly: terminate learned-selector escalation for this paper and retain
  the infrastructure/oracle benchmark as a negative-result contribution.
- Near-pass: permit at most one predeclared diagnostic iteration, only when
  both VLAs pass at least 3/5 seeds and no mean primary metric misses its limit
  by more than 2 percentage points.  The change must address a measured data or
  calibration defect; no unregistered larger model or new feature search.

### Gate 1C-U: shared multi-VLA claim

Only after both per-VLA gates pass, run `shared_calib` (more accurately:
shared core plus behavior-descriptor-conditioned FiLM).  If it fails while
per-VLA passes, the valid claim is policy-specific early risk control, not a
universal selector.  The optional shared/shared_id/shared_desc/LOO/zero-shot
ladder requires a second explicit approval marker.

Pure zero-shot is a challenge result.  The primary generalization claim is low
shot adaptation measured at 0/8/16/32 calibration trajectories.

### Later locks

R6-D world-model residual/disagreement remains locked until the no-WM
per-VLA and shared-calibrated selectors pass.  Independent validation, test,
100+ paired closed-loop episodes, and a third VLA remain locked behind their
existing gates.  A world model is retained only if it adds state-level Pareto
value on both VLAs without unacceptable latency.

## Stop rule

R6-C.1C is the last planned method attempt for the current observation-only,
early-window selector.  A clear per-VLA failure ends this method escalation.
It does not prove all risk prediction is impossible; it proves this data,
representation, arm pair, and early controller do not support the paper's
predeclared success/safety/cost claim.
