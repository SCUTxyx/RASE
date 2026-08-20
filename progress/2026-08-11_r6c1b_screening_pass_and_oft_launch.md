# R6-C.1B screening PASS and targeted OFT-label launch

Date: 2026-08-11

## Decision

The 432-trajectory source-only screening is complete and passes the frozen
data-value gate.  Proceed once with the targeted, reproducibility-audited OFT
label collection.  This is not a selector result: source-only screening cannot
establish whether the corrective arm rescues a failure at t={0,8,16}, nor
whether observations identify the correct decision.

## Screening result

The frozen go/no-go report is
`runs/pre_c0_r6/r6c1b_screen_v1/screening_go_no_go.json` and records
`GO_LABEL_COLLECTION` with 432/432 expected trajectories complete.

Final source-rollout outcomes are summarized separately in
`runs/pre_c0_r6/r6c1b_screen_v1/screening_outcome_analysis.json`:

- Pi0.5 natural evaluation: 8/96 failures, 6/48 unique failure states, six
  tasks, all four suites.
- Pi0.5 train enrichment: 52/192 failures, 33/96 unique failure states, 23
  tasks, all four suites.  Fourteen states change outcome across seeds 2/3;
  this is cross-seed deployment variation, not exact-repeat nondeterminism.
- Pi0Fast natural evaluation: 28/48 failures, 28 unique failure states, 28
  tasks, all four suites.
- Pi0Fast train enrichment: 61/96 failures, 61 unique failure states, 35
  tasks, all four suites.

The enrichment therefore fixes the immediate failure-support shortage,
especially for Pi0.5.  It does not yet fix natural-evaluation prevalence:
Pi0.5 still has only six unique natural failure states, so uncertainty and
task-cluster intervals remain essential.

The legacy `hard_enrichment_states` field in the gate report must not be used
as a scientific statistic.  It also marks a successful trajectory as hard
when the task is not already complete within 16 environment steps, which is
not a meaningful hardness definition for these manipulation tasks.  The new
descriptive analysis uses final source-rollout outcomes only and leaves the
frozen gate/hash unchanged.

## Frozen collection selection

`runs/pre_c0_r6/r6c1b_oft_selection_v2.json` selects every enrichment failure
state plus deterministic same-suite success-only controls.  It never consults
OFT outcomes and never uses cross-suite controls.

- Pi0.5: 33 failure states + 31 matched controls = 64 enrichment states.  Two
  Object failures lack same-suite controls but remain retained.
- Pi0Fast: 61 failure states + 33 matched controls = 94 enrichment states.
  Twenty-eight failures lack a one-to-one same-suite control but remain
  retained.
- The 48-state natural cohort is collected in full for each policy.

The pre-execution plan audit passes all gates and freezes 732 exact-repeat
trajectory groups and 2,196 t={0,8,16} OFT counterfactual branches.  The
natural/enrichment split, initial-state manifest hash, screening-audit hash,
and selection hash are checked before collection.

## Execution status

The explicit approval marker was created only after reviewing the complete
screening and frozen plan.  At 13:04 CST,
`scripts/run_r6c1b_resume.sh` launched the targeted collection into
`runs/pre_c0_r6/r6c1b_collect_v1/`.  The Spatial OFT service passed readiness,
the first Pi0.5 trajectory and all three boundaries completed, and the
pipeline remains live.  Free space was about 15 GiB; the driver refuses a
fresh launch below 5 GiB.

### Early exact-repeat audit correction

An in-flight read-only audit of the first paired replicas found that five of
six pairs were incorrectly marked `needs_third` even though rollout seed,
source outcome, source terminal step and all three boundary success labels
agreed.  The only differences were 1--2 OFT teacher calls on successful
branches.  The audit had included teacher cost in the hard-label signature,
contradicting the frozen rule that cost spread is supervised by replica-level
median/quantile.

`audit_r6c1b_repro.py` now separates boundary success labels from teacher cost:

- seed/source-outcome/source-terminal-step/boundary-label disagreement still
  triggers rep2;
- teacher-cost-only spread is recorded as `cost_variability`, keeps both
  replicas, and does not trigger rep2;
- after a true three-replica label disagreement, the group remains a
  probabilistic-label group rather than a hard label.

The corrected partial audit had 2 fully identical pairs, 7 cost-variable
pairs, 0 label-variable pairs, 0 seed mismatches and 0 rep2 requests among the
first 9 complete pairs.  Two regression tests cover cost-only variation and a
true boundary-success flip; both pass on the server.  No collected label was
changed or discarded by this correction.

## Next gates

1. Complete rep0/rep1 with identical rollout seeds.  Collect rep2 only for
   disagreements, then freeze the reproducibility exclusion manifest.
2. Before any model training, require per VLA at least 30 canonical source
   failures, at least 20 early-rescuable groups, at least 12 tasks, and both
   failure/matched-success support in all four suites.
3. If either policy fails label support, stop before training.  A risk model
   cannot compensate for an ineffective corrective arm.
4. If both pass, build the leakage-safe merged dataset.  Enrichment and extra
   replicas are outer-training only; natural rep0 alone defines calibration
   and OOF evaluation.
5. Run the decisive five-seed per-VLA selector first.  Both policies require
   >=4/5 seeds satisfying success gap >= -5pp, frozen false-continue <=5%,
   paired harm <=5%, teacher savings >=20%, and no suite-concentrated harm.
6. Run shared+calibration only after both per-VLA gates pass.  World-model
   features, independent validation, test, closed loop, and a third VLA stay
   locked.

## Stop rule

A clear corrected per-VLA 1C failure ends method escalation for the current
observation-only early selector.  A near pass permits at most one
predeclared diagnostic iteration; it does not authorize an unbounded model or
feature search.
