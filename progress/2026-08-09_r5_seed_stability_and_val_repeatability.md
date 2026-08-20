# R5 seed stability and validation-boundary repeatability

Date: 2026-08-09  
Status: **MODEL NO-GO; FULL VAL/TEST EVALUATION PAUSED FOR PROBABILISTIC RELABELING**

## Why these experiments were run

The corrected R4-D baseline retained high row-level AUC but failed the safe-handback
success gate. Before spending simulator budget on a conference-scale evaluation, R5 had to
answer two cheaper questions:

1. Is the corrected result stable across training seeds and calibration folds?
2. Are exact-state handback counterfactual labels reproducible on a truly task-disjoint
   validation split?

Both are necessary. A high-AUC model with seed-sensitive stopping decisions is not a safe
controller, and a deterministic classifier cannot be justified when its supposed binary
targets are themselves stochastic or protocol-dependent.

## Frozen split and provenance

`runs/pre_c0_r5/r5_split_manifest_v1.json` freezes:

- train: 72 states / 24 tasks;
- validation: 24 states / 8 tasks;
- test: 24 states / 8 tasks;
- train/validation/test state overlap: 0;
- train/validation/test task overlap: 0.

The previous M4 subset is explicitly marked invalid because 24/24 states overlapped the
training set. The new test split has not been collected, scored, or used for any decision.

The historical validation opportunity audit contains 24 complete states and eight true task
IDs. Persistent OFT succeeds on 23/24; finite handback succeeds on 16/24; the cost oracle
would save 36.7% OFT steps. Its formal `safe_handback_status=not_ready` is caused by applying
train-scale opportunity thresholds to a 24-state validation split. R5 therefore permits an
explicit, provenance-recorded bypass only for unfiltered `val`/`test` collection; train can
never use that bypass.

## Five-seed corrected OOF stability

Seeds: 20260809--20260813. Each run used leakage-safe nested task folds, fresh model and
ensemble initialization, train-only normalization, inner calibration tasks, and one earliest
handback decision per state.

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| row AUC | 0.9270 | 0.9221 | 0.9319 |
| policy success | 83.10% | 81.69% | 85.92% |
| gap vs persistent | -8.45pp | -9.86pp | -5.63pp |
| conditional false handback | 9.23% | 6.15% | 10.77% |
| OFT savings | 22.97% | 18.52% | 30.60% |

Results:

- all four gates passed: **0/5 seeds**;
- per-fold thresholds span **0.3323--0.9998**;
- 27/71 states (38.03%) changed exact handback time across seeds;
- only 77.32% mean pairwise exact-time agreement;
- each fold has only 11--12 calibration states;
- even observing zero errors in 11 calibration states gives a one-sided 95% error upper
  bound of 23.84%; at least 59 independent zero-error calibration examples are needed to put
  that bound below 5%.

Interpretation: representation ranking is stable, but the safety-critical operating point is
not. More epochs, another random seed, or a slightly larger MLP is not the remedy. The project
needs substantially more independent persistent-rescuable calibration states and a
conservative finite-sample calibration method.

## True-validation smoke and repeatability

The versioned collector was tested on one frozen validation state from each LIBERO suite.
The first collection produced 27 boundary rows over four states:

- persistent replay parity: **4/4**;
- fresh finite-safe states: 2/4;
- fresh cost-oracle saving: 31.22%;
- historical-vs-fresh handback-label agreement: **22/27 = 81.48%**;
- provenance records true task join, projection hash, source gate state and the eval-only
  bypass.

Because 81.48% agreement was insufficient, the same four states were recollected from
scratch. Fresh-run versus fresh-run results were:

- common boundary-label agreement: **25/26 = 96.15%**;
- exact minimum-successful-boundary agreement: **4/4**;
- one Goal boundary changed from failure to success at h64;
- the Goal trajectory exposed h128 in one run but terminated before h128 in the other;
- late Goal features/actions diverged despite identical state ID and policy checkpoints;
- Object was bit-exact over all seven boundaries; Spatial labels were 6/6 identical, though
  one-step Student latent differed at a later boundary.

This separates two effects. Much of the 22/27 historical disagreement is an old-vs-new
protocol difference, but the Goal repeat proves that some live counterfactuals are genuinely
trajectory-sensitive. A single 0/1 `success_if_handback_now` is therefore not a sufficiently
reliable scientific target.

## Same-snapshot K=5 probabilistic-label pilot

The collector was upgraded and then executed on the unstable Goal validation state at
h={0,64}. Each boundary was saved once and restored for five Student continuations with
recorded independent policy seeds.

| Boundary | Successes | Empirical p(success) | one-sided-95% Wilson LCB | Conservative label |
|---|---:|---:|---:|---|
| h0 | 3/5 | 0.60 | 0.272 | unsafe |
| h64 | 5/5 | 1.00 | 0.649 | safe over observed repeats, not certified at 0.95 |

At h0, continuation lengths were 290, 131, 291, 138 and 163 steps, with three successes.
This directly demonstrates that the former binary label hides policy-outcome variance even
from the same saved simulator boundary. At h64 all five continuations succeeded, but their
lengths still ranged from 80 to 167 steps.

Under the implemented Wilson rule, at least 52/52 successes are required before the lower
bound exceeds 0.95. Thus K=5 is a label-entropy/development probe, not a certification sample
size. Formal deployment calibration still needs the larger independent cohort and paired
closed-loop evidence described below.

## Go / No-Go

- Corrected 0.6M baseline as a deployment controller: **NO-GO**.
- V-JEPA delta as a mainline feature: **NO-GO** under the completed exact-pair ablation.
- Existing high AUC as a paper safety claim: **NO-GO**.
- Full 24-state validation collection with one rollout per boundary: **PAUSED**; it would
  scale a noisy binary-label protocol.
- Frozen test and large paired closed-loop evaluation: **BLOCKED**.
- R5 probabilistic handback relabeling and larger calibration cohort: **GO**.

## Revised execution order

1. Extend the boundary collector to take `K` Student continuations from the *same saved
   boundary snapshot*, record seeds, successes, continuation lengths and Beta/Wilson bounds.
   Do not estimate repeatability by rerunning the entire OFT prefix alone.
2. Run a stratified 16-state relabeling pilot with `K=5`; include all four suites, finite-safe
   and persistent-only states. Gate: at least 95% same-snapshot replay integrity and report
   label entropy by suite/boundary.
3. Replace the binary handback head with a probabilistic/Beta-binomial or soft-label head;
   controller handback requires a one-sided lower confidence bound, two consecutive safe
   boundaries, and persistent-rescuable support.
4. Expand development data to at least 300 states and reserve at least 100 independent
   persistent-rescuable states for calibration. Do not reuse model-development states for
   threshold certification.
5. Pre-register the controller and rerun task-held-out validation. Required gates remain:
   success gap >= -5pp, conditional false handback <=5%, OFT savings >=20%, all with
   task-cluster intervals and seed stability.
6. Add a second VLA through the existing canonical action adapter. Compare shared encoder +
   VLA-conditioned head against separate per-VLA heads; leave-one-VLA-out is the primary
   universality test.
7. Touch the frozen test only after validation gates pass; then run at least 100 paired
   closed-loop episodes, two policy pairs, two seeds and all four suites.

## Completed code and artifacts

- `scripts/train_r4d_light_risk_student_v2.py`
- `scripts/analyze_r4d_seed_stability_v2.py`
- `scripts/freeze_r5_split_manifest.py`
- `scripts/collect_r4_boundary_transitions_v2.py`
- `scripts/collect_r4_boundary_transitions_v3.py` (same-snapshot K-repeat labels)
- `scripts/run_pre_c0_r5_collect.sh`
- `scripts/run_pre_c0_r5_probabilistic_collect.sh`
- `scripts/compare_r5_boundary_repeats.py`
- `runs/pre_c0_r5/seed_stability_v1.json`
- `runs/pre_c0_r5/r5_split_manifest_v1.json`
- `runs/pre_c0_r5/opportunity_audit_costaware_val_qc.json`
- `runs/pre_c0_r5/boundary_val_smoke4_v1/`
- `runs/pre_c0_r5/boundary_val_smoke4_repeat_v1/`
- `runs/pre_c0_r5/boundary_repeatability_smoke4_v1.json`
- `runs/pre_c0_r5/boundary_goal_k5_v1/`
