# R7 multi-VLA risk and selector plan (OpenVLA role clarified)

Date: 2026-08-12

## Decision

Keep the project goal, but split it into two claims that require different
evidence:

1. **multi-VLA source-risk prediction**: predict whether the currently active
   source policy will fail from deployable state, task and source-action intent;
2. **source-general selector with a fixed corrective family**: use the risk
   score together with fallback recoverability/cost to decide whether to keep
   the source or enter OpenVLA-OFT.

The first claim may use every runnable source policy.  The second claim may use
only a source/fallback pair whose model-free opportunity audit passes.  A policy
can therefore be useful for risk generalization even when it is scientifically
invalid as a selector pair.

The main transfer claim is **shared risk core + outcome-free behavior descriptor
+ small calibration**, not one zero-shot threshold for every possible VLA.  The
latter was contradicted by R6 and remains a challenge metric only.

## Why OpenVLA is not listed as a direct source today

The server does contain OpenVLA.  The available checkpoints are
`ckpts/oft_spatial`, `ckpts/oft_object`, `ckpts/oft_goal` and `ckpts/oft_10`.
Their configs identify `OpenVLAForActionPrediction`, backed by Llama-2 7B, and
each checkpoint is tied to one LIBERO suite.  They are loaded in the isolated
OpenVLA-OFT environment through `rase.oracle.openvla_oft_adapter` and a ZeroMQ
oracle server.

This differs from the current source path.  `Pi0Fast`, `Pi0.5` and `SmolVLA`
are LeRobot checkpoints accepted by `load_lerobot_policy_bundle`, so the source
collector can reset their queues, seed them, run a complete continuation and
record action traces in process.  The OpenVLA-OFT checkpoints currently expose
only the fallback/oracle service contract.  No separate generic base-OpenVLA
LIBERO source checkpoint has been verified on this server.

Consequences:

- OpenVLA-OFT is already present in the project as the corrective policy;
- it is not yet a fourth *direct source* under the current collector API;
- using the same OpenVLA-OFT checkpoint as both source and fallback would make
  the selector pair degenerate and cannot support a correction claim;
- it may later be adapted as a risk-only source and architecture-held-out test,
  but requires an oracle-backed source continuation, source-only parity audit
  and substantial 7B inference cost;
- an OpenVLA source may enter a selector experiment only with a distinct
  corrective arm and a fresh model-free opportunity audit.

## Policy inventory and admissible roles

| Policy | Direct source now | Risk cohort | Selector with OpenVLA-OFT | Current role |
|---|---:|---:|---:|---|
| Pi0Fast | yes | yes | yes, opportunity exists | R7-A canonical source |
| Pi0.5 | yes | yes | yes, but natural failures are sparse | high-capability/safety cohort |
| SmolVLA | yes | yes | no, R6 model-free opportunity failed | lower-capability transfer cohort |
| OpenVLA-OFT | oracle service only | optional after adapter | no, same arm is degenerate | fallback; later architecture holdout |

This table is deliberately asymmetric.  “Universal risk” does not require the
same policies as “universal selector.”  With only one validated fallback
family, the defensible selector claim is source-general but fallback-specific.
A fallback-general claim requires at least one additional corrective policy.

## R7-A: finish Pi0Fast source-risk decisively

The running cohort is 48 tasks x four independent init states.  Complete all
192 source-only rollouts before reading the formal label gate.  It contains no
OFT outcomes, actions or cost.

Execution order is fail closed:

1. source-label support audit;
2. frozen 16-state exact-repeat audit, stratified by suite and outcome;
3. hash-bound immutable dataset;
4. five fixed seeds x five task-held-out folds x three task-bootstrap members;
5. stability verdict.

The exact-repeat and dataset stages now use file locks because both the original
driver and the independent post-label driver may reach them.  Completed PASS
artifacts are accepted only when their upstream hashes match.

The canonical action feature has also been corrected.  The old builder reduced
Pi0Fast to the first 7-DoF action even though the policy is configured with
`n_action_steps=10`.  The dataset now reconstructs the causal initial proposal
from `source_action_trace[0:10]` and retains the old one-step summary only as an
ablation.  This does not require recollection and uses no future observation,
outcome or OFT information.

Pi0Fast representation gate:

- task-held-out AUROC >= 0.75;
- task-bootstrap AUROC 95% lower bound >= 0.65;
- AP minus failure prevalence >= 0.10;
- calibrated ECE <= 0.10;
- every suite AUROC > 0.60;
- at least 4/5 fixed seeds pass.

If the full model fails but at least three seeds have AUROC >=0.65 and AP gain
>=0.05, permit exactly one additive policy-native feature adapter.  A near-
random result stops the representation line; a world model may not rescue it.

## R7-B: construct real multi-VLA risk evidence

Only after the Pi0Fast canonical/allowed-native gate passes, run source-only
screening on the *same 192 frozen reset states* for Pi0.5 and SmolVLA.  Same-
state evaluation removes environment-cohort confounding, but task remains the
bootstrap/split unit.

Each additional VLA must independently pass:

- 192 complete source-only rows and exact provenance;
- at least 40 successes and 40 failures;
- failures in at least 12 tasks and mixed outcomes in at least eight tasks;
- at least four examples of each label in every suite;
- a 16-state exact-repeat gate;
- every fit/calibration partition contains both classes.

These thresholds prevent selecting a VLA merely because it fails often.
Failure-only or success-only cohorts cannot learn calibrated risk.  SmolVLA is
likely the better second *training* source if it retains enough successes;
Pi0.5 remains important as a high-success false-alarm and calibration stress
test even if it cannot support a full per-VLA learner.

For every qualified VLA run the same per-VLA source-risk probe first.  Shared
training is unlocked only after at least two policies show real within-policy
signal.  Pooling policies must never turn a between-policy base-rate difference
into a false “universal risk” result.

## R7-C: shared risk model and transfer ladder

The target is

`P(source fails | observation, task, source action proposal, policy condition)`.

The lightweight `SourceRiskStudent` supports two independent policy conditions:

- a small learned policy-ID embedding for seen VLAs;
- an outcome-free behavior descriptor for unseen/adapted VLAs.

The descriptor is computed only from deployable RGB moments, proprioception and
canonical action summaries.  It must never contain success rate, failure label,
OFT rescueability or teacher cost.  For seen-policy OOF it is fit from outer-
train tasks only.  For a held-out VLA it is estimated from a separate unlabeled
calibration cohort, never from evaluation outcomes.

Run this fixed ladder:

1. per-VLA models (performance upper bound);
2. pooled shared model without policy condition (negative/control baseline);
3. shared + seen-VLA ID embedding;
4. shared + outcome-free descriptor (canonical transfer method);
5. shared + descriptor + tiny temperature/bias or FiLM calibration;
6. leave-one-VLA-out zero-shot (challenge metric);
7. held-out adaptation curves using 0/8/16/32 unlabeled trajectories, followed
   by a separately reported 0/8/16/32 labeled calibration curve.

Required reporting:

- per-policy and per-suite AUROC/AP/ECE, never only pooled AUROC;
- task-cluster bootstrap confidence intervals;
- task-only, observation-only, one-step-action and full-chunk-action ablations;
- parameter count, TorchScript/ONNX parity and end-to-end latency;
- descriptor/calibration data excluded from the held-out evaluation tasks.

Shared-model gate: at least two qualified VLAs pass 4/5 seeds, and the canonical
shared+descriptor model is within 3 AUROC points of each per-VLA model while
meeting its calibration/suite gates.  Zero-shot failure does not by itself fail
the method; failure to adapt with 32 trajectories does.

## R7-D: selector re-entry, still model-free first

Risk generalization and selector success are not interchangeable.  For each
admissible source/fallback pair collect fresh t0 persistent-OFT counterfactuals
on the independent R7 cohort.  Before fitting a selector require:

- privileged success gap >= -5pp versus persistent fallback;
- privileged teacher-step savings >=30%;
- zero privileged paired harm;
- opportunity in all four suites and at least 25 tasks.

Only Pi0Fast+OpenVLA-OFT and Pi0.5+OpenVLA-OFT are currently candidates.
SmolVLA may not be promoted without a new opportunity PASS.  OpenVLA-OFT cannot
be paired with itself.

The selector then combines:

- the shared source-risk core;
- a fallback-specific recoverability head;
- a fallback cost-quantile head;
- train-fold-only calibration and conservative decision thresholds.

The source-risk backbone may be shared across VLAs, but recoverability/cost are
conditioned on the corrective arm.  Require per pair, for at least 4/5 seeds:
success gap >= -5pp, false continue <=5%, absolute paired harm <=5%, teacher
savings >=20%, and no suite-concentrated harm.  Independent validation and test
stay sealed until two source/fallback pairs pass.

## R7-E: world-model and OpenVLA-source extensions

World-model features are tested only after the no-WM shared risk model passes.
The two preregistered signals are action-conditioned multi-step latent residual
and dynamics-ensemble disagreement.  They are additive features, never pooled-
latent replacements.  Retain them only if they improve the state-level
AUROC/AP/calibration and downstream selector Pareto frontier for at least two
source VLAs at acceptable real-time latency.

An OpenVLA-OFT source-risk extension is allowed only after R7-C:

1. implement an oracle-backed `SourceContinuation` supporting reset, complete
   source rollout and `return_mode=chunk`;
2. freeze suite checkpoint/config hashes and prevent server-side fallback calls;
3. collect source-only outcomes on the same reset design;
4. pass exact-repeat and label-support audits;
5. use it first as a leave-one-architecture-out risk test;
6. do not run a selector until a distinct fallback passes a model-free audit.

Downloading or training a generic base OpenVLA is not justified before these
gates because it adds major storage/inference cost without resolving the
current Pi0Fast source-risk uncertainty.

## Immediate execution and stop points

1. Let the current Pi0Fast source-only collection finish; do not pause it.
2. Run the already waiting label audit, exact-repeat audit and canonical OOF.
3. If label support fails, stop without training or OFT collection.
4. If source-risk is near random, stop model escalation and report the R7
   representation negative result.
5. If Pi0Fast passes, collect Pi0.5 and SmolVLA source-only cohorts in parallel
   by suite; select qualified cohorts by the frozen balance/stability gates.
6. Train the fixed multi-VLA ladder; do not tune a universal threshold on held-
   out labels.
7. Only after shared risk passes, reopen model-free selector opportunities.
8. Only after no-WM selector success, run the world-model ablation.
9. Only after two policy pairs pass 4/5 seeds, collect independent validation;
   then and only then unseal test and 100+ paired closed-loop episodes.

This plan preserves the original idea while preventing three invalid shortcuts:
pooled base-rate discrimination masquerading as universal risk, a high-failure
policy masquerading as useful training data, and OpenVLA-OFT being counted as
both source and corrective arm.

## Execution update (2026-08-12 evening)

The gate chain is now executable end to end without manually editing policy
names or result paths:

- `audit_r7a_source_labels.py` accepts a declared policy and frozen per-policy
  support thresholds while preserving the Pi0Fast defaults;
- source collection, exact repeat, dataset construction and per-VLA OOF runners
  are parameterized for Pi0Fast, Pi0.5 and SmolVLA;
- `run_r7b_multivla_source_pipeline.sh` refuses to start unless Pi0Fast reaches
  `FULL_PASS`, then independently gates Pi0.5 and SmolVLA at 40 successes, 40
  failures, 12 failure tasks, eight mixed tasks and four samples per class per
  suite;
- `build_r7c_multivla_source_dataset.py` requires identical 192-state keys and
  bit-identical t0 image/proprio/language/task features across policies;
- `train_r7c_multivla_source_risk.py` implements pooled, shared+ID,
  shared+outcome-free-descriptor and shared+per-policy calibration modes;
- `audit_r7c_multivla_stability.py` requires every included policy to pass at
  least 4/5 seeds and remain within 0.03 AUROC of its per-VLA model;
- `train_r7c_lovo_adaptation.py` implements task-held-out leave-one-VLA-out
  zero-shot and hash-selected 0/8/16/32-trajectory behavior adaptation.  The
  descriptor cohort comes only from outer calibration tasks; validation labels
  and outcomes never enter it.  Labeled temperature/bias curves are reported
  separately and only when the selected calibration subset contains both
  classes;
- `audit_r7c_lovo_stability.py` treats zero-shot as a challenge metric and
  requires 32-unlabeled adaptation to reach AUROC >=0.65 and AP gain >=0.05 in
  at least 4/5 seeds.

The independent `tmux:r7_after_gate` watcher is active.  It waits for the
canonical Pi0Fast stability report; a FAIL exits cleanly, while `FULL_PASS`
launches R7-B and the qualified-policy R7-C ladder.  It cannot launch OFT,
selector, world-model, validation or test jobs.

At the latest audit, Pi0Fast source-only collection is 77/192: Spatial 48/48,
Object 29/48, Goal 0/48 and Long 0/48.  Disk free space is 14 GiB.  This is
enough for the source-only NPZ/JSON, exact repeats and OOF reports because model
checkpoints are not dumped per fold.  No frozen data or checkpoint has been
deleted.  Cache/smoke cleanup remains a conditional action only if free space
falls below 8 GiB before an admitted next stage.

The current R7 regression suite passes 19 tests, including same-state merge
rejection, shared calibration and LOVO/adaptation end-to-end smoke.  The LOVO
smoke caught and fixed an allocation bug (`len(data)` counted fields rather
than sample rows) before any formal experiment used the code.

`audit_r7d_selector_readiness.py` makes the risk-to-control boundary explicit.
Even if two per-VLA probes and the shared calibrated model pass, it unlocks
only fresh model-free t0 fallback opportunity audits.  Selector training stays
locked until each concrete source/fallback pair passes that audit.  SmolVLA is
listed as risk-only unless a future SmolVLA/fallback opportunity gate changes
its role.

A non-authoritative partial audit at 80/192 rows was written only to `/tmp`
(never to the watched formal gate path).  It contains 43 successes and 37
failures over 20 tasks, 13 failure tasks and nine mixed-outcome tasks, with zero
data-contract errors.  Spatial is 25 success / 23 failure; the first 32 Object
rows are 18 / 14.  Thus success support and mixed-task support already exceed
their final minima, while failure count and failure-task coverage are close.
Goal and Long have not started, so this is evidence that collection remains
worthwhile, not a prediction that the full four-suite gate will pass.
