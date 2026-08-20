# R7-A: independent-episode source-risk restart

Date: 2026-08-12

Outcome update (2026-08-13): the exclusion-bound 191-state reproducibility
gate passed, but canonical source-risk OOF failed 0/5 seeds and froze
`STOP_SOURCE_RISK_ESCALATION`.  See
`progress/2026-08-13_r7a_191_exact_repeat_and_source_risk_oof_negative.md`.

## Decision

Do not make another selector architecture change on the frozen R6 split.  R6-C.1
is a canonical negative result.  Restart only the scientifically supported part
of the idea: a lightweight, VLA-conditioned, real-time source-risk model.  Arm
recoverability, selector control, shared multi-VLA training and world-model
features remain gated behind source-risk transfer on genuinely independent
episodes.

## Evidence for the restart

The corrected R6 audit makes two facts coexist:

- the deployed controller fails badly: mean success gap -11.67pp, corrected
  paired harm 13.54%, false-continue 13.42%, savings 21.90%, 0/5 seeds;
- a privileged t0 controller that knows only whether the source will fail gets
  success gap +5.21pp, zero paired harm and 34.63% OFT savings.

No post-hoc controller family on the existing predictions satisfies the formal
gate.  The safest cross-fit t0 rule has success gap -2.08pp and harm 2.08%, but
only 7.51% savings.  Thus threshold, LCB, dwell and advantage-head tuning cannot
recover the missing 20% cost target.  The remaining opportunity specifically
requires better source-failure representation.

The old 740-snapshot pool is not 740 independent episodes: it contains exactly
one episode/seed per task and several temporal snapshots of that episode.  That
is the central data defect.  More snapshots or replicas do not increase
task-held-out state diversity.

## R7-A0: frozen protocol and reset pool

- Freeze 48 tasks x 4 independent LIBERO init states = 192 reset episodes.
- Keep all four episodes of a task in one outer fold.
- Select states before outcomes; never use reset-pool `episode_outcome` as a
  label.  The legacy pool schema requires an outcome token, so reset-only
  collection writes an explicit placeholder and downstream audit rejects it as
  supervision.
- Capture at step 0 and execute zero source actions.  This does not load the
  Pi0Fast checkpoint and is idempotently resumable.
- Frozen design:
  `runs/pre_c0_r7/r7a_multi_episode_design_v1.json`, design SHA256
  `9b78cf470e1c659d26b3789c5e113a6f51470c4e6a950cb146fc87c8c207f72f`.

Gate: exactly 192 readable, unique step-0 states; 48 tasks; four distinct init
states per task; exact task/init/seed/design provenance; all suites represented.

Status: **PASS**.  Real collection produced 192/192 readable unique states and
the frozen state-key manifest has SHA256
`3a083717bbcc4786264c716820e83a31c7fddc28cb9b31dfac89822f194ed81a`.

The first real rollout exposed a legacy-schema restore incompatibility.  V1
StatePool records contain flattened simulator state but no raw `mujoco_data`,
while newer records may contain controller runtime caches.  Constructing a
hybrid snapshot made robosuite reject the controller keys.  The compatibility
fix is deliberately narrow: `bundle_to_env_snapshot` deep-copies the in-memory
robot controller payload and removes only the runtime-cache fields when raw
MuJoCo data is unavailable.  Stored pool bytes, hashes and provenance are not
rewritten.  A real restore audit now passes 12/12 cells (four suites x clean,
camera and robot perturbations), all at the expected post-reset stabilization
timestep 10.

## R7-A1: true source labels before expensive counterfactuals

Run Pi0Fast once from every frozen reset state.  Collect only the deployable t0
features and final source outcome; do not call OFT.  The dataset contains the
two camera views, proprioception, proposed source action/chunk summary,
instruction/task metadata and final failure label.

Label-support gate, frozen before outcomes:

- 192 complete rows and 48 task clusters;
- at least 48 source failures and 32 source successes;
- failures in at least 16 tasks;
- at least eight tasks whose four independent init states contain both labels;
- at least four failures and four successes in every suite.

A PASS unlocks only a source-risk representation probe and a hash-selected
exact-repeat stability audit.  It does not unlock OFT labels or selector
training.

## R7-A2: dedicated source-risk representation probe

Use five task-held-out folds and task-bootstrap confidence intervals.  Train a
single failure target first; do not share its trunk with persistent-success,
advantage or cost losses in this stage.

Pre-register the following ladder:

1. canonical lightweight baseline: reset images + proprio + proposed Pi0Fast
   action summary + task text embedding;
2. policy-native adapter: frozen Pi0Fast intermediate visual/action features
   projected into the same small risk core;
3. optional world-model add-on: only action-conditioned multi-step residual and
   ensemble disagreement, concatenated as features.  Never replace the
   discriminative state representation with a pooled V-JEPA latent.

Representation gate:

- task-held-out AUROC >= 0.75;
- task-bootstrap AUROC 95% lower bound >= 0.65;
- AP above prevalence by at least 0.10;
- ECE <= 0.10 after train-fold-only calibration;
- all suites AUROC above 0.60;
- at least 4/5 training seeds satisfy the point gates.

World-model features remain only if they improve task-held-out AUROC or AP and
the controller Pareto frontier without degrading calibration, on both source
VLAs once a second VLA is admitted.  Otherwise they are a negative ablation.

The canonical probe implementation is now frozen before labels complete.  It
uses five suite-balanced task folds, three task-bootstrap members, an
outer-train task-disjoint calibration subset, Platt temperature/bias fitting,
and task-cluster bootstrap intervals.  Five training seeds are fixed to
`{2026081207,...,2026081211}`; stability requires at least four seeds to pass
all point gates.  The runner is not connected to the collection driver and
will refuse to start unless the 192-row label-support audit and bound dataset
hash both pass.

The representation stop rule is also frozen before outcomes.  If the canonical
model passes at least four seeds, run the policy-native adapter as a planned
ablation and unlock a new-cohort t0 OFT opportunity audit.  If it misses the
full gate but at least three seeds retain AUROC >=0.65 and AP at least 0.05
above prevalence (with mean AUROC >=0.65), allow exactly one frozen
policy-native additive-adapter attempt.  Otherwise stop source-risk escalation.
The world model cannot be used to rescue a near-random canonical/native result.

Before model fitting, a second source-only reproducibility gate now reruns a
hash-selected 16-state subset (two successes and two failures per suite) with
the exact same rollout seeds.  Outcome, terminal step, stop reason, t0 features
and the full action trace must agree.  The repeat stage and dataset builder are
serialized with file locks and content-hash checks because two independent
drivers may reach the gate; duplicate execution cannot create a second
training cohort.

The canonical action input was also corrected before dataset construction.
Pi0Fast is configured with ten queued action steps, whereas the first builder
used only the first 7-DoF command.  R7 now summarizes
`source_action_trace[0:10]`, which is the causal initial proposal produced by
the first forward pass.  The one-step summary remains in the immutable dataset
as an ablation.  No recollection, outcome, future observation or OFT signal is
used by this reconstruction.

## R7-A3: counterfactual and selector gates

Only after the source-risk representation gate:

1. collect t0 persistent-OFT outcomes first; add t8/t16 only if the t0
   controller cannot reach the cost target and the model-free oracle proves
   later entry useful;
2. require the new natural cohort's privileged source-risk oracle to have
   success gap >= -5pp, savings >=30%, harm 0, all suites and >=25 tasks;
3. train one prespecified within-Pi0Fast t0 selector with thresholds calibrated
   inside training folds;
4. require 4/5 seeds with success gap >= -5pp, false-continue <=5%, corrected
   absolute paired harm <=5% and savings >=20%;
5. only then add a second source VLA and compare per-VLA, shared+VLA descriptor,
   small calibration adapter, few-shot curve and leave-one-VLA-out.  Pure
   zero-shot is a challenge metric, not the main gate.

Independent validation, test and 100+ paired closed-loop evaluation stay sealed
until two policy pairs pass.  R7 recovery-student LoRA+DAgger is complementary
but remains a separate project and cannot retroactively rescue R6.

## Implemented artifacts

- `rase/collect/r7_schedule.py`
- `rase/risk/r7_source_protocol.py`
- `scripts/freeze_r7a_multi_episode_design.py`
- `scripts/freeze_r7a_reset_keys.py`
- `scripts/audit_r7a_restore_compat.py`
- `scripts/run_r7a_source_labels.sh`
- `scripts/audit_r7a_source_labels.py`
- `scripts/build_r7a_source_risk_dataset.py`
- `scripts/freeze_r7a_exact_repeat_manifest.py`
- `scripts/audit_r7a_exact_repeat.py`
- `scripts/run_r7a_exact_repeat.sh`
- `scripts/run_r7a_build_source_dataset.sh`
- `scripts/train_r7a_source_risk_probe.py`
- `scripts/run_r7a_source_risk_oof.sh`
- `scripts/audit_r7a_source_risk_stability.py`
- `scripts/run_r7a_after_reset.sh`
- `configs/r7a_pi0fast_reset_pool_v1.json`
- `scripts/audit_r6c1_posthoc_pareto.py`
- `rase/risk/light_risk_student.py::SourceRiskStudent` (single target;
  policy-native and WM streams are additive-only)
- `rase/risk/multi_vla_descriptor.py` (outcome-free action/state behavior
  descriptor for future multi-VLA transfer)

Reset-state collection, reset-key provenance audit and the 12-cell real-restore
audit are complete and PASS.  Pi0Fast source-only labeling is running in tmux
session `r7a_pipeline`.  The driver stops after label-support audit and dataset
construction; it cannot automatically start OFT, selector, WM or validation
experiments.

End-to-end data validation has also passed on the first real outputs.  Each
record contains one t0 row, two RGB views `(2,3,96,96)`, proprio `(8,)`, a
canonical source-action summary `(20,)`, final source outcome and no OFT arrays
or labels.  `obs_recorded=false` means the live source trajectory was not
perturbed by an in-loop forced observation refresh; features are captured
post-hoc from the frozen t0 branch snapshot and are present in the NPZ.  The
final audit now checks these shapes, metadata equality, hashes, action-trace
length and absence of OFT leakage.

One model-interface defect found by this real sample was fixed before training:
`SourceRiskStudent` now encodes the agent and wrist views separately with the
shared encoder and averages their embeddings.  Previously the generic encoder
would have interpreted the two-camera axis as time and produced an incompatible
rank at feature fusion.  A 192-row/48-task/five-fold synthetic end-to-end smoke
passes; synthetic metrics are execution checks only and are not evidence.

## 2026-08-12 deterministic Pi0Fast decoder failure and recovery

The source-only driver stopped after 138/192 completed states because Pi0Fast's
FAST detokenizer asserted that a generated sequence did not begin with the
required `Action :` grammar.  Disk space, GPU memory, tmux and simulator restore
were ruled out.  An isolated same-checkpoint/same-seed reproduction on
`sp1_53e7e397b73fb24d71ac47c0fc9e3efd` failed identically at source step 20
(simulator timestep 30), proving that this is a deterministic source-policy
runtime failure rather than a transient collector error.

The collector now fails closed before ten valid source actions, because no
complete causal t0 proposal would exist.  At or after ten actions it records a
structured `policy_inference_error`, labels the source outcome as failure and
preserves the real action trace and t0 features.  This matches deployment
semantics without changing seed, retrying into a different policy, or inventing
an action.  The label audit validates the structured error contract and reports
the count separately.  The repaired state produced a 20x7 action trace, one
two-camera t0 feature row, and a valid hashed NPZ; 12 targeted regression tests
pass.  Formal collection resumed from the existing 138 records with 54 states
remaining in tmux `r7a_pipeline`.
