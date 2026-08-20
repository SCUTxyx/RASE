# PRE-C0-R2 Gate Data Scaling + Conditional Evaluation

**Date:** 2026-08-08  
**Status:** COMPLETE — `TRAIN-GATE-FAIL`; learned-gate dev arm stopped by gate  
**Research question:** after task-disjoint label scaling, can deployment-observable lean features predict when the spatial F0 correction rescues rather than harms, and improve over the same deterministic safety envelope?

## Identity and execution

- Server root: `/root/autodl-tmp/RASE`
- Git base: `8d8212643893935368152efce4bc571bb83b59db` (dirty working tree; R2 files were uncommitted)
- Protocol: `runs/route_c_final/protocol_frozen.json`
  - SHA-256: `fb85cda2b6c58ab9901bc51f76a6b790fa773988aca646ac4e58f38e55a20237`
- Student model identity is frozen by the protocol.
- F0 vector: `runs/pre_c0_r0/f0_constant_vector.json`
  - SHA-256: `efee1a5aae51f4014f77438d1a1225749a8c27116e6904fd660bdaa5c5a344a3`
- F0 plugin checkpoint: `runs/route_c_controls/F0/plugin_best.pt`
  - SHA-256: `b781eb67576f7693ebf5c5cf6bb0f929154ef92101823cab877bf083cc81da1d`
- Diagnostic gate checkpoint: `runs/pre_c0_r2/gate_checkpoint.pt`
  - SHA-256: `371d8e4d7d9201da03ece01a9ec7df1aec17e46c395587e616ec6120757779a1`
  - This checkpoint failed the training gate and is not approved for deployment.
- Environment lock SHA-256: `b3e18d916dee8941105ed26ad10a2523fc2a82c02e8835478023682b6501adbb`
- Python 3.12.13; PyTorch 2.10.0+cu128; NumPy 2.2.6.
- Execution: `tmux 0`, RTX 5090; GPU idle after completion.

## Why the original R2 plan was amended

The original collector used `protocol["splits"][suite]["dev"]`; the final 40-episode evaluation used the same two dev tasks and overlapping seed construction. Snapshot-level random train/validation splitting also placed correlated snapshots from the same source episode on both sides.

The following issues were fixed before R2:

1. activation labels now come from `train` tasks, not dev tasks;
2. validation is leave-one-task-out, with episode-grouped fallback;
3. raw validation accuracy is not a gate because an always-off model already gets 90.3% on R2;
4. the deployment threshold is selected from OOF predictions using rescue/harm utility;
5. collector/eval lean context length and progress delta definitions are aligned;
6. collector and gate eval use the same constant F0 delta instead of training with a constant and evaluating with a history-dependent F0 output;
7. `takeover_steps` now counts executed corrected action steps; entry count is separate;
8. OOF predictions and the complete threshold sweep are saved.

Original scripts were preserved as `scripts/*.20260808_pre_c0_r2.bak`.

## Step 1: task-disjoint label scaling

Resolved configuration:

```yaml
suite: libero_spatial
source_split: train
task_limit: 4
tasks_observed_before_snapshot_cap:
  - libero_spatial_000001
  - libero_spatial_000003
  - libero_spatial_000005
n_episodes_per_task: 8
snapshot_limit_requested: 60
snapshots_written: 62
source_episode_groups: 21
max_snapshots_per_episode: 3
seed: 20260808
dev_high: 0.15
dev_low: 0.05
dev_recover: 0.10
feature_schema: rase-activation-gate-lean/v2
```

The count exceeded 60 by two because the stop check occurs between source episodes; the final included episode contributed multiple snapshots. No outcomes were removed post hoc.

### Counterfactual outcome matrix

| Label type | Count | Fraction |
|---|---:|---:|
| Rescue: Base fail, F0 success | 6 | 9.7% |
| Harm: Base success, F0 fail | 3 | 4.8% |
| Neutral: both success | 30 | 48.4% |
| Both fail | 23 | 37.1% |
| Total | 62 | 100% |

Aggregate snapshot continuations:

- Base: 33/62 = 53.2%
- F0: 36/62 = 58.1%
- Oracle(Base,F0): 39/62 = 62.9%
- `H_activation = 39/62 - 36/62 = 3/62 = 4.84pp`

This task-disjoint estimate is below the frozen 5pp opportunity threshold and substantially below R1's dev-selected 12.5pp estimate.

Per task:

| Task | N | Rescue | Harm | Neutral | Both fail |
|---|---:|---:|---:|---:|---:|
| spatial_000001 | 23 | 1 | 0 | 16 | 6 |
| spatial_000003 | 24 | 3 | 3 | 10 | 8 |
| spatial_000005 | 15 | 2 | 0 | 4 | 9 |

All observed harm is concentrated in one task, while rescue exists in all three. This is already a sign that the activation boundary is task-dependent.

## Step 2: grouped OOF gate training

Resolved training configuration:

```yaml
model: LeanActivationGate
hidden_dim: 16
epochs_max: 200
learning_rate: 0.001
patience: 35
weight_decay: 0.001
seed: 20260808
validation: leave_one_task_out
harm_sample_weight: 2.0
threshold_constraints:
  rescue_recall_min: 0.50
  harm_activation_rate_max: 0.25
```

OOF results:

- positive prevalence: 0.0968;
- average precision: 0.2937;
- ROC-AUC: 0.6458;
- two folds selected epoch 0; the third selected epoch 126;
- zero thresholds met both recall and harm constraints.

The rescue and harm distributions overlap:

| Type | N | OOF probability min / mean / max |
|---|---:|---:|
| rescue | 6 | 0.355 / 0.476 / 0.636 |
| harm | 3 | 0.440 / 0.468 / 0.488 |
| neutral | 30 | 0.234 / 0.425 / 0.532 |
| both fail | 23 | 0.173 / 0.431 / 0.628 |

Trade-off:

- enforcing harm activation ≤25% gives threshold 0.63, rescue recall 1/6=16.7%, and activation rate 1/62=1.6%; this is effectively always-off;
- obtaining rescue recall ≥50% requires threshold 0.45, which recalls 5/6 rescues but activates 2/3 harms and 45.2% of snapshots; utility is non-positive.

Decision: `TRAIN-GATE-FAIL`. The failure is not just insufficient accuracy; the available lean features do not separate rescue from harm across held-out tasks.

## Step 3: conditional safety-envelope evaluation

The learned-gate 40-episode arm was intentionally not run after the training gate failed.

The gate-independent `Envelope-only` control was run because it is the deterministic fallback and was not yet isolated with the same new runner. All arms use the exact same 40-key spatial dev manifest.

| Arm | Success |
|---|---:|
| B0 | 27/40 = 67.5% |
| Legacy bounded F0 | 32/40 = 80.0% |
| New envelope-only constant F0 | 31/40 = 77.5% |
| Learned gate + envelope | not run: training gate failed |

Key-based paired comparisons:

- Envelope-only vs B0: rescue 7, harm 3, net +4/40 = +10.0pp; exact McNemar p=0.3438.
- Envelope-only vs legacy bounded: rescue 4, harm 5, net -1/40 = -2.5pp.

Intervention burden for envelope-only:

- mean: 88.0 corrected action steps/episode;
- successful episodes: mean 59.9;
- failed episodes: mean 185.0.

Envelope-only gives a positive dev point estimate over B0 but is not statistically confirmed and intervenes heavily, especially on failures. Legacy bounded F0 remains the best observed spatial engineering baseline by one episode.

## Final decision

```yaml
pre_c0_r2: complete
activation_opportunity_gate: fail_marginal_4.84pp
learned_gate_training: fail
learned_gate_dev_eval: stopped_by_gate
envelope_only_dev: 31/40
recommended_spatial_fallback: legacy_bounded_f0
cross_suite_deployment: prohibited
world_model_gate: closed
```

Do not respond by increasing hidden size or repeatedly tuning threshold on spatial dev. The limitation is semantic: current lean features assign almost identical scores to rescue and harm states.

## Artifacts

```text
runs/pre_c0_r2/
  activation_labels.jsonl
  pipeline.log
  oft_spatial.log
  gate_checkpoint.pt                  # diagnostic, failed gate
  gate_training_report.json           # includes 62 OOF predictions + 91 thresholds
  gate_training_rerun.log
  decision.txt
  eval_envelope_only/
    paired_results_b3.jsonl
  paired_results_envelope_only.jsonl
  summary.json
  summary.md
```

## Recommended next stage

1. Freeze the learned activation-gate branch as `NO-GO`; do not collect more of the same feature/label pairs unless a power analysis and new signal are specified.
2. Keep legacy bounded F0 only as a spatial diagnostic/baseline, not a universal method. The spatial vector is known to oppose object corrections.
3. If pursuing a learned decision model, change the target and information, not only scale:
   - predict counterfactual intervention utility (`rescue`, `harm`, `neutral`) rather than a binary rescue label;
   - add task/object-relative progress and contact/grasp state, or a validated policy-risk signal;
   - evaluate failure probability and recoverability as separate heads;
   - query only at deployment-defined candidate boundaries rather than all frames.
4. Before a new model branch, first verify that the new feature set separates rescue from harm on task-held-out OOF data and that oracle headroom exceeds 5pp.
5. For the current paper, this result strengthens a diagnosis: a simple directional correction can help spatial dev, but state-conditioned activation does not generalize from small counterfactual data, and intervention burden remains high.
