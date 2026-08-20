# R6-D World-Model Residual/Disagreement Ablation — Pre-registration

Date: 2026-08-10

## Status

**PRE-REGISTERED.** The ablation is sealed until the no-world-model R6-C
baseline passes the per-VLA stage gate (≥4/5 seeds on both Pi0Fast and Pi0.5).
This is the only world-model experiment the frozen protocol
(`r6b1_dynamic_boundary_protocol_v1.json` -> `world_model_lock`) allows.

## What R4-D already settled

The V-JEPA pooled-latent **replacement** of the baseline latent is not an
experiment worth repeating. The original AUC drop (0.865 → 0.573) was invalid:
the teacher cache had 1407/1407 identical Student/OFT deltas and a missing
`window_start` field, so it did not test action-conditioned distillation at all
(`progress/2026-08-09_r4d_protocol_reaudit_and_delta_results.md`). Re-audited
baseline AUC 0.9221 still failed the gate (false handback 7.69% > 5%); the true
bottleneck is sample size and calibration, not representation.

Consequence: the world model may only enter the R6-C risk model as an **additional
input feature**, never as a replacement for the action/state/history features.

## Hypothesis

For a source policy that is about to fail irreversibly, the multi-step
action-conditioned prediction under the *source* action should diverge from the
real next latent faster than for a source policy that will succeed.  If true, the
following derived features carry signal beyond the baseline features:

1. `multi-step residual`: MSE between the predicted pooled latent
   `z_{t+k}` (k-step AC rollout conditioned on the recorded source action) and
   the real pooled latent of the time-aligned real frame `z_{t+k}^real`;
2. `ensemble disagreement`: variance of per-step delta directions across the
   K-horizon rollout (an internal forecast-uncertainty proxy).

## Pre-registered protocol

| Item | Value |
|---|---|
| Dataset rows | every B1.2 boundary with a restorable snapshot (pool). One feature row per boundary, aligned to the frozen B1.2 row by `state_key:policy_id:seed_index:elapsed_source_steps`. |
| Real frames | restore the frozen pool snapshot at the boundary, run the *source* policy with the same rollout seed for up to 8 additional steps, record agent-view frames `t+1..t+8`. Time-aligned real latents are `encoder.pooled_latent(frame_{t+k})`. |
| Conditioned actions | the recorded `source_action` at the boundary (and the trace after it) — never the counterfactual OFT action (that is a forbidden future/counterfactual input). |
| Horizon set K | {1, 4, 8} |
| Feature dims | `residual_k` ∈ R^D per k; `disagreement` ∈ R^2 (direction var, magnitude var); `latent_z_t` ∈ R^D. D = V-JEPA 2-AC pooled latent dim. |
| Model input | baseline features + WM features as concatenated auxiliary inputs (same shared backbone, extra linear projection). |
| Training | identical 5-seed task-held-out OOF as R6-C; three task-bootstrap members; 1.64-sigma LCB; two-boundary dwell; per-VLA calibration. |
| Comparison | state-level Pareto vs the no-WM R6-C baseline, same data and seeds. |

## Pre-registered gate (keep the WM arm only if)

1. task-held-out risk ranking improves: per-VLA and pooled AUROC gain ≥ +0.02
   (or a strict dominance across ≥4/5 seeds on at least one VLA);
2. no success-gap regression: success gap ≥ −5pp per VLA;
3. no false-continue regression: false continue ≤ 5%;
4. savings unchanged or better (≥ 20%);
5. inference latency still within the real-time budget (single pooled-latent
   encode + one K-step rollout).

If any criterion fails, the world-model arm is written up as an honest negative
result and is NOT integrated into the deployed selector.

## Scripts

- Feature cache: `scripts/cache_r6d_wm_features.py` (offline; uses the frozen
  V-JEPA 2-AC encoder at `ckpts`/`/root/autodl-tmp/vjepa2`, never in the
  deployment path).
- Ablation training/eval: extends `scripts/train_r6c_candidate_arm_student.py`
  with a `--wm-features <cache.jsonl>` flag; the report must record the same
  metrics schema as R6-C so `audit_r6c_dynamic_stability.py` can aggregate.
- Pareto comparison: `scripts/eval_r6d_wm_ablation.py`.

## Files

- Protocol lock: `configs/r6b1_dynamic_boundary_protocol_v1.json`
- Evidence correction: `progress/2026-08-09_r4d_protocol_reaudit_and_delta_results.md`
