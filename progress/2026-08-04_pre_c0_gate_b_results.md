# PRE-C0 Gate B results (offline privileged upper bound)

Date: 2026-08-04

## Trigger

- Natural Gate A decision: `run_privileged_guidance_upper_bound`
- Gate A natural_headroom_pp: `2.083`
- Naming: **privileged trust-region action refinement** / Best-of-K ceiling
- Explicitly **not** SmolVLA flow API guidance

## Offline audit

- Audit: `runs/rase_pre_c0_guided_audit_v1.json`
- Decision: `runs/rase_pre_c0_guided_decision_v1.json`
- Trust-region probe: `artifacts/pre_c0/guidance_trust_region_audit.json`
- Artifacts summary: `artifacts/pre_c0/gate_b_results.json`

## Decision (frozen)

```yaml
decision: frozen_same_policy_recovery_nogo
guided_gain_pp: 0.0
frozen_same_policy_recovery: NOGO
guided_generation_gate: closed
learned_recovery_critic_gate: closed
```

## Interpretation

Offline Best-of-K ranking over already-executed natural `strict_resample` candidates does **not** beat the nested natural oracle (`guided_gain_pp=0`). Ceiling is below the frozen 5pp NOGO threshold.

Per protocol:

- Do **not** expand to closed-loop privileged refinement (B2) in this round
- Do **not** train candidate / recovery critic under frozen same-policy recovery
- Do **not** reopen PRE-A3 / world-model / handback gates
- Next research iteration (out of this round): recovery adapter / LoRA, OFT-to-SmolVLA distillation, or abstention — without changing Gate A/B thresholds post hoc

## Driver

```bash
bash scripts/run_pre_c0_privileged_guidance_audit.sh
```
