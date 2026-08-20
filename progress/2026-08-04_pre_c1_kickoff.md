# PRE-C1 kickoff: OFT→SmolVLA Recovery LoRA

Date: 2026-08-04

## Trigger (frozen predecessors)

| Artifact | Decision |
|----------|----------|
| `runs/rase_pre_c0_decision_v1.json` | `run_privileged_guidance_upper_bound` (`natural_headroom_pp≈2.08`) |
| `runs/rase_pre_c0_guided_decision_v1.json` | `frozen_same_policy_recovery_nogo` (`guided_gain_pp=0`) |

Interpretation: frozen SmolVLA same-policy candidate space lacks recovery support. PRE-C1 moves to **same-backbone recovery competence** via offline OFT teacher → recovery LoRA.

## Gate status

```yaml
recovery_adapter_gate: open_for_audit
learned_recovery_critic_gate: closed   # Gate B NOGO
candidate_critic_gate: closed          # Gate A FAIL
world_model_gate: closed
pre_a3_method_gate: closed
hidden_test24: sealed
```

## Method naming

- **recovery LoRA / OFT action distillation**
- Runtime does **not** call OFT
- Not SmolVLA flow-API guidance
- Not a runtime policy switch to OFT

## Protocol lock

`artifacts/pre_c1/pre_c1_protocol_lock.yaml`

Pre-registered gates:

- `recovery_gain_pp >= 8`
- `clean_retention_drop_pp <= 2`
- episode-cluster bootstrap CI lower bound `> 0`

## Next

1. Build distill dataset (OFT teacher chunks on PRE-C0 failure states + clean retention)
2. Train recovery LoRA
3. Eval dual gate → PASS / abstention / capacity review
