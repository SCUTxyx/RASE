# PRE-C1.1 kickoff: long-horizon successful OFT distillation

Date: 2026-08-04

## Why

PRE-C1 LoRA train loss fell but `recovery_gain_pp=0`. Root cause (frozen):

| Item | Value |
|------|------:|
| Teacher states | 42 |
| Teacher steps | **10** for all |
| Teacher `rollout_success` | **0 / 42** |
| Val adapted successes | **0 / 10** |

PRE-A3 showed OFT recovery concentrates at **h≈96–128**. Distilling short *failed* prefixes cannot teach recovery.

## Protocol

- Lock: `artifacts/pre_c1/pre_c1_1_protocol_lock.yaml`
- `teacher_horizon_steps: 128`
- `keep_only_successful_oft: true`
- `record_every_oft_chunk: true`
- Gate thresholds **unchanged**: recovery ≥8pp, retention drop ≤2pp

## Naming

recovery LoRA / OFT action distillation — offline teacher only; not runtime OFT; not flow-API guidance.
