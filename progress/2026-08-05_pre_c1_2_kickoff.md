# PRE-C1.2 kickoff

Date: 2026-08-05

## Inheritance

- PRE-C1.1 decision: `capacity_or_data_review` (recovery 0/9, retention intact)
- Offline teacher only; runtime OFT forbidden
- Gate thresholds unchanged: recovery ≥8pp, retention drop ≤2pp
- PRE-A3 / WM / handback remain closed

## Protocol

- Lock: `artifacts/pre_c1/pre_c1_2_protocol_lock.yaml`
- Phase1 adapter under test: `runs/rase_pre_c1_1_lora_train_v1/adapter_final`
- Recovery comparator: `adapted_minus_base_same_horizon`
- Retention lock: `n_action_steps=10`
- Batch schedule: 9 recovery + 1 clean
- Loss: native flow matching + piecewise horizon weights (mean-normalized); no aux sampled-action MSE
- DAgger: forked teacher query; distinguish `student_query_state` vs `teacher_suffix`

## Execution order

1. Phase 0 successor + restore-repeatability (blocking)
2. Phase 1 same-H horizon sweep + fresh-forward invariants → freeze H
3. Phase 2 student-query OFT-DAgger Round 1
4. Phase 3 train E3 (schedule) → E4 (prefix flow weights) → primary dual gate
5. Phase 4 capacity ladder only if E4 still fails
