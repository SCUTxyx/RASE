# RASE Phase A/B CPU-only scaffold

**Date:** 2026-08-15  
**Scope:** Additive files only; frozen confirmation runner, manifest, protocol, collectors, and output are unchanged.

## Added Phase A framework

- `rase/vnext/phase_a_audit.py`
  - frozen manifest/row integrity;
  - contract-mask agreement;
  - exact-K replica coverage;
  - finite raw metric checks;
  - full and non-abort strict opportunity reports;
  - physical-root winner denominator;
  - actual winner-root task/suite coverage;
  - unique/co-best and configurable tie margin;
  - paired-trial soft preference evidence;
  - `A_PASS`, `A_PARTIAL`, `A_FAIL`, and `INTEGRITY_FAIL` verdicts.
- `scripts/audit_rase_vnext_phase_a.py`
  - read-only JSONL CLI;
  - input hashes embedded in output;
  - exit 0 only for `A_PASS`, 2 for integrity failure, 3 otherwise.
- `scripts/run_rase_vnext_phase_a_after_complete.sh`
  - refuses to run before `confirmation_v1/COMPLETE.json` exists;
  - writes only to `runs/rase_vnext/phase_a_v1`;
  - uses exact-tie formal audit (`tie_margin=0`) and the frozen bootstrap seed.

## Added Phase B framework

- `rase/vnext/motion_trace.py`
  - conservative semantic mapping from `CanonicalActionToken`;
  - physical EE delta and integrated xyz+xyzw pose;
  - base-frame and EEF-frame translation support;
  - velocity, acceleration, jerk and derivative-valid order;
  - path length and direction reversals;
  - gripper state/event masks;
  - previous/current chunk cosine consistency;
  - optional workspace margin and camera projection;
  - missing physical dimensions are masked, never invented.
- `rase/vnext/adapter_parity.py`
  - raw/canonical action round-trip audit;
  - MotionTrace conversion audit;
  - empirical resample diversity audit;
  - declared-versus-observed operator capability report.

## Tests

- `tests/test_vnext_phase_a_audit.py`
- `tests/test_vnext_motion_trace.py`

Server verification:

```text
24 focused tests passed
39 complete test_vnext_* tests passed
```

The partial confirmation integrity dry-run correctly returned `FAIL` because the schedule is incomplete. It found no row/manifest contract mismatches; it did not compute a scientific opportunity verdict.

## Locked actions

- Do not execute `run_rase_vnext_phase_a_after_complete.sh` until `COMPLETE.json` exists.
- Do not train MotionTrace/action-semantic models until the Phase A verdict unlocks them.
- Do not modify the current confirmation manifest, protocol, K, roots, operators, costs, or seeds.
