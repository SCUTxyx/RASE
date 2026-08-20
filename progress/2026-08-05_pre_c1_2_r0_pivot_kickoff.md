# PRE-C1.2 R0 Pivot Kickoff

**Date:** 2026-08-05  
**Status:** Engineering carrier kept; legacy E3/E4 paused; R0 path implemented

## Decision

Do **not** wipe the table and do **not** run original C1.2 Phase 3 full OFT-action BC after DAgger R1.  
Keep the running student-query DAgger Round 1, then switch the objective through R0 diagnostics.

## Implemented controls

1. **Hard stop before legacy E3/E4**
   - [`scripts/run_pre_c1_2_train_eval.sh`](../scripts/run_pre_c1_2_train_eval.sh) exits `42` unless `ALLOW_LEGACY_E3_E4=1` or unlock file exists.
   - This also protects the already-running parent pipeline (it re-reads this script from disk).
   - Capacity ladder similarly blocked in [`scripts/run_pre_c1_2_capacity_ladder.py`](../scripts/run_pre_c1_2_capacity_ladder.py).
2. **Pipeline path revision**
   - [`scripts/run_pre_c1_2_full_pipeline.sh`](../scripts/run_pre_c1_2_full_pipeline.sh): R1 → global QC → R0; legacy E3/E4 only if explicitly unlocked.
   - Continuation helper for the in-flight run: [`scripts/run_pre_c1_2_after_dagger_r0.sh`](../scripts/run_pre_c1_2_after_dagger_r0.sh).
3. **Protocol revision**
   - [`artifacts/pre_c1/pre_c1_2_protocol_lock.yaml`](../artifacts/pre_c1/pre_c1_2_protocol_lock.yaml) adds `revision`, `r0`, `revised_training`.
   - Preserves E0/E1/R1 provenance hash `87916082ceacf3415bc28c40b0d968c86f5b36924fd05e224907994504c57790`.
   - Frozen `H=2` unchanged.

## New diagnostics / training entrypoints

| Step | Script |
|---|---|
| Global DAgger QC | `scripts/analyze_pre_c1_2_dagger_global_qc.py` |
| Teacher-forced fit | `scripts/eval_pre_c1_2_teacher_forced_fit.py` |
| One-step + R(k) | `scripts/eval_pre_c1_2_student_prefix_teacher_handover.py` |
| R0 decision | `scripts/analyze_pre_c1_2_r0.py` |
| R0 runner | `scripts/run_pre_c1_2_r0.sh` |
| Revised dataset | `scripts/build_pre_c1_2_revised_dataset.py` |
| Revised train | `scripts/train_smolvla_recovery_lora_c1_2_revised.py` |
| Revised runner | `scripts/run_pre_c1_2_revised_train.sh` |

## Current live state

- DAgger R1 still collecting (Object nearly/fully done; Goal/Long remaining at kickoff).
- Partial global QC already runnable from root run summaries; final numbers freeze only after 9/9 anchors × 5 seeds.
- Legacy E3/E4 block verified (`exit 42`).

## Next

1. Let DAgger R1 finish.
2. Build dataset + freeze global QC.
3. Run R0-A/B/C and emit branch decision.
4. Only then run revised short-horizon training (not legacy E3/E4).
