# PRE-C0 privileged guidance utilities (follow-up)

Date: 2026-08-04

## Context

Subagent [Build privileged guidance](3a5ca67e-3ca8-41e9-b403-492cf986557e) reached API limit after writing the CPU guidance package and tests, before confirming green tests.

## Completed follow-up

- Fixed `GuidanceResult.__eq__` so numpy action tensors compare deterministically (`rase/guidance/flow_guidance.py`).
- Confirmed guidance unit tests: `tests/test_guidance_trust_region.py`, `tests/test_guidance_determinism.py` (14 passed).
- Added Gate B analyzer `analyze_guided_headroom()` in `rase/collect/pre_c0.py`.
- Added `scripts/run_privileged_guidance_audit.py` (matched-compute Best-of-K ceiling + numerical trust-region probe; no SmolVLA API injection claim).
- Wired `scripts/generate_smolvla_corrective_candidates.py` to `RecedingHorizonSmolVLAContinuation` from `rase/collect/same_policy_corrective.py`.
- Extended `tests/test_pre_c0_protocol.py` for guided gate PASS / frozen-NOGO paths.

## Still pending (not blocked by guidance infra)

- 24-trajectory collection still running under tmux `pre-c0` (no `PRE_C0_COLLECT_EXIT` yet).
- Deviation mining / stage-key QC after collection finishes.
- Natural Gate A smoke + 48-state audit; run privileged guidance audit only if Gate A FAIL.

## Sealed bounds

- PRE-A3 method gate remains closed.
- Hidden PRE-A3 test24 remains sealed.
- World model remains closed.
