# RASE-UI Phase 0H suffix-prefix mechanism audit kickoff

Date: 2026-08-01 22:03 CST  
Status: **RUNNING — exploratory selected-subset mechanism audit**

## Research question and claim boundary

Phase 0G found four immediate-only and two deferred-only OFT states but missed
the preregistered aggregate opportunity gate. Phase 0H asks only whether those
six frozen disagreements have a stable causal boundary as progressively more of
the already-active action suffix is executed before OFT takes control.

This cohort is selected using Phase 0G outcomes. Therefore Phase 0H cannot
estimate population prevalence, validate a selector, or reopen model training.
It is a mechanism audit whose only admissible next decision is whether a new,
independent targeted screen is scientifically justified.

## Frozen intervention grid and stop rules

- Source analysis: `runs/rase_ui_phase0g_independent48_analysis_v2.json`.
- Cohort: exactly the Phase 0G four `direct_only` plus two `deferred_only`
  states; no additions or removals after interior-prefix outcomes are observed.
- Prefix source: exact `decision_context.active_action_suffix` stored in each
  Phase 0G snapshot.
- Arms: every prefix length `k=0,1,2,3,4,5`, followed by the same OFT
  continuation.
- `k=0` must match the Phase 0G immediate arm in both prefix SHA-256 and success.
- `k=5` must match the Phase 0G deferred arm in both prefix SHA-256 and success.
- Any endpoint mismatch invalidates interpretation of the interior curve.
- A one-flip curve is a state-level stable boundary; more than one flip is
  classified as non-monotonic. No threshold is tuned to maximize these results.

## Identity, environment, and configuration

- Base Git commit: `454f76384e5195a750584dd9753c29b0701bb6af`.
- Config: `configs/collect_rase_ui_phase0g_independent48.json`.
- Pool: `runs/rase_ui_phase0g_independent48_pool/`.
- Suites present in the frozen subset: Spatial, Goal, Long; Object is empty and
  is not launched.
- Six unique states, tasks, and episodes inherited from the Phase 0G frozen
  design; one deterministic continuation per arm.
- Phase 0G design/collection seed: `2026081807`.
- SmolVLA client: Python 3.12.13, NumPy 2.2.6, PyTorch 2.10.0+cu128.
- OFT checkpoint remains frozen and suite-specific under `ckpts/oft_*`; no
  parameters are trained.
- Base `env.lock.md` SHA-256:
  `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2`.

## Code added before outcome access

- `freeze_timing_disagreement_keys.py`: freezes the disclosed outcome-selected
  cohort and source-analysis checksum.
- `prefix_ablation.py`: constructs and validates every exact suffix prefix.
- `rollout_oft_prefix_ablation.py`: adds the resumable
  `suffix-prefix-grid` protocol plus action-geometry descriptors.
- `analyze_suffix_prefix_mechanism.py`: enforces endpoint success/SHA parity,
  classifies curve flips, and emits JSON and Markdown reports.
- `run_rase_ui_phase0h_suffix_prefix6.sh`: suite-serial, resumable entrypoint.

Pre-run validation: 25 relevant tests passed; Ruff, bash syntax, and
`git diff --check` passed.

## Frozen scientific decision

If the six curves share an interpretable, outcome-independent boundary feature,
Phase 0H may motivate a separately preregistered task/episode-disjoint targeted
screen. If curves are non-monotonic or state-specific, close the timing-selector
direction and retain immediate OFT as the fixed deployment baseline. Neither
outcome authorizes world-model training.
