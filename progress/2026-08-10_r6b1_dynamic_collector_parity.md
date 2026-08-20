# R6-B1 Dynamic Boundary Collector Parity Smoke

Date: 2026-08-10

## Result: PASS

The R6-B1 collector now reproduces the frozen R6-A source trajectory exactly
for both qualified source VLAs before writing any scientific dynamic-boundary
data. Two earlier engineering-only smokes used a stage-specific salt or an
incorrect relative horizon and are explicitly excluded from the dataset.

| Source policy | Suite | Frozen R6-A seed | R6-A source outcome | R6-B1 outcome | t=0 persistent OFT |
|---|---|---:|---|---|---|
| Pi0Fast | Spatial | 1096176436 | failure, 270 steps | failure, 270 steps | success, 79 steps |
| Pi0.5 | Spatial | 1103197255 | failure, 270 steps | failure, 270 steps | success, 79 steps |

The Pi0Fast run also collected a t=16 boundary: the source remained a failure
and persistent OFT rescued from that exact snapshot in 81 steps. Images,
proprioception, source/OFT actions, and canonical action summaries are finite
and change over time.

## Corrections made before acceptance

1. Reused the exact R6-A seed formula rather than introducing a B1-specific
   salt.
2. Wrapped each counterfactual branch in Python/NumPy/Torch/CUDA RNG state
   preservation. Simulator restore must not affect the main source trajectory.
3. Applied the simulator's absolute horizon (`current_timestep < horizon`),
   matching the R6-A rollout implementation. The saved states begin at timestep
   10, so using a relative 280-step horizon incorrectly produced 280 rather
   than 270 source actions.

## Verified collector outputs

- `scripts/collect_r6b1_dynamic_boundaries.py`
- `configs/r6b1_dynamic_boundary_protocol_v1.json`
- `runs/pre_c0_r6/r6b1_smoke_pi0fast_spatial_parity_v4/`
- `runs/pre_c0_r6/r6b1_smoke_pi05_spatial_parity_v2/`

The collector records a true trajectory group ID, observations/actions available
at deployment, and source-final/future/persistent-takeover outcomes as
simulator-only labels. It does not use a world-model feature.

## Next stage

Freeze an outcome-balanced cross-suite pilot manifest from the R6-A source
summaries. Run both qualified VLAs across all four suites at
`h={0,16,32}`. Only after the pilot has source-success and source-failure
support, persistent counterfactual parity, and non-empty later boundaries will
the full grouped R6-B1 collection be authorized.
