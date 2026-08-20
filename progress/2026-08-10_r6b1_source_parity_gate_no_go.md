# R6-B1.0 Source-Parity Gate NO-GO: collector boundary bookkeeping perturbs the source trajectory

Date: 2026-08-10

## Verdict

The frozen two-state Pi0Fast pair-parity diagnostic is **FAILED**. One of the two
R6-A reference source trajectories is not reproduced exactly, so the dynamic
collector is **not frozen**, no dynamic-boundary training data may be collected,
and the R6-B1.1 pilot manifest stays locked.

Artifact: `runs/pre_c0_r6/r6b1_smoke_pi0fast_spatial_pair_parity_v1/`

Reference: `runs/pre_c0_r6/policy_pair_atlas_v1.json`

## Observed parity table (Pi0Fast, libero_spatial, seed-index 0)

| State key | rollout seed | source success | env steps (R6-A) | env steps (B1) | Result |
|---|---|---|---|---|---|
| `sp1_0632d5ef6c45e2f304a01f2c133f0bfe` | 997110703 | failure | 270 | 270 | PASS |
| `sp1_0660d272e7256c6b204caf666e94c875` | 154127683 | success | 116 | **149** | FAIL |

The rollout seed and final success match on both states, but the successful
state's environment step count differs (149 vs 116). The invariant

```text
rollout_seed identical
and source success identical
and source terminal/env steps identical
and saved features all finite
```

therefore fails on the env-steps term, which is exactly the term the first full
pilot (`r6b1_pilot_v3`) also violated (155 vs 116).

## Evidence pointing at boundary bookkeeping rather than native VLA sampling

1. Pi0Fast checkpoint `temperature = 0.0`; `sample_actions_fast` in
   `lerobot/policies/pi0_fast/modeling_pi0_fast.py` decodes greedily with
   `torch.argmax`, so native sampling is deterministic.
2. Pi0.5's flow-matching noise uses the global CUDA RNG, and every policy
   forward is preceded by `seed_everything(...)` in `rase/collect/candidates.py`
   (via `InProcessLeRobotContinuation.reset()`); its single-state parity smoke
   passed.
3. The divergence grows with the amount of in-loop boundary bookkeeping:
   - R6-A source runner (no boundary recording): 116 steps;
   - B1 collector with `--boundary 0`: 149 steps;
   - pilot with `--boundary 0 16 32`: 155 steps.
4. The collector's in-loop bookkeeping performs `single._get_observations(force_update=True)`
   and `raw_libero_to_oracle_arrays(force_update=True)`, which call robosuite
   `_update_observables(force=True)`. That resamples every observable's
   `_current_delay` from the environment RNG and toggles `_sampled`, changing the
   observation schedule the source policy observes on subsequent steps.

Conclusion: the failure mode is **collector-injected side effects during source
rollout**, not VLA-native nondeterminism. This is testable by construction.

## Lock status (unchanged or tightened)

- R6-B1 dynamic collector: **NOT FROZEN**; parity diagnostic failed.
- R6-B1.1 pilot manifest `runs/pre_c0_r6/r6b1_pilot_manifest_v1.json`: **frozen,
  not executed** until pair parity passes.
- World-model residual/disagreement: **LOCKED**.
- Independent validation: **LOCKED**.
- Test and 100+ paired closed-loop episodes: **SEALED**.

## Next step

Bisect the in-loop bookkeeping with a `--bookkeeping-mode` control
(`none` / `snapshot_only` / `obs_only` / `full`) on one Pi0Fast success state
and one Pi0.5 success state, with repeated independent processes, to confirm the
side-effect source before repairing the collector.
