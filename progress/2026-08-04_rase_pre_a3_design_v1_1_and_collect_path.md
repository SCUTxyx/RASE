# PRE-A3 design v1.1 + design-driven collect path — 2026-08-04

## Status

**Step 0 complete.** Collection path is ready for the confirmatory 120-state pool.

## Amendments

- Design `runs/rase_pre_a3_design120_v1.1.json`
  - parent sha `db5d12e7…`
  - new sha `ee202c35556c1c3b0ea639b3ea016cd9119e4fb7f62341a91757d235271df375`
  - clean concrete IDs rewritten `000000–000009` → `000001–000010` (adapter contract)
  - no outcomes used; amended before first confirmatory collection
- `rase/collect/pre_a3_schedule.py`: design → `PerturbationRequest`
- `pipeline.collect`: protocol `rase-pre-a3-recovery120/v1`
- Plus adapter honors explicit `task_id` for camera/robot
- Keys join by `episode_id`

## Smoke

2-episode real collect on smoke design subset produced step-0 snapshots for:

- Spatial clean
- Spatial camera L1

## Next

Full 120 collect in tmux, then freeze `runs/rase_pre_a3_keys120_v1.json`.
