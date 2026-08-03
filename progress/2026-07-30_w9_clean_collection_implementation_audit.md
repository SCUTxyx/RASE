# W9 clean-collection implementation audit

## Status

**Audit complete. No new collection. W9 pipeline not restarted.**

Pool `pool/ngc_w9_clean_controls` is marked
`diagnostic_invalid_for_control` (retained on disk). Selector Gate W9-C must
not start on this pool.

## Environment

- Repo: `/root/autodl-tmp/RASE` @ `ea7ad403…`
- LeRobot LiberoEnv:
  `/root/autodl-tmp/envs/smolvla/lib/python3.12/site-packages/lerobot/envs/libero.py`
- `git add -N` applied to W9 untracked sources; `git diff --check` → **clean**
  (exit 0)

## 1. Diff hygiene

Intent-to-add covered W9 configs, selector package, pipeline/scripts, focused
tests, and W9 progress notes. `git diff --check` reported no whitespace errors.

## 2. Fixed `episode_index=0` (root cause)

In `rase/collect/lerobot_libero_plus_adapter.py` every episode builds:

```python
LiberoEnv(..., init_states=True, episode_index=0, n_envs=1, ...)
```

`rase/collect/libero_env_factory.py` uses the same hardcode.

### LeRobot mapping (installed source)

```text
self.init_state_id = self.episode_index
# on reset:
set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
self.init_state_id += self._reset_stride   # n_envs
```

Official factory `create_libero_envs` assigns `episode_index = 0..n_envs-1` so
parallel envs start on distinct init rows. With `n_envs=1` and a **new** env per
RASE episode, `episode_index=0` ⇒ **every episode loads `init_states[0]`**.

Live check (`libero_spatial_000005`, 50 official inits):

| Condition | Result |
|---|---|
| `episode_index=0`, seeds 1 vs 2 | **identical** post-reset qpos |
| `episode_index` 0/1/2, same seed | **3 distinct** qpos hashes |

Seed does **not** select the init row; only `episode_index` does.

## 3. Pool step-0 sim_state hashes

Artifact: `runs/ngc_w9_step0_sim_only_hash_audit.json`

Hashing only `sim_state.npz['sim_state']` (not controller/rng JSON blobs):

- Spatial / Goal / Long: **within-task collapse** (exact equality) for all
  multi-episode tasks checked.
- Object: byte hashes differ, but pairwise max |Δ| ≈ **1e-14** (float noise only).
- Effective conclusion: W9 clean rollouts did **not** sample the 50-init bank;
  they repeatedly evaluated `init_states[0]` (plus negligible physics noise).

Earlier “unique hashes” that included `controller_state_json` /
`rng_state_json` were false diversity.

## 4. Terminal info / reward / success detector

Artifacts:

- `runs/ngc_w9_init_and_terminal_diagnostic.json`
- `runs/ngc_w9_success_detector_probe.json`

Observed contracts:

| Layer | Reward | terminated | truncated | success signal |
|---|---|---|---|---|
| Raw `OffScreenRenderEnv.step` | `0.0` (non-success) | via `done` | n/a | `check_success()` |
| `LiberoEnv.step` | float | `done or is_success` | **always False** | `info['is_success']`; on terminal also `info['final_info']` |
| `SyncVectorEnv.step` | `[0.0]` | bool array | bool array | top-level `is_success` + on terminal `final_info` |

`success_from_info` (used by the adapter) reads **only**
`info['final_info']['is_success']`. Forced-success probe: on the terminating
vector step, `final_info` is present and `success_from_info` → **True**.
So the 7% clean success rate is **not** explained by a broken detector; labels
that did succeed were recorded.

`LiberoEnv` also auto-`reset()`s inside `step` on success (and advances
`init_state_id`). Because RASE closes the env after each episode, that advance
does not diversify the **next** episode’s starting init.

## 5. 140 requested → 138 in manifest

Not two crash skips. Accounting:

1. **Batch `20260730`:** 6× `already_in_pool` skips (indices 0–5 from the
   pre-fingerprint-hotfix partial run). Those episodes **are** in the pool.
2. **Manifest gap (140→138):** episodes `ep-0135277a-00000032` and
   `…00000037` both **completed as success** (9 and 11 snapshots) but
   `retain_snapshot(..., success_fraction=0.20)` kept **zero** steps, so they
   never appear as episode-groups in `manifest.json`.

`episodes_skipped_crash_list` remained 0 for all three batches.

## 6. Failure24 contingency (W9 Smol × W8 direct OFT)

Artifact: `runs/ngc_w9_failure24_smol_oft_contingency.json`

Intersection: **24/24** state_keys.

| both | Smol only | OFT only | neither |
|---:|---:|---:|---:|
| **0** | **0** | **9** | **15** |

Matches frozen W8 pairing (`direct_only=9`, Smol portfolio/direct-Smol 0).

## 7. Pool validity mark

- `pool/ngc_w9_clean_controls/POOL_VALIDITY.json`
- `runs/ngc_w9_pool_validity.json`

`status: diagnostic_invalid_for_control`; `do_not_delete: true`.

## 8. Proposed W9B protocol (preregistered; **do not collect until approved**)

### Scientific intent

Recover a **clean-success control cohort** with genuine init-state diversity so
clean-regret is estimable under deployable features—without adaptive sampling
past a hard stop.

### Frozen fixes (implementation gates before any GPU collect)

1. **Init schedule (required):** pass
   `episode_index = request.index % n_official_inits` (or an explicit
   `init_state_id` field on `PerturbationRequest`), with `n_official_inits`
   read from `len(get_task_init_states(...))` (typically 50). Record
   `init_state_id` in episode/state metadata.
2. **Factory parity:** same schedule in `libero_env_factory.py` / restore paths.
3. **Success retention:** for control collection, either keep all success
   snapshots or use a floor (`≥1` snapshot per success episode) so success
   episodes cannot vanish from the manifest.
4. **Idempotent resume:** keep `already_in_pool` skip semantics; do not reuse
   the invalid W9A pool as train/control evidence.
5. **Focused pytest + live fingerprint smoke** remain mandatory before GPU.

### Preregistered collection schedule (proposal)

| Item | Value |
|---|---|
| Pool path | `pool/ngc_w9b_clean_controls` (**new**; do not append to W9A) |
| Episodes | 60 + optional 40 + optional 40 (**max 140**), seeds `20260801/02/03` |
| Suites | Spatial / Object / Goal / Long balanced |
| Tasks | original-10 clean mapping unchanged |
| Init | `init_state_id = episode_index % 50` (or measured `n_inits`) |
| Cadence / retention | cadence 2; success retention **1.0** for W9B control pool **or** guaranteed ≥1 keep |
| Freeze | same 32-state protocol: suite × {early,mid} × 4, `distinct_episodes`, success only |
| Kill | incomplete coverage after 140 → stop + audit; no adaptive top-up |

### Post-collect gates (unchanged science)

Direct Smol + direct OFT on failure24 and clean32; cohort-semantic readiness;
ridge only if ready; task-heldout vs action-matched random decides method branch.

### Explicit non-goals

No DQN/CQL/PER/MLP; no portfolio proxy labels; no training on W9A pool; no
lowering `min_train_states` / `per_cell`.

## Next gate

Await approval of W9B before any code-backed collect. Optional prep (still no
collect): implement `episode_index`/`init_state_id` plumbing + unit tests that
prove distinct inits for indices 0 vs 1.
