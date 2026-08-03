# W10 Object/Spatial benchmark protocol

W10 is a fixed diagnostic benchmark, not a selector-development experiment. Its new collection is only an independent Object/Spatial failure extension; it does not invent or recollect clean episodes. The held-out action dataset pairs those failure rows with frozen, identity-valid W9C Object/Spatial clean-control rows. It measures direct SmolVLA continuation and direct OFT escalation and does not authorize selector training, MLPs, RL, policy tuning, or claims beyond benchmark diagnosis.

## Frozen protocol

- Collection: exactly 80 attempted episodes, seed `20260731`, Object/Spatial 50/50, camera/robot 50/50, L1/L2, cadence 2 action chunks, and zero retained successful snapshots.
- Benchmark: 16 failure states: two random states from each `suite × dim × level` cell, seed `20260731`, at least 100 remaining steps, and 16 distinct `(task_id, episode_id)` groups.
- Policies: frozen SmolVLA hash `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`; suite-matched `ckpts/oft_object` and `ckpts/oft_spatial`; one direct rollout per state and policy; Smol continuation temperature `0.5`.
- Clean source: frozen `runs/ngc_w9c_clean_action_dataset.jsonl`, filtered read-only to Object/Spatial and `clean_control` before merging. This is not a wholly new W10 clean cohort.
- Split seed: `20260731`, episode grouping. Every train/val/test split must contain both `clean_control` and `failure_challenge`, both Object and Spatial, observed abstain/Smol/OFT arms, and at least two supported optimal actions. Splits are support/diagnosis artifacts only; do not train a selector from W10.
- Hard stop: no top-up, replacement, resampling, seed change, cell relaxation, or outcome-adaptive selection after the 80 attempted episodes. An incomplete inventory is a reported `NOT_READY` result, not permission to collect more.

Before any GPU run, save SHA-256 values for both W10 configs, this runbook, policy/checkpoint trees, and a frozen 80-row collection identity manifest. The manifest must record row index, request seed, suite, perturbation dimension and level, resolved LIBERO-plus task identity, init-state identity, and all prior cohort episode identities used for exclusion.

## Manual BLOCKED gates

Current code does **not** enforce a W10 schedule hash, resolved task identity, init-state identity, the 80-episode hard stop across invocations, or cross-pool cohort deduplication. `collect_state_pool` deterministically samples from the seed, but catalog changes can change resolved tasks; for this non-scheduled protocol the adapter derives init-state identity from request index.

1. **MANUAL BLOCKED — identity freeze:** do not collect until the reviewed 80-row manifest and its SHA-256 exist. There is no current CLI that creates or validates this W10 manifest; do not treat the JSON seed as equivalent.
2. **MANUAL BLOCKED — prior-cohort exclusion:** compare `(resolved task, init_state_id)` and `(task_id, episode_id)` against all W3–W9 benchmark, pilot, control, and selector cohorts. Because those cohorts may live in other pools, `sample_state_keys --exclude-episode-keys-json` cannot enforce this cross-pool gate. Any match blocks W10; do not replace it adaptively.
3. **MANUAL BLOCKED — hard stop:** verify the W10 pool is new and launch the collection command exactly once. Resume may skip already written episode IDs, but no code caps cumulative invocations at 80.
4. **MANUAL BLOCKED — selected identity audit:** after sampling, independently verify 16 unique episode groups, two states in every frozen cell, zero prior identity overlap, and hashes matching the preregistration manifest.
5. **MANUAL BLOCKED — evaluation schedule/seed:** before direct outcomes, freeze the ordered 16 state keys, their SHA-256, suite order `object,spatial`, arm order, and output names. Direct Smol derives a deterministic rollout seed from each state key; the OFT direct runner exposes no rollout-seed CLI or config field, so the manifest must state this limitation. Do not claim bitwise repeatability or restart a completed arm; only resume its existing scheduler artifact.
6. **MANUAL BLOCKED — W9C clean identity and cross-source audit:** do not filter or merge until `runs/ngc_w9c_clean_action_dataset.jsonl` is confirmed to be the frozen identity-valid W9C artifact and its SHA-256 is recorded. Before splitting, audit the union for duplicate `state_key` and overlapping `(task_id, episode_id)` or resolved task/init identities across W9C clean and W10 failure. The filter and merge tools reject duplicate state keys, but only the split audit checks episode-group leakage across splits; neither automatically validates cross-source task/init identity. Any overlap blocks analysis and must not be replaced.

Record reviewer, UTC timestamp, input artifact paths, ordered execution schedule, hashes, and PASS/BLOCK for every gate in `runs/ngc_w10_manual_gate_manifest.json`. This is an operator artifact, not an automatically generated guarantee.

## 1. CPU preflight

Use `smolvla` for configuration and client commands; the OFT suite runner switches its server process to `oft` itself.

```bash
cd /root/autodl-tmp/RASE
conda run -n smolvla python -m pytest -q \
  tests/test_filter_selector_dataset.py tests/test_w10_protocol.py
conda run -n smolvla python scripts/collect_state_pool.py --help
conda run -n smolvla python scripts/sample_state_keys.py --help
conda run -n smolvla python scripts/rollout_direct_smol.py --help
conda run -n smolvla python scripts/filter_selector_dataset.py --help
conda run -n smolvla python scripts/merge_selector_datasets.py --help
conda run -n smolvla python scripts/build_selector_splits.py --help
./scripts/run_oft_verify_suites.sh --help
```

The shell runner has no argparse help mode: the last command is only a parse smoke and is expected to exit nonzero with an `output tag suffix` usage message. Do not proceed unless the manual identity, prior-cohort, and new-pool gates PASS.

## 2. Fixed collection and inventory gate

```bash
conda run -n smolvla python scripts/collect_state_pool.py \
  --config configs/collect_w10_object_spatial_failures.json \
  --summary-output runs/ngc_w10_object_spatial_collect80.json

conda run -n smolvla python scripts/sample_state_keys.py \
  --config configs/ngc_w10_object_spatial_benchmark.yaml \
  --inventory-only --require-complete \
  --output runs/ngc_w10_object_spatial_inventory.json
```

Stop permanently at 80 attempted episodes. The inventory must report all eight `suite × dim × level` cells with at least two eligible distinct failure episode groups. If not, report `NOT_READY`; never top up or relax the protocol.

After the manual cross-pool exclusion gate passes, freeze keys:

```bash
conda run -n smolvla python scripts/sample_state_keys.py \
  --config configs/ngc_w10_object_spatial_benchmark.yaml \
  --require-complete \
  --output runs/ngc_w10_object_spatial_state_keys.json
```

The sampler automatically guarantees uniqueness among selected episode groups within this pool. It does not guarantee cross-pool identity exclusion.

## 3. Direct outcomes only

Do not inspect either policy's outcomes to alter keys, order, or policy settings. First mark the manual evaluation schedule/seed gate PASS; otherwise this stage remains blocked.

```bash
conda run -n smolvla python scripts/rollout_direct_smol.py \
  --config configs/ngc_w10_object_spatial_benchmark.yaml \
  --state-keys-json runs/ngc_w10_object_spatial_state_keys.json \
  --output-dir runs/ngc_w10_direct_smol_object_spatial16 \
  --fresh-run

SMOLVLA_ENV=smolvla OFT_ENV=oft \
OFT_SUITE_SHORTS=object,spatial \
OUTPUT_PREFIX=ngc_w10_direct_oft \
STATE_KEYS_JSON=runs/ngc_w10_object_spatial_state_keys.json \
CANDIDATES_DIR=runs/ngc_w10_object_spatial_state_keys.json \
OFT_RUNNER=prefix-ablation OFT_PREFIX_ARMS=direct FRESH_RUN=1 \
./scripts/run_oft_verify_suites.sh \
  configs/ngc_w10_object_spatial_benchmark.yaml object_spatial16
```

The exact OFT summaries are:

- `runs/ngc_w10_direct_oft_object_object_spatial16/summary.json`
- `runs/ngc_w10_direct_oft_spatial_object_spatial16/summary.json`

One missing, failed, or mismatched state blocks paired analysis. No outcome-based reruns or substitutions are allowed; only scheduler resume of the same frozen state/policy arm is permitted.

## 4. Action datasets, read-only clean filter, merge, then split gate

The split action-support gate is forbidden before both W10 direct outcome arms, the W10 failure action dataset, the frozen W9C clean source validation, read-only filtering, and the merged action dataset exist. Feature extraction is descriptive input preparation, not selector training.

First create the new W10 failure rows:

```bash
conda run -n smolvla python scripts/extract_selector_features.py \
  --pool pool/ngc_w10_object_spatial_failures \
  --state-keys runs/ngc_w10_object_spatial_state_keys.json \
  --output runs/ngc_w10_object_spatial_features.json

conda run -n smolvla python scripts/export_selector_action_dataset.py \
  --smol-direct-summary runs/ngc_w10_direct_smol_object_spatial16/summary.json \
  --oft-direct-summary runs/ngc_w10_direct_oft_object_object_spatial16/summary.json \
  --oft-direct-summary runs/ngc_w10_direct_oft_spatial_object_spatial16/summary.json \
  --features runs/ngc_w10_object_spatial_features.json \
  --pool pool/ngc_w10_object_spatial_failures \
  --cohort failure_challenge \
  --output runs/ngc_w10_object_spatial_failure_action_dataset.jsonl
```

After the manual W9C identity/hash gate passes, filter the frozen source without editing JSONL rows. The filter rejects duplicate source `state_key` values and records source/output hashes, row counts, suite counts, cohort counts, and verbatim-row preservation:

```bash
conda run -n smolvla python scripts/filter_selector_dataset.py \
  --dataset runs/ngc_w9c_clean_action_dataset.jsonl \
  --suite Object --suite Spatial \
  --cohort clean_control \
  --output runs/ngc_w10_w9c_object_spatial_clean_action_dataset.jsonl \
  --manifest-output runs/ngc_w10_w9c_object_spatial_clean_filter_manifest.json
```

Manually verify the manifest source SHA against the approved frozen W9C hash and confirm both suites have nonzero clean rows. Then merge using the existing CLI; it rejects duplicate state keys across sources and records source/output SHA-256 values:

```bash
conda run -n smolvla python scripts/merge_selector_datasets.py \
  --dataset runs/ngc_w10_object_spatial_failure_action_dataset.jsonl \
  --dataset runs/ngc_w10_w9c_object_spatial_clean_action_dataset.jsonl \
  --output runs/ngc_w10_object_spatial_heldout_action_dataset.jsonl \
  --manifest runs/ngc_w10_object_spatial_heldout_merge_manifest.json
```

Complete and record the manual cross-source episode/task/init identity audit before building splits. Only then run the support gate:

```bash
conda run -n smolvla python scripts/build_selector_splits.py \
  --dataset runs/ngc_w10_object_spatial_heldout_action_dataset.jsonl \
  --output runs/ngc_w10_object_spatial_episode_splits.json \
  --seed 20260731 --grouping episode \
  --requirements configs/ngc_w10_split_requirements.json \
  --fail-not-ready
```

`READY` means only that current `audit_split_support` fields pass: every train/val/test split contains clean and failure cohorts, Object and Spatial, required episode-group counts, observed abstain/Smol/OFT arms, at least two supported optimal actions, and no episode-group leakage. It does not validate schedule/task/init identity, W9C identity validity, or cross-source task/init deduplication. With this small deterministic greedy split, `NOT_READY` is acceptable and final: do not change the seed, split, costs, cohort, sampled states, or collection size, and do not top up.

## Prohibited actions and reporting

Never run `train_lightweight_selector.py`, any MLP, or RL for W10. Never choose states from observed outcomes. Report frozen counts and hashes, W9C filter and merge manifests, cross-source identity audit, inventory and manual gate status, paired direct outcomes by suite/cell, missingness, and the split support audit. Label every conclusion benchmark/diagnosis only.
