# W4/OFT runner hardening

Run the read-only preflight before allocating a GPU run:

```bash
python scripts/preflight_runner.py \
  --conda-root /root/miniconda3 \
  --smolvla-env smolvla --oft-env oft \
  --libero-plus-root /root/autodl-tmp/src/LIBERO-plus
```

It fails on missing `pyzmq`, a non-Plus benchmark import, missing
`get_benchmark_dict` suites, suites with at most 10 tasks, or an OFT
checkpoint whose dataset statistics do not match its suite. Existing GPU
compute processes are a warning so the operator can decide whether sharing is
intentional. The preflight does not install, edit, or launch model servers.

Formal W4 runs consume one frozen key artifact for candidate generation,
SmolVLA, and all four OFT suites:

```bash
KEYS_JSON=runs/ngc_w4_adequate_state_keys.json \
CAND_DIR=runs/ngc_w4_adequate_candidates \
./scripts/run_w4_adequate_pipeline.sh
```

The default behavior is resume. Completed suite summaries are skipped,
scheduler records are reused, and a lock prevents duplicate OFT runners.
Server cleanup targets only the PID started by the runner. Do not delete a
partial scheduler to resume it.

Fresh execution is opt-in and refuses any existing run output directory:

```bash
FRESH_RUN=1 SMOL_OUT=runs/w4_new_smol OUTPUT_PREFIX=w4_new_oft \
TAG=trial ./scripts/run_w4_adequate_pipeline.sh
```

To rebuild compact reports without starting candidate generation, policies,
or OFT servers:

```bash
SUMMARY_ONLY=1 ./scripts/run_w4_adequate_pipeline.sh
```

Candidate and rollout summaries record the raw key-artifact checksum and a
canonical ordered-key checksum. OFT summaries additionally declare
`verification_semantics: deterministic_one_shot`; their one trial per
candidate is portfolio evidence, not Wilson Set A/B/C certification.
