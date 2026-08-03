# W6 benchmark release and statistical audit

This runbook turns the frozen W4 dual-oracle artifacts into a leakage-safe
research release. It performs no policy rollout.

## 1. Validate code and environment

```bash
cd /root/autodl-tmp/RASE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

python scripts/preflight_runner.py \
  --libero-plus-root /root/autodl-tmp/src/LIBERO-plus \
  --checkpoints-root ckpts
pytest -q
```

## 2. Regenerate the W4 summary

The current aggregator adds state-level cross-oracle agreement, Cohen's kappa,
and an exact two-sided McNemar test. Agreement remains descriptive because the
two oracle tracks have different measurement semantics.

```bash
SUMMARY_ONLY=1 ./scripts/run_w4_adequate_pipeline.sh
```

Inspect these fields before export:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/ngc_w4_adequate_dual_oracle_summary.json")
payload = json.loads(path.read_text())
print(json.dumps(payload["cross_oracle_agreement"], indent=2))
print(json.dumps(payload["warnings"], indent=2))
PY
```

## 3. Export candidate rows and grouped benchmark splits

```bash
python scripts/export_recovery_dataset.py \
  --dual-oracle runs/ngc_w4_adequate_dual_oracle_summary.json \
  --pool pool/ngc_step1_scale200 \
  --candidates-dir runs/ngc_w4_adequate_candidates \
  --output runs/ngc_w4_recovery_dataset.jsonl \
  --split-seed 20260727
```

Outputs:

- `runs/ngc_w4_recovery_dataset.jsonl`: one row per state/candidate;
- `runs/ngc_w4_recovery_dataset.splits.json`: label index only;
- `runs/ngc_w4_recovery_dataset.benchmark-splits.json`: episode-grouped
  train/validation/test manifest with an exact audit.

Only the benchmark-splits file may be used for model development. The label
index is not a train/test split.

## 4. Audit the grouped release

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/ngc_w4_recovery_dataset.benchmark-splits.json")
payload = json.loads(path.read_text())
assert payload["audit"]["group_leakage"] is False
assert sum(len(keys) for keys in payload["splits"].values()) == payload["audit"]["n_states"]
print(json.dumps(payload["audit"], indent=2))
PY
```

W4 has only 32 states and all were selected at `t0=0`; this split is an
engineering gate, not a sufficient training benchmark. Do not report selector
generalization from W4 alone.

## 5. Optional final SmolVLA temperature sensitivity

Run t=0.3 and t=1.0 only if the compute budget is acceptable. If both remain
zero-hit, stop this diagnostic line and move to the L1–L2 / cross-policy
recoverability matrix described in
`plan/RASE_top_conference_execution_v4.md`.

The exact generation and screen loop is in `docs/runbooks/w5_smol_frontier.md`.
