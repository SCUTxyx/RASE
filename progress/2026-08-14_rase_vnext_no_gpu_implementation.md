# RASE vNext no-GPU implementation checkpoint

Date: 2026-08-14

No GPU experiment, collection, training, or learned-stage evaluation was started.
R10-B remains the active prerequisite and all learned descendants remain locked.

## Current R10 status

The contract-aware monitor reports 10/54 valid trajectories (18.5%). A trajectory
counts only when both t8 and t16 contain a non-empty per-chunk query trace. The
collection `COMPLETE` marker and G0 audit are absent.

The resumable launcher reuses the frozen 18-group diagnostic manifest and existing
output directory. The existing collector skips each valid per-record JSON, so a
restart does not repeat the ten completed trajectories. It refuses a real launch
when the GPU is unavailable, free data-disk space is below 20 GiB, the manifest is
not frozen, an audit already exists, or the tmux session already exists.

## Implemented CPU-side contracts

- per-cell, all-three-pair R10 A/B/C/D root-cause matrix audit;
- exact progress validator, watcher, dry-run, resumable tmux launcher, and auto-audit;
- lossless variable-dimension canonical robot/observation/action schema;
- minimal LIBERO benchmark, policy, and correction-operator adapters;
- explicit five-layer seed ledger and semantic operator masking;
- fail-closed vNext protocol validator with unresolved scientific fields visible;
- deterministic outcome-independent fixed-K3 discovery manifest freezer;
- root/task-nested fixed-K5 model-free opportunity audit with raw-metric reporting.

Server verification: 20 selected CPU contract tests passed. The draft protocol
structure passes, but activation correctly remains blocked on G0 and on unresolved
utility/cost, feasibility, meaningful-effect, and minimum-task fields.

## Commands

From `/root/autodl-tmp/RASE`:

```bash
bash scripts/run_rase_vnext_no_gpu_checks.sh
python scripts/status_r10b_chunk_diagnostic.py
python scripts/status_r10b_chunk_diagnostic.py --watch --interval 30
RASE_DRY_RUN=1 bash scripts/run_r10b_chunk_diagnostic_resume.sh
```

When a GPU is attached:

```bash
bash scripts/run_r10b_chunk_diagnostic_resume.sh
tmux attach -t r10b_chunk_full
```

Detaching from tmux (`Ctrl-b`, then `d`) does not stop collection. On successful
54/54 completion the same tmux chain writes
`runs/pre_c0_r10/r10b_chunk_input_divergence_audit_v1.json`. Do not run discovery
until the protocol config is scientifically completed, set to `frozen`, validated,
and hash-bound to that contract-valid G0 audit.
