# W8 lightweight selector and direct-escalation arm

## Status

**Completed on the server.** The 24 direct OFT rollouts finished and produced
9/24 successes: Long 5/10, Goal 4/8, Object 0/2, Spatial 0/4. See the immutable
result record `progress/2026-07-29_w8_direct_escalation_results.md` for the
frozen statistics, interpretation, artifacts, and W9 gate.

The final W7 matrix is Smol 0/24 states versus prefix+OFT 8/24 states, exact
state-level McNemar `p=0.0078125`. W8 direct OFT is 9/24; the new post-analysis
code must align the 9 and 8 successes by state before comparing those two OFT
routes.

## Implemented

- `rase/selector/lightweight.py`
  - state-level three-arm dataset;
  - hard rejection of any-of-K portfolio proxies;
  - episode-group leakage, clean-control, support, and label-collapse gates;
  - multi-output ridge utility model with explicit parameter count;
  - always-continue, always-escalate, always-abstain, matched-random-trigger,
    and oracle-upper-bound metrics.
- `scripts/rollout_oft_prefix_ablation.py --arms direct`: direct OFT only,
  preserving the full causal grid as the default behavior.
- `scripts/run_w8_direct_escalation.sh`: resumable four-suite W8 runner.
- `scripts/export_direct_escalation_dataset.py`: historical W8 export joining
  frozen Smol candidate-0 and direct OFT. W9 supersedes its continuation label
  with true empty-prefix direct Smol.
- `scripts/build_selector_splits.py`: frozen episode- or task-disjoint splits.
- `scripts/train_lightweight_selector.py`: audit-first training; exits without a
  model when scientific readiness fails.

Server verification before W8: 20 focused tests passed in 0.18 seconds. The
synthetic model used 18 parameters and learned the cost-sensitive escalation
rule. W8 execution and dataset export completed without a reported exception.

## Scientific interpretation

W6/W7 `oft_portfolio_hit` means at least one of eight *Smol prefixes followed by
OFT* succeeded. It is not the result of a direct escalation action. Training a
selector on that bit would leak an oracle portfolio and inflate deployable
performance. W8 measures the correct counterfactual: invoke OFT immediately
from exactly the same snapshot.

The failure-only W7 rows will still fail selector readiness because clean
success controls are absent. That failure is expected and is the next data
collection gate, not a software error.

## Frozen execution order

1. **Done:** read and freeze final W7 matrix.
2. **Done:** run W8 direct OFT on all 24 W7 states.
3. Export direct action rows and run the readiness audit; preserve its expected
   `NOT_READY` report as evidence that no failure-only selector was trained.
4. Build an independent success-retained control cohort, run the same direct
   arms, combine JSONL files, and freeze episode/task-disjoint splits.
5. Only after readiness passes, train ridge; add a 2-layer MLP only if it beats
   matched random on held-out tasks.

## CPU post-analysis command

```bash
cd /root/autodl-tmp/RASE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

pytest -q tests/test_escalation_analysis.py tests/test_lightweight_selector.py
bash scripts/run_w8_posthoc_analysis.sh | tee runs/ngc_w8_posthoc_analysis.log
```
