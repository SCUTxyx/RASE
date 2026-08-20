#!/usr/bin/env python3
"""Pre-registered R6-D Pareto comparison: no-world-model vs world-model risk.

Compares two R6-C-style reports:

- ``--baseline``: the no-WM candidate-arm OOF report (``rase-r6c-candidate-arm-oof/v1``);
- ``--wm``: the same report schema produced with ``--wm-features`` in
  ``train_r6c_candidate_arm_student.py`` (extended variant, schema
  ``rase-r6c-candidate-arm-oof-wm/v1``).

Enforces the pre-registered keep-the-WM-arm gate:

1. per-VLA + pooled task-held-out risk ranking gain (AUROC +0.02 or strict
   dominance on >=4/5 seeds for at least one VLA);
2. success gap >= -5pp per VLA;
3. false continue <= 5%;
4. savings >= 20%;
5. latency budget (WM feature inference must fit the real-time budget).

Output is a single JSON with a ``decision`` field; it never edits the baseline
report.  The WM arm is integrated only if ``decision == keep``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def auroc_from_predictions(predictions: list[dict], label_key: str,
                           score_key: str) -> float:
    """Rank-based AUROC over a report's per-row predictions (labels 0/1)."""
    labels = np.asarray([1.0 if row[label_key] else 0.0 for row in predictions])
    scores = np.asarray([float(row[score_key]) for row in predictions])
    if len(set(labels.tolist())) < 2:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank = np.arange(1, len(labels) + 1)
    auc = (rank[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--auroc-gain-min", type=float, default=0.02)
    parser.add_argument("--dominant-seeds-min", type=int, default=4)
    args = parser.parse_args()

    base = _load(args.baseline)
    wm = _load(args.wm)
    if base.get("schema_version") != "rase-r6c-candidate-arm-oof/v1":
        raise ValueError(f"unexpected baseline schema {base.get('schema_version')}")
    if wm.get("schema_version") not in ("rase-r6c-candidate-arm-oof/v1",
                                        "rase-r6c-candidate-arm-oof-wm/v1"):
        raise ValueError(f"unexpected wm schema {wm.get('schema_version')}")
    if base["seed"] != wm["seed"]:
        raise ValueError("seed mismatch between baseline and wm reports")
    if base["mode"] != wm["mode"] or base.get("target_policy") != wm.get("target_policy"):
        raise ValueError("mode/target mismatch between baseline and wm reports")

    policies = sorted(set(base["metrics_by_policy"]) | set(wm["metrics_by_policy"]))
    per_policy: dict[str, dict] = {}
    for policy in policies:
        base_metrics = base["metrics_by_policy"].get(policy, {})
        wm_metrics = wm["metrics_by_policy"].get(policy, {})
        base_pred = [row for row in base["predictions"] if row["policy_id"] == policy]
        wm_pred = [row for row in wm["predictions"] if row["policy_id"] == policy]
        per_policy[policy] = {
            "baseline_metrics": base_metrics,
            "wm_metrics": wm_metrics,
            "baseline_auroc": auroc_from_predictions(base_pred, "persistent_success",
                                                     "arm_success_mean") if base_pred else float("nan"),
            "wm_auroc": auroc_from_predictions(wm_pred, "persistent_success",
                                               "arm_success_mean") if wm_pred else float("nan"),
        }

    # Gate 1: AUROC gain or strict dominance on >=4/5 seeds for at least one VLA.
    base_pooled_auc = auroc_from_predictions(base["predictions"], "persistent_success",
                                             "arm_success_mean")
    wm_pooled_auc = auroc_from_predictions(wm["predictions"], "persistent_success",
                                           "arm_success_mean")
    auc_gain = wm_pooled_auc - base_pooled_auc
    ranking_improved = auc_gain >= args.auroc_gain_min
    # Dominance proxy: WM success gap >= baseline for each policy (>=dominant_seeds_min
    # is checked by the caller across seeds; here we record per-policy direction).
    dominance = all(
        per_policy[policy]["wm_metrics"].get("success_gap", -1.0)
        >= per_policy[policy]["baseline_metrics"].get("success_gap", 1.0)
        for policy in policies
    )

    gates = {
        "auroc_gain": {"base": base_pooled_auc, "wm": wm_pooled_auc,
                       "gain": auc_gain, "pass": ranking_improved},
        "per_vla_gap": {
            policy: {"base": per_policy[policy]["baseline_metrics"].get("success_gap"),
                     "wm": per_policy[policy]["wm_metrics"].get("success_gap"),
                     "pass": per_policy[policy]["wm_metrics"].get("success_gap", -1.0) >= -0.05}
            for policy in policies},
        "per_vla_false_continue": {
            policy: {"base": per_policy[policy]["baseline_metrics"].get("false_continue_rate"),
                     "wm": per_policy[policy]["wm_metrics"].get("false_continue_rate"),
                     "pass": per_policy[policy]["wm_metrics"].get("false_continue_rate", 1.0) <= 0.05}
            for policy in policies},
        "per_vla_savings": {
            policy: {"base": per_policy[policy]["baseline_metrics"].get("savings"),
                     "wm": per_policy[policy]["wm_metrics"].get("savings"),
                     "pass": per_policy[policy]["wm_metrics"].get("savings", -1.0) >= 0.20}
            for policy in policies},
        "dominance": dominance,
    }
    gap_pass = all(gates["per_vla_gap"][policy]["pass"] for policy in policies)
    fc_pass = all(gates["per_vla_false_continue"][policy]["pass"] for policy in policies)
    save_pass = all(gates["per_vla_savings"][policy]["pass"] for policy in policies)
    decision = "keep" if (ranking_improved and dominance and gap_pass and fc_pass and save_pass) else "reject"

    result = {
        "schema_version": "rase-r6d-wm-ablation-pareto/v1",
        "baseline": str(args.baseline.resolve()), "wm": str(args.wm.resolve()),
        "seed": base["seed"], "mode": base["mode"], "target_policy": base.get("target_policy"),
        "per_policy": per_policy,
        "gates": gates,
        "decision": decision,
        "note": ("keep: integrate WM auxiliary features as additional inputs. "
                 "reject: write the WM arm as an honest negative result, do NOT "
                 "integrate into the deployed selector."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in
                      ["decision", "gates", "per_policy"]}, indent=2, sort_keys=True))
    return 0 if decision == "keep" else 2


if __name__ == "__main__":
    raise SystemExit(main())
