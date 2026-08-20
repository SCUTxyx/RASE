#!/usr/bin/env python3
"""Summarize PRE-C0-R2 with key-based paired comparisons."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict):
    return str(row["task_id"]), int(row["init_state_id"]), int(row["seed"])


def index(rows):
    out = {}
    for row in rows:
        k = key(row)
        if k in out:
            raise ValueError(f"duplicate episode key: {k}")
        out[k] = row
    return out


def paired(a_rows, b_rows):
    a, b = index(a_rows), index(b_rows)
    if set(a) != set(b):
        raise ValueError(f"manifest mismatch: only_a={len(set(a)-set(b))}, only_b={len(set(b)-set(a))}")
    rescue = harm = both_ok = both_fail = 0
    for k in a:
        av, bv = bool(a[k]["success"]), bool(b[k]["success"])
        if not av and bv: rescue += 1
        elif av and not bv: harm += 1
        elif av and bv: both_ok += 1
        else: both_fail += 1
    n = len(a)
    return {
        "n": n, "a_success": sum(bool(x["success"]) for x in a.values()),
        "b_success": sum(bool(x["success"]) for x in b.values()),
        "rescue": rescue, "harm": harm, "net": rescue - harm,
        "both_success": both_ok, "both_fail": both_fail,
        "delta_pp": 100.0 * (rescue - harm) / max(1, n),
    }


def exact_two_sided_mcnemar(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    lo = min(rescue, harm)
    cdf = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2.0 * cdf)


def arm_summary(rows):
    n = len(rows)
    takeover_semantics = (
        "executed_corrected_action_steps"
        if any("takeover_entries" in r for r in rows)
        else "legacy_counter_not_comparable" if any(float(r.get("takeover_steps", 0)) > 0 for r in rows)
        else "none"
    )
    success_rows = [r for r in rows if r.get("success")]
    fail_rows = [r for r in rows if not r.get("success")]
    by_task = {}
    for task in sorted({str(r["task_id"]) for r in rows}):
        task_rows = [r for r in rows if str(r["task_id"]) == task]
        by_task[task] = {
            "n": len(task_rows),
            "success": sum(bool(r.get("success")) for r in task_rows),
        }
    return {
        "n": n,
        "success": sum(bool(r.get("success")) for r in rows),
        "success_rate": sum(bool(r.get("success")) for r in rows) / max(1, n),
        "takeover_steps_mean": sum(float(r.get("takeover_steps", 0)) for r in rows) / max(1, n),
        "takeover_counter_semantics": takeover_semantics,
        "takeover_steps_mean_success": sum(float(r.get("takeover_steps", 0)) for r in success_rows) / max(1, len(success_rows)),
        "takeover_steps_mean_failure": sum(float(r.get("takeover_steps", 0)) for r in fail_rows) / max(1, len(fail_rows)),
        "gate_activation_rate": (
            sum(float(r.get("gate_positive_decisions", 0)) for r in rows)
            / max(1.0, sum(float(r.get("gate_queries", 0)) for r in rows))
        ),
        "by_task": by_task,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--b0", type=Path, required=True)
    ap.add_argument("--bounded", type=Path, required=True)
    ap.add_argument("--gate", type=Path, default=None)
    ap.add_argument("--envelope-only", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--training-report", type=Path, required=True)
    args = ap.parse_args()

    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    b0, bounded = load_jsonl(args.b0), load_jsonl(args.bounded)
    gate = load_jsonl(args.gate) if args.gate and args.gate.exists() else []
    envelope = load_jsonl(args.envelope_only)
    labels = load_jsonl(args.labels)
    training = json.loads(args.training_report.read_text())

    comparisons = {
        "envelope_vs_b0": paired(b0, envelope),
        "legacy_bounded_vs_b0": paired(b0, bounded),
        "envelope_vs_legacy_bounded": paired(bounded, envelope),
    }
    if gate:
        comparisons["gate_vs_b0"] = paired(b0, gate)
        comparisons["gate_vs_envelope"] = paired(envelope, gate)
    for item in comparisons.values():
        item["mcnemar_p"] = exact_two_sided_mcnemar(item["rescue"], item["harm"])

    if not training.get("gate_pass", False):
        decision = "TRAIN-GATE-FAIL"
    elif not gate:
        decision = "GATE-EVAL-MISSING"
    else:
        gate_b0 = comparisons["gate_vs_b0"]
        gate_env = comparisons["gate_vs_envelope"]
        gate_harm_rate = gate_b0["harm"] / max(1, gate_b0["n"])
        if gate_env["net"] >= 2 and gate_harm_rate <= 0.05:
            decision = "PASS"
        elif abs(gate_env["net"]) <= 1 and gate_harm_rate <= 0.05:
            decision = "TIE-PREFER-ENVELOPE"
        else:
            decision = "FAIL-PREFER-ENVELOPE"

    label_types = Counter(r.get("label_type", "unknown") for r in labels)
    report = {
        "schema_version": "rase-pre-c0-r2-summary/v1",
        "decision": decision,
        "note": "40 episodes resolve success differences in 2.5pp increments; the preregistered 3pp margin therefore requires at least two net episodes (5pp).",
        "labels": {
            "n": len(labels), "types": dict(label_types),
            "tasks": sorted({str(r["task_id"]) for r in labels}),
            "episodes": len({r.get("episode_id", str(key(r))) for r in labels}),
        },
        "training": training,
        "arms": {
            "b0": arm_summary(b0), "legacy_bounded": arm_summary(bounded),
            "envelope_only": arm_summary(envelope),
        },
        "paired": comparisons,
        "limitations": [
            "This is a development decision, not a 2pp non-inferiority proof.",
            "Training uses spatial train tasks; evaluation uses spatial dev tasks.",
            "The constant F0 vector and gate are suite-specific.",
            "Legacy bounded takeover_steps were written before the counter fix and count entries; they are not comparable to R2 executed corrected action steps.",
        ],
    }
    if gate:
        report["arms"]["gate_envelope"] = arm_summary(gate)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    def rate(name):
        a = report["arms"][name]
        return f"{a['success']}/{a['n']} = {100*a['success_rate']:.1f}%"
    env_b0 = comparisons["envelope_vs_b0"]
    env_legacy = comparisons["envelope_vs_legacy_bounded"]
    gate_row = f"| Learned gate + envelope | {rate('gate_envelope')} |" if gate else "| Learned gate + envelope | NOT RUN: training gate failed |"
    gate_detail = ""
    if gate:
        gate_b0 = comparisons["gate_vs_b0"]
        gate_env = comparisons["gate_vs_envelope"]
        gate_detail = f"- Gate vs envelope-only: net {gate_env['net']}/40 ({gate_env['delta_pp']:+.1f}pp), rescue={gate_env['rescue']}, harm={gate_env['harm']}, McNemar p={gate_env['mcnemar_p']:.4f}\n- Gate vs B0: net {gate_b0['net']}/40 ({gate_b0['delta_pp']:+.1f}pp), rescue={gate_b0['rescue']}, harm={gate_b0['harm']}"

    md = f"""# PRE-C0-R2: Gate Data Scaling + Conditional Evaluation

**Decision: {decision}**

## Data and training

- Labels: {len(labels)} from {report['labels']['episodes']} episodes and {len(report['labels']['tasks'])} train tasks
- Types: `{dict(label_types)}`
- OOF AP: {training.get('average_precision', float('nan')):.3f}; prevalence: {training.get('prevalence', float('nan')):.3f}
- Threshold: {training.get('selected',{}).get('threshold', float('nan')):.3f}
- Rescue recall: {training.get('selected',{}).get('rescue_recall', float('nan')):.3f}
- Harm activation rate: {training.get('selected',{}).get('harm_activation_rate', float('nan')):.3f}

## Spatial dev (same 40-episode manifest)

| Arm | Success |
|---|---:|
| B0 | {rate('b0')} |
| Legacy bounded F0 | {rate('legacy_bounded')} |
| Envelope-only (same new runner) | {rate('envelope_only')} |
{gate_row}

## Paired decisions

- Envelope-only vs B0: net {env_b0['net']}/40 ({env_b0['delta_pp']:+.1f}pp), rescue={env_b0['rescue']}, harm={env_b0['harm']}, McNemar p={env_b0['mcnemar_p']:.4f}
- Envelope-only vs legacy bounded: net {env_legacy['net']}/40 ({env_legacy['delta_pp']:+.1f}pp), rescue={env_legacy['rescue']}, harm={env_legacy['harm']}
{gate_detail}
- Envelope-only takeover burden: mean {report['arms']['envelope_only']['takeover_steps_mean']:.1f} action steps/episode; success mean {report['arms']['envelope_only']['takeover_steps_mean_success']:.1f}; failure mean {report['arms']['envelope_only']['takeover_steps_mean_failure']:.1f}.
- 40 episodes only resolve 2.5pp increments; a nominal 3pp margin requires at least 2 net episodes (5pp).

The learned gate arm was intentionally not run because its leakage-free training gate failed. This remains a dev decision and cannot establish 2pp clean non-inferiority.
"""
    (out / "summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
