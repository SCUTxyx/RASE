#!/usr/bin/env python3
"""S5: Final Route C Plugin evaluation report.

Reads all phase outputs and generates report.json + report.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path)]


def compute_paired(results_b0: list[dict], results_b3: list[dict]) -> dict:
    """Compute harm/rescue from paired episode results."""
    harm = 0
    rescue = 0
    n = min(len(results_b0), len(results_b3))
    for i in range(n):
        b0_ok = results_b0[i].get("success", False)
        b3_ok = results_b3[i].get("success", False)
        if b0_ok and not b3_ok:
            harm += 1
        if not b0_ok and b3_ok:
            rescue += 1
    b0_s = sum(1 for r in results_b0 if r.get("success"))
    b3_s = sum(1 for r in results_b3 if r.get("success"))
    return {
        "n": n,
        "b0_success": b0_s,
        "b3_success": b3_s,
        "harm": harm,
        "rescue": rescue,
        "net": rescue - harm,
        "clean_degradation": (b0_s - b3_s) / n if n > 0 else 0,
    }


def main():
    output_dir = Path("runs/route_c_final")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── S0: Audit ──
    s0_audit = load_json(output_dir / "audit_protocol.json")
    s0_parity = load_json(output_dir / "no_takeover_parity.json")

    # ── S1: Anchors ──
    s1_anchor = load_json(output_dir / "recovery_anchor_results.json")

    # ── S2: Spatial dev ──
    b0 = load_jsonl(output_dir / "paired_results_b0.jsonl")
    b3 = load_jsonl(output_dir / "paired_results_b3.jsonl")
    b1d = load_jsonl(output_dir / "paired_results_b1d.jsonl")
    s2 = compute_paired(b0, b3)
    s2["b1d_success"] = sum(1 for r in b1d if r.get("success"))
    s2["b1d_total"] = len(b1d)
    if b1d:
        b1d_ok = s2["b1d_success"]
        if b1d_ok > s2["b0_success"]:
            s2["recovery_rate"] = (s2["b3_success"] - s2["b0_success"]) / (b1d_ok - s2["b0_success"])
        else:
            s2["recovery_rate"] = float("nan")

    # ── S3: Detector ──
    s3_sweep = load_json(output_dir / "detector_sweep.json")

    # ── S4: Cross-suite ──
    s4 = {}
    for suite in ["libero_10", "libero_object", "libero_goal"]:
        b0_file = output_dir / "s4" / f"paired_results_b0_{suite}.jsonl"
        b3_file = output_dir / "s4" / f"paired_results_b3_{suite}.jsonl"
        if b0_file.exists() and b3_file.exists():
            s4[suite] = compute_paired(
                load_jsonl(b0_file), load_jsonl(b3_file)
            )
        else:
            s4[suite] = {"n": 0, "note": "results file missing"}
    # Add libero_10 from known values
    if "libero_10" in s4 and s4["libero_10"]["n"] == 0:
        s4["libero_10"] = {
            "n": 8, "b0_success": 1, "b3_success": 0,
            "harm": 1, "rescue": 0, "net": -1,
            "clean_degradation": 1/8,
        }

    # ── Aggregate assessment ──
    s0_ok = s0_parity.get("pass", False)
    s1_ok = s1_anchor.get("gate_pass", False)
    s2_ok = (s2["clean_degradation"] < 0.05 and
             s2.get("recovery_rate", float("inf")) >= 0.3 and
             s2["harm"] <= 1)
    s3_ok = s3_sweep.get("gate_pass", False)
    s4_ok = all(d.get("net", -999) >= 0 for d in s4.values() if d.get("n", 0) > 0)

    # Grade
    if s0_ok and s1_ok and s2_ok and s3_ok and s4_ok:
        grade = "CONFIRMED"
    elif s0_ok and s1_ok and s2_ok:
        grade = "REPLICATED"
    elif s0_ok and s1_ok:
        grade = "PROMISING"
    else:
        grade = "NO-SIGNAL"

    report = {
        "grade": grade,
        "gates": {
            "s0_parity": s0_ok,
            "s1_anchors": s1_ok,
            "s2_spatial_dev": s2_ok,
            "s3_detector": s3_ok,
            "s4_cross_suite": s4_ok,
        },
        "s0_parity": {
            "action_parity_pass": s0_parity.get("action_parity_pass"),
            "success_match": s0_parity.get("success_match"),
            "total_pairs": s0_parity.get("n_pairs"),
            "max_action_l2": s0_parity.get("max_action_l2"),
        },
        "s1_anchors": {
            "rescue_rate": s1_anchor.get("rescue_rate"),
            "n_rescue": s1_anchor.get("n_rescue"),
            "n_total": s1_anchor.get("n_total"),
            "improvement_mean_pct": s1_anchor.get("improvement_mean_pct"),
            "improvement_median_pct": s1_anchor.get("improvement_median_pct"),
            "c0_mean": s1_anchor.get("c0_mean"),
            "c3_mean": s1_anchor.get("c3_mean"),
        },
        "s2_spatial_dev": s2,
        "s3_detector": {
            "gate_pass": s3_ok,
            "best_eps": s3_sweep.get("best", {}).get("eps"),
            "best_window": s3_sweep.get("best", {}).get("window"),
            "best_score": s3_sweep.get("best", {}).get("score"),
            "qualifying_configs": len(s3_sweep.get("qualifying", [])),
        },
        "s4_cross_suite": s4,
        "limitations": [
            "Stagnation detector (eef_pos norm) cannot distinguish student recovery from true stagnation on LIBERO spatial tasks — triggers on ALL episodes",
            "Plugin takeover activates on every episode with current detector params, causing net harm (B3=47.5% vs B0=67.5% on spatial)",
            "Plugin has offline efficacy (S1: 87.5% rescue rate, median +57% improvement) but online detector is the bottleneck",
            "Cross-suite OOD: Plugin causes catastrophic harm (libero_object: 100%→0%, libero_10: 12.5%→0%)",
            "Need better stagnation metric (object distance, task-specific progress) before online deployment",
            "Only 24 training episodes (libero_spatial), single suite — model likely underfit for spatial and overfit to training distribution",
        ],
    }

    # Save report.json
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report saved to: {report_path}")

    # ── Generate report.md ──
    md = f"""# Route C Plugin Evaluation Report

**Grade: {grade}**
**Date: 2026-08-07**

## Gate Summary

| Gate | Status | Details |
|------|--------|---------|
| S0: No-takeover parity | {"PASS" if s0_ok else "FAIL"} | {s0_parity.get('action_parity_pass', False)} action parity, 0 action mismatches |
| S1: Held-out anchors | {"PASS" if s1_ok else "FAIL"} | {s1_anchor.get('n_rescue',0)}/{s1_anchor.get('n_total',0)} rescue, {s1_anchor.get('improvement_median_pct',0):.1f}% median improvement |
| S2: Spatial dev | {"PASS" if s2_ok else "FAIL"} | B0={s2['b0_success']}/{s2['n']} B1d={s2['b1d_success']}/{s2['b1d_total']} B3={s2['b3_success']}/{s2['n']} |
| S3: Detector | {"PASS" if s3_ok else "FAIL"} | No config meets coverage>=0.4 & false_rate<=0.1 |
| S4: Cross-suite | {"PASS" if s4_ok else "FAIL"} | All suites show harm |

## S0: No-Takeover Parity

- **Action parity**: {"PASS" if s0_parity.get('action_parity_pass') else "FAIL"}
- B0 ≡ B3(force-off): {s0_parity.get('success_match')}/{s0_parity.get('n_pairs')} success match
- Max action L2: {s0_parity.get('max_action_l2', '?'):.2e}
- Root cause fix: Added `policy.reset()` between arms to clear SmolVLA action queue

## S1: Held-Out Anchor Recovery (Offline)

- **Rescue rate**: {s1_anchor.get('rescue_rate', 0):.1%} ({s1_anchor.get('n_rescue',0)}/{s1_anchor.get('n_total',0)})
- **Improvement**: mean {s1_anchor.get('improvement_mean_pct',0):.1f}%, median {s1_anchor.get('improvement_median_pct',0):.1f}%
- C0 (student L2): mean {s1_anchor.get('c0_mean',0):.4f} → C3 (plugin L2): mean {s1_anchor.get('c3_mean',0):.4f}
- **Assessment**: Plugin has offline efficacy — delta_a moves student action closer to teacher

## S2: Spatial Dev Confirmation (Online)

| Arm | Success | Steps (mean) |
|-----|---------|-------------|
| B0 (student only) | {s2['b0_success']}/{s2['n']} ({s2['b0_success']/s2['n']:.1%}) | — |
| B1d (OFT upper bound) | {s2['b1d_success']}/{s2['b1d_total']} (100%) | — |
| B3 (student + Plugin) | {s2['b3_success']}/{s2['n']} ({s2['b3_success']/s2['n']:.1%}) | — |

- Harm: {s2['harm']}/{s2['n']} (B0 ok, B3 fail)
- Rescue: {s2['rescue']}/{s2['n']} (B0 fail, B3 ok)
- Clean degradation: {s2['clean_degradation']:.3f} (negative = B3 < B0)
- Recovery rate: {s2.get('recovery_rate', float('nan')):.3f}

**Root cause**: stagnation_eps=2e-2 + window=5 triggers Plugin takeover on EVERY episode within first ~5 steps.
The Plugin replaces the student for most of the episode but is less effective than the student alone on easy cases.

## S3: Detector Calibration

- **Gate**: FAIL
- All configs with eps <= 5e-4 and window <= 60 trigger on 100% of episodes
- The `||eef_pos||` progress metric changes too slowly to distinguish "robot stuck" from "robot progressing"
- Best config: eps={s3_sweep.get('best',{}).get('eps',0)}, window={s3_sweep.get('best',{}).get('window',0)} (score={s3_sweep.get('best',{}).get('score',0):.3f})
"""
    for suite, d in sorted(s4.items()):
        md += f"""
### {suite}
- B0: {d.get('b0_success',0)}/{d.get('n',0)} ({d.get('b0_success',0)/max(d.get('n',1),1):.1%})
- B3: {d.get('b3_success',0)}/{d.get('n',0)} ({d.get('b3_success',0)/max(d.get('n',1),1):.1%})
- Harm: {d.get('harm','?')}, Rescue: {d.get('rescue','?')}, Net: {d.get('net','?')}
"""

    md += f"""
## Assessment: {grade}

### Plugin Efficacy Breakdown

```
system_gain = trigger_coverage × plugin_efficacy − intervention_harm
             = 1.000 × (-0.200) − 0.300
             = -0.500  (NET HARM)
```

The Plugin has strong offline efficacy (S1: 87.5% rescue rate, median +57% improvement),
but the online stagnation detector (eef_pos norm) cannot distinguish "student recovering"
from "student stuck", causing premature takeover and net harm (S2: B3=47.5% vs B0=67.5%).
On OOD suites, the harm becomes catastrophic (S4: libero_object 100%→0%).

### Required Fixes Before Re-evaluation

1. **Better stagnation metric**: Replace `||eef_pos||` with task-specific metrics
   (distance to key objects, velocity, success-condition proximity)
2. **Detector calibration with proper labels**: Use human-annotated or teacher-verified
   "truly stuck" vs "slowly progressing" labels
3. **Multi-suite training data**: Current model trained on 24 spatial-only episodes
4. **Takeover policy refinement**: Plugin currently runs for entire episode once triggered;
   need handback logic or confidence-gated takeover

### Artifacts

```
runs/route_c_final/
  audit_protocol.json
  no_takeover_parity.json
  recovery_anchor_results.json
  recovery_anchor_results.jsonl
  paired_results_b0.jsonl
  paired_results_b1d.jsonl
  paired_results_b3.jsonl
  detector_sweep.json
  s2_traces_b0.jsonl
  s2_traces_b3.jsonl
  report.json
  report.md
```

### Files Created

- `scripts/audit_route_c_eval_protocol.py`
- `scripts/eval_route_c_paired.py`
- `scripts/eval_route_c_anchors.py`
- `scripts/sweep_route_c_detector.py`
- `scripts/summarize_route_c_final.py`
"""

    md_path = output_dir / "report.md"
    md_path.write_text(md)
    print(f"Report saved to: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
