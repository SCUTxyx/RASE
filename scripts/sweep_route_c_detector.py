#!/usr/bin/env python3
"""S3: Detector calibration sweep.

Sweeps stagnation_eps × stagnation_window on episode traces to find the
Pareto-optimal detector configuration (maximize trigger_coverage × (1 - false_trigger_rate)).

Online: re-runs a subset of episodes with different detector parameters,
collecting progress traces and analyzing stagnation triggers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def detect_stagnation(progress_vals: list[float], eps: float, window: int) -> int:
    """Return first timestep (0-indexed) where stagnation is detected, or -1."""
    for t in range(window, len(progress_vals)):
        if np.std(progress_vals[t - window:t]) < eps:
            return t
    return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, default=None,
                        help="pre-collected trace jsonl")
    parser.add_argument("--eps-values", type=float, nargs="+",
                        default=[1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1])
    parser.add_argument("--window-values", type=int, nargs="+",
                        default=[3, 5, 10, 15, 20])
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.trace_jsonl:
        # Offline sweep: use pre-collected progress traces
        traces = [json.loads(line) for line in open(args.trace_jsonl)]
        print(f"Loaded {len(traces)} episode traces")
    else:
        print("No trace file provided. Use --trace-jsonl for offline sweep.")
        print("Collecting traces fresh (not implemented). Run B0 with traces first.")
        return 1

    # Compute sweep
    sweep_results = []
    for eps in args.eps_values:
        for window in args.window_values:
            triggers = 0
            true_triggers = 0  # stagnation detected AND episode eventually failed
            false_triggers = 0  # stagnation detected BUT episode eventually succeeded
            lead_times = []  # steps from trigger to episode end
            total_episodes = len(traces)

            for trace_entry in traces:
                trace = trace_entry.get("trace", [])
                success = trace_entry.get("success", False)
                progress_vals = [t.get("progress", 0.0) for t in trace]

                trigger_step = detect_stagnation(progress_vals, eps, window)
                if trigger_step >= 0:
                    triggers += 1
                    if success:
                        false_triggers += 1
                    else:
                        true_triggers += 1
                    lead_times.append(len(progress_vals) - trigger_step)

            trigger_coverage = triggers / total_episodes if total_episodes > 0 else 0
            false_trigger_rate = false_triggers / triggers if triggers > 0 else 0

            score = trigger_coverage * (1 - false_trigger_rate)

            sweep_results.append({
                "eps": eps,
                "window": window,
                "triggers": triggers,
                "true_triggers": true_triggers,
                "false_triggers": false_triggers,
                "trigger_coverage": round(trigger_coverage, 4),
                "false_trigger_rate": round(false_trigger_rate, 4),
                "score": round(score, 4),
                "mean_lead_time": float(np.mean(lead_times)) if lead_times else 0,
            })

    # Sort by score descending
    sweep_results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 60)
    print("DETECTOR SWEEP RESULTS")
    print("=" * 60)
    print(f"{'eps':<10} {'window':<8} {'triggers':<10} {'coverage':<10} "
          f"{'false_rate':<10} {'score':<10} {'lead_time':<10}")
    print("-" * 60)
    top10 = sweep_results[:10]
    for r in top10:
        print(f"{r['eps']:<10.0e} {r['window']:<8} {r['triggers']:<10} "
              f"{r['trigger_coverage']:<10.3f} {r['false_trigger_rate']:<10.3f} "
              f"{r['score']:<10.3f} {r['mean_lead_time']:<10.1f}")

    best = top10[0]
    print()
    print("=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"  stagnation_eps: {best['eps']}")
    print(f"  stagnation_window: {best['window']}")
    print(f"  trigger_coverage: {best['trigger_coverage']:.3f}")
    print(f"  false_trigger_rate: {best['false_trigger_rate']:.3f}")
    print(f"  score: {best['score']:.3f}")

    # Find configs meeting gate: coverage >= 0.4 AND false_rate <= 0.1
    qualifying = [r for r in sweep_results
                  if r["trigger_coverage"] >= 0.4 and r["false_trigger_rate"] <= 0.1]
    qualifying.sort(key=lambda x: x["score"], reverse=True)

    print()
    print(f"Qualifying configs (coverage>=0.4, false_rate<=0.1): {len(qualifying)}")
    for r in qualifying[:5]:
        print(f"  eps={r['eps']:.0e} window={r['window']} "
              f"coverage={r['trigger_coverage']:.3f} "
              f"false_rate={r['false_trigger_rate']:.3f}")

    gate_pass = len(qualifying) > 0
    print(f"\n  DETECTOR GATE: {'PASS' if gate_pass else 'FAIL'}")

    # Save
    results = {
        "sweep": sweep_results,
        "best": best,
        "qualifying": qualifying,
        "gate_pass": gate_pass,
    }
    (output_dir / "detector_sweep.json").write_text(
        json.dumps(results, indent=2) + "\n")

    # Freeze optimal params into protocol
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol["plugin_config"]["stagnation_eps"] = best["eps"]
    protocol["plugin_config"]["stagnation_window"] = best["window"]
    (output_dir / "protocol_frozen.json").write_text(
        json.dumps(protocol, indent=2) + "\n")
    print(f"Frozen protocol saved to: {output_dir / 'protocol_frozen.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
