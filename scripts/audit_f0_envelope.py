#!/usr/bin/env python3
"""PRE-C0-R1 Step B1/B2: F0 deployment envelope audit via trace replay.

Audits magnitude, onset, duration, and exit strategies for F0 activation
using existing Always-On trace data. No additional env rollouts needed.

Inputs:
  - f0_controls_summary.json (magnitude audit, already collected)
  - always_on_f0_*.jsonl (per-episode traces with step-level action/progress)

Outputs:
  - runs/pre_c0_r1/envelope_audit.json
  - runs/pre_c0_r1/envelope_audit.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit_magnitude(controls_summary_path: Path, controls_dir: Path) -> dict:
    summary = json.loads(controls_summary_path.read_text(encoding="utf-8"))
    arms = {}
    for fpath in sorted(controls_dir.glob("f0_controls_*.jsonl")):
        if "summary" in fpath.name:
            continue
        arm_name = fpath.stem.replace("f0_controls_", "")
        episodes = [json.loads(l) for l in fpath.read_text().strip().split("\n") if l.strip()]
        arms[arm_name] = episodes

    task_results = defaultdict(dict)
    for arm_name, episodes in arms.items():
        for ep in episodes:
            tid = ep.get("task_id", "unknown")
            task_results[tid][arm_name] = {
                "success": sum(1 for e in episodes if e["task_id"] == tid and e["success"]),
                "total": sum(1 for e in episodes if e["task_id"] == tid),
            }

    alpha_arms = ["B0_baseline", "alpha_0.00", "alpha_0.25", "alpha_0.50",
                  "alpha_0.75", "alpha_1.00", "alpha_1.25"]
    control_arms = ["sign_flip", "random", "dim_shuffled"]

    global_rates = {}
    for arm_name in alpha_arms + control_arms:
        if arm_name in arms:
            eps_list = arms[arm_name]
            global_rates[arm_name] = sum(1 for e in eps_list if e["success"]) / max(len(eps_list), 1)

    best_alpha = max(
        [a for a in alpha_arms if a in global_rates],
        key=lambda a: global_rates[a]
    )
    alpha_1_rate = global_rates.get("alpha_1.00", 0)
    is_1_optimal = abs(alpha_1_rate - global_rates.get(best_alpha, 0)) < 1e-6

    per_task_optima = {}
    for tid, tdata in task_results.items():
        best_rate = -1
        best_a = "unknown"
        for a in alpha_arms:
            if a in tdata and tdata[a]["total"] > 0:
                rate = tdata[a]["success"] / tdata[a]["total"]
                if rate > best_rate:
                    best_rate = rate
                    best_a = a
        per_task_optima[tid] = best_a

    control_rank = sorted(
        [a for a in control_arms if a in global_rates],
        key=lambda a: global_rates[a], reverse=True
    )

    return {
        "global_alpha_rates": global_rates,
        "best_alpha": best_alpha,
        "alpha_1_0_rate": alpha_1_rate,
        "is_1_0_optimal": is_1_optimal,
        "per_task_optimum": per_task_optima,
        "control_ranking": control_rank,
        "controls_pass": (
            global_rates.get("sign_flip", 1) <
            global_rates.get("random", 1) <
            global_rates.get("dim_shuffled", 1) <
            global_rates.get("alpha_1.00", 0)
        ),
    }


def load_always_on_traces(trace_paths: list[Path]) -> list[dict]:
    episodes = []
    for path in trace_paths:
        if not path.is_file():
            continue
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                ep = json.loads(line)
                if ep.get("trace"):
                    episodes.append(ep)
    return episodes


def compute_trace_statistics(episodes: list[dict]) -> dict:
    successes = [ep for ep in episodes if ep.get("success")]
    failures = [ep for ep in episodes if not ep.get("success")]
    return {
        "n_episodes": len(episodes),
        "n_success": len(successes),
        "n_fail": len(failures),
        "success_rate": len(successes) / max(len(episodes), 1),
    }


def simulate_envelope_strategies(episodes: list[dict]) -> dict:
    strategies = {}

    for stag_window in [5, 10, 20]:
        for stag_eps in [0.01, 0.02, 0.05]:
            for dur in [4, 8, 16]:
                name = f"stag_w{stag_window}_e{stag_eps}_d{dur}"
                results = _simulate_stagnation(episodes, stag_window, stag_eps, dur)
                strategies[name] = results

    for tau_on in [0.01, 0.02, 0.05]:
        for tau_off in [0.005, 0.01, 0.02]:
            if tau_off >= tau_on:
                continue
            name = f"hyst_on{tau_on}_off{tau_off}"
            results = _simulate_hysteresis(episodes, tau_on, tau_off)
            strategies[name] = results

    return strategies


def _compute_stagnation(progress_seq: list[float], window: int) -> float:
    if len(progress_seq) < window:
        return float("inf")
    return float(np.std(progress_seq[-window:]))


def _simulate_stagnation(episodes, stag_window, stag_eps, dur):
    results = []
    for ep in episodes:
        trace = ep.get("trace", [])
        progress_seq = []
        active_count = 0
        in_activation = False
        activation_remaining = 0
        activation_events = 0

        for frame in trace:
            progress_seq.append(frame.get("progress", 0.0))
            if len(progress_seq) > stag_window * 2:
                progress_seq = progress_seq[-stag_window * 2:]

            if not in_activation and len(progress_seq) >= stag_window:
                if _compute_stagnation(progress_seq, stag_window) < stag_eps:
                    in_activation = True
                    activation_remaining = dur
                    activation_events += 1

            if in_activation:
                active_count += 1
                activation_remaining -= 1
                if activation_remaining <= 0:
                    in_activation = False

        results.append({
            "ep_success": ep.get("success", False),
            "active_count": active_count,
            "activation_events": activation_events,
            "total_steps": len(trace),
        })

    n = len(results)
    return {
        "n_episodes": n,
        "mean_active_per_ep": float(np.mean([r["active_count"] for r in results])) if n else 0,
        "mean_events_per_ep": float(np.mean([r["activation_events"] for r in results])) if n else 0,
    }


def _simulate_hysteresis(episodes, tau_on, tau_off):
    results = []
    for ep in episodes:
        trace = ep.get("trace", [])
        progress_seq = []
        active_count = 0
        active = False
        toggle_count = 0
        prev_state = False

        for frame in trace:
            progress_seq.append(frame.get("progress", 0.0))
            if len(progress_seq) > 30:
                progress_seq = progress_seq[-30:]

            if len(progress_seq) < 10:
                continue

            recent_std = float(np.std(progress_seq[-10:]))

            if not active and recent_std < tau_on:
                active = True
            elif active and recent_std > tau_off:
                active = False

            if active:
                active_count += 1
            if active != prev_state:
                toggle_count += 1
                prev_state = active

        results.append({
            "ep_success": ep.get("success", False),
            "active_count": active_count,
            "toggle_count": toggle_count,
            "total_steps": len(trace),
        })

    n = len(results)
    return {
        "n_episodes": n,
        "mean_active_per_ep": float(np.mean([r["active_count"] for r in results])) if n else 0,
        "mean_toggles_per_ep": float(np.mean([r["toggle_count"] for r in results])) if n else 0,
    }


def find_best_envelope(strategies: dict) -> dict:
    best_stag = None
    best_stag_score = float("inf")
    for name, stats in strategies.items():
        if not name.startswith("stag_"):
            continue
        active_mean = stats.get("mean_active_per_ep", 0)
        events_mean = stats.get("mean_events_per_ep", 0)
        target_active = 40
        score = abs(active_mean - target_active) + 2 * abs(events_mean - 1)
        if score < best_stag_score:
            best_stag_score = score
            best_stag = name

    best_hyst = None
    best_hyst_score = float("inf")
    for name, stats in strategies.items():
        if not name.startswith("hyst_"):
            continue
        active_mean = stats.get("mean_active_per_ep", 0)
        toggles_mean = stats.get("mean_toggles_per_ep", 0)
        score = abs(active_mean - 40) + 5 * abs(toggles_mean - 1)
        if score < best_hyst_score:
            best_hyst_score = score
            best_hyst = name

    return {
        "best_stagnation_strategy": best_stag,
        "best_hysteresis_strategy": best_hyst,
    }


def design_safety_envelope() -> dict:
    return {
        "hard_constraints": [
            {"name": "action_rate_limit", "delta_max": 0.3},
            {"name": "f0_max_continuous_steps", "T_max": 8},
            {"name": "workspace_bounds", "enabled": True},
            {"name": "gripper_safety", "enabled": True},
        ],
        "soft_constraints": [
            {"name": "post_recovery_disable", "N": 3},
            {"name": "cooldown_after_handback", "cooldown": 4},
            {"name": "f0_norm_limit", "max_norm": 0.5},
        ],
        "bounded_takeover_params": {
            "max_takeover_steps": 8,
            "handback_window": 3,
            "cooldown_steps": 4,
            "mix_ramp": [0.0, 0.3, 0.6, 1.0],
            "delta_clip": 0.5,
            "action_rate_limit": 0.1,
        },
    }


def format_report(magnitude: dict, trace_stats: dict,
                  safety: dict, best_env: dict) -> str:
    lines = ["# PRE-C0-R1: F0 Deployment Envelope Audit\n"]
    lines.append("## B1: Magnitude Audit\n")
    lines.append("| Arm | Success Rate |")
    lines.append("|-----|-------------|")
    for arm in ["B0_baseline", "alpha_0.00", "alpha_0.25", "alpha_0.50",
                 "alpha_0.75", "alpha_1.00", "alpha_1.25",
                 "sign_flip", "random", "dim_shuffled"]:
        rate = magnitude["global_alpha_rates"].get(arm)
        if rate is not None:
            mark = " **<-- BEST**" if arm == magnitude["best_alpha"] else ""
            lines.append(f"| {arm} | {rate:.1%}{mark} |")
    lines.append(f"\nAlpha=1.0 optimal: **{magnitude['is_1_0_optimal']}**")
    lines.append(f"Controls pass: **{magnitude['controls_pass']}**\n")

    lines.append("## B2: Trace Statistics\n")
    lines.append(f"- Episodes: {trace_stats['n_episodes']}")
    lines.append(f"- Success: {trace_stats['n_success']}")
    lines.append(f"- Fail: {trace_stats['n_fail']}")
    lines.append(f"- Rate (AO): {trace_stats['success_rate']:.1%}\n")

    lines.append("## B2: Recommended Envelope\n")
    lines.append(f"- Stagnation: **{best_env.get('best_stagnation_strategy', 'N/A')}**")
    lines.append(f"- Hysteresis: **{best_env.get('best_hysteresis_strategy', 'N/A')}**\n")

    lines.append("## B3: Safety Envelope\n")
    for c in safety["hard_constraints"]:
        params = {k: v for k, v in c.items() if k != "name"}
        lines.append(f"- **{c['name']}**: {params}")
    for c in safety["soft_constraints"]:
        params = {k: v for k, v in c.items() if k != "name"}
        lines.append(f"- **{c['name']}**: {params}")

    params = safety["bounded_takeover_params"]
    lines.append(f"\nBounded takeover: max_steps={params['max_takeover_steps']}, "
                 f"handback={params['handback_window']}, cooldown={params['cooldown_steps']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-summary", type=Path,
                        default=ROOT / "runs/pre_c0_r0/f0_controls_summary.json")
    parser.add_argument("--controls-dir", type=Path,
                        default=ROOT / "runs/pre_c0_r0")
    parser.add_argument("--always-on-dir", type=Path,
                        default=ROOT / "runs/pre_c0_r0")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/pre_c0_r1")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("B1: Magnitude audit...")
    magnitude = audit_magnitude(args.controls_summary, args.controls_dir)
    print(f"  Best alpha: {magnitude['best_alpha']} (1.0 optimal: {magnitude['is_1_0_optimal']})")

    print("B2: Trace replay audit...")
    trace_paths = list(args.always_on_dir.glob("always_on_f0_*.jsonl"))
    print(f"  Found {len(trace_paths)} trace files")
    episodes = load_always_on_traces(trace_paths)
    print(f"  Loaded {len(episodes)} episodes")
    trace_stats = compute_trace_statistics(episodes)
    print(f"  AO success rate: {trace_stats['success_rate']:.1%}")

    strategies = simulate_envelope_strategies(episodes)
    print(f"  Simulated {len(strategies)} strategies")
    best_env = find_best_envelope(strategies)
    print(f"  Best stag: {best_env['best_stagnation_strategy']}")
    print(f"  Best hyst: {best_env['best_hysteresis_strategy']}")

    print("B3: Safety envelope design...")
    safety = design_safety_envelope()

    envelope_result = {
        "magnitude_audit": magnitude,
        "trace_statistics": {k: v for k, v in trace_stats.items()},
        "best_strategies": best_env,
        "safety_envelope": safety,
    }

    json_path = output_dir / "envelope_audit.json"
    json_path.write_text(json.dumps(envelope_result, indent=2, default=str) + "\n",
                         encoding="utf-8")
    print(f"Saved: {json_path}")

    md = format_report(magnitude, trace_stats, safety, best_env)
    md_path = output_dir / "envelope_audit.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
