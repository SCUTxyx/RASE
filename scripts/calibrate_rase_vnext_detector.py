#!/usr/bin/env python3
"""Phase 1 detector calibration: record source prefix action norms to set
phase/stagnation thresholds (one preregistered diagnostic adjustment)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.forked_rollout import (  # noqa: E402
    InProcessLeRobotContinuation,
    load_lerobot_policy_bundle,
    restore_pool_state,
)
from rase.collect.state_pool import StatePool  # noqa: E402
from scripts.collect_rase_vnext_discovery import _policy_action, prefix_to_decision  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()

    roots = json.loads(args.roots_file.read_text())
    pool = StatePool(args.pool.resolve())
    bundle = load_lerobot_policy_bundle(
        args.policy_path, device="cuda", num_steps=10, n_action_steps=10,
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )

    class _NormRecorder:
        def __init__(self) -> None:
            self.norms: list[float] = []
            self.proprio: list[np.ndarray] = []

        def update(self, action: np.ndarray, proprio: np.ndarray | None = None) -> None:
            self.norms.append(float(np.linalg.norm(action)))
            if proprio is not None:
                self.proprio.append(np.asarray(proprio, dtype=np.float64).reshape(-1))

        def evaluate(self, timestep: int, now_s: float):
            return None  # never trigger; just record

    report: dict = {"schema_version": "phase1-norm-calibration/v1", "roots": {}}
    for root_spec in roots:
        state_key = str(root_spec["state_key"])
        recorder = _NormRecorder()
        restored = restore_pool_state(
            pool, state_key, libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
        )
        source = InProcessLeRobotContinuation(
            bundle, seed=12345, capture=True, capture_horizon=10,
        )
        try:
            _p = prefix_to_decision(
                restored, source, decision_step=None, detector=recorder,
                max_steps=args.steps,
            )
            print("PREFIX", state_key[:12], "available=", _p.get("available"), "reason=", _p.get("reason"), "elapsed=", _p.get("elapsed"), "steps=", _p.get("source_prefix_steps"), "exc=", str(_p.get("exception"))[:140], flush=True)
            norms = np.asarray(recorder.norms, dtype=np.float64)
            proprio_deltas: list[float] = []
            for i in range(1, len(recorder.proprio)):
                proprio_deltas.append(float(np.linalg.norm(recorder.proprio[i] - recorder.proprio[i - 1])))
            deltas = np.asarray(proprio_deltas, dtype=np.float64) if proprio_deltas else np.zeros(0)
            report["roots"][state_key[:20]] = {
                "steps": len(norms),
                "norm_min": round(float(norms.min()), 4),
                "norm_p10": round(float(np.percentile(norms, 10)), 4),
                "norm_p25": round(float(np.percentile(norms, 25)), 4),
                "norm_median": round(float(np.median(norms)), 4),
                "norm_max": round(float(norms.max()), 4),
                "first_below_0_1": int(np.argmax(norms < 0.1)) if (norms < 0.1).any() else None,
                "first_below_0_2": int(np.argmax(norms < 0.2)) if (norms < 0.2).any() else None,
                "min_window5_mean": round(float(min(
                    norms[i:i + 5].mean() for i in range(len(norms) - 4)
                )), 4) if len(norms) >= 5 else None,
                "proprio_delta_min": round(float(deltas.min()), 5) if len(deltas) else None,
                "proprio_delta_p10": round(float(np.percentile(deltas, 10)), 5) if len(deltas) else None,
                "proprio_delta_p25": round(float(np.percentile(deltas, 25)), 5) if len(deltas) else None,
                "proprio_delta_median": round(float(np.median(deltas)), 5) if len(deltas) else None,
                "proprio_delta_min_window5_mean": round(float(min(
                    deltas[i:i + 4].mean() for i in range(len(deltas) - 3)
                )), 5) if len(deltas) >= 4 else None,
            }
        except Exception as exc:
            report["roots"][state_key[:20]] = {"error": str(exc)[:200]}
        finally:
            restored.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
