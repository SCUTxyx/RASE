#!/usr/bin/env python3
"""Phase 1: dynamic-boundary feasibility smoke on positive controls.

For each root: run the causal detector prefix until trigger (or max_steps),
freeze the boundary, then execute continue / requery / fallback with K=3
matched seeds.  Positive controls are the two known heterogeneous regions
(goal_000625@16 requery-wins, object_001041@8 continue-wins); at least two
fallback-dominant controls are preregistered.  Output: per-root boundary
step/rule and per-candidate success -> heterogeneous matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.forked_rollout import (  # noqa: E402
    InProcessLeRobotContinuation,
    load_lerobot_policy_bundle,
    restore_pool_state,
)
from rase.collect.pool_candidates import observation_from_libero_env  # noqa: E402
from rase.collect.state_pool import StatePool  # noqa: E402
from rase.vnext.boundary import DetectorConfig, DynamicBoundaryDetector  # noqa: E402
from scripts.collect_rase_vnext_discovery import (  # noqa: E402
    _policy_action,
    action_hash,
    isolated_force_inference,
    prefix_to_decision,
    rollout_policy,
)
from scripts.collect_r6b1_dynamic_boundaries import persistent_branch, preserve_rng_state  # noqa: E402


def stable_seed(*parts: object) -> int:
    token = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--policy-id", default="pi0fast.libero")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    roots = json.loads(args.roots_file.read_text())
    pool = StatePool(args.pool.resolve())
    bundle = load_lerobot_policy_bundle(
        args.policy_path, device="cuda", num_steps=10, n_action_steps=10,
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )
    from rase.oracle.client import OracleClient

    client = OracleClient(args.endpoint, timeout_ms=60_000)

    config = DetectorConfig(max_steps=args.max_steps)
    records: list[dict[str, Any]] = []
    for root_spec in roots:
        suite = str(root_spec["suite"])
        state_key = str(root_spec["state_key"])
        label = str(root_spec.get("label", ""))
        instruction_source = str(root_spec.get("instruction", ""))
        detector = DynamicBoundaryDetector(config)
        restored = restore_pool_state(
            pool, state_key, libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
        )
        source = InProcessLeRobotContinuation(
            bundle, seed=stable_seed("dyn-smoke", state_key, "source"),
            capture=True, capture_horizon=10,
        )
        entry: dict[str, Any] = {
            "suite": suite, "state_key": state_key, "label": label,
            "candidates": {}, "boundary": None, "trigger": None,
        }
        try:
            prefix = prefix_to_decision(
                restored, source, decision_step=None, detector=detector,
                max_steps=args.max_steps,
            )
            if not prefix["available"]:
                entry["prefix_available"] = False
                entry["reason"] = prefix.get("reason")
                records.append(entry)
                restored.close()
                continue
            entry["prefix_available"] = True
            entry["boundary"] = {
                "step": int(prefix["source_prefix_steps"]),
                "rule": prefix.get("boundary_rule"),
                "provenance": prefix.get("trigger_provenance"),
            }
            snapshot = prefix["snapshot"]
            instruction = str(prefix["instruction"]) or instruction_source
            decision_step = int(prefix["source_prefix_steps"])

            # continue: read boundary inference event (full chunk)
            cont_event = source.current_inference_event()
            cont_cursor = source.consumed_in_current_event()
            if cont_event is not None and cont_cursor < cont_event.env_chunk.shape[0]:
                continue_chunk = cont_event.env_chunk[cont_cursor:]
            else:
                continue_chunk = cont_event.env_chunk if cont_event is not None else None

            # requery: isolated forced inference
            requery_seed = stable_seed("dyn-smoke", state_key, "requery")
            requery_first, requery_event = isolated_force_inference(
                bundle, prefix["observation"], task=instruction,
                boundary_step=decision_step, generation_seed=requery_seed, horizon=10,
            )
            requery_chunk = requery_event.env_chunk

            # fallback: OFT persistent branch
            branch = restore_pool_state(pool, state_key, libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"))
            with preserve_rng_state():
                fallback = persistent_branch(
                    branch, snapshot, client, instruction, record_chunk_trace=True,
                    return_action_trace=True,
                )
            fallback_chunk = np.asarray(fallback.pop("action_trace"), dtype=np.float32)
            branch.close()

            candidates = {
                "continue.source": continue_chunk,
                "requery.source": requery_chunk,
                "fallback.persistent": fallback_chunk[:10],
            }
            for operator, chunk in candidates.items():
                if chunk is None or len(chunk) < 1:
                    entry["candidates"][operator] = {"success": [None] * args.repeats, "error": "no chunk"}
                    continue
                for rep in range(args.repeats):
                    branch = restore_pool_state(pool, state_key, libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"))
                    cont = InProcessLeRobotContinuation(
                        bundle, seed=stable_seed("dyn-smoke", state_key, operator, rep),
                        capture=True, capture_horizon=10,
                    )
                    observation = observation_from_libero_env(branch.handle.vector_env.envs[0])
                    result = rollout_policy(
                        branch, cont, observation=observation, instruction=instruction,
                        first_action=chunk[0],
                    )
                    entry["candidates"].setdefault(operator, {"success": [], "steps": []})
                    entry["candidates"][operator]["success"].append(bool(result["success"]))
                    entry["candidates"][operator]["steps"].append(int(result["post_decision_env_steps"]))
                    branch.close()
            # heterogeneous verdict for this root
            succ = {
                op: [v for v in entry["candidates"].get(op, {}).get("success", []) if v is not None]
                for op in ("continue.source", "requery.source", "fallback.persistent")
            }
            fb_ok = any(succ.get("fallback.persistent", []))
            others_ok = any(v for op in ("continue.source", "requery.source") for v in succ.get(op, []))
            entry["verdict"] = (
                "heterogeneous" if not fb_ok and others_ok else
                "all_fail" if not fb_ok and not others_ok else
                "fallback_dominates"
            )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            restored.close()
        records.append(entry)

    report = {
        "schema_version": "rase-vnext-dynamic-smoke/v1",
        "policy_id": args.policy_id,
        "detector_config": {
            "max_steps": args.max_steps, "first_eligible": config.first_eligible_step,
            "phase": (config.phase_window, config.phase_rel_depth, config.phase_threshold),
            "stagnation": (config.stagnation_window, config.stagnation_norm),
        },
        "roots": records,
        "summary": {
            "heterogeneous": sum(1 for r in records if r.get("verdict") == "heterogeneous"),
            "fallback_dominates": sum(1 for r in records if r.get("verdict") == "fallback_dominates"),
            "all_fail": sum(1 for r in records if r.get("verdict") == "all_fail"),
            "triggered": sum(1 for r in records if (r.get("boundary") or {}).get("rule") != "none"),
            "boundary_steps": [(r["state_key"][:12], (r.get("boundary") or {}).get("step"), (r.get("boundary") or {}).get("rule")) for r in records],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
