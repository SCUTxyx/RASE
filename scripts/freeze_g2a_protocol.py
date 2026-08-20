#!/usr/bin/env python3
"""Freeze the outcome-blind G2a Pi0Fast clean-LIBERO-10 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "rase-g2a-pi0fast-clean-direct-protocol/v1"


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def canonical_sha256(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_protocol(seed_base: int) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for task_index in range(1, 11):
        task_id = f"libero_10_{task_index:06d}"
        for init_state_id in range(8):
            episode_id = f"g2a-long-t{task_index:02d}-i{init_state_id:02d}"
            records.append(
                {
                    "episode_id": episode_id,
                    "suite": "libero_10",
                    "task_id": task_id,
                    "clean_task_index": task_index,
                    "init_state_id": init_state_id,
                    "environment_seed": stable_seed(
                        SCHEMA, seed_base, "environment", task_id, init_state_id
                    ),
                    "policy_seed": stable_seed(
                        SCHEMA, seed_base, "policy", "pi0fast_libero", task_id, init_state_id
                    ),
                }
            )
    protocol: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "frozen",
        "scientific_scope": "development eligibility screen; no held-out claim",
        "selection_uses_outcomes": False,
        "policy_id": "pi0fast_libero",
        "policy_path": "ckpts/pi0fast_libero",
        "tokenizer_path": "ckpts/paligemma_tokenizer_35e4f46",
        "action_tokenizer_path": "ckpts/pi0fast_action_tokenizer_79ae83e",
        "libero_flavor": "clean",
        "suite": "libero_10",
        "seed_base": int(seed_base),
        "num_steps": 10,
        "n_action_steps": 10,
        "observation_height": 360,
        "observation_width": 360,
        "n_tasks": 10,
        "episodes_per_task": 8,
        "n_episodes": 80,
        "gate": {
            "metric": "micro_episode_success_rate",
            "long_pair_eligible_interval": [0.30, 0.70],
            "lower_inclusive": True,
            "upper_inclusive": True,
            "below_interval_decision": "PI0FAST_TOO_WEAK_STOP_LONG_PAIR",
            "inside_interval_decision": "PROCEED_G2B_LONG",
            "above_interval_decision": "PI0FAST_DOMINATES_TRY_SPATIAL_PAIR",
            "uncertainty_reporting_only": [
                "wilson_95_interval",
                "task_cluster_bootstrap_95_interval",
            ],
            "note": "The point-estimate gate is preregistered; confidence intervals do not override it.",
        },
        "cost_reporting": [
            "environment_steps",
            "wall_seconds",
            "model_forward_calls",
            "action_select_seconds",
        ],
        "records": records,
    }
    protocol["records_sha256"] = canonical_sha256(records)
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=2026082002)
    args = parser.parse_args()
    protocol = build_protocol(args.seed_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: protocol[key] for key in ("n_episodes", "records_sha256", "protocol_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
