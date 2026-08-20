#!/usr/bin/env python3
"""Validate the vNext protocol and refuse activation with unresolved fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_OPERATOR_IDS = {
    "continue.source", "requery.source", "resample.source",
    "fallback.persistent", "abort.safe",
}
REQUIRED_SEEDS = {
    "init_state_id", "environment_seed", "source_sampling_seed",
    "operator_seed", "exact_repeat_replica",
}


def unresolved_paths(value: object, prefix: str = "") -> list[str]:
    result: list[str] = []
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(unresolved_paths(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(unresolved_paths(child, f"{prefix}[{index}]"))
    return result


def validate(config: dict, *, allow_draft: bool) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != "rase-vnext-protocol/v1":
        errors.append("unexpected schema_version")
    if config.get("status") not in {"draft_locked", "frozen"}:
        errors.append("status must be draft_locked or frozen")
    operators = config.get("operators", [])
    operator_ids = [row.get("operator_id") for row in operators]
    if set(operator_ids) != REQUIRED_OPERATOR_IDS or len(operator_ids) != 5:
        errors.append("operator prior must contain each of the five semantic operators exactly once")
    for row in operators:
        if row.get("kind") != "resample" and row.get("candidate_ids"):
            errors.append(f"only resample may own candidates: {row.get('operator_id')}")
    collection = config.get("collection", {})
    if collection.get("discovery_repeats") != 3 or collection.get("confirmation_repeats") != 5:
        errors.append("fixed K must be discovery=3 and confirmation=5")
    if collection.get("outcome_dependent_sampling") is not False:
        errors.append("outcome-dependent sampling must be false")
    if set(collection.get("seed_layers", [])) != REQUIRED_SEEDS:
        errors.append("all five independent seed layers are required")
    points = collection.get("decision_points", [])
    point_ids = [row.get("decision_point_id") for row in points if isinstance(row, dict)]
    if len(points) != 2 or len(set(point_ids)) != 2:
        errors.append("exactly two distinct causal decision points are required")
    for row in points:
        if not isinstance(row, dict) or row.get("rule") != "source_elapsed_step":
            errors.append("decision points must use the causal source_elapsed_step rule")
            continue
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append("source_elapsed_step decision values must be positive integers")
    unresolved = unresolved_paths({"utility": config.get("utility"), "gates": config.get("gates")})
    if config.get("status") == "frozen" and unresolved:
        errors.append("frozen protocol contains unresolved fields: " + ", ".join(unresolved))
    if config.get("status") == "draft_locked" and not allow_draft:
        errors.append("protocol is draft_locked; use --allow-draft for no-GPU structural checks")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    errors = validate(config, allow_draft=args.allow_draft)
    result = {
        "status": "PASS_DRAFT_STRUCTURE" if not errors and config["status"] == "draft_locked" else (
            "PASS_FROZEN" if not errors else "FAIL"
        ),
        "config_status": config.get("status"),
        "unresolved": unresolved_paths({"utility": config.get("utility"), "gates": config.get("gates")}),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
