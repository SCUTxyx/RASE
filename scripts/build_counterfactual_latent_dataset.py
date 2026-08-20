#!/usr/bin/env python3
"""Join exact-state PRE-A3 outcomes with deployment-time latent transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FEATURES = ("state_key", "task_id", "latent", "action", "next_latent")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opportunity-audit", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True,
                    help="One JSONL row per exact state with frozen online features")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-non-ready", action="store_true", help="diagnostic only")
    args = ap.parse_args()
    audit = json.loads(args.opportunity_audit.read_text())
    if audit.get("status") != "ready" and not args.allow_non_ready:
        raise SystemExit(f"Opportunity gate is {audit.get('status')}; selector dataset is blocked")
    operators = list(audit.get("operators", []))
    if "CONTINUE" not in operators:
        raise SystemExit("Audit has no strict CONTINUE operator")
    outcomes = {}
    for row in audit.get("per_state", []):
        successes = set(row["successful_operators"])
        outcomes[str(row["state_key"])] = {
            "task_id": str(row["task_id"]), "suite": str(row.get("suite", "unknown")),
            "operator_success": {op: int(op in successes) for op in operators}}
    features = load_jsonl(args.features)
    seen, joined = set(), []
    for row in features:
        missing = [key for key in REQUIRED_FEATURES if key not in row]
        if missing: raise ValueError(f"feature row missing {missing}: {row.get('state_key')}")
        key = str(row["state_key"])
        if key in seen: raise ValueError(f"duplicate feature state_key: {key}")
        seen.add(key)
        if key not in outcomes: continue
        outcome = outcomes[key]
        if str(row["task_id"]) != outcome["task_id"]:
            raise ValueError(f"task mismatch for {key}: {row['task_id']} vs {outcome['task_id']}")
        if len(row["latent"]) != len(row["next_latent"]):
            raise ValueError(f"latent transition dimension mismatch for {key}")
        joined.append({**row, **outcome})
    missing_features = sorted(set(outcomes) - seen)
    coverage = len(joined) / max(1, len(outcomes))
    if coverage < .95:
        raise SystemExit(f"Only {len(joined)}/{len(outcomes)} states joined ({coverage:.1%}); "
                         f"examples missing={missing_features[:10]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in joined))
    report = {"schema_version": "rase-counterfactual-latent-dataset/v1",
              "n_outcome_states": len(outcomes), "n_feature_states": len(features),
              "n_joined": len(joined), "coverage": coverage, "operators": operators,
              "tasks": sorted({row["task_id"] for row in joined}),
              "latent_dim": len(joined[0]["latent"]),
              "action_dim": len(joined[0]["action"]),
              "output": str(args.output.resolve())}
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
