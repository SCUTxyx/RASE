#!/usr/bin/env python3
"""Summarize PRE-C0 deviation stage coverage, ordering, fallback, and reliability."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "rase-pre-c0-deviation-timeline-analysis/v1"
STAGES = (
    ("T0", "last_stable"),
    ("T1", "first_deviation"),
    ("T2", "sustained_deviation"),
    ("T3", "failure_in_progress"),
    ("T4", "terminal"),
)


def analyze(keys: Mapping[str, Any]) -> dict[str, Any]:
    episodes = keys.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("keys JSON must contain at least one episode")

    stage_ids = [stage for stage, _ in STAGES]
    stage_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    order_violations: list[dict[str, Any]] = []
    duplicate_key_episodes: list[str] = []
    fallback_episodes: list[str] = []
    unreliable_episodes: list[str] = []
    scores: list[float] = []
    coverage: list[float] = []
    by_suite: dict[str, Counter[str]] = defaultdict(Counter)
    by_cell: dict[str, Counter[str]] = defaultdict(Counter)

    for episode in episodes:
        episode_id = str(episode["episode_id"])
        stages = episode.get("stages") or {}
        missing = [stage for stage in stage_ids if stage not in stages]
        for stage in stage_ids:
            if stage in stages:
                stage_counts[stage] += 1
        if missing:
            order_violations.append({"episode_id": episode_id, "missing_stages": missing})
        else:
            indices = [int(stages[stage]["index"]) for stage in stage_ids]
            steps = [int(stages[stage]["step"]) for stage in stage_ids]
            if any(right <= left for left, right in zip(indices, indices[1:], strict=False)):
                order_violations.append(
                    {
                        "episode_id": episode_id,
                        "reason": "non_monotonic_indices",
                        "indices": indices,
                    }
                )
            if any(right <= left for left, right in zip(steps, steps[1:], strict=False)):
                order_violations.append(
                    {
                        "episode_id": episode_id,
                        "reason": "non_monotonic_steps",
                        "steps": steps,
                    }
                )
            state_keys = [str(stages[stage]["state_key"]) for stage in stage_ids]
            if len(set(state_keys)) != len(state_keys):
                duplicate_key_episodes.append(episode_id)

        fallback = bool(episode.get("temporal_fallback"))
        reliability = episode.get("reliability") or {}
        reliable = bool(reliability.get("reliable"))
        if fallback:
            fallback_episodes.append(episode_id)
        fallback_reasons.update(
            str(reason) for reason in episode.get("temporal_fallback_reasons", [])
        )
        if not reliable:
            unreliable_episodes.append(episode_id)
        scores.append(float(reliability.get("score", 0.0)))
        coverage.append(float(reliability.get("signal_coverage", 0.0)))
        for group, target in (
            (str(episode.get("suite") or "unknown"), by_suite),
            (str(episode.get("cell") or "unknown"), by_cell),
        ):
            target[group]["episodes"] += 1
            target[group]["fallback"] += fallback
            target[group]["reliable"] += reliable

    count = len(episodes)

    def group_summary(groups: Mapping[str, Counter[str]]) -> dict[str, Any]:
        return {
            name: {
                "episodes": values["episodes"],
                "fallback_episodes": values["fallback"],
                "fallback_rate": values["fallback"] / values["episodes"],
                "reliable_episodes": values["reliable"],
                "reliable_rate": values["reliable"] / values["episodes"],
            }
            for name, values in sorted(groups.items())
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": keys.get("schema_version"),
        "n_episodes": count,
        "stage_counts": {stage: stage_counts[stage] for stage in stage_ids},
        "ordering": {
            "strict_order_episodes": count - len({row["episode_id"] for row in order_violations}),
            "all_strict": not order_violations,
            "violations": order_violations,
            "duplicate_key_episodes": sorted(duplicate_key_episodes),
        },
        "temporal_fallback": {
            "episodes": len(fallback_episodes),
            "rate": len(fallback_episodes) / count,
            "episode_ids": sorted(fallback_episodes),
            "reason_counts": dict(sorted(fallback_reasons.items())),
        },
        "reliability": {
            "reliable_episodes": count - len(unreliable_episodes),
            "reliable_rate": (count - len(unreliable_episodes)) / count,
            "unreliable_episode_ids": sorted(unreliable_episodes),
            "mean_score": sum(scores) / count,
            "min_score": min(scores),
            "mean_signal_coverage": sum(coverage) / count,
            "min_signal_coverage": min(coverage),
        },
        "by_suite": group_summary(by_suite),
        "by_cell": group_summary(by_cell),
        "qc_pass": (
            not order_violations
            and not duplicate_key_episodes
            and all(stage_counts[stage] == count for stage in stage_ids)
        ),
        "provenance": {
            "analyzer": "scripts/analyze_deviation_timeline.py",
            "source_design_sha256": (keys.get("provenance") or {}).get("design_sha256"),
            "source_pool_manifest_sha256": (keys.get("provenance") or {}).get(
                "pool_manifest_sha256"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --overwrite")
    result = analyze(json.loads(args.keys_json.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_episodes": result["n_episodes"],
                "qc_pass": result["qc_pass"],
                "fallback_rate": result["temporal_fallback"]["rate"],
                "reliable_rate": result["reliability"]["reliable_rate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
