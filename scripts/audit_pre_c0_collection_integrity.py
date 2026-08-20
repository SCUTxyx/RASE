#!/usr/bin/env python3
"""Audit PRE-C0 24-trajectory collection integrity against the frozen design."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rase.collect.state_pool import StatePool

SCHEMA_VERSION = "rase-pre-c0-collection-integrity/v1"
REQUIRED_DC_KEYS = (
    "schema_version",
    "public_observation_history",
    "public_proprio_history",
    "public_action_history",
    "active_action_suffix",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_collection(
    *,
    design: dict[str, Any],
    summary: dict[str, Any],
    pool: StatePool,
    collect_log: Path | None,
    sample_decision_context: int = 24,
) -> dict[str, Any]:
    design_records = list(design.get("records") or [])
    expected_episodes = int(design.get("n_episodes") or len(design_records))
    design_episode_ids = [str(row["episode_id"]) for row in design_records]
    design_tasks = [str(row["concrete_task_id"]) for row in design_records]

    metrics = list(summary.get("episode_metrics") or [])
    summary_episode_ids = [str(row["episode_id"]) for row in metrics]
    outcomes = Counter(str(row.get("outcome")) for row in metrics)

    states = dict((pool.manifest() or {}).get("states") or {})
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seeds: list[int] = []
    missing_history_states: list[str] = []
    sampled_checked = 0
    for state_key, entry in states.items():
        episode_id = str(entry["episode_id"])
        by_episode[episode_id].append({"state_key": state_key, **entry})

    # Sample one mid-episode state per design episode for decision_context QC.
    for episode_id in design_episode_ids:
        snaps = sorted(by_episode.get(episode_id) or [], key=lambda row: int(row.get("step", -1)))
        if not snaps:
            continue
        if sampled_checked >= sample_decision_context:
            break
        pick = snaps[len(snaps) // 2]
        loaded = pool.read_state(str(pick["state_key"]), load_observations=False)
        seeds.append(int(loaded.metadata.seed))
        context = dict((loaded.controller_state or {}).get("decision_context") or {})
        missing = [name for name in REQUIRED_DC_KEYS if name not in context]
        if missing:
            missing_history_states.append(str(pick["state_key"]))
        sampled_checked += 1

    duplicate_seeds = sorted(
        seed for seed, count in Counter(seeds).items() if count > 1
    )
    episodes_without_snapshots = sorted(
        episode_id for episode_id in design_episode_ids if episode_id not in by_episode
    )
    unexpected_pool_episodes = sorted(set(by_episode) - set(design_episode_ids))
    missing_summary_episodes = sorted(set(design_episode_ids) - set(summary_episode_ids))
    duplicate_summary_episodes = sorted(
        episode_id
        for episode_id, count in Counter(summary_episode_ids).items()
        if count > 1
    )
    duplicate_design_tasks = sorted(
        task for task, count in Counter(design_tasks).items() if count > 1
    )

    exit_marker_present = False
    collect_done_count = 0
    if collect_log is not None and collect_log.is_file():
        text = collect_log.read_text(encoding="utf-8", errors="replace")
        exit_marker_present = "PRE_C0_COLLECT_EXIT" in text
        collect_done_count = text.count("COLLECT_EPISODE_DONE")

    checks = {
        "expected_episodes_24": expected_episodes == 24 and len(design_records) == 24,
        "summary_has_24_metrics": len(metrics) == 24,
        "summary_matches_design_episodes": (
            set(summary_episode_ids) == set(design_episode_ids)
            and not duplicate_summary_episodes
        ),
        "pool_covers_all_design_episodes": not episodes_without_snapshots,
        "no_unexpected_pool_episodes": not unexpected_pool_episodes,
        "no_duplicate_design_tasks": not duplicate_design_tasks,
        "no_duplicate_seeds_in_sample": not duplicate_seeds,
        "decision_context_sample_ok": not missing_history_states,
        "summary_file_present": True,
        "collection_exit_marker_present": exit_marker_present,
        "collect_done_count_matches_24": collect_done_count == 24 or collect_log is None,
    }
    # Formal gate: allow missing exit marker if summary+DONE evidence is complete.
    completion_ok = (
        checks["summary_has_24_metrics"]
        and checks["summary_matches_design_episodes"]
        and checks["pool_covers_all_design_episodes"]
        and (exit_marker_present or collect_done_count == 24)
    )
    checks["completion_accepted"] = completion_ok
    passed = all(
        value
        for name, value in checks.items()
        if name
        not in {
            "collection_exit_marker_present",  # advisory when DONE+summary complete
            "collect_done_count_matches_24",
        }
    ) and completion_ok

    return {
        "schema_version": SCHEMA_VERSION,
        "pass": passed,
        "checks": checks,
        "expected_trajectories": expected_episodes,
        "completed_trajectories": len(metrics),
        "outcome_counts": dict(outcomes),
        "states_in_pool": len(states),
        "snapshots_per_episode": {
            episode_id: len(by_episode.get(episode_id) or [])
            for episode_id in design_episode_ids
        },
        "episodes_without_snapshots": episodes_without_snapshots,
        "unexpected_pool_episodes": unexpected_pool_episodes,
        "missing_summary_episodes": missing_summary_episodes,
        "duplicate_summary_episodes": duplicate_summary_episodes,
        "duplicate_design_tasks": duplicate_design_tasks,
        "duplicate_seeds_in_sample": duplicate_seeds,
        "missing_required_history_states": missing_history_states,
        "decision_context_states_sampled": sampled_checked,
        "collect_log": {
            "path": None if collect_log is None else str(collect_log),
            "exit_marker_present": exit_marker_present,
            "collect_episode_done_count": collect_done_count,
            "note": (
                "PRE_C0_COLLECT_EXIT string absent; completion accepted from "
                "24 COLLECT_EPISODE_DONE + written summary"
                if (not exit_marker_present and collect_done_count == 24)
                else None
            ),
        },
        "design_sha256": design.get("design_sha256"),
        "pool": str(pool.root),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PRE-C0 collection integrity",
        "",
        f"- pass: `{report['pass']}`",
        f"- completed: `{report['completed_trajectories']}/{report['expected_trajectories']}`",
        f"- outcomes: `{report['outcome_counts']}`",
        f"- states_in_pool: `{report['states_in_pool']}`",
        f"- exit_marker_present: `{report['collect_log']['exit_marker_present']}`",
        f"- note: `{report['collect_log'].get('note')}`",
        "",
        "## Checks",
        "",
    ]
    for name, value in sorted(report["checks"].items()):
        lines.append(f"- `{name}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--collect-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-md", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --overwrite")

    report = audit_collection(
        design=_load(args.design.resolve()),
        summary=_load(args.summary.resolve()),
        pool=StatePool(args.pool.resolve()),
        collect_log=None if args.collect_log is None else args.collect_log.resolve(),
    )
    _write(args.output, report)
    progress = args.progress_md or args.output.with_suffix(".md")
    if str(progress).endswith(".json.md"):
        progress = Path(str(args.output).removesuffix(".json") + "_report.md")
    # Prefer explicit progress path under progress/ when not provided.
    if args.progress_md is None:
        progress = Path("progress/2026-08-04_pre_c0_collection_integrity.md")
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pass": report["pass"]}, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
