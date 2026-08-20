#!/usr/bin/env python3
"""Mine deterministic T0-T4 deviation snapshots from a PRE-C0 StatePool."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rase.collect.state_pool import LoadedState, StatePool

SCHEMA_VERSION = "rase-pre-c0-deviation-keys/v1"
STAGES = (
    ("T0", "last_stable"),
    ("T1", "first_deviation"),
    ("T2", "sustained_deviation"),
    ("T3", "failure_in_progress"),
    ("T4", "terminal"),
)
_SUFFIX_KEYS = ("active_suffix", "action_suffix", "current_suffix", "remaining_actions")


@dataclass(frozen=True)
class MiningConfig:
    """Frozen thresholds used to turn component timelines into stage labels."""

    visual_delta_threshold: float = 0.06
    proprio_velocity_threshold: float = 0.20
    proprio_jerk_threshold: float = 0.20
    active_suffix_norm_threshold: float = 4.0
    no_progress_streak_threshold: int = 3
    progress_visual_delta: float = 0.01
    progress_proprio_velocity: float = 0.01
    deviation_score_threshold: float = 0.75
    failure_score_threshold: float = 1.25
    sustained_steps: int = 2
    min_signal_coverage: float = 0.80
    visual_weight: float = 0.30
    velocity_weight: float = 0.15
    jerk_weight: float = 0.20
    suffix_weight: float = 0.15
    no_progress_weight: float = 0.20

    def __post_init__(self) -> None:
        positive = (
            "visual_delta_threshold",
            "proprio_velocity_threshold",
            "proprio_jerk_threshold",
            "active_suffix_norm_threshold",
            "no_progress_streak_threshold",
            "deviation_score_threshold",
            "failure_score_threshold",
            "sustained_steps",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.min_signal_coverage <= 1.0:
            raise ValueError("min_signal_coverage must be in [0, 1]")
        if self.failure_score_threshold < self.deviation_score_threshold:
            raise ValueError("failure_score_threshold must be >= deviation_score_threshold")
        weights = (
            self.visual_weight,
            self.velocity_weight,
            self.jerk_weight,
            self.suffix_weight,
            self.no_progress_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("component weights must be non-negative with a positive sum")

    @classmethod
    def from_design(cls, design: Mapping[str, Any]) -> "MiningConfig":
        section: Mapping[str, Any] = {}
        for candidate in (
            design.get("deviation_mining"),
            design.get("mining"),
            (design.get("pre_c0") or {}).get("deviation_mining")
            if isinstance(design.get("pre_c0"), Mapping)
            else None,
        ):
            if isinstance(candidate, Mapping):
                section = candidate
                break
        thresholds = section.get("thresholds", {})
        weights = section.get("weights", {})
        merged = dict(section)
        if isinstance(thresholds, Mapping):
            merged.update(thresholds)
        aliases = {
            "visual_delta": "visual_delta_threshold",
            "proprio_velocity": "proprio_velocity_threshold",
            "proprio_jerk": "proprio_jerk_threshold",
            "active_suffix_norm": "active_suffix_norm_threshold",
            "no_progress_streak": "no_progress_streak_threshold",
            "deviation_score": "deviation_score_threshold",
            "failure_score": "failure_score_threshold",
        }
        for old, new in aliases.items():
            if old in merged and new not in merged:
                merged[new] = merged[old]
        if isinstance(weights, Mapping):
            for short, field in (
                ("visual", "visual_weight"),
                ("velocity", "velocity_weight"),
                ("jerk", "jerk_weight"),
                ("suffix", "suffix_weight"),
                ("no_progress", "no_progress_weight"),
            ):
                if short in weights:
                    merged[field] = weights[short]
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in merged.items() if key in allowed})


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _verified_design_sha256(design: Mapping[str, Any]) -> str:
    declared = design.get("design_sha256")
    if declared is None:
        return _canonical_sha256(design)
    unsigned = dict(design)
    unsigned.pop("design_sha256", None)
    actual = _canonical_sha256(unsigned)
    if str(declared) != actual:
        raise ValueError(f"PRE-C0 design checksum mismatch: {declared} != {actual}")
    return actual


def _decode_png(data: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        message = "PNG mining requires Pillow (install the project's video extra)"
        raise RuntimeError(message) from exc
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def decoded_frame_delta(previous: bytes | None, current: bytes | None) -> float | None:
    """Return normalized RGB mean absolute error for two encoded PNG frames."""
    if previous is None or current is None:
        return None
    first = _decode_png(previous)
    second = _decode_png(current)
    if first.shape != second.shape:
        return None
    return float(np.mean(np.abs(second - first), dtype=np.float64) / 255.0)


def _numeric_array(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in "biuf" or array.size == 0:
        return None
    result = np.asarray(array, dtype=np.float64).reshape(-1)
    return result if np.all(np.isfinite(result)) else None


def extract_active_suffix(controller_state: Mapping[str, Any]) -> np.ndarray | None:
    """Find a numeric active action suffix in a nested controller-state mapping."""
    queue: list[Mapping[str, Any]] = [controller_state]
    while queue:
        current = queue.pop(0)
        for key in _SUFFIX_KEYS:
            if key in current:
                suffix = _numeric_array(current[key])
                if suffix is not None:
                    return suffix
        queue.extend(value for value in current.values() if isinstance(value, Mapping))
    return None


def _safe_ratio(value: float | None, threshold: float) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return value / threshold


def compute_component_timeline(
    *,
    steps: Sequence[int],
    proprios: Sequence[np.ndarray | None],
    suffixes: Sequence[np.ndarray | None],
    agentview_deltas: Sequence[float | None],
    wrist_deltas: Sequence[float | None],
    config: MiningConfig,
) -> list[dict[str, Any]]:
    """Compute per-step deviation components; useful with synthetic unit timelines."""
    size = len(steps)
    if not all(
        len(values) == size
        for values in (proprios, suffixes, agentview_deltas, wrist_deltas)
    ):
        raise ValueError("all timeline inputs must have equal length")
    if any(right <= left for left, right in zip(steps, steps[1:], strict=False)):
        raise ValueError("snapshot steps must be strictly increasing within an episode")

    rows: list[dict[str, Any]] = []
    previous_velocity: np.ndarray | None = None
    no_progress_streak = 0
    for index in range(size):
        dt = float(steps[index] - steps[index - 1]) if index else 1.0
        velocity_vector: np.ndarray | None = None
        if index and proprios[index] is not None and proprios[index - 1] is not None:
            current = np.asarray(proprios[index], dtype=np.float64).reshape(-1)
            previous = np.asarray(proprios[index - 1], dtype=np.float64).reshape(-1)
            if current.shape == previous.shape and np.all(np.isfinite(current)):
                velocity_vector = (current - previous) / dt
        velocity = float(np.linalg.norm(velocity_vector)) if velocity_vector is not None else None
        jerk = None
        if velocity_vector is not None and previous_velocity is not None:
            if velocity_vector.shape == previous_velocity.shape:
                jerk = float(np.linalg.norm((velocity_vector - previous_velocity) / dt))
        previous_velocity = velocity_vector

        suffix = suffixes[index]
        suffix_norm = (
            float(np.linalg.norm(np.asarray(suffix, dtype=np.float64).reshape(-1)))
            if suffix is not None
            else None
        )
        agent = agentview_deltas[index]
        wrist = wrist_deltas[index]
        visual_values = [value for value in (agent, wrist) if value is not None]
        visual = max(visual_values) if visual_values else None
        has_motion_signal = visual is not None or velocity is not None
        made_progress = (
            (visual is not None and visual >= config.progress_visual_delta)
            or (velocity is not None and velocity >= config.progress_proprio_velocity)
        )
        if index == 0 or not has_motion_signal or made_progress:
            no_progress_streak = 0
        else:
            no_progress_streak += 1

        ratios = {
            "visual": _safe_ratio(visual, config.visual_delta_threshold),
            "velocity": _safe_ratio(velocity, config.proprio_velocity_threshold),
            "jerk": _safe_ratio(jerk, config.proprio_jerk_threshold),
            "suffix": _safe_ratio(suffix_norm, config.active_suffix_norm_threshold),
            "no_progress": no_progress_streak / config.no_progress_streak_threshold,
        }
        weights = {
            "visual": config.visual_weight,
            "velocity": config.velocity_weight,
            "jerk": config.jerk_weight,
            "suffix": config.suffix_weight,
            "no_progress": config.no_progress_weight,
        }
        score = sum(weights[name] * ratios[name] for name in weights) / sum(weights.values())
        available = {
            "agentview_frame_delta": agent is not None,
            "wrist_frame_delta": wrist is not None,
            "proprio_velocity": velocity is not None,
            "proprio_jerk": jerk is not None,
            "active_suffix_norm": suffix_norm is not None,
            "no_progress_streak": has_motion_signal,
        }
        rows.append(
            {
                "index": index,
                "step": int(steps[index]),
                "agentview_frame_delta": agent,
                "wrist_frame_delta": wrist,
                "visual_frame_delta": visual,
                "proprio_velocity": velocity,
                "proprio_jerk": jerk,
                "active_suffix_norm": suffix_norm,
                "no_progress_streak": no_progress_streak,
                "deviation_score": float(score),
                "is_deviation": score >= config.deviation_score_threshold,
                "is_failure_in_progress": score >= config.failure_score_threshold,
                "available": available,
            }
        )
    return rows


def _temporal_fallback_indices(size: int) -> list[int]:
    if size < len(STAGES):
        raise ValueError(f"at least {len(STAGES)} snapshots are required, got {size}")
    return [(stage * (size - 1)) // (len(STAGES) - 1) for stage in range(len(STAGES))]


def _force_strict_unique(indices: list[int], size: int) -> list[int]:
    """Push colliding indices forward while keeping the terminal fixed."""
    fixed = [int(index) for index in indices]
    fixed[0] = max(0, min(fixed[0], size - 5))
    for position in range(1, len(fixed) - 1):
        lower = fixed[position - 1] + 1
        upper = size - (len(fixed) - position)
        fixed[position] = min(max(fixed[position], lower), upper)
    fixed[-1] = size - 1
    if len(set(fixed)) != len(fixed) or any(
        right <= left for left, right in zip(fixed, fixed[1:], strict=False)
    ):
        return _temporal_fallback_indices(size)
    return fixed


def label_stage_indices(
    timeline: Sequence[Mapping[str, Any]], config: MiningConfig
) -> tuple[dict[str, int], bool, list[str]]:
    """Retrospectively label strict unique stages.

    Full temporal percentile fallback is used only when first deviation (T1)
    cannot be detected. Missing T2/T3 after a valid T1 are soft-filled while
    preserving strict monotonicity; those soft fills are recorded but do not
    count as full temporal fallback for reliability gating.
    """
    size = len(timeline)
    if size < len(STAGES):
        raise ValueError(f"at least {len(STAGES)} snapshots are required, got {size}")
    reasons: list[str] = []
    soft_fill: list[str] = []

    deviation = [bool(row["is_deviation"]) for row in timeline]
    scores = [float(row["deviation_score"]) for row in timeline]
    t1_candidates = [index for index in range(1, size - 3) if deviation[index]]
    used_relative_t1 = False
    if not t1_candidates:
        window_scores = scores[1 : size - 3]
        peak = max(window_scores) if window_scores else 0.0
        relative_threshold = max(0.40, 0.85 * peak)
        t1_candidates = [
            index
            for index in range(1, size - 3)
            if scores[index] >= relative_threshold and scores[index] > 0.0
        ]
        used_relative_t1 = bool(t1_candidates)
    if not t1_candidates:
        reasons.append("first_deviation_not_detected_in_valid_window")
        indices = _temporal_fallback_indices(size)
        used_full_fallback = True
    else:
        if used_relative_t1:
            soft_fill.append("T1_relative")
        t1 = t1_candidates[0]
        stable = [index for index in range(t1) if not deviation[index]]
        t0 = stable[-1] if stable else 0
        if t0 >= t1:
            t0 = max(0, t1 - 1)
        t2 = next(
            (
                end
                for end in range(t1 + 1, size - 2)
                if end - config.sustained_steps + 1 >= t1
                and all(
                    deviation[start]
                    for start in range(
                        end - config.sustained_steps + 1,
                        end + 1,
                    )
                )
            ),
            -1,
        )
        if t2 < 0:
            soft_fill.append("T2")
            t2 = min(t1 + max(1, config.sustained_steps), size - 3)
            if t2 <= t1:
                t2 = t1 + 1
        t3 = next(
            (
                index
                for index in range(t2 + 1, size - 1)
                if bool(timeline[index]["is_failure_in_progress"])
            ),
            -1,
        )
        if t3 < 0:
            soft_fill.append("T3")
            # Prefer the strongest late score after T2; else penultimate frame.
            window = list(range(t2 + 1, size - 1))
            if window:
                t3 = max(
                    window,
                    key=lambda index: float(timeline[index]["deviation_score"]),
                )
            else:
                t3 = size - 2
        indices = _force_strict_unique([t0, t1, t2, t3, size - 1], size)
        if indices == _temporal_fallback_indices(size) and soft_fill:
            reasons.append("soft_fill_collapsed_to_temporal_fallback")
            used_full_fallback = True
        else:
            used_full_fallback = False
            reasons.extend(f"{stage}_soft_filled" for stage in soft_fill)

    if len(set(indices)) != len(STAGES) or any(
        right <= left for left, right in zip(indices, indices[1:], strict=False)
    ):
        raise AssertionError("internal error: stage indices are not strict and unique")
    return (
        {stage: index for (stage, _), index in zip(STAGES, indices, strict=True)},
        used_full_fallback,
        reasons,
    )


def _design_rows(design: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = design.get("records", [])
    if not isinstance(records, list):
        raise ValueError("design records must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping) or not row.get("episode_id"):
            raise ValueError("each design record must contain episode_id")
        episode_id = str(row["episode_id"])
        if episode_id in result:
            raise ValueError(f"duplicate design episode_id: {episode_id}")
        result[episode_id] = row
    return result


def _episode_reliability(
    timeline: Sequence[Mapping[str, Any]],
    temporal_fallback: bool,
    minimum: float,
    *,
    fallback_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    signal_names = tuple(timeline[0]["available"]) if timeline else ()
    # Deltas are undefined at the first snapshot and jerk at the first two.
    # Exclude those structural warm-up positions from coverage denominators.
    warmup = {
        "agentview_frame_delta": 1,
        "wrist_frame_delta": 1,
        "proprio_velocity": 1,
        "proprio_jerk": 2,
        "active_suffix_norm": 0,
        "no_progress_streak": 1,
    }
    coverage = {
        name: (
            sum(bool(row["available"][name]) for row in timeline)
            / max(1, len(timeline) - warmup.get(name, 0))
        )
        for name in signal_names
    }
    signal_coverage = float(sum(coverage.values()) / len(coverage)) if coverage else 0.0
    score = signal_coverage * (0.5 if temporal_fallback else 1.0)
    reasons = []
    # Full temporal fallback (missing T1) fails reliability. Soft-filled T2/T3
    # after a detected T1 remain eligible when signal coverage is adequate.
    if temporal_fallback:
        reasons.append("temporal_fallback")
    if signal_coverage < minimum:
        reasons.append("low_signal_coverage")
    return {
        "reliable": not reasons,
        "score": score,
        "signal_coverage": signal_coverage,
        "component_coverage": coverage,
        "reasons": reasons,
        "fallback_reasons": list(fallback_reasons or []),
    }


def mine_pool(
    pool: StatePool,
    design: Mapping[str, Any],
    *,
    config: MiningConfig | None = None,
) -> dict[str, Any]:
    """Mine selected state keys and QC summaries from an existing StatePool."""
    config = config or MiningConfig.from_design(design)
    design_sha256 = _verified_design_sha256(design)
    design_by_episode = _design_rows(design)
    grouped: dict[str, list[str]] = defaultdict(list)
    for state_key, entry in (pool.manifest().get("states") or {}).items():
        episode_id = str(entry["episode_id"])
        if design_by_episode and episode_id not in design_by_episode:
            continue
        grouped[episode_id].append(str(state_key))
    missing = sorted(set(design_by_episode) - set(grouped))
    if missing:
        raise ValueError(f"design episodes absent from pool: {missing[:5]}")
    if not grouped:
        raise ValueError("no StatePool snapshots matched the design")

    episodes = []
    flattened = []
    for episode_id in sorted(grouped):
        loaded = [pool.read_state(key) for key in grouped[episode_id]]
        loaded.sort(key=lambda state: state.metadata.step)
        identities = {
            (state.metadata.episode_id, state.metadata.task_id, state.metadata.suite)
            for state in loaded
        }
        if len(identities) != 1 or loaded[0].metadata.episode_id != episode_id:
            raise ValueError(f"inconsistent snapshot identity in episode {episode_id}")
        design_row = design_by_episode.get(episode_id, {})
        expected_identity = {
            "task_id": design_row.get("concrete_task_id", design_row.get("task_id")),
            "suite": design_row.get("suite"),
        }
        for field, expected in expected_identity.items():
            actual = getattr(loaded[0].metadata, field)
            if expected is not None and str(expected) != str(actual):
                raise ValueError(
                    f"design/pool {field} mismatch for {episode_id}: "
                    f"{expected!r} != {actual!r}"
                )
        steps = [state.metadata.step for state in loaded]
        agent_deltas = [None]
        wrist_deltas = [None]
        for previous, current in zip(loaded, loaded[1:], strict=False):
            agent_deltas.append(
                decoded_frame_delta(
                    previous.observations.get("agentview"),
                    current.observations.get("agentview"),
                )
            )
            wrist_deltas.append(
                decoded_frame_delta(
                    previous.observations.get("wrist"),
                    current.observations.get("wrist"),
                )
            )
        timeline = compute_component_timeline(
            steps=steps,
            proprios=[state.proprio for state in loaded],
            suffixes=[extract_active_suffix(state.controller_state) for state in loaded],
            agentview_deltas=agent_deltas,
            wrist_deltas=wrist_deltas,
            config=config,
        )
        for row, state in zip(timeline, loaded, strict=True):
            row["state_key"] = state.state_key
        indices, temporal_fallback, fallback_reasons = label_stage_indices(timeline, config)
        reliability = _episode_reliability(
            timeline,
            temporal_fallback,
            config.min_signal_coverage,
            fallback_reasons=fallback_reasons,
        )
        first = loaded[0].metadata
        stages = {}
        for stage, name in STAGES:
            index = indices[stage]
            state = loaded[index]
            stages[stage] = {
                "name": name,
                "state_key": state.state_key,
                "index": index,
                "step": state.metadata.step,
                "components": {
                    key: value
                    for key, value in timeline[index].items()
                    if key not in {"state_key", "index", "step"}
                },
            }
            flattened.append(
                {
                    "episode_id": episode_id,
                    "task_id": first.task_id,
                    "logical_task_id": design_row.get("task_id", first.task_id),
                    "concrete_task_id": design_row.get("concrete_task_id", first.task_id),
                    "suite": first.suite,
                    "stage": stage,
                    "stage_name": name,
                    "state_key": state.state_key,
                    "step": state.metadata.step,
                    "temporal_fallback": temporal_fallback,
                    "reliable": reliability["reliable"],
                }
            )
        episodes.append(
            {
                "episode_id": episode_id,
                "task_id": first.task_id,
                "logical_task_id": design_row.get("task_id", first.task_id),
                "concrete_task_id": design_row.get("concrete_task_id", first.task_id),
                "suite": first.suite,
                "cell": design_row.get(
                    "cell", f"{first.perturb_dim}:L{first.level}"
                ),
                "split": design_row.get("split"),
                "outcome": first.episode_outcome,
                "snapshot_count": len(loaded),
                "temporal_fallback": temporal_fallback,
                "temporal_fallback_reasons": fallback_reasons,
                "reliability": reliability,
                "stages": stages,
                "timeline": timeline,
            }
        )

    reliable = sum(bool(episode["reliability"]["reliable"]) for episode in episodes)
    fallback_count = sum(bool(episode["temporal_fallback"]) for episode in episodes)
    reason_counts = Counter(
        reason for episode in episodes for reason in episode["temporal_fallback_reasons"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stage_definitions": {stage: name for stage, name in STAGES},
        "n_episodes": len(episodes),
        "n_selected_states": len(flattened),
        "selected_states": flattened,
        "qc_summary": {
            "episodes_with_temporal_fallback": fallback_count,
            "temporal_fallback_rate": fallback_count / len(episodes),
            "fallback_reason_counts": dict(sorted(reason_counts.items())),
            "all_stage_orders_strict": True,
            "all_stage_keys_unique_per_episode": True,
        },
        "reliability_summary": {
            "reliable_episodes": reliable,
            "reliable_rate": reliable / len(episodes),
            "mean_reliability_score": float(
                np.mean([episode["reliability"]["score"] for episode in episodes])
            ),
            "mean_signal_coverage": float(
                np.mean(
                    [episode["reliability"]["signal_coverage"] for episode in episodes]
                )
            ),
        },
        "provenance": {
            "miner": "scripts/mine_deviation_snapshots.py",
            "miner_schema_version": SCHEMA_VERSION,
            "pool": str(pool.root.resolve()),
            "design_artifact_version": design.get("artifact_version"),
            "design_sha256": design_sha256,
            "pool_manifest_sha256": _canonical_sha256(pool.manifest()),
            "config": asdict(config),
            "selection_uses_terminal_outcome": False,
        },
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="StatePool root")
    parser.add_argument("--design", type=Path, required=True, help="frozen PRE-C0 design JSON")
    parser.add_argument("--output", type=Path, required=True, help="output keys JSON")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --overwrite")
    design = json.loads(args.design.read_text(encoding="utf-8"))
    result = mine_pool(StatePool(args.pool.resolve()), design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_episodes": result["n_episodes"],
                "n_selected_states": result["n_selected_states"],
                "fallback_rate": result["qc_summary"]["temporal_fallback_rate"],
                "reliable_rate": result["reliability_summary"]["reliable_rate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
