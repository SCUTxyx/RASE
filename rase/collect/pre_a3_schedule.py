"""Design-driven episode schedule for PRE-A3 confirmatory collection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rase.collect.perturb_sampler import PerturbationRequest

PROTOCOL_VERSION = "rase-pre-a3-recovery120/v1"
DESIGN_VERSIONS = frozenset({"rase-pre-a3-design/v1"})
_SUBDIMENSIONS = {
    "clean": "none",
    "camera": "viewpoint",
    "robot": "initial_state",
}
_CONCRETE_RE = re.compile(
    r"^(libero_spatial|libero_object|libero_goal|libero_10)_(\d{6})$"
)


def parse_concrete_task_id(concrete_task_id: str) -> tuple[str, int]:
    match = _CONCRETE_RE.fullmatch(str(concrete_task_id))
    if match is None:
        raise ValueError(f"invalid concrete_task_id: {concrete_task_id!r}")
    return match.group(1), int(match.group(2))


def load_pre_a3_design(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_version") not in DESIGN_VERSIONS:
        raise ValueError(
            f"unsupported PRE-A3 design version: {payload.get('artifact_version')!r}"
        )
    n_requests = int(payload.get("n_requests", -1))
    if n_requests <= 0 or n_requests != len(payload.get("records") or []):
        raise ValueError("PRE-A3 design n_requests must match records length")
    digest = str(payload.get("design_sha256") or "")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("PRE-A3 design_sha256 mismatch")
    return payload


def requests_from_design(
    design: Mapping[str, Any],
    *,
    seed: int,
) -> list[PerturbationRequest]:
    records = list(design.get("records") or [])
    expected = int(design.get("n_requests", len(records)))
    if len(records) != expected:
        raise ValueError(f"expected {expected} design records, got {len(records)}")
    requests: list[PerturbationRequest] = []
    for row in sorted(records, key=lambda item: int(item["request_index"])):
        dimension = str(row["dimension"])
        if dimension not in _SUBDIMENSIONS:
            raise ValueError(f"unsupported design dimension: {dimension!r}")
        concrete = str(row.get("concrete_task_id") or "")
        _, numeric_id = parse_concrete_task_id(concrete)
        if dimension == "clean" and numeric_id not in range(1, 11):
            raise ValueError(
                f"clean concrete task must be 000001-000010, got {concrete}"
            )
        index = int(row["request_index"])
        request_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:pre-a3:{index}:{concrete}".encode()).digest()[:8],
            "big",
        ) & 0x7FFFFFFF
        requests.append(
            PerturbationRequest(
                index=index,
                suite=str(row["suite"]),
                dimension=dimension,
                subdimension=_SUBDIMENSIONS[dimension],
                level=int(row["level"]),
                seed=request_seed,
                global_episode_index=index,
                batch_id=1,
                task_id=numeric_id,
                init_state_id=None,
                episode_id=str(row["episode_id"]),
            )
        )
    if len({req.episode_id for req in requests}) != len(requests):
        raise ValueError("PRE-A3 design episode_ids must be unique")
    return requests


def amend_design_v1_to_v1_1(design_v1: Mapping[str, Any]) -> dict[str, Any]:
    """Fix clean concrete IDs from 0-based to 1-based before first collection."""
    payload = json.loads(json.dumps(design_v1))
    changed = 0
    for row in payload["records"]:
        if str(row["dimension"]) != "clean":
            continue
        suite_prefix, numeric = parse_concrete_task_id(str(row["concrete_task_id"]))
        if numeric in range(0, 10):
            row["concrete_task_id"] = f"{suite_prefix}_{numeric + 1:06d}"
            changed += 1
        elif numeric in range(1, 11):
            continue
        else:
            raise ValueError(f"unexpected clean concrete id {row['concrete_task_id']}")
    if changed not in {0, 40}:
        raise ValueError(f"expected 0 or 40 clean ID rewrites, got {changed}")
    payload["artifact_version"] = "rase-pre-a3-design/v1"
    payload["design_amendment"] = {
        "from": "rase_pre_a3_design120_v1",
        "to": "rase_pre_a3_design120_v1.1",
        "reason": (
            "Clean adapter requires official task_id in [1,10]; v1 used 0-based "
            "000000-000009. Amended before any confirmatory collection/outcomes."
        ),
        "clean_ids_rewritten": changed,
    }
    digest = hashlib.sha256(
        repr(
            sorted(
                (
                    r["task_id"],
                    r["dimension"],
                    r["level"],
                    r["split"],
                    r.get("concrete_task_id"),
                )
                for r in payload["records"]
            )
        ).encode()
    ).hexdigest()
    payload["design_sha256"] = digest
    payload["parent_design_sha256"] = design_v1.get("design_sha256")
    return payload
