#!/usr/bin/env python3
"""Capture immutable PRE-C0-R4 code, data, model, and runtime provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle(path: Path, patterns: tuple[str, ...]) -> dict:
    files = sorted({item for pattern in patterns for item in path.rglob(pattern)})
    digest = hashlib.sha256()
    records = []
    for item in files:
        relative = item.relative_to(path).as_posix()
        file_sha = _sha256(item)
        size = item.stat().st_size
        digest.update(relative.encode())
        digest.update(file_sha.encode())
        records.append({"path": relative, "bytes": size, "sha256": file_sha})
    return {
        "path": str(path.resolve()),
        "n_files": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "bundle_sha256": digest.hexdigest(),
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--student-policy", type=Path, required=True)
    parser.add_argument(
        "--artifact", type=Path, action="append", default=[],
        help="Additional immutable dataset/report artifact to hash (repeatable)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conda", default="/root/miniconda3/bin/conda")
    args = parser.parse_args()
    root = args.root.resolve()
    sources = [
        root / "scripts/audit_pre_a3_operator_opportunity.py",
        root / "scripts/collect_r4_boundary_transitions.py",
        root / "scripts/run_pre_c0_r4_collect.sh",
        root / "scripts/train_r4_safe_handback_world_model.py",
        root / "scripts/run_pre_c0_r4_end_to_end.sh",
        root / "scripts/analyze_r4_live_boundary_opportunity.py",
        root / "scripts/rebuild_r4_merged_report.py",
        root / "scripts/probe_r4_dynamics_baselines.py",
        root / "scripts/evaluate_r4_policy_sweep.py",
    ]
    packages = {}
    for env in ("smolvla", "oft"):
        raw = _run([args.conda, "list", "-n", env, "--json"])
        try:
            packages[env] = json.loads(raw)
        except json.JSONDecodeError:
            packages[env] = {"error": raw}
    external = Path("/root/autodl-tmp/src/openvla-oft")
    report = {
        "schema_version": "rase-pre-c0-r4-provenance/v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "rase_git_head": _run(["git", "rev-parse", "HEAD"], root),
        "rase_git_status_sha256": hashlib.sha256(
            _run(["git", "status", "--porcelain=v1"], root).encode()
        ).hexdigest(),
        "openvla_oft_git_head": _run(["git", "rev-parse", "HEAD"], external),
        "openvla_oft_git_status": _run(["git", "status", "--porcelain=v1"], external),
        "inputs": {
            str(path.resolve()): _sha256(path.resolve())
            for path in (args.config, args.state_keys, args.audit, *args.artifact)
        },
        "source_sha256": {
            str(path.relative_to(root)): _sha256(path)
            for path in sources
        },
        "student_policy": _bundle(
            args.student_policy.resolve(),
            ("*.safetensors", "*.bin", "*.pt", "*.json", "*.model"),
        ),
        "conda_packages": packages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "rase_git_head": report["rase_git_head"],
        "openvla_oft_git_head": report["openvla_oft_git_head"],
        "student_policy_bundle_sha256": report["student_policy"]["bundle_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
