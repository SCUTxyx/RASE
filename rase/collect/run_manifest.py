"""Immutable run provenance; reject resume when fingerprints change."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rase.collect.schema import canonical_json

MANIFEST_VERSION = "rase-run-manifest/v1"
MANIFEST_NAME = "run_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_sha(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def artifact_fingerprint(path: Path) -> str:
    """Stable fingerprint for a candidate ``.npz`` or directory tree."""
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    digests = []
    for file_path in sorted(p for p in target.rglob("*") if p.is_file()):
        rel = file_path.relative_to(target).as_posix()
        digests.append({"path": rel, "sha256": sha256_file(file_path)})
    return sha256_bytes(canonical_json(digests))


def build_run_manifest(
    *,
    repo_root: Path,
    resolved_config: Mapping[str, Any],
    pool_root: Path,
    candidates_dir: Path,
    policy_path: Path | None,
    policy_hash: str | None,
    protocol_version: str,
    oracle_model_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env_lock = repo_root / "env.lock.md"
    pool_manifest = Path(pool_root) / "manifest.json"
    return {
        "manifest_version": MANIFEST_VERSION,
        "git_sha": git_sha(repo_root),
        "env_lock_sha256": sha256_file(env_lock) if env_lock.is_file() else None,
        "pool_root": str(Path(pool_root).resolve()),
        "pool_manifest_sha256": (
            sha256_file(pool_manifest) if pool_manifest.is_file() else None
        ),
        "candidates_dir": str(Path(candidates_dir).resolve()),
        "candidates_dir_sha256": artifact_fingerprint(Path(candidates_dir)),
        "policy_path": str(Path(policy_path).resolve()) if policy_path else None,
        "policy_hash": policy_hash,
        "protocol_version": protocol_version,
        "oracle_model_info": dict(oracle_model_info or {}),
        "resolved_config": dict(resolved_config),
    }


def _compare_keys() -> tuple[str, ...]:
    return (
        "manifest_version",
        "git_sha",
        "env_lock_sha256",
        "pool_root",
        "pool_manifest_sha256",
        "candidates_dir",
        "candidates_dir_sha256",
        "policy_path",
        "policy_hash",
        "protocol_version",
        "oracle_model_info",
        "resolved_config",
    )


def write_run_manifest(run_root: Path, manifest: Mapping[str, Any]) -> Path:
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / MANIFEST_NAME
    payload = dict(manifest)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in _compare_keys():
            if existing.get(key) != payload.get(key):
                raise ValueError(
                    f"run_manifest fingerprint mismatch on {key!r}; "
                    "use a new run root or delete the stale scheduler"
                )
        return path
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_run_manifest(run_root: Path) -> dict[str, Any] | None:
    path = Path(run_root) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
