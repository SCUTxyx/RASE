#!/usr/bin/env python3
"""Create a provenance-gated manifest for an official LIBERO-PRO layout.

This is a preflight tool, not an evaluator.  It prevents a copied or
home-grown BDDL/init-state directory from silently being reported as an
official LIBERO-PRO position-perturbation result.  A run is ``ready`` only
when the selected suite assets, their hashes, and a source provenance record
are all present.

Example
-------
python scripts/audit_libero_pro_assets.py \
  --pro-root /path/to/selected/layout/root \
  --suite libero_object --variant position_x0.3 \
  --provenance /path/to/provenance.json \
  --output runs/provenance/object_position_x0p3.json

The provenance JSON must contain ``upstream``, ``commit``, ``variant`` and
``source_assets``.  ``source_assets`` identifies the original official
BDDL/init-state locations before they were copied into the runtime root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "rase-libero-pro-provenance/v1"
EXPECTED_UPSTREAM = "https://github.com/Zxy-MLlab/LIBERO-PRO"
REQUIRED_PROVENANCE = ("upstream", "commit", "variant", "source_assets")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def inventory(directory: Path, *, suffix: str | None = None) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    if suffix is not None:
        paths = [path for path in paths if path.suffix == suffix]
    return [
        {
            "path": str(path.relative_to(directory)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def git_metadata(path: Path) -> dict[str, Any]:
    """Return local git facts only; an absent checkout is intentionally not OK."""
    if not (path / ".git").exists():
        return {"present": False}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        remotes = subprocess.check_output(
            ["git", "-C", str(path), "remote", "-v"], text=True
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"present": True, "read_error": str(exc)}
    return {"present": True, "commit": commit, "remotes": remotes}


def read_provenance(path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, ["missing --provenance record"]
    if not path.is_file():
        return None, [f"provenance file missing: {path}"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid provenance JSON: {exc}"]
    if not isinstance(value, dict):
        return None, ["provenance must be a JSON object"]
    missing = [key for key in REQUIRED_PROVENANCE if not value.get(key)]
    return value, [f"provenance missing required field: {key}" for key in missing]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pro-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument(
        "--variant", required=True,
        help="Pre-registered official layout label, e.g. position_x0.3.",
    )
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--expected-upstream", default=EXPECTED_UPSTREAM)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.pro_root.resolve()
    bddl_dir = root / "bddl_files" / args.suite
    init_dir = root / "init_states" / args.suite
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"missing pro root: {root}")
    if not bddl_dir.is_dir():
        errors.append(f"missing BDDL suite directory: {bddl_dir}")
    if not init_dir.is_dir():
        errors.append(f"missing init-state suite directory: {init_dir}")

    bddl = inventory(bddl_dir, suffix=".bddl")
    init = inventory(init_dir)
    if len(bddl) != 10:
        errors.append(f"expected exactly 10 Object-suite BDDL files, found {len(bddl)}")
    if len(init) < 10:
        errors.append(f"expected at least 10 Object-suite init-state files, found {len(init)}")

    provenance, provenance_errors = read_provenance(args.provenance)
    errors.extend(provenance_errors)
    if provenance is not None:
        if str(provenance.get("variant")) != str(args.variant):
            errors.append(
                "provenance variant does not equal requested variant: "
                f"{provenance.get('variant')!r} != {args.variant!r}"
            )
        upstream = str(provenance.get("upstream", "")).rstrip("/")
        expected = str(args.expected_upstream).rstrip("/")
        if upstream != expected:
            errors.append(f"unexpected upstream: {upstream!r}; expected {expected!r}")

    repo = args.official_repo.resolve() if args.official_repo else root
    repo_git = git_metadata(repo)
    if not repo_git.get("present"):
        errors.append(f"official repository checkout is unavailable: {repo}")
    elif repo_git.get("read_error"):
        errors.append(f"cannot read official repository metadata: {repo_git['read_error']}")

    report = {
        "schema_version": SCHEMA,
        "status": "ready" if not errors else "not_ready",
        "requested": {
            "suite": args.suite,
            "variant": args.variant,
            "expected_upstream": args.expected_upstream,
        },
        "runtime_root": str(root),
        "paths": {"bddl": str(bddl_dir), "init_states": str(init_dir)},
        "counts": {"bddl": len(bddl), "init_state_files": len(init)},
        "bddl_inventory": bddl,
        "init_state_inventory": init,
        "provenance": provenance,
        "official_repo": {"path": str(repo), "git": repo_git},
        "errors": errors,
        "official_claim_permitted": not errors,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "official_claim_permitted": report["official_claim_permitted"],
        "bddl": len(bddl),
        "init_state_files": len(init),
        "errors": errors,
    }, indent=2), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
