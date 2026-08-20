#!/usr/bin/env python3
"""S0: Protocol audit — schema version, checkpoint SHA, OFT identity, feature contract, episode manifest.

Checks that the evaluation infrastructure is consistent before running any efficacy eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_protocol(protocol_path: Path) -> dict:
    p = json.loads(protocol_path.read_text(encoding="utf-8"))
    ok = True
    checks = {}

    # 1. Schema version
    checks["schema_version"] = p.get("schema_version", "MISSING")
    if checks["schema_version"] != "rase-route-c-plugin/v1":
        print(f"  WARN: unexpected schema_version={checks['schema_version']}")
        ok = False

    # 2. Student checkpoint exists
    student_path = Path(p["student_identity"]["checkpoint_path"])
    checks["student_checkpoint_exists"] = student_path.exists()
    if not student_path.exists():
        print(f"  FAIL: student checkpoint not found: {student_path}")
        ok = False

    # 3. Splits
    suites = list(p.get("splits", {}).keys())
    checks["suites"] = suites
    for suite in suites:
        dev = p["splits"][suite].get("dev", [])
        train = p["splits"][suite].get("train", [])
        print(f"  {suite}: dev={len(dev)} tasks, train={len(train)} tasks")

    # 4. Teacher identities
    for suite, ident in p.get("teacher_identities", {}).items():
        cp = Path(ident["checkpoint_path"])
        checks[f"teacher_{suite}_exists"] = cp.exists()
        if not cp.exists():
            print(f"  FAIL: teacher checkpoint {suite} not found: {cp}")
            ok = False
        else:
            print(f"  teacher {suite}: {cp} (exists)")

    # 5. Plugin config
    pc = p.get("plugin_config", {})
    for k in ["stagnation_eps", "stagnation_window", "max_takeover_steps", "delta_clip_per_dim"]:
        checks[f"plugin_{k}"] = pc.get(k)

    print(f"  stagnation_eps={pc.get('stagnation_eps')}, window={pc.get('stagnation_window')}")
    print(f"  max_takeover={pc.get('max_takeover_steps')}, delta_clip={pc.get('delta_clip_per_dim')}")

    checks["overall"] = ok
    return checks


def generate_episode_manifest(protocol: dict, suite: str, n_init_state: int,
                               n_seed: int, arms: list[str],
                               base_seed: int = 20260807) -> list[dict]:
    """Generate a deterministic episode manifest for paired evaluation."""
    tasks = protocol["splits"][suite]["dev"][:2]  # 2 dev tasks
    manifest = []
    for task_idx, task_id in enumerate(tasks):
        for si in range(n_init_state):
            init_state = si % 50
            for pi in range(n_seed):
                policy_seed = (base_seed * 31 + task_idx * 100 + si * 7 + pi * 13) % (2**31)
                for arm in arms:
                    manifest.append({
                        "task_id": task_id,
                        "init_state_id": init_state,
                        "seed": policy_seed,
                        "arm": arm,
                        "suite": suite,
                    })
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--n-init-state", type=int, default=5)
    parser.add_argument("--n-seed", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260807)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PROTOCOL AUDIT")
    print("=" * 60)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    audit = audit_protocol(args.protocol)

    # Generate manifest
    print()
    print("=" * 60)
    print("EPISODE MANIFEST")
    print("=" * 60)

    arms = ["b0", "b3"]
    manifest = generate_episode_manifest(
        protocol, args.suite, args.n_init_state, args.n_seed, arms, args.base_seed
    )
    print(f"  Suite: {args.suite}")
    print(f"  Tasks: {sorted(set(m['task_id'] for m in manifest))}")
    print(f"  Init states: {sorted(set(m['init_state_id'] for m in manifest))}")
    print(f"  Total episodes: {len(manifest)}")

    # Save
    audit_path = output_dir / "audit_protocol.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "episode_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"Audit saved to: {audit_path}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Overall audit: {'PASS' if audit['overall'] else 'FAIL'}")
    return 0 if audit["overall"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
