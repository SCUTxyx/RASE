#!/usr/bin/env python3
"""PRE-C1.4-R3 Phase 0: Identity manifest and split generation.

Generates `pre_c1_4_r3_identity_manifest.json` recording all frozen identities
(C1.1 adapter, teacher checkpoint, PRE-A3 audit, code SHAs, protocol hashes)
and the anchor split manifest (calibration / train / dev / confirmation).

The confirmation manifest is sealed and must not be read by training code.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git_sha(repo_root: Path) -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _sha256_path(p: Path) -> str:
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _all_available_failure_keys(
    c0_failure_dir: Path, limit: int = 200, max_scan: int = 500
) -> list[dict]:
    """Scan PRE-C0 failure rollout JSON files for candidate state keys.

    Returns list of {state_key, suite, task_id, episode_seed, ...}.
    Limits total files scanned to avoid hanging on large directories.
    """
    rows = []
    scanned = 0
    # Only scan top-level and one level deep to avoid massive directory trees
    for jf in sorted(c0_failure_dir.iterdir()):
        if scanned >= max_scan or (rows and len(rows) >= limit):
            break
        if jf.is_dir():
            for sub in sorted(jf.iterdir()):
                if scanned >= max_scan or (rows and len(rows) >= limit):
                    break
                if sub.suffix == ".json":
                    scanned += 1
                    rows = _extract_keys_from_json(sub, rows, limit)
                if scanned % 100 == 0:
                    print(f"    scanned {scanned} files, {len(rows)} keys...",
                          flush=True)
        elif jf.suffix == ".json":
            scanned += 1
            rows = _extract_keys_from_json(jf, rows, limit)

    # If not enough, also check known PRE-A3 dirs
    pre_a3_dirs = [
        c0_failure_dir.parent / "rase_pre_a3_confirmatory",
        c0_failure_dir.parent / "rase_pre_a3_confirmatory_smoke4_v1",
    ]
    for pd in pre_a3_dirs:
        if pd.exists() and len(rows) < limit:
            for jf in sorted(pd.glob("*.json")):
                if scanned >= max_scan or len(rows) >= limit:
                    break
                scanned += 1
                rows = _extract_keys_from_json(jf, rows, limit)

    return rows


def _extract_keys_from_json(jf: Path, rows: list, limit: int) -> list:
    """Extract state keys from a single JSON file."""
    if len(rows) >= limit:
        return rows
    try:
        data = json.loads(jf.read_text())
    except Exception:
        return rows
    if not isinstance(data, dict):
        return rows

    # Format 1: top-level state_key + suite (PRE-C0 rollout format)
    sk = data.get("state_key", "")
    suite = data.get("suite", "")
    if sk and suite:
        rows.append({
            "state_key": sk,
            "suite": suite,
            "task_id": data.get("task_id", data.get("concrete_task_id", "")),
            "episode_seed": data.get("episode_id", ""),
            "source_file": jf.name,
        })
        return rows

    # Format 2: nested "episodes" array
    for ep in data.get("episodes", []):
        if len(rows) >= limit:
            break
        si = ep.get("suite_info", {}) or {}
        suite_ep = si.get("suite_short", "")
        task_id = ep.get("task_id", "")
        init_seed = ep.get("episode_seed", ep.get("seed", ""))
        key = ep.get("state_key", "")
        if key and suite_ep:
            rows.append({
                "state_key": key,
                "suite": suite_ep,
                "task_id": task_id,
                "episode_seed": init_seed,
                "source_file": jf.name,
            })

    # Format 3: "state_keys" array with "by_suite"
    if len(rows) < limit:
        for sk_list in data.get("state_keys", []):
            if not isinstance(sk_list, str) or not sk_list:
                continue
            for suite_name, keys in data.get("by_suite", {}).items():
                if sk_list in keys:
                    rows.append({
                        "state_key": sk_list,
                        "suite": suite_name,
                        "task_id": "",
                        "episode_seed": "",
                        "source_file": jf.name,
                    })
                    break
            else:
                rows.append({
                    "state_key": sk_list,
                    "suite": "unknown",
                    "task_id": "",
                    "episode_seed": "",
                    "source_file": jf.name,
                })
            if len(rows) >= limit:
                break
    return rows


def _assign_splits(
    candidates: list[dict],
    calibration_n: int = 8,
    train_n: int = 16,
    dev_n: int = 8,
    confirm_n: int = 16,
):
    """Assign candidates to splits by suite-stratified shuffle."""
    import random

    if not candidates:
        return {
            "calibration": [],
            "train_collection": [],
            "development": [],
            "locked_confirmation": [],
        }

    rng = random.Random(20260806)
    candidates.sort(key=lambda r: (r["suite"], r["task_id"], r["episode_seed"]))
    rng.shuffle(candidates)

    by_suite = {}
    for c in candidates:
        by_suite.setdefault(c["suite"], []).append(c)

    splits = {
        "calibration": [],
        "train_collection": [],
        "development": [],
        "locked_confirmation": [],
    }

    all_candidates = list(candidates)
    rng.shuffle(all_candidates)

    targets = [
        ("calibration", calibration_n),
        ("train_collection", train_n),
        ("development", dev_n),
        ("locked_confirmation", confirm_n),
    ]

    idx = 0
    for split_name, n_needed in targets:
        for _ in range(min(n_needed, len(all_candidates) - idx)):
            if idx < len(all_candidates):
                splits[split_name].append(all_candidates[idx])
                idx += 1

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="PRE-C1.4-R3 identity and split manifest generation"
    )
    parser.add_argument(
        "--c0-failure-dir",
        default=str(ROOT / "runs" / "rase_pre_c0_same_policy_pilot48_v1"),
    )
    parser.add_argument(
        "--c11-adapter-dir",
        default=str(ROOT / "runs" / "rase_pre_c1_1_lora_train_v1" / "adapter_final"),
    )
    parser.add_argument(
        "--teacher-ckpt-dir",
        default=str(ROOT / "ckpts" / "oft_object"),
    )
    parser.add_argument(
        "--pre-a3-audit",
        default=str(
            ROOT / "runs" / "rase_pre_a3_confirmatory_smoke4_v1" / "summary.json"
        ),
    )
    parser.add_argument(
        "--calibration-n", type=int, default=8,
    )
    parser.add_argument(
        "--train-n", type=int, default=16,
    )
    parser.add_argument(
        "--dev-n", type=int, default=8,
    )
    parser.add_argument(
        "--confirm-n", type=int, default=16,
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "runs" / "rase_pre_c1_4_r3_protocol"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=20260806,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- collect identity SHAs ----
    identity = {
        "schema_version": "rase-pre-c1-4-r3-identity/v1",
        "c11_student_adapter_dir": str(Path(args.c11_adapter_dir).resolve()),
        "c11_adapter_sha256": _sha256_path(
            Path(args.c11_adapter_dir) / "adapter_config.json"
        ),
        "c11_lora_sha256": _sha256_path(
            Path(args.c11_adapter_dir) / "adapter_model.safetensors"
        ),
        "teacher_ckpt_dir": str(Path(args.teacher_ckpt_dir).resolve()),
        "pre_a3_audit_sha256": _sha256_path(Path(args.pre_a3_audit)),
        "code_git_sha": _git_sha(ROOT),
        "protocol_sha256": _sha256_text(
            json.dumps(
                {
                    "causal_unit_candidates": [4, 8, 64],
                    "screen_seeds": 3,
                    "verification_seeds": 5,
                    "branch_continuation_policy": "frozen_c1_1_student",
                    "calibration_n": args.calibration_n,
                    "train_n": args.train_n,
                    "dev_n": args.dev_n,
                    "confirm_n": args.confirm_n,
                },
                sort_keys=True,
            )
        ),
    }

    # ---- collect candidate state keys ----
    c0_failure_dir = Path(args.c0_failure_dir)
    candidates = _all_available_failure_keys(c0_failure_dir)

    total_needed = args.calibration_n + args.train_n + args.dev_n + args.confirm_n
    if len(candidates) < total_needed:
        print(
            f"WARNING: only {len(candidates)} candidate states found, "
            f"need at least {total_needed}. "
            f"Consider running with smaller split sizes or adding more failure data."
        )

    splits = _assign_splits(
        candidates,
        calibration_n=args.calibration_n,
        train_n=args.train_n,
        dev_n=args.dev_n,
        confirm_n=args.confirm_n,
    )

    # ---- build manifest ----
    manifest = {
        "schema_version": "rase-pre-c1-4-r3-identity-manifest/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "identity": identity,
        "splits": {
            "calibration": {
                "n": len(splits["calibration"]),
                "state_keys": [r["state_key"] for r in splits["calibration"]],
                "by_suite": {},
            },
            "train_collection": {
                "n": len(splits["train_collection"]),
                "state_keys": [r["state_key"] for r in splits["train_collection"]],
                "by_suite": {},
            },
            "development": {
                "n": len(splits["development"]),
                "state_keys": [r["state_key"] for r in splits["development"]],
                "by_suite": {},
            },
            "locked_confirmation": {
                "n": len(splits["locked_confirmation"]),
                "state_keys": [
                    r["state_key"] for r in splits["locked_confirmation"]
                ],
                "by_suite": {},
            },
        },
        "overlap_audit": {
            "calibration_vs_train": 0,
            "calibration_vs_dev": 0,
            "calibration_vs_confirm": 0,
            "train_vs_dev": 0,
            "train_vs_confirm": 0,
            "dev_vs_confirm": 0,
        },
    }

    # ---- by_suite grouping and overlap audit ----
    split_names = ["calibration", "train_collection", "development", "locked_confirmation"]
    for spam in split_names:
        for r in splits[spam]:
            manifest["splits"][spam]["by_suite"].setdefault(r["suite"], []).append(
                r["state_key"]
            )

    # Check overlap
    keys_by_split = {
        s: set(manifest["splits"][s]["state_keys"]) for s in split_names
    }
    pairs = [
        ("calibration", "train_collection"),
        ("calibration", "development"),
        ("calibration", "locked_confirmation"),
        ("train_collection", "development"),
        ("train_collection", "locked_confirmation"),
        ("development", "locked_confirmation"),
    ]
    for a, b in pairs:
        overlap = len(keys_by_split[a] & keys_by_split[b])
        key = f"{a}_vs_{b.split('_')[0]}"
        if b == "locked_confirmation":
            key = f"{a}_vs_confirm"
        elif b == "train_collection":
            key = f"{a}_vs_train"
        elif b == "development":
            key = f"{a}_vs_dev"
        manifest["overlap_audit"][f"{a}_vs_{b}"] = overlap

    # ---- generate sealed confirmation manifest (no outcomes) ----
    confirm_manifest = {
        "schema_version": "rase-pre-c1-4-r3-confirmation-protocol/v1",
        "sealed_at": manifest["generated_at"],
        "identity_sha256": _sha256_text(json.dumps(identity, sort_keys=True)),
        "locked_anchors": manifest["splits"]["locked_confirmation"]["state_keys"],
        "n_anchors": manifest["splits"]["locked_confirmation"]["n"],
        "protocol": {
            "training_seeds": 3,
            "eval_seeds_per_anchor": 5,
            "clean_episodes": 100,
            "bootstrap_hierarchy": ["training_seed", "anchor", "episode_seed"],
            "gate_requirements": {
                "training_seeds_positive": "2 of 3",
                "paired_ci_lower_bound": "> 0",
                "reproducible_recovery_anchors": ">= 6 of 16",
                "suite_coverage": ">= 3",
                "clean_retention_drop_max_pp": 2,
            },
        },
    }

    # ---- write outputs ----
    manifest_path = output_dir / "pre_c1_4_r3_identity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Identity manifest written to {manifest_path}")
    print(f"  calibration: {manifest['splits']['calibration']['n']} anchors")
    print(f"  train:       {manifest['splits']['train_collection']['n']} anchors")
    print(f"  dev:         {manifest['splits']['development']['n']} anchors")
    print(f"  confirm:     {manifest['splits']['locked_confirmation']['n']} anchors (SEALED)")

    # Seal confirmation manifest separately
    confirm_path = output_dir / "confirmation_protocol_frozen.json"
    confirm_path.write_text(
        json.dumps(confirm_manifest, indent=2, sort_keys=True) + "\n"
    )
    # Write SHA of sealed manifest
    confirm_sha = _sha256_path(confirm_path)
    seal_path = output_dir / "confirmation_protocol_frozen.json.sha256"
    seal_path.write_text(confirm_sha + "\n")
    print(f"Confirmation protocol sealed at {confirm_path}")
    print(f"  SHA256: {confirm_sha}")

    # Summarize overlap
    if any(v > 0 for v in manifest["overlap_audit"].values()):
        print("\n*** WARNING: ANCHOR OVERLAP DETECTED ***")
        for k, v in manifest["overlap_audit"].items():
            if v > 0:
                print(f"  {k}: {v} overlapping keys")
    else:
        print("\nAll splits are disjoint. No anchor overlap.")

    print("\nDone. Phase 0 identity manifest generated.")


if __name__ == "__main__":
    main()
