from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_rollout(path: Path, *, state_key: str, suite: str, success: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    npz = path.with_suffix(".npz")
    np.savez_compressed(
        npz,
        image=np.zeros((1, 2, 3, 96, 96), dtype=np.uint8),
        proprio=np.zeros((1, 8), dtype=np.float32),
        source_action=np.zeros((1, 7), dtype=np.float32),
        source_action_summary=np.zeros((1, 20), dtype=np.float32),
        source_action_trace=np.zeros((3, 7), dtype=np.float32),
        oft_action=np.zeros((0,), dtype=np.float32),
        oft_action_summary=np.zeros((0,), dtype=np.float32),
    )
    path.write_text(json.dumps({
        "source_success": success, "source_steps": 3,
        "stop_reason": "success" if success else "horizon",
        "npz": str(npz.resolve()), "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "rows": [{
            "state_key": state_key, "suite": suite, "task_id": f"task_{state_key}",
            "policy_id": "pi0fast_libero", "seed_index": 0, "rollout_seed": 17,
            "elapsed_source_steps": 0, "persistent_success_if_enter_now": None,
        }],
    }, sort_keys=True))


def test_hash_selected_exact_repeat_manifest_and_audit(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    repeat = tmp_path / "repeat"
    for suite in ("Spatial", "Object", "Goal", "Long"):
        for success in (False, True):
            for index in range(2):
                key = f"{suite.lower()}_{int(success)}_{index}"
                canonical_path = root / f"suite_{suite.lower()}" / "seed_0" / f"{key}__seed0.json"
                repeat_path = repeat / f"suite_{suite.lower()}" / "seed_0" / f"{key}__seed0__rep1.json"
                write_rollout(canonical_path, state_key=key, suite=suite, success=success)
                write_rollout(repeat_path, state_key=key, suite=suite, success=success)
    audit = tmp_path / "label.json"
    audit.write_text(json.dumps({"status": "PASS"}))
    manifest, output = tmp_path / "manifest.json", tmp_path / "repeat_audit.json"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(repo / "scripts/freeze_r7a_exact_repeat_manifest.py"),
        "--label-audit", str(audit), "--input-root", str(root), "--output", str(manifest),
    ], cwd=repo, check=True)
    subprocess.run([
        sys.executable, str(repo / "scripts/audit_r7a_exact_repeat.py"),
        "--manifest", str(manifest), "--repeat-root", str(repeat), "--output", str(output),
    ], cwd=repo, check=True)
    report = json.loads(output.read_text())
    assert report["status"] == "PASS"
    assert report["audited_records"] == 16


def test_exact_repeat_audit_rejects_action_trace_change() -> None:
    from scripts.audit_r7a_exact_repeat import array_difference

    assert array_difference(np.zeros((2, 7), np.float32), np.ones((2, 7), np.float32)) == 1.0


def test_amended_manifest_replaces_excluded_state_by_frozen_rank(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    repo = Path(__file__).resolve().parents[1]
    for suite in ("Spatial", "Object", "Goal", "Long"):
        for success in (False, True):
            for index in range(3):
                key = f"{suite.lower()}_{int(success)}_{index}"
                write_rollout(
                    root / f"suite_{suite.lower()}" / "seed_0" / f"{key}__seed0.json",
                    state_key=key, suite=suite, success=success,
                )
    original_audit = tmp_path / "label_original.json"
    original_audit.write_text(json.dumps({"status": "PASS"}))
    original_manifest = tmp_path / "manifest_original.json"
    subprocess.run([
        sys.executable, str(repo / "scripts/freeze_r7a_exact_repeat_manifest.py"),
        "--label-audit", str(original_audit), "--input-root", str(root),
        "--output", str(original_manifest),
    ], cwd=repo, check=True)
    original = json.loads(original_manifest.read_text())
    target = next(row for row in original["records"]
                  if row["suite"] == "Long" and not row["source_success"])
    selected = {row["state_key"] for row in original["records"]}
    same_group = [f"long_0_{index}" for index in range(3)]
    replacement = next(key for key in same_group if key not in selected)
    exclusion = tmp_path / "exclusion.json"
    exclusion.write_text(json.dumps({
        "status": "frozen", "excluded_state_keys": [target["state_key"]],
        "proposed_exact_repeat_replacement": {"state_key": replacement},
    }))
    amended_audit = tmp_path / "label_amended.json"
    amended_audit.write_text(json.dumps({
        "status": "PASS",
        "exclusion_manifest_sha256": hashlib.sha256(exclusion.read_bytes()).hexdigest(),
    }))
    amended_manifest = tmp_path / "manifest_amended.json"
    subprocess.run([
        sys.executable, str(repo / "scripts/freeze_r7a_exact_repeat_manifest.py"),
        "--label-audit", str(amended_audit), "--input-root", str(root),
        "--exclusion-manifest", str(exclusion), "--output", str(amended_manifest),
    ], cwd=repo, check=True)
    amended = json.loads(amended_manifest.read_text())
    amended_keys = {row["state_key"] for row in amended["records"]}
    assert target["state_key"] not in amended_keys
    assert replacement in amended_keys
    assert len(amended_keys) == 16
