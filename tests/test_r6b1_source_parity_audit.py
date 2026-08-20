"""Regression tests for the R6-B1 source-parity hard gate.

The gate must catch exactly the R6-B1.0 failure mode: a source trajectory that
preserves the rollout seed and final success but ends at a different number of
environment steps than the frozen R6-A reference (149 vs 116), plus nonfinite
saved features and states missing from the reference.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.audit_r6b1_source_parity import main as audit_main


STATE = "sp1_0660d272e7256c6b204caf666e94c875"
SUITE = "Spatial"


def write_atlas(tmp_path, *, expected_steps: int, expected_seed: int,
                expected_success: bool) -> None:
    root = tmp_path / "atlas"
    summary_dir = root / "pi0fast_libero" / "seed_0"
    summary_dir.mkdir(parents=True)
    summary = {
        "per_state": [{
            "state_key": STATE, "rollout_seed": expected_seed,
            "source_success": expected_success, "suite": SUITE,
            "task_id": "libero_spatial_000495",
            "result": {"env_steps": expected_steps, "success": expected_success},
        }],
    }
    (summary_dir / "summary.json").write_text(json.dumps(summary))
    atlas = {"atlas_root": str(root), "pairs": {"pi0fast_libero": {"seeds": {"0": {}}}}}
    (tmp_path / "atlas.json").write_text(json.dumps(atlas))


def write_collector(tmp_path, *, steps: int, seed: int, success: bool,
                    finite: bool = True) -> None:
    data_dir = tmp_path / "collector"
    data_dir.mkdir()
    npz = data_dir / f"{STATE}__seed0.npz"
    arrays = {"image": np.ones((1, 2, 3, 96, 96), np.uint8)}
    if not finite:
        arrays["proprio"] = np.array([[1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], np.float32)
    np.savez_compressed(npz, **arrays)
    rows = [{
        "state_key": STATE, "policy_id": "pi0fast_libero", "seed_index": 0,
        "rollout_seed": seed, "elapsed_source_steps": 0,
        "source_final_success": success, "source_total_steps": steps,
    }]
    metadata = {"rows": rows, "npz": str(npz), "source_success": success, "source_steps": steps}
    (data_dir / f"{STATE}__seed0.json").write_text(json.dumps(metadata))
    report = {"collector_sha256": "abc", "bookkeeping_mode": "full", "boundaries": [0]}
    (data_dir / "report.json").write_text(json.dumps(report))


def run_audit(tmp_path):
    output = tmp_path / "audit.json"
    rc = audit_main([
        "--atlas", str(tmp_path / "atlas.json"),
        "--input-root", str(tmp_path / "collector"),
        "--output", str(output),
    ])
    return rc, json.loads(output.read_text())


def test_gate_passes_when_exactly_reproducing_r6a(tmp_path):
    write_atlas(tmp_path, expected_steps=116, expected_seed=154127683, expected_success=True)
    write_collector(tmp_path, steps=116, seed=154127683, success=True)
    rc, result = run_audit(tmp_path)
    assert rc == 0
    assert result["status"] == "pass"
    assert result["parity_failures"] == []
    assert result["n_rows"] == 1


def test_gate_catches_env_steps_mismatch_149_vs_116(tmp_path):
    # The exact R6-B1.0 regression: seed and success match, env steps do not.
    write_atlas(tmp_path, expected_steps=116, expected_seed=154127683, expected_success=True)
    write_collector(tmp_path, steps=149, seed=154127683, success=True)
    rc, result = run_audit(tmp_path)
    assert rc == 1
    assert result["status"] == "fail"
    failure = result["parity_failures"][0]
    assert failure["observed"]["source_total_steps"] == 149
    assert failure["expected"]["source_total_steps"] == 116
    assert failure["observed"]["rollout_seed"] == failure["expected"]["rollout_seed"]
    assert failure["observed"]["source_final_success"] == failure["expected"]["source_final_success"]


def test_gate_catches_rollout_seed_mismatch(tmp_path):
    write_atlas(tmp_path, expected_steps=116, expected_seed=154127683, expected_success=True)
    write_collector(tmp_path, steps=116, seed=1, success=True)
    rc, result = run_audit(tmp_path)
    assert rc == 1
    assert result["status"] == "fail"


def test_gate_catches_success_mismatch(tmp_path):
    write_atlas(tmp_path, expected_steps=116, expected_seed=154127683, expected_success=True)
    write_collector(tmp_path, steps=116, seed=154127683, success=False)
    rc, result = run_audit(tmp_path)
    assert rc == 1
    assert result["status"] == "fail"


def test_gate_catches_nonfinite_features(tmp_path):
    write_atlas(tmp_path, expected_steps=116, expected_seed=154127683, expected_success=True)
    write_collector(tmp_path, steps=116, seed=154127683, success=True, finite=False)
    rc, result = run_audit(tmp_path)
    assert rc == 1
    assert result["status"] == "fail"
    assert result["nonfinite_files"]


def test_gate_fails_on_missing_reference_state(tmp_path):
    write_atlas(tmp_path, expected_steps=116, expected_seed=154127683, expected_success=True)
    write_collector(tmp_path, steps=116, seed=154127683, success=True)
    # Drop the state from the reference to simulate a state not in the atlas.
    summary_path = tmp_path / "atlas" / "pi0fast_libero" / "seed_0" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["per_state"] = []
    summary_path.write_text(json.dumps(summary))
    rc, result = run_audit(tmp_path)
    assert rc == 1
    assert result["status"] == "fail"
    assert len(result["missing_reference"]) == 1


def test_gate_fails_with_no_collector_rows(tmp_path):
    write_atlas(tmp_path, expected_steps=116, expected_seed=154127683, expected_success=True)
    data_dir = tmp_path / "collector"
    data_dir.mkdir()
    npz = data_dir / f"{STATE}__seed0.npz"
    np.savez_compressed(npz, image=np.ones((1, 2, 3, 96, 96), np.uint8))
    (data_dir / f"{STATE}__seed0.json").write_text(
        json.dumps({"rows": [], "npz": str(npz)}))
    (data_dir / "report.json").write_text(
        json.dumps({"collector_sha256": "abc", "bookkeeping_mode": "full"}))
    rc, result = run_audit(tmp_path)
    assert rc == 1
    assert result["status"] == "fail"
    assert "no collector boundary rows found" in result["reasons"]


def test_gate_finds_nested_pilot_layout(tmp_path):
    """The pilot runner writes suite_*/<policy>/seed_<k>/*__seed<k>.json; the
    audit must discover those recursively."""
    write_atlas(tmp_path, expected_steps=116, expected_seed=154127683, expected_success=True)
    data_dir = tmp_path / "collector" / "suite_spatial" / "pi0fast_libero" / "seed_0"
    data_dir.mkdir(parents=True)
    npz = data_dir / f"{STATE}__seed0.npz"
    np.savez_compressed(npz, image=np.ones((1, 2, 3, 96, 96), np.uint8))
    rows = [{
        "state_key": STATE, "policy_id": "pi0fast_libero", "seed_index": 0,
        "rollout_seed": 154127683, "elapsed_source_steps": 0,
        "source_final_success": True, "source_total_steps": 116,
    }]
    (data_dir / f"{STATE}__seed0.json").write_text(
        json.dumps({"rows": rows, "npz": str(npz)}))
    (data_dir / "report.json").write_text(
        json.dumps({"collector_sha256": "abc", "bookkeeping_mode": "full", "boundaries": [0]}))
    rc, result = run_audit(tmp_path)
    assert rc == 0
    assert result["status"] == "pass"
    assert result["n_rows"] == 1


def test_gate_rejects_non_full_bookkeeping_mode(tmp_path):
    write_atlas(tmp_path, expected_steps=116, expected_seed=154127683, expected_success=True)
    write_collector(tmp_path, steps=116, seed=154127683, success=True)
    report_path = tmp_path / "collector" / "report.json"
    report = json.loads(report_path.read_text())
    report["bookkeeping_mode"] = "obs_only"
    report_path.write_text(json.dumps(report))
    rc, result = run_audit(tmp_path)
    assert rc == 1
    assert result["status"] == "fail"
    assert any("non-full bookkeeping mode" in reason for reason in result["reasons"])
