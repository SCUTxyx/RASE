from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_cohort(root: Path, policy: str, *, alter_image: bool = False) -> tuple[Path, Path]:
    rows = 192
    dataset = root / f"{policy}.npz"
    image = np.zeros((rows, 2, 3, 8, 8), dtype=np.uint8)
    if alter_image:
        image[0, 0, 0, 0, 0] = 1
    task = np.asarray([f"task_{index // 4:02d}" for index in range(rows)])
    suite = np.asarray([["Spatial", "Object", "Goal", "Long"][index // 48]
                        for index in range(rows)])
    np.savez_compressed(
        dataset,
        image=image,
        proprio=np.zeros((rows, 8), dtype=np.float32),
        action_summary=np.full((rows, 20), 1 if policy == "a" else 2, dtype=np.float32),
        action_summary_single_step=np.zeros((rows, 20), dtype=np.float32),
        language_hash=np.zeros((rows, 256), dtype=np.float32),
        instruction=np.asarray([f"instruction {value}" for value in task]),
        source_failure=np.tile([0, 0, 1, 1], 48).astype(np.float32),
        source_success=np.tile([1, 1, 0, 0], 48).astype(np.float32),
        source_steps=np.full(rows, 100, dtype=np.int32),
        state_key=np.asarray([f"state_{index:03d}" for index in range(rows)]),
        task_id=task, suite=suite,
        perturb_dim=np.asarray(["clean"] * rows),
        init_state_id=np.tile(np.arange(4), 48).astype(np.int32),
        policy_id=np.asarray([policy] * rows),
    )
    report = dataset.with_suffix(".npz.report.json")
    report.write_text(json.dumps({
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "rows": rows, "tasks": 48, "policy_id": policy,
    }))
    return dataset, report


def test_multivla_builder_requires_and_preserves_same_state_alignment(tmp_path: Path) -> None:
    first, first_report = write_cohort(tmp_path, "a")
    second, second_report = write_cohort(tmp_path, "b")
    output = tmp_path / "merged.npz"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(repo / "scripts/build_r7c_multivla_source_dataset.py"),
        "--dataset", str(first), "--dataset-report", str(first_report),
        "--dataset", str(second), "--dataset-report", str(second_report),
        "--output", str(output),
    ], cwd=repo, check=True)
    with np.load(output, allow_pickle=False) as data:
        assert data["source_failure"].shape == (384,)
        assert set(data["policy_index"].tolist()) == {0, 1}
        assert np.all(data["action_summary"][:192] == 1)
        assert np.all(data["action_summary"][192:] == 2)
    report = json.loads(output.with_suffix(".npz.report.json").read_text())
    assert report["policies"] == ["a", "b"]
    assert all(all(checks.values()) for checks in report["same_state_alignment"].values())


def test_multivla_builder_rejects_observation_misalignment(tmp_path: Path) -> None:
    first, first_report = write_cohort(tmp_path, "a")
    second, second_report = write_cohort(tmp_path, "b", alter_image=True)
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run([
        sys.executable, str(repo / "scripts/build_r7c_multivla_source_dataset.py"),
        "--dataset", str(first), "--dataset-report", str(first_report),
        "--dataset", str(second), "--dataset-report", str(second_report),
        "--output", str(tmp_path / "bad.npz"),
    ], cwd=repo, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "alignment failed" in completed.stderr


def test_shared_calibration_multivla_oof_smoke(tmp_path: Path) -> None:
    first, first_report = write_cohort(tmp_path, "a")
    second, second_report = write_cohort(tmp_path, "b")
    merged = tmp_path / "merged.npz"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(repo / "scripts/build_r7c_multivla_source_dataset.py"),
        "--dataset", str(first), "--dataset-report", str(first_report),
        "--dataset", str(second), "--dataset-report", str(second_report),
        "--output", str(merged),
    ], cwd=repo, check=True)
    output = tmp_path / "shared.json"
    subprocess.run([
        sys.executable, str(repo / "scripts/train_r7c_multivla_source_risk.py"),
        "--dataset", str(merged),
        "--dataset-report", str(merged.with_suffix(".npz.report.json")),
        "--output", str(output), "--mode", "shared_calib", "--seed", "11",
        "--members", "1", "--epochs", "1", "--bootstrap-samples", "10",
        "--device", "cpu",
    ], cwd=repo, check=True)
    report = json.loads(output.read_text())
    assert report["mode"] == "shared_calib"
    assert set(report["metrics_by_policy"]) == {"a", "b"}
    predictions = np.load(output.with_suffix(".predictions.npz"))
    assert predictions["calibrated_oof_probability"].shape == (384,)
    assert np.isfinite(predictions["calibrated_oof_probability"]).all()


def test_lovo_adaptation_curve_smoke(tmp_path: Path) -> None:
    first, first_report = write_cohort(tmp_path, "a")
    second, second_report = write_cohort(tmp_path, "b")
    merged = tmp_path / "merged.npz"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(repo / "scripts/build_r7c_multivla_source_dataset.py"),
        "--dataset", str(first), "--dataset-report", str(first_report),
        "--dataset", str(second), "--dataset-report", str(second_report),
        "--output", str(merged),
    ], cwd=repo, check=True)
    output = tmp_path / "heldout_b.json"
    subprocess.run([
        sys.executable, str(repo / "scripts/train_r7c_lovo_adaptation.py"),
        "--dataset", str(merged),
        "--dataset-report", str(merged.with_suffix(".npz.report.json")),
        "--heldout-policy", "b", "--output", str(output), "--seed", "13",
        "--members", "1", "--epochs", "1", "--bootstrap-samples", "10",
        "--device", "cpu",
    ], cwd=repo, check=True)
    report = json.loads(output.read_text())
    assert set(report["curves"]["unlabeled"]) == {"0", "8", "16", "32"}
    assert report["heldout_policy"] == "b"
    predictions = np.load(output.with_suffix(".predictions.npz"))
    assert np.isfinite(predictions["unlabeled_32"]).all()
