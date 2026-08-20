from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.train_r7a_source_risk_probe import (
    average_precision,
    binary_auc,
    expected_calibration_error,
    task_folds,
)
from scripts.build_r7a_source_risk_dataset import initial_proposal_summary
from rase.risk.canonical_action import summary_from_chunk
from rase.risk.vla_action_adapters import create_vla_adapter


def test_binary_metrics_are_exact_for_perfect_ranking() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.float32)
    scores = np.asarray([0.1, 0.8, 0.2, 0.9], dtype=np.float32)
    assert binary_auc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0
    assert expected_calibration_error(labels, labels) == 0.0


def test_initial_proposal_summary_uses_first_ten_actions_only() -> None:
    trace = np.zeros((12, 7), dtype=np.float32)
    trace[:10, 0] = np.arange(10, dtype=np.float32)
    trace[10:, 0] = 1000.0
    expected = summary_from_chunk(
        create_vla_adapter("pi0fast_libero").to_canonical(trace[:10])
    ).numpy()
    actual = initial_proposal_summary(
        trace, policy_id="pi0fast_libero", chunk_steps=10
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert actual[0] < 10.0  # the post-replan commands were not admitted


def test_initial_proposal_summary_rejects_truncated_queue() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        initial_proposal_summary(
            np.zeros((9, 7), dtype=np.float32),
            policy_id="pi0fast_libero", chunk_steps=10,
        )


def test_task_folds_keep_all_four_rows_together_and_cover_tasks() -> None:
    tasks = np.asarray([f"task_{i:02d}" for i in range(12) for _ in range(4)])
    suites = np.asarray([f"suite_{i // 3}" for i in range(12) for _ in range(4)])
    folds = task_folds(tasks, suites, count=5, seed=17)
    assert set().union(*folds) == set(tasks.tolist())
    assert sum(len(fold) for fold in folds) == 12
    assert all(not (folds[i] & folds[j]) for i in range(5) for j in range(i))


def test_full_five_fold_probe_smoke(tmp_path: Path) -> None:
    rows = 192
    repeats = np.tile(np.arange(4), 48)
    task_number = np.repeat(np.arange(48), 4)
    task_id = np.asarray([f"task_{value:02d}" for value in task_number])
    suite = np.asarray([f"suite_{value // 12}" for value in task_number])
    labels = (repeats >= 2).astype(np.float32)
    rng = np.random.default_rng(7)
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        image=rng.integers(0, 256, size=(rows, 2, 3, 16, 16), dtype=np.uint8),
        proprio=rng.normal(size=(rows, 8)).astype(np.float32),
        action_summary=rng.normal(size=(rows, 20)).astype(np.float32),
        language_hash=rng.normal(size=(rows, 256)).astype(np.float32),
        source_failure=labels,
        source_success=1.0 - labels,
        state_key=np.asarray([f"state_{i:03d}" for i in range(rows)]),
        task_id=task_id,
        suite=suite,
        policy_id=np.asarray(["pi0fast_libero"] * rows),
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "status": "PASS", "states": 192, "tasks": 48,
        "gate": {"complete": True, "support": True},
    }))
    repeat = tmp_path / "repeat.json"
    repeat.write_text(json.dumps({
        "status": "PASS", "audited_records": 16,
        "label_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    }))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "label_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
        "exact_repeat_audit_sha256": hashlib.sha256(repeat.read_bytes()).hexdigest(),
        "policy_id": "pi0fast_libero",
        "rows": 192, "tasks": 48,
    }))
    output = tmp_path / "probe.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "scripts/train_r7a_source_risk_probe.py"),
        "--dataset", str(dataset), "--dataset-report", str(report),
        "--label-audit", str(audit), "--exact-repeat-audit", str(repeat),
        "--output", str(output),
        "--seed", "3", "--members", "1", "--epochs", "1",
        "--bootstrap-samples", "10", "--device", "cpu",
    ], cwd=root, check=True)
    result = json.loads(output.read_text())
    assert len(result["fold_reports"]) == 5
    assert result["target"] == "source final failure only"
    prediction = np.load(output.with_suffix(".predictions.npz"))
    assert prediction["calibrated_oof_probability"].shape == (192,)
    assert np.isfinite(prediction["calibrated_oof_probability"]).all()
