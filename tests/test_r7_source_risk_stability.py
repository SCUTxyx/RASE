from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.audit_r7a_source_risk_stability import SEEDS


def test_stability_requires_four_of_five(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    for index, seed in enumerate(SEEDS):
        path = root / f"seed_{seed}" / "report.json"
        path.parent.mkdir(parents=True)
        value = 0.8 + index * 0.001
        path.write_text(json.dumps({
            "seed": seed, "status": "PASS" if index < 4 else "FAIL",
            "policy_id": "pi0fast_libero",
            "dataset_sha256": "same",
            "metrics": {
                "auroc": value, "average_precision": value,
                "ap_above_prevalence": 0.2, "ece_10_equal_width": 0.08,
                "brier": 0.15,
            },
            "gate": {"point": index < 4},
            "task_bootstrap": {"auroc": {"mean": value, "lower_95": 0.7,
                                            "upper_95": 0.9}},
        }))
    output = tmp_path / "stability.json"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(repo / "scripts/audit_r7a_source_risk_stability.py"),
        "--input-root", str(root), "--output", str(output),
    ], cwd=repo, check=True)
    result = json.loads(output.read_text())
    assert result["status"] == "PASS"
    assert result["decision"] == "FULL_PASS"
    assert result["seeds_passed"] == 4
