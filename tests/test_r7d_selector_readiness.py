from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_readiness_unlocks_opportunity_audit_not_selector(tmp_path: Path) -> None:
    per_paths = []
    for policy in ("pi0fast_libero", "smolvla_libero"):
        path = tmp_path / f"{policy}.json"
        path.write_text(json.dumps({
            "policy_id": policy, "status": "PASS", "decision": "FULL_PASS",
        }))
        per_paths.append(path)
    shared = tmp_path / "shared.json"
    shared.write_text(json.dumps({
        "status": "PASS", "policies": ["pi0fast_libero", "smolvla_libero"],
    }))
    output = tmp_path / "readiness.json"
    repo = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, str(repo / "scripts/audit_r7d_selector_readiness.py"),
    ]
    for path in per_paths:
        command += ["--per-vla-stability", str(path)]
    command += ["--shared-stability", str(shared), "--output", str(output)]
    subprocess.run(command, cwd=repo, check=True)
    report = json.loads(output.read_text())
    assert report["status"] == "READY_FOR_MODEL_FREE_OPPORTUNITY_AUDIT"
    assert report["candidate_pairs"] == ["pi0fast_libero+openvla_oft"]
    assert report["risk_only_policies"] == ["smolvla_libero"]
    assert any("selector training" in value for value in report["still_locked_even_when_ready"])
