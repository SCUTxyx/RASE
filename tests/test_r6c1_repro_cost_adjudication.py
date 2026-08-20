from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_replica(root: Path, state: str, replica: int, *, t8_success: bool,
                  t8_cost: int) -> None:
    directory = (root / "suite_spatial" / "pi05_libero" /
                 "natural_development_eval" / "seed_2" / f"rep{replica}")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if replica == 0 else f"__rep{replica}"
    path = directory / f"{state}__seed2{suffix}.json"
    rows = []
    for elapsed, success, cost in ((0, True, 50), (8, t8_success, t8_cost),
                                   (16, False, 271)):
        rows.append({
            "policy_id": "pi05_libero",
            "seed_index": 2,
            "state_key": state,
            "rollout_seed": 1234,
            "elapsed_source_steps": elapsed,
            "source_final_success": True,
            "persistent_success_if_enter_now": success,
            "persistent_teacher_steps_if_enter_now": cost,
        })
    path.write_text(json.dumps({
        "rollout_index": replica,
        "source_success": True,
        "source_steps": 100,
        "stop_reason": "success",
        "rows": rows,
    }))


def run_audit(tmp_path: Path) -> dict:
    script = (Path(__file__).resolve().parents[1] / "scripts" /
              "audit_r6c1b_repro.py")
    output = tmp_path / "audit.json"
    subprocess.run([
        sys.executable, str(script), "--input-root", str(tmp_path / "collect"),
        "--exclusions-output", str(output),
    ], check=True, capture_output=True, text=True)
    return json.loads(output.read_text())


def test_teacher_cost_only_variation_does_not_trigger_third_replica(tmp_path: Path):
    root = tmp_path / "collect"
    write_replica(root, "cost_only", 0, t8_success=True, t8_cost=97)
    write_replica(root, "cost_only", 1, t8_success=True, t8_cost=99)
    result = run_audit(tmp_path)
    assert result["status"] == "frozen"
    assert result["repro_summary"]["n_cost_variability"] == 1
    assert result["repro_summary"]["n_needs_third"] == 0
    assert result["needs_third"] == []


def test_boundary_success_variation_still_requires_third_replica(tmp_path: Path):
    root = tmp_path / "collect"
    write_replica(root, "label_flip", 0, t8_success=True, t8_cost=97)
    write_replica(root, "label_flip", 1, t8_success=False, t8_cost=271)
    result = run_audit(tmp_path)
    assert result["status"] == "incomplete_needs_third"
    assert result["repro_summary"]["n_needs_third"] == 1
    assert result["needs_third"][0]["label_disagrees"] is True
