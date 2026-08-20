from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_r6c1_replica_aggregated_dataset.py"
    spec = importlib.util.spec_from_file_location("r6c1_replica_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_replica(root: Path, replica: int, successes: tuple[bool, bool, bool],
                  costs: tuple[int, int, int]) -> None:
    directory = (root / "suite_spatial" / "pi0fast_libero" /
                 "natural_development_eval" / "seed_1" / f"rep{replica}")
    directory.mkdir(parents=True)
    suffix = "" if replica == 0 else f"__rep{replica}"
    npz = directory / f"state__seed1{suffix}.npz"
    np.savez_compressed(
        npz,
        image=np.zeros((3, 2, 3, 8, 8), dtype=np.uint8),
        proprio=np.zeros((3, 8), dtype=np.float32),
        source_action_summary=np.zeros((3, 20), dtype=np.float32),
        source_action_trace=np.zeros((16, 7), dtype=np.float32),
    )
    rows = []
    for elapsed, success, cost in zip((0, 8, 16), successes, costs):
        rows.append({
            "policy_id": "pi0fast_libero", "seed_index": 1,
            "state_key": "state", "task_id": "libero_spatial_000001",
            "suite": "Spatial", "instruction": "pick up object",
            "group_id": "state:pi0fast_libero:seed1",
            "elapsed_source_steps": elapsed, "rollout_seed": 123,
            "source_final_success": False, "source_total_steps": 100,
            "source_success_within_8": False,
            "source_success_within_16": False,
            "source_success_within_32": False,
            "persistent_success_if_enter_now": success,
            "persistent_teacher_steps_if_enter_now": cost,
        })
    (directory / f"state__seed1{suffix}.json").write_text(json.dumps({
        "rows": rows, "source_success": False, "source_steps": 100,
        "rollout_index": replica, "npz": str(npz),
    }))


def test_replicas_become_counts_not_duplicate_rows(tmp_path: Path) -> None:
    write_replica(tmp_path, 0, (True, False, True), (10, 20, 30))
    write_replica(tmp_path, 1, (False, False, True), (30, 40, 50))
    write_replica(tmp_path, 2, (True, True, True), (20, 60, 70))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema_version": "rase-r6b1-dynamic-boundary-protocol/v1",
        "qualified_source_policies": [{
            "policy_id": "pi0fast_libero", "dynamic_seed_indices": [1],
        }],
    }))
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"status": "frozen", "excluded": []}))
    output = tmp_path / "dataset.npz"

    builder = load_builder()
    report = builder.build_dataset(
        input_roots=[tmp_path], protocol=protocol, output=output,
        exclusions=exclusions,
    )
    data = np.load(output)

    assert report["n_groups"] == 1
    assert report["n_rows"] == 3
    assert np.all(data["arm_trials"][:, 1] == 3)
    assert np.array_equal(data["arm_successes"][:, 1], np.asarray([2, 1, 3]))
    assert np.allclose(data["arm_success"][:, 1], np.asarray([2 / 3, 1 / 3, 1]))
    assert np.allclose(data["arm_teacher_step_quantiles"][0, 1], [12, 20, 28])
    assert len(set(data["group_id"].tolist())) == 1
