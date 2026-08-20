from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np


def load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_r6c_dynamic_dataset.py"
    spec = importlib.util.spec_from_file_location("r6c1_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_trajectory(root: Path, role: str, *, replicate: int, state: str) -> None:
    directory = root / "suite_spatial" / "pi05_libero" / role / "seed_2" / f"rep{replicate}"
    directory.mkdir(parents=True)
    suffix = "" if replicate == 0 else f"__rep{replicate}"
    npz = directory / f"{state}__seed2{suffix}.npz"
    np.savez_compressed(
        npz,
        image=np.zeros((3, 2, 3, 8, 8), dtype=np.uint8),
        proprio=np.zeros((3, 8), dtype=np.float32),
        source_action_summary=np.zeros((3, 20), dtype=np.float32),
        source_action_trace=np.zeros((16, 7), dtype=np.float32),
    )
    group = f"{state}:pi05_libero:2" + (f":rep{replicate}" if replicate else "")
    rows = []
    for elapsed in (0, 8, 16):
        rows.append({
            "policy_id": "pi05_libero", "seed_index": 2,
            "state_key": state, "group_id": group,
            "task_id": "libero_spatial_000001", "suite": "libero_spatial",
            "elapsed_source_steps": elapsed, "instruction": "pick up object",
            "rollout_seed": 123, "source_final_success": False,
            "source_total_steps": 100, "source_success_within_8": False,
            "source_success_within_16": False, "source_success_within_32": False,
            "persistent_success_if_enter_now": True,
            "persistent_teacher_steps_if_enter_now": 40,
        })
    metadata = {
        "rows": rows, "source_success": False, "source_steps": 100,
        "rollout_index": replicate, "npz": str(npz),
    }
    (directory / f"{state}__seed2{suffix}.json").write_text(json.dumps(metadata))


def test_natural_replicas_are_training_only():
    builder = load_builder()
    with tempfile.TemporaryDirectory() as value:
        root = Path(value)
        protocol = root / "protocol.json"
        protocol.write_text(json.dumps({
            "schema_version": "rase-r6b1-dynamic-boundary-protocol/v1",
            "scientific_scope": "test",
            "qualified_source_policies": [{
                "policy_id": "pi05_libero", "dynamic_seed_indices": [2],
            }],
        }))
        write_trajectory(root, "natural_development_eval", replicate=0, state="natural")
        write_trajectory(root, "natural_development_eval", replicate=1, state="natural")
        write_trajectory(root, "train_enrichment", replicate=0, state="hard")
        output = root / "dataset.npz"
        _, report = builder.build_dataset(input_root=root, protocol=protocol, output=output)
        data = np.load(output)
        roles = {group: role for group, role in zip(data["group_id"], data["cohort_role"])}
        assert roles["natural:pi05_libero:2"] == "natural"
        assert roles["natural:pi05_libero:2:rep1"] == "replicate_training"
        assert roles["hard:pi05_libero:2"] == "enrichment"
        assert report["n_groups"] == 3
        assert report["n_base_groups"] == 2
