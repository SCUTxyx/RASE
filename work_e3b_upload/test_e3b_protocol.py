from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e3b_partition_is_task_disjoint_and_balanced() -> None:
    module = load_script("freeze_e3b_partitions")
    records = []
    request = 0
    for suite in ("Goal", "Long", "Object", "Spatial"):
        for task_index in range(6):
            task = f"{suite}-task-{task_index}"
            for dimension, level in (("clean", 0), ("camera", 1), ("robot", 1)):
                records.append(
                    {
                        "split": "train",
                        "suite": suite,
                        "logical_task_id": task,
                        "dimension": dimension,
                        "level": level,
                        "request_index": request,
                        "state_key": f"state-{request}",
                        "episode_id": f"episode-{request}",
                    }
                )
                request += 1
    manifest, roles = module.build_artifacts(
        {"records": records, "design_sha256": "design", "pool": "pool"}
    )
    assert manifest["decision"] == "PASS"
    assert manifest["role_counts"] == {
        "b0_smoke": 12,
        "b1_collect": 36,
        "b2_qualification": 24,
    }
    task_sets = [
        {row["logical_task_id"] for row in value["records"]}
        for value in roles.values()
    ]
    assert task_sets[0].isdisjoint(task_sets[1])
    assert task_sets[0].isdisjoint(task_sets[2])
    assert task_sets[1].isdisjoint(task_sets[2])


def test_e3b_teacher_summary_rejects_outcome_drift(tmp_path, monkeypatch) -> None:
    module = load_script("summarize_e3b_teacher_calibration")
    for repeat in ("a", "b"):
        for suite in module.SUITES:
            rows = []
            for index in range(3):
                success = not (repeat == "b" and suite == "long" and index == 0)
                rows.append(
                    {
                        "state_key": f"{suite}-{index}",
                        "direct_oft_success": success,
                    }
                )
            path = tmp_path / repeat / suite
            path.mkdir(parents=True)
            (path / "summary.json").write_text(json.dumps({"per_state": rows}))
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--input-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    assert module.main() == 2
    artifact = json.loads(output.read_text())
    assert artifact["decision"] == "FAIL"
    assert artifact["outcome_drift"]["n_states"] == 1


def test_e3b_b0_correction_modes_and_chunk_shape() -> None:
    module = load_script("collect_e3b_b0_onpolicy")
    assert not module.should_correct("source_h8", 0)
    assert module.should_correct("one_shot_h8", 0)
    assert not module.should_correct("one_shot_h8", 1)
    assert module.should_correct("persistent_h8", 0)
    assert module.should_correct("persistent_h8", 9)

    class Policy:
        def __init__(self):
            self.index = 0

        def act(self, observation, *, task):
            del observation, task
            value = np.full(7, self.index, dtype=np.float32)
            self.index += 1
            return value

    chunk = module.collect_chunk(Policy(), {}, "task", 8)
    assert chunk.shape == (8, 7)
    assert np.array_equal(chunk[:, 0], np.arange(8, dtype=np.float32))


def test_e3b_chunk_residual_feature_contract_and_size() -> None:
    from rase.recovery import e3b_chunk_residual as module

    state = module.state_features(
        np.zeros(8, dtype=np.float32),
        np.zeros((8, 7), dtype=np.float32),
        np.zeros((8, 23), dtype=np.float32),
        np.zeros(64, dtype=np.float32),
    )
    vision = module.vision_features(
        np.zeros((24, 24, 3), dtype=np.uint8),
        np.zeros((24, 24, 3), dtype=np.uint8),
    )
    assert state.shape == (module.STATE_DIM,)
    assert vision.shape == (2 * 24 * 24 * 3,)
    network = module.make_network()
    assert sum(parameter.numel() for parameter in network.parameters()) < 1_000_000
