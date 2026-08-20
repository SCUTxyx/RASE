from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rase.collect.r7_schedule import build_design, load_design, requests_from_design


def base_design() -> dict:
    records = []
    prefixes = {
        "Spatial": "libero_spatial", "Object": "libero_object",
        "Goal": "libero_goal", "Long": "libero_10",
    }
    index = 0
    for suite in prefixes:
        for task in range(1, 13):
            dimension, level = (("clean", 0) if task <= 4 else
                                ("camera", 1) if task <= 8 else ("robot", 1))
            records.append({
                "request_index": index, "episode_id": f"base-{index}",
                "suite": suite, "task_id": f"{prefixes[suite]}_{task:06d}",
                "dimension": dimension, "level": level,
            })
            index += 1
    return {"records": records}


def test_r7_schedule_has_independent_init_states_and_task_clusters(tmp_path: Path) -> None:
    design = build_design(base_design(), repeats_per_task=4, seed=17)
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design))
    loaded = load_design(path, expected_sha256=design["design_sha256"])
    requests = requests_from_design(loaded, seed=17)

    assert len(requests) == 192
    assert len({request.episode_id for request in requests}) == 192
    by_task = Counter((request.suite, request.task_id) for request in requests)
    assert set(by_task.values()) == {4}
    init_by_task: dict[tuple[str, int], set[int]] = {}
    for request in requests:
        init_by_task.setdefault((request.suite, request.task_id), set()).add(
            int(request.init_state_id)
        )
    assert all(values == {0, 1, 2, 3} for values in init_by_task.values())
    assert len({request.seed for request in requests}) == 192
