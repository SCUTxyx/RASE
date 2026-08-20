import json

from rase.collect.pre_a3_schedule import (
    amend_design_v1_to_v1_1,
    parse_concrete_task_id,
    requests_from_design,
)


def _minimal_design():
    rows = []
    idx = 0
    for suite in ("Spatial", "Object", "Goal", "Long"):
        prefix = {
            "Spatial": "libero_spatial",
            "Object": "libero_object",
            "Goal": "libero_goal",
            "Long": "libero_10",
        }[suite]
        for slot in range(10):
            logical = f"pre_a3_{prefix}_task{slot:02d}"
            concrete = {
                "clean": f"{prefix}_{slot:06d}",  # 0-based, to be amended
                "camera": f"{prefix}_{100 + slot:06d}",
                "robot": f"{prefix}_{200 + slot:06d}",
            }
            for dim, level in (("clean", 0), ("camera", 1), ("robot", 1)):
                rows.append(
                    {
                        "request_index": idx,
                        "suite": suite,
                        "task_id": logical,
                        "concrete_task_id": concrete[dim],
                        "dimension": dim,
                        "level": level,
                        "split": "train" if slot < 6 else ("val" if slot < 8 else "test"),
                        "episode_id": f"ep-{idx}",
                    }
                )
                idx += 1
    return {
        "artifact_version": "rase-pre-a3-design/v1",
        "n_requests": 120,
        "design_sha256": "parent",
        "split_counts": {"train": 72, "val": 24, "test": 24},
        "records": rows,
    }


def test_parse_and_amend_clean_ids():
    suite, numeric = parse_concrete_task_id("libero_spatial_000003")
    assert suite == "libero_spatial"
    assert numeric == 3
    amended = amend_design_v1_to_v1_1(_minimal_design())
    cleans = [
        row["concrete_task_id"]
        for row in amended["records"]
        if row["dimension"] == "clean"
    ]
    assert all(parse_concrete_task_id(value)[1] in range(1, 11) for value in cleans)
    assert amended["design_amendment"]["clean_ids_rewritten"] == 40


def test_requests_from_amended_design():
    amended = amend_design_v1_to_v1_1(_minimal_design())
    requests = requests_from_design(amended, seed=2026080401)
    assert len(requests) == 120
    clean = [req for req in requests if req.dimension == "clean"]
    assert len(clean) == 40
    assert all(req.task_id in range(1, 11) for req in clean)
    assert len({req.episode_id for req in requests}) == 120
