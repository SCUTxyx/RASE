from rase.collect.pre_a3 import (
    analyze_recovery_duration,
    assign_task_disjoint_splits,
    build_design,
    decide_method_gate,
)


def _design_records():
    rows = []
    idx = 0
    for suite_i, suite in enumerate(("Spatial", "Object", "Goal", "Long")):
        for task_slot in range(10):
            task = f"{suite.lower()}_task{task_slot:02d}"
            for dim, level in (("clean", 0), ("camera", 1), ("robot", 1)):
                rows.append(
                    {
                        "request_index": idx,
                        "suite": suite,
                        "task_id": task,
                        "dimension": dim,
                        "level": level,
                        "episode_id": f"ep-{idx}",
                    }
                )
                idx += 1
    return rows


def test_build_design_and_splits():
    design = build_design(_design_records(), seed=123)
    assert design["n_requests"] == 120
    assert design["n_unique_tasks"] == 40
    assert design["split_counts"] == {"train": 72, "val": 24, "test": 24}
    assert all(row["split"] in {"train", "val", "test"} for row in design["records"])


def test_assign_task_disjoint_splits_counts():
    tasks = {
        suite: [f"{suite}_{i}" for i in range(10)]
        for suite in ("Spatial", "Object", "Goal", "Long")
    }
    assignment = assign_task_disjoint_splits(tasks, seed=7)
    assert sum(value == "train" for value in assignment.values()) == 24
    assert sum(value == "val" for value in assignment.values()) == 8
    assert sum(value == "test" for value in assignment.values()) == 8


def _duration_payload(outcomes, directs, splits=None, cells=None):
    lengths = [0, 8, 32, 64]
    default_cells = [("camera", 1), ("robot", 1), ("camera", 1), ("robot", 1), ("clean", 0), ("clean", 0)]
    rows = []
    for i, row in enumerate(outcomes):
        dim, level = (cells or default_cells)[i % len(default_cells)]
        rows.append(
            {
                "state_key": f"s{i}",
                "task_id": f"t{i}",
                "suite": ["Spatial", "Object", "Goal", "Long"][i % 4],
                "perturbation_dimension": dim,
                "perturbation_level": level,
                "split": (splits or ["test"] * len(outcomes))[i],
                "arms": [{"success": value} for value in row],
                "direct_oft_success": directs[i],
            }
        )
    return {"prefix_lengths": lengths, "per_state": rows}


def test_pre_a3_gate_pass_and_method_decision():
    # Construct a hidden-test cohort that satisfies all confirmatory conditions.
    # Rescues peak at different durations so adaptive oracle beats any fixed h.
    outcomes = [
        [False, False, True, False],  # only h=32
        [False, True, False, False],  # only h=8
        [False, False, False, True],  # only h=64
        [False, False, True, True],  # h=32/64
        [True, True, True, True],  # base success preserved
        [True, True, True, True],
    ]
    directs = [True, True, True, True, True, True]
    duration = _duration_payload(outcomes, directs, splits=["test"] * 6)
    keys = {
        "records": [
            {
                "state_key": f"s{i}",
                "task_id": f"t{i}",
                "suite": duration["per_state"][i]["suite"],
                "cell": (
                    f"{duration['per_state'][i]['perturbation_dimension']}:"
                    f"L{duration['per_state'][i]['perturbation_level']}"
                ),
                "split": "test",
            }
            for i in range(6)
        ]
    }
    audit = analyze_recovery_duration(duration, keys=keys, bootstrap_replicates=200, bootstrap_seed=0)
    assert audit["gate_pass"] is True
    assert audit["termination_model_gate"] == "open"
    gate = decide_method_gate(audit, val_audit=audit)
    assert gate["decision"] == "enter_safe_handback_method"
    assert gate["world_model_gate"] == "closed"


def test_pre_a3_gate_fail_keeps_benchmark_track():
    outcomes = [[False, False, False, False] for _ in range(4)]
    directs = [True, True, True, False]
    duration = _duration_payload(outcomes, directs, splits=["test"] * 4)
    keys = {
        "records": [
            {
                "state_key": f"s{i}",
                "task_id": f"t{i}",
                "suite": "Goal",
                "cell": "camera:L1",
                "split": "test",
            }
            for i in range(4)
        ]
    }
    audit = analyze_recovery_duration(duration, keys=keys, bootstrap_replicates=50, bootstrap_seed=1)
    assert audit["gate_pass"] is False
    gate = decide_method_gate(audit)
    assert gate["decision"] == "benchmark_diagnosis_only"
    assert gate["termination_model_gate"] == "closed"
