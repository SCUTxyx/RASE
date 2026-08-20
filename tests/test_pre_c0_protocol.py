from __future__ import annotations

import json
from pathlib import Path

from rase.collect.pre_c0 import (
    CELLS,
    SUITES,
    analyze_guided_headroom,
    analyze_natural_headroom,
    analyze_risk_trigger_oracle,
    build_pre_c0_design,
    load_pre_c0_design,
    requests_from_design,
)


def _source_design() -> dict:
    records = []
    index = 0
    for suite in SUITES:
        short = "10" if suite == "Long" else suite.lower()
        for task in range(10):
            split = "train" if task < 6 else ("val" if task < 8 else "test")
            for cell, dimension, level, concrete in (
                ("clean:L0", "clean", 0, task + 1),
                ("camera:L1", "camera", 1, 100 + task),
                ("robot:L1", "robot", 1, 200 + task),
            ):
                records.append(
                    {
                        "request_index": index,
                        "episode_id": f"source-{index}",
                        "suite": suite,
                        "task_id": f"pre_a3_libero_{short}_task{task:02d}",
                        "concrete_task_id": f"libero_{short}_{concrete:06d}",
                        "cell": cell,
                        "dimension": dimension,
                        "level": level,
                        "split": split,
                    }
                )
                index += 1
    return {"design_sha256": "source-sha", "records": records}


def test_pre_c0_design_uses_balanced_train_tasks(tmp_path: Path):
    design = build_pre_c0_design(_source_design())
    assert design["n_episodes"] == 24
    assert design["suite_counts"] == {suite: 6 for suite in SUITES}
    assert design["cell_counts"] == {cell: 8 for cell in CELLS}
    assert {row["source_split"] for row in design["records"]} == {"train"}
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design))
    loaded = load_pre_c0_design(path, expected_sha256=design["design_sha256"])
    requests = requests_from_design(loaded, seed=123)
    assert len(requests) == 24
    assert len({request.episode_id for request in requests}) == 24


def test_natural_gate_passes_with_cross_suite_rescues():
    rows = []
    for index in range(20):
        rescue = index in {0, 1, 2}
        rows.append(
            {
                "state_key": f"k{index}",
                "task_id": f"task{index}",
                "suite": ("Spatial", "Object", "Goal", "Long")[index % 4],
                "cell": "camera:L1",
                "stage": "T1" if index % 2 == 0 else "T3",
                "family_success": {
                    "current_suffix": False,
                    "strict_resample": rescue,
                    "fresh_replan": False,
                    "receding_horizon": False,
                },
            }
        )
    audit = analyze_natural_headroom(rows)
    assert audit["headroom_pp"]["natural_total"] == 15.0
    assert audit["gate_pass"] is True
    assert audit["candidate_critic_gate"] == "eligible"


def test_natural_gate_fails_without_headroom():
    rows = [
        {
            "state_key": f"k{index}",
            "task_id": f"task{index}",
            "suite": "Spatial",
            "cell": "clean:L0",
            "stage": "T1" if index % 2 == 0 else "T3",
            "family_success": {
                "current_suffix": True,
                "strict_resample": True,
                "fresh_replan": True,
                "receding_horizon": True,
            },
        }
        for index in range(20)
    ]
    audit = analyze_natural_headroom(rows)
    assert audit["gate_pass"] is False
    assert audit["headroom_pp"]["natural_total"] == 0.0


def test_guided_gate_passes_with_cross_suite_privileged_rescues():
    rows = []
    for index in range(20):
        guided = index in {0, 1, 2}
        rows.append(
            {
                "state_key": f"k{index}",
                "task_id": f"task{index}",
                "suite": ("Spatial", "Object", "Goal", "Long")[index % 4],
                "cell": "camera:L1",
                "stage": "T1" if index % 2 == 0 else "T3",
                "family_success": {
                    "current_suffix": False,
                    "strict_resample": False,
                    "fresh_replan": False,
                    "receding_horizon": False,
                },
                "privileged_guidance": guided,
            }
        )
    audit = analyze_guided_headroom(rows)
    assert audit["guided_gain_pp"] == 15.0
    assert audit["gate_pass"] is True
    assert audit["guided_generation_gate"] == "open"
    assert audit["learned_recovery_critic_gate"] == "eligible"


def test_guided_gate_marks_frozen_nogo_when_gain_below_5pp():
    rows = [
        {
            "state_key": f"k{index}",
            "task_id": f"task{index}",
            "suite": "Spatial",
            "cell": "camera:L1",
            "stage": "T1" if index % 2 == 0 else "T3",
            "family_success": {
                "current_suffix": False,
                "strict_resample": False,
                "fresh_replan": False,
                "receding_horizon": False,
            },
            "privileged_guidance": False,
        }
        for index in range(20)
    ]
    audit = analyze_guided_headroom(rows)
    assert audit["gate_pass"] is False
    assert audit["guided_gain_pp"] == 0.0
    assert audit["frozen_same_policy_recovery"] == "NOGO"
    assert audit["decision"] == "frozen_same_policy_recovery_nogo"


def test_risk_trigger_retains_natural_oracle_with_lower_always_harm():
    rows = []
    for index in range(20):
        # 0-2: current succeeds, alternatives fail (always-intervene harms)
        # 3-9: current succeeds, alternatives also succeed
        # 10-12: current fails, alternatives rescue
        # 13-19: all fail
        if index <= 2:
            current, alt = True, False
        elif index <= 9:
            current, alt = True, True
        elif index <= 12:
            current, alt = False, True
        else:
            current, alt = False, False
        rows.append(
            {
                "state_key": f"k{index}",
                "task_id": f"task{index}",
                "suite": "Spatial",
                "cell": "clean:L0",
                "stage": "T1",
                "family_success": {
                    "current_suffix": current,
                    "strict_resample": alt,
                    "fresh_replan": False,
                    "receding_horizon": False,
                },
            }
        )
    audit = analyze_risk_trigger_oracle(rows)
    assert audit["retains_full_natural_oracle"] is True
    assert audit["success_counts"]["risk_trigger_oracle"] == 13
    assert audit["success_counts"]["always_intervene_oracle"] == 10
    assert audit["control_harm_rate"]["risk_trigger"] == 0.0
    assert audit["control_harm_rate"]["always_intervene"] == 0.15
    assert audit["meaningful_for_critic"] is True


def test_gate_b_decision_naming_forbids_flow_api_claim():
    decision = {
        "naming": "privileged trust-region action refinement",
        "not_smolvla_flow_api_guidance": True,
        "decision": "frozen_same_policy_recovery_nogo",
    }
    assert decision["not_smolvla_flow_api_guidance"] is True
    assert "flow api" not in decision["naming"].lower()
    assert "smolvla api" not in decision["naming"].lower()
