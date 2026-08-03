import copy

from rase.selector.lightweight import (
    ABSTAIN,
    CONTINUE_SMOL,
    ESCALATE_OFT,
    LightweightSelector,
    audit_selector_dataset,
    build_direct_escalation_rows,
    build_direct_policy_rows,
    build_policy_matrix_proxy_rows,
    evaluate_selector,
    fit_lightweight_selector,
)


def _row(index, *, split_signal, clean, direct=True):
    continue_success = clean
    escalate_success = clean or split_signal > 0
    return {
        "state_key": f"state-{index}",
        "task_id": f"task-{index}",
        "episode_id": f"episode-{index}",
        "suite": "Spatial" if index % 2 else "Object",
        "perturb_dim": "camera" if index % 2 else "robot",
        "perturb_sub": "x",
        "episode_outcome": "success" if clean else "failure",
        "cohort": "clean_control" if clean else "failure_challenge",
        "features": {"risk": float(split_signal), "progress": index / 20},
        "arms": {
            CONTINUE_SMOL: {
                "success": continue_success,
                "cost": 0.02,
                "observed": True,
                "proxy": not direct,
            },
            ESCALATE_OFT: {
                "success": escalate_success,
                "cost": 0.1,
                "observed": True,
                "proxy": not direct,
            },
            ABSTAIN: {
                "success": False,
                "cost": 0.0,
                "observed": True,
                "proxy": False,
            },
        },
    }


def test_matrix_rows_are_explicit_non_trainable_proxies():
    matrix = {
        "schema_version": "rase-one-shot-policy-matrix/v1",
        "cohort": "failure_challenge",
        "per_state": [
            {
                "state_key": "s",
                "smol_portfolio_hit": False,
                "oft_portfolio_hit": True,
                "level": 1,
            }
        ],
    }
    rows = build_policy_matrix_proxy_rows(
        matrix,
        metadata_by_state={
            "s": {
                "task_id": "task",
                "episode_id": "ep",
                "suite": "Spatial",
                "perturb_dim": "camera",
                "perturb_sub": "x",
                "level": 1,
                "step": 4,
                "episode_outcome": "failure",
            }
        },
    )
    splits = {"splits": {"train": ["s"], "val": [], "test": []}}
    audit = audit_selector_dataset(rows, splits, min_train_states=1)
    assert not audit.ready
    assert rows[0]["arms"][ESCALATE_OFT]["proxy"] is True
    assert any("proxy outcomes" in reason for reason in audit.reasons)
    assert any("clean-success" in reason for reason in audit.reasons)


def test_direct_escalation_rows_are_deployable_not_portfolio_proxies():
    smol = {
        "per_state": [
            {
                "state_key": "s",
                "candidates": [
                    {"successes": 0, "trials": 1},
                    {"successes": 1, "trials": 1},
                ],
            }
        ]
    }
    direct = [{
        "schema_version": "rase-oft-direct-escalation/v1",
        "status": "complete",
        "per_state": [{"state_key": "s", "direct_oft_success": True}],
    }]
    metadata = {
        "s": {
            "task_id": "task",
            "episode_id": "ep",
            "suite": "Spatial",
            "perturb_dim": "camera",
            "perturb_sub": "viewpoint",
            "level": 1,
            "step": 2,
            "episode_outcome": "failure",
        }
    }
    rows = build_direct_escalation_rows(
        smol, direct, metadata_by_state=metadata, candidate_index=0
    )
    assert rows[0]["arms"][CONTINUE_SMOL]["success"] is False
    assert rows[0]["arms"][ESCALATE_OFT]["success"] is True
    assert rows[0]["arms"][CONTINUE_SMOL]["proxy"] is False
    assert rows[0]["arms"][ESCALATE_OFT]["proxy"] is False
    assert rows[0]["arms"][CONTINUE_SMOL]["cost"] > rows[0]["arms"][ABSTAIN]["cost"]


def test_direct_policy_rows_use_true_direct_smol_and_deployable_features():
    smol = {
        "schema_version": "rase-smol-direct-continuation/v1",
        "status": "complete",
        "per_state": [{"state_key": "s", "direct_smol_success": True}],
    }
    oft = [{
        "schema_version": "rase-oft-direct-escalation/v1",
        "status": "complete",
        "per_state": [{"state_key": "s", "direct_oft_success": False}],
    }]
    metadata = {"s": {
        "task_id": "task", "episode_id": "ep", "suite": "Goal",
        "perturb_dim": "clean", "perturb_sub": "none", "level": 0,
        "step": 4, "episode_outcome": "success",
    }}
    rows = build_direct_policy_rows(
        smol,
        oft,
        metadata_by_state=metadata,
        features_by_state={"s": {"t0": 4.0, "image_mean": 0.5}},
        cohort="clean_control",
    )
    assert rows[0]["arms"][CONTINUE_SMOL]["success"] is True
    assert rows[0]["arms"][CONTINUE_SMOL]["outcome_semantics"] == "direct_smol_from_snapshot"
    assert rows[0]["features"] == {"t0": 4.0, "image_mean": 0.5}


def test_audit_rejects_ground_truth_perturbation_feature():
    rows = [_row(index, split_signal=(-1 if index % 2 else 1), clean=bool(index % 2)) for index in range(8)]
    rows[0]["features"]["level"] = 3.0
    splits = {"splits": {"train": [f"state-{index}" for index in range(8)]}}
    audit = audit_selector_dataset(rows, splits, min_train_states=1)
    assert not audit.ready
    assert any("forbidden deployment features" in reason for reason in audit.reasons)


def test_audit_detects_episode_group_leakage():
    rows = [_row(0, split_signal=-1, clean=True), _row(1, split_signal=1, clean=False)]
    rows[1]["task_id"] = rows[0]["task_id"]
    rows[1]["episode_id"] = rows[0]["episode_id"]
    splits = {"splits": {"train": ["state-0"], "val": [], "test": ["state-1"]}}
    audit = audit_selector_dataset(rows, splits, min_train_states=1)
    assert not audit.ready
    assert audit.group_leakage


def test_small_selector_learns_cost_sensitive_escalation():
    rows = []
    for index in range(40):
        kind = index % 3
        clean = kind == 0
        signal = -1 if clean else (1 if kind == 1 else 0)
        rows.append(_row(index, split_signal=signal, clean=clean))
    splits = {
        "splits": {
            "train": [f"state-{index}" for index in range(30)],
            "val": [f"state-{index}" for index in range(30, 35)],
            "test": [f"state-{index}" for index in range(35, 40)],
        }
    }
    audit = audit_selector_dataset(rows, splits, min_train_states=20)
    assert audit.ready, audit.reasons
    model = fit_lightweight_selector(rows[:30], ridge=0.01)
    assert model.n_parameters < 100
    payload = model.to_dict()
    assert payload["n_parameters"] == model.n_parameters
    metrics = evaluate_selector(model, rows[35:])
    assert metrics["learned"]["success_rate"] == 0.6
    assert metrics["learned"]["mean_utility"] > metrics["always_escalate"]["mean_utility"]


def test_action_matched_random_matches_escalate_and_abstain_budgets():
    rows = [
        _row(0, split_signal=-1, clean=True),
        _row(1, split_signal=0, clean=False),
        _row(2, split_signal=1, clean=False),
    ]
    for row, choice in zip(rows, (-1.0, 0.0, 1.0)):
        row["features"] = {"choice": choice}
    model = LightweightSelector(
        numeric_features=["choice"],
        categorical_fields=[],
        categories={},
        means={"choice": 0.0},
        scales={"choice": 1.0},
        coefficients={
            CONTINUE_SMOL: [0.0, 0.0],
            ESCALATE_OFT: [0.0, 1.0],
            ABSTAIN: [0.0, -1.0],
        },
        ridge=1.0,
        success_reward=1.0,
    )
    metrics = evaluate_selector(model, rows)
    assert metrics["learned"]["action_counts"] == {
        CONTINUE_SMOL: 1,
        ESCALATE_OFT: 1,
        ABSTAIN: 1,
    }
    assert metrics["matched_random_actions"]["action_counts"] == metrics["learned"][
        "action_counts"
    ]
    assert metrics["matched_random_trigger"]["action_counts"][ABSTAIN] == 0
    delta = metrics["paired_utility_differences"][
        "learned_minus_matched_random_actions"
    ]
    assert delta["n_pairs"] == 3
    assert delta["bootstrap_ci_95"]["lower"] <= delta["mean_difference"]
    assert delta["mean_difference"] <= delta["bootstrap_ci_95"]["upper"]
    assert evaluate_selector(model, rows)["paired_utility_differences"] == metrics[
        "paired_utility_differences"
    ]


def test_audit_rejects_proxy_even_with_balanced_labels():
    rows = [
        _row(index, split_signal=(-1 if index % 2 else 1), clean=bool(index % 2))
        for index in range(8)
    ]
    rows[0] = copy.deepcopy(rows[0])
    rows[0]["arms"][CONTINUE_SMOL]["proxy"] = True
    splits = {"splits": {"train": [f"state-{index}" for index in range(8)]}}
    audit = audit_selector_dataset(rows, splits, min_train_states=1)
    assert not audit.ready
    assert audit.arm_counts[CONTINUE_SMOL]["proxy"] == 1
