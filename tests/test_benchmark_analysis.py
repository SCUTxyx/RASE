from rase.selector.benchmark_analysis import (
    analyze_selector_benchmark,
    paired_bootstrap_gap,
)
from rase.selector.lightweight import ABSTAIN, CONTINUE_SMOL, ESCALATE_OFT


def _row(key, suite, task, smol, oft, *, cohort="failure_challenge"):
    return {
        "state_key": key,
        "task_id": task,
        "episode_id": f"ep-{key}",
        "suite": suite,
        "cohort": cohort,
        "episode_outcome": "success" if cohort == "clean_control" else "failure",
        "perturb_dim": "camera",
        "level": 3,
        "arms": {
            CONTINUE_SMOL: {"observed": True, "success": smol, "cost": 0.02},
            ESCALATE_OFT: {"observed": True, "success": oft, "cost": 0.10},
            ABSTAIN: {"observed": True, "success": False, "cost": 0.0},
        },
    }


def test_composition_includes_zero_cells_and_oracle_gaps_are_deterministic():
    rows = [_row("a", "Spatial", "ta", True, False), _row("b", "Object", "tb", False, True)]
    splits = {"splits": {"train": ["a"], "test": ["b"]}}
    first = analyze_selector_benchmark(rows, splits=splits, bootstrap_seed=9, bootstrap_samples=100)
    second = analyze_selector_benchmark(
        rows, splits=splits, bootstrap_seed=9, bootstrap_samples=100
    )
    assert first == second
    cells = first["composition"]["cells"]
    assert first["composition"]["fields"] == [
        "split",
        "cohort",
        "suite",
        "episode_outcome",
        "direct_outcome",
    ]
    assert any(
        cell["split"] == "train"
        and cell["suite"] == "Object"
        and cell["direct_outcome"] == "oft_only"
        and cell["n_states"] == 0
        for cell in cells
    )
    assert first["oracle_action_support"]["action_counts"] == {
        CONTINUE_SMOL: 1,
        ESCALATE_OFT: 1,
        ABSTAIN: 0,
    }
    smol_gap = first["oracle_minus_fixed_policy_utility_gaps"][CONTINUE_SMOL]
    assert smol_gap["n_pairs"] == 2
    assert smol_gap["mean_difference"] == 0.46


def test_paired_bootstrap_empty_and_deterministic():
    assert paired_bootstrap_gap([], bootstrap_samples=10)["mean_difference"] is None
    assert paired_bootstrap_gap([0.0, 1.0], seed=2, bootstrap_samples=20) == paired_bootstrap_gap(
        [0.0, 1.0], seed=2, bootstrap_samples=20
    )


def test_suite_shortcut_uses_train_only():
    rows = [
        _row("a", "Spatial", "ta", True, False),
        _row("b", "Spatial", "tb", False, True),
        _row("c", "Spatial", "tc", False, True),
    ]
    result = analyze_selector_benchmark(
        rows,
        splits={"splits": {"train": ["a"], "test": ["b", "c"]}},
        bootstrap_samples=10,
    )
    shortcut = result["train_only_suite_shortcut"]
    assert shortcut["fit_scope"] == "train_only"
    assert shortcut["n_train_evaluable"] == 1
    assert shortcut["suite_majority_action"] == {"Spatial": CONTINUE_SMOL}
    assert shortcut["per_split"]["test"]["oracle_action_accuracy"] == 0.0
    assert shortcut["per_split"]["test"]["mean_oracle_utility_gap"] == 0.92


def test_direct_outcome_missing_is_not_both_fail():
    missing = _row("m", "Spatial", "tm", False, False)
    missing["arms"][CONTINUE_SMOL]["observed"] = False
    result = analyze_selector_benchmark([missing], bootstrap_samples=10)
    cells = result["composition"]["cells"]
    assert next(cell["n_states"] for cell in cells if cell["direct_outcome"] == "missing_smol") == 1
    assert next(cell["n_states"] for cell in cells if cell["direct_outcome"] == "both_fail") == 0


def test_split_warnings_report_cohorts_oracle_support_and_learned_unavailable():
    rows = [
        _row(str(index), "Spatial", f"t{index}", True, False, cohort="clean_control")
        for index in range(8)
    ]
    result = analyze_selector_benchmark(
        rows,
        splits={"splits": {"train": [], "test": [str(index) for index in range(8)]}},
        bootstrap_samples=10,
    )
    warnings = result["task_split_composition_warnings"]
    assert "split test cohort composition: clean_control=8, failure_challenge=0" in warnings
    assert "split test has 0 escalation oracle support" in warnings
    assert "split test learned action unavailable: no learned_action annotations" in warnings
    assert not any("0 learned escalation actions" in warning for warning in warnings)


def test_shortcut_unseen_suite_fallback_and_loso_semantics():
    rows = [
        _row("a", "Spatial", "ta", True, False),
        _row("b", "Object", "tb", False, True),
    ]
    result = analyze_selector_benchmark(
        rows,
        splits={"splits": {"train": ["a"], "test": ["b"]}},
        bootstrap_samples=10,
    )
    shortcut = result["train_only_suite_shortcut"]
    assert shortcut["fallback"]["semantics"] == "train_global_majority_then_preregistered"
    assert shortcut["fallback"]["applied_action"] == CONTINUE_SMOL
    assert shortcut["per_split"]["test"]["n_unseen_suite_fallback"] == 1
    assert shortcut["per_split"]["test"]["mean_oracle_utility_gap"] == 0.92

    object_fold = next(
        fold
        for fold in result["leave_suite_out_descriptive_folds"]
        if fold["held_out_suite"] == "Object"
    )
    fold_shortcut = object_fold["train_only_suite_shortcut"]
    assert fold_shortcut["fit_scope"] == "train_only"
    assert fold_shortcut["per_split"]["test"]["n_unseen_suite_fallback"] == 1
    assert fold_shortcut["fallback"]["applied_action"] == CONTINUE_SMOL
