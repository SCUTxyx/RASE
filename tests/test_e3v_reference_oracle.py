from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(state: str, task: str, rollout: int, success: bool, baseline: bool = False) -> dict:
    return {
        "state_key": state,
        "task_id": task,
        "suite": "Spatial",
        "rollout_seed": rollout + 10,
        "single_reference_success": baseline,
        "executed_action_steps": 3,
        "initial_chunk_sha256": f"chunk-{state}-{rollout}",
        "trace_sha256": f"trace-{state}-{rollout}",
        "result": {"success": success, "continuation_steps": 3},
    }


def test_balanced_freeze_prioritizes_single_reference_failures() -> None:
    module = load_script("freeze_e3v_reference_roots")
    rows = [
        {"state_key": "a", "task_id": "t1", "step": 0, "single_reference_success": True},
        {"state_key": "b", "task_id": "t1", "step": 2, "single_reference_success": False},
        {"state_key": "c", "task_id": "t2", "step": 0, "single_reference_success": False},
    ]
    selected = module.balanced_records(rows, 2)
    assert {item["state_key"] for item in selected} == {"b", "c"}


def test_audit_passes_sufficient_diverse_rescues() -> None:
    module = load_script("analyze_e3v_reference_oracle")
    rows = []
    for state_index in range(4):
        for rollout in range(2):
            rows.append(row(f"s{state_index}", f"t{state_index % 2}", rollout, rollout == 1))
    result = module.analyze(
        rows,
        min_roots=4,
        min_tasks=2,
        min_successful_trajectories=4,
        min_successful_tasks=2,
        min_oracle_coverage=0.5,
        min_single_failure_rescue_tasks=2,
    )
    assert result["decision"] == "PASS"
    assert result["metrics"]["rescued_single_reference_failure_roots"] == 4


def test_small_cohort_requires_expansion_not_false_pass() -> None:
    module = load_script("analyze_e3v_reference_oracle")
    result = module.analyze(
        [row("s0", "t0", 0, True)],
        min_roots=4,
        min_tasks=2,
        min_successful_trajectories=1,
        min_successful_tasks=1,
        min_oracle_coverage=0.1,
        min_single_failure_rescue_tasks=1,
    )
    assert result["decision"] == "EXPAND_REQUIRED"


def test_oft_cohort_gate_keeps_supervision_and_system_claims_separate(tmp_path: Path) -> None:
    module = load_script("analyze_e3v_oft_cohort")
    records = []
    summaries = []
    summary_records = []
    actions_by_path = {}
    for index in range(4):
        key = f"s{index}"
        records.append(
            {
                "state_key": key,
                "task_id": f"t{index}",
                "suite": "Spatial",
                "single_reference_success": index < 3,
            }
        )
        trace_path = tmp_path / f"{key}.npz"
        trace_path.write_bytes(f"trace-{index}".encode())
        actions_by_path[trace_path] = np.zeros((1, 3, 7), dtype=np.float32)
        summary_records.append(
            {
                "state_key": key,
                "direct_oft_result": {
                    "success": index < 3,
                    "continuation_steps": 3,
                    "stop_reason": "success" if index < 3 else "horizon",
                },
            }
        )
    protocol = {"protocol_sha256": "abc", "records": records}
    summaries.append(("summary.json", {"status": "complete", "suite": "libero_spatial", "records": summary_records}))
    result = module.audit_cohort(
        protocol,
        summaries,
        tmp_path,
        action_loader=lambda path: actions_by_path[path],
        min_successful_trajectories=3,
        min_successful_tasks=3,
        min_coverage=0.5,
    )
    assert result["decision"] == "PASS"
    assert result["scientific_scope"] == "development_only_residual_supervision_viability"
    assert result["next_gate"]["name"].startswith("E3-U")


def test_oft_cohort_gate_fails_outcome_drift(tmp_path: Path) -> None:
    module = load_script("analyze_e3v_oft_cohort")
    trace_path = tmp_path / "s0.npz"
    trace_path.write_bytes(b"trace")
    protocol = {
        "records": [
            {
                "state_key": "s0",
                "task_id": "t0",
                "suite": "Spatial",
                "single_reference_success": True,
            }
        ]
    }
    summaries = [
        (
            "summary.json",
            {
                "status": "complete",
                "records": [
                    {
                        "state_key": "s0",
                        "direct_oft_result": {
                            "success": False,
                            "continuation_steps": 2,
                            "stop_reason": "horizon",
                        },
                    }
                ],
            },
        )
    ]
    result = module.audit_cohort(
        protocol,
        summaries,
        tmp_path,
        action_loader=lambda _: np.zeros((1, 2, 7), dtype=np.float32),
        min_successful_trajectories=0,
        min_successful_tasks=0,
        min_coverage=0,
    )
    assert result["decision"] == "FAIL"
    assert result["errors"]["outcome_mismatches"] == ["s0"]


def test_teacher_prefix_validates_shape_and_horizon() -> None:
    module = load_script("rollout_e3v_oft_prefix_smol")
    actions = np.arange(1 * 8 * 7, dtype=np.float32).reshape(1, 8, 7)
    prefix = module.teacher_prefix(actions, 5)
    assert prefix.shape == (5, 7)
    assert np.array_equal(prefix, actions[0, :5])
    try:
        module.teacher_prefix(actions, 9)
    except ValueError as exc:
        assert "needs 9" in str(exc)
    else:
        raise AssertionError("short trajectory must be rejected")


def test_residual_dataset_canonicalizes_plus_instruction() -> None:
    module = load_script("build_e3_residual_dataset")
    clean = "put the bowl on the plate"
    plus = "put the bowl on the plate view 0 0 100 356 8 initstate 0"
    assert module.canonical_instruction(clean) == clean
    assert module.canonical_instruction(plus) == clean
    assert np.array_equal(module.language_hash(clean), module.language_hash(plus))
    assert np.isclose(np.linalg.norm(module.language_hash(clean)), 1.0)


def test_residual_ridge_group_folds_and_fit() -> None:
    module = load_script("train_e3_residual_ridge")
    groups = ["a", "a", "b", "b", "c", "c"]
    folds = module.group_folds(groups, 3)
    assert sum(mask.astype(int) for mask in folds).tolist() == [1] * len(groups)
    for mask in folds:
        validation_groups = {groups[index] for index in np.flatnonzero(mask)}
        training_groups = {groups[index] for index in np.flatnonzero(~mask)}
        assert validation_groups.isdisjoint(training_groups)
    x = np.arange(18, dtype=np.float32).reshape(6, 3)
    y = np.column_stack((x[:, 0] * 2, x[:, 1] - 1))
    model = module.fit_ridge(x, y, 1e-5)
    assert np.mean((module.predict(model, x) - y) ** 2) < 1e-6


def test_residual_chunk_is_additive_and_action_bounded() -> None:
    module = load_script("rollout_e3_residual_smol")
    source = np.zeros((5, 7), dtype=np.float32)
    delta = np.full((5, 7), 2.0, dtype=np.float32)
    result = module.corrected_chunk(source, delta)
    assert result.shape == (5, 7)
    assert np.all(result == 1.0)


def test_native_capture_supports_smol_private_get_action_chunk(monkeypatch) -> None:
    import torch
    from rase.collect import policy_step

    class FakePolicy:
        def __init__(self):
            self._queues = {"action": []}

        def predict_action_chunk(self, batch):
            raise AssertionError("Smol select_action does not call this public method")

        def _get_action_chunk(self, batch):
            del batch
            return torch.arange(21, dtype=torch.float32).reshape(1, 3, 7)

    policy = FakePolicy()
    bundle = {
        "policy": policy,
        "postprocessor": lambda value: value,
        "env_postprocessor": lambda transition: transition,
    }

    def fake_select(_bundle, _observation, *, task):
        del _observation, task
        return policy._get_action_chunk({})[0, 0].numpy()

    monkeypatch.setattr(policy_step, "select_env_action", fake_select)
    first, event = policy_step.capture_inference_event(
        bundle, {}, task="task", boundary_step=0, generation_seed=1, horizon=2
    )
    assert event.env_chunk.shape == (2, 7)
    assert np.array_equal(first, event.env_chunk[0])


def test_policy_fingerprint_handles_lerobot_dict_of_tensor_deques() -> None:
    from collections import deque
    import torch
    from rase.collect.policy_step import policy_state_fingerprint

    policy = type("Policy", (), {})()
    policy._queues = {"action": deque([torch.arange(7), torch.arange(7) + 1])}
    digest = policy_state_fingerprint({"policy": policy})
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_step_residual_weighted_ridge_fits_linear_targets() -> None:
    module = load_script("train_e3_step_residual")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    truth = rng.normal(size=(5, 2)).astype(np.float32)
    y = x @ truth + np.array([0.3, -0.2], dtype=np.float32)
    weights = np.linspace(1, 2, len(x))
    model = module.fit_weighted_ridge(x, y, weights, 1e-8)
    assert np.mean((module.predict(model, x) - y) ** 2) < 1e-8


def test_step_demo_canonical_action_removes_only_singleton_batch() -> None:
    module = load_script("collect_e3_step_demos")
    expected = np.arange(7, dtype=np.float32)
    assert np.array_equal(module.canonical_action(expected), expected)
    assert np.array_equal(module.canonical_action(expected[None, :]), expected)
    for invalid in (np.zeros((2, 7)), np.zeros((1, 1, 7)), np.zeros(6)):
        try:
            module.canonical_action(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid action shape {invalid.shape} must be rejected")


def test_step_residual_corrected_action_scales_and_clips() -> None:
    module = load_script("rollout_e3_step_residual")
    source = np.array([0.9, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    delta = np.array([0.4, 0.4, 0, 0, 0, 0, 0], dtype=np.float32)
    result = module.corrected_action(source, delta, 0.5)
    assert np.allclose(result[:2], [1.0, 0.2])
    try:
        module.corrected_action(source, delta, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero residual scale must be rejected")


def test_route_c_history_is_right_aligned_and_schema_exact() -> None:
    module = load_script("rollout_e3_step_residual")
    entry = {
        "proprio": np.arange(8, dtype=np.float32),
        "source_action": np.arange(7, dtype=np.float32) + 10,
        "progress": 0.25,
        "executed_action": np.arange(7, dtype=np.float32) + 20,
    }
    result = module.route_c_history([entry], window=3)
    assert result.shape == (3, 23)
    assert np.all(result[:2] == 0)
    assert np.array_equal(result[-1, :8], entry["proprio"])
    assert np.array_equal(result[-1, 8:15], entry["source_action"])
    assert result[-1, 15] == 0.25
    assert np.array_equal(result[-1, 16:], entry["executed_action"])
