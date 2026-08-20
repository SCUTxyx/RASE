import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT if (ROOT / "export_decision_context_keys.py").exists() else ROOT.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_filters_steps_suite_and_task():
    module = _load("export_decision_context_keys")
    rows = [
        {"state_key": f"k{i}", "task_id": task, "episode_id": task, "step": step, "suite": suite}
        for i, (task, suite, step) in enumerate(
            [("g1", "Goal", 0), ("g1", "Goal", 2), ("l1", "Long", 2), ("s1", "Spatial", 2)]
        )
    ]
    selected = module.select_records(
        rows, steps={0, 2}, suites={"Goal", "Long"}, task_ids={"g1"}
    )
    assert [row["state_key"] for row in selected] == ["k0", "k1"]


def _fixture():
    records = []
    continue_rows = []
    fallback_rows = []
    # t1 flips: CONTINUE wins at step 0, fallback wins at step 2.
    # t2 is tied at both states. Oracle gain = mean(0.5, 0.0) = 0.25.
    outcomes = {
        ("t1", 0): (True, False),
        ("t1", 2): (False, True),
        ("t2", 0): (True, True),
        ("t2", 2): (False, False),
    }
    for task, step in outcomes:
        key = f"{task}-{step}"
        records.append(
            {
                "state_key": key,
                "task_id": task,
                "episode_id": f"ep-{task}",
                "suite": "Goal",
                "step": step,
                "perturbation_dimension": "clean",
                "perturbation_level": 0,
            }
        )
        c, f = outcomes[(task, step)]
        continue_rows.append(
            {
                "state_key": key,
                "continue_smol_active_chunk": c,
                "continue_smol_active_chunk_env_steps": 10,
            }
        )
        fallback_rows.append(
            {
                "state_key": key,
                "direct_oft_success": f,
                "result": {
                    "prefix_source": "direct",
                    "prefix_steps": 0,
                    "env_steps": 10,
                    "stop_reason": "success" if f else "horizon",
                    "success": f,
                },
            }
        )
    keys = [row["state_key"] for row in records]
    return (
        {"state_keys": keys, "records": records},
        {"status": "complete", "per_pair": continue_rows},
        {"schema_version": "rase-oft-direct-escalation/v1", "status": "complete", "per_state": fallback_rows},
    )


def test_analysis_uses_strict_winners_and_task_best_fixed():
    module = _load("analyze_continue_fallback_opportunity")
    keys, continue_summary, fallback = _fixture()
    result = module.analyze(
        keys, continue_summary, [fallback], bootstrap_replicates=100, bootstrap_seed=7
    )
    assert result["metrics"]["within_task_heterogeneity"] == 0.5
    assert result["metrics"]["n_tasks_with_winner_flip"] == 1
    assert result["metrics"]["oracle_minus_task_best_fixed"] == 0.25
    assert result["metrics"]["state_winner_counts"] == {
        "continue": 1,
        "fallback": 1,
        "tie": 2,
    }
    assert result["status"] == "pass"
    assert result["metrics"]["fallback_weakly_dominates_every_state"] is False
    assert result["by_group"]["suite"]["Goal"]["n_states"] == 4
    assert result["by_group"]["suite"]["Goal"]["oracle_minus_task_best_fixed"] == 0.25
