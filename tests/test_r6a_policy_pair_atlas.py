from scripts.audit_r6a_policy_pair_atlas import evaluate_pair


def make_rows(source_successes):
    source = []
    oft = {}
    suites = ("Spatial", "Object", "Goal", "Long")
    for index, success in enumerate(source_successes):
        state = f"s{index}"
        source.append({
            "state_key": state,
            "suite": suites[index % 4],
            "task_id": index,
            "source_success": success,
        })
        oft[state] = {"success": True, "env_steps": 100}
    return source, oft


def test_pair_gate_passes_with_distributed_source_safe_support():
    source, oft = make_rows([True] * 16 + [False] * 16)
    report = evaluate_pair(source, oft, min_savings=0.3, min_source_safe_tasks=12)
    assert report["privileged_teacher_savings"] == 0.5
    assert report["privileged_trigger_successes"] == 32
    assert report["all_seed_gates_passed"]


def test_pair_gate_rejects_insufficient_savings():
    source, oft = make_rows([True] * 8 + [False] * 24)
    report = evaluate_pair(source, oft, min_savings=0.3, min_source_safe_tasks=8)
    assert report["privileged_teacher_savings"] == 0.25
    assert not report["gates"]["teacher_savings_at_least_margin"]
    assert not report["all_seed_gates_passed"]
