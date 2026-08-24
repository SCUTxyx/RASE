from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _branch(root_id: str, task: str, arm: str, success: bool, *, index=None, seed=None):
    source_seed = 1000 + int(root_id[-1])
    return {
        "schema_version": "rase-v6-stage0-branch/v1",
        "status": "complete",
        "root_id": root_id,
        "state_key": f"state-{root_id}",
        "task_id": task,
        "suite": "libero_10",
        "perturb_dim": "position",
        "perturb_level": 0.2,
        "cursor": 5,
        "native_chunk_horizon": 10,
        "actual_cursor_fraction": 0.5,
        "root_snapshot_sha256": f"snapshot-{root_id}",
        "source_generation_seed": source_seed,
        "source_temperature": 0.5,
        "arm": arm,
        "arm_index": index,
        "candidate_generation_seed": seed,
        "candidate_chunk_steps": 5,
        "downstream_controller": "same_source_fixed_mu",
        "downstream_seed": 9000 + int(root_id[-1]),
        "downstream_temperature": 0.5,
        "success": success,
    }


def _root_rows(root_index: int, task: str, *, c: bool, new: list[bool], same: bool | None = None):
    root_id = f"root-{root_index}"
    source_seed = 1000 + root_index
    rows = [
        _branch(root_id, task, "C", c),
        _branch(root_id, task, "R_same", c if same is None else same, seed=source_seed),
    ]
    rows.extend(
        _branch(root_id, task, "R_new", value, index=index, seed=2000 + root_index * 10 + index)
        for index, value in enumerate(new)
    )
    return rows


def test_root_plan_is_outcome_blind_balanced_and_uses_unique_states(tmp_path) -> None:
    module = load_script("freeze_v6_stage0_roots")
    pool = tmp_path / "pool"
    states = {}
    for task_index in range(3):
        for episode in range(4):
            key = f"sp1_{task_index}{episode:07d}"
            relative = f"task{task_index}/episode{episode}/000000"
            states[key] = {"path": relative}
            path = pool / relative
            path.mkdir(parents=True)
            raw_dimension = "position" if episode % 2 == 0 else "robot"
            metadata = {
                "suite": "libero_10",
                "task_id": f"libero_10/task_{task_index}",
                "episode_id": f"episode-{episode}",
                "seed": episode,
                "perturb_dim": raw_dimension,
                "level": 0.2,
            }
            if raw_dimension == "robot":
                metadata["perturb_sub"] = "position"
            (path / "meta.json").write_text(json.dumps(metadata))
    (pool / "manifest.json").write_text(json.dumps({"states": states}))
    args = SimpleNamespace(
        pool=pool, suite="long", perturb_dim="position", perturb_level=0.2,
        n_roots=9, native_chunk_horizon=10, cursors="3,5,8", r_new_k=4, seed=7,
    )
    plan = module.build_plan(args)
    assert plan["selection_outcomes_used"] is False
    assert len(plan["roots"]) == 9
    assert {row["cursor"] for row in plan["roots"]} == {3, 5, 8}
    assert len({row["state_key"] for row in plan["roots"]}) == 9
    assert {row["pool_perturb_dim"] for row in plan["roots"]} <= {"position", "robot"}
    for row in plan["roots"]:
        assert row["source_generation_seed"] not in row["r_new_generation_seeds"]
        assert len(set(row["r_new_generation_seeds"])) == 4


def test_pilot_gate_requires_two_sided_statewise_opportunity() -> None:
    module = load_script("analyze_v6_refresh_opportunity")
    rows = []
    # Three roots favor C and three favor refresh, spread across task clusters.
    for index in range(6):
        rows.extend(_root_rows(index, f"task-{index % 3}", c=index < 3, new=[index >= 3] * 4))
    artifact = module.analyse(rows, expected_k=4, mode="pilot", n_bootstrap=200, seed=3)
    summary = artifact["summary"]
    assert summary["continue_better_roots"] == 3
    assert summary["refresh_better_roots"] == 3
    assert summary["ac_refresh"] > 0.0
    assert artifact["gate"]["decision"] == "PASS"


def test_primary_refresh_value_is_the_mean_not_best_of_k() -> None:
    module = load_script("analyze_v6_refresh_opportunity")
    rows = _root_rows(0, "task-0", c=True, new=[True, False, False, False])
    artifact = module.analyse(rows, expected_k=4, mode="pilot", n_bootstrap=0, seed=1)
    root = artifact["per_root"][0]
    assert root["q_refresh_new_mean"] == 0.25
    assert root["q_refresh_new_best_diagnostic"] == 1.0
    assert root["adv_refresh_new"] == -0.75


def test_adaptive_k8_analysis_accepts_only_preregistered_k_values() -> None:
    module = load_script("analyze_v6_refresh_opportunity")
    rows = _root_rows(0, "task-0", c=False, new=[True] * 8)
    artifact = module.analyse(
        rows, expected_k=None, allowed_k={4, 8}, mode="pilot", n_bootstrap=0, seed=1,
    )
    assert artifact["n_auditable_roots"] == 1
    assert artifact["per_root"][0]["r_new_k"] == 8
    artifact = module.analyse(
        rows[:-1], expected_k=None, allowed_k={4, 8}, mode="pilot", n_bootstrap=0, seed=1,
    )
    assert artifact["n_auditable_roots"] == 0
    assert "expected_R_new_k_4_or_8_got_7" in artifact["audit_failures"]["root-0"]


def test_k8_refinement_selects_all_and_only_mixed_original_roots(tmp_path) -> None:
    module = load_script("refine_v6_stage0_k8")
    roots = tmp_path / "roots"
    roots.mkdir()
    for index, outcomes in enumerate(([False] * 4, [False, True, False, True], [True] * 4)):
        root_id = f"root-{index}"
        record = {
            "status": "complete",
            "root": {"root_id": root_id},
            "branches": _root_rows(index, "task-0", c=False, new=list(outcomes)),
        }
        (roots / f"{root_id}.json").write_text(json.dumps(record))
    selected = module.select_mixed_roots(tmp_path)
    assert [entry["root"]["root_id"] for entry in selected] == ["root-1"]
    root = dict(selected[0]["root"])
    root["source_generation_seed"] = 1001
    assert len(module.refinement_seeds(root, selected[0]["original_r_new_seeds"])) == 4


def test_audit_rejects_unmatched_r_same_seed() -> None:
    module = load_script("analyze_v6_refresh_opportunity")
    rows = _root_rows(0, "task-0", c=False, new=[True, True, True, True])
    rows[1]["candidate_generation_seed"] = 123456
    artifact = module.analyse(rows, expected_k=4, mode="pilot", n_bootstrap=0, seed=1)
    assert artifact["n_auditable_roots"] == 0
    assert artifact["audit_failures"]["root-0"] == ["r_same_seed_not_matched"]


def test_audit_accepts_v2_stale_action_causality_metadata() -> None:
    module = load_script("analyze_v6_refresh_opportunity")
    rows = _root_rows(0, "task-0", c=False, new=[True, False, False, False])
    for row in rows:
        row.update({
            "schema_version": "rase-v6-stage0-branch/v2",
            "old_chunk_source_qpos_sha256": "old-clean-state",
            "pre_perturb_boundary_qpos_sha256": "after-old-prefix",
            "perturbation_timing": "after_old_prefix_before_branch_decision",
            "position_target_mode": "first_goal_subject",
        })
    artifact = module.analyse(rows, expected_k=4, mode="pilot", n_bootstrap=0, seed=1)
    assert artifact["n_auditable_roots"] == 1
    assert artifact["n_audit_failures"] == 0


def test_capture_path_receives_smolvla_temperature() -> None:
    policy_step = (Path(__file__).resolve().parents[1] / "rase/collect/policy_step.py").read_text()
    rollout = (Path(__file__).resolve().parents[1] / "rase/collect/forked_rollout.py").read_text()
    assert "temperature: float | None = None" in policy_step
    assert "temperature=self.temperature" in rollout


def test_direct_position_plan_is_all_task_balanced_and_outcome_blind() -> None:
    module = load_script("collect_v6_stage0_direct_position")
    args = SimpleNamespace(
        cursors="3,5,8", native_horizon=10, tasks="1,2,3,4,5,6,7,8,9,10",
        suite="libero_10", init_states=10, seed=11, r_new_k=4, perturb_level=0.2,
        position_target_mode="all_goal_subjects", replicates_per_cell=1,
        plan_label="testpilot",
    )
    plan = module.make_root_plan(args)
    assert plan["selection_outcomes_used"] is False
    assert len(plan["roots"]) == 30
    for cursor in (3, 5, 8):
        roots = [row for row in plan["roots"] if row["cursor"] == cursor]
        assert len(roots) == 10
        assert {row["task_id"] for row in roots} == {
            f"libero_10_{task:06d}" for task in range(1, 11)
        }
        assert {row["position_target_mode"] for row in roots} == {"all_goal_subjects"}


def test_formal_plan_uses_independent_replicates_and_labels() -> None:
    module = load_script("collect_v6_stage0_direct_position")
    args = SimpleNamespace(
        cursors="3,5,8", native_horizon=10, tasks="1,2,3,4,5,6,7,8,9,10",
        suite="libero_10", init_states=10, seed=17, r_new_k=4, perturb_level=0.02,
        position_target_mode="first_goal_subject", replicates_per_cell=4,
        plan_label="formal_v1",
    )
    plan = module.make_root_plan(args)
    assert len(plan["roots"]) == 120
    assert len({row["root_id"] for row in plan["roots"]}) == 120
    assert {row["replicate"] for row in plan["roots"]} == {0, 1, 2, 3}
    assert all(row["root_id"].startswith("formal_v1_") for row in plan["roots"])
    assert plan["selection"]["replicates_per_cell"] == 4


def test_position_asset_names_are_normalized_before_perturbation() -> None:
    module = load_script("collect_v6_stage0_direct_position")
    assert module.canonical_object_name("bigger_alphabet_soup_1") == "alphabet_soup_1"
    assert module.canonical_object_name("red_cream_cheese_1") == "cream_cheese_1"
    assert module.canonical_object_name("alphabet_soup_1_main") == "alphabet_soup_1"
    assert module.target_matches_body("black_bowl_1", "akita_black_bowl_1_main")
    assert not module.target_matches_body("black_bowl_1", "white_cabinet_1_main")


def test_position_perturbation_resolves_the_exact_catalog_bddl(tmp_path) -> None:
    module = load_script("collect_v6_stage0_direct_position")
    requested = "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
    (tmp_path / f"{requested}.bddl").write_text(
        "(:obj_of_interest\n  book_1\n  desk_caddy_1\n)\n",
        encoding="utf-8",
    )
    # A shared natural-language prefix must not cause this decoy to be read.
    (tmp_path / "STUDY_SCENE1_pick_up_the_book_and_place_it_on_the_table.bddl").write_text(
        "(:obj_of_interest\n  decoy_1\n)\n", encoding="utf-8"
    )
    targets, path = module.exact_bddl_objects(tmp_path, requested)
    assert path.name == f"{requested}.bddl"
    assert targets == ["book_1", "desk_caddy_1"]


def test_position_perturbation_uses_execution_asset_bddl_tree(tmp_path) -> None:
    module = load_script("collect_v6_stage0_direct_position")
    clean = tmp_path / "clean"
    expected = clean / "libero" / "libero" / "bddl_files" / "libero_10"
    expected.mkdir(parents=True)
    assert module.clean_bddl_directory(clean, "libero_10") == expected


def test_position_perturbation_moves_goal_subjects_not_passive_receptacles(tmp_path) -> None:
    module = load_script("collect_v6_stage0_direct_position")
    bddl = tmp_path / "task.bddl"
    bddl.write_text(
        """
        (define (problem example)
          (:objects black_book_1 white_mug_1 - object)
          (:fixtures desk_caddy_1 - caddy)
          (:obj_of_interest black_book_1 white_mug_1 desk_caddy_1)
          (:goal (And (In black_book_1 desk_caddy_1_back_region)
                      (On white_mug_1 desk_caddy_1)))
        )
        """,
        encoding="utf-8",
    )
    assert module.bddl_goal_manipulated_objects(bddl) == ["black_book_1", "white_mug_1"]
    assert module.choose_perturbation_targets(
        ["black_book_1", "white_mug_1"], mode="all_goal_subjects"
    ) == ["black_book_1", "white_mug_1"]
    assert module.choose_perturbation_targets(
        ["black_book_1", "white_mug_1"], mode="first_goal_subject"
    ) == ["black_book_1"]


def test_stage0_orders_the_perturbation_after_old_prefix() -> None:
    module = load_script("collect_v6_stage0_direct_position")
    source = Path(module.__file__).read_text(encoding="utf-8")
    old_event = source.index("old_event = force_fresh_inference")
    prefix = source.index("for action in old_chunk[:cursor]:", old_event)
    perturb = source.index("moved = apply_position_perturbation(", prefix)
    boundary = source.index("boundary_snapshot = forkable.snapshot()", perturb)
    assert old_event < prefix < perturb < boundary
    assert module.parse_optional_temperature("default") is None
    assert module.parse_optional_temperature("0.5") == 0.5


def test_source_viability_temperature_parser() -> None:
    module = load_script("measure_v6_source_viability")
    assert module.parse_temperature("none") is None
    assert module.parse_temperature("default") is None
    assert module.parse_temperature("0.5") == 0.5
