"""PRE-C0 same-policy corrective protocol and analysis helpers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .perturb_sampler import PerturbationRequest

PROTOCOL_VERSION = "rase-pre-c0-same-policy-corrective/v1"
DESIGN_VERSION = "rase-pre-c0-design/v1"
AUDIT_VERSION = "rase-pre-c0-same-policy-audit/v1"
GUIDED_AUDIT_VERSION = "rase-pre-c0-guided-audit/v1"
RISK_TRIGGER_AUDIT_VERSION = "rase-pre-c0-risk-trigger-audit/v1"
SUITES = ("Spatial", "Object", "Goal", "Long")
CELLS = ("clean:L0", "camera:L1", "robot:L1")
STAGES = ("T0", "T1", "T2", "T3", "T4")
NATURAL_FAMILIES = (
    "current_suffix",
    "strict_resample",
    "fresh_replan",
    "receding_horizon",
)
NATURAL_ALTERNATIVES = NATURAL_FAMILIES[1:]


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _task_number(task_id: str) -> int:
    marker = "task"
    if marker not in task_id:
        raise ValueError(f"invalid logical PRE-A3 task id: {task_id!r}")
    return int(task_id.rsplit(marker, 1)[1])


def build_pre_c0_design(
    pre_a3_design: Mapping[str, Any],
    *,
    seed: int = 2_026_080_405,
) -> dict[str, Any]:
    """Choose one outcome-independent train episode per logical task.

    Each suite has six train tasks. Cycling clean/camera/robot twice within each
    suite yields exactly 8 episodes per perturbation cell over 24 tasks.
    """

    train = [
        dict(row)
        for row in pre_a3_design.get("records") or []
        if str(row.get("split")) == "train"
    ]
    by_suite_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        by_suite_task[(str(row["suite"]), str(row["task_id"]))].append(row)

    records: list[dict[str, Any]] = []
    request_index = 0
    for suite in SUITES:
        tasks = sorted(
            {
                task_id
                for row_suite, task_id in by_suite_task
                if row_suite == suite
            },
            key=_task_number,
        )
        if len(tasks) != 6:
            raise ValueError(f"{suite} must have exactly 6 PRE-A3 train tasks")
        for task_offset, task_id in enumerate(tasks):
            wanted_cell = CELLS[task_offset % len(CELLS)]
            matches = [
                row
                for row in by_suite_task[(suite, task_id)]
                if str(row["cell"]) == wanted_cell
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{suite}/{task_id}/{wanted_cell} must map to one PRE-A3 row"
                )
            source = matches[0]
            records.append(
                {
                    "request_index": request_index,
                    "episode_id": f"ep-pre-c0-{seed:010d}-{request_index:06d}",
                    "source_pre_a3_episode_id": source["episode_id"],
                    "suite": suite,
                    "task_id": task_id,
                    "concrete_task_id": source["concrete_task_id"],
                    "cell": source["cell"],
                    "dimension": source["dimension"],
                    "level": int(source["level"]),
                    "source_split": "train",
                }
            )
            request_index += 1

    cell_counts = Counter(str(row["cell"]) for row in records)
    suite_counts = Counter(str(row["suite"]) for row in records)
    if cell_counts != Counter({cell: 8 for cell in CELLS}):
        raise ValueError(f"PRE-C0 cells are not balanced: {dict(cell_counts)}")
    if suite_counts != Counter({suite: 6 for suite in SUITES}):
        raise ValueError(f"PRE-C0 suites are not balanced: {dict(suite_counts)}")

    payload: dict[str, Any] = {
        "artifact_version": DESIGN_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "source_design": "runs/rase_pre_a3_design120_v1.1.json",
        "source_design_sha256": str(pre_a3_design["design_sha256"]),
        "source_split": "train",
        "selection_uses_outcomes": False,
        "hidden_test_sealed": True,
        "n_episodes": len(records),
        "suite_counts": dict(suite_counts),
        "cell_counts": dict(cell_counts),
        "snapshot_stages": list(STAGES),
        "audit_stages": ["T1", "T3"],
        "records": records,
    }
    payload["design_sha256"] = canonical_sha256(payload)
    return payload


def load_pre_c0_design(
    path: Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_version") != DESIGN_VERSION:
        raise ValueError("unsupported PRE-C0 design version")
    declared = str(payload.get("design_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("design_sha256", None)
    actual = canonical_sha256(unsigned)
    if declared != actual:
        raise ValueError(f"PRE-C0 design checksum mismatch: {declared} != {actual}")
    if expected_sha256 is not None and declared != expected_sha256:
        raise ValueError("PRE-C0 design differs from frozen expected checksum")
    records = list(payload.get("records") or [])
    if len(records) != 24 or len({row["episode_id"] for row in records}) != 24:
        raise ValueError("PRE-C0 design requires 24 unique episodes")
    if any(str(row.get("source_split")) != "train" for row in records):
        raise ValueError("PRE-C0 pilot may only use PRE-A3 train tasks")
    return payload


def requests_from_design(
    design: Mapping[str, Any], *, seed: int
) -> list[PerturbationRequest]:
    result = []
    subdimensions = {"clean": "none", "camera": "viewpoint", "robot": "initial_state"}
    for row in sorted(design["records"], key=lambda item: int(item["request_index"])):
        concrete = str(row["concrete_task_id"])
        try:
            task_id = int(concrete.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid concrete task id {concrete!r}") from exc
        index = int(row["request_index"])
        dimension = str(row["dimension"])
        result.append(
            PerturbationRequest(
                index=index,
                suite=str(row["suite"]),
                dimension=dimension,
                subdimension=subdimensions[dimension],
                level=int(row["level"]),
                seed=int(
                    canonical_sha256(
                        {"protocol": PROTOCOL_VERSION, "seed": seed, "index": index}
                    )[:8],
                    16,
                ),
                global_episode_index=index,
                batch_id=1,
                task_id=task_id,
                init_state_id=None,
                episode_id=str(row["episode_id"]),
            )
        )
    return result


def analyze_natural_headroom(
    per_state: Sequence[Mapping[str, Any]],
    *,
    minimum_headroom_pp: float = 5.0,
    maximum_harm_rate: float = 0.05,
) -> dict[str, Any]:
    """Analyze nested same-policy family oracles on a frozen pilot cohort."""

    rows = [dict(row) for row in per_state]
    if not rows:
        raise ValueError("PRE-C0 audit requires states")
    required = set(NATURAL_FAMILIES)
    for row in rows:
        outcomes = dict(row.get("family_success") or {})
        if not required.issubset(outcomes):
            raise ValueError(f"incomplete PRE-C0 family outcomes for {row.get('state_key')}")

    nested_sets = (
        ("S0_current", ("current_suffix",)),
        ("S1_resample", ("current_suffix", "strict_resample")),
        ("S2_replan", ("current_suffix", "strict_resample", "fresh_replan")),
        (
            "S3_closed_loop",
            (
                "current_suffix",
                "strict_resample",
                "fresh_replan",
                "receding_horizon",
            ),
        ),
    )
    successes: dict[str, int] = {}
    for name, families in nested_sets:
        successes[name] = sum(
            any(bool(row["family_success"][family]) for family in families)
            for row in rows
        )
    n = len(rows)
    total_headroom_pp = 100.0 * (
        successes["S3_closed_loop"] - successes["S0_current"]
    ) / n

    rescued = [
        row
        for row in rows
        if not bool(row["family_success"]["current_suffix"])
        and any(
            bool(row["family_success"][family])
            for family in NATURAL_FAMILIES[1:]
        )
    ]
    rescue_suites = sorted({str(row["suite"]) for row in rescued})
    rescue_tasks = sorted({str(row["task_id"]) for row in rescued})
    controls = [
        row
        for row in rows
        if str(row.get("stage")) == "T0"
        or str(row.get("cell")) == "clean:L0"
    ]
    harmed = sum(
        bool(row["family_success"]["current_suffix"])
        and not any(
            bool(row["family_success"][family])
            for family in NATURAL_FAMILIES[1:]
        )
        for row in controls
    )
    harm_rate = harmed / max(1, len(controls))
    stage_rates = {}
    for stage in ("T1", "T3"):
        stage_rows = [row for row in rows if str(row.get("stage")) == stage]
        if stage_rows:
            base = sum(bool(row["family_success"]["current_suffix"]) for row in stage_rows)
            oracle = sum(
                any(bool(row["family_success"][family]) for family in NATURAL_FAMILIES)
                for row in stage_rows
            )
            stage_rates[stage] = 100.0 * (oracle - base) / len(stage_rows)

    pass_conditions = {
        "natural_headroom_ge_5pp": total_headroom_pp >= minimum_headroom_pp,
        "rescues_cover_ge_2_suites": len(rescue_suites) >= 2,
        "rescues_cover_ge_3_tasks": len(rescue_tasks) >= 3,
        "control_harm_le_5pct": harm_rate <= maximum_harm_rate,
        "early_late_direction_nonnegative": (
            "T1" in stage_rates
            and "T3" in stage_rates
            and stage_rates["T1"] >= 0.0
            and stage_rates["T3"] >= 0.0
        ),
    }
    passed = all(pass_conditions.values())
    return {
        "schema_version": AUDIT_VERSION,
        "status": "natural_same_policy_gate_pass" if passed else "natural_gate_fail",
        "n_states": n,
        "nested_successes": successes,
        "headroom_pp": {
            "sampling": 100.0
            * (successes["S1_resample"] - successes["S0_current"])
            / n,
            "reconditioning": 100.0
            * (successes["S2_replan"] - successes["S1_resample"])
            / n,
            "closed_loop": 100.0
            * (successes["S3_closed_loop"] - successes["S2_replan"])
            / n,
            "natural_total": total_headroom_pp,
        },
        "stage_headroom_pp": stage_rates,
        "rescue_tasks": rescue_tasks,
        "rescue_suites": rescue_suites,
        "control_harm_rate": harm_rate,
        "pass_conditions": pass_conditions,
        "gate_pass": passed,
        "candidate_critic_gate": "eligible" if passed else "closed",
        "world_model_gate": "closed",
    }


def _nested_success(row: Mapping[str, Any], families: Sequence[str]) -> bool:
    outcomes = dict(row.get("family_success") or {})
    return any(bool(outcomes.get(family)) for family in families)


def episode_cluster_bootstrap_natural_headroom(
    per_state: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 2_026_080_405,
) -> dict[str, Any]:
    """Episode-cluster bootstrap 95% CI for nested natural headroom (pp)."""

    rows = [dict(row) for row in per_state]
    if not rows:
        raise ValueError("bootstrap requires states")
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_id") or row.get("state_key"))].append(row)
    episode_ids = sorted(by_episode)
    n_episodes = len(episode_ids)
    if n_episodes < 1:
        raise ValueError("bootstrap requires at least one episode cluster")

    def headroom_pp(selected_ids: Sequence[str]) -> float:
        selected = [row for episode_id in selected_ids for row in by_episode[episode_id]]
        n = len(selected)
        if n == 0:
            return 0.0
        s0 = sum(_nested_success(row, ("current_suffix",)) for row in selected)
        s3 = sum(_nested_success(row, NATURAL_FAMILIES) for row in selected)
        return 100.0 * (s3 - s0) / n

    point = headroom_pp(episode_ids)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sample = rng.choice(episode_ids, size=n_episodes, replace=True)
        draws[index] = headroom_pp(sample)
    lower, upper = (float(value) for value in np.quantile(draws, [0.025, 0.975]))
    return {
        "cluster_unit": "episode_id",
        "n_episodes": n_episodes,
        "n_states": len(rows),
        "replicates": int(replicates),
        "seed": int(seed),
        "point_estimate_pp": point,
        "ci95_pp": [lower, upper],
        "ci95_lower_positive": lower > 0.0,
    }


def horizon_decomposition(per_state: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report fixed H=1/2/4 success and adaptive horizon oracle over receding arms."""

    rows = [dict(row) for row in per_state]
    horizons = (1, 2, 4)
    per_horizon_success = {horizon: 0 for horizon in horizons}
    best_fixed = 0
    adaptive = 0
    n = len(rows)
    for row in rows:
        arms = list(row.get("arms") or [])
        by_h = {
            int(arm["execution_horizon"]): bool(arm.get("success"))
            for arm in arms
            if str(arm.get("family")) == "receding_horizon"
            and arm.get("execution_horizon") is not None
        }
        for horizon in horizons:
            per_horizon_success[horizon] += int(bool(by_h.get(horizon)))
        fixed_flags = [bool(by_h.get(horizon)) for horizon in horizons]
        best_fixed += int(any(fixed_flags))
        adaptive += int(any(fixed_flags))  # same as best-of fixed for oracle over {1,2,4}
    rates = {
        f"H{horizon}": (per_horizon_success[horizon] / n if n else 0.0)
        for horizon in horizons
    }
    best_fixed_rate = best_fixed / n if n else 0.0
    adaptive_rate = adaptive / n if n else 0.0
    # Best single fixed horizon across the cohort (not per-state).
    best_cohort_horizon = max(horizons, key=lambda horizon: per_horizon_success[horizon])
    best_cohort_rate = per_horizon_success[best_cohort_horizon] / n if n else 0.0
    return {
        "n_states": n,
        "per_horizon_successes": {f"H{h}": per_horizon_success[h] for h in horizons},
        "per_horizon_rates": rates,
        "best_fixed_horizon_oracle_successes": best_fixed,
        "best_fixed_horizon_oracle_rate": best_fixed_rate,
        "per_state_horizon_oracle_successes": adaptive,
        "per_state_horizon_oracle_rate": adaptive_rate,
        "best_cohort_fixed_horizon": best_cohort_horizon,
        "best_cohort_fixed_rate": best_cohort_rate,
        "H_adaptive_horizon_pp": 100.0 * (adaptive_rate - best_cohort_rate),
        "note": (
            "H_adaptive_horizon compares per-state best-of-{1,2,4} against the "
            "single best fixed horizon chosen on the full cohort."
        ),
    }


def leave_one_task_out_natural_direction(
    per_state: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Leave-one-task-out sign stability for nested natural headroom."""

    rows = [dict(row) for row in per_state]
    tasks = sorted({str(row["task_id"]) for row in rows})
    folds = []
    nonnegative = 0
    for held_out in tasks:
        kept = [row for row in rows if str(row["task_id"]) != held_out]
        if not kept:
            continue
        s0 = sum(_nested_success(row, ("current_suffix",)) for row in kept)
        s3 = sum(_nested_success(row, NATURAL_FAMILIES) for row in kept)
        headroom_pp = 100.0 * (s3 - s0) / len(kept)
        folds.append(
            {
                "held_out_task": held_out,
                "n_states": len(kept),
                "natural_headroom_pp": headroom_pp,
            }
        )
        nonnegative += int(headroom_pp >= 0.0)
    return {
        "n_tasks": len(tasks),
        "n_folds": len(folds),
        "nonnegative_folds": nonnegative,
        "all_folds_nonnegative": nonnegative == len(folds) and bool(folds),
        "folds": folds,
    }


def analyze_guided_headroom(
    per_state: Sequence[Mapping[str, Any]],
    *,
    minimum_guided_gain_pp: float = 8.0,
    frozen_nogo_gain_pp: float = 5.0,
    maximum_harm_rate: float = 0.05,
) -> dict[str, Any]:
    """Analyze privileged-guided gain over the nested natural oracle.

    Each row must include ``family_success`` for the natural families and a
    boolean ``privileged_guidance`` outcome for the matched-compute guided arm.
    """

    rows = [dict(row) for row in per_state]
    if not rows:
        raise ValueError("PRE-C0 guided audit requires states")
    required = set(NATURAL_FAMILIES)
    for row in rows:
        outcomes = dict(row.get("family_success") or {})
        if not required.issubset(outcomes):
            raise ValueError(f"incomplete PRE-C0 family outcomes for {row.get('state_key')}")
        if "privileged_guidance" not in row:
            raise ValueError(f"missing privileged_guidance for {row.get('state_key')}")

    n = len(rows)
    natural = sum(
        any(bool(row["family_success"][family]) for family in NATURAL_FAMILIES)
        for row in rows
    )
    guided = sum(
        any(bool(row["family_success"][family]) for family in NATURAL_FAMILIES)
        or bool(row["privileged_guidance"])
        for row in rows
    )
    guided_gain_pp = 100.0 * (guided - natural) / n

    guided_rescues = [
        row
        for row in rows
        if bool(row["privileged_guidance"])
        and not any(bool(row["family_success"][family]) for family in NATURAL_FAMILIES)
    ]
    rescue_suites = sorted({str(row["suite"]) for row in guided_rescues})
    t1_rows = [row for row in rows if str(row.get("stage")) == "T1"]
    t1_gain_pp = 0.0
    if t1_rows:
        t1_natural = sum(
            any(bool(row["family_success"][family]) for family in NATURAL_FAMILIES)
            for row in t1_rows
        )
        t1_guided = sum(
            any(bool(row["family_success"][family]) for family in NATURAL_FAMILIES)
            or bool(row["privileged_guidance"])
            for row in t1_rows
        )
        t1_gain_pp = 100.0 * (t1_guided - t1_natural) / len(t1_rows)

    controls = [
        row
        for row in rows
        if str(row.get("stage")) == "T0" or str(row.get("cell")) == "clean:L0"
    ]
    # Control harm: clean/T0 current successes where privileged selection finds
    # no recovering candidate among the matched-K set.
    guided_harmed = sum(
        bool(row["family_success"]["current_suffix"])
        and not bool(row["privileged_guidance"])
        for row in controls
    )
    harm_rate = guided_harmed / max(1, len(controls))

    pass_conditions = {
        "guided_gain_ge_8pp": guided_gain_pp >= minimum_guided_gain_pp,
        "guided_rescues_cover_ge_2_suites": len(rescue_suites) >= 2,
        "t1_guided_gain_nonnegative": t1_gain_pp >= 0.0,
        "control_harm_le_5pct": harm_rate <= maximum_harm_rate,
    }
    passed = all(pass_conditions.values())
    frozen_nogo = guided_gain_pp < frozen_nogo_gain_pp
    if passed:
        decision = "learned_recovery_critic_eligible"
        guided_gate = "open"
    elif frozen_nogo:
        decision = "frozen_same_policy_recovery_nogo"
        guided_gate = "closed"
    else:
        decision = "guided_gain_inconclusive_review"
        guided_gate = "closed"

    return {
        "schema_version": GUIDED_AUDIT_VERSION,
        "status": "guided_generation_gate_pass" if passed else "guided_gate_fail",
        "n_states": n,
        "nested_successes": {
            "S3_natural": natural,
            "S4_privileged_guided": guided,
        },
        "guided_gain_pp": guided_gain_pp,
        "t1_guided_gain_pp": t1_gain_pp,
        "rescue_suites": rescue_suites,
        "control_harm_rate": harm_rate,
        "pass_conditions": pass_conditions,
        "gate_pass": passed,
        "guided_generation_gate": guided_gate,
        "learned_recovery_critic_gate": "eligible" if passed else "closed",
        "frozen_same_policy_recovery": "NOGO" if frozen_nogo else "open_for_review",
        "decision": decision,
        "world_model_gate": "closed",
    }


def analyze_risk_trigger_oracle(
    per_state: Sequence[Mapping[str, Any]],
    *,
    maximum_harm_rate: float = 0.05,
) -> dict[str, Any]:
    """Compare risk-triggered intervention vs always-intervene on natural oracles.

    Risk-trigger oracle: intervene with the nested natural alternative oracle only
    when ``current_suffix`` fails. Always-intervene: replace current with the
    nested alternative oracle on every state (even when current already succeeds).
    """

    rows = [dict(row) for row in per_state]
    if not rows:
        raise ValueError("risk-trigger audit requires states")
    required = set(NATURAL_FAMILIES)
    for row in rows:
        outcomes = dict(row.get("family_success") or {})
        if not required.issubset(outcomes):
            raise ValueError(f"incomplete PRE-C0 family outcomes for {row.get('state_key')}")

    n = len(rows)
    current_success = 0
    risk_trigger_success = 0
    always_intervene_success = 0
    interventions = 0
    clean_rows = 0
    risk_trigger_harm = 0
    always_intervene_harm = 0
    preserved_natural_headroom = 0
    natural_rescues = 0

    for row in rows:
        outcomes = dict(row["family_success"])
        current = bool(outcomes["current_suffix"])
        alt = any(bool(outcomes[family]) for family in NATURAL_ALTERNATIVES)
        natural = current or alt
        current_success += int(current)
        natural_rescues += int((not current) and alt)
        preserved_natural_headroom += int(natural)

        # Risk-trigger: keep current when it succeeds; else take alternative oracle.
        if current:
            risk_trigger_success += 1
            risk_outcome = True
            intervened = False
        else:
            interventions += 1
            intervened = True
            risk_outcome = alt
            risk_trigger_success += int(alt)

        # Always intervene: replace current with the alternative oracle on every state.
        always_outcome = alt
        always_intervene_success += int(always_outcome)

        is_control = str(row.get("stage")) == "T0" or str(row.get("cell")) == "clean:L0"
        if is_control:
            clean_rows += 1
            # Harm: current would succeed, selected intervention fails.
            if current and intervened and not risk_outcome:
                risk_trigger_harm += 1
            if current and not always_outcome:
                always_intervene_harm += 1

    risk_rate = risk_trigger_success / n
    always_rate = always_intervene_success / n
    current_rate = current_success / n
    natural_rate = preserved_natural_headroom / n
    risk_harm_rate = risk_trigger_harm / max(1, clean_rows)
    always_harm_rate = always_intervene_harm / max(1, clean_rows)
    retained_headroom_pp = 100.0 * (risk_trigger_success - current_success) / n
    always_headroom_pp = 100.0 * (always_intervene_success - current_success) / n
    harm_reduction = always_harm_rate - risk_harm_rate
    meaningful = (
        retained_headroom_pp > 0.0
        and risk_harm_rate <= maximum_harm_rate
        and harm_reduction >= 0.0
        and abs(risk_rate - natural_rate) < 1e-9
    )
    return {
        "schema_version": RISK_TRIGGER_AUDIT_VERSION,
        "n_states": n,
        "n_control_states": clean_rows,
        "success_rates": {
            "current": current_rate,
            "risk_trigger_oracle": risk_rate,
            "always_intervene_oracle": always_rate,
            "natural_nested_oracle": natural_rate,
        },
        "success_counts": {
            "current": current_success,
            "risk_trigger_oracle": risk_trigger_success,
            "always_intervene_oracle": always_intervene_success,
            "natural_nested_oracle": preserved_natural_headroom,
        },
        "intervention_count": interventions,
        "intervention_rate": interventions / n,
        "natural_rescue_count": natural_rescues,
        "headroom_pp": {
            "risk_trigger_vs_current": retained_headroom_pp,
            "always_intervene_vs_current": always_headroom_pp,
            "natural_vs_current": 100.0 * (preserved_natural_headroom - current_success) / n,
        },
        "control_harm_rate": {
            "risk_trigger": risk_harm_rate,
            "always_intervene": always_harm_rate,
        },
        "harm_reduction_vs_always": harm_reduction,
        "retains_full_natural_oracle": abs(risk_rate - natural_rate) < 1e-9,
        "meaningful_for_critic": meaningful,
        "decision": (
            "candidate_critic_training_eligible"
            if meaningful
            else "risk_trigger_insufficient_for_critic"
        ),
        "world_model_gate": "closed",
        "note": (
            "Risk-trigger assumes perfect knowledge of current failure and uses "
            "the nested natural alternative oracle only on those states."
        ),
    }
